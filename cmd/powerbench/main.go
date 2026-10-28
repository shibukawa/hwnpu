// Command powerbench measures how much power Core ML models draw while they run, using macOS
// powermetrics (CPU, GPU and ANE rails). It answers "is the slower Neural Engine run still
// cheaper in energy?", which latency alone cannot.
//
//	go run ./cmd/powerbench                 # asks for the sudo password once
//	go run ./cmd/powerbench -rate 200       # fixed request rate instead of back-to-back runs
//	go run ./cmd/powerbench -model my.mlpackage -model my_ane.mlpackage
//	go run ./cmd/powerbench -model 'build/sweep/*.mlpackage'   # tools/coreml/gen_sweep.py
//
// A model that reaches the Neural Engine is also measured with Core ML limited to the CPU
// (-cpu=false skips that), so the same computation is compared on both.
// Each round measures idle power, then every workload for -d while it runs continuously (or at
// -rate calls per second), and the result subtracts idle power from the workload's power.
// Energy per call = net power / calls per second.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/fixtures"
)

type Workload struct {
	Name      string   `json:"name"`
	Path      string   `json:"path"`
	Units     string   `json:"units"`    // "npu" (CPU and Neural Engine), "cpu" or "gpu"
	CPUOnly   bool     `json:"cpu_only"` // Units == "cpu"
	Placement string   `json:"placement"`
	Rows      int      `json:"rows"`     // logical batch the caller gets back per call
	Physical  int      `json:"physical"` // batch the model computes (padded batch)
	Rounds    []Result `json:"rounds"`
	Mean      Result   `json:"mean"`
	Err       string   `json:"error,omitempty"`

	req   *hwnpu.Request
	close func()
	onNPU bool // at least one operation runs on the NPU
}

type Result struct {
	CallsPerS     float64 `json:"calls_per_s"`
	MeanLatency   float64 `json:"mean_latency_us"`
	Gross         Rails   `json:"gross"`
	Net           Rails   `json:"net"`                        // gross - idle
	EnergyPerCall float64 `json:"energy_per_call_mj"`         // net combined / calls per second
	EnergyPerRow  float64 `json:"energy_per_row_mj"`          // per row the caller gets back
	EnergyPerComp float64 `json:"energy_per_computed_row_mj"` // per row computed, padding included
}

type Report struct {
	Time     time.Time   `json:"time"`
	Host     string      `json:"host"`
	Duration string      `json:"duration_per_phase"`
	Rate     float64     `json:"rate,omitempty"`
	Idle     []Rails     `json:"idle"`
	IdleMean Rails       `json:"idle_mean"`
	Work     []*Workload `json:"workloads"`
}

type modelList []string

func (m *modelList) String() string { return strings.Join(*m, ",") }

// Set accepts a path or a glob pattern ('dir/*.mlpackage'), quoted so the shell leaves it alone.
func (m *modelList) Set(v string) error {
	if !strings.ContainsAny(v, "*?[") {
		*m = append(*m, v)
		return nil
	}
	paths, err := filepath.Glob(v)
	if err != nil {
		return err
	}
	if len(paths) == 0 {
		return fmt.Errorf("%s matches no file", v)
	}
	*m = append(*m, paths...)
	return nil
}

func main() {
	var models modelList
	dur := flag.Duration("d", 10*time.Second, "measured time per phase")
	interval := flag.Duration("i", 500*time.Millisecond, "powermetrics sample interval")
	warm := flag.Duration("warm", time.Second, "time a workload runs before sampling starts")
	cool := flag.Duration("cool", 3*time.Second, "pause between phases")
	rounds := flag.Int("rounds", 2, "repeat idle + all workloads this many times")
	rate := flag.Float64("rate", 0, "calls per second (0: back-to-back)")
	samplers := flag.String("samplers", "cpu_power,gpu_power,ane_power", "powermetrics samplers")
	rawDir := flag.String("raw", "", "keep the raw powermetrics output in this directory")
	jsonOut := flag.String("json", "", "also write the result as JSON to this file")
	out := flag.String("o", "", "also write the Markdown report to this file")
	cpu := flag.Bool("cpu", true, "also measure each model that reaches the NPU on the CPU alone")
	one := flag.Bool("one", false, "load one model at a time and run each variant's rounds back to back (for models of hundreds of MB and more)")
	gpu := flag.Bool("gpu", false, "also measure each model on the GPU (Core ML CPU and GPU)")
	flag.Var(&models, "model", "Core ML model to measure instead of the built-in mlp_small pair (repeatable)")
	flag.Parse()

	if runtime.GOOS != "darwin" {
		fail("powerbench uses macOS powermetrics; on Linux and Windows use cmd/report -n, which reads RAPL or the NPU's own sensor")
	}
	if *rawDir != "" {
		if err := os.MkdirAll(*rawDir, 0o755); err != nil {
			fail(err.Error())
		}
	}
	ctx := context.Background()
	s := &sampler{interval: *interval, samplers: *samplers, rawDir: *rawDir}
	if err := s.authorize(); err != nil {
		fail("sudo: " + err.Error())
	}

	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer))
	if err != nil {
		fail(err.Error())
	}
	defer dev.Close()

	if len(models) == 0 {
		dir, err := os.MkdirTemp("", "hwnpu-powerbench")
		if err != nil {
			fail(err.Error())
		}
		defer os.RemoveAll(dir)
		if err := fixtures.Extract(dir); err != nil {
			fail(err.Error())
		}
		models = modelList{filepath.Join(dir, fixtures.CoreMLSmall), filepath.Join(dir, fixtures.CoreMLSmallANE)}
	}
	var baselines []string
	if *cpu {
		baselines = append(baselines, "cpu")
	}
	if *gpu {
		baselines = append(baselines, "gpu")
	}
	rep := &Report{Time: time.Now().UTC().Truncate(time.Second), Host: host(), Duration: dur.String(), Rate: *rate}
	n := int(*dur / *interval) + 1 // the first sample is dropped
	measureIdle := func(label string) {
		fmt.Fprintf(os.Stderr, "%s: idle\n", label)
		samples, err := s.sample(ctx, n, label+"-idle")
		if err == nil && len(samples) > 1 {
			var r Rails
			if r, err = average(samples[1:]); err == nil {
				rep.Idle = append(rep.Idle, r)
				return
			}
		}
		fail(fmt.Sprint("idle: ", err))
	}
	run := func(label string, w *Workload) {
		if w.req == nil {
			return
		}
		time.Sleep(*cool)
		fmt.Fprintf(os.Stderr, "%s: %s\n", label, w.Name)
		res, err := measure(ctx, s, w, n, *warm, *rate, fmt.Sprintf("%s-%s", strings.ReplaceAll(label, " ", ""), strings.NewReplacer(" ", "", "[", "", "]", "").Replace(w.Name)))
		if err != nil {
			w.Err = err.Error()
			w.req = nil
			return
		}
		w.Rounds = append(w.Rounds, res)
	}

	if *one {
		// Large models: only one model is loaded at a time, and each variant runs all its rounds
		// before the next one is loaded. Idle is measured after every model to follow drift.
		measureIdle("start")
		for i, m := range models {
			w := prepare(ctx, dev, m, "npu")
			var ws []*Workload
			for r := 1; r <= *rounds; r++ {
				run(fmt.Sprintf("model %d round %d", i+1, r), w)
			}
			if w.close != nil {
				w.close()
			}
			for _, u := range baselines {
				if !w.onNPU && u == "cpu" { // the normal run is already the CPU baseline
					continue
				}
				c := prepare(ctx, dev, m, u)
				for r := 1; r <= *rounds; r++ {
					run(fmt.Sprintf("model %d round %d", i+1, r), c)
				}
				if c.close != nil {
					c.close()
				}
				ws = append(ws, c)
			}
			rep.Work = append(rep.Work, append(ws, w)...)
			time.Sleep(*cool)
			measureIdle(fmt.Sprintf("model %d", i+1))
		}
	} else {
		for _, m := range models {
			w := prepare(ctx, dev, m, "npu")
			if w.close != nil {
				defer w.close()
			}
			// Baselines: the CPU alone (unless the normal run already stays on the CPU) and the GPU.
			for _, u := range baselines {
				if w.req == nil || (!w.onNPU && u == "cpu") {
					continue
				}
				c := prepare(ctx, dev, m, u)
				if c.close != nil {
					defer c.close()
				}
				rep.Work = append(rep.Work, c)
			}
			rep.Work = append(rep.Work, w)
		}
		for round := 1; round <= *rounds; round++ {
			label := fmt.Sprintf("round %d", round)
			measureIdle(label)
			for _, w := range rep.Work {
				run(label, w)
			}
			time.Sleep(*cool)
		}
		measureIdle("final")
	}

	for _, r := range rep.Idle {
		rep.IdleMean.CPU += r.CPU / float64(len(rep.Idle))
		rep.IdleMean.GPU += r.GPU / float64(len(rep.Idle))
		rep.IdleMean.ANE += r.ANE / float64(len(rep.Idle))
		rep.IdleMean.Combined += r.Combined / float64(len(rep.Idle))
	}
	for _, w := range rep.Work {
		finish(w, rep.IdleMean)
	}

	md := markdown(rep)
	fmt.Print(md)
	if *out != "" {
		if err := os.WriteFile(*out, []byte(md), 0o644); err != nil {
			fail(err.Error())
		}
	}
	if *jsonOut != "" {
		b, _ := json.MarshalIndent(rep, "", "  ")
		if err := os.WriteFile(*jsonOut, append(b, '\n'), 0o644); err != nil {
			fail(err.Error())
		}
	}
}

// prepare compiles the model and binds inputs filled with x[i] = sin(i * 0.01).
// units "cpu" and "gpu" load the model with MLComputeUnitsCPUOnly and MLComputeUnitsCPUAndGPU
// (through HWNPU_COREML_UNITS), as baselines; "npu" is hwnpu's normal CPU and Neural Engine.
func prepare(ctx context.Context, dev *hwnpu.Device, path, units string) *Workload {
	w := &Workload{Name: strings.TrimSuffix(filepath.Base(path), ".mlpackage"), Path: path, Units: units, CPUOnly: units == "cpu"}
	switch units {
	case "cpu":
		w.Name += " [CPU only]"
	case "gpu":
		w.Name += " [GPU]"
	}
	gr, err := dev.Graph()
	if err != nil {
		w.Err = err.Error()
		return w
	}
	if units != "npu" {
		os.Setenv("HWNPU_COREML_UNITS", units)
	}
	g, err := gr.Compile(ctx, hwnpu.GraphSource{Path: path})
	os.Unsetenv("HWNPU_COREML_UNITS")
	if err != nil {
		w.Err = err.Error()
		return w
	}
	rep := g.Placement()
	w.onNPU = rep.NPUOps > 0
	devs := map[string]int{}
	for _, op := range rep.Ops {
		devs[strings.ToUpper(op.Device)]++
	}
	w.Placement = fmt.Sprintf("NPU %d", rep.NPUOps)
	for _, d := range []string{"CPU", "GPU", "UNKNOWN"} {
		if devs[d] > 0 {
			w.Placement += fmt.Sprintf(", %s %d", d, devs[d])
		}
	}
	w.Placement += " ops"
	req, err := g.NewRequest()
	if err != nil {
		g.Close()
		w.Err = err.Error()
		return w
	}
	var bufs []*hwnpu.Buffer
	w.close = func() {
		req.Close()
		for _, b := range bufs {
			b.Close()
		}
		g.Close()
	}
	for _, a := range g.Args() {
		b, err := dev.AllocBuffer(a.Bytes)
		if err != nil {
			w.Err = err.Error()
			return w
		}
		bufs = append(bufs, b)
		if a.Kind == hwnpu.ArgInput {
			if err := fill(b.Bytes(), a.DType); err != nil {
				w.Err = err.Error()
				return w
			}
			b.SyncToDevice(0, a.Bytes)
		}
		if err := req.Bind(a.Index, b); err != nil {
			w.Err = err.Error()
			return w
		}
		if w.Rows == 0 && len(a.Dims) > 0 {
			w.Rows, w.Physical = a.Dims[0], a.Dims[0]
			if len(a.PaddedDims) > 0 {
				w.Physical = a.PaddedDims[0]
			}
		}
	}
	if err := req.Run(ctx); err != nil {
		w.Err = "run: " + err.Error()
		return w
	}
	w.req = req
	return w
}

func fill(dst []byte, t hwnpu.DType) error {
	switch t {
	case hwnpu.F32:
		for i := 0; i+4 <= len(dst); i += 4 {
			v := math.Float32bits(float32(math.Sin(float64(i/4) * 0.01)))
			dst[i], dst[i+1], dst[i+2], dst[i+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
		}
	case hwnpu.F16:
		v := make([]float32, len(dst)/2)
		for i := range v {
			v[i] = float32(math.Sin(float64(i) * 0.01))
		}
		hwnpu.EncodeF16(dst, v)
	case hwnpu.I32:
		// 1 is valid as a token id and as an attention mask value; the cost does not depend on it.
		for i := 0; i+4 <= len(dst); i += 4 {
			dst[i], dst[i+1], dst[i+2], dst[i+3] = 1, 0, 0, 0
		}
	default:
		return fmt.Errorf("input type %s is not supported", t)
	}
	return nil
}

// measure keeps the workload running on its own thread while powermetrics samples, and counts
// the calls completed while powermetrics ran.
func measure(ctx context.Context, s *sampler, w *Workload, n int, warm time.Duration, rate float64, raw string) (Result, error) {
	var calls, busy atomic.Int64
	var stop atomic.Bool
	var runErr error
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		var period time.Duration
		if rate > 0 {
			period = time.Duration(float64(time.Second) / rate)
		}
		next := time.Now()
		for !stop.Load() {
			if period > 0 {
				if d := time.Until(next); d > 0 {
					time.Sleep(d)
				}
				next = next.Add(period)
				if time.Until(next) < -period { // fell behind: do not burst to catch up
					next = time.Now()
				}
			}
			t := time.Now()
			if err := w.req.Run(ctx); err != nil {
				runErr = err
				return
			}
			busy.Add(int64(time.Since(t)))
			calls.Add(1)
		}
	}()
	time.Sleep(warm)
	c0, b0, t0 := calls.Load(), busy.Load(), time.Now()
	samples, err := s.sample(ctx, n, raw)
	c1, b1, t1 := calls.Load(), busy.Load(), time.Now()
	stop.Store(true)
	wg.Wait()
	if runErr != nil {
		return Result{}, fmt.Errorf("run: %w", runErr)
	}
	if err != nil {
		return Result{}, err
	}
	if len(samples) < 2 {
		return Result{}, errors.New("powermetrics returned fewer than 2 samples")
	}
	gross, err := average(samples[1:])
	if err != nil {
		return Result{}, err
	}
	r := Result{Gross: gross, CallsPerS: float64(c1-c0) / t1.Sub(t0).Seconds()}
	if c1 > c0 {
		r.MeanLatency = float64(b1-b0) / float64(c1-c0) / 1e3
	}
	return r, nil
}

func finish(w *Workload, idle Rails) {
	if len(w.Rounds) == 0 {
		return
	}
	k := float64(len(w.Rounds))
	for i := range w.Rounds {
		r := &w.Rounds[i]
		r.Net = r.Gross.sub(idle)
		if r.CallsPerS > 0 {
			r.EnergyPerCall = r.Net.Combined / r.CallsPerS // mW / (1/s) = mJ
			r.EnergyPerRow = r.EnergyPerCall / float64(max(w.Rows, 1))
			r.EnergyPerComp = r.EnergyPerCall / float64(max(w.Physical, 1))
		}
		m := &w.Mean
		m.CallsPerS += r.CallsPerS / k
		m.MeanLatency += r.MeanLatency / k
		m.Gross = Rails{m.Gross.CPU + r.Gross.CPU/k, m.Gross.GPU + r.Gross.GPU/k, m.Gross.ANE + r.Gross.ANE/k, m.Gross.Combined + r.Gross.Combined/k}
		m.EnergyPerCall += r.EnergyPerCall / k
		m.EnergyPerRow += r.EnergyPerRow / k
		m.EnergyPerComp += r.EnergyPerComp / k
	}
	w.Mean.Net = w.Mean.Gross.sub(idle)
}

func markdown(r *Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# hwnpu power benchmark\n\n%s, %s, %s per phase", r.Time.Format(time.RFC3339), r.Host, r.Duration)
	if r.Rate > 0 {
		fmt.Fprintf(&b, ", %.0f calls/s", r.Rate)
	} else {
		b.WriteString(", back-to-back calls")
	}
	fmt.Fprintf(&b, "\n\nIdle (mean of %d): CPU %.0f mW, GPU %.0f mW, ANE %.0f mW, combined %.0f mW\n\n",
		len(r.Idle), r.IdleMean.CPU, r.IdleMean.GPU, r.IdleMean.ANE, r.IdleMean.Combined)
	b.WriteString("| model | placement | rows (computed) | calls/s | latency µs | net CPU mW | net GPU mW | net ANE mW | net total mW | mJ/call | µJ/computed row |\n")
	b.WriteString("|---|---|---|---|---|---|---|---|---|---|---|\n")
	for _, w := range r.Work {
		if len(w.Rounds) == 0 {
			fmt.Fprintf(&b, "| %s | %s | | | | | | | | | error: %s |\n", w.Name, w.Placement, w.Err)
			continue
		}
		m := w.Mean
		fmt.Fprintf(&b, "| %s | %s | %d (%d) | %.0f | %.1f | %.0f | %.0f | %.0f | %.0f | %.4f | %.3f |\n",
			w.Name, w.Placement, w.Rows, w.Physical, m.CallsPerS, m.MeanLatency, m.Net.CPU, m.Net.GPU, m.Net.ANE, m.Net.Combined, m.EnergyPerCall, m.EnergyPerComp*1000)
	}
	b.WriteString("\nnet = power while running - idle power. mJ/call = net total / calls per second.\n" +
		"µJ/computed row counts padded rows too: it is what each row would cost if the padding carried real requests.\n")
	for _, w := range r.Work {
		if w.Mean.Net.Combined <= 0 && len(w.Rounds) > 0 {
			fmt.Fprintf(&b, "\n%s: net power is not above idle; the difference is within noise (use a longer -d or more -rounds).\n", w.Name)
		}
		if w.Err != "" && len(w.Rounds) > 0 {
			fmt.Fprintf(&b, "\n%s: stopped early: %s\n", w.Name, w.Err)
		}
	}
	return b.String()
}

// host names the machine by model and macOS version, not by host name.
func host() string {
	model, _ := exec.Command("sysctl", "-n", "hw.model").Output()
	ver, _ := exec.Command("sw_vers", "-productVersion").Output()
	return strings.TrimSpace(string(model)) + ", macOS " + strings.TrimSpace(string(ver))
}

func fail(msg string) {
	fmt.Fprintln(os.Stderr, "powerbench:", msg)
	os.Exit(1)
}
