package main

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/probe"
)

// result is one model loaded and run.
type result struct {
	Format      string  `json:"format"`
	Layers      int     `json:"layers"`
	Tokens      int     `json:"tokens"`         // tokens per call: 1, or K drafted tokens checked at once
	Units       string  `json:"units"`          // npu, or cpu and gpu for the Core ML baselines
	Copy        int     `json:"copy,omitempty"` // several models at once: which one
	Params      int64   `json:"params"`
	WeightBytes int64   `json:"weight_bytes"`
	FileBytes   int64   `json:"file_bytes"`
	BuildS      float64 `json:"build_s"`
	CompileS    float64 `json:"compile_s"`
	Verified    bool    `json:"verified"`
	NPUOps      int     `json:"npu_ops"`
	OtherOps    int     `json:"other_ops"`
	OtherWhere  string  `json:"other_where,omitempty"`
	PlanError   string  `json:"plan_error,omitempty"` // why the placement could not be verified
	Error       string  `json:"error,omitempty"`
	Calls       int     `json:"calls,omitempty"`
	MsPerCall   float64 `json:"ms_per_call,omitempty"`
	CPUUse      float64 `json:"cpu_use,omitempty"`     // process CPU time / wall time while running
	NetW        float64 `json:"net_w,omitempty"`       // meter power while running minus idle
	MJPerCall   float64 `json:"mj_per_call,omitempty"` // NetW / calls per second
	// With every model of the several-models test loaded.
	MsAllLoaded  float64 `json:"ms_all_loaded,omitempty"`
	CPUAllLoaded float64 `json:"cpu_all_loaded,omitempty"`
	OnNPU        bool    `json:"on_npu"`
}

func (r result) placement() string {
	switch {
	case r.Error != "" && r.NPUOps == 0 && r.OtherOps == 0:
		return "error"
	case r.NPUOps == 0 && r.OtherOps == 0:
		return "unverified"
	case r.OtherOps == 0 && r.NPUOps > 0:
		return "NPU"
	case r.NPUOps == 0:
		return strings.ToUpper(r.OtherWhere)
	}
	return fmt.Sprintf("partial (NPU %d, %s)", r.NPUOps, r.OtherWhere)
}

func (r result) gbps() float64 {
	if r.MsPerCall == 0 {
		return 0
	}
	return float64(r.WeightBytes) / (r.MsPerCall / 1000) / 1e9
}

type prober struct {
	ctx    context.Context
	dev    *hwnpu.Device
	gr     hwnpu.GraphRunner
	coreml bool
	dir    string
	minRun time.Duration
	idleW  float64 // meter power at rest, when the device has a meter
	meter  bool
}

type loaded struct {
	g       *hwnpu.Graph
	req     *hwnpu.Request
	bufs    []*hwnpu.Buffer // owned, closed with the model
	in, out *hwnpu.Buffer   // bound to x and y
	path    string
}

func (l *loaded) close() {
	if l.req != nil {
		l.req.Close()
	}
	for _, b := range l.bufs {
		b.Close()
	}
	if l.g != nil {
		l.g.Close()
	}
	if l.path != "" {
		os.RemoveAll(l.path)
	}
}

// load builds a model that takes the given tokens per call, compiles it for the given units and
// binds its buffers. It fills r with what it learned; the returned model is nil when it could not
// be compiled.
func (p *prober) load(format string, layers, tokens int, units string, seed uint64, r *result) *loaded {
	part := probe.Whole(layers)
	part.Tokens = tokens
	return p.loadPart(format, part, units, seed, nil, r)
}

// loadPart loads one part of a split model as load does a whole one. When in is not nil it is
// bound to the input instead of a new buffer, so that the part reads what the one before wrote.
func (p *prober) loadPart(format string, part probe.Part, units string, seed uint64, in *hwnpu.Buffer, r *result) *loaded {
	layers := part.Layers
	r.Format, r.Layers, r.Tokens, r.Units = format, layers, max(part.Tokens, 1), units
	r.Params, r.WeightBytes = paramsAndBytes(format, layers)
	l := &loaded{}
	var src hwnpu.GraphSource
	t := time.Now()
	if p.coreml {
		l.path = filepath.Join(p.dir, fmt.Sprintf("probe_l%d_%d_%s_t%d_%d.mlpackage", part.First, layers, format, r.Tokens, seed))
		if err := probe.WriteMLPackagePart(l.path, format, part, seed); err != nil {
			r.Error = "build: " + err.Error()
			l.close()
			return nil
		}
		r.FileBytes = probe.DirSize(l.path)
		src = hwnpu.GraphSource{Path: l.path}
	} else {
		xmlData, bin, err := probe.BuildIRPart(format, part, seed)
		if err != nil {
			r.Error = "build: " + err.Error()
			return nil
		}
		r.FileBytes = int64(len(xmlData) + len(bin))
		src = hwnpu.GraphSource{Format: hwnpu.GraphFormatOpenVINOIR, XML: xmlData, Bin: bin}
	}
	r.BuildS = time.Since(t).Seconds()

	if p.coreml && units != "npu" {
		// The measurement switch of the Core ML backend, read when the model is loaded.
		os.Setenv("HWNPU_COREML_UNITS", units)
		defer os.Unsetenv("HWNPU_COREML_UNITS")
	}
	t = time.Now()
	g, err := p.gr.Compile(p.ctx, src)
	r.CompileS = time.Since(t).Seconds()
	src = hwnpu.GraphSource{}
	debug.FreeOSMemory()
	var pe *hwnpu.PlacementError
	if errors.As(err, &pe) {
		fillPlacement(r, pe.Report)
	}
	if err != nil {
		r.Error = err.Error()
		l.close()
		return nil
	}
	l.g = g
	fillPlacement(r, g.Placement())
	if l.req, err = g.NewRequest(); err != nil {
		r.Error = "request: " + err.Error()
		l.close()
		return nil
	}
	for _, a := range g.Args() {
		b := in
		if a.Kind != hwnpu.ArgInput || in == nil {
			if b, err = p.dev.AllocBuffer(a.Bytes); err != nil {
				r.Error = "buffer: " + err.Error()
				l.close()
				return nil
			}
			l.bufs = append(l.bufs, b)
			if a.Kind == hwnpu.ArgInput {
				fillInput(b.Bytes(), a.DType)
				b.SyncToDevice(0, a.Bytes)
			}
		}
		if a.Kind == hwnpu.ArgInput {
			l.in = b
		} else {
			l.out = b
		}
		if err := l.req.Bind(a.Index, b); err != nil {
			r.Error = "bind: " + err.Error()
			l.close()
			return nil
		}
	}
	return l
}

func fillPlacement(r *result, rep hwnpu.PlacementReport) {
	r.Verified, r.NPUOps, r.OtherOps = rep.Verified, rep.NPUOps, len(rep.Ops)
	count := map[string]int{}
	for _, o := range rep.Ops {
		count[o.Device]++
	}
	var parts []string
	for _, d := range slices.Sorted(maps.Keys(count)) {
		parts = append(parts, fmt.Sprintf("%s %d", d, count[d]))
	}
	r.OtherWhere = strings.Join(parts, ", ")
	if !rep.Verified {
		r.PlanError = rep.Reason
	}
}

// fillInput writes x[i] = sin(i/100) in the buffer's element type.
func fillInput(p []byte, dt hwnpu.DType) {
	switch dt {
	case hwnpu.F16:
		x := make([]float32, len(p)/2)
		for i := range x {
			x[i] = float32(math.Sin(float64(i) / 100))
		}
		hwnpu.EncodeF16(p, x)
	case hwnpu.F32:
		for i := 0; i+4 <= len(p); i += 4 {
			v := math.Float32bits(float32(math.Sin(float64(i/4) / 100)))
			p[i], p[i+1], p[i+2], p[i+3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
		}
	}
}

// measureIdle reads the device's power meter at rest, when it has one that can be read.
func (p *prober) measureIdle() error {
	a, err := p.dev.PowerSample()
	if err != nil {
		return err
	}
	time.Sleep(3 * time.Second)
	b, err := p.dev.PowerSample()
	if err != nil {
		return err
	}
	p.idleW, p.meter = hwnpu.AveragePower(a, b), true
	return nil
}

// time runs the model for at least p.minRun and 3 calls after 2 warm-up calls, and fills the
// speed, CPU use and, with a power meter, energy of r. quick runs it once, for models that left
// the NPU and may be very slow on the CPU.
func (p *prober) time(l *loaded, r *result, quick bool) error {
	return p.measure(func() error { return l.req.Run(p.ctx) }, r, quick)
}

// measure times run as time does a model.
func (p *prober) measure(run func() error, r *result, quick bool) error {
	warm, minCalls, minRun := 2, 3, p.minRun
	if quick {
		warm, minCalls, minRun = 0, 1, 0
	}
	for range warm {
		if err := run(); err != nil {
			return err
		}
	}
	var e0 hwnpu.PowerSample
	if p.meter {
		e0, _ = p.dev.PowerSample()
	}
	c0, t0 := cpuTime(), time.Now()
	calls := 0
	for calls < minCalls || (time.Since(t0) < minRun && calls < 1000) {
		if err := run(); err != nil {
			return err
		}
		calls++
	}
	wall := time.Since(t0)
	r.Calls = calls
	r.MsPerCall = wall.Seconds() * 1000 / float64(calls)
	r.CPUUse = float64(cpuTime()-c0) / float64(wall)
	if p.meter {
		if e1, err := p.dev.PowerSample(); err == nil {
			r.NetW = hwnpu.AveragePower(e0, e1) - p.idleW
			r.MJPerCall = r.NetW * r.MsPerCall
		}
	}
	return nil
}

func paramsAndBytes(format string, layers int) (int64, int64) {
	return int64(layers) * probe.LayerParams, probe.WeightBytes(format, layers)
}

// try loads, times and closes one model.
func (p *prober) try(format string, layers, tokens int, units string, quickOffNPU bool) result {
	var r result
	l := p.load(format, layers, tokens, units, uint64(layers), &r)
	if l != nil {
		quick := quickOffNPU && !(r.OtherOps == 0 && r.NPUOps > 0)
		if err := p.time(l, &r, quick); err != nil {
			r.Error = "run: " + err.Error()
		}
		l.close()
	}
	r.OnNPU = r.Error == "" && r.Verified && r.NPUOps > 0 && r.OtherOps == 0
	return r
}

func logResult(r result) {
	line := fmt.Sprintf("  %-5s %3d layers %6.2f GiB %-4s %-18s build %5.1fs compile %6.1fs", r.Format, r.Layers, probe.GiB(r.WeightBytes), r.Units, r.placement(), r.BuildS, r.CompileS)
	if r.Tokens > 1 {
		line = fmt.Sprintf("  K=%-2d", r.Tokens) + line[1:]
	}
	if r.Copy > 0 {
		line = fmt.Sprintf("  model %d: ", r.Copy) + line[2:]
	}
	if r.MsPerCall > 0 {
		line += fmt.Sprintf("  %8.1f ms/call, CPU use %.2f", r.MsPerCall, r.CPUUse)
	}
	if r.MJPerCall != 0 {
		line += fmt.Sprintf(", %.0f mJ/call", r.MJPerCall)
	}
	if r.Error != "" {
		line += "  error: " + short(r.Error, 160)
	}
	fmt.Println(line)
}

func short(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}
