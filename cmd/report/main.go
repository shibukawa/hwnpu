// Command report checks every NPU hwnpu can find on this machine with bundled test models and
// prints a report that can be pasted into an issue.
//
//	go run ./cmd/report                  print a Markdown report
//	go run ./cmd/report -o report.md     also write it to a file
//	go run ./cmd/report -json report.json
//	go run ./cmd/report -n 50            number of timed runs per model
//	go run ./cmd/report -power low       open devices with a power policy (default, low, balanced, performance)
//	go run ./cmd/report -cache DIR       use a compile cache (run twice to see cached compile times)
//	go run ./cmd/report -xclbin final.xclbin -insts insts.bin
//	                                     AMD: also run an IRON passthrough design (see README)
//
// For each device it compiles the models that suit its backend in strict mode, runs them,
// compares the output with a reference computed in fp32, and measures latency.
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/accelfs"
	"github.com/shibukawa/hwnpu/internal/fixtures"
	"github.com/shibukawa/hwnpu/internal/ze"
)

// Status of one check.
type Status string

const (
	Pass Status = "PASS"
	Fail Status = "FAIL"
	Skip Status = "SKIP"
)

// Report is the full result, also written as JSON.
type Report struct {
	Time        time.Time      `json:"time"`
	Environment Environment    `json:"environment"`
	EnumError   string         `json:"enumerate_error,omitempty"`
	Devices     []DeviceResult `json:"devices"`
}

type Environment struct {
	OS          string   `json:"os"`
	Arch        string   `json:"arch"`
	GoVersion   string   `json:"go_version"`
	CGO         string   `json:"cgo_enabled"`
	Module      string   `json:"module_version"`
	ZeLoaderEnv string   `json:"hwnpu_ze_loader,omitempty"`
	ZeLoaders   []string `json:"ze_loader_candidates"`
	ZeLoader    string   `json:"ze_loader_status"`
	AccelNodes  []string `json:"accel_nodes,omitempty"`
}

type DeviceResult struct {
	Info    hwnpu.DeviceInfo `json:"info"`
	Caps    string           `json:"caps"`
	Power   string           `json:"power_policy"`
	Meter   string           `json:"power_meter"`
	OpenErr string           `json:"open_error,omitempty"`
	Checks  []Check          `json:"checks"`
}

type Check struct {
	Name       string   `json:"name"`
	Status     Status   `json:"status"`
	Detail     string   `json:"detail,omitempty"`
	Args       []string `json:"args,omitempty"`
	Placement  string   `json:"placement,omitempty"`
	MaxAbsDiff float64  `json:"max_abs_diff,omitempty"`
	CompileMS  float64  `json:"compile_ms,omitempty"`
	FirstRunMS float64  `json:"first_run_ms,omitempty"`
	MedianMS   float64  `json:"median_ms,omitempty"`
	MinMS      float64  `json:"min_ms,omitempty"`
	Runs       int      `json:"runs,omitempty"`
	Cached     bool     `json:"cached,omitempty"`
	// Energy over the timed runs, when the device has a power meter.
	EnergyPerRunMJ float64 `json:"energy_per_run_mj,omitempty"`
	AverageW       float64 `json:"average_w,omitempty"`
	EnergyScope    string  `json:"energy_scope,omitempty"`
}

// testCase is a model run on devices of one backend.
type testCase struct {
	name    string
	backend string
	path    string // inside the extracted fixtures
	// expectReject: strict mode must refuse the model (ErrNotOnNPU).
	expectReject bool
	input        func() []float32
	want         func() []float32
	// tolerance: |got - want| <= abs + rel*|want|. NPUs compute in fp16.
	abs, rel float64
}

var cases = []testCase{
	{name: "linear_relu 64x512x512 (Core ML, fp16)", backend: "coreml", path: fixtures.CoreMLFP16,
		input: fixtures.LinearInput, want: fixtures.LinearOutput, abs: 0.05, rel: 0.02},
	{name: "linear_relu 64x512x512 (Core ML, fp32: strict mode must reject)", backend: "coreml", path: fixtures.CoreMLFP32,
		expectReject: true},
	{name: "mlp_small 1x128, 3 layers (Core ML, fp16: too small for the Neural Engine, strict mode must reject)", backend: "coreml", path: fixtures.CoreMLSmall,
		expectReject: true},
	{name: "mlp_small 1x128 padded to batch 64 by ane_pad.py (Core ML, fp16)", backend: "coreml", path: fixtures.CoreMLSmallANE,
		input: fixtures.SmallInput, want: fixtures.SmallOutput, abs: 0.05, rel: 0.02},
	{name: "affine_relu 1x8 (OpenVINO IR, fp32 I/O)", backend: "levelzero", path: fixtures.OpenVINOAffine,
		input: func() []float32 { return fixtures.AffineInput }, want: fixtures.AffineOutput, abs: 0.01, rel: 0.01},
	{name: "linear_relu 64x512x512 (OpenVINO IR, fp32 I/O)", backend: "levelzero", path: fixtures.OpenVINOLinear,
		input: fixtures.LinearInput, want: fixtures.LinearOutput, abs: 0.05, rel: 0.02},
}

func main() {
	out := flag.String("o", "", "also write the Markdown report to this file")
	jsonOut := flag.String("json", "", "also write the report as JSON to this file")
	runs := flag.Int("n", 20, "timed runs per model")
	powerFlag := flag.String("power", "default", "power policy: default, low, balanced, performance")
	cacheDir := flag.String("cache", "", "compile cache directory")
	flag.StringVar(&amdDesign.xclbin, "xclbin", "", "AMD: xclbin of an IRON passthrough design (opcode, instr, ninstr, in, unused, out)")
	flag.StringVar(&amdDesign.insts, "insts", "", "AMD: instruction stream of that design (insts.bin, or insts.txt with one hex word per line)")
	flag.IntVar(&amdDesign.bytes, "bytes", 16384, "AMD: bytes the passthrough design copies")
	flag.StringVar(&amdDesign.kernel, "kernel", "MLIR_AIE", "AMD: kernel name in the xclbin")
	flag.Parse()

	opts := []hwnpu.Option{hwnpu.WithCacheDir(*cacheDir)}
	switch *powerFlag {
	case "default":
	case "low":
		opts = append(opts, hwnpu.WithPowerPolicy(hwnpu.PowerLow))
	case "balanced":
		opts = append(opts, hwnpu.WithPowerPolicy(hwnpu.PowerBalanced))
	case "performance":
		opts = append(opts, hwnpu.WithPowerPolicy(hwnpu.PowerPerformance))
	default:
		fmt.Fprintln(os.Stderr, "unknown -power value:", *powerFlag)
		os.Exit(1)
	}

	rep := run(context.Background(), *runs, opts)
	md := markdown(rep)
	fmt.Print(md)
	if *out != "" {
		if err := os.WriteFile(*out, []byte(md), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	if *jsonOut != "" {
		b, _ := json.MarshalIndent(rep, "", "  ")
		if err := os.WriteFile(*jsonOut, append(b, '\n'), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	for _, d := range rep.Devices {
		for _, c := range d.Checks {
			if c.Status == Fail {
				os.Exit(2)
			}
		}
		if d.OpenErr != "" {
			os.Exit(2)
		}
	}
	if len(rep.Devices) == 0 {
		os.Exit(1)
	}
}

func run(ctx context.Context, runs int, opts []hwnpu.Option) Report {
	rep := Report{Time: time.Now().UTC().Truncate(time.Second), Environment: environment()}
	infos, err := hwnpu.Enumerate(ctx)
	if err != nil {
		rep.EnumError = err.Error()
	}
	dir, err := os.MkdirTemp("", "hwnpu-report")
	if err != nil {
		rep.EnumError += "\n" + err.Error()
		return rep
	}
	defer os.RemoveAll(dir)
	if err := fixtures.Extract(dir); err != nil {
		rep.EnumError += "\n" + err.Error()
		return rep
	}
	for _, info := range infos {
		rep.Devices = append(rep.Devices, checkDevice(ctx, info, dir, runs, opts))
	}
	return rep
}

func environment() Environment {
	env := Environment{
		OS: runtime.GOOS, Arch: runtime.GOARCH, GoVersion: runtime.Version(),
		ZeLoaderEnv: os.Getenv(ze.LoaderEnv), ZeLoaders: ze.LoaderCandidates(), Module: "(devel)",
	}
	if bi, ok := debug.ReadBuildInfo(); ok {
		if bi.Main.Version != "" {
			env.Module = bi.Main.Version
		}
		for _, s := range bi.Settings {
			if s.Key == "CGO_ENABLED" {
				env.CGO = s.Value
			}
		}
	}
	// Level Zero is loaded here as well, so that a missing loader is visible even on systems
	// where no device node tells that an Intel NPU exists (Windows).
	if api, err := ze.Load(); err != nil {
		env.ZeLoader = "not loaded: " + err.Error()
	} else {
		env.ZeLoader = "loaded " + api.Lib.Path
	}
	if nodes, err := accelfs.List(); err == nil {
		for _, n := range nodes {
			env.AccelNodes = append(env.AccelNodes, fmt.Sprintf("%s driver=%s pci=%s:%s", n.Path, n.Driver, n.Vendor, n.Device))
		}
	}
	return env
}

func checkDevice(ctx context.Context, info hwnpu.DeviceInfo, dir string, runs int, opts []hwnpu.Option) DeviceResult {
	res := DeviceResult{Info: info, Caps: info.Caps.String()}
	dev, err := hwnpu.OpenDevice(ctx, info, opts...)
	if err != nil {
		res.OpenErr = err.Error()
		return res
	}
	defer dev.Close()
	res.Power = dev.PowerPolicy().String()
	if info.Caps&hwnpu.CapPowerControl == 0 {
		res.Power += " (ignored: no power controls)"
	}
	if s, err := dev.PowerSample(); err == nil {
		res.Meter = fmt.Sprintf("%s (%s)", s.Source, s.Scope)
	} else {
		res.Meter = "none: " + err.Error()
	}
	if info.Caps&hwnpu.CapKernel != 0 {
		res.Checks = append(res.Checks, checkKernelDevice(ctx, dev, runs)...)
		return res
	}
	any := false
	for _, tc := range cases {
		if tc.backend != info.Backend {
			continue
		}
		any = true
		res.Checks = append(res.Checks, safeCheck(ctx, dev, tc, dir, runs))
	}
	if !any {
		res.Checks = append(res.Checks, Check{Name: "models", Status: Skip, Detail: "no bundled model for backend " + info.Backend})
	}
	return res
}

func safeCheck(ctx context.Context, dev *hwnpu.Device, tc testCase, dir string, runs int) (c Check) {
	defer func() {
		if r := recover(); r != nil {
			c = Check{Name: tc.name, Status: Fail, Detail: fmt.Sprintf("panic: %v\n%s", r, debug.Stack())}
		}
	}()
	return checkModel(ctx, dev, tc, dir, runs)
}

func checkModel(ctx context.Context, dev *hwnpu.Device, tc testCase, dir string, runs int) Check {
	c := Check{Name: tc.name}
	gr, err := dev.Graph()
	if err != nil {
		c.Status, c.Detail = Skip, err.Error()
		return c
	}
	start := time.Now()
	g, err := gr.Compile(ctx, hwnpu.GraphSource{Path: filepath.Join(dir, filepath.FromSlash(tc.path))})
	c.CompileMS = ms(time.Since(start))
	if tc.expectReject {
		var pe *hwnpu.PlacementError
		switch {
		case errors.As(err, &pe):
			c.Status, c.Placement = Pass, placement(pe.Report)
			c.Detail = "rejected as expected: " + err.Error()
		case err == nil:
			c.Status, c.Placement = Fail, placement(g.Placement())
			c.Detail = "strict mode accepted a model that should not run on the NPU"
			g.Close()
		default:
			c.Status, c.Detail = Fail, err.Error()
		}
		return c
	}
	if err != nil {
		c.Status, c.Detail = Fail, err.Error()
		var pe *hwnpu.PlacementError
		if errors.As(err, &pe) {
			c.Placement = placement(pe.Report)
		}
		return c
	}
	defer g.Close()
	c.Placement = placement(g.Placement())
	c.Cached = g.Cached()

	var in, out *hwnpu.ArgInfo
	for _, a := range g.Args() {
		c.Args = append(c.Args, a.String()+modelType(a))
		if a.Kind == hwnpu.ArgInput && in == nil {
			in = &a
		} else if a.Kind == hwnpu.ArgOutput && out == nil {
			out = &a
		}
	}
	if in == nil || out == nil {
		c.Status, c.Detail = Fail, "the graph has no input or output"
		return c
	}
	inBuf, err := dev.AllocBuffer(in.Bytes)
	if err != nil {
		c.Status, c.Detail = Fail, err.Error()
		return c
	}
	defer inBuf.Close()
	outBuf, err := dev.AllocBuffer(out.Bytes)
	if err != nil {
		c.Status, c.Detail = Fail, err.Error()
		return c
	}
	defer outBuf.Close()
	if err := encode(inBuf.Bytes(), in.DType, tc.input()); err != nil {
		c.Status, c.Detail = Fail, err.Error()
		return c
	}
	inBuf.SyncToDevice(0, in.Bytes)

	req, err := g.NewRequest()
	if err != nil {
		c.Status, c.Detail = Fail, err.Error()
		return c
	}
	defer req.Close()
	if err := req.Bind(in.Index, inBuf); err != nil {
		c.Status, c.Detail = Fail, err.Error()
		return c
	}
	if err := req.Bind(out.Index, outBuf); err != nil {
		c.Status, c.Detail = Fail, err.Error()
		return c
	}
	start = time.Now()
	if err := req.Run(ctx); err != nil {
		c.Status, c.Detail = Fail, "run: "+err.Error()
		return c
	}
	c.FirstRunMS = ms(time.Since(start))
	outBuf.SyncFromDevice(0, out.Bytes)

	got, err := decode(outBuf.Bytes(), out.DType)
	if err != nil {
		c.Status, c.Detail = Fail, err.Error()
		return c
	}
	want := tc.want()
	if len(got) < len(want) {
		c.Status, c.Detail = Fail, fmt.Sprintf("output has %d values, want %d", len(got), len(want))
		return c
	}
	c.Status = Pass
	for i, w := range want {
		d := math.Abs(float64(got[i] - w))
		c.MaxAbsDiff = max(c.MaxAbsDiff, d)
		if d > tc.abs+tc.rel*math.Abs(float64(w)) && c.Status == Pass {
			c.Status = Fail
			c.Detail = fmt.Sprintf("y[%d] = %v, want %v", i, got[i], w)
		}
	}

	times := make([]float64, 0, runs)
	before, meterErr := dev.PowerSample()
	for i := 0; i < runs; i++ {
		start := time.Now()
		if err := req.Run(ctx); err != nil {
			c.Status, c.Detail = Fail, fmt.Sprintf("run %d: %v", i+2, err)
			return c
		}
		times = append(times, ms(time.Since(start)))
	}
	if after, err := dev.PowerSample(); meterErr == nil && err == nil && len(times) > 0 {
		c.EnergyPerRunMJ = (after.EnergyJ - before.EnergyJ) / float64(len(times)) * 1000
		c.AverageW = hwnpu.AveragePower(before, after)
		c.EnergyScope = string(after.Scope)
	}
	if len(times) > 0 {
		slices.Sort(times)
		c.Runs, c.MinMS, c.MedianMS = len(times), times[0], times[len(times)/2]
	}
	return c
}

func modelType(a hwnpu.ArgInfo) string {
	if a.ModelDType != hwnpu.DTypeUnknown && a.ModelDType != a.DType {
		return fmt.Sprintf(" model=%s", a.ModelDType)
	}
	return ""
}

func placement(r hwnpu.PlacementReport) string {
	switch {
	case !r.Verified:
		return "not verified: " + r.Reason
	case len(r.Ops) == 0:
		return fmt.Sprintf("all %d ops on NPU (%s)", r.NPUOps, r.Method)
	}
	ops := make([]string, 0, len(r.Ops))
	for _, op := range r.Ops {
		ops = append(ops, fmt.Sprintf("%s(%s)->%s", op.Type, op.Name, op.Device))
	}
	return fmt.Sprintf("%d on NPU, %d elsewhere (%s): %s", r.NPUOps, len(r.Ops), r.Method, strings.Join(ops, ", "))
}

func encode(dst []byte, t hwnpu.DType, v []float32) error {
	switch t {
	case hwnpu.F32:
		for i, x := range v {
			binary.LittleEndian.PutUint32(dst[4*i:], math.Float32bits(x))
		}
	case hwnpu.F16:
		hwnpu.EncodeF16(dst[:2*len(v)], v)
	default:
		return fmt.Errorf("input type %s is not supported by the report", t)
	}
	return nil
}

func decode(src []byte, t hwnpu.DType) ([]float32, error) {
	switch t {
	case hwnpu.F32:
		out := make([]float32, len(src)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[4*i:]))
		}
		return out, nil
	case hwnpu.F16:
		out := make([]float32, len(src)/2)
		hwnpu.DecodeF16(out, src)
		return out, nil
	}
	return nil, fmt.Errorf("output type %s is not supported by the report", t)
}

func ms(d time.Duration) float64 { return math.Round(float64(d.Microseconds())) / 1000 }

func markdown(r Report) string {
	var b strings.Builder
	w := func(format string, a ...any) { fmt.Fprintf(&b, format, a...) }
	w("# hwnpu report\n\n")
	e := r.Environment
	w("| item | value |\n|---|---|\n")
	w("| time | %s |\n| os/arch | %s/%s |\n| go | %s (CGO_ENABLED=%s) |\n| hwnpu | %s |\n", r.Time.Format(time.RFC3339), e.OS, e.Arch, e.GoVersion, orDash(e.CGO), e.Module)
	w("| Level Zero loader | %s |\n", loaderLine(e))
	if len(e.AccelNodes) > 0 {
		w("| /dev/accel | %s |\n", strings.Join(e.AccelNodes, "<br>"))
	}
	w("\n")
	if r.EnumError != "" {
		w("## Enumeration warnings\n\n```\n%s\n```\n\n", strings.TrimSpace(r.EnumError))
	}
	if len(r.Devices) == 0 {
		w("**No NPU found.**\n")
		return b.String()
	}
	for _, d := range r.Devices {
		w("## [%d] %s\n\n", d.Info.Index, d.Info.Name)
		w("- vendor: %s, backend: %s, caps: %s\n", d.Info.Vendor, d.Info.Backend, d.Caps)
		if d.Info.Driver != "" {
			w("- driver: %s\n", d.Info.Driver)
		}
		if d.Info.Path != "" {
			w("- path: %s\n", d.Info.Path)
		}
		if d.Power != "" {
			w("- power policy: %s\n- power meter: %s\n", d.Power, d.Meter)
		}
		w("\n")
		if d.OpenErr != "" {
			w("**FAIL** open: `%s`\n\n", d.OpenErr)
			continue
		}
		w("| model | result | max abs diff | compile ms | first run ms | median ms | min ms | energy/run mJ | avg W |\n|---|---|---|---|---|---|---|---|---|\n")
		for _, c := range d.Checks {
			compile := num(c.CompileMS, 1)
			if c.Cached {
				compile += " (cached)"
			}
			energy := num(c.EnergyPerRunMJ, 3)
			if c.EnergyScope != "" {
				energy += " (" + c.EnergyScope + ")"
			}
			w("| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", c.Name, c.Status, num(c.MaxAbsDiff, 5), compile, num(c.FirstRunMS, 2), num(c.MedianMS, 3), num(c.MinMS, 3), energy, num(c.AverageW, 2))
		}
		w("\n")
		for _, c := range d.Checks {
			if c.Placement == "" && c.Detail == "" && len(c.Args) == 0 {
				continue
			}
			w("<details><summary>%s: %s</summary>\n\n", c.Status, c.Name)
			if c.Placement != "" {
				w("- placement: %s\n", c.Placement)
			}
			for _, a := range c.Args {
				w("- arg: %s\n", a)
			}
			if c.Detail != "" {
				w("\n```\n%s\n```\n", strings.TrimSpace(c.Detail))
			}
			w("\n</details>\n\n")
		}
	}
	return b.String()
}

func loaderLine(e Environment) string {
	search := "search: " + strings.Join(e.ZeLoaders, ", ")
	if e.ZeLoaderEnv != "" {
		search = fmt.Sprintf("%s=%s", ze.LoaderEnv, e.ZeLoaderEnv)
	}
	return search + "<br>" + strings.ReplaceAll(e.ZeLoader, "|", "\\|")
}

func num(v float64, prec int) string {
	if v == 0 {
		return "-"
	}
	return fmt.Sprintf("%.*f", prec, v)
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
