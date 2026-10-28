package hwnpu_test

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/fixtures"
)

const (
	fp16Model = "internal/fixtures/" + fixtures.CoreMLFP16
	fp32Model = "internal/fixtures/" + fixtures.CoreMLFP32
	batch     = fixtures.Batch
	outDim    = fixtures.Out
)

func openApple(t *testing.T, opts ...hwnpu.Option) *hwnpu.Device {
	t.Helper()
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, append([]hwnpu.Option{hwnpu.WithVendor(hwnpu.VendorApple)}, opts...)...)
	if errors.Is(err, hwnpu.ErrNoDevice) {
		t.Skip("no Apple Neural Engine on this machine")
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.Close() })
	return dev
}

func input() []float32 { return fixtures.LinearInput() }

func reference(t *testing.T) []float32 { return fixtures.LinearOutput() }

// run compiles the model, feeds the reference input and returns the output as float32.
func run(t *testing.T, dev *hwnpu.Device, model string) (*hwnpu.Graph, []float32) {
	t.Helper()
	ctx := context.Background()
	gr, err := dev.Graph()
	if err != nil {
		t.Fatal(err)
	}
	g, err := gr.Compile(ctx, hwnpu.GraphSource{Path: model})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { g.Close() })

	in, ok := g.Arg("x")
	if !ok {
		t.Fatalf("no input x in %v", g.Args())
	}
	out, ok := g.Arg("y")
	if !ok {
		t.Fatalf("no output y in %v", g.Args())
	}
	inBuf, err := dev.AllocBuffer(in.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	defer inBuf.Close()
	outBuf, err := dev.AllocBuffer(out.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	defer outBuf.Close()

	x := input()
	switch in.DType {
	case hwnpu.F16:
		hwnpu.EncodeF16(inBuf.Bytes(), x)
	case hwnpu.F32:
		for i, v := range x {
			binary.LittleEndian.PutUint32(inBuf.Bytes()[4*i:], math.Float32bits(v))
		}
	default:
		t.Fatalf("unexpected input type %v", in.DType)
	}

	req, err := g.NewRequest()
	if err != nil {
		t.Fatal(err)
	}
	defer req.Close()
	if err := req.BindName("x", inBuf); err != nil {
		t.Fatal(err)
	}
	if err := req.BindName("y", outBuf); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ { // the second run checks that a request can be reused
		if err := req.Run(ctx); err != nil {
			t.Fatal(err)
		}
	}

	y := make([]float32, batch*outDim)
	switch out.DType {
	case hwnpu.F16:
		hwnpu.DecodeF16(y, outBuf.Bytes())
	case hwnpu.F32:
		for i := range y {
			y[i] = math.Float32frombits(binary.LittleEndian.Uint32(outBuf.Bytes()[4*i:]))
		}
	}
	return g, y
}

func compare(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	worst := 0.0
	for i := range want {
		d := math.Abs(float64(got[i] - want[i]))
		if d > tol+0.02*math.Abs(float64(want[i])) {
			t.Fatalf("y[%d] = %v, want %v (diff %v)", i, got[i], want[i], d)
		}
		worst = max(worst, d)
	}
	t.Logf("max abs diff %.5f", worst)
}

func TestEnumerateApple(t *testing.T) {
	dev := openApple(t)
	info := dev.Info()
	t.Logf("%+v caps=%v", info, info.Caps)
	if info.Vendor != hwnpu.VendorApple || info.Caps&hwnpu.CapGraph == 0 {
		t.Fatalf("unexpected info %+v", info)
	}
}

// Acceptance milestone:m1-apple/infer and requirement:model-execution-api/run-basic.
func TestFP16RunsOnNeuralEngine(t *testing.T) {
	dev := openApple(t)
	g, y := run(t, dev, fp16Model)
	rep := g.Placement()
	if !rep.OnNPU() || rep.NPUOps == 0 {
		t.Fatalf("placement %+v", rep)
	}
	t.Logf("placement: %d ops on the Neural Engine (method %s)", rep.NPUOps, rep.Method)
	compare(t, y, reference(t), 0.03)
}

// Acceptance milestone:m1-apple/strict and requirement:npu-only-execution/strict-rejects.
func TestStrictRejectsFP32(t *testing.T) {
	dev := openApple(t)
	gr, _ := dev.Graph()
	_, err := gr.Compile(context.Background(), hwnpu.GraphSource{Path: fp32Model})
	if !errors.Is(err, hwnpu.ErrNotOnNPU) {
		t.Fatalf("err = %v, want ErrNotOnNPU", err)
	}
	var pe *hwnpu.PlacementError
	if !errors.As(err, &pe) || !pe.Report.Verified || len(pe.Report.Ops) == 0 {
		t.Fatalf("report %+v", pe)
	}
	found := false
	for _, op := range pe.Report.Ops {
		if op.Name == "linear" && op.Device == "cpu" {
			found = true
		}
	}
	if !found {
		t.Fatalf("linear is not reported as placed on the CPU: %+v", pe.Report.Ops)
	}
	t.Log(err)
}

// Acceptance requirement:npu-only-execution/prefer-reports.
func TestPreferRunsAndReports(t *testing.T) {
	dev := openApple(t, hwnpu.WithPlacement(hwnpu.PlacementPrefer))
	g, y := run(t, dev, fp32Model)
	if rep := g.Placement(); rep.OnNPU() || len(rep.Ops) == 0 {
		t.Fatalf("expected off-NPU operations, got %+v", rep)
	}
	compare(t, y, reference(t), 1e-4)
}

// Acceptance requirement:model-execution-api/tensor-check.
func TestBindRejectsSmallBuffer(t *testing.T) {
	dev := openApple(t)
	gr, _ := dev.Graph()
	g, err := gr.Compile(context.Background(), hwnpu.GraphSource{Path: fp16Model})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	req, _ := g.NewRequest()
	defer req.Close()
	small, _ := dev.AllocBuffer(16)
	defer small.Close()
	if err := req.BindName("x", small); err == nil {
		t.Fatal("Bind accepted a buffer that is too small")
	}
	if _, err := req.Submit(); err == nil {
		t.Fatal("Submit accepted unbound arguments")
	}
}

func TestUnsupportedSource(t *testing.T) {
	dev := openApple(t)
	gr, _ := dev.Graph()
	_, err := gr.Compile(context.Background(), hwnpu.GraphSource{Format: hwnpu.GraphFormatOpenVINOIR})
	if !errors.Is(err, hwnpu.ErrUnsupported) {
		t.Fatalf("err = %v, want ErrUnsupported", err)
	}
}

// Acceptance requirement:compile-cache/reuse and /invalidate (Apple).
func TestCoreMLCompileCache(t *testing.T) {
	dir := t.TempDir()
	dev := openApple(t, hwnpu.WithCacheDir(dir), hwnpu.WithPlacement(hwnpu.PlacementPrefer))
	gr, _ := dev.Graph()
	compile := func(path string) (*hwnpu.Graph, time.Duration) {
		t.Helper()
		start := time.Now()
		g, err := gr.Compile(context.Background(), hwnpu.GraphSource{Path: path})
		if err != nil {
			t.Fatal(err)
		}
		return g, time.Since(start)
	}
	g, cold := compile(fp16Model)
	if g.Cached() {
		t.Fatal("first compile reported a cache hit")
	}
	g.Close()
	entries, _ := filepath.Glob(filepath.Join(dir, "*.mlmodelc"))
	if len(entries) != 1 {
		t.Fatalf("cache entries after Close: %v", entries)
	}
	g, warm := compile(fp16Model)
	if !g.Cached() {
		t.Fatal("second compile did not hit the cache")
	}
	if !g.Placement().OnNPU() {
		t.Fatalf("cached placement %+v", g.Placement())
	}
	g.Close()
	t.Logf("compile: cold %v, cached %v", cold, warm)
	g, _ = compile(fp32Model) // another model: a miss
	if g.Cached() {
		t.Fatal("a different model hit the cache")
	}
	g.Close()
	if entries, _ := filepath.Glob(filepath.Join(dir, "*.mlmodelc")); len(entries) != 2 {
		t.Fatalf("cache entries: %v", entries)
	}
	// the cached graph still computes the right result
	_, y := run(t, dev, fp16Model)
	compare(t, y, reference(t), 0.03)
}

func TestCoreMLHasNoPowerControls(t *testing.T) {
	dev := openApple(t, hwnpu.WithPowerPolicy(hwnpu.PowerLow))
	if c := dev.Info().Caps; c&(hwnpu.CapPowerControl|hwnpu.CapPowerMeter) != 0 {
		t.Fatalf("caps %v", c)
	}
	if _, err := dev.PowerSample(); !errors.Is(err, hwnpu.ErrUnsupported) {
		t.Fatalf("PowerSample err = %v", err)
	}
}

// A batch 1 model is placed on the CPU; the same model padded by tools/coreml/ane_pad.py runs on
// the Neural Engine, while callers keep using the batch 1 shape.
func TestCoreMLBatchPadding(t *testing.T) {
	ctx := context.Background()
	dev := openApple(t)
	gr, _ := dev.Graph()
	if _, err := gr.Compile(ctx, hwnpu.GraphSource{Path: "internal/fixtures/" + fixtures.CoreMLSmall}); !errors.Is(err, hwnpu.ErrNotOnNPU) {
		t.Fatalf("unpadded model: err = %v, want ErrNotOnNPU", err)
	}
	g, err := gr.Compile(ctx, hwnpu.GraphSource{Path: "internal/fixtures/" + fixtures.CoreMLSmallANE})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if rep := g.Placement(); !rep.OnNPU() || rep.NPUOps != 6 {
		t.Fatalf("placement %+v", rep)
	}
	args := g.Args()
	for _, a := range args {
		if len(a.Dims) != 2 || a.Dims[0] != 1 || a.Dims[1] != 128 || a.Bytes != 256 || len(a.PaddedDims) != 2 || a.PaddedDims[0] != 64 {
			t.Fatalf("arg %s dims %v padded %v bytes %d", a.Name, a.Dims, a.PaddedDims, a.Bytes)
		}
	}
	in, _ := dev.AllocBuffer(args[0].Bytes)
	out, _ := dev.AllocBuffer(args[1].Bytes)
	defer in.Close()
	defer out.Close()
	hwnpu.EncodeF16(in.Bytes(), fixtures.SmallInput())
	req, _ := g.NewRequest()
	defer req.Close()
	req.Bind(0, in)
	req.Bind(1, out)
	want := fixtures.SmallOutput()
	for i := 0; i < 2; i++ {
		if err := req.Run(ctx); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, 128)
		hwnpu.DecodeF16(got, out.Bytes())
		compare(t, got, want, 0.02)
	}
}
