//go:build !windows

package hwnpu_test

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/ebitengine/purego"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/fixtures"
	"github.com/shibukawa/hwnpu/internal/ze"
)

// mockLib is the path of the zemock Level Zero stand-in built by TestMain, or "" when no C
// compiler is available. When HWNPU_ZE_LOADER is already set (e.g. to a real loader on an Intel
// machine) the mock is not used and the mock tests skip.
var mockLib string

func TestMain(m *testing.M) {
	if os.Getenv(ze.LoaderEnv) == "" && runtime.GOOS != "windows" {
		if cc, err := exec.LookPath("cc"); err == nil {
			dir, err := os.MkdirTemp("", "zemock")
			if err == nil {
				ext := ".so"
				if runtime.GOOS == "darwin" {
					ext = ".dylib"
				}
				lib := filepath.Join(dir, "libzemock"+ext)
				out, err := exec.Command(cc, "-shared", "-fPIC", "-o", lib, "testdata/zemock/zemock.c").CombinedOutput()
				if err != nil {
					fmt.Fprintf(os.Stderr, "building zemock failed: %v\n%s", err, out)
				} else {
					mockLib = lib
					os.Setenv(ze.LoaderEnv, lib)
				}
				defer os.RemoveAll(dir)
			}
		}
	}
	code := m.Run()
	if mockLib != "" {
		os.RemoveAll(filepath.Dir(mockLib))
	}
	os.Exit(code)
}

func openMock(t *testing.T, opts ...hwnpu.Option) *hwnpu.Device {
	t.Helper()
	if mockLib == "" {
		t.Skip("zemock is not available (no C compiler, or a real Level Zero loader is configured)")
	}
	dev, err := hwnpu.Open(context.Background(), append([]hwnpu.Option{hwnpu.WithBackend("levelzero")}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.Close() })
	return dev
}

func mockString(t *testing.T, sym string) string {
	t.Helper()
	h, err := purego.Dlopen(mockLib, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		t.Fatal(err)
	}
	var fn func() string
	purego.RegisterLibFunc(&fn, h, sym)
	return fn()
}

func mockBool(t *testing.T, sym string) bool {
	t.Helper()
	h, _ := purego.Dlopen(mockLib, purego.RTLD_NOW|purego.RTLD_LOCAL)
	var fn func() int32
	purego.RegisterLibFunc(&fn, h, sym)
	return fn() != 0
}

func f32s(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func putF32s(b []byte, v []float32) {
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
}

// runAffine binds x to the input, runs and returns y.
func runAffine(t *testing.T, dev *hwnpu.Device, g *hwnpu.Graph, x []float32) []float32 {
	t.Helper()
	args := g.Args()
	if len(args) != 2 || args[0].Kind != hwnpu.ArgInput || args[1].Kind != hwnpu.ArgOutput || args[0].DType != hwnpu.F32 {
		t.Fatalf("args %v", args)
	}
	in, _ := dev.AllocBuffer(args[0].Bytes)
	out, _ := dev.AllocBuffer(args[1].Bytes)
	defer in.Close()
	defer out.Close()
	putF32s(in.Bytes(), x)
	req, err := g.NewRequest()
	if err != nil {
		t.Fatal(err)
	}
	defer req.Close()
	if err := req.Bind(0, in); err != nil {
		t.Fatal(err)
	}
	if err := req.Bind(1, out); err != nil {
		t.Fatal(err)
	}
	if err := req.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	return f32s(out.Bytes())
}

var affineInput = fixtures.AffineInput

func checkAffine(t *testing.T, got []float32, scale, bias float32) {
	t.Helper()
	for i, x := range affineInput {
		want := max(scale*x+bias, 0)
		if got[i] != want {
			t.Fatalf("y[%d] = %v, want %v (all: %v)", i, got[i], want, got)
		}
	}
}

func TestMockEnumerate(t *testing.T) {
	dev := openMock(t)
	info := dev.Info()
	t.Logf("%+v caps=%v", info, info.Caps)
	if info.Vendor != hwnpu.VendorIntel || info.Name != "Mock Intel NPU" || info.Caps&hwnpu.CapCompile == 0 ||
		!strings.Contains(info.Driver, "ZE_extension_graph 1.16") || !strings.Contains(info.Driver, "compiler 7.3") {
		t.Fatalf("unexpected info %+v", info)
	}
}

// Acceptance milestone:m2-intel/native-blob.
func TestMockNativeBlob(t *testing.T) {
	dev := openMock(t)
	blob := []byte("ZEMOCKB1")
	blob = binary.LittleEndian.AppendUint32(blob, 8)
	blob = binary.LittleEndian.AppendUint32(blob, math.Float32bits(3))
	blob = binary.LittleEndian.AppendUint32(blob, math.Float32bits(-1))
	gr, _ := dev.Graph()
	g, err := gr.Compile(context.Background(), hwnpu.GraphSource{Blob: blob})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if rep := g.Placement(); !rep.OnNPU() || rep.Method != "levelzero-native-blob" {
		t.Fatalf("placement %+v", rep)
	}
	checkAffine(t, runAffine(t, dev, g, affineInput), 3, -1)
}

// Acceptance milestone:m2-intel/ir-compile.
func TestMockOpenVINOIR(t *testing.T) {
	dev := openMock(t)
	gr, _ := dev.Graph()
	g, err := gr.Compile(context.Background(), hwnpu.GraphSource{Path: "internal/fixtures/openvino/affine_relu.xml", BuildFlags: `--config PERFORMANCE_HINT="LATENCY"`})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	rep := g.Placement()
	if !rep.OnNPU() || rep.Method != "levelzero-compiler" || rep.NPUOps != 3 {
		t.Fatalf("placement %+v", rep)
	}
	flags := mockString(t, "zemock_last_build_flags")
	want := `--inputs_precisions="0:FP32" --inputs_layouts="0:NC" --outputs_precisions="0:FP32" --outputs_layouts="0:NC" --config PERFORMANCE_HINT="LATENCY"`
	if flags != want {
		t.Fatalf("build flags\n got %s\nwant %s", flags, want)
	}
	if !mockBool(t, "zemock_last_rtinfo_ok") {
		t.Fatal("is_new_api / use_indices_for_io_metadata were not injected")
	}
	checkAffine(t, runAffine(t, dev, g, affineInput), 2, -1)
}

// Acceptance milestone:m2-intel/unsupported-ops and requirement:npu-only-execution (Intel).
func TestMockUnsupportedOperation(t *testing.T) {
	dev := openMock(t, hwnpu.WithPlacement(hwnpu.PlacementPrefer)) // even Prefer cannot run it
	src, err := os.ReadFile("internal/fixtures/openvino/affine_relu.xml")
	if err != nil {
		t.Fatal(err)
	}
	xml := strings.Replace(string(src), `type="ReLU"`, `type="Sin"`, 1)
	bin, _ := os.ReadFile("internal/fixtures/openvino/affine_relu.bin")
	gr, _ := dev.Graph()
	_, err = gr.Compile(context.Background(), hwnpu.GraphSource{Format: hwnpu.GraphFormatOpenVINOIR, XML: []byte(xml), Bin: bin})
	if !errors.Is(err, hwnpu.ErrNotOnNPU) {
		t.Fatalf("err = %v, want ErrNotOnNPU", err)
	}
	var pe *hwnpu.PlacementError
	errors.As(err, &pe)
	if len(pe.Report.Ops) != 1 || pe.Report.Ops[0] != (hwnpu.OpPlacement{Name: "y", Type: "Sin", Device: "unsupported"}) || pe.Report.NPUOps != 2 {
		t.Fatalf("report %+v", pe.Report)
	}
	var be *hwnpu.BuildError
	if !errors.As(err, &be) || !strings.Contains(be.Log, "unsupported operation Sin") {
		t.Fatalf("build log missing: %v", err)
	}
	t.Log(err)
}

// Acceptance milestone:m2-intel/build-log.
func TestMockBuildError(t *testing.T) {
	dev := openMock(t)
	gr, _ := dev.Graph()
	_, err := gr.Compile(context.Background(), hwnpu.GraphSource{Blob: []byte("garbage")})
	var be *hwnpu.BuildError
	if !errors.As(err, &be) || be.Log != "not a zemock blob" || errors.Is(err, hwnpu.ErrNotOnNPU) {
		t.Fatalf("err = %#v", err)
	}
	var de *hwnpu.DriverError
	if !errors.As(err, &de) || de.Op != "pfnCreate2" || de.Msg != "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY" {
		t.Fatalf("driver error %#v", de)
	}
}

// Arguments are captured when the graph is appended, so two requests on one graph keep their
// own buffers even though pfnSetArgumentValue writes to the shared graph.
func TestMockRequestsKeepTheirArguments(t *testing.T) {
	dev := openMock(t)
	gr, _ := dev.Graph()
	g, err := gr.Compile(context.Background(), hwnpu.GraphSource{Path: "internal/fixtures/openvino/affine_relu.xml"})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	type pair struct {
		req     *hwnpu.Request
		in, out *hwnpu.Buffer
	}
	var ps []pair
	for k := 0; k < 2; k++ {
		in, _ := dev.AllocBuffer(32)
		out, _ := dev.AllocBuffer(32)
		x := make([]float32, 8)
		for i := range x {
			x[i] = float32(k*10 + i)
		}
		putF32s(in.Bytes(), x)
		req, _ := g.NewRequest()
		req.Bind(0, in)
		req.Bind(1, out)
		ps = append(ps, pair{req, in, out})
	}
	for _, p := range ps {
		if err := p.req.Run(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	for k, p := range ps {
		y := f32s(p.out.Bytes())
		if y[1] != 2*float32(k*10+1)-1 {
			t.Fatalf("request %d: y = %v", k, y)
		}
		p.req.Close()
		p.in.Close()
		p.out.Close()
	}
}

func mockInt(t *testing.T, sym string, args ...uint32) int32 {
	t.Helper()
	h, _ := purego.Dlopen(mockLib, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if len(args) == 0 {
		var fn func() int32
		purego.RegisterLibFunc(&fn, h, sym)
		return fn()
	}
	var fn func(uint32) int32
	purego.RegisterLibFunc(&fn, h, sym)
	return fn(args[0])
}

// Acceptance milestone:m3-power/policy-mapping (Intel).
func TestMockPowerPolicy(t *testing.T) {
	cases := []struct {
		policy                      hwnpu.PowerPolicy
		priority, options, workload int32
	}{
		{hwnpu.PowerDefault, 0, 0, -1},
		{hwnpu.PowerLow, 1, 0, 1},          // low priority, background workload
		{hwnpu.PowerBalanced, 0, 0, -1},    // normal priority
		{hwnpu.PowerPerformance, 2, 1, -1}, // high priority, turbo
	}
	for _, c := range cases {
		dev := openMock(t, hwnpu.WithPowerPolicy(c.policy))
		if dev.Info().Caps&hwnpu.CapPowerControl == 0 || dev.PowerPolicy() != c.policy {
			t.Fatalf("%v: caps %v policy %v", c.policy, dev.Info().Caps, dev.PowerPolicy())
		}
		gr, _ := dev.Graph()
		g, err := gr.Compile(context.Background(), hwnpu.GraphSource{Path: "internal/fixtures/openvino/affine_relu.xml"})
		if err != nil {
			t.Fatal(err)
		}
		checkAffine(t, runAffine(t, dev, g, affineInput), 2, -1)
		got := [3]int32{mockInt(t, "zemock_last_queue_priority"), mockInt(t, "zemock_last_queue_options"), mockInt(t, "zemock_last_workload")}
		if got != [3]int32{c.priority, c.options, c.workload} {
			t.Errorf("%v: priority/options/workload = %v, want %v", c.policy, got, [3]int32{c.priority, c.options, c.workload})
		}
		g.Close()
	}
}

// Acceptance requirement:compile-cache/reuse and /invalidate (Intel).
func TestMockCompileCache(t *testing.T) {
	dir := t.TempDir()
	dev := openMock(t, hwnpu.WithCacheDir(dir))
	gr, _ := dev.Graph()
	compile := func(flags string) *hwnpu.Graph {
		t.Helper()
		g, err := gr.Compile(context.Background(), hwnpu.GraphSource{Path: "internal/fixtures/openvino/affine_relu.xml", BuildFlags: flags})
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	irBefore, nativeBefore := mockInt(t, "zemock_creates", 2), mockInt(t, "zemock_creates", 1)

	g := compile("")
	if g.Cached() || mockInt(t, "zemock_creates", 2) != irBefore+1 {
		t.Fatal("first compile should compile the IR")
	}
	if h, _ := purego.Dlopen(mockLib, purego.RTLD_NOW|purego.RTLD_LOCAL); h != 0 {
		var fn func() uint32
		purego.RegisterLibFunc(&fn, h, "zemock_last_graph_flags")
		if fn()&1 == 0 {
			t.Error("the driver cache was not disabled while hwnpu caches")
		}
	}
	g.Close()
	entries, _ := filepath.Glob(filepath.Join(dir, "*.blob"))
	if len(entries) != 1 {
		t.Fatalf("cache entries: %v", entries)
	}

	g = compile("")
	if !g.Cached() || mockInt(t, "zemock_creates", 2) != irBefore+1 || mockInt(t, "zemock_creates", 1) != nativeBefore+1 {
		t.Fatal("second compile should load the cached blob")
	}
	if rep := g.Placement(); !rep.OnNPU() || rep.NPUOps != 3 {
		t.Fatalf("cached placement %+v", rep)
	}
	checkAffine(t, runAffine(t, dev, g, affineInput), 2, -1)
	g.Close()

	g = compile(`--config PERFORMANCE_HINT="LATENCY"`) // different build flags: a different entry
	if g.Cached() || mockInt(t, "zemock_creates", 2) != irBefore+2 {
		t.Fatal("changed build flags must not hit the cache")
	}
	g.Close()
}

func TestMockPowerSample(t *testing.T) {
	dev := openMock(t)
	s, err := dev.PowerSample()
	if dev.Info().Caps&hwnpu.CapPowerMeter == 0 {
		if !errors.Is(err, hwnpu.ErrUnsupported) {
			t.Fatalf("err = %v, want ErrUnsupported", err)
		}
		return
	}
	if err != nil || s.Scope != hwnpu.ScopePackage {
		t.Fatalf("%+v %v", s, err)
	}
}
