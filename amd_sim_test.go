package hwnpu_test

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/amd"
	"github.com/shibukawa/hwnpu/internal/xdna"
	"github.com/shibukawa/hwnpu/internal/xdna/xdnasim"
)

// A simulated AMD NPU registered as backend "xdna-sim" (see internal/xdna/xdnasim).
func init() {
	backend.Register(amd.Provider{
		ProviderName: "xdna-sim",
		Nodes: func() ([]amd.Node, error) {
			return []amd.Node{{Path: "/dev/accel/accel9", Device: "0x17f0", Revision: "0x10"}}, nil
		},
		OpenDriver: func(string) (xdna.Driver, error) {
			sim := xdnasim.New()
			sim.Meta.Cols = 8 // npu4 (Strix Point) has 8 columns
			sim.Run = xdnasim.RunPassthrough
			return sim, nil
		},
	})
}

func openSimAMD(t *testing.T, opts ...hwnpu.Option) *hwnpu.Device {
	t.Helper()
	dev, err := hwnpu.Open(context.Background(), append([]hwnpu.Option{hwnpu.WithBackend("xdna-sim")}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.Close() })
	return dev
}

// Acceptance milestone:m4-amd/kernel-run through the public API, written like IRON host code.
func TestAMDPassthroughPublicAPI(t *testing.T) {
	ctx := context.Background()
	dev := openSimAMD(t, hwnpu.WithPowerPolicy(hwnpu.PowerLow))
	info := dev.Info()
	if info.Vendor != hwnpu.VendorAMD || info.Name != "AMD NPU npu4 (Strix Point), 8 columns" || info.Caps&hwnpu.CapKernel == 0 {
		t.Fatalf("%+v", info)
	}
	if _, err := dev.Graph(); !errors.Is(err, hwnpu.ErrUnsupported) {
		t.Fatalf("Graph() = %v, want ErrUnsupported", err)
	}

	kr, err := dev.Kernel()
	if err != nil {
		t.Fatal(err)
	}
	prog, err := kr.Load(ctx, hwnpu.ProgramSource{Xclbin: xdnasim.PassthroughXclbin()})
	if err != nil {
		t.Fatal(err)
	}
	defer prog.Close()
	if names := prog.Kernels(); len(names) != 1 || names[0] != "MLIR_AIE" {
		t.Fatalf("kernels %v", names)
	}
	k, err := prog.Kernel("MLIR_AIE")
	if err != nil {
		t.Fatal(err)
	}
	if args := k.Args(); len(args) != 6 || args[3].Name != "bo0" || !args[3].Buffer {
		t.Fatalf("args %+v", args)
	}

	const n = 4096
	insts := xdnasim.PassthroughInstructions(n)
	instr, _ := dev.AllocBuffer(len(insts), hwnpu.WithBufferKind(hwnpu.BufferInstructions))
	in, _ := dev.AllocBuffer(n)
	unused, _ := dev.AllocBuffer(n)
	out, _ := dev.AllocBuffer(n)
	for _, b := range []*hwnpu.Buffer{instr, in, unused, out} {
		defer b.Close()
	}
	copy(instr.Bytes(), insts)
	for i := range in.Bytes() {
		in.Bytes()[i] = byte(i)
	}
	instr.SyncToDevice(0, len(insts))
	in.SyncToDevice(0, n)

	run, err := k.NewRun()
	if err != nil {
		t.Fatal(err)
	}
	defer run.Close()
	run.SetUint64(0, 3).SetBuffer(1, instr).SetUint32(2, uint32(len(insts)/4)).
		SetBuffer(3, in).SetBuffer(4, unused).SetBuffer(5, out)
	if err := run.Run(ctx); err != nil {
		t.Fatal(err)
	}
	out.SyncFromDevice(0, n)
	if !bytes.Equal(in.Bytes(), out.Bytes()) {
		t.Fatal("output differs from input")
	}

	// a setter error is kept and reported by Run
	if err := run.SetUint32(0, 1).Run(ctx); err == nil {
		t.Fatal("wrong scalar size was not reported")
	}

	s, err := dev.PowerSample()
	if err != nil || s.Scope != hwnpu.ScopeNPU || s.Watts <= 0 {
		t.Fatalf("power %+v %v", s, err)
	}
}

func TestKernelUnsupportedOnGraphDevices(t *testing.T) {
	ctx := context.Background()
	infos, _ := hwnpu.Enumerate(ctx)
	for _, info := range infos {
		if info.Caps&hwnpu.CapKernel != 0 {
			continue
		}
		dev, err := hwnpu.OpenDevice(ctx, info)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := dev.Kernel(); !errors.Is(err, hwnpu.ErrUnsupported) {
			t.Errorf("%s: Kernel() = %v", info.Name, err)
		}
		dev.Close()
	}
}
