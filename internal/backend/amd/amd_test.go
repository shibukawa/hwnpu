package amd

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/types"
	"github.com/shibukawa/hwnpu/internal/xdna"
	"github.com/shibukawa/hwnpu/internal/xdna/xdnasim"
)

func simProvider(sim *xdnasim.Sim) Provider {
	return Provider{
		ProviderName: "xdna-sim",
		Nodes: func() ([]Node, error) {
			return []Node{{Path: "/dev/accel/accel0", Device: "0x1502", Revision: "0x0"}}, nil
		},
		OpenDriver: func(string) (xdna.Driver, error) { return sim, nil },
	}
}

func open(t *testing.T, sim *xdnasim.Sim, opts backend.Options) *device {
	t.Helper()
	p := simProvider(sim)
	infos, err := p.Enumerate(context.Background())
	if err != nil || len(infos) != 1 {
		t.Fatalf("enumerate: %v %v", infos, err)
	}
	be, err := p.Open(context.Background(), infos[0], opts)
	if err != nil {
		t.Fatal(err)
	}
	return be.(*device)
}

func TestEnumerate(t *testing.T) {
	infos, err := simProvider(xdnasim.New()).Enumerate(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	i := infos[0]
	if i.Vendor != types.VendorAMD || !strings.Contains(i.Name, "npu1") || !strings.Contains(i.Name, "4 columns") ||
		i.Firmware != "1.5.4.361" || i.Caps&(types.CapKernel|types.CapPowerMeter|types.CapPowerControl) == 0 || i.Caps&types.CapGraph != 0 {
		t.Fatalf("%+v caps=%v", i, i.Caps)
	}
}

type passthrough struct {
	prog    backend.Program
	run     backend.Run
	instr   backend.Buffer
	a, b, c backend.Buffer
	n       int
}

func setup(t *testing.T, d *device, n int) *passthrough {
	t.Helper()
	prog, err := d.LoadProgram(context.Background(), types.ProgramSource{Xclbin: xdnasim.PassthroughXclbin()})
	if err != nil {
		t.Fatal(err)
	}
	ks := prog.Kernels()
	if len(ks) != 1 || ks[0].Name != xdnasim.PassthroughKernel || len(ks[0].Args) != 6 || !ks[0].Args[1].Buffer || ks[0].Args[2].Buffer {
		t.Fatalf("kernels %+v", ks)
	}
	p := &passthrough{prog: prog, n: n}
	ins := xdnasim.PassthroughInstructions(n)
	if p.instr, err = d.AllocBuffer(len(ins), types.BufferInstructions); err != nil {
		t.Fatal(err)
	}
	copy(p.instr.Bytes(), ins)
	p.instr.SyncToDevice(0, len(ins))
	for _, b := range []*backend.Buffer{&p.a, &p.b, &p.c} {
		if *b, err = d.AllocBuffer(n, types.BufferDefault); err != nil {
			t.Fatal(err)
		}
	}
	for i := range p.a.Bytes() {
		p.a.Bytes()[i] = byte(i * 7)
	}
	p.a.SyncToDevice(0, n)
	if p.run, err = prog.NewRun(xdnasim.PassthroughKernel); err != nil {
		t.Fatal(err)
	}
	r := p.run
	for _, err := range []error{
		r.SetArg(0, binary.LittleEndian.AppendUint64(nil, xdnasim.PassthroughOpcode)),
		r.SetBuffer(1, p.instr),
		r.SetArg(2, binary.LittleEndian.AppendUint32(nil, uint32(len(ins)/4))),
		r.SetBuffer(3, p.a), r.SetBuffer(4, p.b), r.SetBuffer(5, p.c),
	} {
		if err != nil {
			t.Fatal(err)
		}
	}
	return p
}

func (p *passthrough) close() {
	p.run.Close()
	for _, b := range []backend.Buffer{p.instr, p.a, p.b, p.c} {
		b.Close()
	}
	p.prog.Close()
}

// Acceptance milestone:m4-amd/kernel-run.
func TestPassthroughRun(t *testing.T) {
	sim := xdnasim.New()
	sim.Run = xdnasim.RunPassthrough
	d := open(t, sim, backend.Options{})
	p := setup(t, d, 4096)
	if sim.LiveHwctxs != 1 || sim.HeapUsed() == 0 {
		t.Fatalf("hwctx %d heap %d", sim.LiveHwctxs, sim.HeapUsed())
	}
	for i := 0; i < 2; i++ { // a run can be submitted again
		f, err := p.run.Submit()
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Wait(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	p.c.SyncFromDevice(0, p.n)
	if !bytes.Equal(p.a.Bytes(), p.c.Bytes()) {
		t.Fatal("output differs from input")
	}
	if len(sim.LastArgs) != 4 {
		t.Fatalf("argument BOs passed with the command: %v (want instr, bo0, bo1, bo2)", sim.LastArgs)
	}
	p.close()
	if sim.LiveHwctxs != 0 {
		t.Fatal("hardware context leaked")
	}
	if err := d.Close(); err != nil {
		t.Fatal(err)
	}
	if sim.LiveBOs() != 0 {
		t.Fatalf("%d buffer objects leaked", sim.LiveBOs())
	}
}

func TestArgumentChecks(t *testing.T) {
	sim := xdnasim.New()
	sim.Run = xdnasim.RunPassthrough
	d := open(t, sim, backend.Options{})
	defer d.Close()
	prog, err := d.LoadProgram(context.Background(), types.ProgramSource{Xclbin: xdnasim.PassthroughXclbin()})
	if err != nil {
		t.Fatal(err)
	}
	defer prog.Close()
	r, _ := prog.NewRun(xdnasim.PassthroughKernel)
	defer r.Close()
	buf, _ := d.AllocBuffer(64, types.BufferDefault)
	defer buf.Close()
	if err := r.SetArg(0, []byte{1, 2, 3, 4}); err == nil {
		t.Error("wrong scalar size accepted")
	}
	if err := r.SetArg(1, make([]byte, 8)); err == nil {
		t.Error("scalar set on a buffer argument")
	}
	if err := r.SetBuffer(0, buf); err == nil {
		t.Error("buffer set on a scalar argument")
	}
	if err := r.SetArg(9, nil); err == nil {
		t.Error("unknown argument accepted")
	}
	if _, err := r.Submit(); err == nil || !strings.Contains(err.Error(), "not set") {
		t.Errorf("submit with unset arguments: %v", err)
	}
	if _, err := prog.NewRun("nope"); err == nil {
		t.Error("unknown kernel accepted")
	}
}

func TestCommandErrorAndCancel(t *testing.T) {
	sim := xdnasim.New()
	sim.Run = func(xdnasim.Exec) error { return errors.New("simulated fault") }
	d := open(t, sim, backend.Options{})
	defer d.Close()
	p := setup(t, d, 64)
	defer p.close()
	f, err := p.run.Submit()
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Wait(context.Background()); err == nil || !strings.Contains(err.Error(), "ERT_CMD_STATE_ERROR") {
		t.Fatalf("error state not reported: %v", err)
	}

	sim.Run, sim.Delay = xdnasim.RunPassthrough, 300*time.Millisecond
	f, err = p.run.Submit()
	if err != nil {
		t.Fatal(err)
	}
	if done, _ := f.Done(); done {
		t.Fatal("Done before completion")
	}
	if _, err := p.run.Submit(); err == nil {
		t.Fatal("resubmitted while running")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := f.Wait(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Wait = %v, want DeadlineExceeded", err)
	}
	if err := f.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestLoadRejectsXclbinWithoutPDI(t *testing.T) {
	d := open(t, xdnasim.New(), backend.Options{})
	defer d.Close()
	x := xdnasim.PassthroughXclbin()
	// Corrupt the kernel id in the metadata so that no PDI matches.
	x = bytes.Replace(x, []byte(`dpu_kernel_id="0x901"`), []byte(`dpu_kernel_id="0x999"`), 1)
	if _, err := d.LoadProgram(context.Background(), types.ProgramSource{Xclbin: x}); err == nil || !strings.Contains(err.Error(), "no valid DPU kernel") {
		t.Fatalf("err = %v", err)
	}
}

// Acceptance milestone:m4-amd/power.
func TestPowerPolicyAndSensor(t *testing.T) {
	for policy, want := range map[types.PowerPolicy]uint32{
		types.PowerDefault: 0, types.PowerLow: xdna.QoSLow, types.PowerBalanced: xdna.QoSNormal, types.PowerPerformance: xdna.QoSHigh,
	} {
		sim := xdnasim.New()
		d := open(t, sim, backend.Options{Power: policy})
		prog, err := d.LoadProgram(context.Background(), types.ProgramSource{Xclbin: xdnasim.PassthroughXclbin()})
		if err != nil {
			t.Fatal(err)
		}
		if sim.LastQoS.Priority != want {
			t.Errorf("%v: QoS priority %#x, want %#x", policy, sim.LastQoS.Priority, want)
		}
		prog.Close()
		d.Close()
	}

	sim := xdnasim.New()
	d := open(t, sim, backend.Options{})
	defer d.Close()
	a, err := d.PowerSample()
	if err != nil || a.Watts != 1.5 || a.Scope != types.ScopeNPU {
		t.Fatalf("%+v %v", a, err)
	}
	time.Sleep(50 * time.Millisecond)
	b, _ := d.PowerSample()
	if e := b.EnergyJ - a.EnergyJ; e < 1.5*0.04 || e > 1.5*0.5 {
		t.Fatalf("integrated energy %v J over %v", e, b.Time.Sub(a.Time))
	}
}
