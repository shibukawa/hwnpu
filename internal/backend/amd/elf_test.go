package amd

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"testing"

	"github.com/shibukawa/hwnpu/internal/aieelf"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/types"
	"github.com/shibukawa/hwnpu/internal/xdna/xdnasim"
)

var testPDI = []byte("SIMULATED-PDI:elf-copy")

// copyELF is a full ELF whose control code copies n bytes from argument 0 to argument 1:
// word 0-1 the PDI's address (address_64), word 2 n, a buffer descriptor for argument 0 at
// byte 16 and one for argument 1 at byte 48 (shim_dma_48, argument 1 at an addend of 0x40).
func copyELF(n int) []byte {
	ctrl := make([]byte, 64)
	binary.LittleEndian.PutUint32(ctrl[8:], uint32(n))
	return aieelf.Build(aieelf.BuildSpec{Columns: 4, PDIs: map[string][]byte{".pdi.1": testPDI},
		Instances: []aieelf.BuildInstance{{Kernel: "main", Name: "sequence", Args: 2, Ctrl: ctrl, Relocs: []aieelf.Reloc{
			{Offset: 0, Scheme: aieelf.SchemeAddress64, Target: aieelf.TargetPDI, PDI: ".pdi.1"},
			{Offset: 16, Scheme: aieelf.SchemeShimDMA48, Target: aieelf.TargetArg, Arg: 0},
			{Offset: 48, Scheme: aieelf.SchemeShimDMA48, Target: aieelf.TargetArg, Arg: 1, Addend: 0x40},
		}}}})
}

// runCopyELF is the simulated work of copyELF's control code.
func runCopyELF(e xdnasim.ExecELF) error {
	le := binary.LittleEndian
	pdi, err := e.Mem(le.Uint64(e.Ctrl), len(testPDI))
	if err != nil || !bytes.Equal(pdi, testPDI) {
		return fmt.Errorf("PDI: %v", err)
	}
	n := int(le.Uint32(e.Ctrl[8:]))
	bd := func(o int) uint64 {
		return (uint64(le.Uint32(e.Ctrl[o+8:])&0xffff)<<32 | uint64(le.Uint32(e.Ctrl[o+4:]))) - aieelf.DDRApertureOffset
	}
	src, err := e.Mem(bd(16), n)
	if err != nil {
		return err
	}
	dst, err := e.Mem(bd(48), n)
	if err != nil {
		return err
	}
	copy(dst, src)
	return nil
}

func TestELFRun(t *testing.T) {
	sim := xdnasim.New()
	sim.RunELF = runCopyELF
	d := open(t, sim, backend.Options{})
	defer d.Close()
	const n = 256
	prog, err := d.LoadProgram(context.Background(), types.ProgramSource{Xclbin: copyELF(n)}) // told apart by content
	if err != nil {
		t.Fatal(err)
	}
	ks := prog.Kernels()
	if len(ks) != 1 || ks[0].Name != "main:sequence" || len(ks[0].Args) != 2 || !ks[0].Args[1].Buffer {
		t.Fatalf("%+v", ks)
	}
	if sim.LiveHwctxs != 1 {
		t.Fatal("no hardware context")
	}
	r, err := prog.NewRun("main:sequence")
	if err != nil {
		t.Fatal(err)
	}
	in, _ := d.AllocBuffer(n, types.BufferDefault)
	outA, _ := d.AllocBuffer(n+0x40, types.BufferDefault)
	outB, _ := d.AllocBuffer(n+0x40, types.BufferDefault)
	for i := range in.Bytes() {
		in.Bytes()[i] = byte(i*7 + 3)
	}
	in.SyncToDevice(0, n)
	for _, out := range []backend.Buffer{outA, outB, outA} { // rebinding patches again
		clear(out.Bytes())
		if err := r.SetBuffer(0, in); err != nil {
			t.Fatal(err)
		}
		if err := r.SetBuffer(1, out); err != nil {
			t.Fatal(err)
		}
		f, err := r.Submit()
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Wait(context.Background()); err != nil {
			t.Fatal(err)
		}
		out.SyncFromDevice(0, len(out.Bytes()))
		if !bytes.Equal(out.Bytes()[0x40:0x40+n], in.Bytes()) {
			t.Fatal("output differs from input")
		}
		if len(sim.LastArgs) != 2 {
			t.Fatalf("buffer objects passed with the command: %v (want the two arguments only)", sim.LastArgs)
		}
	}
	if err := r.SetArg(0, []byte{1, 2, 3, 4}); !errors.Is(err, types.ErrUnsupported) {
		t.Errorf("scalar argument: %v", err)
	}
	if err := r.SetBuffer(2, in); err == nil {
		t.Error("argument 2 accepted")
	}
	if _, err := prog.NewRun("main:other"); err == nil {
		t.Error("unknown kernel accepted")
	}
	r.Close()
	in.Close()
	outA.Close()
	outB.Close()
	prog.Close()
	if sim.LiveHwctxs != 0 || sim.HeapUsed() < 0 {
		t.Fatal("hardware context leaked")
	}
}

func TestELFRejects(t *testing.T) {
	sim := xdnasim.New() // npu1: 4 columns
	d := open(t, sim, backend.Options{})
	defer d.Close()
	wide := aieelf.Build(aieelf.BuildSpec{Columns: 8, Instances: []aieelf.BuildInstance{{Kernel: "main", Name: "sequence", Args: 1, Ctrl: make([]byte, 16)}}})
	if _, err := d.LoadProgram(context.Background(), types.ProgramSource{ELF: wide}); err == nil {
		t.Error("an ELF wider than the device was accepted")
	}
	sim.RunELF = func(xdnasim.ExecELF) error { return errors.New("fault") }
	prog, err := d.LoadProgram(context.Background(), types.ProgramSource{ELF: copyELF(16)})
	if err != nil {
		t.Fatal(err)
	}
	defer prog.Close()
	r, _ := prog.NewRun("main:sequence")
	defer r.Close()
	f, err := r.Submit() // unset arguments are patched as address 0, as XRT leaves them
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Wait(context.Background()); err == nil {
		t.Error("a failed command reported success")
	}
}
