package aieelf

import (
	"encoding/binary"
	"os"
	"testing"
)

func TestBuildParse(t *testing.T) {
	ctrl := make([]byte, 96)
	binary.LittleEndian.PutUint32(ctrl[48+8:], 0xabcd0000) // high bits above the 16-bit address field
	spec := BuildSpec{Columns: 8, PDIs: map[string][]byte{".pdi.1": []byte("pdi-one"), ".pdi.7": []byte("unused")},
		Instances: []BuildInstance{{Kernel: "main", Name: "sequence", Args: 3, Ctrl: ctrl, Relocs: []Reloc{
			{Offset: 0, Scheme: SchemeAddress64, Target: TargetPDI, PDI: ".pdi.1"},
			{Offset: 16, Scheme: SchemeShimDMA48, Target: TargetArg, Arg: 0, Addend: 0x200},
			{Offset: 48, Scheme: SchemeShimDMA48, Target: TargetArg, Arg: 2},
			{Offset: 80, Scheme: SchemeScalar32, Target: TargetScratchpad, Size: 0xffff},
		}}}}
	f, err := Parse(Build(spec))
	if err != nil {
		t.Fatal(err)
	}
	if f.Columns != 8 || len(f.Instances) != 1 || string(f.PDIs[".pdi.1"]) != "pdi-one" {
		t.Fatalf("%+v", f)
	}
	in, ok := f.Instance("main:sequence")
	if !ok || in.Args != 3 || len(in.Ctrl) != 96 || len(in.Relocs) != 4 || in.Scratchpad != 0xffff {
		t.Fatalf("%+v", in)
	}
	if p := in.PDIs(); len(p) != 1 || p[0] != ".pdi.1" {
		t.Fatalf("PDIs %v", p)
	}
	r := in.Relocs[1]
	if r.Target != TargetArg || r.Arg != 0 || r.Addend != 0x200 || r.Offset != 16 {
		t.Fatalf("%+v", r)
	}
	code := append([]byte(nil), in.Ctrl...)
	const addr = 0x7a5b_1234_5000
	for i := 0; i < 2; i++ { // patching twice gives the same words
		Patch(code, in.Ctrl, in.Relocs[0], 0x4038000)
		Patch(code, in.Ctrl, in.Relocs[1], addr+uint64(i)*0x1000)
	}
	le := binary.LittleEndian
	if le.Uint64(code[0:]) != 0x4038000 {
		t.Errorf("PDI address %#x", le.Uint64(code[0:]))
	}
	want := uint64(addr+0x1000) + 0x200 + DDRApertureOffset
	got := uint64(le.Uint32(code[16+8:])&0xffff)<<32 | uint64(le.Uint32(code[16+4:]))
	if got != want {
		t.Errorf("shim DMA address %#x, want %#x", got, want)
	}
	Patch(code, in.Ctrl, in.Relocs[2], addr)
	if w2 := le.Uint32(code[48+8:]); w2&0xffff0000 != 0xabcd0000 {
		t.Errorf("high bits of word 2 lost: %#x", w2)
	}
}

func TestDemangle(t *testing.T) {
	for _, c := range []struct {
		in   string
		name string
		args int
	}{{"_Z4mainPcPcPc", "main", 3}, {"_Z13ReLU_b9799113PcPcPc", "ReLU_b9799113", 3}, {"_Z3fooPvPv", "foo", 2}, {"plain", "plain", 0}} {
		n, a, err := demangle(c.in)
		if err != nil || n != c.name || a != c.args {
			t.Errorf("%s: %s %d %v", c.in, n, a, err)
		}
	}
	if _, _, err := demangle("_Z3fooi"); err == nil {
		t.Error("an int argument was accepted")
	}
}

func TestRejects(t *testing.T) {
	ctrl := make([]byte, 32)
	bad := []BuildSpec{
		{Columns: 8, Instances: []BuildInstance{{Kernel: "main", Name: "s", Args: 1, Ctrl: ctrl, Relocs: []Reloc{{Offset: 0, Scheme: 9, Target: TargetArg}}}}},
		{Columns: 8, Instances: []BuildInstance{{Kernel: "main", Name: "s", Args: 1, Ctrl: ctrl, Relocs: []Reloc{{Offset: 0, Scheme: SchemeShimDMA48, Target: TargetArg, Arg: 3}}}}},
		{Columns: 8, Instances: []BuildInstance{{Kernel: "main", Name: "s", Args: 1, Ctrl: ctrl, Relocs: []Reloc{{Offset: 24, Scheme: SchemeShimDMA48, Target: TargetArg}}}}},
		{Columns: 0, Instances: []BuildInstance{{Kernel: "main", Name: "s", Args: 1, Ctrl: ctrl}}},
	}
	for i, s := range bad {
		if _, err := Parse(Build(s)); err == nil {
			t.Errorf("case %d accepted", i)
		}
	}
}

// TestIRON reads a full ELF built by IRON (HWNPU_AIE_ELF), as the ReLU then AXPY graph of
// IRON's programming guide, section 2, compiles it: main:sequence patches three arguments
// and two PDIs.
func TestIRON(t *testing.T) {
	path := os.Getenv("HWNPU_AIE_ELF")
	if path == "" {
		t.Skip("HWNPU_AIE_ELF not set")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	f, err := Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	in, ok := f.Instance("main:sequence")
	if !ok {
		t.Fatalf("no main:sequence in %d instances", len(f.Instances))
	}
	args := map[int]int{}
	for _, r := range in.Relocs {
		if r.Target == TargetArg {
			args[r.Arg]++
		}
	}
	t.Logf("columns %d, %d instances, main:sequence: %d bytes of control code, %d relocations, args %v, PDIs %v",
		f.Columns, len(f.Instances), len(in.Ctrl), len(in.Relocs), args, in.PDIs())
	if f.Columns != 8 || in.Args != 3 || len(args) != 3 || len(in.PDIs()) == 0 {
		t.Fatal("unexpected layout")
	}
}
