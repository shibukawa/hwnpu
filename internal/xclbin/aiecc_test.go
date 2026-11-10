package xclbin

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// testdata/aiecc holds one build of mlir-aie v1.4.4's passthrough_dmas for npu2: the JSON
// aiecc handed to xclbinutil (PDI path made relative), the PDI, and the xclbin that XRT 2.25's
// xclbinutil packaged from them. The design ran on a Strix Point NPU (design.md 16.39, 16.40).

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata/aiecc", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// The xclbin made by xclbinutil parses into what the NPU path needs.
func TestParseXclbinutilOutput(t *testing.T) {
	f, err := Parse(readFixture(t, "passthrough_npu2.xclbin"))
	if err != nil {
		t.Fatal(err)
	}
	k, ok := f.Kernel("MLIR_AIE")
	if !ok {
		t.Fatal("no MLIR_AIE kernel")
	}
	if k.KernelID != 0x901 || k.Functional != 0 {
		t.Errorf("kernel id %#x functional %d", k.KernelID, k.Functional)
	}
	want := []struct {
		name         string
		offset, size int
		buffer       bool
	}{
		{"opcode", 0x00, 8, false}, {"instr", 0x08, 8, true}, {"ninstr", 0x10, 4, false},
		{"bo0", 0x14, 8, true}, {"bo1", 0x1c, 8, true}, {"bo2", 0x24, 8, true}, {"bo3", 0x2c, 8, true}, {"bo4", 0x34, 8, true},
	}
	if len(k.Args) != len(want) {
		t.Fatalf("%d arguments, want %d", len(k.Args), len(want))
	}
	for i, w := range want {
		a := k.Args[i]
		if a.Index != i || a.Name != w.name || a.Offset != w.offset || a.Size != w.size || a.IsBuffer() != w.buffer {
			t.Errorf("argument %d: %+v, want %+v", i, a, w)
		}
	}
	if len(f.CUs) != 1 || f.CUs[0].Kernel != "MLIR_AIE" || f.CUs[0].KernelID != 0x901 {
		t.Errorf("CUs %+v", f.CUs)
	}
	p := f.Partition
	if p.Columns != 1 || len(p.StartColumns) != 8 || p.OpsPerCycle != 2048 {
		t.Errorf("partition: %d columns, start %v, %d ops/cycle", p.Columns, p.StartColumns, p.OpsPerCycle)
	}
	pdi, ok := f.PDIForKernel(0x901)
	if !ok || !reflect.DeepEqual(pdi, readFixture(t, "main.pdi")) {
		t.Errorf("PDI for 0x901 differs from main.pdi (found %v)", ok)
	}
}

// hwnpu-dev xclbin (SpecFromAiecc + Build) gives the same parse as xclbinutil from the same JSON.
func TestSpecFromAiecc(t *testing.T) {
	spec, err := SpecFromAiecc(readFixture(t, "kernels_main.json"), readFixture(t, "partition_main.json"), func(path string) ([]byte, error) {
		if filepath.Base(path) != "main.pdi" {
			t.Errorf("PDI path %q", path)
		}
		return readFixture(t, "main.pdi"), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := Parse(Build(spec))
	if err != nil {
		t.Fatal(err)
	}
	want, err := Parse(readFixture(t, "passthrough_npu2.xclbin"))
	if err != nil {
		t.Fatal(err)
	}
	// The container UUIDs differ (xclbinutil makes a random one); everything else must match.
	got.UUID, want.UUID = [16]byte{}, [16]byte{}
	if !reflect.DeepEqual(got.Kernels, want.Kernels) {
		t.Errorf("kernels:\n got %+v\nwant %+v", got.Kernels, want.Kernels)
	}
	if !reflect.DeepEqual(got.CUs, want.CUs) {
		t.Errorf("CUs:\n got %+v\nwant %+v", got.CUs, want.CUs)
	}
	if !reflect.DeepEqual(got.Partition, want.Partition) {
		t.Errorf("partition:\n got %+v\nwant %+v", got.Partition, want.Partition)
	}
}
