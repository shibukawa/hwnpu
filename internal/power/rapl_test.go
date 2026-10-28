package power

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func write(t *testing.T, path, v string) {
	t.Helper()
	os.MkdirAll(filepath.Dir(path), 0o755)
	if err := os.WriteFile(path, []byte(v+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestRAPL(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "intel-rapl:0:0", "name"), "core")
	write(t, filepath.Join(root, "intel-rapl:0:0", "energy_uj"), "1")
	pkg := filepath.Join(root, "intel-rapl:0")
	write(t, filepath.Join(pkg, "name"), "package-0")
	write(t, filepath.Join(pkg, "max_energy_range_uj"), "10000000") // 10 J
	write(t, filepath.Join(pkg, "energy_uj"), "9000000")

	r, err := OpenRAPL(root)
	if err != nil {
		t.Fatal(err)
	}
	a, err := r.Sample()
	if err != nil || a.EnergyJ != 9 || a.Scope != "package" || a.Source != "rapl:intel-rapl:0" {
		t.Fatalf("%+v %v", a, err)
	}
	write(t, filepath.Join(pkg, "energy_uj"), "500000") // wrapped: 9 J -> 10.5 J
	b, _ := r.Sample()
	if math.Abs(b.EnergyJ-10.5) > 1e-9 {
		t.Fatalf("wrap not handled: %v", b.EnergyJ)
	}
	if _, err := OpenRAPL(t.TempDir()); err == nil {
		t.Fatal("expected error without a package domain")
	}
}
