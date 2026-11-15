package cache

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestParseSize(t *testing.T) {
	for s, want := range map[string]int64{"0": 0, "123": 123, "200GB": 200e9, "200 gb": 200e9, "1.5TiB": 3 << 39, "500M": 500e6, "2KiB": 2048} {
		if got, err := ParseSize(s); err != nil || got != want {
			t.Errorf("ParseSize(%q) = %d, %v; want %d", s, got, err, want)
		}
	}
	for _, s := range []string{"", "GB", "-1GB", "ten"} {
		if _, err := ParseSize(s); err == nil {
			t.Errorf("ParseSize(%q) took it", s)
		}
	}
}

func TestLimitsFromEnv(t *testing.T) {
	SetLimits(nil)
	t.Setenv(MinFreeEnv, "5%")
	t.Setenv(MaxSizeEnv, "200GB")
	l, err := CurrentLimits()
	if err != nil || l.MaxSize != 200e9 || l.MinFreePercent != 5 || l.MinFreeOf(1000e9) != 50e9 {
		t.Fatalf("%+v, %v", l, err)
	}
	t.Setenv(MaxSizeEnv, "lots")
	if _, err := CurrentLimits(); err == nil {
		t.Error("a bad size was taken")
	}
}

// fill makes a cache root with a unit of each kind, each of size bytes and used age ago (by its
// index: the first oldest).
func fill(t *testing.T) (string, []string) {
	root := t.TempDir()
	SetRoot(root)
	t.Cleanup(func() { SetRoot(""); scanned = nil })
	scanned = nil
	key := strings.Repeat("ab", 32)
	units := []string{
		"original/lab/old/model.safetensors",
		"models/Old/tables.bin",
		"entries/amd-xdna2/Old.amd-npu2/model.json",
		"iron/0123/kernel.xclbin",
		key + ".blob",
		"converted/Old/coreai/Old.coreai/v1/model.bin",
		"original/lab/new/model.safetensors",
	}
	for i, u := range units {
		p := filepath.Join(root, filepath.FromSlash(u))
		os.MkdirAll(filepath.Dir(p), 0o755)
		if err := os.WriteFile(p, make([]byte, 1000), 0o644); err != nil {
			t.Fatal(err)
		}
		if strings.HasPrefix(u, "converted/") {
			os.WriteFile(filepath.Join(filepath.Dir(p), "entry.json"), []byte("{}"), 0o644)
		}
		when := time.Now().Add(-time.Duration(len(units)-i) * 24 * time.Hour)
		if i == len(units)-1 {
			when = time.Now() // the newest: used now
		}
		unit, ok := UnitOf(p)
		if !ok {
			t.Fatalf("%s: no unit", u)
		}
		filepath.Walk(unit, func(q string, _ os.FileInfo, _ error) error { return os.Chtimes(q, when, when) })
	}
	os.MkdirAll(filepath.Join(root, "prompts"), 0o755)
	os.WriteFile(filepath.Join(root, "prompts", "p.json"), make([]byte, 50), 0o644)
	return root, units
}

func TestScan(t *testing.T) {
	root, _ := fill(t)
	// a file linked from another repository is counted once
	if runtime.GOOS != "windows" {
		if err := os.Link(filepath.Join(root, "original/lab/new/model.safetensors"), filepath.Join(root, "original/lab/new/copy.safetensors")); err != nil {
			t.Fatal(err)
		}
	}
	u, err := Scan()
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, x := range u.Units {
		got[x.Name] = x.Kind
		if x.Bytes != 1000 && !(x.Kind == KindConverted && x.Bytes == 1002) {
			t.Errorf("%s: %d bytes", x.Name, x.Bytes)
		}
	}
	want := map[string]string{
		"original/lab/old": KindOriginal, "original/lab/new": KindOriginal, "models/Old": KindConverted,
		"entries/amd-xdna2/Old.amd-npu2": KindExport, "iron/0123": KindKernels,
		"compiled/" + strings.Repeat("ab", 32): KindCompiled, "converted/Old/coreai/Old.coreai/v1": KindConverted,
	}
	for n, k := range want {
		if got[n] != k {
			t.Errorf("%s: kind %q, want %q (units %v)", n, got[n], k, got)
		}
	}
	if len(got) != len(want) || u.Other["prompts"] != 50 || u.Bytes != 7*1000+2+50 || u.Other["converted"] != 0 {
		t.Errorf("units %v, other %v, %d bytes", got, u.Other, u.Bytes)
	}
	if by := u.ByUse(); by[0].Name != "original/lab/old" || by[len(by)-1].Name != "original/lab/new" {
		t.Errorf("by use: %s ... %s", by[0].Name, by[len(by)-1].Name)
	}
}

// TestRoomLRU removes the least recently used units to make room under MaxSize, keeps the one
// used now, and says when it cannot.
func TestRoomLRU(t *testing.T) {
	root, _ := fill(t)
	SetLimits(&Limits{MaxSize: 5100})
	defer SetLimits(nil)
	var logged []string
	log := func(f string, a ...any) { logged = append(logged, f) }
	// 7052 held, 1000 more: 2952 over, so the three oldest
	if err := Room(1000, filepath.Join(root, "original/lab/x/f"), log); err != nil {
		t.Fatal(err)
	}
	for i, gone := range []bool{true, true, true, false} {
		p := []string{"original/lab/old", "models/Old", "entries/amd-xdna2/Old.amd-npu2", "iron/0123"}[i]
		if _, err := os.Stat(filepath.Join(root, p)); os.IsNotExist(err) != gone {
			t.Errorf("%s: gone %v, want %v", p, !gone, gone)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "original/lab")); err != nil {
		t.Errorf("original/lab went with original/lab/old: %v", err)
	}
	if len(logged) != 3 {
		t.Errorf("logged %d removals", len(logged))
	}
	// more than all but the newest (used now: kept) hold
	err := Room(6000, filepath.Join(root, "original/lab/x/f"), nil)
	if !errors.Is(err, ErrNoRoom) {
		t.Fatalf("err %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "original/lab/new/model.safetensors")); err != nil {
		t.Errorf("the unit used now went: %v", err)
	}
}

// TestRoomMinFree refuses a download that leaves the disk short, without a MaxSize removing
// nothing.
func TestRoomMinFree(t *testing.T) {
	root, _ := fill(t)
	free, _, err := DiskSpace(root)
	if err != nil {
		t.Skip(err)
	}
	SetLimits(&Limits{MinFree: free + 1e9})
	defer SetLimits(nil)
	if err := Room(1, filepath.Join(root, "original/lab/x/f"), nil); !errors.Is(err, ErrNoRoom) {
		t.Fatalf("err %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "original/lab/old")); err != nil {
		t.Errorf("removed without a MaxSize: %v", err)
	}
	SetLimits(&Limits{MinFree: 1})
	if err := Room(1000, filepath.Join(root, "original/lab/x/f"), nil); err != nil {
		t.Fatal(err)
	}
}

func TestPlanPrune(t *testing.T) {
	fill(t)
	u, err := Scan()
	if err != nil {
		t.Fatal(err)
	}
	old := PlanPrune(u, PruneOptions{OlderThan: 5*24*time.Hour + time.Hour})
	if len(old) != 2 || old[0].Name != "original/lab/old" || old[1].Name != "models/Old" {
		t.Errorf("older than 5 days: %v", names(old))
	}
	all := PlanPrune(u, PruneOptions{MaxSize: 1})
	if len(all) != 6 { // the one used now is kept
		t.Errorf("to 1 byte: %v", names(all))
	}
	kernels := PlanPrune(u, PruneOptions{MaxSize: 1, Kinds: []string{KindKernels}})
	if len(kernels) != 1 || kernels[0].Kind != KindKernels {
		t.Errorf("kernels: %v", names(kernels))
	}
	SetBusy(func() []string { return []string{filepath.Join(u.Root, "models/Old/tables.bin")} })
	defer SetBusy(nil)
	for _, x := range PlanPrune(u, PruneOptions{MaxSize: 1}) {
		if x.Name == "models/Old" {
			t.Error("a unit with work queued is removed")
		}
	}
}

func names(us []Unit) []string {
	var n []string
	for _, x := range us {
		n = append(n, x.Name)
	}
	return n
}
