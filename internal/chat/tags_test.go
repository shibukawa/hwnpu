package chat

import (
	"github.com/goccy/go-yaml"

	"strings"
	"testing"
)

func TestPick(t *testing.T) {
	for _, c := range []struct {
		name    string
		machine int
		want    string
	}{
		{"plamo3translate", 16, "plamo-3-translate-2b"},
		{"plamo3translate", 8, "plamo-3-translate-0.8b"},
		{"plamo3translate", 4, "plamo-3-translate-0.8b"},
		{"PLaMo-3-Translate", 4, "plamo-3-translate-0.8b"}, // folded: case, '-'
		{"plamo3translate:2B", 4, "plamo-3-translate-2b"},  // a tag wins over the memory
		{"plamo3translate:0.8b", 64, "plamo-3-translate-0.8b"},
		{"translate", 16, "plamo-3-translate-2b"},
		{"translate:0.8b", 16, "plamo-3-translate-0.8b"},
		{"translation", 4, "plamo-3-translate-0.8b"},
		{"chat", 8, "LFM3-1.7B"},
		{"chat", 16, "gemma-4-E2B-it"},
		{"chat", 24, "gemma-4-E4B-it"},
		{"chat", 128, "gemma-4-E4B-it"},
		{"chat", 4, "LFM3-1.7B"}, // none fits: the smallest
		{"chat:e4b", 8, "gemma-4-E4B-it"},
		{"gemma4", 16, "gemma-4-E2B-it"}, // MODEL.md: E4B from 24 GB
		{"gemma4", 24, "gemma-4-E4B-it"},
		{"audio", 8, "LFM3-Audio-2B"},
		{"audio", 16, "LFM3-Omni-2B"},
		{"embed", 16, "plamo-embedding-1b"},
		{"qwen3.5", 8, "Qwen3.5-0.8B"},
		{"qwen3.5", 16, "Qwen3.5-2B"},
		{"qwen3.5", 24, "Qwen3.5-4B"},
		{"qwen35:4b", 16, "Qwen3.5-4B"},
		// memory tags: MODEL.md's columns, whatever this machine has
		{"plamo3translate:8g", 64, "plamo-3-translate-0.8b"},
		{"plamo3translate:16g", 8, "plamo-3-translate-2b"},
		{"plamo3translate:4g", 64, "plamo-3-translate-0.8b"},
		{"chat:16GB", 8, "gemma-4-E2B-it"},
		{"chat:24g", 8, "gemma-4-E4B-it"},
		{"gemma4:8g", 64, "gemma-4-E2B-it"},
	} {
		m, why, ok, err := Pick(c.name, c.machine)
		if err != nil || !ok || m.Name != c.want {
			t.Errorf("Pick(%q, %d) = %s (%s), %v, %v; want %s", c.name, c.machine, m.Name, why, ok, err, c.want)
		}
	}
	if _, _, ok, _ := Pick("LFM3-1.7B", 16); ok {
		t.Error("a model's name is not a family")
	}
	if _, _, ok, _ := Pick("no-such", 16); ok {
		t.Error("no-such is a family")
	}
	if _, _, ok, err := Pick("audio:2b", 16); !ok || err == nil || !strings.Contains(err.Error(), "lfm3-omni:2b") {
		t.Errorf("a tag of two models of a category: %v, %v", ok, err)
	}
	if _, _, ok, err := Pick("plamo3translate:8b", 16); !ok || err == nil || !strings.Contains(err.Error(), "0.8b") {
		t.Errorf("unknown tag: %v, %v", ok, err)
	}
}

// Every pick of a category is in the catalog, and the categories and families do not share a
// name.
func TestCategories(t *testing.T) {
	fams := map[string]bool{}
	for _, m := range Catalog {
		if m.Family == "" || m.Tag == "" || m.Memory <= 0 {
			t.Errorf("%s: family %q, tag %q, memory %.1f", m.Name, m.Family, m.Tag, m.Memory)
		}
		fams[fold(m.Family)] = true
	}
	for _, c := range Categories {
		for _, n := range append([]string{c.Name}, c.Aliases...) {
			if fams[fold(n)] {
				t.Errorf("category name %s is a family's", n)
			}
		}
		for _, p := range c.Picks {
			byName(p.Model) // panics if missing
		}
	}
}

func TestMachineMemory(t *testing.T) {
	t.Setenv(MemoryEnv, "24GB")
	if n := MachineMemory(); n != 24 {
		t.Errorf("HWNPU_MEMORY=24GB: %d", n)
	}
	t.Setenv(MemoryEnv, "")
	if n := MachineMemory(); n <= 0 {
		t.Errorf("machine memory %d", n)
	}
}

func TestRouteByFamily(t *testing.T) {
	t.Setenv(MemoryEnv, "16")
	names := []string{"LFM3-1.7B", "plamo-3-translate-2b"}
	if i, _, msg := route("translate", names, nil, KindChat, ""); i != 1 {
		t.Errorf("translate: %d %s", i, msg)
	}
	if i, _, msg := route("plamo3translate:2b", names, nil, KindChat, ""); i != 1 {
		t.Errorf("plamo3translate:2b: %d %s", i, msg)
	}
	if i, _, _ := route("plamo3translate:0.8b", names, nil, KindChat, ""); i != -1 {
		t.Errorf("a tag not loaded routed to %d", i)
	}
}

// TestImageMemoryTags checks the image families picked by a machine's memory written as a tag
// (hwnpu-image generate -m sana:8g), in any case, with or without a space, and of any size, not
// only MODEL.md's columns.
func TestImageMemoryTags(t *testing.T) {
	for name, want := range map[string]string{
		"sana:8g": "Sana-600M", "sana:8G": "Sana-600M", "sana:8GB": "Sana-600M", "sana:8 G": "Sana-600M",
		"sana:12g": "Sana-600M", "sana:4g": "Sana-600M", "sana:16G": "SANA1.5-1.6B", "sana:100g": "SANA1.5-1.6B",
		"anima": "Anima-Aesthetic", "anima:64g": "Anima-Aesthetic", "anima:base": "Anima-Base", "anima-turbo": "Anima-Turbo", "moge:20g": "moge-2-vitb", "image:8g": "Anima-Turbo", "depth:64g": "DA3MONO-LARGE", "depth:8g": "DA3-SMALL", "depth:16g": "DA3-BASE",
	} {
		m, _, ok, err := Pick(name, 59)
		if !ok || err != nil || m.Name != want {
			t.Errorf("Pick(%q) = %s, %v, %v; want %s", name, m.Name, ok, err, want)
		}
	}
}

func TestDevice(t *testing.T) {
	m, ok := Lookup("YuNet")
	if !ok {
		t.Fatal("no YuNet")
	}
	if m.DeviceFor("amd") != "cpu" || m.DeviceFor("apple") != "npu" || m.DeviceFor("intel") != "npu" {
		t.Errorf("YuNet's devices %v", m.Device)
	}
	t.Setenv("HWNPU_DEVICE", "")
	if d, why, _ := ChooseDevice(m, "amd", ""); d != "cpu" || !strings.Contains(why, "models.yaml") {
		t.Errorf("amd: %s (%s)", d, why)
	}
	if d, _, _ := ChooseDevice(m, "amd", "npu"); d != "npu" {
		t.Errorf("--device npu: %s", d)
	}
	t.Setenv("HWNPU_DEVICE", "NPU")
	if d, why, _ := ChooseDevice(m, "amd", ""); d != "npu" || why != "HWNPU_DEVICE" {
		t.Errorf("HWNPU_DEVICE: %s (%s)", d, why)
	}
	if _, _, err := ChooseDevice(m, "amd", "gpu"); err == nil {
		t.Error("--device gpu: no error")
	}
}

// TestDeviceMerge: a file over another changes the vendors it names and keeps the others.
func TestDeviceMerge(t *testing.T) {
	base := map[string]string{"amd": "cpu", "apple": "npu"}
	over := map[string]string{"amd": "npu"}
	dst := &modelEntry{Device: &base}
	merge(dst, &modelEntry{Device: &over})
	if got := *dst.Device; got["amd"] != "npu" || got["apple"] != "npu" || base["amd"] != "cpu" {
		t.Errorf("merged %v, base %v", got, base)
	}
}

func TestPerVendor(t *testing.T) {
	var e modelEntry
	if err := yaml.UnmarshalWithOptions([]byte("memory: {amd: 3.5, apple: 4.1}\nbytes: 5.9e9\n"), &e, yaml.DisallowUnknownField()); err != nil {
		t.Fatal(err)
	}
	for v, want := range map[string]float64{"amd": 3.5, "apple": 4.1, "intel": 4.1, "": 4.1} {
		if got := e.Memory.For(v); got != want {
			t.Errorf("memory for %q: %v, want %v", v, got, want)
		}
	}
	if got := e.Bytes.For("amd"); got != 5.9e9 {
		t.Errorf("bytes for amd: %v", got)
	}
	if err := yaml.Unmarshal([]byte("memory: {nvidia: 1}\n"), &e); err == nil {
		t.Error("an unknown vendor was taken")
	}
	// a table over a number changes its vendors; a number over a table replaces it
	dst := &modelEntry{Memory: &PerVendor{"": 4.1}}
	merge(dst, &modelEntry{Memory: &PerVendor{"amd": 3.5}})
	if m := *dst.Memory; m.For("amd") != 3.5 || m.For("apple") != 4.1 {
		t.Errorf("table over a number: %v", m)
	}
	merge(dst, &modelEntry{Memory: &PerVendor{"": 2}})
	if m := *dst.Memory; m.For("amd") != 2 || len(m) != 1 {
		t.Errorf("number over a table: %v", m)
	}
}
