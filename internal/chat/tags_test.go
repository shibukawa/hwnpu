package chat

import (
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
		"sana:8g": "Sana-Sprint-0.6B", "sana:8G": "Sana-Sprint-0.6B", "sana:8GB": "Sana-Sprint-0.6B", "sana:8 G": "Sana-Sprint-0.6B",
		"sana:12g": "Sana-Sprint-0.6B", "sana:4g": "Sana-Sprint-0.6B", "sana:16G": "Sana-Sprint-1.6B", "sana:100g": "Sana-Sprint-1.6B",
		"sana-full:8g": "Sana-600M", "moge:20g": "moge-2-vitb", "image:8g": "Anima-Turbo", "depth:64g": "DA3MONO-LARGE",
	} {
		m, _, ok, err := Pick(name, 59)
		if !ok || err != nil || m.Name != want {
			t.Errorf("Pick(%q) = %s, %v, %v; want %s", name, m.Name, ok, err, want)
		}
	}
}
