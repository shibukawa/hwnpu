package chat

import (
	"strconv"
	"strings"
	"testing"
)

// Qwen3-1.7B: one program below 16 GB, a program of its own to read prompts from there, by its
// name, its tag, its family and a tag of memory.
func TestBuiltinVariants(t *testing.T) {
	if err := LoadModels(nil, false, nil); err != nil {
		t.Fatal(err)
	}
	m, ok := Lookup("Qwen3-1.7B")
	if !ok || m.Run.Prefill != PrefillOff || m.Memory != 4.5 || m.Variants[16].Run.Prefill != PrefillAuto {
		t.Fatalf("as defined: %+v", m)
	}
	for _, c := range []struct {
		machine        int
		prefill        string
		memory         float64
		family, tagged string // what qwen3 and qwen3:1.7b pick
	}{
		{8, PrefillOff, 4.5, "Qwen3-1.7B", "Qwen3-1.7B"},
		{16, PrefillAuto, 7.6, "Qwen3-1.7B", "Qwen3-1.7B"},
		{64, PrefillAuto, 7.6, "Qwen3-1.7B", "Qwen3-1.7B"},
	} {
		t.Setenv(MemoryEnv, strconv.Itoa(c.machine))
		got, ok, err := lookupFor("Qwen3-1.7B", nil)
		if !ok || err != nil || got.Run.Prefill != c.prefill || got.Memory != c.memory {
			t.Errorf("%d GB, by name: %+v %v", c.machine, got.Run, got.Memory)
		}
		for _, name := range []string{"qwen3", "qwen3:1.7b"} {
			p, _, ok, err := Pick(name, c.machine)
			if !ok || err != nil || p.Name != c.family || p.Run.Prefill != c.prefill || p.Memory != c.memory {
				t.Errorf("%d GB, %s: %s %+v %v (%v)", c.machine, name, p.Name, p.Run, p.Memory, err)
			}
		}
	}
	// a tag of memory is that machine's variant, whatever this one has
	if p, _, _, _ := Pick("qwen3:16g", 8); p.Name != "Qwen3-1.7B" || p.Run.Prefill != PrefillAuto {
		t.Errorf("qwen3:16g on 8 GB: %s %+v", p.Name, p.Run)
	}
	if p, _, _, _ := Pick("qwen3:0.6b", 16); p.Name != "Qwen3-0.6B" || p.Run != (Run{}) {
		t.Errorf("qwen3:0.6b: %s %+v", p.Name, p.Run)
	}
}

func TestOn(t *testing.T) {
	m := Model{Machine: 8, Memory: 4, MemoryBy: PerVendor{"": 4}, Run: Run{Prefill: PrefillOff, Format: "lut6"},
		Variants: map[int]Variant{
			16: {Memory: PerVendor{"": 7}, Run: Run{Prefill: PrefillAuto}},
			32: {Run: Run{Format: "lut8", Context: 8192}},
		}}
	for _, c := range []struct {
		machine int
		want    Run
		memory  float64
	}{
		{4, Run{Prefill: PrefillOff, Format: "lut6"}, 4},
		{8, Run{Prefill: PrefillOff, Format: "lut6"}, 4},
		{16, Run{Prefill: PrefillAuto, Format: "lut6"}, 7},
		{24, Run{Prefill: PrefillAuto, Format: "lut6"}, 7},
		// the largest variant alone, over the model: not the 16 GB one's too
		{32, Run{Prefill: PrefillOff, Format: "lut8", Context: 8192}, 4},
	} {
		got := m.On(c.machine)
		if got.Run != c.want || got.Memory != c.memory || got.MemoryFor("apple") != c.memory {
			t.Errorf("On(%d) = %+v, %v; want %+v, %v", c.machine, got.Run, got.Memory, c.want, c.memory)
		}
	}
	if m.Run.Prefill != PrefillOff || m.Memory != 4 {
		t.Error("On changed the model")
	}
	// a variant's table of vendors changes the vendors it names
	v := Model{Machine: 8, MemoryBy: PerVendor{"amd": 3, "apple": 4}, Variants: map[int]Variant{16: {Memory: PerVendor{"apple": 6}}}}
	if got := v.On(16); got.MemoryFor("amd") != 3 || got.MemoryFor("apple") != 6 {
		t.Errorf("by vendor: %v", got.MemoryBy)
	}
}

// A file of the user's sets how a model runs, changes the variants it names and keeps the others;
// a model like another takes its variants.
func TestUserVariants(t *testing.T) {
	warn, err := userFiles(t, `version: 1
models:
  Qwen3-1.7B:
    format: lut6
    variants:
      32: {context: 8192}
  my-qwen:
    like: Qwen3-1.7B
    tag: mine
    paged: 1024
`)
	if err != nil || warn != "" {
		t.Fatal(err, warn)
	}
	m, _ := Lookup("Qwen3-1.7B")
	if m.Run != (Run{Prefill: PrefillOff, Format: "lut6"}) || len(m.Variants) != 2 || m.Variants[16].Run.Prefill != PrefillAuto {
		t.Errorf("Qwen3-1.7B: %+v %+v", m.Run, m.Variants)
	}
	if got := m.On(32).Run; got != (Run{Prefill: PrefillOff, Format: "lut6", Context: 8192}) {
		t.Errorf("on 32 GB: %+v", got)
	}
	d, _ := Lookup("my-qwen")
	if d.Run != (Run{Prefill: PrefillOff, Format: "lut6", Paged: 1024}) || len(d.Variants) != 2 || d.Memory != 4.5 {
		t.Errorf("like: %+v %+v %v", d.Run, d.Variants, d.Memory)
	}
}

func TestVariantErrors(t *testing.T) {
	chat := "version: 1\nmodels:\n  x:\n    from: hf:a/b\n    kind: chat\n    machine: 8\n"
	for _, tc := range []struct{ name, more, want string }{
		{"prefill", "    prefill: sometimes\n", "not auto, on or off"},
		{"context", "    context: -1\n", "positive"},
		{"paged", "    paged: -2\n", "or -1"},
		{"not above", "    variants:\n      8: {prefill: auto}\n", "not above machine"},
		{"nothing", "    variants:\n      16: {}\n", "gives nothing"},
		{"variant prefill", "    variants:\n      16: {prefill: never}\n", "variant of 16 GB: prefill"},
		{"unknown key", "    variants:\n      16: {memroy: 3}\n", "memroy"},
		{"not chat", "    kind: depth\n    prefill: \"off\"\n", "for chat models"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			file := chat + tc.more
			if strings.Contains(tc.more, "kind:") {
				file = strings.Replace(chat, "    kind: chat\n", "", 1) + tc.more
			}
			_, err := userFiles(t, file)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("got %v, want an error with %q", err, tc.want)
			}
		})
	}
	// written without quotes, off is the word, not YAML 1.1's false
	if _, err := userFiles(t, chat+"    prefill: off\n"); err != nil {
		t.Errorf("prefill: off: %v", err)
	} else if m, _ := Lookup("x"); m.Run.Prefill != PrefillOff {
		t.Errorf("prefill: off read as %q", m.Run.Prefill)
	}
}
