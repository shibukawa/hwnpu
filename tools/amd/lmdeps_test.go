package amdtools

import (
	"regexp"
	"slices"
	"testing"
	"testing/fstest"
)

// TestLMArchCoversModels checks lmArch has every model of iron_export_lm.py's MODELS, with the
// module of its entry when that is one of tools/amd, and that every module of tools/amd
// iron_export_lm.py imports is a root of some model, flag or of all.
func TestLMArchCoversModels(t *testing.T) {
	b, err := files.ReadFile("iron_export_lm.py")
	if err != nil {
		t.Fatal(err)
	}
	entries := regexp.MustCompile(`(?m)^\s+"(\w+)": \("([\w.]+)", "\w+"\),`).FindAllStringSubmatch(string(b), -1)
	if len(entries) < 10 {
		t.Fatalf("MODELS: %d entries read", len(entries))
	}
	for _, e := range entries {
		roots, ok := lmArch[e[1]]
		if !ok {
			t.Errorf("lmArch has no %s", e[1])
			continue
		}
		if _, err := files.ReadFile(e[2] + ".py"); err == nil && !slices.Contains(roots, e[2]+".py") {
			t.Errorf("lmArch[%s] lacks %s.py, its MODELS module", e[1], e[2])
		}
	}
	known := slices.Clone(lmCommon)
	for _, r := range lmArch {
		known = append(known, r...)
	}
	for _, r := range lmFlags {
		known = append(known, r...)
	}
	for _, m := range importRe.FindAllStringSubmatch(string(b), -1) {
		p := m[1] + m[2] + ".py"
		if _, err := files.ReadFile(p); err == nil && !slices.Contains(known, p) {
			t.Errorf("iron_export_lm.py imports %s, which no model, flag or lmCommon names", p)
		}
	}
}

// TestLMClosure follows the imports of the roots (not those of iron_export_lm.py), and takes every
// kernel.
func TestLMClosure(t *testing.T) {
	fsys := fstest.MapFS{
		"iron_export_lm.py": {Data: []byte("import iron_a\nimport iron_b\n")},
		"iron_a.py":         {Data: []byte("def f():\n    from iron_c import x\n    import iron_export_lm\n")},
		"iron_b.py":         {Data: []byte("import numpy\n")},
		"iron_c.py":         {Data: []byte("")},
		"kernels/k.cc":      {Data: []byte("")},
	}
	got := lmClosure(fsys, []string{"iron_export_lm.py", "iron_a.py"})
	if want := []string{"iron_a.py", "iron_c.py", "iron_export_lm.py", "kernels/k.cc"}; !slices.Equal(got, want) {
		t.Errorf("%v, want %v", got, want)
	}
}

// TestLMFiles: a Gemma 4 export reads iron_hostple.py, an LFM3 one does not read Gemma 4's, an
// image model's exporter is no one's, --verify adds iron_verify.py, an unknown model is nil.
func TestLMFiles(t *testing.T) {
	g := LMFiles([]string{"gemma4", "--max-seq-len", "4096", "--q4"})
	l := LMFiles([]string{"lfm3", "--q4"})
	if !slices.Contains(g, "iron_hostple.py") || slices.Contains(l, "iron_gemma4.py") || slices.Contains(l, "iron_sana.py") {
		t.Errorf("gemma4 %v\nlfm3 %v", g, l)
	}
	if q := LMFiles([]string{"qwen35", "--verify", "3"}); !slices.Contains(q, "iron_verify.py") {
		t.Errorf("qwen35 --verify: %v", q)
	}
	if LMFiles([]string{"nothing"}) != nil {
		t.Error("an unknown model")
	}
	a, _ := LMHash([]string{"lfm3", "--q4"})
	h, _ := Hash()
	if a == h || a == "" {
		t.Error("lfm3's hash is all of them")
	}
}
