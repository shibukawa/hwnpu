package chat

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAMDSuffix(t *testing.T) {
	for args, want := range map[string]string{
		"--max-seq-len 12288 --q4":                   "-q4",
		"--max-seq-len 4096 --q4 --mlp q6 --head q6": "-q6",
		"--max-seq-len 4096":                         "",
		"--q4 --mlp q8":                              "-q8",
	} {
		if got := amdSuffix(strings.Fields(args)); got != want {
			t.Errorf("%s: %q, want %q", args, got, want)
		}
	}
}

// The built-in definitions' amd_export: an exporter's model first, and the export names the
// exports made by hand have (Qwen3.5's small models in q6, the others in q4).
func TestAMDExportRecipes(t *testing.T) {
	models := map[string]bool{"lfm3": true, "lfm3vl": true, "lfm3audio": true, "lfm3omni": true, "gemma4": true,
		"plamo3": true, "qwen35": true, "gptoss": true}
	want := map[string]string{"LFM3-1.7B": "-q4", "Qwen3.5-0.8B": "-q6", "Qwen3.5-2B": "-q6", "Qwen3.5-4B": "-q4",
		"plamo-3-translate-2b": "-q4", "gemma-4-E2B-it": "-q4", "gpt-oss-20b": ""}
	for _, m := range Catalog {
		if len(m.AMDExport) == 0 {
			continue
		}
		if !models[m.AMDExport[0]] {
			t.Errorf("%s: amd_export starts with %q, not a model of iron_export_lm.py", m.Name, m.AMDExport[0])
		}
		if w, ok := want[m.Name]; ok && !strings.HasSuffix(AMDExportDir(m), m.Name+".amd-npu2"+w) {
			t.Errorf("%s: export %s, want the suffix %s", m.Name, AMDExportDir(m), w)
		}
	}
	for name := range want {
		if m, ok := Lookup(name); !ok || len(m.AMDExport) == 0 {
			t.Errorf("%s has no amd_export", name)
		}
	}
}

func TestFindAMDExport(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	m := Model{Name: "X", AMDExport: []string{"lfm3", "--q4", "--mlp", "q6"}}
	if d := FindAMDExport(m); d != "" {
		t.Fatalf("none: %s", d)
	}
	mk := func(name string, manifest bool) string {
		d := filepath.Join(home, "models", name)
		os.MkdirAll(d, 0o755)
		if manifest {
			os.WriteFile(filepath.Join(d, "manifest.json"), []byte("{}"), 0o644)
		}
		return d
	}
	plain := mk("X.amd-npu2", true)
	if d := FindAMDExport(m); d != plain {
		t.Errorf("by hand: %s", d)
	}
	mk("X.amd-npu2-q6", false) // cut off: no manifest
	if d := FindAMDExport(m); d != plain {
		t.Errorf("cut off: %s", d)
	}
	q6 := mk("X.amd-npu2-q6", true)
	if d := FindAMDExport(m); d != q6 {
		t.Errorf("its own: %s", d)
	}
}

func TestLastLine(t *testing.T) {
	l := &lastLine{w: io.Discard}
	l.Write([]byte("one\ntw"))
	if got := l.get(); got != "one" {
		t.Errorf("%q", got)
	}
	l.Write([]byte("o\n\n"))
	if got := l.get(); got != "two" {
		t.Errorf("%q", got)
	}
	l.Write([]byte("50%\r60%\r"))
	if got := l.get(); got != "60%" {
		t.Errorf("%q", got)
	}
}
