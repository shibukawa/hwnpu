package chat

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAMDSuffix(t *testing.T) {
	for args, want := range map[string]string{
		"--max-seq-len 12288 --q4":                           "-q4",
		"--max-seq-len 4096 --q4 --mlp q6 --head q6":         "-q6",
		"--max-seq-len 4096":                                 "",
		"--q4 --mlp q8":                                      "-q8",
		"--max-seq-len 8192 --expert-places 16 --host-route": "-p16",
		"--q4 --expert-places 24":                            "-q4-p24",
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
		"plamo3": true, "qwen35": true, "qwen35moe": true, "gptoss": true, "ouro": true, "d1": true, "d1omni": true}
	want := map[string]string{"LFM3-1.7B": "-q4", "Qwen3.5-0.8B": "-q6", "Qwen3.5-2B": "-q6", "Qwen3.5-4B": "-q4",
		"plamo-3-translate-2b": "-q4", "gemma-4-E2B-it": "-q4", "gpt-oss-20b": "-p16",
		"Qwen3.6-35B-A3B": "-q4-p32"}
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
	// gpt-oss-20b: 16 places a layer on 16 GB (an export of its own), every expert from 24 GB
	if m, _ := Lookup("Qwen3.6-35B-A3B"); !strings.HasSuffix(AMDExportDir(m.On(16)), "Qwen3.6-35B-A3B.amd-npu2-q4-p32") ||
		!strings.HasSuffix(AMDExportDir(m.On(24)), "Qwen3.6-35B-A3B.amd-npu2-q4-p128") ||
		!strings.HasSuffix(AMDExportDir(m.On(64)), "Qwen3.6-35B-A3B.amd-npu2-q4") || m.Machine != 16 {
		t.Errorf("Qwen3.6-35B-A3B: machine %d, on 16 GB %s, on 24 GB %s, on 64 GB %s", m.Machine, AMDExportDir(m.On(16)),
			AMDExportDir(m.On(24)), AMDExportDir(m.On(64)))
	}
	if m, _ := Lookup("gpt-oss-20b"); !strings.HasSuffix(AMDExportDir(m.On(16)), "gpt-oss-20b.amd-npu2-p16") ||
		!strings.HasSuffix(AMDExportDir(m.On(24)), "gpt-oss-20b.amd-npu2") || m.Machine != 16 {
		t.Errorf("gpt-oss-20b: machine %d, on 16 GB %s, on 24 GB %s", m.Machine, AMDExportDir(m.On(16)), AMDExportDir(m.On(24)))
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

// TestRunIRONStaleDone makes an export again whose out.done was there before: the caller did not
// take it (made with other tools, gemma-4's int8 table), so it is not of a process that made it
// while this one waited.
func TestRunIRONStaleDone(t *testing.T) {
	t.Setenv("HWNPU_CACHE", t.TempDir())
	t.Setenv("NPU_CACHE_HOME", t.TempDir())
	out := filepath.Join(t.TempDir(), "m.amd-npu2")
	os.MkdirAll(out, 0o755)
	os.WriteFile(filepath.Join(out, "f"), []byte("old"), 0o644)
	os.WriteFile(out+".done", nil, 0o644)
	// "python" writes the export: $1 is the script, $2 the directory it writes
	python := []string{"/bin/sh", "-c", `mkdir -p "$2" && printf new > "$2/f"`, "sh"}
	if err := RunIRON(context.Background(), python, "m", "iron_export_lm.py", []string{ExportOut}, out, io.Discard); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(out, "f")); string(b) != "new" {
		t.Errorf("f: %q, the export of before kept", b)
	}
	if _, err := os.Stat(out + ".done"); err != nil {
		t.Error("no out.done after it was made")
	}
}
