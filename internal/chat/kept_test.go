package chat

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The memory a user chose to run a model as is kept in bonsaiai's directory of settings, by model,
// and read by every command that resolves the model: a 16 GB machine that keeps 8 for
// LFM3-1.7B runs it as models.yaml defines it for 8 GB (no whole program of short
// conversations) and says so; a machine given (--memory) is over what is kept; 0 forgets it; a
// file that is not one of bonsaiai's keeps nothing.
func TestChosenMemory(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv(MemoryEnv, "16")
	lfm, ok, err := lookupFor("LFM3-1.7B", nil)
	if !ok || err != nil || lfm.Run.Tier != "auto" || ChosenMemory(lfm) != 0 {
		t.Fatalf("on 16 GB: %+v, %v, %v", lfm.Run, ok, err)
	}
	if err := ChooseMemory(lfm, 8); err != nil {
		t.Fatal(err)
	}
	if gb := ChosenMemory(lfm); gb != 8 {
		t.Errorf("chosen %d", gb)
	}
	var log bytes.Buffer
	if m, _, _ := lookupFor("LFM3-1.7B", &log); m.Run.Tier != "none" || !strings.Contains(log.String(), "runs as on a machine of 8 GB, as chosen") {
		t.Errorf("chosen 8: %+v, log %q", m.Run, log.String())
	}
	if m, _, _ := lookupOn("LFM3-1.7B", 16, nil); m.Run.Tier != "auto" {
		t.Errorf("--memory 16 over the chosen: %+v", m.Run)
	}
	// another model: its own
	qwen, _, _ := lookupFor("Qwen3-1.7B", nil)
	if qwen.Run.Prefill == PrefillOff || ChosenMemory(qwen) != 0 {
		t.Errorf("Qwen3-1.7B on 16 GB: %+v", qwen.Run)
	}
	if err := ChooseMemory(qwen, 8); err != nil {
		t.Fatal(err)
	}
	if got := ChosenMemories(); len(got) != 2 || got["LFM3-1.7B"] != 8 || got["Qwen3-1.7B"] != 8 {
		t.Errorf("chosen: %v", got)
	}
	if m, _, _ := lookupFor("Qwen3-1.7B", nil); m.Run.Prefill != PrefillOff {
		t.Errorf("Qwen3-1.7B chosen 8: %+v", m.Run)
	}
	// forgotten: as on this machine; the file goes with the last
	if err := ChooseMemory(lfm, 0); err != nil || ChosenMemory(lfm) != 0 || ChosenMemory(qwen) != 8 {
		t.Errorf("forget LFM3-1.7B: %v, %v", ChosenMemories(), err)
	}
	if err := ChooseMemory(qwen, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(RunFile()); !os.IsNotExist(err) {
		t.Errorf("%s with nothing chosen: %v", RunFile(), err)
	}
	if err := ChooseMemory(qwen, 0); err != nil {
		t.Errorf("forget with nothing chosen: %v", err)
	}
	if err := ChooseMemory(qwen, -8); err == nil {
		t.Error("-8 GB chosen")
	}
	if err := os.MkdirAll(filepath.Dir(RunFile()), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{`{"models": {"LFM3-1.7B": {"memory": -8}}}`, `{"models": {"LFM3-1.7B": {"memory": "8"}}}`, `not json`} {
		if err := os.WriteFile(RunFile(), []byte(bad), 0o644); err != nil {
			t.Fatal(err)
		}
		if gb := ChosenMemory(lfm); gb != 0 {
			t.Errorf("%s: chosen %d", bad, gb)
		}
	}
}

// go test never reads the machine's settings.
func TestRunFileInTests(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", "")
	if f := RunFile(); f != "" {
		t.Errorf("RunFile in a test: %q", f)
	}
}
