package main

import (
	"bytes"
	"testing"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// The memory to run a model as: a tag of memory over --memory, none (0) without either; kept
// once the model is ready, and said so once.
func TestMemoryGiven(t *testing.T) {
	for _, c := range []struct {
		name   string
		memory int
		want   int
	}{
		{"LFM3-1.7B", 0, 0},
		{"LFM3-1.7B", 8, 8},
		{"lfm3:8g", 0, 8},
		{"lfm3:16gb", 8, 16},
		{"qwen3.5:2b", 8, 8},
		{"LFM3-1.7B", -1, 0},
	} {
		if got := memoryGiven(c.name, &convertFlags{Memory: c.memory}); got != c.want {
			t.Errorf("%s --memory %d: %d, want %d", c.name, c.memory, got, c.want)
		}
	}
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	m, ok := chat.Lookup("LFM3-1.7B")
	if !ok {
		t.Fatal("no LFM3-1.7B")
	}
	var log bytes.Buffer
	keepMemory(m, 0, &log)
	if chat.ChosenMemory(m) != 0 || log.Len() != 0 {
		t.Errorf("nothing given: chosen %d, log %q", chat.ChosenMemory(m), log.String())
	}
	keepMemory(m, 8, &log)
	if chat.ChosenMemory(m) != 8 || !bytes.Contains(log.Bytes(), []byte("runs as on a machine of 8 GB from now on")) {
		t.Errorf("8 GB: chosen %d, log %q", chat.ChosenMemory(m), log.String())
	}
	log.Reset()
	if keepMemory(m, 8, &log); log.Len() != 0 {
		t.Errorf("kept again: %q", log.String())
	}
}
