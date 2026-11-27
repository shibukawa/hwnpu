package main

import (
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/systemone"
)

// The memory to run a model as: a tag of memory over --memory, none (0) without either (hwnpud
// keeps it: internal/aiengine's TestKeepMemory).
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
}

// The defaults the command copies, so as to import no engine, are the engine's.
func TestDefaultsOfTheEngine(t *testing.T) {
	if mtpAuto != chat.MTPAuto || !slices.Equal(defaultEmbedSeqs, chat.DefaultEmbedSeqs) || !slices.Equal(defaultSystemSeqs, systemone.DefaultSeqs) {
		t.Errorf("mtp %q, embed seqs %v, systemone seqs %v: not internal/chat's and internal/systemone's", mtpAuto, defaultEmbedSeqs, defaultSystemSeqs)
	}
	if backendEnv != llm.BackendEnv || backendCoreML != llm.BackendCoreML || backendCoreAI != llm.BackendCoreAI {
		t.Errorf("backends %q %q %q: not internal/llm's", backendEnv, backendCoreML, backendCoreAI)
	}
}
