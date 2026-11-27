package aiengine

import (
	"bytes"
	"testing"

	"github.com/shibukawa/hwnpu/internal/aieng"
	"github.com/shibukawa/hwnpu/internal/chat"
)

// The memory given to run a model as is kept once it is ready (hwnpu-chat --memory, a tag of
// memory), and said so once; none (0) keeps nothing.
func TestKeepMemory(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	m, ok := chat.Lookup("LFM3-1.7B")
	if !ok {
		t.Fatal("no LFM3-1.7B")
	}
	var log bytes.Buffer
	aieng.Chat.KeepMemory(m.Name, 0, &log)
	if chat.ChosenMemory(m) != 0 || log.Len() != 0 {
		t.Errorf("nothing given: chosen %d, log %q", chat.ChosenMemory(m), log.String())
	}
	aieng.Chat.KeepMemory(m.Name, 8, &log)
	if chat.ChosenMemory(m) != 8 || !bytes.Contains(log.Bytes(), []byte("runs as on a machine of 8 GB from now on")) {
		t.Errorf("8 GB: chosen %d, log %q", chat.ChosenMemory(m), log.String())
	}
	log.Reset()
	if aieng.Chat.KeepMemory(m.Name, 8, &log); log.Len() != 0 {
		t.Errorf("kept again: %q", log.String())
	}
}

// The list of hwnpu-chat list has the models and what the features mean; that of a kind its
// models alone.
func TestChatList(t *testing.T) {
	var full, kind bytes.Buffer
	aieng.Chat.List(t.Context(), &full, nil, true, "coreai")
	aieng.Chat.List(t.Context(), &kind, []string{"embedding"}, false, "")
	if !bytes.Contains(full.Bytes(), []byte("LFM3-1.7B")) || !bytes.Contains(full.Bytes(), []byte("\nFeatures:")) {
		t.Errorf("hwnpu-chat list:\n%s", full.String())
	}
	if bytes.Contains(kind.Bytes(), []byte("\nFeatures:")) || bytes.Contains(kind.Bytes(), []byte("LFM3-1.7B  ")) {
		t.Errorf("the embedding models:\n%s", kind.String())
	}
}
