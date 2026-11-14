package chat

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// A token is kept for its kana, as one or two kanji, or as ASCII symbols and spaces; longer kanji
// words, Hangul and Latin text are left to the low IDs.
func TestDraftToken(t *testing.T) {
	for s, want := range map[string]bool{"の": true, "カタ": true, "漢": true, "東京": true, "東京都": false, "다": false, "the": false, "": false, "すごい!": true, "```": true, "**:": true, " \n": true, "a1": false} {
		if got := draftToken(s); got != want {
			t.Errorf("draftToken(%q) = %v, want %v", s, got, want)
		}
	}
}

// The drafter turns the largest logit's index into its token through the rows of its vocabulary,
// or keeps it as the token without them; the rows written into an export are read back.
func TestDraftVocabRoundTrip(t *testing.T) {
	dir := t.TempDir()
	ids := []int32{0, 1, 2, 40000, 40002, 40100}
	if err := writeDraftVocab(dir, ids); err != nil {
		t.Fatal(err)
	}
	got, err := readDraftVocab(filepath.Join(dir, "model.parts.json"))
	if err != nil || !slices.Equal(got, ids) {
		t.Fatalf("read %v (%v), want %v", got, err, ids)
	}
	if none, err := readDraftVocab(filepath.Join(t.TempDir(), "model.aimodel")); none != nil || err != nil {
		t.Errorf("an export without rows: %v (%v)", none, err)
	}
	m := &mtpDrafter{ids: ids}
	if tok := m.token([]float32{0, 1, 0, 0, 5, 2}); tok != 40002 {
		t.Errorf("token %d, want 40002", tok)
	}
	m.ids = nil
	if tok := m.token([]float32{0, 1, 0, 0, 5, 2}); tok != 4 {
		t.Errorf("token %d without rows, want 4", tok)
	}
}

// Qwen3.5's vocabulary (when its tokenizer is downloaded): the rows are increasing, a multiple of
// 16, the low IDs and Japanese tokens among them, and cover most of a Japanese sentence.
func TestDraftVocabQwen35(t *testing.T) {
	path := filepath.Join(ModelsDir(), "Qwen3.5-4B", "tokenizer.json")
	if _, err := os.Stat(path); err != nil {
		t.Skip("Qwen3.5-4B's tokenizer is not downloaded")
	}
	tok, err := tokenizer.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	ids := draftVocab(tok)
	if len(ids) == 0 || len(ids)%16 != 0 || !slices.IsSorted(ids) || ids[draftVocabLow-1] != draftVocabLow-1 {
		t.Fatalf("%d rows", len(ids))
	}
	text := tok.Encode("日本の首都は東京です。ひらがなとカタカナと漢字で書いた文を下書きする。")
	in := 0
	for _, id := range text {
		if _, ok := slices.BinarySearch(ids, id); ok {
			in++
		}
	}
	if in*10 < len(text)*8 {
		t.Errorf("%d of %d tokens of a Japanese sentence in the %d rows", in, len(text), len(ids))
	}
	t.Logf("%d rows of %d; %d of %d tokens of the sentence", len(ids), tok.VocabSize(), in, len(text))
}
