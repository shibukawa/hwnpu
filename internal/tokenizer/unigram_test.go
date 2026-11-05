package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestUnigramT5 compares Unigram with transformers' T5TokenizerFast (testdata/t5_golden.json,
// written by it) on Anima's t5_tokenizer/tokenizer.json, which HWNPU_T5_TOKENIZER names or
// ~/models/Anima/diffusers holds; skipped without it.
func TestUnigramT5(t *testing.T) {
	path := os.Getenv("HWNPU_T5_TOKENIZER")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "Anima", "diffusers", "t5_tokenizer", "tokenizer.json")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skip("no T5 tokenizer.json:", path)
	}
	u, err := LoadUnigram(path)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/t5_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases [][2]json.RawMessage
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		var text string
		var want []int32
		if err := json.Unmarshal(c[0], &text); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(c[1], &want); err != nil {
			t.Fatal(err)
		}
		if got := u.Encode(text); !slices.Equal(got, want) {
			t.Errorf("%q:\n got  %v\n want %v", text, got, want)
		}
	}
}
