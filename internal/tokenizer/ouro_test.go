package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestOuro checks Ouro's tokenizer (SmolLM's: individual Digits, then a ByteLevel pre-tokenizer
// with GPT-2's pattern) against the ids Hugging Face gave (testdata/Ouro-1.4B-Thinking.json, from
// tools/ref/tokenizer_golden.py). It needs HWNPU_OURO_DIR with tokenizer.json; HWNPU_OURO_GOLDEN
// names another golden file (one with --random cases).
func TestOuro(t *testing.T) {
	dir := os.Getenv("HWNPU_OURO_DIR")
	if dir == "" {
		t.Skip("HWNPU_OURO_DIR not set")
	}
	tk, err := Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	golden := "testdata/Ouro-1.4B-Thinking.json"
	if g := os.Getenv("HWNPU_OURO_GOLDEN"); g != "" {
		golden = g
	}
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Cases []struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	for tok, want := range map[string]int32{"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2} {
		if id, ok := tk.ID(tok); !ok || id != want {
			t.Errorf("%s = %d %v, want %d", tok, id, ok, want)
		}
	}
	bad := 0
	for _, c := range g.Cases {
		got := tk.Encode(c.Text)
		if !slices.Equal(got, c.IDs) {
			if bad++; bad <= 20 {
				t.Errorf("Encode(%q)\n = %v\nwant %v", c.Text, got, c.IDs)
			}
		}
		if back := tk.Decode(got, false); back != c.Text {
			t.Errorf("round trip %q -> %q", c.Text, back)
		}
	}
	t.Logf("%d cases, %d wrong", len(g.Cases), bad)
}

// Individual digits come apart before GPT-2's pattern runs on what is between them.
func TestSplitGPT2Digits(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"it's 12345 I'M", []string{"it", "'s", " ", "1", "2", "3", "4", "5", " I", "'", "M"}},
		{"a 1", []string{"a", " ", "1"}},
		{"x  9y", []string{"x", "  ", "9", "y"}},
		{"v2.0", []string{"v", "2", ".", "0"}},
		{"１２", []string{"１", "２"}},
		{"Hello world", []string{"Hello", " world"}},
	} {
		if got := splitStyle(c.in, gpt2Digits); !slices.Equal(got, c.want) {
			t.Errorf("split %q = %q, want %q", c.in, got, c.want)
		}
	}
}
