package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// tinyFallback is a tokenizer.json with byte fallback in Gemma's form, with the 256 byte tokens
// at ids 10 to 265.
func tinyFallback() string {
	vocab := map[string]int{"<pad>": 0, "<eos>": 1, "a": 2, "b": 3, "▁": 4, "ab": 5, "▁ab": 6, "abab": 7, "c": 8, "bc": 9}
	for b := 0; b < 256; b++ {
		vocab[fmt.Sprintf("<0x%02X>", b)] = 10 + b
	}
	vj, _ := json.Marshal(vocab)
	return `{
  "added_tokens": [{"id": 0, "content": "<pad>", "special": true}, {"id": 1, "content": "<eos>", "special": true}],
  "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
  "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
  "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}]},
  "model": {"type": "BPE", "byte_fallback": true, "ignore_merges": false, "vocab": ` + string(vj) + `,
    "merges": [["a", "b"], ["▁", "ab"], ["ab", "ab"], ["b", "c"]]}
}`
}

func TestFallback(t *testing.T) {
	tk, err := parse([]byte(tinyFallback()))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		in   string
		want []int32
	}{
		{"ab", []int32{5}},
		{"abab ab", []int32{7, 6}}, // one piece: "abab▁ab"
		{"abc", []int32{5, 8}},     // "a b" ranks before "b c"
		{"bc<eos>a", []int32{9, 1, 2}},
		{" b", []int32{4, 3}},                    // no ▁ prepended
		{"aé", []int32{2, 10 + 0xC3, 10 + 0xA9}}, // é is not in the vocabulary: its bytes
		{"", nil},
	} {
		got := tk.Encode(c.in)
		if !slices.Equal(got, c.want) {
			t.Errorf("Encode(%q) = %v, want %v", c.in, got, c.want)
		}
		if back := tk.Decode(got, false); back != c.in {
			t.Errorf("Decode(Encode(%q)) = %q", c.in, back)
		}
	}
	if got := tk.Decode([]int32{2, 1, 10 + 0xE3}, true); got != "a�" {
		t.Errorf("Decode skipping specials, with a lone byte = %q", got)
	}
	for _, bad := range []struct{ old, new string }{
		{`"content": "▁"}`, `"content": "_"}`},
		{`{"type": "Fuse"}`, `{"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}`},
		{`"\u003c0xFF\u003e"`, `"\u003c0xFG\u003e"`},
		{`"type": "Split"`, `"type": "Metaspace"`},
	} {
		raw := strings.Replace(tinyFallback(), bad.old, bad.new, 1)
		if raw == tinyFallback() {
			t.Fatalf("%s is not in the tokenizer", bad.old)
		}
		if _, err := parse([]byte(raw)); err == nil {
			t.Errorf("%s: no error", bad.new)
		}
	}
}

// TestGemma4 checks Gemma 4's tokenizer against the ids Hugging Face gave
// (testdata/gemma-4-E2B-it.json, from tools/ref/tokenizer_golden.py). It needs HWNPU_GEMMA4_DIR
// with tokenizer.json; HWNPU_GEMMA4_GOLDEN may name another golden file (one with --random).
func TestGemma4(t *testing.T) {
	dir := os.Getenv("HWNPU_GEMMA4_DIR")
	if dir == "" {
		t.Skip("HWNPU_GEMMA4_DIR not set")
	}
	tk, err := Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	golden := "testdata/gemma-4-E2B-it.json"
	if g := os.Getenv("HWNPU_GEMMA4_GOLDEN"); g != "" {
		golden = g
	}
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		BOS   int32 `json:"bos"`
		Cases []struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if id, ok := tk.ID("<bos>"); !ok || id != g.BOS {
		t.Errorf("<bos> = %d %v, want %d", id, ok, g.BOS)
	}
	for _, c := range g.Cases {
		got := tk.Encode(c.Text)
		if !slices.Equal(got, c.IDs) {
			t.Errorf("Encode(%q)\n = %v\nwant %v", c.Text, got, c.IDs)
		}
		if back := tk.Decode(got, false); back != c.Text && !strings.Contains(c.Text, "▁") {
			t.Errorf("round trip %q -> %q", c.Text, back)
		}
	}
	t.Logf("%d cases", len(g.Cases))
}
