package tokenizer

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestSplit(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"Hello world", []string{"Hello", " world"}},
		{"I'm here, they'LL go", []string{"I", "'m", " here", ",", " they", "'LL", " go"}},
		{"abc1234567", []string{"abc", "123", "456", "7"}},
		{"a  b", []string{"a", " ", " b"}},
		{"a\n\nb", []string{"a", "\n\n", "b"}},
		{"x  \n  y", []string{"x", "  \n", " ", " y"}},
		{"hi!!\n", []string{"hi", "!!\n"}},
		{"end   ", []string{"end", "   "}},
		{"a .b", []string{"a", " .", "b"}},
		{"日本語のテキスト。次", []string{"日本語のテキスト", "。次"}},
		{"\tx", []string{"\tx"}},
	} {
		if got := split(c.in); !slices.Equal(got, c.want) {
			t.Errorf("split(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

const tiny = `{
  "added_tokens": [{"id": 9, "content": "<|end|>", "special": true}],
  "normalizer": null,
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},
  "model": {"type": "BPE", "ignore_merges": false, "byte_fallback": false,
    "vocab": {"a": 0, "b": 1, "Ġ": 2, "ab": 3, "Ġab": 4, "abab": 5, "c": 6, "bc": 7},
    "merges": ["a b", "Ġ ab", ["ab", "ab"], "b c"]}
}`

func TestEncodeDecode(t *testing.T) {
	tk, err := parse([]byte(tiny))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		in   string
		want []int32
	}{
		{"ab", []int32{3}},
		{"abab ab", []int32{5, 4}},
		{"abc", []int32{3, 6}}, // "a b" ranks before "b c"
		{"bc<|end|>a", []int32{7, 9, 0}},
	} {
		got := tk.Encode(c.in)
		if !slices.Equal(got, c.want) {
			t.Errorf("Encode(%q) = %v, want %v", c.in, got, c.want)
		}
		if back := tk.Decode(got, false); back != c.in {
			t.Errorf("Decode(%v) = %q, want %q", got, back, c.in)
		}
	}
	if got := tk.Decode([]int32{3, 9}, true); got != "ab" {
		t.Errorf("Decode skipping specials = %q", got)
	}
}

// TestLFM3 checks the LFM3 tokenizer against the ids Hugging Face gave for the reference prompt
// (tools/ref/ref_logits.py). It needs HWNPU_LFM3_DIR with tokenizer.json.
func TestLFM3(t *testing.T) {
	dir := os.Getenv("HWNPU_LFM3_DIR")
	if dir == "" {
		t.Skip("HWNPU_LFM3_DIR not set")
	}
	tk, err := Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	text := "The Neural Engine runs small language models at low power. hwnpu is a Go library for CPU integrated NPUs:"
	want := []int32{597, 57760, 8944, 10295, 1710, 3368, 4841, 462, 2419, 1909, 22, 311, 1121, 19724, 355, 267, 4156, 9798, 374, 29609, 13418, 38284, 7335, 34}
	if got := tk.Encode(text); !slices.Equal(got, want) {
		t.Errorf("Encode = %v\nwant     %v", got, want)
	}
	for _, s := range []string{"日本語の文章も、そのまま往復する。\n\n  Tabs\tand  spaces  ", "<|im_start|>user\nこんにちは<|im_end|>\n"} {
		if back := tk.Decode(tk.Encode(s), false); back != s {
			t.Errorf("round trip %q -> %q", s, back)
		}
	}
	if id, ok := tk.ID("<|im_end|>"); !ok || id != 124900 {
		t.Errorf("<|im_end|> = %d %v", id, ok)
	}
}
