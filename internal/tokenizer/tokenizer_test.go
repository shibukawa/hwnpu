package tokenizer

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
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

// Qwen3.5's pattern against the pieces Hugging Face's tokenizers gave for the same text after its
// NFC normalizer (tokenizers 0.23, pre_tokenizer.pre_tokenize_str): marks join letters, numbers
// split digit by digit.
func TestSplitQwen35(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"Hello world", []string{"Hello", " world"}},
		{"abc1234567", []string{"abc", "1", "2", "3", "4", "5", "6", "7"}},
		{"I'm here, they'LL go", []string{"I", "'m", " here", ",", " they", "'LL", " go"}},
		{"a  b", []string{"a", " ", " b"}},
		{"x  \n  y", []string{"x", "  \n", " ", " y"}},
		{"hi!!\n", []string{"hi", "!!\n"}},
		{"日本語のテキスト。次", []string{"日本語のテキスト", "。次"}},
		{"é́x", []string{"é́x"}},
		{"́abc", []string{"́abc"}},
		{" ́x", []string{" ́x"}},
		{"नमस्ते दुनिया", []string{"नमस्ते", " दुनिया"}},
		{"a .b", []string{"a", " .", "b"}},
		{"end   ", []string{"end", "   "}},
		{"\tx", []string{"\tx"}},
		{"ﬁ ①②③ Ａ", []string{"ﬁ", " ", "①", "②", "③", " Ａ"}},
	} {
		if got := splitStyle(c.in, qwen35); !slices.Equal(got, c.want) {
			t.Errorf("splitStyle(%q, qwen35) = %q, want %q", c.in, got, c.want)
		}
	}
	// the GPT-4 pattern puts a mark with the letters after it, not before, and digits in threes
	if got := split("é\u0301x 1234"); !slices.Equal(got, []string{"é", "\u0301x", " ", "123", "4"}) {
		t.Errorf("split = %q", got)
	}
}

// A tokenizer with the NFC normalizer encodes a decomposed character as the composed one; added
// tokens are matched before normalizing.
func TestNFC(t *testing.T) {
	raw := strings.Replace(tiny, `"normalizer": null`, `"normalizer": {"type": "NFC"}`, 1)
	raw = strings.Replace(raw, `"c": 6`, `"Ã©": 6`, 1) // é in GPT-2's byte-level form
	raw = strings.Replace(raw, `"b c"]`, `"b c", "Ã ©"]`, 1)
	tk, err := parse([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if got := tk.Encode("é<|end|>"); !slices.Equal(got, []int32{6, 9}) {
		t.Errorf("Encode(e + combining acute) = %v, want [6 9]", got)
	}
	bad := strings.Replace(tiny, `"normalizer": null`, `"normalizer": {"type": "NFKC"}`, 1)
	if _, err := parse([]byte(bad)); err == nil || !strings.Contains(err.Error(), "only NFC") {
		t.Errorf("NFKC: %v", err)
	}
}

// TestQwen35 checks the Qwen3.5 tokenizer against the ids Hugging Face's tokenizers gave. It needs
// HWNPU_QWEN35_DIR with tokenizer.json (any Qwen3.5 size: they share it).
func TestQwen35(t *testing.T) {
	dir := os.Getenv("HWNPU_QWEN35_DIR")
	if dir == "" {
		t.Skip("HWNPU_QWEN35_DIR not set")
	}
	tk, err := Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		text string
		want []int32
	}{
		{"The Neural Engine runs small language models at low power. hwnpu is a Go library for CPU integrated NPUs: Apple, Intel and AMD. 日本の首都は東京です。",
			[]int32{760, 58439, 7943, 8213, 2526, 3992, 3983, 506, 3238, 2279, 13, 304, 1468, 5409, 369, 264, 5805, 6522, 364, 13540, 17708, 41814, 3440, 25, 7905, 11, 15153, 321, 23547, 13, 237829, 110161, 14876, 115197, 36298, 1710}},
		{"Café 1234 ﬁ नमस्ते <|im_start|>user\nこんにちは<|im_end|>\n<think>\n\n</think>\n\n",
			[]int32{34, 2492, 933, 220, 16, 17, 18, 19, 31422, 68337, 149518, 84237, 150104, 153348, 220, 248045, 846, 198, 85951, 248046, 198, 248068, 271, 248069, 271}},
	} {
		if got := tk.Encode(c.text); !slices.Equal(got, c.want) {
			t.Errorf("Encode(%q) = %v\nwant %v", c.text, got, c.want)
		}
	}
	if id, ok := tk.ID("<|im_end|>"); !ok || id != 248046 {
		t.Errorf("<|im_end|> = %d %v", id, ok)
	}
}
