package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// o200k's pattern against the pieces Hugging Face's tokenizers gave for the same text
// (pre_tokenizer.pre_tokenize_str of gpt-oss-20b's tokenizer.json): letter runs split where a
// lower-case letter follows upper-case ones, contractions join the run, numbers go in threes, and
// a run of other characters takes the slashes and line breaks after it.
func TestSplitO200k(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"Hello world", []string{"Hello", " world"}},
		{"HELLOworld isn't", []string{"HELLOworld", " isn't"}},
		{"I'M here, they'LL go", []string{"I'M", " here", ",", " they'LL", " go"}},
		{"abc1234567", []string{"abc", "123", "456", "7"}},
		{"日本語のテキスト。次", []string{"日本語のテキスト", "。次"}},
		{"a/b//c\n/x", []string{"a", "/b", "//", "c", "\n", "/x"}},
		{"path/to/file\n\n", []string{"path", "/to", "/file", "\n\n"}},
		{"é́x", []string{"é́x"}},
		{"́abc", []string{"́abc"}},
		{" ́x", []string{" ́x"}},
		{"ﬁ ①②③ Ａ", []string{"ﬁ", " ", "①②③", " Ａ"}},
		{"CamelCaseWORDSHere", []string{"Camel", "Case", "WORDSHere"}},
		{"x  \n  y", []string{"x", "  \n", " ", " y"}},
		{"end   ", []string{"end", "   "}},
		{"\tx", []string{"\tx"}},
		{"ǅungla Ǆ", []string{"ǅungla", " Ǆ"}},
		{"ʰa ʰA aʰB", []string{"ʰa", " ʰ", "A", " aʰ", "B"}},
	} {
		if got := splitStyle(c.in, o200k); !slices.Equal(got, c.want) {
			t.Errorf("splitStyle(%q, o200k) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestGptOss checks gpt-oss's tokenizer (o200k_harmony) against the ids Hugging Face gave
// (testdata/gpt-oss-20b.json, from tools/ref/tokenizer_golden.py). It needs HWNPU_GPTOSS_DIR with
// tokenizer.json; HWNPU_GPTOSS_GOLDEN names another golden file (one with --random cases).
func TestGptOss(t *testing.T) {
	dir := os.Getenv("HWNPU_GPTOSS_DIR")
	if dir == "" {
		t.Skip("HWNPU_GPTOSS_DIR not set")
	}
	tk, err := Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	golden := "testdata/gpt-oss-20b.json"
	if g := os.Getenv("HWNPU_GPTOSS_GOLDEN"); g != "" {
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
	for tok, want := range map[string]int32{"<|endoftext|>": 199999, "<|start|>": 200006, "<|end|>": 200007, "<|message|>": 200008, "<|channel|>": 200005, "<|return|>": 200002, "<|call|>": 200012} {
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
