package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestWhisper checks Whisper's tokenizer (a ByteLevel pre-tokenizer with GPT-2's pattern)
// against the ids Hugging Face gave (testdata/whisper-large-v3-turbo.json, from
// tools/ref/tokenizer_golden.py). It needs HWNPU_WHISPER_DIR with tokenizer.json;
// HWNPU_WHISPER_GOLDEN names another golden file (one with --random cases).
func TestWhisper(t *testing.T) {
	dir := os.Getenv("HWNPU_WHISPER_DIR")
	if dir == "" {
		t.Skip("HWNPU_WHISPER_DIR not set")
	}
	tk, err := Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	golden := "testdata/whisper-large-v3-turbo.json"
	if g := os.Getenv("HWNPU_WHISPER_GOLDEN"); g != "" {
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
	for tok, want := range map[string]int32{"<|endoftext|>": 50257, "<|startoftranscript|>": 50258, "<|ja|>": 50266, "<|transcribe|>": 50360, "<|notimestamps|>": 50364} {
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

// GPT-2's pattern splits as transformers' ByteLevel pre-tokenizer does.
func TestSplitGPT2(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"Hello world", []string{"Hello", " world"}},
		{"it's 12345 I'M", []string{"it", "'s", " 12345", " I", "'", "M"}},
		{"a  b", []string{"a", " ", " b"}},
		{"a\n\nb ", []string{"a", "\n", "\n", "b", " "}},
		{"x !? y", []string{"x", " !?", " y"}},
		{"日本語、テスト", []string{"日本語", "、", "テスト"}},
	} {
		if got := splitStyle(c.in, gpt2); !slices.Equal(got, c.want) {
			t.Errorf("split %q = %q, want %q", c.in, got, c.want)
		}
	}
}
