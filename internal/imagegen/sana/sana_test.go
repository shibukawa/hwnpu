package sana

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// TestTokens checks Tokens against the ids Hugging Face's tokenizer gives in diffusers'
// encode_prompt (testdata/tokens.json): the instruction, the prompt lowered and stripped, <bos>
// first, and the padded length. It needs HWNPU_SANA_TOKENIZER, the tokenizer.json of a Sana
// repository (tokenizer/) or export (text/).
func TestTokens(t *testing.T) {
	path := os.Getenv("HWNPU_SANA_TOKENIZER")
	if path == "" {
		t.Skip("HWNPU_SANA_TOKENIZER is not set")
	}
	tok, err := tokenizer.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	bos, _ := tok.ID("<bos>")
	p := &Pipeline{tok: tok, bos: bos}
	b, err := os.ReadFile(filepath.Join("testdata", "tokens.json"))
	if err != nil {
		t.Fatal(err)
	}
	var golden []struct {
		Prompt string  `json:"prompt"`
		IDs    []int32 `json:"ids"`
		Total  int     `json:"total"`
	}
	if err := json.Unmarshal(b, &golden); err != nil {
		t.Fatal(err)
	}
	for _, g := range golden {
		ids, total := p.Tokens(g.Prompt)
		if !slices.Equal(ids, g.IDs) || total != g.Total {
			t.Errorf("%q: %d ids (total %d), want %d (total %d)\n got %v\nwant %v", g.Prompt, len(ids), total, len(g.IDs), g.Total, ids[200:], g.IDs[200:])
		}
	}
}

func TestTextStart(t *testing.T) {
	prefix := make([]int32, 208) // <bos> and the instruction
	for i := range prefix {
		prefix[i] = int32(i + 2)
	}
	prompt := append(append([]int32(nil), prefix[:207]...), 9001, 9002, 9003) // the last token merged with the prompt's
	for _, c := range []struct {
		name        string
		ids         []int32
		held, chunk int
		want        int
	}{
		{"instruction", prompt, 192, 64, 192},
		{"no held rows", prompt, 0, 64, 0},
		{"negative prompt", []int32{2, 9001, 9002}, 192, 64, 0},
		{"the instruction alone", prefix, 192, 64, 192},
		{"shorter than a chunk", prefix[:50], 192, 64, 0},
		{"a smaller chunk", prompt, 192, 32, 192},
		{"held less than shared", prompt, 128, 64, 128},
	} {
		if got := textStart(c.ids, prefix, c.held, c.chunk); got != c.want {
			t.Errorf("%s: %d, want %d", c.name, got, c.want)
		}
	}
}
