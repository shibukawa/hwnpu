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
