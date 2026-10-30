package chat

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// Complete renders, generates and parses in one call on the tiny model (CPU interpreter):
// the prompt holds the messages, emit sees every token, and the stats say how it ended.
func TestComplete(t *testing.T) {
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache = 64
	short, long := cpuProgram(t, cfg, w, 1, cache), cpuProgram(t, cfg, w, 4, cache)
	for name := range long.In {
		if strings.HasPrefix(name, "kv_") || strings.HasPrefix(name, "win_") {
			long.In[name] = short.In[name]
		}
	}
	sess, err := decode.New(short, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := sess.Add(long); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, []byte(imageTokenizer), 0o644); err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	e := &Engine{sess: sess, w: w, model: cfg, emb: map[int32][]float32{}, Tok: tok, Template: LFM3, Stops: map[int32]bool{}}

	msgs := []Message{{Role: "user", Content: Content{Text: "abab"}}}
	var tools []Tool
	if err := json.Unmarshal([]byte(`[{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]`), &tools); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	n, err := e.Count(ctx, msgs, tools, Options{})
	if err != nil || n == 0 {
		t.Fatalf("Count = %d, %v", n, err)
	}
	seen := 0
	c, err := e.Complete(ctx, msgs, tools, Options{NoThink: true}, Sampling{}, 5, 5, func(Answer) bool { seen++; return false })
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(c.Rendered, "abab") || !strings.Contains(c.Rendered, `"name": "bash"`) {
		t.Errorf("prompt %q", c.Rendered)
	}
	if c.Stats.Generated != 5 || c.Finish != "length" || seen != 5 || c.Stats.Prompt != n {
		t.Errorf("generated %q, stats %+v, emit called %d times", c.Raw, c.Stats, seen)
	}
	// emit stops the generation
	c, err = e.Complete(ctx, msgs, tools, Options{NoThink: true}, Sampling{}, 5, 0, func(Answer) bool { return true })
	if err != nil || c.Stats.Generated != 1 || c.Finish != "stop" {
		t.Errorf("stopped: stats %+v, %v", c.Stats, err)
	}
}
