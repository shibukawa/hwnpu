package chat

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/ovrun"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// hwnpu-chat's path on Intel's NPU with the real LFM3-1.7B, but on OpenVINO's CPU plugin:
// Prepare converts the programs as hwnpu-chat does (int8, a q8 program to generate and a q64 one
// to read prompts, a short cache here), and the engine answers a question greedily. Needs
// HWNPU_LFM3_CHAT_DIR (config.json, tokenizer.json and model.safetensors of LiquidAI/LFM3-1.7B)
// and HWNPU_OV_PYTHON; the programs take about 3.5 GB in a temporary directory, and every call
// of a program starts a Python process that reads it, so this takes minutes.
func TestIntelLFM3(t *testing.T) {
	src, py := os.Getenv("HWNPU_LFM3_CHAT_DIR"), ovcpu.Python()
	if src == "" || py == "" {
		t.Skip("HWNPU_LFM3_CHAT_DIR or HWNPU_OV_PYTHON is not set")
	}
	dir := t.TempDir()
	for _, f := range []string{"config.json", "tokenizer.json", "model.safetensors"} {
		if err := os.Symlink(filepath.Join(src, f), filepath.Join(dir, f)); err != nil {
			t.Fatal(err)
		}
	}
	ctx := context.Background()
	c, err := Prepare(ctx, Setup{Dir: dir, Repo: "LiquidAI/LFM3-1.7B", Context: 256, Draft: 8, Backend: "intel", Log: io.Discard})
	if err != nil {
		t.Fatal(err)
	}
	a, model, err := llm.LoadConfig(c.ModelConfig, "")
	if err != nil {
		t.Fatal(err)
	}
	var progs []decode.Program
	for _, p := range c.Packages {
		prog, err := ovrun.Program(py, p, ovcpu.Options{})
		if err != nil {
			t.Fatal(err)
		}
		if len(progs) > 0 {
			for name := range prog.In {
				if strings.HasPrefix(name, "kv_") || strings.HasPrefix(name, "win_") {
					prog.In[name] = progs[0].In[name]
				}
			}
		}
		progs = append(progs, prog)
	}
	sess, err := decode.New(progs[0], model.(llm.Cached))
	if err != nil {
		t.Fatal(err)
	}
	if err := sess.Add(progs[1]); err != nil {
		t.Fatal(err)
	}
	w, err := llm.OpenWeights(c.Weights, a)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	e := &Engine{sess: sess, w: w, model: model, emb: map[int32][]float32{}, Stops: map[int32]bool{}, Tok: tok, Template: LFM3, nextID: -1, drafter: LookupDraft}
	for _, s := range e.Template.Stops() {
		if id, ok := tok.ID(s); ok {
			e.Stops[id] = true
		}
	}
	text, _, err := e.Template.Render([]Message{{Role: "user", Content: Content{Text: "日本の首都はどこですか？一文で答えてください。"}}}, nil, Options{NoThink: true})
	if err != nil {
		t.Fatal(err)
	}
	var answer []int32
	st, err := e.Generate(ctx, Request{Prompt: tok.Encode(text), MaxTokens: 24}, func(gen []int32) (bool, error) {
		answer = gen
		return false, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	got := tok.Decode(answer, true)
	t.Logf("%d tokens in %d calls (%d of %d drafts accepted): %q", st.Generated, st.Calls, st.Accepted, st.Drafted, got)
	if !strings.Contains(got, "東京") {
		t.Errorf("the answer does not name Tokyo: %q", got)
	}
}
