package main

import (
	"slices"
	"testing"

	"github.com/alecthomas/kong"
	"github.com/shibukawa/hwnpu/internal/chat"
)

func parse(t *testing.T, args ...string) (*cli, string) {
	t.Helper()
	var c cli
	p, err := kong.New(&c, kong.Vars{"models_usage": chat.ModelsUsage, "tiers": chat.DefaultTiers,
		"embed_seqs": ints(chat.DefaultEmbedSeqs), "mtp": chat.MTPAuto})
	if err != nil {
		t.Fatal(err)
	}
	k, err := p.Parse(args)
	if err != nil {
		t.Fatalf("%v: %v", args, err)
	}
	return &c, k.Command()
}

func TestParse(t *testing.T) {
	c, cmd := parse(t)
	if cmd != "chat" || c.Chat.Prompt != "" || c.Chat.Model != "LFM3-1.7B" || c.Chat.Think != nil || !c.Chat.Sparse {
		t.Errorf("no arguments: %s %+v", cmd, c.Chat)
	}
	c, cmd = parse(t, "-m", "translate", "--no-think", "桜が満開です。")
	if cmd != "chat <prompt>" || c.Chat.Prompt != "桜が満開です。" || c.Chat.Model != "translate" || c.Chat.Think == nil || *c.Chat.Think {
		t.Errorf("a prompt: %s %+v", cmd, c.Chat)
	}
	c, _ = parse(t, "agent", "-C", ".", "--yes", "fix", "the", "bug")
	if !slices.Equal(c.Agent.Task, []string{"fix", "the", "bug"}) || !c.Agent.Yes || c.Agent.N != 2048 {
		t.Errorf("agent: %+v", c.Agent)
	}
	c, _ = parse(t, "opencode", "-m", "chat", "--", "run", "-m", "x")
	if !slices.Equal(c.OpenCode.openCodeArgs(), []string{"run", "-m", "x"}) || c.OpenCode.Model != "chat" {
		t.Errorf("opencode: %+v", c.OpenCode)
	}
	c, _ = parse(t, "serve", "-m", "LFM3-1.7B,plamo-embedding-1b", "--no-sparse")
	if c.Serve.Listen != "127.0.0.1:8080" || c.Serve.Sparse || c.Serve.Tier != chat.DefaultTiers {
		t.Errorf("serve: %+v", c.Serve)
	}
}

func TestIsPath(t *testing.T) {
	for name, want := range map[string]bool{
		"LFM3-1.7B": false, "translate": false, "LiquidAI/LFM3-1.7B": false, "chat:16g": false,
		"~/models/LFM3-1.7B.amd-npu2": true, "./x": true, "/abs/x": true, "a/b/c": true,
	} {
		if isPath(name) != want {
			t.Errorf("isPath(%q) = %v", name, !want)
		}
	}
}
