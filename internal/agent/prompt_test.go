package agent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The system prompt stays short (about 400 tokens with the four tools), lists only the rules
// of the tools offered, and cuts long project instructions.
func TestPrompt(t *testing.T) {
	dir := t.TempDir()
	for _, f := range []string{"main.go", "go.mod", ".hidden", "pkg/x.go"} {
		if err := os.MkdirAll(filepath.Dir(filepath.Join(dir, f)), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, f), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	s, err := New(&script{}, Config{Dir: dir})
	if err != nil {
		t.Fatal(err)
	}
	p := s.System()
	if len(p) > 1500 {
		t.Errorf("the system prompt is %d characters:\n%s", len(p), p)
	}
	if strings.Contains(p, dir) {
		t.Errorf("the prompt names the directory:\n%s", p)
	}
	for _, want := range []string{"Files in the current directory: go.mod, main.go, pkg/\n", "relative to the current directory", "Read a file before", "Use edit", "Use write", "Use bash", "stop calling tools"} {
		if !strings.Contains(p, want) {
			t.Errorf("the prompt lacks %q:\n%s", want, p)
		}
	}
	if strings.Contains(p, "Project instructions") {
		t.Error("project instructions without AGENTS.md")
	}
	if len(s.Tools()) != 4 {
		t.Errorf("%d tools", len(s.Tools()))
	}

	s, err = New(&script{}, Config{Dir: "/work", Tools: []string{"bash", "read"}, Instructions: strings.Repeat("x", 5000)})
	if err != nil {
		t.Fatal(err)
	}
	p = s.System()
	if strings.Contains(p, "Use edit") || strings.Contains(p, "Use write") || !strings.Contains(p, "Use bash") {
		t.Errorf("rules with read and bash only:\n%s", p)
	}
	if !strings.Contains(p, "Project instructions:\n"+strings.Repeat("x", MaxInstructions)+"\n[...]") || strings.Contains(p, strings.Repeat("x", MaxInstructions+1)) {
		t.Errorf("instructions not cut at %d:\n%s", MaxInstructions, p)
	}
	if tools := s.Tools(); len(tools) != 2 || tools[0].Name() != "read" || tools[1].Name() != "bash" {
		t.Errorf("tools %v", tools)
	}

	s, err = New(&script{}, Config{Dir: "/work", System: "You fix Go code."})
	if err != nil {
		t.Fatal(err)
	}
	if p = s.System(); !strings.HasPrefix(p, "You fix Go code.\n\nRules:") {
		t.Errorf("custom opening:\n%s", p)
	}
}
