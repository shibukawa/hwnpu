package finish

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/recipe"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

const tiny = `{
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}
}`

func TestRun(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "text"), 0o755)
	for _, f := range []string{"tokenizer.json", "text/qwen_tokenizer.json"} {
		os.WriteFile(filepath.Join(dir, f), []byte(tiny), 0o644)
	}
	os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{}`), 0o644)

	// one left out stops the build
	err := Run(dir, []recipe.Step{{Tokenizer: []string{"tokenizer.json"}}})
	if err == nil || !strings.Contains(err.Error(), "text/qwen_tokenizer.json") {
		t.Fatalf("a tokenizer left out: %v", err)
	}
	steps := []recipe.Step{{Tokenizer: []string{"tokenizer.json", "text/qwen_tokenizer.json"}}}
	if err := Run(dir, steps); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"tokenizer.json", "text/qwen_tokenizer.json"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err == nil {
			t.Errorf("%s is still there", f)
		}
		tk, err := tokenizer.Load(filepath.Join(dir, f))
		if err != nil || len(tk.Encode("ab")) != 1 {
			t.Errorf("%s: its binary file: %v", f, err)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer_config.json")); err != nil {
		t.Error("tokenizer_config.json went")
	}
	// done again on a finished export: nothing to do
	if err := Run(dir, steps); err != nil {
		t.Errorf("again: %v", err)
	}
	// a file the export does not have
	if err := Run(t.TempDir(), steps); err == nil {
		t.Error("a missing tokenizer file passed")
	}
}

// The recipe's version of the tokenizer step is the binary file's: a new layout raises the
// exports' versions (their fingerprints have it).
func TestTokenizerStepVersion(t *testing.T) {
	if recipe.StepVersions["tokenizer"] != tokenizer.BinaryVersion {
		t.Errorf("recipe.StepVersions tokenizer %d, the binary file's version %d", recipe.StepVersions["tokenizer"], tokenizer.BinaryVersion)
	}
}

// Check refuses a tokenizer's text, naming what makes it binary; a finished export passes.
func TestCheck(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "text", "check"), 0o755)
	os.WriteFile(filepath.Join(dir, "text", "manifest.json"), []byte(`{}`), 0o644)
	os.WriteFile(filepath.Join(dir, "text", "check", "words.json"), []byte(`{}`), 0o644)
	os.WriteFile(filepath.Join(dir, "text", "tokenizer.json"), []byte(tiny), 0o644)
	err := Check(dir, "--finish")
	if err == nil || !strings.Contains(err.Error(), "text/tokenizer.json") || !strings.Contains(err.Error(), "--finish") {
		t.Fatalf("a tokenizer's text: %v", err)
	}
	if err := Run(dir, []recipe.Step{{Tokenizer: []string{"text/tokenizer.json"}}}); err != nil {
		t.Fatal(err)
	}
	if err := Check(dir, ""); err != nil { // check/ is left out of the entry, not refused
		t.Errorf("a finished export: %v", err)
	}
	// no finish for it: the message says what the recipe needs
	other := t.TempDir()
	os.WriteFile(filepath.Join(other, "tokenizer.json"), []byte(tiny), 0o644)
	if err := Check(other, ""); err == nil || !strings.Contains(err.Error(), "finish: [{tokenizer: [tokenizer.json]}]") {
		t.Errorf("no finish: %v", err)
	}
}

// A remove step removes files the runtime does not read, after a tokenizer step before it that
// may read them (PLaMo's tokenizer_config.json), and passes when they are gone already.
func TestRemove(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tiny), 0o644)
	os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{}`), 0o644)
	steps := []recipe.Step{{Tokenizer: []string{"tokenizer.json"}}, {Remove: []string{"tokenizer_config.json"}}}
	if err := Run(dir, steps); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer_config.json")); err == nil {
		t.Error("tokenizer_config.json is still there")
	}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.tok")); err != nil {
		t.Error("no tokenizer.tok")
	}
	if err := Run(dir, steps); err != nil {
		t.Errorf("again: %v", err)
	}
}
