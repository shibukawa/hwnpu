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
