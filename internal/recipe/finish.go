package recipe

import (
	"fmt"
	"maps"
	"path"
	"slices"
	"strings"
)

// Step is one thing the build does to an export once its script or conversion has made it (a
// model's finish, in order): exactly one of its fields is given.
type Step struct {
	// Tokenizer are tokenizer files of the export (paths in it: tokenizer.json,
	// text/qwen_tokenizer.json, tokenizer.jsonl), each written as its binary file beside it
	// (tokenizer.BinaryPath, which the runtime maps rather than parse the text) and removed.
	Tokenizer Strings `yaml:"tokenizer,omitempty" json:"tokenizer,omitempty"`
	// Remove are files of the export (paths in it) the runtime does not read, removed: what the
	// script writes for its own use or for a step before (tokenizer_config.json, which PLaMo's
	// tokenizer step reads and the binary file then holds). One missing is gone already.
	Remove Strings `yaml:"remove,omitempty" json:"remove,omitempty"`
}

// StepVersions are the versions of what each kind of step makes: a model's finish stands for
// its steps and their versions (Fingerprint), so that a step that comes to make other files
// asks for the exports' new versions as a changed recipe does. tokenizer: the binary tokenizer
// file's layout (tokenizer's tokbinVersion; the tokenizer package's test checks they agree).
var StepVersions = map[string]int{"tokenizer": 1, "remove": 1}

// Kind is the step's kind (its field's name in the recipe).
func (s Step) Kind() string {
	switch {
	case len(s.Tokenizer) > 0 && len(s.Remove) > 0:
		return "" // one of them
	case len(s.Tokenizer) > 0:
		return "tokenizer"
	case len(s.Remove) > 0:
		return "remove"
	}
	return ""
}

func checkFinish(at string, steps []Step) error {
	seen := map[string]bool{}
	for i, s := range steps {
		at := fmt.Sprintf("%s[%d]", at, i)
		if s.Kind() == "" {
			return fmt.Errorf("%s: no step, or more than one (one of %s)", at, strings.Join(stepKinds(), " "))
		}
		for _, f := range s.Remove {
			if f == "" || path.IsAbs(f) || path.Clean(f) != f || strings.HasPrefix(f, "../") || f == ".." {
				return fmt.Errorf("%s.remove: %q is not a path in the export", at, f)
			}
			if IsTokenizerFile(f) {
				return fmt.Errorf("%s.remove: %q is a tokenizer file: a tokenizer step makes it binary (and removes it)", at, f)
			}
		}
		for _, f := range s.Tokenizer {
			if f == "" || path.IsAbs(f) || path.Clean(f) != f || strings.HasPrefix(f, "../") || f == ".." {
				return fmt.Errorf("%s.tokenizer: %q is not a path in the export", at, f)
			}
			if !IsTokenizerFile(f) {
				return fmt.Errorf("%s.tokenizer: %q is no tokenizer file (tokenizer.json, NAME_tokenizer.json, tokenizer.jsonl)", at, f)
			}
			if seen[f] {
				return fmt.Errorf("%s.tokenizer: %q twice", at, f)
			}
			seen[f] = true
		}
	}
	return nil
}

func stepKinds() []string { return slices.Sorted(maps.Keys(StepVersions)) }

// IsTokenizerFile reports whether the name (a path in an export) is that of a tokenizer file a
// finish step makes binary: tokenizer.json, NAME_tokenizer.json (Anima's qwen_tokenizer.json),
// PLaMo's tokenizer.jsonl. The build stops when an export holds one its finish does not name.
func IsTokenizerFile(name string) bool {
	base := path.Base(name)
	return base == "tokenizer.json" || base == "tokenizer.jsonl" || strings.HasSuffix(base, "_tokenizer.json")
}

// finishVersions are the versions of the kinds of steps steps has, for a fingerprint.
func finishVersions(steps []Step) map[string]int {
	if len(steps) == 0 {
		return nil
	}
	out := map[string]int{}
	for _, s := range steps {
		out[s.Kind()] = StepVersions[s.Kind()]
	}
	return out
}

// AMDFinish is what the build does to model's export for AMD's NPU once its script has made it.
func (f *File) AMDFinish(model string) []Step {
	m, _ := f.AMDOf(model)
	return m.Finish
}

// AppleFinish is what the build does to model's Core AI exports once they are converted.
func (f *File) AppleFinish(model string) []Step {
	if m, ok := f.Models[model]; ok && m.Apple != nil {
		return m.Apple.Finish
	}
	return nil
}
