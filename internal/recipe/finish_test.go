package recipe

import (
	"strings"
	"testing"
)

// A finish's steps are of one kind each, with paths in the export; remove does not take a
// tokenizer file (its tokenizer step makes it binary and removes it).
func TestCheckFinish(t *testing.T) {
	ok := []Step{{Tokenizer: Strings{"tokenizer.jsonl"}}, {Remove: Strings{"tokenizer_config.json"}}}
	if err := checkFinish("f", ok); err != nil {
		t.Fatal(err)
	}
	if k := ok[1].Kind(); k != "remove" {
		t.Errorf("kind %q", k)
	}
	for _, c := range []struct {
		steps []Step
		want  string
	}{
		{[]Step{{Tokenizer: Strings{"tokenizer.json"}, Remove: Strings{"a.json"}}}, "more than one"},
		{[]Step{{Remove: Strings{"../a.json"}}}, "not a path"},
		{[]Step{{Remove: Strings{"text/tokenizer.json"}}}, "tokenizer file"},
	} {
		if err := checkFinish("f", c.steps); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%+v: %v", c.steps, err)
		}
	}
}
