package chat

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// TestPlamo3Template checks Plamo3 against the prompts PLaMo 3 Translate's chat_template.jinja
// rendered (testdata/plamo3_template.json, from tools/ref/plamo3_template_golden.py).
func TestPlamo3Template(t *testing.T) {
	raw, err := os.ReadFile("testdata/plamo3_template.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Messages []Message   `json:"messages"`
		Kwargs   Translation `json:"kwargs"`
		Want     string      `json:"want"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for i, c := range cases {
		got, thinking, err := Plamo3.Render(c.Messages, nil, Options{Translation: &c.Kwargs})
		if err != nil {
			t.Errorf("case %d: %v", i, err)
			continue
		}
		if thinking {
			t.Errorf("case %d: thinking", i)
		}
		if got != c.Want {
			t.Errorf("case %d:\n got  %q\n want %q", i, got, c.Want)
		}
	}
}

func TestPlamo3Format(t *testing.T) {
	u := func(s string) Message { return Message{Role: "user", Content: Content{Text: s}} }
	// no target language: by the text
	p, _, err := Plamo3.Render([]Message{u("こんにちは")}, nil, Options{})
	if err != nil || !strings.HasSuffix(p, "<|plamo:val|>en<|plamo:msg|>") {
		t.Errorf("Japanese text: %q %v", p, err)
	}
	p, _, err = Plamo3.Render([]Message{u("Hello")}, nil, Options{})
	if err != nil || !strings.HasSuffix(p, "<|plamo:val|>ja<|plamo:msg|>") {
		t.Errorf("English text: %q %v", p, err)
	}
	// no target language: each pair of the history by its own text
	a := func(s string) Message { return Message{Role: "assistant", Content: Content{Text: s}} }
	p, _, err = Plamo3.Render([]Message{u("Hello"), a("こんにちは"), u("さようなら")}, nil, Options{})
	if err != nil || !strings.Contains(p, "<|plamo:val|>ja<|plamo:msg|>こんにちは<|plamo:tag|>input") || !strings.HasSuffix(p, "<|plamo:val|>en<|plamo:msg|>") {
		t.Errorf("history: %q %v", p, err)
	}
	// a system message is the context
	p, _, err = Plamo3.Render([]Message{{Role: "system", Content: Content{Text: "ref"}}, u("Hello")}, nil, Options{})
	if err != nil || !strings.Contains(p, "translation<|plamo:tag|>context<|plamo:msg|>ref<|plamo:tag|>input") {
		t.Errorf("system: %q %v", p, err)
	}
	for name, msgs := range map[string][]Message{
		"empty":          {u("  ")},
		"pfml":           {u("a <|plamo:tag|> b")},
		"two users":      {u("a"), u("b")},
		"ends assistant": {u("a"), {Role: "assistant", Content: Content{Text: "b"}}},
	} {
		if _, _, err := Plamo3.Render(msgs, nil, Options{}); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	for _, tr := range []Translation{{TargetLang: "EN"}, {TargetLang: "en-us"}, {TargetLang: "en", Form: "plain"}, {TargetLang: "ja", Form: "casual"}} {
		if _, _, err := Plamo3.Render([]Message{u("a")}, nil, Options{Translation: &tr}); err == nil {
			t.Errorf("%+v: no error", tr)
		}
	}
	for _, c := range []struct {
		text  string
		final bool
		want  string
	}{
		{"会議は月曜日です。<|plamo:tag|>", true, "会議は月曜日です。"},
		{"会議は月曜日です。<|pla", false, "会議は月曜日です。"},
		{"会議は月曜日です。<", false, "会議は月曜日です。"},
		{"a < b", false, "a < b"},
	} {
		if got := Plamo3.Parse(c.text, false, nil, c.final).Content; got != c.want {
			t.Errorf("Parse(%q) = %q, want %q", c.text, got, c.want)
		}
	}
}
