package chat

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

// Qwen35.Render writes what Qwen3.5's chat_template.jinja does. The expected prompts in
// testdata/qwen35_template.json were rendered by transformers' apply_chat_template
// (tools/ref/template_golden.py with Qwen/Qwen3.5-4B).
func TestQwen35RenderMatchesTemplate(t *testing.T) {
	raw, err := os.ReadFile("testdata/qwen35_template.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name     string    `json:"name"`
		NoThink  bool      `json:"no_think"`
		Messages []Message `json:"messages"`
		Tools    []Tool    `json:"tools"`
		Want     string    `json:"want"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			// the template has no preserve_thinking: keep_thinking cases render as without it
			got, thinking, err := Qwen35.Render(c.Messages, c.Tools, Options{NoThink: c.NoThink})
			if err != nil {
				t.Fatal(err)
			}
			if got != c.Want {
				t.Errorf("got\n%s\nwant\n%s", got, c.Want)
			}
			if want := strings.HasSuffix(got, "<think>\n"); thinking != want {
				t.Errorf("thinking %v, want %v", thinking, want)
			}
		})
	}
}

func TestQwen35Parse(t *testing.T) {
	call := "<tool_call>\n<function=edit>\n<parameter=path>\na.go\n</parameter>\n<parameter=count>\n3\n</parameter>\n</function>\n</tool_call>"
	for _, c := range []struct {
		name     string
		text     string
		thinking bool
		final    bool
		want     Answer
		args     []string
	}{
		{name: "thinking so far", text: "Let me see\n", thinking: true, want: Answer{Reasoning: "Let me see\n", Thinking: true}},
		{name: "thought and answer", text: "Hmm.\n</think>\n\nHi<|im_end|>", thinking: true, final: true, want: Answer{Reasoning: "Hmm.", Content: "Hi"}},
		{name: "without thinking", text: "Hi<|im_end|>", final: true, want: Answer{Content: "Hi"}},
		{name: "call", text: "Edit it.\n</think>\n\nEditing.\n\n" + call + "<|im_end|>", thinking: true, final: true,
			want: Answer{Reasoning: "Edit it.", Content: "Editing.", Calling: true}, args: []string{`{"path":"a.go","count":3}`}},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := Qwen35.Parse(c.text, c.thinking, nil, c.final)
			var args []string
			for _, tc := range got.ToolCalls {
				args = append(args, tc.Function.Arguments)
			}
			got.ToolCalls = nil
			if !reflect.DeepEqual(got, c.want) || !reflect.DeepEqual(args, c.args) {
				t.Errorf("got %+v %q, want %+v %q", got, args, c.want, c.args)
			}
		})
	}
	if tp, err := TemplateFor("qwen35"); err != nil || tp != Qwen35 {
		t.Errorf("TemplateFor(qwen35) = %v, %v", tp, err)
	}
}
