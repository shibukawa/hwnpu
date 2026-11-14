package chat

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
	"github.com/shibukawa/hwnpu/internal/llm/qwen3"
)

// Qwen3.Render writes what Qwen3's chat template does, and "<think>\n" after it when thinking,
// which the model writes first. The expected prompts in testdata/qwen3_template.json were rendered
// by transformers' apply_chat_template (tools/ref/template_golden.py --text-content with
// Qwen/Qwen3-0.6B).
func TestQwen3RenderMatchesTemplate(t *testing.T) {
	raw, err := os.ReadFile("testdata/qwen3_template.json")
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
			got, thinking, err := Qwen3.Render(c.Messages, c.Tools, Options{NoThink: c.NoThink})
			if err != nil {
				t.Fatal(err)
			}
			want := c.Want
			if !c.NoThink {
				want += "<think>\n"
			}
			if got != want {
				t.Errorf("got\n%s\nwant\n%s", got, want)
			}
			if thinking == c.NoThink {
				t.Errorf("thinking %v with no_think %v", thinking, c.NoThink)
			}
		})
	}
}

func TestQwen3Parse(t *testing.T) {
	call := "<tool_call>\n{\"name\": \"edit\", \"arguments\": {\"path\": \"a.go\", \"count\": 3}}\n</tool_call>"
	for _, c := range []struct {
		name     string
		text     string
		thinking bool
		final    bool
		want     Answer
		names    []string
		args     []string
	}{
		{name: "thinking so far", text: "Let me see\n", thinking: true, want: Answer{Reasoning: "Let me see\n", Thinking: true}},
		{name: "thought and answer", text: "Hmm.\n</think>\n\nHi<|im_end|>", thinking: true, final: true, want: Answer{Reasoning: "Hmm.", Content: "Hi"}},
		{name: "without thinking", text: "Hi<|im_end|>", final: true, want: Answer{Content: "Hi"}},
		{name: "call", text: "Edit it.\n</think>\n\nEditing.\n" + call + "<|im_end|>", thinking: true, final: true,
			want: Answer{Reasoning: "Edit it.", Content: "Editing.", Calling: true}, names: []string{"edit"}, args: []string{`{"path":"a.go","count":3}`}},
		{name: "two calls, one broken, arguments as a string", final: true,
			text: call + "\n<tool_call>\n{\"name\": \"edit\", \"arguments\": \n</tool_call>\n<tool_call>\n{\"name\": \"run\", \"arguments\": \"{\\\"x\\\": 1}\"}\n</tool_call>",
			want: Answer{Calling: true}, names: []string{"edit", "run"}, args: []string{`{"path":"a.go","count":3}`, `{"x": 1}`}},
		{name: "call not ended", text: "</think>\n\n<tool_call>\n{\"name\": \"ed", thinking: true,
			want: Answer{Calling: true}},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := Qwen3.Parse(c.text, c.thinking, nil, c.final)
			var names, args []string
			for _, tc := range got.ToolCalls {
				names = append(names, tc.Function.Name)
				args = append(args, tc.Function.Arguments)
			}
			got.ToolCalls = nil
			if !reflect.DeepEqual(got, c.want) || !reflect.DeepEqual(names, c.names) || !reflect.DeepEqual(args, c.args) {
				t.Errorf("got %+v %q %q, want %+v %q %q", got, names, args, c.want, c.names, c.args)
			}
		})
	}
	if tp, err := TemplateFor("qwen3"); err != nil || tp != Qwen3 {
		t.Errorf("TemplateFor(qwen3) = %v, %v", tp, err)
	}
	if !strings.HasSuffix(mustRender(t), "<|im_start|>assistant\n<think>\n\n</think>\n\n") {
		t.Error("no_think does not close the thinking as enable_thinking=false does")
	}
}

func mustRender(t *testing.T) string {
	t.Helper()
	p, _, err := Qwen3.Render([]Message{{Role: "user", Content: Content{Text: "Hi"}}}, nil, Options{NoThink: true})
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// Qwen3 (llm.Unpaged) is split at its attention only when -paged asks for it.
func TestQwen3NotPagedByDefault(t *testing.T) {
	m, err := qwen3.ParseConfig([]byte(`{"model_type": "qwen3", "num_hidden_layers": 2, "max_position_embeddings": 40960}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := pagedChunk(m, 0, llm.BackendCoreML); got != 0 {
		t.Errorf("pagedChunk(qwen3, 0) = %d, want 0 (whole)", got)
	}
	if got := pagedChunk(m, 1024, llm.BackendCoreML); got != 1024 {
		t.Errorf("pagedChunk(qwen3, 1024) = %d, want 1024", got)
	}
	if got := pagedChunk(m, 0, llm.BackendCoreAI); got != paged.DefaultChunk {
		t.Errorf("pagedChunk(qwen3, 0) on Core AI = %d, want %d", got, paged.DefaultChunk)
	}
	if got := m.UnpagedContext(); got != 4096 {
		t.Errorf("UnpagedContext %d, want 4096", got)
	}
}
