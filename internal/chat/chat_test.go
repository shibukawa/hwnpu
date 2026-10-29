package chat

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

// Render writes what the template (chat_template.jinja of LiquidAI/LFM3-1.7B) does. The
// expected prompts in testdata/template.json were rendered with jinja2 the way transformers'
// apply_chat_template renders them.
func TestRenderMatchesTemplate(t *testing.T) {
	raw, err := os.ReadFile("testdata/template.json")
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
			got, err := Render(c.Messages, c.Tools, Options{NoThink: c.NoThink})
			if err != nil {
				t.Fatal(err)
			}
			if got != c.Want {
				t.Errorf("got\n%q\nwant\n%q", got, c.Want)
			}
		})
	}
}

func TestParse(t *testing.T) {
	var tools []Tool
	if err := json.Unmarshal([]byte(`[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"number"}}}}}]`), &tools); err != nil {
		t.Fatal(err)
	}
	call := "<tool_call>\n<function=bash>\n<parameter=command>\necho 1\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>"
	for _, c := range []struct {
		name     string
		text     string
		thinking bool
		final    bool
		want     Answer
		args     []string
	}{
		{name: "thinking", text: "Let me see", thinking: true, want: Answer{Reasoning: "Let me see", Thinking: true}},
		{name: "answer", text: "hmm</think>\n\nHello <tool", thinking: true, want: Answer{Reasoning: "hmm", Content: "Hello "}},
		{name: "answer final", text: "hmm</think>\n\nHello <tool", thinking: true, final: true, want: Answer{Reasoning: "hmm", Content: "Hello <tool"}},
		{name: "no thinking", text: "Hi<|im_end|>", final: true, want: Answer{Content: "Hi"}},
		{name: "thinking anyway", text: "So.</think>Hi", final: true, want: Answer{Reasoning: "So.", Content: "Hi"}},
		{name: "call started", text: "Running.\n<tool_call>\n<function=bash>\n<par", want: Answer{Content: "Running.", Calling: true}},
		{name: "calls", text: "x</think>Running.\n" + call + "\n\n" + call + "<|im_end|>", thinking: true, final: true,
			want: Answer{Reasoning: "x", Content: "Running.", Calling: true}, args: []string{`{"command":"echo 1","timeout":30}`, `{"command":"echo 1","timeout":30}`}},
		{name: "unknown tool", text: "<tool_call>\n<function=other>\n<parameter=n>\n007\n</parameter>\n<parameter=s>\nnot json\n</parameter>\n</function>\n</tool_call>", final: true,
			want: Answer{Calling: true}, args: []string{`{"n":"007","s":"not json"}`}},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := Parse(c.text, c.thinking, tools, c.final)
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
}

// A generated call rendered back into the prompt reads as the model wrote it.
func TestCallRoundTrip(t *testing.T) {
	var tools []Tool
	json.Unmarshal([]byte(`[{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"n":{"type":"integer"}}}}}]`), &tools)
	call := "<tool_call>\n<function=edit>\n<parameter=path>\na.go\n</parameter>\n<parameter=old>\n\tx := 1\n\n</parameter>\n<parameter=n>\n2\n</parameter>\n</function>\n</tool_call>"
	a := Parse("Editing."+call, false, tools, true)
	if len(a.ToolCalls) != 1 {
		t.Fatalf("got %d calls", len(a.ToolCalls))
	}
	p, err := Render([]Message{{Role: "user", Content: Content{Text: "go"}}, {Role: "assistant", Content: Content{Text: a.Content}, ToolCalls: a.ToolCalls}}, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>assistant\n<think></think>Editing." + call + "<|im_end|>\n"
	if !strings.Contains(p, want) {
		t.Errorf("rendered\n%q\nwant it to contain\n%q", p, want)
	}
}

func TestLooping(t *testing.T) {
	seq := func(n int, f func(i int) int32) []int32 {
		s := make([]int32, n)
		for i := range s {
			s[i] = f(i)
		}
		return s
	}
	for _, c := range []struct {
		name string
		gen  []int32
		want bool
	}{
		{"one token", seq(80, func(int) int32 { return 7 }), true},
		{"period 16", append(seq(20, func(i int) int32 { return int32(100 + i) }), seq(64, func(i int) int32 { return int32(i % 16) })...), true},
		{"period 17", seq(100, func(i int) int32 { return int32(i % 17) }), false},
		{"too short", seq(63, func(int) int32 { return 7 }), false},
		{"broken once", append(seq(40, func(int) int32 { return 7 }), append([]int32{8}, seq(23, func(int) int32 { return 7 })...)...), false},
	} {
		if got := looping(c.gen); got != c.want {
			t.Errorf("%s: looping %v, want %v", c.name, got, c.want)
		}
	}
}

func TestStream(t *testing.T) {
	var s Stream
	var r, c string
	for _, step := range []struct{ reasoning, content string }{
		{"", "I think"},
		{"", "I think so"},
		{"", "I thi"},          // held back for a stop string: nothing new
		{"I think so.", "Yes"}, // the model closed a thinking it was not asked for
		{"I think so.", "Yes!"},
	} {
		dr, dc := s.Next(step.reasoning, step.content)
		r, c = r+dr, c+dc
	}
	if r != "I think so." || c != "I think so\n\nYes!" {
		t.Errorf("sent reasoning %q and content %q", r, c)
	}
}
