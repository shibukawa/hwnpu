package chat

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
)

// LFM2.Render writes what LFM2-8B-A1B's chat template does: testdata/lfm2_template.json was
// rendered by transformers' apply_chat_template (tools/ref/template_golden.py --text-content).
func TestLFM2RenderMatchesTemplate(t *testing.T) {
	raw, err := os.ReadFile("testdata/lfm2_template.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name     string    `json:"name"`
		Messages []Message `json:"messages"`
		Tools    []Tool    `json:"tools"`
		Want     string    `json:"want"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			got, thinking, err := LFM2.Render(c.Messages, c.Tools, Options{})
			if err != nil {
				t.Fatal(err)
			}
			if got != c.Want {
				t.Errorf("got\n%s\nwant\n%s", got, c.Want)
			}
			if thinking {
				t.Error("thinking")
			}
		})
	}
}

func TestLFM2Parse(t *testing.T) {
	for _, c := range []struct {
		name    string
		text    string
		final   bool
		content string
		calling bool
		names   []string
		args    []string
	}{
		{name: "text", text: "Tokyo.<|im_end|>", final: true, content: "Tokyo."},
		{name: "a tail held back", text: "Let me look.<|tool_ca", content: "Let me look."},
		{name: "two calls", final: true, text: "Looking.<|tool_call_start|>[get_weather(city='Tokyo', days=2, unit=\"celsius\"), get_weather(city='Osaka')]<|tool_call_end|><|im_end|>",
			content: "Looking.", calling: true, names: []string{"get_weather", "get_weather"},
			args: []string{`{"city":"Tokyo","days":2,"unit":"celsius"}`, `{"city":"Osaka"}`}},
		{name: "literals", final: true,
			text:    `<|tool_call_start|>[edit(edits=[{"path": "a.go", "new": 'it\'s\n'}], options={'dry_run': False}, big=1e+20, none=None, neg=-3, t=(1, 2))]<|tool_call_end|>`,
			calling: true, names: []string{"edit"},
			args: []string{`{"edits":[{"path":"a.go","new":"it's\n"}],"options":{"dry_run":false},"big":1e+20,"none":null,"neg":-3,"t":[1,2]}`}},
		{name: "no arguments", final: true, text: "<|tool_call_start|>[now()]<|tool_call_end|>", calling: true, names: []string{"now"}, args: []string{`{}`}},
		{name: "not ended", text: "<|tool_call_start|>[get_weather(city='To", calling: true},
		{name: "a call it cannot read, then one it can", final: true, text: "<|tool_call_start|>[bad(1 2), ok(x=1)]<|tool_call_end|>",
			calling: true, names: []string{"ok"}, args: []string{`{"x":1}`}},
	} {
		t.Run(c.name, func(t *testing.T) {
			a := LFM2.Parse(c.text, false, nil, c.final)
			if a.Content != c.content || a.Calling != c.calling {
				t.Errorf("content %q calling %v, want %q %v", a.Content, a.Calling, c.content, c.calling)
			}
			var names, args []string
			for _, tc := range a.ToolCalls {
				names, args = append(names, tc.Function.Name), append(args, tc.Function.Arguments)
			}
			if !slices.Equal(names, c.names) || !slices.Equal(args, c.args) {
				t.Errorf("calls %v %v, want %v %v", names, args, c.names, c.args)
			}
		})
	}
	// a call written by Render reads back
	var tc ToolCall
	tc.Function.Name, tc.Function.Arguments = "f", `{"s":"a'b\\c","n":0.5,"l":[1,"x"]}`
	p, _, err := LFM2.Render([]Message{{Role: "assistant", ToolCalls: []ToolCall{tc}}}, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	_, body, _ := cutString(p, "<|im_start|>assistant\n")
	a := LFM2.Parse(body, false, nil, true)
	if len(a.ToolCalls) != 1 || a.ToolCalls[0].Function.Arguments != `{"s":"a'b\\c","n":0.5,"l":[1,"x"]}` {
		t.Errorf("round trip: %q gave %+v", body, a.ToolCalls)
	}
}

func cutString(s, sep string) (string, string, bool) {
	for i := 0; i+len(sep) <= len(s); i++ {
		if s[i:i+len(sep)] == sep {
			return s[:i], s[i+len(sep):], true
		}
	}
	return s, "", false
}
