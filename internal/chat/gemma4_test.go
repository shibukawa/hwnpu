package chat

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

// Gemma4.Render writes what Gemma 4's chat_template.jinja does. The expected prompts in
// testdata/gemma4_template.json were rendered by transformers' apply_chat_template
// (tools/ref/template_golden.py).
func TestGemma4RenderMatchesTemplate(t *testing.T) {
	raw, err := os.ReadFile("testdata/gemma4_template.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name         string    `json:"name"`
		NoThink      bool      `json:"no_think"`
		KeepThinking bool      `json:"keep_thinking"`
		Messages     []Message `json:"messages"`
		Tools        []Tool    `json:"tools"`
		Want         string    `json:"want"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			got, thinking, err := Gemma4.Render(c.Messages, c.Tools, Options{NoThink: c.NoThink, KeepThinking: c.KeepThinking})
			if err != nil {
				t.Fatal(err)
			}
			if got != c.Want {
				t.Errorf("got\n%s\nwant\n%s", got, c.Want)
			}
			if want := len(got) > 0 && got[len(got)-len("<|channel>thought\n"):] == "<|channel>thought\n"; thinking != want {
				t.Errorf("thinking %v, want %v", thinking, want)
			}
		})
	}
}

func TestGemma4Parse(t *testing.T) {
	call := `<|tool_call>call:edit{edits:[{new:<|"|>x := 1.0<|"|>,path:<|"|>a.go<|"|>}],big:1e+20,none:null,ok:true,n:3}<tool_call|>`
	for _, c := range []struct {
		name     string
		text     string
		thinking bool
		final    bool
		want     Answer
		args     []string
	}{
		{name: "answer", text: "Hello<turn|>", final: true, want: Answer{Content: "Hello"}},
		{name: "thinking so far", text: "<|channel>thought\nLet me", want: Answer{Reasoning: "Let me", Thinking: true}},
		{name: "thought and answer", text: "<|channel>thought\nHmm.\n<channel|>Hi", final: true, want: Answer{Reasoning: "Hmm.", Content: "Hi"}},
		{name: "empty thought", text: "<|channel>thought\n<channel|>Hi", final: true, want: Answer{Content: "Hi"}},
		{name: "prompt inside the channel", text: "Next.\n<channel|>Done", thinking: true, final: true, want: Answer{Reasoning: "Next.", Content: "Done"}},
		{name: "call", text: "Editing.\n" + call + "<|tool_response>", final: true, want: Answer{Content: "Editing.", Calling: true},
			args: []string{`{"edits":[{"new":"x := 1.0","path":"a.go"}],"big":1e+20,"none":null,"ok":true,"n":3}`}},
		{name: "call not ended", text: `<|tool_call>call:edit{edits:[`, want: Answer{Calling: true}},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := Gemma4.Parse(c.text, c.thinking, nil, c.final)
			var args []string
			for _, tc := range got.ToolCalls {
				args = append(args, tc.Function.Arguments)
				if !json.Valid([]byte(tc.Function.Arguments)) {
					t.Errorf("arguments %s are not JSON", tc.Function.Arguments)
				}
			}
			got.ToolCalls = nil
			if !reflect.DeepEqual(got, c.want) || !reflect.DeepEqual(args, c.args) {
				t.Errorf("got %+v %q, want %+v %q", got, args, c.want, c.args)
			}
		})
	}
}

// A call parsed from the model's text renders back as the same text.
func TestGemma4CallRoundTrip(t *testing.T) {
	call := `<|tool_call>call:get_weather{city:<|"|>Tokyo, "Japan"<|"|>,days:2,opts:{deep:[1,2.5,<|"|>x<|"|>]}}<tool_call|>`
	a := Gemma4.Parse(call+"<|tool_response>", false, nil, true)
	if len(a.ToolCalls) != 1 {
		t.Fatalf("parsed %+v", a)
	}
	p, _, err := Gemma4.Render([]Message{{Role: "user", Content: Content{Text: "go"}}, {Role: "assistant", ToolCalls: a.ToolCalls}}, nil, Options{NoThink: true})
	if err != nil {
		t.Fatal(err)
	}
	want := "<bos><|turn>user\ngo<turn|>\n<|turn>model\n" + call + "<|tool_response>"
	if p != want {
		t.Errorf("got\n%s\nwant\n%s", p, want)
	}
}
