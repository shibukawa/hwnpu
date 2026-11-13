package chat

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

// GptOss.Render writes what gpt-oss's chat_template.jinja does (testdata/gptoss_template.json:
// tools/ref/template_golden.py --harmony --text-content with openai/gpt-oss-20b, the date fixed at
// 2026-11-13, reasoning_effort low for no_think).
func TestGptOssRenderMatchesTemplate(t *testing.T) {
	raw, err := os.ReadFile("testdata/gptoss_template.json")
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
	defer func(f func() time.Time) { harmonyNow = f }(harmonyNow)
	harmonyNow = func() time.Time { return time.Date(2026, 11, 13, 0, 0, 0, 0, time.UTC) }
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			got, thinking, err := GptOss.Render(c.Messages, c.Tools, Options{NoThink: c.NoThink})
			if err != nil {
				t.Fatal(err)
			}
			if thinking {
				t.Error("the prompt ends before the model picks its channel, not inside thinking")
			}
			if got != c.Want {
				t.Errorf("got\n%q\nwant\n%q", got, c.Want)
			}
		})
	}
	if tp, err := TemplateFor("gptoss"); err != nil || tp != GptOss {
		t.Errorf("TemplateFor(gptoss) = %v, %v", tp, err)
	}
	if a, err := archOf("GptOss"); err != nil || a != "gptoss" {
		t.Errorf("archOf(GptOss) = %q, %v", a, err)
	}
}

// GptOss.Parse reads harmony's messages: analysis as reasoning, final as content, a commentary
// message to=functions.NAME as a tool call; a cut text is still inside its message.
func TestGptOssParse(t *testing.T) {
	text := "<|channel|>analysis<|message|>The user greets.<|end|><|start|>assistant<|channel|>final<|message|>Hello!<|return|>"
	a := GptOss.Parse(text, false, nil, true)
	if a.Reasoning != "The user greets." || a.Content != "Hello!" || a.Thinking || len(a.ToolCalls) != 0 {
		t.Errorf("answer: %+v", a)
	}
	call := "<|channel|>analysis<|message|>Need weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Tokyo\"}<|call|>"
	a = GptOss.Parse(call, false, nil, true)
	if len(a.ToolCalls) != 1 || a.ToolCalls[0].Function.Name != "get_weather" || a.ToolCalls[0].Function.Arguments != `{"city":"Tokyo"}` || !a.Calling || a.Content != "" {
		t.Errorf("call: %+v", a)
	}
	// the recipient before the channel, as the template writes it
	a = GptOss.Parse(" to=functions.run<|channel|>commentary json<|message|>{}<|call|>", false, nil, true)
	if len(a.ToolCalls) != 1 || a.ToolCalls[0].Function.Name != "run" {
		t.Errorf("recipient first: %+v", a)
	}
	// cut while thinking, then while answering, then inside a call's arguments
	if a := GptOss.Parse("<|channel|>analysis<|message|>Thin", false, nil, false); !a.Thinking || a.Reasoning != "Thin" {
		t.Errorf("thinking: %+v", a)
	}
	if a := GptOss.Parse(text[:strings.Index(text, "!")], false, nil, false); a.Thinking || a.Content != "Hello" {
		t.Errorf("answering: %+v", a)
	}
	if a := GptOss.Parse(call[:len(call)-12], false, nil, false); !a.Calling || len(a.ToolCalls) != 0 {
		t.Errorf("calling: %+v", a)
	}
}
