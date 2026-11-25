package text

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/shibukawa/bonsaiai/internal/chat"
)

// The options' zero values are bonsaiai-chat's defaults; Unlimited and a budget below 0 are none.
func TestRequestDefaults(t *testing.T) {
	r := request(nil, DefaultSampling, 0, 0)
	if r.MaxTokens != 1024 || r.Temperature != 0.1 || r.TopK != 50 || r.Repetition != 1.05 || r.TopP != 0 {
		t.Errorf("defaults %+v", r)
	}
	r = request(&Sampling{}, DefaultSampling, Unlimited, 7)
	if r.MaxTokens != 0 || r.Temperature != 0 || r.TopK != 0 || r.Repetition != 0 || r.Seed != 7 {
		t.Errorf("greedy, unlimited, seed 7: %+v", r)
	}
	if r := request(nil, Sampling{}, 50, 0); r.MaxTokens != 50 || r.Temperature != 0 {
		t.Errorf("a translation's default: %+v", r)
	}
	for b, want := range map[int]int{0: 1024, -1: 0, 300: 300} {
		if got := thinkBudget(b); got != want {
			t.Errorf("thinkBudget(%d) = %d, want %d", b, got, want)
		}
	}
}

// Messages become the engine's: media before the text as markers (data URLs, base64 WAV),
// parts in order, the thinking and the calls kept.
func TestChatMessages(t *testing.T) {
	png := []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
	msgs, err := chatMessages([]Message{
		{Role: "user", Text: "what is it?", Media: []Media{{Data: png}, {Audio: true, Data: []byte("RIFF")}}},
		{Role: "assistant", Text: "a dot", Reasoning: "it is small", ToolCalls: []ToolCall{{ID: "c1", Name: "look", Arguments: `{"x":1}`}}},
		{Role: "tool", Text: "ok", ToolCallID: "c1", Name: "look"},
		{Role: "user", Parts: []Part{{Text: "and "}, {Media: &Media{URL: "data:image/png;base64,AAAA"}}, {Text: " this?"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	u := msgs[0].Content
	if u.Text != chat.MediaMarker+chat.MediaMarker+"what is it?" || len(u.Media) != 2 ||
		!strings.HasPrefix(u.Media[0].Data, "data:image/png;base64,") || !u.Media[1].Audio || u.Media[1].Data != "UklGRg==" {
		t.Errorf("user message %+v", u)
	}
	a := msgs[1]
	if a.Reasoning == nil || *a.Reasoning != "it is small" || len(a.ToolCalls) != 1 || a.ToolCalls[0].Function.Name != "look" || a.ToolCalls[0].Type != "function" {
		t.Errorf("assistant message %+v", a)
	}
	if msgs[2].ToolCallID != "c1" || msgs[2].Name != "look" {
		t.Errorf("tool message %+v", msgs[2])
	}
	p := msgs[3].Content
	if !p.Parts || p.Text != "and "+chat.MediaMarker+" this?" || len(p.Media) != 1 || p.Media[0].Data != "data:image/png;base64,AAAA" {
		t.Errorf("parts %+v", p)
	}
	if _, err := chatMessages([]Message{{Role: "user", Media: []Media{{Data: []byte("plain text")}}}}); !errors.Is(err, ErrMedia) {
		t.Errorf("a text as an image: %v, want ErrMedia", err)
	}
}

// A tool read from the API's JSON is written back as it was; one made in Go is the API's form.
func TestTools(t *testing.T) {
	raw := `{"type": "function", "function": {"name": "bash", "description": "run it", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}`
	tool, err := ParseTool([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if tool.Name != "bash" || tool.Description != "run it" || !strings.Contains(string(tool.Parameters), "command") {
		t.Errorf("parsed %+v", tool)
	}
	if b, _ := tool.MarshalJSON(); string(b) != raw {
		t.Errorf("written back %s", b)
	}
	made := Tool{Name: "add", Parameters: json.RawMessage(`{"type":"object"}`)}
	ts, err := chatTools([]Tool{tool, made})
	if err != nil || len(ts) != 2 || ts[0].Name() != "bash" || ts[1].Name() != "add" {
		t.Errorf("chatTools: %v %v", ts, err)
	}
	if _, err := ParseTool([]byte(`{"type": "function", "function": {}}`)); err == nil {
		t.Error("a tool without a name was read")
	}
}

// Format tells the model in the system message and takes the JSON out of the answer, checked
// against the schema.
func TestFormat(t *testing.T) {
	f := &Format{Schema: json.RawMessage(`{"type": "object", "required": ["name", "age"], "additionalProperties": false,
		"properties": {"name": {"type": "string", "minLength": 1}, "age": {"type": "integer", "minimum": 0},
		"tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 2}}}`)}
	msgs := f.instruct([]chat.Message{{Role: "user", Content: chat.Content{Text: "hi"}}})
	if len(msgs) != 2 || msgs[0].Role != "system" || !strings.Contains(msgs[0].Content.Text, `"required":["name","age"]`) {
		t.Errorf("instructed %+v", msgs)
	}
	msgs = f.instruct([]chat.Message{{Role: "system", Content: chat.Content{Text: "Be kind."}}, {Role: "user"}})
	if len(msgs) != 2 || !strings.HasPrefix(msgs[0].Content.Text, "Be kind.\n\nAnswer with JSON") {
		t.Errorf("instructed after a system message %+v", msgs[0])
	}
	for answer, want := range map[string]string{
		"```json\n{\"name\": \"Ann\", \"age\": 3}\n```":       `{"name": "Ann", "age": 3}`,
		`Sure: {"name": "Bo", "age": 0, "tags": ["a"]} done.`: `{"name": "Bo", "age": 0, "tags": ["a"]}`,
		`{"name": "Ann"}`:                                  "",
		`{"name": "Ann", "age": 1.5}`:                      "",
		`{"name": "Ann", "age": -1}`:                       "",
		`{"name": "", "age": 1}`:                           "",
		`{"name": "Ann", "age": 1, "x": 1}`:                "",
		`{"name": "Ann", "age": 1, "tags": ["c"]}`:         "",
		`{"name": "Ann", "age": 1, "tags": ["a","b","a"]}`: "",
		`no JSON here`:                                     "",
	} {
		got, err := f.check(answer)
		var fe *FormatError
		switch {
		case want == "" && !errors.As(err, &fe):
			t.Errorf("%q: %q, %v; want a FormatError", answer, got, err)
		case want != "" && (err != nil || got != want):
			t.Errorf("%q: %q, %v; want %q", answer, got, err, want)
		}
	}
	anyObject := &Format{}
	if got, err := anyObject.check(`here: {"a": [1, 2]}`); err != nil || got != `{"a": [1, 2]}` {
		t.Errorf("any object: %q, %v", got, err)
	}
	if _, err := anyObject.check(`[1, 2]`); err == nil {
		t.Error("an array taken for an object")
	}
	oneOf := &Format{Schema: json.RawMessage(`{"anyOf": [{"type": "string"}, {"type": "null"}]}`)}
	if _, err := oneOf.check(`[null]`); err == nil {
		t.Error("an array matched anyOf string or null")
	}
}

// Stats tell the speed.
func TestStats(t *testing.T) {
	s := stats(chat.Stats{Prompt: 10, Reused: 4, Generated: 20, Decode: 2e9, Drafted: 6, Accepted: 3, Seed: 9, ThinkCut: true})
	if s.TokensPerSecond() != 10 || s.Prompt != 10 || s.Reused != 4 || s.Accepted != 3 || s.Seed != 9 || !s.ThinkCut {
		t.Errorf("%+v", s)
	}
}
