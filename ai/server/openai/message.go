package openai

import (
	"encoding/json"
	"fmt"

	"github.com/shibukawa/bonsaiai/ai/text"
)

// message is a message of a chat request.
type message struct {
	Role    string  `json:"role"`
	Content content `json:"content"`
	// Reasoning is the thinking before the answer (reasoning_content, as in vLLM and DeepSeek).
	Reasoning  *string `json:"reasoning_content"`
	ToolCalls  []call  `json:"tool_calls"`
	ToolCallID string  `json:"tool_call_id"`
	Name       string  `json:"name"`
}

type call struct {
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// content is a message's content: a string, null or a list of parts (text, image_url with a
// data URL, input_audio with base64 WAV).
type content struct {
	text  string
	parts []text.Part // nil for a string
}

func (c *content) UnmarshalJSON(b []byte) error {
	*c = content{}
	switch {
	case string(b) == "null":
		return nil
	case len(b) > 0 && b[0] == '"':
		return json.Unmarshal(b, &c.text)
	}
	var parts []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL *struct {
			URL string `json:"url"`
		} `json:"image_url"`
		InputAudio *struct {
			Data string `json:"data"`
		} `json:"input_audio"`
	}
	if err := json.Unmarshal(b, &parts); err != nil {
		return fmt.Errorf("content is neither a string nor a list of parts: %w", err)
	}
	c.parts = []text.Part{}
	for _, p := range parts {
		switch {
		case p.Type == "text":
			c.parts = append(c.parts, text.Part{Text: p.Text})
		case p.Type == "image_url" && p.ImageURL != nil:
			c.parts = append(c.parts, text.Part{Media: &text.Media{URL: p.ImageURL.URL}})
		case p.Type == "input_audio" && p.InputAudio != nil:
			c.parts = append(c.parts, text.Part{Media: &text.Media{Audio: true, URL: p.InputAudio.Data}})
		}
	}
	return nil
}

// message is m as the text package takes it.
func (m message) message() text.Message {
	out := text.Message{Role: m.Role, Text: m.Content.text, Parts: m.Content.parts, ToolCallID: m.ToolCallID, Name: m.Name}
	if m.Content.parts != nil && len(m.Content.parts) == 0 {
		out.Parts = []text.Part{{}} // a list of no parts is still one: an empty text
	}
	if m.Reasoning != nil {
		out.Reasoning = *m.Reasoning
	}
	for _, tc := range m.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, text.ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
	}
	return out
}
