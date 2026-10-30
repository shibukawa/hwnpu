// Package chat turns a conversation into a prompt and a generated answer back into a message,
// with the chat template of LFM3 (tokenizer chat_template.jinja of LiquidAI/LFM3-1.7B), and
// runs it on a model with a KV cache (Engine): an interactive chat and an OpenAI compatible
// server (Server) share it. Tool calls are the template's XML form, converted to and from the
// OpenAI function calls.
package chat

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"strings"

	"github.com/shibukawa/hwnpu/internal/pyjson"
)

// Message is one message of a conversation, in the shape of the OpenAI chat API.
type Message struct {
	Role    string  `json:"role"`
	Content Content `json:"content"`
	// Reasoning is the thinking before the answer (reasoning_content, as in vLLM and
	// DeepSeek); nil when the client did not send it.
	Reasoning  *string    `json:"reasoning_content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// Content is a message's text. The API sends it as a string, null or a list of parts; the text
// parts are kept, and each image (image_url, a data URL) or audio (input_audio, base64 WAV) part
// becomes MediaMarker in the text where it was, as llama.cpp's server does, and goes to Media.
type Content struct {
	Text  string
	Parts bool // sent as a list of parts
	parts []string
	Media []Media
}

// Media is an image or an audio clip of a message.
type Media struct {
	Audio bool
	Data  string // an image's data URL, or a clip's base64 WAV (or data URL)
}

// MediaMarker stands for an image or a clip in a rendered prompt; Engine.Tokens puts its
// embeddings there. It ends in a random number, so that a text cannot write it by accident.
var MediaMarker = fmt.Sprintf("<__media_%016x__>", rand.Uint64())

func (c *Content) UnmarshalJSON(b []byte) error {
	*c = Content{}
	switch {
	case string(b) == "null":
		return nil
	case len(b) > 0 && b[0] == '"':
		return json.Unmarshal(b, &c.Text)
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
		return fmt.Errorf("chat: content is neither a string nor a list of parts: %w", err)
	}
	c.Parts = true
	for _, p := range parts {
		switch {
		case p.Type == "text":
			c.parts = append(c.parts, p.Text)
			c.Text += p.Text
		case p.Type == "image_url" && p.ImageURL != nil:
			c.Media = append(c.Media, Media{Data: p.ImageURL.URL})
			c.Text += MediaMarker
		case p.Type == "input_audio" && p.InputAudio != nil:
			c.Media = append(c.Media, Media{Audio: true, Data: p.InputAudio.Data})
			c.Text += MediaMarker
		}
	}
	return nil
}

func (c Content) MarshalJSON() ([]byte, error) { return json.Marshal(c.Text) }

// ToolCall is a call of a function by the assistant. Arguments is a JSON object as a string.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// Tool is a function the model may call: the request's {"type": "function", "function": {...}}.
type Tool struct {
	raw *pyjson.Value // the function object
}

func (t *Tool) UnmarshalJSON(b []byte) error {
	v, err := pyjson.Parse(b)
	if err != nil {
		return err
	}
	if v.Kind != '{' {
		return fmt.Errorf("chat: a tool is not an object")
	}
	if f := v.Get("function"); f != nil {
		v = f
	}
	if n := v.Get("name"); n == nil || n.Kind != '"' {
		return fmt.Errorf("chat: a tool has no name")
	}
	t.raw = v
	return nil
}

// Name is the function's name.
func (t Tool) Name() string { return t.raw.Get("name").Str }

// paramType is the JSON schema type of a parameter ("" when not given).
func (t Tool) paramType(param string) string {
	p := t.raw.Get("parameters").Get("properties").Get(param)
	if ty := p.Get("type"); ty != nil && ty.Kind == '"' {
		return ty.Str
	}
	return ""
}

// Template options.
type Options struct {
	// NoThink starts the answer with an empty <think></think>, so the model answers at once.
	NoThink bool
	// KeepThinking keeps the thinking of assistant messages before the last user message
	// (the template's preserve_thinking).
	KeepThinking bool
}

// Render writes the conversation as the prompt of the next assistant message.
func Render(msgs []Message, tools []Tool, o Options) (string, error) {
	var sb strings.Builder
	sb.WriteString("<|startoftext|>")
	system := ""
	if len(msgs) > 0 && msgs[0].Role == "system" {
		system = msgs[0].Content.Text
		msgs = msgs[1:]
	}
	if len(tools) > 0 {
		if system != "" {
			system += "\n\n"
		}
		system += "<tools>\n"
		for i, t := range tools {
			if i > 0 {
				system += "\n"
			}
			system += toolJSON(t.raw)
		}
		system += "\n</tools>"
	}
	if system != "" {
		sb.WriteString("<|im_start|>system\n" + system + "<|im_end|>\n")
	}
	lastUser := -1
	for i, m := range msgs {
		if m.Role == "user" {
			lastUser = i
		}
	}
	var pending []string // tool results merged with the next (same call)
	for i, m := range msgs {
		if m.Role == "tool" {
			items := m.Content.parts
			if !m.Content.Parts && m.Content.Text != "" {
				items = []string{m.Content.Text}
			}
			if i+1 < len(msgs) && msgs[i+1].Role == "tool" && m.ToolCallID != "" && msgs[i+1].ToolCallID == m.ToolCallID {
				pending = append(pending, items...)
				continue
			}
			sb.WriteString("<|im_start|>tool\n")
			switch {
			case len(pending) > 0:
				toolResponse(&sb, strings.Join(append(pending, items...), "\n"))
				pending = nil
			case m.Content.Parts:
				toolResponse(&sb, strings.Join(items, "\n"))
			default:
				toolResponse(&sb, m.Content.Text)
			}
			sb.WriteString("<|im_end|>\n")
			continue
		}
		sb.WriteString("<|im_start|>" + m.Role + "\n")
		if m.Role != "assistant" {
			sb.WriteString(m.Content.Text + "<|im_end|>\n")
			continue
		}
		keep := o.KeepThinking || i > lastUser
		content := m.Content.Text
		switch {
		case m.Reasoning != nil && keep:
			sb.WriteString("<think>" + *m.Reasoning + "</think>")
		case keep && !strings.Contains(content, "</think>"):
			sb.WriteString("<think></think>")
		}
		if !keep {
			if j := strings.LastIndex(content, "</think>"); j >= 0 {
				content = strings.TrimSpace(content[j+len("</think>"):])
			}
			content = "<think></think>" + content
		}
		sb.WriteString(content)
		for j, tc := range m.ToolCalls {
			if j > 0 {
				sb.WriteString("\n\n")
			}
			if err := renderCall(&sb, tc); err != nil {
				return "", err
			}
		}
		sb.WriteString("<|im_end|>\n")
	}
	sb.WriteString("<|im_start|>assistant\n")
	if o.NoThink {
		sb.WriteString("<think></think>")
	} else {
		sb.WriteString("<think>")
	}
	return sb.String(), nil
}

func toolResponse(sb *strings.Builder, s string) {
	if strings.Contains(s, "<tool_response>") {
		sb.WriteString(s)
		return
	}
	sb.WriteString("<tool_response>\n" + s + "\n</tool_response>")
}

// toolJSON writes a tool's function object without defer_loading.
func toolJSON(f *pyjson.Value) string {
	var sb strings.Builder
	sb.WriteByte('{')
	n := 0
	for i, k := range f.Keys {
		if k == "defer_loading" {
			continue
		}
		if n > 0 {
			sb.WriteString(", ")
		}
		n++
		pyjson.String(&sb, k)
		sb.WriteString(": ")
		f.Fields[i].Write(&sb)
	}
	sb.WriteByte('}')
	return sb.String()
}

func renderCall(sb *strings.Builder, tc ToolCall) error {
	sb.WriteString("<tool_call>\n<function=" + tc.Function.Name + ">\n")
	if a := strings.TrimSpace(tc.Function.Arguments); a != "" && a != "{}" && a != "null" {
		args, err := pyjson.Parse([]byte(a))
		if err != nil {
			return fmt.Errorf("chat: arguments of %s: %w", tc.Function.Name, err)
		}
		for i, k := range args.Keys {
			sb.WriteString("<parameter=" + k + ">\n")
			if v := args.Fields[i]; v.Kind == '"' {
				sb.WriteString(v.Str)
			} else {
				v.Write(sb)
			}
			sb.WriteString("\n</parameter>\n")
		}
	}
	sb.WriteString("</function>\n</tool_call>")
	return nil
}
