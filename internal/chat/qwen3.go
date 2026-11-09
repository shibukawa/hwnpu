package chat

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/shibukawa/hwnpu/internal/pyjson"
)

// Qwen3 is the chat format of Qwen/Qwen3 (the chat_template of Qwen3-0.6B's tokenizer_config.json,
// the same for the dense sizes): ChatML turns, thinking in <think> at the start of the answers
// after the last user question, and tool calls as JSON in <tool_call> (Hermes's form,
// {"name": ..., "arguments": {...}}), the tools declared in the system turn.
//
// The template ends the prompt at "<|im_start|>assistant\n" and lets the model open its thinking;
// Render writes the "<think>\n" the model writes first, so that the answer starts inside it as
// Qwen3.5's does. With o.NoThink it writes the empty thinking of enable_thinking=false.
var Qwen3 Template = qwen3Format{}

type qwen3Format struct{}

func (qwen3Format) Stops() []string { return []string{"<|im_end|>", "<|endoftext|>"} }

func (qwen3Format) Think() (open, close, end string) {
	return "<think>", "</think>", thinkOver + "</think>\n\n"
}

// qwen3Tools are what the template writes before and after the tools' declarations.
const (
	qwen3ToolsHead = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
	qwen3ToolsTail = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
)

// Render writes what the template does with add_generation_prompt, and enable_thinking unless
// o.NoThink. The thinking of the answers before the last user question is left out, as the
// template does, unless o.KeepThinking.
func (qwen3Format) Render(msgs []Message, tools []Tool, o Options) (string, bool, error) {
	if len(msgs) == 0 {
		return "", false, fmt.Errorf("chat: no messages")
	}
	var sb strings.Builder
	hasSystem := msgs[0].Role == "system"
	if len(tools) > 0 {
		sb.WriteString("<|im_start|>system\n")
		if hasSystem {
			sb.WriteString(msgs[0].Content.Text + "\n\n")
		}
		sb.WriteString(qwen3ToolsHead)
		for _, t := range tools {
			sb.WriteString("\n{\"type\": \"function\", \"function\": " + toolJSON(t.raw) + "}")
		}
		sb.WriteString(qwen3ToolsTail)
	} else if hasSystem {
		sb.WriteString("<|im_start|>system\n" + msgs[0].Content.Text + "<|im_end|>\n")
	}
	// the last user message that is not a tool's response (the last message when there is none)
	lastQuery := len(msgs) - 1
	for i := len(msgs) - 1; i >= 0; i-- {
		c := msgs[i].Content.Text
		if msgs[i].Role == "user" && !(strings.HasPrefix(c, "<tool_response>") && strings.HasSuffix(c, "</tool_response>")) {
			lastQuery = i
			break
		}
	}
	for i, m := range msgs {
		content := m.Content.Text
		switch m.Role {
		case "user", "system":
			if m.Role == "system" && i == 0 {
				continue
			}
			sb.WriteString("<|im_start|>" + m.Role + "\n" + content + "<|im_end|>\n")
		case "assistant":
			reasoning := ""
			if m.Reasoning != nil {
				reasoning = *m.Reasoning
			} else if strings.Contains(content, "</think>") {
				r, _, _ := strings.Cut(content, "</think>")
				r = strings.TrimRight(r, "\n")
				if k := strings.LastIndex(r, "<think>"); k >= 0 {
					r = r[k+len("<think>"):]
				}
				reasoning = strings.TrimLeft(r, "\n")
				content = strings.TrimLeft(content[strings.LastIndex(content, "</think>")+len("</think>"):], "\n")
			}
			if (i > lastQuery || o.KeepThinking) && (i == len(msgs)-1 || reasoning != "") {
				sb.WriteString("<|im_start|>assistant\n<think>\n" + strings.Trim(reasoning, "\n") + "\n</think>\n\n" + strings.TrimLeft(content, "\n"))
			} else {
				sb.WriteString("<|im_start|>assistant\n" + content)
			}
			for j, tc := range m.ToolCalls {
				if j > 0 || content != "" {
					sb.WriteString("\n")
				}
				sb.WriteString("<tool_call>\n{\"name\": \"" + tc.Function.Name + "\", \"arguments\": ")
				if a := strings.TrimSpace(tc.Function.Arguments); a == "" {
					sb.WriteString("{}")
				} else {
					// the server parses the arguments into an object, which tojson writes
					args, err := pyjson.Parse([]byte(a))
					if err != nil {
						return "", false, fmt.Errorf("chat: arguments of %s: %w", tc.Function.Name, err)
					}
					pyDumps(&sb, args)
				}
				sb.WriteString("}\n</tool_call>")
			}
			sb.WriteString("<|im_end|>\n")
		case "tool":
			if i == 0 || msgs[i-1].Role != "tool" {
				sb.WriteString("<|im_start|>user")
			}
			sb.WriteString("\n<tool_response>\n" + content + "\n</tool_response>")
			if i == len(msgs)-1 || msgs[i+1].Role != "tool" {
				sb.WriteString("<|im_end|>\n")
			}
		default:
			return "", false, fmt.Errorf("chat: unexpected message role %q", m.Role)
		}
	}
	sb.WriteString("<|im_start|>assistant\n")
	if o.NoThink {
		sb.WriteString("<think>\n\n</think>\n\n")
		return sb.String(), false, nil
	}
	sb.WriteString("<think>\n")
	return sb.String(), true, nil
}

var jsonCallRe = regexp.MustCompile(`(?s)<tool_call>\s*(.*?)\s*</tool_call>`)

// Parse takes apart an answer: the thinking up to </think> (without the newlines the template
// puts around it), the content up to the first <tool_call>, and each call's JSON object; one that
// is not an object with a name is left out, as vLLM's Hermes parser does.
func (qwen3Format) Parse(text string, thinking bool, tools []Tool, final bool) Answer {
	a := Parse(text, thinking, tools, final)
	a.Reasoning = strings.TrimLeft(a.Reasoning, "\n")
	if !a.Thinking {
		a.Reasoning = strings.TrimRight(a.Reasoning, "\n")
	}
	if !a.Calling {
		return a
	}
	a.ToolCalls = nil
	_, calls, _ := strings.Cut(text, callTag)
	for _, m := range jsonCallRe.FindAllStringSubmatch(callTag+calls, -1) {
		var call struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(m[1]), &call); err != nil || call.Name == "" {
			continue
		}
		var tc ToolCall
		tc.ID = fmt.Sprintf("call_%d", len(a.ToolCalls))
		tc.Type = "function"
		tc.Function.Name = call.Name
		var s string
		var buf bytes.Buffer
		switch {
		case len(call.Arguments) == 0 || string(call.Arguments) == "null":
			tc.Function.Arguments = "{}"
		case json.Unmarshal(call.Arguments, &s) == nil:
			tc.Function.Arguments = s // the arguments written as a string of JSON
		case json.Compact(&buf, call.Arguments) == nil:
			tc.Function.Arguments = buf.String()
		}
		a.ToolCalls = append(a.ToolCalls, tc)
	}
	return a
}

// pyDumps writes v as json.dumps does (tojson), the numbers as Python writes the int or float it
// read (1e20 as 1e+20).
func pyDumps(sb *strings.Builder, v *pyjson.Value) {
	switch v.Kind {
	case '{', '[':
		open, close := "[", "]"
		if v.Kind == '{' {
			open, close = "{", "}"
		}
		sb.WriteString(open)
		for i, f := range v.Fields {
			if i > 0 {
				sb.WriteString(", ")
			}
			if v.Kind == '{' {
				pyjson.String(sb, v.Keys[i])
				sb.WriteString(": ")
			}
			pyDumps(sb, f)
		}
		sb.WriteString(close)
	case 'n':
		sb.WriteString(pyNumber(v.Str))
	default:
		v.Write(sb)
	}
}
