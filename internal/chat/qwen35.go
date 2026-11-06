package chat

import (
	"fmt"
	"strings"

	"github.com/shibukawa/hwnpu/internal/pyjson"
)

// Qwen35 is the chat format of Qwen/Qwen3.5 (chat_template.jinja of Qwen3.5-4B, the same for
// every size): ChatML turns, thinking in <think> at the start of the answers after the last user
// question, and tool calls in the XML form LFM3 also uses (<tool_call><function=NAME>
// <parameter=KEY>), the tools declared in the system turn with Qwen's instructions.
var Qwen35 Template = qwen35Format{}

// Qwen36 is the chat format of Qwen/Qwen3.6 (chat_template.jinja of Qwen3.6-35B-A3B): Qwen35's,
// but a tool call's argument that is not a string is written as JSON (true, null, 1.5) where
// Qwen3.5 writes Python's str() of it (True, None). Its preserve_thinking is o.KeepThinking.
var Qwen36 Template = qwen35Format{json: true}

type qwen35Format struct {
	json bool // Qwen3.6: every argument but a string as JSON
}

func (qwen35Format) Stops() []string { return []string{"<|im_end|>", "<|endoftext|>"} }

func (qwen35Format) Think() (open, close, end string) {
	return "<think>", "</think>", thinkOver + "</think>\n\n"
}

// qwen35Tools is what the template writes after the tools' declarations.
const qwen35Tools = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

// Render writes what the template does with add_generation_prompt, and enable_thinking unless
// o.NoThink. The thinking of the answers before the last user question is left out, as the
// template does, unless o.KeepThinking.
func (f qwen35Format) Render(msgs []Message, tools []Tool, o Options) (string, bool, error) {
	if len(msgs) == 0 {
		return "", false, fmt.Errorf("chat: no messages")
	}
	var sb strings.Builder
	first := 0
	system, hasSystem := "", msgs[0].Role == "system"
	if hasSystem {
		system, first = strings.TrimSpace(msgs[0].Content.Text), 1
	}
	if len(tools) > 0 {
		sb.WriteString("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>")
		for _, t := range tools {
			sb.WriteString("\n{\"type\": \"function\", \"function\": " + toolJSON(t.raw) + "}")
		}
		sb.WriteString(qwen35Tools)
		if system != "" {
			sb.WriteString("\n\n" + system)
		}
		sb.WriteString("<|im_end|>\n")
	} else if hasSystem {
		sb.WriteString("<|im_start|>system\n" + system + "<|im_end|>\n")
	}
	// the last user message that is not a tool's response
	lastQuery := -1
	for i := len(msgs) - 1; i >= 0; i-- {
		c := strings.TrimSpace(msgs[i].Content.Text)
		if msgs[i].Role == "user" && !(strings.HasPrefix(c, "<tool_response>") && strings.HasSuffix(c, "</tool_response>")) {
			lastQuery = i
			break
		}
	}
	if lastQuery < 0 {
		return "", false, fmt.Errorf("chat: no user query found in the messages")
	}
	for i := first; i < len(msgs); i++ {
		m := msgs[i]
		content := strings.TrimSpace(m.Content.Text)
		switch m.Role {
		case "system":
			return "", false, fmt.Errorf("chat: a system message must be at the beginning")
		case "user":
			sb.WriteString("<|im_start|>user\n" + content + "<|im_end|>\n")
		case "assistant":
			reasoning := ""
			if m.Reasoning != nil {
				reasoning = *m.Reasoning
			} else if j := strings.Index(content, "</think>"); j >= 0 {
				r := strings.TrimRight(content[:j], "\n")
				if k := strings.LastIndex(r, "<think>"); k >= 0 {
					r = r[k+len("<think>"):]
				}
				reasoning = strings.TrimLeft(r, "\n")
				content = strings.TrimLeft(content[strings.LastIndex(content, "</think>")+len("</think>"):], "\n")
			}
			reasoning = strings.TrimSpace(reasoning)
			if i > lastQuery || o.KeepThinking {
				sb.WriteString("<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content)
			} else {
				sb.WriteString("<|im_start|>assistant\n" + content)
			}
			for j, tc := range m.ToolCalls {
				switch {
				case j > 0:
					sb.WriteString("\n")
				case strings.TrimSpace(content) != "":
					sb.WriteString("\n\n")
				}
				if err := qwen35Call(&sb, tc, f.json); err != nil {
					return "", false, err
				}
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

// qwen35Call writes a tool call: each argument a parameter, a string as it is, an object or a
// list as JSON, and anything else as Python's str() of it, or (asJSON, Qwen3.6) as JSON.
func qwen35Call(sb *strings.Builder, tc ToolCall, asJSON bool) error {
	sb.WriteString("<tool_call>\n<function=" + tc.Function.Name + ">\n")
	if a := strings.TrimSpace(tc.Function.Arguments); a != "" && a != "null" {
		args, err := pyjson.Parse([]byte(a))
		if err != nil {
			return fmt.Errorf("chat: arguments of %s: %w", tc.Function.Name, err)
		}
		for i, k := range args.Keys {
			sb.WriteString("<parameter=" + k + ">\n")
			switch v := args.Fields[i]; {
			case v.Kind == '{' || v.Kind == '[':
				v.Write(sb)
			case asJSON && v.Kind == 'n':
				sb.WriteString(pyNumber(v.Str)) // json.dumps of the float or int Python read
			case asJSON && v.Kind != '"':
				v.Write(sb)
			default:
				sb.WriteString(pyStr(v))
			}
			sb.WriteString("\n</parameter>\n")
		}
	}
	sb.WriteString("</function>\n</tool_call>")
	return nil
}

// Parse takes apart an answer: the thinking up to </think> (without the newlines the template
// puts around it), then the content and the tool calls as in LFM3's Parse.
func (qwen35Format) Parse(text string, thinking bool, tools []Tool, final bool) Answer {
	a := Parse(text, thinking, tools, final)
	a.Reasoning = strings.TrimLeft(a.Reasoning, "\n")
	if !a.Thinking {
		a.Reasoning = strings.TrimRight(a.Reasoning, "\n")
	}
	return a
}
