package chat

import (
	"fmt"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/pyjson"
)

// GptOss is the chat format of openai/gpt-oss (chat_template.jinja of gpt-oss-20b, the harmony
// format): every message is <|start|>ROLE<|channel|>CHANNEL<|message|>TEXT<|end|>, the model
// thinking in the analysis channel and answering in the final one, a tool call a commentary
// message to=functions.NAME whose text is the arguments' JSON, ended by <|call|>. The system
// message gives the reasoning effort (low with o.NoThink, else medium, the template's default)
// and the date; a system message of the conversation becomes the developer message's
// instructions, the tools its TypeScript namespace. As the template, only the first tool call of
// an assistant message is written, and the thinking of earlier answers is dropped but before a
// tool call no answer follows yet.
var GptOss Template = gptOssFormat{}

type gptOssFormat struct{}

// harmonyNow is the date the system message names (the template's strftime_now).
var harmonyNow = time.Now

func (gptOssFormat) Stops() []string { return []string{"<|return|>", "<|call|>", "<|endoftext|>"} }

// Think has no tokens of its own: the analysis channel opens with the same <|channel|> as the
// final one, so a budget counting from it would cut the answer.
func (gptOssFormat) Think() (open, close, end string) { return "", "", "" }

func (gptOssFormat) Render(msgs []Message, tools []Tool, o Options) (string, bool, error) {
	if len(msgs) == 0 {
		return "", false, fmt.Errorf("chat: no messages")
	}
	var sb strings.Builder
	effort := "medium"
	if o.NoThink {
		effort = "low"
	}
	sb.WriteString("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n")
	sb.WriteString("Knowledge cutoff: 2024-06\nCurrent date: " + harmonyNow().Format("2006-01-02") + "\n\n")
	sb.WriteString("Reasoning: " + effort + "\n\n")
	sb.WriteString("# Valid channels: analysis, commentary, final. Channel must be included for every message.")
	if len(tools) > 0 {
		sb.WriteString("\nCalls to these tools must go to the commentary channel: 'functions'.")
	}
	sb.WriteString("<|end|>")
	developer, loop := "", msgs
	if msgs[0].Role == "system" || msgs[0].Role == "developer" {
		developer, loop = msgs[0].Content.Text, msgs[1:]
	}
	if developer != "" || len(tools) > 0 {
		sb.WriteString("<|start|>developer<|message|>")
		if developer != "" {
			sb.WriteString("# Instructions\n\n" + developer + "\n\n")
		}
		if len(tools) > 0 {
			sb.WriteString("# Tools\n\n")
			harmonyTools(&sb, tools)
		}
		sb.WriteString("<|end|>")
	}
	lastCall := ""
	for i, m := range loop {
		switch m.Role {
		case "assistant":
			if len(m.ToolCalls) > 0 {
				answered := false
				for _, f := range loop[i+1:] {
					if f.Role == "assistant" && len(f.ToolCalls) == 0 {
						answered = true
					}
				}
				thinking := m.Reasoning != nil && *m.Reasoning != ""
				switch {
				case m.Content.Text != "" && thinking:
					return "", false, fmt.Errorf("chat: an assistant message with tool calls has both content and thinking")
				case m.Content.Text != "" && !answered:
					sb.WriteString("<|start|>assistant<|channel|>analysis<|message|>" + m.Content.Text + "<|end|>")
				case thinking && !answered:
					sb.WriteString("<|start|>assistant<|channel|>analysis<|message|>" + *m.Reasoning + "<|end|>")
				}
				tc := m.ToolCalls[0]
				sb.WriteString("<|start|>assistant to=functions." + tc.Function.Name + "<|channel|>commentary json<|message|>")
				if err := harmonyArgs(&sb, tc.Function.Arguments); err != nil {
					return "", false, fmt.Errorf("chat: arguments of %s: %w", tc.Function.Name, err)
				}
				sb.WriteString("<|call|>")
				lastCall = tc.Function.Name
				continue
			}
			sb.WriteString("<|start|>assistant<|channel|>final<|message|>" + m.Content.Text + "<|end|>")
			lastCall = ""
		case "tool":
			if lastCall == "" {
				return "", false, fmt.Errorf("chat: a tool message with no assistant tool call before it")
			}
			sb.WriteString("<|start|>functions." + lastCall + " to=assistant<|channel|>commentary<|message|>")
			pyjson.String(&sb, m.Content.Text)
			sb.WriteString("<|end|>")
		case "user":
			sb.WriteString("<|start|>user<|message|>" + m.Content.Text + "<|end|>")
		}
	}
	sb.WriteString("<|start|>assistant")
	return sb.String(), false, nil
}

// harmonyArgs writes a tool call's arguments (a JSON string) as the template's tojson writes the
// object it parsed: json.dumps of Python's values.
func harmonyArgs(sb *strings.Builder, arguments string) error {
	a := strings.TrimSpace(arguments)
	if a == "" {
		a = "{}"
	}
	v, err := pyjson.Parse([]byte(a))
	if err != nil {
		return err
	}
	pyValue(v)
	v.Write(sb)
	return nil
}

// pyValue turns v's numbers into the text Python writes for the value it read (1e20 is 1e+20).
func pyValue(v *pyjson.Value) {
	if v.Kind == 'n' {
		v.Str = pyNumber(v.Str)
	}
	for _, f := range v.Fields {
		pyValue(f)
	}
}

// harmonyTools writes the template's render_tool_namespace("functions", tools).
func harmonyTools(sb *strings.Builder, tools []Tool) {
	sb.WriteString("## functions\n\nnamespace functions {\n\n")
	for _, t := range tools {
		f := t.raw
		sb.WriteString("// " + f.Get("description").Str + "\n")
		sb.WriteString("type " + t.Name() + " = ")
		params := f.Get("parameters")
		props := params.Get("properties")
		if !props.Truthy() {
			sb.WriteString("() => any;\n\n")
			continue
		}
		required := params.Get("required")
		sb.WriteString("(_: {\n")
		for i, name := range props.Keys {
			p := props.Fields[i]
			if d := p.Get("description"); d.Truthy() {
				sb.WriteString("// " + d.Str + "\n")
			}
			sb.WriteString(name)
			if !contains(required, name) {
				sb.WriteString("?")
			}
			sb.WriteString(": " + tsType(p))
			if def := p.Get("default"); def != nil {
				switch {
				case p.Get("enum").Truthy():
					sb.WriteString(", // default: " + def.Str)
				case p.Get("oneOf").Truthy():
					sb.WriteString("// default: " + def.Str)
				default:
					sb.WriteString(", // default: ")
					pyValue(def)
					def.Write(sb)
				}
			}
			sb.WriteString(",\n")
		}
		sb.WriteString("}) => any;\n\n")
	}
	sb.WriteString("} // namespace functions")
}

// contains says whether the list l (a JSON array, or nil) holds the string s.
func contains(l *pyjson.Value, s string) bool {
	if l == nil {
		return false
	}
	for _, f := range l.Fields {
		if f.Kind == '"' && f.Str == s {
			return true
		}
	}
	return false
}

// tsType is the template's render_typescript_type of a JSON schema.
func tsType(p *pyjson.Value) string {
	ty := p.Get("type")
	nullable := p.Get("nullable").Truthy()
	switch {
	case ty != nil && ty.Kind == '"' && ty.Str == "array":
		items := p.Get("items")
		out := "any[]"
		if items.Truthy() {
			it := items.Get("type")
			switch {
			case it != nil && it.Kind == '"' && it.Str == "string":
				out = "string[]"
			case it != nil && it.Kind == '"' && (it.Str == "number" || it.Str == "integer"):
				out = "number[]"
			case it != nil && it.Kind == '"' && it.Str == "boolean":
				out = "boolean[]"
			default:
				inner := tsType(items)
				if inner == "object | object" || len([]rune(inner)) > 50 {
					out = "any[]"
				} else {
					out = inner + "[]"
				}
			}
		}
		if nullable {
			out += " | null"
		}
		return out
	case ty != nil && ty.Kind == '[' && len(ty.Fields) > 0:
		var parts []string
		for _, f := range ty.Fields {
			parts = append(parts, f.Str)
		}
		return strings.Join(parts, " | ")
	case p.Get("oneOf").Truthy():
		// the template's has_object_variants never leaves its loop (a jinja2 set in a for), so
		// every variant is written in turn
		var sb strings.Builder
		vs := p.Get("oneOf").Fields
		for i, v := range vs {
			sb.WriteString(tsType(v))
			if d := v.Get("description"); d.Truthy() {
				sb.WriteString("// " + d.Str)
			}
			if i < len(vs)-1 {
				sb.WriteString(" | ")
			}
		}
		return sb.String()
	case ty != nil && ty.Kind == '"' && ty.Str == "string":
		if e := p.Get("enum"); e.Truthy() {
			var parts []string
			for _, f := range e.Fields {
				parts = append(parts, f.Str)
			}
			return `"` + strings.Join(parts, `" | "`) + `"`
		}
		if nullable {
			return "string | null"
		}
		return "string"
	case ty != nil && ty.Kind == '"' && (ty.Str == "number" || ty.Str == "integer"):
		return "number"
	case ty != nil && ty.Kind == '"' && ty.Str == "boolean":
		return "boolean"
	case ty != nil && ty.Kind == '"' && ty.Str == "object":
		props := p.Get("properties")
		if !props.Truthy() {
			return "object"
		}
		required := p.Get("required")
		var sb strings.Builder
		sb.WriteString("{\n")
		for i, name := range props.Keys {
			sb.WriteString(name)
			if !contains(required, name) {
				sb.WriteString("?")
			}
			// the template's "\n" and indent between ": " and the type (a {{ }} without "-")
			sb.WriteString(": \n                " + tsType(props.Fields[i]))
			if i < len(props.Keys)-1 {
				sb.WriteString(", ")
			}
		}
		sb.WriteString("}")
		return sb.String()
	}
	return "any"
}

// Parse takes apart an answer in harmony's messages: the analysis channel's text is the
// reasoning, the final channel's the content (a commentary message to no one too), and a
// commentary message to=functions.NAME a tool call whose arguments are its text. The text
// starts inside the first message's header (the prompt ends in <|start|>assistant).
func (gptOssFormat) Parse(text string, thinking bool, tools []Tool, final bool) Answer {
	var a Answer
	var reasoning, content []string
	for n, msg := range strings.Split(text, "<|start|>") {
		if n > 0 {
			// another message of the answer: its role (assistant) before the header
			msg = strings.TrimPrefix(msg, "assistant")
		}
		header, body, ok := strings.Cut(msg, "<|message|>")
		if !ok {
			if strings.Contains(header, "to=functions.") {
				a.Calling = true
			}
			continue
		}
		ended := false
		for _, end := range []string{"<|end|>", "<|return|>", "<|call|>"} {
			if i := strings.Index(body, end); i >= 0 {
				body, ended = body[:i], true
				break
			}
		}
		channel, to := harmonyHeader(header)
		switch {
		case strings.HasPrefix(to, "functions."):
			a.Calling = true
			if !ended && !final {
				continue
			}
			var tc ToolCall
			tc.ID = fmt.Sprintf("call_%d", len(a.ToolCalls))
			tc.Type = "function"
			tc.Function.Name = strings.TrimPrefix(to, "functions.")
			tc.Function.Arguments = strings.TrimSpace(body)
			a.ToolCalls = append(a.ToolCalls, tc)
		case channel == "analysis":
			if !ended {
				body = validPrefix(body)
				a.Thinking = !final
			}
			reasoning = append(reasoning, body)
		default:
			if !ended {
				body = validPrefix(body)
			}
			content = append(content, body)
		}
	}
	a.Reasoning = strings.Join(reasoning, "\n")
	a.Content = strings.Join(content, "\n")
	return a
}

// harmonyHeader returns a message header's channel and recipient (to=...), each up to a space
// or a special token.
func harmonyHeader(h string) (channel, to string) {
	word := func(s string) string {
		if i := strings.IndexAny(s, " <"); i >= 0 {
			return s[:i]
		}
		return s
	}
	if _, rest, ok := strings.Cut(h, "<|channel|>"); ok {
		channel = word(rest)
	}
	if _, rest, ok := strings.Cut(h, "to="); ok {
		to = word(rest)
	}
	return channel, to
}
