package chat

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/shibukawa/bonsaiai/internal/pyjson"
)

// LFM2 is the chat format of LiquidAI's LFM2 models (LFM2-8B-A1B): ChatML turns without thinking,
// the tools listed in the system turn as JSON between <|tool_list_start|> and <|tool_list_end|>,
// a call as Python, <|tool_call_start|>[get_weather(city='Tokyo', days=2)]<|tool_call_end|>, and
// a tool's response between <|tool_response_start|> and <|tool_response_end|>. Render writes what
// its chat_template.jinja does (testdata/lfm2_template.json, tools/ref/template_golden.py).
var LFM2 Template = lfm2Format{}

type lfm2Format struct{}

const (
	lfm2CallStart = "<|tool_call_start|>"
	lfm2CallEnd   = "<|tool_call_end|>"
)

func (lfm2Format) Render(msgs []Message, tools []Tool, o Options) (string, bool, error) {
	var sb strings.Builder
	sb.WriteString("<|startoftext|>")
	system := ""
	if len(msgs) > 0 && msgs[0].Role == "system" {
		system = msgs[0].Content.Text
		msgs = msgs[1:]
	}
	if len(tools) > 0 {
		if system != "" {
			system += "\n"
		}
		system += "List of tools: <|tool_list_start|>["
		for i, t := range tools {
			if i > 0 {
				system += ", "
			}
			system += `{"type": "function", "function": ` + t.raw.String() + "}"
		}
		system += "]<|tool_list_end|>"
	}
	if system != "" {
		sb.WriteString("<|im_start|>system\n" + system + "<|im_end|>\n")
	}
	for _, m := range msgs {
		sb.WriteString("<|im_start|>" + m.Role + "\n")
		content := m.Content.Text
		if m.Role == "tool" {
			content = "<|tool_response_start|>" + content + "<|tool_response_end|>"
		}
		if m.Role == "assistant" && len(m.ToolCalls) > 0 {
			sb.WriteString(content)
			sb.WriteString(lfm2CallStart + "[")
			for i, tc := range m.ToolCalls {
				if i > 0 {
					sb.WriteString(", ")
				}
				if err := lfm2Call(&sb, tc); err != nil {
					return "", false, err
				}
			}
			sb.WriteString("]" + lfm2CallEnd + "<|im_end|>\n")
			continue
		}
		sb.WriteString(content + "<|im_end|>\n")
	}
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String(), false, nil
}

// lfm2Call writes a call as the template's render_tool_calls does: name(key=value, ...), a
// string in single quotes, an object or array as JSON, anything else as Python's str.
func lfm2Call(sb *strings.Builder, tc ToolCall) error {
	sb.WriteString(tc.Function.Name + "(")
	args := strings.TrimSpace(tc.Function.Arguments)
	if args != "" {
		v, err := pyjson.Parse([]byte(args))
		if err != nil {
			return fmt.Errorf("chat: the arguments of %s: %w", tc.Function.Name, err)
		}
		if v.Kind != '{' {
			return fmt.Errorf("chat: the arguments of %s are not an object", tc.Function.Name)
		}
		for i, k := range v.Keys {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(k + "=")
			f := v.Fields[i]
			switch f.Kind {
			case '"':
				r := strings.NewReplacer(`\`, `\\`, `'`, `\'`, "\n", `\n`, "\r", `\r`)
				sb.WriteString("'" + r.Replace(f.Str) + "'")
			case '{', '[':
				pyJSON(sb, f)
			case 't':
				sb.WriteString("True")
			case 'f':
				sb.WriteString("False")
			case 'n':
				sb.WriteString(pyNumber(f.Str))
			default:
				sb.WriteString("None")
			}
		}
	}
	sb.WriteString(")")
	return nil
}

// pyJSON writes v as json.dumps does of what Python's json.loads read: as pyjson, the numbers
// as Python writes them (1e20 read as a float is 1e+20).
func pyJSON(sb *strings.Builder, v *pyjson.Value) {
	switch v.Kind {
	case '{', '[':
		open, close := byte('['), byte(']')
		if v.Kind == '{' {
			open, close = '{', '}'
		}
		sb.WriteByte(open)
		for i, f := range v.Fields {
			if i > 0 {
				sb.WriteString(", ")
			}
			if v.Kind == '{' {
				pyjson.String(sb, v.Keys[i])
				sb.WriteString(": ")
			}
			pyJSON(sb, f)
		}
		sb.WriteByte(close)
	case 'n':
		sb.WriteString(pyNumber(v.Str))
	default:
		v.Write(sb)
	}
}

func (lfm2Format) Parse(text string, thinking bool, tools []Tool, final bool) Answer {
	var a Answer
	text = strings.TrimSuffix(strings.TrimSuffix(text, "<|im_end|>"), "<|endoftext|>")
	content, calls, ok := strings.Cut(text, lfm2CallStart)
	if !ok {
		if !final { // hold back a tail that may become <|tool_call_start|>
			for n := min(len(lfm2CallStart)-1, len(content)); n > 0; n-- {
				if strings.HasSuffix(content, lfm2CallStart[:n]) {
					content = content[:len(content)-n]
					break
				}
			}
		}
		a.Content = validPrefix(content)
		return a
	}
	a.Content, a.Calling = strings.TrimRight(content, " \t\r\n"), true
	for {
		body, rest, done := strings.Cut(calls, lfm2CallEnd)
		if !done {
			break // the call goes on
		}
		for _, c := range parsePyCalls(body) {
			c.ID = fmt.Sprintf("call_%d", len(a.ToolCalls))
			a.ToolCalls = append(a.ToolCalls, c)
		}
		// another list of calls may follow
		if _, more, ok := strings.Cut(rest, lfm2CallStart); ok {
			calls = more
			continue
		}
		break
	}
	return a
}

func (lfm2Format) Stops() []string { return []string{"<|im_end|>", "<|endoftext|>"} }

func (lfm2Format) Think() (open, close, end string) { return "", "", "" }

// parsePyCalls reads a Python list of calls, [f(a=1, b='x'), g()], each argument's value a
// Python literal (strings in either quotes, numbers, True, False, None, lists, tuples, dicts)
// written as JSON. A call it cannot read is left out.
func parsePyCalls(s string) []ToolCall {
	p := &pyParser{s: s}
	p.space()
	bracket := p.eat('[')
	var out []ToolCall
	for {
		p.space()
		if p.done() || bracket && p.eat(']') {
			break
		}
		start := p.i
		tc, ok := p.call()
		if !ok {
			// skip to the next call
			p.i = start
			depth := 0
			for !p.done() {
				c := p.s[p.i]
				p.i++
				if c == '(' {
					depth++
				}
				if c == ')' {
					depth--
					if depth <= 0 {
						break
					}
				}
			}
		} else {
			out = append(out, tc)
		}
		p.space()
		p.eat(',')
	}
	return out
}

type pyParser struct {
	s string
	i int
}

func (p *pyParser) done() bool { return p.i >= len(p.s) }

func (p *pyParser) space() {
	for !p.done() && strings.IndexByte(" \t\r\n", p.s[p.i]) >= 0 {
		p.i++
	}
}

func (p *pyParser) eat(c byte) bool {
	if !p.done() && p.s[p.i] == c {
		p.i++
		return true
	}
	return false
}

func (p *pyParser) ident() string {
	start := p.i
	for !p.done() {
		r, n := utf8.DecodeRuneInString(p.s[p.i:])
		if !(unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '.' || r == '-') {
			break
		}
		p.i += n
	}
	return p.s[start:p.i]
}

func (p *pyParser) call() (ToolCall, bool) {
	var tc ToolCall
	tc.Type = "function"
	name := p.ident()
	p.space()
	if name == "" || !p.eat('(') {
		return tc, false
	}
	tc.Function.Name = name
	var sb strings.Builder
	sb.WriteByte('{')
	n := 0
	for {
		p.space()
		if p.eat(')') {
			break
		}
		if p.done() {
			return tc, false
		}
		key := p.ident()
		p.space()
		if key == "" || !p.eat('=') {
			return tc, false
		}
		p.space()
		var vb strings.Builder
		if !p.value(&vb) {
			return tc, false
		}
		if n > 0 {
			sb.WriteByte(',')
		}
		kb, _ := json.Marshal(key)
		sb.Write(kb)
		sb.WriteByte(':')
		sb.WriteString(vb.String())
		n++
		p.space()
		p.eat(',')
	}
	sb.WriteByte('}')
	tc.Function.Arguments = sb.String()
	return tc, true
}

// value reads a Python literal and writes it as JSON.
func (p *pyParser) value(sb *strings.Builder) bool {
	p.space()
	if p.done() {
		return false
	}
	switch c := p.s[p.i]; {
	case c == '\'' || c == '"':
		s, ok := p.str()
		if !ok {
			return false
		}
		b, _ := json.Marshal(s)
		sb.Write(b)
		return true
	case c == '[' || c == '(':
		closer := byte(']')
		if c == '(' {
			closer = ')'
		}
		p.i++
		sb.WriteByte('[')
		for n := 0; ; n++ {
			p.space()
			if p.eat(closer) {
				break
			}
			if n > 0 {
				sb.WriteByte(',')
			}
			if !p.value(sb) {
				return false
			}
			p.space()
			p.eat(',')
		}
		sb.WriteByte(']')
		return true
	case c == '{':
		p.i++
		sb.WriteByte('{')
		for n := 0; ; n++ {
			p.space()
			if p.eat('}') {
				break
			}
			if n > 0 {
				sb.WriteByte(',')
			}
			var kb strings.Builder
			if !p.value(&kb) {
				return false
			}
			k := kb.String()
			if !strings.HasPrefix(k, `"`) { // a key that is not a string, as JSON writes it
				b, _ := json.Marshal(k)
				k = string(b)
			}
			sb.WriteString(k)
			p.space()
			if !p.eat(':') {
				return false
			}
			sb.WriteByte(':')
			if !p.value(sb) {
				return false
			}
			p.space()
			p.eat(',')
		}
		sb.WriteByte('}')
		return true
	}
	if c := p.s[p.i]; c == '-' || c == '+' || c == '.' || c >= '0' && c <= '9' {
		start := p.i
		for !p.done() && strings.IndexByte("0123456789+-.eE_", p.s[p.i]) >= 0 {
			p.i++
		}
		num := strings.ReplaceAll(strings.TrimPrefix(p.s[start:p.i], "+"), "_", "")
		if _, err := strconv.ParseFloat(num, 64); err != nil {
			return false
		}
		sb.WriteString(num)
		return true
	}
	word := p.ident()
	switch word {
	case "True", "true":
		sb.WriteString("true")
		return true
	case "False", "false":
		sb.WriteString("false")
		return true
	case "None", "null":
		sb.WriteString("null")
		return true
	}
	return false
}

// str reads a Python string literal in single or double quotes.
func (p *pyParser) str() (string, bool) {
	q := p.s[p.i]
	p.i++
	var sb strings.Builder
	for !p.done() {
		c := p.s[p.i]
		p.i++
		switch {
		case c == q:
			return sb.String(), true
		case c == '\\' && !p.done():
			e := p.s[p.i]
			p.i++
			switch e {
			case 'n':
				sb.WriteByte('\n')
			case 't':
				sb.WriteByte('\t')
			case 'r':
				sb.WriteByte('\r')
			case '0':
				sb.WriteByte(0)
			case 'u':
				if p.i+4 <= len(p.s) {
					if r, err := strconv.ParseUint(p.s[p.i:p.i+4], 16, 32); err == nil {
						sb.WriteRune(rune(r))
						p.i += 4
						continue
					}
				}
				sb.WriteString(`\u`)
			default:
				sb.WriteByte(e)
			}
		default:
			sb.WriteByte(c)
		}
	}
	return "", false
}
