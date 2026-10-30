package chat

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/pyjson"
)

// Gemma4 is the chat format of google/gemma-4-E2B-it: its chat_template.jinja ("Gemma 4
// Canonical Chat Template", 2026-07-09) in Go. Turns are <|turn>ROLE ... <turn|>, the assistant's
// role is model, thinking is turned on by <|think|> at the top of the system turn and comes back
// in <|channel>thought ... <channel|>, and tools are declared, called and answered in the
// template's own notation (<|tool>declaration:NAME{...}<tool|>, <|tool_call>call:NAME{...}
// <tool_call|>, <|tool_response>response:NAME{...}<tool_response|>), strings quoted with <|"|>.
var Gemma4 Template = gemma4{}

type gemma4 struct{}

// Stops are the tokens that end an answer (generation_config.json's eos_token_id): the end of a
// turn, and the start of a tool response, which follows a tool call.
func (gemma4) Stops() []string { return []string{"<turn|>", "<eos>", "<|tool_response>"} }

const gq = `<|"|>` // the template's string quote

func (gemma4) Render(msgs []Message, tools []Tool, o Options) (string, bool, error) {
	var sb strings.Builder
	sb.WriteString("<bos>")
	think := !o.NoThink
	prevType := "" // ns.prev_message_type
	loop := msgs
	if think || len(tools) > 0 || len(msgs) > 0 && (msgs[0].Role == "system" || msgs[0].Role == "developer") {
		sb.WriteString("<|turn>system\n")
		if think {
			sb.WriteString("<|think|>\n")
			prevType = "think"
		}
		if len(msgs) > 0 && (msgs[0].Role == "system" || msgs[0].Role == "developer") {
			if c := msgs[0].Content; c.Parts {
				for _, p := range c.parts {
					sb.WriteString(pyTrim(p) + " ")
				}
			} else {
				sb.WriteString(pyTrim(c.Text))
			}
			loop = msgs[1:]
		}
		for _, t := range tools {
			sb.WriteString("<|tool>" + pyTrim(declaration(t.raw)) + "<tool|>")
			prevType = "tool"
		}
		sb.WriteString("<turn|>\n")
	}
	lastUser := -1
	for i, m := range loop {
		if m.Role == "user" {
			lastUser = i
		}
	}
	prevRole := "" // ns.prev_non_tool_role
	for i, m := range loop {
		if m.Role == "tool" {
			continue
		}
		prevType = ""
		role := m.Role
		if role == "assistant" {
			role = "model"
		}
		if !(role == "model" && prevRole == "assistant") {
			sb.WriteString("<|turn>" + role + "\n")
		}
		if m.Reasoning != nil && *m.Reasoning != "" && (i > lastUser || o.KeepThinking && len(m.ToolCalls) > 0) {
			sb.WriteString("<|channel>thought\n" + *m.Reasoning + "\n<channel|>")
		}
		for _, tc := range m.ToolCalls {
			sb.WriteString("<|tool_call>call:" + tc.Function.Name + "{")
			if a := strings.TrimSpace(tc.Function.Arguments); a != "" && a != "null" {
				args, err := pyjson.Parse([]byte(a))
				if err != nil {
					return "", false, fmt.Errorf("chat: arguments of %s: %w", tc.Function.Name, err)
				}
				if args.Kind != '{' {
					return "", false, fmt.Errorf("chat: arguments of %s are not a JSON object", tc.Function.Name)
				}
				for j, k := range dictsort(args) {
					if j > 0 {
						sb.WriteByte(',')
					}
					sb.WriteString(k + ":")
					argument(&sb, args.Get(k), false)
				}
			}
			sb.WriteString("}<tool_call|>")
			prevType = "tool_call"
		}
		responded := false // ns_tr_out.flag
		if len(m.ToolCalls) > 0 {
			// the tool messages that follow, as the responses
			for _, f := range loop[i+1:] {
				if f.Role != "tool" {
					break
				}
				name := f.Name
				if name == "" {
					name = "unknown"
				}
				for _, tc := range m.ToolCalls {
					if tc.ID == f.ToolCallID {
						name = tc.Function.Name
					}
				}
				sb.WriteString("<|tool_response>response:" + name + "{value:")
				argument(&sb, &pyjson.Value{Kind: '"', Str: f.Content.Text}, false)
				sb.WriteString("}<tool_response|>")
				responded, prevType = true, "tool_response"
			}
		}
		var content string
		texts := []string{m.Content.Text}
		if m.Content.Parts {
			texts = m.Content.parts
		}
		for _, t := range texts {
			if role == "model" {
				content += stripThinking(t)
			} else {
				content += pyTrim(t)
			}
		}
		sb.WriteString(content)
		next := ""
		for _, f := range loop[i+1:] {
			if f.Role != "tool" {
				next = f.Role
				break
			}
		}
		continues := role == "model" && next == "assistant" && (len(m.ToolCalls) == 0 || responded)
		switch {
		case prevType == "tool_call" && !responded:
			sb.WriteString("<|tool_response>")
		case continues:
		case !(responded && pyTrim(content) == "" && next == ""):
			sb.WriteString("<turn|>\n")
		}
		prevRole = m.Role
	}
	thinking := false
	switch {
	case prevType != "tool_response" && prevType != "tool_call":
		sb.WriteString("<|turn>model\n")
	case prevType == "tool_response" && think:
		sb.WriteString("<|channel>thought\n")
		thinking = true
	}
	return sb.String(), thinking, nil
}

// stripThinking is the template's strip_thinking: the text without its <|channel>...<channel|>
// parts, trimmed.
func stripThinking(text string) string {
	var out strings.Builder
	for _, part := range strings.Split(text, "<channel|>") {
		if before, _, ok := strings.Cut(part, "<|channel>"); ok {
			out.WriteString(before)
		} else {
			out.WriteString(part)
		}
	}
	return pyTrim(out.String())
}

// pyTrim is Jinja's trim (Python's str.strip()).
func pyTrim(s string) string { return strings.TrimSpace(s) }

// dictsort is the keys of an object in the order of Jinja's dictsort: case-insensitive.
func dictsort(v *pyjson.Value) []string {
	keys := slices.Clone(v.Keys)
	slices.SortStableFunc(keys, func(a, b string) int { return strings.Compare(strings.ToLower(a), strings.ToLower(b)) })
	return keys
}

// argument is the template's format_argument.
func argument(sb *strings.Builder, v *pyjson.Value, escapeKeys bool) {
	switch {
	case v == nil || v.Kind == '0':
		sb.WriteString("null")
	case v.Kind == '"':
		sb.WriteString(gq + v.Str + gq)
	case v.Kind == 't':
		sb.WriteString("true")
	case v.Kind == 'f':
		sb.WriteString("false")
	case v.Kind == '{':
		sb.WriteByte('{')
		for i, k := range dictsort(v) {
			if i > 0 {
				sb.WriteByte(',')
			}
			if escapeKeys {
				sb.WriteString(gq + k + gq)
			} else {
				sb.WriteString(k)
			}
			sb.WriteByte(':')
			argument(sb, v.Get(k), escapeKeys)
		}
		sb.WriteByte('}')
	case v.Kind == '[':
		sb.WriteByte('[')
		for i, f := range v.Fields {
			if i > 0 {
				sb.WriteByte(',')
			}
			argument(sb, f, escapeKeys)
		}
		sb.WriteByte(']')
	default:
		sb.WriteString(pyNumber(v.Str))
	}
}

// pyNumber writes a JSON number as Python's str() of what json.loads makes of it: an integer as
// it is, a float in repr's shortest form ("1000.0", "1e-05", "1e+20").
func pyNumber(s string) string {
	if !strings.ContainsAny(s, ".eE") {
		return s
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return s
	}
	if a := math.Abs(f); a != 0 && (a < 1e-4 || a >= 1e16) {
		return strconv.FormatFloat(f, 'e', -1, 64)
	}
	r := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.Contains(r, ".") {
		r += ".0"
	}
	return r
}

// truthy is Jinja's truth of a value (nil for a missing key).
func truthy(v *pyjson.Value) bool {
	switch {
	case v == nil:
		return false
	case v.Kind == '"':
		return v.Str != ""
	case v.Kind == '{', v.Kind == '[':
		return len(v.Fields) > 0
	case v.Kind == 'n':
		f, _ := strconv.ParseFloat(v.Str, 64)
		return f != 0
	}
	return v.Kind == 't'
}

// upper is Jinja's upper filter: of a string, or of Python's str() of anything else.
func upper(v *pyjson.Value) string {
	switch {
	case v == nil:
		return ""
	case v.Kind == '"':
		return strings.ToUpper(v.Str)
	}
	return strings.ToUpper(pyStr(v))
}

// pyStr is Python's str() of a JSON value (a list of strings as ['a', 'b']).
func pyStr(v *pyjson.Value) string {
	switch v.Kind {
	case '"':
		return v.Str
	case '[':
		var items []string
		for _, f := range v.Fields {
			if f.Kind == '"' {
				items = append(items, "'"+f.Str+"'")
			} else {
				items = append(items, pyStr(f))
			}
		}
		return "[" + strings.Join(items, ", ") + "]"
	case '{':
		var items []string
		for i, k := range v.Keys {
			items = append(items, "'"+k+"': "+pyStr(v.Fields[i]))
		}
		return "{" + strings.Join(items, ", ") + "}"
	case 'n':
		return pyNumber(v.Str)
	case 't':
		return "True"
	case 'f':
		return "False"
	}
	return "None"
}

// declaration is the template's format_function_declaration of a tool's function object.
func declaration(f *pyjson.Value) string {
	var sb strings.Builder
	sb.WriteString("declaration:")
	if n := f.Get("name"); n != nil {
		sb.WriteString(pyStr(n))
	}
	sb.WriteString("{description:" + gq)
	if d := f.Get("description"); d != nil {
		sb.WriteString(pyStr(d))
	}
	sb.WriteString(gq)
	if params := f.Get("parameters"); truthy(params) {
		sb.WriteString(",parameters:{")
		if props := params.Get("properties"); truthy(props) {
			sb.WriteString("properties:{")
			parameters(&sb, props, false)
			sb.WriteString("},")
		}
		if req := params.Get("required"); truthy(req) {
			sb.WriteString("required:[")
			for i, r := range req.Fields {
				if i > 0 {
					sb.WriteByte(',')
				}
				sb.WriteString(gq + pyStr(r) + gq)
			}
			sb.WriteString("],")
		}
		if ty := params.Get("type"); truthy(ty) {
			sb.WriteString("type:" + gq + upper(ty) + gq + "}")
		}
	}
	if resp := f.Get("response"); resp != nil {
		sb.WriteString(",response:{")
		if d := resp.Get("description"); truthy(d) {
			sb.WriteString("description:" + gq + pyStr(d) + gq + ",")
		}
		if upper(resp.Get("type")) == "OBJECT" {
			sb.WriteString("type:" + gq + upper(resp.Get("type")) + gq + "}")
		}
	}
	sb.WriteByte('}')
	return sb.String()
}

var standardKeys = []string{"description", "type", "properties", "required", "nullable"}

// parameters is the template's format_parameters over an object's properties; with filterKeys it
// leaves out the keys of a schema itself (an object without properties lists its other keys).
func parameters(sb *strings.Builder, props *pyjson.Value, filterKeys bool) {
	if props == nil || props.Kind != '{' {
		return
	}
	first := true
	for _, key := range dictsort(props) {
		if filterKeys && slices.Contains(standardKeys, key) {
			continue
		}
		v := props.Get(key)
		if !first {
			sb.WriteByte(',')
		}
		first = false
		sb.WriteString(key + ":{")
		comma := false
		sep := func() {
			if comma {
				sb.WriteByte(',')
			}
			comma = true
		}
		if d := v.Get("description"); truthy(d) {
			sb.WriteString("description:" + gq + pyStr(d) + gq)
			comma = true
		}
		switch upper(v.Get("type")) {
		case "STRING":
			if e := v.Get("enum"); truthy(e) {
				sep()
				sb.WriteString("enum:")
				argument(sb, e, true)
			}
		case "ARRAY":
			if items := v.Get("items"); items != nil && items.Kind == '{' && truthy(items) {
				sep()
				sb.WriteString("items:{")
				firstItem := true
				for _, ik := range dictsort(items) {
					iv := items.Get(ik)
					if iv.Kind == '0' {
						continue
					}
					if !firstItem {
						sb.WriteByte(',')
					}
					firstItem = false
					switch ik {
					case "properties":
						sb.WriteString("properties:{")
						if iv.Kind == '{' {
							parameters(sb, iv, false)
						}
						sb.WriteString("}")
					case "required":
						sb.WriteString("required:[")
						for i, r := range iv.Fields {
							if i > 0 {
								sb.WriteByte(',')
							}
							sb.WriteString(gq + pyStr(r) + gq)
						}
						sb.WriteString("]")
					case "type":
						sb.WriteString("type:")
						if iv.Kind == '"' {
							argument(sb, &pyjson.Value{Kind: '"', Str: strings.ToUpper(iv.Str)}, true)
						} else {
							up := &pyjson.Value{Kind: '['}
							for _, f := range iv.Fields {
								up.Fields = append(up.Fields, &pyjson.Value{Kind: '"', Str: upper(f)})
							}
							argument(sb, up, true)
						}
					default:
						sb.WriteString(ik + ":")
						argument(sb, iv, true)
					}
				}
				sb.WriteString("}")
			}
		}
		if truthy(v.Get("nullable")) {
			sep()
			sb.WriteString("nullable:true")
		}
		if upper(v.Get("type")) == "OBJECT" {
			if p := v.Get("properties"); p != nil && p.Kind == '{' {
				sep()
				sb.WriteString("properties:{")
				parameters(sb, p, false)
				sb.WriteString("}")
			} else {
				sep()
				sb.WriteString("properties:{")
				parameters(sb, v, true)
				sb.WriteString("}")
			}
			if req := v.Get("required"); truthy(req) {
				sep()
				sb.WriteString("required:[")
				for i, r := range req.Fields {
					if i > 0 {
						sb.WriteByte(',')
					}
					sb.WriteString(gq + pyStr(r) + gq)
				}
				sb.WriteString("]")
			}
		}
		sep()
		sb.WriteString("type:" + gq + upper(v.Get("type")) + gq + "}")
	}
}

var (
	gemmaCallRe = regexp.MustCompile(`(?s)<\|tool_call>call:([^{]+)\{(.*?)\}<tool_call\|>`)
)

// Parse takes apart the text generated after a prompt from Render. thinking says that the prompt
// ended inside a thought channel.
func (gemma4) Parse(text string, thinking bool, tools []Tool, final bool) Answer {
	var a Answer
	for _, s := range []string{"<turn|>", "<eos>", "<|tool_response>"} {
		text = strings.TrimSuffix(text, s)
	}
	if !thinking {
		if rest, ok := strings.CutPrefix(strings.TrimLeft(text, " \t\r\n"), "<|channel>"); ok {
			text, thinking = strings.TrimPrefix(rest, "thought\n"), true
		}
	}
	if thinking {
		r, rest, ok := strings.Cut(text, "<channel|>")
		if !ok {
			a.Reasoning, a.Thinking = validPrefix(strings.TrimPrefix(r, "thought\n")), true
			return a
		}
		a.Reasoning, text = strings.TrimSuffix(r, "\n"), strings.TrimLeft(rest, " \t\r\n")
	}
	content, calls, ok := strings.Cut(text, "<|tool_call>")
	if !ok {
		if final {
			a.Content = content
		} else {
			a.Content = validPrefix(content)
		}
		return a
	}
	a.Content, a.Calling = strings.TrimRight(content, " \t\r\n"), true
	for i, m := range gemmaCallRe.FindAllStringSubmatch("<|tool_call>"+calls, -1) {
		var tc ToolCall
		tc.ID = fmt.Sprintf("call_%d", i)
		tc.Type = "function"
		tc.Function.Name = strings.TrimSpace(m[1])
		args, err := readArguments("{" + m[2] + "}")
		if err != nil {
			args = "{}"
		}
		tc.Function.Arguments = args
		a.ToolCalls = append(a.ToolCalls, tc)
	}
	return a
}

// readArguments turns the template's notation of a call's arguments ({key:value,...}, strings in
// <|"|>, keys bare) into JSON, keeping the order.
func readArguments(s string) (string, error) {
	p := &notation{s: s}
	var sb strings.Builder
	if err := p.value(&sb); err != nil {
		return "", err
	}
	if p.i != len(p.s) {
		return "", fmt.Errorf("chat: trailing %q after the arguments", p.s[p.i:])
	}
	return sb.String(), nil
}

type notation struct {
	s string
	i int
}

func (p *notation) value(sb *strings.Builder) error {
	switch {
	case strings.HasPrefix(p.s[p.i:], gq):
		end := strings.Index(p.s[p.i+len(gq):], gq)
		if end < 0 {
			return fmt.Errorf("chat: a string without its end")
		}
		str := p.s[p.i+len(gq) : p.i+len(gq)+end]
		p.i += 2*len(gq) + end
		b, _ := jsonString(str)
		sb.WriteString(b)
	case p.i < len(p.s) && p.s[p.i] == '{':
		p.i++
		sb.WriteByte('{')
		for n := 0; p.i < len(p.s) && p.s[p.i] != '}'; n++ {
			if n > 0 {
				if p.s[p.i] != ',' {
					return fmt.Errorf("chat: expected , at %d", p.i)
				}
				p.i++
				sb.WriteByte(',')
			}
			var key string
			if strings.HasPrefix(p.s[p.i:], gq) {
				end := strings.Index(p.s[p.i+len(gq):], gq)
				if end < 0 {
					return fmt.Errorf("chat: a key without its end")
				}
				key = p.s[p.i+len(gq) : p.i+len(gq)+end]
				p.i += 2*len(gq) + end
			} else {
				colon := strings.IndexByte(p.s[p.i:], ':')
				if colon < 0 {
					return fmt.Errorf("chat: a key without a value")
				}
				key = p.s[p.i : p.i+colon]
				p.i += colon
			}
			if p.i >= len(p.s) || p.s[p.i] != ':' {
				return fmt.Errorf("chat: expected : after %q", key)
			}
			p.i++
			b, _ := jsonString(key)
			sb.WriteString(b + ":")
			if err := p.value(sb); err != nil {
				return err
			}
		}
		if p.i >= len(p.s) {
			return fmt.Errorf("chat: an object without its end")
		}
		p.i++
		sb.WriteByte('}')
	case p.i < len(p.s) && p.s[p.i] == '[':
		p.i++
		sb.WriteByte('[')
		for n := 0; p.i < len(p.s) && p.s[p.i] != ']'; n++ {
			if n > 0 {
				if p.s[p.i] != ',' {
					return fmt.Errorf("chat: expected , at %d", p.i)
				}
				p.i++
				sb.WriteByte(',')
			}
			if err := p.value(sb); err != nil {
				return err
			}
		}
		if p.i >= len(p.s) {
			return fmt.Errorf("chat: a list without its end")
		}
		p.i++
		sb.WriteByte(']')
	default:
		// a number, true, false or null, up to the next delimiter
		end := p.i
		for end < len(p.s) && !strings.ContainsRune(",}]", rune(p.s[end])) {
			end++
		}
		tok := strings.TrimSpace(p.s[p.i:end])
		switch tok {
		case "true", "false", "null":
		default:
			if _, err := strconv.ParseFloat(tok, 64); err != nil {
				b, _ := jsonString(tok) // a bare word: keep it as a string
				tok = b
			}
		}
		sb.WriteString(tok)
		p.i = end
	}
	return nil
}

// jsonString writes s as a JSON string, leaving <, > and & as they are.
func jsonString(s string) (string, error) {
	var b bytes.Buffer
	e := json.NewEncoder(&b)
	e.SetEscapeHTML(false)
	if err := e.Encode(s); err != nil {
		return "", err
	}
	return strings.TrimSuffix(b.String(), "\n"), nil
}
