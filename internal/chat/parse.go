package chat

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"
)

// Answer is a generated assistant message taken apart.
type Answer struct {
	Reasoning string
	Content   string
	ToolCalls []ToolCall
	// Thinking is true while the text is still inside <think>.
	Thinking bool
	// Calling is true once a tool call has started: Content does not grow after it.
	Calling bool
}

var (
	callRe  = regexp.MustCompile(`(?s)<tool_call>\s*<function=([^\n>]+)>(.*?)</function>\s*</tool_call>`)
	paramRe = regexp.MustCompile(`(?s)<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>`)
)

const callTag = "<tool_call>"

// Parse takes apart the text generated after a prompt from Render (with the special tokens).
// thinking says that the prompt ended inside <think>. Unless final, the text may be cut
// anywhere: a tool call that has not ended yet is not in ToolCalls, and Content leaves out a
// tail that may be the start of one, and an incomplete UTF-8 character.
func Parse(text string, thinking bool, tools []Tool, final bool) Answer {
	var a Answer
	text = strings.TrimSuffix(text, "<|im_end|>")
	r, rest, ok := strings.Cut(text, "</think>")
	switch {
	case ok:
		// also when the prompt closed the thinking: the model may think anyway
		a.Reasoning, text = r, strings.TrimLeft(rest, " \t\r\n")
	case thinking:
		a.Reasoning, a.Thinking = validPrefix(r), true
		return a
	}
	content, calls, ok := strings.Cut(text, callTag)
	if !ok {
		if final {
			a.Content = content
			return a
		}
		// hold back a tail that may become <tool_call>
		for n := min(len(callTag)-1, len(content)); n > 0; n-- {
			if strings.HasSuffix(content, callTag[:n]) {
				content = content[:len(content)-n]
				break
			}
		}
		a.Content = validPrefix(content)
		return a
	}
	a.Content, a.Calling = strings.TrimRight(content, " \t\r\n"), true
	for i, m := range callRe.FindAllStringSubmatch(callTag+calls, -1) {
		var tc ToolCall
		tc.ID = fmt.Sprintf("call_%d", i)
		tc.Type = "function"
		tc.Function.Name = strings.TrimSpace(m[1])
		var tool *Tool
		for j := range tools {
			if tools[j].Name() == tc.Function.Name {
				tool = &tools[j]
			}
		}
		args := map[string]any{}
		var keys []string
		for _, p := range paramRe.FindAllStringSubmatch(m[2], -1) {
			k := strings.TrimSpace(p[1])
			keys = append(keys, k)
			args[k] = argValue(tool, k, p[2])
		}
		tc.Function.Arguments = objectJSON(keys, args)
		a.ToolCalls = append(a.ToolCalls, tc)
	}
	return a
}

// argValue turns a parameter's text into a JSON value: a string when the tool says so or the
// text is not JSON, else the JSON it reads as.
func argValue(t *Tool, k, s string) any {
	ty := ""
	if t != nil {
		ty = t.paramType(k)
	}
	if ty == "string" {
		return s
	}
	var v any
	if err := json.Unmarshal([]byte(s), &v); err == nil {
		return v
	}
	return s
}

// objectJSON writes an object with the keys in order.
func objectJSON(keys []string, m map[string]any) string {
	var sb strings.Builder
	sb.WriteByte('{')
	for i, k := range keys {
		if i > 0 {
			sb.WriteByte(',')
		}
		kb, _ := json.Marshal(k)
		vb, _ := json.Marshal(m[k])
		sb.Write(kb)
		sb.WriteByte(':')
		sb.Write(vb)
	}
	sb.WriteByte('}')
	return sb.String()
}

// validPrefix drops an incomplete UTF-8 character at the end.
func validPrefix(s string) string {
	for i := 0; i < utf8.UTFMax && len(s) > 0; i++ {
		if utf8.ValidString(s) {
			return s
		}
		s = s[:len(s)-1]
	}
	return s
}

// Stream follows an answer as it grows, for sending it in pieces.
type Stream struct{ reasoning, content string }

// Next takes the reasoning and the content parsed so far and returns what to add to what was
// sent. Both grow at the end, except when the model closes a thinking it was not asked for: the
// content sent until then was its thinking, and the content starts over after a blank line.
func (s *Stream) Next(reasoning, content string) (dr, dc string) {
	if strings.HasPrefix(reasoning, s.reasoning) {
		dr, s.reasoning = reasoning[len(s.reasoning):], reasoning
	}
	switch {
	case strings.HasPrefix(s.content, content): // nothing new, or less held back
	case strings.HasPrefix(content, s.content):
		dc, s.content = content[len(s.content):], content
	default:
		if s.content != "" {
			dc = "\n\n"
		}
		dc, s.content = dc+content, content
	}
	return dr, dc
}
