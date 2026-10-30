package systemone

import (
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/pyjson"
)

// Prompt writes the d1 prompt of one question, as the systemone template llama.cpp writes for
// LFM3 d1 renders it: the d1 prompt of LFM2 (prompt.py of the model repository) and the empty
// thought that the LFM3 template writes before an answer. labels are the options' codes
// (Labels). escape is applied to every text taken from the request (Escaper).
func Prompt(state *pyjson.Value, q Question, labels []string, escape func(string) string) string {
	var sb strings.Builder
	text := func(v *pyjson.Value, indent int) {
		v = escapeValue(v, escape)
		switch {
		case v.Kind == '"':
			sb.WriteString(v.Str)
		case indent > 0:
			v.WriteIndent(&sb, indent)
		default:
			v.Write(&sb)
		}
	}
	sb.WriteString("<|startoftext|><|im_start|>user\n")
	if !state.IsNull() {
		text(state, 2)
		sb.WriteString("\n\n\nQUESTION:\n")
	}
	text(q.Instructions, 0)
	switch q.Type {
	case "choice":
		sb.WriteString("\n\nOptions:\n")
		for i, o := range q.Options {
			sb.WriteString(labels[i] + " ")
			if o.Description.Truthy() {
				text(o.Description, 0)
			} else {
				sb.WriteString(strings.ReplaceAll(escape(o.Key), "_", " "))
			}
			if i+1 < len(q.Options) {
				sb.WriteByte('\n')
			}
		}
		sb.WriteString("\n\nReply with the option code only.")
	case "noul":
		criteria := false
		for _, o := range q.Options {
			criteria = criteria || !o.Description.IsNull()
		}
		if criteria {
			for _, o := range q.Options {
				if o.Key == "true" {
					sb.WriteString("\nYes: ")
				} else {
					sb.WriteString("\nNo: ")
				}
				if o.Description.IsNull() {
					sb.WriteString("None")
				} else {
					text(o.Description, 0)
				}
			}
		}
		sb.WriteString("\n\nReply with yes or no only.")
	default: // score
		sb.WriteString("\n\n")
		for _, o := range q.Options {
			sb.WriteString(o.Key + " ")
			if o.Description == nil {
				sb.WriteString("null")
			} else {
				text(o.Description, 0)
			}
			sb.WriteByte('\n')
		}
		sb.WriteString("\nReply with a single digit 0-" + strconv.Itoa(len(q.Options)-1) + " only.")
	}
	sb.WriteString("<|im_end|>\n<|im_start|>assistant\n<think></think>")
	return sb.String()
}

// escapeValue applies escape to the strings of v, object keys included.
func escapeValue(v *pyjson.Value, escape func(string) string) *pyjson.Value {
	switch v.Kind {
	case '"':
		return &pyjson.Value{Kind: '"', Str: escape(v.Str)}
	case '{', '[':
		out := &pyjson.Value{Kind: v.Kind}
		for i, f := range v.Fields {
			if v.Kind == '{' {
				out.Keys = append(out.Keys, escape(v.Keys[i]))
			}
			out.Fields = append(out.Fields, escapeValue(f, escape))
		}
		return out
	}
	return v
}

// Escaper returns the escape llama.cpp applies to the texts of a d1 request (d1_break): the
// spelling of a control token written in them (such as "<|im_end|>") would be read as that token,
// so a zero width space goes after its first character. Spellings shorter than 2 bytes are left.
func Escaper(control []string) func(string) string {
	var spellings []string
	for _, s := range control {
		if len(s) >= 2 {
			spellings = append(spellings, s)
		}
	}
	// longest first, the order kept among equal lengths
	for i := 1; i < len(spellings); i++ {
		for j := i; j > 0 && len(spellings[j]) > len(spellings[j-1]); j-- {
			spellings[j], spellings[j-1] = spellings[j-1], spellings[j]
		}
	}
	const zwsp = "\u200b"
	return func(text string) string {
		// again for a match that overlapped the one broken before it
		for changed := true; changed; {
			changed = false
			var out strings.Builder
			for i := 0; i < len(text); {
				hit := ""
				for _, s := range spellings {
					if strings.HasPrefix(text[i:], s) {
						hit = s
						break
					}
				}
				if hit == "" {
					out.WriteByte(text[i])
					i++
					continue
				}
				c := hit[0]
				n := 1
				switch {
				case c >= 0xF0:
					n = 4
				case c >= 0xE0:
					n = 3
				case c >= 0x80:
					n = 2
				}
				out.WriteString(hit[:n] + zwsp + hit[n:])
				i += len(hit)
				changed = true
			}
			text = out.String()
		}
		return text
	}
}
