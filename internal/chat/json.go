package chat

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// value is a JSON value that keeps the order of object keys and the text of numbers, so that a
// tool definition or a call's arguments render in the prompt as the client wrote them.
type value struct {
	kind   byte // '{', '[', '"', 'n' (number), 't', 'f', '0' (null)
	str    string
	keys   []string
	fields []*value // object values by keys, or array items
}

func parseValue(raw []byte) (*value, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	v, err := decodeValue(d)
	if err != nil {
		return nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, fmt.Errorf("chat: trailing data after JSON value")
	}
	return v, nil
}

func decodeValue(d *json.Decoder) (*value, error) {
	t, err := d.Token()
	if err != nil {
		return nil, err
	}
	switch t := t.(type) {
	case json.Delim:
		v := &value{kind: byte(t)}
		for d.More() {
			if t == '{' {
				k, err := d.Token()
				if err != nil {
					return nil, err
				}
				v.keys = append(v.keys, k.(string))
			}
			f, err := decodeValue(d)
			if err != nil {
				return nil, err
			}
			v.fields = append(v.fields, f)
		}
		if _, err := d.Token(); err != nil { // the closing delimiter
			return nil, err
		}
		return v, nil
	case string:
		return &value{kind: '"', str: t}, nil
	case json.Number:
		return &value{kind: 'n', str: string(t)}, nil
	case bool:
		if t {
			return &value{kind: 't'}, nil
		}
		return &value{kind: 'f'}, nil
	default:
		return &value{kind: '0'}, nil
	}
}

// get returns the value of an object's key, or nil.
func (v *value) get(key string) *value {
	if v == nil || v.kind != '{' {
		return nil
	}
	for i, k := range v.keys {
		if k == key {
			return v.fields[i]
		}
	}
	return nil
}

// pyJSON writes v the way the chat template's tojson does (Python's json.dumps with
// ensure_ascii off): ", " and ": " between items, keys in their order, non-ASCII as is.
func (v *value) pyJSON(sb *strings.Builder) {
	switch v.kind {
	case '{':
		sb.WriteByte('{')
		for i, k := range v.keys {
			if i > 0 {
				sb.WriteString(", ")
			}
			pyString(sb, k)
			sb.WriteString(": ")
			v.fields[i].pyJSON(sb)
		}
		sb.WriteByte('}')
	case '[':
		sb.WriteByte('[')
		for i, f := range v.fields {
			if i > 0 {
				sb.WriteString(", ")
			}
			f.pyJSON(sb)
		}
		sb.WriteByte(']')
	case '"':
		pyString(sb, v.str)
	case 'n':
		sb.WriteString(v.str)
	case 't':
		sb.WriteString("true")
	case 'f':
		sb.WriteString("false")
	default:
		sb.WriteString("null")
	}
}

func (v *value) String() string {
	var sb strings.Builder
	v.pyJSON(&sb)
	return sb.String()
}

func pyString(sb *strings.Builder, s string) {
	sb.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			sb.WriteString(`\"`)
		case '\\':
			sb.WriteString(`\\`)
		case '\n':
			sb.WriteString(`\n`)
		case '\r':
			sb.WriteString(`\r`)
		case '\t':
			sb.WriteString(`\t`)
		case '\b':
			sb.WriteString(`\b`)
		case '\f':
			sb.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(sb, `\u%04x`, r)
			} else {
				sb.WriteRune(r)
			}
		}
	}
	sb.WriteByte('"')
}
