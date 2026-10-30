// Package pyjson holds JSON values that keep the order of object keys and the text of numbers,
// and writes them the way Python's json.dumps does with ensure_ascii off (what the tojson of a
// chat template writes): ", " and ": " between items, or with an indent "," and newlines, keys in
// their order, non-ASCII as is. Prompts of the chat and decision templates are built from them.
package pyjson

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Value is a JSON value.
type Value struct {
	Kind   byte // '{', '[', '"', 'n' (number), 't', 'f', '0' (null)
	Str    string
	Keys   []string
	Fields []*Value // object values by keys, or array items
}

// Parse reads one JSON value.
func Parse(raw []byte) (*Value, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	v, err := decode(d)
	if err != nil {
		return nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, fmt.Errorf("pyjson: trailing data after JSON value")
	}
	return v, nil
}

func decode(d *json.Decoder) (*Value, error) {
	t, err := d.Token()
	if err != nil {
		return nil, err
	}
	switch t := t.(type) {
	case json.Delim:
		v := &Value{Kind: byte(t)}
		for d.More() {
			if t == '{' {
				k, err := d.Token()
				if err != nil {
					return nil, err
				}
				v.Keys = append(v.Keys, k.(string))
			}
			f, err := decode(d)
			if err != nil {
				return nil, err
			}
			v.Fields = append(v.Fields, f)
		}
		if _, err := d.Token(); err != nil { // the closing delimiter
			return nil, err
		}
		return v, nil
	case string:
		return &Value{Kind: '"', Str: t}, nil
	case json.Number:
		return &Value{Kind: 'n', Str: string(t)}, nil
	case bool:
		if t {
			return &Value{Kind: 't'}, nil
		}
		return &Value{Kind: 'f'}, nil
	default:
		return &Value{Kind: '0'}, nil
	}
}

// UnmarshalJSON lets a Value be a field of a struct decoded with encoding/json.
func (v *Value) UnmarshalJSON(b []byte) error {
	p, err := Parse(b)
	if err != nil {
		return err
	}
	*v = *p
	return nil
}

// Get returns the value of an object's key, or nil.
func (v *Value) Get(key string) *Value {
	if v == nil || v.Kind != '{' {
		return nil
	}
	for i, k := range v.Keys {
		if k == key {
			return v.Fields[i]
		}
	}
	return nil
}

// IsNull reports whether v is missing or null.
func (v *Value) IsNull() bool { return v == nil || v.Kind == '0' }

// Truthy is the value's truth in a template's if: false for null, false, "", 0, [] and {}.
func (v *Value) Truthy() bool {
	switch {
	case v.IsNull() || v.Kind == 'f':
		return false
	case v.Kind == '"':
		return v.Str != ""
	case v.Kind == 'n':
		f, err := strconv.ParseFloat(v.Str, 64)
		return err != nil || f != 0
	case v.Kind == '{' || v.Kind == '[':
		return len(v.Fields) > 0
	}
	return true
}

// Write writes v as json.dumps(v, ensure_ascii=False) does.
func (v *Value) Write(sb *strings.Builder) { v.write(sb, -1, 0) }

// WriteIndent writes v as json.dumps(v, indent=indent, ensure_ascii=False) does.
func (v *Value) WriteIndent(sb *strings.Builder, indent int) { v.write(sb, indent, 0) }

func (v *Value) write(sb *strings.Builder, indent, level int) {
	sep := ", "
	if indent >= 0 {
		sep = ","
	}
	// item starts each item and ends the container: a newline and the indent of the level
	item := func(lvl int) {
		if indent >= 0 {
			sb.WriteByte('\n')
			sb.WriteString(strings.Repeat(" ", lvl*indent))
		}
	}
	switch v.Kind {
	case '{', '[':
		open, close := byte('['), byte(']')
		if v.Kind == '{' {
			open, close = '{', '}'
		}
		sb.WriteByte(open)
		for i, f := range v.Fields {
			if i > 0 {
				sb.WriteString(sep)
			}
			item(level + 1)
			if v.Kind == '{' {
				String(sb, v.Keys[i])
				sb.WriteString(": ")
			}
			f.write(sb, indent, level+1)
		}
		if len(v.Fields) > 0 {
			item(level)
		}
		sb.WriteByte(close)
	case '"':
		String(sb, v.Str)
	case 'n':
		sb.WriteString(v.Str)
	case 't':
		sb.WriteString("true")
	case 'f':
		sb.WriteString("false")
	default:
		sb.WriteString("null")
	}
}

// String is v as Write writes it.
func (v *Value) String() string {
	var sb strings.Builder
	v.Write(&sb)
	return sb.String()
}

// String writes s as a JSON string the way Python does with ensure_ascii off.
func String(sb *strings.Builder, s string) {
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
