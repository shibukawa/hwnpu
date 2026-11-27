package text

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
)

// Format asks a chat model for its answer as JSON (the OpenAI API's response_format): any JSON
// object, or one that matches a JSON Schema. The model is told so in the system message and its
// answer is checked: the JSON is taken out of it (code fences and text around it are left out)
// and matched against the schema, and an answer that is not such JSON is a *FormatError. The
// decoding is not constrained by the schema yet: a small model may fail it, and the app may ask
// again.
//
// The schema's keywords checked are type, properties, required, additionalProperties (false or
// a schema), items, enum, const, minimum, maximum, exclusiveMinimum, exclusiveMaximum,
// minLength, maxLength, minItems, maxItems, anyOf, oneOf and allOf; the others ($ref, pattern,
// format, ...) are not.
type Format struct {
	// Schema is the JSON Schema of the answer; nil asks for any JSON object.
	Schema json.RawMessage
}

// FormatError is the error of an answer that is not the JSON Format asked for.
type FormatError struct {
	// Text is the answer, Why what is wrong with it.
	Text, Why string
}

func (e *FormatError) Error() string { return "text: the answer is not the JSON asked for: " + e.Why }

// Instruction is what the conversation is told of the format, at the end of its system message
// (or as one): JSON only, of the schema when there is one.
func (f *Format) Instruction() string {
	if len(f.Schema) == 0 {
		return "Answer with a JSON object only, without any other text or code fences."
	}
	var b bytes.Buffer
	if json.Compact(&b, f.Schema) != nil {
		b.Reset()
		b.Write(f.Schema)
	}
	return "Answer with JSON only, without any other text or code fences. The JSON must match this JSON Schema:\n" + b.String()
}

// Check takes the JSON out of an answer and matches it against the schema: the JSON, or a
// *FormatError of why it is not the JSON asked for.
func (f *Format) Check(text string) (string, error) {
	s := strings.TrimSpace(text)
	i := strings.IndexAny(s, "{[")
	if i < 0 {
		return text, &FormatError{Text: text, Why: "no JSON in it"}
	}
	s = s[i:]
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return text, &FormatError{Text: text, Why: err.Error()}
	}
	out := strings.TrimSpace(s[:dec.InputOffset()])
	if len(f.Schema) == 0 {
		if _, ok := v.(map[string]any); !ok {
			return text, &FormatError{Text: text, Why: "not a JSON object"}
		}
		return out, nil
	}
	var schema any
	if err := json.Unmarshal(f.Schema, &schema); err != nil {
		return text, fmt.Errorf("text: the schema: %w", err)
	}
	if why := match(v, schema, "$"); why != "" {
		return text, &FormatError{Text: text, Why: why}
	}
	return out, nil
}

// match is "" when v matches the schema s, else what does not, at path.
func match(v any, s any, path string) string {
	sm, ok := s.(map[string]any)
	if !ok {
		if b, ok := s.(bool); ok && !b {
			return path + ": nothing is allowed here"
		}
		return ""
	}
	if t, ok := sm["type"]; ok {
		var types []string
		switch t := t.(type) {
		case string:
			types = []string{t}
		case []any:
			for _, x := range t {
				if s, ok := x.(string); ok {
					types = append(types, s)
				}
			}
		}
		if len(types) > 0 && !slices.ContainsFunc(types, func(t string) bool { return isType(v, t) }) {
			return fmt.Sprintf("%s: %s is not of type %s", path, kind(v), strings.Join(types, " or "))
		}
	}
	if e, ok := sm["enum"].([]any); ok && !slices.ContainsFunc(e, func(x any) bool { return equal(x, v) }) {
		return fmt.Sprintf("%s: %s is not one of the enum", path, short(v))
	}
	if c, ok := sm["const"]; ok && !equal(c, v) {
		return fmt.Sprintf("%s: %s is not the const", path, short(v))
	}
	for _, k := range []string{"allOf", "anyOf", "oneOf"} {
		subs, ok := sm[k].([]any)
		if !ok {
			continue
		}
		n := 0
		first := ""
		for _, sub := range subs {
			if why := match(v, sub, path); why == "" {
				n++
			} else if first == "" {
				first = why
			}
		}
		switch {
		case k == "allOf" && n < len(subs):
			return first
		case k == "anyOf" && n == 0, k == "oneOf" && n != 1:
			return fmt.Sprintf("%s: matches %d of the %s", path, n, k)
		}
	}
	switch v := v.(type) {
	case map[string]any:
		props, _ := sm["properties"].(map[string]any)
		if req, ok := sm["required"].([]any); ok {
			for _, r := range req {
				if name, ok := r.(string); ok {
					if _, has := v[name]; !has {
						return fmt.Sprintf("%s: %q is required", path, name)
					}
				}
			}
		}
		for _, k := range sortedKeys(v) {
			if p, ok := props[k]; ok {
				if why := match(v[k], p, path+"."+k); why != "" {
					return why
				}
				continue
			}
			if ap, ok := sm["additionalProperties"]; ok {
				if why := match(v[k], ap, path+"."+k); why != "" {
					return why
				}
			}
		}
	case []any:
		if n, ok := number(sm["minItems"]); ok && float64(len(v)) < n {
			return fmt.Sprintf("%s: fewer than %v items", path, n)
		}
		if n, ok := number(sm["maxItems"]); ok && float64(len(v)) > n {
			return fmt.Sprintf("%s: more than %v items", path, n)
		}
		if it, ok := sm["items"]; ok {
			for i, x := range v {
				if why := match(x, it, fmt.Sprintf("%s[%d]", path, i)); why != "" {
					return why
				}
			}
		}
	case string:
		n := float64(len([]rune(v)))
		if m, ok := number(sm["minLength"]); ok && n < m {
			return fmt.Sprintf("%s: shorter than %v characters", path, m)
		}
		if m, ok := number(sm["maxLength"]); ok && n > m {
			return fmt.Sprintf("%s: longer than %v characters", path, m)
		}
	case json.Number:
		x, _ := v.Float64()
		if m, ok := number(sm["minimum"]); ok && x < m {
			return fmt.Sprintf("%s: %v is below the minimum %v", path, v, m)
		}
		if m, ok := number(sm["maximum"]); ok && x > m {
			return fmt.Sprintf("%s: %v is above the maximum %v", path, v, m)
		}
		if m, ok := number(sm["exclusiveMinimum"]); ok && x <= m {
			return fmt.Sprintf("%s: %v is not above %v", path, v, m)
		}
		if m, ok := number(sm["exclusiveMaximum"]); ok && x >= m {
			return fmt.Sprintf("%s: %v is not below %v", path, v, m)
		}
	}
	return ""
}

func isType(v any, t string) bool {
	switch t {
	case "object":
		_, ok := v.(map[string]any)
		return ok
	case "array":
		_, ok := v.([]any)
		return ok
	case "string":
		_, ok := v.(string)
		return ok
	case "boolean":
		_, ok := v.(bool)
		return ok
	case "null":
		return v == nil
	case "number":
		_, ok := v.(json.Number)
		return ok
	case "integer":
		n, ok := v.(json.Number)
		if !ok {
			return false
		}
		x, err := n.Float64()
		return err == nil && x == math.Trunc(x)
	}
	return true
}

func kind(v any) string {
	switch v.(type) {
	case map[string]any:
		return "an object"
	case []any:
		return "an array"
	case string:
		return "a string"
	case bool:
		return "a boolean"
	case nil:
		return "null"
	}
	return "a number"
}

// number is a schema's number (json.Unmarshal into any gives float64).
func number(x any) (float64, bool) {
	switch x := x.(type) {
	case float64:
		return x, true
	case json.Number:
		f, err := x.Float64()
		return f, err == nil
	}
	return 0, false
}

// equal compares a schema's value (numbers as float64) with the answer's (json.Number).
func equal(schema, v any) bool {
	if n, ok := v.(json.Number); ok {
		f, _ := n.Float64()
		s, ok := number(schema)
		return ok && s == f
	}
	a, _ := json.Marshal(schema)
	b, _ := json.Marshal(v)
	return bytes.Equal(a, b)
}

func short(v any) string {
	b, _ := json.Marshal(v)
	if len(b) > 40 {
		return string(b[:40]) + "…"
	}
	return string(b)
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}
