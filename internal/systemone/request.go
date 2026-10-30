// Package systemone answers typed questions about a state with a d1 decision model (Liquid AI's
// d1-2B on LFM3), as llama.cpp's /v1/systemone (the TypeSafe System One API) does: each question
// is a prompt of its own, read in one call of the program (internal/llm/lfm3 with Config.D1), and
// the answer is the softmax of the logits of the options' labels at the prompt's last token.
//
// The prompt, the labels and the answers follow llama.cpp (tools/server/server-decision.cpp and
// the systemone template that convert_hf_to_gguf.py writes for LFM3 d1), which follows prompt.py
// and api.py of the model repository.
package systemone

import (
	"fmt"
	"strconv"

	"github.com/shibukawa/hwnpu/internal/pyjson"
)

// Request is a parsed System One request.
type Request struct {
	// State is the content to evaluate: text, or a JSON value given to the model as JSON text;
	// null to ask about nothing but the question.
	State     *pyjson.Value
	Questions []Question
}

// Question is one question of a request.
type Question struct {
	ID           string
	Type         string // choice, score or noul
	Instructions *pyjson.Value
	// Options are the possible answers in the order the prompt lists them: a choice's criteria,
	// a score's levels "0".., a noul's "true" then "false".
	Options []Option
}

// Option is one possible answer.
type Option struct {
	Key         string
	Description *pyjson.Value // nil when not given
}

// MaxOptions is the most options a choice question can have.
const MaxOptions = 255

// BadRequest is an error in the request (HTTP 400).
type BadRequest struct{ Msg string }

func (e *BadRequest) Error() string { return e.Msg }

func bad(format string, args ...any) error { return &BadRequest{fmt.Sprintf(format, args...)} }

// ParseRequest reads a request body. Image and audio input are not supported yet: a request with
// files or images is rejected.
func ParseRequest(raw []byte) (*Request, error) {
	body, err := pyjson.Parse(raw)
	if err != nil {
		return nil, bad("invalid JSON: %v", err)
	}
	if body.Kind != '{' {
		return nil, bad("the request must be an object")
	}
	if v := body.Get("videos"); !v.IsNull() && len(v.Fields) > 0 {
		return nil, bad(`"videos" is not supported`)
	}
	for _, key := range []string{"files", "images"} {
		if v := body.Get(key); !v.IsNull() {
			if v.Kind != '[' {
				return nil, bad("%q must be an array", key)
			}
			if len(v.Fields) > 0 {
				return nil, &Unsupported{"image input is not supported by this server yet"}
			}
		}
	}
	r := &Request{State: body.Get("state")}
	if r.State == nil {
		return nil, bad(`"state" must be provided`)
	}
	if hasMedia(r.State) {
		return nil, &Unsupported{"image and audio parts in the state are not supported by this server yet"}
	}
	qs := body.Get("questions")
	if qs == nil || qs.Kind != '{' || len(qs.Fields) == 0 {
		return nil, bad(`"questions" must be a non-empty object`)
	}
	for i, id := range qs.Keys {
		q, err := parseQuestion(id, qs.Fields[i])
		if err != nil {
			return nil, err
		}
		r.Questions = append(r.Questions, q)
	}
	return r, nil
}

// Unsupported is a request the server cannot serve (HTTP 501).
type Unsupported struct{ Msg string }

func (e *Unsupported) Error() string { return e.Msg }

// hasMedia reports whether a state made of chat messages (an array of messages, or an object
// with one) has image or audio parts, which llama.cpp takes out of the state as media.
func hasMedia(state *pyjson.Value) bool {
	msgs := state
	if state.Kind == '{' {
		msgs = state.Get("messages")
	}
	if msgs == nil || msgs.Kind != '[' {
		return false
	}
	for _, m := range msgs.Fields {
		content := m.Get("content")
		if content == nil || content.Kind != '[' {
			continue
		}
		for _, part := range content.Fields {
			if t := part.Get("type"); t != nil && t.Kind == '"' && (t.Str == "image_url" || t.Str == "input_audio" || t.Str == "video_url") {
				return true
			}
		}
	}
	return false
}

func parseQuestion(id string, q *pyjson.Value) (Question, error) {
	errf := func(format string, args ...any) error {
		return bad("questions.%s: %s", id, fmt.Sprintf(format, args...))
	}
	if q.Kind != '{' {
		return Question{}, errf("must be an object")
	}
	out := Question{ID: id, Instructions: q.Get("instructions")}
	if out.Instructions.IsNull() {
		return Question{}, errf(`"instructions" must be provided`)
	}
	if t := out.Instructions.Get("type"); t != nil && t.Kind == '"' && t.Str == "input_audio" {
		return Question{}, errf("spoken instructions need a model with audio input")
	}
	if t := q.Get("type"); t != nil && t.Kind == '"' {
		out.Type = t.Str
	}
	criteria := q.Get("criteria")
	switch out.Type {
	case "choice":
		if criteria == nil || criteria.Kind != '{' || len(criteria.Fields) == 0 {
			return Question{}, errf(`"criteria" must be a non-empty object`)
		}
		for i, k := range criteria.Keys {
			out.Options = append(out.Options, Option{Key: k, Description: criteria.Fields[i]})
		}
	case "score":
		if criteria == nil || criteria.Kind != '[' || len(criteria.Fields) < 2 || len(criteria.Fields) > 10 {
			return Question{}, errf(`"criteria" must be an array of 2 to 10 levels`)
		}
		for i, f := range criteria.Fields {
			out.Options = append(out.Options, Option{Key: strconv.Itoa(i), Description: f})
		}
	case "noul":
		if !criteria.IsNull() && criteria.Kind != '{' {
			return Question{}, errf(`"criteria" must be an object`)
		}
		// true first, as llama.cpp lists it for d1
		for _, k := range []string{"true", "false"} {
			out.Options = append(out.Options, Option{Key: k, Description: criteria.Get(k)})
		}
	default:
		return Question{}, errf(`"type" must be one of: choice, score, noul`)
	}
	if len(out.Options) > MaxOptions {
		return Question{}, errf("too many options (%d), this model supports at most %d", len(out.Options), MaxOptions)
	}
	return out, nil
}
