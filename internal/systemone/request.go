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
	"image"
	"strconv"

	"github.com/shibukawa/hwnpu/internal/pyjson"
	"github.com/shibukawa/hwnpu/internal/vision"
)

// Request is a parsed System One request.
type Request struct {
	// State is the content to evaluate: text, or a JSON value given to the model as JSON text;
	// null to ask about nothing but the question.
	State     *pyjson.Value
	Questions []Question
	// Images come first in the prompt: those of files (or images), then those taken out of a
	// state made of chat messages.
	Images []image.Image
}

// MaxImages is the most images a request can have, as llama.cpp's server allows.
const MaxImages = 8

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

// ParseRequest reads a request body. Images are data URLs, in files (or its alias images) or as
// image_url parts of a state made of chat messages (an array of messages, or an object with
// one), which are taken out of the state. Video and audio are not supported.
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
	r := &Request{State: body.Get("state")}
	if r.State == nil {
		return nil, bad(`"state" must be provided`)
	}
	addImage := func(url *pyjson.Value) error {
		if url == nil || url.Kind != '"' {
			return bad("media must be data URLs (data:image/...;base64,...)")
		}
		if len(r.Images) >= MaxImages {
			return bad("too many images, the maximum is %d", MaxImages)
		}
		img, err := vision.DecodeDataURL(url.Str)
		if err != nil {
			return bad("%v", err)
		}
		r.Images = append(r.Images, img)
		return nil
	}
	for _, key := range []string{"files", "images"} {
		v := body.Get(key)
		if v.IsNull() {
			continue
		}
		if v.Kind != '[' {
			return nil, bad("%q must be an array", key)
		}
		for _, url := range v.Fields {
			if err := addImage(url); err != nil {
				return nil, err
			}
		}
	}
	if r.State, err = takeMedia(r.State, addImage); err != nil {
		return nil, err
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

// takeMedia takes the image_url parts out of the content of a state made of chat messages (an
// array of messages, or an object with one) and gives their URLs to add, in order, as llama.cpp
// does. Audio and video parts are not supported.
func takeMedia(state *pyjson.Value, add func(url *pyjson.Value) error) (*pyjson.Value, error) {
	msgs := state
	if state.Kind == '{' {
		msgs = state.Get("messages")
	}
	if msgs == nil || msgs.Kind != '[' {
		return state, nil
	}
	out := &pyjson.Value{Kind: '['}
	for _, m := range msgs.Fields {
		content := m.Get("content")
		if content == nil || content.Kind != '[' {
			out.Fields = append(out.Fields, m)
			continue
		}
		kept := &pyjson.Value{Kind: '['}
		for _, part := range content.Fields {
			typ := ""
			if t := part.Get("type"); t != nil && t.Kind == '"' {
				typ = t.Str
			}
			switch {
			case typ == "image_url" && part.Get("image_url") != nil:
				url := part.Get("image_url")
				if u := url.Get("url"); u != nil {
					url = u
				}
				if err := add(url); err != nil {
					return nil, err
				}
			case typ == "input_audio" && part.Get("input_audio") != nil:
				return nil, &Unsupported{"audio input needs a model with audio input"}
			case typ == "video_url" && part.Get("video_url") != nil:
				return nil, &Unsupported{"video input is not supported by this server"}
			default:
				kept.Fields = append(kept.Fields, part)
			}
		}
		mo := &pyjson.Value{Kind: '{'}
		for i, k := range m.Keys {
			f := m.Fields[i]
			if k == "content" {
				f = kept
			}
			mo.Keys, mo.Fields = append(mo.Keys, k), append(mo.Fields, f)
		}
		out.Fields = append(out.Fields, mo)
	}
	if state.Kind != '{' {
		return out, nil
	}
	so := &pyjson.Value{Kind: '{'}
	for i, k := range state.Keys {
		f := state.Fields[i]
		if k == "messages" {
			f = out
		}
		so.Keys, so.Fields = append(so.Keys, k), append(so.Fields, f)
	}
	return so, nil
}

// Unsupported is a request the server cannot serve (HTTP 501).
type Unsupported struct{ Msg string }

func (e *Unsupported) Error() string { return e.Msg }

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
