// Package openai serves an app's text models with the OpenAI API, as an http.Handler for the
// app to serve or to mount in a server of its own (hwnpu-chat serve uses it):
//
//	GET  /v1/models            the models served
//	POST /v1/chat/completions  the chat models (text.Chat, and translation models), streamed with
//	                           server-sent events or not, with tools and response_format
//	POST /v1/embeddings        the embedding models (text.Embed)
//	GET  /health               ok
//
// A request goes to the model of its endpoint's kind that its model field names (its name, or
// a use or a family of the definitions that picks it here), or to the only one of that kind
// whatever it names.
//
// Of the chat completions' fields it reads messages (with image_url and input_audio parts for
// the models that read images and audio, tool_calls and tool messages), tools and tool_choice
// (none drops the tools), stream and stream_options.include_usage, max_tokens and
// max_completion_tokens (none: until the cache is full), temperature, top_p, top_k,
// repetition_penalty, seed, stop, reasoning_effort (none, minimal and low turn the thinking off,
// medium and high on), response_format (json_object, json_schema: the answer is checked, not
// constrained, text.Format) and chat_template_kwargs (enable_thinking, preserve_thinking, and a
// translation model's target_lang, source_lang, form, context and glossary). Of the embeddings'
// it reads input (a string or a list of them), encoding_format (float or base64), dimensions (the
// model's only), and two fields of other embedding APIs: input_type (query, search_query;
// document, search_document, passage) and truncate.
package openai

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/text"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/chatwire"
)

// Server serves chat and embedding models with the OpenAI API. Each model runs one request at a
// time; the others wait.
type Server struct {
	Chats     []*text.Chatter
	Embedders []*text.Embedder
	// Think is whether a chat model thinks before it answers when a request does not say
	// (reasoning_effort, chat_template_kwargs.enable_thinking); nil for each model's own
	// (text.Chatter.Thinks).
	Think *bool
	// ThinkBudget is the most tokens a model thinks before it is made to answer, as
	// text.ChatOptions takes it (0: 1024; below 0: no limit).
	ThinkBudget int
	// Sampling is the sampling of requests that do not say (nil: text.DefaultSampling).
	Sampling *text.Sampling
	// Log receives a line per request (nil for none).
	Log io.Writer
	// Dump, when set, is a directory that receives each chat request, its prompt and the
	// generated text (ID.json), for finding out what a client sends.
	Dump string
	// Catalog is the definitions a request's model is looked up in when it names none of the
	// models served (a use or a family names the model it picks here, Catalog.Find), and those
	// of the embedding models' features; Load sets it (Runtime.Catalog). nil: the models' names
	// alone.
	Catalog *ai.Catalog

	n atomic.Int64
}

// Load is the Server of the uses' models in rt, loaded now (Use.Get).
func Load(ctx context.Context, rt *ai.Runtime, chats []ai.Use[*text.Chatter], embedders []ai.Use[*text.Embedder]) (*Server, error) {
	s := &Server{}
	for _, u := range chats {
		c, err := u.Get(ctx, rt)
		if err != nil {
			return nil, err
		}
		s.Chats = append(s.Chats, c)
	}
	for _, u := range embedders {
		e, err := u.Get(ctx, rt)
		if err != nil {
			return nil, err
		}
		s.Embedders = append(s.Embedders, e)
	}
	cat, err := rt.Catalog()
	if err != nil {
		return nil, err
	}
	s.Catalog = cat
	return s, nil
}

// Handler returns the API's routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("POST /v1/chat/completions", s.completions)
	mux.HandleFunc("POST /v1/embeddings", s.embeddings)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "ok\n") })
	return mux
}

// ServeHTTP serves the API (Handler).
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.Handler().ServeHTTP(w, r) }

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	data := []map[string]any{}
	for _, c := range s.Chats {
		in := []string{"text"}
		if c.Images() {
			in = append(in, "image")
		}
		if c.Audio() {
			in = append(in, "audio")
		}
		data = append(data, describe(c.Name(), catalog.KindChat, c.Features(), map[string]any{"context_length": c.Context(),
			"architecture": map[string]any{"input_modalities": in, "output_modalities": []string{"text"}}}))
	}
	for _, e := range s.Embedders {
		f := []string{}
		if s.Catalog != nil {
			if m, ok := s.Catalog.Model(e.Model()); ok && m.Features != nil {
				f = m.Features
			}
		}
		data = append(data, describe(e.Name(), catalog.KindEmbed, f, map[string]any{"context_length": e.MaxTokens(), "dimensions": e.Dim()}))
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

// describe is a model's entry in /v1/models: with its kind and its features (capabilities).
func describe(name string, kind catalog.Kind, features []string, more map[string]any) map[string]any {
	m := map[string]any{"id": name, "object": "model", "created": 0, "owned_by": "hwnpu", "kind": kind,
		"capabilities": append([]string{}, features...)}
	for k, v := range more {
		m[k] = v
	}
	return m
}

func (s *Server) names() (chats, embedders []string) {
	for _, c := range s.Chats {
		chats = append(chats, c.Name())
	}
	for _, e := range s.Embedders {
		embedders = append(embedders, e.Name())
	}
	return chats, embedders
}

// route picks the model a request names among those of the endpoint's kind (names): the one of
// that name, or the only one. A name of a model of the other kind (others) is an error, so that
// a client set up with the wrong model finds out. pick is the model a family or a use names
// here ("" for none; pick may be nil). It returns the index, or -1 with the error's status and
// message.
func route(model string, names, others []string, kind catalog.Kind, other string, pick func(string) string) (int, int, string) {
	for i, n := range names {
		if strings.EqualFold(model, n) {
			return i, 0, ""
		}
	}
	// a family or a use (translate, plamo3translate:2b) names the model it picks here
	if pick != nil {
		if p := pick(model); p != "" {
			for i, n := range names {
				if strings.EqualFold(p, n) {
					return i, 0, ""
				}
			}
			model = p
		}
	}
	for _, n := range others {
		if strings.EqualFold(model, n) {
			return -1, http.StatusBadRequest, fmt.Sprintf("%s is not a %s model: use %s", n, kind, other)
		}
	}
	switch len(names) {
	case 0:
		return -1, http.StatusNotFound, fmt.Sprintf("no %s model is loaded: add one to --model (hwnpu-chat list shows them)", kind)
	case 1:
		return 0, 0, ""
	}
	return -1, http.StatusNotFound, fmt.Sprintf("no %s model %q: the model is one of %s", kind, model, strings.Join(names, ", "))
}

// pick is the model a family or a use named model picks here (Catalog.Find, as a chat model,
// else as an embedding model), "" for none: a model's own name picks none, as no name of the
// definitions does.
func (s *Server) pick(model string) string {
	if s.Catalog == nil || model == "" {
		return ""
	}
	if _, ok := s.Catalog.Model(model); ok {
		return ""
	}
	for _, u := range []ai.Requirement{text.Chat(model), text.Embed(model)} {
		if m, err := s.Catalog.Find(model, u); err == nil {
			return m.Name
		}
	}
	return ""
}

type chatRequest struct {
	Model         string            `json:"model"`
	Messages      []message         `json:"messages"`
	Tools         []json.RawMessage `json:"tools"`
	ToolChoice    json.RawMessage   `json:"tool_choice"`
	Stream        bool              `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	MaxTokens           int             `json:"max_tokens"`
	MaxCompletionTokens int             `json:"max_completion_tokens"`
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	TopK                *int            `json:"top_k"`
	RepetitionPenalty   *float64        `json:"repetition_penalty"`
	Seed                uint64          `json:"seed"`
	Stop                json.RawMessage `json:"stop"`
	ReasoningEffort     string          `json:"reasoning_effort"`
	ResponseFormat      *struct {
		Type       string `json:"type"`
		JSONSchema *struct {
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	} `json:"response_format"`
	ChatTemplateKwargs struct {
		EnableThinking   *bool `json:"enable_thinking"`
		PreserveThinking bool  `json:"preserve_thinking"`
		// a translation model's (PLaMo 3's)
		TargetLang string `json:"target_lang"`
		SourceLang string `json:"source_lang"`
		Form       string `json:"form"`
		Context    string `json:"context"`
		Glossary   []struct {
			Source string `json:"source"`
			Target string `json:"target"`
		} `json:"glossary"`
	} `json:"chat_template_kwargs"`
}

// thinks decides whether the model thinks before answering req: the server's default (Think),
// else the model's own, then what the request asks.
func (s *Server) thinks(c *text.Chatter, req *chatRequest) bool {
	think := c.Thinks()
	if s.Think != nil {
		think = *s.Think
	}
	switch req.ReasoningEffort {
	case "none", "minimal", "low":
		// A small model thinks at length whatever it is asked; the side tasks of agents
		// (titles) ask for little thinking and want the answer.
		think = false
	case "medium", "high":
		think = true
	}
	if req.ChatTemplateKwargs.EnableThinking != nil {
		think = *req.ChatTemplateKwargs.EnableThinking
	}
	return think
}

// options are the request's ChatOptions.
func (s *Server) options(req *chatRequest, think bool) (text.ChatOptions, error) {
	o := text.ChatOptions{Think: &think, ThinkBudget: s.ThinkBudget, KeepThinking: req.ChatTemplateKwargs.PreserveThinking,
		Seed: req.Seed, MaxTokens: text.Unlimited}
	if string(req.ToolChoice) != `"none"` {
		for _, raw := range req.Tools {
			t, err := text.ParseTool(raw)
			if err != nil {
				return o, err
			}
			o.Tools = append(o.Tools, t)
		}
	}
	if o.Seed == 0 {
		o.Seed = rand.Uint64() // drawn here so that the response and the log tell it
	}
	if req.MaxTokens > 0 {
		o.MaxTokens = req.MaxTokens
	}
	if req.MaxCompletionTokens > 0 {
		o.MaxTokens = req.MaxCompletionTokens
	}
	sm := text.DefaultSampling
	if s.Sampling != nil {
		sm = *s.Sampling
	}
	if req.Temperature != nil {
		sm.Temperature = *req.Temperature
	}
	if req.TopP != nil {
		sm.TopP = *req.TopP
	}
	if req.TopK != nil {
		sm.TopK = *req.TopK
	}
	if req.RepetitionPenalty != nil {
		sm.Repetition = *req.RepetitionPenalty
	}
	o.Sampling = &sm
	if len(req.Stop) > 0 && string(req.Stop) != "null" {
		var one string
		if json.Unmarshal(req.Stop, &one) == nil {
			o.Stop = []string{one}
		} else if err := json.Unmarshal(req.Stop, &o.Stop); err != nil {
			return o, errors.New("stop is neither a string nor a list of strings")
		}
	}
	if f := req.ResponseFormat; f != nil {
		switch f.Type {
		case "", "text":
		case "json_object":
			o.Format = &text.Format{}
		case "json_schema":
			o.Format = &text.Format{}
			if f.JSONSchema != nil {
				o.Format.Schema = f.JSONSchema.Schema
			}
		default:
			return o, fmt.Errorf("response_format %q is neither text, json_object nor json_schema", f.Type)
		}
	}
	k := req.ChatTemplateKwargs
	o.Translation = &text.TranslateOptions{To: k.TargetLang, From: k.SourceLang, Form: k.Form, Context: k.Context}
	for _, g := range k.Glossary {
		o.Translation.Glossary = append(o.Translation.Glossary, text.Pair{Source: g.Source, Target: g.Target})
	}
	return o, nil
}

func (s *Server) completions(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	body, err := io.ReadAll(r.Body)
	if err == nil {
		err = json.Unmarshal(body, &req)
	}
	if err != nil {
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", err.Error())
		return
	}
	chats, embedders := s.names()
	i, code, msg := route(req.Model, chats, embedders, catalog.KindChat, "/v1/embeddings", s.pick)
	if i < 0 {
		apiError(w, code, "invalid_request_error", "model_not_found", msg)
		return
	}
	c := s.Chats[i]
	if len(req.Messages) == 0 {
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", "messages is empty")
		return
	}
	o, err := s.options(&req, s.thinks(c, &req))
	if err != nil {
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", err.Error())
		return
	}
	msgs := make([]text.Message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = m.message()
	}
	n, err := c.Count(r.Context(), msgs, o)
	if err != nil {
		// a template that refuses the conversation, an image or a clip the model does not read
		code, typ := http.StatusBadRequest, "invalid_request_error"
		if errors.Is(err, chatwire.ErrClosed) || r.Context().Err() != nil {
			code, typ = http.StatusInternalServerError, "server_error"
		}
		apiError(w, code, typ, "", err.Error())
		return
	}

	id := fmt.Sprintf("chatcmpl-%d-%d", time.Now().Unix(), s.n.Add(1))
	created := time.Now().Unix()
	// a long prompt on a large cache takes minutes before its first token
	s.logf("%s: %d tokens of prompt, %d messages, %d tools, seed %d", id, n, len(req.Messages), len(o.Tools), o.Seed)
	chunk := func(delta map[string]any, finish any) map[string]any {
		return map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": c.Name(), "seed": o.Seed,
			"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}},
		}
	}
	var fn func(text.Delta) bool
	flush := func() {}
	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		rc := http.NewResponseController(w)
		flush = func() { rc.Flush() }
		sse(w, chunk(map[string]any{"role": "assistant", "content": ""}, nil))
		flush()
		fn = func(d text.Delta) bool {
			if d.Reasoning != "" {
				sse(w, chunk(map[string]any{"reasoning_content": d.Reasoning}, nil))
			}
			if d.Text != "" {
				sse(w, chunk(map[string]any{"content": d.Text}, nil))
			}
			flush()
			return true
		}
	}
	a, err := c.Reply(r.Context(), msgs, o, fn)
	var bad *text.FormatError
	if err != nil && !errors.As(err, &bad) {
		code := http.StatusInternalServerError
		typ, ecode := "server_error", ""
		if errors.Is(err, text.ErrContext) {
			code, typ, ecode = http.StatusBadRequest, "invalid_request_error", "context_length_exceeded"
		}
		s.logf("%s: %v", id, err)
		if req.Stream {
			sse(w, map[string]any{"error": map[string]any{"message": err.Error(), "type": typ, "code": ecode}})
			flush()
			return
		}
		apiError(w, code, typ, ecode, err.Error())
		return
	}
	stats := a.Stats
	if s.Dump != "" {
		b, _ := json.MarshalIndent(map[string]any{"request": json.RawMessage(body), "prompt": a.Prompt, "generated": a.Raw, "stats": stats}, "", " ")
		if err := os.WriteFile(filepath.Join(s.Dump, id+".json"), b, 0o644); err != nil {
			s.logf("%s: %v", id, err)
		}
	}
	looped := ""
	if stats.Looped {
		looped = " (stopped repeating itself)"
	}
	if stats.ThinkCut {
		looped += fmt.Sprintf(" (thinking cut at %d tokens)", s.ThinkBudget)
	}
	if stats.ThinkLooped {
		looped += " (thinking cut going round in circles)"
	}
	if stats.Drafted > 0 {
		looped += fmt.Sprintf(" (%d of %d drafts accepted, %d calls)", stats.Accepted, stats.Drafted, stats.Calls)
	}
	if stats.Cache > 0 && stats.Cache < c.Context() {
		looped += fmt.Sprintf(" (cache %d)", stats.Cache)
	}
	if stats.Restored > 0 {
		looped += fmt.Sprintf(" (%d tokens from the prompt store)", stats.Restored)
	}
	s.logf("%s: prompt %d tokens (%d reused) in %.1fs, %d tokens in %.1fs (%.1f ms/token), %s%s",
		id, stats.Prompt, stats.Reused, stats.Prefill.Seconds(), stats.Generated, stats.Decode.Seconds(),
		float64(stats.Decode.Milliseconds())/float64(max(stats.Generated, 1)), a.Finish, looped)
	if bad != nil {
		s.logf("%s: %v", id, err)
		if req.Stream {
			sse(w, map[string]any{"error": map[string]any{"message": err.Error(), "type": "server_error", "code": "response_format_failed"}})
			flush()
			return
		}
		apiError(w, http.StatusInternalServerError, "server_error", "response_format_failed", err.Error())
		return
	}
	usage := map[string]any{
		"prompt_tokens": stats.Prompt, "completion_tokens": stats.Generated, "total_tokens": stats.Prompt + stats.Generated,
		"prompt_tokens_details": map[string]any{"cached_tokens": stats.Reused},
	}
	calls := toolCalls(a.ToolCalls)
	if !req.Stream {
		msg := map[string]any{"role": "assistant", "content": a.Text}
		if *o.Think {
			msg["reasoning_content"] = a.Reasoning
		}
		if len(calls) > 0 {
			msg["tool_calls"] = calls
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id": id, "object": "chat.completion", "created": created, "model": c.Name(), "seed": o.Seed,
			"choices": []map[string]any{{"index": 0, "message": msg, "finish_reason": a.Finish}},
			"usage":   usage,
		})
		return
	}
	if len(calls) > 0 {
		for i := range calls {
			calls[i]["index"] = i
		}
		sse(w, chunk(map[string]any{"tool_calls": calls}, nil))
	}
	sse(w, chunk(map[string]any{}, a.Finish))
	if req.StreamOptions.IncludeUsage {
		sse(w, map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": c.Name(), "seed": o.Seed, "choices": []any{}, "usage": usage})
	}
	io.WriteString(w, "data: [DONE]\n\n")
	flush()
}

// toolCalls are calls as the API writes them.
func toolCalls(tcs []text.ToolCall) []map[string]any {
	var out []map[string]any
	for _, tc := range tcs {
		fn := struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		}{tc.Name, tc.Arguments}
		out = append(out, map[string]any{"id": tc.ID, "type": "function", "function": fn})
	}
	return out
}

type embedRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format"`
	Dimensions     int             `json:"dimensions"`
	// InputType says whether the texts are search queries ("query", or "search_query") or what
	// is searched ("document", "search_document" or "passage"; the default, also for texts that
	// are not about search). Not in OpenAI's API; other embedding APIs name it so.
	InputType string `json:"input_type"`
	// Truncate cuts a text over the model's length instead of failing (as in text-embeddings-
	// inference).
	Truncate bool `json:"truncate"`
}

func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	var req embedRequest
	body, err := io.ReadAll(r.Body)
	if err == nil {
		err = json.Unmarshal(body, &req)
	}
	if err != nil {
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", err.Error())
		return
	}
	chats, embedders := s.names()
	i, code, msg := route(req.Model, embedders, chats, catalog.KindEmbed, "/v1/chat/completions", s.pick)
	if i < 0 {
		apiError(w, code, "invalid_request_error", "model_not_found", msg)
		return
	}
	emb := s.Embedders[i]
	var texts []string
	var one string
	if json.Unmarshal(req.Input, &one) == nil {
		texts = []string{one}
	} else if err := json.Unmarshal(req.Input, &texts); err != nil {
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", "input is neither a string nor a list of strings (token ids are not taken)")
		return
	}
	if len(texts) == 0 {
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", "input is empty")
		return
	}
	var query bool
	switch req.InputType {
	case "query", "search_query":
		query = true
	case "", "document", "search_document", "passage":
	default:
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", "input_type is neither query nor document: "+req.InputType)
		return
	}
	b64 := false
	switch req.EncodingFormat {
	case "base64":
		b64 = true
	case "", "float":
	default:
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", "encoding_format is neither float nor base64: "+req.EncodingFormat)
		return
	}
	if req.Dimensions != 0 && req.Dimensions != emb.Dim() {
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", fmt.Sprintf("dimensions must be %d, the model's", emb.Dim()))
		return
	}

	id := fmt.Sprintf("embd-%d-%d", time.Now().Unix(), s.n.Add(1))
	t0 := time.Now()
	data := make([]map[string]any, len(texts))
	tokens, truncated := 0, 0
	for i, t := range texts {
		e, err := emb.Embed(r.Context(), t, text.EmbedOptions{Query: query, Truncate: req.Truncate})
		if err != nil {
			code, ecode := http.StatusBadRequest, ""
			switch {
			case errors.Is(err, text.ErrTooLong):
				ecode = "context_length_exceeded"
			case r.Context().Err() != nil:
				return
			case errors.Is(err, text.ErrEmpty):
			default:
				code = http.StatusInternalServerError
			}
			s.logf("%s: input %d: %v", id, i, err)
			typ := "invalid_request_error"
			if code == http.StatusInternalServerError {
				typ = "server_error"
			}
			apiError(w, code, typ, ecode, fmt.Sprintf("input %d: %v", i, err))
			return
		}
		tokens += e.Tokens
		if e.Truncated {
			truncated++
		}
		var v any = e.Vector
		if b64 {
			buf := make([]byte, 4*len(e.Vector))
			for j, x := range e.Vector {
				binary.LittleEndian.PutUint32(buf[4*j:], math.Float32bits(x))
			}
			v = base64.StdEncoding.EncodeToString(buf)
		}
		data[i] = map[string]any{"object": "embedding", "index": i, "embedding": v}
	}
	kind := "documents"
	if query {
		kind = "queries"
	}
	cut := ""
	if truncated > 0 {
		cut = fmt.Sprintf(", %d truncated", truncated)
	}
	s.logf("%s: %d %s, %d tokens in %.2fs%s", id, len(texts), kind, tokens, time.Since(t0).Seconds(), cut)
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list", "data": data, "model": emb.Name(),
		"usage": map[string]any{"prompt_tokens": tokens, "total_tokens": tokens},
	})
}

func (s *Server) logf(format string, args ...any) {
	if s.Log != nil {
		fmt.Fprintf(s.Log, time.Now().Format("15:04:05 ")+format+"\n", args...)
	}
}

func sse(w io.Writer, v any) {
	b, _ := json.Marshal(v)
	fmt.Fprintf(w, "data: %s\n\n", b)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func apiError(w http.ResponseWriter, code int, typ, ecode, msg string) {
	e := map[string]any{"message": msg, "type": typ}
	if ecode != "" {
		e["code"] = ecode
	}
	writeJSON(w, code, map[string]any{"error": e})
}
