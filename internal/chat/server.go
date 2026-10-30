package chat

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
)

// Server serves chat models (Engine) with the OpenAI chat completions API (POST
// /v1/chat/completions, streamed with server-sent events or not, with tools) and embedding
// models (Embedder) with the embeddings API (POST /v1/embeddings); GET /v1/models lists them. A
// request goes to the model of its endpoint's kind that its model field names, or to the only
// one of that kind whatever it names. Each model runs one request at a time; the others wait.
type Server struct {
	Chats     []*Engine
	Embedders []*Embedder
	// Think is the default of thinking before the answer; a request changes it with
	// reasoning_effort (none, minimal and low turn it off, medium and high on) or
	// chat_template_kwargs.enable_thinking.
	Think bool
	// Defaults is the sampling of requests that do not say.
	Defaults Sampling
	// ThinkBudget is the most tokens a model thinks before it is made to answer (0: no limit);
	// see Engine.ThinkLimit.
	ThinkBudget int
	// Log receives a line per request (nil for none).
	Log io.Writer
	// Dump, when set, is a directory that receives each request, its prompt and the generated
	// text (ID.json), for finding out what a client sends.
	Dump string

	n atomic.Int64
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

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	data := []map[string]any{}
	for _, e := range s.Chats {
		in := []string{"text"}
		if e.Images() {
			in = append(in, "image")
		}
		if e.Audio() {
			in = append(in, "audio")
		}
		data = append(data, describe(e.Name, KindChat, map[string]any{"context_length": e.Context(),
			"architecture": map[string]any{"input_modalities": in, "output_modalities": []string{"text"}}}))
	}
	for _, e := range s.Embedders {
		data = append(data, describe(e.Name, KindEmbed, map[string]any{"context_length": e.MaxTokens(), "dimensions": e.Dim()}))
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

// describe is a model's entry in /v1/models: with its kind, and its features (capabilities)
// when it is one of the catalog.
func describe(name string, kind Kind, more map[string]any) map[string]any {
	m := map[string]any{"id": name, "object": "model", "created": 0, "owned_by": "hwnpu", "kind": kind}
	if c, ok := Lookup(name); ok {
		m["capabilities"] = append([]string{}, c.Features...)
	}
	for k, v := range more {
		m[k] = v
	}
	return m
}

func (s *Server) names() (chats, embedders []string) {
	for _, e := range s.Chats {
		chats = append(chats, e.Name)
	}
	for _, e := range s.Embedders {
		embedders = append(embedders, e.Name)
	}
	return chats, embedders
}

// route picks the model a request names among those of the endpoint's kind (names): the one of
// that name, or the only one. A name of a model of the other kind (others) is an error, so that
// a client set up with the wrong model finds out. It returns the index, or -1 with the error's
// status and message.
func route(model string, names, others []string, kind Kind, other string) (int, int, string) {
	for i, n := range names {
		if strings.EqualFold(model, n) {
			return i, 0, ""
		}
	}
	for _, n := range others {
		if strings.EqualFold(model, n) {
			return -1, http.StatusBadRequest, fmt.Sprintf("%s is not a %s model: use %s", n, kind, other)
		}
	}
	switch len(names) {
	case 0:
		return -1, http.StatusNotFound, fmt.Sprintf("no %s model is loaded: add one to -use (hwnpu-chat -list shows them)", kind)
	case 1:
		return 0, 0, ""
	}
	return -1, http.StatusNotFound, fmt.Sprintf("no %s model %q: the model is one of %s", kind, model, strings.Join(names, ", "))
}

type chatRequest struct {
	Model         string          `json:"model"`
	Messages      []Message       `json:"messages"`
	Tools         []Tool          `json:"tools"`
	ToolChoice    json.RawMessage `json:"tool_choice"`
	Stream        bool            `json:"stream"`
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
	ChatTemplateKwargs  struct {
		EnableThinking   *bool `json:"enable_thinking"`
		PreserveThinking bool  `json:"preserve_thinking"`
	} `json:"chat_template_kwargs"`
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
	i, code, msg := route(req.Model, chats, embedders, KindChat, "/v1/embeddings")
	if i < 0 {
		apiError(w, code, "invalid_request_error", "model_not_found", msg)
		return
	}
	e := s.Chats[i]
	if len(req.Messages) == 0 {
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", "messages is empty")
		return
	}
	if string(req.ToolChoice) == `"none"` {
		req.Tools = nil
	}
	think := s.Think
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
	prompt, thinking, err := e.Template.Render(req.Messages, req.Tools, Options{NoThink: !think, KeepThinking: req.ChatTemplateKwargs.PreserveThinking})
	if err != nil {
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", err.Error())
		return
	}
	var stops []string
	if len(req.Stop) > 0 && string(req.Stop) != "null" {
		var one string
		if json.Unmarshal(req.Stop, &one) == nil {
			stops = []string{one}
		} else if err := json.Unmarshal(req.Stop, &stops); err != nil {
			apiError(w, http.StatusBadRequest, "invalid_request_error", "", "stop is neither a string nor a list of strings")
			return
		}
	}
	var media []Media
	for _, m := range req.Messages {
		media = append(media, m.Content.Media...)
	}
	ids, err := e.Tokens(r.Context(), prompt, media)
	if err != nil {
		code, typ := http.StatusInternalServerError, "server_error"
		if errors.Is(err, ErrImage) {
			code, typ = http.StatusBadRequest, "invalid_request_error"
		}
		apiError(w, code, typ, "", err.Error())
		return
	}
	g := Request{Prompt: ids, MaxTokens: req.MaxTokens, Sampling: s.Defaults, Seed: req.Seed, Think: e.ThinkLimit(s.ThinkBudget, thinking)}
	if req.MaxCompletionTokens > 0 {
		g.MaxTokens = req.MaxCompletionTokens
	}
	if req.Temperature != nil {
		g.Temperature = *req.Temperature
	}
	if req.TopP != nil {
		g.TopP = *req.TopP
	}
	if req.TopK != nil {
		g.TopK = *req.TopK
	}
	if req.RepetitionPenalty != nil {
		g.Repetition = *req.RepetitionPenalty
	}

	id := fmt.Sprintf("chatcmpl-%d-%d", time.Now().Unix(), s.n.Add(1))
	created := time.Now().Unix()
	// a long prompt on a large cache takes minutes before its first token
	s.logf("%s: %d tokens of prompt, %d messages, %d tools", id, len(ids), len(req.Messages), len(req.Tools))
	chunk := func(delta map[string]any, finish any) map[string]any {
		return map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": e.Name,
			"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}},
		}
	}
	var flush func()
	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		rc := http.NewResponseController(w)
		flush = func() { rc.Flush() }
		sse(w, chunk(map[string]any{"role": "assistant", "content": ""}, nil))
		flush()
	}

	var a Answer
	var sent Stream
	hold := 0 // content held back while it may become a stop string
	for _, st := range stops {
		hold = max(hold, len(st)-1)
	}
	cut := func(c string) (string, bool) {
		for _, st := range stops {
			if i := strings.Index(c, st); st != "" && i >= 0 {
				return c[:i], true
			}
		}
		return c, false
	}
	var gen []int32
	stats, err := e.Generate(r.Context(), g, func(sofar []int32) (bool, error) {
		gen = sofar
		a = e.Template.Parse(e.Tok.Decode(sofar, false), thinking, req.Tools, false)
		c, stopped := cut(a.Content)
		if !req.Stream {
			return stopped, nil
		}
		if !stopped && !a.Calling {
			c = validPrefix(c[:max(0, len(c)-hold)])
		}
		dr, dc := sent.Next(a.Reasoning, c)
		if dr != "" {
			sse(w, chunk(map[string]any{"reasoning_content": dr}, nil))
		}
		if dc != "" {
			sse(w, chunk(map[string]any{"content": dc}, nil))
		}
		flush()
		return stopped, nil
	})
	if err != nil {
		code := http.StatusInternalServerError
		typ, ecode := "server_error", ""
		if errors.Is(err, ErrContext) {
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
	text := e.Tok.Decode(gen, false)
	a = e.Template.Parse(text, thinking, req.Tools, true)
	a.Content, _ = cut(a.Content)
	if s.Dump != "" {
		b, _ := json.MarshalIndent(map[string]any{"request": json.RawMessage(body), "prompt": prompt, "generated": text, "stats": stats}, "", " ")
		if err := os.WriteFile(filepath.Join(s.Dump, id+".json"), b, 0o644); err != nil {
			s.logf("%s: %v", id, err)
		}
	}
	finish := stats.Finish
	if len(a.ToolCalls) > 0 {
		finish = "tool_calls"
	}
	looped := ""
	if stats.Looped {
		looped = " (stopped repeating itself)"
	}
	if stats.ThinkCut {
		looped += fmt.Sprintf(" (thinking cut at %d tokens)", s.ThinkBudget)
	}
	s.logf("%s: prompt %d tokens (%d reused) in %.1fs, %d tokens in %.1fs (%.1f ms/token), %s%s",
		id, stats.Prompt, stats.Reused, stats.Prefill.Seconds(), stats.Generated, stats.Decode.Seconds(),
		float64(stats.Decode.Milliseconds())/float64(max(stats.Generated, 1)), finish, looped)
	usage := map[string]any{
		"prompt_tokens": stats.Prompt, "completion_tokens": stats.Generated, "total_tokens": stats.Prompt + stats.Generated,
		"prompt_tokens_details": map[string]any{"cached_tokens": stats.Reused},
	}
	if !req.Stream {
		msg := map[string]any{"role": "assistant", "content": a.Content}
		if think {
			msg["reasoning_content"] = a.Reasoning
		}
		if len(a.ToolCalls) > 0 {
			msg["tool_calls"] = a.ToolCalls
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"id": id, "object": "chat.completion", "created": created, "model": e.Name,
			"choices": []map[string]any{{"index": 0, "message": msg, "finish_reason": finish}},
			"usage":   usage,
		})
		return
	}
	dr, dc := sent.Next(a.Reasoning, a.Content)
	if dr != "" {
		sse(w, chunk(map[string]any{"reasoning_content": dr}, nil))
	}
	if dc != "" {
		sse(w, chunk(map[string]any{"content": dc}, nil))
	}
	if len(a.ToolCalls) > 0 {
		var calls []map[string]any
		for i, tc := range a.ToolCalls {
			calls = append(calls, map[string]any{"index": i, "id": tc.ID, "type": tc.Type, "function": tc.Function})
		}
		sse(w, chunk(map[string]any{"tool_calls": calls}, nil))
	}
	sse(w, chunk(map[string]any{}, finish))
	if req.StreamOptions.IncludeUsage {
		sse(w, map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": e.Name, "choices": []any{}, "usage": usage})
	}
	io.WriteString(w, "data: [DONE]\n\n")
	flush()
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
	i, code, msg := route(req.Model, embedders, chats, KindEmbed, "/v1/chat/completions")
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
	for i, text := range texts {
		e, err := emb.Embed(r.Context(), text, query, req.Truncate)
		if err != nil {
			code, ecode := http.StatusBadRequest, ""
			switch {
			case errors.Is(err, ErrTooLong):
				ecode = "context_length_exceeded"
			case r.Context().Err() != nil:
				return
			case errors.Is(err, ErrEmpty):
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
		"object": "list", "data": data, "model": emb.Name,
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
