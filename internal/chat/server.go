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

// Server serves an Engine with the OpenAI chat completions API (POST /v1/chat/completions,
// streamed with server-sent events or not, with tools) and an Embedder with the embeddings API
// (POST /v1/embeddings); GET /v1/models lists both. Either may be missing. Each model runs one
// request at a time; the others wait.
type Server struct {
	E   *Engine   // the chat model (nil for none)
	Emb *Embedder // the embedding model (nil for none)
	// Think is the default of thinking before the answer; a request changes it with
	// reasoning_effort (none, minimal and low turn it off, medium and high on) or
	// chat_template_kwargs.enable_thinking.
	Think bool
	// Defaults is the sampling of requests that do not say.
	Defaults Sampling
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
	if s.E != nil {
		data = append(data, map[string]any{
			"id": s.E.Name, "object": "model", "created": 0, "owned_by": "hwnpu",
			"context_length": s.E.Context(),
		})
	}
	if s.Emb != nil {
		data = append(data, map[string]any{
			"id": s.Emb.Name, "object": "model", "created": 0, "owned_by": "hwnpu",
			"context_length": s.Emb.MaxTokens(), "dimensions": s.Emb.Dim(),
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
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
	if s.E == nil {
		apiError(w, http.StatusNotFound, "invalid_request_error", "model_not_found", "no chat model is loaded (start hwnpu-chat with -use chat or -use chat,embed)")
		return
	}
	if s.Emb != nil && req.Model == s.Emb.Name && req.Model != s.E.Name {
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", req.Model+" is an embedding model: use /v1/embeddings, or the chat model "+s.E.Name)
		return
	}
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
	prompt, err := Render(req.Messages, req.Tools, Options{NoThink: !think, KeepThinking: req.ChatTemplateKwargs.PreserveThinking})
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
	g := Request{Prompt: s.E.Tok.Encode(prompt), MaxTokens: req.MaxTokens, Sampling: s.Defaults, Seed: req.Seed}
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
	chunk := func(delta map[string]any, finish any) map[string]any {
		return map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": s.E.Name,
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
	stats, err := s.E.Generate(r.Context(), g, func(sofar []int32) (bool, error) {
		gen = sofar
		a = Parse(s.E.Tok.Decode(sofar, false), think, req.Tools, false)
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
	text := s.E.Tok.Decode(gen, false)
	a = Parse(text, think, req.Tools, true)
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
			"id": id, "object": "chat.completion", "created": created, "model": s.E.Name,
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
		sse(w, map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": s.E.Name, "choices": []any{}, "usage": usage})
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
	if s.Emb == nil {
		apiError(w, http.StatusNotFound, "invalid_request_error", "model_not_found", "no embedding model is loaded (start hwnpu-chat with -use embed or -use chat,embed)")
		return
	}
	if s.E != nil && req.Model == s.E.Name && req.Model != s.Emb.Name {
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", req.Model+" is a chat model: use /v1/chat/completions, or the embedding model "+s.Emb.Name)
		return
	}
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
	if req.Dimensions != 0 && req.Dimensions != s.Emb.Dim() {
		apiError(w, http.StatusBadRequest, "invalid_request_error", "", fmt.Sprintf("dimensions must be %d, the model's", s.Emb.Dim()))
		return
	}

	id := fmt.Sprintf("embd-%d-%d", time.Now().Unix(), s.n.Add(1))
	t0 := time.Now()
	data := make([]map[string]any, len(texts))
	tokens, truncated := 0, 0
	for i, text := range texts {
		e, err := s.Emb.Embed(r.Context(), text, query, req.Truncate)
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
		"object": "list", "data": data, "model": s.Emb.Name,
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
