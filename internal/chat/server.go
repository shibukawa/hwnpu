package chat

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
)

// Server serves an Engine with the OpenAI chat completions API: GET /v1/models and POST
// /v1/chat/completions, streamed (server-sent events) or not, with tools. The engine runs one
// request at a time; the others wait.
type Server struct {
	E *Engine
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
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "ok\n") })
	return mux
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	in := []string{"text"}
	if s.E.Images() {
		in = append(in, "image")
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id": s.E.Name, "object": "model", "created": 0, "owned_by": "hwnpu",
			"context_length": s.E.Context(),
			"architecture":   map[string]any{"input_modalities": in, "output_modalities": []string{"text"}},
		}},
	})
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
	var images []string
	for _, m := range req.Messages {
		images = append(images, m.Content.Images...)
	}
	ids, err := s.E.Tokens(r.Context(), prompt, images)
	if err != nil {
		code, typ := http.StatusInternalServerError, "server_error"
		if errors.Is(err, ErrImage) {
			code, typ = http.StatusBadRequest, "invalid_request_error"
		}
		apiError(w, code, typ, "", err.Error())
		return
	}
	g := Request{Prompt: ids, MaxTokens: req.MaxTokens, Sampling: s.Defaults, Seed: req.Seed}
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
