package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Decider answers requests: a loaded Model.
type Decider interface {
	Decide(ctx context.Context, r *Request) (*Response, error)
}

// Server serves POST /v1/systemone, and GET /v1/models with the model as a decision model
// (output_modalities ["decisions"]), as llama.cpp's server does.
type Server struct {
	M    Decider
	Name string
	// Log receives a line per request (nil for none).
	Log io.Writer
}

// Handler returns the API's routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("POST /v1/systemone", s.decide)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "ok\n") })
	return mux
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id": s.Name, "object": "model", "created": 0, "owned_by": "hwnpu",
			"architecture": map[string]any{"input_modalities": []string{"text"}, "output_modalities": []string{"decisions"}},
		}},
	})
}

func (s *Server) decide(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		apiError(w, http.StatusBadRequest, err.Error())
		return
	}
	req, err := ParseRequest(body)
	if err == nil {
		var res *Response
		if res, err = s.M.Decide(r.Context(), req); err == nil {
			writeJSON(w, http.StatusOK, res)
			s.logf("systemone: %d questions, %d tokens, %.2fs", len(req.Questions), res.InputTokens, time.Since(t0).Seconds())
			return
		}
	}
	var badReq *BadRequest
	var unsup *Unsupported
	switch {
	case errors.As(err, &badReq):
		apiError(w, http.StatusBadRequest, err.Error())
	case errors.As(err, &unsup):
		apiError(w, http.StatusNotImplemented, err.Error())
	default:
		apiError(w, http.StatusInternalServerError, err.Error())
	}
	s.logf("systemone: %v", err)
}

func (s *Server) logf(format string, args ...any) {
	if s.Log != nil {
		fmt.Fprintf(s.Log, time.Now().Format("15:04:05 ")+format+"\n", args...)
	}
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

// apiError writes an error the way llama.cpp's server does.
func apiError(w http.ResponseWriter, code int, msg string) {
	typ := map[int]string{
		http.StatusBadRequest:     "invalid_request_error",
		http.StatusNotImplemented: "not_supported_error",
	}[code]
	if typ == "" {
		typ = "server_error"
	}
	writeJSON(w, code, map[string]any{"error": map[string]any{"code": code, "message": msg, "type": typ}})
}
