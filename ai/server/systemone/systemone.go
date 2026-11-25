// Package systemone serves a decision model (text.Decide) with the System One API of llama.cpp's
// server, as an http.Handler (bonsaiai-chat systemone --listen uses it):
//
//	POST /v1/systemone  typed questions about a state (text.Decider.Decide)
//	GET  /v1/models     the model, as a decision model (output_modalities ["decisions"])
//	GET  /health        ok
package systemone

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/shibukawa/bonsaiai/ai/text"
)

// Server serves a decision model.
type Server struct {
	D *text.Decider
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

// ServeHTTP serves the API (Handler).
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.Handler().ServeHTTP(w, r) }

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id": s.D.Name(), "object": "model", "created": 0, "owned_by": "bonsaiai",
			"architecture": map[string]any{"input_modalities": []string{"text"}, "output_modalities": []string{"decisions"}},
		}},
	})
}

func (s *Server) decide(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		apiError(w, http.StatusBadRequest, err.Error())
		return
	}
	d, err := s.D.Decide(r.Context(), body)
	if err == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(append(d.JSON, '\n'))
		s.logf("systemone: %d questions, %d tokens, %.2fs", d.Questions, d.InputTokens, d.Took.Seconds())
		return
	}
	switch {
	case errors.Is(err, text.ErrBadRequest):
		apiError(w, http.StatusBadRequest, err.Error())
	case errors.Is(err, text.ErrUnsupported):
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
