package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/shibukawa/bonsaiai/ai/text"
	"github.com/shibukawa/bonsaiai/internal/chat"
)

// A server without models answers the requests it can without one: the list, the health, and
// the errors of requests to no model or of bad requests.
func TestServerErrors(t *testing.T) {
	srv := httptest.NewServer((&Server{}).Handler())
	defer srv.Close()
	get := func(path string) (int, string) {
		r, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer r.Body.Close()
		b, _ := io.ReadAll(r.Body)
		return r.StatusCode, string(b)
	}
	if code, body := get("/v1/models"); code != 200 || !strings.Contains(body, `"data":[]`) {
		t.Errorf("models: %d %s", code, body)
	}
	if code, body := get("/health"); code != 200 || body != "ok\n" {
		t.Errorf("health: %d %q", code, body)
	}
	for _, c := range []struct {
		path, body string
		code       int
		want       string
	}{
		{"/v1/chat/completions", `{"model": "x", "messages": [{"role": "user", "content": "hi"}]}`, 404, "no chat model is loaded"},
		{"/v1/chat/completions", `{"messages": [}`, 400, "invalid character"},
		{"/v1/chat/completions", `{"messages": [{"role": "user", "content": 3}]}`, 400, "neither a string nor a list of parts"},
		{"/v1/embeddings", `{"input": "hi"}`, 404, "no embedding model is loaded"},
	} {
		r, err := http.Post(srv.URL+c.path, "application/json", strings.NewReader(c.body))
		if err != nil {
			t.Fatal(err)
		}
		var e struct {
			Error struct{ Message, Type string }
		}
		json.NewDecoder(r.Body).Decode(&e)
		r.Body.Close()
		if r.StatusCode != c.code || !strings.Contains(e.Error.Message, c.want) {
			t.Errorf("%s %s: %d %q, want %d %q", c.path, c.body, r.StatusCode, e.Error.Message, c.code, c.want)
		}
	}
}

// route picks the model named, the only one whatever is named, and refuses a model of the
// other kind.
func TestRoute(t *testing.T) {
	names, others := []string{"LFM3-1.7B", "Qwen3.5-0.8B"}, []string{"plamo-embedding-1b"}
	if i, _, _ := route("qwen3.5-0.8b", names, others, chat.KindChat, "/v1/embeddings"); i != 1 {
		t.Errorf("by name: %d", i)
	}
	if i, _, _ := route("gpt-4", names[:1], others, chat.KindChat, "/v1/embeddings"); i != 0 {
		t.Errorf("the only one: %d", i)
	}
	if i, code, msg := route("plamo-embedding-1b", names, others, chat.KindChat, "/v1/embeddings"); i >= 0 || code != 400 || !strings.Contains(msg, "/v1/embeddings") {
		t.Errorf("of the other kind: %d %d %s", i, code, msg)
	}
	if i, code, _ := route("gpt-4", names, others, chat.KindChat, "/v1/embeddings"); i >= 0 || code != 404 {
		t.Errorf("none of several: %d %d", i, code)
	}
}

// A request's messages and fields become the text package's messages and options.
func TestRequest(t *testing.T) {
	var req chatRequest
	body := `{"messages": [
		{"role": "system", "content": "be brief"},
		{"role": "user", "content": [{"type": "text", "text": "what is "}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}, {"type": "input_audio", "input_audio": {"data": "UklG"}}]},
		{"role": "assistant", "content": null, "reasoning_content": "hm", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "look", "arguments": "{}"}}]},
		{"role": "tool", "tool_call_id": "c1", "name": "look", "content": "a dot"}],
		"tools": [{"type": "function", "function": {"name": "look", "parameters": {"type": "object"}}}],
		"max_tokens": 5, "temperature": 0, "top_k": 3, "stop": "END", "seed": 4,
		"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
		"chat_template_kwargs": {"target_lang": "ja", "glossary": [{"source": "dot", "target": "点"}]}}`
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatal(err)
	}
	var msgs []text.Message
	for _, m := range req.Messages {
		msgs = append(msgs, m.message())
	}
	if msgs[0].Text != "be brief" || len(msgs[1].Parts) != 3 || msgs[1].Parts[1].Media.URL != "data:image/png;base64,AA" || !msgs[1].Parts[2].Media.Audio {
		t.Errorf("messages %+v", msgs[:2])
	}
	if msgs[2].Reasoning != "hm" || len(msgs[2].ToolCalls) != 1 || msgs[2].ToolCalls[0].Name != "look" || msgs[3].ToolCallID != "c1" {
		t.Errorf("calls %+v", msgs[2:])
	}
	s := &Server{ThinkBudget: 100}
	o, err := s.options(&req, false)
	if err != nil {
		t.Fatal(err)
	}
	if o.MaxTokens != 5 || o.Sampling.Temperature != 0 || o.Sampling.TopK != 3 || o.Sampling.Repetition != text.DefaultSampling.Repetition ||
		len(o.Stop) != 1 || o.Seed != 4 || len(o.Tools) != 1 || o.Format == nil || string(o.Format.Schema) != `{"type": "object"}` ||
		o.Translation.To != "ja" || len(o.Translation.Glossary) != 1 || *o.Think || o.ThinkBudget != 100 {
		t.Errorf("options %+v", o)
	}
	req = chatRequest{ToolChoice: json.RawMessage(`"none"`), Tools: req.Tools}
	if o, _ := s.options(&req, true); len(o.Tools) != 0 || o.MaxTokens != text.Unlimited || o.Seed == 0 {
		t.Errorf("tool_choice none, no max_tokens, no seed: %+v", o)
	}
	req = chatRequest{Stop: json.RawMessage(`3`)}
	if _, err := s.options(&req, true); err == nil {
		t.Error("a stop of a number was taken")
	}
}
