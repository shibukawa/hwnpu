package chat

import (
	"encoding/json"
	"testing"
)

// TestServerThinks checks the default of thinking: the server's, off for a model that answers
// without thinking by default (Qwen3.5-0.8B and 2B), and what a request asks over both.
func TestServerThinks(t *testing.T) {
	for _, c := range []struct {
		name            string
		server, noThink bool
		body            string
		want            bool
	}{
		{"server default", true, false, `{}`, true},
		{"server off", false, false, `{}`, false},
		{"model without thinking", true, true, `{}`, false},
		{"model without thinking, enable_thinking", true, true, `{"chat_template_kwargs":{"enable_thinking":true}}`, true},
		{"model without thinking, reasoning_effort high", true, true, `{"reasoning_effort":"high"}`, true},
		{"reasoning_effort low", true, false, `{"reasoning_effort":"low"}`, false},
		{"enable_thinking over reasoning_effort", true, false, `{"reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":false}}`, false},
	} {
		var req chatRequest
		if err := json.Unmarshal([]byte(c.body), &req); err != nil {
			t.Fatal(err)
		}
		s := &Server{Think: c.server}
		if got := s.thinks(&Engine{NoThink: c.noThink}, &req); got != c.want {
			t.Errorf("%s: thinks %v, want %v", c.name, got, c.want)
		}
	}
	for _, name := range []string{"Qwen3.5-0.8B", "Qwen3.5-2B"} {
		if m, ok := Lookup(name); !ok || !m.NoThink {
			t.Errorf("%s: in the catalog %v, NoThink %v; want a model that does not think by default", name, ok, m.NoThink)
		}
	}
	if m, _ := Lookup("Qwen3.5-4B"); m.NoThink {
		t.Errorf("Qwen3.5-4B thinks by default, as its chat template")
	}
}
