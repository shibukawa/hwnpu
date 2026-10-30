package main

import (
	"encoding/json"
	"net"
	"testing"
)

// listenFree gives a port from 30000 below the ephemeral range, and another while the first is
// taken.
func TestListenFree(t *testing.T) {
	var ls []net.Listener
	defer func() {
		for _, l := range ls {
			l.Close()
		}
	}()
	seen := map[int]bool{}
	for i := 0; i < 5; i++ {
		l, err := listenFree()
		if err != nil {
			t.Fatal(err)
		}
		ls = append(ls, l)
		a := l.Addr().(*net.TCPAddr)
		if !a.IP.IsLoopback() || a.Port < portLow || a.Port >= portHigh || seen[a.Port] {
			t.Errorf("listening on %v", a)
		}
		seen[a.Port] = true
	}
}

// The config names the server as the provider hwnpu, its model with the context as the limit,
// and that model as the one to use.
func TestOpenCodeConfig(t *testing.T) {
	b, err := openCodeConfig("http://127.0.0.1:31234/v1", "LFM3-1.7B", 16320)
	if err != nil {
		t.Fatal(err)
	}
	var c struct {
		Provider map[string]struct {
			NPM     string `json:"npm"`
			Options struct {
				BaseURL string `json:"baseURL"`
			} `json:"options"`
			Models map[string]struct {
				ToolCall bool `json:"tool_call"`
				Limit    struct {
					Context, Output int
				} `json:"limit"`
			} `json:"models"`
		} `json:"provider"`
		Model string `json:"model"`
	}
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	p := c.Provider["hwnpu"]
	m, ok := p.Models["LFM3-1.7B"]
	if p.NPM != "@ai-sdk/openai-compatible" || p.Options.BaseURL != "http://127.0.0.1:31234/v1" || !ok || !m.ToolCall ||
		m.Limit.Context != 16320 || m.Limit.Output != 4080 || c.Model != "hwnpu/LFM3-1.7B" {
		t.Errorf("config %s", b)
	}
}
