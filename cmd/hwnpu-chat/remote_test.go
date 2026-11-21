package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/daemon"
)

func TestWireMessages(t *testing.T) {
	r := "thought"
	msgs := []chat.Message{
		{Role: "system", Content: chat.Content{Text: "be brief"}},
		{Role: "user", Content: chat.Content{Text: chat.MediaMarker + "what is this?" + chat.MediaMarker,
			Media: []chat.Media{{Data: "data:image/png;base64,AA"}, {Audio: true, Data: "UklG"}}}},
		{Role: "assistant", Content: chat.Content{Text: "a cat"}, Reasoning: &r},
	}
	got := wireMessages(msgs)
	b, _ := json.Marshal(got)
	var back []chat.Message
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back[0].Content.Text != "be brief" || back[2].Content.Text != "a cat" || *back[2].Reasoning != "thought" {
		t.Fatalf("text messages: %s", b)
	}
	u := back[1].Content
	if len(u.Media) != 2 || u.Media[0].Audio || !u.Media[1].Audio || u.Media[0].Data != "data:image/png;base64,AA" || u.Media[1].Data != "UklG" {
		t.Fatalf("media: %+v", u.Media)
	}
	// the server's markers in the same places: image, text, clip
	if want := chat.MediaMarker + "what is this?" + chat.MediaMarker; u.Text != want {
		t.Fatalf("text %q, want %q", u.Text, want)
	}
	parts := got[1]["content"].([]map[string]any)
	var types []string
	for _, p := range parts {
		types = append(types, p["type"].(string))
	}
	if !reflect.DeepEqual(types, []string{"image_url", "text", "input_audio"}) {
		t.Fatalf("parts %v", types)
	}
}

// serveUnix serves h on a Unix domain socket for the test and returns a client of it.
func serveUnix(t *testing.T, h http.Handler) *daemon.Client {
	t.Helper()
	dir, err := os.MkdirTemp("/tmp", "hwnpuchat")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	socket := filepath.Join(dir, "s.sock")
	ln, err := daemon.ListenUnix(socket)
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: h}
	go srv.Serve(ln)
	t.Cleanup(func() { srv.Close() })
	return daemon.NewClient(socket)
}

func TestRemoteReply(t *testing.T) {
	var got map[string]any
	var budget string
	c := serveUnix(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		budget = r.Header.Get(daemon.ThinkBudgetHeader)
		b, _ := io.ReadAll(r.Body)
		json.Unmarshal(b, &got)
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"seed":42,"choices":[{"delta":{"role":"assistant","content":""}}]}`,
			`{"seed":42,"choices":[{"delta":{"reasoning_content":"hmm"}}]}`,
			`{"seed":42,"choices":[{"delta":{"content":"Hel"}}]}`,
			`{"seed":42,"choices":[{"delta":{"content":"lo"}}]}`,
			`{"seed":42,"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":12},"hwnpu":{"prefill_ms":300,"decode_ms":250,"drafted":4,"accepted":3,"think_cut":true}}}`,
		} {
			fmt.Fprintf(w, "data: %s\n\n", ev)
		}
		io.WriteString(w, "data: [DONE]\n\n")
	}))
	r := remote{c: c, info: daemon.ModelInfo{Name: "LFM3-1.7B"}}
	var p printer
	msg, st, err := r.reply(context.Background(), []chat.Message{{Role: "user", Content: chat.Content{Text: "hi"}}},
		turn{think: true, thinkBudget: 256, maxNew: 64, s: chat.Sampling{Temperature: 0.1, TopK: 50, Repetition: 1.05},
			tr: &chat.Translation{TargetLang: "ja"}}, &p)
	if err != nil {
		t.Fatal(err)
	}
	if msg.Content.Text != "Hello" || *msg.Reasoning != "hmm" || !p.thought || !p.answered {
		t.Fatalf("message %+v (%q), printer %+v", msg, *msg.Reasoning, p)
	}
	want := turnStats{prompt: 20, reused: 12, generated: 5, prefill: 300 * time.Millisecond, decode: 250 * time.Millisecond,
		thinkCut: true, drafted: 4, accepted: 3, seed: 42}
	if st != want {
		t.Fatalf("stats %+v, want %+v", st, want)
	}
	if budget != "256" || got["model"] != "LFM3-1.7B" || got["stream"] != true || got["max_tokens"] != float64(64) {
		t.Fatalf("request: budget %q, %v", budget, got)
	}
	kw := got["chat_template_kwargs"].(map[string]any)
	if kw["enable_thinking"] != true || kw["target_lang"] != "ja" {
		t.Fatalf("kwargs %v", kw)
	}
}

func TestRemoteReplyError(t *testing.T) {
	c := serveUnix(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"the prompt does not fit","type":"invalid_request_error"}}`)
	}))
	var p printer
	_, _, err := remote{c: c, info: daemon.ModelInfo{Name: "m"}}.reply(context.Background(), nil, turn{}, &p)
	if err == nil || err.Error() != "the prompt does not fit" {
		t.Fatalf("error %v", err)
	}
}

func TestNotThroughDaemon(t *testing.T) {
	var c ChatCmd
	// kong's defaults
	c.Idle, c.Draft, c.MTP, c.Sparse, c.Model = 5*time.Minute, 8, chat.MTPAuto, true, "LFM3-1.7B"
	if got := c.notThroughDaemon(); len(got) != 0 {
		t.Fatalf("defaults: %v", got)
	}
	c.Format, c.Draft, c.Model = "lut4", 4, "./export"
	if got := c.notThroughDaemon(); !reflect.DeepEqual(got, []string{"--format", "--draft", "--model with an export's directory"}) {
		t.Fatalf("given: %v", got)
	}
}
