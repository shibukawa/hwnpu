// Package agent runs a conversation with a chat model (ai/text) that calls tools: each request
// is a turn, in which the model may call the tools, round after round, before it answers.
//
//	a := agent.New(chat, agent.Config{System: "...", Tools: []agent.Tool{clock, weather}})
//	answer, err := a.Turn(ctx, "明日の天気は？", agent.TurnOptions{Text: func(d string) { fmt.Print(d) }})
//
// A turn can start before its request is certain (a voice assistant starts on what was said
// at a pause, ai/agent/voice): nothing of its answer is passed on and no tool runs before its
// Gate is closed, while the model reads the request and writes; a turn cancelled before that
// leaves the conversation as it was.
//
// The conversation is kept short for a quick prompt: it is forgotten after ResetAfter without
// a request, and its oldest turns are dropped past MaxMessages. The system prompt and the tools
// do not change, so the model's cache holds them (Prime reads them ahead).
package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/ai/text"
)

// Model answers a conversation: *text.Chatter, or a fake in tests.
type Model interface {
	Reply(ctx context.Context, msgs []text.Message, o text.ChatOptions, fn func(text.Delta) bool) (text.Answer, error)
}

// Tool is a function the model may call.
type Tool struct {
	Name, Description string
	// Parameters are its arguments' JSON Schema ("" for none).
	Parameters string
	// Slow says that it takes a while (it goes to the network): TurnOptions.Tool is told, so
	// that a word of waiting can be said.
	Slow bool
	// Run does the call and returns what the model is told. An error is told too, as the
	// result ("失敗しました: ...", Config.Failed).
	Run func(ctx context.Context, args Args) (string, error)
}

// Args are a call's arguments, the JSON object the model wrote.
type Args map[string]any

// String is a string argument, trimmed ("" when absent or not a string).
func (a Args) String(k string) string {
	s, _ := a[k].(string)
	return strings.TrimSpace(s)
}

// Int is a number argument, rounded (0 when absent or not a number; a number in a string is
// read too, as small models write it).
func (a Args) Int(k string) int {
	switch v := a[k].(type) {
	case float64:
		return int(math.Round(v))
	case string:
		var f float64
		if _, err := fmt.Sscan(strings.TrimSpace(v), &f); err == nil {
			return int(math.Round(f))
		}
	}
	return 0
}

// Config says how the agent works. The zero values of its fields take their defaults.
type Config struct {
	// System is the system prompt.
	System string
	Tools  []Tool
	// Options are how the model answers (their Tools are the Tools'). The zero value answers
	// without thinking, in up to 256 tokens, with DefaultSampling.
	Options text.ChatOptions
	// ResetAfter is the time without a request after which the conversation is forgotten
	// (default 5 minutes; below 0, never).
	ResetAfter time.Duration
	// MaxMessages is the most messages of earlier turns kept (default 16).
	MaxMessages int
	// Rounds is the most answers of the model in a turn, its rounds of calls and its answer
	// (default 4); the last round is offered no tools, so that it answers.
	Rounds int
	// Failed is what the model is told before the error of a tool (default "失敗しました: ").
	Failed string
	// Log receives what the agent does: the model's stats, the calls (nil for none).
	Log func(format string, args ...any)
}

// DefaultSampling is how an agent's tokens are drawn when Config.Options gives none: nearly the
// most likely, for tool calls that parse.
var DefaultSampling = text.Sampling{Temperature: 0.2, TopK: 40, Repetition: 1.05}

// Agent is the conversation. It runs one turn at a time; another waits for the one before.
type Agent struct {
	m     Model
	cfg   Config
	opts  text.ChatOptions
	tools map[string]Tool

	mu      sync.Mutex
	history []text.Message // without the system prompt
	last    time.Time
}

// New is an agent of the model with the config.
func New(m Model, cfg Config) *Agent {
	if cfg.ResetAfter == 0 {
		cfg.ResetAfter = 5 * time.Minute
	}
	if cfg.MaxMessages <= 0 {
		cfg.MaxMessages = 16
	}
	if cfg.Rounds <= 0 {
		cfg.Rounds = 4
	}
	if cfg.Failed == "" {
		cfg.Failed = "失敗しました: "
	}
	if cfg.Log == nil {
		cfg.Log = func(string, ...any) {}
	}
	o := cfg.Options
	if o.Think == nil {
		no := false
		o.Think = &no
	}
	if o.MaxTokens == 0 {
		o.MaxTokens = 256
	}
	if o.Sampling == nil {
		s := DefaultSampling
		o.Sampling = &s
	}
	a := &Agent{m: m, cfg: cfg, tools: map[string]Tool{}}
	for _, t := range cfg.Tools {
		params := t.Parameters
		if params == "" {
			params = `{"type": "object", "properties": {}}`
		}
		o.Tools = append(o.Tools, text.Tool{Name: t.Name, Description: t.Description, Parameters: json.RawMessage(params)})
		a.tools[t.Name] = t
	}
	a.opts = o
	return a
}

// Reset forgets the conversation.
func (a *Agent) Reset() {
	a.mu.Lock()
	a.history = nil
	a.mu.Unlock()
}

// History is the conversation kept (without the system prompt).
func (a *Agent) History() []text.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]text.Message(nil), a.history...)
}

// Prime reads the system prompt, the tools and the conversation into the model's cache, so
// that the next request reads its own words alone: at the start, and after a reset, while
// nobody waits.
func (a *Agent) Prime(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	o := a.opts
	o.MaxTokens = 1
	_, err := a.m.Reply(ctx, append(a.messages(), text.Message{Role: "user", Text: "こんにちは"}), o, nil)
	return err
}

func (a *Agent) messages() []text.Message {
	return append([]text.Message{{Role: "system", Text: a.cfg.System}}, a.history...)
}

// TurnOptions are where a turn's answer goes. Any may be nil.
type TurnOptions struct {
	// Gate, when not nil, holds the turn's answer and tool calls until it is closed.
	Gate <-chan struct{}
	// Text gets the answer's text as the model writes it (after Gate).
	Text func(delta string)
	// Tool is called before a tool runs (after Gate), with whether it is Slow.
	Tool func(name, args string, slow bool)
}

// ErrRounds is a turn whose rounds ran out without an answer.
var ErrRounds = errors.New("agent: the model did not answer within the rounds allowed")

// Turn answers a request: the model's rounds of tool calls, then its answer, which is returned
// and kept in the conversation with the request and the calls. When ctx is cancelled (or a
// round fails), the conversation is as it was.
func (a *Agent) Turn(ctx context.Context, request string, o TurnOptions) (string, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.cfg.ResetAfter > 0 && !a.last.IsZero() && time.Since(a.last) > a.cfg.ResetAfter && len(a.history) > 0 {
		a.cfg.Log("the conversation is forgotten: %v without a request", time.Since(a.last).Round(time.Second))
		a.history = nil
	}
	gate := o.Gate
	if gate == nil {
		open := make(chan struct{})
		close(open)
		gate = open
	}
	confirmed := func() error {
		select {
		case <-gate:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	out := o.Text
	if out == nil {
		out = func(string) {}
	}
	held := newRelease(gate, out)
	defer held.close()
	msgs := append(a.messages(), text.Message{Role: "user", Text: request})
	added := msgs[len(msgs)-1:]
	for round := 1; round <= a.cfg.Rounds; round++ {
		opts := a.opts
		if round == a.cfg.Rounds {
			opts.Tools = nil
		}
		ans, err := a.m.Reply(ctx, msgs, opts, func(d text.Delta) bool {
			if d.Text != "" {
				held.add(d.Text)
			}
			return ctx.Err() == nil
		})
		if err != nil {
			return "", err
		}
		a.cfg.Log("model: %d tokens read, %d of them cached, %.2f s; %d written at %.1f tokens/s",
			ans.Stats.Prompt, ans.Stats.Reused, ans.Stats.Prefill.Seconds(), ans.Stats.Generated, ans.Stats.TokensPerSecond())
		if err := confirmed(); err != nil {
			return "", err
		}
		if len(ans.ToolCalls) == 0 {
			a.keep(append(added, text.Message{Role: "assistant", Text: ans.Text}))
			return ans.Text, nil
		}
		msg := text.Message{Role: "assistant", Text: ans.Text, ToolCalls: ans.ToolCalls}
		msgs, added = append(msgs, msg), append(added, msg)
		for _, tc := range ans.ToolCalls {
			res := a.call(ctx, tc, o.Tool)
			m := text.Message{Role: "tool", Text: res, ToolCallID: tc.ID, Name: tc.Name}
			msgs, added = append(msgs, m), append(added, m)
		}
		if err := ctx.Err(); err != nil {
			return "", err
		}
	}
	return "", ErrRounds
}

// call runs a tool call and returns what the model is told.
func (a *Agent) call(ctx context.Context, tc text.ToolCall, before func(name, args string, slow bool)) string {
	t, ok := a.tools[tc.Name]
	if !ok {
		return "そういう道具はありません: " + tc.Name
	}
	var args Args
	if strings.TrimSpace(tc.Arguments) != "" {
		if err := json.Unmarshal([]byte(tc.Arguments), &args); err != nil {
			return "引数が JSON ではありません: " + err.Error()
		}
	}
	if before != nil {
		before(tc.Name, tc.Arguments, t.Slow)
	}
	t0 := time.Now()
	res, err := t.Run(ctx, args)
	if err != nil {
		res = a.cfg.Failed + err.Error()
	}
	a.cfg.Log("tool %s %s: %s, %.2f s", tc.Name, tc.Arguments, oneLine(res, 80), time.Since(t0).Seconds())
	return res
}

// keep adds a turn's messages to the conversation, dropping the oldest turns past MaxMessages
// (whole turns: from a user's message).
func (a *Agent) keep(turn []text.Message) {
	a.history = append(a.history, turn...)
	a.last = time.Now()
	for len(a.history) > a.cfg.MaxMessages {
		i := 1
		for i < len(a.history) && a.history[i].Role != "user" {
			i++
		}
		a.history = a.history[i:]
	}
}

// oneLine is s on one line, cut to n letters.
func oneLine(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + "…"
	}
	return s
}

// release passes texts on once a gate is closed, in their order: those that come before it
// wait; after close (the turn ended or was cancelled), what waits is dropped.
type release struct {
	gate    <-chan struct{}
	mu      sync.Mutex
	open    bool
	waiting []string
	out     func(string)
	stop    chan struct{}
}

func newRelease(gate <-chan struct{}, out func(string)) *release {
	r := &release{gate: gate, out: out, stop: make(chan struct{})}
	go func() {
		select {
		case <-gate:
		case <-r.stop:
			return
		}
		r.mu.Lock()
		defer r.mu.Unlock()
		r.flush()
	}()
	return r
}

// flush passes on what waits, the gate being closed (r.mu held).
func (r *release) flush() {
	r.open = true
	for _, s := range r.waiting {
		r.out(s)
	}
	r.waiting = nil
}

func (r *release) add(s string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.open {
		r.out(s)
		return
	}
	r.waiting = append(r.waiting, s)
}

// close ends the release: what waits is passed on when the gate is closed (the goroutine may
// not have seen it yet), and dropped when it is not (the turn was cancelled).
func (r *release) close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.open {
		return
	}
	close(r.stop)
	select {
	case <-r.gate:
		r.flush()
	default:
		r.waiting = nil
		r.open = true
	}
	r.out = func(string) {}
}
