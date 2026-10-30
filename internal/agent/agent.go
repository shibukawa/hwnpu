// Package agent is a coding agent of four tools, read, edit, write and bash, on a chat model
// called in this process (chat.Engine): the loop of Pi (pi.dev) without its harness. The
// model gets a system prompt of a few lines and the four declarations, well under 1K tokens,
// calls tools one round at a time, and ends a task by answering without a call. The
// conversation is kept within the model's context by folding old tool results, not by asking
// the model for a summary.
//
// A small model needs guards more than rules: edit and write want the file read first, the
// thinking is cut at a budget and the answer asked without it, a call made again is not run,
// and an answer repeated is the end of the task.
package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// Model answers a conversation: *chat.Engine, or a fake in tests.
type Model interface {
	Complete(ctx context.Context, msgs []chat.Message, tools []chat.Tool, o chat.Options, s chat.Sampling, maxTokens int, emit func(partial chat.Answer) (stop bool)) (chat.Completion, error)
	Count(ctx context.Context, msgs []chat.Message, tools []chat.Tool, o chat.Options) (int, error)
	Context() int
}

// Config says how the agent works.
type Config struct {
	// Dir is the working directory: paths are relative to it, commands run in it, and files
	// outside it are not written. Default: the process's.
	Dir string
	// Tools names the tools to offer, of Names. Default: all four.
	Tools []string
	// System replaces the opening of the system prompt (DefaultSystem).
	System string
	// Instructions are the project's (AGENTS.md), appended to the system prompt.
	Instructions string
	// Think lets the model think before each answer; KeepThinking keeps the thinking of earlier
	// rounds of a task in the conversation (it costs context).
	Think, KeepThinking bool
	// ThinkBudget is the most tokens of thinking before an answer: past it the generation
	// stops and the answer is asked without thinking. Default 1024; negative for no limit.
	ThinkBudget int
	Sampling    chat.Sampling
	// MaxTokens is the most tokens of one answer (0: until the cache is full).
	MaxTokens int
	// Steps is the most rounds (answers of the model) of one task. Default 30.
	Steps int
	// Reserve is the tokens kept free for the answer when the conversation is folded to fit the
	// context. Default 2048.
	Reserve int
	// Keep is how many of the latest tool results are never folded. Default 4.
	Keep int
	// ReadLines and ReadBytes limit what read returns; TailLines and TailBytes what bash
	// returns (its last lines). Defaults 200 lines and 8 KB, 100 lines and 8 KB.
	ReadLines, ReadBytes, TailLines, TailBytes int
	// Timeout is the default time a command may take. Default 60 seconds.
	Timeout time.Duration
	// Allow is asked before a tool that writes or runs anything (name and what it does); nil
	// allows everything.
	Allow func(tool, what string) bool
	// AllowList are patterns of commands that run without asking Allow, such as "go *" or
	// "ls*" (see Allowed).
	AllowList []string
	// Dump is a directory where each request is written as JSON ("" for none).
	Dump string
	// Log receives warnings (nil for none).
	Log io.Writer
}

func (c *Config) defaults() error {
	if c.Dir == "" {
		c.Dir = "."
	}
	dir, err := filepath.Abs(c.Dir)
	if err != nil {
		return err
	}
	c.Dir = dir
	def := func(p *int, v int) {
		if *p <= 0 {
			*p = v
		}
	}
	def(&c.Steps, 30)
	def(&c.Reserve, 2048)
	def(&c.Keep, 4)
	def(&c.ReadLines, 200)
	def(&c.ReadBytes, 8192)
	def(&c.TailLines, 100)
	def(&c.TailBytes, 8192)
	if c.ThinkBudget == 0 {
		c.ThinkBudget = 1024
	}
	if c.Timeout <= 0 {
		c.Timeout = time.Minute
	}
	return nil
}

// Events are called as a task runs (any may be nil).
type Events struct {
	// Delta gets the reasoning and the content of the answer so far, as it is generated.
	Delta func(reasoning, content string)
	// Call is called before a tool runs, with its name and what it does.
	Call func(name, what string)
	// Result is called after a tool ran, with what the model is told.
	Result func(name, result string)
	// Done is called after each answer of the model, with its stats.
	Done func(c chat.Completion)
	// Note is called when the agent steps in, such as cutting the thinking.
	Note func(text string)
}

var (
	// ErrSteps says that the task used all its rounds.
	ErrSteps = errors.New("agent: the task did not end within the rounds allowed")
	// ErrCut says that the last answer was cut by MaxTokens or the context.
	ErrCut = errors.New("agent: the answer was cut short")
	// ErrLoop says that the model made the same call three times in a row.
	ErrLoop = errors.New("agent: the model repeats the same call")
)

// nudge is the result of a call the model just made, the second time.
const nudge = "Same call as the one before; the result would be the same. If the task is done, answer without calling a tool."

// closing asks for the answer when the rounds run out or the model repeats itself: a request
// without tools, so that it can only answer.
const closing = "Stop using tools. Answer now with a short summary of what you did and whether it worked."

// Session is a conversation with the agent: a task and the tasks after it share it until
// Reset.
type Session struct {
	cfg    Config
	m      Model
	tools  []*Tool
	decls  []chat.Tool
	msgs   []chat.Message
	folded map[int]bool    // tool messages shortened to fit the context
	seen   map[string]bool // files read (or written) in this conversation, by absolute path
	start  time.Time
	n      int // requests, for Dump
}

// New makes a session of the model with the config.
func New(m Model, cfg Config) (*Session, error) {
	if err := cfg.defaults(); err != nil {
		return nil, err
	}
	s := &Session{cfg: cfg, m: m, start: time.Now()}
	tools, err := s.buildTools()
	if err != nil {
		return nil, err
	}
	s.tools = tools
	for _, t := range tools {
		s.decls = append(s.decls, t.Decl)
	}
	s.Reset()
	return s, nil
}

// Reset forgets the conversation.
func (s *Session) Reset() {
	s.msgs = []chat.Message{{Role: "system", Content: chat.Content{Text: Prompt(s.cfg, s.tools)}}}
	s.folded = map[int]bool{}
	s.seen = map[string]bool{}
}

// System is the system prompt.
func (s *Session) System() string { return s.msgs[0].Content.Text }

// Messages is the conversation so far.
func (s *Session) Messages() []chat.Message { return slices.Clone(s.msgs) }

// Tools are the declarations sent with each request.
func (s *Session) Tools() []chat.Tool { return s.decls }

// Count is the tokens the conversation takes.
func (s *Session) Count(ctx context.Context) (int, error) {
	return s.m.Count(ctx, s.msgs, s.decls, chat.Options{NoThink: !s.cfg.Think})
}

// Run gives the model a task and runs its tool calls until it answers without one, and returns
// that answer. The task and the rounds stay in the conversation, except when the context is
// cancelled: then they are dropped.
//
// A small model does not always stop: it writes its final answer and adds a call anyway, round
// after round, or makes one call again and again. An answer much like the one before it ends
// the task without its calls; a call made a second time is not run but answered with a hint to
// stop, and a third time, like the last round, is followed by one request without tools for the
// answer (ErrLoop or ErrSteps when even that gives none).
func (s *Session) Run(ctx context.Context, task string, ev Events) (string, error) {
	start := len(s.msgs)
	s.msgs = append(s.msgs, chat.Message{Role: "user", Content: chat.Content{Text: task}})
	var prev, prevCall string // the content and the call of the round before
	same := 0                 // rounds the call has been made in a row
	for range s.cfg.Steps {
		if err := s.fit(ctx); err != nil {
			return "", err
		}
		c, err := s.complete(ctx, s.decls, ev)
		if err != nil {
			if ctx.Err() != nil {
				s.msgs = s.msgs[:start]
			}
			return "", err
		}
		msg := chat.Message{Role: "assistant", Content: chat.Content{Text: c.Content}, ToolCalls: c.ToolCalls}
		if s.cfg.KeepThinking && c.Reasoning != "" {
			r := c.Reasoning
			msg.Reasoning = &r
		}
		if len(c.ToolCalls) > 0 && similar(c.Content, prev) {
			msg.ToolCalls = nil // the answer is final; the calls are a habit
			s.msgs = append(s.msgs, msg)
			return c.Content, nil
		}
		s.msgs = append(s.msgs, msg)
		if len(c.ToolCalls) == 0 {
			if c.Finish == "length" {
				return c.Content, ErrCut
			}
			return c.Content, nil
		}
		prev = c.Content
		if sig := signature(c.ToolCalls); sig == prevCall {
			same++
		} else {
			prevCall, same = sig, 0
		}
		for _, tc := range c.ToolCalls {
			if ctx.Err() != nil {
				s.msgs = s.msgs[:start]
				return "", ctx.Err()
			}
			out := nudge
			if same == 0 {
				out = s.call(ctx, tc, ev)
			}
			s.msgs = append(s.msgs, chat.Message{Role: "tool", Name: tc.Function.Name, ToolCallID: tc.ID, Content: chat.Content{Text: out}})
		}
		if same >= 2 {
			return s.close(ctx, ev, ErrLoop)
		}
	}
	return s.close(ctx, ev, ErrSteps)
}

// complete asks the model for the next answer. The thinking is cut at ThinkBudget tokens
// (or by MaxTokens): then the answer is asked again without thinking.
func (s *Session) complete(ctx context.Context, tools []chat.Tool, ev Events) (chat.Completion, error) {
	noThink := !s.cfg.Think
	for {
		thought := 0
		emit := func(a chat.Answer) bool {
			if ev.Delta != nil {
				ev.Delta(a.Reasoning, a.Content)
			}
			if a.Thinking {
				thought++
				return s.cfg.ThinkBudget > 0 && thought > s.cfg.ThinkBudget
			}
			return false
		}
		c, err := s.m.Complete(ctx, s.msgs, tools, chat.Options{NoThink: noThink}, s.cfg.Sampling, s.cfg.MaxTokens, emit)
		if err != nil {
			return c, err
		}
		s.dump(c)
		if ev.Done != nil {
			ev.Done(c)
		}
		if c.Thinking && !noThink {
			if ev.Note != nil {
				ev.Note(fmt.Sprintf("thinking cut after %d tokens; answering without it", c.Stats.Generated))
			}
			noThink = true
			continue
		}
		return c, nil
	}
}

// close asks for the answer without tools; cause is the error when there is none.
func (s *Session) close(ctx context.Context, ev Events, cause error) (string, error) {
	s.msgs = append(s.msgs, chat.Message{Role: "user", Content: chat.Content{Text: closing}})
	c, err := s.complete(ctx, nil, ev)
	if err != nil {
		return "", err
	}
	s.msgs = append(s.msgs, chat.Message{Role: "assistant", Content: chat.Content{Text: c.Content}})
	if strings.TrimSpace(c.Content) == "" {
		return "", cause
	}
	return c.Content, nil
}

// signature tells the calls of a round apart.
func signature(calls []chat.ToolCall) string {
	var sb strings.Builder
	for _, tc := range calls {
		sb.WriteString(tc.Function.Name + "(" + tc.Function.Arguments + ")\n")
	}
	return sb.String()
}

// similar reports whether two answers say the same: both are at least 20 words long and share
// at least 70% of their words (the model rewords its final answer a little each round).
func similar(a, b string) bool {
	wa, wb := strings.Fields(a), strings.Fields(b)
	if len(wa) < 20 || len(wb) < 20 {
		return false
	}
	set := map[string]bool{}
	for _, w := range wa {
		set[w] = true
	}
	both := map[string]bool{}
	for _, w := range wb {
		if set[w] {
			both[w] = true
		}
	}
	all := len(set)
	for _, w := range wb {
		if !set[w] {
			set[w] = true
			all++
		}
	}
	return len(both)*10 >= all*7
}

// call runs one tool call and returns what the model is told.
func (s *Session) call(ctx context.Context, tc chat.ToolCall, ev Events) string {
	args := map[string]any{}
	if a := strings.TrimSpace(tc.Function.Arguments); a != "" && a != "null" {
		if err := json.Unmarshal([]byte(a), &args); err != nil {
			return "The arguments are not a JSON object: " + err.Error()
		}
	}
	var t *Tool
	for _, u := range s.tools {
		if u.Name == tc.Function.Name {
			t = u
		}
	}
	if t == nil {
		var names []string
		for _, u := range s.tools {
			names = append(names, u.Name)
		}
		return fmt.Sprintf("There is no tool %q. The tools are %s.", tc.Function.Name, strings.Join(names, ", "))
	}
	what := t.Describe(args)
	if ev.Call != nil {
		ev.Call(t.Name, what)
	}
	ask := t.Writes && s.cfg.Allow != nil
	if t.Name == "bash" && Allowed(what, s.cfg.AllowList) {
		ask = false
	}
	var out string
	if ask && !s.cfg.Allow(t.Name, what) {
		out = "Not allowed by the user."
	} else {
		out = t.Run(ctx, args)
	}
	if ev.Result != nil {
		ev.Result(t.Name, out)
	}
	return out
}

// fit folds old tool results until the conversation leaves Reserve tokens of the context for
// the answer: each becomes its first line and a count of the rest, the oldest first, the
// latest Keep untouched. A conversation that does not fit even so is an error (chat.ErrContext).
func (s *Session) fit(ctx context.Context) error {
	limit := s.m.Context() - s.cfg.Reserve
	n, err := s.Count(ctx)
	if err != nil || n <= limit {
		return err
	}
	var idx []int
	for i, m := range s.msgs {
		if m.Role == "tool" && !s.folded[i] {
			idx = append(idx, i)
		}
	}
	if len(idx) <= s.cfg.Keep {
		idx = nil
	} else {
		idx = idx[:len(idx)-s.cfg.Keep]
	}
	for _, i := range idx {
		s.msgs[i].Content.Text = fold(s.msgs[i].Content.Text)
		s.folded[i] = true
		if n, err = s.Count(ctx); err != nil || n <= limit {
			return err
		}
	}
	return fmt.Errorf("%w: the conversation takes %d tokens, %d are allowed (/reset starts over)", chat.ErrContext, n, limit)
}

// fold keeps the first line of a tool result and counts the rest.
func fold(text string) string {
	first, rest, ok := strings.Cut(text, "\n")
	if !ok {
		return text
	}
	return fmt.Sprintf("%s\n[... %d more lines omitted]", first, strings.Count(rest, "\n")+1)
}

// dump writes the request and the answer to Config.Dump.
func (s *Session) dump(c chat.Completion) {
	if s.cfg.Dump == "" {
		return
	}
	s.n++
	b, _ := json.MarshalIndent(map[string]any{"messages": s.msgs, "prompt": c.Rendered, "generated": c.Raw, "stats": c.Stats}, "", " ")
	name := filepath.Join(s.cfg.Dump, fmt.Sprintf("agent-%d-%03d.json", s.start.Unix(), s.n))
	if err := os.WriteFile(name, b, 0o644); err != nil && s.cfg.Log != nil {
		fmt.Fprintln(s.cfg.Log, "agent:", err)
	}
}
