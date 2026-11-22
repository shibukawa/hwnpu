package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/text"
	"github.com/shibukawa/hwnpu/internal/agent"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
)

// AgentCmd is "hwnpu-chat agent": a coding agent, a chat model with four tools, read, edit, write
// and bash, in the loop of Pi (pi.dev) written in Go on the chat engine. Its system prompt and
// tool declarations take well under 1K tokens (OpenCode's take 7.4K), and it makes no side
// requests, so a task starts at once.
//
// With a task on the command line it runs that task and exits (0 when the model answered, 1
// otherwise); without one it takes tasks in the terminal, one conversation (/reset starts over,
// /exit quits, Ctrl-C stops the task at hand). The agent works in -C (the current directory):
// paths are relative to it, commands run in it, and it writes no file outside it. AGENTS.md
// there, if any, goes into the system prompt. Before a command or a change to a file it asks on
// the terminal, unless --yes or the command matches --allow (e.g. --allow "go *,ls*"); without a
// terminal and without --yes it runs nothing else.
//
// The model must read a file before edit or write changes it, its thinking is closed at
// --think-budget tokens so that it answers, a call it makes twice is not run again, and an answer
// it repeats ends the task.
//
// A fifth tool, search, looks facts up, which a 1.7B model recalls badly: --tools search alone
// makes an assistant that answers from what it finds, and --search-first searches before the
// model answers each task, with a query it writes for the task, so that the search is not left
// to its judgment. --search picks Wikipedia (no key), the Brave Search API (BRAVE_API_KEY) or a
// SearXNG instance for the web, and fetch reads a page of the results as text.
//
//	hwnpu-chat agent --tools search,fetch --search-first --no-think
type AgentCmd struct {
	Task        []string `arg:"" optional:"" help:"The task; without it, tasks are taken in the terminal."`
	Model       string   `short:"m" default:"LFM3-1.7B" placeholder:"NAME|EXPORT" help:"The chat model, one with tool calls: a name of hwnpu-chat list, chat for the one recommended for the machine's memory, a family with or without a tag (gemma4, gemma4:e4b), a Hugging Face repository OWNER/NAME, or an export's directory (AMD)."`
	WorkDir     string   `short:"C" name:"C" default:"." type:"existingdir" placeholder:"DIR" help:"The working directory of the agent."`
	Tools       string   `default:"read,edit,write,bash" help:"The tools to offer, comma separated: read, edit, write, bash, search, fetch (search alone makes an assistant that looks facts up)."`
	Search      string   `default:"wikipedia" help:"What the search tool searches: wikipedia, brave (the key in BRAVE_API_KEY), or the URL of a SearXNG instance such as http://localhost:8888."`
	SearchFirst bool     `help:"Search before the model answers each task, with a query it writes for the task, so that the facts are in front of it (search must be among --tools)."`
	System      string   `type:"existingfile" placeholder:"FILE" help:"A file whose text replaces the opening of the system prompt."`
	Yes         bool     `short:"y" help:"Run commands and change files without asking."`
	Allow       string   `placeholder:"PATTERNS" help:"Commands that run without asking, comma separated, with * for any text, e.g. \"go *,ls*,cat *\" (each part of a command joined with && or ; must match)."`
	Keep        bool     `name:"keep-thinking" help:"Keep the thinking of earlier rounds of a task in the conversation (costs context)."`
	Steps       int      `default:"30" help:"Most rounds of one task."`
	N           int      `short:"n" default:"2048" placeholder:"TOKENS" help:"Most tokens of one answer; also the tokens kept free when old tool results are folded to fit the context."`
	Dump        string   `type:"path" placeholder:"DIR" help:"Write each request, its prompt and the generated text to this directory."`
	loadFlags
	convertFlags
	samplingFlags
}

// Run loads the model and runs the task, or takes tasks in the terminal.
func (c *AgentCmd) Run(ctx context.Context, stop context.CancelFunc) error {
	st, cat, err := c.loadFlags.prepare(ctx)
	if err != nil {
		return err
	}
	work, err := filepath.Abs(c.WorkDir)
	if err != nil {
		return err
	}
	opening := ""
	if c.System != "" {
		b, err := os.ReadFile(c.System)
		if err != nil {
			return err
		}
		opening = string(b)
	}
	instructions := ""
	if b, err := os.ReadFile(filepath.Join(work, "AGENTS.md")); err == nil {
		instructions = string(b)
	}
	searcher, err := agent.NewSearcher(c.Search, os.Getenv("BRAVE_API_KEY"))
	if err != nil {
		return err
	}
	tools := split(c.Tools)
	for _, t := range tools {
		if !slices.Contains(agent.Names, t) {
			return fmt.Errorf("--tools: no tool %q (%s)", t, strings.Join(agent.Names, ", "))
		}
	}
	if c.SearchFirst && !slices.Contains(tools, "search") {
		return errors.New("--search-first needs search among --tools")
	}

	cm, err := chatModelOf(cat, c.Model, &c.convertFlags, nil)
	if err != nil {
		return err
	}
	st.Definitions = cm.defs
	app := ai.NewApp("hwnpu-chat", cm.chat)
	app.Store = st
	rt, err := open(ctx, app, options(&c.loadFlags, &c.convertFlags, "agent")...)
	if err != nil {
		return err
	}
	defer rt.Close()
	e, err := cm.chat.Get(ctx, rt)
	if err != nil {
		return err
	}
	prepared(rt)
	cm.keep()
	if f := e.Features(); len(f) > 0 && !slices.Contains(f, "tools") {
		return fmt.Errorf("%s has no tool calls; the agent needs a model with the tools feature (hwnpu-chat list)", cm.base)
	}
	stop()

	budget := c.ThinkBudget
	if budget <= 0 {
		budget = -1 // no limit
	}
	ed := console.NewEditor("agent")
	cfg := agent.Config{
		Dir: work, Tools: tools, System: opening, Instructions: instructions,
		Think: c.think(e), KeepThinking: c.Keep, ThinkBudget: budget, Sampling: chatSampling(c.sampling()),
		MaxTokens: c.N, Reserve: c.N, Steps: c.Steps, Dump: c.Dump, Log: os.Stderr,
		Search: searcher, SearchFirst: c.SearchFirst, AllowList: split(c.Allow),
	}
	switch {
	case c.Yes:
	case console.Terminal(os.Stdin):
		cfg.Allow = func(tool, what string) bool {
			line, _ := ed.Ask(fmt.Sprintf("%s%s %s%s [y/N] ", bold, tool, what, reset))
			line = strings.ToLower(strings.TrimSpace(line))
			return line == "y" || line == "yes"
		}
	default:
		cfg.Allow = func(string, string) bool { return false }
		fmt.Fprintln(os.Stderr, "no terminal: the agent runs no command and changes no file without --yes")
	}
	s, err := agent.New(agentModel{e}, cfg)
	if err != nil {
		return err
	}
	if n, err := s.Count(context.Background()); err == nil {
		fmt.Fprintf(logw(), "%s, context %d tokens, %d for the system prompt and the tools\n", e.Name(), e.Context(), n)
	}
	if task := strings.TrimSpace(strings.Join(c.Task, " ")); task != "" {
		return runTask(s, task)
	}
	console.Notef("/reset starts a new conversation, /exit (or Ctrl-D) quits; Ctrl-C stops the task, ↑ ↓ go through what was typed\n")
	for {
		text, ok := read(ed)
		if !ok {
			return nil
		}
		switch text {
		case "":
			continue
		case "/exit", "/quit":
			return nil
		case "/reset":
			s.Reset()
			console.Notef("[conversation cleared]\n")
			continue
		}
		if err := runTask(s, text); err != nil {
			printError(err)
		}
	}
}

// runTask gives the agent one task, showing what it does: the answer and the calls on standard
// output, the thinking and the tools' results dim, the statistics in gray on standard error.
func runTask(s *agent.Session, task string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var sent chat.Stream
	thought, answered := false, false
	ev := agent.Events{
		Delta: func(reasoning, content string) {
			dr, dc := sent.Next(reasoning, content)
			if dr != "" {
				fmt.Fprint(os.Stderr, dimmed(dr))
				thought = true
			}
			if dc != "" {
				if !answered && thought {
					fmt.Fprint(os.Stderr, "\n\n")
				}
				fmt.Print(dc)
				answered = true
			}
		},
		Call: func(name, what string) {
			if thought || answered {
				fmt.Println()
			}
			if name == "bash" {
				fmt.Printf("%s$ %s%s\n", bold, what, reset)
			} else {
				fmt.Printf("%s%s %s%s\n", bold, name, what, reset)
			}
		},
		Result: func(name, result string) {
			fmt.Fprintln(os.Stderr, dimmed(preview(result, 5)))
			sent, thought, answered = chat.Stream{}, false, false
		},
		Note: func(text string) {
			if thought || answered {
				fmt.Println()
			}
			console.Notef("[%s]\n", text)
			sent, thought, answered = chat.Stream{}, false, false
		},
		Done: func(c chat.Completion) {
			if thought || answered {
				fmt.Println() // the statistics on a line of their own
			}
			sent, thought, answered = chat.Stream{}, false, false
			looped := ""
			if c.Looped {
				looped = ", stopped repeating itself"
			}
			if c.Drafted > 0 {
				looped += fmt.Sprintf("; %d of %d drafts accepted", c.Accepted, c.Drafted)
			}
			if c.Cache > 0 {
				looped += fmt.Sprintf("; cache %d", c.Cache)
			}
			console.Notef("[prompt %d tokens, %d new in %.1fs; %d tokens at %.1f tokens/s%s]\n",
				c.Stats.Prompt, c.Stats.Prompt-c.Reused, c.Prefill.Seconds(), c.Stats.Generated, float64(c.Stats.Generated)/max(c.Decode.Seconds(), 1e-9), looped)
		},
	}
	_, err := s.Run(ctx, task, ev)
	if thought || answered {
		fmt.Println()
	}
	switch {
	case errors.Is(err, context.Canceled):
		console.Notef("[stopped]\n")
		return nil
	case errors.Is(err, agent.ErrCut):
		return fmt.Errorf("%w; -n allows longer answers", err)
	case errors.Is(err, agent.ErrLoop):
		return fmt.Errorf("%w; what it answered so far is above", err)
	}
	return err
}

// preview is the first lines of a tool result, with a count of the rest.
func preview(s string, n int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) <= n {
		return strings.Join(lines, "\n")
	}
	return strings.Join(lines[:n], "\n") + fmt.Sprintf("\n… %d more lines", len(lines)-n)
}

// bold and reset mark the agent's calls (the escape sequences go to a terminal only: the calls
// are on standard output, which may be a file).
var bold, reset = func() (string, string) {
	if console.Colors(os.Stdout) {
		return "\x1b[1m", "\x1b[0m"
	}
	return "", ""
}()

// agentModel is a chat model as the agent's loop asks it (agent.Model): the conversation, the
// tools and the answer in the engine's types, through text.Chatter.
type agentModel struct{ c *text.Chatter }

func (m agentModel) Context() int { return m.c.Context() }

func (m agentModel) Count(ctx context.Context, msgs []chat.Message, tools []chat.Tool, o chat.Options) (int, error) {
	tm, tt, err := textConversation(msgs, tools)
	if err != nil {
		return 0, err
	}
	think := !o.NoThink
	return m.c.Count(ctx, tm, text.ChatOptions{Tools: tt, Think: &think, KeepThinking: o.KeepThinking})
}

func (m agentModel) Complete(ctx context.Context, msgs []chat.Message, tools []chat.Tool, o chat.Options, s chat.Sampling, maxTokens, thinkBudget int, emit func(partial chat.Answer) (stop bool)) (chat.Completion, error) {
	var c chat.Completion
	tm, tt, err := textConversation(msgs, tools)
	if err != nil {
		return c, err
	}
	think := !o.NoThink
	co := text.ChatOptions{Tools: tt, Think: &think, KeepThinking: o.KeepThinking, MaxTokens: maxTokens, ThinkBudget: thinkBudget,
		Sampling: &text.Sampling{Temperature: s.Temperature, TopK: s.TopK, TopP: s.TopP, Repetition: s.Repetition}}
	if maxTokens <= 0 {
		co.MaxTokens = text.Unlimited
	}
	if thinkBudget <= 0 {
		co.ThinkBudget = -1 // none
	}
	var fn func(text.Delta) bool
	if emit != nil {
		var sofar chat.Answer // what the answer has added up to
		fn = func(d text.Delta) bool {
			sofar.Reasoning += d.Reasoning
			sofar.Content += d.Text
			return !emit(sofar)
		}
	}
	a, err := m.c.Reply(ctx, tm, co, fn)
	c.Reasoning, c.Content, c.Rendered, c.Raw = a.Reasoning, a.Text, a.Prompt, a.Raw
	for _, tc := range a.ToolCalls {
		var call chat.ToolCall
		call.ID, call.Type, call.Function.Name, call.Function.Arguments = tc.ID, "function", tc.Name, tc.Arguments
		c.ToolCalls = append(c.ToolCalls, call)
	}
	st := a.Stats
	c.Stats = chat.Stats{Prompt: st.Prompt, Reused: st.Reused, Generated: st.Generated, Restored: st.Restored, Finish: a.Finish,
		Looped: st.Looped, ThinkCut: st.ThinkCut, ThinkLooped: st.ThinkLooped, Drafted: st.Drafted, Accepted: st.Accepted, Calls: st.Calls,
		Cache: st.Cache, Prefill: st.Prefill, Decode: st.Decode, Seed: st.Seed}
	if c.Finish == "tool_calls" {
		c.Finish = "stop"
	}
	return c, err
}

// textConversation is the agent's conversation and tools as text.Chatter takes them.
func textConversation(msgs []chat.Message, tools []chat.Tool) ([]text.Message, []text.Tool, error) {
	out := make([]text.Message, len(msgs))
	for i, m := range msgs {
		tm := text.Message{Role: m.Role, Text: m.Content.Text, ToolCallID: m.ToolCallID, Name: m.Name}
		if m.Reasoning != nil {
			tm.Reasoning = *m.Reasoning
		}
		for _, tc := range m.ToolCalls {
			tm.ToolCalls = append(tm.ToolCalls, text.ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
		}
		out[i] = tm
	}
	var tt []text.Tool
	for _, t := range tools {
		raw, err := json.Marshal(t)
		if err != nil {
			return nil, nil, err
		}
		tool, err := text.ParseTool(raw)
		if err != nil {
			return nil, nil, err
		}
		tt = append(tt, tool)
	}
	return out, tt, nil
}

// chatSampling is s as the agent's loop takes it.
func chatSampling(s text.Sampling) chat.Sampling {
	return chat.Sampling{Temperature: s.Temperature, TopK: s.TopK, TopP: s.TopP, Repetition: s.Repetition}
}
