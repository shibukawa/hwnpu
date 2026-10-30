// hwnpu-agent is a coding agent on the Neural Engine: a chat model (LFM3-1.7B) with four tools,
// read, edit, write and bash, in the loop of Pi (pi.dev) written in Go on hwnpu-chat's engine.
// Its system prompt and tool declarations take well under 1K tokens of the 16K context (OpenCode's
// take 7.4K), and it makes no side requests, so a task starts at once.
//
//	hwnpu-agent "fix the bug in the add function of main.go"
//	hwnpu-agent -C ~/src/demo -yes "add a mul function and print mul(2, 3)"
//	hwnpu-agent
//
// With a task on the command line it runs that task and exits (0 when the model answered, 1
// otherwise); without one it takes tasks in the terminal, one conversation (/reset starts
// over, /exit quits, Ctrl-C stops the task at hand). The agent works in -C (the current
// directory): paths are relative to it, commands run in it, and it writes no file outside it.
// AGENTS.md there, if any, goes into the system prompt. Before a command or a change to a file
// it asks on the terminal, unless -yes or the command matches -allow (e.g. -allow "go *,ls*");
// without a terminal and without -yes it runs nothing else.
//
// The model must read a file before edit or write changes it, its thinking is cut at
// -think-budget tokens and the answer asked without thinking, a call it makes twice is not run
// again, and an answer it repeats ends the task.
//
// The model is -use, prepared like hwnpu-chat's (downloaded and converted into -dir on the
// first run, with a context of -context tokens). -tools offers fewer tools, -system replaces
// the opening of the system prompt, and -dump writes each request and answer as JSON.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/agent"
	"github.com/shibukawa/hwnpu/internal/chat"
)

func main() {
	use := flag.String("use", "LFM3-1.7B", "the chat model: a name that hwnpu-chat -list shows (one with tool calls), or a Hugging Face repository OWNER/NAME")
	dir := flag.String("dir", chat.ModelsDir(), "where the models are kept, each in a directory of its name")
	context_ := flag.Int("context", chat.DefaultContext, "tokens of the conversation and the answer together (the KV cache of the model's programs)")
	pkgs := flag.String("package", "", "programs converted by hwnpu-convert instead, comma separated: .mlpackage or out.parts.json, with the same -cache and different -seq")
	config := flag.String("config", "", "config.json of the model (default: in its directory)")
	weights := flag.String("weights", "", "checkpoint the token embeddings are read from (default: in the model's directory)")
	tok := flag.String("tokenizer", "", "tokenizer.json (default: next to config.json)")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory)")
	workDir := flag.String("C", ".", "the working directory of the agent")
	tools := flag.String("tools", strings.Join(agent.Names, ","), "the tools to offer, comma separated")
	system := flag.String("system", "", "a file whose text replaces the opening of the system prompt")
	yes := flag.Bool("yes", false, "run commands and change files without asking")
	think := flag.Bool("think", true, "let the model think before each answer")
	keep := flag.Bool("keep-thinking", false, "keep the thinking of earlier rounds of a task in the conversation (costs context)")
	thinkBudget := flag.Int("think-budget", 1024, "most tokens of thinking before an answer; past it the answer is asked without thinking (-1: no limit)")
	allow := flag.String("allow", "", "commands that run without asking, comma separated, with * for any text, e.g. \"go *,ls*,cat *\" (each part of a command joined with && or ; must match)")
	steps := flag.Int("steps", 30, "most rounds of one task")
	maxNew := flag.Int("n", 2048, "most tokens of one answer; also the tokens kept free when old tool results are folded to fit the context")
	temp := flag.Float64("temp", 0.1, "sampling temperature; 0 picks the most likely token")
	topK := flag.Int("top-k", 50, "sample from the k most likely tokens")
	topP := flag.Float64("top-p", 0, "sample from the most likely tokens up to this probability (0: off)")
	rep := flag.Float64("rep", 1.05, "repetition penalty on the tokens of the answer (1: none)")
	dump := flag.String("dump", "", "write each request, its prompt and the generated text to this directory")
	flag.Parse()
	task := strings.TrimSpace(strings.Join(flag.Args(), " "))
	if *cacheDir == "" {
		if d, err := os.UserCacheDir(); err == nil {
			*cacheDir = filepath.Join(d, "hwnpu")
		}
	}
	work, err := filepath.Abs(*workDir)
	if err == nil {
		var info os.FileInfo
		if info, err = os.Stat(work); err == nil && !info.IsDir() {
			err = fmt.Errorf("-C %s is not a directory", *workDir)
		}
	}
	if err != nil {
		fail(err)
	}
	opening := ""
	if *system != "" {
		b, err := os.ReadFile(*system)
		if err != nil {
			fail(err)
		}
		opening = string(b)
	}
	instructions := ""
	if b, err := os.ReadFile(filepath.Join(work, "AGENTS.md")); err == nil {
		instructions = string(b)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	m, err := chat.Resolve(ctx, *use, *dir, os.Stderr)
	if err != nil {
		fail(err)
	}
	if m.Kind != chat.KindChat {
		fail(fmt.Errorf("%s is a %s model; the agent needs a chat model with tool calls", m.Name, m.Kind))
	}
	if len(m.Features) > 0 && !slices.Contains(m.Features, "tools") {
		fail(fmt.Errorf("%s has no tool calls; the agent needs a model with the tools feature (hwnpu-chat -list)", m.Name))
	}
	md := filepath.Join(*dir, m.Name)
	c := chat.Config{ModelConfig: *config, Weights: *weights}
	if *pkgs != "" {
		c.Packages, c.Name = strings.Split(*pkgs, ","), m.Name
		if c.ModelConfig == "" {
			c.ModelConfig = filepath.Join(md, "config.json")
		}
		if c.Weights == "" {
			c.Weights = filepath.Join(md, "model.safetensors")
		}
	} else if c, err = chat.Prepare(ctx, chat.Setup{Dir: md, Repo: m.Repo, Context: *context_, Log: os.Stderr}); err != nil {
		fail(err)
	}
	c.Tokenizer, c.CacheDir, c.Log = *tok, *cacheDir, os.Stderr
	e, err := chat.Load(ctx, c)
	if err != nil {
		fail(err)
	}
	defer e.Close()
	stop()

	in := bufio.NewReader(os.Stdin)
	cfg := agent.Config{
		Dir: work, Tools: strings.Split(*tools, ","), System: opening, Instructions: instructions,
		Think: *think, KeepThinking: *keep, ThinkBudget: *thinkBudget, Sampling: chat.Sampling{Temperature: *temp, TopK: *topK, TopP: *topP, Repetition: *rep},
		MaxTokens: *maxNew, Reserve: *maxNew, Steps: *steps, Dump: *dump, Log: os.Stderr,
	}
	if *allow != "" {
		cfg.AllowList = strings.Split(*allow, ",")
	}
	switch {
	case *yes:
	case terminal(os.Stdin):
		cfg.Allow = func(tool, what string) bool {
			fmt.Fprintf(os.Stderr, "%s%s %s%s [y/N] ", bold, tool, what, reset)
			line, _ := in.ReadString('\n')
			line = strings.ToLower(strings.TrimSpace(line))
			return line == "y" || line == "yes"
		}
	default:
		cfg.Allow = func(string, string) bool { return false }
		fmt.Fprintln(os.Stderr, "no terminal: the agent runs no command and changes no file without -yes")
	}
	s, err := agent.New(e, cfg)
	if err != nil {
		fail(err)
	}
	if n, err := s.Count(ctx); err == nil {
		fmt.Fprintf(os.Stderr, "%s[%s, context %d tokens, %d for the system prompt and the tools]%s\n", dim, e.Name, e.Context(), n, reset)
	}
	if task != "" {
		if err := run(s, task); err != nil {
			fail(err)
		}
		return
	}
	fmt.Fprintf(os.Stderr, "%s/reset starts a new conversation, /exit (or Ctrl-D) quits; Ctrl-C stops the task%s\n", dim, reset)
	for {
		text, ok := read(in)
		if !ok {
			return
		}
		switch text {
		case "":
			continue
		case "/exit", "/quit":
			return
		case "/reset":
			s.Reset()
			fmt.Fprintln(os.Stderr, "[conversation cleared]")
			continue
		}
		if err := run(s, text); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
		}
	}
}

// run gives the agent one task, showing what it does.
func run(s *agent.Session, task string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var sent chat.Stream
	thought, answered := false, false
	ev := agent.Events{
		Delta: func(reasoning, content string) {
			dr, dc := sent.Next(reasoning, content)
			if dr != "" {
				fmt.Print(dim + dr + reset)
				thought = true
			}
			if dc != "" {
				if !answered && thought {
					fmt.Print("\n\n")
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
			fmt.Print(dim + preview(result, 5) + reset + "\n")
			sent, thought, answered = chat.Stream{}, false, false
		},
		Note: func(text string) {
			if thought || answered {
				fmt.Println()
			}
			fmt.Fprintf(os.Stderr, "%s[%s]%s\n", dim, text, reset)
			sent, thought, answered = chat.Stream{}, false, false
		},
		Done: func(c chat.Completion) {
			looped := ""
			if c.Looped {
				looped = ", stopped repeating itself"
			}
			fmt.Fprintf(os.Stderr, "%s[prompt %d tokens, %d new in %.1fs; %d tokens at %.1f tokens/s%s]%s\n", dim,
				c.Stats.Prompt, c.Stats.Prompt-c.Reused, c.Prefill.Seconds(), c.Stats.Generated, float64(c.Stats.Generated)/max(c.Decode.Seconds(), 1e-9), looped, reset)
		},
	}
	answer, err := s.Run(ctx, task, ev)
	if answered || answer != "" {
		fmt.Println()
	}
	switch {
	case errors.Is(err, context.Canceled):
		fmt.Fprintln(os.Stderr, "[stopped]")
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

// read prompts for and reads the next task. A line ending in \ continues on the next line. It
// returns false at the end of the input.
func read(in *bufio.Reader) (string, bool) {
	fmt.Print("\n> ")
	var lines []string
	for {
		line, err := in.ReadString('\n')
		if err != nil && line == "" {
			fmt.Println()
			return "", false
		}
		line = strings.TrimRight(line, "\r\n")
		if strings.HasSuffix(line, `\`) {
			lines = append(lines, strings.TrimSuffix(line, `\`))
			fmt.Print(". ")
			continue
		}
		lines = append(lines, line)
		return strings.TrimSpace(strings.Join(lines, "\n")), true
	}
}

func terminal(f *os.File) bool {
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "hwnpu-agent:", err)
	os.Exit(1)
}

const (
	dim   = "\x1b[2m"
	bold  = "\x1b[1m"
	reset = "\x1b[0m"
)
