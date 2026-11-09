// hwnpu-agent is a coding agent on the Neural Engine: a chat model (LFM3-1.7B) with four tools,
// read, edit, write and bash, in the loop of Pi (pi.dev) written in Go on hwnpu-chat's engine.
// Its system prompt and tool declarations take well under 1K tokens (OpenCode's take 7.4K), and it
// makes no side requests, so a task starts at once. The context is LFM3's longest, 262144 tokens,
// with the model split at its attention as hwnpu-chat does (-paged).
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
// The model must read a file before edit or write changes it, its thinking is closed at
// -think-budget tokens so that it answers, a call it makes twice is not run again, and an
// answer it repeats ends the task.
//
// A fifth tool, search, looks facts up, which a 1.7B model recalls badly: -tools search
// alone makes an assistant that answers from what it finds, and -search-first searches
// before the model answers each task, with a query it writes for the task, so that the
// search is not left to its judgment. -search picks Wikipedia (no key), the Brave Search
// API (BRAVE_API_KEY) or a SearXNG instance for the web, and fetch reads a page of the
// results as text.
//
//	hwnpu-agent -tools search,fetch -search-first -think=false
//
// The model is -use, prepared like hwnpu-chat's (downloaded and converted into -dir on the
// first run, with a context of -context tokens). -tools offers fewer tools, -system replaces
// the opening of the system prompt, and -dump writes each request and answer as JSON.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/agent"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
)

func main() {
	use := flag.String("use", "LFM3-1.7B", "the chat model: a name that hwnpu-chat -list shows (one with tool calls), chat for the one recommended for the machine's memory, a family with or without a tag (gemma4, gemma4:e4b), or a Hugging Face repository OWNER/NAME")
	dir := flag.String("dir", chat.ModelsDir(), "where the models are kept, each in a directory of its name")
	context_ := flag.Int("context", 0, "tokens of a prompt and its answer together; 0: the longest the model was trained for when it is split at its attention (-paged: 262144 for LFM3-1.7B and Qwen3.5, 131072 for Gemma 4, its cache growing with the conversation within a quarter of the memory), else 16320 (over 16320 a whole model's cache is in chunks of 16384, slower per token)")
	paged := flag.Int("paged", 0, "split a chat model at its full attention layers and attend over the cache in chunks of this many tokens, so that a step costs in proportion to the conversation, not to -context: 0 for 8192 where the model allows it (LFM3 and its VL, Audio and Omni checkpoints, Gemma 4, Qwen3.5), -1 never (then -tier applies)")
	sparse := flag.Bool("sparse", true, "past about 48K tokens of a model split at its attention (-paged), read 16384 keys of the cache exactly (32768 while reading a prompt), those of the 8-token buckets that weigh most, and every other bucket as its mean (two-level attention, a few times faster at 128K and over); false reads the whole cache on every call")
	pkgs := flag.String("package", "", "programs converted by hwnpu-convert instead, comma separated: .mlpackage, .xml (Intel) or out.parts.json, with the same -cache and different -seq")
	config := flag.String("config", "", "config.json of the model (default: in its directory)")
	weights := flag.String("weights", "", "checkpoint the token embeddings are read from (default: in the model's directory)")
	tok := flag.String("tokenizer", "", "tokenizer.json (default: next to config.json)")
	backend := flag.String("backend", "", "the NPU to convert for and run on: coreml (Apple's Neural Engine), levelzero or intel (Intel's NPU), openvino (OpenVINO, HWNPU_OV_DEVICE, for checking without the NPU) (default: HWNPU_BACKEND, else the NPU found)")
	format := flag.String("format", "", "weight format of the chat model's programs: int8, int4, int4-g128 or fp16 on Intel's NPU, lut4, lut6, lut8, int4, int8 or fp16 on the Neural Engine (default: the model's for the NPU, int8 for LFM3 on Intel's NPU)")
	codes := flag.String("codes", "", "directory of lookup tables chosen against the activations (tools/amd/q4_gptq.py --fmt-q4 lut6 ...) for the chat model's lutN weights; the programs are converted anew for them")
	formatFor_ := flag.String("format-for", "", "comma separated SUBSTRING=FORMAT: another weight format for the chat model's tensors whose names contain SUBSTRING, e.g. per_layer=lut8 (the last match wins); the programs are converted anew for them")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory)")
	promptDir := flag.String("prompt-cache", "", "keep the prompts read on disk, so that a prompt beginning like an earlier one is not read again after a restart (default: prompts in the compile cache; \"off\" for none)")
	workDir := flag.String("C", ".", "the working directory of the agent")
	tools := flag.String("tools", strings.Join(agent.DefaultTools, ","), "the tools to offer, comma separated: "+strings.Join(agent.Names, ", ")+" (search alone makes an assistant that looks facts up)")
	search := flag.String("search", "wikipedia", "what the search tool searches: wikipedia, brave (the key in BRAVE_API_KEY), or the URL of a SearXNG instance such as http://localhost:8888")
	searchFirst := flag.Bool("search-first", false, "search before the model answers each task, with a query it writes for the task, so that the facts are in front of it (search must be among -tools)")
	system := flag.String("system", "", "a file whose text replaces the opening of the system prompt")
	yes := flag.Bool("yes", false, "run commands and change files without asking")
	think := flag.Bool("think", true, "let the model think before each answer")
	keep := flag.Bool("keep-thinking", false, "keep the thinking of earlier rounds of a task in the conversation (costs context)")
	thinkBudget := flag.Int("think-budget", 1024, "most tokens the model thinks before it is made to answer (0: no limit); a small model may think for thousands")
	tier := flag.String("tier", chat.DefaultTiers, "caches shorter than -context to run on while the conversation is short, comma separated (a step costs in proportion to the cache); empty or none for none, auto for 4096 with a model in 4 bit tables and none with 6 or 8 bits (four such programs overflow the Neural Engine's compile cache and every start compiles them again)")
	draft := flag.Int("draft", 8, "tokens per call of the program that generates: a token and N-1 guesses checked at once (speculative decoding); 0 or 1 generates one token per call")
	mtp := flag.String("mtp", chat.MTPAuto, "auto: drafts also from the checkpoint's multi-token prediction layer when it has one (Qwen3.5); off: from the prompt alone")
	prefill := flag.String("prefill", chat.PrefillAuto, "a program of 64 tokens per call (128 for a model split at its attention) to read prompts beside the one that generates: auto (unless the two would not stay in the Neural Engine's compile cache of about 4 GiB, as Gemma 4 E4B's do not: then the one that generates reads the prompts, slower for long ones), on or off")
	allow := flag.String("allow", "", "commands that run without asking, comma separated, with * for any text, e.g. \"go *,ls*,cat *\" (each part of a command joined with && or ; must match)")
	steps := flag.Int("steps", 30, "most rounds of one task")
	maxNew := flag.Int("n", 2048, "most tokens of one answer; also the tokens kept free when old tool results are folded to fit the context")
	temp := flag.Float64("temp", 0.1, "sampling temperature; 0 picks the most likely token")
	topK := flag.Int("top-k", 50, "sample from the k most likely tokens")
	topP := flag.Float64("top-p", 0, "sample from the most likely tokens up to this probability (0: off)")
	rep := flag.Float64("rep", 1.05, "repetition penalty on the tokens of the answer (1: none)")
	dump := flag.String("dump", "", "write each request, its prompt and the generated text to this directory")
	ironDir := flag.String("iron", "", "run a model compiled by amd/IRON for AMD's NPU (tools/amd/iron_export_lm.py), this directory, in place of -use")
	idle := flag.Duration("idle", 5*time.Minute, "with -iron, give the NPU's memory back after this long without a request and load the model again for the next one (0: never); the conversation goes on from the host's copy")
	modelFiles := flag.String("models", "", chat.ModelsUsage)
	flag.Parse()
	if err := chat.LoadUserModels(*modelFiles, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-agent:", err)
		os.Exit(2)
	}
	task := strings.TrimSpace(strings.Join(flag.Args(), " "))
	tiers, err := chat.Tiers(*tier)
	if err != nil {
		fail(fmt.Errorf("-tier: %w", err))
	}
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
	searcher, err := agent.NewSearcher(*search, os.Getenv("BRAVE_API_KEY"))
	if err != nil {
		fail(err)
	}
	if *searchFirst && !slices.Contains(strings.Split(*tools, ","), "search") {
		fail(errors.New("-search-first needs search among -tools"))
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	var c chat.Config
	noThink := false // the model's own default (chat.Model.NoThink) when -think is not given
	if *ironDir == "" && *pkgs == "" {
		if *ironDir, err = chat.AMDExport(ctx, *use, *backend); err != nil {
			fail(err)
		} else if *ironDir != "" {
			fmt.Fprintf(os.Stderr, "hwnpu-agent: AMD's NPU: -iron %s\n", *ironDir)
		}
	}
	if *ironDir != "" {
		c = chat.Config{IRON: *ironDir, Tokenizer: *tok, Log: console.Log(os.Stderr), PromptDir: chat.PromptStore(*promptDir, *cacheDir), Idle: *idle}
	} else {
		m, err := chat.Resolve(ctx, *use, *dir, console.Log(os.Stderr))
		if err != nil {
			fail(err)
		}
		if m.Kind != chat.KindChat {
			fail(fmt.Errorf("%s is a %s model; the agent needs a chat model with tool calls", m.Name, m.Kind))
		}
		if len(m.Features) > 0 && !slices.Contains(m.Features, "tools") {
			fail(fmt.Errorf("%s has no tool calls; the agent needs a model with the tools feature (hwnpu-chat -list)", m.Name))
		}
		noThink = m.NoThink && !flagGiven("think")
		md := filepath.Join(*dir, m.DirName())
		c = chat.Config{ModelConfig: *config, Weights: *weights}
		if *pkgs != "" {
			c.Packages, c.Name, c.Backend = strings.Split(*pkgs, ","), m.Name, *backend
			if c.ModelConfig == "" {
				c.ModelConfig = filepath.Join(md, "config.json")
			}
			if c.Weights == "" {
				c.Weights = chat.WeightsIn(md)
			}
		} else if c, err = chat.Prepare(ctx, chat.Setup{Dir: md, Repo: m.Source(), Arch: m.Arch, Name: m.Name, Context: *context_, Paged: *paged, Draft: *draft, MTP: *mtp, Prefill: *prefill, Tiers: tiers, Backend: *backend, Format: *format, Codes: *codes, FormatFor: formatFor(*formatFor_), Log: console.Log(os.Stderr)}); err != nil {
			fail(err)
		}
		c.Tokenizer, c.CacheDir, c.Log = *tok, *cacheDir, console.Log(os.Stderr)
		c.PromptDir = chat.PromptStore(*promptDir, *cacheDir)
		c.Dense = !*sparse
	}
	e, err := chat.Load(ctx, c)
	if err != nil {
		fail(err)
	}
	defer e.Close()
	stop()

	if *thinkBudget <= 0 {
		*thinkBudget = -1 // no limit
	}
	ed := console.NewEditor("agent")
	cfg := agent.Config{
		Dir: work, Tools: strings.Split(*tools, ","), System: opening, Instructions: instructions,
		Think: *think && !noThink, KeepThinking: *keep, ThinkBudget: *thinkBudget, Sampling: chat.Sampling{Temperature: *temp, TopK: *topK, TopP: *topP, Repetition: *rep},
		MaxTokens: *maxNew, Reserve: *maxNew, Steps: *steps, Dump: *dump, Log: os.Stderr,
		Search: searcher, SearchFirst: *searchFirst,
	}
	if *allow != "" {
		cfg.AllowList = strings.Split(*allow, ",")
	}
	switch {
	case *yes:
	case console.Terminal(os.Stdin):
		cfg.Allow = func(tool, what string) bool {
			line, _ := ed.Ask(fmt.Sprintf("%s%s %s%s [y/N] ", bold, tool, what, reset))
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
	console.Notef("/reset starts a new conversation, /exit (or Ctrl-D) quits; Ctrl-C stops the task, ↑ ↓ go through what was typed\n")
	for {
		text, ok := read(ed)
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
			console.Notef("[conversation cleared]\n")
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
			if c.Drafted > 0 {
				looped += fmt.Sprintf("; %d of %d drafts accepted", c.Accepted, c.Drafted)
			}
			if c.Cache > 0 {
				looped += fmt.Sprintf("; cache %d", c.Cache)
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

// read prompts for and reads the next task with the line editor (a history on the arrows; a line
// ending in \ continues on the next). It returns false at the end of the input.
func read(ed *console.Editor) (string, bool) {
	fmt.Println()
	text, err := ed.Read("> ")
	if err != nil {
		return "", false
	}
	return strings.TrimSpace(text), true
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

// flagGiven reports whether the flag of that name was set on the command line.
func flagGiven(name string) bool {
	given := false
	flag.Visit(func(f *flag.Flag) { given = given || f.Name == name })
	return given
}

// formatFor splits -format-for into its SUBSTRING=FORMAT entries (chat.Setup.FormatFor).
func formatFor(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(s, ",")
}
