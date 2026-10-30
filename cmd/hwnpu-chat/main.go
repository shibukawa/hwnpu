// hwnpu-chat talks with LFM3-1.7B on the Neural Engine: interactively in the terminal, or as an
// OpenAI compatible server (-listen) for clients such as OpenCode.
//
//	hwnpu-chat
//	hwnpu-chat -listen 127.0.0.1:8080
//	hwnpu-chat -opencode [opencode arguments]
//
// With no -package, the model lives in -model (hwnpu/models/LFM3-1.7B in the user cache
// directory): the first run downloads the checkpoint from Hugging Face (-repo, 3.5 GB) and
// converts it twice with a KV cache of -context tokens, 1 token per call to generate and 64 to
// read prompts about 20 times faster; later runs load what is there. The two share one cache,
// and a conversation reads only what is new since the last answer.
//
// -package takes programs converted by hwnpu-convert instead, with the same -cache and
// different -seq, separated by commas.
//
// -opencode serves on a free port from 30000 (or -listen), writes an OpenCode config for the
// server, and runs OpenCode on this terminal with the arguments after the flags; when it exits,
// so does the server.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/chat"
)

func main() {
	model := flag.String("model", chat.DefaultDir(chat.DefaultRepo), "directory of the model: the checkpoint and the converted programs, downloaded and converted when missing")
	repo := flag.String("repo", chat.DefaultRepo, "Hugging Face repository the checkpoint comes from")
	context_ := flag.Int("context", chat.DefaultContext, "tokens of a prompt and its answer together (the KV cache of the converted programs); over 16320 the cache is held in chunks of 16384, slower per token")
	pkgs := flag.String("package", "", "programs converted by hwnpu-convert instead, comma separated: .mlpackage or out.parts.json, with the same -cache and different -seq")
	config := flag.String("config", "", "config.json of the model (default: in -model)")
	weights := flag.String("weights", "", "checkpoint the token embeddings are read from: .safetensors or .gguf (default: in -model)")
	tok := flag.String("tokenizer", "", "tokenizer.json (default: next to config.json)")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory)")
	listen := flag.String("listen", "", "serve the OpenAI chat completions API at this address instead of chatting, e.g. 127.0.0.1:8080")
	dump := flag.String("dump", "", "write each request, its prompt and the generated text to this directory (server)")
	withOpenCode := flag.Bool("opencode", false, "serve on a free port from 30000 (or -listen) and run OpenCode on this terminal with this server as its model; the arguments after the flags go to OpenCode")
	openCodeBin := flag.String("opencode-bin", "opencode", "the OpenCode command (-opencode)")
	think := flag.Bool("think", true, "let the model think (<think>) before it answers")
	system := flag.String("system", "", "system message (chat)")
	maxNew := flag.Int("n", 0, "most tokens of an answer (0: until the cache is full) (chat)")
	temp := flag.Float64("temp", 0.1, "sampling temperature; 0 picks the most likely token")
	topK := flag.Int("top-k", 50, "sample from the k most likely tokens")
	topP := flag.Float64("top-p", 0, "sample from the most likely tokens up to this probability (0: off)")
	rep := flag.Float64("rep", 1.05, "repetition penalty on the tokens of the answer (1: none)")
	flag.Parse()
	if flag.NArg() > 0 && !*withOpenCode {
		flag.Usage()
		os.Exit(2)
	}
	if *cacheDir == "" {
		if d, err := os.UserCacheDir(); err == nil {
			*cacheDir = filepath.Join(d, "hwnpu")
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	c := chat.Config{ModelConfig: *config, Weights: *weights}
	var err error
	if *pkgs != "" {
		c.Packages = strings.Split(*pkgs, ",")
		if c.ModelConfig == "" {
			c.ModelConfig = filepath.Join(*model, "config.json")
		}
		if c.Weights == "" {
			c.Weights = filepath.Join(*model, "model.safetensors")
		}
	} else if c, err = chat.Prepare(ctx, chat.Setup{Dir: *model, Repo: *repo, Context: *context_, Log: os.Stderr}); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-chat:", err)
		os.Exit(1)
	}
	c.Tokenizer, c.CacheDir, c.Log = *tok, *cacheDir, os.Stderr
	e, err := chat.Load(ctx, c)
	if err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-chat:", err)
		os.Exit(1)
	}
	defer e.Close()
	s := chat.Sampling{Temperature: *temp, TopK: *topK, TopP: *topP, Repetition: *rep}
	if *withOpenCode {
		stop()
		code, err := openCode(&chat.Server{E: e, Think: *think, Defaults: s, Dump: *dump}, *listen, openCodeDir(*cacheDir), *openCodeBin, flag.Args())
		if err != nil {
			fmt.Fprintln(os.Stderr, "hwnpu-chat:", err)
		}
		e.Close()
		os.Exit(code)
	}
	if *listen != "" {
		err = serve(ctx, &chat.Server{E: e, Think: *think, Defaults: s, Log: os.Stderr, Dump: *dump}, *listen)
	} else {
		stop() // Ctrl-C stops an answer, not the chat
		err = talk(e, *system, *think, *maxNew, s)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-chat:", err)
		os.Exit(1)
	}
}

func serve(ctx context.Context, s *chat.Server, addr string) error {
	srv := &http.Server{Addr: addr, Handler: s.Handler()}
	go func() {
		<-ctx.Done()
		srv.Close()
	}()
	fmt.Fprintf(os.Stderr, "serving %s at http://%s/v1 (context %d tokens)\n", s.E.Name, addr, s.E.Context())
	if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

const (
	dim   = "\x1b[2m"
	reset = "\x1b[0m"
)

// talk is the interactive chat. A line ending in \ continues on the next line.
func talk(e *chat.Engine, system string, think bool, maxNew int, s chat.Sampling) error {
	var msgs []chat.Message
	if system != "" {
		msgs = append(msgs, chat.Message{Role: "system", Content: chat.Content{Text: system}})
	}
	fmt.Fprintln(os.Stderr, "/reset clears the conversation, /think on|off, /exit (or Ctrl-D) quits; Ctrl-C stops an answer")
	in := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\n> ")
		var lines []string
		for {
			line, err := in.ReadString('\n')
			if err != nil && line == "" {
				fmt.Println()
				return nil
			}
			line = strings.TrimRight(line, "\r\n")
			if strings.HasSuffix(line, `\`) {
				lines = append(lines, strings.TrimSuffix(line, `\`))
				fmt.Print(". ")
				continue
			}
			lines = append(lines, line)
			break
		}
		text := strings.TrimSpace(strings.Join(lines, "\n"))
		switch {
		case text == "":
			continue
		case text == "/exit" || text == "/quit":
			return nil
		case text == "/reset":
			msgs = msgs[:0]
			if system != "" {
				msgs = append(msgs, chat.Message{Role: "system", Content: chat.Content{Text: system}})
			}
			fmt.Fprintln(os.Stderr, "[conversation cleared]")
			continue
		case strings.HasPrefix(text, "/think"):
			think = strings.TrimSpace(strings.TrimPrefix(text, "/think")) != "off"
			fmt.Fprintf(os.Stderr, "[thinking %v]\n", think)
			continue
		}
		msgs = append(msgs, chat.Message{Role: "user", Content: chat.Content{Text: text}})
		prompt, err := chat.Render(msgs, nil, chat.Options{NoThink: !think})
		if err != nil {
			return err
		}
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		var sent chat.Stream
		var gen []int32
		thought, answered := false, false
		st, err := e.Generate(ctx, chat.Request{Prompt: e.Tok.Encode(prompt), MaxTokens: maxNew, Sampling: s}, func(g []int32) (bool, error) {
			gen = g
			a := chat.Parse(e.Tok.Decode(g, false), think, nil, false)
			dr, dc := sent.Next(a.Reasoning, a.Content)
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
			return false, nil
		})
		stop()
		fmt.Println()
		if err != nil && !errors.Is(err, context.Canceled) {
			fmt.Fprintln(os.Stderr, "error:", err)
			msgs = msgs[:len(msgs)-1]
			continue
		}
		a := chat.Parse(e.Tok.Decode(gen, false), think, nil, true)
		r := a.Reasoning
		msgs = append(msgs, chat.Message{Role: "assistant", Content: chat.Content{Text: a.Content}, Reasoning: &r})
		fmt.Fprintf(os.Stderr, "%s[prompt %d tokens, %d new in %.1fs; %d tokens at %.1f tokens/s]%s\n", dim,
			st.Prompt, st.Prompt-st.Reused, st.Prefill.Seconds(), st.Generated, float64(st.Generated)/max(st.Decode.Seconds(), 1e-9), reset)
	}
}
