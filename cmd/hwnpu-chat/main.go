// hwnpu-chat runs two models on the Neural Engine: LFM3-1.7B to chat with, and PLaMo-Embedding-1B
// to turn texts into embeddings for search. -use picks them: chat (the default) talks in the
// terminal, embed searches texts typed in the terminal by their embeddings, and with -listen
// the OpenAI compatible server serves the chat completions API, the embeddings API or both.
//
//	hwnpu-chat
//	hwnpu-chat -use embed
//	hwnpu-chat -listen 127.0.0.1:8080
//	hwnpu-chat -listen 127.0.0.1:8080 -use chat,embed
//
// With no -package, the chat model lives in -model (hwnpu/models/LFM3-1.7B in the user cache
// directory): the first run downloads the checkpoint from Hugging Face (-repo, 3.5 GB) and
// converts it twice with a KV cache of -context tokens, 1 token per call to generate and 64 to
// read prompts about 20 times faster; later runs load what is there. The two share one cache,
// and a conversation reads only what is new since the last answer.
//
// -package takes programs converted by hwnpu-convert instead, with the same -cache and
// different -seq, separated by commas.
//
// The embedding model is the same with the -embed- flags: it lives in -embed-model
// (hwnpu/models/plamo-embedding-1b), comes from -embed-repo (2.1 GB) and is converted once per
// length in -embed-seq; a text runs on the shortest that holds it.
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
	"sort"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/chat"
)

func main() {
	use := flag.String("use", "chat", "models to run, comma separated: chat (LFM3), embed (PLaMo-Embedding); both only with -listen")
	model := flag.String("model", chat.DefaultDir(chat.DefaultRepo), "directory of the chat model: the checkpoint and the converted programs, downloaded and converted when missing")
	repo := flag.String("repo", chat.DefaultRepo, "Hugging Face repository the chat model comes from")
	context_ := flag.Int("context", chat.DefaultContext, "tokens of a prompt and its answer together (the KV cache of the converted programs)")
	pkgs := flag.String("package", "", "chat programs converted by hwnpu-convert instead, comma separated: .mlpackage or out.parts.json, with the same -cache and different -seq")
	config := flag.String("config", "", "config.json of the chat model (default: in -model)")
	weights := flag.String("weights", "", "checkpoint the token embeddings are read from: .safetensors or .gguf (default: in -model)")
	tok := flag.String("tokenizer", "", "tokenizer.json (default: next to config.json)")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory)")
	listen := flag.String("listen", "", "serve the OpenAI compatible API (chat completions with chat, embeddings with embed) at this address instead of the terminal, e.g. 127.0.0.1:8080")
	dump := flag.String("dump", "", "write each request, its prompt and the generated text to this directory (server)")
	think := flag.Bool("think", true, "let the model think (<think>) before it answers")
	system := flag.String("system", "", "system message (chat)")
	maxNew := flag.Int("n", 0, "most tokens of an answer (0: until the cache is full) (chat)")
	temp := flag.Float64("temp", 0.1, "sampling temperature; 0 picks the most likely token")
	topK := flag.Int("top-k", 50, "sample from the k most likely tokens")
	topP := flag.Float64("top-p", 0, "sample from the most likely tokens up to this probability (0: off)")
	rep := flag.Float64("rep", 1.05, "repetition penalty on the tokens of the answer (1: none)")
	embedModel := flag.String("embed-model", chat.DefaultDir(chat.DefaultEmbedRepo), "directory of the embedding model, downloaded and converted when missing")
	embedRepo := flag.String("embed-repo", chat.DefaultEmbedRepo, "Hugging Face repository the embedding model comes from")
	embedSeq := flag.String("embed-seq", ints(chat.DefaultEmbedSeqs), "sequence lengths the embedding model is converted for, comma separated; the longest is the most tokens of a text")
	embedPkgs := flag.String("embed-package", "", "embedding programs converted by hwnpu-convert instead, comma separated, with different -seq (config.json and the checkpoint in -embed-model)")
	flag.Parse()
	if flag.NArg() > 0 {
		flag.Usage()
		os.Exit(2)
	}
	useChat, useEmbed, err := parseUse(*use, *listen != "")
	if err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-chat:", err)
		os.Exit(2)
	}
	seqs, err := parseInts(*embedSeq)
	if err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-chat: -embed-seq:", err)
		os.Exit(2)
	}
	if *cacheDir == "" {
		if d, err := os.UserCacheDir(); err == nil {
			*cacheDir = filepath.Join(d, "hwnpu")
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var e *chat.Engine
	if useChat {
		c := chat.Config{ModelConfig: *config, Weights: *weights}
		if *pkgs != "" {
			c.Packages = strings.Split(*pkgs, ",")
			if c.ModelConfig == "" {
				c.ModelConfig = filepath.Join(*model, "config.json")
			}
			if c.Weights == "" {
				c.Weights = filepath.Join(*model, "model.safetensors")
			}
		} else if c, err = chat.Prepare(ctx, chat.Setup{Dir: *model, Repo: *repo, Context: *context_, Log: os.Stderr}); err != nil {
			fail(err)
		}
		c.Tokenizer, c.CacheDir, c.Log = *tok, *cacheDir, os.Stderr
		if e, err = chat.Load(ctx, c); err != nil {
			fail(err)
		}
		defer e.Close()
	}
	var emb *chat.Embedder
	if useEmbed {
		var c chat.EmbedConfig
		if *embedPkgs != "" {
			c = chat.EmbedConfig{Packages: strings.Split(*embedPkgs, ","), ModelConfig: filepath.Join(*embedModel, "config.json"), Weights: filepath.Join(*embedModel, "model.safetensors")}
		} else if c, err = chat.PrepareEmbedder(ctx, chat.EmbedSetup{Dir: *embedModel, Repo: *embedRepo, Seqs: seqs, Log: os.Stderr}); err != nil {
			fail(err)
		}
		c.CacheDir, c.Log = *cacheDir, os.Stderr
		if emb, err = chat.LoadEmbedder(ctx, c); err != nil {
			fail(err)
		}
		defer emb.Close()
	}
	s := chat.Sampling{Temperature: *temp, TopK: *topK, TopP: *topP, Repetition: *rep}
	switch {
	case *listen != "":
		err = serve(ctx, &chat.Server{E: e, Emb: emb, Think: *think, Defaults: s, Log: os.Stderr, Dump: *dump}, *listen)
	case useChat:
		stop() // Ctrl-C stops an answer, not the chat
		err = talk(e, *system, *think, *maxNew, s)
	default:
		stop()
		err = search(emb)
	}
	if err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "hwnpu-chat:", err)
	os.Exit(1)
}

// parseUse reads -use: which of the chat and the embedding model to run. The terminal takes one.
func parseUse(use string, server bool) (useChat, useEmbed bool, err error) {
	for _, u := range strings.Split(use, ",") {
		switch strings.TrimSpace(u) {
		case "chat":
			useChat = true
		case "embed":
			useEmbed = true
		default:
			return false, false, fmt.Errorf("-use %q: each must be chat or embed", u)
		}
	}
	if useChat && useEmbed && !server {
		return false, false, errors.New("-use chat,embed needs -listen; in the terminal, use one of them")
	}
	return useChat, useEmbed, nil
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("%q is not a positive number", f)
		}
		out = append(out, n)
	}
	return out, nil
}

func ints(ns []int) string {
	var s []string
	for _, n := range ns {
		s = append(s, strconv.Itoa(n))
	}
	return strings.Join(s, ",")
}

func serve(ctx context.Context, s *chat.Server, addr string) error {
	srv := &http.Server{Addr: addr, Handler: s.Handler()}
	go func() {
		<-ctx.Done()
		srv.Close()
	}()
	var what []string
	if s.E != nil {
		what = append(what, fmt.Sprintf("%s (chat, context %d tokens)", s.E.Name, s.E.Context()))
	}
	if s.Emb != nil {
		what = append(what, fmt.Sprintf("%s (embeddings, up to %d tokens)", s.Emb.Name, s.Emb.MaxTokens()))
	}
	fmt.Fprintf(os.Stderr, "serving %s at http://%s/v1\n", strings.Join(what, " and "), addr)
	if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

const (
	dim   = "\x1b[2m"
	reset = "\x1b[0m"
)

// read prompts for and reads the next input. A line ending in \ continues on the next line. It
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

// talk is the interactive chat.
func talk(e *chat.Engine, system string, think bool, maxNew int, s chat.Sampling) error {
	var msgs []chat.Message
	if system != "" {
		msgs = append(msgs, chat.Message{Role: "system", Content: chat.Content{Text: system}})
	}
	fmt.Fprintf(os.Stderr, "context %d tokens. /reset clears the conversation, /think on|off, /exit (or Ctrl-D) quits; Ctrl-C stops an answer\n", e.Context())
	in := bufio.NewReader(os.Stdin)
	for {
		text, ok := read(in)
		if !ok {
			return nil
		}
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

// search is the interactive search by embeddings: each input is a document kept for searching,
// and an input starting with ? is a query, which lists the documents most similar to it.
func search(e *chat.Embedder) error {
	type doc struct {
		text string
		v    []float32
	}
	var docs []doc
	fmt.Fprintf(os.Stderr, "%s: %d dimensions, up to %d tokens a text. Each input is a document to search; ?TEXT searches them, /list shows them, /reset clears them, /exit (or Ctrl-D) quits\n",
		e.Name, e.Dim(), e.MaxTokens())
	in := bufio.NewReader(os.Stdin)
	for {
		text, ok := read(in)
		if !ok {
			return nil
		}
		query := false
		switch {
		case text == "":
			continue
		case text == "/exit" || text == "/quit":
			return nil
		case text == "/reset":
			docs = docs[:0]
			fmt.Fprintln(os.Stderr, "[documents cleared]")
			continue
		case text == "/list":
			for i, d := range docs {
				fmt.Printf("%3d  %s\n", i+1, oneLine(d.text))
			}
			continue
		case strings.HasPrefix(text, "?") || strings.HasPrefix(text, "？"):
			query = true
			text = strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(text, "?"), "？"))
		}
		r, err := e.Embed(context.Background(), text, query, true)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			continue
		}
		cut := ""
		if r.Truncated {
			cut = fmt.Sprintf(", cut to %d", e.MaxTokens())
		}
		if !query {
			docs = append(docs, doc{text, r.Vector})
			fmt.Fprintf(os.Stderr, "%s[document %d: %d tokens%s in %.2fs]%s\n", dim, len(docs), r.Tokens, cut, r.Time.Seconds(), reset)
			continue
		}
		if len(docs) == 0 {
			fmt.Fprintln(os.Stderr, "[no documents yet: type some first]")
			continue
		}
		order := make([]int, len(docs))
		sim := make([]float64, len(docs))
		for i, d := range docs {
			order[i], sim[i] = i, chat.Cosine(r.Vector, d.v)
		}
		sort.SliceStable(order, func(a, b int) bool { return sim[order[a]] > sim[order[b]] })
		for _, i := range order[:min(len(order), 10)] {
			fmt.Printf("%.3f  %3d  %s\n", sim[i], i+1, oneLine(docs[i].text))
		}
		fmt.Fprintf(os.Stderr, "%s[query: %d tokens%s in %.2fs]%s\n", dim, r.Tokens, cut, r.Time.Seconds(), reset)
	}
}

// oneLine shortens a text to one line of at most 60 characters for a list.
func oneLine(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > 60 {
		return string(r[:59]) + "…"
	}
	return s
}
