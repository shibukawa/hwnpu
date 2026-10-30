// hwnpu-chat runs models on the Neural Engine by name: chat models (LFM3-1.7B) to talk with, and
// embedding models (PLaMo-Embedding-1B) to turn texts into vectors for search. -list shows the
// models it knows and what each is for; -use picks them. In the terminal, a chat model talks
// and an embedding model searches the texts typed by their embeddings; with -listen, the OpenAI
// compatible server serves the chat models with the chat completions API and the embedding
// models with the embeddings API, a request going to the model it names.
//
//	hwnpu-chat
//	hwnpu-chat -list
//	hwnpu-chat -use plamo-embedding-1b
//	hwnpu-chat -listen 127.0.0.1:8080 -use LFM3-1.7B,plamo-embedding-1b
//
// A model lives in -dir/NAME (hwnpu/models/NAME in the user cache directory). The first run
// downloads it from Hugging Face and converts it; later runs load what is there. A chat model is
// converted twice with a KV cache of -context tokens, 1 token per call to generate and 64 to read
// prompts about 20 times faster; the two share one cache, and a conversation reads only what is
// new since the last answer. An embedding model is converted once per length in -embed-seq, and
// a text runs on the shortest that holds it.
//
// -package (or -embed-package) takes programs converted by hwnpu-convert instead, separated by
// commas: for a chat model with the same -cache and different -seq, for an embedding model with
// different -seq.
// For LFM3-VL, -vision takes its vision encoder (hwnpu-convert -arch lfm3-vision, a package per
// image size): image_url parts of user messages are read, and in the terminal /image FILE sends an
// image. For LFM3-Audio, -audio takes its audio encoder (hwnpu-convert -arch lfm3-audio-encoder, a
// package per clip length): input_audio parts (WAV) are read, and /audio FILE.wav sends a clip.
// Both go with -package and one chat model. Speech output is not supported (see hwnpu-tts).
package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/shibukawa/hwnpu/internal/chat"
)

func main() {
	use := flag.String("use", "LFM3-1.7B", "models to run, comma separated: names that -list shows, or Hugging Face repositories OWNER/NAME; one in the terminal")
	list := flag.Bool("list", false, "list the models and what each is for, and exit")
	dir := flag.String("dir", chat.ModelsDir(), "where the models are kept, each in a directory of its name: the checkpoint and the converted programs, downloaded and converted when missing")
	context_ := flag.Int("context", chat.DefaultContext, "tokens of a prompt and its answer together (the KV cache of a chat model's programs); over 16320 the cache is held in chunks of 16384, slower per token")
	pkgs := flag.String("package", "", "chat programs converted by hwnpu-convert instead, comma separated: .mlpackage or out.parts.json, with the same -cache and different -seq")
	config := flag.String("config", "", "config.json of the chat model (default: in its directory)")
	weights := flag.String("weights", "", "checkpoint the chat model's token embeddings are read from: .safetensors or .gguf (default: in its directory)")
	tok := flag.String("tokenizer", "", "tokenizer.json of the chat model (default: next to config.json)")
	embedSeq := flag.String("embed-seq", ints(chat.DefaultEmbedSeqs), "sequence lengths an embedding model is converted for, comma separated; the longest is the most tokens of a text")
	embedPkgs := flag.String("embed-package", "", "embedding programs converted by hwnpu-convert instead, comma separated, with different -seq (config.json and the checkpoint in the model's directory)")
	vis := flag.String("vision", "", "the LFM3-VL vision encoder converted by hwnpu-convert -arch lfm3-vision, a package per image size, comma separated: images in user messages (with -package)")
	aud := flag.String("audio", "", "the LFM3-Audio audio encoder converted by hwnpu-convert -arch lfm3-audio-encoder, a package per clip length, comma separated: audio in user messages (with -package)")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory)")
	listen := flag.String("listen", "", "serve the OpenAI compatible API (chat completions for chat models, embeddings for embedding models) at this address instead of the terminal, e.g. 127.0.0.1:8080")
	dump := flag.String("dump", "", "write each chat request, its prompt and the generated text to this directory (server)")
	think := flag.Bool("think", true, "let the model think (<think>) before it answers")
	system := flag.String("system", "", "system message (chat)")
	maxNew := flag.Int("n", 0, "most tokens of an answer (0: until the cache is full) (chat)")
	temp := flag.Float64("temp", 0.1, "sampling temperature; 0 picks the most likely token")
	topK := flag.Int("top-k", 50, "sample from the k most likely tokens")
	topP := flag.Float64("top-p", 0, "sample from the most likely tokens up to this probability (0: off)")
	rep := flag.Float64("rep", 1.05, "repetition penalty on the tokens of the answer (1: none)")
	flag.Parse()
	if flag.NArg() > 0 {
		flag.Usage()
		os.Exit(2)
	}
	if *list {
		showModels(os.Stdout, *dir)
		return
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
	models, err := resolve(ctx, *use, *dir, *listen != "")
	if err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-chat:", err)
		os.Exit(2)
	}
	var chats []chat.Model
	for _, m := range models {
		if m.Kind == chat.KindChat {
			chats = append(chats, m)
		}
	}
	if len(chats) > 1 && (*pkgs != "" || *config != "" || *weights != "" || *tok != "" || *vis != "" || *aud != "") {
		fail(errors.New("-package, -config, -weights, -tokenizer, -vision and -audio are for one chat model; -use names several"))
	}
	if len(models)-len(chats) > 1 && *embedPkgs != "" {
		fail(errors.New("-embed-package is for one embedding model; -use names several"))
	}

	srv := &chat.Server{Think: *think, Defaults: chat.Sampling{Temperature: *temp, TopK: *topK, TopP: *topP, Repetition: *rep}, Log: os.Stderr, Dump: *dump}
	for _, m := range models {
		md := filepath.Join(*dir, m.Name)
		switch m.Kind {
		case chat.KindChat:
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
			if *vis != "" {
				c.Vision = strings.Split(*vis, ",")
			}
			if *aud != "" {
				c.Audio = strings.Split(*aud, ",")
			}
			e, err := chat.Load(ctx, c)
			if err != nil {
				fail(err)
			}
			defer e.Close()
			srv.Chats = append(srv.Chats, e)
		case chat.KindEmbed:
			var c chat.EmbedConfig
			if *embedPkgs != "" {
				c = chat.EmbedConfig{Packages: strings.Split(*embedPkgs, ","), ModelConfig: filepath.Join(md, "config.json"), Weights: filepath.Join(md, "model.safetensors"), Name: m.Name}
			} else if c, err = chat.PrepareEmbedder(ctx, chat.EmbedSetup{Dir: md, Repo: m.Repo, Seqs: seqs, Log: os.Stderr}); err != nil {
				fail(err)
			}
			c.CacheDir, c.Log = *cacheDir, os.Stderr
			e, err := chat.LoadEmbedder(ctx, c)
			if err != nil {
				fail(err)
			}
			defer e.Close()
			srv.Embedders = append(srv.Embedders, e)
		}
	}
	switch {
	case *listen != "":
		err = serve(ctx, srv, *listen)
	case len(srv.Chats) > 0:
		stop() // Ctrl-C stops an answer, not the chat
		err = talk(srv.Chats[0], *system, *think, *maxNew, srv.Defaults)
	default:
		stop()
		err = search(srv.Embedders[0])
	}
	if err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "hwnpu-chat:", err)
	os.Exit(1)
}

// resolve reads -use: the models by name, each once. The terminal takes one.
func resolve(ctx context.Context, use, dir string, server bool) ([]chat.Model, error) {
	var models []chat.Model
	seen := map[string]bool{}
	for _, name := range strings.Split(use, ",") {
		m, err := chat.Resolve(ctx, strings.TrimSpace(name), dir, os.Stderr)
		if err != nil {
			return nil, err
		}
		if !seen[m.Name] {
			seen[m.Name] = true
			models = append(models, m)
		}
	}
	if len(models) > 1 && !server {
		return nil, fmt.Errorf("-use %s: the terminal runs one model; several need -listen", use)
	}
	return models, nil
}

// showModels writes the catalog: each model's name, what it is for and what else it can do, its
// size and repository, whether it is downloaded to dir, and what it is; then what the features
// mean.
func showModels(w io.Writer, dir string) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tKIND\tFEATURES\tSIZE\tREPOSITORY\tDOWNLOADED\tABOUT")
	for _, m := range chat.Catalog {
		got := "no"
		if chat.Downloaded(filepath.Join(dir, m.Name)) {
			got = "yes"
		}
		features := strings.Join(m.Features, ",")
		if features == "" {
			features = "-"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%.1f GB\t%s\t%s\t%s\n", m.Name, m.Kind, features, m.Bytes/1e9, m.Repo, got, m.About)
	}
	tw.Flush()
	fmt.Fprintln(w, "\nFeatures:")
	tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	for _, f := range chat.Features {
		fmt.Fprintf(tw, "  %s\t%s\n", f.Name, f.About)
	}
	tw.Flush()
	fmt.Fprintf(w, "\n-use takes names, comma separated (one in the terminal, any with -listen), or a Hugging Face\n"+
		"repository OWNER/NAME of a supported architecture. The default is -use LFM3-1.7B.\n"+
		"In the terminal, a chat model talks and an embedding model searches the texts typed; with -listen,\n"+
		"chat models answer /v1/chat/completions and embedding models /v1/embeddings.\nModels are kept in %s.\n", dir)
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
	for _, e := range s.Chats {
		what = append(what, fmt.Sprintf("%s (chat, context %d tokens)", e.Name, e.Context()))
	}
	for _, e := range s.Embedders {
		what = append(what, fmt.Sprintf("%s (embeddings, up to %d tokens)", e.Name, e.MaxTokens()))
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
	media := ""
	if e.Images() {
		media += "/image FILE [text] sends an image, "
	}
	if e.Audio() {
		media += "/audio FILE.wav [text] sends a clip, "
	}
	fmt.Fprintf(os.Stderr, "%s/reset clears the conversation, /think on|off, /exit (or Ctrl-D) quits; Ctrl-C stops an answer\n", media)
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
		content := chat.Content{Text: text}
		// /image FILE [text], /audio FILE.wav [text]: the image or the clip, then the text
		for _, cmd := range []string{"/image ", "/audio "} {
			rest, ok := strings.CutPrefix(text, cmd)
			if !ok {
				continue
			}
			file, after, _ := strings.Cut(strings.TrimSpace(rest), " ")
			m, err := mediaFile(file, cmd == "/audio ")
			if err != nil {
				fmt.Fprintln(os.Stderr, "error:", err)
				content.Text = ""
				break
			}
			content = chat.Content{Text: chat.MediaMarker + strings.TrimSpace(after), Media: []chat.Media{m}}
		}
		if content.Text == "" {
			continue
		}
		msgs = append(msgs, chat.Message{Role: "user", Content: content})
		prompt, err := chat.Render(msgs, nil, chat.Options{NoThink: !think})
		if err != nil {
			return err
		}
		var media []chat.Media
		for _, m := range msgs {
			media = append(media, m.Content.Media...)
		}
		ids, err := e.Tokens(context.Background(), prompt, media)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			msgs = msgs[:len(msgs)-1]
			continue
		}
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		var sent chat.Stream
		var gen []int32
		thought, answered := false, false
		st, err := e.Generate(ctx, chat.Request{Prompt: ids, MaxTokens: maxNew, Sampling: s}, func(g []int32) (bool, error) {
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

// mediaFile reads an image file as a data URL, or a WAV file as base64.
func mediaFile(path string, isAudio bool) (chat.Media, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return chat.Media{}, err
	}
	if isAudio {
		return chat.Media{Audio: true, Data: base64.StdEncoding.EncodeToString(raw)}, nil
	}
	mime := http.DetectContentType(raw)
	if !strings.HasPrefix(mime, "image/") {
		return chat.Media{}, fmt.Errorf("%s is %s, not an image", path, mime)
	}
	return chat.Media{Data: "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(raw)}, nil
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
