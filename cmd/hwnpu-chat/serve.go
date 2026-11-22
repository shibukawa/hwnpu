package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/server/openai"
	"github.com/shibukawa/hwnpu/ai/text"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
)

// serverFlags are the models a server runs and how they answer (serve, opencode).
type serverFlags struct {
	Model string `short:"m" default:"LFM3-1.7B" placeholder:"NAME,..." help:"The models to serve, comma separated: names of hwnpu-chat list, uses or families (translate, plamo3translate:2b), Hugging Face repositories OWNER/NAME, or an export's directory (AMD: one chat model). A request goes to the model it names."`
	Dump  string `type:"path" placeholder:"DIR" help:"Write each chat request, its prompt and the generated text to this directory."`
	loadFlags
	convertFlags
	samplingFlags
	encoderFlags
	embedFlags
}

// server loads the models of --model into the OpenAI API's server (ai/server/openai), on the
// Runtime returned, which the caller closes.
func (f *serverFlags) server(ctx context.Context) (*openai.Server, *ai.Runtime, error) {
	st, cat, err := f.loadFlags.prepare(ctx)
	if err != nil {
		return nil, nil, err
	}
	var (
		chats  []ai.Use[*text.Chatter]
		embeds []ai.Use[*text.Embedder]
		defs   []ai.Definition
		kept   []chatModel
	)
	for _, name := range split(f.Model) {
		name = strings.TrimSpace(name)
		if !isPath(name) {
			if m, err := cat.Find(name, text.Embed(name)); err == nil && m.Kind == string(chat.KindEmbed) {
				if len(embeds) > 0 && f.EmbedPackage != "" {
					return nil, nil, errors.New("--embed-package is for one embedding model; --model names several")
				}
				u, d, err := embedUse(cat, name, &f.embedFlags)
				if err != nil {
					return nil, nil, err
				}
				embeds, defs = append(embeds, u), append(defs, d...)
				continue
			}
		}
		if len(chats) > 0 && (f.Package != "" || f.Config != "" || f.Weights != "" || f.Tokenizer != "" || f.Vision != "" || f.AudioEncoder != "") {
			return nil, nil, errors.New("--package, --config, --weights, --tokenizer, --vision and --audio-encoder are for one chat model; --model names several")
		}
		cm, err := chatModelOf(cat, name, &f.convertFlags, &f.encoderFlags)
		if err != nil {
			return nil, nil, err
		}
		chats, defs, kept = append(chats, cm.chat), append(defs, cm.defs...), append(kept, cm)
	}
	st.Definitions = defs
	app := ai.NewApp("hwnpu-chat")
	for _, u := range chats {
		app.Use(u)
	}
	for _, u := range embeds {
		app.Use(u)
	}
	app.Store = st
	rt, err := open(ctx, app, options(&f.loadFlags, &f.convertFlags, "serve")...)
	if err != nil {
		return nil, nil, err
	}
	s, err := openai.Load(ctx, rt, chats, embeds)
	if err != nil {
		rt.Close()
		return nil, nil, err
	}
	prepared(rt)
	for _, cm := range kept {
		cm.keep()
	}
	s.Think, s.ThinkBudget, s.Log, s.Dump = f.Think, f.budget(), logw(), f.Dump
	sm := f.sampling()
	s.Sampling = &sm
	return s, rt, nil
}

// ServeCmd is "hwnpu-chat serve": the OpenAI compatible API, chat completions for the chat models
// and embeddings for the embedding models.
type ServeCmd struct {
	Listen string `default:"127.0.0.1:8080" placeholder:"ADDR" help:"The address to serve at."`
	serverFlags
}

// Run loads the models and serves until Ctrl-C.
func (c *ServeCmd) Run(ctx context.Context, stop context.CancelFunc) error {
	s, rt, err := c.server(ctx)
	if err != nil {
		return err
	}
	defer rt.Close()
	srv := &http.Server{Addr: c.Listen, Handler: s.Handler()}
	go func() {
		<-ctx.Done()
		stop() // Ctrl-C is how serving ends: the models are closed and the command returns, not cut short (console.Interrupt)
		srv.Close()
	}()
	var what []string
	for _, e := range s.Chats {
		what = append(what, fmt.Sprintf("%s (chat, context %d tokens)", e.Name(), e.Context()))
	}
	for _, e := range s.Embedders {
		what = append(what, fmt.Sprintf("%s (embeddings, up to %d tokens)", e.Name(), e.MaxTokens()))
	}
	fmt.Fprintf(os.Stderr, "serving %s at http://%s/v1\n", strings.Join(what, " and "), c.Listen)
	if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

// OpenCodeCmd is "hwnpu-chat opencode": the server on a free port from 30000 (or --listen), an
// OpenCode config for its chat models, and OpenCode on this terminal with the arguments after
// --; when it exits, so does the server.
type OpenCodeCmd struct {
	Args   []string `arg:"" optional:"" passthrough:"" help:"OpenCode's arguments (after --)."`
	Listen string   `placeholder:"ADDR" help:"The address to serve at (default: a free port of 127.0.0.1 from 30000)."`
	Bin    string   `default:"opencode" help:"The OpenCode command."`
	serverFlags
}

// Run serves and runs OpenCode, and exits with its code.
func (c *OpenCodeCmd) Run(ctx context.Context, stop context.CancelFunc) error {
	s, rt, err := c.server(ctx)
	if err != nil {
		return err
	}
	stop()
	code, err := openCode(s, c.Listen, openCodeDir(cache.Root()), c.Bin, c.openCodeArgs())
	rt.Close()
	if err != nil {
		return err
	}
	os.Exit(code)
	return nil
}

// openCodeArgs are the arguments for OpenCode, without the -- that kong keeps.
func (c *OpenCodeCmd) openCodeArgs() []string {
	if len(c.Args) > 0 && c.Args[0] == "--" {
		return c.Args[1:]
	}
	return c.Args
}

// SearchCmd is "hwnpu-chat search": each text typed is a document kept for searching, and a text
// starting with ? is a query, which lists the documents most similar to it.
type SearchCmd struct {
	Model string `short:"m" default:"embedding" placeholder:"NAME" help:"The embedding model: a name of hwnpu-chat list, embedding for the one picked for the machine's memory, or a Hugging Face repository OWNER/NAME."`
	loadFlags
	embedFlags
}

// Run loads the model and searches.
func (c *SearchCmd) Run(ctx context.Context, stop context.CancelFunc) error {
	st, cat, err := c.loadFlags.prepare(ctx)
	if err != nil {
		return err
	}
	u, defs, err := embedUse(cat, c.Model, &c.embedFlags)
	if err != nil {
		return err
	}
	st.Definitions = defs
	app := ai.NewApp("hwnpu-chat", u)
	app.Store = st
	rt, err := open(ctx, app)
	if err != nil {
		return err
	}
	defer rt.Close()
	e, err := u.Get(ctx, rt)
	if err != nil {
		return err
	}
	stop()
	return search(e)
}

// search is the interactive search by embeddings.
func search(e *text.Embedder) error {
	type doc struct {
		text string
		v    []float32
	}
	var docs []doc
	console.Notef("%s: %d dimensions, up to %d tokens a text. Each input is a document to search; ?TEXT searches them, /list shows them, /reset clears them, /exit (or Ctrl-D) quits\n",
		e.Name(), e.Dim(), e.MaxTokens())
	ed := console.NewEditor("search")
	for {
		line, ok := read(ed)
		if !ok {
			return nil
		}
		query := false
		switch {
		case line == "":
			continue
		case line == "/exit" || line == "/quit":
			return nil
		case line == "/reset":
			docs = docs[:0]
			console.Notef("[documents cleared]\n")
			continue
		case line == "/list":
			for i, d := range docs {
				fmt.Printf("%3d  %s\n", i+1, oneLine(d.text))
			}
			continue
		case strings.HasPrefix(line, "?") || strings.HasPrefix(line, "？"):
			query = true
			line = strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(line, "?"), "？"))
		}
		r, err := e.Embed(context.Background(), line, text.EmbedOptions{Query: query, Truncate: true})
		if err != nil {
			printError(err)
			continue
		}
		cut := ""
		if r.Truncated {
			cut = fmt.Sprintf(", cut to %d", e.MaxTokens())
		}
		if !query {
			docs = append(docs, doc{line, r.Vector})
			console.Notef("[document %d: %d tokens%s in %.2fs]\n", len(docs), r.Tokens, cut, r.Took.Seconds())
			continue
		}
		if len(docs) == 0 {
			console.Notef("[no documents yet: type some first]\n")
			continue
		}
		order := make([]int, len(docs))
		sim := make([]float64, len(docs))
		for i, d := range docs {
			order[i], sim[i] = i, text.Cosine(r.Vector, d.v)
		}
		sort.SliceStable(order, func(a, b int) bool { return sim[order[a]] > sim[order[b]] })
		for _, i := range order[:min(len(order), 10)] {
			fmt.Printf("%.3f  %3d  %s\n", sim[i], i+1, oneLine(docs[i].text))
		}
		console.Notef("[query: %d tokens%s in %.2fs]\n", r.Tokens, cut, r.Took.Seconds())
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

// ListCmd is "hwnpu-chat list": the chat and embedding models of the definitions (models.yaml),
// whether they are downloaded (on AMD's NPU: the export each runs from), and what the features
// mean.
type ListCmd struct {
	Models string `placeholder:"FILES" help:"${models_usage}."`
}

// Run lists the models.
func (c *ListCmd) Run(ctx context.Context) error {
	if _, err := store(c.Models).Catalog(); err != nil { // the definitions, as the commands read them
		return err
	}
	showModels(os.Stdout, chat.OnlyAMD(ctx, ""))
	return nil
}

// showModels writes the models of the definitions hwnpu-chat runs (chat.List) and what the
// features mean.
func showModels(w io.Writer, amd bool) {
	if amd { // what runs: the export here, or the one the first run makes
		chat.ListHave(w, "RUNS FROM", func(m chat.Model) string {
			if d := chat.FindAMDExport(m); d != "" {
				return shortPath(d)
			}
			if len(m.AMDExport) > 0 {
				return "exported on first run"
			}
			return "- (no form for AMD)"
		}, chat.KindChat, chat.KindEmbed, chat.KindDecision)
	} else {
		chat.List(w, chat.KindChat, chat.KindEmbed, chat.KindDecision)
	}
	fmt.Fprintln(w, "\nFeatures:")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	for _, f := range chat.Features {
		fmt.Fprintf(tw, "  %s\t%s\n", f.Name, f.About)
	}
	tw.Flush()
	fmt.Fprintf(w, "\n--model (-m) takes a model's name, a use or a family (translate, plamo3translate), either with a\n"+
		"tag (plamo3translate:2b, chat:e4b, chat:16g), or a Hugging Face repository OWNER/NAME of a supported\n"+
		"architecture; serve takes several, comma separated. The default is LFM3-1.7B (search: embedding).\n"+
		"Requests may name a model the same ways. Only the model picked is downloaded.\n"+
		"On AMD's NPU a name runs its export, NAME.amd-npu2-q4 (-q6, or none for bf16) in %s or ~/models;\n"+
		"the first run takes it from the sources (hwnpu-dev sources), else downloads the checkpoint\n"+
		"and makes it with IRON (tools/amd/iron_export_lm.py, %s, INSTALL_AMD.md).\n"+
		"Models are kept in %s (HWNPU_CACHE moves it).\n", shortPath(chat.AMDRoot()), chat.IRONEnv, shortPath(cache.Root()))
}

// shortPath is path with the home directory written ~.
func shortPath(path string) string {
	home, err := os.UserHomeDir()
	if err == nil && home != "" && strings.HasPrefix(path, home+string(filepath.Separator)) {
		return "~" + path[len(home):]
	}
	return path
}
