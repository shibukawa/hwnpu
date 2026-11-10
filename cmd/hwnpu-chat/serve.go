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

// server loads the models of --model into a chat.Server.
func (f *serverFlags) server(ctx context.Context) (*chat.Server, error) {
	if err := f.prepare(); err != nil {
		return nil, err
	}
	srv := &chat.Server{Think: f.Think == nil || *f.Think, ThinkBudget: f.ThinkBudget, Defaults: f.sampling(), Log: logw(), Dump: f.Dump}
	fail := func(err error) (*chat.Server, error) {
		closeServer(srv)
		return nil, err
	}
	seen := map[string]bool{}
	for _, name := range split(f.Model) {
		name = strings.TrimSpace(name)
		var m chat.Model
		if !isPath(name) {
			var err error
			if m, err = chat.Resolve(ctx, name, f.Dir, logw()); err != nil {
				return fail(err)
			}
			if seen[m.Name] {
				continue
			}
			seen[m.Name] = true
		}
		if m.Kind == chat.KindEmbed {
			if len(srv.Embedders) > 0 && f.EmbedPackage != "" {
				return fail(errors.New("--embed-package is for one embedding model; --model names several"))
			}
			e, err := loadEmbedder(ctx, m, &f.loadFlags, &f.embedFlags)
			if err != nil {
				return fail(err)
			}
			srv.Embedders = append(srv.Embedders, e)
			continue
		}
		if len(srv.Chats) > 0 && (f.Package != "" || f.Config != "" || f.Weights != "" || f.Tokenizer != "" || f.Vision != "" || f.AudioEncoder != "") {
			return fail(errors.New("--package, --config, --weights, --tokenizer, --vision and --audio-encoder are for one chat model; --model names several"))
		}
		if !isPath(name) && !strings.Contains(name, "/") {
			name = m.Name // picked once (a family or a use logs what it picks)
		}
		e, err := loadChat(ctx, name, &f.loadFlags, &f.convertFlags, &f.encoderFlags)
		if err != nil {
			return fail(err)
		}
		if f.Think != nil {
			e.NoThink = false // --think or --no-think says it for every model
		}
		srv.Chats = append(srv.Chats, e.Engine)
	}
	return srv, nil
}

func closeServer(s *chat.Server) {
	for _, e := range s.Chats {
		e.Close()
	}
	for _, e := range s.Embedders {
		e.Close()
	}
}

// ServeCmd is "hwnpu-chat serve": the OpenAI compatible API, chat completions for the chat models
// and embeddings for the embedding models.
type ServeCmd struct {
	Listen string `default:"127.0.0.1:8080" placeholder:"ADDR" help:"The address to serve at."`
	serverFlags
}

// Run loads the models and serves until Ctrl-C.
func (c *ServeCmd) Run(ctx context.Context) error {
	s, err := c.server(ctx)
	if err != nil {
		return err
	}
	defer closeServer(s)
	srv := &http.Server{Addr: c.Listen, Handler: s.Handler()}
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
	s, err := c.server(ctx)
	if err != nil {
		return err
	}
	stop()
	code, err := openCode(s, c.Listen, openCodeDir(c.Cache), c.Bin, c.openCodeArgs())
	closeServer(s)
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
	if err := c.prepare(); err != nil {
		return err
	}
	m, err := chat.Resolve(ctx, c.Model, c.Dir, logw())
	if err != nil {
		return err
	}
	if m.Kind != chat.KindEmbed {
		return fmt.Errorf("%s is a %s model, not an embedding model", m.Name, m.Kind)
	}
	e, err := loadEmbedder(ctx, m, &c.loadFlags, &c.embedFlags)
	if err != nil {
		return err
	}
	defer e.Close()
	stop()
	return search(e)
}

// search is the interactive search by embeddings.
func search(e *chat.Embedder) error {
	type doc struct {
		text string
		v    []float32
	}
	var docs []doc
	console.Notef("%s: %d dimensions, up to %d tokens a text. Each input is a document to search; ?TEXT searches them, /list shows them, /reset clears them, /exit (or Ctrl-D) quits\n",
		e.Name, e.Dim(), e.MaxTokens())
	ed := console.NewEditor("search")
	for {
		text, ok := read(ed)
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
			console.Notef("[documents cleared]\n")
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
			printError(err)
			continue
		}
		cut := ""
		if r.Truncated {
			cut = fmt.Sprintf(", cut to %d", e.MaxTokens())
		}
		if !query {
			docs = append(docs, doc{text, r.Vector})
			console.Notef("[document %d: %d tokens%s in %.2fs]\n", len(docs), r.Tokens, cut, r.Time.Seconds())
			continue
		}
		if len(docs) == 0 {
			console.Notef("[no documents yet: type some first]\n")
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
		console.Notef("[query: %d tokens%s in %.2fs]\n", r.Tokens, cut, r.Time.Seconds())
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
	Dir    string `type:"path" placeholder:"DIR" help:"Where the models are kept (default: hwnpu/models in the user cache directory)."`
	Models string `placeholder:"FILES" help:"${models_usage}."`
}

// Run lists the models.
func (c *ListCmd) Run(ctx context.Context) error {
	lf := loadFlags{Dir: c.Dir, Models: c.Models}
	if err := lf.prepare(); err != nil {
		return err
	}
	showModels(os.Stdout, lf.Dir, chat.OnlyAMD(ctx, ""))
	return nil
}

// showModels writes the models of the definitions hwnpu-chat runs (chat.List) and what the
// features mean.
func showModels(w io.Writer, dir string, amd bool) {
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
		chat.List(w, dir, chat.KindChat, chat.KindEmbed, chat.KindDecision)
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
		"On AMD's NPU a name runs its export, ~/models/NAME.amd-npu2-q4 (-q6, or none for bf16); the first run\n"+
		"downloads the checkpoint and makes it with IRON (tools/amd/iron_export_lm.py, %s, INSTALL_AMD.md).\n"+
		"Models are kept in %s.\n", chat.IRONEnv, dir)
}

// shortPath is path with the home directory written ~.
func shortPath(path string) string {
	home, err := os.UserHomeDir()
	if err == nil && home != "" && strings.HasPrefix(path, home+string(filepath.Separator)) {
		return "~" + path[len(home):]
	}
	return path
}
