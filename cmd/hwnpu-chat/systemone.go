package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/server/systemone"
	"github.com/shibukawa/hwnpu/ai/text"
	"github.com/shibukawa/hwnpu/internal/console"
	sysone "github.com/shibukawa/hwnpu/internal/systemone"
)

// SystemOneCmd is "hwnpu-chat systemone": typed questions about a state answered by a d1
// decision model (Liquid AI's d1-2B) on the Neural Engine, as the System One API of llama.cpp's
// server (POST /v1/systemone): one request read from a file (or stdin) and answered on stdout,
// or served with --listen.
//
//	hwnpu-chat systemone request.json                     # -m decision: d1-2B, downloaded and converted when missing
//	hwnpu-chat systemone -m d1-omni --listen 127.0.0.1:8080
//
// The first run downloads the model and converts its programs by itself
// (internal/systemone Prepare: the model for prompts of --seq tokens, and the vision and audio
// encoders the checkpoint has). Programs converted by hand are given with --checkpoint and
// --package:
//
//	hwnpu-dev convert -config d1-2B/config.json -weights d1-2B/model.safetensors -seq 512 -o d1-512.mlpackage
//	hwnpu-chat systemone --checkpoint d1-2B --package d1-512.mlpackage request.json
//	hwnpu-chat systemone --checkpoint d1-2B --package d1-512.mlpackage,d1-2048.mlpackage --listen 127.0.0.1:8080
//
// A request is {"state": ..., "questions": {"id": {"type": "choice" | "score" | "noul",
// "instructions": ..., "criteria": ...}}}, as llama.cpp's tools/server/README.md describes it.
// Each question is one call of a program: the shortest of --package that holds its prompt.
// --checkpoint holds config.json, model.safetensors (the token embeddings and the output layer's
// rows of the answer labels are read from it), tokenizer.json and calibration.json.
//
// d1-2B also answers about images (files, images, or image_url parts of a state made of chat
// messages, as data URLs) with --vision, its vision encoder converted for one or more image sizes:
//
//	hwnpu-dev convert -arch lfm3-vision -image 512x512 -config d1-2B/config.json -weights d1-2B/model.safetensors -format fp16 -o vision-512x512.mlpackage
//
// d1-Omni-2B answers about audio clips too (WAV: data:audio/ URLs in files, or input_audio parts
// of the state), and questions asked as a clip (instructions {"type": "input_audio",
// "input_audio": {"data": BASE64}}), with --audio-encoder, its audio encoder converted for one or
// more clip lengths:
//
//	hwnpu-dev convert -arch lfm3-audio-encoder -frames 3000 -config d1-Omni-2B/config.json -weights d1-Omni-2B/model.safetensors -format fp16 -o audio-30s.mlpackage
type SystemOneCmd struct {
	Request string `arg:"" optional:"" placeholder:"REQUEST.json" help:"The request to answer (- or none: standard input)."`
	Model   string `short:"m" default:"decision" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (decision, d1, d1-omni), or a Hugging Face repository OWNER/NAME of a d1 checkpoint."`
	Listen  string `placeholder:"ADDR" help:"Serve POST /v1/systemone at this address (127.0.0.1:8080) instead of answering one request."`
	Seq     string `default:"512" placeholder:"N,..." help:"Prompt lengths of the programs converted, comma separated: a prompt runs on the shortest that holds it."`
	Format  string `placeholder:"FORMAT" help:"Weight format of the d1 model's programs (default: lut8)."`
	Models  string `placeholder:"FILES" help:"${models_usage}."`
	CPU     bool   `name:"cpu" help:"Run on the CPU interpreter instead of the Neural Engine (slow; for checking)."`

	Checkpoint  string `group:"Converted by hand" type:"existingdir" placeholder:"DIR" help:"With --package: the checkpoint's directory (config.json, model.safetensors, tokenizer.json, calibration.json)."`
	Package     string `group:"Converted by hand" placeholder:"PKG,..." help:"The model converted by hwnpu-dev convert, comma separated: .mlpackage or out.parts.json with different -seq."`
	Config      string `group:"Converted by hand" type:"existingfile" placeholder:"FILE" help:"config.json (default: in --checkpoint)."`
	Weights     string `group:"Converted by hand" type:"existingfile" placeholder:"FILE" help:"The checkpoint the embeddings and the label rows are read from: .safetensors or .gguf (default: in --checkpoint)."`
	Tokenizer   string `group:"Converted by hand" type:"existingfile" placeholder:"FILE" help:"tokenizer.json (default: next to config.json)."`
	Calibration string `group:"Converted by hand" type:"existingfile" placeholder:"FILE" help:"calibration.json (default: next to config.json)."`
	Vision      string `group:"Converted by hand" placeholder:"PKG,..." help:"The vision encoder (hwnpu-dev convert -arch lfm3-vision), a package per image size, comma separated: questions about images."`
	Audio       string `group:"Converted by hand" name:"audio-encoder" placeholder:"PKG,..." help:"The audio encoder (hwnpu-dev convert -arch lfm3-audio-encoder), a package per clip length, comma separated: d1-Omni's questions about audio and spoken questions."`
}

// Run answers the request or serves.
func (c *SystemOneCmd) Run(ctx context.Context, stop context.CancelFunc) error {
	lw := logw()
	if c.Listen != "" && c.Request != "" {
		return errors.New("--listen serves; it takes no request")
	}
	lf := loadFlags{Models: c.Models}
	st, cat, err := lf.prepare(ctx)
	if err != nil {
		return err
	}
	u, defs, err := c.use(cat)
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
	d, err := u.Get(ctx, rt)
	if err != nil {
		return err
	}
	if c.Listen != "" {
		return serveSystemOne(ctx, stop, &systemone.Server{D: d, Log: lw}, c.Listen, d.MaxTokens())
	}
	console.Running() // a server's or a conversation's times are not of the command's work
	return answerSystemOne(ctx, d, c.Request)
}

// use is the use of the decision model --model names, as the flags make it: a model of its own
// for --seq, --format, --cpu and the programs converted by hand.
func (c *SystemOneCmd) use(cat *ai.Catalog) (ai.Use[*text.Decider], []ai.Definition, error) {
	u, defs, err := useFor(cat, text.Decide(c.Model))
	if err != nil {
		return u, nil, err
	}
	if _, err := parseInts(c.Seq); err != nil {
		return u, nil, fmt.Errorf("--seq: %w", err)
	}
	d := ai.Definition{Like: u.Name()}
	set, ps := map[string]string{}, map[string][]string{}
	if c.Seq != ints(sysone.DefaultSeqs) {
		set["seqs"] = c.Seq
	}
	if c.Format != "" {
		set["format"] = c.Format
	}
	if c.CPU {
		d.Device = "cpu"
	}
	if c.Package != "" {
		if c.Checkpoint == "" && (c.Config == "" || c.Weights == "") {
			return u, nil, errors.New("--package needs --checkpoint (or --config and --weights)")
		}
		config, weights := c.Config, c.Weights
		if config == "" {
			config = filepath.Join(c.Checkpoint, "config.json")
		}
		if weights == "" {
			weights = filepath.Join(c.Checkpoint, "model.safetensors")
		}
		ps["packages"], ps["config"], ps["weights"] = absolutes(c.Package), []string{absolute(config)}, []string{absolute(weights)}
	}
	for name, v := range map[string]string{"tokenizer": c.Tokenizer, "calibration": c.Calibration} {
		if v != "" {
			ps[name] = []string{absolute(v)}
		}
	}
	if c.Vision != "" {
		ps["vision"] = absolutes(c.Vision)
	}
	if c.Audio != "" {
		ps["audio"] = absolutes(c.Audio)
	}
	if len(set) > 0 {
		d.Settings = set
	}
	if len(ps) > 0 {
		d.Programs = ps
	}
	if d.Settings == nil && d.Programs == nil && d.Device == "" {
		return u, defs, nil
	}
	return text.Decide(d.ID()), append(defs, d), nil
}

func answerSystemOne(ctx context.Context, d *text.Decider, path string) error {
	var raw []byte
	var err error
	if path == "" || path == "-" {
		raw, err = io.ReadAll(os.Stdin)
	} else {
		raw, err = os.ReadFile(path)
	}
	if err != nil {
		return err
	}
	res, err := d.Decide(ctx, raw)
	if err != nil {
		return err
	}
	var out bytes.Buffer
	json.Indent(&out, res.JSON, "", "  ")
	out.WriteByte('\n')
	_, err = os.Stdout.Write(out.Bytes())
	return err
}

func serveSystemOne(ctx context.Context, stop context.CancelFunc, s *systemone.Server, addr string, maxTokens int) error {
	base, cancelReqs := context.WithCancel(context.Background())
	defer cancelReqs()
	srv := &http.Server{
		Addr:        addr,
		Handler:     s.Handler(),
		BaseContext: func(net.Listener) context.Context { return base },
	}
	listenDone := make(chan struct{})
	shutdownDone := make(chan struct{})
	go func() {
		defer close(shutdownDone)
		select {
		case <-ctx.Done():
			stop() // as ServeCmd.Run: Ctrl-C waits for active calls before closing models
			cancelReqs()
			sctx, scancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer scancel()
			if err := srv.Shutdown(sctx); err != nil {
				srv.Close()
			}
		case <-listenDone:
		}
	}()
	fmt.Fprintf(os.Stderr, "serving %s (prompts up to %d tokens) at http://%s/v1/systemone\n", s.D.Name(), maxTokens, addr)
	err := srv.ListenAndServe()
	close(listenDone)
	<-shutdownDone
	if !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
