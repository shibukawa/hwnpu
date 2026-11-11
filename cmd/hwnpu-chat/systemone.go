package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/systemone"
)

// SystemOneCmd is "hwnpu-chat systemone": typed questions about a state answered by a d1
// decision model (Liquid AI's d1-2B) on the Neural Engine, as the System One API of llama.cpp's
// server (POST /v1/systemone): one request read from a file (or stdin) and answered on stdout,
// or served with --listen.
//
//	hwnpu-chat systemone request.json                     # -m decision: d1-2B, downloaded and converted when missing
//	hwnpu-chat systemone -m d1-omni --listen 127.0.0.1:8080
//
// The first run downloads the model into --dir and converts its programs by itself
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
	Dir     string `type:"path" placeholder:"DIR" help:"Where the models are kept (default: hwnpu/models in the user cache directory)."`
	Cache   string `placeholder:"DIR" help:"Compile cache (default: hwnpu in the user cache directory)."`
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
func (c *SystemOneCmd) Run(ctx context.Context) error {
	lw := logw()
	lf := loadFlags{Dir: c.Dir, Cache: c.Cache, Models: c.Models}
	if err := lf.prepare(); err != nil {
		return err
	}
	if c.Listen != "" && c.Request != "" {
		return errors.New("--listen serves; it takes no request")
	}
	var cfg systemone.Config
	if c.Package != "" {
		if c.Checkpoint == "" && (c.Config == "" || c.Weights == "") {
			return errors.New("--package needs --checkpoint (or --config and --weights)")
		}
		if c.Config == "" {
			c.Config = filepath.Join(c.Checkpoint, "config.json")
		}
		if c.Weights == "" {
			c.Weights = filepath.Join(c.Checkpoint, "model.safetensors")
		}
		cfg = systemone.Config{Packages: split(c.Package), ModelConfig: c.Config, Weights: c.Weights}
	} else {
		seqs, err := parseInts(c.Seq)
		if err != nil {
			return fmt.Errorf("--seq: %w", err)
		}
		m, err := chat.ResolveFor(c.Model, chat.KindDecision, lw)
		if err != nil {
			return err
		}
		backend := ""
		if !c.CPU { // the CPU interpreter runs Core ML packages
			if backend, err = llm.ProgramFormat(ctx, lf.Backend); err != nil {
				return err
			}
		}
		if cfg, err = systemone.Prepare(ctx, systemone.Setup{Dir: filepath.Join(lf.Dir, m.DirName()), Repo: m.Source(), Seqs: seqs, Format: c.Format, Backend: backend, Log: lw}); err != nil {
			return err
		}
		cfg.Name = m.Name
	}
	cfg.Tokenizer, cfg.Calibration, cfg.CacheDir, cfg.CPU, cfg.Log, cfg.Backend = c.Tokenizer, c.Calibration, lf.Cache, c.CPU, lw, lf.Backend
	cfg.Vision, cfg.Audio = split(c.Vision), split(c.Audio)
	m, err := systemone.Load(ctx, cfg)
	if err != nil {
		return err
	}
	defer m.Close()
	if c.Listen != "" {
		return serveSystemOne(ctx, &systemone.Server{M: m, Name: m.Name, Log: lw}, c.Listen, m.MaxTokens())
	}
	console.Running() // a server's or a conversation's times are not of the command's work
	return answerSystemOne(ctx, m, c.Request)
}

func answerSystemOne(ctx context.Context, m *systemone.Model, path string) error {
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
	req, err := systemone.ParseRequest(raw)
	if err != nil {
		return err
	}
	res, err := m.Decide(ctx, req)
	if err != nil {
		return err
	}
	b, err := json.Marshal(res)
	if err != nil {
		return err
	}
	var out bytes.Buffer
	json.Indent(&out, b, "", "  ")
	out.WriteByte('\n')
	_, err = os.Stdout.Write(out.Bytes())
	return err
}

func serveSystemOne(ctx context.Context, s *systemone.Server, addr string, maxTokens int) error {
	srv := &http.Server{Addr: addr, Handler: s.Handler()}
	go func() {
		<-ctx.Done()
		srv.Close()
	}()
	fmt.Fprintf(os.Stderr, "serving %s (prompts up to %d tokens) at http://%s/v1/systemone\n", s.Name, maxTokens, addr)
	if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
