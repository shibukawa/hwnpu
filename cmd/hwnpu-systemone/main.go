// hwnpu-systemone answers typed questions about a state with a d1 decision model (Liquid AI's
// d1-2B) on the Neural Engine: the System One API of llama.cpp's server (POST /v1/systemone),
// served with -listen, or one request read from a file (or stdin) and answered on stdout.
//
//	hwnpu-convert -config d1-2B/config.json -weights d1-2B/model.safetensors -seq 512 -o d1-512.mlpackage
//	hwnpu-systemone -model d1-2B -package d1-512.mlpackage request.json
//	hwnpu-systemone -model d1-2B -package d1-512.mlpackage,d1-2048.mlpackage -listen 127.0.0.1:8080
//
// A request is {"state": ..., "questions": {"id": {"type": "choice" | "score" | "noul",
// "instructions": ..., "criteria": ...}}}, as llama.cpp's tools/server/README.md describes it.
// Each question is one call of a program: the shortest of -package that holds its prompt. -model
// holds config.json, model.safetensors (the token embeddings and the output layer's rows of the
// answer labels are read from it), tokenizer.json and calibration.json. -cpu runs the programs
// with the CPU interpreter instead, slowly, to check the answers on a machine without a Neural
// Engine.
//
// d1-2B also answers about images (files, images, or image_url parts of a state made of chat
// messages, as data URLs) with -vision, its vision encoder converted for one or more image sizes:
//
//	hwnpu-convert -arch lfm3-vision -image 512x512 -config d1-2B/config.json -weights d1-2B/model.safetensors -format fp16 -o vision-512x512.mlpackage
//
// d1-Omni-2B answers about audio clips too (WAV: data:audio/ URLs in files, or input_audio parts
// of the state), and questions asked as a clip (instructions {"type": "input_audio",
// "input_audio": {"data": BASE64}}), with -audio, its audio encoder converted for one or more
// clip lengths:
//
//	hwnpu-convert -arch lfm3-audio-encoder -frames 3000 -config d1-Omni-2B/config.json -weights d1-Omni-2B/model.safetensors -format fp16 -o audio-30s.mlpackage
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/systemone"
)

func main() {
	model := flag.String("model", "", "directory of the checkpoint: config.json, model.safetensors, tokenizer.json and calibration.json")
	pkgs := flag.String("package", "", "the model converted by hwnpu-convert, comma separated: .mlpackage or out.parts.json with different -seq")
	config := flag.String("config", "", "config.json (default: in -model)")
	weights := flag.String("weights", "", "checkpoint the embeddings and the label rows are read from: .safetensors or .gguf (default: in -model)")
	tok := flag.String("tokenizer", "", "tokenizer.json (default: next to config.json)")
	calib := flag.String("calibration", "", "calibration.json (default: next to config.json)")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory)")
	cpu := flag.Bool("cpu", false, "run on the CPU interpreter instead of the Neural Engine (slow; for checking)")
	listen := flag.String("listen", "", "serve POST /v1/systemone at this address, e.g. 127.0.0.1:8080")
	vis := flag.String("vision", "", "the LFM3-VL vision encoder converted by hwnpu-convert -arch lfm3-vision, a package per image size, comma separated: questions about images")
	aud := flag.String("audio", "", "the LFM3-Omni audio encoder converted by hwnpu-convert -arch lfm3-audio-encoder, a package per clip length, comma separated: d1-Omni's questions about audio and spoken questions")
	flag.Parse()
	if *pkgs == "" || *model == "" && (*config == "" || *weights == "") || flag.NArg() > 1 || *listen != "" && flag.NArg() > 0 {
		flag.Usage()
		os.Exit(2)
	}
	if *config == "" {
		*config = filepath.Join(*model, "config.json")
	}
	if *weights == "" {
		*weights = filepath.Join(*model, "model.safetensors")
	}
	if *cacheDir == "" {
		if d, err := os.UserCacheDir(); err == nil {
			*cacheDir = filepath.Join(d, "hwnpu")
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	c := systemone.Config{Packages: strings.Split(*pkgs, ","), ModelConfig: *config, Weights: *weights,
		Tokenizer: *tok, Calibration: *calib, CacheDir: *cacheDir, CPU: *cpu, Log: os.Stderr}
	if *vis != "" {
		c.Vision = strings.Split(*vis, ",")
	}
	if *aud != "" {
		c.Audio = strings.Split(*aud, ",")
	}
	m, err := systemone.Load(ctx, c)
	if err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-systemone:", err)
		os.Exit(1)
	}
	defer m.Close()
	if *listen != "" {
		err = serve(ctx, &systemone.Server{M: m, Name: m.Name, Log: os.Stderr}, *listen, m.MaxTokens())
	} else {
		err = answer(ctx, m, flag.Arg(0))
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-systemone:", err)
		os.Exit(1)
	}
}

func answer(ctx context.Context, m *systemone.Model, path string) error {
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

func serve(ctx context.Context, s *systemone.Server, addr string, maxTokens int) error {
	srv := &http.Server{Addr: addr, Handler: s.Handler()}
	go func() {
		<-ctx.Done()
		srv.Close()
	}()
	fmt.Fprintf(os.Stderr, "serving %s at http://%s/v1/systemone (prompts up to %d tokens)\n", s.Name, addr, maxTokens)
	if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
