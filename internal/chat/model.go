package chat

import (
	"cmp"
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
)

var hubURL = "https://huggingface.co/"

// DefaultContext is the longest cache the Neural Engine takes with 64 tokens per call: a tensor
// dimension goes up to llm.NeuralEngineMaxDim, and the key axis is the cache plus the tokens.
const DefaultContext = llm.NeuralEngineMaxDim - prefillSeq

const prefillSeq = 64

// Setup says where the model is and how to get it.
type Setup struct {
	Dir     string // holds config.json, tokenizer.json, model.safetensors and the converted programs
	Repo    string // Hugging Face repository the missing checkpoint files come from
	Context int    // cache length of the programs
	// Draft makes the program that generates take this many tokens per call instead of 1, for
	// checking the drafts of speculative decoding (Engine.Draft); 0 or 1 for none. It is one
	// program either way: the Neural Engine takes about 100 ms to switch between programs, so
	// generation runs on one.
	Draft int
	Log   io.Writer
}

// Prepare makes the model ready in Dir: it downloads the checkpoint files that are missing from
// Hugging Face, and converts the programs that are missing, one of 1 token per call to generate
// and one of 64 to read prompts, both with a cache of Context tokens. It returns the Config that
// loads them.
func Prepare(ctx context.Context, s Setup) (Config, error) {
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return Config{}, err
	}
	for _, f := range []string{"config.json", "tokenizer.json", "model.safetensors"} {
		if err := fetch(ctx, s.Repo, f, filepath.Join(s.Dir, f), s.Log); err != nil {
			return Config{}, err
		}
	}
	c := Config{ModelConfig: filepath.Join(s.Dir, "config.json"), Weights: filepath.Join(s.Dir, "model.safetensors"), Name: filepath.Base(s.Repo)}
	a, _, err := llm.LoadConfig(c.ModelConfig, "")
	if err != nil {
		return Config{}, err
	}
	progs := []struct{ seq, parts int }{{1, 2}, {prefillSeq, 0}}
	if s.Draft > 1 {
		progs[0] = struct{ seq, parts int }{s.Draft, 0}
	}
	for _, p := range progs {
		out := filepath.Join(s.Dir, fmt.Sprintf("%s-c%d-q%d.mlpackage", a.Name, s.Context, p.seq))
		path, err := convert(ctx, c.ModelConfig, c.Weights, out, program{seq: p.seq, parts: p.parts, cache: s.Context}, s.Log)
		if err != nil {
			return Config{}, err
		}
		c.Packages = append(c.Packages, path)
	}
	for _, e := range encoders {
		if _, _, err := llm.LoadConfig(c.ModelConfig, e.arch); err != nil {
			continue // the checkpoint has no such encoder
		}
		for _, f := range e.files {
			if err := fetch(ctx, s.Repo, f, filepath.Join(s.Dir, f), s.Log); err != nil {
				return Config{}, err
			}
		}
		out := filepath.Join(s.Dir, fmt.Sprintf("%s-%s.mlpackage", e.arch, e.value))
		path, err := convert(ctx, c.ModelConfig, c.Weights, out, program{arch: e.arch, format: "fp16", params: map[string]string{e.param: e.value}}, s.Log)
		if err != nil {
			return Config{}, err
		}
		*e.packages(&c) = []string{path}
	}
	return c, nil
}

// encoders are the programs of the multimodal checkpoints besides the text model, converted when
// config.json has them: LFM3-VL's and LFM3-Omni's vision encoder for one image size (an image is
// stretched to it), LFM3-Audio's and LFM3-Omni's audio encoder for clips of up to 30 seconds. They
// stay in fp16 as hwnpu-convert's examples convert them (docs/design.md 16.28, 16.29); files are
// what else of the repository they read.
var encoders = []struct {
	arch, param, value string
	files              []string
	packages           func(*Config) *[]string
}{
	{"lfm3-vision", "image", "512x512", []string{"processor_config.json"}, func(c *Config) *[]string { return &c.Vision }},
	{"lfm3-audio-encoder", "frames", "3000", nil, func(c *Config) *[]string { return &c.Audio }},
}

// program is what convert writes: the program of arch (empty: the one config.json names) with
// the weights in format (empty: the architecture's default) and the architecture's own settings
// in params; seq tokens per call, in parts (0: one) and with a KV cache of cache tokens (0: none).
type program struct {
	arch, format      string
	params            map[string]string
	seq, parts, cache int
}

// convert writes the program of the model (config.json and the checkpoint) unless it is there,
// and returns its path. NAME.done, written last, holds the path, so that a conversion cut off is
// done again.
func convert(ctx context.Context, config, weights, out string, p program, log io.Writer) (string, error) {
	done := out[:len(out)-len(".mlpackage")] + ".done"
	if b, err := os.ReadFile(done); err == nil {
		return string(b), nil
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	a, model, err := llm.LoadConfig(config, p.arch)
	if err != nil {
		return "", err
	}
	w, err := llm.OpenWeights(weights, a)
	if err != nil {
		return "", err
	}
	defer w.Close()
	// the converter's defaults and the architecture's format, with the table fitting that keeps
	// LFM3's answers as they were (also used for the embedding model)
	fs := flag.NewFlagSet("", flag.ContinueOnError)
	var opts llm.Options
	set := llm.BindFlags(fs, &opts, "")
	if err := fs.Parse([]string{"-quant", "kmeans-minmax"}); err != nil {
		return "", err
	}
	set()
	opts.Format = cmp.Or(p.format, a.DefaultFormat(model))
	opts.Seq, opts.Cache = p.seq, p.cache
	for k, v := range p.params {
		opts.Params[k] = v
	}
	switch {
	case p.cache > 0:
		fmt.Fprintf(log, "converting %s for the Neural Engine (cache %d, %d per call); this takes a minute or two\n", filepath.Base(out), p.cache, p.seq)
	case p.seq > 0:
		fmt.Fprintf(log, "converting %s for the Neural Engine (%d tokens); this takes a minute or two\n", filepath.Base(out), p.seq)
	default:
		fmt.Fprintf(log, "converting %s for the Neural Engine; this takes a minute or two\n", filepath.Base(out))
	}
	path, err := llm.Convert(a, model, w, out, "any", p.parts, opts, log)
	if err != nil {
		return "", err
	}
	return path, os.WriteFile(done, []byte(path), 0o644)
}

// fetch downloads a file of a Hugging Face repository to path unless it is there. A download cut
// off goes on from where it stopped (path.part). HF_TOKEN, when set, is sent for gated models.
func fetch(ctx context.Context, repo, name, path string, log io.Writer) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	url := hubURL + repo + "/resolve/main/" + name
	part := path + ".part"
	var have int64
	if fi, err := os.Stat(part); err == nil {
		have = fi.Size()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	if have > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", have))
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	flags := os.O_CREATE | os.O_WRONLY | os.O_TRUNC
	switch resp.StatusCode {
	case http.StatusOK:
		have = 0
	case http.StatusPartialContent:
		flags = os.O_CREATE | os.O_WRONLY | os.O_APPEND
	default:
		return fmt.Errorf("chat: downloading %s: %s", url, resp.Status)
	}
	total := have + resp.ContentLength
	size := ""
	if resp.ContentLength >= 0 {
		size = fmt.Sprintf(" (%.2f GB)", float64(total)/1e9)
	}
	fmt.Fprintf(log, "downloading %s from %s%s to %s\n", name, repo, size, filepath.Dir(path))
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	pw := &progress{w: f, done: have, total: total, name: name, log: log}
	_, err = io.Copy(pw, resp.Body)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if pw.shown {
		fmt.Fprintln(log)
	}
	if err != nil {
		return fmt.Errorf("chat: downloading %s: %w (run again to go on)", name, err)
	}
	return os.Rename(part, path)
}

// progress writes through and shows how far a download is, once a second.
type progress struct {
	w           io.Writer
	done, total int64
	name        string
	log         io.Writer
	last        time.Time
	shown       bool
}

func (p *progress) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.done += int64(n)
	if time.Since(p.last) >= time.Second && p.total > 1e7 {
		fmt.Fprintf(p.log, "\r%s: %.2f / %.2f GB", p.name, float64(p.done)/1e9, float64(p.total)/1e9)
		p.last, p.shown = time.Now(), true
	}
	return n, err
}
