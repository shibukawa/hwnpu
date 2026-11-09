package systemone

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"sync"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
)

// DefaultSeqs are the prompt lengths of the programs Prepare converts: one call reads a whole
// prompt, on the shortest program that holds it.
var DefaultSeqs = []int{512}

// Setup says what Prepare downloads and converts.
type Setup struct {
	// Dir is where the checkpoint and the programs are kept; Repo is the Hugging Face repository
	// it is downloaded from when it is not there (OWNER/NAME, or OWNER/NAME@REVISION).
	Dir, Repo string
	// Seqs are the prompt lengths of the programs (none: DefaultSeqs).
	Seqs []int
	// Format is the weight format of the d1 model's programs (empty: lut8; the d1 models answer
	// worse in 4 bit tables). The encoders stay in fp16.
	Format string
	Log    io.Writer
}

// Prepare downloads the d1 checkpoint into s.Dir unless it is there, converts its programs for
// the Neural Engine unless they are (the model for each prompt length; the vision encoder for
// images of 512x512 and, for d1-Omni, the audio encoder for clips up to 30 seconds, when the
// checkpoint has them), and writes the tables the host reads (the token embeddings and the
// output layer's rows of the answer labels), so that it runs without the checkpoint. The
// checkpoint is downloaded only when something is converted or written from it.
func Prepare(ctx context.Context, s Setup) (Config, error) {
	log := cmp.Or[io.Writer](s.Log, io.Discard)
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return Config{}, err
	}
	for _, f := range []string{"config.json", "tokenizer.json", "calibration.json"} {
		if err := chat.Fetch(ctx, s.Repo, f, filepath.Join(s.Dir, f), log); err != nil {
			return Config{}, err
		}
	}
	config := filepath.Join(s.Dir, "config.json")
	a, model, err := llm.LoadConfig(config, "")
	if err != nil {
		return Config{}, err
	}
	cfg, ok := model.(*lfm3.Config)
	if !ok || !cfg.D1 {
		return Config{}, fmt.Errorf("systemone: %s is not a d1 model (LFM3 with architectures D1Model)", s.Repo)
	}
	checkpoint := sync.OnceValues(func() (string, error) {
		return filepath.Join(s.Dir, "model.safetensors"), chat.FetchWeights(ctx, s.Repo, s.Dir, log)
	})
	convert := func(name string, p chat.Program) (string, error) {
		out := filepath.Join(s.Dir, name+".mlpackage")
		weights := ""
		if !chat.Converted(out) {
			var err error
			if weights, err = checkpoint(); err != nil {
				return "", err
			}
		}
		return chat.ConvertProgram(ctx, config, weights, out, p, log)
	}
	c := Config{ModelConfig: config}
	format := cmp.Or(s.Format, "lut8")
	seqs := slices.Clone(s.Seqs)
	if len(seqs) == 0 {
		seqs = slices.Clone(DefaultSeqs)
	}
	slices.Sort(seqs)
	for _, seq := range slices.Compact(seqs) {
		path, err := convert(fmt.Sprintf("d1-%s%s-%s-s%d", a.Name, a.Rev(), format, seq), chat.Program{Format: format, Seq: seq})
		if err != nil {
			return Config{}, err
		}
		c.Packages = append(c.Packages, path)
	}
	if _, _, err := llm.LoadConfig(config, lfm3vl.Arch.Name); err == nil {
		if err := chat.Fetch(ctx, s.Repo, "processor_config.json", filepath.Join(s.Dir, "processor_config.json"), log); err != nil && !chat.NotFound(err) {
			return Config{}, err
		}
		path, err := convert(lfm3vl.Arch.Name+lfm3vl.Arch.Rev()+"-512x512", chat.Program{Arch: lfm3vl.Arch.Name, Format: "fp16", Params: map[string]string{"image": "512x512"}})
		if err != nil {
			return Config{}, err
		}
		c.Vision = []string{path}
	}
	if _, _, err := llm.LoadConfig(config, lfm3audio.Arch.Name); err == nil {
		path, err := convert(lfm3audio.Arch.Name+lfm3audio.Arch.Rev()+"-3000", chat.Program{Arch: lfm3audio.Arch.Name, Format: "fp16", Params: map[string]string{"frames": "3000"}})
		if err != nil {
			return Config{}, err
		}
		c.Audio = []string{path}
	}
	names := []string{cfg.Embedding(), cfg.Head()}
	slices.Sort(names)
	if c.Weights, err = chat.Tables(filepath.Join(s.Dir, chat.TablesFile), a, slices.Compact(names), checkpoint, log); err != nil {
		return Config{}, err
	}
	// a conversion reads the checkpoint into memory: give it back now
	debug.FreeOSMemory()
	return c, nil
}
