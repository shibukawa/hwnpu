package transcribe

import (
	"cmp"
	"context"
	"flag"
	"fmt"
	"github.com/shibukawa/hwnpu/internal/cache"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/whisper"
)

// DefaultRepo is the checkpoint hwnpu-audio transcribe uses when not told.
const DefaultRepo = "openai/whisper-large-v3-turbo"

// Setup says what Prepare downloads and converts.
type Setup struct {
	// Dir is where the checkpoint and the programs are kept; Repo is the Hugging Face repository
	// it is downloaded from when it is not there.
	Dir, Repo string
	// Format is the weight format of the programs (empty: the architectures', lut8; int8 on
	// Intel's NPU, which reads no lookup tables).
	Format string
	// Backend is the program format (llm.Options.Backend): Core ML packages (the default, what
	// the CPU interpreter runs), Core AI's programs (coreai, kept in Dir/coreai) or OpenVINO IR
	// for Intel's NPU, kept in Dir/intel.
	Backend string
	Log     io.Writer
}

// files are the checkpoint's files besides the weights.
var files = []string{"config.json", "generation_config.json", "tokenizer.json"}

// Prepare downloads the checkpoint into s.Dir unless it is there, and converts its three
// programs for the Neural Engine (or Intel's NPU, s.Backend) unless they are: the encoder (whisper-encoder), the cross
// program (whisper-cross) and the decoder with a KV cache of every position it takes, one token
// per call.
func Prepare(ctx context.Context, s Setup) (Config, error) {
	log := cmp.Or[io.Writer](s.Log, io.Discard)
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return Config{}, err
	}
	for _, f := range files {
		if err := chat.Fetch(ctx, s.Repo, f, filepath.Join(s.Dir, f), log); err != nil {
			return Config{}, err
		}
	}
	config := filepath.Join(s.Dir, "config.json")
	raw, err := os.ReadFile(config)
	if err != nil {
		return Config{}, err
	}
	wc, err := whisper.ParseConfig(raw)
	if err != nil {
		return Config{}, err
	}
	c := Config{Dir: s.Dir}
	intel := s.Backend == llm.BackendIntel
	// the checkpoint is downloaded only to convert a program or write the tables from it: what
	// was converted runs without it
	checkpoint := sync.OnceValues(func() (string, error) {
		return filepath.Join(s.Dir, "model.safetensors"), chat.FetchWeights(ctx, s.Repo, s.Dir, log)
	})
	for _, p := range []struct {
		arch *llm.Arch
		opts llm.Options
		path *string
	}{
		{whisper.EncoderArch, llm.Options{}, &c.Encoder},
		{whisper.CrossArch, llm.Options{}, &c.Cross},
		{whisper.Arch, llm.Options{Seq: 1, Cache: wc.TargetPositions}, &c.Decoder},
	} {
		format := cmp.Or(s.Format, p.arch.Format)
		dir, ext := s.Dir, llm.ProgramExt(s.Backend)
		if intel || s.Backend == llm.BackendCoreAI {
			if intel {
				format = llm.IntelFormat(format)
			}
			dir = filepath.Join(s.Dir, s.Backend)
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return Config{}, err
			}
		}
		p.opts.Backend = s.Backend
		name := p.arch.Name + p.arch.Rev() + "-" + format
		if p.opts.Cache > 0 {
			name += fmt.Sprintf("-c%d-q%d", p.opts.Cache, p.opts.Seq)
		}
		out := filepath.Join(dir, name+ext)
		done := strings.TrimSuffix(out, ext) + ".done"
		if path, ok := cache.ReadDone(done); ok {
			*p.path = path
			continue
		}
		weights, err := checkpoint()
		if err != nil {
			return Config{}, err
		}
		path, err := convert(config, weights, out, p.arch, format, p.opts, log)
		if err != nil {
			return Config{}, err
		}
		if err := cache.WriteDone(done, path); err != nil {
			return Config{}, err
		}
		*p.path = path
	}
	// the decoder's token embeddings, which the host looks up
	_, dec, err := llm.LoadConfig(config, whisper.Arch.Name)
	if err != nil {
		return Config{}, err
	}
	if c.Weights, err = chat.Tables(filepath.Join(s.Dir, chat.TablesFile), whisper.Arch, llm.HostTables(dec), checkpoint, log); err != nil {
		return Config{}, err
	}
	// a conversion reads the checkpoint into memory: give it back now
	debug.FreeOSMemory()
	return c, nil
}

// convert writes one program with the converter's defaults (hwnpu-dev convert's), the lookup tables
// fitted as hwnpu-chat fits them.
func convert(config, weights, out string, a *llm.Arch, format string, o llm.Options, log io.Writer) (string, error) {
	_, model, err := llm.LoadConfig(config, a.Name)
	if err != nil {
		return "", err
	}
	w, err := llm.OpenWeights(weights, a)
	if err != nil {
		return "", err
	}
	defer w.Close()
	fs := flag.NewFlagSet("", flag.ContinueOnError)
	var opts llm.Options
	set := llm.BindFlags(fs, &opts, "")
	if err := fs.Parse(nil); err != nil {
		return "", err
	}
	set()
	opts.Format, opts.Seq, opts.Cache, opts.Backend = format, o.Seq, o.Cache, o.Backend
	if strings.HasPrefix(format, "lut") {
		opts.Quant = "kmeans-minmax"
	}
	target := "the Neural Engine"
	switch o.Backend {
	case llm.BackendIntel:
		target = "Intel's NPU"
	case llm.BackendCoreAI:
		target = "the Neural Engine (Core AI)"
	}
	fmt.Fprintf(log, "converting %s for %s (%s); this takes a minute\n", filepath.Base(out), target, format)
	return llm.Convert(a, model, w, out, "any", 0, opts, log)
}
