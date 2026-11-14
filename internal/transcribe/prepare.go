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
	"github.com/shibukawa/hwnpu/internal/console"
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
	// the CPU interpreter runs), Core AI's programs (coreai, the three in one export
	// whisper-….coreai, llm.CoreAIExport) or OpenVINO IR for Intel's NPU, kept in Dir/intel.
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
	progs := []struct {
		role string // its name in the Core AI export
		arch *llm.Arch
		opts llm.Options
		path *string
	}{
		{"encoder", whisper.EncoderArch, llm.Options{}, &c.Encoder},
		{"cross", whisper.CrossArch, llm.Options{}, &c.Cross},
		{"decoder", whisper.Arch, llm.Options{Seq: 1, Cache: wc.TargetPositions}, &c.Decoder},
	}
	if s.Backend == llm.BackendCoreAI {
		// one export of the three, written side by side (as every model's Core AI programs)
		format := cmp.Or(s.Format, whisper.Arch.Format)
		export, done := llm.CoreAIExport(s.Dir, fmt.Sprintf("whisper-r%d.%d.%d-%s-c%d", whisper.EncoderArch.Revision,
			whisper.CrossArch.Revision, whisper.Arch.Revision, format, wc.TargetPositions))
		if !llm.FetchExport(ctx, export, done) {
			weights, err := checkpoint()
			if err != nil {
				return Config{}, err
			}
			if err := llm.StartExport(export); err != nil {
				return Config{}, err
			}
			fmt.Fprintf(log, "converting %s for the Neural Engine (Core AI, %s); this takes a minute\n", filepath.Base(export), format)
			jobs := console.NewJobs(log)
			for _, p := range progs {
				f := cmp.Or(s.Format, p.arch.Format)
				p.opts.Backend = s.Backend
				jobs.Add(p.role, func() (string, error) {
					_, err := convert(config, weights, filepath.Join(export, p.role+".aimodel"), p.arch, f, p.opts, io.Discard)
					return f, err
				})
			}
			// a job holds its program's weights (up to the target's limit) and the tensors it reads
			if err := jobs.Run(4 << 30); err != nil {
				return Config{}, err
			}
			if err := llm.FinishExport(export, done, filepath.Join(s.Dir, llm.BackendCoreAI)); err != nil {
				return Config{}, err
			}
			llm.ShareExport(ctx, export)
		}
		for _, p := range progs {
			*p.path = exportProgram(export, p.role)
		}
	}
	for _, p := range progs {
		if s.Backend == llm.BackendCoreAI {
			break
		}
		format := cmp.Or(s.Format, p.arch.Format)
		dir, ext := s.Dir, llm.ProgramExt(s.Backend)
		if intel {
			format = llm.IntelFormat(format)
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
	if c.Weights, err = chat.Tables(ctx, filepath.Join(s.Dir, chat.TablesFile), whisper.Arch, llm.HostTables(dec), checkpoint, log); err != nil {
		return Config{}, err
	}
	// a conversion reads the checkpoint into memory: give it back now
	debug.FreeOSMemory()
	return c, nil
}

// exportProgram is the program role of a Core AI export: role.aimodel, or role.parts.json when it
// was split under the Neural Engine's limit (llm.Convert).
func exportProgram(export, role string) string {
	if p := filepath.Join(export, role+".parts.json"); fileExists(p) {
		return p
	}
	return filepath.Join(export, role+".aimodel")
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
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
