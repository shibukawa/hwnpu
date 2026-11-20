package main

import (
	"cmp"
	"context"
	"fmt"
	"github.com/shibukawa/hwnpu/internal/cache"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm/experts"
)

// loadFlags are where the models are and how they are made ready, for every command that runs one.
type loadFlags struct {
	Backend     string        `enum:",coreai,intel,levelzero,openvino" default:"" help:"The NPU to convert for and run on: coreai (Apple's Neural Engine through Core AI, macOS 27: a Mac's only one; the models converted for it from their checkpoints), levelzero or intel (Intel's NPU), openvino (OpenVINO, HWNPU_OV_DEVICE, for checking without the NPU) (default: HWNPU_BACKEND, else the NPU found)."`
	Cache       string        `placeholder:"DIR" help:"Compile cache (default: hwnpu in the user cache directory)."`
	PromptCache string        `placeholder:"DIR" help:"Keep the prompts read on disk, so that a prompt beginning like an earlier one is not read again after a restart (default: prompts in the compile cache; \"off\" for none)."`
	Idle        time.Duration `default:"5m" help:"AMD's NPU: give the NPU's memory back after this long without a request and load the model again for the next one (0: never); the conversation goes on from the host's copy. A mixture of experts whose experts run on the CPU: keep only the core ones wired after this long."`
	ExpertsKey  string        `placeholder:"KEY" help:"A mixture of experts whose experts do not all fit in memory: the name of this application's counts of the experts it runs, which choose the ones kept wired (default: the command, chat, agent or serve; off: none read or kept)."`
	Models      string        `placeholder:"FILES" help:"${models_usage}."`
}

// prepare reads the definitions and fills in the defaults of the directories.
func (f *loadFlags) prepare() error {
	if err := chat.LoadUserModels(f.Models, logw()); err != nil {
		return err
	}
	var err error
	if f.Backend, err = resolveBackend(context.Background(), f.Backend); err != nil {
		return err
	}
	if f.Cache == "" {
		f.Cache = cache.Root()
	}
	return nil
}

// convertFlags are how a chat model is converted and run.
type convertFlags struct {
	Context   int     `placeholder:"TOKENS" help:"Tokens of a prompt and its answer together; 0: the model's definition (models.yaml's context), else the longest the model was trained for when it is split at its attention (--paged: 262144 for LFM3-1.7B and Qwen3.5, 131072 for Gemma 4, its cache growing with the conversation within a quarter of the memory), else 16320 (4096 for Qwen3 not split, whose 28 layers of keys and values in one program fail past it; over 16320 a whole model's cache is in chunks of 16384, slower per token)."`
	Paged     int     `placeholder:"TOKENS" help:"Split a chat model at its full attention layers and attend over the cache in chunks of this many tokens, so that a step costs in proportion to the conversation, not to --context: 0 for the model's definition (models.yaml's paged), else 8192 where the model allows it (LFM3 and its VL, Audio and Omni checkpoints, Gemma 4, Qwen3.5; Qwen3 on coreai, elsewhere only when given one, --paged 1024 for its long context), -1 never (then --tier applies)."`
	Shortest  int     `placeholder:"TOKENS" help:"With --paged, also attention programs over chunks of this many tokens and twice as many... below --paged, and hold the cache in the shortest that covers the conversation (a call over a chunk of 1024 costs a third of one over 8192): 0 for the backend's (1024 on coreai, 4096 on coreml, whose Neural Engine leaves the attention of shorter chunks to the CPU, none on intel), -1 none."`
	Sparse    bool    `default:"true" negatable:"" help:"Past about 48K tokens of a model split at its attention (--paged), read 16384 keys of the cache exactly (32768 while reading a prompt), those of the 8-token buckets that weigh most, and every other bucket as its mean (two-level attention, a few times faster at 128K and over); --no-sparse reads the whole cache on every call."`
	Package   string  `placeholder:"FILES" help:"Chat programs converted by hwnpu-dev convert instead, comma separated: .mlpackage, .xml (Intel) or out.parts.json, with the same -cache and different -seq."`
	Config    string  `type:"path" placeholder:"FILE" help:"config.json of the chat model (default: in its directory)."`
	Weights   string  `type:"path" placeholder:"FILE" help:"Checkpoint the chat model's token embeddings are read from: .safetensors or .gguf (default: in its directory)."`
	Tokenizer string  `type:"path" placeholder:"FILE" help:"tokenizer.json of the chat model (default: next to config.json)."`
	Format    string  `help:"Weight format of the chat model's programs: int8, int4, int4-g128 or fp16 on Intel's NPU, lut4, lut6, lut8, int4, int8 or fp16 on the Neural Engine (default: the model's definition (models.yaml's format), else its architecture's for the NPU, int8 for LFM3 on Intel's NPU)."`
	Codes     string  `type:"path" placeholder:"DIR" help:"Directory of lookup tables chosen against the activations (tools/amd/q4_gptq.py --fmt-q4 lut6 ...) for the chat model's lutN weights; the programs are converted anew for them."`
	FormatFor string  `placeholder:"SUBSTRING=FORMAT,..." help:"Another weight format for the chat model's tensors whose names contain SUBSTRING, e.g. per_layer=lut8 (the last match wins); the programs are converted anew for them."`
	Tier      *string `placeholder:"TOKENS,..." help:"Caches shorter than --context to run on while a conversation is short, comma separated (a step costs in proportion to the cache); empty or none for none, auto for 4096 with a model in 4 bit tables and none with 6 or 8 bits (four such programs overflow the Neural Engine's compile cache and every start compiles them again). On coreai, a model split at its attention (--paged) runs a whole program with its cache inside first: auto for the longest of 512 to 4096 whose keys and values take at most 4 MiB a layer (2048 for LFM3-1.7B, 1024 for Qwen3-0.6B), when the Neural Engine holds its weights twice (default: the model's definition for this machine's memory (models.yaml's tier: none for LFM3-1.7B, Qwen3-0.6B and Qwen3.5-0.8B on 8 GB), else auto)."`
	Draft     int     `default:"8" help:"Tokens per call of the program that checks drafts (speculative decoding: tokens guessed from the prompt and the answer so far are checked in one call); 0 or 1 turns it off."`
	MTP       string  `name:"mtp" default:"${mtp}" help:"auto: drafts also from the checkpoint's multi-token prediction layer when it has one (Qwen3.5); off: from the prompt alone."`
	Prefill   string  `help:"A program of 64 tokens per call (128 for a model split at its attention) to read prompts beside the one that generates: auto (unless the two would not stay in the Neural Engine's compile cache of about 4 GiB, as Gemma 4 E4B's do not: then the one that generates reads the prompts, slower for long ones), on or off (default: the model's definition for this machine's memory, else auto)."`
	Memory    int     `placeholder:"GB" help:"Run the model as on a machine of this much memory, as MODEL.md's columns: --memory 8 on a 16 GB Mac for the smaller programs models.yaml gives 8 GB (LFM3-1.7B and Qwen3.5-0.8B without the whole program of short conversations, Qwen3-1.7B without the one that reads prompts), to leave room for other programs. Kept for the model once it is ready, and every hwnpu command runs it so (hwnpu memory shows and changes it; a tag of memory, lfm3:8g, does the same) (default: the memory chosen for the model, else HWNPU_MEMORY, else the machine's)."`
}

// given is what the flags give of how a model runs (chat.Run), the zero for those not given.
func (f *convertFlags) given() chat.Run {
	r := chat.Run{Prefill: f.Prefill, Format: f.Format, Context: f.Context, Paged: f.Paged}
	if f.Tier != nil {
		r.Tier = cmp.Or(strings.TrimSpace(*f.Tier), "none") // an empty --tier is none
	}
	return r
}

// memoryGiven is the memory name and the flags say to run the model as: a tag of memory
// (lfm3:8g), else --memory, 0 for none.
func memoryGiven(name string, cf *convertFlags) int {
	if _, tag, tagged := strings.Cut(name, ":"); tagged {
		if gb, ok := chat.MemoryTag(tag); ok {
			return gb
		}
	}
	return max(cf.Memory, 0)
}

// keepMemory keeps gb, the memory given (memoryGiven), as the one to run model m as
// (chat.ChooseMemory), now that it is ready on it. Not keeping it fails no run.
func keepMemory(m chat.Model, gb int, log io.Writer) {
	if gb == 0 || chat.ChosenMemory(m) == gb {
		return
	}
	if err := chat.ChooseMemory(m, gb); err != nil {
		fmt.Fprintf(log, "%s: %d GB not kept for the next runs: %v\n", m.Name, gb, err)
		return
	}
	fmt.Fprintf(log, "%s: runs as on a machine of %d GB from now on, in every hwnpu command (%s; hwnpu memory %s --forget forgets it)\n", m.Name, gb, chat.RunFile(), m.Name)
}

// samplingFlags are how a chat model answers.
type samplingFlags struct {
	Think       *bool   `negatable:"" help:"Let the model think (<think>) before it answers (default: on, unless the model's definition turns it off)."`
	ThinkBudget int     `default:"1024" placeholder:"TOKENS" help:"Most tokens the model thinks before it is made to answer (0: no limit); a small model may think for thousands."`
	Temp        float64 `default:"0.1" help:"Sampling temperature; 0 picks the most likely token."`
	TopK        int     `default:"50" help:"Sample from the k most likely tokens."`
	TopP        float64 `help:"Sample from the most likely tokens up to this probability (0: off)."`
	Rep         float64 `default:"1.05" help:"Repetition penalty on the tokens of the answer (1: none)."`
}

func (f samplingFlags) sampling() chat.Sampling {
	return chat.Sampling{Temperature: f.Temp, TopK: f.TopK, TopP: f.TopP, Repetition: f.Rep}
}

// think is whether the model thinks: --think, else not when its definition says so.
func (f samplingFlags) think(e *chat.Engine) bool {
	if f.Think != nil {
		return *f.Think
	}
	return !e.NoThink
}

// encoderFlags are the encoders of a chat model that reads images or audio, converted by
// hwnpu-dev convert (with --package).
type encoderFlags struct {
	Vision       string `placeholder:"FILES" help:"The LFM3-VL vision encoder converted by hwnpu-dev convert -arch lfm3-vision, a package per image size, comma separated (default: the 512x512 one a model with vision converts)."`
	AudioEncoder string `placeholder:"FILES" help:"The LFM3-Audio audio encoder converted by hwnpu-dev convert -arch lfm3-audio-encoder, a package per clip length, comma separated (default: the 30 second one a model with audio converts)."`
}

// logw is where what goes on is written: standard error, in gray on a terminal, with the
// placements in green or red.
func logw() io.Writer { return console.Log(os.Stderr) }

// expandHome is path with a leading ~ made the home directory.
func expandHome(path string) string {
	if path == "~" || strings.HasPrefix(path, "~/") || strings.HasPrefix(path, "~"+string(filepath.Separator)) {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, path[1:])
	}
	return path
}

// isPath reports whether --model names a directory (an export) rather than a model of the
// definitions or a repository: a leading /, . or ~, or more than one / (OWNER/NAME is a
// repository).
func isPath(name string) bool {
	return strings.HasPrefix(name, "/") || strings.HasPrefix(name, ".") || strings.HasPrefix(name, "~") ||
		strings.Count(name, "/") > 1 || strings.Contains(name, `\`)
}

// isIRON reports whether dir is an export of tools/amd/iron_export_lm.py.
func isIRON(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "manifest.json"))
	return err == nil
}

// loaded is a chat model loaded, with its definition (empty for an export's directory).
type loaded struct {
	*chat.Engine
	model chat.Model
}

// loadChat makes ready and loads the chat model name names: an export's directory, the export
// of AMD's NPU, the programs of --package, or the model of the definitions, downloaded and
// converted when missing (chat.RepoDir, chat.ModelDir).
func loadChat(ctx context.Context, name string, lf *loadFlags, cf *convertFlags, ef *encoderFlags) (loaded, error) {
	iron := ""
	if isPath(name) {
		iron = expandHome(name)
		if !isIRON(iron) {
			return loaded{}, fmt.Errorf("%s is not an export of tools/amd/iron_export_lm.py (no manifest.json)", name)
		}
	} else if cf.Package == "" {
		var err error
		if iron, err = chat.AMDExport(ctx, name, lf.Backend, logw()); err != nil {
			return loaded{}, err
		}
	}
	promptDir := chat.PromptStore(lf.PromptCache, lf.Cache)
	if iron != "" {
		c := chat.Config{IRON: iron, Tokenizer: cf.Tokenizer, Log: logw(), PromptDir: promptDir, Idle: lf.Idle}
		// a model of the definitions: its experts' places get its core and this application's
		// counts first (an export's directory has neither)
		if m, ok := chat.LookupFor(name); ok && !isPath(name) {
			c.Experts = expertsPlacement(m, lf)
		}
		e, err := chat.Load(ctx, c)
		return loaded{Engine: e}, err
	}
	gb := memoryGiven(name, cf)
	m, err := chat.ResolveOn(ctx, name, gb, logw())
	if err != nil {
		return loaded{}, err
	}
	if m.Kind != chat.KindChat {
		return loaded{}, fmt.Errorf("%s is a %s model, not a chat model", m.Name, m.Kind)
	}
	md, src := chat.ModelDir(m), chat.RepoDir(m.Source())
	c := chat.Config{ModelConfig: cf.Config, Weights: cf.Weights}
	if cf.Package != "" {
		c.Packages, c.Name, c.Backend = strings.Split(cf.Package, ","), m.Name, lf.Backend
		c.ModelConfig = cmp.Or(c.ModelConfig, filepath.Join(src, "config.json"))
		c.Weights = cmp.Or(c.Weights, chat.WeightsIn(m))
	} else {
		// the flags, else the model's definition for the memory it runs as (Model.On)
		run := m.Run.Over(cf.given())
		tiers, err := chat.Tiers(cmp.Or(run.Tier, chat.DefaultTiers))
		if err != nil {
			return loaded{}, fmt.Errorf("tier %q: %w", run.Tier, err)
		}
		if c, err = chat.Prepare(ctx, chat.Setup{Dir: md, Repo: m.Source(), Arch: m.Arch, Name: m.Name,
			Context: run.Context, Paged: run.Paged, Shortest: cf.Shortest, Prefill: run.Prefill,
			Format: run.Format, Draft: cf.Draft, MTP: cf.MTP, Tiers: tiers, Backend: lf.Backend, Codes: cf.Codes, CodesHash: run.Codes,
			FormatFor: split(cf.FormatFor), Log: logw()}); err != nil {
			return loaded{}, err
		}
		keepMemory(m, gb, logw())
	}
	c.Tokenizer, c.CacheDir, c.Log = cf.Tokenizer, lf.Cache, logw()
	c.PromptDir = promptDir
	c.Dense = !cf.Sparse
	c.Idle = lf.Idle
	// a model without experts on the host ignores it
	c.Experts = expertsPlacement(m, lf)
	// the flags override the encoders Prepare converted (a model with vision or audio); without
	// them those stay
	if ef != nil && ef.Vision != "" {
		c.Vision = split(ef.Vision)
	}
	if ef != nil && ef.AudioEncoder != "" {
		c.Audio = split(ef.AudioEncoder)
	}
	e, err := chat.Load(ctx, c)
	if err != nil {
		return loaded{}, err
	}
	console.Prepared(logw())
	e.NoThink = m.NoThink
	return loaded{Engine: e, model: m}, nil
}

// embedFlags are how an embedding model is converted.
type embedFlags struct {
	EmbedSeq     string `default:"${embed_seqs}" placeholder:"N,..." help:"Sequence lengths an embedding model is converted for, comma separated; the longest is the most tokens of a text."`
	EmbedPackage string `placeholder:"FILES" help:"Embedding programs converted by hwnpu-dev convert instead, comma separated, with different -seq (config.json and the checkpoint in the model's directory)."`
}

// loadEmbedder makes ready and loads the embedding model m.
func loadEmbedder(ctx context.Context, m chat.Model, lf *loadFlags, ef *embedFlags) (*chat.Embedder, error) {
	seqs, err := parseInts(ef.EmbedSeq)
	if err != nil {
		return nil, fmt.Errorf("--embed-seq: %w", err)
	}
	md, src := chat.ModelDir(m), chat.RepoDir(m.Source())
	var c chat.EmbedConfig
	if ef.EmbedPackage != "" {
		c = chat.EmbedConfig{Packages: strings.Split(ef.EmbedPackage, ","), ModelConfig: filepath.Join(src, "config.json"), Weights: chat.WeightsIn(m), Name: m.Name, Backend: lf.Backend}
	} else if c, err = chat.PrepareEmbedder(ctx, chat.EmbedSetup{Dir: md, Repo: m.Source(), Name: m.Name, Seqs: seqs, Backend: lf.Backend, Log: logw()}); err != nil {
		return nil, err
	}
	c.CacheDir, c.Log = lf.Cache, logw()
	return chat.LoadEmbedder(ctx, c)
}

// split splits a comma separated list; nil for "".
func split(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(s, ",")
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

// expertsPlacement is how a mixture of experts' experts that do not all fit are placed: model
// m's core (models.yaml's experts_core) first, then by this application's counts (--experts-key).
func expertsPlacement(m chat.Model, lf *loadFlags) experts.Placement {
	idle := lf.Idle
	if idle <= 0 {
		idle = -1 // never
	}
	p := experts.Placement{Core: m.ExpertsCore, Idle: idle, Ceiling: filepath.Join(chat.ModelDir(m), "experts-usage", "ceiling")}
	if lf.ExpertsKey != "off" {
		p.Usage = chat.ExpertsUsage(chat.ModelDir(m), lf.ExpertsKey)
	}
	return p
}
