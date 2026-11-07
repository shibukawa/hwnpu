package chat

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"github.com/shibukawa/hwnpu/internal/console"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
)

var hubURL = "https://huggingface.co/"

// DefaultContext is the longest cache the Neural Engine takes with 64 tokens per call in one
// piece: a tensor dimension goes up to llm.NeuralEngineMaxDim, and the key axis is the cache plus
// the tokens. It is the context of a model that cannot be split at its attention; one that can
// (Setup.Paged) gets the longest it was trained for.
const DefaultContext = llm.NeuralEngineMaxDim - prefillSeq

// PagedOff in Setup.Paged converts the model whole even when it can be split at its attention.
const PagedOff = -1

// prefillSeq is the tokens per call of the program that reads prompts, of a model in one piece:
// its key axis is the cache plus these (DefaultContext).
const prefillSeq = 64

// pagedPrefillSeq is that of a model split at its attention on the Neural Engine, which has no
// such bound. A call costs little more for more tokens up to about 128 (the weights are read once
// per call): on an M3, LFM3-1.7B reads 4K, 16K and 64K tokens in 3.9, 16.0 and 136 s with 128
// rather than 4.7, 25.2 and 160 s with 64, and 256 is no faster (design.md 16.82).
const pagedPrefillSeq = 128

// The values of Setup.Prefill (-prefill): whether a program of 64 tokens per call (128 for a
// model split at its attention) reads the prompts beside the one that generates.
const (
	// PrefillAuto has one unless the two programs' weights would not stay in the Neural Engine's
	// compile cache (compileCacheBytes), as Gemma 4 E4B's (2 × 3.26 GiB in lut6) do not: then
	// every start compiles them again (77 to 87 seconds on an M3) and every question pays 1 to 3
	// seconds of switching between them, while the one that generates alone starts in 4 seconds
	// and answers faster. It reads a long prompt 4.4 times slower (16K tokens in 183 seconds
	// against 42; docs/design.md 16.65)
	PrefillAuto = "auto"
	PrefillOn   = "on"  // always a program to read prompts
	PrefillOff  = "off" // never: the program that generates reads them
)

// compileCacheBytes is about what the Neural Engine's own cache of compiled programs holds
// (TierAuto).
var compileCacheBytes int64 = 4 << 30

// DefaultTiers is the -tier of hwnpu-chat and hwnpu-agent: auto, a cache of 4096 to run on
// while a conversation is short when the model is in 4 bit tables, none otherwise (TierAuto).
const DefaultTiers = "auto"

// TierAuto in Setup.Tiers (Tiers("auto")) lets Prepare choose: a tier of 4096 for a model in
// 4 bit tables, none for 6 or 8 bits. The Neural Engine keeps its compiled programs in a cache
// of its own that holds about 4 GiB: four programs of LFM3-1.7B in lut4 (3.2 GiB) stay in it and
// load in a second each, four in lut6 (4.8 GiB) push each other out and every start compiles
// them anew, 17 seconds a part (docs/design.md 16.41); two in lut6 (2.4 GiB) stay.
const TierAuto = -1

// autoTiers is the choice TierAuto stands for, by the weight format and the backend: the cache
// that overflows is the Neural Engine's, so another NPU gets the tier whatever the format.
func autoTiers(format, backend string) []int {
	if backend != llm.BackendIntel && format != "lut4" {
		return nil
	}
	return []int{4096}
}

// Tiers parses a comma separated list of caches (Setup.Tiers); empty or "none" for none (an
// empty argument is hard to pass in some shells, PowerShell's).
func Tiers(s string) ([]int, error) {
	switch strings.TrimSpace(s) {
	case "auto":
		return []int{TierAuto}, nil
	case "none":
		return nil, nil
	}
	var out []int
	for _, f := range strings.Split(s, ",") {
		if f = strings.TrimSpace(f); f == "" {
			continue
		}
		n, err := strconv.Atoi(f)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("%q is not a positive number", f)
		}
		out = append(out, n)
	}
	return out, nil
}

// Setup says where the model is and how to get it.
type Setup struct {
	Dir  string // holds config.json, tokenizer.json, model.safetensors and the converted programs
	Repo string // Hugging Face repository the missing checkpoint files come from
	// Context is the longest prompt and answer together; 0 for the longest the model was trained
	// for when it is split at its attention (Paged), else DefaultContext.
	Context int
	// Paged is the chunk length of a model split at its attention (llm.Options.Paged): a step
	// then costs in proportion to the tokens in the cache, and the cache's memory grows with the
	// conversation, so the context can be the model's longest. 0 for paged.DefaultChunk when the
	// model can be split, PagedOff for never. A model split at its attention has no tiers.
	Paged int
	// Draft makes the program that generates take this many tokens per call instead of 1, for
	// checking the drafts of speculative decoding (Engine.Draft); 0 or 1 for none. It is one
	// program either way: the Neural Engine takes about 100 ms to switch between programs, so
	// generation runs on one.
	Draft int
	// Prefill is PrefillAuto (also for ""), PrefillOn or PrefillOff: whether a program of 64
	// tokens per call (128 split at its attention) reads the prompts, or the one that generates does (it must take several
	// tokens per call, Draft).
	Prefill string
	// Tiers are caches shorter than Context to run on while a conversation is short (Engine
	// tiers): the programs are converted for each, and a conversation moves up as it grows. A
	// step costs in proportion to the cache, whatever it holds.
	Tiers []int
	// Backend picks the device the programs are converted for and run on (llm.FindDevice): a
	// device backend (coreml, levelzero, openvino) or a program format (coreml, intel); empty for
	// HWNPU_BACKEND or the NPU found.
	Backend string
	// Format is the weight format of the chat model's programs (llm.Options.Format); empty for the
	// architecture's for the backend (llm.Arch.DefaultFormatFor). The encoders stay in fp16.
	Format string
	Log    io.Writer
}

// TokenizerFiles are the files of an architecture's tokenizer in its checkpoint, the one
// tokenizer.Load reads first: PLaMo 3's tokenizer.jsonl and the thresholds of its
// tokenizer_config.json, every other model's tokenizer.json.
func TokenizerFiles(arch string) []string {
	if arch == "plamo3" {
		return []string{"tokenizer.jsonl", "tokenizer_config.json"}
	}
	return []string{"tokenizer.json"}
}

// programDir is where the programs for a backend go in a model's directory: Core ML's next to the
// checkpoint (as before), Intel's in intel/, so that one directory serves both.
func programDir(dir, backend string) (string, error) {
	if backend != llm.BackendIntel {
		return dir, nil
	}
	d := filepath.Join(dir, "intel")
	return d, os.MkdirAll(d, 0o755)
}

// programExt is the file name extension of a program for a backend.
func programExt(backend string) string {
	if backend == llm.BackendIntel {
		return ".xml"
	}
	return ".mlpackage"
}

// targetName is what logs call the NPU the programs are for.
func targetName(backend string) string {
	if backend == llm.BackendIntel {
		return "Intel's NPU"
	}
	return "the Neural Engine"
}

// Prepare makes the model ready in Dir: it downloads the checkpoint files that are missing from
// Hugging Face, and converts the programs that are missing, one of 1 token per call to generate
// and one of 64 to read prompts (128 for a model split at its attention on the Neural Engine,
// with the logits after the last token alone where the architecture can: -last), both with a
// cache of Context tokens, for the NPU found (Setup.Backend), named by the architecture, the
// weight format and the attention (lfm3-lut6-grouped-p8192-c262144-q8.mlpackage). It returns the Config that loads them.
func Prepare(ctx context.Context, s Setup) (Config, error) {
	switch s.Prefill {
	case "", PrefillAuto, PrefillOn, PrefillOff:
	default:
		return Config{}, fmt.Errorf("chat: -prefill %q is not auto, on or off", s.Prefill)
	}
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return Config{}, err
	}
	backend, err := llm.ProgramFormat(ctx, s.Backend)
	if err != nil {
		return Config{}, err
	}
	pdir, err := programDir(s.Dir, backend)
	if err != nil {
		return Config{}, err
	}
	if err := fetch(ctx, s.Repo, "config.json", filepath.Join(s.Dir, "config.json"), s.Log); err != nil {
		return Config{}, err
	}
	c := Config{ModelConfig: filepath.Join(s.Dir, "config.json"), Weights: filepath.Join(s.Dir, "model.safetensors"), Name: filepath.Base(s.Repo), Backend: cmp.Or(s.Backend, backend)}
	a, model, err := llm.LoadConfig(c.ModelConfig, "")
	if err != nil {
		return Config{}, err
	}
	for _, f := range TokenizerFiles(a.Name) {
		if err := fetch(ctx, s.Repo, f, filepath.Join(s.Dir, f), s.Log); err != nil {
			return Config{}, err
		}
	}
	if err := FetchWeights(ctx, s.Repo, s.Dir, s.Log); err != nil {
		return Config{}, err
	}
	// The name says what the program is: the format always, since a change of the defaults
	// (LFM3 to lut6 and the grouped attention, 2026-11) must convert anew instead of loading
	// what an earlier version wrote, and the attention where the architecture has the setting.
	format := cmp.Or(s.Format, a.DefaultFormatFor(model, backend))
	kind := a.Name + a.Rev() + "-" + format
	if attn := a.Param(llm.Options{}, "attn"); attn != "" {
		kind += "-" + attn
	}
	chunk, context := pagedChunk(model, s.Paged), s.Context
	if context <= 0 {
		context = DefaultContext
		if pm, ok := model.(llm.Paged); ok && chunk > 0 && pm.MaxContext() > 0 {
			context = pm.MaxContext()
		}
	}
	if chunk > 0 {
		chunk = min(chunk, context)
		kind += fmt.Sprintf("-p%d", chunk)
	}
	// the Neural Engine's program of one token per call is in two parts; Intel's is whole, since
	// each part is a graph its driver compiles, slowly for a large one, and no limit is known to
	// call for parts
	// the program that reads prompts computes the logits after the last token alone where the
	// architecture can (llm.Options.LastRow): the output layer is the model's widest
	type prog struct {
		seq, parts int
		lastRow    bool
	}
	pre := prog{prefillSeq, 0, a.LastRow && backend != llm.BackendIntel}
	if chunk > 0 && backend != llm.BackendIntel {
		pre.seq = min(pagedPrefillSeq, chunk) // a call's tokens fit in a chunk
	}
	progs := []prog{{1, 2, false}, pre}
	if backend == llm.BackendIntel {
		progs[0].parts = 0
	}
	if s.Draft > 1 {
		progs[0] = prog{s.Draft, 0, false}
	}
	tiers := s.Tiers
	if chunk > 0 {
		// the cost follows the tokens used, not the cache: no shorter caches to run on
		if len(tiers) > 0 && !slices.Contains(tiers, TierAuto) && s.Log != nil {
			fmt.Fprintf(s.Log, "no cache tiers for %s: split at its attention, a step costs in proportion to the conversation, not to the context of %d\n", c.Name, context)
		}
		tiers = nil
	}
	if slices.Contains(tiers, TierAuto) {
		tiers = autoTiers(format, backend)
		if len(tiers) == 0 && s.Log != nil {
			fmt.Fprintf(s.Log, "no shorter cache tier for %s: its programs are %s, and four of them would not stay in the Neural Engine's compile cache (-tier 4096 to have one anyway)\n", c.Name, format)
		}
	}
	var caches []int
	for _, t := range tiers {
		if t > 0 && t < context {
			caches = append(caches, t)
		}
	}
	slices.Sort(caches)
	caches = slices.Compact(caches)
	if one, why := onePrograms(model, format, backend, s.Prefill, s.Draft, len(progs)*(len(caches)+1)); one {
		progs = progs[:1]
		if s.Log != nil {
			fmt.Fprintf(s.Log, "one program for %s, which reads the prompts %d tokens per call too: %s (-prefill on for one of %d to read long prompts faster)\n", c.Name, s.Draft, why, pre.seq)
		}
	}
	for _, cache := range append(caches, context) {
		var pkgs []string
		for _, p := range progs {
			last := ""
			if p.lastRow {
				last = "-last"
			}
			out := filepath.Join(pdir, fmt.Sprintf("%s-c%d-q%d%s%s", kind, cache, p.seq, last, programExt(backend)))
			path, err := convert(ctx, c.ModelConfig, c.Weights, out, program{backend: backend, format: s.Format, seq: p.seq, parts: p.parts, cache: cache, paged: chunk, lastRow: p.lastRow}, s.Log)
			if err != nil {
				return Config{}, err
			}
			if chunk > 0 {
				// a conversion from before the two-level attention gets its mass program, which
				// has no weights, instead of being converted again
				if err := llm.AddMass(path, cmp.Or(a.Param(llm.Options{}, "attn"), "grouped"), model.(llm.Paged).PagedShape().Scale); err != nil {
					return Config{}, err
				}
			}
			pkgs = append(pkgs, path)
		}
		c.Tiers = append(c.Tiers, pkgs)
	}
	c.Packages = c.Tiers[len(c.Tiers)-1]
	shareOnce(pdir, s.Log)
	for _, e := range encoders {
		ea, _, err := llm.LoadConfig(c.ModelConfig, e.arch)
		if err != nil {
			continue // the checkpoint has no such encoder
		}
		for _, f := range e.files {
			if err := fetch(ctx, s.Repo, f, filepath.Join(s.Dir, f), s.Log); err != nil {
				return Config{}, err
			}
		}
		out := filepath.Join(pdir, fmt.Sprintf("%s%s-%s%s", e.arch, ea.Rev(), e.value, programExt(backend)))
		path, err := convert(ctx, c.ModelConfig, c.Weights, out, program{backend: backend, arch: e.arch, format: "fp16", params: map[string]string{e.param: e.value}}, s.Log)
		if err != nil {
			return Config{}, err
		}
		*e.packages(&c) = []string{path}
	}
	// A conversion reads the checkpoint into memory (2.4 GiB of PLaMo 3 Translate 2B left after
	// it): give it back now rather than as Go's scavenger does, a little at a time.
	debug.FreeOSMemory()
	return c, nil
}

// onePrograms decides Setup.Prefill: whether the program that generates (of draft tokens per
// call) reads the prompts too, instead of n programs of the model's weights in format, and why.
func onePrograms(model llm.Model, format, backend, prefill string, draft, n int) (bool, string) {
	switch {
	case prefill == PrefillOff:
		return true, "-prefill off"
	case prefill == PrefillOn || draft <= 1 || backend == llm.BackendIntel:
		return false, "" // reading prompts one token per call would be far slower; the cache is the Neural Engine's
	}
	var o llm.Options
	set := llm.BindFlags(flag.NewFlagSet("", flag.ContinueOnError), &o, "")
	set()
	o.Format = format
	pre, layers, post := llm.EstimateBytes(model, o)
	total := pre + post
	for _, l := range layers {
		total += l
	}
	if int64(n)*total <= compileCacheBytes {
		return false, ""
	}
	return true, fmt.Sprintf("%d programs would hold %.2f GiB of weights, over the about %d GiB the Neural Engine keeps compiled, so every start would compile them again and every question switch between them", n, float64(int64(n)*total)/(1<<30), compileCacheBytes>>30)
}

// sharedMark is the file in a directory of programs that says ShareWeights went over it: the
// conversions share the weights of what they write since, and the programs written before were
// shared then.
const sharedMark = ".weights-shared"

// shareOnce makes the programs in dir that have the same weights share them on the disk
// (llm.ShareWeights), once per directory: the programs of 8 and 64 tokens per call written before
// conversions did it themselves.
func shareOnce(dir string, log io.Writer) {
	mark := filepath.Join(dir, sharedMark)
	if _, err := os.Stat(mark); err == nil {
		return
	}
	llm.ShareWeights(dir)
	os.WriteFile(mark, nil, 0o644)
}

// cacheAge is how long an entry of the compile cache stays unused before Load removes it.
const cacheAge = 30 * 24 * time.Hour

// pruneCache removes the entries of the compile cache that are not used any more (cache.Prune):
// those of programs deleted or not loaded for cacheAge.
func pruneCache(dir string, logf func(string, ...any)) {
	if dir == "" {
		return
	}
	if n, bytes, err := cache.Prune(dir, cacheAge); err == nil && n > 0 {
		logf("compile cache: removed %d entries not used any more (%.2f GiB)", n, float64(bytes)/(1<<30))
	}
}

// pagedChunk is the chunk length to split the model at its attention with (Setup.Paged), 0 for
// none: the model must be Paged (LFM3 and its VL, Audio and Omni checkpoints, Gemma 4), on any
// NPU.
func pagedChunk(model llm.Model, want int) int {
	pm, ok := model.(llm.Paged)
	if !ok || want < 0 || len(pm.PagedLayers()) == 0 {
		return 0
	}
	return cmp.Or(want, paged.DefaultChunk)
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

// program is what convert writes: the program of arch (empty: the one config.json names) for
// backend (llm.Options.Backend) with the weights in format (empty: the architecture's default for
// the backend) and the architecture's own settings in params; seq tokens per call, in parts (0:
// one) and with a KV cache of cache tokens (0: none).
type program struct {
	backend           string
	arch, format      string
	params            map[string]string
	seq, parts, cache int
	paged             int  // llm.Options.Paged
	lastRow           bool // llm.Options.LastRow
}

// convert writes the program of the model (config.json and the checkpoint) unless it is there,
// and returns its path. NAME.done, written last, holds the path, so that a conversion cut off is
// done again.
func convert(ctx context.Context, config, weights, out string, p program, log io.Writer) (string, error) {
	done := strings.TrimSuffix(out, filepath.Ext(out)) + ".done"
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
	// LFM3's answers as they were on the Neural Engine (also used for the embedding model); the
	// integer formats keep the converter's scale search
	fs := flag.NewFlagSet("", flag.ContinueOnError)
	var opts llm.Options
	set := llm.BindFlags(fs, &opts, "")
	if err := fs.Parse(nil); err != nil {
		return "", err
	}
	set()
	opts.Backend = p.backend
	opts.Format = cmp.Or(p.format, a.DefaultFormatFor(model, p.backend))
	if p.backend != llm.BackendIntel && !strings.HasPrefix(opts.Format, "int") {
		opts.Quant = "kmeans-minmax"
	}
	opts.Seq, opts.Cache, opts.Paged, opts.LastRow = p.seq, p.cache, p.paged, p.lastRow
	for k, v := range p.params {
		opts.Params[k] = v
	}
	npu := targetName(p.backend)
	switch {
	case p.paged > 0:
		fmt.Fprintf(log, "converting %s for %s (split at its attention, a context of %d held by the host in chunks of %d, %d per call, %s); this takes a few minutes\n", filepath.Base(out), npu, p.cache, p.paged, p.seq, opts.Format)
	case p.cache > 0:
		fmt.Fprintf(log, "converting %s for %s (cache %d, %d per call, %s); this takes a minute or two\n", filepath.Base(out), npu, p.cache, p.seq, opts.Format)
	case p.seq > 0:
		fmt.Fprintf(log, "converting %s for %s (%d tokens, %s); this takes a minute or two\n", filepath.Base(out), npu, p.seq, opts.Format)
	default:
		fmt.Fprintf(log, "converting %s for %s (%s); this takes a minute or two\n", filepath.Base(out), npu, opts.Format)
	}
	path, err := llm.Convert(a, model, w, out, "any", p.parts, opts, log)
	if err != nil {
		return "", err
	}
	return path, os.WriteFile(done, []byte(path), 0o644)
}

// errNotFound is the error of fetch for a file the repository does not have.
var errNotFound = errors.New("not in the repository")

// FetchWeights downloads the checkpoint into dir: model.safetensors, or when the repository has
// it in several files (Qwen3.5-4B), its index model.safetensors.index.json and the files it
// names. llm.OpenWeights opens either by the name model.safetensors.
func FetchWeights(ctx context.Context, repo, dir string, log io.Writer) error {
	single := filepath.Join(dir, "model.safetensors")
	index := single + ".index.json"
	if _, err := os.Stat(index); err != nil {
		err := fetch(ctx, repo, "model.safetensors", single, log)
		if err == nil || !errors.Is(err, errNotFound) {
			return err
		}
		if err := fetch(ctx, repo, "model.safetensors.index.json", index, log); err != nil {
			return err
		}
	}
	files, err := shards(index)
	if err != nil {
		return err
	}
	for _, f := range files {
		if err := fetch(ctx, repo, f, filepath.Join(dir, f), log); err != nil {
			return err
		}
	}
	return nil
}

// shards are the files an index of a checkpoint in several files names, in order.
func shards(index string) ([]string, error) {
	raw, err := os.ReadFile(index)
	if err != nil {
		return nil, err
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, fmt.Errorf("chat: %s: %w", index, err)
	}
	var files []string
	for _, f := range idx.WeightMap {
		if f != filepath.Base(f) {
			return nil, fmt.Errorf("chat: %s names %q, not a file next to it", index, f)
		}
		if !slices.Contains(files, f) {
			files = append(files, f)
		}
	}
	slices.Sort(files)
	return files, nil
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
	case http.StatusNotFound:
		return fmt.Errorf("chat: downloading %s: %s: %w", url, resp.Status, errNotFound)
	default:
		return fmt.Errorf("chat: downloading %s: %s", url, resp.Status)
	}
	total := int64(-1)
	if resp.ContentLength >= 0 {
		total = have + resp.ContentLength
	}
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	pw := &progress{w: f, done: have, from: have, total: total,
		bar: console.NewBar(log, fmt.Sprintf("downloading %s from %s", filepath.Base(name), repo))}
	_, err = io.Copy(pw, resp.Body)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		pw.bar.Done(fmt.Sprintf("stopped at %s: %v (run again to go on)", gb(pw.done, pw.total), err))
		return fmt.Errorf("chat: downloading %s: %w (run again to go on)", name, err)
	}
	pw.bar.Done(fmt.Sprintf("%s in %s to %s", gb(pw.done, -1), console.Duration(pw.bar.Elapsed()), filepath.Dir(path)))
	return os.Rename(part, path)
}

// progress writes through and shows how far a download is on its bar: the bytes, the speed and
// the time left.
type progress struct {
	w                 io.Writer
	done, from, total int64 // total < 0: not known
	bar               *console.Bar
	last              time.Time
}

func (p *progress) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.done += int64(n)
	if time.Since(p.last) >= 200*time.Millisecond {
		p.last = time.Now()
		frac := -1.0
		detail := gb(p.done, p.total)
		if sec := p.bar.Elapsed().Seconds(); sec > 0.5 {
			rate := float64(p.done-p.from) / sec
			detail += fmt.Sprintf("  %.1f MB/s", rate/1e6)
			if p.total > 0 && rate > 0 {
				detail += ", " + console.Duration(time.Duration(float64(p.total-p.done)/rate*float64(time.Second))) + " left"
			}
		}
		if p.total > 0 {
			frac = float64(p.done) / float64(p.total)
		}
		p.bar.Set(frac, detail)
	}
	return n, err
}

// gb is bytes (of total, when known) for a person: "1.23 / 4.18 GB", "12.30 MB", "2.11 kB".
func gb(n, total int64) string {
	unit, div := "GB", 1e9
	switch m := max(n, total); {
	case m < 1e6:
		unit, div = "kB", 1e3
	case m < 1e9:
		unit, div = "MB", 1e6
	}
	if total < 0 {
		return fmt.Sprintf("%.2f %s", float64(n)/div, unit)
	}
	return fmt.Sprintf("%.2f / %.2f %s", float64(n)/div, float64(total)/div, unit)
}

// Fetch downloads a file of a Hugging Face repository to path unless it is there, as Prepare does
// for the checkpoint.
func Fetch(ctx context.Context, repo, name, path string, log io.Writer) error {
	return fetch(ctx, repo, name, path, log)
}
