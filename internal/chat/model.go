package chat

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
	"github.com/shibukawa/hwnpu/internal/palette"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
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

// The values of Setup.MTP (-mtp).
const (
	MTPAuto = "auto" // drafts from the multi-token prediction layer when the checkpoint has one
	MTPOff  = "off"  // drafts from the prompt alone
)

// mtpSeq is the tokens per call of the multi-token prediction layer's program: it reads the
// tokens the model kept in a call (up to Setup.Draft) and guesses from the last, then one at a
// time for the drafts after the first; the prompt in calls of as many.
const mtpSeq = 8

// mtpArch is the architecture of a model's multi-token prediction layer, "" for none: that of
// the arch with -mtp, when the checkpoint has its tensors. A program of the layer converted
// before in pdir says it has them; else the checkpoint is downloaded to look (tables.go).
func mtpArch(config string, a *llm.Arch, pdir string, checkpoint func() (string, error)) (string, error) {
	name := a.Name + "-mtp"
	ma, mm, err := llm.LoadConfig(config, name)
	if err != nil {
		return "", nil
	}
	if done, _ := filepath.Glob(filepath.Join(pdir, ma.Name+"*-last*.done")); len(done) > 0 {
		return name, nil
	}
	weights, err := checkpoint()
	if err != nil {
		return "", err
	}
	w, err := llm.OpenWeights(weights, ma)
	if err != nil {
		return "", nil
	}
	defer w.Close()
	for t := range mm.Tensors() {
		if strings.HasPrefix(t, "mtp.") && strings.HasSuffix(t, "norm.weight") {
			if _, _, err := w.Get(t); err != nil {
				return "", nil
			}
		}
	}
	return name, nil
}

// compileCacheBytes is about what the Neural Engine's own cache of compiled programs holds
// (TierAuto), with some room to spare: on an M3 the two programs of Qwen3.5-2B (3.56 GiB in
// lut8) were compiled again at every start (4 to 5 minutes), one alone starts in 1.5 seconds
// (docs/design.md 16.99). The cache is about 4 GiB.
var compileCacheBytes int64 = 7 << 29 // 3.5 GiB

// m1HoldBytes is what an M1's Neural Engine holds of the programs loaded together before every
// call slows down about 12 times, as if parts were loaded again on each: lut4 parts holding 2.86
// GiB ran at their own times added up, 2.97 GiB at 828 against 72 ms a token, whatever the number
// of parts (hwnpu-dev probe split, PROBE.md). The M2 (3.65 GiB at full speed), the M3 and the M4 have no
// such limit below the compile cache, so the programs are the same files on every chip and an M1
// only loads fewer of them: the one that generates reads the prompts too (onePrograms).
const m1HoldBytes = 2867 << 20 // 2.8 GiB

// neuralEngineHold is what the machine's Neural Engine holds of the programs loaded together
// before it slows down, 0 for no such limit: m1HoldBytes on an M1 (the M1 Pro, Max and Ultra,
// with the same Neural Engine, are taken to be alike; not measured).
var neuralEngineHold = func() int64 {
	if strings.HasPrefix(cpuBrand(), "Apple M1") {
		return m1HoldBytes
	}
	return 0
}

// DefaultTiers is the --tier of hwnpu-chat (chat, agent, serve): auto, a cache of 4096 to run on
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
	// Arch is the structure to convert the model as (llm.Lookup), "" for the one config.json
	// says (Model.Arch: a distilled model whose model_type is a name of its own).
	Arch string
	// Name is what the API calls the model (Model.Name), "" for the repository's name.
	Name string
	// Context is the longest prompt and answer together; 0 for the longest the model was trained
	// for when it is split at its attention (Paged), else DefaultContext (an llm.Unpaged model's
	// own, Qwen3's 4096).
	Context int
	// Paged is the chunk length of a model split at its attention (llm.Options.Paged): a step
	// then costs in proportion to the tokens in the cache, and the cache's memory grows with the
	// conversation, so the context can be the model's longest. 0 for paged.DefaultChunk when the
	// model can be split, PagedOff for never. A model split at its attention has no tiers.
	Paged int
	// Shortest is the shortest chunk of the attention programs besides Paged's (llm.AddSmaller):
	// the host holds the cache in the shortest that covers the conversation, 1024, 2048, 4096...
	// up to Paged, as a call costs a fixed part and a part in proportion to its chunk. 0 for the
	// backend's (shortestChunk), PagedOff for none.
	Shortest int
	// Draft makes the program that generates take this many tokens per call instead of 1, for
	// checking the drafts of speculative decoding (Engine.Draft); 0 or 1 for none. It is one
	// program either way: the Neural Engine takes about 100 ms to switch between programs, so
	// generation runs on one.
	Draft int
	// MTP is MTPAuto (also for "") or MTPOff: whether the drafts come from the checkpoint's
	// multi-token prediction layer (Qwen3.5's mtp., -arch qwen35-mtp) as well as from the prompt,
	// when it has one and Draft is above 1. The programs of the model then also output their
	// final hidden states (hidden-out), which the layer reads.
	MTP string
	// Prefill is PrefillAuto (also for ""), PrefillOn or PrefillOff: whether a program of 64
	// tokens per call (128 split at its attention) reads the prompts, or the one that generates does (it must take several
	// tokens per call, Draft).
	Prefill string
	// Tiers are caches shorter than Context to run on while a conversation is short (Engine
	// tiers): the programs are converted for each, and a conversation moves up as it grows. A
	// step costs in proportion to the cache, whatever it holds.
	Tiers []int
	// Backend picks the device the programs are converted for and run on (llm.FindDevice): a
	// device backend (coreml, coreai, levelzero, openvino) or a program format (coreml, intel);
	// empty for HWNPU_BACKEND or the NPU found. coreai converts Core AI's programs (.aimodel, in
	// an export NAME.coreai beside the checkpoint) from the checkpoint, the parts of the same
	// layers of the programs of one cache as the functions of one (llm.ConvertAI).
	Backend string
	// Format is the weight format of the chat model's programs (llm.Options.Format); empty for the
	// architecture's for the backend (llm.Arch.DefaultFormatFor). The encoders stay in fp16.
	Format string
	// Codes is a directory of lookup tables and indices chosen against the activations
	// (tools/amd/q4_gptq.py, llm.Options.Codes) for the lutN weights it names; empty for none.
	// The programs' names carry a hash of its index (-gptq-HASH), so that other codes or none
	// convert anew.
	Codes string
	// FormatFor gives some tensors another format than Format (llm.Options.FormatFor,
	// "SUBSTRING=FORMAT" entries, such as per_layer=lut8); the programs' names carry them.
	FormatFor []string
	Log       io.Writer
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

// programDir is where the programs for a backend go in a model's directory: Core ML's and Core
// AI's exports next to the checkpoint (as the image models' exports), Intel's in intel/, so that
// one directory serves all.
func programDir(dir, backend string) (string, error) {
	if backend != llm.BackendIntel {
		return dir, nil
	}
	d := filepath.Join(dir, "intel")
	return d, os.MkdirAll(d, 0o755)
}

// programExt is the file name extension of a program for a backend. Core AI's programs are an
// export, a directory NAME.coreai (as the image models'): a program's model.aimodel or its parts
// and manifest model.parts.json (convert), or the programs of one cache together (convertShared).
func programExt(backend string) string {
	switch backend {
	case llm.BackendIntel:
		return ".xml"
	case llm.BackendCoreAI:
		return ".coreai"
	}
	return ".mlpackage"
}

// doneFile is the file that says the program or export out is whole (cache.WriteDone): NAME.done
// of a program NAME.mlpackage or NAME.xml, NAME.coreai.done of an export NAME.coreai (as the
// image models' exports).
func doneFile(out string) string {
	if filepath.Ext(out) == ".coreai" {
		return out + ".done"
	}
	return strings.TrimSuffix(out, filepath.Ext(out)) + ".done"
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
	switch s.MTP {
	case "", MTPAuto, MTPOff:
	default:
		return Config{}, fmt.Errorf("chat: -mtp %q is not auto or off", s.MTP)
	}
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return Config{}, err
	}
	backend, err := llm.NativeFormat(ctx, s.Backend)
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
	c := Config{ModelConfig: filepath.Join(s.Dir, "config.json"), Weights: filepath.Join(s.Dir, "model.safetensors"), Name: cmp.Or(s.Name, repoName(s.Repo)), Backend: cmp.Or(s.Backend, llm.DeviceBackend(""), backend)}
	a, model, err := llm.LoadConfig(c.ModelConfig, s.Arch)
	if err != nil {
		return Config{}, err
	}
	for _, f := range TokenizerFiles(a.Name) {
		if err := fetch(ctx, s.Repo, f, filepath.Join(s.Dir, f), s.Log); err != nil {
			return Config{}, err
		}
	}
	// the checkpoint is downloaded only when a program or the tables are to be written from it:
	// what was converted runs without it (tables.go)
	checkpoint := sync.OnceValues(func() (string, error) {
		return c.Weights, FetchWeights(ctx, s.Repo, s.Dir, s.Log)
	})
	// The name says what the program is: the format always, since a change of the defaults
	// (LFM3 to lut6 and the grouped attention, 2026-11) must convert anew instead of loading
	// what an earlier version wrote, and the attention where the architecture has the setting.
	format := cmp.Or(s.Format, a.DefaultFormatFor(model, backend))
	kind := a.Name + a.Rev() + "-" + format
	if attn := a.Param(llm.Options{}, "attn"); attn != "" {
		kind += "-" + attn
	}
	if s.Codes != "" {
		if !strings.HasPrefix(format, "lut") {
			return Config{}, fmt.Errorf("chat: -codes are lookup tables, for the lutN formats, not %s", format)
		}
		h, err := codesHash(s.Codes)
		if err != nil {
			return Config{}, err
		}
		kind += "-gptq-" + h
	}
	for _, r := range s.FormatFor {
		sub, f, ok := strings.Cut(r, "=")
		if !ok || sub == "" || f == "" || strings.ContainsAny(r, "/\\ ") {
			return Config{}, fmt.Errorf("chat: -format-for %q is not SUBSTRING=FORMAT", r)
		}
		kind += "-" + sub + "." + f
	}
	// with drafts from the multi-token prediction layer, the programs also output their final
	// hidden states, which the layer reads: other programs, named -h
	var mtp string
	params := map[string]string{}
	if s.Draft > 1 && s.MTP != MTPOff && backend != llm.BackendIntel {
		if mtp, err = mtpArch(c.ModelConfig, a, pdir, checkpoint); err != nil {
			return Config{}, err
		}
		if mtp != "" {
			kind += "-h"
			params["hidden-out"] = "1"
		}
	}
	// with drafts, the program that checks them, of a model whose state a step replaces as a
	// whole (Qwen3.5's Gated DeltaNet), also outputs what each row adds to it, so that keeping a
	// beginning of the drafts does not run them again (llm.StateReplayer): another program,
	// named -sr
	stateRows := s.Draft > 1 && a.Param(llm.Options{}, "state-rows") != ""
	chunk, context := pagedChunk(model, s.Paged, backend), s.Context
	if context <= 0 {
		context = DefaultContext
		if pm, ok := model.(llm.Paged); ok && chunk > 0 && pm.MaxContext() > 0 {
			context = pm.MaxContext()
		} else if u, ok := model.(llm.Unpaged); ok && chunk == 0 {
			context = u.UnpagedContext()
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
	// architecture can (llm.Options.LastRow): the output layer is the model's widest. Split at
	// its attention, a model whose later layers write no cache (Gemma 4's shared layers) has one
	// that only writes the caches (llm.Options.KVOnly), without those layers and the output layer,
	// and the program that generates reads the last tokens of a prompt
	type prog struct {
		seq, parts      int
		lastRow, kvOnly bool
	}
	pre := prog{seq: prefillSeq, lastRow: a.LastRow && backend != llm.BackendIntel}
	if chunk > 0 && backend != llm.BackendIntel {
		pre.seq = min(pagedPrefillSeq, chunk) // a call's tokens fit in a chunk
		if cw, ok := model.(llm.CacheWriters); ok && cw.CacheLayers() < model.Layers() {
			pre.lastRow, pre.kvOnly = false, true
		}
	}
	progs := []prog{{seq: 1, parts: 2}, pre}
	if backend == llm.BackendIntel {
		progs[0].parts = 0
	}
	if s.Draft > 1 {
		progs[0] = prog{seq: s.Draft}
	}
	tiers := s.Tiers
	switch {
	case chunk > 0 && backend == llm.BackendCoreAI:
		// whole programs with short caches (inside) before the one split at its attention: a
		// step in fewer calls while the conversation is short (wholeTiers)
		if slices.Contains(tiers, TierAuto) {
			gen, _ := programBytes(model, format, false)
			tiers = wholeTiers(model, gen)
		}
	case chunk > 0:
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
	// the programs loaded beside the model's, which the Neural Engine holds too where it holds
	// only so much (neuralEngineHold): the multi-token prediction layer's and the encoders'
	var others int64
	hold := neuralEngineHold()
	if hold > 0 && backend != llm.BackendIntel {
		others = besideBytes(c.ModelConfig, mtp, s.Format, backend)
	}
	shared := sharesWeights(c.Backend)
	one, why := onePrograms(model, format, backend, s.Prefill, s.Draft, len(caches)+1, pre.kvOnly, shared, others)
	if one {
		progs = progs[:1]
		if s.Log != nil {
			fmt.Fprintf(s.Log, "one program for %s, which reads the prompts %d tokens per call too: %s (-prefill on for one of %d to read long prompts faster)\n", c.Name, s.Draft, why, pre.seq)
		}
	}
	if gen, read := programBytes(model, format, pre.kvOnly); hold > 0 && backend != llm.BackendIntel && s.Log != nil {
		if held := heldBytes(gen, read, len(caches)+1, one, shared) + others; held > hold {
			fmt.Fprintf(s.Log, "%s's programs hold %.2f GiB on the Neural Engine, over the %.1f GiB an M1 holds before every call slows down about 12 times: expect it to be slow on this machine (a smaller -format holds less)\n", c.Name, float64(held)/(1<<30), float64(hold)/(1<<30))
		}
	}
	// the programs of each cache: each one's file without its extension, function name (Core AI)
	// and settings. A model split at its attention with tiers has whole programs (their caches
	// inside) for the shorter caches: on Core AI all of them are written together, so that the
	// whole programs' parts share the weights of the split ones' in one Core AI program
	type tierProgs struct {
		outs, names []string
		ps          []program
		base        string
	}
	hybrid := chunk > 0 && len(caches) > 0
	var tps []tierProgs
	missing := false
	for _, cache := range append(caches, context) {
		var outs, names []string
		var ps []program
		tkind, tchunk, tprogs := kind, chunk, progs
		if hybrid && cache < context {
			tkind, tchunk = strings.Replace(kind, fmt.Sprintf("-p%d", chunk), "", 1), 0
			tprogs = []prog{progs[0]}
			if len(progs) > 1 {
				tprogs = append(tprogs, prog{seq: min(prefillSeq, cache), lastRow: a.LastRow})
			}
		}
		base := filepath.Join(pdir, fmt.Sprintf("%s-c%d", tkind, cache))
		for i, p := range tprogs {
			last := ""
			switch {
			case p.lastRow:
				last = "-last"
			case p.kvOnly:
				last = "-kv"
			}
			pk, pp, name := tkind, params, fmt.Sprintf("q%d%s", p.seq, last)
			if i == 0 && stateRows {
				pk, pp, name = tkind+"-sr", maps.Clone(params), "sr-"+name
				pp["state-rows"] = "1"
			}
			if hybrid {
				name = fmt.Sprintf("c%d-%s", cache, name) // the functions of all the caches together
			}
			out := filepath.Join(pdir, fmt.Sprintf("%s-c%d-q%d%s", pk, cache, p.seq, last))
			if backend != llm.BackendCoreAI { // Core AI's are checked by their export below
				missing = missing || !converted(ctx, out+programExt(backend))
			}
			outs, names, base = append(outs, out), append(names, name), base+"-"+name
			ps = append(ps, program{backend: backend, arch: s.Arch, format: s.Format, codes: s.Codes, formatFor: s.FormatFor, params: pp, seq: p.seq, parts: p.parts, cache: cache, paged: tchunk, lastRow: p.lastRow, kvOnly: p.kvOnly})
		}
		tps = append(tps, tierProgs{outs, names, ps, base})
	}
	var all tierProgs // a model split at its attention with tiers on Core AI: every tier's programs
	if hybrid && backend == llm.BackendCoreAI {
		all.base = filepath.Join(pdir, kind)
		for _, tp := range tps {
			all.outs, all.ps = append(all.outs, tp.outs...), append(all.ps, tp.ps...)
			for _, n := range tp.names {
				all.names, all.base = append(all.names, n), all.base+"-"+n
			}
		}
		missing = !converted(ctx, all.base+programExt(backend))
	} else if backend == llm.BackendCoreAI {
		for _, tp := range tps {
			missing = missing || !converted(ctx, tp.base+programExt(backend))
		}
	}
	if missing {
		if _, err := checkpoint(); err != nil {
			return Config{}, err
		}
	}
	if hybrid && backend == llm.BackendCoreAI {
		pkgs, err := convertShared(ctx, c.ModelConfig, c.Weights, all.base, all.names, all.ps, s.Log)
		if err != nil {
			return Config{}, err
		}
		for i := range tps {
			n := len(tps[i].outs)
			tps[i].outs, pkgs = pkgs[:n], pkgs[n:] // the manifests in place of the files
		}
	}
	for ti, tp := range tps {
		outs, names, ps, base := tp.outs, tp.names, tp.ps, tp.base
		var pkgs []string
		switch {
		case hybrid && backend == llm.BackendCoreAI:
			pkgs = outs
		case backend == llm.BackendCoreAI:
			// the programs of one cache together, their parts of the same layers the functions
			// of one Core AI program: the weights once on the disk and on the Neural Engine
			if pkgs, err = convertShared(ctx, c.ModelConfig, c.Weights, base, names, ps, s.Log); err != nil {
				return Config{}, err
			}
		default:
			for i, p := range ps {
				path, err := convert(ctx, c.ModelConfig, c.Weights, outs[i]+programExt(backend), p, s.Log)
				if err != nil {
					return Config{}, err
				}
				pkgs = append(pkgs, path)
			}
		}
		if chunk > 0 && (!hybrid || ti == len(tps)-1) {
			for _, path := range pkgs {
				// a conversion from before the two-level attention gets its mass program, which
				// has no weights, instead of being converted again
				form, scale := cmp.Or(a.Param(llm.Options{}, "attn"), "grouped"), model.(llm.Paged).PagedShape().Scale
				if err := llm.AddMass(path, form, scale); err != nil {
					return Config{}, err
				}
				// the attention over shorter chunks, also without weights
				if err := llm.AddSmaller(path, form, scale, shortestChunk(backend, s.Shortest)); err != nil {
					return Config{}, err
				}
			}
		}
		c.Tiers = append(c.Tiers, pkgs)
	}
	c.Packages = c.Tiers[len(c.Tiers)-1]
	if mtp != "" {
		// the multi-token prediction layer, with a cache of the context: it reads the model's
		// hidden states from where it starts, and only the last row's logits are drafts
		ma, mm, err := llm.LoadConfig(c.ModelConfig, mtp)
		if err != nil {
			return Config{}, err
		}
		mkind := ma.Name + ma.Rev() + "-" + cmp.Or(s.Format, ma.DefaultFormatFor(mm, backend))
		if chunk > 0 {
			mkind += fmt.Sprintf("-p%d", chunk)
		}
		// Core AI's drafts are of a part of the vocabulary (draftVocab), listed in the export
		var rows []int32
		if backend == llm.BackendCoreAI {
			tok, err := tokenizer.Load(filepath.Join(s.Dir, TokenizerFiles(a.Name)[0]))
			if err != nil {
				return Config{}, err
			}
			if rows = draftVocab(tok); rows != nil {
				mkind += "-" + draftVocabRule
			}
		}
		out := filepath.Join(pdir, fmt.Sprintf("%s-c%d-q%d-last%s", mkind, context, mtpSeq, programExt(backend)))
		if !converted(ctx, out) {
			if _, err := checkpoint(); err != nil {
				return Config{}, err
			}
		}
		path, err := convert(ctx, c.ModelConfig, c.Weights, out, program{backend: backend, arch: mtp, format: s.Format, seq: mtpSeq, cache: context, paged: chunk, lastRow: true, headRows: rows}, s.Log)
		if err != nil {
			return Config{}, err
		}
		if pm, ok := mm.(llm.Paged); ok && chunk > 0 {
			// the layer's attention over shorter chunks too, as the model's (without weights):
			// over the whole chunk it took 3 of its 9 ms a call (Qwen3.5-4B, docs/design.md 16.192)
			if err := llm.AddSmaller(path, cmp.Or(ma.Param(llm.Options{}, "attn"), "grouped"), pm.PagedShape().Scale, shortestChunk(backend, s.Shortest)); err != nil {
				return Config{}, err
			}
		}
		c.MTP = []string{path}
	}
	shareOnce(pdir, s.Log)
	shareMade(ctx) // the exports split at their attention, their programs all added
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
		if !converted(ctx, out) {
			if _, err := checkpoint(); err != nil {
				return Config{}, err
			}
		}
		path, err := convert(ctx, c.ModelConfig, c.Weights, out, program{backend: backend, arch: e.arch, format: "fp16", params: map[string]string{e.param: e.value}}, s.Log)
		if err != nil {
			return Config{}, err
		}
		*e.packages(&c) = []string{path}
	}
	if c.Weights, err = writeTables(ctx, s.Dir, a, model, checkpoint, s.Log); err != nil {
		return Config{}, err
	}
	// A conversion reads the checkpoint into memory (2.4 GiB of PLaMo 3 Translate 2B left after
	// it): give it back now rather than as Go's scavenger does, a little at a time.
	debug.FreeOSMemory()
	return c, nil
}

// sharesWeights reports whether the programs of a model run on the device backend (Config.Backend)
// share their weights on the Neural Engine: on Core AI the programs of one model's weights are
// functions of one program (internal/backend/coreai), which the Neural Engine holds once
// (LFM3-1.7B's two: 1.34 GiB against 2.56 GiB lowered apart, docs/design.md 16.177).
func sharesWeights(device string) bool { return device == "coreai" }

// heldBytes is what a model's programs hold on the Neural Engine: tiers of the one that generates
// (gen bytes) and, unless one, of the one that reads prompts (read); once gen when they share
// their weights (the one that reads prompts has a part of the other's: sharesWeights).
func heldBytes(gen, read int64, tiers int, one, shared bool) int64 {
	if shared {
		return gen
	}
	held := int64(tiers) * gen
	if !one {
		held += int64(tiers) * read
	}
	return held
}

// onePrograms decides Setup.Prefill: whether the program that generates (of draft tokens per
// call) reads the prompts too, instead of a pair of programs for each of tiers cache lengths, and
// why. The one that reads prompts has the model's weights in format, or with kvOnly those of the
// layers that write the caches (llm.Options.KVOnly). Two are too many when they would not stay
// in the Neural Engine's compile cache (compileCacheBytes), or, with the others loaded beside
// them (bytes of weights), would hold more than the machine's Neural Engine holds before it slows
// down (neuralEngineHold, an M1's). Programs that share their weights (shared, sharesWeights)
// hold no more for the one that reads prompts, and Core AI keeps what it compiles in a cache of
// its own on the disk: they are never too many.
func onePrograms(model llm.Model, format, backend, prefill string, draft, tiers int, kvOnly, shared bool, others int64) (bool, string) {
	switch {
	case prefill == PrefillOff:
		return true, "prefill off (--prefill, or the model's definition for this machine's memory)"
	case prefill == PrefillOn || draft <= 1 || backend == llm.BackendIntel || shared:
		return false, "" // reading prompts one token per call would be far slower; the cache is the Neural Engine's
	}
	gen, read := programBytes(model, format, kvOnly)
	total := heldBytes(gen, read, tiers, false, false)
	if hold := neuralEngineHold(); hold > 0 && total+others > hold {
		why := fmt.Sprintf("%d programs would hold %.2f GiB on the Neural Engine", 2*tiers, float64(total)/(1<<30))
		if others > 0 {
			why += fmt.Sprintf(" with the %.2f GiB of the programs beside them", float64(others)/(1<<30))
		}
		return true, why + fmt.Sprintf(", over the %.1f GiB an M1 holds before every call slows down about 12 times", float64(hold)/(1<<30))
	}
	if total <= compileCacheBytes {
		return false, ""
	}
	return true, fmt.Sprintf("%d programs would hold %.2f GiB of weights, over the about %.1f GiB the Neural Engine keeps compiled, so every start would compile them again and every question switch between them", 2*tiers, float64(total)/(1<<30), float64(compileCacheBytes)/(1<<30))
}

// programBytes estimates the weights of a model's program that generates and of the one that
// reads prompts, in format; with kvOnly the latter has only the layers that write the caches and
// no output layer.
func programBytes(model llm.Model, format string, kvOnly bool) (gen, read int64) {
	pre, layers, post := estimate(model, format)
	n := len(layers) // the layers of the program that reads prompts
	if kvOnly {
		n = model.(llm.CacheWriters).CacheLayers()
	}
	gen, read = pre+post, pre
	if !kvOnly {
		read += post
	}
	for i, l := range layers {
		gen += l
		if i < n {
			read += l
		}
	}
	return gen, read
}

// estimate is llm.EstimateBytes with the converter's defaults and the weights in format.
func estimate(model llm.Model, format string) (pre int64, layers []int64, post int64) {
	var o llm.Options
	set := llm.BindFlags(flag.NewFlagSet("", flag.ContinueOnError), &o, "")
	set()
	o.Format = format
	return llm.EstimateBytes(model, o)
}

// besideBytes estimates the weights of the programs Prepare converts beside a model's and the
// engine loads with them: the multi-token prediction layer's (arch mtp, "" for none, in format or
// its default) and the encoders' of a multimodal checkpoint (fp16).
func besideBytes(config, mtp, format, backend string) int64 {
	var n int64
	add := func(m llm.Model, f string) {
		pre, layers, post := estimate(m, f)
		n += pre + post
		for _, l := range layers {
			n += l
		}
	}
	if mtp != "" {
		if ma, mm, err := llm.LoadConfig(config, mtp); err == nil {
			add(mm, cmp.Or(format, ma.DefaultFormatFor(mm, backend)))
		}
	}
	for _, e := range encoders {
		if _, em, err := llm.LoadConfig(config, e.arch); err == nil {
			add(em, "fp16")
		}
	}
	return n
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

// shortestChunk is the shortest chunk of the attention programs of a model split at its attention
// besides its own (Setup.Shortest; llm.AddSmaller), 0 for none: want, else the backend's. On an
// M3's Neural Engine with Core AI and LFM3-1.7B, a call over a chunk of 1024 took 0.50 ms a
// layer, 2048 0.63, 4096 0.93 and 8192 1.49: a conversation of 150 tokens went from 44.1 to 53.7
// tokens a second. Core ML puts the attention of 8 tokens a call over chunks under 4096 on the
// CPU (the chain leaves such a program out), so it starts at 4096: 42.4 to 47.9. Intel's NPU is
// not measured (docs/design.md 16.186).
func shortestChunk(backend string, want int) int {
	switch {
	case want == PagedOff:
		return 0
	case want > 0:
		return want
	case backend == llm.BackendCoreAI:
		return 1024
	case backend == "" || backend == llm.BackendCoreML:
		return 4096
	}
	return 0
}

// wholeLayerBytes is the most bytes of keys and values a layer of a whole program's caches holds
// (wholeTiers): a call copies all of them into Core AI's arrays and attends over them. On an M3's
// Neural Engine with Core AI, LFM3-1.7B (8 kv heads of 64, 2 KiB a position) generated 64, 62, 59
// and 52 tokens a second whole with caches of 512 to 4096 against 55 split at its attention, and
// Qwen3-0.6B (8 of 128, 4 KiB) 68, 53, 35 and 20 against 34 to 40: both even at about 4 MiB a
// layer (docs/design.md 16.188).
const wholeLayerBytes = 4 << 20

// wholeHold is the most weights the whole program and those split at their attention hold on
// the Neural Engine together (wholeTiers): each has its own Core AI programs (in one with the
// others, or with a whole program of another cache, a whole program ran at half its speed,
// docs/design.md 16.188). An M1 slows down about 12 times past 2.9 GiB (16.118).
const wholeHold = 5 << 29 // 2.5 GiB

// wholeTiers is the cache of the whole program (its caches inside, none of its attention on the
// host) a model split at its attention runs on with Core AI while the conversation is short,
// before it moves to the one split at its attention (decode.Session.Migrate): the longest of 512,
// 1024, 2048 and 4096 whose keys and values a layer are at most wholeLayerBytes. None for a model
// whose weights (gen bytes) twice are over wholeHold.
func wholeTiers(model llm.Model, gen int64) []int {
	pm, ok := model.(llm.Paged)
	if !ok || 2*gen > wholeHold || len(pm.PagedLayers()) == 0 {
		return nil
	}
	sh := pm.PagedShape()
	row := 2 * 2 * sh.KVHeads * sh.HeadDim // fp16 keys and values of a position
	longest := 0
	for _, c := range []int{512, 1024, 2048, 4096} {
		if row > 0 && c*row <= wholeLayerBytes {
			longest = c
		}
	}
	if longest == 0 {
		return nil
	}
	return []int{longest}
}

// pagedChunk is the chunk length to split the model at its attention with (Setup.Paged), 0 for
// none: the model must be Paged (LFM3 and its VL, Audio and Omni checkpoints, Gemma 4), on any
// NPU, and an Unpaged one (Qwen3) is split only when Setup.Paged asks, but with Core AI, where it
// runs faster split (with the attention over shorter chunks, shortestChunk; docs/design.md
// 16.187).
func pagedChunk(model llm.Model, want int, backend string) int {
	pm, ok := model.(llm.Paged)
	if !ok || want < 0 || len(pm.PagedLayers()) == 0 {
		return 0
	}
	if _, whole := model.(llm.Unpaged); whole && want == 0 && backend != llm.BackendCoreAI {
		return 0 // split only when told (Qwen3)
	}
	return cmp.Or(want, paged.DefaultChunk)
}

// encoders are the programs of the multimodal checkpoints besides the text model, converted when
// config.json has them: LFM3-VL's and LFM3-Omni's vision encoder for one image size (an image is
// stretched to it), LFM3-Audio's and LFM3-Omni's audio encoder for clips of up to 30 seconds. They
// stay in fp16 as hwnpu-dev convert's examples convert them (docs/design.md 16.28, 16.29); files are
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
	paged             int      // llm.Options.Paged
	lastRow           bool     // llm.Options.LastRow
	kvOnly            bool     // llm.Options.KVOnly
	codes             string   // llm.Options.Codes
	formatFor         []string // llm.Options.FormatFor
	headRows          []int32  // llm.Options.HeadRows, listed in a Core AI export (draftVocabFile)
}

// codesHash names a codes directory (llm.Options.Codes) by its index.json: the first 8 hex
// digits of its SHA-256. The index has each weight's key, a hash of its values.
func codesHash(dir string) (string, error) {
	b, err := os.ReadFile(filepath.Join(dir, "index.json"))
	if err != nil {
		return "", fmt.Errorf("chat: -codes: %w", err)
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:4]), nil
}

// convert writes the program of the model (config.json and the checkpoint) unless it is there,
// and returns its path. NAME.done, written last, holds the path (relative to it, cache.WriteDone), so that a conversion cut off is
// done again.
func convert(ctx context.Context, config, weights, out string, p program, log io.Writer) (string, error) {
	done := doneFile(out)
	if p.backend == llm.BackendCoreAI {
		converted(ctx, out) // from the shared layers when one has it
	}
	if p, ok := cache.ReadDone(done); ok {
		return p, nil
	}
	unlock, err := cache.Lock(ctx, out, filepath.Base(out), log)
	if err != nil {
		return "", err
	}
	defer unlock()
	if p, ok := cache.ReadDone(done); ok { // another process converted it meanwhile
		return p, nil
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
	opts, err := programOptions(a, model, p)
	if err != nil {
		return "", err
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
	var codes *palette.Codes
	hits := 0
	if p.codes != "" {
		if codes, err = blocks.LoadCodes(p.codes); err != nil {
			return "", err
		}
		hits = codes.Hits()
	}
	dst := out
	if p.backend == llm.BackendCoreAI { // an export: its program, or its parts and manifest, inside
		if err := freshDir(out); err != nil {
			return "", err
		}
		dst = filepath.Join(out, "model.aimodel")
	}
	path, err := llm.Convert(a, model, w, dst, "any", p.parts, opts, log)
	if err != nil {
		return "", err
	}
	if codes != nil && log != nil {
		fmt.Fprintf(log, "-codes: %d weights took the chosen tables (%d in %s)\n", codes.Hits()-hits, codes.Len(), p.codes)
	}
	if len(p.headRows) > 0 {
		if p.backend != llm.BackendCoreAI {
			return "", fmt.Errorf("chat: a part of the vocabulary is listed in a Core AI export alone")
		}
		if err := writeDraftVocab(out, p.headRows); err != nil {
			return "", err
		}
	}
	if err := cache.WriteDone(done, path); err != nil {
		return "", err
	}
	if p.backend == llm.BackendCoreAI {
		shareExport(ctx, out, p.paged > 0)
	}
	return path, nil
}

// programOptions are the converter's options for p: its defaults and the architecture's format,
// with the table fitting that keeps LFM3's answers as they were on the Neural Engine (also used
// for the embedding model); the integer formats keep the converter's scale search.
func programOptions(a *llm.Arch, model llm.Model, p program) (llm.Options, error) {
	fs := flag.NewFlagSet("", flag.ContinueOnError)
	var opts llm.Options
	set := llm.BindFlags(fs, &opts, "")
	if err := fs.Parse(nil); err != nil {
		return opts, err
	}
	set()
	opts.Backend = p.backend
	opts.Format = cmp.Or(p.format, a.DefaultFormatFor(model, p.backend))
	if p.backend != llm.BackendIntel && !strings.HasPrefix(opts.Format, "int") {
		opts.Quant = "kmeans-minmax"
	}
	opts.Seq, opts.Cache, opts.Paged, opts.LastRow, opts.KVOnly = p.seq, p.cache, p.paged, p.lastRow, p.kvOnly
	opts.Codes, opts.FormatFor, opts.HeadRows = p.codes, p.formatFor, p.headRows
	for k, v := range p.params {
		opts.Params[k] = v
	}
	return opts, nil
}

// convertShared writes the programs ps of one model for Core AI together (llm.ConvertAI) into the
// export base.coreai, the parts of the same layers as the functions of one Core AI program
// model.partK.aimodel and each program's manifest NAME.parts.json (names[i]), unless they are
// there, and returns the manifests' paths. base.coreai.done, written last, says they are; a
// conversion cut off is done again, as a whole since the programs share files.
func convertShared(ctx context.Context, config, weights, base string, names []string, ps []program, log io.Writer) ([]string, error) {
	export := base + programExt(llm.BackendCoreAI)
	done := doneFile(export)
	manifests := make([]string, len(names))
	for i, n := range names {
		manifests[i] = filepath.Join(export, n+".parts.json")
	}
	converted(ctx, export) // from the shared layers when one has it
	if _, ok := cache.ReadDone(done); ok {
		return manifests, nil
	}
	unlock, err := cache.Lock(ctx, export, filepath.Base(export), log)
	if err != nil {
		return nil, err
	}
	defer unlock()
	if _, ok := cache.ReadDone(done); ok { // another process converted them meanwhile
		return manifests, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	a, model, err := llm.LoadConfig(config, ps[0].arch)
	if err != nil {
		return nil, err
	}
	w, err := llm.OpenWeights(weights, a)
	if err != nil {
		return nil, err
	}
	defer w.Close()
	if err := freshDir(export); err != nil {
		return nil, err
	}
	var progs []llm.AIProgram
	for i, p := range ps {
		opts, err := programOptions(a, model, p)
		if err != nil {
			return nil, err
		}
		progs = append(progs, llm.AIProgram{Out: filepath.Join(export, names[i]), Name: names[i], Parts: p.parts, Options: opts})
	}
	p := ps[0]
	if p.paged > 0 {
		fmt.Fprintf(log, "converting %s for the Neural Engine (Core AI, %s; split at its attention, a context of %d held by the host in chunks of %d, %s per call); this takes a few minutes\n", filepath.Base(export), progs[0].Options.Format, p.cache, p.paged, strings.Join(names, " and "))
	} else {
		fmt.Fprintf(log, "converting %s for the Neural Engine (Core AI, %s; cache %d, %s per call); this takes a few minutes\n", filepath.Base(export), progs[0].Options.Format, p.cache, strings.Join(names, " and "))
	}
	paths, err := llm.ConvertAI(a, model, w, filepath.Join(export, "model"), "any", progs, log)
	if err != nil {
		return nil, err
	}
	if err := cache.WriteDone(done, paths[0]); err != nil {
		return nil, err
	}
	shareExport(ctx, export, p.paged > 0)
	return paths, nil
}

// madePaged are the Core AI exports of models split at their attention converted here, put in
// the shared layers once their programs without weights are added (shareMade): putting one
// describes its files, which AddMass and AddSmaller change after the conversion.
var madePaged struct {
	sync.Mutex
	exports []string
}

// shareExport puts the Core AI export made here in the shared layers (llm.ShareExport), or,
// one split at its attention (paged), once Prepare has added its programs (shareMade).
func shareExport(ctx context.Context, export string, paged bool) {
	if !paged {
		llm.ShareExport(ctx, export)
		return
	}
	madePaged.Lock()
	madePaged.exports = append(madePaged.exports, export)
	madePaged.Unlock()
}

// shareMade puts the exports shareExport kept in the shared layers.
func shareMade(ctx context.Context) {
	madePaged.Lock()
	exports := madePaged.exports
	madePaged.exports = nil
	madePaged.Unlock()
	for _, e := range exports {
		llm.ShareExport(ctx, e)
	}
}

// freshDir makes dir an empty directory: what a conversion cut off left in an export is removed.
func freshDir(dir string) error {
	if err := os.RemoveAll(dir); err != nil {
		return err
	}
	return os.MkdirAll(dir, 0o755)
}

// repoName is a repository's name: NAME of OWNER/NAME or OWNER/NAME@REVISION.
func repoName(repo string) string {
	repo, _, _ = strings.Cut(repo, "@")
	return filepath.Base(repo)
}

// errNotFound is the error of fetch for a file the repository does not have.
var errNotFound = errors.New("not in the repository")

// NotFound reports whether err is Fetch's for a file the repository does not have.
func NotFound(err error) bool { return errors.Is(err, errNotFound) }

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
	var want []File
	for _, f := range files {
		want = append(want, File{repo, f, filepath.Join(dir, f)})
	}
	Expect(ctx, want)
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

// fetchHF downloads a file of a Hugging Face repository to path (fetch). A download cut off goes
// on from where it stopped (path.part). A gated model's repository (401, 403) goes through gate:
// a token (hfToken), signing in, agreeing to the terms. repo is OWNER/NAME, or
// OWNER/NAME@REVISION for a revision other than main.
func fetchHF(ctx context.Context, repo, name, path string, log io.Writer) error {
	repo, rev, _ := strings.Cut(repo, "@")
	url := hubURL + repo + "/resolve/" + cmp.Or(rev, "main") + "/" + name
	part := path + ".part"
	var have int64
	if fi, err := os.Stat(part); err == nil {
		have = fi.Size()
	}
	// the version of the file the part is of (its ETag): the rest is asked for only if the file
	// is still that one (If-Range), else the whole file comes again. A part of an earlier hwnpu,
	// with no ETag kept, goes on as before
	etag, _ := os.ReadFile(part + ".etag")
	var resp *http.Response
	for try := 0; ; try++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		tok := hfToken()
		if tok != "" {
			req.Header.Set("Authorization", "Bearer "+tok)
		}
		if have > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", have))
			if len(etag) > 0 {
				req.Header.Set("If-Range", string(etag))
			}
		}
		resp, err = http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		if (resp.StatusCode != http.StatusUnauthorized && resp.StatusCode != http.StatusForbidden) || try == 4 ||
			!isGated(ctx, repo) {
			break
		}
		resp.Body.Close()
		if log == nil {
			log = io.Discard
		}
		if err := gate(ctx, repo, resp.StatusCode, tok, log); err != nil {
			return err
		}
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
	noteCommit(repo, rev, resp)
	if e := resp.Header.Get("ETag"); e != "" && !strings.HasPrefix(e, "W/") {
		os.WriteFile(part+".etag", []byte(e), 0o644)
	} else {
		os.Remove(part + ".etag")
	}
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	pw := &progress{w: f, done: have, from: have, total: total, plan: planned(path),
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
	os.Remove(part + ".etag")
	return os.Rename(part, path)
}

// commits are the commits Hugging Face served the repositories' files of a revision from
// (X-Repo-Commit), by OWNER/NAME@REVISION: what a copy of a file is put under in a shared layer
// (docs/cache-sharing.md 7.1), never a branch's name.
var (
	commitsMu sync.Mutex
	commits   = map[string]string{}
)

var commitRE = regexp.MustCompile(`^[0-9a-f]{40}$`)

// noteCommit keeps the commit a response of the hub (or one it redirected from) names.
func noteCommit(repo, rev string, resp *http.Response) {
	for r := resp; r != nil; {
		if c := r.Header.Get("X-Repo-Commit"); commitRE.MatchString(c) {
			commitsMu.Lock()
			commits[repo+"@"+cmp.Or(rev, "main")] = c
			commitsMu.Unlock()
			return
		}
		if r.Request == nil {
			return
		}
		r = r.Request.Response
	}
}

// servedCommit is the commit noteCommit kept for repo (OWNER/NAME or OWNER/NAME@REVISION): the
// revision itself when it is a commit.
func servedCommit(repo string) (string, bool) {
	base, rev, _ := strings.Cut(repo, "@")
	if commitRE.MatchString(rev) {
		return rev, true
	}
	commitsMu.Lock()
	defer commitsMu.Unlock()
	c, ok := commits[base+"@"+cmp.Or(rev, "main")]
	return c, ok
}

// progress writes through and shows how far a download is on its bar: the bytes, the speed and
// the time left. For a file of the files Expect looked up, the bytes and the time left are also
// those of all of them still to come.
type progress struct {
	w                 io.Writer
	done, from, total int64 // total < 0: not known
	plan              *plannedFile
	bar               *console.Bar
	last              time.Time
}

func (p *progress) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.done += int64(n)
	if p.plan != nil {
		planWrote(p.plan, p.done)
	}
	if time.Since(p.last) >= 200*time.Millisecond {
		p.last = time.Now()
		frac := -1.0
		detail := gb(p.done, p.total)
		left := p.total - p.done
		if p.plan != nil {
			if done, total := planNow(); total > p.total {
				detail += ", all " + gb(done, total)
				left = total - done
			}
		}
		if sec := p.bar.Elapsed().Seconds(); sec > 0.5 {
			rate := float64(p.done-p.from) / sec
			detail += fmt.Sprintf("  %.1f MB/s", rate/1e6)
			if p.total > 0 && rate > 0 {
				detail += ", " + console.Duration(time.Duration(float64(left)/rate*float64(time.Second))) + " left"
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

// Fetch puts a file of a model's repository (Model.Source) at path unless it is there, from the
// model's source or the sources of Search, as Prepare does for the checkpoint.
func Fetch(ctx context.Context, repo, name, path string, log io.Writer) error {
	return fetch(ctx, repo, name, path, log)
}
