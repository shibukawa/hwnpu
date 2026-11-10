// Command hwnpu-dev probe measures the NPU of this machine with generated models, without Python, model
// files or root:
//
//	go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe          # bench, then limit
//	go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe bench    # speed only
//	go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe limit    # size limit only
//	go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe split    # a model over the limit in parts
//	go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe verify   # K tokens per call
//	go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe kv       # attention over a KV cache
//	go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe attn     # attention per chunk of a cache
//	go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe seam     # weighted parts and that attention interleaved
//	go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe cache    # the Neural Engine's cache of compiled programs
//
// The models have the shape of LLM decoding (internal/probe): one token through Llama 3.2 1B
// sized layers with random weights, so every call reads every weight once, as generating one
// token does.
//
// bench runs 8 and 16 layers (0.5 B and 1 B weights) in each weight format and gives the time per
// token, the weight bandwidth and, where the device has a readable power meter (Intel and AMD
// on Linux as root), the energy per token. On Apple it also runs each model on the CPU and the GPU
// through Core ML; for energy there use hwnpu-dev powerbench -decode.
//
// limit grows the model in each format until it leaves the NPU, bisects to the boundary, then
// loads several models just under it at once to tell a limit per model from a limit on the total.
//
// split, run only when asked, builds a model larger than the limit (64 layers, 1.8 GiB of lut4 by
// default), splits it into parts small enough for the NPU and runs the parts one after another,
// the output buffer of each bound as the input of the next, as a split LLM would generate a token.
// It checks that every part stays on the NPU with all of them loaded, compares the time per token
// with the parts' own times added up, and compares the output after every part with the same parts
// run one at a time on the GPU.
//
// verify, run only when asked for, runs the bench models on the NPU with K tokens per call (1 to
// 128 by default), the call that checks K drafted tokens in speculative decoding or reads a chunk
// of the prompt. The K at which the time per call starts to grow is about the most speculative
// decoding can gain and the chunk that uses the NPU best; past it the call is bound by the compute,
// which the report gives in TFLOPS.
//
// attn, run only when asked for (Core ML), builds the attention without weights over one chunk of
// a KV cache that a model split at its attention would call once per used chunk and layer
// (docs/model-fit.md T2), checks that it stays on the Neural Engine, times one call, and then
// times a step's attention over caches of several used lengths (up to LFM3-1.7B's 262144): every
// chunk in buffers of its own, the outputs merged on the host. A step then costs in proportion to
// the tokens used instead of to the cache's capacity; the passes show the cost per call, what the
// host's merge adds, and the memory of the process holding the cache.
//
// seam, run only when asked for (Core ML), loads a model split at its attention: weighted parts
// (40 lut4 layers by default, about the 1.2 GiB of LFM3-1.7B in lut6) in one more part than there
// are attention layers, and the chunk attention program of attn between each two. It times a
// step of the parts alone, of the attention alone and of the two interleaved; what interleaving
// adds, per switch between a program with weights and one without, is the cost of the seams of
// such a model. Alternating two programs with weights has cost a reload of the Neural Engine
// (35 ms a call); a seam within one chain 0.1 ms.
//
// cache, run only when asked for (Core ML), measures the Neural Engine's own cache of compiled
// programs: a program hwnpu's compile cache holds loads in 0.1 s when the Neural Engine's cache
// hits and compiles again for seconds or minutes when it misses. A configuration writes N
// programs of S GiB (-cache-sizes; N from -cache-totals) once, then loads them in order and runs
// each once in a new process per round (-cache-rounds), keeping hwnpu's compile cache; a load at
// or under -cache-hit is a hit. The report gives the hits of every round and the largest total
// that stays warm for each size. -cache-hold open keeps every program open instead of closing
// each before the next, -cache-order reverse reverses every other round (an LRU then hits more),
// -cache-passes 2 loads again within each process, -cache-names runs rounds under other
// executable names (copies of this one), and -cache-keep with -dir keeps the programs for a
// later run, after a restart say (written again, a program is another one to the caches). The
// probe holds the Neural Engine's lock (internal/llm) while it runs, so that other hwnpu
// processes wait; anything else compiling for the Neural Engine spoils the times.
//
// Placement comes from Core ML's compute plan on Apple; on Intel the driver compiler either builds
// the whole graph for the NPU or fails. The report also gives the process CPU time per wall time
// while a model runs: near 0 when the NPU does the work, 1 or more when the CPU does.
//
// The report is written to npuprobe-<machine>-<time>.md and .json.
package probe

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/probe"
)

type report struct {
	Date      string        `json:"date"`
	CPU       string        `json:"cpu"`
	Model     string        `json:"model"`
	OS        string        `json:"os"`
	RAMBytes  int64         `json:"ram_bytes"`
	Device    string        `json:"device"`
	Backend   string        `json:"backend"`
	Driver    string        `json:"driver,omitempty"`
	Firmware  string        `json:"firmware,omitempty"`
	Version   string        `json:"hwnpu_version"`
	Go        string        `json:"go"`
	IdleW     float64       `json:"idle_w,omitempty"`
	Meter     string        `json:"meter,omitempty"`
	MaxBytes  int64         `json:"max_bytes"`
	Bench     []result      `json:"bench,omitempty"`
	Verify    []result      `json:"verify,omitempty"`
	KV        []kvResult    `json:"kv,omitempty"`
	Attn      []attnResult  `json:"attn,omitempty"`
	Seam      []seamResult  `json:"seam,omitempty"`
	Sweep     []result      `json:"sweep,omitempty"`
	Several   []result      `json:"several,omitempty"`
	SeveralOf string        `json:"several_of,omitempty"`
	Split     []splitResult `json:"split,omitempty"`
	Cache     []cacheResult `json:"cache,omitempty"`
	Notes     []string      `json:"notes,omitempty"`
}

func Main() {
	if len(os.Args) == 3 && os.Args[1] == childMode {
		if err := runChild(os.Args[2]); err != nil {
			fmt.Fprintln(os.Stderr, "hwnpu-dev probe:", err)
			os.Exit(1)
		}
		return
	}
	mode := "all"
	args := os.Args[1:]
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		mode, args = args[0], args[1:]
	}
	fs := flag.NewFlagSet("hwnpu-dev probe", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: hwnpu-dev probe [all|bench|limit|split|verify|kv|attn|seam|cache] [flags]")
		fs.PrintDefaults()
	}
	formats := fs.String("formats", "", "weight formats (default: fp16,int8,lut4 and int4 on the CPU and GPU on Apple; fp16,int8,int4 on Intel; split: lut4 on Apple, int4 on Intel)")
	layers := fs.String("layers", "8,16", "bench and verify: layer counts (61 M weights each)")
	tokens := fs.String("tokens", "1,2,4,8,16,32,64,128", "verify: tokens per call (K drafted tokens checked at once, or a chunk of the prompt)")
	baselines := fs.String("baselines", "cpu,gpu", "bench on Apple: also run each model on these Core ML units (empty for none)")
	maxGB := fs.Float64("max-gb", 0, "limit: largest weight size to try, GiB (default: a quarter of the RAM on Apple, a sixth on Intel, at most 8)")
	several := fs.Bool("several", true, "limit: after the sweep, load several models just under the limit at once")
	splitLayers := fs.Int("split-layers", 64, "split: layers of the whole model (61 M weights each; 64 is 1.8 GiB of lut4)")
	parts := fs.String("parts", "2,4", "split: numbers of parts to try (1 runs the model unsplit)")
	layout := fs.String("layout", "files", "split: files (a package per part), functions (one package, a function per part; Core ML on macOS 15) or both, comma separated")
	ref := fs.String("ref", "", "split: Core ML units for the reference output, cpu or gpu, or none (default gpu on Apple, none elsewhere)")
	kvLayers := fs.Int("kv-layers", 11, "kv: layers with attention over the cache (11 is LFM3-1.7B's attention layers)")
	kvCache := fs.String("kv-cache", "256,1024,4096", "kv: cache lengths in tokens")
	kvModes := fs.String("kv-modes", "host,state", "kv: host (buffers bound as inputs and outputs) and state (Core ML MLState); -mha gives every query head its own cached head, so nothing is repeated; -int8 holds the host cache as int8 with fp16 scales, cast back in the program; -chunk holds the host cache in chunks of at most 16384 tokens, so that caches over the Neural Engine's dimension limit stay on it")
	attnChunks := fs.String("attn-chunks", "1024,2048,4096,8192,16384", "attn: chunk lengths in tokens (at most 16384, the Neural Engine's dimension limit; Core ML puts the shortest on the CPU with few query rows)")
	attnTokens := fs.String("attn-tokens", "1,8,64", "attn: query rows per call (1 a step, 8 a step checking drafts, 64 a chunk of the prompt)")
	attnForms := fs.String("attn-forms", "grouped", "attn: grouped (the query heads of a kv head stacked in one matrix product, LFM3-1.7B's default) and tile (keys and values copied per query head)")
	attnUsed := fs.String("attn-used", "4096,16384,65536,262144", "attn: used cache lengths of the passes, ascending; the chunk buffers of every layer are held at once (12 KiB a token with 6 layers: 3 GiB at 262144)")
	attnLayers := fs.Int("attn-layers", 6, "attn: layers per pass (6 is LFM3-1.7B's full attention layers)")
	seamFormat := fs.String("seam-format", "lut4", "seam: weight format of the weighted parts")
	seamLayers := fs.Int("seam-layers", 40, "seam: weighted layers (61 M weights each; 40 in lut4 is 1.14 GiB, about LFM3-1.7B in lut6), split into -attn-layers + 1 parts")
	seamTokens := fs.String("seam-tokens", "8,64", "seam: query rows per call of the parts and the attention (8 a step checking drafts, 64 a chunk of the prompt)")
	seamForm := fs.String("seam-form", "grouped", "seam: form of the attention, grouped or tile")
	seamChunk := fs.Int("seam-chunk", 8192, "seam: chunk length of the attention")
	seamUsed := fs.String("seam-used", "8192,65536", "seam: used cache lengths, ascending")
	cacheFormat := fs.String("cache-format", "fp16", "cache: weight format of the programs (fp16, int8, lut4)")
	cacheSizes := fs.String("cache-sizes", "0.5,1", "cache: sizes of one program, GiB, comma separated (rounded to whole layers of 61 M weights)")
	cacheTotals := fs.String("cache-totals", "1,2,3,4,5,6,8", "cache: totals of the programs of a configuration, GiB, comma separated (programs: total / size)")
	cacheRounds := fs.Int("cache-rounds", 3, "cache: rounds per configuration, each a new process loading every program (the first round compiles them)")
	cacheHold := fs.String("cache-hold", "close", "cache: close (each program closed before the next loads, as Sana's pipeline does) and open (all held until the round ends), comma separated")
	cacheOrder := fs.String("cache-order", "same", "cache: same (every round in one order) and reverse (every other round reversed: an LRU then hits more), comma separated")
	cachePasses := fs.Int("cache-passes", 1, "cache: passes over the programs within each round's process (2: also loading again in the same process)")
	cacheNames := fs.String("cache-names", "", "cache: after the rounds, a round under each of these executable names (a copy of this one), each followed by a round under this name again")
	cacheHit := fs.Duration("cache-hit", time.Second, "cache: a load and first call at or under this is a hit of the Neural Engine's cache")
	cachePause := fs.Duration("cache-pause", 2*time.Second, "cache: pause between rounds")
	cacheKeep := fs.Bool("cache-keep", false, "cache: keep the programs in -dir and reuse them in later runs (after a restart, say); a program written again is another program to the caches")
	minRun := fs.Duration("run", 3*time.Second, "time each model for at least this long")
	work := fs.String("dir", "", "directory for generated models (default: a temporary directory; Apple needs free space of about twice -max-gb)")
	out := fs.String("o", "", "report file (default npuprobe-<machine>-<time>.md, and .json beside it)")
	fs.Parse(args)
	if mode != "all" && mode != "bench" && mode != "limit" && mode != "split" && mode != "verify" && mode != "kv" && mode != "attn" && mode != "seam" && mode != "cache" {
		fs.Usage()
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	sp := splitOpts{layers: *splitLayers, parts: *parts, layout: *layout, ref: *ref, kvLayers: *kvLayers, kvCache: *kvCache, kvModes: *kvModes,
		attnChunks: *attnChunks, attnTokens: *attnTokens, attnForms: *attnForms, attnUsed: *attnUsed, attnLayers: *attnLayers,
		seamFormat: *seamFormat, seamLayers: *seamLayers, seamTokens: *seamTokens, seamForm: *seamForm, seamChunk: *seamChunk, seamUsed: *seamUsed}
	co := cacheOpts{format: *cacheFormat, rounds: *cacheRounds, holds: split(*cacheHold), orders: split(*cacheOrder), passes: *cachePasses,
		names: split(*cacheNames), hit: *cacheHit, pause: *cachePause, keep: *cacheKeep}
	if mode == "cache" {
		if err := co.parse(*cacheSizes, *cacheTotals, *work); err != nil {
			fmt.Fprintln(os.Stderr, "hwnpu-dev probe:", err)
			os.Exit(2)
		}
	}
	if err := run(ctx, mode, *formats, *layers, *tokens, *baselines, *maxGB, *several, sp, co, *minRun, *work, *out); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-dev probe:", err)
		os.Exit(1)
	}
}

type splitOpts struct {
	layers             int
	parts, layout, ref string
	kvLayers           int
	kvCache, kvModes   string

	attnChunks, attnTokens, attnForms, attnUsed string
	attnLayers                                  int

	seamFormat, seamTokens, seamForm, seamUsed string
	seamLayers, seamChunk                      int
}

func run(ctx context.Context, mode, formats, layerList, tokenList, baselines string, maxGB float64, several bool, sp splitOpts, co cacheOpts, minRun time.Duration, work, out string) error {
	infos, err := hwnpu.Enumerate(ctx)
	if err != nil {
		return err
	}
	var info *hwnpu.DeviceInfo
	for i := range infos {
		if (infos[i].Backend == "coreml" || infos[i].Backend == "levelzero") && infos[i].Caps&hwnpu.CapGraph != 0 {
			info = &infos[i]
			break
		}
	}
	if info == nil {
		return errors.New("no NPU that runs models was found (Apple: macOS 14.4 or later on Apple Silicon; Intel: the NPU driver and Level Zero, see INSTALL_INTEL.md; run hwnpu-dev report for details)")
	}
	dev, err := hwnpu.OpenDevice(ctx, *info, hwnpu.WithPlacement(hwnpu.PlacementPrefer))
	if err != nil {
		return err
	}
	defer dev.Close()
	gr, err := dev.Graph()
	if err != nil {
		return err
	}

	rep := report{Date: time.Now().Format(time.RFC3339), Device: info.Name, Backend: info.Backend, Driver: info.Driver, Firmware: info.Firmware, Go: runtime.Version()}
	rep.CPU, rep.Model, rep.OS, rep.RAMBytes = machine()
	if bi, ok := debug.ReadBuildInfo(); ok {
		rep.Version = bi.Main.Version
		for _, s := range bi.Settings {
			if s.Key == "vcs.revision" {
				rep.Version += " " + s.Value
			}
		}
	}
	p := &prober{ctx: ctx, dev: dev, gr: gr, coreml: info.Backend == "coreml", minRun: minRun}

	benchFormats, limitFormats := split(formats), split(formats)
	if formats == "" {
		benchFormats, limitFormats = []string{"fp16", "int8", "int4"}, []string{"fp16", "int8", "int4"}
		if p.coreml {
			benchFormats, limitFormats = []string{"fp16", "int8", "lut4", "int4"}, []string{"fp16", "int8", "lut4"}
		}
	}
	for _, f := range benchFormats {
		if _, ok := probe.BytesPerParam[f]; !ok || (!p.coreml && f == "lut4") {
			return fmt.Errorf("format %q is not available on %s", f, info.Backend)
		}
	}
	limitFormats = slices.DeleteFunc(limitFormats, func(f string) bool { return p.coreml && f == "int4" })
	if mode == "verify" && len(limitFormats) == 0 {
		return errors.New("-formats: verify runs on the NPU only, and Core ML never places int4 on the Neural Engine")
	}
	splitFormats := split(formats)
	if formats == "" {
		splitFormats = []string{"int4"}
		if p.coreml {
			splitFormats = []string{"lut4"}
		}
	}
	var splitParts []int
	for _, s := range split(sp.parts) {
		n, err := strconv.Atoi(s)
		if err != nil || n < 1 || n > sp.layers {
			return fmt.Errorf("-parts: %q is not a number of parts of %d layers", s, sp.layers)
		}
		splitParts = append(splitParts, n)
	}
	var layouts []string
	for _, s := range split(sp.layout) {
		if s != "files" && s != "functions" || s == "functions" && !p.coreml {
			return fmt.Errorf("-layout: %q is not files or functions (functions is for Core ML)", s)
		}
		layouts = append(layouts, s)
	}
	switch {
	case sp.ref == "none":
		sp.ref = ""
	case sp.ref == "" && p.coreml:
		sp.ref = "gpu"
	case sp.ref != "" && (!p.coreml || sp.ref != "cpu" && sp.ref != "gpu"):
		return fmt.Errorf("-ref: %q is not available on %s", sp.ref, info.Backend)
	}
	var layers []int
	for _, s := range split(layerList) {
		n, err := strconv.Atoi(s)
		if err != nil || n < 1 {
			return fmt.Errorf("-layers: %q is not a layer count", s)
		}
		layers = append(layers, n)
	}
	var tokens []int
	for _, s := range split(tokenList) {
		n, err := strconv.Atoi(s)
		if err != nil || n < 1 {
			return fmt.Errorf("-tokens: %q is not a token count", s)
		}
		tokens = append(tokens, n)
	}
	var units []string
	if p.coreml {
		for _, u := range split(baselines) {
			if u != "cpu" && u != "gpu" {
				return fmt.Errorf("-baselines: %q is not cpu or gpu", u)
			}
			units = append(units, u)
		}
	}

	rep.MaxBytes = int64(maxGB * (1 << 30))
	if rep.MaxBytes == 0 {
		div := int64(6)
		if p.coreml {
			div = 4
		}
		rep.MaxBytes = min(rep.RAMBytes/div, 8<<30)
		if rep.MaxBytes == 0 {
			rep.MaxBytes = 2 << 30
		}
	}
	if work == "" {
		if work, err = os.MkdirTemp("", "hwnpu-npuprobe-"); err != nil {
			return err
		}
		defer os.RemoveAll(work)
	}
	p.dir = work
	if out == "" {
		name := rep.Model
		if name == "" {
			name = runtime.GOOS
		}
		name = strings.Map(func(r rune) rune {
			if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' {
				return r
			}
			return '-'
		}, name)
		out = fmt.Sprintf("npuprobe-%s-%s.md", name, time.Now().Format("20060102-1504"))
	}
	save := func() { write(out, &rep) } // keep what was measured if the run is cut short

	fmt.Printf("%s, %s, %s, RAM %.0f GiB\nNPU: %s (%s)\n", rep.CPU, rep.Model, rep.OS, probe.GiB(rep.RAMBytes), info.Name, info.Backend)
	if info.Caps&hwnpu.CapPowerMeter != 0 {
		if err := p.measureIdle(); err != nil {
			rep.Notes = append(rep.Notes, "power meter not readable ("+short(err.Error(), 120)+"); on Linux run as root for energy")
		} else {
			rep.IdleW = p.idleW
			if s, err := dev.PowerSample(); err == nil {
				rep.Meter = s.Source
			}
			fmt.Printf("power meter %s, idle %.2f W\n", rep.Meter, p.idleW)
		}
	}
	fmt.Println()

	if mode == "split" {
		p.splits(&rep, splitFormats, sp.layers, splitParts, layouts, sp.ref, save)
	}
	if mode == "kv" {
		if !p.coreml {
			return errors.New("kv: the KV cache probe models are Core ML only")
		}
		var caches []int
		for _, s := range split(sp.kvCache) {
			n, err := strconv.Atoi(s)
			if err != nil || n < 1 {
				return fmt.Errorf("-kv-cache: %q is not a token count", s)
			}
			caches = append(caches, n)
		}
		modes := split(sp.kvModes)
		for _, m := range modes {
			if b, _, _ := strings.Cut(m, "-"); b != "host" && b != "state" {
				return fmt.Errorf("-kv-modes: %q is not host or state, optionally -mha, -int8 or -chunk", m)
			}
		}
		p.kvs(&rep, sp.kvLayers, caches, modes, save)
	}
	if mode == "attn" {
		if !p.coreml {
			return errors.New("attn: the chunk attention probe models are Core ML only")
		}
		chunks, err := counts("-attn-chunks", sp.attnChunks)
		if err != nil {
			return err
		}
		for _, c := range chunks {
			if c > 16384 {
				return fmt.Errorf("-attn-chunks: %d is over the Neural Engine's dimension limit of 16384", c)
			}
		}
		tokens, err := counts("-attn-tokens", sp.attnTokens)
		if err != nil {
			return err
		}
		used, err := counts("-attn-used", sp.attnUsed)
		if err != nil {
			return err
		}
		if !slices.IsSorted(used) {
			return errors.New("-attn-used: the lengths go up")
		}
		forms := split(sp.attnForms)
		for _, f := range forms {
			if !slices.Contains(probe.AttnForms, f) {
				return fmt.Errorf("-attn-forms: %q is not grouped or tile", f)
			}
		}
		if sp.attnLayers < 1 {
			return errors.New("-attn-layers: at least 1")
		}
		p.attns(&rep, sp.attnLayers, chunks, tokens, forms, used, save)
	}
	if mode == "seam" {
		if !p.coreml {
			return errors.New("seam: the chunk attention probe models are Core ML only")
		}
		if _, ok := probe.BytesPerParam[sp.seamFormat]; !ok || sp.seamFormat == "int4" {
			return fmt.Errorf("-seam-format: %q is not a format the Neural Engine runs (fp16, int8, lut4)", sp.seamFormat)
		}
		if sp.seamLayers < sp.attnLayers+1 {
			return fmt.Errorf("-seam-layers: %d layers do not make %d parts", sp.seamLayers, sp.attnLayers+1)
		}
		if sp.seamChunk < 1 || sp.seamChunk > 16384 {
			return fmt.Errorf("-seam-chunk: %d is not 1 to 16384", sp.seamChunk)
		}
		if !slices.Contains(probe.AttnForms, sp.seamForm) {
			return fmt.Errorf("-seam-form: %q is not grouped or tile", sp.seamForm)
		}
		tokens, err := counts("-seam-tokens", sp.seamTokens)
		if err != nil {
			return err
		}
		used, err := counts("-seam-used", sp.seamUsed)
		if err != nil {
			return err
		}
		if !slices.IsSorted(used) {
			return errors.New("-seam-used: the lengths go up")
		}
		if sp.attnLayers < 1 {
			return errors.New("-attn-layers: at least 1")
		}
		p.seams(&rep, sp.seamFormat, sp.seamLayers, sp.attnLayers, tokens, sp.seamForm, sp.seamChunk, used, save)
	}
	if mode == "cache" {
		if !p.coreml {
			return errors.New("cache: the cache of compiled programs measured is the Neural Engine's (Core ML)")
		}
		// No other process compiling for the Neural Engine while the loads are timed: the
		// children open the device without the lock, under this process's.
		if err := llm.LockNPU(ctx); err != nil {
			return err
		}
		p.caches(&rep, co, save)
	}
	if mode == "verify" {
		// The formats the NPU runs: int4 never reaches the Neural Engine.
		p.verify(&rep, limitFormats, layers, tokens, save)
		fmt.Println()
	}
	if mode == "all" || mode == "bench" {
		p.bench(&rep, benchFormats, layers, units, save)
		fmt.Println()
	}
	if (mode == "all" || mode == "limit") && ctx.Err() == nil {
		fmt.Printf("limit: models up to %.1f GiB of weights\n", probe.GiB(rep.MaxBytes))
		p.limits(&rep, limitFormats, several, save)
	}
	if ctx.Err() != nil {
		rep.Notes = append(rep.Notes, "interrupted")
	}
	if err := write(out, &rep); err != nil {
		return err
	}
	fmt.Printf("\nreport: %s (and %s)\n", out, strings.TrimSuffix(out, ".md")+".json")
	return nil
}

// counts parses a comma separated list of positive counts.
func counts(flagName, s string) ([]int, error) {
	var out []int
	for _, f := range split(s) {
		n, err := strconv.Atoi(f)
		if err != nil || n < 1 {
			return nil, fmt.Errorf("%s: %q is not a positive count", flagName, f)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: empty", flagName)
	}
	return out, nil
}

func split(s string) []string {
	var out []string
	for _, f := range strings.Split(s, ",") {
		if f = strings.TrimSpace(f); f != "" {
			out = append(out, f)
		}
	}
	return out
}

func write(out string, rep *report) error {
	if err := os.WriteFile(out, []byte(markdown(rep)), 0o644); err != nil {
		return err
	}
	b, _ := json.MarshalIndent(rep, "", "  ")
	return os.WriteFile(strings.TrimSuffix(out, ".md")+".json", append(b, '\n'), 0o644)
}
