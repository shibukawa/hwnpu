// Command npuprobe measures the NPU of this machine with generated models, without Python, model
// files or root:
//
//	go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest          # bench, then limit
//	go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest bench    # speed only
//	go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest limit    # size limit only
//	go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest split    # a model over the limit in parts
//	go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest verify   # K tokens per call
//
// The models have the shape of LLM decoding (internal/probe): one token through Llama 3.2 1B
// sized layers with random weights, so every call reads every weight once, as generating one
// token does.
//
// bench runs 8 and 16 layers (0.5 B and 1 B weights) in each weight format and gives the time per
// token, the weight bandwidth and, where the device has a readable power meter (Intel and AMD
// on Linux as root), the energy per token. On Apple it also runs each model on the CPU and the GPU
// through Core ML; for energy there use cmd/powerbench -decode.
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
// Placement comes from Core ML's compute plan on Apple; on Intel the driver compiler either builds
// the whole graph for the NPU or fails. The report also gives the process CPU time per wall time
// while a model runs: near 0 when the NPU does the work, 1 or more when the CPU does.
//
// The report is written to npuprobe-<machine>-<time>.md and .json.
package main

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
	Sweep     []result      `json:"sweep,omitempty"`
	Several   []result      `json:"several,omitempty"`
	SeveralOf string        `json:"several_of,omitempty"`
	Split     []splitResult `json:"split,omitempty"`
	Notes     []string      `json:"notes,omitempty"`
}

func main() {
	if len(os.Args) == 3 && os.Args[1] == childMode {
		if err := runChild(os.Args[2]); err != nil {
			fmt.Fprintln(os.Stderr, "npuprobe:", err)
			os.Exit(1)
		}
		return
	}
	mode := "all"
	args := os.Args[1:]
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		mode, args = args[0], args[1:]
	}
	fs := flag.NewFlagSet("npuprobe", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: npuprobe [all|bench|limit|split|verify|kv] [flags]")
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
	minRun := fs.Duration("run", 3*time.Second, "time each model for at least this long")
	work := fs.String("dir", "", "directory for generated models (default: a temporary directory; Apple needs free space of about twice -max-gb)")
	out := fs.String("o", "", "report file (default npuprobe-<machine>-<time>.md, and .json beside it)")
	fs.Parse(args)
	if mode != "all" && mode != "bench" && mode != "limit" && mode != "split" && mode != "verify" && mode != "kv" {
		fs.Usage()
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	sp := splitOpts{layers: *splitLayers, parts: *parts, layout: *layout, ref: *ref, kvLayers: *kvLayers, kvCache: *kvCache, kvModes: *kvModes}
	if err := run(ctx, mode, *formats, *layers, *tokens, *baselines, *maxGB, *several, sp, *minRun, *work, *out); err != nil {
		fmt.Fprintln(os.Stderr, "npuprobe:", err)
		os.Exit(1)
	}
}

type splitOpts struct {
	layers             int
	parts, layout, ref string
	kvLayers           int
	kvCache, kvModes   string
}

func run(ctx context.Context, mode, formats, layerList, tokenList, baselines string, maxGB float64, several bool, sp splitOpts, minRun time.Duration, work, out string) error {
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
		return errors.New("no NPU that runs models was found (Apple: macOS 14.4 or later on Apple Silicon; Intel: the NPU driver and Level Zero, see INSTALL_INTEL.md; run cmd/report for details)")
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
