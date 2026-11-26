package llm

import (
	"fmt"
	"io"
	"math"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm/experts"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// AIProgram is one of the programs ConvertAI writes: the model built with Options (tokens per
// call, the last row alone, the caches alone...) into Out.parts.json, as the function Name of
// the Core AI programs it shares with the others, in Parts parts (0: as the target needs).
type AIProgram struct {
	Out     string // the manifest's path without .parts.json (OUT.parts.json)
	Name    string // the function's name in the programs it shares
	Parts   int
	Options Options
}

// ConvertAI writes programs of one model (the same weights, different Options) for Core AI, from
// the checkpoint: each split as Convert splits it, with one KV cache (no tiers), and the parts of
// the programs that begin at the same layer written as the functions of one Core AI program,
// base.partK.aimodel (WriteAIFunctions), so that their weights are on the disk, in Core AI's
// cache and on the Neural Engine once (LFM3-1.7B's programs of 8 and 128 tokens per call: 1.34
// GiB against 2.56 GiB apart, docs/design/apple-coreai.md). A program split at its attention
// (Options.Paged) gets its programs without weights of its own, OUT.attn.aimodel and
// OUT.mass.aimodel. Each program's OUT.parts.json lists its parts as PATH#NAME; the paths of the
// manifests are returned in order.
func ConvertAI(a *Arch, model Model, w Weights, base, targetName string, progs []AIProgram, log io.Writer) ([]string, error) {
	var target Target
	budget := int64(math.MaxInt64)
	if targetName != "none" {
		var err error
		if target, err = LookupTargetFor(BackendCoreAI, targetName); err != nil {
			return nil, err
		}
		if !target.Measured {
			fmt.Fprintf(log, "note: the %s limit (%.2f GiB) is not measured yet: %s\n", target.Name, float64(target.MaxBytes)/(1<<30), target.Note)
		}
		budget = int64(SplitMargin * float64(target.MaxBytes))
	}
	names := map[string]bool{}
	plans := make([][]Part, len(progs))
	for i := range progs {
		p := &progs[i]
		p.Options.Backend = BackendCoreAI
		o := p.Options
		if p.Name == "" || strings.ContainsAny(p.Name, "#/ ") || names[p.Name] {
			return nil, fmt.Errorf("llm: the program %s needs a name of its own without # / or spaces, not %q", p.Out, p.Name)
		}
		names[p.Name] = true
		if o.LastRow && (!a.LastRow || o.Cache <= 0) {
			return nil, fmt.Errorf("llm: %s cannot build the output layer for the last row alone (-last-row needs an architecture that has it and -cache)", a.Name)
		}
		if o.KVOnly && (o.Paged <= 0 || o.LastRow) {
			return nil, fmt.Errorf("llm: -kv-only takes -paged, and no -last-row (the program outputs no logits)")
		}
		if o.Start != 0 || o.Layers != 0 {
			return nil, fmt.Errorf("llm: ConvertAI converts the whole model")
		}
		var err error
		_, routed := model.(Routed)
		switch {
		case routed && o.Paged > 0:
			if _, ok := model.(Paged); !ok {
				return nil, fmt.Errorf("llm: %s cannot be split at its attention (-paged)", a.Title)
			}
			switch {
			case o.Cache <= 0:
				return nil, fmt.Errorf("llm: -paged needs -cache, the longest sequence")
			case o.Seq < 1 || o.Seq > o.Paged:
				return nil, fmt.Errorf("llm: -paged %d takes 1 to %d tokens per call, not %d", o.Paged, o.Paged, o.Seq)
			}
			plans[i], err = PlanRouted(model, o, budget)
		case routed:
			plans[i], err = PlanRouted(model, o, budget)
		case o.Paged > 0:
			if _, ok := model.(Paged); !ok {
				return nil, fmt.Errorf("llm: %s cannot be split at its attention (-paged)", a.Title)
			}
			switch {
			case o.Cache <= 0:
				return nil, fmt.Errorf("llm: -paged needs -cache, the longest sequence")
			case o.Seq < 1 || o.Seq > o.Paged:
				return nil, fmt.Errorf("llm: -paged %d takes 1 to %d tokens per call, not %d", o.Paged, o.Paged, o.Seq)
			case o.Paged > NeuralEngineMaxDim:
				return nil, fmt.Errorf("llm: chunks of %d are over the Neural Engine's %d per dimension", o.Paged, NeuralEngineMaxDim)
			}
			plans[i], err = PlanPaged(model, o, budget)
		case targetName != "none" || p.Parts > 1:
			plans[i], err = PlanSplit(model, o, budget, p.Parts)
		default:
			plans[i] = []Part{{}}
		}
		if err != nil {
			return nil, err
		}
		if o.Cache > 0 && o.Paged == 0 {
			lens, chunked := CacheChunks(o.Cache)
			if !chunked && o.Cache+o.Seq > NeuralEngineMaxDim {
				fmt.Fprintf(log, "warning: attention over a cache of %d and %d tokens per call spans %d keys, over the Neural Engine's %d per dimension; it will run on the CPU\n", o.Cache, o.Seq, o.Cache+o.Seq, NeuralEngineMaxDim)
			}
			if chunked {
				fmt.Fprintf(log, "note: the cache of %d is held in %d chunks of up to %d tokens\n", o.Cache, len(lens), CacheChunkLen)
			}
		}
	}
	// the parts by the layer they begin at: a program's parts of the same first layer have the
	// same weights, or (the program that writes the caches alone) the first of them
	// (a part of no layers, NoLayers, is a group of its own). Whole programs (their caches
	// inside) beside programs split at their attention have groups of their own, by cache: in one
	// Core AI program with the whole programs of a cache of 256, Qwen3-0.6B's of 1024 ran at half
	// the speed it has alone (docs/design/apple-coreai.md)
	type member struct{ prog, part int }
	type key struct {
		start int
		some  bool // the part has layers
		whole int  // the cache of a program not split at its attention, beside one that is
	}
	var starts []key
	groups := map[key][]member{}
	var paged bool
	for _, p := range progs {
		paged = paged || p.Options.Paged > 0
	}
	for i, plan := range plans {
		for j, p := range plan {
			k := key{p.Start, p.Layers != NoLayers, 0}
			if paged && progs[i].Options.Paged == 0 {
				k.whole = progs[i].Options.Cache
			}
			if _, ok := groups[k]; !ok {
				starts = append(starts, k)
			}
			groups[k] = append(groups[k], member{i, j})
		}
	}
	slices.SortFunc(starts, func(x, y key) int {
		if x.whole != y.whole {
			return y.whole - x.whole // the whole programs' first, longest cache first
		}
		if x.start != y.start {
			return x.start - y.start
		}
		if x.some == y.some {
			return 0
		}
		if y.some {
			return -1
		}
		return 1
	})
	recs := make([][]PartRecord, len(progs))
	for i := range progs {
		recs[i] = make([]PartRecord, len(plans[i]))
	}
	// the programs are written side by side (console.Jobs), their lines erased when all are done
	jobs := console.NewJobs(log)
	for k, s := range starts {
		path := fmt.Sprintf("%s.part%d.aimodel", base, k+1)
		if len(starts) == 1 {
			path = base + ".aimodel"
		}
		var fns []AIFunction
		for _, m := range groups[s] {
			o := progs[m.prog].Options
			if len(plans[m.prog]) > 1 || o.Paged > 0 {
				o.Start, o.Layers = plans[m.prog][m.part].Start, plans[m.prog][m.part].Layers
			}
			fns = append(fns, AIFunction{Name: progs[m.prog].Name, Model: model, Options: o})
		}
		jobs.Add(filepath.Base(path), func() (string, error) {
			rs, err := WriteAIFunctions(path, w, fns)
			if err != nil {
				return "", err
			}
			for i, m := range groups[s] {
				recs[m.prog][m.part] = rs[i]
			}
			return fmt.Sprintf("%d functions, %.2f GiB", len(fns), float64(rs[0].WeightBytes)/(1<<30)), nil
		})
	}
	// a job holds its program's weights (up to the target's limit, a few GiB) and the tensors it reads
	if err := jobs.Run(4 << 30); err != nil {
		return nil, err
	}
	warned := map[string]bool{}
	for _, r := range recs {
		for _, rec := range r {
			path, _, _ := strings.Cut(rec.Path, "#") // a function's record: PATH#NAME
			if target.MaxBytes > 0 && rec.WeightBytes >= target.MaxBytes && !warned[path] {
				warned[path] = true
				fmt.Fprintf(log, "warning: %s: weights of %.3f GiB are over the %s limit of %.3f GiB, so the NPU may not take it\n", path, float64(rec.WeightBytes)/(1<<30), target.Name, float64(target.MaxBytes)/(1<<30))
			}
		}
	}
	var ex *ExpertsRecord
	if hm, ok := model.(HostRouted); ok {
		cp, ok := w.(CheckpointPath)
		if !ok {
			return nil, fmt.Errorf("llm: %s's experts run from the checkpoint, and %s is not a file", a.Title, w.Source())
		}
		path, err := filepath.Abs(cp.Path())
		if err != nil {
			return nil, err
		}
		rec := &experts.HostRecord{Checkpoint: path, Layers: map[int]experts.MXFP4{}, Act: hm.ExpertAct()}
		for _, l := range hm.RoutedLayers() {
			rec.Layers[l] = hm.HostExperts(l)
		}
		if ls, ok := w.(lister); ok {
			if n := rec.Layers[hm.RoutedLayers()[0]].GateUpBlocks; n != "" {
				if _, has := ls.Shape(n); !has {
					hint := ""
					if _, ok := model.(HostDerived); ok {
						hint = " (its release's experts are not MXFP4: write a checkpoint with them in MXFP4 with hwnpu-dev mxfp4, and convert from that)"
					}
					return nil, fmt.Errorf("llm: %s's experts run from the checkpoint's MXFP4 tensors, and %s has no %s%s", a.Title, path, n, hint)
				}
			}
		}
		ex = &ExpertsRecord{Layers: hm.RoutedLayers(), Shape: hm.ExpertShape(), Routing: hm.Routing(), Host: rec}
		fmt.Fprintf(log, "%s: the experts of %d layers run on the CPU from %s\n", a.Title, len(rec.Layers), path)
	} else if rm, ok := model.(Routed); ok {
		var err error
		var seqs []int
		for _, p := range progs {
			if q := p.Options.Seq; q > 1 && !slices.Contains(seqs, q) {
				seqs = append(seqs, q)
			}
		}
		if ex, err = writeAIExperts(rm, w, base, a.Title, seqs, log); err != nil {
			return nil, err
		}
	}
	var out []string
	for i, p := range progs {
		o := p.Options
		man := Manifest{Model: a.Title, Source: w.Source(), Format: o.Format, Seq: o.Seq, Cache: o.Cache, Target: target.Name, MaxBytes: target.MaxBytes,
			Backend: BackendCoreAI, Meta: metadata(a, model, w, o), Paged: o.Paged, KVOnly: o.KVOnly}
		man.Parts = recs[i]
		if ex != nil {
			e := *ex
			man.Experts = &e
		}
		if o.Paged > 0 {
			if err := writeAIPaged(&man, a, model.(Paged), p.Out, o); err != nil {
				return nil, err
			}
		}
		if err := WriteManifest(p.Out+".parts.json", &man); err != nil {
			return nil, err
		}
		out = append(out, p.Out+".parts.json")
	}
	if len(progs) > 1 {
		fmt.Fprintf(log, "%s: %d programs in %d Core AI programs, the parts of the same layers as functions of one\n", filepath.Base(base), len(progs), len(starts))
	}
	return out, nil
}

// writeAIPaged writes the programs without weights of a model split at its attention for Core AI,
// out.attn.aimodel and out.mass.aimodel, into man.
func writeAIPaged(man *Manifest, a *Arch, pm Paged, out string, o Options) error {
	form := "grouped"
	sh := pm.PagedShape()
	arec, err := writeAttention(out+".attn", BackendCoreAI, sh, form, o.Seq, o.Paged, a.Title)
	if err != nil {
		return err
	}
	man.Attention, man.AttnVersion, man.AttnOutScale = arec, paged.Version, sh.OutScale
	if af := attentionForm(BackendCoreAI, form, sh); af != form {
		man.AttnForm = af
	}
	var ins, outs []mlpackage.Feature
	mpath := out + ".mass.aimodel"
	_, ins, outs, err = writeAI(mpath, map[string]string{MetaDescription: fmt.Sprintf("%s weights of a chunk of %d bucket summaries, %d per call, %s (hwnpu-dev convert)", a.Title, o.Paged, o.Seq, form)},
		func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
			return paged.BuildMass(b, sh, form, o.Seq, o.Paged)
		})
	if err != nil {
		return err
	}
	man.Mass = &PartRecord{Path: mpath, Inputs: records(ins...), Output: records(outs[0])[0]}
	return nil
}

// writeAIExperts writes the experts of a Routed model for Core AI: the program without weights of
// a row's chosen experts, base.experts.aimodel, and every layer's experts in int8,
// base.experts.safetensors (experts.WriteFile).
func writeAIExperts(rm Routed, w Weights, base, title string, seqs []int, log io.Writer) (*ExpertsRecord, error) {
	sh := rm.ExpertShape()
	ppath := base + ".experts.aimodel"
	desc := fmt.Sprintf("%s: %d of %d experts of %d x %d on a row, their weights given at run time (hwnpu-dev convert)", title, sh.TopK, sh.Experts, sh.Hidden, sh.Inter)
	_, ins, outs, err := writeAI(ppath, map[string]string{MetaDescription: desc}, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
		return experts.Build(b, sh)
	})
	if err != nil {
		return nil, err
	}
	var rows []PartRecord
	for _, q := range seqs {
		rpath := fmt.Sprintf("%s.experts.q%d.aimodel", base, q)
		desc := fmt.Sprintf("%s: an expert of %d x %d on %d rows, its weights given at run time (hwnpu-dev convert)", title, sh.Hidden, sh.Inter, q)
		_, ins, outs, err := writeAI(rpath, map[string]string{MetaDescription: desc}, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
			return experts.BuildRows(b, sh, q)
		})
		if err != nil {
			return nil, err
		}
		rows = append(rows, PartRecord{Path: rpath, Inputs: records(ins...), Output: records(outs[0])[0]})
	}
	wpath := base + ".experts.safetensors"
	layers := rm.RoutedLayers()
	stop := console.Spin(log, filepath.Base(wpath))
	err = experts.WriteFile(wpath, sh, layers, rm.ExpertNames, func(name string) ([]float32, error) {
		v, _, err := w.Get(name)
		return v, err
	})
	stop()
	if err != nil {
		io.WriteString(log, "\n")
		return nil, err
	}
	fmt.Fprintf(log, ": the experts of %d layers in int8, %.2f GiB\n", len(layers), float64(len(layers)*sh.Experts*3*sh.Hidden*sh.Inter)/(1<<30))
	return &ExpertsRecord{Program: PartRecord{Path: ppath, Inputs: records(ins...), Output: records(outs[0])[0]}, Rows: rows, Weights: wpath, Layers: layers, Shape: sh, Routing: rm.Routing()}, nil
}
