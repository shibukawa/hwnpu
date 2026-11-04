package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/probe"
)

// seamResult is a model split at its attention (docs/model-fit.md T2): weighted parts that run
// one after another, and between each two of them the chunk attention program over a cache,
// called once per chunk in use. It tells whether switching between a program with weights and
// one without costs the Neural Engine anything: alternating two programs of 0.81 GiB of weights
// each cost 35 ms a call (docs/design.md 16.36), a seam between two parts of one chain 0.1 ms
// (16.12).
type seamResult struct {
	Format      string    `json:"format"`
	Layers      int       `json:"layers"` // weighted layers, in AttnLayers+1 parts
	WeightBytes int64     `json:"weight_bytes"`
	Tokens      int       `json:"tokens"` // query rows per call, of the parts and the attention
	Form        string    `json:"form"`
	Chunk       int       `json:"chunk"`
	AttnLayers  int       `json:"attn_layers"`
	Parts       []result  `json:"parts"` // placement of each weighted part
	Attn        result    `json:"attn"`  // placement of the attention program
	Runs        []seamRun `json:"runs"`
	Error       string    `json:"error,omitempty"`
}

// seamRun is one used cache length: a step of the weighted parts alone, of the attention alone,
// and of the two interleaved, each the fastest of two rounds.
type seamRun struct {
	Used      int     `json:"used"`
	Chunks    int     `json:"chunks"` // per attention layer
	MsWeights float64 `json:"ms_weights,omitempty"`
	MsAttn    float64 `json:"ms_attn,omitempty"`
	MsMixed   float64 `json:"ms_mixed,omitempty"`
	Seams     int     `json:"seams"` // switches between a part and the attention per step
	CPUUse    float64 `json:"cpu_use,omitempty"`
	MaxRSS    int64   `json:"max_rss,omitempty"`
	Error     string  `json:"error,omitempty"`
}

// msPerSeam is what interleaving adds over the parts and the attention run apart, per switch.
func (r seamRun) msPerSeam() float64 {
	if r.MsMixed == 0 || r.Seams == 0 {
		return 0
	}
	return (r.MsMixed - r.MsWeights - r.MsAttn) / float64(r.Seams)
}

// seams measures the split model for each number of query rows.
func (p *prober) seams(rep *report, format string, layers, attnLayers int, tokens []int, form string, chunk int, used []int, save func()) {
	fmt.Printf("seam, %s %d layers (%.2f GiB) in %d parts with %s attention over chunks of %d between them:\n",
		format, layers, probe.GiB(probe.WeightBytes(format, layers)), attnLayers+1, form, chunk)
	for _, q := range tokens {
		if p.ctx.Err() != nil {
			return
		}
		var s seamResult
		spec := childSpec{Kind: "seam", Format: format, Layers: layers, AttnLayers: attnLayers, Tokens: q, N: chunk, Layout: form, Used: used}
		if err := p.child(spec, &s); err != nil {
			s = seamResult{Format: format, Layers: layers, WeightBytes: probe.WeightBytes(format, layers), Tokens: q, Form: form, Chunk: chunk, AttnLayers: attnLayers, Error: err.Error()}
		}
		fmt.Printf("  Q=%d: parts %s, attention %s%s\n", s.Tokens, partsPlacement(s.Parts), s.Attn.placement(), errSuffix(s.Error))
		for _, r := range s.Runs {
			fmt.Printf("      used %6d: weights %7.2f ms + attention %7.2f ms, interleaved %7.2f ms: %+.3f ms per seam (%d seams), peak RSS %.2f GiB%s\n",
				r.Used, r.MsWeights, r.MsAttn, r.MsMixed, r.msPerSeam(), r.Seams, probe.GiB(r.MaxRSS), errSuffix(r.Error))
		}
		rep.Seam = append(rep.Seam, s)
		save()
	}
}

// partsPlacement is NPU when every part is on it, else each part's placement.
func partsPlacement(parts []result) string {
	if len(parts) == 0 {
		return "none"
	}
	var off []string
	for i, r := range parts {
		if pl := r.placement(); pl != "NPU" {
			off = append(off, fmt.Sprintf("%d: %s", i+1, pl))
		}
	}
	if len(off) == 0 {
		return "NPU"
	}
	return fmt.Sprint(off)
}

// seam is the child side: load the weighted parts and the attention program, all at once, and
// time the three ways of running them for each used length.
func (p *prober) seam(format string, layers, attnLayers, tokens int, form string, chunk int, used []int) seamResult {
	s := seamResult{Format: format, Layers: layers, WeightBytes: probe.WeightBytes(format, layers), Tokens: tokens, Form: form, Chunk: chunk, AttnLayers: attnLayers}
	var models []*loaded
	defer func() {
		for _, l := range models {
			l.close()
		}
	}()
	var in *hwnpu.Buffer
	for i, part := range probe.Split(layers, attnLayers+1) {
		part.Tokens = tokens
		r := result{Copy: i + 1}
		l := p.loadPart(format, part, "npu", splitSeed, in, nil, &r)
		r.OnNPU = r.Error == "" && r.Verified && r.NPUOps > 0 && r.OtherOps == 0
		s.Parts = append(s.Parts, r)
		if l == nil {
			s.Error = fmt.Sprintf("part %d: %s", i+1, r.Error)
			return s
		}
		models = append(models, l)
		in = l.out
	}
	path := filepath.Join(p.dir, fmt.Sprintf("probe_seam_attn_%s_q%d_c%d.mlpackage", form, tokens, chunk))
	if err := probe.WriteChunkAttnPackage(path, form, tokens, chunk); err != nil {
		s.Error = "build: " + err.Error()
		return s
	}
	defer os.RemoveAll(path)
	al := p.loadPart("none", probe.Whole(0), "npu", splitSeed, nil, &sharedPart{path: path, fileBytes: probe.DirSize(path)}, &s.Attn)
	s.Attn.Tokens = tokens
	if al == nil {
		s.Error = "attention: " + s.Attn.Error
		return s
	}
	models = append(models, al)
	s.Attn.OnNPU = s.Attn.Error == "" && s.Attn.Verified && s.Attn.NPUOps > 0 && s.Attn.OtherOps == 0
	a, err := p.newAttnRunner(al, attnLayers, tokens, chunk)
	if err != nil {
		s.Error = err.Error()
		return s
	}
	defer a.close()
	parts := models[:attnLayers+1]

	weights := func() error {
		for _, l := range parts {
			if err := l.req.Run(p.ctx); err != nil {
				return err
			}
		}
		return nil
	}
	mixed := func() error {
		for i, l := range parts {
			if err := l.req.Run(p.ctx); err != nil {
				return err
			}
			if i < attnLayers {
				if err := a.layer(i); err != nil {
					return err
				}
			}
		}
		return nil
	}
	for _, u := range used {
		if p.ctx.Err() != nil {
			break
		}
		run := seamRun{Used: u, Chunks: a.chunks(u), Seams: 2 * attnLayers}
		if err := a.use(u); err != nil {
			run.Error = err.Error()
			s.Runs = append(s.Runs, run)
			break
		}
		// Two rounds of the three, the fastest of each kept: the order of the runs and the
		// machine's state between them weigh less.
		for round := 0; round < 2 && run.Error == ""; round++ {
			for _, m := range []struct {
				run func() error
				ms  *float64
			}{{weights, &run.MsWeights}, {a.step, &run.MsAttn}, {mixed, &run.MsMixed}} {
				var r result
				if err := p.measure(m.run, &r, false); err != nil {
					run.Error = "run: " + err.Error()
					break
				}
				if *m.ms == 0 || r.MsPerCall < *m.ms {
					*m.ms = r.MsPerCall
				}
				if m.ms == &run.MsMixed {
					run.CPUUse = r.CPUUse
				}
			}
		}
		run.MaxRSS = maxRSS()
		s.Runs = append(s.Runs, run)
		if run.Error != "" {
			break
		}
	}
	return s
}
