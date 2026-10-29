package main

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/probe"
)

// splitResult is one model split into parts that run one after another, the output buffer of
// each part bound as the input of the next.
type splitResult struct {
	Format      string    `json:"format"`
	Layers      int       `json:"layers"`
	WeightBytes int64     `json:"weight_bytes"`
	Parts       []result  `json:"parts"` // each part timed alone, with every part loaded
	OnNPU       bool      `json:"on_npu"`
	MsPerToken  float64   `json:"ms_per_token,omitempty"` // every part once
	MsSumParts  float64   `json:"ms_sum_parts,omitempty"` // the parts' own times added up
	CPUUse      float64   `json:"cpu_use,omitempty"`
	NetW        float64   `json:"net_w,omitempty"`
	MJPerToken  float64   `json:"mj_per_token,omitempty"`
	Ref         string    `json:"ref,omitempty"`     // units of the reference run
	RelErr      []float64 `json:"rel_err,omitempty"` // after each part: rms(y - reference) / rms(reference)
	Error       string    `json:"error,omitempty"`
	// y after each part, passed from the child processes and dropped from the report.
	Outputs [][]float32 `json:"outputs,omitempty"`
}

func (s splitResult) gbps() float64 {
	if s.MsPerToken == 0 {
		return 0
	}
	return float64(s.WeightBytes) / (s.MsPerToken / 1000) / 1e9
}

const splitSeed = 7

// split loads every part of a model of the given layers split n ways on the NPU, keeps all of
// them loaded, runs them once in order to record the outputs, then times each part alone and
// the whole chain.
func (p *prober) split(format string, layers, n int) splitResult {
	s := splitResult{Format: format, Layers: layers, WeightBytes: probe.WeightBytes(format, layers)}
	var models []*loaded
	defer func() {
		for _, l := range models {
			l.close()
		}
	}()
	var in *hwnpu.Buffer
	for i, part := range probe.Split(layers, n) {
		r := result{Copy: i + 1}
		l := p.loadPart(format, part, "npu", splitSeed, in, &r)
		r.OnNPU = r.Error == "" && r.Verified && r.NPUOps > 0 && r.OtherOps == 0
		s.Parts = append(s.Parts, r)
		if l == nil {
			s.Error = fmt.Sprintf("part %d: %s", i+1, r.Error)
			return s
		}
		models = append(models, l)
		in = l.out
	}
	s.OnNPU = true
	for _, r := range s.Parts {
		s.OnNPU = s.OnNPU && r.OnNPU
	}
	chain := func() error {
		for _, l := range models {
			if err := l.req.Run(p.ctx); err != nil {
				return err
			}
		}
		return nil
	}
	if err := chain(); err != nil {
		s.Error = "run: " + err.Error()
		return s
	}
	for _, l := range models {
		s.Outputs = append(s.Outputs, readF16(l.out))
	}
	for i, l := range models {
		if err := p.time(l, &s.Parts[i], false); err != nil {
			s.Error = fmt.Sprintf("part %d: run: %s", i+1, err)
			return s
		}
		s.MsSumParts += s.Parts[i].MsPerCall
	}
	var r result
	if err := p.measure(chain, &r, false); err != nil {
		s.Error = "run: " + err.Error()
		return s
	}
	s.MsPerToken, s.CPUUse, s.NetW, s.MJPerToken = r.MsPerCall, r.CPUUse, r.NetW, r.MJPerCall
	return s
}

// chain is the reference for split: the same parts on the given Core ML units, loaded one at a
// time so that only one part is in memory, each reading what the one before wrote.
func (p *prober) chain(format string, layers, n int, units string) splitResult {
	s := splitResult{Format: format, Layers: layers, WeightBytes: probe.WeightBytes(format, layers), Ref: units}
	carry, err := p.dev.AllocBuffer(2 * probe.DModel)
	if err != nil {
		s.Error = "buffer: " + err.Error()
		return s
	}
	defer carry.Close()
	fillInput(carry.Bytes(), hwnpu.F16)
	carry.SyncToDevice(0, carry.Size())
	for i, part := range probe.Split(layers, n) {
		var r result
		l := p.loadPart(format, part, units, splitSeed, carry, &r)
		if l == nil {
			s.Error = fmt.Sprintf("part %d: %s", i+1, r.Error)
			return s
		}
		err := l.req.Run(p.ctx)
		if err == nil {
			y := readF16(l.out)
			s.Outputs = append(s.Outputs, y)
			hwnpu.EncodeF16(carry.Bytes(), y)
			carry.SyncToDevice(0, carry.Size())
		}
		l.close()
		if err != nil {
			s.Error = fmt.Sprintf("part %d: run: %s", i+1, err)
			return s
		}
	}
	return s
}

func readF16(b *hwnpu.Buffer) []float32 {
	b.SyncFromDevice(0, b.Size())
	y := make([]float32, b.Size()/2)
	hwnpu.DecodeF16(y, b.Bytes())
	return y
}

// relErr is rms(y - ref) / rms(ref).
func relErr(y, ref []float32) float64 {
	var d, r float64
	for i := range ref {
		e := float64(y[i]) - float64(ref[i])
		d += e * e
		r += float64(ref[i]) * float64(ref[i])
	}
	if r == 0 {
		return math.Inf(1)
	}
	return math.Sqrt(d / r)
}

// splits runs each format split each way on the NPU and, when ref is set, compares the output
// after every part with the same parts run one at a time on ref.
func (p *prober) splits(rep *report, formats []string, layers int, parts []int, ref string, save func()) {
	for _, f := range formats {
		for _, n := range parts {
			if p.ctx.Err() != nil {
				return
			}
			fmt.Printf("split, %s %d layers (%.2f GiB) in %d parts:\n", f, layers, probe.GiB(probe.WeightBytes(f, layers)), n)
			var s splitResult
			if err := p.child(childSpec{Kind: "split", Format: f, Layers: layers, N: n}, &s); err != nil {
				s = splitResult{Format: f, Layers: layers, WeightBytes: probe.WeightBytes(f, layers), Error: err.Error()}
			}
			if ref != "" && s.Error == "" && p.ctx.Err() == nil {
				var c splitResult
				err := p.child(childSpec{Kind: "chain", Format: f, Layers: layers, N: n, Units: ref}, &c)
				switch {
				case err != nil:
					s.Error = "reference: " + err.Error()
				case c.Error != "":
					s.Error = "reference: " + c.Error
				default:
					s.Ref = ref
					for i := range min(len(s.Outputs), len(c.Outputs)) {
						s.RelErr = append(s.RelErr, relErr(s.Outputs[i], c.Outputs[i]))
					}
				}
			}
			s.Outputs = nil
			logSplit(s)
			rep.Split = append(rep.Split, s)
			save()
		}
	}
}

func logSplit(s splitResult) {
	for _, r := range s.Parts {
		logResult(r)
	}
	if s.MsPerToken > 0 {
		fmt.Printf("  all %d parts: %8.1f ms/token (parts alone add up to %.1f), %.0f GB/s, CPU use %.2f", len(s.Parts), s.MsPerToken, s.MsSumParts, s.gbps(), s.CPUUse)
		if s.MJPerToken != 0 {
			fmt.Printf(", %.0f mJ/token", s.MJPerToken)
		}
		fmt.Println()
	}
	if len(s.RelErr) > 0 {
		fmt.Printf("  error against %s after each part:", s.Ref)
		for _, e := range s.RelErr {
			fmt.Printf(" %.2f%%", 100*e)
		}
		fmt.Println()
	}
	if s.Error != "" {
		fmt.Println("  error:", short(s.Error, 200))
	}
}
