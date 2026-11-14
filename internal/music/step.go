package music

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/milcpu"
)

// step runs a program of named fp16 inputs and outputs, on a device or the CPU interpreter.
type step struct {
	run    func(in map[string][]float32) (map[string][]float32, error)
	onNPU  bool
	closer io.Closer
	name   string
	units  string // GraphSource.Units it was loaded on ("": the Neural Engine)
	// where says what of the program Core ML does not put on the NPU ("" when all is)
	where string
}

// openStep loads the program at path on dev (nil: the CPU interpreter).
func openStep(ctx context.Context, dev *hwnpu.Device, path string) (*step, error) {
	return openStepOn(ctx, dev, path, "")
}

// openStepOn is openStep on the Core ML units given ("gpu", "cpu"; "": the Neural Engine).
func openStepOn(ctx context.Context, dev *hwnpu.Device, path, units string) (*step, error) {
	if dev == nil {
		prog, f, err := milcpu.OpenPackage(path)
		if err != nil {
			return nil, err
		}
		shapes := prog.Inputs()
		return &step{closer: f, run: func(in map[string][]float32) (map[string][]float32, error) {
			feed := map[string]milcpu.Tensor{}
			for name, sh := range shapes {
				v, ok := in[name]
				if !ok {
					return nil, fmt.Errorf("music: no input %s", name)
				}
				feed[name] = milcpu.Tensor{Shape: sh, Data: v}
			}
			res, err := prog.Run(feed, milcpu.Options{})
			if err != nil {
				return nil, err
			}
			out := map[string][]float32{}
			for k, t := range res {
				out[k] = t.Data
			}
			return out, nil
		}}, nil
	}
	c, err := chain.OpenOn(ctx, dev, path, units)
	if err != nil {
		return nil, err
	}
	where := ""
	for _, part := range c.Parts() {
		pl := part.Placement
		switch {
		case !pl.Verified:
			where = "placement not known: " + pl.Reason
		case len(pl.Ops) > 0:
			kinds := map[string]int{}
			var order []string
			for _, op := range pl.Ops {
				t := op.Type
				if strings.HasPrefix(t, "ios") {
					if _, after, ok := strings.Cut(t, "."); ok {
						t = after
					}
				}
				k := t + " on the " + op.Device
				if kinds[k] == 0 {
					order = append(order, k)
				}
				kinds[k]++
			}
			sort.Slice(order, func(i, j int) bool { return kinds[order[i]] > kinds[order[j]] })
			var parts []string
			for _, k := range order {
				parts = append(parts, fmt.Sprintf("%d %s", kinds[k], k))
			}
			where = fmt.Sprintf("%d operations on the NPU, %d elsewhere: %s", pl.NPUOps, len(pl.Ops), strings.Join(parts, ", "))
		}
	}
	p := c.Program(context.Background())
	outs := map[string][]float32{}
	for name, b := range p.Out {
		outs[name] = make([]float32, len(b)/2)
	}
	return &step{closer: c, onNPU: c.OnNPU(), where: where, units: units, name: strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)), run: func(in map[string][]float32) (map[string][]float32, error) {
		for name, b := range p.In {
			v, ok := in[name]
			if !ok {
				return nil, fmt.Errorf("music: no input %s", name)
			}
			if len(v)*2 != len(b) {
				return nil, fmt.Errorf("music: input %s of %d values, want %d", name, len(v), len(b)/2)
			}
			hwnpu.EncodeF16(b, v)
		}
		if err := p.Run(); err != nil {
			return nil, err
		}
		for name, b := range p.Out {
			hwnpu.DecodeF16(outs[name], b)
		}
		return outs, nil
	}}, nil
}

func (s *step) Close() error { return s.closer.Close() }
