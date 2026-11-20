package iron

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// IsGraph reports whether path is a graph export (tools/amd/iron_graph.py: a directory with a
// manifest.json of kind graph).
func IsGraph(path string) bool {
	raw, err := os.ReadFile(filepath.Join(path, "manifest.json"))
	if err != nil {
		return false
	}
	var m struct {
		Kind string `json:"kind"`
	}
	return json.Unmarshal(raw, &m) == nil && m.Kind == "graph"
}

// GraphProgram is a graph export as a runner.Program (tools/amd/iron_lfm3speech.py's depth/ and
// codec/, iron_moss_sfx.py's, iron_magenta.py's): the export's meta "program" gives the shapes the converted program has, which the
// graph's pad: each input's rows go to the graph's rows (each repeated "repeat" times: the codec's
// upsampling), their values padded with zeros to the graph's columns; the output's rows and
// columns are the first of the graph's. Its meta "values" are the per-call values.
type GraphProgram struct {
	g       *Graph
	ins     map[string][]int
	repeat  map[string]int
	out     []int
	values  map[string]int64
	graphIn map[string][]int
	in      map[string][]byte
}

// programMeta is what a graph export's meta says of the program it stands for.
type programMeta struct {
	Values  map[string]int64 `json:"values"`
	Program struct {
		Inputs map[string]struct {
			Shape  []int `json:"shape"`
			Repeat int   `json:"repeat"`
		} `json:"inputs"`
		Output []int `json:"output"`
	} `json:"program"`
}

func OpenGraphProgram(ctx context.Context, dev *hwnpu.Device, dir string) (*GraphProgram, error) {
	g, err := OpenGraph(ctx, dev, dir)
	if err != nil {
		return nil, fmt.Errorf("iron: %s: %w", filepath.Base(dir), err)
	}
	fail := func(format string, args ...any) (*GraphProgram, error) {
		g.Close()
		return nil, fmt.Errorf("iron: %s: "+format, append([]any{dir}, args...)...)
	}
	var meta programMeta
	if len(g.Manifest.Meta) > 0 {
		if err := json.Unmarshal(g.Manifest.Meta, &meta); err != nil {
			return fail("meta: %w", err)
		}
	}
	p := &GraphProgram{g: g, ins: map[string][]int{}, repeat: map[string]int{}, graphIn: map[string][]int{}, in: map[string][]byte{}, values: meta.Values}
	for _, h := range g.Inputs() {
		if h.DType != "bfloat16" || len(h.Shape) == 0 {
			return fail("input %s is %s %v, not bfloat16", h.Name, h.DType, h.Shape)
		}
		p.graphIn[h.Name] = h.Shape
		p.ins[h.Name] = h.Shape
		p.repeat[h.Name] = 1
		if d, ok := meta.Program.Inputs[h.Name]; ok {
			p.ins[h.Name] = d.Shape
			p.repeat[h.Name] = max(1, d.Repeat)
		}
		p.in[h.Name] = make([]byte, h.Bytes)
	}
	outs := g.Outputs()
	if len(outs) != 1 || outs[0].DType != "bfloat16" {
		return fail("not one bfloat16 output")
	}
	p.out = outs[0].Shape
	if len(meta.Program.Output) > 0 {
		p.out = meta.Program.Output
	}
	return p, nil
}

func (p *GraphProgram) Inputs() map[string][]int    { return p.ins }
func (p *GraphProgram) Output() []int               { return p.out }
func (p *GraphProgram) Metadata() map[string]string { return nil }
func (p *GraphProgram) OnNPU() bool                 { return true }
func (p *GraphProgram) Close() error                { return p.g.Close() }

func (p *GraphProgram) Run(ctx context.Context, in map[string][]float32) ([]float32, error) {
	for name, shape := range p.ins {
		x, ok := in[name]
		if !ok {
			return nil, fmt.Errorf("iron: %s: no input %s", p.g.Manifest.Name, name)
		}
		gs := p.graphIn[name]
		cols, gcols := shape[len(shape)-1], gs[len(gs)-1]
		rep := p.repeat[name]
		b := p.in[name]
		clear(b)
		if len(gs) == 1 && rep == 1 {
			// a flat input (the DiT's mod): the values as they are
			cols = gcols
			if len(x) > gcols {
				return nil, fmt.Errorf("iron: %s: %d values for input %s of %d", p.g.Manifest.Name, len(x), name, gcols)
			}
			putBF16Row(b, x)
			if err := p.g.SetInput(name, b); err != nil {
				return nil, err
			}
			continue
		}
		if cols > gcols || len(x)%cols != 0 || len(x)/cols*rep*gcols*2 > len(b) {
			return nil, fmt.Errorf("iron: %s: %d values of rows of %d for input %s %v", p.g.Manifest.Name, len(x), cols, name, gs)
		}
		for r := 0; r < len(x)/cols; r++ {
			for k := 0; k < rep; k++ {
				putBF16Row(b[2*(r*rep+k)*gcols:], x[r*cols:(r+1)*cols])
			}
		}
		if err := p.g.SetInput(name, b); err != nil {
			return nil, err
		}
	}
	if err := p.g.Run(ctx, p.values); err != nil {
		return nil, fmt.Errorf("iron: %s: %w", p.g.Manifest.Name, err)
	}
	o := p.g.Outputs()[0]
	b, err := p.g.Output(o.Name)
	if err != nil {
		return nil, err
	}
	gcols, cols := o.Shape[len(o.Shape)-1], p.out[len(p.out)-1]
	n := 1
	for _, d := range p.out {
		n *= d
	}
	rows := n / cols
	if cols > gcols || rows*gcols*2 > len(b) {
		return nil, fmt.Errorf("iron: %s: output %v from the graph's %v", p.g.Manifest.Name, p.out, o.Shape)
	}
	y := make([]float32, n)
	for r := 0; r < rows; r++ {
		vecops.BF16ToF32(y[r*cols:(r+1)*cols], b[2*r*gcols:])
	}
	return y, nil
}

// putBF16Row writes xs as bf16 (to the nearest even) into b.
func putBF16Row(b []byte, xs []float32) {
	for i, x := range xs {
		binary.LittleEndian.PutUint16(b[2*i:], bf16Even(x))
	}
}

func bf16Even(x float32) uint16 {
	bits := math.Float32bits(x)
	if bits&0x7fffffff > 0x7f800000 {
		return uint16(bits>>16) | 0x40
	}
	return uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
}
