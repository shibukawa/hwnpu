package speech

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// HiddenBuffer is the state the export of tools/amd/iron_lfm3tts.py writes the text model's final
// hidden state to (after the final norm, before the head), in a prompt chunk and a decode step.
const HiddenBuffer = "hwnpu_hidden"

// ironText is the text model on AMD's NPU: an export of tools/amd/iron_lfm3tts.py (iron.Model),
// its token embeddings the host's, so that a decode step reads the row the host writes
// (iron.Model.StepRow: a frame's feedback).
type ironText struct {
	m      *iron.Model
	ctx    context.Context
	hidden []float32
	at     int
	row    []byte
}

func openIronText(ctx context.Context, dev *hwnpu.Device, dir string, hidden int, logf func(string, ...any)) (*ironText, error) {
	m, err := iron.Open(ctx, dev, dir, logf)
	if err != nil {
		return nil, fmt.Errorf("speech: text model: %w", err)
	}
	if _, ok := m.Buffers("decode")[HiddenBuffer]; !ok {
		m.Close()
		return nil, fmt.Errorf("speech: %s does not write its hidden state (%s): export it with iron_export_lm.py lfm3tts", dir, HiddenBuffer)
	}
	return &ironText{m: m, ctx: ctx, hidden: make([]float32, hidden), row: make([]byte, 2*hidden)}, nil
}

func (t *ironText) Reset()            { t.at = 0 }
func (t *ironText) Q() int            { return 1 }
func (t *ironText) Pos() int          { return t.at }
func (t *ironText) Cache() int        { return t.m.Manifest.Config.MaxSeqLen }
func (t *ironText) Hidden() []float32 { return t.hidden }
func (t *ironText) Close() error      { return t.m.Close() }

// ReadPrompt reads ids from position 0 in prompt chunks.
func (t *ironText) ReadPrompt(ctx context.Context, ids []int32) error {
	t.ctx = ctx
	if _, err := t.m.Prompt(ctx, ids, 0); err != nil {
		return fmt.Errorf("speech: text model: %w", err)
	}
	t.at = len(ids)
	return t.readHidden()
}

// Step reads each row (an embedding) in a decode step.
func (t *ironText) Step(rows [][]float32) ([]float32, error) {
	var logits []float32
	for _, r := range rows {
		if len(r) != len(t.hidden) {
			return nil, fmt.Errorf("speech: a row of %d for the text model's %d", len(r), len(t.hidden))
		}
		putBF16(t.row, r)
		var err error
		if logits, err = t.m.StepRow(t.ctx, t.row, t.at); err != nil {
			return nil, fmt.Errorf("speech: text model: %w", err)
		}
		t.at++
	}
	return logits, t.readHidden()
}

func (t *ironText) readHidden() error {
	b, _, err := t.m.Buffer("decode", HiddenBuffer)
	if err != nil {
		return err
	}
	if len(b) < 2*len(t.hidden) {
		return fmt.Errorf("speech: %s holds %d bytes, not a row of %d", HiddenBuffer, len(b), len(t.hidden))
	}
	vecops.BF16ToF32(t.hidden, b[:2*len(t.hidden)])
	return nil
}

// isGraph reports whether path is a graph export (tools/amd/iron_graph.py: a directory with a
// manifest.json of kind graph).
func isGraph(path string) bool {
	raw, err := os.ReadFile(filepath.Join(path, "manifest.json"))
	if err != nil {
		return false
	}
	var m struct {
		Kind string `json:"kind"`
	}
	return json.Unmarshal(raw, &m) == nil && m.Kind == "graph"
}

// graphProgram is a graph export as a runner.Program (tools/amd/iron_lfm3speech.py's depth/ and
// codec/): the export's meta "program" gives the shapes the converted program has, which the
// graph's pad: each input's rows go to the graph's rows (each repeated "repeat" times: the codec's
// upsampling), their values padded with zeros to the graph's columns; the output's rows and
// columns are the first of the graph's. Its meta "values" are the per-call values.
type graphProgram struct {
	g       *iron.Graph
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

func openGraphProgram(ctx context.Context, dev *hwnpu.Device, dir string) (*graphProgram, error) {
	g, err := iron.OpenGraph(ctx, dev, dir)
	if err != nil {
		return nil, fmt.Errorf("speech: %s: %w", filepath.Base(dir), err)
	}
	fail := func(format string, args ...any) (*graphProgram, error) {
		g.Close()
		return nil, fmt.Errorf("speech: %s: "+format, append([]any{dir}, args...)...)
	}
	var meta programMeta
	if len(g.Manifest.Meta) > 0 {
		if err := json.Unmarshal(g.Manifest.Meta, &meta); err != nil {
			return fail("meta: %w", err)
		}
	}
	p := &graphProgram{g: g, ins: map[string][]int{}, repeat: map[string]int{}, graphIn: map[string][]int{}, in: map[string][]byte{}, values: meta.Values}
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

func (p *graphProgram) Inputs() map[string][]int    { return p.ins }
func (p *graphProgram) Output() []int               { return p.out }
func (p *graphProgram) Metadata() map[string]string { return nil }
func (p *graphProgram) OnNPU() bool                 { return true }
func (p *graphProgram) Close() error                { return p.g.Close() }

func (p *graphProgram) Run(ctx context.Context, in map[string][]float32) ([]float32, error) {
	for name, shape := range p.ins {
		x, ok := in[name]
		if !ok {
			return nil, fmt.Errorf("speech: %s: no input %s", p.g.Manifest.Name, name)
		}
		gs := p.graphIn[name]
		cols, gcols := shape[len(shape)-1], gs[len(gs)-1]
		rep := p.repeat[name]
		b := p.in[name]
		clear(b)
		if cols > gcols || len(x)%cols != 0 || len(x)/cols*rep*gcols*2 > len(b) {
			return nil, fmt.Errorf("speech: %s: %d values of rows of %d for input %s %v", p.g.Manifest.Name, len(x), cols, name, gs)
		}
		for r := 0; r < len(x)/cols; r++ {
			for k := 0; k < rep; k++ {
				putBF16(b[2*(r*rep+k)*gcols:], x[r*cols:(r+1)*cols])
			}
		}
		if err := p.g.SetInput(name, b); err != nil {
			return nil, err
		}
	}
	if err := p.g.Run(ctx, p.values); err != nil {
		return nil, fmt.Errorf("speech: %s: %w", p.g.Manifest.Name, err)
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
		return nil, fmt.Errorf("speech: %s: output %v from the graph's %v", p.g.Manifest.Name, p.out, o.Shape)
	}
	y := make([]float32, n)
	for r := 0; r < rows; r++ {
		vecops.BF16ToF32(y[r*cols:(r+1)*cols], b[2*r*gcols:])
	}
	return y, nil
}

// putBF16 writes xs as bf16 (to the nearest even) into b.
func putBF16(b []byte, xs []float32) {
	for i, x := range xs {
		bits := math.Float32bits(x)
		var h uint16
		if bits&0x7fffffff > 0x7f800000 {
			h = uint16(bits>>16) | 0x40
		} else {
			h = uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
		}
		binary.LittleEndian.PutUint16(b[2*i:], h)
	}
}
