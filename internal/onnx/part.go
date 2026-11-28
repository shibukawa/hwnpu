package onnx

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"

	"github.com/shibukawa/hwnpu/internal/mapped"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// A part file (PartExt) holds what a model's run reads of its ONNX file beside the program
// converted from it, as it is in memory, so that loading it is mapping it (as the tokenizer's
// binary file, internal/tokenizer): the part of the graph the host evaluates (Graph.Subgraph:
// pose's decoding after the program, notes' CQT before it, segment's whole decoder), the tensors
// it reads alone (pitch's filter banks, the values the program folds) and the model's settings
// (meta: JSON of the model's own). A machine that takes the program and its part file from the
// sources needs neither the ONNX file nor its parsing.
//
//	"HWNPUONX"  u32 version  u32 n  n bytes of JSON: the graph (its nodes, inputs, outputs,
//	            opset and initializers), the other tensors, meta; a tensor is its type, shape
//	            and where its bytes are
//	the tensors' bytes, each from a multiple of 8: little endian, as TensorProto.raw_data
//
// The weights (tensors of lazyMin elements or more) are read lazily (Tensor.Lazy) from the mapped
// file, which stays mapped while one of them is; the others are read at once, as LoadLazy does. A file of another version is not read (ReadPart's error: the ONNX file is taken then).
const PartExt = ".onnxpart"

// PartVersion is the version of the part file's layout.
const PartVersion = 1

const partMagic = "HWNPUONX"

// Part is what a part file holds.
type Part struct {
	// Graph is the part of the graph the host evaluates (nil for none): its initializers are
	// those of its nodes alone.
	Graph *Graph
	// Tensors are tensors read by name beside the graph's.
	Tensors map[string]*Tensor
	// Meta is the model's settings, JSON of the model's own.
	Meta json.RawMessage
}

// Subgraph is the part of g that computes outputs from given (values the host has: a program's
// outputs, the graph's inputs): the nodes needed, in order, the initializers they read, given's
// values as its inputs and outputs as its outputs. It shares g's tensors.
func (g *Graph) Subgraph(given, outputs []string) *Graph {
	stop := map[string]bool{}
	for _, v := range given {
		stop[v] = true
	}
	s := &Graph{Init: map[string]*Tensor{}, Opset: g.Opset}
	var add func(nodes []*Node)
	add = func(nodes []*Node) {
		for _, n := range nodes {
			for _, in := range n.In {
				if t := g.Init[in]; t != nil {
					s.Init[in] = t
				}
			}
			for _, sub := range n.Graphs {
				add(sub.Nodes) // the outer values a branch reads are among n.In (If, Loop)
			}
		}
	}
	s.Nodes = g.needed(outputs, stop, nil)
	add(s.Nodes)
	for _, v := range given {
		vi := ValueInfo{Name: v}
		if i := slices.IndexFunc(g.Inputs, func(x ValueInfo) bool { return x.Name == v }); i >= 0 {
			vi = g.Inputs[i]
		}
		s.Inputs = append(s.Inputs, vi)
	}
	for _, o := range outputs {
		vi := ValueInfo{Name: o}
		if i := slices.IndexFunc(g.Outputs, func(x ValueInfo) bool { return x.Name == o }); i >= 0 {
			vi = g.Outputs[i]
		}
		s.Outputs = append(s.Outputs, vi)
	}
	return s
}

// the part file's JSON
type (
	partJSON struct {
		Graph   *graphJSON           `json:"graph,omitempty"`
		Tensors map[string]tensorRef `json:"tensors,omitempty"`
		Meta    json.RawMessage      `json:"meta,omitempty"`
	}
	graphJSON struct {
		Nodes   []nodeJSON           `json:"nodes"`
		Init    map[string]tensorRef `json:"init,omitempty"`
		Inputs  []ValueInfo          `json:"inputs,omitempty"`
		Outputs []ValueInfo          `json:"outputs,omitempty"`
		Opset   int                  `json:"opset,omitempty"`
	}
	nodeJSON struct {
		Op     string              `json:"op"`
		Name   string              `json:"name,omitempty"`
		Domain string              `json:"domain,omitempty"`
		In     []string            `json:"in"`
		Out    []string            `json:"out"`
		Attrs  map[string]attrJSON `json:"attrs,omitempty"`
		Graphs []*graphJSON        `json:"graphs,omitempty"`
	}
	attrJSON struct {
		Kind   int        `json:"kind"`
		F      float64    `json:"f,omitempty"`
		I      int64      `json:"i,omitempty"`
		S      string     `json:"s,omitempty"`
		T      *tensorRef `json:"t,omitempty"`
		Floats []float64  `json:"floats,omitempty"`
		Ints   []int64    `json:"ints,omitempty"`
		G      *graphJSON `json:"g,omitempty"`
	}
	tensorRef struct {
		Type  int   `json:"type"`
		Shape []int `json:"shape"`
		Off   int   `json:"off"`
		Size  int   `json:"size"`
	}
)

// partWriter lays out the tensors' bytes after the header, each from a multiple of 8.
type partWriter struct {
	data [][]byte
	at   int // the offset in the data section of the next tensor
}

func (w *partWriter) tensor(t *Tensor) (tensorRef, error) {
	raw, typ, err := rawOf(t)
	if err != nil {
		return tensorRef{}, err
	}
	r := tensorRef{Type: typ, Shape: slices.Clone(t.Shape), Off: w.at, Size: len(raw)}
	if r.Shape == nil {
		r.Shape = []int{}
	}
	w.data = append(w.data, raw)
	w.at = align8(w.at + len(raw))
	return r, nil
}

func (w *partWriter) graph(g *Graph) (*graphJSON, error) {
	out := &graphJSON{Inputs: g.Inputs, Outputs: g.Outputs, Opset: g.Opset}
	names := make([]string, 0, len(g.Init))
	for n := range g.Init {
		names = append(names, n)
	}
	slices.Sort(names) // the same bytes for the same graph
	if len(names) > 0 {
		out.Init = map[string]tensorRef{}
	}
	for _, n := range names {
		r, err := w.tensor(g.Init[n])
		if err != nil {
			return nil, fmt.Errorf("%s: %w", n, err)
		}
		out.Init[n] = r
	}
	for _, n := range g.Nodes {
		nj := nodeJSON{Op: n.Op, Name: n.Name, Domain: n.Domain, In: n.In, Out: n.Out}
		if len(n.Attrs) > 0 {
			nj.Attrs = map[string]attrJSON{}
		}
		for k, a := range n.Attrs {
			aj := attrJSON{Kind: a.Kind, F: a.F, I: a.I, S: a.S, Floats: a.Floats, Ints: a.Ints}
			if !finite(a.F) || slices.ContainsFunc(a.Floats, func(v float64) bool { return !finite(v) }) {
				return nil, fmt.Errorf("node %s: attribute %s is not finite (no JSON for it)", n.Name, k)
			}
			if a.T != nil {
				r, err := w.tensor(a.T)
				if err != nil {
					return nil, fmt.Errorf("node %s: attribute %s: %w", n.Name, k, err)
				}
				aj.T = &r
			}
			if a.G != nil {
				gj, err := w.graph(a.G)
				if err != nil {
					return nil, err
				}
				aj.G = gj
			}
			nj.Attrs[k] = aj
		}
		for _, sub := range n.Graphs {
			gj, err := w.graph(sub)
			if err != nil {
				return nil, err
			}
			nj.Graphs = append(nj.Graphs, gj)
		}
		out.Nodes = append(out.Nodes, nj)
	}
	return out, nil
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// WritePart writes p to path (PartExt), through a temporary file beside it.
func WritePart(path string, p Part) error {
	w := &partWriter{}
	var pj partJSON
	if p.Graph != nil {
		gj, err := w.graph(p.Graph)
		if err != nil {
			return fmt.Errorf("onnx: %s: %w", filepath.Base(path), err)
		}
		pj.Graph = gj
	}
	names := make([]string, 0, len(p.Tensors))
	for n := range p.Tensors {
		names = append(names, n)
	}
	slices.Sort(names)
	if len(names) > 0 {
		pj.Tensors = map[string]tensorRef{}
	}
	for _, n := range names {
		r, err := w.tensor(p.Tensors[n])
		if err != nil {
			return fmt.Errorf("onnx: %s: %s: %w", filepath.Base(path), n, err)
		}
		pj.Tensors[n] = r
	}
	pj.Meta = p.Meta
	head, err := json.Marshal(pj)
	if err != nil {
		return fmt.Errorf("onnx: %s: %w", filepath.Base(path), err)
	}
	var buf bytes.Buffer
	buf.WriteString(partMagic)
	binary.Write(&buf, binary.LittleEndian, uint32(PartVersion))
	binary.Write(&buf, binary.LittleEndian, uint32(len(head)))
	buf.Write(head)
	buf.Write(make([]byte, align8(buf.Len())-buf.Len()))
	base := buf.Len()
	for _, d := range w.data {
		buf.Write(make([]byte, base+align8(buf.Len()-base)-buf.Len()))
		buf.Write(d)
	}
	// the tensors' bytes may be in a mapped file (LoadLazy's), unmapped once they are gone
	runtime.KeepAlive(p)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, buf.Bytes(), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ReadPart maps the part file at path. Its tensors are lazy, read from the mapping, which is
// kept while one of them is.
func ReadPart(path string) (*Part, error) {
	f, err := mapped.Open(path)
	if err != nil {
		return nil, err
	}
	b := f.Bytes()
	if len(b) < 16 || string(b[:8]) != partMagic {
		f.Close()
		return nil, fmt.Errorf("onnx: %s is no part file", path)
	}
	if v := binary.LittleEndian.Uint32(b[8:]); v != PartVersion {
		f.Close()
		return nil, fmt.Errorf("onnx: %s is a part file of version %d, not %d", path, v, PartVersion)
	}
	n := int(binary.LittleEndian.Uint32(b[12:]))
	if 16+n > len(b) {
		f.Close()
		return nil, fmt.Errorf("onnx: %s is cut short", path)
	}
	var pj partJSON
	if err := json.Unmarshal(b[16:16+n], &pj); err != nil {
		f.Close()
		return nil, fmt.Errorf("onnx: %s: %w", path, err)
	}
	r := &partReader{data: b[align8(16+n):], keep: f}
	p := &Part{Meta: pj.Meta}
	if pj.Graph != nil {
		if p.Graph, err = r.graph(pj.Graph); err != nil {
			return nil, fmt.Errorf("onnx: %s: %w", path, err)
		}
	}
	if len(pj.Tensors) > 0 {
		p.Tensors = map[string]*Tensor{}
	}
	for name, ref := range pj.Tensors {
		if p.Tensors[name], err = r.tensor(ref); err != nil {
			return nil, fmt.Errorf("onnx: %s: %s: %w", path, name, err)
		}
	}
	// the mapping goes with the last lazy tensor (none: now)
	if r.lazy == 0 {
		f.Close()
	}
	return p, nil
}

type partReader struct {
	data []byte
	keep *mapped.File
	lazy int
}

func (r *partReader) tensor(ref tensorRef) (*Tensor, error) {
	if ref.Off < 0 || ref.Off+ref.Size > len(r.data) {
		return nil, errors.New("a tensor's bytes are past the file's end")
	}
	shape := ref.Shape
	if len(shape) == 0 {
		shape = nil
	}
	if ref.Size == 0 {
		return &Tensor{Shape: shape, Type: ref.Type, F: []float64{}}, nil
	}
	t, err := NewLazy(shape, ref.Type, r.data[ref.Off:ref.Off+ref.Size], r.keep)
	if err != nil {
		return nil, err
	}
	// as LoadLazy: the weights stay in the mapped file, shapes, axes and scales are read
	if Len(shape) < lazyMin {
		t.Decode()
	} else {
		r.lazy++
	}
	return t, nil
}

func (r *partReader) graph(gj *graphJSON) (*Graph, error) {
	g := &Graph{Init: map[string]*Tensor{}, Inputs: gj.Inputs, Outputs: gj.Outputs, Opset: gj.Opset}
	for name, ref := range gj.Init {
		t, err := r.tensor(ref)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		g.Init[name] = t
	}
	for _, nj := range gj.Nodes {
		n := &Node{Op: nj.Op, Name: nj.Name, Domain: nj.Domain, In: nj.In, Out: nj.Out}
		if len(nj.Attrs) > 0 {
			n.Attrs = map[string]Attr{}
		}
		for k, aj := range nj.Attrs {
			a := Attr{Kind: aj.Kind, F: aj.F, I: aj.I, S: aj.S, Floats: aj.Floats, Ints: aj.Ints}
			if aj.T != nil {
				t, err := r.tensor(*aj.T)
				if err != nil {
					return nil, err
				}
				t.Decode() // the evaluator reads an attribute's values as they are (Constant's)
				a.T = t
			}
			if aj.G != nil {
				sub, err := r.graph(aj.G)
				if err != nil {
					return nil, err
				}
				a.G = sub
			}
			n.Attrs[k] = a
		}
		for _, gj := range nj.Graphs {
			sub, err := r.graph(gj)
			if err != nil {
				return nil, err
			}
			n.Graphs = append(n.Graphs, sub)
		}
		g.Nodes = append(g.Nodes, n)
	}
	return g, nil
}

// rawOf are the tensor's bytes as TensorProto.raw_data has them, and their type: a lazy tensor's
// as they were read, else its values written in its type.
func rawOf(t *Tensor) ([]byte, int, error) {
	if t.Lazy() {
		return t.raw, t.Type, nil
	}
	typ := t.Type
	if typ == 0 {
		typ = Float
	}
	size := map[int]int{Float: 4, Double: 8, Int64: 8, Int32: 4, Bool: 1, Uint8: 1, Int8: 1, Float16: 2}[typ]
	if size == 0 {
		return nil, 0, fmt.Errorf("no part file bytes for tensors of type %d", typ)
	}
	out := make([]byte, size*len(t.F))
	for i, v := range t.F {
		p := out[i*size:]
		switch typ {
		case Float:
			binary.LittleEndian.PutUint32(p, math.Float32bits(float32(v)))
		case Double:
			binary.LittleEndian.PutUint64(p, math.Float64bits(v))
		case Int64:
			binary.LittleEndian.PutUint64(p, uint64(int64(v)))
		case Int32:
			binary.LittleEndian.PutUint32(p, uint32(int32(v)))
		case Bool, Uint8:
			p[0] = byte(v)
		case Int8:
			p[0] = byte(int8(v))
		case Float16:
			binary.LittleEndian.PutUint16(p, vecops.F16Bits(float32(v)))
		}
	}
	return out, typ, nil
}

func align8(n int) int { return (n + 7) &^ 7 }

// programLayout is a built Program's layout of its outputs as a part file keeps it (ProgramPart):
// what Split, Widths, Folded and Shapes read.
type programLayout struct {
	Outputs   []string         `json:"outputs"`
	Axis      int              `json:"axis,omitempty"`
	Flat      bool             `json:"flat,omitempty"`
	Rows      int              `json:"rows,omitempty"`
	Widths    []int            `json:"widths,omitempty"`
	OutShapes [][]int          `json:"out_shapes,omitempty"`
	Shapes    map[string][]int `json:"shapes,omitempty"`
	Folded    []string         `json:"folded,omitempty"`
}

// ProgramPart is the part file of the built program p (after Build, as llm.WritePart builds it):
// the part of its graph the host evaluates from given to outputs (none when outputs is empty),
// the values p folds (Folded, among the tensors) and p's layout of its outputs (meta's
// "program"), what ProgramOf makes p again from. meta's other keys are the model's own.
func (p *Program) ProgramPart(given, outputs []string, meta map[string]any) (Part, error) {
	if p.outShapes == nil && p.widths == nil {
		return Part{}, errors.New("onnx: the program's part before it is built")
	}
	l := programLayout{Outputs: p.Outputs, Axis: p.Axis, Flat: p.Flat, Rows: p.rows, Widths: p.widths, OutShapes: p.outShapes, Shapes: p.shapes}
	part := Part{Tensors: map[string]*Tensor{}}
	for name, t := range p.folded {
		l.Folded = append(l.Folded, name)
		part.Tensors[name] = t
	}
	slices.Sort(l.Folded)
	if len(outputs) > 0 && p.G != nil {
		part.Graph = p.G.Subgraph(given, outputs)
	}
	m := map[string]any{"program": l}
	for k, v := range meta {
		m[k] = v
	}
	var err error
	part.Meta, err = json.Marshal(m)
	return part, err
}

// ProgramOf is the program a part file of ProgramPart describes: its layout of the outputs and
// the values it folds (Split, Widths, OutShapes, Folded, Shapes), with no graph (nothing to
// build: the program is the converted one), and the part's graph.
func ProgramOf(part *Part) (*Program, error) {
	var m struct {
		Program *programLayout `json:"program"`
	}
	if err := json.Unmarshal(part.Meta, &m); err != nil || m.Program == nil {
		return nil, fmt.Errorf("onnx: a part file with no program's layout (%v)", err)
	}
	l := m.Program
	p := &Program{G: part.Graph, Outputs: l.Outputs, Axis: l.Axis, Flat: l.Flat, rows: l.Rows, widths: l.Widths, outShapes: l.OutShapes, shapes: l.Shapes, folded: map[string]*Tensor{}}
	for _, name := range l.Folded {
		t := part.Tensors[name]
		if t == nil {
			return nil, fmt.Errorf("onnx: the part file has no folded value %s", name)
		}
		p.folded[name] = t.Decoded()
	}
	return p, nil
}

// MetaOf reads the model's setting key of the part's meta into v.
func (p *Part) MetaOf(key string, v any) error {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(p.Meta, &m); err != nil {
		return fmt.Errorf("onnx: the part file's meta: %w", err)
	}
	raw, ok := m[key]
	if !ok {
		return fmt.Errorf("onnx: the part file's meta has no %s", key)
	}
	return json.Unmarshal(raw, v)
}
