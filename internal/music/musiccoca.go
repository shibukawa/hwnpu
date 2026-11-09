package music

import (
	"encoding/binary"
	"fmt"
	"os"
	"runtime/debug"
	"slices"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/onnx"
	"github.com/shibukawa/hwnpu/internal/tflite"
)

// musicCoCaRevision counts the changes to MusicCoCa's programs.
const musicCoCaRevision = 1

// MusicCoCa's sizes: the text encoder's tokens and vocabulary, the music encoder's log mel
// spectrogram of a 10 second clip at 16 kHz.
const (
	textLength     = 128 // tokens
	textVocab      = 16000
	StyleDim       = 768
	ClipRate       = 16000
	ClipSamples    = 10 * ClipRate
	melFrames      = 992
	melBins        = 128
	musicPatches   = 496
	musicPatchSize = 256
)

// onnxStep is an onnx.Program as a Program of named inputs (x, then Names) and y.
type onnxStep struct{ p *onnx.Program }

func (s onnxStep) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	ins, out, err := s.p.Build(b, w, opts)
	if err != nil {
		return nil, nil, err
	}
	return ins, []mlpackage.Feature{out}, nil
}

// encoderCut is how a MusicCoCa encoder's graph is split: the host computes the values Inputs
// (from the model's inputs) and the program the rest from them.
type encoderCut struct {
	m       *tflite.Model
	program *onnx.Program
	inputs  []string // the program's inputs, as graph values (x first)
	table   *onnx.Tensor
}

// textEncoderCut puts the text encoder on the NPU after its embedding lookup (the host looks up
// the tokens' rows of table, (16000, 768)): the program reads the tokens' embeddings (128, 768),
// the attention's condition (1 for a token, 0 for padding; the graph compares the paddings times
// -2.4e38 with a constant, and a program has no booleans) and the paddings.
func textEncoderCut(m *tflite.Model) (*encoderCut, error) {
	g := m.Graph
	var lookup *onnx.Node
	conds := map[string]bool{}
	for _, n := range g.Nodes {
		if n.Op == "Gather" && lookup == nil {
			if t, ok := g.Init[n.In[0]]; ok && len(t.Shape) == 2 && t.Shape[0] == textVocab && t.Shape[1] == StyleDim {
				lookup = n
			}
		}
		if n.Op == "Where" {
			conds[n.In[0]] = true
		}
	}
	if lookup == nil || len(conds) != 1 {
		return nil, fmt.Errorf("music: the text encoder has no embedding lookup or not one attention condition (%d)", len(conds))
	}
	cond := ""
	for k := range conds {
		cond = k
	}
	shapes, err := evalShapes(m, map[string]*onnx.Tensor{
		m.InputValues[0]: {Shape: []int{1, textLength}, Type: onnx.Int32, F: make([]float64, textLength)},
		m.InputValues[1]: {Shape: []int{1, textLength}, Type: onnx.Float, F: make([]float64, textLength)},
	}, []string{cond})
	if err != nil {
		return nil, err
	}
	table := g.Init[lookup.In[0]]
	emb := lookup.Out[0]
	pad := m.InputValues[1]
	p := &onnx.Program{G: g, Outputs: m.OutputValues[:1], Names: map[string]string{cond: "mask", pad: "paddings"},
		Inputs: []onnx.ValueInfo{{Name: emb, Shape: []int{textLength, StyleDim}}, {Name: cond, Shape: shapes[0]}, {Name: pad, Shape: []int{1, textLength}}}}
	return &encoderCut{m: m, program: p, inputs: []string{emb, cond, pad}, table: table}, nil
}

func evalShapes(m *tflite.Model, given map[string]*onnx.Tensor, values []string) ([][]int, error) {
	out, err := m.Graph.Eval(given, values)
	if err != nil {
		return nil, err
	}
	var shapes [][]int
	for _, v := range values {
		shapes = append(shapes, slices.Clone(out[v].Shape))
	}
	return shapes, nil
}

// textFeed is the text program's inputs for the encoder's ids and paddings (TextIDs), the
// embedding table (16000, 768) row major.
func textFeed(table []float32, ids, pad []float64) map[string][]float32 {
	x := make([]float32, textLength*StyleDim)
	mask := make([]float32, textLength)
	paddings := make([]float32, textLength)
	for i, id := range ids {
		if id >= 0 && int(id) < textVocab {
			copy(x[i*StyleDim:(i+1)*StyleDim], table[int(id)*StyleDim:(int(id)+1)*StyleDim])
		}
		mask[i], paddings[i] = float32(1-pad[i]), float32(pad[i])
	}
	return map[string][]float32{"x": x, "mask": mask, "paddings": paddings}
}

// musicEncoderCut puts the music encoder on the NPU from its first layer, the loop over the
// layers unrolled: the host cuts the log mel spectrogram (1, 992, 128) into its 496 patches
// (rank 6 on the way, more than Core ML's reshape takes) and the program reads them (496, 256).
func musicEncoderCut(m *tflite.Model) (*encoderCut, error) {
	if err := m.Graph.Unroll(); err != nil {
		return nil, err
	}
	var patches string
	for _, n := range m.Graph.Nodes {
		if n.Op == "MatMul" {
			patches = n.In[0]
			break
		}
	}
	if patches == "" {
		return nil, fmt.Errorf("music: the music encoder has no first layer")
	}
	shapes, err := evalShapes(m, map[string]*onnx.Tensor{m.InputValues[0]: {Shape: []int{1, melFrames, melBins}, Type: onnx.Float, F: make([]float64, melFrames*melBins)}}, []string{patches})
	if err != nil {
		return nil, err
	}
	p := &onnx.Program{G: m.Graph, Outputs: m.OutputValues[:1], Inputs: []onnx.ValueInfo{{Name: patches, Shape: shapes[0]}}}
	return &encoderCut{m: m, program: p, inputs: []string{patches}}, nil
}

// feed computes the program's inputs on the host, by their program names (x, then Names).
func (c *encoderCut) feed(given map[string]*onnx.Tensor) (map[string][]float32, error) {
	out, err := c.m.Graph.Eval(given, c.inputs)
	if err != nil {
		return nil, err
	}
	feed := map[string][]float32{}
	for i, v := range c.inputs {
		name := "x"
		if i > 0 {
			name = c.program.Names[v]
		}
		f := make([]float32, len(out[v].F))
		for j, x := range out[v].F {
			f[j] = float32(x)
		}
		feed[name] = f
	}
	return feed, nil
}

// loadTFLite loads a TFLite model with the garbage collector run often: the host evaluator
// keeps weights as float64, a few hundred MB to 1.5 GB.
func loadTFLite(path string) (*tflite.Model, func(), error) {
	old := debug.SetGCPercent(10)
	m, err := tflite.Load(path)
	return m, func() { debug.SetGCPercent(old); debug.FreeOSMemory() }, err
}

// textProgram is the text encoder's program (textEncoderCut); it writes the embedding table to
// table in fp16.
func textProgram(model, table string) (Program, error) {
	m, done, err := loadTFLite(model)
	defer done()
	if err != nil {
		return nil, err
	}
	c, err := textEncoderCut(m)
	if err != nil {
		return nil, err
	}
	// the condition the runtime computes as 1 - paddings: check it on a clip of 5 tokens
	ids, pad := make([]float64, textLength), make([]float64, textLength)
	for i := range pad {
		if i >= 5 {
			pad[i] = 1
		}
	}
	f, err := m.Graph.Eval(map[string]*onnx.Tensor{
		m.InputValues[0]: {Shape: []int{1, textLength}, Type: onnx.Int32, F: ids},
		m.InputValues[1]: {Shape: []int{1, textLength}, Type: onnx.Float, F: pad},
	}, c.inputs[1:2])
	if err != nil {
		return nil, err
	}
	for i, v := range f[c.inputs[1]].F {
		if v != 1-pad[i] {
			return nil, fmt.Errorf("music: the text encoder's attention condition is not 1 - paddings")
		}
	}
	b := make([]byte, 2*len(c.table.F))
	f32 := make([]float32, len(c.table.F))
	for i, v := range c.table.F {
		f32[i] = float32(v)
	}
	hwnpu.EncodeF16(b, f32)
	if err := os.WriteFile(table, b, 0o644); err != nil {
		return nil, err
	}
	return onnxStep{c.program}, nil
}

// musicProgram is the music encoder's program (musicEncoderCut); it writes the patches' indexes
// into the spectrogram to patches.
func musicProgram(model, patches string) (Program, error) {
	m, done, err := loadTFLite(model)
	defer done()
	if err != nil {
		return nil, err
	}
	c, err := musicEncoderCut(m)
	if err != nil {
		return nil, err
	}
	// the cut of an arange is the index each patch value takes
	idx := &onnx.Tensor{Shape: []int{1, melFrames, melBins}, Type: onnx.Float, F: make([]float64, melFrames*melBins)}
	for i := range idx.F {
		idx.F[i] = float64(i)
	}
	out, err := m.Graph.Eval(map[string]*onnx.Tensor{m.InputValues[0]: idx}, c.inputs)
	if err != nil {
		return nil, err
	}
	v := out[c.inputs[0]].F
	if len(v) != musicPatches*musicPatchSize {
		return nil, fmt.Errorf("music: the music encoder's patches are %v", out[c.inputs[0]].Shape)
	}
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(int32(x)))
	}
	if err := os.WriteFile(patches, b, 0o644); err != nil {
		return nil, err
	}
	return onnxStep{c.program}, nil
}
