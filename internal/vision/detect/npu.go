package detect

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/onnx"
)

// CNNExport is model.json of an export of tools/amd/iron_onnxcnn.py: the graph's input grid and
// its outputs, each a channel-last grid zero-padded by P pixels (pixel (y, x) at row band + y
// wp + P + x, Cp channels a row, the first C real).
type CNNExport struct {
	Kind   string `json:"kind"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
	P      int    `json:"P"`
	Input  struct {
		Rows int `json:"rows"`
		Cp   int `json:"cp"`
		Wp   int `json:"wp"`
	} `json:"input"`
	Outputs []CNNOutput `json:"outputs"`
}

// CNNOutput is one output grid of a CNNExport: the ONNX value it is and its layout.
type CNNOutput struct {
	Name string `json:"name"`
	C    int    `json:"C"`
	Cp   int    `json:"Cp"`
	H    int    `json:"H"`
	W    int    `json:"W"`
	Wp   int    `json:"wp"`
	Rows int    `json:"rows"`
	Band int    `json:"band"`
}

// ReadCNNExport reads dir's model.json.
func ReadCNNExport(dir string) (*CNNExport, error) {
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	var x CNNExport
	if err := json.Unmarshal(b, &x); err != nil {
		return nil, fmt.Errorf("detect: %s: %w", dir, err)
	}
	if x.Kind != "onnx-cnn" || len(x.Outputs) == 0 || x.Input.Cp != 4 {
		return nil, fmt.Errorf("detect: %s is not an export of tools/amd/iron_onnxcnn.py of a picture's 3 channels", dir)
	}
	return &x, nil
}

// InputGrid is the graph's input of x (3, H, W, normalized: Pixels'): the input grid, bf16, its
// fourth channel one inside the picture (the biases').
func (x *CNNExport) InputGrid(px []float32) []byte {
	W, H, P, Wp := x.Width, x.Height, x.P, x.Input.Wp
	band := P*Wp + P
	out := make([]byte, 2*x.Input.Rows*4)
	one := bf16(1)
	for y := range H {
		for i := range W {
			r := band + y*Wp + P + i
			for c := range 3 {
				binary.LittleEndian.PutUint16(out[2*(r*4+c):], bf16(px[(c*H+y)*W+i]))
			}
			binary.LittleEndian.PutUint16(out[2*(r*4+3):], one)
		}
	}
	return out
}

// Planes are an output grid's real channels, channels first (C, H, W) in float32.
func (o CNNOutput) Planes(b []byte, P int) []float32 {
	out := make([]float32, o.C*o.H*o.W)
	n := o.H * o.W
	for y := range o.H {
		for x := range o.W {
			r := o.Band + y*o.Wp + P + x
			row := b[2*r*o.Cp:]
			for c := range o.C {
				out[c*n+y*o.W+x] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(row[2*c:])) << 16)
			}
		}
	}
	return out
}

// NPUOutputs are the graph's values an export of tools/amd/iron_onnxcnn.py gives for the heads,
// in HeadShapes' order (iron_onnxcnn.py --outputs): a class head's logits as they are (scale 0),
// a box head's value before its Mul by the stride (scale: the stride) and its Exp where it has
// one (scale: minus the stride), which the host does.
func NPUOutputs(g *onnx.Graph) (names []string, scales []float64, err error) {
	cls, box, err := Heads(g)
	if err != nil {
		return nil, nil, err
	}
	prod := map[string]*onnx.Node{}
	for _, n := range g.Nodes {
		for _, o := range n.Out {
			prod[o] = n
		}
	}
	for i := range Strides {
		names, scales = append(names, cls[i]), append(scales, 0)
		// Mul(Exp(conv), stride) (RTMDet-m) or Mul(conv, stride) (tiny, s): the export's output is
		// the conv; a negative scale stands for the Exp before it
		m := prod[box[i]]
		var c *onnx.Tensor
		if m != nil && m.Op == "Mul" && len(m.In) == 2 {
			c = g.Init[m.In[1]]
		}
		if c == nil || len(c.F) != 1 || c.F[0] <= 0 {
			return nil, nil, fmt.Errorf("detect: box head %s is not a convolution (or its exp) times the stride", box[i])
		}
		if e := prod[m.In[0]]; e != nil && e.Op == "Exp" {
			names, scales = append(names, e.In[0]), append(scales, -c.F[0])
		} else {
			names, scales = append(names, m.In[0]), append(scales, c.F[0])
		}
	}
	return names, scales, nil
}

// headsFile is the file in an export that ExportNPU writes: NPUOutputs' names and scales, so
// that a run need not read the ONNX file's graph (its weights in float64).
const headsFile = "heads.json"

type headsJSON struct {
	Names  []string  `json:"names"`
	Scales []float64 `json:"scales"`
}

// readHeads is NPUOutputs of the ONNX file model, from dir's heads.json when it has one.
func readHeads(dir, model string) ([]string, []float64, error) {
	if b, err := os.ReadFile(filepath.Join(dir, headsFile)); err == nil {
		var h headsJSON
		if err := json.Unmarshal(b, &h); err == nil && len(h.Names) == len(h.Scales) && len(h.Names) == 2*len(Strides) {
			return h.Names, h.Scales, nil
		}
	}
	g, err := onnx.Load(model)
	if err != nil {
		return nil, nil, err
	}
	return NPUOutputs(g)
}

// ExportNPU makes the export of tools/amd/iron_onnxcnn.py of the ONNX file model in out, with
// IRON's python (chat.IRONPython): its heads (NPUOutputs) at Size x Size.
func ExportNPU(ctx context.Context, python []string, name, model, out string, log io.Writer) error {
	g, err := onnx.Load(model)
	if err != nil {
		return err
	}
	names, scales, err := NPUOutputs(g)
	if err != nil {
		return err
	}
	args := append([]string{model, chat.ExportOut, "--size", fmt.Sprintf("%dx%d", Size, Size), "--outputs"}, names...)
	if err := chat.RunIRON(ctx, python, name, "iron_onnxcnn.py", args, out, log); err != nil {
		return err
	}
	b, err := json.Marshal(headsJSON{names, scales})
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, headsFile), b, 0o644)
}

// npuHeads runs an export of tools/amd/iron_onnxcnn.py on AMD's NPU and gives the heads in
// HeadShapes' order: each head an export's output, the boxes through the graph's tail on the
// host (exp, times the stride: what the export leaves out).
type npuHeads struct {
	x     *CNNExport
	g     *iron.Graph
	order []int     // export output of each head (HeadShapes' order)
	scale []float64 // a box head's stride (negative: after an Exp; 0: a class head, as it is)
}

// OpenNPU loads an export of tools/amd/iron_onnxcnn.py (dir) of the ONNX file model onto dev:
// the heads of Heads, or the values before their Exp and Mul by the stride.
func OpenNPU(ctx context.Context, dev *hwnpu.Device, dir, model string) (*Detector, error) {
	x, err := ReadCNNExport(dir)
	if err != nil {
		return nil, err
	}
	if x.Width != Size || x.Height != Size {
		return nil, fmt.Errorf("detect: %s is for %dx%d, not %dx%d", dir, x.Width, x.Height, Size, Size)
	}
	names, scales, err := readHeads(dir, model)
	if err != nil {
		return nil, err
	}
	h := &npuHeads{x: x, scale: scales}
	at := map[string]int{}
	for i, o := range x.Outputs {
		at[o.Name] = i
	}
	for _, v := range names {
		j, ok := at[v]
		if !ok {
			return nil, fmt.Errorf("detect: %s has no output %s (iron_onnxcnn.py --outputs)", dir, v)
		}
		h.order = append(h.order, j)
	}
	shapes := HeadShapes(len(COCO))
	classes := COCO
	if c := x.Outputs[h.order[0]].C; c != len(COCO) {
		classes = make([]string, c)
		for i := range classes {
			classes[i] = fmt.Sprint(i)
		}
		shapes = HeadShapes(c)
	}
	for k, j := range h.order {
		o := x.Outputs[j]
		if s := shapes[k]; o.C != s[1] || o.H != s[2] || o.W != s[3] {
			return nil, fmt.Errorf("detect: output %s is %dx%dx%d, not the head's %v", o.Name, o.C, o.H, o.W, s)
		}
	}
	if h.g, err = iron.OpenGraph(ctx, dev, dir); err != nil {
		return nil, err
	}
	return &Detector{Score: 0.3, NMS: 0.65, Max: 300, Classes: classes, npu: h}, nil
}

func (h *npuHeads) run(ctx context.Context, x []float32) ([][]float32, error) {
	g := h.g
	if err := g.SetInput(g.Inputs()[0].Name, h.x.InputGrid(x)); err != nil {
		return nil, err
	}
	if err := g.Run(ctx, nil); err != nil {
		return nil, fmt.Errorf("detect: %w", err)
	}
	outs := g.Outputs()
	heads := make([][]float32, len(h.order))
	for k, j := range h.order {
		b, err := g.Output(outs[j].Name)
		if err != nil {
			return nil, err
		}
		p := h.x.Outputs[j].Planes(b, h.x.P)
		switch s := h.scale[k]; {
		case s < 0:
			for i, v := range p {
				p[i] = float32(math.Exp(float64(v)) * -s)
			}
		case s > 0:
			for i, v := range p {
				p[i] = v * float32(s)
			}
		}
		heads[k] = p
	}
	return heads, nil
}

func bf16(x float32) uint16 {
	bits := math.Float32bits(x)
	if bits&0x7fffffff > 0x7f800000 {
		return uint16(bits>>16) | 0x40
	}
	return uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
}
