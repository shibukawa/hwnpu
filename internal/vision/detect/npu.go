package detect

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/onnx"
	"github.com/shibukawa/hwnpu/internal/vision/ironcnn"
)

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

// ExportVersion is the version of ExportNPU's exports (a key of the shared layers: chat.PinnedRecipe),
// bumped by hand when what it passes iron_onnxcnn.py or writes beside its export changes.
const ExportVersion = 1

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
	if err := ironcnn.Make(ctx, python, name, model, out, Size, names, nil, log); err != nil {
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
	g     *ironcnn.Graph
	order []int     // export output of each head (HeadShapes' order)
	scale []float64 // a box head's stride (negative: after an Exp; 0: a class head, as it is)
}

// OpenNPU loads an export of tools/amd/iron_onnxcnn.py (dir) of the ONNX file model onto dev:
// the heads of Heads, or the values before their Exp and Mul by the stride.
func OpenNPU(ctx context.Context, dev *hwnpu.Device, dir, model string) (*Detector, error) {
	x, err := ironcnn.Read(dir)
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
	h := &npuHeads{scale: scales}
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
	if h.g, err = ironcnn.Open(ctx, dev, dir); err != nil {
		return nil, err
	}
	return &Detector{Score: 0.3, NMS: 0.65, Max: 300, Classes: classes, npu: h}, nil
}

func (h *npuHeads) run(ctx context.Context, x []float32) ([][]float32, error) {
	planes, err := h.g.Run(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("detect: %w", err)
	}
	heads := make([][]float32, len(h.order))
	for k, j := range h.order {
		p := planes[j]
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
