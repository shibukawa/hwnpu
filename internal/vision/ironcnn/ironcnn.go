// Package ironcnn runs the exports of tools/amd/iron_onnxcnn.py on AMD's NPU: a convolutional
// ONNX graph (RTMDet, RTMO) up to some of its values as one IRON graph (a full ELF), its input a
// picture's channels in a zero-padded grid and its outputs grids read back channels first. The
// packages of the models (detect, pose) choose the values and run the rest of the graph on the
// host. See docs/amd-npu-models.md 20.
package ironcnn

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/onnx"
)

// Export is model.json of an export: the graph's input grid and its outputs, each a
// channel-last grid zero-padded by P pixels (pixel (y, x) at row band + y wp + P + x, Cp
// channels a row, the first C real), and the per-call values its Run takes (RTMO's attention:
// "nk", the tokens it reads).
type Export struct {
	Kind   string `json:"kind"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
	P      int    `json:"P"`
	Input  struct {
		Rows int `json:"rows"`
		Cp   int `json:"cp"`
		Wp   int `json:"wp"`
	} `json:"input"`
	Values  map[string]int64 `json:"values"`
	Outputs []Output         `json:"outputs"`
}

// Output is one output grid of an Export: the ONNX value it is and its layout.
type Output struct {
	Name string `json:"name"`
	C    int    `json:"C"`
	Cp   int    `json:"Cp"`
	H    int    `json:"H"`
	W    int    `json:"W"`
	Wp   int    `json:"wp"`
	Rows int    `json:"rows"`
	Band int    `json:"band"`
}

// Dir is where the export of a model of models.yaml is kept: ~/models/NAME.amd-npu2, NAME in
// lower case (as hwnpu-chat's AMD exports).
func Dir(name string) string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "models", strings.ToLower(name)+".amd-npu2")
}

// Read reads dir's model.json.
func Read(dir string) (*Export, error) {
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	var x Export
	if err := json.Unmarshal(b, &x); err != nil {
		return nil, fmt.Errorf("ironcnn: %s: %w", dir, err)
	}
	if x.Kind != "onnx-cnn" || len(x.Outputs) == 0 || x.Input.Cp != 4 {
		return nil, fmt.Errorf("ironcnn: %s is not an export of tools/amd/iron_onnxcnn.py of a picture's 3 channels", dir)
	}
	return &x, nil
}

// InputGrid is the graph's input of x (3, H, W, as the model reads it): the input grid, bf16,
// its fourth channel one inside the picture (the biases').
func (x *Export) InputGrid(px []float32) []byte {
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
func (o Output) Planes(b []byte, P int) []float32 {
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

// Graph is an export loaded on AMD's NPU.
type Graph struct {
	X *Export
	g *iron.Graph
}

// Open loads the export in dir onto dev.
func Open(ctx context.Context, dev *hwnpu.Device, dir string) (*Graph, error) {
	x, err := Read(dir)
	if err != nil {
		return nil, err
	}
	g, err := iron.OpenGraph(ctx, dev, dir)
	if err != nil {
		return nil, err
	}
	return &Graph{X: x, g: g}, nil
}

// Run runs the graph on px (3, H, W) and gives each output's planes (Output.Planes), in the
// export's order.
func (g *Graph) Run(ctx context.Context, px []float32) ([][]float32, error) {
	if err := g.g.SetInput(g.g.Inputs()[0].Name, g.X.InputGrid(px)); err != nil {
		return nil, err
	}
	var values map[string]int64
	if len(g.X.Values) > 0 {
		values = g.X.Values
	}
	if err := g.g.Run(ctx, values); err != nil {
		return nil, fmt.Errorf("ironcnn: %w", err)
	}
	outs := g.g.Outputs()
	planes := make([][]float32, len(outs))
	for j, o := range outs {
		b, err := g.g.Output(o.Name)
		if err != nil {
			return nil, err
		}
		planes[j] = g.X.Outputs[j].Planes(b, g.X.P)
	}
	return planes, nil
}

func (g *Graph) Close() error { return g.g.Close() }

// Make makes the export of the ONNX file model in out with IRON's python (chat.IRONPython):
// the graph from its input (size x size) to the values outputs, with iron_onnxcnn.py's options
// opts.
func Make(ctx context.Context, python []string, name, model, out string, size int, outputs, opts []string, log io.Writer) error {
	args := append([]string{model, chat.ExportOut, "--size", fmt.Sprintf("%dx%d", size, size)}, opts...)
	args = append(append(args, "--outputs"), outputs...)
	return chat.RunIRON(ctx, python, name, "iron_onnxcnn.py", args, out, log)
}

// LastConvs are the outputs of g's convolutions that no convolution reads (directly or
// through any other node): a network's heads.
func LastConvs(g *onnx.Graph) []string {
	users := map[string][]*onnx.Node{}
	for _, n := range g.Nodes {
		for _, i := range n.In {
			users[i] = append(users[i], n)
		}
	}
	reaches := map[string]bool{} // whether a convolution reads the value, or what it makes
	var reach func(v string, seen map[string]bool) bool
	reach = func(v string, seen map[string]bool) bool {
		if r, ok := reaches[v]; ok {
			return r
		}
		if seen[v] {
			return false
		}
		seen[v] = true
		r := false
		for _, u := range users[v] {
			if u.Op == "Conv" {
				r = true
				break
			}
			for _, o := range u.Out {
				if reach(o, seen) {
					r = true
					break
				}
			}
			if r {
				break
			}
		}
		reaches[v] = r
		return r
	}
	var out []string
	for _, n := range g.Nodes {
		if n.Op == "Conv" && !reach(n.Out[0], map[string]bool{}) {
			out = append(out, n.Out[0])
		}
	}
	return out
}

func bf16(x float32) uint16 {
	bits := math.Float32bits(x)
	if bits&0x7fffffff > 0x7f800000 {
		return uint16(bits>>16) | 0x40
	}
	return uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
}
