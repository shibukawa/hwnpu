package whisper

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// CrossArch registers the program that turns the encoder's output into the decoder's cross
// attention keys and values, once per clip; it is chosen with -arch.
var CrossArch = &llm.Arch{
	Name:       "whisper-cross",
	Title:      "Whisper cross attention keys and values",
	Format:     "lut8",
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseCrossConfig(raw) },
}

func init() { llm.Register(CrossArch) }

// Cross is the cross attention's key and value projections of every decoder layer (llm.Model).
type Cross struct{ *Config }

// ParseCrossConfig parses config.json for the cross program.
func ParseCrossConfig(raw []byte) (*Cross, error) {
	c, err := ParseConfig(raw)
	if err != nil {
		return nil, err
	}
	return &Cross{c}, nil
}

// Hidden is the encoder's output width, the program's input.
func (c *Cross) Hidden() int { return c.DModel }

// Layers is 0: the program has no layers of its own, the projections of all the decoder's
// layers at once.
func (c *Cross) Layers() int { return 0 }

// Output is the keys and values of every decoder layer per position.
func (c *Cross) Output() (int, bool) { return 2 * c.DecoderLayers * c.DModel, false }

// Embedding is empty: the input is the encoder's output.
func (c *Cross) Embedding() string { return "" }

// Tensors lists the cross attention's key and value projections (k_proj has no bias).
func (c *Cross) Tensors() map[string][]int {
	d := c.DModel
	t := map[string][]int{}
	for i := 0; i < c.DecoderLayers; i++ {
		p := c.decoderLayer(i) + "encoder_attn."
		t[p+"k_proj.weight"] = []int{d, d}
		t[p+"v_proj.weight"] = []int{d, d}
		t[p+"v_proj.bias"] = []int{d}
	}
	return t
}

// CrossHeads is the rows of the cross program's output y (1, CrossHeads, positions, head_dim):
// layer N's keys are heads [2N heads, (2N+1) heads), its values the next heads, each as the
// decoder's input xk_N and xv_N (1, heads, positions, head_dim) takes them.
func (c *Config) CrossHeads() int { return 2 * c.DecoderLayers * c.DecoderHeads }

// Build writes the program: its input x is the encoder's last hidden state (1, positions,
// d_model), its output y the keys and values (1, CrossHeads, positions, head_dim).
func (c *Cross) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Cache > 0 || opts.Start > 0 {
		return nil, out, fmt.Errorf("whisper: the cross program is in one piece, without a cache")
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = CrossArch.Format
	}
	g := &blocks.Graph{Name: "whisper", B: b, W: w, Opts: opts}
	n, nh := c.SourcePositions, c.DecoderHeads
	hd := c.DModel / nh
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, n, c.DModel)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		var parts []*mlpackage.Var
		for i := 0; i < c.DecoderLayers; i++ {
			p := c.decoderLayer(i) + "encoder_attn."
			parts = append(parts, heads(g, x, p+"k_proj.weight", "", nh), heads(g, x, p+"v_proj.weight", p+"v_proj.bias", nh))
		}
		y := g.B.NamedOp("concat", "y", map[string]any{"values": parts, "axis": b.ScalarI32(1), "interleave": b.ScalarBool(false)}, mlpackage.Float16, 1, c.CrossHeads(), n, hd)
		out = mlpackage.Feature{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}
	})
	return ins, out, err
}
