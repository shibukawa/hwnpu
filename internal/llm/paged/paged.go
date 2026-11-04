// Package paged is the attention of a model split at its attention (llm.Options.Paged): the
// host holds a long KV cache in chunks of a fixed length, and a program without weights attends
// over one chunk at a time, called for the chunks in use only, so that a step costs in proportion
// to the tokens in the cache rather than to its capacity (docs/model-fit.md T2, docs/design.md
// 16.58). The host merges the chunks' outputs (Merge), as Flash-Decoding does.
//
// The program's inputs are q (1, heads, tokens, head_dim), RoPE already applied; k and v (1,
// kv_heads, chunk, head_dim), one chunk of the cache with the step's own keys and values already
// written in by the host; mask (1, 1, tokens, chunk), 0 or -1e4. Its outputs are o (1, heads,
// tokens, head_dim), the attention normalized over the chunk alone, and m and l (1, heads,
// tokens, 1), each row's largest scaled score and the sum of exp(score - m) over the chunk.
//
// Core ML puts the forms with little work on the CPU: on an M3 and an M4 (macOS 27.0.1), the
// grouped form with 1 query row stays off the Neural Engine up to chunks of 8192, with 8 rows up
// to 2048; 64 rows are on it from 1024. Chunks of 8192 take 8 and 64 rows on the Neural Engine.
package paged

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Shape is the attention's heads, and the scale of its scores.
type Shape struct {
	Heads, KVHeads, HeadDim int
	// Scale multiplies the scores; 0 for 1/sqrt(HeadDim). Gemma 4 takes 1: its query and key
	// norms take the place of the scaling.
	Scale float64
}

// DefaultChunk is the chunk length hwnpu-chat converts with: 8 and 64 query rows are on the
// Neural Engine, 8 rows at their fastest per key (1.09 ms per 1K over 6 layers on an M3), and the
// host's merge is a few percent of a step.
const DefaultChunk = 8192

// Build writes the program for tokens query rows over a chunk of the given length into b and
// returns its inputs and outputs. form is "grouped" (the query heads of a kv head stacked along
// the rows of one matrix product, as blocks.SDPAGrouped) or "tile" (the keys and values copied for
// each query head, as blocks.SDPA). The sum of the exps is a matrix product with ones, as
// blocks.ChunkedSDPA takes it: a reduce_sum made the placement of 64 rows per call change from
// one compile to the next.
func Build(b *mlpackage.Builder, sh Shape, form string, tokens, chunk int) (ins, outs []mlpackage.Feature, err error) {
	nh, nkv, d := sh.Heads, sh.KVHeads, sh.HeadDim
	if form != "grouped" && form != "tile" {
		return nil, nil, fmt.Errorf("paged: attention form %q is not grouped or tile", form)
	}
	if tokens < 1 || chunk < 1 || nh < 1 || nkv < 1 || d < 1 || nh%nkv != 0 {
		return nil, nil, fmt.Errorf("paged: %d tokens over a chunk of %d with %d heads on %d kv heads of %d", tokens, chunk, nh, nkv, d)
	}
	op := func(typ string, in map[string]any, shape ...int) *mlpackage.Var {
		return b.Op(typ, in, mlpackage.Float16, shape...)
	}
	reshape := func(v *mlpackage.Var, sh ...int) *mlpackage.Var {
		s := make([]int32, len(sh))
		for i, n := range sh {
			s[i] = int32(n)
		}
		return op("reshape", map[string]any{"x": v, "shape": b.Ints(s...)}, sh...)
	}
	matmul := func(x, y *mlpackage.Var, ty bool, sh ...int) *mlpackage.Var {
		return op("matmul", map[string]any{"x": x, "y": y, "transpose_x": b.ScalarBool(false), "transpose_y": b.ScalarBool(ty)}, sh...)
	}
	rep, s, c := nh/nkv, tokens, chunk
	input := func(name string, sh ...int) *mlpackage.Var {
		ins = append(ins, mlpackage.Feature{Name: name, DType: mlpackage.Float16, Shape: sh})
		return b.Input(name, mlpackage.Float16, sh...)
	}
	q := input("q", 1, nh, s, d)
	k := input("k", 1, nkv, c, d)
	v := input("v", 1, nkv, c, d)
	mask := input("mask", 1, 1, s, c)
	ones := make([]float32, c)
	for i := range ones {
		ones[i] = 1
	}
	one, err := b.F16s("ones", []int{c, 1}, ones)
	if err != nil {
		return nil, nil, err
	}
	scale := b.ScalarF16(float32(1 / math.Sqrt(float64(d))))
	if sh.Scale != 0 {
		scale = b.ScalarF16(float32(sh.Scale))
	}
	rows := []int{1, nh, s, 1}
	tile := func(x *mlpackage.Var) *mlpackage.Var {
		if rep == 1 {
			return x
		}
		x = reshape(x, 1, nkv, 1, c, d)
		x = op("tile", map[string]any{"x": x, "reps": b.Ints(1, 1, int32(rep), 1, 1)}, 1, nkv, rep, c, d)
		return reshape(x, 1, nh, c, d)
	}
	grouped := form == "grouped" && rep > 1
	var sc *mlpackage.Var
	if grouped {
		sc = reshape(matmul(reshape(q, 1, nkv, rep*s, d), k, true, 1, nkv, rep*s, c), 1, nh, s, c)
	} else {
		k, v = tile(k), tile(v)
		sc = matmul(q, k, true, 1, nh, s, c)
	}
	sc = op("add", map[string]any{"x": op("mul", map[string]any{"x": sc, "y": scale}, 1, nh, s, c), "y": mask}, 1, nh, s, c)
	m := op("reduce_max", map[string]any{"x": sc, "axes": b.Ints(3), "keep_dims": b.ScalarBool(true)}, rows...)
	e := op("exp", map[string]any{"x": op("sub", map[string]any{"x": sc, "y": m}, 1, nh, s, c)}, 1, nh, s, c)
	var o, l *mlpackage.Var
	if grouped {
		e = reshape(e, 1, nkv, rep*s, c)
		l = matmul(e, one, false, 1, nkv, rep*s, 1)
		o = op("real_div", map[string]any{"x": matmul(e, v, false, 1, nkv, rep*s, d), "y": l}, 1, nkv, rep*s, d)
		o, l = reshape(o, 1, nh, s, d), reshape(l, rows...)
	} else {
		l = matmul(e, one, false, rows...)
		o = op("real_div", map[string]any{"x": matmul(e, v, false, 1, nh, s, d), "y": l}, 1, nh, s, d)
	}
	for _, x := range []struct {
		name string
		v    *mlpackage.Var
	}{{"o", o}, {"m", m}, {"l", l}} {
		out := b.NamedOp("identity", x.name, map[string]any{"x": x.v}, mlpackage.Float16, x.v.Shape...)
		outs = append(outs, mlpackage.Feature{Name: out.Name, DType: mlpackage.Float16, Shape: out.Shape})
	}
	return ins, outs, nil
}

// Write writes the program as the package at path.
func Write(path string, sh Shape, form string, tokens, chunk int, description string) (ins, outs []mlpackage.Feature, err error) {
	pkg, err := mlpackage.Create(path)
	if err != nil {
		return nil, nil, err
	}
	pkg.Description = description
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	if ins, outs, err = Build(b, sh, form, tokens, chunk); err != nil {
		return nil, nil, err
	}
	var vars []*mlpackage.Var
	for _, f := range outs {
		vars = append(vars, &mlpackage.Var{Name: f.Name})
	}
	return ins, outs, pkg.Finish(b.Finish(vars...), 9, ins, outs)
}

// Merge combines the outputs of the program over the chunks of a cache into the attention over
// all of them, in float32: the softmax over every key is each chunk's own, weighted by its sum
// of exps from the largest score seen so far (the merge of Flash-Decoding, online, so that a
// chunk's outputs can be dropped once added).
type Merge struct {
	rows, dim int
	acc       []float32 // rows × dim: the sum of the chunks' outputs, each times its weight
	top, sum  []float32 // per row: the largest score so far, and the weights' sum from it
}

// NewMerge returns an empty merge of rows rows (heads × tokens, as o is laid out) of dim values.
func NewMerge(rows, dim int) *Merge {
	c := &Merge{rows: rows, dim: dim, acc: make([]float32, rows*dim), top: make([]float32, rows), sum: make([]float32, rows)}
	c.Reset()
	return c
}

// Reset empties the merge for the next layer or step.
func (c *Merge) Reset() {
	clear(c.acc)
	clear(c.sum)
	for i := range c.top {
		c.top[i] = float32(math.Inf(-1))
	}
}

// Add adds a chunk: o (rows × dim) normalized over the chunk, m and l (rows) its largest scores
// and sums of exps from them. A row whose keys are all masked in the chunk has a score near the
// mask's -1e4 and weighs nothing next to a chunk where it sees keys.
func (c *Merge) Add(o, m, l []float32) {
	for r := 0; r < c.rows; r++ {
		if l[r] <= 0 {
			continue
		}
		mj := m[r]
		if mj > c.top[r] {
			if c.sum[r] > 0 {
				f := float32(math.Exp(float64(c.top[r] - mj)))
				c.sum[r] *= f
				row := c.acc[r*c.dim : (r+1)*c.dim]
				for i := range row {
					row[i] *= f
				}
			}
			c.top[r] = mj
		}
		w := l[r] * float32(math.Exp(float64(mj-c.top[r])))
		c.sum[r] += w
		row, oj := c.acc[r*c.dim:(r+1)*c.dim], o[r*c.dim:(r+1)*c.dim]
		for i := range row {
			row[i] += w * oj[i]
		}
	}
}

// Result writes the merged attention, rows × dim, into out.
func (c *Merge) Result(out []float32) {
	for r := 0; r < c.rows; r++ {
		inv := float32(0)
		if c.sum[r] > 0 {
			inv = 1 / c.sum[r]
		}
		for i := 0; i < c.dim; i++ {
			out[r*c.dim+i] = c.acc[r*c.dim+i] * inv
		}
	}
}
