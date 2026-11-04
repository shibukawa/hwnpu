package probe

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Chunk attention probe models (npuprobe attn): attention without weights over one chunk of a
// KV cache, the program a model would call once per used chunk and layer when its layers are
// split at the attention (docs/model-fit.md T2), so that a step costs in proportion to the tokens
// used, not to the cache's capacity. The host merges the chunks (ChunkMerge).
//
// Inputs: q (1, QHeads, tokens, HeadDim), RoPE already applied; k and v (1, KVHeads, chunk,
// HeadDim), one chunk of the cache with the step's own keys and values already written in by the
// host; mask (1, 1, tokens, chunk), 0 or -1e4. Outputs: o (1, QHeads, tokens, HeadDim), the
// chunk's attention normalized over the chunk alone; m and l (1, QHeads, tokens, 1), the largest
// scaled score of each row and the sum of exp(score - m) over the chunk.
//
// Forms: "grouped" stacks the query heads of a kv head along the rows of one matrix product, as
// blocks.SDPAGrouped (the default attention of LFM3-1.7B); "tile" copies the keys and values for
// every query head, as blocks.SDPA. Both take the sum of the exps as a matrix product with ones,
// as blocks.ChunkedSDPA does: a reduce_sum made the placement of 64 tokens per call change from
// one compile to the next (docs/design.md 16.22).

// AttnForms are the forms of WriteChunkAttnPackage.
var AttnForms = []string{"grouped", "tile"}

// WriteChunkAttnPackage writes a chunk attention probe model for tokens query rows and a chunk
// of the given length.
func WriteChunkAttnPackage(dir, form string, tokens, chunk int) error {
	if form != "grouped" && form != "tile" {
		return fmt.Errorf("probe: attention form %q is not grouped or tile", form)
	}
	if tokens < 1 || chunk < 1 {
		return fmt.Errorf("probe: %d tokens over a chunk of %d", tokens, chunk)
	}
	pkg, err := mlpackage.Create(dir)
	if err != nil {
		return err
	}
	pkg.Metadata["com.github.shibukawa.hwnpu.probe"] = fmt.Sprintf("attn %s %d tokens chunk %d", form, tokens, chunk)
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	op := func(typ string, in map[string]any, shape ...int) *mlpackage.Var {
		return b.Op(typ, in, mlpackage.Float16, shape...)
	}
	reshape := func(v *mlpackage.Var, sh ...int) *mlpackage.Var {
		s := make([]int32, len(sh))
		for i, d := range sh {
			s[i] = int32(d)
		}
		return op("reshape", map[string]any{"x": v, "shape": b.Ints(s...)}, sh...)
	}
	matmul := func(x, y *mlpackage.Var, ty bool, sh ...int) *mlpackage.Var {
		return op("matmul", map[string]any{"x": x, "y": y, "transpose_x": b.ScalarBool(false), "transpose_y": b.ScalarBool(ty)}, sh...)
	}
	const nh, nkv, d = QHeads, KVHeads, HeadDim
	rep, s, c := nh/nkv, tokens, chunk
	var ins []mlpackage.Feature
	input := func(name string, sh ...int) *mlpackage.Var {
		ins = append(ins, mlpackage.Feature{Name: name, DType: mlpackage.Float16, Shape: sh})
		return b.Input(name, mlpackage.Float16, sh...)
	}
	q := input("q", 1, nh, s, d)
	k := input("k", 1, nkv, c, d)
	v := input("v", 1, nkv, c, d)
	mask := input("mask", 1, 1, s, c)
	ones, err := b.F16s("ones", []int{c, 1}, constOnes(c))
	if err != nil {
		return err
	}
	scale := b.ScalarF16(float32(1 / math.Sqrt(d)))
	rows := []int{1, nh, s, 1}

	var sc *mlpackage.Var
	if form == "grouped" {
		qg := reshape(q, 1, nkv, rep*s, d)
		sc = reshape(matmul(qg, k, true, 1, nkv, rep*s, c), 1, nh, s, c)
	} else {
		k, v = tileHeads(b, k, rep), tileHeads(b, v, rep)
		sc = matmul(q, k, true, 1, nh, s, c)
	}
	sc = op("add", map[string]any{"x": op("mul", map[string]any{"x": sc, "y": scale}, 1, nh, s, c), "y": mask}, 1, nh, s, c)
	m := op("reduce_max", map[string]any{"x": sc, "axes": b.Ints(3), "keep_dims": b.ScalarBool(true)}, rows...)
	e := op("exp", map[string]any{"x": op("sub", map[string]any{"x": sc, "y": m}, 1, nh, s, c)}, 1, nh, s, c)
	var o, l *mlpackage.Var
	if form == "grouped" {
		e = reshape(e, 1, nkv, rep*s, c)
		l = matmul(e, ones, false, 1, nkv, rep*s, 1)
		o = op("real_div", map[string]any{"x": matmul(e, v, false, 1, nkv, rep*s, d), "y": l}, 1, nkv, rep*s, d)
		o, l = reshape(o, 1, nh, s, d), reshape(l, rows...)
	} else {
		l = matmul(e, ones, false, rows...)
		o = op("real_div", map[string]any{"x": matmul(e, v, false, 1, nh, s, d), "y": l}, 1, nh, s, d)
	}
	outs := []*mlpackage.Var{
		b.NamedOp("identity", "o", map[string]any{"x": o}, mlpackage.Float16, 1, nh, s, d),
		b.NamedOp("identity", "m", map[string]any{"x": m}, mlpackage.Float16, rows...),
		b.NamedOp("identity", "l", map[string]any{"x": l}, mlpackage.Float16, rows...),
	}
	var feats []mlpackage.Feature
	for _, v := range outs {
		feats = append(feats, mlpackage.Feature{Name: v.Name, DType: mlpackage.Float16, Shape: v.Shape})
	}
	return pkg.Finish(b.Finish(outs...), 9, ins, feats)
}

// tileHeads copies (1, nkv, t, d) for rep query heads each: (1, nkv*rep, t, d).
func tileHeads(b *mlpackage.Builder, x *mlpackage.Var, rep int) *mlpackage.Var {
	nkv, t, d := x.Shape[1], x.Shape[2], x.Shape[3]
	x = b.Op("reshape", map[string]any{"x": x, "shape": b.Ints(1, int32(nkv), 1, int32(t), int32(d))}, mlpackage.Float16, 1, nkv, 1, t, d)
	x = b.Op("tile", map[string]any{"x": x, "reps": b.Ints(1, 1, int32(rep), 1, 1)}, mlpackage.Float16, 1, nkv, rep, t, d)
	return b.Op("reshape", map[string]any{"x": x, "shape": b.Ints(1, int32(nkv*rep), int32(t), int32(d))}, mlpackage.Float16, 1, nkv*rep, t, d)
}

func constOnes(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

// AttnChunkBytes is the size of one chunk's keys and values in fp16.
func AttnChunkBytes(chunk int) int64 { return 2 * int64(KVHeads) * int64(chunk) * HeadDim * 2 }

// ChunkMerge combines the outputs of the chunk attention program over the chunks of a cache into
// the attention over all of them, in float32: the softmax over every key is each chunk's own,
// weighted by its sum of exps taken from the largest score seen so far (the merge of
// Flash-Decoding, online, so that a chunk's outputs can be dropped once added).
type ChunkMerge struct {
	rows, dim int
	acc       []float32 // rows × dim: the sum of the chunks' outputs, each times its weight
	top, sum  []float32 // per row: the largest score so far, and the weights' sum from it
}

// NewChunkMerge returns an empty merge of rows rows (query heads × tokens) of dim values.
func NewChunkMerge(rows, dim int) *ChunkMerge {
	c := &ChunkMerge{rows: rows, dim: dim, acc: make([]float32, rows*dim), top: make([]float32, rows), sum: make([]float32, rows)}
	c.Reset()
	return c
}

// Reset empties the merge for the next layer or step.
func (c *ChunkMerge) Reset() {
	clear(c.acc)
	clear(c.sum)
	for i := range c.top {
		c.top[i] = float32(math.Inf(-1))
	}
}

// Add adds a chunk: o (rows × dim) normalized over the chunk, m and l (rows) its largest scores
// and sums of exps from them. A row whose keys are all masked in the chunk has a score near the
// mask's -1e4 and weighs nothing next to a chunk where it sees keys.
func (c *ChunkMerge) Add(o, m, l []float32) {
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
func (c *ChunkMerge) Result(out []float32) {
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
