package paged

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// The two-level attention (decode.Sparse) reads a long cache in two ways: the keys of the buckets
// that weigh most exactly, and every other bucket of S keys as one summary key, their mean, with
// log S added to its score (the bucket's S keys taken as equal), whose value is their values'
// mean. The buckets to read exactly are chosen by their weight over the call's queries: the
// softmax of each query row over the summaries, summed over the rows and query heads of a kv
// head. The mass program gives that weight for one chunk of summaries, the rows' normalizers
// computed beforehand by the attention program over all of them (Merge.Norm).
//
// Its inputs are q (1, heads, tokens, head_dim) as the attention program's; k (1, kv_heads,
// chunk, head_dim), the summaries' keys; mask (1, 1, tokens, chunk), log S for the summaries
// counted and -1e4 for the others; and r (1, heads, tokens, 1), minus each row's log of its sum
// of exps over all the summaries (-1e4 for a row not counted). Its output w (1, kv_heads, 1,
// chunk) is each summary's softmax summed over the rows of its kv head.

// BuildMass writes the mass program for tokens query rows over a chunk of summaries into b and
// returns its inputs and outputs. The form is the attention program's (Build).
func BuildMass(b *mlpackage.Builder, sh Shape, form string, tokens, chunk int) (ins, outs []mlpackage.Feature, err error) {
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
	mask := input("mask", 1, 1, s, c)
	r := input("r", 1, nh, s, 1)
	ones := make([]float32, nkv*rep*s)
	for i := range ones {
		ones[i] = 1
	}
	// the sum over the rows is a matrix product with ones, as the attention program's sum of exps
	one, err := b.F16s("ones", []int{1, nkv, 1, rep * s}, ones)
	if err != nil {
		return nil, nil, err
	}
	scale := b.ScalarF16(float32(1 / math.Sqrt(float64(d))))
	var sc *mlpackage.Var
	if form == "grouped" && rep > 1 {
		sc = reshape(matmul(reshape(q, 1, nkv, rep*s, d), k, true, 1, nkv, rep*s, c), 1, nh, s, c)
	} else {
		if rep > 1 {
			k = reshape(k, 1, nkv, 1, c, d)
			k = op("tile", map[string]any{"x": k, "reps": b.Ints(1, 1, int32(rep), 1, 1)}, 1, nkv, rep, c, d)
			k = reshape(k, 1, nh, c, d)
		}
		sc = matmul(q, k, true, 1, nh, s, c)
	}
	sc = op("add", map[string]any{"x": op("mul", map[string]any{"x": sc, "y": scale}, 1, nh, s, c), "y": mask}, 1, nh, s, c)
	e := op("exp", map[string]any{"x": op("add", map[string]any{"x": sc, "y": r}, 1, nh, s, c)}, 1, nh, s, c)
	w := matmul(one, reshape(e, 1, nkv, rep*s, c), false, 1, nkv, 1, c)
	out := b.NamedOp("identity", "w", map[string]any{"x": w}, mlpackage.Float16, w.Shape...)
	outs = append(outs, mlpackage.Feature{Name: out.Name, DType: mlpackage.Float16, Shape: out.Shape})
	return ins, outs, nil
}

// WriteMass writes the mass program as the package at path.
func WriteMass(path string, sh Shape, form string, tokens, chunk int, description string) (ins, outs []mlpackage.Feature, err error) {
	pkg, err := mlpackage.Create(path)
	if err != nil {
		return nil, nil, err
	}
	pkg.Description = description
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	if ins, outs, err = BuildMass(b, sh, form, tokens, chunk); err != nil {
		return nil, nil, err
	}
	var vars []*mlpackage.Var
	for _, f := range outs {
		vars = append(vars, &mlpackage.Var{Name: f.Name})
	}
	return ins, outs, pkg.Finish(b.Finish(vars...), 9, ins, outs)
}

// Norm writes each row's log of its sum of exps over the chunks added, log(sum) + top, into out
// (rows), and -Inf for a row that saw nothing.
func (c *Merge) Norm(out []float32) {
	for r := range c.rows {
		if c.sum[r] > 0 {
			out[r] = c.top[r] + float32(math.Log(float64(c.sum[r])))
		} else {
			out[r] = float32(math.Inf(-1))
		}
	}
}

// Sub takes out a chunk added before, or a part of what was added (the summaries of the buckets
// read exactly, which the attention over all the summaries counted too): its o, m and l as Add
// takes them. What remains of a row is not let below nothing, which only the rounding of fp16
// could make it.
func (c *Merge) Sub(o, m, l []float32) {
	for r := 0; r < c.rows; r++ {
		if l[r] <= 0 || c.sum[r] <= 0 {
			continue
		}
		w := l[r] * float32(math.Exp(float64(m[r]-c.top[r])))
		row, oj := c.acc[r*c.dim:(r+1)*c.dim], o[r*c.dim:(r+1)*c.dim]
		if w >= c.sum[r] {
			c.sum[r] = 0
			clear(row)
			continue
		}
		c.sum[r] -= w
		for i := range row {
			row[i] -= w * oj[i]
		}
	}
}

// AddMerge adds what another merge of the same rows holds, as if its chunks had been added here.
func (c *Merge) AddMerge(o *Merge) {
	for r := 0; r < c.rows; r++ {
		if o.sum[r] <= 0 {
			continue
		}
		mj := o.top[r]
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
		f := float32(math.Exp(float64(mj - c.top[r])))
		c.sum[r] += o.sum[r] * f
		row, oj := c.acc[r*c.dim:(r+1)*c.dim], o.acc[r*c.dim:(r+1)*c.dim]
		for i := range row {
			row[i] += f * oj[i]
		}
	}
}
