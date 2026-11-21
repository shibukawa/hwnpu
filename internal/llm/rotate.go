package llm

import (
	"fmt"
	"math"
	"math/rand"
)

// Rotation says how an orthogonal matrix Q passes through a model's residual stream, so that the
// weights can be rotated offline (QuaRot, SpinQuant: the residual r becomes r Q). The values in
// the matrices that read and write the residual are mixed across its dimension, which spreads a
// row's outliers over the row: a lookup table of whole rows (the Neural Engine's only kind,
// docs/design/llm-qwen.md) then fits them with fewer bits. Nothing is added to the program: Q is in the
// weights.
//
// It holds for a pre-norm model whose every reader of the residual is a linear after an RMS
// norm (the norm's weight folded into the linear), and whose every writer is a linear whose
// output is added to the residual as it is. A norm on a writer's output (LFM3's and Gemma's
// sandwich norms) does not let Q through.
type Rotation struct {
	Dim int
	// Embed is the token embedding table (vocab, Dim): E Q, as Rows reads it.
	Embed string
	// Head is the output layer (vocab, Dim) after the final norm HeadNorm: a reader. Get gives it
	// even when it is Embed (tied), whose Rows stay the embedding's.
	Head, HeadNorm string
	// Readers are the matrices (out, Dim) that read the residual after a norm, by the norm whose
	// weight is folded into them: W diag(norm) Q.
	Readers map[string]string
	// Writers are the matrices (Dim, in) whose output is added to the residual: Qt W.
	Writers map[string]bool
	// Norms are the residual's norms: their weights become the identity's (zeros when
	// ZeroCentered, the norm then times 1 + w, else ones).
	Norms        map[string]bool
	ZeroCentered bool
}

// Rotator is a model that can be rotated (Rotation).
type Rotator interface {
	Rotation() (Rotation, error)
}

// RotateParam is the architecture setting (Options.Params) that builds a Rotator's programs from
// its weights rotated (RotateFor), "1"; RotationSeed is the seed of their signs.
const (
	RotateParam        = "rotate"
	RotationSeed int64 = 1
)

// RotateFor gives w rotated (Rotate, RotationSeed) when o asks for it (RotateParam), else w.
func RotateFor(m Model, w Weights, o Options) (Weights, error) {
	if o.Params[RotateParam] != "1" {
		return w, nil
	}
	if _, ok := w.(*rotated); ok {
		return w, nil
	}
	r, ok := m.(Rotator)
	if !ok {
		return nil, fmt.Errorf("llm: the model cannot be rotated (%s=1)", RotateParam)
	}
	rot, err := r.Rotation()
	if err != nil {
		return nil, err
	}
	return Rotate(w, rot, RotationSeed)
}

// Rotate gives w's tensors rotated as r says, with Q = diag(s) H: random signs s (from seed)
// and the Walsh-Hadamard matrix H of the largest power of two dividing Dim, block diagonal when
// that is less than Dim.
func Rotate(w Weights, r Rotation, seed int64) (Weights, error) {
	s, b, err := rotationSigns(r.Dim, seed)
	if err != nil {
		return nil, err
	}
	return &rotated{Weights: w, r: r, signs: s, block: b}, nil
}

func rotationSigns(dim int, seed int64) ([]float32, int, error) {
	b := dim & -dim
	if b < 64 {
		return nil, 0, fmt.Errorf("rotate: hidden size %d has no Hadamard block of 64 or more", dim)
	}
	rng := rand.New(rand.NewSource(seed))
	s := make([]float32, dim)
	for i := range s {
		s[i] = 1
		if rng.Intn(2) == 0 {
			s[i] = -1
		}
	}
	return s, b, nil
}

// Unrotate returns what takes a row of the rotated residual (r Q, Rotate with seed) back to r in
// place: r Q Qt, Qt = H diag(s), the Walsh-Hadamard transform and the signs (the host's rows for
// a program built unrotated: the MTP layer's).
func Unrotate(dim int, seed int64) (func(row []float32), error) {
	s, b, err := rotationSigns(dim, seed)
	if err != nil {
		return nil, err
	}
	w := &rotated{signs: s, block: b}
	return func(row []float32) {
		w.hadamard(row)
		for i := range row {
			row[i] *= s[i]
		}
	}, nil
}

// Table gives a table the host reads rows of (WriteTables): the embedding as E Q (Rows' form,
// where Get gives a tied table the output layer's), the final norm's weight as the checkpoint
// has it (for Unrotate's callers that apply it after), any other as Get.
func (w *rotated) Table(name string) ([]float32, []int, error) {
	switch name {
	case w.r.Embed:
		v, shape, err := w.Weights.Get(name)
		if err != nil {
			return nil, nil, err
		}
		return v, shape, w.rows(v, shape, nil)
	case w.r.HeadNorm:
		return w.Weights.Get(name)
	}
	return w.Get(name)
}

type rotated struct {
	Weights
	r     Rotation
	signs []float32
	block int
}

func (w *rotated) Source() string { return w.Weights.Source() + " (rotated)" }

func (w *rotated) Get(name string) ([]float32, []int, error) {
	v, shape, err := w.Weights.Get(name)
	if err != nil {
		return nil, nil, err
	}
	d := w.r.Dim
	switch {
	case w.r.Norms[name]:
		v = make([]float32, len(v))
		if !w.r.ZeroCentered {
			for i := range v {
				v[i] = 1
			}
		}
	case name == w.r.Head:
		err = w.reader(v, shape, w.r.HeadNorm)
	case w.r.Readers[name] != "" || hasKey(w.r.Readers, name):
		err = w.reader(v, shape, w.r.Readers[name])
	case w.r.Writers[name]:
		if len(shape) != 2 || shape[0] != d {
			return nil, nil, fmt.Errorf("rotate: writer %s is %v, not (%d, in)", name, shape, d)
		}
		w.writer(v, shape[1])
	case name == w.r.Embed:
		err = w.rows(v, shape, nil)
	}
	return v, shape, err
}

func hasKey(m map[string]string, k string) bool { _, ok := m[k]; return ok }

// Rows of the embedding are E Q; of another matrix, the rows of what Get gives.
func (w *rotated) Rows(name string, begin, end int) ([]float32, error) {
	if name == w.r.Embed {
		v, err := w.Weights.Rows(name, begin, end)
		if err != nil {
			return nil, err
		}
		return v, w.rows(v, []int{end - begin, w.r.Dim}, nil)
	}
	if !w.r.Norms[name] && name != w.r.Head && !hasKey(w.r.Readers, name) && !w.r.Writers[name] {
		return w.Weights.Rows(name, begin, end)
	}
	v, shape, err := w.Get(name)
	if err != nil {
		return nil, err
	}
	cols := 1
	for _, s := range shape[1:] {
		cols *= s
	}
	return v[begin*cols : end*cols], nil
}

// reader makes the (out, Dim) matrix v W diag(1 + w or w) Q, the norm's weight w.
func (w *rotated) reader(v []float32, shape []int, norm string) error {
	if len(shape) != 2 || shape[1] != w.r.Dim {
		return fmt.Errorf("rotate: reader of shape %v, not (out, %d)", shape, w.r.Dim)
	}
	var g []float32
	if norm != "" {
		nv, _, err := w.Weights.Get(norm)
		if err != nil {
			return err
		}
		g = make([]float32, len(nv))
		for i, x := range nv {
			g[i] = x
			if w.r.ZeroCentered {
				g[i] = 1 + x
			}
		}
	}
	return w.rows(v, shape, g)
}

// rows makes each row x of the (n, Dim) matrix v into (x g s) H.
func (w *rotated) rows(v []float32, shape []int, g []float32) error {
	d := w.r.Dim
	if len(v)%d != 0 {
		return fmt.Errorf("rotate: %d values are not rows of %d", len(v), d)
	}
	for o := 0; o < len(v); o += d {
		row := v[o : o+d]
		for i := range row {
			if g != nil {
				row[i] *= g[i]
			}
			row[i] *= w.signs[i]
		}
		w.hadamard(row)
	}
	return nil
}

// writer makes the (Dim, in) matrix v Qt W = H diag(s) W: each column y becomes H (s y).
func (w *rotated) writer(v []float32, in int) {
	d := w.r.Dim
	col := make([]float32, d)
	for c := 0; c < in; c++ {
		for i := 0; i < d; i++ {
			col[i] = v[i*in+c] * w.signs[i]
		}
		w.hadamard(col)
		for i := 0; i < d; i++ {
			v[i*in+c] = col[i]
		}
	}
}

// hadamard multiplies x by the normalized Walsh-Hadamard matrix of each block (in float64).
func (w *rotated) hadamard(x []float32) {
	b := w.block
	buf := make([]float64, b)
	scale := 1 / math.Sqrt(float64(b))
	for o := 0; o < len(x); o += b {
		for i := range buf {
			buf[i] = float64(x[o+i])
		}
		for h := 1; h < b; h *= 2 {
			for i := 0; i < b; i += 2 * h {
				for j := i; j < i+h; j++ {
					a, c := buf[j], buf[j+h]
					buf[j], buf[j+h] = a+c, a-c
				}
			}
		}
		for i := range buf {
			x[o+i] = float32(buf[i] * scale)
		}
	}
}
