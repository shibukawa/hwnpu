package llm

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
)

type memWeights map[string][]float32

func (m memWeights) shape(name string) []int {
	switch name {
	case "emb", "head":
		return []int{5, 128}
	case "read":
		return []int{32, 128}
	case "write":
		return []int{128, 32}
	}
	return []int{128}
}

func (m memWeights) Get(name string) ([]float32, []int, error) {
	v, ok := m[name]
	if !ok {
		return nil, nil, fmt.Errorf("no %s", name)
	}
	return slices.Clone(v), m.shape(name), nil
}

func (m memWeights) Rows(name string, begin, end int) ([]float32, error) {
	v, s, err := m.Get(name)
	if err != nil {
		return nil, err
	}
	return v[begin*s[1] : end*s[1]], nil
}

func (memWeights) Source() string { return "mem" }
func (memWeights) Close() error   { return nil }

// tinyModel is a pre-norm model of one block: r = emb[t]; r += write silu(read norm1(r));
// logits = head norm2(r), the norms zero centered.
func tinyModel(t *testing.T, w Weights, tok int) []float64 {
	t.Helper()
	get := func(n string) []float32 {
		v, _, err := w.Get(n)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	r64 := func(v []float32) []float64 {
		o := make([]float64, len(v))
		for i, x := range v {
			o[i] = float64(x)
		}
		return o
	}
	rows, err := w.Rows("emb", tok, tok+1)
	if err != nil {
		t.Fatal(err)
	}
	r := r64(rows)
	norm := func(x []float64, g []float32) []float64 {
		var ss float64
		for _, v := range x {
			ss += v * v
		}
		s := 1 / math.Sqrt(ss/float64(len(x))+1e-6)
		o := make([]float64, len(x))
		for i := range x {
			o[i] = x[i] * s * (1 + float64(g[i]))
		}
		return o
	}
	matvec := func(m []float32, rows, cols int, x []float64) []float64 {
		o := make([]float64, rows)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				o[i] += float64(m[i*cols+j]) * x[j]
			}
		}
		return o
	}
	a := matvec(get("read"), 32, 128, norm(r, get("n1")))
	for i, v := range a {
		a[i] = v / (1 + math.Exp(-v))
	}
	add := matvec(get("write"), 128, 32, a)
	for i := range r {
		r[i] += add[i]
	}
	return matvec(get("head"), 5, 128, norm(r, get("n2")))
}

func TestRotateKeepsOutput(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	m := memWeights{}
	for _, n := range []string{"emb", "head", "read", "write", "n1", "n2"} {
		s := m.shape(n)
		v := make([]float32, s[0]*max(1, func() int {
			if len(s) > 1 {
				return s[1]
			}
			return 1
		}()))
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		m[n] = v
	}
	m["emb"][7] = 40 // an outlier channel
	rot := Rotation{Dim: 128, Embed: "emb", Head: "head", HeadNorm: "n2", Readers: map[string]string{"read": "n1"},
		Writers: map[string]bool{"write": true}, Norms: map[string]bool{"n1": true, "n2": true}, ZeroCentered: true}
	rw, err := Rotate(m, rot, 1)
	if err != nil {
		t.Fatal(err)
	}
	for tok := 0; tok < 5; tok++ {
		want, got := tinyModel(t, m, tok), tinyModel(t, rw, tok)
		for i := range want {
			if math.Abs(want[i]-got[i]) > 1e-3*math.Max(1, math.Abs(want[i])) {
				t.Fatalf("token %d logit %d: rotated %g, unrotated %g", tok, i, got[i], want[i])
			}
		}
	}
	// the embedding's outlier is spread: no rotated value near 40
	e, _ := rw.Rows("emb", 0, 1)
	for i, v := range e {
		if math.Abs(float64(v)) > 10 {
			t.Fatalf("rotated embedding value %d is %g", i, v)
		}
	}
}

// Unrotate takes a rotated embedding row back to the checkpoint's.
func TestUnrotate(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	m := memWeights{"emb": make([]float32, 5*128)}
	for i := range m["emb"] {
		m["emb"][i] = float32(rng.NormFloat64())
	}
	rw, err := Rotate(m, Rotation{Dim: 128, Embed: "emb"}, RotationSeed)
	if err != nil {
		t.Fatal(err)
	}
	un, err := Unrotate(128, RotationSeed)
	if err != nil {
		t.Fatal(err)
	}
	row, _ := rw.Rows("emb", 2, 3)
	un(row)
	for i, v := range row {
		if math.Abs(float64(v-m["emb"][2*128+i])) > 1e-5 {
			t.Fatalf("value %d: %g, want %g", i, v, m["emb"][2*128+i])
		}
	}
}
