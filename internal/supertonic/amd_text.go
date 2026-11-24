package supertonic

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/llm/iron"
	"github.com/shibukawa/bonsaiai/internal/safetensors"
	"github.com/shibukawa/bonsaiai/internal/sgemm"
	"github.com/shibukawa/bonsaiai/internal/vecops"
)

// The text's two graphs on AMD's NPU (tools/amd/iron_supertonic.py's TextGraph): the text
// encoder and the duration predictor's sentence encoder, each of the Sizes' texts. The windowed
// relative attention of their attention encoders' layers (VITS's) is the host's, between two
// commands (relAttention); the duration predictor's last layers (proj_out, the predictor, exp)
// too.

// relWindow is the relative attention's window: positions -4 .. 4.
const relWindow = 4

// amdText is the text's graphs of each size and what the host computes of them.
type amdText struct {
	enc, dp         []*amdTextSize
	encRows, dpRows []float32 // the embedding tables, (rows, width)
	// relK, relV: by graph kind, each layer's relative embeddings (9, D)
	relK, relV map[string][][]float32
	// the duration predictor's sentence token, proj_out (64, 64) and predictor
	token, proj, l0w, l0b, l1w, l1b []float32
	prelu                           float32
}

type amdTextSize struct {
	kind string
	g    *iron.Graph
	meta struct {
		T        int `json:"T"`
		Rows     int `json:"rows"`
		P        int `json:"P"`
		R        int `json:"R"`
		Channels int `json:"channels"`
		Heads    int `json:"heads"`
		Layers   int `json:"layers"`
		Style    int `json:"style_rows"`
	}
	n int // the call's valid rows (the host's attention reads them)
}

func textDir(dir, kind string, s Size) string {
	return filepath.Join(dir, fmt.Sprintf("%s-t%d", kind, s.T))
}

func openAMDText(ctx context.Context, amd *bonsaiai.Device, c Config, dir string) (*amdText, error) {
	t := &amdText{relK: map[string][][]float32{}, relV: map[string][][]float32{}}
	var err error
	if t.dpRows, err = table(filepath.Join(c.Dir, "onnx", Graphs.Duration), dpTable, dpWidth); err != nil {
		return nil, err
	}
	if t.encRows, err = table(filepath.Join(c.Dir, "onnx", Graphs.Encoder), encTable, TextDim); err != nil {
		return nil, err
	}
	f, err := safetensors.Open(filepath.Join(dir, "host.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	read := func(name string) ([]float32, error) {
		v, err := f.Float32(name)
		if err != nil {
			return nil, fmt.Errorf("supertonic: %s's host.safetensors: %w (export it again)", dir, err)
		}
		return v, nil
	}
	for _, x := range []struct {
		name string
		to   *[]float32
	}{{"dp.token", &t.token}, {"dp.proj", &t.proj}, {"dp.l0.w", &t.l0w}, {"dp.l0.b", &t.l0b}, {"dp.l1.w", &t.l1w}, {"dp.l1.b", &t.l1b}} {
		if *x.to, err = read(x.name); err != nil {
			return nil, err
		}
	}
	pr, err := read("dp.prelu")
	if err != nil {
		return nil, err
	}
	t.prelu = pr[0]
	fail := func(err error) (*amdText, error) {
		t.Close()
		return nil, err
	}
	for _, kind := range []string{"encoder", "duration"} {
		for _, size := range Sizes {
			p := &amdTextSize{kind: kind}
			if p.g, err = iron.OpenGraph(ctx, amd, textDir(dir, kind, size)); err != nil {
				return fail(fmt.Errorf("supertonic: the %s of %d characters: %w", kind, size.T, err))
			}
			if kind == "encoder" {
				t.enc = append(t.enc, p)
			} else {
				t.dp = append(t.dp, p)
			}
			if err := json.Unmarshal(p.g.Manifest.Meta, &p.meta); err != nil || p.meta.Layers == 0 || p.meta.Heads == 0 {
				return fail(fmt.Errorf("supertonic: %s has no layout in its meta: export it again", textDir(dir, kind, size)))
			}
			p.g.After = t.after(p)
		}
		layers := t.enc[0].meta.Layers
		if kind == "duration" {
			layers = t.dp[0].meta.Layers
		}
		for l := range layers {
			k, err := read(fmt.Sprintf("%s.rel_k.%d", kind, l))
			if err != nil {
				return fail(err)
			}
			v, err := read(fmt.Sprintf("%s.rel_v.%d", kind, l))
			if err != nil {
				return fail(err)
			}
			t.relK[kind] = append(t.relK[kind], k)
			t.relV[kind] = append(t.relV[kind], v)
		}
	}
	return t, nil
}

// after is a graph's host step: after command l (of its layers), layer l's relative attention
// from the state qkv into the state att.
func (t *amdText) after(p *amdTextSize) func(int) error {
	return func(l int) error {
		if l >= p.meta.Layers {
			return nil
		}
		raw, from, _, err := p.g.Named("qkv")
		if err != nil {
			return err
		}
		if err := from(); err != nil {
			return err
		}
		C, R := p.meta.Channels, p.meta.R
		qkv := make([]float32, R*3*C)
		vecops.BF16ToF32(qkv, raw[:2*len(qkv)])
		att := make([]float32, R*C)
		relAttention(qkv, att, p.n, p.meta.Heads, C/p.meta.Heads, t.relK[p.kind][l], t.relV[p.kind][l])
		out, _, to, err := p.g.Named("att")
		if err != nil {
			return err
		}
		putBF16(out, att)
		return to()
	}
}

// relAttention is VITS's windowed relative attention of n rows: qkv (rows, 3 H D) the
// projections (q divided by sqrt(D)), ek and ev (9, D) the relative embeddings; out (rows, H D),
// its rows past n left zero.
func relAttention(qkv, out []float32, n, H, D int, ek, ev []float32) {
	C := H * D
	ld := 3 * C
	clear(out)
	s := make([]float32, n*n)
	for h := range H {
		q, k, v := qkv[h*D:], qkv[C+h*D:], qkv[2*C+h*D:]
		clear(s)
		sgemm.MulAdd(n, n, D, sgemm.A{Data: q, Stride: ld}, sgemm.Transposed{Data: k, Stride: ld, N: n}, s, n, 0)
		for i := range n {
			qi := q[i*ld : i*ld+D]
			row := s[i*n : (i+1)*n]
			for r := range 2*relWindow + 1 {
				if j := i + r - relWindow; j >= 0 && j < n {
					var acc float32
					for d, x := range qi {
						acc += x * ek[r*D+d]
					}
					row[j] += acc
				}
			}
			softmaxRow(row)
		}
		o := out[h*D:]
		sgemm.MulAdd(n, D, n, sgemm.A{Data: s, Stride: n}, sgemm.Matrix{Data: v, Stride: ld, N: D}, o, C, 0)
		for i := range n {
			oi := o[i*C : i*C+D]
			for r := range 2*relWindow + 1 {
				if j := i + r - relWindow; j >= 0 && j < n {
					w := s[i*n+j]
					for d := range oi {
						oi[d] += w * ev[r*D+d]
					}
				}
			}
		}
	}
}

func softmaxRow(row []float32) {
	m := row[0]
	for _, x := range row {
		m = max(m, x)
	}
	var sum float64
	for i, x := range row {
		e := math.Exp(float64(x - m))
		row[i] = float32(e)
		sum += e
	}
	for i := range row {
		row[i] = float32(float64(row[i]) / sum)
	}
}

// size is the graphs (of sizes) whose text holds T characters, nil for none.
func textSize(sizes []*amdTextSize, T int) *amdTextSize {
	for _, p := range sizes {
		if T <= p.meta.T {
			return p
		}
	}
	return nil
}

// Encode is the text encoder's output (TextDim, T) of ids and a voice's style (50, 256).
func (t *amdText) Encode(ctx context.Context, ids []int64, style []float32) ([]float32, error) {
	T := len(ids)
	p := textSize(t.enc, T)
	if p == nil {
		return nil, ErrTooLong
	}
	m := p.meta
	x, mask := make([]float32, m.Rows*m.Channels), make([]float32, m.Rows*m.Channels)
	for j, id := range ids {
		r := (m.P + j) * m.Channels
		copy(x[r:r+m.Channels], t.encRows[int(id)*TextDim:(int(id)+1)*TextDim])
		for c := range m.Channels {
			mask[r+c] = 1
		}
	}
	st := make([]float32, m.Style*256)
	copy(st, style)
	for _, in := range []struct {
		name string
		v    []float32
	}{{"x", x}, {"mask", mask}, {"style", st}} {
		if err := setBF16(p.g, in.name, in.v); err != nil {
			return nil, err
		}
	}
	p.n = T
	if err := p.g.Run(ctx, map[string]int64{"ns": 50}); err != nil {
		return nil, fmt.Errorf("supertonic: the text encoder: %w", err)
	}
	y, err := output(p.g)
	if err != nil {
		return nil, err
	}
	emb := make([]float32, TextDim*T)
	for j := range T {
		for c := range TextDim {
			emb[c*T+j] = y[j*TextDim+c]
		}
	}
	return emb, nil
}

// Duration is the speech's length in seconds of ids and a voice's duration style (8 x 16): the
// sentence encoder's token row through proj_out, then the predictor with the style, exp.
func (t *amdText) Duration(ctx context.Context, ids []int64, style []float32) (float32, error) {
	T := len(ids)
	p := textSize(t.dp, T)
	if p == nil || T+1 > p.meta.R {
		return 0, ErrTooLong
	}
	m := p.meta
	C := m.Channels
	x, mask := make([]float32, m.Rows*C), make([]float32, m.Rows*C)
	copy(x[m.P*C:(m.P+1)*C], t.token)
	for j, id := range ids {
		r := (m.P + 1 + j) * C
		copy(x[r:r+C], t.dpRows[int(id)*dpWidth:(int(id)+1)*dpWidth])
	}
	for r := m.P; r <= m.P+T; r++ {
		for c := range C {
			mask[r*C+c] = 1
		}
	}
	for _, in := range []struct {
		name string
		v    []float32
	}{{"x", x}, {"mask", mask}} {
		if err := setBF16(p.g, in.name, in.v); err != nil {
			return 0, err
		}
	}
	p.n = T + 1
	if err := p.g.Run(ctx, nil); err != nil {
		return 0, fmt.Errorf("supertonic: the duration predictor: %w", err)
	}
	y, err := output(p.g)
	if err != nil {
		return 0, err
	}
	h := y[:C]
	z := make([]float64, 0, C+len(style))
	for o := range C {
		var acc float64
		for i, v := range h {
			acc += float64(t.proj[o*C+i]) * float64(v)
		}
		z = append(z, acc)
	}
	for _, v := range style {
		z = append(z, float64(v))
	}
	var out float64 = float64(t.l1b[0])
	for o := range len(t.l0b) {
		acc := float64(t.l0b[o])
		for i, v := range z {
			acc += float64(t.l0w[o*len(z)+i]) * v
		}
		if acc < 0 {
			acc *= float64(t.prelu)
		}
		out += float64(t.l1w[o]) * acc
	}
	return float32(math.Exp(out)), nil
}

func (t *amdText) Close() error {
	for _, p := range append(append([]*amdTextSize(nil), t.enc...), t.dp...) {
		if p.g != nil {
			p.g.Close()
		}
	}
	return nil
}
