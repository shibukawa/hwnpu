package depth

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
)

// MultiConfig is model.json of an export of tools/amd/iron_da3.py --views: DA3's any-view
// model (DA3-SMALL, DA3-BASE) on Views pictures of one size at once.
type MultiConfig struct {
	Config
	Views  int `json:"views"`
	Tall   int `json:"tall"`  // the backbone's rows: the views' tokens one after another, then zeros
	Width  int `json:"width"` // the size every picture is stretched to
	Height int `json:"height"`
	CamDec []struct {
		Name string `json:"name"`
		Out  int    `json:"out"`
		Inp  int    `json:"inp"`
	} `json:"cam_dec"`
}

// IsMultiExport reports whether dir is an export of tools/amd/iron_da3.py --views.
func IsMultiExport(dir string) bool {
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return false
	}
	var c MultiConfig
	return json.Unmarshal(b, &c) == nil && c.Kind == "depth-mv"
}

// Multi is a loaded multi-view export.
type Multi struct {
	MultiConfig
	vitA, vitB, head *iron.Graph
	camDec           [][]float32 // each layer's weight (out x inp) then bias
	// Ref, when 0 or more, is the reference view (the official strategies "first" 0, "middle"
	// Views / 2); -1 (OpenMulti's) picks it as DA3 does ("saddle_balanced", 3 views or more),
	// which near-identical views can tie.
	Ref  int
	Logf func(format string, args ...any)
}

// Camera is a view's camera as DA3's cam_dec gives it: World-to-camera [R | t] (3 x 4, row
// major) and the intrinsics fx, fy, cx, cy at the processed size (Width x Height).
type Camera struct {
	W2C            [12]float64
	Fx, Fy, Cx, Cy float64
}

// OpenMulti loads the export in dir onto dev.
func OpenMulti(ctx context.Context, dev *hwnpu.Device, dir string) (*Multi, error) {
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	m := &Multi{Ref: -1}
	if err := json.Unmarshal(b, &m.MultiConfig); err != nil {
		return nil, fmt.Errorf("depth: %w", err)
	}
	if m.Kind != "depth-mv" || m.Views < 1 {
		return nil, fmt.Errorf("depth: %s is not an export of iron_da3.py --views", dir)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "cam_dec.bin"))
	if err != nil {
		return nil, err
	}
	off := 0
	for _, l := range m.CamDec {
		for _, n := range []int{l.Out * l.Inp, l.Out} {
			xs := make([]float32, n)
			for i := range xs {
				xs[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*(off+i):]))
			}
			off += n
			m.camDec = append(m.camDec, xs)
		}
	}
	open := func(n string) (*iron.Graph, error) { return iron.OpenGraph(ctx, dev, filepath.Join(dir, n)) }
	if m.vitA, err = open("vit_a"); err == nil {
		if m.vitB, err = open("vit_b"); err == nil {
			m.head, err = open("head")
		}
	}
	if err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
}

// Close releases the device memory.
func (m *Multi) Close() error {
	var err error
	for _, g := range []*iron.Graph{m.vitA, m.vitB, m.head} {
		if g != nil {
			if e := g.Close(); err == nil {
				err = e
			}
		}
	}
	return err
}

// Estimate runs the model on imgs (Views of them, each stretched to the export's size): each
// view's depth (relative, one scale for all the views) and confidence, and its camera.
func (m *Multi) Estimate(ctx context.Context, imgs []image.Image) ([]*Result, []Camera, error) {
	S, t1, E := m.Views, m.Rows, m.Model.Hidden
	if len(imgs) != S {
		return nil, nil, fmt.Errorf("depth: the export takes %d pictures at once, got %d", S, len(imgs))
	}
	W, H := m.Width, m.Height
	t0 := time.Now()
	pat := make([]float32, m.Tall*patchIn)
	for i, img := range imgs {
		rows := PatchRows(lfm3vl.Resize(img, W, H, "bicubic"), W, H, m.Patch, m.Tokens, m.Mean, m.Std)
		copy(pat[i*t1*patchIn:(i+1)*t1*patchIn], rows[:t1*patchIn])
	}
	vals := map[string]int64{"nk": int64(t1), "na": int64(S * t1)}
	if err := m.vitA.SetInput("patches", toBF16(pat)); err != nil {
		return nil, nil, err
	}
	if err := m.vitA.Run(ctx, vals); err != nil {
		return nil, nil, fmt.Errorf("depth: vit_a: %w", err)
	}
	x, err := m.vitA.Output(m.vitA.Outputs()[0].Name)
	if err != nil {
		return nil, nil, err
	}
	// the reference view (DA3's saddle_balanced, 3 views or more) first
	cls := make([][]float32, S)
	for i := range cls {
		cls[i] = bf16Row(x, i*t1, E)
	}
	ref := 0
	switch {
	case m.Ref >= 0 && m.Ref < S:
		ref = m.Ref
	case S >= 3:
		ref = SaddleBalanced(cls)
	}
	order := []int{ref}
	for i := 0; i < S; i++ {
		if i != ref {
			order = append(order, i)
		}
	}
	xb := make([]byte, len(x))
	for j, i := range order {
		copy(xb[2*j*t1*E:2*(j+1)*t1*E], x[2*i*t1*E:2*(i+1)*t1*E])
	}
	if err := m.vitB.SetInput("x", xb); err != nil {
		return nil, nil, err
	}
	t1s := time.Now()
	if err := m.vitB.Run(ctx, vals); err != nil {
		return nil, nil, fmt.Errorf("depth: vit_b: %w", err)
	}
	outs := m.vitB.Outputs() // the four features, the last local, the last global
	feats := make([][]byte, 4)
	for k := 0; k < 4; k++ {
		f, err := m.vitB.Output(outs[k].Name)
		if err != nil {
			return nil, nil, err
		}
		feats[k] = append([]byte(nil), f...)
	}
	local, err := m.vitB.Output(outs[4].Name)
	if err != nil {
		return nil, nil, err
	}
	last, err := m.vitB.Output(outs[5].Name)
	if err != nil {
		return nil, nil, err
	}
	results := make([]*Result, S)
	cams := make([]Camera, S)
	D := 2 * E
	in := make([]byte, 2*m.Tokens*D)
	t2 := time.Now()
	for j, i := range order {
		cams[i] = m.camera(append(bf16Row(local, j*t1, E), bf16Row(last, j*t1, E)...))
		for k := 0; k < 4; k++ {
			clear(in)
			copy(in, feats[k][2*j*t1*D:2*(j+1)*t1*D])
			if err := m.head.SetInput(fmt.Sprintf("f%d", k), in); err != nil {
				return nil, nil, err
			}
		}
		if err := m.head.Run(ctx, nil); err != nil {
			return nil, nil, fmt.Errorf("depth: head: %w", err)
		}
		out, err := m.head.Output(m.head.Outputs()[0].Name)
		if err != nil {
			return nil, nil, err
		}
		r := &Result{W: W, H: H, Depth: make([]float32, W*H), Conf: make([]float32, W*H)}
		for p := range r.Depth {
			r.Depth[p] = float32(math.Exp(float64(fromBF16(binary.LittleEndian.Uint16(out[2*64*p:])))))
			r.Conf[p] = float32(math.Exp(float64(fromBF16(binary.LittleEndian.Uint16(out[2*64*p+2:]))))) + 1
		}
		results[i] = r
	}
	if m.Logf != nil {
		m.Logf("depth: %d views at %dx%d (reference %d), backbone %.2f s (the second part %.2f s), heads %.2f s", S, W, H, ref,
			t2.Sub(t0).Seconds(), t2.Sub(t1s).Seconds(), time.Since(t2).Seconds())
	}
	return results, cams, nil
}

func bf16Row(b []byte, row, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = fromBF16(binary.LittleEndian.Uint16(b[2*(row*n+i):]))
	}
	return out
}

// SaddleBalanced is DA3's reference view (select_reference_view, "saddle_balanced") of the
// views' class tokens: the view whose mean cosine to the others, token norm and normalized
// token variance are each nearest the middle of the views' range.
func SaddleBalanced(cls [][]float32) int {
	S := len(cls)
	unit := make([][]float64, S)
	nrm := make([]float64, S)
	vr := make([]float64, S)
	for i, c := range cls {
		var s float64
		for _, v := range c {
			s += float64(v) * float64(v)
		}
		nrm[i] = math.Sqrt(s)
		unit[i] = make([]float64, len(c))
		var mean float64
		for k, v := range c {
			unit[i][k] = float64(v) / nrm[i]
			mean += unit[i][k]
		}
		mean /= float64(len(c))
		for _, u := range unit[i] {
			vr[i] += (u - mean) * (u - mean)
		}
		vr[i] /= float64(len(c) - 1) // torch's var: unbiased
	}
	sim := make([]float64, S)
	for i := range unit {
		for j := range unit {
			if i != j {
				var d float64
				for k := range unit[i] {
					d += unit[i][k] * unit[j][k]
				}
				sim[i] += d
			}
		}
		sim[i] /= float64(S - 1)
	}
	norm := func(m []float64) []float64 {
		lo, hi := m[0], m[0]
		for _, v := range m {
			lo, hi = math.Min(lo, v), math.Max(hi, v)
		}
		out := make([]float64, len(m))
		for i, v := range m {
			out[i] = (v - lo) / (hi - lo + 1e-8)
		}
		return out
	}
	a, b, c := norm(sim), norm(nrm), norm(vr)
	best, bi := math.Inf(1), 0
	for i := range a {
		if s := math.Abs(a[i]-0.5) + math.Abs(b[i]-0.5) + math.Abs(c[i]-0.5); s < best {
			best, bi = s, i
		}
	}
	return bi
}

// camera runs cam_dec (Linear, ReLU, Linear, ReLU, then fc_t, fc_qvec, fc_fov + ReLU) on a camera
// token (2 hidden) and converts the pose encoding as DA3's pose_encoding_to_extri_intri and
// affine_inverse do.
func (m *Multi) camera(tok []float32) Camera {
	lin := func(k int, x []float32, relu bool) []float32 {
		w, b := m.camDec[2*k], m.camDec[2*k+1]
		l := m.CamDec[k]
		y := make([]float32, l.Out)
		for o := 0; o < l.Out; o++ {
			s := float64(b[o])
			for i := 0; i < l.Inp; i++ {
				s += float64(w[o*l.Inp+i]) * float64(x[i])
			}
			if relu && s < 0 {
				s = 0
			}
			y[o] = float32(s)
		}
		return y
	}
	h := lin(1, lin(0, tok, true), true)
	t, q, fov := lin(2, h, false), lin(3, h, false), lin(4, h, true)
	// quaternion XYZW (scalar last) to a rotation
	i, j, k, r := float64(q[0]), float64(q[1]), float64(q[2]), float64(q[3])
	two := 2 / (i*i + j*j + k*k + r*r)
	R := [9]float64{
		1 - two*(j*j+k*k), two * (i*j - k*r), two * (i*k + j*r),
		two * (i*j + k*r), 1 - two*(i*i+k*k), two * (j*k - i*r),
		two * (i*k - j*r), two * (j*k + i*r), 1 - two*(i*i+j*j),
	}
	var c Camera
	// c2w = [R | t]; w2c = [R^T | -R^T t]
	for a := 0; a < 3; a++ {
		var s float64
		for b := 0; b < 3; b++ {
			c.W2C[a*4+b] = R[b*3+a]
			s -= R[b*3+a] * float64(t[b])
		}
		c.W2C[a*4+3] = s
	}
	W, H := float64(m.Width), float64(m.Height)
	c.Fy = H / 2 / math.Max(math.Tan(float64(fov[0])/2), 1e-6)
	c.Fx = W / 2 / math.Max(math.Tan(float64(fov[1])/2), 1e-6)
	c.Cx, c.Cy = W/2, H/2
	return c
}
