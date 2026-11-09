package pitch

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// synthetic is a Model with random weights of SwiftF0's shapes (no front end).
func synthetic() *Model {
	r := rand.New(rand.NewPCG(1, 2))
	t := map[string]tensor{}
	add := func(c conv, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		fan := float64(n / shape[0])
		w := tensor{shape: shape, vals: make([]float64, n)}
		for i := range w.vals {
			w.vals[i] = r.NormFloat64() / math.Sqrt(fan)
		}
		b := tensor{shape: []int{shape[0]}, vals: make([]float64, shape[0])}
		for i := range b.vals {
			b.vals[i] = 0.3 * r.NormFloat64()
		}
		t[c.weight], t[c.bias] = w, b
	}
	add(stem, 12, 3, 3, 3)
	kernels := [][2]int{{7, 3}, {7, 1}, {5, 1}, {3, 1}, {3, 1}, {7, 3}}
	for i, blk := range body {
		add(blk[0], 12, 12, kernels[i][0], kernels[i][1])
		add(blk[1], 12, 12, kernels[i][0], kernels[i][1])
	}
	add(head1, 4, 12, 3, 3)
	add(head2, 4, 4, 5, 1)
	add(pitchConv, 1, 4, 1, 1)
	add(voice1, 16, 5, 3)
	add(voice2, 1, 16, 1)
	return &Model{t: t, source: "synthetic"}
}

// refConv is a zero-padded 2 D convolution in float64 of x (c, h, w) with the weights rounded to
// fp16, as the program holds them.
func refConv(m *Model, x []float64, c, h, w int, cv conv) ([]float64, int, int) {
	wt, b := m.t[cv.weight], m.t[cv.bias]
	sh := wt.shape
	if len(sh) == 3 {
		sh = []int{sh[0], sh[1], 1, sh[2]}
	}
	co, kh, kw := sh[0], sh[2], sh[3]
	oh := h + cv.pad[0] + cv.pad[1] - cv.dil[0]*(kh-1)
	ow := w + cv.pad[2] + cv.pad[3] - cv.dil[1]*(kw-1)
	out := make([]float64, co*oh*ow)
	for o := range co {
		for i := range oh {
			for j := range ow {
				s := r16(b.vals[o])
				for ci := range c {
					for a := range kh {
						y := i - cv.pad[0] + a*cv.dil[0]
						if y < 0 || y >= h {
							continue
						}
						for e := range kw {
							xx := j - cv.pad[2] + e*cv.dil[1]
							if xx < 0 || xx >= w {
								continue
							}
							s += r16(wt.vals[((o*c+ci)*kh+a)*kw+e]) * x[(ci*h+y)*w+xx]
						}
					}
				}
				out[(o*oh+i)*ow+j] = s
			}
		}
	}
	return out, oh, ow
}

func relu(x []float64) []float64 {
	for i, v := range x {
		x[i] = math.Max(v, 0)
	}
	return x
}

func r16(x float64) float64 {
	return float64(mlpackage.Float16Value(mlpackage.Float16Bits(float32(x))))
}

// cnn is the graph's CNN from the features x (3, 128, T): y (96, T), from the ONNX graph's nodes
// in float64.
func cnn(m *Model, x []float64, T int) []float64 {
	h, H, W := refConv(m, x, 3, Bands, T, stem)
	relu(h)
	for _, blk := range body {
		a, _, _ := refConv(m, h, 12, H, W, blk[0])
		r, _, _ := refConv(m, relu(a), 12, H, W, blk[1])
		for i := range h {
			h[i] = math.Max(h[i]+r[i], 0)
		}
	}
	low := make([]float64, 12*headBands*T)
	for c := range 12 {
		copy(low[c*headBands*T:(c+1)*headBands*T], h[c*Bands*T:c*Bands*T+headBands*T])
	}
	h1, H1, _ := refConv(m, low, 12, headBands, T, head1)
	relu(h1)
	h2, _, _ := refConv(m, h1, 4, H1, T, head2)
	v := make([]float64, 5*T)
	for t := range T {
		for c := range 4 {
			mx := math.Inf(-1)
			for f := range H1 {
				mx = math.Max(mx, h2[(c*H1+f)*T+t])
			}
			v[c*T+t] = mx
		}
		var s float64
		for c := range 3 {
			for f := range Bands {
				s += x[(c*Bands+f)*T+t]
			}
		}
		v[4*T+t] = s / (3 * Bands)
	}
	v1, _, _ := refConv(m, v, 5, 1, T, voice1)
	relu(v1)
	v2, _, _ := refConv(m, v1, 16, 1, T, voice2)
	p, _, _ := refConv(m, h1, 4, H1, T, pitchConv)
	return append(p, v2...)
}

// program builds the CNN for frames and loads it on the CPU interpreter.
func program(t *testing.T, m *Model, frames int) func(x, mask []float32) []float32 {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "p.mlpackage")
	if _, err := llm.WritePart(dir, &Program{Frames: frames}, m.Weights(), llm.Options{Format: "fp16"}, nil); err != nil {
		t.Fatal(err)
	}
	prog, f, err := milcpu.OpenPackage(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	return func(x, mask []float32) []float32 {
		res, err := prog.Run(map[string]milcpu.Tensor{
			"x":    {Shape: []int{1, 3, Bands, frames}, Data: x},
			"mask": {Shape: []int{1, 1, 1, frames}, Data: mask},
		}, milcpu.Options{})
		if err != nil {
			t.Fatal(err)
		}
		return res["y"].Data
	}
}

func relErr(got []float32, want []float64) float64 {
	var se, sw float64
	for i := range want {
		d := float64(got[i]) - want[i]
		se += d * d
		sw += want[i] * want[i]
	}
	return math.Sqrt(se / sw)
}

func ones(n, of int) []float32 {
	m := make([]float32, of)
	for i := range n {
		m[i] = 1
	}
	return m
}

// The program in fp16 on the CPU gives the float64 CNN's logits; a clip shorter than the program,
// zero padded and masked, gives those of a program of its own length.
func TestProgramMatchesReference(t *testing.T) {
	m := synthetic()
	r := rand.New(rand.NewPCG(3, 4))
	const T, short, long = 20, 13, 24
	x := make([]float64, 3*Bands*T)
	for i := range x {
		x[i] = 2*r.NormFloat64() - 3
	}
	x32 := make([]float32, len(x))
	for i, v := range x {
		x32[i] = float32(v)
		x[i] = r16(v) // the program's input is fp16
	}
	want := cnn(m, x, T)
	got := program(t, m, T)(x32, ones(T, T))
	if e := relErr(got, want); e > 2e-3 {
		t.Errorf("relative error %g against the float64 CNN", e)
	} else {
		t.Logf("relative error %.2g", e)
	}
	// the first short frames of x, in a program of long frames
	xs := make([]float64, 3*Bands*short)
	in := make([]float32, 3*Bands*long)
	for c := range 3 * Bands {
		for f := range short {
			xs[c*short+f] = x[c*T+f]
			in[c*long+f] = x32[c*T+f]
		}
	}
	want = cnn(m, xs, short)
	y := program(t, m, long)(in, ones(short, long))
	got = make([]float32, (Bins+1)*short)
	for row := range Bins + 1 {
		copy(got[row*short:(row+1)*short], y[row*long:row*long+short])
	}
	if e := relErr(got, want); e > 2e-3 {
		t.Errorf("a clip of %d frames in a program of %d: relative error %g", short, long, e)
	}
}

func TestFFT(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	const n = 64
	a := make([]complex128, n)
	for i := range a {
		a[i] = complex(r.NormFloat64(), 0)
	}
	got := append([]complex128(nil), a...)
	fft(got)
	for k := range n {
		var s complex128
		for j := range n {
			ang := -2 * math.Pi * float64(k*j) / n
			s += a[j] * complex(math.Cos(ang), math.Sin(ang))
		}
		if d := got[k] - s; math.Hypot(real(d), imag(d)) > 1e-9 {
			t.Fatalf("bin %d: %v, want %v", k, got[k], s)
		}
	}
}

func readBin[T float32 | float64](t *testing.T, path string) []T {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var z T
	size := int(binary.Size(z))
	out := make([]T, len(raw)/size)
	if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, out); err != nil {
		t.Fatal(err)
	}
	return out
}

// With the real model (HWNPU_SWIFTF0 = model.onnx) and a reference of tools/ref/ref_swiftf0.py
// (HWNPU_SWIFTF0_REF = its OUT_PREFIX), the features, the logits and the pitch are the graph's.
func TestMatchesONNX(t *testing.T) {
	path, ref := os.Getenv("HWNPU_SWIFTF0"), os.Getenv("HWNPU_SWIFTF0_REF")
	if path == "" || ref == "" {
		t.Skip("HWNPU_SWIFTF0 and HWNPU_SWIFTF0_REF are not set")
	}
	m, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		F    int     `json:"frames"`
		W    int     `json:"window_frames"`
		FMin float64 `json:"fmin"`
		FMax float64 `json:"fmax"`
	}
	raw, err := os.ReadFile(ref + ".json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	audio := readBin[float32](t, ref+".audio.bin")
	wantF := readBin[float32](t, ref+".features.bin")
	wantY := readBin[float32](t, ref+".logits.bin")
	want := readBin[float64](t, ref+".pitch.bin")
	first := audio // the reference's first window
	if meta.F > WindowFrames {
		first = audio[:(WindowFrames+LookaheadFrames)*Hop]
	}
	f := m.Features(first)
	f64 := make([]float64, len(wantF))
	for i, v := range wantF {
		f64[i] = float64(v)
	}
	if len(f) != len(f64) {
		t.Fatalf("%d features, want %d", len(f), len(f64))
	}
	// bands above e^-6 of magnitude: in the weaker ones the graph's float32 transform (a DFT
	// factored as matrix products) is off by its rounding of the frame's strong bands (0.04 at
	// e^-8.5 in a frame of a 41 s clip), which the logits do not feel
	var worst float64
	for i := range f {
		if f64[i] > -6 {
			worst = math.Max(worst, math.Abs(f[i]-f64[i]))
		}
	}
	t.Logf("features: largest difference %.3g (bands above -6)", worst)
	if worst > 2e-3 {
		t.Errorf("features: largest difference %g", worst)
	}
	run := program(t, m, ProgramFrames)
	y := run(func() []float32 {
		in := make([]float32, 3*Bands*ProgramFrames)
		W := meta.W
		for c := range 3 * Bands {
			for t := range W {
				in[c*ProgramFrames+t] = float32(f[c*W+t])
			}
		}
		return in
	}(), ones(meta.W, ProgramFrames))
	got := make([]float32, (Bins+1)*meta.W)
	for row := range Bins + 1 {
		copy(got[row*meta.W:(row+1)*meta.W], y[row*ProgramFrames:row*ProgramFrames+meta.W])
	}
	wy := make([]float64, len(wantY))
	for i, v := range wantY {
		wy[i] = float64(v)
	}
	if e := relErr(got, wy); e > 5e-3 {
		t.Errorf("logits: relative error %g", e)
	} else {
		t.Logf("logits: relative error %.3g", e)
	}
	res, err := m.Detect(audio, meta.FMin, meta.FMax, func(x, mask []float32) ([]float32, error) { return run(x, mask), nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(res.PitchHz) != meta.F {
		t.Fatalf("%d frames, want %d", len(res.PitchHz), meta.F)
	}
	var cents, agree, loud float64
	for i := range meta.F {
		wp, wc, wl := want[i], want[meta.F+i], want[2*meta.F+i]
		if (res.Confidence[i] >= 0.5) == (wc >= 0.5) {
			agree++
		}
		if wc >= 0.5 && res.Confidence[i] >= 0.5 {
			cents = math.Max(cents, math.Abs(1200*math.Log2(res.PitchHz[i]/wp)))
		}
		loud = math.Max(loud, math.Abs(res.LoudnessDB[i]-wl))
	}
	t.Logf("%d frames: voicing the same in %.0f, largest pitch difference of the voiced %.2f cents, loudness %.2g dB", meta.F, agree, cents, loud)
	if agree < float64(meta.F)-1 || cents > 5 || loud > 1e-6 {
		t.Errorf("voicing the same in %.0f of %d, %.2f cents, loudness %g dB", agree, meta.F, cents, loud)
	}
}
