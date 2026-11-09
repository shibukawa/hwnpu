package music

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// On the Neural Engine (HWNPU_MRT_DEVICE: the programs' format, lut8 or fp16), the prepared
// programs stepped with the reference's tokens forced give logits close to the reference's (fp32):
// the KL of the codebooks' distributions (at temperature 1.3) and how often the most likely token
// is the same.
func TestLMOnDevice(t *testing.T) {
	dir, ref := refDirs(t)
	format := os.Getenv("HWNPU_MRT_DEVICE")
	if format == "" {
		t.Skip("HWNPU_MRT_DEVICE is not set")
	}
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "", "")
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	p, err := Prepare(ctx, Setup{Dir: dir, Repo: "google/magenta-realtime-2", Format: format, Backend: llm.BackendCoreML})
	if err != nil {
		t.Fatal(err)
	}
	lw, err := OpenWeights(map[string]string{filepath.Join(dir, "checkpoints/mrt2_small.safetensors"): "params/"})
	if err != nil {
		t.Fatal(err)
	}
	defer lw.Close()
	ts, err := openStep(ctx, dev, p.Temporal)
	if err != nil {
		t.Fatal(err)
	}
	defer ts.Close()
	ds, err := openStep(ctx, dev, p.Depth)
	if err != nil {
		t.Fatal(err)
	}
	defer ds.Close()
	t.Logf("on the NPU: %v %v", ts.onNPU, ds.onNPU)
	block := readBin[int32](t, ref+".block.bin")
	var c Conditioning
	for i := range c {
		c[i] = int(block[i]) - ReservedTokens - 1
	}
	enc, err := Encode(lw, c)
	if err != nil {
		t.Fatal(err)
	}
	l := newLM(lw, ts, ds)
	tokens := readBin[int32](t, ref+".tokens.bin")
	want := readBin[float32](t, ref+".logits.bin")
	frames := len(tokens) / Codebooks
	var kl float64
	agree := 0
	for f := range frames {
		forced := make([]int, Codebooks)
		for i := range forced {
			forced[i] = int(tokens[f*Codebooks+i])
		}
		_, lg, err := l.Step(enc, nil, forced, true)
		if err != nil {
			t.Fatal(err)
		}
		for cb := range Codebooks {
			lo := ReservedTokens + cb*CodebookSize
			w := want[(f*Codebooks+cb)*Vocab+lo : (f*Codebooks+cb)*Vocab+lo+CodebookSize]
			g := lg[cb*CodebookSize : (cb+1)*CodebookSize]
			kl += klDiv(w, g, 1.3)
			if argmax(g) == argmax(w) {
				agree++
			}
		}
	}
	n := frames * Codebooks
	t.Logf("%s: mean KL %.4f nats, the same most likely token in %d of %d", format, kl/float64(n), agree, n)
}

func klDiv(p, q []float32, temp float64) float64 {
	sm := func(v []float32) []float64 {
		out := make([]float64, len(v))
		hi := math.Inf(-1)
		for _, x := range v {
			hi = math.Max(hi, float64(x)/temp)
		}
		var z float64
		for i, x := range v {
			out[i] = math.Exp(float64(x)/temp - hi)
			z += out[i]
		}
		for i := range out {
			out[i] /= z
		}
		return out
	}
	a, b := sm(p), sm(q)
	var s float64
	for i := range a {
		if a[i] > 0 {
			s += a[i] * math.Log(a[i]/math.Max(b[i], 1e-30))
		}
	}
	return s
}

// MusicCoCa's encoders on the Neural Engine (HWNPU_MRT_DEVICE set) give the references'
// embeddings, and the text's codes.
func TestMusicCoCaOnDevice(t *testing.T) {
	if os.Getenv("HWNPU_MRT_DEVICE") == "" {
		t.Skip("HWNPU_MRT_DEVICE is not set")
	}
	p, ref := mccRef(t)
	aref := os.Getenv("HWNPU_MUSICCOCA_AUDIO_REF")
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "", "")
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	t0 := time.Now()
	st, err := OpenStyler(ctx, dev, p)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	t.Logf("loaded in %.1f s, on the NPU: %v", time.Since(t0).Seconds(), st.OnNPU())
	feed := textFeed(st.table, f64(readF32(t, ref+".ids.bin", true)), f64(readF32(t, ref+".pad.bin", false)))
	var out map[string][]float32
	for range 3 {
		t0 = time.Now()
		if out, err = st.text.run(feed); err != nil {
			t.Fatal(err)
		}
		t.Logf("text: ran in %.0f ms", float64(time.Since(t0).Microseconds())/1000)
	}
	t.Logf("text: relative error %.3g", math.Sqrt(relErr32(out["y"], readF32(t, ref+".emb.bin", false))))
	t0 = time.Now()
	emb, err := st.Text("disco funk")
	if err != nil {
		t.Fatal(err)
	}
	codes, err := st.Tokens(emb)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("text to codes: %.1f s, %v", time.Since(t0).Seconds(), codes)
	if aref == "" {
		return
	}
	feat := readF32(t, aref+".feat.bin", false)
	x := make([]float32, len(st.patches))
	for i, j := range st.patches {
		x[i] = feat[j]
	}
	for range 3 {
		t0 = time.Now()
		if out, err = st.music.run(map[string][]float32{"x": x}); err != nil {
			t.Fatal(err)
		}
		t.Logf("music: ran in %.0f ms", float64(time.Since(t0).Microseconds())/1000)
	}
	t.Logf("music: relative error %.3g", math.Sqrt(relErr32(out["y"], readF32(t, aref+".emb.bin", false))))
}

// SpectroStream's encoder on the Neural Engine (HWNPU_MRT_DEVICE set; HWNPU_MRT_DIR,
// HWNPU_MRT_ENC_REF) gives the reference's codes.
func TestEncodeAudioOnDevice(t *testing.T) {
	dir, ref := os.Getenv("HWNPU_MRT_DIR"), os.Getenv("HWNPU_MRT_ENC_REF")
	if os.Getenv("HWNPU_MRT_DEVICE") == "" || dir == "" || ref == "" {
		t.Skip("HWNPU_MRT_DEVICE, HWNPU_MRT_DIR and HWNPU_MRT_ENC_REF are not set")
	}
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "", "")
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	p, err := Prepare(ctx, Setup{Dir: dir, Repo: "google/magenta-realtime-2", Encoder: true})
	if err != nil {
		t.Fatal(err)
	}
	g, err := Open(ctx, dev, p)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	wav := readF32(t, ref+".wav.bin", false)
	t0 := time.Now()
	spec, _ := STFT(wav)
	t.Logf("STFT of %.1f s: %.2f s", float64(len(wav)/2)/SampleRate, time.Since(t0).Seconds())
	_ = spec
	codes, err := g.EncodeAudio(wav)
	if err != nil {
		t.Fatal(err)
	}
	want := readF32(t, ref+".codes.bin", true)
	same, first := 0, 0
	for f, c := range codes {
		for l, v := range c {
			if v == int(want[f*Codebooks+l]) {
				same++
				if l == 0 {
					first++
				}
			}
		}
	}
	t.Logf("on the NPU: %v; %d frames in %.2f s: %d / %d codes the same (first level %d)", g.enc.onNPU, len(codes), g.Timing.Encode.Seconds(), same, len(codes)*Codebooks, first)
}

// SpectroStream's decoder on the units of HWNPU_MRT_UNITS (gpu, cpu; "" the Neural Engine) gives
// the reference's samples of its codes (HWNPU_MRT, HWNPU_MRT_REF: mrt_ss_ref.py's prefix).
func TestDecoderOnUnits(t *testing.T) {
	if os.Getenv("HWNPU_MRT_DEVICE") == "" {
		t.Skip("HWNPU_MRT_DEVICE is not set")
	}
	dir, ref := refDirs(t)
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "", "")
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	p, err := Prepare(ctx, Setup{Dir: dir, Repo: "google/magenta-realtime-2"})
	if err != nil {
		t.Fatal(err)
	}
	g, err := OpenOn(ctx, dev, p, os.Getenv("HWNPU_MRT_UNITS"))
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	codes := readBin[int32](t, ref+".codes.bin")
	var cs [][]int
	for f := range len(codes) / Codebooks {
		c := make([]int, Codebooks)
		for i := range c {
			c[i] = int(codes[f*Codebooks+i])
		}
		cs = append(cs, c)
	}
	wav, err := g.Decode(cs)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("units %q: samples relative error %.3g (%.2f s)", os.Getenv("HWNPU_MRT_UNITS"), relErr(wav, readBin[float32](t, ref+".wav.bin")), g.Timing.Decode.Seconds())
}
