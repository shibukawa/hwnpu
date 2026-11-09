package music

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
)

func readBin[T float32 | int32](t *testing.T, path string) []T {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var z T
	out := make([]T, len(raw)/binary.Size(z))
	if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, out); err != nil {
		t.Fatal(err)
	}
	return out
}

func relErr(got, want []float32) float64 {
	var se, sw float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		se += d * d
		sw += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(se / sw)
}

// refDir is where the Hugging Face repository is (HWNPU_MRT, the layout of
// google/magenta-realtime-2) and the reference of mrt_ss_ref.py (HWNPU_MRT_REF).
func refDirs(t *testing.T) (string, string) {
	dir, ref := os.Getenv("HWNPU_MRT"), os.Getenv("HWNPU_MRT_REF")
	if dir == "" || ref == "" {
		t.Skip("HWNPU_MRT and HWNPU_MRT_REF are not set")
	}
	return dir, ref
}

func openSS(t *testing.T, dir string) *Weights {
	t.Helper()
	w, err := OpenWeights(map[string]string{
		filepath.Join(dir, "resources/spectrostream/decoder.safetensors"):   "params/",
		filepath.Join(dir, "resources/spectrostream/quantizer.safetensors"): "params/",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { w.Close() })
	return w
}

// The decoder program on the CPU interpreter in fp16 gives SpectroStream's STFT of a clip of
// codes, whole and as a window of a longer program with the frames before the clip masked; the
// inverse STFT gives its samples.
func TestDecoderMatchesReference(t *testing.T) {
	dir, ref := refDirs(t)
	w := openSS(t, dir)
	codes := readBin[int32](t, ref+".codes.bin")
	T := len(codes) / Codebooks
	x := make([]float32, T*EmbeddingSize)
	for f := range T {
		c := make([]int, Codebooks)
		for i := range c {
			c[i] = int(codes[f*Codebooks+i])
		}
		if err := Embed(w, c, x[f*EmbeddingSize:(f+1)*EmbeddingSize]); err != nil {
			t.Fatal(err)
		}
	}
	if e := relErr(x, readBin[float32](t, ref+".emb.bin")); e > 1e-6 {
		t.Errorf("embeddings: relative error %g", e)
	}
	run := func(frames, offset int) []float32 {
		p := filepath.Join(t.TempDir(), "d.mlpackage")
		if _, err := llm.WritePart(p, &Decoder{Frames: frames}, w, llm.Options{Format: "fp16"}, nil); err != nil {
			t.Fatal(err)
		}
		prog, f, err := milcpu.OpenPackage(p)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		in := make([]float32, frames*EmbeddingSize)
		mask := make([]float32, frames)
		for i := range frames {
			src := i - offset
			if src >= 0 && src < T {
				for c := range EmbeddingSize { // channels first
					in[c*frames+i] = x[src*EmbeddingSize+c]
				}
				mask[i] = 1
			}
		}
		feed := map[string][]float32{}
		ssMasks(feed, mask)
		tensors := map[string]milcpu.Tensor{"x": {Shape: []int{1, EmbeddingSize, frames, 1}, Data: in}}
		for k, v := range feed {
			tensors[k] = milcpu.Tensor{Shape: []int{1, 1, len(v), 1}, Data: v}
		}
		opts := milcpu.Options{RoundF16: os.Getenv("HWNPU_MRT_F16") != "", FlushSubnormals: os.Getenv("HWNPU_MRT_F16") == "ftz"}
		if os.Getenv("HWNPU_MRT_TRACE") != "" {
			opts.Trace = func(typ, name string, m, rms float32) {
				if m > 1e4 || m != m || os.Getenv("HWNPU_MRT_TRACE") == "all" {
					t.Logf("trace: %s %s: max %.4g rms %.4g", typ, name, m, rms)
				}
			}
		}
		res, err := prog.Run(tensors, opts)
		if err != nil {
			t.Fatal(err)
		}
		y := res["y"].Data
		for i := range y {
			y[i] *= DecoderOutScale
		}
		return y
	}
	// the reference is (1, 4T, 480, 4); the program's (1, 4, 4T', 480)
	want := readBin[float32](t, ref+".stft.bin")
	compare := func(name string, y []float32, frames, offset int) {
		n := 4 * frames
		var got, w []float32
		for c := range 4 {
			for s := range 4 * T {
				for b := range stftBins {
					got = append(got, y[(c*n+s+4*offset)*stftBins+b])
					w = append(w, want[(s*stftBins+b)*4+c])
				}
			}
		}
		e := relErr(got, w)
		t.Logf("%s: STFT relative error %.3g", name, e)
		if e > 0.01 {
			t.Errorf("%s: STFT relative error %g", name, e)
		}
	}
	y := run(T, 0)
	compare("whole", y, T, 0)
	compare("after 7 masked frames", run(T+7, 7), T+7, 7)
	// the inverse STFT of the frames after the lookahead
	spec := make([]float32, 4*(4*T-lookahead)*stftBins)
	frames := 4*T - lookahead
	for c := range 4 {
		copy(spec[c*frames*stftBins:(c+1)*frames*stftBins], y[(c*4*T+lookahead)*stftBins:(c*4*T+4*T)*stftBins])
	}
	wav, _ := InverseSTFT(spec, frames, [2][]float64{})
	wantWav := readBin[float32](t, ref+".wav.bin")
	if len(wav) != len(wantWav) {
		t.Fatalf("%d samples, want %d", len(wav), len(wantWav))
	}
	e := relErr(wav, wantWav)
	t.Logf("samples: relative error %.3g", e)
	if e > 0.02 {
		t.Errorf("samples: relative error %g", e)
	}
}

// The temporal and depth programs on the CPU interpreter in fp16, stepped with the reference's
// tokens forced, give the reference's encoder output, temporal outputs (through the depth
// steps' logits) and soft-capped logits of every codebook, past the window of 41 frames.
func TestLMMatchesReference(t *testing.T) {
	dir, ref := refDirs(t)
	lw, err := OpenWeights(map[string]string{filepath.Join(dir, "checkpoints/mrt2_small.safetensors"): "params/"})
	if err != nil {
		t.Fatal(err)
	}
	defer lw.Close()
	block := readBin[int32](t, ref+".block.bin")
	var c Conditioning
	for i := range c {
		c[i] = int(block[i]) - ReservedTokens - 1
	}
	want := NewConditioning(c[:12], Guidance{Style: 3, Notes: 0.1, Drums: 1}, c[condDrums])
	if want != c {
		t.Errorf("conditioning %v, NewConditioning gives %v", c, want)
	}
	enc, err := Encode(lw, c)
	if err != nil {
		t.Fatal(err)
	}
	wantEnc := readBin[float32](t, ref+".enc.bin")
	if e := relErr(enc, wantEnc[:encDim]); e > 0.01 {
		t.Errorf("encoder: relative error %g", e)
	} else {
		t.Logf("encoder: relative error %.3g", e)
	}
	dirp := t.TempDir()
	open := func(name string, p Program) *step {
		path := filepath.Join(dirp, name+".mlpackage")
		if err := WriteProgram(path, p, lw, llm.Options{Format: "fp16"}); err != nil {
			t.Fatal(err)
		}
		s, err := openStep(context.Background(), nil, path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { s.Close() })
		return s
	}
	l := newLM(lw, open("temporal", Temporal{}), open("depth", Depth{}))
	tokens := readBin[int32](t, ref+".tokens.bin")
	wantLogits := readBin[float32](t, ref+".logits.bin")
	frames := len(tokens) / Codebooks
	var worst float64
	agree, total := 0, 0
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
			w := wantLogits[(f*Codebooks+cb)*Vocab+lo : (f*Codebooks+cb)*Vocab+lo+CodebookSize]
			g := lg[cb*CodebookSize : (cb+1)*CodebookSize]
			worst = math.Max(worst, relErr(g, w))
			if argmax(g) == argmax(w) {
				agree++
			}
			total++
		}
		if f == 0 || f == frames-1 {
			t.Logf("frame %d: worst relative error so far %.3g", f, worst)
		}
	}
	t.Logf("%d frames: worst logits relative error %.3g, the same most likely token in %d of %d", frames, worst, agree, total)
	if worst > 0.05 || agree < total*9/10 {
		t.Errorf("worst logits relative error %g, the same most likely token in %d of %d", worst, agree, total)
	}
}

func argmax(v []float32) int {
	b := 0
	for i, x := range v {
		if x > v[b] {
			b = i
		}
	}
	return b
}

// A session's audio, a chunk at a time as the frames come, is the reference's samples of the
// clip of codes (the decoder and the inverse STFT over the whole clip), on the CPU interpreter.
func TestSessionAudioMatchesReference(t *testing.T) {
	dir, ref := refDirs(t)
	ctx := context.Background()
	p, err := Prepare(ctx, Setup{Dir: dir, Repo: "google/magenta-realtime-2"})
	if err != nil {
		t.Fatal(err)
	}
	g, err := Open(ctx, nil, p)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	codes := readBin[int32](t, ref+".codes.bin")
	s := g.NewSession()
	var wav []float32
	for f := range len(codes) / Codebooks {
		c := make([]int, Codebooks)
		for i := range c {
			c[i] = int(codes[f*Codebooks+i])
		}
		s.codes = append(s.codes, c)
		if (f+1)%DecodeFrames == 0 {
			pcm, err := s.Audio(false)
			if err != nil {
				t.Fatal(err)
			}
			wav = append(wav, pcm...)
		}
	}
	pcm, err := s.Audio(true)
	if err != nil {
		t.Fatal(err)
	}
	wav = append(wav, pcm...)
	want := readBin[float32](t, ref+".wav.bin")
	if len(wav) != len(want) {
		t.Fatalf("%d samples, want %d", len(wav), len(want))
	}
	for c := 0; c < len(want)/2; c += SamplesPerFrame * 5 {
		e := min(c+SamplesPerFrame*5, len(want)/2)
		t.Logf("frames %d-%d: relative error %.3g", c/SamplesPerFrame, e/SamplesPerFrame, relErr(wav[2*c:2*e], want[2*c:2*e]))
	}
	if e := relErr(wav, want); e > 0.01 {
		t.Errorf("samples: relative error %g", e)
	} else {
		t.Logf("samples: relative error %.3g", e)
	}
}
