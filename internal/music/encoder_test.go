package music

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
)

// SpectroStream's encoder (HWNPU_MRT_DIR, HWNPU_MRT_ENC_REF: mrt_enc_ref.py's prefix, the bins
// written as enc.*.bin) gives the reference's STFT, embeddings and codes on the CPU interpreter.
func TestEncoderMatchesReference(t *testing.T) {
	dir, ref := os.Getenv("HWNPU_MRT_DIR"), os.Getenv("HWNPU_MRT_ENC_REF")
	if dir == "" || ref == "" {
		t.Skip("HWNPU_MRT_DIR and HWNPU_MRT_ENC_REF are not set")
	}
	wav := readF32(t, ref+".wav.bin", false)
	spec, frames := STFT(wav)
	want := readF32(t, ref+".stft.bin", false)
	if len(spec) != len(want) {
		t.Fatalf("STFT: %d values (%d frames), want %d", len(spec), frames, len(want))
	}
	t.Logf("STFT: relative error %.3g", math.Sqrt(relErr32(spec, want)))
	T := frames / timeStride
	ew, err := OpenWeights(map[string]string{filepath.Join(dir, "resources/spectrostream/encoder.safetensors"): "params/"})
	if err != nil {
		t.Fatal(err)
	}
	defer ew.Close()
	path := filepath.Join(t.TempDir(), "enc.mlpackage")
	if err := WriteProgram(path, &Encoder{Frames: T}, ew, llm.Options{Format: "fp16"}); err != nil {
		t.Fatal(err)
	}
	if os.Getenv("HWNPU_MRT_TRACE") != "" {
		prog, f, err := milcpu.OpenPackage(path)
		if err != nil {
			t.Fatal(err)
		}
		mask := make([]float32, T)
		for i := range mask {
			mask[i] = 1
		}
		in := map[string][]float32{"x": spec}
		ssMasks(in, mask)
		shapes := prog.Inputs()
		feed := map[string]milcpu.Tensor{}
		for k, v := range in {
			feed[k] = milcpu.Tensor{Shape: shapes[k], Data: v}
		}
		_, err = prog.Run(feed, milcpu.Options{Trace: func(typ, name string, m, rms float32) {
			if m > 1e4 || m != m {
				t.Logf("trace: %s %s: max %.4g rms %.4g", typ, name, m, rms)
			}
		}})
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
	}
	s, err := openStep(context.Background(), nil, path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	mask := make([]float32, T)
	for i := range mask {
		mask[i] = 1
	}
	t0 := time.Now()
	in := map[string][]float32{"x": spec}
	ssMasks(in, mask)
	out, err := s.run(in)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("encoder (CPU interpreter): %.1f s", time.Since(t0).Seconds())
	emb := readF32(t, ref+".emb.bin", false)
	y := make([]float32, len(out["y"])) // (256, T) to (T, 256)
	for c := range EmbeddingSize {
		for f := range T {
			y[f*EmbeddingSize+c] = out["y"][c*T+f]
		}
	}
	out["y"] = y
	t.Logf("embeddings: relative error %.3g", math.Sqrt(relErr32(out["y"], emb)))
	sw, err := OpenWeights(map[string]string{filepath.Join(dir, "checkpoints/mrt2_small.safetensors"): "params/soundstream/"})
	if err != nil {
		t.Fatal(err)
	}
	defer sw.Close()
	codes := readF32(t, ref+".codes.bin", true)
	same, total := 0, 0
	firstLevel := 0
	for f := range T {
		c, err := Quantize(sw, out["y"][f*EmbeddingSize:(f+1)*EmbeddingSize], Codebooks)
		if err != nil {
			t.Fatal(err)
		}
		for l, v := range c {
			if v == int(codes[f*Codebooks+l]) {
				same++
				if l == 0 {
					firstLevel++
				}
			}
			total++
		}
	}
	t.Logf("codes: %d / %d the same (first level %d / %d)", same, total, firstLevel, T)
}

// EncodeAudio, the encoder's program of 41 frames over windows of 25 with 16 before, gives the
// reference's codes over the whole clip (CPU interpreter).
func TestEncodeAudioMatchesReference(t *testing.T) {
	dir, ref := os.Getenv("HWNPU_MRT_DIR"), os.Getenv("HWNPU_MRT_ENC_REF")
	if dir == "" || ref == "" {
		t.Skip("HWNPU_MRT_DIR and HWNPU_MRT_ENC_REF are not set")
	}
	ctx := context.Background()
	p, err := Prepare(ctx, Setup{Dir: dir, Repo: "google/magenta-realtime-2", Encoder: true})
	if err != nil {
		t.Fatal(err)
	}
	g, err := Open(ctx, nil, p)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	codes, err := g.EncodeAudio(readF32(t, ref+".wav.bin", false))
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
	t.Logf("%d frames in %.1f s: %d / %d codes the same (first level %d)", len(codes), g.Timing.Encode.Seconds(), same, len(codes)*Codebooks, first)
	if first < len(codes)*95/100 {
		t.Errorf("first level: %d of %d the same", first, len(codes))
	}
}
