package speech

import (
	"context"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3gen"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// An STFT taken as the codec's frames are (a Hann window of n_fft every hop over the signal
// padded by (n_fft-hop)/2 on both sides), given as log magnitudes and phases, comes back as the
// signal.
func TestISTFTReconstructs(t *testing.T) {
	const nfft, hop, n = 16, 4, 48
	rng := rand.New(rand.NewPCG(1, 2))
	x := make([]float64, n)
	for i := range x {
		x[i] = rng.Float64()*2 - 1
	}
	pad := (nfft - hop) / 2
	padded := make([]float64, n+2*pad)
	copy(padded[pad:], x)
	frames := (len(padded)-nfft)/hop + 1
	t1 := NewISTFT(nfft, hop)
	var got []float32
	for f := 0; f < frames; f++ {
		polar := make([]float32, 2*(nfft/2+1))
		for k := 0; k <= nfft/2; k++ {
			var re, im float64
			for j := 0; j < nfft; j++ {
				w := 0.5 * (1 - math.Cos(2*math.Pi*float64(j)/nfft))
				a := 2 * math.Pi * float64(k*j) / nfft
				re += w * padded[f*hop+j] * math.Cos(a)
				im -= w * padded[f*hop+j] * math.Sin(a)
			}
			polar[k] = float32(math.Log(math.Hypot(re, im)))
			polar[nfft/2+1+k] = float32(math.Atan2(im, re))
		}
		got = append(got, t1.Frame(polar)...)
	}
	got = append(got, t1.Flush()...)
	if len(got) != n {
		t.Fatalf("%d samples, want %d", len(got), n)
	}
	for i, v := range got {
		if math.Abs(float64(v)-x[i]) > 1e-4 {
			t.Errorf("sample %d is %g, want %g", i, v, x[i])
			break
		}
	}
}

const tinyCodec = `{"model_type": "liquid_codec", "fsq_levels": [4, 3], "n_q": 3, "quantizer_dim": 24, "encoder_dim": 512,
	"stft_n_fft": 16, "stft_hop_length": 4, "stft_win_length": 16, "stft_frame_stack": 2, "decoder_window": 3, "sampling_rate": 24000}`

func convert(t *testing.T, a *llm.Arch, m llm.Model, w llm.Weights, opts llm.Options) runner.Program {
	t.Helper()
	path, err := llm.Convert(a, m, w, filepath.Join(t.TempDir(), "m.mlpackage"), "none", 0, opts, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	p, err := runner.Open(context.Background(), nil, path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { p.Close() })
	return p
}

// Decoding in chunks that overlap by the look back gives what one call over all the frames
// gives; with too little overlap it would not.
func TestDecodeCodecChunks(t *testing.T) {
	c, err := lfm3gen.ParseCodec([]byte(tinyCodec))
	if err != nil {
		t.Fatal(err)
	}
	c.SetLayers([]string{"conv", "full_attention", "conv"}, 8, 4, 16, 64, 3)
	w := llm.NewSynthetic(c)
	rng := rand.New(rand.NewPCG(3, 4))
	frames := make([][]int32, 13)
	for i := range frames {
		for range c.Codebooks {
			frames[i] = append(frames[i], int32(rng.IntN(12)))
		}
	}
	whole := convert(t, lfm3gen.CodecArch, c, w, llm.Options{Format: "fp16", Params: map[string]string{"codec-frames": "16"}})
	want, err := DecodeCodec(context.Background(), c, whole, 16, frames)
	if err != nil {
		t.Fatal(err)
	}
	chunk := convert(t, lfm3gen.CodecArch, c, w, llm.Options{Format: "fp16", Params: map[string]string{"codec-frames": "5"}})
	got, err := DecodeCodec(context.Background(), c, chunk, 5, frames)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) || len(got) != 13*c.FrameStack*2*c.Bins() {
		t.Fatalf("%d and %d values", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("value %d: %g in chunks, %g at once", i, got[i], want[i])
		}
	}
	if _, err := DecodeCodec(context.Background(), c, chunk, 2, frames); err == nil {
		t.Error("chunks no longer than the look back accepted")
	}
}

func TestSampling(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	l := []float32{1, 3, 2, 3, -1}
	if got := (&Sampling{}).pick(l, rng); got != 1 {
		t.Errorf("greedy pick %d", got)
	}
	// top 2 keeps the two 3s (and nothing else)
	seen := map[int]bool{}
	for range 200 {
		seen[(&Sampling{Temperature: 1, TopK: 2}).pick(l, rng)] = true
	}
	if len(seen) != 2 || !seen[1] || !seen[3] {
		t.Errorf("top-k picked %v", seen)
	}
	// the penalty divides a positive logit and multiplies a negative one, for the codes the
	// codebook took in the window
	sp := &Sampling{Repetition: 2, Window: 1}
	pl := []float32{4, -4, 4}
	sp.penalize(pl, [][]int32{{2, 0}, {1, 0}, {0, 2}}, 0)
	if !slices.Equal(pl, []float32{2, -4, 4}) {
		t.Errorf("penalized %v", pl)
	}
	sp.Window = 3
	pl = []float32{4, -4, 4}
	sp.penalize(pl, [][]int32{{2, 0}, {1, 0}, {0, 2}}, 0)
	if !slices.Equal(pl, []float32{2, -8, 2}) {
		t.Errorf("penalized over 3 frames %v", pl)
	}
}

func TestPrompt(t *testing.T) {
	got := Prompt("こんにちは", "JA", "Male", false)
	want := "<|startoftext|><|im_start|>system\nPerform TTS in Japanese using a male voice.<|im_end|>\n<|im_start|>user\nこんにちは<|im_end|>\n<|im_start|>assistant\n<|audio_start|>"
	if got != want {
		t.Errorf("prompt %q", got)
	}
	if got := Prompt("x", "Klingon", "robot", true); !strings.Contains(got, "TTS in Klingon using a female voice") || !strings.HasSuffix(got, "<think></think><|audio_start|>") {
		t.Errorf("prompt %q", got)
	}
}

// spyProgram records the inputs of each call.
type spyProgram struct {
	runner.Program
	calls []map[string][]float32
}

func (p *spyProgram) Run(ctx context.Context, in map[string][]float32) ([]float32, error) {
	c := map[string][]float32{}
	for k, v := range in {
		c[k] = slices.Clone(v)
	}
	p.calls = append(p.calls, c)
	return p.Program.Run(ctx, in)
}

// tiny LFM3-Audio: a text model of width 48, a depthformer of 4 codebooks of 17 codes
const tinyAudio = `{"model_type": "lfm3_audio", "codebooks": 4, "audio_vocab_size": 17, "maskgit_steps": 2,
	"lfm": {"model_type": "lfm3", "hidden_size": 48, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 6,
		"num_key_value_heads": 2, "head_dim": 8, "vocab_size": 40, "norm_eps": 1e-5, "conv_L_cache": 3, "sliding_window": 4,
		"use_sandwich_norm": true, "tie_word_embeddings": true, "layer_types": ["conv", "full_attention"],
		"rope_parameters": {"full_attention": {"rope_theta": 10000}}},
	"depthformer": {"hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
		"norm_eps": 1e-5, "layer_types": ["full_attention"], "use_sandwich_norm": false, "is_causal": false,
		"rope_parameters": {"full_attention": {"rope_theta": 1000000.0}}}}`

// The frame loop: each frame's codes are the depthformer's best for each codebook in the pass
// that reveals it (greedy), with the codes revealed before it in the pass's input; each frame's
// feedback embedding is the text model's next input.
func TestFrames(t *testing.T) {
	text, err := lfm3.ParseConfig([]byte(tinyAudio))
	if err != nil {
		t.Fatal(err)
	}
	depth, err := lfm3gen.ParseDepth([]byte(tinyAudio))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.Synthetic{}
	for _, m := range []llm.Model{text, depth} {
		for k, v := range m.Tensors() {
			w[k] = v
		}
	}
	w[depth.Feedback()] = []int{depth.Codebooks * depth.Vocab, depth.TextHidden}
	path, err := llm.Convert(lfm3.Arch, text, w, filepath.Join(t.TempDir(), "text.mlpackage"), "none", 0,
		llm.Options{Seq: 1, Cache: 16, Format: "fp16", Params: map[string]string{"hidden-out": "1", "post-scale": "1"}}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	sess, cl, err := runner.OpenSession(context.Background(), nil, []string{path}, text)
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	s := &Synth{text: text, depth: depth, w: w, sess: sess, rows: map[string][]float32{},
		dprog: convert(t, lfm3gen.DepthArch, depth, w, llm.Options{Format: "fp16"})}
	for _, id := range []int32{3, 7, 1} {
		e, _ := llm.Embed(w, text, []int32{id})
		if _, err := sess.Step([][]float32{e}); err != nil {
			t.Fatal(err)
		}
	}
	h0 := slices.Clone(sess.Hidden())
	if len(h0) != 48 {
		t.Fatalf("hidden state of %d values", len(h0))
	}
	frames, err := s.Frames(context.Background(), h0, rand.New(rand.NewPCG(9, 9)), 3)
	if err != nil {
		t.Fatal(err)
	}
	// a feedback step after each frame kept, none after the end of audio
	if len(frames) == 0 || sess.Pos() != 3+len(frames) {
		t.Fatalf("%d frames, %d positions", len(frames), sess.Pos())
	}
	for i, f := range frames {
		if slices.Contains(f, int32(depth.Vocab-1)) {
			t.Errorf("frame %d %v has the end of audio code", i, f)
		}
	}
	n, v, d := depth.Codebooks, depth.Vocab, depth.Hidden()
	mask, _, _ := w.Get(depth.MaskEmbedding())
	byHand := func(h []float32, order []int) []int32 {
		codes := slices.Repeat([]int32{-1}, n)
		k := 0
		for _, pick := range depth.Schedule {
			x := make([]float32, n*d)
			for cb, c := range codes {
				e := mask
				if c >= 0 {
					e, _ = w.Rows(depth.Table(cb), int(c), int(c)+1)
				}
				copy(x[cb*d:], e)
			}
			logits, err := s.dprog.Run(context.Background(), map[string][]float32{"x": x, "h": h})
			if err != nil {
				t.Fatal(err)
			}
			for range pick {
				cb := order[k]
				k++
				codes[cb] = int32(argmax(logits[cb*v : (cb+1)*v]))
			}
		}
		return codes
	}
	orders := rand.New(rand.NewPCG(9, 9))
	if codes := byHand(h0, orders.Perm(n)); !slices.Equal(codes, frames[0]) {
		t.Errorf("first frame %v, by hand %v", frames[0], codes)
	}
	if len(frames) < 3 {
		for range len(frames) - 1 {
			orders.Perm(n)
		}
		if next := byHand(sess.Hidden(), orders.Perm(n)); !slices.Contains(next, int32(v-1)) {
			t.Errorf("stopped after %d frames without an end of audio code (next %v)", len(frames), next)
		}
	}
	fb, err := s.feedback(frames[0])
	if err != nil {
		t.Fatal(err)
	}
	r, _ := w.Rows(depth.Feedback(), int(frames[0][1])+v, int(frames[0][1])+v+1)
	if fb[5] == r[5] {
		t.Error("the feedback is one codebook's row, not the sum")
	}
	// the passes' inputs: the codes revealed so far as their embeddings, the others masked, and
	// the hidden state after the last feedback
	spy := &spyProgram{Program: s.dprog}
	s.dprog = spy
	sess.Reset()
	for _, id := range []int32{3, 7, 1} {
		e, _ := llm.Embed(w, text, []int32{id})
		sess.Step([][]float32{e})
	}
	frames2, err := s.Frames(context.Background(), h0, rand.New(rand.NewPCG(9, 9)), 2)
	if err != nil {
		t.Fatal(err)
	}
	s.dprog = spy.Program
	steps := len(depth.Schedule)
	if len(spy.calls) <= steps {
		t.Fatalf("%d passes for %d frames", len(spy.calls), len(frames2))
	}
	for call, in := range spy.calls {
		if call >= steps && slices.Equal(in["h"], h0) {
			t.Errorf("pass %d of frame 2 reads the hidden state before the first frame's feedback", call)
		}
		if call/steps >= len(frames2) {
			continue // the frame that ended the audio
		}
		f := frames2[call/steps]
		known := 0
		for _, n := range depth.Schedule[:call%steps] {
			known += n
		}
		revealed := 0
		for cb := range n {
			row := in["x"][cb*d : (cb+1)*d]
			if e, _ := w.Rows(depth.Table(cb), int(f[cb]), int(f[cb])+1); slices.Equal(row, e) {
				revealed++
			} else if !slices.Equal(row, mask) {
				t.Errorf("pass %d: codebook %d's input is neither its code's embedding nor the mask", call, cb)
			}
		}
		if revealed != known {
			t.Errorf("pass %d: %d codebooks revealed, want %d", call, revealed, known)
		}
	}
	again, _ := s.Frames(context.Background(), h0, rand.New(rand.NewPCG(9, 9)), 1)
	if fmt.Sprint(again[0]) != fmt.Sprint(frames[0]) {
		t.Errorf("not deterministic: %v then %v", frames[0], again[0])
	}
}
