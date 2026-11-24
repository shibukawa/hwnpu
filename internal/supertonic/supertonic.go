// Package supertonic speaks texts with Supertone's Supertonic 3 (Supertone/supertonic-3,
// OpenRAIL-M): 31 languages from their characters, with no phonemizer, at 44.1 kHz. It is four
// ONNX graphs, run as the reference does (supertonic 1.3.1, core.py):
//
//	text → ids → duration_predictor → the speech's length
//	          → text_encoder → the text's embedding
//	noise of that length → vector_estimator, Steps times (flow matching) → vocoder → speech
//
// The graphs run on Stages: the NPU (programs of fixed lengths, the text and the speech padded
// and masked, with no ONNX Runtime: the host takes the embedding tables' rows and works out the
// flow's sin and cos), or the CPU with ONNX Runtime (OpenCPU, BONSAIAI_ONNXRUNTIME).
package supertonic

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"time"
)

// The model's constants (onnx/tts.json).
const (
	SampleRate = 44100
	// Chunk is the samples of a frame of the latent: the autoencoder's chunk (512) times the
	// frames compressed into one (6).
	Chunk = 512 * 6
	// Latent is the latent's channels: 24 times the 6 frames.
	Latent = 24 * 6
	// TextDim is the text embedding's channels.
	TextDim = 256
)

// The reference's defaults and limits (config.py).
const (
	DefaultSteps = 8
	DefaultSpeed = 1.05
	MinSpeed     = 0.7
	MaxSpeed     = 2.0
)

// Stages run the four graphs on one text (a batch of 1). The text's ids and its mask are of
// length T, the latent of length L.
type Stages interface {
	// Duration is the speech's length in seconds.
	Duration(ctx context.Context, ids []int64, style []float32) (float32, error)
	// Encode is the text's embedding, (TextDim, T).
	Encode(ctx context.Context, ids []int64, style []float32) ([]float32, error)
	// Estimate is one step of the flow: the latent (Latent, L) a step on.
	Estimate(ctx context.Context, x []float32, L int, emb []float32, T int, style []float32, step, total int) ([]float32, error)
	// Vocode is the speech of a latent (Latent, L): L * Chunk samples.
	Vocode(ctx context.Context, x []float32, L int) ([]float32, error)
	Close() error
}

// Request is a text to speak.
type Request struct {
	Text, Lang string
	Style      Style
	// Steps are the flow's steps (0: DefaultSteps), Speed how fast it is said (0:
	// DefaultSpeed), Seed the noise's (0: a new one).
	Steps int
	Speed float64
	Seed  uint64
}

// Times are what speaking took, by stage.
type Times struct {
	Duration, Encode, Estimate, Vocode time.Duration
}

// Synth speaks with the stages.
type Synth struct {
	Stages
	Indexer Indexer
	// Dir is the model's directory (voice_styles/).
	Dir   string
	Times Times // the last Speak's
}

// Speak is the speech of a request at SampleRate, and the characters it did not know (left
// out).
func (s *Synth) Speak(ctx context.Context, r Request) (pcm []float32, unknown []string, err error) {
	steps, speed := r.Steps, r.Speed
	if steps <= 0 {
		steps = DefaultSteps
	}
	if speed == 0 {
		speed = DefaultSpeed
	}
	if speed < MinSpeed || speed > MaxSpeed {
		return nil, nil, fmt.Errorf("supertonic: speed %g, from %g to %g", speed, MinSpeed, MaxSpeed)
	}
	text, err := Preprocess(r.Text, r.Lang)
	if err != nil {
		return nil, nil, err
	}
	ids, unknown := s.Indexer.IDs(text)
	s.Times = Times{}
	t0 := time.Now()
	dur, err := s.Duration(ctx, ids, r.Style.DP)
	if err != nil {
		return nil, unknown, err
	}
	dur /= float32(speed)
	s.Times.Duration = time.Since(t0)
	t0 = time.Now()
	emb, err := s.Encode(ctx, ids, r.Style.TTL)
	if err != nil {
		return nil, unknown, err
	}
	s.Times.Encode = time.Since(t0)
	samples := int(float64(dur) * SampleRate)
	L := max(1, (samples+Chunk-1)/Chunk)
	seed := r.Seed
	if seed == 0 {
		seed = rand.Uint64()
	}
	rng := rand.New(rand.NewPCG(seed, 0))
	x := make([]float32, Latent*L)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	t0 = time.Now()
	for step := range steps {
		if x, err = s.Estimate(ctx, x, L, emb, len(ids), r.Style.TTL, step, steps); err != nil {
			return nil, unknown, err
		}
	}
	s.Times.Estimate = time.Since(t0)
	t0 = time.Now()
	wav, err := s.Vocode(ctx, x, L)
	if err != nil {
		return nil, unknown, err
	}
	s.Times.Vocode = time.Since(t0)
	return wav[:min(len(wav), samples)], unknown, nil
}

// Peak is the largest magnitude of a clip (for checks).
func Peak(pcm []float32) float64 {
	m := 0.0
	for _, v := range pcm {
		m = math.Max(m, math.Abs(float64(v)))
	}
	return m
}
