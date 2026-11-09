package music

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/onnx"
	"github.com/shibukawa/hwnpu/internal/tflite"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// StyleCodes are the codes of a MusicCoCa style the model reads.
const StyleCodes = 12

// Styler makes MusicCoCa's styles: an embedding (768) of a text or of audio, a weighted mean of
// several, and its 12 codes. The text and the music encoders run as programs (on the NPU); the
// tokenizer, the mapper of text embeddings, the audio's log mel spectrogram and the quantizer on
// the host.
type Styler struct {
	p                       Paths
	text, music             *step
	sp                      *tokenizer.Unigram
	patches                 []int32
	table                   []float32 // the text encoder's embeddings
	mapper, prep, quantizer *tflite.Model
}

// OpenStyler loads the programs prepared (Setup.Text, Setup.AudioStyle) on dev (nil: the CPU
// interpreter).
func OpenStyler(ctx context.Context, dev *hwnpu.Device, p Paths) (*Styler, error) {
	s := &Styler{p: p}
	var err error
	if p.TextProgram != "" {
		if s.sp, err = tokenizer.LoadSentencePieceUnigram(p.SPM); err != nil {
			return nil, err
		}
		raw, err := os.ReadFile(p.TextTable)
		if err != nil {
			return nil, err
		}
		s.table = make([]float32, len(raw)/2)
		hwnpu.DecodeF16(s.table, raw)
		if s.text, err = openStep(ctx, dev, p.TextProgram); err != nil {
			return nil, err
		}
	}
	if p.MusicProgram != "" {
		raw, err := os.ReadFile(p.MusicPatches)
		if err != nil {
			s.Close()
			return nil, err
		}
		s.patches = make([]int32, len(raw)/4)
		for i := range s.patches {
			s.patches[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		if s.music, err = openStep(ctx, dev, p.MusicProgram); err != nil {
			s.Close()
			return nil, err
		}
	}
	return s, nil
}

// OnNPU reports whether the encoders opened run on the NPU.
func (s *Styler) OnNPU() bool {
	return (s.text == nil || s.text.onNPU) && (s.music == nil || s.music.onNPU)
}

// Placement says, for each program not all on the NPU, what Core ML puts elsewhere.
func (s *Styler) Placement() []string { return placement(s.text, s.music) }

func (s *Styler) Close() error {
	for _, st := range []*step{s.text, s.music} {
		if st != nil {
			st.Close()
		}
	}
	return nil
}

func (s *Styler) host(m **tflite.Model, path string) (*tflite.Model, error) {
	if *m == nil {
		var err error
		if *m, err = tflite.Load(path); err != nil {
			return nil, err
		}
	}
	return *m, nil
}

// Text is the style embedding of a text, as magenta-rt's MusicCoCa.embed(text, use_mapper=True):
// the lowercased text's SentencePiece ids, the text encoder, the mapper with numpy's
// RandomState(0) noise, L2-normalized.
func (s *Styler) Text(text string) ([]float32, error) {
	if s.text == nil {
		return nil, fmt.Errorf("music: MusicCoCa's text side is not prepared")
	}
	ids, pad := TextIDs(s.sp, text)
	out, err := s.text.run(textFeed(s.table, ids, pad))
	if err != nil {
		return nil, fmt.Errorf("music: text encoder: %w", err)
	}
	mp, err := s.host(&s.mapper, s.p.Mapper)
	if err != nil {
		return nil, err
	}
	emb := out["y"]
	noise := LegacyNormal(0, len(emb))
	for i, v := range noise {
		noise[i] = float64(float32(v))
	}
	e := make([]float64, len(emb))
	for i, v := range emb {
		e[i] = float64(v)
	}
	mapped, err := evalOne(mp, map[string]*onnx.Tensor{
		mp.InputValues[0]: {Shape: []int{1, len(e)}, Type: onnx.Float, F: e},
		mp.InputValues[1]: {Shape: []int{1, len(noise)}, Type: onnx.Float, F: noise},
	})
	if err != nil {
		return nil, fmt.Errorf("music: mapper: %w", err)
	}
	var norm float64
	for _, v := range mapped.F {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	res := make([]float32, len(mapped.F))
	for i, v := range mapped.F {
		res[i] = float32(v / norm)
	}
	return res, nil
}

// Audio is the style embedding of mono samples at 16 kHz (ClipRate), as MusicCoCa.embed(audio):
// clips of 10 seconds (the last padded with zeros), each the audio preprocessor's log mel
// spectrogram and the music encoder, averaged.
func (s *Styler) Audio(samples []float32) ([]float32, error) {
	if s.music == nil {
		return nil, fmt.Errorf("music: MusicCoCa's audio side is not prepared")
	}
	prep, err := s.host(&s.prep, s.p.AudioPreprocessor)
	if err != nil {
		return nil, err
	}
	sum := make([]float64, StyleDim)
	clips := 0
	for at := 0; at < len(samples) || at == 0; at += ClipSamples {
		clip := make([]float64, ClipSamples)
		for i := range clip {
			if at+i < len(samples) {
				clip[i] = float64(samples[at+i])
			}
		}
		feat, err := evalOne(prep, map[string]*onnx.Tensor{prep.InputValues[0]: {Shape: []int{1, ClipSamples}, Type: onnx.Float, F: clip}})
		if err != nil {
			return nil, fmt.Errorf("music: audio preprocessor: %w", err)
		}
		x := make([]float32, len(s.patches))
		for i, j := range s.patches {
			x[i] = float32(feat.F[j])
		}
		out, err := s.music.run(map[string][]float32{"x": x})
		if err != nil {
			return nil, fmt.Errorf("music: music encoder: %w", err)
		}
		for i, v := range out["y"] {
			sum[i] += float64(v)
		}
		clips++
	}
	res := make([]float32, StyleDim)
	for i, v := range sum {
		res[i] = float32(v / float64(clips))
	}
	return res, nil
}

// Tokens are the 12 codes of a style embedding, the quantizer's.
func (s *Styler) Tokens(emb []float32) ([]int, error) {
	vq, err := s.host(&s.quantizer, s.p.Quantizer)
	if err != nil {
		return nil, err
	}
	e := make([]float64, len(emb))
	for i, v := range emb {
		e[i] = float64(v)
	}
	codes, err := evalOne(vq, map[string]*onnx.Tensor{vq.InputValues[0]: {Shape: []int{1, len(e)}, Type: onnx.Float, F: e}})
	if err != nil {
		return nil, fmt.Errorf("music: quantizer: %w", err)
	}
	if len(codes.F) < StyleCodes {
		return nil, fmt.Errorf("music: the quantizer gave %d codes", len(codes.F))
	}
	out := make([]int, StyleCodes)
	for i := range out {
		out[i] = int(codes.F[i])
	}
	return out, nil
}

// Mix is a weighted mix of style embeddings: the weighted mean of their directions, at the
// weighted mean of their lengths. MusicCoCa's audio embeddings are about 28 long and its text
// ones (mapped) 1, and the quantizer is not blind to the length, so that a plain mean would be
// the audio's alone; one embedding is itself.
func Mix(embs [][]float32, weights []float64) []float32 {
	dir := make([]float64, StyleDim)
	var total, length float64
	for k, e := range embs {
		var n float64
		for _, v := range e {
			n += float64(v) * float64(v)
		}
		n = math.Sqrt(n)
		if n == 0 {
			continue
		}
		for i, v := range e {
			dir[i] += weights[k] * float64(v) / n
		}
		total += weights[k]
		length += weights[k] * n
	}
	var n float64
	for _, v := range dir {
		n += v * v
	}
	n = math.Sqrt(n)
	out := make([]float32, StyleDim)
	if n == 0 || total == 0 {
		return out
	}
	for i, v := range dir {
		out[i] = float32(v / n * length / total)
	}
	return out
}

// TextIDs are the text encoder's inputs: 1 then the lowercased text's ids (at most 127 of them),
// zeros after; and the paddings, 0 for each token and 1 after.
func TextIDs(sp *tokenizer.Unigram, text string) (ids, pad []float64) {
	t := sp.Encode(strings.ToLower(text))
	ids, pad = make([]float64, textLength), make([]float64, textLength)
	ids[0] = 1
	for i := 1; i < textLength; i++ {
		if i-1 < len(t) {
			ids[i] = float64(t[i-1])
		} else {
			pad[i] = 1
		}
	}
	return ids, pad
}

func evalOne(m *tflite.Model, in map[string]*onnx.Tensor) (*onnx.Tensor, error) {
	out, err := m.Graph.Eval(in, m.OutputValues[:1])
	if err != nil {
		return nil, err
	}
	return out[m.OutputValues[0]], nil
}

// LegacyNormal is numpy's np.random.RandomState(seed).randn(n): MT19937 seeded by init_genrand
// and the polar method, the second of each pair first.
func LegacyNormal(seed uint32, n int) []float64 {
	var mt [624]uint32
	mt[0] = seed
	for i := 1; i < 624; i++ {
		mt[i] = 1812433253*(mt[i-1]^(mt[i-1]>>30)) + uint32(i)
	}
	pos := 624
	next := func() uint32 {
		if pos >= 624 {
			for i := 0; i < 624; i++ {
				y := mt[i]&0x80000000 | mt[(i+1)%624]&0x7fffffff
				v := mt[(i+397)%624] ^ y>>1
				if y&1 != 0 {
					v ^= 0x9908b0df
				}
				mt[i] = v
			}
			pos = 0
		}
		y := mt[pos]
		pos++
		y ^= y >> 11
		y ^= y << 7 & 0x9d2c5680
		y ^= y << 15 & 0xefc60000
		return y ^ y>>18
	}
	double := func() float64 {
		a, b := next()>>5, next()>>6
		return (float64(a)*67108864 + float64(b)) / 9007199254740992
	}
	out := make([]float64, 0, n+1)
	for len(out) < n {
		var x1, x2, r2 float64
		for r2 >= 1 || r2 == 0 {
			x1, x2 = 2*double()-1, 2*double()-1
			r2 = x1*x1 + x2*x2
		}
		f := math.Sqrt(-2 * math.Log(r2) / r2)
		out = append(out, f*x2, f*x1)
	}
	return out[:n]
}
