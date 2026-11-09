package music

import (
	"fmt"
	"sync"
	"time"
)

// Session generates music frame by frame and turns the frames into audio as they come: the
// language model's caches, the frames' codes, and how far the decoder and the inverse STFT have
// gone.
type Session struct {
	g     *Generator
	lm    *LM
	mu    sync.Mutex // codes, between Step and an Audio running alongside (the decoder on the GPU, the language model on the NPU)
	codes [][]int
	encs  map[Conditioning][]float32
	// decoded are the frames the decoder has turned into STFT frames, a whole chunk
	// (DecodeFrames) at a time from the chunk start; emitFrom the first sample (after the
	// lookahead) Audio returns; emitted the samples returned or skipped so far; tail the inverse
	// STFT's overlap.
	decoded, emitFrom, emitted int
	tail                       [2][]float64
}

// NewSession starts a session at the beginning of the music.
func (g *Generator) NewSession() *Session {
	return &Session{g: g, lm: newLM(g.lw, g.temporal, g.depth), encs: map[Conditioning][]float32{}}
}

// Codes are the codes (0 to 1023) of the frames so far.
func (s *Session) Codes() [][]int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.codes
}

// Frames are the frames so far.
func (s *Session) Frames() int { return len(s.Codes()) }

func (s *Session) encode(c Conditioning) ([]float32, error) {
	if e, ok := s.encs[c]; ok {
		return e, nil
	}
	e, err := Encode(s.g.lw, c)
	if err != nil {
		return nil, err
	}
	if len(s.encs) > 4096 {
		clear(s.encs)
	}
	s.encs[c] = e
	return e, nil
}

// Step generates a frame for the conditioning, sampling with pick, and returns its codes.
func (s *Session) Step(c Conditioning, pick Sampler) ([]int, error) {
	e, err := s.encode(c)
	if err != nil {
		return nil, err
	}
	tokens, _, err := s.lm.Step(e, pick, nil, false)
	if err != nil {
		return nil, err
	}
	codes := Codes(tokens)
	s.mu.Lock()
	s.codes = append(s.codes, codes)
	s.mu.Unlock()
	return codes, nil
}

// Prefill runs the language model over frames of codes (a clip's, EncodeAudio) for the
// conditioning, as the C++ runtime's prefill does, so that the music goes on from them; only the
// last frames the model can see (its window through its layers, about 20 seconds) matter. Audio
// then returns the samples after them.
func (s *Session) Prefill(codes [][]int, c Conditioning) error {
	if len(s.codes) > 0 {
		return fmt.Errorf("music: prefill after frames were generated")
	}
	e, err := s.encode(c)
	if err != nil {
		return err
	}
	const horizon = 500 // frames that can still reach the next one through the 12 layers of 41
	from := max(0, len(codes)-horizon)
	forced := make([]int, Codebooks)
	for _, f := range codes[from:] {
		for l, v := range f {
			forced[l] = ReservedTokens + l*CodebookSize + v
		}
		if _, _, err := s.lm.Step(e, nil, forced, false); err != nil {
			return err
		}
	}
	s.mu.Lock()
	s.codes = append(s.codes, codes...)
	s.mu.Unlock()
	// decode from the chunk before the end of the clip (its STFT tail and context), returning
	// the samples after the clip
	P := len(codes)
	s.decoded = max(0, (P-2)/DecodeFrames*DecodeFrames)
	s.emitted = s.decoded*SamplesPerFrame - lookahead*stftHop
	s.emitFrom = P * SamplesPerFrame
	return nil
}

// Audio returns the stereo samples (interleaved) of the frames decoded since the last call: the
// whole chunks of DecodeFrames frames, or all the frames when final. The samples follow the
// clip's (the decoder's lookahead dropped), so that the calls' samples joined are the music. It
// may run alongside Step (one Audio at a time).
func (s *Session) Audio(final bool) ([]float32, error) {
	t0 := time.Now()
	defer func() { s.g.Timing.Decode += time.Since(t0) }()
	codes := s.Codes() // the frames so far; those before T do not change
	T := len(codes)
	W := DecodeFrames + decodeContext
	x := make([]float32, W*EmbeddingSize)
	row := make([]float32, EmbeddingSize)
	mask := make([]float32, W)
	var out []float32
	for s.decoded < T && (final || s.decoded+DecodeFrames <= T) {
		start := s.decoded
		clear(x)
		clear(mask)
		for i := range W {
			f := start - decodeContext + i
			if f >= 0 && f < T {
				if err := Embed(s.g.sw, codes[f], row); err != nil {
					return nil, err
				}
				for c, v := range row { // channels first: (256, W)
					x[c*W+i] = v
				}
				mask[i] = 1
			}
		}
		in := map[string][]float32{"x": x}
		ssMasks(in, mask)
		res, err := s.g.dec.run(in)
		if err != nil {
			return nil, err
		}
		n := timeStride * min(DecodeFrames, T-start) // the chunk's STFT frames
		spec := make([]float32, 4*n*stftBins)
		y := res["y"]
		for c := range 4 {
			for k := range n {
				src := (c*timeStride*W + timeStride*decodeContext + k) * stftBins
				for b, v := range y[src : src+stftBins] {
					spec[(c*n+k)*stftBins+b] = v * DecoderOutScale
				}
			}
		}
		// samples of the STFT frames from 4 start on, the lookahead's 4 dropped: at the start
		// of the music before the inverse STFT, as the decoder's layer drops them
		pos := start*SamplesPerFrame - lookahead*stftHop // the chunk's first sample after the lookahead
		if start == 0 {
			k := min(lookahead, n)
			cut := make([]float32, 4*(n-k)*stftBins)
			for c := range 4 {
				copy(cut[c*(n-k)*stftBins:(c+1)*(n-k)*stftBins], spec[(c*n+k)*stftBins:(c*n+n)*stftBins])
			}
			spec, n, pos = cut, n-k, 0
		}
		var wav []float32
		wav, s.tail = InverseSTFT(spec, n, s.tail)
		for i := 0; i < len(wav)/2; i++ {
			if p := pos + i; p >= s.emitFrom && p >= s.emitted {
				out = append(out, wav[2*i], wav[2*i+1])
			}
		}
		s.emitted = max(s.emitted, pos+len(wav)/2)
		s.decoded = start + min(DecodeFrames, T-start)
	}
	return out, nil
}
