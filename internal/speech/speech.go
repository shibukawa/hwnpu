// Package speech speaks text with LFM3-Audio (or LFM3-Omni) on the Neural Engine, as llama.cpp's
// lfm3a pipeline does (tools/mtmd/mtmd-helper-gen.cpp): the text model reads the TTS prompt (its
// "tts" chat template) up to <|audio_start|>; then for each audio frame the depthformer turns
// the text model's final hidden state into the frame's codes in MaskGIT passes (the codebooks
// revealed in a random order, each code sampled with temperature, top-k and a repetition
// penalty), the sum of the codes' feedback embeddings is the text model's next input, and a
// frame with an end of audio code ends the speech. The liquid codec's decoder turns the codes
// into STFT frames and the inverse STFT into samples at 24 kHz.
//
// The programs: the text model converted with a KV cache and -hidden-out 1 (so that it outputs
// its final hidden state), the depthformer (-arch lfm3-depthformer) and the codec decoder
// (-arch lfm3-codec, on audio_tokenizer/). The audio is decoded after the last frame, in chunks
// that overlap by the decoder's look back, which gives what decoding it as a stream gives.
package speech

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3gen"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// Config says where the model is.
type Config struct {
	// Text are the text model converted with a KV cache and -hidden-out 1, one or more packages
	// of the same cache and different -seq (1 to generate; a longer one reads the prompt fast).
	Text []string
	// Depth is the depthformer (-arch lfm3-depthformer), Codec the codec decoder (-arch
	// lfm3-codec).
	Depth, Codec string
	// Dir holds the checkpoint: config.json, model.safetensors (the text model's embeddings and
	// the depthformer's code tables), tokenizer.json, generation_config.json (optional), and
	// audio_tokenizer/config.json.
	Dir string
	// CacheDir is the compile cache ("" for none). CPU runs everything with the CPU interpreter.
	CacheDir string
	CPU      bool
	Log      io.Writer
}

// encoder is the tokenizer.
type encoder interface {
	Encode(text string) []int32
}

// Synth speaks text.
type Synth struct {
	tok   encoder
	text  *lfm3.Config
	depth *lfm3gen.Depth
	codec *lfm3gen.Codec
	w     llm.Weights // the checkpoint
	sess  *decode.Session
	dprog runner.Program
	cprog runner.Program
	cfr   int // the codec program's frames

	dev     *hwnpu.Device
	closers []io.Closer

	// Sampling of the speech codes.
	Sampling Sampling
	rows     map[string][]float32 // rows of the code and feedback tables read so far
}

// Sampling is how the codes are drawn: a temperature (0 picks the best), the k best (0: all,
// ties with the k-th kept) and a CTRL style penalty on the codes a codebook took in the last
// window frames.
type Sampling struct {
	Temperature float64
	TopK        int
	Repetition  float64
	Window      int
}

// Load loads the programs.
func Load(ctx context.Context, c Config) (*Synth, error) {
	logf := func(format string, args ...any) {
		if c.Log != nil {
			fmt.Fprintf(c.Log, format+"\n", args...)
		}
	}
	_, tm, err := llm.LoadConfig(filepath.Join(c.Dir, "config.json"), lfm3.Arch.Name)
	if err != nil {
		return nil, err
	}
	_, dm, err := llm.LoadConfig(filepath.Join(c.Dir, "config.json"), lfm3gen.DepthArch.Name)
	if err != nil {
		return nil, err
	}
	_, cm, err := llm.LoadConfig(filepath.Join(c.Dir, "audio_tokenizer", "config.json"), lfm3gen.CodecArch.Name)
	if err != nil {
		return nil, err
	}
	s := &Synth{text: tm.(*lfm3.Config), depth: dm.(*lfm3gen.Depth), codec: cm.(*lfm3gen.Codec), rows: map[string][]float32{}}
	s.Sampling = Sampling{s.depth.Temperature, s.depth.TopK, s.depth.Repetition, s.depth.RepWindow}
	if err := s.Sampling.override(filepath.Join(c.Dir, "generation_config.json")); err != nil {
		return nil, err
	}
	tok, err := tokenizer.Load(filepath.Join(c.Dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	s.tok = tok
	if s.w, err = llm.OpenWeights(filepath.Join(c.Dir, "model.safetensors"), lfm3.Arch); err != nil {
		return nil, err
	}
	s.closers = append(s.closers, s.w)
	if !c.CPU {
		opts := []hwnpu.Option{hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer)}
		if c.CacheDir != "" {
			opts = append(opts, hwnpu.WithCacheDir(c.CacheDir))
		}
		if s.dev, err = hwnpu.Open(ctx, opts...); err != nil {
			s.Close()
			return nil, err
		}
	}
	t0 := time.Now()
	sess, cl, err := runner.OpenSession(ctx, s.dev, c.Text, s.text)
	if err != nil {
		s.Close()
		return nil, err
	}
	s.sess = sess
	s.closers = append(s.closers, cl)
	if s.dprog, err = runner.Open(ctx, s.dev, c.Depth); err != nil {
		s.Close()
		return nil, err
	}
	s.closers = append(s.closers, s.dprog)
	if s.cprog, err = runner.Open(ctx, s.dev, c.Codec); err != nil {
		s.Close()
		return nil, err
	}
	s.closers = append(s.closers, s.cprog)
	if err := s.check(); err != nil {
		s.Close()
		return nil, err
	}
	logf("speech: text model (cache %d), depthformer, codec decoder (%d frames per call) loaded in %.1fs", s.sess.Cache(), s.cfr, time.Since(t0).Seconds())
	return s, nil
}

// check checks the programs' shapes against the configs.
func (s *Synth) check() error {
	x, h, y := s.dprog.Inputs()["x"], s.dprog.Inputs()["h"], s.dprog.Output()
	if len(x) != 3 || x[1] != s.depth.Codebooks || len(h) != 3 || h[2] != s.depth.TextHidden || len(y) != 3 || y[2] != s.depth.Vocab {
		return fmt.Errorf("speech: the depthformer takes %v and %v and gives %v", x, h, y)
	}
	cx, cy := s.cprog.Inputs()["x"], s.cprog.Output()
	if len(cx) != 3 || cx[2] != s.codec.CodeDim() || len(cy) != 3 || cy[1] != cx[1]*s.codec.FrameStack || cy[2] != 2*s.codec.Bins() {
		return fmt.Errorf("speech: the codec decoder takes %v and gives %v", cx, cy)
	}
	s.cfr = cx[1]
	if s.cfr*s.codec.FrameStack <= s.codec.Lookback() {
		return fmt.Errorf("speech: the codec decoder's %d frames are no more than its look back of %d positions", s.cfr, s.codec.Lookback())
	}
	return nil
}

func (sp *Sampling) override(path string) error {
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	} else if err != nil {
		return err
	}
	var g struct {
		Temp   *float64 `json:"audio_temperature"`
		TopK   *int     `json:"audio_top_k"`
		Rep    *float64 `json:"audio_repetition_penalty"`
		Window *int     `json:"audio_repetition_window"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		return fmt.Errorf("speech: %s: %w", path, err)
	}
	if g.Temp != nil {
		sp.Temperature = *g.Temp
	}
	if g.TopK != nil {
		sp.TopK = *g.TopK
	}
	if g.Rep != nil {
		sp.Repetition = *g.Rep
	}
	if g.Window != nil {
		sp.Window = *g.Window
	}
	return nil
}

// Close releases the programs.
func (s *Synth) Close() error {
	for i := len(s.closers) - 1; i >= 0; i-- {
		s.closers[i].Close()
	}
	s.closers = nil
	if s.dev != nil {
		s.dev.Close()
		s.dev = nil
	}
	return nil
}

// SampleRate is the rate of the samples Speak returns.
func (s *Synth) SampleRate() int { return s.codec.SampleRate }

// languages maps the language codes of the model card to its names; another language is given
// by its name.
var languages = map[string]string{
	"ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "American English", "en-us": "American English",
	"en-gb": "British English", "fi": "Finnish", "fr": "French", "he": "Hebrew", "it": "Italian", "ja": "Japanese",
	"ko": "Korean", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "es": "Spanish", "sv": "Swedish", "tr": "Turkish",
	"zh": "Chinese",
}

// Prompt is the TTS prompt of the model card, as the "tts" chat template llama.cpp writes for
// LFM3-Audio renders it: language a code of the card (default en) or a language's name, voice
// female (default) or male; LFM3-Omni opens the answer with an empty thought.
func Prompt(text, language, voice string, omni bool) string {
	code := strings.ToLower(cmp.Or(language, "en"))
	lang, ok := languages[code]
	if !ok {
		lang = language
	}
	v := strings.ToLower(voice)
	if v != "female" && v != "male" {
		v = "female"
	}
	think := ""
	if omni {
		think = "<think></think>"
	}
	return "<|startoftext|><|im_start|>system\nPerform TTS in " + lang + " using a " + v + " voice.<|im_end|>\n" +
		"<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n" + think + "<|audio_start|>"
}

// Request is one utterance.
type Request struct {
	Text, Language, Voice string
	Seed                  uint64 // 0: random
	MaxFrames             int    // 0: until the end of audio or a full cache
}

// Speak returns the speech of a request: samples at SampleRate, and the codes of its frames.
func (s *Synth) Speak(ctx context.Context, r Request) ([]float32, [][]int32, error) {
	ids := s.tok.Encode(Prompt(r.Text, r.Language, r.Voice, s.depth.Omni))
	s.sess.Reset()
	for i := 0; i < len(ids); i += s.sess.Q() {
		emb, err := llm.Embed(s.w, s.text, ids[i:min(i+s.sess.Q(), len(ids))])
		if err != nil {
			return nil, nil, err
		}
		var rows [][]float32
		for j := 0; j < len(emb); j += s.text.HiddenSize {
			rows = append(rows, emb[j:j+s.text.HiddenSize])
		}
		if _, err := s.sess.Step(rows); err != nil {
			return nil, nil, err
		}
	}
	if s.sess.Hidden() == nil {
		return nil, nil, errors.New("speech: the text model does not output its hidden state h (convert it with -hidden-out 1)")
	}
	seed := cmp.Or(r.Seed, uint64(time.Now().UnixNano()))
	frames, err := s.Frames(ctx, s.sess.Hidden(), rand.New(rand.NewPCG(seed, 0)), r.MaxFrames)
	if err != nil {
		return nil, nil, err
	}
	pcm, err := s.Decode(ctx, frames)
	return pcm, frames, err
}

// Frames generates audio frames from the text model's hidden state h after the prompt, until a
// frame with the end of audio code, limit frames (0: no limit) or a full cache.
func (s *Synth) Frames(ctx context.Context, h []float32, rng *rand.Rand, limit int) ([][]int32, error) {
	var frames [][]int32
	for limit == 0 || len(frames) < limit {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		codes, err := s.frame(ctx, h, frames, rng)
		if err != nil {
			return nil, err
		}
		if slices.Contains(codes, int32(s.depth.Vocab-1)) {
			break
		}
		frames = append(frames, codes)
		if s.sess.Pos() >= s.sess.Cache() {
			break
		}
		fb, err := s.feedback(codes)
		if err != nil {
			return nil, err
		}
		if _, err := s.sess.Step([][]float32{fb}); err != nil {
			return nil, err
		}
		h = s.sess.Hidden()
	}
	return frames, nil
}

// row reads row i of a table, once.
func (s *Synth) row(name string, i int) ([]float32, error) {
	key := fmt.Sprint(name, "/", i)
	if r, ok := s.rows[key]; ok {
		return r, nil
	}
	r, err := s.w.Rows(name, i, i+1)
	if err != nil {
		return nil, err
	}
	if len(s.rows) > 1<<16 {
		clear(s.rows)
	}
	s.rows[key] = r
	return r, nil
}

// frame samples one frame's codes: the codebooks in a random order, revealed as the schedule
// says, each pass of the depthformer reading the codes known so far (the others masked).
func (s *Synth) frame(ctx context.Context, h []float32, hist [][]int32, rng *rand.Rand) ([]int32, error) {
	n, v := s.depth.Codebooks, s.depth.Vocab
	d := s.depth.Hidden()
	order := rng.Perm(n)
	codes := slices.Repeat([]int32{-1}, n)
	mask, _, err := s.w.Get(s.depth.MaskEmbedding())
	if err != nil {
		return nil, err
	}
	k := 0
	for _, pick := range s.depth.Schedule {
		x := make([]float32, n*d)
		for cb, c := range codes {
			e := mask
			if c >= 0 {
				if e, err = s.row(s.depth.Table(cb), int(c)); err != nil {
					return nil, err
				}
			}
			copy(x[cb*d:], e)
		}
		logits, err := s.dprog.Run(ctx, map[string][]float32{"x": x, "h": h})
		if err != nil {
			return nil, err
		}
		for range pick {
			cb := order[k]
			k++
			l := slices.Clone(logits[cb*v : (cb+1)*v])
			s.Sampling.penalize(l, hist, cb)
			codes[cb] = int32(s.Sampling.pick(l, rng))
		}
	}
	return codes, nil
}

// penalize applies the repetition penalty over the codes codebook cb took in the last frames.
func (sp *Sampling) penalize(l []float32, hist [][]int32, cb int) {
	if sp.Repetition <= 1 || sp.Window <= 0 {
		return
	}
	seen := map[int32]bool{}
	for _, f := range hist[max(0, len(hist)-sp.Window):] {
		c := f[cb]
		if c >= 0 && int(c) < len(l) && !seen[c] {
			seen[c] = true
			if l[c] > 0 {
				l[c] /= float32(sp.Repetition)
			} else {
				l[c] *= float32(sp.Repetition)
			}
		}
	}
}

// pick draws a code from logits.
func (sp *Sampling) pick(l []float32, rng *rand.Rand) int {
	if sp.Temperature <= 0 {
		return argmax(l)
	}
	lmin := float32(math.Inf(-1))
	if sp.TopK > 0 && sp.TopK < len(l) {
		sorted := slices.Clone(l)
		slices.SortFunc(sorted, func(a, b float32) int { return cmp.Compare(b, a) })
		lmin = sorted[sp.TopK-1]
	}
	lmax := l[argmax(l)]
	p := make([]float64, len(l))
	var sum float64
	for i, x := range l {
		if x >= lmin {
			p[i] = math.Exp(float64(x-lmax) / sp.Temperature)
			sum += p[i]
		}
	}
	r := rng.Float64() * sum
	for i, x := range p {
		if r -= x; r <= 0 {
			return i
		}
	}
	return len(l) - 1
}

func argmax(l []float32) int {
	best := 0
	for i, x := range l {
		if x > l[best] {
			best = i
		}
	}
	return best
}

// feedback is the text model's next input after a frame: the sum of its codes' rows of the
// feedback table (code + codebook * vocab).
func (s *Synth) feedback(codes []int32) ([]float32, error) {
	fb := make([]float32, s.depth.TextHidden)
	for cb, c := range codes {
		r, err := s.row(s.depth.Feedback(), int(c)+cb*s.depth.Vocab)
		if err != nil {
			return nil, err
		}
		for j, x := range r {
			fb[j] += x
		}
	}
	return fb, nil
}

// Decode turns frames of codes into samples: the codec decoder over chunks of its frames that
// overlap by its look back (the first of a chunk's outputs, computed from a zero state, are
// dropped), then the inverse STFT.
func (s *Synth) Decode(ctx context.Context, frames [][]int32) ([]float32, error) {
	stft, err := DecodeCodec(ctx, s.codec, s.cprog, s.cfr, frames)
	if err != nil {
		return nil, err
	}
	t := NewISTFT(s.codec.NFFT, s.codec.Hop)
	var pcm []float32
	w := 2 * s.codec.Bins()
	for i := 0; i < len(stft); i += w {
		pcm = append(pcm, t.Frame(stft[i:i+w])...)
	}
	return append(pcm, t.Flush()...), nil
}

// DecodeCodec runs the codec decoder program (of fr frames per call) over frames, in chunks
// that overlap by the decoder's look back, and returns the STFT frames (FrameStack per frame,
// 2*Bins each).
func DecodeCodec(ctx context.Context, c *lfm3gen.Codec, p runner.Program, fr int, frames [][]int32) ([]float32, error) {
	overlap := (c.Lookback() + c.FrameStack - 1) / c.FrameStack
	if overlap >= fr {
		return nil, fmt.Errorf("speech: the codec decoder's %d frames per call do not exceed its look back of %d frames", fr, overlap)
	}
	w := 2 * c.Bins() * c.FrameStack // values per frame
	var out []float32
	for e := 0; e < len(frames); {
		start := max(0, e-overlap)
		x := make([]float32, fr*c.CodeDim())
		for f := start; f < min(start+fr, len(frames)); f++ {
			for cb, code := range frames[f] {
				copy(x[(f-start)*c.CodeDim()+cb*len(c.Levels):], c.Codes(int(code)))
			}
		}
		y, err := p.Run(ctx, map[string][]float32{"x": x})
		if err != nil {
			return nil, err
		}
		end := min(start+fr, len(frames))
		out = append(out, y[(e-start)*w:(end-start)*w]...)
		e = end
	}
	return out, nil
}
