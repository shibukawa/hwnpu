package music

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// The files of google/magenta-realtime-2 the small model needs: the checkpoint (the
// depthformer and SpectroStream's decoder and quantizer), MusicCoCa's text side (TextFiles), its
// audio side (AudioStyleFiles) and SpectroStream's encoder (EncoderFiles).
var (
	Files     = []string{"checkpoints/mrt2_small.safetensors"}
	TextFiles = []string{
		"resources/musiccoca/text_encoder.tflite",
		"resources/musiccoca/mapper.tflite",
		"resources/musiccoca/pretrained_vector_quantizer.tflite",
		"resources/musiccoca/spm.model",
	}
	AudioStyleFiles = []string{
		"resources/musiccoca/audio_preprocessor.tflite",
		"resources/musiccoca/music_encoder.tflite",
		"resources/musiccoca/pretrained_vector_quantizer.tflite",
	}
	EncoderFiles = []string{"resources/spectrostream/encoder.safetensors"}
)

// Revision counts the changes to the language model's programs that call for converting them
// again; ssRevision SpectroStream's (2: their inputs laid out on the host; 4: the decoder's
// blocks scaled into fp16's range).
const (
	Revision   = 1
	ssRevision = 4
)

// DecodeFrames are the frames a call of the decoder or the encoder writes; decodeContext the
// frames before them it reads again (more than its receptive field).
const (
	DecodeFrames  = 25
	decodeContext = 16
)

// Setup says what Prepare downloads and converts.
type Setup struct {
	Dir, Repo string
	// Format is the depthformer's weight format (lut8 by default); the other programs are fp16.
	Format  string
	Backend string
	Log     io.Writer
	// Text prepares MusicCoCa's text side (styles from texts), AudioStyle its audio side
	// (styles from audio), Encoder SpectroStream's encoder (continuing a clip).
	Text, AudioStyle, Encoder bool
}

// Paths are the files and programs of a prepared model ("" for the parts not prepared).
type Paths struct {
	Dir, Checkpoint          string
	Temporal, Depth, Decoder string
	// SpectroStream's encoder: its weights and its program
	EncoderWeights, Encoder string
	// MusicCoCa: the models, and the programs of the text and the music encoders (the music
	// encoder's patches the indexes of its input each of its program's inputs takes)
	TextEncoder, MusicEncoder, AudioPreprocessor, Mapper, Quantizer, SPM string
	TextProgram, TextTable, MusicProgram, MusicPatches                   string
}

func (s Setup) files() []string {
	files := slices.Clone(Files)
	add := func(fs []string) {
		for _, f := range fs {
			if !slices.Contains(files, f) {
				files = append(files, f)
			}
		}
	}
	if s.Text {
		add(TextFiles)
	}
	if s.AudioStyle {
		add(AudioStyleFiles)
	}
	if s.Encoder {
		add(EncoderFiles)
	}
	return files
}

// Prepare downloads the files unless they are there and converts the programs unless they are.
func Prepare(ctx context.Context, s Setup) (Paths, error) {
	log := cmp.Or[io.Writer](s.Log, io.Discard)
	format := cmp.Or(s.Format, "lut8")
	at := func(f string) string { return filepath.Join(s.Dir, filepath.FromSlash(f)) }
	p := Paths{Dir: s.Dir, Checkpoint: at(Files[0])}
	var want []chat.File
	for _, f := range s.files() {
		want = append(want, chat.File{Repo: s.Repo, Name: f, Path: at(f)})
	}
	chat.Expect(ctx, want)
	for _, f := range s.files() {
		path := at(f)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return p, err
		}
		if err := chat.Fetch(ctx, s.Repo, f, path, log); err != nil {
			return p, err
		}
	}
	if s.Backend == llm.BackendIntel {
		return p, fmt.Errorf("music: Intel's NPU is not supported yet")
	}
	name := func(kind, f string) string {
		return filepath.Join(s.Dir, fmt.Sprintf("mrt2-small-%s-r%d-%s.mlpackage", kind, Revision, f))
	}
	p.Temporal, p.Depth = name("temporal", format), name("depth", format)
	ssName := func(kind string) string {
		return filepath.Join(s.Dir, fmt.Sprintf("mrt2-small-%s-t%d-r%d-fp16.mlpackage", kind, DecodeFrames+decodeContext, ssRevision))
	}
	p.Decoder = ssName("decoder")
	if s.Text || s.AudioStyle {
		p.Quantizer = at(TextFiles[2])
	}
	if s.Text {
		p.TextEncoder, p.Mapper, p.SPM = at(TextFiles[0]), at(TextFiles[1]), at(TextFiles[3])
		p.TextProgram = filepath.Join(s.Dir, fmt.Sprintf("mrt2-musiccoca-text-r%d-fp16.mlpackage", musicCoCaRevision))
		p.TextTable = strings.TrimSuffix(p.TextProgram, ".mlpackage") + ".table"
	}
	if s.AudioStyle {
		p.AudioPreprocessor, p.MusicEncoder = at(AudioStyleFiles[0]), at(AudioStyleFiles[1])
		p.MusicProgram = filepath.Join(s.Dir, fmt.Sprintf("mrt2-musiccoca-music-r%d-fp16.mlpackage", musicCoCaRevision))
		p.MusicPatches = strings.TrimSuffix(p.MusicProgram, ".mlpackage") + ".patches"
	}
	if s.Encoder {
		p.EncoderWeights = at(EncoderFiles[0])
		p.Encoder = ssName("encoder")
	}
	var lw, sw, ew *Weights
	defer func() {
		for _, w := range []*Weights{lw, sw, ew} {
			if w != nil {
				w.Close()
			}
		}
	}()
	type job struct {
		path   string
		prog   func() (Program, error)
		w      func() (*Weights, error)
		format string
	}
	open := func(to **Weights, path, prefix string) func() (*Weights, error) {
		return func() (*Weights, error) {
			var err error
			if *to == nil {
				*to, err = OpenWeights(map[string]string{path: prefix})
			}
			return *to, err
		}
	}
	none := func() (*Weights, error) { return nil, nil }
	fixed := func(pr Program) func() (Program, error) { return func() (Program, error) { return pr, nil } }
	jobs := []job{
		{p.Temporal, fixed(Temporal{}), open(&lw, p.Checkpoint, "params/"), format},
		{p.Depth, fixed(Depth{}), open(&lw, p.Checkpoint, "params/"), format},
		{p.Decoder, fixed(decoderProgram{&Decoder{Frames: DecodeFrames + decodeContext}}), open(&sw, p.Checkpoint, "params/soundstream/"), "fp16"},
	}
	if s.Encoder {
		jobs = append(jobs, job{p.Encoder, fixed(&Encoder{Frames: DecodeFrames + decodeContext}), open(&ew, p.EncoderWeights, "params/"), "fp16"})
	}
	if s.Text {
		jobs = append(jobs, job{p.TextProgram, func() (Program, error) { return textProgram(p.TextEncoder, p.TextTable) }, none, "fp16"})
	}
	if s.AudioStyle {
		jobs = append(jobs, job{p.MusicProgram, func() (Program, error) { return musicProgram(p.MusicEncoder, p.MusicPatches) }, none, "fp16"})
	}
	for _, c := range jobs {
		if _, err := os.Stat(c.path + ".done"); err == nil {
			continue
		}
		w, err := c.w()
		if err != nil {
			return p, err
		}
		bar := console.NewBar(log, "converting "+filepath.Base(c.path))
		prog, err := c.prog()
		if err != nil {
			bar.Done("failed")
			return p, err
		}
		os.RemoveAll(c.path)
		opts := llm.Options{Format: c.format}
		if c.format == "lut8" || c.format == "lut6" || c.format == "lut4" {
			opts.Quant = "kmeans-minmax"
		}
		var lwArg llm.Weights
		if w != nil {
			lwArg = w
		}
		if err := WriteProgram(c.path, prog, lwArg, opts); err != nil {
			bar.Done("failed")
			return p, err
		}
		bar.Done(console.Duration(bar.Elapsed()))
		if err := os.WriteFile(c.path+".done", nil, 0o644); err != nil {
			return p, err
		}
	}
	return p, nil
}

// decoderProgram is the decoder as a Program of one output.
type decoderProgram struct{ d *Decoder }

func (p decoderProgram) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	ins, y, err := p.d.Build(b, w, opts)
	return ins, []mlpackage.Feature{y}, err
}

// Generator generates music.
type Generator struct {
	lw, sw                    *Weights
	temporal, depth, dec, enc *step
	Log                       io.Writer
	Timing                    struct{ LM, Decode, Encode time.Duration }
}

// SpectroStreamUnits are where SpectroStream's decoder and encoder run unless told: the Neural
// Engine, as everything (the GPU, "gpu", and the CPU, "cpu", for comparing).
func SpectroStreamUnits() string { return "" }

// Open loads the programs on dev (nil: the CPU interpreter); the encoder's when prepared;
// SpectroStream's on SpectroStreamUnits.
func Open(ctx context.Context, dev *hwnpu.Device, p Paths) (*Generator, error) {
	return OpenOn(ctx, dev, p, SpectroStreamUnits())
}

// OpenOn is Open with SpectroStream's decoder and encoder on the Core ML units given ("gpu",
// "cpu"; "": the Neural Engine).
func OpenOn(ctx context.Context, dev *hwnpu.Device, p Paths, ssUnits string) (*Generator, error) {
	g := &Generator{}
	var err error
	if g.lw, err = OpenWeights(map[string]string{p.Checkpoint: "params/"}); err != nil {
		return nil, err
	}
	if g.sw, err = OpenWeights(map[string]string{p.Checkpoint: "params/soundstream/"}); err != nil {
		g.Close()
		return nil, err
	}
	type prog struct {
		path, units string
		to          **step
	}
	progs := []prog{{p.Temporal, "", &g.temporal}, {p.Depth, "", &g.depth}, {p.Decoder, ssUnits, &g.dec}}
	if p.Encoder != "" {
		progs = append(progs, prog{p.Encoder, ssUnits, &g.enc})
	}
	for _, s := range progs {
		if *s.to, err = openStepOn(ctx, dev, s.path, s.units); err != nil {
			g.Close()
			return nil, err
		}
	}
	// SpectroStream's programs run once on silence, so that a failure shows now and not in
	// the middle of the music
	W := DecodeFrames + decodeContext
	for _, s := range []*step{g.dec, g.enc} {
		if s == nil {
			continue
		}
		size := EmbeddingSize * W
		if s == g.enc {
			size = 4 * timeStride * W * stftBins
		}
		in := map[string][]float32{"x": make([]float32, size)}
		ssMasks(in, make([]float32, W))
		if _, err := s.run(in); err != nil {
			g.Close()
			return nil, fmt.Errorf("music: %s fails on the %s: %w", s.name, cmp.Or(s.units, "Neural Engine"), err)
		}
	}
	return g, nil
}

// OnNPU reports whether all the programs meant for the NPU run on it (those loaded on other
// units are not).
func (g *Generator) OnNPU() bool { return len(g.Placement()) == 0 }

// Elsewhere names the programs loaded on other units than the Neural Engine, and the units.
func (g *Generator) Elsewhere() []string {
	var out []string
	for _, s := range []*step{g.temporal, g.depth, g.dec, g.enc} {
		if s != nil && s.units != "" {
			out = append(out, s.name+" on the "+s.units)
		}
	}
	return out
}

// Placement says, for each program not all on the NPU, what Core ML puts elsewhere.
func (g *Generator) Placement() []string { return placement(g.temporal, g.depth, g.dec, g.enc) }

func placement(steps ...*step) []string {
	var out []string
	for _, s := range steps {
		if s != nil && s.units == "" && !s.onNPU {
			w := s.where
			if w == "" {
				w = "the CPU interpreter"
			}
			out = append(out, s.name+": "+w)
		}
	}
	return out
}

func (g *Generator) Close() error {
	for _, s := range []*step{g.temporal, g.depth, g.dec, g.enc} {
		if s != nil {
			s.Close()
		}
	}
	if g.lw != nil {
		g.lw.Close()
	}
	if g.sw != nil {
		g.sw.Close()
	}
	return nil
}

// Generate writes frames of music (25 a second) for a conditioning, sampling each codebook with
// pick, and returns the stereo samples at 48 kHz, interleaved, and the frames' codes.
func (g *Generator) Generate(c Conditioning, frames int, pick Sampler) ([]float32, [][]int, error) {
	s := g.NewSession()
	t0 := time.Now()
	for f := range frames {
		if _, err := s.Step(c, pick); err != nil {
			return nil, nil, err
		}
		if g.Log != nil && (f+1)%25 == 0 {
			fmt.Fprintf(g.Log, "\r%d / %d frames (%.0f ms a frame)", f+1, frames, float64(time.Since(t0).Milliseconds())/float64(f+1))
		}
	}
	if g.Log != nil {
		fmt.Fprintln(g.Log)
	}
	g.Timing.LM = time.Since(t0)
	t0 = time.Now()
	wav, err := s.Audio(true)
	g.Timing.Decode = time.Since(t0)
	return wav, s.Codes(), err
}

// Decode turns frames of codes into stereo samples, as SpectroStream's decoder and inverse STFT
// do over the whole sequence: frames T give (4T - 4) * 480 samples (the decoder's lookahead of
// one frame dropped).
func (g *Generator) Decode(codes [][]int) ([]float32, error) {
	s := g.NewSession()
	s.codes = codes
	return s.Audio(true)
}

// EncodeAudio is SpectroStream's codes (12 levels) of stereo samples at 48 kHz, interleaved, as
// its STFT, encoder and quantizer give them over the whole clip: a frame every 1920 samples, the
// samples after the last whole frame left out.
func (g *Generator) EncodeAudio(samples []float32) ([][]int, error) {
	if g.enc == nil {
		return nil, fmt.Errorf("music: SpectroStream's encoder is not prepared")
	}
	t0 := time.Now()
	defer func() { g.Timing.Encode += time.Since(t0) }()
	P := len(samples) / 2 / SamplesPerFrame
	if P == 0 {
		return nil, nil
	}
	spec, n := STFT(samples[:2*P*SamplesPerFrame])
	if n != timeStride*P {
		return nil, fmt.Errorf("music: %d STFT frames for %d frames", n, P)
	}
	W := DecodeFrames + decodeContext
	x := make([]float32, 4*timeStride*W*stftBins)
	mask := make([]float32, W)
	codes := make([][]int, P)
	for start := 0; start < P; start += DecodeFrames {
		clear(x)
		clear(mask)
		for i := range W {
			f := start - decodeContext + i
			if f < 0 || f >= P {
				continue
			}
			mask[i] = 1
			for c := range 4 {
				for s := range timeStride {
					copy(x[(c*timeStride*W+timeStride*i+s)*stftBins:][:stftBins], spec[(c*n+timeStride*f+s)*stftBins:][:stftBins])
				}
			}
		}
		in := map[string][]float32{"x": x}
		ssMasks(in, mask)
		out, err := g.enc.run(in)
		if err != nil {
			return nil, err
		}
		emb := make([]float32, EmbeddingSize)
		for i := decodeContext; i < W; i++ {
			f := start - decodeContext + i
			if f >= P {
				break
			}
			for c := range emb { // y is channels first: (256, W)
				emb[c] = out["y"][c*W+i]
			}
			if codes[f], err = Quantize(g.sw, emb, Codebooks); err != nil {
				return nil, err
			}
		}
	}
	return codes, nil
}
