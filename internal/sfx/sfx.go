// Package sfx generates sound effects with MOSS-SoundEffect v2.0 (OpenMOSS-Team/MOSS-SoundEffect-v2.0,
// Apache-2.0) on the NPU: Qwen3-1.7B reads the prompt, a Wan 2.1 DiT of 1.3B parameters denoises
// 30 seconds of 128-channel latents (50 a second) in flow-matching Euler steps with
// classifier-free guidance, and the DAC VAE's decoder turns them into 48 kHz mono, of which the
// duration asked for is kept. The host tokenizes, looks up the embedding rows, applies the text
// encoder's final norm, computes the time embedding and the modulation of each step, and steps
// the latents.
//
// The structures follow OpenMOSS/MOSS-TTS moss_soundeffect_v2 (diffsynth's WanAudioModel, DAC
// and Qwen3TextEncoder, and its pipeline); tools/ref/ref_moss_sfx.py is the reference.
package sfx

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/qwen3"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// The files of the repository the programs are converted from and run with: the text
// encoder's checkpoint stays for its embedding table and final norm, the DiT's for the host's
// time embedding.
var Files = []string{
	"model_index.json",
	"scheduler/scheduler_config.json",
	"tokenizer/tokenizer.json",
	"tokenizer/tokenizer_config.json",
	"text_encoder/config.json",
	"text_encoder/model.safetensors.index.json",
	"text_encoder/model-00001-of-00002.safetensors",
	"text_encoder/model-00002-of-00002.safetensors",
	"transformer/config.json",
	"transformer/diffusion_pytorch_model.safetensors",
	"vae/config.json",
	"vae/vae_128d_48k.pth",
}

// Revision counts the changes to the programs that call for converting them again (in their
// names).
const Revision = 1

// The defaults of the model card: 100 steps, guidance 4, shift 5.
const (
	DefaultSteps = 100
	DefaultCFG   = 4.0
	DefaultShift = 5.0
	// MaxSeconds is the longest sound (max_inference_seconds).
	MaxSeconds = 30
	// DitGroup is the DiT's blocks a program holds.
	DitGroup = 5
)

// Formats are the text encoder's and the DiT's weight formats for a machine of memory GB: the
// DiT in lut8 (2.7 % off the reference's velocity against fp16's 2.1 %, 1.3 s a call against
// 1.7 s, half the memory); the text encoder in fp16 (0.7 % off, 2.8 GB) from 16 GB, lut8 with
// TextFP16 (4.9 %, 1.7 GB) under.
func Formats(memory int) (text, dit string) {
	if memory >= 16 {
		return "fp16", "lut8"
	}
	return "lut8", "lut8"
}

// Setup says what Prepare downloads and converts.
type Setup struct {
	Dir, Repo string
	// Machine is the memory, in GB, of the machine the model is picked for (its column of the
	// definitions: moss-soundeffect:8g), which Formats chooses by.
	Machine    int
	TextFormat string // fp16 or lut8 (Formats by default)
	DitFormat  string // lut8 or fp16
	Log        io.Writer
}

// Paths are the files and programs of a prepared model.
type Paths struct {
	Dir     string
	Text    []string // the text encoder's parts
	Context string
	Blocks  []string // the DiT's groups, DitGroup blocks each
	Stages  []string // the decoder's stages
	TextCfg *qwen3.Config
}

func (p Paths) at(f string) string { return filepath.Join(p.Dir, filepath.FromSlash(f)) }

// Prepare downloads the files unless they are there and converts the programs unless they are.
func Prepare(ctx context.Context, s Setup) (Paths, error) {
	log := cmp.Or[io.Writer](s.Log, io.Discard)
	text, dit := Formats(cmp.Or(s.Machine, chat.MachineMemory()))
	text, dit = cmp.Or(s.TextFormat, text), cmp.Or(s.DitFormat, dit)
	p := Paths{Dir: s.Dir}
	var want []chat.File
	for _, f := range Files {
		want = append(want, chat.File{Repo: s.Repo, Name: f, Path: p.at(f)})
	}
	chat.Expect(ctx, want)
	for _, f := range Files {
		path := p.at(f)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return p, err
		}
		if err := chat.Fetch(ctx, s.Repo, f, path, log); err != nil {
			return p, err
		}
	}
	cfg, err := TextConfig(s.Dir)
	if err != nil {
		return p, err
	}
	p.TextCfg = cfg
	name := func(kind, format string) string {
		return p.at(fmt.Sprintf("hwnpu-%s-r%d-%s.mlpackage", kind, Revision, format))
	}
	for i := range TextSplit(text) {
		p.Text = append(p.Text, name(fmt.Sprintf("text-%d", i), text))
	}
	p.Context = name("dit-context", "fp16")
	for first := 0; first < DitLayers; first += DitGroup {
		p.Blocks = append(p.Blocks, name(fmt.Sprintf("dit-blocks-%d", first), dit))
	}
	for _, st := range DACStages() {
		p.Stages = append(p.Stages, name(fmt.Sprintf("dac-%d", st.Index), "fp16"))
	}

	type job struct {
		path, format string
		part         llm.Model
		w            func() (*Weights, error)
		opts         llm.Options
	}
	var tw, dw, vw *Weights
	defer func() {
		for _, w := range []*Weights{tw, dw, vw} {
			if w != nil {
				w.Close()
			}
		}
	}()
	lazy := func(to **Weights, pth []string, st map[string][]string) func() (*Weights, error) {
		return func() (*Weights, error) {
			var err error
			if *to == nil {
				*to, err = OpenWeights(pth, st)
			}
			return *to, err
		}
	}
	textW := lazy(&tw, nil, map[string][]string{"": {p.at(Files[6]), p.at(Files[7])}})
	ditW := lazy(&dw, nil, map[string][]string{"dit.": {p.at(Files[9])}})
	vaeW := lazy(&vw, []string{p.at(Files[11])}, nil)
	var jobs []job
	start := 0
	for i, n := range TextSplit(text) {
		o := TextOptions(text)
		o.Seq, o.Start, o.Layers, o.NoHead = PromptRows, start, n, true
		jobs = append(jobs, job{p.Text[i], text, cfg, textW, o})
		start += n
	}
	jobs = append(jobs, job{p.Context, "fp16", Context(), ditW, llm.Options{}})
	for i, first := 0, 0; first < DitLayers; i, first = i+1, first+DitGroup {
		jobs = append(jobs, job{p.Blocks[i], dit, Blocks(first, DitGroup), ditW, llm.Options{}})
	}
	for i, st := range DACStages() {
		jobs = append(jobs, job{p.Stages[i], "fp16", st.Part(), vaeW, llm.Options{}})
	}
	todo := 0
	for _, j := range jobs {
		if _, err := os.Stat(j.path + ".done"); err != nil {
			todo++
		}
	}
	done := 0
	for _, j := range jobs {
		if _, err := os.Stat(j.path + ".done"); err == nil {
			continue
		}
		done++
		w, err := j.w()
		if err != nil {
			return p, err
		}
		bar := console.NewBar(log, fmt.Sprintf("  [%2d/%d] converting %s", done, todo, filepath.Base(j.path)))
		os.RemoveAll(j.path)
		opts := j.opts
		opts.Backend, opts.Format = llm.BackendCoreML, j.format
		if _, err := llm.WritePart(j.path, j.part, w, opts, nil); err != nil {
			bar.Done("failed")
			return p, err
		}
		bar.Done(console.Duration(bar.Elapsed()) + ", " + j.format)
		if err := os.WriteFile(j.path+".done", nil, 0o644); err != nil {
			return p, err
		}
	}
	return p, nil
}

// Generator generates sound effects.
type Generator struct {
	p     Paths
	dev   *hwnpu.Device
	tok   *tokenizer.Tokenizer
	textW *Weights // the embedding table and the final norm
	ditW  *Weights // the host's tensors
	dit   *DiT
	dac   []Program
	progs []runner.Program
}

// Open reads the tokenizer and the weights the host computes with; Load loads the DiT's and
// the decoder's programs, and Encode the text encoder's for a prompt.
func Open(p Paths, dev *hwnpu.Device) (*Generator, error) {
	g := &Generator{p: p, dev: dev}
	var err error
	if g.tok, err = tokenizer.Load(p.at("tokenizer/tokenizer.json")); err != nil {
		return nil, err
	}
	if g.textW, err = OpenWeights(nil, map[string][]string{"": {p.at(Files[6]), p.at(Files[7])}}); err != nil {
		return nil, err
	}
	if g.ditW, err = OpenWeights(nil, map[string][]string{"dit.": {p.at(Files[9])}}); err != nil {
		g.Close()
		return nil, err
	}
	timing, err := LoadTiming(g.ditW, "dit.")
	if err != nil {
		g.Close()
		return nil, err
	}
	g.dit = &DiT{Timing: timing}
	return g, nil
}

// Load loads the DiT's and the decoder's programs on the device (nil: the CPU interpreter).
// loaded is called after each.
func (g *Generator) Load(ctx context.Context, loaded func(name string, p runner.Program, took time.Duration)) error {
	open := func(path string) (runner.Program, error) {
		t0 := time.Now()
		pr, err := runner.Open(ctx, g.dev, path)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", filepath.Base(path), err)
		}
		g.progs = append(g.progs, pr)
		if loaded != nil {
			loaded(filepath.Base(path), pr, time.Since(t0))
		}
		return pr, nil
	}
	var err error
	if g.dit.Ctx, err = open(g.p.Context); err != nil {
		return err
	}
	for _, b := range g.p.Blocks {
		pr, err := open(b)
		if err != nil {
			return err
		}
		g.dit.Groups = append(g.dit.Groups, pr)
		g.dit.Sizes = append(g.dit.Sizes, DitGroup)
	}
	for _, s := range g.p.Stages {
		pr, err := open(s)
		if err != nil {
			return err
		}
		g.dac = append(g.dac, pr)
	}
	return nil
}

// Close lets the programs and the weights go.
func (g *Generator) Close() error {
	for _, p := range g.progs {
		p.Close()
	}
	g.progs = nil
	for _, w := range []*Weights{g.textW, g.ditW} {
		if w != nil {
			w.Close()
		}
	}
	return nil
}

// Encode is the text encoder's output for a prompt and a duration (Prompt): the text encoder's
// parts are loaded, run and let go. loaded is called after each part.
func (g *Generator) Encode(ctx context.Context, prompt string, seconds float64, loaded func(name string, p runner.Program, took time.Duration)) ([]float32, []int32, error) {
	var parts []Program
	var progs []runner.Program
	defer func() {
		for _, p := range progs {
			p.Close()
		}
	}()
	for _, path := range g.p.Text {
		t0 := time.Now()
		pr, err := runner.Open(ctx, g.dev, path)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", filepath.Base(path), err)
		}
		progs = append(progs, pr)
		parts = append(parts, pr)
		if loaded != nil {
			loaded(filepath.Base(path), pr, time.Since(t0))
		}
	}
	e := NewTextEncoder(g.tok, g.textW, g.p.TextCfg, parts)
	return e.Encode(ctx, Prompt(prompt, seconds))
}

// Noise is the initial latents (Frames x 128, rows) of a seed: standard normal.
func Noise(seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	x := make([]float32, Frames*LatentDim)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	return x
}

// Options are a generation's settings.
type Options struct {
	Steps        int
	CFG, Shift   float64
	Seed         uint64
	Noise        []float32 // in place of the seed's (Frames x 128, rows)
	Step         func(i, steps int, took time.Duration)
	DecodeLoaded func()
}

// Generate is the samples (48 kHz mono) of seconds of sound for the text encoder's output.
func (g *Generator) Generate(ctx context.Context, text []float32, seconds float64, o Options) ([]float32, error) {
	if seconds <= 0 || seconds > MaxSeconds {
		return nil, fmt.Errorf("sfx: %.1f s: from 0 to %d seconds", seconds, MaxSeconds)
	}
	o.Steps = cmp.Or(o.Steps, DefaultSteps)
	if o.CFG == 0 {
		o.CFG = DefaultCFG
	}
	if o.Shift == 0 {
		o.Shift = DefaultShift
	}
	cond, err := g.dit.Context(ctx, text)
	if err != nil {
		return nil, err
	}
	cond = append([]float32(nil), cond...)
	uncond, err := g.dit.Context(ctx, make([]float32, TextRows*TextDim))
	if err != nil {
		return nil, err
	}
	uncond = append([]float32(nil), uncond...)
	x := o.Noise
	if x == nil {
		x = Noise(o.Seed)
	}
	if len(x) != Frames*LatentDim {
		return nil, errors.New("sfx: the noise is not 1500 x 128")
	}
	x, err = g.dit.Denoise(ctx, x, cond, uncond, Sigmas(o.Steps, o.Shift), o.CFG, func(i int, took time.Duration) {
		if o.Step != nil {
			o.Step(i, o.Steps, took)
		}
	})
	if err != nil {
		return nil, err
	}
	z := channels(x, Frames, LatentDim)
	samples := int(math.Round(seconds*10)) * SampleRate / 10
	return DecodeLatents(ctx, g.dac, z, Frames, Frames, samples)
}

// channels is x (n rows of c) channel major (c x n).
func channels(x []float32, n, c int) []float32 {
	y := make([]float32, len(x))
	for j := range n {
		for i := range c {
			y[i*n+j] = x[j*c+i]
		}
	}
	return y
}

// String names the formats of the paths, for the log.
func (p Paths) String() string {
	f := func(path string) string {
		b := strings.TrimSuffix(filepath.Base(path), ".mlpackage")
		return b[strings.LastIndex(b, "-")+1:]
	}
	return fmt.Sprintf("text encoder %s, DiT %s", f(p.Text[0]), f(p.Blocks[0]))
}
