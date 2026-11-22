package imagegen

import (
	"errors"
	"image"
	"math/rand/v2"
	"os"
	"time"
)

// Options are how a picture is drawn.
type Options struct {
	// References are pictures the prompt edits or composes from, up to 4 (FLUX.2 klein): "image
	// 1", "image 2", ... in the prompt. Each is cropped about its centre to the picture's shape
	// and scaled to its size.
	References []image.Image
	// ReferenceScales are the references' sizes as the picture's sides over them, one a
	// reference (missing ones, and 0, are 1: the picture's size): 2 reads a reference at half the
	// picture's sides (a quarter of its rows in each step: an edit at 1024 x 1024 in about two
	// thirds of the time, with less of the reference's fine detail), where the export runs it.
	ReferenceScales []int
	// Width and Height are the picture's size wanted: the nearest of Sizes in shape is drawn.
	// 0: the shape of the first reference, else the first of Sizes.
	Width, Height int
	// Seed is the noise's seed (of the first picture; the i-th's is Seed + i); 0 takes one from
	// the time for each picture (Pictures.Seeds tells them).
	Seed uint64
	// Steps are the denoising steps; 0 is the model's own (FLUX.2 klein: 4; FLUX.3 klein: 16,
	// the only ones it takes; Anima-Turbo 8, Anima-Aesthetic 25; Chroma1-Flash 8, Chroma1-HD 40).
	Steps int
	// Negative is the negative prompt of the classifier-free guidance, and Guidance its scale,
	// the DiT twice a step (Anima, Chroma; 0: the model's own, 1: none). A Negative without a
	// Guidance on a model that runs none by default (Anima-Turbo) runs a weak one (1.5); the
	// distilled models (FLUX.2 and FLUX.3 klein) run none and refuse them.
	Negative string
	Guidance float64
	// CFGSigmas, [lo, hi], run the guidance only at the steps whose sigma is in it (1 is pure
	// noise; 0, 1 for every step); the others run the DiT once. Nil: the model's own (Anima 0.6
	// to 1; Chroma its export's with its own sampler and steps).
	CFGSigmas []float64
	// Sampler is the solver (Anima: euler or dpm2m, DPM-Solver++ 2M; Chroma: euler, heun or
	// dpm2m; "": the model's own).
	Sampler string
	// Shift is Anima's sigma schedule's shift (0: 3, its scheduler's).
	Shift float64
	// Batch is how many pictures GenerateAll draws together where the model does (FLUX.2 klein;
	// Chroma on Apple's Neural Engine): their prompts are read, then their steps run together
	// (a group of the DiT's weights read once for all of them), then the VAE. 0 and 1 draw a
	// picture after another.
	Batch int
	// UnloadDiT unloads Chroma's DiT between the pictures of GenerateAll on Apple's Neural
	// Engine, which keeps it loaded otherwise (the text encoder and the VAE are loaded for each).
	UnloadDiT bool

	// Progress, when not nil, is called after each step (step from 1) and once more when the
	// pictures of the step are decoded (step == steps + 1).
	Progress func(step, steps int)
	// Phase, when not nil, is called as Generate begins each of what it does before its steps
	// (PhaseOpen, PhaseReferences, PhasePrompt), as the steps begin (PhaseSteps) and as the
	// picture is decoded after them (PhaseDecode): the first picture of a size opens its
	// programs, which can take a while with nothing else to show.
	Phase func(phase string)
	// OnBatch, when not nil, is called as the steps of a batch of pictures begin (GenerateAll
	// draws a batch after another; Generate one), OnStep after each of its steps, OnStage after
	// each stage of the VAE's decoder, and Drawn with each picture as it is decoded, before the
	// next batch begins.
	OnBatch func(Batch)
	OnStep  func(Step)
	OnStage func(Stage)
	Drawn   func(i int, img *image.RGBA)

	// Checks are what a model's numbers are checked with against a reference (tools/ref).
	Checks *Checks
}

// Checks are what a developer checks a model's numbers with against a reference implementation
// (tools/ref/ref_*.py): the latents to start from, the latents it ends with, and its
// intermediate values.
type Checks struct {
	// Noise is the initial latents of every picture, in place of the seed's noise (raw float32:
	// Anima (16, H/8, W/8)).
	Noise []float32
	// NoSteps runs no step: Noise is then the final latents, for a check of the VAE alone
	// (FLUX.2 klein).
	NoSteps bool
	// Latents, when not nil, is given each picture's final latents.
	Latents func(i int, x []float32)
	// RefLatents are references' latents (FLUX.2 klein: tokens x 128, normalized), before those
	// of References.
	RefLatents [][]float32
	// Dump is a directory to write the conditioning and each step's values to (and FLUX.2
	// klein's references' latents, ref_N.f32), as the reference's are.
	Dump string
}

// The phases of Generate before its steps (Options.Phase).
const (
	PhaseOpen       = "opening the programs" // of the picture's size, at its first picture
	PhaseReferences = "reading the reference pictures"
	PhasePrompt     = "reading the prompt"
	PhaseSteps      = "drawing"
	PhaseDecode     = "decoding the picture"
)

// Batch is a batch of pictures whose steps begin (Options.OnBatch).
type Batch struct {
	// First is the index of its first prompt, N its pictures, Steps the steps they take.
	First, N, Steps int
	// Seeds are its pictures' seeds (none with Checks.Noise).
	Seeds []uint64
}

// Step is a denoising step of a batch done (Options.OnStep).
type Step struct {
	// Step is from 0, of Steps.
	Step, Steps int
	// From and To are the sigmas the step went between (1: pure noise).
	From, To float64
	// Took is how long it took, for all of the batch's pictures.
	Took time.Duration
}

// Stage is a stage of the VAE's decoder done (Options.OnStage).
type Stage struct {
	Stage, Stages int // from 0
	Took          time.Duration
}

// Pictures are what GenerateAll drew, with what it took.
type Pictures struct {
	// Images are the pictures, a prompt each, and Seeds their seeds (0 with Checks.Noise).
	Images []*image.RGBA
	Seeds  []uint64
	// Size is the size drawn (the nearest of Sizes to Options'), Steps the steps a picture.
	Size  image.Point
	Steps int
	// Batches are the batches drawn together, in order.
	Batches []BatchStats
	// Open is the time opening the size's programs took (0 when they were open), References
	// encoding the references, Prompts the text encoder's (all the prompts', and the negative
	// one's), and Took the whole call.
	Open, References, Prompts, Took time.Duration
}

// BatchStats are what a batch of pictures drawn together took.
type BatchStats struct {
	First, N int
	// Steps are each step's time, for all of its pictures; Denoise all of them.
	Steps   []time.Duration
	Denoise time.Duration
	// Stages are each stage's time of the VAE's decoder (where it tells them); Decode the
	// decoding of all of its pictures.
	Stages []time.Duration
	Decode time.Duration
}

// run is a call of GenerateAll: its prompts, its options and what it took.
type run struct {
	g       *Generator
	prompts []string
	o       Options
	size    image.Point
	out     Pictures
	t0      time.Time
	cur     *BatchStats
}

// newRun is a run of the prompts at the size o asks (of the first reference, else the first).
func (g *Generator) newRun(prompts []string, o Options) *run {
	w, h := o.Width, o.Height
	if (w <= 0 || h <= 0) && len(o.References) > 0 && o.References[0] != nil {
		b := o.References[0].Bounds()
		w, h = b.Dx(), b.Dy()
	}
	r := &run{g: g, prompts: prompts, o: o, size: g.SizeFor(w, h), t0: time.Now()}
	r.out.Size = r.size
	r.out.Images = make([]*image.RGBA, len(prompts))
	r.out.Seeds = make([]uint64, len(prompts))
	return r
}

func (r *run) phase(s string) {
	if r.o.Phase != nil {
		r.o.Phase(s)
	}
}

// checks are the run's Checks, the zero value without.
func (r *run) checks() Checks {
	if r.o.Checks == nil {
		return Checks{}
	}
	return *r.o.Checks
}

// dumpDir is Checks.Dump, made.
func (r *run) dumpDir() (string, error) {
	d := r.checks().Dump
	if d == "" {
		return "", nil
	}
	return d, os.MkdirAll(d, 0o755)
}

// batch begins the batch of prompts lo to hi of steps steps: the noise of each picture (noise of
// its seed, or Checks.Noise).
func (r *run) batch(lo, hi, steps int, noise func(seed uint64) []float32) [][]float32 {
	r.out.Steps = steps
	b := Batch{First: lo, N: hi - lo, Steps: steps}
	xs := make([][]float32, hi-lo)
	for i := lo; i < hi; i++ {
		if n := r.checks().Noise; n != nil {
			xs[i-lo] = append([]float32(nil), n...)
			continue
		}
		s := r.o.Seed
		if s == 0 {
			s = rand.Uint64()
		} else {
			s += uint64(i)
		}
		r.out.Seeds[i] = s
		b.Seeds = append(b.Seeds, s)
		xs[i-lo] = noise(s)
	}
	r.out.Batches = append(r.out.Batches, BatchStats{First: lo, N: hi - lo})
	r.cur = &r.out.Batches[len(r.out.Batches)-1]
	r.phase(PhaseSteps)
	if r.o.OnBatch != nil {
		r.o.OnBatch(b)
	}
	return xs
}

// step is told step i of n done in d, between sigmas from and to.
func (r *run) step(i, n int, from, to float64, d time.Duration) {
	r.cur.Steps = append(r.cur.Steps, d)
	r.cur.Denoise += d
	if r.o.OnStep != nil {
		r.o.OnStep(Step{Step: i, Steps: n, From: from, To: to, Took: d})
	}
	if r.o.Progress != nil {
		r.o.Progress(i+1, n)
	}
}

// stepsOf is a pipeline's step function (i of n, d) over sigmas.
func (r *run) stepsOf(sig []float64) func(i, n int, d time.Duration) {
	return func(i, n int, d time.Duration) {
		from, to := 0.0, 0.0
		if i+1 < len(sig) {
			from, to = sig[i], sig[i+1]
		}
		r.step(i, n, from, to, d)
	}
}

// stage is a pipeline's stage function of the VAE's decoder.
func (r *run) stage(i, n int, d time.Duration) {
	r.cur.Stages = append(r.cur.Stages, d)
	if r.o.OnStage != nil {
		r.o.OnStage(Stage{Stage: i, Stages: n, Took: d})
	}
}

// latents tells Checks.Latents the final latents of the pictures from lo.
func (r *run) latents(lo int, xs ...[]float32) {
	if fn := r.checks().Latents; fn != nil {
		for i, x := range xs {
			fn(lo+i, x)
		}
	}
}

// drawn keeps the pictures from lo, decoded in d, and tells them.
func (r *run) drawn(lo int, d time.Duration, imgs ...*image.RGBA) error {
	if len(imgs) == 0 {
		return errors.New("ai/imagegen: no picture decoded")
	}
	r.cur.Decode = d
	for i, img := range imgs {
		r.out.Images[lo+i] = img
	}
	if r.o.Progress != nil {
		r.o.Progress(r.out.Steps+1, r.out.Steps)
	}
	if r.o.Drawn != nil {
		for i, img := range imgs {
			r.o.Drawn(lo+i, img)
		}
	}
	return nil
}

// finish is what the run drew.
func (r *run) finish() *Pictures {
	r.out.Took = time.Since(r.t0)
	return &r.out
}

// opened adds the time since t0, of opening a size's programs, to Pictures.Open.
func (r *run) opened(t0 time.Time) { r.out.Open += time.Since(t0) }
