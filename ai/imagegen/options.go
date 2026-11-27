package imagegen

import (
	"image"
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
	// the only ones it takes; Anima-Turbo 8, Anima-Aesthetic 25).
	Steps int
	// Negative is the negative prompt of the classifier-free guidance, and Guidance its scale,
	// the DiT twice a step (Anima; 0: the model's own, 1: none). A Negative without a
	// Guidance on a model that runs none by default (Anima-Turbo) runs a weak one (1.5); the
	// distilled models (FLUX.2 and FLUX.3 klein) run none and refuse them.
	Negative string
	Guidance float64
	// CFGSigmas, [lo, hi], run the guidance only at the steps whose sigma is in it (1 is pure
	// noise; 0, 1 for every step); the others run the DiT once. Nil: the model's own (Anima 0.6
	// to 1).
	CFGSigmas []float64
	// Sampler is the solver (Anima: euler or dpm2m, DPM-Solver++ 2M; "": the model's own).
	Sampler string
	// Shift is Anima's sigma schedule's shift (0: 3, its scheduler's).
	Shift float64
	// Batch is how many pictures GenerateAll draws together where the model does (FLUX.2 klein):
	// their prompts are read, then their steps run together
	// (a group of the DiT's weights read once for all of them), then the VAE. 0 and 1 draw a
	// picture after another.
	Batch int

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
