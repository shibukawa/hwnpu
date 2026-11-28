package sound

import (
	"context"
	"io"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/sound/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/soundwire"
)

// The sound effects' audio: 48 kHz mono, up to EffectMaxSeconds.
const (
	EffectRate       = soundwire.EffectRate
	EffectMaxSeconds = soundwire.EffectMaxSeconds
	// EffectPromptTokens are the tokens of a description the text encoder reads; the rest is
	// cut.
	EffectPromptTokens = soundwire.EffectPromptTokens
)

// Effect is the use of a model that makes sound effects from a description (MOSS-SoundEffect)
// for an app: a model, a family or a use of the definitions ("sfx", "moss-soundeffect:8gb").
func Effect(name string) ai.Use[*EffectGenerator] {
	return ai.Define[*EffectGenerator](effectRole, name)
}

// The settings a sound effects model's definition may give (ai.Definition):
//
//	settings: text-format  the text encoder's weight format, fp16 or lut8 (default: by the
//	                       model's machine, fp16 from 16 GB)
//	          dit-format   the DiT's weight format, lut8 (the default) or fp16
//
// A model of device cpu runs on hwnpu's CPU interpreter (slow). On AMD's NPU the model runs as
// the graphs of tools/amd/iron_moss_sfx.py (exported on the first preparation; dit-format does
// not change them). The role's Prepare and Load are the engine's (ai/sound/engine).
var effectRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &EffectGenerator{p: airpc.RemoteProxy[effectGeneratorImpl](h)} },
	Name:   "sfx",
	Kinds:  []catalog.Kind{catalog.KindSFX},
	// 1: the decoder's export of this Mac's chip's rewrites (model-recipes.yaml: sin as it is on
	// M3 and later), not the one prepared before; 2: the tokenizer as its binary file
	// (chat.PrepareTokenizer) in the files, as "tokenizer": a store prepared before named none,
	// and Load then read the checkpoint's tokenizer.json, which a checkpoint fetched since is
	// without
	Revision: 2,
}

// EffectGenerator makes sound effects (MOSS-SoundEffect): the text encoder reads the
// description, the DiT denoises 30 seconds of latents, and the decoder turns the duration
// asked for into 48 kHz mono. The text encoder's programs are loaded for each description and
// let go before the DiT's, which are loaded at the first sound and kept. It runs one request at
// a time; others wait.
type EffectGenerator struct {
	p airpc.Proxy[effectGeneratorImpl]
}

// effectGeneratorImpl is a sound effects model of the engine in this process.
type effectGeneratorImpl interface {
	Model() string
	OnCPU() bool
	Programs() string
	Generate(ctx context.Context, description string, o EffectOptions) (EffectSound, error)
	Warm(ctx context.Context) error
	Close() error
}

// Model is the name of the model in the definitions.
func (g *EffectGenerator) Model() string {
	return airpc.ProxyValue(g.p, "Model", effectGeneratorImpl.Model)
}

// OnCPU reports whether the model runs on hwnpu's CPU interpreter (its definition's device cpu).
func (g *EffectGenerator) OnCPU() bool {
	return airpc.ProxyValue(g.p, "OnCPU", effectGeneratorImpl.OnCPU)
}

// Programs names the weight formats of the text encoder's and the DiT's programs, for a log.
func (g *EffectGenerator) Programs() string {
	return airpc.ProxyValue(g.p, "Programs", effectGeneratorImpl.Programs)
}

// EffectOptions are a sound effect's settings.
type EffectOptions struct {
	// Duration is how long the sound lasts, to a tenth of a second, up to EffectMaxSeconds (0:
	// 10 seconds).
	Duration time.Duration
	// Steps are the flow-matching steps (0: 100; each two calls of the DiT: fewer is faster and
	// rougher), CFG how strongly the description guides (0: 4) and Shift the sigma schedule's
	// shift (0: 5).
	Steps      int
	CFG, Shift float64
	// Seed makes the sound repeatable (0: a new one, told in EffectStats.Seed).
	Seed uint64
	// Fade fades the last of the sound out (0: none; a sound effect may end where it should).
	Fade time.Duration
	// Progress, when not nil, is told what is being done, in the call's goroutine.
	Progress func(EffectProgress)
}

// EffectStage is what making a sound effect is doing.
type EffectStage int

const (
	EffectReading EffectStage = iota // the text encoder reads the description
	EffectRead                       // it has (Tokens, Took, Where)
	EffectLoading                    // the DiT's and the decoder's programs are loaded (the first sound)
	EffectLoaded                     // they are (Took, Where)
	EffectStep                       // a step of the DiT is done (Step of Steps, Took the step's)
)

// EffectProgress is how far making a sound effect is.
type EffectProgress struct {
	Stage       EffectStage
	Tokens      int
	Step, Steps int
	Took        time.Duration
	// Where says where the programs just loaded run: "all on the NPU", "NOT all on the NPU",
	// "placement not known" or "on the CPU interpreter".
	Where string
}

// EffectSound is a sound effect made.
type EffectSound struct {
	Samples []float32 // mono, in [-1, 1]
	Rate    int
	Stats   EffectStats
}

// EffectStats are what making a sound effect took.
type EffectStats struct {
	// Seed is the seed of the noise (the one drawn when EffectOptions had none: given again, it
	// makes the same sound).
	Seed uint64
	// Seconds is the sound's length; Steps the DiT's steps.
	Seconds float64
	Steps   int
	// Tokens are the description's tokens, Cut true when it was cut at EffectPromptTokens.
	Tokens int
	Cut    bool
	// Text is the time the text encoder took (its programs' loads with it), Load the DiT's and
	// the decoder's loads (0 after the first sound), Took the generation's.
	Text, Load, Took time.Duration
}

// Generate makes the sound of a description (English or Chinese).
func (g *EffectGenerator) Generate(ctx context.Context, description string, o EffectOptions) (EffectSound, error) {
	return airpc.ProxyCall1(g.p, ctx, "Generate", func(l effectGeneratorImpl) (EffectSound, error) {
		return l.Generate(ctx, description, o)
	}, description, o)
}

// Warm loads the DiT's and the decoder's programs and reads a description once, which compiles
// them all for the NPU (ai.Warmer: the load of "models prepare").
func (g *EffectGenerator) Warm(ctx context.Context) error {
	return airpc.ProxyCall0(g.p, ctx, "Warm", effectGeneratorImpl.Warm)
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (g *EffectGenerator) Close() error { return g.p.Close() }

func init() {
	airole.Register(effectRole)
	bridge.Roles[effectRole.Name] = effectRole
	bridge.Wrap[effectRole.Name] = func(l any) io.Closer { return &EffectGenerator{p: airpc.LocalProxy(l.(effectGeneratorImpl))} }
	airpc.RegisterErrorType[*OffNPUError]("sound.OffNPUError")
}
