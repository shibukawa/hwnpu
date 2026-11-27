package text

import (
	"context"
	"errors"
	"io"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/text/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
)

// Decide is the use of a decision model (Liquid AI's d1) for an app: a model, a family or a use
// of the definitions ("decision", "d1", "d1-omni"). It answers typed questions about a state, as
// the System One API of llama.cpp's server does.
func Decide(name string) ai.Use[*Decider] {
	return ai.Define[*Decider](decideRole, name)
}

// The settings and programs a decision model's definition may give (ai.Definition):
//
//	settings: seqs         the prompt lengths of the programs, comma separated (512): a prompt
//	                       runs on the shortest that holds it
//	          format       the programs' weight format (lut8)
//	programs: packages     the model converted by hwnpu-dev convert, with different -seq; with
//	                       config, weights, tokenizer and calibration (default: next to config)
//	          vision       the vision encoder, a package per image size: questions about images
//	          audio        the audio encoder, a package per clip length (d1-Omni)
//
// A model of device cpu runs on hwnpu's CPU interpreter (slow; for checking). On AMD's NPU a
// model runs from its export of tools/amd/iron_export_lm.py d1 (its recipe, model-recipes.yaml).
// decideRole's Prepare and Load are the engine's (ai/text/engine).
var decideRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &Decider{remote: h} },
	Name:   "decide",
	Kinds:  []catalog.Kind{catalog.KindDecision},
	// 1: the tokenizer as its binary file (chat.PrepareTokenizer) in the files, as "tokenizer":
	// a store prepared before named none, and Load then read the checkpoint's tokenizer.json, which
	// a checkpoint fetched since is without
	Revision: 1,
}

var (
	// ErrBadRequest is the error of a request that is not one of the System One API.
	ErrBadRequest = errors.New("text: a bad request")
	// ErrUnsupported is the error of a request of what the model does not do (video).
	ErrUnsupported = errors.New("text: not supported")
)

// Decision is the answers to a request.
type Decision struct {
	// JSON is the response as llama.cpp's server writes it.
	JSON []byte
	// Questions are the request's questions, InputTokens the tokens read for them.
	Questions, InputTokens int
	Took                   time.Duration
}

// Decider answers typed questions about a state with a decision model. It runs one question at a
// time.
type Decider struct {
	remote *airpc.Handle // the model in hwnpud (ai's HWNPU_DAEMON), nil when it is here
	local  deciderImpl   // the engine's (ai/text/engine), when it is here
}

// deciderImpl is a decision model of the engine in this process.
type deciderImpl interface {
	Model() string
	Name() string
	MaxTokens() int
	OnCPU() bool
	Decide(ctx context.Context, request []byte) (Decision, error)
	Close() error
}

// Model is the name of the model in the definitions, Name the one it calls itself.
func (d *Decider) Model() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Model")
	}
	return d.local.Model()
}
func (d *Decider) Name() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Name")
	}
	return d.local.Name()
}

// MaxTokens is the most tokens of a question's prompt.
func (d *Decider) MaxTokens() int {
	if d.remote != nil {
		return airpc.Value[int](d.remote, "MaxTokens")
	}
	return d.local.MaxTokens()
}

// OnCPU reports whether it runs on the CPU interpreter.
func (d *Decider) OnCPU() bool {
	if d.remote != nil {
		return airpc.Value[bool](d.remote, "OnCPU")
	}
	return d.local.OnCPU()
}

// Decide answers a request of the System One API: {"state": ..., "questions": {"id": {"type":
// "choice" | "score" | "noul", "instructions": ..., "criteria": ...}}}, as llama.cpp's
// tools/server/README.md describes it. A request that is not one is ErrBadRequest.
func (d *Decider) Decide(ctx context.Context, request []byte) (Decision, error) {
	if d.remote != nil {
		return airpc.Call1[Decision](d.remote, ctx, "Decide", request)
	}
	return d.local.Decide(ctx, request)
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (d *Decider) Close() error {
	if d.remote != nil {
		return d.remote.Release() // the Runtime in hwnpud unloads it
	}
	return d.local.Close()
}

func init() {
	airole.Register(decideRole)
	bridge.Roles[decideRole.Name] = decideRole
	bridge.Wrap[decideRole.Name] = func(l any) io.Closer { return &Decider{local: l.(deciderImpl)} }
	airpc.RegisterError("text.ErrBadRequest", ErrBadRequest)
	airpc.RegisterError("text.ErrUnsupported", ErrUnsupported)
}
