package text

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"io"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/systemone"
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
var decideRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &Decider{remote: h} },
	Name:   "decide",
	Kinds:  []chat.Kind{chat.KindDecision},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		m, ps := e.Model, e.Model.Programs
		if e.NPU.AMD() && len(ps["packages"]) == 0 && !e.NPU.CPU(m) {
			// AMD's NPU: an export of tools/amd/iron_export_lm.py d1 (its recipe, model-recipes.yaml),
			// from the sources or made with IRON, as a chat model's
			if len(m.AMDExport) == 0 {
				return f, noAMDForm(e)
			}
			dir, err := chat.AMDExportOf(e.Ctx, m, e.Log)
			if err != nil {
				return f, err
			}
			f.Set("export", dir)
			f.SetValue("name", e.Base.Name)
			return f, nil
		}
		if pk := ps["packages"]; len(pk) > 0 {
			config := first(ps["config"])
			if config == "" {
				return f, errors.New("text: a decision model's packages need its config (the checkpoint's config.json)")
			}
			f.Set("packages", pk...)
			f.Set("config", config)
			f.Set("weights", cmp.Or(first(ps["weights"]), filepath.Join(filepath.Dir(config), "model.safetensors")))
		} else {
			seqs, err := ints(m.Settings["seqs"], systemone.DefaultSeqs)
			if err != nil {
				return f, err
			}
			backend := ""
			if !e.NPU.CPU(m) { // the CPU interpreter runs Core ML packages
				if backend, err = llm.NativeFormat(e.Ctx, e.NPU.Device); err != nil {
					return f, err
				}
			}
			c, err := systemone.Prepare(e.Ctx, systemone.Setup{Dir: e.Dir, Repo: m.Source(), Seqs: seqs, Format: m.Settings["format"], Backend: backend, Log: e.Log})
			if err != nil {
				return f, err
			}
			f.Set("packages", c.Packages...)
			f.Set("config", c.ModelConfig)
			f.Set("weights", c.Weights)
			f.Set("tokenizer", c.Tokenizer)
			f.Set("vision", c.Vision...)
			f.Set("audio", c.Audio...)
			f.SetValue("name", e.Base.Name)
		}
		if t := ps["tokenizer"]; len(t) > 0 { // one converted by hand
			f.Set("tokenizer", t...)
		}
		f.Set("calibration", ps["calibration"]...)
		if v := ps["vision"]; len(v) > 0 {
			f.Set("vision", v...)
		}
		if a := ps["audio"]; len(a) > 0 {
			f.Set("audio", a...)
		}
		return f, nil
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		if dir := f.Path("export"); dir != "" {
			m, err := systemone.LoadAMD(ctx, systemone.AMDConfig{Dir: dir, Name: f.Values["name"], Log: e.Log})
			if err != nil {
				return nil, err
			}
			return &Decider{m: m, model: e.Model}, nil
		}
		m, err := systemone.Load(ctx, systemone.Config{Packages: f.Paths["packages"], ModelConfig: f.Path("config"), Weights: f.Path("weights"),
			Tokenizer: f.Path("tokenizer"), Calibration: f.Path("calibration"), Vision: f.Paths["vision"], Audio: f.Paths["audio"],
			Name: f.Values["name"], CacheDir: e.Cache, CPU: e.NPU.CPU(e.Def), Backend: e.NPU.Device, Log: e.Log})
		if err != nil {
			return nil, err
		}
		return &Decider{m: m, model: e.Model, cpu: e.NPU.CPU(e.Def)}, nil
	},
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
	m      *systemone.Model
	model  string
	cpu    bool
}

// Model is the name of the model in the definitions, Name the one it calls itself.
func (d *Decider) Model() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Model")
	}
	return d.model
}
func (d *Decider) Name() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Name")
	}
	return d.m.Name
}

// MaxTokens is the most tokens of a question's prompt.
func (d *Decider) MaxTokens() int {
	if d.remote != nil {
		return airpc.Value[int](d.remote, "MaxTokens")
	}
	return d.m.MaxTokens()
}

// OnCPU reports whether it runs on the CPU interpreter.
func (d *Decider) OnCPU() bool {
	if d.remote != nil {
		return airpc.Value[bool](d.remote, "OnCPU")
	}
	return d.cpu
}

// Decide answers a request of the System One API: {"state": ..., "questions": {"id": {"type":
// "choice" | "score" | "noul", "instructions": ..., "criteria": ...}}}, as llama.cpp's
// tools/server/README.md describes it. A request that is not one is ErrBadRequest.
func (d *Decider) Decide(ctx context.Context, request []byte) (Decision, error) {
	if d.remote != nil {
		return airpc.Call1[Decision](d.remote, ctx, "Decide", request)
	}
	t0 := time.Now()
	req, err := systemone.ParseRequest(request)
	if err == nil {
		var res *systemone.Response
		if res, err = d.m.Decide(ctx, req); err == nil {
			b, err := json.Marshal(res)
			return Decision{JSON: b, Questions: len(req.Questions), InputTokens: res.InputTokens, Took: time.Since(t0)}, err
		}
	}
	var bad *systemone.BadRequest
	var unsup *systemone.Unsupported
	switch {
	case errors.As(err, &bad):
		return Decision{}, kindError{err, ErrBadRequest}
	case errors.As(err, &unsup):
		return Decision{}, kindError{err, ErrUnsupported}
	}
	return Decision{}, err
}

// kindError is err, which also is kind (errors.Is), with err's message.
type kindError struct{ err, kind error }

func (e kindError) Error() string   { return e.err.Error() }
func (e kindError) Unwrap() []error { return []error{e.kind, e.err} }

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (d *Decider) Close() error {
	if d.remote != nil {
		return d.remote.Release() // the Runtime in hwnpud unloads it
	}
	return d.m.Close()
}

func init() {
	airole.Register(decideRole)
	airpc.RegisterError("text.ErrBadRequest", ErrBadRequest)
	airpc.RegisterError("text.ErrUnsupported", ErrUnsupported)
}
