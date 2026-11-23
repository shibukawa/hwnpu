// Package airole is what the ai package and its packages of models (ai/speech, ai/text,
// ai/vision) agree on: how a kind of model is made ready and loaded. It is internal so that the
// packages of models are hwnpu's own; an app gets its uses from them.
package airole

import (
	"context"
	"fmt"
	"io"
	"os"
	"runtime"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// Role is a kind of use: what its models are, how one is made ready and how it is loaded.
type Role struct {
	// Name is the role's name (transcribe, translate, speak, depth, faces).
	Name string
	// Kinds are the kinds of the definitions' models that fill the role; Feature, when not "",
	// is a feature the model must have besides (translation).
	Kinds   []chat.Kind
	Feature string
	// Prepare makes the model ready as the command of its kind does on its first run: it takes
	// what is missing from the shared layers or downloads it, and converts the programs unless
	// a layer has them. It returns the files Load reads.
	Prepare func(PrepareEnv) (Files, error)
	// Load loads the model from the files alone: no network, no conversion.
	Load func(ctx context.Context, f Files, env LoadEnv) (io.Closer, error)
	// Revision counts the changes to what Prepare makes that call for preparing again: a store
	// prepared by an earlier revision counts as not prepared.
	Revision int
	// Usable, when not nil, says whether what Prepare made still runs here (nil when it does):
	// a store whose files depend on what this machine had when it was prepared (ONNX Runtime's
	// library for a model it put on the CPU) counts as not prepared when that is gone, so that
	// Prepare decides again instead of the load failing.
	Usable func(Files) error
	// Remote is the role's model of an app whose models run in hwnpud (HWNPU_DAEMON): a value
	// of the type Load returns whose methods call h's, the model Load loaded in hwnpud's engine
	// (airpc.Call1(x.remote, ctx, "Estimate", img)). nil for a role that does not run through
	// hwnpud yet: an app of it runs with HWNPU_DAEMON=off.
	Remote func(h *airpc.Handle) io.Closer
}

// Accepts is nil when the role runs model m.
func (r *Role) Accepts(m chat.Model) error {
	if !slices.Contains(r.Kinds, m.Kind) {
		return fmt.Errorf("ai: %s is a %s model, which the role %s does not run", m.Name, m.Kind, r.Name)
	}
	if r.Feature != "" && !slices.Contains(m.Features, r.Feature) {
		return fmt.Errorf("ai: %s has no %s, which the role %s needs", m.Name, r.Feature, r.Name)
	}
	return nil
}

// PrepareEnv is what a role's Prepare is given.
type PrepareEnv struct {
	Ctx   context.Context
	Model chat.Model
	// Base is the model of the definitions Model was made from by the app (ai.Definition's Like):
	// what it shares where its settings do not change it (the export of AMD's NPU, found by the
	// model's name). It is Model itself for a model of the definitions.
	Base chat.Model
	// Dir is the model's directory of programs (chat.ModelDir).
	Dir string
	// NPU is what the programs are converted for.
	NPU NPU
	Log io.Writer
}

// LoadEnv is what a role's Load is given.
type LoadEnv struct {
	// Model is the model's name in the definitions, and Def its definition (its Device, Settings
	// and Programs among the rest).
	Model string
	Def   chat.Model
	NPU   NPU
	// AllowOffNPU loads a model part of whose programs the NPU's compiler placed off it (on the
	// GPU or the CPU), which a role that checks the placement refuses otherwise.
	AllowOffNPU bool
	// Cache is the compile cache.
	Cache string
	Log   io.Writer
	// Settings are how the app runs the role's models beyond their definition, settings that
	// change nothing of what is prepared (ai.LoadSettings); nil for none.
	Settings map[string]string
}

// Files are what a role's Prepare made and its Load reads: paths by name (files, or the
// directories of programs and exports) and settings that are not paths.
type Files struct {
	Paths  map[string][]string `json:"paths,omitempty"`
	Values map[string]string   `json:"values,omitempty"`
}

// Set names paths; no paths, or only empty ones, name nothing.
func (f *Files) Set(name string, paths ...string) {
	paths = slices.DeleteFunc(slices.Clone(paths), func(p string) bool { return p == "" })
	if len(paths) == 0 {
		return
	}
	if f.Paths == nil {
		f.Paths = map[string][]string{}
	}
	f.Paths[name] = paths
}

// SetValue names a setting; an empty one names nothing.
func (f *Files) SetValue(name, value string) {
	if value == "" {
		return
	}
	if f.Values == nil {
		f.Values = map[string]string{}
	}
	f.Values[name] = value
}

// Path is the first path of name, "" for none.
func (f Files) Path(name string) string {
	if p := f.Paths[name]; len(p) > 0 {
		return p[0]
	}
	return ""
}

// NPU is this machine's NPU as the models are prepared for it and run on it.
type NPU struct {
	// Vendor is amd, apple or intel; "" without an NPU.
	Vendor string
	// Target names what the programs are for, as the shared layers name it: coreai, intel or
	// amd-xdna2 (cache.AMDTarget); "" when no NPU runs here (the programs are converted, not
	// loaded).
	Target string
	// Device is the device backend to open (llm.FindDevice, "" for the NPU found).
	Device string
	// Format is the program format the converted models are made in (llm.Options.Backend); ""
	// on AMD's NPU, which runs exports of IRON.
	Format string
}

// CPU reports whether model m runs on the CPU on this NPU's machine (its definition's device).
func (n NPU) CPU(m chat.Model) bool { return m.DeviceFor(n.Vendor) == "cpu" }

// AMD reports whether the NPU is AMD's, which runs IRON's exports instead of converted programs.
func (n NPU) AMD() bool { return n.Target == cache.AMDTarget }

// Here is this machine's NPU, picked as hwnpu-chat, hwnpu-audio and hwnpu-image pick it:
// HWNPU_BACKEND, else the NPU found; on a Mac Core AI's (macOS 27), which runs the Neural Engine
// for them (Core ML is not used).
func Here(ctx context.Context) NPU {
	device := strings.TrimSpace(os.Getenv(llm.BackendEnv))
	if runtime.GOOS == "darwin" {
		switch device {
		case "", llm.BackendCoreML, llm.BackendCoreAI:
			device = llm.BackendCoreAI
		}
	}
	n := NPU{Vendor: chat.HereVendor(), Device: device}
	if chat.OnlyAMD(ctx, device) {
		n.Target = cache.AMDTarget
		return n
	}
	if device == llm.BackendCoreAI {
		n.Format = llm.BackendCoreAI
		if _, _, err := llm.FindDevice(ctx, device); err == nil {
			n.Target = cache.CoreAITarget
		}
		return n
	}
	f, err := llm.ProgramFormat(ctx, device)
	if err != nil {
		return n
	}
	n.Target, n.Format = f, f
	return n
}

// registered are the roles of the packages of models, in the order they were registered.
var registered []*Role

// Register adds r to the roles For finds: each package of models registers its roles, so that a
// tool (hwnpu-dev models package) makes a model ready as the app would, by its kind alone.
func Register(r *Role) { registered = append(registered, r) }

// Lookup is the role registered as name (Role.Name), nil when none is: how hwnpud's engine finds
// the roles of an app's uses (internal/aihost), whose packages it imports.
func Lookup(name string) *Role {
	for _, r := range registered {
		if r.Name == name {
			return r
		}
	}
	return nil
}

// For is the first role registered that runs m; nil when none does (or its package is not
// imported).
func For(m chat.Model) *Role {
	for _, r := range registered {
		if r.Accepts(m) == nil {
			return r
		}
	}
	return nil
}

// ForTarget is the NPU of a target as the shared layers name it (amd-xdna2, coreai), to prepare
// a model for it on any machine: the programs are converted and put in the layers, not loaded,
// so no device of the target needs to be here.
func ForTarget(target string) (NPU, error) {
	switch target {
	case cache.AMDTarget:
		return NPU{Vendor: "amd", Target: cache.AMDTarget}, nil
	case cache.CoreAITarget:
		return NPU{Vendor: "apple", Target: cache.CoreAITarget, Device: llm.BackendCoreAI, Format: llm.BackendCoreAI}, nil
	}
	return NPU{}, fmt.Errorf("ai: no target %q to prepare models for (%s, %s)", target, cache.AMDTarget, cache.CoreAITarget)
}
