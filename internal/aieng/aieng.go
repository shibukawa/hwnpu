// Package aieng is what the packages of hwnpu's clients (ai, modelstore) call of the engine that
// runs the models in this process, without importing it: the engine (ai/engine, which hwnpud,
// hwnpu-dev and the tests import) sets E. A process of a client alone has none, and its models
// run through hwnpud (docs/design/hwnpud.md, "Clients without the engine").
package aieng

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/shibukawa/configdir"
	"github.com/shibukawa/hwnpu/internal/catalog"
)

// Engine is the engine of this process: the definitions it read, how it fetches and keeps the
// models, and the memory they run as.
type Engine interface {
	// LoadModels reads the definition files and texts over the built-in ones (user: the user's
	// and the machine's too).
	LoadModels(files []string, texts []catalog.Definitions, user bool, log io.Writer) error
	// Table is what the definitions read hold: the models, the uses, the aliases and the files.
	Table() (models []catalog.Model, uses []catalog.Category, aliases map[string]string, files []string)
	// Lookup and LookupOn find a model by its name, family, use or repository (LookupOn for a
	// machine of machine GB, 0 for this one); Pick is the model of a use for machine GB; Named a
	// model of the definitions by name alone.
	Lookup(name string) (catalog.Model, bool)
	LookupOn(name string, machine int, log io.Writer) (catalog.Model, bool, error)
	Pick(name string, machine int) (m catalog.Model, why string, ok bool, err error)
	Named(name string) (catalog.Model, error)
	// Downloaded is whether m's checkpoint or converted programs are here; ModelDir its directory.
	Downloaded(m catalog.Model) bool
	ModelDir(m catalog.Model) string
	// AllowedAlso is nil when the policy, narrowed by items, allows m; AcceptLicense asks for or
	// records the acceptance of m's license.
	AllowedAlso(m catalog.Model, items string) error
	AcceptLicense(m catalog.Model, yes, ask bool, in io.Reader, out io.Writer) error
	// WithModel is ctx for preparing m (what the engine reads of it while it converts).
	WithModel(ctx context.Context, m catalog.Model) context.Context
	// MachineMemory is the memory the models are picked by; HereVendor this machine's NPU vendor.
	MachineMemory() int
	HereVendor() string
	// ChosenMemory, ChooseMemory and ChosenMemories are the memory a model was chosen to run as.
	ChosenMemory(m catalog.Model) int
	ChooseMemory(m catalog.Model, gb int) error
	ChosenMemories() map[string]int
	// BundleDirs are the folders of models bundled with an app; Fetch and FetchWeights bring a
	// file and the weights of a repository here, RepoDir is where; NotFound is whether err is
	// of a file the repository does not have.
	BundleDirs(app configdir.ConfigDir, folder string) []string
	Fetch(ctx context.Context, repo, name string, log io.Writer) (string, error)
	FetchWeights(ctx context.Context, repo string, log io.Writer) (string, error)
	RepoDir(repo string) string
	NotFound(err error) bool
}

// E is the engine of this process, nil without one.
var E Engine

// ErrNone is the error of a process without the engine asked to run the models itself.
var ErrNone = errors.New("this process has no engine to run the models in: go through hwnpud (HWNPU_DAEMON), or import _ \"github.com/shibukawa/hwnpu/ai/engine\" to run them here")

// Get is E, or an error that says how to have one.
func Get() (Engine, error) {
	if E == nil {
		return nil, ErrNone
	}
	return E, nil
}

// Must is E; without one it panics with ErrNone (for the paths only the engine's process reaches).
func Must() Engine {
	if E == nil {
		panic(fmt.Errorf("aieng: %w", ErrNone))
	}
	return E
}
