// Package aiengine is the engine the ai package and modelstore call (internal/aieng.Engine), of
// internal/chat: the definitions, the policy, the licenses, the memory and fetching. ai/engine
// imports it with the engines of the roles; the tests of ai and modelstore import it alone.
package aiengine

import (
	"context"

	"io"

	"github.com/shibukawa/configdir"
	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/aieng"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/daemon"
)

func init() {
	aieng.E = local{}
	// hwnpud and its clients compare the definitions they were built with (daemon.Hello)
	daemon.Builtin = [][]byte{hwnpu.ModelsYAML, hwnpu.RecipesYAML}
}

// local is the engine of internal/chat.
type local struct{}

func (local) LoadModels(files []string, texts []catalog.Definitions, user bool, log io.Writer) error {
	return chat.LoadModelsWith(files, texts, user, log)
}

func (local) Table() ([]catalog.Model, []catalog.Category, map[string]string, []string) {
	return chat.Catalog, chat.Categories, chat.Aliases, chat.ModelFiles
}

func (local) Lookup(name string) (catalog.Model, bool) { return chat.Lookup(name) }

func (local) LookupOn(name string, machine int, log io.Writer) (catalog.Model, bool, error) {
	return chat.LookupOn(name, machine, log)
}

func (local) Pick(name string, machine int) (catalog.Model, string, bool, error) {
	return chat.Pick(name, machine)
}

func (local) Named(name string) (catalog.Model, error)        { return chat.Named(name) }
func (local) Downloaded(m catalog.Model) bool                 { return chat.Downloaded(m) }
func (local) ModelDir(m catalog.Model) string                 { return chat.ModelDir(m) }
func (local) AllowedAlso(m catalog.Model, items string) error { return chat.AllowedAlso(m, items) }

func (local) AcceptLicense(m catalog.Model, yes, ask bool, in io.Reader, out io.Writer) error {
	return chat.AcceptLicense(m, yes, ask, in, out)
}

func (local) WithModel(ctx context.Context, m catalog.Model) context.Context {
	return chat.WithModel(ctx, m)
}

func (local) MachineMemory() int                         { return chat.MachineMemory() }
func (local) HereVendor() string                         { return chat.HereVendor() }
func (local) ChosenMemory(m catalog.Model) int           { return chat.ChosenMemory(m) }
func (local) ChooseMemory(m catalog.Model, gb int) error { return chat.ChooseMemory(m, gb) }
func (local) ChosenMemories() map[string]int             { return chat.ChosenMemories() }

func (local) BundleDirs(app configdir.ConfigDir, folder string) []string {
	return chat.BundleDirs(app, folder)
}

func (local) Fetch(ctx context.Context, repo, name string, log io.Writer) (string, error) {
	return chat.Fetch(ctx, repo, name, log)
}

func (local) FetchWeights(ctx context.Context, repo string, log io.Writer) (string, error) {
	return chat.FetchWeights(ctx, repo, log)
}

func (local) RepoDir(repo string) string { return chat.RepoDir(repo) }
func (local) NotFound(err error) bool    { return chat.NotFound(err) }
