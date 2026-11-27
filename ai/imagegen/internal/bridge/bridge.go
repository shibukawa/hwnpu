// Package bridge joins ai/imagegen, the client of the picture models, to ai/imagegen/engine, which
// runs them in this process: the roles whose Prepare, Load and Usable the engine sets, how a model
// of the engine becomes the client's type, and what of the exports the client asks of the engine
// (docs/design/hwnpud.md, "Clients without the engine").
package bridge

import (
	"io"

	"github.com/shibukawa/hwnpu/internal/airole"
)

// Roles are ai/imagegen's roles by name (image, image-edit), which the engine gives their
// Prepare, Load and Usable.
var Roles = map[string]*airole.Role{}

// Wrap makes a model of the engine (of the role's interface in ai/imagegen) the client's type of
// the role, by the role's name.
var Wrap = map[string]func(local any) io.Closer{}

// The families of the models the generator draws with (catalog.Model's Family's prefix).
const (
	FamKlein = "flux2-klein"
	FamFlux3 = "flux3-klein"
	FamAnima = "anima"
)

// ExportFamily is the family of the export in dir, "" for none (imagegen.Export): the engine's,
// nil without it.
var ExportFamily func(dir string) string
