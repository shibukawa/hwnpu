// Package bridge joins ai/sound, the client of the sound models, to ai/sound/engine, which runs
// them in this process: the roles whose Prepare and Load the engine sets, and how a model of the
// engine becomes the client's type (docs/design/hwnpud.md, "Clients without the engine").
package bridge

import (
	"io"

	"github.com/shibukawa/hwnpu/internal/airole"
)

// Roles are ai/sound's roles by name (music, sfx, pitch, notes), which the engine gives their
// Prepare and Load.
var Roles = map[string]*airole.Role{}

// Wrap makes a model of the engine (of the role's interface in ai/sound) the client's type of
// the role, by the role's name.
var Wrap = map[string]func(local any) io.Closer{}
