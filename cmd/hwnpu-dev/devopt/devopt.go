// Package devopt holds what the groups of hwnpu-dev's subcommands share: the definition files
// they read (--models).
package devopt

import (
	"os"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// Defs is the --models flag of a group of subcommands: the user's definition files, read
// over hwnpu's. A group embeds it and binds it for its subcommands (Bind).
type Defs struct {
	Models string `help:"${models_usage}" placeholder:"FILE,..."`
}

// Load reads the definitions and sets the sources of the environment (HWNPU_MODEL_SOURCE...).
func (d *Defs) Load() error { return chat.LoadUserModels(d.Models, os.Stderr) }

// Bind gives the subcommands of a group d, their Run(d *devopt.Defs).
func (d *Defs) Bind(k *kong.Context) { k.Bind(d) }

// Vars are kong's variables of the flag.
var Vars = map[string]string{"models_usage": chat.ModelsUsage}
