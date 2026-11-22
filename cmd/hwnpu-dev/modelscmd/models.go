// Package modelscmd is hwnpu-dev models: the conversions of the models of the definitions, here,
// in the sources (docs/cache-sharing-design.md 7) and on Hugging Face (docs/prebuilt.md). Each
// is named MODEL/TARGET by the target it is for, as the sources' folders are
// (converted/MODEL/TARGET): fetch brings them here, package makes them, push puts what was made
// by hand. The commands are the same on every OS; what one of them needs that this OS has not
// (Core AI's compiler for coreai-ARCH) it says, and stops.
package modelscmd

import (
	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/hfcmd"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/sourcescmd"

	// the roles package prepares the models with (airole.For)
	_ "github.com/shibukawa/hwnpu/ai/imagegen"
	_ "github.com/shibukawa/hwnpu/ai/speech"
	_ "github.com/shibukawa/hwnpu/ai/text"
	_ "github.com/shibukawa/hwnpu/ai/vision"
)

// Cmd is hwnpu-dev models.
type Cmd struct {
	devopt.Defs `embed:""`

	Fetch   FetchCmd           `cmd:"" help:"Bring models here without converting anything, to try conversions on them: the checkpoint and the conversions for each target the sources have (what none has is listed); what comes from Hugging Face goes in the sources written back to too."`
	Package PackageCmd         `cmd:"" help:"Make the conversions of models for targets (amd-xdna2, coreai, coreai-ARCH) on this machine, whatever its NPU, unless a source has them, and put them in the sources written back to."`
	Push    sourcescmd.PushCmd `cmd:"" help:"Push exports for AMD's NPU that are here, made by hand, in the sources written back to (with entry.json, made for them when missing); what package makes goes there itself."`
	HF      hfcmd.Cmd          `cmd:"" name:"hf" help:"Put the conversions the sources hold on Hugging Face, one repository a model, and pin the commits in models.yaml (docs/prebuilt.md): plan, push, check, sizes."`
}

// AfterApply gives the subcommands the definitions' flag.
func (c *Cmd) AfterApply(k *kong.Context) error { c.Bind(k); return nil }
