// Package modelscmd is hwnpu-dev models: the conversions of the models of the definitions, here
// and in the sources (docs/cache-sharing-design.md 7) and on Hugging Face (docs/prebuilt.md).
// Its commands are the same on every OS; what one of them needs that this OS has not (Core AI's
// compiler for prebuild) it says, and stops.
package modelscmd

import (
	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/cachecmd"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/hfcmd"

	// the roles export prepares the models with (airole.For)
	_ "github.com/shibukawa/hwnpu/ai/imagegen"
	_ "github.com/shibukawa/hwnpu/ai/speech"
	_ "github.com/shibukawa/hwnpu/ai/text"
	_ "github.com/shibukawa/hwnpu/ai/vision"
)

// Cmd is hwnpu-dev models.
type Cmd struct {
	devopt.Defs `embed:""`

	Sources  cachecmd.SourcesCmd `cmd:"" help:"Show the cache's root, the sources in effect (HWNPU_MODEL_SOURCE, _1, ...), whether each is written back to, whether Hugging Face is read, and the .env files read."`
	Ls       cachecmd.LsCmd      `cmd:"" help:"List the conversions the sources hold (or one source: its number in hwnpu-dev models sources); with --original the repositories' locks."`
	Fetch    FetchCmd            `cmd:"" help:"Bring models here without converting anything, to try conversions on them: the checkpoint and the conversions for each target the sources have (what none has is listed); what comes from Hugging Face goes in the sources written back to too."`
	Export   ExportCmd           `cmd:"" help:"Make the conversions of models for a target (amd-xdna2, coreai) on this machine, whatever its NPU, unless a source has them, and put them in the sources written back to."`
	Prebuild PrebuildCmd         `cmd:"" help:"Compile the Core AI exports of models (or export folders) for the Neural Engine's architectures with coreai-build, and put them in the sources written back to (macOS, Xcode 27's Metal Toolchain)."`
	Put      cachecmd.PushCmd    `cmd:"" help:"Put exports for AMD's NPU that are here in the sources written back to (with entry.json, made for them when missing)."`
	Import   cachecmd.ImportCmd  `cmd:"" help:"Put folders made by hand (~/models) in the sources written back to: a checkpoint's files that are its Hugging Face repository's (original/), an export for AMD's NPU (converted/)."`
	Update   cachecmd.UpdateCmd  `cmd:"" help:"Move the repositories' files in the sources written back to to the commit Hugging Face serves now (their hwnpu-lock.json): the files that changed are downloaded again."`
	Verify   cachecmd.VerifyCmd  `cmd:"" help:"Check directories against their entry.json: the sizes, and the SHA-256 with --sums."`
	Flush    cachecmd.FlushCmd   `cmd:"" help:"Do the work the background left in the queue (what goes in the sources) and wait for it."`
	GC       cachecmd.GCCmd      `cmd:"" name:"gc" help:"Remove conversions from a source (by model, kind, key or age); with --orphans what a writeback cut off left too. Shows what would go unless --apply."`
	Migrate  cachecmd.MigrateCmd `cmd:"" help:"Move what a folder source holds in the layout before (blobs/, upstream/, entries/) to original/ and converted/, by hard links."`
	HF       hfcmd.Cmd           `cmd:"" name:"hf" help:"Put the conversions the sources hold on Hugging Face, one repository a model, and pin the commits in models.yaml (docs/prebuilt.md): plan, push, check, sizes."`
}

// AfterApply gives the subcommands the definitions' flag.
func (c *Cmd) AfterApply(k *kong.Context) error { c.Bind(k); return nil }
