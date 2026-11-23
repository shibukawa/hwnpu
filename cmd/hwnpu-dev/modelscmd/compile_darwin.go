package modelscmd

import (
	"context"
	"os"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/prebuild"
)

// canCompile is nil: coreai-build is asked for when it runs (coreai.Build).
func canCompile() error { return nil }

// compile compiles the Core AI exports and programs at paths for archs as the recipe of model
// says (model-recipes.yaml; the recipe's architectures when archs is empty, the recipe of each
// export's model when model is "") and puts them in the sources written back to (compileFlags).
func compile(ctx context.Context, paths, archs []string, model string, p compileFlags) error {
	return prebuild.Run(ctx, paths, prebuild.Options{Model: model, Archs: archs, Jobs: p.Jobs, Force: p.Force, DryRun: p.DryRun,
		Packed: p.Packed || !p.NoPush, Put: !p.NoPush && !p.Packed, AnyTools: p.AnyTools, Log: os.Stderr})
}
