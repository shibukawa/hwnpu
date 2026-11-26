package modelscmd

import (
	"context"
	"errors"
	"os"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/prebuild"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// canCompile is nil: coreai-build is asked for when it runs (coreai.Build).
func canCompile() error { return nil }

// compile compiles the Core AI exports and programs at paths for archs as the recipe of model
// says (model-recipes.yaml; when archs is empty the recipe's architectures for a model it prebuilds, none for another, the recipe of each
// export's model when model is "") and puts them in the sources written back to (compileFlags).
func compile(ctx context.Context, paths, archs []string, model string, p compileFlags) error {
	// with no source written back to, the programs stay packed beside their export (Run says so
	// at its start): a put would end the build in an error once everything compiled
	put := !p.NoPush && !p.Packed && len(shared.Writebacks()) > 0
	return prebuild.Run(ctx, paths, prebuild.Options{Model: model, Archs: archs, Jobs: p.Jobs, Force: p.Force, DryRun: p.DryRun,
		Packed: p.Packed || !p.NoPush, Put: put, AnyTools: p.AnyTools, Log: os.Stderr})
}

// canExportAMD is why this OS cannot make exports for AMD's NPU: IRON (mlir-aie, Peano) runs on
// Linux and Windows.
func canExportAMD() error {
	return errors.New("the exports for AMD's NPU are made with IRON, on Linux or Windows")
}
