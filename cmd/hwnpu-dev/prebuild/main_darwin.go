// Command hwnpu-dev prebuild compiles Core AI programs ahead of time with coreai-build (Xcode 27's
// Metal Toolchain), writing X.ARCH.aimodelc beside each X.aimodel of an export. On a device of
// that architecture hwnpu loads the prebuilt program while Core AI's cache has no entry for the
// program, and the device finishes it (ANELargeModelCompilerService) in a third of the time the
// Neural Engine compiler takes over the source (docs/design/apple-coreai.md).
//
// It builds as the model's recipe says (model-recipes.yaml, docs/model-recipes.md): the
// architectures, the compiles at once, the programs left to the device, and the version of
// coreai-build. The flags override the first two.
//
//	hwnpu-dev prebuild -put EXPORT_DIR...        the recipe's architectures, put in the sources
//	hwnpu-dev prebuild -arch h14g,h15g EXPORT    some architectures (-arch all: every one)
//	hwnpu-dev prebuild -j 4 -force EXPORT        4 compiles at once, over what is there
//	hwnpu-dev prebuild -n EXPORT                 list what would be compiled
//
// With -put (or -packed) the programs are packed beside the export (EXPORT.prebuilt/ARCH, each
// without its weights, which a device makes again from the export's bytecode) and -put puts
// each architecture's in the sources written back to (HWNPU_MODEL_SOURCE...?writeback=true): a
// machine that fetches the export fetches its architecture's too, and loads the programs
// without the Neural Engine compiler (docs/cache-sharing-design.md 7.9). An architecture of an
// export with a program that failed is not put: the run says the rule of the recipe that would
// leave the program to the device.
package prebuild

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"

	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/chat"
)

func Main() {
	archs := flag.String("arch", "", "the architectures, comma separated (default: the recipe's, model-recipes.yaml; all: "+strings.Join(coreai.Archs, " ")+")")
	jobs := flag.Int("j", 0, fmt.Sprintf("compiles at once (default: the recipe's, else %d here: one for every 4 GB of memory beyond 6, at most half the cores)", coreai.DefaultJobs()))
	force := flag.Bool("force", false, "compile again over prebuilt programs that are there")
	dry := flag.Bool("n", false, "list what would be compiled")
	packed := flag.Bool("packed", false, "pack the programs beside the export (EXPORT.prebuilt/ARCH) rather than whole beside their sources")
	put := flag.Bool("put", false, "pack the programs and put each architecture's in the sources written back to")
	model := flag.String("model", "", "the model whose recipe to build with (default: the one each export is of, its entry.json)")
	anyTools := flag.Bool("any-tools", false, "build with a coreai-build of another version than model-recipes.yaml's")
	flag.Parse()
	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: hwnpu-dev prebuild [-arch LIST] [-j N] [-force] [-n] [-put] [-model NAME] EXPORT_DIR_OR_PROGRAM...")
		os.Exit(2)
	}
	// the definitions (the models of the exports) and the sources of the environment
	if err := chat.LoadUserModels("", os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-dev prebuild:", err)
		os.Exit(1)
	}
	o := Options{Model: *model, Jobs: *jobs, Force: *force, DryRun: *dry, Packed: *packed, Put: *put, AnyTools: *anyTools, Log: os.Stderr}
	switch *archs {
	case "":
	case "all":
		o.Archs = coreai.Archs
	default:
		o.Archs = strings.Split(*archs, ",")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := Run(ctx, flag.Args(), o); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-dev prebuild:", err)
		os.Exit(1)
	}
}
