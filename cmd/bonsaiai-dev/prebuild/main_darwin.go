// Command bonsaiai-dev prebuild compiles Core AI programs ahead of time with coreai-build (Xcode 27's
// Metal Toolchain), writing X.ARCH.aimodelc beside each X.aimodel of an export. On a device of
// that architecture bonsaiai loads the prebuilt program while Core AI's cache has no entry for the
// program, and the device finishes it (ANELargeModelCompilerService) in a third of the time the
// Neural Engine compiler takes over the source (docs/design/apple-coreai.md).
//
// It builds as the model's recipe says (model-recipes.yaml, docs/model-recipes.md): the
// architectures, the compiles at once, the programs left to the device, and the version of
// coreai-build. The flags override the first two.
//
//	bonsaiai-dev prebuild -put EXPORT_DIR...        the recipe's architectures, put in the sources
//	bonsaiai-dev prebuild -arch h14g,h15g EXPORT    some architectures (-arch all: every one)
//	bonsaiai-dev prebuild -j 4 -force EXPORT        4 compiles at once, over what is there
//	bonsaiai-dev prebuild -n EXPORT                 list what would be compiled
//	bonsaiai-dev prebuild -mark failedToSpecialize -arch h15g -programs 'dit_single_*' EXPORT
//	                                             write what a device made of the prebuilt programs
//
// A packed build (-put, -packed) takes a program's prebuilt program from another export of the
// same source (an earlier version of the model's, beside the export) instead of compiling it
// again, when the record of its build (EXPORT.prebuilt/ARCH/build.json) says it was good: the
// same coreai-build, no rule leaving it to the device, no more compiles at once, on the Neural
// Engine, and no device failed it (-mark, or a run that failed to finish it: failedToSpecialize). One a
// device failed is not compiled again (it would be the same): the run says the rule that leaves
// it to the device. Prebuilt programs made before records are given theirs once with -record
// (here and in the sources written back to); without one a program is kept where it is but taken for
// no other export (docs/prebuilt-reuse.md).
//
// With -put (or -packed) the programs are packed beside the export (EXPORT.prebuilt/ARCH, each
// without its weights, which a device makes again from the export's bytecode) and -put puts
// each architecture's in the sources written back to (BONSAIAI_MODEL_SOURCE...?writeback=true): a
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

	"github.com/shibukawa/bonsaiai/internal/backend/coreai"
	"github.com/shibukawa/bonsaiai/internal/chat"
)

func Main() {
	archs := flag.String("arch", "", "the architectures, comma separated (default: the recipe's for a model it prebuilds, model-recipes.yaml prebuilt, none for another; all: "+strings.Join(coreai.Archs, " ")+")")
	jobs := flag.Int("j", 0, fmt.Sprintf("compiles at once (default: the recipe's, else %d here: one for every 4 GB of memory beyond 6, at most half the cores)", coreai.DefaultJobs()))
	force := flag.Bool("force", false, "compile again over prebuilt programs that are there")
	dry := flag.Bool("n", false, "list what would be compiled")
	packed := flag.Bool("packed", false, "pack the programs beside the export (EXPORT.prebuilt/ARCH) rather than whole beside their sources")
	put := flag.Bool("put", false, "pack the programs and put each architecture's in the sources written back to")
	model := flag.String("model", "", "the model whose recipe to build with (default: the one each export is of, its entry.json)")
	anyTools := flag.Bool("any-tools", false, "build with a coreai-build of another version than model-recipes.yaml's")
	record := flag.Bool("record", false, "write the records (build.json) of prebuilt programs made before records, compiling nothing: of the exports given, else of every export in the cache and every entry of prebuilt programs in the sources written back to (fetched, recorded, put again)")
	mark := flag.String("mark", "", "write what a device made of the exports' prebuilt programs for -arch (ok, or the failure: failedToSpecialize) in their build.json, compiling nothing")
	programs := flag.String("programs", "", "with -mark, the programs, comma separated patterns (dit_single_*; default: all)")
	function := flag.String("function", "", "with -mark, the function that failed (s1024x1024)")
	flag.Parse()
	if flag.NArg() == 0 && !*record {
		fmt.Fprintln(os.Stderr, "usage: bonsaiai-dev prebuild [-arch LIST] [-j N] [-force] [-n] [-put] [-model NAME] EXPORT_DIR_OR_PROGRAM...\n       bonsaiai-dev prebuild -mark STATUS -arch LIST [-programs LIST] [-function F] EXPORT_DIR...\n       bonsaiai-dev prebuild -record [-n] [EXPORT_DIR...]")
		os.Exit(2)
	}
	// the definitions (the models of the exports) and the sources of the environment
	if err := chat.LoadUserModels("", os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "bonsaiai-dev prebuild:", err)
		os.Exit(1)
	}
	if *mark != "" {
		if *archs == "" || *archs == "all" {
			fmt.Fprintln(os.Stderr, "bonsaiai-dev prebuild -mark: name the architectures (-arch h15g)")
			os.Exit(2)
		}
		var progs []string
		if *programs != "" {
			progs = strings.Split(*programs, ",")
		}
		if err := Mark(flag.Args(), strings.Split(*archs, ","), progs, *mark, *function, os.Stderr); err != nil {
			fmt.Fprintln(os.Stderr, "bonsaiai-dev prebuild -mark:", err)
			os.Exit(1)
		}
		return
	}
	if *record {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		if err := Record(ctx, flag.Args(), *dry, os.Stderr); err != nil {
			fmt.Fprintln(os.Stderr, "bonsaiai-dev prebuild -record:", err)
			os.Exit(1)
		}
		return
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
		fmt.Fprintln(os.Stderr, "bonsaiai-dev prebuild:", err)
		os.Exit(1)
	}
}
