// Command hwnpu-dev prebuild compiles Core AI programs ahead of time with coreai-build (Xcode 27's
// Metal Toolchain), writing X.ARCH.aimodelc beside each X.aimodel of an export. On a device of
// that architecture hwnpu loads the prebuilt program while Core AI's cache has no entry for the
// program, and the device finishes it (ANELargeModelCompilerService) in a third of the time the
// Neural Engine compiler takes over the source (design.md 16.229).
//
//	hwnpu-dev prebuild EXPORT_DIR...            this device's architecture
//	hwnpu-dev prebuild -arch h14g,h15g EXPORT    some architectures (-arch all: every one)
//	hwnpu-dev prebuild -j 4 -force EXPORT        4 compiles at once, over what is there
//	hwnpu-dev prebuild -n EXPORT                 list what would be compiled
//	hwnpu-dev prebuild -arch all -put EXPORT     packed, and put in the sources written back to
//
// With -put (or -packed) the programs are packed beside the export (EXPORT.prebuilt/ARCH, each
// without its weights, which a device makes again from the export's bytecode) and -put puts
// each architecture's in the sources written back to (HWNPU_MODEL_SOURCE...?writeback=true): a
// machine that fetches the export fetches its architecture's too, and loads the programs
// without the Neural Engine compiler (docs/cache-sharing-design.md 7.9).
package prebuild

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/chat"
)

func Main() {
	archs := flag.String("arch", "", "the architectures, comma separated (default: this device's; all: "+strings.Join(coreai.Archs, " ")+")")
	jobs := flag.Int("j", 0, fmt.Sprintf("compiles at once (default %d here: one for every 4 GB of memory beyond 6, at most half the cores)", coreai.DefaultJobs()))
	force := flag.Bool("force", false, "compile again over prebuilt programs that are there")
	dry := flag.Bool("n", false, "list what would be compiled")
	packed := flag.Bool("packed", false, "pack the programs beside the export (EXPORT.prebuilt/ARCH) rather than whole beside their sources")
	put := flag.Bool("put", false, "pack the programs and put each architecture's in the sources written back to")
	flag.Parse()
	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: hwnpu-dev prebuild [-arch LIST] [-j N] [-force] [-n] EXPORT_DIR_OR_PROGRAM...")
		os.Exit(2)
	}
	if *put {
		// the user's definitions and the sources of the environment (HWNPU_MODEL_SOURCE...)
		if err := chat.LoadUserModels("", os.Stderr); err != nil {
			fmt.Fprintln(os.Stderr, "hwnpu-dev prebuild:", err)
			os.Exit(1)
		}
	}
	o := coreai.BuildOptions{Jobs: *jobs, Force: *force, DryRun: *dry, Log: os.Stderr, Packed: *packed || *put}
	switch *archs {
	case "":
	case "all":
		o.Archs = coreai.Archs
	default:
		o.Archs = strings.Split(*archs, ",")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	t0 := time.Now()
	results, err := coreai.Build(ctx, flag.Args(), o)
	built, skipped, failed := 0, 0, 0
	var code, weights int64
	for _, r := range results {
		switch {
		case r.Err != nil:
			failed++
		case r.Skipped:
			skipped++
		default:
			built++
			code += r.Code
			weights += r.Weights
		}
	}
	if !*dry {
		fmt.Fprintf(os.Stderr, "%d compiled in %s (%.0f MB of compiled regions beside %.1f GB of weights), %d prebuilt already, %d failed\n", built, time.Since(t0).Round(time.Second), float64(code)/1e6, float64(weights)/1e9, skipped, failed)
	}
	if *put && !*dry {
		// what failed is left out (the devices compile it): the rest is put all the same
		puts := map[[2]string]bool{}
		for _, r := range results {
			if export, ok := coreai.ExportOf(r.Asset); ok && r.Err == nil {
				puts[[2]string{export, r.Arch}] = true
			}
		}
		for k := range puts {
			if perr := chat.PutPrebuilt(k[0], k[1]); perr != nil {
				fmt.Fprintln(os.Stderr, "hwnpu-dev prebuild:", perr)
				os.Exit(1)
			}
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-dev prebuild:", err)
		if !*put {
			os.Exit(1)
		}
		return
	}
	if len(results) > 0 && !*dry {
		where := "beside their sources, as " + filepath.Base(results[0].Out)
		if o.Packed {
			where = "packed in " + filepath.Dir(filepath.Dir(results[0].Out))
		}
		fmt.Fprintf(os.Stderr, "the prebuilt programs lie %s\n", where)
	}
}
