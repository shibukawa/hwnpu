package modelscmd

import (
	"os"
	"path/filepath"
	"strings"
)

// PrebuildCmd is hwnpu-dev models prebuild: Core AI's programs compiled ahead of time with
// coreai-build (Xcode 27's Metal Toolchain), X.ARCH.aimodelc for each X.aimodel of an export. On
// a device of that architecture hwnpu loads the prebuilt program while Core AI's cache has no
// entry for it, and the device finishes it in a third of the time the Neural Engine compiler
// takes over the source (docs/design/apple-coreai.md). A model is made ready for coreai first
// (export: taken from the sources, else converted); its exports are compiled.
//
// What is compiled goes packed (EXPORT.prebuilt/ARCH, each program without its weights, which a
// device makes again from the export's bytecode) in the sources written back to, an
// architecture's at a time: a machine that fetches the export fetches its architecture's too
// (docs/cache-sharing-design.md 7.9). --packed packs them without putting them, --no-put leaves
// them whole beside their sources.
type PrebuildCmd struct {
	Sel    []string `arg:"" name:"MODEL|DIR" help:"Models of the definitions (their Core AI exports), or export folders and programs (.coreai, .aimodel)."`
	Arch   []string `help:"The architectures (h14g, ...; all: every one); this device's when left out." placeholder:"ARCH"`
	Jobs   int      `short:"j" help:"Compiles at once (0: one for every 4 GB of memory beyond 6, at most half the cores)."`
	Force  bool     `help:"Compile again over prebuilt programs that are there."`
	DryRun bool     `short:"n" help:"List what would be compiled."`
	Packed bool     `help:"Pack the programs beside the export (EXPORT.prebuilt/ARCH) without putting them in the sources."`
	NoPut  bool     `help:"Leave the programs whole beside their sources, not packed nor put in the sources."`
	Memory []int    `help:"Of a model, only the forms of machines of these GB (as export)." placeholder:"GB"`
	Chip   []string `help:"Of a model, only the exports of Macs of these chips (as export)." placeholder:"CHIP"`
}

// isPath reports whether a selection is a folder or a program rather than a model's name.
func isPath(s string) bool {
	if strings.HasSuffix(s, ".coreai") || strings.HasSuffix(s, ".aimodel") || strings.ContainsRune(s, filepath.Separator) {
		return true
	}
	_, err := os.Stat(s)
	return err == nil
}

// coreAIPaths are the exports and programs among the files a preparation gave.
func coreAIPaths(paths map[string][]string) []string {
	var out []string
	seen := map[string]bool{}
	for _, ps := range paths {
		for _, p := range ps {
			// a program of an export (NAME.coreai/model.aimodel) is compiled with its export
			for d := p; d != "." && d != string(filepath.Separator) && d != ""; d = filepath.Dir(d) {
				if strings.HasSuffix(d, ".coreai") {
					p = d
					break
				}
			}
			if (strings.HasSuffix(p, ".coreai") || strings.HasSuffix(p, ".aimodel")) && !seen[p] {
				seen[p] = true
				out = append(out, p)
			}
		}
	}
	return out
}
