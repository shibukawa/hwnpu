package modelscmd

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/bonsaiai/internal/chat"
)

// compileFlags say how package compiles Core AI's programs for coreai-ARCH (compile): ahead of
// time with coreai-build (Xcode 27's Metal Toolchain), X.ARCH.aimodelc for each X.aimodel of an
// export. On a device of that architecture bonsaiai loads the prebuilt program while Core AI's cache
// has no entry for it, and the device finishes it in a third of the time the Neural Engine
// compiler takes over the source (docs/design/apple-coreai.md).
//
// What is compiled goes packed (EXPORT.prebuilt/ARCH, each program without its weights, which a
// device makes again from the export's bytecode) in the sources written back to, an
// architecture's at a time: a machine that fetches the export fetches its architecture's too
// (docs/cache-sharing-design.md 7.9). --packed packs them without putting them, --no-push leaves
// them whole beside their sources. The architectures, the compiles at once and the programs left
// to the device are the model's recipe's (model-recipes.yaml, docs/model-recipes.md) where the
// target and -j do not say.
type compileFlags struct {
	Jobs     int  `short:"j" help:"For coreai-ARCH, compiles at once (0: one for every 4 GB of memory beyond 6, at most half the cores)."`
	Force    bool `help:"For coreai-ARCH, compile again over prebuilt programs that are there."`
	DryRun   bool `short:"n" help:"For coreai-ARCH, list what would be compiled (the exports are made all the same)."`
	Packed   bool `help:"For coreai-ARCH, pack the programs beside the export (EXPORT.prebuilt/ARCH) without putting them in the sources."`
	NoPush   bool `help:"For coreai-ARCH, leave the programs whole beside their sources, not packed nor put in the sources."`
	AnyTools bool `help:"For coreai-ARCH, build with a coreai-build of another version than model-recipes.yaml's tools (to check a new Metal Toolchain)."`
}

// isPath reports whether a selection is a folder or a program rather than a model's name.
func isPath(s string) bool {
	if strings.HasSuffix(s, ".coreai") || strings.HasSuffix(s, ".aimodel") || strings.ContainsRune(s, filepath.Separator) {
		return true
	}
	return exists(s)
}

// exists reports whether there is a file or folder at path.
func exists(path string) bool { _, err := os.Stat(path); return err == nil }

// isModel reports whether name is a model's of the definitions.
func isModel(name string) bool { _, ok := chat.Lookup(name); return ok }

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
