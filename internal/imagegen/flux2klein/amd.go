package flux2klein

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/recipe"
)

// AMDTool is the IRON exporter of FLUX.2 klein (tools/amd, built into hwnpu).
const AMDTool = "iron_flux2klein.py"

// AMDExportVersion is the version of PrepareAMD's exports (a key of the shared layers:
// chat.PinnedRecipe), raised by hand when what PrepareAMD passes AMDTool changes.
const AMDExportVersion = 2

// AMDFiles are the repository's files AMDTool reads: Files and model_index.json (whether the
// checkpoint is distilled).
var AMDFiles = append(slices.Clone(Files), "model_index.json")

// AMDExportName is the name of the export for AMD's NPU of every size its caller gives:
// NAME.amd-npu2 in lower case, as the other image models' (docs/design/image-amd.md). An export of that
// name here that lacks a size asked for (the one of 16:9 at 1360 x 752 before, docs/design/image-models.md)
// is replaced by PrepareAMD.
func AMDExportName(name string) string {
	return strings.ToLower(name) + ".amd-npu2"
}

// sizeArg is a size as AMDTool's --size takes it: N for N x N, else WxH.
func sizeArg(s [2]int) string {
	if s[0] == s[1] {
		return fmt.Sprint(s[0])
	}
	return fmt.Sprintf("%dx%d", s[0], s[1])
}

// AMDRecipe is what PrepareAMD's export of m at sizes (several: one export of them all, the
// first laid out as an export of its own) is made from (a key of the shared layers):
// chat.PinnedRecipe and the exporter's arguments. nil when the definitions pin none of m's files.
func AMDRecipe(m chat.Model, sizes ...[2]int) map[string]string {
	r := chat.PinnedRecipe(m, AMDTool, AMDExportVersion)
	if r != nil {
		r["args"] = strings.Join(chat.AMDOptions(m, amdVars(m, sizes...), AMDTool), " ")
	}
	return r
}

// amdVars are the values of the recipe's placeholders (model-recipes.yaml) for pictures of each
// of sizes: AMDTool's defaults otherwise (the blocks for every mix of up to 4 references of the
// picture's size and of half its sides, and the VAE's encoders of those sizes).
func amdVars(m chat.Model, sizes ...[2]int) map[string][]string {
	var ss []string
	for _, s := range sizes {
		ss = append(ss, sizeArg(s))
	}
	return map[string][]string{recipe.In: {chat.RepoDir(m.Repo)}, recipe.Sizes: ss}
}

// PrepareAMD makes an export of FLUX.2 klein (m) for AMD's NPU that draws pictures of w x h
// ready, as hwnpu-chat makes its models' (chat.PrepareAMD), and returns its directory: the export
// of every size of sizes (AMDExportName) here (cache.ExportRoots; one made by hand goes in the sources
// written back to: chat.ShareMadeByHand), else from the shared layers (AMDRecipe, or one made by
// hand that hwnpu-dev sources import put there under its name), else it downloads the checkpoint
// (AMDFiles) and exports every size of sizes in one with IRON in IRON's environment (chat.IRONEnv), in
// the cache's entries/amd-xdna2, and puts that in the sources written back to.
func PrepareAMD(ctx context.Context, m chat.Model, sizes [][2]int, w, h int, log io.Writer) (string, error) {
	size := [2]int{w, h}
	if !slices.Contains(sizes, size) {
		return "", fmt.Errorf("--size %dx%d: FLUX.2 klein's exports for AMD's NPU are of %s (tools/amd/%s --size exports another, run with -m DIR)",
			w, h, sizeNames(sizes), AMDTool)
	}
	if log == nil {
		log = io.Discard
	}
	draws := func(d string) bool { return drawsSize(d, w, h) }
	name := AMDExportName(m.Name)
	if d := cache.FindExport(name, draws); d != "" {
		chat.ShareMadeByHand(m, d, log) // one made by hand here goes in the sources written back to
		return d, nil
	}
	out := filepath.Join(chat.AMDRoot(), name)
	// one of this name here that does not draw w x h is of other sizes (16:9 at 1360 x 752 before):
	// set aside for the one of sizes from the sources (or IRON), and put back when there is none
	if isIronExport(out) {
		aside := out + ".replaced"
		os.RemoveAll(aside)
		if err := os.Rename(out, aside); err != nil {
			return "", err
		}
		os.Remove(out + ".done")
		fmt.Fprintf(log, "%s: does not draw %dx%d (an export of other sizes); getting the one of %s\n", name, w, h, sizeNames(sizes))
		defer func() {
			if isIronExport(out) {
				os.RemoveAll(aside)
			} else {
				os.RemoveAll(out)
				os.Rename(aside, out)
			}
		}()
	}
	imported := []chat.SharedRecipe{{Kind: chat.ImportKind, Recipe: chat.ImportedRecipe(m, cache.AMDTarget, name)}}
	err := chat.SharedExportOr(ctx, chat.ExportKind(AMDTool), AMDRecipe(m, sizes...), imported, out, log, func() error {
		python, err := chat.IRONPython(ctx)
		if err != nil {
			return fmt.Errorf("%s has no export for AMD's NPU here (%s) or in the sources, and making one needs IRON (amd/IRON with mlir-aie and Peano, Python 3.12): %w. Install it as INSTALL_AMD.md says and set %s to its virtual environment or to a script that sets it up, or put an export made elsewhere in a source (hwnpu-dev sources import)",
				m.Name, out, err, chat.IRONEnv)
		}
		fmt.Fprintf(log, "%s: no export for AMD's NPU here or in the sources; downloading the checkpoint and exporting it with IRON (%s in one, the blocks for up to 4 references too; %s on a Ryzen AI 9 365)\n",
			m.Name, sizeNames(sizes), amdExportTime)
		want := make([]chat.File, len(AMDFiles))
		for i, f := range AMDFiles {
			want[i] = chat.File{Repo: m.Repo, Name: f}
		}
		if err := chat.FetchFiles(ctx, want, log); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		return chat.RunAMDRecipe(ctx, python, m, amdVars(m, sizes...), out, log, AMDTool)
	})
	if err != nil {
		return "", err
	}
	if !draws(out) {
		return "", fmt.Errorf("flux2klein: %s is no whole export of %s at %dx%d", out, AMDTool, w, h)
	}
	return out, nil
}

// amdExportTime is how long the export of every size in one took (IRON's kernel cache empty).
const amdExportTime = "about 30 minutes"

// drawsSize reports whether dir is an IRON export that draws pictures of w x h.
func drawsSize(dir string, w, h int) bool {
	return isIronExport(dir) && slices.Contains(IronSizes(dir), [2]int{w, h})
}

// sizeNames are sizes as WIDTHxHEIGHT, comma separated: "1024x1024, 512x512, ...".
func sizeNames(sizes [][2]int) string {
	s := make([]string, len(sizes))
	for i, z := range sizes {
		s[i] = fmt.Sprintf("%dx%d", z[0], z[1])
	}
	return strings.Join(s, ", ")
}
