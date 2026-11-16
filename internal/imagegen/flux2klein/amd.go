package flux2klein

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
)

// AMDTool is the IRON exporter of FLUX.2 klein (tools/amd, built into hwnpu).
const AMDTool = "iron_flux2klein.py"

// AMDExportVersion is the version of PrepareAMD's exports (a key of the shared layers:
// chat.PinnedRecipe), raised by hand when what PrepareAMD passes AMDTool changes.
const AMDExportVersion = 1

// AMDSizes are the pictures' sides PrepareAMD exports for: the exports measured (design.md
// 16.206 to 16.209). A step's rows are the picture's and the text's 512, whole rounds of the
// attention's 512 rows at these sizes.
var AMDSizes = []int{512, 1024}

// AMDFiles are the repository's files AMDTool reads: Files and model_index.json (whether the
// checkpoint is distilled).
var AMDFiles = append(slices.Clone(Files), "model_index.json")

// AMDExportName is the name of the export for AMD's NPU of the model named name for pictures of
// size x size: NAME.amd-npu2 at 1024, else NAME-SIZE.amd-npu2, NAME in lower case (the names
// tools/amd's guide gives and the shared layers hold the exports made by hand under).
func AMDExportName(name string, size int) string {
	n := strings.ToLower(name)
	if size != 1024 {
		n += fmt.Sprintf("-%d", size)
	}
	return n + ".amd-npu2"
}

// AMDRecipe is what PrepareAMD's export of m at size is made from (a key of the shared layers):
// chat.PinnedRecipe and the exporter's arguments. nil when the definitions pin none of m's files.
func AMDRecipe(m chat.Model, size int) map[string]string {
	r := chat.PinnedRecipe(m, AMDTool, AMDExportVersion)
	if r != nil {
		r["args"] = strings.Join(amdArgs(size), " ")
	}
	return r
}

// amdArgs are AMDTool's options for pictures of size x size: its defaults otherwise (the blocks
// for every mix of up to 4 references of the picture's size and of half its sides, and the VAE's
// encoders of both sizes).
func amdArgs(size int) []string { return []string{"--size", fmt.Sprint(size)} }

// PrepareAMD makes the export of FLUX.2 klein (m) for AMD's NPU for pictures of size x size
// ready, as hwnpu-chat makes its models' (chat.PrepareAMD), and returns its directory: one here
// (cache.ExportRoots, by AMDExportName; NAME-1024.amd-npu2 too at 1024; one made by hand there
// goes in the sources written back to: chat.ShareMadeByHand), else from the shared
// layers (the export a first run made elsewhere, AMDRecipe, or one made by hand that hwnpu-dev
// cache import put there under the same name), else it downloads the checkpoint (AMDFiles) and
// exports it with IRON in IRON's environment (chat.IRONEnv; 9 minutes at 512 x 512 on a Ryzen
// AI 9 365 with IRON's kernel cache empty), in the cache's entries/amd-xdna2, and puts that in the
// sources written back to.
func PrepareAMD(ctx context.Context, m chat.Model, size int, log io.Writer) (string, error) {
	if !slices.Contains(AMDSizes, size) {
		return "", fmt.Errorf("--size %d: FLUX.2 klein's exports for AMD's NPU are of %s pixels (tools/amd/%s --size exports another, run with -m DIR)",
			size, sizeNames(AMDSizes), AMDTool)
	}
	if log == nil {
		log = io.Discard
	}
	name := AMDExportName(m.Name, size)
	names := []string{name}
	if size == 1024 {
		names = append(names, strings.TrimSuffix(name, ".amd-npu2")+"-1024.amd-npu2")
	}
	for _, n := range names {
		if d := cache.FindExport(n, isIronExport); d != "" {
			chat.ShareMadeByHand(m, d, log) // one made by hand here goes in the sources written back to
			return d, nil
		}
	}
	out := filepath.Join(chat.AMDRoot(), name)
	imported := []chat.SharedRecipe{{Kind: chat.ImportKind, Recipe: chat.ImportedRecipe(m, cache.AMDTarget, name)}}
	err := chat.SharedExportOr(ctx, chat.ExportKind(AMDTool), AMDRecipe(m, size), imported, out, log, func() error {
		python, err := chat.IRONPython(ctx)
		if err != nil {
			return fmt.Errorf("%s has no export for AMD's NPU here (%s) or in the sources, and making one needs IRON (amd/IRON with mlir-aie and Peano, Python 3.12): %w. Install it as INSTALL_AMD.md says and set %s to its virtual environment or to a script that sets it up, or put an export made elsewhere in a source (hwnpu-dev cache import)",
				m.Name, out, err, chat.IRONEnv)
		}
		fmt.Fprintf(log, "%s: no export for AMD's NPU here or in the sources; downloading the checkpoint and exporting it with IRON (%dx%d, the blocks for up to 4 references too; 9 minutes at 512x512 on a Ryzen AI 9 365)\n",
			m.Name, size, size)
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
		args := append([]string{chat.RepoDir(m.Repo), chat.ExportOut}, amdArgs(size)...)
		return chat.RunIRON(ctx, python, m.Name, AMDTool, args, out, log)
	})
	if err != nil {
		return "", err
	}
	if !isIronExport(out) {
		return "", fmt.Errorf("flux2klein: %s is no whole export of %s", out, AMDTool)
	}
	return out, nil
}

// sizeNames are sizes as "512 x 512 and 1024 x 1024".
func sizeNames(sizes []int) string {
	var s []string
	for _, n := range sizes {
		s = append(s, fmt.Sprintf("%d x %d", n, n))
	}
	return strings.Join(s, " and ")
}
