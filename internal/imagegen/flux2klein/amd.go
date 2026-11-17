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
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
)

// AMDTool is the IRON exporter of FLUX.2 klein (tools/amd, built into hwnpu).
const AMDTool = "iron_flux2klein.py"

// AMDExportVersion is the version of PrepareAMD's exports (a key of the shared layers:
// chat.PinnedRecipe), raised by hand when what PrepareAMD passes AMDTool changes.
const AMDExportVersion = 2

// Aspects are FLUX.2 klein's sizes of the eight aspect ratios of --size: aspect.All's, but 16:9
// and 9:16 at 1344 x 768 (1.03 M pixels, multiples of 64 pixels; 1360 x 752 before, 2026-11-18),
// which the other models of multiples of 16 pixels keep.
var Aspects = aspect.With(aspect.All, aspect.Aspect{Name: "9:16", Width: 768, Height: 1344},
	aspect.Aspect{Name: "16:9", Width: 1344, Height: 768})

// AMDSizes are the pictures' sizes (width, height) PrepareAMD exports for: Aspects, 1024 x 1024
// first (laid out as an export of its own). A step whose rows are no whole rounds of the
// attention's 512 runs padded, the text's rows first.
var AMDSizes = aspect.SizesOf(Aspects)

// AMDFiles are the repository's files AMDTool reads: Files and model_index.json (whether the
// checkpoint is distilled).
var AMDFiles = append(slices.Clone(Files), "model_index.json")

// AMDExportName is the name of the export for AMD's NPU of every AMDSizes. The distinct suffix
// prevents an imported export with the previous 16:9 and 9:16 sizes from being reused.
func AMDExportName(name string) string {
	return strings.ToLower(name) + "-aspects2.amd-npu2"
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
		r["args"] = strings.Join(amdArgs(sizes...), " ")
	}
	return r
}

// amdArgs are AMDTool's options for pictures of each of sizes: its defaults otherwise (the
// blocks for every mix of up to 4 references of the picture's size and of half its sides, and
// the VAE's encoders of those sizes).
func amdArgs(sizes ...[2]int) []string {
	args := []string{"--size"}
	for _, s := range sizes {
		args = append(args, sizeArg(s))
	}
	return args
}

// PrepareAMD makes an export of FLUX.2 klein (m) for AMD's NPU that draws pictures of w x h
// ready, as hwnpu-chat makes its models' (chat.PrepareAMD), and returns its directory: the export
// of every AMDSizes (AMDExportName) here (cache.ExportRoots; one made by hand goes in the sources
// written back to: chat.ShareMadeByHand), else from the shared layers (AMDRecipe, or one made by
// hand that hwnpu-dev cache import put there under its name), else it downloads the checkpoint
// (AMDFiles) and exports every AMDSizes in one with IRON in IRON's environment (chat.IRONEnv), in
// the cache's entries/amd-xdna2, and puts that in the sources written back to.
func PrepareAMD(ctx context.Context, m chat.Model, w, h int, log io.Writer) (string, error) {
	size := [2]int{w, h}
	if !slices.Contains(AMDSizes, size) {
		return "", fmt.Errorf("--size %dx%d: FLUX.2 klein's exports for AMD's NPU are of %s (tools/amd/%s --size exports another, run with -m DIR)",
			w, h, aspectNames(), AMDTool)
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
	imported := []chat.SharedRecipe{{Kind: chat.ImportKind, Recipe: chat.ImportedRecipe(m, cache.AMDTarget, name)}}
	err := chat.SharedExportOr(ctx, chat.ExportKind(AMDTool), AMDRecipe(m, AMDSizes...), imported, out, log, func() error {
		python, err := chat.IRONPython(ctx)
		if err != nil {
			return fmt.Errorf("%s has no export for AMD's NPU here (%s) or in the sources, and making one needs IRON (amd/IRON with mlir-aie and Peano, Python 3.12): %w. Install it as INSTALL_AMD.md says and set %s to its virtual environment or to a script that sets it up, or put an export made elsewhere in a source (hwnpu-dev cache import)",
				m.Name, out, err, chat.IRONEnv)
		}
		fmt.Fprintf(log, "%s: no export for AMD's NPU here or in the sources; downloading the checkpoint and exporting it with IRON (%s in one, the blocks for up to 4 references too; %s on a Ryzen AI 9 365)\n",
			m.Name, aspectNames(), amdExportTime)
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
		args := append([]string{chat.RepoDir(m.Repo), chat.ExportOut}, amdArgs(AMDSizes...)...)
		return chat.RunIRON(ctx, python, m.Name, AMDTool, args, out, log)
	})
	if err != nil {
		return "", err
	}
	if !draws(out) {
		return "", fmt.Errorf("flux2klein: %s is no whole export of %s at %dx%d", out, AMDTool, w, h)
	}
	return out, nil
}

// amdExportTime is how long the export of every AMDSizes in one took (IRON's kernel cache empty).
const amdExportTime = "about 30 minutes"

// drawsSize reports whether dir is an IRON export that draws pictures of w x h.
func drawsSize(dir string, w, h int) bool {
	return isIronExport(dir) && slices.Contains(IronSizes(dir), [2]int{w, h})
}

// aspectNames are Aspects' names and sizes: "1:1 (1024x1024), 1:1s (512x512), ...".
func aspectNames() string {
	s := make([]string, len(Aspects))
	for i, a := range Aspects {
		s[i] = fmt.Sprintf("%s (%dx%d)", a.Name, a.Width, a.Height)
	}
	return strings.Join(s, ", ")
}
