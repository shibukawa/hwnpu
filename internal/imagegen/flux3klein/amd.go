package flux3klein

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
)

// AMDTool is the IRON exporter of FLUX.3 klein (tools/amd, built into hwnpu).
const AMDTool = "iron_flux3klein.py"

// AMDExportVersion is the version of PrepareAMD's exports (a key of the shared layers:
// chat.PinnedRecipe), raised by hand when what PrepareAMD passes AMDTool changes.
const AMDExportVersion = 2

// AMDFiles are the repository's files AMDTool reads: the DiT's single-file copy (model.safetensors,
// BFL's names) and transformer/config.json, Qwen3-VL's language model and tokenizer, the VAE. The
// sources mirror the repository's top level; its diffusers shards of the DiT are the same weights
// (Files: Apple's Neural Engine reads those).
var AMDFiles = []string{
	"transformer/config.json",
	"model.safetensors",
	"text_encoder/config.json",
	"text_encoder/tokenizer.json",
	"text_encoder/model.safetensors.index.json",
	"text_encoder/model-00001-of-00002.safetensors",
	"text_encoder/model-00002-of-00002.safetensors",
	"vae/config.json",
	"vae/diffusion_pytorch_model.safetensors",
}

// AMDExportName is the name of the export for AMD's NPU of every size its caller gives (models.yaml
// npu.amd's eight): NAME.amd-npu2 in lower case, as the other image models' (design.md 16.243). An
// export of that name here that lacks a size asked for (the one of 1024 x 1024 and 512 x 512 alone
// before, design.md 16.250) is replaced by PrepareAMD.
func AMDExportName(name string) string {
	return strings.ToLower(name) + ".amd-npu2"
}

// amdArgs are AMDTool's options: every size of sizes in one export (the first's graphs hold the
// banks).
func amdArgs(sizes [][2]int) []string {
	args := []string{"--size"}
	for _, s := range sizes {
		if s[0] == s[1] {
			args = append(args, fmt.Sprint(s[0]))
		} else {
			args = append(args, fmt.Sprintf("%dx%d", s[0], s[1]))
		}
	}
	return args
}

// AMDRecipe is what PrepareAMD's export of m at sizes is made from (a key of the shared layers):
// chat.PinnedRecipe and the exporter's arguments. nil when the definitions pin none of m's files.
func AMDRecipe(m chat.Model, sizes [][2]int) map[string]string {
	r := chat.PinnedRecipe(m, AMDTool, AMDExportVersion)
	if r != nil {
		r["args"] = strings.Join(amdArgs(sizes), " ")
	}
	return r
}

// amdExportTime is how long the export of the eight sizes took on a Ryzen AI 9 365 (IRON's kernels
// built already, as FLUX.2 klein's export leaves them: their first build takes longer).
const amdExportTime = "about 11 minutes once IRON's kernels are built"

// PrepareAMD makes an export of FLUX.3 klein (m) for AMD's NPU that draws pictures of w x h ready
// and returns its directory: the export of every size of sizes (AMDExportName) here
// (cache.ExportRoots; one made by hand goes in the sources written back to: chat.ShareMadeByHand),
// else from the shared layers (AMDRecipe, or one made by hand that hwnpu-dev cache import put there
// under its name), else it downloads the checkpoint (AMDFiles) and exports every size of sizes in
// one with IRON in IRON's environment (chat.IRONEnv), in the cache's entries/amd-xdna2, and puts
// that in the sources written back to.
func PrepareAMD(ctx context.Context, m chat.Model, sizes [][2]int, w, h int, log io.Writer) (string, error) {
	if !slices.Contains(sizes, [2]int{w, h}) {
		return "", fmt.Errorf("--size %dx%d: FLUX.3 klein's export for AMD's NPU draws %s (tools/amd/%s --size exports another, run with -m DIR)",
			w, h, sizeNames(sizes), AMDTool)
	}
	if log == nil {
		log = io.Discard
	}
	draws := func(d string) bool { return IsIronExport(d) && slices.Contains(IronSizes(d), [2]int{w, h}) }
	name := AMDExportName(m.Name)
	if d := cache.FindExport(name, draws); d != "" {
		chat.ShareMadeByHand(m, d, log)
		return d, nil
	}
	out := filepath.Join(chat.AMDRoot(), name)
	// one of this name here that does not draw w x h is of other sizes (1024 x 1024 and 512 x 512
	// alone before): set aside for the one of sizes from the sources (or IRON), and put back when
	// there is none
	if IsIronExport(out) {
		aside := out + ".replaced"
		os.RemoveAll(aside)
		if err := os.Rename(out, aside); err != nil {
			return "", err
		}
		os.Remove(out + ".done")
		fmt.Fprintf(log, "%s: does not draw %dx%d (an export of other sizes); getting the one of %s\n", name, w, h, sizeNames(sizes))
		defer func() {
			if IsIronExport(out) {
				os.RemoveAll(aside)
			} else {
				os.RemoveAll(out)
				os.Rename(aside, out)
			}
		}()
	}
	imported := []chat.SharedRecipe{{Kind: chat.ImportKind, Recipe: chat.ImportedRecipe(m, cache.AMDTarget, name)}}
	err := chat.SharedExportOr(ctx, chat.ExportKind(AMDTool), AMDRecipe(m, sizes), imported, out, log, func() error {
		python, err := chat.IRONPython(ctx)
		if err != nil {
			return fmt.Errorf("%s has no export for AMD's NPU here (%s) or in the sources, and making one needs IRON (amd/IRON with mlir-aie and Peano, Python 3.12): %w. Install it as INSTALL_AMD.md says and set %s to its virtual environment or to a script that sets it up, or put an export made elsewhere in a source (hwnpu-dev cache import)",
				m.Name, out, err, chat.IRONEnv)
		}
		fmt.Fprintf(log, "%s: no export for AMD's NPU here or in the sources; downloading the checkpoint and exporting it with IRON (%s in one; %s on a Ryzen AI 9 365)\n",
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
		args := append([]string{chat.RepoDir(m.Repo), chat.ExportOut}, amdArgs(sizes)...)
		return chat.RunIRON(ctx, python, m.Name, AMDTool, args, out, log)
	})
	if err != nil {
		return "", err
	}
	if !draws(out) {
		return "", fmt.Errorf("flux3klein: %s is no whole export of %s at %dx%d", out, AMDTool, w, h)
	}
	return out, nil
}
