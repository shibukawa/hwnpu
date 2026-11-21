package transcribe

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
)

// AMDTool is the IRON exporter of Whisper (tools/amd, built into hwnpu).
const AMDTool = "iron_whisper.py"

// AMDExportVersion is the version of PrepareAMD's exports (a key of the shared layers:
// chat.PinnedRecipe), raised by hand when what PrepareAMD passes AMDTool changes.
const AMDExportVersion = 1

// AMDFiles are the repository's files AMDTool reads.
var AMDFiles = []string{"config.json", "generation_config.json", "tokenizer.json", "model.safetensors"}

// AMDExportName is the name of the export for AMD's NPU: NAME.amd-npu2 in lower case.
func AMDExportName(name string) string { return strings.ToLower(name) + ".amd-npu2" }

// AMDRecipe is what PrepareAMD's export of m is made from (a key of the shared layers); nil when
// the definitions pin none of m's files.
func AMDRecipe(m chat.Model) map[string]string {
	return chat.PinnedRecipe(m, AMDTool, AMDExportVersion)
}

// PrepareAMD makes the export of Whisper (m) for AMD's NPU ready and returns its directory: the
// export (AMDExportName) here (cache.ExportRoots; one made by hand goes in the sources written
// back to), else from the shared layers, else it downloads the checkpoint and exports it with
// IRON in IRON's environment (chat.IRONEnv) into the cache's entries/amd-xdna2, and puts it in
// the sources written back to.
func PrepareAMD(ctx context.Context, m chat.Model, log io.Writer) (string, error) {
	if log == nil {
		log = io.Discard
	}
	name := AMDExportName(m.Name)
	if d := cache.FindExport(name, IsIronExport); d != "" {
		chat.ShareMadeByHand(m, d, log)
		return d, nil
	}
	out := filepath.Join(chat.AMDRoot(), name)
	imported := []chat.SharedRecipe{{Kind: chat.ImportKind, Recipe: chat.ImportedRecipe(m, cache.AMDTarget, name)}}
	err := chat.SharedExportOr(ctx, chat.ExportKind(AMDTool), AMDRecipe(m), imported, out, log, func() error {
		python, err := chat.IRONPython(ctx)
		if err != nil {
			return fmt.Errorf("%s has no export for AMD's NPU here (%s) or in the sources, and making one needs IRON (amd/IRON with mlir-aie and Peano, Python 3.12): %w. Install it as INSTALL_AMD.md says and set %s to its virtual environment or to a script that sets it up, or put an export made elsewhere in a source (hwnpu-dev models import)",
				m.Name, out, err, chat.IRONEnv)
		}
		fmt.Fprintf(log, "%s: no export for AMD's NPU here or in the sources; downloading the checkpoint and exporting it with IRON (about a minute once IRON's kernels are built)\n", m.Name)
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
		return chat.RunIRON(ctx, python, m.Name, AMDTool, []string{chat.RepoDir(m.Repo), chat.ExportOut}, out, log)
	})
	if err != nil {
		return "", err
	}
	if !IsIronExport(out) {
		return "", fmt.Errorf("transcribe: %s is no whole export of %s", out, AMDTool)
	}
	return out, nil
}
