package speech

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
)

// AMDTool is the IRON exporter of LFM3-Audio's speech (tools/amd, built into hwnpu): the text
// model (iron_export_lm.py lfm3tts), the depthformer's and the codec decoder's graphs, and the
// host's tables.
const AMDTool = "iron_lfm3speech.py"

// AMDExportVersion is the version of PrepareAMD's exports (a key of the shared layers:
// chat.PinnedRecipe), raised by hand when what PrepareAMD passes AMDTool changes.
const AMDExportVersion = 1

// AMDFiles are the repository's files AMDTool reads.
var AMDFiles = []string{"config.json", "generation_config.json", "tokenizer.json", "model.safetensors",
	"audio_tokenizer/config.json", "audio_tokenizer/model.safetensors"}

// AMDExportName is the name of the export for AMD's NPU: NAME.amd-npu2 in lower case.
func AMDExportName(name string) string { return strings.ToLower(name) + ".amd-npu2" }

// AMDRecipe is what PrepareAMD's export of m is made from (a key of the shared layers); nil when
// the definitions pin none of m's files.
func AMDRecipe(m chat.Model) map[string]string {
	return chat.PinnedRecipe(m, AMDTool, AMDExportVersion)
}

// IsIronExport reports whether dir is an export of tools/amd/iron_lfm3speech.py.
func IsIronExport(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return false
	}
	var m struct {
		Kind string `json:"kind"`
	}
	if json.Unmarshal(raw, &m) != nil || m.Kind != "lfm3speech" {
		return false
	}
	for _, f := range []string{"text/manifest.json", "depth/manifest.json", "codec/manifest.json", "tables.safetensors", "audio_tokenizer/config.json"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			return false
		}
	}
	return true
}

// IronConfig is the Config of the export dir of tools/amd/iron_lfm3speech.py.
func IronConfig(dir string) Config {
	return Config{Dir: dir, Weights: filepath.Join(dir, "tables.safetensors"), IronText: filepath.Join(dir, "text"),
		Depth: filepath.Join(dir, "depth"), Codec: filepath.Join(dir, "codec")}
}

// PrepareAMD makes the export of LFM3-Audio's speech (m) for AMD's NPU ready and returns its
// Config: the export (AMDExportName) here (cache.ExportRoots; one made by hand goes in the
// sources written back to), else from the shared layers, else it downloads the checkpoint and
// exports it with IRON in IRON's environment (chat.IRONEnv) into the cache's entries/amd-xdna2,
// and puts it in the sources written back to.
func PrepareAMD(ctx context.Context, m chat.Model, log io.Writer) (Config, error) {
	if log == nil {
		log = io.Discard
	}
	name := AMDExportName(m.Name)
	if d := cache.FindExport(name, IsIronExport); d != "" {
		chat.ShareMadeByHand(m, d, log)
		return IronConfig(d), nil
	}
	out := filepath.Join(chat.AMDRoot(), name)
	imported := []chat.SharedRecipe{{Kind: chat.ImportKind, Recipe: chat.ImportedRecipe(m, cache.AMDTarget, name)}}
	err := chat.SharedExportOr(ctx, chat.ExportKind(AMDTool), AMDRecipe(m), imported, out, log, func() error {
		python, err := chat.IRONPython(ctx)
		if err != nil {
			return fmt.Errorf("%s has no export for AMD's NPU here (%s) or in the sources, and making one needs IRON (amd/IRON with mlir-aie and Peano, Python 3.12): %w. Install it as INSTALL_AMD.md says and set %s to its virtual environment or to a script that sets it up, or put an export made elsewhere in a source (hwnpu-dev sources import)",
				m.Name, out, err, chat.IRONEnv)
		}
		fmt.Fprintf(log, "%s: no export for AMD's NPU here or in the sources; downloading the checkpoint and exporting it with IRON (about two minutes once IRON's kernels are built)\n", m.Name)
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
		return chat.RunAMDRecipe(ctx, python, m, chat.InVar(chat.RepoDir(m.Repo)), out, log, AMDTool)
	})
	if err != nil {
		return Config{}, err
	}
	if !IsIronExport(out) {
		return Config{}, fmt.Errorf("speech: %s is no whole export of %s", out, AMDTool)
	}
	return IronConfig(out), nil
}
