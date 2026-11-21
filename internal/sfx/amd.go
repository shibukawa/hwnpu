package sfx

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

// AMDTool is the IRON exporter of MOSS-SoundEffect (tools/amd, built into hwnpu): the text
// encoder's, the DiT's and the DAC decoder's stages' graphs (iron_moss_dac.py) and the host's
// tables.
const AMDTool = "iron_moss_sfx.py"

// AMDExportVersion is the version of PrepareAMD's exports (a key of the shared layers:
// chat.PinnedRecipe), raised by hand when what PrepareAMD makes changes (2: the decoder's stages
// as graphs, not Core ML packages for the CPU).
const AMDExportVersion = 2

// AMDExportName is the name of the export for AMD's NPU: NAME.amd-npu2 in lower case.
func AMDExportName(name string) string { return strings.ToLower(name) + ".amd-npu2" }

// AMDTextFormat is the text encoder's format in the export for a text format of Formats: bf16
// for fp16, q8 (iron_q4's 8.5 bits, TextFP16's layers in bf16) for lut8.
func AMDTextFormat(text string) string {
	if text == "lut8" || text == "int8" {
		return "q8"
	}
	return "bf16"
}

// AMDRecipe is what PrepareAMD's export of m is made from (a key of the shared layers); nil when
// the definitions pin none of m's files.
func AMDRecipe(m chat.Model, text string) map[string]string {
	r := chat.PinnedRecipe(m, AMDTool, AMDExportVersion)
	if r != nil {
		r["args"] = "--text " + text
	}
	return r
}

// dacStage is the decoder's stage i in an export for AMD's NPU (a graph).
func dacStage(dir string, i int) string {
	return filepath.Join(dir, fmt.Sprintf("dac-%d", i))
}

// IsIronExport reports whether dir is a whole export for AMD's NPU (PrepareAMD's).
func IsIronExport(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return false
	}
	var m struct {
		Kind string `json:"kind"`
	}
	if json.Unmarshal(raw, &m) != nil || m.Kind != "moss-sfx" {
		return false
	}
	need := []string{"text/manifest.json", "dit-context/manifest.json", "host/text.safetensors", "host/timing.safetensors",
		"tokenizer/tokenizer.json", "text_encoder/config.json"}
	for first := 0; first < DitLayers; first += DitGroup {
		need = append(need, fmt.Sprintf("dit-blocks-%d/manifest.json", first))
	}
	for _, f := range need {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			return false
		}
	}
	for _, st := range DACStages() {
		if _, err := os.Stat(filepath.Join(dacStage(dir, st.Index), "manifest.json")); err != nil {
			return false
		}
	}
	return true
}

// AMDPaths are the Paths of an export for AMD's NPU.
func AMDPaths(dir string) (Paths, error) {
	p := Paths{Dir: dir, Text: []string{filepath.Join(dir, "text")}, Context: filepath.Join(dir, "dit-context"), AMD: true,
		Host: map[string][]string{"": {filepath.Join(dir, "host", "text.safetensors")}, "dit.": {filepath.Join(dir, "host", "timing.safetensors")}}}
	for first := 0; first < DitLayers; first += DitGroup {
		p.Blocks = append(p.Blocks, filepath.Join(dir, fmt.Sprintf("dit-blocks-%d", first)))
	}
	for _, st := range DACStages() {
		p.Stages = append(p.Stages, dacStage(dir, st.Index))
	}
	cfg, err := TextConfig(dir)
	if err != nil {
		return p, err
	}
	p.TextCfg = cfg
	return p, nil
}

// PrepareAMD makes the export of MOSS-SoundEffect (m) for AMD's NPU ready, its text encoder in
// AMDTextFormat(text), and returns its Paths: the export (AMDExportName) here (cache.ExportRoots;
// one made by hand goes in the sources written back to), else from the shared layers, else it
// downloads the checkpoint and exports the graphs with IRON in IRON's environment (chat.IRONEnv)
// into the cache's entries/amd-xdna2, and puts it in the sources written back to.
func PrepareAMD(ctx context.Context, m chat.Model, text string, log io.Writer) (Paths, error) {
	if log == nil {
		log = io.Discard
	}
	name := AMDExportName(m.Name)
	if d := cache.FindExport(name, IsIronExport); d != "" {
		chat.ShareMadeByHand(m, d, log)
		return AMDPaths(d)
	}
	out := filepath.Join(chat.AMDRoot(), name)
	fmtText := AMDTextFormat(text)
	imported := []chat.SharedRecipe{{Kind: chat.ImportKind, Recipe: chat.ImportedRecipe(m, cache.AMDTarget, name)}}
	err := chat.SharedExportOr(ctx, chat.ExportKind(AMDTool), AMDRecipe(m, fmtText), imported, out, log, func() error {
		python, err := chat.IRONPython(ctx)
		if err != nil {
			return fmt.Errorf("%s has no export for AMD's NPU here (%s) or in the sources, and making one needs IRON (amd/IRON with mlir-aie and Peano, Python 3.12): %w. Install it as INSTALL_AMD.md says and set %s to its virtual environment or to a script that sets it up, or put an export made elsewhere in a source (hwnpu-dev models import)",
				m.Name, out, err, chat.IRONEnv)
		}
		fmt.Fprintf(log, "%s: no export for AMD's NPU here or in the sources; downloading the checkpoint and exporting it with IRON (about three minutes once IRON's kernels are built)\n", m.Name)
		want := make([]chat.File, len(Files))
		for i, f := range Files {
			want[i] = chat.File{Repo: m.Repo, Name: f}
		}
		if err := chat.FetchFiles(ctx, want, log); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		return chat.RunIRON(ctx, python, m.Name, AMDTool, []string{chat.RepoDir(m.Repo), chat.ExportOut, "--text", fmtText}, out, log)
	})
	if err != nil {
		return Paths{}, err
	}
	if !IsIronExport(out) {
		return Paths{}, fmt.Errorf("sfx: %s is no whole export of %s", out, AMDTool)
	}
	return AMDPaths(out)
}
