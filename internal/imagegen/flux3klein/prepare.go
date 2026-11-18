package flux3klein

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// Setup is what Prepare makes ready.
type Setup struct {
	// Dir holds the exports; the checkpoint is the repository's files (chat.RepoDir).
	Dir string
	// Repo is the Hugging Face repository (black-forest-labs/flux-3-image-klein).
	Repo string
	// Name names the export.
	Name string
	ConvertOptions
	Log io.Writer
}

// StatsFile is the file in Setup.Dir Prepare takes the DiT's activation statistics from when the
// options name none (ConvertOptions.Stats): tools/ref/flux3_quant.py's calibration, 4 prompts.
const StatsFile = "flux3-klein.x2.safetensors"

// Files are the files of the repository Convert reads (models.yaml pins the large ones' sha256):
// the diffusers layout, not its single-file copy of the DiT nor its fp8 and nvfp4 variants.
var Files = []string{
	"transformer/config.json",
	"transformer/diffusion_pytorch_model.safetensors.index.json",
	"transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
	"transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
	"text_encoder/config.json",
	"text_encoder/tokenizer.json",
	"text_encoder/model.safetensors.index.json",
	"text_encoder/model-00001-of-00002.safetensors",
	"text_encoder/model-00002-of-00002.safetensors",
	"vae/config.json",
	"vae/diffusion_pytorch_model.safetensors",
}

// Prepare makes FLUX.3 klein ready in s.Dir for Apple's Neural Engine: it downloads the checkpoint
// files that are missing, converts the export that is missing and returns its directory.
// NAME.done, written last, says an export is whole; one cut off is converted again.
func Prepare(ctx context.Context, s Setup) (string, error) {
	o := s.ConvertOptions
	log := s.Log
	if log == nil {
		log = io.Discard
	}
	if o.Stats == "" {
		if p := filepath.Join(s.Dir, StatsFile); fileExists(p) {
			o.Stats = p
		}
	}
	o.defaults()
	out := filepath.Join(s.Dir, ExportName(s.Name, o))
	if llm.FetchExport(ctx, out, out+".done") { // here, or from the shared layers
		return out, nil
	}
	var want []chat.File
	for _, f := range Files {
		want = append(want, chat.File{Repo: s.Repo, Name: f})
	}
	if err := chat.FetchFiles(ctx, want, log); err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return "", err
	}
	if err := os.RemoveAll(out); err != nil {
		return "", err
	}
	if o.Stats == "" {
		fmt.Fprintf(log, "no activation statistics (%s): the DiT's lut8 tables are spaced evenly, which loses fine detail; see docs/design.md (FLUX.3 klein)\n",
			filepath.Join(s.Dir, StatsFile))
	}
	fmt.Fprintf(log, "converting %s (Qwen3-VL %s, DiT %s): several minutes; the first load then compiles the programs for the NPU\n",
		filepath.Base(out), o.TextFormat, o.DitFormat)
	t0 := time.Now()
	if err := Convert(chat.RepoDir(s.Repo), out, o, log); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converted in %s\n", time.Since(t0).Round(time.Second))
	debug.FreeOSMemory()
	if err := cache.WriteDone(out+".done", out); err != nil {
		return "", err
	}
	llm.ShareExport(ctx, out)
	return out, nil
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}
