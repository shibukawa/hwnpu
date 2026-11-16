package flux2klein

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
	// Repo is the Hugging Face repository (black-forest-labs/FLUX.2-klein-4B).
	Repo string
	// Name names the export (FLUX.2-klein-4B).
	Name string
	ConvertOptions
	Log io.Writer
}

// Files are the files of the repository Convert reads (models.yaml pins the large ones' sha256);
// its single-file copy of the DiT is not fetched.
var Files = []string{
	"transformer/config.json",
	"transformer/diffusion_pytorch_model.safetensors",
	"scheduler/scheduler_config.json",
	"text_encoder/config.json",
	"text_encoder/model.safetensors.index.json",
	"text_encoder/model-00001-of-00002.safetensors",
	"text_encoder/model-00002-of-00002.safetensors",
	"tokenizer/tokenizer.json",
	"vae/config.json",
	"vae/diffusion_pytorch_model.safetensors",
}

// Prepare makes FLUX.2 klein ready in s.Dir for Apple's Neural Engine, as hwnpu-chat makes its
// models ready: it downloads the checkpoint files that are missing, converts the export that is
// missing and returns its directory. NAME.done, written last, says an export is whole; one cut
// off is converted again.
func Prepare(ctx context.Context, s Setup) (string, error) {
	o := s.ConvertOptions
	o.Only = nil
	o.defaults()
	log := s.Log
	if log == nil {
		log = io.Discard
	}
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
	fmt.Fprintf(log, "converting %s (%dx%d, Qwen3 %s, DiT %s): a few minutes; the first load then compiles the programs for the NPU\n",
		filepath.Base(out), o.Width, o.Height, o.TextFormat, o.DitFormat)
	t0 := time.Now()
	if err := Convert(chat.RepoDir(s.Repo), out, o, log); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converted in %s\n", time.Since(t0).Round(time.Second))
	debug.FreeOSMemory()
	if err := cache.WriteDone(out+".done", out); err != nil {
		return "", err
	}
	// a Core AI export goes in the shared layers that are written (llm.SharedExports)
	llm.ShareExport(ctx, out)
	return out, nil
}
