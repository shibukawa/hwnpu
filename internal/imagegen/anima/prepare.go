package anima

import (
	"cmp"
	"context"
	"fmt"
	"github.com/shibukawa/hwnpu/internal/cache"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima/mil"
	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// ExportVersion is the layout of the programs Convert writes. Prepare names its exports by it,
// so that an export of an earlier layout is not loaded by a later hwnpu: 2 is the time part of
// mil.TimeRows rows and the patch embedding in the first group of blocks (docs/design/image-apple.md); 3 the
// silu of the DiT and the text written with exp and real_div (blocks.SiLU, docs/design/apple-neural-engine.md); 4
// the DiT's attention scale in the queries' norm, no multiply over the scores, its blocks' norms
// scaled with abs (milpart.AbsScaled; docs/design/image-apple.md) and its attentions a head at a time
// (milpart.HeadAttention; docs/design/image-apple.md).
const ExportVersion = 4

// PaddedExportVersion is the version of an Apple export whose patches are not a multiple of
// milpart.DeferredRows (880 x 1168, 752 x 1360 and their turns): its cross-attention's queries
// padded to one, which M2's and later Neural Engines' compiler takes (docs/design/amd.md). The
// other sizes' programs are as ExportVersion's, byte for byte, and keep its name.
const PaddedExportVersion = 5

// exportVersion is the version in the name of o's export: PaddedExportVersion when one of its
// sizes is padded.
func exportVersion(o ConvertOptions) int {
	for _, sz := range o.sizes() {
		gh, gw := mil.Grid(sz[0], sz[1])
		if o.Backend != "intel" && (gh*gw)%milpart.DeferredRows != 0 {
			return PaddedExportVersion
		}
	}
	return ExportVersion
}

// The Hugging Face repositories Prepare downloads from: the official split files (the diffusion
// model and Qwen3-0.6B) and, of the Diffusers version, what Convert reads besides (the VAE and
// its config, the two tokenizers).
const (
	RepoSplit     = "circlestone-labs/Anima"
	RepoDiffusers = "circlestone-labs/Anima-Base-v1.0-Diffusers"
)

// File is a file Prepare downloads: Name of the Hugging Face repository Repo.
type File struct{ Repo, Name string }

// Files are the files Prepare downloads for the diffusion model dit (models.yaml pins their
// sha256: the split files under the model's repository, the Diffusers ones under diffusers/).
func Files(dit string) []File {
	return []File{
		{RepoSplit, "split_files/diffusion_models/" + dit},
		{RepoSplit, "split_files/text_encoders/qwen_3_06b_base.safetensors"},
		{RepoDiffusers, "vae/config.json"},
		{RepoDiffusers, "vae/diffusion_pytorch_model.safetensors"},
		{RepoDiffusers, "tokenizer/tokenizer.json"},
		{RepoDiffusers, "t5_tokenizer/tokenizer.json"},
	}
}

// Setup is what Prepare makes ready.
type Setup struct {
	// Dir holds the exports; the checkpoint is the repositories' files (RepoCheckpoint).
	Dir string
	// ConvertOptions are the export's: Backend, the size, the diffusion model's file and the
	// formats (empty: Formats), Group and QChunk. Only is not used.
	ConvertOptions
	Log io.Writer
}

// Formats are the default weight formats of the text and the DiT for a backend: int8 for both on
// Core ML and Core AI (the text's 1.0 GB at fp16 is over a Core ML program's limit, and the DiT is
// as fast at int8 as at fp16 and nearer it than lut8, docs/design/image-apple.md; Core AI ran these, docs/design/apple-coreai.md);
// fp16 on Intel's NPU.
func Formats(backend string) (text, dit string) {
	if backend == "coreml" || backend == "coreai" {
		return "int8", "int8"
	}
	return "fp16", "fp16"
}

// DefaultGroup is the DiT's blocks a program holds when none is asked for: 4, and on Intel's NPU
// as many as keep a program's work near 4 blocks of 512 x 512 (1024 patches). Windows stops a job
// on the NPU that runs past its time limit (TDR, 2 s by default) with ZE_RESULT_ERROR_DEVICE_LOST,
// and a group of 4 blocks at 1024 x 1024 (4096 patches, attention 16 times the work) did on the
// Core Ultra 5 135U (docs/design/image-apple.md): there it is 1.
func DefaultGroup(backend string, width, height int) int {
	g := 4
	if backend != "intel" {
		return g
	}
	patches := (width / 16) * (height / 16)
	for g > 1 && g*patches > 4*1024 {
		g /= 2
	}
	return g
}

// FromRecipe fills what a Core AI export's o leaves out from model name's recipe
// (model-recipes.yaml apple.convert): the formats of the text and the DiT (text_format,
// dit_format), the DiT's blocks a program holds (group) and the queries an attention scores at a
// time (qchunk). Other backends take Formats, DefaultGroup and the code's qchunk.
func FromRecipe(name string, o *ConvertOptions) error {
	if o.Backend != llm.BackendCoreAI {
		return nil
	}
	if err := chat.FillConvert(name, map[string]*int{"group": &o.Group, "qchunk": &o.QChunk},
		map[string]*string{"text_format": &o.TextFormat, "dit_format": &o.DitFormat}); err != nil {
		return fmt.Errorf("anima: %w", err)
	}
	return nil
}

// ExportName is the directory Prepare writes the export of o into, in Setup.Dir: what changes
// the programs is in its name.
// An export of several sizes (Core AI) names them with aspect.Tag ("8sizes-HASH"); it is one
// export whatever size is drawn (new-model-checklist.md 11.4).
func ExportName(o ConvertOptions) string {
	return fmt.Sprintf("%s-%s-%s-%s-g%d-q%d-v%d.%s", strings.TrimSuffix(o.Dit, ".safetensors"), aspect.Tag(o.sizes()),
		o.TextFormat, o.DitFormat, o.Group, o.QChunk, exportVersion(o), cmp.Or(o.Backend, "coreml"))
}

// Prepare makes Anima ready in s.Dir (its checkpoint: RepoCheckpoint), as hwnpu-chat makes its models ready: it downloads the
// checkpoint files that are missing, converts the export that is missing and returns its
// directory. NAME.done, written last, says an export is whole; one cut off is converted again.
func Prepare(ctx context.Context, s Setup) (string, error) {
	o := s.ConvertOptions
	o.Only = nil
	if o.Dit == "" {
		o.Dit = "anima-turbo-v1.1.safetensors"
	}
	text, dit := Formats(o.Backend)
	if o.TextFormat == "" {
		o.TextFormat = text
	}
	if o.DitFormat == "" {
		o.DitFormat = dit
	}
	if o.Group <= 0 {
		o.Group = DefaultGroup(o.Backend, o.Width, o.Height)
	}
	log := s.Log
	if log == nil {
		log = io.Discard
	}
	out := filepath.Join(s.Dir, ExportName(o))
	done := out + ".done"
	if llm.FetchExport(ctx, out, done) { // here, or from the shared layers
		return out, nil
	}
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return "", err
	}
	var want []chat.File
	for _, f := range Files(o.Dit) {
		want = append(want, chat.File{Repo: f.Repo, Name: f.Name})
	}
	if err := chat.FetchFiles(ctx, want, log); err != nil {
		return "", err
	}
	if err := llm.MayConvert(ctx, out); err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	// an export cut off is written again from nothing
	if err := os.RemoveAll(out); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converting %s (%s, text %s, DiT %s); this takes a minute or two\n", filepath.Base(out), aspect.Tag(o.sizes()), o.TextFormat, o.DitFormat)
	t0 := time.Now()
	if err := Convert(RepoCheckpoint(), out, o, log); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converted in %.0f s\n", time.Since(t0).Seconds())
	// what is left of the checkpoints read goes back to the OS before the programs load
	debug.FreeOSMemory()
	if err := cache.WriteDone(done, out); err != nil {
		return "", err
	}
	// a Core AI export goes in the shared layers that are written (llm.SharedExports)
	llm.ShareExport(ctx, out)
	return out, nil
}
