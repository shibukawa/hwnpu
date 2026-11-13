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
)

// ExportVersion is the layout of the programs Convert writes. Prepare names its exports by it,
// so that an export of an earlier layout is not loaded by a later hwnpu: 2 is the time part of
// mil.TimeRows rows and the patch embedding in the first group of blocks (design.md 16.85); 3 the
// silu of the DiT and the text written with exp and real_div (blocks.SiLU, design.md 16.95).
const ExportVersion = 3

// The Hugging Face repositories Prepare downloads from: the official split files (the diffusion
// model and Qwen3-0.6B) and, of the Diffusers version, what Convert reads besides (the VAE and
// its config, the two tokenizers).
const (
	RepoSplit     = "circlestone-labs/Anima"
	RepoDiffusers = "circlestone-labs/Anima-Base-v1.0-Diffusers"
)

// File is a file Prepare downloads: Name of the Hugging Face repository Repo, at Path in Dir.
type File struct{ Repo, Name, Path string }

// Files are the files Prepare downloads for the diffusion model dit (models.yaml pins their
// sha256: the split files under the model's repository, the Diffusers ones under diffusers/).
func Files(dit string) []File {
	return []File{
		{RepoSplit, "split_files/diffusion_models/" + dit, "diffusion_models/" + dit},
		{RepoSplit, "split_files/text_encoders/qwen_3_06b_base.safetensors", "text_encoders/qwen_3_06b_base.safetensors"},
		{RepoDiffusers, "vae/config.json", "diffusers/vae/config.json"},
		{RepoDiffusers, "vae/diffusion_pytorch_model.safetensors", "diffusers/vae/diffusion_pytorch_model.safetensors"},
		{RepoDiffusers, "tokenizer/tokenizer.json", "diffusers/tokenizer/tokenizer.json"},
		{RepoDiffusers, "t5_tokenizer/tokenizer.json", "diffusers/t5_tokenizer/tokenizer.json"},
	}
}

// Setup is what Prepare makes ready.
type Setup struct {
	// Dir holds the checkpoint in Convert's layout (diffusion_models/, text_encoders/,
	// diffusers/) and the exports.
	Dir string
	// ConvertOptions are the export's: Backend, the size, the diffusion model's file and the
	// formats (empty: Formats), Group and QChunk. Only is not used.
	ConvertOptions
	Log io.Writer
}

// Formats are the default weight formats of the text and the DiT for a backend: int8 for both on
// Core ML and Core AI (the text's 1.0 GB at fp16 is over a Core ML program's limit, and the DiT is
// as fast at int8 as at fp16 and nearer it than lut8, design.md 16.85; Core AI ran these, 16.169);
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
// Core Ultra 5 135U (design.md 16.146): there it is 1.
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

// ExportName is the directory Prepare writes the export of o into, in Setup.Dir: what changes
// the programs is in its name.
func ExportName(o ConvertOptions) string {
	return fmt.Sprintf("%s-%dx%d-%s-%s-g%d-q%d-v%d.%s", strings.TrimSuffix(o.Dit, ".safetensors"), o.Width, o.Height,
		o.TextFormat, o.DitFormat, o.Group, o.QChunk, ExportVersion, cmp.Or(o.Backend, "coreml"))
}

// Prepare makes Anima ready in s.Dir, as hwnpu-chat makes its models ready: it downloads the
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
	if _, err := os.Stat(done); err == nil {
		return out, nil
	}
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return "", err
	}
	var want []chat.File
	for _, f := range Files(o.Dit) {
		want = append(want, chat.File{Repo: f.Repo, Name: f.Name, Path: filepath.Join(s.Dir, f.Path)})
	}
	chat.Expect(ctx, want)
	for _, f := range Files(o.Dit) {
		path := filepath.Join(s.Dir, f.Path)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return "", err
		}
		if err := chat.Fetch(ctx, f.Repo, f.Name, path, log); err != nil {
			return "", err
		}
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	// an export cut off is written again from nothing
	if err := os.RemoveAll(out); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converting %s (%dx%d, text %s, DiT %s); this takes under a minute\n", filepath.Base(out), o.Width, o.Height, o.TextFormat, o.DitFormat)
	t0 := time.Now()
	if err := Convert(s.Dir, out, o, log); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converted in %.0f s\n", time.Since(t0).Seconds())
	// what is left of the checkpoints read goes back to the OS before the programs load
	debug.FreeOSMemory()
	return out, cache.WriteDone(done, out)
}
