package chroma

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
)

// Setup is what Prepare makes ready.
type Setup struct {
	// Dir holds the checkpoints (a directory per repository; T5-XXL and the VAE in SharedDir,
	// the same files in every Chroma) and the exports.
	Dir string
	// Repo is the Hugging Face repository (lodestones/Chroma1-Flash).
	Repo string
	// Name names the export (Chroma1-Flash).
	Name string
	ConvertOptions
	Log io.Writer
}

// SharedDir is the directory of Dir that holds T5-XXL (text_encoder/), the VAE (vae/) and T5's
// tokenizer.json: Chroma1-Flash's and Chroma1-HD's are the same files (models.yaml's sha256).
const SharedDir = "t5-xxl"

// File is a file of a Chroma repository; Shared ones go to SharedDir.
type File struct {
	Name   string
	Shared bool
}

// Files are the files of a Chroma repository Prepare downloads (models.yaml pins their sha256).
var Files = []File{
	{"transformer/config.json", false},
	{"transformer/diffusion_pytorch_model.safetensors.index.json", false},
	{"transformer/diffusion_pytorch_model-00001-of-00002.safetensors", false},
	{"transformer/diffusion_pytorch_model-00002-of-00002.safetensors", false},
	{"scheduler/scheduler_config.json", false},
	{"text_encoder/config.json", true},
	{"text_encoder/model.safetensors.index.json", true},
	{"text_encoder/model-00001-of-00002.safetensors", true},
	{"text_encoder/model-00002-of-00002.safetensors", true},
	{"vae/config.json", true},
	{"vae/diffusion_pytorch_model.safetensors", true},
}

// tokenizerFile is T5's tokenizer.json of Anima's repository: Chroma's has only spiece.model, and
// FLUX's tokenizer.json normalizes with a Sequence the Go tokenizer does not read (the same
// vocabulary and charsmap, design.md 16.156).
const tokenizerFile = "t5_tokenizer/tokenizer.json"

// ExportName is the directory Prepare writes the export of name and o into: what changes the
// programs is in it.
func ExportName(name string, o ConvertOptions) string {
	o.defaults()
	tag := func(f string) string { return strings.NewReplacer(" ", "+", "=", "").Replace(f) }
	s := fmt.Sprintf("%s-%dx%d-%s-%s", strings.ToLower(name), o.Width, o.Height, tag(o.t5Formats()), tag(o.ditFormats()))
	if o.TextRows != 512 {
		s += fmt.Sprintf("-t%d", o.TextRows)
	}
	return fmt.Sprintf("%s-v%d.coreml", s, ExportVersion)
}

// Prepare makes a Chroma ready in s.Dir, as hwnpu-chat makes its models ready: it downloads the
// checkpoint files that are missing, converts the export that is missing and returns its
// directory. NAME.done, written last, says an export is whole; one cut off is converted again.
func Prepare(ctx context.Context, s Setup) (string, error) {
	o := s.ConvertOptions
	o.Only = nil
	o.defaults()
	log := s.Log
	if log == nil {
		log = io.Discard
	}
	out := filepath.Join(s.Dir, ExportName(s.Name, o))
	if _, err := os.Stat(out + ".done"); err == nil {
		return out, nil
	}
	root := filepath.Join(s.Dir, filepath.Base(s.Repo))
	shared := filepath.Join(s.Dir, SharedDir)
	at := func(f File) string {
		if f.Shared {
			return filepath.Join(shared, f.Name)
		}
		return filepath.Join(root, f.Name)
	}
	var want []chat.File
	for _, f := range Files {
		want = append(want, chat.File{Repo: s.Repo, Name: f.Name, Path: at(f)})
	}
	tok := filepath.Join(shared, "tokenizer.json")
	want = append(want, chat.File{Repo: anima.RepoDiffusers, Name: tokenizerFile, Path: tok})
	chat.Expect(ctx, want)
	for _, f := range want {
		if err := os.MkdirAll(filepath.Dir(f.Path), 0o755); err != nil {
			return "", err
		}
		if err := chat.Fetch(ctx, f.Repo, f.Name, f.Path, log); err != nil {
			return "", err
		}
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := os.RemoveAll(out); err != nil {
		return "", err
	}
	o.Tokenizer = tok
	// the parts another export here holds the same: T5's programs (every size, Flash and HD
	// but the last group), the VAE's (the same size)
	done, _ := filepath.Glob(filepath.Join(s.Dir, "*.coreml.done"))
	for _, d := range done {
		o.LinkFrom = append(o.LinkFrom, strings.TrimSuffix(d, ".done"))
	}
	fmt.Fprintf(log, "converting %s (%dx%d, T5 %s, DiT %s): about a minute; the first load then compiles the programs for the NPU (over an hour at 1024 x 1024 on an M1)\n",
		filepath.Base(out), o.Width, o.Height, o.TextFormat, o.DitFormat)
	t0 := time.Now()
	if err := Convert(root, shared, out, o, log); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converted in %s\n", time.Since(t0).Round(time.Second))
	debug.FreeOSMemory()
	return out, os.WriteFile(out+".done", []byte(out), 0o644)
}
