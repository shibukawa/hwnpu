package qwenimage

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// Setup is what Prepare makes ready.
type Setup struct {
	// Dir holds the checkpoints (a directory per repository, Qwen3-VL-8B in TextDir) and the
	// exports.
	Dir string
	// Repo is the Hugging Face repository (Qwen/Qwen-Image-2.1-Turbo).
	Repo string
	// Name names the export (Qwen-Image-2.1-Turbo).
	Name string
	ConvertOptions
	Log io.Writer
}

// TextDir is the directory of Dir that holds Qwen3-VL-8B (text_encoder/ and processor/): the same
// weights in every Qwen-Image-2.1 repository (image-models-survey-qwen-chroma.md 2).
const TextDir = "Qwen3-VL-8B"

// File is a file of a Qwen-Image-2.1 repository Prepare downloads: Text, one of TextDir's.
type File struct {
	Name string
	Text bool
}

// Files are what Convert reads (models.yaml pins the large ones' sha256). The text encoder's
// vision tower is in model.safetensors too; it is not read.
var Files = []File{
	{"model_index.json", false},
	{"scheduler/scheduler_config.json", false},
	{"transformer/config.json", false},
	{"transformer/diffusion_pytorch_model.safetensors.index.json", false},
	{"transformer/diffusion_pytorch_model-00001-of-00002.safetensors", false},
	{"transformer/diffusion_pytorch_model-00002-of-00002.safetensors", false},
	{"vae/config.json", false},
	{"vae/diffusion_pytorch_model.safetensors", false},
	{"text_encoder/config.json", true},
	{"text_encoder/model.safetensors", true},
	{"processor/tokenizer.json", true},
}

// ExportName is the directory Prepare writes the export of name and o into: what changes the
// programs is in it.
func ExportName(name string, o ConvertOptions) string {
	o.defaults()
	q := func(format, quant string) string {
		if !strings.HasPrefix(format, "lut") || quant == "uniform" {
			return format
		}
		return format + "-" + map[string]string{"kmeans": "km", "kmeans-minmax": "kmm"}[quant]
	}
	return fmt.Sprintf("%s-%dx%d-%s-%s-%s-g%d-v%d.coreml", strings.ToLower(name), o.Width, o.Height, q(o.TextFormat, o.TextQuant),
		q(o.DitTextFormat, o.DitQuant), q(o.DitFormat, o.DitQuant), o.Group, ExportVersion)
}

// Prepare makes a Qwen-Image-2.1 ready in s.Dir as hwnpu-chat makes its models ready: it downloads
// the checkpoint files that are missing, converts the export that is missing and returns its
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
	text := filepath.Join(s.Dir, TextDir)
	at := func(f File) string {
		if f.Text {
			return filepath.Join(text, f.Name)
		}
		return filepath.Join(root, f.Name)
	}
	var want []chat.File
	for _, f := range Files {
		want = append(want, chat.File{Repo: s.Repo, Name: f.Name, Path: at(f)})
	}
	chat.Expect(ctx, want)
	for _, f := range Files {
		path := at(f)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return "", err
		}
		if err := chat.Fetch(ctx, s.Repo, f.Name, path, log); err != nil {
			return "", err
		}
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := os.RemoveAll(out); err != nil {
		return "", err
	}
	if from := sharedExport(s.Dir, o); from != "" {
		if err := linkShared(from, out); err != nil {
			return "", err
		}
		o.Only = map[string]bool{"blocks": true, "time": true, "vae": true}
		fmt.Fprintf(log, "the text encoder's and the DiT's text pass's programs from %s\n", filepath.Base(from))
	}
	fmt.Fprintf(log, "converting %s (%dx%d, text %s, DiT %s); this takes a few minutes\n", filepath.Base(out), o.Width, o.Height, o.TextFormat, o.DitFormat)
	t0 := time.Now()
	if err := Convert(root, text, out, o, log); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converted in %.0f s\n", time.Since(t0).Seconds())
	debug.FreeOSMemory()
	return out, os.WriteFile(out+".done", []byte(out), 0o644)
}

// sharedExport is a whole export in dir whose text encoder's and DiT text pass's programs are those
// o asks for (they do not depend on the picture's size), or "".
func sharedExport(dir string, o ConvertOptions) string {
	done, _ := filepath.Glob(filepath.Join(dir, "*.coreml.done"))
	for _, d := range done {
		e := strings.TrimSuffix(d, ".done")
		m, err := ReadProgramManifest(e)
		if err == nil && m.Version == ExportVersion && m.Formats["text"] == o.TextFormat && m.Formats["text_quant"] == o.TextQuant &&
			m.Formats["dit_text"] == o.DitTextFormat && m.Formats["dit_quant"] == o.DitQuant && m.Dit.Group == o.Group &&
			len(m.TextIn) > 0 && len(m.DitText) > 0 {
			return e
		}
	}
	return ""
}

// linkShared hard links text/ and the programs of the text encoder and of the DiT's text pass of
// the export from into out.
func linkShared(from, out string) error {
	m, err := ReadProgramManifest(from)
	if err != nil {
		return err
	}
	for _, sub := range append(append([]string{"text"}, m.TextIn...), m.DitText...) {
		src := filepath.Join(from, sub)
		err := filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			rel, _ := filepath.Rel(from, path)
			dst := filepath.Join(out, rel)
			if d.IsDir() {
				return os.MkdirAll(dst, 0o755)
			}
			return os.Link(path, dst)
		})
		if err != nil {
			return fmt.Errorf("qwenimage: linking %s: %w", sub, err)
		}
	}
	return nil
}
