package sana

import (
	"context"
	"fmt"
	"github.com/shibukawa/hwnpu/internal/cache"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// Setup is what Prepare makes ready.
type Setup struct {
	// Dir holds the checkpoints (a directory per repository, Gemma 2 in TextDir, shared by every
	// Sana) and the exports.
	Dir string
	// Repo is the Hugging Face repository (Efficient-Large-Model/Sana_600M_1024px_diffusers).
	Repo string
	// Name names the export (Sana-600M).
	Name string
	ConvertOptions
	Log io.Writer
}

// TextDir is the directory of Dir that holds Gemma 2 (text_encoder/ and tokenizer/): every Sana
// repository has the same weights and tokenizer (see Files).
const TextDir = "gemma-2-2b-it"

// File is a file of a Sana repository: Text, one of Gemma 2's.
type File struct {
	Name string
	Text bool
}

// Files are the files of a Sana repository Prepare downloads (models.yaml pins their sha256).
// Gemma 2's (Text) go to TextDir, downloaded once from the first Sana's repository: the other
// repositories' copies differ only in config.json's metadata (_name_or_path, transformers_version).
var Files = []File{
	{"transformer/config.json", false},
	{"transformer/diffusion_pytorch_model.safetensors", false},
	{"vae/config.json", false},
	{"vae/diffusion_pytorch_model.safetensors", false},
	{"scheduler/scheduler_config.json", false},
	{"text_encoder/config.json", true},
	{"text_encoder/model.safetensors.index.json", true},
	{"text_encoder/model-00001-of-00002.safetensors", true},
	{"text_encoder/model-00002-of-00002.safetensors", true},
	{"tokenizer/tokenizer.json", true},
}

// ExportName is the directory Prepare writes the export of name and o into: what changes the
// programs is in it.
func ExportName(name string, o ConvertOptions) string {
	o.defaults(name)
	var sizes []string
	for _, s := range o.sizes() {
		if s[0] == s[1] {
			sizes = append(sizes, fmt.Sprint(s[0]))
		} else {
			sizes = append(sizes, fmt.Sprintf("%dx%d", s[0], s[1]))
		}
	}
	size := strings.Join(sizes, "+")
	if len(sizes) == 1 {
		size = fmt.Sprintf("%dx%d", o.Width, o.Height)
	}
	return fmt.Sprintf("%s-%s-%s-%s-g%d-v%d.%s", strings.ToLower(name), size, o.TextFormat, o.DitFormat, o.Group, ExportVersion, exportSuffix(o.Backend))
}

// exportSuffix is the extension of an export's directory for a program format.
func exportSuffix(backend string) string {
	switch backend {
	case llm.BackendIntel, llm.BackendCoreAI:
		return backend
	}
	return "coreml"
}

// Prepare makes a Sana ready in s.Dir, as hwnpu-chat makes its models ready: it downloads the
// checkpoint files that are missing, converts the export that is missing and returns its
// directory. NAME.done, written last, says an export is whole; one cut off is converted again.
// Gemma 2's programs are the same for every Sana: an export of another takes them as hard links.
func Prepare(ctx context.Context, s Setup) (string, error) {
	o := s.ConvertOptions
	o.Only = nil
	o.defaults(s.Name)
	log := s.Log
	if log == nil {
		log = io.Discard
	}
	out := filepath.Join(s.Dir, ExportName(s.Name, o))
	if llm.FetchExport(ctx, out, out+".done") { // here, or from the shared layers
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
	if from := textExport(s.Dir, o); from != "" {
		if err := linkText(from, out); err != nil {
			return "", err
		}
		o.Only = map[string]bool{"caption": true, "time": true, "blocks": true, "final": true, "vae": true}
		fmt.Fprintf(log, "Gemma 2's programs from %s\n", filepath.Base(from))
	}
	fmt.Fprintf(log, "converting %s (%dx%d, text %s, DiT %s); this takes a few minutes\n", filepath.Base(out), o.Width, o.Height, o.TextFormat, o.DitFormat)
	t0 := time.Now()
	if err := Convert(root, text, out, o, log); err != nil {
		return "", err
	}
	fmt.Fprintf(log, "converted in %.0f s\n", time.Since(t0).Seconds())
	debug.FreeOSMemory()
	if err := cache.WriteDone(out+".done", out); err != nil {
		return "", err
	}
	// a Core AI export goes in the shared layers that are written (llm.SharedExports)
	llm.ShareExport(ctx, out)
	return out, nil
}

// textExport is a whole export in dir whose Gemma 2 programs are those o asks for (the program
// format, the text format and ExportVersion), or "".
func textExport(dir string, o ConvertOptions) string {
	done, _ := filepath.Glob(filepath.Join(dir, "*."+exportSuffix(o.Backend)+".done"))
	for _, d := range done {
		e := strings.TrimSuffix(d, ".done")
		m, err := ReadProgramManifest(e)
		if err == nil && m.Version == ExportVersion && m.Formats["text"] == o.TextFormat && len(m.TextParts) > 0 {
			return e
		}
	}
	return ""
}

// linkText hard links the Gemma 2 programs and text/ of the export from into out.
func linkText(from, out string) error {
	m, err := ReadProgramManifest(from)
	if err != nil {
		return err
	}
	for _, sub := range append([]string{"text"}, m.TextParts...) {
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
			return fmt.Errorf("sana: linking %s: %w", sub, err)
		}
	}
	return nil
}
