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
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// Setup is what Prepare makes ready.
type Setup struct {
	// Dir holds the exports; the checkpoint is the repository's files (chat.RepoDir).
	Dir string
	// Repo is the Hugging Face repository (Efficient-Large-Model/Sana_600M_1024px_diffusers).
	Repo string
	// Name names the export (Sana-600M).
	Name string
	ConvertOptions
	Log io.Writer
}

// File is a file of a Sana repository: Text, one of Gemma 2's.
type File struct {
	Name string
	Text bool
}

// Files are the files of a Sana repository Prepare downloads (models.yaml pins their sha256).
// Gemma 2's (Text) are the same in every Sana repository but config.json's metadata
// (_name_or_path, transformers_version): the weights are linked from another's (pinned).
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

// exportNameAt is ExportName for the layout version.
func exportNameAt(name string, o ConvertOptions, version int) string {
	return strings.Replace(ExportName(name, o), fmt.Sprintf("-v%d.", ExportVersion), fmt.Sprintf("-v%d.", version), 1)
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
	// what is converted: all of it, less what another export gives as hard links (the same
	// bytes, so Core AI's cache holds them compiled): Gemma 2's programs from an export of this
	// layout, the others from this Sana's export of an older layout whose DiT is this one's
	only := map[string]bool{"text": true, "caption": true, "time": true, "blocks": true, "final": true, "vae": true}
	textFrom, ditFrom := textExport(s.Dir, o), ditExport(s.Dir, s.Name, o)
	if textFrom != "" {
		delete(only, "text")
	}
	if ditFrom != "" {
		only = map[string]bool{"text": only["text"]}
	}
	// the repository's files (chat.RepoDir) that are read: the configurations, and the weights of
	// what is converted. Gemma 2's are the same in every Sana's (pinned: linked from another's
	// when it is here)
	root := chat.RepoDir(s.Repo)
	text := root
	var want []chat.File
	for _, f := range Files {
		if needs(f, only) {
			want = append(want, chat.File{Repo: s.Repo, Name: f.Name})
		}
	}
	if err := chat.FetchFiles(ctx, want, log); err != nil {
		return "", err
	}
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := os.RemoveAll(out); err != nil {
		return "", err
	}
	if textFrom != "" {
		if err := linkText(textFrom, out); err != nil {
			return "", err
		}
		fmt.Fprintf(log, "Gemma 2's programs from %s\n", filepath.Base(textFrom))
	}
	if ditFrom != "" {
		if err := linkDit(ditFrom, out); err != nil {
			return "", err
		}
		fmt.Fprintf(log, "the DiT's and the VAE's programs from %s\n", filepath.Base(ditFrom))
	}
	if len(only) < 6 {
		o.Only = only // nothing of it true: the manifest alone
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

// needs reports whether Prepare reads the file f of the repository when it converts only: a
// configuration always, weights (and Gemma 2's tokenizer and index) when their part is converted.
func needs(f File, only map[string]bool) bool {
	switch {
	case f.Text:
		return only["text"] || f.Name == "text_encoder/config.json"
	case strings.HasSuffix(f.Name, ".safetensors"):
		return only["blocks"]
	}
	return true
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

// ditExport is a whole export in dir of the Sana name and o's options in an older layout whose
// programs but Gemma 2's are this layout's (DitVersion on), or "".
func ditExport(dir, name string, o ConvertOptions) string {
	for v := ExportVersion - 1; v >= DitVersion; v-- {
		e := filepath.Join(dir, exportNameAt(name, o, v))
		if _, err := os.Stat(e + ".done"); err != nil {
			continue
		}
		if m, err := ReadProgramManifest(e); err == nil && m.Version == v && m.Formats["dit"] == o.DitFormat {
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
	return linkParts(from, out, append([]string{"text"}, m.TextParts...))
}

// linkDit hard links the programs of the export from but Gemma 2's into out: the caption's, the
// time embedding's weights, the DiT's and the VAE's, of every size.
func linkDit(from, out string) error {
	m, err := ReadProgramManifest(from)
	if err != nil {
		return err
	}
	names := slices.Concat([]string{m.Caption, m.Time, m.Final}, m.Blocks, m.Vae)
	for _, sp := range m.Sizes {
		names = slices.Concat(names, []string{sp.Final}, sp.Blocks, sp.Vae)
	}
	return linkParts(from, out, names)
}

// linkParts hard links the files and directories names of from into out, each once: a name is a
// path in the export, a program's with its function after "#" or without.
func linkParts(from, out string, names []string) error {
	seen := map[string]bool{}
	for _, name := range names {
		sub, _, _ := strings.Cut(name, "#")
		if sub == "" || seen[sub] {
			continue
		}
		seen[sub] = true
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
