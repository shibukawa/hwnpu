package sana

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
	// Dir holds the checkpoints (a directory per repository, Gemma 2 in TextDir, shared by every
	// Sana) and the exports.
	Dir string
	// Repo is the Hugging Face repository (Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers).
	Repo string
	// Name names the export (Sana-Sprint-0.6B).
	Name string
	ConvertOptions
	Log io.Writer
}

// TextDir is the directory of Dir that holds Gemma 2 (text_encoder/ and tokenizer/): every Sana
// repository has the same files.
const TextDir = "gemma-2-2b-it"

// ExportName is the directory Prepare writes the export of name and o into: what changes the
// programs is in it.
func ExportName(name string, o ConvertOptions) string {
	o.defaults()
	return fmt.Sprintf("%s-%dx%d-%s-%s-g%d-v%d.coreml", strings.ToLower(name), o.Width, o.Height, o.TextFormat, o.DitFormat, o.Group, ExportVersion)
}

// Prepare makes a Sana ready in s.Dir, as hwnpu-chat makes its models ready: it downloads the
// checkpoint files that are missing, converts the export that is missing and returns its
// directory. NAME.done, written last, says an export is whole; one cut off is converted again.
// Gemma 2's programs are the same for every Sana: an export of another takes them as hard links.
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
	for _, f := range []struct{ dir, name string }{
		{root, "transformer/config.json"},
		{root, "transformer/diffusion_pytorch_model.safetensors"},
		{root, "vae/config.json"},
		{root, "vae/diffusion_pytorch_model.safetensors"},
		{root, "scheduler/scheduler_config.json"},
		{text, "text_encoder/config.json"},
		{text, "text_encoder/model.safetensors.index.json"},
		{text, "text_encoder/model-00001-of-00002.safetensors"},
		{text, "text_encoder/model-00002-of-00002.safetensors"},
		{text, "tokenizer/tokenizer.json"},
	} {
		path := filepath.Join(f.dir, f.name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return "", err
		}
		if err := chat.Fetch(ctx, s.Repo, f.name, path, log); err != nil {
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
	return out, os.WriteFile(out+".done", []byte(out), 0o644)
}

// textExport is a whole export in dir whose Gemma 2 programs are those o asks for (the text
// format and ExportVersion), or "".
func textExport(dir string, o ConvertOptions) string {
	done, _ := filepath.Glob(filepath.Join(dir, "*.coreml.done"))
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
