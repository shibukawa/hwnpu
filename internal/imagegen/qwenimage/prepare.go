package qwenimage

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime/debug"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/qwenimage/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
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

// textWeights is the text encoder's checkpoint: one file (Qwen-Image-2.1-Turbo) or the shards of
// its index (Qwen-Image-2.1). TextDir holds either, whichever repository put it there.
const textWeights = "text_encoder/model.safetensors"

// fetchSharded puts the checkpoint name of repo at path, or when the repository has it in
// several files, its index (path.index.json) and the files it names beside it. Shards already
// there (of another repository's download) are kept.
func fetchSharded(ctx context.Context, repo, name, path string, log io.Writer) error {
	index := path + ".index.json"
	if _, err := os.Stat(index); err != nil {
		err := chat.Fetch(ctx, repo, name, path, log)
		if err == nil || !chat.NotFound(err) {
			return err
		}
		if err := chat.Fetch(ctx, repo, name+".index.json", index, log); err != nil {
			return err
		}
	}
	files, err := mil.Shards(filepath.Dir(path), strings.TrimSuffix(filepath.Base(path), ".safetensors"))
	if err != nil {
		return err
	}
	if len(files) == 1 && files[0] == path {
		return nil // the one file, there
	}
	dir := filepath.Dir(name)
	var want []chat.File
	for _, f := range files {
		want = append(want, chat.File{Repo: repo, Name: dir + "/" + filepath.Base(f), Path: f})
	}
	chat.Expect(ctx, want)
	for _, f := range want {
		if err := chat.Fetch(ctx, repo, f.Name, f.Path, log); err != nil {
			return err
		}
	}
	return nil
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
	return fmt.Sprintf("%s-%dx%d-%s-%s-%s-g%d-v%d.%s", strings.ToLower(name), o.Width, o.Height, q(o.TextFormat, o.TextQuant),
		q(o.DitTextFormat, o.DitQuant), q(o.DitFormat, o.DitQuant), o.Group, ExportVersion, o.Backend)
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
		fetch := chat.Fetch
		if f.Name == textWeights {
			fetch = fetchSharded
		}
		if err := fetch(ctx, s.Repo, f.Name, path, log); err != nil {
			return "", err
		}
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := os.RemoveAll(out); err != nil {
		return "", err
	}
	if from, same := sharedExport(s.Dir, s.Name, o); from != "" {
		if err := linkShared(from, out, same, same && !o.SharedDit()); err != nil {
			return "", err
		}
		if same && !o.SharedDit() {
			o.Only = map[string]bool{"blocks": true, "time": true, "vae": true}
			fmt.Fprintf(log, "the text encoder's and the DiT's text pass's programs from %s\n", filepath.Base(from))
		} else {
			o.Only = map[string]bool{"text_in": true, "dit_text": true, "blocks": true, "time": true, "vae": true}
			if same { // its DiT's text pass is one program with its blocks (SharedDit): written again
				delete(o.Only, "text_in")
			}
			fmt.Fprintf(log, "the text encoder's programs from %s\n", filepath.Base(from))
		}
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

// sharedExport is a whole export in dir whose text encoder's programs are those o asks for (they do
// not depend on the picture's size), or "". same says it is of the same model, name, so that its
// DiT's text pass and the text encoder's last group (which ends with the DiT's txt_in) are this
// one's too; one of another model (Qwen-Image-2.1 and its Turbo hold the same Qwen3-VL-8B) shares
// the text encoder's other groups only. One of the same model is taken first.
func sharedExport(dir, name string, o ConvertOptions) (from string, same bool) {
	done, _ := filepath.Glob(filepath.Join(dir, "*."+o.Backend+".done"))
	for _, d := range done {
		e := strings.TrimSuffix(d, ".done")
		m, err := ReadProgramManifest(e)
		if err != nil || m.Version != ExportVersion || m.Formats["text"] != o.TextFormat || m.Formats["text_quant"] != o.TextQuant ||
			len(m.TextIn) == 0 || len(m.DitText) == 0 {
			continue
		}
		if exportModel(e) == strings.ToLower(name) && m.Formats["dit_text"] == o.DitTextFormat && m.Formats["dit_quant"] == o.DitQuant &&
			m.Dit.Group == o.Group {
			return e, true
		}
		if from == "" {
			from = e
		}
	}
	return from, false
}

var exportSize = regexp.MustCompile(`-[0-9]+x[0-9]+-`)

// exportModel is the model an export directory of ExportName is of (qwen-image-2.1-turbo).
func exportModel(dir string) string {
	b := filepath.Base(dir)
	if i := exportSize.FindStringIndex(b); i != nil {
		return b[:i[0]]
	}
	return ""
}

// linkShared hard links text/ and the text encoder's programs of the export from into out; with
// last, its last group too (an export of the same model, sharedExport); with ditText, the DiT's
// text pass too (not when it is one program with the blocks, SharedDit).
func linkShared(from, out string, last, ditText bool) error {
	m, err := ReadProgramManifest(from)
	if err != nil {
		return err
	}
	subs := append([]string{"text"}, m.TextIn...)
	if !last {
		subs = subs[:len(subs)-1]
	}
	if ditText {
		for _, d := range m.DitText {
			if p, fn := llm.SplitFunction(d); fn == "" {
				subs = append(subs, p)
			} // one program with its blocks (SharedDit): not this export's
		}
	}
	for _, sub := range subs {
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
