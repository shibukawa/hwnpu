package anima

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"github.com/shibukawa/hwnpu/internal/chat"
	"io"
	"os"
	"path/filepath"
	"slices"

	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// ConvertOptions select what Convert writes.
type ConvertOptions struct {
	// Backend is "coreml" (Core ML packages for Apple's Neural Engine), "coreai" (Core AI programs
	// for it, macOS 27) or "intel" (OpenVINO IR for Intel's NPU).
	Backend string
	// Width and Height are the image's size in pixels, multiples of 16.
	Width, Height int
	// Dit is the diffusion model's file in diffusion_models/ (anima-turbo-v1.1.safetensors).
	Dit string
	// TextFormat and DitFormat are the weight formats of the text encoder and the DiT, as
	// hwnpu-dev convert's -format: fp16, lut4/6/8 (Core ML), int8, int4... (Intel). The VAE's are fp16.
	TextFormat, DitFormat string
	// Group is the DiT's blocks a program holds.
	Group int
	// QChunk is the queries the attention of the DiT and the VAE scores at a time.
	QChunk int
	// Only, when not empty, writes these parts alone (text, time, blocks, final, vae).
	Only map[string]bool
	// Sizes are the picture sizes (width, height) a Core AI export holds: each part that depends
	// on the size one program with a function each (SizeFunction; llm.WriteAIFunctions, the
	// weights once), the manifest's Sizes. Empty: Width x Height alone. Width x Height is one of
	// them (the manifest's own). The other backends take one size.
	Sizes [][2]int
}

// sizes is what o's export holds.
func (o ConvertOptions) sizes() [][2]int {
	if len(o.Sizes) == 0 {
		return [][2]int{{o.Width, o.Height}}
	}
	return o.Sizes
}

// SizeFunction is the function of a program of several sizes that runs at w x h.
func SizeFunction(w, h int) string { return fmt.Sprintf("s%dx%d", w, h) }

// Checkpoint is where Convert reads Anima from: Models holds diffusion_models/ and
// text_encoders/ (split_files/ of RepoSplit), Diffusers vae/, tokenizer/ and t5_tokenizer/ (of
// RepoDiffusers).
type Checkpoint struct{ Models, Diffusers string }

// LayoutCheckpoint is the Checkpoint of a folder in the layout of tools/amd/iron_anima.py:
// diffusion_models/, text_encoders/, diffusers/.
func LayoutCheckpoint(root string) Checkpoint {
	return Checkpoint{Models: root, Diffusers: filepath.Join(root, "diffusers")}
}

// RepoCheckpoint is the Checkpoint of the repositories' files here (chat.RepoDir), which
// Prepare downloads.
func RepoCheckpoint() Checkpoint {
	return Checkpoint{Models: filepath.Join(chat.RepoDir(RepoSplit), "split_files"), Diffusers: chat.RepoDir(RepoDiffusers)}
}

// Convert writes an export of Anima from ck into out: the programs, the tokenizers, the
// embedding tables (text/, as the IRON export's) and manifest.json.
func Convert(ck Checkpoint, out string, o ConvertOptions, log io.Writer) error {
	if o.Backend != llm.BackendCoreML && o.Backend != llm.BackendCoreAI && o.Backend != llm.BackendIntel {
		return fmt.Errorf("anima: backend %q is not coreml, coreai or intel", o.Backend)
	}
	sizes := o.sizes()
	if len(sizes) > 1 && o.Backend != llm.BackendCoreAI {
		return fmt.Errorf("anima: an export of several sizes is Core AI's (%s takes one)", o.Backend)
	}
	if !slices.Contains(sizes, [2]int{o.Width, o.Height}) {
		return fmt.Errorf("anima: %dx%d is not one of the sizes %v", o.Width, o.Height, sizes)
	}
	for _, sz := range sizes {
		if sz[0]%16 != 0 || sz[1]%16 != 0 || sz[0] <= 0 || sz[1] <= 0 {
			return fmt.Errorf("anima: %dx%d is no whole 16-pixel patches", sz[0], sz[1])
		}
		if len(mil.VaeStages(sz[0], sz[1])) != len(mil.VaeStages(o.Width, o.Height)) {
			return fmt.Errorf("anima: the VAE's stages at %dx%d are not as many as at %dx%d", sz[0], sz[1], o.Width, o.Height)
		}
	}
	if o.Group <= 0 {
		o.Group = DefaultGroup(o.Backend, o.Width, o.Height)
	}
	if mil.DitLayers%o.Group != 0 {
		return fmt.Errorf("anima: %d blocks are no whole groups of %d", mil.DitLayers, o.Group)
	}
	if o.Dit == "" {
		o.Dit = "anima-turbo-v1.1.safetensors"
	}
	w, err := mil.OpenWeights(ck.Models, ck.Diffusers, o.Dit)
	if err != nil {
		return err
	}
	defer w.Close()
	if err := os.MkdirAll(filepath.Join(out, "text"), 0o755); err != nil {
		return err
	}
	ext := map[string]string{llm.BackendCoreML: ".mlpackage", llm.BackendCoreAI: ".aimodel", llm.BackendIntel: ".xml"}[o.Backend]
	var vcfg struct {
		Mean []float64 `json:"latents_mean"`
		Std  []float64 `json:"latents_std"`
	}
	raw, err := os.ReadFile(filepath.Join(ck.Diffusers, "vae", "config.json"))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, &vcfg); err != nil || len(vcfg.Mean) != 16 || len(vcfg.Std) != 16 {
		return fmt.Errorf("anima: vae/config.json has no 16 latents_mean and latents_std")
	}
	// the programs are written side by side once all are listed (jobs.Run)
	jobs := console.NewJobs(log)
	m := &ProgramManifest{Backend: o.Backend, Width: o.Width, Height: o.Height, Dit: o.Dit,
		Formats: map[string]string{"text": o.TextFormat, "dit": o.DitFormat, "vae": "fp16"}}
	write := func(kind string, p *mil.Part, format string) (string, error) {
		name := p.Name + ext
		if len(o.Only) > 0 && !o.Only[kind] {
			return name, nil
		}
		jobs.Add(name, func() (string, error) {
			opts := llm.Options{Backend: o.Backend, Format: format}
			if _, err := llm.WritePart(filepath.Join(out, name), p, w, opts, nil); err != nil {
				return "", fmt.Errorf("anima: %s: %w", p.Name, err)
			}
			return format, nil
		})
		return name, nil
	}
	if m.Text, err = write("text", mil.Text(), o.TextFormat); err != nil {
		return err
	}
	if m.Time, err = write("time", mil.Time(), o.DitFormat); err != nil {
		return err
	}
	m.TimeRows = mil.TimeRows
	// writeSized writes the part of each size (part(w, h)): for one size as write does, else one
	// program with a function each (SizeFunction), the weights once. The names are one a size.
	writeSized := func(kind string, part func(w, h int) *mil.Part, format string) ([]string, error) {
		if len(sizes) == 1 {
			name, err := write(kind, part(sizes[0][0], sizes[0][1]), format)
			return []string{name}, err
		}
		name := part(o.Width, o.Height).Name + ext
		var names []string
		var fns []llm.AIFunction
		for _, sz := range sizes {
			f := SizeFunction(sz[0], sz[1])
			names = append(names, name+"#"+f)
			fns = append(fns, llm.AIFunction{Name: f, Model: part(sz[0], sz[1]), Options: llm.Options{Backend: o.Backend, Format: format}})
		}
		if len(o.Only) > 0 && !o.Only[kind] {
			return names, nil
		}
		jobs.Add(name, func() (string, error) {
			if _, err := llm.WriteAIFunctions(filepath.Join(out, name), w, fns); err != nil {
				return "", fmt.Errorf("anima: %s: %w", name, err)
			}
			return fmt.Sprintf("%s, %d sizes", format, len(sizes)), nil
		})
		return names, nil
	}
	sized := make([]SizeParts, len(sizes))
	for i, sz := range sizes {
		sized[i] = SizeParts{Width: sz[0], Height: sz[1]}
		if q := mil.DitQChunk(sz[0], sz[1], o.QChunk); q != o.QChunk {
			fmt.Fprintf(log, "the DiT's attention at %dx%d scores %d queries at a time (%d would be over 64 MiB of scores)\n", sz[0], sz[1], q, o.QChunk)
		}
	}
	for first := 0; first < mil.DitLayers; first += o.Group {
		names, err := writeSized("blocks", func(w, h int) *mil.Part {
			return mil.Blocks(w, h, first, o.Group, mil.DitQChunk(w, h, o.QChunk))
		}, o.DitFormat)
		if err != nil {
			return err
		}
		for i := range sized {
			sized[i].Blocks = append(sized[i].Blocks, names[i])
		}
	}
	names, err := writeSized("final", func(w, h int) *mil.Part { return mil.Final(w, h) }, o.DitFormat)
	if err != nil {
		return err
	}
	for i := range sized {
		sized[i].Final = names[i]
	}
	for st := range mil.VaeStages(o.Width, o.Height) {
		names, err := writeSized("vae", func(w, h int) *mil.Part { return mil.Vae(w, h, st, vcfg.Mean, vcfg.Std, o.QChunk) }, "fp16")
		if err != nil {
			return err
		}
		for i := range sized {
			sized[i].Vae = append(sized[i].Vae, names[i])
		}
	}
	for _, sp := range sized {
		if sp.Width == o.Width && sp.Height == o.Height {
			m.Blocks, m.Final, m.Vae = sp.Blocks, sp.Final, sp.Vae
		}
	}
	if len(sized) > 1 {
		m.Sizes = sized
	}
	// a job takes up to 0.4 GB at 20 at a time (Anima Turbo at 1024 x 1024: 6.7 GB in all)
	if err := jobs.Run(1 << 30); err != nil {
		return err
	}
	if len(o.Only) == 0 || o.Only["text"] {
		if err := writeTextFiles(ck, out, w); err != nil {
			return err
		}
	}
	return WriteProgramManifest(out, m)
}

// writeTextFiles writes text/: the tokenizers, the embedding tables (bf16) the host reads the
// prompt's rows from and the empty prompt's conditioning (EmptyCondFile).
func writeTextFiles(ck Checkpoint, out string, w *mil.Weights) error {
	for _, c := range []struct{ src, dst string }{
		{filepath.Join(ck.Diffusers, "tokenizer", "tokenizer.json"), "qwen_tokenizer.json"},
		{filepath.Join(ck.Diffusers, "t5_tokenizer", "tokenizer.json"), "t5_tokenizer.json"},
	} {
		b, err := os.ReadFile(c.src)
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(out, "text", c.dst), b, 0o644); err != nil {
			return err
		}
	}
	for _, t := range []struct{ name, file string }{
		{"qwen.embed_tokens.weight", "qwen_embed.bin"},
		{"dit.llm_adapter.embed.weight", "t5_embed.bin"},
	} {
		v, _, err := w.Get(t.name)
		if err != nil {
			return err
		}
		b := make([]byte, 2*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint16(b[2*i:], bf16(x))
		}
		if err := os.WriteFile(filepath.Join(out, "text", t.file), b, 0o644); err != nil {
			return err
		}
	}
	row, err := EmptyCond(w)
	if err != nil {
		return err
	}
	return WriteEmptyCond(filepath.Join(out, "text"), row)
}
