package anima

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

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
}

// Convert writes an export of Anima from root (the layout of tools/amd/iron_anima.py:
// diffusion_models/, text_encoders/, diffusers/) into out: the programs, the tokenizers, the
// embedding tables (text/, as the IRON export's) and manifest.json.
func Convert(root, out string, o ConvertOptions, log io.Writer) error {
	if o.Width%16 != 0 || o.Height%16 != 0 || o.Width <= 0 || o.Height <= 0 {
		return fmt.Errorf("anima: %dx%d is no whole 16-pixel patches", o.Width, o.Height)
	}
	if o.Backend != llm.BackendCoreML && o.Backend != llm.BackendCoreAI && o.Backend != llm.BackendIntel {
		return fmt.Errorf("anima: backend %q is not coreml, coreai or intel", o.Backend)
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
	w, err := mil.OpenWeights(root, o.Dit)
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
	raw, err := os.ReadFile(filepath.Join(root, "diffusers", "vae", "config.json"))
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
	qchunk := mil.DitQChunk(o.Width, o.Height, o.QChunk)
	if qchunk != o.QChunk {
		fmt.Fprintf(log, "the DiT's attention scores %d queries at a time (%d would be over 64 MiB of scores)\n", qchunk, o.QChunk)
	}
	for first := 0; first < mil.DitLayers; first += o.Group {
		name, err := write("blocks", mil.Blocks(o.Width, o.Height, first, o.Group, qchunk), o.DitFormat)
		if err != nil {
			return err
		}
		m.Blocks = append(m.Blocks, name)
	}
	if m.Final, err = write("final", mil.Final(o.Width, o.Height), o.DitFormat); err != nil {
		return err
	}
	for i := range mil.VaeStages(o.Width, o.Height) {
		name, err := write("vae", mil.Vae(o.Width, o.Height, i, vcfg.Mean, vcfg.Std, o.QChunk), "fp16")
		if err != nil {
			return err
		}
		m.Vae = append(m.Vae, name)
	}
	// a job takes up to 0.4 GB at 20 at a time (Anima Turbo at 1024 x 1024: 6.7 GB in all)
	if err := jobs.Run(1 << 30); err != nil {
		return err
	}
	if len(o.Only) == 0 || o.Only["text"] {
		if err := writeTextFiles(root, out, w); err != nil {
			return err
		}
	}
	return WriteProgramManifest(out, m)
}

// writeTextFiles writes text/: the tokenizers, the embedding tables (bf16) the host reads the
// prompt's rows from and the empty prompt's conditioning (EmptyCondFile).
func writeTextFiles(root, out string, w *mil.Weights) error {
	for _, c := range []struct{ src, dst string }{
		{filepath.Join(root, "diffusers", "tokenizer", "tokenizer.json"), "qwen_tokenizer.json"},
		{filepath.Join(root, "diffusers", "t5_tokenizer", "tokenizer.json"), "t5_tokenizer.json"},
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
