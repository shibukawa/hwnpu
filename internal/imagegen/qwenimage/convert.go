package qwenimage

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/qwenimage/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// ExportVersion is the layout of the programs Convert writes; an export of another is converted
// again.
const ExportVersion = 1

// SysPrompt and Template are QwenImage21Pipeline's prompt template (prompt_template_t2i): the system
// message's tokens are dropped from the DiT's text rows.
const (
	SysPrompt = "Comprehend and analyze the provided prompt."
	Template  = "<|im_start|>system\n" + SysPrompt + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

// ConvertOptions select what Convert writes.
type ConvertOptions struct {
	// Backend is the program format: coreml (Core ML packages, the default) or coreai (Core AI
	// programs, written from the checkpoint without a Core ML package).
	Backend string
	// Width and Height are the picture's size in pixels, multiples of 32.
	Width, Height int
	// TextFormat and DitFormat are the weight formats of Qwen3-VL-8B and of the DiT's blocks, as
	// hwnpu-convert's -format (int8, lut8, lut6, lut4...); the VAE's are fp16.
	TextFormat, DitFormat string
	// DitTextFormat is the DiT's text pass's (run once a prompt, a group loaded at a time, so not
	// held through the steps; "" DefaultDitTextFormat).
	DitTextFormat string
	// TextQuant and DitQuant are the quantizers of the lut formats (uniform, kmeans,
	// kmeans-minmax; "" the defaults, DefaultTextQuant and DefaultDitQuant).
	TextQuant, DitQuant string
	// Group is the DiT's blocks a program holds; TextGroup the text encoder's layers.
	Group, TextGroup int
	// QChunk is the image attention's queries a chunk (0: mil.QChunk's).
	QChunk int
	// Only, when not empty, writes these parts alone (text, text_in: the text encoder's last group,
	// with the DiT's txt_in; dit_text, blocks, time, vae).
	Only map[string]bool
}

// DefaultFormats are the formats of the text encoder and the DiT: lut8 for Qwen3-VL-8B (run once a
// prompt; 7 GB, read a group at a time), lut4 for the DiT (3.6 GB: its 8 groups are held through
// the steps, under the Neural Engine's compile cache of about 4 GiB; design.md 16.165). Not int8:
// the Neural Engine takes the product of a per-row int8 weight's codes before its scale, and the
// attention sink's row (13000 out of layer 6's down_proj, 3300 at TextScale) passed fp16 there and
// came out NaN; a table's weights are fp16 before the product.
const (
	DefaultTextFormat = "lut8"
	DefaultDitFormat  = "lut4"
	// DefaultDitQuant fits each table to its rows' weights with their extremes kept: one group of
	// the DiT's blocks on the first step's input came out 3.2 % off fp16 (relative RMS), uniform
	// tables 4.9 %, k-means without the extremes 12.6 % (lut6 1.2 %, lut8 0.5 %; design.md 16.165)
	DefaultDitQuant = "kmeans-minmax"
	// DefaultDitTextFormat is the DiT's text pass's: its keys and values are what every step's image
	// attends to. The fox at 512 x 512 against diffusers' float32 (the latents' cosine): text pass
	// and image blocks lut4 0.965, text pass lut8 0.980, and the image blocks lut6 too 0.987 (3.2 s
	// a step against 2.3 s, 5.1 GB held; design.md 16.165)
	DefaultDitTextFormat = "lut8"
	DefaultTextQuant     = "uniform"
	// DefaultCoreAIDitFormat is the DiT's, its text pass's too, on Core AI: one format for both, so
	// that a group's two passes are one program with their weights once (SharedDit). lut6 between
	// the text pass's lut8 and the blocks' lut4 on Core ML (the fox at 512 x 512: text pass lut8
	// and blocks lut6 0.987; design.md 16.165, 16.180)
	DefaultCoreAIDitFormat = "lut6"
)

// SharedDit reports whether o writes each group of the DiT's blocks as one Core AI program with
// two functions, text (the text pass) and blocks (the image blocks), over the same weights written
// once (llm.WriteAIFunctions): on Core AI with the text pass in the DiT's format.
func (o ConvertOptions) SharedDit() bool {
	o.defaults()
	return o.Backend == llm.BackendCoreAI && o.DitTextFormat == o.DitFormat
}

func (o *ConvertOptions) defaults() {
	if o.Backend == "" {
		o.Backend = llm.BackendCoreML
	}
	if o.TextFormat == "" {
		o.TextFormat = DefaultTextFormat
	}
	if o.DitFormat == "" {
		o.DitFormat = DefaultDitFormat
		if o.Backend == llm.BackendCoreAI {
			o.DitFormat = DefaultCoreAIDitFormat
		}
	}
	if o.DitTextFormat == "" {
		o.DitTextFormat = DefaultDitTextFormat
		if o.Backend == llm.BackendCoreAI {
			o.DitTextFormat = o.DitFormat // one copy of the DiT's weights (SharedDit)
		}
	}
	if o.DitQuant == "" {
		o.DitQuant = DefaultDitQuant
	}
	if o.TextQuant == "" {
		o.TextQuant = DefaultTextQuant
	}
	if o.Width == 0 {
		o.Width = 1024
	}
	if o.Height == 0 {
		o.Height = o.Width
	}
	if o.Group <= 0 {
		o.Group = 4
	}
	if o.TextGroup <= 0 {
		// a program within about 0.8 GB of weights (193 MB a layer at 8 bits)
		o.TextGroup = 4
		if o.TextFormat == "fp16" {
			o.TextGroup = 2
		} else if strings.HasPrefix(o.TextFormat, "lut4") || strings.HasPrefix(o.TextFormat, "int4") {
			o.TextGroup = 8
		}
	}
}

// ProgramKind is the kind of manifest.json of an export of Convert.
const ProgramKind = "qwenimage-programs"

// ProgramManifest is manifest.json of an export of Convert: the shapes and the programs (Core ML
// packages, or Core AI programs for Backend coreai) by part, relative to the export's directory.
type ProgramManifest struct {
	Kind    string            `json:"kind"`
	Backend string            `json:"backend"`
	Version int               `json:"version"`
	Formats map[string]string `json:"formats"`
	Dit     DitConfig         `json:"dit"`
	Text    TextConfig        `json:"text"`
	// Template holds {} for the prompt; Drop is its system message's tokens.
	Template string   `json:"template"`
	Drop     int      `json:"drop"`
	TextIn   []string `json:"text_parts"` // Qwen3-VL-8B's groups, in order; the last ends with txt_in
	DitText  []string `json:"dit_text"`   // the DiT's text pass, a group of blocks each
	Blocks   []string `json:"blocks"`     // the DiT's image blocks; the first embeds, the last gives the velocity
	Time     string   `json:"time"`
	Vae      []string `json:"vae"`
}

// ProgramExt is the extension of a program of the format backend: .aimodel for coreai, else
// .mlpackage.
func ProgramExt(backend string) string {
	if backend == llm.BackendCoreAI {
		return ".aimodel"
	}
	return ".mlpackage"
}

// ExportFormat is the program format of the export dir (its manifest's backend), "" for none.
func ExportFormat(dir string) string {
	m, err := ReadProgramManifest(dir)
	if err != nil {
		return ""
	}
	return m.Backend
}

// IsProgramExport reports whether dir holds an export of Convert.
func IsProgramExport(dir string) bool {
	_, err := ReadProgramManifest(dir)
	return err == nil
}

// ReadProgramManifest reads dir/manifest.json of an export of Convert.
func ReadProgramManifest(dir string) (*ProgramManifest, error) {
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m ProgramManifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if m.Kind != ProgramKind {
		return nil, fmt.Errorf("qwenimage: %s is not an export of hwnpu-image convert", dir)
	}
	return &m, nil
}

// Convert writes a Core ML export of the Qwen-Image-2.1 repository root (transformer/, vae/,
// scheduler/, model_index.json; Qwen3-VL-8B's text_encoder/ and processor/ in text, which may be
// root) into out: the programs, text/ (the tokenizer and the embedding table, bf16),
// time.safetensors, model.json and manifest.json.
func Convert(root, text, out string, o ConvertOptions, log io.Writer) error {
	o.defaults()
	if log == nil {
		log = io.Discard
	}
	dc, err := mil.ReadDit(root, o.Width, o.Height)
	if err != nil {
		return err
	}
	tc, err := mil.ReadText(text)
	if err != nil {
		return err
	}
	vc, err := mil.ReadVae(root)
	if err != nil {
		return err
	}
	if tc.Hidden != dc.Hidden {
		return fmt.Errorf("qwenimage: a text encoder of %d for a DiT of %d", tc.Hidden, dc.Hidden)
	}
	tok, err := tokenizer.Load(filepath.Join(text, "processor", "tokenizer.json"))
	if err != nil {
		return err
	}
	drop := len(tok.Encode("<|im_start|>system\n" + SysPrompt + "<|im_end|>\n"))
	w, err := mil.OpenWeights(root, text)
	if err != nil {
		return err
	}
	defer w.Close()
	if err := os.MkdirAll(filepath.Join(out, "text"), 0o755); err != nil {
		return err
	}
	qchunk := o.QChunk
	if qchunk <= 0 {
		qchunk = mil.QChunk(dc, 0)
	}
	textGroups := mil.Groups(tc.Layers, o.TextGroup)
	ditGroups := mil.Groups(dc.Layers, o.Group)
	stages := mil.VaeStages(vc, o.Width, o.Height)
	total := len(textGroups) + 2*len(ditGroups) + len(stages)
	if o.SharedDit() {
		total -= len(ditGroups)
	}
	written := 0
	write := func(kind string, p *mil.Part, format string) (string, error) {
		quant := o.DitQuant
		if kind == "text" || kind == "text_in" {
			quant = o.TextQuant
		}
		name := p.Name + ProgramExt(o.Backend)
		if len(o.Only) > 0 && !o.Only[kind] {
			return name, nil
		}
		written++
		bar := console.NewBar(log, fmt.Sprintf("  [%2d/%d] %s", written, total, name))
		opts := llm.Options{Backend: o.Backend, Format: format}
		if strings.HasPrefix(format, "lut") {
			opts.Quant = quant
		}
		if _, err := llm.WritePart(filepath.Join(out, name), p, w, opts, nil); err != nil {
			bar.Done("failed")
			return "", fmt.Errorf("qwenimage: %s: %w", p.Name, err)
		}
		bar.Done(fmt.Sprintf("%s, %s", console.Duration(bar.Elapsed()), format))
		return name, nil
	}
	m := &ProgramManifest{Kind: ProgramKind, Backend: o.Backend, Version: ExportVersion,
		Formats: map[string]string{"text": o.TextFormat, "dit": o.DitFormat, "dit_text": o.DitTextFormat, "vae": "fp16", "text_quant": o.TextQuant, "dit_quant": o.DitQuant},
		Dit: DitConfig{Width: o.Width, Height: o.Height, Hidden: dc.Hidden, Heads: dc.Heads, HeadDim: dc.HeadDim, Layers: dc.Layers,
			Latent: dc.Latent, Axes: dc.Axes[:], Theta: dc.Theta, Group: o.Group, TextRows: mil.TextRows, VAEScale: mil.VAEScale},
		Text:     TextConfig{Rows: mil.TextLen, Cache: mil.TextLen, Hidden: tc.Hidden, Vocab: tc.Vocab, HeadDim: tc.HeadDim, Theta: tc.Theta},
		Template: Template, Drop: drop, Time: TimeFile}
	for i, gr := range textGroups {
		kind := "text"
		if i == len(textGroups)-1 {
			kind = "text_in" // ends with the DiT's txt_in: of the model, not of Qwen3-VL-8B alone
		}
		name, err := write(kind, mil.TextGroup(tc, dc, gr[0], gr[1], i == len(textGroups)-1), o.TextFormat)
		if err != nil {
			return err
		}
		m.TextIn = append(m.TextIn, name)
	}
	if o.SharedDit() {
		for i, gr := range ditGroups {
			name := fmt.Sprintf("dit_%d", gr[0]) + ProgramExt(o.Backend)
			m.DitText = append(m.DitText, name+"#text")
			m.Blocks = append(m.Blocks, name+"#blocks")
			if len(o.Only) > 0 && !o.Only["dit_text"] && !o.Only["blocks"] {
				continue
			}
			written++
			bar := console.NewBar(log, fmt.Sprintf("  [%2d/%d] %s", written, total, name))
			opts := llm.Options{Backend: o.Backend, Format: o.DitFormat}
			if strings.HasPrefix(o.DitFormat, "lut") {
				opts.Quant = o.DitQuant
			}
			_, err := llm.WriteAIFunctions(filepath.Join(out, name), w, []llm.AIFunction{
				{Name: "text", Model: mil.TextBlocks(dc, gr[0], gr[1]), Options: opts},
				{Name: "blocks", Model: mil.Blocks(dc, gr[0], gr[1], qchunk, i == len(ditGroups)-1), Options: opts}})
			if err != nil {
				bar.Done("failed")
				return fmt.Errorf("qwenimage: %s: %w", name, err)
			}
			bar.Done(fmt.Sprintf("%s, %s, text pass and blocks", console.Duration(bar.Elapsed()), o.DitFormat))
		}
	} else {
		for _, gr := range ditGroups {
			name, err := write("dit_text", mil.TextBlocks(dc, gr[0], gr[1]), o.DitTextFormat)
			if err != nil {
				return err
			}
			m.DitText = append(m.DitText, name)
		}
		for i, gr := range ditGroups {
			name, err := write("blocks", mil.Blocks(dc, gr[0], gr[1], qchunk, i == len(ditGroups)-1), o.DitFormat)
			if err != nil {
				return err
			}
			m.Blocks = append(m.Blocks, name)
		}
	}
	if len(o.Only) == 0 || o.Only["time"] {
		if err := writeTimeFile(filepath.Join(out, TimeFile), w); err != nil {
			return err
		}
	}
	for i := range stages {
		name, err := write("vae", mil.VaePart(vc, o.Width, o.Height, i, 0), "fp16")
		if err != nil {
			return err
		}
		m.Vae = append(m.Vae, name)
	}
	if len(o.Only) == 0 || o.Only["text"] {
		if err := writeTextFiles(text, out, w, tc); err != nil {
			return err
		}
	}
	if err := writeModel(root, out, o.Width, o.Height); err != nil {
		return err
	}
	b, err := json.MarshalIndent(m, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, "manifest.json"), append(b, '\n'), 0o644)
}

// writeModel writes model.json as tools/amd/iron_qwenimage.py does: the checkpoint's sigmas and
// its scheduler's settings.
func writeModel(root, out string, width, height int) error {
	var mi struct {
		Sigmas []float64 `json:"sample_sigmas"`
	}
	b, err := os.ReadFile(filepath.Join(root, "model_index.json"))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, &mi); err != nil {
		return err
	}
	sb, err := os.ReadFile(filepath.Join(root, "scheduler", "scheduler_config.json"))
	if err != nil {
		return err
	}
	m := map[string]any{"name": filepath.Base(root), "sample_sigmas": mi.Sigmas, "scheduler": json.RawMessage(sb),
		"width": width, "height": height}
	b, err = json.MarshalIndent(m, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, "model.json"), append(b, '\n'), 0o644)
}

// writeTextFiles writes text/: the tokenizer and Qwen3-VL-8B's embedding table (bf16) the host
// reads the prompt's rows from.
func writeTextFiles(text, out string, w *mil.Weights, tc mil.Text) error {
	b, err := os.ReadFile(filepath.Join(text, "processor", "tokenizer.json"))
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "text", "tokenizer.json"), b, 0o644); err != nil {
		return err
	}
	f, err := os.Create(filepath.Join(out, "text", "embed.bin"))
	if err != nil {
		return err
	}
	const chunk = 8192
	buf := make([]byte, 2*chunk*tc.Hidden)
	for r := 0; r < tc.Vocab; r += chunk {
		e := min(r+chunk, tc.Vocab)
		v, err := w.Rows("text.embed_tokens.weight", r, e)
		if err != nil {
			f.Close()
			return err
		}
		for i, x := range v {
			binary.LittleEndian.PutUint16(buf[2*i:], bf16(x))
		}
		if _, err := f.Write(buf[:2*len(v)]); err != nil {
			f.Close()
			return err
		}
	}
	return f.Close()
}
