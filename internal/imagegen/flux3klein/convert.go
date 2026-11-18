// Package flux3klein runs FLUX.3 [klein] (black-forest-labs/flux-3-image-klein, text to image) on
// Apple's Neural Engine: Qwen3-VL reads the prompt (the hidden states after layers 4, 8, ..., 32),
// the 5.5B DiT (each stream's mode blocks, then single blocks over [text, image]) takes 16
// flow-map steps without guidance, and the FLUX.2 VAE decodes the latents. Convert writes the
// programs (internal/imagegen/flux3klein/mil) as a Core AI export; Pipeline runs them.
package flux3klein

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux3klein/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// ExportVersion is the layout of the programs Convert writes; an export of another is converted
// again.
const ExportVersion = 1

// ConvertOptions select what Convert writes.
type ConvertOptions struct {
	// Sizes are the picture sizes (width, height) the export holds (default 512 x 512 and 1024 x
	// 1024), and TextRows the text lengths (multiples of mil.TextPad: default 80 and 160, a
	// prompt of up to 160 tokens in the chat template). The parts that depend on them are
	// programs of a function each (llm.WriteAIFunctions: the weights once).
	Sizes    [][2]int
	TextRows []int
	// TextFormat and DitFormat are the weight formats of Qwen3-VL and of the DiT, as
	// hwnpu-convert's -format (default lut8 both). The VAE's and context_embedder's are fp16.
	TextFormat, DitFormat string
	// Stats, when set, is a safetensors file of the DiT's inputs' mean squares (FitCodes): the
	// DiT's lut8 tables are then fitted to the activations (weighted Lloyd from the uniform
	// levels), else spaced evenly (uniform), which loses fine detail (a portrait's hair: latent
	// cosine 0.964 against 0.998 at 512 x 512, docs/design.md).
	Stats string
	// TextGroup is Qwen3-VL's layers a program holds at most (a group ends at each hidden state
	// the DiT reads); SingleGroup the single blocks.
	TextGroup, SingleGroup int
}

// DefaultSizes and DefaultTextRows are what an export holds by default.
var (
	DefaultSizes    = [][2]int{{512, 512}, {1024, 1024}}
	DefaultTextRows = []int{80, 160}
)

func (o *ConvertOptions) defaults() {
	if len(o.Sizes) == 0 {
		o.Sizes = DefaultSizes
	}
	if len(o.TextRows) == 0 {
		o.TextRows = DefaultTextRows
	}
	if o.TextFormat == "" {
		o.TextFormat = "lut8" // Qwen3-VL's attention sinks: NaN at int8 on the Neural Engine (design.md 16.165)
	}
	if o.DitFormat == "" {
		o.DitFormat = "lut8" // the Neural Engine reads lutN fastest; per-row int8 overflows on large outputs
	}
	if o.TextGroup <= 0 {
		o.TextGroup = 4 // 0.4 GB at 8 bits (101 MB a layer)
	}
	if o.SingleGroup <= 0 {
		o.SingleGroup = 5 // 0.6 GB at 8 bits (123 MB a block)
	}
}

// Files of an export: the tokenizer and Qwen3-VL's embedding rows (bf16).
const (
	TokenizerFile = "text/tokenizer.json"
	EmbedFile     = "text/embed.bin"
	CodesDir      = "codes"
)

// fnName names a function of a part that depends on the picture's size (w > 0) and on the text's
// rows (rows > 0).
func fnName(w, h, rows int) string {
	var s []string
	if w > 0 {
		s = append(s, fmt.Sprintf("s%dx%d", w, h))
	}
	if rows > 0 {
		s = append(s, fmt.Sprintf("t%d", rows))
	}
	return strings.Join(s, "_")
}

// Convert writes a Core AI export of the FLUX.3 klein repository root (transformer/,
// text_encoder/, vae/) into out: the programs, text/, TimeFile and manifest.json.
func Convert(root, out string, o ConvertOptions, log io.Writer) (err error) {
	o.defaults()
	if log == nil {
		log = io.Discard
	}
	ditAt := func(sz [2]int, rows int) (mil.Dit, error) { return mil.ReadDit(root, sz[0], sz[1], rows) }
	d0, err := ditAt(o.Sizes[0], o.TextRows[0])
	if err != nil {
		return err
	}
	for _, sz := range o.Sizes {
		for _, r := range o.TextRows {
			if _, err := ditAt(sz, r); err != nil {
				return err
			}
		}
	}
	tc, err := mil.ReadText(root)
	if err != nil {
		return err
	}
	vc, err := mil.ReadVae(root)
	if err != nil {
		return err
	}
	w, err := mil.OpenWeights(root, d0)
	if err != nil {
		return err
	}
	defer w.Close()
	if err := os.MkdirAll(filepath.Join(out, "text"), 0o755); err != nil {
		return err
	}
	codes := ""
	if o.Stats != "" && strings.HasPrefix(o.DitFormat, "lut") {
		codes = filepath.Join(out, CodesDir)
		if o.DitFormat != "lut8" {
			return fmt.Errorf("flux3klein: tables fitted to the activations are lut8's; the DiT is %s", o.DitFormat)
		}
		if err := FitCodes(w, d0, o.Stats, codes, log); err != nil {
			return err
		}
	}
	jobs := console.NewJobs(log)
	type fn struct {
		name string
		part *mil.Part
	}
	// program writes the functions fns as one program (or a plain program for one) named after
	// the first's part; it returns each function's name as the runner opens it.
	program := func(fns []fn, format string, dit bool) map[string]string {
		file := fns[0].part.Name + ".aimodel"
		names := map[string]string{}
		opts := llm.Options{Backend: llm.BackendCoreAI, Format: format}
		if dit && codes != "" {
			opts.Codes = codes
		}
		if len(fns) == 1 {
			names[fns[0].name] = file
			p := fns[0].part
			jobs.Add(file, func() (string, error) {
				if _, err := llm.WritePart(filepath.Join(out, file), p, w, opts, nil); err != nil {
					return "", fmt.Errorf("flux3klein: %s: %w", p.Name, err)
				}
				return format, nil
			})
			return names
		}
		var afs []llm.AIFunction
		for _, f := range fns {
			names[f.name] = file + "#" + f.name
			afs = append(afs, llm.AIFunction{Name: f.name, Model: f.part, Options: opts})
		}
		jobs.Add(file, func() (string, error) {
			if _, err := llm.WriteAIFunctions(filepath.Join(out, file), w, afs); err != nil {
				return "", fmt.Errorf("flux3klein: %s: %w", file, err)
			}
			return fmt.Sprintf("%s, %d functions", format, len(afs)), nil
		})
		return names
	}
	byRows := func(part func(rows int) *mil.Part, format string, dit bool) map[string]string {
		var fns []fn
		for _, r := range o.TextRows {
			fns = append(fns, fn{fnName(0, 0, r), part(r)})
		}
		return program(fns, format, dit)
	}
	bySize := func(part func(c mil.Dit) *mil.Part, format string, dit bool) map[string]string {
		var fns []fn
		for _, sz := range o.Sizes {
			c, _ := ditAt(sz, o.TextRows[0])
			fns = append(fns, fn{fnName(sz[0], sz[1], 0), part(c)})
		}
		return program(fns, format, dit)
	}
	byBoth := func(part func(c mil.Dit) *mil.Part, format string, dit bool) map[string]string {
		var fns []fn
		for _, sz := range o.Sizes {
			for _, r := range o.TextRows {
				c, _ := ditAt(sz, r)
				fns = append(fns, fn{fnName(sz[0], sz[1], r), part(c)})
			}
		}
		return program(fns, format, dit)
	}
	m := &ProgramManifest{Backend: llm.BackendCoreAI, Version: ExportVersion,
		Formats:    map[string]string{"text": o.TextFormat, "dit": o.DitFormat, "vae": "fp16", "ctx": "fp16"},
		EmbedScale: 1.0 / mil.TextScale, FittedTables: codes != "",
		Dit: Config{Hidden: d0.Hidden, HeadDim: d0.HeadDim, Latent: d0.Latent, Context: d0.Context, TimeIn: d0.TimeIn,
			Modes: d0.Modes, Singles: d0.Singles, SingleGroup: o.SingleGroup},
		Text: TextConfig{Hidden: tc.Hidden, HeadDim: tc.HeadDim, Theta: tc.Theta, Vocab: tc.Vocab}}
	type textProg struct {
		names map[string]string
		tap   int
	}
	var texts []textProg
	for _, gr := range mil.TextGroups(o.TextGroup) {
		tap := -1
		for i, t := range mil.TextTaps {
			if gr[0]+gr[1] == t {
				tap = i
			}
		}
		names := byRows(func(r int) *mil.Part { return mil.TextGroup(tc, r, gr[0], gr[1]) }, o.TextFormat, false)
		texts = append(texts, textProg{names, tap})
	}
	ctx := byRows(func(r int) *mil.Part { c := d0; c.TextRows = r; return mil.Ctx(c) }, "fp16", false)
	textModes := byRows(func(r int) *mil.Part { c := d0; c.TextRows = r; return mil.TextModes(c, 0, c.Modes) }, o.DitFormat, true)
	imageModes := bySize(func(c mil.Dit) *mil.Part { return mil.ImageModes(c, 0, c.Modes) }, o.DitFormat, true)
	var singles []struct {
		names      map[string]string
		first, cnt int
	}
	for _, gr := range mil.Groups(d0.Singles, o.SingleGroup) {
		names := byBoth(func(c mil.Dit) *mil.Part { return mil.Singles(c, gr[0], gr[1]) }, o.DitFormat, true)
		singles = append(singles, struct {
			names      map[string]string
			first, cnt int
		}{names, gr[0], gr[1]})
	}
	var vaes []map[string]string
	for s := range mil.VaeStages(vc, o.Sizes[0][0], o.Sizes[0][1]) {
		vaes = append(vaes, bySize(func(c mil.Dit) *mil.Part { return mil.VaeProgram(vc, c.Width, c.Height, s) }, "fp16", false))
	}
	for _, sz := range o.Sizes {
		for _, r := range o.TextRows {
			v := Variant{Width: sz[0], Height: sz[1], TextRows: r}
			fr, fs, fb := fnName(0, 0, r), fnName(sz[0], sz[1], 0), fnName(sz[0], sz[1], r)
			for _, t := range texts {
				v.TextGroups = append(v.TextGroups, TextProgram{Name: t.names[fr], Tap: t.tap})
			}
			v.Ctx, v.TextModes, v.ImageModes = ctx[fr], textModes[fr], imageModes[fs]
			for _, s := range singles {
				v.Singles = append(v.Singles, Group{Name: s.names[fb], First: s.first, Count: s.cnt})
			}
			for _, st := range vaes {
				v.Vae = append(v.Vae, st[fs])
			}
			m.Variants = append(m.Variants, v)
		}
	}
	if m.BNMean, m.BNStd, err = bnStats(w, vc, d0.Latent); err != nil {
		return err
	}
	if err := writeTextFiles(root, out, w, tc); err != nil {
		return err
	}
	if err := writeTimeFile(filepath.Join(out, TimeFile), w); err != nil {
		return err
	}
	// a job takes up to about 1.5 GB (a group of the DiT's blocks read as float32)
	if err := jobs.Run(2 << 30); err != nil {
		return err
	}
	if codes != "" { // the tables are in the programs now
		if err := os.RemoveAll(codes); err != nil {
			return err
		}
	}
	return writeProgramManifest(out, m)
}

// bnStats are the VAE's batch-norm mean and standard deviation of the patched latents' channels.
func bnStats(w *mil.Weights, v mil.Vae, n int) (mean, std []float32, err error) {
	mean, _, err = w.Get("vae.bn.running_mean")
	if err != nil {
		return nil, nil, err
	}
	vr, _, err := w.Get("vae.bn.running_var")
	if err != nil {
		return nil, nil, err
	}
	if len(mean) != n || len(vr) != n {
		return nil, nil, fmt.Errorf("flux3klein: the VAE's batch norm has %d and %d channels for patches of %d", len(mean), len(vr), n)
	}
	std = make([]float32, n)
	for i, x := range vr {
		std[i] = float32(math.Sqrt(float64(x) + v.BNEps))
	}
	return mean, std, nil
}

// writeTextFiles writes text/: the tokenizer and Qwen3-VL's embedding table (bf16) the host reads
// the prompt's rows from.
func writeTextFiles(root, out string, w *mil.Weights, tc mil.Text) error {
	b, err := os.ReadFile(filepath.Join(root, "text_encoder", "tokenizer.json"))
	if err != nil {
		return fmt.Errorf("flux3klein: text_encoder/tokenizer.json: %w", err)
	}
	if err := os.WriteFile(filepath.Join(out, TokenizerFile), b, 0o644); err != nil {
		return err
	}
	f, err := os.Create(filepath.Join(out, EmbedFile))
	if err != nil {
		return err
	}
	const chunk = 4096
	buf := make([]byte, 2*chunk*tc.Hidden)
	for r := 0; r < tc.Vocab; r += chunk {
		e := min(r+chunk, tc.Vocab)
		v, err := w.Rows("text.model.language_model.embed_tokens.weight", r, e)
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

// IsRepository reports whether dir is a FLUX.3 klein repository (transformer/config.json of a
// Flux3Transformer2DModel).
func IsRepository(dir string) bool {
	b, err := os.ReadFile(filepath.Join(dir, "transformer", "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		Class string `json:"_class_name"`
	}
	return json.Unmarshal(b, &c) == nil && c.Class == "Flux3Transformer2DModel"
}

// ExportName is the directory Prepare writes the export of name and o into: what changes the
// programs is in it.
func ExportName(name string, o ConvertOptions) string {
	o.defaults()
	var sizes, rows []string
	for _, s := range o.Sizes {
		if s[0] == s[1] {
			sizes = append(sizes, fmt.Sprint(s[0]))
		} else {
			sizes = append(sizes, fmt.Sprintf("%dx%d", s[0], s[1]))
		}
	}
	for _, r := range o.TextRows {
		rows = append(rows, fmt.Sprint(r))
	}
	dit := o.DitFormat
	if o.Stats != "" && o.DitFormat == "lut8" {
		dit += "x2" // the tables fitted to the activations
	}
	return fmt.Sprintf("%s-%s-t%s-%s-%s-v%d.%s", strings.ToLower(name), strings.Join(sizes, "+"), strings.Join(rows, "+"),
		o.TextFormat, dit, ExportVersion, llm.BackendCoreAI)
}
