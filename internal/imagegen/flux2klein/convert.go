package flux2klein

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
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// ExportVersion is the layout of the programs Convert writes; an export of another is converted
// again. 2: 512 x 512's attention takes its 1536 queries as two chunks of 768 (mil's headChunk:
// the prebuilt programs for an M3 failed to load).
const ExportVersion = 2

// ConvertOptions select what Convert writes.
type ConvertOptions struct {
	// Width and Height are the picture's size in pixels, multiples of 16.
	Width, Height int
	// TextFormat and DitFormat are the weight formats of Qwen3-4B and of the DiT, as
	// hwnpu-convert's -format (default lut8 both). The VAE's are fp16.
	TextFormat, DitFormat string
	// TextGroup is Qwen3's layers a program holds at most (a group ends at each hidden state the
	// DiT reads, mil.TextGroups); DoubleGroup and SingleGroup the DiT's blocks.
	TextGroup, DoubleGroup, SingleGroup int
	// Only, when not empty, writes these parts alone (text, ctx, doubles, singles, vae, files).
	Only map[string]bool
	// Sizes are the picture sizes (width, height) the export holds, each part that depends on the
	// size one program with a function each (llm.WriteAIFunctions: the weights once, on the disk
	// and in Core AI's cache). Empty: DefaultSizes when Width x Height is one of them, else Width x
	// Height alone.
	Sizes [][2]int
}

// DefaultSizes are the sizes of an export by default: 512 x 512 and 1024 x 1024, one export for
// both (the functions of the one not drawn are never loaded; they share the weights' bytes with
// the other's), as Sana's (docs/design/image-apple.md).
var DefaultSizes = [][2]int{{512, 512}, {1024, 1024}}

// sizes is what o's export holds (after defaults).
func (o ConvertOptions) sizes() [][2]int {
	if len(o.Sizes) > 0 {
		return o.Sizes
	}
	for _, s := range DefaultSizes {
		if s == [2]int{o.Width, o.Height} {
			return DefaultSizes
		}
	}
	return [][2]int{{o.Width, o.Height}}
}

// SizeFunction is the function of a program of several sizes that runs at w x h.
func SizeFunction(w, h int) string { return fmt.Sprintf("s%dx%d", w, h) }

// defaults fills what o leaves out.
func (o *ConvertOptions) defaults() {
	if o.TextFormat == "" {
		// lut8: Qwen3-VL-8B's attention sink came out NaN at int8 on the Neural Engine
		// (qwenimage's DefaultTextFormat, docs/design/image-apple.md)
		o.TextFormat = "lut8"
	}
	if o.DitFormat == "" {
		// lut8: Chroma's (FLUX.1's) DiT put infinities in its later blocks at per-row int8 (the
		// Neural Engine's integer sums of large inputs, docs/design/image-apple.md)
		o.DitFormat = "lut8"
	}
	if o.TextGroup <= 0 {
		o.TextGroup = 3 // 0.3 GB of weights at 8 bits (101 MB a layer)
	}
	if o.DoubleGroup <= 0 {
		o.DoubleGroup = 2 // 0.5 GB at 8 bits (245 MB a block)
	}
	if o.SingleGroup <= 0 {
		o.SingleGroup = 5 // 0.6 GB at 8 bits (123 MB a block)
	}
	if o.Width == 0 {
		o.Width = 1024
	}
	if o.Height == 0 {
		o.Height = o.Width
	}
}

// Files of an export of Convert: the tokenizer and Qwen3's embedding rows (bf16, as the IRON
// export's text/), the time embedding's weights (TimeFile) and the VAE's batch-norm statistics
// (in the manifest).
const (
	TokenizerFile = "text/tokenizer.json"
	EmbedFile     = "text/embed.bin"
)

// Convert writes a Core AI export of the FLUX.2 klein repository root (transformer/,
// text_encoder/, tokenizer/, vae/, scheduler/) into out: the programs, text/, TimeFile,
// model.json and manifest.json.
func Convert(root, out string, o ConvertOptions, log io.Writer) (err error) {
	o.defaults()
	if log == nil {
		log = io.Discard
	}
	sizes := o.sizes()
	dcs := make([]mil.Dit, len(sizes))
	for i, sz := range sizes {
		if dcs[i], err = mil.ReadDit(root, sz[0], sz[1]); err != nil {
			return err
		}
	}
	dc, err := mil.ReadDit(root, o.Width, o.Height)
	if err != nil {
		return err
	}
	tc, err := mil.ReadText(root)
	if err != nil {
		return err
	}
	vc, err := mil.ReadVae(root)
	if err != nil {
		return err
	}
	w, err := mil.OpenWeights(root, dc)
	if err != nil {
		return err
	}
	defer w.Close()
	if err := os.MkdirAll(filepath.Join(out, "text"), 0o755); err != nil {
		return err
	}
	jobs := console.NewJobs(log)
	const ext = ".aimodel"
	write := func(kind string, p *mil.Part, format string) string {
		name := p.Name + ext
		if len(o.Only) > 0 && !o.Only[kind] {
			return name
		}
		opts := llm.Options{Backend: llm.BackendCoreAI, Format: format}
		jobs.Add(name, func() (string, error) {
			if _, err := llm.WritePart(filepath.Join(out, name), p, w, opts, nil); err != nil {
				return "", fmt.Errorf("flux2klein: %s: %w", p.Name, err)
			}
			return format, nil
		})
		return name
	}
	// writeSized writes the part of each size (part(dc)): one program as write does for one size,
	// else a program with a function each (SizeFunction), the weights once. The names are one a
	// size.
	writeSized := func(kind string, part func(c mil.Dit) *mil.Part, format string) []string {
		if len(sizes) == 1 {
			return []string{write(kind, part(dcs[0]), format)}
		}
		name := part(dcs[0]).Name + ext
		var names []string
		var fns []llm.AIFunction
		for i, sz := range sizes {
			f := SizeFunction(sz[0], sz[1])
			names = append(names, name+"#"+f)
			fns = append(fns, llm.AIFunction{Name: f, Model: part(dcs[i]), Options: llm.Options{Backend: llm.BackendCoreAI, Format: format}})
		}
		if len(o.Only) > 0 && !o.Only[kind] {
			return names
		}
		jobs.Add(name, func() (string, error) {
			if _, err := llm.WriteAIFunctions(filepath.Join(out, name), w, fns); err != nil {
				return "", fmt.Errorf("flux2klein: %s: %w", name, err)
			}
			return fmt.Sprintf("%s, %d sizes", format, len(sizes)), nil
		})
		return names
	}
	m := &ProgramManifest{Backend: llm.BackendCoreAI, Version: ExportVersion,
		Formats:    map[string]string{"text": o.TextFormat, "dit": o.DitFormat, "vae": "fp16"},
		EmbedScale: 1.0 / mil.TextScale,
		Dit: Config{Width: o.Width, Height: o.Height, TextRows: dc.TextRows, Hidden: dc.Hidden, HeadDim: dc.HeadDim, Latent: dc.Latent,
			Text: tc.Hidden, Axes: dc.Axes, Theta: dc.Theta, Doubles: dc.Doubles, Singles: dc.Singles,
			DoubleGroup: o.DoubleGroup, SingleGroup: o.SingleGroup},
		Text: TextConfig{Rows: dc.TextRows, Hidden: tc.Hidden, Heads: tc.Heads, HeadDim: tc.HeadDim, Theta: tc.Theta}}
	m.Vocab = tc.Vocab
	for _, gr := range mil.TextGroups(o.TextGroup) {
		name := write("text", mil.TextGroup(tc, gr[0], gr[1]), o.TextFormat)
		tap := -1
		for i, t := range mil.TextTaps {
			if gr[0]+gr[1] == t {
				tap = i
			}
		}
		m.TextGroups = append(m.TextGroups, TextProgram{Name: name, Tap: tap})
	}
	// fp16: once a prompt, 24 M weights
	m.Ctx = write("ctx", mil.Ctx(dc), "fp16")
	sized := make([]SizeParts, len(sizes))
	for i, sz := range sizes {
		sized[i] = SizeParts{Width: sz[0], Height: sz[1]}
	}
	for _, gr := range mil.Groups(dc.Doubles, o.DoubleGroup) {
		names := writeSized("doubles", func(c mil.Dit) *mil.Part { return mil.Doubles(c, gr[0], gr[1]) }, o.DitFormat)
		for i := range sized {
			sized[i].Doubles = append(sized[i].Doubles, Group{names[i], gr[0], gr[1]})
		}
	}
	for _, gr := range mil.Groups(dc.Singles, o.SingleGroup) {
		names := writeSized("singles", func(c mil.Dit) *mil.Part { return mil.Singles(c, gr[0], gr[1]) }, o.DitFormat)
		for i := range sized {
			sized[i].Singles = append(sized[i].Singles, Group{names[i], gr[0], gr[1]})
		}
	}
	for s := range mil.VaeStages(vc, o.Width, o.Height) {
		names := writeSized("vae", func(c mil.Dit) *mil.Part { return mil.VaeProgram(vc, c.Width, c.Height, s) }, "fp16")
		for i := range sized {
			sized[i].Vae = append(sized[i].Vae, names[i])
		}
	}
	for _, sp := range sized {
		if sp.Width == o.Width && sp.Height == o.Height {
			m.Doubles, m.Singles, m.Vae = sp.Doubles, sp.Singles, sp.Vae
		}
	}
	if len(sized) > 1 {
		m.Sizes = sized
	}
	if m.BNMean, m.BNStd, err = bnStats(w, vc, dc.Latent); err != nil {
		return err
	}
	if len(o.Only) == 0 || o.Only["files"] {
		if err := writeTextFiles(root, out, w, tc); err != nil {
			return err
		}
		if err := writeTimeFile(filepath.Join(out, TimeFile), w); err != nil {
			return err
		}
	}
	// a job takes up to about 1.5 GB (a group of the DiT's blocks read as float32)
	if err := jobs.Run(2 << 30); err != nil {
		return err
	}
	if err := writeModel(root, out); err != nil {
		return err
	}
	return writeProgramManifest(out, m)
}

// bnStats are the VAE's batch-norm mean and standard deviation of the patched latents' channels
// (bn.running_mean, the root of bn.running_var + batch_norm_eps).
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
		return nil, nil, fmt.Errorf("flux2klein: the VAE's batch norm has %d and %d channels for patches of %d", len(mean), len(vr), n)
	}
	std = make([]float32, n)
	for i, x := range vr {
		std[i] = float32(math.Sqrt(float64(x) + v.BNEps))
	}
	return mean, std, nil
}

// writeModel writes model.json: 4 steps without guidance (the checkpoint is distilled) and the
// scheduler's base shift.
func writeModel(root, out string) error {
	b, err := os.ReadFile(filepath.Join(root, "scheduler", "scheduler_config.json"))
	if err != nil {
		return err
	}
	var sched map[string]any
	if err := json.Unmarshal(b, &sched); err != nil {
		return err
	}
	m := map[string]any{"name": filepath.Base(root), "scheduler": sched, "steps": 4, "guidance": 1.0}
	mb, err := json.MarshalIndent(m, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, "model.json"), append(mb, '\n'), 0o644)
}

// writeTextFiles writes text/: the tokenizer and Qwen3's embedding table (bf16) the host reads the
// prompt's rows from.
func writeTextFiles(root, out string, w *mil.Weights, tc mil.Text) error {
	b, err := os.ReadFile(filepath.Join(root, "tokenizer", "tokenizer.json"))
	if err != nil {
		return fmt.Errorf("flux2klein: tokenizer/tokenizer.json: %w", err)
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
		v, err := w.Rows("text.model.embed_tokens.weight", r, e)
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

func bf16(x float32) uint16 {
	u := math.Float32bits(x)
	return uint16((u + 0x7fff + (u>>16)&1) >> 16)
}

// IsRepository reports whether dir is a FLUX.2 klein repository (transformer/config.json of a
// Flux2Transformer2DModel).
func IsRepository(dir string) bool {
	b, err := os.ReadFile(filepath.Join(dir, "transformer", "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		Class string `json:"_class_name"`
	}
	return json.Unmarshal(b, &c) == nil && c.Class == "Flux2Transformer2DModel"
}

// ExportName is the directory Prepare writes the export of name and o into: what changes the
// programs is in it.
func ExportName(name string, o ConvertOptions) string {
	o.defaults()
	size := aspect.Tag(o.sizes())
	return fmt.Sprintf("%s-%s-%s-%s-v%d.%s", strings.ToLower(name), size, o.TextFormat, o.DitFormat, ExportVersion, llm.BackendCoreAI)
}
