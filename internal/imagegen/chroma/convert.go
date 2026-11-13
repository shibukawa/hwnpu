package chroma

import (
	"cmp"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/chroma/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// ExportVersion is the layout of the programs Convert writes; an export of another is converted
// again.
const ExportVersion = 1

// ConvertOptions select what Convert writes.
type ConvertOptions struct {
	// Backend is the program format: coreml (Core ML packages, the default) or coreai (Core AI
	// programs, written from the checkpoint without a Core ML package).
	Backend string
	// Width and Height are the picture's size in pixels, multiples of 16.
	Width, Height int
	// TextFormat and DitFormat are the weight formats of T5 (default lut8, with TextFormatFor wo=fp16) and of the DiT
	// (default lut8), as hwnpu-convert's -format: fp16, int8, lut8, lut6... The VAE's are fp16.
	TextFormat, DitFormat string
	// DitFormatFor are rules over the DiT's weights (llm.Options.FormatFor: "proj_mlp=lut4").
	DitFormatFor []string
	// TextFormatFor are rules over T5's weights, as DitFormatFor ("wo=fp16").
	TextFormatFor []string
	// DitQuant is the quantizer of the DiT's lut formats, as hwnpu-convert's -quant (uniform,
	// kmeans, kmeans-minmax; "" uniform).
	DitQuant string
	// TextGroup is T5's layers a program holds; DoubleGroup and SingleGroup the DiT's blocks.
	TextGroup, DoubleGroup, SingleGroup int
	// TextRows is the text's rows (512, ChromaPipeline's max_sequence_length; fewer cut a
	// longer prompt).
	TextRows int
	// Tokenizer is a tokenizer.json of T5's (Chroma's repository has only spiece.model; the one
	// of Anima's t5_tokenizer is the same vocabulary).
	Tokenizer string
	// Only, when not empty, writes these parts alone (text, mods, doubles, singles, vae).
	Only map[string]bool
	// LinkFrom are whole exports to hard link the parts that would come out the same from: T5's
	// groups but the last (the same T5 format and text rows), the last and the empty prompt's rows
	// too (the same model), the VAE's stages (the same size). The Core ML cache keys a compiled
	// program by its bytes, which a conversion makes anew (random UUIDs): linked, the programs
	// are compiled once for every export, and take the disk and the compile cache once.
	LinkFrom []string
}

// defaults fills what o leaves out.
func (o *ConvertOptions) defaults() {
	if o.Backend == "" {
		o.Backend = llm.BackendCoreML
	}
	if o.TextFormat == "" {
		// lut8 with wo, the FF's output projection whose inputs carry T5's outliers, at fp16
		// (5.5 GB against fp16's 9.5): the same pictures as fp16 (latent cosine 0.996 to 0.999 to
		// fp16's T5), where lut8 throughout took one prompt row to a cosine of 0.77 and int8 put
		// infinities on the Neural Engine (wo's integer sums, design.md 16.170)
		o.TextFormat, o.TextFormatFor = "lut8", []string{"wo=fp16"}
	}
	if o.DitFormat == "" {
		// lut8: per-row int8 put infinities in the later double blocks' text stream on the Neural
		// Engine (its integer sums of large inputs), and the velocity was 14 % off float32 at
		// sigma 0.25; lut8 1.2 % (design.md 16.170)
		o.DitFormat = "lut8"
	}
	if o.TextGroup <= 0 {
		o.TextGroup = 4 // 0.77 GB of weights at 8 bits
		if o.TextFormat == "fp16" {
			o.TextGroup = 2
		} else if len(o.TextFormatFor) > 0 {
			o.TextGroup = 3 // a layer with some weights at fp16: up to 0.24 GB
		}
	}
	if o.DoubleGroup <= 0 {
		o.DoubleGroup = 4 // 0.9 GB at 8 bits
	}
	if o.SingleGroup <= 0 {
		o.SingleGroup = 8
	}
	if o.TextRows <= 0 {
		o.TextRows = mil.TextRows
	}
	if o.Width == 0 {
		o.Width = 1024
	}
	if o.Height == 0 {
		o.Height = o.Width
	}
}

// ProgramExt is the extension of a program of the format backend: .aimodel for coreai, else
// .mlpackage.
func ProgramExt(backend string) string {
	if backend == llm.BackendCoreAI {
		return ".aimodel"
	}
	return ".mlpackage"
}

// Convert writes an export (Core ML's, or Core AI's for o.Backend coreai) of the Chroma
// repository root (transformer/, scheduler/; T5's text_encoder/ and the VAE's vae/ in shared,
// which may be root) into out: the programs, text/ (the tokenizer and T5's embedding table,
// bf16, as the IRON export's), model.json and manifest.json.
func Convert(root, shared, out string, o ConvertOptions, log io.Writer) error {
	o.defaults()
	if log == nil {
		log = io.Discard
	}
	dc, err := mil.ReadDit(root, o.Width, o.Height)
	if err != nil {
		return err
	}
	dc.TextRows = o.TextRows
	tc, err := mil.ReadT5(shared)
	if err != nil {
		return err
	}
	vc, err := mil.ReadVae(shared)
	if err != nil {
		return err
	}
	w, err := mil.OpenWeights(root, shared, shared)
	if err != nil {
		return err
	}
	defer w.Close()
	if err := os.MkdirAll(filepath.Join(out, "text"), 0o755); err != nil {
		return err
	}
	texts := mil.TextGroups(tc.Layers, o.TextGroup)
	doubles, singles := mil.DoubleGroups(dc, o.DoubleGroup), mil.SingleGroups(dc, o.SingleGroup)
	stages := mil.VaeStages(vc, o.Width, o.Height)
	total := len(texts) + 1 + len(doubles) + len(singles) + len(stages)
	written := 0
	name := filepath.Base(root)
	from := linkSources(o.LinkFrom, name, o, dc)
	ext := ProgramExt(o.Backend)
	lastText := fmt.Sprintf("text_%d%s", texts[len(texts)-1][0], ext)
	write := func(kind string, p *mil.Part, format string, rules []string) (string, error) {
		opts := llm.Options{Backend: o.Backend, Format: format, FormatFor: rules}
		if kind == "doubles" || kind == "singles" {
			opts.Quant = o.DitQuant
		}
		name := p.Name + ext
		if len(o.Only) > 0 && !o.Only[kind] {
			return name, nil
		}
		written++
		if src := from.part(kind, name, name == lastText); src != "" {
			if err := linkTree(filepath.Join(src, name), filepath.Join(out, name)); err != nil {
				return "", err
			}
			fmt.Fprintf(log, "  [%2d/%d] %s linked from %s\n", written, total, name, filepath.Base(src))
			return name, nil
		}
		bar := console.NewBar(log, fmt.Sprintf("  [%2d/%d] %s", written, total, name))
		if _, err := llm.WritePart(filepath.Join(out, name), p, w, opts, nil); err != nil {
			bar.Done("failed")
			return "", fmt.Errorf("chroma: %s: %w", p.Name, err)
		}
		bar.Done(fmt.Sprintf("%s, %s", console.Duration(bar.Elapsed()), format))
		return name, nil
	}
	m := &ProgramManifest{Backend: o.Backend, Version: ExportVersion,
		Formats:    map[string]string{"t5": o.t5Formats(), "dit": o.ditFormats(), "vae": "fp16"},
		EmbedScale: 1.0 / mil.T5Scale,
		Dit: Config{Width: o.Width, Height: o.Height, TextRows: dc.TextRows, Hidden: dc.Hidden, Latent: dc.Latent, T5: tc.Hidden,
			Doubles: dc.Doubles, Singles: dc.Singles, Mods: dc.Mods(), ModRows: dc.Mods(), ApproxIn: dc.ApproxIn}}
	m.T5.Hidden, m.T5.Layers, m.T5.Vocab = tc.Hidden, tc.Layers, tc.Vocab
	for _, gr := range texts {
		name, err := write("text", mil.Text(tc, dc, gr[0], gr[1], dc.TextRows), o.TextFormat, o.TextFormatFor)
		if err != nil {
			return err
		}
		m.Text = append(m.Text, name)
	}
	// fp16: it runs once a sigma before the steps and is closed, so its bytes are read a few times
	if m.Mods, err = write("mods", mil.Mods(dc), "fp16", nil); err != nil {
		return err
	}
	for _, gr := range doubles {
		name, err := write("doubles", mil.Doubles(dc, gr[0], gr[1]), o.DitFormat, o.DitFormatFor)
		if err != nil {
			return err
		}
		m.Doubles = append(m.Doubles, Group{name, gr[0], gr[1]})
	}
	for _, gr := range singles {
		name, err := write("singles", mil.Singles(dc, gr[0], gr[1]), o.DitFormat, o.DitFormatFor)
		if err != nil {
			return err
		}
		m.Singles = append(m.Singles, Group{name, gr[0], gr[1]})
	}
	for i := range stages {
		name, err := write("vae", mil.VaeProgram(vc, o.Width, o.Height, i), "fp16", nil)
		if err != nil {
			return err
		}
		m.Vae = append(m.Vae, name)
	}
	if len(o.Only) == 0 || o.Only["text"] {
		if o.Tokenizer == "" {
			o.Tokenizer = filepath.Join(shared, "tokenizer.json")
		}
		if src := from.text; src != "" {
			for _, f := range []string{"tokenizer.json", "embed.bin"} {
				if err := linkTree(filepath.Join(src, "text", f), filepath.Join(out, "text", f)); err != nil {
					return err
				}
			}
		} else if err := writeTextFiles(o.Tokenizer, out, w, tc); err != nil {
			return err
		}
		if src := from.model; src != "" {
			if err := linkTree(filepath.Join(src, EmptyFile), filepath.Join(out, EmptyFile)); err != nil {
				return err
			}
		} else if err := func() error {
			bar := console.NewBar(log, "  the empty prompt's text rows")
			rows, err := emptyRows(w, tc, dc)
			if err == nil {
				err = writeEmpty(out, rows)
			}
			if err != nil {
				bar.Done("failed")
				return err
			}
			bar.Done(console.Duration(bar.Elapsed()))
			return nil
		}(); err != nil {
			return err
		}
	}
	if err := writeModel(root, out); err != nil {
		return err
	}
	return writeProgramManifest(out, m)
}

// writeModel writes model.json, as tools/amd/iron_chroma.py's write_model: Chroma1-Flash Heun's 8
// steps without guidance; Chroma1-HD guidance 3 with DPM-Solver++ 2M 25 steps, guided at sigma
// >= 0.6 only (design.md 16.157).
func writeModel(root, out string) error {
	b, err := os.ReadFile(filepath.Join(root, "scheduler", "scheduler_config.json"))
	if err != nil {
		return err
	}
	var sched map[string]any
	if err := json.Unmarshal(b, &sched); err != nil {
		return err
	}
	name := filepath.Base(root)
	m := map[string]any{"name": name, "scheduler": sched}
	if strings.Contains(strings.ToLower(name), "flash") {
		m["steps"], m["guidance"], m["sampler"], m["cfg_sigmas"] = 8, 1.0, "heun", nil
	} else {
		m["steps"], m["guidance"], m["sampler"], m["cfg_sigmas"] = 25, 3.0, "dpm2m", []float64{0.6, 1}
	}
	mb, err := json.MarshalIndent(m, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, "model.json"), append(mb, '\n'), 0o644)
}

// writeTextFiles writes text/: the tokenizer and T5's embedding table (bf16) the host reads the
// prompt's rows from.
func writeTextFiles(tokenizer, out string, w *mil.Weights, tc mil.T5) error {
	b, err := os.ReadFile(tokenizer)
	if err != nil {
		return fmt.Errorf("chroma: T5's tokenizer.json: %w", err)
	}
	if err := os.WriteFile(filepath.Join(out, "text", "tokenizer.json"), b, 0o644); err != nil {
		return err
	}
	f, err := os.Create(filepath.Join(out, "text", "embed.bin"))
	if err != nil {
		return err
	}
	const chunk = 4096
	buf := make([]byte, 2*chunk*tc.Hidden)
	for r := 0; r < tc.Vocab; r += chunk {
		e := min(r+chunk, tc.Vocab)
		v, err := w.Rows("t5.shared.weight", r, e)
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

// IsRepository reports whether dir is a Chroma repository (transformer/config.json of a
// ChromaTransformer2DModel).
func IsRepository(dir string) bool {
	b, err := os.ReadFile(filepath.Join(dir, "transformer", "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		Class string `json:"_class_name"`
	}
	return json.Unmarshal(b, &c) == nil && c.Class == "ChromaTransformer2DModel"
}

// linkFrom is the exports Convert links parts from: text the one with T5's programs, model the
// one of the same model too (its last T5 program, its empty rows), vae the one of the same size.
type linkFrom struct{ text, model, vae, dit string }

// t5Formats and ditFormats are the formats of T5's and the DiT's weights as the manifest records
// them: the format, then the rules and the quantizer.
func (o ConvertOptions) t5Formats() string {
	return strings.Join(append([]string{o.TextFormat}, o.TextFormatFor...), " ")
}

func (o ConvertOptions) ditFormats() string {
	f := append([]string{o.DitFormat}, o.DitFormatFor...)
	if o.DitQuant != "" {
		f = append(f, o.DitQuant)
	}
	return strings.Join(f, " ")
}

// linkSources picks from the whole exports dirs those whose parts Convert can link (LinkFrom).
func linkSources(dirs []string, name string, o ConvertOptions, dc mil.Dit) linkFrom {
	var f linkFrom
	for _, d := range dirs {
		m, err := ReadProgramManifest(d)
		if err != nil || m.Version != ExportVersion || cmp.Or(m.Backend, llm.BackendCoreML) != cmp.Or(o.Backend, llm.BackendCoreML) {
			continue
		}
		var md struct {
			Name string `json:"name"`
		}
		same := false
		if b, err := os.ReadFile(filepath.Join(d, "model.json")); err == nil && json.Unmarshal(b, &md) == nil {
			same = md.Name == name
		}
		if m.Formats["t5"] == o.t5Formats() && m.Dit.TextRows == dc.TextRows && len(m.Text) > 0 {
			if f.text == "" {
				f.text = d
			}
			if same && f.model == "" {
				f.model = d
			}
		}
		// the approximator and the DiT's groups: the same model, size, text rows and formats
		if same && f.dit == "" && m.Formats["dit"] == o.ditFormats() && m.Dit.Width == o.Width && m.Dit.Height == o.Height &&
			m.Dit.TextRows == dc.TextRows && len(m.Doubles) > 0 {
			f.dit = d
		}
		if m.Dit.Width == o.Width && m.Dit.Height == o.Height && len(m.Vae) > 0 && f.vae == "" {
			f.vae = d
		}
	}
	return f
}

// part is the export to link the program name of kind from, or "" (last: T5's last group, which
// holds the model's context_embedder).
func (f linkFrom) part(kind, name string, last bool) string {
	src := ""
	switch {
	case kind == "text" && last:
		src = f.model
	case kind == "text":
		src = f.text
	case kind == "vae":
		src = f.vae
	case kind == "mods" || kind == "doubles" || kind == "singles":
		src = f.dit
	}
	if src == "" {
		return ""
	}
	if _, err := os.Stat(filepath.Join(src, name)); err != nil {
		return ""
	}
	return src
}

// linkTree hard links the file or directory src as dst.
func linkTree(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, path)
		to := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(to, 0o755)
		}
		os.Remove(to)
		return os.Link(path, to)
	})
}
