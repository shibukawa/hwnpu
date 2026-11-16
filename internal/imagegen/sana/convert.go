package sana

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/sana/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// ExportVersion is the layout of the programs Convert writes; an export of another is converted
// again. 2: the time embedding on the host (TimeFile); 3: Gemma 2's GELU written with exp and
// real_div (mil.Text); 4: the caption's second linear scaled down (mil.Caption); 5: the
// cross-attention's keys and values once a prompt (mil.Caption gives them, mil.Blocks takes
// them), Core AI's exports of several sizes (Sizes), the VAE's channel repeat as a convolution;
// 6: Core AI's Gemma 2 programs hold the instruction's first rows as keys and values and have
// functions for the rows after them (mil.TextFrom, ProgramManifest.TextFrom); 7: Core AI's DiT
// blocks end in a multiplication by 1 and its VAE takes a ResBlock a stage and the norm's max
// with the channels last, so that coreai-build compiles them (design.md 16.233).
const ExportVersion = 7

// DitVersion is the oldest ExportVersion whose programs but Gemma 2's are those of this one: an
// export of it gives them to a new export (Prepare), which then converts Gemma 2 alone.
const DitVersion = 7

// TextFunction is the function of a Core AI Gemma 2 program that runs rows rows after the held
// ones (mil.TextFrom); WholeFunction the one that runs every row from the first (mil.Text).
func TextFunction(rows int) string { return fmt.Sprintf("from%d", rows) }

// WholeFunction is the function of a Core AI Gemma 2 program over all its rows.
const WholeFunction = "whole"

// HeldFile and HeldOutFile are, in an export whose Gemma 2 programs hold the instruction's first
// rows (TextConfig.Prefix): those rows' tokens, as JSON ({"prefix": [...]}), and their last hidden
// states, float32 rows of the hidden width.
const (
	HeldFile    = "text/held.json"
	HeldOutFile = "text/held_out.bin"
)

// ConvertOptions select what Convert writes.
type ConvertOptions struct {
	// Width and Height are the picture's size in pixels, multiples of 32.
	Width, Height int
	// TextFormat and DitFormat are the weight formats of Gemma 2 and of the DiT (and the
	// caption), as hwnpu-convert's -format: fp16, int8, lut8, lut6... The VAE's are fp16.
	TextFormat, DitFormat string
	// TextQuant is the quantizer of Gemma 2's lut formats, as hwnpu-convert's -quant (uniform,
	// kmeans, kmeans-minmax; "" uniform).
	TextQuant string
	// Group is the DiT's blocks a program holds; TextGroup Gemma 2's layers.
	Group, TextGroup int
	// Only, when not empty, writes these parts alone (text, caption, time, blocks, final, vae).
	Only map[string]bool
	// Backend is the program format (llm.Options.Backend): Core ML packages (the default), Core
	// AI programs (coreai) or OpenVINO IR for Intel's NPU.
	Backend string
	// Sizes are the picture sizes (width, height) a Core AI export holds, each part that depends on
	// the size one program with a function each (llm.WriteAIFunctions: the weights once). Empty:
	// DefaultSizes when Width x Height is one of them, else Width x Height alone. The other
	// formats hold Width x Height alone.
	Sizes [][2]int
}

// DefaultSizes are the sizes of a Core AI export by default: 512 x 512 and 1024 x 1024, one
// export for both (the parts of the one not drawn are never loaded; their functions share the
// weights' bytes with the other's).
var DefaultSizes = [][2]int{{512, 512}, {1024, 1024}}

// sizes is what o's export holds (after defaults).
func (o ConvertOptions) sizes() [][2]int {
	if o.Backend != llm.BackendCoreAI {
		return [][2]int{{o.Width, o.Height}}
	}
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

// SizeFunction is the function of a Core AI program of several sizes that runs at w x h.
func SizeFunction(w, h int) string { return fmt.Sprintf("s%dx%d", w, h) }

// ProgramExt is the extension of a program of the format backend.
func ProgramExt(backend string) string {
	switch backend {
	case llm.BackendIntel:
		return ".xml"
	case llm.BackendCoreAI:
		return ".aimodel"
	}
	return ".mlpackage"
}

// CaptionFormat is the caption program's weight format: it runs once a prompt and is closed
// before the steps, so its keys' and values' projections cost the steps nothing and take fp16
// (new-model-checklist 5.2: a pass run once a prompt takes a format up for free).
const CaptionFormat = "fp16"

// Formats are the default weight formats of Gemma 2 and the DiT of the Sana name (a models.yaml
// name or the repository's): int8 for Gemma 2 (its 5.2 GB at fp16 would be a dozen programs;
// int8 is near fp16, design.md 16.85). The DiT at 8 bits draws the same pictures in half the
// Neural Engine's memory (design.md 16.160): int8 for Sana 600M (1.14 -> 0.60 GiB on the M1),
// lut8 for SANA 1.5 1.6B, whose per-row int8 breaks the pictures (2.98 -> 1.56 GiB, under the
// M1's 2.9: 43 -> 31 s an image). Others keep fp16.
func Formats(name string) (text, dit string) {
	switch n := strings.ToLower(name); {
	case strings.Contains(n, "600m"):
		return "int8", "int8"
	case strings.Contains(n, "1.6b"):
		return "int8", "lut8"
	}
	return "int8", "fp16"
}

// FormatsFor is Formats for a program format (llm.Options.Backend). Intel's NPU reads no lookup
// tables: SANA 1.5 1.6B's DiT, whose per-row int8 breaks the pictures, stays in fp16 there
// (the NPU shares the system's memory).
func FormatsFor(name, backend string) (text, dit string) {
	text, dit = Formats(name)
	if backend == llm.BackendIntel && strings.HasPrefix(dit, "lut") {
		dit = "fp16"
	}
	return text, dit
}

// defaults fills what o leaves out for the Sana name (FormatsFor).
func (o *ConvertOptions) defaults(name string) {
	if o.Backend == "" {
		o.Backend = llm.BackendCoreML
	}
	text, dit := FormatsFor(name, o.Backend)
	if o.TextFormat == "" {
		o.TextFormat = text
	}
	if o.DitFormat == "" {
		o.DitFormat = dit
	}
	if o.Group <= 0 {
		o.Group = 4
	}
	if o.TextGroup <= 0 {
		// a program within about 500 MB of weights (78 MB a layer at int8, 156 at fp16)
		o.TextGroup = 6
		if o.TextFormat == "fp16" {
			o.TextGroup = 3
		}
	}
	if o.Width == 0 {
		o.Width = 1024
	}
	if o.Height == 0 {
		o.Height = o.Width
	}
}

// Convert writes a Core ML (or Core AI's, or OpenVINO IR, o.Backend) export of the Sana
// repository root (transformer/, vae/, scheduler/; Gemma 2's text_encoder/ and tokenizer/ in
// text, which may be root) into out: the programs, text/ (the tokenizer and the embedding table,
// bf16, as the IRON export's), model.json and manifest.json.
func Convert(root, text, out string, o ConvertOptions, log io.Writer) error {
	o.defaults(filepath.Base(root))
	if log == nil {
		log = io.Discard
	}
	sizes := o.sizes()
	dcs := make([]mil.Dit, len(sizes))
	for i, sz := range sizes {
		var err error
		if dcs[i], err = mil.ReadDit(root, sz[0], sz[1]); err != nil {
			return err
		}
		dcs[i].CoreAI = o.Backend == llm.BackendCoreAI
	}
	dc, err := mil.ReadDit(root, o.Width, o.Height)
	if err != nil {
		return err
	}
	if dc.Layers%o.Group != 0 {
		return fmt.Errorf("sana: %d blocks are no whole groups of %d", dc.Layers, o.Group)
	}
	gc, err := mil.ReadGemma(text)
	if err != nil {
		return err
	}
	var vcfg struct {
		Channels []int   `json:"decoder_block_out_channels"`
		Up       string  `json:"upsample_block_type"`
		Latent   int     `json:"latent_channels"`
		Scaling  float64 `json:"scaling_factor"`
		Layers   []int   `json:"decoder_layers_per_block"`
	}
	b, err := os.ReadFile(filepath.Join(root, "vae", "config.json"))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, &vcfg); err != nil {
		return fmt.Errorf("sana: vae/config.json: %w", err)
	}
	if fmt.Sprint(vcfg.Channels) != fmt.Sprint(mil.VaeChannels) || vcfg.Up != "interpolate" || vcfg.Latent != dc.Latent ||
		fmt.Sprint(vcfg.Layers) != "[3 3 3 3 3 3]" {
		return fmt.Errorf("sana: the VAE (%v, %s, %v) is not DC-AE f32c32's", vcfg.Channels, vcfg.Up, vcfg.Layers)
	}
	model, err := readModel(root)
	if err != nil {
		return err
	}
	// the weights of what is written: with o.Only, the others' files need not be here
	wroot, wtext := root, text
	if len(o.Only) > 0 {
		if !o.Only["caption"] && !o.Only["time"] && !o.Only["blocks"] && !o.Only["final"] && !o.Only["vae"] {
			wroot = ""
		}
		if !o.Only["text"] {
			wtext = ""
		}
	}
	w, err := mil.OpenWeights(wroot, wtext)
	if err != nil {
		return err
	}
	defer w.Close()
	if err := os.MkdirAll(filepath.Join(out, "text"), 0o755); err != nil {
		return err
	}
	groups := mil.TextGroups(gc.Layers, o.TextGroup)
	stages := mil.VaeStages(o.Width, o.Height, o.Backend == llm.BackendCoreAI)
	// the programs are written side by side once all are listed (jobs.Run)
	jobs := console.NewJobs(log)
	ext := ProgramExt(o.Backend)
	opts := func(kind, format string) llm.Options {
		opts := llm.Options{Backend: o.Backend, Format: format}
		if kind == "text" {
			opts.Quant = o.TextQuant
		}
		return opts
	}
	write := func(kind string, p *mil.Part, format string) (string, error) {
		name := p.Name + ext
		if len(o.Only) > 0 && !o.Only[kind] {
			return name, nil
		}
		jobs.Add(name, func() (string, error) {
			if _, err := llm.WritePart(filepath.Join(out, name), p, w, opts(kind, format), nil); err != nil {
				return "", fmt.Errorf("sana: %s: %w", p.Name, err)
			}
			return format, nil
		})
		return name, nil
	}
	// writeSized writes the part of each size (part(dc)): one program as write does for one size,
	// else a Core AI program with a function each (SizeFunction), the weights once. The names are
	// the manifest's, one a size.
	writeSized := func(kind string, part func(c mil.Dit) *mil.Part, format string) ([]string, error) {
		if len(sizes) == 1 {
			name, err := write(kind, part(dcs[0]), format)
			return []string{name}, err
		}
		name := part(dcs[0]).Name + ext
		var names []string
		var fns []llm.AIFunction
		for i, sz := range sizes {
			f := SizeFunction(sz[0], sz[1])
			names = append(names, name+"#"+f)
			fns = append(fns, llm.AIFunction{Name: f, Model: part(dcs[i]), Options: opts(kind, format)})
		}
		if len(o.Only) > 0 && !o.Only[kind] {
			return names, nil
		}
		jobs.Add(name, func() (string, error) {
			if _, err := llm.WriteAIFunctions(filepath.Join(out, name), w, fns); err != nil {
				return "", fmt.Errorf("sana: %s: %w", name, err)
			}
			return fmt.Sprintf("%s, %d sizes", format, len(sizes)), nil
		})
		return names, nil
	}
	m := &ProgramManifest{Backend: o.Backend, Version: ExportVersion,
		Formats: map[string]string{"text": o.TextFormat, "caption": CaptionFormat, "dit": o.DitFormat, "vae": "fp16"},
		Dit: DitConfig{Width: o.Width, Height: o.Height, Hidden: dc.Hidden, Layers: dc.Layers, Latent: dc.Latent,
			TextRows: mil.TextRows, Group: o.Group,
			VAEScale: mil.VAEScale, Caption: dc.Caption},
		Text: TextConfig{Rows: mil.TextLen, Hidden: gc.Hidden, Vocab: gc.Vocab}}
	if o.Backend == llm.BackendCoreAI {
		// Gemma 2's programs hold the instruction's first rows as keys and values, computed here on
		// the host, and have a function for each length of what follows them (mil.TextFrom), the
		// weights once; the function over all the rows stays for a prompt without the instruction
		// (the negative prompt)
		m.Text.Prefix, m.Text.Cache = mil.TextHeld, mil.TextLen
		for _, rows := range mil.TextFromRows {
			m.TextFrom = append(m.TextFrom, TextFunc{Rows: rows})
		}
		var held [][][2][]float32 // each group's layers' keys and values of the held rows
		if len(o.Only) == 0 || o.Only["text"] {
			ids, err := heldTokens(text)
			if err != nil {
				return err
			}
			var last []float32
			if held, last, err = heldRows(w, gc, groups, ids, o, out); err != nil {
				return err
			}
			if err := writeHeld(out, ids, last); err != nil {
				return err
			}
		}
		for gi, gr := range groups {
			name := mil.Text(gc, gr[0], gr[1]).Name + ext
			m.TextParts = append(m.TextParts, name+"#"+WholeFunction)
			for i, rows := range mil.TextFromRows {
				m.TextFrom[i].Parts = append(m.TextFrom[i].Parts, name+"#"+TextFunction(rows))
			}
			if held == nil {
				continue
			}
			fns := []llm.AIFunction{{Name: WholeFunction, Model: mil.Text(gc, gr[0], gr[1]), Options: opts("text", o.TextFormat)}}
			for _, rows := range mil.TextFromRows {
				fns = append(fns, llm.AIFunction{Name: TextFunction(rows), Model: mil.TextFrom(gc, gr[0], gr[1], mil.TextHeld, rows, held[gi]),
					Options: opts("text", o.TextFormat)})
			}
			jobs.Add(name, func() (string, error) {
				if _, err := llm.WriteAIFunctions(filepath.Join(out, name), w, fns); err != nil {
					return "", fmt.Errorf("sana: %s: %w", name, err)
				}
				return fmt.Sprintf("%s, %d functions", o.TextFormat, len(fns)), nil
			})
		}
	} else {
		for _, gr := range groups {
			name, err := write("text", mil.Text(gc, gr[0], gr[1]), o.TextFormat)
			if err != nil {
				return err
			}
			m.TextParts = append(m.TextParts, name)
		}
	}
	if m.Caption, err = write("caption", mil.Caption(dc), CaptionFormat); err != nil {
		return err
	}
	m.Time = TimeFile
	if len(o.Only) == 0 || o.Only["time"] {
		if err := writeTimeFile(filepath.Join(out, TimeFile), w); err != nil {
			return err
		}
	}
	sized := make([]SizeParts, len(sizes))
	for i, sz := range sizes {
		sized[i] = SizeParts{Width: sz[0], Height: sz[1]}
	}
	for first := 0; first < dc.Layers; first += o.Group {
		names, err := writeSized("blocks", func(c mil.Dit) *mil.Part { return mil.Blocks(c, first, o.Group) }, o.DitFormat)
		if err != nil {
			return err
		}
		for i := range sized {
			sized[i].Blocks = append(sized[i].Blocks, names[i])
		}
	}
	names, err := writeSized("final", mil.Final, o.DitFormat)
	if err != nil {
		return err
	}
	for i := range sized {
		sized[i].Final = names[i]
	}
	for s := range stages {
		names, err := writeSized("vae", func(c mil.Dit) *mil.Part { return mil.Vae(c.Width, c.Height, s, vcfg.Scaling, c.CoreAI) }, "fp16")
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
	// a job takes up to about 2 GB (Gemma 2's layers, the 1.6B DiT's groups)
	if err := jobs.Run(2 << 30); err != nil {
		return err
	}
	if len(o.Only) == 0 || o.Only["text"] {
		if err := writeTextFiles(text, out, w, gc); err != nil {
			return err
		}
	}
	mb, err := json.MarshalIndent(model, "", " ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "model.json"), append(mb, '\n'), 0o644); err != nil {
		return err
	}
	return writeProgramManifest(out, m)
}

// heldTokens are the tokens Gemma 2's programs hold the rows of: <bos> and the first of the
// instruction's, mil.TextHeld in all. A prompt's tokens start with them (Pipeline.tokens).
func heldTokens(text string) ([]int32, error) {
	tok, err := tokenizer.Load(filepath.Join(text, "tokenizer", "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	bos, ok := tok.ID("<bos>")
	if !ok {
		return nil, errors.New("sana: the tokenizer has no <bos>")
	}
	ids := append([]int32{bos}, tok.Encode(Instruction)...)
	// the instruction's last tokens are not held: they may merge with the prompt's first
	if len(ids) < mil.TextHeld+4 {
		return nil, fmt.Errorf("sana: the instruction is %d tokens, too few to hold %d of", len(ids), mil.TextHeld)
	}
	return ids[:mil.TextHeld], nil
}

// heldRows runs Gemma 2 over the held tokens ids on the host (milcpu, float32, the weights in the
// text format: the layers the programs run): each group's layers' keys and values of those rows
// (mil.TextKV), and the rows' last hidden states. A group's program is written under out for the
// run and removed.
func heldRows(w *mil.Weights, gc mil.Gemma, groups [][2]int, ids []int32, o ConvertOptions, out string) (kv [][][2][]float32, last []float32, err error) {
	E, n := gc.Hidden, len(ids)
	// the rows as the pipeline feeds them: the table's bf16 values times sqrt(hidden), in bf16
	x := make([]float32, n*E)
	norm := float32(math.Sqrt(float64(E)))
	for i, id := range ids {
		row, err := w.Rows("gemma.embed_tokens.weight", int(id), int(id)+1)
		if err != nil {
			return nil, nil, err
		}
		for j, v := range row {
			x[i*E+j] = fromBF16(bf16(fromBF16(bf16(v)) * norm))
		}
	}
	tmp, err := os.MkdirTemp(out, "held-")
	if err != nil {
		return nil, nil, err
	}
	defer os.RemoveAll(tmp)
	for _, gr := range groups {
		p := mil.TextKV(gc, gr[0], gr[1], n)
		path := filepath.Join(tmp, p.Name+".mlpackage")
		if _, err := llm.WritePart(path, p, w, llm.Options{Backend: llm.BackendCoreML, Format: o.TextFormat, Quant: o.TextQuant}, nil); err != nil {
			return nil, nil, fmt.Errorf("sana: %s: %w", p.Name, err)
		}
		prog, closer, err := milcpu.OpenPackage(path)
		if err != nil {
			return nil, nil, fmt.Errorf("sana: %s: %w", p.Name, err)
		}
		y, err := prog.Run(map[string]milcpu.Tensor{"x": {Shape: []int{1, n, E}, Data: x}}, milcpu.Options{})
		closer.Close()
		if err != nil {
			return nil, nil, fmt.Errorf("sana: %s: %w", p.Name, err)
		}
		if err := os.RemoveAll(path); err != nil {
			return nil, nil, err
		}
		v, ok := y["y"]
		if !ok || len(v.Data) != mil.TextKVLen(gc, gr[1], n) {
			return nil, nil, fmt.Errorf("sana: %s gave %d values, not %d", p.Name, len(v.Data), mil.TextKVLen(gc, gr[1], n))
		}
		var k [][2][]float32
		x, k = mil.SplitTextKV(gc, gr[1], n, v.Data)
		kv = append(kv, k)
	}
	return kv, x, nil
}

// writeHeld writes HeldFile and HeldOutFile: the held tokens and their last hidden states.
func writeHeld(out string, ids []int32, last []float32) error {
	b, err := json.Marshal(struct {
		Prefix []int32 `json:"prefix"`
	}{ids})
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, HeldFile), append(b, '\n'), 0o644); err != nil {
		return err
	}
	buf := make([]byte, 4*len(last))
	for i, v := range last {
		binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
	}
	return os.WriteFile(filepath.Join(out, HeldOutFile), buf, 0o644)
}

// readHeld reads HeldFile and HeldOutFile of the export dir: the tokens whose rows Gemma 2's
// programs hold (tc.Prefix of them) and their last hidden states.
func readHeld(dir string, tc TextConfig) ([]int32, []float32, error) {
	b, err := os.ReadFile(filepath.Join(dir, HeldFile))
	if err != nil {
		return nil, nil, err
	}
	var x struct {
		Prefix []int32 `json:"prefix"`
	}
	if err := json.Unmarshal(b, &x); err != nil || len(x.Prefix) != tc.Prefix {
		return nil, nil, fmt.Errorf("text: %s holds %d tokens, the programs %d rows (%v)", HeldFile, len(x.Prefix), tc.Prefix, err)
	}
	if b, err = os.ReadFile(filepath.Join(dir, HeldOutFile)); err != nil {
		return nil, nil, err
	}
	if len(b) != 4*tc.Prefix*tc.Hidden {
		return nil, nil, fmt.Errorf("text: %s is %d bytes, not %d rows of %d", HeldOutFile, len(b), tc.Prefix, tc.Hidden)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return x.Prefix, out, nil
}

// nonFinite matches a JSON value Python writes for a non-finite number.
var nonFinite = regexp.MustCompile(`(:\s*)-?(Infinity|NaN)\b`)

// readModel is model.json of root: which Sana it is and its scheduler's settings (as
// tools/amd/iron_sana.py's write_model).
func readModel(root string) (Model, error) {
	var tc struct {
		Guidance bool `json:"guidance_embeds"`
	}
	b, err := os.ReadFile(filepath.Join(root, "transformer", "config.json"))
	if err != nil {
		return Model{}, err
	}
	if err := json.Unmarshal(b, &tc); err != nil {
		return Model{}, err
	}
	var sc struct {
		Class      string  `json:"_class_name"`
		Prediction string  `json:"prediction_type"`
		Algorithm  string  `json:"algorithm_type"`
		Order      int     `json:"solver_order"`
		FlowSigmas bool    `json:"use_flow_sigmas"`
		FlowShift  float64 `json:"flow_shift"`
		Train      int     `json:"num_train_timesteps"`
	}
	if b, err = os.ReadFile(filepath.Join(root, "scheduler", "scheduler_config.json")); err != nil {
		return Model{}, err
	}
	// Python's json writes infinities as bare words (Sana 600M's "lambda_min_clipped": -Infinity),
	// which no field read here holds
	b = nonFinite.ReplaceAll(b, []byte("${1}0"))
	if err := json.Unmarshal(b, &sc); err != nil {
		return Model{}, err
	}
	m := Model{Name: filepath.Base(root), Guidance: 4.5}
	if tc.Guidance {
		return m, mil.ErrSprint
	}
	if sc.Class != "DPMSolverMultistepScheduler" || sc.Prediction != "flow_prediction" || sc.Algorithm != "dpmsolver++" ||
		sc.Order != 2 || !sc.FlowSigmas {
		return m, fmt.Errorf("sana: scheduler %s (%s, %s, order %d) is not Sana's", sc.Class, sc.Prediction, sc.Algorithm, sc.Order)
	}
	m.Steps, m.FlowShift, m.TrainSteps = 20, sc.FlowShift, sc.Train
	return m, nil
}

// writeTextFiles writes text/: the tokenizer and Gemma 2's embedding table (bf16) the host reads
// the prompt's rows from.
func writeTextFiles(text, out string, w *mil.Weights, gc mil.Gemma) error {
	b, err := os.ReadFile(filepath.Join(text, "tokenizer", "tokenizer.json"))
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
	const chunk = 8192 // rows at a time: the whole table is 2.4 GB as float32
	buf := make([]byte, 2*chunk*gc.Hidden)
	for r := 0; r < gc.Vocab; r += chunk {
		e := min(r+chunk, gc.Vocab)
		v, err := w.Rows("gemma.embed_tokens.weight", r, e)
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
