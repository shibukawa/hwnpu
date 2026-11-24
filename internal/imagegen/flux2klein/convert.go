package flux2klein

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/internal/console"
	"github.com/shibukawa/bonsaiai/internal/imagegen/aspect"
	"github.com/shibukawa/bonsaiai/internal/imagegen/flux2klein/mil"
	"github.com/shibukawa/bonsaiai/internal/llm"
)

// ExportVersion is the layout of the programs Convert writes; an export of another is converted
// again. 2: the edits of PrebuiltKeys keys or more in programs of their own (DeviceSuffix); 3:
// those in an export of their own beside it (DeviceExport); 4: 512 x 512's attention takes its
// 1536 queries as two chunks of 768 (mil's headChunk: the prebuilt programs for an M3 failed to
// load, docs/known-issues.md 12); 5: an attention over 8192 keys or more joins its parts by their
// log-sum-exp (milpart.PieceKeys), so that the edit at 1024 x 1024 with a reference of 1024 is in
// the export, prebuilt (PrebuiltKeys); 6: two single blocks a program (model-recipes.yaml
// single_group: five left the edit at 1024 x 1024 from three references of 512 off the Neural
// Engine on an M1 and an M2, "RegAlloc: failed to allocate tensors with fixed L2 hint", and every
// size off it on an M1).
const ExportVersion = 6

// ConvertOptions select what Convert writes.
type ConvertOptions struct {
	// Width and Height are the picture's size in pixels, multiples of 16.
	Width, Height int
	// TextFormat and DitFormat are the weight formats of Qwen3-4B and of the DiT, as
	// bonsaiai-convert's -format (default lut8 both). The VAE's are fp16.
	TextFormat, DitFormat string
	// TextGroup is Qwen3's layers a program holds at most (a group ends at each hidden state the
	// DiT reads, mil.TextGroups); DoubleGroup and SingleGroup the DiT's blocks.
	TextGroup, DoubleGroup, SingleGroup int
	// Only, when not empty, writes these parts alone (text, ctx, doubles, singles, vae, encoder,
	// files).
	Only map[string]bool
	// Sizes are the picture sizes (width, height) the export holds, each part that depends on the
	// size one program with a function each (llm.WriteAIFunctions: the weights once, on the disk
	// and in Core AI's cache). Empty: DefaultSizes when Width x Height is one of them, else Width x
	// Height alone.
	Sizes [][2]int
	// Edits are the edits the export holds programs for too (Core AI): the DiT's groups over the
	// picture's rows and the references' as more functions of the same programs (the weights
	// once), and the VAE's encoder at each reference size. Each is at one of Sizes.
	Edits []Edit
}

// Edit is the edits of an export at pictures of Width x Height: the references of each step, each
// a list of the picture's sides over 1 (a reference of the picture's size) or 2 (of half its
// sides).
type Edit struct {
	Width, Height int
	Refs          [][]int
}

// PrebuiltKeys is the most keys (the DiT's rows) a program coreai-build compiles for the Neural
// Engine takes, plus one. An attention over 8192 keys or more joined by one softmax it left to the
// GPU ("NEFUSED_EW: KernelPreScales"); with its parts joined by their log-sum-exp
// (milpart.PieceKeys) the edit at 1024 x 1024 with a reference of 1024 (8704 keys) compiles for
// the Neural Engine (Metal Toolchain 27.2, h13g and h15g, 2026-11-24). More, with half-size
// references beside it (9728 and 10752 keys), still fail a group of single blocks ("Failed to
// compile function ...: unexpected failure", h15g; a group of double blocks on h13g) though the
// attention alone compiles: those edits' steps are a program of their own (DeviceSuffix), where
// the device's compiler takes them, so that the others keep their prebuilt.
const PrebuiltKeys = 8704 + 1

// DeviceSuffix ends the name of a program the device compiles (coreai.LeftToDevice): the DiT's
// groups of the edits of PrebuiltKeys keys or more.
const DeviceSuffix = "_device"

// DeviceExport is the export beside the export out that holds the programs of its edits of
// PrebuiltKeys keys or more (DeviceSuffix): NAME-device.coreai. Drawing from a prompt, and the
// edits of fewer keys, never download it (PrepareDevice).
func DeviceExport(out string) string {
	return strings.TrimSuffix(out, "."+llm.BackendCoreAI) + "-device." + llm.BackendCoreAI
}

// DeviceKind is the kind of manifest.json of a DeviceExport.
const DeviceKind = "flux2klein-device"

// EditFunction is the function of a program of several sizes that runs a step at w x h with
// references of the picture's sides over divs.
func EditFunction(w, h int, divs []int) string {
	return SizeFunction(w, h) + "_" + mil.RefsName(slices.Sorted(slices.Values(divs)))
}

// editTag names o's edits in its export's name: "" for none.
func (o ConvertOptions) editTag() string {
	if len(o.Edits) == 0 {
		return ""
	}
	var b strings.Builder
	n := 0
	for _, e := range o.Edits {
		fmt.Fprintf(&b, "%dx%d:", e.Width, e.Height)
		for _, r := range e.Refs {
			fmt.Fprintf(&b, "%s,", mil.RefsName(slices.Sorted(slices.Values(r))))
			n++
		}
		b.WriteString(";")
	}
	return fmt.Sprintf("-%dedits-%08x", n, crc32.ChecksumIEEE([]byte(b.String())))
}

// ParseReferences is the references bonsaiai-image names name ("1", "2@half", "1+2@half"; models.yaml
// npu's edits) as the picture's sides over each: 1 for one of its size, 2 for half its sides.
func ParseReferences(name string) ([]int, error) {
	var divs []int
	for _, part := range strings.Split(name, "+") {
		n, div := part, 1
		if s, ok := strings.CutSuffix(part, "@half"); ok {
			n, div = s, 2
		}
		k, err := strconv.Atoi(n)
		if err != nil || k <= 0 {
			return nil, fmt.Errorf("flux2klein: references %q: N, N@half or a sum of them", name)
		}
		for range k {
			divs = append(divs, div)
		}
	}
	if len(divs) > MaxReferences {
		return nil, fmt.Errorf("flux2klein: references %q: more than %d", name, MaxReferences)
	}
	return divs, nil
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

// FromRecipe fills, for model name, the layers and blocks of a program o leaves out (text_group, double_group,
// single_group) from model name's recipe (model-recipes.yaml apple.convert).
func (o *ConvertOptions) FromRecipe(name string) error {
	if err := chat.FillConvert(name, map[string]*int{"text_group": &o.TextGroup, "double_group": &o.DoubleGroup, "single_group": &o.SingleGroup}, nil); err != nil {
		return fmt.Errorf("flux2klein: %w", err)
	}
	return nil
}

// defaults fills what o leaves out (after FromRecipe).
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
	if o.TextGroup <= 0 || o.DoubleGroup <= 0 || o.SingleGroup <= 0 {
		return fmt.Errorf("flux2klein: no programs' groups (FromRecipe: model-recipes.yaml apple.convert)")
	}
	if log == nil {
		log = io.Discard
	}
	sizes := o.sizes()
	for _, ed := range o.Edits {
		if !slices.Contains(sizes, [2]int{ed.Width, ed.Height}) {
			return fmt.Errorf("flux2klein: edits at %dx%d, not one of the export's sizes %v", ed.Width, ed.Height, sizes)
		}
		for _, refs := range ed.Refs {
			if len(refs) == 0 || len(refs) > MaxReferences {
				return fmt.Errorf("flux2klein: an edit of %d references (1 to %d)", len(refs), MaxReferences)
			}
		}
	}
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
	// writeFns writes a part of each of vs: one program as write does for one (unless name is
	// given), else a program with a function each, the weights once, named name or the first
	// part's. The names are one each.
	type variant struct {
		fn   string // the function (SizeFunction, EditFunction)
		part func() *mil.Part
	}
	dout := DeviceExport(out)
	writeFns := func(kind, name string, vs []variant, format string) []string {
		if len(vs) == 1 && name == "" {
			return []string{write(kind, vs[0].part(), format)}
		}
		if name == "" {
			name = vs[0].part().Name + ext
		}
		var names []string
		var fns []llm.AIFunction
		for _, v := range vs {
			names = append(names, name+"#"+v.fn)
			fns = append(fns, llm.AIFunction{Name: v.fn, Model: v.part(), Options: llm.Options{Backend: llm.BackendCoreAI, Format: format}})
		}
		if len(o.Only) > 0 && !o.Only[kind] {
			return names
		}
		dir := out
		if kind == "device" {
			dir = dout
			if err := os.MkdirAll(dir, 0o755); err != nil {
				jobs.Add(name, func() (string, error) { return "", err })
				return names
			}
		}
		jobs.Add(name, func() (string, error) {
			if _, err := llm.WriteAIFunctions(filepath.Join(dir, name), w, fns); err != nil {
				return "", fmt.Errorf("flux2klein: %s: %w", name, err)
			}
			return fmt.Sprintf("%s, %d functions", format, len(fns)), nil
		})
		return names
	}
	// the DiT's configurations a group of its blocks has a function for: each size's, then each
	// edit's step. An edit's step of PrebuiltKeys keys or more is a function of a program of its
	// own (DeviceSuffix), the group's weights again: compiled on the device
	type ditFn struct {
		fn string
		c  mil.Dit
	}
	var dits []ditFn
	for i, sz := range sizes {
		dits = append(dits, ditFn{SizeFunction(sz[0], sz[1]), dcs[i]})
	}
	for _, ed := range o.Edits {
		for _, refs := range ed.Refs {
			c, err := mil.ReadDit(root, ed.Width, ed.Height)
			if err != nil {
				return err
			}
			c.Refs = refs
			if err := c.CheckRefs(); err != nil {
				return err
			}
			dits = append(dits, ditFn{EditFunction(ed.Width, ed.Height, refs), c})
		}
	}
	writeDit := func(kind string, part func(c mil.Dit) *mil.Part) []string {
		var vs, dvs []variant
		var at []int // each of dits' index in vs (>= 0) or in dvs (-1 - i)
		for _, d := range dits {
			v := variant{d.fn, func() *mil.Part { return part(d.c) }}
			if d.c.Rows() >= PrebuiltKeys {
				at = append(at, -1-len(dvs))
				dvs = append(dvs, v)
			} else {
				at = append(at, len(vs))
				vs = append(vs, v)
			}
		}
		names := writeFns(kind, "", vs, o.DitFormat)
		var dnames []string
		if len(dvs) > 0 {
			dnames = writeFns("device", part(dcs[0]).Name+DeviceSuffix+ext, dvs, o.DitFormat)
		}
		out := make([]string, len(dits))
		for i, k := range at {
			if k >= 0 {
				out[i] = names[k]
			} else {
				out[i] = dnames[-1-k]
			}
		}
		return out
	}
	writeSized := func(kind string, part func(c mil.Dit) *mil.Part, format string) []string {
		vs := make([]variant, len(sizes))
		for i, sz := range sizes {
			vs[i] = variant{SizeFunction(sz[0], sz[1]), func() *mil.Part { return part(dcs[i]) }}
		}
		return writeFns(kind, "", vs, format)
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
	edits := make([]EditSize, len(o.Edits))
	for i, ed := range o.Edits {
		edits[i] = EditSize{Width: ed.Width, Height: ed.Height, Encoders: map[int][]string{}}
		for _, refs := range ed.Refs {
			edits[i].Steps = append(edits[i].Steps, EditStep{Refs: refs})
		}
	}
	// add puts each function's name of a group of the DiT's blocks in its size's or edit's step
	add := func(names []string, gr [2]int, double bool) {
		k := len(sizes)
		for i := range sized {
			if double {
				sized[i].Doubles = append(sized[i].Doubles, Group{names[i], gr[0], gr[1]})
			} else {
				sized[i].Singles = append(sized[i].Singles, Group{names[i], gr[0], gr[1]})
			}
		}
		for i := range edits {
			for j := range edits[i].Steps {
				st := &edits[i].Steps[j]
				if double {
					st.Doubles = append(st.Doubles, Group{names[k], gr[0], gr[1]})
				} else {
					st.Singles = append(st.Singles, Group{names[k], gr[0], gr[1]})
				}
				k++
			}
		}
	}
	for _, gr := range mil.Groups(dc.Doubles, o.DoubleGroup) {
		add(writeDit("doubles", func(c mil.Dit) *mil.Part { return mil.Doubles(c, gr[0], gr[1]) }), gr, true)
	}
	for _, gr := range mil.Groups(dc.Singles, o.SingleGroup) {
		add(writeDit("singles", func(c mil.Dit) *mil.Part { return mil.Singles(c, gr[0], gr[1]) }), gr, false)
	}
	// the VAE's encoder of each reference size of the edits: one program a stage, a function for
	// each size
	var encSizes [][2]int
	for _, ed := range o.Edits {
		for _, refs := range ed.Refs {
			for _, d := range refs {
				if es := [2]int{ed.Width / d, ed.Height / d}; !slices.Contains(encSizes, es) {
					encSizes = append(encSizes, es)
				}
			}
		}
	}
	if len(encSizes) > 0 {
		n := len(mil.VaeEncStages(vc, encSizes[0][0], encSizes[0][1]))
		for _, es := range encSizes[1:] {
			if k := len(mil.VaeEncStages(vc, es[0], es[1])); k != n {
				return fmt.Errorf("flux2klein: the VAE's encoder in %d stages at %dx%d, %d at %dx%d", n, encSizes[0][0], encSizes[0][1], k, es[0], es[1])
			}
		}
		for s := 0; s < n; s++ {
			vs := make([]variant, len(encSizes))
			for i, es := range encSizes {
				vs[i] = variant{SizeFunction(es[0], es[1]), func() *mil.Part { return mil.VaeEncProgram(vc, es[0], es[1], s) }}
			}
			names := writeFns("encoder", "", vs, "fp16")
			for i := range edits {
				for _, st := range edits[i].Steps {
					for _, d := range st.Refs {
						if _, ok := edits[i].Encoders[d]; ok && len(edits[i].Encoders[d]) > s {
							continue
						}
						k := slices.Index(encSizes, [2]int{edits[i].Width / d, edits[i].Height / d})
						edits[i].Encoders[d] = append(edits[i].Encoders[d], names[k])
					}
				}
			}
		}
	}
	m.Edits = edits
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
	// the edits' steps whose programs are the DeviceExport's
	for i := range m.Edits {
		for j := range m.Edits[i].Steps {
			st := &m.Edits[i].Steps[j]
			if len(st.Doubles) > 0 && strings.Contains(st.Doubles[0].Name, DeviceSuffix+ext) {
				st.Device, m.Device = true, filepath.Base(dout)
			}
		}
	}
	if m.Device != "" && (len(o.Only) == 0 || o.Only["device"]) {
		if err := os.MkdirAll(dout, 0o755); err != nil {
			return err
		}
		b, err := json.MarshalIndent(map[string]any{"kind": DeviceKind, "version": ExportVersion, "of": filepath.Base(out)}, "", " ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dout, "manifest.json"), append(b, '\n'), 0o644); err != nil {
			return err
		}
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
	o.FromRecipe(name) // Prepare says what it lacks
	o.defaults()
	size := aspect.Tag(o.sizes()) + o.editTag()
	return fmt.Sprintf("%s-%s-%s-%s-v%d.%s", strings.ToLower(name), size, o.TextFormat, o.DitFormat, ExportVersion, llm.BackendCoreAI)
}
