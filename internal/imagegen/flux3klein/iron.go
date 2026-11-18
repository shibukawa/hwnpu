package flux3klein

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"

	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux3klein/mil"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// IronConfig is the DiT's shape in an export of tools/amd/iron_flux3klein.py (its Flux3Config).
type IronConfig struct {
	Width          int     `json:"width"`
	Height         int     `json:"height"`
	TextSlot       int     `json:"text_slot"`
	TextPad        int     `json:"text_pad"`
	Hidden         int     `json:"hidden"`
	HeadDim        int     `json:"head_dim"`
	Latent         int     `json:"latent"`
	Text           int     `json:"text"`
	TextLayers     []int   `json:"text_layers"`
	Axes           []int   `json:"axes"`
	Theta          float64 `json:"theta"`
	ImageRopeScale float64 `json:"image_rope_scale"`
}

// Tokens is the picture's patches.
func (c IronConfig) Tokens() int { return (c.Width / 16) * (c.Height / 16) }

// ironRound is the attention's rows in whole rounds (its 8 pipelines' 64).
const ironRound = 512

// ImageRows is the image's rows in the graphs: its patches padded with zeros to whole rounds of
// the attention (iron_flux3klein.Flux3Config.image_rows; the pads are no keys).
func (c IronConfig) ImageRows() int { return (c.Tokens() + ironRound - 1) / ironRound * ironRound }

// MaxText is the longest prompt the export takes, in tokens of the chat template: the text slot's
// rows in whole TextPads.
func (c IronConfig) MaxText() int { return c.TextSlot / c.TextPad * c.TextPad }

// ironText is the text encoder's shape in an IRON export.
type ironText struct {
	Rows    int     `json:"rows"`
	Hidden  int     `json:"hidden"`
	Heads   int     `json:"heads"`
	HeadDim int     `json:"head_dim"`
	Theta   float64 `json:"theta"`
}

// ironSizesDir holds an IRON export's graphs for pictures of another size than its first:
// sizes/WxH/ holds that size's singles (on the first size's banks) and vae, and its dit_in,
// image_modes (on the first's banks) and dit_out where its image's padded rows (ImageRows) are not
// the first's (a size of the first's rows runs the first's: ironPart).
const ironSizesDir = "sizes"

// ironSizeParts are the graphs that depend on the picture's size.
var ironSizeParts = []string{"dit_in", "image_modes", "singles", "dit_out", "vae"}

// ironPart is the directory in the IRON export dir of the graph part (ironSizeParts) for pictures
// of w x h, whose first size is first's: sizes/WxH/part, else the first size's where it serves (all
// of the first size; dit_in, image_modes and dit_out of as many padded rows); "" when none does.
func ironPart(dir string, first IronConfig, w, h int, part string) string {
	if w == first.Width && h == first.Height {
		return part
	}
	sized := filepath.Join(ironSizesDir, fmt.Sprintf("%dx%d", w, h), part)
	if _, err := os.Stat(filepath.Join(dir, sized, "manifest.json")); err == nil {
		return sized
	}
	c := first
	c.Width, c.Height = w, h
	switch part {
	case "dit_in", "image_modes", "dit_out":
		if c.ImageRows() == first.ImageRows() {
			return part
		}
	}
	return ""
}

// IsIronExport reports whether dir is an export of tools/amd/iron_flux3klein.py.
func IsIronExport(dir string) bool {
	for _, p := range []string{"text/manifest.json", "ctx/manifest.json", "text_modes/manifest.json", "image_modes/manifest.json",
		"singles/manifest.json", "dit_in/manifest.json", "dit_out/manifest.json", "vae/manifest.json", TimeFile} {
		if _, err := os.Stat(filepath.Join(dir, p)); err != nil {
			return false
		}
	}
	return true
}

func readIronConfig(dir string) (IronConfig, error) {
	m, err := iron.ReadGraphManifest(filepath.Join(dir, "singles"))
	if err != nil {
		return IronConfig{}, err
	}
	var x struct {
		Config IronConfig `json:"config"`
	}
	if err := json.Unmarshal(m.Meta, &x); err != nil {
		return IronConfig{}, err
	}
	return x.Config, nil
}

// IronSizes are the picture sizes the IRON export in dir draws: its first, then those of
// sizes/WxH with every graph (ironPart); nil when dir is no IRON export.
func IronSizes(dir string) [][2]int {
	c, err := readIronConfig(dir)
	if err != nil {
		return nil
	}
	out := [][2]int{{c.Width, c.Height}}
	entries, _ := os.ReadDir(filepath.Join(dir, ironSizesDir))
	for _, e := range entries {
		var w, h int
		if _, err := fmt.Sscanf(e.Name(), "%dx%d", &w, &h); err != nil || fmt.Sprintf("%dx%d", w, h) != e.Name() {
			continue
		}
		whole := w%16 == 0 && h%16 == 0
		for _, p := range ironSizeParts {
			if ironPart(dir, c, w, h, p) == "" {
				whole = false
			}
		}
		if whole {
			out = append(out, [2]int{w, h})
		}
	}
	return out
}

// ironPipe runs an IRON export on AMD's NPU (Pipeline's, when it is one).
type ironPipe struct {
	dev   *hwnpu.Device
	c     IronConfig // the size drawn
	text  ironText
	taps  []int             // the text graph's groups whose outputs the DiT reads, in mil.TextTaps' order
	parts map[string]string // the directories of the size's graphs (ironPart)
	vae   *anima.VAE
}

// openIron reads the IRON export in dir for pictures of w x h into p.
func (p *Pipeline) openIron(ctx context.Context, dev *hwnpu.Device, dir string, w, h int) error {
	if dev == nil {
		return errors.New("an IRON export runs on AMD's NPU only")
	}
	c, err := readIronConfig(dir)
	if err != nil {
		return err
	}
	ip := &ironPipe{dev: dev, parts: map[string]string{}}
	if sizes := IronSizes(dir); !slices.Contains(sizes, [2]int{w, h}) {
		return fmt.Errorf("%s draws %s, not %dx%d", filepath.Base(dir), sizeNames(sizes), w, h)
	}
	for _, part := range ironSizeParts {
		ip.parts[part] = ironPart(dir, c, w, h, part)
	}
	c.Width, c.Height = w, h
	ip.c = c
	tm, err := iron.ReadGraphManifest(filepath.Join(dir, "text"))
	if err != nil {
		return err
	}
	var t struct {
		Config ironText `json:"config"`
		Taps   []int    `json:"taps"`
	}
	if err := json.Unmarshal(tm.Meta, &t); err != nil {
		return err
	}
	if t.Config.Rows != c.TextSlot || t.Config.Hidden != c.Text || len(t.Taps) != len(mil.TextTaps) || c.Text*len(mil.TextTaps) != 20480 {
		return fmt.Errorf("text: %d rows of %d and %d hidden states for the DiT's %d of %d and %d", t.Config.Rows, t.Config.Hidden,
			len(t.Taps), c.TextSlot, c.Text, len(mil.TextTaps))
	}
	ip.text, ip.taps = t.Config, t.Taps
	vd := filepath.Join(dir, ip.part("vae"))
	if ip.vae, err = anima.OpenVAE(ctx, dev, vd); err != nil {
		return err
	}
	b, err := os.ReadFile(filepath.Join(vd, "manifest.json"))
	if err != nil {
		return err
	}
	var vm struct {
		Mean []float32 `json:"bn_mean"`
		Std  []float32 `json:"bn_std"`
	}
	if err := json.Unmarshal(b, &vm); err != nil || len(vm.Mean) != c.Latent || len(vm.Std) != c.Latent {
		return fmt.Errorf("vae: %d and %d batch-norm statistics for %d channels (%v)", len(vm.Mean), len(vm.Std), c.Latent, err)
	}
	// the manifest Pipeline's host parts read (Noise, Unpack, the modulations)
	p.m = &ProgramManifest{Kind: ProgramKind, Backend: "iron", Version: ExportVersion, FittedTables: true,
		Dit:  Config{Hidden: c.Hidden, HeadDim: c.HeadDim, Latent: c.Latent, Context: c.Text * len(mil.TextTaps), TimeIn: 256},
		Text: TextConfig{Hidden: t.Config.Hidden, HeadDim: t.Config.HeadDim, Theta: t.Config.Theta}, EmbedScale: 1,
		BNMean: vm.Mean, BNStd: vm.Std}
	if p.tok, err = tokenizer.Load(filepath.Join(dir, "text", "tokenizer.json")); err != nil {
		return err
	}
	if p.emb, err = os.Open(filepath.Join(dir, "text", "embed.bin")); err != nil {
		return err
	}
	p.iron = ip
	return nil
}

// part is the directory in the export of the graph name for the drawn size.
func (ip *ironPipe) part(name string) string {
	if d, ok := ip.parts[name]; ok {
		return d
	}
	return name
}

func sizeNames(sizes [][2]int) string {
	s := make([]string, len(sizes))
	for i, z := range sizes {
		s[i] = fmt.Sprintf("%dx%d", z[0], z[1])
	}
	return strings.Join(s, ", ")
}

func (p *Pipeline) openGraph(ctx context.Context, name string, swap, prefetch bool) (*iron.Graph, error) {
	return iron.OpenGraphWith(ctx, p.iron.dev, filepath.Join(p.dir, name), iron.GraphOptions{SwapVariants: swap, Prefetch: prefetch, Barrier: true})
}

func runGraph(ctx context.Context, g *iron.Graph, in map[string][]byte, values map[string]int64) ([]byte, error) {
	for k, v := range in {
		if err := g.SetInput(k, v); err != nil {
			return nil, err
		}
	}
	if err := g.Run(ctx, values); err != nil {
		return nil, err
	}
	outs := g.Outputs()
	b, err := g.Output(outs[len(outs)-1].Name)
	return append([]byte(nil), b...), err
}

// encodeIron is Encode on AMD's NPU: the text graph over the slot's rows (the prompt's tokens in the
// template, then pads), a group of Qwen3-VL's layers at a time (the prompt's rows causal, the pads
// attending to the prompt's keys only), and ctx of the eight hidden states.
func (p *Pipeline) encodeIron(ctx context.Context, prompt string) (Cond, error) {
	t0 := time.Now()
	ip := p.iron
	T := ip.text
	R, E := T.Rows, T.Hidden
	ids := p.Tokens(prompt)
	if n := len(ids); n > ip.c.MaxText() {
		return Cond{}, fmt.Errorf("flux3klein: the prompt is %d tokens in the chat template; the export takes up to %d", n, ip.c.MaxText())
	}
	rows := (len(ids) + ip.c.TextPad - 1) / ip.c.TextPad * ip.c.TextPad
	p.logf("text: %d tokens, %d rows", len(ids), rows)
	x := make([]byte, 2*R*E)
	for i := 0; i < R; i++ {
		id := int32(PadID)
		if i < len(ids) {
			id = ids[i]
		}
		if _, err := p.emb.ReadAt(x[2*i*E:2*(i+1)*E], int64(id)*int64(2*E)); err != nil {
			return Cond{}, fmt.Errorf("flux3klein: %w", err)
		}
	}
	W := T.Heads * T.HeadDim
	sel := make([]byte, 2*R*W)
	one := toBF16([]float32{1})
	for i := 0; i < len(ids)*W; i++ {
		copy(sel[2*i:], one)
	}
	angles := toBF16(ironTextAngles(mil.TextPositions(len(ids), R), T.HeadDim, T.Theta))
	g, err := p.openGraph(ctx, "text", true, false)
	if err != nil {
		return Cond{}, fmt.Errorf("flux3klein: %w", err)
	}
	taps := make([][]byte, len(ip.taps))
	for gi := 0; gi < max(g.Variants(), 1); gi++ {
		if g.Variants() > 0 {
			if err := g.Use(gi); err != nil {
				g.Close()
				return Cond{}, fmt.Errorf("flux3klein: %w", err)
			}
		}
		if x, err = runGraph(ctx, g, map[string][]byte{"x": x, "angles": angles, "sel": sel}, map[string]int64{"nk": int64(len(ids))}); err != nil {
			g.Close()
			return Cond{}, fmt.Errorf("flux3klein: text group %d: %w", gi, err)
		}
		for i, tap := range ip.taps {
			if tap == gi {
				taps[i] = x
			}
		}
	}
	g.Close()
	for i, t := range taps {
		if t == nil {
			return Cond{}, fmt.Errorf("flux3klein: the text graph has no group %d (hidden state %d)", ip.taps[i], i)
		}
	}
	t1 := time.Now()
	if p.Dump != "" { // the eight hidden states side by side, as tools/ref/ref_flux3klein.py's text rows
		n := len(taps)
		all := make([]float32, rows*n*E)
		for i, t := range taps {
			v := fromBF16s(t)
			for r := 0; r < rows; r++ {
				copy(all[(r*n+i)*E:(r*n+i+1)*E], v[r*E:(r+1)*E])
			}
		}
		p.dump("ctx.f32", all)
	}
	cg, err := p.openGraph(ctx, "ctx", false, false)
	if err != nil {
		return Cond{}, fmt.Errorf("flux3klein: %w", err)
	}
	defer cg.Close()
	in := map[string][]byte{}
	for i, t := range taps {
		in[fmt.Sprintf("h%d", i)] = t
	}
	stream, err := runGraph(ctx, cg, in, nil)
	if err != nil {
		return Cond{}, fmt.Errorf("flux3klein: ctx: %w", err)
	}
	p.logf("the text in %.2f s: Qwen3-VL %.2f s, the projection %.2f s", time.Since(t0).Seconds(), t1.Sub(t0).Seconds(), time.Since(t1).Seconds())
	return Cond{ironStream: stream, rows: rows, Tokens: len(ids)}, nil
}

// ironTextAngles is IRON's RoPE table (rows x dim, cos and sin interleaved) of Qwen's rotary
// embedding (two halves) at positions (iron_flux3klein.text_angles).
func ironTextAngles(positions []int, dim int, theta float64) []float32 {
	out := make([]float32, len(positions)*dim)
	for r, pos := range positions {
		for k := 0; k < dim/2; k++ {
			a := float64(pos) / math.Pow(theta, float64(2*k)/float64(dim))
			out[r*dim+2*k] = float32(math.Cos(a))
			out[r*dim+2*k+1] = float32(math.Sin(a))
		}
	}
	return out
}

// ironAngles is IRON's RoPE table of the DiT's rows at ids (iron_dit.rope_angles): each axis's
// frequencies theta^(-2k / d), the pair of frequency i at (2i, 2i + 1).
func (c IronConfig) ironAngles(ids []mil.Position) []float32 {
	var freqs []float64
	var axis []int
	for a, d := range c.Axes {
		for k := 0; k < d; k += 2 {
			freqs = append(freqs, 1/math.Pow(c.Theta, float64(k)/float64(d)))
			axis = append(axis, a)
		}
	}
	out := make([]float32, len(ids)*c.HeadDim)
	for r, pos := range ids {
		for i, f := range freqs {
			a := pos[axis[i]] * f
			out[r*c.HeadDim+2*i] = float32(math.Cos(a))
			out[r*c.HeadDim+2*i+1] = float32(math.Sin(a))
		}
	}
	return out
}

// ironIDs are the position ids of the picture's rows and of the text slot's (the image's pads
// past its patches: ImageRows, at 0).
func (c IronConfig) ironIDs() (img, txt []mil.Position) {
	gw := c.Width / 16
	img = make([]mil.Position, c.ImageRows())
	for r := range c.Tokens() {
		img[r] = mil.Position{0, float64(r/gw) * c.ImageRopeScale, float64(r%gw) * c.ImageRopeScale, 0}
	}
	txt = make([]mil.Position, c.TextSlot)
	for i := range txt {
		txt[i] = mil.Position{0, 0, 0, float64(i)}
	}
	return img, txt
}

// generateIron is Generate on AMD's NPU: each step the text's mode blocks over its slot (the keys
// the prompt's rows), the latents embedded (dit_in, the image's rows padded with zeros to whole
// rounds: ImageRows) and the image's mode blocks (the keys its patches), the single blocks over
// [image rows, text slot] (the keys the image's patches and the prompt's), a group at a time (its
// weights read into the bank), and norm_out and proj_out of the image's rows (dit_out), of which
// the patches' are read.
func (p *Pipeline) generateIron(ctx context.Context, cond Cond, noise []float32, sigmas []float64, step StepFunc) ([]float32, error) {
	ip := p.iron
	c := ip.c
	E, N, P, L := c.Hidden, c.Tokens(), c.ImageRows(), c.Latent
	if cond.ironStream == nil {
		return nil, errors.New("flux3klein: the DiT runs without a prompt (Encode)")
	}
	if len(noise) != N*L {
		return nil, fmt.Errorf("flux3klein: %d noise values for latents of %d", len(noise), N*L)
	}
	mods, err := p.modulations(sigmas)
	if err != nil {
		return nil, fmt.Errorf("flux3klein: %w", err)
	}
	names := []string{ip.part("dit_in"), "text_modes", ip.part("image_modes"), ip.part("singles"), ip.part("dit_out")}
	swap := []bool{false, true, true, true, false}
	prefetch := anima.PrefetchFits(ip.vae, p.dir, names, names[1], names[2], names[3])
	gs := make([]*iron.Graph, len(names))
	defer func() {
		for _, g := range gs {
			if g != nil {
				g.Close()
			}
		}
	}()
	for i, n := range names {
		if gs[i], err = p.openGraph(ctx, n, swap[i], prefetch && swap[i]); err != nil {
			return nil, fmt.Errorf("flux3klein: %w", err)
		}
	}
	in, textModes, imageModes, singles, out := gs[0], gs[1], gs[2], gs[3], gs[4]
	imgIDs, txtIDs := c.ironIDs()
	txtAngles := toBF16(c.ironAngles(txtIDs))
	imgAngles := toBF16(c.ironAngles(imgIDs))
	jointAngles := toBF16(c.ironAngles(append(slices.Clone(imgIDs), txtIDs...)))
	var tRead, tRun time.Duration
	tLoop := time.Now()
	// blocks runs g's groups over h in place
	blocks := func(g *iron.Graph, h, m, angles []byte, nk int, what string) ([]byte, error) {
		for gi := 0; gi < max(g.Variants(), 1); gi++ {
			if g.Variants() > 0 {
				t := time.Now()
				if err := g.Use(gi); err != nil {
					return nil, err
				}
				tRead += time.Since(t)
			}
			t := time.Now()
			var err error
			if h, err = runGraph(ctx, g, map[string][]byte{"x": h, "mods": m, "angles": angles}, map[string]int64{"nk": int64(nk)}); err != nil {
				return nil, fmt.Errorf("%s group %d: %w", what, gi, err)
			}
			tRun += time.Since(t)
		}
		return h, nil
	}
	x := slices.Clone(noise)
	n := len(sigmas) - 1
	for i := 0; i < n; i++ {
		t0 := time.Now()
		rows := func(lo, hi int) []float32 { return mods[i][lo*E : hi*E] }
		txt, err := blocks(textModes, cond.ironStream, toBF16(rows(mil.ModTextEarly, mil.ModTextEarly+3)), txtAngles, cond.rows, "text mode blocks")
		if err != nil {
			return nil, fmt.Errorf("flux3klein: step %d: %w", i, err)
		}
		xin := make([]byte, 2*P*L) // the patches, then the pads' zeros
		copy(xin, toBF16(x))
		img, err := runGraph(ctx, in, map[string][]byte{"x": xin}, nil)
		if err != nil {
			return nil, fmt.Errorf("flux3klein: step %d: dit_in: %w", i, err)
		}
		if img, err = blocks(imageModes, img, toBF16(rows(mil.ModImageEarly, mil.ModImageEarly+3)), imgAngles, N, "image mode blocks"); err != nil {
			return nil, fmt.Errorf("flux3klein: step %d: %w", i, err)
		}
		if len(img) != 2*P*E || len(txt) != 2*c.TextSlot*E {
			return nil, fmt.Errorf("flux3klein: step %d: the mode blocks give %d and %d bytes for %d and %d rows", i, len(img), len(txt), P, c.TextSlot)
		}
		sm := append(toBF16(rows(mil.ModImageSingle, mil.ModImageSingle+3)), toBF16(rows(mil.ModTextSingle, mil.ModTextSingle+3))...)
		h, err := blocks(singles, append(img, txt...), sm, jointAngles, N+cond.rows, "single blocks")
		if err != nil {
			return nil, fmt.Errorf("flux3klein: step %d: %w", i, err)
		}
		vb, err := runGraph(ctx, out, map[string][]byte{"x": h[:2*P*E], "scale": toBF16(rows(mil.ModOut, mil.ModOut+1)),
			"shift": toBF16(rows(mil.ModOut+1, mil.ModOut+2))}, nil)
		if err != nil {
			return nil, fmt.Errorf("flux3klein: step %d: dit_out: %w", i, err)
		}
		v := fromBF16s(vb[:2*N*L])
		p.dump(fmt.Sprintf("v_%d.f32", i), v)
		dt := float32(sigmas[i+1] - sigmas[i])
		for j := range x {
			x[j] += dt * v[j]
		}
		if step != nil {
			step(i, n, time.Since(t0))
		}
	}
	all := time.Since(tLoop)
	p.logf("the steps' %.2f s: the blocks %.2f s, the groups' weights %.2f s, the rest %.2f s",
		all.Seconds(), tRun.Seconds(), tRead.Seconds(), (all - tRun - tRead).Seconds())
	return x, nil
}

// decodeIron is Decode on AMD's NPU: the VAE's stages, one picture.
func (p *Pipeline) decodeIron(ctx context.Context, x []float32, stage func(i, n int, d time.Duration)) (*image.RGBA, error) {
	p.iron.vae.Stage = stage
	img, err := p.iron.vae.Decode(ctx, p.Unpack(x))
	if err != nil {
		return nil, fmt.Errorf("flux3klein: vae: %w", err)
	}
	return img, nil
}

func toBF16(xs []float32) []byte {
	b := make([]byte, 2*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint16(b[2*i:], bf16(x))
	}
	return b
}
