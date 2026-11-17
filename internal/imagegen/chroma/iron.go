package chroma

import (
	"context"
	"encoding/json"
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
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// isIronExport reports whether dir is an export of tools/amd/iron_chroma.py.
func isIronExport(dir string) bool {
	for _, p := range []string{"doubles/manifest.json", "singles/manifest.json", "mods/manifest.json", "t5/manifest.json"} {
		if _, err := os.Stat(filepath.Join(dir, p)); err != nil {
			return false
		}
	}
	return true
}

// ironEngine runs the IRON graphs of an export of tools/amd/iron_chroma.py on AMD's NPU.
type ironEngine struct {
	C     Config
	dev   *hwnpu.Device
	dir   string
	t5dim int
	vae   *anima.VAE
	logf  func(format string, args ...any)
	d     *dit
	// sfx and singles are the names' suffixes of the picture's size (iron_chroma.export_dit):
	// "" for the export's first size, else _WxH (singles "" when its rows are the first size's)
	sfx, singles string
	angles       []byte // the blocks' input of the rows' RoPE; nil for an export that holds its own
}

// SizesDir is the directory of an IRON export's VAEs for pictures of another size than its first
// (tools/amd/iron_chroma.py --size 1024 512 832x1248 ...): SizesDir/WxH/vae; the DiT's graphs of
// that size are dit_in_WxH, doubles_WxH, dit_out_WxH and singles_WxH or the first size's.
const SizesDir = "sizes"

// IronSizes are the pictures' sizes (width, height) the IRON export in dir draws: its first size,
// then those with a VAE in SizesDir and their doubles; nil when dir is no IRON export.
func IronSizes(dir string) [][2]int {
	if !isIronExport(dir) {
		return nil
	}
	c, err := ironConfig(dir, "doubles")
	if err != nil {
		return nil
	}
	out := [][2]int{{c.Width, c.Height}}
	entries, _ := os.ReadDir(filepath.Join(dir, SizesDir))
	for _, en := range entries {
		var w, h int
		if _, err := fmt.Sscanf(en.Name(), "%dx%d", &w, &h); err != nil || fmt.Sprintf("%dx%d", w, h) != en.Name() {
			continue
		}
		_, ev := os.Stat(filepath.Join(dir, SizesDir, en.Name(), "vae", "manifest.json"))
		_, ed := os.Stat(filepath.Join(dir, "doubles_"+en.Name(), "manifest.json"))
		if ev == nil && ed == nil {
			out = append(out, [2]int{w, h})
		}
	}
	return out
}

// ironConfig is the DiT's shape in the meta of the export's graph name.
func ironConfig(dir, name string) (Config, error) {
	m, err := iron.ReadGraphManifest(filepath.Join(dir, name))
	if err != nil {
		return Config{}, err
	}
	var x struct {
		Config Config `json:"config"`
	}
	err = json.Unmarshal(m.Meta, &x)
	return x.Config, err
}

func openIron(ctx context.Context, dev *hwnpu.Device, dir string, w, h int, c *Config, model *Model, logf func(string, ...any)) (*ironEngine, error) {
	e := &ironEngine{dev: dev, dir: dir, logf: logf}
	base, err := ironConfig(dir, "doubles")
	if err != nil {
		return nil, err
	}
	cfg, vae := base, filepath.Join(dir, "vae")
	if w != 0 && (w != base.Width || h != base.Height) {
		size := fmt.Sprintf("%dx%d", w, h)
		if !slices.Contains(IronSizes(dir), [2]int{w, h}) {
			var have []string
			for _, s := range IronSizes(dir) {
				have = append(have, fmt.Sprintf("%dx%d", s[0], s[1]))
			}
			return nil, fmt.Errorf("the export draws %s, not %s", strings.Join(have, ", "), size)
		}
		e.sfx = "_" + size
		if cfg, err = ironConfig(dir, "doubles"+e.sfx); err != nil {
			return nil, err
		}
		if cfg.GraphRows() != base.GraphRows() {
			e.singles = e.sfx
		}
		vae = filepath.Join(dir, SizesDir, size, "vae")
	}
	e.C, *c = cfg, cfg
	m, err := iron.ReadGraphManifest(filepath.Join(dir, "doubles"+e.sfx))
	if err != nil {
		return nil, err
	}
	for _, in := range m.Graph.Inputs {
		if in.Name == "angles" { // the RoPE an input (an export before it holds its own)
			e.angles = toBF16(ropeAngles(cfg))
		}
	}
	*model = Model{Name: "Chroma1-Flash", Steps: 8, Guidance: 1, Sampler: "heun"}
	if b, err := os.ReadFile(filepath.Join(dir, "model.json")); err == nil {
		if err := json.Unmarshal(b, model); err != nil {
			return nil, err
		}
	}
	tm, err := iron.ReadGraphManifest(filepath.Join(dir, "t5"))
	if err != nil {
		return nil, err
	}
	var t struct {
		Config struct {
			Hidden int `json:"hidden"`
			Rows   int `json:"rows"`
		} `json:"config"`
	}
	if err := json.Unmarshal(tm.Meta, &t); err != nil || t.Config.Rows != c.TextRows {
		return nil, fmt.Errorf("t5: %d rows for the DiT's %d (%v)", t.Config.Rows, c.TextRows, err)
	}
	e.t5dim = t.Config.Hidden
	if e.vae, err = anima.OpenVAE(ctx, dev, vae); err != nil {
		return nil, err
	}
	return e, nil
}

func (e *ironEngine) textDir() string { return filepath.Join(e.dir, "t5") }

func (e *ironEngine) close() error { return e.closeDiT() }

func (e *ironEngine) open(ctx context.Context, name string, swap bool) (*iron.Graph, error) {
	return e.openWith(ctx, name, swap, false)
}

// openWith is open with iron.GraphOptions.Prefetch (anima.PrefetchFits decides).
func (e *ironEngine) openWith(ctx context.Context, name string, swap, prefetch bool) (*iron.Graph, error) {
	return iron.OpenGraphWith(ctx, e.dev, filepath.Join(e.dir, name), iron.GraphOptions{SwapVariants: swap, Prefetch: prefetch, Barrier: true})
}

// encode runs T5 on each prompt. An export with T5 in groups of layers (iron_chroma_parts.py's
// T5Config.group) holds one ELF and a bank image per group, read into one bank in turn: every
// prompt runs a group before the next is read, so each group's weights are read once and T5's 5
// GB of q8 is never pinned at once. Each group gives its residual (the next group's x) and the
// final norm of it (the last group's is T5's output). An export of one graph gives the norm only.
func (e *ironEngine) encode(ctx context.Context, xs [][]float32, ns []int) ([]Cond, error) {
	t0 := time.Now()
	R := e.C.TextRows
	g, err := e.openWith(ctx, "t5", true, anima.PrefetchFits(e.vae, e.dir, []string{"t5"}, "t5"))
	if err != nil {
		return nil, err
	}
	in := make([][]byte, len(xs))
	masks := make([][]byte, len(xs))
	for k, x := range xs {
		mask := make([]float32, R)
		for i := 0; i < ns[k]; i++ {
			mask[i] = 1
		}
		in[k], masks[k] = toBF16(x), toBF16(mask)
	}
	groups := max(g.Variants(), 1)
	if groups > 1 && len(g.Outputs()) != 2 {
		g.Close()
		return nil, fmt.Errorf("chroma: t5: %d groups but %d outputs (the residual and the norm)", groups, len(g.Outputs()))
	}
	hs := make([][]byte, len(xs))
	for gi := 0; gi < groups; gi++ {
		if g.Variants() > 0 {
			if err := g.Use(gi); err != nil {
				g.Close()
				return nil, err
			}
		}
		for k := range xs {
			if err := run(ctx, g, map[string][]byte{"x": in[k], "mask": masks[k]}, nil); err != nil {
				g.Close()
				return nil, fmt.Errorf("chroma: t5 group %d: %w", gi, err)
			}
			last := len(g.Outputs()) - 1 // the norm
			if gi < groups-1 {
				in[k], err = output(g, 0) // the residual
			} else {
				hs[k], err = output(g, last)
			}
			if err != nil {
				g.Close()
				return nil, err
			}
		}
	}
	g.Close()
	t1 := time.Now()
	c, err := e.open(ctx, "ctx", false)
	if err != nil {
		return nil, err
	}
	defer c.Close()
	conds := make([]Cond, len(xs))
	for k := range xs {
		if err := run(ctx, c, map[string][]byte{"x": hs[k]}, nil); err != nil {
			return nil, err
		}
		out, err := output(c, 0)
		if err != nil {
			return nil, err
		}
		conds[k] = Cond{ctx: out}
	}
	e.logf("the text in %.2f s: T5-XXL %.2f s, the projection %.2f s", time.Since(t0).Seconds(), t1.Sub(t0).Seconds(), time.Since(t1).Seconds())
	return conds, nil
}

func run(ctx context.Context, g *iron.Graph, in map[string][]byte, values map[string]int64) error {
	for k, v := range in {
		if err := g.SetInput(k, v); err != nil {
			return err
		}
	}
	return g.Run(ctx, values)
}

func output(g *iron.Graph, i int) ([]byte, error) {
	b, err := g.Output(g.Outputs()[i].Name)
	return append([]byte(nil), b...), err
}

// dit holds the DiT's graphs while the steps run.
type dit struct {
	mods, in, doubles, singles, out *iron.Graph
}

func (d *dit) close() {
	for _, g := range []*iron.Graph{d.mods, d.in, d.doubles, d.singles, d.out} {
		if g != nil {
			g.Close()
		}
	}
}

func (e *ironEngine) openDiT(ctx context.Context, _ []float64) error {
	d := &dit{}
	var err error
	stage := []string{"mods", "dit_in", "doubles", "singles", "dit_out"}
	prefetch := anima.PrefetchFits(e.vae, e.dir, stage, "doubles", "singles")
	for _, x := range []struct {
		g    **iron.Graph
		name string
		swap bool
	}{{&d.mods, "mods", false}, {&d.in, "dit_in" + e.sfx, false}, {&d.doubles, "doubles" + e.sfx, true},
		{&d.singles, "singles" + e.singles, true}, {&d.out, "dit_out" + e.sfx, false}} {
		if *x.g, err = e.openWith(ctx, x.name, x.swap, prefetch && x.swap); err != nil {
			d.close()
			return err
		}
	}
	if e.angles != nil {
		for _, g := range []*iron.Graph{d.doubles, d.singles} {
			if err := g.SetInput("angles", e.angles); err != nil {
				d.close()
				return err
			}
		}
	}
	e.d = d
	return nil
}

func (e *ironEngine) closeDiT() error {
	if e.d != nil {
		e.d.close()
		e.d = nil
	}
	return nil
}

// velocity runs the DiT on the latents x (tokens x 64) at sigma for each conditioning, back to
// back in each group of blocks (the weights read once): the velocities.
func (e *ironEngine) velocity(ctx context.Context, conds []Cond, x []float32, sigma float64) ([][]float32, error) {
	c := e.C
	d := e.d
	E, N := c.Hidden, c.Tokens()
	if err := run(ctx, d.mods, map[string][]byte{"x": toBF16(ModInput(1000*sigma, c.Mods, c.ModRows, c.ApproxIn))}, nil); err != nil {
		return nil, err
	}
	mods, err := output(d.mods, 0)
	if err != nil {
		return nil, err
	}
	row := func(k int) []byte { return mods[2*k*E : 2*(k+1)*E] }
	// dit_in and dit_out take the image's rows in whole 256s (iron_chroma.ChromaConfig.img_rows):
	// zeros past the tokens, their outputs dropped
	np := inputRows(d.in, N)
	xin := toBF16(x)
	xin = append(xin, make([]byte, 2*(np-N)*c.Latent)...)
	if err := run(ctx, d.in, map[string][]byte{"x": xin}, nil); err != nil {
		return nil, err
	}
	img, err := output(d.in, 0)
	if err != nil {
		return nil, err
	}
	img = img[:2*N*E]
	hs := make([][]byte, len(conds))
	pad := 0
	if e.angles != nil { // the padding to the graphs' rows (zeros, its keys past nk)
		pad = c.GraphRows() - c.Rows()
	}
	for k, cd := range conds {
		hs[k] = append(append(append([]byte(nil), img...), cd.ctx...), make([]byte, 2*pad*E)...) // [image rows, text rows, padding]
	}
	blocks := func(g *iron.Graph, group, nmods int, rows func(b int) []int, out int, what string) error {
		for gi := 0; gi < g.Variants(); gi++ {
			var m []byte
			for j := 0; j < group; j++ {
				for _, r := range rows(gi*group + j) {
					m = append(m, row(r)...)
				}
			}
			if err := g.Use(gi); err != nil {
				return err
			}
			for k, cd := range conds {
				if err := run(ctx, g, map[string][]byte{"x": hs[k], "mods": m}, map[string]int64{"nk": int64(N + cd.Keys)}); err != nil {
					return fmt.Errorf("%s group %d: %w", what, gi, err)
				}
				o := out
				if o < 0 {
					o = len(g.Outputs()) - 1
				}
				if hs[k], err = output(g, o); err != nil {
					return err
				}
			}
		}
		return nil
	}
	err = blocks(d.doubles, c.DoubleGroup, 12, func(b int) []int {
		var r []int // the image's rows, then the text's
		for i := 0; i < 6; i++ {
			r = append(r, 3*c.Singles+6*b+i)
		}
		for i := 0; i < 6; i++ {
			r = append(r, 3*c.Singles+6*c.Doubles+6*b+i)
		}
		return r
	}, -1, "double blocks")
	if err == nil {
		err = blocks(d.singles, c.SingleGroup, 3, func(b int) []int { return []int{3 * b, 3*b + 1, 3*b + 2} }, 0, "single blocks")
	}
	if err != nil {
		return nil, err
	}
	last := c.Mods - 2
	vs := make([][]float32, len(conds))
	for k := range conds {
		xo := hs[k][:2*N*E]
		if no := inputRows(d.out, N); no > N {
			xo = append(append([]byte(nil), xo...), make([]byte, 2*(no-N)*E)...)
		}
		if err := run(ctx, d.out, map[string][]byte{"x": xo, "shift": row(last), "scale": row(last + 1)}, nil); err != nil {
			return nil, err
		}
		v, err := output(d.out, 0)
		if err != nil {
			return nil, err
		}
		vs[k] = fromBF16s(v)[:N*c.Latent]
	}
	return vs, nil
}

func (e *ironEngine) decode(ctx context.Context, z []float32, stage anima.StageFunc) (*image.RGBA, error) {
	e.vae.Stage = stage
	return e.vae.Decode(ctx, z)
}

// ropeAngles is the blocks' input angles (GraphRows x HeadDim, cos and sin interleaved: IRON's
// RoPE) of the image's rows at (0, row, column), then the text's and the padding's at (0, 0, 0)
// (iron_chroma.angles, iron_qwenimage.rope_angles).
func ropeAngles(c Config) []float32 {
	var freqs []float64
	var axis []int
	for a, d := range c.Axes {
		for k := 0; k < d; k += 2 {
			freqs = append(freqs, 1/math.Pow(c.Theta, float64(k)/float64(d)))
			axis = append(axis, a)
		}
	}
	out := make([]float32, c.GraphRows()*c.HeadDim)
	for r := range c.GraphRows() {
		pos := [3]float64{}
		if r < c.Tokens() {
			pos = [3]float64{0, float64(r / c.GW()), float64(r % c.GW())}
		}
		for i, f := range freqs {
			a := pos[axis[i]] * f
			out[r*c.HeadDim+2*i] = float32(math.Cos(a))
			out[r*c.HeadDim+2*i+1] = float32(math.Sin(a))
		}
	}
	return out
}

// inputRows is the rows of g's input x, n when it says none.
func inputRows(g *iron.Graph, n int) int {
	for _, in := range g.Inputs() {
		if in.Name == "x" && len(in.Shape) > 0 && in.Shape[0] > n {
			return in.Shape[0]
		}
	}
	return n
}
