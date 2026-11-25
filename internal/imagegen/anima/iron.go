package anima

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

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/llm/iron"
)

// ironEngine runs the IRON graphs of tools/amd/iron_anima.py on AMD's NPU: text, dit_in, the
// groups of dit_blocks (one ELF, a bank per group), dit_out and the VAE's stages. The graphs
// take and give bf16.
//
// A graph's banks are pinned host memory the process's resident size does not count, and
// the three stages of a picture never run together, so each is on the device only while it
// runs: the text graph for the conditioning, the DiT for the steps, the VAE a stage at a time.
// That keeps the peak to the DiT's (4.9 GiB at 1024 x 1024) instead of all of them (14.2 GiB);
// loading costs a few seconds a picture, from the page cache.
// ResidentDiT keeps every group of the DiT's blocks on the NPU (7 banks at 1024 x 1024); by
// default one bank is kept and each group's weights are read into it before the group runs
// (iron.GraphOptions.SwapVariants).
var ResidentDiT = false

type ironEngine struct {
	dev                      *bonsaiai.Device
	dir                      string
	dit                      DitConfig
	// sfx names the DiT's graphs of the picture's rows (rows: its patches padded to whole
	// rounds of 512; "" for the export's first size, else _nROWS), sized the directory of its
	// VAE ("" for the first size's vae, else SizesDir/WxH)
	sfx, sized   string
	tokens, rows int
	angles       []byte // dit_blocks' RoPE (bf16), nil for an export that holds its own
	ditIn, ditBlocks, ditOut *iron.Graph // open while the steps run
	ctxSet                   []float32   // the conditioning dit_blocks holds
	vae                      *VAE
	checkpoint               string // the diffusion model of the export (dit_blocks' meta)
}

func meta[T any](m *iron.GraphManifest) (T, error) {
	var x struct {
		Config T `json:"config"`
	}
	err := json.Unmarshal(m.Meta, &x)
	return x.Config, err
}

// SizesDir is the directory of an IRON export's VAEs for pictures of another size than its first
// (tools/amd/iron_anima.py --size 1024 512 832x1248 ...): SizesDir/WxH/vae. The DiT's graphs
// depend on the picture's rows alone (its patches padded to whole rounds of 512, the RoPE an
// input): named after the first size's (none) or _nROWS, on one set of banks.
const SizesDir = "sizes"

// IronSizes are the pictures' sizes (width, height) the IRON export in dir draws: its first size,
// then those of SizesDir with a VAE; nil when dir is no IRON export.
func IronSizes(dir string) [][2]int {
	m, err := iron.ReadGraphManifest(filepath.Join(dir, "dit_blocks"))
	if err != nil {
		return nil
	}
	dc, err := meta[DitConfig](m)
	if err != nil {
		return nil
	}
	out := [][2]int{{dc.Width, dc.Height}}
	entries, _ := os.ReadDir(filepath.Join(dir, SizesDir))
	for _, e := range entries {
		var w, h int
		if _, err := fmt.Sscanf(e.Name(), "%dx%d", &w, &h); err != nil || fmt.Sprintf("%dx%d", w, h) != e.Name() {
			continue
		}
		if _, err := os.Stat(filepath.Join(dir, SizesDir, e.Name(), "vae", "manifest.json")); err == nil {
			out = append(out, [2]int{w, h})
		}
	}
	return out
}

// graphRows is the DiT's graphs' rows for tokens patches: tokens, or whole rounds of the
// attention's 512 (iron_anima.DitConfig.rows).
func graphRows(tokens int) int {
	if tokens%512 == 0 {
		return tokens
	}
	return (tokens + 511) / 512 * 512
}

// openIron reads the export's manifests for pictures of w x h (0 x 0: its first size); the graphs
// are loaded when they run.
func openIron(ctx context.Context, dev *bonsaiai.Device, dir string, w, h int) (*ironEngine, DitConfig, TextConfig, error) {
	e := &ironEngine{dev: dev, dir: dir}
	var dc DitConfig
	var tc TextConfig
	fail := func(err error) (*ironEngine, DitConfig, TextConfig, error) {
		return nil, dc, tc, err
	}
	bm, err := iron.ReadGraphManifest(filepath.Join(dir, "dit_blocks"))
	if err != nil {
		return fail(err)
	}
	if dc, err = meta[DitConfig](bm); err != nil {
		return fail(err)
	}
	gh, gw := dc.Grid()
	baseRows := gh * gw
	if (w != 0 || h != 0) && (w != dc.Width || h != dc.Height) {
		sizes := IronSizes(dir)
		if !slices.Contains(sizes, [2]int{w, h}) {
			var names []string
			for _, s := range sizes {
				names = append(names, fmt.Sprintf("%dx%d", s[0], s[1]))
			}
			return fail(fmt.Errorf("%s draws %s, not %dx%d", filepath.Base(dir), strings.Join(names, ", "), w, h))
		}
		dc.Width, dc.Height = w, h
		e.sized = filepath.Join(SizesDir, fmt.Sprintf("%dx%d", w, h))
	}
	gh, gw = dc.Grid()
	e.tokens = gh * gw
	e.rows = graphRows(e.tokens)
	if e.rows != baseRows {
		e.sfx = fmt.Sprintf("_n%d", e.rows)
	}
	ms := map[string]*iron.GraphManifest{}
	for _, name := range []string{"text", "dit_in" + e.sfx, "dit_blocks" + e.sfx, "dit_out" + e.sfx} {
		m, err := iron.ReadGraphManifest(filepath.Join(dir, name))
		if err != nil {
			return fail(err)
		}
		ms[strings.TrimSuffix(name, e.sfx)] = m
	}
	for _, in := range ms["dit_blocks"].Graph.Inputs {
		if in.Name == "angles" { // the RoPE an input (an export before it holds its own)
			e.angles = toBF16(ropeAngles(dc, e.rows))
		}
	}
	if e.angles == nil && e.rows != e.tokens {
		return fail(fmt.Errorf("%dx%d: %d patches, and the export's DiT takes no pads (made before them)", w, h, e.tokens))
	}
	var ck struct {
		Checkpoint string `json:"checkpoint"`
	}
	if err := json.Unmarshal(ms["dit_blocks"].Meta, &ck); err != nil {
		return fail(err)
	}
	e.checkpoint = ck.Checkpoint
	if tc, err = meta[TextConfig](ms["text"]); err != nil {
		return fail(err)
	}
	e.dit = dc
	if n, v := dc.Layers/dc.Group, max(1, len(ms["dit_blocks"].Variants)); v != n {
		return fail(fmt.Errorf("dit_blocks holds %d groups, the DiT %d", v, n))
	}
	vae := filepath.Join(dir, e.sized, "vae")
	if _, err := os.Stat(filepath.Join(vae, "manifest.json")); err == nil {
		if e.vae, err = OpenVAE(ctx, dev, vae); err != nil {
			return fail(err)
		}
	}
	return e, dc, tc, nil
}

func (e *ironEngine) close() error { return e.closeDiT() }

// closeDiT releases the DiT's graphs, which the next velocity call loads again.
func (e *ironEngine) closeDiT() error {
	var errs []error
	for _, g := range []**iron.Graph{&e.ditIn, &e.ditBlocks, &e.ditOut} {
		if *g != nil {
			errs = append(errs, (*g).Close())
			*g = nil
		}
	}
	return errors.Join(errs...)
}

// openDiT loads the DiT's graphs and gives them the conditioning.
func (e *ironEngine) openDiT(ctx context.Context) error {
	if e.ditBlocks != nil {
		return nil
	}
	s := e.sfx
	prefetch := !ResidentDiT && PrefetchFits(e.vae, e.dir, []string{"dit_in" + s, "dit_blocks" + s, "dit_out" + s}, "dit_blocks"+s)
	for _, x := range []struct {
		g    **iron.Graph
		name string
	}{{&e.ditIn, "dit_in" + s}, {&e.ditBlocks, "dit_blocks" + s}, {&e.ditOut, "dit_out" + s}} {
		g, err := iron.OpenGraphWith(ctx, e.dev, filepath.Join(e.dir, x.name), iron.GraphOptions{SwapVariants: !ResidentDiT, Prefetch: prefetch})
		if err != nil {
			e.closeDiT()
			return err
		}
		*x.g = g
	}
	if e.angles != nil {
		if err := e.ditBlocks.SetInput("angles", e.angles); err != nil {
			e.closeDiT()
			return err
		}
	}
	e.ctxSet = nil
	return nil
}

func toBF16(xs []float32) []byte {
	b := make([]byte, 2*len(xs))
	for i, x := range xs {
		binary.LittleEndian.PutUint16(b[2*i:], bf16(x))
	}
	return b
}

func fromBF16s(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		out[i] = fromBF16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return out
}

// text loads the text graph, runs it and releases it.
func (e *ironEngine) text(ctx context.Context, qx, tx []float32, nq, nt int) ([]float32, error) {
	g, err := iron.OpenGraph(ctx, e.dev, filepath.Join(e.dir, "text"))
	if err != nil {
		return nil, err
	}
	defer g.Close()
	if err := g.SetInput("qx", toBF16(qx)); err != nil {
		return nil, err
	}
	if err := g.SetInput("tx", toBF16(tx)); err != nil {
		return nil, err
	}
	if err := g.Run(ctx, map[string]int64{"nq": int64(nq), "nt": int64(nt)}); err != nil {
		return nil, err
	}
	out, err := g.Output(g.Outputs()[0].Name)
	if err != nil {
		return nil, err
	}
	return fromBF16s(out), nil // a copy: the graph's memory goes with Close
}

// velocity runs dit_in once (it does not read the conditioning), each group of dit_blocks on
// each conditioning before the next group's weights are read, and dit_out on each, loading the
// DiT on the first call.
func (e *ironEngine) velocity(ctx context.Context, patches, tproj []float32, conds [][]float32) ([][]float32, error) {
	if err := e.openDiT(ctx); err != nil {
		return nil, err
	}
	pb := toBF16(patches)
	if pad := 2 * 64 * e.rows; len(pb) < pad { // the pads' rows: zeros
		pb = append(pb, make([]byte, pad-len(pb))...)
	}
	if err := e.ditIn.SetInput("patches", pb); err != nil {
		return nil, err
	}
	if err := e.ditIn.SetInput("tproj", toBF16(tproj)); err != nil {
		return nil, err
	}
	if err := e.ditIn.Run(ctx, nil); err != nil {
		return nil, err
	}
	outs := e.ditIn.Outputs() // x, temb, et
	var x, temb, et []byte
	var err error
	if x, err = e.ditIn.Output(outs[0].Name); err != nil {
		return nil, err
	}
	if temb, err = e.ditIn.Output(outs[1].Name); err != nil {
		return nil, err
	}
	if et, err = e.ditIn.Output(outs[2].Name); err != nil {
		return nil, err
	}
	for _, in := range []struct {
		name string
		b    []byte
	}{{"temb", temb}, {"et", et}} {
		if err := e.ditBlocks.SetInput(in.name, in.b); err != nil {
			return nil, err
		}
		if err := e.ditOut.SetInput(in.name, in.b); err != nil {
			return nil, err
		}
	}
	rows := make([]map[string]int64, len(conds))
	for k, cond := range conds {
		rows[k] = map[string]int64{"nt": int64(condRows(cond, e.dit.TextDim))}
		if e.angles != nil {
			rows[k]["nk"] = int64(e.tokens)
		}
	}
	xs := make([][]byte, len(conds))
	for k := range xs {
		xs[k] = append([]byte(nil), x...) // Output's bytes are the graph's, which the next run writes
	}
	for g := 0; g < e.ditBlocks.Variants(); g++ {
		if err := e.ditBlocks.Use(g); err != nil {
			return nil, err
		}
		for k, cond := range conds {
			if len(e.ctxSet) == 0 || &e.ctxSet[0] != &cond[0] {
				if err := e.ditBlocks.SetInput("ctx", toBF16(cond)); err != nil {
					return nil, err
				}
				e.ctxSet = cond
			}
			if err := e.ditBlocks.SetInput("x", xs[k]); err != nil {
				return nil, err
			}
			if err := e.ditBlocks.Run(ctx, rows[k]); err != nil {
				return nil, fmt.Errorf("blocks %d..%d: %w", g*e.dit.Group, (g+1)*e.dit.Group-1, err)
			}
			out, err := e.ditBlocks.Output(e.ditBlocks.Outputs()[0].Name)
			if err != nil {
				return nil, err
			}
			xs[k] = append(xs[k][:0], out...)
		}
	}
	vs := make([][]float32, len(conds))
	for k := range xs {
		if err := e.ditOut.SetInput("x", xs[k]); err != nil {
			return nil, err
		}
		if err := e.ditOut.Run(ctx, nil); err != nil {
			return nil, err
		}
		v, err := e.ditOut.Output(e.ditOut.Outputs()[0].Name)
		if err != nil {
			return nil, err
		}
		vs[k] = fromBF16s(v[:min(len(v), 2*64*e.tokens)]) // the patches' rows, not the pads'
	}
	return vs, nil
}

// condRows is the rows of a conditioning (rows of dim) before its trailing zero rows, at least 1:
// dit_blocks' per-call nt. An export with --dit-cross trim attends those rows and counts the
// zero rows' keys (zeros too) in the softmax, so the output is the same as over every row; an
// export before it takes no nt.
func condRows(cond []float32, dim int) int {
	if dim <= 0 {
		return len(cond)
	}
	n := len(cond) / dim
	for ; n > 1; n-- {
		for _, x := range cond[(n-1)*dim : n*dim] {
			if x != 0 {
				return n
			}
		}
	}
	return 1
}

func (e *ironEngine) hasVAE() bool { return e.vae != nil }

// decode releases the DiT and runs the VAE, which loads a stage at a time.
func (e *ironEngine) decode(ctx context.Context, latents []float32, stage StageFunc) (*image.RGBA, error) {
	if err := e.closeDiT(); err != nil {
		return nil, err
	}
	e.vae.Stage = stage
	return e.vae.Decode(ctx, latents)
}

// ropeAngles is dit_blocks' angles (rows x HeadDim, cos and sin interleaved: IRON's RoPE) for the
// picture of c: CosmosRotaryPosEmbed's, pair i of a head turned by the frame (all at 0), then the
// row, then the column, each axis at its NTK-scaled theta (iron_anima.rope_table); the rows past
// the patches no turn (iron_anima.angles_input).
func ropeAngles(c DitConfig, rows int) []float32 {
	D := c.HeadDim
	dimH := D / 6 * 2
	dimW := dimH
	dimT := D - dimH - dimW
	scale := func(i int) float64 {
		if i < len(c.RopeScale) {
			return c.RopeScale[i]
		}
		return 1
	}
	theta := func(s float64, dim int) float64 { return 10000 * math.Pow(s, float64(dim)/float64(dim-2)) }
	freqs := func(th float64, dim int) []float64 {
		f := make([]float64, dim/2)
		for i := range f {
			f[i] = 1 / math.Pow(th, float64(2*i)/float64(dim))
		}
		return f
	}
	fh, fw, ft := freqs(theta(scale(1), dimH), dimH), freqs(theta(scale(2), dimW), dimW), freqs(theta(scale(0), dimT), dimT)
	gh, gw := c.Grid()
	out := make([]float32, rows*D)
	for r := 0; r < rows; r++ {
		a := make([]float64, 0, D/2)
		if r < gh*gw {
			y, x := float64(r/gw), float64(r%gw)
			for range ft {
				a = append(a, 0)
			}
			for _, f := range fh {
				a = append(a, y*f)
			}
			for _, f := range fw {
				a = append(a, x*f)
			}
		} else {
			a = make([]float64, D/2)
		}
		for i, v := range a {
			out[r*D+2*i] = float32(math.Cos(v))
			out[r*D+2*i+1] = float32(math.Sin(v))
		}
	}
	return out
}
