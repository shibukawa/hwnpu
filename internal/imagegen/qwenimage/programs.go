package qwenimage

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagegen/qwenimage/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// progEngine runs the programs of internal/imagegen/qwenimage/mil, each a runner.Program on the
// Neural Engine (or the CPU interpreter). The text encoder's groups and the DiT's text pass are
// loaded one at a time while a prompt is read, each VAE stage while it runs, and the DiT's image
// blocks from the first step to the decoding: at most the DiT's weights are held (3.6 GB at lut4).
// On Core AI the DiT's groups are loaded in turn as well (swapDiT).
type progEngine struct {
	m    *ProgramManifest
	dir  string
	dev  *hwnpu.Device
	log  io.Writer
	time *timeEmbed
	blks []runner.Program
	// swapDiT loads each group of the DiT's image blocks for each step, runs it and closes it,
	// rather than holding them all (Core AI: a group loads from Core AI's cache in 20 ms). On M1,
	// Core AI fails to run the eight groups held at once (5.2 GiB on the Neural Engine) rather than
	// evicting them as Core ML does, runs a group in 0.95 s with four held and in 0.41 s alone
	// (design.md 16.184). ready says openDiT checked them.
	swapDiT, ready bool
}

func openPrograms(dev *hwnpu.Device, dir string, log io.Writer) (*progEngine, error) {
	m, err := ReadProgramManifest(dir)
	if err != nil {
		return nil, err
	}
	if m.Version != ExportVersion {
		return nil, fmt.Errorf("%s is of layout %d, this hwnpu reads %d: convert it again", dir, m.Version, ExportVersion)
	}
	e := &progEngine{m: m, dir: dir, dev: dev, log: log, swapDiT: dev != nil && dev.Info().Backend == "coreai"}
	if e.time, err = readTimeFile(filepath.Join(dir, m.Time), m.Dit.Hidden); err != nil {
		return nil, err
	}
	return e, nil
}

// open loads one program, on the device unless it is nil, and fails unless it runs on the NPU
// alone. With a log it shows a line while it loads (the first load compiles it).
func (e *progEngine) open(ctx context.Context, name string) (runner.Program, error) {
	var bar *console.Bar
	if e.log != nil {
		bar = console.NewLoadBar(e.log, "  loading "+name, llm.PartBytes(filepath.Join(e.dir, name)))
	}
	p, err := e.load(ctx, name)
	if bar != nil {
		switch {
		case err != nil:
			bar.Failed("failed")
		case e.dev != nil:
			bar.Done(console.Duration(bar.Elapsed()) + ", all on the NPU")
		default:
			bar.Done(console.Duration(bar.Elapsed()))
		}
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return p, nil
}

// load is open without a line: the program on the device unless it is nil, on the NPU alone.
func (e *progEngine) load(ctx context.Context, name string) (runner.Program, error) {
	p, err := runner.Open(ctx, e.dev, filepath.Join(e.dir, name))
	if err == nil && e.dev != nil && !p.OnNPU() {
		p.Close()
		err = fmt.Errorf("%s does not run on the NPU alone", name)
	}
	return p, err
}

// each loads the program name, runs f with it and closes it.
func (e *progEngine) each(ctx context.Context, name string, f func(p runner.Program) error) error {
	p, err := e.open(ctx, name)
	if err != nil {
		return err
	}
	defer p.Close()
	if err := f(p); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return nil
}

// inTurn runs f with each of the programs names in order, each loaded just before and closed
// after: one is held at a time. Loading the next while one runs did not save time (the Neural
// Engine's compiles are mostly one at a time) and held two (design.md 16.165).
func (e *progEngine) inTurn(ctx context.Context, names []string, f func(i int, p runner.Program) error) error {
	for i, name := range names {
		p, err := e.open(ctx, name)
		if err != nil {
			return err
		}
		err = f(i, p)
		p.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	return nil
}

// encode runs the text encoder on each prompt's ids (the template's), a group of layers at a time
// (every prompt through a group before the next is loaded), then the DiT's text pass at t = 0 the
// same way: each prompt's keys and values of every block.
func (e *progEngine) encode(ctx context.Context, embed *os.File, ids [][]int32) ([]Cond, error) {
	m := e.m
	E, L, TR, drop := m.Text.Hidden, m.Text.Rows, m.Dit.TextRows, m.Drop
	xs := make([][]float32, len(ids))
	row := make([]byte, 2*E)
	for k, ii := range ids {
		x := make([]float32, L*E)
		for i, id := range ii {
			if id < 0 || int(id) >= m.Text.Vocab {
				return nil, fmt.Errorf("qwenimage: token %d past the vocabulary", id)
			}
			if _, err := embed.ReadAt(row, int64(id)*int64(2*E)); err != nil {
				return nil, err
			}
			for j := 0; j < E; j++ {
				x[i*E+j] = fromBF16(binary.LittleEndian.Uint16(row[2*j:])) / mil.TextScale
			}
		}
		xs[k] = x
	}
	t0 := time.Now()
	err := e.inTurn(ctx, m.TextIn, func(_ int, p runner.Program) error {
		for k := range xs {
			y, err := p.Run(ctx, map[string][]float32{"x": xs[k]})
			if err != nil {
				return err
			}
			xs[k] = y
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	t1 := time.Now()
	// the DiT's text rows: txt_in's after the system message's, zeros past them
	conds := make([]Cond, len(ids))
	for k, ii := range ids {
		T := len(ii) - drop
		x := make([]float32, TR*E)
		copy(x, xs[k][drop*E:len(ii)*E])
		xs[k], conds[k].Rows = x, T
	}
	mod0, _ := e.time.modulation(0)
	err = e.inTurn(ctx, m.DitText, func(gi int, p runner.Program) error {
		count := min(m.Dit.Group, m.Dit.Layers-gi*m.Dit.Group)
		{
			for k := range xs {
				y, err := p.Run(ctx, map[string][]float32{"x": xs[k], "mod": mod0})
				if err != nil {
					return err
				}
				if len(y) != (1+2*count)*TR*E {
					return fmt.Errorf("%d values, not %d blocks' rows, keys and values", len(y), count)
				}
				xs[k] = y[:TR*E]
				tk, tv := make([]float32, 0, count*TR*E), make([]float32, 0, count*TR*E)
				for j := 0; j < count; j++ {
					tk = append(tk, y[(1+2*j)*TR*E:(2+2*j)*TR*E]...)
					tv = append(tv, y[(2+2*j)*TR*E:(3+2*j)*TR*E]...)
				}
				conds[k].ptk, conds[k].ptv = append(conds[k].ptk, tk), append(conds[k].ptv, tv)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if e.log != nil {
		fmt.Fprintf(e.log, "Qwen3-VL-8B %.1f s, the DiT's text pass %.1f s\n", t1.Sub(t0).Seconds(), time.Since(t1).Seconds())
	}
	return conds, nil
}

// openDiT loads the DiT's image blocks, held until closeDiT. One at a time: of 8 programs the
// Neural Engine's cache had dropped (M3), four at once took 56 s against 71 s in a test, but 76 s
// against 62 s in the pipeline, with a peak footprint of 7.4 GB against 4.5 (design.md 16.165).
func (e *progEngine) openDiT(ctx context.Context) error {
	if e.blks != nil || e.ready {
		return nil
	}
	if e.swapDiT { // each loaded once here (the first load compiles it), then for each step
		for _, name := range e.m.Blocks {
			p, err := e.open(ctx, name)
			if err != nil {
				return err
			}
			p.Close()
		}
		e.ready = true
		return nil
	}
	for _, name := range e.m.Blocks {
		p, err := e.open(ctx, name)
		if err != nil {
			e.closeDiT()
			return err
		}
		e.blks = append(e.blks, p)
	}
	return nil
}

func (e *progEngine) closeDiT() error {
	e.ready = false
	var errs []error
	for _, p := range e.blks {
		errs = append(errs, p.Close())
	}
	e.blks = nil
	return errors.Join(errs...)
}

// imageInputs are a conditioning's image RoPE (cos, sin) and key bias for the image blocks.
type imageInputs struct{ cos, sin, kbias []float32 }

func (e *progEngine) imageInputs(cd Cond) imageInputs {
	c := mil.Dit{Width: e.m.Dit.Width, Height: e.m.Dit.Height, HeadDim: e.m.Dit.HeadDim, Theta: e.m.Dit.Theta}
	copy(c.Axes[:], e.m.Dit.Axes)
	cos, sin := mil.RopeTables(c, mil.ImagePositions(c, cd.Rows))
	N := c.Tokens()
	kb := make([]float32, N+e.m.Dit.TextRows)
	for i := N + cd.Rows; i < len(kb); i++ {
		kb[i] = -1e4
	}
	return imageInputs{cos, sin, kb}
}

// velocity runs the DiT's image blocks on x (tokens x latent, channel last) at sigma for each
// conditioning, back to back in each group (a group the Neural Engine has to load again is loaded
// once a step): the velocities, the same layout.
func (e *progEngine) velocity(ctx context.Context, x []float32, sigma float64, conds []Cond, ins []imageInputs) ([][]float32, error) {
	mod, scale := e.time.modulation(sigma)
	xs := make([][]float32, len(conds))
	for k := range xs {
		xs[k] = x
	}
	for gi, name := range e.m.Blocks {
		var b runner.Program
		if e.swapDiT {
			var err error
			if b, err = e.load(ctx, name); err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
		} else {
			b = e.blks[gi]
		}
		for k, cd := range conds {
			in := map[string][]float32{"x": xs[k], "mod": mod, "cos": ins[k].cos, "sin": ins[k].sin,
				"tk": cd.ptk[gi], "tv": cd.ptv[gi], "kbias": ins[k].kbias}
			if gi == len(e.m.Blocks)-1 {
				in["scale"] = scale
			}
			y, err := b.Run(ctx, in)
			if err != nil {
				if e.swapDiT {
					b.Close()
				}
				return nil, fmt.Errorf("%s: %w", name, err)
			}
			xs[k] = y
		}
		if e.swapDiT {
			b.Close()
		}
	}
	return xs, nil
}

// decode runs the VAE's stages on each of the latents (latent, rows, columns), stage by stage (each
// loaded once for all of them, one load ahead); the last gives the pictures (1, 4, h, w), RGBA.
func (e *progEngine) decode(ctx context.Context, latents [][]float32, stage StageFunc) ([]*image.RGBA, error) {
	xs := append([][]float32(nil), latents...)
	t0 := time.Now()
	err := e.inTurn(ctx, e.m.Vae, func(i int, p runner.Program) error {
		for k := range xs {
			var y []float32
			var err error
			if i < len(e.m.VaeTiles) && e.m.VaeTiles[i].tiled() {
				y, err = runTiled(ctx, p, xs[k], e.m.VaeTiles[i])
			} else {
				y, err = p.Run(ctx, map[string][]float32{"x": xs[k]})
			}
			if err != nil {
				return fmt.Errorf("vae stage %d: %w", i, err)
			}
			xs[k] = y
		}
		if stage != nil {
			stage(i, len(e.m.Vae), time.Since(t0))
		}
		t0 = time.Now()
		return nil
	})
	if err != nil {
		return nil, err
	}
	W, H := e.m.Dit.Width, e.m.Dit.Height
	imgs := make([]*image.RGBA, len(xs))
	for k, x := range xs {
		if len(x) != 4*W*H {
			return nil, fmt.Errorf("vae: %d values for a %dx%d RGBA picture", len(x), W, H)
		}
		img := image.NewRGBA(image.Rect(0, 0, W, H))
		for y := 0; y < H; y++ {
			for xx := 0; xx < W; xx++ {
				i := y*W + xx
				var px [4]uint8
				for c := range px {
					px[c] = pixel(x[c*W*H+i])
				}
				for c := 0; c < 3; c++ { // straight alpha into image.RGBA's premultiplied
					px[c] = uint8((int(px[c])*int(px[3]) + 127) / 255)
				}
				img.SetRGBA(xx, y, color.RGBA{px[0], px[1], px[2], px[3]})
			}
		}
		imgs[k] = img
	}
	return imgs, nil
}

func pixel(v float32) uint8 {
	x := (math.Max(-1, math.Min(1, float64(v))) + 1) * 127.5
	return uint8(math.Round(x))
}

func fromBF16(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }

// generatePrograms is Generate with a Core ML export's programs.
func (p *Pipeline) generatePrograms(ctx context.Context, cond Cond, neg *Cond, noise []float32, sigmas []float64, smp Sampling, step StepFunc) ([]float32, error) {
	e := p.prog
	N, L := p.Dit.Tokens(), p.Dit.Latent
	if len(cond.ptk) != len(e.m.Blocks) {
		return nil, fmt.Errorf("qwenimage: the text's keys of %d groups for %d", len(cond.ptk), len(e.m.Blocks))
	}
	if err := e.openDiT(ctx); err != nil {
		return nil, fmt.Errorf("qwenimage: %w", err)
	}
	conds := []Cond{cond}
	if neg != nil && smp.Guidance != 1 {
		conds = append(conds, *neg)
	}
	ins := make([]imageInputs, len(conds))
	for k, cd := range conds {
		ins[k] = e.imageInputs(cd)
	}
	x := make([]float32, N*L)
	for ch := 0; ch < L; ch++ {
		for t := 0; t < N; t++ {
			x[t*L+ch] = noise[ch*N+t]
		}
	}
	steps := len(sigmas) - 1
	var d0, d1 []float32
	var hLast float64
	for i := 0; i < steps; i++ {
		t0 := time.Now()
		run := conds[:1]
		if smp.guided(sigmas[i]) {
			run = conds
		}
		vs, err := e.velocity(ctx, x, sigmas[i], run, ins)
		if err != nil {
			return nil, fmt.Errorf("qwenimage: step %d: %w", i, err)
		}
		v := vs[0]
		if len(vs) == 2 { // the pipeline's true_cfg_scale: neg + g (cond - neg)
			g := float32(smp.Guidance)
			for j := range v {
				v[j] = vs[1][j] + g*(v[j]-vs[1][j])
			}
		}
		if smp.Solver == "dpm2m" {
			hLast = anima.DPM2M(x, v, &d0, &d1, sigmas[i], sigmas[i+1], hLast)
		} else {
			d := float32(sigmas[i+1] - sigmas[i])
			for j := range x {
				x[j] += d * v[j]
			}
		}
		if step != nil {
			step(i, steps, time.Since(t0))
		}
	}
	res := make([]float32, N*L)
	for ch := 0; ch < L; ch++ {
		for t := 0; t < N; t++ {
			res[ch*N+t] = x[t*L+ch]
		}
	}
	return res, nil
}

// runTiled runs the program of a tiled VAE stage on each window of its input x (C, H, W) and puts
// the tiles of the outputs together: (OutC, Scale H, Scale W). A window's output beyond its tile
// is dropped (its edge inside the input was zero padded).
func runTiled(ctx context.Context, p runner.Program, x []float32, t VaeTiling) ([]float32, error) {
	if len(x) != t.C*t.H*t.W {
		return nil, fmt.Errorf("%d values for a (%d, %d, %d) input", len(x), t.C, t.H, t.W)
	}
	wh, ww := t.Rows.Window(), t.Cols.Window()
	s, H, W := t.Scale, t.H, t.W
	oh, ow := s*wh, s*ww
	out := make([]float32, t.OutC*s*H*s*W)
	win := make([]float32, t.C*wh*ww)
	for _, r := range t.Rows.Windows() {
		for _, c := range t.Cols.Windows() {
			for ch := 0; ch < t.C; ch++ {
				for y := 0; y < wh; y++ {
					copy(win[(ch*wh+y)*ww:(ch*wh+y+1)*ww], x[(ch*H+r.Start+y)*W+c.Start:])
				}
			}
			y, err := p.Run(ctx, map[string][]float32{"x": win})
			if err != nil {
				return nil, err
			}
			if len(y) != t.OutC*oh*ow {
				return nil, fmt.Errorf("a tile of %d values, want (%d, %d, %d)", len(y), t.OutC, oh, ow)
			}
			for ch := 0; ch < t.OutC; ch++ {
				for yy := 0; yy < s*r.Len; yy++ {
					src := y[(ch*oh+s*r.Off+yy)*ow+s*c.Off:][:s*c.Len]
					copy(out[(ch*s*H+s*(r.Start+r.Off)+yy)*s*W+s*(c.Start+c.Off):], src)
				}
			}
		}
	}
	return out, nil
}
