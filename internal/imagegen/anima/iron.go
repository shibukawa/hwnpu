package anima

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
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
	dev                      *hwnpu.Device
	dir                      string
	dit                      DitConfig
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

// openIron reads the export's manifests; the graphs are loaded when they run.
func openIron(ctx context.Context, dev *hwnpu.Device, dir string) (*ironEngine, DitConfig, TextConfig, error) {
	e := &ironEngine{dev: dev, dir: dir}
	var dc DitConfig
	var tc TextConfig
	fail := func(err error) (*ironEngine, DitConfig, TextConfig, error) {
		return nil, dc, tc, err
	}
	ms := map[string]*iron.GraphManifest{}
	for _, name := range []string{"text", "dit_in", "dit_blocks", "dit_out"} {
		m, err := iron.ReadGraphManifest(filepath.Join(dir, name))
		if err != nil {
			return fail(err)
		}
		ms[name] = m
	}
	var err error
	if dc, err = meta[DitConfig](ms["dit_blocks"]); err != nil {
		return fail(err)
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
	if _, err := os.Stat(filepath.Join(dir, "vae", "manifest.json")); err == nil {
		if e.vae, err = OpenVAE(ctx, dev, filepath.Join(dir, "vae")); err != nil {
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
	for _, x := range []struct {
		g    **iron.Graph
		name string
	}{{&e.ditIn, "dit_in"}, {&e.ditBlocks, "dit_blocks"}, {&e.ditOut, "dit_out"}} {
		g, err := iron.OpenGraphWith(ctx, e.dev, filepath.Join(e.dir, x.name), iron.GraphOptions{SwapVariants: !ResidentDiT})
		if err != nil {
			e.closeDiT()
			return err
		}
		*x.g = g
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
	if err := e.ditIn.SetInput("patches", toBF16(patches)); err != nil {
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
			if err := e.ditBlocks.Run(ctx, nil); err != nil {
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
		vs[k] = fromBF16s(v)
	}
	return vs, nil
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
