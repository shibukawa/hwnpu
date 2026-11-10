package sana

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

// ironEngine runs the graphs of tools/amd/iron_sana.py on AMD's NPU. They take and give bf16;
// the values the pipeline hands over (bf16 that went through float32) come back bit for bit.
type ironEngine struct {
	dev  *hwnpu.Device
	dir  string
	dit  DitConfig
	text TextConfig
	vae  *VAE

	ditIn, ditBlocks, ditOut *iron.Graph // open while the steps run
	ctxSet                   []float32   // the caption dit_blocks holds
}

func meta[T any](m *iron.GraphManifest) (T, error) {
	var x struct {
		Config T `json:"config"`
	}
	err := json.Unmarshal(m.Meta, &x)
	return x.Config, err
}

// isIronExport reports whether dir is an export of tools/amd/iron_sana.py.
func isIronExport(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "model.json"))
	if err != nil {
		return false
	}
	_, err = os.Stat(filepath.Join(dir, "caption", "manifest.json"))
	return err == nil
}

func openIron(dev *hwnpu.Device, dir string) (*ironEngine, DitConfig, TextConfig, error) {
	e := &ironEngine{dev: dev, dir: dir}
	var tc TextConfig
	mb, err := iron.ReadGraphManifest(filepath.Join(dir, "dit_blocks"))
	if err != nil {
		return nil, e.dit, tc, err
	}
	if e.dit, err = meta[DitConfig](mb); err != nil {
		return nil, e.dit, tc, err
	}
	if n, v := e.dit.Layers/e.dit.Group, max(1, len(mb.Variants)); v != n {
		return nil, e.dit, tc, fmt.Errorf("dit_blocks holds %d groups, the DiT %d", v, n)
	}
	mt, err := iron.ReadGraphManifest(filepath.Join(dir, "text"))
	if err != nil {
		return nil, e.dit, tc, err
	}
	if tc, err = meta[TextConfig](mt); err != nil {
		return nil, e.dit, tc, err
	}
	e.text = tc
	if e.vae, err = OpenVAE(dev, filepath.Join(dir, "vae")); err != nil {
		return nil, e.dit, tc, err
	}
	return e, e.dit, tc, nil
}

// graphPart is an IRON graph as the pipeline's part: bf16 in and out; with prefix, run passes
// the call's start position (the text graph that holds the instruction's first rows).
type graphPart struct {
	g      *iron.Graph
	input  string
	prefix bool
}

func (e *ironEngine) open(ctx context.Context, name, input string, prefix bool) (part, error) {
	g, err := iron.OpenGraph(ctx, e.dev, filepath.Join(e.dir, name))
	if err != nil {
		return nil, err
	}
	return &graphPart{g: g, input: input, prefix: prefix}, nil
}

func (gp *graphPart) run(ctx context.Context, in []float32, start int) ([]float32, error) {
	if err := gp.g.SetInput(gp.input, toBF16(in)); err != nil {
		return nil, err
	}
	var values map[string]int64
	if gp.prefix {
		values = map[string]int64{"start": int64(start)}
	}
	if err := gp.g.Run(ctx, values); err != nil {
		return nil, err
	}
	out, err := gp.g.Output(gp.g.Outputs()[0].Name)
	if err != nil {
		return nil, err
	}
	return fromBF16s(out), nil
}

func (gp *graphPart) close() error { return gp.g.Close() }

func (e *ironEngine) openText(ctx context.Context) (part, error) {
	return e.open(ctx, "text", "x", e.text.Prefix > 0)
}

func (e *ironEngine) openCaption(ctx context.Context) (part, error) {
	return e.open(ctx, "caption", "y", false)
}

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

func (e *ironEngine) openDiT(ctx context.Context, resident bool) error {
	if e.ditBlocks != nil {
		return nil
	}
	for _, x := range []struct {
		g    **iron.Graph
		name string
	}{{&e.ditIn, "dit_in"}, {&e.ditBlocks, "dit_blocks"}, {&e.ditOut, "dit_out"}} {
		g, err := iron.OpenGraphWith(ctx, e.dev, filepath.Join(e.dir, x.name), iron.GraphOptions{SwapVariants: !resident})
		if err != nil {
			e.closeDiT()
			return err
		}
		*x.g = g
	}
	e.ctxSet = nil
	return nil
}

// velocity runs the DiT on patches (tokens x latent, channel last) once for each conditioning.
// Each group of blocks runs every call before the next group's weights are read.
func (e *ironEngine) velocity(ctx context.Context, patches, tproj []float32, conds []Cond) ([][]float32, error) {
	c := e.dit
	n, L := c.Tokens(), c.Latent
	padded := make([]float32, c.NTok()*L) // zeros past the tokens
	copy(padded, patches)
	g := e.ditIn
	if err := g.SetInput("patches", toBF16(padded)); err != nil {
		return nil, err
	}
	if err := g.SetInput("tproj", toBF16(tproj)); err != nil {
		return nil, err
	}
	if err := g.Run(ctx, nil); err != nil {
		return nil, err
	}
	outs := g.Outputs() // x, t6, emb
	h, err := g.Output(outs[0].Name)
	if err != nil {
		return nil, err
	}
	t6, err := g.Output(outs[1].Name)
	if err != nil {
		return nil, err
	}
	emb, err := g.Output(outs[2].Name)
	if err != nil {
		return nil, err
	}
	hs := make([][]byte, len(conds))
	for k := range hs {
		hs[k] = append([]byte(nil), h...)
	}
	b := e.ditBlocks
	if err := b.SetInput("t6", t6); err != nil {
		return nil, err
	}
	if err := e.ditOut.SetInput("emb", emb); err != nil {
		return nil, err
	}
	for gi := 0; gi < b.Variants(); gi++ {
		if err := b.Use(gi); err != nil {
			return nil, err
		}
		for k, cd := range conds {
			if len(e.ctxSet) == 0 || &cd.ctx[0] != &e.ctxSet[0] {
				if err := b.SetInput("ctx", toBF16(cd.ctx)); err != nil {
					return nil, err
				}
				e.ctxSet = cd.ctx
			}
			if err := b.SetInput("x", hs[k]); err != nil {
				return nil, err
			}
			if err := b.Run(ctx, map[string]int64{"nk": int64(cd.Rows)}); err != nil {
				return nil, fmt.Errorf("blocks %d..%d: %w", gi*c.Group, (gi+1)*c.Group-1, err)
			}
			o, err := b.Output(b.Outputs()[0].Name)
			if err != nil {
				return nil, err
			}
			hs[k] = append(hs[k][:0], o...)
		}
	}
	res := make([][]float32, len(conds))
	for k := range conds {
		if err := e.ditOut.SetInput("x", hs[k]); err != nil {
			return nil, err
		}
		if err := e.ditOut.Run(ctx, nil); err != nil {
			return nil, err
		}
		v, err := e.ditOut.Output(e.ditOut.Outputs()[0].Name)
		if err != nil {
			return nil, err
		}
		res[k] = fromBF16s(v)[:n*L]
	}
	return res, nil
}

func (e *ironEngine) decode(ctx context.Context, latents []float32, stage StageFunc) (*image.RGBA, error) {
	e.vae.Stage = stage
	return e.vae.Decode(ctx, latents)
}

func (e *ironEngine) close() error { return e.closeDiT() }
