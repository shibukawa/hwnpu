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
type ironEngine struct {
	dit                             DitConfig
	textG, ditIn, ditBlocks, ditOut *iron.Graph
	vae                             *VAE
}

func meta[T any](g *iron.Graph) (T, error) {
	var m struct {
		Config T `json:"config"`
	}
	err := json.Unmarshal(g.Manifest.Meta, &m)
	return m.Config, err
}

func openIron(ctx context.Context, dev *hwnpu.Device, dir string) (*ironEngine, DitConfig, TextConfig, error) {
	e := &ironEngine{}
	var dc DitConfig
	var tc TextConfig
	fail := func(err error) (*ironEngine, DitConfig, TextConfig, error) {
		e.close()
		return nil, dc, tc, err
	}
	var err error
	for _, x := range []struct {
		g    **iron.Graph
		name string
	}{{&e.textG, "text"}, {&e.ditIn, "dit_in"}, {&e.ditBlocks, "dit_blocks"}, {&e.ditOut, "dit_out"}} {
		if *x.g, err = iron.OpenGraph(ctx, dev, filepath.Join(dir, x.name)); err != nil {
			return fail(err)
		}
	}
	if dc, err = meta[DitConfig](e.ditBlocks); err != nil {
		return fail(err)
	}
	if tc, err = meta[TextConfig](e.textG); err != nil {
		return fail(err)
	}
	e.dit = dc
	if n := dc.Layers / dc.Group; e.ditBlocks.Variants() != n {
		return fail(fmt.Errorf("dit_blocks holds %d groups, the DiT %d", e.ditBlocks.Variants(), n))
	}
	if _, err := os.Stat(filepath.Join(dir, "vae", "manifest.json")); err == nil {
		if e.vae, err = OpenVAE(ctx, dev, filepath.Join(dir, "vae")); err != nil {
			return fail(err)
		}
	}
	return e, dc, tc, nil
}

func (e *ironEngine) close() error {
	var errs []error
	for _, g := range []*iron.Graph{e.textG, e.ditIn, e.ditBlocks, e.ditOut} {
		if g != nil {
			errs = append(errs, g.Close())
		}
	}
	if e.vae != nil {
		errs = append(errs, e.vae.Close())
	}
	return errors.Join(errs...)
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

func (e *ironEngine) text(ctx context.Context, qx, tx []float32, nq, nt int) ([]float32, error) {
	if err := e.textG.SetInput("qx", toBF16(qx)); err != nil {
		return nil, err
	}
	if err := e.textG.SetInput("tx", toBF16(tx)); err != nil {
		return nil, err
	}
	if err := e.textG.Run(ctx, map[string]int64{"nq": int64(nq), "nt": int64(nt)}); err != nil {
		return nil, err
	}
	out, err := e.textG.Output(e.textG.Outputs()[0].Name)
	if err != nil {
		return nil, err
	}
	return fromBF16s(out), nil
}

func (e *ironEngine) setCond(cond []float32) error { return e.ditBlocks.SetInput("ctx", toBF16(cond)) }

// velocity runs dit_in, each group of dit_blocks and dit_out.
func (e *ironEngine) velocity(ctx context.Context, patches, tproj []float32) ([]float32, error) {
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
	for g := 0; g < e.ditBlocks.Variants(); g++ {
		if err := e.ditBlocks.Use(g); err != nil {
			return nil, err
		}
		if err := e.ditBlocks.SetInput("x", x); err != nil {
			return nil, err
		}
		if err := e.ditBlocks.Run(ctx, nil); err != nil {
			return nil, fmt.Errorf("blocks %d..%d: %w", g*e.dit.Group, (g+1)*e.dit.Group-1, err)
		}
		if x, err = e.ditBlocks.Output(e.ditBlocks.Outputs()[0].Name); err != nil {
			return nil, err
		}
	}
	if err := e.ditOut.SetInput("x", x); err != nil {
		return nil, err
	}
	if err := e.ditOut.Run(ctx, nil); err != nil {
		return nil, err
	}
	v, err := e.ditOut.Output(e.ditOut.Outputs()[0].Name)
	if err != nil {
		return nil, err
	}
	return fromBF16s(v), nil
}

func (e *ironEngine) hasVAE() bool { return e.vae != nil }

func (e *ironEngine) decode(ctx context.Context, latents []float32, logf func(string, ...any)) (*image.RGBA, error) {
	e.vae.Logf = logf
	return e.vae.Decode(ctx, latents)
}
