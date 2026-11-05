package iron

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
)

// GraphManifest is manifest.json of a graph exported by tools/amd/iron_graph.py: one IRON graph
// called with inputs (an image model's text encoder, a denoiser step, a VAE decoder), not a
// language model.
type GraphManifest struct {
	Version int             `json:"version"`
	Kind    string          `json:"kind"` // "graph"
	Name    string          `json:"name"`
	Banks   map[string]Bank `json:"banks"`
	Graph   *Version        `json:"graph"`
	// Meta is the application's own description (shapes, what the inputs hold).
	Meta json.RawMessage `json:"meta"`
	// Variants are other images of the banks for the same program: by variant, a bank's
	// image by kind (a DiT's groups of blocks, one ELF with each group's weights). Variant 0
	// is the banks' own images.
	Variants []map[string]string `json:"variants"`
}

// ReadGraphManifest reads dir/manifest.json of a graph export.
func ReadGraphManifest(dir string) (*GraphManifest, error) {
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m GraphManifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("iron: %s: manifest: %w", dir, err)
	}
	if m.Version != ManifestVersion || m.Kind != "graph" {
		return nil, fmt.Errorf("iron: %s: not a graph export (version %d, kind %q)", dir, m.Version, m.Kind)
	}
	v := m.Graph
	if v == nil || v.ELF == "" || len(v.Kernels) == 0 {
		return nil, fmt.Errorf("iron: %s: the graph is no full ELF", dir)
	}
	if len(v.Encoders) > 0 || v.Carried != nil {
		return nil, fmt.Errorf("iron: %s: a graph takes no gathers and carries nothing", dir)
	}
	return &m, nil
}

// Graph is a loaded graph export: its banks, its program and its own buffers.
type Graph struct {
	Manifest *GraphManifest
	dev      *hwnpu.Device
	banks    map[string]*hwnpu.Buffer   // the variant in use
	variants []map[string]*hwnpu.Buffer // every variant's banks (one, with no variants)
	v        *version
	// Timing, when set, is told each command's run time.
	Timing func(command int, run time.Duration)
}

// OpenGraph loads the graph exported in dir onto dev.
func OpenGraph(ctx context.Context, dev *hwnpu.Device, dir string) (*Graph, error) {
	man, err := ReadGraphManifest(dir)
	if err != nil {
		return nil, err
	}
	g := &Graph{Manifest: man, dev: dev}
	fail := func(err error) (*Graph, error) {
		g.Close()
		return nil, fmt.Errorf("iron: %s: %w", man.Name, err)
	}
	images := man.Variants
	if len(images) == 0 {
		images = []map[string]string{nil}
	}
	for i, img := range images {
		banks := map[string]*hwnpu.Buffer{}
		g.variants = append(g.variants, banks)
		for kind, b := range man.Banks {
			buf, err := dev.AllocBuffer(max(b.Bytes, 4096))
			if err != nil {
				return fail(fmt.Errorf("bank %s (%d MiB): %w", kind, b.Bytes>>20, err))
			}
			banks[kind] = buf
			name := b.Image
			if s, ok := img[kind]; ok {
				name = s
			}
			if name != "" {
				if err := readInto(filepath.Join(dir, name), buf.Bytes()); err != nil {
					return fail(fmt.Errorf("variant %d: bank %s: %w", i, kind, err))
				}
			}
			if err := buf.SyncToDevice(0, buf.Size()); err != nil {
				return fail(err)
			}
		}
	}
	g.banks = g.variants[0]
	kr, err := dev.Kernel()
	if err != nil {
		return fail(err)
	}
	v := &version{m: man.Graph, args: map[string]*hwnpu.Buffer{}}
	g.v = v
	if v.prog, err = kr.Load(ctx, hwnpu.ProgramSource{Path: filepath.Join(dir, v.m.ELF)}); err != nil {
		return fail(err)
	}
	for _, a := range v.m.Arguments {
		if _, ok := g.banks[a.Kind]; ok {
			continue
		}
		buf, err := dev.AllocBuffer(max(a.Bytes, 4096))
		if err != nil {
			return fail(fmt.Errorf("argument %s: %w", a.Kind, err))
		}
		v.args[a.Kind] = buf
	}
	return g, nil
}

// Close releases the device memory.
func (g *Graph) Close() error {
	var errs []error
	if v := g.v; v != nil {
		for _, b := range v.args {
			errs = append(errs, b.Close())
		}
		if v.prog != nil {
			errs = append(errs, v.prog.Close())
		}
	}
	for _, banks := range g.variants {
		for _, b := range banks {
			errs = append(errs, b.Close())
		}
	}
	return errors.Join(errs...)
}

// Variants is the number of bank variants (1 without any).
func (g *Graph) Variants() int { return len(g.variants) }

// Use selects the variant the next runs bind.
func (g *Graph) Use(i int) error {
	if i < 0 || i >= len(g.variants) {
		return fmt.Errorf("iron: %s: no variant %d of %d", g.Manifest.Name, i, len(g.variants))
	}
	g.banks = g.variants[i]
	return nil
}

func (g *Graph) buffer(kind string) *hwnpu.Buffer {
	if b, ok := g.banks[kind]; ok {
		return b
	}
	return g.v.args[kind]
}

func findHandle(hs []Handle, name string) (*Handle, bool) {
	for i := range hs {
		if hs[i].Name == name {
			return &hs[i], true
		}
	}
	return nil, false
}

// Inputs and Outputs are the graph's handles.
func (g *Graph) Inputs() []Handle  { return g.v.m.Inputs }
func (g *Graph) Outputs() []Handle { return g.v.m.Outputs }

// Input returns the host view of input name, which a caller fills before Run.
func (g *Graph) Input(name string) ([]byte, error) {
	h, ok := findHandle(g.v.m.Inputs, name)
	if !ok {
		return nil, fmt.Errorf("iron: %s: no input %s", g.Manifest.Name, name)
	}
	return g.buffer(h.Argument).Bytes()[h.Offset : h.Offset+h.Bytes], nil
}

// SetInput copies data into input name, zeros after it.
func (g *Graph) SetInput(name string, data []byte) error {
	dst, err := g.Input(name)
	if err != nil {
		return err
	}
	if len(data) > len(dst) {
		return fmt.Errorf("iron: %s: %d bytes for input %s of %d", g.Manifest.Name, len(data), name, len(dst))
	}
	copy(dst, data)
	clear(dst[len(data):])
	return nil
}

// Output returns output name as the device left it after Run.
func (g *Graph) Output(name string) ([]byte, error) {
	h, ok := findHandle(g.v.m.Outputs, name)
	if !ok {
		return nil, fmt.Errorf("iron: %s: no output %s", g.Manifest.Name, name)
	}
	buf := g.buffer(h.Argument)
	if err := buf.SyncFromDevice(h.Offset, h.Bytes); err != nil {
		return nil, err
	}
	return buf.Bytes()[h.Offset : h.Offset+h.Bytes], nil
}

// Run syncs the inputs to the device and runs every command with the per-call values.
func (g *Graph) Run(ctx context.Context, values map[string]int64) error {
	v := g.v
	for _, h := range v.m.Inputs {
		if err := g.buffer(h.Argument).SyncToDevice(h.Offset, h.Bytes); err != nil {
			return err
		}
	}
	all := make(map[string]int64, len(v.m.Values)+len(values))
	for _, n := range v.m.Values {
		all[n] = 0
	}
	for k, x := range values {
		all[k] = x
	}
	for _, a := range v.m.Addresses {
		base, ok := g.buffer(a.Argument).DeviceAddress()
		if !ok {
			return errors.New("iron: the device does not expose buffer addresses")
		}
		addr := base + uint64(a.Offset)
		switch a.Member {
		case "base_lo":
			all[a.Name] = int64(addr & (1<<a.LowBits - 1))
		case "base_hi":
			all[a.Name] = int64(addr >> a.LowBits)
		default:
			return fmt.Errorf("iron: address value %s of %s", a.Name, a.Member)
		}
	}
	for k := range v.m.Kernels {
		kern, err := v.prog.Kernel(v.m.Kernels[k].Name)
		if err != nil {
			return err
		}
		run, err := kern.NewRun()
		if err != nil {
			return err
		}
		for _, a := range v.m.Arguments {
			run.SetBuffer(a.Index, g.buffer(a.Kind))
		}
		err = run.Err()
		if err == nil {
			err = writeWords(run.Scratchpad(), v.m.Kernels[k].Parameters, all)
		}
		t0 := time.Now()
		if err == nil {
			err = run.Run(ctx)
		}
		if g.Timing != nil {
			g.Timing(k, time.Since(t0))
		}
		run.Close()
		if err != nil {
			return fmt.Errorf("iron: %s: command %d: %w", g.Manifest.Name, k, err)
		}
	}
	return nil
}
