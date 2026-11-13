package sana

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/imagegen/sana/mil"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// ProgramKind is the kind of manifest.json of an export of Convert.
const ProgramKind = "sana-programs"

// ProgramManifest is manifest.json of an export of Convert: the shapes and the programs (Core ML
// packages, or OpenVINO IR with Backend intel) by part, relative to the export's directory.
type ProgramManifest struct {
	Kind    string `json:"kind"`
	Backend string `json:"backend"`
	Version int    `json:"version"` // ExportVersion
	// Formats are the weight formats of the parts (text, dit, vae), as hwnpu-convert's -format.
	Formats map[string]string `json:"formats"`
	Dit     DitConfig         `json:"dit"`
	Text    TextConfig        `json:"text"`
	// TextParts are Gemma 2's groups of layers, in order; the last ends with the final norm.
	TextParts []string `json:"text_parts"`
	Caption   string   `json:"caption"`
	Time      string   `json:"time"`   // TimeFile
	Blocks    []string `json:"blocks"` // the DiT's groups of blocks, in order
	Final     string   `json:"final"`
	Vae       []string `json:"vae"` // the VAE's stages, in order
}

// IsProgramExport reports whether dir holds an export of Convert.
func IsProgramExport(dir string) bool {
	_, err := ReadProgramManifest(dir)
	return err == nil
}

// ReadProgramManifest reads dir/manifest.json of an export of Convert.
func ReadProgramManifest(dir string) (*ProgramManifest, error) {
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m ProgramManifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if m.Kind != ProgramKind {
		return nil, fmt.Errorf("sana: %s is not an export of hwnpu-image convert", dir)
	}
	return &m, nil
}

func writeProgramManifest(dir string, m *ProgramManifest) error {
	m.Kind = ProgramKind
	b, err := json.MarshalIndent(m, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "manifest.json"), append(b, '\n'), 0o644)
}

// progEngine runs the programs of internal/imagegen/sana/mil, each a runner.Program on the Neural
// Engine (or the CPU interpreter). Gemma 2, the caption and each VAE stage are loaded only while
// they run, the DiT from the first step to the decoding: at most the DiT's weights are held.
type progEngine struct {
	m    *ProgramManifest
	dir  string
	dev  *hwnpu.Device
	log  io.Writer
	time *timeEmbed
	fin  runner.Program
	blks []runner.Program
}

func openPrograms(dev *hwnpu.Device, dir string, log io.Writer) (*progEngine, DitConfig, TextConfig, error) {
	m, err := ReadProgramManifest(dir)
	if err != nil {
		return nil, DitConfig{}, TextConfig{}, err
	}
	if m.Version != ExportVersion {
		return nil, m.Dit, m.Text, fmt.Errorf("%s is of layout %d, this hwnpu reads %d: convert it again", dir, m.Version, ExportVersion)
	}
	return &progEngine{m: m, dir: dir, dev: dev, log: log}, m.Dit, m.Text, nil
}

// open loads one program, on the device unless it is nil, and fails unless it runs on the NPU
// alone. With a log it shows a line while it loads (the first load compiles it).
func (e *progEngine) open(ctx context.Context, name string) (runner.Program, error) {
	if name == "" {
		return nil, errors.New("the manifest misses a part")
	}
	var bar *console.Bar
	if e.log != nil {
		bar = console.NewBar(e.log, "  loading "+name)
	}
	p, err := runner.Open(ctx, e.dev, filepath.Join(e.dir, name))
	// OpenVINO's CPU plugin (the Intel path checked without Intel's NPU) is not the NPU
	if err == nil && e.dev != nil && !p.OnNPU() && e.dev.Info().Backend != "openvino" {
		p.Close()
		err = fmt.Errorf("%s does not run on the NPU alone", name)
	}
	if bar != nil {
		switch {
		case err != nil:
			bar.Done("failed")
		case e.dev != nil && e.dev.Info().Backend == "openvino":
			bar.Done(console.Duration(bar.Elapsed()) + ", on " + e.dev.Info().Name)
		case e.dev != nil: // console.Log colors it green, as Anima's loads say it
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

// once loads the program name, runs it on in and closes it.
func (e *progEngine) once(ctx context.Context, name string, in map[string][]float32) ([]float32, error) {
	p, err := e.open(ctx, name)
	if err != nil {
		return nil, err
	}
	defer p.Close()
	y, err := p.Run(ctx, in)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return y, nil
}

// textPart runs Gemma 2's programs in turn, each loaded only while it runs: at most one
// group's weights are held (the five are 2.2 GB at int8).
type textPart struct{ e *progEngine }

func (t textPart) run(ctx context.Context, x []float32, start int) ([]float32, error) {
	if start != 0 {
		return nil, fmt.Errorf("sana: a Core ML export runs the prompt from its start, not %d", start)
	}
	var err error
	for _, name := range t.e.m.TextParts {
		if x, err = t.e.once(ctx, name, map[string][]float32{"x": x}); err != nil {
			return nil, err
		}
	}
	return x, nil
}

func (t textPart) close() error { return nil }

func (e *progEngine) openText(ctx context.Context) (part, error) { return textPart{e}, nil }

// programPart is one program as the pipeline's part.
type programPart struct {
	p     runner.Program
	input string
}

func (pp programPart) run(ctx context.Context, in []float32, _ int) ([]float32, error) {
	return pp.p.Run(ctx, map[string][]float32{pp.input: in})
}

func (pp programPart) close() error { return pp.p.Close() }

func (e *progEngine) openCaption(ctx context.Context) (part, error) {
	p, err := e.open(ctx, e.m.Caption)
	if err != nil {
		return nil, err
	}
	return programPart{p, "text"}, nil
}

func (e *progEngine) openDiT(ctx context.Context, _ bool) error {
	if e.time != nil {
		return nil
	}
	var err error
	if e.time, err = readTimeFile(filepath.Join(e.dir, e.m.Time), e.m.Dit.Hidden); err != nil {
		return err
	}
	for _, name := range e.m.Blocks {
		p, err := e.open(ctx, name)
		if err != nil {
			e.closeDiT()
			return err
		}
		e.blks = append(e.blks, p)
	}
	if e.fin, err = e.open(ctx, e.m.Final); err != nil {
		e.closeDiT()
		return err
	}
	return nil
}

func (e *progEngine) closeDiT() error {
	var errs []error
	for _, p := range append([]runner.Program{e.fin}, e.blks...) {
		if p != nil {
			errs = append(errs, p.Close())
		}
	}
	e.time, e.fin, e.blks = nil, nil, nil
	return errors.Join(errs...)
}

// cbias is the additive mask of the caption's rows: 0 for the n that count, -10000 past them.
func cbias(n int) []float32 {
	b := make([]float32, mil.TextRows)
	for i := n; i < len(b); i++ {
		b[i] = -1e4
	}
	return b
}

func (e *progEngine) velocity(ctx context.Context, patches, tproj []float32, conds []Cond) ([][]float32, error) {
	t6, emb := e.time.embed(tproj)
	var err error
	// each group runs every conditioning before the next group, as on AMD (design.md 16.148):
	// a group the Neural Engine has to load again (an M1 past about 2.9 GiB) loads once a step
	xs := make([][]float32, len(conds))
	bias := make([][]float32, len(conds))
	for k, cd := range conds {
		xs[k], bias[k] = patches, cbias(cd.Rows)
	}
	for i, b := range e.blks {
		for k, cd := range conds {
			if xs[k], err = b.Run(ctx, map[string][]float32{"x": xs[k], "t6": t6, "ctx": cd.ctx, "cbias": bias[k]}); err != nil {
				return nil, fmt.Errorf("blocks group %d: %w", i, err)
			}
		}
	}
	res := make([][]float32, len(conds))
	for k := range conds {
		if res[k], err = e.fin.Run(ctx, map[string][]float32{"x": xs[k], "emb": emb}); err != nil {
			return nil, fmt.Errorf("dit_final: %w", err)
		}
	}
	return res, nil
}

// decode runs the VAE's stages, each loaded while it runs; the last gives the picture (1, 3, h,
// w).
func (e *progEngine) decode(ctx context.Context, latents []float32, stage StageFunc) (*image.RGBA, error) {
	x := latents
	for i, name := range e.m.Vae {
		t0 := time.Now()
		var err error
		if x, err = e.once(ctx, name, map[string][]float32{"x": x}); err != nil {
			return nil, fmt.Errorf("vae stage %d: %w", i, err)
		}
		if stage != nil {
			stage(i, len(e.m.Vae), time.Since(t0))
		}
	}
	W, H := e.m.Dit.W(), e.m.Dit.H()
	if len(x) != 3*W*H {
		return nil, fmt.Errorf("vae: %d values for a %dx%d picture", len(x), W, H)
	}
	img := image.NewRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for xx := 0; xx < W; xx++ {
			i := y*W + xx
			img.SetRGBA(xx, y, color.RGBA{pixel(x[i]), pixel(x[W*H+i]), pixel(x[2*W*H+i]), 255})
		}
	}
	return img, nil
}

func (e *progEngine) close() error { return e.closeDiT() }
