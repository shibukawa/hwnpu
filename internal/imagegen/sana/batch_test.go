package sana

import (
	"context"
	"fmt"
	"image"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// fakeLoader loads programs that write down what happens to them: "+name" a load, "name(v)" a run
// on an x that starts with v, "-name" a close. A run gives out values (1 when 0) of x's first plus
// 1.
type fakeLoader struct {
	log []string
	out map[string]int
}

func (l *fakeLoader) load(_ context.Context, name string) (runner.Program, error) {
	l.log = append(l.log, "+"+name)
	return &fakeProgram{l: l, name: name}, nil
}

type fakeProgram struct {
	l    *fakeLoader
	name string
}

func (p *fakeProgram) Inputs() map[string][]int    { return nil }
func (p *fakeProgram) Output() []int               { return nil }
func (p *fakeProgram) Metadata() map[string]string { return nil }
func (p *fakeProgram) OnNPU() bool                 { return true }
func (p *fakeProgram) Close() error                { p.l.log = append(p.l.log, "-"+p.name); return nil }
func (p *fakeProgram) Run(_ context.Context, in map[string][]float32) ([]float32, error) {
	x := in["x"]
	p.l.log = append(p.l.log, fmt.Sprintf("%s(%g)", p.name, x[0]))
	y := make([]float32, max(1, p.l.out[p.name]))
	for i := range y {
		y[i] = x[0] + 1
	}
	return y, nil
}

// Gemma 2's programs are loaded once each for all the prompts of a run, and a prompt's rows are
// what they are when it runs alone.
func TestTextRunAll(t *testing.T) {
	l := &fakeLoader{}
	e := &progEngine{m: &ProgramManifest{TextParts: []string{"t0", "t1", "t2"}}, loader: l.load}
	ys, err := textPart{e}.runAll(context.Background(), [][]float32{{1}, {10}, {100}})
	if err != nil {
		t.Fatal(err)
	}
	if ys[0][0] != 4 || ys[1][0] != 13 || ys[2][0] != 103 {
		t.Errorf("rows %v", ys)
	}
	want := "+t0 t0(1) t0(10) t0(100) -t0 +t1 t1(2) t1(11) t1(101) -t1 +t2 t2(3) t2(12) t2(102) -t2"
	if got := strings.Join(l.log, " "); got != want {
		t.Errorf("the programs ran as\n%s\nwant\n%s", got, want)
	}
	l.log = nil
	y, err := textPart{e}.run(context.Background(), []float32{10}, 0)
	if err != nil || y[0] != ys[1][0] {
		t.Errorf("alone: %v, %v; together %v", y, err, ys[1])
	}
}

// The VAE runs stage by stage over the pictures (runner.Stages), each stage loaded once for
// them, and a picture is what it is when decoded alone.
func TestDecodeAllStageByStage(t *testing.T) {
	const W, H = 1024, 1024
	l := &fakeLoader{out: map[string]int{"v1": 3 * W * H}}
	e := &progEngine{m: &ProgramManifest{Vae: []string{"v0", "v1"}, Dit: DitConfig{Width: W, Height: H}}, loader: l.load}
	var stages []string
	stage := func(i, n int, _ time.Duration) { stages = append(stages, fmt.Sprintf("%d/%d", i, n)) }
	// the latents' first values put a picture's pixels at -0.5, 0 and 0.5
	lat := [][]float32{{-2.5}, {-2}, {-1.5}}
	imgs, err := e.decodeAll(context.Background(), lat, stage)
	if err != nil {
		t.Fatal(err)
	}
	want := "+v0 v0(-2.5) v0(-2) v0(-1.5) -v0 +v1 v1(-1.5) v1(-1) v1(-0.5) -v1"
	if got := strings.Join(l.log, " "); got != want {
		t.Errorf("the stages ran as\n%s\nwant\n%s", got, want)
	}
	if got := strings.Join(stages, " "); got != "2/6 5/6" {
		t.Errorf("stages told as %s", got)
	}
	if len(imgs) != 3 {
		t.Fatalf("%d pictures", len(imgs))
	}
	for k, img := range imgs {
		one, err := e.decode(context.Background(), lat[k], nil)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(img.Pix, one.Pix) {
			t.Errorf("picture %d is not the one decoded alone", k)
		}
		if want := pixel(lat[k][0] + 2); img.Pix[0] != want || img.Bounds() != image.Rect(0, 0, W, H) {
			t.Errorf("picture %d: first value %d (want %d), bounds %v", k, img.Pix[0], want, img.Bounds())
		}
	}
}

// togetherEngine decodes several pictures at once; plainEngine one.
type togetherEngine struct {
	countEngine
	closed int
	all    [][]float32
}

func (e *togetherEngine) closeDiT() error { e.closed++; return nil }
func (e *togetherEngine) decodeAll(_ context.Context, latents [][]float32, _ StageFunc) ([]*image.RGBA, error) {
	e.all = latents
	return make([]*image.RGBA, len(latents)), nil
}

type plainEngine struct {
	engine
	closed, decoded int
}

func (e *plainEngine) closeDiT() error { e.closed++; return nil }
func (e *plainEngine) decode(context.Context, []float32, StageFunc) (*image.RGBA, error) {
	e.decoded++
	return nil, nil
}

// DecodeAll hands an engine of programs every picture at once, and decodes one after another with
// an engine that takes one; the DiT is released first unless it is kept.
func TestDecodeAll(t *testing.T) {
	ctx := context.Background()
	te := &togetherEngine{}
	p := &Pipeline{eng: te}
	if !p.Batched() {
		t.Error("an engine that decodes pictures together is not batched")
	}
	imgs, err := p.DecodeAll(ctx, []float32{1}, []float32{2}, []float32{3})
	if err != nil || len(imgs) != 3 || len(te.all) != 3 || te.closed != 1 {
		t.Errorf("%d pictures of %d latents, %d closes, %v", len(imgs), len(te.all), te.closed, err)
	}
	p.KeepDiT = true
	if _, err := p.DecodeAll(ctx, []float32{1}); err != nil || te.closed != 1 {
		t.Errorf("the DiT kept: %d closes, %v", te.closed, err)
	}
	pe := &plainEngine{}
	p = &Pipeline{eng: pe}
	if p.Batched() {
		t.Error("an engine that decodes one picture is batched")
	}
	imgs, err = p.DecodeAll(ctx, []float32{1}, []float32{2})
	if err != nil || len(imgs) != 2 || pe.decoded != 2 || pe.closed != 1 {
		t.Errorf("%d pictures, %d decodes, %d closes, %v", len(imgs), pe.decoded, pe.closed, err)
	}
}
