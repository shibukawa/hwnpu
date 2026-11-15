package qwenimage

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/imagegen/qwenimage/mil"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// fakeLoader loads programs that write down what happens to them: "+name" a load, "name" a run,
// "-name" a close. A run gives, for each value of x, half of it plus the text's first key (0
// without one) and a hundredth of its index; out values of x's first plus 1 for a program named
// in out.
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
	p.l.log = append(p.l.log, p.name)
	if n := p.l.out[p.name]; n > 0 {
		y := make([]float32, n)
		for i := range y {
			y[i] = x[0] + 1
		}
		return y, nil
	}
	var key float32
	if tk := in["tk"]; len(tk) > 0 {
		key = tk[0]
	}
	y := make([]float32, len(x))
	for j := range y {
		y[j] = 0.5*x[j] + key + 0.01*float32(j)
	}
	return y, nil
}

// Pictures drawn together are the pictures drawn one after another, with both solvers and with
// guidance at some of the steps; each group of the DiT's blocks is loaded once a step and runs
// every pass of every picture before the next group.
func TestGenerateAllPrograms(t *testing.T) {
	ctx := context.Background()
	dc := DitConfig{Width: 32, Height: 32, HeadDim: 6, Axes: []int{2, 2, 2}, Theta: 10000, Latent: 2, TextRows: 4, VAEScale: mil.VAEScale}
	pipeline := func() (*Pipeline, *fakeLoader) {
		l := &fakeLoader{}
		e := &progEngine{m: &ProgramManifest{Dit: dc, Blocks: []string{"b0", "b1"}}, loader: l.load, swapDiT: true,
			time: &timeEmbed{E: 1, t1: make([]float32, 256), t2: []float32{0}, mod: make([]float32, 4), norm: []float32{0}}}
		return &Pipeline{Dit: dc, prog: e}, l
	}
	cond := func(v float32, rows int) Cond {
		return Cond{ptk: [][]float32{{v}, {v}}, ptv: [][]float32{{v}, {v}}, Rows: rows}
	}
	conds := []Cond{cond(0.1, 2), cond(-0.2, 3), cond(0.3, 1)}
	neg := cond(0.05, 1)
	noises := make([][]float32, len(conds))
	for k := range noises {
		noises[k] = make([]float32, dc.Tokens()*dc.Latent)
		for j := range noises[k] {
			noises[k][j] = float32(math.Sin(float64(7*k + j + 1)))
		}
	}
	sig := []float64{1, 0.75, 0.5, 0.25, 0}
	for _, solver := range []string{"", "dpm2m"} {
		for _, g := range []float64{1, 3} {
			name := fmt.Sprintf("%q, guidance %g", solver, g)
			smp := Sampling{Solver: solver, Guidance: g, CFGMin: 0.6, CFGMax: 1}
			p, l := pipeline()
			steps := 0
			all, err := p.GenerateAll(ctx, conds, &neg, noises, sig, smp, func(i, n int, _ time.Duration) {
				if i != steps || n != 4 {
					t.Errorf("%s: step %d of %d after %d", name, i, n, steps)
				}
				steps++
			})
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if steps != 4 {
				t.Errorf("%s: told of %d steps", name, steps)
			}
			for k := range conds {
				p1, _ := pipeline()
				one, err := p1.Generate(ctx, conds[k], &neg, noises[k], sig, smp, nil)
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				if !slices.Equal(all[k], one) {
					t.Errorf("%s: picture %d drawn with the others is %v, alone %v", name, k, all[k], one)
				}
			}
			// the groups are loaded once to check them, then for each step: guided (two passes a
			// picture) at sigma 1 and 0.75
			want := []string{"+b0 -b0 +b1 -b1"}
			for i := 0; i < 4; i++ {
				passes := 3
				if g != 1 && i < 2 {
					passes = 6
				}
				for _, b := range []string{"b0", "b1"} {
					want = append(want, "+"+b+strings.Repeat(" "+b, passes)+" -"+b)
				}
			}
			if got := strings.Join(l.log, " "); got != strings.Join(want, " ") {
				t.Errorf("%s: the groups ran as\n%s\nwant\n%s", name, got, strings.Join(want, " "))
			}
		}
	}
	p, _ := pipeline()
	if _, err := p.GenerateAll(ctx, conds, nil, noises[:2], sig, Sampling{}, nil); err == nil {
		t.Error("three conditionings for two noises ran")
	}
}

// The VAE runs stage by stage over the pictures (runner.Stages), each stage loaded once for
// them.
func TestDecodeStageByStage(t *testing.T) {
	const W, H = 1024, 1024
	l := &fakeLoader{out: map[string]int{"v0": 1, "v1": 4 * W * H}}
	e := &progEngine{m: &ProgramManifest{Vae: []string{"v0", "v1"}, Dit: DitConfig{Width: W, Height: H}}, loader: l.load}
	var stages []string
	zs := [][]float32{{-2.5}, {-2}, {-1.5}}
	imgs, err := e.decode(context.Background(), zs, func(i, n int, _ time.Duration) { stages = append(stages, fmt.Sprintf("%d/%d", i, n)) })
	if err != nil {
		t.Fatal(err)
	}
	want := "+v0 v0 v0 v0 -v0 +v1 v1 v1 v1 -v1"
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
		// every channel, alpha too, is the latents' first value plus 2: premultiplied
		v := int(pixel(zs[k][0] + 2))
		if want := uint8((v*v + 127) / 255); img.Pix[0] != want || img.Pix[3] != uint8(v) {
			t.Errorf("picture %d: %d with alpha %d, want %d with %d", k, img.Pix[0], img.Pix[3], want, v)
		}
	}
}
