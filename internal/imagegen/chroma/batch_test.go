package chroma

import (
	"context"
	"fmt"
	"image"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// stepEngine's DiT gives a velocity that depends on the latents, the conditioning and sigma, a
// picture at a time; it counts its calls.
type stepEngine struct {
	engine
	calls int
}

func (e *stepEngine) openDiT(context.Context, []float64) error { return nil }
func (e *stepEngine) closeDiT() error                          { return nil }

func fakeVelocity(cd Cond, x []float32, sigma float64) []float32 {
	v := make([]float32, len(x))
	for j := range v {
		v[j] = 0.5*x[j] + cd.rows[0] + float32(sigma) + 0.01*float32(j)
	}
	return v
}

func (e *stepEngine) velocity(_ context.Context, conds []Cond, x []float32, sigma float64) ([][]float32, error) {
	e.calls++
	out := make([][]float32, len(conds))
	for k, cd := range conds {
		out[k] = fakeVelocity(cd, x, sigma)
	}
	return out, nil
}

// togetherEngine is stepEngine taking every picture's passes in one call, and every picture's
// latents in one decoding; it writes down the passes of each call.
type togetherEngine struct {
	stepEngine
	passes  []int
	decoded int
}

func (e *togetherEngine) velocities(_ context.Context, conds []Cond, xs [][]float32, sigma float64) ([][]float32, error) {
	e.calls++
	e.passes = append(e.passes, len(conds))
	out := make([][]float32, len(conds))
	for k, cd := range conds {
		out[k] = fakeVelocity(cd, xs[k], sigma)
	}
	return out, nil
}

func (e *togetherEngine) decodeAll(_ context.Context, zs [][]float32, _ anima.StageFunc) ([]*image.RGBA, error) {
	e.decoded += len(zs)
	return make([]*image.RGBA, len(zs)), nil
}

// Pictures drawn together are the pictures drawn one after another, with every solver and with
// guidance at some of the steps; an engine of programs gets each call's passes of every picture
// at once.
func TestGenerateAll(t *testing.T) {
	ctx := context.Background()
	c := Config{Width: 32, Height: 32, Latent: 2}
	conds := []Cond{{rows: []float32{0.1}}, {rows: []float32{-0.2}}, {rows: []float32{0.3}}}
	neg := Cond{rows: []float32{0.05}}
	noises := make([][]float32, len(conds))
	for k := range noises {
		noises[k] = make([]float32, c.Tokens()*c.Latent)
		for j := range noises[k] {
			noises[k][j] = float32(math.Sin(float64(7*k + j + 1)))
		}
	}
	sig := Sigmas(4) // 1, 0.75, 0.5, 0.25, 0
	for _, solver := range []string{"heun", "euler", "dpm2m"} {
		for _, g := range []float64{1, 3} {
			name := fmt.Sprintf("%s, guidance %g", solver, g)
			smp := Sampling{Solver: solver, Guidance: g, CFGMin: 0.6, CFGMax: 1}
			te := &togetherEngine{}
			steps := 0
			all, err := (&Pipeline{C: c, eng: te}).GenerateAll(ctx, conds, &neg, noises, sig, smp, func(i, n int, _ time.Duration) {
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
				one, err := (&Pipeline{C: c, eng: &stepEngine{}}).Generate(ctx, conds[k], &neg, noises[k], sig, smp, nil)
				if err != nil {
					t.Fatalf("%s: %v", name, err)
				}
				if !slices.Equal(all[k], one) {
					t.Errorf("%s: picture %d drawn with the others is %v, alone %v", name, k, all[k], one)
				}
			}
			// the calls of the DiT: Heun's two a step but the last; guided (two passes a picture)
			// at sigma 1 and 0.75
			want := []int{3, 3, 3, 3}
			if solver == "heun" {
				want = []int{3, 3, 3, 3, 3, 3, 3} // at 1, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25
			}
			if g != 1 {
				want[0], want[1] = 6, 6
				if solver == "heun" {
					want[2] = 6
				}
			}
			if !slices.Equal(te.passes, want) {
				t.Errorf("%s: the calls ran %v passes, want %v", name, te.passes, want)
			}
			// an engine that takes one picture at a time draws the same
			pe := &stepEngine{}
			again, err := (&Pipeline{C: c, eng: pe}).GenerateAll(ctx, conds, &neg, noises, sig, smp, nil)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			for k := range conds {
				if !slices.Equal(again[k], all[k]) {
					t.Errorf("%s: picture %d differs with an engine of one picture", name, k)
				}
			}
			if pe.calls != len(conds)*len(want) {
				t.Errorf("%s: %d calls of an engine of one picture, want %d", name, pe.calls, len(conds)*len(want))
			}
		}
	}
	if _, err := (&Pipeline{C: c, eng: &stepEngine{}}).GenerateAll(ctx, conds, nil, noises[:2], sig, Sampling{Solver: "euler"}, nil); err == nil {
		t.Error("three conditionings for two noises ran")
	}
}

func TestBatched(t *testing.T) {
	te := &togetherEngine{}
	p := &Pipeline{C: Config{Width: 32, Height: 32, Latent: 64}, eng: te}
	if !p.Batched() || (&Pipeline{eng: &stepEngine{}}).Batched() {
		t.Error("Batched is not the engine of programs'")
	}
	x := make([]float32, p.C.Tokens()*p.C.Latent)
	if imgs, err := p.DecodeAll(context.Background(), x, x, x); err != nil || len(imgs) != 3 || te.decoded != 3 {
		t.Errorf("%d pictures, %d decoded, %v", len(imgs), te.decoded, err)
	}
}

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

// A call of the DiT runs each group of its blocks on every pass before the next group, a group
// loaded once for all of them; so does the VAE its stages.
func TestProgramsGroupByGroup(t *testing.T) {
	ctx := context.Background()
	const W, H = 1024, 1024
	l := &fakeLoader{out: map[string]int{"v1": 3 * W * H}}
	m := &ProgramManifest{Doubles: []Group{{"d0", 0, 1}, {"d1", 1, 1}}, Singles: []Group{{"s0", 0, 2}}, Vae: []string{"v0", "v1"}}
	m.Dit = Config{Width: W, Height: H, TextRows: 2, Hidden: 1, Doubles: 2, Singles: 2, Mods: 3*2 + 12*2 + 2}
	e := &progEngine{m: m, loader: l.load, swapDiT: true, logf: func(string, ...any) {}}
	e.mods = map[float64][]float32{0.5: make([]float32, m.Dit.Mods)}
	e.ring = runner.NewRing(e.groups(), false, e.load)
	conds := []Cond{{rows: []float32{0}, Keys: 1}, {rows: []float32{0}, Keys: 2}, {rows: []float32{0}, Keys: 1}}
	vs, err := e.velocities(ctx, conds, [][]float32{{1}, {10}, {100}}, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if vs[0][0] != 4 || vs[1][0] != 13 || vs[2][0] != 103 {
		t.Errorf("velocities %v", vs)
	}
	want := "+d0 d0(1) d0(10) d0(100) -d0 +d1 d1(2) d1(11) d1(101) -d1 +s0 s0(3) s0(12) s0(102) -s0"
	if got := strings.Join(l.log, " "); got != want {
		t.Errorf("the groups ran as\n%s\nwant\n%s", got, want)
	}
	l.log = nil
	one, err := e.velocity(ctx, conds[1:2], []float32{10}, 0.5)
	if err != nil || one[0][0] != vs[1][0] {
		t.Errorf("alone: %v, %v; together %v", one, err, vs[1])
	}
	// the VAE: stage by stage over the pictures (runner.Stages)
	l.log = nil
	var stages []string
	zs := [][]float32{{-2.5}, {-2}, {-1.5}}
	imgs, err := e.decodeAll(ctx, zs, func(i, n int, _ time.Duration) { stages = append(stages, fmt.Sprintf("%d/%d", i, n)) })
	if err != nil {
		t.Fatal(err)
	}
	want = "+v0 v0(-2.5) v0(-2) v0(-1.5) -v0 +v1 v1(-1.5) v1(-1) v1(-0.5) -v1"
	if got := strings.Join(l.log, " "); got != want {
		t.Errorf("the stages ran as\n%s\nwant\n%s", got, want)
	}
	if got := strings.Join(stages, " "); got != "2/6 5/6" {
		t.Errorf("stages told as %s", got)
	}
	for k, img := range imgs {
		if want := pixel(zs[k][0] + 2); img.Pix[0] != want {
			t.Errorf("picture %d: first value %d, want %d", k, img.Pix[0], want)
		}
	}
}
