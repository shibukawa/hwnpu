package flux2klein

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func testPipeline(size int) *Pipeline {
	p := &Pipeline{C: Config{Width: size, Height: size, TextRows: 512, Hidden: 3072, HeadDim: 128, Latent: 128, Text: 2560,
		Axes: []int{32, 32, 32, 32}, Theta: 2000, Doubles: 5, Singles: 20, DoubleGroup: 1, SingleGroup: 2}}
	p.base = p.C // an export of one size
	p.Model.Steps = 4
	p.bnMean, p.bnStd = make([]float32, 128), make([]float32, 128)
	for i := range p.bnMean {
		p.bnMean[i], p.bnStd[i] = float32(i)/100-0.5, 1+float32(i%7)/10
	}
	return p
}

// The pipeline's sigmas of a 512 x 512 picture in 4 steps (tools/ref/ref_flux2klein.py prints them).
func TestSigmas(t *testing.T) {
	got := testPipeline(512).Sigmas(0)
	want := []float64{1, 0.95808536, 0.8839819, 0.7174966, 0}
	if len(got) != len(want) {
		t.Fatalf("%d sigmas, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-5 {
			t.Errorf("sigma %d: %.6f, want %.6f", i, got[i], want[i])
		}
	}
	// past 4300 rows the shift no longer depends on the steps
	if a, b := EmpiricalMu(8192, 4), EmpiricalMu(8192, 50); a != b {
		t.Errorf("mu of 8192 rows: %g at 4 steps, %g at 50", a, b)
	}
}

// Unpack un-normalizes and lays the 2 x 2 patches out: channel c of the 32, at (2 y + dy, 2 x + dx),
// is the row's value c*4 + dy*2 + dx (the pipeline's _unpatchify_latents).
func TestUnpack(t *testing.T) {
	p := testPipeline(64)
	c := p.C
	x := make([]float32, c.Tokens()*c.Latent)
	for i := range x {
		x[i] = float32(i%251) / 50
	}
	z := p.Unpack(x)
	H, W := 2*c.GH(), 2*c.GW()
	if len(z) != 32*H*W {
		t.Fatalf("%d values, want %d", len(z), 32*H*W)
	}
	for _, at := range [][5]int{{0, 0, 0, 0, 0}, {5, 1, 2, 1, 0}, {31, 3, 3, 1, 1}, {17, 2, 0, 0, 1}} {
		ch, py, px, dy, dx := at[0], at[1], at[2], at[3], at[4]
		k := ch*4 + dy*2 + dx
		want := x[(py*c.GW()+px)*c.Latent+k]*p.bnStd[k] + p.bnMean[k]
		if got := z[(ch*H+2*py+dy)*W+2*px+dx]; got != want {
			t.Errorf("channel %d at patch (%d, %d) + (%d, %d): %g, want %g", ch, py, px, dy, dx, got, want)
		}
	}
}

// The rows' positions: the picture's, each reference's on the first axis at 10, 20, ..., then the text's.
func TestPositions(t *testing.T) {
	p := testPipeline(64)
	N := p.C.Tokens()
	pos := p.positions([]int{1, 1})
	if len(pos) != 3*N+p.C.TextRows {
		t.Fatalf("%d rows, want %d", len(pos), 3*N+p.C.TextRows)
	}
	for _, c := range []struct {
		row  int
		want [4]float64
	}{{0, [4]float64{0, 0, 0, 0}}, {N - 1, [4]float64{0, 3, 3, 0}}, {N + 5, [4]float64{10, 1, 1, 0}}, {2 * N, [4]float64{20, 0, 0, 0}},
		{3 * N, [4]float64{0, 0, 0, 0}}, {3*N + 7, [4]float64{0, 0, 0, 7}}} {
		if pos[c.row] != c.want {
			t.Errorf("row %d: %v, want %v", c.row, pos[c.row], c.want)
		}
	}
	// references of half the sides: a quarter of the rows each, their own grid
	half := p.positions([]int{2})
	if len(half) != N+N/4+p.C.TextRows {
		t.Fatalf("%d rows with a half-size reference, want %d", len(half), N+N/4+p.C.TextRows)
	}
	if got, want := half[N+3], ([4]float64{10, 1, 1, 0}); got != want {
		t.Errorf("row %d of a half-size reference: %v, want %v", N+3, got, want)
	}
	// a mix: the second reference's rows follow the first's, on its own grid
	mix := p.positions([]int{2, 1})
	if len(mix) != N+N/4+N+p.C.TextRows {
		t.Fatalf("%d rows with a half-size and a full-size reference, want %d", len(mix), N+N/4+N+p.C.TextRows)
	}
	if got, want := mix[N+N/4+5], ([4]float64{20, 1, 1, 0}); got != want {
		t.Errorf("row %d of the second reference: %v, want %v", N+N/4+5, got, want)
	}
	// the graphs' names: by a step's rows, whatever the references that make them up
	base := N + p.C.TextRows
	for _, c := range []struct {
		divs []int
		rows int
	}{{nil, base}, {[]int{1}, base + N}, {[]int{2, 1}, base + N + N/4}, {[]int{1, 2}, base + N + N/4}, {[]int{2, 2, 2, 2}, base + N}} {
		rows, ok := p.rows(c.divs)
		if !ok || rows != c.rows {
			t.Errorf("rows of %v: %d %v, want %d", c.divs, rows, ok, c.rows)
		}
	}
	if _, ok := p.rows([]int{3}); ok {
		t.Error("rows: a reference of a third of a 4 x 4 grid's sides")
	}
	if a, b := p.suffix(base), p.suffix(base+N); a != "" || b != fmt.Sprintf("_n%d", base+N) {
		t.Errorf("suffixes %q %q", a, b)
	}
	a := p.ropeAngles(pos[:1])
	for i := 0; i < p.C.HeadDim; i += 2 { // position 0: no turn
		if a[i] != 1 || a[i+1] != 0 {
			t.Fatalf("angle %d of position 0: cos %g sin %g", i/2, a[i], a[i+1])
		}
	}
}

// The references an export runs: the mixes whose rows it has blocks for and whose sizes it has
// an encoder for; a step whose rows are no whole rounds of 512 runs padded, the text's rows first.
func TestReferences(t *testing.T) {
	p := testPipeline(512) // 1024 picture rows, 512 text rows; a reference 1024 rows, a half-size one 256
	p.dir = t.TempDir()
	have := func(names ...string) {
		for _, n := range names {
			d := filepath.Join(p.dir, n)
			if err := os.MkdirAll(d, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(d, "manifest.json"), []byte("{}"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	if got := p.References(); !reflect.DeepEqual(got, []string{"0"}) {
		t.Errorf("no references: %v", got)
	}
	// 2560 rows: one of the picture's size, or four of half its sides
	have("vae_enc", "doubles_n2560", "singles_n2560", "dit_in_n2560")
	if got := p.References(); !reflect.DeepEqual(got, []string{"0", "1"}) {
		t.Errorf("without the half-size encoder: %v", got)
	}
	// one of half the sides: 1792 rows, padded to 2048 with the text's first
	have("vae_enc_d2", "doubles_t2048", "singles_n2048", "dit_in_n2048")
	if got := p.References(); !reflect.DeepEqual(got, []string{"0", "1@half", "4@half", "1"}) {
		t.Errorf("with it: %v", got)
	}
	// two of half the sides are 2048 rows exactly: the image's rows first (doubles_n2048)
	if !p.HasReferences([]int{2, 2, 2, 2}) || !p.HasReferences([]int{2}) || p.HasReferences([]int{2, 2}) || p.HasReferences([]int{1, 2}) {
		t.Error("HasReferences: 4@half and 1@half run, 2@half and 1+1@half do not")
	}
	for _, c := range []struct {
		divs    []int
		want    step
		imageAt int
	}{
		{nil, step{rows: 1536, padded: 1536, doubles: "doubles", singles: "singles", ditIn: "dit_in"}, 0},
		{[]int{2}, step{rows: 1792, padded: 2048, textFirst: true, doubles: "doubles_t2048", singles: "singles_n2048", ditIn: "dit_in_n2048"}, 512},
		{[]int{2, 2}, step{rows: 2048, padded: 2048, doubles: "doubles_n2048", singles: "singles_n2048", ditIn: "dit_in_n2048"}, 0},
	} {
		got, ok := p.step(c.divs)
		if !ok || got != c.want || got.imageAt(p.C.TextRows) != c.imageAt {
			t.Errorf("step of %v: %+v (image at %d), want %+v (at %d)", c.divs, got, got.imageAt(p.C.TextRows), c.want, c.imageAt)
		}
	}
	// a picture that is no square: 832 x 1248 is 4056 rows and the text's, padded to the 1024 x 1024 export's 4608
	q := testPipeline(1024)
	q.C.Width, q.C.Height = 832, 1248
	if got, _ := q.step(nil); got != (step{rows: 4568, padded: 4608, textFirst: true, doubles: "doubles_t4608", singles: "singles", ditIn: "dit_in"}) {
		t.Errorf("832 x 1248: %+v", got)
	}
}
