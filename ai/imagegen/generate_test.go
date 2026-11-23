package imagegen

import (
	"image"
	"slices"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
)

func TestSizeFor(t *testing.T) {
	g := &Generator{sizes: []image.Point{{1024, 1024}, {512, 512}, {832, 1248}, {1248, 832}, {1360, 752}}}
	for _, c := range []struct{ w, h, wantW, wantH int }{
		{0, 0, 1024, 1024}, // nothing asked: the first
		{1024, 1024, 1024, 1024},
		{2000, 2000, 1024, 1024}, // a square: the larger one for a large picture
		{400, 400, 512, 512},     // and the smaller for a small one
		{1920, 1080, 1360, 752},
		{3000, 2000, 1248, 832},
		{600, 900, 832, 1248},
		{1100, 1000, 1024, 1024}, // the nearest shape
	} {
		if got := g.SizeFor(c.w, c.h); got != image.Pt(c.wantW, c.wantH) {
			t.Errorf("SizeFor(%d, %d) = %v, want %d x %d", c.w, c.h, got, c.wantW, c.wantH)
		}
	}
}

// A run's batches: the seeds from Options.Seed, the steps' and stages' times, the pictures told
// as they are drawn.
func TestRunBatches(t *testing.T) {
	g := &Generator{sizes: []image.Point{{512, 512}}}
	var batches []Batch
	var steps []Step
	var drawn []int
	o := Options{Seed: 10, OnBatch: func(b Batch) { batches = append(batches, b) }, OnStep: func(s Step) { steps = append(steps, s) },
		Drawn: func(i int, _ *image.RGBA) { drawn = append(drawn, i) }}
	r := g.newRun([]string{"a", "b", "c"}, o)
	noise := func(seed uint64) []float32 { return []float32{float32(seed)} }
	sig := []float64{1, 0.5, 0}
	for lo := 0; lo < 3; lo += 2 {
		hi := min(lo+2, 3)
		xs := r.batch(lo, hi, 2, noise)
		if len(xs) != hi-lo || xs[0][0] != float32(10+lo) {
			t.Fatalf("batch %d: noise %v", lo, xs)
		}
		step := r.stepsOf(sig)
		step(0, 2, time.Second)
		step(1, 2, 2*time.Second)
		r.stage(0, 1, time.Second)
		if err := r.drawn(lo, time.Second, make([]*image.RGBA, hi-lo)...); err != nil {
			t.Fatal(err)
		}
	}
	p := r.finish()
	if got := p.Seeds; len(got) != 3 || got[0] != 10 || got[2] != 12 {
		t.Errorf("seeds %v", got)
	}
	if len(batches) != 2 || batches[1].First != 2 || batches[1].N != 1 || batches[0].Steps != 2 || len(batches[0].Seeds) != 2 {
		t.Errorf("batches %+v", batches)
	}
	if len(steps) != 4 || steps[1].From != 0.5 || steps[1].To != 0 {
		t.Errorf("steps %+v", steps)
	}
	if b := p.Batches[0]; len(b.Steps) != 2 || b.Denoise != 3*time.Second || len(b.Stages) != 1 || b.Decode != time.Second {
		t.Errorf("stats %+v", b)
	}
	if len(drawn) != 3 || drawn[2] != 2 || p.Size != image.Pt(512, 512) {
		t.Errorf("drawn %v, size %v", drawn, p.Size)
	}
	// Checks.Noise in place of the seeds'
	r = g.newRun([]string{"a"}, Options{Checks: &Checks{Noise: []float32{3}}})
	if xs := r.batch(0, 1, 1, noise); xs[0][0] != 3 || r.out.Seeds[0] != 0 {
		t.Errorf("noise %v, seeds %v", xs, r.out.Seeds)
	}
}

func TestReferenceNames(t *testing.T) {
	if got := referenceNames([]int{2, 1, 2}); got != "1+2@half" {
		t.Errorf("names: %q", got)
	}
}

// The settings of a definition: Anima's conversion, Chroma's square.
func TestSettings(t *testing.T) {
	m := chat.Model{Name: "Anima-Turbo", File: "anima-turbo-v1.1.safetensors", Settings: map[string]string{SettingGroup: "2", SettingSize: "16:9", SettingDitFormat: "fp16"}}
	o, err := animaOptions(m, "intel")
	if err != nil {
		t.Fatal(err)
	}
	if o.Group != 2 || o.QChunk != 512 || o.Width != 1360 || o.Height != 752 || o.Dit != m.File || o.DitFormat != "fp16" || o.Sizes != nil {
		t.Errorf("anima %+v", o)
	}
	if o, _ := animaOptions(chat.Model{}, "coreai"); len(o.Sizes) != 8 || o.Width != 1024 {
		t.Errorf("anima on Core AI: %+v", o)
	}
	if _, err := animaOptions(chat.Model{Settings: map[string]string{SettingQChunk: "x"}}, "intel"); err == nil {
		t.Error("qchunk x: no error")
	}
	for spec, want := range map[string]int{"": 1024, "1:1s": 512, "768": 768, "16:9": 0} {
		side, err := chromaSide(chat.Model{Settings: map[string]string{SettingSize: spec}})
		if side != want || (want == 0) != (err != nil) {
			t.Errorf("chroma %q: %d, %v", spec, side, err)
		}
	}
}

func TestExportNone(t *testing.T) {
	if _, err := Export(t.TempDir()); err == nil {
		t.Error("an empty directory is an export")
	}
}

// An edit's size and references: the nearest size with an edit of that many references (Apple's
// edits of models.yaml), or any number of the picture's size (AMD's).
func TestEditFor(t *testing.T) {
	big, small := image.Pt(1024, 1024), image.Pt(512, 512)
	g := &Generator{edits: true, sizes: []image.Point{big, small, {768, 1152}}, editSizes: []image.Point{big, small},
		editRefs: map[image.Point][][]int{big: {{1}, {1, 2}, {1, 2, 2}}, small: {{1}, {1, 1}}}}
	for _, c := range []struct {
		w, h, n int
		size    image.Point
		divs    []int
	}{
		{1024, 1024, 1, big, []int{1}},
		{1024, 1024, 2, big, []int{1, 2}},
		{1024, 1024, 3, big, []int{1, 2, 2}},
		{512, 512, 1, small, []int{1}},
		{512, 512, 2, small, []int{1, 1}},
		{512, 512, 3, big, []int{1, 2, 2}}, // three only at 1024
		{1920, 1080, 2, big, []int{1, 2}},  // the nearest of the two
	} {
		s, d, err := g.editFor(c.w, c.h, c.n)
		if err != nil || s != c.size || !slices.Equal(d, c.divs) {
			t.Errorf("editFor(%d, %d, %d) = %v %v %v, want %v %v", c.w, c.h, c.n, s, d, err, c.size, c.divs)
		}
	}
	if _, _, err := g.editFor(1024, 1024, 4); err == nil {
		t.Error("4 references: no error")
	}
	if r := g.newRun([]string{"a"}, Options{Width: 512, Height: 512, References: make([]image.Image, 3)}); r.size != big {
		t.Errorf("a run of 3 references at 512: %v", r.size)
	}
	// AMD's NPU: any number at every size, of the picture's size
	g = &Generator{edits: true, sizes: []image.Point{big, small}}
	if s, d, err := g.editFor(512, 512, 2); err != nil || s != small || !slices.Equal(d, []int{1, 1}) {
		t.Errorf("AMD: %v %v %v", s, d, err)
	}
	if got := g.EditSizes(); len(got) != 2 {
		t.Errorf("AMD's edit sizes %v", got)
	}
}
