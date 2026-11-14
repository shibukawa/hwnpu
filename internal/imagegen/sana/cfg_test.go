package sana

import (
	"context"
	"image"
	"testing"
)

// countEngine's DiT gives each conditioning's first value as the velocity, and counts the calls
// and the passes.
type countEngine struct {
	engine
	calls, passes int
}

func (e *countEngine) openDiT(context.Context, bool) error { return nil }
func (e *countEngine) closeDiT() error                     { return nil }
func (e *countEngine) decode(context.Context, []float32, StageFunc) (*image.RGBA, error) {
	return nil, nil
}

func (e *countEngine) velocity(_ context.Context, patches, _ []float32, conds []Cond) ([][]float32, error) {
	e.calls++
	e.passes += len(conds)
	res := make([][]float32, len(conds))
	for k, cd := range conds {
		res[k] = make([]float32, len(patches))
		for j := range res[k] {
			res[k][j] = cd.ctx[0]
		}
	}
	return res, nil
}

// TestCFGSigmas checks that only the steps of sigma in [CFGMin, CFGMax] run the DiT on both
// conditionings, and that the others take the prompt's velocity alone.
func TestCFGSigmas(t *testing.T) {
	cond, neg := Cond{ctx: []float32{1}, Rows: 1}, Cond{ctx: []float32{-1}, Rows: 1}
	run := func(lo, hi float64) (*countEngine, []float32) {
		e := &countEngine{}
		p := &Pipeline{eng: e, Dit: DitConfig{Width: 32, Height: 32, Latent: 1, VAEScale: 32}, Model: Model{FlowShift: 3, TrainSteps: 1000}, CFGMin: lo, CFGMax: hi}
		x, err := p.Generate(context.Background(), cond, neg, []float32{0.5}, 20, 4.5, nil)
		if err != nil {
			t.Fatal(err)
		}
		return e, x
	}
	// 20 steps of shift 3 (TestDPM's sigmas): 14 at sigma 0.6 or more
	all, xa := run(0, 0)
	if all.calls != 20 || all.passes != 40 {
		t.Errorf("every step: %d calls, %d passes, want 20 and 40", all.calls, all.passes)
	}
	some, xs := run(0.6, 1)
	if some.calls != 20 || some.passes != 34 {
		t.Errorf("sigma 0.6 to 1: %d calls, %d passes, want 20 and 34", some.calls, some.passes)
	}
	if xa[0] == xs[0] {
		t.Errorf("the guidance left out changed nothing (%v)", xa[0])
	}
	if e, x := run(0, 1); e.passes != 40 || x[0] != xa[0] {
		t.Errorf("sigma 0 to 1: %d passes, x %v; want 40 and %v (every step)", e.passes, x[0], xa[0])
	}
}

// TestFormats checks the DiT's default format by model: int8 for Sana 600M, lut8 for SANA 1.5
// 1.6B (its int8 DiT breaks the pictures).
func TestFormats(t *testing.T) {
	for name, want := range map[string]string{
		"Sana-600M":                     "sana-600m-1024x1024-int8-int8-g4-v5.coreml",
		"SANA1.5-1.6B":                  "sana1.5-1.6b-1024x1024-int8-lut8-g4-v5.coreml",
		"Sana_600M_1024px_diffusers":    "sana_600m_1024px_diffusers-1024x1024-int8-int8-g4-v5.coreml",
		"SANA1.5_1.6B_1024px_diffusers": "sana1.5_1.6b_1024px_diffusers-1024x1024-int8-lut8-g4-v5.coreml",
	} {
		if got := ExportName(name, ConvertOptions{}); got != want {
			t.Errorf("%s: %s, want %s", name, got, want)
		}
	}
	if got := ExportName("Sana-600M", ConvertOptions{DitFormat: "fp16"}); got != "sana-600m-1024x1024-int8-fp16-g4-v5.coreml" {
		t.Errorf("Sana-600M at fp16: %s", got)
	}
}

// TestCoreAISizes checks the sizes of Core AI's exports: 512 x 512 and 1024 x 1024 in one export
// (by either size), any other size alone; and that a manifest of several sizes gives each size's
// parts.
func TestCoreAISizes(t *testing.T) {
	for _, c := range []struct {
		size int
		want string
	}{
		{1024, "sana-600m-512+1024-int8-int8-g4-v5.coreai"},
		{512, "sana-600m-512+1024-int8-int8-g4-v5.coreai"},
		{768, "sana-600m-768x768-int8-int8-g4-v5.coreai"},
	} {
		if got := ExportName("Sana-600M", ConvertOptions{Backend: "coreai", Width: c.size}); got != c.want {
			t.Errorf("%d: %s, want %s", c.size, got, c.want)
		}
	}
	m := &ProgramManifest{Dit: DitConfig{Width: 1024, Height: 1024}, Blocks: []string{"b.aimodel#s1024x1024"},
		Sizes: []SizeParts{
			{Width: 512, Height: 512, Blocks: []string{"b.aimodel#s512x512"}, Final: "f.aimodel#s512x512"},
			{Width: 1024, Height: 1024, Blocks: []string{"b.aimodel#s1024x1024"}, Final: "f.aimodel#s1024x1024"},
		}}
	if err := m.At(512, 512); err != nil || m.Blocks[0] != "b.aimodel#s512x512" || m.Final != "f.aimodel#s512x512" || m.Dit.W() != 512 {
		t.Errorf("at 512: %v, %v %s %d", err, m.Blocks, m.Final, m.Dit.W())
	}
	if err := m.At(768, 768); err == nil {
		t.Error("at 768: no error")
	}
}
