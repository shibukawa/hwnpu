package blocks

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// SiLU writes silu without the Neural Engine's own silu (or sigmoid), and in fp16 it stays within
// fp16's rounding of the exact silu, the tail past exp's overflow (x < -11) included, where it is
// 0 rather than NaN.
func TestSiLU(t *testing.T) {
	const n = 641
	x := make([]float32, n)
	for i := range x {
		x[i] = -20 + 32*float32(i)/(n-1) // -20 to 12
	}
	dir := filepath.Join(t.TempDir(), "m.mlpackage")
	pkg, err := mlpackage.Create(dir)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	g := &Graph{Name: "t", B: b}
	in := b.Input("x", mlpackage.Float16, 1, n)
	y := b.NamedOp("identity", "y", map[string]any{"x": g.SiLU(in)}, mlpackage.Float16, 1, n)
	prog := b.Finish(y)
	fn := prog.GetFunctions()["main"]
	for _, op := range fn.GetBlockSpecializations()[fn.GetOpset()].GetOperations() {
		if op.GetType() == "silu" || op.GetType() == "sigmoid" {
			t.Errorf("a %s operation: the Neural Engine's is off by about 1.4%%", op.GetType())
		}
	}
	feat := []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: []int{1, n}}}
	if err := pkg.Finish(prog, 9, feat, []mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: []int{1, n}}}); err != nil {
		t.Fatal(err)
	}
	p, c, err := milcpu.OpenPackage(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	out, err := p.Run(map[string]milcpu.Tensor{"x": {Shape: []int{1, n}, Data: x}}, milcpu.Options{RoundF16: true})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range x {
		want := float64(v) / (1 + math.Exp(-float64(v)))
		got := float64(out["y"].Data[i])
		if math.IsNaN(got) || math.Abs(got-want) > 4e-3*math.Abs(want)+2e-4 {
			t.Errorf("silu(%g) = %g, want %g", v, got, want)
		}
	}
}
