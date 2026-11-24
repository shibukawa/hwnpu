package milcpu

import (
	"math/rand/v2"
	"os"
	"testing"
)

// BenchmarkPackage runs the .mlpackage at BONSAIAI_TEST_MLPACKAGE on random inputs in [0, 1) (one
// thread unless -cpu says otherwise): a check of the executor's speed on a real program (YuNet's).
func BenchmarkPackage(b *testing.B) {
	path := os.Getenv("BONSAIAI_TEST_MLPACKAGE")
	if path == "" {
		b.Skip("BONSAIAI_TEST_MLPACKAGE: an .mlpackage")
	}
	prog, f, err := OpenPackage(path)
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()
	r := rand.New(rand.NewPCG(1, 2))
	in := map[string]Tensor{}
	for name, sh := range prog.Inputs() {
		d := make([]float32, numel(sh))
		for i := range d {
			d[i] = r.Float32()
		}
		in[name] = Tensor{Shape: sh, Data: d}
	}
	for b.Loop() {
		res, err := prog.Run(in, Options{})
		if err != nil {
			b.Fatal(err)
		}
		for _, t := range res {
			prog.pool.give(t.Data)
		}
	}
}
