// Command lutfake writes a Core AI program of one linear layer with a lut8 (or lut4/lut6)
// weight, for bisecting coreai-build's ANE compile of lookup-table weights:
//
//	go run ./tools/coreai/lutfake -rows 3072 -cols 3072 -tokens 1024 -group 16 -tables random OUT.aimodel
//
// -tables: random (each table its own random levels), same (all tables identical), first32
// (the tables of the first 32 rows identical, the rest random), pairs (each 32-row group's
// tables identical within the group, different between groups), sorted (uniform levels in
// [-a, a] with a different a per table: the layout palette.Uniform writes).
package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/coreai/ailower"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

func main() {
	rows := flag.Int("rows", 3072, "output channels")
	cols := flag.Int("cols", 3072, "input channels")
	tokens := flag.Int("tokens", 1024, "rows of the input")
	group := flag.Int("group", 16, "rows per table")
	bits := flag.Int("bits", 8, "4, 6 or 8")
	tables := flag.String("tables", "random", "random, same, first32, pairs, sorted")
	seed := flag.Int64("seed", 1, "random seed")
	flag.Parse()
	if flag.NArg() != 1 {
		log.Fatal("usage: lutfake [flags] OUT.aimodel")
	}
	r := rand.New(rand.NewSource(*seed))
	k := 1 << *bits
	nt := *rows / *group
	lut := make([]uint16, nt*k)
	level := func(t, e int, a float64) uint16 {
		return mlpackage.Float16Bits(float32(a * (2*float64(e)/float64(k-1) - 1)))
	}
	fill := func(t int, src *rand.Rand, sorted bool) {
		a := 0.01 + 0.05*src.Float64()
		for e := 0; e < k; e++ {
			if sorted {
				lut[t*k+e] = level(t, e, a)
			} else {
				lut[t*k+e] = mlpackage.Float16Bits(float32(a * src.NormFloat64()))
			}
		}
	}
	for t := 0; t < nt; t++ {
		switch *tables {
		case "random":
			fill(t, r, false)
		case "sorted":
			fill(t, r, true)
		case "same":
			fill(t, rand.New(rand.NewSource(*seed)), true)
		case "first32":
			if t*(*group) < 32 {
				fill(t, rand.New(rand.NewSource(*seed)), true)
			} else {
				fill(t, r, true)
			}
		case "pairs":
			g := t * (*group) / 32
			fill(t, rand.New(rand.NewSource(*seed+int64(g))), true)
		default:
			log.Fatalf("unknown -tables %q", *tables)
		}
	}
	n := *rows * *cols
	idx := make([]byte, (n**bits+7)/8)
	r.Read(idx)

	path := filepath.Join(os.TempDir(), fmt.Sprintf("lutfake-%d.bin", os.Getpid()))
	defer os.Remove(path)
	bw, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		log.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", bw)
	f16 := mlpackage.Float16
	x := b.Input("x", f16, 1, *tokens, *cols)
	w, err := b.LUT("w", *bits, []int{*rows, *cols}, []int{*group, *cols}, idx, lut)
	if err != nil {
		log.Fatal(err)
	}
	y := b.NamedOp("linear", "y", map[string]any{"x": x, "weight": w}, f16, 1, *tokens, *rows)
	prog := b.Finish(y)
	if err := bw.Close(); err != nil {
		log.Fatal(err)
	}
	weights, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	p, err := ailower.Lower(prog.GetFunctions()["main"], bytes.NewReader(weights))
	if err != nil {
		log.Fatal(err)
	}
	if err := p.WriteAsset(flag.Arg(0)); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s: %d x %d, group %d, %d tables (%s), %d tokens\n", flag.Arg(0), *rows, *cols, *group, nt, *tables, *tokens)
}
