// Command fbrepro writes the small Core AI programs that reproduce the problems reported to Apple
// (docs/apple-feedback): each one operation or one short chain, fp16, constant weights, no
// checkpoint.
//
//	go run ./tools/coreai/fbrepro sin OUT.aimodel          sin alone (M1: off the Neural Engine)
//	go run ./tools/coreai/fbrepro clipew OUT.aimodel       conv -> reshape -> per-channel mul -> clip
//	go run ./tools/coreai/fbrepro prescale OUT.aimodel     a*c1 - b*c2
//	go run ./tools/coreai/fbrepro snake [-clip] [-square] OUT.aimodel
//	                                                       a Snake block as the DAC's (conv, sin, conv,
//	                                                       reshape, mul), then a clip with -clip; -square
//	                                                       writes sin(y) as y*y
//	go run ./tools/coreai/fbrepro attn [-rows 1536] [-keys 1536] [-chunk 1024] [-head 128] OUT.aimodel
//	                                                       one head's attention, the queries in chunks
//	                                                       (1536 rows by 1024: a chunk of 512 is left)
//
// Flags go before OUT. -c and -t set channels and frames of the elementwise programs.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/shibukawa/bonsaiai/internal/imagegen/milpart"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/mlpackage"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	cmd := os.Args[1]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	c := fs.Int("c", 256, "channels")
	t := fs.Int("t", 4096, "frames")
	clip := fs.Bool("clip", false, "snake: end with a clip")
	square := fs.Bool("square", false, "snake: y*y in place of sin(y)")
	rows := fs.Int("rows", 1536, "attn: query rows")
	keys := fs.Int("keys", 1536, "attn: keys")
	chunk := fs.Int("chunk", 1024, "attn: query rows a chunk")
	head := fs.Int("head", 128, "attn: head size")
	fs.Parse(os.Args[2:])
	if fs.NArg() != 1 {
		usage()
	}
	out := fs.Arg(0)
	r := rand.New(rand.NewSource(1))
	vals := func(n int, a float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(a * r.NormFloat64())
		}
		return v
	}
	var body func(g *milpart.Graph)
	switch cmd {
	case "sin":
		body = func(g *milpart.Graph) {
			x := g.Input("x", 1, *c, 1, *t)
			g.Output(g.Unary("sin", x))
		}
	case "clipew":
		body = func(g *milpart.Graph) {
			x := g.Input("x", 1, *c, *t/8, 8)
			y := g.Conv2D(x, "conv", vals(*c**c, 1/math.Sqrt(float64(*c))), vals(*c, 0.1), *c, 1, 1)
			y = g.Reshape(y, 1, *c, 1, *t)
			alpha, _ := g.B.F16s("alpha", []int{1, *c, 1, 1}, vals(*c, 1))
			y = g.Mul(y, alpha)
			g.Output(clipOp(g, y, -1500, 1500))
		}
	case "prescale":
		body = func(g *milpart.Graph) {
			a := g.Input("a", 1, *c, 1, *t)
			b := g.Input("b", 1, *c, 1, *t)
			g.Output(g.Sub(g.Scale(a, 0.75), g.Scale(b, 1.25)))
		}
	case "snake":
		body = func(g *milpart.Graph) {
			x := g.Input("x", 1, *c, *t/8, 8)
			y := g.Conv2D(x, "conv0", vals(*c**c, 1/math.Sqrt(float64(*c))), vals(*c, 0.1), *c, 1, 1)
			alpha, _ := g.B.F16s("alpha", []int{1, *c, 1, 1}, absVals(vals(*c, 1), 0.1))
			inv, _ := g.B.F16s("inv", []int{1, *c, 1, 1}, recip(alpha32(g, *c)))
			ay := g.Mul(y, alpha)
			var s *mlpackage.Var
			if *square {
				s = g.Mul(ay, ay)
			} else {
				s = g.Unary("sin", ay)
			}
			y = g.Add(y, g.Mul(g.Mul(s, s), inv)) // Snake: y + sin(a y)^2 / a
			y = g.Conv2D(y, "conv1", vals(*c**c, 1/math.Sqrt(float64(*c))), vals(*c, 0.1), *c, 1, 1)
			y = g.Reshape(y, 1, *c, 1, *t)
			y = g.Mul(y, alpha)
			if *clip {
				y = clipOp(g, y, -1500, 1500)
			}
			g.Output(y)
		}
	case "attn":
		body = func(g *milpart.Graph) {
			q := g.Input("q", 1, 1, *rows, *head)
			k := g.Input("k", 1, 1, *keys, *head)
			v := g.Input("v", 1, 1, *keys, *head)
			g.Output(g.HeadAttention(q, k, v, milpart.HeadOptions{Chunk: *chunk, Whole: true}))
		}
	default:
		usage()
	}
	part := &milpart.Part{Name: cmd, Model: "fbrepro", Body: body}
	_, err := llm.WriteAI(out, map[string]string{"fbrepro": cmd}, func(b *mlpackage.Builder) (ins, outs []mlpackage.Feature, err error) {
		ins, o, err := part.Build(b, nil, llm.Options{})
		return ins, []mlpackage.Feature{o}, err
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(out)
}

// alphas are the snake's alpha values as the program has them, for their reciprocals.
var alphas []float32

func absVals(v []float32, min float32) []float32 {
	for i := range v {
		v[i] = float32(math.Abs(float64(v[i]))) + min
	}
	alphas = v
	return v
}

func alpha32(_ *milpart.Graph, _ int) []float32 { return alphas }

func recip(v []float32) []float32 {
	o := make([]float32, len(v))
	for i, x := range v {
		o[i] = 1 / x
	}
	return o
}

func clipOp(g *milpart.Graph, x *mlpackage.Var, lo, hi float32) *mlpackage.Var {
	return g.Op("clip", map[string]any{"x": x, "alpha": g.B.ScalarF16(lo), "beta": g.B.ScalarF16(hi)}, x.Shape...)
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: fbrepro sin|clipew|prescale|snake|attn [flags] OUT.aimodel")
	os.Exit(2)
}
