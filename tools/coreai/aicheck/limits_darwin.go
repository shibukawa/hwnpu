package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/types"
)

// limits measures the Neural Engine's limits that arXiv 2606.22283 and ANEForge (2606.17090)
// give, on this Mac through Core AI (docs/design/apple-neural-engine.md):
//
//   - floor: what one call of a nearly empty program costs (ANEForge: about 70 us; the paper:
//     about 190 us on an M1)
//
//   - programs: N small programs loaded at once and called in turn, N up to -max (the paper:
//     about 128 programs a process), with the wired memory
//
//   - operand: an elementwise program over 0.25 .. 16 MB (the paper: a working set past about
//     2 MB leaves the chip)
//
//   - conv: a 1 x 1 convolution of 1024 channels over 32 x 32 to 64 .. 1024 channels (the
//     handoff: a 1x1 conv's cost steps up past 192 to 256 output channels)
//
//   - formats: a linear of a constant 4096 x 4096 (32 MB in fp16) in fp16, int8, int4, lut8,
//     lut6 and lut4 (rows whole, lut tables of 16 rows), on 1 and 64 rows (arXiv 2611.19101: the
//     weights stream at bytes for fp16, at the dequantizer's elements for int8, faster for lut4)
//
//     aicheck limits [-only floor,programs,operand,conv,formats] [-max 200] [-calls 200]
func limits(args []string) error {
	fs := flag.NewFlagSet("limits", flag.ExitOnError)
	only := fs.String("only", "", "the measurements to run, comma separated (default all)")
	maxProgs := fs.Int("max", 200, "programs: the most loaded at once")
	calls := fs.Int("calls", 200, "calls timed a measurement")
	fs.Parse(args)
	ctx := context.Background()
	if err := llm.LockNPU(ctx); err != nil {
		return err
	}
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "aicheck-limits-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	m := &limitRunner{ctx: ctx, dev: dev, dir: dir, calls: *calls}
	run := func(name string, f func() error) {
		if *only != "" && !strings.Contains(","+*only+",", ","+name+",") {
			return
		}
		fmt.Printf("== %s\n", name)
		if err := f(); err != nil {
			fmt.Printf("  FAIL %v\n", err)
		}
	}
	run("floor", m.floor)
	run("programs", func() error { return m.programs(*maxProgs) })
	run("operand", m.operand)
	run("conv", m.conv)
	run("formats", m.formats)
	return nil
}

type limitRunner struct {
	ctx   context.Context
	dev   backend.Device
	dir   string
	n     int
	calls int
}

// loaded is a program loaded with its buffers bound to random inputs.
type loaded struct {
	g     backend.Graph
	req   backend.Request
	where string
}

func (m *limitRunner) load(body func(g *milpart.Graph)) (*loaded, error) {
	return m.loadAs(body, llm.Options{Backend: llm.BackendCoreML, Format: "fp16"})
}

// loadAs is load with the weights (blocks.Graph.Matrix) in the options' format.
func (m *limitRunner) loadAs(body func(g *milpart.Graph), opts llm.Options) (*loaded, error) {
	m.n++
	name := fmt.Sprintf("l%d", m.n)
	path := filepath.Join(m.dir, name+".mlpackage")
	if _, err := llm.WritePart(path, &milpart.Part{Name: name, Model: "limits", Body: body}, nil, opts, nil); err != nil {
		return nil, err
	}
	g, err := m.dev.Compile(m.ctx, types.GraphSource{Path: path})
	if err != nil {
		return nil, err
	}
	where := "ane"
	if p := g.Placement(); !p.OnNPU() {
		where = fmt.Sprintf("NOT-ANE(%v)", p.Ops)
	}
	req, err := g.NewRequest()
	if err != nil {
		g.Close()
		return nil, err
	}
	rng := rand.New(rand.NewSource(int64(m.n)))
	for _, a := range g.Args() {
		b, err := m.dev.AllocBuffer(a.Bytes, types.BufferKind(0))
		if err != nil {
			g.Close()
			return nil, err
		}
		if a.Kind == types.ArgInput {
			f := make([]float32, a.Bytes/2)
			for i := range f {
				f[i] = float32(rng.NormFloat64())
			}
			hwnpu.EncodeF16(b.Bytes(), f)
		}
		req.Bind(a.Index, b)
	}
	return &loaded{g: g, req: req, where: where}, nil
}

func (p *loaded) call(ctx context.Context) error {
	f, err := p.req.Submit()
	if err == nil {
		err = f.Wait(ctx)
	}
	return err
}

// time is the median of 5 means of n/5 calls each, after 3 calls.
func (m *limitRunner) time(p *loaded) (time.Duration, error) {
	for range 3 {
		if err := p.call(m.ctx); err != nil {
			return 0, err
		}
	}
	per := max(m.calls/5, 1)
	var means []time.Duration
	for range 5 {
		t := time.Now()
		for range per {
			if err := p.call(m.ctx); err != nil {
				return 0, err
			}
		}
		means = append(means, time.Since(t)/time.Duration(per))
	}
	sort.Slice(means, func(i, j int) bool { return means[i] < means[j] })
	return means[2], nil
}

func (m *limitRunner) floor() error {
	for _, n := range []int{64, 1024, 16 * 1024, 256 * 1024} {
		p, err := m.load(func(g *milpart.Graph) {
			g.Output(g.Add(g.Input("x", 1, 64, n/64), g.Input("z", 1, 64, n/64)))
		})
		if err != nil {
			return err
		}
		d, err := m.time(p)
		p.g.Close()
		if err != nil {
			return err
		}
		fmt.Printf("  add of %7d fp16 (%7.1f KB a buffer): %-6s %8.1f us a call\n", n, float64(2*n)/1024, p.where, float64(d.Nanoseconds())/1e3)
	}
	return nil
}

// wiredGB is the wired memory vm_stat gives.
func wiredGB() float64 {
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		return -1
	}
	page := 16384.0
	for _, line := range strings.Split(string(out), "\n") {
		if strings.Contains(line, "page size of") {
			f := strings.Fields(line)
			for i, w := range f {
				if w == "of" && i+1 < len(f) {
					v, _ := strconv.ParseFloat(f[i+1], 64)
					page = v
				}
			}
		}
		if strings.HasPrefix(line, "Pages wired down:") {
			v, _ := strconv.ParseFloat(strings.TrimSuffix(strings.TrimSpace(strings.TrimPrefix(line, "Pages wired down:")), "."), 64)
			return v * page / (1 << 30)
		}
	}
	return -1
}

// programs loads small programs one by one (each a 256 x 256 linear of its own weights on 64
// rows, distinct so that none is shared) and at 1, 8, 16, ... loaded calls them all in turn.
func (m *limitRunner) programs(most int) error {
	var ps []*loaded
	defer func() {
		for _, p := range ps {
			p.g.Close()
		}
	}()
	base := wiredGB()
	fmt.Printf("  wired before: %.2f GB\n", base)
	marks := map[int]bool{1: true, 8: true, 16: true, 32: true, 64: true, 96: true, 112: true, 120: true, 124: true, 126: true, 127: true, 128: true, 129: true, 130: true, 136: true, 144: true, 160: true, 192: true, 200: true, 256: true}
	for i := 1; i <= most; i++ {
		seed := int64(i)
		t := time.Now()
		p, err := m.load(func(g *milpart.Graph) {
			rng := rand.New(rand.NewSource(seed))
			w := make([]float32, 256*256)
			for j := range w {
				w[j] = float32(rng.NormFloat64() / 16)
			}
			g.Output(g.LinearConst(g.Input("x", 1, 64, 256), "w", w, 256, nil))
		})
		load := time.Since(t)
		if err != nil {
			fmt.Printf("  program %d: FAIL load after %v: %v\n", i, load, err)
			return nil
		}
		ps = append(ps, p)
		if err := p.call(m.ctx); err != nil {
			fmt.Printf("  program %d: FAIL first call: %v\n", i, err)
			return nil
		}
		if !marks[i] {
			continue
		}
		// all loaded, called in turn: 3 rounds, the last timed
		var round time.Duration
		for r := 0; r < 3; r++ {
			t := time.Now()
			for k, q := range ps {
				if err := q.call(m.ctx); err != nil {
					fmt.Printf("  %d loaded: FAIL calling program %d: %v\n", i, k+1, err)
					return nil
				}
			}
			round = time.Since(t)
		}
		notANE := 0
		for _, q := range ps {
			if q.where != "ane" {
				notANE++
			}
		}
		fmt.Printf("  %3d loaded (last load %6.0f ms): a round %7.2f ms = %6.1f us a call; wired +%.2f GB; off the ANE %d\n",
			i, float64(load.Microseconds())/1e3, float64(round.Microseconds())/1e3, float64(round.Microseconds())/float64(len(ps)), wiredGB()-base, notANE)
	}
	return nil
}

// operand: y = x * s + z over n fp16 of x and z, n from 128 K (0.25 MB) to 8 M (16 MB): the
// time a MB of the three tensors.
func (m *limitRunner) operand() error {
	for _, mb := range []float64{0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 12, 16} {
		n := int(mb * (1 << 20) / 2)
		n = n / 512 * 512
		p, err := m.load(func(g *milpart.Graph) {
			x := g.Input("x", 1, 512, n/512)
			z := g.Input("z", 1, 512, n/512)
			g.Output(g.Add(g.Scale(x, 1.5), z))
		})
		if err != nil {
			return err
		}
		d, err := m.time(p)
		p.g.Close()
		if err != nil {
			return err
		}
		moved := 3 * float64(2*n) / (1 << 20)
		fmt.Printf("  tensor %5.2f MB: %-6s %8.1f us, %6.2f GB/s of the three tensors\n", mb, p.where, float64(d.Nanoseconds())/1e3, moved/1024/d.Seconds())
	}
	return nil
}

// conv: a 1 x 1 convolution of (1, 1024, 32, 32) to out channels, out 64 .. 1024 (2 GFLOP at
// 1024): the time an output channel.
func (m *limitRunner) conv() error {
	const cin, hw = 1024, 32
	for _, out := range []int{64, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 768, 1024} {
		rng := rand.New(rand.NewSource(int64(out)))
		w := make([]float32, out*cin)
		for j := range w {
			w[j] = float32(rng.NormFloat64() / 32)
		}
		p, err := m.load(func(g *milpart.Graph) {
			g.Output(g.Conv2D(g.Input("x", 1, cin, hw, hw), "w", w, nil, out, 1, 1))
		})
		if err != nil {
			return err
		}
		d, err := m.time(p)
		p.g.Close()
		if err != nil {
			return err
		}
		tf := 2 * float64(cin*hw*hw*out) / d.Seconds() / 1e12
		fmt.Printf("  out %4d: %-6s %8.1f us, %5.2f us an output channel, %.2f TFLOP/s\n", out, p.where, float64(d.Nanoseconds())/1e3, float64(d.Nanoseconds())/1e3/float64(out), tf)
	}
	return nil
}

func (m *limitRunner) formats() error {
	const out, in = 4096, 4096
	rng := rand.New(rand.NewSource(7))
	w := make([]float32, out*in)
	for j := range w {
		w[j] = float32(rng.NormFloat64() / 64)
	}
	for _, rows := range []int{1, 64} {
		for _, f := range []string{"fp16", "int8", "int4", "lut8", "lut6", "lut4"} {
			opts := llm.Options{Backend: llm.BackendCoreML, Format: f}
			if strings.HasPrefix(f, "lut") {
				opts.Group = 16
			}
			p, err := m.loadAs(func(g *milpart.Graph) {
				x := g.Input("x", 1, rows, in)
				g.Output(g.Op("linear", map[string]any{"x": x, "weight": g.Matrix("w", w, out, in)}, 1, rows, out))
			}, opts)
			if err != nil {
				fmt.Printf("  rows %2d %-5s FAIL %v\n", rows, f, err)
				continue
			}
			d, err := m.time(p)
			p.g.Close()
			if err != nil {
				return err
			}
			bits := map[string]float64{"fp16": 16, "int8": 8, "int4": 4, "lut8": 8, "lut6": 6, "lut4": 4}[f]
			bytes := float64(out*in) * bits / 8
			fmt.Printf("  rows %2d %-5s: %-6s %8.1f us, %6.1f GB/s of weights, %5.1f G weights/s\n", rows, f, p.where, float64(d.Nanoseconds())/1e3, bytes/d.Seconds()/1e9, float64(out*in)/d.Seconds()/1e9)
		}
	}
	return nil
}
