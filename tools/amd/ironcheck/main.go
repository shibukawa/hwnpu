// Command ironcheck runs a graph exported by tools/amd/iron_export.py on an AMD NPU through
// hwnpu (no IRON, no XRT) and compares its outputs with graph.reference()'s.
//
//	go run ./tools/amd/ironcheck DIR
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/shibukawa/hwnpu"
)

type handle struct {
	Name     string `json:"name"`
	Shape    []int  `json:"shape"`
	DType    string `json:"dtype"`
	Argument string `json:"argument"`
	Offset   int    `json:"offset"`
	Bytes    int    `json:"bytes"`
}

type manifest struct {
	Version   int      `json:"version"`
	ELF       string   `json:"elf"`
	Kernels   []string `json:"kernels"`
	Arguments []struct {
		Index int    `json:"index"`
		Kind  string `json:"kind"`
		Bytes int    `json:"bytes"`
	} `json:"arguments"`
	Inputs     []handle `json:"inputs"`
	Parameters []struct {
		Name  string `json:"name"`
		Index int    `json:"index"`
		DType string `json:"dtype"`
		Kind  string `json:"kind"`
		Form  form   `json:"form"`
	} `json:"parameters"`
	Outputs []handle          `json:"outputs"`
	Plan    string            `json:"plan"`
	Images  map[string]string `json:"images"`
	Carried *struct {
		Carry *struct {
			Argument string   `json:"argument"`
			Offset   int      `json:"offset"`
			Names    []string `json:"names"`
		} `json:"carry"`
		Next map[string]struct {
			Form  *form `json:"form"`
			Plane int   `json:"plane"`
			Index int   `json:"index"`
		} `json:"next"`
	} `json:"carried"`
}

// form is how a scratchpad word follows from one per-call value v of the graph (IRON's
// Affine): clamp(((v*scale + bias) >> down)*mul + add, lo, hi) % mod + post, >> flooring.
type form struct {
	Value string `json:"value"`
	Scale int64  `json:"scale"`
	Bias  int64  `json:"bias"`
	Down  uint   `json:"down"`
	Mul   int64  `json:"mul"`
	Add   int64  `json:"add"`
	Lo    *int64 `json:"lo"`
	Hi    *int64 `json:"hi"`
	Mod   *int64 `json:"mod"`
	Post  int64  `json:"post"`
}

func (f form) eval(v int64) int64 {
	x := ((v*f.Scale+f.Bias)>>f.Down)*f.Mul + f.Add // >> on int64 floors
	if f.Lo != nil && x < *f.Lo {
		x = *f.Lo
	}
	if f.Hi != nil && x > *f.Hi {
		x = *f.Hi
	}
	if f.Mod != nil {
		x &= *f.Mod - 1 // a power of two: floors like Python's %
	}
	return x + f.Post
}

// writeWords writes the scratchpad words of a call with the graph's per-call values: a
// value a core reads is held shifted left by 2, an address as it is (IRON's Parameter.shift).
func writeWords(m *manifest, run *hwnpu.Run, values map[string]int64, verbose bool) {
	if len(m.Parameters) == 0 {
		return
	}
	sp := run.Scratchpad()
	for _, p := range m.Parameters {
		gv, ok := values[p.Form.Value]
		if !ok {
			log.Fatalf("parameter %s needs the per-call value %s (have %v)", p.Name, p.Form.Value, values)
		}
		v := p.Form.eval(gv)
		if verbose {
			fmt.Printf("parameter %s (slot %d, %s) = %d from %s = %d\n", p.Name, p.Index, p.Kind, v, p.Form.Value, gv)
		}
		if p.Kind == "core" {
			v <<= 2
		}
		if 4*p.Index+4 > len(sp) {
			log.Fatalf("parameter %s at word %d: the scratchpad has %d bytes", p.Name, p.Index, len(sp))
		}
		binary.LittleEndian.PutUint32(sp[4*p.Index:], uint32(v))
	}
}

// runCarried runs check/calls.json, a host-driven sequence of a graph with carried values:
// each call's values go to the scratchpad and the carry buffer's first plane, and its next
// values come from Affine forms or the carry buffer's second plane.
func runCarried(ctx context.Context, dir string, m *manifest, run *hwnpu.Run, args map[string]*hwnpu.Buffer) {
	var calls []struct {
		Values map[string]int64 `json:"values"`
		Next   map[string]int64 `json:"next"`
	}
	b, err := os.ReadFile(filepath.Join(dir, "check", "calls.json"))
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(b, &calls); err != nil {
		log.Fatal(err)
	}
	c := m.Carried
	le := binary.LittleEndian
	bad := 0
	t0 := time.Now()
	for i, call := range calls {
		writeWords(m, run, call.Values, i == 0)
		var carry *hwnpu.Buffer
		if c.Carry != nil {
			carry = args[c.Carry.Argument]
			for j, name := range c.Carry.Names {
				le.PutUint32(carry.Bytes()[c.Carry.Offset+4*j:], uint32(call.Values[name]))
			}
			carry.SyncToDevice(c.Carry.Offset, 8*len(c.Carry.Names))
		}
		if err := run.Run(ctx); err != nil {
			log.Fatalf("call %d: %v", i, err)
		}
		if carry != nil {
			carry.SyncFromDevice(c.Carry.Offset, 8*len(c.Carry.Names))
		}
		got := map[string]int64{}
		for name, n := range c.Next {
			if n.Form != nil {
				got[name] = n.Form.eval(call.Values[n.Form.Value])
			} else {
				o := c.Carry.Offset + 4*(n.Plane*len(c.Carry.Names)+n.Index)
				got[name] = int64(int32(le.Uint32(carry.Bytes()[o:])))
			}
			if got[name] != call.Next[name] {
				bad++
			}
		}
		if i < 4 || bad > 0 && bad < 4 {
			fmt.Printf("call %d: values %v -> next %v (reference %v)\n", i, call.Values, got, call.Next)
		}
	}
	fmt.Printf("%d host-driven calls in %.1f ms, %d next values differ from the reference\n", len(calls), time.Since(t0).Seconds()*1000, bad)
	if bad > 0 {
		fmt.Println("FAIL")
		os.Exit(1)
	}
	fmt.Println("PASS")
}

func bf16(h uint16) float64 { return float64(math.Float32frombits(uint32(h) << 16)) }

func main() {
	runs := flag.Int("runs", 100, "timed runs")
	tol := flag.Float64("tol", 0.02, "tolerance relative to the output's largest magnitude")
	flag.Parse()
	dir := flag.Arg(0)
	var m manifest
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(b, &m); err != nil {
		log.Fatal(err)
	}
	if len(m.Kernels) != 1 {
		log.Fatalf("%d commands: only one is supported yet", len(m.Kernels))
	}
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	kr, err := dev.Kernel()
	if err != nil {
		log.Fatal(err)
	}
	t0 := time.Now()
	prog, err := kr.Load(ctx, hwnpu.ProgramSource{Path: filepath.Join(dir, m.ELF)})
	if err != nil {
		log.Fatal(err)
	}
	defer prog.Close()
	loadMS := time.Since(t0).Seconds() * 1000
	k, err := prog.Kernel(m.Kernels[0])
	if err != nil {
		log.Fatal(err)
	}
	args := map[string]*hwnpu.Buffer{}
	run, err := k.NewRun()
	if err != nil {
		log.Fatal(err)
	}
	defer run.Close()
	for _, a := range m.Arguments {
		buf, err := dev.AllocBuffer(max(a.Bytes, 4096))
		if err != nil {
			log.Fatal(err)
		}
		defer buf.Close()
		clear(buf.Bytes())
		args[a.Kind] = buf
		run.SetBuffer(a.Index, buf)
	}
	for kind, name := range m.Images {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			log.Fatal(err)
		}
		copy(args[kind].Bytes(), data)
	}
	if m.Carried != nil {
		for _, buf := range args {
			buf.SyncToDevice(0, buf.Size())
		}
		runCarried(ctx, dir, &m, run, args)
		return
	}
	for _, in := range m.Inputs {
		data, err := os.ReadFile(filepath.Join(dir, "check", "in_"+in.Name+".bin"))
		if err != nil {
			log.Fatal(err)
		}
		if len(data) != in.Bytes {
			log.Fatalf("input %s: %d bytes, the manifest says %d", in.Name, len(data), in.Bytes)
		}
		copy(args[in.Argument].Bytes()[in.Offset:], data)
	}
	for _, buf := range args {
		buf.SyncToDevice(0, buf.Size())
	}
	// Per-call values go to the ctrl scratchpad, a word per parameter: a value a core reads
	// is held shifted left by 2, an address as it is (IRON's Parameter.shift).
	values := map[string]int64{}
	if b, err := os.ReadFile(filepath.Join(dir, "check", "values.json")); err == nil {
		if err := json.Unmarshal(b, &values); err != nil {
			log.Fatal(err)
		}
	}
	writeWords(&m, run, values, true)
	t0 = time.Now()
	if err := run.Run(ctx); err != nil {
		log.Fatal(err)
	}
	firstMS := time.Since(t0).Seconds() * 1000
	ok := true
	for _, out := range m.Outputs {
		buf := args[out.Argument]
		buf.SyncFromDevice(0, buf.Size())
		want, err := os.ReadFile(filepath.Join(dir, "check", "out_"+out.Name+".bin"))
		if err != nil {
			log.Fatal(err)
		}
		if out.DType != "bfloat16" {
			log.Fatalf("output %s: dtype %s is not compared yet", out.Name, out.DType)
		}
		// A bounded call writes the first part only; the reference holds just that part.
		got := buf.Bytes()[out.Offset : out.Offset+min(out.Bytes, len(want))]
		scale, worst, at := 0.0, 0.0, 0
		for i := 0; i+1 < len(want); i += 2 {
			scale = max(scale, math.Abs(bf16(binary.LittleEndian.Uint16(want[i:]))))
		}
		for i := 0; i+1 < len(want); i += 2 {
			d := math.Abs(bf16(binary.LittleEndian.Uint16(got[i:])) - bf16(binary.LittleEndian.Uint16(want[i:])))
			if d > worst {
				worst, at = d, i/2
			}
		}
		if len(want) < out.Bytes {
			// The call was bounded. The device works in whole tiles, so it may write past the
			// bound up to the tile; the reference holds the first part only.
			written := 0
			for _, c := range buf.Bytes()[out.Offset+len(want) : out.Offset+out.Bytes] {
				if c != 0 {
					written++
				}
			}
			fmt.Printf("output %s: bounded to %d of %d values, %d bytes written past the bound (to the tile)\n", out.Name, len(want)/2, out.Bytes/2, written)
		}
		rel := worst / max(scale, 1e-30)
		fmt.Printf("output %s: %d values, worst difference %.4g at %d (%.2e of the largest |value| %.4g)\n", out.Name, len(want)/2, worst, at, rel, scale)
		if rel > *tol {
			ok = false
		}
	}
	times := make([]float64, *runs)
	for i := range times {
		t := time.Now()
		if err := run.Run(ctx); err != nil {
			log.Fatal(err)
		}
		times[i] = time.Since(t).Seconds() * 1000
	}
	slices.Sort(times)
	fmt.Printf("%s: load %.1f ms, first run %.3f ms, median %.3f ms, min %.3f ms\n", m.Kernels[0], loadMS, firstMS, times[len(times)/2], times[0])
	if !ok {
		fmt.Println("FAIL")
		os.Exit(1)
	}
	fmt.Println("PASS")
}
