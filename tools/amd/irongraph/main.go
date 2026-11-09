// Command irongraph runs a graph exported by tools/amd/iron_graph.py on an AMD NPU through
// hwnpu and compares its outputs with graph.reference()'s (check/).
//
//	go run ./tools/amd/irongraph [-runs N] DIR
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
	"runtime/pprof"
	"syscall"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
)

func values(b []byte, dtype string) []float64 {
	switch dtype {
	case "bfloat16":
		out := make([]float64, len(b)/2)
		for i := range out {
			out[i] = float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16))
		}
		return out
	case "float32":
		out := make([]float64, len(b)/4)
		for i := range out {
			out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:])))
		}
		return out
	case "int32":
		out := make([]float64, len(b)/4)
		for i := range out {
			out[i] = float64(int32(binary.LittleEndian.Uint32(b[4*i:])))
		}
		return out
	}
	log.Fatalf("dtype %s", dtype)
	return nil
}

func main() {
	runs := flag.Int("runs", 3, "timed runs")
	save := flag.Bool("save", false, "write each output as check/got_<name>.bin")
	variant := flag.Int("variant", 0, "the bank variant to run (the check is variant 0's)")
	tol := flag.Float64("tol", 0.03, "tolerance of the largest error relative to the output's largest magnitude")
	keep := flag.Bool("keep", false, "keep each command's run (iron.GraphOptions.KeepRuns)")
	cpuprofile := flag.String("cpuprofile", "", "write the runs' CPU profile (go tool pprof) to this file")
	pin := flag.Float64("pin", 0, "GiB of device buffers to allocate, fill and keep before loading the graph (a test of large pinned memory)")
	flag.Parse()
	dir := flag.Arg(0)
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	for left := int(*pin * (1 << 30)); left > 0; {
		n := min(left, 7<<29)
		b, err := dev.AllocBuffer(n)
		if err != nil {
			log.Fatalf("pin: after %.1f GiB: %v", *pin-float64(left)/(1<<30), err)
		}
		defer b.Close()
		for i := range b.Bytes() {
			b.Bytes()[i] = byte(i)
		}
		if err := b.SyncToDevice(0, b.Size()); err != nil {
			log.Fatal(err)
		}
		left -= n
	}
	if *pin > 0 {
		fmt.Printf("pinned %.1f GiB\n", *pin)
	}
	t0 := time.Now()
	g, err := iron.OpenGraphWith(ctx, dev, dir, iron.GraphOptions{KeepRuns: *keep})
	if err != nil {
		log.Fatal(err)
	}
	defer g.Close()
	fmt.Printf("%s: %d commands, %d variants, loaded in %.1f s\n", g.Manifest.Name, len(g.Manifest.Graph.Kernels), g.Variants(), time.Since(t0).Seconds())
	if err := g.Use(*variant); err != nil {
		log.Fatal(err)
	}
	var vals map[string]int64
	if b, err := os.ReadFile(filepath.Join(dir, "check", "values.json")); err == nil {
		if err := json.Unmarshal(b, &vals); err != nil {
			log.Fatal(err)
		}
	}
	for _, h := range g.Inputs() {
		b, err := os.ReadFile(filepath.Join(dir, "check", "in_"+h.Name+".bin"))
		if err != nil {
			log.Fatal(err)
		}
		if err := g.SetInput(h.Name, b); err != nil {
			log.Fatal(err)
		}
	}
	per := make([]time.Duration, len(g.Manifest.Graph.Kernels))
	g.Timing = func(k int, d time.Duration) { per[k] += d }
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal(err)
		}
	}
	cpu0, wall0 := cpuTime(), time.Now()
	for i := 0; i < max(1, *runs); i++ {
		clear(per)
		t := time.Now()
		if err := g.Run(ctx, vals); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("run %d: %.1f ms (commands %v)\n", i, time.Since(t).Seconds()*1000, per)
	}
	if *cpuprofile != "" {
		pprof.StopCPUProfile()
	}
	n := float64(max(1, *runs))
	fmt.Printf("mean %.2f ms a run, the host's CPU %.3f ms a run (user and system)\n",
		time.Since(wall0).Seconds()*1000/n, (cpuTime()-cpu0).Seconds()*1000/n)
	bad := false
	for _, h := range g.Outputs() {
		got, err := g.Output(h.Name)
		if err != nil {
			log.Fatal(err)
		}
		want, err := os.ReadFile(filepath.Join(dir, "check", "out_"+h.Name+".bin"))
		if err != nil {
			log.Fatal(err)
		}
		if *save {
			if err := os.WriteFile(filepath.Join(dir, "check", "got_"+h.Name+".bin"), got, 0o644); err != nil {
				log.Fatal(err)
			}
		}
		gv, wv := values(got, h.DType), values(want, h.DType)
		n := min(len(gv), len(wv))
		var maxErr, maxRef, sumErr, dot, gg, ww float64
		for i := 0; i < n; i++ {
			e := math.Abs(gv[i] - wv[i])
			maxErr = max(maxErr, e)
			maxRef = max(maxRef, math.Abs(wv[i]))
			sumErr += e
			dot += gv[i] * wv[i]
			gg += gv[i] * gv[i]
			ww += wv[i] * wv[i]
		}
		cos := dot / math.Sqrt(gg*ww)
		fmt.Printf("output %s %v: max error %.4g of %.4g, mean %.4g, cosine %.6f\n", h.Name, h.Shape, maxErr, maxRef, sumErr/float64(n), cos)
		if !(maxErr <= *tol*maxRef) {
			bad = true
		}
	}
	if bad {
		fmt.Println("FAIL")
		os.Exit(1)
	}
	fmt.Println("PASS")
}

// cpuTime is the process's user and system CPU time so far.
func cpuTime() time.Duration {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}
