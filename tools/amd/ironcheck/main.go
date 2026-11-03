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
	Inputs  []handle `json:"inputs"`
	Outputs []handle `json:"outputs"`
	Plan    string   `json:"plan"`
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
		got := buf.Bytes()[out.Offset : out.Offset+out.Bytes]
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
