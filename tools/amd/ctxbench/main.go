// Command ctxbench measures what switching between designs costs on an AMD NPU: it loads
// several IRON GEMV designs at once (one hardware context each) and compares running each
// alone with running them in turn, the way a model runs one design per operator.
//
//	go run ./tools/amd/ctxbench DIR1:M1:K1 DIR2:M2:K2 ...
//
// Each DIR holds final.xclbin and insts.bin.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
)

type design struct {
	name string
	run  *hwnpu.Run
}

func median(t []float64) float64 {
	slices.Sort(t)
	return t[len(t)/2]
}

func main() {
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	kr, _ := dev.Kernel()
	var ds []design
	for _, spec := range os.Args[1:] {
		f := strings.Split(spec, ":")
		dir := f[0]
		m, _ := strconv.Atoi(f[1])
		k, _ := strconv.Atoi(f[2])
		prog, err := kr.Load(ctx, hwnpu.ProgramSource{Path: filepath.Join(dir, "final.xclbin")})
		if err != nil {
			log.Fatalf("%s: %v", dir, err)
		}
		defer prog.Close()
		kern, _ := prog.Kernel("MLIR_AIE")
		code, _ := os.ReadFile(filepath.Join(dir, "insts.bin"))
		instr, err := dev.AllocBuffer(len(code), hwnpu.WithBufferKind(hwnpu.BufferInstructions))
		if err != nil {
			log.Fatal(err)
		}
		copy(instr.Bytes(), code)
		instr.SyncToDevice(0, len(code))
		a, _ := dev.AllocBuffer(m * k * 2)
		b, _ := dev.AllocBuffer(k * 2)
		c, _ := dev.AllocBuffer(m * 2)
		run, _ := kern.NewRun()
		run.SetUint64(0, 3).SetBuffer(1, instr).SetUint32(2, uint32(len(code)/4)).SetBuffer(3, a).SetBuffer(4, b).SetBuffer(5, c)
		if err := run.Run(ctx); err != nil {
			log.Fatalf("%s: %v", dir, err)
		}
		ds = append(ds, design{fmt.Sprintf("%dx%d", m, k), run})
	}
	const n = 200
	alone := 0.0
	for _, d := range ds {
		t := make([]float64, n)
		for i := range t {
			s := time.Now()
			if err := d.run.Run(ctx); err != nil {
				log.Fatal(err)
			}
			t[i] = time.Since(s).Seconds() * 1000
		}
		md := median(t)
		alone += md
		fmt.Printf("%s alone: median %.3f ms\n", d.name, md)
	}
	// In turn: each run follows a run of another design.
	t := make([]float64, n)
	for i := range t {
		s := time.Now()
		for _, d := range ds {
			if err := d.run.Run(ctx); err != nil {
				log.Fatal(err)
			}
		}
		t[i] = time.Since(s).Seconds() * 1000
	}
	md := median(t)
	fmt.Printf("in turn: median %.3f ms per round of %d designs, alone they sum to %.3f ms: %.3f ms per switch\n",
		md, len(ds), alone, (md-alone)/float64(len(ds)))
	// Submitted back to back without waiting in between (one fence each).
	t = t[:0]
	for i := 0; i < n; i++ {
		s := time.Now()
		var fs []hwnpu.Fence
		for _, d := range ds {
			f, err := d.run.Submit()
			if err != nil {
				log.Fatal(err)
			}
			fs = append(fs, f)
		}
		for _, f := range fs {
			if err := f.Wait(ctx); err != nil {
				log.Fatal(err)
			}
		}
		t = append(t, time.Since(s).Seconds()*1000)
	}
	fmt.Printf("submitted together: median %.3f ms per round\n", median(t))
}
