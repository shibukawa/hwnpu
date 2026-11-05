// Command ironbench (tools/iron) runs an IRON design (opcode 3, instr, ninstr, bo0..bo4) through hwnpu's
// public API and times it. Buffers are bound in order: inputs from files, then outputs.
//
//	ironbench -xclbin final.xclbin -insts insts.bin -in a.bin,b.bin -out 1048576 -save c.bin -runs 50
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
)

type result struct {
	LoadMS, FirstMS, MinMS, MedianMS, MeanMS, MaxMS float64
	Runs                                            int
	SyncInMS, SyncOutMS                             float64
}

func ms(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

func main() {
	xclbin := flag.String("xclbin", "", "xclbin")
	insts := flag.String("insts", "", "insts.bin")
	in := flag.String("in", "", "comma-separated input files, bound first")
	out := flag.String("out", "", "comma-separated output sizes in bytes, bound after the inputs")
	save := flag.String("save", "", "comma-separated files for the outputs")
	runs := flag.Int("runs", 20, "timed runs")
	warmup := flag.Int("warmup", 3, "untimed runs")
	flag.Parse()
	if err := run(*xclbin, *insts, *in, *out, *save, *runs, *warmup); err != nil {
		fmt.Fprintln(os.Stderr, "ironbench:", err)
		os.Exit(1)
	}
}

func split(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(s, ",")
}

func run(xclbin, instsPath, in, out, save string, runs, warmup int) error {
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		return fmt.Errorf("open: %w", err)
	}
	defer dev.Close()
	kr, err := dev.Kernel()
	if err != nil {
		return err
	}
	var r result
	start := time.Now()
	prog, err := kr.Load(ctx, hwnpu.ProgramSource{Path: xclbin})
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}
	defer prog.Close()
	r.LoadMS = ms(time.Since(start))
	k, err := prog.Kernel("MLIR_AIE")
	if err != nil {
		return err
	}
	insts, err := os.ReadFile(instsPath)
	if err != nil {
		return err
	}
	instr, err := dev.AllocBuffer(len(insts), hwnpu.WithBufferKind(hwnpu.BufferInstructions))
	if err != nil {
		return fmt.Errorf("alloc instr: %w", err)
	}
	defer instr.Close()
	copy(instr.Bytes(), insts)
	instr.SyncToDevice(0, len(insts))

	var bufs, outs []*hwnpu.Buffer
	for _, f := range split(in) {
		data, err := os.ReadFile(f)
		if err != nil {
			return err
		}
		b, err := dev.AllocBuffer(len(data))
		if err != nil {
			return fmt.Errorf("alloc input: %w", err)
		}
		defer b.Close()
		copy(b.Bytes(), data)
		bufs = append(bufs, b)
	}
	for _, s := range split(out) {
		n, err := strconv.Atoi(s)
		if err != nil {
			return err
		}
		b, err := dev.AllocBuffer(n)
		if err != nil {
			return fmt.Errorf("alloc output: %w", err)
		}
		defer b.Close()
		bufs = append(bufs, b)
		outs = append(outs, b)
	}
	start = time.Now()
	for _, b := range bufs {
		if err := b.SyncToDevice(0, len(b.Bytes())); err != nil {
			return fmt.Errorf("sync: %w", err)
		}
	}
	r.SyncInMS = ms(time.Since(start))

	rn, err := k.NewRun()
	if err != nil {
		return err
	}
	defer rn.Close()
	rn.SetUint64(0, 3).SetBuffer(1, instr).SetUint32(2, uint32(len(insts)/4))
	for i, b := range bufs {
		rn.SetBuffer(3+i, b)
	}
	start = time.Now()
	if err := rn.Run(ctx); err != nil {
		return fmt.Errorf("first run: %w", err)
	}
	r.FirstMS = ms(time.Since(start))
	start = time.Now()
	for _, b := range outs {
		if err := b.SyncFromDevice(0, len(b.Bytes())); err != nil {
			return err
		}
	}
	r.SyncOutMS = ms(time.Since(start))
	for i, f := range split(save) {
		if err := os.WriteFile(f, outs[i].Bytes(), 0o644); err != nil {
			return err
		}
	}
	for range warmup {
		if err := rn.Run(ctx); err != nil {
			return fmt.Errorf("warmup run: %w", err)
		}
	}
	times := make([]float64, 0, runs)
	for range runs {
		start := time.Now()
		if err := rn.Run(ctx); err != nil {
			return fmt.Errorf("timed run: %w", err)
		}
		times = append(times, ms(time.Since(start)))
	}
	slices.Sort(times)
	sum := 0.0
	for _, t := range times {
		sum += t
	}
	r.Runs, r.MinMS, r.MedianMS, r.MeanMS, r.MaxMS = len(times), times[0], times[len(times)/2], sum/float64(len(times)), times[len(times)-1]
	return json.NewEncoder(os.Stdout).Encode(r)
}
