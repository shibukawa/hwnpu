// Command gemvbench runs a bf16 GEMV design built by amd/IRON (iron.operators.GEMV) on an
// AMD NPU through hwnpu, checks it against the CPU and reports the time and the bandwidth
// of reading the matrix.
//
//	go run ./tools/amd/gemvbench -xclbin final.xclbin -insts insts.bin -M 8192 -K 2048
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
)

func bf16(f float32) uint16 {
	b := math.Float32bits(f)
	b += 0x7fff + (b>>16)&1
	return uint16(b >> 16)
}

func f32(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }

func main() {
	xclbin := flag.String("xclbin", "", "xclbin")
	insts := flag.String("insts", "", "instruction stream")
	kernel := flag.String("kernel", "MLIR_AIE", "kernel name")
	m := flag.Int("M", 8192, "rows of A")
	k := flag.Int("K", 2048, "columns of A")
	runs := flag.Int("runs", 50, "timed runs")
	seconds := flag.Float64("seconds", 0, "also run back to back this long and report the package power (amdgpu PPT)")
	flag.Parse()
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
	prog, err := kr.Load(ctx, hwnpu.ProgramSource{Path: *xclbin})
	if err != nil {
		log.Fatal(err)
	}
	defer prog.Close()
	loadMS := time.Since(t0).Seconds() * 1000
	kern, err := prog.Kernel(*kernel)
	if err != nil {
		log.Fatal(err)
	}
	for _, a := range kern.Args() {
		fmt.Printf("arg %d %s %s offset=%#x size=%d buffer=%v\n", a.Index, a.Name, a.Type, a.Offset, a.Size, a.Buffer)
	}
	code, err := os.ReadFile(*insts)
	if err != nil {
		log.Fatal(err)
	}
	instr, err := dev.AllocBuffer(len(code), hwnpu.WithBufferKind(hwnpu.BufferInstructions))
	if err != nil {
		log.Fatal(err)
	}
	defer instr.Close()
	copy(instr.Bytes(), code)
	instr.SyncToDevice(0, len(code))

	M, K := *m, *k
	a, err := dev.AllocBuffer(M * K * 2)
	if err != nil {
		log.Fatal(err)
	}
	defer a.Close()
	b, _ := dev.AllocBuffer(K * 2)
	defer b.Close()
	c, _ := dev.AllocBuffer(M * 2)
	defer c.Close()
	rng := rand.New(rand.NewPCG(1, 2))
	av, bv := a.Bytes(), b.Bytes()
	for i := 0; i < M*K; i++ {
		binary.LittleEndian.PutUint16(av[2*i:], bf16(float32(rng.NormFloat64())*0.05))
	}
	for i := 0; i < K; i++ {
		binary.LittleEndian.PutUint16(bv[2*i:], bf16(float32(rng.NormFloat64())))
	}
	a.SyncToDevice(0, M*K*2)
	b.SyncToDevice(0, K*2)
	clear(c.Bytes())
	c.SyncToDevice(0, M*2)

	run, err := kern.NewRun()
	if err != nil {
		log.Fatal(err)
	}
	defer run.Close()
	run.SetUint64(0, 3).SetBuffer(1, instr).SetUint32(2, uint32(len(code)/4)).
		SetBuffer(3, a).SetBuffer(4, b).SetBuffer(5, c)
	t0 = time.Now()
	if err := run.Run(ctx); err != nil {
		log.Fatal(err)
	}
	firstMS := time.Since(t0).Seconds() * 1000
	c.SyncFromDevice(0, M*2)

	// Check against float64 on the CPU, relative to the row's scale.
	cv := c.Bytes()
	worst, bad := 0.0, 0
	for r := 0; r < M; r++ {
		var sum, mag float64
		for j := 0; j < K; j++ {
			p := float64(f32(binary.LittleEndian.Uint16(av[2*(r*K+j):]))) * float64(f32(binary.LittleEndian.Uint16(bv[2*j:])))
			sum += p
			mag += math.Abs(p)
		}
		got := float64(f32(binary.LittleEndian.Uint16(cv[2*r:])))
		e := math.Abs(got-sum) / (mag/math.Sqrt(float64(K)) + 1e-6)
		worst = max(worst, e)
		if e > 0.05 {
			bad++
		}
	}
	fmt.Printf("check: worst error %.4f of the row scale, %d of %d rows over 0.05\n", worst, bad, M)

	times := make([]float64, 0, *runs)
	for i := 0; i < *runs; i++ {
		t := time.Now()
		if err := run.Run(ctx); err != nil {
			log.Fatal(err)
		}
		times = append(times, time.Since(t).Seconds()*1000)
	}
	slices.Sort(times)
	med := times[len(times)/2]
	gb := float64(M*K*2) / 1e9
	fmt.Printf("M=%d K=%d: load %.1f ms, first %.2f ms, median %.3f ms, min %.3f ms, %.1f GB/s (median), %.2f GFLOP/s\n",
		M, K, loadMS, firstMS, med, times[0], gb/(med/1000), 2*float64(M*K)/(med/1000)/1e9)
	if *seconds > 0 {
		idle := pptAverage(2 * time.Second)
		done := make(chan float64)
		stop := time.Now().Add(time.Duration(*seconds * float64(time.Second)))
		go func() { done <- pptAverage(time.Duration(*seconds*float64(time.Second)) - 200*time.Millisecond) }()
		n := 0
		for time.Now().Before(stop) {
			if err := run.Run(ctx); err != nil {
				log.Fatal(err)
			}
			n++
		}
		busy := <-done
		per := *seconds / float64(n)
		fmt.Printf("loop: %d runs in %.1f s (%.3f ms each), package power idle %.2f W, busy %.2f W, %.2f mJ per run above idle, %.1f GB/s\n",
			n, *seconds, per*1000, idle, busy, (busy-idle)*per*1000, gb/per)
	}
	if bad > 0 {
		os.Exit(1)
	}
}

// pptAverage samples the package power the amdgpu driver reports on an APU (PPT, the
// whole socket) every 50 ms for d and returns the mean in watts, or NaN without it.
func pptAverage(d time.Duration) float64 {
	path := ""
	hw, _ := filepath.Glob("/sys/class/hwmon/hwmon*")
	for _, h := range hw {
		if name, _ := os.ReadFile(filepath.Join(h, "name")); strings.TrimSpace(string(name)) == "amdgpu" {
			path = filepath.Join(h, "power1_input")
		}
	}
	if path == "" {
		time.Sleep(d)
		return math.NaN()
	}
	var sum float64
	n := 0
	for end := time.Now().Add(d); time.Now().Before(end); time.Sleep(50 * time.Millisecond) {
		b, err := os.ReadFile(path)
		if err != nil {
			return math.NaN()
		}
		v, _ := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
		sum += v / 1e6
		n++
	}
	return sum / float64(n)
}
