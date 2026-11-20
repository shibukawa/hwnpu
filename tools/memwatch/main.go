// Command memwatch runs a command and samples its memory every 0.1 s until it exits
// (docs/new-model-checklist.md 6 and 11.2): the process's footprint (Activity Monitor's) and its
// wired memory (mlock: the experts a host wires), the Neural Engine's kernel memory
// (mem.NPUWired) and the system's free percent and pressure. It prints the peak of each and of
// footprint + wired + Neural Engine together, models.yaml's memory, and with -csv FILE writes
// the samples.
//
//	go run ./tools/memwatch [-csv FILE] hwnpu-chat chat -m MODEL "question"
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/shibukawa/hwnpu/internal/mem"
	"github.com/shibukawa/hwnpu/internal/sysload"
)

func main() {
	csv := flag.String("csv", "", "write the samples to this file")
	every := flag.Duration("every", 100*time.Millisecond, "the time between samples")
	flag.Parse()
	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: memwatch [-csv FILE] COMMAND ARGS...")
		os.Exit(2)
	}
	cmd := exec.Command(flag.Arg(0), flag.Args()[1:]...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(os.Stderr, "memwatch:", err)
		os.Exit(1)
	}
	var out *os.File
	if *csv != "" {
		var err error
		if out, err = os.Create(*csv); err != nil {
			fmt.Fprintln(os.Stderr, "memwatch:", err)
			os.Exit(1)
		}
		defer out.Close()
		fmt.Fprintln(out, "seconds,footprint,wired,npu,free_percent,pressure")
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	var s sysload.ProcSampler
	base, _ := mem.NPUWired() // what the Neural Engine held before, of other processes
	var peakFoot, peakWired, peakNPU, peakAll int64
	minFree := 100
	start := time.Now()
	tick := time.NewTicker(*every)
	defer tick.Stop()
	var err error
loop:
	for {
		select {
		case err = <-done:
			break loop
		case <-tick.C:
		}
		p, ok := s.Read([]int{cmd.Process.Pid})[cmd.Process.Pid]
		if !ok {
			continue
		}
		npu, _ := mem.NPUWired()
		npu = max(npu-base, 0)
		pr, _ := mem.PressureNow()
		peakFoot, peakWired, peakNPU = max(peakFoot, p.Memory-p.Wired), max(peakWired, p.Wired), max(peakNPU, npu)
		peakAll = max(peakAll, p.Memory+npu)
		minFree = min(minFree, pr.Free)
		if out != nil {
			fmt.Fprintf(out, "%.1f,%d,%d,%d,%d,%d\n", time.Since(start).Seconds(), p.Memory-p.Wired, p.Wired, npu, pr.Free, pr.Level)
		}
	}
	gb := func(v int64) float64 { return float64(v) / 1e9 }
	fmt.Fprintf(os.Stderr, "memwatch: peak footprint %.2f GB, wired %.2f GB, Neural Engine +%.2f GB (%.2f GB before); together %.2f GB; the system's free at least %d%%; %.0f s\n",
		gb(peakFoot), gb(peakWired), gb(peakNPU), gb(base), gb(peakAll), minFree, time.Since(start).Seconds())
	if err != nil {
		os.Exit(1)
	}
}
