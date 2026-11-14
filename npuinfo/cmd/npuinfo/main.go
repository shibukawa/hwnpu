// Command npuinfo prints what the OS tells about NPUs as JSON (package npuinfo).
//
//	go run ./npuinfo/cmd/npuinfo            # once
//	go run ./npuinfo/cmd/npuinfo -n 5 -i 1s # five times, a second apart, one JSON per line
//	go run ./npuinfo/cmd/npuinfo -wake      # also query a sleeping NPU (wakes it)
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/shibukawa/hwnpu/npuinfo"
)

func main() {
	wake := flag.Bool("wake", false, "run queries that wake a sleeping NPU")
	n := flag.Int("n", 1, "times to read")
	interval := flag.Duration("i", time.Second, "time between reads")
	sample := flag.Duration("sample", 0, "time between the two samples of rate counters (Windows, macOS IOReport; <0 skips them on macOS)")
	noCaches := flag.Bool("no-caches", false, "do not walk compile cache directories (macOS)")
	flag.Parse()

	opts := npuinfo.Options{Wake: *wake, SampleInterval: *sample, SkipCaches: *noCaches}
	enc := json.NewEncoder(os.Stdout)
	if *n == 1 {
		enc.SetIndent("", "  ")
	}
	for i := 0; i < *n; i++ {
		if i > 0 {
			time.Sleep(*interval)
		}
		info, err := npuinfo.GetWithContext(context.Background(), opts)
		if err != nil {
			fmt.Fprintln(os.Stderr, "npuinfo:", err)
		}
		if err := enc.Encode(info); err != nil {
			fmt.Fprintln(os.Stderr, "npuinfo:", err)
			os.Exit(1)
		}
	}
}
