// Command aicheck lowers Core ML packages to Core AI programs (internal/coreai/ailower) and
// checks them.
//
//	go run ./tools/coreai/aicheck ops PKG...      the MIL operation types of the packages
//	go run ./tools/coreai/aicheck run [-cache DIR] [-scale S] [-cpu] PKG...
//	                                               load on Core AI, run, compare with milcpu
package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: aicheck ops|lower|run PKG...")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "ops":
		err = opTypes(os.Args[2:])
	case "list":
		n := 50
		fmt.Sscan(os.Getenv("N"), &n)
		err = listOps(os.Args[2], n)
	case "probe":
		err = probe(os.Args[2:], os.Getenv("KEEP") != "")
	case "lower":
		err = lowerTo(os.Args[2], os.Args[3:])
	case "run":
		fs := flag.NewFlagSet("run", flag.ExitOnError)
		cache := fs.String("cache", "", "the compile cache (default: none, lowered each time)")
		scale := fs.Float64("scale", 0.5, "the inputs' standard deviation")
		cpu := fs.Bool("cpu", false, "compare with the CPU interpreter")
		fs.Parse(os.Args[2:])
		err = run(fs.Args(), *cache, *scale, *cpu)
	default:
		err = fmt.Errorf("unknown command %s", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
