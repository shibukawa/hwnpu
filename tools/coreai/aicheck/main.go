// Command aicheck lowers Core ML packages to Core AI programs (internal/coreai/ailower) and
// checks them.
//
//	go run ./tools/coreai/aicheck ops PKG...      the MIL operation types of the packages
//	go run ./tools/coreai/aicheck run [-cache DIR] [-scale S] [-cpu] [-coreml] [-calls N] [-ref PKG] PKG...
//	                                               load on Core AI, run, compare with milcpu
//	go run ./tools/coreai/aicheck merge OUT.aimodel NAME=PKG...
//	                                               one program with a function of each package, the weights shared
//
// run takes an .aimodel too, and PROGRAM.aimodel#NAME for its function NAME. With
// HWNPU_AILOWER_CUT=N, -cpu compares the cut program's output with milcpu's value of that
// operation; AICHECK_WORST=1 also prints the elements that differ most; AICHECK_CONST=NAME=V,...
// gives inputs a value instead of random ones (masks, a constant passed as an input). HWNPU_COREAI_UNITS=cpu
// runs on Core AI's CPU (a lowering's errors apart from the Neural Engine's, and without its
// lock); HWNPU_AILOWER_DIVIDE=1 keeps a division by a constant as divide_v1.
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
	case "merge": // merge OUT.aimodel NAME=PKG...
		err = merge(os.Args[2], os.Args[3:])
	case "run":
		fs := flag.NewFlagSet("run", flag.ExitOnError)
		cache := fs.String("cache", "", "the compile cache (default: none, lowered each time)")
		scale := fs.Float64("scale", 0.5, "the inputs' standard deviation")
		cpu := fs.Bool("cpu", false, "compare with the CPU interpreter")
		coreML := fs.Bool("coreml", false, "run on Core ML instead (its errors against milcpu)")
		calls := fs.Int("calls", 1, "calls after the first, timed together")
		ref := fs.String("ref", "", "with -cpu: the package milcpu runs for the comparison, when PKG is an .aimodel")
		fs.Parse(os.Args[2:])
		err = run(fs.Args(), *cache, *scale, *cpu, *coreML, *calls, *ref)
	default:
		err = fmt.Errorf("unknown command %s", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
