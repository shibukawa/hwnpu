// Command llirtasks lists the Neural Engine tasks of an .aimodelc's LLIR (the Neural Engine
// compiler's MLIR bytecode, binary_0.llir.bundle/.../X.mlir.bc) in their order, the order of
// the device's finishing service ("Convert 1/1 procedures of ANEHLO task N/M" in os_log): what
// each computes and the kernel it loads.
//
//	go run ./tools/coreai/llirtasks [-task N] [-around K] [-check] X.mlir.bc
//
// A kernel's line ends with OVERRUN when the task reads coefficients past the end of the kernel
// it loads: coreai-build shares one kernel between the activations of the same constants and
// gives it the size of one of them, so a task with a coefficient set a channel (8 of 128 bytes)
// can be left with the kernel of a task with one set (128 bytes). On an M1 such a program is
// finished and then does not end its first call (docs/known-issues.md 16). -check prints these
// tasks alone and fails when there is one.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/shibukawa/hwnpu/internal/coreai/llir"
	"github.com/shibukawa/hwnpu/internal/coreai/mlirbc"
)

var (
	f        *mlirbc.File
	check    bool
	overruns int
)

func main() {
	task := flag.Int("task", 0, "print this task (from 1) in full")
	around := flag.Int("around", 0, "with -task, the tasks this many before and after too")
	flag.BoolVar(&check, "check", false, "print only the tasks that read past their kernel, and fail when there is one")
	flag.Parse()
	b, err := os.ReadFile(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}
	if f, err = mlirbc.Read(b); err != nil {
		log.Fatal(err)
	}
	n := llir.Tasks(f, func(n int, op *mlirbc.Op, sc *llir.Scope) {
		if *task == 0 || n >= *task-*around && n <= *task+*around {
			printTask(n, op, sc, *task != 0)
		}
	})
	if check {
		fmt.Printf("%d tasks, %d read past their kernel\n", n, overruns)
		if overruns > 0 {
			os.Exit(1)
		}
		return
	}
	fmt.Printf("%d tasks\n", n)
}

func printTask(n int, op *mlirbc.Op, sc *llir.Scope, full bool) {
	var names []string
	var kernels []string
	over := false
	child := llir.Inside(op, sc)
	loads := llir.Kernels(f, op, child)
	for _, r := range op.Regions {
		for _, blk := range r.Blocks {
			for _, o := range blk.Ops {
				switch o.Name {
				case "memref.reinterpret_cast", "memref.alloc", "memref.subview", "memref.view", "anehlo.bind", "anehlo.raster", "anehlo.yield", "llir.l2_alias", "anehlo.fifo", "memref.transpose":
				default:
					names = append(names, strings.TrimPrefix(o.Name, "anehlo."))
				}
				if full {
					fmt.Printf("      %s props %x operands %v%s\n", o.Name, o.Props, o.Operands, typesOf(o))
				}
			}
		}
	}
	for _, kl := range loads {
		k := chain(kl.Load.Operands[0], child, 0)
		for _, c := range kl.Coeffs {
			if c.OK {
				k += fmt.Sprintf(" coeff %d %d %d", c.N, c.Offset, c.Size)
			} else {
				k += " coeff ?"
			}
		}
		if kl.Overrun() {
			k += fmt.Sprintf(" OVERRUN: reads %d bytes of a kernel of %d", kl.End, kl.Size)
			over = true
		}
		kernels = append(kernels, k)
	}
	if over {
		overruns++
	} else if check {
		return
	}
	fmt.Printf("task %d %s: %s\n", n, strings.TrimPrefix(op.Name, "anehlo."), strings.Join(names, " "))
	for _, k := range kernels {
		fmt.Printf("    kernel %s\n", k)
	}
}

func typesOf(o *mlirbc.Op) string {
	var s []string
	for _, t := range o.Results {
		if int(t) < len(f.Types) {
			if m, ok := llir.Memref(f, f.Types[t]); ok {
				s = append(s, m)
			} else {
				s = append(s, short(f.Types[t].String()))
			}
		}
	}
	if len(s) == 0 {
		return ""
	}
	return " -> " + strings.Join(s, ", ")
}

func short(s string) string {
	if len(s) > 110 {
		return s[:110] + "..."
	}
	return s
}

// chain follows a value to what it is made of: the views and casts back to a global or a kernel.
func chain(v uint64, sc *llir.Scope, depth int) string {
	d, ok := sc.Get(v)
	if !ok || d.Op == nil {
		return fmt.Sprintf("%%%d(?)", v)
	}
	s := d.Op.Name + propAttrs(d.Op) + typesOf(d.Op)
	if d.Op.Name == "memref.view" && len(d.Op.Operands) > 1 { // its offset in the buffer it views
		if n, ok := llir.Constant(f, d.Op.Operands[1], sc); ok {
			s += fmt.Sprintf(" at %d", n)
		}
	}
	if depth < 6 && len(d.Op.Operands) > 0 && strings.HasPrefix(d.Op.Name, "memref.") {
		s += " <- " + chain(d.Op.Operands[0], sc, depth+1)
	}
	return s
}

// propAttrs reads an operation's properties as attribute indices where they are (the text of the
// symbols and types the dialects' own encodings refer to).
func propAttrs(o *mlirbc.Op) string {
	var out []string
	for i := 0; i < len(o.Props); {
		c := o.Props[i]
		if c&1 == 0 {
			break
		}
		if a := int(c >> 1); a < len(f.Attrs) && !f.Attrs[a].Custom {
			out = append(out, short(f.Attrs[a].String()))
		}
		i++
	}
	if len(out) == 0 {
		return fmt.Sprintf(" props %x", o.Props)
	}
	return " {" + strings.Join(out, "; ") + "}"
}
