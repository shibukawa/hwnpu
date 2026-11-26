// Command llirtasks lists the Neural Engine tasks of an .aimodelc's LLIR (the Neural Engine
// compiler's MLIR bytecode, binary_0.llir.bundle/.../X.mlir.bc) in their order, the order of
// the device's finishing service ("Convert 1/1 procedures of ANEHLO task N/M" in os_log): what
// each computes and the kernel it loads.
//
//	go run ./tools/coreai/llirtasks [-task N] [-around K] X.mlir.bc
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/shibukawa/hwnpu/internal/coreai/mlirbc"
)

type def struct {
	op  *mlirbc.Op
	res int
}

// scope is the values of a region and of the regions around it up to one isolated from above.
type scope struct {
	base   int
	defs   []def // values base.. of this region
	parent *scope
}

func (s *scope) get(v uint64) (def, bool) {
	for ; s != nil; s = s.parent {
		if int(v) >= s.base {
			if i := int(v) - s.base; i < len(s.defs) {
				return s.defs[i], true
			}
			return def{}, false
		}
	}
	return def{}, false
}

var f *mlirbc.File

func main() {
	task := flag.Int("task", 0, "print this task (from 1) in full")
	around := flag.Int("around", 0, "with -task, the tasks this many before and after too")
	flag.Parse()
	b, err := os.ReadFile(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}
	if f, err = mlirbc.Read(b); err != nil {
		log.Fatal(err)
	}
	n := 0
	var walk func(blk *mlirbc.Block, sc *scope)
	region := func(op *mlirbc.Op, r *mlirbc.Region, sc *scope) *scope {
		child := &scope{parent: sc}
		if op.Isolated || sc == nil {
			child.parent = nil
		} else {
			child.base = sc.base + len(sc.defs)
		}
		for _, blk := range r.Blocks {
			for range blk.Args {
				child.defs = append(child.defs, def{})
			}
			for _, o := range blk.Ops {
				for i := range o.Results {
					child.defs = append(child.defs, def{o, i})
				}
			}
		}
		return child
	}
	walk = func(blk *mlirbc.Block, sc *scope) {
		for _, op := range blk.Ops {
			if op.Name == "anehlo.pe" || op.Name == "anehlo.ne" {
				n++
				if *task == 0 || n >= *task-*around && n <= *task+*around {
					printTask(n, op, sc, *task != 0)
				}
				continue
			}
			for _, r := range op.Regions {
				child := region(op, r, sc)
				for _, b := range r.Blocks {
					walk(b, child)
				}
			}
		}
	}
	top := &scope{}
	for _, o := range f.Top.Ops {
		for i := range o.Results {
			top.defs = append(top.defs, def{o, i})
		}
	}
	walk(f.Top, top)
	fmt.Printf("%d tasks\n", n)
}

func printTask(n int, op *mlirbc.Op, sc *scope, full bool) {
	var names []string
	var kernels []string
	child := &scope{parent: sc, base: sc.base + len(sc.defs)}
	for _, r := range op.Regions {
		for _, blk := range r.Blocks {
			for _, o := range blk.Ops {
				for i := range o.Results {
					child.defs = append(child.defs, def{o, i})
				}
			}
		}
	}
	for _, r := range op.Regions {
		for _, blk := range r.Blocks {
			for _, o := range blk.Ops {
				switch o.Name {
				case "memref.reinterpret_cast", "memref.alloc", "memref.subview", "memref.view", "anehlo.bind", "anehlo.raster", "anehlo.yield", "llir.l2_alias", "anehlo.fifo", "memref.transpose":
				default:
					names = append(names, strings.TrimPrefix(o.Name, "anehlo."))
				}
				if o.Name == "anehlo.kload" && len(o.Operands) > 0 {
					kernels = append(kernels, chain(o.Operands[0], child, 0))
				}
				if full {
					fmt.Printf("      %s props %x operands %v%s\n", o.Name, o.Props, o.Operands, typesOf(o))
				}
			}
		}
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
			if m, ok := memref(f.Types[t]); ok {
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

// memref reads a builtin memref type's own encoding: its shape and element type (the type's
// index). ok is false for another type.
func memref(e mlirbc.Entry) (string, bool) {
	if !e.Custom || e.Dialect != "builtin" {
		return "", false
	}
	b := e.Data
	i := 0
	varint := func() (uint64, bool) {
		if i >= len(b) {
			return 0, false
		}
		c := b[i]
		if c&1 != 0 {
			i++
			return uint64(c >> 1), true
		}
		if c == 0 {
			if i+9 > len(b) {
				return 0, false
			}
			var v uint64
			for k := 8; k >= 1; k-- {
				v = v<<8 | uint64(b[i+k])
			}
			i += 9
			return v, true
		}
		extra := 0
		for c>>extra&1 == 0 {
			extra++
		}
		if i+extra+1 > len(b) {
			return 0, false
		}
		var v uint64
		for k := extra; k >= 0; k-- {
			v = v<<8 | uint64(b[i+k])
		}
		i += extra + 1
		return v >> (extra + 1), true
	}
	code, ok := varint()
	if !ok || code != 10 && code != 11 {
		return "", false
	}
	if code == 11 {
		if _, ok = varint(); !ok { // the memory space
			return "", false
		}
	}
	n, ok := varint()
	if !ok || n > 16 {
		return "", false
	}
	var dims []string
	for range n {
		v, ok := varint()
		if !ok {
			return "", false
		}
		dims = append(dims, fmt.Sprint(int64(v>>1)^-int64(v&1)))
	}
	elem, _ := varint()
	et := fmt.Sprintf("t%d", elem)
	if int(elem) < len(f.Types) && !f.Types[elem].Custom {
		et = f.Types[elem].String()
	} else if int(elem) < len(f.Types) {
		et = fmt.Sprintf("%x", f.Types[elem].Data)
	}
	return "memref<" + strings.Join(dims, "x") + " " + et + ">", true
}

func short(s string) string {
	if len(s) > 110 {
		return s[:110] + "..."
	}
	return s
}

// chain follows a value to what it is made of: the views and casts back to a global or a kernel.
func chain(v uint64, sc *scope, depth int) string {
	d, ok := sc.get(v)
	if !ok || d.op == nil {
		return fmt.Sprintf("%%%d(?)", v)
	}
	s := d.op.Name + propAttrs(d.op) + typesOf(d.op)
	if depth < 6 && len(d.op.Operands) > 0 && strings.HasPrefix(d.op.Name, "memref.") {
		s += " <- " + chain(d.op.Operands[0], sc, depth+1)
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
