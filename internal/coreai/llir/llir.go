// Package llir reads the LLIR of a prebuilt Core AI program (.aimodelc): the Neural Engine
// compiler's MLIR bytecode of a function's region, binary_0.llir.bundle/.../X.mlir.bc, a list of
// tasks (anehlo.ne for the Neural Engine's convolution engine, anehlo.pe for its planar engine)
// in the order the device's finishing service converts them ("Convert 1/1 procedures of ANEHLO
// task N/M" in os_log).
//
// What is read is what was worked out from coreai-build 3605.6.3's output (docs/design/
// apple-coreai.md): the tasks, the kernels they load (anehlo.kload of a memref.view into the
// weights) and the coefficient sets a task reads of its kernel (anehlo.coeff_dma_offset_size:
// number, offset, size).
package llir

import (
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/coreai/mlirbc"
)

// Def is a value: the result res of op (op nil for a block's argument).
type Def struct {
	Op  *mlirbc.Op
	Res int
}

// Scope is the values of a region and of the regions around it up to one isolated from above.
type Scope struct {
	base   int
	defs   []Def // values base.. of this region
	parent *Scope
}

// Get is the value numbered v in the scope.
func (s *Scope) Get(v uint64) (Def, bool) {
	for ; s != nil; s = s.parent {
		if int(v) >= s.base {
			if i := int(v) - s.base; i < len(s.defs) {
				return s.defs[i], true
			}
			return Def{}, false
		}
	}
	return Def{}, false
}

// Inside is the scope of op's regions inside sc: the values its blocks define, after sc's.
func Inside(op *mlirbc.Op, sc *Scope) *Scope {
	child := &Scope{parent: sc}
	if sc != nil {
		child.base = sc.base + len(sc.defs)
	}
	for _, r := range op.Regions {
		for _, blk := range r.Blocks {
			for _, o := range blk.Ops {
				for i := range o.Results {
					child.defs = append(child.defs, Def{o, i})
				}
			}
		}
	}
	return child
}

// Tasks calls visit for each task of f in order, numbered from 1, with the scope around it.
func Tasks(f *mlirbc.File, visit func(n int, task *mlirbc.Op, sc *Scope)) int {
	n := 0
	var walk func(blk *mlirbc.Block, sc *Scope)
	region := func(op *mlirbc.Op, r *mlirbc.Region, sc *Scope) *Scope {
		child := &Scope{parent: sc}
		if op.Isolated || sc == nil {
			child.parent = nil
		} else {
			child.base = sc.base + len(sc.defs)
		}
		for _, blk := range r.Blocks {
			for range blk.Args {
				child.defs = append(child.defs, Def{})
			}
			for _, o := range blk.Ops {
				for i := range o.Results {
					child.defs = append(child.defs, Def{o, i})
				}
			}
		}
		return child
	}
	walk = func(blk *mlirbc.Block, sc *Scope) {
		for _, op := range blk.Ops {
			if op.Name == "anehlo.pe" || op.Name == "anehlo.ne" {
				n++
				visit(n, op, sc)
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
	top := &Scope{}
	for _, o := range f.Top.Ops {
		for i := range o.Results {
			top.defs = append(top.defs, Def{o, i})
		}
	}
	walk(f.Top, top)
	return n
}

// Coeff is a coefficient set a task reads of its kernel: its number, and its bytes in the kernel
// (OK false when one of them is not a constant).
type Coeff struct {
	N, Offset, Size int64
	OK              bool
}

// Kernel is a kernel a task loads (anehlo.kload).
type Kernel struct {
	Load   *mlirbc.Op
	Coeffs []Coeff
	// Size is the bytes of the kernel's view (-1 when its type is not a one-dimensional memref),
	// End the end of the last coefficient set read.
	Size, End int64
}

// Overrun reports whether the task reads coefficients past the end of the kernel.
func (k Kernel) Overrun() bool { return k.Size >= 0 && k.End > k.Size }

// Kernels are the kernels the task loads; in is Inside(task, sc).
func Kernels(f *mlirbc.File, task *mlirbc.Op, in *Scope) []Kernel {
	var out []Kernel
	for _, r := range task.Regions {
		for _, blk := range r.Blocks {
			for _, o := range blk.Ops {
				if o.Name != "anehlo.kload" || len(o.Operands) == 0 {
					continue
				}
				k := Kernel{Load: o, Size: -1}
				if n, ok := ViewBytes(f, o.Operands[0], in); ok {
					k.Size = n
				}
				for _, kr := range o.Regions {
					for _, kb := range kr.Blocks {
						for _, ko := range kb.Ops {
							if ko.Name != "anehlo.coeff_dma_offset_size" || len(ko.Operands) != 3 {
								continue
							}
							c := Coeff{OK: true}
							for i, dst := range []*int64{&c.N, &c.Offset, &c.Size} {
								v, ok := Constant(f, ko.Operands[i], in)
								*dst, c.OK = v, c.OK && ok
							}
							if c.OK {
								k.End = max(k.End, c.Offset+c.Size)
							}
							k.Coeffs = append(k.Coeffs, c)
						}
					}
				}
				out = append(out, k)
			}
		}
	}
	return out
}

// Overrun is a task that reads past the end of a kernel it loads.
type Overrun struct {
	Task          int   // from 1
	Reads, Kernel int64 // the bytes it reads from the kernel's start, and the kernel's
}

// Overruns are the tasks of the LLIR bytecode b that read coefficients past the end of the
// kernel they load, and the number of its tasks.
//
// coreai-build's compiler for the M1 (h13g, 3605.6.3) keeps one kernel for the activations of a
// program that have the same constants and gives it the size one of them needs (the last): a
// task with a coefficient set a channel (8 of 128 bytes) is left with the kernel of a task with
// one set (128 bytes). The M1 finishes such a program and does not end its first call: the
// Neural Engine gives it up after 100 s and restarts, and runs nothing for any process until
// then; in a program of several functions the finishing service crashes instead
// (docs/known-issues.md 16 and 17, 2026-11-27). Every program that did so on the M1 had such
// tasks, and none of the 40 that ran had one.
func Overruns(b []byte) (tasks int, over []Overrun, err error) {
	f, err := mlirbc.Read(b)
	if err != nil {
		return 0, nil, err
	}
	tasks = Tasks(f, func(n int, task *mlirbc.Op, sc *Scope) {
		for _, k := range Kernels(f, task, Inside(task, sc)) {
			if k.Overrun() {
				over = append(over, Overrun{Task: n, Reads: k.End, Kernel: k.Size})
				break
			}
		}
	})
	return tasks, over, nil
}

// Files are the LLIR files of the prebuilt program at dir (X.ARCH.aimodelc), one for each region
// of each function, by the function's name.
func Files(dir string) (map[string][]string, error) {
	out := map[string][]string{}
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".mlir.bc") {
			return err
		}
		// FUNCTION_HASH_N_ANE_region_I_J.mlir.bc, the hash 22 characters of base64url
		name := filepath.Base(path)
		fn := name
		if i := strings.Index(name, "_ANE_region_"); i > 0 {
			fn = name[:i]
			if j := strings.LastIndexByte(fn, '_'); j > 0 { // _N
				fn = fn[:j]
			}
			if len(fn) > 23 && fn[len(fn)-23] == '_' {
				fn = fn[:len(fn)-23]
			}
		}
		out[fn] = append(out[fn], path)
		return nil
	})
	return out, err
}

// CheckKernels reads the LLIR of the prebuilt program at dir and returns an error that names the
// functions with tasks that read past their kernel (Overruns), nil when there is none. unread
// are the files that could not be read as LLIR, which tell nothing.
func CheckKernels(dir string) (unread []string, err error) {
	files, err := Files(dir)
	if err != nil {
		return nil, err
	}
	var bad []string
	for _, fn := range slices.Sorted(maps.Keys(files)) {
		n, first := 0, Overrun{}
		for _, path := range files[fn] {
			b, err := os.ReadFile(path)
			if err != nil {
				return unread, err
			}
			_, over, err := Overruns(b)
			if err != nil {
				unread = append(unread, path)
				continue
			}
			if len(over) > 0 && n == 0 {
				first = over[0]
			}
			n += len(over)
		}
		if n > 0 {
			bad = append(bad, fmt.Sprintf("%s: %d (task %d reads %d bytes of a kernel of %d)", fn, n, first.Task, first.Reads, first.Kernel))
		}
	}
	if len(bad) > 0 {
		return unread, fmt.Errorf("%w: %s", ErrKernelOverrun, strings.Join(bad, "; "))
	}
	return unread, nil
}

// ErrKernelOverrun is CheckKernels' error of a program with tasks that read past their kernel.
var ErrKernelOverrun = fmt.Errorf("tasks read past the kernel they load (the Neural Engine does not end the first call of such a program, or the device's finishing service crashes on it: docs/known-issues.md 16)")

// ViewBytes is the size of the one-dimensional memref that v names (a kernel's view).
func ViewBytes(f *mlirbc.File, v uint64, sc *Scope) (int64, bool) {
	d, ok := sc.Get(v)
	if !ok || d.Op == nil || d.Res >= len(d.Op.Results) || int(d.Op.Results[d.Res]) >= len(f.Types) {
		return 0, false
	}
	dims, _, ok := MemrefShape(f, f.Types[d.Op.Results[d.Res]])
	if !ok || len(dims) != 1 {
		return 0, false
	}
	return dims[0], true
}

// Memref is the text of a builtin memref type: its shape and element type. ok is false for
// another type.
func Memref(f *mlirbc.File, e mlirbc.Entry) (string, bool) {
	dims, elem, ok := MemrefShape(f, e)
	if !ok {
		return "", false
	}
	s := make([]string, len(dims))
	for i, d := range dims {
		s[i] = strconv.FormatInt(d, 10)
	}
	et := fmt.Sprintf("t%d", elem)
	if int(elem) < len(f.Types) && !f.Types[elem].Custom {
		et = f.Types[elem].String()
	} else if int(elem) < len(f.Types) {
		et = fmt.Sprintf("%x", f.Types[elem].Data)
	}
	return "memref<" + strings.Join(s, "x") + " " + et + ">", true
}

// MemrefShape reads a builtin memref type's own encoding: its shape and its element type's
// index. ok is false for another type.
func MemrefShape(f *mlirbc.File, e mlirbc.Entry) (dims []int64, elem uint64, ok bool) {
	if !e.Custom || e.Dialect != "builtin" {
		return nil, 0, false
	}
	b := e.Data
	i := 0
	next := func() (uint64, bool) {
		v, n, ok := Uvarint(b, i)
		i += n
		return v, ok
	}
	code, ok := next()
	if !ok || code != 10 && code != 11 {
		return nil, 0, false
	}
	if code == 11 {
		if _, ok = next(); !ok { // the memory space
			return nil, 0, false
		}
	}
	n, ok := next()
	if !ok || n > 16 {
		return nil, 0, false
	}
	for range n {
		v, ok := next()
		if !ok {
			return nil, 0, false
		}
		dims = append(dims, int64(v>>1)^-int64(v&1))
	}
	elem, _ = next()
	return dims, elem, true
}

// Uvarint reads the bytecode's variable length integer at b[i:]: its value and the bytes it took.
func Uvarint(b []byte, i int) (v uint64, n int, ok bool) {
	if i >= len(b) {
		return 0, 0, false
	}
	c := b[i]
	if c&1 != 0 {
		return uint64(c >> 1), 1, true
	}
	if c == 0 {
		if i+9 > len(b) {
			return 0, 0, false
		}
		for k := 8; k >= 1; k-- {
			v = v<<8 | uint64(b[i+k])
		}
		return v, 9, true
	}
	extra := 0
	for c>>extra&1 == 0 {
		extra++
	}
	if i+extra+1 > len(b) {
		return 0, 0, false
	}
	for k := extra; k >= 0; k-- {
		v = v<<8 | uint64(b[i+k])
	}
	return v >> uint(extra+1), extra + 1, true
}

// Constant is the value of an arith.constant of an integer that v names.
func Constant(f *mlirbc.File, v uint64, sc *Scope) (int64, bool) {
	d, ok := sc.Get(v)
	if !ok || d.Op == nil || d.Op.Name != "arith.constant" {
		return 0, false
	}
	a, _, ok := Uvarint(d.Op.Props, 0)
	if !ok || int(a) >= len(f.Attrs) || !f.Attrs[a].Custom {
		return 0, false
	}
	b := f.Attrs[a].Data // the builtin IntegerAttr: its code (8), its type, the value zigzag
	if len(b) < 3 || b[0] != 0x11 {
		return 0, false
	}
	_, n, ok := Uvarint(b, 1) // the type
	if !ok {
		return 0, false
	}
	u, _, ok := Uvarint(b, 1+n)
	if !ok {
		return 0, false
	}
	return int64(u>>1) ^ -int64(u&1), true
}
