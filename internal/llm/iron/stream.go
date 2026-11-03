package iron

import (
	"encoding/json"
	"fmt"
	"os"
)

// Program builds an instruction stream (a TXN) from a step's dispatch-time scalars: on an
// xclbin per step (NPU1) mlir-aie generates C++ for it, and tools/amd/iron_stream.py turns
// that into this form, with the same semantics as its run.
type Program struct {
	ABI  []string `json:"abi"`
	Regs int      `json:"regs"`
	Msgs []string `json:"msgs"`
	Ops  []sop    `json:"ops"`
}

// sop is one operation: its code and its integer operands (types and the operator of "bin"
// are held apart).
type sop struct {
	code byte
	typ  ctype
	op   string
	args []int64
}

type ctype uint8

const (
	tI8 ctype = iota
	tU8
	tI16
	tU16
	tI32
	tU32
	tI64
	tU64
	tBool
)

var ctypes = map[string]ctype{
	"i8": tI8, "u8": tU8, "i16": tI16, "u16": tU16, "i32": tI32, "u32": tU32, "i64": tI64, "u64": tU64, "b": tBool,
	"int8_t": tI8, "uint8_t": tU8, "int16_t": tI16, "uint16_t": tU16, "int32_t": tI32, "uint32_t": tU32,
	"int64_t": tI64, "uint64_t": tU64, "bool": tBool, "size_t": tU64,
}

const (
	opK byte = iota
	opCast
	opBin
	opSel
	opArr
	opSet
	opChk
	opW32
	opMW32
	opMP32
	opSync
	opBW
	opPatch
	opPDI
	opPre
	opHdr
)

var opcodes = map[string]struct {
	code  byte
	typed bool // [code, dst, type, ...]
	n     int  // integer operands after the type (and the operator)
}{
	"k": {opK, true, 1}, "cast": {opCast, true, 1}, "bin": {opBin, true, 2}, "sel": {opSel, true, 3},
	"arr": {opArr, false, 2}, "set": {opSet, false, 3}, "chk": {opChk, false, 2},
	"w32": {opW32, false, 2}, "mw32": {opMW32, false, 3}, "mp32": {opMP32, false, 3}, "sync": {opSync, false, 6},
	"bw": {opBW, false, 5}, "patch": {opPatch, false, 4}, "pdi": {opPDI, false, 3}, "pre": {opPre, false, 1},
	"hdr": {opHdr, false, 5},
}

func (o *sop) UnmarshalJSON(b []byte) error {
	var raw []json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	if len(raw) == 0 {
		return fmt.Errorf("iron: empty stream operation")
	}
	var code string
	if err := json.Unmarshal(raw[0], &code); err != nil {
		return err
	}
	c, ok := opcodes[code]
	if !ok {
		return fmt.Errorf("iron: stream operation %q", code)
	}
	o.code = c.code
	rest := raw[1:]
	num := func(r json.RawMessage) (int64, error) {
		var v int64
		err := json.Unmarshal(r, &v)
		return v, err
	}
	if c.typed {
		// [code, dst, type, (op,) operands...]
		if len(rest) < 2 {
			return fmt.Errorf("iron: stream operation %s: %d operands", code, len(rest))
		}
		dst, err := num(rest[0])
		if err != nil {
			return err
		}
		var t string
		if err := json.Unmarshal(rest[1], &t); err != nil {
			return err
		}
		if o.typ, ok = ctypes[t]; !ok {
			return fmt.Errorf("iron: stream type %q", t)
		}
		rest = rest[2:]
		if code == "bin" {
			if len(rest) == 0 {
				return fmt.Errorf("iron: bin without an operator")
			}
			if err := json.Unmarshal(rest[0], &o.op); err != nil {
				return err
			}
			rest = rest[1:]
		}
		o.args = append(o.args, dst)
	}
	if len(rest) != c.n {
		return fmt.Errorf("iron: stream operation %s: %d operands, want %d", code, len(rest), c.n)
	}
	for _, r := range rest {
		v, err := num(r)
		if err != nil {
			return fmt.Errorf("iron: stream operation %s: %w", code, err)
		}
		o.args = append(o.args, v)
	}
	return nil
}

// ReadProgram reads a program written by tools/amd/iron_export_lm.py and checks its
// registers.
func ReadProgram(path string) (*Program, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var p Program
	if err := json.Unmarshal(b, &p); err != nil {
		return nil, fmt.Errorf("iron: %s: %w", path, err)
	}
	return &p, p.check()
}

func (p *Program) check() error {
	for _, t := range p.ABI {
		if _, ok := ctypes[t]; !ok {
			return fmt.Errorf("iron: stream scalar type %q", t)
		}
	}
	reg := func(r int64) error {
		if r < 0 || r >= int64(p.Regs) {
			return fmt.Errorf("iron: stream register %d of %d", r, p.Regs)
		}
		return nil
	}
	for i, o := range p.Ops {
		regs := o.args
		switch o.code {
		case opK:
			regs = o.args[:1]
		case opArr:
			regs = o.args[:1]
		case opSet:
			regs = []int64{o.args[0], o.args[2]}
		case opChk:
			regs = o.args[:1]
			if o.args[1] < 0 || o.args[1] >= int64(len(p.Msgs)) {
				return fmt.Errorf("iron: stream operation %d: refusal %d", i, o.args[1])
			}
		case opHdr:
			regs = nil
		}
		for _, r := range regs {
			if err := reg(r); err != nil {
				return fmt.Errorf("operation %d: %w", i, err)
			}
		}
	}
	return nil
}

// RefusedError is a program's refusal of a call's scalars (the dispatch library returns -2).
type RefusedError struct{ Why string }

func (e *RefusedError) Error() string { return "iron: the stream refuses the call: " + e.Why }

func width(t ctype) uint {
	switch t {
	case tI8, tU8:
		return 8
	case tI16, tU16:
		return 16
	case tI32, tU32:
		return 32
	case tBool:
		return 1
	}
	return 64
}

func signed(t ctype) bool { return t == tI8 || t == tI16 || t == tI32 || t == tI64 }

// conv is v as a value of type t, held sign-extended (signed) or zero-extended (unsigned) in
// an int64's bits.
func conv(v int64, t ctype) int64 {
	if t == tBool {
		if v != 0 {
			return 1
		}
		return 0
	}
	w := width(t)
	if w == 64 {
		return v
	}
	if signed(t) {
		return v << (64 - w) >> (64 - w)
	}
	return int64(uint64(v) & (1<<w - 1))
}

func promote(t ctype) ctype {
	switch t {
	case tBool, tI8, tU8, tI16, tU16:
		return tI32
	}
	return t
}

// common is C's usual arithmetic conversion of two operand types.
func common(a, b ctype) ctype {
	a, b = promote(a), promote(b)
	if a == b {
		return a
	}
	rank := map[ctype]int{tI32: 0, tU32: 1, tI64: 2, tU64: 3}
	hi := a
	if rank[b] > rank[a] {
		hi = b
	}
	switch hi {
	case tU64:
		return tU64
	case tI64:
		return tI64
	}
	return tU32
}

func binop(op string, x int64, tx ctype, y int64, ty ctype) (int64, error) {
	b := func(c bool) int64 {
		if c {
			return 1
		}
		return 0
	}
	switch op {
	case "&&":
		return b(x != 0 && y != 0), nil
	case "||":
		return b(x != 0 || y != 0), nil
	case "<<", ">>":
		t := promote(tx)
		x = conv(x, t)
		if y < 0 || y >= 64 {
			return 0, nil // past the width: C leaves it undefined; the generated code guards it
		}
		if op == "<<" {
			return conv(x<<uint(y), t), nil
		}
		if signed(t) {
			return x >> uint(y), nil
		}
		return int64(uint64(x) >> uint(y)), nil
	}
	t := common(tx, ty)
	x, y = conv(x, t), conv(y, t)
	us := !signed(t)
	switch op {
	case "+":
		return conv(x+y, t), nil
	case "-":
		return conv(x-y, t), nil
	case "*":
		return conv(x*y, t), nil
	case "/", "%":
		if y == 0 {
			return 0, &RefusedError{"division by zero"}
		}
		if us {
			if op == "/" {
				return conv(int64(uint64(x)/uint64(y)), t), nil
			}
			return conv(int64(uint64(x)%uint64(y)), t), nil
		}
		if op == "/" {
			return conv(x/y, t), nil // Go truncates, as C
		}
		return conv(x%y, t), nil
	case "&":
		return conv(x&y, t), nil
	case "|":
		return conv(x|y, t), nil
	case "^":
		return conv(x^y, t), nil
	case "==":
		return b(x == y), nil
	case "!=":
		return b(x != y), nil
	}
	lt := func() bool {
		if us {
			return uint64(x) < uint64(y)
		}
		return x < y
	}
	eq := x == y
	switch op {
	case "<":
		return b(lt()), nil
	case "<=":
		return b(lt() || eq), nil
	case ">":
		return b(!lt() && !eq), nil
	case ">=":
		return b(!lt()), nil
	}
	return 0, fmt.Errorf("iron: stream operator %q", op)
}

// Run builds the stream for the scalars into dst (reset), and returns it.
func (p *Program) Run(dst []uint32, scalars []int64) ([]uint32, error) {
	if len(scalars) != len(p.ABI) {
		return nil, fmt.Errorf("iron: stream of %d scalars, given %d", len(p.ABI), len(scalars))
	}
	val := make([]int64, p.Regs)
	typ := make([]ctype, p.Regs)
	arrs := map[int64][]uint32{}
	for i, t := range p.ABI {
		typ[i] = ctypes[t]
		val[i] = conv(scalars[i], typ[i])
	}
	u32 := func(r int64) uint32 { return uint32(val[r]) }
	txn := append(dst[:0], 0, 0, 0, 0)
	for _, o := range p.Ops {
		a := o.args
		switch o.code {
		case opK:
			val[a[0]], typ[a[0]] = conv(a[1], o.typ), o.typ
		case opCast:
			val[a[0]], typ[a[0]] = conv(val[a[1]], o.typ), o.typ
		case opBin:
			v, err := binop(o.op, val[a[1]], typ[a[1]], val[a[2]], typ[a[2]])
			if err != nil {
				return nil, err
			}
			val[a[0]], typ[a[0]] = conv(v, o.typ), o.typ
		case opSel:
			v := val[a[3]]
			if val[a[1]] != 0 {
				v = val[a[2]]
			}
			val[a[0]], typ[a[0]] = conv(v, o.typ), o.typ
		case opArr:
			arrs[a[0]] = make([]uint32, a[1])
		case opSet:
			arr := arrs[a[0]]
			if a[1] < 0 || a[1] >= int64(len(arr)) {
				return nil, fmt.Errorf("iron: stream array index %d of %d", a[1], len(arr))
			}
			arr[a[1]] = u32(a[2])
		case opChk:
			if val[a[0]] == 0 {
				return nil, &RefusedError{p.Msgs[a[1]]}
			}
		case opW32:
			txn = append(txn, 0, 0, u32(a[0]), 0, u32(a[1]), 24)
		case opMW32, opMP32:
			c := uint32(3)
			if o.code == opMP32 {
				c = 4
			}
			txn = append(txn, c, 0, u32(a[0]), 0, u32(a[1]), u32(a[2]), 28)
		case opSync:
			col, row, d, ch, nc, nr := u32(a[0]), u32(a[1]), u32(a[2]), u32(a[3]), u32(a[4]), u32(a[5])
			txn = append(txn, 0x80, 16, d&0xff|(row&0xff)<<8|(col&0xff)<<16, (nr&0xff)<<8|(nc&0xff)<<16|(ch&0xff)<<24)
		case opBW:
			data := arrs[a[1]]
			count := uint64(val[a[2]])
			if count > uint64(len(data)) {
				return nil, fmt.Errorf("iron: stream block write of %d words from %d", count, len(data))
			}
			col, row := u32(a[3]), u32(a[4])
			txn = append(txn, 1, col&0xff|(row&0xff)<<8, u32(a[0]), uint32((4+count)*4))
			txn = append(txn, data[:count]...)
		case opPatch:
			idx := uint32(conv(val[a[1]], tI32))
			plus := uint64(val[a[2]])
			if val[a[3]] != 0 && idx >= 5 {
				plus += 0x80000000
			}
			txn = append(txn, 0x81, 48, 0, 0, 0, 0, u32(a[0]), 0, idx, 0, uint32(plus), uint32(plus>>32))
		case opPDI:
			addr := uint64(val[a[2]])
			txn = append(txn, 8|u32(a[0])<<16, u32(a[1]), uint32(addr), uint32(addr>>32))
		case opPre:
			txn = append(txn, 6|u32(a[0])<<8)
		case opHdr:
			ops, gen, rows, cols, mem := uint32(a[0]), uint32(a[1]), uint32(a[2]), uint32(a[3]), uint32(a[4])
			txn[0] = rows<<24 | gen<<16 | 1<<8
			txn[1] = mem<<8 | cols
			txn[2] = ops
			txn[3] = uint32(len(txn) * 4)
		}
	}
	return txn, nil
}
