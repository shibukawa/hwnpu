"""The per-call instruction streams of an xclbin per step (NPU1), as programs hwnpu runs.

On an xclbin a per-call value is a dispatch-time scalar: mlir-aie generates, per design, C++
that builds the instruction stream (a TXN) from the scalars (dispatch_gen.cpp, compiled to a
dispatch library). That code is straight-line: typed assignments, arithmetic, selects, guards
that refuse a call, and calls of aie_runtime's TXN encoders (aie/Runtime/TxnEncoding.h).
``parse`` turns it into a program of those operations, ``run`` runs one as hwnpu's Go
interpreter does (internal/llm/iron/stream.go), so that an export can hold each design's
program and check it against the dispatch library.

Program (JSON): {"abi": [C types of the scalars], "regs": N, "msgs": [refusals], "ops": [...]}
with registers numbered, the scalars first (0..), and each operation a list:

  ["k", dst, type, value]            dst = value
  ["cast", dst, type, src]           dst = (type) src
  ["bin", dst, type, op, a, b]       dst = a op b, in C's usual conversions, then (type)
  ["sel", dst, type, c, a, b]        dst = c ? a : b
  ["arr", dst, n]                    dst = uint32_t[n] of zeros
  ["set", arr, i, src]               arr[i] = (uint32_t) src
  ["chk", c, msg]                    refuse the call with msgs[msg] unless c
  ["w32", addr, val]                 txn_append_write32
  ["mw32", addr, val, mask]          txn_append_maskwrite32
  ["mp32", addr, val, mask]          txn_append_maskpoll32
  ["sync", col, row, dir, chan, ncol, nrow]
  ["bw", addr, arr, count, col, row] txn_append_blockwrite
  ["patch", addr, idx, plus, fold]   txn_append_arg_patch
  ["pdi", id, size, addr]            txn_append_loadpdi
  ["pre", level]                     txn_append_preempt
  ["hdr", ops, gen, rows, cols, memrows]  txn_prepend_header with txn_device_info

Types: i8 u8 i16 u16 i32 u32 i64 u64 b. Anything else in the C++ is refused, so that a newer
mlir-aie's output is caught at export rather than run wrong.
"""

from __future__ import annotations

import re

TYPES = {
    "int8_t": "i8", "uint8_t": "u8", "int16_t": "i16", "uint16_t": "u16",
    "int32_t": "i32", "uint32_t": "u32", "int64_t": "i64", "uint64_t": "u64",
    "bool": "b", "size_t": "u64",
}
WIDTH = {"i8": 8, "u8": 8, "i16": 16, "u16": 16, "i32": 32, "u32": 32, "i64": 64, "u64": 64, "b": 1}
BINOPS = {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "==", "!=", "<", "<=", ">", ">=", "&&", "||"}
CMPS = {"==", "!=", "<", "<=", ">", ">=", "&&", "||"}


class Refused(Exception):
    """The program refused the call (the dispatch library returns -2)."""


def conv(v: int, t: str) -> int:
    """v as a value of C type t."""
    if t == "b":
        return 1 if v else 0
    w = WIDTH[t]
    v &= (1 << w) - 1
    if t[0] == "i" and v >= 1 << (w - 1):
        v -= 1 << w
    return v


def common(a: str, b: str) -> str:
    """C's usual arithmetic conversions of two operand types (integer promotion first)."""
    def promote(t):
        return "i32" if t in ("b", "i8", "u8", "i16", "u16") else t
    a, b = promote(a), promote(b)
    rank = {"i32": 0, "u32": 1, "i64": 2, "u64": 3}
    if a == b:
        return a
    hi, lo = (a, b) if rank[a] > rank[b] else (b, a)
    if hi == "u64":
        return "u64"
    if hi == "i64":
        return "i64"  # lo is i32 or u32: both fit in int64
    return "u32"  # hi u32, lo i32


def binop(op: str, x: int, tx: str, y: int, ty: str) -> int:
    t = common(tx, ty)
    if op in ("&&", "||"):
        return int(bool(x) and bool(y)) if op == "&&" else int(bool(x) or bool(y))
    if op in ("<<", ">>"):
        t = common(tx, tx)  # the left operand's promoted type
        x = conv(x, t)
        if y < 0 or y >= 64:
            return 0  # past the width: C leaves it undefined; the generated code guards it
        if op == "<<":
            return conv(x << y, t)
        return x >> y  # arithmetic for signed (Python's >> floors), logical for unsigned
    x, y = conv(x, t), conv(y, t)
    if op == "+":
        return conv(x + y, t)
    if op == "-":
        return conv(x - y, t)
    if op == "*":
        return conv(x * y, t)
    if op in ("/", "%"):
        if y == 0:
            raise Refused("division by zero")
        q = abs(x) // abs(y) * (1 if (x >= 0) == (y >= 0) else -1)  # C truncates
        return conv(q, t) if op == "/" else conv(x - q * y, t)
    if op == "&":
        return conv(x & y, t)
    if op == "|":
        return conv(x | y, t)
    if op == "^":
        return conv(x ^ y, t)
    return int({"==": x == y, "!=": x != y, "<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y}[op])


_SIG = re.compile(r"generate_txn_\w+\((.*?)\)\s*\{")
_DECL = re.compile(r"^(\w+) v(\d+) = (.+);$")
_ARR = re.compile(r"^uint32_t v(\d+)\[(\d+)\] = \{(.*)\};$")
_SET = re.compile(r"^v(\d+)\[(\d+)\] = v(\d+);$")
_CHK = re.compile(r'^if \(!\(v(\d+)\)\) return aie_runtime::txn_refused\("(.*)"\);$')
_CALL = re.compile(r"^aie_runtime::txn_(\w+)\(txn, (.*)\);$")
_LIT = re.compile(r"^(-?(?:0x[0-9a-fA-F]+|\d+))[uU]?(?:[lL]{1,2})?$")
_CAST = re.compile(r"^\((\w+)\) v(\d+)$")
_SEL = re.compile(r"^v(\d+) \? (v\d+(?: \S+ v\d+)?) : (v\d+(?: \S+ v\d+)?)$")
_BIN = re.compile(r"^v(\d+) (\S+) v(\d+)$")
_HDR = re.compile(r"^(\d+)u?, aie_runtime::txn_device_info\((\d+), (\d+), (\d+), (\d+)\)$")
_SKIP = {"std::vector<uint32_t> txn;", "aie_runtime::txn_init(txn);", "return std::move(txn);", "return txn;"}
_CALLS = {"append_write32": ("w32", 2), "append_maskwrite32": ("mw32", 3), "append_maskpoll32": ("mp32", 3),
          "append_sync": ("sync", 6), "append_blockwrite": ("bw", 5), "append_arg_patch": ("patch", 4),
          "append_loadpdi": ("pdi", 3), "append_preempt": ("pre", 1)}


def parse(cpp: str) -> dict:
    """The program of a dispatch_gen.cpp; ValueError on a line it does not know."""
    m = _SIG.search(cpp)
    if m is None:
        raise ValueError("no generate_txn_ function")
    body = cpp[m.end():]
    end = body.find('\n}\nextern "C"')
    if end < 0:
        raise ValueError("no end of the generate_txn_ function")
    body = body[:end]
    abi, reg = [], {}
    for p in [x.strip() for x in m.group(1).split(",") if x.strip()]:
        t, name = p.rsplit(" ", 1)
        if t not in TYPES or not name.startswith("v"):
            raise ValueError(f"scalar {p!r}")
        reg[int(name[1:])] = (len(abi), TYPES[t])
        abi.append(t)
    ops, msgs = [], []
    nregs = len(abi)

    def new(t):
        nonlocal nregs
        nregs += 1
        return nregs - 1

    def ref(v):
        if v not in reg:
            raise ValueError(f"v{v} used before it is set")
        return reg[v][0]

    def arg(s):
        """A call's argument: a register, or a literal put in a new register."""
        s = s.strip()
        if s.startswith("v") and s[1:].isdigit():
            return ref(int(s[1:]))
        if s in ("true", "false"):
            r = new("b")
            ops.append(["k", r, "b", 1 if s == "true" else 0])
            return r
        lm = _LIT.match(s)
        if lm is None:
            raise ValueError(f"call argument {s!r}")
        r = new("u64")
        ops.append(["k", r, "i64" if not s.lower().endswith("u") and int(lm.group(1), 0) < 0 else "u64", int(lm.group(1), 0)])
        return r

    for raw in body.split("\n"):
        line = raw.strip()
        if not line or line in _SKIP:
            continue
        if (mm := _ARR.match(line)) is not None:
            r = new("arr")
            reg[int(mm.group(1))] = (r, "arr")
            ops.append(["arr", r, int(mm.group(2))])
            init = [x.strip() for x in mm.group(3).split(",") if x.strip()]
            if len(init) > int(mm.group(2)):
                raise ValueError(f"array initializer {line!r}")
            for i, x in enumerate(init):
                lm = _LIT.match(x)
                if lm is None:
                    raise ValueError(f"array initializer {x!r}")
                k = new("u32")
                ops.append(["k", k, "u32", conv(int(lm.group(1), 0), "u32")])
                ops.append(["set", r, i, k])
        elif (mm := _SET.match(line)) is not None:
            ops.append(["set", ref(int(mm.group(1))), int(mm.group(2)), ref(int(mm.group(3)))])
        elif (mm := _CHK.match(line)) is not None:
            msgs.append(mm.group(2))
            ops.append(["chk", ref(int(mm.group(1))), len(msgs) - 1])
        elif (mm := _CALL.match(line)) is not None:
            fn, args = mm.group(1), mm.group(2)
            if fn == "prepend_header":
                hm = _HDR.match(args)
                if hm is None:
                    raise ValueError(f"header {args!r}")
                ops.append(["hdr", *(int(hm.group(i)) for i in range(1, 6))])
                continue
            if fn not in _CALLS:
                raise ValueError(f"TXN encoder {fn}")
            code, n = _CALLS[fn]
            parts = [p.strip() for p in args.split(",")]
            if len(parts) != n:
                raise ValueError(f"{fn} with {len(parts)} arguments")
            ops.append([code, *(arg(p) for p in parts)])
        elif (mm := _DECL.match(line)) is not None:
            ctype, v, rhs = mm.group(1), int(mm.group(2)), mm.group(3).strip()
            if ctype not in TYPES:
                raise ValueError(f"type {ctype} in {line!r}")
            t = TYPES[ctype]
            r = new(t)
            if rhs in ("true", "false"):
                ops.append(["k", r, t, 1 if rhs == "true" else 0])
            elif (lm := _LIT.match(rhs)) is not None:
                ops.append(["k", r, t, conv(int(lm.group(1), 0), t)])
            elif (cm := _CAST.match(rhs)) is not None:
                if cm.group(1) not in TYPES:
                    raise ValueError(f"cast to {cm.group(1)}")
                ops.append(["cast", r, TYPES[cm.group(1)], ref(int(cm.group(2)))])
                if TYPES[cm.group(1)] != t:
                    r2 = new(t)
                    ops.append(["cast", r2, t, r])
                    r = r2
            elif (sm := _SEL.match(rhs)) is not None:
                # a branch is a register, or a binary operation (a shift guarded against its
                # width: both branches are computed, so a shift past it gives 0, not an error)
                def branch(x):
                    x = x.strip()
                    if (vm := re.match(r"^v(\d+)$", x)) is not None:
                        return ref(int(vm.group(1)))
                    bm = _BIN.match(x)
                    if bm is None or bm.group(2) not in BINOPS:
                        raise ValueError(f"select branch {x!r}")
                    b = new(t)
                    ops.append(["bin", b, t, bm.group(2), ref(int(bm.group(1))), ref(int(bm.group(3)))])
                    return b
                ops.append(["sel", r, t, ref(int(sm.group(1))), branch(sm.group(2)), branch(sm.group(3))])
            elif (bm := _BIN.match(rhs)) is not None and bm.group(2) in BINOPS:
                ops.append(["bin", r, t, bm.group(2), ref(int(bm.group(1))), ref(int(bm.group(3)))])
            else:
                raise ValueError(f"expression {rhs!r}")
            reg[v] = (r, t)
        else:
            raise ValueError(f"line {line!r}")
    return dict(abi=abi, regs=nregs, msgs=msgs, ops=ops)


def run(prog: dict, scalars: list[int]) -> list[int]:
    """The stream the program builds for the scalars; Refused when it refuses them."""
    n = prog["regs"]
    val, typ = [0] * n, ["i64"] * n
    arrs: dict[int, list[int]] = {}
    for i, t in enumerate(prog["abi"]):
        typ[i] = TYPES[t]
        val[i] = conv(int(scalars[i]), typ[i])
    txn = [0, 0, 0, 0]
    u32 = lambda r: conv(val[r], "u32")  # noqa: E731
    for op in prog["ops"]:
        c = op[0]
        if c == "k":
            val[op[1]], typ[op[1]] = conv(op[3], op[2]), op[2]
        elif c == "cast":
            val[op[1]], typ[op[1]] = conv(val[op[3]], op[2]), op[2]
        elif c == "bin":
            _, d, t, o, a, b = op
            val[d], typ[d] = conv(binop(o, val[a], typ[a], val[b], typ[b]), t), t
        elif c == "sel":
            _, d, t, cc, a, b = op
            val[d], typ[d] = conv(val[a] if val[cc] else val[b], t), t
        elif c == "arr":
            arrs[op[1]] = [0] * op[2]
        elif c == "set":
            arrs[op[1]][op[2]] = u32(op[3])
        elif c == "chk":
            if not val[op[1]]:
                raise Refused(prog["msgs"][op[2]])
        elif c == "w32":
            txn += [0, 0, u32(op[1]), 0, u32(op[2]), 24]
        elif c in ("mw32", "mp32"):
            txn += [3 if c == "mw32" else 4, 0, u32(op[1]), 0, u32(op[2]), u32(op[3]), 28]
        elif c == "sync":
            col, row, d, ch, nc, nr = (u32(r) for r in op[1:])
            txn += [0x80, 16, (d & 0xFF) | ((row & 0xFF) << 8) | ((col & 0xFF) << 16),
                    ((nr & 0xFF) << 8) | ((nc & 0xFF) << 16) | ((ch & 0xFF) << 24)]
        elif c == "bw":
            data, count = arrs[op[2]], conv(val[op[3]], "u64")
            col, row = u32(op[4]), u32(op[5])
            txn += [1, (col & 0xFF) | ((row & 0xFF) << 8), u32(op[1]), conv((4 + count) * 4, "u32")] + data[:count]
        elif c == "patch":
            addr, idx, plus, fold = u32(op[1]), conv(val[op[2]], "i32"), conv(val[op[3]], "u64"), val[op[4]]
            if fold and conv(idx, "u32") >= 5:
                plus = conv(plus + 0x80000000, "u64")
            txn += [0x81, 48, 0, 0, 0, 0, addr, 0, conv(idx, "u32"), 0, plus & 0xFFFFFFFF, plus >> 32]
        elif c == "pdi":
            pid, size, addr = u32(op[1]), u32(op[2]), conv(val[op[3]], "u64")
            txn += [8 | (pid << 16), size, addr & 0xFFFFFFFF, addr >> 32]
        elif c == "pre":
            txn += [6 | (u32(op[1]) << 8)]
        elif c == "hdr":
            _, ops_, gen, rows, cols, mem = op
            txn[0] = (rows << 24) | (gen << 16) | (1 << 8) | 0
            txn[1] = (mem << 8) | cols
            txn[2] = ops_
            txn[3] = len(txn) * 4
        else:
            raise ValueError(f"operation {c}")
    return txn
