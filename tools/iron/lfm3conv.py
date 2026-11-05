"""lfm3conv.py build|data|check OUT [MODEL_DIR LAYER]: one decode step of an LFM3 convolution layer in one NPU call.

The layer (LFM3-1.7B: hidden 2048, intermediate 7680, convolution width 7) runs on 8 columns x 2
cores. Core k owns 128 channels: its rows of in_proj (B, C and X of those channels), out_proj,
w1 and w3 (480 each) and w2, all in q4nx. Each core keeps the full residual vector and runs the
norms on full vectors, so the only data that moves between cores are the four vectors that feed
a weight product, gathered in DDR and broadcast back within the one runtime sequence:

  A: x, operator_norm -> in_proj -> y = C * conv(B * X) and the new convolution window
  B: y -> out_proj -> o
  C: o, post_operator_norm, ffn_norm -> x1 = x + norm(o); w1, w3 -> g = silu(w1 h) * w3 h
  D: g -> w2 -> f
  E: f, post_ffn_norm -> x1 + norm(f), each core its channels

Kernel arguments: x (f32), the weights (q4nx blocks and a block of convolution taps per core),
the convolution window (a block per core), the four norm weights (bf16) and a scratch buffer
that holds every stage's outputs (the layer's output in the last region).
"""
import json
import os
import struct
import sys

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern as TAP
from aie.iron import Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.runtime import TaskGroup
from ml_dtypes import bfloat16

HERE = os.path.dirname(os.path.abspath(__file__))
D, I, KC = 2048, 7680, 7
COLS, ROWS = 8, 2
NC = COLS * ROWS
CH, FI = D // NC, I // NC
BLK_W = 1280  # words in a 5120 byte block
CHUNK_W = 256  # words in a 1024 byte chunk
PIECE_W = 16  # words in a 64 byte output piece
# stage outputs per core, in words: y and the window, o, g, f, the output
OUT_W = {"A": (2 * CH + 2 * (KC - 1) * CH) // 4, "B": CH, "C": FI // 2, "D": CH, "E": CH}
SCR = {}
_off = 0
for _s in "ABCDE":
    SCR[_s] = _off
    _off += NC * OUT_W[_s]
SCR_W = _off
# q4nx blocks per core and stage: (row blocks, column blocks)
STAGES = {"A": (3 * CH // 32, D // 256), "B": (CH // 32, D // 256), "C": (2 * FI // 32, D // 256), "D": (CH // 32, I // 256)}
BLOCKS = sum(r * c for r, c in STAGES.values())
COL_W = (2 + 2 * BLOCKS) * BLK_W  # a column's weight stream: the parameter pair, then the stages


@iron.jit
def lfm3conv(x: In, w: In, state: In, norms: In, scr: Out, *, eps: CompileTime[float]):
    i32 = np.dtype[np.int32]
    blk_ty = np.ndarray[(BLK_W,), i32]
    chunk_ty = np.ndarray[(CHUNK_W,), i32]
    piece_ty = np.ndarray[(PIECE_W,), i32]
    ak = os.path.join(os.environ["MLIR_AIE_SRC"], "aie_kernels")
    r_ty = np.ndarray[(D,), np.dtype[np.float32]]
    xb_ty = np.ndarray[(I,), np.dtype[bfloat16]]
    n_ty1 = np.ndarray[(D,), np.dtype[bfloat16]]
    mv = ExternalFunction("lyr_mv", source_file=os.path.join(HERE, "lfm3conv.cc"), arg_types=[blk_ty, np.int32, np.int32, xb_ty],
                          include_dirs=[ak], compile_flags=[f"-DLYR_EPS={eps}f", f"-DLYR_D={D}", f"-DLYR_I={I}", f"-DLYR_CORES={NC}"])
    bind = mv.object_file.bind
    k_ld = bind("lyr_ld", [chunk_ty, np.int32, np.int32, r_ty, n_ty1, xb_ty])
    k_ld_blk = bind("lyr_ld_blk", [blk_ty, np.int32])
    k_rms_x = bind("lyr_rms_x", [r_ty, n_ty1, xb_ty])
    k_resid = bind("lyr_resid", [r_ty, n_ty1, xb_ty])
    k_zero = bind("lyr_zero", [np.int32])
    k_conv = bind("lyr_conv", [])
    k_glu = bind("lyr_glu", [])
    k_out = bind("lyr_out", [np.int32, r_ty])
    k_st = bind("lyr_st", [piece_ty, np.int32])

    w_l3, w_l1, x_l3, x_l1, outs = [], [], [], [], []
    for c in range(COLS):
        wf = ObjectFifo(np.ndarray[(ROWS * BLK_W,), i32], name=f"w{c}")
        w_l3.append(wf)
        w_l1.append(wf.cons().split([BLK_W * r for r in range(ROWS)], obj_types=[blk_ty] * ROWS, names=[f"w{c}_{r}" for r in range(ROWS)]))
        xf = ObjectFifo(chunk_ty, name=f"x{c}")
        x_l3.append(xf)
        x_l1.append(xf.cons().forward(obj_type=chunk_ty, name=f"xl{c}"))
        for r in range(ROWS):
            outs.append(ObjectFifo(piece_ty, name=f"o{c}_{r}"))

    def make_core(k):
        def core(fw, fx, fo, R, X, N, mv, ld, ld_blk, rms_x, resid, zero, conv, glu, out, st):
            def chunks(n, dst):
                for ch in range_(n):
                    c = fx.acquire(1)
                    ld(c, ch, dst, R, N, X)
                    fx.release(1)

            def product(rows, cols):
                zero(rows * 32)
                for rb in range_(rows):
                    for cb in range_(cols):
                        b = fw.acquire(1)
                        mv(b, cb, rb, X)
                        fw.release(1)

            def send(n):
                for i in range_(n):
                    p = fo.acquire(1)
                    st(p, i)
                    fo.release(1)

            for which in range_(2):  # the taps, then the window
                b = fw.acquire(1)
                ld_blk(b, which)
                fw.release(1)
            # A: x and operator_norm; in_proj; the gated convolution
            chunks(D * 4 // 1024, 0)
            chunks(D * 2 // 1024, 1)
            rms_x(R, N, X)
            product(*STAGES["A"])
            conv()
            send(OUT_W["A"] // PIECE_W)
            # B: y; out_proj
            chunks(D * 2 // 1024, 2)
            product(*STAGES["B"])
            out(-1, R)
            send(OUT_W["B"] // PIECE_W)
            # C: o, post_operator_norm, ffn_norm; w1 and w3; silu(a) * b
            chunks(D * 4 // 1024, 2)
            chunks(D * 2 // 1024, 1)
            resid(R, N, X)
            chunks(D * 2 // 1024, 1)
            rms_x(R, N, X)
            product(*STAGES["C"])
            glu()
            send(OUT_W["C"] // PIECE_W)
            # D: g; w2
            chunks(I * 2 // 1024, 2)
            product(*STAGES["D"])
            out(-1, R)
            send(OUT_W["D"] // PIECE_W)
            # E: f, post_ffn_norm; the core's channels of x1 + norm(f)
            chunks(D * 4 // 1024, 2)
            chunks(D * 2 // 1024, 1)
            resid(R, N, X)
            out(k, R)
            send(OUT_W["E"] // PIECE_W)

        return core

    kernels = [mv, k_ld, k_ld_blk, k_rms_x, k_resid, k_zero, k_conv, k_glu, k_out, k_st]
    workers = []
    for c in range(COLS):
        for r in range(ROWS):
            k = c * ROWS + r
            bufs = [Buffer(r_ty, name=f"R{k}"), Buffer(xb_ty, name=f"X{k}"), Buffer(n_ty1, name=f"N{k}")]
            workers.append(Worker(make_core(k), [w_l1[c][r].cons(), x_l1[c].cons(), outs[k].prod(), *bufs, *kernels]))

    wp = [f.prod() for f in w_l3]
    xp = [f.prod() for f in x_l3]
    oc = [f.cons() for f in outs]
    x_ty = np.ndarray[(D,), np.dtype[np.float32]]
    w_ty = np.ndarray[(COLS * COL_W,), i32]
    s_ty = np.ndarray[(COLS * ROWS * BLK_W,), i32]
    n_ty = np.ndarray[(4 * D // 2,), i32]
    scr_ty = np.ndarray[(SCR_W,), i32]

    def flat(n, off, size):
        return TAP((n,), off, [size], [1])

    def sequence(x, w, state, norms, scr, wh, xh, oh):
        tg = TaskGroup()
        for c in range(COLS):
            wh[c].fill(w, flat(COLS * COL_W, c * COL_W, 2 * BLK_W), group=tg)
            wh[c].fill(state, flat(COLS * ROWS * BLK_W, c * ROWS * BLK_W, ROWS * BLK_W), group=tg)
            wh[c].fill(w, flat(COLS * COL_W, c * COL_W + 2 * BLK_W, COL_W - 2 * BLK_W), group=tg)

        def norm(i):  # the i-th norm weight, D bf16
            return flat(4 * D // 2, i * D // 2, D // 2)

        def stage(s, *inputs):
            # the stage's broadcasts are consumed once its outputs are back: free their BDs then
            td = TaskGroup()
            for c in range(COLS):
                for buf, tap in inputs:
                    xh[c].fill(buf, tap, group=td)
            for k in range(NC):
                oh[k].drain(scr, flat(SCR_W, SCR[s] + k * OUT_W[s], OUT_W[s]), wait=True, group=td)
            td.finish()

        stage("A", (x, TAP((D,), 0, [D], [1])), (norms, norm(0)))
        # y: the first 2*CH bytes of each core's A output
        stage("B", (scr, TAP((SCR_W,), SCR["A"], [NC, CH // 2], [OUT_W["A"], 1])))
        stage("C", (scr, flat(SCR_W, SCR["B"], NC * OUT_W["B"])), (norms, norm(1)), (norms, norm(2)))
        stage("D", (scr, flat(SCR_W, SCR["C"], NC * OUT_W["C"])))
        stage("E", (scr, flat(SCR_W, SCR["D"], NC * OUT_W["D"])), (norms, norm(3)))
        tg.finish()

    rt = Runtime(sequence, [x_ty, w_ty, s_ty, n_ty, scr_ty, wp, xp, oc])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# --- host side -------------------------------------------------------------------------------

def load_safetensors(path, names):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        out = {}
        for name in names:
            e = hdr[name]
            a, b = e["data_offsets"]
            f.seek(8 + n + a)
            raw = np.frombuffer(f.read(b - a), dtype=np.uint16)
            assert e["dtype"] == "BF16", e["dtype"]
            out[name] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])
    return out


def quantize(w):
    """q4nx with each group's whole range: codes (M, K) uint8, scales and minima (M, K/32) bf16."""
    M, K = w.shape
    g = w.reshape(M, K // 32, 32)
    lo, hi = g.min(axis=2), g.max(axis=2)
    s = ((hi - lo) / 15).astype(bfloat16)
    s = np.where(s.astype(np.float32) == 0, np.abs(lo).astype(bfloat16), s).astype(bfloat16)
    m = lo.astype(bfloat16)
    sf, mf = s.astype(np.float32)[:, :, None], m.astype(np.float32)[:, :, None]
    q = np.clip(np.rint((g - mf) / np.where(sf == 0, 1, sf)), 0, 15).astype(np.uint8).reshape(M, K)
    return q, s, m


def blocks(q, s, m):
    """The kernel's 32 x 256 blocks of a (rows, K) q4nx matrix, row blocks outer (palette.Q4NX.Blocks)."""
    M, K = q.shape
    out = []
    for r0 in range(0, M, 32):
        for c0 in range(0, K, 256):
            sc = s[r0:r0 + 32, c0 // 32:(c0 + 256) // 32].T.ravel()
            mn = m[r0:r0 + 32, c0 // 32:(c0 + 256) // 32].T.ravel()
            codes = q[r0:r0 + 32, c0:c0 + 256].reshape(2, 16, 256).transpose(0, 2, 1).ravel()
            out.append(np.concatenate([sc.view(np.uint8), mn.view(np.uint8), (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)]))
    return out


def pad_block(b):
    out = np.zeros(BLK_W * 4, np.uint8)
    out[:len(b)] = b
    return out


def layer_weights(model, layer):
    p = f"model.layers.{layer}."
    names = [p + n for n in ("conv.in_proj.weight", "conv.out_proj.weight", "conv.conv.weight", "feed_forward.w1.weight", "feed_forward.w2.weight",
                             "feed_forward.w3.weight", "operator_norm.weight", "post_operator_norm.weight", "ffn_norm.weight", "post_ffn_norm.weight")]
    t = load_safetensors(os.path.join(model, "model.safetensors"), names)
    return {n[len(p):]: v for n, v in t.items()}


def core_rows(k):
    """Rows of in_proj, out_proj, w1|w3 (stacked) and w2 that core k computes."""
    ch = np.arange(k * CH, (k + 1) * CH)
    fi = np.arange(k * FI, (k + 1) * FI)
    return np.concatenate([ch, D + ch, 2 * D + ch]), ch, np.concatenate([fi, I + fi]), ch


def bf16r(a, floor=False):
    b = np.asarray(a, np.float32).view(np.uint32).astype(np.uint64)
    if floor:
        b = b + (((b >> np.uint64(31)) != 0) & ((b & np.uint64(0xFFFF)) != 0)).astype(np.uint64) * np.uint64(0x10000)
    else:
        b = b + np.uint64(0x7FFF) + ((b >> np.uint64(16)) & np.uint64(1))
    return (b & np.uint64(0xFFFF0000)).astype(np.uint32).view(np.float32)


def gemv_kernel(qsm, x):
    """y as the NPU's q4nx_accumulate computes it (a bf16 partial sum per 32 columns, tools/iron/q4gemv.py)."""
    sys.path.insert(0, HERE)
    from q4gemv import kernel_model

    q, s, m = qsm
    return kernel_model(q, s, m, x.astype(bfloat16), False).astype(np.float32)  # lyr_ld_conv sets nearest


def gemv_exact(qsm, x):
    q, s, m = qsm
    W = np.repeat(s.astype(np.float64), 32, axis=1) * q + np.repeat(m.astype(np.float64), 32, axis=1)
    return (W @ x.astype(np.float64)).astype(np.float32)


def rms(v):
    v = v.astype(np.float64)
    return 1 / np.sqrt(np.mean(v * v) + 1e-5)


def reference(W, Q, x, win, kernel):
    """The layer's decode step. kernel: the NPU's arithmetic (bf16 where the cores round, the
    q4nx kernel's products); otherwise float32 with the q4nx weights. Returns (out, new window)."""
    mv = gemv_kernel if kernel else gemv_exact
    r = (lambda v: bf16r(v)) if kernel else (lambda v: v.astype(np.float32))
    h = r(x * rms(x) * W["operator_norm.weight"])
    bcx = mv(Q["in_proj"], h)
    B, C, X = bcx[:D], bcx[D:2 * D], bcx[2 * D:]
    bx = r(B * X)
    taps = W["conv.conv.weight"].reshape(D, KC)
    if kernel:
        taps = bf16r(taps)
    conv = (win * taps[:, :KC - 1].T).sum(axis=0) + bx * taps[:, KC - 1]
    y = r(C * conv)
    o = mv(Q["out_proj"], y)
    x1 = x + o * rms(o) * W["post_operator_norm.weight"]
    h2 = r(x1 * rms(x1) * W["ffn_norm.weight"])
    a, b = mv(Q["w1"], h2), mv(Q["w3"], h2)
    silu = a / (1 + np.exp(-a.astype(np.float64)))
    g = r(silu * b)
    f = mv(Q["w2"], g)
    out = x1 + f * rms(f) * W["post_ffn_norm.weight"]
    return out.astype(np.float32), np.concatenate([win[1:], bx[None]], axis=0)


def prepare(model, layer):
    W = layer_weights(model, layer)
    Q = {"in_proj": quantize(W["conv.in_proj.weight"]), "out_proj": quantize(W["conv.out_proj.weight"]),
         "w1": quantize(W["feed_forward.w1.weight"]), "w3": quantize(W["feed_forward.w3.weight"]), "w2": quantize(W["feed_forward.w2.weight"])}
    return W, Q


def inputs(model, ref):
    """x entering layer 0 for the last token of the reference, and the window of the 6 before."""
    W, Q = prepare(model, 0)
    meta = json.load(open(ref + ".json"))
    hid = np.fromfile(ref + ".hidden.bin", dtype="<f4").reshape(-1, meta["seq"], D)
    emb = hid[0]
    win = []
    for t in range(meta["seq"] - KC, meta["seq"] - 1):
        h = (emb[t] * rms(emb[t]) * W["operator_norm.weight"]).astype(np.float32)
        bcx = gemv_exact(Q["in_proj"], h)
        win.append(bf16r(bcx[:D] * bcx[2 * D:]))
    return W, Q, emb[-1].astype(np.float32), np.array(win, np.float32)


def main():
    cmd, out = sys.argv[1], sys.argv[2]
    if cmd == "build":
        from ironcompile import compile

        compile(lfm3conv, out, eps=1e-5)
        return
    if cmd == "data":
        model, ref = sys.argv[3], sys.argv[4]
        W, Q, x, win = inputs(model, ref)
        cols = []
        for c in range(COLS):
            parts = []
            per_core = []
            for r in range(ROWS):
                k = c * ROWS + r
                taps = W["conv.conv.weight"].reshape(D, KC)[k * CH:(k + 1) * CH].T.astype(bfloat16)  # [tap][channel]
                parts.append(pad_block(np.ascontiguousarray(taps).view(np.uint8).ravel()))
                ra, rb, rc, rd = core_rows(k)
                st = []
                for name, rows in (("in_proj", ra), ("out_proj", rb), ("w13", rc), ("w2", rd)):
                    if name == "w13":
                        q = np.concatenate([Q["w1"][0], Q["w3"][0]])[rows]
                        s = np.concatenate([Q["w1"][1], Q["w3"][1]])[rows]
                        m = np.concatenate([Q["w1"][2], Q["w3"][2]])[rows]
                    else:
                        q, s, m = (a[rows] for a in Q[name])
                    st.extend(blocks(q, s, m))
                per_core.append(st)
            for i in range(len(per_core[0])):  # interleave the two cores' blocks
                for r in range(ROWS):
                    parts.append(per_core[r][i])
            cols.append(np.concatenate(parts))
        np.concatenate(cols).tofile(os.path.join(out, "w.bin"))
        state = [pad_block(win[:, k * CH:(k + 1) * CH].astype(bfloat16).view(np.uint8).ravel()) for k in range(NC)]
        np.concatenate(state).tofile(os.path.join(out, "state.bin"))
        norms = np.concatenate([W[n + ".weight"] for n in ("operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm")]).astype(bfloat16)
        norms.tofile(os.path.join(out, "norms.bin"))
        x.tofile(os.path.join(out, "x.bin"))
        np.savez(os.path.join(out, "ref_inputs.npz"), x=x, win=win)
        print(SCR_W * 4)
        return
    model = sys.argv[3]
    W, Q = prepare(model, 0)
    d = np.load(os.path.join(out, "ref_inputs.npz"))
    x, win = d["x"], d["win"]
    scr = np.fromfile(os.path.join(out, "scr.bin"), dtype=np.uint8)
    got = scr.view(np.float32)[SCR["E"]:SCR["E"] + D]
    a = scr.view(np.int32)[SCR["A"]:SCR["A"] + NC * OUT_W["A"]].reshape(NC, OUT_W["A"]).view(np.uint8)
    newwin = np.concatenate([a[k, 2 * CH:].view(bfloat16).astype(np.float32).reshape(KC - 1, CH) for k in range(NC)], axis=1)
    want_k, win_k = reference(W, Q, x, win, True)
    want_f, _ = reference(W, Q, x, win, False)
    W16 = {"in_proj": W["conv.in_proj.weight"], "out_proj": W["conv.out_proj.weight"], "w1": W["feed_forward.w1.weight"], "w3": W["feed_forward.w3.weight"], "w2": W["feed_forward.w2.weight"]}

    def rel(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(b.astype(np.float64) ** 2)))

    # the bf16 weights of the checkpoint, float32 arithmetic: the quantization's own error
    Qbf = {n: (None,) for n in W16}
    mv_bf = lambda name, v: (W16[name].astype(np.float64) @ v.astype(np.float64)).astype(np.float32)
    h = x * rms(x) * W["operator_norm.weight"]
    bcx = mv_bf("in_proj", h)
    bx = bcx[:D] * bcx[2 * D:]
    taps = W["conv.conv.weight"].reshape(D, KC)
    y = bcx[D:2 * D] * ((win * taps[:, :KC - 1].T).sum(axis=0) + bx * taps[:, KC - 1])
    o = mv_bf("out_proj", y)
    x1 = x + o * rms(o) * W["post_operator_norm.weight"]
    h2 = x1 * rms(x1) * W["ffn_norm.weight"]
    av, bv = mv_bf("w1", h2), mv_bf("w3", h2)
    f = mv_bf("w2", (av / (1 + np.exp(-av.astype(np.float64)))) * bv)
    want_bf = (x1 + f * rms(f) * W["post_ffn_norm.weight"]).astype(np.float32)
    res = dict(rel_vs_npu_model=rel(got, want_k), rel_vs_q4nx_float=rel(got, want_f), rel_vs_bf16_weights=rel(got, want_bf),
               q4nx_float_vs_bf16_weights=rel(want_f, want_bf), window_rel=rel(newwin, win_k), finite=bool(np.isfinite(got).all()))
    res["ok"] = res["finite"] and res["rel_vs_npu_model"] < 0.02 and res["window_rel"] < 0.02
    print(json.dumps(res))


if __name__ == "__main__":
    main()
