"""A mixture of experts whose routing runs on the device (Qwen3.6-35B-A3B's), for IRON graphs.

A decode step's MoE layer is two commands (an empty host step between them, as
iron_moeprobe's layers):

  A  ... the layer's mixer, its residual and post-attention norm, then the router: one GEMV of
     the router's rows and the shared expert's gate row, and MoERoute (kernels/moe_route.cc):
     the 8 largest logits, their weights (softmax over the chosen), sigmoid of the shared
     gate, and the words of command B's scratchpad: each chosen expert's byte offset in the
     expert tables, as B's operators read them (``e0`` .. ``e7``, times an expert's bytes)
  B  iron_moeprobe's MoEGateUp and MoEDown (the 8 experts and the shared one, read in place)

MoERoute's ``image`` is a buffer of the version's feedback argument (``Feedback``): hwnpu puts
command B's ctrl scratchpad there, so B reads the offsets A wrote, and the host neither reads
the logits nor writes the experts. A word of the image is ``(id * scale + bias) << shift`` by a
program (a weight, four int32 a word), written once the lowering has placed B's words
(``program_rows``).

On the host (IRON's reference of the graph), MoERoute's reference keeps the experts it chose in
iron_moeprobe.PICKED by its ``picked`` key, which MoEGateUp and MoEDown read in place of their
per-call offsets.
"""

import functools
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))

import iron_moeprobe  # noqa: E402

SLOTS = 32  # a command's scratchpad words (IRON's SCRATCHPAD_WORDS): the image's words
WV = iron_moeprobe.WV  # routing weights MoEGateUp reads: the 8 chosen, the shared expert's, zeros


def route_reference(logits, experts: int, top: int):
    """The chosen experts (by descending logit, ties to the lower index) and the routing weights."""
    x = np.asarray(logits, np.float32).reshape(-1)
    ids = np.argsort(-x[:experts], kind="stable")[:top]
    v = x[ids]
    e = np.exp(v - v[0])
    w = np.zeros(WV, np.float32)
    w[:top] = e / e.sum()
    w[top] = 1 / (1 + np.exp(-x[experts]))
    return ids.astype(np.int32), w.astype(bfloat16)


@functools.cache
def route_operator():
    from aie.iron import In, ObjectFifo, Out, TaskGroup, Worker
    from aie.iron.kernel import ExternalFunction
    from aie.utils import config
    from aie.utils.compile.jit.params import param
    from aie.utils.verify import Tolerance

    from iron.common import Built, Operator

    class MoERoute(Operator):
        """``w, image = route(logits, program)``: one token's top-``top`` of ``experts`` routed
        experts (logit ``experts`` is the shared expert's gate) and the scratchpad image of
        their offsets (module docstring)."""

        n: int = param(array=True)  # the logits row, the shared gate at index experts
        experts: int = param(default=256, array=True)
        top: int = param(default=8, array=True)
        slots: int = param(default=SLOTS, array=True)
        words: int = param(derive=lambda op: 4 * op.slots, repr=False)
        wv: int = param(default=WV, init=False, repr=False)
        picked: int = param(default=-1, repr=False)  # PICKED's key on the host

        logits = In(n, obj_shape=(n,), depth=1)
        program = In(words, dtype=np.int32, obj_shape=(words,), depth=1)
        w = Out(wv, obj_shape=(wv,), depth=1)
        image = Out(slots, dtype=np.int32, obj_shape=(slots,), depth=1)

        def build(self) -> Built:
            kdir = Path(config.aie_kernels_dir())
            here = Path(__file__).resolve().parent / "kernels"
            kernel = ExternalFunction(
                "moe_route",
                source_file=str(here / "moe_route.cc"),
                include_dirs=[str(kdir), config.cxx_header_path()],
                compile_flags=[f"-DROUTE_EXPERTS={self.experts}", f"-DROUTE_TOP={self.top}",
                               f"-DROUTE_SLOTS={self.slots}", f"-DROUTE_W={self.wv}"],
                arg_types=[self.logits.obj_type, self.program.obj_type, self.w.obj_type, self.image.obj_type],
            )
            fl, fp = ObjectFifo(self.logits.obj_type, depth=1), ObjectFifo(self.program.obj_type, depth=1)
            fw, fi = ObjectFifo(self.w.obj_type, depth=1), ObjectFifo(self.image.obj_type, depth=1)

            def core(fl, fp, fw, fi, k):
                lg, pr, w, im = fl.acquire(1), fp.acquire(1), fw.acquire(1), fi.acquire(1)
                k(lg, pr, w, im)
                fl.release(1)
                fp.release(1)
                fw.release(1)
                fi.release(1)

            worker = Worker(core, [fl.cons(), fp.cons(), fw.prod(), fi.prod(), kernel])
            self.logits.bind(fl.prod())
            self.program.bind(fp.prod())
            self.w.bind(fw.cons())
            self.image.bind(fi.cons())
            return Built([worker], [])

        def sequence(self, rt):
            tg = TaskGroup()
            self.logits.fill(group=tg)
            self.program.fill(group=tg)
            self.w.drain(group=tg, wait=True)
            self.image.drain(group=tg, wait=True)
            tg.finish()

        def ops(self) -> int:
            return self.top * self.experts

        def reference(self, logits, program):
            ids, w = route_reference(logits, self.experts, self.top)
            if self.picked >= 0:
                iron_moeprobe.PICKED[self.picked] = [int(i) for i in ids]
            p = np.asarray(program, np.int64).reshape(self.slots, 4)
            img = np.where(p[:, 0] < 0, p[:, 2], ids[np.clip(p[:, 0], 0, self.top - 1)] * p[:, 1] + p[:, 2])
            img = (img << p[:, 3]).astype(np.int32)
            return w, img

        def tolerance(self) -> Tolerance:
            return Tolerance.relative(0.02, 1e-3, note="exp_f32 and a float divide, rounded to bf16")

    return MoERoute


def quantize_experts(weights: list, fmt: str = "q4") -> np.ndarray:
    """Experts' (n, k) weights, each in iron_q4's blocks, one after another: an expert table."""
    import iron_q4

    return np.concatenate([iron_q4.quantize_fmt(np.asarray(w, np.float32), fmt).reshape(-1, iron_q4.FORMATS[fmt])
                           for w in weights])


def _check(args) -> int:
    """A graph of one MoERoute (random logits and a program of every word kind) and its check."""
    import iron
    from iron_graph import bind_device, export_graph, write_check

    bind_device("npu2", 8)
    R = route_operator()
    n = 320
    rng = np.random.default_rng(args.seed)
    logits = (rng.standard_normal(n) * 2).astype(bfloat16)
    if args.ties:  # equal logits: the lower index first
        logits[[7, 100, 200]] = logits.max()
    program = np.zeros((SLOTS, 4), np.int32)
    program[:, 0] = -1
    program[:, 2] = np.arange(SLOTS) * 3
    for s in range(16):  # the 8 chosen's offsets twice: addr words, then core words
        program[s] = (s % 8, 1 << 18, 64 * s, 2 if s >= 8 else 0)

    class Graph(iron.Graph):
        def body(self, logits, program):
            return R.step(logits, program, n=n)

        def reference(self, logits, program):
            ids, w = route_reference(logits, 256, 8)
            p = np.asarray(program, np.int64).reshape(SLOTS, 4)
            img = np.where(p[:, 0] < 0, p[:, 2], ids[np.clip(p[:, 0], 0, 7)] * p[:, 1] + p[:, 2])
            return w, (img << p[:, 3]).astype(np.int32)

    g = Graph()
    cg = g.compile(logits=(n,), program=((4 * SLOTS,), np.int32))
    m = export_graph(cg, args.out, "moeroute", dict(n=n))
    write_check(g, m, args.out, dict(logits=logits, program=program.reshape(-1)), None)
    print(f"{args.out}: MoERoute, top 8 of 256 and the shared gate")
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="export a graph of one MoERoute and its check")
    c.add_argument("out", type=Path)
    c.add_argument("--seed", type=int, default=5)
    c.add_argument("--ties", action="store_true")
    return _check(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
