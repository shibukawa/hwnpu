#!/usr/bin/env python3
"""MoGe-2 (Ruicheng/moge-2-vit{s,b,l}-normal, MIT) on IRON: the graphs cmd/hwnpu-depth runs on
AMD's NPU for a metric point map, depth, normals and a mask from one picture.

MoGe-2 is DINOv2's ViT (some of its layers after the final norm, each projected by a 1 x 1
convolution and summed), a ConvStack neck over five scales of the patch grid (1x to 16x; the
first takes the features beside the view plane's UV, the others the UV alone), three ConvStack
heads (points, normal, mask) that take every scale of the neck, and an MLP on the class token
for the metric scale. A ConvStack level adds its input (a 1 x 1 convolution) to the running
features, runs residual blocks (ReLU, 3 x 3, ReLU, 3 x 3, plus the input; no norms in MoGe-2)
and resamples to the next (a 2 x 2 transposed convolution, or bilinear 2x, then a 3 x 3).
Every 3 x 3 pads by replicating the edge (replicate grids, iron_da3.Grid). The graphs, each
exported by iron_graph.py:

  vit     the patch embedding and the blocks: the summed projections (tokens, dim) and the
          metric scale's logit
  neck    the five scales of the neck, dense (H W, C) each
  heads   the points, normal and mask heads: one program, a bank image each (variants 0, 1,
          2); the 16x scale's output (H W, 64), its first 3, 3 or 1 channels the head's

The host (internal/vision/moge) stretches the picture to the export's size, normalizes it, cuts
the patches; resizes the heads' outputs to the picture (bilinear), takes exp of z (and times xy),
sigmoid of the mask, normalizes the normals, solves the focal length and z shift on the point
map (MoGe's recover_focal_shift) and scales by exp of the scale's logit: meters.

  python tools/amd/iron_moge.py ~/models/moge-2-vits-normal OUT   # --tokens 2500

See docs/amd-npu-models.md.
"""

import argparse
import dataclasses
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

import iron
import iron.functional as F
from iron import Scratchpad
from iron.operators.copy import Copy
from iron.operators.relu import ReLU

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iron_da3 as D  # noqa: E402
from iron_anima import bf, command_boundary, variant_bank  # noqa: E402
from iron_da3 import PATCH, PE_IN, ConvOps, Grid, f32, pad64, padded  # noqa: E402
from iron_graph import bind_device, export_graph, write_check  # noqa: E402

# --- model.pt without torch ---------------------------------------------------------------------


class TorchZip:
    """A torch.save zip's tensors as numpy, read without torch: the pickle's objects are only
    dicts, lists and the tensor rebuild calls, the storages the zip's data/ files."""

    DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "BFloat16Storage": np.uint16,
              "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_}

    def __init__(self, path: Path):
        self.z = zipfile.ZipFile(path)
        names = self.z.namelist()
        self.root = names[0].split("/")[0]
        z = self

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                    return z._rebuild
                if (module, name) == ("collections", "OrderedDict"):
                    import collections

                    return collections.OrderedDict
                if module == "torch" and name in TorchZip.DTYPES:
                    return name
                raise pickle.UnpicklingError(f"{module}.{name} is not allowed")

            def persistent_load(self, pid):
                _, storage, key, _, numel = pid
                return (storage, key)

        with self.z.open(f"{self.root}/data.pkl") as f:
            self.obj = Unpickler(f).load()

    def _rebuild(self, storage, offset, size, stride, *rest):
        kind, key = storage
        raw = np.frombuffer(self.z.read(f"{self.root}/data/{key}"), dtype=self.DTYPES[kind])
        if kind == "BFloat16Storage":
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        n = int(np.prod(size)) if size else 1
        a = np.lib.stride_tricks.as_strided(raw[offset:], shape=tuple(size), strides=tuple(s * raw.itemsize for s in stride)) if n else raw[:0]
        return np.array(a, np.float32) if raw.dtype != np.bool_ else np.array(a)


# --- the configuration -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Config:
    base_w: int = 50
    base_h: int = 50
    hidden: int = 384
    heads: int = 6
    layers: int = 12
    out_layers: tuple = (5, 11)
    dim: int = 384  # the encoder's projections
    neck: tuple = (384, 256, 128, 64, 32)  # dim_res_blocks
    res_blocks: tuple = (0, 1, 1, 1, 0)  # the neck's
    head_res_blocks: tuple = (0, 1, 1, 1, 0)  # the heads' (MoGe-2 ViT-L: 2 in the neck, 1 in the heads)
    resamplers: tuple = ("conv_transpose", "conv_transpose", "conv_transpose", "bilinear")
    scale: tuple = (384, 384, 384, 1)
    split: int = 6

    # what iron_da3's VitBlock reads of its config
    qknorm_start: int = -1
    rope_start: int = -1
    alt_start: int = -1

    @property
    def mlp(self) -> int:
        return 4 * self.hidden

    @property
    def gh(self) -> int:
        return self.base_h

    @property
    def gw(self) -> int:
        return self.base_w

    @property
    def width(self) -> int:
        return 14 * self.base_w

    @property
    def height(self) -> int:
        return 14 * self.base_h

    @property
    def patches(self) -> int:
        return self.base_w * self.base_h

    @property
    def tokens(self) -> int:
        return -(-(1 + self.patches) // 512) * 512

    def level(self, i: int) -> tuple[int, int]:
        """Scale i's rows and columns (2^i times the patch grid)."""
        return self.base_h << i, self.base_w << i

    @classmethod
    def of(cls, mc: dict, base_w: int, base_h: int) -> "Config":
        enc, neck = mc["encoder"], mc["neck"]
        vit = {"dinov2_vits14": (384, 6, 12), "dinov2_vitb14": (768, 12, 12), "dinov2_vitl14": (1024, 16, 24)}[enc["backbone"]]
        n_out = enc["intermediate_layers"]
        out_layers = tuple(n_out) if isinstance(n_out, list) else tuple(range(vit[2] - n_out, vit[2]))
        for h in ("points_head", "normal_head", "mask_head"):
            assert mc[h]["dim_res_blocks"] == neck["dim_res_blocks"] and mc[h]["resamplers"] == neck["resamplers"], h
            assert mc[h]["num_res_blocks"] == mc["points_head"]["num_res_blocks"], h
            assert mc[h].get("res_block_in_norm") == "none" and mc[h].get("res_block_hidden_norm") == "none", h
        assert neck.get("res_block_in_norm") == "none" and neck.get("res_block_hidden_norm") == "none", "norms are not supported"
        assert mc.get("remap_output") == "exp", mc.get("remap_output")
        return cls(base_w=base_w, base_h=base_h, hidden=vit[0], heads=vit[1], layers=vit[2], out_layers=out_layers,
                   dim=enc["dim_out"], neck=tuple(neck["dim_res_blocks"]), res_blocks=tuple(neck["num_res_blocks"]),
                   head_res_blocks=tuple(mc["points_head"]["num_res_blocks"]),
                   resamplers=tuple(neck["resamplers"]), scale=tuple(mc["scale_head"]["dims"]))


def base_size(tokens: int, aspect: float) -> tuple[int, int]:
    """MoGe's forward: the patch grid of num_tokens at an aspect (columns, rows)."""
    return round((tokens * aspect) ** 0.5), round((tokens / aspect) ** 0.5)


class Weights:
    """model.pt's state dict, read as iron_da3's Safetensors (a name to an array), with the DA3
    names VitBlock reads mapped to MoGe's."""

    def __init__(self, sd: dict):
        self.sd = sd

    def __contains__(self, n):
        return self._name(n) in self.sd

    @staticmethod
    def _name(n: str) -> str:
        return n.replace("model.backbone.pretrained.", "encoder.backbone.")

    def __getitem__(self, n: str):
        return self.sd[self._name(n)]


# --- the ViT -----------------------------------------------------------------------------------


class Vit(iron.Graph):
    """The patch embedding, the blocks, the encoder's projections and the scale head.

    Input ``patches`` (tokens, 640), as iron_da3's. Per call ``nk`` (1 + patches). Outputs
    ``f`` (tokens, dim): the summed 1 x 1 projections of the layers (row 0 the class token's,
    unused), and ``scale`` (64,): the metric scale's logit first.
    """

    def __init__(self, c: Config, w: Weights):
        E = c.hidden
        self.c = c
        self.blocks = [D.VitBlock(c, w, i) for i in range(max(c.out_layers) + 1)]
        pw = f32(w["encoder.backbone.patch_embed.proj.weight"]).reshape(E, -1)
        self.pe = iron.param(bf(padded(pw, E, PE_IN)))
        pos = np.zeros((c.tokens, E), np.float32)
        pe = D.pos_embed(_PosWeights(w), c)
        pos[: pe.shape[0]] = pe
        pos[1 : pe.shape[0]] += f32(w["encoder.backbone.patch_embed.proj.bias"])
        pos[0] += f32(w["encoder.backbone.cls_token"]).reshape(-1)
        self.pos = iron.param(bf(pos))
        self.norm = iron.param(bf(w["encoder.backbone.norm.weight"]))
        self.normb = iron.param(bf(w["encoder.backbone.norm.bias"]))
        self.proj = [iron.param(bf(f32(w[f"encoder.output_projections.{i}.weight"])[:, :, 0, 0])) for i in range(len(c.out_layers))]
        self.projb = iron.param(bf(sum(f32(w[f"encoder.output_projections.{i}.bias"]) for i in range(len(c.out_layers)))))
        dims = c.scale
        self.mlp = []
        for j in range(len(dims) - 1):
            wt, b = f32(w[f"scale_head.{2 * j}.weight"]), f32(w[f"scale_head.{2 * j}.bias"])
            n = pad64(wt.shape[0])
            self.mlp.append((iron.param(bf(padded(wt, n, wt.shape[1] if j == 0 else pad64(wt.shape[1])))), iron.param(bf(padded(b, n)))))

    def body(self, patches, *, nk: Scratchpad[np.int32]):
        c = self.c
        E = c.hidden
        x = F.linear(patches, self.pe, precision=D.PREC.vit) + self.pos
        f = None
        n = None
        for i, b in enumerate(self.blocks):
            if i and i % c.split == 0:
                command_boundary()
            x = b(x, nk)
            if i in c.out_layers:
                n = F.layer_norm(x, (E,), self.norm, self.normb, eps=1e-6)
                y = F.linear(n, self.proj[c.out_layers.index(i)], precision=D.PREC.vit)
                f = y if f is None else f + y
        f = f + self.projb
        # the scale head on the last layer's normed class token (one row: GEMV)
        h = n[0:1].reshape(E)
        for j, (wt, b) in enumerate(self.mlp):
            h = F.linear(h, wt, b, precision=D.PREC.vit)
            if j < len(self.mlp) - 1:
                h = ReLU.step(h)
        return f, h


class _PosWeights:
    """pos_embed's one name for iron_da3.pos_embed."""

    def __init__(self, w):
        self.w = w

    def __getitem__(self, n):
        return self.w["encoder.backbone.pos_embed"]


# --- the ConvStacks --------------------------------------------------------------------------------


def view_uv(w: int, h: int, aspect: float) -> np.ndarray:
    """MoGe's normalized_view_plane_uv: (h w, 2), u then v."""
    sx = aspect / (1 + aspect**2) ** 0.5
    sy = 1 / (1 + aspect**2) ** 0.5
    u = np.linspace(-sx * (w - 1) / w, sx * (w - 1) / w, w, dtype=np.float32)
    v = np.linspace(-sy * (h - 1) / h, sy * (h - 1) / h, h, dtype=np.float32)
    return np.stack([np.tile(u, h), np.repeat(v, w)], axis=1)


def grid_image(dense: np.ndarray, g: Grid) -> np.ndarray:
    """A constant (H W, C) laid out as a replicate grid (rows, C), its pads the edge pixels."""
    out = np.zeros((g.rows, g.C), np.float32)
    img = dense.reshape(g.H, g.W, -1)
    img = np.concatenate([img[:, :1], img, img[:, -1:]], axis=1)  # (H, W + 2, C)
    img = np.concatenate([img[:1], img, img[-1:]], axis=0)  # (H + 2, W + 2, C)
    for y in range(-1, g.H + 1):
        s = g.pixel(y, -1)
        out[s : s + g.W + 2, : img.shape[2]] = img[y + 1]
    return out


class Stack(ConvOps):
    """A ConvStack: the neck (``name`` "neck": input the encoder's features ``f`` (tokens,
    dim); outputs every scale, dense) or a head (inputs the neck's scales ``n0``..``n4``;
    output the last scale's projection, (H W, 64))."""

    def __init__(self, c: Config, w: Weights, name: str):
        self.c, self.part = c, name
        self.init_ops()
        p = f"{name}."
        g = lambda n: f32(w[p + n])  # noqa: E731
        ch = c.neck
        chp = [pad64(x) for x in ch]
        self.chp = chp
        self.nres = c.res_blocks if name == "neck" else c.head_res_blocks
        aspect = c.width / c.height
        for i in range(5):
            H, W = c.level(i)
            gr = Grid(H, W, chp[i], True)
            if name == "neck":
                wt, b = g(f"input_blocks.{i}.weight")[:, :, 0, 0], g(f"input_blocks.{i}.bias")
                uv = view_uv(W, H, aspect) @ wt[:, -2:].T + b  # the UV's share and the bias, a constant
                if i == 0:
                    self._param("in0", padded(wt[:, : c.dim], chp[0], c.dim))
                    const = np.zeros((c.tokens, chp[0]), np.float32)
                    const[1 : 1 + c.patches, : ch[0]] = uv
                    self._param("in0.c", const)
                else:
                    self._param(f"in{i}.c", grid_image(padded(uv, H * W, chp[i]), gr))
            else:
                self._param(f"in{i}", padded(g(f"input_blocks.{i}.weight")[:, :, 0, 0], chp[i], chp[i]))
                self._param(f"in{i}.b", padded(g(f"input_blocks.{i}.bias"), chp[i]))
            for r in range(self.nres[i]):
                for j, k in ((1, 2), (2, 5)):
                    self._conv3(f"res{i}.{r}.{j}", g(f"res_blocks.{i}.{r}.layers.{k}.weight"), g(f"res_blocks.{i}.{r}.layers.{k}.bias"), chp[i], chp[i])
            if i < 4:
                kind = c.resamplers[i]
                if kind == "conv_transpose":
                    self.deconv_weight(f"up{i}", g(f"resamplers.{i}.0.weight"), g(f"resamplers.{i}.0.bias"), chp[i + 1], chp[i])
                elif kind == "bilinear":
                    H2, W2 = c.level(i + 1)
                    self.add_interp(H2, H, align=False)
                    self.add_interp(W2, W, align=False)
                else:
                    raise SystemExit(f"resampler {kind} is not supported")
                cin = chp[i + 1] if kind == "conv_transpose" else chp[i]
                self._conv3(f"up{i}.conv", g(f"resamplers.{i}.1.weight"), g(f"resamplers.{i}.1.bias"), chp[i + 1], cin)
        if name != "neck":
            wt, b = g("output_blocks.4.weight")[:, :, 0, 0], g("output_blocks.4.bias")
            self._param("out", padded(wt, 64, chp[4]))
            self._param("out.b", padded(b, 64))

    def res(self, x, g: Grid, name: str):
        a, ga = self.conv3(ReLU.step(x), g, f"{name}.1", g.C, post=lambda y, r0: ReLU.step(y))

        def post(y, r0):
            s = g.band + r0 * g.Wp
            R = y.shape[0] // g.Wp
            return y + x[s : s + R * g.Wp]

        return self.conv3(a, ga, f"{name}.2", g.C, post=post)

    def resample(self, x, g: Grid, i: int):
        c = self.c
        H2, W2 = c.level(i + 1)
        cout = self.chp[i + 1]
        if c.resamplers[i] == "conv_transpose":
            u, gu = self.deconv(self.to_dense(x, g), g.H, g.W, f"up{i}", 2, cout, rep=True)
        else:
            d = self.resize(self.to_dense(x, g), g.H, g.W, H2, W2, g.C, align=False)
            gu = Grid(H2, W2, g.C, True)
            u = self.to_grid(d, gu)
        return self.conv3(u, gu, f"up{i}.conv", cout)

    def run(self, inputs):
        """The stack over the scales: inputs[i] (dense) or None; returns each scale's grid."""
        c = self.c
        outs = []
        x = gx = None
        for i in range(5):
            H, W = c.level(i)
            g = Grid(H, W, self.chp[i], True)
            if self.part == "neck":
                if i == 0:
                    d = F.linear(inputs[0], self.w("in0"), precision=D.PREC.head) + self.w("in0.c")
                    x, gx = self.to_grid(d, g, first=1), g
                else:
                    x = x + self.w(f"in{i}.c")
            else:
                d = F.linear(inputs[i], self.w(f"in{i}"), self.w(f"in{i}.b"), precision=D.PREC.head)
                y = self.to_grid(d, g)
                x, gx = (y, g) if i == 0 else (x + y, gx)
            for r in range(self.nres[i]):
                x, gx = self.res(x, gx, f"res{i}.{r}")
            outs.append((x, gx))
            if i < 4:
                x, gx = self.resample(x, gx, i)
        return outs


class Neck(Stack):
    def __init__(self, c: Config, w: Weights):
        super().__init__(c, w, "neck")

    def body(self, f):
        # a state's view is no output of the graph's own: copies
        return tuple(Copy.step(self.to_dense(x, g)) for x, g in self.run([f, None, None, None, None]))


class Head(Stack):
    def body(self, n0, n1, n2, n3, n4):
        x, g = self.run([n0, n1, n2, n3, n4])[-1]
        return F.linear(self.to_dense(x, g), self.w("out"), self.w("out.b"), precision=D.PREC.head)


# --- export -----------------------------------------------------------------------------------------

HEADS = ("points_head", "normal_head", "mask_head")


def export(w: Weights, mc: dict, out: Path, c: Config, check: bool) -> None:
    meta = dict(config=dataclasses.asdict(c), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], patch=PATCH,
                tokens=c.tokens, rows=1 + c.patches, width=c.width, height=c.height)
    rng = np.random.default_rng(0)
    t0 = time.time()
    g = Vit(c, w)
    cg = g.compile(patches=(c.tokens, PE_IN))
    m = export_graph(cg, out / "vit", "vit", meta)
    print(f"vit: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    img = rng.standard_normal((c.height, c.width, 3)).astype(np.float32)
    if check:
        write_check(g, m, out / "vit", dict(patches=bf(D.patches(img, c))), dict(nk=1 + c.patches))
    t0 = time.time()
    g = Neck(c, w)
    cg = g.compile(f=(c.tokens, c.dim))
    m = export_graph(cg, out / "neck", "neck", meta)
    print(f"neck: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    if check:
        write_check(g, m, out / "neck", dict(f=bf(rng.standard_normal((c.tokens, c.dim)))))
    shapes = {}
    for i in range(5):
        H, W = c.level(i)
        n = -(-H * W // 256) * 256
        shapes[f"n{i}"] = (n, pad64(c.neck[i]))
    # the three heads: one program, a bank image each (the same shapes), read in turn into
    # one bank by hwnpu (iron.GraphOptions.SwapVariants): the work area of one head pinned
    t0 = time.time()
    g = Head(c, w, HEADS[0])
    cg = g.compile(**shapes)
    d = out / "heads"
    m = export_graph(cg, d, "heads", dict(meta, heads=[h.removesuffix("_head") for h in HEADS]))
    variants = [{k: b["image"] for k, b in m["banks"].items()}]
    for i, h in enumerate(HEADS[1:], 1):
        variants.append(variant_bank(cg, Head(c, w, h), d, f"bank_h{i}"))
    m["variants"] = variants
    (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    print(f"heads: compiled in {time.time() - t0:.0f} s; {m['graph']['plan']}", flush=True)
    if check:
        write_check(g, m, d, {k: bf(rng.standard_normal(s) * 0.5) for k, s in shapes.items()})
    (out / "model.json").write_text(json.dumps(dict(kind="moge", **meta), indent=1) + "\n")


ASPECTS = {"1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3, "16:10": 16 / 10, "10:16": 10 / 16,
           "16:9": 16 / 9, "9:16": 9 / 16}

ARGS = None


def main(argv=None) -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a MoGe-2 repository (model.pt)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--tokens", type=int, default=2500, help="the ViT's patches (MoGe's num_tokens: 1200 to 3600)")
    ap.add_argument("--aspects", nargs="*", default=list(ASPECTS), help="the aspects to export, each a directory")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ARGS = a = ap.parse_args(argv)
    bind_device(a.device, a.cols)
    tz = TorchZip(a.root / "model.pt")
    mc, w = tz.obj["model_config"], Weights(tz.obj["model"])
    sizes = []
    for asp in a.aspects:
        bw, bh = base_size(a.tokens, ASPECTS[asp])
        c = Config.of(mc, bw, bh)
        print(f"== {asp}: {bw}x{bh} patches, {c.width}x{c.height} pixels", flush=True)
        d = f"{c.width}x{c.height}"
        export(w, mc, a.out / d, c, a.check)
        sizes.append([c.width, c.height])
    (a.out / "model.json").write_text(json.dumps(dict(name=a.root.name, kind="moge", sizes=sizes, tokens=a.tokens), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
