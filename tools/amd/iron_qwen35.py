"""Qwen3.5 / Qwen3.6 text models (``qwen3_5``, ``qwen3_5_moe``) on IRON's shared decoder (``iron.lm``).

An application of amd/IRON kept in hwnpu (IRON has none for Qwen3.5), exported by
``iron_export_lm.py qwen35``. Its shape is read from the checkpoint's ``config.json`` (the
root, or its ``text_config``); the structure follows transformers' modeling_qwen3_5.py:

- ``layer_types``: ``linear_attention`` layers are Gated DeltaNet (``GatedDeltaNet``),
  ``full_attention`` ones gated attention (``GatedAttention``).
- Gated DeltaNet: ``in_proj_qkv``, ``in_proj_a`` and ``in_proj_b`` are one projection here,
  its rows in iron_gdn's column order (``gdn_layout``), through one depth-wise causal
  convolution of ``linear_conv_kernel_dim`` taps (a and b pass it unchanged: their taps are
  0, 0, 0, 1), then iron_gdn's GatedDeltaRule (silu on q, k and v, the l2 norms, the decay
  and the delta rule), its float32 state a slot per prompt-chunk boundary (``Checkpoints``).
  Its output is RMS-normed per value head (``norm``, a plain weight), times ``silu(z)``
  (``in_proj_z``), and ``out_proj``.
- Gated attention: ``q_proj`` holds each head's query then its gate (``2 head_dim`` rows a
  head); q and k are RMS-normed per head; RoPE turns the first ``partial_rotary_factor`` of
  each head (rotate_half over those columns; text positions, so mrope is plain RoPE); the
  output times ``sigmoid(gate)`` before ``o_proj``.
- Every RMSNorm but the gated one scales by ``1 + weight`` (``RMSNorm.plus_one``).
- The feed-forward is a SwiGLU MLP (``qwen3_5``), or for ``qwen3_5_moe`` (Qwen3.6-35B-A3B,
  ``Qwen35Moe``) 8 of 256 experts and a shared one gated by ``sigmoid(shared_expert_gate)``,
  routed on the device (iron_moe.py): a layer is two commands of the decode step, and the
  experts are q4nx tables read in place. The MoE model has a decode step only; hwnpu runs a
  prompt through it a token at a time (the prompt chunk's MoE, experts by token, is not
  built yet).
"""

import dataclasses

import numpy as np
from ml_dtypes import bfloat16

import iron
import iron.functional as F
from iron import Precision, lm
from iron.lm import (
    Attention,
    CausalConv1d,
    CausalLM,
    Checkpoints,
    DecoderConfig,
    DecoderLayer,
    Linear,
    RMSNorm,
)
from iron.lm.checkpoint import renamed
from iron.operators.copy import Copy
from iron.operators.sigmoid import Sigmoid

import iron_gdn
import iron_moe
import iron_moeprobe

COLUMNS = 8  # the columns GatedDeltaRule spreads over (npu2)

# install()'s formats (iron_q4's), by part: the mixers' projections (Gated DeltaNet's and the
# attention's), a dense MLP's, the head. Empty: all bf16.
QUANT: dict[str, str] = {}


def install(attn: str = "q8", mlp: str = "q4", head: str = "q4") -> None:
    """Build Qwen3.5 with its projections in these formats ("q4", "q8" or "bf16"), F.linear
    running iron_q4's on them. The router and the experts' tables are as they are without it."""
    import iron_q4

    for part, fmt in dict(attn=attn, mlp=mlp, head=head).items():
        if fmt not in ("q4", "q8", "bf16"):
            raise ValueError(f"{part} in {fmt}: q4, q8 or bf16")
    iron_q4.install_linear()
    QUANT.update(attn=attn, mlp=mlp, head=head)


@dataclasses.dataclass(frozen=True)
class Qwen35Config(DecoderConfig):
    """``DecoderConfig`` with Qwen3.5's layer kinds, its Gated DeltaNet's shape and partial RoPE."""

    rope_theta: float = dataclasses.field(metadata={"json": "rope_parameters.rope_theta"})
    partial_rotary_factor: float = dataclasses.field(
        default=1.0, kw_only=True, metadata={"json": "rope_parameters.partial_rotary_factor"})
    layer_types: tuple[str, ...] = dataclasses.field(kw_only=True, metadata={"convert": tuple})
    linear_conv_kernel_dim: int = dataclasses.field(default=4, kw_only=True)
    linear_key_head_dim: int = dataclasses.field(default=128, kw_only=True)
    linear_value_head_dim: int = dataclasses.field(default=128, kw_only=True)
    linear_num_key_heads: int = dataclasses.field(default=16, kw_only=True)
    linear_num_value_heads: int = dataclasses.field(default=32, kw_only=True)
    num_experts: int = dataclasses.field(default=0, kw_only=True)
    num_experts_per_tok: int = dataclasses.field(default=8, kw_only=True)
    moe_intermediate_size: int = dataclasses.field(default=0, kw_only=True)
    shared_expert_intermediate_size: int = dataclasses.field(default=0, kw_only=True)
    precision: Precision = lm.not_json(Precision.MEDIUM, kw_only=True)

    @classmethod
    def from_json(cls, data: dict, source="config.json") -> "Qwen35Config":
        """``DecoderConfig.from_json`` on the text model's object (the root or ``text_config``).

        Raises:
            ValueError: ``data`` asks for what is not built: another model, RoPE other than
                the default, attention biases, or experts.
        """
        text = data.get("text_config", data)
        kind = text.get("model_type")
        if kind not in ("qwen3_5_text", "qwen3_5_moe_text", "qwen3_5", "qwen3_5_moe"):
            raise ValueError(f"{source}: model_type {kind!r} is not Qwen3.5")
        rope = text.get("rope_parameters", {})
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"{source}: RoPE {rope['rope_type']!r} is not built here")
        if text.get("attention_bias"):
            raise ValueError(f"{source}: attention biases are not built here")
        if not text.get("attn_output_gate", True):
            raise ValueError(f"{source}: attention without its output gate is not built here")
        kinds = set(text.get("layer_types", ())) - {"linear_attention", "full_attention"}
        if kinds:
            raise ValueError(f"{source}: layer kinds {sorted(kinds)} are not built here")
        if text.get("num_experts", 0):
            if (text.get("num_experts_per_tok"), text.get("num_experts")) != (8, 256):
                raise ValueError(f"{source}: experts other than 8 of 256 are not built here")
            if text.get("moe_intermediate_size") != 512 or text.get("shared_expert_intermediate_size") != 512:
                raise ValueError(f"{source}: experts of other than 512 are not built here")
            # DecoderConfig's intermediate_size, which a dense layer would read
            text = {**text, "intermediate_size": text["moe_intermediate_size"]}
        return super().from_json(text, source)

    def __post_init__(self):
        super().__post_init__()
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"Qwen35Config: {len(self.layer_types)} layer_types for "
                             f"{self.num_hidden_layers} layers")
        if self.partial_rotary_factor != 1.0:
            object.__setattr__(self, "rotary_dim", int(self.head_dim * self.partial_rotary_factor))

    def linear(self, i: int) -> bool:
        """Whether layer ``i`` is Gated DeltaNet."""
        return self.layer_types[i] == "linear_attention"


class GatedNorm(iron.Graph):
    """transformers' Qwen3_5RMSNormGated: each value head's rows RMS-normed, times ``weight``
    (no ``1 +``), times ``silu(z)``."""

    def __init__(self, dim: int, eps: float):
        self.dim, self.eps = dim, eps
        self.weight = iron.param(dim)

    def body(self, y, z):
        n, width = y.shape
        h = F.rms_norm(y.reshape(n * width // self.dim, self.dim), (self.dim,), self.weight, self.eps)
        return h.reshape(n, width) * F.silu(z)


class GatedDeltaNet(iron.Graph):
    """Qwen3.5's linear attention, as the checkpoint's ``linear_attn`` names it (module docstring)."""

    def __init__(self, c: Qwen35Config):
        E = c.hidden_size
        H, KH = c.linear_num_value_heads, c.linear_num_key_heads
        dk, dv, K = c.linear_key_head_dim, c.linear_value_head_dim, c.linear_conv_kernel_dim
        self.shape = (H, KH, dk, dv)
        self.qkv = 2 * KH * dk + H * dv
        self.in_proj_qkv = Linear(E, self.qkv, False, precision=c.precision)
        self.in_proj_a = Linear(E, H, False, precision=c.precision)
        self.in_proj_b = Linear(E, H, False, precision=c.precision)
        self.in_proj_z = Linear(E, H * dv, False, precision=c.precision)
        self.conv1d = iron.Graph()
        self.conv1d.weight = iron.param(self.qkv, 1, K)
        self.dt_bias = iron.param(H)
        self.A_log = iron.param(H)
        self.norm = GatedNorm(dv, c.eps)
        self.out_proj = Linear(H * dv, E, False, precision=c.precision)
        self.config = c  # the fused projection and convolution are made once loaded
        self.state = Checkpoints.of((H * dk * dv,), c.max_seq_len, c.prefill_chunk, np.float32)
        self.words = None

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        c = self.config
        H, KH, dk, dv = self.shape
        W = self.qkv + 2 * H
        # The fused projection and convolution, made from the checkpoint's own (bound here, so
        # that nothing reads them from the checkpoint).
        self.in_proj = Linear(c.hidden_size, W, False, precision=c.precision)
        self.conv = CausalConv1d(c, W, c.linear_conv_kernel_dim)
        self.in_proj._bound = self.conv._bound = True
        order = iron_gdn.gdn_layout(H, KH, dk, dv, COLUMNS)
        rows = np.concatenate([np.asarray(p.weight) for p in (self.in_proj_qkv, self.in_proj_a, self.in_proj_b)])
        self.in_proj.weight = iron.param(np.ascontiguousarray(rows[order]))
        w = np.asarray(self.conv1d.weight)
        K = w.shape[-1]
        taps = np.zeros((self.qkv + 2 * H, 1, K), w.dtype)
        taps[: self.qkv] = w
        taps[self.qkv :, 0, K - 1] = 1  # a and b: the row itself
        taps = taps[order]
        self.conv.weight = iron.param(np.ascontiguousarray(taps))
        # A row per tap, as CausalConv1d.load_state_dict makes them.
        self.conv.taps = iron.param(np.ascontiguousarray(taps.reshape(len(order), K).T))
        A = -np.exp(np.asarray(self.A_log, np.float32))
        self.words = iron_gdn.head_words(np.asarray(self.dt_bias, np.float32), A)
        # read once here: the fused copies are what the device holds
        self.in_proj_qkv = self.in_proj_a = self.in_proj_b = self.conv1d = None
        self.dt_bias = self.A_log = None

    def body(self, x, call):
        H, KH, dk, dv = self.shape
        GDR = iron_gdn.gdn_operator(COLUMNS, H // (2 * COLUMNS))
        p = self.in_proj(x)
        u = self.conv(p, call)
        read, write = self.state.at(call)
        y, final = GDR.step(u, Copy.step(read, dtype=np.float32), heads=H, key_heads=KH, key_dim=dk,
                            value_dim=dv, words=self.words)
        Copy.step(final, write, dtype=np.float32)
        return self.out_proj(self.norm(y, self.in_proj_z(x)))


class GatedAttention(Attention):
    """``Attention`` whose ``q_proj`` also gives a gate per head column, ``sigmoid(gate)`` times
    the heads' output before ``o_proj``; q and k normed per head."""

    def __init__(self, c, i: int):
        super().__init__(c, i, norms="qk")

    def projections(self, c) -> None:
        super().projections(c)
        H, D = self.heads, self.head_dim
        self.q_proj = Linear(c.hidden_size, 2 * H * D, False, precision=c.precision)

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        H, D = self.heads, self.head_dim
        w = np.asarray(self.q_proj.weight).reshape(H, 2, D, -1)
        self.q_proj.weight = iron.param(np.ascontiguousarray(w[:, 0].reshape(H * D, -1)))
        self.gate = iron.param(np.ascontiguousarray(w[:, 1].reshape(H * D, -1)))

    def output(self, o, x):
        g = Sigmoid.step(F.linear(x, self.gate, precision=self.q_proj.precision))
        return self.o_proj(o * g)


class Qwen35Layer(DecoderLayer):
    """Layer ``i``: Gated DeltaNet or gated attention, then the feed-forward, pre-normed."""

    def __init__(self, c: Qwen35Config, i: int):
        if c.linear(i):
            super().__init__(c, i, self_attn=iron.Graph(), cache=iron.Graph())
            del self.self_attn, self.cache
            self.linear_attn = GatedDeltaNet(c)
        else:
            super().__init__(c, i, GatedAttention(c, i))
            self.linear_attn = None

    def mix(self, h, call):
        if self.linear_attn is not None:
            return self.linear_attn(h, call)
        return super().mix(h, call)


class Qwen35(CausalLM):
    """Qwen3.5 on ``config``'s shape and a checkpoint's ``weights``."""

    config = Qwen35Config
    layer_class = Qwen35Layer

    def __init__(self, config: Qwen35Config, weights):
        # A multimodal checkpoint's text model is model.language_model.
        super().__init__(config, renamed(weights, {"model.language_model": "model"}))
        for i, layer in enumerate(self.layers):
            if config.linear(i):
                layer.rope = None
        RMSNorm.plus_one(self)
        if QUANT:
            self.quantize(QUANT)

    def quantize(self, fmts: dict) -> None:
        """Put the loaded projections in ``fmts``' formats (``install``)."""
        import iron_q4

        def q(holder, name, fmt):
            if fmt != "bf16":
                setattr(holder, name, iron_q4.q_param(np.asarray(getattr(holder, name), np.float32), fmt))

        for layer in self.layers:
            if layer.linear_attn is not None:
                g = layer.linear_attn
                for lin in (g.in_proj, g.in_proj_z, g.out_proj):
                    q(lin, "weight", fmts["attn"])
            else:
                a = layer.self_attn
                for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                    q(lin, "weight", fmts["attn"])
                q(a, "gate", fmts["attn"])
            if not isinstance(layer.mlp, SparseMoe):
                for lin in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                    q(lin, "weight", fmts["mlp"])
        # the head its own quantized copy where it is tied: the embeddings stay bf16
        q(self.lm_head, "weight", fmts["head"])


class SparseMoe(iron.Graph):
    """Qwen3.6's ``mlp``: the router (``gate``), the 256 ``experts`` and the ``shared_expert``
    with its ``shared_expert_gate``; on the device MoERoute, MoEGateUp and MoEDown (iron_moe.py)."""

    def __init__(self, c: Qwen35Config, index: int):
        E, I, N = c.hidden_size, c.moe_intermediate_size, c.num_experts
        self.index, self.hidden, self.inter, self.n = index, E, I, N
        self.gate = Linear(E, N, False, precision=c.precision)
        self.experts = iron.Graph()
        self.experts.gate_up_proj = iron.param(N, 2 * I, E)
        self.experts.down_proj = iron.param(N, E, I)
        self.shared_expert = iron.Graph()
        S = c.shared_expert_intermediate_size
        for name, shape in (("gate_proj", (S, E)), ("up_proj", (S, E)), ("down_proj", (E, S))):
            setattr(self.shared_expert, name, Linear(shape[1], shape[0], False, precision=c.precision))
        self.shared_expert_gate = Linear(E, 1, False, precision=c.precision)
        self.precision = c.precision

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        E, I, N = self.hidden, self.inter, self.n
        gu = np.asarray(self.experts.gate_up_proj, np.float32)
        down = np.asarray(self.experts.down_proj, np.float32)
        sh = self.shared_expert
        tables = [iron_moeprobe.gu_table(gu[e, :I], gu[e, I:]) for e in range(N)]
        tables.append(iron_moeprobe.gu_table(np.asarray(sh.gate_proj.weight, np.float32),
                                             np.asarray(sh.up_proj.weight, np.float32)))
        self.gu = iron.param(iron_moe.quantize_experts(tables))
        downs = [down[e] for e in range(N)] + [np.asarray(sh.down_proj.weight, np.float32)]
        self.down = iron.param(iron_moe.quantize_experts(downs))
        self.gu_bytes = self.gu.nbytes // (N + 1)
        self.down_bytes = self.down.nbytes // (N + 1)
        # the router's logits, then the shared expert's gate, padded to whole column blocks
        rows = -(-(N + 1) // 64) * 64
        router = np.zeros((rows, E), bfloat16)
        router[:N] = np.asarray(self.gate.weight)
        router[N] = np.asarray(self.shared_expert_gate.weight)[0]
        self.router = iron.param(router)
        # B's scratchpad image: written once the lowering has placed its words (program_rows)
        self.program = iron.param(np.zeros(4 * iron_moe.SLOTS, np.int32))
        self.gate = self.experts = self.shared_expert = self.shared_expert_gate = None

    def body(self, hn, h, call):
        E, I = self.hidden, self.inter
        Route = iron_moe.route_operator()
        logits = F.linear(hn, self.router, precision=self.precision)
        n = self.router.shape[0]
        w, image = Route.step(logits.reshape(n), self.program, n=n, experts=self.n, picked=self.index)
        call.feedback.append((self.index, image.name))
        iron.on_host(lambda: None)  # command B: the experts, at the offsets the image holds
        GU, D = iron_moeprobe.fused_operators()
        e = call.experts[self.index]
        a = GU.step(self.gu, hn.reshape(E), w, K=E, NE=I, picked=self.index,
                    **{f"e{i}": e[i] * self.gu_bytes for i in range(len(e))})
        y = D.step(self.down, a, h.reshape(E), KE=I, N=E, picked=self.index,
                   **{f"e{i}": e[i] * self.down_bytes for i in range(len(e))})
        iron.on_host(lambda: None)
        return y.reshape(1, E)


class Qwen35MoeLayer(Qwen35Layer):
    """``Qwen35Layer`` whose feed-forward is ``SparseMoe`` (its residual added in MoEDown)."""

    def __init__(self, c: Qwen35Config, i: int):
        super().__init__(c, i)
        self.mlp = SparseMoe(c, i)

    def body(self, x, call):
        h = x + self.mix(self.input_layernorm(x), call)
        return self.mlp(self.post_attention_layernorm(h), h, call)


class Qwen35Moe(Qwen35):
    """Qwen3.6-35B-A3B (``qwen3_5_moe``): ``Qwen35`` with ``SparseMoe`` layers, a decode step only.

    Its body takes a scratchpad value per layer and chosen expert (``e<layer>_<i>``), which no
    host gives: command B's scratchpad is in the feedback argument, where the layer's MoERoute
    writes them (``hwnpu_fed``)."""

    layer_class = Qwen35MoeLayer
    hwnpu_name = "Qwen35Moe"

    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        if "e0_0" in getattr(cls, "_values_made", ()):
            return super().from_checkpoint(path, **kwargs)
        from iron.lm.checkpoint import weights_file

        layers = cls.default_config(weights_file(path)).num_hidden_layers
        return moe_class(cls, layers).from_checkpoint(path, **kwargs)

    def shapes(self):
        return [{}]  # the decode step alone

    def draw(self, logits, call):
        # No draw on the device: its commands end at host steps, so no Emit carries a drawn
        # token, and hwnpu draws from the logits anyway.
        return logits, iron.carry(token=call.token + 0, position=call.position + 1)

    def record(self, **shapes):
        self._fed = []
        recording = super().record(**shapes)
        recording.feedback.extend(name for _, name in self._fed)
        return recording

    def moe_body(self, values, **standard):
        experts = [[values[f"e{l}_{i}"] for i in range(8)] for l in range(self.config.num_hidden_layers)]
        import iron.lm.causal as causal

        made = causal.Call

        def call(**kw):
            c = made(**kw)
            c.experts, c.feedback = experts, self._fed
            return c

        causal.Call = call
        try:
            return CausalLM.body(self, **standard)
        finally:
            causal.Call = made

    def program_rows(self, kernels) -> list:
        """Fill each layer's MoERoute program from the decode step's commands (the manifest's
        ``kernels``): command B of layer l is the one whose words are layer l's offsets. Returns
        the manifest's ``fed``: by such command, the feedback buffer its scratchpad is."""
        images = dict(self._fed)
        fed = []
        for layer in self.layers:
            l, m = layer.mlp.index, layer.mlp
            found = [(k, kern["parameters"]) for k, kern in enumerate(kernels)
                     if any((p.get("form") or {}).get("value", "").startswith(f"e{l}_") for p in kern["parameters"])]
            if len(found) != 1:
                raise SystemExit(f"layer {l}: {len(found)} commands read its experts' offsets")
            k, words = found[0]
            prog = np.zeros((iron_moe.SLOTS, 4), np.int32)
            prog[:, 0] = -1
            for p in words:
                f = p.get("form")
                if not f or not f["value"].startswith(f"e{l}_"):
                    raise SystemExit(f"layer {l}: command {k} reads {p['name']}, which no routing gives")
                plain = (f["down"], f["mul"], f["add"], f["post"], f["lo"], f["hi"], f["mod"]) == (0, 1, 0, 0, None, None, None)
                if not plain or p["index"] >= iron_moe.SLOTS:
                    raise SystemExit(f"layer {l}: {p['name']} at {p['index']} is {f}, not id * scale + bias")
                prog[p["index"]] = (int(f["value"].split("_")[1]), f["scale"], f["bias"],
                                    2 if p["kind"] == "core" else 0)
            np.asarray(m.program).reshape(-1)[:] = prog.reshape(-1)
            fed.append(dict(command=k, buffer=images[l], words=len(words)))
        return fed


def moe_class(base, layers: int):
    """``base`` whose body takes the 8 offsets of each of ``layers`` layers as scratchpad values."""
    from iron import Carried, Scratchpad

    names = [f"e{l}_{i}" for l in range(layers) for i in range(8)]
    params = ", ".join(f"{n}: Scratchpad[np.int32] = np.int32(0)" for n in names)
    src = (
        "def body(self, ids=None, *, token: Carried[np.int32], position: Carried[np.int32], "
        "start: Scratchpad[np.int32] = np.int32(0), rows: Scratchpad[np.int32] = np.int32(1), "
        f"rope_start: Scratchpad[np.int32] = np.int32(0), {params}):\n"
        "    values = dict(" + ", ".join(f"{n}={n}" for n in names) + ")\n"
        "    return self.moe_body(values, ids=ids, token=token, position=position, start=start, "
        "rows=rows, rope_start=rope_start)\n"
    )
    ns = dict(Carried=Carried, Scratchpad=Scratchpad, np=np)
    exec(src, ns)
    return type(base.__name__, (base,), dict(body=ns["body"], _values_made=frozenset(names)))


if __name__ == "__main__":
    lm.main(Qwen35)
