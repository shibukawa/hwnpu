"""Qwen3 and Qwen2 dense models (``qwen3``: Qwen3-0.6B, 1.7B; ``qwen2``: Qwen2.5-Coder-7B-Instruct)
on IRON's shared decoder (``iron.lm``).

An application of amd/IRON kept in hwnpu (IRON has none for Qwen3 or Qwen2), exported by
``iron_export_lm.py qwen3`` (``qwen2``). Its shape is read from the checkpoint's ``config.json``;
the structure follows transformers' modeling_qwen3.py and modeling_qwen2.py:

- Every layer is pre-normed attention then a SwiGLU MLP (IRON's ``DecoderLayer``), RMSNorm by
  ``weight`` (not ``1 + weight``), rotate_half RoPE of ``rope_theta`` over the whole head.
- Qwen3: q and k RMS-normed per head before RoPE (``q_norm``, ``k_norm``), no biases.
- Qwen2: biases on q, k and v (none on o), no norms over the heads.
- The head is the embeddings where ``tie_word_embeddings`` (Qwen3's small sizes), else
  ``lm_head`` (Qwen2.5-Coder-7B).

q, k and v are one projection (``qkv``, rows q, k, v; Qwen2's biases one vector): a decode
step's one QGEMV whose output is cut, the heads of q then k normed by one operator (iron_headnorm)
and turned by one RoPE, as Qwen3.5's gated attention (iron_qwen35.py). In a decode step the
residual adds and the norms are made in the GEMVs that read the norms (iron_q4.sandwich_linear
without a post norm: the qkv projection, the MLP's gate and up, the head), as LFM3's sandwiches
(iron_lfm3.py); Qwen2's biased qkv takes its norm alone and down_proj adds its residual.

With install() (``iron_export_lm.py qwen3 --q4``) the attention's projections are in ``attn`` (q8 by
default), the MLP's in ``mlp`` and the head in ``head`` (iron_q4's formats); the MLP's gate and up
are one table (QGEMVGlu), the head's rows padded to whole blocks of the 8 columns, and the token
embeddings go to the host (iron_hostple's embed.bin, as Qwen3.5's).
"""

import dataclasses
import functools
import os

import numpy as np
from ml_dtypes import bfloat16

import iron
from iron import Precision, lm
from iron.lm import Attention, CausalLM, DecoderConfig, DecoderLayer, Linear

import iron_verify
from iron.operators.copy import Copy

# A q6 or q8 MLP's gate and up as one table (iron_q4's glu_param), as iron_qwen35.FUSE_GLU.
FUSE_GLU = True

# install()'s formats (iron_q4's), by part: the attention's projections, the MLP's, the head.
# Empty: all bf16.
QUANT: dict[str, str] = {}

# A prompt chunk's commands are cut at this many scratchpad words (as iron_qwen35's dense
# models: IRON's 32 left a command's control code over the 64 MiB device heap on Qwen3.5-2B).
SCRATCHPAD_WORDS = 24

# A prompt chunk's command starts every LAYERS_PER_COMMAND layers (install_cuts): attention
# layers ask few scratchpad words, so IRON's own cuts left Qwen3-0.6B's 1024-row chunk two
# commands of 66 MB of control code, which the 64 MiB device heap cannot take (CREATE_BO:
# resource temporarily unavailable), as Ouro's (iron_ouro.install_cuts).
# HWNPU_QWEN3_LAYERS_PER_COMMAND=0 leaves the cuts to IRON's packaging alone.
LAYERS_PER_COMMAND = int(os.environ.get("HWNPU_QWEN3_LAYERS_PER_COMMAND", "7"))

# The head's rows are padded to a multiple of this (64-row blocks over 8 columns).
HEAD_ALIGN = 512

# The last step recorded before each cut's layer (Qwen3.layer), for install_cuts.
_CUT_AFTER: list = []


def install(attn: str = "q8", mlp: str = "q4", head: str = "q4") -> None:
    """Build Qwen3 / Qwen2 with their projections in these formats ("q4", "q6", "q8" or "bf16"),
    F.linear running iron_q4's on them, the token embeddings on the host."""
    import iron_hostple
    import iron_q4
    from iron.common.image import packaging

    for part, fmt in dict(attn=attn, mlp=mlp, head=head).items():
        if fmt not in ("q4", "q6", "q8", "bf16"):
            raise ValueError(f"{part} in {fmt}: q4, q6, q8 or bf16")
    iron_q4.install_linear()
    QUANT.update(attn=attn, mlp=mlp, head=head)
    packaging.SCRATCHPAD_WORDS = min(packaging.SCRATCHPAD_WORDS, SCRATCHPAD_WORDS, iron_hostple.SCRATCHPAD_WORDS)


def install_cuts() -> None:
    """Cut a prompt chunk's commands at SCRATCHPAD_WORDS scratchpad words and every
    ``LAYERS_PER_COMMAND`` layers, in IRON's packaging. The decode step (no tensor input: IRON
    keeps it one command) is left as it is."""
    import iron.common.graph.compiled as compiled
    from iron.common.image import packaging

    packaging.SCRATCHPAD_WORDS = min(packaging.SCRATCHPAD_WORDS, SCRATCHPAD_WORDS)
    if LAYERS_PER_COMMAND <= 0 or getattr(compiled.plan, "_hwnpu_qwen3", False):
        return
    plan = compiled.plan

    def cut_plan(dev, recording, *args, **kwargs):
        chosen = plan(dev, recording, *args, **kwargs)
        steps = recording.steps
        cuts = {i + 1 for i, s in enumerate(steps) if any(s is t for t in _CUT_AFTER)}
        cuts = {i for i in cuts if 0 < i < len(steps)}
        if not cuts or not chosen.full_elf or not recording.inputs:
            return chosen
        commands = tuple(sorted(set(chosen.commands) | cuts))
        reasons = [*chosen.reasons, f"Qwen3: a command starts every {LAYERS_PER_COMMAND} layers: {sorted(cuts)}"]
        return dataclasses.replace(chosen, reasons=reasons, commands=commands)

    cut_plan._hwnpu_qwen3 = True
    compiled.plan = cut_plan


def _mark_cut() -> None:
    """Note that a command starts after the last step recorded so far (install_cuts)."""
    from iron.common.declare.operator import graph_recorder

    recorder = graph_recorder.get()
    if recorder is not None and recorder.steps:
        _CUT_AFTER.append(recorder.steps[-1])


@dataclasses.dataclass(frozen=True)
class Qwen3Config(DecoderConfig):
    """``DecoderConfig`` with Qwen3's norms over the heads (``head_norms``) or Qwen2's q, k and v biases."""

    head_norms: bool = lm.not_json(True, kw_only=True)
    qkv_bias: bool = lm.not_json(False, kw_only=True)
    precision: Precision = lm.not_json(Precision.MEDIUM, kw_only=True)

    @classmethod
    def from_json(cls, data: dict, source="config.json") -> "Qwen3Config":
        """``DecoderConfig.from_json``.

        Raises:
            ValueError: ``data`` asks for what is not built: another model, scaled RoPE, a
                sliding window, an activation other than SiLU, or (Qwen3) biases.
        """
        kind = data.get("model_type")
        if kind not in ("qwen3", "qwen2"):
            raise ValueError(f"{source}: model_type {kind!r} is not qwen3 or qwen2")
        if data.get("rope_scaling"):
            raise ValueError(f"{source}: rope_scaling {data['rope_scaling']} is not built here")
        if data.get("use_sliding_window"):
            raise ValueError(f"{source}: use_sliding_window is not built here")
        if data.get("hidden_act", "silu") != "silu":
            raise ValueError(f"{source}: hidden_act {data['hidden_act']!r} is not built here (SwiGLU)")
        if kind == "qwen3" and data.get("attention_bias"):
            raise ValueError(f"{source}: Qwen3 with attention biases is not built here")
        # Qwen2's q, k and v biases are the model's own (no attention_bias in its config); none on
        # o, which DecoderConfig's attention_bias would add
        text = {**data, "attention_bias": False, "use_sliding_window": False}
        c = super().from_json(text, source)
        return dataclasses.replace(c, head_norms=kind == "qwen3", qkv_bias=kind == "qwen2")


class Qwen3Attention(Attention):
    """``Attention`` over one fused projection of q, k and v (``qkv``), q and k normed per head
    (Qwen3) or the three biased (Qwen2)."""

    def __init__(self, c: Qwen3Config, i: int):
        super().__init__(c, i, norms="qk" if c.head_norms else "")

    def projections(self, c) -> None:
        super().projections(c)
        if c.qkv_bias:
            E, H, G, D = c.hidden_size, self.heads, self.groups, self.head_dim
            self.q_proj = Linear(E, H * D, True, precision=c.precision)
            self.k_proj = Linear(E, G * D, True, precision=c.precision)
            self.v_proj = Linear(E, G * D, True, precision=c.precision)

    def load_state_dict(self, tree, *, path: str = "") -> None:
        super().load_state_dict(tree, path=path)
        H, G, D = self.heads, self.groups, self.head_dim
        lins = (self.q_proj, self.k_proj, self.v_proj)
        # q, k and v as one projection (rows in that order): a decode step's one QGEMV whose
        # output is cut, a prompt chunk's GEMM on each run of its rows (Linear.split)
        self.runs = (H * D, G * D, G * D)
        E = np.asarray(self.q_proj.weight).shape[-1]
        bias = self.q_proj.bias is not None
        self.qkv = Linear(E, sum(self.runs), bias, precision=self.q_proj.precision)
        self.qkv._bound = True
        self.qkv.weight = iron.param(np.ascontiguousarray(np.concatenate([np.asarray(p.weight) for p in lins])))
        if bias:
            self.qkv.bias = iron.param(np.ascontiguousarray(np.concatenate([np.asarray(p.bias) for p in lins])))
        self.q_proj = self.k_proj = self.v_proj = None

    def project(self, x, **gemv) -> list:
        return self.qkv.split(x, *self.runs)

    def fast(self, x, angles) -> bool:
        """Whether a call on ``x`` runs the decode step's attention (decode_from) on a QGEMV."""
        import iron_q4

        if x.shape[0] != 1 or angles is None or angles[0] is not angles[1] or self.v_norm or iron_verify.is_rows(x):
            return False
        entry = iron_q4._Q4.get(id(self.qkv.weight))
        return entry is not None and entry[0] is self.qkv.weight  # bf16: no QGEMV to cut

    def body(self, x, cache, call, angles=None):
        """A decode step's attention on the fused projection (decode_from)."""
        if not self.fast(x, angles):
            return super().body(x, cache, call, angles)
        import iron_q4

        _, rows, K, fmt = iron_q4._Q4[id(self.qkv.weight)]
        adds = () if self.qkv.bias is None else (self.qkv.bias,)
        whole = iron_q4.q_linear(x.reshape(K), self.qkv.weight, rows, K, fmt, adds=adds)
        return self.decode_from(whole, cache, call, angles, x)

    def decode_from(self, whole, cache, call, angles, x):
        """A decode step's attention from the fused projection's output ``whole`` (q, k, v): q and
        k are one contiguous run of it, normed per head by one operator (iron_headnorm, Qwen3) and
        turned by one RoPE, where the step's q and k take the same angles; then o_proj."""
        import iron_headnorm

        H, G, D = self.heads, self.groups, self.head_dim
        QK = (H + G) * D
        whole = whole.reshape(QK + G * D)
        v = whole[QK:]
        if self.q_norm is not None:
            if getattr(self, "_qk", None) is None:
                # q's heads then k's, normed by one operator (their weights one row), for one RoPE
                qn = np.tile(np.asarray(self.q_norm.weight, np.float32), H)
                kn = np.tile(np.asarray(self.k_norm.weight, np.float32), G)
                self._qkn = iron.param(np.concatenate([qn, kn]).astype(bfloat16))
                self._qk = iron.state((QK,), bfloat16)
            iron_headnorm.HeadRMSNorm.step(whole[:QK].reshape(1, QK), self._qk.reshape(1, QK), weight=self._qkn,
                                           epsilon=self.eps, head_dim=D)
            qk = self._qk
        else:
            qk = whole[:QK]
        qk = self.rotate(qk.reshape(H + G, D), angles[0])
        q, k = qk[:H].reshape(1, H, D), qk[H:].reshape(1, G, D)
        o = cache.attend(call, q, k, v.reshape(1, G, D), scale=self.scaling)
        return self.output(o, x)


# A prompt chunk's gate and up GEMMs of more out-features than this run in pieces: Qwen2.5-Coder-7B's
# 18944 (296 column tiles) made ProjectionGEMM's per-call output pattern more than one buffer
# descriptor takes.
GLU_PIECE = 12288


class Qwen3Layer(DecoderLayer):
    """Layer ``i``: attention then the SwiGLU MLP, pre-normed."""

    def __init__(self, c: Qwen3Config, i: int):
        super().__init__(c, i, Qwen3Attention(c, i))

    def ffn(self, hn, residual=None, g=None):
        """The MLP on the normed rows ``hn`` (or from its gate-up activations ``g``, one row), plus
        ``residual`` where given (a decode step's, added in the last GEMV). An MLP of more than
        GLU_PIECE out-features (``mlp.pieces``, Qwen3.quantize) runs gate and up in a prompt chunk,
        and down always, in pieces of in-features (each its own table)."""
        import iron.functional as F
        import iron_q4
        from iron import Precision

        mlp = self.mlp
        pieces = getattr(mlp, "pieces", None)
        if pieces is None:
            if g is None:
                if residual is None:
                    return mlp(hn)
                with iron_q4.into_linear(mlp.down_proj, residual) as st:
                    y = mlp(hn)
            else:
                if residual is None:
                    return mlp.down_proj(g)
                with iron_q4.into_linear(mlp.down_proj, residual) as st:
                    y = mlp.down_proj(g)
            return y if st["used"] else residual.reshape(y.shape) + y
        _, I, K, fmt, act = iron_q4._GLU[id(mlp.gu)]
        one_row = g is not None or len(hn.shape) == 1 or hn.shape[0] == 1
        if one_row:
            if g is None:
                g = iron_q4.glu_linear(hn, mlp.gu)
            g = g.reshape(I)
            y = None
            for i, (a, b, w) in enumerate(pieces):
                last = residual if i == len(pieces) - 1 else None
                y = iron_q4.linear_add(g[a:b], w, *[v.reshape(int(np.prod(v.shape))) for v in (y, last) if v is not None])
            return y.reshape(1, iron_q4._Q4[id(pieces[0][2])][1])
        few = hn.shape[0] <= iron_q4.ROWS_GEMV  # a call checking drafts: QGEMVRows
        g = iron_q4.glu_linear(hn, mlp.gu) if few else None
        ops, q = iron_q4.glu_operators(), mlp.gu.reshape(-1, iron_q4.FORMATS[fmt])
        per = q.shape[0] // (I // iron_q4.CB)  # blocks of a column block's gate and up rows
        y = None
        for a, b, w in pieces:
            if few:
                ga = g[:, a:b]
            else:
                part = q[a // iron_q4.CB * per : b // iron_q4.CB * per]
                gp, up = (iron_q4._BASE["linear"](hn, ops["dequant"].step(part, fmt=fmt, K=K, parity=par,
                                                                          blocks=part.shape[0] // 2),
                                                   precision=Precision.MEDIUM) for par in (0, 1))
                ga = F.silu(gp) * up
            _, n, k, f = iron_q4._Q4[id(w)]
            t = iron_q4.q_linear(ga, w, n, k, f)
            y = t if y is None else y + t
        return y if residual is None else residual + y

    def body(self, x, call):
        # A decode step's two residuals added in the blocks' last projections' GEMVs (iron_q4's
        # into_linear: o_proj and down_proj), not by adds of their own (Qwen3.fused runs the
        # decode step's layers by fused_body instead where it can)
        if call.prompt or x.shape[0] != 1 or not QUANT or iron_verify.is_rows(x):
            from iron.lm.blocks import residual_add

            h = residual_add(x, self.mix(self.input_layernorm(x), call), self.residual_multiplier)
            return residual_add(h, self.ffn(self.post_attention_layernorm(h)), self.residual_multiplier)
        import iron_q4

        with iron_q4.into_linear(self.self_attn.o_proj, x) as st:
            y = self.mix(self.input_layernorm(x), call)
        h = y if st["used"] else x + y
        return self.ffn(self.post_attention_layernorm(h), h)

    def fused_body(self, x, pend, call):
        """A decode step's layer on the residual ``x`` and ``pend``: ("done",) when ``x`` is whole, or
        ("pending", h) with the last MLP's output h not yet added. The residual add and the next
        norm are made in the GEMV that reads the norm (iron_q4.sandwich_linear without a post norm:
        the qkv projection where it has no bias, the MLP's QGEMVGlu), o_proj adding nothing.
        Returns the residual and the next pend: ("pending", down_proj's output), or ("done",) with
        Qwen2's biased qkv, whose residual down_proj adds (into_linear)."""
        import iron_q4

        a, mlp, eps = self.self_attn, self.mlp, self.input_layernorm.eps
        angles = call.ropes[self.rope]
        if pend[0] == "pending":
            whole, x, _ = iron_q4.sandwich_linear(pend[1], x, None, self.input_layernorm.weight, eps, a.qkv.weight)
            h = a.decode_from(whole, self.cache, call, angles, x)
        else:
            h = self.mix(self.input_layernorm(x), call)
        g, x, _ = iron_q4.sandwich_linear(h, x, None, self.post_attention_layernorm.weight, eps, mlp.gu)
        if a.qkv.bias is not None:
            return self.ffn(None, x, g=g).reshape(x.shape), ("done",)
        return x, ("pending", self.ffn(None, g=g))


class Qwen3(CausalLM):
    """Qwen3 (or Qwen2) on ``config``'s shape and a checkpoint's ``weights``."""

    config = Qwen3Config
    layer_class = Qwen3Layer
    hwnpu_name = "Qwen3"

    def __init__(self, config: Qwen3Config, weights):
        if config.head_norms != (self.hwnpu_name == "Qwen3"):
            kind = "qwen3" if config.head_norms else "qwen2"
            raise ValueError(f"{type(self).__name__}: a {kind} checkpoint (iron_export_lm.py {kind})")
        super().__init__(config, weights)
        self.host_embed = False
        if QUANT:
            self.quantize(QUANT)
            self.embed_on_host()

    def fused(self, call, x) -> bool:
        """Whether this call's layers run fused_body (a decode step's row, quantized, gate and up
        one table, every norm of one eps)."""
        c = self.config
        return (not call.prompt and x.shape[0] == 1 and bool(QUANT) and not iron_verify.is_rows(x)
                and all(getattr(l.mlp, "gu", None) is not None for l in self.layers)
                and all(l.self_attn.fast(x, call.ropes[l.rope]) for l in self.layers)
                and all(n.eps == c.eps for l in self.layers for n in (l.input_layernorm, l.post_attention_layernorm))
                and self.model.norm.eps == c.eps)

    def layer(self, call, i, w, x):
        if i and LAYERS_PER_COMMAND > 0 and i % LAYERS_PER_COMMAND == 0:
            _mark_cut()
        if not self.fused(call, x):
            return super().layer(call, i, w, x)
        x, self._pend = w.fused_body(x, ("done",) if i == 0 else self._pend, call)
        return x

    def head(self, x):
        pend, self._pend = getattr(self, "_pend", None), None
        if pend is not None and pend[0] == "pending":
            import iron_q4

            # the last layer's residual and the final norm made in the head's GEMV
            norm = self.model.norm
            logits, _, _ = iron_q4.sandwich_linear(pend[1], x, None, norm.weight, norm.eps, self.lm_head.weight)
        else:
            logits = super().head(x)
        return logits

    def draw(self, logits, call):
        """CausalLM.draw on the head's rows padded to whole blocks of the 8 columns (quantize): the
        output holds them all (hwnpu reads the vocabulary's alone), Sample draws from the
        vocabulary's."""
        from aie.iron.kernels.sample import ROW_WORDS
        from iron.operators.sample import Sample

        c, n = self.config, int(np.prod(logits.shape))
        if n == c.vocab_size + c.vocab_size % 2:
            return super().draw(logits, call)
        _, drawn = Sample.step(logits.reshape(n), self.draws, self.drawn, row=call.position * ROW_WORDS,
                               at=call.position, drawable=c.vocab_size)
        return logits, iron.carry(token=drawn, position=call.position + 1)

    def quantize(self, fmts: dict) -> None:
        """Put the loaded projections in ``fmts``' formats (``install``)."""
        import iron_q4

        def q(holder, fmt):
            if fmt != "bf16":
                holder.weight = iron_q4.q_param(np.asarray(holder.weight, np.float32), fmt)

        for layer in self.layers:
            a, mlp = layer.self_attn, layer.mlp
            for lin in (a.qkv, a.o_proj):
                q(lin, fmts["attn"])
            if fmts["mlp"] != "bf16" and FUSE_GLU:
                # gate and up as one table: the decode step's one operator (iron_q4's QGEMVGlu) in
                # place of two GEMVs, a SiLU and a product; a prompt chunk reads it in parts
                iron_q4.install_glu()
                mlp.gu = iron_q4.glu_param(np.asarray(mlp.gate_proj.weight, np.float32),
                                           np.asarray(mlp.up_proj.weight, np.float32), fmts["mlp"])
                mlp.gate_proj = mlp.up_proj = None
                I = mlp.down_proj.weight.shape[1]
                if I > GLU_PIECE:
                    # down in pieces of in-features, whole 512s (DequantBFP's), each its own table
                    w = np.asarray(mlp.down_proj.weight, np.float32)
                    cut = I // 2 // 512 * 512
                    mlp.pieces = [(a, b, iron_q4.q_param(np.ascontiguousarray(w[:, a:b]), fmts["mlp"]))
                                  for a, b in ((0, cut), (cut, I))]
                    mlp.down_proj = None
                    continue
                q(mlp.down_proj, fmts["mlp"])
                continue
            for lin in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                q(lin, fmts["mlp"])
        # the head its own quantized copy where it is tied: the embeddings stay bf16. Its rows padded
        # with zeros to whole blocks of the 8 columns (HEAD_ALIGN): Qwen3's 151936 is 2374 blocks of
        # 64, which a QGEMV ran on 2 columns
        if fmts["head"] != "bf16":
            w = np.asarray(self.lm_head.weight, np.float32)
            pad = -len(w) % HEAD_ALIGN
            self.lm_head.weight = iron_q4.q_param(np.pad(w, ((0, pad), (0, 0))), fmts["head"])

    def embed_on_host(self) -> None:
        """The token embeddings to the host (iron_hostple's embed.bin)."""
        import iron_hostple

        c, m = self.config, self.model
        embed = np.asarray(m.embed_tokens.weight).astype(bfloat16)
        iron_hostple._TABLES.setdefault(id(self), {})[iron_hostple.EMBED] = embed
        m.embed_tokens.weight = iron.param(np.zeros((1, c.hidden_size), bfloat16))  # no longer read
        self.embed_rows = iron.state((c.prefill_chunk, c.hidden_size), bfloat16, name=iron_hostple.EMBED)
        self.host_embed = True

    def lookup(self, ids):
        # The embeddings from embed_rows, which the host filled with the call's tokens in order.
        if not self.host_embed:
            return super().lookup(ids)
        shape = getattr(ids, "shape", ())
        if len(shape) == 0:
            return Copy.step(self.embed_rows[:1])
        bound = getattr(ids, "valid", None) or {}
        return Copy.step(self.embed_rows[: bound[0]] if 0 in bound else self.embed_rows[: shape[0]])

    @functools.wraps(CausalLM.body)
    def body(self, ids=None, **values):
        if ids is not None and self.host_embed:
            # a version's input must be read, and no table is left to gather from (iron_hostple)
            Copy.step(ids, dtype=np.int32)
        return super().body(ids, **values)


class Qwen2(Qwen3):
    """Qwen2 (Qwen2.5-Coder): Qwen3's layers with biases on q, k and v and no norms over the heads."""

    hwnpu_name = "Qwen2"  # the manifest's model: hwnpu's chat format for Qwen2.5


if __name__ == "__main__":
    lm.main(Qwen3)
