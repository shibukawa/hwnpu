"""Ouro (ByteDance's OuroForCausalLM, e.g. ByteDance/Ouro-1.4B-Thinking) on IRON's shared decoder (``iron.lm``).

An application of amd/IRON kept in hwnpu (IRON has none for Ouro), exported by
``iron_export_lm.py ouro`` like PLaMo 3's (iron_plamo3.py). Its shape is read
from the checkpoint's ``config.json``; the structure follows its remote code
``modeling_ouro.py``:

- A looped decoder: the whole stack of ``num_hidden_layers`` layers runs
  ``total_ut_steps`` times on each token (4: Ouro-1.4B's 24 layers make 96
  layer runs a token), the same weights each pass. After every pass the final
  norm (``model.norm``) is applied, and that normed state is what the next pass
  starts from; the head reads the last pass's.
- Each pass keeps keys and values of its own for every layer
  (UniversalTransformerCache's slot ``pass * layers + layer``): pass 0's are
  the layers' own ``cache``, the later passes' are in ``passes`` (their buffers
  ``passes.P.layers.L.cache.keys``, which hwnpu reads as KV caches by the
  ``.cache.`` in the name).
- Each layer is sandwich-normed as Gemma's (IRON's ``SandwichLayer``): Ouro's
  ``input_layernorm``, the attention, ``input_layernorm_2``, the residual;
  ``post_attention_layernorm``, the SwiGLU MLP, ``post_attention_layernorm_2``,
  the residual (``renames``). Plain multi-head attention, rotate_half RoPE, no
  biases, no norms over the heads, a head of its own (``lm_head``).
- ``model.early_exit_gate`` is not read: with ``early_exit_threshold`` 1.0 (the
  checkpoints') transformers takes the last pass.

With install() (``iron_export_lm.py ouro --q4``) the attention's projections are in ``attn`` (q8 by
default), the MLP's in ``mlp`` and the head in ``head`` (q4nx by default; iron_q4's formats; q6 is the
catalog's: a weight read four times a token compounds its rounding, docs/design/llm-models.md), a
q6 or q8 MLP's gate and up as one table (FUSE_GLU), and the token embeddings go to the host as PLaMo
3's do (iron_hostple: embed.bin).
"""

import dataclasses
import functools
import os

import numpy as np
from ml_dtypes import bfloat16

import iron
from iron import Precision, lm
from iron.lm import CausalLM, DecoderConfig, KVCache, SandwichLayer
from iron.lm.checkpoint import renamed
from iron.operators.copy import Copy

# A q6 or q8 MLP's gate and up as one table (iron_q4's glu_param), as iron_qwen35.FUSE_GLU.
# HWNPU_OURO_FUSE_GLU=0 keeps them apart.
FUSE_GLU = os.environ.get("HWNPU_OURO_FUSE_GLU", "1") != "0"

# install()'s formats: on, the attention's (q8, q4 or bf16), the MLP's and the head's (q4, q6, q8).
_QUANT = {"on": False, "attn": "q8", "mlp": "q4", "head": "q4"}


def install(attn: str = "q8", mlp: str = "q4", head: str = "q4") -> None:
    """Make Ouro load its projections and head quantized (iron_q4) and its embeddings on the host."""
    import iron_hostple
    import iron_q4
    from iron.common.image import packaging

    if attn not in ("q8", "q4", "bf16"):
        raise ValueError(f"attention in {attn}: q8, q4 or bf16")
    for part, fmt in (("MLP", mlp), ("head", head)):
        if fmt not in ("q4", "q6", "q8"):
            raise ValueError(f"{part} in {fmt}: q4, q6 or q8")
    iron_q4.install_linear()
    _QUANT.update(on=True, attn=attn, mlp=mlp, head=head)
    # Without the gather of the embeddings, a prompt chunk's commands would merge (iron_hostple).
    packaging.SCRATCHPAD_WORDS = min(packaging.SCRATCHPAD_WORDS, iron_hostple.SCRATCHPAD_WORDS)


@dataclasses.dataclass(frozen=True)
class OuroConfig(DecoderConfig):
    """``DecoderConfig`` with Ouro's passes over the layers (``total_ut_steps``)."""

    total_ut_steps: int = dataclasses.field(default=4, kw_only=True)
    precision: Precision = lm.not_json(Precision.MEDIUM, kw_only=True)

    @classmethod
    def from_json(cls, data: dict, source="config.json") -> "OuroConfig":
        """``DecoderConfig.from_json``.

        Raises:
            ValueError: ``data`` asks for what is not built: an exit before the
                last pass, a layer other than full attention, or another model.
        """
        if data.get("model_type") != "ouro":
            raise ValueError(f"{source}: model_type {data.get('model_type')!r} is not ouro")
        if data.get("hidden_act", "silu") != "silu":
            raise ValueError(f"{source}: hidden_act {data['hidden_act']!r} is not built here (SwiGLU)")
        threshold = data.get("early_exit_threshold")
        if threshold is not None and threshold < 1 or data.get("early_exit_step") is not None:
            raise ValueError(f"{source}: an exit before the last pass is not built here")
        for i, t in enumerate(data.get("layer_types") or []):
            if t != "full_attention":
                raise ValueError(f"{source}: layer {i} is {t!r}: only full_attention is built here")
        if data.get("total_ut_steps", 4) < 1:
            raise ValueError(f"{source}: total_ut_steps {data['total_ut_steps']}")
        return super().from_json(data, source)


class OuroLayer(SandwichLayer):
    """Layer ``i``: attention over the cache of the pass running it, then SwiGLU, each sandwich-normed."""

    def mix(self, h, call):
        # the pass's own keys and values (Ouro.layer puts them on the call), pass 0's the layer's
        angles = None if self.rope is None else call.ropes[self.rope]
        return self.self_attn(h, getattr(call, "ouro_cache", None) or self.cache, call, angles)


class _Cached(iron.Graph):
    """A layer's keys and values in a pass after the first."""

    def __init__(self, c):
        self.cache = KVCache(c, c.window)


class _Pass(iron.Graph):
    """The caches of every layer in a pass after the first."""

    def __init__(self, c):
        self.layers = [_Cached(c) for _ in range(c.num_hidden_layers)]


class Ouro(CausalLM):
    """Ouro on ``config``'s shape and a checkpoint's ``weights``."""

    config = OuroConfig
    layer_class = OuroLayer
    hwnpu_name = "Ouro"
    # Ouro's sandwich norms by SandwichLayer's names (whole dotted names: post_attention_layernorm
    # does not take post_attention_layernorm_2)
    renames = {
        "input_layernorm_2": "post_attention_layernorm",
        "post_attention_layernorm": "pre_feedforward_layernorm",
        "post_attention_layernorm_2": "post_feedforward_layernorm",
    }

    def __init__(self, config: OuroConfig, weights):
        # the early exit gate is not read (the last pass's logits): out of a copy of the tree,
        # whose namespaces renamed makes new
        weights = renamed(weights, {})
        if hasattr(weights.model, "early_exit_gate"):
            del weights.model.early_exit_gate
        super().__init__(config, weights)
        c = config
        self.passes = [_Pass(c) for _ in range(c.total_ut_steps - 1)]
        # every layer once a pass: Ouro.layer finds the pass from the index
        self.layers = list(self.model.layers) * c.total_ut_steps
        self.host_embed = False
        if _QUANT["on"]:
            self.quantize(_QUANT["attn"], _QUANT["mlp"], _QUANT["head"])

    def layer(self, call, i, w, x):
        L = self.config.num_hidden_layers
        p, j = divmod(i, L)
        call.ouro_cache = None if p == 0 else self.passes[p - 1].layers[j].cache
        x = w(x, call)
        call.ouro_cache = None
        if j == L - 1 and p < self.config.total_ut_steps - 1:
            x = self.model.norm(x)  # the next pass starts from the normed state
        return x

    def quantize(self, attn: str, mlp: str, head: str) -> None:
        """The projections in iron_q4's formats (the attention's attn, the MLP's mlp, the head's
        head), the token embeddings to the host (iron_hostple's embed.bin)."""
        import iron_hostple
        import iron_q4

        def q(lin, fmt):
            lin.weight = iron_q4.q_param(np.asarray(lin.weight, np.float32), fmt)

        for layer in self.model.layers:
            a, m = layer.self_attn, layer.mlp
            if attn != "bf16":
                for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj):
                    q(lin, attn)
            if mlp in ("q6", "q8") and FUSE_GLU:
                # gate and up as one table: the decode step's one operator (iron_q4's QGEMVGlu) in
                # place of two GEMVs, a SiLU and a product, as Qwen3.5's (iron_qwen35)
                iron_q4.install_glu()
                m.gu = iron_q4.glu_param(np.asarray(m.gate_proj.weight, np.float32),
                                         np.asarray(m.up_proj.weight, np.float32), mlp)
                m.gate_proj = m.up_proj = None
                q(m.down_proj, mlp)
                continue
            for lin in (m.gate_proj, m.up_proj, m.down_proj):
                q(lin, mlp)
        q(self.lm_head, head)
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


if __name__ == "__main__":
    lm.main(Ouro)
