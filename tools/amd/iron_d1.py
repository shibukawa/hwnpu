"""d1 (Liquid AI's d1-2B and d1-Omni-2B, the System One decision models) on IRON.

An application of amd/IRON kept in bonsaiai, exported by ``iron_export_lm.py d1`` (d1-2B: LFM3-VL's
checkpoint layout, ``model_type`` lfm3_vl) and ``d1omni`` (d1-Omni-2B: LFM3-Omni's). The models
are LFM3-VL and LFM3-Omni (``iron_lfm3vl``, ``iron_lfm3omni``) read as d1 reads them
(``architectures`` D1Model, ``d1_attention`` "set"), as llama.cpp and bonsaiai's Core ML programs do:

- one call reads the whole prompt from position 0: the full attention layers attend both ways
  over the call's rows (``SetCache``), the sliding attention layers and the short convolutions
  stay causal;
- the output is the final hidden state of the last row (the final norm, no head): the host reads
  the logits of the answer labels from it and the rows of the token embeddings (tied to the head),
  which a ``--q4`` export holds on the host (``embed.bin``) as LFM3's do.

So bonsaiai runs the export's prompt chunk once a question, from 0, with a prompt of at most a chunk
(``--prefill-chunk``, ``--max-seq-len`` the same); images and clips go through the towers and the
placed chunk as LFM3-VL's and LFM3-Omni's. The decode step is built (IRON's ``CausalLM`` has one)
and never run.
"""

import numpy as np
from ml_dtypes import bfloat16

import iron
from iron.lm import KVCache, Multimodal, subtree
from iron.operators.mha import MHA

import iron_lfm3
from iron_lfm3 import Lfm3, Lfm3Layer
from iron_lfm3omni import Lfm3OmniConfig, OmniAudioTower
from iron_lfm3vl import Lfm3VlConfig, VisionTower


class SetCache(KVCache):
    """A full attention layer's ``KVCache`` that a prompt chunk attends both ways (d1's "set").

    The chunk's rows are written as ``KVCache`` writes them and every query attends to every key
    of the call (MHA with ``causal`` off, its keys bounded by the call's rows): of the whole
    prompt when the call starts at 0, which is how bonsaiai runs a d1 export. A decode step attends
    as ``KVCache`` does (causal; never run).
    """

    def attend(self, call, q, k, v, scale: float | None = None):
        if not call.prompt or self.ring is not None:
            return super().attend(call, q, k, v, scale)
        C, (n, H, D) = self.chunk, q.shape
        for x, cache in ((k, self.keys), (v, self.values)):
            cache[call.start : call.start + C][: call.rows] = x
        keys, values = (x[: call.position + 1] for x in (self.keys, self.values))
        o = MHA.step(q, keys, values, heads_interleaved=True, kv_interleaved=True, causal=False, scale=scale)
        return o.reshape(n, H * D)


class D1Layer(Lfm3Layer):
    """``Lfm3Layer``, a full attention layer's cache a ``SetCache``."""

    def __init__(self, c, i: int):
        super().__init__(c, i)
        if self.kind == "full_attention":
            self.cache = SetCache(c, None)


class D1(Lfm3):
    """LFM3's text model as d1 reads it: ``D1Layer``s, and the final norm as the output (no head)."""

    layer_class = D1Layer

    def head(self, x):
        pend, self._pend = getattr(self, "_pend", None), None
        norm = self.model.norm
        if pend is None:
            return norm(x)
        import iron_sandwich

        # a decode step (never run): the last layer's sandwich, its next norm the final norm
        _, h, wpost = pend
        _, xn = iron_sandwich.sandwich_add(h, x, wpost, norm.weight, norm.eps)
        return xn

    def draw(self, hidden, call):
        # No token drawn on the device (the output is no logits); bonsaiai reads the hidden state.
        # A placed chunk's call carries no token (IRON's from_rows): its position stands in.
        token = call.token if getattr(call, "token", None) is not None else call.position
        return hidden, iron.carry(token=token + 0, position=call.position + 1)

    def quantize(self, fmts: dict) -> None:
        """``Lfm3.quantize`` with no head on the device: the embeddings (tied to the head) go to
        the host as a quantized head's would, whose rows the host reads for the labels too."""
        import iron_hostple

        super().quantize(dict(fmts, head="bf16"))
        c = self.config
        embed = np.asarray(self.model.embed_tokens.weight)
        iron_hostple._TABLES.setdefault(id(self), {})[iron_hostple.EMBED] = embed.astype(bfloat16)
        unread = iron.param(np.zeros((1, c.hidden_size), bfloat16))
        self.model.embed_tokens.weight = unread
        self.lm_head.weight = unread
        self.embed_rows = iron.state((c.prefill_chunk, c.hidden_size), bfloat16, name=iron_hostple.EMBED)


class D1Vl(Multimodal):
    """d1-2B: ``D1`` with LFM3-VL's vision tower, an image's soft tokens at its placeholders."""

    config = Lfm3VlConfig
    bonsaiai_kind = "lfm3vl"  # the towers and the placed chunk as LFM3-VL's (iron_export_lm.py)

    def __init__(self, config: Lfm3VlConfig, weights):
        assert config.vision is not None
        self.config, self.weights = config, weights
        towers = {config.image_token_id: VisionTower(config.vision, weights.model)}
        text = D1(config, subtree(weights, model=weights.model.language_model))
        super().__init__(text, towers)


class D1Omni(Multimodal):
    """d1-Omni-2B: ``D1`` with LFM3-Omni's vision tower and audio encoder."""

    config = Lfm3OmniConfig
    bonsaiai_kind = "lfm3omni"

    def __init__(self, config: Lfm3OmniConfig, weights):
        assert config.vision is not None and config.audio is not None
        self.config, self.weights = config, weights
        towers = {
            config.image_token_id: VisionTower(config.vision, weights.model),
            config.audio_token_id: OmniAudioTower(config.audio, weights.model),
        }
        text = D1(config, subtree(weights, model=weights.model.language_model))
        super().__init__(text, towers)


# the exporter's q4 install (iron_lfm3.install_q4) is the LFM3 family's: kept for d1 too
install_q4 = iron_lfm3.install_q4
