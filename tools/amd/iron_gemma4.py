"""Gemma 4 whose prompt chunks skip the KV-shared layers for every row but the last.

An application kept in hwnpu over amd/IRON's Gemma 4 (``iron.applications.gemma4``),
exported by ``iron_export_lm.py gemma4`` and ``gemma4mm``.

Gemma 4's last ``num_kv_shared_layers`` layers (20 of E2B's 35) write no keys or
values: they attend over the cache of the last earlier layer of their kind. A prompt
chunk keeps only the caches, plus the logits after its last row, so the shared layers'
outputs matter only for that last row. A chunk therefore runs every row through the
layers that own a cache, then the last row alone through the shared layers, as the
decode step runs them (one query at its position over the owners' caches, already
holding the chunk's rows). The caches and the logits are the full chunk's; the
shared layers' work is one row's, not ``rows``'.

With iron_hostple (the per-layer token table on the host), the host writes every row's
per-layer inputs as for any chunk; the last row reads its own (``_Tail.row``). Its chunk keeps
IRON's 32 scratchpad words a command (``SCRATCHPAD_WORDS``), not iron_hostple's 24.
"""

import dataclasses
from typing import Any

from iron.applications.gemma4.audio import AudioTower
from iron.applications.gemma4.model import Gemma4, Gemma4Multimodal
from iron.applications.gemma4.vision import VisionTower
from iron.lm import Multimodal
from iron.lm.causal import Call
from iron.operators.copy import Copy

# The scratchpad words of a prompt chunk's command with iron_hostple. Its 24 splits the full
# model's long commands (iron_hostple.SCRATCHPAD_WORDS); here the shared layers are light (the
# slowest command 0.87 s at position 3072 for E2B, 6 commands), and 24 cut them in two: then the
# next chunk's first command hung on the device until amdxdna's timeout (a decode step between
# chunks hid it). 32 runs them in one command.
SCRATCHPAD_WORDS = 32


@dataclasses.dataclass(kw_only=True)
class _Tail(Call):
    """A prompt chunk's last row run as a decode step: ``row``, its row in the chunk."""

    row: Any = None


class Gemma4Tail(Gemma4):
    """``Gemma4`` whose prompt chunks run the shared layers on their last row alone."""

    hwnpu_name = "Gemma4"  # the manifest's model: hwnpu's chat format for Gemma 4

    @property
    def first_shared(self) -> int:
        c = self.config
        return c.num_hidden_layers - c.num_kv_shared_layers

    def from_rows(self, x, ids, *, token, position, start, rows, rope_start):
        # As CausalLM.from_rows, with the token the tail gathers its per-layer rows by.
        return self.forward(
            x,
            Call(prompt=True, start=start, rows=rows, position=position, rope_start=rope_start, ids=ids, token=token),
        )

    def forward(self, x, call: Call):
        first = self.first_shared
        if not call.prompt or call.token is None or first >= len(self.layers):
            return super().forward(x, call)
        E = self.config.hidden_size
        call.embeddings = x
        for rope in (layer.rope for layer in self.layers):
            if rope is not None and rope not in call.ropes:
                call.ropes[rope] = rope.rows(call)
        x = self.positions(x, call)
        for i in range(first):
            x = self.layer(call, i, self.layers[i], x)
        # The last row, as a decode step at the chunk's last position: its embedding
        # (the per-layer inputs' projection reads it) and its token (their gather).
        last = call.rows - 1
        tail = _Tail(
            prompt=False,
            start=call.position,
            rows=call.rows,
            position=call.position,
            rope_start=call.rope_start,
            ids=call.token,
            token=call.token,
            embeddings=Copy.step(call.embeddings[last]).reshape(1, E),
            row=last,
        )
        for rope in (layer.rope for layer in self.layers[first:]):
            if rope is not None and rope not in tail.ropes:
                tail.ropes[rope] = rope.rows(tail)
        x = Copy.step(x[last]).reshape(1, E)
        for i in range(first, len(self.layers)):
            x = self.layer(tail, i, self.layers[i], x)
        return self.draw(self.head(x), call)


class Gemma4TailMultimodal(Gemma4Multimodal):
    """``Gemma4Multimodal`` on ``Gemma4Tail``."""

    def __init__(self, config, weights):
        assert config.vision is not None and config.audio is not None
        assert config.image_token_id is not None and config.audio_token_id is not None
        self.config, self.weights = config, weights
        towers = {
            config.image_token_id: VisionTower(config.vision, weights.model),
            config.audio_token_id: AudioTower(config.audio, weights.model),
        }
        Multimodal.__init__(self, Gemma4Tail(config, weights), towers)

