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


# The step before each recording's tail (its last row through the shared layers), by identity.
_TAIL_AFTER: list = []


def _mark_tail() -> None:
    """Note where the recording's tail starts, for ``install_tail_boundary``."""
    from iron.common.declare.operator import graph_recorder

    recorder = graph_recorder.get()
    if recorder is not None and recorder.steps:
        _TAIL_AFTER.append(recorder.steps[-1])


def install_tail_boundary() -> None:
    """Start a command where a prompt chunk's tail starts, in IRON's packaging.

    IRON starts a command wherever the next step's words would overflow a command's scratchpad
    (``SCRATCHPAD_WORDS``), so the cuts move with every change to the layers' words. A cut inside
    the tail made the next chunk's first command hang (design.md 16.80), as did, with
    iron_window, one that left the head's draw a command of its own (16.84; Go's chunkBarrier
    is what that hang needed). Starting a command exactly at the tail keeps the tail, the head and
    the draw in one command (their words fit), as in the exports measured, wherever the layers
    before put their cuts.
    """
    import iron.common.graph.compiled as compiled
    from iron.common.image import packaging

    if getattr(compiled.plan, "_hwnpu_tail", False):
        return
    plan = compiled.plan

    def tail_plan(dev, recording, *args, step_words=None, **kwargs):
        chosen = plan(dev, recording, *args, step_words=step_words, **kwargs)
        steps = recording.steps
        at = next((i + 1 for i, s in enumerate(steps) if any(s is t for t in _TAIL_AFTER)), None)
        if at is None or not chosen.full_elf or step_words is None or at >= len(step_words):
            return chosen
        hosted = {s.at for s in recording.host if 0 < s.at < len(steps)}
        commands, held = [], frozenset()
        for i, words in enumerate(step_words):
            if i and (i in hosted or i == at or len(held | words) > packaging.SCRATCHPAD_WORDS):
                commands.append(i)
                held = frozenset()
            held |= words
        reasons = [*chosen.reasons, f"Gemma4Tail: a command starts at the tail (step {at}): {commands}"]
        return dataclasses.replace(chosen, reasons=reasons, commands=tuple(commands))

    tail_plan._hwnpu_tail = True
    compiled.plan = tail_plan


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
        _mark_tail()
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

