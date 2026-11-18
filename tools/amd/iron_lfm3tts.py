"""LFM3-Audio's text model for speaking (hwnpu-audio tts) on IRON's shared decoder (``iron.lm``).

``iron_export_lm.py lfm3tts --q4 --mlp q8 --head q8`` exports LFM3-Audio-2B's text model (``lfm.``
of the checkpoint, no audio encoder) as ``Lfm3`` does, with two differences for internal/speech:

- the final norm's output, the hidden state the depthformer reads, is written to the state
  ``hwnpu_hidden`` (1, hidden) before the head, both in a prompt chunk (its last row) and in a
  decode step; the host reads it after each call
- the token embeddings are the host's (``--q4``: iron_hostple's ``hwnpu_embed_rows``), so the
  host writes the row a decode step reads: a token's, or after an audio frame the sum of its
  codes' feedback rows, which is no token's

The head stays (its logits are not read when speaking): the decode step's version is the
shared decoder's.
"""

import dataclasses

import iron
from iron_lfm3 import Lfm3, Lfm3Config
from ml_dtypes import bfloat16

HIDDEN = "hwnpu_hidden"


@dataclasses.dataclass(frozen=True)
class Lfm3TtsConfig(Lfm3Config):
    """``Lfm3Config`` under ``lfm`` (LFM3-Audio's config.json)."""

    json_prefix = "lfm."


class Lfm3Tts(Lfm3):
    """LFM3-Audio's text model, its final hidden state out in ``hwnpu_hidden``."""

    config = Lfm3TtsConfig

    def __init__(self, config: Lfm3TtsConfig, weights):
        # the released checkpoint names the text model lfm.model
        super().__init__(config, weights.lfm)
        self.hidden = iron.state((1, config.hidden_size), bfloat16, name=HIDDEN)

    def head(self, x):
        h = self.model.norm(x)
        self.hidden[0:1] = h
        return self.lm_head(h)
