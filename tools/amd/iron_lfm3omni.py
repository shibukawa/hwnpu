"""LFM3-Omni (Liquid AI's LFM3-Omni-2B) on IRON: LFM3 with LFM3-VL's vision tower and LFM3-Audio's audio encoder.

An application of amd/IRON kept in hwnpu, exported by ``iron_export_lm.py lfm3omni``: images
and audio in, text out (the speech output is not built). The text model is ``iron_lfm3.Lfm3``
under ``text_config``, the vision tower ``iron_lfm3vl.VisionTower`` under ``vision_config`` and
the audio encoder ``iron_lfm3audio.AudioTower`` under ``audio_encoder_config`` (its LFM2 stack
under ``audio_encoder_config.text_config``). The checkpoint names its parts as transformers'
modules do (``model.language_model``, ``model.vision_tower``, ``model.audio_encoder``...).
An image's and a clip's placeholders are between ``<|vision_start|>`` and ``<|vision_end|>``.
"""

import dataclasses
from pathlib import Path

from iron import lm
from iron.lm import Multimodal, subtree

from iron_lfm3 import Lfm3, Lfm3Config
from iron_lfm3audio import AudioConfig, AudioTower, _token
from iron_lfm3vl import VisionConfig, VisionTower


@dataclasses.dataclass(frozen=True)
class OmniAudioConfig(AudioConfig):
    """``AudioConfig`` read from LFM3-Omni's ``audio_encoder_config``."""

    json_prefix = "audio_encoder_config."

    mels: int = dataclasses.field(metadata={"json": "num_mel_bins"})
    conv_channels: int = dataclasses.field(metadata={"json": "conv_channels"})
    sliding_window: int = dataclasses.field(metadata={"json": "sliding_window"})
    hidden_size: int = dataclasses.field(metadata={"json": "text_config.hidden_size"})
    ff: int = dataclasses.field(metadata={"json": "text_config.intermediate_size"})
    num_hidden_layers: int = dataclasses.field(metadata={"json": "text_config.num_hidden_layers"})
    num_attention_heads: int = dataclasses.field(metadata={"json": "text_config.num_attention_heads"})
    num_key_value_heads: int = dataclasses.field(metadata={"json": "text_config.num_key_value_heads"})
    layer_types: tuple[str, ...] = dataclasses.field(metadata={"json": "text_config.layer_types", "convert": tuple})
    rope_theta: float = dataclasses.field(default=1e6, metadata={"json": "text_config.rope_parameters.rope_theta"})
    conv_L_cache: int = dataclasses.field(default=3, metadata={"json": "text_config.conv_L_cache"})
    auto_ff: bool = dataclasses.field(default=False, metadata={"json": "text_config.block_auto_adjust_ff_dim"})
    ff_multiplier: float | None = dataclasses.field(default=None, metadata={"json": "text_config.block_ffn_dim_multiplier"})
    ff_multiple_of: int = dataclasses.field(default=256, metadata={"json": "text_config.block_multiple_of"})
    eps: float = dataclasses.field(default=1e-5, kw_only=True, metadata={"json": "text_config.norm_eps"})
    text_dim: int = dataclasses.field(default=2048, kw_only=True, metadata={"json": "/text_config.hidden_size"})


class OmniAudioTower(AudioTower):
    """``AudioTower`` at the names of transformers' modules, as LFM3-Omni's checkpoint holds them."""

    names = dict(
        layers="audio_encoder.lfm.layers.",
        downsample="audio_encoder.downsample.",
        norm="audio_encoder.lfm.embedding_norm.",
        projector=dict(layer_norm="audio_projector.layer_norm.", linear_1="audio_projector.linear_1.",
                       linear_2="audio_projector.linear_2."),
    )


@dataclasses.dataclass(frozen=True)
class Lfm3OmniConfig(Lfm3Config):
    """``Lfm3Config`` under ``text_config``, the placeholders and markers, and both towers' shapes."""

    json_prefix = "text_config."

    image_token_id: int = dataclasses.field(default=125030, kw_only=True, metadata={"json": "/image_token_id"})
    audio_token_id: int = dataclasses.field(default=125031, kw_only=True, metadata={"json": "/audio_token_id"})
    vision_start_token_id: int = lm.not_json(125009, kw_only=True)
    vision_end_token_id: int = lm.not_json(125010, kw_only=True)
    vision: VisionConfig | None = lm.not_json(None, kw_only=True)
    audio: OmniAudioConfig | None = lm.not_json(None, kw_only=True)

    @classmethod
    def from_checkpoint(cls, path) -> "Lfm3OmniConfig":
        path = Path(path)
        d = path if path.is_dir() else path.parent
        text = super().from_checkpoint(path)
        return dataclasses.replace(
            text,
            vision=VisionConfig.from_checkpoint(path),
            audio=OmniAudioConfig.from_checkpoint(path),
            vision_start_token_id=_token(d, "<|vision_start|>"),
            vision_end_token_id=_token(d, "<|vision_end|>"),
        )


class Lfm3Omni(Multimodal):
    """LFM3-Omni with images and audio in: ``Lfm3`` with both towers, their soft tokens at their placeholders."""

    config = Lfm3OmniConfig
    hwnpu_kind = "lfm3omni"  # the multimodal manifest iron_export_lm.py writes

    def __init__(self, config: Lfm3OmniConfig, weights):
        assert config.vision is not None and config.audio is not None
        self.config, self.weights = config, weights
        towers = {
            config.image_token_id: VisionTower(config.vision, weights.model),
            config.audio_token_id: OmniAudioTower(config.audio, weights.model),
        }
        text = Lfm3(config, subtree(weights, model=weights.model.language_model))
        super().__init__(text, towers)
