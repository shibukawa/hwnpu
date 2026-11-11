#!/usr/bin/env python3
"""Anima (circlestone-labs/Anima) on the CPU with diffusers' modules: the reference hwnpu's
NPU pipeline (cmd/hwnpu-image, tools/amd/iron_anima.py) is checked against.

Reads ComfyUI's split files as the official repository ships them (the diffusion model with
its LLM adapter, Qwen3-0.6B, the Qwen-Image VAE) and the configs and tokenizers of
circlestone-labs/Anima-Base-v1.0-Diffusers, all under one directory:

  ANIMA/diffusion_models/anima-turbo-v1.1.safetensors
  ANIMA/text_encoders/qwen_3_06b_base.safetensors
  ANIMA/diffusers/{tokenizer,t5_tokenizer,text_encoder,transformer,vae,text_conditioner,scheduler}/...
  ANIMA/diffusers/vae/diffusion_pytorch_model.safetensors   (the same VAE in diffusers' names)

Writes OUT.npz with every intermediate (token ids, Qwen3's last hidden state, the adapter's
conditioning, the noise, each step's sigma and velocity, the final latents, the image) and
OUT.png. Run in the reference virtual environment (torch, transformers, diffusers 0.42+):

  python tools/ref/ref_anima.py ~/models/Anima OUT --prompt "..." --size 1024 --steps 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

PREFIX = "model.diffusion_model."


def _no_torchvision() -> None:
    """transformer_cosmos resizes the padding mask with torchvision, when installed; nearest
    interpolation in torch is the same (the mask is zeros for a whole image)."""
    from types import SimpleNamespace

    import diffusers.models.transformers.transformer_cosmos as tc

    if not hasattr(tc, "transforms"):
        resize = lambda m, size, interpolation=None: torch.nn.functional.interpolate(m, size=size, mode="nearest")  # noqa: E731
        tc.transforms = SimpleNamespace(functional=SimpleNamespace(resize=resize),
                                        InterpolationMode=SimpleNamespace(NEAREST=None))


def load(root: Path, dit_file: str, dtype):
    _no_torchvision()
    from diffusers import AutoencoderKLQwenImage, CosmosTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers
    from diffusers.models.condition_embedders.condition_embedder_anima import AnimaTextConditioner
    from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

    d = root / "diffusers"
    raw = load_file(root / "diffusion_models" / dit_file)
    prefix = "net." if any(k.startswith("net.") for k in raw) else PREFIX  # anima-base-v1.0, the previews
    sd = {k.removeprefix(prefix): v for k, v in raw.items()}
    adapter = {k.removeprefix("llm_adapter."): v for k, v in sd.items() if k.startswith("llm_adapter.")}
    adapter = {k.replace("mlp.0.", "mlp.0.").replace("mlp.2.", "mlp.2."): v for k, v in adapter.items()}
    dit = convert_cosmos_transformer_checkpoint_to_diffusers({k: v for k, v in sd.items() if not k.startswith("llm_adapter.")})

    tc = AnimaTextConditioner.from_config(json.loads((d / "text_conditioner" / "config.json").read_text()))
    missing, unexpected = tc.load_state_dict(adapter, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    tr = CosmosTransformer3DModel.from_config(json.loads((d / "transformer" / "config.json").read_text()))
    missing, unexpected = tr.load_state_dict(dit, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    qc = Qwen3Config.from_pretrained(d / "text_encoder")
    qwen = Qwen3Model(qc)
    qsd = {k.removeprefix("model."): v for k, v in load_file(root / "text_encoders" / "qwen_3_06b_base.safetensors").items()}
    missing, unexpected = qwen.load_state_dict(qsd, strict=False)
    assert not [m for m in missing if "rotary" not in m] and not unexpected, (missing, unexpected)

    vae = AutoencoderKLQwenImage.from_pretrained(d / "vae")
    tok = AutoTokenizer.from_pretrained(d / "tokenizer")
    t5 = AutoTokenizer.from_pretrained(d / "t5_tokenizer")
    for m in (tc, tr, qwen, vae):
        m.to(dtype).eval()
    return tok, t5, qwen, tc, tr, vae


def sigmas(steps: int, shift: float) -> np.ndarray:
    """FlowMatchEulerDiscreteScheduler's sigmas for Anima: linspace(1, 1/steps) shifted, then 0."""
    s = np.linspace(1.0, 1.0 / steps, steps)
    s = shift * s / (1 + (shift - 1) * s)
    return np.append(s, 0.0)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--dit", default="anima-turbo-v1.1.safetensors")
    ap.add_argument("--prompt", default="masterpiece, best quality, score_7, safe, 1girl, solo, smile, long hair, blue eyes, cherry blossoms, outdoors")
    ap.add_argument("--size", type=int, default=1024, help="the image's width and height")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--noise", type=Path, help="the initial latents (16, H/8, W/8) as float32 .npy, in place of the seed's")
    ap.add_argument("--stop", type=int, default=0, help="run only this many steps (0: all), then decode")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="classifier-free guidance's scale (1: none, Anima-Turbo; Anima-Aesthetic 4 to 5)")
    ap.add_argument("--negative", default="", help="the negative prompt of the guidance (default empty)")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    W, H = a.width or a.size, a.height or a.size
    tok, t5, qwen, tc, tr, vae = load(a.root, a.dit, dtype)
    rec = {}

    def encode(prompt: str):
        """AnimaTextEncoderStep and the conditioner: an empty prompt is one masked Qwen3 token."""
        q = tok([prompt], return_tensors="pt", padding="longest", max_length=512, truncation=True)
        ids, mask = q.input_ids, q.attention_mask
        if ids.shape[-1] == 0:
            ids, mask = ids.new_zeros((1, 1)), mask.new_zeros((1, 1))
        hidden = qwen(input_ids=ids, attention_mask=mask).last_hidden_state * mask.unsqueeze(-1).to(dtype)
        t = t5([prompt], return_tensors="pt", padding="longest", max_length=512, truncation=True)
        c = tc(source_hidden_states=hidden, target_input_ids=t.input_ids,
               target_attention_mask=t.attention_mask, source_attention_mask=mask)
        return ids, hidden, t.input_ids, c

    ids, hidden, t5_ids, cond = encode(a.prompt)
    rec["qwen_ids"] = ids[0].numpy().astype(np.int32)
    rec["qwen_hidden"] = hidden[0].float().numpy()
    rec["t5_ids"] = t5_ids[0].numpy().astype(np.int32)
    rec["cond"] = cond[0].float().numpy()
    uncond = None
    if a.guidance != 1:
        uncond = encode(a.negative)[3]
        rec["uncond"] = uncond[0].float().numpy()

    if a.noise is not None:
        noise = torch.from_numpy(np.load(a.noise)).reshape(1, 16, 1, H // 8, W // 8)
    else:
        g = torch.Generator().manual_seed(a.seed)
        noise = torch.randn((1, 16, 1, H // 8, W // 8), generator=g, dtype=torch.float32)
    rec["noise"] = noise[0, :, 0].numpy()
    sig = sigmas(a.steps, a.shift)
    rec["sigmas"] = sig.astype(np.float32)
    x = noise.clone()
    pad = torch.zeros(1, 1, H, W, dtype=dtype)
    vs = []
    steps = a.stop or a.steps
    for i in range(steps):
        def dit(c):
            return tr(hidden_states=x.to(dtype), timestep=torch.tensor([sig[i]], dtype=dtype),
                      encoder_hidden_states=c.to(dtype), padding_mask=pad, return_dict=False)[0].float()

        v = dit(cond)
        if uncond is not None:  # diffusers' ClassifierFreeGuidance
            u = dit(uncond)
            v = u + a.guidance * (v - u)
        vs.append(v[0, :, 0].numpy())
        x = x + (sig[i + 1] - sig[i]) * v
        print(f"step {i}: sigma {sig[i]:.4f} -> {sig[i + 1]:.4f}, |v| {v.abs().mean():.4f}", flush=True)
    rec["velocity"] = np.stack(vs)
    rec["latents"] = x[0, :, 0].numpy()

    mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
    z = (x * std + mean).to(dtype)
    rec["vae_in"] = z[0, :, 0].float().numpy()
    img = vae.decode(z, return_dict=False)[0][:, :, 0].float()
    rec["decoded"] = img[0].numpy()
    pixels = ((img[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
    rec["image"] = pixels
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out.with_suffix(".npz"), **rec)
    from PIL import Image

    Image.fromarray(pixels).save(a.out.with_suffix(".png"))
    print(f"{a.out.with_suffix('.png')}: {W}x{H}, {steps} steps, qwen {len(rec['qwen_ids'])} tokens, t5 {len(rec['t5_ids'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
