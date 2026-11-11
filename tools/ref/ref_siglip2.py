#!/usr/bin/env python3
"""SigLIP 2 (google/siglip2-*, Apache-2.0) on the CPU with Hugging Face's Transformers: the
reference hwnpu-image label and embed (tools/amd/iron_siglip2.py, internal/vision/siglip) are
checked against.

The images go through the model's own processor (resized to the model's size with PIL's
bilinear, scaled to [-1, 1]); the texts are lowercased (SigLIP 2 was trained on lowercased
text) and tokenized as the zero-shot pipeline does (padded to 64 tokens, no attention mask,
the tokenizer appends <eos>). With --template, each label becomes the template's sentence.

Writes OUT.npz: the images' pixel values, the patch embedding's rows, the vision tower's last
hidden state and pooled output (before normalizing), the texts' token ids, the text tower's
last hidden state and pooled output, the normalized embeddings and the sigmoid probabilities
(images x texts), and prints each image's labels by probability.

  python tools/ref/ref_siglip2.py ~/models/siglip2-base-patch16-224 OUT --images cat.jpg \
      --texts "2 cats" "a plane" "a remote" --template "this is a photo of {}."
"""

import argparse
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a SigLIP 2 repository (config.json, model.safetensors, tokenizer)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--images", type=Path, nargs="*", default=[])
    ap.add_argument("--texts", nargs="*", default=[])
    ap.add_argument("--template", default="{}", help="each text's sentence, {} the text")
    ap.add_argument("--fp64", action="store_true", help="run in float64")
    a = ap.parse_args()

    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    dtype = torch.float64 if a.fp64 else torch.float32
    model = AutoModel.from_pretrained(a.root, dtype=dtype).eval()
    proc = AutoProcessor.from_pretrained(a.root)
    out = {}
    img_emb = txt_emb = None
    if a.images:
        imgs = [Image.open(p).convert("RGB") for p in a.images]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dtype)
        vm = model.vision_model
        emb = vm.embeddings(px)
        o = vm(pixel_values=px)
        out.update(pixel_values=px.numpy(), embeddings=emb.numpy(), image_hidden=o.last_hidden_state.numpy(),
                   image_pooled=o.pooler_output.numpy())
        img_emb = o.pooler_output / o.pooler_output.norm(dim=-1, keepdim=True)
        out["image_embeds"] = img_emb.numpy()
    if a.texts:
        sentences = [a.template.format(t).lower() for t in a.texts]
        tok = proc.tokenizer(sentences, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        ids = tok["input_ids"]
        o = model.text_model(input_ids=ids)
        out.update(input_ids=ids.numpy().astype(np.int32), text_hidden=o.last_hidden_state.numpy(),
                   text_pooled=o.pooler_output.numpy())
        txt_emb = o.pooler_output / o.pooler_output.norm(dim=-1, keepdim=True)
        out["text_embeds"] = txt_emb.numpy()
        for s, row in zip(sentences, ids.numpy()):
            print(f"{s!r}: {[int(t) for t in row if t != 0]}")
    out["logit_scale"] = model.logit_scale.exp().numpy()
    out["logit_bias"] = model.logit_bias.numpy()
    if img_emb is not None and txt_emb is not None:
        logits = img_emb @ txt_emb.T * model.logit_scale.exp() + model.logit_bias
        probs = torch.sigmoid(logits)
        out["probs"] = probs.numpy()
        for p, row in zip(a.images, probs.numpy()):
            print(p.name + ":")
            for j in np.argsort(-row):
                print(f"  {row[j]:.4f} {a.texts[j]}")
    np.savez(a.out.with_suffix(".npz"), **out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
