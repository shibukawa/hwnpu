"""Reference activations of a Hugging Face Whisper model, for checking its conversion.

    python ref_whisper.py MODEL_DIR AUDIO.wav OUT_PREFIX [--language ja] [--tokens 32]

MODEL_DIR holds config.json, preprocessor_config.json, the tokenizer and model.safetensors
(WhisperForConditionalGeneration). AUDIO is a 16 kHz mono WAV of up to 30 seconds. Runs on the
CPU in float32 and writes, little endian and row major, in hwnpu-check's layout (tools/ref/
ref_logits.py):

    OUT_PREFIX.enc.x.bin       float32 (1, 3000, mel bins)  the log mel frames, time major: the
                                                            encoder program's input
    OUT_PREFIX.enc.ids.bin     int32 (1500)                 0 .. 1499 (positions, no tokens)
    OUT_PREFIX.enc.hidden.bin  float32 (layers + 1, 1500, d_model): the hidden state entering
                               layer 0 (after the convolutions and the position embeddings), then
                               each layer's output (before the final LayerNorm)
    OUT_PREFIX.enc.logits.bin  float32 (1500, d_model)      the last hidden state
    OUT_PREFIX.enc.json

    OUT_PREFIX.samples.bin     float32                      the clip's samples, for the features
    OUT_PREFIX.dec.*           the decoder's run over the greedy transcript (when --tokens > 0):
                               ids (the prompt and the tokens it generated), hidden, logits, and
                               the cross attention's keys and values of each layer
                               (OUT_PREFIX.dec.xk_N.bin, xv_N, (1, heads, 1500, head_dim))
    OUT_PREFIX.text.txt        the transcript

Needs torch, transformers and numpy.
"""
import argparse
import json
import wave

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def read_wav(path):
    with wave.open(path, "rb") as f:
        if f.getframerate() != 16000 or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise SystemExit(f"{path}: want 16 kHz mono 16 bit PCM")
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768


def save(path, a, dtype="<f4"):
    np.ascontiguousarray(a, dtype=dtype).tofile(path)


def hooks(layers):
    """Captures the input of the first layer and each layer's output."""
    hidden = []

    def first(_, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        hidden.append(x.detach()[0].clone())

    def each(_, __, out):
        hidden.append((out[0] if isinstance(out, tuple) else out).detach()[0].clone())

    hs = [layers[0].register_forward_pre_hook(first, with_kwargs=True)]
    hs += [l.register_forward_hook(each) for l in layers]
    return hidden, hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--language", default=None, help="the language token, e.g. ja (default: detected)")
    ap.add_argument("--tokens", type=int, default=32, help="most tokens to generate (0: the encoder alone)")
    a = ap.parse_args()

    torch.manual_seed(0)
    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation="eager").eval()
    cfg = model.config
    samples = read_wav(a.audio)
    save(a.out + ".samples.bin", samples)
    feats = proc.feature_extractor(samples, sampling_rate=16000, return_tensors="pt").input_features  # (1, mels, 3000)

    enc = model.model.encoder
    hidden, hs = hooks(enc.layers)
    with torch.no_grad():
        last = enc(feats).last_hidden_state[0]
    for h in hs:
        h.remove()
    n = cfg.max_source_positions
    save(a.out + ".enc.x.bin", feats[0].T.numpy())
    save(a.out + ".enc.ids.bin", np.arange(n), "<i4")
    save(a.out + ".enc.hidden.bin", torch.stack(hidden).numpy())
    save(a.out + ".enc.logits.bin", last.numpy())
    json.dump({"model_type": "whisper", "seq": n, "hidden_size": cfg.d_model, "num_layers": cfg.encoder_layers,
               "vocab_size": cfg.d_model, "text": a.audio, "dtype": "float32", "output": "hidden"},
              open(a.out + ".enc.json", "w"), indent=1)
    print(f"encoder: hidden {tuple(torch.stack(hidden).shape)}, max |hidden| per layer",
          [f"{h.abs().max():.1f}" for h in hidden])
    if a.tokens <= 0:
        return

    gen = {"max_new_tokens": a.tokens, "do_sample": False, "num_beams": 1, "task": "transcribe"}
    if a.language:
        gen["language"] = a.language
    with torch.no_grad():
        ids = model.generate(feats, **gen)[0]
    gc = model.generation_config
    sot = cfg.decoder_start_token_id
    if ids[0] != sot:
        # transformers 5 returns the generated tokens alone: the decoder reads them after the
        # prompt <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|>
        if a.language:
            lang = gc.lang_to_id[f"<|{a.language}|>"]
        else:
            with torch.no_grad():
                lang = int(model.detect_language(feats)[0])
        prompt = torch.tensor([sot, lang, gc.task_to_id["transcribe"], gc.no_timestamps_token_id])
        ids = torch.cat([prompt, ids])
    text = proc.tokenizer.decode(ids, skip_special_tokens=True)
    open(a.out + ".text.txt", "w").write(text)
    print("transcript:", ids.tolist(), repr(text))

    dec = model.model.decoder
    hidden, hs = hooks(dec.layers)
    with torch.no_grad():
        out = model(encoder_outputs=(last[None],), decoder_input_ids=ids[None], use_cache=False)
    for h in hs:
        h.remove()
    # hidden[0] is the input of layer 0: the embeddings and the position embeddings, summed; the
    # program's x is the token embeddings alone, which hwnpu-check looks up (REF.ids)
    with torch.no_grad():
        emb = dec.embed_tokens(ids[None])[0]
    hidden[0] = emb * (cfg.d_model ** 0.5 if cfg.scale_embedding else 1)
    s = len(ids)
    save(a.out + ".dec.ids.bin", ids.numpy(), "<i4")
    save(a.out + ".dec.hidden.bin", torch.stack(hidden).numpy())
    save(a.out + ".dec.logits.bin", out.logits[0].numpy())
    nh, hd = cfg.decoder_attention_heads, cfg.d_model // cfg.decoder_attention_heads
    for i, l in enumerate(dec.layers):
        at = l.encoder_attn
        with torch.no_grad():
            k = at.k_proj(last).view(n, nh, hd).transpose(0, 1)
            v = at.v_proj(last).view(n, nh, hd).transpose(0, 1)
        save(f"{a.out}.dec.xk_{i}.bin", k.numpy())
        save(f"{a.out}.dec.xv_{i}.bin", v.numpy())
    json.dump({"model_type": "whisper", "seq": s, "hidden_size": cfg.d_model, "num_layers": cfg.decoder_layers,
               "vocab_size": cfg.vocab_size, "text": text, "dtype": "float32", "output": "logits"},
              open(a.out + ".dec.json", "w"), indent=1)


if __name__ == "__main__":
    main()
