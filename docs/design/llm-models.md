# LLM and encoder models

[日本語](llm-models.ja.md)

Per-model notes for the LFM3 family (LFM3-1.7B, LFM3-VL, LFM3-Audio, LFM3-Omni, d1), Gemma 4,
PLaMo 3 Translate, Ouro, PLaMo-Embedding-1B and the BERT family, plus the tokenizers
(`internal/tokenizer`) and chat formats (`internal/chat`). Qwen models are in
[llm-qwen.md](llm-qwen.md), mixtures of experts (LFM2-8B-A1B, Qwen3.6, gpt-oss) in [moe.md](moe.md),
engine mechanisms (KV cache, rings, split at the attention, prompt store, last-row logits) in
[llm-engine.md](llm-engine.md), weight formats in depth in [quantization.md](quantization.md), AMD
forms in [amd-llm.md](amd-llm.md). The models and their defaults are in
[models.yaml](../../models.yaml).

## Overview

"Port" = Hugging Face's `modeling_*.py` moved to Go (`internal/llm/<arch>`); see
[new-model-checklist.md](new-model-checklist.md) and the `port-architecture` skill. On a Mac,
`bonsaiai-chat` runs only Core AI ([apple-coreai.md](apple-coreai.md)); the Go converter still writes
a `.mlpackage` first, then lowers it.

| Model (models.yaml) | Arch | Kind | Apple default | Intel | AMD |
|---|---|---|---|---|---|
| LFM3-1.7B | `lfm3` | chat (tools, thinking) | lut6, grouped attention, context 262144 | int8, converts, not run on hardware | q4 ([amd-llm.md](amd-llm.md)) |
| LFM3-VL-2B | `lfm3` + `lfm3-vision` | chat + images 512×512 | text lut8, encoder fp16 | converts | q4 |
| LFM3-Audio-2B | `lfm3` + `lfm3-audio-encoder` | chat + audio in (30 s) | text lut8, encoder fp16 | converts | q4 |
| LFM3-Omni-2B | both towers | chat + images + audio | text lut8 | converts | q4 |
| LFM3-Audio-2B-TTS | `lfm3` + `lfm3-depthformer` + `lfm3-codec` | speech (`bonsaiai-audio tts`) | text lut8, depthformer and codec fp16 | — | see [audio-models.md](audio-models.md) |
| d1-2B / d1-Omni-2B | `lfm3` in d1 mode (+ towers) | decision (`bonsaiai-chat systemone`) | lut8 | converts | — |
| gemma-4-E2B-it / E4B-it | `gemma4` | chat (tools, thinking), text only | lut6 | converts | q4, images and audio too |
| plamo-3-translate-0.8b / 2b / 8b | `plamo3` | chat (translation) | lut8; 8B lut4 | int8; 8B int4 by rows | q4 |
| Ouro-1.4B-Thinking / 2.6B-Thinking | `ouro` | chat (thinking, no tools) | lut8, not run on a Mac | int8, converts, not run on hardware | q4 with MLP and head q6: KL 0.026 against HF (top-1 98.4 %), 3.6 tokens/s |
| plamo-embedding-1b | `plamo-bi` | embedding (query) | lut8 + head-scale | int8 | — |
| BERT family (all-MiniLM-L6-v2 …) | `bert` | — (not in models.yaml) | converts only, lut8 | converts only, int8 | — |

Defaults come from `llm.Arch.Format` / `IntelFormat`, or per checkpoint from `Config.Format()` /
`IntelFormat()` (`llm.Formatter`, `llm.IntelFormatter`). Override with `--format` and
`--format-for SUBSTRING=FORMAT` (last match wins; `-keep-fp16` beats all).

## LFM3 family

### LFM3-1.7B (`internal/llm/lfm3`)

- Structure: short convolutions, 5 sliding-attention layers (window 256, rings), 6 full-attention
  layers, QK norm, partial RoPE, per-head gates, sandwich norms, SwiGLU.
- Arch params: `post-scale` (default 64; a power of two in `out_proj`, `o_proj`, `w2`, absorbed by
  the sandwich norms; keeps the ANE's error to 1.4–2.7% a layer), `conv` (`taps` default: the short
  convolution as shifted multiply-adds, avoiding the ANE conv's error), `attn` (`grouped` default,
  `blocks.SDPAGrouped`: one product per KV head over stacked query heads; M3, cache 16320: 73.5 → 40.7
  ms a call vs `tile`), `hidden-out` (1: also output the final hidden state h; used by TTS).
- Default lut6: KL 0.23 nats against bf16 (lut4 1.39); a coding task took 6 rounds vs 13–27 with lut4.
  1.21 GiB a program vs 0.81 for lut4.
- Checked against an independent fp64 forward (`reference_test.go`, ~3e-7) and `bonsaiai-dev check`.
- Split into 7 parts at the attention, context 262144 ([llm-engine.md](llm-engine.md)).
- Core AI, M1 16 GB: 35.2–35.5 tok/s; second start 4.5 s. Memory per models.yaml (1.7 GB; 3.1 GB on
  16 GB with the short-context tier).
- Intel: int8. With int4 the next token matches fp16 at 22 of 52 positions (int4-g128 29, int4-g32
  33, ANE lut4 26), int8 at 51.

### Multimodal checkpoints: text part

- LFM3-VL, LFM3-Audio, LFM3-Omni and d1 have LFM3's text structure. `ParseConfig` reads nested
  configs; `Config.Prefix` selects the tensor names:

| `model_type` | Config under | Tensor prefix | Output layer |
|---|---|---|---|
| `lfm3` | top | `model.` | `lm_head.weight` |
| `lfm3_vl` (VL, d1-2B) | `text_config` | `model.language_model.` | `lm_head.weight` |
| `lfm3_omni` (Omni, d1-Omni) | `text_config` | `model.language_model.` | `lm_head.weight` |
| `lfm3_audio` | `lfm` | `lfm.model.` | `lfm.lm_head.weight` |

- Top-level `tie_word_embeddings` beats the one in `text_config`.
- **Text part is lut8** (`lfm3.Config.Multimodal`). With lut4 (M3): d1-2B sent the README's ticket to
  the wrong team (shipping 0.54; fp16 and lut8 billing 0.57–0.62), LFM3-VL described a red circle and
  blue square as "1000pink and blue icon", LFM3-Audio's TTS was no longer understood by its own ASR.
  lut8 matches fp16 on all three. lut6 was on the edge (billing 0.52 vs shipping 0.42).
- Core AI, M1: LFM3-VL 26.2 tok/s, Audio 24.7, Omni 22.4; second start 3.2–5.5 s. Memory per models.yaml
  (2.6–3.1 GB on Apple).
- On the Neural Engine every program (d1, vision, audio encoder, text, depthformer, codec) is all on
  the NPU.
- Tensors the converter does not read: host-side embedding tables, `codebook_embedding`,
  `mask_embedding`, d1-Omni's depthformer, the codec's encoder side,
  `depth_embeddings.N.to_logits.weight` (tied to the embeddings; HF ignores it on load).

### Vision encoder (`internal/llm/lfm3vl`, arch `lfm3-vision`)

- Chosen with `-arch lfm3-vision` (the `model_type` `lfm3_vl` belongs to the text part). Also reads
  `lfm3_omni` configs.
- Structure (LFM-style): patchifier (a linear on the patch pixels; the (d, 3, t, p, p) weights are
  summed over t at conversion, since a still image fills every time slice), embedding RMSNorm,
  pre-norm layers of two kinds, both with SwiGLU, no final norm:
  - conv layers: `in_proj` → B, C, x; a 3×3 depthwise conv (zero padding) on the patch grid over B·x,
    times C. Written as 9 shifted multiply-adds in (row, column, channel) order (runs on `milcpu`,
    avoids the ANE conv error).
  - attention layers: QK RMSNorm, 2D RoPE, GQA, bidirectional.
- 2D RoPE = ggml's `GGML_ROPE_TYPE_VISION`: pairs (k, k+head_dim/2); k < head_dim/4 rotates by row,
  the rest by column, both with head_dim/2-wide frequencies. A rebuilt table lets
  `blocks.ApplyRope` do it.
- Projector: per-patch LayerNorm (eps 1e-6), 2×2 patch concat (upper left, upper right, lower left,
  lower right; a 5-D reshape and transpose), linear, GELU (erf), linear.
- Fixed shapes: one program per image size (`-image WxH`, multiples of 2×patch; default 512x512).
  `-seq` is not used. The package metadata carries the arch and param values
  (`com.github.shibukawa.bonsaiai.arch`, `...param.image`) so the runtime knows the size.
- Preprocessing (`lfm3vl.Processor`): mean/std, `resample` (2 bilinear, 3 bicubic) and pixel limits
  from `preprocessor_config.json` / `processor_config.json`. `SmartResize` = llama.cpp's still-image
  smart_resize. `Resize` = PIL/torchvision antialiased resize in float (horizontal then vertical,
  8-bit rounding per pass), within ±1 of PIL.
- `internal/vision`: loads programs per size and stretches the image to the program whose aspect is
  nearest smart_resize's choice. In the prompt: `<|vision_start|>`, the embedding rows,
  `<|vision_end|>` (as llama.cpp mtmd). No video.
- Chat: an `image_url` part becomes a media marker (random per start) at its place in the text;
  `Engine.Tokens` turns it into image rows with negative token IDs from the data URL's SHA-256, so
  the same image reuses the prompt cache. The last 16 images are kept; an evicted one is re-embedded
  under a new ID. `/v1/models` lists `image` in `input_modalities`. Terminal: `/image FILE question`.
- M3 16 GB, Core AI, lut8: a 512 × 512 picture with its question (275 tokens) reads in 0.43 s, encoder
  included; 32.5 tok/s on the picture's description, 38 on text. First run 2 min 16 s (convert and
  compile), the NE holds 1.64 GiB of weights.

### Audio encoder (`internal/llm/lfm3audio`, arch `lfm3-audio-encoder`)

- Reads both layouts: LFM3-Audio (`encoder.lfm` LFM2 config, `dim_in`, `sliding_window`; tensors
  `encoder.downsample.`, `encoder.lfm.lfm.`, projector `mlp.model.0/1/3`) and LFM3-Omni
  (`audio_encoder_config.text_config`, `num_mel_bins`; `model.audio_encoder.*`,
  `model.audio_projector.*`).
- Structure: three stride-2 convs on (time, mel) cut time to 1/8 (3×3 → SiLU, then twice depthwise 3×3
  → pointwise → SiLU; kernels read as ggml's [out][in][mel][time]); channel-first flatten + linear
  (`conv_channels` 256, `lin_out` (768, 4096, 1)); causal LFM2 layers (short conv, sliding-window
  attention over `sliding_window` keys, QK norm, full-head rotate_half RoPE θ = 1e6); final RMSNorm;
  projector (LayerNorm eps 1e-5, linear, GELU, linear; 768 → 2048). FFN width by LFM2's
  `block_auto_adjust_ff_dim`.
- Fixed length: one program per max mel frames (`-frames`, default 3000 = 30 s). Inputs `x` (1,
  frames, 128) and `mask` (1, frames). Each conv stage zeroes outputs past the clip with a
  halved mask (llama.cpp pads with zeros each time); later layers are causal, so a clip in a longer
  program gives the same rows. The clip is the first `ceil(ceil(ceil(frames/2)/2)/2)` rows.
- Features (`lfm3audio.Features`, = llama.cpp `mtmd_audio_preprocessor_lfm3a`): 16 kHz, right zero pad
  so the last frame holds the last sample, periodic Hann 400 / hop 160, power spectrum, 128 Slaney mel
  filters (0–8 kHz, area-normalized), log10 (floor 1e-12), (v + 4) / 4. Within 1e-4 of librosa.
  WAV: 8/16/24/32-bit PCM, 32-bit float, channels averaged, other rates resampled with a Hann-windowed
  sinc (llama.cpp uses miniaudio, so non-16 kHz input is approximate).
- `internal/audio`: runs the shortest program that holds the clip, else `ErrTooLong`. LFM3-Audio puts
  the embeddings in the prompt as they are; LFM3-Omni wraps them in `<|vision_start|>` …
  `<|vision_end|>` (as llama.cpp mtmd).
- Chat: `input_audio` parts (base64 WAV) as media markers like images (`Content.Media`). Terminal:
  `/audio FILE.wav question`.
- M3: "Perform ASR." transcribed `say` English word for word and Japanese with one character off.

### Speech output (`internal/llm/lfm3gen`, `internal/speech`)

Model-level parts; the pipeline, sampling and command are in [audio-models.md](audio-models.md).

- Text part with `-hidden-out 1`: the cached program outputs logits and h; `decode.Session.Hidden`
  reads it.
- `lfm3-depthformer`: one MaskGIT round. Inputs x (per codebook, a fixed code's embedding or the mask
  embedding; the host looks them up) and h. h projection (`depth_linear`) + x, bidirectional LFM3
  attention layers (QK norm, full-head RoPE θ = 1e6, per-head gates, no sandwich), final RMSNorm,
  per-codebook RMSNorm, product with each codebook's code table (shared with the output; a batched
  matmul).
- `lfm3-codec` (`audio_tokenizer/`, `model_type` liquid_codec): FSQ code values (levels on the host),
  two projections, each frame copied to 6 positions, 8 causal LFM3 layers (sandwich norms, window-30
  attention, short conv), output norm, projection to log-magnitude and phase. Layer kinds and widths
  are read from the weight shapes (as llama.cpp). `-codec-frames` (default 64 = 5.1 s) per call; longer
  audio runs in chunks overlapped by the decoder's look-back, dropping each chunk's head (same as
  streaming because it is causal and RoPE is relative).
- Host ISTFT (`speech.ISTFT`, = llama.cpp `mtmd_audio_streaming_istft`), 24 kHz.
- Config param names are shared across archs, so the codec uses `-codec-frames`, not `-frames`;
  `TestParamNames` checks that same-named params agree on defaults.
- M3: lut4 text ran at about 1.4× real time but was unintelligible; lut8 (2 parts) about 0.5×.

### d1-2B / d1-Omni-2B (System One API)

- d1-2B is an LFM3-VL checkpoint with `architectures: ["D1Model"]` and stock weights; only the reading
  differs: the whole prompt in one pass, full-attention layers bidirectional, sliding layers and short
  convs causal (llama.cpp `LLAMA_NON_CAUSAL_TYPE_DENSE_ONLY`), no KV cache. The answer is the last
  token's logits over the label tokens only.
- `Config.D1` requires `d1_codes` 2 and `d1_attention` `set`, else error. Program inputs `x` and `mask`
  (1, seq); full-attention layers mask padding only (`blocks.MaskBias`). Output is the final hidden
  state; the output layer (~130M params) stays on the host, which reads only the label rows. A cached
  conversion is an error.
- `internal/systemone`: port of llama.cpp's `POST /v1/systemone` (TypeSafe-compatible).
  - Requests: `state`, `questions` of kind `choice` / `score` / `noul`, key order kept
    (`internal/pyjson`: ordered JSON, `json.dumps(indent=2)` form, Jinja truthiness; numbers written
    as read).
  - Prompt: llama.cpp's systemone template (LFM2 d1 prompt + `<think></think>`).
  - Labels (`d1_labels`): choice keys if all are single letters, else A, B …; non-single-token labels
    from a spare table; spaced forms count as the same choice (max taken). noul: yes/Yes/YES vs
    no/No/NO. score: digits.
  - Answers: softmax at `calibration.json`'s `default` temperature. choice = max; score = the median
    level; noul = P(true); confidence = the answer's probability.
  - Control-token spellings in request strings (`<|im_end|>` …) get a zero-width space after the
    first character (`d1_break`).
  - One run per question (prompts differ); the shortest of several `-seq` programs that fits.
  - Media: `files` / `images` (data URLs) and `image_url` parts of a chat-shaped state, up to 8, in that
    order. `<__media__>` in strings becomes a space; one marker per image at the start of the user turn.
    Images are embedded once per request. d1-Omni: `data:audio/` files and `input_audio` parts the same
    way; a spoken question (`instructions` of type `input_audio`) is a marker at the question's place.
    All wrapped in `<|vision_start|>`/`<|vision_end|>`. Missing encoder → 501.
- Command: `bonsaiai-chat systemone` (`--vision`, `--audio-encoder`; `/v1/systemone` and `/v1/models`
  with `output_modalities: ["decisions"]`). CPU mode via `milcpu` for machines without an NPU (slow).
- M3, lut8: NPU vs CPU interpreter probabilities differ by ~1e-3. After the first call, 4 text
  questions 0.13 s, 4 image questions 0.9 s. Japanese tickets, synthetic images (colors and sides),
  d1-Omni spoken questions all answered as expected.

## Gemma 4 E2B / E4B (`internal/llm/gemma4`)

- Target: the `-it` checkpoints (`Gemma4ForConditionalGeneration`); only the text model is ported.
  Audio and image towers (1411 tensors in E2B) are unused on Apple and Intel.
- E2B text: 35 layers, hidden 1536, 8 heads / 1 KV head. Pattern 4 sliding (window 512, head_dim 256)
  + 1 full (head_dim 512). Each layer: sandwich-norm attention, GeGLU (`gelu_pytorch_tanh`), the
  per-layer embedding branch, then the whole residual times `layer_scalar`.
- Attention scores not scaled (`scaling = 1.0`, QK norm instead; `blocks.SDPAScale`). Unweighted RMSNorm
  on V too. RMSNorm is `w * x` (not Gemma 3's `(1 + w) * x`). Embedding × sqrt(hidden) (unrounded, as
  the fp32 reference). Logit softcap 30 (tanh). Full layers use proportional RoPE: only the first 64
  frequencies of 512 rotate.
- **Per-layer embeddings (PLE)**: `embed_tokens_per_layer` (262144 × 8960; 4.7 GB in bf16) stays on the
  host, stored as int8 tables (E2B 2.4 GB) and only used rows are read (`llm.TokenInputs`, input `ple`
  (1, seq, 8960)). The program starting at layer 0 computes
  `(norm(per_layer_model_projection(emb)) + ple * 16) / sqrt(2)`; layer i uses slice i (256). Split
  programs pass `per_layer_inputs` by name. `per_layer_model_projection` is counted in the first part
  (`llm.Prologue`).
- **KV sharing**: in E2B, layers 15–34 have no K/V of their own: sliding layers read layer 13's, full
  layers layer 14's. Only 15 layers keep caches (3 full, 12 sliding rings). Across parts the source part
  outputs `shared_k_sliding` etc.; `chain` binds an input to the earlier part's output of the same name.
  `llm.WritePart` uses `BuildCached` for cached models so these outputs are kept.
- **fp16 overflow**: layer 0's `input_layernorm` weights reach 270, so QK norm squares overflow fp16
  (attention output 45% off), and residuals reach 1900 before `layer_scalar`. Every gemma4 norm
  normalizes `x/64` (eps / 64²; exact). `-f16` check: KL 3.17 → 0.0004. The PLE projection's norm is
  not scaled (its input RMS 0.3; scaling made squares subnormal, 0.04% → 0.5%).
- `-logits host` (default, arch revision 1): the program outputs logits before the softcap and the host
  applies it (`llm.LogitSoftcap`). `flat` / `folded` keep it in the program: on Core ML the 262144-wide
  softcap fell to the CPU unless folded to (16, 16384), and the final reshape back stayed on the CPU.
- `SDPAGrouped` (not `tile`): `tile` copied K and V 8× per step (128 MB each at cache 16320). M3 lut4:
  1024 cache 31–35 → 23.6–25.8 ms a token, 16320 cache 90–150 → 47 ms.
- Split at the full-attention layers (`llm.Paged`, context 131072). Unsplit, on Core ML, the lut6 parts
  never stayed in the ANE cache (~100 s every start); split into 8 parts at the attention, 1.9 s.
- Checks: tiny synthetic model vs a loop-written fp64 forward of `modeling_gemma4.py` 1.6e-6; 15
  mutations all caught. Real E2B (`bonsaiai-dev check`, 31 tokens from `<bos>`): 0 missing / mismatched,
  argmax 31/31, KL 0.0000 (fp16); `-f16` worst layer 0.0098, KL 0.0004. Reference
  `tools/ref/ref_gemma4.py` builds `Gemma4ForCausalLM` in fp32 from the text config and reads PLE rows
  lazily (9 GB instead of 20 GB).
- **Weights: lut6** (`Arch.Format`):

| Format (E2B, CPU check, 31 tokens) | Layer error mean / worst | Argmax | KL (nats) |
|---|---|---|---|
| lut4 | 15% / 31% | 13/31 | 2.54 |
| lut6 | 4.7% / 11% | 25/31 | 0.25 |
| lut8 | 1.2% / 3.0% | 27/31 | 0.048 |
| fp16 | 0.01% | 31/31 | 0.0000 |

  - lut4 is coarse because sandwich norms rebuild most of a layer's output inside the layer, so
    4-bit weight error shows almost directly; keeping one group in fp16 only halves it. lut4 answered
    Mt. Fuji 3778 m and skipped factual questions; lut6 and lut8 answer them (3776 m, 夏目漱石, 信濃川).
  - On the ANE (128 tokens, fp32 reference): lut6 rounding KL mean 1.02 (en) / 1.16 (ja); lut8 0.10 /
    0.10 at 2.16 vs 1.60 GiB.
- **GPTQ codes (optional)**: `bonsaiai-chat --codes DIR --format-for per_layer=lut8` on E2B: ANE KL
  1.02 → 0.56 (en), 1.16 → 0.94 (ja), median 0.25 → 0.08; same speed, +0.01 GiB. 12 factual questions:
  10/12 → 11/12. The `per_layer` projections (41M) are not in the calibration set, so lut8 for them
  matters with GPTQ. Calibration fits a 16 GB Mac with `q4_gptq.py`'s `load_gemma4_lean` (memmapped
  tables, ~9 GB; 87 min on an M1). Not default: the 2.1 GB of codes has no distribution yet. Export
  names carry the codes hash and rules. Details: [quantization.md](quantization.md).
- Speed and memory: Core AI M1 E2B 24.0–24.5 tok/s, second start 2.7 s; Core AI M2 E4B (8 parts) 20.3–20.7
  tok/s, 3.27 GiB on the Neural Engine, a 2149-token prompt 1.8 s, second start 0.9–1.0 s. Memory per
  models.yaml (E2B 2.3 GB, E4B 4.0 GB on Apple; E4B from 24 GB machines).
- Tokenizer: byte-fallback BPE; no BOS added (the chat format writes `<bos>`).

## PLaMo 3 Translate (`internal/llm/plamo3`)

- `Plamo3ForCausalLM`. Shape = Gemma 4 without PLE, KV sharing or softcap. Sandwich norms; RMSNorm
  offsets per norm (1, 1/5, 1/5^1.5) added into fp16 weights. Norm inputs scaled by 1/64 (max |hidden|
  1089 at layer 18 of the 2B).
- `qkv_proj` and `gate_up_proj` split by row ranges (`blocks.LinearRows`). `lm_head` rows are the
  vocabulary padded to a multiple of 16; extra rows are dropped.
- One layer in 8 is full attention with θ = 10⁶ YaRN (factor 256); the rest sliding window 2048 (rings).
  YaRN's concentration (0.1 ln 256 + 1 = 1.5545) is not put into cos/sin; its square scales the full
  layers' scores (`scoreScale`, `PagedShape.Scale`), so cached keys carry none and the host attention
  only applies a scale.
- Split at the full-attention layers (`llm.Paged`), context 262144.
- The attention output is taken as `attn_ot_N` (1, seq, heads, head_dim), token-first; the host
  transposes (`decode.putAttnOut`). Taking (1, heads, seq, head_dim) left one `transpose` on the CPU in
  the last part.
- Architecture shared by all sizes; shapes come from config:

| | 0.8B | 2B | 8B |
|---|---|---|---|
| hidden / layers / KV heads | 1536 / 16 / 1 (16 heads) | 2560 / 24 / 4 | 4096 / 32 / 4 (32 heads) |
| Full-attention layers | 7, 15 | 7, 15, 23 | 7, 15, 23, 31 |
| KV a token | 1 KiB | 7 KiB | 9 KiB |
| Apple format | lut8 | lut8 | **lut4** (`Config.Format`, hidden ≥ 4096) |
| Intel format | int8 | int8 | **int4 by rows** (`Config.IntelFormat`) |
| Programs (bonsaiai-chat) | two | two on Core AI (functions of one `.aimodel`) | one (`prefill: off`) |
| models.yaml `machine` | 8 | 16 | 16, `not_on: [M1]` |

- 2B formats (M3 ANE vs fp16, cache 4096, 8 tokens a call): next token same lut6 89%, lut8 98%; call
  fp16 105.4 ms, lut6 53.7, lut8 57.8. CPU KL: lut4 1.91 (unusable), lut6 0.073, lut8 0.0066, int4 0.72,
  int4-g128 0.32, int8 0.0025.
- 0.8B (M3, Core AI): lut8, a short sentence 72 tok/s; a 1214-token Japanese document read in 0.52 s and
  translated at 118 tok/s (prompt-lookup drafts take the copied code); 20/20 translations same as fp16
  (lut6 16/20).
- 2B (M3, Core AI): short sentence 23 tok/s; the same 1214-token document read in 1.9 s and translated
  at 38.6 tok/s; NE weights 2.47 GiB.
- 8B: lut8 7.58 GiB, lut6 5.66, lut4 3.77 on the Neural Engine. lut4 keeps translations: 13/20 sentences
  word for word as lut8, the rest correct rewordings; answer-token KL vs lut8 0.305, argmax 88%
  (`bonsaiai-dev generate -write-ref` / `-score`). Speed (lut4): M3, Core AI: short sentence 14.5 tok/s,
  the 1214-token document read in 12.8 s (one program of 8 tokens a call: prefill off) and translated at
  23 tok/s; start 0.3–0.4 s from the second run (first 2 min 22 s with a 4 GB download of the export);
  NE holds 3.80 GiB. Core ML era: M2 9.4–10.1 tok/s but 35–41 s every start (compile cache keeps only 1.2–2.1 GiB);
  M1 0.8 tok/s (over the 2.8 GiB an M1's NE runs at full speed) → `not_on: [M1]`, the 2B is picked
  there. M3 lut8 5.8–6.5 tok/s (Core ML). Memory per models.yaml (Apple 8.2 GB peak).
- Intel 8B: int8 (7.6 GiB) is over the ~4.6 GB a Core Ultra 5 135U NPU held; int4 by rows (3.8 GiB, one
  program). int4 rounding run on the M1 ANE: 12/20 same as lut8, rest rewordings.
- Checks (2B): tiny model vs fp64 6.5e-5; `bonsaiai-dev check` 0 missing / unused, argmax 31/31 (fp32),
  `-f16` KL 0.0001. 0.8B and 8B not checked against HF fp32 (structure as the 2B).
- Translation in bonsaiai-chat: earlier pairs are kept as history; `--to`, `--from`, `--form`, `/to LANG`;
  the server takes `glossary` ([cli.md](cli.md)). Draft tokens never hit (output language differs from
  the prompt).
- Tokenizer files are chosen by arch (`tokenizer.jsonl`, `tokenizer_config.json`).

## Ouro (`internal/llm/ouro`, `tools/amd/iron_ouro.py`)

- ByteDance's looped language models (`OuroForCausalLM`, remote code `modeling_ouro.py`; Apache 2.0).
  Llama's layer with sandwich norms (`input_layernorm`, attention, `input_layernorm_2`, residual;
  `post_attention_layernorm`, SwiGLU, `post_attention_layernorm_2`, residual), plain multi-head
  attention (16 heads, no GQA), RoPE θ 10⁶, no biases, no norms over the heads, an untied `lm_head`,
  vocabulary 49152 (SmolLM's). The whole stack runs `total_ut_steps` (4) times on each token with the
  same weights; after each pass `model.norm` is applied and the next pass starts from the normed
  state; the head reads the last pass's. Each pass keeps keys and values of its own for every layer.
- BonsaiAI treats the passes as layers: `Layers()` is layers × passes (1.4B: 24 × 4 = 96, 2.6B: 48 × 4 =
  192), layer v the weights of layer v mod layers (`llm.Looped`), the caches `kv_k_v` per pass. The
  caches, the parts and `bonsaiai-dev check` then work as for any decoder. A program holds a layer's
  weights once however many of its passes it builds (the builder keeps the constants by name:
  `TestPassesShareWeights`); `llm.EstimateBytes` counts them for every pass (`llm.Looped`), which is
  over the stored size of a part that builds more than one pass. The KV cache is 4× a 24-layer
  model's: 768 KiB a token for the 1.4B in fp16, 1.5 MiB for the 2.6B.
- The early exit gate (`model.early_exit_gate`) is not read: with `early_exit_threshold` 1.0 (the
  checkpoints') transformers takes the last pass. A threshold under 1 or `early_exit_step` is refused
  (an exit per token would leave the later passes' caches without that token's keys).
  `total_ut_steps` can be lowered in config.json, as the model card says, for fewer passes.
- Norms: the residual reaches 212 (1.4B), its rows' root mean square 4.7, so the norms over it
  (`input_layernorm`, `post_attention_layernorm`, `model.norm`) take it divided by 4. The norms after
  the attention and the MLP read at most 54 and 25 but rows down to a root mean square of 0.0002, so
  they take their input as it is (1/64, as Qwen3's, would square such rows under fp16's smallest
  values). The first layer's norm reads the embeddings as they are.
- Checks (1.4B-Thinking): the tiny model (3 layers × 3 passes) against fp64 8e-7, layer by layer 6e-7,
  with a cache in parts 3.5e-4; breaking the norm between passes, sharing a layer's cache between
  passes or dropping the norm after the MLP each fail them. `bonsaiai-dev check` (30 tokens): 0 missing,
  0 mismatched, 2 unused (the gate); fp32 worst layer 0.0004, argmax 30/30, KL 0.0000; `-f16` worst
  layer 0.0012, argmax 30/30, KL 0.0000. The reference is `tools/ref/ref_logits.py` with transformers
  4.54 (the remote code fails on 5.x: `pad_token_id`, `create_causal_mask`); it records each layer
  once a pass and the norm between passes (`hook_passes`). 2.6B-Thinking (192 layers, residual up to 161): `-f16` worst layer 0.0020, argmax 30/30, KL 0.0000.
- Formats (CPU, `-f16`, the whole model against HF fp32, 30 tokens): lut8 KL 0.023 (argmax 25/30),
  lut6 0.18 (23/30), lut4 5.1 (3/30, unusable), int8 0.006 (29/30). The default is lut8 (Intel's int8).
  A weight read four times a token compounds its rounding. The 2.6B, whose 48 layers make 192 runs, loses more: lut8 KL 0.13
  (argmax 27/30), int8 0.067 (26/30); it is not in models.yaml.
- AMD (`iron_export_lm.py ouro`): IRON's `SandwichLayer` with Ouro's norms renamed; `Ouro.layer`
  runs `model.layers` once a pass (`self.layers` is them four times), gives a pass after the first its
  own `KVCache` through the call (`passes.P.layers.L.cache.*`, which the runtime reads as caches by
  `.cache.`), and puts `model.norm` between passes. q6 MLP and head (host rounding KL q4 0.130, q6 0.013). On the NPU
  (`ironlm -forced` against `forced_logits.py`, 64 rows): q6 KL 0.026 (top-1 98.4 %), bf16 0.0022 (98.4 %); details in [amd-llm.md](amd-llm.md).
- Tokenizer: SmolLM's (individual `Digits`, then `ByteLevel` with GPT-2's pattern), `gpt2Digits` in
  `internal/tokenizer`. Its golden ids come from the tokenizers library (`tokenizer_golden.py --tokenizers`):
  transformers 5 puts GPT2Tokenizer's own pre-tokenizer in place of the file's and loses the Digits
  step (" ①" one piece).
- Chat format: `chat.Ouro` (see below). The Thinking models think by default; the base models
  (Ouro-1.4B, 2.6B) have no chat template.
- Not in models.yaml: on the NPU a looped model is slow for its size and its cache grows fast (an evaluation
  record, with LFM3-1.7B and others beside it: [model-estimation.md](../model-estimation.md); to take it up again:
  [not-adopted/ouro.md](../not-adopted/ouro.md)).

## PLaMo-Embedding-1B (`internal/llm/plamo`, arch `plamo-bi`)

- `PlamoBiModel`: PLaMo's decoder with bidirectional attention, 16 pre-norm layers + final RMSNorm.
  Fused `qkv_proj` (2304 = 2048 + 128 + 128), 16 heads / 1 KV head, head_dim 128, per-head QK-norm
  (`q_weight` (16, 128), `k_weight` (1, 128), eps 1e-6), rotate_half RoPE base 10000. `gate_up_proj`
  (16384, 2048), first half gate. Output = hidden state after the final norm; pooling on the host.
- Chosen by `architectures: PlamoBiModel` (`model_type` `plamo` is shared with the causal LM). MoE
  (`n_expert`, `sparse_step`) and `eval_*_n_bit` are rejected; `linear_type: fp8` and `group_size`
  (training only) are accepted.
- `gate_up_proj` split into two 8192-row linears (`blocks.WeightRows` / `LinearRows`, constants named
  `NAME_begin_end`): one linear would hit the ANE's 16384 output limit exactly. Cuts at multiples of 16
  keep lut tables identical to whole-matrix quantization.
- Inputs `x` and `mask`; positions 0..seq-1 regardless of padding. Right padding equals HF's left padding
  (bidirectional + mask + distance-only RoPE).
- 1.05 billion params (the 103M embedding table on the host). lut8: one part, 0.89 GiB.
- **8 bits**: cosine to fp16 (milcpu) lut4 `uniform` 0.20, `kmeans-minmax` 0.76, lut8 0.998.
- **head-scale** (param, default 64): the last layer's `down_proj` output reaches 5e4 (fp32), and the ANE
  returned -Inf. The last layer's residual after attention and `down_proj`'s input are divided by 64 and
  the final norm's eps by 64² (exact, since only the final norm reads them). Programs without the final
  norm (`NoHead`, middle parts) are not scaled. ANE vs CPU cosine 0.9745 → 0.99986; vs HF 0.9969–0.9987
  on 8 texts (`TestPlamoEmbeddingGolden`, `tools/ref/embed_golden.py`). All ops on the ANE.
- ANE note: layer 3's MLP puts 1.2e3 in one BOS channel that stays in the residual; RMSNorm's `x*x`
  exceeds fp16 there, but the ANE did not break (it seems to sum squares in a wider range). milcpu's
  `RoundF16` gives inf there, so it is useless past layer 4 for this model.
- Embedding API (`chat.Embedder`, `/v1/embeddings`, terminal `bonsaiai-chat search`):
  - Encode with BOS (`<|startoftext|>`). A query is `QueryPrefix + text` ("次の文章に対して、関連する文章を検索してください: "),
    and BOS plus the prefix's tokens are left out of the mean. Mean pooling over the text's tokens,
    normalized to length 1.
  - Programs per `--embed-seq` (default 128, 1024); the shortest that fits, right padding + mask.
  - Too long → `context_length_exceeded`, or cut with `truncate: true`. `input_type` (`query` /
    `search_query`, `document` / `search_document` / `passage`; default document).
    `encoding_format: base64` = float32 little endian.
  - Program names include the format, so an old lut4 export is not reused.

## BERT family (`internal/llm/bert`)

- Encoder ported; converts with lut8 (Apple) or int8 (Intel). all-MiniLM-L6-v2: cosine to fp16 lut4
  0.23, lut8 0.9988.
- Not usable from bonsaiai-chat: no WordPiece tokenizer (`tokenizer.Load` rejects it). Not in models.yaml
  ([not-adopted/bert.md](../not-adopted/bert.md)).

## Tokenizers (`internal/tokenizer`)

Depends only on `golang.org/x/text`. Each is checked against HF tokenizers.

| Kind | Models | How it works | Checked |
|---|---|---|---|
| byte-level BPE (`Load`) | LFM3 family, Qwen | Added tokens cut first; GPT-4-style pre-tokenizer by a hand-written scanner (`\s+(?!\S)` lookahead, which Go's regexp lacks); merges by rank. Qwen3.5: NFC, `[\p{L}\p{M}]+`, digits one by one ([llm-qwen.md](llm-qwen.md)). Ouro: individual Digits, then GPT-2's pattern on the runs between them (`gpt2Digits`) | matches HF (Ouro: 40 texts + 500 random, 18,441 tokens) |
| byte-fallback BPE (`fallback.go`, via `Load`) | Gemma 4 | Spaces → ▁, text between added tokens merged by rank as one piece, unknown chars → UTF-8 bytes (`<0xE3>`). No BOS added | 35 texts + 300 random (7,026 tokens) |
| SentencePiece BPE (`LoadSentencePiece`) | PLaMo-Embedding | Reads `tokenizer.model` protobuf by hand. Identity normalization, ▁ prefix. User-defined pieces (runs of 2/4/8/16 ▁, 1/2/4 tabs, newline) longest-match, never merged. Merges by piece score (ties: left), not a merge list. Unknown → `<0xNN>`. Special tokens cut first and a ▁ added after them too (as HF's slow tokenizer). No BOS added | 32 texts (599 tokens) + 3,032 random (88,072 tokens), 0 mismatches |
| PLaMo (`LoadPlamo`; `Load` picks it for `.jsonl`) | PLaMo 3 | Least-cost split from `tokenizer.jsonl`'s scored pieces, searched from the end; cost = −round(score × 10⁴) (Python round); a map of all piece prefixes instead of Aho-Corasick; ties → longer piece; uncovered chars → UTF-8 bytes. Cuts at special tokens, runs of ≥4 same char, ≥2 spaces | 40 cases + 3,040 random |
| Unigram (`LoadUnigram`) | T5 (image models) | Precompiled charsmap + Viterbi; unknown costs lowest score − 10; no byte fallback | 11 texts |
| WordPiece | BERT | **Not implemented**; `Load` errors | — |

- `LoadSentencePiece` rejects: Unigram, normalization rule tables (nmt_nfkc …),
  `remove_extra_whitespaces`, trailing-space form, unused pieces, BPE without byte fallback, missing
  byte pieces.
- PLaMo-Embedding `Decode(ids, false)` inverts `Encode` (drops the leading ▁ per segment); HF's decode
  adds spaces before special tokens, so HF does not round-trip. `skipSpecial` joins segments first, as
  HF's `skip_special_tokens=True`.
- Golden data: `tools/ref/tokenizer_golden.py` → `internal/tokenizer/testdata/*.json`; real-vocab tests
  need `BONSAIAI_TEST_PLAMO_DIR`, `BONSAIAI_TEST_GEMMA4_DIR`. Score ties and merges across user-defined pieces never
  occur in the real vocab, so the tiny-model tests (`TestSentencePiece`) cover them.
- Reference env for PLaMo's tokenizer: Python 3.12, `sentencepiece==0.2.0` (0.2.2 lacks
  `SentencePieceProcessor.Init`), `protobuf`, `transformers<5`.
- AMD reads the same Go tokenizers from the export directory.

## Chat formats (`internal/chat`)

Each ports `chat_template.jinja` to Go (`chat.Template`, chosen by `TemplateFor(arch)`) and matches
what jinja2 / transformers render, character for character (`tools/ref/template_golden.py`).

| Format | Arch | Checked | Tools | Thinking |
|---|---|---|---|---|
| LFM3 (`chat.LFM3`) | `lfm3` | 6 conversations | `<tools>` JSON at the end of the system message (key order kept); calls as `<tool_call><function=name><parameter=key>value</parameter>` | `<think>…</think>`; earlier replies' thinking dropped |
| Gemma 4 (`chat.Gemma4`) | `gemma4` | 9 conversations (`gemma4_template.json`) | its own notation (strings `<\|"\|>`, keys dictsort) | `<\|think\|>`, `<\|channel>thought … <channel\|>` |
| PLaMo 3 / PFML3 (`chat.Plamo3`) | `plamo3` | 8 cases (`plamo3_template_golden.py`) | none; `target_lang`, `source_lang`, `form`, `context`, `glossary` (`Options.Translation`; API `chat_template_kwargs` as vLLM) | none |
| Ouro (`chat.Ouro`) | `ouro` | 4 conversations (`ouro_template.json`, transformers' `apply_chat_template`) | none (refused) | `<think>` written after the prompt unless `NoThink` (the template's `enable_thinking`); no-think writes Qwen3's empty `<think>\n\n</think>\n\n`, for which the template has no switch |
| System One (`internal/systemone`) | d1 | 7 cases (`systemone_golden.py`) | d1's choice / score / noul | `<think></think>` |
| LFM2, Qwen3, Qwen2.5, Qwen3.5, Qwen3.6, harmony | — | see [llm-qwen.md](llm-qwen.md), [moe.md](moe.md) | | |

- Gemma 4: turns `<|turn>ROLE`; `Render` also reports whether the prompt ends inside thinking (after a
  tool response with thinking on, it ends in `<|channel>thought\n`). Stops: `<turn|>`, `<eos>`,
  `<|tool_response>`.
- PLaMo 3: a system message becomes `context`. Without a target language, text with Japanese goes to
  English, other text to Japanese. Each past translation's label is chosen from its source text, so the
  prompt's start can be reused.
- The think budget feeds the format's closing mark (`</think>`, `<channel|>`)
  ([llm-decoding.md](llm-decoding.md)).
- Model metadata (kind, features `tools` / `thinking` / `vision` / `audio` / `query` / `translation`) come
  from models.yaml; `/v1/models` returns them as `capabilities`. Requests are routed by `model`; a model
  of the wrong kind is an error ([cli.md](cli.md)).

## Not adopted

- lut4 for Gemma 4 (KL 2.54, factual answers wrong), for the LFM3 multimodal and d1 text parts (wrong
  team, broken image descriptions, unintelligible speech), for PLaMo 3 0.8B/2B (KL 1.91) and for
  embeddings (cosine 0.20–0.76).
- Gemma 4 QAT Q4_0 weights as they are: the ANE's lut4 is one table per 16 rows × whole row, so 32-column
  scales do not fit (layer error only 0.17 → 0.16–0.20). Q4_0 as a 32-column grouped conv runs on the
  ANE but 12.3 ms vs lut4 1.81 (7×); as batched matmul it falls to the CPU (42 ms).
- Gemma 4 lut6 + MLP in lut8 (1.99 GiB, over two parts on Core ML) and lut6 + attention in lut8 (most
  of the error is the MLP).
- Gemma 4 norms scaled by dividing weights by 64 (`blocks.RMSNorm` inScale): small weights (0.16)
  become fp16 subnormals (2.5e-4 off on a tiny model).
- Gemma 4 logits left folded as the output: all on the ANE but 137 ms a token (4×). The host softcap is
  the default instead.
- PLaMo-Embedding head-scale by scaling weights: same on the ANE, but small weights and int8 scales
  became subnormal and Intel's int8 IR missed 1e-3 (0.00101).
- PLaMo 3 8B int8 on Intel (7.6 GiB, over the NPU's ~4.6 GB); PLaMo 3 2B int4 on Intel (KL 0.72).
- Core ML int4 for the PLaMo 3 8B head (the 107520-wide linear falls to the CPU): Macs never use int4.
- GPTQ codes as default for Gemma 4: no distribution for 2.1 GB of codes yet.
- d1-omni-600M (LFM2-based, SigLIP2 + FastConformer): out of scope.
- LFM3-Audio raw live ASR (interleaved "Perform live ASR."), chat replies as speech (`<|mixed_start|>`,
  `<|audio_start|>` switching) and streaming speech out: not handled. Video input: not handled.

## Open

- d1-Omni-2B: the README example's team comes out billing 0.45 vs shipping 0.52 (README: billing), the
  same in fp16 on the CPU. HF reads up to `d1_set_size` (16) questions together; BonsaiAI reads one at a
  time as llama.cpp does. Not compared with HF (no torch).
- d1-2B / Omni not compared against HF or llama.cpp's `/v1/systemone` probabilities.
- LFM3-Audio: with thinking at temperature 0 the thinking can loop and never answer.
- PLaMo-Embedding: the last layer's maximum on 1024-token texts (5e4 may grow); batching several texts
  per call (now one at a time).
- Gemma 4: GPTQ for E4B (too large to calibrate on 16 GB), `per_layer` inside GPTQ, more questions and
  agent tasks, M2/M3.
- PLaMo 3 8B: HF fp32 KL, GPTQ codes and `--lut-fit wkmeans` (fp32 calibration ~34 GB), long-context
  passkeys, M4, Intel and AMD on hardware, Core AI numbers.
- PLaMo 3 0.8B: `bonsaiai-dev check` against HF fp32, passkeys, AMD and Intel.
- BERT: a WordPiece tokenizer.
- Ouro: the Neural Engine (placement, speed, the 96 / 192 caches of a whole program; `UnpagedContext`
  1024 is a guess), Intel's NPU, GPTQ codes for lut6/lut4 (and for the 2.6B's lut8), the 2.6B on AMD (q8), and whether fewer passes
  (`total_ut_steps` 3) are worth their speed.
