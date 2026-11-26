# Quantization

[日本語](quantization.ja.md)

Weight formats on each NPU, how a format and a calibration method are chosen, the default for each model family, and the traps. The format sets the bytes read per token, and with them the speed and the energy. Calibration changes only which codes are picked, so it costs nothing at run time.

## Formats

### Apple Neural Engine (Core ML and Core AI)

| Format | Layout | Notes |
|---|---|---|
| `lut4`, `lut6`, `lut8` | One fp16 table of 2^bits values for each group of 16 rows (`Options.Group`, 16 is what the Neural Engine reads fast). Indices packed (uint6 from the low bits) | `mlpackage.Builder.LUT`, `palette.QuantizeBits`. The main formats |
| `lut4r`, `lut6r`, `lut8r` | lutN of each row divided by its scale, then the output multiplied back by fp16 row scales (`blocks.linearRows`) | For matrices whose rows differ a lot in size. gpt-oss's attention rows differ up to 6600×: on the Neural Engine KL 0.28 (lut8), 0.069 (lut8r), 0.092 (int8), 0.033 (fp16); a step on an M3 57 ms for lut8 and lut8r, 73 for int8, 79 for fp16 |
| `int8`, `int4` | Symmetric, one scale per row. `-quant mse` (default) searches the clipping point, `uniform` takes the largest weight | Accurate on the NE, but see the int8 overflow below |
| `int4-g128`, `int8-g32` … | A scale per block of columns | Wrong on the Neural Engine through Core AI (below); the converter refuses them there. Core ML puts int8-gN on the CPU |
| `fp16` | | The reference. Encoders, VAEs and small vision models stay fp16 |

Mixing: `-format-for SUBSTRING=FORMAT` (`Options.FormatFor`, the last match wins); `-keep-fp16` beats all.

Speed follows the weight bytes. M2, Core AI, a constant 4096 × 4096 linear (`aicheck limits -only formats`):

| | fp16 | int8 (row) | lut8 | lut6 | int4 (row) | lut4 |
|---|---|---|---|---|---|---|
| 1 row | 742 µs | 466 | 449 | 387 | 329 | 351 |
| 64 rows | 808 | 506 | 515 | 477 | 379 | 382 |

Minus the ~160 µs call floor: fp16 about 58 GB/s, int8 and lut8 about 55 GB/s; lut4 tops out near 90 G elements/s, so lut4 is only 1.3–1.4× lut8. On an M3 (Core ML, six E2B-size layers) lut4 / lut6 / lut8 / fp16 took 1.81 / 2.65 / 3.49 / 6.52 ms. `hwnpu-dev probe`'s speeds are not the model's: on an M1, probe reads int8 at 62 GB/s but Gemma 4 E2B's parts at about 48 GB/s, so int8 was slower than lut6. Measure the real parts with `HWNPU_PROFILE=1`.

A format also changes how many programs fit the Neural Engine's compile cache (about 3.5–4 GiB). LFM3 lut6 with four programs did not fit and every start recompiled, so `-tier auto` drops the tiers for 6 and 8 bit formats. See [Neural Engine](apple-neural-engine.md).

### Intel NPU

`int4` (per row), `int4-g128`, `int8`, `fp16`. No lutN (`checkIntelFormats` refuses it). `llm.IntelFormat` maps the Neural Engine format: lut4 → int4, lut6 and lut8 → int8. An architecture can set its own (`Arch.IntelFormat`, `IntelFormatter`). int8 is the default for most models: next token as fp16 at 51 of 52 against int4's 22 (weights 1.6 GiB, twice int4). PLaMo 3 2B: int4 KL 0.72, int4-g128 0.32, int8 0.0025. See [Intel](intel.md).

### AMD XDNA (IRON)

| Format | Bits a weight | Layout | Use |
|---|---|---|---|
| `q4nx` | 5 | FastFlowLM's: 16 uniform levels with scale and min per 32. The fit picks the least error of 9 candidates (weight error 8.2 → 7.3 %) | MLP and head with `--q4` |
| `q6` | 6.5 | q8's layout (block 32 out × 256 in, bf16 scale per 32 inputs), codes −31..31 in two planes (low nibble as q4nx, top 2 bits per 128 in 32 bytes) | `--mlp q6 --head q6` for small models |
| `q8` | 8.5 | int8 and a scale per 32 | Attention and short convolutions |
| `q8x` | 8.5 | q8's codes in mx4's block size (32 out × 160 in, with bias), used by decode and prompt | gpt-oss |
| `mx4` | 4.3 | MXFP4 kept: e2m1 nibbles decoded in the kernel, the exponent byte as the scale | gpt-oss's experts |
| `bf16` | 16 | | Router, references |
| `bfp16` | ~9 | 8 values share an exponent, 8-bit mantissas | The prompt GEMMs (q4nx is dequantized into it; gpt-oss's prompt experts) |

GEMV of 8192 × 2048: q4 0.30–0.35 ms, q6 0.42–0.47, q8 0.43–0.50. q6 takes about q8's time (bound by unpacking the top bits); its gain is memory (76 % of q8) and accuracy (relative RMS q4 0.071, q6 0.022, q8 0.005). Numeric details that matter for accuracy:

- q8 and q6 GEMV split each 32-term partial sum into two bf16s (high and rest) before the scale: one bf16 gives cosine 0.9998, two 0.999992
- The cores start in floor rounding; hwnpu sets round to nearest even (Anima DiT velocity error q8 0.088 → 0.031). Takes effect on a new export

See [AMD LLMs](amd-llm.md) and [AMD](amd.md).

### CPU: MXFP4 and int8 experts

Mixture-of-experts models on a Mac run their experts on the CPU (see [MoE](moe.md)):

- gpt-oss-20b: the checkpoint's own MXFP4 (e2m1, a power-of-two scale per 32), not rounded again
- Qwen3.6-35B-A3B: experts derived from bf16 to gpt-oss's MXFP4 layout (`hwnpu-dev mxfp4`, `qwen35.DeriveMXFP4`, 17.1 GB of experts, 22.0 GB in all). `mxfp4.Encode` tries the OCP scale (exponent of the max − 2) and the 3 around it, keeps the least squared error, rounds each value to the nearest e2m1 (ties to even). Relative error 0.113 on Gaussian values; KL not measured
- LFM2-8B-A1B: experts in int8 as runtime weights on the Neural Engine (Core AI, M2 and later)

## Default per model family

From the code (`Arch.Format`, `Config.Format()`, `defaultFormats`) and [models.yaml](../../models.yaml). hwnpu-chat fits lut tables with `kmeans-minmax`; the plain converter's default is `uniform`.

| Model | Apple (Neural Engine) | Intel | AMD (model-recipes.yaml `amd`) |
|---|---|---|---|
| LFM3-1.7B | lut6 (with grouped attention) | int8 | `--q4` (MLP, head q4nx; attention, conv q8) |
| LFM3-VL / Audio / Omni, d1 (text parts) | lut8 (`Config.Multimodal`) | int8 | `--q4` |
| Gemma 4 E2B / E4B | lut6 | int8 | `--q4` |
| Qwen3 | lut8 | int8 | — |
| Qwen3.5-0.8B, 4B | lut8 | int8 | 0.8B `--q4 --mlp q6 --head q6`; 4B `--q4` |
| Qwen3.5-2B | lut6 (hidden 2048) | int8 | `--q4 --mlp q6 --head q6` |
| Qwen2.5-Coder-7B | lut4 over whole rows with GPTQ codes (`codes: faace07a`) | — | — |
| PLaMo 3 / Translate 0.8B, 2B | lut8 | int8 | `--q4` |
| PLaMo 3 Translate 8B | lut4 (hidden ≥ 4096) | int4 | `--q4` |
| PLaMo-Embedding-1B, MiniLM (`plamo-bi`, `bert`) | lut8 (+ head-scale for PLaMo) | int8 | — |
| Whisper | lut8 | int8 (mapped) | see [audio](audio-models.md) |
| gpt-oss-20b | lut8; `self_attn=lut8r`, `lm_head=lut6r`; experts MXFP4 on the CPU | — | experts mx4, attention q8x, head q4, router bf16 |
| Qwen3.6-35B-A3B | lut8 mixers and shared expert, `lm_head=lut6r`, router gates fp16; experts MXFP4 on the CPU | — | `--q4` (experts and head q4, mixers q8) |
| LFM2-8B-A1B | lut8, experts int8 | — | — |
| Image models | text encoder and DiT lut8 (FLUX.3 klein's DiT lut8 with tables fitted to activations); Anima text and DiT int8 (the DiT as fast as fp16 and nearer it than lut8) | Anima fp16 | Anima's DiT q8 |

Image and audio details are in [image models](image-models.md), [image on Apple](image-apple.md), [image on AMD](image-amd.md), [audio](audio-models.md).

Why the LLM defaults are what they are (KL against the float reference unless said):

- LFM3 lut6: KL 0.232 against lut4's 1.387; a coding task in 6 rounds against 13–27. Grouped attention made a call 38 % faster than the old lut4 + tiled
- Multimodal and d1 at lut8: with lut4 d1 sent a ticket to the wrong team, LFM3-VL described pictures wrongly, LFM3-Audio's TTS stopped being understood
- Gemma 4 lut6: lut4 KL 2.54 (argmax 13/31), lut6 0.25 (25/31), lut8 0.048 (27/31) but 3 parts and 41 ms against 31.5
- Qwen3.5-2B lut6: lut8's next token at 260 of 283, the same 10 of 12 questions, 1.32 GiB against 1.78. The 0.8B's lut6 is 242/283 and gets 17 × 23 wrong; the 4B loses too much
- Qwen2.5-Coder-7B: rounding lut4 KL 0.90, GPTQ codes over whole rows 0.131 (lut8 0.0012, 7.1 GB against 3.5)
- PLaMo 3: 2B lut4 KL 1.91, lut6 89 % next-token match, lut8 98 % (M3). The 8B in lut4 translated 13 of 20 sentences word for word as lut8 and the rest as correct rewordings, at half the memory
- Embeddings: lut4 cosine 0.20–0.76, lut8 0.998. PLaMo-Embedding also needs head-scale (below)

## Measuring

- **KL(reference ‖ quantized)**: mean, median, 95th percentile and top-1 agreement over 400 positions (256 tokens read as a prompt, then one at a time), **with a BOS** (LFM3 without one looks 4× worse)
- Five texts in `~/models/calib`: README, GPL-3 (never used for calibration, to catch overfitting), wiki40b-ja (eval), wikitext-2 (test), Go code. Frozen copies of the changing ones are in `testdata/texts`
- Reference: fp32 on the CPU (HF) or bf16 on the same NPU. The NPU's quantized KL must match the CPU estimate (LFM3 0.036 against 0.035); if not, suspect the NPU's arithmetic first
- Neural Engine: `hwnpu-dev generate -package P.parts.json -score REF,...` runs a program without cache (`hwnpu-dev convert -seq 128`) over reference tokens (`ref_logits.py`, `ref_gemma4.py`). The first tokens weigh much (KL up to 8–13), so compare settings only on the same texts
- Remove op error before blaming quantization: compare the same program on the Neural Engine and on Core ML's CPU. The ANE's `silu` alone was most of LFM3 lut8's loss (KL 0.058 → 0.022 with `blocks.SiLU`)
- `hwnpu-dev formats` (`-ref lut8` when fp16 does not fit a part): next-token agreement and short questions. On short texts perplexity can be lower quantized; use agreement
- Image generation: velocity relative RMS error along the reference trajectory (Chroma1's `TestCallsOnDevice`, removed), and the final-latent cosine of free runs over several seeds. One-step error does not predict free runs (FLUX.3: 2.33 → 2.29 % velocity error, yet latent cosine 0.964 → 0.983)
- Audio and image encoders: by WER / CER (Whisper lut6 WER 4.43 → 4.87 %, so lut8)

## Choosing the format

1. Sweep bits per part on the CPU (attention projections, MLP, output matrix, convolutions, towers; `q4_gptq.py --method rtn` with `--q8`, `--head`, `--fmt-*`; `tools/quant/rotkl` for Apple's lut)
2. Decide by the guide:

| Measure | Decision |
|---|---|
| KL < 0.05 (CPU, against HF) | 4 bits can be the default |
| KL 0.05–0.2 | Decide by the answers (LFM3 lut6 is here) |
| KL ≥ 1 | Not usable |
| `hwnpu-dev formats` ≥ 90 % and as many questions right | Take the smaller |
| 85–90 %, or new arithmetic or fact errors | Keep the next larger |
| < 80 %, or repeating answers | Not usable (Qwen3.5 lut4: 60–76 %) |

What the sweeps found:

- **Attention projections at 8 bits**: about 10 % of the weights but most of the error at 4 bits (AMD E2B KL 0.097 → 0.229 with `--attn q4`; bf16 0.091). +0.12 GB
- MLP and output matrix at 4 bits on AMD; the head at 4 bits barely matters. Short convolutions at 8 bits
- Sandwich-norm layers (Gemma 4, PLaMo 3) are weak at 4 bits; on the Neural Engine lut6 or more
- **A larger size of the same structure can take fewer bits** (PLaMo 3 8B lut4). Decide per checkpoint (`llm.Formatter`); do not carry a small size's verdict to a large one
- **Small models (≤ 2B) can break in the MLP at 4 bits**: Qwen3.5-0.8B README 0.055 but GPL 0.34 (GPTQ 0.30), AMD q6 0.044. Always look at the GPL
- **Keep a checkpoint already in 4 bits in its own format**: gpt-oss's MXFP4 rounded to q4nx added 6.6 % weight error, KL README 0.052 → 0.135, wiki-ja 0.043 → 0.107. MXFP4 fits q6 exactly and mx4 as is
- Distilled 8-step DiTs are weak at 4 bits (Anima q4nx latent cosine 0.777, q8 0.961). A large DiT can hold (Qwen-Image-2.1 20B lut4 with kmeans-minmax 0.935–0.981). Large DiT: MLP at 6 bits, attention at 8 (Qwen-Image-2.1's 7B DiT q6 MLP: −1.21 GB, pictures kept)
- A pass run once a prompt (text encoder, approximator) can take a larger format: it is not held through the steps. It still costs load, disk and compile-cache room, so raise only the weights that need it (Chroma1's T5, removed: lut8 with `wo=fp16`, 5.5 GB against fp16's 9.5)
- Weights the calibration list leaves out are still rounded in the base format. Keep small ones a step up (Gemma 4's `per_layer` projections, 41M, `-format-for per_layer=lut8`, +0.01 GiB)

## Calibration

Keep the format, choose the codes better. Try in order:

| Order | Method | Tools | Result |
|---|---|---|---|
| 1 | Rounding (RTN) | `iron_q4.quantize_fmt`, `palette.QuantizeBits` | Baseline. Least weight error is not least output error |
| 1b | **k-means keeping each block's min and max** (no data) | `-quant kmeans-minmax` (`palette.KMeansMinMax`) | Qwen-Image-2.1 DiT, 4 blocks vs fp16: uniform 4.9 % → 3.2 % (plain k-means 12.6 %: it clips outliers). hwnpu-chat's and the music model's default |
| 2 | **Tables weighted by E[x²]** of each input channel (Lloyd from the uniform levels) | `q4_gptq.py --lut-fit wkmeans`, `palette.QuantizeWeightedMinMax`, `tools/coreai/lutfit` | LFM3 lut6 0.050 → 0.038, lut4 0.66 → 0.33 (with GPTQ). FLUX.3 DiT lut8 latent cosine 0.964 → 0.998 |
| 3 | **GPTQ** (each column's error pushed onto later columns, H = E[xxᵀ]) | `tools/amd/q4_gptq.py`, `qwen2_gptq.py`, `gptoss_gptq.py`, `anima_gptq.py` … | AMD wiki-ja: E2B 0.221 → 0.081, E4B 0.083 → 0.049, LFM3 0.071 → 0.038, PLaMo 3 0.058 → 0.040. Apple LFM3 lut6 0.074 → 0.050 |
| 4 | Rotation before lut (below) | `llm.Rotate`, `tools/quant/rotkl` | Qwen3.5 KL about half at every format |
| 5 | AWQ-style scales folded into the previous op | Not written | When lut4 must go lower |
| — | One format up (AMD q6) | `--mlp q6`, `q4_gptq.py --fmt-q4 q6` | When calibration does not fix it |

GPTQ settings that matter:

- Damp H by **0.1 × the mean of its diagonal**. 0.01 fitted the calibration texts and made the GPL worse than rounding
- Calibrate on general text: wikitext-2 and wiki40b-ja, 64 × 512 tokens (`tools/amd/calib_data.py`). The repository's documents alone overfit. Leave the BOS position out of H
- Quantize earlier layers before collecting the next ones' H (a few layers a pass)
- On Apple lutN the tables are fitted first (uniform or wkmeans) and stay fixed; GPTQ picks each weight's index. On q4nx the group's scale and min are refit to the weights as GPTQ reaches the group
- `--q8 gptq` is needed to calibrate attention and convolution projections too (LFM3: 73 of 143 weights without it, 143 with; English KL 0.255 → 0.247)
- A multimodal model's text part is retrained: calibrate each (LFM3-VL / Audio / Omni), do not reuse the base model's codes
- Adopt only if the GPL (not calibrated on) is not worse than rounding. LFM3-Audio's GPL 0.154 → 0.161 was accepted because the other four texts halved
- For a DiT judge on free runs from other seeds (see Not adopted)

Cost: fp32 on the CPU, 25–60 min for 2B models, 2.4 h for E4B (over 30 GB). On Linux run it under `systemd-run --user`. On a 16 GB Mac: Gemma 4 E2B with its embedding tables read from the file (`q4_gptq.py` `lean`, about 9 GB, 87 min on an M1); a lean model must first match transformers' logits (KL ≤ 0.003, same top-1). `qwen2_gptq.py` builds one layer at a time for 7B (77 min on an M1). Qwen3.6 (`qwen36_gptq.py`) 4.3 h.

Codes:

- Written as `palette.Codes` (`index.json`, per-weight `.idx` and `.tab`; `palette.CodesWriter` from Go). Looked up by a hash of the checkpoint weight's shape and bf16 values (`iron_q4.weight_key` = `palette.WeightKey`), so no name mapping per model. If an export reorders or concatenates weights they do not match; check the log's "N weights took the chosen codes". Fused weights quantized as one (PLaMo 3's `gate_up_proj`) match
- AMD: `iron_export_lm.py --q4-codes DIR`. Apple: `hwnpu-dev convert -codes DIR`, `hwnpu-chat --codes DIR` (lutN only; the program name gets the codes hash, e.g. `gemma4-r1-lut6-gptq-27b3b448-per_layer.lut8-...`)
- models.yaml `codes: HASH` makes a model convert with codes. They are put to and fetched from the source with the export (`chat.putCodes`, `fetchCodes`, `converted/MODEL/any/codes-HASH`), since without them nobody can rebuild the export when its key changes. See [distribution](distribution.md)
- GPTQ is the default only for Qwen2.5-Coder-7B. Elsewhere it is opt-in; the AMD recipes do not calibrate

## Rotation before lut

`llm.Rotate` (`llm.Rotation`, `Rotator`) passes an orthogonal Q = diag(s) H (random signs and a Walsh–Hadamard of the largest power of two, block-diagonal otherwise) through the residual stream; only weights change: embedding E Q, matrices that read the residual W diag(1 + w) Q (norm weight then 0), matrices that write it Qᵀ W, head W diag(1 + w_final) Q.

- Works only where every reader of the residual is a linear after a norm and every writer is added as is. Among the Mac chat models only Qwen3.5 (`qwen35.Config.Rotation`). LFM3 (sandwich norm), Gemma 4 and PLaMo 3 have weighted norms on the writer side. gpt-oss would need its MXFP4 experts rotated and re-quantized
- `rotkl` on the CPU (milcpu, float32, 256 tokens, kmeans-minmax): Qwen3.5-2B lut8 0.0047 → 0.0033, lut6 0.0560 → 0.0267, lut4 0.6026 → 0.3664; 0.8B lut6 0.0749 → 0.0388. fp16 rounding does not break the range
- lut4 stays unusable even rotated, and no lut4 mix reaches today's lut6 (closest: head lut4, 0.062), so rotation buys accuracy for lut6, not speed
- On the Neural Engine (`rotkl -ane -per-part 6`, M2, 2B, 128 tokens against the CPU's fp16): lut6 0.0537 → **0.0288**; lut8 0.0079 → 0.0091 (the NE's own fp16 error hides the gain)
- **Default for Qwen3.5's lut6 exports on Apple** (the 2B): `chat.Prepare` rotates rotatable models at lut6 and below on Core AI and Core ML (`-rot` in the export name). AMD and Intel exports are not rotated; Qwen3.6 is not (its experts are MXFP4 on the CPU and would need re-quantizing). Results and the MTP / table handling: [Qwen](llm-qwen.md)

## Failure modes

- **Per-row int8 overflows on the Neural Engine** on large outputs: it seems to sum the integer products and apply the row scale last, so the middle passes fp16. Qwen3-VL's text encoder gave NaN at attention sinks (layer 6 `down_proj` row 0: 13120); Chroma's T5 `wo` and DiT gave infinities; SANA 1.5's int8 DiT broke. lut8 (tables turned back to fp16 before the sum) is clean. Use lut8 in large models
- **Linear and conv outputs must stay under 2^15**, partial sums included (M2: Inf at 32768 even when the final sum is 0). See [Core AI](apple-coreai.md)
- **Per-block tables and scales are silently wrong on the Neural Engine** (Core AI, macOS 27.2, M1): every row group uses its first column block's table or scale for all columns. Fully placed, same speed, wrong values (int4-g32 cosine 0.966–0.984; a 7B with 32-column GPTQ tables answered garbage). Core AI's CPU is right. `blocks.Graph.rowsOnly` refuses them for Core AI; whole-row tables only. Splitting a matrix (`MatrixSlice`) keeps whole-matrix tables, so codes still apply
- **Adjacent lut tables must differ**: `coreai-build` (offline ANE compiler) treats identical neighbouring 16-row tables at the start of a 256-row block as a shared 32-row table and fails (`EmitGlobalBackedCoalescedLutTensor: ... not content-uniform`); it shows only in os_log. Uniform tables collide often. `mlpackage.DistinctNeighbours` (from `Builder.LUT`, every lut export) moves the largest-magnitude value of a duplicate table one fp16 step toward 0. `tools/coreai/lutscan -tile 16` finds duplicates; `tools/coreai/lutfake` reproduces. Reported to Apple (FB25624151). The on-device compiler is not affected
- **k-means started from quantiles crushes large weights**: the top level lands at the 99.8 % point (FLUX.3 DiT lut8 latent cosine 0.56). Start from the uniform (min–max) levels. For wkmeans, starting from the least-weight-error k-means gave the worst KL
- Final-layer outputs past fp16 (PLaMo-Embedding `down_proj` up to 5e4 → −Inf, cosine 0.9745): head-scale divides the last residual and the `down_proj` input by 64 (exact, a power of two; the final RMSNorm does not care) → 0.99986. Scaling the weights instead made Intel's int8 scales denormal
- Dense AMD: the floor rounding mode (above). bfp16 is fine for GEMMs (errors average out) but not where a softmax follows
- The ANE's `silu` / `softplus` errors look like quantization loss; see [Neural Engine](apple-neural-engine.md) and [Core AI](apple-coreai.md)

## Not adopted

- Q4_0 (Gemma QAT grid) on the Neural Engine: grouped conv 12.3 ms against lut4's 1.81 (7×); batched matmul falls to the CPU (42 ms)
- Per-row int8 for LLMs and large DiTs on Apple: no faster than lut8 and overflows on large outputs (above). Anima keeps int8
- lut4 for LFM3, Gemma 4, PLaMo 3 ≤ 2B, Qwen3.5, multimodal parts, embeddings, Qwen2.5-Coder by rounding: KL 0.9–2.7 or broken answers
- lut4 even after rotation (Qwen3.5-2B 0.37); a lut4 head on top of rotated lut6: +3.4 % speed (33.5 vs 32.4 tok/s) but KL 0.062 against 0.056
- Rotating Qwen3.5's lut8 exports (0.8B, 4B): same answers, 0–3.5 % faster, and re-exporting takes 10–27 min
- Gemma 4 lut6 with the MLP at lut8: 1.99 GiB, over two parts; attention alone leaves most error
- Tables or scales per column block on Core AI's Neural Engine: wrong values (above). (Qwen2.5-Coder lut4 with 32-column GPTQ tables would be KL 0.060 against 0.131)
- GPTQ as the default for LFM3 and Gemma 4 on Apple: KL drops (LFM3 English −23 %, Japanese −35 %; E2B median 0.25 → 0.08 with `per_layer` lut8) at the same speed, but 12 questions differ by at most one (GPTQ even got Mount Fuji wrong on LFM3), and the 1.6–2.1 GB of codes have no distribution plan. Available with `--codes`
- GPTQ on DiTs (Qwen-Image-2.1 MLP q4): teacher-forced velocity error −13–15 % on held-out prompts, but free runs worse than rounding on 2 of 3 seeds. Anima's DiT stays q8 rather than q4 + GPTQ (latent cosine 0.939 against q8's 0.962)
- GPTQ on gpt-oss's AMD attention q8: wiki-ja 0.027 → 0.066 (three positions KL 1.6–2.9); only the head's q4 is calibrated, and only when codes are passed
- q6 attention for gpt-oss on AMD: KL 2–3× (README 0.133)
- Re-rounding MXFP4 experts to q4nx (above)
- bfp16 for gpt-oss's decode attention scores: outputs 3.6 % of their largest off (cosine 0.99987); bf16 kept
- Folding norm weights into q8 table columns (gpt-oss AMD): widens the range, top-1 changed in 32 tokens
- Plain k-means for lut4 DiTs (clips outliers, 12.6 %); k-means from quantiles (above)
- H damping 0.01; calibrating on the repository's documents alone (GPL 0.105 → 0.129, worse than rounding)
- Anima's DiT at fp16 on Apple: latent +0.006–0.025 for 8 % more time and 1.8 GB more; int8 stays
- Whisper at lut6: WER 4.43 → 4.87 %
- Chroma's DiT MLP at lut4 (latent cosine 0.87–0.93 against lut8's 0.9965); lut6 kept the picture but was no faster at 1024²

## Open

- AWQ-style scales (not written)
- `hwnpu-chat --package` with a rotated export picks the unrotated `tables.safetensors`
- GPTQ on the Neural Engine for more models: E4B (too big for a 16 GB Mac even lean), `per_layer` in the calibration, lut4 + GPTQ chat, larger question sets and hwnpu-agent tasks, M2 and M3
- Accuracy of Qwen3.6's derived MXFP4 experts (KL vs bf16, GPTQ for MXFP4); its AMD GPTQ codes are not in the export
- Qwen2.5-Coder 14B (not converted or calibrated); AMD form of Qwen2.5-Coder
- AMD q6 attention with GPTQ for gpt-oss (attention is 36 % of a layer's transfer)
- ANE compile-cache capacity (seen between 3.2 and 4.8 GiB) is not confirmed on the aned side
