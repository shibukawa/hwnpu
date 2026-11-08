# Checks when bringing in a new model

[日本語](new-model-checklist.ja.md)

Written 2026-11-07. While bringing in LFM3 (VL / Audio / Omni, d1), Gemma 4 E2B / E4B, Qwen3.5-4B, PLaMo 3 Translate, PLaMo-Embedding and Anima, some problems were found only late, and some methods paid off. This page lists them **in the order to check them first**. Checking in this order from the start gives a form with little memory and good speed, without redoing the work.

The numbers come from chapter 16 of [design.md](design.md) (section numbers in parentheses), [improvements.md](improvements.md), [model-fit.md](model-fit.md) and [amd-npu-models.md](amd-npu-models.md). The steps and pass conditions for moving a structure to Go are in `.claude/skills/port-architecture` and are not repeated here.

| Stage | What it decides | Where |
|---|---|---|
| 1 | Whether it fits and is fast, on paper (whether to choose it) | Anywhere |
| 2 | Whether the structure is right (fp64, against HF, tokenizer, chat format) | CPU |
| 3 | Whether it fits on the Mac's Neural Engine, and all of it runs on the NPU | Mac |
| 4 | The AMD / Intel form | Linux / Windows |
| 5 | **The quantization format and the calibration method** | Estimated on the CPU, checked on the NPU |
| 6 | Memory (resident, pinned, disk) | Each NPU |
| 7 | Speed (generation, prompts, startup) | Each NPU |
| 8 | Long contexts | Mac (Intel) |
| 9 | Checks in use, and the record | Each NPU |
| 10 | Values for combining it as a part (residency, input and output kinds) | Each NPU |

Each "pass" is a guide. When a model does not meet it, write the reason in design.md before going on.

## 1. Estimates on paper (before porting)

Estimate from config.json alone, with the formula and criteria in sections 2 and 4 of [model-fit.md](model-fit.md).

- **Time per token (ANE)** ≈ weight bytes ÷ 65 GB/s + layers that compute attention over the long capacity × 0.22 ms (the `SDPAGrouped` form) × capacity / 1K + 3 ms
  - Sharing KV does not reduce how many attentions are computed (only memory). Window, short-convolution, Mamba2 and GDN layers do not depend on the capacity
- **Weights**: lut4 0.5 B, lut6 0.75 B, lut8 1 B a weight. On an M3 (16 GB), keep resident memory under 3 GiB. On AMD, 4 bits (q4nx at 5 bits + attention at q8) pins 1/2–1/5 of bf16
- **Vocabulary**: the output layer is vocabulary × hidden. A 262144 axis falls to the CPU on the ANE, so it has to be folded to (16, 16384) (Gemma 4, 16.25)
- **Structures that cannot be used**: MoE routing (`gather`), learned sparse attention, variable shapes. On AMD only the MoE building blocks were measured (16.75)
- **License and languages** (whether it goes into hwnpu-chat's catalog)

**Pass**: an estimate of 50 ms or less at 1K and 100 ms or less at 16K, and a format in which resident memory fits in 3 GiB on an M3.

## 2. A correct structure (CPU)

On top of the pass conditions of the `port-architecture` skill (1e-4 against an fp64 reference, one layer at a time matching all at once, breaking it to see the test fail, layer-by-layer checks with `hwnpu-check`), these were found later:

| What to check | Tools | Lesson |
|---|---|---|
| Compare **with a BOS** | `tools/ref/ref_logits.py`, `hwnpu-check`, `ironlm -forced -bos` | Without a BOS, LFM3's KL looks 4× worse and layer 0's activations become outliers (amd-npu-models.md section 13). Measure with the tokens the chat format actually puts first |
| Check against the real weights | `hwnpu-check`, `iron_plamo3_check.py`; on AMD `ironlm -forced` + `tools/ref/forced_logits.py` | With random weights the NPU and the CPU reference disagree (Gemma 4, PLaMo 3): their logits are flat, and bf16 rounding and the bfp16 GEMMs spread layer by layer (16.107). Decide pass or fail with the real weights |
| The tokenizer matches HF | `tools/ref/tokenizer_golden.py` | 0 mismatches over fixed texts plus thousands of random ones. Also check that the tests fail when ties (PLaMo), NFC and combining marks (Qwen3.5) or byte fallback (Gemma 4) are handled the other way |
| The chat format matches jinja2 character for character | `tools/ref/template_golden.py` | Tool calls, thinking, dropping earlier thinking |
| Multimodal preprocessing | `resize_golden.py`, `mel_golden.py` | ±1 against PIL, 1e-4 against librosa |
| Embedding models | `embed_golden.py` | Cosine 0.997 or more against HF |
| Image generation | `tools/ref/ref_anima.py` | Latent cosine against diffusers' fp32, PSNR of the VAE |

## 3. Does it fit on the Mac's Neural Engine?

**Check this first**: nothing here can be settled without a Mac. Write a handoff (`docs/handoff-*.md`) and measure on a Mac.

### 3.1 Limits

| Limit | What to do | Section |
|---|---|---|
| 1 GiB a package (0.9 GiB a part by default) | Split by layers (separate packages; multifunction cannot get past it) | 16.12–13 |
| 16384 per dimension | Attention split into chunks (`-paged`), the softcap folded to (16, 16384) | 16.21, 16.25 |
| A compile cache of about 4 GiB (LRU) | If the two programs (Q=8 and Q=64) exceed it, use one (`-prefill auto`). Tiers (`-tier`) only with lut4 | 16.41, 16.65, 16.71 |
| Tables only per 16 rows | How to choose the format is in section 5 | |
| `gather` runs on the CPU | Look up embedding rows on the host (PLE too) | 16.25 |
| Attention scores under 64 MiB a tensor | Past it, 2–3× slower. Chunk the queries (`mil.DitQChunk`) | 16.85 |

### 3.2 Does all of it run on the NPU?

Check that each part is `all on the NPU` with `hwnpuinfo -model PART.mlpackage`. What fell to the CPU before:

- Attention that copies K/V with `tile`, and the form with the queries as a matrix → `SDPAGrouped` (16.25, 16.39). The order alone changes the placement
- A `transpose` / `reshape` in the last part → take the output token-first (`attn_ot_N`, PLaMo 3). The `reshape` in Gemma 4's last part is not solved yet (improvements.md 11.1)
- Parts with one row, and parts as programs of their own → 8 rows, or put into the neighbouring group (Anima, 16.85)
- `reduce_sum` → sum with a matrix multiply (chunked attention: on the ANE 30 times of 30)

### 3.3 Does fp16 overflow?

Check that each layer's relative RMS error is under 0.01 with `hwnpu-check -f16`. What helped:

- Scale the norms' inputs by 1/64 (Gemma 4: KL 3.17 → 0.0004; PLaMo 3 too)
- Shrink the last layer with head-scale (PLaMo-Embedding), `-post-scale` (LFM3, 1.4–2.7 % a layer)
- Keep residuals at 1/32, and divide by each row's maximum before a norm (Anima; a fixed scale made subnormals)
- Fill padding rows with the last token (Gemma 4's eps becomes 0 in fp16 and gives NaN)
- The ANE's own `softplus` and `silu` are inaccurate (softplus returned 0 for some inputs; silu off by 1.4 % in its tail). Use `builder.softplus` and `builder.silu` (written with exp and log). Products below fp16's smallest normal (6e-5) lose digits on the ANE: scale them up and divide later (Qwen3.5's GDN output, 16.88). `hwnpu-check` on the CPU does not show these: compare the per-position KL on the ANE with Core ML's CPU (`HWNPU_COREML_UNITS=cpu`)

### 3.4 Pass

- Every part is `all on the NPU`
- The second start takes a few seconds (the parts stay in the compile cache). In the split form (`-paged`), Gemma 4 went from 98 s to 1.9 s
- Record the KL and argmax of `hwnpu-check -package` (for each format, section 5)

## 4. The AMD (IRON) and Intel forms

### 4.1 AMD

| What to check | Lesson | Section |
|---|---|---|
| Does IRON have the same application? If not, write it from IRON's building blocks | LFM3 (`iron_lfm3.py`) and PLaMo 3 (`iron_plamo3.py`) were written from the building blocks | 16.66, 16.70 |
| Check against HF (400 rows of `ironlm -forced` against HF fp32 with `tools/ref/forced_logits.py`) | A bf16 export lands at KL 0.001–0.014, top-1 96–99 % (E2B, E4B, PLaMo 3, Qwen3.5). Long prompts (over 2000 tokens) too | 16.76, 16.102 |
| **The prompt's MHA on several pipelines** (`--mha-pipelines`, default 8) | Without a profile in the export, MHA ran on one pipeline, 60 % of a chunk. On 8, chunks are 2–5× faster with the same logits. A chunk of 256 takes 4 at most. Check `prompt attention: N pipelines` in the export's log and `num_pipelines` in the kernels' names | 16.107 |
| The chunk's size | 1024 for attention models (LFM3 too: short prompts −24 % against 2048, long ones the same). 512 for GDN models (Qwen3.5: 256 slows long prompts, 1024 short ones) | 16.54, 16.108 |
| Quantized projections in multiples of 64 out by 512 in | Pad a fused projection (GDN's in_proj, 6176 rows) per column with rows of zeros (`iron_gdn.column_widths`) | 16.104 |
| A command's control code within the device heap (64 MiB) | Dense models cut at `SCRATCHPAD_WORDS` 24 (Qwen3.5-2B's q4 at 32: `CREATE_BO: invalid argument`) | 16.104 |
| Where a slow chunk's time goes | `--command-per-step` (a command a step) and `HWNPU_IRON_TIMING`. Run one chunk only (16.80's hang) | 16.107 |
| One full ELF | Switching xclbins costs 0.46 ms each | 16.45–50 |
| Prompt chunks of 1024 | 20 % faster than 2048. At 256 the window layers become rings and cannot roll back | 16.54 |
| At most 3 transfers queued on one shim channel | Queuing 16 without waiting hung the command, and two hangs in a row needed a reboot. Wait for the oldest before queuing more | 16.89 (PR #66) |
| Each command under 2 s | Longer ones end in `ERT_CMD_STATE_TIMEOUT`. Split them at export (`SCRATCHPAD_WORDS`) | 16.65 |
| Window layers read only their window (`iron_window.py`) | Prompts −12 %. Without an empty command before each chunk (`chunkBarrier`) the second chunk hangs. The caches' rows are written as `cache_rows` | 16.84, 16.87 |
| KV-shared layers on the prompt's last row only | Gemma 4 E2B's prompts 2.3× faster | 16.80 |
| Tables on the host (PLE, embeddings) | E2B pinned 9.08 → 4.60 GB, the same output. With a head tied to the embeddings, put them on the host too once the head holds a quantized copy (Qwen3.5 −0.47 to −0.95 GiB) | 16.72–73, 16.104 |
| Towers loaded only when used (`growBanks`) | No text-only export needed | 16.81 |
| `-idle` gives back pinned memory | The next request pays 0.8–2 s | 16.74 |
| The cores' rounding to nearest even | Floor rounding pulls sums of inputs with a mean downward (Anima's DiT) | 16.78 |
| MHA's head_dim a multiple of 64 | Pad 96 to 128 (LFM3-VL's tower) | 16.65 |
| Images and audio in the final export form | Towers loaded on first use replace the banks; the chunk barrier bound to the old bank failed (ENOENT) until 2026-11-07. Send an image and a clip through `hwnpu-chat` after any change to the export | 16.81, 16.87 |
| Layers with state (convolutions) copied at chunk boundaries | For rolling back to marks and for the prompt store | 16.66 |
| **No** speculative decoding | Checking a draft costs 4.5 decode steps | |
| A graph's groups (`variants`, an image model's DiT) in one bank, read in turn (`iron.GraphOptions.SwapVariants`) | Anima's DiT, 7 groups: pinned 3445 → 1236 MiB, +2 % | 16.106 |

### 4.2 Intel

- To IR with `ovlower`. int8 by default (with int4 the next token matches fp16 only 22/52). For now only accuracy can be checked, on OpenVINO's CPU (speed and memory on a real NPU are not measured)

## 5. Quantization: choosing the format and the calibration method

**The stage that takes the most work and pays the most.** How much 4 bits lose differs tenfold between models (AMD q4 KL: PLaMo 3 on its translations 0.009, LFM3 0.036, E4B 0.049, E2B 0.097). Decide in this order.

### 5.1 Measure the same way

- **KL(reference ‖ quantized)**: mean, median, 95th percentile and top-1 agreement over 400 positions (the first 256 tokens read as a prompt, then one at a time), **with a BOS**
- Five texts: the README (Japanese and English mixed), GPL-3 (**not used for calibration**, to catch overfitting), the evaluation part of wiki40b-ja, the test part of wikitext-2, and Go code. They live in `~/models/calib`
- The reference is fp32 on the CPU (HF) or bf16 on the same NPU. Check that the quantized KL on the NPU matches the CPU estimate (LFM3 0.036 against 0.035; LFM3-VL the same way). If it does not, suspect the NPU's arithmetic (rounding, bfp16)
- For image generation, instead of KL: the velocity's relative RMS error along the reference trajectory, and the cosine of the final latents of a free run (`tools/amd/anima_gptq.py`)

### 5.2 Sweep the bits per part on the CPU

Switch each part (attention projections, MLP, output matrix, convolutions, towers) between 4 bits, 8 bits and bf16, and measure KL on the CPU (combine `--q8`, `--head` and `--fmt-*` of `q4_gptq.py --method rtn`). The conclusions so far:

- **Attention projections at 8 bits**: a little over 10 % of the weights, yet at 4 bits they cause most of the error (E2B 0.097 → 0.229). Only +0.12 GB pinned
- MLP and output matrix at 4 bits. The output matrix at 4 bits barely matters
- Short convolutions at 8 bits (+0.01 at 4 bits)
- Layers with sandwich norms (Gemma 4, PLaMo 3) are weak at 4 bits. With the ANE's tables, lut6 or more (Gemma 4 lut4 KL 2.54, PLaMo 3 1.91)
- lut8 for multimodal models' text parts, TTS, d1's judgments and embeddings (they broke down with lut4)
- A distilled 8-step DiT is weak at 4 bits (q4nx: latent cosine 0.777). q8 is the default (q6 keeps the composition)
- **A small model's (2B or less) MLP can break at 4 bits**: Qwen3.5-0.8B README 0.055 but GPL 0.34 (0.30 with GPTQ). AMD's q6 (6.5 bits, `--mlp q6 --head q6`) gives 0.044. Always look at the GPL. 4B makes 0.042 at q4 and stays at q4

### 5.3 Try the calibration methods in order

Keep the format and raise accuracy by choosing the codes (levels) better. The format does not change, so speed and memory stay the same.

| Order | Method | Tools | Example results | Notes |
|---|---|---|---|---|
| 1 | Rounding (RTN). q4nx picks the least error of 9 candidates | `iron_q4.quantize_fmt`, `palette.QuantizeBits` | Weight error 8.2 → 7.3 % | The baseline. The least weight error is not the least output error |
| 2 | **GPTQ** (pushes each column's error onto the later columns, with H = E[xxᵀ]) | `tools/amd/q4_gptq.py` → AMD: `iron_export_lm.py --q4-codes`; Apple: `hwnpu-convert -codes` | NPU wiki-ja: E2B 0.221 → 0.081, E4B 0.083 → 0.049, LFM3 0.071 → 0.038, LFM3-VL 0.067 → 0.034, PLaMo 3 0.058 → 0.040. Anima's DiT q4: latent cosine 0.868 → 0.939 (q8 0.962) | Damp by **0.1** × the mean of H's diagonal (0.01 overfitted the calibration texts and made the GPL worse than rounding). Calibrate on general text (wikitext-2 and wiki40b-ja, 64 × 512 tokens); the repository's documents alone overfit too. Leave the BOS position out of H. Quantize the earlier layers before measuring the next ones (a few layers a pass) |
| 3 | **Fitting the tables** (Apple's lutN): k-means weighted by E[x²] | `q4_gptq.py --lut-fit wkmeans` | LFM3 lut6 0.050 → 0.038, lut4 0.66 → 0.33 (through Go's converter) | Start from the uniform tables (starting from the k-means of least weight error gives the worst KL). The ANE check is in handoff-m3-gptq.md |
| 4 | AWQ-style scales folded into the previous op | Not written | | model-fit.md T8, 2. When lut4 has to go lower |
| — | One format up (AMD's q6) | `iron_export_lm.py --mlp q6`, `q4_gptq.py --fmt-q4 q6` | Qwen3.5-0.8B GPL 0.34 → 0.044 | When calibration does not fix it. The GEMV takes about q8's time; the gain is memory (76 % of q8) |

**Whether to adopt it**: the GPL, not used for calibration, must not get worse than rounding. For LFM3-Audio GPTQ leaves only the GPL unchanged (0.154 → 0.161), but the other four texts drop by half or more, so it was adopted. A multimodal model's text part is often trained again (LFM3-VL / Audio / Omni): calibrate each, do not reuse the base model's codes. If a text gets worse, raise the damping (0.3 was tried) or add calibration text.

**Cost of calibration**: fp32 on the CPU, 25–60 minutes for 2B models, 2.4 hours for E4B (over 30 GB in fp32; running two side by side ran out of memory). On Linux, run it with `systemd-run --user` (so it is not killed with the session).

**Matching the codes**: codes are looked up by a hash of the checkpoint weight's shape and bf16 values (`iron_q4.weight_key` = `palette.WeightKey`). If the export reorders or concatenates weights, they do not match. Check the count in the export's log, "N weights took the chosen codes" (LFM3-VL: 73; the q8 parts are rounded by default and not included). A weight quantized as one, like PLaMo 3's `gate_up_proj`, matches.

### 5.4 Choosing the format

| Guide | Decision |
|---|---|
| KL under 0.05 (against HF on the CPU) | port-architecture's guide. 4 bits can be the default |
| KL 0.05–0.2 | Decide by the quality of the answers (section 9). The ANE's lut6 (LFM3 0.232) is here |
| KL 1 or more | Not usable (lut4 LFM3 got 3 of 4 factual questions wrong) |

The trade with speed follows the weight bytes (ANE lut4 / lut6 / lut8: 1.81 / 2.65 / 3.49 ms). lut6's slowness was won back with `SDPAGrouped` (16.41).

## 6. Memory

| What to measure | Tools | Lesson |
|---|---|---|
| AMD pinned memory (the rise of `Unevictable`) | `/proc/meminfo` (not in RSS), the banks in the export's `manifest.json` | Mostly weights and work areas. KV is under 5 % |
| The Mac's `phys_footprint` | `npuprobe split` | The ANE's side (aned) is not measured (model-fit.md T10) |
| Size of conversation marks | | Copy only the ring rows in use (`ringRows`, 578 → 183 MB) |
| Right after a conversion | | `debug.FreeOSMemory` (2.4 GB → 154 MB) |
| KV size and context | | Past 1/4 of physical memory, shrink by whole chunks |
| Disk | `cache.Clone` | Share identical files (the Q=8 and Q=64 `weight.bin`, the common start of exports). Nothing is shared on ext4 |

**Pass**: enough numbers to add a row to the table in [improvements.md](improvements.md) 5.1.

## 7. Speed

| What to measure | Tools |
|---|---|
| One call (Q=8, Q=64) | `TestCallTimes` (`HWNPU_CALL_PKGS`) |
| Generation tokens/s, prompts, startup (the second) | `hwnpu-generate`, `hwnpu-chat`'s log |
| Comparing formats | `hwnpu-formats` |
| AMD GEMV, power | `tools/amd/gemvbench`, `powerbench` |

Things to bring in next (they helped every model):

- Two programs (generation and prompt reading) on one cache, reuse of the previous prompt, the prompt store (disk)
- Speculative decoding (n-grams from the prompt, Apple / Intel), the think budget (`-think-budget`), repetition detection

Compare only pairs measured back to back under the same conditions (on the M3 swap, on AMD the power state, make numbers vary by up to 1.4×).

## 8. Long contexts

- Split at the attention (`-paged`): the cost follows the length used. 262144 by default
- Two-level attention (`-sparse`): from about 48K. Passkey tests (128K, 250K, and three at 10 / 50 / 90 % of a document)
- Layers with state (GDN): one token at a time must match chunks of Q=64 (the form of `TestCachedMatchesFull`)

## 9. Checks in use, and the record

Do not decide by KL alone. Put the model's answers next to bf16 (or fp16) for what it is used for:

| Use | How to check |
|---|---|
| Chat | 16.26's four questions (Mt. Fuji, 17 × 23, *Kokoro*, the Shinano River), hwnpu-agent's task (does `go test` pass?) |
| Translation | Measure KL and agreement **on the translation's tokens only** (the source side is not predicted, so it scores high). How many of 44 texts are word for word the same as bf16 |
| Image descriptions, transcription, TTS | Compare the answers for the same images and audio. Run TTS output back through ASR |
| Embeddings | Cosine against HF |
| Image generation | Compare images from the same noise. The composition can change with some seeds (one in four) |

**The record**:

- Add a section to chapter 16 of design.md (numbers, what was adopted, what was not and why)
- The tables in [improvements.md](improvements.md) and [improvements.ja.md](improvements.ja.md) (speed, memory, accuracy, the model list in section 10, not yet verified in section 11)
- What could not be measured on a Mac goes into `docs/handoff-*.md`, with what to bring, the commands and the expected values

## 10. Values for combining it as a part

Later, models will be chained, such as speech recognition → LLM → TTS, and run at the same time in one process (design.md 16.94, a plan, not implemented). Measure and record the values that the residency management and the matching with the connected models will need when the model comes in.

| What to record | How to measure | What it is for |
|---|---|---|
| The pinned size of each part (stage, tower, DiT, VAE, ...) | AMD: the size of the process's `/dev/accel/accel0` mappings (`smaps`). Mac: each part's weights and `phys_footprint` | The memory budget |
| Each part's reload time (the second load) | AMD: `Release` / `Reload`. Mac: when the ANE's cache hits | Whether it may be evicted. Parts whose delay is heard, as in a speech chain, stay resident |
| Parts that run together, and parts that run in turn | From the flow of the processing | Only parts that run in turn can be loaded a stage at a time (Anima's text → DiT → VAE) |
| AMD: the longest command | Logs of `hwnpu-chat` / `irongraph` | Other models wait for it. The further below 2 s (4.1), the easier to run beside others |
| Mac: the parts' total weights and the switching time | `TestCallTimes`'s alternating calls | Past the ANE's cache (about 4 GiB), parts are evicted and recompiled in minutes. A switch costs 15–35 ms |
| The data kinds and conditions of the inputs and outputs | The model card and the preprocessing | Text (tokenizer and languages), structured text (can it follow a schema), images (sizes it takes), audio (sample rate, channels). For matching against the connected models |
| Whether it can stream, and the time to the first output | Feed it chunks and take chunks out | Estimating the delay of a speech chain |

- Pass data to other models as text (token IDs when the tokenizer is the same). Hidden states are not compatible with another model's, even with the same tokenizer. Connect vectors directly only between parts trained together (LFM3-Audio's text model and depthformer, a tower and its projection, Anima's text encoder and adapter)
- Record a model that keeps audio as audio (Audio, Omni) as one part whose input and output are audio

**Pass**: the values in the table above are written in the design.md section.

## 11. Tried without effect (do not repeat)

The gist of section 8 of [improvements.md](improvements.md): padding small models into batches for the ANE (slower than the CPU), fixing a small Qwen3.5's 4-bit MLP with GPTQ (the GPL stays broken; q6 instead), a faster GEMV from q6 (q8's time), getting past 1 GiB with multifunction (all on the CPU), the KV cache in MLState, an int8 KV cache, loading the next tier in the background, one Q=16 program, E4B at lut4, compressing the model, sparse attention on the top keys alone, AMD attention at 4 bits, a GEMV with one core per column, copying MoE experts, the DiT at `HIGH`, the DiT at q4 by default, q4nx prompts through bf16, RMSNorm with a fixed scale.

## 12. In progress (2026-11-08)

- The AMD exports here were exported again with 8 pipelines and the new chunk sizes (16.108). When an export changes, pass an image and a clip through `hwnpu-chat`
- Qwen3.6's prompt attention pipelines are not measured (it keeps 1)

### As of 2026-11-07

- Done on AMD: GPTQ codes for E2B, E4B, LFM3 (and VL / Audio / Omni) and PLaMo 3, measured on the NPU (amd-npu-models.md 13, "continued"). Against bf16: E2B with images and audio gave the same answers to text, an image and a clip; PLaMo 3's translations word for word the same 14/20 rounded, 17/20 with GPTQ
- Apple: checking the GPTQ lut6 / lut4 tables on the ANE ([handoff-m3-gptq.md](handoff-m3-gptq.md)). hwnpu-chat does not take `-codes` yet
- Anima: the 4-bit DiT with GPTQ (latent cosine 0.939 on the CPU at 512×512) not yet checked as an image at 1024×1024 on the NPU

When results come in, update the table in 5.3 and this section.
