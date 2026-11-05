# What each improvement bought (as of 2026-11-06, main up to #46)

[日本語](improvements.ja.md)

This page sorts the changes by what they improved. It draws on the implementation log in design.md chapter 16 (up to the MoE measurements in 16.75 and to 16.77), model-fit.md, intel-npu-models.md, amd-npu-models.md (sections 11–13) and handoff-anima-mac-intel.md. The numbers are the measurements recorded there. Unless noted, they are from **the Neural Engine (ANE) of an Apple M3 (16 GB)** with LFM3-1.7B. AMD numbers are from a Ryzen AI 9 365 (Strix Point). Numbers in parentheses are design.md sections.

Main additions since the previous version (up to #39):
- **4-bit and 8-bit weights on AMD** (q4nx and q8). Pinned memory drops to 1/2–1/5, and generation is 1.3–1.6 times faster
- **Less pinned memory on AMD**. The PLE and embedding tables stay on the host, Anima loads one stage at a time, and the NPU is released while idle (`-idle`)
- **PLaMo 3 Translate 2B** on all three kinds of NPU (on an M3, checked as far as placement and weight formats)
- **Anima image generation** (`hwnpu-image`). It runs on AMD; on Apple and Intel it is checked on the CPU only
- **Three more tokenizers** (PLaMo, T5's Unigram, Qwen3's pre-tokenizer)
- **Measurements of MoE building blocks at Qwen3.6-35B-A3B's sizes**. Not a port of the model
- **Host memory fixes**. A conversation mark keeps only the ring slots in use, memory goes back to the OS after a conversion, and the `/dev/shm` left by OpenVINO is cleaned up

- "Generation" = how fast one token is made (ms/token or tokens/s)
- "Prefill" = how fast the prompt is read (time to the first answer)
- "Startup" = time to load the model (second and later starts)
- There are no GPU optimizations. Using the GPU is a non-goal (the aim is to be fast on the NPU alone and to cut energy). Section 7 lists **the work for each kind of NPU (Apple / Intel / AMD)** instead

## 1. At a glance

| Change | Generation | Prefill | Startup | Memory | Disk | Context | Accuracy | Where |
|---|---|---|---|---|---|---|---|---|
| Split by layers (past the 1 GiB limit) (16.12–13) | ○ fits on the ANE | | ○ | ◎ RSS ≈ one part | | | | Apple |
| KV cache (16.14) | ◎ 34 → 17–21 ms | | | | | ○ | | All |
| Sliding layers as rings (16.16) | ○ −26% at 4K | | | ○ KV −43% | | | | All |
| Two programs (generation and prefill) on one cache (16.17) | | ◎ 860 tok 35 s → 1.6 s | | △ weights loaded twice | | | | All |
| Reusing the previous prompt (in memory) (16.17) | | ◎ 0.1–0.5 s from the second turn | | | | | | All |
| Cache in chunks (past 16K) (16.21–22) | | | | | | ◎ 16K → 64K | | Apple |
| `SDPAGrouped` (grouped attention) (16.25, 16.39) | ◎ −45% | ○ −25% | | ○ no tile copies | | | | Mainly Apple |
| lut6 / lut8 (weight formats) (16.26, 16.31, 16.41) | △ heavier is slower | | △ | △ bigger weights | | | ◎ KL 1/6–1/10 | Apple |
| Think budget `-think-budget` (16.32) | ◎ felt (stops minutes of spinning) | | | | | | | All |
| Speculative decoding (prompt lookup) (16.33, 16.36–37) | ○ 1.2–1.5× | | | | | | unchanged | Apple / Intel |
| KV tiers `-tier` (16.20, 16.37) | ◎ 1.95× (short chats) | ◎ over 2× | ✕ worse with lut6 | △ +0.5–0.7 GiB | | | | Apple (now lut4 only) |
| Caching the placement result (16.20) | | | ○ 42 → 16 s | | | | | Apple |
| Prompt store (disk) (16.38, 16.52) | | ◎ 7.9K tok 14.6 s → 0.2 s | | | △ about 145 MB a sequence | | | All (AMD too) |
| hwnpu-agent (short system prompt) (16.35) | | ◎ 7.5K → 629 tok | | | | | | All |
| Split at the attention (`-paged`) (16.59, 16.63) | ◎ 1.3–1.7× | | ◎ Gemma 98 s → 1.9 s | ◎ only the length used | | ◎ 262144 by default | unchanged | Apple / Intel |
| Two-level attention (`-sparse`) (16.61, 16.65) | ○ −10–23% at long contexts | ○ −9–27% | | △ +1/8 | | | △ approximate | Apple / Intel |
| Repetition detection (16.60) | ◎ felt (stops going round in circles) | | | | | | | All |
| E4B as one program (`-prefill auto`) (16.65) | ○ +25% | ✕ 4.4× slower when long | ◎ 77–87 s → 4 s | ◎ half the weights | | | | Apple |
| Identical weights shared with clonefile, cleanup (16.64) | | | △ slow once right after the switch | | ◎ about 50 → 20 GiB | | | macOS |
| AMD: full ELF (16.45–50) | ◎ (avoids a 0.46 ms switch per op × 100) | | ○ | | | | | AMD |
| AMD: prompt chunk 2048 → 1024 (16.54) | | ○ −20% | | | | | | AMD |
| AMD: 4-bit / 8-bit weights `--q4` (16.73, 16.76–77) | ◎ 1.3–1.6× | unchanged | | ◎ pinned 1/2–1/5 | ○ about half | | △ KL 0.009–0.097 | AMD |
| AMD: PLE and embedding tables on the host (16.72–73) | unchanged | unchanged | | ◎ E2B 9.08 → 4.60 GB | | | unchanged (word for word) | AMD |
| AMD: release the NPU while idle `-idle` (16.74) | | | △ reload 0.8–2 s on the next request | ◎ 0 while idle | | | | AMD |
| Anima loaded one stage at a time (16.72) | 69.3 → 67.7 s/image | | | ◎ 14.2 → 5.0 GiB | | | bit for bit the same | AMD |
| Conversation marks keep only the ring slots in use (16.71) | | | | ○ 578 → 183 MB over 5 translations | | | | All |
| Memory back to the OS after a conversion (16.71) | | | | ○ 2.4 GB → 154 MB right after startup | | | | All |
| Cleaning up the `/dev/shm` left by OpenVINO (16.72) | | | | | ○ 8.4 GB a day | | | Intel / OpenVINO |

◎ large effect, ○ helps, △ small or with a cost, ✕ makes it worse

## 2. Generation throughput

| Change | Before | After | Notes |
|---|---|---|---|
| KV cache (16.14) | 34.2 ms/token (a window of 128 every time) | 16.8 ms (c256), 20.6 ms (c1024) | The cost follows the **capacity**, not the amount used (about 5.4 ms per 1K) |
| Rings for the sliding layers (16.16) | c4096 37.6 ms | 27.7 ms (−26%) | Only the 6 full-attention layers have a long cache. 5.4 → 2.8 ms per 1K |
| `SDPAGrouped` (Gemma 4, 16.25) | c16320 90–150 ms | 47 ms | No more copying K/V 8 times with `tile` |
| `SDPAGrouped` (LFM3, 16.39) | c16320 Q=8 73.5 ms | 40.7 ms (−45%) | 18–24 → 27–39 tokens/s. Answers word for word the same |
| lut6 + grouped as the default (16.41) | lut4 + tile 69.2 ms | 42.8 ms (−38%) | Accuracy rose too (KL 1.387 → 0.232). A coding task went from 27 round trips to 6 |
| Speculative decoding (16.36) | 15.9 tokens/s (Q=1) | 23.7 tokens/s | 1.5–2× when copying, 1.2–1.5× over a whole task, a little over 10% lost when only thinking. The output distribution is unchanged |
| n-grams 3..2 → 4..1 (16.37) | 1.552 tokens/call | 1.673 (+7.6%) | Checking is free at Q=8, so 1-grams are drafted too even though they hit less |
| KV tiers (c4096 → c16320) (16.37) | 19.4 tokens/s | 37.8 tokens/s (1.95×) | lut4 only. With lut6 startup gets worse, so `-tier auto` turns it off (section 5) |
| Split at the attention (16.59) | c16320 (unsplit) 19–28 tokens/s | 31–36 tokens/s (1.3–1.7×) | The cost now follows **the length used**. 42.7 ms/token at 16K, 92.8 at 64K, 165 at 128K, 320 at 250K |
| Same (Gemma 4 E2B, 16.63) | 13.3–28.2 tokens/s | 23.6–36.0 tokens/s | Not one token of the answers differs |
| Same (LFM3-VL / Audio, 16.65) | 11.9–29.4 / 16.3–23.4 | 24.0–36.0 / 18.8–28.6 | |
| Two-level attention (Gemma 4 E2B, 16.65) | 82.4 ms at 64K, 119.5 at 125K | 73.9, 92.4 ms | For LFM3 on an M3 there is only an estimate (165 → about 90 ms at 128K) |
| E4B as one program (16.65) | 10–13 tokens/s | 13.4–16.4 tokens/s | No more reloading weights when switching between Q=8 and Q=64 |
| Think budget (16.32) | A title request thought for 3440 tokens, 226 s | Cut at 1024 tokens and made to answer | Cuts waiting time, not the time per token |
| Repetition detection (16.60) | Could run for hours with a 262144 context | Finds the loop and stops it / closes the thinking | Never stopped any of 24 normal answers |

**The effect of 4 bits (`--q4`) on AMD:** the MLP and output matrices are q4nx, and the attention (and LFM3's convolution) projections are q8. Prompt reading speed is unchanged.

| Model | Generation (bf16 → q4) | Energy per token | Notes |
|---|---|---|---|
| Gemma 4 E2B (16.73) | 6.1 → 7.9 tokens/s | 2.31 → 1.46 J | Of the remaining ~125 ms, the weight GEMVs take a little over 30 ms. The rest is attention (reading the whole context) and small ops |
| Gemma 4 E4B (16.73) | 3.7 → 5.6 tokens/s | | |
| LFM3-1.7B (16.77) | 8.7 → 11.2 tokens/s (117 → 92 ms at position 2048) | 1.34 → 1.05 J | VL: 8.7 → 11.3 tokens/s, 1.35 → 1.03 J |
| PLaMo 3 Translate 2B (16.76) | 7.3 → 11.6 tokens/s | | Measured with another job loading the machine; likely faster on a quiet one. Reading weights is most of the decode time, so the gain is large |

Improvements to the GEMV itself:
- With 2 cores per column it reaches 55 GB/s, the same as bf16. With 1 core it is bound by compute
- The fused `qkv_proj` and `gate_up_proj` are one `QGEMV` call in decode (6 % faster than splitting)

## 3. Prefill (time to the first answer)

| Change | Before | After | Notes |
|---|---|---|---|
| A Q=64 program for prefill (16.17) | 860 tokens 35.4 s (Q=1) | 1.6 s | Shares the cache with the generation program (Q=1/8) |
| Reusing the previous prompt (16.17) | Reads 7.4K every time (13 s) | 99% reused from the second turn, 0.1–0.5 s | Marks for the rings and convolution windows (about 3 MB), and two conversations set aside |
| Grouped attention (16.39) | OpenCode's 7.9K 14.6 s | 10.9 s | |
| KV tiers (16.37) | Total prefill 15.2 s | 6.7 s | With lut4 |
| Prompt store (16.38) | 7,854 tokens after a restart 14.6 s | 0.2 s; 0.7 s even with a changed text | Writing adds under 0.5 s. Greedy output matches exactly |
| Same (AMD, 16.52) | 2,736 tokens 13.2 s | 0.0 s | 51 MB an entry |
| hwnpu-agent (16.35) | OpenCode's first request 7,516 tokens (13 s) | 629 tokens (1.3 s) | A short system prompt with 4 tools |
| Two-level attention (16.65) | Gemma 4 E2B at 125K 709 s | 521 s | |
| AMD: prompt chunk 1024 (16.54) | 2,683 tokens 13.0 s | 10.2 s | Short ones too: 0.56 → 0.43 s |
| 4 bits on AMD (16.73, 16.76–77) | — | Unchanged (LFM3 2048 tokens 2.29 → 2.38 s, PLaMo 3 2596 tokens 11.2 → 11.8 s) | Weights are turned back from q4nx and q8 into bfp16 or bf16 on the device for the GEMM, so prompt speed does not gain |
| PLaMo 3's translation history (16.70–71) | Reads the whole history of pairs every time | From the second text on, reads only 17 of 50 tokens | When the target language is chosen automatically, the label of each past translation is fixed by its source text. Otherwise the start of the prompt could not be reused |
| Coming back from `-idle` (16.74) | — | 830 tokens restored from the host copy, only the next question's 58 tokens read | |

## 4. Startup time

| Change | Before | After | Notes |
|---|---|---|---|
| hwnpu's cache of compiled models (16.13) | First run 15–30 s | 0.8–1.1 s | |
| Caching Core ML's placement (`MLComputePlan`) (16.20) | 42 s in all | 16 s (placement check 5–21 s → 0.01 s) | |
| Loading every tier up front (16.20) | Loading in the background made generation over 2× slower | All at startup by default | Longer startup, but a net gain |
| `-tier auto` (16.41) | 4 programs with lut6 (4.8 GiB): 62–110 s every time | 2 programs (2.4 GiB): 3 s | The ANE's compile cache is an LRU of about 4 GiB. Past that, everything is recompiled every time |
| Split at the attention (Gemma 4 E2B, 16.63) | 98–109 s every time (the lut6 parts do not stay in the cache) | **1.8–1.9 s** | Smaller parts stay. `-logits folded` and `-parts 3` did not help (16.43) |
| Same (LFM3, 16.59) | | 1.0 + 1.2 s from the third start | 14 parts plus 2 attention programs still fit in 2.4 GiB |
| E4B as one program (16.65) | 77–87 s | 3.7–5.5 s | Two programs (6.5 GiB) do not fit in the cache |
| Sharing with clonefile (16.64) | — | Once, right after the switch, 20–50 s a program | New conversions share from the start, so this happens once |
| AMD (16.47, 16.50) | — | xclbin 0.5 ms, full ELF 1.0 ms, a 9 GB image 2–3 s from the page cache | |
| AMD reloading after `-idle` (16.74) | — | E2B (4-bit) 0.8 s, E2B with images and audio 2.0 s | The Neural Engine and Intel take tens of seconds or more to reload, so only IRON releases |
| Trimming trailing zeros from Anima's bank images (amd-npu-models.md 11.2) | VAE image 8 GB, loads in 10 s | 0.2 GB, 4 s | Startup reads only the manifest (0.5 s); each stage loads when it runs |
| PLaMo 3 as one lut8 program (`-prefill auto`) (16.71) | Two programs (4.9 GiB) exceed the ANE's compile cache and are recompiled every time | One program fits the cache | The cost: a document of thousands of tokens is read about 10× slower. `-format lut6` keeps two programs |

## 5. Memory

| Change | Effect | Notes |
|---|---|---|
| Split by layers (separate packages) (16.13) | Peak RSS ≈ one part (a 1.81 GiB model: 1.03 GiB in 2 parts, 0.55 GiB in 4) | multifunction (several functions in one file) cannot get past the limit and falls to the CPU (5.26 GiB) |
| Rings for the sliding layers (16.16) | KV 88 → 50.5 MiB (c4096) | |
| Gemma 4's KV sharing (16.25, 16.63) | 6 KiB a token (0.75 GiB at 131072) | Sharing saves only memory. The amount of attention computed is the same |
| Split at the attention (16.59) | KV is allocated only as used (12 KiB/token). phys_footprint 416 MB at 16K → 3.2 GB at 256K | `Save` also copies only the rows used (before: the whole cache, 3 GiB each time at 256K) |
| Shrinking the context to fit memory (16.59) | If the cache is over 1/4 of physical memory, it shrinks by whole chunks | Up to 4 GiB on a 16 GB M3 |
| PLE table on the host (16.25) | Gemma 4's 4.7 GB table is kept off the ANE | |
| E4B as one program (16.65) | Resident weights 6.5 → 3.26 GiB | |
| KV tiers (16.37) | **Cost**: 2 programs 2.29 GiB → 4 programs 2.73–2.96 GiB (RSS), footprint 1,968 → 2,807 MB | Each tier has its own copy of the weights |
| Two-level attention (16.61) | **Cost**: +1/8 for the summaries (6 → 7 KiB/token) | |
| Conversation marks keep only the ring slots in use (`ringRows`) (16.71) | 578 → 183 MB over 5 PLaMo 3 translations (live heap +85 MB a turn → 36 MB in all) | Each mark used to copy the whole ring of the window layers (88 MB). Helps short chats with Gemma 4 and LFM3 too |
| `debug.FreeOSMemory` after a conversion (16.71) | Startup right after a conversion 2.4 GB → 154 MB | Go was slow to give back what was left of the checkpoint read during the conversion |

**Pinned memory on AMD** (16.72–74, amd-npu-models.md sections 12–13). Every page of an NPU buffer is pinned, so it shows neither in the process's RSS nor in swap. What grows is `Unevictable`. Nearly all of it is weights and work areas; the KV cache is under 5 %.

| Model | Original (bf16) | Tables on the host | 4-bit (attention q8 by default) | While idle (`-idle`) |
|---|---|---|---|---|
| Gemma 4 E2B | 9.08 GB | 4.60 GB | **1.78 GB** | 0 |
| Gemma 4 E4B | 14.7 GB | 9.25 GB | **3.53 GB** | 0 |
| Gemma 4 E2B (images and audio) | 10.1 GB | 5.6 GB | **2.82 GB** | 0 |
| LFM3-1.7B | 3.75 GiB | — | **1.64 GiB** | 0 |
| LFM3-VL / Audio / Omni | 4.12 / 4.14 / 4.51 GiB | — | **2.00 / 2.03 / 2.39 GiB** | 0 |
| PLaMo 3 Translate 2B | 5.96 GiB | — | **2.16 GiB** (the 0.51 GiB embedding is a file on the host) | 0 |
| Anima (1024 × 1024) | 14.2 GiB | **5.0 GiB** loading one stage at a time | — | — |

- Tables on the host: PLE reads only one row a token, yet the whole table was pinned. The table now stays a file, and before each call only the tokens' rows are written into a small device buffer (18 MB for E2B)
- The embedding is shared with the output matrix, so making the output 4-bit leaves no bf16 rows. The embedding became a host table the same way

## 6. Accuracy

| Change | Before | After | Cost |
|---|---|---|---|
| Gemma 4's norm on `x/64` (16.25) | KL 3.17 with `-f16` (fp16 overflow) | KL 0.0004 | None (mathematically the same) |
| Gemma 4 to lut6 (16.26) | lut4: KL 2.54, argmax 13/31 | lut6: KL 0.25, 25/31 | 23 → 31.5 ms a token |
| LFM3 to lut6 (16.39, 16.41) | lut4: KL 1.387, argmax 16/32 | lut6: KL 0.232, 23/32 | Weights +0.4 GiB. Speed was won back with grouped attention |
| Multimodal and d1 text parts to lut8 (16.31) | With lut4, TTS, image descriptions and d1's judgments broke down | Answers as in the README | Two parts. TTS 1.4 → 0.5× real time |
| Embeddings with lut8 + head-scale (16.34) | Cosine 0.76 with lut4; 0.9745 on the ANE with lut8 alone | 0.99986 (0.997–0.999 against HF) | None |
| int8 as Intel's default (intel-npu-models.md section 11) | int4: next token matches fp16 22/52 | int8: 51/52 | Weights twice int4's (1.6 GiB) |
| Search and fetch tools (16.40) | lut4 LFM3 got 3 of 4 factual questions wrong | 4 of 5 with SearXNG | More round trips |
| AMD: attention projections at 8 bits (16.73) | Attention at 4 bits too: E2B KL 0.229, 84.8 % match | q8: KL 0.097, 90.0 % match (0.091 with bf16) | +0.12 GB pinned. Attention is a little over 10% of the weights, yet at 4 bits it causes most of the error |
| AMD: q8 GEMV with the partial sum in two bf16s (16.73) | Rounded to one bf16: cosine 0.9998 | Split into high and remainder: 0.999992 | None |
| AMD: q4nx quantization choosing from 9 candidates (amd-npu-models.md section 13) | Plain min-max: weight error 8.2 % | 7.3 % | None; at export only |
| AMD 4-bit KL by model | | LFM3 0.036 (92 % match), E4B 0.049, E2B 0.097, PLaMo 3 0.007–0.009 on its translations (36 of 44 the same as bf16) | LFM3 and translation hold up well at 4 bits. Measure LFM3 with a BOS (without it, it looks 4× worse) |
| Anima: `HIGHEST` for the text only (16.68) | All `MEDIUM` (bfp16): latent cosine 0.938 | 0.966 (bf16 on the CPU: 0.980) | Same time |
| Anima's fp16 handling (Apple / Intel) (16.69) | DiT residuals reach 2.7e5 and overflow fp16 | Residuals kept at 1/32, norms divided by each row's maximum first: latent cosine 0.99985 on the CPU | None |
| PLaMo 3 to lut8 (16.71) | lut4: KL 1.91; lut6: 89 % match | lut8: 98 % match (M3) | Weights 1.82 → 2.45 GiB |

## 7. Work for each kind of NPU

### 7.1 Apple Neural Engine (Core ML) only

Fitted to the ANE's limits: 1 GiB a package, 16384 per dimension, tables only per 16 rows, `gather` on the CPU, fixed shapes, a compile cache of about 4 GiB.

| Change | What it helps | Numbers |
|---|---|---|
| Split by layers (separate packages) | Puts models over 1 GiB on the ANE. Memory | 0.1–0.16 ms per boundary (M3/M4) |
| Cache in chunks | Gets past the 16384 limit (32K, 64K) | Summing with a matmul instead of `reduce_sum` keeps placement stable (ANE 30 times of 30) |
| lut4 / lut6 / lut8 | Choice between accuracy and speed | Speed follows the weight bytes (1.81 / 2.65 / 3.49 ms). The Q4_0 layout is 7× slower or goes to the CPU |
| `SDPAGrouped` | Generation −45% | 16.15's "queries as a matrix" form fell to the CPU. The ordering changes the placement |
| Folding the softcap to (16, 16384) (Gemma 4) | Keeps a 262144 axis from falling to the CPU | |
| Caching the placement check | Startup 42 → 16 s | |
| `-tier auto`, E4B as one program (`-prefill auto`) | Fits the ANE's compile cache (about 4 GiB) for fast startup | 4 programs 62–110 s → 2 programs 3 s; E4B 77–87 s → 4 s |
| Attention in programs without weights | Smaller parts stay in the ANE's cache; the cost follows the length used | Gemma 4 startup 98 s → 1.9 s |
| head-scale (embeddings), `-post-scale` (LFM3) | fp16 overflow and the error floor on the ANE | |
| clonefile (APFS) | Disk about 50 → 20 GiB | Does nothing outside macOS |
| PLaMo 3's attention output taken token-first (`attn_ot_N`) | Avoids the one `transpose` in the last part that fell to the CPU | Everything on the ANE on an M3. The host reorders (one more copy) |
| Anima's fp16 handling (DiT residuals at 1/32, `maxScaled` norms) | No fp16 overflow and no subnormals | Not yet checked on hardware |

### 7.2 Intel NPU (OpenVINO IR / Level Zero)

| Change | What it helps | State |
|---|---|---|
| Lowering MIL to OpenVINO IR (`ovlower`) | One converter for both Apple and Intel | Checked on OpenVINO's CPU |
| int8 / int4 / int4-gN formats (read compressed by the NPU) | int8 is nearly as accurate as fp16 | int8 by default |
| IR split at the attention | 262144 context | Checked on OpenVINO's CPU |
| Padding rows filled with the last token | Prevents Gemma 4's NaN (eps becomes 0 in fp16) | Helps Apple too |
| PLaMo 3's IR (int8, split at the attention) | Translation | Same translations as AMD on OpenVINO's CPU |
| Anima's IR (`hwnpu-image -convert -backend intel`) | Image generation | On OpenVINO's CPU, latent cosine 0.99985, PSNR 37.9 dB |
| Cleaning up crashed processes' `/dev/shm` on open | Leftover shared buffers (8.4 GB in a day) | |

**Note: neither speed nor memory has been measured on a real Intel NPU.** Every number here is an accuracy check on OpenVINO's CPU.

### 7.3 AMD XDNA (Ryzen AI, IRON)

Measured on a Ryzen AI 9 365 (Strix Point, NPU2).

| Change | What it helps | Numbers |
|---|---|---|
| full ELF (the whole graph in one context, one command) | Avoids switching xclbins per op (0.46 ms each, 46 ms for 100 ops) | Gemma 4 E2B 5.5–6.1 tok/s, LFM3-1.7B 8.7 tok/s (bf16) |
| Prompt chunk 2048 → 1024 | Prefill −20% | 256 is as fast, but Gemma 4's sliding layers become rings and can no longer roll back |
| **No** speculative decoding | A draft-checking call costs 4.5 decode steps | Loses unless over 4.5 tokens are accepted per call |
| Prompt store and setting conversations aside | Prefill after a restart 13.2 s → 0.0 s | LFM3 copies its convolution state at each chunk boundary to roll back (16.66) |
| CLFLUSH from user space, a flush right after mapping | Fixes first results coming out as 0 from dirty cache lines | |
| One xclbin per step (for NPU1) | Runs on Phoenix / Hawk Point, which have no full ELF | 2.6 tok/s (full ELF 6.1). Not yet run on a real NPU1 |
| xrtshim (Windows) | Windows through XRT | 6.1 tok/s on Linux. Not yet run on Windows hardware |
| Power (PPT) | Measurement | 11.8 W while generating, 1.93 J a token (idle included). GEMV at 60 % of the CPU's speed, 3–5 mJ per 32 MiB (CPU 7–12 mJ) |
| 4-bit (q4nx) and 8-bit (q8) weights (`iron_q4.py`, `q4_mv.cc`, `q8_mv.cc`) | Generation 1.3–1.6×, pinned 1/2–1/5 | Decode uses `QGEMV`; with 2 cores per column it reaches 55 GB/s like bf16. Prompts turn weights back into bfp16 or bf16 for the GEMM. Swapped in on hwnpu's side without changing IRON |
| Host tables (`iron.HostTable`) | PLE and embeddings are not pinned | The tokens' rows are written before each call. Output word for word the same |
| `-idle` (`Release` / `Reload`) | A resident server gives back pinned memory while idle | E2B 1775 MiB → 0. The conversation is copied to the host and continues |
| A general graph export (`iron_graph.py`, `iron.Graph`) | Models other than language models (Anima) | One ELF can hold several weight banks (`variants`). The DiT's 7 sets compile once |
| Splitting commands (`SCRATCHPAD_WORDS` 24) | Keeps each command under amdxdna's 2 s timeout and the 64 MiB heap | Without the gather, commands grew too long; they are now split at export |
| Retrying when the heap runs out (`ENOSPC`) | Releases the decode run and tries again | Same as IRON |
| MoE building blocks (at Qwen3.6-35B-A3B's sizes, 16.75) | Reads the chosen experts in place without copying (`ExpertQGEMV`). A layer in 4 ops (`MoEGateUp`, `MoEDown`) | 5.2 → 2.3 → 1.27 ms a layer. Estimated 8–10 tokens/s. Pinning 35 GB does not change the speed |

### 7.4 Common (any NPU, on the host)

Prompt reuse and the prompt store, speculative decoding (Apple / Intel), the think budget, repetition detection, hwnpu-agent's guards (no editing a file it has not read, stopping repeated calls, refusing a `cd` outside), and the search tools.

## 8. Tried without effect, or not adopted

| Tried | Result | Section |
|---|---|---|
| Padding small models into batches for the ANE | 73 µs (CPU) → 280 µs (ANE). A loss | 16.6 |
| Splitting inside one file with multifunction | The 1 GiB limit counts the whole package, so all of it went to the CPU | 16.13 |
| KV cache in MLState | Same speed. Frees only 0.15 of a CPU core (kept as an option for power) | 16.15 |
| int8 KV cache | Half the memory, 10% slower. KV is under 10% of resident memory, so not adopted | 16.15 |
| Loading the next tier in the background | Generation over 2× slower | 16.20 |
| One Q=16 program for generation and prefill | +15% with grouped attention; not worth it | 16.40 addendum |
| Draft width Q=4 | Only 3% cheaper than Q=8, 10% fewer hits | 16.37 |
| Gemma 4's `-logits folded`, `-parts 3` | Startup unchanged | 16.43 |
| E4B at lut4 | Only 20% faster, matches 90% → 64% | 16.65 |
| Compressing the model (lz4, zstd) | 71–100%. Mapped files would need unpacking | 16.64 |
| Sparse attention picking single keys (top-k) | Catches only 53% of the weight at 128K. Replaced by two levels (bucket means) | 16.61 |
| AMD: attention projections at 4 bits too (`--attn q4`) | Saves only 0.12 GB pinned, E2B KL 0.097 → 0.229 | 16.73 |
| AMD: q4nx GEMV with 1 core per column | Compute-bound (36 GB/s on the output matrix). Went to 2 cores | amd-npu-models.md section 13 |
| AMD MoE: copying the experts, then GEMV | Moves the chosen experts' bytes three times. `Copy` runs at 26 GB/s, slower than the GEMV. Replaced by reading in place | 16.75 |
| AMD MoE: down as "one column = one expert" | Fast as a GEMV alone, but with 7 adds to sum the 8 outputs it is 1.7× `MoEDown` | 16.75 |
| AMD MoE: the transposed down as a one-row GEMM | Slower than the GEMV (1.92 ms) | 16.75 |
| Splitting Anima's DiT into commands with host steps | The first command stopped with `ERT_CMD_STATE_TIMEOUT`. Made one graph of 4 blocks instead | 16.68 |
| Anima's DiT at `HIGH` | Latent cosine 0.966 → 0.976, but 7.2 → 10.9 s a step | amd-npu-models.md 11.3 |
| RMSNorm with a fixed scale (Anima, fp16) | Squares of rows without large values become subnormals. Replaced by dividing by each row's maximum | 16.69 |

## 9. Caveats when combining (costs)

- **lut6 and tiers (`-tier`)**: tiers with lut6 exceed the ANE's cache (about 4 GiB) and every startup takes 1–2 minutes. `-tier auto` makes tiers only with lut4. The split form (`-paged`, now the default) has no tiers
- **E4B as one program**: startup and short chats are fast, but reading a 16K prompt goes from 41.6 to 182.6 s (4.4×). `-prefill on` brings back two programs
- **Two-level attention**: it is an approximation, so the error grows when attention focuses on a single key (relative error 0.068–1.6 on a small Gemma 4). It passes the passkey test. `-sparse=false` turns it off
- **Prefill of long contexts**: reading 256K at once takes about 29 minutes (LFM3). The prompt store is assumed
- **Noise in the numbers**: many M3 measurements were taken with 5–11 GB of swap. Only pairs measured back to back under the same conditions are compared. On AMD too, a single op can differ by about 1.4× between the first and second run (thought to be the power state). PLaMo 3's 4-bit numbers were measured with another job loading the machine
- **4 bits on AMD**: how much accuracy drops depends on the model (KL 0.036 for LFM3, 0.097 for E2B). For bf16, export without `--q4`
- **`-idle`**: memory is given back while idle, but the next request first pays 0.8–2 s to reload. `-idle 0` turns it off
- **PLaMo 3 as one lut8 program**: short texts are fast, but long documents are read about 10× slower (`-format lut6` gives two programs at lower accuracy)

## 10. Support for each model

### 10.1 Overview

"Port" means Hugging Face's `modeling_*.py` moved to Go (`internal/llm/<arch>`). For Apple and Intel, hwnpu's converter (Go only) converts the model. For AMD, an IRON application is exported and hwnpu runs it from Go (amd-npu-models.md).

| Model | Port (arch) | Tokenizer | Chat format | Images / audio | Apple (ANE) | Intel | AMD |
|---|---|---|---|---|---|---|---|
| LFM3-1.7B | `lfm3` | byte-level BPE (`tokenizer.json`) | LFM3 | — | ◎ lut6 + grouped, context 262144 | Converts (int8), not run on hardware | ◎ q4 11.2 tok/s, 1.64 GiB pinned (bf16 8.7 tok/s), context 12288 |
| LFM3-VL-2B | `lfm3` + `lfm3-vision` | Same | LFM3 | Images (512×512) | ◎ lut8, context 128000 | Converts | ◎ images, q4 11.3 tok/s, 2.00 GiB |
| LFM3-Audio-2B | `lfm3` + `lfm3-audio-encoder` + `lfm3-depthformer` / `lfm3-codec` | Same | LFM3 | Audio in (30 s), audio out (`hwnpu-tts`) | ◎ lut8 | Converts | ○ audio in only (41 s), no output. q4 2.03 GiB |
| LFM3-Omni-2B | Both towers above | Same | LFM3 | Images and audio | ◎ lut8 | Converts | ◎ q4 2.39 GiB |
| d1-2B / d1-Omni-2B | `lfm3` in d1 mode | Same | System One API (`hwnpu-systemone`) | Images (d1-2B), images and audio (d1-Omni) | ◎ lut8 | Converts | — |
| gemma-4-E2B-it | `gemma4` | byte-fallback BPE (`tokenizer.json`) | Gemma 4 | Text only on Apple / Intel | ◎ lut6, context 131072 | Converts | ◎ q4 7.9 tok/s, 1.78 GB (bf16 6.1 tok/s), images and audio too |
| gemma-4-E4B-it | `gemma4` | Same | Gemma 4 | Text only | ○ lut6, one program (`-prefill auto`) | Converts | ○ q4 5.6 tok/s, 3.53 GB, context 4096 |
| Qwen3.5-4B | `qwen35` | byte-level BPE + NFC + Qwen's split rules | Qwen3.5 | Text only (the vision tower is not converted) | △ checked on the CPU only, not yet on the ANE | Converts | — |
| **plamo-3-translate-2b** (new) | `plamo3` | PLaMo (`tokenizer.jsonl`) | PLaMo 3 (PFML3, translation) | — | ○ lut8, one program, context 262144. Checked on an M3 that all of it is on the ANE, and the weight formats | Converts (int8). Translates on OpenVINO's CPU | ◎ q4 11.6 tok/s, 2.16 GiB, context 8192 |
| plamo-embedding-1b | `plamo-bi` | SentencePiece BPE (`tokenizer.model`) | — (embedding API) | — | ◎ lut8 + head-scale | Converts (int8) | — |
| BERT family (all-MiniLM-L6-v2 and so on) | `bert` | **Not implemented** (WordPiece) | — | — | Converts only (lut8) | Converts only (int8) | — |
| **Anima-Turbo** (new, image generation) | `internal/imagegen/anima` (built from one MIL description) | Qwen3 BPE + T5 Unigram | — (`hwnpu-image`) | Images out | △ Converts (fp16). Checked with the CPU interpreter, not on hardware | △ Converts. Checked on OpenVINO's CPU | ◎ 1024×1024, 8 steps in about 70 s, 5.0 GiB pinned |
| Qwen3.6-35B-A3B (reference) | Not ported | — | — | — | — | — | Only MoE layers measured, with random weights (16.75) |

### 10.2 Tokenizers (`internal/tokenizer`, depending only on `golang.org/x/text`)

| Kind | Models | Implementation | Checked |
|---|---|---|---|
| byte-level BPE (`Load`) | LFM3 family, Qwen3.5 | Added tokens are cut out first, then the GPT-4 style pre-tokenizer pattern is applied by a hand-written scanner (it uses the lookahead `\s+(?!\S)`, which Go's regexp lacks). Merges by rank | LFM3: matches HF on 24 tokens (16.13) |
| Qwen3.5's extensions | Qwen3.5 | NFC normalization (adds `x/text/unicode/norm`), combining marks counted as letters with `[\p{L}\p{M}]+`, digits one at a time | Matches HF tokenizers 0.23 on Japanese, Hindi, combining marks, full-width and special tokens; 15 split cases match (16.62) |
| byte-fallback BPE (`fallback.go`) | Gemma 4 | Spaces become ▁, text between added tokens merged by rank as one piece, characters outside the vocabulary become UTF-8 bytes (`<0xE3>`). No BOS added (the chat format writes `<bos>`) | Matches HF on 35 texts + 300 random ones (7,026 tokens) (16.25) |
| SentencePiece BPE (`LoadSentencePiece`) | PLaMo-Embedding | Reads `tokenizer.model`'s protobuf by hand without dependencies. Merges by piece scores, not a merge list; user-defined pieces (runs of ▁, tabs, newlines) are matched longest-first and not merged; a ▁ is added after special tokens too (as HF's slow tokenizer does) | 0 mismatches over 32 texts (599 tokens) + 3,032 random texts (88,072 tokens) (16.19) |
| PLaMo (`LoadPlamo`; `Load` picks it for a `.jsonl`) (new) | PLaMo 3 Translate | From `tokenizer.jsonl`'s scored pieces, finds the split of least total cost, searching from the end. The cost is the score × 10⁴, rounded as Python's round does, with the sign flipped. HF uses Aho-Corasick; hwnpu looks pieces up in a map of every piece's prefixes. On a tie the longer piece wins. Characters no piece covers become UTF-8 bytes. Cuts at special tokens, at 4 or more of the same character in a row, and at 2 or more spaces | Matches HF on 40 cases + 3,040 random ones. Flipping the tie rule fails the tests (16.70) |
| Qwen3's pre-tokenizer (new) | Anima's text (Qwen3-0.6B) | GPT-4 style with digits split one by one (Qwen2 / Qwen3's rule) | Matches transformers on 11 texts (16.68) |
| T5's Unigram (`LoadUnigram`) (new) | Anima's LLM adapter | Normalizes with sentencepiece's precompiled charsmap and splits with Viterbi. An unknown character costs "the lowest score − 10" (as HF). No byte fallback | Matches transformers on 11 texts (full-width, accents, Japanese, emoji) (16.68) |
| WordPiece | BERT and others | Not implemented. `Load` returns an error | — |

On AMD (IRON), the tokenizer and chat format read the `tokenizer.json` in the export directory with the same Go code. Embedding lookups used to be device DMA. Now the 4-bit exports and Gemma 4's PLE are host tables (`embed.bin`, `ple.bin`), and only the tokens' rows are written. PLaMo 3 has no `tokenizer.json`, so the IRON side does not read a tokenizer (Go's `LoadPlamo` is used).

### 10.3 Chat formats (`internal/chat`)

Each one ports `chat_template.jinja` to Go and is checked to match, character for character, what jinja2 / transformers render.

| Format | Checked | Tools | Thinking |
|---|---|---|---|
| LFM3 (`chat.Render`) | 6 conversations (`testdata/template.json`) | `<tools>` and JSON at the end of the system message (key order kept); calls as `<tool_call><function=name><parameter=key>value</parameter>` XML | `<think>…</think>`. Thinking in earlier replies is dropped |
| Gemma 4 (`chat.Gemma4`) | 9 conversations (`gemma4_template.json`) | Its own notation (strings as `<\|"\|>`, keys by dictsort) | `<\|think\|>` and `<\|channel>thought … <channel\|>` |
| Qwen3.5 (`chat.Qwen35`) | 9 conversations | Qwen's declarations with descriptions. Objects and arrays as JSON arguments, others as Python's `str()` | Kept only in replies after the last question |
| System One (`internal/systemone`) | 7 cases (llama.cpp's template rendered with jinja2) | d1's choice / score / noul questions | `<think></think>` |
| PLaMo 3 / PFML3 (`chat.Plamo3`) (new) | 8 cases (`tools/ref/plamo3_template_golden.py`) | No tools. `target_lang`, `source_lang`, `form`, `context`, `glossary` (`Options.Translation`; in the API, `chat_template_kwargs` as in vLLM) | None. A system message becomes `context`. Without a target language, a text with Japanese goes to English and others to Japanese. The label of each past translation is chosen from its source text, so the start of the prompt can be reused |

In every format, the think budget (`-think-budget`) feeds in the closing mark the format gives (`</think>`, `<channel|>`) to move on to the answer.

### 10.4 Notes for each model

**LFM3-1.7B** (16.8, 16.13–17, 16.39–41, 16.59, 16.61, 16.66)
- Structure: short convolutions (split into `taps` to avoid the error of the ANE's conv), 5 sliding layers with a window of 256 (rings), 6 full-attention layers, per-head gates, sandwich norms. `-post-scale` keeps the ANE's arithmetic error to 1.4–2.7% a layer
- Checked against an independent fp64 reference (`reference_test.go`, 16.27) and with `hwnpu-check`. KL 0.232 with lut6
- Default: lut6 + grouped attention, split into 7 parts at the attention, context 262144, two-level attention past 48K
- Intel: int8 by default (with int4 the next token matches fp16 only 22/52)
- AMD: IRON has no LFM3, so hwnpu wrote it from IRON's building blocks (`tools/amd/iron_lfm3.py`). KL 9e-7 against HF. The convolution state is copied at each chunk boundary, for rolling back to marks and for the prompt store
- 4 bits on AMD (16.77):
  - Layout: the MLP and output matrices are q4nx, the attention (gates included) and short-convolution projections are q8, and the embedding is a host table
  - Which parts get how many bits was compared on the CPU. Most of the error is the MLP at 4 bits (KL 0.031); attention and convolutions are nearly lossless at 8 bits
  - KL on the NPU is 0.036, the same as the CPU estimate. The README's export now uses `--q4`
  - On hardware it gave the same answers as bf16 to the same questions. hwnpu-agent solved its task and passed `go test`

**LFM3-VL / Audio / Omni, d1-2B / d1-Omni-2B** (16.27–31, 16.42, 16.63–65, 16.67)
- The text part has LFM3's structure; only the tensor name prefixes (`model.language_model.`, `lfm.model.`) differ. With lut4, TTS, image descriptions and d1's judgments broke down, so the text part is lut8
- Vision (`lfm3vl`): a patchifier, a 3×3 depthwise convolution as 9 shifted multiply-adds, 2D RoPE. Preprocessing is llama.cpp's smart_resize, with resizing that matches PIL within ±1. The ANE needs fixed shapes, so there is a program per image size
- Audio in (`lfm3audio`): log-mel (within 1e-4 of librosa), WAV reading and resampling to 16 kHz, packed with a mask into a program of up to 3,000 frames (30 s)
- Audio out (`lfm3gen`, `hwnpu-tts`, Apple only): the depthformer's MaskGIT, the codec decoder run in overlapping chunks, ISTFT on the host. About 0.5× real time with lut8
- d1 (`hwnpu-systemone`): one bidirectional pass over the full-attention layers only, answering by the probabilities of the choice labels. A port of llama.cpp's `/v1/systemone`
- Images in a chat get negative token IDs from the SHA-256 of their data URL, so the same image keeps the prompt cache working (the last 16 are kept)
- AMD: hwnpu's vision and audio towers on IRON's `Multimodal` (vision 512×512, audio up to 4,096 frames. IRON's MHA needs a head_dim that is a multiple of 64, so 96 is zero-padded to 128)

**Gemma 4 E2B / E4B** (16.25–26, 16.42–43, 16.50, 16.55, 16.63, 16.65)
- Structure: per-layer embeddings (PLE; the 4.7 GB table stays on the host and rows are looked up), KV sharing (on E2B, layers 15–34 read the K/V of layers 13 and 14), proportional RoPE, sandwich norms, a logit softcap
- For the ANE: norm inputs scaled by 1/64 against fp16 overflow (`-f16` KL 3.17 → 0.0004), the softcap folded to (16, 16384), `SDPAGrouped`, lut6 (lut4 has KL 2.54 and cannot answer factual questions)
- Startup: unsplit, the lut6 parts do not stay in the ANE's cache, about 100 s every time. Split into 8 parts at the attention, 1.9 s. E4B's two programs (6.5 GiB) exceed the cache, so it runs on the generation program alone
- Text only on Apple / Intel (the audio and image towers are not converted)
- AMD: IRON's existing application is exported and run from Go. E2B also runs images and audio with IRON's `Gemma4Multimodal` (the vision tower's tokens for all 1,600 possible sizes were tabled at export)
- Pinned memory on AMD (16.72–73):
  - The PLE table (4.4 GB for E2B) moved to the host: 9.08 → 4.60 GB. Output word for word the same
  - With `--q4` (MLP and output q4nx, attention q8, embedding on the host): 1.78 GB. Generation 6.1 → 7.9 tok/s, KL 0.097
  - E4B: 14.7 → 3.53 GB, 3.7 → 5.6 tok/s, KL 0.049
- E4B's old export hit `ERT_CMD_STATE_TIMEOUT` on a 2483-token prompt. The new export (62 commands) gets through

**Qwen3.5-4B** (16.62)
- Structure: 24 Gated DeltaNet layers + 8 full-attention layers. The GDN state (128×128 per value head) cannot be written per position, so a new kind of state `st_` was added (replaced whole by the new state the program outputs). `step_valid` keeps padding tokens from changing the state. When speculative decoding accepts only part of a draft, it reruns with the accepted tokens only
- Zero-centered RMSNorm (`1 + w`), an output gate, RoPE on the first 64 dimensions only. Sharded safetensors (`model.safetensors.index.json`) can now be read
- Checked: 3.7e-7 against an fp64 reference. Only 2 layers of the real weights were fetched (a slow line): argmax 32/32 (fp16), KL 0.016 with lut6
- Split into 9 parts at the attention (2.95 GiB with lut6), context 262144 (KV 32 KiB/token)
- **Placement and speed on the ANE not yet checked**. Two programs at 5.9 GiB may exceed the ANE's cache

**PLaMo 3 Translate 2B** (16.70–71, 16.76)

Structure:
- Layers use sandwich norms. The RMSNorm offset differs by norm (1, 1/5, 1/5^1.5)
- One layer in 8 (layers 7, 15, 23) is full attention with θ = 10⁶ YaRN (factor 256). The others use a sliding window of 2048
- The fused `qkv_proj` and `gate_up_proj` are split by row ranges

Apple / Intel (`internal/llm/plamo3`):
- The shape is Gemma 4 without per-layer embeddings, KV sharing or the softcap. Norm inputs are scaled by 1/64
- YaRN's concentration is not applied to cos and sin; its square scales the scores of the full-attention layers instead. Cached keys then carry no concentration, so the host's attention only applies a scale
- Split at the 3 full-attention layers, context 262144 (7 KiB/token)
- Checked: relative error 6.5e-5 against an fp64 reference. `hwnpu-check` argmax 31/31 (fp32)
- Weight formats (M3): with lut8 the next token matches fp16 98 % of the time, with lut6 89 %; lut4 is unusable (KL 1.91)
- Default: one lut8 program (`-prefill auto`)

AMD (`tools/amd/iron_plamo3.py`; written from IRON's building blocks alone):
- Checked: top-1 17/17 against HF fp32, 9/9 also over 2596 tokens
- bf16: 7.7 tok/s, 1024-token chunks in 2.2–10.5 s
- `--q4` (16.76): 2.16 GiB, 11.6 tok/s. KL on the translations 0.009 or below
- With random weights the NPU and `OnHost` disagree (as with Gemma 4), so checks use the real weights

hwnpu-chat:
- Translates each text typed and keeps the earlier pairs in the prompt as history
- `-to`, `-from`, `-form` and `/to LANG` set the languages and style. The server also takes a `glossary`
- A terminal input line was added: history, line editing, two columns for wide characters, multi-line paste, colors

**Anima-Turbo (image generation, `hwnpu-image`)** (16.68–69, 16.72)

Parts:
- Text: Qwen3-0.6B and an LLM adapter (T5 tokens attend to themselves and to Qwen3's hidden states)
- DiT: Cosmos-Predict2 2B (28 blocks)
- VAE: the decoder of Qwen-Image's VAE
- 8 steps, CFG 1
- The host's work: tokenization, fetching embedding rows, rearranging patches, the Euler steps, PNG

AMD (`tools/amd/iron_anima.py`):
- About 70 s at 1024×1024 (7.2 s a DiT step, 60% of it self-attention)
- Against diffusers in fp32: latent cosine 0.966 (bf16 on the CPU: 0.980). The VAE alone: PSNR 44.5 dB
- Loaded one stage at a time, at most 5.0 GiB pinned

Apple / Intel (`hwnpu-image -convert`):
- One MIL description (`internal/imagegen/anima/mil`) is written as a Core ML package or OpenVINO IR, in Go only
- fp16 handling: DiT residuals kept at 1/32, norms divided by each row's maximum first
- Checked on the CPU: latent cosine 0.99985, PSNR 37–38 dB

**Qwen3.6-35B-A3B (reference, MoE building blocks only)** (16.75)
- Before bringing the model over, MoE layers that today's IRON can express were measured at its sizes (random weights)
- Reading 4-bit experts in place and folding a layer into 4 ops: 1.27 ms a layer
- Estimated 100–120 ms a token (8–10 tok/s)
- There is no IRON form of Gated DeltaNet yet

**PLaMo-Embedding-1B** (16.18–19, 16.23, 16.34)
- Structure: PLaMo's decoder with bidirectional attention. The fused `qkv_proj` and `gate_up_proj` are split by row ranges (8192 at a time, because of the 16384 limit)
- hwnpu-chat's embedding API (`/v1/embeddings`): the query prefix, mean pooling over the text's tokens only, `input_type`, `truncate`, and the shortest of the programs of different lengths (128 / 1024) that fits
- The last layer's MLP exceeds fp16 (5e4), so head-scale brings it to 1/64. Cosine similarity 0.997–0.999 against HF

**BERT family** (16.11)
- The structure is ported and converts (lut8 / int8), but without a WordPiece tokenizer it cannot be used from `hwnpu-chat`

## 11. Not yet verified

Collected from what the records call "not yet checked", "not measured" or "estimated". The items are split into "implemented, waiting for a check on hardware" and "cause unknown", with how to check them where there is a way. Features not yet written (T8's activation-aware quantization, changing the split at run time and so on) are listed briefly in 11.5.

### 11.1 Apple (Neural Engine)

| Item | Unknown | How to check / what is needed | Section |
|---|---|---|---|
| Qwen3.5-4B as a whole | Placement on the ANE, speed, startup. Two programs at 5.9 GiB may exceed the compile cache (about 4 GiB) | `hwnpu-chat -use Qwen3.5-4B` on a Mac. If every start is slow, one program (`-prefill off`) or lut4 | 16.62 |
| Qwen3.5's GDN state | Whether keeping it in fp16 accumulates error over long chats (PyTorch uses fp32) | Compare logits with an fp32 reference over a long chat | 16.62 |
| Qwen3.5's Q=64 program | Compile time and placement of the update unrolled 64 times (about 18,000 ops) | If it does not fit, add the chunked form (WY representation) through `Params` | 16.62 |
| Checking all 32 layers of Qwen3.5 | Only 2 layers were checked with the real weights. Does fp16 overflow in deeper layers? | `ref_logits.py` → `hwnpu-check` on a machine with a good line | 16.62 |
| Qwen3.5 with two-level attention | Do answers hold with two-level attention (`-sparse`)? | Passkey test | 16.62 |
| LFM3's two-level attention | Speed on an M3 (estimated 165 → about 90 ms/token at 128K), placement of the mass program, host time | The steps in [handoff-m3-sparse.md](handoff-m3-sparse.md) | 16.61 |
| Two-level attention's `Refresh` | Accuracy when reselecting only every few steps | Compare with the default (every step) on real weights | 16.61 |
| Long contexts for VL / Audio / Omni / E4B | Speed with two-level attention (16.65's table is from before it) | The same test as 16.65 | 16.65 |
| Slowing down past 128K | Why the slope per 1K grows from 1.14 to 1.43 ms (memory pressure, or attention lagging when interleaved?) | | 16.59 |
| Slowness right after a long prefill | Why the first few calls after reading 256K take 1.29 s each (350 ms after) | | 16.59 |
| The ANE's memory | Footprint of aned and ANECompilerService (not in RSS) | Needs sudo | 16.39, model-fit.md T10 |
| The ANE's compile cache | Its size (about 4 GiB is an estimate) and location, and why only unsplit Gemma 4 at lut6 does not stay | Looking into aned (sudo) | 16.41, 16.43 |
| Limits on M4 and later | The default per-part limit (0.9 GiB now; an M4 ran with 1.8 GiB and more), memory by number of parts | Measurements from M4 users | 16.12 |
| Gemma 4 E4B against the unsplit form | Skipped for lack of disk | | 16.65 |
| Gemma 4 and the prompt store | Save and restore time and size with Gemma 4 E2B | Replay the same OpenCode request as 16.38 | 16.38 |
| hwnpu-agent on Apple with Gemma 4 / Qwen3.5 | The templates support tools, but the agent tasks were not run (Gemma 4 E4B passed on AMD) | The tasks in CHAT.md section 7 | 16.35 |
| The rest of speculative decoding | Acceptance at Q=4, acceptance in OpenCode's round trips (7.4K, long XML), comparison with temperature 0 | Replays with `-dump` | 16.36 |
| d1-Omni-2B differing from its README | The team answer is the opposite of the README's. HF's code reads up to 16 questions together, hwnpu one at a time. Not compared with HF | Compare with HF's code where torch is available | 16.31 |
| LFM3-Audio's TTS | Does it sound the same as the reference (llama.cpp)? Only checked by transcribing it again with ASR | Compare audio decoded from the same codes with llama.cpp's server TTS | 16.30–31 |
| Long texts with PLaMo-Embedding | Do the last layer's values grow further in the 1024-token program (is head-scale enough)? | Vary the texts in `TestANEPrecision` | 16.34 |
| BERT family | `hwnpu-check` on real weights and placement on the ANE (only checked on the CPU for Intel) | | 16.11 |
| **Anima (image generation) as a whole** (new) | Does all of it go on the ANE (a single part on the CPU stops it)? Compile time, speed, memory, and whether the 1024×1024 VAE (one head of 384 dims, 16384 pixels) fits | Section 3 of [handoff-anima-mac-intel.md](handoff-anima-mac-intel.md) | 16.69 |
| Details of PLaMo 3's accuracy and speed (new) | Placement and weight formats are checked. Translation quality and time for long documents are not measured in detail | Compare KL with fp16 on translation pairs | 16.71 |
| PLaMo 3's two lut6 programs (new) | Do they (about 3.6 GiB) fit the compile cache (about 4 GiB) and keep startup fast? | Start twice with `-format lut6` | 16.71 |
| Why macOS memory seemed to shrink with use (new) | The guess is that macOS compressed the rings, which are mostly zeros | Measure the footprint on an M3 after the mark fix (`ringRows`) | 16.71 |

### 11.2 Intel (NPU)

**Nothing has been run on a real NPU.** Only accuracy on OpenVINO's CPU has been checked.

| Item | Unknown | Section |
|---|---|---|
| Does the compiler accept it? | Whether the driver's compiler accepts LFM3-1.7B (int8, `-seq 8` and `64`) and the split form (attention graphs over chunks of 8192), and how long it compiles | intel-npu-models.md section 11 |
| Argument names | Whether the names Level Zero returns map to the IR's names under `argNames`'s rules (if not, it stops with `chain: part 0 has no output y`) | Same |
| Speed of the weight formats | The speed difference between int8 and int4-g128 (if int4-g128 is clearly faster, it is a candidate default). Measure with `hwnpu-formats` | Same |
| Size limits | Is there a limit like Apple's 1 GiB? The 2 GiB of `-target intel` is a placeholder | 16.12, intel-npu-models.md P1 |
| Cost of attention per chunk | The cost of re-recording command lists when each step calls "full-attention layers × chunks in use" times | intel-npu-models.md, the 16.63 section |
| Memory | Whether there is enough with tiers, where the 4 graphs keep separate weights (6.4 GiB in int8) | intel-npu-models.md section 11 |
| Real checkpoints | Running Gemma 4 and LFM3-VL / Audio in hwnpu-chat (conversion and numbers checked on the CPU) | Same |
| PLaMo 3 (new) | Whether the int8 IR (two programs of 4 parts each) runs on a real NPU, and its speed. On OpenVINO's CPU it gave the same translations as AMD | 16.71 |
| Anima (new) | Whether all of it runs on a real NPU, speed, memory. Checked on OpenVINO's CPU (f32). With f16 inference, PSNR drops to 25.5 dB | 16.69, handoff-anima-mac-intel.md section 4 |

### 11.3 AMD (XDNA)

| Item | Unknown | Section |
|---|---|---|
| NPU1 (Phoenix, Hawk Point) | Whether one xclbin per step works on real hardware (checked only on a simulated 4-column NPU, and with the same export shape on NPU2). 6 xclbins use up the limit of 6 hardware contexts | 16.57 |
| Windows | xrtshim on real hardware and building with MSVC (checked only through XRT on Linux, and compiling with zig) | 16.56 |
| Checking Gemma 4 E2B | Not compared with HF's logits on the real weights (LFM3 and E4B were). With random weights it disagrees with IRON's CPU reference (cosine 0.55–0.83); the reason is not known | 16.50 |
| Power of the NPU alone | Upstream Linux 7.0 has no sensor. Current numbers are for the whole package (PPT) | 16.44, 16.53 |
| `CREATE_HWCTX` right after another process | Can fail with `EINVAL` for a moment. No retry yet | 16.48 |
| Long contexts | IRON exports have a fixed context (4096 / 12288). Attention over chunks is IRON's side's job | 16.63 |
| Speed of PLaMo 3 at 4 bits (new) | Measured with another job loading the machine (a single GEMV 1.6× slower). Measure again on a quiet machine | 16.76 |
| Other sizes of PLaMo 3 (new) | 0.8B, 8B and 31B share the structure and should export, but have not been tried | 16.70 |
| 4-bit KL against HF (new) | The q4 KL for Gemma 4 and PLaMo 3 is against bf16 on the NPU. Only LFM3 was compared with HF fp32 (0.036) | 16.73, 16.76 |
| MoE (Qwen3.6-35B-A3B) (new) | Loading and running all 40 layers (20 GB of tables) at q4 (tried up to 12 layers; 35 GB pinned checked with bf16). KL with real weights (with random weights, stacking layers gives cosine 0.997 at 12 layers). IRON forms of Gated DeltaNet and attention, MoE for prompts. 8–10 tok/s is an estimate | 16.75 |
| MoE warning about reusing BDs (new) | aiecc warns about reusing a BD after a `dma_free_task` with no completion guarantee. Thought harmless, but not proven statically | 16.75 |

### 11.4 Behaviour with unknown causes

| Behaviour | Current guess | Section |
|---|---|---|
| Only unsplit Gemma 4 at lut6 / lut8 does not stay in the ANE's cache | A policy on the ANE's side (aned). The split form stays, so it no longer matters | 16.43 |
| Which shapes fall to the CPU at the 16384 per-dimension limit and which stay (chunks along an axis, weighting by ratios go to the CPU) | Inside Core ML's compiler | 16.21 |
| Only the 8-part split loads slowly from the cache (23.7 s) | Not looked into | 16.13 |
| On a 128K English document, two-level attention has lower loss than normal attention | Normal attention gets distracted by distant tokens, and two levels narrow that | 16.61 |
| AMD's prefill is slower than IRON's Llama 3.2 1B | Without rings for the sliding layers, the whole context is read with a mask (skipping outside the window should be several times faster) | 16.50, 16.66 |
| AMD: splitting Anima's DiT into commands with host steps stops the first command with `ERT_CMD_STATE_TIMEOUT` (new) | Not looked into. Avoided with one graph of 4 blocks | 16.68 |
| AMD: with Gemma 4 E4B's old export, the first command of the second chunk stops for 61 s (new) | Not looked into. The new export (62 commands) gets through | amd-npu-models.md section 12 |
| AMD: with random weights the NPU and IRON's reference disagree (PLaMo 3 too, besides Gemma 4) (new) | They agree with the real weights. Random weights may make outliers through bf16 rounding, but this is not checked | 16.50, 16.70 |
| AMD: MoE layers briefly slowed to 170–520 ms each (new) | Suspected another session using the NPU at the time. The 8 runs after were around 5 ms | 16.75 |
| AMD: a single op differs by 1.4× between the first and second run (new) | Thought to be the power state | 16.75 |

### 11.5 Not yet written (for reference)

- Fitting tables with activations in view (weighted k-means, AWQ-style scales) to make lut4 / lut6 more accurate (model-fit.md T8)
- Writing the embedding tables at conversion so the checkpoint (LFM3 3.5 GB, Gemma 4 10.2 GB) is not opened at run time (T5)
- Conversations set aside moved out of the process (to disk), halved with int8 (the rest of T4)
- Changing the number of parts at run time (`MemoryBudget`, `Reshape`) (16.12)
- A WordPiece tokenizer (to use the BERT family from hwnpu-chat)
- Speculative decoding with a draft model (such as Qwen3.5's `mtp.` layers) (T9)
- Repairing hwnpu-agent's broken calls (output-parser)
- AMD: attention that skips outside the sliding window, an IRON version for checking drafts
- The rest of AMD's pinned memory (amd-npu-models.md section 12):
  - Load the towers only when used (about 1 GB for E2B with images and audio)
  - One DiT work area for all 7 sets (about 1 GB)
  - Split the VAE into tiles
- A 6-bit format on AMD. It can follow q8's layout
- The rest of Anima:
  - CFG (negative prompts). A non-Turbo Anima is expected to take over 10 minutes an image
  - Calls from ComfyUI nodes
  - A 512×512 export
- Porting Qwen3.6-35B-A3B: an IRON form of Gated DeltaNet, and choosing the route on the device
