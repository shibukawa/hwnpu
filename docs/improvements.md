# What each improvement bought (as of 2026-11-08, main up to #83, with AMD's checks and changes: Hugging Face comparisons, PLaMo 3 0.8B and Qwen3.5-0.8B / 2B, 6-bit weights, Anima's DiT in one bank, the prompt's attention on 8 pipelines, and checking drafts in one call)

[日本語](improvements.ja.md)

This page sorts the changes by what they improved. It draws on the implementation log in design.md chapter 16 (up to 16.117), model-fit.md, intel-npu-models.md, amd-npu-models.md (sections 11–13), amd-xdna2-bench.md, handoff-anima-mac-intel.md and handoff-m3-gptq.md. The numbers are the measurements recorded there. Unless noted, they are from **the Neural Engine (ANE) of an Apple M3 (16 GB)** with LFM3-1.7B. AMD numbers are from a Ryzen AI 9 365 (Strix Point). Numbers in parentheses are design.md sections.

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
| AMD: checking drafts in one call (`--verify`, speculative decoding with prompt-lookup drafts) (16.117) | ◎ Qwen3.5-0.8B 82 → 72 ms/token (code), 82 → 62 (Japanese) | | | △ +30 MB of control code in the device heap, the GDN states after each row (0.8B 2 × 20 MB) | | | unchanged (logits bit for bit the decode step's: the text is the same at temperature 0) | AMD (needs exporting again; Qwen3.5 only for now) |
| Speculative decoding: drafts from Qwen3.5's MTP layer, and the GDN state brought back on the host (`state-rows`) (16.109) | ◎ 0.8B (M1) 26.1 → 43.8 tok/s (code), 25.8 → 30.3 (Japanese) | | △ one more program (0.27 GiB) | △ +0.27 GiB on the ANE | | | unchanged (the distribution; fp16 rounding may change the text partway) | Apple |
| KV tiers `-tier` (16.20, 16.37) | ◎ 1.95× (short chats) | ◎ over 2× | ✕ worse with lut6 | △ +0.5–0.7 GiB | | | | Apple (now lut4 only) |
| Caching the placement result (16.20) | | | ○ 42 → 16 s | | | | | Apple |
| Prompt store (disk) (16.38, 16.52) | | ◎ 7.9K tok 14.6 s → 0.2 s | | | △ about 145 MB a sequence | | | All (AMD too) |
| hwnpu-agent (short system prompt) (16.35) | | ◎ 7.5K → 629 tok | | | | | | All |
| Split at the attention (`-paged`) (16.59, 16.63) | ◎ 1.3–1.7× | | ◎ Gemma 98 s → 1.9 s | ◎ only the length used | | ◎ 262144 by default | unchanged | Apple / Intel |
| Two-level attention (`-sparse`) (16.61, 16.65) | ◎ a call −52% at 128K, −58% at 250K (LFM3) | ○ −24% at 128K, −44% at 250K (+11% at 64K) | | △ +1/8 and the copies (+0.8 GB at 250K) | | | △ approximate (passkeys kept) | Apple / Intel |
| The Neural Engine reads prompts 128 tokens per call, logits of the last row alone (16.82) | unchanged | ○ at 16K LFM3 about −20%, E2B −25%, VL −22% | △ converted again once | | | | unchanged | Apple (models split at their attention) |
| Neural Engine: Gemma 4's KV-sharing layers on the last row alone (16.83) | unchanged | ◎ E2B 16K 27.8 → 12.6–13.2 s, 64K 205 → 94.6 s (2.1–2.2 times) | | ○ footprint 1805 → 1392 MB (64K) | | | unchanged (matches the program without a cache on the CPU) | Apple |
| Neural Engine: Gemma 4's prompt program with only the layers that write the caches (`-kv-only`) (16.112) | ○ +13–17% on the M1 (no more clash with the second program) | ◎ E2B on the M1, 4249 tokens 4.2–4.6 → 2.4–2.7 s | ○ second program 2.6–3.5 → 0.4–0.5 s | ◎ weights 3.20 → 1.98 GiB, ANE 1.96 → 1.73 GiB | | | unchanged (the same answers to the letter) | Apple |
| Neural Engine: Gemma 4's soft cap on the host (`-logits host`) (16.113) | ○ −2.3 ms a step on the M1 (4%) | | | | | | unchanged (the same answers to the letter) | Apple |
| Neural Engine: silu written with exp and real_div (`blocks.SiLU`, 16.95) | △ +3% | △ +9% | △ converted again once | | | | ◎ lut8 KL 1/2.5 (LFM3 0.058 → 0.022), PLaMo 3 −20–40%; lut6 under a tenth. ◎ Anima's DiT: latent cosine with diffusers' fp32 0.55–0.85 → 0.67–0.98, no slower | Apple (Intel unchanged) |
| Repetition detection (16.60) | ◎ felt (stops going round in circles) | | | | | | | All |
| E4B as one program (`-prefill auto`) (16.65, 16.83) | ○ +25% | ✕ 7.5–8× slower when long (than `-prefill on`'s two) | ◎ 77–87 s → 4 s | ◎ half the weights | | | | Apple |
| Identical files shared on the disk, cleanup (16.64, 16.79) | | | △ Apple: slow once right after the switch | | ◎ M3 about 50 → 20 GiB, two AMD exports 15.2 → 8.2 GiB | | | macOS / Linux (Btrfs, XFS) / Windows (ReFS, unverified) |
| AMD: full ELF (16.45–50) | ◎ (avoids a 0.46 ms switch per op × 100) | | ○ | | | | | AMD |
| AMD: prompt chunk 2048 → 1024 (16.54) | | ○ −20% | | | | | | AMD |
| AMD: 4-bit / 8-bit weights `--q4` (16.73, 16.76–77) | ◎ 1.3–1.6× | unchanged | | ◎ pinned 1/2–1/5 | ○ weights about half, exports 0.6–0.8× | | △ KL 0.009–0.097 | AMD |
| AMD: PLE and embedding tables on the host (16.72–73) | unchanged | unchanged | | ◎ E2B 9.08 → 4.60 GB | | | unchanged (word for word) | AMD |
| AMD: release the NPU while idle `-idle` (16.74) | | | △ reload 0.8–2 s on the next request | ◎ 0 while idle | | | | AMD |
| AMD: Gemma 4's KV-shared layers on the prompt's last row only (16.80) | unchanged | ◎ E2B 2.3× (4K 18.3 → 7.8 s), E4B 1.7× | | | | | ○ caches bit for bit the same (last row's logits cosine 0.9996 or more) | AMD |
| AMD: sliding-window layers read only their window (16.84) | △ −2 % at 4K (PLaMo 3 −14 % at 8K) | ○ E2B −12 % over 4K on top of 16.80 (last chunk −20 %), PLaMo 3 −12 % over 8K | | △ +12 MB (a chunk's rows more in each window cache) | | | unchanged (top token the same, cosine 0.9997 or more) | AMD |
| AMD: towers loaded only when used (16.81) | | | △ the first image or clip +0.6–0.7 s | ◎ E2B with images and audio 2818 → 1826 MiB while only text | ○ no text-only export needed | | unchanged | AMD |
| AMD: the prompt's attention on 8 pipelines (`--mha-pipelines`, 16.107) | unchanged | ◎ chunks 2.3–4× faster on E2B (from 0 1.16 → 0.50 s, from 3072 2.53 → 0.64 s), LFM3 2.1–4.8×, PLaMo 3 2B 2.7–4.8×, 0.8B 2.4–4.8×. Energy −66 to −83 % | | | | | unchanged (logits bit for bit the same) | AMD (needs exporting again) |
| AMD: Qwen3.6's prompt experts' gate and up on four cores a column, rows an expert 128 → 256 (16.116) | unchanged | ◎ chunk of 256 3.9 → 3.0 s (always one pass of the experts' weights), 2.7 s with the host's combine on all CPUs, 2.6 s with the experts' down on four cores a column | | | | | unchanged (logits bit for bit the same) | AMD (needs exporting again) |
| AMD: Gated DeltaNet's columns aligned to 64 (`COLUMN_ALIGN`, 16.117) | ○ Qwen3.5-0.8B decode 85.6 → 82.5 ms (the fused projection on 8 columns, not 1) | | | △ zero rows (0.8B +7 % of that projection) | | | unchanged (logits bit for bit the same) | AMD (needs exporting again) |
| AMD: the prompt GDN's silu once before the operator (`GdnSiLU`, 16.111) | unchanged | ◎ Qwen3.6 chunk of 256 4.5 → 3.9 s (GDN layer 43 → 22 ms), Qwen3.5-0.8B chunk of 512 0.9 → 0.5 s | | | | | unchanged (logits bit for bit the same) | AMD (needs exporting again) |
| AMD: Qwen3.5-0.8B / 2B with 6-bit MLPs (q6) and the embeddings on the host (16.104–105) | unchanged | | | ◎ 0.8B 1858 → 1069 MiB, 2B 4017 → 2014 MiB | | | ◎ KL from HF on the GPL 0.34 at q4 → 0.044 (0.8B) | AMD |
| AMD: Anima's DiT in one bank (`SwapVariants`, 16.106) | △ denoising +2 % | | | ◎ pinned 3445 → 1236 MiB | | | bit for bit the same | AMD |
| Anima loaded one stage at a time (16.72) | 69.3 → 67.7 s/image | | | ◎ 14.2 → 5.0 GiB | | | bit for bit the same | AMD |
| AMD: Anima's DiT at 8 bits (q8) (16.78) | +2 % (67.3 → 68.3 s/image) | | | ◎ pinned 5020 → 3445 MiB | ○ 5.7 → 4.0 GB | | nearly unchanged (latent cosine 0.966 → 0.961) | AMD |
| AMD: the cores' rounding set to nearest even (16.78) | unchanged | unchanged | | | | | ◎ Anima's quantized DiT (step-0 error q8 0.088 → 0.031); language models unchanged | AMD |
| Apple: Anima on the Neural Engine (16.85) | ◎ runs (`dit_time`, `dit_patch` and the fp16 text had fallen to the CPU) | | | | | | | Apple |
| Apple: `hwnpu-image` with the compile cache (16.85) | | | ◎ 118 → 4.4 s (512×512), every start had compiled the DiT again | | ◎ no 4 GB of bundles a start | | | Apple |
| Apple: Anima's attention 512 queries at a time (16.85) | ○ a step 7.0 → 4.5 s (1024×1024) | | | | | | unchanged | Apple |
| Apple: Anima's VAE with `upsample_nearest_neighbor`, loaded one stage at a time (16.85) | ◎ VAE 14.5 → 3.7 s | | | ◎ 6.9 → 3.8 GB | | | stage 5 rel. error 4.8e-4 | Apple |
| Apple: Anima at int8 by default (16.85) | ○ a step 4.5 → 4.2 s | | ○ | ○ | ○ export 4.6 → 2.8 GB | | △ latent cosine 0.902 against fp16 (lut8 0.858); with the DiT's silu fixed (16.95), fp16 is only 0.006–0.025 nearer diffusers' fp32 | Apple |
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
| KV tiers (c4096 → c16320) (16.37) | 19.4 tokens/s | 37.8 tokens/s (1.95×) | lut4 only. With lut6 startup gets worse, so `-tier auto` turns it off (section 4) |
| Split at the attention (16.59) | c16320 (unsplit) 19–28 tokens/s | 31–36 tokens/s (1.3–1.7×) | The cost now follows **the length used**. 42.7 ms/token at 16K, 92.8 at 64K, 165 at 128K, 320 at 250K |
| Same (Gemma 4 E2B, 16.63) | 13.3–28.2 tokens/s | 23.6–36.0 tokens/s | Not one token of the answers differs |
| Same (LFM3-VL / Audio, 16.65) | 11.9–29.4 / 16.3–23.4 | 24.0–36.0 / 18.8–28.6 | |
| Two-level attention (LFM3, 16.61) | a call 101 ms at 64K, 221 at 128K, 345 at 250K (the whole cache, the same day) | 84, 105, 144 ms (generating at 78, 90, 126 ms/token) | As estimated (about 90 ms/token at 128K). The host's copies in `choose` take 13 ms a call (12%) |
| Two-level attention (Gemma 4 E2B, 16.65) | 82.4 ms at 64K, 119.5 at 125K | 73.9, 92.4 ms | |
| Two-level attention (LFM3-VL / Audio / Omni, 16.65) | a call 179 ms at 120K (VL, the same day) | VL 111, Audio 120, Omni 118 ms | |
| Two-level attention's `Refresh` 4 / 16 (16.61) | a call 105–111 ms at 128K (choosing every call) | 81–90 with 4, 76–83 with 16 | The three passkeys kept. The default stays every call |
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
| Two-level attention (16.61) | LFM3 at 128K 616 s, at 250K 2186 s (the whole cache, the same day) | 469 s, 1217 s | Slower at 64K, 155 → 173 s (the prompt is read in two levels from about 57K) |
| Two-level attention (16.65) | Gemma 4 E2B at 125K 709 s, LFM3-VL at 120K 583 s | 521 s, 431 s | Audio 516 s, Omni 501 s |
| Neural Engine: Gemma 4's KV-sharing layers on the last row alone (16.83) | E2B 16K 27.8 s, 64K 205.2 s | 12.6–13.2 s, 94.6 s | The shared layers (15 to 34), and the host's attention of their 4 full attention layers, on the last token's row. That attention runs on the program of 8 tokens that generates |
| The same (E4B, `-prefill on`) (16.83) | The default one program: 16K 184.3 s, 64K 1382.8 s | 24.3–25.4 s, 168.4 s (7.5–8.2×) | Two programs (6.5 GiB) do not fit the compile cache: every start takes 135–165 s, and the first tokens after a prompt are slow (3 s at 16K, 14 s at 64K). Generation at 16K is a fifth slower. The default stays one program |
| Neural Engine: Gemma 4's prompt program with only the layers that write the caches (16.112) | E2B on the M1, 4249 tokens 4.2–4.6 s (q128-last, 8 parts, 1.6 GiB) | 2.4–2.7 s (3 parts, 0.38 GiB) | Layers 0 to 14 alone read 128 tokens at a time; the last 8 tokens go through the program that generates, which gives the logits. The output layer's `reshape` on the CPU is gone from the prompt program too |
| The Neural Engine's prompt chunk of 128, logits of the last row alone (16.82) | LFM3 4K 3.9–5.4 s, 16K 19.3–25.2 s, 64K 160 s; Gemma 4 E2B 16K 31.2 s; LFM3-VL 16K 21.6 s | 3.3–3.9 s, 15.2–16.1 s, 136–146 s; 23.5 s; 16.9 s | Even at 64 rows a call is mostly reading the weights, so one of 128 costs 1.5 times as much. 256 is no faster. Models in one piece and Intel stay at 64 |
| AMD: prompt chunk 1024 (16.54) | 2,683 tokens 13.0 s | 10.2 s | Short ones too: 0.56 → 0.43 s |
| 4 bits on AMD (16.73, 16.76–77) | — | Unchanged (LFM3 2048 tokens 2.29 → 2.38 s, PLaMo 3 2596 tokens 11.2 → 11.8 s) | Weights are turned back from q4nx and q8 into bfp16 or bf16 on the device for the GEMM, so prompt speed does not gain |
| PLaMo 3's translation history (16.70–71) | Reads the whole history of pairs every time | From the second text on, reads only 17 of 50 tokens | When the target language is chosen automatically, the label of each past translation is fixed by its source text. Otherwise the start of the prompt could not be reused |
| Coming back from `-idle` (16.74) | — | 830 tokens restored from the host copy, only the next question's 58 tokens read | |
| AMD: Gemma 4's KV-shared layers on the last row only (16.80) | 4K prompt: E2B 18.3 s, 218 J (q4 the same), E4B 22.9 s, 279 J | E2B 7.8 s, 97 J (q4 8.3 s, 94 J), E4B 13.7 s, 166 J | Only a chunk's last row needs the shared layers' output (E2B 20 of 35, E4B 18 of 42). Faster by computing less, not by adding the GPU |
| AMD: the prompt's attention on 8 pipelines (16.107) | E2B q4, a chunk of 1024: from 0 1.16 s, 13.7 J; from 3072 2.53 s, 49.6 J. LFM3, a chunk of 2048: from 0 2.73 s, from 10240 11.39 s | 0.50 s, 4.7 J; 0.64 s, 8.6 J. 1.30 s, 2.39 s | 60 % of a chunk was MHA, all with `num_pipelines` 1 (the exports had no profile). Now 8, as IRON's Llama. Logits bit for bit the same. Qwen3.5 (chunks of 256) takes 4: 0.63 → 0.59 s (from 3840 1.04 → 0.69 s) |
| AMD: sliding-window layers read only their window (16.84) | 4K prompt after 16.80: E2B 7.84 s, 92 J; PLaMo 3 q4 8K 60.3 s | E2B 6.87 s, 80 J (the chunk from 3072 2.76 → 2.21 s); PLaMo 3 53.1 s (the chunk from 7168 10.75 → 8.39 s) | IRON's attention read every key from position 0 and only skipped the compute outside the window. A chunk now copies its window's keys (chunk + window) and attends over the copy; a decode step reads its window in place. The chunks before the window's end are unchanged |

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
| Sharing AMD exports (16.79) | — | The first load of an export +4 s (5.3 GiB of whole files and a 1.67 GiB bank start, from the page cache); 2–11 ms after | Nothing to compile again on AMD. Only where the file system clones |
| AMD (16.47, 16.50) | — | xclbin 0.5 ms, full ELF 1.0 ms, a 9 GB image 2–3 s from the page cache | |
| AMD reloading after `-idle` (16.74) | — | E2B (4-bit) 0.8 s, E2B with images and audio 2.0 s | The Neural Engine and Intel take tens of seconds or more to reload, so only IRON releases |
| AMD: the towers on the first image or clip (16.81) | Loaded with the model | 0.5–0.7 s once: Gemma 4 E2B 4-bit 992 MiB 0.7 s, LFM3-Omni 768 MiB 0.6 s, Gemma 4 E2B bf16 (16.80's export) 1058 MiB 0.5 s | Text alone starts with the text's part of the bank. After `-idle` the towers are left out again |
| Trimming trailing zeros from Anima's bank images (amd-npu-models.md 11.2) | VAE image 8 GB, loads in 10 s | 0.2 GB, 4 s | Startup reads only the manifest (0.5 s); each stage loads when it runs |
| PLaMo 3 as one lut8 program (`-prefill auto`) (16.71) | Two programs (4.9 GiB) exceed the ANE's compile cache and are recompiled every time | One program fits the cache | The cost: a document of thousands of tokens is read about 10× slower. `-format lut6` keeps two programs |

## 5. Memory and disk

### 5.1 Memory

| Change | Effect | Notes |
|---|---|---|
| Split by layers (separate packages) (16.13) | Peak RSS ≈ one part (a 1.81 GiB model: 1.03 GiB in 2 parts, 0.55 GiB in 4) | multifunction (several functions in one file) cannot get past the limit and falls to the CPU (5.26 GiB) |
| Rings for the sliding layers (16.16) | KV 88 → 50.5 MiB (c4096) | |
| Gemma 4's KV sharing (16.25, 16.63) | 6 KiB a token (0.75 GiB at 131072) | Sharing saves only memory. The amount of attention computed is the same |
| Split at the attention (16.59) | KV is allocated only as used (12 KiB/token). phys_footprint 416 MB at 16K → 3.2 GB at 256K | `Save` also copies only the rows used (before: the whole cache, 3 GiB each time at 256K) |
| Shrinking the context to fit memory (16.59) | If the cache is over 1/4 of physical memory, it shrinks by whole chunks | Up to 4 GiB on a 16 GB M3 |
| PLE table on the host (16.25) | Gemma 4's 4.7 GB table is kept off the ANE | |
| E4B as one program (16.65) | Resident weights 6.5 → 3.26 GiB | |
| Gemma 4's prompt program with only the layers that write the caches (16.112) | E2B's two programs' weights 3.20 → 1.98 GiB (the Neural Engine holds 1.96 → 1.73 GiB). E4B 6.51 → 4.88 GiB | E4B is still over the compile cache's mark, so it stays one program |
| KV tiers (16.37) | **Cost**: 2 programs 2.29 GiB → 4 programs 2.73–2.96 GiB (RSS), footprint 1,968 → 2,807 MB | Each tier has its own copy of the weights |
| Two-level attention (16.61) | **Cost**: +1/8 for the summaries (LFM3 12 → 14, Gemma 4 E2B 6 → 7 KiB/token) and the copies of the chunks read exactly | LFM3's footprint at 250K 3.28 → 4.09 GB |
| Conversation marks keep only the ring slots in use (`ringRows`) (16.71) | 578 → 183 MB over 5 PLaMo 3 translations (live heap +85 MB a turn → 36 MB in all) | Each mark used to copy the whole ring of the window layers (88 MB). Helps short chats with Gemma 4 and LFM3 too |
| `debug.FreeOSMemory` after a conversion (16.71) | Startup right after a conversion 2.4 GB → 154 MB | Go was slow to give back what was left of the checkpoint read during the conversion |

**Pinned memory on AMD** (16.72–74, amd-npu-models.md sections 12–13). Every page of an NPU buffer is pinned, so it shows neither in the process's RSS nor in swap. What grows is `Unevictable`. Nearly all of it is weights and work areas; the KV cache is under 5 %.

| Model | Original (bf16) | Tables on the host | Quantized (4-bit with attention q8 by default; Anima 8-bit) | While idle (`-idle`) |
|---|---|---|---|---|
| Gemma 4 E2B | 9.08 GB | 4.60 GB | **1.78 GB** | 0 |
| Gemma 4 E4B | 14.7 GB | 9.25 GB | **3.53 GB** | 0 |
| Gemma 4 E2B (images and audio) | 10.1 GB | 5.6 GB | **2.82 GB**; 1.83 GB until an image or a clip comes (16.81) | 0 |
| LFM3-1.7B | 3.75 GiB | — | **1.64 GiB** | 0 |
| LFM3-VL / Audio / Omni | 4.12 / 4.14 / 4.51 GiB | — | **2.00 / 2.03 / 2.39 GiB**; Omni 1.64 GiB until a tower runs (16.81) | 0 |
| PLaMo 3 Translate 2B | 5.96 GiB | — | **2.16 GiB** (the 0.51 GiB embedding is a file on the host) | 0 |
| PLaMo 3 Translate 0.8B | 1.77 GiB | — | **0.58 GiB** (16.103) | 0 |
| Qwen3.5-0.8B / 2B | 1.81 / 3.92 GiB | — | **1.04 / 1.97 GiB** (6-bit MLP and head, embeddings on the host, 16.104) | 0 |
| Anima (1024 × 1024) | 14.2 GiB | **5.0 GiB** loading one stage at a time | **3.4 GiB** (DiT at 8 bits; 2.7 GiB at 4 bits). **1.2 GiB** with the DiT's 7 groups read in turn into one bank (16.106) | — |

- Tables on the host: PLE reads only one row a token, yet the whole table was pinned. The table now stays a file, and before each call only the tokens' rows are written into a small device buffer (18 MB for E2B)
- The embedding is shared with the output matrix, so making the output 4-bit leaves no bf16 rows. The embedding became a host table the same way

### 5.2 Disk

Most of the disk is weights and their copies: programs for 8 and 64 tokens per call, Core ML's compiled copies, AMD exports of one model with and without images and audio or at 4 bits. Compressing does not help (section 8), so the gains come from fewer bytes (4 bits) and from sharing one copy of the same bytes.

| Change | Effect | Notes |
|---|---|---|
| Sharing identical files on macOS (APFS's clonefile) (16.64) | M3: weight files about 49.6 → 20 GiB | The Q=8 and Q=64 programs' `weight.bin` and the `.mlmodelc` Core ML compiles them to. Measured with APFS's per-file private size |
| Sharing identical files on Linux (FICLONE) (16.79) | Two Gemma 4 E2B 4-bit exports (text 7.0 GB, images and audio 8.2 GB): 15.2 → 9.9 GiB on Btrfs | `ple.bin` 4.4 GiB, `embed.bin` 0.75 GiB, `prompt.elf`, `decode.elf`, `tokenizer.json`. Btrfs, XFS (reflink, the default since xfsprogs 5.1), bcachefs and ZFS with block cloning share; ext4 and tmpfs do nothing |
| Sharing the start of files (FICLONERANGE, `ClonePrefix`) (16.79) | The same two: 9.9 → 8.2 GiB | The text export's bank (1.67 GiB) is the start of the bank with the towers. Files of one name share their common start in whole blocks |
| Same, Anima (16.79) | bf16, q4 and q8 exports: 0.09 GiB | Their `text/` and `vae/` were already symlinks. The DiT banks of q4 and q8 differ after 4 MiB |
| Sharing on Windows (ReFS, Dev Drive) (16.79) | Not measured | FSCTL_DUPLICATE_EXTENTS_TO_FILE, whole files and ranges. NTFS does nothing. Only compiled |
| Same, the potential on the AMD development machine (16.79) | 68.7 GiB of whole files the same in `~/models` (297 GB), found by hashing | Mostly test exports. Its `/` is ext4, so it stays |
| 4 bits on AMD `--q4` (16.73, 16.76–77) | Weight banks about half (3.5 → 1.6–1.7 GiB for Gemma 4 E2B and LFM3-1.7B), whole exports 0.6–0.8× (Gemma 4 E2B 9.0 → 7.0 GB, LFM3-1.7B 3.6 → 2.2 GB) | The host tables (`ple.bin`, `embed.bin`) stay bf16 |
| Anima's DiT at 8 bits (16.78) | Export 5.7 → 4.0 GB | |
| Trimming trailing zeros from Anima's bank images (amd-npu-models.md 11.2) | VAE image 8 GB → 0.2 GB | |
| Cleaning up the `/dev/shm` left by OpenVINO (16.72) | 8.4 GB a day | `/dev/shm` is memory, counted here as it fills like a disk |
| **Cost**: the prompt store (16.38, 16.52) | About 145 MB a sequence (Apple), 51 MB an entry (AMD) | |
| Towers loaded only when used (16.81) | A text-only export need not be kept beside the one with images and audio (Gemma 4 E2B 4-bit: 7.0 GB) | It costs 51 MiB more pinned than the text export. Unlike sharing (16.79), it needs no Btrfs or XFS |
| **Cost**: AMD exports are whole files per variant | Each `-q4`, `.mm`, context or chunk size is a new directory | Shared where the file system clones (above), whole files and common starts; bytes that differ in the middle of a file are not |

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
| AMD: 4-bit codes chosen by GPTQ (16.87, `q4_gptq.py`, `--q4-codes`) | Each weight rounded to its nearest level: KL from the NPU's bf16 on E2B 0.120 (README), 0.221 (Japanese Wikipedia); LFM3 0.032, 0.071 | E2B 0.092, 0.081 (top-1 79 → 88 %); LFM3 0.017, 0.038. The GPL, not calibrated on: E2B 0.161 → 0.140, LFM3 0.110 → 0.092 | Same format, so speed and memory are unchanged. Calibration on the CPU before the export (E2B about 40 min, LFM3 about 25). With H damped by 0.01 instead of 0.1 it fitted the calibration texts and made the GPL worse than rounding. Also on the NPU, wiki-ja: E4B 0.083 → 0.049, LFM3-VL 0.067 → 0.034, Audio 0.046 → 0.022, Omni 0.048 → 0.024, PLaMo 3 0.058 → 0.040 (README and GPL mostly better too; Audio's GPL 0.154 → 0.161). LFM3's multimodal models have their own text weights, so each is calibrated (amd-npu-models.md 13, "continued") |
| Apple: lut6 entries chosen by GPTQ (16.87, `hwnpu-convert -codes`) | LFM3 lut6: KL (CPU estimate) README 0.070, wiki40b-ja 0.112 | 0.027, 0.068. Through Go's converter 0.074 → 0.050 (top-1 105 → 111 of 128) | Same format and speed. On the M3's ANE (16.95, every projection with `--q8 gptq`): KL from PyTorch's fp32 against the chat's rounding English 0.322 → 0.247 (median 0.155 → 0.097), Japanese 0.054 → 0.035, as on Core ML's CPU; all on the ANE, same speed. 12 factual questions: 10 of 12 either way (GPTQ got Mount Fuji's height wrong). Fitting the tables too, by k-means weighted with E[x²] (`--lut-fit wkmeans`): lut6 0.050 → 0.038 through Go's converter (top-1 112 of 128); lut4 2.7 → 0.66 → 0.33, better but still 4× lut6's rounding. **Gemma 4 E2B on the M1** (16.114, calibrated on the 16 GB Mac with the tables read from the file, 87 min): on the ANE over 128 tokens English 1.02 → 0.56, Japanese 1.16 → 0.94 (median 0.25 → 0.08, 0.47 → 0.20; top-1 88 → 97, 82 → 92 of 128) with the `per_layer` projections the calibration leaves out at lut8 (+0.01 GiB); 12 questions 10 → 11 of 12; same speed. `hwnpu-chat -codes DIR -format-for per_layer=lut8` |
| AMD: the cores' rounding to nearest even (16.78) | The cores start in floor mode, and each narrowing to bf16 pulls sums of inputs with a mean downward: Anima's DiT step-0 velocity error q8 0.088, q4 0.244 (bf16 0.025) | q8 0.031, q4 0.087 (the quantization alone on the CPU: 0.007, 0.092). Language models: E2B KL 0.097 → 0.093, LFM3 0.036 unchanged | None (one instruction a call). Takes effect when exported again |
| AMD: Anima's DiT at q8 rather than q4 (16.78) | q4nx: latent cosine 0.777, the face and hands change. GPTQ (`anima_gptq.py`, 16.87) at 512×512 on the CPU: velocity error 0.123 → 0.071, latent cosine 0.868 → 0.939 (q8: 0.013, 0.962) | q8: 0.961 (bf16 0.966), nearly the same picture | 735 MiB more pinned than q4 |
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
| Gemma 4's KV-sharing layers on the last row alone (`-last-row`, 16.83) | Prefill. AMD's 16.80, with the host's attention on one row too | E2B 2.1–2.2 times. The attention of the one-row layers runs on the program that generates |
| Gemma 4's prompt program with only the layers that write the caches (`-kv-only`, 16.112) | Prefill, memory, generation. The shared layers write no keys and values, so reading a prompt does not need them | E2B's second program 1.6 → 0.38 GiB. On the M1 the slowdown of generation with both loaded (one part +3 ms each call) is gone too |
| Gemma 4's soft cap on the host (`-logits host`, 16.113) | Generation and power. The mul, tanh and mul after the output layer and the `reshape` on the CPU are gone: all on the Neural Engine | The cap is monotonic, so the host applies it to the top k that sampling uses and to the tokens the penalty touches. decode applies it to every row it returns to tools and tests |
| The prompt program takes 128 tokens per call, its output layer the last row alone (`-last-row`, 16.82) | Prefill. A call of 64 rows costs 1.06 times one of 8: reading the weights dominates | LFM3 at 16K about −20%, Gemma 4 E2B −25%. The row is taken by a product with a one-hot row (gather is on the CPU) |
| PLaMo 3's attention output taken token-first (`attn_ot_N`) | Avoids the one `transpose` in the last part that fell to the CPU | Everything on the ANE on an M3. The host reorders (one more copy) |
| Anima's fp16 handling (DiT residuals at 1/32, `maxScaled` norms) | No fp16 overflow and no subnormals | Checked on an M3 (16.85) |
| Anima's time part with 8 rows, its patch embedding in the first group of blocks (16.85) | With one row, and as a program of its own, Core ML puts the whole part on the CPU | All of Anima on the ANE |
| Attention scores under 64 MiB a tensor (`mil.DitQChunk`, 16.85) | Past about 65–68 MiB a score tensor, q kᵀ and its softmax run 2–3× slower per score, whatever the shape | 1024×1024 at 512 queries is just under it; larger images take smaller chunks |
| The other models' silu written with exp and real_div too (`blocks.SiLU`, 16.95) | The ANE's `silu` was most of the loss in lut8 | KL on the ANE: LFM3 lut8 0.058 / 0.012 → 0.022 / 0.0048 (English / Japanese, as on Core ML's CPU), PLaMo 3 lut8 0.0067 / 0.0069 → 0.0053 / 0.0042, PLaMo-Embedding cosine 0.99986 → 0.99993. Gemma 4's `gelu` is unchanged by rewriting, so it stays. Anima's DiT (the time embedding and the modulation's input): latent cosine with diffusers' fp32 over 3 prompts 0.846 / 0.555 / 0.751 → 0.948 / 0.672 / 0.977, PSNR 12.6–14.7 → 14.1–25.8 dB; its VAE's silu changes no image and stays |
| Qwen3.5's silu and softplus written with exp, log and real_div, the GDN's q scaled up 1024× (16.88) | The ANE's `softplus` gives 0 for some inputs (the decay becomes 1), its `silu` is 1.4% off and worse on small values in the tail; the products of o = q S fall to fp16 subnormals | KL on the ANE 0.33–0.76 → 0.005–0.007; passkeys in long prompts 1/3 → 3/3 |

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
| Checking drafts in one call (`--verify`, 16.117) | A prompt chunk checking 2–8 tokens costs 3–9 decode steps (the chunk's cost whatever its rows), so a version of 3 rows runs the decode step's operators over them: a GEMV of a few rows (the weights streamed once), the GDN operator over the rows in one call keeping the state after each, attention a row at a time | 3 rows 1.27 decode steps (Qwen3.5-0.8B). The states of the rows not kept are copied back by the host (`Keep`, 1.5 ms), not run again |
| Prompt store and setting conversations aside | Prefill after a restart 13.2 s → 0.0 s | LFM3 copies its convolution state at each chunk boundary to roll back (16.66) |
| CLFLUSH from user space, a flush right after mapping | Fixes first results coming out as 0 from dirty cache lines | |
| One xclbin per step (for NPU1) | Runs on Phoenix / Hawk Point, which have no full ELF | 2.6 tok/s (full ELF 6.1). Not yet run on a real NPU1 |
| xrtshim (Windows) | Windows through XRT | 6.1 tok/s on Linux. Not yet run on Windows hardware |
| Power (PPT) | Measurement | 11.8 W while generating, 1.93 J a token (idle included). GEMV at 60 % of the CPU's speed, 3–5 mJ per 32 MiB (CPU 7–12 mJ) |
| 4-bit (q4nx) and 8-bit (q8) weights (`iron_q4.py`, `q4_mv.cc`, `q8_mv.cc`) | Generation 1.3–1.6×, pinned 1/2–1/5 | Decode uses `QGEMV`; with 2 cores per column it reaches 55 GB/s like bf16. Prompts turn weights back into bfp16 or bf16 for the GEMM. Swapped in on hwnpu's side without changing IRON |
| Host tables (`iron.HostTable`) | PLE and embeddings are not pinned | The tokens' rows are written before each call. Output word for word the same |
| `-idle` (`Release` / `Reload`) | A resident server gives back pinned memory while idle | E2B 1775 MiB → 0. The conversation is copied to the host and continues |
| Towers loaded only when used (`growBanks`, 16.81) | One export with images and audio serves text alone at the text export's memory | Gemma 4 E2B 2818 → 1826 MiB (the text export: 1775), LFM3-Omni 2448 → 1679 MiB until a tower runs; the towers come in 0.6–0.7 s. Same answers. No export again |
| Windowed attention (`iron_window.py`, 16.84) and the chunk barrier (`chunkBarrier`) | Prompt chunks −12 % (E2B 4K, PLaMo 3 8K), decode −2 to −14 % | Attention of one window layer alone 112 → 66 ms (chunk at 3072). The window form made the second chunk's first command hang in the full ELF (16.80's hang); an empty command in the decode step's hardware context before each chunk (0.3–0.5 ms) prevents it. The cause is unknown |
| A general graph export (`iron_graph.py`, `iron.Graph`) | Models other than language models (Anima) | One ELF can hold several weight banks (`variants`). The DiT's 7 sets compile once |
| Splitting commands (`SCRATCHPAD_WORDS` 24) | Keeps each command under amdxdna's 2 s timeout and the 64 MiB heap | Without the gather, commands grew too long; they are now split at export |
| Retrying when the heap runs out (`ENOSPC`) | Releases the decode run and tries again | Same as IRON |
| MoE building blocks (at Qwen3.6-35B-A3B's sizes, 16.75) | Reads the chosen experts in place without copying (`ExpertQGEMV`). A layer in 4 ops (`MoEGateUp`, `MoEDown`) | 5.2 → 2.3 → 1.27 ms a layer. Estimated 8–10 tokens/s. Pinning 35 GB does not change the speed |
| An IRON operator for Gated DeltaNet (`iron_gdn.py`, 16.89) | mlir-aie's recurrence kernel, with the rows gathered per column and the state moved a slice at a time (8 columns × 2 cores) | 0.26–0.33 ms a layer in decode, 0.11 ms a row in a prompt. Cosine 1.000000 against the reference |
| Choosing the MoE route on the device (`iron_moe.py`, 16.89) | The routing command writes the experts' offsets into the experts' command's scratchpad (a feedback buffer in the device heap). The host neither reads the logits nor writes the words | A layer is two commands, A (mixer and routing) and B (experts, 1.0 ms). With Qwen3.6's 2 layers, NPU vs the host graph: cosine 0.99997 |
| Rounding to nearest even in hwnpu's kernels (16.78) | Removes the downward bias of bf16 narrowing that grows with the number of inputs | Unit check with inputs of mean 1: IRON's `DequantBFP` path cosine 0.968, hwnpu's `QDequant` (bf16) 0.99994. Language models keep `DequantBFP` for prompts (bf16 is 5–9 % slower there and changes nothing) |
| Anima's DiT at q8 (`iron_anima.py --dit-attn`, `--dit-mlp`, 16.78) | Pinned 5020 → 3445 MiB at 1024×1024 | 2 % slower (weights dequantized every step), same energy (about 1050 J an image). `q4` can be chosen (2710 MiB) |
| Sharing files between exports (`iron.ShareExport`, 16.79) | Disk: the same files, and the same start of a bank, of exports side by side are kept once | Gemma 4 E2B text and with images and audio 15.2 → 8.2 GiB on Btrfs. Once per export, before it loads. Anima's images stay byte for byte the same |
| The prompt's attention on several pipelines (`iron_export_lm.py --mha-pipelines`, default 8, 16.107) | The prompt chunk's MHA on 8 pipelines (4 for chunks of 256), as IRON's Llama profile has it | Chunks 2–5× faster, energy −66 to −83 %. Logits bit for bit the same. Takes an export again. Qwen3.6 keeps 1 (not measured) |
| 6-bit weights (q6, `kernels/q6_mv.cc`, 16.105) | q8's layout with 6-bit codes (6.5 bits a weight) | Relative RMS weight error q4 0.071, q6 0.022, q8 0.005. The GEMV takes about q8's time (bound by taking out the high bits). For the small Qwen3.5's MLP and Anima's DiT (`--dit-attn q6`) |
| Qwen3.5's embeddings on the host (16.104) | The head holds a quantized copy, so the bf16 table on the device was unused | 0.8B −0.47 GiB, 2B −0.95 GiB. Logits bit for bit the same. Qwen3.6 keeps it (its routing reads it) |
| GDN columns in whole blocks of 64 rows (16.104) | Puts the fused projection of the Qwen3.5 with as many value heads as key heads (0.8B, 2B) on q8 / q4 | A column 772 → 776 channels (rows of zeros). 4B unchanged |
| Anima's DiT in one bank (`iron.GraphOptions.SwapVariants`, 16.106) | One bank for the 7 groups, a group's weight image (304 MiB) read into it when it runs | Pinned 3445 → 1236 MiB, denoising +2 %, latents byte for byte the same. `-resident-dit` keeps all |
| hwnpu-xclbin and tools/iron (amd-xdna2-bench.md) | Builds IRON designs without XRT's xclbinutil. Measures the data types and DMA | Matrix multiply at 1–4 tenths of the peak (int8 7.3–7.6 TOPS, bf16 2.7–3.0 TFLOPS at 4096³). DMA read 55–57 GB/s, write 48 GB/s, both 64 GB/s. A run costs at least 0.06 ms |

### 7.4 Common (any NPU, on the host)

Prompt reuse and the prompt store, speculative decoding (Apple / Intel, AMD's Qwen3.5), the think budget, repetition detection, hwnpu-agent's guards (no editing a file it has not read, stopping repeated calls, refusing a `cd` outside), and the search tools.

Sharing identical files on the disk (`cache.Clone`, 16.64 and 16.79) depends on the file system, not the NPU: clonefile on macOS's APFS, FICLONE and FICLONERANGE on Linux's Btrfs, XFS and others, block cloning on Windows' ReFS and Dev Drive. It covers Apple's programs and compiled copies, Intel's IR (`.bin` of the programs for 8 and 64 tokens) and AMD's exports. Nothing happens on ext4, tmpfs or NTFS. Only macOS cannot share the start of a file.

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
| Anima's DiT at 4 bits (q4nx) as the default | Pinned 3445 → 2710 MiB, but latent cosine 0.777 and the picture changes (the loss is the quantization itself: 0.753 on the CPU). GPTQ in the same format (16.87, 512×512): velocity error 0.123 → 0.071, latent cosine 0.868 → 0.939 (q8: 0.013, 0.962). Closer, still q8 by default; `--q4-codes` makes it a choice | 16.78, 16.87 |
| AMD: language models' q4nx prompts through bf16 (`QDequant`) | KL unchanged (E2B 0.094, LFM3 0.037), prompt chunk 5–9 % slower | 16.78 |
| RMSNorm with a fixed scale (Anima, fp16) | Squares of rows without large values become subnormals. Replaced by dividing by each row's maximum | 16.69 |
| Apple: faster self-attention for Anima | Core ML's `scaled_dot_product_attention` 110 ms (101 now), the scale on q 103, heads in groups 96–118, channels first 122. Heads 2 × queries 256: 90 ms, but 84 s to compile 4 attentions. Without the row max 87 ms, but fp16 overflows (bounds from the norms' weights reach 221) | 16.85 |
| Apple: Anima's DiT at lut8 | As fast as int8 but further from fp16 (latent cosine 0.858 against 0.902) | 16.85 |

## 9. Caveats when combining (costs)

- **lut6 and tiers (`-tier`)**: tiers with lut6 exceed the ANE's cache (about 4 GiB) and every startup takes 1–2 minutes. `-tier auto` makes tiers only with lut4. The split form (`-paged`, now the default) has no tiers
- **E4B as one program**: startup and short chats are fast, but reading a 16K prompt goes from 41.6 to 182.6 s (4.4×). `-prefill on` brings back two programs. After 16.83 the two read 16K in 24–25 s and 64K in 168 s (one program: 184 and 1383 s), but every start takes 135–165 s and generation at 16K is a fifth slower
- **Two-level attention**: it is an approximation, so the error grows when attention focuses on a single key (relative error 0.068–1.6 on a small Gemma 4). It passes the passkey test (LFM3 on the M3 up to 250K, and the three in a document of 128K). Around 64K a prompt reads a tenth slower. `-sparse=false` turns it off
- **Prefill of long contexts**: reading 256K at once takes about 29 minutes (LFM3). The prompt store is assumed
- **Noise in the numbers**: many M3 measurements were taken with 5–11 GB of swap. Only pairs measured back to back under the same conditions are compared. On AMD too, a single op can differ by about 1.4× between the first and second run (thought to be the power state). PLaMo 3's 4-bit numbers were measured with another job loading the machine
- **4 bits on AMD**: how much accuracy drops depends on the model (KL 0.036 for LFM3, 0.097 for E2B). For bf16, export without `--q4`
- **Anima's DiT at 8 bits**: 2 % slower than bf16. With a seed whose trajectory branches easily, the composition can still change (one of 4 prompts: PSNR 15.4 dB against bf16, the furniture in front differs). For bf16, export with `--dit-attn bf16 --dit-mlp bf16`
- **The rounding fix on AMD**: it is in the exported kernels, so 4-bit exports made before #48 keep the old rounding until exported again (the difference for language models is small)
- **`-idle`**: memory is given back while idle, but the next request first pays 0.8–2 s to reload. `-idle 0` turns it off
- **Towers loaded only when used**: the first image or clip after a start (or after `-idle`) waits 0.5–0.7 s more for the towers, and for that moment the old and the new bank are both pinned (Gemma 4 E2B 4-bit: 1.67 + 2.64 GiB). After that the towers stay until the model is released
- **The chunk barrier** (16.84): every prompt chunk first runs an empty command in the decode step's hardware context (0.3–0.5 ms a chunk). Without it, exports with the windowed attention hang on their second chunk. Exports with an xclbin per step (NPU1) run none
- **PLaMo 3 as one lut8 program**: short texts are fast, but long documents are read about 10× slower (`-format lut6` gives two programs at lower accuracy)
- **Sharing on the disk depends on the file system**: ext4, the default of Ubuntu, Debian and many others, cannot share, so each copy takes its full size there. Fedora and openSUSE default to Btrfs. Put the models on Btrfs or XFS to share (on Windows, a Dev Drive). Whole files and the common start of files of one name are shared; bytes after a difference are kept separately

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
| Qwen3.5-4B | `qwen35` | byte-level BPE + NFC + Qwen's split rules | Qwen3.5 | Text only (the vision tower is not converted) | ◎ lut8, one program (Q=8), all on the ANE, KL 0.005–0.007, about 8 tokens/s | Converts (int8) | ○ `iron_export_lm.py qwen35 --q4`. 32 layers at 6.4 tok/s (16.94), 4.7 GB. Top token 8/9 against HF (the host graph at the same q4 misses the same one, 16.92) |
| plamo-3-translate-2b | `plamo3` | PLaMo (`tokenizer.jsonl`) | PLaMo 3 (PFML3, translation) | — | ○ lut8, one program, context 262144. Checked on an M3 that all of it is on the ANE, and the weight formats | Converts (int8). Translates on OpenVINO's CPU | ◎ q4 11.6 tok/s, 2.16 GiB, context 8192 |
| plamo-embedding-1b | `plamo-bi` | SentencePiece BPE (`tokenizer.model`) | — (embedding API) | — | ◎ lut8 + head-scale | Converts (int8) | — |
| BERT family (all-MiniLM-L6-v2 and so on) | `bert` | **Not implemented** (WordPiece) | — | — | Converts only (lut8) | Converts only (int8) | — |
| Anima-Turbo (image generation) | `internal/imagegen/anima` (built from one MIL description) | Qwen3 BPE + T5 Unigram | — (`hwnpu-image`) | Images out | ◎ M3: 1024×1024, 8 steps in about 40 s, 3.8 GB (int8) | △ Converts. Checked on OpenVINO's CPU | ◎ 1024×1024, 8 steps in about 68 s, 3.4 GiB pinned (DiT at 8 bits) |
| Qwen3.6-35B-A3B | `qwen35moe` (IRON only) | Same as Qwen3.5 | Qwen3.6 (`Qwen36`: tool arguments as JSON, 16.90) | — | — | — | ○ `iron_export_lm.py qwen35moe --q4`. All 40 layers (23 GB): decode at 5.8 tok/s (16.92), prompts in chunks at 25 ms a token (routed on the host, 16.93), cosine 0.989–0.997 against HF (experts and head by GPTQ, `qwen36_gptq.py`, 16.101), every top token agrees. Prompts go a token at a time |

### 10.2 Tokenizers (`internal/tokenizer`, depending only on `golang.org/x/text`)

| Kind | Models | Implementation | Checked |
|---|---|---|---|
| byte-level BPE (`Load`) | LFM3 family, Qwen3.5 | Added tokens are cut out first, then the GPT-4 style pre-tokenizer pattern is applied by a hand-written scanner (it uses the lookahead `\s+(?!\S)`, which Go's regexp lacks). Merges by rank | LFM3: matches HF on 24 tokens (16.13) |
| Qwen3.5's extensions | Qwen3.5 | NFC normalization (adds `x/text/unicode/norm`), combining marks counted as letters with `[\p{L}\p{M}]+`, digits one at a time | Matches HF tokenizers 0.23 on Japanese, Hindi, combining marks, full-width and special tokens; 15 split cases match (16.62) |
| byte-fallback BPE (`fallback.go`) | Gemma 4 | Spaces become ▁, text between added tokens merged by rank as one piece, characters outside the vocabulary become UTF-8 bytes (`<0xE3>`). No BOS added (the chat format writes `<bos>`) | Matches HF on 35 texts + 300 random ones (7,026 tokens) (16.25) |
| SentencePiece BPE (`LoadSentencePiece`) | PLaMo-Embedding | Reads `tokenizer.model`'s protobuf by hand without dependencies. Merges by piece scores, not a merge list; user-defined pieces (runs of ▁, tabs, newlines) are matched longest-first and not merged; a ▁ is added after special tokens too (as HF's slow tokenizer does) | 0 mismatches over 32 texts (599 tokens) + 3,032 random texts (88,072 tokens) (16.19) |
| PLaMo (`LoadPlamo`; `Load` picks it for a `.jsonl`) | PLaMo 3 Translate | From `tokenizer.jsonl`'s scored pieces, finds the split of least total cost, searching from the end. The cost is the score × 10⁴, rounded as Python's round does, with the sign flipped. HF uses Aho-Corasick; hwnpu looks pieces up in a map of every piece's prefixes. On a tie the longer piece wins. Characters no piece covers become UTF-8 bytes. Cuts at special tokens, at 4 or more of the same character in a row, and at 2 or more spaces | Matches HF on 40 cases + 3,040 random ones. Flipping the tie rule fails the tests (16.70) |
| Qwen3's pre-tokenizer | Anima's text (Qwen3-0.6B) | GPT-4 style with digits split one by one (Qwen2 / Qwen3's rule) | Matches transformers on 11 texts (16.68) |
| T5's Unigram (`LoadUnigram`) | Anima's LLM adapter | Normalizes with sentencepiece's precompiled charsmap and splits with Viterbi. An unknown character costs "the lowest score − 10" (as HF). No byte fallback | Matches transformers on 11 texts (full-width, accents, Japanese, emoji) (16.68) |
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
| PLaMo 3 / PFML3 (`chat.Plamo3`) | 8 cases (`tools/ref/plamo3_template_golden.py`) | No tools. `target_lang`, `source_lang`, `form`, `context`, `glossary` (`Options.Translation`; in the API, `chat_template_kwargs` as in vLLM) | None. A system message becomes `context`. Without a target language, a text with Japanese goes to English and others to Japanese. The label of each past translation is chosen from its source text, so the start of the prompt can be reused |

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
- AMD: hwnpu's vision and audio towers on IRON's `Multimodal` (vision 512×512, audio up to 4,096 frames. IRON's MHA needs a head_dim that is a multiple of 64, so 96 is zero-padded to 128). The towers are loaded only when used (16.81): LFM3-Omni 4-bit 2448 → 1679 MiB while only text

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
  - With images and audio, the towers are loaded only when an image or a clip comes (16.81): 2818 → 1826 MiB while only text (the text export: 1775 MiB), the same answers
- E4B's old export hit `ERT_CMD_STATE_TIMEOUT` on a 2483-token prompt. The new export (62 commands) gets through

**Qwen3.5-4B** (16.62)
- Structure: 24 Gated DeltaNet layers + 8 full-attention layers. The GDN state (128×128 per value head) cannot be written per position, so a new kind of state `st_` was added (replaced whole by the new state the program outputs). `step_valid` keeps padding tokens from changing the state. When speculative decoding accepts only part of a draft, it reruns with the accepted tokens only
- Zero-centered RMSNorm (`1 + w`), an output gate, RoPE on the first 64 dimensions only. Sharded safetensors (`model.safetensors.index.json`) can now be read
- Checked: 3.7e-7 against an fp64 reference. All 32 layers against PyTorch fp32 (16.88): KL 0.0000 in fp16, 0.0015–0.0050 with lut8, 0.031–0.135 with lut6 (lut6 breaks answers, so lut8 is the default)
- Split into 9 parts at the attention (3.9 GiB with lut8), context 262144 (KV 32 KiB/token; 114688 on 16 GB)
- On an M3 (16.88): all on the ANE. Two programs exceed the ANE's cache, so one (Q=8). Second start 2–4.5 s. The ANE's `silu` and `softplus` made it lose long prompts; written with exp, log and real_div it reads them (KL 0.33–0.76 → 0.005–0.007, passkeys at 13K tokens 3/3)

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

**Anima-Turbo (image generation, `hwnpu-image`)** (16.68–69, 16.72, 16.78)

Parts:
- Text: Qwen3-0.6B and an LLM adapter (T5 tokens attend to themselves and to Qwen3's hidden states)
- DiT: Cosmos-Predict2 2B (28 blocks)
- VAE: the decoder of Qwen-Image's VAE
- 8 steps, CFG 1
- The host's work: tokenization, fetching embedding rows, rearranging patches, the Euler steps, PNG

AMD (`tools/amd/iron_anima.py`):
- About 70 s at 1024×1024 (7.2 s a DiT step, 60% of it self-attention)
- Against diffusers in fp32: latent cosine 0.966 (bf16 on the CPU: 0.980). The VAE alone: PSNR 44.5 dB
- Loaded one stage at a time, at most 5.0 GiB pinned; 3.4 GiB with the DiT's block projections at 8 bits (the default, 16.78), the picture nearly the same. 4 bits (2.7 GiB) change the picture

Apple / Intel (`hwnpu-image -convert`):
- One MIL description (`internal/imagegen/anima/mil`) is written as a Core ML package or OpenVINO IR, in Go only
- fp16 handling: DiT residuals kept at 1/32, norms divided by each row's maximum first
- Checked on the CPU: latent cosine 0.99985, PSNR 37–38 dB
- M3 (16.85): all on the Neural Engine. 1024×1024, 8 steps: 40.4 s (a step 4.2–4.4 s, VAE 3.7 s, startup 2.8 s from the cache; 3–5 min the first time), footprint 3.8 GB. 512×512: 0.8 s a step
- Made ready as hwnpu-chat makes its models: without `-model`, the checkpoint is downloaded into `hwnpu/models/Anima` and converted for the NPU found, the export named by its size, formats and layout version (`anima.Prepare`)
- Core ML's defaults are int8 for the text and the DiT: as fast as fp16 or faster (the DiT is bound by its attention, not its weights), and nearer fp16 than lut8. The text does not fit at fp16 (1.0 GB)
- Self-attention is about half of a block. Its cost follows the number of scores (q kᵀ about 16 G scores/s, the softmax about 8 G/s), not the inner dimension; the chunking, Core ML's own `scaled_dot_product_attention` and other layouts gain at most 11 %

**Qwen3.6-35B-A3B** (16.75, 16.89)
- Before bringing the model over, MoE layers that today's IRON can express were measured at its sizes (random weights). Reading 4-bit experts in place and folding a layer into 4 ops: 1.27 ms a layer
- AMD (16.89): Gated DeltaNet is an IRON operator of hwnpu's (`iron_gdn.py`). The route is chosen on the device: a layer splits into a routing command A and an experts' command B, and A writes the experts' offsets into B's scratchpad
- Cosine 0.998–0.9998 against HF on 2 layers of real weights (layers 0 and 3; q4nx experts). Decode only; a prompt runs a token at a time
- A takes 2.9–5 ms, B 1.0 ms, the head 5.9 ms (q4). Estimated about 210 ms (5 tok/s) for 40 layers. Most of the rest is A's op count (IRON's short convolution is about 10 ops)

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
| Qwen3.5's Q=64 program | Two programs (Q=8 and Q=64) take 7.9 GiB with lut8, over the ANE's cache, so it is not made by default. One program reads long prompts at 66 tokens/s (13K). Q=64's speed and compile time not measured | `-prefill on` | 16.62, 16.88 |
| `blocks.SiLU` in the towers and multimodal models | LFM3-VL / Audio / Omni / d1 with their vision and audio towers, TTS take the same rewrite and are converted again, but only LFM3-1.7B, PLaMo 3, PLaMo-Embedding and Anima were measured | Compare VL's image descriptions and TTS with the earlier version | 16.95 |
| Qwen3.5 with two-level attention | Do answers hold with two-level attention (`-sparse`)? | Passkey test | 16.62 |
| Two-level attention's `Refresh` as the default | With 4 or 16 the passkeys hold and a call is 14–32% faster (checked 2026-11-06), but finer differences such as loss or KL are not measured on the Neural Engine | Compare the logits with choosing every call on real weights | 16.61 |
| Where two-level attention starts in the prompt | At 64K reading in two levels is a tenth slower than the whole cache. Should the threshold of about 57K go up? | Compare the reading times from 64K to 96K | 16.61 |
| The copies in two-level attention's `choose` | Copying the buckets that change at each choice is 12% (13 ms) of a call generating at 128K. Would a choice that changes fewer buckets cut it? | pprof (`decode.(*Session).choose`) | 16.61 |
| Gemma 4's soft cap on the host (16.113) | Checked on the M1. The speed on the M3 and M2, the answers with temperature on the real model, Intel's IR | Compare the output layer's part with `HWNPU_PROFILE=1` on the same question | 16.113 |
| Slowing down past 128K | Why the slope per 1K grows from 1.14 to 1.43 ms (memory pressure, or attention lagging when interleaved?) | | 16.59 |
| Slowness right after a long prefill | Why the first few calls after reading 256K take 1.29 s each (350 ms after) | | 16.59 |
| The ANE's memory (the rest) | The weights' share is now read by `mem.NPUWired` (16.95, no root: 2.43 GiB for LFM3's two lut6 programs). The KV's and intermediate values' share, and the compile cache's, are not told apart | Compare after loading and while generating with a long context | 16.95, model-fit.md T10 |
| The ANE's compile cache | Its size (about 4 GiB is an estimate) and location, and why only unsplit Gemma 4 at lut6 does not stay | Looking into aned (sudo) | 16.41, 16.43 |
| Limits on M4 and later | The default per-part limit (0.9 GiB now; an M4 ran with 1.8 GiB and more), memory by number of parts | Measurements from M4 users | 16.12 |
| Gemma 4 E4B's default `-prefill` | `-prefill on` (two programs) reads long prompts 7.5–8× faster, but every start takes 135–165 s (one program: 86 s) and generation at 16K is a fifth slower. Switch by the prompt's length (load the program of 128 only to read, say), or let `auto` weigh the length? | Compare the round trips including the reload (3–14 s) and the start | 16.65, 16.83 |
| Gemma 4 E4B against the unsplit form | Skipped for lack of disk | | 16.65 |
| Gemma 4 and the prompt store | Save and restore time and size with Gemma 4 E2B | Replay the same OpenCode request as 16.38 | 16.38 |
| hwnpu-agent on Apple with Gemma 4 / Qwen3.5 | The templates support tools, but the agent tasks were not run (Gemma 4 E4B passed on AMD) | The tasks in CHAT.md section 7 | 16.35 |
| The rest of speculative decoding | Acceptance at Q=4, acceptance in OpenCode's round trips (7.4K, long XML), comparison with temperature 0 | Replays with `-dump` | 16.36 |
| d1-Omni-2B differing from its README | The team answer is the opposite of the README's. HF's code reads up to 16 questions together, hwnpu one at a time. Not compared with HF | Compare with HF's code where torch is available | 16.31 |
| LFM3-Audio's TTS | Does it sound the same as the reference (llama.cpp)? Only checked by transcribing it again with ASR | Compare audio decoded from the same codes with llama.cpp's server TTS | 16.30–31 |
| Long texts with PLaMo-Embedding | Do the last layer's values grow further in the 1024-token program (is head-scale enough)? | Vary the texts in `TestANEPrecision` | 16.34 |
| BERT family | `hwnpu-check` on real weights and placement on the ANE (only checked on the CPU for Intel) | | 16.11 |
| GPTQ's lut6 tables in the answers | On the ANE the KL falls as on the CPU, all on the ANE, same speed (16.95). Whether the answers get better is not told by 12 questions (10/12 either way). lut4 + wkmeans not tried on the ANE. Gemma 4 E2B (16.114): KL on the ANE 1.02 → 0.56 (English), 12 questions 10 → 11 of 12; hwnpu-chat takes them with `-codes`, not by default (2.1 GB of codes to hand out) | More questions and hwnpu-agent's tasks with and without the codes; a way to hand out the codes | 16.87, 16.95, 16.114 |
| Anima's remaining error against diffusers' fp32 | Measured in 16.95 (latent cosine 0.67–0.98 over 3 prompts after the DiT's silu). What is left is not split between the text's int8, the DiT's int8 and fp16 arithmetic | Swap the text and the DiT to fp16 one at a time | 16.85, 16.95 |
| Anima above 1024×1024 | The smaller attention chunks for sizes past 1024 (`mil.DitQChunk`) are not measured, nor whether the VAE stages fit | Convert 1280×1280 and run | 16.85 |
| Details of PLaMo 3's accuracy and speed | Placement and weight formats are checked. Translation quality and time for long documents are not measured in detail | Compare KL with fp16 on translation pairs | 16.71 |
| PLaMo 3's two lut6 programs | Do they (about 3.6 GiB) fit the compile cache (about 4 GiB) and keep startup fast? | Start twice with `-format lut6` | 16.71 |
| Why macOS memory seemed to shrink with use | The guess is that macOS compressed the rings, which are mostly zeros | Measure the footprint on an M3 after the mark fix (`ringRows`) | 16.71 |

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
| IR shared on Linux | The `.bin` of the programs for 8 and 64 tokens should be shared like Apple's `weight.bin`. Checked only by tests on Btrfs and XFS, not with a real conversion | 16.79 |
| PLaMo 3 | Whether the int8 IR (two programs of 4 parts each) runs on a real NPU, and its speed. On OpenVINO's CPU it gave the same translations as AMD | 16.71 |
| Anima | Whether all of it runs on a real NPU, speed, memory. Checked on OpenVINO's CPU (f32). With f16 inference, PSNR drops to 25.5 dB | 16.69, handoff-anima-mac-intel.md section 4 |

### 11.3 AMD (XDNA)

| Item | Unknown | Section |
|---|---|---|
| NPU1 (Phoenix, Hawk Point) | Whether one xclbin per step works on real hardware (checked only on a simulated 4-column NPU, and with the same export shape on NPU2). 6 xclbins use up the limit of 6 hardware contexts | 16.57 |
| Windows | xrtshim on real hardware and building with MSVC (checked only through XRT on Linux, and compiling with zig) | 16.56 |
| Sharing on other file systems | Tried on loop devices only: Btrfs (the tests, two Gemma 4 exports and three Anima exports through hwnpu-chat and hwnpu-image) and XFS (the tests); not bcachefs, ZFS or NFS | 16.79 |
| Sharing on Windows | ReFS and Dev Drive block cloning is only compiled (windows/amd64, 386, arm64). Whether it shares, the rounding of the last cluster, integrity streams | 16.79 |
| Power of the NPU alone | Upstream Linux 7.0 has no sensor. Current numbers are for the whole package (PPT) | 16.44, 16.53 |
| Long contexts | IRON exports have a fixed context (4096 / 12288). Attention over chunks is IRON's side's job | 16.63 |
| Other sizes of PLaMo 3 | 8B and 31B not tried | 16.70 |
| MoE warning about reusing BDs | aiecc warns about reusing a BD after a `dma_free_task` with no completion guarantee. Thought harmless, but not proven statically | 16.75 |
| Drafts on Qwen3.5-2B and 4B | Checking drafts in one call (16.117) is measured only on 0.8B (82 → 62–72 ms/token). 2B (q6) not measured; 4B is q4 and waits for a GEMV of a few rows at q4 | 16.117 |
| Pipelines of Qwen3.6's prompt attention | The exports' default of 8 (16.107) is not measured on Qwen3.6, which keeps 1 | 16.107 |
| Speed of q6's GEMV | About q8's time, bound by taking out the high 2 bits. Another layout might be faster | 16.105 |
| `iron_moep.py check` from 64 tokens | aiecc fails in `AIEDMATasksToNPUPass` (main as well). The prompt experts were checked against HF through Go instead | 16.116 |
| `iron_q4.py check --valid` | Bounding the prompt path's rows fails the check for q8 as well (unbounded it passes). Thought to be the check's, not fixed | 16.105 |

### 11.4 Behaviour with unknown causes

| Behaviour | Current guess | Section |
|---|---|---|
| Only unsplit Gemma 4 at lut6 / lut8 does not stay in the ANE's cache | A policy on the ANE's side (aned). The split form stays, so it no longer matters | 16.43 |
| Which shapes fall to the CPU at the 16384 per-dimension limit and which stay (chunks along an axis, weighting by ratios go to the CPU) | Inside Core ML's compiler | 16.21 |
| Only the 8-part split loads slowly from the cache (23.7 s) | Not looked into | 16.13 |
| On a 128K English document, two-level attention has lower loss than normal attention | Normal attention gets distracted by distant tokens, and two levels narrow that | 16.61 |
| AMD: with the windowed attention, a full ELF's second prompt chunk hangs in its first command | Not the words' values, the cut, the bank offsets or the operators (each passes alone and as an xclbin per step). A pause does not help; another hardware context's command does. Avoided with `chunkBarrier` | 16.80, 16.84 |
| AMD: splitting Anima's DiT into commands with host steps stops the first command with `ERT_CMD_STATE_TIMEOUT` | Not looked into. Avoided with one graph of 4 blocks | 16.68 |
| AMD: with `--command-per-step`, Qwen3.6's 2 layers give logits at cosine 0.32 against HF on a chunk of 256 (the normal export 0.98) | Not confirmed. Perhaps the per-step export routes before the router's logits are written (it picks experts 0–7). Use it for timing only | 16.116 |
| AMD: a single op differs by 1.4× between the first and second run | Thought to be the power state | 16.75 |

### 11.5 Not yet written (for reference)

- AWQ-style scales for lut4 (model-fit.md T8, 2). GPTQ choosing each weight's entry and k-means weighted with E[x²] fitting the tables are in since 16.87 (lut6 0.074 → 0.038 through Go's converter; lut4 2.7 → 0.33, still 4× lut6's rounding)
- hwnpu-chat fetching GPTQ's codes and using them by default: it takes them with `-codes` (16.114, a `-gptq-HASH` in the package name); how to hand out the codes (Gemma 4 E2B 2.1 GB, 1 byte an index) is open
- Writing the embedding tables at conversion so the checkpoint (LFM3 3.5 GB, Gemma 4 10.2 GB) is not opened at run time (T5)
- Conversations set aside moved out of the process (to disk), halved with int8 (the rest of T4)
- Changing the number of parts at run time (`MemoryBudget`, `Reshape`) (16.12)
- A WordPiece tokenizer (to use the BERT family from hwnpu-chat)
- Speculative decoding with a draft model (such as Qwen3.5's `mtp.` layers) (T9): Apple done (16.109); AMD not yet (the model's hidden states out and a version of the MTP layer)
- Repairing hwnpu-agent's broken calls (output-parser)
- AMD: the rest of checking drafts (16.117): a GEMV of a few rows for q4 (Gemma 4, PLaMo 3, LFM3 at 4 bits), Gemma 4's shared layers (last row only) and ring caches, attention over the rows in one MHA, fewer copies of the states
- The rest of AMD's pinned memory (amd-npu-models.md section 12):
  - Split the VAE into tiles (with the DiT in one bank (16.106), a VAE stage is now the largest)
- The rest of Anima:
  - CFG (negative prompts). A non-Turbo Anima is expected to take over 10 minutes an image
  - Calls from ComfyUI nodes
- The rest of Qwen3.6-35B-A3B's prompt (16.116; a chunk of 256 2.6 s): the experts' command (B, 1.49 s) is bound by the kernel's arithmetic (streaming the weights alone takes 8.4 + 4.4 ms of 19–23 + 15–17 ms a layer; a scalar MAC, no mmul. bf16 mmul in groups of 8 rows was tried and was not faster), the GDN and attention command (A, about 0.7 s; `GatedDeltaRule` receives every row 16 times), the host's routing (about 2.8 ms a layer, mostly a 16 MB copy of x)

## 12. The speed ceiling and what software can still move (estimate of 2026-11-08)

This section uses `npuprobe` on the M1, M2 and M3 (PROBE.md, `npu_profiles/`) and the AMD measurements (amd-xdna2-bench.md) to answer two questions: does a faster NPU make generation faster, and what can software still change? 12.2 is an estimate from public specs and has not been measured.

### 12.1 The ceiling is how fast the NPU reads DRAM

With 16 layers (0.97 B):

| | M1 | M2 | M3 | AMD (Ryzen AI 9 365) |
|---|---|---|---|---|
| NPU, rated | 11 TOPS | 15.8 TOPS | 18 TOPS | 50 TOPS |
| DRAM bandwidth | 68 GB/s | 100 GB/s | 100 GB/s | 90–120 GB/s |
| int8 weights read at | 62 GB/s | 62 GB/s | 23 GB/s (slow on the M3 only) | — |
| 4-bit weights read at | 44 GB/s (lut4) | 63 GB/s (lut4) | 64 GB/s (lut4) | 55 GB/s (q4nx GEMV) |

- **The ANE reads weights at about 62–64 GB/s on every Apple chip measured.** DRAM grew 1.47× from the M1 to the M2, but int8 stayed at 62 GB/s. On the M2 and M3 the limit is the ANE's own path to memory, not DRAM (100 GB/s)
- **Moving bytes is the limit, not computing.** On the M2, int8 and lut4 differ 2× in work per byte yet both read at about 63 GB/s
- **Only the M1's lut4 was compute-bound.** Expanding the 4-bit table is slow there, and the M2 was 1.41× faster (close to the TOPS ratio of 1.44). The M2 reached the ceiling. The M3 has 14% more TOPS and went from 63 to 64 GB/s
- **AMD is the same.** DMA tops out at 55–57 GB/s with 8 channels (14 GB/s each). The 4-bit GEMV reaches that with 16 cores, and 32 cores are no faster (2.4× spare compute)
- TOPS helps only prefill and attention over a long context (the one-row matrix-vector products the ANE cannot fill). One generated token follows model-fit.md section 2: weight bytes ÷ read bandwidth + attention + about 3 ms. A faster NPU shrinks only the second term

### 12.2 What newer chips may bring (not measured)

Generation of LFM3-1.7B (lut4, short context) compared with the M3. The NPU's share of DRAM is assumed to be 50–60%, from Apple's 64% and AMD's roughly half.

| | NPU read, expected | Generation | Prefill | Runs on hwnpu |
|---|---|---|---|---|
| Apple M6 (two Neural Engines, 170 GB/s) | The same if one engine still reads 64 GB/s. Up to about 2× if both can be used at once or the path is wider | 1.1–2× | About 2× the M5's compute, 2–4× the M3 | Yes (Core ML) |
| Snapdragon X2 Elite (80 TOPS, 152 GB/s) | 75–100 GB/s | 1.3–1.6× | About 4× the M3 on paper (effective unknown; AMD reaches 10–40% of its rating) | No (no Qualcomm backend) |
| Snapdragon X2 Elite Extreme (228 GB/s) | 115–150 GB/s | 1.8–2.2× | Same as above | Same as above |

- The M6's second engine helps generation only if a layer's matrices can be split into two programs called at the same time. That doubles the fixed cost per call
- To check: run `npuprobe` on the new machine and read "weights GB/s" for lut4 at 16 layers. If it is about 64, generation is the same as the M3 and only prefill gains

### 12.3 What software can move

Tokens/s ≈ read bandwidth ÷ bytes read per token × tokens advanced per read (minus the time not spent reading). The bandwidth is close to the ceiling (12.1), so the other terms are what is left.

| Term | Method | State | Expected or measured | Section |
|---|---|---|---|---|
| Bytes read per token | Fewer bits (lut3 and lut4 mixed per layer, tables fitted to activations) | Up to choosing levels with GPTQ (16.87). Fitting the table is not done | A trade against accuracy | model-fit.md T8 |
| 〃 | MoE (read only the experts in use) | Done on AMD with Qwen3.6-35B-A3B (16.89–16.93). Not on the ANE (the routing `gather` goes to the CPU, model-fit.md section 4) | LFM2.5-8B-A1B (8.3B total, 1.5B active) reads about 0.75 GB a token at lut4, a floor of about 12 ms on the ANE. Computing every expert densely reads 4.2 GB, about 65 ms | — |
| Tokens advanced per read | Speculative decoding (prompt lookup) | Done on Apple and Intel. A Q=8 call costs 13% more than Q=1 | 15.9 → 23.7 tokens/s, 1.2–1.5× over a whole task | 16.33, 16.36, 16.37 |
| 〃 | A draft model (such as Qwen3.5's `mtp.` layers) for more hits | Apple done (16.109, M1 Qwen3.5-0.8B 26.1 → 43.8 tok/s on code). AMD not yet | Gains if it beats prompt lookup's 1.07–1.67 tokens a call | model-fit.md T9 |
| 〃 | Read once for several requests (batching) | Not done | One read serves several conversations. Total throughput grows when subagents run side by side | — |
| 〃 | Draft checking on AMD | Done for q8 / q6 (16.117): a check of 3 rows costs 1.27 decode steps. q4 not yet | Qwen3.5-0.8B 82 → 62–72 ms/token with prompt-lookup drafts | 16.117 |
| Time not spent reading | Fixed cost per call (about 3 ms), switching parts, host work | LFM3-1.7B at a 1K capacity: of 19.1 ms, 13.4 ms reading weights, 2.8 ms attention, 3 ms fixed | 15–16 ms expected after trimming the fixed cost and attention | model-fit.md section 2 |
| 〃 | AMD: one submission for a whole layer | Done (a full ELF is 2.7× faster than submitting each op). A fused layer reaches 83% of the read floor | Avoids piling up the 0.06 ms per call | amd-xdna2-bench.md section 4.10, amd-npu-models.md section 2 |
| Attention over a long context (compute-bound) | `SDPAGrouped`, splitting around attention to compute only the length used | Done (16.39, 16.59). Two-level attention (16.61) | Both TOPS and how the program is written shrink this | 16.39, 16.59, 16.61 |
| (For reference) Another reader | Use the DRAM bandwidth the ANE leaves (100 − 64 GB/s on the M3) from the CPU, for example drafting on the CPU and checking on the ANE | Not tried. Using the GPU is a non-goal | Whether syncing pays off is not measured | — |

Expected (M3, LFM3-1.7B, lut4, short context; rough): 19–21 ms now (about 48 tokens/s) → 15–16 ms after trimming the fixed cost and attention (about 60–65 tokens/s; the weight floor is 13.4 ms, about 75) → about 80–90 tokens/s effective with speculative decoding. The default lut6 reads 1.5× the weight bytes, so its weight term is that much larger.
