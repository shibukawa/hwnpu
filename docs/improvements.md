# What each improvement bought (as of 2026-11-08, main up to #83, with AMD's checks and changes: Hugging Face comparisons, PLaMo 3 0.8B and Qwen3.5-0.8B / 2B, 6-bit weights, Anima's DiT in one bank, the prompt's attention on 8 pipelines, and checking drafts in one call)

[日本語](improvements.ja.md)

This page sorts the changes by what they improved. It draws on the implementation log in design.md chapter 16 (up to 16.117), model-fit.md, intel-npu-models.md, amd-npu-models.md (sections 11–13), amd-xdna2-bench.md, handoff-anima-mac-intel.md and handoff-m3-gptq.md. The numbers are the measurements recorded there. Unless noted, they are from **the Neural Engine (ANE) of an Apple M3 (16 GB)** with LFM3-1.7B. AMD numbers are from a Ryzen AI 9 365 (Strix Point). Numbers in parentheses are design.md sections. Section 7.5 and the rows marked "Core AI" add the Core AI backend of macOS 27 (16.169–16.183), measured on an M1 and an M2 (16 GB, macOS 27.2) unless noted (16.182 and 16.183: an M3). The gpt-oss-20b rows (16.179) are AMD's numbers of 2026-11-14.

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
| AMD: the decode step in fewer operators: biases and residuals added in the GEMVs (QGEMVAdd), a gated MLP's gate and up as one table and one operator (QGEMVGlu), gpt-oss's q and k one RoPE (16.179) | ◎ Qwen3.5-0.8B 87 → 73 ms (−16 %), gpt-oss-20b 97.5 → 89.5 ms | unchanged (the same operators) | | unchanged (one table, no second copy) | | | unchanged (cosine 0.9999 to the export before) | AMD (needs exporting again) |
| AMD: gpt-oss-20b's decode step as one resident array, the output head a phase of it, no barrier before a step (`GptOssStep`, 16.179) | ◎ 79.2 → 50.6 ms a token (12.6 → 19.7 tokens/s): an operator that follows another is loaded again, 30 of the 79 ms | unchanged | | △ +1.06 GiB (the array's own tables and its array of keys and values; 0.67 GiB of it back with the next row) | | | unchanged (KL 0.031) | AMD (needs exporting again) |
| AMD: one table for the prompt and the decode step (`Q8XDequant`, 16.179) | unchanged | △ +1.9 % | | ◎ gpt-oss pinned 14.1 → 13.4 GB | ○ the image 0.7 GB smaller | | unchanged (the prompt's output bit for bit) | AMD (needs exporting again) |
| AMD: the decode attention's exponential and sums, the router on every column (16.179) | ○ 50.6 → 49.4 ms with the host's row below; ◎ long contexts: 66.1 → 56.1 ms at 7680 (15.1 → 17.8 tokens/s) | | | | | | unchanged (KL 0.0315, the same errors against the reference) | AMD (needs exporting again) |
| AMD: a sliding layer's decode cache as a ring of its window (16.179) | unchanged | | | ○ gpt-oss 12.41 → 12.22 GiB | | unchanged (going back and the prompt store as before) | bit for bit the same after going back | AMD (needs exporting again, and the hwnpu of the same change) |
| AMD: gpt-oss's prompt experts on bfp16, by slots, each block decoded once (16.179) | | ◎ a chunk of 256 tokens about 6 → 1.8 s (7 ms a token; a token at a time 106 ms) | | | | | △ KL of 5 texts 0.027 → 0.030 | AMD |
| AMD: a checkpoint's own 4 bits kept (gpt-oss's MXFP4 experts, mx4, 16.179) | (4.3 bits a weight, where q4nx takes 5) | | | | | | ◎ rounding them again to q4nx: KL 2–3× | AMD |
| The host's logits and top k in vector loops (16.179) | ○ gpt-oss 19.4 → 19.7 tokens/s (201088 logits: 1.33 → 0.23 ms a token) | | | | | | the same output | AMD (the logits), all (top k) |
| AMD: Qwen3.6's prompt experts' gate and up on four cores a column, rows an expert 128 → 256 (16.116) | unchanged | ◎ chunk of 256 3.9 → 3.0 s (always one pass of the experts' weights), 2.7 s with the host's combine on all CPUs, 2.6 s with the experts' down on four cores a column | | | | | unchanged (logits bit for bit the same) | AMD (needs exporting again) |
| AMD: Gated DeltaNet's columns aligned to 64 (`COLUMN_ALIGN`, 16.117) | ○ Qwen3.5-0.8B decode 85.6 → 82.5 ms (the fused projection on 8 columns, not 1) | | | △ zero rows (0.8B +7 % of that projection) | | | unchanged (logits bit for bit the same) | AMD (needs exporting again) |
| AMD: the prompt GDN's silu once before the operator (`GdnSiLU`, 16.111) | unchanged | ◎ Qwen3.6 chunk of 256 4.5 → 3.9 s (GDN layer 43 → 22 ms), Qwen3.5-0.8B chunk of 512 0.9 → 0.5 s | | | | | unchanged (logits bit for bit the same) | AMD (needs exporting again) |
| AMD: Qwen3.5-0.8B / 2B with 6-bit MLPs (q6) and the embeddings on the host (16.104–105) | unchanged | | | ◎ 0.8B 1858 → 1069 MiB, 2B 4017 → 2014 MiB | | | ◎ KL from HF on the GPL 0.34 at q4 → 0.044 (0.8B) | AMD |
| AMD: image DiTs' weights packed as bfp16 on the host (`bfp16`, 16.200) | ◎ a step: Anima-Turbo 1024² 4.1 → 3.6–3.8 s, Qwen-Image 1024² 11.7 → 8.4 s (with the head norms), Chroma1-Flash 29.9 → 22.8–25.4 s, Chroma1-HD 27.5 → 20.2 s; Sana 600M −5 % alone (few of its projections tile) | | | △ Anima's bank 464 → 479 MiB; the runs' peaks the same | △ 9 bits a weight: Qwen-Image's DiT 6.2 → 7.6 GB (its MLP was q6), Chroma's 8.6 → 9.2 GB; exporting 3–4 × as long | | ○ the first step's velocity error between bf16's and q8's (fp32: Anima 0.0246, q8 0.0253, bf16 0.0232) | AMD (needs exporting again) |
| AMD: GELU's tanh argument in float32 (`GELUEven`, 16.200) | ○ Anima 1024² 4.6 → 4.2 s a step (IRON's exact GELU was 21 of a block's 154 ms) | | | | | | unchanged (IRON's tanh kernel, its argument in bf16, changed the clothes of a test prompt) | AMD (needs exporting again) |
| AMD: the next group's weights read while a group runs (`GraphOptions.Prefetch`, 16.200) | ○ Qwen-Image-Turbo 1024² 1m57s → 1m50s, Anima-Turbo 43 → 42 s | ○ Qwen-Image's text 7.45 → 4.76 s | | △ one bank more at the DiT stage, only where the VAE's peak covers it (512² mostly not) | | | bit for bit the same | AMD |
| AMD: Sana's depthwise convolutions as one operator a branch (`DWConv3`, 16.200) | ◎ Sana 600M 1024² denoising 42 → 29 s, SANA 1.5 83 → 57 s (−31 %) | | | unchanged | ○ the MLP's channels padded with zeros (600M 2880 → 3072) | | unchanged (latent cosine 0.9997–0.9999) | AMD (needs exporting again) |
| AMD: the modulation's scale in LayerNorm's weight, q and k RMS-normed a whole row a call (16.200) | ○ LayerNorm and scale 1.23 → 0.61 ms, the head norms 1.4 → 0.83 ms (4096 × 2048) | | | | | | bit for bit the same (Anima) | AMD (needs exporting again) |
| AMD: Anima's DiT in one bank (`SwapVariants`, 16.106) | △ denoising +2 % | | | ◎ pinned 3445 → 1236 MiB | | | bit for bit the same | AMD |
| Anima loaded one stage at a time (16.72) | 69.3 → 67.7 s/image | | | ◎ 14.2 → 5.0 GiB | | | bit for bit the same | AMD |
| AMD: Anima's DiT at 8 bits (q8) (16.78) | +2 % (67.3 → 68.3 s/image) | | | ◎ pinned 5020 → 3445 MiB | ○ 5.7 → 4.0 GB | | nearly unchanged (latent cosine 0.966 → 0.961) | AMD |
| AMD: the cores' rounding set to nearest even (16.78) | unchanged | unchanged | | | | | ◎ Anima's quantized DiT (step-0 error q8 0.088 → 0.031); language models unchanged | AMD |
| Apple: Anima on the Neural Engine (16.85) | ◎ runs (`dit_time`, `dit_patch` and the fp16 text had fallen to the CPU) | | | | | | | Apple |
| Apple: `hwnpu-image` with the compile cache (16.85) | | | ◎ 118 → 4.4 s (512×512), every start had compiled the DiT again | | ◎ no 4 GB of bundles a start | | | Apple |
| Apple: Anima's attention 512 queries at a time (16.85) | ○ a step 7.0 → 4.5 s (1024×1024) | | | | | | unchanged | Apple |
| Apple: Anima's VAE with `upsample_nearest_neighbor`, loaded one stage at a time (16.85) | ◎ VAE 14.5 → 3.7 s | | | ◎ 6.9 → 3.8 GB | | | stage 5 rel. error 4.8e-4 | Apple |
| Apple: Anima at int8 by default (16.85) | ○ a step 4.5 → 4.2 s | | ○ | ○ | ○ export 4.6 → 2.8 GB | | △ latent cosine 0.902 against fp16 (lut8 0.858); with the DiT's silu fixed (16.95), fp16 is only 0.006–0.025 nearer diffusers' fp32 | Apple |
| Apple: Core AI backend (`--backend coreai`, macOS 27) (16.169, 16.173) | unchanged (±10 %) | | ◎ second start 1.6–5.5 s for every chat model (Core ML: 4 s to 13 min, when parts fall out of its ~4 GiB cache); Qwen-Image prototype 46 s an image (Core ML 4 min 26 s) | | △ the weights on the disk three times (below) | | unchanged (Qwen3.5-2B, Gemma 4 E2B, LFM3 word for word; Anima's latent cosine 0.971 against Core ML) | Apple (macOS 27) |
| Core AI: each program's input arrays made once and filled by memcpy (16.172) | ○ PLaMo 3 0.8B 25 → 38–44 tok/s (Core ML 35–42) | | | | | | unchanged | Apple (Core AI) |
| Core AI: parts with the same weights in one `.aimodel` of several functions (16.177, 16.178) | ○ LFM3-1.7B +13 % | | △ the first compile pays every function | ◎ the Neural Engine holds the weights once: LFM3-1.7B 2.56 → 1.34 GiB | ◎ chat models about 9.3 GiB, Anima 3.0 GiB; Chroma1-Flash's 512² beside 1024² adds 49 MB instead of about 17 GB | | bit for bit the same (Anima and Chroma images, greedy answers) | Apple (Core AI) |
| Core AI: the prompt program kept (weights counted once, 16.177) | | ◎ 2.3–3.2× (Qwen3.5-2B, LFM3-VL / Audio / Omni, PLaMo 3 2B) | △ +1–3.6 s | △ +0.1–0.14 GiB | | | unchanged | Apple (Core AI) |
| Core AI: an image model's sizes in one export, a function each (Sana's 512² and 1024², 16.189) | | | ◎ the second size compiles with the first: Sana 600M's 512² in 7.0 s right after its first 1024² | | ◎ Sana 600M 4.0 GB for both (Core ML 3.9 GB a size) | | Core AI against Core ML 0.9976 (latents) | Apple (Core AI) |
| Core AI: the DiT's attention scale in the queries' norm, the norms' magnitudes by `abs` (16.201), the attentions a head at a time with the scores laid out (1, queries, 1, keys) (Anima, 16.202) | ◎ Anima-Turbo 7.1 → 5.6 s a step at 1024² (a picture 1 min 02 s → 48 s); Anima-Aesthetic 25 steps 4 min 48 s → 3 min 51 s | | | | | | the same distance from the CPU's float32 (512² latents 0.9984, 0.9992, 0.9872; the attention before 0.9979, 0.9992, 0.9837); against before 0.970–0.998 (Turbo, 9 pictures at 0.982–0.9998 against the heads together), 0.99995 (Aesthetic) | Apple (needs exporting again) |
| A DiT's cross-attention keys and values once a prompt (Sana, 16.189) | ○ steps as before (0.63 s, 1.37 s at 1024²) | | | ○ the DiT the steps hold 555 → 497 MiB (600M), about 1.49 → 1.30 GiB (1.6B) | | | ○ the projections in fp16: latents 0.964 → 0.972 against float32 | Apple (needs exporting again) |
| Core AI: Qwen3.5-4B and gemma-4-E4B-it from the checkpoint, both programs kept (16.191) | ◎ Qwen3.5-4B 20.5–21.9, E4B 20.3–20.7 tok/s (M2; Core ML 14.5, 15.7) | ◎ a 2K-token prompt 4.0 / 1.8 s (Core ML 29.5 / 22.8 s: one program there) | ◎ 0.7–1.0 s (Core ML 4.4–5.8 s) | | ○ no `.mlpackage` | | right answers; parts' cosines as Core ML's | Apple (Core AI) |
| Core AI: Qwen3.5's MTP drafts over a part of the vocabulary (74272 of 248320 rows: low IDs, kana, kanji words of up to 2, ASCII symbols; `llm.Options.HeadRows`, as FR-Spec) and its attention over shorter chunks (16.192) | ◎ Qwen3.5-4B 17.4–18.2 → 19.6–20.1 tok/s (an MTP step 13.6 → 6.6 ms), then 19.2–21.0 → 20.5–21.9 (its attention 3 → 0.9 ms) | | | ○ the MTP's head part 0.68 → 0.26 GiB | ○ the same | | bit for bit the same answer (every draft is checked) | Apple (Core AI) |
| Qwen3.5's Gated DeltaNet in chunks of 8 rows (the chunked WY form: the 1 MiB state read and written 3 times a chunk instead of 5 times a row; the inverse applied as B − N(B − N(…)), since an 8 x 8 by 8 x 8 product puts a program off the Neural Engine; 16.193) | ○ Qwen3.5-4B a step 96 → 83–89 ms (a GDN part 9.6 → 8.3 ms, near the read ceiling) | ◎ 128 tokens 511 → 304–344 ms | ○ a part's first compile 15.8 → 7.0 s | | | | the same error per part (cosine 0.999996); greedy answers part near the end | all (the same MIL; checked on Core AI) |
| Core AI: dropping the `.aimodel`'s weights once compiled (`HWNPU_COREAI_DROP_BYTECODE=1`, opt-in, 16.176) | | | ○ Chroma's DiT group 6.5 → 0.5 s | | ○ Chroma1-Flash 512²: hwnpu's cache 28 → 15 GB | | unchanged | Apple (Core AI). △ a purged entry means lowering again (or a download, for distributed `.aimodel`s) |
| Apple: Chroma1-Flash on Core AI (16.176) | ○ M2 1024² 6 min 58 s an image (M1 on Core ML: 9 min 17 s) | | △ first run 1 hour at 1024² | DiT held 8.15 GiB | | | latent cosine 0.9976, PSNR 35.4 dB against Core ML | Apple (Core AI) |
| Apple: the image-reading models on Core AI (PP-OCR, RTMDet, RTMO, YuNet, DA3, MoGe-2, EfficientSAM; 16.182, 16.183) | unchanged (a call ±3 ms) | | unchanged with both caches warm (0.05–0.9 s a command, three starts each); Core AI keeps its programs where Core ML compiles again what left its ~4 GiB cache (1–16 s seen); first 0.2–38 s a part | △ the Neural Engine holds a little more (RTMDet 56 → 89 MB) | | | the same OCR text, objects, people and face; depth cosine 0.999999–1.000000 against Core ML (M3) | Apple (Core AI) |
| Apple: SigLIP 2 (`hwnpu-image label`, `embed`) on the Neural Engine, converted in Go on the first run (16.183) | base 7.3–7.9 ms a picture and 15–17 ms for 8 texts, large 33–36 ms and 47–59 ms (M3; AMD: base 84 ms and 131 ms) | | first load 13 s (base), 33 s (large) | base 0.45 GB of the process and 0.36–0.48 GB on the Neural Engine, large 1.8 GB and 1.2–1.6 GB | export 0.73 GB (base), 1.7 GB (large) | | embeddings' cosine against Transformers 0.99996 (base) and 0.9999 (large) for pictures, 0.99999 for texts, on Core ML and Core AI | Apple |
| Conversation marks keep only the ring slots in use (16.71) | | | | ○ 578 → 183 MB over 5 translations | | | | All |
| Memory back to the OS after a conversion (16.71) | | | | ○ 2.4 GB → 154 MB right after startup | | | | All |
| Cleaning up the `/dev/shm` left by OpenVINO (16.72) | | | | | ○ 8.4 GB a day | | | Intel / OpenVINO |
| Apple: sigmoid and SiLU written with exp in ONNX programs (`ExpSigmoid`, 16.149) | △ RTMDet-s 6.6 → 9.3 ms an image | | | | | | ◎ RTMDet-s box 98 → 5.5 px, score 0.35 → 0.063 off ONNX Runtime | Apple |
| Host resize without closures, rows on every core (16.149) | ◎ 37–55 → 5–7 ms a picture (RTMDet, RTMO, LFM3-VL) | | | | | | the same bytes | All |
| Apple: PP-OCR's recognizer on the NPU (logits before the softmax, the 18710-wide output in two parts, the bias in the linear, fp16 read as it is, decoding during the next call) (16.149) | ◎ a page of 98 lines 1528 → 488 ms | | | | | | the same lines | Apple |
| PP-OCR's recognizer at seven widths padded as PaddleOCR (16.149) | | | △ a program to compile for each width (5–20 s, once) | △ 425 MB at most | | | ○ the receipt 11 → 13 of 13 lines as PaddleOCR | All |

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

**gpt-oss-20b on AMD, a decode step by step (16.179).** A token moves 2.53 GB: 24 layers of 88 MB (the 4 experts 57 MB, q/k/v 17.6 MB, o_proj 13.7 MB) and the head's 415 MB. At 55 GB/s that is 46 ms.

| Change | Before | After | Notes |
|---|---|---|---|
| Biases and residuals in the GEMVs, a gated MLP's gate and up as one operator, q and k one RoPE | 97.5 ms | 89.5 ms | Generation was bound by the number of operators (80–150 µs each) |
| A layer's experts as one operator, scales as MXFP4's exponent bytes (`MX4Experts`) | 89.5 ms | 80.5 ms | Each operator carried about 0.4 ms of its streams' start and end |
| q, k, v with the norm, the bias and RoPE in one operator (`QGEMVQKV`) | 80.4 ms | 79.2 ms | Two operators fewer a layer |
| The step as the phases of one resident array (`GptOssStep`) | 79.2 ms | 54 ms | An operator following another is loaded again: about 67 µs and 1.2 µs a KB of its cores' programs, 30 ms a token |
| The head a phase of the array, no barrier before a step | 54 ms | 50.6 ms | The barrier's change of hardware context put the whole array back (2.4 ms a step); the head's q4 table is read in place |
| The attention's exponential and sums, the router on every column, the host's logits | 50.6 ms | 49.4 ms (20.2 tokens/s) | The host 1.1 ms of it; the rest shows with the context, below |

| Context | Before the attention's change | After |
|---|---|---|
| 256 | 51.4 ms | 49.4 ms |
| 2048 | 54.7 ms | 50.8 ms |
| 4096 | 58.7 ms | 52.3 ms |
| 7680 | 66.1 ms | 56.1 ms |

- The steps' own time, 256 steps (`ironlm -forced`), both columns with the host's change. An object of 128 rows went from 21 to 9.6 µs a full-attention layer, 5–6 µs of it the transfer
- What is left above the floor's 46 ms: the 24 command boundaries (0.1 ms each: one command a token and a chain of commands both failed, section 8), each phase's fixed 25–35 µs (its x objects and descriptors), and the host's 1 ms
- Energy: the same answer of 600 tokens took 779 J in 62 s when a step was about 90 ms and 553 J in 38 s now (the package: 1.3 → 0.92 J a token, the load included)

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
| AMD gpt-oss: the prompt's experts through the matrix units (16.179) | A chunk of 256 tokens about 6 s (the vector mac; a token at a time 106 ms a token) | 5.4 s (bf16 `mmul`), 3.4 s (bfp16, the native float matrix instruction), 1.8 s (slots of 56 rows, each weight block decoded once): 7 ms a token | The bf16 `mmul<4,8,8>` is emulated with broadcast macs on AIE2P. A layer of a chunk now: the experts 35 ms, the attention's side 12 ms, the host 4 ms |
| AMD gpt-oss: the prompt reads the decode step's q8x tables (`Q8XDequant`, 16.179) | 2048 tokens 15.8 s | 16.1 s (+1.9 %) | The price of holding the projections once (−0.67 GiB): writing rows of 320 bytes runs at 15 GB/s |
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
| gpt-oss-20b | — (its experts are MXFP4 in the checkpoint) | — | **12.22 GiB** (experts mx4, attention q8, head q4, embeddings a file on the host; 13.1 GiB before the prompt and the decode step shared the projections' tables, 12.41 before the sliding layers' ring, 16.179) | not tried |
| Anima (1024 × 1024) | 14.2 GiB | **5.0 GiB** loading one stage at a time | **3.4 GiB** (DiT at 8 bits; 2.7 GiB at 4 bits). **1.2 GiB** with the DiT's 7 groups read in turn into one bank (16.106) | — |

- Tables on the host: PLE reads only one row a token, yet the whole table was pinned. The table now stays a file, and before each call only the tokens' rows are written into a small device buffer (18 MB for E2B)
- A decode step on its own array (gpt-oss, 16.179) holds what it reads in its own order: its tables (the same codes as the prompt's until `Q8XDequant` let the prompt read them: −0.67 GiB) and an array of keys and values beside the prompt's caches (0.4 GiB; the sliding layers' half became rings of 128 rows, −0.19 GiB; the full layers' 0.19 GiB is left). Measure a process's pinned memory by the sizes of its `/dev/accel/accel0` mappings in smaps
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
| **Cost**: Core AI holds the weights three times (16.173, 16.176) | The `.mlpackage`, hwnpu's lowered `.aimodel`, and Core AI's own cache (`~/Library/Caches/coreai-cache/<OS build>/<executable>/<main.hash>/`). Chroma1-Flash: 14 + 14 + 14–17 GB, 73 GB with its checkpoints | Below about 15 GiB free, macOS purged `~/Library/Caches` — Core AI's entries and hwnpu's cache root with them. Delete the checkpoints once converted |
| Core AI: one `.aimodel` of several functions (16.174, 16.177, 16.178) | LFM3-1.7B `.aimodel` 2.43 → 1.22 GiB, Core AI's cache 2.53 → 1.32 GiB; Anima 7.36 → 4.32 GiB; Chroma1-Flash 512² + 1024²: +31 MB `.aimodel`, +18 MB Core AI | Siblings: the packages in the same directory, and of the same name in the exports beside it, that have the 3 largest weight blobs of the same bytes. `LowerFunctions` writes a blob of the same bytes once |
| Core AI: dropping the bytecode (`HWNPU_COREAI_DROP_BYTECODE=1`) (16.176) | Chroma1-Flash 512²: hwnpu's cache 28 → 15 GB | Opt-in: the default keeps the `.aimodel` (16.174), as a purged entry needs it to compile again |
| Distributing `.aimodel`s instead of checkpoints (estimate, 16.174, 16.176) | LFM3-1.7B: 3.5 GB to download and about 10 GB on the disk → 1.8 GB and 3.2 GB. Chroma1-Flash: 27.5 → about 14 GB, about 73 → 28 GB | No conversion and no checkpoints. Not implemented (the runner has to read `.aimodel#NAME` and `hwnpu-io-*.json`) |

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
| Apple: lut6 entries chosen by GPTQ (16.87, `hwnpu-dev convert -codes`) | LFM3 lut6: KL (CPU estimate) README 0.070, wiki40b-ja 0.112 | 0.027, 0.068. Through Go's converter 0.074 → 0.050 (top-1 105 → 111 of 128) | Same format and speed. On the M3's ANE (16.95, every projection with `--q8 gptq`): KL from PyTorch's fp32 against the chat's rounding English 0.322 → 0.247 (median 0.155 → 0.097), Japanese 0.054 → 0.035, as on Core ML's CPU; all on the ANE, same speed. 12 factual questions: 10 of 12 either way (GPTQ got Mount Fuji's height wrong). Fitting the tables too, by k-means weighted with E[x²] (`--lut-fit wkmeans`): lut6 0.050 → 0.038 through Go's converter (top-1 112 of 128); lut4 2.7 → 0.66 → 0.33, better but still 4× lut6's rounding. **Gemma 4 E2B on the M1** (16.114, calibrated on the 16 GB Mac with the tables read from the file, 87 min): on the ANE over 128 tokens English 1.02 → 0.56, Japanese 1.16 → 0.94 (median 0.25 → 0.08, 0.47 → 0.20; top-1 88 → 97, 82 → 92 of 128) with the `per_layer` projections the calibration leaves out at lut8 (+0.01 GiB); 12 questions 10 → 11 of 12; same speed. `hwnpu-chat -codes DIR -format-for per_layer=lut8` |
| AMD: the cores' rounding to nearest even (16.78) | The cores start in floor mode, and each narrowing to bf16 pulls sums of inputs with a mean downward: Anima's DiT step-0 velocity error q8 0.088, q4 0.244 (bf16 0.025) | q8 0.031, q4 0.087 (the quantization alone on the CPU: 0.007, 0.092). Language models: E2B KL 0.097 → 0.093, LFM3 0.036 unchanged | None (one instruction a call). Takes effect when exported again |
| AMD: a checkpoint already in 4 bits kept in its format (gpt-oss's MXFP4 experts, mx4, 16.179) | Rounded again to q4nx: weight error 6.6 %, KL README 0.135, wiki-ja 0.107 (CPU) | As they are: 0.052, 0.043 | None (4.3 bits a weight; the e2m1 nibbles are decoded in the kernel) |
| AMD gpt-oss: only the head's q4 by GPTQ (16.179) | Rounding: KL 0.060 (README), 0.0074 (GPL), 0.027 (wiki-ja), 0.027 (wikitext), 0.032 (Go) | 0.055, 0.0065, 0.021, 0.023, 0.030 | The attention's q8 by GPTQ too made four texts better and wiki-ja 0.027 → 0.066 (three positions at KL 1.6–2.9): left rounded |
| AMD: bfp16 where a GEMM takes it, not where a softmax follows (16.179) | gpt-oss's prompt experts on bfp16 (8 in-features sharing an exponent, 8-bit mantissas): KL 0.027 → 0.030, kept for 1.6× the speed | The decode attention's scores on it: the heads' outputs 3.6 % of their largest off on random keys (cosine 0.99987). Kept in bf16 | A score's error is its softmax's; a GEMM's averages out |
| AMD: a shorter exponential on the cores (`gl_exp32`, 16.179) | `exp_f32`, exact in float32 | A quartic on two bf16 limbs, within 1e-4: the same errors against the reference, KL of 5 texts 0.0318 → 0.0315 | None (7 µs an attention object less). A cubic in bf16 alone (0.2 %) put a routing weight 1.3 % off |
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
| Anima's IR (`hwnpu-image convert --backend intel`) | Image generation | On OpenVINO's CPU, latent cosine 0.99985, PSNR 37.9 dB |
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
| Image DiTs: weights packed as bfp16 on the host (`iron_q4.bfp16_pack`, 16.200) | flm.ProjectionGEMM's B in bfp16ebs8 blocks (what AIE2P's matrix instruction multiplies), written at export and read by the GEMM as they are: no dequantize operator (q8's QDequant: 8 ms of an Anima block), no conversion in the GEMM, one rounding to nearest even | An MLP of 4096 rows 25.9 ms (q8) / 23.0 ms (bf16 weights) → 18.5 ms. The tiling is pinned on both sides (tile_n 64, k tiles 512 / 384 / 256 / 128, ct_k 128); the weight is registered as a v8bfp16ebs8 operand under its param's name |
| Image DiTs: a block's parts timed as graphs of their own (16.200) | Anima's 4 blocks as the whole, the GEMMs alone, the attention alone, the elementwise operators alone | 616 = attention 297 + GEMMs 178 (of which q8's dequantize 32) + elementwise 187 (of which the exact GELU about 84) + adaLN 6 ms. Elementwise operators run at the memory's 58 GB/s; the exact GELU computes 36 cycles a value |
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
| hwnpu-dev xclbin and tools/iron (amd-xdna2-bench.md) | Builds IRON designs without XRT's xclbinutil. Measures the data types and DMA | Matrix multiply at 1–4 tenths of the peak (int8 7.3–7.6 TOPS, bf16 2.7–3.0 TFLOPS at 4096³). DMA read 55–57 GB/s, write 48 GB/s, both 64 GB/s. A run costs at least 0.06 ms |
| A checkpoint's own 4 bits kept (mx4, `iron_gptoss_moe.py`, 16.179) | gpt-oss's MXFP4 experts read as they are: e2m1 nibbles decoded in the kernel (sign × max(m, 2m − 4, 4m − 16)), the group's exponent byte as the scale. 4.3 bits a weight, no error | Rounding them again to q4nx added 6.6 % weight error and took KL 2–3× up. q6 also holds them exactly, at 6.5 bits |
| Attention sinks in IRON's MHA (`iron_sinks.py`, 16.179) | gpt-oss's learned logit per head: its term added to the softmax's running sum before the last division, on the P×V cores | No work added, the KV cache as without sinks. Cosine 0.999988 against the formula, sinks of 6 included |
| Fewer operators in the decode step (16.179) | A projection's bias and the block's residual added in its GEMV as the output leaves (`QGEMVAdd`, `kernels/mv_add.h`); a gated MLP's gate and up as one table and one operator (`QGEMVGlu`); q and k one RoPE; a router's bias in MoERoute | Qwen3.5-0.8B 87 → 73 ms, gpt-oss 97.5 → 89.5 ms; one table, no second copy of the weights. Generation is bound by the count of operators (80–150 µs each) |
| A MoE layer's experts in one operator (`MX4Experts`, 16.179) | Gate-up and down as two phases of one operator: one set of core programs, one weight stream, the activations round-tripping through DRAM in between; down's weights queued before the gate-up ends | Each operator carried about 0.4 ms of its streams' start and end: the experts 2.16 → 1.84 ms a layer, with the exponent-byte scales 1.81; gpt-oss 89.5 → 80.5 ms/token |
| A prompt's experts through AIE2P's bf16 matrix unit (`kernels/moe_mx4_rows.cc`, 16.179) | `mac_4x8_8x8` (256 products an instruction against the vector mac's 16): a k-step's codes decoded and scaled once, 4 rows a tile; a strip of 16 out-features at a time, so that only 2 G accumulators are live | 773 → 462 ms a pass with the vector mac; 4 cores a column 241 ms; a strip at a time: 256 tokens of 2 layers 571 → 488 ms |
| A prompt's experts on bfp16, AIE2P's native float matrix instruction (`MX_BFP`, 16.179) | The bf16 `mmul<4,8,8>` is emulated with broadcast macs on AIE2P; `mac_8x8_8x8T` (512 products) takes bfp16ebs8. The weights' tiles built directly from the codes (mantissa v = 2 e2m1, exponent the group's E + 5: exact, an exponent's 8 values one out-feature's 8 in-features), the activations converted (rounded to 8-bit mantissas) | Experts of a 256-token chunk 113.5 → 56.8 ms; gpt-oss's prompt 256 tokens 5.4 → 3.4 s. KL of 5 texts 0.027 → 0.030 |
| Timing an operator's switch (16.179) | The same operator twice in a row (no reload) against once; extra unused code (PDI size) | A switch is about 67 µs plus 1.2 µs a KB of PDI (32 cores' programs and the init); gpt-oss's decode spends about 30 of 80 ms on switches |
| A decode step as the phases of one resident array (`GptOssStep`, 16.179) | Every operator of a gpt-oss layer (experts, q/k/v with the norm and RoPE, attention, o_proj, router, routing) a phase of one core program on one array; a graph step a turn of the cores' loop, its phases runtime words; a full ELF of one array is loaded once (IRON leaves its load unexpanded, the firmware skips it), the output head a phase too | An operator following another is loaded again: about 30 of a token's 79 ms. 79.2 → 54 ms/token (12.6 → 18.5 tok/s), 58 ms at 2048 tokens of context; KL the same; +1.06 GiB (its own tables and KV array) |
| The output head a phase of the resident array, no barrier before a step (16.179) | The final RMSNorm and the q4 GEMV as phase H of `GptOssStep`: the head's own q4 table read in place as 5504 byte windows stepping 5120 (no copy), a row's logit two partial sums hwnpu adds (`logit_parts`); the embedding row and the RoPE angles read without Copy operators. With unexpanded loads the step barrier (an empty command in a second hardware context) put the whole array back each step: the manifest's `no_step_barrier` drops it | The decode ELF is one array: 54 → 50.6 ms/token (18.5 → 19.7 tok/s; the first command 3.4 → 0.82 ms, of which the barrier 2.4), about 55.6 ms at 2048 tokens; KL 0.0310 → 0.0315 (5 texts); memory the same. A token's bytes (2.53 GB) at 55 GB/s are 46 ms |
| One table for the prompt and the decode step (`Q8XDequant`, 16.179) | The attention projections' weights were held twice with the same codes and scales: q8 blocks for the prompt chunk's dequantize and GEMM, q8x blocks for the decode array. The prompt chunk now dequantizes the q8x table (read in the decode step's order, 4 cores a column, a block 32 rows of 160 in-features written into the bf16 weight; the last k-block's surplus to a junk buffer) | gpt-oss-20b 14.1 → 13.4 GiB pinned (0.67 GiB); the prompt's output bit for bit the same; the prompt 1.9 % slower (rows of 320 bytes are written at 15 GB/s), decode the same |
| Timing a command's phases and its boundary (`iron_gptoss_layer.py check --first`, `--one`, 16.179) | The first command with fewer phases; two steps as one command | Q 0.47, A +0.07, O +0.26, R +0.05, T +0.07 ms: Q, O and the experts are their bytes at 55 GB/s. A command boundary is about 0.1 ms; a chain of commands (ERT_CMD_CHAIN) does not remove it, and one command a token cannot read the routed words (mlir-aie reads the scratchpad once a sequence) |
| The decode attention down to its transfers (`gl_exp32`, the values' sums as broadcast macs, 16.179) | gpt-oss's attention cores: the softmax's exponential from `exp_f32` (exact in float32, 16 values, 33 calls for an object of 128 rows) to 2^k p(f) on 32 values a call (a quartic on two bf16 limbs, within 1e-4; the experts' glu and the routing use the same one). The values' sums no longer build 8 by 8 tiles for `mmul`: a token's value (two vectors of 32 as they lie) times its p, by mac | An object about 20 → 9.6 µs (5–6 of them transfer). 58.7 → 52.3 ms a token at a context of 4096, 66.1 → 56.1 ms at 7680 (15.1 → 17.8 tokens/s). The same error against the reference and the same KL |
| Timing a kernel stopped part way (`GL_ATTN=9` and builds for measuring, 16.179) | Build a phase's kernel to return after the transfers alone, after the scores, after the softmax, and compare the command's times | Of an attention object's 20 µs: transfers 6, the scores' matrix products about 0 (done while the next object arrives), the exponentials 7, the values' sums 7. Faster matrix products (bfp16, 4 cores) would have gained nothing |
| The router's weights on every column (`router_table`, 16.179) | gpt-oss's router k-blocks go to the last two cores of every column instead of column 0's; each column's partial sums come back and the routing core adds them | The R phase 46 → 23 µs (440 KB went down one column), about 20 µs a layer |
| A sliding layer's decode cache as a ring (`GPTOSS_KV_RING`, rows.go's `kvJoin.ring`, 16.179) | A sliding layer's attention reads its last 128 rows in any order: the decode step's array is a ring of 128 rows (position p's row is p % 128). hwnpu fills it from the keys and values by position after a prompt chunk and before a step elsewhere, and splits the steps' rows back every 128 steps | Pinned 12.41 → 12.22 GiB (12 layers of 16.5 MiB). Steps after going back, and after rows are exported and imported, are bit for bit the same (a test on the device). The same speed |
| The host's logits and top k in vector loops (16.179) | An IRON decode step's logits (bf16; gpt-oss's two partial sums a row) through `vecops.BF16ToF32` and `f32.Axpy`; top k passes over blocks of 256 whose largest is not above the k-th | gpt-oss (201088 tokens): 1.33 → 0.23 ms a token, 19.4 → 19.7 tokens/s. What is left on the host is about 1 ms a token (by pprof) |
| A prompt's experts by slots, each block decoded once (16.179) | 40 slots of 56 rows: slot e < 32 expert e's first rows, 8 overflow slots for heavy experts at per-call offsets (a command's scratchpad holds 32 words); a weight pair decoded once into bfp16 tiles kept on the core for all its rows; the host writes and reads only the used rows | x and activations streamed CAP rows whatever the count (141 of a column's 198 MB). 256 tokens 3.4 → 1.8 s, values the same |
| gpt-oss's decode q, k, v in one operator (`QGEMVQKV`, 16.179) | The input RMSNorm done by each core on its x (the weight on x's channel), the bias, and RoPE on q's and k's rows as they leave (their rows in interleaved pairs: a q8 scale is a row's) | Two operators fewer a layer: 80.4 → 79.2 ms. Folding the norm's weight into the table's columns took the q8 error up (top-1 changed) |
| Telling transfers from compute (`MX4_NOCOMPUTE=1`, 16.179) | Build the operator's kernels with the products skipped and time the transfers alone | gpt-oss's decode experts were DMA-bound (2.05 of 2.16 ms), its prompt experts compute-bound (38 of 237 ms). The disassembly (`llvm-objdump`) then showed where: the codes' decoding, 5000 of a call's 5500 cycles |
| A hidden size off the 8 columns padded (16.179) | gpt-oss's 2880 (45 blocks of 64) carried as 3072 with zeros; a norm's weight times sqrt(2880 / 3072), its eps times 2880 / 3072 | The GEMVs take all 8 columns and in-features in multiples of 512, with the same outputs |

### 7.4 Common (any NPU, on the host)

Prompt reuse and the prompt store, speculative decoding (Apple / Intel, AMD's Qwen3.5), the think budget, repetition detection, hwnpu-agent's guards (no editing a file it has not read, stopping repeated calls, refusing a `cd` outside), and the search tools.

Sharing identical files on the disk (`cache.Clone`, 16.64 and 16.79) depends on the file system, not the NPU: clonefile on macOS's APFS, FICLONE and FICLONERANGE on Linux's Btrfs, XFS and others, block cloning on Windows' ReFS and Dev Drive. It covers Apple's programs and compiled copies, Intel's IR (`.bin` of the programs for 8 and 64 tokens) and AMD's exports. Nothing happens on ext4, tmpfs or NTFS. Only macOS cannot share the start of a file.

The host's own CPU code (16.183): the matrix product (`internal/sgemm`) has a NEON kernel on arm64 beside amd64's AVX2 and AVX-512 ones (106 GFLOPS on one thread of an M3, where the Go loop took about 10 times as long in EfficientSAM's decoder), which the CPU interpreter (`milcpu`, `--cpu`) and the ONNX evaluator share; and the evaluator's elementwise operations no longer build tables of indices for the common broadcasts. EfficientSAM's decoder, which runs on the host for each prompt, went from 1.06 s to 0.28 s (`hwnpu-image segment`: 1.5 → 0.72 s a picture).

The host's share of a token (16.179): with gpt-oss on AMD pprof showed 2.3 ms a token on the host, 1.33 ms of it turning 201088 bf16 logits to float32 one at a time. The logits now go through `vecops.BF16ToF32` and `f32.Axpy` (0.23 ms), and the sampling's top k (every backend's) passes over blocks of 256 logits whose largest (`vecops.MinMax`) is not above the k-th. About 1 ms is left: the commands' submits and waits, the logits' copy, the token's text.

### 7.5 Apple: Core AI (macOS 27)

Core AI keeps what it specialized for the device in its own cache, keyed only by `main.hash` (the sha256 of `main.mlirb`), for each OS build and executable name. It has no ~4 GiB limit like Core ML's compile cache, and one program may hold 2.5–3 GiB on an M3 (Core ML: 1 GiB).

| Change | What it helps | Numbers |
|---|---|---|
| Lowering MIL to Core AI's bytecode in Go (`internal/coreai/mlirbc`, `ailower`, 16.169) | No Python, no Xcode toolchain; the same `.mlpackage` as Core ML | Written to match coreai-core's output, read back with `tools/coreai/bcdump` |
| What Core AI needs to keep a program on the Neural Engine (16.169) | Placement | Core AI's own debug locations (also on block arguments); reductions as the composite `reduce_*_v1` around `primitive.reduce_v1`; matmul as the composite `broadcasting_batch_matmul_v1` (bare ops with int8 weights move the whole program off); softmax as `exp(x − max) × reciprocal(sum)`; binary ops on equal shapes, widened with `broadcast_in_dims` |
| The Swift shim loaded with purego (`libcaishim.dylib`, go:embed) | `CGO_ENABLED=0` stays | Placement read from the MPSGraph manifest's `mps.fullyPlacedOnANE` |
| Input arrays made once per program (16.172) | Generation | A new `NDArray` each call 10.0 ms, the same one 7.2 ms (PLaMo 3 0.8B's part1) |
| A broken cache entry deleted and loaded again (16.169) | Loads that pass yet fail in `loadFunction` (`_GenericObjCError 0`) | Once per entry |
| `aicheck run -cpu` / `-coreml` / `-calls`, `probe`, `HWNPU_AILOWER_CUT`, `HWNPU_AILOWER_TRACE` (16.169, 16.172) | Finding the op that leaves the Neural Engine before Core AI aborts the process | Bisects by the first N ops |
| Functions in one `.aimodel` (`ailower.LowerFunctions`, `PROGRAM.aimodel#NAME`, 16.174) and found automatically (16.177) | Disk and the Neural Engine's memory | Records `KEY.aifn` per package, so later loads hash only their own package; the function's name is from its digest (the same on every machine). A third size makes a larger group, which replaces the smaller one |
| An empty `.aimodel` opened only with its entry there (`hasEntry`, `kindNoRetry`, 16.176) | Core AI parsing an empty `main.mlirb` aborts the process (LLVM: `cannot unwrap empty odiec_module_t`) | `main-this.odix` must be a non-empty file; otherwise lower again. The shim does not delete and reload a cut asset; Go lowers again instead |
| `max_pool`, `conv_transpose` (cropped to MIL's pads), `pad` replicate / reflect, a division by a constant as a product (16.182) | The image models' ONNX graphs and depth heads | PP-OCR's detector at cosine 0.05 on Core AI alone: `divide_v1` by a constant came out multiplied (about c x plus an offset a channel) on the Neural Engine. `HWNPU_AILOWER_CUT` now compares by cosine with milcpu's value of the cut op |
| `HWNPU_COREAI_UNITS=cpu`, `HWNPU_AILOWER_DIVIDE=1`, `aicheck` taking the Neural Engine's lock, `elu` (16.183) | Telling a lowering's error from the Neural Engine's, without the Neural Engine | The division's error is the Neural Engine's alone (Core AI's CPU and GPU agree with the product). Core AI's CPU overflows a `reduce_mean` whose sum passes fp16 (57600 values of mean −1.37): the sum goes back to fp16 before the 1 / n. Magenta RealTime 2's decoder stopped at `elu` |

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
| AMD gpt-oss: the experts re-rounded to q4nx | Weight error 6.6 %, KL 2–3× (README 0.052 → 0.135). Kept as MXFP4 (mx4) | 16.179 |
| AMD gpt-oss: GPTQ on the attention's q8 too | Four texts better, wiki-ja 0.027 → 0.066 (three positions at KL 1.6–2.9). The head alone by GPTQ: all five better | 16.179 |
| AMD gpt-oss: sinks seeded into the softmax's running maximum | Every score's exponent far below zero, where the device's exp2 is up to twice too high: outputs 1.5× at a sink of 6. Added at the division instead | 16.179 |
| AMD gpt-oss: decode experts on 2 cores a column | Transfers alone faster (1.78 against 2.05 ms), but compute-bound with the products (2.39 ms). 4 cores kept | 16.179 |
| AMD gpt-oss: deeper weight buffers in the memtile (3, 4) | Ran out of the channel's 24 descriptors in aiecc | 16.179 |
| AMD gpt-oss: prompt experts at 64 rows a pass, 16 an object | Fewer decodes a row, but 3 passes where 128 rows take 1–2: 256 tokens of 2 layers 571 → 714 ms. 128 and 8 with a strip at a time: 488 ms | 16.179 |
| AMD gpt-oss: smaller core programs for the decode experts (glu once, accumulators looped) | 7.8 → 7.6 KB a core, 1.78 → 1.80 ms: IRON unrolls a loop by its fifos' depths | 16.179 |
| AMD gpt-oss: the decode attention's scores on the bfp16 matrix instruction | As fast as the bf16 mmul (both end within the transfers). On random keys the heads' outputs are off by 3.6 % of their largest (cosine 0.99987, bf16 1.000000): 8 values sharing an exponent with 8-bit mantissas, spread by the softmax | 16.179 |
| AMD gpt-oss: the exponential as a cubic in bf16 alone | Its 0.2 % put a routing weight 1.3 % off. A quartic on two limbs (within 1e-4) instead | 16.179 |
| AMD gpt-oss: the cache's size as an array parameter differing by layer | Sliding and full layers became two arrays, loaded again each command (a layer's command 1.80 → 4.27 ms); the check still passes. A parameter outside the array's key | 16.179 |
| AMD gpt-oss: the attention's cache read in one run (to time the transfers) | The command timed out (ERT_CMD_STATE_TIMEOUT). Keys and values stay read 64 elements each in turn (5–6 µs an object) | 16.179 |
| AMD gpt-oss: one command a token | The routing writes the scratchpad words the next layer's experts read; mlir-aie reads a sequence's scratchpad once, at its start: cosine 0.877 from the second step. A command boundary a layer stays (0.1 ms each) | 16.179 |
| AMD gpt-oss: a token's 25 commands as one chain (ERT_CMD_CHAIN) | Linux 7.0's amdxdna takes a chain of ELF commands, and it ran: 19.3–19.6 tokens/s against 19.6–19.7 one by one. The boundary's 0.1 ms is the firmware starting a command, not the round trip | 16.179 |
| AMD image DiTs: IRON's tanh GELU | 4× as fast as the exact one, but its argument from bf16 products (x² and s 0.044715 x) changed a test prompt's clothes. The same table with the argument in float32 (`GELUEven`) keeps them | 16.200 |
| AMD image DiTs: GELU as x sigmoid(2u) from gpt-oss's exponential and the core's float inverse | Exact to 2e-4, but as slow as IRON's exact GELU (5.5 against 5.2 ms at 4096 × 2048: the exponential 2.5, the inverse 1.25) | 16.200 |
| AMD image DiTs: the gated residual x + y g as one operator | Three inputs (x, y and the gate's row) do not fit a core's two input DMA channels. Two pipelined cores would; about 1 % of a block | 16.200 |
| AMD gpt-oss: the decode attention on 4 cores a column | Not needed: once the exponential was short the computing ended within the transfers (an object 9.6 µs, 5–6 of them transfer). It would have needed room in the cores that hold the router | 16.179 |
| AMD gpt-oss: the attention's outputs written in one run instead of 64-byte pieces | The same time: a phase's fixed 25–35 µs is its x objects and descriptors, not the shape of its writes | 16.179 |
| AMD gpt-oss: the input norm's weight folded into the q8 table's columns | The q8 error went up and a top token changed. The cores multiply it on their x | 16.179 |
| AMD gpt-oss: the attention's projections at q6 | KL 2–3× on the CPU (README 0.133). q8 kept; GPTQ at q6 is not tried | 16.179 |
| AMD gpt-oss: the prompt experts' x resent from the memtile, their outputs in objects of 8 rows | 24.7 → 26.1 ms and 24.66 ms a gate-up: the weights' stream and decoding, x's transfer and the products are each about a third | 16.179 |
| Core AI: `.aimodelc` (`coreai-build compile`) | First load 5–10× slower than `.aimodel` (26–77 s for 0.5 GiB on an M3), weights still copied into Core AI's cache, and kept per architecture | 16.169 |
| Core AI: siblings by the whole weight file | PLaMo-Embedding's 128 and 1024 differ by a 458 KB table and did not group; the 3 largest blobs are compared instead | 16.177 |
| Chroma1-Flash: 512² as a preview of 1024² | With the 1024² noise averaged down to 512², the composition matched in about 1 of 3 seeds; starting 1024² from the 512² latent stretched (hires fix) showed a grid of the token spacing. A preview was not needed | 16.176 |
| Chroma's DiT at lut6 or int8 | lut6: the same picture but not faster at 1024² (saves 1.7 GB of disk). int8: stripes (PSNR 23.7 dB), the Neural Engine's int8 sums overflow fp16 | 16.170 |

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
- **gpt-oss's export and hwnpu go together** (16.179): the resident decode step needs what hwnpu learned with it (a row's logits as two partial sums, no barrier before a step, the sliding layers' rings). An export made after #148 is refused by an older hwnpu ("kv is no rows of its keys and values"); an older export still runs. `GPTOSS_RESIDENT=0` and `GPTOSS_KV_RING=0` export the older forms
- **No barrier before a gpt-oss step**: the empty command in a second hardware context that keeps Qwen3.6 from hanging (16.92) is skipped for this export alone (the manifest's `no_step_barrier`): with it every step loads the array again (2.4 ms). Run without it: 399 steps on 2 layers, the KL runs, the agent's task, chats. `HWNPU_IRON_BARRIER=1` puts it back
- **A decode step's own keys and values**: hwnpu copies a prompt chunk's rows into the step's array and the steps' rows back (before the next chunk, an export of rows, and every 128 steps for the rings). A step at a position that does not follow the last one fills the rings again first
- **Timing a decode step on AMD**: `HWNPU_IRON_TIMING=1` waits for each command and adds its round trip (gpt-oss: the commands sum to 50.7 ms where a step takes 49.0). A forced run's whole time carries the prompt's variance (±0.3 s, ±1 ms a token over 256 steps): read the steps' own time, which `ironlm -forced` prints. After any change to an export read the commands' times too: an array parameter that differs by layer loads the array again each command and still passes the check
- **PLaMo 3 as one lut8 program**: short texts are fast, but long documents are read about 10× slower (`-format lut6` gives two programs at lower accuracy)
- **Sharing on the disk depends on the file system**: ext4, the default of Ubuntu, Debian and many others, cannot share, so each copy takes its full size there. Fedora and openSUSE default to Btrfs. Put the models on Btrfs or XFS to share (on Windows, a Dev Drive). Whole files and the common start of files of one name are shared; bytes after a difference are kept separately
- **Core AI's first run**: every part is compiled once, and a group of several functions compiles all of them. Chroma1-Flash took 12 min 45 s at 512², 59 min 44 s at 1024², and 1 h 14 min for the 512² that joined the 1024² (after which 1024² loaded in about 1 s a part)
- **Core AI's cache is per OS build and executable name**: after an OS update, or from `aicheck` or `go test`, programs compile again from `main.mlirb`. macOS removes entries when the disk runs low (`storagePressure`, seen at 11–20 GB free), so the next start compiles again
- **Converting needs room**: converting Chroma1-Flash's 512² beside the 1024² export with its checkpoints took the M2 down to 16 GB free, just above where macOS purges `~/Library/Caches`

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
| Qwen3-0.6B | `qwen3` (16.127) | Qwen3's pre-tokenizer (the `tokenizer.json` Anima reads) | Qwen3 | — | ○ lut8, one program not split at its attention (context 4096, `llm.Unpaged`), all on the ANE, 9.5–12.4 tokens/s on an M1, 1.16 GiB on the Neural Engine + 2.75 GB of the process. KL 0.014 for lut8 on the CPU | Converts (int8), not run on the NPU | — (no export) |
| Qwen3-1.7B | `qwen3` (16.150) | Same | Qwen3 | — | ○ lut8, one program not split at its attention (context 4096), all on the ANE, 13–22 tokens/s on an M3; below 16 GB one program (`prefill: off`, `variants`), 1.63 GiB on the Neural Engine + 2.2–2.8 GB of the process at its peak; from 16 GB two, 3.26 GiB + 3.0–4.1 GB. KL 0.018 for lut8 on the CPU (32/32) | Converts (int8), not run on the NPU | — (no export) |
| plamo-3-translate-2b | `plamo3` | PLaMo (`tokenizer.jsonl`) | PLaMo 3 (PFML3, translation) | — | ○ lut8, one program, context 262144. Checked on an M3 that all of it is on the ANE, and the weight formats | Converts (int8). Translates on OpenVINO's CPU | ◎ q4 11.6 tok/s, 2.16 GiB, context 8192 |
| plamo-3-translate-8b | `plamo3` (16.164) | Same | Same | — | ○ lut4 (the 8B only, `Config.Format`), one program of 5 parts, all on the ANE, 3.81 GiB + the process's 4.1 GB at its peak; 0.8 tokens/s on an M1 (over its 2.8 GiB). 13/20 sentences word for word as lut8, the rest rewordings | int4 (by rows, the 8B only, `Config.IntelFormat`), one program (prefill off), 3.8 GiB; int8 (7.6 GiB) is over what a Meteor Lake NPU held. Not run, precision not checked | ? `iron_export_lm.py plamo3 --q4` not tried (docs/handoff-plamo3-8b.md) |
| plamo-embedding-1b | `plamo-bi` | SentencePiece BPE (`tokenizer.model`) | — (embedding API) | — | ◎ lut8 + head-scale | Converts (int8) | — |
| BERT family (all-MiniLM-L6-v2 and so on) | `bert` | **Not implemented** (WordPiece) | — | — | Converts only (lut8) | Converts only (int8) | — |
| Anima-Turbo (image generation) | `internal/imagegen/anima` (built from one MIL description) | Qwen3 BPE + T5 Unigram | — (`hwnpu-image`) | Images out | ◎ M3: 1024×1024, 8 steps in about 40 s, 3.8 GB (int8) | △ Converts. Checked on OpenVINO's CPU | ◎ 1024×1024, 8 steps in about 68 s, 3.4 GiB pinned (DiT at 8 bits) |
| gpt-oss-20b | `gptoss` (IRON only) | o200k_harmony, byte-level BPE (`tokenizer.json`, o200k's pattern) | harmony (`GptOss`: analysis, final and commentary channels, tools as a TypeScript namespace) | — | — (MoE routing does not fit the Neural Engine) | — | ○ `iron_export_lm.py gptoss`. 24 layers (13.1 GB pinned): decode 20.2 tok/s (the step one resident array with the head in it, 17.8 at a context of 7680; 12.4 with the operators apart, 10.3 at first), prompts in chunks of 256 at 7 ms a token (routed on the host, the experts on bfp16), experts in the checkpoint's MXFP4 as they are (mx4), attention sinks on the device (16.179). KL against HF fp32 0.0065–0.055 (the head by GPTQ, `gptoss_gptq.py`), every top token of the check agrees; 1.3 J a token at the package |
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
| Qwen3 (`chat.Qwen3`) | 9 conversations (`qwen3_template.json`, `--text-content`) | Hermes's: JSON in `<tools>`, calls as `<tool_call>{"name": …, "arguments": {…}}</tool_call>` | Kept only in replies after the last question. With thinking, the prompt ends in `<think>\n` (the template leaves it to the model) |
| System One (`internal/systemone`) | 7 cases (llama.cpp's template rendered with jinja2) | d1's choice / score / noul questions | `<think></think>` |
| PLaMo 3 / PFML3 (`chat.Plamo3`) | 8 cases (`tools/ref/plamo3_template_golden.py`) | No tools. `target_lang`, `source_lang`, `form`, `context`, `glossary` (`Options.Translation`; in the API, `chat_template_kwargs` as in vLLM) | None. A system message becomes `context`. Without a target language, a text with Japanese goes to English and others to Japanese. The label of each past translation is chosen from its source text, so the start of the prompt can be reused |

In every format, the think budget (`-think-budget`) feeds in the closing mark the format gives (`</think>`, `<channel|>`) to move on to the answer.

### 10.4 Notes for each model

**LFM3-1.7B** (16.8, 16.13–17, 16.39–41, 16.59, 16.61, 16.66)
- Structure: short convolutions (split into `taps` to avoid the error of the ANE's conv), 5 sliding layers with a window of 256 (rings), 6 full-attention layers, per-head gates, sandwich norms. `-post-scale` keeps the ANE's arithmetic error to 1.4–2.7% a layer
- Checked against an independent fp64 reference (`reference_test.go`, 16.27) and with `hwnpu-dev check`. KL 0.232 with lut6
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
- Checked: relative error 6.5e-5 against an fp64 reference. `hwnpu-dev check` argmax 31/31 (fp32)
- Weight formats (M3): with lut8 the next token matches fp16 98 % of the time, with lut6 89 %; lut4 is unusable (KL 1.91)
- Default: one lut8 program (`-prefill auto`)

AMD (`tools/amd/iron_plamo3.py`; written from IRON's building blocks alone):
- Checked: top-1 17/17 against HF fp32, 9/9 also over 2596 tokens
- bf16: 7.7 tok/s, 1024-token chunks in 2.2–10.5 s
- `--q4` (16.76): 2.16 GiB, 11.6 tok/s. KL on the translations 0.009 or below
- With random weights the NPU and `OnHost` disagree (as with Gemma 4), so checks use the real weights

hwnpu-chat:
- Translates each text typed and keeps the earlier pairs in the prompt as history
- `--to`, `--from`, `--form` and `/to LANG` set the languages and style. The server also takes a `glossary`
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

Apple / Intel (`hwnpu-image convert`):
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

**gpt-oss-20b** (16.179)
- AMD only (IRON; the routing does not fit the Neural Engine). 24 layers, 32 experts of which 4 a token, sliding (128 positions) and full attention in turn with a learned sink a head, a hidden size of 2880 carried as 3072. It always reasons first (the analysis channel of harmony)
- Formats: the experts in the checkpoint's MXFP4 as they are (mx4), the attention's projections q8 in one table for the prompt and the decode step, the head q4, the embeddings a file on the host. The head's GPTQ codes come from `gptoss_gptq.py` and go in with `--q4-codes`; models.yaml's `amd_export` rounds it
- Decode is one resident array of the 32 cores, a command a layer (the routing writes the next command's scratchpad words): 49.4 ms a token (20.2 tokens/s), 56.1 ms at a context of 7680. A prompt chunk of 256 tokens 1.8 s (routed on the host). Pinned 12.22 GiB. 0.92 J a token at the package
- Checks: against HF fp32 every top token and cosine 0.99711–0.99984 (`ironlm` on 32 tokens and 3 steps), KL of 5 texts 0.0093–0.070 (mean 0.0315; `ironlm -forced -bos 200006`, then `ref_gptoss.py forced`), hwnpu-chat agent's task, and going back on the device (`TestKVRingRewindOnDevice`)
- Switches for measuring: `GPTOSS_RESIDENT=0` (operators apart), `GPTOSS_KV_RING=0`, `GPTOSS_ATTN=0` (the attention and exponential before) or `9` (transfers alone), `MX4_BFP=0` (the prompt's experts on bf16), `HWNPU_IRON_BARRIER`, `HWNPU_IRON_TIMING`; `iron_gptoss_layer.py check --first`, `--one`, `--ring`
- Left: fewer bytes (the attention below q8 needs GPTQ; the head's windows overlap by 7 %), the full layers' second copy of keys and values (0.19 GiB), checking drafts (each row has its own experts)

**PLaMo-Embedding-1B** (16.18–19, 16.23, 16.34)
- Structure: PLaMo's decoder with bidirectional attention. The fused `qkv_proj` and `gate_up_proj` are split by row ranges (8192 at a time, because of the 16384 limit)
- hwnpu-chat's embedding API (`/v1/embeddings`): the query prefix, mean pooling over the text's tokens only, `input_type`, `truncate`, and the shortest of the programs of different lengths (128 / 1024) that fits
- The last layer's MLP exceeds fp16 (5e4), so head-scale brings it to 1/64. Cosine similarity 0.997–0.999 against HF

**BERT family** (16.11)
- The structure is ported and converts (lut8 / int8), but without a WordPiece tokenizer it cannot be used from `hwnpu-chat`

**Chroma1-Flash** (16.156, 16.170, 16.176, 16.178)
- Apple (Core ML, M1): T5 in groups of 3 layers at lut8 with `wo` in fp16, the DiT at lut8, the VAE in 10 stages. fp16 pitfalls: an integer division of untyped constants (`1/64 = 0`), subnormals after dividing by the row's maximum, scales folded into weights leaving products below fp16's normal range, int8 sums overflowing on the Neural Engine (lut8 does not), and Core ML folding constant factors back into weights. 512²: 2 min 06 s an image, 1024² about 9 min 30 s
- Core AI (M2): all 29 parts on the Neural Engine without changing the lowering. 512² 1 min 36–48 s, 1024² 6 min 58 s an image. The 512² and 1024² programs of the DiT, the approximator and the larger VAE stages share one `.aimodel` each (16.178)
- AMD: 512² 1 min 42 s, 1024² 4 min 13 s (Heun 8 steps) (16.156)

## 11. Not yet verified

Collected from what the records call "not yet checked", "not measured" or "estimated". The items are split into "implemented, waiting for a check on hardware" and "cause unknown", with how to check them where there is a way. Features not yet written (T8's activation-aware quantization, changing the split at run time and so on) are listed briefly in 11.5.

### 11.1 Apple (Neural Engine)

| Item | Unknown | How to check / what is needed | Section |
|---|---|---|---|
| Qwen3.5-4B's prompt program, and the first compile of GDN programs | 0.8B and 2B (lut6) load two programs by default (Q=8 and Q=128 with `-last`, 16.99, 16.115). 4B keeps one (two at lut8 take 7.9 GiB) and reads long prompts at 66 tokens/s (13K). The Q=128 program with GDN layers compiles for 13–15 minutes the first time. Would Q=64 (or a smaller GDN chunk) compile faster, and how fast would 4B read with two programs at lut6? | `-prefill on`, compile times at Q=32 / 64 / 128 | 16.62, 16.88, 16.99, 16.115 |
| `blocks.SiLU` in the towers and multimodal models | LFM3-VL / Audio / Omni / d1 with their vision and audio towers, TTS take the same rewrite and are converted again, but only LFM3-1.7B, PLaMo 3, PLaMo-Embedding and Anima were measured | Compare VL's image descriptions and TTS with the earlier version | 16.95 |
| Qwen3.5 with two-level attention | Do answers hold with two-level attention (`-sparse`)? | Passkey test | 16.62 |
| Two-level attention's `Refresh` as the default | With 4 or 16 the passkeys hold and a call is 14–32% faster (checked 2026-11-06), but finer differences such as loss or KL are not measured on the Neural Engine | Compare the logits with choosing every call on real weights | 16.61 |
| Where two-level attention starts in the prompt | At 64K reading in two levels is a tenth slower than the whole cache. Should the threshold of about 57K go up? | Compare the reading times from 64K to 96K | 16.61 |
| The copies in two-level attention's `choose` | Copying the buckets that change at each choice is 12% (13 ms) of a call generating at 128K. Would a choice that changes fewer buckets cut it? | pprof (`decode.(*Session).choose`) | 16.61 |
| Gemma 4's soft cap on the host (16.113) | Checked on the M1. The speed on the M3 and M2, the answers with temperature on the real model, Intel's IR | Compare the output layer's part with `HWNPU_PROFILE=1` on the same question | 16.113 |
| Slowing down past 128K | Why the slope per 1K grows from 1.14 to 1.43 ms (memory pressure, or attention lagging when interleaved?) | | 16.59 |
| Slowness right after a long prefill | Why the first few calls after reading 256K take 1.29 s each (350 ms after) | | 16.59 |
| The ANE's memory (the rest) | The weights' share is now read by `mem.NPUWired` (16.95, no root: 2.43 GiB for LFM3's two lut6 programs). The KV's and intermediate values' share, and the compile cache's, are not told apart | Compare after loading and while generating with a long context | 16.95, model-fit.md T10 |
| The ANE's compile cache | Its size (about 4 GiB is an estimate; `-prefill auto` assumes 3.5 GiB, since Qwen3.5-2B's two programs of 3.56 GiB were compiled again at every start, 16.99) and location | Looking into aned (sudo) | 16.41, 16.99 |
| Limits on the M4 and later | The M1 to the M3 are measured: 0.99 GiB a package. Only the M1 slows down 12× once the Neural Engine holds over 2.86 GiB of parts run in turn (16.109); the M2 runs 3.62 GiB at full speed (PROBE.md, 2026-11-09). An M4 ran 3.62 GiB in total, and parts of 1.8 GiB and more, at full speed. Whether a larger default per part (0.9 GiB now) pays on the M4 | `hwnpu-dev probe limit` and `hwnpu-dev probe split` on an M4 | 16.12, 16.109 |
| Two programs on the M1 | Since 16.118, `-prefill auto` on an M1 keeps one program when the two with the programs loaded beside them (MTP, encoders) would hold over 2.8 GiB (Qwen3.5-2B lut6: 2.64 + 0.40 GiB → one). Not measured: how much faster LFM3-VL / Audio / Omni (3.26 GiB for two, plus the encoder) generate on the M1 with one, against how much slower they read long prompts | `HWNPU_PROFILE=1` with `-prefill auto` and `-prefill on` on the M1 | 16.109, 16.118 |
| Gemma 4 E4B's default `-prefill` | With `-kv-only` (16.112) the two programs take 4.88 GiB (6.51 before), still over 3.5 GiB, so `auto` keeps one. `-prefill on` has not been measured since. Before, two programs read long prompts 7.5–8× faster, but every start took 135–165 s (one program: 86 s) and generation at 16K was a fifth slower. Switch by the prompt's length, or let `auto` weigh the length? | Start and round trips with `-prefill on` after `-kv-only` | 16.65, 16.83, 16.112 |
| Gemma 4 E4B against the unsplit form | Skipped for lack of disk | | 16.65 |
| Gemma 4 and the prompt store | Save and restore time and size with Gemma 4 E2B | Replay the same OpenCode request as 16.38 | 16.38 |
| `hwnpu-chat agent` on Apple with Gemma 4 / Qwen3.5 | The templates support tools, but the agent tasks were not run (Gemma 4 E4B passed on AMD) | The tasks in CHAT.md section 7 | 16.35 |
| Qwen3.5-0.8B / 2B | 2B against HF (0.8B passes with the same code), 2B lut6's KL against HF's bf16 (the 2B in fp16 does not fit a part, so so far only against lut8), passkeys over long contexts | `hwnpu-dev check`; the KL with the host's or another split | 16.99, 16.115 |
| The rest of speculative decoding | Prompt lookup: acceptance at Q=4, in OpenCode's round trips (7.4K, long XML), against temperature 0. MTP drafts (16.109): 4B on the M3, the time to the first draft after a long prompt (up to 64 calls) | Replays with `-dump` | 16.36, 16.109 |
| d1-Omni-2B differing from its README | The team answer is the opposite of the README's. HF's code reads up to 16 questions together, hwnpu one at a time. Not compared with HF | Compare with HF's code where torch is available | 16.31 |
| LFM3-Audio's TTS | Does it sound the same as the reference (llama.cpp)? Only checked by transcribing it again with ASR | Compare audio decoded from the same codes with llama.cpp's server TTS | 16.30–31 |
| Long texts with PLaMo-Embedding | Do the last layer's values grow further in the 1024-token program (is head-scale enough)? | Vary the texts in `TestANEPrecision` | 16.34 |
| BERT family | `hwnpu-dev check` on real weights and placement on the ANE (only checked on the CPU for Intel) | | 16.11 |
| GPTQ's lut6 tables in the answers | On the ANE the KL falls as on the CPU, all on the ANE, same speed (16.95). Whether the answers get better is not told by 12 questions (10/12 either way). lut4 + wkmeans not tried on the ANE. Gemma 4 E2B (16.114): KL on the ANE 1.02 → 0.56 (English), 12 questions 10 → 11 of 12; hwnpu-chat takes them with `--codes`, not by default (2.1 GB of codes to hand out) | More questions and hwnpu-agent's tasks with and without the codes; a way to hand out the codes | 16.87, 16.95, 16.114 |
| Anima's remaining error against diffusers' fp32 | Latent cosine 0.67–0.98 over 3 prompts after the DiT's silu; the DiT in fp16 brings it only 0.006–0.025 closer (16.95). How much of the rest is the text's int8 is not measured | Swap the text encoder to fp16 | 16.85, 16.95 |
| Anima above 1024×1024 | The smaller attention chunks for sizes past 1024 (`mil.DitQChunk`) are not measured, nor whether the VAE stages fit | Convert 1280×1280 and run | 16.85 |
| Details of PLaMo 3's accuracy and speed | Placement and weight formats are checked. Translation quality and time for long documents are not measured in detail | Compare KL with fp16 on translation pairs | 16.71 |
| Why macOS memory seemed to shrink with use | The guess is that macOS compressed the rings, which are mostly zeros | Measure the footprint on an M3 after the mark fix (`ringRows`) | 16.71 |
| Core AI: distributing `.aimodel`s | The runner reading `PATH.aimodel#NAME` and output shapes from `hwnpu-io-*.json` instead of the `.mlpackage`; whether one `.aimodel` loads across OS versions; whether to use `.persistent` (no purges, but hwnpu would delete its entries) | Download one, run it on another machine and OS build | 16.174, 16.177 |
| Core AI on other models and chips | Chroma1-HD, Qwen-Image's grouping of 512² and 1024²; the M1 and the M3 for Qwen3.5-4B and Gemma 4 E4B (16.191); the M3 for chat models; Chroma's warm Core ML run on the M2 (for a like-for-like comparison) | Run with `--backend coreai` | 16.173, 16.176, 16.177 |

### 11.2 Intel (NPU)

**Nothing has been run on a real NPU.** Only accuracy on OpenVINO's CPU has been checked.

| Item | Unknown | Section |
|---|---|---|
| Does the compiler accept it? | Whether the driver's compiler accepts LFM3-1.7B (int8, `-seq 8` and `64`) and the split form (attention graphs over chunks of 8192), and how long it compiles | intel-npu-models.md section 11 |
| Argument names | Whether the names Level Zero returns map to the IR's names under `argNames`'s rules (if not, it stops with `chain: part 0 has no output y`) | Same |
| Speed of the weight formats | The speed difference between int8 and int4-g128 (if int4-g128 is clearly faster, it is a candidate default). Measure with `hwnpu-dev formats` | Same |
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
| gpt-oss without the step barrier | Long sessions (hours, `serve`) are not run: 399 steps on 2 layers, KL runs of 5 × 144 steps, the agent's task and answers of 600 tokens were | 16.179 |
| gpt-oss with `-idle` | Release and reload are not tried since the decode step became a resident array with rings | 16.179 |
| gpt-oss's attention below 8 bits | q6 by rounding was 2–3× the KL on the CPU; GPTQ at q6 is not tried. The projections are 36 % of a layer's bytes | 16.179 |
| gpt-oss's energy by part | Only the package's total is measured (0.92 J a token); the NPU's own share (`gpu_metrics`) is not read for it | 16.179 |
| Drafts on Qwen3.5-2B and 4B | Checking drafts in one call (16.117) is measured only on 0.8B (82 → 62–72 ms/token). 2B (q6) not measured; 4B is q4 and waits for a GEMV of a few rows at q4 | 16.117 |
| Pipelines of Qwen3.6's prompt attention | The exports' default of 8 (16.107) is not measured on Qwen3.6, which keeps 1 | 16.107 |
| Speed of q6's GEMV | About q8's time, bound by taking out the high 2 bits. Another layout might be faster | 16.105 |
| `iron_moep.py check` from 64 tokens | aiecc fails in `AIEDMATasksToNPUPass` (main as well). The prompt experts were checked against HF through Go instead | 16.116 |
| `iron_q4.py check --valid` | Bounding the prompt path's rows fails the check for q8 as well (unbounded it passes). Thought to be the check's, not fixed | 16.105 |

### 11.4 Behaviour with unknown causes

| Behaviour | Current guess | Section |
|---|---|---|
| Which shapes fall to the CPU at the 16384 per-dimension limit and which stay (chunks along an axis, weighting by ratios go to the CPU) | Inside Core ML's compiler | 16.21 |
| Only the 8-part split loads slowly from the cache (23.7 s) | Not looked into | 16.13 |
| AMD gpt-oss: the attention's cache read as one run an object (a descriptor of [objects, 16, 1024] in place of [objects, 128, 2, 64]) timed the command out | The descriptor's shape with a per-call repeat count; not looked into and not run again (two hangs can wedge the device) | 16.179 |
| On a 128K English document, two-level attention has lower loss than normal attention | Normal attention gets distracted by distant tokens, and two levels narrow that | 16.61 |
| AMD: with the windowed attention, a full ELF's second prompt chunk hangs in its first command | Not the words' values, the cut, the bank offsets or the operators (each passes alone and as an xclbin per step). A pause does not help; another hardware context's command does. Avoided with `chunkBarrier` | 16.80, 16.84 |
| AMD: splitting Anima's DiT into commands with host steps stops the first command with `ERT_CMD_STATE_TIMEOUT` | Not looked into. Avoided with one graph of 4 blocks | 16.68 |
| AMD: with `--command-per-step`, Qwen3.6's 2 layers give logits at cosine 0.32 against HF on a chunk of 256 (the normal export 0.98) | Not confirmed. Perhaps the per-step export routes before the router's logits are written (it picks experts 0–7). Use it for timing only | 16.116 |
| AMD: a single op differs by 1.4× between the first and second run | Thought to be the power state | 16.75 |

### 11.5 Not yet written (for reference)

- AWQ-style scales for lut4 (model-fit.md T8, 2). GPTQ choosing each weight's entry and k-means weighted with E[x²] fitting the tables are in since 16.87 (lut6 0.074 → 0.038 through Go's converter; lut4 2.7 → 0.33, still 4× lut6's rounding)
- hwnpu-chat fetching GPTQ's codes and using them by default: it takes them with `--codes` (16.114, a `-gptq-HASH` in the package name); how to hand out the codes (Gemma 4 E2B 2.1 GB, 1 byte an index) is open
- Writing the embedding tables at conversion so the checkpoint (LFM3 3.5 GB, Gemma 4 10.2 GB) is not opened at run time (T5)
- Conversations set aside moved out of the process (to disk), halved with int8 (the rest of T4)
- Changing the number of parts at run time (`MemoryBudget`, `Reshape`) (16.12)
- A WordPiece tokenizer (to use the BERT family from hwnpu-chat)
- Speculative decoding with a draft model (such as Qwen3.5's `mtp.` layers) (T9): Apple done (16.109); AMD not yet (the model's hidden states out and a version of the MTP layer)
- Repairing `hwnpu-chat agent`'s broken calls (output-parser)
- AMD: the rest of checking drafts (16.117): a GEMV of a few rows for q4 (Gemma 4, PLaMo 3, LFM3 at 4 bits), Gemma 4's shared layers (last row only) and ring caches, attention over the rows in one MHA, fewer copies of the states
- The rest of AMD's pinned memory (amd-npu-models.md section 12):
  - Split the VAE into tiles (with the DiT in one bank (16.106), a VAE stage is now the largest)
- The rest of Anima:
  - CFG (negative prompts) done in 16.148 (Anima-Aesthetic, 3 min 19 s an image at 1024² on AMD with DPM-Solver++ 2M in 25 steps and the guidance at the noisier steps only, 16.154; 4 min 40 s with the model card's 30 Euler steps); not yet checked on Core ML and OpenVINO devices
  - Calls from ComfyUI nodes
- The rest of Qwen3.6-35B-A3B's prompt (16.116; a chunk of 256 2.6 s): the experts' command (B, 1.49 s) is bound by the kernel's arithmetic (streaming the weights alone takes 8.4 + 4.4 ms of 19–23 + 15–17 ms a layer; a scalar MAC, no mmul. bf16 mmul in groups of 8 rows was tried and was not faster), the GDN and attention command (A, about 0.7 s; `GatedDeltaRule` receives every row 16 times), the host's routing (about 2.8 ms a layer, mostly a 16 MB copy of x)

## 12. The speed ceiling and what software can still move (estimate of 2026-11-08)

This section uses `hwnpu-dev probe` on the M1, M2 and M3 (PROBE.md, `npu_profiles/`) and the AMD measurements (amd-xdna2-bench.md) to answer two questions: does a faster NPU make generation faster, and what can software still change? 12.2 is an estimate from public specs and has not been measured.

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
- **gpt-oss-20b on AMD runs at 93 % of this floor** (16.179): a token moves 2.53 GB, 46 ms at 55 GB/s, and takes 49.4 ms. Getting there took removing everything between the reads (operators loaded again, a barrier, exponentials, the host's loop over the logits), not faster arithmetic: the 4 cores a column have compute to spare
- TOPS helps only prefill and attention over a long context (the one-row matrix-vector products the ANE cannot fill). One generated token follows model-fit.md section 2: weight bytes ÷ read bandwidth + attention + about 3 ms. A faster NPU shrinks only the second term

### 12.2 What newer chips may bring (not measured)

Generation of LFM3-1.7B (lut4, short context) compared with the M3. The NPU's share of DRAM is assumed to be 50–60%, from Apple's 64% and AMD's roughly half.

| | NPU read, expected | Generation | Prefill | Runs on hwnpu |
|---|---|---|---|---|
| Apple M6 (two Neural Engines, 170 GB/s) | The same if one engine still reads 64 GB/s. Up to about 2× if both can be used at once or the path is wider | 1.1–2× | About 2× the M5's compute, 2–4× the M3 | Yes (Core ML) |
| Snapdragon X2 Elite (80 TOPS, 152 GB/s) | 75–100 GB/s | 1.3–1.6× | About 4× the M3 on paper (effective unknown; AMD reaches 10–40% of its rating) | No (no Qualcomm backend) |
| Snapdragon X2 Elite Extreme (228 GB/s) | 115–150 GB/s | 1.8–2.2× | Same as above | Same as above |

- The M6's second engine helps generation only if a layer's matrices can be split into two programs called at the same time. That doubles the fixed cost per call
- To check: run `hwnpu-dev probe` on the new machine and read "weights GB/s" for lut4 at 16 layers. If it is about 64, generation is the same as the M3 and only prefill gains

### 12.3 What software can move

Tokens/s ≈ read bandwidth ÷ bytes read per token × tokens advanced per read (minus the time not spent reading). The bandwidth is close to the ceiling (12.1), so the other terms are what is left.

| Term | Method | State | Expected or measured | Section |
|---|---|---|---|---|
| Bytes read per token | Fewer bits (lut3 and lut4 mixed per layer, tables fitted to activations) | Up to choosing levels with GPTQ (16.87). Fitting the table is not done | A trade against accuracy | model-fit.md T8 |
| 〃 | MoE (read only the experts in use) | Done on AMD with Qwen3.6-35B-A3B (16.89–16.93). Not on the ANE (the routing `gather` goes to the CPU, model-fit.md section 4) | LFM2.5-8B-A1B (8.3B total, 1.5B active) reads about 0.75 GB a token at lut4, a floor of about 12 ms on the ANE. Computing every expert densely reads 4.2 GB, about 65 ms. gpt-oss-20b (21B, 3.6B active) on AMD reads 2.53 GB a token: a floor of 46 ms, run at 49.4 (16.179) | — |
| Tokens advanced per read | Speculative decoding (prompt lookup) | Done on Apple and Intel. A Q=8 call costs 13% more than Q=1 | 15.9 → 23.7 tokens/s, 1.2–1.5× over a whole task | 16.33, 16.36, 16.37 |
| 〃 | A draft model (such as Qwen3.5's `mtp.` layers) for more hits | Apple done (16.109, M1 Qwen3.5-0.8B 26.1 → 43.8 tok/s on code). AMD not yet | Gains if it beats prompt lookup's 1.07–1.67 tokens a call | model-fit.md T9 |
| 〃 | Read once for several requests (batching) | Not done | One read serves several conversations. Total throughput grows when subagents run side by side | — |
| 〃 | Draft checking on AMD | Done for q8 / q6 (16.117): a check of 3 rows costs 1.27 decode steps. q4 not yet | Qwen3.5-0.8B 82 → 62–72 ms/token with prompt-lookup drafts | 16.117 |
| Time not spent reading | Fixed cost per call (about 3 ms), switching parts, host work | LFM3-1.7B at a 1K capacity: of 19.1 ms, 13.4 ms reading weights, 2.8 ms attention, 3 ms fixed | 15–16 ms expected after trimming the fixed cost and attention | model-fit.md section 2 |
| 〃 | AMD: one submission for a whole layer | Done (a full ELF is 2.7× faster than submitting each op). A fused layer reaches 83% of the read floor | Avoids piling up the 0.06 ms per call | amd-xdna2-bench.md section 4.10, amd-npu-models.md section 2 |
| 〃 | AMD: a decode step as one resident array (no operator loaded again), the host's loops in vector code | Done for gpt-oss-20b (16.179). The other models still run their operators apart (about 67 µs and 1.2 µs a KB of programs each time one follows another) | gpt-oss 79.2 → 49.4 ms. What is left: a command boundary a layer (0.1 ms), a phase's fixed 25–35 µs, the host's 1 ms | 16.179 |
| Attention over a long context (compute-bound) | `SDPAGrouped`, splitting around attention to compute only the length used | Done (16.39, 16.59). Two-level attention (16.61). AMD gpt-oss's decode: 128 rows 21 → 9.6 µs a layer, down to their transfer (16.179) | Both TOPS and how the program is written shrink this. On AMD the cost was the softmax's exponentials, not the products | 16.39, 16.59, 16.61, 16.179 |
| (For reference) Another reader | Use the DRAM bandwidth the ANE leaves (100 − 64 GB/s on the M3) from the CPU, for example drafting on the CPU and checking on the ANE | Not tried. Using the GPU is a non-goal | Whether syncing pays off is not measured | — |

Expected (M3, LFM3-1.7B, lut4, short context; rough): 19–21 ms now (about 48 tokens/s) → 15–16 ms after trimming the fixed cost and attention (about 60–65 tokens/s; the weight floor is 13.4 ms, about 75) → about 80–90 tokens/s effective with speculative decoding. The default lut6 reads 1.5× the weight bytes, so its weight term is that much larger.

### 12.4 What AMD's work of 2026-11-13 may bring to Apple and Intel (not tried)

gpt-oss-20b on AMD (16.179) took generation from 97.5 to 80.5 ms and Qwen3.5-0.8B's from 87 to 73 ms by cutting operators and their fixed costs. Core ML and OpenVINO compile a whole program and fuse elementwise work into it themselves, so most of it has no counterpart there. What may carry over, in the order worth trying:

| Technique | Apple (Neural Engine) | Intel (NPU) | Why |
|---|---|---|---|
| A gated MLP's gate and up as one linear (one matrix of 2 I rows, the glu after it) | Worth a measurement | Worth a measurement | One wider matrix multiply in place of two: fewer kernels inside the program and longer runs of weights. hwnpu's `blocks` builds them as two linears. Rows up to 16384 on the ANE (2 I of the chat models fits) |
| A checkpoint's own quantization kept (MXFP4, a power-of-two scale per 32 in-features) | Not exact: the ANE's tables are per 16 rows, not per 32 in-features | Worth checking: OpenVINO has f4e2m1 with f8e8m0 scales; whether the NPU plugin runs them is not known | Rounding again costs accuracy (gpt-oss: KL 2–3×) |
| Fewer bytes for scales (a byte of exponent, larger groups) | — (Core ML's formats are fixed) | int4 with larger groups (now int8 by default) | Scales are 6 % of a 4-bit weight's bytes at groups of 32 in bf16 |
| Telling transfers from compute | `hwnpu-dev probe` already reads weights alone; a part's time against its weight bytes says the same | The same with OpenVINO's profiling | Decides whether fewer bytes or less compute is the lever |
| Attention sinks (a model that has them) | Easy in MIL: one column of the sink's logit concatenated to the scores before the softmax, dropped after | The same in the IR | gpt-oss itself does not run there (its routing), a dense model with sinks would |
| Biases and residuals in the linear | Not needed: Core ML folds them | Not needed: OpenVINO fuses them | AMD's gain came from removing operators IRON runs one by one |
| The host's share of a token read with pprof (2026-11-14) | Worth a measurement: top k already gained from the change; what a step does with its logits rows is in `HWNPU_PROFILE`'s `rows` | The same | gpt-oss on AMD had 1.33 ms a token in a loop over its logits; a larger vocabulary or a faster NPU shows such loops |
| A whole step as one resident program | Already so: a compiled program stays loaded (but large Core AI programs run in turn pay 50–70 ms each, 12.5) | Already so | AMD's gain came from IRON loading an operator each time it follows another |

### 12.5 gpt-oss-20b's lessons on the image models through Core AI (measured 2026-11-14, M1)

gpt-oss-20b on AMD doubled by removing what sat between the weight reads, not by faster arithmetic (12.1). The heavy image models on Core AI (Anima, Chroma, Qwen-Image, Sana) are the other case: a DiT pass at 1024² is 4096 tokens, bound by compute, not by reading weights. Each lesson was measured on Anima's DiT (1024², int8, groups of 4 blocks; design.md 16.201) first.

| gpt-oss's lesson (16.179) | On Anima's DiT through Core AI | Result | The other models |
|---|---|---|---|
| Tell transfers from compute (`MX4_NOCOMPUTE`) | Parts of a block written as programs of their own and timed (`probe_test.go`, `aicheck run -calls`) | A group 928 ms. The MLP 54 ms a block (5.1 TFLOPS, near the ANE's peak), **the self-attention's core 102 ms (1.3 TFLOPS)**: about 40 % of a step | Chroma and Qwen-Image attend over 4096 image tokens at 1024² too. Sana's self-attention is linear (no score matrix) |
| The attention's cost was its exponentials, not its products | Core AI writes softmax as `exp(x − max) × reciprocal(sum)` (16.169), passes over the 16 × 512 × 4096 scores; one pass is about 11 ms of a block | **The attention scale in the queries' RMSNorm weight**, no multiply over the scores: a group 928 → 884 ms (−4.7 %). With the norms' magnitudes by `abs` (below): Anima-Turbo 7.1 → 6.5 s a step, 1 min 02 s → 56 s a picture; Anima-Aesthetic guided steps 13.6 → 12.7 s, the rest 7.0 → 6.5 s, 25 steps 4 min 48 s → 4 min 27 s (latents 0.99998). The same error against milcpu | Chroma (FLUX) and Qwen-Image norm their q and k with RMSNorm: the same change applies. Not measured |
| (the same lesson, by layout, 16.202) | The self-attention a head at a time, a chunk's scores (1, queries, 1, keys) with the softmax over the last axis (`milpart.HeadAttention`); the cross-attention the same with the softmax's sum divided after the product with the values | The self-attention's core 90 → 72 ms a block (its two products are 31), the cross-attention's 14 → 10; a group 864 → 759 ms; **Anima-Turbo 6.5 → 5.6 s a step, Anima-Aesthetic 4 min 27 s → 3 min 51 s**. Every head in one tensor of the same layout is slower than before. The sum divided last is 12 ms faster for the self-attention too, but its product reaches fp16's largest in the last block, and a scale in the values' int8 weights for it changed the pictures (the ANE drops products under fp16's normal range) | Chroma and Qwen-Image: the same attention over 4096 image tokens, not measured |
| An operator that follows another is loaded again (`GptOssStep`) | `aicheck cycle`: a 4-block group that follows another group costs **50–70 ms more** (6 %), whatever runs between them; two small programs, or a group and a small one, cost nothing | About 0.4 s a step (each of the 7 groups pays it). Groups of 7 blocks: −1.2 % only, not adopted | Chroma's and Qwen-Image's groups are larger; check with `aicheck cycle` before choosing a group size. 12.4's "a compiled program stays loaded" does not hold for large Core AI programs in turn |
| A gated MLP's gate and up as one table (`QGEMVGlu`) | Anima's MLP is not gated; its q, k, v as one linear (2048 → 6144) | A group 928 → 946 ms (slower). Not adopted | Core AI's compiler already runs the three as long matrix products |
| Attention sinks | Anima zeroes the text rows past the prompt's T5 tokens: their keys and values are 0, their scores 0. A short bucket of keys with ln(dropped + 1) added on its last zero row is exact | The cross-attention's core 16.1 ms (512 keys) → 9.3 (256) → 5.8 (128); a group −2.3 % at 256. Needs a 512-row function too, and Core AI compiles all of an `.aimodel`'s functions at once (the first compile doubles): pending | Chroma's pads are masked (its `--text-rows`); Qwen-Image cuts the text per call already |
| The host's loops in vector code | float32 ↔ fp16 of a group's input and output: 2–4 ms a call | Nothing to gain (the conversions are already vector code) | The same |
| Work that does not change with the step, out of the loop | The cross-attention's keys and values once a prompt (as Sana, 16.189) | Saves about 3.5 ms of compute a group but adds 16 MB of input a call; the weights the ANE holds −117 MB (4 %). Not adopted for the speed | Sana has it (memory) |
| Fewer bytes per weight (MXFP4 kept, bfp16 GEMMs) | — | Measured the other way round: a block without its attention cores int8 97.4 ms, fp16 111.9 ms, lut8 111.6 ms. The ANE pays nothing to expand int8 and reads half the bytes: more bits are slower and double the memory it holds | — |
| AMD's own kernels (a new GELU, LayerNorm, the heads' RMSNorm: Anima-Aesthetic −21 % with bfp16) | Only the graph's form can change. Of a block's 221 ms: the q and k norms 11.3 ms (4.8 of it the division by the row's maximum that keeps fp16 squares in range), RoPE 3.4, the LayerNorms' same division 3.7, the exact GELU 5.8 (the tanh and sigmoid forms are slower) | **The magnitude by `abs`** in place of the maximum of x and −x (`milpart.AbsScaled`): bit for bit the same pictures, a group 884 → 864 ms (−2.3 %). The division itself cannot become a constant: measured in a real run (`mil.BlockRanges`), q and k reach 384 (the square is past fp16) and their rows' RMS spans 1400× | The softmax's passes over the scores (about 59 ms a block) are what AMD removed with its own kernel; on Core AI the layout above takes 18 of them |
