# NPU papers we drew on, and what held up in hwnpu

[日本語](reference-paper.ja.md)

Written 2026-11-21. The paper survey that began on 2026-11-20 ([handoff-amd-papers.md](handoff-amd-papers.md)) picked out the papers below. For each one, this page says what it claims and whether it held on our hardware. The detailed numbers and how to rerun the tests are in the [docs/design/README.md](design/README.md) sections cited. This page is the list and a guide to those sections.

- The focus is optimization ideas for the AMD NPU (XDNA2, Ryzen AI 9 365). It also covers MoE papers, and the Apple Neural Engine papers checked on an M2 around the same time
- The full table of MoE papers is chapter 13 of [docs/design/performance.md](design/performance.md). Only the ones relevant to AMD are here

**How closely each paper was read** (same marks as handoff-amd-papers.md)
- ✓: the full text (HTML) was read
- (a): the abstract only
- †: a widely known technique; the paper itself was not read

Most of the 2026 papers are preprints that have not been peer-reviewed.

**Result labels**
- **Held**: the claim held on our hardware, and we used it to speed something up or fix something
- **Already done**: hwnpu already did the same thing
- **Partly**: parts held and parts did not
- **Not started**: not tried yet

## At a glance

| Paper | Claim | Result | hwnpu sections |
|---|---|---|---|
| HeteroMosaic 2607.12839 ✓ | Reconfiguring the datapath (reloading the PDI) is expensive. Switching DMA settings is nearly free | **Held.** One reconfiguration costs 40 µs + 2.3 µs/KB, about 70 % of dense-model decode time. Fusing ops took LFM3 from 11.2 to 16.9 tok/s, among others | [amd](design/amd.md), [README](design/README.md).278 |
| 2606.07586 ✓ (Llama-3.2-1B on XDNA2) | Reuse the XRT context. Pass BOs zero-copy. Skip host syncs for buffers only the NPU touches | **Already done.** The IRON path already syncs only the ranges it uses | end of [amd](design/amd.md) |
| 2512.13282 ✓ (GEMM on two Ryzen AI generations) | Reorder with multi-dimensional DMA addressing instead of tiling weights on the host. Use a single C buffer for the output | **Not started.** The 1023-word limit on one DMA dimension came up again in [amd-llm](design/amd-llm.md) | [moe](design/moe.md), [amd-llm](design/amd-llm.md) |
| Asymmetric Tile Buffering 2511.16041 (a) | Use different tile sizes for input and output (4.54× on a BFP16 GEMM on XDNA2) | **Not started** | |
| 2609.21264 ✓ (FlashAttention in IRON and MLIR-AIR) | Fuse attention, keeping QKᵀ in core memory. It pays off on XDNA2 | **Not started.** Fusion also cuts reconfigurations, which agrees with [amd](design/amd.md) | |
| STEEL 2607.09385 (a) | Place FlashAttention pipeline stages with the causal mask's sparsity in mind | **Not started** | |
| Zen-Attention 2508.17593 (a) | Fold attention to cut DRAM bandwidth | **Not started** | |
| MR-GPTQ 2509.23202 (a), QuaRot / SpinQuant † | Rotate with a Hadamard transform before quantizing, to flatten outliers | **Not started on AMD.** On Apple's lut formats it roughly halved the KL (not in the product yet) | [apple-neural-engine](design/apple-neural-engine.md) |
| TileFuse 2606.11357 ✓ | BFP16 kernels are not accurate enough for end-to-end LLMs | **Check only.** hwnpu checks every bfp16 use with KL or cosine | [moe](design/moe.md), [image-amd](design/image-amd.md) |
| ENEC 2604.03298 (a, Ascend) | Lossless weight compression removes the transfer bound | **Not started.** Decode today is bound by reconfiguration, not bandwidth, so this waits | |
| 2505.16056 (Not All Models Suit Expert Offloading) | A cache about twice the active experts is enough. LFU beats LRU | **Partly.** AMD's expert places (`--expert-places`) are now chosen by usage count | [moe](design/moe.md), [moe](design/moe.md) |
| EStream 2609.06551 (Hexagon) | Keep experts in a bank × slot pool, addressed as base + slot × size. Read with O_DIRECT | **Held.** It is the basis of AMD's expert places. gpt-oss-120b reads with O_DIRECT | [moe](design/moe.md), handoff-amd-moe.md |
| Fate 2502.12224 | Predict the next layer's gate from the current layer's input, and prefetch | **Held.** Apple's gpt-oss-20b: 86–87 % in the top 4. AMD's gpt-oss-120b: misses 9.8 → 3.1 % | [docs/design/performance.md](design/performance.md), handoff-amd-moe.md 4 |
| NPUMoE 2604.18788 | Only the prompt on the Neural Engine; routing and gather on the CPU | **Already done.** It matches the decision not to put gather on the Neural Engine ([moe](design/moe.md)) | [moe](design/moe.md) |
| Pre-gated MoE 2308.12066 and others | Compute the next gate one layer early | Read as background only | [docs/design/performance.md](design/performance.md) |
| 2606.22283 (Neural Engine reverse engineering) | Numeric rules: outputs below 2^15, width slices, softplus, and more | **Partly.** Most rules hold on the M2. The width-slice rule triggers under narrower conditions than the paper says | [apple-neural-engine](design/apple-neural-engine.md) |
| 2611.19101 (Neural Engine roofline) | A slow activation path (about 14 GB/s). Runtime weights run at half the speed of constant weights | **Held.** On the M2, elementwise ops reach 16–20 GB/s; runtime weights 26 GB/s against 50–52 GB/s for constants | [apple-neural-engine](design/apple-neural-engine.md) |
| ANEForge 2606.17090 | A direct call costs about 70 µs | **Partly.** Through Core AI a call costs about 160–175 µs | [apple-neural-engine](design/apple-neural-engine.md) |

## AMD NPU

### HeteroMosaic 2607.12839 ✓ — held

**Claim**
- Reconfiguring the datapath (reloading the PDI) is expensive. Switching DMA settings is nearly free
- Group operations by xclbin. Drive the NPU from a single thread

**What we checked** ([amd](design/amd.md))
- In hwnpu's full ELF, every change of operator puts a `load_pdi` and the array configuration (including core program memory) into the control code
- One reconfiguration costs 40 µs + 2.3 µs/KB. This is what lay behind 16.75's "about 80 µs per operator of a different kind"
- The estimate matched real decode times within 10 % (LFM3-1.7B: 98 ms estimated against 92.6 ms measured; Gemma 4 E2B: 138 ms against 137 ms)
- gpt-oss-20b's decode does not fit the pattern: it is a single resident operation (`GptOssStep`)

**What we did with it**: cut the number of operations per layer. Reordering barely helps (because of dependencies, at most 2 % per token on Qwen3.6).

| Section | Change | Result |
|---|---|---|
| [amd-llm](design/amd-llm.md) | Moved GDN's .bss arrays into Worker-local buffers (removing the blocks of zeros) | Qwen3.5-0.8B 13.2 → 14.1 tok/s, logits bit-identical |
| [amd-llm](design/amd-llm.md) | `ShortConvStep`, `SandwichAdd` and `QGEMVGlu` in LFM3's decode | LFM3-1.7B 11.2 → 15.6 tok/s |
| [amd-llm](design/amd-llm.md) | Gemma 4's per-layer inputs computed at once, and `SandwichAdd` | Gemma 4 E2B 7.7 → 9.3 tok/s |
| [moe](design/moe.md) | Merged t3/qwen36-integrate (GDN reads and writes its state slots directly) | Qwen3.5-0.8B 13.8 → 18.7, Qwen3.6 8.5 → 8.9 tok/s |
| [amd](design/amd.md) | QGEMV on one core per column (about half the configuration), sandwich inside the GEMV | LFM3-1.7B 15.6 → 17.2, Gemma 4 E2B 9.3 → 9.7 tok/s |
| [amd-llm](design/amd-llm.md) | The prompt's short convolution as one operator (`ShortConvChunk`, `GdnConvChunk`). No state copies in LFM3's decode | LFM3 16-token prompt −13 %, Qwen3.5-0.8B −25 %. LFM3 decode 16.5 → 16.9 tok/s |

**What did not work, or could not be used**
- `--reconfig-teardown`: faster for small configurations, slower for large ones (GEMM)
- `--load-pdi-to-ctrl-pkt` (streaming the configuration over DMA): aiecc stops with "operand #0 does not dominate this use"
- "Drive the NPU from a single thread" was not tried

**Remaining**: to cut decode's remaining reconfigurations by much, a layer has to become one array (the `GptOssStep` form). On Qwen3.6 the big cost is GDN's 374 KB configuration × 30 layers.

### 2606.07586 ✓ — already done

**Claim**: Llama-3.2-1B decodes at 10.8 tok/s on XDNA2. Reuse the XRT context and pass each layer's BOs zero-copy. Skip host syncs for buffers only the NPU reads and writes.

**What we checked** (end of [amd](design/amd.md)): the IRON path (`readRegion`, `writeRegion`, `writeCopies` and `rows.go` in `internal/llm/iron`) already syncs only the ranges it uses. Whole-buffer syncs remain only at load time and in the non-IRON path (`internal/llm/chain`, small buffers). Nothing was changed.

### 2512.13282 ✓ (GEMM on two Ryzen AI generations) — not started

**Claim**
- int8 38 TOPS and bf16 14.7 TOPS on XDNA2. bf16 goes through the bfp16 datapath
- Reorder with multi-dimensional DMA addressing instead of tiling weights on the host ahead of time
- When the output transfer can be amortized, use a single C buffer (+13–18 %)

**Relation to hwnpu**
- The bfp16 datapath is already used for the DiT and for gpt-oss's prompt experts
- One DMA dimension holds at most 1023 words, so the 4-D TAP that would reorder weights into consumption order does not fit ([moe](design/moe.md)). The expert tables are reordered on the host instead
- 16.278's `ShortConvChunk` also hits this limit: `length_parameter`, which would cut the DMA by row count, cannot be used, so every row of the chunk is moved
- Splitting a dimension longer than 1023 words into two has not been tried

### Asymmetric Tile Buffering 2511.16041 (a) — not started

**Claim**: keep tiles of different sizes for input and output. 24.6 TFLOPS, 4.54×, on a BFP16-BF16 GEMM on XDNA2.

**Candidate targets**: the image DiTs' GEMMs (Anima, FLUX, Chroma) and gpt-oss's prompt experts. It may also bear on 16.278's finding that IRON's GEMM streams the whole weight to every row unit regardless of the call's rows (60 % of a short prompt).

### Attention fusion: 2609.21264 ✓, STEEL 2607.09385 (a), Zen-Attention 2508.17593 (a) — not started

**Claims**
- 2609.21264: FlashAttention in IRON and MLIR-AIR, up to 128K. The fused version reaches 3.62 TFLOP/s on XDNA2. QKᵀ stays in core memory and is gathered over the cascade. Fusion pays off on XDNA2 and barely at all on XDNA1
- STEEL: places the pipeline stages with the causal mask's sparsity in mind
- Zen-Attention: folds attention and chooses tiles to cut DRAM bandwidth. Attention up to 4×, end to end up to 32 %

**Relation to hwnpu**: the targets are the prompt's causal attention and the DiT's attention. Fusion also cuts the number of reconfigurations, which agrees with [amd](design/amd.md). 2609.21264 is written in IRON, so if its code is published, the quickest route is to compare it in the same shape as today's `iron_mha.py`.

### Rotation: MR-GPTQ 2509.23202 (a), QuaRot / SpinQuant † — not started on AMD, partly on Apple

**Claim**: rotate the residual dimension with a Hadamard transform before quantizing, to flatten outliers. The rotation is folded into the weights, so it adds no runtime operations.

**What we checked** (Apple, [apple-neural-engine](design/apple-neural-engine.md)): on Qwen3.5's lut formats the KL roughly halved.
- Qwen3.5-2B lut6: 0.056 → 0.027
- Qwen3.5-0.8B lut6: 0.075 → 0.039
- lut4 is unusable even rotated (2B: 0.60 → 0.37)

**Limits on bringing it to AMD** (found in [apple-neural-engine](design/apple-neural-engine.md))
- Only models where everything reading the residual is a linear after a norm, and everything writing it is added as is, can be rotated
- Among the chat models, only the Qwen3.5 family qualifies
- LFM3, Gemma 4 and PLaMo 3 have weighted norms on the writing side (sandwich norms), so they cannot be rotated
- gpt-oss would need its experts rotated and requantized too

AMD's q4 might benefit (most of Qwen3.6's gap to HF, cosine 0.986–0.996, comes from q4), but this has not been tried.

### TileFuse 2606.11357 ✓ — check only

**Claim**: fused W4A16/W8A16 kernels on XDNA2. "BFP16 kernels are not accurate enough for end-to-end LLMs."

**Relation to hwnpu**: hwnpu uses bfp16 for the DiT's attention and MLP (checked by latent cosine, [image-amd](design/image-amd.md)) and for gpt-oss's prompt experts (KL measured, [moe](design/moe.md)). A new LLM that uses bfp16 should go through the same KL sweep. We have not re-measured.

### ENEC 2604.03298 (a, Ascend) — not started

**Claim**: fixed-length block codes and branch-free decoding remove the weight-transfer bound. Up to 6.3× end to end.

**Relation to hwnpu**: as 16.266 shows, dense-model decode is bound by reconfiguration, not bandwidth, so this only helps once reconfiguration is cut. First measure the entropy of q4nx's codes and scales; if it does not fall below 2.5 bits per code, decoding probably costs more than it saves.

## MoE

The full table is chapter 13 of [docs/design/performance.md](design/performance.md). Only the papers relevant to AMD are here.

### 2505.16056 (Not All Models Suit Expert Offloading) — partly

**Claim**: measures local routing consistency across 20 MoEs. A cache of about twice the active experts is enough. LFU beats LRU by 3–5 points.

**Relation to hwnpu**
- AMD's expert places (`--expert-places`, [moe](design/moe.md)) choose which experts to keep by calibrated usage count
- Qwen3.6-35B-A3B: 7.5 GiB on a 16 GB machine, 5.8–6.0 tok/s ([moe](design/moe.md))
- The places are larger than "twice the active experts" (gpt-oss-20b on a 16 GB machine keeps about 5×: 20 of the 32 per layer)

### EStream 2609.06551 (Hexagon) — held

**Claim**
- One compiled expert graph serves every expert
- Weights live in a bank × slot DMA-BUF pool, addressed as base + slot × stride. The route descriptor is passed at call time
- Experts are read with O_DIRECT (two at once reach 95 % of the bandwidth)

**Relation to hwnpu**
- The positions `MoERoute` writes to the scratchpad ([moe](design/moe.md)) follow the same idea as the paper's "expert virtualization". It became the basis of AMD's places, which hold only some of the experts ([moe](design/moe.md))
- gpt-oss-120b reads its expert files with O_DIRECT (handoff-amd-moe.md 4)

### Fate 2502.12224 — held

**Claim**: predicts the next layer's gate from the current layer's input. 79–84 % correct during generation.

**Relation to hwnpu**
- Apple's gpt-oss-20b: the next layer's prediction holds 86–87 % of the experts in its top 4. Prefetching made every budget 3–6 % faster
- AMD's gpt-oss-120b: prefetching the next layer's top 4 cut misses from 9.8 to 3.1 % and took it from 9.2 to 9.7–9.9 tok/s (handoff-amd-moe.md 4)

### NPUMoE 2604.18788 — already done

**Claim**: only the prompt runs on the Neural Engine, through Core ML. Routing, gather and write-back stay on the CPU.

**Relation to hwnpu**: it matches hwnpu's decision not to put gather on the Neural Engine ([moe](design/moe.md)). The paper does not cover generation.

## Apple Neural Engine (checked on an M2, [apple-neural-engine](design/apple-neural-engine.md))

Around the same time as the AMD survey, we checked the Neural Engine papers' rules and limits on an M2. Nothing got faster. What went in: correctness fixes, limit detection, and a measuring tool (`tools/coreai/aicheck`).

### 2606.22283 (reverse engineering) — partly

| Rule in the paper | On the M2 | What hwnpu does |
|---|---|---|
| linear / conv outputs must stay below 2^15 | Holds, **including partial sums**: Inf once one reaches 32768 | Ranges were already kept down. Added to the checks for a new model |
| A width-axis slice with a nonzero start multiplies by 16 | Only when the start mod 32 is 1–14 | `aicheck lint` looks for it. Nothing in our exports is affected |
| softplus(+inf) is 0 | 0 for x > 10.4 | ailower's softplus was rewritten (worst relative error 1 → 0.0014) |
| Denormals inside a matmul become 0 (the paper says M1) | The M2 too | Covered by the range checks |
| About 128 programs per process | Up to **112**. Beyond that Core AI silently runs the program on the GPU | hwnpu now says so when a load fails near the limit |
| A step at 5 experts per program (M1) | **Not on the M2** (time grows with top-k) | |

### 2611.19101 (roofline) — held

- The slow activation path (about 14 GB/s on an M2 Pro): elementwise ops on the M2 level off at 16–20 GB/s
- Two weight paths: runtime weights 26 GB/s, constant weights 50–52 GB/s, about half
- lut8 takes 0.6× the time of fp16, as in the paper's M2 table

### ANEForge 2606.17090 — partly

A direct call costs about 70 µs. Through Core AI it costs about 160–175 µs; Core AI and the driver add about 100 µs.

## Things the papers led to that they did not cover

- **GDN's blocks of zeros** ([amd](design/amd.md), [amd-llm](design/amd-llm.md)): the tool that counts reconfigurations (`tools/amd/elfconfigs`) showed aiecc writing zeros to the GDN kernel's .bss arrays on every reconfiguration. No paper mentions this
- **QGEMV's configuration size** ([amd](design/amd.md)): since the cost grows with size (2.3 µs/KB), QGEMV moved to one core per column, about halving its configuration. GEMV speed is set by bandwidth, so it did not change
- **The limits of IRON's GEMM** ([amd-llm](design/amd-llm.md)): 60 % of a short prompt's time is the GEMM, which streams the whole weight regardless of the rows, plus weight expansion. This is a limit of how IRON's GEMM is built, outside the papers' scope
- **Why the Qwen3.6 export was slow** ([amd-llm](design/amd-llm.md), [moe](design/moe.md)): the shared export had been built from a branch that was not in main (t3/qwen36-integrate)

## Candidates not yet tried (in priority order)

Sections 3–9 of handoff-amd-papers.md describe how to try each.

1. Avoid the 1023-word limit with multi-dimensional DMA (2512.13282). Less host reordering, and prompt operators could cut their DMA by row count
2. Attention fusion (2609.21264). It also cuts reconfigurations
3. Different tile sizes for input and output (2511.16041). For the image DiTs and long prompts
4. Rotation before q4 (QuaRot style). For the accuracy of AMD's q4 on Qwen3.5 / 3.6
5. Lossless weight compression (ENEC). After reconfiguration is cut
