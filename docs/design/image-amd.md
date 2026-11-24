# Image models on AMD's NPU

[日本語](image-amd.ja.md)

How the DiTs, VAEs and text encoders of the image models run on AMD XDNA 2 through IRON. The models, samplers, guidance, sizes and the export rule: [image-models.md](image-models.md). The IRON infrastructure, driver and kernels shared with the LLMs: [amd.md](amd.md). Numbers are on a Ryzen AI 9 365 unless said.

## Graph exports

- An image model is exported as IRON graphs, input to output (`tools/amd/iron_graph.py`, Go `iron.Graph` in `internal/llm/iron/graph.go`), not as the LLMs' decode/prompt programs. Exporters: `tools/amd/iron_anima.py`, `iron_flux2klein.py`, `iron_flux3klein.py`; shared pieces in `iron_dit.py` (`Checkpoint`, `qparam`, `head_perm`, `rope_angles`, `share_banks`, `VaeStage`, `timestep_sinusoid`, `text_linear`) and FLUX.1's VAE in `iron_chroma_parts.py` (`FluxVaeStage`; the rest of it, T5, was Chroma1's, removed: [not-adopted/chroma1.md](../not-adopted/chroma1.md)). The export directory is `NAME.amd-npu2` with `model.json` and a manifest.
- **Weight banks** (`variants`): one ELF holds several banks; a DiT's blocks are one graph of a few blocks and one bank per group (Anima: 28 blocks = a 4-block graph × 7 banks).
- **`SwapVariants`** (default): one bank is allocated and each group's image is read into it before the group runs (the arena puts weights first, scratch after; nothing in scratch survives a call). Anima 1024²: pinned peak 3445 → 1236 MiB for +2 % time, the same bytes. `--resident-dit` keeps every bank. The text encoders (Qwen3, T5, Qwen3-VL) are grouped the same way.
- **Prefetch** (`iron.GraphOptions.Prefetch`): a second bank; after `Use(i)` group i+1 is read in the background, and the next `Use` swaps banks. Used when the DiT stage plus the second bank plus a 384 MiB margin (`anima.prefetchMargin`: programs and control code pin memory too) stays under the VAE's peak, or under `anima.PrefetchFloor` (2 GiB). `HWNPU_DEV=iron.prefetch=0` turns it off. Anima hides only 0.1 of 0.5 s a step: the read and the NPU share the memory bandwidth.
- Reads that are not prefetched use 8 threads (`iron.readPrefix`): 5–7 → 15 GB/s.
- **`share_banks`**: a graph whose weights are the same bytes in the same order as an earlier graph's drops its own bank image and points at `../GRAPH/` (checked byte for byte). For this the weights' layout must not depend on the size or rows: **RoPE tables are inputs** (`angles`, rows × head_dim, made on the host by Go `ropeAngles`, `TestRopeAngles`), never baked into a bank. Older exports without `angles` still run.
- **`iron.GraphOptions.Barrier`**: a graph run a second time can time out on its first command (`ERT_CMD_STATE_TIMEOUT`; graphs writing K/V at per-call positions, Chroma1's 6-command approximator, removed). Loading the same ELF into another hardware context and running an empty command first avoids it (irongraph `-barrier`).
- Graphs are compiled per number of rows. A step's rows must be whole rounds of **512** (the attention's 8 pipelines × 64 rows); a size that is not is padded (below).

## Attention

`tools/amd/iron_mha.py`, `ImageMHA` (self- and cross-attention of every DiT):

- **Keys in blocks of 64.** mha.cc's fast softmax (`partial_softmax_rows`) runs only at B_kv = 64; at d = 128, IRON picks 32 × 32 and the softmax core bound the time. Without causal masking or a window, the key block need not equal the query block: B_q 32, B_kv 64, P·V on 2 cores (`pv_cores=2`). 4096² × 16 × 128: 146 → 103 ms.
- **QK^T in bfp16** (`qk`): AIE2P's native bf16 mmul does about 24 MACs a cycle; bfp16 (8 values share an exponent, 8-bit mantissas) is the matrix unit's native form. 103 → 59 ms (cosine 0.999936). **P·V in bfp16 too** (`qkpv`): 54 ms.
- **Softmax exponent with one bf16 limb** (`kernels/exp2_mid.h`, `mha_mid.cc`, `exp=mid`): IRON's accurate exp2 evaluates its cubic in float32 precision (two bf16 limbs a product: 10 macs + 6 limb splits for 32 values); the fraction and each Horner stage in one limb (4 macs) err within 0.2 %, the size of the final bf16 rounding. FLUX.2 attention alone: 102.7 → 83.2 ms, cosine 0.999650 → 0.999649. IRON's `linalg/mha.cc` is included after the header to replace `exp2_bf16`.
- The qk/qkpv flags and others are declared kernel parameters (`param(array=True)`): IRON's compile cache (`~/.npu/cache`) keys on operator parameters, so a method patched from outside silently reuses the old ELF.
- **Per-call key count `nk`**: keys past `nk` are cut (text padding, padded image rows). It need not be a multiple of 64 (4608 rows at `nk = 4568`: cosine 0.99995).
- **Zero keys counted in the softmax** (`ImageMHA(zero_keys=N)`, `kernels/mha_zero_keys.cc`, Anima `--dit-cross trim`): Anima's conditioning rows past the prompt are zero, and its cross-attention k, v have no bias, so those keys score 0 and add nothing but e^{0−m} each to the sum. `dit_blocks` takes `nt` (the prompt's rows) per call, reads `k[:nt]`, `v[:nt]`, and the P·V core adds (N − nt)·2^{−m s} before the division (where gpt-oss's sinks go). The query rows (4096) may now outnumber the keys (512), so the 4096 patches go in one call instead of 8 × 512. Cross-attention alone 12.7 → 4.6 ms; Anima-Turbo 1024² 3.7 → 3.2 s a step; Anima-Aesthetic 1024² −11 %. `nt = 512` is bit-identical to the chunked form. The empty negative is `nt = 1`. A new export needs a new hwnpu (an old one sends `nt = 0`, a call with no key blocks, which can hang the NPU).
- IRON's MHA holds scores in bf16: when scores are large (~1750) and a row's differences small, the conditional output drifts and CFG breaks. Subtracting the first key from the keys (softmax unchanged) fixed it on a removed model; Anima's cross-attention was not checked for it.
- The max is still subtracted. A bound from the q/k norm weights (√128·max|w_q|·max|w_k|: 15–45 in Anima, 221 in block 25) would allow skipping it, but the self-attention balances P·V (bf16, ~6000 cycles a call) and the softmax (~6000 a block); not tried.

| Model | `mha` | `exp` |
|---|---|---|
| Anima | qk | exact (`--dit-exp`; mid moved the picture, latent 0.976 / 19.6 dB, for 3 %) |
| FLUX.2 klein, FLUX.3 klein | qkpv | mid |

## DiT blocks

Measured on Anima's 4-block graph (1024², before the changes below): attention 48 %, GEMMs 29 %, elementwise 30 % (mostly bandwidth, ~58 GB/s), adaLN 1 %. Elementwise ops cost 0.6–0.8 ms each at 4096 × 2048; IRON's exact GELU 5.2 ms stood out.

- **Weights packed as bfp16 on the host** (`iron_q4.bfp16_pack`, `bfp16_linear`, format `bfp16`; default for every DiT's blocks): MEDIUM GEMMs round bf16 weights to bfp16 on every call, and q8 also dequantizes each step; packing at export into the `bfp16ebs8` layout `flm.ProjectionGEMM` reads removes both. One MLP: q8 25.9, bf16 23.0, bfp16 18.5 ms; error as bf16's. 9 bits a weight (q8 8.5). The packer and the GEMM must get the same `pack_b` tile_n 64, k tile 512 (384/256/128 when K is not a multiple), ct_k 128, or the order silently breaks. Registered as block-typed weights (`bfp16_operand`). Costs: export time (Anima DiT 45 s → 3 min 18 s; `bfp.encode` serial); `graph.reference()` cannot read packed B, so `--check` needs an export with `--proj bf16`. Embeddings, adaLN and the final layer stay bf16.
- **GELU** (`iron_gelu.GELUEven`, `kernels/gelu_even.cc`, `--dit-gelu even`): IRON's tanh table with the argument in float32 (x² exact, the coefficient as two bf16 limbs), rounded to nearest even: 1.76 ms vs the exact kernel's 5.2. Chroma1 (trained on tanh GELU, removed) used it too.
- **LayerNorm with the modulation multiply** (`F.layer_norm(x, weight=scale + 1)`, IRON's WeightedRowwise on a second core): 1.23 → 0.61 ms, bit-identical. Every adaLN.
- **Per-head RMSNorm** (`iron_headnorm.py`, `kernels/head_rms.cc`): q/k norms over (rows, heads × 128) in one call, weight tiled per head: 1.4 → 0.83 ms, bit-identical.
- **SiLU in float32** (`iron_silu.SiLUF32`, `kernels/silu_f32.cc`): IRON's SiLU multiplies x by a bf16 table sigmoid, off 1 % on average (2.3 % max) on [−6, 2] and exactly 0 at x ≤ −8. In an LLM's residual, channels of tens to hundreds make SwiGLU's up branch large there, so the error reaches the MLP's output. x / (1 + exp(−x)) with IRON's `exp_f32` and a float reciprocal, one bf16 rounding: 0.14 % mean error, 512 × 9728 in 12 ms. Used in FLUX.2 klein's Qwen3 text encoder (final latent 0.916 → 0.977). The DiT's SwiGLU keeps the table (its gates are not large negatives: first-step velocity 0.99999). Other SwiGLU LLM exports may be hit too; not checked.

## Text encoders

| Model | Encoder | Form |
|---|---|---|
| Anima | Qwen3-0.6B + LLM adapter | GEMMs at `HIGHEST`: MEDIUM (bfp16) spread error in the text (latent 0.938 → 0.966 with HIGHEST, same time) |
| FLUX.2 klein | Qwen3-4B, 9 groups of 3 layers, bf16, HIGHEST | the hidden states after layers 9, 18, 27 (7680 columns); `ctx` (context_embedder) as 3 GEMMs (K 7680 does not fit a HIGHEST GEMM) |
| FLUX.3 klein | Qwen3-VL-4B, 32 layers in 8 banks of 4, bf16 | FLUX.2's `Qwen3Layer`; bf16 holds the text stream's 9.7e6 without scaling |

- **T5's relative position bias**: IRON's MHA has no bias. 8 heads at a time: block-diagonal K^T and V^T (the projection weights on the GEMM's left give each head contiguous) put 8 heads' scores in one GEMM; the bias is a constant (shared by every layer, 64 MiB) added in the float32 Softmax; the padding mask goes into the scores GEMM's contraction (one row of K^T = (mask − 1) × 16384, a q column of 1). One run 3.2 s, cosine 0.9997 to transformers' float32.
- **Padding rows of a Qwen3 encoder (FLUX)**: the pipeline pads the prompt to 512 tokens and the DiT reads the pad rows. In transformers' mask a pad row attends to the prompt's keys only (not its own), prompt rows are causal. So attention runs twice — causal over all rows, and mask-free over the first `nk` keys — and an input `sel` (1 on prompt rows) picks per row (`op + (oc − op)·sel`). 512 rows in one call (64-row chunks would put a pad row's own key into the causal pass).
- The pad rows are sensitive to weight rounding (layer 18's pad rows: q6 0.904, q8 0.995, bf16 0.9995): FLUX.2's text encoder stays bf16 (5.9 GB with the 0.8 GB embedding table; a layer 61 → 52 ms).
- FLUX.3: the text lives in a 512-row slot (`text_slot`); only the prompt's R rows (tokens rounded up to 80, 80–480) are keys (`nk`); pad positions are 1. Run at 512 rows, the first R used (the same as an 80-row run).

## VAE

- **Convolutions on the grid** (`iron_gridgemm.GridGEMM`, `--vae-conv grid`): 3 × 3 convs without im2col copies (one GEMM if 3C ≤ 512, else three summed per kernel row). Upsampling as four output phases, each a 2 × 2 conv at the low resolution (`PHASE_TAPS`). Anima's VAE 9.3 → 5.2 s; PSNR 52.7 dB to the im2col form.
- `iron_anima.gemm_tile_n`: a tile that fills all 8 columns (16 for 128 channels, 32 for 256; the default 64 used 2 columns). Conv alone 46 → 38 ms, a stage only 1.37 → 1.30 s: convs are bound by A's transfer (9C a pixel).
- **Buffer reuse** (`iron.GraphOptions.Buffers`, `anima.VAE.run`): allocating a device buffer pins and zeroes pages (0.33 s a GiB, freeing 0.11 s); FLUX.2's 10 decoder stages allocated 17.6 GiB, 7.6 s of 14. Now one buffer per bank kind (the largest stage's size) serves every stage, zeroed after the image is read (multi-threaded), and a stage's output buffer is the next stage's input (no copy, no host round trip). FLUX.2 1024² decode 14 → 8.5 s, encode 7.3 → 4.8 s, the process's RSS 1.17 → 0.16 GiB, the peak 4.9 → 3.8 GB, the same bytes. Anima-Turbo's decode 5.2 → 4.0 s. (Prefetching VAE stages was only hiding allocation; it is gone.)
- **GroupNorm** (`iron_groupnorm.py`, `kernels/chan_stats.cc`, `kernels/row_affine.cc`, `anima.HostNorm`, `--host-norm`, default; FLUX's VAE in FLUX.2/FLUX.3 klein): IRON has neither GroupNorm nor rsqrt.
  - `ChannelStats`: per-channel sum and sum of squares (float32) from bf16 pixel rows, read once.
  - A host stage (`iron.Graph.After`, `anima.hostNorms`): named graph states (`gn0.stats`, `gn0.row`) located from the manifest's `buffers`; after each command, if the next norm's stats are written (a NaN marker is gone), the host computes group mean and variance in float64 and writes a = γ/√(var + eps), b = β − mean·a. The command is cut right after the stats op.
  - `RowAffine`: y = x·a + b, a and b as two bf16 limbs, float32 math, one bf16 rounding (x·a and b reach many times the result: |mean|/std up to 6.3 in this decoder).
  - Computed on the grid (zero pads don't touch the sums; divide by the real pixel count); after norm + SiLU the pads are no longer zero, so the image pixels are copied to a fresh grid.
  - 128 channels at 1024²: about 157 ms → a few ms. Decode 512² 2.6 → 1.8 s, 1024² 8.5 → 7.0 s; PSNR to float32 52.6 dB. `graph.reference()` does not know host stages: check on the NPU (`HWNPU_DEV=klein.decode=1` decodes `--noise`'s latents; `--dump`).
  - The older form (~20 ops of GEMMs and an RMSNorm: 1/√(var + eps) as an RMSNorm of the group's row with a constant row) needs no host and remains with `--no-host-norm`.
- **Latents not a multiple of 16** (rectangles, e.g. 104 × 156 for 832 × 1248; `iron_anima.GridShape`, Go `anima.GridShape`): the row width `Wp` = W + 16 pad rounded to 16; convs iterate over `Ha` (height rounded to 16) in 16-row chunks and copy only image rows. The middle attention pads pixels to multiples of 256 and adds −30000 to padded keys' scores (the mask built once as a GEMM of a ones column with the mask row: a 16384-column rowwise add exceeds 16 DMA descriptors) (`attn_masked`). Squares compute as before.
- DMA repeats are at most 64 (bound the column blocks); a transpose GEMM over 1 M pixels never finished, so it runs in commands of 262144 pixels (Chroma1 1024², removed).
- Stride-2 conv (FLUX.2 encoder): a same-size 3 × 3 conv, then every other pixel copied to the half grid (output (Y, X) is (2Y + 1, 2X + 1) of a conv padded one pixel right and down).
- The VAE runs a picture at a time even with `--batch` ([image-models.md](image-models.md)).

## Padding rows to 512

| Model | Row order | Padding |
|---|---|---|
| Anima | [patches] | patches rounded up to 512 (832 × 1248: 4056 → 4608); self-attention keys cut at `nk` = patches; pad rows' angle is position 0; output's first `patches` rows kept. Graphs named by rows (`dit_blocks_n1024`; 4096 rows unnamed) |
| FLUX.2 klein | [image, refs] or, when not a multiple of 512, [text 512, image, refs, zeros] | the double blocks' weights differ for text and image and the boundary moves with the order, so the text-first case has `doubles_tROWS` (same banks). singles, `dit_in`, `dit_out` depend on rows only and are shared. Text-first vs normal at 1024²: PSNR 36.9 dB (key-block order rounding) |
| FLUX.3 klein | singles: [image P, text slot 512]; image mode blocks: P rows | image rows zero-padded to P (`image_rows`, 4096 for all 7 large sizes); keys repacked as [N image, text R] (`parallel(keys=...)`), so one graph serves every prompt length (text first would interleave non-key rows). `dit_in`, image mode blocks, `dit_out` depend on P only and fall back to the first size's (`flux3klein.ironPart`); `singles` and the VAE are per size |

Padding changes nothing: a 2:3 FLUX.2 picture at 4608 and 5120 rows is byte-identical. Large sizes all run 4608 rows (as 1024²), so their time and memory match 1024².

## Per model

### Anima-Turbo, Anima-Aesthetic

- DiT: 28 blocks, 7 banks of 4; bfp16 weights, `qk`, exact exp, `trim` cross-attention, GELUEven. `dit_in` once a step; CFG's two passes per group. Empty negative from `text/empty_cond.bin` (`iron_anima.empty_cond`, `anima.EmptyCond`).
- Export of the eight sizes (`--size 1024 512 832x1248 880x1168 752x1360 1248x832 1168x880 1360x752`, `--dit anima-aesthetic-v1.1.safetensors` for Aesthetic): 6 min 38 s, peak RSS 7.1 GB, 5.8 GB. Only the DiT can be remade (`--only dit`).

| | Time | Peak (NPU + RSS) |
|---|---|---|
| Turbo 1:1 1024², 8 steps | 32 s (3.2 s a step) | 2.00 GiB (NPU 1.21) |
| Turbo 1:1s | 9.9 s | 1.74 GiB |
| Turbo rectangles | 31–32 s | 2.04–2.11 GiB |
| Aesthetic 1024², 25 steps, CFG | 2 min 22–23 s, 2.71 kJ | 2.14 GiB |

- Energy: Turbo 1024² 572–615 J (package, `HWNPU_ENERGY`). Accuracy (Turbo 1024², diffusers fp32): latent cosine about 0.975–0.98.
- models.yaml's AMD memory 3.5 GB (NPU peak 2.4 GiB in the VAE + process 0.9 GiB).

### Chroma1-Flash, Chroma1-HD (removed)

Removed on 2026-11-24: the DiT, the text mask, the export and its numbers are in [not-adopted/chroma1.md](../not-adopted/chroma1.md).

### FLUX.2 klein 4B

| Graph | Content | Calls |
|---|---|---|
| text | Qwen3-4B, 3 layers × 9 groups, 512 rows in one call | 9 a prompt |
| ctx | context_embedder | 1 a prompt |
| mods | modulation rows (17 × 3072) from the time embedding | all steps' rows before the steps (0.3 GB not held) |
| dit_in | x_embedder (picture and reference rows) | 1 a step |
| doubles, singles | 1 double × 5 groups, 2 singles × 10 groups | 15 groups a step |
| dit_out | norm_out, proj_out (picture rows) | 1 a step |
| vae/s0..s9 | decoder (`iron_chroma_parts.FluxVaeStage`; s0's 1 × 1 is post_quant_conv) | 1 a picture |
| vae_enc/s0..s5 | encoder; conv_out and quant_conv folded into one conv, the mean's 32 channels only | 1 a reference |

- DiT: width 3072, 24 heads × 128, no biases, SwiGLU MLP (3×), three modulation linears shared by every block; RoPE 4 axes × 32 (reference index, row, column, text position), theta 2000.
- Block graphs depend only on a step's rows (positions are inputs) and are named by them (`_nROWS`; the first size's unnamed); a full-size reference and four half-size ones at 1024² are both 8704 rows and the same ELF. Encoders by side: the first size's `vae_enc`, `vae_enc_dD` at 1/D of its side, else `sizes/WxH/vae_enc`.
- Default reference combinations (`default_refs`, `--refs N`, `NdD`, `A+BdD`): up to 4 references of full and half size whose rows are a multiple of 512, up to 12800 rows. More rows compile and run if named (`--refs 3 4`). Each row count is ~12 MB and ~70 s of compile.
- **Step time** ≈ 0.70 ms·n + 71 ns·n² for n rows (projections linear, attention quadratic): 4608 rows 4.75 s, 12800 rows 21 s. Up to 12800 rows the VAE is the peak (3.9 GB); past it the DiT banks (0.088 MiB a row in a double) grow it (20992 rows: 5.6 GB).
- Export of all eight: 12 min 58 s, peak RSS 19.1 GB, 16.0 GB (rectangles' VAEs are larger: 16-row chunks mean more control code).

| | Time | Peak |
|---|---|---|
| 1:1 1024², 4 steps | 28 s | 3.60 GiB |
| 1:1s 512² | 10 s | 1.90 GiB |
| rectangles | 27–28 s | 3.53–3.64 GiB |
| 1024² edit, 1 full reference | 59 s | 3.9 GB |
| 1024² edit, 1 half reference | 37 s | 3.8 GB |
| 1024² 2 full references | 1 min 41 s | 3.9 GB |
| 1024² edit, 2 half references (6656 rows, 8.5 s a step) | 47 s | 3.8 GB |
| 1024² edit, 3 half references (7680 rows, 10.4 s a step) | 56 s | 3.8 GB |
| 512² 2 full references | 20 s | 1.6 GB |

- References without a size asked (2026-11-25): models.yaml `npu.amd.edits`, the same as `npu.apple.edits` (1024²: 1 full, 2 half, 3 half; 512²: 1 or 2 full), so one call draws one picture on both. Another number at those sizes is refused (as on Apple); `--image FILE@half` still picks any combination the export holds; the rectangles take any number of full ones. The times above are wall clock of `hwnpu-image generate` (1 full reference: 13.1 s a step, 1 min 8 s).

- Accuracy against CPU float32: 1024² generation latent 0.961, 512² 0.975; edits 0.996–0.999 (2 full references 0.9989 / 40.0 dB). The remaining gap comes from the text's pad rows.
- Simulated per-format error in the DiT (first-step velocity): bfp16 weights 0.99992, bfp16 linear inputs 0.99989, bfp16 q/k 0.99989, all three 0.99964.

### FLUX.3 klein

- Exporter reads the single-file DiT `model.safetensors` (BFL's names, 11.1 GB; renamed as diffusers 0.43.0's `convert_flux3_transformer_checkpoint_to_diffusers`, `norm_out` rows swapped to scale, shift) when the diffusers layout is absent (`AMDFiles`). Modulations (14 rows) on the host (`time.safetensors`). DiT weights bfp16, `qkpv`, mid exp.
- Export of all eight: 10 min 45 s by hand, 11 min 27 s from `PrepareAMD`, peak RSS 16.8 GB, 15 GB (the 7 extra sizes add only `singles` and `vae`, 0.2–0.5 GB each).

| | Time | Peak |
|---|---|---|
| 1024², 16 steps | 1 min 43 s (text 2.8 s, 5.9 s a step, VAE 6.5 s) | 3.59 GiB |
| 512² | 32 s (1.7 s a step) | 2.49 GiB |
| 3:4 | 1 min 43 s (5.7 s a step) | 3.61 GiB |

- Accuracy: 512² latent 0.9985 / 39.4 dB (a 91-token prompt 0.984 / 32.2 dB: details diverge in free runs); 3:4 against diffusers bf16 0.990 / 34.1 dB.

## Measuring

- Peak memory: every 0.1 s, `drm-total-memory` of the accel fd in `/proc/PID/fdinfo` (amdxdna's buffers) plus `RssAnon + RssShmem`; the max of the sum.
- Energy: `HWNPU_ENERGY=10 hwnpu-image …` (package joules by phase, idle subtracted). See [memory-and-power.md](memory-and-power.md).
- No other user of `/dev/accel` (another session made a layer 52 → 500 ms). Measure with the page cache warm unless testing cold starts.

## Open

- Anima's mid exponent against float32 (which is closer is unknown).
- Anima: the cross-attention's K/V out of the step loop (0.7 % of the work), same-shaped VAE stages in one graph (`up_blocks.3`'s three ResBlocks; the VAE is ~12 % of Turbo).
- The self-attention: P·V (bf16) and softmax balance; then a fixed max bound.
- The projections' and MLP's GEMMs and norms (about 60 % of a block): kernel work.
- `bfp.encode` in parallel (export time).
- FLUX.2 references: cropping without reshaping, `@quarter` (the run handles it by length; encoder and flag missing).
- Energy per picture for FLUX.2 and FLUX.3 klein; editing with FLUX.3 klein.
- The time of exporting all eight sizes from an empty kernel cache.

## Not adopted

| What | Why |
|---|---|
| AIE2P's exp2 instruction | as fast as one limb, error 1.6× (FLUX.2 latent 0.9645 → 0.9629, 27.1 → 25.7 dB); 4× in Anima's MHA |
| IRON's tanh GELU | the argument's bf16 products (0.4 % each) changed the clothes in a test picture |
| x·sigmoid(2u) GELU with gpt-oss's `gl_exp32` | accurate to 2e-4 but as slow as the exact kernel (5.5 ms) |
| `qkpv` for Anima | 1024² −5 % time, −12 % energy, but details moved further from the reference (chest's shape changed); every export would be remade |
| Gate and residual (`x + y·g`) in one operator | 3 inputs exceed the core's 2 input DMA channels; a 2-core pipeline buys ~1 % |
| adaLN folded into `dit_in` | ~1 % |
| A resident array (gpt-oss's GptOssStep form) | operators take ms each; switching ~5 % at 1024² |
| Anima DiT at q6 / GPTQ q4 | q6 median latent 0.912 vs q8 0.957; q4 changed compositions (one knight became two) |
| bfp16 (MEDIUM) text encoder for FLUX.2 | pad rows 0.983 at layer 18; the DiT is sensitive to them |
| FLUX.2 text encoder in 64-row chunks | a pad row's own key enters the causal pass |
| Last single block on picture rows only | 2 % of an edit step; another graph and bank-sharing complexity |
| GroupNorm stats as sum of squares in one pass on bf16 vectors | cancellation 40×; sums stay float32, the subtraction in float64 on the host |
| FLUX.3 short text slot (80–240 rows) | the MHA's 8 pipelines take only multiples of 512 rows; ~1.8 s a picture |
| Batching the VAE across pictures | copies stage outputs to the host (0.5–0.8 s a picture) and raises the peak |
| Prefetching the next VAE stage | it only hid buffer allocation, which buffer reuse removed |
| Larger conv row blocks (64 → 128, 256) | 38.0 → 36.5 ms: not a switching cost |
