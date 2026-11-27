# Image models on Apple and Intel

[日本語](image-apple.ja.md)

How the image generation models run on Apple's Neural Engine and on Intel's NPU: the programs of each model, the DiT attention forms, the fp16 traps, the weight formats, how programs are loaded, and the current numbers. The catalog, samplers, CFG, aspect ratios and which models stay are in [image-models.md](image-models.md). The AMD forms are in [image-amd.md](image-amd.md). Core AI mechanics (`.aimodel`, cache, prebuilt, NE compile) are in [apple-coreai.md](apple-coreai.md), Core ML limits in [apple-coreml.md](apple-coreml.md), Level Zero and OpenVINO IR in [intel.md](intel.md).

## Backends and where each model runs

| model | Apple (Core AI) | Intel NPU | package |
|---|---|---|---|
| Anima-Turbo, Anima-Aesthetic | yes | yes (OpenVINO IR, not verified end to end on hardware) | `internal/imagegen/anima` |
| FLUX.2 klein 4B | yes (text to image; editing with the edit's programs, below) | no | `internal/imagegen/flux2klein` |
| FLUX.3 klein | yes (text to image) | no | `internal/imagegen/flux3klein` |

- On a Mac, `hwnpu-image` runs Core AI only (macOS 27). `--backend coreml` or `HWNPU_BACKEND=coreml` also opens Core AI (`imageDevice` in `cmd/hwnpu-image/backend.go`). A Mac without Core AI has no device for `hwnpu-image`.
- Core ML exports are still written by `hwnpu-image convert --backend coreml`. They are what `--cpu` runs: the CPU interpreter (`milcpu`) in float32, the reference where no PyTorch runs. `HWNPU_DEV=cpu.f16` rounds every op to fp16 on the CPU.
- Each model has one MIL description (`internal/imagegen/<model>/mil`). It is written as a Core AI `.aimodel`, as a Core ML package, or as OpenVINO IR (`ovlower`, Anima only). The pipeline (tokenizer, sampler, CFG, batch) is shared with the AMD path. Only the engine (`progEngine` in `programs.go`) differs.
- First run: `Prepare` downloads the missing checkpoint files (`chat.Fetch`, sha256 pinned in [../../models.yaml](../../models.yaml)), converts in Go only, and writes `NAME.done` last. The export name holds what changes the programs, including `ExportVersion`; a new layout gets a new name and is converted again. Old exports stay until removed by hand.
- Every model takes the Neural Engine lock before it opens an Apple export (`llm.LockNPU`, "waiting for the Neural Engine: process N … is using it"). Without it, another hwnpu process on the NE made VAE stages fail with `Inference failed — request aborted (underlying=0x15)`. See [apple-coreml.md](apple-coreml.md).

## Exports: one per model holds every size

The rule (see [image-models.md](image-models.md) and [new-model-checklist.md](new-model-checklist.md)): one export per model and NPU vendor, with every aspect ratio of models.yaml and the weights stored once. No export per size, and no export of only some of the sizes.

- Apple form: a part whose graph depends on the size is one `.aimodel` with a function per size, `s{W}x{H}` (`llm.WriteAIFunctions`: identical weight bytes stored once). The manifest's `sizes` lists the part names per size, and `ProgramManifest.At` picks them. The size part of the name is `aspect.Tag` (three or more sizes: `8sizes-` plus the CRC-32 of the sizes). One prebuilt per chip per export holds all sizes.
- What it costs (accepted: duplicate downloads cost more):
  - Core AI compiles every function of an `.aimodel` together. The first run compiles all sizes. Where the VAE is not prebuilt (M1, and Anima on every chip: below), its sizes are compiled on the device on the first run.
  - If the NE compiler rejects one size, every size of that part leaves the NE. Check each chip and each size offline with `coreai-build` before placing an export.
  - Adding a size rebuilds the export and every prebuilt.
- Parts that do not depend on the size (text encoders) are one function. Parts shared between models or exports are hard links (`ConvertOptions.LinkFrom`, `linkFrom`, `TextVersion`), so Core AI's cache hits them and they are not compiled again. The cache keys on program bytes, and each conversion writes a new UUID into the package, so separately converted copies of the same part would be compiled and held twice.

Current state (migration is open):

| model | Apple export now |
|---|---|
| Anima | one export per size (`anima-turbo-v1.1-WxH-int8-int8-g4-q512-v4.coreai`; `-v5` for the sizes whose patches are not a multiple of 8) |
| FLUX.2 klein | 512² + 1024² in one (7.5 GB, against 7.3 + 7.5 GB apart); the other six sizes of `npu.apple.sizes` separately |
| FLUX.3 klein | 512² and 1024² × text rows 80 and 160 in one (`DefaultSizes`, `DefaultTextRows`) |

Migration order: Anima-Turbo → Anima-Aesthetic → FLUX.2 klein (the eight sizes of `npu.apple.sizes`) → FLUX.3 klein (Chroma1 was removed: [not-adopted/chroma1.md](../not-adopted/chroma1.md)); chips M1, M2, M3 (h13g, h14g, h15g) first.

## What is not prebuilt

`coreai.LeftToDevice` keeps programs whose names start with `text` (and those ending in `_device`) on the device on every chip: `hwnpu-dev models package MODEL/coreai-ARCH` leaves them out ("left to the device") and `coreai.Prebuilt` ignores such prebuilts already in a source. The VAE is the recipe's choice per model and chip (`model-recipes.yaml` apple `prebuild`, `do: device`):

| part | chips | why |
|---|---|---|
| FLUX.2 klein VAE (`vae_s0` on M1: its attention's softmax reads past its kernel, `llirtasks -check`), FLUX.3 klein VAE (the same decoder; every stage on M1 and M2 until FLUX.2's are run there) | M1 (`h13*`), M2 (`h14*`: FLUX.3 alone) | `coreai-build`'s FLUX.2 klein VAE for h13g drew stripes a few rows high every 112 rows at 512² (the 1024² function of the same `.aimodel` was right). A 1024 × 768 `vae_s0` failed with `ANEProgramProcessRequestDirect() … Request cancelled`. No gain there either: the M1 compiles the ten stages at 1024² in 104 s. On M2 every picture at 512² had bands 8 to 16 rows high every 112 rows (rows 112, 224, 336, 448), from the h14g prebuilt and from the device's compile alike, FLUX.2 and FLUX.3 klein; 1024² was right (2026-11-26). Left to the device on M2 too until the program is fixed |
| Anima VAE (`vae_*`) | M2 (`h14*`) | Up to export v5 `coreai-build` failed 8 of the 10 stages on every chip (the native silu, below); since v6 all are placed on the Neural Engine offline for h13g and h14g too. On M2 the pictures from the h14g prebuilt had bands every 160 pixels (and twice or three times as often at some aspect ratios), none compiled on the device (2026-11-27). Prebuilt on M1 (drew right) and on M3 and later (the same bytes as compiled on the device, 2026-11-26) |
| text encoders (`text_*`) | M1 (`h13*`), M2 (`h14*`) | Prebuilt on M3 and later since 2026-11-26, for the memory: the device's compiler grew by 1.5 GB compiling FLUX.2 klein's nine Qwen3 programs, the prebuilts' finishing service by 0.5 GB; the caches took 7.9 GB either way. Their first load took 46–190 s from prebuilts against 46–52 s compiled on the device (it varied from run to run). Fresh programs (never compiled on the machine) gave the same bytes from the prebuilt as from the device's compile. Left to the device on M1 and M2 until they are run there (Chroma1's T5, removed, ended in `Request cancelled` from h13g prebuilts) |

- FLUX.2 and FLUX.3 klein's VAE is prebuilt for M3 and later. On an M3 the device's compiler took 2799 s over the ten decoder stages at the first run (`vae_s6`–`s8` about 600 s each, every size at once), finishing the prebuilt 0.9 s; the outputs of every stage were the same bytes at 1024² and 512² (2026-11-25). A whole run on the M3 (2026-11-26): FLUX.2 klein's first decode at 1024² 10m32s → 4.5 s and a 512² edit's encoder 430 s → 20 s; FLUX.3 klein's first run at 1024² 19m27s → 2m43s (the VAE 640 s → 1 s). Every picture was the same bytes as with the VAE compiled on the device. A real picture through the VAE at all eight sizes of FLUX.2 klein was the same bytes from the prebuilt and from the device's compile, without bands (2026-11-26). FLUX.3 klein's VAE is left to the device on M1 like FLUX.2 klein's, untried there.
- A machine that already drew with a prebuilt VAE on M1 keeps the finished entry in Core AI's cache (`hasEntry` wins over prebuilt) and keeps the stripes. Remove `~/Library/Caches/coreai-cache/<OS build>/hwnpu-image/` (the VAE entries are enough).
- Open: what makes `coreai-build` wrong on the VAE for h13g (the 112-row period may be a row split inside a stage), and what cancels T5. Drop the recipe's rules when fixed. Report to Apple (FB).

## Patterns shared by the models

### DiT attention (Core AI)

At 1024² a DiT attends over 4096 image tokens. The time is in the element-wise passes over the scores (scale, max, subtract, exp, sum, reciprocal), not in the products: Core AI writes softmax as `exp(x − max) × reciprocal(sum)` (Core AI's own attention op does not run on the NE). One pass over 16 × 512 × 4096 scores is about 11 ms. So the forms below cut passes or change the layout.

| form | where | effect (M1, 1024², one block's attention core) |
|---|---|---|
| attention scale 1/√d in the queries' RMSNorm weight; `SDPAScale` 1 puts no multiply | every model (q is RMS-normed); FLUX's text paths read the same weights | Anima self-attention 102 → 89 ms, a 4-block group −4.7 %; the same error against milcpu (RoPE is linear) |
| `milpart.HeadAttention`: a head at a time, a chunk's scores reshaped to (1, queries, 1, keys), softmax over the last axis | image self-attention, chunks of 1024 queries (`HeadChunk`), division before the product with V | Anima 90 → 72 ms. Both conditions are needed: every head in one tensor of that layout is slower than before, and so are smaller chunks alone |
| keys split into power-of-two parts of 4096 at most (`milpart.KeyParts`), image keys and text keys as separate score tensors (`HeadOptions.Split`) under one softmax (max of maxima, sum of sums); the text pad bias on the text part only | FLUX.2 klein, FLUX.3 klein (text rows in the same attention); first on Chroma1 (removed) | per score 0.27 ns at exactly 4096 keys, 0.38–0.47 ns at 2304…8192. Chroma1 205 → 139 ms (24 heads, 4608 keys); a double group 1.68 → 1.40 s, a single group 3.47 → 2.91 s |
| cross-attention: every query at once, the sum divided after the product with V (`Deferred`) | Anima only (512 text keys) | 14 → 10 ms. Safe because the product before division is at most 8.7; `TestBlockRanges` fails under 8× headroom |
| `Deferred` queries padded with zero rows to a multiple of `milpart.DeferredRows` (8), dropped after (`paddedDeferred`) | Anima | M2's and later NE compilers (h14g, h15g, h16g) put a `Deferred` attention with a query count not a multiple of 8 on the GPU (`no compiled region`); h13g took it. Fixed sizes: 880 × 1168, 752 × 1360 and their turns (export `-v5`, `PaddedExportVersion`); the other sizes stay `-v4` byte for byte. M2 3:4: 4.3 s a step, all on the NE |

Rules learned:
- Do not defer the division in self-attention: the product reaches fp16's max in Anima's last block (27). Scaling V's int8 weights by 1/256 to avoid it changed the pictures (products under fp16's normal range are dropped by the NE).
- Writing that left the NE: two chunks divided by the same sum (use the sum's reciprocal), `maximum` of maxima of different widths (`Interleave should be matched for dual-input PEs`; concatenate the maxima and `reduce_max` again). Qwen-Image's query chunk of 2048 also left it.
- Intel always uses the heads-together form (`QChunkAttention`).
- A single score tensor over about 64 MiB is 2–3× slower per score on Core ML's NE (a step between 65 and 68 MiB). This is why `mil.DitQChunk` halves the query chunk at conversion until 16 × chunk × patches × 2 B ≤ 64 MiB (Anima's heads-together form: 1024² 512, 1280² 256, 2048² 128). The size of the on-chip memory is not the cause: per-score speed is flat from 0.5 MB to 64 MiB.

### fp16 traps and their fixes

| trap | fix | where |
|---|---|---|
| residual streams pass fp16 (Anima DiT 2.7e5, Chroma1's T5 2.8e5 and DiT text rows 2.4e5, FLUX.3 text stream 9.7e6) | hold the stream scaled: Anima `DitScale` 1/32 (patch embedding and gates ×1/32); FLUX.2 klein DiT 1/32, `TextScale` 1/8; FLUX.3 image 1/32, text `TextStreamScale` 1/1024 with the gate adding the block output (1/32) into the text times 32/1024 | every DiT and text encoder (Chroma1, removed: `T5Scale` 1/64, `DitScale` 1/32) |
| squares in RMSNorm/LayerNorm overflow; a fixed scale makes small rows subnormal | divide each row by its largest magnitude first (`milpart.AbsScaled`, `AbsScaledLayerNorm`; `abs` is one op, the older `MaxScaled` used `maximum(x, −x)`) | all norms in DiTs. It cannot become a constant: Anima's q and k reach 384 and their row RMS spans 1400× (measured by `mil.BlockRanges`), weight-derived scales still leave 700× |
| dividing by the max leaves outlier-free channels near subnormal (T5 rows have one channel 1000× the rest) | take the mean square from the divided copy, build the output from the original x times a scalar (Chroma1's `mil.t5Norm`) | Chroma1's T5 (Chroma1, removed: [not-adopted/chroma1.md](../not-adopted/chroma1.md)) |
| a scale folded into weights makes products under fp16's normal range (6e-5); the NE drops them | keep scales on outputs: o's 1/64 after the product; wo 1/8 in the weight and 1/8 after; norm output ×16 before the projection and ÷16 after (`t5Up`) | Chroma1's T5, removed (three prompt rows were at cosine 0.93, 0.75, 0.36 before) |
| Core ML folds constant multiplies around a linear back into the weight | the `t5Up` trick only helps when an add follows the scale; on the DiT it changed nothing and broke the last single group (reverted) | Chroma1's DiT (removed) |
| untyped integer constant division: `1/64 = 0`, `8/32 = 0` | write float constants; checked by unit tests (`TestCtx`) | Chroma1's T5 FF output vanished (removed); FLUX.2 klein text rows all zero |
| per-row int8: the NE seems to accumulate the signed products and apply the row scale last, so the accumulation passes fp16 on large inputs → inf/NaN | lut8 (table expanded to fp16 weights before the product). The same int8 program is right in the CPU interpreter | Chroma1's T5 layers 4–11 and late double blocks (removed); Qwen3-VL attention sinks (NaN); hence lut8 for every text encoder with attention sinks and for the FLUX DiTs |
| a dimension over 16384 runs on the CPU | split: FLUX.3 `context_embedder` input 20480 as two inputs and two half linears; FFN `linear_in` 18432 read as `#gate` and `#up`; Sana's KV output put layers on axis 1 | |
| a program of one row (Anima `dit_time`, input (1, 1, 2048)) or a tiny program (`dit_patch`, one linear) runs on the CPU | 8 rows (`mil.TimeRows`, sinusoid in row 0); put small parts into a neighbor group (patch embedding into `dit_blocks_0`, norm_out/proj_out into the last) | |
| modulations from one timestep row drift on the NE | compute on the host in float32 (FLUX.2/3 klein's time embedding and shared modulation, `time.safetensors`, once per σ) | |

### Weight formats on Apple

| part | format | reason |
|---|---|---|
| Anima text (Qwen3-0.6B) and DiT | int8 / int8 | DiT int8 nearer fp16 than lut8 (latent cosine 0.902 against 0.858, M3); on Core AI a block without attention cores is int8 97.4 ms, fp16 111.9, lut8 111.6 (the NE pays nothing to expand int8 and reads half the bytes); fp16 doubles what the NE holds (2.62 → 5.2 GiB) |
| FLUX.2 klein Qwen3-4B, DiT | lut8, lut8 | attention sinks at int8; lut6 changes the picture |
| FLUX.3 klein Qwen3-VL, DiT | lut8; DiT tables fitted to activations (`lut8x2`) | below |
| context embedders (FLUX.2 `Ctx`, FLUX.3 `ctx`), VAE | fp16 | run once per prompt or picture: more bits cost nothing at steps |

- Speed: on Core AI, int8 vs lut8 depends on the shape. Anima int8 was 13–17 % faster than lut8; Qwen-Image's (hidden 4096) was not (369 vs 361 ms a group, 6.2 TFLOPS, the NE's ceiling). Measure on the model's group before choosing a format for speed.
- 4 bits is not enough for distilled DiTs (Chroma1's MLP at lut4 changed poses or collapsed; removed).
- See [quantization.md](quantization.md) for the formats.

### Loading programs: swapping DiT groups and preloading

- On Core AI the DiT groups are not held. Each group is loaded for its turn, run, and closed (`runner.Swap`; Anima `progEngine.swap`, FLUX `runner.Ring`). While a group runs, the next one loads (`runner.InTurn` for in-turn parts, `runner.Ring` for the DiT: after the last group it loads the first for the next call; an unused load is closed). At most 2 programs open. A failed ahead-load is retried once at its turn (`TestRing`, `TestInTurn`).
- Why: a load from Core AI's cache is 0.02–0.3 s, about one run. Holding everything is no faster, because a large group that follows another program pays a switch cost anyway (`aicheck cycle`: a 4-block Anima group after another group +50–70 ms; 2 small programs or a group plus a small one +0). And holding a lot makes things slower or fail: Chroma1-Flash 1024² holding all 8.8 GiB is 10 % slower; Qwen-Image's lut4 DiT held at 5.2 GiB failed with `underlying=0x15` (Core AI errors instead of evicting like Core ML).
- Text encoders are closed after the prompt, the approximator before the steps, the groups loaded for the last step's next before the VAE. VAE stages are loaded one at a time (10–20 ms each, no preload needed).
- Core ML and Intel load one at a time (Intel holds what fits and closes others when out of memory).
- A first call after a load (or after another program) is slow: a VAE stage and a text encoder group always run as first calls.
- Several prompts together (`--batch`, `GenerateAll`, `DecodeAll`, `EncodeAll`): the text once for all prompts, each DiT group runs every picture, each VAE stage every picture. See [image-models.md](image-models.md).
- Memory at 1024² peaks in the last VAE stages (process float32 buffers), not in the DiT. FLUX.2 klein drops the time embedding weights (0.7 GB) after building every σ's rows and returns step garbage before the VAE and after each stage (`debug.FreeOSMemory`): 5.8 → 4.1 GB footprint.

| measured on M1, 1024² (Core AI) | hold all | swap, no preload | swap + preload (default) |
|---|---|---|---|
| Chroma1-Flash (removed): a picture / NE held | 6 m 16 s / 8.80 GiB | 6 m 11 s / 1.80 GiB | 5 m 36 s / 2.53 GiB |
| Anima-Turbo: a picture / NE held / process peak | 49 s / 2.53 GiB / 3.88 GB | — | 49 s / 0.57 GiB (0.84 with two groups) / 3.32 GB |

Pictures are byte for byte the same in all forms.

### Text encoders

| model | encoder | programs | notes |
|---|---|---|---|
| Anima | Qwen3-0.6B | `text` (one program) | 1.0 GB at fp16 was over a Core ML program's limit; int8 |
| FLUX.2 klein | Qwen3-4B, first 27 layers | 3 layers a program, 9 (`mil.TextGroups`: the hidden states after layers 9, 18, 27 each one output); `Ctx` concatenates them | 512 rows; prompt rows causal, pad rows see only prompt keys (`kbias` + causal mask). The DiT reads pad rows, so they cannot be skipped |
| FLUX.3 klein | Qwen3-VL-4B language part, 32 of 36 layers | the hidden states after layers 4, 8, …, 32 concatenated (20480) | text rows padded to a multiple of 80 (80 or 160); pad position ID 1; the same mask as FLUX.2 klein (checked against transformers 5.23) |

- Text encoders run once a prompt: the residual scale holds the attention sinks (Qwen3 21700 in FLUX.2's ctx; Qwen3-VL 11264).
- Prefix K/V as program constants (`mil.TextFrom`, `mil.TextKV`) was built for Sana. None of the remaining models uses it: FLUX.2 klein's chat-template prefix is 3 tokens.

### VAE

- Run stage by stage (Anima 10, FLUX 10), loaded per stage, closed after. Each stage reads the previous output directly.
- Upsampling is `upsample_nearest_neighbor` (Anima stage 5 alone 1275 → 412 ms, 3× faster than reshape + tile); OpenVINO IR keeps reshape + tile.
- FLUX's GroupNorm (`fluxvae.GroupNorm`): group values as (1, 32, a, b) (every axis ≤ 16384), divide by the group max, subtract the mean before the squares (|mean|/std reaches 6.3). 512² VAE vs diffusers fp32: 66.3 dB.
- FLUX.2/3 klein: FLUX.1's decoder (`internal/imagegen/fluxvae`, once Chroma1's) plus `post_quant_conv` (`fluxvae.Vae.PostQuant`); batch-norm denormalization and the 2 × 2 unpack on the host.
- Anima's VAE values stay under 8: no fp16 care.
- Anima's VAE for Core AI writes the residual blocks' silu in the exp form (`blocks.SiLU`, export v6, `anima.CoreAIExportVersion`). With the Neural Engine's own `silu` after both of a block's norms, `coreai-build` (Metal Toolchain 27.2.47) failed every stage with a residual block on h13g, h14g and h15g (`ANECCompileOffline: CompilationFailure`; norm, silu, conv, norm, silu is enough; exp after the second norm alone still failed s0–s2 and s4). With it all 10 stages × 8 sizes are placed whole on the Neural Engine for the three chips. M3, 512²: the device compiled the ten stages in 482 s (the native silu's: 3552 s, up to 12 GB of temporary disk a stage), from the h15g prebuilt they load at once; prebuilt and device outputs are the same bytes and every stage matches an fp64 reference from its own input at 75–89 dB, 1024² too (2026-11-26). Core ML and OpenVINO IR keep the native silu.
- Anima's prebuilt on M1 (but `vae_s0`) and on M3 and later, on the device on M2; FLUX.2 and FLUX.3 klein's prebuilt on M3 and later, on the device on M1 and M2 (above).
- Passing fp16 between stages instead of float32 gave nothing (0–0.3 s, Sana 1024²).

## Anima (Cosmos-Predict2 2B, Qwen3-0.6B)

- Programs: `text`, `dit_time` (8 rows), DiT in groups of 4 blocks (`dit_blocks_N`, 7 groups of 28 blocks; the patch embedding in `dit_blocks_0`), the final layer, VAE 10 stages. Formats int8/int8 on Apple (`anima.Formats`), fp16 on Intel.
- Attention: self-attention `HeadAttention` 1024 chunks, division first; cross-attention whole, division deferred, queries padded to 8. Scale in q's norm weight. Norms by `abs`. `ExportVersion` 4, `PaddedExportVersion` 5, `CoreAIExportVersion` 6 (every Core AI export: the VAE's exp silu).
- `--group` (default 0 = `anima.DefaultGroup`): 4 on Apple. `--qchunk` 512 (Core ML/Intel heads-together form).
- Breakdown of a 1024² block (M1, Core AI, before the attention forms): 232 ms; MLP 54 ms (5.1 TFLOPS), self-attention core 102 ms (1.3 TFLOPS), cross-attention core 14 ms. Element-wise ops other than attention are about 24 ms of a 221 ms block (11 %): q/k RMSNorm 11.3 (4.8 of it the max division), RoPE 3.4, LayerNorm max divisions 3.7, exact GELU 5.8 (tanh and sigmoid approximations are slower: +4.7, +12.5 ms).
- Its CFG step costs exactly twice a plain step (11.1–11.2 vs 5.6–5.7 s): the groups are small (0.28 GiB), so it pays no switch cost, and batching groups does not speed it up.
- Accuracy: Anima-Turbo 512² against a CPU float32 run of the same int8 weights: latent cosine 0.987–0.999. Against diffusers fp32 at 1024² (M1, Core ML, int8): 0.876–0.948, same composition; the 8-step distilled model amplifies small differences. Anima-Aesthetic (25 small steps) stays close (0.997–0.9999 between attention forms).

## Chroma1 (removed)

Chroma1 (Flash, HD) was removed on 2026-11-24: its programs, formats and numbers are in [not-adopted/chroma1.md](../not-adopted/chroma1.md).

## FLUX.2 klein 4B

- Programs: Qwen3-4B 9 groups (lut8), `Ctx` (fp16), DiT 2 double blocks a program (3) and 5 single blocks a program (4) at lut8, VAE.
- DiT in the form of Chroma1's (removed): no biases, SwiGLU, modulation shared across blocks (double image 6 rows, text 6, single 3, norm_out 2) computed on the host per σ; fused single-block projection read by part names (`to_qkv_mlp_proj.weight#q`, `mil.Weights`); RoPE 4 axes × 32, θ 2000, text rows on axis 4. Attention a head at a time, image and text keys apart, no mask (the DiT sees pads).
- 512² + 1024² in one export; latents bit for bit as per-size exports.
- Accuracy (512², first-step velocity, against the same programs in float32 on the CPU, `flux2klein.TestF32Check`): 3.8 % rel. RMS = lut8 2.7 % + NE fp16 2.8 %; per group 0.2–0.8 %, accumulating over 25 blocks. No diffusers comparison (no PyTorch on that Mac).
- Does not apply: CFG (distilled), text K/V out of the steps (text rows are modulated each step), fixed prefix (3 tokens), fewer text rows (the DiT sees pads).

### Editing: in the same export (`edit.go`)

- An edit needs programs text to image does not: the DiT's groups over the picture's rows and the references' (`mil.Dit.Refs`; 1024² with one full-size reference is 8704 rows, against 4608) and the VAE's encoder. They are **in the one export** of every size: models.yaml `npu.apple.edits` names the edits by aspect ratio (`ConvertOptions.Edits`, `AppleOptions`), and each DiT program holds a function for each size and each edit step (`EditFunction`, `s1024x1024_ref1h2`), the weights once: 8 sizes + 5 edits = 13 functions (2026-11-22). An edit of 8192 keys or more (the DiT's rows: at 1024² a reference of 1024 is 8704) is not in them: coreai-build leaves an attention over that many keys to the GPU on every chip (Metal Toolchain 27.2, `NEFUSED_EW: KernelPreScales`; the keys in pieces of their own, or each piece's softmax by itself and the pieces weighed after, fail the same; 8064 keys compile), and one such function would cost the whole program its prebuilt. Those steps are programs of their own, a function each (`dit_double_0_device.aimodel`, `flux2klein.PrebuiltKeys`, `DeviceSuffix`): `coreai.LeftToDevice` leaves them to the device's compiler, which takes them, at the first such edit. They hold the group's weights again (about 4 GB more). So the main programs are 10 functions (8 sizes, the two edits at 512²) and prebuilt; the three edits with a reference of 1024 compile on the device. Export layout v2. The edits: at 1024² from one reference of 1024, 1024 + 512, or 1024 + 512 × 2; at 512² from one or two of 512. The encoder's stages are programs of their own, a function for 1024² and for 512² (a reference of 512 is the same function at either picture). The manifest lists them under `edits` (`EditSize`); the name gains `-5edits-HASH` (`flux.2-klein-4b-8sizes-4d02f40a-5edits-fa7f0afe-lut8-lut8-v1.coreai`).
- Since v5 (2026-11-24) an attention over 8192 keys or more joins its parts by their log-sum-exp (`milpart.PieceKeys`), which coreai-build compiles for the Neural Engine, so the edit at 1024² from a reference of 1024 (8704 keys) is a function of the same programs too and the `-device` export is gone: 8 sizes + 5 edits = 13 functions, all prebuilt on an M2 and an M3 (v6: two single blocks a program).
- **An M1 has an export of its own** (2026-11-25, models.yaml `npu.apple.generation_edits`): its one reference of a picture of 1024² is read at 512 (`1@half`, 5632 keys), so that none of its 13 functions is of 8192 keys or more. coreai-build's h13g program of the 8704-key function is wrong on the M1 ([known-issues 13](../known-issues.md)): the single blocks' finish and compute noise, the double blocks' crash the service that finishes them, and Core AI then compiles the whole program on the device (54 min for one of 13 programs). The same function compiled on the M1 is right, and so are coreai-build's of 4608 and 6656 keys. Splitting that edit into programs of its own (the `-device` export of v2 to v4) would have kept it at the cost of 3.9 GB more and about two hours of compiling at its first use; the M1 is the machine with the least memory and storage, so it reads the reference smaller in place of that. `--image FILE` picks the reference's size from the export's table by itself (`ai/imagegen`: the log says `reference 1 at 512x512`); the picture is still 1024². The other chips' export and AMD's are as before. `hwnpu-dev models package` prebuilds each chip's export for the architectures that take it alone (`chat.TakesExportsOf`).
- Until then the edit was an export of its own (`ConvertEdit`, `PrepareEdit`, `Pipeline.UseEdit`, `flux.2-klein-4b-edit-…`), converted at the first edit, so that text to image compiled none of it: Core AI compiles every function of an `.aimodel` together. With the prebuilt in the sources nothing compiles at the first run, and the second export only held the DiT's weights again (4 GB more) and a second prebuilt, so it is gone. An edit at another size is refused (`References` lists what the export runs).
- The DiT's programs of a step with references: the image stream is the picture's rows then the references' (the same modulation rows), the RoPE's first axis 10, 20, … for the references, `HeadAttention`'s split at the image stream's end, `proj_out` over the picture's rows only. `TestDitReferences` checks them against diffusers' blocks in float64 (rel. RMS 3e-4, the same as without references).
- The encoder (`fluxvae.VaeEncStages`, 7 stages: a residual block a stage at the picture's size and half of it, the other down blocks whole, the mid block with the head). The downsampler pads (0, 1, 0, 1) (`milpart.Conv2DPad`); `quant_conv` is folded into `conv_out`'s first 32 channels (the means). Against diffusers (`tools/ref/ref_flux2klein.py --encode`, the fox at 512²): rel. RMS 0.39 %, cosine 0.99999.
- Numbers (M3 16 GB, warm, the separate export of before): a step at 512² with one reference 3.0 s; `samples/layers` draws a layer in 39 s at 512² and 89 s at 1024², the reference's encoding included.

## FLUX.3 klein

- DiT 5.55B: per-stream mode blocks (5 layers each, own weights, attention only within the stream, attention and FFN in parallel from the same modulated input, `x + gate (attn + ff)`), then 28 single blocks over [text, image]. FFN SwiGLU, `linear_in` first half is gate.
- Modulation on the host (14 rows): image `silu(temb(t) + temb'(t'))`, text `silu(temb(0) + temb'(t'))`. The flow-map target t' reaches the text too, so the text mode blocks run every step. `norm_out` is scale then shift. The timestep is rounded to bf16, ×1000, rounded again (as the pipeline does). Rel. RMS 1e-7 vs diffusers (`TestModulation`).
- RoPE: 4 axes × 32, θ 10000, image row/column ×0.5 (`image_rope_scale`), text (0, 0, 0, i); `RopePerm`.
- Export: 512² and 1024² × text 80 and 160 rows in one (text, `ctx` and text mode blocks per row count; image mode blocks and VAE per size; single blocks per combination). Qwen3-VL and DiT lut8, `ctx` and VAE fp16. Convert 2 m 37 s (tables 1 m 23 s), 9.4 GB; first compile about 50 min (a single-block group 6.5 min; 512²'s first run compiles 1024² too).
- DiT lut8 tables are fitted to activations: weighted Lloyd from the uniform (min/max) levels with E[x²] per input column (`palette.QuantizeWeightedMinMax`; export name `lut8x2`). Statistics: `flux3-klein.x2.safetensors` in the exports' folder, else from the source `original/hwnpu/flux-3-image-klein-calibration/` (not on Hugging Face; SHA-256 fixed in code, `StatsSHA256`); without either, uniform with a warning. Made by `ref_flux3klein.py --stats` (int8 DiT, 4 prompts separate from evaluation).

| DiT format (MPS bf16 reference, 512², final latent cosine) | portrait | 8 prompts mean / median |
|---|---|---|
| per-row int8 (clip search) | 0.9967 | 0.988 / 0.996 |
| lut8 uniform | 0.964 | 0.956 (3) |
| lut8 k-means from quantiles | 0.56 (broken) | — |
| lut8 weighted k-means from weighted quantiles | 0.93 | — |
| **lut8 weighted Lloyd from uniform (E[x²], 4 calibration prompts)** | **0.998** | 0.981 / 0.997 |

- Starting from quantiles puts the top level at the 99.8 % point and crushes the few large weights. Unweighted k-means from uniform equals uniform. 12 calibration prompts = 4. A one-step velocity error does not predict a free run (single-block attention at int8: 2.33 → 2.29 % velocity, but latent 0.964 → 0.983).
- fp16: image stream 1/32, text 1/1024 (above). Pad rows' text state drifts 50 % from the reference (prompt rows 3–7 %): pad position 1 equals the attention-sink row 2, which picks up Qwen3-VL's lut8 error. Not visible in pictures.
- M3, 512², same noise as reference: 37.0 dB, first-step velocity cosine 0.9997 (rel. RMS 2.5 %), last step 0.993.

## Current numbers (Core AI, from the second start, cache warm)

| model | machine | 512² | 1024² | memory peak (process + NE) |
|---|---|---|---|---|
| FLUX.2 klein 4B (4 steps) | M1 16 GB | 10–11 s (text 0.9, step 2.2–2.5 s, VAE 1.2); `--batch 4` 9.0 s a picture | 41 s (step 9.1–9.2 s, VAE 4.7) | 4.93 GB at 1024² (3.83 + 0.76 GiB in the VAE; NE 2.32 GiB at most in the DiT); 3.09 GB at 512² |
| FLUX.3 klein (16 steps) | M3 16 GB | 29 s (1.7 s a step) | 2 m 02 s (text 1.0 s, 7.4 s a step, VAE 3.8 s) | footprint 4.1 GB; NE not measured (models.yaml 5.5 GB is an estimate) |
| Anima-Turbo (8 steps) | M1 16 GB | 7.5 s of steps | 48–49 s (5.6–5.7 s a step, VAE 2.8–3.2 s) | 4.81 GB (3.06 + 1.42 GiB) |
| Anima-Turbo 3:4 (880 × 1168) | M2 | | 34 s of 8 steps (4.3 s a step; M1 5.4 s) | |
| Anima-Aesthetic (25 steps, CFG on 17) | M1 16 GB | | 3 m 51–57 s of steps (CFG step 11.0–11.2 s, other 5.6–5.7 s) | as Anima-Turbo |

- First run (conversion + NE compile): FLUX.2 klein 40 min (both sizes), FLUX.3 klein about 50 min, Anima-Turbo 1024² about 21 min. The first run's memory peak is higher (FLUX.2 klein 1024²: 7.00 GB).
- FLUX.2 klein v6 on an M1 from its own export's prebuilt programs (2026-11-25, macOS 27.2.1; the row above is v1 to v5, compiled on the device, five single blocks a program):

  | | time | a step |
  |---|---|---|
  | first run from an empty Core AI cache (512²) | 18 min 15 s: the 13 DiT programs finished from their prebuilts (a double group 1 min 35 s, a single group 1 min 16 s), the VAE compiled on the device (1 min 21 s) | |
  | 512² | 15 s of steps and VAE | 3.5 s |
  | 1024² | 58 s | 12.3–12.4 s |
  | 1280 × 704, 704 × 1280 | 49 s | 10.0–10.1 s |
  | 768 × 1024 | 44 s | 9.0 s |
  | edit at 1024², one reference (read at 512) | 1 min 29 s (the reference 16 s at its first use, the encoder compiled on the device) | 16.0 s |
  | edit at 1024², two and three references of 512 | 1 min 29 s, 1 min 44 s | 19.9 s, 23.4 s |
  | edit at 512², one and two references | 28 s, 43 s | 5.5 s, 9.0 s |

  No prebuilt program failed and none of the DiT was compiled on the device. A step at 1024² is 12.4 s where v5's groups of five single blocks compiled on the device took 9.1 s: 13 programs a step in place of 7 (a group of two single blocks alone takes 1.0 s at 4608 rows, a double group 1.0 to 1.3 s). Larger single groups for the M1's export would win some of it back (three a program compile offline for h13g but for the edit from three references of 512).
- FLUX.3 klein on an M1 from the sources' export and h13g prebuilt programs (2026-11-25): 512² in 1 min 14 s (4.0–4.4 s a step), 1024² in 4 min 45 s (16.8–17.2 s a step, VAE 9.3 s); the first run 49 min 33 s with the export's download (7 min 25 s) while three offline compiles ran beside it. No prebuilt program failed.
- Measure with an executable named `hwnpu-image`: Core AI's cache is per executable name, and another name compiles everything again (see [apple-coreai.md](apple-coreai.md)). To time one part from the production cache, build `tools/coreai/aicheck` as `hwnpu-image` and run `run -calls N EXPORT/PART.aimodel#FUNCTION`.
- Memory is the peak of `ri_phys_footprint + ri_neural_footprint` (`proc_pid_rusage`). `mem.NPUWired` is machine-wide and includes other processes. See [memory-and-power.md](memory-and-power.md).
- Low power mode halved image generation speed on an M3; no difference seen on an M1.

## Intel's NPU (Anima)

- OpenVINO IR from the same MIL (`ovlower`), fp16 weights, heads-together attention (`QChunkAttention`, chunks by `DitQChunk`), upsampling as reshape + tile.
- Group size (`anima.DefaultGroup`): halve from 4 until blocks × patches ≤ 4096 (512²: 4, 640²: 2, 768² and 1024²: 1). Windows' TDR (2 s) stops a longer job with `ZE_RESULT_ERROR_DEVICE_LOST` (Core Ultra 5 135U, 1024², 4 blocks). On DEVICE_LOST with a group ≥ 2 the error suggests halving `--group` (`types.IsDeviceLost`).
- "bad allocation" / "bad_alloc" from the driver's compiler counts as out of memory (`types.IsOutOfMemory`): other parts are closed and it loads again. At 1024² g1 it hit this at `dit_blocks_20` after the text and 20 blocks were loaded; reloading per prompt (text plus a few blocks) is expected.
- Checked: OpenVINO 2026.4 CPU vs diffusers fp32 at 256², 8 steps: latent cosine 0.99985, 37.9 dB. g4 vs g1 at 512² int8: the same picture, latents 3.6 % apart (rounding between groups). Core Ultra 5 135U at 512² g4: about 0.4 s a group, 2.8 s a step.

## Lessons kept from dropped models (Sana, Qwen-Image)

Sana and Qwen-Image are no longer offered (see [image-models.md](image-models.md), Not adopted; the notes for taking them up again are in [not-adopted/sana.md](../not-adopted/sana.md) and [not-adopted/qwen-image.md](../not-adopted/qwen-image.md)). What still applies:

- Core AI's NE compiler can reject a wide conv: Qwen-Image's 1152 → 1152 channel VAE stages at 64 × 64 and above (M1, `Allocation failure for L2-dep pair`) went to the GPU; the same package ran on Core ML. Splitting the conv by channels or rows did not help. Tiling the stage input into overlapping windows (≤ 48 × 48) and stitching is exact when only pixels farther than the stage's reach from the inner edge are used (overlap 7 for two tiles, 14 for three or more). FLUX's VAE (512 channels at most) does not need it.
- When one function of a grouped `.aimodel` fails, the whole file goes to the GPU: the backend ungroups (`ungroup`, `record.Alone`) and lowers the members alone (see [apple-coreai.md](apple-coreai.md)).
- `coreai-build` fixes (Sana): `x × 1` at a program's end breaks an output fusion whose constants got weight names (`redefinition of symbol named 'hwnpu_N'`); a channel max with channels moved to the last axis; one block per VAE stage; a grouped 1 × 1 conv as a block-diagonal dense 1 × 1; a channel repeat (`tile`) as a 0/1 1 × 1 conv (`mil.repeatChannels`).
- A 128-row text function hung the NE for 1–2 min then failed with `Request cancelled`, and left it failing for every process until idle; 64 and 320 rows of the same build worked. Cause unknown.
- Few-step distilled models (Sana-Sprint, Anima-Turbo) spread small condition differences: compare forms by row error against fp32 and by the composition of several pictures, not by one latent cosine.
- Program reference without much RAM: `tools/ref/stream_weights.py` loads repeating layers' weights just before use (8B float32 in a few GB).

## Not adopted

| what | why |
|---|---|
| Core ML for `hwnpu-image` | Core AI only on a Mac; Core ML exports are kept for `--cpu` |
| text encoder prebuilts, and the VAE's on M1 | wrong pictures / `Request cancelled`, no time saved (above) |
| Anima: groups of 7 blocks | −1.2 % a step, larger programs harder on 8 GB machines |
| Anima: q, k, v as one linear (2048 → 6144) | slower (a group 928 → 946 ms) |
| Anima: normalization after P·V in self-attention | overflows fp16 in the last block; with a V scale the pictures change |
| scores with keys first (1, 16, keys, queries) | 128 ms vs 89 |
| Anima: short cross-attention key buckets (exact, sink-style ln(n+1) on the last zero row) | −2.3 % a group at 256 keys, but a 512-row function is needed too and all functions compile together (first compile doubles); pending |
| Anima: cross-attention K/V once a prompt (Sana's form) | saves ~3.5 ms compute a group, adds 16 MB input a call; −117 MB NE weights only |
| Anima: GELU tanh/sigmoid forms | slower than exact erf |
| replacing the max division before norms by constants | ranges span 1400× |
| higher weight bits for speed (fp16 / lut8 on Anima) | slower than int8 and doubles NE memory |
| holding every DiT group on Core AI | no faster, up to 10 % slower at 1024², more NE memory, can fail |
| FLUX.2 klein DiT lut6 | −9 % a step but latent 0.957, the picture changes |
| FLUX.2 klein larger groups (3 double, 7 single) | the same 2.2 s a step (preload hides the switch); longer compiles |
| FLUX.3 klein lut8 k-means from quantiles | breaks the picture (0.56) |
| loading the next program ahead on Core ML | a load is a compile there, and the NE compiles about one at a time; loading 4 DiT programs at once was slower and used 7.4 GB |
| VAE outputs passed as fp16 between stages | 0–0.3 s, within noise |
| Anima subpixel conv upsampling (`depth_to_space`) | the same speed as `upsample_nearest_neighbor`; weight order not matched |
| removing softmax's row max | the bound from weights (2–221 per layer) is past fp16's exp limit of 11 |

## Open

- Move every Apple export to one export of all sizes (order above). Check each size per chip with `coreai-build` first.
- Why the VAE draws bands at 512² on h13g and h14g (on M2 compiled on the device too), and what cancels T5; drop the recipe's rules and `LeftToDevice` when fixed.
- FLUX.3 klein on Apple: Neural Engine memory, editing, other sizes, prompts over 160 tokens, the pad-row drift (e.g. later Qwen3-VL layers at fp16), how the calibration statistics are distributed.
- FLUX.2 klein editing on Apple.
- Anima on Intel's NPU: end-to-end run on hardware at 1024², the v4/v5 exports on Intel. Anima-Aesthetic ranges (`TestBlockRanges`).
- M2 and M3: block-switch costs (`aicheck cycle`), query chunk sizes, the padded `-v5` sizes on M3 and M4. Energy per picture (powerbench needs sudo).
- The prebuilt finish floor of about 10.2 s per part after the first part of a process (see [apple-coreai.md](apple-coreai.md)).
