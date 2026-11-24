# Mixture of experts

[日本語](moe.ja.md)

How BonsaiAI runs mixture-of-experts (MoE) models: gpt-oss-20b, Qwen3.6-35B-A3B and LFM2-8B-A1B on Apple, and gpt-oss-20b and Qwen3.6-35B-A3B on AMD's NPU. It covers where the routing and the experts run, how the experts are kept in memory, the current numbers and what limits them.

## Models and where they run

| Model | Shape | Apple (Core AI) | AMD (IRON) | Intel |
|---|---|---|---|---|
| gpt-oss-20b (`gpt_oss`, 21B, 3.6B active) | 24 layers, sliding (128) and full attention in turn, sinks, YaRN; 32 experts, top 4, MXFP4 in the checkpoint | Parts on the Neural Engine, experts on the CPU in the checkpoint's MXFP4 (`llm.HostRouted`, `experts.Host`) | Experts kept as MXFP4 (`mx4`), routing on the NPU; 16 GB machines use 16 places and host routing | No |
| Qwen3.6-35B-A3B (`qwen3_5_moe`, arch `qwen35moe`, 35B, 3B active) | 40 layers = 30 Gated DeltaNet + 10 full attention, hidden 2048; 256 experts of 512, top 8, plus one shared expert (512) with a sigmoid gate; bf16 72 GB | Same form as gpt-oss: experts on the CPU in MXFP4 from a derived checkpoint | q4 experts; 32 / 128 / all places by memory | No |
| LFM2-8B-A1B (`lfm2_moe`, 8.3B, 1.5B active) | 32 experts, top 4 | Experts on the Neural Engine in a program without weights (`llm.Routed`); M2 and later | No | No |
| gpt-oss-120b | 36 layers, 128 experts, top 4 | — | Checked only, not in models.yaml (see Not adopted) | — |

- The Neural Engine and Intel's NPU cannot do the routing's `gather`. On Apple the host picks the experts between parts. No Intel form exists.
- models.yaml picks the export by machine memory (`variants`):

| Model | 16 GB | 24 GB | 48 GB |
|---|---|---|---|
| gpt-oss-20b, AMD | `--expert-places 16 --host-route` (8.3 GB) | all experts (14.6 GB) | all |
| gpt-oss-20b, Apple | budget of wired experts (6.4 GB) | all wired (12.2 GB) | all wired |
| Qwen3.6-35B-A3B, AMD | `--expert-places 32` (8.2 GB) | `--expert-places 128` (17.2 GB) | all experts (26.3 GB) |
| Qwen3.6-35B-A3B, Apple | none wired (2.7 GB + page cache) | same | same |

- A 32 GB AMD machine stays on 128 places: all experts (26 GB) leave too little room.
- bonsaiai-chat (`internal/chat`) turns off drafts and cache tiers for a MoE model (see Not adopted).

## Expert usage, the core, and prefetch (both vendors)

- **Counts.** `experts.Usage` records each layer's choices. bonsaiai-chat keeps a file a model and application key: `~/Library/Caches/bonsaiai/models/<model>/experts-usage/<key>.bin` on a Mac, `~/.cache/bonsaiai/models/<model>/experts-usage/<key>.bin` on Linux (`--experts-key`; `off` for measurements). `BONSAIAI_DEV=moe.usage=FILE` writes the same form. On AMD, `iron.Model.CountExperts` counts the prompt's host routing and reads the decode's routed words back from the next command's scratchpad (no measurable cost).
- **`tools/moeusage`**: `prompts.tsv` (110 prompts, 19 fields, about 25 languages), `collect.sh`, `analyze.py`, `place.py`. Results:

| | gpt-oss-20b (4 of 32) | Qwen3.6 Mac (8 of 256) | Qwen3.6 AMD | gpt-oss-120b (4 of 128) |
|---|---|---|---|---|
| Top half within a field | 74–90 % | 89–95 % | — | — |
| Top half, all fields mixed | 81 % | 81 % | top 64: 59 %, top 128: 83 % | top 64: 88 % |
| Half chosen from other fields: computer, math | 59 %, 63 % | 60 %, 63 % | 45 %, 48 % (top 64) | 69 %, 70 % (top 64) |
| Core (at least an even share in every field) | 141 experts (1.9 GB), 31–41 % of runs | 841 (1.4 GB), 21–28 % | 901, 24–31 % | 515, 29–40 % |
| Experts never run | none | none (8 under 10 runs) | a few (78 under 10) | some |

- The AMD counts of gpt-oss-20b match the Mac's (140 vs 141 core experts, 131 shared), because the attention and the mx4 experts give the same routing. models.yaml's `experts_core` is one list for both vendors.
- On AMD, a layer's core larger than half its places is left out (`internal/llm/iron/pool.go`): with 32 places the core took most places and copies rose 34.4 → 36.7 %.
- **Next-layer guess.** The residual that entered layer l's experts, run through layer l+1's norm weight and router, guesses layer l+1's experts (Fate, MoE-CORE). Hit rate: gpt-oss-20b top 4 86–87 %, top 8 97–98 %. Qwen3.6 top 4 44–48 %, top 8 72–82 %, top 16 95 %. The lower rate is due to picking 8 of 256, not to how the experts split.

## Apple: experts on the CPU (gpt-oss-20b, Qwen3.6)

### Split

- `llm.PlanRouted` cuts the model at every MoE layer. The Neural Engine parts run the token mixer, the router's logits and (Qwen3.6) the shared expert. A part's y is "residual + shared expert"; the next part adds the host's expert output (`moe_o_N`).
- The host takes the top k (`experts.Routing{softmax, Norm}`: softmax over all, renormalized over the chosen) and runs the chosen experts with `experts.Host`. SwiGLU (`ActSiLU`) and no-bias experts are supported. The record carries the router and norm names (`MXFP4.Router`, `Norm`, `NormOffset`: +1 for zero-centered norms), so the prefetch works for any model.
- Formats: gpt-oss attention `lut8r` (lut8 with a scale per row), head `lut6r`, router fp16 (architecture Revision 4). Qwen3.6: router and shared-expert gate fp16, the rest lut8, head lut6r.
- The experts are mapped from the checkpoint (`safetensors.File.Map`, nothing copied). gpt-oss: 10.2 GB.

### MXFP4 kernels (`internal/mxfp4`, NEON)

- e2m1 by table lookup; the activation row quantized to int8 by blocks of 32; SDOT for one vector, SMMLA (`mm28NEON`, FEAT_I8MM, M2 and later; `BONSAIAI_DEV=mxfp4.i8mm=0` off) for 8 or more rows of one expert.
- Zero accumulators with `MOVI`, not `EOR v,v,v`: on the M3 EOR waits for the register's old value and the sums serialize. MOVI doubled the kernels (one vector row 65 → 32 µs).
- `mxfp4.Rows` (`mvNEON`): many rows a call. 45.6–47.3 GMAC/s on one core. It is bound by NEON instructions (about 20 a 2 blocks), not memory. Bit-identical to `Dot`.
- Encoding (`mxfp4.Encode`): the power-of-two scale per 32 values is the best (least squared error) of the OCP rule (max exponent − 2) and its neighbours; values round to the nearest e2m1 (ties to even). Relative error 0.113 on normal values.

### Generation: a token's experts

- `Host.runRow`: an expert is split into pieces (128 × 2048 weights) that any core takes. The core that ends an expert's gate/up runs its activation and opens its down. Workers are woken once a layer. The caller waits only for workers inside the work; late workers see the closed run and sleep again. Before this, an expert a core made a layer wait for efficiency cores or late wakers (the last 4 of 8 workers take 85 µs to 3 ms to wake, `TestWake`).
- The activation is `f32.GLU` (NEON on arm64: LD2 splits the gates and ups, Cephes' expf), not `math.Exp` in float64 a value: a gpt-oss expert's 2880 took 22 µs on an M2 while its down products waited. The biases are added into the products and the experts' outputs added with `f32.Axpy`. `runRow`'s arrays are kept for the next token (`Host.rowBufs`): 217 KB a layer had made the collector's clearing and returning 16 % of the experts' CPU. Synthetic gpt-oss-20b layer, M2, 1.5 ms between layers (`TestHostRunGap`): mean 1.51 → 1.39 ms, median 1.11 → 1.07, p99 6.7 → 5.9. Back to back (`BenchmarkHostRun`) about 2 %, within noise: the products read 425 MB of weights.
- `mem.Probe`: ask `mincore` for 2 pages of a range only (a range is read and used whole), and not again within a second (`hostLayer.fresh`). `mincore` on a mapped file is 0.25 µs a page and had been 47 % of the experts' CPU time.
- A layer of 8 experts, all in memory (`TestHostTokenTimeQwen36`): back to back 0.28 ms on 8 cores (0.64 on 1); after a 1.2 ms gap 0.79–0.90 ms. The gap slowdown comes from the calling thread sleeping (`BONSAIAI_TEST_QWEN36_SPIN` removes it), see Open.

### Memory: wiring (`experts.Host.wiring`, `experts.Placement`)

| Experts vs memory | Default |
|---|---|
| ≤ 1/2 | all wired (mlock). M4 32 GB: 1.8 → 0.9 ms a layer, 13.5 → 18.0 tok/s against unwired |
| > 1/2 and ≤ 3/4 (gpt-oss-20b on 16 GB) | a budget of 1/4 of memory (4.3 GB on 16 GiB), wired an expert at a time: core first, then the application's counts. Rebalanced every 30 s (first after 5 s); after `--idle` (5 min) all but the core are unwired |
| > 3/4 (Qwen3.6, 17.1 GB on 16 GiB) | none wired; the page cache keeps what fits (6–7 GB), the system counts it as free |

- `BONSAIAI_MOE_WIRE`: `0` none, `1` all, a size (`6GB`) for the budget.
- Wiring Qwen3.6 only moves the page cache's share: none 10.6, 2 GB 9.1, 4.3 GB 8.6–8.8 tok/s, and even fully resident layers got slower (system 19 % free vs 51 %).
- **Budget that follows free memory** (`placer.effective`, `mem.PressureNow`, every 5 s). Budget = free memory + what this process wired − 25 % (35 % for 5 min after a system warning); core only at critical (level 4). Changes within 2 % of free memory are ignored. Counting own wiring as free gives a fixed point. Using free percent alone made it oscillate.
- **After a warning** with more than the core wired (`placer.warned`): ceiling one step (1/4 of the budget above the core) below what was wired, held 10 min, then rises a step at a time. A warning soon after a step up doubles the hold (max 80 min). Kept across starts in `experts-usage/ceiling` (`Placement.Ceiling`).
- ENOMEM while loading is retried in place.
- **Replacement policy**: decayed counts (halved, plus recent share). Alternatives in `place.py` all gave 72–74 % hit on average; kept as is.

### Reading experts not in memory

- A mapped file faulted by each core reads 16 KB at a time: 0.4–0.6 GB/s. The same SSD gives 2.6 GB/s for one 1 MiB `pread`, 4.7 GB/s for 4.
- `Host.readIn` / `readChosen` / `experts.reader` (`reader.need`): before a row's (or chunk's) experts run, ranges not in memory are read whole with `pread` (1 MiB pieces, several readers) into the page cache, then used through the mapping. Where `mem.Resident` is missing it falls back to `madvise(WILLNEED)` (0.5 GB/s). `BONSAIAI_DEV=moe.need=0` leaves it to faults.
- **Prefetch** (`experts.prefetcher`, on by default): the guess runs in its own goroutine during the Neural Engine part; guessed experts not wired get `madvise(WILLNEED)` and are read by the readers. Count `prefetchN` = experts a token picks, at least 4 (gpt-oss 4, Qwen3.6 8). For 10 s after a run had to read experts from disk, only the top 2 (`prefetcher.auto`, `reader.short`). `BONSAIAI_DEV=moe.prefetch=N` fixes it, `0` off.
- Cold-read benchmark (`tools/moeusage/read_bench.py`, page cache dropped): one gpt-oss-20b expert (13.2 MB) took 13.9 ms by mapping/page faults, 3.0 ms with O_DIRECT, or 2.6 ms with four 2 MiB reads; a Qwen3.6 expert (6.3 MB) took 6.9 vs 1.4 ms. A warm page-cache read took 0.04 ms. Mac direct reads have not been measured.

### Prompts

- **Wave** (`decode.Session.Steps`, `decode.Program.RunMany`, `chain.tier.runMany`): run every 128-token step through a part before the next part. Stages are cut at attention layers (`RunPart(i)`). A step at a time through all parts made every step read nearly all experts (Qwen3.6: 17 GB a step, 190.9 GB for 2602 tokens). A wave reads them once (14.6 GB).
  - Carried a step between stages: `decode.Program.Carry(i)` (Qwen3.6 2.6 MB a step). A wave holds up to 64 MB of carries (25 steps, 3200 tokens).
  - Marks (the prompt store's every 512 tokens) can be taken inside a wave. `ctx` is checked between steps and parts; an interrupted wave goes back to the mark before it.
  - Step masks written as fp16 bytes directly (`step.stepMasks`, 3 → 0.1 ms).
  - `TestPagedWave`: bit-identical to step-at-a-time on a synthetic model. Long prompts differ in rounding (the wave sizes the attention cache to its final chunk from the start).
- **Several steps' rows per call** (`chain.moe.runMany`, `hostRows` 512): an expert is read once a call. Per row per layer: 128 rows 0.139 ms, 512 0.089, 1024 0.085. `Host.Run` works in groups of 1024 (expert × row) so memory stays one step's (12 MB). `BONSAIAI_DEV=moe.many=0` runs a step at a time.
- LFM2-8B-A1B and earlier: `experts.BuildRows`, `chain.moe.runRows` run each chosen expert once over all rows of a call.

### Qwen3.6 on Apple

- **Derived checkpoint** (`bonsaiai-dev mxfp4`, `qwen35.DeriveMXFP4`): the bf16 experts (64 of 72 GB) neither fit 16 GB nor the disk. Each layer's `gate_up_proj` (256, 1024, 2048) and `down_proj` (256, 2048, 512) are written as gpt-oss-style MXFP4 (`*_blocks`, `*_scales`, gate and up rows interleaved, gate even). 17.1 GB of experts, 22.0 GB in all (vision tower and MTP dropped).
  - Read from Hugging Face with Range requests a layer at a time (`HTTPSource`, 4 streams): 45–70 s a layer, 39 min on an M3. Per-layer files, so it resumes. `bonsaiai_mxfp4` is added to config.json last.
  - bonsaiai-chat (`llm.HostDerived`) writes it to the cache's `derived/Qwen/Qwen3.6-35B-A3B` (`chat.DerivedDir`) without downloading the release. `bonsaiai-dev convert` refuses a checkpoint without MXFP4 experts and points to `bonsaiai-dev mxfp4`.
  - It is an entry of the sources (`chat.DerivedKind`, `converted/Qwen3.6-35B-A3B/any/Qwen3.6-35B-A3B/`, keyed by the release's commit and `derivedVersion`): a machine takes it from there in place of deriving it, and one that derived it puts it in the sources written back to (`bonsaiai-dev models package Qwen3.6-35B-A3B/coreai`; 2026-11-22). AMD does not use it: its experts run on the NPU from the IRON export.
- **Unsupported configs** are refused by `ParseMoE`: `mlp_only_layers`, `decoder_sparse_step` > 1, `norm_topk_prob` false, sizes not a multiple of 32.
- **Neural Engine inner-dim-1 matmul bug**: in the Gated DeltaNet output o = (QKᵀ ⊙ E) U + Γ Q S0, a chunk of one row gives `matmul (1×1)·(1×128)`, which the Neural Engine computes wrong (off by about 8 on values of 40–70). One-token decoding repeated tokens. A one-row chunk now uses a broadcast mul (`deltaRuleChunks`). The dense Qwen3.5 one-token program got the same change (Revision 2).
- **Conv window output** (Revision 2): the GDN conv window `win_conv_N_new` outputs only the 3 rows the host keeps (one-token program: a slice; 128-token program: a matmul with input `step_window` (1, 3, 131), built from the step's valid tokens by `PagedStepInputs`). Saves 64 MB. This form cannot keep part of a step (partial acceptance of drafts), so such a step runs again (`wholeWindow`). Dense Qwen3.5 is unchanged.
- Checks: `TestMoEAgainstFP64` (synthetic, float64 reference from modeling_qwen3_5_moe.py, relative error 2.5e-6; catches a missing shared gate sigmoid, a missing shared expert, pre-norm input), `TestMoEPagedSteps`, `TestDeriveMXFP4` (cosine ≥ 0.99995 vs dequantized MXFP4), `TestMoEStepWindow`. The real model's float32 reference `tools/ref/ref_qwen36_moe.py` (layer outputs max 48.5: fp16 does not overflow). KL is not measured.
- First start: conversion and Core AI compile about 10 min (51 parts, 2 programs); later loads 0.9 s.
- Context on 16 GB: 180224 (KV 22 KiB a token).

### gpt-oss-20b on Apple

- Parts: attention with sinks, YaRN, the router with its bias. The experts run between parts with the kernels above.
- Accuracy (Neural Engine, KL vs float32 transformers on 5 texts): attention lut8 (one table per 16 rows) 0.284, `lut8r` 0.069, fp16 0.033. Projection rows differ up to 6600× in magnitude, so 16-row tables starve small rows; `lut8r` divides each row by its max (fp16 ≤ 1) and multiplies back before the bias. Head `lut6r` against lut8: about +0.004 mean KL, 10.2 → 8.1 ms a step. `--format-for=self_attn=fp16` gives the best answers.

### LFM2-8B-A1B on the Neural Engine

- A part ends at each MoE layer with fp16 router logits and the experts' input (`moe_x_N`, `moe_logits_N`). The host takes the top 4 and runs a program without weights on the chosen experts. Their int8 weights live in arrays of that program that hold every layer's experts (7.22 GiB, read once, shared by chains that `Share`).
- The experts' scales are picked by `gather_nd` with int16 indices: the M2 and M3 run it, the M1 does not (`not_on: [M1]`, `chat` refuses on an M1).
- `PlanRouted` with `Paged` also splits at attention (a part of no layers, `NoLayers`) for a 128000 context.
- Short of memory, macOS compresses the expert arrays and prompts take up to 3× as long: `BONSAIAI_MOE_WIRE=1` wires them (off by default).

### Numbers (Apple)

| Model, machine | Generation | Prompt | Memory |
|---|---|---|---|
| Qwen3.6-35B-A3B, M3 16 GB (MacBook Air 15, macOS 27.2) | 10.9–11.0 tok/s (8 prompts × 150 tokens) | 2602 tokens 21.0–21.5 s; 621 tokens 8.1 s | process 0.75 GB + Neural Engine 1.94 GB = 2.69 GB, plus 6–7 GB of page cache |
| gpt-oss-20b, M3 16 GB, quiet | all wired 17.0–18.1; budget 4.3 GB 17.5–18.1 | 3914 tokens 33–38 s | process 0.5 + budget + Neural Engine 1.44 GiB |
| gpt-oss-20b, M3 16 GB, busy page cache | 12.2–12.4 → 14.1 tok/s with the CPU changes of the generation section | 707 tokens over 8 prompts 14.2–14.5 s | |
| gpt-oss-20b, M4 32 GB | wired 17.4–18.0 | 2157 tokens 26.7–30.5 s | |
| gpt-oss-20b, M2 16 GB | all wired 13.2; default budget 10.1; unwired 7.9 | — | |
| LFM2-8B-A1B, M3 16 GB | 15.0–16.1 | 1503 tokens 8.6–10.9 s | 8.0 GB |
| LFM2-8B-A1B, M2 | 12.0–13.7 | 1624 tokens 13–14.5 s | 8.0 GB |

- Qwen3.6 step at 10 tok/s (106 ms): experts 51 ms, Neural Engine about 55 ms (a GDN part 1.3 ms, head about 8 ms). Since then, layers whose experts are all in memory went 0.57 → 0.36–0.42 ms; layers that read (about 30 %) stay at 1.6 ms. Another process taking 3 GB dropped 9.8–11.5 to 8.4 tok/s.
- Qwen3.6 prompt of 2602 tokens: experts about 17.5 s (reading 14.6 GB about 4 s), Neural Engine parts and attention about 3.5 s.
- Power, M2 16 GB, gpt-oss-20b: about 7.0 W whatever the wiring; 0.50 J a token with memory to spare, 0.71 J while reading experts from disk. M3: 8.96 W, 0.52 J.

## AMD: IRON (Ryzen AI 9 365)

The NPU reads only pinned buffers (banks). An expert's position in its table is chosen at call time: the scratchpad words of a command (at most 32 words) are added to the shim DMA's addresses (`Affine`), so an operator reads the chosen experts in place, with no copy.

### Qwen3.6 decode

- A layer is 2 commands with an empty host step between (decode has 81 commands a step):
  - **A**: the mixer, residual, post-attention norm, router GEMV (256 rows + 1 shared-gate row), `MoERoute`.
  - **B**: the experts. One operator `MoEExperts` (gate/up and down in one program, as gpt-oss's `MX4Experts`; the activation goes through DRAM). Its table holds the routed experts and then the shared expert as the last entry. SiLU × up × routing weight is applied at the gate/up exit (`moe_glu`), and the residual is added at the down exit.
- **`MoERoute`** (`kernels/moe_route.cc`, one core, vectorized, 0.1 ms): top 8 (ties to the lower index), softmax over the chosen (bf16, as transformers), shared sigmoid. It writes command B's scratchpad words (`id × expert bytes`). Which words to write comes from a table filled at export from IRON's lowering (`params.txt`, `Qwen35Moe.program_rows`).
  - Its output is a feedback buffer used as B's ctrl scratchpad (manifest `fed`; `Run.UseScratchpad`). The host never reads the logits.
  - The feedback buffer must be in the device heap (`BufferInstructions`); an ordinary BO as scratchpad fails with `ERT_CMD_STATE_ERROR`. The command processor reads device addresses, but the shim DMA uses the process's virtual address (PASID), so relocations of a heap buffer use the virtual address (`buffer.dmaAddr`).
- **Gated DeltaNet operator** (`tools/amd/iron_gdn.py`, `kernels/gdn.cc`): 8 columns × 2 cores; the recurrence after mlir-aie's `gated_delta_rule.cc`. The short conv (`gdn_conv`, history in the state's first pair of each column) and the gated norm (`gdn_norm`, y/rms(y)·silu(z) in float32, norm weight folded into `out_proj`) run inside the operator. The operator reads and writes the checkpoint slots directly (`gdn_operator(slots=)`). q/k SiLU once per key head (`gdn_gather_conv_v`). in_proj (8704) and z (4096) share one table of 12800 rows (200 column blocks, 25 per column).
- Attention layer: q, k, v and gate in one table (9216 rows); q/k norms written to one state so RoPE runs once. Residual added in the out_proj / o_proj GEMV (`into_linear`).
- **`stepBarrier`**: before each decode step, an empty instance of the decode ELF runs on a separate hardware context (0.3 ms). Without it, command 0 after the last layer's B hangs (`ERT_CMD_STATE_TIMEOUT`) once the conv is inside GDN. The cause is unknown; running the barrier on the same context also hangs.
- Formats: experts and head q4 (q4nx), mixers q8.

### Qwen3.6 prompt (chunks of 256)

- A scratchpad cannot hold 2048 positions, and no DMA groups tokens by data-dependent indices. So the prompt does:
  - **A**: mixer, residual h (copied to state `moe_h`), normed hn, router logits (C × 320).
  - **Host route** (`moe_route`, `topK` selection): top 8 per token; rows sorted by expert; per-expert counts (512 words: 256 counts plus the pad count); only real rows written.
  - **B**: shared expert (q4 Linear), `MoEGroupGateUp`, `MoEGroupDown` (`tools/amd/iron_moep.py`, `kernels/moe_rows.cc`). The weight stream is static: every expert's weights flow once a pass; each core loops an expert's row count read from the data. Pad rows go last, are read through and output 0.
  - **Host combine** (`moeHostCombine`, split per token over CPUs): out = h + shared gate × shared + Σ weight × expert row (float32, rounded to bf16 once).
- Rows per expert per pass `MOE_CAP` = 256 (= the chunk), so a chunk is always one pass. gate/up and down are spread over a column's 4 cores, each core holding half a block (16 outputs, one q4nx strip). 256 rows then fit in 32 KB of accumulators.
- Go reads the manifest's `cap`, so old exports still run.
- When a chunk needs several passes, the host writes hn back before each later B (`moeRoute.hn`). IRON's allocator had put B's output over hn.
- Conv in prompts: IRON `CausalConv1d` with its own state. The GDN operator skips the pair slots (`pairs=True`). BonsaiAI copies history between the conv state and the GDN pairs around each chunk (`internal/llm/iron/convpairs.go`, manifest `conv_pairs`).
- Host steps are described in the manifest (`host`) and run by Go between commands (`internal/llm/iron/moe.go`).

### gpt-oss-20b on AMD

- **mx4 experts** (`tools/amd/iron_gptoss_moe.py`, `kernels/moe_mx4.cc`): q8's block shape (32 outputs × 320 inputs), e2m1 nibbles, the MXFP4 exponent byte kept as the scale (put into bf16's exponent field in the kernel), the block's bias. 4.3 bits a weight. Decoded as sign × max(m, 2m − 4, 4m − 16). 2880 = 9 × 320, so no input padding; gate/up outputs padded to 3072.
- **Sinks** (`tools/amd/iron_sinks.py`, `kernels/mha_sinks.cc`): the P×V core adds 2^(sink·log2e − m·scale·log2e) to the row sum l before the final divide. Starting the softmax state at the sink was wrong: the device's exp2 is up to 2× high far below 0. Only the P×V worker program of IRON's MHA is replaced. `sink_form` was added to the operator args because IRON's compile cache keys on them.
- Residual padded to 3072 (norm weight × sqrt(2880/3072), eps × 2880/3072). Head rows padded to 201216; BonsaiAI reads the first 201088. Attention projections q8, head q4 (GPTQ codes optional via `--q4-codes`), router bf16, embedding on the host. Window layers read only the window.
- **Decode as one resident array** (`GptOssStep`, `tools/amd/iron_gptoss_layer.py`, `kernels/gptoss_layer.cc`): every layer op is a phase of one array. A command = previous layer's experts (E) → q/k/v with RMSNorm, bias and RoPE (Q) → attention (A) → o_proj with bias and residual (O) → router (R) → routing (T: writes the next command's scratchpad words). The last command is [E, head (H)]. 25 commands a token, the PDI loaded once (`unexpanded_loads`), no step barrier (`no_step_barrier`; `BONSAIAI_DEV=iron.barrier=1/0` overrides).
  - Projections in q8x blocks of 5504 bytes, the size of an mx4 block. The prompt dequantizes the same q8x tables (`Q8XDequant`), so there is one copy.
  - Head read in place from the q4 table as 5504-byte windows stepping 5120 bytes. A row's two partial sums are added by BonsaiAI (`logit_parts`).
  - KV in one array for decode (`cache.kv`); BonsaiAI joins and splits it with the prompt's key and value arrays (`joinKV`, `splitKV`). Window layers keep a 128-row ring (`GPTOSS_KV_RING`, position % 128); `ringsTo` rebuilds it after rewinds. `TestKVRingRewindOnDevice` checks it.
  - One exponential `gl_exp32` (32 lanes, 2^k·p(f), bf16 two-limb) for softmax, glu and routing weights.
- **Prompt experts** (`MX4GroupGateUp`, `MX4GroupDown`): host routing (`moeHostRouteMX4`). Each block is decoded straight into bfp16ebs8 (mantissa = 2·e2m1, exponent = E + 5, exact for weights; activations rounded to 8-bit mantissas) for the bfp16 matrix instruction `mac_8x8_8x8T`. The bf16 `mmul<4,8,8>` on AIE2P is emulated with broadcast macs.
  - Slots: `SLOTS = 40`, `CAP = 56`. Slot e < 32 holds expert e's first rows; 8 overflow slots take busy experts' further rows (positions as per-call words, `slot<i>`). A weight block is decoded once into the core and shared by the slot's rows. A chunk that does not fit runs in 2 passes.
- The host reads logits with AVX2 (`vecops.BF16ToF32`, `f32.Axpy`) and top k skipping 256-blocks (`vecops.MinMax`).

### Expert places (`--expert-places P`)

- The export holds P places a layer in the bank plus, for Qwen3.6, the shared expert as the last entry. The places are zero in the bank and holes in the image file (sparse). The full tables are in `experts.bin` (per layer: all gate-up, then all down). Manifest `experts_pool`. Every Qwen3.6 operator reads the shared expert as "the last in the table", not index 256.
- Load (`PlaceExperts`, `experts.Placement` from bonsaiai-chat): fills places by core, then app counts, then index. bonsaiai-chat passes `experts_core` and the app's counts only when the model is called by name (`chat.LookupFor`); an export directory has no definition.
- **Decode, device routing** (gpt-oss-20b all/24-place variants, Qwen3.6): commands run one at a time. Before a fed command, the host reads the routed words back, copies chosen experts not in a place over the least recently used place (none chosen this token; core last), and rewrites the words to place positions (`route`, `placeChoices`). Kernels are unchanged. The wait costs about 1.2 ms a token (gpt-oss, 24 commands).
- Scores: +1 a use, × 0.995 a token (half-life about 140 tokens), seeded with the counts' steady state.
- **Decode, host routing** (`--host-route`, `internal/llm/iron/route.go`; gpt-oss 16 GB export and 120b): a layer's command ends after attention. The host reads the residual (`yo`), runs RMSNorm and the router on the CPU (`router.bin`, transposed float32), takes top 4 and softmax, and writes the next command's position words and routing weights (`moe_rw<l>`). Then it guesses layer l+1's top 4 and reads missing ones into l+1's places in the background (`guessNext`, `expertPool.inflight`). vs device routing (20b): median cosine 0.99998, KL 0.0095; 50.3 → 51.9 ms a step.
- **Prompt** (`poolPasses` / Qwen3.6 `poolRowPasses`, `rowsPass`): experts already in a place go in the first pass; others are copied into places of experts not needed later, before their pass. `TestPoolRowPasses` checks it.
- **Copying**: 4 goroutines copy from the page cache mapping, then clflush (`SyncToDevice`). With `directReads` (the experts file is at least half of physical memory; `BONSAIAI_DEV=moe.direct=0/1`), reads use O_DIRECT through per-goroutine staging buffers (the bank mapping rejects O_DIRECT with EFAULT), 8 pieces an expert. Measured costs: gpt-oss expert (14.3 MB) 1.1 ms from page cache, 2.6 ms O_DIRECT (vs 9.4 ms by faults); Qwen3.6 expert (1.97 MB) 0.4–0.7 ms O_DIRECT, 0.1–0.3 ms from cache.
- `BONSAIAI_DEV=moe.check` compares every place with `experts.bin` after filling and at close. Place exports give the same logits bit for bit as all-expert exports (same weights, other places).

### Numbers (AMD, Ryzen AI 9 365)

| Model, export | Pinned / peak | Decode | Copied per choice | Prompt |
|---|---|---|---|---|
| gpt-oss-20b, all | 12.22 GiB + RSS 0.3 | 20.2 tok/s (49.4 ms; ctx 4096 52.3, 7680 56.1) | — | 256 tokens 1.8–1.9 s; 2248 tokens 15.5 s |
| gpt-oss-20b, P = 24 | 9.44 GiB | 19.0 | 2.4–2.6 % | 2248 tokens 17.1 s |
| gpt-oss-20b, P = 16, host route, O_DIRECT (16 GB) | 7.7 GiB peak | 14.4 | 3.5 % | 1217 tokens 10.5 s |
| Qwen3.6, all (48 GB) | 24.4–25.2 GiB | 8.8–8.9 tok/s (112 ms) | — | 1173 tokens 14.1 s; 256-token chunk about 2.5 s |
| Qwen3.6, P = 128 (24 GB) | 14.9–16.0 GiB | 7.5–7.6 | 6–7 % | 1173 tokens 19.1 s |
| Qwen3.6, P = 32 (16 GB, O_DIRECT) | 7.2–7.6 GiB | 5.8–6.0 | 33–38 % | 1173 tokens 29.7–29.8 s |
| gpt-oss-120b, P = 56, host route, O_DIRECT, prefetch (64 GB) | 32.05 GiB + 1.0 | 9.7–9.9 | 3.2 % | 74 tokens 6.1 s |

- gpt-oss-20b accuracy: vs HF float32 top 4/4, cosine 0.9968–0.9997; KL over 5 texts mean about 0.03. Energy: 0.92 J a token including the prompt (15.6 W while running).
- Qwen3.6 accuracy (q4 rounded, as models.yaml exports): cosine 0.989–0.995 vs HF float32 (32-token prompt and 8 greedy steps), top 9/9; two calls stay below `ironlm`'s 0.99 threshold. With GPTQ codes (`qwen36_gptq.py`, `--q4-codes`): 0.989–0.997, CPU KL 0.0243 → 0.0185. Place exports are bit-identical to all-expert exports.
- Qwen3.6 export: 29–37 min, peak RSS 36–50 GiB depending on P. A 16 GB machine cannot export it and takes it from the source. 20.1 GB `experts.bin`, 1.97 MB an expert.
- gpt-oss-20b export: 3 min 16 s, peak RSS 38 GB.

## What limits speed

| Path | Limit now |
|---|---|
| gpt-oss-20b AMD decode | DMA: 2.53 GB a token at 55 GB/s = 46 ms against 49.4 ms run (93 %). The rest: command boundaries (0.1 ms each, firmware start; 24 a token) and fixed phase cost (A, R, T about 70 µs a layer) |
| Qwen3.6 AMD decode | Operators loaded onto the cores each time one follows another (not one resident array like `GptOssStep`). GDN operator 1.06 ms, mostly load and stream setup |
| Qwen3.6 AMD prompt | Expert kernels expand q4nx to bf16 and use 16-lane vector MACs: compute-bound (weights stream alone at about 40 GB/s). With places, a pass streams the shared expert and all P places, and a 256-token chunk touches about 200 experts a layer |
| Places on AMD | The copy of misses (P = 32: about 50 ms a token); device routing gives no time to prefetch |
| gpt-oss-120b | SSD bandwidth (about 300 MB a token read) |
| Qwen3.6 Mac generation | Experts missing from memory (about 30 % of layers, 1.5 MB an expert); the Neural Engine parts (about 55 ms); cores slow after sleeping during a part |
| Qwen3.6 Mac prompt | Expert compute: 0.085 ms a row a layer even with ≥ 16 rows an expert; the 4-bit → int8 unpack costs per row |
| gpt-oss Mac 16 GB | Page cache share; M2 kernels 1.85 ms a layer (29 GB/s) against the M3's 0.9 |

## Pitfalls

- **Neural Engine**: `matmul (1×1)·(1×N)` (inner dim 1) is computed wrong on the M3 through Core AI: use a broadcast mul.
- **IRON region allocator does not count host-step reads**: a buffer the host reads can be overwritten by a later op. The host must read only the output of the op just before the host step, or a copy in a state. This caused hn being overwritten (NaN; broken shared expert in later passes), and gpt-oss's hn breaking at 24 layers but not 2.
- **Shim channel task queue is 4 deep**: keep at most 3 outstanding per channel. Two hangs in a row left the SMU dead (`aie2_smu_init: Access power failed`) until reboot. `MoEExperts` hung when down's 9 streams were queued ahead: `iron_moeprobe.AHEAD` = 2.
- DMA: a 1-D size ≤ 1023 words; descriptor repeat ≤ 64; a command scratchpad ≤ 32 words; scratchpad reads are hoisted to the start of the runtime sequence (device-written words are not seen mid-command).
- An array-key arg that differs between layers (gpt-oss `kvn`) makes separate arrays and reloads every command (1.80 → 4.27 ms a layer) while checks still pass: watch the times.
- Core program memory 16 KB, data memory overflows; vector loads must be 64-byte aligned; `vector.extract<8>` with a runtime index returns wrong values; llvm-aie crashes on some loops and alignments.
- `--command-per-step` exports give unreliable Qwen3.6 prompt logits (cosine 0.32). A 2-layer truncated checkpoint routes most tokens of layer 0 to the same 8 experts, unlike the full model.
- `--check` at export fails for `qwen35moe --q4` (host reference GEMM gets q4 tables); check with `ironlm -record` and `iron_qwen35_check.py ref --stream` instead.
- Polling `bonsaiai status` every 0.5 s while a model loads was followed once by a firmware mailbox timeout and a dead SMU (cause not confirmed). While measuring, another process on the NPU makes one-command-at-a-time decoding much slower.
- **Mac**: bonsaiai-chat writes thinking to stderr: compare text from stderr's "ready" line to "[prompt". Core AI's cache is per process name: renamed test binaries add 1.8 GB each. Measure with `--experts-key off` or the app counts bias hit rates.

## Not adopted

AMD:
- Moving experts in frequent rounds: replaying gpt-oss routing traces gave 81.8% hits when refreshed every 450 tokens (0.83 GB read per 1,000 tokens), 85.3% with 8 experts every 16 tokens (6.5 GB), and 89.0% on every miss (140 GB). The small hit-rate gains cost 3–8× more reads; the existing prefetch makes misses cheaper.
- Skipping or penalizing unloaded experts (float32 reference only): dropping the lowest of gpt-oss's top 4 cut misses 29% but gave KL 0.014 on generated answers and 0.095 on wikitext-like text; dropping one of Qwen3.6's top 8 gave KL 0.0095. Outputs change, and no NPU implementation was tested.
- Copying chosen experts into a work state, then GEMV: bytes moved 3 times, `Copy` reads 26 GB/s at best. Reading in place by per-call offsets replaced it.
- Down as "a column = an expert": needs 7 adds to sum, 1.7× `MoEDown`. A one-row GEMM of a transposed down is slower than GEMV.
- Requantizing gpt-oss experts to q4nx: weight error 6.6 %, KL 2–3×. mx4 kept.
- gpt-oss attention in q6: KL 2–3× (CPU). GPTQ on all of gpt-oss: wiki-ja worse (0.027 → 0.066); only the head uses GPTQ codes.
- Prompt GEMM over real rows only: `valid` is in units of 256 rows, no gain; per-row GEMV is slower.
- Keeping Qwen3.6 prompt experts pre-dequantized (bfp16): more pinned memory.
- Qwen3.6 prompt gate/up with bf16 `mmul` 8×8×4: not faster (code reorder via `filter_even`/`filter_odd` about 290k cycles a call; per-row scale and min eat the gain).
- Qwen3.6 B and the next A in one command (81 → 41 commands): 17.6 → 17.6 ms.
- Double-buffered GDN state objects: data memory overflow; input-only no gain.
- One command a token for gpt-oss: device-written scratchpad words are not reread inside a command (cosine 0.877). `ERT_CMD_CHAIN`: same or slower (19.3–19.6 vs 19.6–19.7 tok/s).
- gpt-oss: smaller expert core programs (IRON unrolls by ObjectFifo depth), KV writes straight from GEMV/RoPE/`QGEMVQKV` (one more switch), norm weights folded into the table (q8 error changes the top token), glu by tanh (6 KB less PDI, bf16 precision), router GEMV inside `MoERoute` (costs a switch), 2 cores a column for `MX4Experts` (compute 2.39 ms), x placed in memtile once (`repeat_count`) or 8-row output objects (no gain), bfp16 attention scores (3.6 % error), bf16-only exp (routing weights 1.3 % off), contiguous KV read (hung).
- Core experts on Qwen3.6 with 32 places: copies rose 34.4 → 36.7 %. A core over half the places is left out, and models.yaml's core for Qwen3.6 serves the Mac.
- **gpt-oss-120b in models.yaml**: 32 GiB pinned and about 10 tok/s bound by the SSD; suits a GPU. Revisit with NPUs over 80 TOPS and cheaper memory. Export: `iron_export_lm.py gptoss CKPT OUT --max-seq-len 8192 --prefill-chunk 256 --expert-places 56`.
- `not_on` naming an NPU vendor: not merged (catalog conflict); no longer needed since Qwen3.6 runs on Apple.

Apple:
- Routing `gather` on the Neural Engine, and any Intel form: neither NPU runs it.
- Core AI's MX types: GPU only (M3–M5). An exact MXFP4 lookup on the Neural Engine costs 49 ms an expert. So the experts run on the CPU.
- Drafts for a MoE model: experts run row by row, 7 drafts are 8 rows of experts (2.3 vs 14.1 tok/s on LFM2-8B-A1B). Cache tiers: each tier is an export with all the experts.
- Wiring experts that exceed 3/4 of memory (Qwen3.6 on 16 GB): slower at every budget.
- Prefetching 16 for Qwen3.6 (7.3 tok/s, wrong guesses eat SSD reads); 8 for gpt-oss (no better than 4, worse when memory is short); a guess from two layers back (80–85 % top 4, more reading).
- Keeping workers spinning (`mxfp4.Hold`, spinners): 0.55 → 1.0 ms, and the Neural Engine parts slow by as much. Thread QoS `USER_INTERACTIVE`: no effect.
- SMMLA pad threshold (`BONSAIAI_DEV=mxfp4.pad=N`): no difference. 2 rows × 4 vectors SMMLA: slower than SDOT.
- Other replacement policies for wired experts: all 72–74 % average hit.
- gpt-oss attention: plain lut8 (KL 0.284), int8 (slow on the Neural Engine); GPTQ codes for `lut8r` attention or lut6 head are not the default (not better on every text; codes not shipped).

## Open

- Qwen3.6 Mac: prompt expert compute (about 13 s of 21 s excluding reads); missed-expert reads in generation; cores slow after sleeping (keeping one spinning slows the Neural Engine parts); 24 GB and larger Macs not measured (budget wiring, `experts_core`); KL not measured; whether the inner-dim-1 fix changed dense Qwen3.5.
- gpt-oss Mac: top-2 vs top-4 prefetch on an M3 with other applications busy (if top 2 is slower there, set `fewWhenShort` back to 4).
- Qwen3.6 AMD: prompt with places (P = 32 is 2.2× all; ideas: compute misses on the CPU, widen places during prompts); decode prefetch (needs host routing); GPTQ codes (`qwen36_gptq.py`, 4.3 h, 18 GB) not in the models.yaml export; one resident array for the decode step; prompt experts on the matrix unit (q4nx group scales are not powers of two and must be applied to partial sums).
- gpt-oss AMD: more than 56 places (prompt slots `COUNT_WORDS` 64); per-layer place counts (the array key needs equal sizes); attention below q8 with GPTQ (36 % of a layer's transfer); KV laid out [group][position][key, value] for long contexts; draft verification for MoE.
- Computing missed experts on the AMD host CPU instead of copying them: not tried.

## Related work

| Work | Relation |
|---|---|
| NPUMoE (arXiv 2604.18788, Core ML, prompts only) | Agrees: no gather on the Neural Engine |
| EStream (2609.06551, Hexagon) | Per-call slot offsets as AMD's scratchpad words; O_DIRECT reads like `directReads` |
| MoE-CORE (2610.01950), Fate (2502.12224), Mixtral offloading (2312.17238) | The next-layer router guess used on both vendors |
| 2505.16056, 2609.04895 | A cache of about 2× the active experts suffices; BonsaiAI's decayed counts are LRFU-like |
| 2412.00099 (cache-conditional routing), HOBBIT (2411.01433) | Change outputs: only behind a KL check; not done |
| AMD ONNX Runtime GenAI gpt-oss-20b (INT4, CPU routing) | 13.4 tok/s in 13.7 GB all resident, 2.4 in 2.7 GB with 24 dynamic layers; BonsaiAI 20.2 in 12.2 GiB, 17.2 with 16 places in 6.9 GiB (not on the same machine) |
| FastFlowLM gpt-oss-20b (Q4_1, closed kernels) | 18.2–19 tok/s on XDNA2 |
| llama.cpp Hexagon | MXFP4 experts decoded in an NPU kernel, 18.4 tok/s |
