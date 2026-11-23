# Performance

[日本語](performance.ja.md)

What each improvement bought, what to watch when combining them, where the speed ceiling is and what software can still move, and what is not yet verified. One row per improvement; the design and the full measurements are in the linked topic files.

Machines: Apple numbers are from the Neural Engine of an M1, M2 or M3 (16 GB, macOS 27, Core AI) as noted; AMD numbers from a Ryzen AI 9 365 (Strix Point, XDNA 2). "Generation" is one token, "prompt" is reading the prompt, "start" is the second and later start. The GPU is never used.

## At a glance

### Language models: Apple

| Change | What it bought | Cost | Details |
|---|---|---|---|
| Split by layers (1 GiB a package, 0.9 GiB a part) | Fits on the ANE; RSS about one part | | [apple-coreml.md](apple-coreml.md) |
| KV cache | 34 → 17–21 ms a token (M3, LFM3-1.7B) | | [llm-engine.md](llm-engine.md) |
| Sliding layers as rings | −26 % at 4K; KV −43 % | | [llm-engine.md](llm-engine.md) |
| `SDPAGrouped` (grouped attention) | Generation −45 %, prompts −25 %; no tile copies | | [llm-engine.md](llm-engine.md) |
| Two programs (generation, prompt) on one cache | 860-token prompt 35 s → 1.6 s | Weights once only as functions of one `.aimodel` | [llm-engine.md](llm-engine.md) |
| Previous prompt reused in memory | 0.1–0.5 s from the second turn | | [llm-engine.md](llm-engine.md) |
| Prompt store on disk | 7.9K tokens 14.6 s → 0.2 s | About 145 MB a sequence | [llm-engine.md](llm-engine.md) |
| Split at the attention (`--paged`, default) | Generation 1.3–1.7×; context 262144; cost follows the length used | Two NE calls an attention layer: an all-attention model (Qwen3-0.6B) runs 2–4× faster unsplit (`llm.Unpaged`) | [llm-engine.md](llm-engine.md) |
| Attention chunks per kv head (Core AI, `paged.Heads`) | M1: Qwen3-0.6B 30.3 → 32 tok/s; 128-token call 88–91 → 85 ms | One-kv-head models (Gemma 4 E2B, PLaMo 3 0.8B) stay grouped | [apple-coreai.md](apple-coreai.md) |
| Two-level attention (`-sparse`) | A call −52 % at 128K, −58 % at 250K (LFM3); prompts −24 % / −44 % | Approximate (passkeys kept); +1/8 memory; 64K reads 10 % slower | [llm-engine.md](llm-engine.md) |
| Prompts 128 tokens a call, logits of the last row (`LastRow`) | At 16K −20–25 %; reading a document −25 % | | [llm-engine.md](llm-engine.md) |
| Gemma 4's KV-sharing layers on the last row only | E2B 16K 27.8 → 12.6–13.2 s; 64K 205 → 94.6 s | | [llm-models.md](llm-models.md) |
| Prompt program up to the last cache writer (`-kv-only`) | E2B on an M1: 4249 tokens 4.2–4.6 → 2.4–2.7 s; weights 3.20 → 1.98 GiB | | [llm-models.md](llm-models.md) |
| Soft cap on the host (`llm.LogitSoftcap`) | −2.3 ms a step on an M1 (4 %); output layer all on the NE | | [llm-models.md](llm-models.md) |
| `blocks.SiLU` (silu with exp) | lut8 KL 2.5× lower (LFM3 0.058 → 0.022), PLaMo 3 −20–40 % | +3 % generation | [apple-neural-engine.md](apple-neural-engine.md) |
| Weight formats lut4 / lut6 / lut8 | A call 1.81 / 2.65 / 3.49 ms; lut6 / lut8 KL 1/6–1/10 of lut4 | Bytes | [quantization.md](quantization.md) |
| Rotation before lut (`llm.Rotate`, Qwen3.5 2B lut6, default) | NE KL 0.054 → 0.029; with MTP 45 → 52 tok/s (M2) | Re-export on first run | [quantization.md](quantization.md) |
| Prompt-lookup drafts | 1.2–1.5× over a task (15.9 → 23.7 tok/s) | A Q=8 call costs 13 % more than Q=1 | [llm-decoding.md](llm-decoding.md) |
| MTP drafts, GDN state rebuilt on the host (`state-rows`) | Qwen3.5-0.8B (M1) 26.1 → 43.8 tok/s on code, 25.8 → 30.3 Japanese | +0.27 GiB on the NE | [llm-decoding.md](llm-decoding.md) |
| MTP head over part of the vocabulary (`HeadRows`, 74272 rows) | Qwen3.5-4B 17.4–18.2 → 19.6–20.1 tok/s; head 0.68 → 0.26 GiB | | [llm-decoding.md](llm-decoding.md) |
| Think budget, repetition detection | Stops minutes of spinning | | [llm-decoding.md](llm-decoding.md) |
| Chunked Gated DeltaNet (WY form) | Qwen3.5-4B step 96 → 83–89 ms; 128 tokens 511 → 304–344 ms; first compile 15.8 → 7.0 s | | [llm-qwen.md](llm-qwen.md) |
| Core AI as the Mac backend | Second start 0.7–5.5 s for every chat model; Qwen3.5-4B 20.5–21.9, E4B 20.3–20.7 tok/s (M2); 2K prompt 4.0 / 1.8 s | First compile per Mac | [apple-coreai.md](apple-coreai.md) |
| Parts with the same weights as functions of one `.aimodel` | NE holds weights once (LFM3-1.7B 2.56 → 1.34 GiB), +13 % | First compile pays every function | [apple-coreai.md](apple-coreai.md) |
| Input arrays made once; filled in place by the caller | PLaMo 3 0.8B 25 → 38–44 tok/s; LFM3-1.7B at 3K tokens 39.8 → 43.0 tok/s (M1) | | [apple-coreai.md](apple-coreai.md) |
| No whole program on an M1 (`chat.wholeSlower`) | M1: LFM3-1.7B 33.6 → 39.9, Qwen3-0.6B 20.2 → 30.3 tok/s | | [apple-coreai.md](apple-coreai.md) |
| Short-context programs while the conversation is short (`ringTiers`) | PLaMo 3 Translate 2B on an M1 12.9 → 19.0 tok/s | First compile twice as long | [llm-models.md](llm-models.md) |
| PLE table in int8 a layer's slice | E2B table 4.4 → 2.2 GiB; Mac download E2B 4.94 GB (checkpoint 10.2); KL 0.00005 | None: speed and peak memory unchanged | [llm-models.md](llm-models.md) |

### Language models: AMD

| Change | What it bought | Cost | Details |
|---|---|---|---|
| One full ELF | Avoids a 0.46 ms xclbin switch per op | | [amd.md](amd.md) |
| Fewer operator switches in the decode step (`QGEMVGlu`, `SandwichAdd`, `ShortConvStep`, state slots in place, PLE batched, QGEMV one core a column) | LFM3-1.7B 11.2 → 17 tok/s, Gemma 4 E2B 7.7 → 9.7, E4B 5.5 → 6.4, Qwen3.5-0.8B 13.7 → 18.7, Qwen3.6 8.4 → 8.9 | KL unchanged within noise (LFM3 0.0642 → 0.0646) | [amd-llm.md](amd-llm.md) |
| A decode step as one resident array (`GptOssStep`) | gpt-oss-20b 79.2 → 49.4 ms (20.2 tok/s), 93 % of the read floor | +1.06 GiB (0.67 back with `Q8XDequant`) | [moe.md](moe.md) |
| One short-convolution operator in prompt chunks (`ShortConvChunk`, `GdnConvChunk`) | LFM3-1.7B 16 tokens 443 → 386 ms; Qwen3.5-0.8B 343 → 259 ms | | [amd-llm.md](amd-llm.md) |
| Prompt MHA on 8 pipelines (`--mha-pipelines`) | Chunks 2–5× faster; energy −66 to −83 % | Export again | [amd-llm.md](amd-llm.md) |
| Prompt chunk 1024 (attention), 512 (GDN) | −20 % against 2048 | | [amd-llm.md](amd-llm.md) |
| 4-bit weights (`--q4`, attention q8) | Generation 1.3–1.6×; pinned 1/2–1/5 | KL 0.009–0.097 by model | [quantization.md](quantization.md) |
| q6 MLP for small Qwen3.5 | 0.8B GPL KL 0.34 → 0.044; 0.8B 1858 → 1069 MiB with host embeddings | | [llm-qwen.md](llm-qwen.md) |
| A checkpoint's own MXFP4 kept (mx4) | 4.3 bits a weight; re-rounding to q4nx had KL 2–3× | | [moe.md](moe.md) |
| PLE and embeddings on the host | E2B 9.08 → 4.60 GB pinned | | [amd-llm.md](amd-llm.md) |
| Gemma 4's KV-shared layers on the prompt's last row | E2B prompts 2.3×, E4B 1.7× | | [amd-llm.md](amd-llm.md) |
| Window layers read only their window | Prompts −12 % | +12 MB; needs `chunkBarrier` | [amd-llm.md](amd-llm.md) |
| Towers loaded when used | E2B text-only 2818 → 1826 MiB | First image / clip +0.6–0.7 s | [amd-llm.md](amd-llm.md) |
| `-idle` | 0 pinned while idle | Next request 0.8–2 s | [amd.md](amd.md) |
| Drafts checked in one call (`--verify 3`) | A 3-row check costs 1.27 decode steps; Qwen3.5-0.8B 1.15–1.3× with prompt lookup | +30 MB control code; q8 / q6 only | [llm-decoding.md](llm-decoding.md) |
| Experts in places (`--expert-places`) | gpt-oss-20b P=16: 6.90 GiB pinned, 17.2 tok/s (all: 12.22 GiB, 20.2); Qwen3.6 32 places 7.5 GiB, 5.8–6.0 tok/s (all: 8.8–8.9) | Copies chosen experts not placed | [moe.md](moe.md) |
| Host logits and top k in vector loops | gpt-oss 201088 logits 1.33 → 0.23 ms a token | | [amd-llm.md](amd-llm.md) |

### MoE on the Mac

| Change | What it bought | Details |
|---|---|---|
| Routed experts through Core AI (`llm.Routed`) | LFM2-8B-A1B runs; prompts 138 tok/s against 24 a row at a time | [moe.md](moe.md) |
| Experts on the CPU in MXFP4 (`llm.HostRouted`), read whole, prompt by parts (wave) | Qwen3.6-35B-A3B on an M3: 11.0 tok/s, 2602-token prompt 21.5 s, 2.69 GB with the NE | [moe.md](moe.md) |
| gpt-oss-20b, experts on the CPU, wired by a budget | M3 17.3 tok/s, 0.52 J a token; M2 13.9 tok/s, 0.50 J | [moe.md](moe.md) |

### Image generation

| Change | What it bought | Cost | Details |
|---|---|---|---|
| CFG's two passes per weight group | Anima 512² 3.0 → 2.7 s a step | | [image-models.md](image-models.md) |
| DPM-Solver++ 2M (`--sampler dpm2m`) | 25 steps nearer 100 Euler than 30 Euler | | [image-models.md](image-models.md) |
| CFG at the noisier steps only (`--cfg-sigmas 0.6,1`) | Anima-Aesthetic 1024² on AMD 4 min 40 s → 3 min 19 s, 4.8 → 3.8 kJ | | [image-models.md](image-models.md) |
| Modulations once a sigma; keys / values once a prompt | Not held during the steps | | [image-models.md](image-models.md) |
| Core AI: attention scale in the q norm, `abs` magnitudes, a head at a time, keys in power-of-two parts | Anima-Turbo 1024² 7.1 → 5.6 s a step (M1); Chroma1-Flash 1024² 5 min 36 s a picture (M1) | Export again | [image-apple.md](image-apple.md) |
| Core AI: load the next program ahead, swap DiT groups | A Chroma DiT call 7.2 → 5.4–5.8 s; NE holds 8.3 → 2.2–2.6 GiB | | [image-apple.md](image-apple.md) |
| `--batch` on Core AI | A Chroma DiT call 5.4 → 5.05 s a picture | Host memory per picture | [image-apple.md](image-apple.md) |
| Every size as functions of one export | One download for all sizes (Anima's per-size exports were 2.9 GB each) | First compile pays all sizes | [image-apple.md](image-apple.md) |
| Prebuilt per chip (`hwnpu-dev models package MODEL/coreai-ARCH`) | FLUX.2 klein first run on an M1 40 min → 7 min 48 s | Not for chat models or VAEs | [apple-coreai.md](apple-coreai.md) |
| FLUX.2 klein 4B on Core AI | M1: 512² 11 s, 1024² 41 s a picture | | [image-apple.md](image-apple.md) |
| Host memory back between VAE stages | FLUX.2 klein 1024² peak 5.8 → 4.1 GB | +0.4 s decoding | [image-apple.md](image-apple.md) |
| AMD: DiT weights as bfp16 on the host | A step: Anima-Turbo 1024² 4.1 → 3.6–3.8 s, Chroma1-Flash 29.9 → 22.8–25.4 s, Chroma1-HD 27.5 → 20.2 s | 9 bits a weight; export 3–4× longer | [image-amd.md](image-amd.md) |
| AMD: GELU's tanh argument in fp32 (`GELUEven`) | Anima 1024² 4.6 → 4.2 s a step | | [image-amd.md](image-amd.md) |
| AMD: cross-attention over prompt rows only (`zero_keys`) | Anima-Turbo 1024² 3.7 → 3.2 s a step | Pictures change in details | [image-amd.md](image-amd.md) |
| AMD: image attention in 64-key blocks (`ImageMHA`) | 146 → 59 ms (4096² × 16 × 128) | | [image-amd.md](image-amd.md) |
| AMD: VAE on the grid (`GridGEMM`) | Anima VAE 9.3 → 5.2 s | | [image-amd.md](image-amd.md) |
| AMD: DiT groups in one bank (`SwapVariants`) | Anima pinned 3445 → 1236 MiB | +2 % | [image-amd.md](image-amd.md) |
| AMD: DiT at q8 | Pinned 5020 → 3445 MiB; latent cosine 0.966 → 0.961 | +2 % | [image-amd.md](image-amd.md) |
| AMD: cores round to nearest even | Anima q8 step-0 error 0.088 → 0.031 | | [amd.md](amd.md) |
| AMD: all aspect ratios in one export (`share_banks`) | FLUX.2 klein 2 sizes 22.68 → 11.77 GB; 8 sizes 16.23 GB | | [image-amd.md](image-amd.md) |
| AMD: FLUX.3 klein | 1024² 1 min 43 s, 512² 32 s | | [image-amd.md](image-amd.md) |

### Vision and audio

| Change | What it bought | Cost | Details |
|---|---|---|---|
| Sigmoid / SiLU with exp in ONNX programs (`ExpSigmoid`) | RTMDet-s box error 98 → 5.5 px | 6.6 → 9.3 ms an image | [vision-models.md](vision-models.md) |
| PP-OCR's recognizer on the NPU | 98-line page 1528 → 488 ms | A program per width | [vision-models.md](vision-models.md) |
| Host resize without closures, on every core | 37–55 → 5–7 ms a picture | | [vision-models.md](vision-models.md) |
| SigLIP 2 on the NE | base 7.3–7.9 ms a picture (M3; AMD 84 ms) | | [vision-models.md](vision-models.md) |
| AMD: Whisper large-v3-turbo | Decoding 78 tok/s; 30 s window encoder 0.84 s | | [audio-models.md](audio-models.md) |
| AMD: LFM3-Audio speech, MOSS-SoundEffect, Magenta RT2 decoder | 0.43× real time; 2.8 s a step; 0.3× real time | | [audio-models.md](audio-models.md) |
| `milcpu` for small CNNs: buffers kept between values, AVX2 depthwise, direct and 1×1 convolutions, conv + ReLU in one pass | YuNet 320×256 on a Ryzen AI 9 365 core 12.6 → 2.3 ms, 243 → about 30 mJ a picture (ONNX Runtime 20 mJ, AMD's NPU 184 mJ) | | [vision-models.md](vision-models.md) |

### Everywhere

| Change | What it bought | Details |
|---|---|---|
| Identical files shared on disk (`cache.Clone`) | M3 about 50 → 20 GiB; two AMD exports 15.2 → 8.2 GiB (not on ext4) | [distribution.md](distribution.md) |
| Downloads 4 at a time | Wi-Fi and a NAS 5 → 9 MB/s | [distribution.md](distribution.md) |
| Conversation marks keep ring rows in use; memory back after conversion | 578 → 183 MB; 2.4 GB → 154 MB | [memory-and-power.md](memory-and-power.md) |

## Caveats when combining

- **lut6 and tiers**: `--tier auto` makes short-context tiers only with 4-bit tables; 8 GB machines run none (models.yaml `tier`)
- **Two-level attention** is an approximation: the error grows when attention focuses on one key. It passes passkeys to 250K. Around 64K a prompt reads 10 % slower. `-sparse=false` turns it off
- **Long prompts**: 256K at once takes about 29 min (LFM3); the prompt store is assumed
- **Noise**: swap on the M3 and the power state on AMD move numbers up to 1.4×. Compare only pairs measured back to back. On AMD check no other process holds `/dev/accel`; on a Mac look at `top` (`mediaanalysisd`)
- **4 bits on AMD**: loss depends on the model (LFM3 KL 0.036, E2B 0.097); export without `--q4` for bf16
- **Anima's DiT at 8 bits on AMD**: a seed whose trajectory branches easily can change the composition (1 of 4 prompts)
- **Kernel fixes live in the export**: an AMD export keeps the kernels it was made with (rounding, merged operators) until exported again. An export from a newer exporter can be refused by an older hwnpu (gpt-oss's resident step)
- **`-idle`**: the next request pays 0.8–2 s. **Towers on use**: the first image or clip waits 0.5–0.7 s, and the old and new banks are both pinned for that moment
- **AMD barriers**: the chunk barrier costs 0.3–0.5 ms a chunk; the step barrier 0.3 ms a step (skipped for gpt-oss's resident step, where it would reload the array: 2.4 ms)
- **Timing AMD**: `HWNPU_DEV=iron.timing=1` adds each command's round trip; read `ironlm -forced`'s step time. An `array=True` parameter that differs by layer reloads the array each command and still passes the check
- **Disk sharing depends on the file system**: ext4 shares nothing; use Btrfs or XFS (Windows: Dev Drive)
- **Core AI's first run** compiles every part once, and a multi-function `.aimodel` compiles all functions (Chroma1-Flash: 1 h 14 min for 512² and 1024²). The cache is per OS build and process name: after an OS update, or from `aicheck` / `go test`, programs compile again. macOS purges it when the disk is low (seen at 11–22 GB free); converting needs room too
- **A layout version in an export's name** means converting and compiling again once; pictures are not bit for bit the previous version's (sums in another order); the old export stays until deleted
- **Judging a numeric change on the NE**: two NPU forms can differ by 0.97 from each other and both be as near the fp32 reference. Compare each with the reference and look at pictures

## The speed ceiling

### The ceiling is how fast the NPU reads DRAM

`hwnpu-dev probe` ([PROBE.md](../../PROBE.md)), 16 layers (0.97 B):

| | M1 | M2 | M3 | AMD (Ryzen AI 9 365) |
|---|---|---|---|---|
| NPU, rated | 11 TOPS | 15.8 TOPS | 18 TOPS | 50 TOPS |
| DRAM bandwidth | 68 GB/s | 100 GB/s | 100 GB/s | 90–120 GB/s |
| int8 weights read at | 62 GB/s | 62 GB/s | 23 GB/s (M3 only) | — |
| 4-bit weights read at | 44 GB/s (lut4) | 63 GB/s | 64 GB/s | 53–55 GB/s (q4, q8, bf16 back to back) |
| Matrix products, best measured | about 5 TFLOPS (Core AI) | — | 13.7 TFLOPS (K=64, Core ML) | 15.2 TFLOPS (bfp16 weights, 4096 rows) |
| Checking drafts: 4 tokens a call cost | — | — | 1.0 steps (64: 1.10) | 1.28 steps (8: 2.83) |

- **The ANE reads weights at about 62–64 GB/s on every Apple chip measured**: the limit is the ANE's own path to memory, not DRAM. Moving bytes is the limit, not computing (int8 and lut4 read alike on the M2). Only the M1's lut4 is compute-bound (table expansion)
- **The NE's own costs** (M2, `aicheck limits`): a call's floor 160–175 µs; elementwise ops 16–20 GB/s; 4096 × 4096 one row: fp16 742 µs, int8 466, lut8 449, lut6 387, lut4 351, int4 329 ([apple-coreai.md](apple-coreai.md))
- **AMD's DMA tops out at 55–57 GB/s** with 8 channels; a 4-bit GEMV reaches it with 16 cores. But a chat model's step as exported reaches about a third of it: each operator switch rewrites the array's configuration (85–100 µs, `elfconfigs`). gpt-oss-20b's resident step runs at 93 % of its floor (2.53 GB a token: 46 ms at 55 GB/s, run at 49.4)
- **Compute is the other way round**: AMD's GEMM with bfp16 weights reaches 15.2 TFLOPS, above the M3's NE and about 3× the M1's. Compute-bound work (image DiTs, prompts) is where AMD is ahead (FLUX.2 klein 1024² about 29 s against the M1's 41)
- TOPS helps only prompts and attention over long contexts. A generated token follows [model-fit.md](../model-fit.md) section 2: weight bytes ÷ read bandwidth + attention + about 3 ms

### What newer chips may bring (estimate, not measured)

| | NPU read, expected | Generation vs M3 | Runs on hwnpu |
|---|---|---|---|
| Apple M6 (two Neural Engines, 170 GB/s) | The same if one engine still reads 64 GB/s; up to 2× if both run at once | 1.1–2× | Yes |
| Snapdragon X2 Elite (80 TOPS, 152 GB/s) | 75–100 GB/s | 1.3–1.6× | No backend |
| Snapdragon X2 Elite Extreme (228 GB/s) | 115–150 GB/s | 1.8–2.2× | No backend |

To check a new machine: `hwnpu-dev probe`, "weights GB/s" for lut4 at 16 layers. About 64 means generation like the M3's.

### What software can still move

Tokens/s ≈ read bandwidth ÷ bytes read a token × tokens advanced a read, minus time not spent reading.

| Term | Method | State |
|---|---|---|
| Bytes a token | Fewer bits (mixed lut3/lut4, fitted tables, AWQ-style scales) | GPTQ and fitted tables in; AWQ not written; a trade against accuracy |
| 〃 | MoE (read only the experts used) | In on AMD and the Mac |
| Tokens a read | Prompt lookup, MTP drafts | In on Apple; AMD has prompt lookup with `--verify` (q8 / q6); AMD MTP not yet |
| 〃 | Batching several requests on one read | Not done |
| 〃 | A smaller model's drafts (PLaMo 3 0.8B for 2B / 8B; prompt lookup hits 1 of 113 in translation) | Not measured: about 1.9× at 80 % agreement |
| Time not reading | AMD: operator switches in the decode step | Cut for every model; one resident array is in for gpt-oss only (other models: worth about 2.7× in q4) |
| 〃 | AMD prompts: weights as bfp16 like the DiTs (no dequantize a call) | Not done: probe 256 rows q4 / q8 86–88 ms against 28 |
| 〃 | Apple: fixed cost a call, switching parts, host work | M1 LFM3-1.7B step 31.2 ms = weights 26.8 (48 GB/s) + attention 4.2 + host 0.5: what is left is a tenth |
| Attention over long contexts | `SDPAGrouped`, the split, two-level attention | In |

Techniques from AMD that may carry to Apple and Intel (not tried): a gated MLP's gate and up as one linear; fewer bytes for scales on Intel; attention sinks as one concatenated column. Not needed there: biases and residuals in the linear, a resident step (Core ML / OpenVINO fuse and keep programs loaded; but a large Core AI program following another pays 50–70 ms).

## Not adopted

Cross-cutting speed ideas measured and left out (model-bringing ones are in [new-model-checklist.md](new-model-checklist.md)):

- Draft width Q=4: only 3 % cheaper than Q=8, 10 % fewer hits
- A function of 1 token a call for when no draft is kept: 4 % from an 8-token call (M1)
- Shorter rings for Gemma 4: 40.1 against 40.4 ms
- int8 instead of lut8 on the M1 alone: PLaMo 3 2B's 128-token part was cancelled on the NE (status 0xf); also needs an export per chip
- Splitting Anima's DiT into commands with host steps (AMD): the first command timed out; one graph of 4 blocks instead
- A token's commands as one chain (`ERT_CMD_CHAIN`): 19.3–19.6 against 19.6–19.7 tok/s; the 0.1 ms boundary is the firmware starting a command
- One command a token (gpt-oss): mlir-aie reads the scratchpad once a sequence, so routing broke (cosine 0.877)
- The bfp16 matrix instruction for decode attention scores: no faster, outputs 3.6 % off
- Prompt sandwich norms as one operator (`SandwichRows`): no faster (small configurations); GDN kernels at `-Os`: no smaller
- Gemma 4's MLP gate and up with GELU as one operator: 9.3 → 9.9 tok/s but KL up (0.154 → 0.158) and logits drift; off by default (`HWNPU_GEMMA4_FUSE_GLU=1`)
- Polling expert workers on the Mac (`HWNPU_MXFP4_SPIN`), threads at USER_INTERACTIVE, workers kept awake over a layer: slowed the NE's calls or no gain; code removed
- Wiring Qwen3.6's experts up to a budget on 16 GB: 8.6–8.8 against 10.6 tok/s unwired; experts over 3/4 of memory are not wired
- Core AI on image DiTs: more bits a weight (int8 is fastest on the NE), q / k / v as one linear, groups of 7 blocks, the softmax sum divided after P·V in self-attention (overflows fp16), 1/256 in int8 value weights (pictures changed), a norm's row-max division as a constant
- Core AI: a VAE stage's fp16 output passed on as is, Sana-style loading ahead for small parts: within the noise
- AMD image DiTs: IRON's tanh GELU (changed a prompt's clothes), GELU as x sigmoid(2u) (no faster), the gated residual as one operator (does not fit two input DMA channels)

## Not yet verified

### Apple

- Two-level attention: Qwen3.5 passkeys; `Refresh` 4 or 16 by default (14–32 % faster a call, KL not measured on the NE); where it starts (about 57K); the copies in `choose` (12 % of a call at 128K)
- Slowing past 128K (1.14 → 1.43 ms a 1K); the first calls after a 256K prompt (1.29 s each)
- The NE's memory beyond the weights (KV, intermediates) is not told apart
- The M4 and later: limits (`hwnpu-dev probe limit`, `probe split`), and whether a larger default part pays
- `blocks.SiLU` in the towers and multimodal models (VL descriptions, TTS) not compared with before
- `hwnpu-chat agent` with Gemma 4 / Qwen3.5 on Apple; Gemma 4 and the prompt store
- d1-Omni-2B's team answer differs from its README (HF reads 16 questions together); LFM3-Audio TTS against llama.cpp's audio; PLaMo-Embedding long texts; BERT on the ANE with real weights
- GPTQ codes in the answers (KL falls on the NE; 12 questions do not tell); lut4 + wkmeans on the NE
- Qwen3.5-2B lut6 against HF's bf16 (only against lut8 so far); Whisper's formats on real recordings
- Core AI: `.aimodel` across OS versions; the whole program again on M2 / M3 after the copies went; Qwen3.5's GDN state passed as the next inputs (18–24 MiB a step); attention chunks in 1.5× steps (3–6 % expected); keys / values as int8 inputs
- Anima: the remaining error against diffusers' fp32 (text encoder int8?), above 1024², v5 on the M3 / M4
- MoE: the M1 (no `gather_nd` of the scales) and M4; binding cost growing with experts held (memory pressure or Core AI's limit); LFM2-8B-A1B's agent tasks; gpt-oss's prompt experts (SMMLA 112 GMAC/s a core against about 190), GPTQ codes for it, the M3 busy and the M4 unwired

### Intel

**Nothing has run on a real NPU**; only accuracy on OpenVINO's CPU ([intel.md](intel.md)). Open: whether the driver's compiler accepts the models and the split form, argument names (`argNames`), int8 against int4-g128 speed, size limits (the 2 GiB of `-target intel` is a placeholder), the cost of re-recording command lists, memory with tiers, real checkpoints in hwnpu-chat, IR sharing on Linux, PLaMo 3 and Anima on hardware.

### AMD

- NPU1 (Phoenix, Hawk Point): one xclbin per step checked only on a simulated 4-column NPU
- Windows: xrtshim on real hardware, building with MSVC; ReFS / Dev Drive cloning only compiled
- Sharing on bcachefs, ZFS, NFS
- The NPU's own power: Linux 7.0 has no sensor (package numbers only; gpt-oss 0.92 J a token)
- Long contexts: exports have a fixed context (4096 / 12288)
- gpt-oss: hours-long sessions without the step barrier; `-idle` with the resident array; attention projections below q8 with GPTQ
- Drafts on Qwen3.5-2B (q6) and 4B (needs a q4 GEMV of a few rows); Gemma 4's shared layers and rings in `--verify`
- Qwen3.6's prompt attention keeps 1 pipeline (8 not measured)
- Short prompts: `ProjectionGEMM` and dequantizing are 60 % of a 16-token chunk regardless of rows (needs a row-following GEMM)
- GDN on one core a column (about 12 ms a token expected for Qwen3.6)
- A faster q6 GEMV layout; aiecc's BD-reuse warning; `iron_moep.py check` from 64 tokens; `iron_q4.py check --valid`

### Causes unknown

- Which shapes fall to the CPU at the 16384 limit and which stay (inside Core ML's compiler)
- On a 128K English document two-level attention has lower loss than full attention (full attention distracted by distant tokens?)
- AMD: a full ELF's second chunk hangs without `chunkBarrier`; Anima's DiT split with host steps timed out; a single op 1.4× apart between first and second runs (power state?); Qwen3.6 with `--command-per-step` gives wrong logits (use for timing only)
