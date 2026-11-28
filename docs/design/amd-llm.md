# LLMs on AMD's NPU

[日本語](amd-llm.ja.md)

Dense language models and their multimodal towers on AMD XDNA 2 (Ryzen AI, NPU2) through IRON. The driver, the runtime, full ELF, shared kernels and power are in [AMD](amd.md). Mixtures of experts (Qwen3.6, gpt-oss) are in [MoE](moe.md). The weight formats themselves are in [Quantization](quantization.md). Model structure, tokenizers and chat formats are in [LLM models](llm-models.md) and [Qwen](llm-qwen.md).

All numbers are from a Ryzen AI 9 365 (Strix Point) unless noted.

## Models

| Export (`iron_export_lm.py MODEL`) | Application | Models |
|---|---|---|
| `gemma4` | `iron_gemma4.Gemma4Tail` (IRON's `Gemma4` with KV-shared layers on the last row) | Gemma 4 E2B, E4B |
| `gemma4mm` | `iron_gemma4.Gemma4TailMultimodal` | Gemma 4 E2B with vision and audio |
| `gemma4full`, `gemma4mmfull` | IRON's own `Gemma4`, `Gemma4Multimodal` | reference only |
| `lfm3` | `iron_lfm3.Lfm3` (hwnpu's; IRON has none) | LFM3-1.7B |
| `lfm3vl`, `lfm3audio`, `lfm3omni` | `iron_lfm3vl`, `iron_lfm3audio`, `iron_lfm3omni` | LFM3-VL-2B, LFM3-Audio-2B, LFM3-Omni-2B |
| `plamo3` | `iron_plamo3.Plamo3` (hwnpu's) | PLaMo 3 Translate 0.8B, 2B, 8B |
| `d1`, `d1omni` | `iron_d1.D1Vl`, `iron_d1.D1Omni` (hwnpu's) | d1-2B, d1-Omni-2B (System One decision models): LFM3-VL / Omni with the full attention layers bidirectional in the prompt chunk and the final hidden state out ([amd-npu-models.md](../amd-npu-models.md) 29) |
| `ouro` | `iron_ouro.Ouro` (hwnpu's; IRON has none) | Ouro-1.4B-Thinking, evaluated, not in models.yaml ([model-estimation.md](../model-estimation.md)); 2.6B not exported |
| `qwen35` | `iron_qwen35.Qwen35` (GDN: `iron_gdn.py`) | Qwen3.5-0.8B, 2B, 4B |
| `qwen3`, `qwen2` | `iron_qwen3.Qwen3`, `iron_qwen3.Qwen2` (hwnpu's; IRON has none) | Qwen3-0.6B, 1.7B; Qwen2.5-Coder-7B-Instruct |
| `lfm2moe` | `iron_lfm2moe.Lfm2Moe` (hwnpu's) | LFM2-8B-A1B (4 of 32 experts) |

The current export lines are the `args` of `iron_export_lm.py` in [model-recipes.yaml](../../model-recipes.yaml):

| Model | `iron_export_lm.py`'s args (but `{in}` `{out}`) |
|---|---|
| LFM3-1.7B / VL / Audio / Omni | `lfm3 / lfm3vl / lfm3audio / lfm3omni --max-seq-len 12288 --q4` |
| Gemma 4 E2B, E4B | `gemma4 --max-seq-len 4096 --q4` (text only; the towers are not converted for the catalog) |
| PLaMo 3 Translate 0.8B / 2B / 8B | `plamo3 --max-seq-len 8192 --q4` |
| d1-2B / d1-Omni-2B | `d1 / d1omni --max-seq-len 1024 --prefill-chunk 1024 --q4` (a question is one chunk from position 0) |
| Qwen3.5-0.8B, 2B | `qwen35 --max-seq-len 4096 --q4 --mlp q6 --head q6 --verify 3` |
| Qwen3.5-4B | `qwen35 --max-seq-len 4096 --q4` |
| Qwen3-0.6B, 1.7B | `qwen3 --max-seq-len 4096 --q4 --mlp q8 --head q8 --verify 3` |
| Qwen2.5-Coder-7B-Instruct | `qwen2 --max-seq-len 4096 --q4 --mlp q4 --head q4` |
| LFM2-8B-A1B | `lfm2moe --max-seq-len 4096 --q4 --mlp q8 --head q8` (the experts q4nx) |

Export names: `~/models/NAME.amd-npu2-q4` (or `NAME.amd-npu2`). On a machine with only AMD's NPU, `hwnpu-chat -m NAME` picks it. `hwnpu-chat -iron DIR` and `hwnpu-agent -iron DIR` load an export directory; the tokenizer (`tokenizer.json`, PLaMo's `tokenizer.jsonl`) and the chat format come from it. Catalog flags such as `NoThink` do not apply to `-iron DIR` (pass `enable_thinking: false` in the request).

Writing a model without an IRON application: build it from IRON's parts (`Linear.split`, `CausalConv1d`, `Checkpoints`, `MHA`, `SandwichLayer` …). LFM3 and PLaMo 3 needed nothing else.

## The export

`tools/amd/iron_export_lm.py MODEL CKPT OUT` compiles the versions into one arena (as IRON's `load_versions`) and writes the ELF, the bank images with the weights, and `manifest.json`.

- Versions: `decode` (one token), `prompt` (a chunk), and when present `vision`, `audio`, `placed` (a prompt chunk that inserts tower output), `verify` (several rows).
- Per version the manifest holds the commands, the arguments, each command's scratchpad words, the carried values (token, position), the logits output, every buffer's place, and the prompt's embedding encoders.
- Word forms: Affine, Linear (`ceil(v·ratio + offset)`, fractions), row tables, and MHA's derived values computed in Go from the operator's constants (`iron.Derived`). `TestWordsMatchIRON` checks every word against IRON's (`check/words.json`).
- The same weight name can live at different places in different versions: the manifest keeps every version's, not deduplicated by name.
- Manifest `restartable`: true when IRON's `CausalLM.recurrent` is empty. `checkpoints` (state buffers, `*.slots`) and `rings` for models with state.
- Relocation addends are unsigned (bank offsets over 2 GiB).

Main flags (all in `iron_export_lm.py --help`):

| Flag | Default | Meaning |
|---|---|---|
| `--max-seq-len` | 4096 | context |
| `--prefill-chunk` | 512 for `qwen35`, 1024 for the dense others | prompt chunk |
| `--q4` | off | quantized weights (see [Weight formats](#weight-formats)) |
| `--attn`, `--conv`, `--mlp`, `--head` | q8, q8, q4, q4 | per-part format with `--q4` |
| `--q4-codes DIR` | none | GPTQ codes from `q4_gptq.py` instead of rounding |
| `--ple` | int8 | Gemma 4's per-layer token table on the host: int8 with an fp16 scale (E2B 4.4 → 2.2 GiB, KL 0.00005 from bf16) or bf16 |
| `--device-ple` | off | Gemma 4's PLE table on the device (IRON's gathers) |
| `--mha-pipelines` | 8 | prompt MHA pipelines |
| `--verify ROWS` | 0 | add the draft-checking version |
| `--scan-window` | off | IRON's sliding-window attention (reads from position 0) |
| `--per-step` | off | an xclbin per step (NPU1, no full ELF) |
| `--command-per-step` | off | for timing: a command per step, `OUT/steps.json` |
| `--check N`, `--random-weights SEED` | | write a check / random weights |

`HWNPU_EXPORT_STEPS=1` writes each version's operator list (`<version>.steps.txt`).

Export times (context 4096, chunk 1024): E2B about 6 min (peak RSS 19.3 GB), E2B with towers 10.5 min, E4B 8 min (27.4 GB), PLaMo 3 q4 4.5 min, LFM3-1.7B q4 2.5 min, Qwen3.5-0.8B 2 min.

## Runtime (`internal/llm/iron`)

- Banks are allocated as `BO_SHARE` and filled from the bank images (page cache, several threads). `Step(token, position)` and `Prompt(tokens, start)` return logits; the host samples (a token the device draws is not used).
- Device heap (64 MiB on Linux 7.0's amdxdna, also on npu4; 512 MiB only with the DKMS driver): the decode run stays resident; a prompt or tower run that does not fit is made and closed per call (about 2 ms), dropping the decode run (`dropDecode`) and making it again at the next step. Each command must stay under the heap and under amdxdna's 2 s timeout: commands are cut at export by `SCRATCHPAD_WORDS` (below).
- PDIs are allocated at program load for the image's `main` (allocated lazily they scattered and no 40 MiB run was left).
- `chunkBarrier`: before each prompt chunk, an empty instance of the decode ELF (`aiecc_empty`, 16 bytes of control code, no PDI) runs in the decode hardware context (0.3–0.5 ms, one run kept). Without it, in a full ELF with the window attention or Gemma 4's tail, the second chunk's first command hangs until `ERT_CMD_STATE_TIMEOUT` (60 s). The cause is unknown. Not run for `--per-step` exports.
- `stepBarrier`: an empty command in a second hardware context before each decode step of several commands (see [AMD](amd.md)).
- `CREATE_HWCTX` is retried for 5 s (`internal/xdna`): right after another process exits it can fail with `invalid argument` for over 1 s (that process held 3 contexts). `aie2_cmd_submit: Grab job sem failed, ret -4` in the kernel log is EINTR and is retried. A device-heap mmap interrupted by a signal is retried like an ioctl.
- `-idle` (`Release` / `Reload`) gives back pinned memory and keeps the conversation on the host (see [Memory and power](memory-and-power.md)).
- Disk: exports side by side share files (`iron.ShareExport`, see [Distribution](distribution.md)).

### Command size

| Model | `SCRATCHPAD_WORDS` | Why |
|---|---|---|
| PLaMo 3, LFM3 (`iron_hostple.SCRATCHPAD_WORDS`), dense Qwen3.5 (`iron_qwen35.DENSE_SCRATCHPAD_WORDS`) | 24 | at 32, Qwen3.5-2B q4's second prompt command was over 64 MiB (`CREATE_BO: invalid argument`) |
| Gemma 4 tail (`iron_gemma4.SCRATCHPAD_WORDS`) | 32 | at 24 IRON split the shared layers over two commands and the next chunk's first command hung; `install_tail_boundary` always cuts at the shared layers' start, so shared layers, head and draw are one command |
| Qwen3.6 | 32 | its routing assumes it (`iron_moe.SLOTS`, see [MoE](moe.md)) |

## Prompt programs

### Chunk size

| Models | Chunk | Why |
|---|---|---|
| Attention models (Gemma 4, LFM3 family, PLaMo 3) | 1024 | 2048 is 20 % slower on short prompts (LFM3: 30 tokens 536 → 409 ms, 300 tokens 597 → 469 ms; 12288 tokens the same, 11.25 vs 11.53 s) |
| Qwen3.5 (GDN) | 512 | GDN and the projection GEMMs follow the chunk's rows: 256 is faster for short prompts, 1024 for long (0.8B q6: 30 tokens 321 / 354 / 416 ms, 300 tokens 911 / 664 / 750 ms, long 317–362 / 509–565 / 575–625 tok/s at 256 / 512 / 1024) |
| none | < 256 | `flm.ProjectionGEMM` (MEDIUM) takes multiples of 256 rows; MHA's `seq_pad` is a multiple of 64 |

Gemma 4 at 256 would make its sliding layers rings (IRON rings a window layer when `3 × ring < max_seq_len`, ring = the power of two ≥ window + chunk − 1), which breaks position-only rewind and row copies. At 1024 / 4096 it does not ring; LFM3 at 12288 and PLaMo 3 at 8192 do not ring either.

### What makes a prompt chunk fast

| Change | Where | Effect |
|---|---|---|
| Last row only into the head | IRON's `CausalLM` passes `x[call.rows - 1]` | as Apple's `LastRow` |
| MHA on 8 pipelines | `--mha-pipelines` (default 8; 4 at most for a 256 chunk) | chunks 2–5× faster, energy −66 to −83 %, logits bit for bit the same. Check `prompt attention: N pipelines` in the export log and `num_pipelines` in kernel names |
| Gemma 4 KV-shared layers on the last row (`Gemma4Tail`) | E2B: last 20 of 35 layers (E4B 18 of 42) read layers 13/14's cache and write none, so a chunk needs their output only for the last row. All rows go through the cache layers; the last row goes through the shared ones in the decode shape (`Call(prompt=False)`: GEMV, one-query MHA, RoPE row). The PLE row is the last row's (host table) or the chunk's last `token` (`--device-ple`) | 4K prompt: E2B 18.3 → 7.8 s, 218 → 97 J; E4B 22.9 → 13.7 s. Later steps bit for bit the same; prompt logits differ slightly (GEMV vs GEMM: cosine 0.99992) |
| Window layers read only their window (`iron_window.py`, default) | Prompt: copy the `chunk + window` keys/values ending at `start + chunk` (64-rounded, E2B 1536) into a shared work state with `Copy`, then IRON's `MHA`; window caches have `chunk` extra rows. Decode: read the `window + 1` keys in place (`MHA` subclass with `kv_offset`, still named `MHA`). Caches stay by position, so any position can restart | a window layer's attention at 3072: 112 → 66 ms. E2B 4K 7.84 → 6.87 s (−12 %), PLaMo 3 q4 8K 60.3 → 53.1 s; decode −2 % (E2B), −14 % (PLaMo 3 at 8191) |
| GDN rule on active row blocks only (`gdn_zero_y`) | blocks with no valid row only stream x and write 0 | the operator 27 → 5 ms (32 of 256 rows) |
| GDN SiLU once before the rule (`GdnSiLU`, `iron_gdn.silu_operator`, `GDR_PRESILU`) | the rule re-read each row once per 16-column slice and redid q/k/v SiLU every time | the operator 30.4 → 9.8 ms; Qwen3.5-0.8B 512 chunk 0.9 → 0.5 s; logits bit for bit the same |
| LFM3 short convolution as one operator (`ShortConvChunk`, `iron_shortconv.short_conv_chunk_operator`, `kernels/short_conv.cc`) | B·x, 7 taps, C· in one; 8 columns × 1 core, 8-row tiles, the last 6 rows in a `hist` buffer, `valid` by RTP; in_proj as one 6144-out GEMM. IRON's form was about 26 operators a layer. Text layers only (not the LFM3-Audio encoder) | bit for bit the reference |
| Qwen3.5 GDN convolution + SiLU as one (`GdnConvChunk`) | 2 cores a column (4 need 10 memtile DMA channels, over 6); rows after the call's are 0 | 8704 ch: 3 rows 3.99 → 0.95 ms |

Current prompt times (first call of `ironlm -record`, NPU otherwise idle):

| Model | Reconfigurations | 16 tokens | Long |
|---|---|---|---|
| LFM3-1.7B q4 | 632 | 386 ms | 1010 tokens 583 ms |
| Qwen3.5-0.8B q6 | 702 | 259 ms | 300 tokens 439 ms |

Other chunk times after the pipelines: E2B with towers 0.49 s at 0, 0.63 s from 3072; E4B q4 1.04 s / 1.24 s; LFM3-VL/Omni 0.75 s at 0, 1.35 s from 11264; Qwen3.5-0.8B q6 534–494 tok/s, 2B q6 444–409 tok/s, 4B q4 193–214 tok/s.

Where short prompts still go (LFM3-1.7B, 16 tokens, by command): `ProjectionGEMM` 219 ms and weight dequantize (q4/q8 → bf16/bfp16 in DDR) 119 ms are 60 %. Both run over the chunk's full rows whatever the call's rows (`flm/gemm/op.py`: the count is in memtile lock values, out of reach of per-call words). This is a limit of IRON's GEMM.

## Decode programs

AMD's decode is bound by the count of operators, not by weight bytes: each switch reconfigures the array (about 40 µs + 2.3 µs/KB; around 70 % of a dense model's token). Qwen3.5-0.8B decodes in the same time at bf16, q4 and q6. So the work is to fuse operators. Count them with `HWNPU_EXPORT_STEPS=1` and `elfconfigs`. Fusing a layer's operators (2026-11-21, after HeteroMosaic 2607.12839) took decode from 11.2 to 17.2 tok/s on the LFM3 family, 7.7 → 9.3 on Gemma 4 E2B, 13.8 → 18.7 on Qwen3.5-0.8B and 8.5 → 8.9 on Qwen3.6, with the same KL or logits.

| Fusion | Models | Default / switch |
|---|---|---|
| `QGEMVAdd`: a projection's bias and the residual added as the GEMV's output leaves | Qwen3.5, gpt-oss | on |
| `QGEMVGlu`: gate and up as one table, `silu(Gx)·(Ux)` | Qwen3.5 (q6/q8), LFM3 (q4) | on |
| `QGEMVGlu` GELU form (`Q_GLU_GELU`) | Gemma 4 | **off** (see Not adopted) |
| `ShortConvStep` (`iron_shortconv.py`): the decode short conv in one operator, reading and writing the checkpoint slots at per-call offsets (`short_conv_operator(slots=)`, no Copy around it); in_proj as one QGEMV | LFM3 | on |
| `SandwichAdd` (`iron_sandwich.py`, `kernels/sandwich_add.cc`): post norm + residual + next norm in one core; float32 sums, one bf16 rounding; `scale` for Gemma's layer_scalar (rounded then multiplied, as IRON) | LFM3, Gemma 4 | on |
| Gemma 4 per-layer input for all 35 layers before the first layer (one 35 × 256 GEMV, one RMSNorm …) | Gemma 4 | on; logits bit for bit the same |
| GDN with the convolution and gated norm inside the decode operator (`Qwen35Layer.fold`) | Qwen3.5 | on |
| GDN arrays in Worker-local buffers (`gdn_bind`): only the 3 state limbs (`s_hi`, `s_mid`/`s_lo` at fixed offsets); `blk`, `u_row`, `y_keep` static (all of them overflowed the 16 KB program) | Qwen3.5/3.6 | on; zero blob a GDN call 258 → 61 KB |
| GDN columns aligned to 64 (`iron_gdn.COLUMN_ALIGN`) | Qwen3.5 | 0.8B/2B fused projection 97 blocks (prime) ran on 1 column: 571 → 217 µs |

Current decode (q4 exports unless noted):

| Model | Ops reconfigured / token | Speed | Pinned |
|---|---|---|---|
| LFM3-1.7B q4 | 226 | 16.8–16.9 tok/s | 1.64 GiB |
| LFM3-VL-2B q4 | 226 | 15.6 tok/s (before the slots form) | 2.0–2.4 GiB (VL/Audio/Omni) |
| Gemma 4 E2B q4 | 672 | 9.3 tok/s (107 ms) | 1775 MiB |
| Gemma 4 E4B q4 | not rebuilt | 5.6 tok/s | 3.53 GB |
| PLaMo 3 2B q4 GPTQ | | 89 ms (11.2 tok/s), under CPU load | 2.16 GiB |
| PLaMo 3 0.8B q4 | | 40–48 ms (about 22 tok/s) | 599 MiB |
| Qwen3.5-0.8B q6 | | 70.9 ms (14.1 tok/s) | 1069 MiB |
| Qwen3.5-2B q6 | | about 120 ms (before the GDN zero-blob fix) | 1862 MiB |
| Qwen3.5-4B q4 | | 157–162 ms (6.2 tok/s) | 3727 MiB |
| Ouro-1.4B-Thinking q6 | | 268–357 ms (3.7–2.8 tok/s; 96 layer runs a token) | 2.79 GiB |
| Qwen3-0.6B q8 | 314 before the fusions below | 41.6 ms (23 tok/s in hwnpu-chat) | 1166 MiB |
| Qwen3-1.7B q8 | | 68 ms (14.5 tok/s) | 2350 MiB |
| Qwen2.5-Coder-7B q4 | | 169 ms (5.8 tok/s; its weights' reads) | 4989 MiB |
| LFM2-8B-A1B (experts q4, the rest q8) | 45 commands | 53.8 ms (19.1 tok/s in hwnpu-chat) | 5445 MiB |

Energy: E2B q4 1.46 J/token, LFM3-1.7B q4 1.05 J/token (before the fusions).

As a part ([new-model checklist](new-model-checklist.md)): second load / longest command / decode — PLaMo 3 0.8B q4 GPTQ 0.5 s / 84 ms / 40–48 ms; Qwen3.5-0.8B q6 0.8 s / 85 ms / about 90 ms; Qwen3.5-2B q6 1.5 s / 109 ms / about 120 ms.

Left in the LFM3 decode (226 reconfigurations): QGEMV configs are 75–97 KB (about 250 µs to come back to); next would be ops in the QGEMV's output path, the q/k norm and RoPE merged. Left in Gemma 4's: QGEMV 206, RMSNorm 102, ElementwiseMul 72, GEMV 71 (PLE gate/projection in bf16), SandwichAdd 70, GELU 70, RoPE 50.

## Conversation state

- KV caches sit by position in the host-visible arena bank, laid out alike in every version (E2B: 15 layers × keys/values, 18 KiB a position). `Mark`/`Rewind` move only the position: the next call overwrites.
- `iron.Model.ExportRows` / `ImportRows` copy a position range; the prompt store and `Save` use them through `rowSource` (`RowBytes`, `ExportRows`). Store key: model name, manifest mtime, bank size. A 2736-token item is 51 MB. After a restart the same request: 13.2 s → 0.0 s.
- Models with state (LFM3's convolutions, Qwen3.5's GDN): IRON's `Checkpoints` keep a slot per chunk boundary (slot k = state after position k × chunk). `StateAt(pos)` / `SetStateAt` copy only the slot a call from `pos` reads (LFM3 13 layers: 312 KiB); with the KV rows that restores `pos` fully. `Prompt` cuts calls at chunk boundaries. Marks, `Rewind`, `Save` and the store carry the copy. `TestStateRewindOnDevice` (`HWNPU_TEST_IRON_RECURRENT_EXPORT`).
- `Resumable()` refuses exports with rings (`Rings > 0`) or inconsistent slots; the engine does not load a non-restartable export without it.
- `Engine.markTurns`: for state models that are not restartable, prefill also marks after the last turn's header (`<|im_start|>assistant\n`), because the chat template drops earlier thinking and the common prefix ends at the previous assistant turn.
- `ironSession.StepTokens`: a short read goes through decode steps when tokens × step time < chunk time (the session measures both).
- Media: tower soft tokens enter as negative IDs; the store matches only before the first negative ID and does not store sequences with them (IDs are reassigned per process).

## Multimodal towers

Towers stay bf16 with `--q4`. A tower writes soft tokens into the shared state `merged`; the `placed` chunk takes IDs and per-marker `before` and inserts them. One `merged` per tower, so at most one image and one clip a prompt.

| Tower | Form |
|---|---|
| Gemma 4 vision | input: patches after the host resizes to the processor's size (48 px blocks hk × wk, hk·wk ≤ 280; PIL-compatible bicubic, `ImageSize`), so the tower's own resample is 1:1. Words without a closed form tabulated at export for all 1600 (hk, wk) |
| Gemma 4 audio | 16 kHz waveform with a hop of zeros in front (`AudioTokens`); words tabulated by frame count (1–3000). Its control code (39.8 MiB) does not fit beside the decode run, so the decode run is dropped while it runs |
| LFM3-VL vision (`iron_lfm3vl.py`) | one size, 512 × 512 (host stretches); gated 3×3 depthwise conv as 9 shifted reads of a zero-banded grid plus 2 edge masks (IRON's depthwise `Conv2d` builds a k × width × channel band matrix, too big at 1024 ch); 2-D RoPE as weight tables; 2×2 merge by one 4-D DMA read |
| LFM3-Audio encoder (`iron_lfm3audio.py`) | host log-mel (`lfm3audio.Features`), up to 4096 frames; first conv as `Conv2d`, two depthwise as 9 strided reads, the row after each stage's valid rows zeroed; 17 LFM2 layers with `CausalConv1d`; head dim 96 padded to 128 (scale 1/√96, RoPE on 96 columns, head-norm weight and eps × 0.75). Words by frame count. No audio output (text answers) |
| LFM3-Omni (`iron_lfm3omni.py`) | both towers under HF names (`model.vision_tower`, `model.audio_encoder`); image and audio between `<|vision_start|>` and `<|vision_end|>` |

- Manifest `kind` picks the preprocessing (`RunVision`, `RunAudio`, `PromptPlaced`). LFM3-Audio has no audio marker.
- Address-dependent values (`recording.addresses`) are in the manifest and computed in Go.
- **Towers load only when used** (`loadBanks`, `growBanks`, `Manifest.textBanks`): the text weights and work areas lead the bank, so decode and prompt allocate only that prefix. Before the first tower or `placed` run, a full buffer is allocated, the text part copied with the device's caches and state, the tower part read from the image, and the old bank and its decode run freed. The conversation continues. Full until `Release`; `Reload` after `-idle` is text-only again. NPU1 (`--per-step`) reads the whole bank.

| Pinned | text only | after a tower |
|---|---|---|
| Gemma 4 E2B with towers, q4 | 1826 MiB (text export: 1775) | 2818 MiB |
| LFM3-Omni-2B | 1679 MiB | 2448 MiB |

Loading a tower: 0.5–0.7 s; both banks are briefly pinned (E2B q4: 1.67 + 2.64 GiB). Test: `TestLazyTowerBanks`. One export with towers serves text alone, so no text-only twin is needed.

## Speculative verify

A prompt chunk checking 2–8 tokens costs a full chunk (3–9 decode steps): Qwen3.5-0.8B q6 350 ms against 87.6, PLaMo 3 0.8B 345 against 40, E2B q4 411 against 128.5. So `--verify ROWS` (`iron_verify.py`) adds a version that runs the decode step's operators over ROWS rows:

- `QGEMVRows` (`kernels/mv_rows.h`): weights streamed once, each block's codes made bf16 once for all rows; per-row sums and rounding as the 1-row GEMV (bit for bit). q8 3584 × 1024: 1 row 159 µs, 3 rows 155–162, 4 rows 188, 8 rows 389. q8 and q6 only, up to 8 rows (`ROWS_GEMV`).
- Embeddings: host rows. Short conv: `causal_conv1d_rows`. RoPE: rows from `start`. Attention: per row, write K/V then the decode `MHA`.
- GDN: the decode operator over ROWS rows in one call (`gdn_operator(..., steps=ROWS)`, `GDR_STEPS`), state carried in the core; each row's state out, the last to the slot, the others to `rows_state`. (Three calls cost 1.0 ms × 54 and made the ELF 47 MB, over the heap.)
- `iron.Model.Keep`: when fewer rows are accepted, the host copies row kept−1 of `rows_state` to the slot (Qwen3.5-0.8B 18 layers × 1.1 MB, 1.5 ms). Extra K/V rows are overwritten later. Checks across a chunk boundary use decode steps (`SpeculateTokens`).
- Control code: verify ELF 30 MB beside decode 26 MB in the 64 MiB heap; otherwise per call (`perCall`).
- Pass: `ironlm -verify` — 3 rows' logits bit for bit 3 decode steps; steps after `Keep` bit for bit.
- Qwen3.5-0.8B q6 (at 82 ms decode): verify 104.5 ms (1.27 steps). With 2 prompt-lookup drafts (`LookupDraft`): code 82.4 → 72.1 ms/token, Japanese prose 82.5 → 61.7. `hwnpu-chat -iron` uses the version when present; `HWNPU_DEV=iron.drafts=0` turns it off. In the catalog only Qwen3.5-0.8B and 2B carry `--verify 3`.

### Selective draft checks

The three-row check now costs 1.96 decode steps for Qwen3.5-0.8B and 1.67 for 2B. On Ryzen AI 9 365, checking every draft made prose slower than decoding without drafts. The AMD-only `chat.draftPolicy` tracks the acceptance rate q by the length of the preceding match and checks only when `1 + q + q²` exceeds the check cost (default 1.8 decode steps, with a 3% margin). Drafts it skips still update q when compared with later tokens. `HWNPU_DEV=iron.lookup.cost` changes the cost and `HWNPU_DEV=iron.lookup.trace=FILE` records decisions (checking every draft, `draft.policy=0`, was removed on 2026-11-27). Apple and Intel are unchanged.

On 300-token, temperature-0 runs, prose on the 0.8B model went from 14.6 tok/s with every draft checked to 18.8 with the policy (18.8 without drafts); on 2B, 9.6 → 11.5 (11.2 without drafts). Copy-heavy answers retained their gains: code on 0.8B 26.9 → 27.2 tok/s, JSON on 2B 14.2 → 13.6. Temperature-0 output can differ between drafted and undrafted decoding; this is not investigated.

See [Decoding](llm-decoding.md) for the drafting side.

## Weight formats

Formats are defined in [Quantization](quantization.md). As used here (with `--q4`):

| Part | Format | Decode | Prompt |
|---|---|---|---|
| MLP, head | q4nx (FastFlowLM's, 5 bits) | `QGEMV`, 2 cores a column, 55 GB/s as bf16 | `flm.DequantBFP` + bfp16 GEMM |
| Attention projections (incl. LFM3's gate), LFM3 conv projections | q8 (int8, bf16 scale per 32 in, 8.5 bits) | `QGEMV` with partial sums as two bf16 (high + rest) | `Q8Dequant` / `QDequant` + bf16 GEMM |
| Qwen3.5-0.8B/2B MLP and head | q6 (6.5 bits, q8's layout, `kernels/q6_mv.cc`) | about q8's time (bound by extracting the high bits) | `q6_dequant` |
| Token embeddings | host table (`embed.bin`, `embed_rows`) | rows written before each call | |
| Gemma 4 PLE | host table (`ple.bin`, int8) | | |
| Towers | bf16 | | |

- Fused weights (`qkv_proj`, `gate_up_proj`, LFM3 `in_proj`) are quantized whole; row ranges cut as column-block byte ranges (`iron_q4.q_rows`). Decode runs one QGEMV and cuts the output (6 % faster than split).
- Relative RMS weight error: q4 0.071, q6 0.022, q8 0.005. GEMV 8192 × 2048: q4 0.30–0.35, q6 0.42–0.47, q8 0.43–0.50 ms.
- Qwen3.5's head is tied to the embeddings but holds a quantized copy, so the bf16 table on the device was unused and moved to the host: 0.8B −0.47 GiB, 2B −0.95 GiB, logits bit for bit the same. Qwen3.6 keeps it on the device (its routing reads it).
- GDN fused projection padded per column with zero rows (`iron_gdn.column_widths`: 772 → 776, then `COLUMN_ALIGN` 64) so quantized blocks (64 out × 512 in) fit.

Effect of `--q4`:

| Model | Pinned | Decode |
|---|---|---|
| Gemma 4 E2B | 9.08 → 1.78 GB | 6.1 → 7.9 tok/s (before the fusions) |
| Gemma 4 E4B | 14.7 → 3.53 GB | 3.7 → 5.6 tok/s |
| LFM3-1.7B | 3.75 → 1.64 GiB | 8.7 → 11.2 tok/s (before the fusions) |
| PLaMo 3 2B | 5.96 → 2.16 GiB | 7.3 → 11.6 tok/s |
| PLaMo 3 0.8B | 1814 → 599 MiB | 17.5 → 22.8 tok/s |
| Qwen3.5-0.8B (q6) | 1858 → 1069 MiB | same (op-bound) |
| Qwen3.5-2B (q6) | 4017 → 1862 MiB (chunk 512) | 131 → 123 ms |

### Accuracy against HF fp32

KL(HF ‖ NPU) over 400 rows each of README / GPL-3 / wiki40b-ja (`ironlm -forced`, `tools/ref/forced_logits.py`):

| Export | README | GPL-3 | wiki |
|---|---|---|---|
| Gemma 4 E2B bf16 | 0.0059 | 0.0056 | 0.0038 |
| E2B q4 round / GPTQ | 0.123 / 0.091 | 0.171 / 0.145 | 0.220 / 0.081 |
| Gemma 4 E4B bf16 | 0.0138 | 0.0040 | 0.0031 |
| E4B q4 round / GPTQ | 0.047 / 0.025 | 0.089 / 0.077 | 0.085 / 0.051 |
| PLaMo 3 2B bf16 | 0.0040 | 0.0118 | 0.0022 |
| PLaMo 3 2B q4 round / GPTQ | 0.097 / 0.044 | 0.093 / 0.088 | 0.055 / 0.040 |
| PLaMo 3 0.8B bf16 | 0.0046 | 0.0038 | 0.0031 |
| PLaMo 3 0.8B q4 round / GPTQ | 0.095 / 0.043 | 0.094 / 0.061 | 0.094 / 0.072 |
| Qwen3.5-0.8B bf16 / q6 | 0.0014 / 0.0079 | 0.0056 / 0.044 | 0.0012 / 0.0077 |
| Qwen3.5-2B bf16 / q6 | 0.0009 / 0.0047 | 0.0017 / 0.019 | 0.0007 / 0.0062 |
| Qwen3.5-4B q4 / q6 | 0.023 / 0.004 | 0.042 / 0.005 | 0.046 / 0.040 |

Current fused exports, mean / median / p95 over 320 rows after 256 tokens: LFM3-1.7B q4 0.0643 / 0.0494 / 0.171 (top-1 86.6 %); LFM3-VL-2B q4 (text) 0.1188 / 0.0589 / 0.511; Qwen3.5-0.8B q6 0.0084 / 0.0074 / 0.020 (94.1 %); Gemma 4 E2B q4 0.1539 / 0.0684 / 0.600 (82.2 %).

- PLaMo 3 translation at q4: KL on the translation side 0.007–0.009; 36 of 44 sentences the same as bf16, the rest correct rewordings. PLaMo 3 0.8B GPTQ: 20/20 sentences as bf16.
- LFM3 at q4: most error is the 4-bit MLP (0.031); attention and conv at q8 are nearly lossless (0.0004).
- Qwen3.5 0.8B/2B at q4 collapse on GPL (0.8B 0.336, GPTQ still 0.300; 2B 0.163) from the MLP: use q6. 4B holds at q4.

## Qwen3, Qwen2 and LFM2-MoE (2026-11-28)

`iron_qwen3.py` holds Qwen3 (q and k normed per head) and Qwen2 (q, k and v biased); `iron_lfm2moe.py` reuses its attention (LFM2's `q_layernorm`, `k_layernorm`) and LFM3's short convolution.

- **q8 against q6 on op-bound models.** Qwen3-0.6B's decode step was 63 ms at q6 and 50 ms at q8 (1.7B: 92 against 68 ms), and the GPL's KL 0.099 against 0.023 (1.7B: 0.081 against 0.019). q6's GEMV unpacks two planes a block and costs more than its fewer bytes save while a step is bound by operator switches. A q4 MLP broke Qwen3-1.7B (KL 1.31). Qwen3.5-0.8B and 2B (q6) may gain the same way: not measured yet.
- **The head's rows on 8 columns.** Qwen3's vocabulary, 151936, is 2374 blocks of 64 (2 × 1187), so its QGEMV ran on 2 columns. The quantized head is padded with zero rows to a multiple of 512 (`HEAD_ALIGN`), Sample draws from the vocabulary's rows (`Qwen3.draw`), and hwnpu reads `vocab_size` of the output. `iron_verify` takes the head's own width for the last row.
- **A pre-norm decoder's sandwich GEMV.** `iron_q4.sandwich_linear(h, r, None, w, eps, W)` (`sandwich_operator(plain=True)`, `Q_SANDWICH_PLAIN`): y = r + h, n = rms(y) w, then the GEMV of n: one operator in place of an add, an RMSNorm and the GEMV. Qwen3's decode step makes the residual and the next norm in the qkv projection, the MLP's QGEMVGlu and the head (o_proj and down_proj add nothing), and one HeadRMSNorm norms q's heads then k's: 48 → 41.6 ms on 0.6B, the same KL. Qwen2's biased qkv keeps its own norm, and down_proj adds the residual (into_linear).
- **Prompt commands.** Attention layers ask few scratchpad words, so IRON's cuts left Qwen3-0.6B's 1024-row chunk two commands of 66 MB of control code (CREATE_BO: resource temporarily unavailable): a command starts every 7 layers (`install_cuts`, as Ouro's).
- **Qwen2.5-Coder's MLP of 18944.** A prompt chunk's gate and up GEMM of 296 column tiles needs more than one buffer descriptor for its per-call output: gate and up run in two pieces, and down is two tables of in-features (9216 + 9728, whole 512s) used by every version (the decode step's second GEMV adds the first's output and the residual). Rounded q4nx (no codes) gives KL 0.028 (README) / 0.018 (GPL) against fp32, where Apple's rounded lut4 gave 0.90. 1024 tokens of prompt in 3.0 s at 0, 3.9 s from 3072.
- **LFM2-MoE.** A decode step's MoE layer is two commands, as Qwen3.6's (`iron_moe.py`): `MoERouteSigmoid` (`kernels/moe_route_sigmoid.cc`: sigmoid scores, the top 4 of score + expert bias, weights over the chosen's sum) writes command B's scratchpad; `MoE4GateUp` puts each chosen expert on two columns (half its column blocks each) and `MoE4Down` reads the 4 experts' down side by side and adds the residual (kernels/moe_mv.cc unchanged). Qwen3.6's prompt group operators take 64 out-features a column and 2 k-blocks an expert (LFM2's 1792 is 28 blocks and 7 k-blocks), so a prompt chunk at first ran every expert on every row (8 times the chosen's work) into a state, the host routing and combining; it now runs only what its rows chose, in two forms (below). q4nx now packs a K of whole 256s (the same bytes as before for whole 512s), and `q4_mv.cc` takes them for its dequantization. KL against transformers in bf16: 0.075 (README) / 0.079 (GPL); against the same with its experts rounded to q4nx 0.035 / 0.026, and the export with everything but the experts in bf16 gave 0.033 / 0.028: the rest is not q8's, but routing that flips on near ties (medians 0.005–0.02). fp32 (33 GB) did not fit beside the other sessions. A prompt chunk ran an expert's operators in turn at first (dequantize, GEMM, SiLU, ...: 9 changes of operator an expert, 6724 reconfigurations and 499 MB of configuration a chunk): 1024 tokens in 3.91 s, and about 125 ms a MoE layer whatever the rows, so no prompt was under 2.8 s. Then the experts ran in groups of 4, a group's same operators in a row: 1796 reconfigurations, 1024 tokens in 2.94 s (348 tok/s), a 33-token prompt 2.9 → 1.9 s in hwnpu-chat, the same pinned memory and the same logits bit for bit (the operators and their inputs are the same); prompt.elf 577 → 190 MB. Groups of 8 gave 2.84 s for 48 MB more pinned (the group's dequantized weights are held together), 16 gave 2.81 s for 165 MB. Measured on one expert's operators at 256 rows (a graph of 8 experts): 3.25 ms an expert in turn, 2.16 ms with the same operators in a row, 1.15 ms with the weights already expanded (bfp16), so dequantizing is about 1 ms of it.
- **What the rows of a prompt choose (LFM2-MoE, 5 texts).** A 33-token prompt already uses 23 of a layer's 32 experts (128 tokens: 28.5; 512: 30.9), about 6 rows each, so leaving out the experts no row chose saves little: the waste is the rows an expert was not chosen for. An expert's GEMMs on 256 rows cost about 2.2 ms (above), a row through a q4 GEMV about 0.11 ms (Qwen3.6's grouped operator, 0.1 TMAC/s against the GEMM's 6), so a chunk of few rows goes faster row by row and a long one in the GEMMs. Porting Qwen3.6's grouped operator alone would make long prompts slower (about 450 ms a layer at 1024 tokens).
- **A short chunk's rows run their own experts alone (LFM2-MoE).** A MoE layer of a prompt chunk has two forms and the host takes one by the chunk's rows. Up to 80 rows (`ROW_MAX`), the host routes the rows from the router's logits (`moe_rows_plan`: the kernel's rule, internal/llm/iron/moe.go) and runs one command once for every 2 rows (`ROW_TOKENS`: a pass's values are the rows' places and their experts, 10 scratchpad words a row); the command is the decode step's `MoE4GateUp` and `MoE4Down` with a row's place in the buffers as one more per-call offset (`expert_operators(at=True)`), so a row reads its 4 experts in place and nothing is dequantized. The other form's command is left out; `moe_gather_combine` then only copies the rows into the layer's output (a state the host writes cannot be a step's output too). Measured with ironlm (a prompt and one decode step): 4 tokens 0.3 s, 16 0.5 s, 33 0.8 s, 64 1.3 s, 80 1.6 s, 96 1.9 s, so about 17 ms a row (0.8 ms a row a MoE layer: a row reads its 4 experts' 27 MB again, and a pass is a command of its own with two changes of operator); in hwnpu-chat a 33-token prompt 1.9 → 0.7 s, one of 13 tokens 0.4 s, and a conversation's further turns (22 new tokens after the kept ones) 0.5 s each. The two forms' logits differ as a decode step's do from a prompt's (the q4 kernels' sums against the GEMMs'): KL 0.007 between them after a 32-token prompt, the top token the same in every row; against transformers in fp32, after prompts of 48 and 90 tokens of two texts, 0.088 / 0.050 / 0.180 / 0.153 by rows against 0.081 / 0.055 / 0.180 / 0.143 by the GEMMs, and after their prompts of 1100 tokens (the last 76 by rows) 0.054 / 0.077 against 0.053 / 0.077.
- **A longer chunk gathers each expert's rows (LFM2-MoE).** Every expert on every row took 81 ms a layer at 256 rows, 93 at 512 and 117 at 1024 (`HWNPU_DEV=iron.timing`), and the host 4 to 5 ms more to combine (it flushed all 134 MB of the experts' output): 88 % of a 1024-token chunk's 3.0 s. The rows are the small part: 12 ms for 256 rows of the 32 experts' GEMMs (7.5 TMAC/s) on 69 ms that do not depend on them, 2.2 ms an expert of which about 1 ms dequantizes it. So the chunk now pays for the experts it uses and the rows they were chosen for: the host routes the rows (`moe_gather`), gives each expert the rows that chose it in slots of 256 rows (`SLOT_ROWS`, the GEMM's unit; an expert of more rows takes further slots, 33 to 37 slots for 1024 tokens), and runs one command of 4 slots (`GATHER_SLOTS`) once for every 4: the slots' rows written into the command's input (padded with zero rows to whole 64, so that nothing a slot held before shares a block of the GEMM with them), their experts per-call offsets on the dequantizing operators (`dequant_at`: iron_q4's QDequant with `offset_parameter`), the GEMMs IRON's as before, and the slots' rows read back; `moe_gather_combine` adds each row's 4, weighted, to the residual in float32. The logits are the same bit for bit as with every expert on every row (a row of the GEMM does not depend on the others; checked at 90, 300 and 1100 tokens). Measured with ironlm (a prompt and one decode step): 97 tokens 2.0 → 1.6 s, 200 2.3 → 1.7 s, 500 2.5 → 1.9 s, 1024 3.1 → 2.3 s; a layer of a 1024-row chunk 79 to 93 ms and 5 to 8 ms on the host; in hwnpu-chat prompts of 110, 225, 507, 989 and 1375 tokens take 1.6, 1.7, 1.8, 2.1 and 3.8 s (507 and 1375 took 2.4 and 5.3 s). The state of every expert's output is gone: 126 MB less pinned (5445 MiB on the device against 5565, 5554 MiB at the peak of a 1542-token prompt), prompt.elf 195 → 53 MB. The forms cross at 80 rows (64 rows 1.3 s by rows against 1.5 s gathered, 80 1.6 against 1.6, 96 1.9 against 1.6). What is left is the slot itself (below), and a last run's unused slots, which still run (1.5 of 4 on average).
- **What a slot costs, and what was tried on it (LFM2-MoE, 2026-11-28).** Each operator alone, 16 in a row in one command: QDequant 0.30 ms a table, the GEMM 0.28 ms at 256 rows (0.225 where B is bfp16 already; about 0.125 of it the products), the Copy of a slot's rows 0.17, the product 0.075, SiLU 0.06. A slot's 2.03 ms were so 0.90 dequantizing, 0.84 in the GEMMs and 0.30 after them, and their sum is the run's: the 8 reconfigurations of a run of 4 slots are little. Tried on a run of 4 slots, the variants timed in turn over 8 rounds (min / median), against 7.96 / 8.32 ms as it was and 4.58 / 4.67 with the weights packed on the host, nothing dequantized (the bound of anything that keeps IRON's GEMM):
  - QDequant on two cores a column (`dequant_at`: a column block's pair of blocks split in the memory tile, the halves' rows joined there): 0.24 ms a table, 7.34 / 7.61 ms, the weights the same bits (compared on a layer's tables). It is bound by what it moves (2.3 MB in and 7.3 MB of bf16 out a table): a stream out for each core gave no more. **Taken**: a layer of the gathered form 63 → 59 ms at 128 tokens, 71 → 66 at 500, 85 → 78 at 1024, the logits the same bits; in hwnpu-chat prompts of 110, 507, 989 and 1375 tokens 1.5, 1.8, 2.0 and 3.7 s (1.6, 1.8, 2.1 and 3.8 before).
  - IRON's DequantBFP for gate and up (q4nx straight to the GEMM's bfp16 on 4 cores a column; down's 1792 in-features are no whole 512s) with the GEMM reading bfp16: 0.23 ms a table and 0.225 a GEMM, 7.21 / 7.61 ms alone and 6.63 / 6.71 with two cores for down and no Copy. But it rounds toward −∞, and a slot's output moved by 2 % (rms) from the nearest-even one: not taken.
  - The down's GEMM given the slot's rows (a view of the state) as its output, no Copy: 6.89 / 7.04 ms with the two cores, the same bits. But in the model the chunk after one that ran so (1024 rows gathered, then 76 a row at a time) stopped at the command after the experts with ERT_CMD_STATE_TIMEOUT, as other operators writing into a state's view did ([amd-npu-models](../amd-npu-models.md)); the ELF holds one more configuration after such a GEMM. Not taken.
  - SiLU as the gate GEMM's epilogue: nothing (7.07 / 7.29 against 7.03 / 7.28): another configuration of the GEMM.
  - Dequantizing inside the GEMM's cores: not built. B is broadcast to a column's 4 cores, so each would expand the same weights: at QDequant's rate about 0.25 ms a table, what the operator of its own takes now, for a GEMM kernel of our own.
- **Fewer places than experts (LFM2-MoE, `--expert-places P`).** As Qwen3.6's and gpt-oss's ([moe](moe.md)): a MoE layer's tables hold P places, every expert's tables are in `experts.bin`, and hwnpu copies an expert into a place when a decode step, a pass of rows (its 8) or a pass of slots (its 4) chooses one that is in none (`placePass`); the dense layers have no entry in `experts_pool`. With 16 places the logits are the same bit for bit as with all 32 (and `HWNPU_DEV=iron.moe.check` finds every place its expert's bytes), 3203 MiB at its peak against 5644 (3133 MiB pinned; the experts' file, 4.8 GB, is the page cache's to keep or drop), loaded in 0.8 s. A decode step copies in 11 to 22 % of its 88 choices (three texts, 300 tokens each; the simulation said 18 %): 62.8 to 66.9 ms (15.0 to 15.9 tok/s) against 51.7 ms with the experts' file in the page cache, 69 to 81 ms (12.3 to 14.5 tok/s) when every copy is read from the SSD (`HWNPU_DEV=iron.moe.direct=1`, what a machine without memory to spare for the page cache gets). Of the 11 ms, 3 to 6 are the copies, 3 the commands waited for one at a time. Prompts: 33 tokens 1.1 s (1.8 from the SSD) against 0.9, 128 1.9 (2.4) against 1.6, 1024 2.5 (3.0) against 2.3. A pass of rows copies more than a pass of slots, so the rows form ends at 48 rows there (`ROW_MAX_PLACES`). LFM2-MoE uses its experts evenly (26 of 32 a layer within 64 tokens), so fewer places cost more than gpt-oss's 3.5 % at 16 of 32. In hwnpu-chat as an 8 GB machine: 3182 MiB at its peak, 16.9 tokens/s, 989 tokens of prompt in 2.0 s. It is AMD's export for machines under 16 GB: models.yaml's `machine` takes a table by vendor (`{apple: 16, amd: 8}`; Apple's form needs 8 GB), the recipe's base args have `--expert-places 16` and its variant of 16 GB every expert. The ai package's record of what a model was prepared with (its spec) now holds the export where variants change it: before, a machine whose memory picks another export took the one of the first run (gpt-oss-20b's 16 and 24 GB too).
- **Fewer experts cost too much here** (CPU, float32, KL against the model itself over 5 texts; the q4nx experts alone cost 0.082): top 3 of 4 everywhere 0.31, top 2 1.25, dropping experts under 0.3 of the top weight 0.22 (3.64 experts a token), under 0.5 without renormalizing 0.29 (2.77), a 256-token chunk's 24 most voted experts 0.20. LFM2-MoE's weights are flat (0.40, 0.26, 0.19, 0.15 on average). A 4-bit mantissa with a power-of-two exponent a block of 8 (the matrix instruction's own form, 5 bits a weight as q4nx) cost 0.116.
- **Speed left.** Coder's decode reads its weights (about 4.7 GB a token); it has no check of drafts, as q4 has no rows GEMV. LFM2-MoE's decode step is 250 reconfigurations a token (about half its time by the cost model of [amd](amd.md)): one resident array a layer would remove them.

## Porting notes

- **LFM3** (`iron_lfm3.py`): conv layer = `in_proj` split B, C, x; `CausalConv1d` (width 7) on B·x with state in `Checkpoints`; × C; `out_proj`. Full attention: per-head q/k RMSNorm, RoPE with the back half of frequencies zero ("zero-tail"), per-head sigmoid gate as `gate_proj` rows repeated per head column and an elementwise product. Sliding layers: window 256, θ 10000. Sandwich norm, SwiGLU, final `embedding_norm`, tied head.
- **PLaMo 3** (`iron_plamo3.py`, check: `iron_plamo3_check.py`): RMSNorm `(offset + w)·x` with per-norm offsets (1; after attention 1/5; after MLP 1/5^1.5) added into the weights in float32 then rounded to bf16. One layer in 8 global (7, 15, 23), others a 2048 window. Window RoPE θ 10000; global θ 10⁶ with YaRN (factor 256, original 4096, beta 32/1; built from `rope_scaling_factor` and `initial_context_length`, not config.json's `rope_parameters`), `rope_attention_scaling` 0.1 ln 256 + 1. Untied `lm_head`, no embedding scale. Tokenizer and PFML3 chat format: [LLM models](llm-models.md).
- **Ouro** (`iron_ouro.py`): a looped decoder, its 24 layers run four times a token with the same weights ([LLM models](llm-models.md)). IRON's `SandwichLayer` with Ouro's norms renamed (`input_layernorm_2` → `post_attention_layernorm`, `post_attention_layernorm` → `pre_feedforward_layernorm`, `post_attention_layernorm_2` → `post_feedforward_layernorm`). `self.layers` is `model.layers` four times; `Ouro.layer` gives a pass after the first its own `KVCache` through the call (`call.ouro_cache`, buffers `passes.P.layers.L.cache.keys`, which the runtime reads as KV caches by `.cache.`) and runs `model.norm` after a pass's last layer. The weights stay in the banks once (IRON's `walk` takes a module once). The checkpoint's `model.early_exit_gate` is taken out of the tree (IRON refuses extra weights). q6 MLP and head with gate and up as one table (`QGEMVGlu`, `HWNPU_OURO_FUSE_GLU=0`), q8 attention: host rounding KL over the five evaluation texts (`q4_gptq.py ouro --method rtn`, transformers 4.54) q4 0.130, q6 0.013, q8 0.0012. Context 2048: the caches are 96 layers × 8 KiB a position (1.6 GB at 2048), the banks 2.86 GB; export 43–48 s, peak RSS 7.7 GB. A prompt chunk's command starts every 12 layer runs (`install_cuts`, `HWNPU_OURO_LAYERS_PER_COMMAND`): the passes read the same weights, so IRON's cuts (by a command's scratchpad words) never came, and a chunk was one command of 233 MB of control code (160 MB in bf16), which the 64 MiB device heap refused (`control code buffer: CREATE_BO: resource temporarily unavailable`; other exports' largest command is 37 MB). Now 8 commands of about 29 MB; the decode step stays one (36 MB). On a Ryzen AI 9 365 (2026-11-22): against Hugging Face (`ironlm -forced` on readme.md, 256 + 64 tokens, `forced_logits.py` with transformers 4.54) q6 KL 0.026 / median 0.0036 / p95 0.108, top-1 98.4 %; bf16 KL 0.0022, 98.4 %. q6 decode 268 ms a token at 300 (3.6–3.7 tokens/s), 312 ms at 1024, 357 ms at 2047; bf16 321 ms. A chunk of 1024 2.50 s from 0, 3.11 s from 1024 (409 / 329 tokens/s; each command well under the 2 s limit). Pinned 2.79 GiB (`drm-total-memory`), process 0.82 GB, load 1.1 s. hwnpu-chat runs it (3.6 tokens/s greedy).
- **Gemma 4**: IRON's application; hwnpu replaces the prompt forward (`Gemma4Tail`) and moves PLE and embeddings to the host (`iron_hostple`). Model name stays `Gemma4` (`hwnpu_name`), which picks the chat format.
- **Qwen3.5 dense**: see [Qwen](llm-qwen.md) for the model; GDN operator details are shared with Qwen3.6 ([MoE](moe.md)).

## Checking a port

- Against HF fp32 with real weights: `ironlm -forced` then `tools/ref/forced_logits.py MODEL CKPT RUN.bin`. A bf16 export lands at KL 0.001–0.014, top-1 96–99 %.
- Random weights at real shapes do not match IRON's `OnHost` reference on the NPU (cosine 0.45–0.83, Gemma 4 and PLaMo 3): check with real weights. Small random models (few layers) do match on the host.
- Measure KL with BOS in front (LFM3 without BOS: 0.139 instead of 0.035). Qwen3.5 uses `<|endoftext|>` 248044; LFM3 BOS 124894.
- `forced_logits.py` on LFM3: the checkpoint's `auto_map` makes transformers load remote code; pass a directory with `auto_map` removed from config.json.
- `ironlm -record` writes an export's own logits as `check/`; another export run on it compares (used for every refactor: "bit for bit the same").
- `ironlm -depth` times a chunk and a step at each boundary up to the context; `-bench` times few-token calls; `-steps N` decode speed; run old and new alternately, NPU otherwise idle.
- Per-op timing: `--command-per-step` + `HWNPU_DEV=iron.timing=1` (sums about 10 % above a normal export).
- Check every tower with `hwnpu-chat` (an image and a clip) after any export change.
- Check downloaded checkpoints' sha256 against `x-linked-etag`: LFM3-Audio and Omni safetensors once had bit flips that no random-weight check found.

## Not adopted

- Speeding prefill with a GPU or a second machine (oMLX's Prefill Accelerator): hwnpu targets low power; reduce the work instead (Gemma 4's tail).
- Checking drafts with the prompt chunk: costs 3–9 decode steps, more than prompt lookup's 1.07–1.3 accepted tokens can repay.
- A two-stage output head (a 2–3-bit copy to shortlist rows, then exact scores): greedy tests read 34% of Qwen3.5-0.8B's head, 50% of Qwen3.5-2B / Qwen3.6's and 40% of gpt-oss's, but keeping 64 candidates reads 56–83% or more; Gemma 4 E2B's copy error was unusable. The head is only 5–15% of a step, so the estimated gain is a few percent without an NPU implementation.
- Prompt chunk 2048 (20 % slower) or 256 for attention models (Gemma 4's window layers become rings).
- Attention at 4 bits (`--attn q4`): E2B KL 0.229 vs 0.097 at q8; attention is about a tenth of the weights but most of the 4-bit error.
- q8 partial sums rounded to one bf16 (q4nx's way): cosine 0.9998 against 0.999992 with two.
- 4-bit MLP for Qwen3.5-0.8B/2B: GPL collapses (above); q6 instead.
- `QGEMVGlu` GELU for Gemma 4 by default: 9.3 → 9.9 tok/s but KL 0.1539 / 0.0684 / 0.600 → 0.1580 / 0.0736 / 0.625 and cosine to before 0.9897. Suspected: the prompt chunk reads the GLU table through the MEDIUM GEMM; not confirmed.
- `SandwichRows` (the prompt's RMSNorm + add + RMSNorm as one operator): 632 → 538 reconfigurations but the same time (RMSNorm and add configs are about 5 KB). Code not kept.
- GDN kernel at `-Os` / `-Oz` / `-O1`: program size about the same (1045 → 1037 KB), `-Os` 32 bytes over the limit. Note IRON's build cache keys on name and parameters, so flags alone do not rebuild.
- A separate text-only export beside the multimodal one: towers load on use.
- `gemma4full` / `gemma4mmfull` (IRON's own forward) and `--scan-window`: kept only as references.

## Open

- Why the second prompt chunk hangs in a full ELF without `chunkBarrier` (aiecc warns about about 500 buffer descriptors reused after a `dma_free_task` in the decode-shaped design).
- Gemma 4 with towers: answers to one question varied between runs of the same export; not investigated.
- The host writes every layer's PLE for every chunk row, but the shared layers read only the last row (20 layers × 512 bytes a row wasted).
- Short-prompt GEMM and dequantize (60 % of a 16-token prompt): needs a GEMM that follows the call's rows, or a few-row GEMM reading quantized weights. `QGEMVRows` at 16 rows would match today's GEMM.
- Decode: beyond the current fusions, a resident one-array form like gpt-oss's `GptOssStep` ([MoE](moe.md)); first hurdle: core programs within 16 KB. GDN on 1 core a column (about half the config, about 12 ms a token estimated) needs the state layout shared by decode, prompt and hwnpu's `conv_pairs` changed.
- Gemma 4: PLE gate/projection to q8 QGEMV; GELU and product fused without the KL loss.
- Rebuild with the decode fusions and measure: Gemma 4 E4B, `gemma4mm`, LFM3-Audio, LFM3-Omni.
- Verify: q4 `QGEMVRows` (Gemma 4, PLaMo 3, LFM3), Gemma 4's shared layers and window attention, ring caches, MTP drafts, attention as one multi-row MHA, fewer state copies.
- PLaMo 3 2B decode was measured under another session's CPU load; measure again on a quiet machine.
- `iron_q4.py check --valid` disagrees for q8 too (the checker's problem, not fixed).
