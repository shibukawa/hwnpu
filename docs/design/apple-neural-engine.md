# Apple Neural Engine

[日本語](apple-neural-engine.ja.md)

Facts about Apple's Neural Engine (ANE) that hold whichever backend drives it: limits, numeric
pitfalls, prompt programs, the lock between processes and the ANE's memory. The backends are
[Core AI](apple-coreai.md) (what the commands use on macOS 27) and [Core ML](apple-coreml.md)
(best effort: the library's default and the developer tools).

## Limits

| Limit | Value | What hwnpu does |
|---|---|---|
| Weights per program | Core ML: 0.99 GiB runs on the ANE, 1.02 GiB falls to the CPU (M1–M3; M4 2 GB). Core AI's own limit is in [Core AI](apple-coreai.md) | Split by layers ([Core ML](apple-coreml.md)), or at the attention ([LLM engine](llm-engine.md)) |
| Any tensor dimension | 16384 fits; 16385 puts the ops on the CPU (M3), even an identity naming the output | `llm.NeuralEngineMaxDim`; KV caches in chunks of 16384 (`CacheChunks`); wide outputs, inputs and concats split |
| Weight tables (lut) | One table per 16 rows only | Pad rows to a multiple of 16 and slice back (`blocks.LinearPadded`); see [Quantization](quantization.md) |
| `gather` | Runs on the CPU | One-hot matmul; embedding rows looked up on the host |
| ANE compile cache (system, `aned`) | LRU of about 4 GiB by bytes (3.2–4.8 GiB; about 3.5 in effect); not visible without sudo. A miss reruns `ANECompilerService` at load: tens of seconds to minutes a program | `-prefill auto` keeps one program when two would pass `compileCacheBytes` (3.5 GiB); keep what runs every step smallest. A start is good when the second **and third** starts take a few seconds |
| Parts run in turn (M1 only) | Up to 2.86 GiB held: full speed; from 2.97 GiB every call is about 12× slower (828 vs 72 ms a token). M2 runs 3.65 GiB, M3 3.96, M4 3.62 at full speed | `neuralEngineHold` = 2.8 GiB on an `Apple M1`: `-prefill auto` keeps one program if two plus the programs beside them (MTP, towers) pass it; warns when one alone does |
| Programs per process | 112 (M2, Core AI): the 113th load fails (`no ANE resources`) and Core AI silently runs it on the GPU | `coreai` counts its live ANE programs and warns near the limit. gpt-oss-20b uses about 84 |
| Call floor | About 160–175 µs a call (M2, Core AI). No switch cost seen across 112 programs | Few, large calls |
| Elementwise ops | 16–20 GB/s (sum of 3 tensors), no cliff past 2 MB (M2) | |
| 1×1 conv | Time flat up to 256 output channels, then grows (1024: 500 µs, 4.3 TFLOP/s) | No reason to split wide convs |
| Run-time weights (input arrays) | Half the speed of constant weights (26 vs 50–52 GB/s, M2) | See [MoE](moe.md) |
| Attention scores | Past about 64–68 MiB a score tensor, q kᵀ and softmax run 2–3× slower per score (Core ML) | Chunk queries (`mil.DitQChunk`), but measure first: Whisper's 90 MB was not faster chunked |

### Weight format speed (M2, Core AI, constant 4096×4096, lut tables per 16 rows)

| Format | 1 row | 64 rows | Weight elements/s (1 row) |
|---|---|---|---|
| fp16 | 742 µs | 808 µs | 22.6 G (45 GB/s) |
| int8 (per row) | 466 | 506 | 36.0 G |
| lut8 | 449 | 515 | 37.4 G |
| lut6 | 387 | 477 | 43.4 G |
| int4 (per row) | 329 | 379 | 51.0 G |
| lut4 | 351 | 382 | 47.9 G |

Minus the 160 µs floor: fp16 about 58 GB/s, int8 and lut8 about 55 GB/s (bytes decide), lut4 tops
out near 90 G elements/s. lut8 takes 0.6× the time of fp16; lut4 is 1.3–1.4× faster than lut8.

## Ops the ANE does not run, and their stand-ins

Seen through Core ML's compute plan (Core AI's own pitfalls are in [Core AI](apple-coreai.md)). The
stand-ins are in the graphs for both backends.

| Op or form | Past what | Stand-in |
|---|---|---|
| `sin`, `cos`, `floor`, `round`, `ceil`, `mod`, `sign`, `cast` | Always | Rounding as `(v + 1536) − 1536` with 1536 an input (a constant is folded away); series after a range reduction |
| `gather` | Always | One-hot matmul |
| `reduce_sum` of exponentials | Varies by compile | Sum with a matmul by ones |
| Attention copying K/V with `tile` | Always | `SDPAGrouped` (op order alone changes placement) |
| A 262144-vocabulary `reshape`; `transpose` in the last part | That op | Reorder on the host (PLaMo 3; Gemma 4 `-logits host`) |
| `conv_transpose` (M1) | Strides 5 and 7; 8, 4, 3 run but are off | Polyphase conv |
| Dilated conv, `max_pool` (M1) | Dilations 17–48; windows 9 and 13 | Shifted copies + 1×1 conv; chains of window 5 |
| Depthwise conv (M1) | Over 4096 channels | Chunks of ≤ 4096 |
| An 8×8 by 8×8 matmul (Qwen3.5 GDN chunk) | Always | 7 products of (8×8)·(8×128) |
| (3, 39) conv + ReLU (Basic Pitch) | Always | Not solved |

## Numeric pitfalls

The placement report does not show wrong values. Compare on the ANE against a CPU run of the same
program.

### Hardware rules (M2, Core AI, `aicheck rules`; Core AI's CPU has none of them)

| Rule | Detail | What hwnpu does |
|---|---|---|
| A linear's or conv's sum must stay under 2^15 | 32768 gives Inf (32256 right), **partial sums too** (+1/−1 halves of 1024 inputs: sum 0 but Inf). `reduce_sum` is right to 51200 | Range control (MaxScaled, residual scales). The per-row int8 overflow below is likely this rule |
| Slice of the last axis starting at 1..14 mod 32 | Values over 4094 become Inf. Starts 0 and 15–31 right; other axes unaffected | `aicheck lint`, `HWNPU_DEV=ailower.notes=1`. Of 27 exports only LFM2-8B-A1B's top-4 routing weights (≤ 1) hit it: harmless |
| softplus fused as log(1 + exp(x)) | 0 for x > 10.4 | `relu(x) + log(1 + exp(−|x|))` |
| Tiny products inside a matmul | Dropped below about 2^-16 (fitted on attention data; products of 1e-6 summed over 512 give 0). Elementwise mul keeps them. In attention's `matmul(e, v)` with small values it erased the many lightly weighted keys (o off by up to 79 %, see [LLM engine](llm-engine.md)) | Scale up, divide later: the split form's attention multiplies e by 1024/l before the product |
| matmul with inner dimension 1 | (1,8,M,1)·(1,8,1,128): M = 1 rel. error 0.28; M = 64 right | Written as a mul (outer product) |
| Activations as 33-point piecewise lines | sigmoid, silu, tanh, gelu max rel. error 0.0009–0.0016 | See below |
| Division by a constant | Fine as a product | Lowered as `x × (1/c)` |

### Activations

- The ANE's `silu` is 1.4 % off, worse on small tail values; `x · sigmoid(x)` gives the same values.
  Use `blocks.SiLU` (`x / (1 + exp(−x))`) and `builder.softplus` everywhere. KL against PyTorch fp32
  (English / Japanese, M3): LFM3-1.7B lut8 0.058 / 0.012 → **0.022 / 0.0048**; PLaMo 3 Translate 2B
  lut8 0.0067 / 0.0069 → **0.0053 / 0.0042**; Qwen3.5 (silu + softplus) 0.33–0.76 → 0.005–0.007. Cost:
  a 3790-token prompt +9 %, a generation call +3 %.
- `sigmoid` through a deep CNN drifts (RTMDet-s, 80 SiLUs: box 98 px off vs 7.5 on the CPU). Write it
  with exp (`onnx.Program.ExpSigmoid`).
- The tanh `gelu` is fine (Gemma 4: no change when rewritten).

### Values

- **Per-row int8 overflows on large outputs**: the ANE multiplies the int8 codes and applies the row
  scale after, so an output of 13000 (Qwen3-VL-8B at the attention sink) became NaN. lut8
  dequantizes to fp16 first and is right.
- Values near fp16's limit (SANA 1.5 `<bos>` row 38 000; SpectroStream up to 1.2e7) give NaN on the ANE
  only. When a norm follows, scale the linear by 1/64; otherwise power-of-2 scales.
- A constant scale next to a linear is folded into its weight, which can push products under fp16's
  normal range (T5's outlier rows: cosine 0.36). Multiply the projection's output instead.
- Fill padding rows with real tokens (an eps of 0 in fp16 gave NaN).
- A one-row computation that every later layer reads (a timestep embedding) drifted up to 1 %;
  compute it on the host in float32.

### How to check

1. Compare the same program on the ANE and on the CPU (`HWNPU_COREAI_UNITS=cpu`,
   `HWNPU_COREML_UNITS=cpu`), per position (KL of logits, or cosine), counting padding apart.
2. Compare with the reference run in fp16 before hunting a cause. If HF's fp16 drifts as much, the
   port is not wrong (Whisper's encoder: an attention sink at silent positions, not an ANE error).
3. Convert the first N layers (`hwnpu-dev convert -layers N`) and find where the drift starts.
4. Bisect a graph op by op between the ANE and the CPU interpreter (`TestBisect` in
   `internal/vision/detect`), or move one part at a time to the CPU.

`hwnpu-dev check` on the CPU does not show any of these.

## Prompt programs

Reading weights dominates a call: 64 rows cost 1.06× of 8. So the prompt program reads more rows
per call and computes logits for the last row only.

- Paged models read prompts at `pagedPrefillSeq` = 128 tokens a call (`…-q128-last`). Unsplit
  models keep 64 so the key axis stays ≤ 16384. Intel keeps 64 without `LastRow`.
- `llm.Options.LastRow` (LFM3, Gemma 4): the last row is taken by a product with a one-hot input
  `last` (gather is on the CPU), so y is (1, 1, vocab).
- Gemma 4 with KV sharing: from the first shared layer on, all layers run on one row (`attendOne`).
- Numbers on Core AI are in [LLM engine](llm-engine.md).

## Lock between processes

Two processes using the ANE at once froze an M1 (predictions never returned, unkillable processes,
back after about 20 minutes). So:

- `llm.OpenDevice` takes an exclusive flock on `hwnpu/neural-engine.lock` in the user cache for a
  Core AI or Core ML device and holds it until the process ends. The second process prints
  `waiting for the Neural Engine: process N (NAME) is using it …`. Tools that open a provider directly
  take `llm.LockNPU`.
- After taking the lock, `settle` waits until the previous process's programs are gone from the ANE
  (`mem.NPUWired`, polled every 0.1 s, up to 30 s). The holder writes the non-hwnpu share it found
  (`base N`) as the file's second line.
- Why: a process's programs stay on the ANE after it ends (up to 7 s on a busy machine); a process
  that loads meanwhile may fail (`Request cancelled`). Another process compiling can also make a
  working program fail.
- The lock cannot be skipped (`HWNPU_NPU_LOCK` was removed on 2026-11-22). Other sessions may
  hold it for hours: queue ANE work.

## Neural Engine memory

- Weights on the ANE appear in neither the process's RSS, its footprint nor `aned`. They are kernel
  wired memory under `com.apple.driver.AppleH11ANEInterface`. `mem.NPUWired` reads it with
  `mach_memory_info` through purego (no root), finding the zone by name. Idle: 144–192 KiB.
- It is the whole machine's value: measure with the lock held.
- The same weights in two programs (Q=8 and Q=128) are held twice: LFM3-1.7B lut6, 2.43 GiB
  (1.21 GiB × 2).
- A program compiled just now is not on the ANE right after loading; it comes on at its first call.
  `hwnpu-chat` logs "the Neural Engine holds N GiB", adding "so far" when it is under half the
  weights.

## Not adopted

- A 2.8 GiB budget on every chip: M2+ would lose prompt programs. Only the M1 loads fewer.
- `x · sigmoid(x)` for silu: same values as the ANE's `silu`.
- The exp form of `gelu` (Gemma 4): no change. The exp form of the VAE's silu in Anima: no change in
  images, VAE 15–20 % slower.
- Prompt calls of 256 tokens: no faster than 128, heavier to load.
- Splitting attention queries by default: no gain on Whisper's 90 MB scores.

## Open

- Why two processes at once freeze the ANE. Intel and AMD were not tried; the lock is Apple only.
- Size and location of the system ANE compile cache (needs sudo); `hwnpu-dev probe cache` results on
  M1–M4 not recorded.
- Which shapes at the 16384 limit fall to the CPU and which stay.
- The ANE's memory beyond weights (KV, intermediates) is not told apart.
- M1 Pro / Max / Ultra hold limit (taken as the M1's).
- `silu` rewrite not measured on LFM3-VL / Audio / Omni / d1, their towers and TTS.
