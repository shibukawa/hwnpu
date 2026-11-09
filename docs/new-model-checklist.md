# Checks when bringing in a new model

[日本語](new-model-checklist.ja.md)

Written 2026-11-07, updated 2026-11-08 (what PLaMo 3 Translate 0.8B, Qwen3.5-0.8B / 2B, Whisper large-v3-turbo, Qwen3.6-35B-A3B and checking drafts on AMD added). While bringing in LFM3 (VL / Audio / Omni, d1), Gemma 4 E2B / E4B, Qwen3.5-4B, PLaMo 3 Translate, PLaMo-Embedding and Anima, some problems were found only late, and some methods paid off. This page lists them **in the order to check them first**. Checking in this order from the start gives a form with little memory and good speed, without redoing the work.

## Policy: power over speed

hwnpu's aim is to run on the NPU alone and use less power. Weigh methods in this order:

- **When the total time is the same, keep the work on the NPU.** A form that gets a little faster by moving work to the CPU or the GPU (attention over a short context on the CPU, say) is not taken. The GPU is not used
- **Read fewer bytes of weights.** Reading the weights is most of a token's energy, and reading less helps speed and power at once (fewer bits, no layers a prompt program does not need, MoE)
- **Cut the waste on the NPU.** Programs that do not fit the NPU's memory and are loaded again, or one operation on the CPU that makes the NPU and the CPU take turns, use power even at the same speed
- Measure power with `powerbench` as mJ per token (the Mac needs sudo). When it cannot be measured, compare the weight bytes read, the memory the NPU holds (`mem.NPUWired`) and the CPU use instead

The numbers come from chapter 16 of [design.md](design.md) (section numbers in parentheses), [improvements.md](improvements.md), [model-fit.md](model-fit.md) and [amd-npu-models.md](amd-npu-models.md). The steps and pass conditions for moving a structure to Go are in `.claude/skills/port-architecture` and are not repeated here.

| Stage | What it decides | Where |
|---|---|---|
| 1 | Whether it fits and is fast, on paper (whether to choose it) | Anywhere |
| 2 | Whether the structure is right (fp64, against HF, tokenizer, chat format) | CPU |
| 3 | Whether it fits on the Mac's Neural Engine, and all of it runs on the NPU | Mac |
| 4 | The AMD / Intel form | Linux / Windows |
| 5 | **The quantization format and the calibration method** | Estimated on the CPU, checked on the NPU |
| 6 | Memory (resident, pinned, disk) | Each NPU |
| 7 | Speed (generation, prompts, startup) | Each NPU |
| 8 | Long contexts | Mac (Intel) |
| 9 | Checks in use, and the record | Each NPU |
| 10 | Values for combining it as a part (residency, input and output kinds) | Each NPU |

Each "pass" is a guide. When a model does not meet it, write the reason in design.md before going on.

## 1. Estimates on paper (before porting)

Estimate from config.json alone, with the formula and criteria in sections 2 and 4 of [model-fit.md](model-fit.md).

- **Time per token (ANE)** ≈ weight bytes ÷ 65 GB/s + layers that compute attention over the long capacity × 0.22 ms (the `SDPAGrouped` form) × capacity / 1K + 3 ms
  - Sharing KV does not reduce how many attentions are computed (only memory). Window, short-convolution, Mamba2 and GDN layers do not depend on the capacity
- **Time per token (AMD)**: generation is bound by the count of operators, not by the weight bytes (Qwen3.5-0.8B 82–86 ms at bf16, q4 and q6 alike; PLaMo 3 2B q4 85 ms; Qwen3.6-35B-A3B's 40 layers 172 ms where reading the weights alone would take about 33 ms). Estimate from the layer count and the times of models already in (16.104, 16.107, 16.92)
- **Weights**: lut4 0.5 B, lut6 0.75 B, lut8 1 B a weight. On an M3 (16 GB), keep resident memory under 3 GiB. On AMD, 4 bits (q4nx at 5 bits + attention at q8) pins 1/2–1/5 of bf16. On an M1 (16 GB), keep the total weights the Neural Engine holds for the parts run in turn under 2.9 GiB: past it every call is about 12× slower (Qwen3.5-4B lut6 at 2.97 GiB 1.1 tok/s, lut4 at 1.99 GiB 10.9; the M2 and the M4 ran 3.62 GiB at full speed, 16.109, PROBE.md)
- **Vocabulary**: the output layer is vocabulary × hidden. The linear takes a 262144 axis on the ANE, but elementwise operations after it fall to the CPU. Leave a monotonic one (the soft cap) to the host (Gemma 4, 16.113)
- **Structures that cannot be used**: MoE routing (`gather`), learned sparse attention, variable shapes. MoE runs on AMD (Qwen3.6-35B-A3B: the routing on the device while generating, on the host for prompts, 16.89–16.93, 16.110)
- **Layers the checkpoint carries for drafting** (Qwen3.5's `mtp.`): do not drop them at conversion. As drafts they beat prompt lookup (Qwen3.5-0.8B on the M1: 26.1 → 43.8 tok/s on code, 16.109)
- **License and languages** (whether it goes into hwnpu-chat's catalog)

**Pass**: an estimate of 50 ms or less at 1K and 100 ms or less at 16K, and a format in which resident memory fits in 3 GiB on an M3.

## 2. A correct structure (CPU)

On top of the pass conditions of the `port-architecture` skill (1e-4 against an fp64 reference, one layer at a time matching all at once, breaking it to see the test fail, layer-by-layer checks with `hwnpu-check`), these were found later:

| What to check | Tools | Lesson |
|---|---|---|
| The downloaded checkpoint is intact | `sha256sum` against the `x-linked-etag` of HF's file | Big downloads here had sparse bit flips (LFM3-Audio and Omni: about 10 places of 2–6 bytes each). Neither random-weight checks nor value ranges show them. Fetch the bad pieces again (16.67) |
| Compare **with a BOS** | `tools/ref/ref_logits.py`, `hwnpu-check`, `ironlm -forced -bos` | Without a BOS, LFM3's KL looks 4× worse and layer 0's activations become outliers (amd-npu-models.md section 13). Measure with the tokens the chat format actually puts first |
| Check against the real weights | `hwnpu-check`, `iron_plamo3_check.py`; on AMD `ironlm -forced` + `tools/ref/forced_logits.py` | With random weights the NPU and the CPU reference disagree (Gemma 4, PLaMo 3): their logits are flat, and bf16 rounding and the bfp16 GEMMs spread layer by layer (16.107). Decide pass or fail with the real weights |
| Check the real weights **of every size** | `hwnpu-check` | The same structure takes other paths through its config at other sizes (Qwen3.5-0.8B has as many value heads as key heads). Qwen3.5-4B was not checked against its real weights until 16.108. Check every size that goes into the catalog |
| The tokenizer matches HF | `tools/ref/tokenizer_golden.py` | 0 mismatches over fixed texts plus thousands of random ones. Also check that the tests fail when ties (PLaMo), NFC and combining marks (Qwen3.5) or byte fallback (Gemma 4) are handled the other way |
| The chat format matches jinja2 character for character | `tools/ref/template_golden.py` | Tool calls, thinking, dropping earlier thinking |
| The chat format's defaults do not differ by size | The template and the model card | Qwen3.5-0.8B / 2B answer without thinking by default, 4B thinks. Made to think, the small ones ran on to the budget, so the catalog's `NoThink` follows the default (-think on the command line overrides it) |
| Multimodal preprocessing | `resize_golden.py`, `mel_golden.py`, `whisper_mel_golden.py` | ±1 against PIL, 1e-4 against librosa |
| Encoder-decoders (speech) | `tools/ref/ref_whisper.py`, `hwnpu-check` (the mel frames as input through `REF.x.bin`) | Check the encoder, the cross attention's keys and values and the decoder each on their own. For audio over 30 seconds, follow HF's windowing too (features over the whole clip, the next window from the timestamps): cut every 30 seconds, HF drops text as well |
| Embedding models | `embed_golden.py` | Cosine 0.997 or more against HF |
| Image generation | `tools/ref/ref_anima.py` | Latent cosine against diffusers' fp32, PSNR of the VAE |

## 3. Does it fit on the Mac's Neural Engine?

**Check this first**: nothing here can be settled without a Mac. Write a handoff (`docs/handoff-*.md`) and measure on a Mac.

### 3.1 Limits

| Limit | What to do | Section |
|---|---|---|
| 1 GiB a package (0.9 GiB a part by default) | Split by layers (separate packages; multifunction cannot get past it) | 16.12–13 |
| 16384 per dimension | Attention split into chunks (`-paged`), the softcap folded to (16, 16384) | 16.21, 16.25 |
| A compile cache of about 4 GiB (LRU), **about 3.5 GiB in effect** | If the two programs (Q=8 and Q=64) exceed it, use one (`-prefill auto`, at 3.5 GiB). Qwen3.5-2B's two programs of 3.56 GiB were compiled again at every start, 4–5 minutes. Tiers (`-tier`) only with lut4 | 16.41, 16.65, 16.71, 16.99 |
| Tables only per 16 rows | How to choose the format is in section 5. An output layer whose rows are not a multiple of 16 (Whisper's vocabulary of 51866) is padded with rows of zeros and sliced back (`blocks.LinearPadded`); unpadded, the lut conversion fails | 16.100 |
| `gather` runs on the CPU | Look up embedding rows on the host (PLE too) | 16.25 |
| Attention scores under 64 MiB a tensor | Past it, 2–3× slower. Chunk the queries (`mil.DitQChunk`). Not always: Whisper's encoder (20 heads × 1500 × 1500 = 90 MB) was no faster in chunks (0 / 300 / 500 / 750 queries: 520 / 518 / 537 / 555 ms, the same output). Measure before splitting | 16.85, 16.115 |

### 3.2 Does all of it run on the NPU?

Check that each part is `all on the NPU` with `hwnpuinfo -model PART.mlpackage`. What fell to the CPU before:

- Attention that copies K/V with `tile`, and the form with the queries as a matrix → `SDPAGrouped` (16.25, 16.39). The order alone changes the placement
- A `transpose` / `reshape` in the last part → take the output token-first (`attn_ot_N`, PLaMo 3). Elementwise operations after the output layer (Gemma 4's soft cap) → the host's sampling applies them to the top values only (`-logits host`, `llm.LogitSoftcap`, 16.113)
- Parts with one row, and parts as programs of their own → 8 rows, or put into the neighbouring group (Anima, 16.85)
- `reduce_sum` → sum with a matrix multiply (chunked attention: on the ANE 30 times of 30)

### 3.3 Does fp16 overflow?

Check that each layer's relative RMS error is under 0.01 with `hwnpu-check -f16`. Layers that pass one at a time can still drift as a whole (Whisper: 0.0003 or less a layer, a cosine of 0.9978 for the whole). Look also at:

- The whole `-f16` run and the cosine **per position**, counting the positions of the speech or prompt apart from the padding's
- Where massive activations (attention sinks) arise: the largest values of each layer of HF's hidden states (`REF.hidden.bin`). Whisper puts 300 at silent positions from layer 21
- **Before hunting the cause, compare with HF run in fp16.** If it drifts as much, the port is not wrong (Whisper: HF's fp16 0.9978 too)
- On the ANE and on Core ML's CPU, programs of the first N layers (`hwnpu-convert -layers N`) against HF's layers show where the drift starts (`TestEncoderOnDevice` in `internal/transcribe`). Whisper stays at 0.9996 or more through layer 20; at layer 21 which silent position becomes a sink varies. The CPU and the ANE drift as much (cosine 0.97) from HF as from each other, the speech positions hold, and the transcripts are HF's. A LayerNorm divided by each row's maximum changed nothing, so it is not an overflow

What helped:

- Scale the norms' inputs by 1/64 (Gemma 4: KL 3.17 → 0.0004; PLaMo 3 too). Not small inputs, though (the embeddings entering layer 0): their squares fall under fp16's range and give NaN (Qwen3-0.6B's embeddings reach about 0.1, 16.127). Always look at layer 0 with `-f16`
- Shrink the last layer with head-scale (PLaMo-Embedding), `-post-scale` (LFM3, 1.4–2.7 % a layer)
- Keep residuals at 1/32, and divide by each row's maximum before a norm (Anima; a fixed scale made subnormals)
- Fill padding rows with the last token (Gemma 4's eps becomes 0 in fp16 and gives NaN)
- The ANE's own `softplus` and `silu` are inaccurate (softplus returned 0 for some inputs; silu off by 1.4 % in its tail). Use `builder.softplus` and `builder.silu` (written with exp and log). Products below fp16's smallest normal (6e-5) lose digits on the ANE: scale them up and divide later (Qwen3.5's GDN output, 16.88). `hwnpu-check` on the CPU does not show these: compare the per-position KL on the ANE with Core ML's CPU (`HWNPU_COREML_UNITS=cpu`)

### 3.4 Pass

- Every part is `all on the NPU`
- The second and **third** starts take a few seconds (the parts stay in the compile cache). In the split form (`-paged`), Gemma 4 went from 98 s to 1.9 s. The second alone misses two programs that evict each other (16.99)
- Record the first compile time of each part. Qwen3.5's 128 tokens a call (GDN) took 13–15 minutes the first time (not solved)
- On an M1, the parts' times in `HWNPU_PROFILE=1` hold steady from call to call. When the parts run in turn hold more than about 2.9 GiB, the same part varies and every call is 10× slower or more (a reload). On an M1, `-prefill auto` keeps one program when the two with the MTP layer's and the encoders' programs would hold over 2.8 GiB (16.118); a model whose program that generates holds over it alone (Qwen3.5-4B lut8, Gemma 4 E4B) gets a warning in the log (16.109, 16.112)
- **A change to the graph that keeps the program's name raises `llm.Arch.Revision`** (the name takes `-rN`). Otherwise hwnpu-chat reuses programs converted before the change (the silu rewrite, lfm3 / plamo3 r1; the host soft cap, gemma4 r1; Anima's `ExportVersion`, 16.95, 16.113)
- Record the KL and argmax of `hwnpu-check -package` (for each format, section 5)

## 4. The AMD (IRON) and Intel forms

### 4.1 AMD

| What to check | Lesson | Section |
|---|---|---|
| Does IRON have the same application? If not, write it from IRON's building blocks | LFM3 (`iron_lfm3.py`) and PLaMo 3 (`iron_plamo3.py`) were written from the building blocks | 16.66, 16.70 |
| Check against HF (400 rows of `ironlm -forced` against HF fp32 with `tools/ref/forced_logits.py`) | A bf16 export lands at KL 0.001–0.014, top-1 96–99 % (E2B, E4B, PLaMo 3, Qwen3.5). Long prompts (over 2000 tokens) too | 16.76, 16.102 |
| **The prompt's MHA on several pipelines** (`--mha-pipelines`, default 8) | Without a profile in the export, MHA ran on one pipeline, 60 % of a chunk. On 8, chunks are 2–5× faster with the same logits. A chunk of 256 takes 4 at most. Check `prompt attention: N pipelines` in the export's log and `num_pipelines` in the kernels' names | 16.107 |
| The chunk's size | 1024 for attention models (LFM3 too: short prompts −24 % against 2048, long ones the same). 512 for GDN models (Qwen3.5: 256 slows long prompts, 1024 short ones). Not under 256: the prompt's `ProjectionGEMM` takes multiples of 256 rows only (16.117) | 16.54, 16.108 |
| Quantized projections in multiples of 64 out by 512 in | Pad a fused projection (GDN's in_proj, 6176 rows) per column with rows of zeros (`iron_gdn.column_widths`) | 16.104 |
| **The GEMV's 64-row blocks divide evenly over the 8 columns** | The GEMV takes only as many columns as divide its blocks. Qwen3.5-0.8B's GDN projection (776 × 8 = 97 blocks, a prime) ran on one column: 571 µs against 217 on 8. Look at `num_aie_columns` in the kernels' names (`--command-per-step`), and pad (`iron_gdn.COLUMN_ALIGN` 64) | 16.117 |
| A command's control code within the device heap (64 MiB) | Dense models cut at `SCRATCHPAD_WORDS` 24 (Qwen3.5-2B's q4 at 32: `CREATE_BO: invalid argument`). Gemma 4's tail export keeps 32: at 24 IRON split its shared layers over two commands and the next chunk's first command hung (16.80) | 16.104 |
| Where a slow chunk's time goes | `--command-per-step` (a command a step) and `HWNPU_IRON_TIMING`. Run one chunk only (16.80's hang). Read the times only: on Qwen3.6 that export's logits do not match HF. Whether transfers or compute bound an operator: compare with a build whose kernel computes nothing (Qwen3.6's experts: 8.4 ms streaming the weights alone, 20 ms with the compute) | 16.107, 16.116 |
| **Check prompts against HF at the chunk's own size** (64, 128, the chunk's size: `iron_qwen35_check.py ref --tokens N --stream` into `EXPORT/check`, then `ironlm`) | A 32-token check takes only the one-pass path. Qwen3.6's MoE broke when a 256-token chunk ran in several passes (cosine 0.578) | 16.110 |
| What a host step reads can be overwritten between commands | IRON's arena does not count a host read as a use. A command run more than once needs its inputs written back from the host (`moeRoute.hn`) | 16.110 |
| Hoist work an operator repeats | The GDN operator re-reads each row once a slice of value columns and redid the SiLU each time (3/4 of the operator); once before it: 30 → 10 ms. The MoE re-reads every expert's weights each pass, so the rows an expert to the chunk's size (four cores a column split the outputs when the accumulators do not fit) | 16.111, 16.116 |
| Host steps split over the CPUs by token | Qwen3.6's combine, 40 layers 336 → 73 ms. Keep each token's order of sums | 16.116 |
| Kernel pitfalls | `filter_even` / `filter_odd` on long vectors (128 or 64 bf16) take about 290 K cycles a call (use `interleave_unzip` on 32-element vectors). Program memory is 16 KB a core (more template copies overflow it; one noinline function). bf16 mmul is native in 8×8×4 only | 16.116 |
| One full ELF | Switching xclbins costs 0.46 ms each | 16.45–50 |
| Prompt chunks of 1024 | 20 % faster than 2048. At 256 the window layers become rings and cannot roll back | 16.54 |
| At most 3 transfers queued on one shim channel | Queuing 16 without waiting hung the command, and two hangs in a row needed a reboot. Wait for the oldest before queuing more | 16.89 (PR #66) |
| Each command under 2 s | Longer ones end in `ERT_CMD_STATE_TIMEOUT`. Split them at export (`SCRATCHPAD_WORDS`) | 16.65 |
| An empty command in a second hardware context before each decode step of several commands (`stepBarrier`) | Without it, Qwen3.6's decode with the convolution folded into the GDN operator hung in the next step's first command. An empty command in the same context does not help. About 0.3 ms a step | 16.92 |
| Window layers read only their window (`iron_window.py`) | Prompts −12 %. Without an empty command before each chunk (`chunkBarrier`) the second chunk hangs. The caches' rows are written as `cache_rows` | 16.84, 16.87 |
| KV-shared layers on the prompt's last row only | Gemma 4 E2B's prompts 2.3× faster | 16.80 |
| Tables on the host (PLE, embeddings) | E2B pinned 9.08 → 4.60 GB, the same output. With a head tied to the embeddings, put them on the host too once the head holds a quantized copy (Qwen3.5 −0.47 to −0.95 GiB) | 16.72–73, 16.104 |
| Towers loaded only when used (`growBanks`) | No text-only export needed | 16.81 |
| `-idle` gives back pinned memory | The next request pays 0.8–2 s | 16.74 |
| The cores' rounding to nearest even | Floor rounding pulls sums of inputs with a mean downward (Anima's DiT) | 16.78 |
| MHA's head_dim a multiple of 64 | Pad 96 to 128 (LFM3-VL's tower) | 16.65 |
| Images and audio in the final export form | Towers loaded on first use replace the banks; the chunk barrier bound to the old bank failed (ENOENT) until 2026-11-07. Send an image and a clip through `hwnpu-chat` after any change to the export | 16.81, 16.87 |
| Layers with state (convolutions) copied at chunk boundaries | For rolling back to marks and for the prompt store | 16.66 |
| **Checking drafts in one call** (`--verify 3`) | A prompt chunk checking 2–8 tokens costs 3–9 decode steps, so the export adds a version running the decode step's operators over 3 rows: every op needs a form of a few rows (the GEMV `QGEMVRows`, q8 and q6 only so far; a recurrent operator over the rows in one call keeping the state after each, as `iron_gdn`'s `steps`; attention a row at a time). Layers that run the last row only (Gemma 4's shared ones) and ring caches need all rows. Pass: `ironlm -verify` (every row and the step after `Keep` bit for bit the decode step's). Qwen3.5-0.8B: 1.27 decode steps, 82 → 62–72 ms/token | 16.117 |
| The verify version's control code beside the decode step's | The verify ELF (30 MB) and the decode ELF (26 MB) both stay in the device heap (64 MiB). When they do not fit, one is made at each call (`perCall`, 2 ms); a prompt chunk or a tower that does not fit drops both (`dropDecode`). Calling the GDN operator three times made it 47 MB, over the heap | 16.117 |
| A graph's groups (`variants`, an image model's DiT) in one bank, read in turn (`iron.GraphOptions.SwapVariants`) | Anima's DiT, 7 groups: pinned 3445 → 1236 MiB, +2 % | 16.106 |
| Big exports | Exporting Qwen3.6's 40 layers took 30 minutes at a peak RSS of 53.8 GB (of 60): do not run it beside other big jobs. /tmp is RAM (tmpfs): write test exports under `~/models`. Run long ones with `systemd-run --user` (so they are not killed with the session) | 16.91–16.92 |
| The export's name | `~/models/NAME.amd-npu2-q4` (or `NAME.amd-npu2`): on a machine with only AMD's NPU, hwnpu-chat and hwnpu-agent pick it for `-use NAME` without `-iron` (#81) | — |

### 4.2 Intel

- To IR with `ovlower`. int8 by default (with int4 the next token matches fp16 only 22/52). For now only accuracy can be checked, on OpenVINO's CPU (speed and memory on a real NPU are not measured)

## 5. Quantization: choosing the format and the calibration method

**The stage that takes the most work and pays the most.** How much 4 bits lose differs tenfold between models (AMD q4 KL: PLaMo 3 on its translations 0.009, LFM3 0.036, E4B 0.049, E2B 0.097). Decide in this order.

### 5.1 Measure the same way

- **KL(reference ‖ quantized)**: mean, median, 95th percentile and top-1 agreement over 400 positions (the first 256 tokens read as a prompt, then one at a time), **with a BOS**
- Five texts: the README (Japanese and English mixed), GPL-3 (**not used for calibration**, to catch overfitting), the evaluation part of wiki40b-ja, the test part of wikitext-2, and Go code. They live in `~/models/calib`
- The reference is fp32 on the CPU (HF) or bf16 on the same NPU. Check that the quantized KL on the NPU matches the CPU estimate (LFM3 0.036 against 0.035; LFM3-VL the same way). If it does not, suspect the NPU's arithmetic (rounding, bfp16)
- On Apple's Neural Engine: `hwnpu-generate -package P.parts.json -score REF,...` runs a program without a cache (`hwnpu-convert -seq 128`) over each reference's tokens (`ref_logits.py` / `ref_gemma4.py`) and prints the KL at every position (mean, median, largest), top-1 and cosine. The first tokens of a text weigh much (KL up to 8–13), so compare settings on the same texts only (16.114)
- For image generation, instead of KL: the velocity's relative RMS error along the reference trajectory, and the cosine of the final latents of a free run (`tools/amd/anima_gptq.py`)

### 5.2 Sweep the bits per part on the CPU

Switch each part (attention projections, MLP, output matrix, convolutions, towers) between 4 bits, 8 bits and bf16, and measure KL on the CPU (combine `--q8`, `--head` and `--fmt-*` of `q4_gptq.py --method rtn`). The conclusions so far:

- **Attention projections at 8 bits**: a little over 10 % of the weights, yet at 4 bits they cause most of the error (E2B 0.097 → 0.229). Only +0.12 GB pinned
- MLP and output matrix at 4 bits. The output matrix at 4 bits barely matters
- Short convolutions at 8 bits (+0.01 at 4 bits)
- Layers with sandwich norms (Gemma 4, PLaMo 3) are weak at 4 bits. With the ANE's tables, lut6 or more (Gemma 4 lut4 KL 2.54, PLaMo 3 1.91)
- lut8 for multimodal models' text parts, TTS, d1's judgments and embeddings (they broke down with lut4)
- A distilled 8-step DiT is weak at 4 bits (q4nx: latent cosine 0.777). q8 is the default (q6 keeps the composition)
- **A small model's (2B or less) MLP can break at 4 bits**: Qwen3.5-0.8B README 0.055 but GPL 0.34 (0.30 with GPTQ). AMD's q6 (6.5 bits, `--mlp q6 --head q6`) gives 0.044. Always look at the GPL. 4B makes 0.042 at q4 and stays at q4

### 5.3 Try the calibration methods in order

Keep the format and raise accuracy by choosing the codes (levels) better. The format does not change, so speed and memory stay the same.

| Order | Method | Tools | Example results | Notes |
|---|---|---|---|---|
| 1 | Rounding (RTN). q4nx picks the least error of 9 candidates | `iron_q4.quantize_fmt`, `palette.QuantizeBits` | Weight error 8.2 → 7.3 % | The baseline. The least weight error is not the least output error |
| 2 | **GPTQ** (pushes each column's error onto the later columns, with H = E[xxᵀ]) | `tools/amd/q4_gptq.py` → AMD: `iron_export_lm.py --q4-codes`; Apple: `hwnpu-convert -codes`, `hwnpu-chat -codes` | NPU wiki-ja: E2B 0.221 → 0.081, E4B 0.083 → 0.049, LFM3 0.071 → 0.038, LFM3-VL 0.067 → 0.034, PLaMo 3 0.058 → 0.040. Anima's DiT q4: latent cosine 0.868 → 0.939 (q8 0.962) | Damp by **0.1** × the mean of H's diagonal (0.01 overfitted the calibration texts and made the GPL worse than rounding). Calibrate on general text (wikitext-2 and wiki40b-ja, 64 × 512 tokens); the repository's documents alone overfit too. Leave the BOS position out of H. Quantize the earlier layers before measuring the next ones (a few layers a pass) |
| 3 | **Fitting the tables** (Apple's lutN): k-means weighted by E[x²] | `q4_gptq.py --lut-fit wkmeans` | LFM3 lut6 0.050 → 0.038, lut4 0.66 → 0.33 (through Go's converter) | Start from the uniform tables (starting from the k-means of least weight error gives the worst KL). The ANE check is in handoff-m3-gptq.md |
| 4 | AWQ-style scales folded into the previous op | Not written | | model-fit.md T8, 2. When lut4 has to go lower |
| — | One format up (AMD's q6) | `iron_export_lm.py --mlp q6`, `q4_gptq.py --fmt-q4 q6` | Qwen3.5-0.8B GPL 0.34 → 0.044 | When calibration does not fix it. The GEMV takes about q8's time; the gain is memory (76 % of q8) |

**Whether to adopt it**: the GPL, not used for calibration, must not get worse than rounding. For LFM3-Audio GPTQ leaves only the GPL unchanged (0.154 → 0.161), but the other four texts drop by half or more, so it was adopted. A multimodal model's text part is often trained again (LFM3-VL / Audio / Omni): calibrate each, do not reuse the base model's codes. If a text gets worse, raise the damping (0.3 was tried) or add calibration text.

**Cost of calibration**: fp32 on the CPU, 25–60 minutes for 2B models, 2.4 hours for E4B (over 30 GB in fp32; running two side by side ran out of memory). On Linux, run it with `systemd-run --user` (so it is not killed with the session). **A 16 GB Mac** calibrates Gemma 4 E2B with its embedding tables read from the file (`q4_gptq.py`'s `lean`, about 9 GB): 87 minutes on an M1 with 4 threads, slowed by swap in its second half. The calibration texts come from `tools/amd/calib_data.py`; PyTorch from Homebrew (16.114).

**Weights the calibration leaves out**: the converter rounds every weight in the format, the ones MODELS does not name too. Gemma 4's `per_layer` projections (41M) at lut6 cost a lot next to GPTQ: `-format-for per_layer=lut8` (+0.01 GiB) took the median KL on the ANE 0.13 → 0.08 (16.114). Look for small projections outside the q4 / q8 lists and keep them a step up.

**Calibrate every projection**: `q4_gptq.py` without `--q8 gptq` leaves the attention and convolution projections rounded (LFM3: codes for 73 of 143 weights; with it 143, and the English KL 0.255 → 0.247, 16.95). A model loaded lean for a 16 GB Mac must first match transformers' logits (KL 0.003 or less, the same top-1): Gemma 4's `layer_scalar` buffers were missed at first and broke the answers (16.114).

**Matching the codes**: codes are looked up by a hash of the checkpoint weight's shape and bf16 values (`iron_q4.weight_key` = `palette.WeightKey`). If the export reorders or concatenates weights, they do not match. Check the count in the export's log, "N weights took the chosen codes" (LFM3-VL: 73; the q8 parts are rounded by default and not included). A weight quantized as one, like PLaMo 3's `gate_up_proj`, matches.

### 5.4 Choosing the format

| Guide | Decision |
|---|---|
| KL under 0.05 (against HF on the CPU) | port-architecture's guide. 4 bits can be the default |
| KL 0.05–0.2 | Decide by the quality of the answers (section 9). The ANE's lut6 (LFM3 0.232) is here |
| KL 1 or more | Not usable (lut4 LFM3 got 3 of 4 factual questions wrong) |
| `hwnpu-formats`: the reference's next token 90% or more, and as many short questions right | Take the smaller one (Qwen3.5-2B's lut6: 92% of lut8's, 10 of 12 right in both. 16.115) |
| 85–90% likewise, or new mistakes in arithmetic or facts | Keep the next larger format (Qwen3.5-0.8B's lut6: 86% of fp16's, and 17 × 23 wrong) |
| Under 80% likewise, or answers that repeat a phrase | Not usable (Qwen3.5's lut4: 60–76%, 5–6 of 12 answers repeating) |
| Audio and image encoders | Decide by the error rates of the text (WER, CER). Whisper's lut6: WER 4.43% → 4.87%, CER 3.60% → 3.94%, so it stays lut8 |

The trade with speed follows the weight bytes (ANE lut4 / lut6 / lut8: 1.81 / 2.65 / 3.49 ms). lut6's slowness was won back with `SDPAGrouped` (16.41).

- Make fp16 the reference. When fp16 does not fit a part (0.97 GiB; Qwen3.5-2B's last layer with its output layer of a 248K vocabulary is 1.04 GiB), use lut8 (`hwnpu-formats -ref lut8`) and check lut8 itself by its KL against HF on the CPU
- On a short text, perplexity can be lower quantized (Qwen3.5's lut6). Compare by agreement with the reference
- npuprobe's read speed per format is not the model's: on the M1, npuprobe reads int8 at 62 GB/s, but Gemma 4 E2B's parts read it at about 48 GB/s, so int8 was slower than lut6 by its extra bytes (16.112). Measure the real parts with `HWNPU_PROFILE=1` before choosing a format for speed
- When a smaller format brings the two programs (8 tokens per call and the one for prompts) under the compile cache (about 3.5 GiB), `-prefill auto` goes from one program to two (Qwen3.5-2B at lut6). Measure the resident memory and the first compile again
- The format may differ by size: within one architecture, decide it per checkpoint with `Formatter` (the config's `Format()`; LFM3 and its VL, Qwen3.5's 2B against the 0.8B and 4B)

## 6. Memory

| What to measure | Tools | Lesson |
|---|---|---|
| AMD pinned memory (the rise of `Unevictable`) | `/proc/meminfo` (not in RSS), the banks in the export's `manifest.json` | Mostly weights and work areas. KV is under 5 % |
| The Mac's `phys_footprint` and the memory the Neural Engine holds | `npuprobe split`, `hwnpu-chat`'s loading log, `HWNPU_PROFILE=1` | The weights are in the kernel's wired memory, read with `mem.NPUWired` (16.95). Programs Core ML has just compiled come onto it at their first call, not at loading (LFM3: 0.27 GiB after loading, 2.43 GiB after one call), so measure after one call. When two programs do not fit there, a part of generation is loaded again on every call (Gemma 4 on the M1, 16.112) |
| Size of conversation marks | | Copy only the ring rows in use (`ringRows`, 578 → 183 MB) |
| Right after a conversion | | `debug.FreeOSMemory` (2.4 GB → 154 MB) |
| KV size and context | | Past 1/4 of physical memory, shrink by whole chunks |
| Copies of the states at chunk boundaries (AMD) | `manifest.json`'s banks | Layers with state (GDN) keep a frame for each chunk boundary, so pinned memory grows with the context (Qwen3.6 at 4096: 0.33 → 1.1 GB) and shrinks with bigger chunks (Qwen3.5-2B 256 → 512: 2014 → 1862 MiB). The verify version keeps every row's states too (Qwen3.5-0.8B 2 × 20 MB) |
| Disk | `cache.Clone` | Share identical files (the Q=8 and Q=64 `weight.bin`, the common start of exports). Nothing is shared on ext4 |

**Pass**: enough numbers to add a row to the table in [improvements.md](improvements.md) 5.1.

## 7. Speed

| What to measure | Tools |
|---|---|
| One call (Q=8, Q=64) | `TestCallTimes` (`HWNPU_CALL_PKGS`) |
| One encoder call (the ANE and Core ML's CPU) | `TestEncoderOnDevice` (`HWNPU_WHISPER_UNITS`) |
| Generation tokens/s, prompts, startup (the second) | `hwnpu-generate`, `hwnpu-chat`'s log |
| Comparing formats | `hwnpu-formats` |
| AMD GEMV, power | `tools/amd/gemvbench`, `powerbench` |

Things to bring in next (they helped every model):

- Two programs (generation and prompt reading) on one cache, reuse of the previous prompt, the prompt store (disk)
- **The prompt program's output layer on the last row only** (`LastRow`, #56). PLaMo 3 and Qwen3.5 lacked it; with it, reading a document took 25 % less (16.99)
- Speculative decoding (n-grams from the prompt, Apple / Intel; AMD with `--verify`, 4.1; drafts from the checkpoint's MTP layer with the GDN state brought back on the host, `-mtp` and `state-rows`, Apple, 16.109), the think budget (`-think-budget`), repetition detection
- A model whose later layers write no cache (shared keys and values) gets a prompt program that stops at the last layer that writes one (`-kv-only`, `llm.CacheWriters`; Gemma 4 E2B 1.6 → 0.38 GiB, 16.112). When the two programs do not fit the Neural Engine, generation slows down too
- A monotonic operation after the output layer (the soft cap) goes to the host's sampling, so that the output layer is all on the Neural Engine (`llm.LogitSoftcap`, 16.113)
- `HWNPU_PROFILE=1` breaks a step down (ms per part, per host attention and for reading the logits, with the memory the Neural Engine holds)

Compare only pairs measured back to back under the same conditions (on the M3 swap, on AMD the power state, make numbers vary by up to 1.4×). On AMD, check first that no other process holds `/dev/accel` (`ls -l /proc/*/fd 2>/dev/null | grep accel`; a `hwnpu-chat -listen` left running counts): another user of the NPU made commands up to 50× slower. On a Mac, macOS's own services take the Neural Engine too: with `mediaanalysisd` (122 % CPU) and Apple Intelligence's inference running, every form fell to 0.7 tokens/s (16.112). Look at `top` before measuring and measure again when it settles.

Drafts from an MTP layer: measure 1, 2 and 3 drafts on both code and prose. Each extra draft costs one call of the MTP layer with the output layer, and the third lost on Japanese (Qwen3.5-0.8B / 2B: 2 by default, 16.109). Bring back the state of layers whose state is replaced each step (GDN) on the host from the rows kept (`state-rows`), not by running them again: it helps prompt-lookup drafts too (26.1 → 32.1 tokens/s).

## 8. Long contexts

- Split at the attention (`-paged`): the cost follows the length used. 262144 by default. Split, though, every attention layer calls the Neural Engine twice: a model whose layers all attend (Qwen3-0.6B's 28) runs 2 to 4 times faster as one program (16.127, `llm.Unpaged`). Count the attention layers and measure both
- Two-level attention (`-sparse`): from about 48K. Passkey tests (128K, 250K, and three at 10 / 50 / 90 % of a document)
- Layers with state (GDN): one token at a time must match chunks of Q=64 (the form of `TestCachedMatchesFull`)

## 9. Checks in use, and the record

Do not decide by KL alone. Put the model's answers next to bf16 (or fp16) for what it is used for:

| Use | How to check |
|---|---|
| Chat | 16.26's four questions (Mt. Fuji, 17 × 23, *Kokoro*, the Shinano River), hwnpu-agent's task (does `go test` pass?). When a small model gets facts wrong, ask fp16 the same: if it errs alike, it is the model's limit, not the quantization or the port (Qwen3.5-0.8B, 16.99) |
| Translation | Measure KL and agreement **on the translation's tokens only** (the source side is not predicted, so it scores high). How many of 44 texts are word for word the same as bf16. `hwnpu-formats` measures on chat text, so its perplexity means nothing for a translation model (28 thousand for PLaMo 3 Translate) |
| Image descriptions, TTS | Compare the answers for the same images and audio. Run TTS output back through ASR |
| Transcription | English: LibriSpeech's 73 validation clips (`hf-internal-testing/librispeech_asr_dummy`, with the truth); Japanese: 12 texts in four `say` voices. Count the clips word for word the same as HF fp32, and the WER / CER (16.115). Audio over 30 seconds too |
| Embeddings | Cosine against HF |
| Image generation | Compare images from the same noise. The composition can change with some seeds (one in four) |

**In practice**:

- fp16 programs for the comparison double the disk used. Remove a format's programs once compared
- Core ML makes `~/Library/Caches/<process>/com.apple.e5rt.e5bundlecache` for every process, test binaries (`transcribe.test`) included, several GB each (6.2 GB from the Whisper comparison)
- A stalled download never times out (`chat.fetch` does not detect stalls; kotoba-whisper stopped at 158 MB for an hour)
- `hwnpu-formats` shows the weights of a single unsplit program as 0.00 GiB

**The record**:

- Add a section to chapter 16 of design.md (numbers, what was adopted, what was not and why)
- The tables in [improvements.md](improvements.md) and [improvements.ja.md](improvements.ja.md) (speed, memory, accuracy, the model list in section 10, not yet verified in section 11)
- What could not be measured on a Mac goes into `docs/handoff-*.md`, with what to bring, the commands and the expected values

## 10. Values for combining it as a part

Later, models will be chained, such as speech recognition → LLM → TTS, and run at the same time in one process (design.md 16.94, a plan, not implemented). Measure and record the values that the residency management and the matching with the connected models will need when the model comes in.

| What to record | How to measure | What it is for |
|---|---|---|
| The pinned size of each part (stage, tower, DiT, VAE, ...) | AMD: the size of the process's `/dev/accel/accel0` mappings (`smaps`). Mac: each part's weights and `phys_footprint` | The memory budget |
| Each part's reload time (the second load) | AMD: `Release` / `Reload`. Mac: when the ANE's cache hits | Whether it may be evicted. Parts whose delay is heard, as in a speech chain, stay resident |
| Parts that run together, and parts that run in turn | From the flow of the processing | Only parts that run in turn can be loaded a stage at a time (Anima's text → DiT → VAE) |
| AMD: the longest command | Logs of `hwnpu-chat` / `irongraph` | Other models wait for it. The further below 2 s (4.1), the easier to run beside others |
| Mac: the parts' total weights and the switching time | `TestCallTimes`'s alternating calls | Past the ANE's cache (about 4 GiB), parts are evicted and recompiled in minutes. A switch costs 15–35 ms |
| The data kinds and conditions of the inputs and outputs | The model card and the preprocessing | Text (tokenizer and languages), structured text (can it follow a schema), images (sizes it takes), audio (sample rate, channels). For matching against the connected models |
| Whether it can stream, and the time to the first output | Feed it chunks and take chunks out | Estimating the delay of a speech chain |

- Pass data to other models as text (token IDs when the tokenizer is the same). Hidden states are not compatible with another model's, even with the same tokenizer. Connect vectors directly only between parts trained together (LFM3-Audio's text model and depthformer, a tower and its projection, Anima's text encoder and adapter)
- Record a model that keeps audio as audio (Audio, Omni) as one part whose input and output are audio

**Pass**: the values in the table above are written in the design.md section.

## 11. Tried without effect (do not repeat)

The gist of section 8 of [improvements.md](improvements.md): padding small models into batches for the ANE (slower than the CPU), fixing a small Qwen3.5's 4-bit MLP with GPTQ (the GPL stays broken; q6 instead), a faster GEMV from q6 (q8's time), getting past 1 GiB with multifunction (all on the CPU), the KV cache in MLState, an int8 KV cache, loading the next tier in the background, one Q=16 program, E4B at lut4, compressing the model, sparse attention on the top keys alone, AMD attention at 4 bits, a GEMV with one core per column, copying MoE experts, the DiT at `HIGH`, the DiT at q4 by default, q4nx prompts through bf16, RMSNorm with a fixed scale. Whisper's encoder LayerNorm divided by each row's maximum (the output did not change, 16.115), Whisper's encoder attention in chunks of queries (no faster).

## 12. In progress (2026-11-08)

- Qwen3.6's prompt attention pipelines are not measured (it keeps 1)
- AMD's checking of drafts (16.117): a GEMV of a few rows for q4 (Gemma 4, PLaMo 3, LFM3, Qwen3.5-4B and Qwen3.6 at 4 bits), Gemma 4's shared layers and ring caches, Qwen3.5's MTP layer as the drafter
- Qwen3.5-2B at lut6: the KL against HF's bf16 (the 2B in fp16 does not fit a part, so so far only against lut8), long contexts and the hwnpu-agent tasks
- Whisper: the formats on real recordings (noise, overlapping speakers)
- Qwen3.5-4B's lut6 (2.97 GiB) on the M1 is just over its 2.8 GiB; 0.1–0.2 GiB less (the output layer at lut4, say) would let it run at full speed (16.109)
- Apple's GPTQ codes: hwnpu-chat takes them with `-codes` and `-format-for`, not by default (how to hand out the codes is open). E4B does not fit a 16 GB Mac to calibrate

When results come in, update the tables in 5.3 and 5.4 and this section.
