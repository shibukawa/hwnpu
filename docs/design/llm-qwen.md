# Qwen models

[日本語](llm-qwen.ja.md)

The dense Qwen chat models: Qwen3 (0.6B, 1.7B), Qwen3.5 (0.8B, 2B, 4B; Gated DeltaNet) and
Qwen2.5-Coder (Qwen2). The mixture of experts Qwen3.6-35B-A3B is in [moe.md](moe.md). Drafts
(prompt lookup, Qwen3.5's MTP layer) are in [llm-decoding.md](llm-decoding.md). The KV cache,
the split at the attention and the tiers are in [llm-engine.md](llm-engine.md). AMD's exports are
in [amd-llm.md](amd-llm.md).

## Models

| Model (models.yaml) | Arch | Apple format | AMD | machine | Memory (Apple) | Notes |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (`qwen3:0.6b`) | `qwen3` | lut8 | no export | 8 | 1.6 GB (M1 1.1) | smallest download (1.5 GB), weak at facts |
| Qwen3-1.7B (`qwen3`, `qwen3:1.7b`) | `qwen3` | lut8 | no export | 8 | 2.1 GB (<16 GB), 2.3 GB (≥16 GB) | `prefill: off` below 16 GB |
| Qwen3.5-0.8B (`qwen3.5:0.8b`) | `qwen35` | lut8 | `--q4 --mlp q6 --head q6 --verify 3` | 8 | 1.5 GB (`tier: none` on 8 GB; 2.3 GB from 16 GB on M2+) | `no_think` |
| Qwen3.5-2B (`qwen3.5` on 8 GB) | `qwen35` | lut6 | same as 0.8B | 8 | 2.2 GB (AMD 2.4) | `no_think`; writes code |
| Qwen3.5-4B (`qwen3.5:4b`) | `qwen35` | lut8 | `--q4` | 24 | 4.2 GB | strongest at code for its size |
| Qwen2.5-Coder-7B-Instruct | `qwen2` | lut4 + GPTQ codes (`codes: faace07a`) | no export | 16, `not_on: [M1]` | 4.4 GB | no thinking, context 32768 |

- On a Mac `hwnpu-chat` runs only Core AI. All parts of all these models are on the Neural
  Engine with Core AI. The generating program and the prompt program are two functions of one
  `.aimodel`, so the weights are held once ([Core AI](apple-coreai.md)).
- Intel: the IR converts (`hwnpu-convert -backend intel`; the architecture's lut8 becomes int8).
  Not measured on the NPU.
- Facts: every small Qwen here gets Japanese facts wrong (Mt. Fuji's height, the author of
  『こころ』, the longest river). HF fp32 gets them wrong the same way, so it is the model, not the
  port or the quantization. They are not in the chat recommendations of MODEL.md.

## Qwen3 (`internal/llm/qwen3`)

Structure: RMSNorm, per-head norms on q and k then RoPE (θ = 1e6), GQA (16 / 8 heads,
head_dim 128), SwiGLU. The output matrix is the embedding (`tie_word_embeddings`; the
checkpoint's `lm_head.weight` is a copy and is not read). 0.6B and 1.7B are the same code
(1.7B: hidden 2048, 28 layers, MLP 6144).

- Rejected configs: `rope_scaling`, windows, `attention_bias`, `layer_types` other than
  `full_attention`. transformers 5's `rope_parameters.rope_theta` is read.
- fp16: the residual grows to 8141 (0.6B, from layer 2) and 14573 (1.7B). Norm inputs are scaled
  by 1/64 before squaring, as in Gemma 4, **except the entry norm of layer 0**: its input is the
  embedding (max about 0.1), and 1/64 makes the square underflow to NaN.
- All 28 layers attend. To split at the attention, the first part has no layers
  (`llm.NoLayers`): layer 0's norm and projections only, outputting `attn_q_0` and K / V.
- Split or not (`llm.Unpaged`, `pagedChunk`):
  - Core AI: split at the attention by default, with the short chunks. Context 40960, cut to a
    quarter of memory (32768 on 16 GB). On an M2 or later a whole program for short conversations
    sits in front (`--tier`, 1024 for Qwen3-0.6B; see [llm-engine.md](llm-engine.md)). M3,
    Qwen3-0.6B lut8: split 36.1 tokens/s (unsplit 19.1); 2149-token prompt 1.7 s (3.6 s).
  - Other backends: one unsplit program with the cache inside (`UnpagedContext` 4096). Split only
    with `--paged N` (`--paged 1024` gives 32768).
  - Pitfall: an unsplit cache of 8192 or 16320 compiles and loads but Core ML's predict fails
    (56 K/V inputs, 0.94 / 1.9 GB).
- Chat (`chat.Qwen3`, `internal/chat/qwen3.go`): ChatML, Hermes-style tools (`<tools>` JSON,
  `<tool_call>{"name": …, "arguments": {…}}</tool_call>`), thinking kept only for answers after
  the last question. Arguments as Python's `json.dumps` (`1e20` → `1e+20`). `Render` writes
  `<think>\n` itself so the stream starts inside the thinking (as Qwen3.5). Matches jinja2 on 9
  conversations (`template_golden.py --text-content`: list contents are made strings first, as
  vLLM does).
- Checks: fp64 reference 1e-4 (tied and untied), layer by layer, cached, split. Real weights
  (`hwnpu-dev check`, 32 tokens):

  | | fp32 / fp16 | lut8 | lut6 |
  |---|---|---|---|
  | 0.6B | KL 0.0000, 32/32 | KL 0.0139, 31/32 | 0.2425, 23/32 |
  | 1.7B | KL 0.0000, 32/32 | KL 0.0181, 32/32 | 0.3148, 25/32 |

- Qwen3-1.7B is byte for byte MOSS-SoundEffect v2.0's `text_encoder/`. models.yaml pins its
  revision and the two files' `sha256`; `fetch` hard links (or copies) a file of the same SHA-256
  from the other model's directory and checks it (`sameFile`, `TestSameFile`). See
  [distribution.md](distribution.md).
- Below 16 GB Qwen3-1.7B runs one program (`prefill: off`); the generating program reads prompts
  8 tokens a call. `--prefill on` adds the prompt function (+0.3 GB with Core AI).
- Speed (Core AI, M1): Qwen3-0.6B 32 tokens/s, Qwen3-1.7B 22 tokens/s.

## Qwen2 / Qwen2.5-Coder (`qwen2`, in `internal/llm/qwen3`)

Qwen3's layer without the per-head q/k norms, with biases on q, k and v. No `head_dim` in
config (hidden / heads = 128). `Qwen2Arch`, `ParseQwen2Config`; paging, cache and `LastRow`
are Qwen3's.

- Chat: `chat.Qwen25`, Qwen3's no-thinking form plus Qwen2.5's fixed system line when there is
  no system message. Matches `apply_chat_template` on 9 conversations (tools, calls, results).
- Checks: a synthetic Qwen2 (bias, no norms, no head_dim, tied and untied) 1.6e-5 against fp64;
  without the bias it fails at 0.70. Real 7B layers 0–2 and the head against a layer-at-a-time
  float32 reference (`tools/ref/ref_qwen2.py`; the 7B in float32 is 30 GB and does not fit on
  16 GB): 7e-5–6.6e-4 per layer, logits 4e-7.
- MLP wider than the Neural Engine's 16384 axis (7B: 18944) is split in multiples of 32 columns:
  gate and up by rows, down by columns (`blocks.Graph.MatrixSlice`; the tables and indices are
  of the whole matrix, so values and codes are unchanged). In one piece every part fell off the
  Neural Engine (0.1 tokens/s). Split MLP part: 2.94 ms (125 MiB, 44 GB/s). `Qwen2Arch.Revision`
  1. 14B's 13824 is not split.

### Weight format

The architecture default is lut8. The 7B in models.yaml is `format: lut4` with GPTQ codes.
CPU, all 28 layers chained, KL against the float32 reference on a 64-token chat
(`TestRealChained`):

| Format | KL mean | median | top-1 | 7B weights |
|---|---|---|---|---|
| lut4 uniform | 0.90 | 0.41 | 36/64 | 3.5 GB |
| lut4 k-means | 0.47 | 0.20 | 45/64 | 3.5 GB |
| lut4 k-means weighted by input E[x²] (`palette.QuantizeWeighted`) | 0.23 | 0.078 | 51/64 | 3.5 GB |
| **lut4 whole-row tables + GPTQ (default, Neural Engine usable)** | **0.131** | **0.051** | **54/64** | 3.5 GB |
| same, `--format-for down_proj=lut8` | 0.078 | 0.035 | 57/64 | +0.95 GB |
| same, attention lut8 | 0.105 | 0.043 | 55/64 | +0.44 GB |
| lut4, tables per 32 columns + GPTQ (wrong on the NE, see below) | 0.060 | 0.038 | 62/64 | 4.5 bit |
| lut6 k-means | 0.18 | 0.078 | 50/64 | 5.3 GB |
| lut8 | 0.0012 | 0.0008 | 63/64 | 7.1 GB |

- Calibration: the repo's Go, Python and English/Japanese READMEs, 512 tokens each (not the
  evaluation chat). Input E[x²] from `ref_qwen2.py --stats`.
- GPTQ (`tools/ref/qwen2_gptq.py`): layer at a time, calibration through the quantized layers,
  inside a layer q/k/v → o → gate/up → down. Tables are fixed first by weighted Lloyd; GPTQ picks
  only each weight's index. down's H (18944²) is 1.4 GB in float32. 77 min for the 7B on M1.
- Codes are written as `palette.Codes` (`index.json`, `.idx`, `.tab`; `palette.CodesWriter`).
  `hwnpu-chat --codes DIR` converts with them. The codes hash is in the program name, so a
  converted export can come from a source; converting needs `models/NAME/codes-HASH` or `--codes`.
- **The Neural Engine silently gets per-column-block tables and scales wrong** (Core AI,
  macOS 27.2, M1): it uses the first column block's table for every column. int4-g32/g128 and
  int8-g32 are wrong too (cosine 0.966–0.984); per-row int4 and whole-row lut4 are exact. Core
  AI's CPU is right, so it is the NE compile. Core AI programs refuse column blocks at conversion
  (`blocks.Graph.rowsOnly`). Core ML (int8-gN to the CPU) and Intel keep them. See
  [apple-coreai.md](apple-coreai.md), [quantization.md](quantization.md).

### On hardware (M1 16 GB, Core AI, `qwen2-r1-lut4-gptq-faace07a`)

- Convert 2 min 35 s (29 programs), first load 1 min 15 s, then 0.5 s. All on the Neural Engine.
  Answers right (an iterative Go Fibonacci, goroutines in two sentences).
- 0.8–0.9 tokens/s: 4.0 GB on the NE is over the 2.8 GiB an M1 holds at full speed, so the
  weights reload every call. Hence `not_on: [M1]`. With all weights resident the estimate is
  about 0.1 s a token (not measured). M2 runs 3.65 GiB at full speed (measured with Core ML).
- Memory: NE 4.0 GB, process 0.35 GB. 6.3 GB during the first conversion.

## Qwen3.5 (`internal/llm/qwen35`)

Ported from `modeling_qwen3_5.py` (transformers 5.22). Checkpoints are image+text
(`Qwen3_5ForConditionalGeneration`, tensors `model.language_model.`); text-only (`qwen3_5_text`,
`model.`) is read too. The vision tower (`model.visual.`) is not converted. Sharded safetensors
(`model.safetensors.index.json`) are read (`llm.OpenWeights`).

### Structure

- Three Gated DeltaNet (GDN) layers then one full-attention layer, repeated. 4B: 32 layers
  (24 GDN, 8 attention), hidden 2560. 0.8B / 2B: 24 layers (attention at 3, 7, …, 23). Vocabulary
  248320, output tied to the embedding.
- RMSNorm is zero-centered (`1 + w`), except the GDN's last norm (`Qwen3_5RMSNormGated`: `w`,
  gated by `silu(z)`). `1 + w` is rounded to fp16 once (rounding w first moves ~3% of elements).
- Full attention: `q_proj` gives the query and an output gate per head (256 + 256), QK norm,
  RoPE on the first 64 dims only (`rotate_half` within them), GQA (4B: 16 heads, 4 KV), output ×
  `sigmoid(gate)` per channel. mrope's 3 positions are equal for text, so it is plain RoPE.
- GDN: `in_proj_qkv` → depthwise causal conv (width 4) + silu; q and k L2-normalized (q also
  × 1/√128; L2 norm written as `RMSNormWith` with eps 1e-6/n times 1/√n). Key heads are shared by
  value heads (`repeat_interleave`; 4B 16 → 32, 0.8B/2B 16 = 16). Decay
  `exp(-exp(A_log) · softplus(a + dt_bias))`, write strength `sigmoid(b)`, state S (128 × 128 per
  value head): `S ← S·decay; S ← S + kᵀ((v − kS)·β); o = qS`.
- MTP layer (`mtp.`): used for drafts, see [llm-decoding.md](llm-decoding.md).

### State

- **Whole-replaced state `st_`** (`llm.Cached`): the GDN state cannot be written per position.
  The program takes `st_rec_N` (1, heads, 128, 128) and outputs `st_rec_N_new`; the host
  replaces the input with it. The conv window is `win_conv_N` (last 3 rows), as LFM3.
- Padding must not change the state: `StepInputs` passes `step_valid` (1 token, 0 padding), and
  padding gets β = 0 and g = 0.
- `Mark`, `Rewind`, `Resume`, `Migrate`, `chain.Share` treat `st_` as `win_`
  (`llm.CacheInput`, `llm.HeldInput`).
- The host keeps the state in fp16. Over 3276 tokens the loss does not drift (positions 2048–4096
  3.356 / 3.310 vs PyTorch fp32).
- Partial draft acceptance: `state-rows` replays the kept rows on the host; otherwise
  `Speculate` reruns. See [llm-decoding.md](llm-decoding.md).

### Gated delta rule over chunks (`delta=chunk`, default)

Row by row (`delta=rows`), the state (4B: 32 × 128 × 128 fp16 = 1 MiB a layer) is read and
written 5 times a row: ~1.3 GB a step over 24 layers, 30% of the 4.25 GB weights.
`deltaRuleChunks` (flash-linear-attention's `chunk_gated_delta_rule`, WY form) reads the state
as `k S0` and `q S0` and writes it once per chunk of 8 rows. With G_t = g_1 + … + g_t
(log decay), Γ_t = exp(G_t):

- (I + N) U = diag(β)(V − diag(Γ) K S0), N_tj = β_t exp(G_t − G_j) k_t·k_j (j < t)
- O = diag(Γ) Q S0 + (Q Kᵀ ⊙ exp(G_t − G_j)[j ≤ t]) U
- exp above the diagonal is taken of min(·, 0) then masked, so it cannot overflow. Padding rows
  (β 0, g 0) leave the state alone.

Neural Engine pitfalls found:
- An (8 × 8)·(8 × 8) matmul pushes the whole part off the NE (k·kᵀ with inner 128 and
  (8 × 8)·(8 × 128) stay on). So no inverse: N is nilpotent, U = B − N(B − N(B − …)) by 7
  (8 × 8)·(8 × 128) products.
- Cumulative sum G and differences G_t − G_j are a constant `linear` over g's rows.
- A tiny test program (45 ops) is put on the CPU whatever its shape, so placement must be probed
  inside a real-sized program.
- The NE miscomputes a matmul of inner dimension 1 (M3). A chunk of one row (Q=1) uses `mul`
  instead (`Revision` 2).

Result (M2, Core AI, 4B): GDN part 9.51 → 8.23 ms (≈56 GB/s, near the 62 GB/s ceiling), its
first compile 15.8 → 7.0 s, model step (Q=8, 9 parts) 96 → 83–89 ms, 128-token prompt 511 →
304–344 ms. Same error per part; greedy text can differ in near ties (summation order).

### Neural Engine numerics

The ANE's own `silu` and `softplus` broke long prompts (lut8 per-position KL 0.33 / 0.76 on
Japanese / English; Core ML CPU ~0.003). Fixes, now in `builder`:

| Problem | Fix |
|---|---|
| ANE `softplus` returns 0 for some inputs (decay 1.0 instead of 0.003–0.13) | `relu(x) + log(1 + exp(−|x|))` |
| ANE `silu` off by 1.4%, worst in the small tail; the key L2 norm magnifies it (30% on rows of size 0.001–0.01) | `x / (1 + exp(−x))` (0 for x < −11). Also the gate and the MLP |
| o = qS terms ~1e-5, below fp16's normal range (6e-5): o off 6%, the gated norm makes it 17% | q × `oScale` (1024), a separate `mul` by 1/`oNormIn` (64) before the norm, eps × (1024/64)² |

- The division is not the norm's `inScale`: that would scale the weight by 1/64 into fp16
  subnormals.
- `TestNoNeuralEngineActivations` checks that programs have no `silu` or `softplus`. `abs` and
  `log` were added to `internal/milcpu` and `internal/ovlower`.
- After the fixes (M3, lut8): per-position KL 0.0073 (JA, 27/27) and 0.0052 (EN, 29/31); 3276-token
  loss 3.362 vs PyTorch 3.354; passkeys 3/3 at 3,276, 12,895 and 59,797 tokens.
- Other models use `blocks.SiLU` too (see [apple-neural-engine.md](apple-neural-engine.md)).

### Formats

| Size | Default | Why |
|---|---|---|
| 0.8B | lut8 | lut6: fp16's next token 242/283 (lut8 274), gets 17 × 23 wrong |
| 2B | lut6 (`Config.Format`, hidden 2048) | next token 260/283 of lut8, perplexity 3.75 vs 3.83, same 10/12 questions; 1.32 GiB vs 1.78 |
| 4B | lut8 | lut6 KL 0.031–0.135, 21–24 of 26–30 tokens; answers break |

- Real weights against HF / PyTorch fp32 (`hwnpu-dev check`): 0.8B fp32 per layer ≤0.0002,
  argmax 32/32, KL 0.0000; lut8 KL 0.0076. 4B all 32 layers (layer-streaming reference
  `tools/ref/ref_qwen35_stream.py`; activations up to 121, no fp16 overflow): fp16 KL
  0.0000–0.0001, lut8 0.0015–0.0050.
- 2B has no fp16 program: the last layer and head make 1.04 GiB, over one part's limit.
- lut4 repeats phrases or miscounts on 0.8B and 2B.
- The MTP layer inherits `Format` (lut6 on 2B).
- **The 2B's lut6 export is rotated** (`-rot`, see [quantization](quantization.md)): weights pass
  through an orthogonal Q, the host tables are `tables-rot.safetensors` (E Q in fp16, 1.02 GB; source key
  `qwen35-rot`). The MTP layer is not rotated: its two inputs (the token embedding and the last hidden
  state) are turned back on the host (`mtpDrafter.plain`, `llm.Unrotate`), then the final norm's
  1 + w is applied. M2, 2B lut6, temperature 0, 200 tokens with MTP: 45.1–45.4 → **51.4–52.4 tok/s**,
  drafts accepted 95/220 → 108/215; 12 questions 9 → 10; NE KL 0.054 → 0.029. The first run after the
  change re-exports and compiles (about 5 min on the 2B).
- 0.8B and 4B (lut8) are not rotated: 0.8B 61.6–62.1 → 63.6–64.5 tok/s, 4B 22.8 → 22.9–23.1, same
  answers on the 12 questions.
- AMD: q4 with the MLP and head at q6 for 0.8B / 2B (they fall apart on long English text at 4
  bits), see [amd-llm.md](amd-llm.md).

### Split at the attention, context, last row

- Split before and after the full-attention layers (`llm.Paged`); the host attends over chunks
  with weightless programs. The part before passes `attn_q_N`, `kv_k_N_new`, `kv_v_N_new` and
  the output gate `attn_gate_N` (1, seq, 16, 256; per channel, LFM3's is per head). GDN state and
  conv window stay inputs of each part. 4B: 9 parts; 0.8B / 2B: 7.
- Context 262144 (`max_position_embeddings`). KV per token: 4B 32 KiB (8 GiB at 262144; 114688 on
  16 GB), 0.8B / 2B 14 KiB (3.38 GiB).
- `LastRow`: the prompt program computes the head for the last row only (`last` one-hot before
  the final norm). Without it the 128 × 248320 output is 60.6 MiB.

### Tokenizer and chat

- tokenizer.json uses NFC and a split rule unlike GPT-4's (`[\p{L}\p{M}]+`, digits one at a time,
  `\p{M}` out of the symbols). `golang.org/x/text/unicode/norm` is a dependency. IDs match HF
  tokenizers (`TestQwen35`, `TestSplitQwen35`). 0.8B / 2B / 4B tokenizers are identical.
- `chat.Qwen35` ports chat_template.jinja: tool declarations with Qwen's text in the system,
  thinking kept only after the last question, arguments JSON for objects and arrays, else
  Python `str()` (True, None). Matches 9 rendered conversations. Parsing is `chat.Parse` (XML, as
  LFM3).
- Thinking: 0.8B and 2B think only when `enable_thinking` is true (template default), so
  models.yaml has `no_think: true`. Asked to think, 0.8B hit the 1024 budget on all 4 test
  questions repeating itself. See [llm-decoding.md](llm-decoding.md).

### Measured

| | Machine | Generation | Prompt | Neural Engine | Start |
|---|---|---|---|---|---|
| 4B lut8 + MTP | M2 16 GB, Core AI | 20.5–21.9 tokens/s | 2041 tokens 4.0 s | 4.05 GiB | first 14 min 39 s (compile 13 min), then 0.7–0.8 s |
| 2B lut6 | M1, Core AI | 20.0–21.0 tokens/s | | 1.67 GiB + process 0.39 GiB | first 1 min 20 s, then 1.6 s |
| 0.8B lut8 | M1, Core AI | 25.3–27.0 tokens/s | | 0.99 GiB + process 0.39 GiB | first 5 min 9 s, then 2.3 s |

- 4B on an M1: 3.96 GiB of lut8 is over the 2.86 GiB the M1's NE holds at full speed; about 1
  token/s. Use 2B on an M1.
- First conversion peaks the process at 6.0 GiB for 2B, 3.0 for 0.8B; on 8 GB machines the
  converted programs are meant to be fetched ([distribution.md](distribution.md)).
- Core ML compiled Qwen3.5's 128-token GDN parts very slowly (0.8B 15–26 min, every start when
  evicted from its ~3.5–4 GiB cache); Core AI keeps them (0.7 s second start).

## Not adopted

- lut6 for Qwen3.5-0.8B / 4B, lut4 for any Qwen3.5: accuracy (see Formats).
- lut6 for Qwen3: KL 0.24 / 0.31.
- Per-column-block tables or scales on Core AI: the NE uses the first block's table (wrong
  answers, part cosine 0.56–0.84 on the 7B). Whole-row tables + GPTQ instead.
- Max-scaled norm (divide by the row max before squaring) for the GDN gated norm: worse on the
  ANE (17% → 40%); the cause was o's magnitude.
- Inverting (I + N) with (I − N)(I + N²)(I + N⁴): the 8 × 8 products leave the NE.
- A Qwen3.5 pair of Q=1 and Q=64 programs: compile did not finish in 24 min (Core ML).
- Sharing Anima's text encoder with Qwen3-0.6B: it is Qwen3-0.6B-Base, a different checkpoint.
- A `gemma2` chat type for Sana's Gemma 2: same size as Gemma 4 E2B, which exists.
- Qwen2.5-Coder on an M1: 0.8 tokens/s (`not_on: [M1]`).

## Open

- Qwen3.5-4B on 16 GB machines: models.yaml says `machine: 24`; M3 / M2 numbers above are with
  one model loaded.
- Qwen2.5-Coder-7B on M2 and later (speed, whether 4.0 GB runs at full speed); 14B (7.0 GB at 4
  bits, checkpoint 29.5 GB, not converted); KL on other evaluation text; long chats and agent
  tasks; an AMD export (q4nx's 32-column scales would be finer than the NE's tables).
- Qwen3 / Qwen2 on AMD (no export) and Intel's NPU (not measured).
- Qwen3.5 on Intel's NPU; the 2B's lut6 KL against HF bf16; GPTQ lut6 for Qwen3.5; long-context
  passkeys and agent tasks on 0.8B / 2B.
- Qwen3.5-4B lut4 on an M1 (fits at full speed, 10.9 tokens/s) needs the KL and passkey checks.
