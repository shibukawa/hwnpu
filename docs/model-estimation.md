# Model evaluation record

[日本語](model-estimation.ja.md)

A record of models checked for whether they are worth running on the NPUs. Models left out of the catalog (models.yaml) are kept here too, so that the same work is not repeated. One section per model (or family); new ones go at the bottom. Each section gives:

- The verdict (added to the catalog, left out, or deferred) and why
- Where and when it was measured, and what it was compared with (models already in the catalog for the same use). A model only estimated from its config.json and weight headers, without running it, says so
- Where its ported code stays, and links to the detailed records (design/)

Candidates surveyed but not yet ported go in survey documents such as [small-models-survey.md](small-models-survey.md).

## Ouro (looped language models) — 2026-11-22

Checked whether a looped language model, which runs the same layers several times a token, is worth using on the NPUs. The port is ByteDance's Ouro (`internal/llm/ouro`, `tools/amd/iron_ouro.py`, the tokenizer's `gpt2Digits`); its code and tests stay. **Verdict: not in the catalog.** Taking speed, memory and quality together on the NPUs, LFM3-1.7B is still the best small chat model.

The structure and checks are in the Ouro section of [LLM models](design/llm-models.md), the AMD export in the porting notes of [amd-llm](design/amd-llm.md). Steps to put it back in the catalog: [not-adopted/ouro.md](not-adopted/ouro.md).

### Candidates

| Model | Size | Loop | Verdict |
|---|---|---|---|
| Inkling 2 Small (thinkingmachines) | 594B total, 17B a token (MoE) | none | × not looped, far too large for an NPU |
| Inkling 2 Mini (thinkingmachines) | 26.4B MoE, 3.4B a token | its 12 layers 6–17 three times | deferred (estimate in the Inkling 2 Mini section below) |
| **Ouro-1.4B-Thinking** (ByteDance) | 1.4B | all 24 layers four times | ported and evaluated (below) |
| Ouro-2.6B-Thinking (ByteDance) | 2.6B | all 48 layers four times | checked on the CPU only; lut8 breaks down (KL 0.13), so not exported for AMD |

ByteDance's paper (arXiv 2510.25741) claims reasoning on par with 4B models for 1.4B and 8B models for 2.6B. BonsaiAI did not benchmark quality.

### Ouro-1.4B-Thinking measured (Ryzen AI 9 365, AMD's NPU)

- Format: MLP and output matrices 6 bit, attention 8 bit, embeddings on the host. Context 2048.
- Against HF (`ironlm -forced` on readme.md, 256 + 64 tokens, `forced_logits.py`): 6 bit KL 0.026, top-1 agreement 98.4 %; bf16 KL 0.0022.
- Speed: decode 268 ms a token at 300, 357 ms at 2047 (3.7–2.8 tokens/s); 3.6 tokens/s in bonsaiai-chat. Prompts about 400 tokens/s.
- Memory: 2.79 GiB pinned on the NPU (1.6 GB of it the KV cache), 0.82 GB of the process, 3.8 GB in all.
- Output tried: on "17 × 23" it made a mistake once, noticed it and redid the sum. Mostly English, no tools.

| Model (AMD) | Decode | Pinned |
|---|---|---|
| LFM3-1.7B q4 | 16.8–16.9 tokens/s | 1.64 GiB |
| PLaMo 3 2B q4 | 89 ms (11.2 tokens/s) | 2.16 GiB |
| Qwen3.5-2B q6 | about 120 ms | 1.97 GiB |
| **Ouro-1.4B-Thinking q6** | 268–357 ms (3.7–2.8 tokens/s) | 2.79 GiB |

(The others' figures are from [amd-llm](design/amd-llm.md) and [USAGE_LLM_AMD](../USAGE_LLM_AMD.md).)

### Findings

- **Compute grows with the loop count.** A token costs what a 96-layer model's does. Decoding on the NPU is bound by reading weights, so it is that many times slower: a third to a fifth of the speed of models with weights of about the same size.
- **Weights and download do not grow, the KV cache does.** Each pass keeps its own cache, so Ouro's is four times a 24-layer model's. Ouro also gives all 16 heads their own K and V (no GQA): 768 KiB a token, 16 times a 24-layer model with 4 K/V heads (48 KiB). The weights (about 1.1 GB at 6 bit) are fixed while the cache grows with the context: at 2048 the cache already exceeds the weights, at 8192 it is 6.4 GB. The smaller the model, the more this shows.
- **The weights cannot be rounded small.** Each weight is read four times, so its rounding compounds: lut4 on the CPU gave KL 5.1 (unusable), AMD's 4 bit KL 0.13. Unlike ordinary small models it needs 6 bit or more. The 2.6B, at 192 layer runs, gave KL 0.13 even with 8 bit tables (lut8).
- **The NPU needed a fix of its own.** All four passes read the same weights, so IRON's command cuts (by the weights a command reads) never came: a prompt chunk was one command, and its 233 MB of control code did not fit the 64 MiB device heap. Cutting a command every 12 layer runs fixed it (`install_cuts`).
- **The reference needs old software.** The remote code fails on transformers 5 and needs 4.54. transformers 5 also replaces GPT2Tokenizer's pre-tokenizer and drops Ouro's Digits step.

### Outlook

- A looped model should pay off where the weights are large and only some layers loop. Inkling 2 Mini is such a case (next section).
- Left for Ouro: the Neural Engine and Intel's NPU were not tried; fewer passes (`total_ut_steps` 3) were not measured for speed or quality.

## Inkling 2 Mini (looped MoE, estimate only) — 2026-11-22

thinkingmachines/Inkling-2-Mini (Apache 2.0, revision b85c3cfa), estimated without running it from config.json, the README and the safetensors header (shapes and types only, read with a range request). **Verdict: deferred.** Its experts do not fit a 16 GB machine, so it is for machines of 24 GB and more, and its many new parts make the port large.

### Structure

- 24 layers, hidden 2304. Attention with 24 query heads and 8 K/V heads (GQA), head_dim 128. Layers 5, 11, 17 and 23 attend globally, the other 20 in a window of 1024. Vocabulary 201024, context up to 1M.
- Every MLP is an MoE: 8 of 128 experts (no shared expert), intermediate 1152; the router has a bias, route_scale 8.
- Loop: layers 6–17 (12 layers) run three times with the same weights, 48 layer runs a token. Rank-64 gates run between passes (`controllers`, forget and gain).
- Parameters (from the header): text 26.07B (experts 24.46B, attention 0.64B, embeddings and head 0.93B), vision tower 0.32B, plus an MTP for speculative decoding (3 layers, 0.64 GB). The README gives 3.4B active a token, 5.4B effective with the passes.
- Only a quantized checkpoint is published (about 20 GB): experts of layers 0–19 in 4 bit (NVFP4: an FP8 scale per 16), of layers 20–23 in FP8 (E4M3). No bf16 release.
- Parts BonsaiAI does not have: `attn_res` (the residual gathered by attention over the outputs of blocks of 6 layers, with a query vector per layer), short convolutions (width 4) on the residual and on K and V, cross-pass queries (`middle_cycle_cross_pass_*`), attention sinks, relative position logits (d_rel 16, extent 1024), log scaling of attention, L2 clipping of q and k, a norm on V, soft caps on the MLP's gate and up (4, 25), a logit soft cap of 30.

### Memory estimate (AMD, in Qwen3.6's form: experts 4 bit, attention 8 bit, embeddings on the host)

- Weights: experts about 13.0 GB, attention and the rest about 0.7 GB, head about 0.25 GB: about 14 GB on the NPU. Embeddings (0.93 GB in bf16) on the host.
- KV cache (bf16, assuming each pass keeps its own): global attention runs 8 times a token (layers 5 and 23 once, 11 and 17 three times each), 32 KiB a token; the windowed layers' 40 runs hold 160 MiB whatever the context.

| Context | KV cache | Against the weights on the NPU |
|---|---|---|
| 8192 | 0.41 GiB | 3 % |
| 32768 | 1.16 GiB | 9 % |
| 131072 | 4.16 GiB | 32 % |

  If the passes share their caches, global attention is 16 KiB a token; the model's code will tell which.
- Against Ouro-1.4B the cache a token is 24 times smaller (32 KiB against 768 KiB). Against the weights it is 3 % at a context of 8192, where Ouro's is 145 % at 2048. Since only half the layers loop, with GQA and windowed attention and large weights, the loop's extra cache hardly shows.
- In all, about 14.5 GB on the NPU at a context of 8192 and 1–1.5 GB of the process. Holding every expert on the NPU needs a machine of 24 GB or more. On 16 GB, as with gpt-oss-20b and Qwen3.6, only some experts would stay in place and the others be read from the disk. But each of the three passes picks its own experts, so a middle layer uses up to 24 a token, and more are read from the disk.
- Download about 20 GB (19.5 GB of the model and 0.64 GB of the MTP).

### Speed estimate (AMD)

- Weights read a token: experts 48 runs × 8 about 1.6 GB, attention 48 runs about 1.4 GB (8 bit), head 0.25 GB: about 3.2 GB.
- Time a layer run from models already measured: gpt-oss-20b (19.4 tokens/s with every expert in place, 24 layers) 2.1 ms, Qwen3.6-35B-A3B (8.5 tokens/s, 40 layers) 2.9 ms. 48 runs make 100–140 ms a token (7–10 tokens/s), slower still for the new parts: **5–9 tokens/s** (not measured).
- Much slower and heavier than LFM3-1.7B (16.8 tokens/s, 1.64 GiB). Its README benchmarks it beside Gemma 4 31B and Qwen3.8-27B, though, so those are what it should be compared with.

### If it is ported

- Reference: transformers supports it (the HF blog in the README); check first whether it needs a newer transformers.
- Quantization: rounding NVFP4 again into BonsaiAI's 4 bit (q4nx) rounds twice. Find out whether NVFP4's blocks of 16 with FP8 scales map onto q4 tables as they are.
- The MoE parts (routing on the host, expert places, experts read from the disk) come from gpt-oss-20b and Qwen3.6. New code: the parts BonsaiAI does not have (above) and the gates between passes.
- Apple: the README lists Macs of 32 GB and more. BonsaiAI would run the experts on the CPU, as for Qwen3.6.
