# Decoding: drafts, think budget, loops

[日本語](llm-decoding.ja.md)

How `chat.Engine` generates beyond one token a step: speculative decoding with prompt-lookup
drafts and Qwen3.5's MTP drafts, the think budget, and finding loops. On all backends; AMD's
verify program is in [amd-llm.md](amd-llm.md). The cache and the split are in
[llm-engine.md](llm-engine.md).

## Flags and defaults

| Setting | Default | Meaning |
|---|---|---|
| `--draft N` | 8 | tokens per call of the generating program; drafts = N − 1. 0 or 1: off (Q=1) |
| `--mtp` | `auto` | also draft from the checkpoint's MTP layer when it has one (Qwen3.5); `off`: prompt only |
| `HWNPU_DEV=mtp.drafts` | 2 (`mtpDrafts`) | MTP drafts in a row |
| `HWNPU_DEV=iron.drafts=0` | | AMD: do not use the verify program |
| `--think` / `--no-think` | on, unless the model's `no_think` | let the model think first |
| `--think-budget N` | 1024 | thinking tokens before it is made to answer; 0: no limit |
| `--rep` | 1.05 | repetition penalty on the answer's tokens |
| `-n` | chat 0 (until the cache is full), agent 2048 | answer length cap |
| API `reasoning_effort`, `chat_template_kwargs.enable_thinking` | | override `--think` and `no_think` per request (`Server.thinks`) |

Logs and stats (`chat.Stats`): `Drafted`, `Accepted`, `Calls` (`(a of d drafts accepted, c
calls)`), `ThinkCut` (`thinking cut at N tokens`), `ThinkLooped` (`thinking cut going round in
circles`), `Looped` (`stopped repeating itself`).

## Speculative decoding

Why: on the Neural Engine a call costs about the same for 1 or 8 tokens (M3, LFM3-1.7B lut4,
Core ML: Q=1 61.5 ms, Q=8 72.0 ms (+17%), Q=64 103.2 ms). Q=8 pays off at 0.15 accepted drafts a
call. Agents copy what they read into what they write, so answers repeat the prompt.

### One program for generation

- With `--draft 8` the generating program is Q=8, and steps without drafts run on it too. Two
  programs are loaded as before: generating and prompt reading.
- Reason: switching programs costs (M3, Core ML: alternating Q=1 and Q=8, 35 ms a switch on a
  quiet machine, over 100 ms under swap). Switching per draft made no-draft calls 2× slower.

### Prompt-lookup drafts (`LookupDraft`)

- The tokens after the latest earlier occurrence of the last n tokens, n = 4 down to 1, up to k.
- 1-grams are drafted too: checking is free at Q=8 (replay of 56 requests: 3..2-gram 1.552
  tokens/call, 4..1-gram 1.673, +7.6%).
- Draft length is cut by the answer cap, the cache left and the think budget left.

### Acceptance is exact

- `Generate` passes "1 real token + k drafts" to `speculate`. Draft i is kept if it equals the
  token sampled from position i−1's logits. The first mismatch's sample, or the last row's when
  all match, is the next real token.
- Greedy output is identical to one token a step. With temperature the distribution is unchanged
  too: each kept token is exactly the token sampled at its position. No rejection sampling, so
  no correction and fewer hits. Drafts cannot add repetition.
- Each kept token goes through `take` (stop tokens, think budget count, `emit`, repeat check,
  limits).

### Keeping part of a step (`decode.Session.Speculate`)

- `Step` is split into `run` (write inputs, run) and `keep(v')` (write the first v' rows to the
  cache). `accept` reads per-row logits (`row(i)`) on demand.
- Full-attention caches: rows past the position are masked, so rejected rows are left. Rings
  write only i < v'. Conv windows: take rows v'..v'+W of `win_*_new` (W+Q rows). No copies.
- Whole-replaced state (Qwen3.5's GDN `st_`): the output state is after all rows, so it cannot be
  cut. Two ways:
  - `state-rows` (`llm.StateReplayer`, `qwen35.Config.ReplayState`, program suffix `-sr`, the
    draft-checking program only): the program also outputs `st_rec_N_rows` (1, value heads, rows,
    dk+dv+2: k, v, β, decay per row); the host advances the delta rule from the pre-step state for
    the kept rows only (fp32, parallel per head). One call. 0.8B: 66 KB a layer.
  - Fallback: rerun the kept tokens (the input buffer still has the pre-step state). Two calls.
- AMD (IRON) has its own verify program and keeps the row state on the device (`iron.Model.Keep`);
  no host replay. See [amd-llm.md](amd-llm.md).

### Measured (prompt lookup)

M3 16 GB, Core AI, LFM3-1.7B lut6, `hwnpu-chat serve`, temperature 0, no thinking, the same prompt three
times (the runs agreed within 1%):

| | Prose, 300 tokens ("how a bicycle gear works") | Copying a 1159-token Go file back, 600 tokens |
|---|---|---|
| `--draft 8` | 49.5 tok/s (20.2 ms/token) | 285 tok/s (3.5 ms/token) |
| `--draft 0` (Q=1) | 44.8 tok/s (22.3 ms/token) | 44.8 tok/s (22.3 ms/token) |

- On Core AI the Q=8 program costs about what Q=1 does, so drafts are 10% faster even on prose; copying
  text the prompt holds is 6.4× faster. (On Core ML, Q=8 cost more: an agent task gained 1.2–1.5× and
  thinking with nothing to copy was 10–15% slower.)
- Hits: tool-call XML, repeated commands (`go run main.go`), summaries restating earlier text.
  Free prose rarely hits (28 of 340 on a "why is the sky blue" answer).

## Qwen3.5 MTP drafts

The Qwen3.5 checkpoints carry one multi-token prediction layer (`mtp.`). hwnpu uses it as the
draft model.

### The layer (`internal/llm/qwen35/mtp.go`, `-arch qwen35-mtp`)

- Ported from vLLM's `qwen3_5_mtp.py` (transformers has none). At position t: the embedding of
  token t+1 and the model's final hidden state h_t (after the last norm), each through a
  zero-centered norm (`pre_fc_norm_embedding`, `pre_fc_norm_hidden`), concatenated embedding
  first, `fc` (2d → d), one full-attention layer `mtp.layers.0`, `mtp.norm`, the model's head →
  logits of token t+2.
- As an `llm.Model` it is 2 layers: layer 0 is the front (norms, concat, fc) so that `-paged` can
  split before the attention (layer 0's attention cannot run on the host). Inputs `x` and `hid`.
  With a cache it also outputs its own `h`; drafts after the first read that instead of the
  model's (as vLLM).
- The model side: Arch param `hidden-out` (program suffix `-h`) outputs every row's final h;
  `decode.Session.HiddenRows` returns the kept rows.
- Checks: fp64 reference of vLLM's forward 1.3e-7 (`TestMTPAgainstFP64`), cached, `Truncate`,
  split (`TestMTPCached`, `TestMTPPaged`). Swapping the concat order fails 8 tests.

### The drafter (`internal/chat/mtp.go`, `mtpDrafter`)

- After each model call it receives the kept tokens' h (`Engine.recordHidden`). Before drafting it
  feeds the positions whose next token is known (Q=8, `LastRow`); the last row's argmax is draft
  1, later drafts one token a call. Then `decode.Session.Truncate` back to the real position
  (all its state is per-position KV, so no marks).
- Positions start from `base`: RoPE attention sees only relative positions. On a rewind or a new
  conversation it keeps the common prefix (`sync`). More than 512 positions behind (`mtpStart`)
  it restarts there instead of reading the whole prompt.
- If a prompt-lookup draft agrees with MTP's first and is longer, it is used.
- `TestMTPDrafts`: output identical to one token a step; first drafts equal a fresh MTP pass.
- Conversion (`chat.Prepare`, `--draft` ≥ 2, `--mtp auto`): model `-h`, draft-checking program
  `-sr`, MTP program `qwen35-mtp-<format>-p8192-[dv2-]c262144-q8-last`.

### Partial vocabulary (`draftVocab`, rule `dv2`, Core AI only)

- The MTP head is the model's 248320 rows: 11.4 of 13.6 ms of an MTP step (4B, M2), twice per
  model step. Drafts are checked by the model, so a vocabulary subset changes no answer; it only
  loses drafts outside it (as FR-Spec). `llm.Options.HeadRows`, `llm.SelectRows`.
- Rule: IDs < 32768, plus tokens with kana, of 1–2 kanji, or of ASCII symbols and spaces only;
  padded to a multiple of 16. About 74K rows (30%): 97% of Japanese text, 96% of English and Go.
  Low IDs alone cover English 95% but Japanese only 65% (its tokens are spread over the vocab).
- Listed in the export's `draft-vocab.json`; `mtpDrafter.token` maps back. The rule is in the
  program name.
- Effect (4B, M2, Core AI): MTP part 2 0.68 → 0.26 GiB, MTP step 13.6 → 6.6–6.9 ms, same bytes of
  answer. With the MTP attention also on short chunks (`AddSmaller`): MTP step 6.5–6.8 ms,
  generation 20.5–21.9 tokens/s.

### Measured (MTP)

M1 16 GB, Core ML, `--prefill off`, temp 0, tokens/s (accepted / drafted), with `state-rows`:

| Drafts | 0.8B code | 0.8B Japanese | 2B lut8 code | 2B lut8 Japanese |
|---|---|---|---|---|
| prompt only (`--mtp off`) | 32.1 (79/696) | 25.8 (11/323) | 23.5 | 19.3 |
| MTP 1 | 40.9 | 27.4 | 29.6 | 21.6 |
| **MTP 2 (default)** | **43.8 (169/308)** | **30.3 (90/263)** | **32.4** | **20.1** |
| MTP 3 | 40.1 | 23.7 | 33.9 | 17.5 |

- `state-rows` vs rerun: prompt-only 26.1 → 32.1, MTP 2 38.1 → 43.8.
- 2B lut6 on M3: 18.3–24.5 → 27.2–43.7 tokens/s with MTP, 4 answers still right.
- Greedy text can drift after ~30 tokens: the NE's fp16 rounding depends on the rows run together
  and the host replay is fp32. Bit-identical on the CPU checks.

## Think budget (`chat.ThinkLimit`, `chat.Request.Think`)

- Inside thinking (between the template's open and close marks, or a prompt `Render` ends inside
  the thinking), after `Budget` tokens without closing, the template's `End` (a "no more time"
  sentence plus the close mark; `Template.Think`; LFM3 `</think>`, Gemma 4 `<channel|>`) is fed as
  if the model generated it, Q tokens per `step`. The fed tokens go to `emit`, so streams see the
  close.
- If the model opens `<think>` by itself in no-think mode, the same budget counts from the open
  mark.
- Why: an OpenCode title request (587 tokens, no tools) thought 3440 tokens for 226 s while the
  real request waited (M3, LFM3-1.7B).
- Per-model default: models.yaml `no_think` (`chat.Model.NoThink` → `Engine.NoThink`) when
  `--think` is not given (Qwen3.5-0.8B and 2B think in loops; see [llm-qwen.md](llm-qwen.md)).
- `TestThinkBudget`: `End` after the budget, fed tokens reach `emit`, logits match reading
  everything from the start within 2e-3.

## Loop detection (`chat.repeats`, `internal/chat/repeat.go`)

Why: small models go round in circles (LFM3-1.7B on "Tokyo's 23 wards": rewriting the list in
the thinking, then in the answer for thousands of tokens). With `-n 0` and a 262144 context that
can run for hours. `--rep` only weakens literal repeats.

Real answers repeat too (a draft copied fair, tests with the same setup lines, JSON keys, table
rows). The difference is how often: a fair copy is the 2nd time, a loop keeps coming back a 3rd
time and more. "Seen once before" overlaps (0.50–0.92 in loops, 0.58 in a normal thinking);
"seen twice or more" separates (normal: lines at most 7/16, 8-grams at most 50/256).

| Counter | Rule |
|---|---|
| Lines | lines with ≥ 2 words (tokens with a letter), hashed by tokens; loop when 12 of the last 16 are a 3rd+ occurrence. Lines without words (`}`, `---`, number-only rows) are not counted. Digits stay distinct (merging them stops on date and weather tables) |
| 8-grams | loop when 230 of the last 256 positions end an 8-gram seen 3+ times (loops without newlines, any period) |
| `looping` | the last 64 tokens are one pattern of period ≤ 16 |

- Thinking and answer count apart (reset at `<think>` / `</think>`): writing twice in thinking and
  once more in the answer is fine.
- In the answer: stop (`Looped`). In the thinking: close it and make it answer, as at the budget
  (`ThinkLooped`).
- Token kinds (newline, letter) are built from the tokenizer's vocabulary at the first
  generation; engines without a tokenizer count 8-grams only.
- Checked: on 24 logged normal thinkings and answers it never fires; it fires on the 23-ward
  loops. It is tuned to be late rather than cut a real answer (one 23-ward answer ran ~3700 tokens
  before it stopped).
- Tests (`repeat_test.go`, `TestThinkLoop`): 20 names cycled in shifting order found after 2
  rounds; fair copy of 40 lines, 20 tests with the same setup, 120 rows differing only in numbers
  not found; a newline-free period of 100 found after 200 tokens.

## Not adopted

- Rejection sampling (keep a mismatched draft with some probability): exact match keeps the
  distribution without correction.
- Switching between Q=1 and Q=8 by whether there are drafts: switching costs 35 ms or more.
- Q=4 for drafts: a call only 2 ms (3%) cheaper than Q=8, and 3 drafts give 10% fewer tokens a
  call than 7.
- 3..2-gram or first-occurrence lookup: 1.552 / 1.550 tokens/call vs 4..1's 1.673.
- 3 MTP drafts: fewer hits and slower on Japanese; each draft pays a head pass.
- Draft vocabulary of low IDs + kana + single kanji (`dv1`, 43618 rows): missed 説明, 計算, ```
  and fell from 82 to 67 accepted drafts, slower than the whole vocabulary.
- A default answer length cap: long answers would be cut. Loops are detected instead.
- `--rep 1.1` against loops: weakens literal repeats only.
- Counting "seen once before" for loops: overlaps with normal answers.
- Speculative decoding through AMD's prompt-chunk program: fixed cost of 4.5 decode steps a call;
  prompt lookup accepts 1.07–1.3 a call. A dedicated verify program instead
  ([amd-llm.md](amd-llm.md)).
- Removing the Core AI shim's input copy: about 0.5 ms a call (4% of a 4B step); views of the
  caller's buffer were slower.

## Open

- MTP drafts on AMD (needs the model's hidden state out and an MTP program) and on Intel.
- Time to the first MTP draft after a long prompt (up to 64 calls).
- argmax over fp16 logits to skip converting 248320 values.
- Drafts at temperature 0 vs 0.1 on agent tasks; acceptance on OpenCode's long XML turns.
