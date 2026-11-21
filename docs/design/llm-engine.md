# LLM engine

[日本語](llm-engine.ja.md)

The model-independent part of running a language model on an NPU: the KV cache and its layout, splitting
a model at its attention, the attention programs, prompt reading, prompt reuse in memory and on disk, and
long contexts. Packages: `internal/llm/decode` (`Session`), `internal/llm/paged` (attention programs
without weights), `internal/llm/chain` (parts in a chain), `internal/llm/runner` (CPU interpreter runner),
`internal/chat` (`Engine`, `Prepare`, `store`). AMD's NPU runs its own path through IRON
([LLMs on AMD](amd-llm.md)). The survey of KV cache methods and their fit to the Neural Engine is
[../kv-cache.md](../kv-cache.md) (kept as is).

## The cache lives on the host

- The cache is host memory passed to the programs as inputs and outputs (not MLState). `Bind` hands a
  buffer to Core ML without a copy, so the cache is the input buffer itself, and the host writes the new
  rows after each step. Buffers are plain fp16 bytes, so the same `decode` code runs on Core ML, Core AI,
  Intel and the CPU interpreter (`milcpu`) the tests use.
- Roles come from the names (`llm.Cached`):

| Input | Shape | Meaning |
|---|---|---|
| `kv_NAME` / output `kv_NAME_new` | (1, kv_heads, cache, head_dim) / (1, kv_heads, Q, head_dim) | Keys after RoPE, and values. The host writes the new rows at the step's positions |
| `win_NAME` / `win_NAME_new` | (1, W, width) / (1, W+Q, width) | A short convolution's last rows (LFM3: 6 rows of B·x). The host keeps the W rows that end at the last real token |
| `kv_NAME_cJ` | per chunk | A whole program's cache over 16320, in chunks (below) |
| other inputs | | Built by the model from the position (`StepInputs`): masks per attention type (1, 1, Q, cache+Q), RoPE tables (1, Q, 1, head_dim). No `gather` (it would go to the CPU) |

- A step takes up to Q tokens. Unused slots are zero and their rows are not written. Each part holds
  the caches of its own layers, so this composes with splitting into parts.
- Keys are stored after RoPE, so the order of rows in a ring does not matter (softmax does not depend on
  key order).

### Sliding-window layers are rings

- A sliding layer's cache is its window (`Config.cacheLen`: the window when it is shorter than the
  cache). The host writes row `pos mod W`. Rings and full caches are told apart by length alone.
- The ring mask: slot j was last written at `pos-1-((pos-1-j) mod W)`; it is visible when that is ≥ 0
  and inside the window.
- LFM3-1.7B: 5 of 11 attention layers slide (window 256). Rings cut the per-1K-capacity cost of a whole
  program from 5.4 to 2.8 ms (M3).

### Tensor limit: 16384 per dimension

- The Neural Engine takes at most 16384 along one dimension (`llm.NeuralEngineMaxDim`). An attention whose
  key axis is 16385 or more goes to the CPU (M3: 16.6 ms → 117 ms for 2 layers; a 16384 cache made a
  token 375 ms, load 140 s).
- So a whole program's cache plus Q must stay ≤ 16384: `chat.DefaultContext` = 16384 − 64 = 16320 for a
  model not split at its attention, with prompt programs of 64 tokens. `hwnpu-dev convert` warns when
  `-cache + -seq` exceeds the limit.
- Past 16320, a whole program's cache is split into input chunks of 16384 (`llm.CacheChunks`,
  `CacheChunkOver` = 16320, `CacheChunkLen` = 16384; the last chunk shorter). `blocks.ChunkedSDPA` merges
  the chunks and the step's own tokens (`mask_TYPE_self`) inside the program with elementwise ops: exponent
  from the global max plus ln 4n (keeps sums in fp16), sums and weighted values as matmuls (`e @ 1`,
  `e @ V`), one division at the end. The rule depends only on the cache length, so the 1-token and 64-token
  programs share buffers. Rings are never chunked. This path is slower per token than splitting at the
  attention, which is the default for every model that allows it.

## Two programs, one cache

- A chat model has a program that generates (Q = `--draft`, default 8: a token plus drafts checked at once,
  see [decoding](llm-decoding.md)) and one that reads prompts (Q = 64, or 128 split at the attention). Both
  are converted for the same cache. `chain.Share` binds the second chain's cache inputs to the first chain's
  buffers; `decode.Session.Add` checks they are the same memory. `Step` picks the smallest program that
  takes the tokens.
- The weights are loaded twice. `--prefill` (`auto` default, `on`, `off`) decides:
  - `auto` keeps one program (the generating one reads prompts too) when the two would not stay in the
    Neural Engine's compile cache (about 4 GiB; budget `compileCacheBytes` = 3.5 GiB) or, on an M1, exceed
    what it holds at full speed (`neuralEngineHold` = 2.8 GiB with the other loaded programs).
    Gemma 4 E4B (2 × 3.26 GiB) runs one program: start 3.7–5.5 s against 77–87 s, generation 13.4–16.4
    against 10–13 tokens/s, but a 16K prompt reads 4.4× slower (183 s against 42 s, M3).
  - Intel and `--draft` ≤ 1 always keep two.
- models.yaml can set `prefill`, `context`, `paged` and `tier` per model and machine memory
  ([distribution](distribution.md)).

## Splitting at the attention (paged)

The default for every model that allows it (`llm.Paged`: LFM3 and its VL, Audio and Omni checkpoints,
Gemma 4, Qwen3.5, PLaMo 3; Qwen3 only on Core AI), on Core ML, Core AI and Intel.

- Conversion (`hwnpu-dev convert -paged N`, `llm.Options.Paged`, `llm.PlanPaged`): parts end before each
  full attention layer N and output `attn_q_N` (after RoPE / norm), the step's `kv_k_N_new` and
  `kv_v_N_new` (and LFM3's gate `attn_gate_N`); the next part takes `attn_o_N` and computes the rest of
  layer N. Rings and convolution windows stay inside the parts. Outputs are bound by name across parts
  (`chain`). A weight budget can split a span further. The manifest has `paged` and `attention`.
- The attention program (`internal/llm/paged`, `out.attn.<ext>`) has no weights: one chunk's attention,
  returning the chunk-normalized output o, row max m and exp-sum l. Sums are a matmul with 1 (on Core ML,
  `reduce_sum` placement flips between ANE and CPU from compile to compile at Q=64).
- `decode.Session` runs the parts in turn. Between them it writes the step's keys and values into its own
  cache (before attending, so the step's tokens see themselves), calls the attention per used chunk, merges
  in float32 (`paged.Merge`, Flash-Decoding style), and writes `attn_o_N`. Only the chunks the step's tokens
  fall in (at most 2) need a mask; earlier chunks use a zero mask.
- Chunks are allocated as the position reaches them (`Attention.NewBuf`, device buffers on Core ML).
  `Limit` lowers the context, `Close` returns the chunks.
- Rebinding a chunk costs nothing, inputs are not copied (RSS ≈ KV bytes + 0.1 GiB), and there is no
  reload at the boundary between weight parts and attention programs (`npuprobe seam`, M3).
- Cost follows the length used, not the capacity. Context defaults to the model's longest
  (`max_position_embeddings`: 262144 for LFM3-1.7B and Qwen3.5, 131072 for Gemma 4, 128000 for LFM3-VL/
  Audio/Omni). The engine shrinks it (in whole chunks) when the full cache would exceed a quarter of
  physical memory (`fitMemory`, `memoryShare` = 4); the ready line shows the growth per token.
- KV sharing (`llm.Paged.PagedKV`): Gemma 4's layers that read another layer's keys (E2B: 19–34 read 14)
  output only `attn_q_N`; the session attends over the owner's cache. Host cache per token: LFM3-1.7B
  12 KiB, Gemma 4 E2B 6 KiB, E4B 16 KiB.
- Score scale: `paged.Shape.Scale` (0 = 1/√d; Gemma 4 uses 1, its q/k norms do the scaling).
- Qwen3 (`llm.Unpaged`, all 28 layers attend) is split only on Core AI by default (`pagedChunk`); on
  Core ML and Intel only with `--paged`. Unsplit, its context is 4096 (28 layers of K/V in one program fail
  past it). On M3 Core AI split + short chunks: 22-token prompt 19.1 → 36.1 tokens/s, 2149-token prompt
  18.0 → 21.6 and prompt 3.6 → 1.7 s. On Core ML the gain is small (M3 19.0 → 21.2) and M1 Core ML was
  slower split, so it stays opt-in there.
- Each attention layer costs a host round trip (about 0.4 ms fixed per call on Core AI); models with many
  attention layers pay it per layer. The whole-program tier (below) removes it for short conversations.

### Flags (hwnpu-chat)

| Flag | Default | Meaning |
|---|---|---|
| `--paged N` | 0: models.yaml, else 8192 (`paged.DefaultChunk`) where allowed; −1 never | Chunk length of the attention |
| `--shortest N` | 0: Core AI 1024, Core ML 4096, Intel none; −1 none | Shorter attention programs (N, 2N, … below `--paged`) |
| `--context N` | 0: models.yaml, else the model's longest when split, else 16320 (Qwen3 unsplit 4096) | Prompt + answer tokens |
| `--sparse` | true (`--no-sparse`) | Two-level attention past about 48K |
| `--tier` | models.yaml, else `auto` | Shorter programs for short conversations (below) |
| `--prefill` | models.yaml, else `auto` | Separate prompt program |
| `--draft` | 8 | Tokens per generating call |
| `--prompt-cache DIR` | `prompts` in the compile cache; `off` none | Prompt store on disk |

### Shortest chunk that covers the conversation

- A call's cost per layer is about 0.36 ms fixed plus 0.14 ms per 1K keys (M3, Core AI, LFM3-1.7B, 8
  tokens: 1024 0.50, 2048 0.63, 4096 0.93, 8192 1.49 ms). One chunk that covers n beats one long chunk and
  beats several short ones.
- `decode.Program.Smaller` holds attention programs for shorter chunks (each divides the next and `Attn`).
  The session keeps the cache in the shortest chunk that covers the conversation (`Session.fit`): before a
  step crosses a chunk, rows move at the same positions to the next length (contiguous rows per head, up
  to 4096 rows at a time); after `Reset`, `Rewind`, `Truncate` the next step moves back down. `ImportRows`
  fits first.
- `llm.AddSmaller` adds them to an existing export without reconverting (`Manifest.Smaller`,
  `OUT.attnC.<ext>`): weight parts do not depend on the chunk length. `chain` drops a shorter program that
  is not all on the NPU (`Chain.Chunks`); hwnpu-chat uses only chunks every program of a tier has
  (`commonSmaller`).
- Core ML puts small attentions on the CPU: grouped Q=1 up to chunk 8192, Q=8 up to 2048, Q=64 from 1024 on
  the ANE (M3 and M4 alike; set by Core ML on macOS 27, not the chip). Hence Core ML starts at 4096.
- Result (M3, LFM3-1.7B, 64 tokens): 150-token conversation Core AI 44.1 → 53.7 tokens/s, Core ML 42.4 →
  47.9; 2183 tokens Core AI 40.3 → 45.0 (prompt 1.6 → 1.1 s).

### Attention program forms

- `grouped` (`blocks.SDPAGrouped`, LFM3's `attn` default): one product per kv head over stacked query
  heads, no `tile` copy of K/V. `tile` remains selectable.
- `paged.Heads` (Core AI, kv heads ≥ 2; `llm.attentionForm`): queries of a kv head as (1, 1, rows, d),
  scores laid out (1, rows, 1, keys), scale on the queries, max and sum over the last axis with `reduce_sum`.
  5–26% faster per call (M1). With 1 kv head (Gemma 4 E2B, PLaMo 3 0.8B) it leaves the Neural Engine, so
  those stay `grouped`. `HWNPU_ATTENTION=grouped` forces the old form for comparison.
- `llm.RefreshAttention` rewrites the attention programs of an older export (`Manifest.AttnForm`); weight
  parts stay. `chat.Prepare` calls it after `AddMass`.
- The mass program (`paged.BuildMass`) stays `grouped`.

## Tiers: shorter programs while a conversation is short

All tiers are loaded at start (`chain.LoadAll`): loading the next one in the background while generating
made generation over 2× slower. Holding all tiers does not slow the active one. A session starts on the
shortest tier that holds the prompt (`tierFor`), and `room(n)` migrates up before a step that would not fit.
A tier switch costs about 15 ms once (Core ML).

| Tier kind | When | How it moves |
|---|---|---|
| Capacity tiers of a whole program (`--tier 4096,…`) | Models not split. `auto` = 4096 for lut4 models, none for lut6/lut8 (four lut6 programs overflow the compile cache and recompile every start) | `Session.Migrate` copies the full layers' rows at the same positions; rings are the same length |
| Whole program before the split one (`wholeTiers`) | Core AI, split models, `auto`. Cache = longest of 512/1024/2048/4096 with ≤ 4 MiB K/V per layer (LFM3-1.7B 2048, Qwen3-0.6B 1024). None when 2 × weights > 2.5 GiB (`wholeHold`), and none on an M1 (`wholeSlower`) | `migratePaged`: full rows move into the session's chunks (shortest covering); same-length rings, windows and states copy as they are |
| Short-context split programs (`ringTiers`) | Core AI, `llm.Windowed` models (PLaMo 3): a quarter of the window, if ≥ 256 (PLaMo 3: 512) | `migrateSplit`: host chunks stay; caches that held every position become rings (`pos mod length`). `Session.Remark` rebuilds marks taken before the move (nil after the ring wrapped) |

- A whole program copies its whole cache into Core AI arrays each call; cost tracks K/V bytes per layer and
  matches the split form near 4 MiB a layer (M3 Core AI tokens/s, whole 512/1024/2048/4096 vs split:
  Qwen3-0.6B 68/53/35/20 vs 34–40, LFM3-1.7B 64/62/59/52 vs 55). With the tier: Qwen3-0.6B 22-token
  conversation 37.4 → 52.5, LFM3-1.7B 54.9 → 59.2 (M3).
- On an M1 the whole program is slower than split programs with chunks of 1024 (LFM3-1.7B 33.6–37.0 vs
  39.9–43.6 tokens/s), so there are none.
- A whole tier is a separate `.aimodel` per cache length; its weights are held twice on the Neural Engine.
  Ring tiers are functions of the same `.aimodel` as the long-context programs (`llm.ConvertAI` puts parts
  starting at the same layer in one `.aimodel`), so weights are held once.
- PLaMo 3 Translate 2B (M1, 21 window layers of 2048 = 84 MiB of rings per call): 78-token paragraph 12.9 →
  18.9–19.0 tokens/s (step 77.3 → 52 ms, near the 49.7 ms weight floor); 438-token prompt with 469-token
  translation 21.7 → 26.0. Gemma 4 (window 512, 14 MiB of rings) gains nothing from a 256 tier, so none.
- models.yaml's `tier: none` turns the whole tier off on 8 GB machines (LFM3-1.7B, Qwen3-0.6B,
  Qwen3.5-0.8B).
- Two-level attention is attached to the last tier only. Prompt marks taken mid-read across a tier move
  are not stored.

## Reading prompts

- Prompt program Q: 64 for models in one piece (key axis must stay ≤ 16384) and on Intel (not measured
  there); 128 (`pagedPrefillSeq`, or the chunk if smaller) for split models on the Neural Engine, and for
  routed MoE models. Even at 64 rows a call is mostly reading weights; 256 is no faster than 128.
  LFM3-1.7B (M3) with Q=128: 4K 3.3–3.9 s, 16K 15.2–16.1 s, 64K 136–146 s (Q=64: 3.9–5.4, 19.3–25.2,
  160 s). Gemma 4 E2B 16K 31.2 → 23.5 s, LFM3-VL 16K 21.6 → 16.9 s.
- Last-row logits (`llm.Options.LastRow`, `-last-row`, `llm.Arch.LastRow`: LFM3, Gemma 4): an input
  `last` (1, 1, Seq) one-hot times the hidden rows (`blocks.Graph.LastRow`; `gather` would go to the CPU)
  picks one row before the final norm, so y is (1, 1, vocab). `decode` writes `last` and reads row 0. Not
  on Intel. `Speculate` never picks such a program (it needs logits per row). Names end `-q128-last`.
- KV-only prompt program (`llm.Options.KVOnly`, `-kv-only` with `-paged`): for models whose later layers
  write no cache (`llm.CacheWriters`: Gemma 4's shared layers), keep only the parts up to the last
  cache-writing layer; the last part ends before that layer's attention and outputs its K/V; no output
  layer (`decode.Program.NoLogits`, `writeUnattended`). The engine reads the prompt minus its last
  generating call's worth of tokens with it, and the rest with the generating program (`Engine.logitsQ`).
  Names end `-q128-kv`. Gemma 4 E2B (M1): prompt program 1.6 → 0.38 GiB, all on the Neural Engine, 4249
  tokens 4.2–4.6 → 2.4–2.7 s, generation after it 18.5–21.5 → 21.6–24.4 tokens/s (the second program no
  longer evicts a generating part). Marks without logits are never used at the prompt's end
  (`errNoLogits`).
- Host logit softcap (Gemma 4, `-logits host`): see [LLM models](llm-models.md). `decode` applies the
  softcap from the part metadata unless `SetRawLogits(true)`.

## Reusing prompts

### In memory (`chat.Engine`)

- Full attention rows need no copy to rewind: rows past the position are masked and later steps overwrite
  them. Rings and convolution windows are overwritten in place, so the engine takes a `Mark` (rings,
  windows, last logits; about 3–3.5 MB for LFM3) every 512 tokens while reading and at the end of each
  prompt, and `Rewind`s to the last mark inside the common prefix.
- Chat templates drop earlier `<think>` blocks, so the common prefix usually ends at the previous prompt's
  end.
- Up to 2 other sequences are set aside (`maxSaved`) with `Session.Save`, which copies only the rows up to
  the position (`ExportRows`), so parallel requests (OpenCode's title and summary requests) do not force
  re-reading the main one. A saved sequence remembers its tier.

### On disk: the prompt store (`internal/chat/store.go`, `--prompt-cache`)

- Full-attention rows are per position and do not depend on tier or chunk. `ExportRows(from, to)` packs
  them in position order (non-ring caches by name, rows per head, fp16); `ImportRows` puts them back into a
  session of any tier. `Resume(rows, mark)` checks that the mark's buffers match the session's rings and
  windows by name and size and that the rows end at the mark.
- Layout: a directory per model (`storeKey`: name, package names and times, weight sizes — reconverting
  makes a new key) with an entry directory per sequence: `meta.json` (tokens, rows, mark positions, last
  use), `rows.bin`, `mark-POS.bin` (gob of rings, windows, logits).
- `find` picks the entry with the longest common prefix and the last mark inside it. `put` appends to an
  entry that the sequence continues (written 1024 positions at a time), starts a new entry on a branch
  (copying the shared marks), or adds missing marks. Marks every 512 tokens (`markEvery`). Sequences under
  1024 tokens are not stored (`minStored`: they read in a second). 4 entries per model (`maxStored`), least
  recently used goes. Writes go to a temporary name and are renamed; `meta.json` is written last.
- Unreadable entries are dropped on use and the next best entry tried; `openStore` removes directories
  without `meta.json` older than 1 hour (`staleAfter`).
- `Engine.prefill` uses the store only when it reaches further than the in-memory reuse; it resumes on the
  shortest tier that holds the prompt. `Stats.Restored` counts restored tokens (part of `Reused` and of
  `usage.cached_tokens`); the server log says `(N tokens from the prompt store)`.
- Size (LFM3-1.7B): rows 12,288 B a token, a mark 3.45 MB, about 18.7 KiB a token with marks; a 7,857-token
  entry 145 MB.
- M3, LFM3-1.7B, OpenCode's first request (7,854 tokens): 14.6 s cold → 0.2 s after a restart (the store
  read itself 17–36 ms, the rest writing ANE buffers); 0.7 s with the request text changed (from mark
  7680). Writing adds under 0.5 s. Greedy output matches a cold read.
- hwnpu-agent's prompts (617 tokens of system and tools) are below `minStored` and are not stored.

## Two-level (sparse) attention (`decode.Sparse`, `--sparse`)

For split models past about 48K tokens (chunk 8192). Adapted from Latent Sparse Attention without distilled
projections.

- Summaries: per full attention layer and kv head, the mean key and mean value of each 8-position bucket
  before the window, in device buffers shaped like a chunk (1, kv_heads, 8192, head_dim); the host adds a
  bucket as it leaves the window (fp32 mean, stored fp16). They cost 1/8 more cache (LFM3 12 → 14 KiB a
  token; Gemma 4 E2B 6 → 7 KiB).
- Weighing: (1) attention over all summaries with the normal attention program (mask log 8 for counted
  buckets, −1e4 otherwise), which also gives the summaries' share of the output; (2) the mass program
  (`paged.BuildMass`) gives each summary's weight from q, the summary keys, the mask and each row's
  −log-sum-exp (`Merge.Norm`).
- Choosing: per kv head, the heaviest buckets that fit `Exact` chunks (2 = 16384 keys) together with the
  window and the call's own tokens, plus the first bucket. A selection algorithm on the host (no sort);
  unchanged buckets stay in place, only changes are copied.
- Exact chunks: each kv head's buckets at the same slots, then the window and the call (same positions for
  all heads); one mask shared by heads. Their summaries are subtracted from step (1) with `Merge.Sub`
  (negative results clamp to 0).
- Calls per step: summary chunks × 2 + 1 + `Exact`. Used only when fewer than reading everything (from about
  48K). Prompt calls use `Prefill` = 4 chunks (32768 keys), from about 57K. `Refresh` (calls between
  choices) defaults to 1.
- `DefaultSparse()` = Bucket 8, Sinks 4, Window 512, Exact 2, Prefill 4, Refresh 1. Summaries past a write
  position are dropped on `run`, `ImportRows`, `Reset` (`sparseTouch`). Short chunks (`Smaller`) never run
  it: it runs only on the longest chunk (`Attn`).
- Export: `convertPaged` writes `out.mass.<ext>`; older exports get it from `llm.AddMass` (seconds, called by
  `chat.Prepare`).
- Ready line: `past about 48K tokens, 16384 keys exactly and the rest as means of 8`.

Checked on CPU (LFM3-1.7B, PyTorch bf16, 128K, pass phrases at 10/50/90%): the default form raised the
next-token loss by 0.007 nats and kept all three; prompt in calls of 64 tokens matters (256 lost the 10%
phrase).

## Long-context numbers (M3 16 GB, Core AI, LFM3-1.7B lut6 grouped)

macOS 27.2.1, `hwnpu-chat serve` with the defaults (`--paged 8192`, two-level attention past about 48K,
drafts of 7), `--prompt-cache off --no-think`, temperature 0, a fresh server per run. Read = time to the
first token. Generate = tokens/s of the answer right after the prompt (100–256 tokens; it moves with how
many drafts are taken, so a range). The documents are this repository's English docs and Go source.

| Prompt (tokens) | Read, two-level | Generate, two-level | Read, all keys (`--no-sparse`) | Generate, all keys |
|---|---|---|---|---|
| short (20) | 0.1 s | 44 tok/s (23 ms/token) | — | — |
| 17,322 | 13.3 s | 34 tok/s (29 ms/token) | — | — |
| 65,133 | 95–97 s | 37–53 tok/s (19–27 ms/token) | 114 s | 27 tok/s (37 ms/token) |
| 130,522 | 264–267 s | 17–30 tok/s (34–61 ms/token) | 318 s | 13 tok/s (77 ms/token) |
| 256,732 | 679 s | 16 tok/s (62 ms/token) | not measured | not measured |

- One pass phrase at the start of repeated filler (the old test) is answered at 35K and 70K tokens.
- Three pass phrases at 10%, 50% and 90% of the documents, asked for together: 2 of 3 at 17K, 0–1 of 3 at
  65K and 130K, and the same with all keys read, so two-level attention is not the cause (see Open).
- Memory: the server's RSS is 0.59 GB at 17K, 1.74 GB at 65K, 1.90 GB at 130K, 3.89 GB at 257K (the
  cache grows with the conversation, 14 KiB a token).
- Other models, two-level, Core ML era (not measured again on Core AI): Gemma 4 E2B 64K 204 s /
  73.9 ms/token, 125K 521 s / 92.4 ms; LFM3-VL 120K 431 s / 101 ms; LFM3-Audio and Omni 120K about
  500–516 s / 115 ms. Gemma 4 E4B (one program) 125K read took 56 min.
- LFM3-1.7B is reliable to about 128K; at 256K even full attention missed the 10% and 50% phrases (the
  model's limit, checked on the CPU).

## Backends compared

| | Core AI | Core ML | Intel NPU |
|---|---|---|---|
| Split at attention | Yes, Qwen3 too | Yes (Qwen3 only with `--paged`) | Yes (parts and attention as OpenVINO IR, `out.partN.xml`, `out.attn.xml`) |
| Shortest chunk | 1024 | 4096 (shorter goes to CPU) | none |
| Attention form | `Heads` when kv heads ≥ 2 | `grouped` | `grouped` |
| Whole tier / ring tier | yes (not M1) / yes | no | no |
| Prompt Q, last row | 128, last row / KV-only | 128, last row / KV-only | 64, no last row |
| Unsplit model's generating program | 2 parts | 2 parts | one graph (no parts) |
| `--prefill auto` | budget 3.5 GiB (M1 2.8) | same | always two programs |

Level Zero and OpenVINO already allow rebinding a finished request; Core ML's `request.Bind` was fixed to
allow it too (`TestBindAfterSubmissionFinished`). AMD is outside this engine: IRON exports fix the context
with `--max-seq-len` and keep the cache in the device bank ([LLMs on AMD](amd-llm.md)).

## Pitfalls

- Attention programs must run with the step program's context, not the loading context: hwnpu-chat
  cancels the load context once ready, and a call under it returned `context canceled` while the
  prediction kept running (empty answers, `cannot bind while a submission is running`, SIGSEGV on close).
  `TestPagedOnNeuralEngine` covers it.
- Core ML: merging chunks inside one program falls to the CPU in several forms (chunks stacked on an axis
  and reduced, weighting each chunk by `sum_J / total`, scaling sums by 1/n first). Add the sums first and
  divide once at the end.
- Core ML at Q=64: `reduce_sum` placement is not stable between compiles; use a matmul with ones.
- MLState: using a value straight after `write_state` fails ANE compile (`error code: -14`); read it back
  first.
- The first few calls right after a very long prompt are slow (256K: 1.29 s a call, then 350 ms). Cause not
  checked.
- Two-level attention approximates peaked attention badly: a small Gemma 4 with norm weights near 1 (scores
  near 30) gives relative error 1.3–1.6 against full attention (0.068 with normal weights).
- `HWNPU_PROFILE=1` (`decode.WriteProfile`) splits a step into parts, host attention per layer, and the
  logits read; hwnpu-chat prints it after each answer with the Neural Engine's memory.

## Tests

`TestCachedMatchesFull`, `TestCachedChunksMatchFull`, `TestMigrateTiers`, `TestEngineTiers`,
`TestPagedMatchesFull`, `TestPagedShareRewindResume`, `TestEngineReusePaged`, `TestPagedLastRow`,
`TestPagedKVOnly`, `TestSmallerChunks*`, `TestShorterChunks(OnNeuralEngine)`, `TestMigrateToPaged(OnNeuralEngine)`,
`TestMigrateRings(OnNeuralEngine)`, `TestEngineRingTiers`, `TestWholeTiers`, `TestSparseMatchesDirect`,
`TestPagedSparse`, `TestMassMatchesSoftmax`, `TestEngineStore`, `TestResume`, `TestMergeMatchesFull`,
`TestRefreshAttention`. Each compares logits with a program without a cache on the CPU interpreter and
checks that deliberate breaks (row offset, missing ring copy, mask shift) fail.

## Not adopted

- MLState for the cache: same speed as host buffers (M3, 256–8192); only CPU use drops (about 0.15 core).
  Host buffers stay: they run on the CPU interpreter and move between tiers easily. The state support stays
  in `mlpackage` and the backend.
- int8 cache inputs: all on ANE and half the memory, but 10% slower; a memory measure only.
- GQA read as a (kv head, group) matrix without `tile` in a whole program: all ops on the CPU (`grouped`
  in its later form works).
- Loading the next tier in the background: generation over 2× slower than loading all up front.
- Q=4 generating program: 3% cheaper per call than Q=8, 10% fewer accepted tokens per call.
- Tiers as functions of one Core ML package: the blob writer does not deduplicate constants, so weights
  double and pass 1 GiB.
- Whole and split programs in one `.aimodel` (shared weights): Qwen3-0.6B became 60 functions, load
  2 min 28 s, some functions fell to the GPU and MPSGraph wrote 0.6 GB each to
  `$TMPDIR/com.apple.MetalPerformanceShadersGraph` until the disk filled.
- Two whole tiers (256, 1024) as functions of one `.aimodel`: the 1024 one ran at half speed.
- Sparse selection by single keys (top-k): even ideal top-4096 caught 53% of the weight at 128K (KL 0.58);
  32K keys needed to match two-level. Quest page bounds were worse; k-means in the RoPE-free dims was no
  better than fixed 8-token buckets; 16-token buckets raised the loss; 4096 exact keys lost the 10% phrase;
  keeping the first 4K exact lost it too.
- A Core AI chunk of 4096 as the default `--paged` (M1 Gemma 4 E2B): host attention 2.05 → 1.36 ms a layer
  but a fixed 0.67 ms a call remains and long contexts get 10% worse per 1K; short chunks (`--shortest`)
  cover the short case instead.
- Attention on the CPU: rejected for power (keep work on the NPU).
- One fused gate+up linear on Core AI (as on AMD): part time 3.31–3.33 vs 3.30–3.32 ms; bound by weight
  reading, and Core AI has no call boundary between the two linears.
- 4-bit weights (lut4) for Gemma 4 E4B to fit two programs: next-token agreement 64% vs 90% for lut6, only
  20% faster.
- A ring tier for Gemma 4 (window 512): 40.1 vs 40.4 ms a step.

## Open

- `Refresh` 4 or 16 is 14–32% faster per call at 128K and kept the pass phrases; default stays 1 until a
  loss check on the Neural Engine.
- The two-level prompt is 10% slower at 64K (its start at about 57K may be too low; not tried).
- Per-1K cost rises past 128K (1.14 → 1.43 ms per 1K) and the slow first calls after a long read: cause
  unknown (memory pressure or buffer count).
- Host time in `choose` (memmove of re-picked buckets) is about 13 ms a call at 128K (12%).
- Store: keep set-aside sequences (`Save`) in the store, thin old per-request end marks, int8 rows.
- KV-only prompt programs for models whose every layer writes a cache (drop LFM3's output layer, about
  0.19 GiB).
- Three pass phrases spread through a 65K–130K document (10/50/90%) are mostly missed on Core AI,
  with or without two-level attention, while one phrase at the start is found. The Core ML era passed a
  similar three-phrase test (docs in Japanese and English, asked in Japanese) at 128K; the two were not
  run on the same text, so a Core AI regression is not ruled out.
- Not measured: `Heads` form on M2/M3, ring tiers on M2/M3 and for PLaMo 3 0.8B / 8B, many server
  conversations moving between tiers, Intel NPU hardware for split models.
