# Checklist for a new model

[日本語](new-model-checklist.ja.md)

Problems found late while bringing in LFM3 (VL / Audio / Omni, d1), Gemma 4, Qwen3.5, PLaMo 3, PLaMo-Embedding, Whisper, the MoE models and the image models, and the methods that paid off, **in the order to check them**. Checking in this order from the start gives a form with little memory and good speed without redoing work. The steps and pass conditions for porting a structure to Go are in `.claude/skills/port-architecture` and are not repeated here.

## Policy: power over speed

hwnpu runs on the NPU alone to use less power. Weigh methods in this order:

- **When the total time is the same, keep the work on the NPU.** A form a little faster with work moved to the CPU (attention over a short context, say) is not taken. The GPU is never used
- **Read fewer weight bytes.** Reading weights is most of a token's energy; fewer bits, no layers a program does not need, and MoE help speed and power at once
- **Cut waste on the NPU.** Programs reloaded because they do not fit the NPU's memory, or one op on the CPU making the NPU and CPU take turns, cost power even at the same speed
- Measure power with `hwnpu-dev powerbench` (mJ a token; the Mac needs sudo) or `HWNPU_ENERGY` ([memory-and-power.md](memory-and-power.md)). When it cannot be measured, compare weight bytes read, `mem.NPUWired` and CPU use

| Stage | What it decides | Where |
|---|---|---|
| 1 | Whether it fits and is fast, on paper | Anywhere |
| 2 | Whether the structure is right (fp64, against HF, tokenizer, chat format) | CPU |
| 3 | Whether it fits on the Mac's Neural Engine and all of it runs there | Mac |
| 4 | The AMD / Intel form | Linux / Windows |
| 5 | **The quantization format and calibration** | Estimated on the CPU, checked on the NPU |
| 6 | Memory (resident, pinned, disk) | Each NPU |
| 7 | Speed (generation, prompts, startup) | Each NPU |
| 8 | Long contexts | Mac (Intel) |
| 9 | Checks in use, and the record | Each NPU |
| 10 | Values for combining it as a part | Each NPU |
| 11 | Running by name on every NPU (models.yaml, sources) | Each NPU |

Each "pass" is a guide. When a model misses it, write the reason in the model's topic file before going on.

## 1. Estimates on paper (before porting)

Estimate from config.json alone with [model-fit.md](../model-fit.md) (sections 2 and 4).

- **Time per token (ANE)** ≈ weight bytes ÷ 62–65 GB/s + (layers attending over the long capacity) × 0.22 ms × capacity / 1K + 3 ms
  - KV sharing does not cut the count of attentions (only memory). Window, short-convolution, Mamba2 and GDN layers do not depend on the capacity
- **Time per token (AMD)**: bound by the count of operator switches, not weight bytes. Each switch rewrites the array's configuration, about 85–100 µs (`tools/amd/elfconfigs` counts them). Now (Ryzen AI 9 365): Qwen3.5-0.8B q6 18.7 tok/s, LFM3-1.7B q4 about 17, Gemma 4 E2B q4 9.7, Qwen3.6-35B-A3B 8.9, gpt-oss-20b 20.2 ([amd-llm.md](amd-llm.md)). Estimate from the layer count and models already in
- **Weights**: lut4 0.5 B, lut6 0.75 B, lut8 1 B a weight. On an M3 16 GB keep resident memory under 3 GiB. On AMD, 4 bits (q4nx at 5 bits + attention at q8) pins 1/2–1/5 of bf16. **On an M1 16 GB keep the weights the Neural Engine holds for parts run in turn under about 2.9 GiB**: past it every call is about 12× slower (Qwen3.5-4B 2.97 GiB 1.1 tok/s against 1.99 GiB 10.9). The M2 and M4 ran 3.62 GiB at full speed ([apple-neural-engine.md](apple-neural-engine.md))
- **Vocabulary**: the output layer is vocabulary × hidden. A linear takes a 262144 axis on the ANE, but elementwise ops after it fall to the CPU: leave a monotonic one (soft cap) to the host
- **Structures that cannot be used**: learned sparse attention, variable shapes. MoE: AMD runs the routing on the device (or the host); the Mac runs experts through Core AI with the host picking them (`llm.Routed`), or on the CPU in MXFP4 (`llm.HostRouted`) ([moe.md](moe.md)). Count every layer's experts against the Mac's memory, and a routed layer at about 3× its active weights' time
- **Layers the checkpoint carries for drafting** (Qwen3.5's `mtp.`): keep them at conversion. As drafts they beat prompt lookup (Qwen3.5-0.8B, M1: 26.1 → 43.8 tok/s on code) ([llm-decoding.md](llm-decoding.md))
- **License and languages** (whether it goes into the catalog) ([distribution.md](distribution.md))

**Pass**: 50 ms or less at 1K, 100 ms or less at 16K, and a format whose resident memory fits 3 GiB on an M3.

## 2. A correct structure (CPU)

On top of `port-architecture`'s pass conditions (1e-4 against fp64, one layer at a time matching all at once, breaking it to see the test fail, `hwnpu-dev check` layer by layer):

| What to check | Tools | Lesson |
|---|---|---|
| The download is intact | `sha256sum` against HF's `x-linked-etag` | Large downloads had sparse bit flips (LFM3-Audio / Omni: about 10 places of 2–6 bytes). Random-weight checks and value ranges do not show them |
| Compare **with a BOS** | `tools/ref/ref_logits.py`, `hwnpu-dev check`, `ironlm -forced -bos` | Without a BOS LFM3's KL looks 4× worse and layer 0's activations become outliers. Use the tokens the chat format really puts first |
| Real weights | `hwnpu-dev check`; AMD `ironlm -forced` + `tools/ref/forced_logits.py` | With random weights the NPU and CPU disagree (flat logits, bf16 / bfp16 rounding spreads). Decide on real weights |
| Real weights **of every size** | `hwnpu-dev check` | Other sizes take other paths through the config (Qwen3.5-0.8B has as many value heads as key heads). Check every size that goes into the catalog |
| Tokenizer matches HF | `tools/ref/tokenizer_golden.py` | 0 mismatches over fixed and thousands of random texts. Make the tests fail when ties (PLaMo), NFC / combining marks (Qwen3.5) or byte fallback (Gemma 4) are handled the other way |
| Chat format matches jinja2 to the character | `tools/ref/template_golden.py` | Tool calls, thinking, dropping earlier thinking |
| Chat defaults per size | The template, the model card | Qwen3.5-0.8B / 2B answer without thinking by default, 4B thinks; made to think, the small ones ran to the budget. The catalog's `no_think` follows the default |
| Multimodal preprocessing | `resize_golden.py`, `mel_golden.py`, `whisper_mel_golden.py` | ±1 against PIL, 1e-4 against librosa |
| Encoder-decoders (speech) | `tools/ref/ref_whisper.py`, `hwnpu-dev check` (`REF.x.bin`) | Check the encoder, the cross keys / values and the decoder apart. Over 30 s follow HF's windowing (next window from the timestamps); cutting every 30 s drops text |
| Embedding models | `embed_golden.py` | Cosine 0.997 or more against HF |
| Image generation | `tools/ref/compare_image.py`, the model's `ref_*.py` | Latent cosine against diffusers' fp32, VAE PSNR. A model too big for a 16 GB Mac in fp32: read the repeated layers' weights just before each layer (`tools/ref/stream_weights.py`), or build diffusers' blocks on the meta device and assign a block's weights at a time (`ref_chroma_lean.py`: a 512² DiT call in 98 s on an M1; `--stats` per-block largest values, `--dump` block inputs) |
| **How the reference masks** | `tools/ref/ref_chroma.py --mask keys` | diffusers' ChromaPipeline adds its 0/1 text mask to the scores: pads are not hidden though Chroma was trained with them hidden. The NPU's correct masking looked wrong (0.9886) until the reference masked keys (0.99985). Patch only the part meant (the same patch also broke T5's bias mask). Read the reference's masks before blaming the NPU |
| Untyped constants in Go | Read the scales | `T5FFScale/T5Scale` of two untyped ints is `1/64 = 0`. Write `float32(a)/b` |

## 3. Does it fit on the Mac's Neural Engine?

Nothing here can be settled without a Mac. If you have none, write a handoff (`docs/handoff-*.md`). The Mac commands run Core AI only (macOS 27): the model is converted to MIL (`.mlpackage`), lowered to `.aimodel` ([apple-coreai.md](apple-coreai.md)); the Core ML package is what `--cpu` and the dev tools read ([apple-coreml.md](apple-coreml.md)).

### 3.1 Limits

| Limit | What to do |
|---|---|
| 1 GiB a package (0.9 GiB a part by default) | Split by layers into separate packages (multifunction does not get past it) |
| 16384 per dimension | Attention in chunks (`--paged`), the soft cap folded to (16, 16384) |
| Core ML's compile cache about 4 GiB (LRU), about 3.5 GiB in effect | Only for Core ML: past it every start compiles again (Qwen3.5-2B's 3.56 GiB, 4–5 min). Core AI keeps programs until the disk runs low. Measure starts three times in new processes, alone |
| Tables only per 16 rows | An output layer whose rows are not a multiple of 16 (Whisper's 51866) is padded with zero rows and sliced back (`blocks.LinearPadded`); unpadded, lut conversion fails |
| `gather` runs on the CPU | Look up embedding rows (PLE too) on the host |
| Attention scores under 64 MiB a tensor | Past it 2–3× slower: chunk the queries (`maxScoreBytes`). Not always: Whisper's encoder (90 MB) was no faster in chunks. Weigh compile time too when programs are past the cache. Measure before splitting |
| 112 Neural Engine programs a process | The 113th goes to the GPU; the Core AI device warns near the limit ([apple-coreai.md](apple-coreai.md)) |

### 3.2 Does all of it run on the NPU?

Check each part with `hwnpu-dev info -model PART.mlpackage` (Core ML's placement) and Core AI's own placement (3.5). What fell off before:

- Attention that copies K/V with `tile`, or queries as a matrix → `SDPAGrouped`. The order alone changes placement
- A `transpose` / `reshape` in the last part → take the output token-first. Elementwise ops after the output layer → the host's sampling on the top values (`llm.LogitSoftcap`)
- Parts of one row or tiny programs → 8 rows, or merge into the neighbouring group. A small program (about 45 ops) goes to the CPU whatever it holds
- `reduce_sum` → a matrix multiply (chunked attention)
- Any op on an axis wider than 16384, even an identity naming y: take logits before the softmax, fold the bias into the linear, split a wide output linear into stacked parts (`internal/onnx`)
- Depthwise convolution over more than 4096 channels: split the channels (`maxDepthwise`)
- `int8-g32` (int8 with groups) puts the linear on the CPU: use per-row int8
- A product of two 8 × 8 matrices puts a program off the NE ((8 × 8)·(8 × 128) and k·kᵀ are fine)
- A cross-attention whose softmax sum is divided after P·V (`Deferred`) with a query count not a multiple of 8 falls to the GPU on M2 and later: pad queries to 8 (`milpart.DeferredRows`)

### 3.3 Does fp16 overflow?

Check each layer's relative RMS error is under 0.01 with `hwnpu-dev check -f16`. Layers that pass alone can drift as a whole. Look also at:

- The whole `-f16` run and the cosine **per position**, the prompt's or speech's positions apart from the padding's
- Where massive activations (attention sinks) arise: each layer's largest values in HF's hidden states (`REF.hidden.bin`)
- **Before hunting a cause, compare with HF in fp16.** If it drifts as much, the port is fine (Whisper: 0.9978 both)
- Programs of the first N layers (`hwnpu-dev convert -layers N`) on the ANE and Core ML's CPU against HF show where drift starts

What helped:

- Scale norm inputs by 1/64 (Gemma 4: KL 3.17 → 0.0004; PLaMo 3). Not small inputs: squares under fp16's range give NaN (Qwen3-0.6B's embeddings ≈ 0.1). Always look at layer 0 with `-f16`
- Shrink the last layer: head-scale (PLaMo-Embedding), `-post-scale` (LFM3)
- Residuals at 1/32 and division by each row's maximum before a norm (Anima; a fixed scale made subnormals)
- Choose the residual's scale from the reference's largest value per layer, not by trying (the fp32 reference prints them)
- Fill padding rows with the last token (Gemma 4's eps becomes 0 in fp16)
- The ANE's own `sigmoid`, `tanh`, `softplus`, `silu` and `gelu` are inaccurate: write them with exp (`blocks.SiLU`, `onnx.Program.ExpSigmoid`, GELU as `x / (1 + exp(-1.5958 (x + 0.044715 x³)))`). `hwnpu-dev check` on the CPU does not show this: compare the per-position KL on the ANE with Core ML's CPU (`HWNPU_COREML_UNITS=cpu`), and bisect values on the ANE against the CPU interpreter (`TestBisect`)
- A one-row computation every later layer takes (a diffusion timestep embedding) drifted 1 % on the ANE: compute it on the host in fp32
- Linear attention's sums pass fp16 though residuals do not: divide each head's query rows by their maximum, scale keys down before the sums, floor the denominator
- One row of a linear's output near fp16's limit became NaN on the ANE only: when a norm follows, scale the linear's weight and bias by 1/64. Find such parts by moving one part at a time to the CPU
- **Per-row int8 weights overflow on large outputs**: the NE sums int8 codes and applies the row's scale after; an output of 13000 at an attention sink came out NaN (Core ML's CPU right). lut8 expands to fp16 before the product and is right. Bisect layer by layer counting non-finite values
- **A scale folded into a weight makes products under fp16's normal range** (small channels lost). Multiply the projection's output instead; Core ML folds a constant scale next to a linear into its weight anyway
- **A norm of rows with outlier channels**: take the mean square of the max-divided copy, but multiply x itself by rsqrt / max (`chroma/mil.t5Norm`)
- **The NE's rules** (`aicheck rules`, measured on an M2; Core AI's CPU has none of them, see [apple-coreai.md](apple-coreai.md)):
  - a linear's / convolution's sum **and partial sums** must stay under 2^15 (reduce_sum keeps the full range)
  - a slice of the last axis starting at 1..14 mod 32 turns values over 4094 into Inf: run `aicheck lint` (or `HWNPU_AILOWER_NOTES=1`)
  - products under fp16's normal range are flushed inside a matmul, not elementwise
  - `ailower` already rewrites softplus and a matmul of inner dimension 1

### 3.4 Pass

- Every part is all on the NPU
- The second and **third** starts take a few seconds (the second alone can miss two programs that evict each other). Record each part's first compile time
- A model whose programs are past what the NE holds must keep what runs every step in the fewest bytes, run other parts once per run, and batch prompts (`--batch`). Record the load times
- On an M1, `HWNPU_PROFILE=1` part times hold steady call to call. When parts run in turn hold more than about 2.9 GiB, the same part varies and calls are 10× slower (a reload). Parts run in the wrong order hit it too (each conditioning through every DiT group). Not every model past 2.9 GiB reloads: test a reload hypothesis by changing the weights' size before acting on it
- **Keep the disk free**: macOS purges compile caches when free space is low (seen at 11–22 GB free; keep 25 GB free when timing a first run), and the next run compiles again (minutes per part). Test binaries get a cache of their own name: build them as the command's name in a directory (`go test -c -o DIR/hwnpu-image`) ([apple-coreai.md](apple-coreai.md))
- **`mem.NPUWired` is the whole machine's.** A measured process takes the NE lock (`llm.LockNPU`) so the value before it starts is near 0; after the lock, wait for it to settle (programs of an ended process stay 0.1–7 s). Check nothing else uses the NE before taking a failure as the program's: another process compiling made programs fail ("Unable to compute the prediction", `Request cancelled`)
- Keep the CPU free while programs compile (a CPU reference beside it timed out a placement check; `planTimeout` is 30 min)
- **A graph change that keeps the program's name raises `llm.Arch.Revision`** (`-rN` in the name; image models' export version), or old programs are reused
- Record the KL and argmax of `hwnpu-dev check -package` per format (section 5)

### 3.5 Core AI (macOS 27)

A model whose parts pass 3.1–3.4 can still fail on Core AI. Check on the M1 too (some parts pass the M3's compiler and fail the M1's). Details in [apple-coreai.md](apple-coreai.md).

- **Every op needs a lowering.** `aicheck lower OUT PKG` stops at the first missing one. Write a new one as coreai-torch does: convert a small torch module with coreai-torch, print the program, read the bytecode with `tools/coreai/bcdump`
- **Compare every part with the CPU interpreter on Core AI and on Core ML**: `aicheck run -cpu PKG...` and `aicheck run -cpu -coreml PKG...`. A part all on the NE can be wrong on Core AI alone (PP-OCR's detector 0.05 against 0.9999)
- **Find the op with `HWNPU_AILOWER_CUT=N aicheck run -cpu PKG`** (`AICHECK_WORST=1`). Look at what the error is, not only where it is largest (fit a line between input and output: the NE multiplied where it should divide). Test against the real part, not a small copy
- **Tell lowering errors from the NE's with `HWNPU_COREAI_UNITS=cpu`.** Traps: Core AI's CPU overflows a `reduce_mean` whose sum passes fp16; the CPU interpreter implements only what it implements (`elu` alpha 1 alone); `runner` computes in fp32 unless `HWNPU_CPU_F16=1`, aicheck with fp16 rounding: decide the reference's precision first
- **NE pitfalls**: `divide_v1` by a constant is miscomputed (ailower writes a product by the reciprocal); `(v + c) - c` is folded to `v` even with c an input; a width convolution with kernel over 1 and 512+ channels whose input is computed comes out wrong (run it along the height); dilated convolutions are wrong at some lengths (lay frames out as rows of the dilation); an explicit `pad` before a convolution went wrong; a (1×1)·(1×N) matmul is wrong on the M3 (use mul)
- **Several functions in one `.aimodel` compile as one**: one function the compiler fails takes all of them off the NE. Look at each size's part alone first, and try small rewrites in `aicheck probe` (a `tile` channel repeat → a 1 × 1 convolution of 0/1 weights). Adjacent lut tables must not be identical (`coreai-build` left a part off the NE)
- **Compare starts with both caches warm** (three starts of each backend in new processes, right after a run of each)
- **Read `HWNPU_PROFILE` by stages** (programs of the same tokens per call are summed apart). Before acting on a part's time, check parts read at about 62 GB/s
- **A recurrent state touched per row costs as much as weights**: count the bytes activations and states move too (Qwen3.5's chunked GDN)
- **A draft head over part of the vocabulary** (`llm.Options.HeadRows`): decide by drafts taken, not coverage
- `aicheck run` / `probe` take the NE lock: two processes on the NE at once have frozen a Mac

### 3.6 Mixture of experts

Details in [moe.md](moe.md).

- **Does the experts' form fit `internal/llm/experts`?** SwiGLU without biases, sigmoid or softmax scores, a choice bias, top k normalized and scaled. Shared experts, biases and clamps (gpt-oss) need the program extended
- **Time the layer's shape first**: `aicheck moe -model NAME -layers L -copy` and `aicheck moepack -two`. `aicheck probe gnd16_const gnd16_pts_w1` tells whether the chip takes the scales' gather (M2 and M3 do, M1 does not)
- **Separate the experts' error from the port's**: the rest in transformers and the MoE blocks through `aicheck moeserve` (`tools/ref/lfm2moe_ane.py`), then `hwnpu-dev generate -score` with and without the cache
- **In the architecture**: implement `llm.Routed`; a part ends where a MoE layer begins (`moe_x_N`, `moe_logits_N` in fp16), the next starts adding `moe_o_N`; cut by `llm.PlanRouted`
- **Count rows**: drafts are off and one tier runs for a MoE model; a prompt call runs each chosen expert once over its rows (`experts.BuildRows`), 128 tokens a call
- **Count which experts run** over many kinds of text (`tools/moeusage`: `collect.sh MODEL OUT`, `analyze.py OUT`). Experts every kind runs go to models.yaml's `experts_core`; check the core pays at the places you ship (Qwen3.6 on AMD has none)
- **Host-run experts (`llm.HostRouted`)**: measure reading and computing apart (`HWNPU_MOE_DEBUG=1`). Measure the SSD first. Do not compare settings on one repeated question (the page cache warms): use a list of questions of different fields and languages (`--prompt-cache off`). Look at a CPU profile (`mincore` and the kernel took 66 %). Time with the NE's pause between layers (`HWNPU_QWEN36_GAP`) and a long prompt (20+ steps). Check the free disk before deriving an MXFP4 checkpoint (`hwnpu-dev mxfp4`)
- **Compare generated text from stderr**: `hwnpu-chat` writes thinking there, so a short run's stdout is empty. Keep measuring binaries' names (Core AI's cache is per process name, 1.8 GB a model)
- **On AMD, experts that do not fit go in places** (`iron_export_lm.py --expert-places P`): check with `HWNPU_MOE_DEBUG=check` and against an export with every expert placed; give a 16 GB machine a `variants` entry; run by the model's name (only the name gives the core and counts)
- **Check a model that is not MoE stays as it was** (same text, same speed, not converted again)

## 4. The AMD (IRON) and Intel forms

### 4.1 AMD

Details in [amd.md](amd.md), [amd-llm.md](amd-llm.md) and [image-amd.md](image-amd.md).

**Correctness**

| What to check | Lesson |
|---|---|
| Does IRON have the application? If not, write it from IRON's blocks | LFM3 and PLaMo 3 were |
| Against HF: 400 rows of `ironlm -forced` with `tools/ref/forced_logits.py` | bf16 lands at KL 0.001–0.014, top-1 96–99 %. Long prompts too |
| **Prompts at the chunk's own size** (`iron_qwen35_check.py ref --tokens N --stream`) | A 32-token check takes the one-pass path only; Qwen3.6 broke when a chunk ran in several passes |
| Each operator against fp32 with real activations | IRON's reference copies its kernels (the tanh GELU's table, none of MEDIUM's bfp16): feed one layer's HF activations through one operator |
| IRON's MHA keeps scores in bf16 | Large scores with small differences lose them: subtract the first key from every key. Check each attention's largest score |
| A constant "ones" channel carrying biases | Must stay exactly constant through activations and rounding (16: silu(16) = 16) |
| MEDIUM rounds A and B to bfp16; AIE2P's vtanh returns x for \|x\| ≤ 0.5 | Where layers cancel, measure the NPU's rounding with an identity GEMM and simulate it before porting |
| A max pool's pads in bf16 | Pad just below the input's least value, not -30000 |
| A graph whose second run hangs | Check every graph with `irongraph -runs 3`; an empty command in its own context before each run (`GraphOptions.Barrier`) |
| What a host step reads can be overwritten between commands | Copy what the host reads into a state; check a host step's export on the whole model |
| IRON's compile cache keys on an operator's parameters | Give an operator whose build you change a form parameter |
| Read the kernel headers IRON compiles, not an mlir-aie checkout | They differ |
| The cores' rounding to nearest even | Floor rounding pulls sums down |
| Images and audio in the final export form | Send an image and a clip through `hwnpu-chat` after any export change |
| Layers with state copied at chunk boundaries | For rollback marks and the prompt store; a ring cache: check rollback and export / import bit for bit (`TestKVRingRewindOnDevice`) |

**Limits**

| Limit | What to do |
|---|---|
| A command under 2 s | Split at export (`SCRATCHPAD_WORDS`); else `ERT_CMD_STATE_TIMEOUT` |
| Control code within the device heap (64 MiB) | Dense models cut at `SCRATCHPAD_WORDS` 24; verify and decode ELFs both stay or one is made per call |
| At most 3 transfers queued on a shim channel | 16 hung the command; two hangs need a reboot |
| A DMA run is at most 1023 words; a descriptor repeats at most 64 | Lay tables out in the order operators read them |
| Program memory 16 KB a core | One noinline function; check after adding kernels |
| Quantized projections: multiples of 64 out, 512 in | Pad fused projections per column (`iron_gdn.column_widths`) |
| GEMV's 64-row blocks divide over the 8 columns | Else it runs on fewer columns (97 blocks: 1 column). Check `num_aie_columns`; pad (`COLUMN_ALIGN` 64) |
| MHA's head_dim a multiple of 64 | Pad 96 to 128 |
| A hidden size not whole blocks on 8 columns (2880) | Carry the residual padded with zeros, rescale the norm's weight and eps |
| Per-call row bounds | Every per-call word must be affine in one per-call value: fixed chunks and one `start`; `ProjectionGEMM` takes multiples of 256 rows |
| HIGHEST GEMMs need N a multiple of 64; RMSNorm weight ≤ 4096; DMA loop ≤ 64 | HIGH / MEDIUM, fold the weight into the next linear, commands of 262144 rows |
| Do not poll `hwnpu status` while a model loads | Followed by a mailbox timeout and an SMU needing a reboot; read `/proc/PID/fdinfo` |
| Big exports | Qwen3.6: 30 min, peak RSS 54 GB. /tmp is RAM: export under `~/models`; run long ones with `systemd-run --user` |

**Speed** (generation is bound by operator switches; prompts too)

| Method | Lesson |
|---|---|
| Count the decode step's operators and switches first | `model.record().steps`, `HWNPU_EXPORT_STEPS=1` and `elfconfigs`; time each with `--command-per-step` and `HWNPU_IRON_TIMING` (a command costs about 0.1 ms of its own). Read times only from that export |
| **Fewer operators in the decode step** | Bias and residual in the GEMV (`into_linear`), gate and up as one table (`QGEMVGlu`), sandwich norms as one (`SandwichAdd`), one RoPE for q and k, short convolution as one (`ShortConvStep`), state slots read in place (`slots=`), QGEMV on one core a column |
| **Fewer operators in a prompt chunk** | One short-convolution operator (`ShortConvChunk`, `GdnConvChunk`). Merging small configurations (RMSNorm, add) does not pay |
| A decode step as one resident array (`GptOssStep`) | Removes reloads (gpt-oss 79 → 49 ms); costs a core program within 16 KB and every operand declared every step. An `array=True` parameter that differs by layer makes an array a layer: read the commands' times after any export change |
| Prompt MHA on several pipelines (`--mha-pipelines`, default 8) | 2–5× faster chunks; check `num_pipelines` in kernel names |
| Chunk size | 1024 for attention models, 512 for GDN models, never under 256 |
| Hoist work an operator repeats; host steps split over CPUs by token | GDN's SiLU once: 30 → 10 ms |
| KV-shared layers on the prompt's last row; window layers read their window (`chunkBarrier` needed) | Gemma 4 E2B prompts 2.3×; −12 % |
| Tables on the host (PLE, embeddings); towers loaded when used; `-idle` | E2B pinned 9.08 → 4.60 GB |
| Attention sinks | Add the sink's term to MHA's running sum before the last division; test sinks of 6 and more |
| Exponentials | Count a core's exponentials and their width; a quartic on two limbs, not a bf16 cubic |
| Time a kernel stopped part way before optimizing a phase | The products were free; the exponentials and sums were not |
| Checking drafts in one call (`--verify 3`) | Every op needs a form of a few rows (`QGEMVRows` q8 / q6). Pass: `ironlm -verify` bit for bit the decode step |
| Time the host's share (pprof) | A loop over 201088 logits took 1.33 ms |
| One weight bank for graphs of the same blocks (`share_banks`) | Compare bank images before storing a second copy |
| Large image parts | Text encoders and DiTs in groups in one bank (`SwapVariants`); check a group switch (`irongraph -swap -variant 1`) |
| No other user of `/dev/accel` while measuring | Another user made commands up to 50× slower |

The export's name: `~/models/NAME.amd-npu2-q4` (or `NAME.amd-npu2`); on a machine with only AMD's NPU, hwnpu-chat picks it for `-m NAME`.

### 4.2 Intel

To IR with `ovlower`, int8 by default (int4 matched fp16's next token only 22/52). Only accuracy on OpenVINO's CPU can be checked so far ([intel.md](intel.md)).

## 5. Quantization: format and calibration

**The stage that takes the most work and pays the most.** 4-bit loss differs tenfold between models (AMD q4 KL: PLaMo 3 translations 0.009, LFM3 0.036, E4B 0.049, E2B 0.097). Details and per-model choices in [quantization.md](quantization.md).

### 5.1 Measure the same way

- **KL(reference ‖ quantized)**: mean, median, 95th percentile and top-1 over 400 positions (256 read as a prompt, then one at a time), **with a BOS**
- Five texts in `~/models/calib`: the README (frozen in `testdata/texts` with `GPL-3` and `model.go`), GPL-3 (**never used for calibration**, to catch overfitting), wiki40b-ja (evaluation part), wikitext-2 (test part), Go code
- The reference is HF fp32 on the CPU or bf16 on the same NPU. The quantized KL on the NPU should match the CPU estimate; if not, suspect the NPU's arithmetic
- Apple: `hwnpu-dev generate -package P.parts.json -score REF,...` on a program without a cache (`hwnpu-dev convert -seq 128`). The first tokens weigh much (KL up to 8–13): compare settings on the same texts only
- Image generation: the velocity's relative RMS along the reference trajectory (each reference DiT input through the NPU, `chroma.TestCallsOnDevice`), and the final latents of free runs. Errors that grow with the steps show only along the trajectory

### 5.2 Sweep bits per part on the CPU

Switch each part (attention projections, MLP, output, convolutions, towers) between 4 bits, 8 bits and bf16 (`q4_gptq.py --method rtn` with `--q8`, `--head`, `--fmt-*`). So far:

- **Attention projections at 8 bits**: about 10 % of the weights but most of 4-bit error (E2B 0.097 → 0.229)
- MLP and output matrix at 4 bits; short convolutions at 8 bits
- Sandwich-norm models (Gemma 4, PLaMo 3) are weak at 4 bits: lut6 or more on the ANE
- **A larger size can take fewer bits** (PLaMo 3 Translate 8B holds at lut4 where the 2B broke): measure each size (`Formatter`)
- lut8 for multimodal text parts, TTS, d1's judgments, embeddings, and the LFM3 family on the M3
- **A small model's (≤ 2B) MLP can break at 4 bits**: Qwen3.5-0.8B README 0.055 but GPL 0.34; AMD q6 gives 0.044. Always look at the GPL
- **A checkpoint already in 4 bits (MXFP4) stays in its own format**: re-rounding to q4nx took KL 2–3× up
- DiTs: try per-row int8 and lut8 and look at the pictures (int8 overflows on some); a distilled few-step DiT is weak at 4 bits; a large DiT can hold at lut4 with k-means-minmax tables; MLP at 6 bits with attention at 8; a pass run once a prompt can take a format up for free
- A part run once a picture and closed (text encoder) need not save bytes like the DiT, but keep only the weights that need fp16 at fp16 (`--text-format-for wo=fp16`)

### 5.3 Calibration methods, in order

The format stays, so speed and memory stay.

| Order | Method | Tools | Notes |
|---|---|---|---|
| 1 | Rounding (RTN); q4nx picks the least error of 9 candidates | `iron_q4.quantize_fmt`, `palette.QuantizeBits` | The baseline; least weight error is not least output error |
| 1b | **k-means tables keeping each block's min and max** (lut4, no data) | `-quant kmeans-minmax` | Default for lut4. Plain k-means clips outliers |
| 2 | **GPTQ** | `tools/amd/q4_gptq.py` → AMD `--q4-codes`; Apple `hwnpu-dev convert -codes`, `hwnpu-chat --codes`, models.yaml `codes:` | Damp 0.1 × mean diag(H) (0.01 overfits). Calibrate on general text (wikitext-2 + wiki40b-ja, 64 × 512); leave the BOS out of H; quantize earlier layers first; `--q8 gptq` to calibrate every projection |
| 3 | Fitting Apple's lut tables: k-means weighted by E[x²] | `q4_gptq.py --lut-fit wkmeans` | Start from uniform tables |
| 4 | Rotation before lut (`llm.Rotate`) | `tools/quant/rotkl` | Halves KL for Qwen3.5 (NE lut6 0.054 → 0.029); default for its lut6 exports |
| — | One format up (AMD q6) | `--mlp q6` | When calibration does not fix it |

- **Adopt** only if the GPL does not get worse than rounding. A multimodal model's text part is often retrained: calibrate each, do not reuse the base's codes
- **For a DiT, judge GPTQ on free runs from other seeds**: it cut the teacher-forced error yet made free runs worse (not adopted for DiTs)
- **Weights the calibration leaves out** are still rounded: look for small projections outside the lists and keep them a step up (Gemma 4's `per_layer`: `-format-for per_layer=lut8`)
- A lean loader for a 16 GB Mac must first match transformers' logits (KL ≤ 0.003)
- **Codes match** by a hash of shape and bf16 values (`palette.WeightKey`): check the export log's "N weights took the chosen codes". Codes are also kept in the sources (`converted/MODEL/any/codes-HASH`), else nobody can convert again when the export key changes
- Cost: 25–60 min for 2B models in fp32, 2.4 h for E4B (two at once ran out of memory); a 16 GB Mac calibrates E2B lean in 87 min. Use `systemd-run --user` on Linux

### 5.4 Choosing the format

| Guide | Decision |
|---|---|
| KL under 0.05 (against HF on the CPU) | 4 bits can be the default |
| KL 0.05–0.2 | Decide by the answers (section 9) |
| KL 1 or more | Not usable |
| `hwnpu-dev formats`: 90 %+ of the reference's next tokens and as many short questions right | Take the smaller format |
| 85–90 %, or new mistakes in arithmetic or facts | Keep the next larger format |
| Under 80 %, or repeating answers | Not usable |
| Audio / image encoders | Decide by WER / CER (Whisper stays lut8) |

- Speed follows weight bytes (ANE lut4 / lut6 / lut8 a call: 1.81 / 2.65 / 3.49 ms); probe's read speed per format is not the model's: measure real parts with `HWNPU_PROFILE=1`
- fp16 is the reference; when it does not fit a part, use lut8 (`-ref lut8`) and check lut8 against HF
- Compare by agreement with the reference, not perplexity (quantized can be lower on short text)
- The format may differ by size: decide per checkpoint (`Formatter`)

## 6. Memory

Details in [memory-and-power.md](memory-and-power.md).

| What to measure | Tools | Lesson |
|---|---|---|
| AMD pinned memory | `/proc/meminfo` `Unevictable`, banks in `manifest.json` | Mostly weights and work areas; KV under 5 %. GDN layers keep a frame a chunk boundary, so pinned memory grows with the context |
| The Mac's footprint and NE memory | `hwnpu-dev probe split`, `HWNPU_PROFILE=1`, `mem.NPUWired` | NE weights are kernel wired memory. They arrive at the first call, not at load: measure after one call |
| Peaks over a run | `go run ./tools/memwatch [-csv FILE] CMD...` (every 0.1 s) | mlock'd file pages are not in the footprint; `footprint PID`, `vmmap -summary PID` for the breakdown |
| Conversation marks | | Copy only ring rows in use (`ringRows`) |
| After a conversion | | `debug.FreeOSMemory` |
| KV and context | | Past 1/4 of physical memory, shrink by whole chunks |
| Disk | `cache.Clone`, `LinkFrom` | Share identical files (nothing shared on ext4); hard-link parts that do not depend on size from another export |
| Host memory between VAE stages | `debug.FreeOSMemory` | Large stage I/O in fp32 stays in Go's heap otherwise; drop host weights needed only before the steps |
| AMD VAE prefetch | `anima.VAE.PrefetchLimit` | Prefetch only when two stages fit 3 GiB |

**Pass**: enough numbers for models.yaml's `memory` and a row in [performance.md](performance.md).

## 7. Speed

| What to measure | Tools |
|---|---|
| One call | `TestCallTimes` (`HWNPU_CALL_PKGS`) |
| Generation, prompts, startup (second start) | `hwnpu-dev generate`, `hwnpu-chat`'s log |
| Comparing formats | `hwnpu-dev formats` |
| AMD GEMV, power | `tools/amd/gemvbench`, `hwnpu-dev powerbench` |
| One run's energy by phase | `HWNPU_ENERGY=10 hwnpu-image ...` (chat, audio too) |
| A step broken down | `HWNPU_PROFILE=1` |

Bring these in from the start ([llm-engine.md](llm-engine.md), [llm-decoding.md](llm-decoding.md)):

- Two programs (generation and prompt) on one cache, prompt reuse, the prompt store
- **The prompt program's output layer on the last row only** (`LastRow`)
- Drafts (prompt lookup; MTP with the GDN state rebuilt on the host from `state-rows`; AMD `--verify`), the think budget, repetition detection. Measure 1, 2 and 3 MTP drafts on code and prose (the third lost on Japanese)
- A prompt program that stops at the last layer writing a cache (`-kv-only`, `llm.CacheWriters`)
- Monotonic ops after the output layer on the host (`llm.LogitSoftcap`)

Compare only pairs measured back to back under the same conditions (swap on the M3, the power state on AMD vary numbers up to 1.4×). Look at `top` first: on a Mac `mediaanalysisd` and Apple Intelligence take the NE (every form fell to 0.7 tok/s).

### 7.1 Image generation

An image's time is the DiT's passes: steps × passes a step (2 with CFG) × one pass. Cut passes first, then a pass's time. Estimate a pass's FLOPs against the NPU to see the room left. Check every change from the same noise against the CPU fp32 reference and on a grid of seeds. Details and per-model state in [image-models.md](image-models.md), [image-apple.md](image-apple.md), [image-amd.md](image-amd.md).

Bring a new image model to each of these (or write why it does not apply):

- CFG's two passes back to back in each weight group (`velocity`), the prompt-independent embedding once
- The empty negative prompt as a constant computed at export
- A second-order solver (DPM-Solver++ 2M), judged against many Euler steps
- CFG at the noisier steps only (`--cfg-sigmas 0.6,1`)
- Keys and values that do not depend on the step once a prompt (cross-attention; text rows modulated at t = 0); modulations once a sigma, closed before the steps
- A fixed instruction prefix held as constants (only when longer than a chunk)
- The negative read in the same text call (`EncodeAll`)
- Batches of pictures (`--batch`): load a part once, run N
- Core AI: the attention scale in the q norm, magnitudes by `abs`, attention a head at a time with scores (1, queries, 1, keys), keys in power-of-two parts (`milpart.KeyParts`); load the next program ahead (`runner.InTurn`); swap DiT groups
- AMD: image attention in 64-key blocks with QKᵀ in bfp16 (`ImageMHA`), VAE convolutions on the grid (`GridGEMM`), same-shaped VAE stages in one graph, DiT weights as bfp16, text encoder in groups (`SwapVariants`)
- Fewer DiT weight bytes (judged on pictures), `mem.NPUWired` on the M1, energy per image, 3 prompts × 2 seeds against diffusers' fp32

Measure before adopting: smaller programs for the first compile, fewer text rows, fewer programs a step (`aicheck cycle`: a large group following another costs 50–70 ms).

## 8. Long contexts

- Split at the attention (`--paged`): cost follows the length used. Split, every attention layer calls the NE twice: a model whose layers all attend (Qwen3-0.6B) can run 2–4× faster as one program (`llm.Unpaged`). Count attention layers and measure both
- Two-level attention (`-sparse`) from about 48K. Passkey tests (128K, 250K, and three at 10 / 50 / 90 % of a document)
- Layers with state (GDN): one token at a time must match chunks (`TestCachedMatchesFull`)

## 9. Checks in use, and the record

Do not decide by KL alone. Put answers next to bf16 / fp16 for the model's use:

| Use | How to check |
|---|---|
| Chat | Four questions (Mt. Fuji, 17 × 23, *Kokoro*, the Shinano River) and `hwnpu-chat agent`'s task (does `go test` pass?). When a small model errs, ask fp16: if it errs alike, it is the model's limit |
| Translation | KL and agreement **on the translation's tokens only**; how many of 44 texts match bf16 word for word. `hwnpu-dev formats` perplexity means nothing here |
| Image descriptions, TTS | The same images / audio; TTS output back through ASR |
| Transcription | LibriSpeech's 73 validation clips; 12 Japanese texts in four `say` voices; clips matching HF fp32, WER / CER; audio over 30 s |
| Embeddings | Cosine against HF |
| Detection, OCR | Objects against ONNX Runtime and a graph with only its weights in fp16 (`tools/ref/onnx_f16_weights.py`). OCR: run the vendor's whole pipeline (`ref_ppocr.py`, `compare_ocr.py`); pad as the vendor does |
| Image generation | Images from the same noise; 3 prompts × 2 seeds next to the reference. A distilled few-step model spreads small conditioning changes: judge a part by its own error against fp32 and a grid of pictures. Generate every aspect ratio once on the lowest machine and look at each |

In practice: fp16 comparison programs double the disk (remove after); Core ML and Core AI make a cache per process name (several GB each); a stalled download never times out; `hwnpu-dev formats` shows an unsplit program's weights as 0.00 GiB.

**The record**: the model's facts, numbers, what was adopted and what was not (with why) in its topic file under `docs/design/`; a row in [performance.md](performance.md); models.yaml's measured values (section 11). What could not be measured goes into `docs/handoff-*.md` with what to bring, the commands and the expected values.

## 10. Values for combining it as a part

Models will be chained (speech recognition → LLM → TTS) and run together in one process (planned; [api.md](api.md)). Record when the model comes in:

| What | How | What for |
|---|---|---|
| Each part's pinned size | AMD: `/dev/accel/accel0` mappings in `smaps`. Mac: weights and `phys_footprint` | Memory budget |
| Each part's reload time | AMD `Release` / `Reload`; Mac with the cache hit | Whether it may be evicted |
| Parts run together vs in turn | The flow | Only parts run in turn load a stage at a time |
| AMD: the longest command | `hwnpu-chat` / `irongraph` logs | Others wait for it |
| Mac: total weights and switch time | `TestCallTimes` alternating | A switch costs 15–35 ms |
| Input / output kinds and conditions | Model card, preprocessing | Matching connected models |
| Streaming, time to first output | Feed chunks | Delay of a speech chain |

- Pass data between models as text (token IDs when the tokenizer is the same); connect vectors only between parts trained together
- A model that keeps audio as audio (Audio, Omni) is one part with audio in and out

## 11. Running by name on every NPU (models.yaml and preparation)

A model is not done until a name, a family or a use prepares and runs it on every NPU without manual steps. Details in [distribution.md](distribution.md).

### 11.1 Order of preparation

1. **An export here** with its `NAME.done` (written last); a cut-off one is made again
2. **An export in the sources** (`HWNPU_MODEL_SOURCE`): the entry of the same recipe, else one a person imported by name (`hwnpu-dev cache import`)
3. **A conversion here**: download only the files the converter reads, check sha256, convert, write `.done`. Apple and Intel use the Go converters (`chat.Prepare`, `anima.Prepare`); AMD runs the IRON exporters built into hwnpu (`chat.PrepareAMD`, `amd_export`, `HWNPU_IRON_ENV`). Missing IRON: stop and say what to install (INSTALL_AMD.md)
4. **Not usable**: stop and say what is missing

- Never download the whole repository (Sana's held 16.5 GB, 8.9 needed); share files models of a family hold in common
- A model with no form for this NPU falls back (the CPU) and says so; only an explicit `--device npu` is an error

### 11.2 Values in models.yaml (measured, not estimated)

| Field | What | How | Pitfalls |
|---|---|---|---|
| `bytes` | The download | Sum of HF's `lfs.size` for the files fetched. On AMD's NPU, where preparation fetches the export from the sources instead: the export's size (`du -sb`), in a vendor table (`{amd: 6.59e9, apple: 1.4e9, intel: 1.4e9}`) | GB is 10⁹; `du -h` prints GiB. An export of pictures holds a program of each size with the weights, several times the checkpoint (DA3MONO-LARGE 6.59 GB against 1.4 GB). A vendor the table leaves out takes the largest value |
| `memory` | The peak while running | Sample every 0.1 s over a whole run (`tools/memwatch`; AMD: accel mappings + RSS) | The peak of the run, not after loading (text stage, VAE) |
| `machine` | Least machine memory the family recommends it for | `memory` + room for the OS and apps | Keep models of the same size alike |
| `file` | The checkpoint file when a repository holds several | File list | Tensor prefixes can differ |
| `device` | Where it runs per vendor | Energy of one run on the NPU and CPU | Small models lose on AMD's NPU (YuNet 137 mJ against 10 mJ) |

- **Licenses to accept** (`license: {id, url, terms, accept: true}`): asked once on a terminal, kept in `~/.config/hwnpu/licenses.json`; otherwise `--accept-license` / `HWNPU_ACCEPT_LICENSE`
- **Values that differ by NPU**: a table by vendor (`memory: {amd: 3.5, apple: 4.1}`), a chip may be named (`M1: 1.7`); a vendor left out takes the largest
- **Uses pick by memory like their families**: give a use's `picks` the same steps as its family
- **Check**: `hwnpu-image list` / `hwnpu-chat list` with `HWNPU_MEMORY=8`, `16`, `24`; each NPU shows the right size, memory and RUNS FROM

An example measurement script (AMD):

```bash
"$@" & pid=$!
while kill -0 $pid 2>/dev/null; do
  awk '/accel/{a=1;next} /^[0-9a-f]+-/{a=0} a&&/^Size:/{s+=$2} END{printf "%d ", s/1024}' /proc/$pid/smaps
  awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status    # the NPU and the RSS (MiB); memory is the peak of their sum
  sleep 0.1
done
```

### 11.3 AMD exports and Go

- IRON's compilers (mlir-aie, Peano) cannot move to Go cheaply. The compiled programs depend only on shapes; weights are in banks. **The same shapes give the same programs** (Sana's exports checked byte for byte): distributing programs per shape and writing banks in Go would need no Python on the user's machine
- The shape includes: dimensions, picture sizes, context length and chunk size, each part's format, the NPU generation and columns, the export format version
- Compiling without Python matters only to those making new shapes and the CI; low priority ([amd.md](amd.md))

### 11.4 Image exports: every aspect ratio in one, in the sources

**The rule (every image model, every NPU)**: one export per model and NPU vendor holds **all** of models.yaml's aspect ratios, weights stored once. Never one export per size or a subset. A user switching `--size` must not download again. An exception needs a measured reason in the model's topic file and models.yaml entry.

- **The sizes are models.yaml's** (`npu.amd.sizes`, `npu.apple.sizes`: the eight ratios `--size` takes, `internal/imagegen/aspect`); the command reads them to parse `--size` and to make the export
- **Apple (Core AI)**: a part whose graph depends on size is one `.aimodel` with a function per size (`s{W}x{H}`, `llm.WriteAIFunctions`); size-independent parts are one function; the manifest's `sizes` lists each size's parts (`ProgramManifest.At`); the name carries the sizes (`aspect.Tag`). The part count must not depend on size
- **Its cost on Apple**: Core AI compiles all functions of an `.aimodel` together, and one size the compiler fails takes that part off the NPU for all sizes. Before putting an export in the sources, compile each size alone offline for every chip (`coreai-build compile … --architecture h13g/h14g/h15g/h16g`, look for `Falling back` in `/usr/bin/log`). Then `hwnpu-dev prebuild -arch ARCH -put EXPORT` must end with `0 failed`. VAEs get no prebuilt (`coreai.LeftToDevice`)
- **AMD (IRON)**: the first size's graphs hold the banks, other sizes' graphs are in `sizes/WxH/` on the same banks (`share_banks`). A size not in whole rounds of 512 rows runs padded with keys bounded by `nk`; graphs that depend only on padded rows are shared
- **The key**: `chat.PinnedRecipe(model, tool, version)` plus the exporter's arguments; it needs every input file's `sha256:` in models.yaml. Raise the export version by hand whenever inputs or outputs change
- **The first run puts it in the sources** (`chat.SharedExportOr`, `?writeback=true`, `hwnpu-dev cache flush`); hand-made exports via `hwnpu-dev cache import DIR` or first use
- **Check from the other side**: `hwnpu-dev cache ls N`, then move the local export aside and run by name: it must come from the source and draw the same bytes
- **The checkpoint in the sources** is read only with its `hwnpu-lock.json`
- **Check every size's picture** on the lowest machine (a broken function shows only there)
- **Record the export's time, size and peak RSS** for models.yaml and the user's message

## Not adopted

Measured and not repeated (one line each; more in [performance.md](performance.md) and the topic files):

- Padding small models into batches for the ANE: 73 µs CPU → 280 µs ANE
- Multifunction to get past 1 GiB: the limit counts the whole package; all of it went to the CPU
- KV cache in MLState: same speed (kept as an option for power); int8 KV cache: half the memory, 10 % slower
- Loading the next tier in the background: generation over 2× slower
- One Q=16 program for generation and prompts: +15 %
- E4B at lut4: 20 % faster, next-token agreement 90 % → 64 %
- Compressing the model files (lz4, zstd): 71–100 %, and mapped files would need unpacking
- Sparse attention on top-k single keys: 53 % of the weight at 128K (two levels instead)
- AMD attention projections at 4 bits: −0.12 GB pinned, E2B KL 0.097 → 0.229
- AMD MoE: copying experts then GEMV (copy at 26 GB/s is slower than reading in place)
- A small Qwen3.5's 4-bit MLP fixed by GPTQ: the GPL stays broken (q6 instead); a faster GEMV from q6: it takes q8's time
- A DiT at `HIGH` on AMD: 7.2 → 10.9 s a step for 0.966 → 0.976; a DiT at q4 by default (latent 0.777); GPTQ for DiTs (free runs worse)
- q4nx prompts through bf16 (`QDequant`): KL unchanged, 5–9 % slower
- RMSNorm with a fixed scale in fp16: subnormals (divide by the row maximum)
- Whisper's encoder LayerNorm divided by the row maximum (no change) and its attention in query chunks (no faster)
- NE MoE: experts gathered from a constant table (right, a sixth of the speed), an expert in one array (slows wide experts), 4-bit experts (no form faster than int8)
- gpt-oss's head at lut4 / lut6 (KL 0.0088 → 0.068 / 0.0136; 0.27 / 0.41 GiB against lut8's 0.55)
- Prompt-lookup drafts for host-run MoE experts: a check of 7 drafts adds about 90 ms to a 60 ms step
- Shipping `.aimodelc` with weights; prebuilt for chat models (first load slower: LFM3-1.7B 2:17 → 8:12); prebuilt for VAEs (stripes, no gain)
- One export per picture size (re-downloads; see 11.4)
