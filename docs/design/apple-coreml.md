# Core ML backend

[日本語](apple-coreml.ja.md)

The Core ML backend (`internal/backend/coreml`, macOS arm64) is **best effort**. The commands run
on [Core AI](apple-coreai.md) (macOS 27); Core ML stays because the library and some tools use it,
and it can run a model on the CPU. This file keeps its limits and a few numbers, not a full survey.
Facts about the Neural Engine itself are in [Neural Engine](apple-neural-engine.md).

## Where it is used

| Where | How |
|---|---|
| Public library (`hwnpu.Open`) | Registered first in `backends_darwin.go`: the default Apple device. Core AI by name (`WithBackend("coreai")`, `HWNPU_BACKEND=coreai`) |
| CPU runs | `GraphSource.Units` / `HWNPU_COREML_UNITS=cpu` run a package on Core ML's CPU (references, models the ANE runs worse) |
| `llm.OpenDevice` with no backend named | Prefers `coreml`; used by `hwnpu-dev generate` and `formats` |
| `hwnpu-dev info`, `powerbench`, `probe`, `report` | Core ML device, placement report (`MLComputePlan` shows each op's device) |
| `aicheck run -coreml` | Core ML as a comparison when Core AI's result looks wrong |

`hwnpu-chat`, `hwnpu-image` and `hwnpu-audio` read `coreml` as Core AI. The program format
`.mlpackage` (MIL) is still what most models convert to; Core AI lowers it, `milcpu` runs it.

## How it works

- Core ML through `purego/objc` (no cgo). Compute units `CPUAndNeuralEngine` (never the GPU by
  default). Inputs wrap a `Buffer` with no copy; outputs go to `outputBackings` (0 copies).
- `GraphSource.Function` loads one function of a multifunction package.
- Placement: `MLComputePlan` (macOS 14.4+) must put each op on the ANE. `PlacementStrict` fails with
  `ErrNotOnNPU`; `PlacementPrefer` runs and lists the ops off it. The plan is an estimate
  (`Verified` is not proof). It waits up to 30 min (`planTimeout`); the result is cached (a start
  42 → 16 s).

## Limits

| Limit | Value | What hwnpu does |
|---|---|---|
| Weights per package | 0.99 GiB runs on the ANE, 1.02 GiB falls to the CPU (M1–M3, every format); M4 2 GB | Split by layers at conversion (below) |
| Too little work | Small models stay on the CPU even in fp16 ((32, 256)→256 on CPU, (64, 512)→512 on ANE; one-row parts) | 8 rows; merge small parts; batch padding (`tools/coreml/ane_pad.py`) |
| ANE compile cache | About 4 GiB (see [Neural Engine](apple-neural-engine.md)); measurements below | Fewer, smaller programs |
| hwnpu's cache | `.mlmodelc` and placement results in `WithCacheDir` (commands: `~/Library/Caches/hwnpu`), keyed by the package's tree digest | `cache.Prune`; converting again leaves old entries |
| e5 bundle cache | `~/Library/Caches/<process>/com.apple.e5rt.e5bundlecache`, no cap seen (49 GB) | Build test binaries with the command's name |

Ops Core ML puts on the CPU and their stand-ins are in [Neural Engine](apple-neural-engine.md). Check a part with `hwnpu-dev info -model PART.mlpackage` (`all on the NPU`).

### The ANE's compile cache

hwnpu's cache (`.mlmodelc` and placement) hits. A slow load from it is the system's cache missing:
`MLModel` runs `ANECompilerService` again. Core AI's cache is another one
(`~/Library/Caches/coreai-cache/<OS>/<executable>/`) and keeps 17 GB exports ([Core AI](apple-coreai.md)).

M1 16 GB, three programs of the Sana-Sprint 0.6B export, each load (`runner.Open`) in a new process:

| | vae_s1 | dit_blocks_0 | vae_s0 |
|---|---|---|---|
| 1st | 129.2 s | 14.7 s | 10.6 s |
| 2nd | 0.1 s | 0.1 s | 0.1 s |
| 3rd | 0.1 s | 0.1 s | 0.1 s |

Misses inside the capacity: Gemma 4 E2B (M3) misses at every start (98 s; each part's `MLModel` load
22–25 s) while hwnpu's cache hits (0.7–0.8 s). Neither ops on the CPU nor smaller parts (three of
0.51–0.55 GiB) change it. The format does (Q=8 loaded 3 times, `TestCallTimes`):

| | Parts | 1st | 2nd | 3rd |
|---|---|---|---|---|
| Gemma 4 lut4 | 2 (0.54 + 0.52 GiB) | miss | miss | **hit** (0.07 + 0.04 s) |
| Gemma 4 lut6 | 2 | miss (29 + 25 s) | miss | miss |
| Gemma 4 lut8 | 3 (0.69–0.74 GiB) | miss (8–10 s) | miss | miss |

LFM3 of the same size hits (lut6 from the 2nd load, lut8 VL from the 3rd), and Sana's Gemma 2 in int8
hits on the M1. Something in Gemma 4's lut6 and lut8 graphs is not kept; the cause is not known.

`hwnpu-dev probe cache` on an M2 (16 GB, macOS 27.2, 2026-11-10):

- fp16 programs of 0.45 GiB hit up to 16 (7.25 GiB) and miss at 20 (9.06 GiB).
- 1.02 GiB: 1 hits, 2–3 hit from the 3rd round, 4 or more miss.
- int8 hits up to about 3 GiB in total, lut4 about 1 GiB.
- So the count of layers seemed to decide (fp16 about 72, int8 about 50, lut4 about 36), not the GiB.
- SANA 1.5 on the M1 does not fit this: Gemma 2 int8 (about 2.2 GB), the DiT in fp16 (about 3.2 GB) and the VAE all hit from the second process when
  no other process uses the ANE. Whether the M1 differs or synthetic programs do is not known.

To measure: reuse the same packages (converting again writes new random UUIDs, so both caches
miss), and make sure no other process compiles for the ANE. How to run the probe: [PROBE](../../PROBE.md).

### Splitting by layers

- `hwnpu-dev convert -target m1|m2|m3|m4|any|none` (default `any`) and `-parts N`; `llm.PlanSplit`
  balances by estimated bytes. Output `out.partN.mlpackage` and `out.parts.json`; `internal/llm/chain`
  binds part i's output as part i+1's input, no copies.
- What crosses a cut must be the residual stream at the reference's scale (the port-architecture
  skill states it); otherwise no cut is found.

## Numbers

| What | Result |
|---|---|
| Split, M3 16 GB, lut4, 64 layers (1.81 GiB) | 1 package on the CPU; 2 parts 30.0 ms/token, 4 parts 30.2, all on the ANE (~65 GB/s) |
| Split, M4 (3.6 GiB) | 2/4/8/16 parts 58.3 / 58.8 / 59.4 / 60.6 ms/token |
| Cost of a cut | 0.1 ms (M3) to 0.16 ms (M4); error identical with 2 and 4 parts |
| Same layers by format, M3 | lut4 / lut6 / lut8 1.81 / 2.65 / 3.49 ms; Q4_0 layout 7× slower or on the CPU |
| Batch padding, 3-layer MLP (width 128) | 64 padded rows all on the ANE: 280 µs; 1 row on the CPU 73 µs, 64 rows 104 µs (use only when chosen) |

## Not adopted

- Multifunction packages to pass 1 GiB: the limit counts the whole package (2 functions all on the
  CPU, 12549 ms/token, vs 2 packages 30.6 ms).
- One export per split count; weights as inputs; reloading parts per use; the GPU (non-goal).
- Run-time splitting of one shipped package (`MemoryBudget`, `Reshape`): the split is decided at
  conversion.
- Covering every model on Core ML: the commands moved to Core AI, so Core ML gets no per-model
  tuning or measurements.

## Open

- Only the 8-part split of LFM3-1.7B loads slowly from the cache (23.7 s).
- Whether `MLState` KV caches count toward the 1 GiB limit.

ANE compile cache, low priority (the commands use Core ML only for `--cpu`), with
`hwnpu-dev probe cache`:

- Capacity on the M1, M3 and M4: the largest total that stays warm, and whether bytes or programs
  decide.
- Holding or closing programs, the order they are pushed out, and the process (loading again in one
  process, the executable's name, after a restart). One machine is enough.
- Gemma 4's misses inside the capacity: lut6 and lut8 in `-cache-format`, a mode that loads
  exported packages, and what `e5bundlecache` holds for the programs that miss.
- A fixed `uuid()`: `internal/mlpackage/package.go` writes random UUIDs (`rand.Read`), so every
  conversion changes the package's bytes. Fixed values would let both caches hit after converting
  again.
