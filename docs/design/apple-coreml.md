# Core ML backend

[日本語](apple-coreml.ja.md)

The Core ML backend (`internal/backend/coreml`, macOS arm64) is **best effort**. The commands run
on [Core AI](apple-coreai.md) (macOS 27); Core ML stays because the library and some tools use it,
and it can run a model on the CPU. This file keeps its limits and a few numbers, not a full survey.
Facts about the Neural Engine itself are in [Neural Engine](apple-neural-engine.md).

## Where it is used

| Where | How |
|---|---|
| Public library (`bonsaiai.Open`) | Registered first in `backends_darwin.go`: the default Apple device. Core AI by name (`WithBackend("coreai")`, `BONSAIAI_BACKEND=coreai`) |
| CPU runs | `GraphSource.Units` / `BONSAIAI_COREML_UNITS=cpu` run a package on Core ML's CPU (references, models the ANE runs worse) |
| `llm.OpenDevice` with no backend named | Prefers `coreml`; used by `bonsaiai-dev generate` and `formats` |
| `bonsaiai-dev info`, `powerbench`, `probe`, `report` | Core ML device, placement report (`MLComputePlan` shows each op's device) |
| `aicheck run -coreml` | Core ML as a comparison when Core AI's result looks wrong |

`bonsaiai-chat`, `bonsaiai-image` and `bonsaiai-audio` read `coreml` as Core AI. The program format
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

| Limit | Value | What BonsaiAI does |
|---|---|---|
| Weights per package | 0.99 GiB runs on the ANE, 1.02 GiB falls to the CPU (M1–M3, every format); M4 2 GB | Split by layers at conversion (below) |
| Too little work | Small models stay on the CPU even in fp16 ((32, 256)→256 on CPU, (64, 512)→512 on ANE; one-row parts) | 8 rows; merge small parts; batch padding (`tools/coreml/ane_pad.py`) |
| ANE compile cache | About 4 GiB (see [Neural Engine](apple-neural-engine.md)) | Fewer, smaller programs |
| BonsaiAI's cache | `.mlmodelc` and placement results in `WithCacheDir` (commands: `~/Library/Caches/bonsaiai`), keyed by the package's tree digest | `cache.Prune`; converting again leaves old entries |
| e5 bundle cache | `~/Library/Caches/<process>/com.apple.e5rt.e5bundlecache`, no cap seen (49 GB) | Build test binaries with the command's name |

Ops Core ML puts on the CPU and their stand-ins are in [Neural Engine](apple-neural-engine.md). Check a part with `bonsaiai-dev info -model PART.mlpackage` (`all on the NPU`).

### Splitting by layers

- `bonsaiai-dev convert -target m1|m2|m3|m4|any|none` (default `any`) and `-parts N`; `llm.PlanSplit`
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
