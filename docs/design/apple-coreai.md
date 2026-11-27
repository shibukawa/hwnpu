# Apple: Core AI backend

[日本語](apple-coreai.ja.md)

Core AI (CoreAI.framework, macOS 27) is how hwnpu runs Apple's Neural Engine. hwnpu writes Core AI programs (`.aimodel`) in Go from the checkpoints and runs them through a small Swift shim loaded with purego. Core ML is kept only for the CPU interpreter (`--cpu`) and as a reference; see [Core ML](apple-coreml.md).

Why Core AI:

- Core AI keeps what it specialized for the device in its own disk cache, keyed by the program's hash. There is no ~4 GiB limit like Core ML's compile cache, so a second start does not compile again (Core ML recompiled whatever its cache had evicted: 4 s to 13 min for the same chat model on an M1).
- One program may hold 2.5–3 GiB on an M3 (fp16: 2.5 GiB stays all on the Neural Engine, 3 GiB goes to the GPU). Core ML: 1 GiB.
- Functions of one `.aimodel` share their weights on disk, in Core AI's cache and in the Neural Engine's memory. A chat model keeps its prompt program on every machine.
- Programs that Core ML put partly off the Neural Engine (Qwen3.5-2B's body, the MTP parts) run all on it.

| Second start, M1 16 GB | Core ML | Core AI |
|---|---|---|
| Qwen3.5-0.8B | 12 min 54 s | 2.3 s |
| LFM3-1.7B | 57 s | 4.5 s |
| Qwen3.5-4B (M2 16 GB) | 5.8 s; generation 14.5 tok/s; 2041-token prompt 29.5 s | 0.7–0.8 s; 20.5–21.9 tok/s; 4.0 s |

Generation speed is otherwise the same within ±10 %. On an M3, chat models start in 0.2–0.5 s from Core AI's cache.

## Which commands use it

| Command | On a Mac |
|---|---|
| `hwnpu-chat` (chat, embeddings, `systemone`) | Core AI only. `--backend` takes `coreai`, `intel`, `levelzero`, `openvino`; `coreml` is not a choice |
| `hwnpu-image` (generation and the image-reading models) | Core AI only. `convert` writes `coreai` by default; `convert --backend coreml` remains to write what `--cpu` reads |
| `hwnpu-audio` (music, sfx, transcribe, tts, pitch, notes) | Core AI only |
| `hwnpu-dev models package MODEL/coreai-ARCH` | Builds prebuilt programs with `coreai-build` (below) |
| `hwnpu-dev convert` | Writes `coreml` by default; `llm.Convert` also takes `coreai` |

- Without Core AI (macOS 26 or earlier) these commands stop with an error saying that Core AI needs macOS 27 (`errNoCoreAI`). They do not fall back to Core ML.
- `HWNPU_BACKEND=coreml`, which may be left set for other tools, is read as `coreai`.
- `--cpu`, and the CPU paths of detect / pose / ocr / pitch / notes on a machine without ONNX Runtime, read a Core ML export (`.mlpackage`) with `milcpu`. These models are small, so Core ML's Neural Engine problems do not matter there.
- `llm.NativeFormat` returns `coreai` when the device is Core AI. `llm.ExportDevice` opens Core AI for a `coreai` export, and can still run an old Core ML export on Core AI by lowering it when loading.

## Writing `.aimodel` in Go

An `.aimodel` is a directory:

| File | What |
|---|---|
| `main.mlirb` | MLIR bytecode v6 in the `aicode` dialect; constants over 256 bytes are resources (the weights) |
| `main.hash` | sha256 of `main.mlirb`, Core AI's cache key |
| `metadata.json` | Core AI's metadata |
| `hwnpu-io.json` / `hwnpu-io-NAME.json` | inputs and outputs, a file for each function |
| `hwnpu-meta.json` | description and other metadata (`llm.MetaFile`, `ReadMeta`) |

- `internal/coreai/mlirbc` writes and reads the bytecode. `internal/coreai/ailower` lowers MIL to it. The format was matched to coreai-core's output, read back with `tools/coreai/bcdump`. Attributes and types are written in the dialect's text form; op properties in the dialect's order (a required one by attribute number, an optional one as `number << 1 | 1`, absent 0).
- No Python and no Xcode toolchain are needed, so exports are made on any OS (Linux too).
- `llm.WritePart` with `BackendCoreAI` builds the MIL and writes the `.aimodel` directly; no `.mlpackage` in between. The bytes equal those of a lowered `.mlpackage` (`TestCoreAIPartMatchesPackage`, fp16 and lut4), so both paths share Core AI's cache entries.
- `ailower.FormatVersion` (now 2) is part of a shared export's key. Raise it by hand when a change makes the programs of an export of the same name different.
- Memory while writing: `mlirbc.Writer.Spill` keeps blobs up to `ailower.SpillKeep` (512 MiB) in memory and the rest in `NAME.aimodel.blobs.tmp`; `WriteTo` streams them into the resource section at the end of the file. The caller deletes `.milblob.tmp` once a part is lowered (otherwise a part holds three times its size on disk). All in a temp file was slower on a nearly full disk; 512 MiB was the best trade.
- Parts are written in parallel (`console.Jobs`): the smaller of the core count and 3/4 of usable memory (Linux MemAvailable, else half the RAM) divided by a part's estimate (Anima 1 GiB, Sana / Chroma / Qwen-Image 2 GiB, `llm.ConvertAI` 4 GiB). `HWNPU_JOBS=N` fixes it. Output is byte-identical to one at a time (except `metadata.json`'s creationDate). Ryzen AI (20 cores, 59 GB), Qwen-Image-2.1 Turbo lut6: 2 min 16 s → 50 s.

### What Core AI needs from the lowering

| Need | How ailower writes it | Otherwise |
|---|---|---|
| Locations | Core AI's own `#aicode.debuginfo.location_v1`, on block arguments too | builtin `loc` cannot become "versioned IR" |
| Reductions | composite `reduce_max_v1` / `reduce_sum_v1` around `primitive.reduce_v1` (shape-free type) | — |
| Matmul | composite `broadcasting_batch_matmul_v1` around `broadcast_to` and `batch_matmul` | bare ops with int8 weights (`blockwise_shift_scale_v2`) put the whole program off the Neural Engine; a reshape of the weights makes MPSGraph get the shape wrong |
| Softmax | `exp(x − max) × reciprocal(sum)` | `exp / sum` or softmax is fused with the matmul into an attention op that leaves the Neural Engine |
| Binary ops | equal shapes only; widen with `broadcast_in_dims` | — |
| Division by a constant | product by the reciprocal (`HWNPU_DEV=ailower.divide=1` keeps the division, for study) | the Neural Engine multiplies instead of dividing (PP-OCR's detector at cosine 0.05) |
| Matmul with inner dimension 1 | written as a multiply (outer product) | (1×1)·(1×N) is wrong on the Neural Engine (relative error 0.28 at M = 1) |
| softplus | `relu(x) + log(1 + exp(−|x|))` | Core AI fuses `log(1 + exp(x))`; it gives 0 for x > 10.4 |
| Neighbouring lut tables | never equal (`mlpackage.DistinctNeighbours`: one value moved 1 ulp toward 0) | `coreai-build`'s ANE compiler fails ("per-cout group … not content-uniform") and puts the part on the GPU |

Lowerings also exist for `max_pool`, `conv_transpose` (cropped to MIL's pads), `pad` replicate / reflect, `elu`, LUT (lut4–lut8), and more. To write a new one, convert a small torch module with coreai-torch (`uv run --offline --prerelease=allow --python 3.13 --with coreai-torch==0.4.3 --with torch`), print the program and read its bytecode with `bcdump`; coreai-core's `_coreai_ops_gen.py` gives the operands in order.

`ailower` notes known Neural Engine traps as it lowers (`HWNPU_DEV=ailower.notes=1` prints them; `aicheck lint` searches exports). Today: a slice of the last axis starting at `begin mod 32` in 1..14 (values above 4094 become Inf).

## Exports: `NAME.coreai`

Every command puts Core AI programs in the same form (`internal/llm/export.go`):

- `NAME.coreai/` beside the checkpoint in the model's directory, and `NAME.coreai.done` (relative path, `cache.WriteDone`) when complete. An export is one unit of conversion; a cut one is emptied and written again (`StartExport`, `FinishExport`). `ExportDone` also reads an old empty `.done`.
- A model of one program: `NAME.coreai/model.aimodel` (`ProgramPath`). A split model: `model.partK.aimodel` with functions, `NAME.parts.json` for each program listing `PATH#FUNCTION`, and the attention programs.
- Things that change version separately are separate exports (music, sfx and transcribe each have several; see [audio models](audio-models.md)).
- When an export replaces an older layout, the old `coreai/` directory and `NAME.aimodel` / `.done` are deleted.
- Core ML exports (for `--cpu`) and Intel's OpenVINO IR keep their own places.
- Exports are packed and shared through sources: see [distribution](distribution.md).

## Several functions in one `.aimodel`

`ailower.LowerFunctions` writes several `graph_v1` functions into one `.aimodel`, a blob of the same bytes once. The shim opens `PROGRAM.aimodel#NAME`. Core AI makes one cache entry for the asset, and the Neural Engine holds the weights once even with each function open as its own `AIModel`.

- LFM3-1.7B on an M1, 8- and 128-token programs as two functions: `.aimodel` 2.43 → 1.22 GiB, Core AI cache 2.53 → 1.32 GiB, `mem.NPUWired` 2.56 → 1.34 GiB, generation +13 %. Footprint does not change (the weights are in driver memory).
- `llm.ConvertAI` (hwnpu-chat): the programs of one model (tokens per call, last-row logits, `KVOnly`, `state-rows`) are split as usual (`PlanPaged`, `PlanSplit`); parts that start at the same layer become functions of one `.aimodel` (`WriteAIFunctions`), e.g. `base.partK.aimodel#q8` and `#q128-last`. Gemma 4's `KVOnly` parts join the generation parts that start at the same layer. Parts with no layers (`NoLayers`) stay separate. The weightless `attn` / `mass` programs are written for each program. The `.aimodel` name lists all programs inside (`…-c262144-q8-q128-last.part1.aimodel`), so changing `--draft` does not overwrite the files another set points to. The MTP layer and the encoders are single programs. `PrepareEmbedder` makes 128 and 1024 two functions of one asset.
- Because weights count once on Core AI (`sharesWeights`, `heldBytes`), `chat.Prepare` keeps the prompt program on every memory class, and does not apply Core ML's compile cache cap (`compileCacheBytes`). On an M1 prompts read 2.3–3.2× faster for Qwen3.5-2B, LFM3-VL / Audio / Omni-2B and PLaMo 3 Translate 2B, for 0.1–0.14 GiB more Neural Engine memory.
- Function names are `p` + 16 hex digits of the content's sha256: the same on every machine.
- Core AI compiles all functions of an asset together. One function its compiler fails on puts the others off the Neural Engine too (Qwen-Image's VAE stage at 1024² took the 512² one to the GPU on an M1, 30 → 109 ms). Check each size's part alone first.

Automatic grouping when a `.mlpackage` is lowered at load (`internal/backend/coreai/group_darwin.go`):

- Siblings: `.mlpackage`s in the same folder, or of the same name in neighbouring folders (image exports of other sizes), that have the 3 largest weight blobs of the same bytes (`groupBlobs`). Only metadata and those blobs are read. Weights under 1 MiB (`groupMin`) are not grouped.
- A record `KEY.aifn` per package names the asset and function, so later loads hash only their own package. `groupRule` (now 3) makes older records be looked at again. A new sibling (a third size) regroups all and drops the smaller group. The older single assets and their Core AI entries are deleted (`cai_forget`). `cache.Lock` keeps two processes from grouping one folder.
- A group whose function is not all on the Neural Engine is taken apart and its members kept alone (`Alone`).
- Downloaded `.aimodel`s are opened as they are: not hashed, regrouped or deleted.

## Running: the shim

- `internal/backend/coreai/shim/shim.swift`, built by `tools/coreai/build-shim.sh` into `libcaishim.dylib`, embedded with go:embed and loaded with purego. `CGO_ENABLED=0` stays.
- `Compile` opens a program with the Neural Engine preferred. Placement is read from Core AI's cache: the MPSGraph manifest's `mps.fullyPlacedOnANE`, per function under `Entry Function Attributes`. The delegate folder is `main-this-delegates` for every function.
- Calls to one program are serialized (`graph.mu`).
- Inputs: `cai_input_array(h, i)` makes an `NDArray` of input i's shape and returns the address of its dense row-major contents. The caller writes there directly (`backend.InputAllocator`, `hwnpu.Graph.AllocInput`); `cai_run` passes that array without a copy. An array binds to any input of its shape, also in other programs (generation and prompt programs share caches; attention programs take chunks in turn). A buffer of another shape is copied. `chain` allocates caches, rings, windows, states, masks and attention chunks this way; image and audio programs still copy. Gains on an M1: LFM3-1.7B after a 3101-token prompt 39.8–40.6 → 43.0–43.1 tok/s (attention call 1.34 → 1.0 ms), PLaMo 3 Translate 2B 12.1 → 12.6–13.1 tok/s; PLaMo 3 2B's rings held 252 → 84 MiB. The copy grows with the chunk: 0.56 of an attention call's 2.07 ms at 8192.
- Copy speed on an M1 is about 29 GB/s. A program that does nothing costs 0.21 ms a call (M1); the floor of a call is 160–175 µs (M2).
- At most 112 programs per process on the Neural Engine (M2). The 113th fails to load ("no ANE resources") and Core AI runs it on the GPU silently. The device counts loaded programs (`aneLive`) and says so when a load fails near the limit. gpt-oss-20b's export is about 84; other chat models under 40. Two models in one process can pass it.
- Core AI cannot choose compute units for a program the way Core ML's `GraphSource.Units` does. `HWNPU_COREAI_UNITS=cpu` runs everything on Core AI's CPU, for tests without the Neural Engine.
- A load from Core AI's cache takes 0.2–0.3 s, so the next program can be loaded while one runs (`runner.InTurn`, `runner.Ring`). See [image models on Apple](image-apple.md).

## Caches and disk

Where a model's weights are:

| Copy | Place | Key / notes |
|---|---|---|
| Export | `NAME.coreai/…/main.mlirb` | Kept by default |
| Core AI's entry | `~/Library/Caches/coreai-cache/<OS build>/<executable name>/<main.hash>/<ID>/model.aimodelx` | Only `main.hash` and one ID shared by all entries: a copy of the asset elsewhere, or a rebuilt executable of the same name, hits in 0.02 s. About the weights' size |
| aned's compiled ANE code | `/Library/Caches/com.apple.aned` (SIP: unreadable even with sudo) | Keyed by the code-signing identifier. Go binaries are ad-hoc signed as `a.out`, so all dev builds share it; a release signed with a real identifier does not. Purgeable; about 1.6 GB more for a 0.8 GB entry |
| MPSGraph scratch | `$TMPDIR/com.apple.MetalPerformanceShadersGraph` (0.6 GB per function that falls to the GPU), `…GraphHost` (`coreai-build`'s work, 12 GB left behind once) | Left by crashed processes |

Rules that follow:

- Core AI's cache is per executable name. Renamed test binaries add a full copy (1.8 GB a model). Keep the name and use directories (`base/hwnpu-chat`, `new/hwnpu-chat`). `aicheck` built under the name `hwnpu-image` hits that command's cache.
- An OS update compiles everything again from `main.mlirb`.
- With little free disk, macOS deletes Core AI entries, aned's code and sometimes hwnpu's cache root under `~/Library/Caches` (the default policy evicts on `storagePressure`, not by least recent use). It started below about 15 GiB free on an M2, and deleted freshly finished prebuilt programs even with 22 GB free. Keep 25 GB or more free for first runs.
- When aned's code is gone but the entry is still there, `specialize` / `loadFunction` fails at once (`_GenericObjCError.nilError`, `_GenericObjCError 0`). The shim deletes the entry and loads again; with no disk left the Neural Engine compiler fails and the program goes to the GPU. `HWNPU_DEV=coreai.debug=1` prints "the cached entry failed (…), compiled again", deletes entries that fell to the GPU, and warns when free space is under 10 GiB.
- Once Core AI has an entry, `main.mlirb` is not read again: the asset loads even with it emptied (`main.hash` must stay).
- `hwnpu cache`, cutting to its free-space floor, deletes first the Core AI entries of parts that have a prebuilt: they come back from it in seconds.
- The Neural Engine's weights are wired kernel memory (read by `mem.NPUWired`), not the process footprint: see [memory and power](memory-and-power.md).

## Compiling on the device

- All time goes to `ANECompilerService`: one root instance, one thread, one serial queue for the whole machine. Other processes or threads wait behind it (10 processes × 10 parts end together after 67 s; in turn 79 s). `SpecializationOptions` has no knob.
- Compile time is superlinear in a part's blocks (Chroma DiT at 1024², M2: 1 block 30 s, 2 blocks 76 s, 4 blocks 219 s); calls are linear (360 → 1428 ms).
- Each resident program costs wired memory on its first call: about 0.36 GB fixed + 1.2× its lut8 weights (Chroma at 1024²). Switching between resident programs is free while they fit; past the RAM the kernel rewires on every pass. So a DiT is not split finer on 16 GB (see [image models on Apple](image-apple.md)).
- `AIModelCache.model(for:options:)` answers in 1–20 ms whether an entry exists (not whether aned's code does).
- A failed conversion inside Core AI aborts the whole process (LLVM). Find lowering mistakes with `aicheck` first.
- A failed Neural Engine compile is not an error: the program runs on the GPU. Check placement.

## Prebuilt programs (`coreai-build`)

The Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`) has `coreai-build`:

```
xcrun coreai-build compile X.aimodel --platform macOS --architecture h14g --preferred-compute neural-engine --output out/
```

- It writes `X.h14g.aimodelc`: `main.hash` (as the source), `metadata.json` (`sourceHash`), `main-ARCH.mlirb` (1 KB, no weights), and an MPSGraph package with `manifest.plist`, `specialized_model_1.mpsgraph`, `original_model_0.mpsgraph`, the ANE LLIR (`binary_0.llir.bundle/…/*.mlir.bc`, about 1 %) and `resources.bin` (the weights, 99 %).
- Only `--preferred-compute neural-engine` makes ANE code; the default (`none`) is GPU only. One thread per process. Without `--architecture` it builds all 22 in turn and stops at the first crash, so build one architecture per process.
- A part the offline compiler cannot place exits 0 with `ANECCompileOffline() failed … CompilationFailure` on stderr, no LLIR, and an `ANERegionsHash` whose second half is the empty sha256. The real reason is in os_log (process `coreai-build`, `ANEHostCompiler`), not stderr.

Loading (`coreai.Prebuilt`):

- `AIModel(contentsOf:)` on an `.aimodelc` makes `ANELargeModelCompilerService` finish the LLIR (CPU 25 %, one thread) in place of the ANE compiler. The entry lands at the same hash and ID as a native compile, with byte-identical `resources.bin`; later loads take the `.aimodel` path in 0.01 s.
- Rule: when `hasEntry` is false and a prebuilt of the same hash for this architecture is there with `mps.fullyPlacedOnANE` in its manifest, open it. `HWNPU_COREAI_PREBUILT=off` compiles on the device. The shim also splits `PATH.aimodelc#NAME`. A prebuilt that fails to finish (`failedToSpecialize`) is compiled on the device.
- `coreai.LeftToDevice`: programs named `*_device` are always compiled on the device, and `Build` skips them. The text encoders (`text*`) are prebuilt on M3 and later since 2026-11-26, for the compiler's memory (1.5 GB against 0.5 GB, [image-apple.md](image-apple.md)); the recipe leaves them to the device on M1 and M2 (`apple.prebuild`). The VAEs are the recipe's (`model-recipes.yaml` apple `prebuild`, `do: device`): FLUX.2 and FLUX.3 klein's on M1 and M2 (on M1 the prebuilt VAE drew a band of noise every 112 rows at 512² and the 1024 × 768 VAE ended in `Request cancelled`; on M2 512² had such bands from the prebuilt and from the device's compile alike), and Anima's on M2 (up to export v5 `coreai-build` failed 8 of its 10 stages on every chip; v6 builds them all; M2's h14g prebuilt drew bands every 160 pixels, M1's h13g drew right). On an M3 FLUX.2 klein's ten decoder stages compiled in 2799 s and the prebuilt finished in 0.9 s, the same bytes out (2026-11-25, [image-apple.md](image-apple.md)): prebuilt there. An M1 that already drew with a prebuilt VAE keeps that entry: delete `~/Library/Caches/coreai-cache/<OS build>/hwnpu-image/`.
- Another architecture's `.aimodelc` does not load (`incompatibleCompiledAssetArchitecture`), even with the name and `deviceDescriptor` rewritten.
- Calls of a program finished from prebuilt are slower: 3–5 % (Chroma, M2), 8–16 % (M1).
- Code that aned compiled on a device cannot be exported (SIP, per signing identifier). Only `coreai-build`'s output can be shipped.

### Architectures

Names from `coreai-build inspect X.ARCH.aimodelc` (`Supported chips`): h13 M1, h14 M2, h15 M3, h16 M4, h17 M5, h18 M6; `g` base, `s` Pro, `c` Max, `d` Ultra. h15m is another M3 Max; h17p is A18 / A18 Pro; h17d shows no name. Other `--platform`s have other lists, so macOS's 22 are the Mac's.

| Result | Architectures | Macs |
|---|---|---|
| Builds and places on the ANE (12) | h13g, h14c, h14g, h15c, h15g, h16c, h16g, h16s, h17c, h17d, h17g, h17s | M1; M2, M2 Max; M3, M3 Max (h15c); all M4; M5, M5 Pro, M5 Max (and h17d) |
| `coreai-build` crashes (SIGSEGV in `ZinIrCompilerParameters::PropogateGlobalOptionsToPerProcedureOptions`, any program) (9) | h13c, h13d, h13s, h14d, h14s, h15d, h15m, h15s, h17p | M1 Pro / Max / Ultra, M2 Pro / Ultra, M3 Pro / Ultra, M3 Max (h15m), A18: compile on the device |
| No ANE code (`redefinition of symbol named 'hwnpu_N'`), all GPU | h18g | M6: not built; on-device compile unknown (no machine) |

`--preferred-compute none` (GPU) builds all 22. Across the 12, `resources.bin`, `original_model_0.mpsgraph`, `main.hash` and the LLIR `Manifest.json` are identical; the ANE LLIR differs per architecture (11 kinds: only h17c and h17d match); `manifest.plist` (with the ANE generation and LLIR hash in `deviceDescriptor`), `specialized_model_1.mpsgraph`, `main-ARCH.mlirb` and `metadata.json` (1–40 KB) differ.

### Packing and shipping

- `resources.bin` is the bytes of `main.mlirb`'s resource blobs in another order. `coreai.Pack` replaces it with `resources.bin.map.json` (an index of 16-byte heads and 64 KiB positions; the few bytes that differ as literals): a Chroma 1-block part 231 MB → 2.97 MB. Files mostly not in `main.mlirb` are kept as they are.
- `coreai.Assemble` hard-links the small files and writes `resources.bin` from `main.mlirb`, checking size and sha256; it never builds from other bytecode. The backend assembles in a temp folder, lets Core AI open it, and deletes it (the entry holds the weights).
- `coreai.Bundle` makes one `prebuilt.tar.zst` per architecture (zstd; a machine takes only its own): 3.02 MB → 1.22 MB for one Chroma block.
- Key: `coreai.ProgramsDigest` (each `.aimodel`'s place and `main.hash`), the architecture and the export name: `converted/MODEL/coreai-ARCH/EXPORT/v0-HASH`. The model comes from the export's `entry.json`.
- When `Compile` meets a part with no entry and no local prebuilt, it asks the source once per export and process (`coreai.FetchPrebuilt`, set by chat) and puts the result in `EXPORT.prebuilt/ARCH/`. `EXPORT.coreai.prebuilt` is a cache unit of kind `prebuilt`, not counted in the model's size, and kept when cutting. The `.aimodel` export is still needed on machines with a prebuilt (the weights come from it). Sources and packing of exports: [distribution](distribution.md).

`hwnpu-dev models package MODEL/coreai-ARCH|EXPORT[/coreai-ARCH] [-j N] [--force] [-n] [--packed] [--no-push]` (`coreai.Build`):

- Runs one `coreai-build` process per part and architecture, `-j` at once (default: one per 4 GB of memory beyond 6, at most half the cores; 2 on 16 GB). Writes into `.prebuild-*` then renames. Builds and packs part by part and deletes the whole `.aimodelc` (Sana's 27 parts × 12 kinds whole would be 50 GB). It packs and puts each architecture's bundle in the sources written back to; `--packed` packs beside the export only, `--no-push` leaves the whole programs beside their sources. Given a model, it first makes the model ready for coreai (`hwnpu-dev models package MODEL/coreai`: from the sources, else converted) and compiles its exports.
- A part the offline compiler fails on counts as failed (no output kept) and is tried once more alone after the others.
- Build one architecture per run (`--arch ARCH`): with all 12 at once nothing was put before the disk filled. Watch free disk and `$TMPDIR/com.apple.MetalPerformanceShadersGraphHost`.
- Chroma1-Flash at 1024²: 29 parts in 9 min (2.3 GB a process peak); ANE code 140 MB in all, weights 15.6 GB.

### Where prebuilt pays

The first run, empty Core AI and aned caches (a copy with a new signing identifier), 1024²:

| Model, machine | On the device | From prebuilt | Picture |
|---|---|---|---|
| FLUX.2 klein 4B, M1 16 GB | 40 min 08 s | 7 min 48 s | byte-identical |
| FLUX.2 klein 4B, M2 16 GB | 52 min 16 s | 16 min 55 s | same layout, details differ (PSNR 28.9 dB) |
| Chroma1-Flash, M2 16 GB | DiT compile about 46 min | all loads and finishes 1 min 07 s; picture 7 min 24 s | byte-identical |
| Anima-Turbo, M2 16 GB | 45 min 01 s | 37 min 50 s (−16 %) | byte-identical |
| LFM3-1.7B, M2 | 2 min 17 s | 8 min 12 s | same answers and speed |
| plamo-3-translate-8b, M2 | 1 min 13 s | 7 min 08 s | same |
| Qwen3.5-4B, M2 | 12 min 26 s | loaded in 7 min 07 s, then no answer for 1 h 40 min | — |

- It pays for large parts that take minutes each to compile (image DiTs: FLUX's `dit_single` about 7 min → 33 s). Small parts lose: finishing costs about 10 s a program (the service exits after each and launchd delays its respawn), and the first prebuilt of a process seemed to carry a fixed cost of minutes.
- So prebuilt is built for FLUX.2 klein and Anima (and was for Chroma1-Flash, removed; DiT and other non-text, non-VAE parts), and not for chat models.
- Step time is the same either way.

Open problems in Core AI and `coreai-build`, Apple's report numbers and what to do when each is fixed: [known issues](../known-issues.md).

## Checking a model on Core AI

Tools in `tools/coreai`:

| Tool | Use |
|---|---|
| `aicheck run -cpu PKG…` | Run parts on Core AI and compare with `milcpu` on random inputs; `-coreml` runs the same part on Core ML (is the error Core AI's?); `-calls N` averages later calls; `-ref` compares with a reference package (an `X.aimodel` is compared with `X.mlpackage` beside it without one); `AICHECK_WORST=1` prints the worst elements; `AICHECK_CONST=name=v,…` gives constant or mask inputs their real values |
| `HWNPU_DEV=ailower.cut=N` | Lower only the first N ops and compare that op by cosine with `milcpu`: bisect for the op that is wrong or leaves the Neural Engine |
| `HWNPU_DEV=ailower.trace` | Map op_ids (Core AI's errors name them) to ops |
| `aicheck probe` | Small MIL programs (SDPA, RoPE, RMSNorm, int8 linear, rewrites): do they stay on the Neural Engine? |
| `aicheck lower OUT PKG` | Stops at the first op with no lowering |
| `aicheck merge OUT.aimodel f8=PKG f128=PKG` | Several packages as functions of one asset |
| `aicheck cycle [-rounds N] PKG…` | What a program pays when it follows another |
| `aicheck rules`, `limits`, `lint` | The Neural Engine's numeric rules against float32, its limits (call floor, program count, formats), and lint of exports |
| `caiload` (Swift) | specialize, run, `-cycle` switching cost, from Core AI alone |
| `lutfake`, `lutscan -tile 16`, `kleinonly`, `bcdump` | Synthetic lut layers, equal neighbouring tables, FLUX.2 klein prepare / put, bytecode dump |
| `llirtasks [-task N [-around K]] [-check] X.mlir.bc` | The Neural Engine tasks of a prebuilt program's LLIR (`X.aimodelc/…/binary_0.llir.bundle/…/X.mlir.bc`) in their order, with what each computes and the kernel it loads: the task the device's finishing service was at when it failed (`Convert 1/1 procedures of ANEHLO task N/M` in os_log). `bcdump` reads the LLIR too `-check` prints only the tasks that read coefficients past the end of the kernel they load, and fails when there is one: an M1 hangs on such a program or its finishing service crashes (known-issues 16). `hwnpu-dev prebuild` runs the same check on every program it compiles (`internal/coreai/llir`). |
| `AICHECK_SENTINEL=PROGRAM aicheck run …` | A program known to run, run first and again after each program that fails until it runs: a prebuilt program that times the Neural Engine out leaves it failing the next ones for a while, which are then not judged |
| `aicheck locked CMD…` | CMD under the Neural Engine lock. The hwnpu processes CMD starts run under the same hold (`HWNPU_NPU_LOCK_FD`), so no other session gets the Neural Engine between them. A process CMD leaves behind keeps the lock: let helpers close descriptor 3 |
| `partprobe -dir CHECKPOINT -o OUT.aimodel NAME=start:layers:seq:paged[:key=value,…] …` | Parts of a chat model as the functions of one program, each with its own options (rows a call, split at the attention or not, the architecture's parameters): one finish of a prebuilt program tells which part fails. With one NAME it writes the export's own part (the same `main.hash`) |

`aicheck run` and `probe` take the Neural Engine lock like the commands (`llm.LockNPU`): two processes on it at once have frozen a Mac, and another process compiling spoils timings.

Checks for a new model (in addition to the [Neural Engine's](apple-neural-engine.md) and the [checklist](new-model-checklist.md)):

- Check on Core AI even when Core ML passes, and on an M1 too: MoGe-2's dense head and Sana's VAE passed on an M3 or on Core ML and failed the M1's compiler on Core AI.
- Tell the lowering's error from the Neural Engine's with `HWNPU_COREAI_UNITS=cpu`.
- Look at what the error is, not only its size (PP-OCR's error was a multiply in place of a divide, found by fitting a line per channel). A tiny program of the suspect ops may be right; test against the real part's cut.
- A small program (about 45 ops) goes to the CPU whatever it holds: find what puts a program off the Neural Engine by adding the piece to a real-size program known to be on it.
- For sizes as functions of one asset, check each size's part alone before cutting.
- Compare start times with both caches warm (three starts each, right after a run of each): an evicted Core ML cache once looked like a Core AI speedup.
- Decide the reference's precision first: the CPU interpreter through `runner` is fp32 unless `HWNPU_DEV=cpu.f16=1`; `aicheck` runs it with fp16 rounding.

## Pitfalls of Core AI and the Neural Engine under it

Shared Neural Engine numerics (fp16 range, inaccurate silu, int8 overflow…) are in [Neural Engine](apple-neural-engine.md). Found on the Core AI path:

| Pitfall | Workaround |
|---|---|
| `divide_v1` by a constant computed as a product on the Neural Engine | ailower writes a product by the reciprocal |
| Core AI's CPU overflows a `reduce_mean` whose sum passes fp16 (the sum returns to fp16 before 1/n; 57600 values of mean −1.37 → −inf). Neural Engine and GPU are right | Keep in mind when using `HWNPU_COREAI_UNITS=cpu` |
| `(v + c) - c` folded to `v` even with c an input (breaks magic-number rounding) | Avoid that rounding form on the Neural Engine |
| Width-axis convolution with a 1-wide kernel over 512+ channels and a computed input comes out wrong | Run it along the height: x as (1, C, T, 1), weight (co, ci, k, 1) |
| Dilated convolutions wrong at some lengths, or sent to the GPU | Lay frames out as rows of the dilation and convolve undilated |
| An explicit `pad` before a convolution went wrong | — |
| Matmul with inner dimension 1 | Written as a multiply |
| softplus 0 for x > 10.4 | Stable form (above) |
| Last-axis slice from `begin mod 32` in 1..14: values over 4094 → Inf | `HWNPU_DEV=ailower.notes`, `aicheck lint` (only LFM2-8B-A1B's routing weights hit it, values ≤ 1) |
| linear / conv output, even a partial sum, reaching 2^15 → Inf (M2) | Keep ranges down (MaxScaled, `mil.BlockRanges`) |
| Denormals inside a matmul become 0 | Range checks |
| "empty reset layer candidates in SS cost model": a pair of ops fails where each alone passes (Sana's VAE `tile` + 1×1 conv; MoGe-2's `pixel_shuffle` after a 1×1 deconv) | Rewrite with the same values: a channel repeat as a 1×1 conv of 0/1 weights; the shuffle in bands of rows (`mil.deconv2`, up to 256 rows) |
| A product of two 8 × 8 matrices puts a program off the Neural Engine ((8 × 8)·(8 × 128) and k·kᵀ do not) | — |
| The attention program a kv head at a time leaves the Neural Engine with one kv head | Keep the grouped form there (see [LLM engine](llm-engine.md)) |
| M1's compiler rejects `sin`, a clip fused into elementwise ops ("NEFUSED_EW: EW: MAX"), prescaled subtraction | Write around them (MOSS DAC's snake as a series): [audio models](audio-models.md), [known issues](../known-issues.md) |
| `coreai-build` stricter than the device compiler | Write forms that pass offline too |
| One failing function puts all functions of an asset on the GPU | Check each function alone; ungroup |
| Over 112 programs in a process run on the GPU silently | Count programs (`aneLive`) |
| Core AI's own conversion failure aborts the process | Check with `aicheck` first |

## Core AI choices described elsewhere

| Choice | File |
|---|---|
| Short conversations on a whole program, then the program split at attention (`wholeTiers`, `decode.Session.Migrate`; none on an M1: `chat.wholeSlower`); Qwen3 split at attention on Core AI; attention a kv head at a time (`paged.Heads`); ring tiers for long windows | [LLM engine](llm-engine.md) |
| DiT groups swapped per call, load ahead, batching, attention layouts, block counts per memory | [image models on Apple](image-apple.md) |
| Experts picked by the host between parts | [MoE](moe.md) |
| Exports in sources, `files.tar.zst`, prebuilt bundles | [distribution](distribution.md) |

## Not adopted

- **Falling back to Core ML** on a Mac without Core AI: the commands need macOS 27 and say so.
- **Core ML as the intermediate form** (`.mlpackage` lowered at load): exports are written from checkpoints directly; it cost a third copy of the weights on disk (export, lowered `.aimodel`, Core AI entry) and conversion time.
- **Shipping whole `.aimodelc`**: 99 % is weights that the `.aimodel` already holds; ship the `.aimodel` plus a weightless packed prebuilt per architecture.
- **Prebuilt for chat models**: slower first runs (LFM3-1.7B 2 min 17 s → 8 min 12 s) and Qwen3.5-4B did not answer.
- **Prebuilt for text encoders** (`LeftToDevice`) **and for VAEs on M1 and M2** (the recipe): wrong pictures / `Request cancelled`, and no gain.
- **Prebuilt for Sana**: `coreai-build` placed only 8 of 27 parts on the ANE when tried, and those finished slower than a native compile. Qwen-Image is not a target either.
- **h18g and the 9 crashing architectures**: not built (a GPU-only prebuilt would keep an M6 off the ANE).
- **Emptying `main.mlirb` once Core AI has an entry** (`HWNPU_DEV=coreai.drop-bytecode`, removed 2026-11-26): it only reached assets lowered from a `.mlpackage` into the compile cache, never the downloaded exports that are the main flow (YuNet on the M1: emptied in the first case, left whole in the second), and for a downloaded export an evicted entry (an OS update, low disk, a new executable name) would mean downloading it again. Disk is freed by the `hwnpu` command's cache cleaning; the library always keeps `main.mlirb`. An empty `main.mlirb` given to Core AI's parser aborts the process (LLVM: `cannot unwrap empty odiec_module_t`).
- **Parallel compiles**: the machine has one ANE compiler queue.
- **Fresh `NDArray` per call, or views of caller buffers** (`NDArray.RawView`, IOSurface): 10.0 / 10.7 / 8.3 ms against 7.2 ms for a reused array (PLaMo 3 0.8B part1, M1); now the caller writes into Core AI's array.
- **A whole program and the split one in one `.aimodel`**: Qwen3-0.6B became 60 functions, loading took 2 min 28 s, some went to the GPU, and each function on the GPU wrote 0.6 GB of MPSGraph scratch until the disk ran out.
- **Two whole-program tiers (256 and 1024) as functions of one `.aimodel`**: the 1024 one ran at half speed (29.4 against 59–60 tok/s). One tier, a separate `.aimodel` per cache length.
- **Switching to a 1-token function when drafts miss**: 8 tokens a call cost only 4 % more than 1 (LFM3-1.7B, M1).
- **Gate and up as one linear** (as on AMD): 3.31 vs 3.30 ms a part on LFM3-1.7B; the parts are bound by weight bytes (~51 GB/s) and Core AI puts no call boundary between two linears.
- **`--decoder-on gpu|cpu`** in hwnpu-audio: removed; Core AI cannot choose units.
- **Finer DiT groups on 16 GB** (Chroma `--group 2`): compile halves but a step is 60 % slower (wired-memory rewiring).

## Open

- Core AI's `.persistent` cache policy (`AIModelCache.specialize(…cachePolicy:)`, kept in `~/Library/Application Support/coreai-cache/`): avoids eviction recompiles, but hwnpu would have to delete entries itself (`deleteEntries(for:)`), and aned's code would still be evicted. Not decided.
- Whether the same `.aimodel` loads across OS versions (the cache folder is per OS build).
- The fixed minutes of the first prebuilt finish in a process, and Anima's 13-minute first step (with or without prebuilt): not investigated.
- Why the VAE stripes at 512² on h13g and h14g (on M2 compiled on the device too), and why T5 is cancelled; remove the recipe's rules and `LeftToDevice` when fixed. Older FLUX.2 klein prebuilts in the sources (of exports the code no longer reads) still contain VAE / text parts; the v6 export's h13g one leaves them out.
- `coreai-build` crashes on 9 architectures and gives no ANE code for h18g (reported to Apple); whether an M6 compiles on the device is unknown.
- aned's disk use not measured directly.
- Input arrays written in place: not measured on M2 / M3; image and audio programs still copy.
- macOS 26 error path and Intel Macs not checked on a machine.
