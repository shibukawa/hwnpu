# hwnpu design

[日本語](README.ja.md)

hwnpu is a Go library and a set of commands that run models on the NPU built into the CPU package:
the Apple Neural Engine (Core ML, Core AI), Intel's NPU (Level Zero, OpenVINO IR) and AMD's XDNA
(amdxdna ioctl, IRON). No cgo, no GPU. These files hold the current design and the current
decisions. History stays in git.

## Goals

- **Run on the NPU, and say so.** ONNX Runtime's EPs and OpenVINO's AUTO move unsupported ops to the
  CPU or GPU without telling. hwnpu is strict by default: a graph that does not run entirely on the NPU
  fails before it runs, with the ops that did not fit ([API](api.md)).
- **Use little power.** The power policy is a first-class option, and the energy is read where the
  backend allows ([memory and power](memory-and-power.md)).
- **Small models done well.** Several shapes of one model (prompt / decode), shared weights, compile
  caches that survive restarts, models kept resident or dropped as memory allows.
- **Models from Hugging Face, converted in Go.** Supported architectures are ported by hand from
  `modeling_*.py` and written straight to each NPU's format ([architecture](architecture.md)).
- **No cgo, few dependencies.** The root package needs only purego and golang.org/x/sys on Linux and
  Windows.

## Non-goals

- The GPU (NVIDIA, Intel XMX, AMD RDNA, Apple's GPU). Use another engine when a GPU is wanted.
- Data-center NPUs (AWS Inferentia, Google TPU and the like).
- A general importer (ONNX, TorchScript) into the converter. Models outside the whitelist are brought
  as vendor files (`.mlpackage`, OpenVINO IR, xclbin / ELF).

## Principles

| Principle | What it means |
|---|---|
| NPU only | Work stays on the NPU even when moving a part to the CPU would be a little faster. The CPU runs only what the NPU cannot (sampling, tokenizers, small host steps), and models that the NPU runs worse than the CPU are placed on the CPU by `models.yaml` (`device: {amd: cpu}`) |
| No GPU | Not a backend and not a fallback. Core ML's compute units never include the GPU by default |
| Power over speed | At equal total time keep the work on the NPU. Read fewer weight bytes (fewer bits, skip unused layers, MoE). Cut waste on the NPU (programs reloaded because they do not fit, NPU and CPU taking turns). Measure mJ/token with `hwnpu-dev powerbench`; when power cannot be read, compare weight bytes read, NPU memory (`mem.NPUWired`) and CPU use |
| No cgo | purego, `golang.org/x/sys` ioctl, and small C or Swift shims loaded at run time (`native/xrtshim`, Core AI's `libcaishim.dylib`). `CGO_ENABLED=0` builds everything |
| Strict by default | `PlacementStrict` refuses a graph that is not verified to run on the NPU |
| Convert once, in Go | From the official bf16 checkpoint (`config.json` + safetensors / GGUF) straight to each NPU's format. No ONNX in between. A published quantized checkpoint is used only when it can be rearranged without loss; it is never re-quantized |
| Check against the reference | Every ported model is checked layer by layer against PyTorch (fp64/fp32) on the CPU before it goes to an NPU ([new-model checklist](new-model-checklist.md)) |
| Record decisions | When a model misses a pass condition, write the reason before going on. Decisions not to do something are kept with their reason |

## Files

| File | Scope |
|---|---|
| [README.md](README.md) | This index, goals and non-goals, principles |
| [architecture.md](architecture.md) | Vendor software stacks, design policy, package layout, purego/ABI rules, testing and diagnostics, the conversion pipeline in Go, the model whitelist, open milestones, references |
| [api.md](api.md) | Public Go API (devices, buffers, GraphRunner, KernelRunner, strict mode, power, errors, options), composing models as parts, `npuinfo`, `modelstore` |
| [apple-neural-engine.md](apple-neural-engine.md) | Apple's Neural Engine for either backend: limits, ops it does not run, numeric pitfalls, prompt programs, the lock between processes, its memory |
| [apple-coreml.md](apple-coreml.md) | The Core ML backend (best effort): where it is used, the size and cache limits, splitting by layers, a few numbers |
| [apple-coreai.md](apple-coreai.md) | The Core AI backend (macOS 27): MIL to `.aimodel`, two-function `.aimodel`, cache, prebuilt programs on Apple, NE compile, Core AI pitfalls |
| [intel.md](intel.md) | Intel's NPU: Level Zero, OpenVINO IR, limits, residency |
| [amd.md](amd.md) | AMD XDNA: the amdxdna driver, ioctl, ERT, xrtshim and Windows, NPU1, the IRON infrastructure and kernels all models share, power |
| [amd-llm.md](amd-llm.md) | Language models on AMD's NPU through IRON: dense models, multimodal towers, prompt and decode programs, AMD weight formats |
| [llm-engine.md](llm-engine.md) | KV cache and tiers, attention split (paged), sparse attention, prefill, last-row logits, prompt store, long context |
| [llm-models.md](llm-models.md) | LFM3 family (VL, Audio, d1), Gemma 4, PLaMo 3 / Translate / Embedding, MiniLM / BERT, tokenizers, chat formats |
| [llm-qwen.md](llm-qwen.md) | Qwen3, Qwen3.5 (Gated DeltaNet), Qwen2.5-Coder |
| [llm-decoding.md](llm-decoding.md) | Speculative decoding (prompt lookup, MTP), think budget, loop detection |
| [moe.md](moe.md) | Mixture of experts: Qwen3.6-35B-A3B (Mac and AMD), gpt-oss-20b |
| [image-models.md](image-models.md) | Image generation across NPUs: catalog, samplers, CFG, editing, batch, aspect ratios, exports, which models stay |
| [image-amd.md](image-amd.md) | Image models on AMD's NPU: DiT, VAE and text encoder forms and kernels |
| [image-apple.md](image-apple.md) | Image models on Apple (Core ML / Core AI) and Intel |
| [vision-models.md](vision-models.md) | Detection, pose, OCR, segmentation, depth, face, SigLIP 2, ONNX-only models |
| [audio-models.md](audio-models.md) | Whisper, TTS, Magenta RT2, MOSS-SoundEffect, SwiftF0, Basic Pitch |
| [distribution.md](distribution.md) | `models.yaml`, the cache root, sources (folder, WebDAV, S3), downloads, gated repositories, licenses, prebuilt distribution, disk sharing (clone / reflink), cleanup |
| [hwnpud.md](hwnpud.md) | hwnpud (design, not built): one per-user process on the Mac and Linux owns the NPU and the models; model and cache management, socket, supervisor and engine, scheduling across apps, installing and distributing, signing |
| [variables.md](variables.md) | hwnpu's environment variables: the settings, the machine's libraries, the developers' switches (`HWNPU_DEV`) and the tests' (`HWNPU_TEST_<GROUP>_<ROLE>`, `HWNPU_TEST_NPU`), with the names before |
| [cli.md](cli.md) | The commands (`hwnpu`, `hwnpu-chat`, `hwnpu-image`, `hwnpu-audio`, `hwnpu-dev` …), the OpenAI API, OpenCode, terminal display, Ctrl+C |
| [memory-and-power.md](memory-and-power.md) | Energy measurement, power policy, memory measurement, machine memory classes, residency |
| [quantization.md](quantization.md) | Weight formats (lut4/6/8, int8, q4/q6/q8, mxfp4), GPTQ, rotation, per-model choices |
| [new-model-checklist.md](new-model-checklist.md) | The checklist for bringing in a new model |
| [performance.md](performance.md) | What each improvement bought, caveats when combining them, the speed ceiling, what is not verified yet |

## Updating these files

- **Write the current state, not a log.** When a change overturns what a file says, replace that text.
  Do not add dated entries, "was X, now Y", or PR numbers. History stays in git.
- **Keep decisions not to do something.** A form tried and dropped, a model removed, a method rejected:
  put it in the file's `## Not adopted` section with a one-line reason (and the number, when the number
  is the reason). A model removed, taken off models.yaml, or evaluated and not implemented also gets a
  note of its own in [../not-adopted/](../not-adopted/README.md) (its config, how far it got, the commits
  that hold its code, its problems, how to resume), and its long passages move there.
- **Edit the topic file the change belongs to** (table above). Add a new file only for a new topic, and
  add it to the table.
- **English first, then Japanese.** Change `NAME.md` and `NAME.ja.md` in the same commit, with the same
  headings and tables.
- **Say the machine with each number** (e.g. "M1 16 GB", "Ryzen AI 9 365"). Replace old numbers with new
  ones instead of keeping both.
- **Cite by file**, e.g. `docs/design/llm-engine.md`, from code comments and other documents. There are
  no section numbers to cite.

## Other documents in docs/

| File | What it is |
|---|---|
| [../developer-guide.md](../developer-guide.md) ([日本語](../developer-guide.ja.md)) | Building apps with the `ai` packages: registering, preparing, running, joining models, constraints |
| [../app-api.md](../app-api.md), [../app-api-tasks.md](../app-api-tasks.md) | Design and task list of the `ai` library (Japanese) |
| [../kv-cache.md](../kv-cache.md) | Survey of KV cache memory and speed methods and how they map to the Neural Engine |
| [../model-fit.md](../model-fit.md) | Which models fit the Neural Engine, and methods to make them faster and smarter in less memory (survey and proposals) |
| [../model-table.md](../model-table.md) | Design of `models.yaml` (model definitions, selection, sources, per-NPU builds) |
| [../model-provenance.md](../model-provenance.md) | Model licenses and content, and the allowlist of models (`HWNPU_MODEL_ALLOWLIST`) |
| [../cache-sharing.md](../cache-sharing.md) ([日本語](../cache-sharing.ja.md)) | User guide to the model cache and sharing it |
| [../cache-sharing-design.md](../cache-sharing-design.md) | Design record of distributing converted models and sharing caches (layers, sources, locks) |
| [../prebuilt.md](../prebuilt.md) | Design for uploading converted models from a source (NAS) to Hugging Face and reading them back |
| [../prebuilt-reuse.md](../prebuilt-reuse.md) | Rebuilding Core AI prebuilt programs: taking unchanged programs from an earlier export (the build record build.json, the conditions, device marks; Japanese) |
| [../known-issues.md](../known-issues.md) | Known Core AI / coreai-build problems, hwnpu's workarounds, and what to do when Apple fixes them |
| [../reference-paper.md](../reference-paper.md) ([日本語](../reference-paper.ja.md)) | NPU papers drawn on, and whether they held up on hwnpu's hardware |
| [../amd-npu-models.md](../amd-npu-models.md) | Design of running converted models on AMD's NPU (IRON) |
| [../amd-xdna2-bench.md](../amd-xdna2-bench.md) | Data types and measured performance of XDNA2 (Strix Point) with IRON samples |
| [../intel-npu-models.md](../intel-npu-models.md) | Design of running converted models on Intel's NPU |
| [../model-estimation.md](../model-estimation.md) ([日本語](../model-estimation.ja.md)) | Record of models evaluated on the NPUs, including those left out of models.yaml, with the verdict and why |
| [../small-models-survey.md](../small-models-survey.md) | Survey of small (≤1B) non-chat models: pitch, singing, music, sound effects, pose |
| [../image-models-survey-qwen-chroma.md](../image-models-survey-qwen-chroma.md) | Groundwork for bringing Qwen-Image-2.1 and Chroma1 to AMD's NPU (both since removed: [../not-adopted/](../not-adopted/README.md)) |
| `../handoff-*.md` | Hand-off notes for checks on other machines (M1–M4, AMD, Intel): what to run and what to record |
