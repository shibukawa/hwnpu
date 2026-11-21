# Image generation models

[日本語](image-models.ja.md)

The image-generation models hwnpu runs (`hwnpu-image generate`), and the pipeline rules shared by every NPU: samplers, guidance, editing, batching, sizes and exports. How each NPU runs them: [AMD](image-amd.md), [Apple and Intel](image-apple.md). Command flags: [cli.md](cli.md). Downloads and sources: [distribution.md](distribution.md).

## Pipeline

A picture is three parts, all on the NPU: a text encoder reads the prompt, a DiT denoises the latents for N steps, a VAE decodes them. The tools ComfyUI and diffusers split a UI/workflow from the model run; hwnpu does only the model run.

- A picture's time is the DiT's passes: steps × passes a step (2 with classifier-free guidance) × one pass. The text and the VAE take seconds. Cut the number of passes first, then a pass's time.
- Judge every change from the same noise against the CPU float32 reference (`tools/ref/ref_*.py`): the final latent's cosine and the picture's PSNR, on several prompts × seeds; look at the pictures too. One seed's cosine varies a lot.
- Packages: `internal/imagegen/anima`, `chroma`, `flux2klein`, `flux3klein`; `internal/imagegen/aspect` (sizes); `internal/imagegen/milpart` (Core AI parts).

## Catalog (models.yaml)

| Name (family:tag) | Parts | Steps / guidance | License, content | `machine` | Memory AMD / Apple (GB) | NPUs, time |
|---|---|---|---|---|---|---|
| **FLUX.2-klein-4B** (`flux2-klein:4b`) | 3.9B DiT of FLUX's shape (5 double, 20 single blocks), Qwen3-4B, FLUX.2 VAE | Euler 4, none | Apache-2.0, safe | 8 | 3.9 / 4.9 | AMD 1024² ~29 s, 512² ~10 s; edits: AMD 1 min, 37 s with a half-size reference; Apple M3 89 s at 1024², 39 s at 512² (the edit's programs). Apple M1 1024² ~41 s, 512² 11 s |
| flux-3-image-klein (`flux3-klein`) | 5.5B DiT (each stream's mode blocks, then single blocks), Qwen3-VL-4B, FLUX.2 VAE | 16 flow-map steps, none | FLUX Kommunity (non-commercial, gated; asked once) | 16 | 3.9 / 5.5 (Apple an estimate) | AMD 1024² ~1 min 45 s, 512² 32 s; Apple M3 1024² ~2 min (lut8). Text to image only |
| Anima-Turbo (`anima-turbo:v1.1`) | Cosmos-Predict2 2B DiT, Qwen3-0.6B + LLM adapter, Qwen-Image VAE | Euler 8, none (distilled) | non-commercial weights (pictures may be used commercially; asked once), nsfw | 8 | 3.5 / 4.8 | AMD 1024² ~32 s; Apple M1 48 s. Intel: the only image model with an Intel form |
| Anima-Aesthetic (`anima:aesthetic`, the family's default) | Anima-Turbo's programs, other weights | DPM-Solver++ 2M 25, CFG 4 at σ ≥ 0.6 | as Anima-Turbo, nsfw | 8 | 3.5 / 4.8 | about 4.5× Anima-Turbo; AMD 1024² 2 min 22 s, 2.71 kJ |
| Chroma1-Flash (`chroma:1-flash`) | FLUX.1 schnell's 8.9B DiT with an FFN for its modulations (approximator), T5-XXL, FLUX VAE | Heun 8 (15 DiT calls), none | Apache-2.0, nsfw ("fully uncensored") | 16 | 3.9 / 4.3 (+6.9 GiB on the Neural Engine) | AMD 1024² 3 min 14 s, 512² 1 min; Apple M1 1024² ~9.5 min |
| Chroma1-HD (`chroma:1-hd`) | the undistilled Chroma1 | DPM-Solver++ 2M 25, CFG 3 at σ ≥ 0.6 | as Flash | 0 (by name only) | as Flash | AMD 1024² 8 min 34 s |

- T5-XXL and the VAE of Chroma1 are FLUX.1 schnell's byte for byte, shared when present; Chroma1-HD's export reuses Flash's.
- Anima's VAE, tokenizers and configs come from `circlestone-labs/Anima-Base-v1.0-Diffusers` (`with: diffusers`); the field `file` picks which checkpoint of the repository a model is (`--dit`'s default). Anima-Base's checkpoints start with `net.` (not `model.diffusion_model.`); `iron_anima.py`, `ref_anima.py` and `mil.OpenWeights` still tell them apart.
- Licenses with `accept: true` are asked once on the terminal and kept in `licenses.json` beside the user's config; without a terminal, `--accept-license` or `HWNPU_ACCEPT_LICENSE=ID` (this run only) (`chat.AcceptLicense`).
- `content: [nsfw]` models (Anima, Chroma) need `+nsfw` in `HWNPU_MODEL_ALLOWLIST` to run by name.
- Memory = the peak of the NPU's memory plus the process's (AMD: `drm-total-memory` of the accel fd + `RssAnon + RssShmem`; Apple: `ri_phys_footprint + ri_neural_footprint`), measured at 1024², the second run. The peak is usually the VAE's last stages. See [memory-and-power.md](memory-and-power.md).

## The main model: use `image`

`image` (alias `generate`, `hwnpu-image generate`'s default) picks **FLUX.2 klein 4B from 8 GB**, on every NPU.

- Why: Apache-2.0, does not draw NSFW unless asked, edits on AMD, about Anima-Turbo's memory (4.9 vs 4.8 GB on an M1) and faster (41 vs 48 s).
- On an 8 GB machine (about 3 GB free for a model) 1024² peaks over that; `--size 1:1s` (512², 3.1 GB on Apple, 1.9 GB on AMD) fits.
- Anima and Chroma are never picked, even with `+nsfw`: named, they run.

## Samplers and steps

| Model | Default | Flags |
|---|---|---|
| Anima-Turbo | Euler 8, shift 3, guidance 1 (1.5 with a `--negative`) | the distillation is Euler's |
| Anima-Aesthetic | DPM-Solver++ 2M 25, CFG 4 at σ ≥ 0.6 (17 of 25 steps guided) | the model card's: `--sampler euler --steps 30 --cfg-sigmas 0,1` |
| Chroma1-Flash | Heun 8 | `--sampler euler` (8 calls) |
| Chroma1-HD | DPM-Solver++ 2M 25, CFG 3 at σ ≥ 0.6 | written in the export's `model.json` (`sampler`, `steps`, `cfg_sigmas`) |
| FLUX.2 klein | Euler 4 | — |
| FLUX.3 klein | 16 flow-map steps (`flux3klein.Steps`) | — |

**DPM-Solver++ 2M** (`--sampler dpm2m`, `anima.Sampling`): diffusers' DPMSolverMultistep on the flow's σ. Data prediction D = x − σv, λ = log((1 − σ)/σ), h = λ_t − λ_s; x_t = (σ_t/σ_s) x_s + (1 − σ_t)(1 − e^−h) D; second order with the previous D and r = h_prev/h: (1 + 1/2r) D_0 − (1/2r) D_1. The first step (from σ = 1) and the last (to 0) are first order, the same as Euler. One DiT call a step, as Euler. `TestDPM2M` checks it on Gaussian data's exact field.

Measured (Anima-Aesthetic, 512², AMD, 6 pictures against 100 Euler steps): Euler 30 0.982 / 25.2 dB in 74 s; DPM 25 0.984 / 24.3 dB; DPM 20 0.970; DPM 15 0.920 (details break). DPM 25 with CFG at σ ≥ 0.6: 0.984 in 53 s. At 1024² against Euler 30 with every step guided: −29 % time and −29 % energy (balanced power profile). Judge a step count against many Euler steps, not only the model card's count.

## Classifier-free guidance

- `--guidance SCALE`, `--negative TEXT`, `--cfg-sigmas LO,HI` (guide only steps with σ in [LO, HI]; `0,1` every step). Default `0.6,1` for the guided models: the steps under σ 0.6 run the DiT once. The same pictures (latent cosine 0.998 to every step guided, Anima); at σ ≥ 0.8 colours fade and small objects go.
- The conditional and unconditional passes run **back to back in each weight group** (the engine's `velocity`, every backend), so a group's weights are read once a step, not twice. The timestep/patch embedding (`dit_in`) runs once.
- The prompt and the negative are encoded in one load of the text encoder (`EncodeAll`).
- **The empty negative is a constant**: diffusers runs an empty prompt as masked tokens; what is left is one row from the weights alone, computed at export (Anima `text/empty_cond.bin`, cosine 1.0000 to diffusers; Chroma on the Mac `text/empty.bin`, its 2 rows through T5 in float32).

## Reference-image editing (FLUX.2 klein)

`hwnpu-image generate -m flux2-klein -i fox.png -i room.png@half "The fox from image 1 ... the living room from image 2."`

- `-i`/`--image` up to 4 times (BFL's limit for klein); the prompt calls them image 1, image 2, … in order. A reference is centre-cropped to the picture's shape and scaled to its size; `FILE@half` reads it at half the picture's sides (a quarter of its rows; `@full` is the default). A file actually named `FILE@half` is read as is.
- A reference's latents (the VAE encoder's mean, patched, normalized by the batch-norm stats) follow the picture's rows; the RoPE's first axis is 10, 20, 30, 40. The modulations are the picture's; only the picture's rows are read at the end.
- Half size changes the subject's fine detail (fur, face flatter: the CPU float32 shows the same) but not a composition reference. A subject at full size with the room at half: as detailed as two full-size, 1 min 41 s → 1 min 08 s.
- The step time follows the rows; the export holds the reference combinations of its default (see [image-amd.md](image-amd.md)). An unheld combination fails with the list it can run (`References`).
- On Apple's Neural Engine an edit runs the edit's programs, an export of their own converted at the first edit ([image-apple.md](image-apple.md)). The other models draw from text only.

## Several pictures: `--batch`

With several prompts, `--batch N` (default 4) draws N together; `--batch 1` one after another. The pictures are byte for byte the ones drawn alone, written when the last is done.

| | Text | Steps | VAE |
|---|---|---|---|
| Apple (Core AI): FLUX.2 klein, Chroma | the N prompts in one load | N together, a weight group once for all | each stage over the N pictures (`runner.Stages`) |
| AMD: FLUX.2 klein | all prompts in one load | a group's weights once for all the pictures' rows | a picture at a time |
| Anima | not batched: one prompt, a group run back to back is no faster | | |

- A part loaded once and run N times pays its slow first call once. Gains are large at 512² (Chroma on an M1: 1 min 24 s → 1 min 18 s a picture) and 1–3 % at 1024² (FLUX.2 klein on AMD: 512² 8.1 → 7.3 s a picture, 1024² 3 %).
- `runner.Stages`: load a VAE stage once, run every picture through it, then the next. Between stages the host holds float32 (up to 1.2 GB a picture at 1024²), so from the first stage whose outputs pass `StageBudget` (1.25 GiB) it runs as many pictures as fit. Holding every stage loaded and calling them in turn is much slower (2.5 → 16.9 s, measured on a 13-stage VAE).
- AMD keeps the VAE a picture at a time: its stages pass their outputs on the device; batching would copy them to the host for 0.5–0.8 s a picture and raise the peak.
- `debug.FreeOSMemory` before each picture's decode keeps the peak at one picture's.

## Sizes and aspect ratios

`--size` takes eight names (`internal/imagegen/aspect`): `1:1`, `1:1s` (the small square), `2:3`, `3:4`, `9:16`, `3:2`, `4:3`, `16:9`, or `N` / `WxH`.

| Name | `aspect.All` (multiples of 16) | FLUX.2 / FLUX.3 klein, AMD | FLUX.2 klein, Apple |
|---|---|---|---|
| 1:1 | 1024 × 1024 | 1024 × 1024 | 1024 × 1024 |
| 1:1s | 512 × 512 | 512 × 512 | 512 × 512 |
| 2:3 / 3:2 | 832 × 1248 | 832 × 1248 | 768 × 1152 |
| 3:4 / 4:3 | 880 × 1168 | 880 × 1168 | 768 × 1024 |
| 9:16 / 16:9 | 752 × 1360 | 768 × 1344 | 704 × 1280 |

- Every size but 1:1s is about 1.0 M pixels, so a VAE's middle attention stays at most 16384 pixels a side.
- The sizes are models.yaml's `npu.<vendor>.sizes` (a generation may override them), not the exporter's defaults. The command reads them (`npuAspects`) both to parse `--size` and to make the export, so `--size 16:9` draws what the export holds. Models without `npu.*.sizes` use `aspect.All`. (`aspect.All32` remains for 32-pixel models; none is in models.yaml now.)
- FLUX's 16:9 is 1344 × 768: ratio 1.750 (−1.6 %), 1.03 M pixels, sides multiples of 64. Apple's FLUX.2 sizes are the ones the Neural Engine compiles (1344 × 768 and 832 × 1248 fail there; the cause is on the Core AI side, see [image-apple.md](image-apple.md)).
- Check every size's picture once on the lowest machine (bands, seams, noise, blank): a broken function for one size shows only there.

## Exports: one for all aspect ratios

**The rule (every image model, every NPU)**: one export per model and NPU vendor holds **all** the aspect ratios of models.yaml, the weights stored once. Never one export per size, nor a subset (`512+1024`) with the others apart. A user who switches `--size` must not download the model again. A missing size is a bug in the exporter; an exception needs a measured reason written in the design docs and in the model's models.yaml entry.

- **AMD**: the first size's graphs hold the weight banks; the other sizes' graphs (in `sizes/WxH/`, or named by their rows) point at the same banks (`share_banks`). See [image-amd.md](image-amd.md).
- **Apple (Core AI)**: a part whose graph depends on the size (the DiT's groups, its last layer, the VAE's stages) is one `.aimodel` with a function per size (`s{W}x{H}`, `llm.WriteAIFunctions`); a part that does not (text encoder, time embedding) is one function. The manifest's `sizes` lists each size's parts; `ProgramManifest.At` picks `--size`'s. The number of parts must not depend on the size (check `VaeStages` for every size). See [image-apple.md](image-apple.md), [apple-coreai.md](apple-coreai.md).

What it saves: Anima's per-size Apple exports were 2.9 GB each with the weights repeated (23 GB for 8 sizes, 46 GB for Turbo and Aesthetic). FLUX.2 klein on AMD: 22.68 → 11.77 GB for 1:1 and 1:1s, 16.0 GB for all eight; the page cache holds half. A picture's time is the same.

What it costs (accepted: the repeated download is worse):

- Core AI compiles every function of an `.aimodel` together: the first run compiles all sizes (FLUX.2 klein on an M1: 10 min for 512², 40 min for 512² and 1024²). With a prebuilt only the finish is local, but the VAE uses no prebuilt, so its sizes compile on every machine's first run.
- One size the Neural Engine compiler rejects takes **all** sizes of that part off the NPU. Before putting an export in the sources, compile each size alone offline per chip (`coreai-build compile … --architecture h13g/h14g/h15g/h16g`, then look for `Falling back` in `/usr/bin/log`); the per-chip prebuilt (`hwnpu-dev models prebuild --arch ARCH EXPORT`) must end with `0 failed`.
- Adding a size remakes the export and every prebuilt.

### Names and keys

| | Name |
|---|---|
| AMD | `NAME.amd-npu2`, the model's name in lower case: `anima-turbo`, `anima-aesthetic`, `chroma1-flash`, `chroma1-hd`, `flux.2-klein-4b`, `flux-3-image-klein`. Anima's (6.13 GB) and Chroma's (18.85 GB) are packed in the sources (`files.tar.zst`). Size names (`-512`, `-aspects`) are gone |
| Apple | the export's name carries its sizes by `aspect.Tag` (`8sizes-` + the sizes' CRC-32), so another set of sizes is another export |

- The key: `chat.PinnedRecipe(model, tool, version)` plus the exporter's arguments (`AMDRecipe`); it needs every read file's `sha256:` in models.yaml (no pins, no sharing). Raise `AMDExportVersion` (FLUX.2 and FLUX.3 klein: 2) by hand when the exporter's input or output changes (sizes, a format, a layout). The bundled `tools/amd` hash is in the key, so any change there misses the recipe entry; an export imported by name (`iron-import`, `hwnpu-dev models import`, `chat.ImportedRecipe`) is found by the name alone, so a changed export needs a new name or the old one removed from the sources.
- The first run puts the export in the sources (`chat.SharedExportOr`, written back to `?writeback=true` sources). A local export of the same name that lacks the asked size is moved to `NAME.amd-npu2.replaced`, the sources' current one fetched (or IRON exports it), and moved back if neither works.
- Where each model comes from on the first run: FLUX.2 and FLUX.3 klein on AMD: local → sources → exported with IRON (`PrepareAMD`); Anima and Chroma on AMD: local or the sources; Apple and Intel: downloaded and converted. Details: [distribution.md](distribution.md).
- Apple's per-size exports are being moved to the one-export form in this order: Anima-Turbo → Anima-Aesthetic → FLUX.2 klein (its eight `npu.apple.sizes`) → Chroma1 (Flash, HD) → FLUX.3 klein; chips M1, M2, M3 (h13g, h14g, h15g) first.

## Open

- Apple's move to one export for all sizes (order above); FLUX.3 klein on Apple has 1024² only.
- Editing on Apple, editing with FLUX.3 klein.
- Energy per picture: FLUX.2 and FLUX.3 klein on AMD; every model on the Mac (`powerbench` needs sudo) and Intel.
- Intel: only Anima has a form, and its run on the hardware is not checked.

## Not adopted

| What | Why |
|---|---|
| Sana 600M, SANA 1.5 1.6B (removed) | dropped with Qwen-Image; FLUX.2 klein took the 8 GB slot. `internal/imagegen/sana`, `iron_sana.py`, `iron_dwconv.py` removed; the parts Anima/Chroma/FLUX used moved to `tools/amd/iron_dit.py`. Kernels such as `kernels/dwconv3.cc` stay: the chat exports' key hashes every kernel (`amdtools.LMHash`) |
| Qwen-Image-2.1, -Turbo (removed) | 7B DiT and Qwen3-VL-8B: too many parameters to use on an NPU (17 GB of programs past the Neural Engine's compile cache; 1024² ~1.5 min on AMD at best) |
| Sana-Sprint 0.6B / 1.6B (removed) | 2 steps and fast (12–32 s at 1024²) but much worse pictures than Sana. Exporters refuse `guidance_embeds` checkpoints |
| Anima-Base v1.0 (removed) | its use is as a LoRA base; hwnpu loads no LoRA. Anima-Aesthetic has the same structure and time. `-m anima:base` fails (`TestImageMemoryTags`); `iron_anima.py --dit anima-base-v1.0.safetensors` still exports |
| Per-size exports, size subsets | repeated downloads (above) |
| LongCat-Image-Edit, OmniGen2 (as edit models) | LongCat: 6.3B DiT + 16.6 GB text encoder; OmniGen2: lower quality |
| Reusing the references' K/V from the first step (FLUX.2 klein 9B-KV's way, on the 4B weights) | latent cosine 0.46 to the right edit: needs a model trained for it |
| First-Block Cache (reuse later groups' residual) | apart for cond/uncond the guidance amplifies the mismatch (colours fade); together 0.02 saves 12 % at 0.989, 0.05 changes the person. CFG at noisier steps saves as much with less change |
| CFG dropped at the noisiest steps (the paper does both ends) | compositions change, colours fade (0.79) |
| The cond and uncond passes batched into one call | the GEMMs already have 4096 rows |
| DPM-Solver++ 15 steps | details break (0.920) |
| The GPU beside the NPU | the policy: NPU only |
| Chroma `--text-rows 256` as default | −7 % at 512², exact only for prompts ≤ 255 tokens |
