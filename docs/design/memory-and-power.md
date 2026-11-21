# Memory and power

[日本語](memory-and-power.ja.md)

How hwnpu measures energy and memory, what power control each NPU offers, how a model is run for a
machine's memory class, and how the NPU's resident limit is probed. The Go API (`WithPowerPolicy`,
`PowerSample`, `AveragePower`) is in [api.md](api.md); NE-specific memory limits in
[apple-neural-engine.md](apple-neural-engine.md); KV cache memory in [llm-engine.md](llm-engine.md).

## Power policy

`WithPowerPolicy(PowerDefault | PowerLow | PowerBalanced | PowerPerformance)`. A device without
`CapPowerControl` ignores it.

| Backend | Low / Balanced / Performance |
|---|---|
| Intel (Level Zero) | Queue priority LOW / NORMAL / HIGH, always. With the NPU queue extension also workload type BACKGROUND for Low and turbo for Performance (`turbo` in ext 1.0, `options` in 1.1) |
| AMD (amdxdna) | Hardware context QoS priority (unprivileged). The device-wide power mode (`SET_STATE`) needs root and affects every user, so it is left alone |
| Apple (Core ML, Core AI) | No effect (no `CapPowerControl`; only compute units can be chosen) |

## Energy meters

`Device.PowerSample()` returns cumulative joules; `Scope` is `npu` or `package` (whole CPU package).

| Backend | Source | Scope | Notes |
|---|---|---|---|
| Intel, Linux | RAPL `/sys/class/powercap/intel-rapl:N` | package | Usually needs root. Counter wrap handled |
| AMD, Linux | NPU sensor (`QUERY_SENSORS`) when the driver has one (AMD's DKMS driver); else amdgpu hwmon `power1_input` (PPT, µW, no root) | npu / package | Linux 7.0's in-tree amdxdna has no NPU sensor. Both are instantaneous: a background goroutine samples every 50 ms from the first `PowerSample` and integrates by trapezoids (`power.Integrator`), stopped on close |
| AMD, Windows (xrtshim) | XRT electrical info (watts) | — | `CapPowerMeter` when it reads |
| Apple | none in the API | — | `hwnpu-dev powerbench` (root) for measurements; `hwnpu status` shows IOReport power only with root |

- Tests: `Provider.Hwmon` moves the hwmon path; `xdnasim.NoSensors` simulates Linux 7.0.
- macOS IOReport energy (`Energy Model` ANE and `CPU Energy`) reads all 0 without root. `hwnpu status`
  says "power not readable without root" rather than 0.00 W ([cli.md](cli.md)).

### HWNPU_ENERGY (per-phase energy of a run)

- `HWNPU_ENERGY=SECONDS` (not a number: 5): `console.MeasureEnergy` at the start of `main` measures the idle
  power for that long (not counted in the run's times), then splits the package energy at each phase
  change (converting/loading/compiling vs running). Under the `took …` line it prints each phase's energy
  and that energy minus idle power × the phase's time (`energyLine`).
- Source: `power.PackagePower` (amdgpu PPT), so AMD APUs on Linux only for now; elsewhere it says so and
  does nothing. Called by hwnpu-image, hwnpu-chat, hwnpu-audio.
- Phase rules are those of the time line ([cli.md](cli.md)): loads and weight-bank re-reads
  (`iron.Graph.Use`, `console.Reading`) during the run count as preparing.
- Measured (AMD APU, Linux; one 1024² image, files in the page cache, idle 6.7–7.1 W):

| Model | Time (prep / run) | Energy (prep / run) | Minus idle (prep / run) | Bank re-reads |
|---|---|---|---|---|
| Anima-Turbo | 46 s (9.6 / 36) | 698 J (121 / 577) | 56 / 329 J | 55×, 30.5 GB, 4.2 s |
| Anima-Aesthetic | 3m18s (11 / 186) | 3.23 kJ (158 / 3.07 k) | 79 J / 1.82 kJ | 174×, 55.5 GB, 6.3 s |
| Chroma1-Flash | 4m22s (23 / 238) | 4.45 kJ (345 / 4.10 k) | 182 J / 2.49 kJ | 573×, 142 GB, 14 s |
| Chroma1-HD | 11m20s (32 / 648) | 11.89 kJ (493 / 11.39 k) | 277 J / 7.02 kJ | 953×, 234 GB, 22 s |

  - Idle is about 40 % of the running energy.
  - Preparing is 3–8 %: graph loads 4.5–8.8 s and bank re-reads from the page cache (10–18 GB/s), which grow
    with steps. Keeping all banks resident removes them but pins memory per bank
    ([image-amd.md](image-amd.md)).
- Other AMD figure: Gemma 4 E2B generation averages 11.8 W package, 1.93 J/token (idle 5–7 W included).

### powerbench (Apple)

- `hwnpu-dev powerbench` (darwin): asks the sudo password once, starts `sudo -n powermetrics -f plist` per
  measurement, runs each model continuously on its own thread (or `-rate` calls/s), and reports CPU / GPU /
  ANE power minus idle and mJ per call. A model that reaches the NE is also measured CPU-only (`-cpu=false`
  skips). `-decode 8,16` generates LLM decode models (energy per token).
- `HWNPU_COREML_UNITS=cpu|gpu` loads Core ML with `MLComputeUnitsCPUOnly` / `CPUAndGPU`; for measurement
  only, not public API. Sweeps: `tools/coreml/gen_sweep.py`.
- Result (M3): break-even about 200 MFLOP per call. Above it the ANE is 2–3× faster at about 1/4 the energy
  (≈1 TFLOP/J vs CPU 0.25 TFLOP/J). Below it Core ML's fixed cost per call (≈55 µs) and ANE startup win,
  and the CPU is better. Details: [MODEL_ARCHITECTURE.md](../../MODEL_ARCHITECTURE.md).

## Measuring memory

| What | How | Pitfall |
|---|---|---|
| Apple NE weights | `mem.NPUWired`: kernel wired memory of `com.apple.driver.AppleH11ANEInterface`, read with `mach_memory_info` (no root, purego) | Not in RSS, footprint or `aned`. Machine-wide (other processes mix in). Found by name: the vm tag number changed between macOS versions. Idle: 144–192 KiB. Programs Core ML just compiled come onto it at the first call, not at load (LFM3: 0.27 GiB after load, 2.43 GiB after one call): measure after a call |
| Apple per process | `proc_pid_rusage` flavor 6: `ri_phys_footprint` + `ri_neural_footprint`, sampled every 0.1 s, max of the sum | Per process, so other NE users do not mix in. mlocked file pages are not in the footprint: add wired (gpt-oss-20b with 6.4 GB of experts wired: footprint 0.53, wired 6.41, resident 9.53 GB). Resident counts unpinned page cache and shared libraries, not swap or compressed |
| A whole run's peaks | `go run ./tools/memwatch [-csv FILE] CMD …`: footprint, wired, `mem.NPUWired`, system free % and pressure every 0.1 s; prints peaks and models.yaml's `memory` | Breakdown: `footprint PID`, `vmmap -summary PID` |
| AMD pinned memory | Rise of `Unevictable` in `/proc/meminfo`; a process's share = sizes of its `/dev/accel/accel0` mappings in smaps; export `manifest.json` banks | Pinned pages are in neither RSS nor swap. Mostly weights and work areas; KV < 5 % |
| Windows | `K32GetProcessMemoryInfo` private bytes; `maxRSS` = peak working set | |
| During a run | `HWNPU_PROFILE=1`, hwnpu-chat's load log ("the Neural Engine holds …"), `hwnpu status` process table | |

- Value to record for a model: the second run (after conversion and compile); the first run's peak is the
  conversion's.
- After a conversion call `debug.FreeOSMemory` (Go was slow to return the checkpoint's pages: 2.4 GB →
  154 MB at startup).
- Core ML holds the same weights twice when two programs carry them; Core AI's multi-function `.aimodel`
  holds them once ([apple-coreai.md](apple-coreai.md)).

### Current values (M1 16 GB, macOS 27.2.1, Core AI)

`hwnpu-chat -m NAME --temp 0 -n 200 "…"`, second run. "Total" = max of NE + process, in GB.

| Model | Neural Engine | Process | Total | Conversion peak (process) |
|---|---|---|---|---|
| LFM3-1.7B (lut6) | 1.36 GiB | 0.23 GiB | 1.71 GB | — |
| same, M2+ form (`--tier 2048`) | 2.64 | 0.21 | 3.05 GB | — |
| Qwen3-0.6B (lut8) | 0.65 | 0.35 | 1.07 GB | — |
| same, M2+ form (`--tier 1024`) | 1.23 | 0.26 | 1.60 GB | 2.0 GiB |
| Qwen3-1.7B (lut8, with prompt function) | 1.80 | 0.36 | 2.32 GB | 3.1 GiB |
| same, `prefill: off` | 1.64 | 0.23 | 2.01 GB | 1.8 GiB |
| Qwen3.5-0.8B (lut8, with MTP layer) | 0.99 | 0.39 | 1.48 GB | 3.0 GiB |
| same, M2+ form (`--tier 2048`) | 1.84 | 0.31 | 2.29 GB | 3.3 GiB |
| Qwen3.5-2B (lut6) | 1.67 | 0.39 | 2.20 GB | 6.0 GiB |
| LFM3-VL-2B (lut8) text / one image | 2.14 / 2.24 | 0.23 / 0.25 | 2.55 / 2.67 GB | 2.9 GiB |
| LFM3-Audio-2B text / one clip | 2.05 / 2.18 | 0.23 / 0.21 | 2.44 / 2.57 GB | 2.8 GiB |
| LFM3-Omni-2B text / image and audio | 2.40 / 2.63 | 0.23 / 0.25 | 2.83 / 3.10 GB | 2.9 GiB |
| PLaMo 3 Translate 0.8B (lut8) | 0.82 | 0.12 | 1.01 GB | 2.3 GiB |
| PLaMo 3 Translate 2B (lut8) | 2.74 | 0.24 | 3.20 GB | 2.7 GiB |
| PLaMo-Embedding-1B (lut8) | 1.05 | 0.03 | 1.16 GB | 1.3 GiB |

- The process's steady part is 0.03–0.4 GiB (tokenizer, caches, the embedding rows read).
- The "M2+ form" adds the short-conversation unpaged program, which holds the weights once more
  ([llm-engine.md](llm-engine.md)). Measured on M1 by forcing `--tier`; not on M2/M3 hardware.
- LFM3 VL and Omni load their encoders from the start (Omni on Apple 2.8 GB before any image or audio).
- Conversion peaks at 2–3 GiB (Qwen3.5-2B 6.0 GiB), before anything is on the NE. Converting Qwen3.5-2B on
  an 8 GB machine swaps, so 8 GB machines rely on prebuilt programs ([distribution.md](distribution.md)).
- PLaMo 3 Translate 2B's NE share (2.74 GiB) is within the 2.86 GiB the M1 holds at full speed.
- AMD pinned memory per model: [amd-llm.md](amd-llm.md).

## Machine memory classes

How a model runs on a machine of some memory is the model definition's job (models.yaml), not the user's
per-flag choice. The user only chooses which memory class this machine counts as.

### In models.yaml

- `machine`: the smallest class the model is offered for. A family or use name (`lfm3`, `qwen3.5`) picks by
  the machine's memory.
- `memory`: GB the model takes; per vendor and per chip (`chat.PerVendor`):
  `memory: {amd: 2.2, apple: 3.1, M1: 1.7}`. On Apple's NPU the chip's key (`chat.Chips`: M1–M5, the same
  names as `not_on`) wins. `hwnpu-chat list` shows it; comparing candidates of one family (`Model.Memory`)
  ignores the chip.
- `variants: {16: {…}}`: per-memory differences of the run fields (`chat.Run`: `prefill`, `format`,
  `context`, `paged`, `tier`). `tier` takes the same values as `--tier` (`auto`, `none`, lengths).
- The 8 GB target is about 3 GB per model.

| Rule | Models | Reason |
|---|---|---|
| No short-conversation unpaged program on 8 GB (`tier: none`; `variants: {16: {tier: auto}}`) | LFM3-1.7B, Qwen3.5-0.8B | It holds the weights again: LFM3-1.7B 1.7 → 3.1 GB, over the 8 GB target |
| Unchanged (keeps the extra program on 8 GB) | Qwen3-0.6B | 1.6 GB with it, inside the target; about 2× faster on M3 (68 vs 34–40 tok/s); `tier` is not per vendor and would also remove Intel's 4096 tier |
| No prompt program below 16 GB (`prefill: off`) | Qwen3-1.7B | Smaller default on 8 GB; Core AI saves only 0.3 GB now. `--prefill on` for speed |
| `machine: 8` | Qwen3.5-2B | 2.2 GB; `-m qwen3.5` picks 2B on 8 GB (`qwen3.5:0.8b` for 0.8B) |
| M1 never loads the short-conversation program | all | Checked at run time (`chipName`) |

8 GB Apple now: LFM3-1.7B 1.7 GB, Qwen3.5-0.8B 1.5 GB, `-m qwen3.5` → Qwen3.5-2B 2.2 GB. Running several
at once on 8 GB: LFM3-1.7B + PLaMo 3 Translate 0.8B (Apple 2.7 GB, AMD 3.2 GB).

### Choosing the class

- `hwnpu memory LFM3-1.7B 8` (no args: list; `--forget`). `hwnpu-chat --memory 8` or a memory tag
  (`lfm3:8g`, `:8GB`, `:8gb`) also keep the choice once the model is prepared (`keepMemory`). API:
  `modelstore.Store` `Memory`, `SetMemory`, `Memories`.
- Kept in `run.json` in the settings folder beside the user's models.yaml (`chat.RunFile`,
  `{"models": {"LFM3-1.7B": {"memory": 8}}}`). Not in the cache (`hwnpu cache prune` would drop it, and a
  shared `HWNPU_CACHE` would carry it to other machines). Not in models.yaml (machine edits would break
  comments). `go test` reads and writes it only with `XDG_CONFIG_HOME` set.
- Applied in `lookupOn` (Resolve, ResolveFor, ResolveForKinds, `serve`, `modelstore`): every command that
  resolves the model runs it so and converts or fetches only that class's programs. One log line at start:
  `LFM3-1.7B: runs as on a machine of 8 GB, as chosen (…/run.json; hwnpu memory LFM3-1.7B --forget forgets it)`.
- Precedence: this run's memory tag or `--memory` → chosen memory → `HWNPU_MEMORY` (whole run, not kept) →
  the machine's memory.
- A family name picks the model by the machine's memory (the choice is per model, unknown before picking);
  the picked model then runs in its chosen class.
- `--tier`, `--prefill`, `--format`, `--context`, `--paged` are not kept: this run only; otherwise the
  definition's values for the class.
- Programs of the previous class stay in the cache until LRU removes them; `hwnpu memory` suggests
  `hwnpu cache clear NAME`.
- Only chat models have `variants` now; image and audio models get the same handling when they add them.
- Within a run the KV cache is capped at 1/4 of physical memory ([llm-engine.md](llm-engine.md)).

## Residency probe (`hwnpu-dev probe resident`)

Measures how much the NPU holds at once. Usage: [PROBE.md](../../PROBE.md).

- Loads programs of `-resident-size` GiB (comma list; whole layers of 61 M weights; default 0.25) one by one
  and runs each once, until a load fails or `-resident-max-gb` (60 % of RAM, at most 16). Then runs all
  again. On out of memory (`types.IsOutOfMemory`) closes the oldest and retries the failed load.
- `-resident-hold open,close`: keep all open, or close each before the next (limit on held vs loaded).
  `-resident-formats int8`: is the limit in weight bytes. `-resident-run=false`: load only (does the crash
  come from load or run).
- `-resident-cycle` (default, with open): after each load, run all held programs in turn twice and record
  the second pass's mean and max (`cycle_ms`, `cycle_max_ms`). Slow = over 2× the best mean (`base_ms`) and
  ≥ 0.5 ms slower. Reports `fast_bytes` (largest total before slow) and `slow_bytes`; after 3 slow in a row
  the cycling stops (each pass would reload everything) but loading goes on. Needed because Apple evicts
  without an error and reloads on the next call: the limit shows as slowness, not failure.
- On Apple, `npu_bytes` = rise of `mem.NPUWired` since before the first load, after each load and cycle;
  `held` is only the sum of weights.
- The child prints a `npuprobe-progress` line per program, so the parent reports what loaded even when the
  child crashes (`crashed`); each stage (load, run, close) goes to stderr, and the failed stage is the last
  line of the error.
- Windows: process private memory after each load.
- Tested against fake NPUs (OUT_OF_HOST_MEMORY by capacity, memory not returned on close, fast only for the
  last 4 held).

Known results:

| Machine | Result |
|---|---|
| Intel Core Ultra 5 135U | Anima fp16 failed at about 4.6 GB with `ERROR_OUT_OF_HOST_MEMORY`; yet 0.23 GiB programs stacked to 16 GB without error. A probe run once ended with 0xc0000005 inside the driver (hence the progress lines). `resident-large` (0.57,1.2 GiB, open, to 8 GiB) separates per-program size from total |
| M2 16 GB, macOS 27.2 | 42 × 0.23 GiB (9.52 GiB) without error, open or closed; each synthetic program compiled for the NE (1.8–5.3 s a load) |
| M4, Intel | No error up to 16 GB |
| M1 | Parts run in turn: full speed until the NE holds 2.86 GiB, about 12× slower per call from 2.97 GiB. M2 full speed at 3.65 GiB; M3, M4 no such step seen ([apple-neural-engine.md](apple-neural-engine.md)) |

## Not adopted

- AMD device-wide power mode (`SET_STATE(SET_POWER_MODE)`): root, and it changes the NPU for every user.
- Selecting the 8 GB form of a model by an env var (`HWNPU_MEMORY`) as the user's means: the model selector
  (tag) and `hwnpu memory` instead; `HWNPU_MEMORY` stays as a whole-run override only.
- Keeping the memory choice in the cache or in models.yaml (see above).
- Users choosing `--tier` / `--prefill` per model to fit memory: the definition decides per class.
- Dropping the short-conversation program for Qwen3-0.6B on 8 GB (fits; 2× faster on M3).
- Zero-padded inputs to amortize ANE call cost: no gain at the sizes measured on M3.

## Open

- Intel: whether the power policies change measured power on hardware (`hwnpu-dev report -power low` vs
  `-power performance`).
- NPU-only power on Intel, Windows and Apple (no API without root).
- `HWNPU_ENERGY` on machines other than AMD APUs (RAPL, Windows).
- M2/M3 hardware for the "M2+ form" memory and 8 GB speed without the extra program (on M3 with it was 8 %
  faster).
- Not measured on Core AI: Gemma 4 E2B process share, Gemma 4 E4B, Qwen3.5-4B, PLaMo 3 Translate 8B
  (expected NE 3.81 GiB + process ≈ 0.3 GiB), d1.
- `probe resident -resident-cycle` on hardware: an M1 `slower from` near 2.9 GiB would confirm the method;
  M2/M4 limits; where `npu_bytes` departs from the weight sum.
- If out of memory on Intel is a crash rather than an error, closing on the error does not work: a self
  imposed limit below the crash point is needed.
