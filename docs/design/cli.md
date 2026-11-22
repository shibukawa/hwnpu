# Commands

[日本語](cli.ja.md)

How the commands are designed: the families, argument parsing, the terminal display, Ctrl+C, and the
`hwnpu` housekeeping command. For usage see [CHAT.md](../../CHAT.md), [USAGE_IMAGE.md](../../USAGE_IMAGE.md),
[USAGE_AUDIO.md](../../USAGE_AUDIO.md), [USAGE_DEV.md](../../USAGE_DEV.md), [USAGE_LLM_AMD.md](../../USAGE_LLM_AMD.md)
and [PROBE.md](../../PROBE.md).

## Command families

Commands are grouped by use. Each family is one binary with subcommands. Old single-purpose binaries
were removed without wrappers.

| Binary | Subcommands | Notes |
|---|---|---|
| `hwnpu` | `status`, `issue`, `cache` (`ls`, `prune`, `clear`, `clean`), `memory` | Looks after the machine: the NPU, the cache, the memory class, bug reports |
| `hwnpu-chat` | `chat` (default), `agent`, `search`, `serve`, `opencode`, `systemone`, `list` | Language models: chat, coding agent, embeddings, OpenAI-compatible API, d1 decisions |
| `hwnpu-image` | `generate`, `depth`, `face-detect`, `pose`, `detect`, `ocr`, `label`, `embed`, `segment`, `convert`, `list` | Image generation and vision models |
| `hwnpu-audio` | `generate` (music; `-m sfx` sound effects), `tts`, `transcribe`, `pitch`, `notes`, `list` | `list` covers music, sfx, speech, transcribe, pitch, notes |
| `hwnpud` | `serve` (default), `install`, `uninstall`, `status`, `stop` | The per-user process that owns the NPU and the models; `hwnpu-chat` goes through it with `HWNPU_DAEMON=on` ([hwnpud](hwnpud.md)) |
| `hwnpu-dev` | `check`, `convert`, `generate`, `formats`, `irdiff`, `info`, `probe`, `powerbench`, `report`, `mxfp4`, `xclbin`, `models` (`fetch`, `package`, `push`, `hf`), `sources` (`list`, `ls`, `import`, `update`, `verify`, `gc`) | Development tools, not for end users |

- Music and sound effects share `hwnpu-audio generate`: only the model differs, the output (48 kHz
  stereo WAV, speaker) and display are the same. Music-only flags (`--codes`, `--continue`, `--notes`)
  are an error with a sound-effect model.
- The 8 GB form of a model is picked by the model selector (models.yaml entry or memory tag), not by an
  env var. See [memory-and-power.md](memory-and-power.md).

## Argument parsing (kong)

- All user commands use `github.com/alecthomas/kong`: one struct per subcommand, fields with tags,
  `--help` for free. Flags have two dashes (`--model`/`-m`, `--out`/`-o`); prompts and files are
  positional.
- Shared conventions: `-m/--model` takes a models.yaml name, family or use (`chat`, `translate`,
  `plamo3translate:2b`; a family alone picks for the machine's memory), a Hugging Face `OWNER/NAME`, or an
  export directory (a value with `/` or a leading `.` or `~`). Each family has `list` instead of a
  `-list` flag.
- Flags that take programs converted by hand are in a "Converted by hand" help group. A checkpoint
  directory is `--checkpoint DIR` (not `-model`, which clashes with `-m`). The audio encoder is
  `--audio-encoder`.
- Flags that only fall back to the model definition are pointers (`Tier *string`, `Think *bool`), so
  "not given" differs from an empty value.
- `hwnpu-dev` is a kong shell with `passthrough:"all"`: each tool is a package `cmd/hwnpu-dev/NAME` with
  `Main`, keeps its std `flag` flags, and gets `os.Args` and `flag.CommandLine` replaced before `Main`.
  `probe` runs itself as a child process, so `probe.ChildArgs` (`["probe"]`) is put in front.
  `hwnpu-dev models` is a real kong subcommand.
- `hwnpu-dev models` gathers what works on the models' conversions here, in the sources and on Hugging
  Face, with the same commands and flags on every OS; what runs behind them differs. Each conversion is
  named `MODEL/TARGET` as the sources' folders are (`converted/MODEL/TARGET`): `amd-xdna2`, `coreai` (the
  generic `.aimodel` exports) and `coreai-ARCH` (those compiled for an architecture of the Neural Engine,
  `h14g`...; `coreai-all` for every one). `fetch` brings them here, `package` makes them, `push` puts in
  the sources what was made by hand (what `package` makes goes there itself). A command that needs what
  this OS lacks says so and stops (`coreai-ARCH` needs macOS's `coreai-build`). `package MODEL[/TARGET]`
  makes a model ready for a target as the apps' first run there does
  (`airole.Role.Prepare`, the role found by the model's kind: `airole.For`), on any machine: the programs
  are converted or taken from the sources, not loaded, and what is made goes in the sources written back
  to. Core AI's exports differ by chip on the M1 alone, so `coreai` is made as on an M1 and as on an M2
  (`chat.ExportChips`, `chat.ActAsChip`); `--chip` and `--memory` narrow it. For `coreai-ARCH` the exports so made
  are then compiled (`coreai.Build`); a Core AI export folder (`DIR[/coreai-ARCH]`) is compiled alone. `fetch MODEL[/TARGET]` runs the same preparation
  converting nothing (`llm.WithFetchOnly`): what is neither here nor in the sources stops where its
  conversion would start (`llm.MayConvert`, in `StartExport`/`StartProgram`, chat's `convert` and
  `convertShared`, the tables, `SharedExportOr` and the image exports) and is listed. It brings the
  checkpoint too (the files the sources' lock names, else a chat model's config, tokenizer and
  weights; `--no-checkpoint` leaves it), so that conversions can be tried on the model afterwards.
- `hwnpu-dev sources` is the upkeep of the sources: `list` (alone: the root and the sources in effect),
  `ls`, `import` (folders made by hand), `update` (locks to a new commit), `verify`, `gc`. What a cut-off
  writeback left is `hwnpu cache clean`'s. `hwnpu-dev cache` and `hwnpu-dev prebuild`, the names of
  before, were removed on 2026-11-22, with `flush` (every command waits for the queue), `gc --orphans`
  and `migrate` (the NAS was moved).
- `hwnpu-dev report` is the hardware check users run for an issue: test models embedded from
  `internal/fixtures` (written to a temp dir, as Core ML needs real files); the same 64×512 · 512×512 + bias
  + ReLU in Core ML and OpenVINO IR to compare vendors; NPU tolerance 0.05 absolute and 2 % relative (fp16
  error is 0.028); checks enumeration, strict compile, results, compile and run times, and that an fp32
  Core ML model is refused in strict mode; always prints the Level Zero loader result (on Windows, with no
  `/dev/accel`, it is the only sign of an Intel NPU). `-power` and `-cache` add energy per call where a
  meter exists.
- `aiecc` calls the `AIE_XCLBINUTIL` binary with no subcommand, so when the executable is named
  `hwnpu-xclbin` or `xclbinutil` it runs as `xclbin` (`go build -o ~/.local/bin/hwnpu-xclbin ./cmd/hwnpu-dev`).

## hwnpu-chat

### Model selection

- `-m` names the model; its use (chat, embedding) and features come from the model. `list` prints the
  catalog (name, use, repository, size, memory, features, downloaded).
- Features follow Ollama's capability names: `tools`, `thinking`, `vision`, `query` (separate query and
  document embedding). Each feature needs a one-line description in `chat.Features` (`TestLookup` rejects
  one without). `/v1/models` returns them as `capabilities`.
- A repository not in the catalog: only `config.json` is fetched first and the architecture decides the
  use (causal LM → chat, `plamo-bi` → embedding, else refused). Its features are unknown and not shown.

### Terminal

- `chat` with a prompt answers once on stdout and exits; without one it is a conversation. With thinking,
  the thinking goes to stderr (compare stderr when checking generated text).
- `search`: each line typed is a document; a line starting with `?` is a query, and documents are listed
  by cosine similarity.

### OpenAI-compatible server (`serve`)

- `GET /v1/models` (all models, with `kind` and `capabilities`), `POST /v1/chat/completions` (SSE,
  `reasoning_content`, `tool_calls`, `stop`, `stream_options.include_usage`,
  `usage.prompt_tokens_details.cached_tokens`), `POST /v1/embeddings`.
- Any number of models; requests are routed by `model`. If only one model has that use, any name works
  (`"model":"lfm3"`). Several and an unknown name: 404. A name of the wrong use (an embedding model on
  `/v1/chat/completions`) is an error, so that a client that picked an embedding model notices.
- Requests are handled one at a time.
- `reasoning_effort` none, minimal or low turns thinking off (a title request once thought for 2000+
  tokens).
- A small model that starts repeating is stopped (the last 64 tokens repeat with period ≤ 16).
- `--dump DIR` writes each request, prompt and generated text.
- Embeddings: too long is `context_length_exceeded` unless `truncate: true` (the name text-embeddings-inference
  uses). Query vs document comes from `input_type` as Cohere and Voyage name it (`query`/`search_query`,
  `document`/`search_document`/`passage`; default document). `encoding_format: base64` is little-endian float32.
- Prompt reuse across requests (common prefix, saved sequences for parallel conversations) is in
  [llm-engine.md](llm-engine.md). Chat template and tool-call XML: [llm-models.md](llm-models.md).

### OpenCode (`opencode`)

- Loads the models, then listens on 127.0.0.1 at a free port in 30000–49151 (below IANA's and macOS's
  ephemeral range from 49152), starting from a random port. The listener found is passed to
  `http.Server.Serve` as it is, so nobody takes the port in between. `--listen` overrides.
- Writes an OpenCode config (provider `hwnpu`, `@ai-sdk/openai-compatible`, each chat model's context and
  output limits) to `opencode/opencode.json` in the cache and passes it in `OPENCODE_CONFIG`. A custom
  config ranks above the global one and below a project's `opencode.json`.
- The model is chosen with `-m hwnpu/NAME`: first when the TUI runs, after `run` for `opencode run`. Other
  OpenCode subcommands reject `-m`, so it is not added and the config's `model` decides. With several
  `-m` models all chat models go in the config and the first is used.
- OpenCode runs by `os/exec` with this process's stdin, stdout and stderr (no pipes, so raw mode, the
  alternate screen, mouse and terminal queries pass through). Server logs go to a file. SIGINT is caught
  and dropped here (ignoring it would be inherited by the child; a caught signal resets to default on
  exec). When OpenCode exits the server closes and its exit code is returned.
- `opencode run` waits until stdin closes (fine on a terminal; use `< /dev/null` in scripts).
- Measured (M3, LFM3-1.7B, cache 16320): OpenCode's first request (system + 10 tools) is 7.4K tokens,
  read in 13 s; a bug fix took 12 rounds, later requests reused 99 % of the prompt, 63–68 ms/token.

### Coding agent (`agent`)

The loop of Pi (pi.dev) written in Go on `internal/chat.Engine`, without HTTP (`internal/agent`). No
mature Go agent library fit. System prompt and 4 tool declarations: 629 tokens (OpenCode's first request:
7,516), and no side requests (OpenCode asks for a title first, 1–2 minutes on a small model).

- `chat.Engine.Complete` / `Count`: Render → Tokens → Generate → Parse in one call. `agent.Model`
  (`Complete`, `Count`, `Context`) is met by `*chat.Engine`; tests use a scripted fake.
- `Session.Run`: add the user message, `Complete`, run the tool calls in order and add `tool` messages
  (with `tool_call_id`), repeat until an answer has no calls.
- Thinking of earlier rounds is not kept (`--keep-thinking` keeps it; it eats the 16K context).
- Context budget: before each request, `Count`; past `Context() - Reserve` (`-n`, default 2048), fold old
  tool results to their first line + `[... N more lines omitted]`, newest 4 kept. No summary generation.
  Still too long: `ErrContext`.
- System prompt in 3 layers: a short fixed core, rules per tool offered (fewer tools, fewer rules), and
  `AGENTS.md` (up to 4000 chars). Nothing per task in the system prompt: the engine does not re-read a
  common prefix, so what changes goes in the user message. `--system FILE` replaces the opening;
  `--dump DIR` to compare prompts on the same task.
- Tools take only string and integer arguments (LFM3's `<parameter=k>v</parameter>` is safe for strings;
  small models cannot write arrays or nesting).

| Tool | Behavior |
|---|---|
| read | 200 lines / 8 KB per call, then `[Lines a-b of n. Use offset=b+1 to read on.]`; no line numbers (they get copied into edit's `old`) |
| edit | Replace when `old` matches exactly once; 0 matches: "read and copy exactly"; 2+: "add surrounding lines". Blank lines at the ends of `old` are ignored (the parser drops one newline around values) |
| write | Creates parent directories |
| bash | `sh -c`, kills the process group, 60 s default, last 100 lines / 8 KB (Pi's 2000 lines / 50 KB is too big for 16K) |
| search | `Searcher`: Wikipedia (MediaWiki API, no key, first 2 sentences of top articles), SearXNG (JSON), Brave (`BRAVE_API_KEY`). `--search` picks one |
| fetch | Page as text, first `FetchChars` (6000) chars; inside `<main>` (else `<article>`), drops script, style, head, nav, footer, aside, comments |

Guards (a 1.7B model needs guards more than rules):

| Guard | Why |
|---|---|
| Working directory not written in the prompt; a list of files and "use relative paths" instead | The model miscopied a 100+ char absolute path and looped |
| An absolute path outside the directory resolves to the file of that name inside it (`resolve`); else the file list is returned | The model writes `/main.go` (OpenCode habit); refusing looped |
| `cd` to outside is refused with the rest shown (`Run: go run main.go`); the absolute work dir in output becomes `.` | The model tried `cd /tmp && go run` repeatedly |
| Same call (name + args) twice: not run, a nudge; third: `ErrLoop` | Repeated calls after the fix |
| Answer ≥ 20 words and ≥ 70 % the same words as the previous answer: final, calls dropped (`similar`) | The model kept appending `echo "done"` to its summary |
| On a loop or the round limit (`--steps`, 30), one request without tools asks for a short summary (`close`) | Tasks ended with no summary |
| edit/write a file not read: refused (`Session.seen`; written files count; `/reset` forgets) | From little-coder |
| Thinking capped at `--think-budget` (1024; 0 none): the engine inserts `</think>` (`chat.ThinkLimit`) | Small models think for thousands of tokens |
| Before write/edit/bash, `Config.Allow`: y/N on the terminal; `--yes` skips; no terminal and no `--yes`: refuse all. `--allow "go *,ls*"`: `*` any text, each part split at `&&`, `\|\|`, `;`, `\|`, newline must match | Safety |

Search-first (`--tools search[,fetch] --search-first`):

- Do not leave the decision to search to the model: changing "Answer in the language of the question" to
  "Always answer in Japanese" alone made it stop searching and invent an author (reproducible at
  temperature 0).
- The first round runs normally with tools declared (the model writes good queries there). If it does not
  call search, that round is dropped and a search for the task's first line is put in its place. Asking
  the model for a query alone produced copies of the question.
- `Searches` (4) per task: after that, "answer from what you have", then one request without tools
  (`ErrSearches`). Without the cap the model kept searching with the answer in front of it (30 rounds,
  14K tokens).
- `search` alone gives `SearchSystem` (255 tokens, no file rules).
- Query and URL hygiene: `『』「」` removed for Wikipedia; 0 hits → retry with the first word; URLs given
  percent-decoded (`readable`) because the model miscopies encoded ones. Rules say measurements need a
  search, arithmetic does not, and names are searched as written (no translation or romanization).
- Measured (M3, lut4 LFM3-1.7B, thinking on, SearXNG, search+fetch): 4 of 5 fact questions right.
  Wikipedia's own search finds less (missed 『こころ』). Remaining errors are the model's: romanizing
  Japanese names, mixing numbers from unrelated sections of a long page.

### systemone

`hwnpu-chat systemone` answers llama.cpp's `POST /v1/systemone` (TypeSafe-compatible System One API)
with a d1 model: one request from a file or stdin, or `--listen ADDR` to serve `/v1/systemone` and
`/v1/models` (`output_modalities: ["decisions"]`). Each question is one run on the shortest `--seq`
program that holds the prompt. `--cpu` runs `milcpu` for checking. Model details: [llm-models.md](llm-models.md).

## Terminal display (`internal/console`)

- Logs (downloads, conversions, loading) go to stderr in gray (`console.Log`). "all on the NPU" is green,
  "NOT all on the NPU" red, "on the CPU interpreter" plain; "placement not known, going on" when
  `MLComputePlan` did not finish (the NE busy compiling), which is not treated as off the NPU. What the
  user asked for (answer, file written, server address) is in normal color. `console.Notef` is a gray
  note on the bar's line (e.g. "prebuilt programs … finished on the device").
- `console.Bar`: on a terminal, label, spinner, bar, percent, elapsed time and details redrawn 10×/s; `Done`
  leaves one result line. Not a terminal: only the `Done` line, no escapes.
- Redrawn lines are cut to the terminal width − 1 (TIOCGWINSZ), shrinking the label with `…` first, and
  are drawn with auto-wrap off (DECAWM `\x1b[?7l` … `\x1b[?7h`), so a miscounted width (East Asian
  ambiguous `█` `░`, a window narrowed later) cannot wrap and leave stale lines. `Done` lines are not cut.
- Downloads (`chat.Fetch`, shared by all commands): size, speed, time left. Files are fetched one at a
  time: parallel bars would each estimate from their share of the bandwidth and overstate the wait.
  `chat.Expect` collects sizes first with 8 parallel HEADs: Hugging Face's `resolve` answers LFS/xet files
  with a CDN redirect whose `Content-Length` is the redirect body, so redirects are not followed and
  `X-Linked-Size` is read. Local sources are stat'ed; S3 and gated repos are left out of the total; `.part`
  counts as done. The bar shows `1.20 / 4.00 GB, all 3.10 / 9.40 GB 85.3 MB/s, 1m26s left` (all
  remaining ÷ current speed). Files found only after reading another (config.json, index.json) and small
  files are not in the plan.
- Load bars (`console.NewLoadBar`): a load is one opaque call (Core AI, Core ML, Level Zero), and compile
  time does not follow size (M1, Anima Turbo 20 programs: cache hits 0.16 s total, correlation 0.98; Core
  AI miss but NE hit 72 s, 0.86; NE evicted 320 s, 0.29 — the same 288 MB DiT block took 0.1 s and 177 s).
  So: remember loads that took ≥ 1 s (cache hits teach nothing), estimate the next from the nearest size
  (log scale, ties → latest) scaled by size, sizes < 1 MiB as 1 MiB. Show a spinner when the estimate is
  < 1 s or nothing is known. Progress = elapsed ÷ estimate, linear to 0.8, then `1 - 0.04 / (x - 0.6)`
  (90 % at the estimate, 97 % at twice), never past 99 %; `(estimated)`, no time left. Split models report
  parts with `console.LoadingPart(bytes)`; failed loads (`Bar.Failed`) are not learned. Estimates live in
  the process only. Sizes: `llm.ProgramBytes`, `llm.PartBytes`.
- hwnpu-chat load lines: `[1/2] Qwen3-0.6B 15s, 8 tokens per call, 1 part, all on the NPU`, then `loaded
  … in 34s, 2 programs, all on the NPU` and `ready after 35s: converting, loading and compiling 35s`
  (`console.Prepared`). File names only with `HWNPU_PROFILE=1`.
- hwnpu-image: per-part counts and spinners for export and load; a step bar (time per step, time left);
  the VAE's stage bar via `Pipeline.Stage` (`StageFunc`). The summary separates a first step that is
  ≥ 5 s longer than the mean of the others (it holds compiles and first calls): `8 steps in 5m25s (3.7 s a
  step; the first 4m59s, with loading and compiling)`.
- Last line (`console.Times`): `took 3m12s: downloading 2m40s, converting, loading and compiling 25s, running
  6.2s`. Only commands that call `console.Running()` print it (not conversations or servers).
  Loads during the run count as preparing (`console.Loading()`; `console.Reading()` for weight banks read
  again by `iron.Graph.Use`); `console.Preparing()` returns to preparing for commands that load another
  model mid-run. Download time is measured in `fetch` (overlaps counted once; sign-in wait and hash check
  included). Core ML's compile at the first call and `--play` count as running. With `HWNPU_ENERGY` the
  energy of each phase follows ([memory-and-power.md](memory-and-power.md)).
- `HWNPU_PROFILE=1` (all commands, `console.Profile`): each download (time, bytes, speed,
  `REPO@REV/FILE`), the size planning, each program load (nested loads indented, same path inside
  `runner.Open` counted once), weight bank re-reads as one line, and `decode.WriteProfile`. A download
  bar starts at the first byte, so the profile's time includes the Hub/CDN wait (swift-f0: 1.68 s vs 0.2 s).

## Ctrl+C

- `console.Interrupt()` returns a context and `stop` like `signal.NotifyContext`. Ctrl+C cancels the
  context; if the command has not ended after `InterruptGrace` (0.1 s), or on a second Ctrl+C, SIGINT is
  reset to default and sent to itself (Unix; elsewhere exit code 130). The 0.1 s lets context-aware work
  (downloads, child processes, deleting half-written files) finish.
- Why: an NPU command in flight cannot be cancelled (`run.Close` waits for it; a VAE stage at 1024² is up
  to 2.7 s), and pinned-buffer allocation (1.4 GB `mmap`) and weight reads (750 MB `io.ReadFull`) do not
  look at the context. SIGKILL at any point was safe: the kernel cleans up contexts and buffers in
  0.001–0.34 s, no driver errors, and the next run gave the same PNG.
- `console.EndInterrupted(err)`: an error after Ctrl+C also ends by SIGINT (not exit 1), so shell loops
  and scripts stop. A newline is printed on a terminal first.
- After `stop()` the command handles Ctrl+C itself: a chat turn's Ctrl+C stops only the answer;
  OpenCode's terminal. `serve` and `systemone --listen` treat Ctrl+C as the normal end: `stop()`, close the
  model, exit 0 (closing takes ~0.2 s, longer than the grace). Ctrl+C during loading still ends at once.
- Used by `hwnpu`, `hwnpu-chat`, `hwnpu-image`, `hwnpu-audio`. `hwnpu-dev probe`, `generate`, `formats`
  keep their own handling (they print results when stopped).
- Measured (AMD, Ctrl+C to exit): Anima Turbo 1024² 0.11–0.27 s; a 1024² VAE stage 0.42–0.87 s;
  gpt-oss-20b during loading 0.72–1.47 s. What remains is the kernel's cleanup, which grows with pinned memory (12 GiB: 1.5 s).

## hwnpu status

- Shows: hwnpu version (module, commit), OS, CPU, memory, the NPU the commands use (`chat.HereVendor`,
  `HWNPU_BACKEND`), devices, the NPU now (`npuinfo` summarized per OS: Linux amdxdna state, firmware,
  tiles, clocks, contexts, gpu_metrics busy and power; macOS ANE awake, IOReport power and reads; Windows
  compute-only adapter and per-process use), processes using the NPU and their NPU memory, the cache and
  its layers. `--json` writes everything.
- Reading must not wake the NPU: by default `hwnpu.Enumerate` is not called (opening the device wakes
  amdxdna). The NPU's presence and kind come from the OS (Linux driver name `amdxdna`/`intel_vpu`, macOS
  I/O Registry ANE, Windows compute-only adapter PCI vendor, or `HWNPU_BACKEND`). `--wake` opens devices
  and queries a sleeping NPU. Checked: `runtime_active_time` does not grow without `--wake`. macOS does
  not print "device: not opened" (opening there wakes nothing and tells nothing more).
- `-w 2s` on a terminal works like top: alternate screen (`\x1b[?1049h`); the fixed part (`writeFixed`:
  version, machine, NPU, devices, cache) is written once, the live part (`writeLive`: time bar, load, NPU
  state, processes) is rewritten in place. Lines cut to the width (wrapping would shift the rows), rows
  past the screen height dropped (scrolling would lose the fixed part), full redraw on resize, original
  screen restored on Ctrl-C. The interval includes read time. Not a terminal, `--json`, or Windows (VT not
  enabled): plain repeated output. Enumerate and the cache scan run once.
- Load (`internal/sysload`, no root):

| | CPU / memory | GPU | NPU |
|---|---|---|---|
| Linux | `/proc/stat` deltas; MemTotal − MemAvailable | amdgpu sysfs (`gpu_busy_percent`, `mem_info_vram_*`, `mem_info_gtt_*`) | gpu_metrics IPU busy and power; sum of process mappings |
| macOS | `host_statistics` ticks; `host_statistics64` "memory used" (app + wired + compressed, as Activity Monitor) | IOAccelerator PerformanceStatistics (`Device Utilization %`, `In use system memory`) | IOReport power; ANE driver wired memory (`mem.NPUWired`); busy % (below) |
| Windows | not read yet | not read yet | performance counters |

- Process table (`sysload.ProcSampler`, % from deltas): PID, name (16 chars), CPU%, memory, GPU%, GPU
  memory, NPU%, NPU memory, and the command line (`-w` and one-shot display; never in `issue`). `-` =
  unknown. Linux: `/proc/PID/stat`, `statm`, DRM fdinfo (`drm-engine-*`, `drm-resident-*`, old
  `drm-memory-*`, deduplicated by `drm-client-id`): amdgpu → GPU, amdxdna and intel_vpu → NPU. macOS:
  `proc_pid_rusage` (times via `mach_timebase_info`; memory = footprint + wired, since mlocked file pages
  are not in the footprint; `wired_bytes` in JSON; other users' processes need root) and the
  IOAccelerator user clients' `accumulatedGPUTime` (no per-process GPU memory).
- macOS NPU readings (M3, macOS 27.2):
  - IOReport energy is all 0 without root (ANE and CPU alike). When CPU energy is also 0 it prints "power
    not readable without root (sudo powermetrics …)" instead of 0.00 W (`DarwinIOReport.ANEPowerKnown`).
  - NPU% = share of time `PMP`/`AF BW`/`ANE0 RD` (ANE read-bandwidth histogram, 32 bins 1–32 GB/s,
    ~0.23 ms per sample on M3) is outside the lowest bin (`ANEBusyPercent`; 39–51 % while generating). The
    ANE has no per-process time, so the whole NPU% goes to a process holding ≥ 90 % of the processes' NPU
    memory (`npuHolder`; the input method and Siri hold a few MB each).
  - Interrupts are `ane N` (M3 is `ane 4`, not `ane 0`). `FirmwareLoaded` is shown only where present
    (`FirmwareKnown`; M3's `H11ANE` has none). `ANE`/`IOP State` stays `Running` while loaded, useless for
    load. `AMC Stats` `ANE RD` does not exist on M3.

## hwnpu issue

- Run log: `hwnpu-chat`, `hwnpu-image`, `hwnpu-audio` record their arguments (`console.RecordRun`), the
  `console.Log` lines (last state of progress lines, no colors) and how they ended
  (`console.EndInterrupted`) to `logs/<prog>-<time>-<pid>.log` under the cache. Last 10 runs per command
  (`console.RunsKept`). `HWNPU_NO_RUN_LOG` turns it off. Terminal detection is unchanged (`teeRun` has
  `Terminal()`).
- `hwnpu issue` (`--list`, `--run N`, `--lines` (80), `--print`, `--title`, `-y`, `--wake`): the last run
  that failed in the past week (else the last run): command, error, log tail and status as Markdown.
- Redacted: home (`~`), user name and real name, host name, `HWNPU_MODEL_SOURCE*` hosts and folders, hosts
  of URLs other than public ones (Hugging Face, GitHub), URL passwords, tokens (`token=`, `hf_`, `AKIA`,
  Bearer), e-mail addresses. Arguments over 40 chars with spaces (prompts) become their length; process
  command lines are left out.
- Sending: no `gh`. It opens `https://github.com/shibukawa/hwnpu/issues/new?title=...&body=...` with `open`
  (macOS), `xdg-open` (Linux), or `rundll32 url.dll,FileProtocolHandler` (Windows; `start` would let cmd
  read the `&`). Nothing is sent until the user presses the button. Over 7500 bytes the log is cut by 2/3
  steps; still too long, the user is told to paste `logs/issue-TIME.md`, which is always written.

## hwnpu cache, hwnpu memory

- `hwnpu cache` (info, `ls`, `prune`, `clear`, `clean`): see [distribution.md](distribution.md).
- `hwnpu memory [MODEL [GB]] [--forget]`: the memory class a model runs as, kept in `run.json`; see
  [memory-and-power.md](memory-and-power.md).

## Not adopted

- Old single binaries (`hwnpu-depth`, `hwnpu-music`, `hwnpu-sfx`, `hwnpu-tts`, `hwnpu-transcribe`,
  `hwnpu-pitch`, `hwnpu-notes`, `hwnpu-systemone`, `hwnpu-pose`, `hwnpu-agent`, `hwnpu-check`,
  `hwnpu-convert`, `hwnpuinfo`, `npuprobe`, `powerbench`, `report` …): folded into the families, no wrappers.
- Choosing a chat model by use (`-use chat|embed`): models of one use will multiply; names instead.
- Parallel downloads: per-file time left is wrong when the bandwidth shifts (4 GB + 1 GB at 100 MB/s show
  80 s and 20 s; both end at 50 s, the same as one at a time).
- A time-left figure on load bars: the estimate is unreliable.
- Keeping load estimates on disk.
- `gh` for issues: the user checks and sends in the browser.
- Agent: an existing Go agent library (kfet/agent has Anthropic types and no provider; pigo and og are CLIs
  with internal packages); Pi's TypeScript extension system (`agent.Config` functions, AGENTS.md,
  `--system`, `--tools` suffice); sub-agents, plan mode, per-turn skill injection (a 1.7B model cannot use
  them); summary-based compaction (folding instead); DuckDuckGo html/lite (202 anomaly from this machine);
  a model-written query alone for search-first (copied the question).
- Ignoring SIGINT for OpenCode's parent: the ignore would be inherited by OpenCode.

## Open

- Agent: repair of broken tool calls (output-parser): once `<parameter=command` came broken and bash's
  "command is required" made the model fix it next round. Several tool_calls in one answer (run in order,
  not observed). Gemma 4 E2B/E4B as the agent model. `--no-think` comparison.
- Search: `fetch` with a `query` to cut the relevant paragraph (long pages give only the first part and
  unrelated numbers); also searching ja → en when ja finds nothing; using the task's text when the model's
  query has no Japanese but the task does.
- Embedding server: batching several texts in one run (now one at a time).
- `hwnpu status`: macOS and Windows summaries only cross-built at first; Windows CPU/GPU load not read.
- `hwnpu issue`: the length GitHub accepts in the URL is not checked in a browser.
- Ctrl+C on `hwnpu-audio` not checked on hardware.
- Load-estimate bars on LLM parts.json chains, Intel and AMD: only built.
- OpenCode's title request delays the first answer by 1–2 minutes on a small model.
