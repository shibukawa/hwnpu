# hwnpud: one process for the NPU and the models

[日本語](hwnpud.ja.md)

Status: step 1 is built on the Mac and on Linux with AMD's NPU (below, "Built"); the rest is design. hwnpud is a per-user daemon on macOS and Linux that owns the NPU and the
models: it downloads and keeps the converted models, holds the caches, loads the programs and runs
them. hwnpu's commands (`hwnpu`, `hwnpu-chat`, `hwnpu-image`, `hwnpu-audio`) and the apps built on
the `ai` package reach it over a Unix domain socket. They pass what their run needs and draw what
comes back; they no longer run the NPU or manage the models in their own processes. Windows comes
after it has been looked into on a real machine.

## Built

Step 1 on the Mac (`cmd/hwnpud`, `internal/daemon`):

- `hwnpud serve` is the supervisor: it takes launchd's socket (`launch_activate_socket`), else
  listens on `HWNPU_DAEMON_SOCKET` or `~/Library/Caches/hwnpu/hwnpud.sock` (mode 0600, the peer's
  uid checked with `LOCAL_PEERCRED`). It answers `GET /hwnpud/v1/hello` and `status` and
  `POST /hwnpud/v1/quit` itself and passes the rest to the engine, which it starts at the first
  such request and again after the engine ends. Under launchd it exits after `HWNPU_DAEMON_IDLE`
  (5m) without an engine or a request, and as soon as it is idle once its executable has been
  replaced.
- The engine (`hwnpud engine`) holds the chat models: `POST /hwnpud/v1/load` makes one ready as
  hwnpu-chat does with its default flags and streams the log (JSON lines, with whether the
  client's terminal redraws bars), then the model; `/v1/chat/completions` is `chat.Server` over
  the model. The usage carries hwnpu's statistics (`chat.UsageStats`). It exits after the idle
  time, which gives the Neural Engine's memory and its lock back.
- `hwnpud install` writes `~/Library/LaunchAgents/com.github.shibukawa.hwnpud.plist` (log in
  `~/Library/Logs/hwnpud.log`) and bootstraps it; it warns of hwnpu's variables set only in the
  shell. `uninstall`, `status`, `stop`.
- `hwnpu-chat [chat]` goes through it with `HWNPU_DAEMON=on` (or `auto`): the load's log, then the
  conversation, the same output and line of statistics. The flags of how a model is converted
  and run are refused there.
- Settings ([Settings](#settings)): `hello` gives hwnpud's settings and the `.env` files read; a
  changed file starts the next engine with the new settings; a client of other settings is told
  (`on`) or runs by itself (`auto`); `hwnpud status` shows them; a client's
  `HWNPU_MODEL_ALLOWLIST` narrows hwnpud's policy.
- M3: the PLaMo 3 Translate 2B program loads in 22 s the first time under the name `hwnpud`
  (Core AI's entry 2.5 GB); the next run answers in 0.6 s from start to end, the model still in
  the engine.

On Linux with AMD's NPU, the same supervisor and engine:

- `hwnpud install` writes systemd's user units `~/.config/systemd/user/hwnpud.socket`
  (`ListenStream=$XDG_RUNTIME_DIR/hwnpud.sock`, mode 0600, `FileDescriptorName=hwnpud`) and
  `hwnpud.service` (`ExecStart=… serve`, `KillMode=mixed`: the supervisor stops the engine
  itself), then `systemctl --user enable --now hwnpud.socket`. The supervisor takes the socket
  from `LISTEN_FDS` (the variables are not passed to the engine) and checks the peer's uid with
  `SO_PEERCRED`. The log is the journal (`journalctl --user -u hwnpud`). systemd's user manager
  gives the service `~/.config/environment.d`'s variables, so `install` warns only of those
  neither it nor a `.env` file sets.
- The engine holds one chat model at a time on AMD's NPU: one model uses all of its hardware
  contexts, so a load of another closes the one held, once its requests have ended (no new
  request gets it meanwhile). The requests of a model take turns on it (`chat.Engine` runs one
  generation at a time), in the order they came.
- The NPU lock (`neural-engine.lock`) is taken on AMD too, when a chat model's IRON export loads
  (in hwnpud's engine and in `hwnpu-chat` run in its own process), so a client with
  `HWNPU_DAEMON=off` prints `waiting for the NPU: process N (hwnpud) is using it` and waits
  for the engine's idle exit. Images and audio on AMD do not take it yet (step 4).
- `hwnpud status` shows the engine's NPU buffers, which are pinned (`drm-total-memory` of
  `/proc/self/fdinfo`, `mem.DRMMemory`).
- Ryzen AI 9 365 (`hwnpud install`'s units, and `systemd-socket-activate` for a second socket):
  LFM3-1.7B's export loads in 2.6 s the first time and answers at 17 tok/s; the next run takes
  3.1 s from start to end with no load (1.62 GiB pinned). Three clients at once are served one
  after the other (12 s, 24 s, 43 s). A load of Qwen3.5-0.8B closes LFM3-1.7B first (1.02 GiB
  pinned after). `hwnpud stop` ends the service; the next connection starts it again.

## Why

### Mac

What grows with each process today (M3, macOS 27; [Core AI](apple-coreai.md), [Apple Neural Engine](apple-neural-engine.md)):

| Copy | Key | How it grows |
|---|---|---|
| Core AI entries (`~/Library/Caches/coreai-cache/<OS build>/<executable name>/`) | The executable's name | About the weights' size again for each name. On the development machine: hwnpu-chat 15 GB, hwnpu-image 15 GB, parallax 8.3 GB, layers 6.5 GB, interpreter 5.2 GB, about 50 GB in all; layers and parallax hold the same depth and image models as hwnpu-image |
| aned's compiled ANE code (`/Library/Caches/com.apple.aned`) | The signing identifier | Shared today because every Go build is ad-hoc signed as `a.out`. Apps signed with their own identifiers would each get their own copy |
| ANE wired memory (`mem.NPUWired`) | The process that loaded the program | One process at a time, because of `neural-engine.lock` |

The lock keeps the memory from adding up, but a process holds it until it exits. While app A is
open, app B prints `waiting for the Neural Engine` and stops. Turning the lock off would load the
weights twice, and two processes on the ANE have frozen an M1.

### Linux (AMD XDNA)

The exports in `entries/amd-xdna2` are already one copy on disk. What adds up is the memory and
the fragility of several processes on the NPU ([AMD](amd.md)):

- **Pinned memory per process.** Every page of an NPU buffer (`BO_SHARE`) is pinned and counted
  against no process's RSS: Gemma 4 E2B raises `Unevictable` by 9.3 GB (Ryzen AI 9 365). Two apps
  that open the same model pin it twice.
- **Several processes are fragile.** `CREATE_HWCTX` right after another process's context ended
  sometimes fails with `EINVAL` (retried for 5 s). Asking the firmware while a model creates its
  contexts has hung the SMU until a reboot. NPU1 has 6 hardware contexts and one model uses them all.
- **Priority** is already set per context (QoS LOW / NORMAL / HIGH) and the firmware orders the
  commands. hwnpu's full ELFs carry no preemption sections (they are refused), so a switch most
  likely comes only between commands (one graph call), the same grain as the gate below. Whether
  the QoS works across processes has not been measured.

### Both

One process on the NPU removes all of this:

- One copy of the caches on disk (on the Mac, the Core AI cache under the name `hwnpud`)
- A model two apps use is on the NPU, and pinned, once
- Apps wait on each other for one program call, not for a whole process
- No two processes ever use the NPU together
- One place that downloads, keeps and removes the models, with the same API on every NPU

## Models and caches

From now on the machines do not build the models. They download them nearly ready from the
sources ([Distribution](distribution.md): Hugging Face, a NAS, S3) and use them. hwnpud is what
does it, and the only writer of hwnpu's cache.

- **The clients draw, hwnpud decides.** `list`, `status` and `models prepare` of each command and
  of the `ai` cli send what they need (the uses, the machine's memory, the app's name) and draw
  the answer: the rows of `chat.ListHave`, an `ai.Plan`, a stream of `ai.Progress`. `hwnpu`'s model
  commands (`hwnpu cache`, `hwnpu memory`) do the same. These types become the wire types.
- **One download for everyone.** Two clients preparing the same model share one download. When a
  client goes away its download goes on while another client waits for it, and stops when none
  does.
- **The locks between processes go away.** The download, convert and export locks
  (`cache.Lock`) become exclusion inside hwnpud.
- **Removing knows what is in use.** `hwnpu cache`'s gc and the trimming to the free-space floor
  skip what is loaded.
- **The sources are read from files**, as the other settings of where the models are and come from
  ([Settings](#settings)).
- **Licenses**: hwnpud has no terminal. For a model with `accept: true` it answers "needs
  acceptance" with the terms; the client asks the user and sends the answer, which goes to
  `licenses.json` as now. `chat.Policy` (`HWNPU_MODEL_ALLOWLIST`, `policy:`) is applied in hwnpud,
  and a client's narrows it ([Settings](#settings)).
- **What still builds**: `hwnpu-dev convert` and putting exports to the sources stay in the
  developer's own process. On the Mac the device still finishes the ANE code (prebuilt or not) and
  compiles again when an entry is gone; that runs in hwnpud, or another name's cache appears again.
- **One code.** The fetching and the cache stay a library that hwnpud hosts; the in-process
  fallback (below) runs the same code.

## Settings

hwnpu's settings are environment variables today, which each process reads from its shell and
from the `.env` files (`envfile`: the current directory's, `~/.config/hwnpu/.env`, the machine's).
With hwnpud they split in two.

| Whose | Variables | How hwnpud gets them |
|---|---|---|
| hwnpud's | Where the models are and come from, and how the machine runs them: `HWNPU_CACHE`, `HWNPU_CACHE_MIN_FREE`, `HWNPU_CACHE_MAX_SIZE`, `HWNPU_MODEL_SOURCE*`, `HWNPU_MODEL_HUGGING_FACE`, `HWNPU_MODELS`, `HWNPU_BACKEND`, `HWNPU_DEVICE`, `HWNPU_MEMORY`, `HWNPU_ACCEPT_LICENSE`, `HWNPU_COREAI_*`, the credentials (`HWNPU_WEBDAV_*`, `HF_TOKEN`, `HF_ENDPOINT`, `AWS_*`) | From the `.env` files and the environment it was started with, never from a request |
| The client's | What this run needs: the model, its memory (`--memory`, a tag), the model policy (`HWNPU_MODEL_ALLOWLIST`) | With each request |

- **Not the shell's.** launchd and systemd do not read the login shell's profile, so
  `HWNPU_MODEL_SOURCE` set in `~/.zprofile` does not reach hwnpud: it goes in
  `~/.config/hwnpu/.env`.
- **Why the cache is not a client's.** hwnpud exists so that the machine has one copy of each
  model, one download and one exclusion. A client with its own `HWNPU_CACHE` would make a second
  copy outside the locks and the use counts; one with its own sources would make what an app
  gets depend on its shell. Credentials have no reason to cross the socket either.
- **The files are the truth, not a command that sets hwnpud's memory.** hwnpud exits when idle and
  launchd starts it again, so a value set only inside it would be lost. The processes that do not
  go through it (`HWNPU_DAEMON=off`, `hwnpu-dev convert`) read the same files, so every path runs
  with the same settings.
- **A changed file is read without restarting hwnpud.** The supervisor keeps the environment it
  was started with, without what the files set (`envfile.Original`), and reads the files again
  for each new engine (`envfile.Environ`). When they changed, the next request that finds no
  other going on stops the engine and starts one with the new settings: a model loaded from the
  old cache is not kept. `hello` gives the settings the next engine gets (credentials only as
  set, passwords taken out of URLs), the files read, and `pending`, the names the running engine
  has other values of.
- **A client of other settings is told.** A client compares its own settings (its shell and its
  `.env` files, with a project's) with `hello`'s. With `HWNPU_DAEMON=on` it goes through and logs
  that hwnpud's settings are used and which files hwnpud reads; with `auto` it runs in its own
  process with its own settings. `hwnpud status` shows the settings and what differs in the
  shell it runs in; `hwnpud install` warns of the variables only the shell sets.
- **The policy narrows, never widens.** A client sends its `HWNPU_MODEL_ALLOWLIST`
  (`LoadRequest.allowlist`, `X-Hwnpu-Model-Allowlist`). hwnpud runs a model when its own policy
  allows it and the definition files' rules with the client's items do too
  (`chat.AllowedAlso`): a client can refuse more, and its `+nsfw` or `+license` does not
  override hwnpud's `-`. An administrator's rules go in the machine's `models.yaml` (`policy:`),
  which every process reads. A family is picked by hwnpud's policy alone so far: when the client
  refuses the pick it is an error, not the next model, until the Accept-like pick (below).
- **A command to change them** comes with step 2: `hwnpu config get|set|unset NAME [VALUE]`
  writes `~/.config/hwnpu/.env` (credentials with `--secret` go to the OS's store) and shows hwnpud's values
  beside the file's. Nothing has to tell hwnpud: the next engine reads the file.

Decided (2026-11-22), not built yet:

- **Credentials in the OS's store.** The NAS's password and the tokens leave `.env`.
  `hwnpu config set --secret NAME` puts one in the Keychain on the Mac (a generic password of
  service `hwnpu`, account `NAME`, through `/usr/bin/security`: the item's access list then names
  Apple's signed tool, not hwnpud's ad-hoc signature, which changes with every `go install` and
  would ask again after each) and in the Secret Service on Linux (`secret-tool` or D-Bus,
  attributes `service=hwnpu name=NAME`). A variable the environment sets still wins, for CI.
  A Linux machine with no keyring unlocked (headless, the systemd user unit before a login)
  falls back to `~/.config/hwnpu/secrets.env` at mode 0600, which hwnpud refuses to read when
  others can.
- **Moving the cache is clearing it.** A new `HWNPU_CACHE` starts empty and fills from the sources
  (the NAS's prebuilt exports, Hugging Face); the old root is removed, not migrated
  (`hwnpu cache clean --root OLD`, or by hand once hwnpud no longer uses it).
- **A family is picked as HTTP picks a representation by Accept.** A client sends what it
  accepts with preferences instead of a single name: `accept: ["plamo3translate;q=1",
  "lfm3;q=0.5", "+license:open"]`, the policy items as filters and the names, families and
  features with q. hwnpud picks among the family's models for the machine's memory the best that
  its policy and the client's accept, and answers which one and why, as `Content-Type` and
  `Vary`; when none is left it answers as `406`, with each candidate and what refused it.
  `HWNPU_MODEL_ALLOWLIST` becomes the filter part of it.
- **The developers' variables are sorted out** ([Variables](variables.md), #294): the switches are
  `HWNPU_DEV`'s keys, which hwnpud does not take from a client (`Settings` leaves them out), and
  the tests' run in their own process with the Neural Engine's lock.

Later:

- **A client's own definitions.** An experiment's model, or a distilled model of another
  architecture compatible with one hwnpud runs, is defined in the client's `models.yaml`. The
  client sends the file with its requests; hwnpud reads it over its own for that client only,
  and keys what it builds and caches by the definition, so two clients' models of the same name
  do not meet. The architecture has to be one hwnpud's build knows; a new one runs with
  `HWNPU_DAEMON=off` until hwnpud is built with it.
- **The definitions of the client's build.** A client newer than hwnpud may know models that
  hwnpud's built-in `models.yaml` does not; names are resolved by hwnpud, so the commands' `list`
  has to come from hwnpud too (step 2), and an older hwnpud is replaced at its next idle exit.
- **Several users on one Mac** each have their own hwnpud and cache; the Neural Engine's lock is
  the machine's, so the second user's engine waits for the first one's to exit.

## Boundary: models, not graphs

The RPC is a model's call at the grain of the `ai` package's uses: `Transcribe(pcm)`,
`Generate(prompt)` with a stream of tokens back, generating an image, and so on.

- Not one RPC per graph call. chain and runner write caches and state straight into the program's
  arrays (Core AI's `cai_input_array`, IRON's banks) and bind the same arrays to several programs,
  so KV caches, rings and windows must live in the daemon. A call costs at least 160 µs (M2) or
  0.06 ms (XDNA), and a token takes several.
- The `ai` package already splits declaring, `Plan`, `Open` and `Get`, so `ai.Runtime` gets a
  remote side behind the same API. An app's code does not change.
- Data is copied over the socket at first: an image at 1024² is a few MB, audio less. Shared
  memory passed with `SCM_RIGHTS` comes when a large tensor needs it.

## Processes

| | Mac | Linux |
|---|---|---|
| Service | Per-user LaunchAgent (`~/Library/LaunchAgents/<label>.plist`) | systemd user units (`~/.config/systemd/user/hwnpud.socket` and `.service`) |
| Started on the socket | The plist's `Sockets`; Go takes the descriptor with `launch_activate_socket` through purego | `hwnpud.socket`; Go reads `LISTEN_FDS` |
| Socket | Under the user's cache (`sun_path` is at most 104 bytes) | `$XDG_RUNTIME_DIR/hwnpud.sock` |
| Peer check | `LOCAL_PEERCRED` | `SO_PEERCRED` |

- **Per user, not root**: the cache, the license answers and `models.yaml` are all per user. Neither
  NPU needs root to run.
- **Idle exit**: it starts at the first connection and exits after it has been idle for a while,
  so it holds no NPU memory (ANE wired, XDNA pinned) and no resident memory when nothing uses it.
  This generalizes `hwnpu-chat -idle` (release after 5 min) to every model.
- **A supervisor and one engine.** Core AI takes the whole process down when its own conversion
  fails (LLVM), and a stuck NPU command must not take the socket with it. The supervisor holds the
  socket, the queue and the model management and never touches the NPU. The engine is the same
  executable run again (`hwnpud engine`), so on the Mac it has the same name and hits the same
  Core AI cache. There is only ever one engine; the supervisor starts a new one when it dies and
  fails the requests that were running. The NPU memory goes with the process.
- **The engine holds the NPU lock only while it has programs on the NPU**: `neural-engine.lock` on
  the Mac as now, and the same lock on Linux (taken by the chat models' IRON exports so far). It releases the lock when it
  unloads after going idle, so tools that open the NPU in their own process (`hwnpu-dev probe`,
  `aicheck`, the IRON tools) still run and wait their turn.

## Scheduling across apps

One process makes some limits shared by every app:

- **Programs**: 112 a process on the Mac (M2): the 113th load fails and Core AI silently runs it
  on the GPU. Hardware contexts on XDNA (6 on NPU1). The engine counts them (`aneLive` on the Mac)
  as a budget and unloads the least recently used models before it is reached.
- **Memory**: `App.InTurns` grows into one plan for the machine: the budget (models.yaml's
  `memory`), one copy of a model shared by every client, and the order of unloading.
- **Priority, at each program call**: the engine passes the NPU from one program call to the next
  with a priority gate around the call (`cai_run`, an IRON command), not around a request. An
  interactive request (chat, speech) waits for the call that is running, not for the image:
  Anima's DiT on the Mac is 7 calls of 4 blocks a step (about 0.8 s each on M1, step 5.6 s) and the
  VAE 10 stages, so speech gets in between two of those. Anima holds only 0.57–0.84 GiB on the ANE
  (`runner.Ring`), so the speech models stay loaded beside it and pay no load. What it costs: a
  large DiT group after another program pays the switch (+50–70 ms, `aicheck cycle`; an XDNA
  context switch 0.46 ms), and the image takes as much longer as the calls run in between. The
  CPU's parts (samplers, tokenizers) run beside the gate. The gate is also what makes the
  priority hold: without it the calls queue in the order they came.

## Clients and versions

- `HWNPU_DAEMON=on|off|auto`. The `ai` package's apps use `auto`: the daemon if it answers with a
  matching protocol, otherwise in their own process as now. hwnpu's commands use `off` unless it is
  set, so `go run ./cmd/hwnpu-chat` in a worktree runs that worktree's code and the
  `base/hwnpu-chat` / `new/hwnpu-chat` comparisons keep working.
- `HWNPU_DAEMON_SOCKET` points to another socket, to try a hwnpud built from a worktree beside the
  installed one.
- The first message gives the protocol version and the build. A daemon older than its client is
  asked to exit once it is idle, and the service manager starts the new binary at the next
  connection.
- The socket is mode 0600 and the daemon checks the peer's uid.

## Installing and distributing

### Now: built with Go

- `go install ./cmd/hwnpud` (or `go build`) names the executable `hwnpud`, which on the Mac is the
  only key of Core AI's entries. `go run ./cmd/hwnpud` runs under the same name from a temporary
  folder and hits the same cache.
- Go's linker signs ad-hoc as `a.out` on the Mac, as every development build is, so aned's code
  stays shared. No `codesign` step.
- `hwnpud install` writes the service files pointing at its own path (`os.Executable`, usually
  `~/go/bin/hwnpud`) and starts them (`launchctl bootstrap gui/$UID`, `systemctl --user enable
  --now hwnpud.socket`). `hwnpud uninstall` undoes it. After `go install` replaces the binary, the
  next start runs the new one.

### Later: binaries for people without Go

- The executable keeps the name `hwnpud` for good, so new versions keep the Core AI cache.
- On the Mac, an ad-hoc signature can carry a chosen identifier without a paid Apple Developer
  account (`codesign -f -s - -i <id>` gives `Identifier=<id>`, `Signature=adhoc`). Release builds
  use the reverse-domain identifier that a Developer ID signature would use later.
- Gatekeeper only checks files that carry the quarantine attribute:

| Way | Quarantine | Without notarization |
|---|---|---|
| `go install` | No | Runs |
| A Homebrew formula (from source or a bottle) | No | Runs |
| `hwnpud install` or the `ai` cli's `daemon install`, downloading with Go and checking the sha256 | No | Runs |
| A browser download from GitHub Releases | Yes | Blocked; System Settings' "Open Anyway" or `xattr -d com.apple.quarantine` |

- The paid account (99 USD a year) is needed to hand `.app` or `.pkg` downloads to people who do not
  use a terminal, or to register a login item from an app (SMAppService). It is not needed to share
  the caches.
- On Linux, the same `daemon install` and the release binaries; distribution packages wait.
- Apps do not bundle their own hwnpud. They look for the installed one and offer to install it
  (`daemon install|status|uninstall` in the `ai` cli, shown like `models prepare`), and run in their
  own process when there is none.

## To measure first

Each is small; the ones on the NPU wait for the lock in the background.

1. Mac: a child exec'd from the same executable (same name, other arguments) hits the Core AI
   cache. Measured (M3): yes; a new engine loads PLaMo 3 Translate 2B in 0.3 s from the entry an
   earlier engine made.
2. Mac: aned keys only on the identifier: two ad-hoc builds with the same chosen identifier share
   the ANE code, and how long the first start takes after the identifier changes (ad-hoc `a.out`
   → chosen identifier → Developer ID with a team).
3. Mac: two apps' models in one process (interpreter and parallax): `mem.NPUWired` and the
   112-program limit.
4. Mac: speech calls between Anima's DiT calls in one process: the speech call's wait and time, the
   image's added time, and whether Core AI takes two programs' `fn.run` in flight at once. Threads
   are not the question: `cai_run`'s thread only waits on a semaphore while a `Task.detached` runs
   the call on Swift's pool, so the goroutine's thread does not matter and needs no
   `runtime.LockOSThread`.
5. Linux: two models in one process side by side (chat and speech): pinned memory, hardware
   contexts, and commands of two models interleaved at the gate (the barrier `stepBarrier` /
   `chunkBarrier` still holds).

## Steps

1. hwnpud with the supervisor, the engine, socket start, idle exit, `install` / `uninstall` on the
   Mac and Linux; chat models only, with `hwnpu-chat` as a client (`HWNPU_DAEMON=on`)
2. The model management in hwnpud: list, status, plan, prepare with progress, licenses, gc; the
   commands' `list` / `status` and `hwnpu cache` as clients; `hwnpu config`; the policy in the pick
3. The remote side of `ai.Runtime` and `HWNPU_DAEMON=auto`; samples interpreter, layers and
   parallax through the daemon, with the Core AI cache in `coreai-cache/<OS>/hwnpud/` only
4. Images and audio; the machine-wide scheduler (program budget, unloading, priority)
5. Release builds (a fixed identifier on the Mac), `daemon install`, a Homebrew tap; notarization
   when the paid account exists
6. Windows, after it has been looked into on a real machine (the service, the socket, XRT)

## Not taken

- **One RPC per graph call**: the KV caches and the arrays bound to several programs live in the
  process that runs them, and every token would cross the socket several times.
- **A root service** (LaunchDaemon, a system unit): the caches, licenses and definitions are per
  user; the NPUs need no root.
- **Lock off, apps on the ANE together**: the weights load twice, and two processes have frozen an
  M1 ([Apple Neural Engine](apple-neural-engine.md)).
- **Only the firmware's QoS on XDNA**: it might order the commands, but each process still pins its
  own copy of a model, and several processes have hung the SMU.
- **The clients pass their environment for the sources**: apps would see different sources and
  fill the cache differently.
- **hwnpud bundled in each app (SMAppService)**: needs a signed app, and copies with different
  labels and versions would split the caches again.
- **A Homebrew cask**: casks get the quarantine attribute, and Homebrew is phasing out casks that fail
  Gatekeeper.
