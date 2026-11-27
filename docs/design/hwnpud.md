# hwnpud: one process for the NPU and the models

[日本語](hwnpud.ja.md)

Status: steps 1 and 3 are built on the Mac and on Linux with AMD's NPU (below, "Built"); the rest is design. hwnpud is a per-user daemon on macOS and Linux that owns the NPU and the
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
- `hwnpu service install` writes `~/Library/LaunchAgents/com.github.shibukawa.hwnpud.plist` (log in
  `~/Library/Logs/hwnpud.log`) and bootstraps it; it warns of hwnpu's variables set only in the
  shell. `uninstall`, `status`, `stop`.
- `hwnpu-chat [chat]` goes through it with `HWNPU_DAEMON=on` (or `auto`): the load's log, then the
  conversation, the same output and line of statistics. The flags of how a model is converted
  and run are refused there. (Step 3 moved it onto the `ai` package, below.)
- Settings ([Settings](#settings)): `hello` gives hwnpud's settings and the `.env` files read; a
  changed file starts the next engine with the new settings; a client of other settings is told
  (`on`) or runs by itself (`auto`); `hwnpu service status` shows them; a client's
  `HWNPU_MODEL_ALLOWLIST` narrows hwnpud's policy.
- M3: the PLaMo 3 Translate 2B program loads in 22 s the first time under the name `hwnpud`
  (Core AI's entry 2.5 GB); the next run answers in 0.6 s from start to end, the model still in
  the engine.

Step 3 on the Mac (`internal/airpc`, `internal/aihost`, `internal/cachecmd`, the `ai` package):

- Every program of the `ai` package goes through hwnpud: `HWNPU_DAEMON` is `on` unless it is set
  (`daemon.DefaultMode`), and a program whose hwnpud does not answer fails with what to do.
  `HWNPU_DAEMON=off` runs the models in the program's own process as before; a test (`go test`)
  and hwnpud's engine itself (which sets it off) run them there too. `hwnpu-chat [chat]` goes
  through hwnpud by default as well, through the `ai` package like the rest (below).
- A session (`internal/airpc`) is a connection to the socket upgraded from HTTP
  (`GET /hwnpud/v1/ai`, `Upgrade: hwnpu-ai/1`, through the supervisor's proxy to the engine),
  with a gob stream each way of messages multiplexed by id. The client calls the methods of the
  engine's objects by name, with the types of the methods of a type of its own (the model's
  type, or `ai.Host`): the arguments and results with gob (a picture behind `image.Image` as its
  concrete type), a `context.Context` as a cancellation, a function argument and the function
  fields of a struct argument (`imagegen.Options`' `Progress`, `Drawn`, ...) as callbacks into
  the client, an iterator (`flow.Stream`, `iter.Seq`) as a stream either way, read as it is
  ranged over and stopped when its reader stops. Errors keep their registered sentinels and
  types (`ai.ErrNotPrepared`, `*ai.LicenseError`, `text.ErrContext`, `*text.FormatError`), so
  `errors.Is` and `errors.As` hold. The engine's handler returns when the session ends, so the
  engine does not exit idle under an app.
- The engine (`internal/aihost`, given to `daemon.Engine.AI` by `cmd/hwnpud`; `internal/daemon`
  imports no package of models) serves `ai.Host` on each session: `App.Plan` and `Plan.Run`
  make the app again from what the client sent (`App.Spec`, `ai.AppFromSpec`: the uses by
  their roles' names, `airole.Lookup`, the groups of `InTurns`, the memory, the store with its
  definitions) and plan or run it in the engine, `Run`'s stages and the models' logs going back
  as they come (the client draws them as `Run` does). `App.Open` opens a Runtime of the
  session; `Use.Get` loads the use there and gives a model of the app's own type
  (`airole.Role.Remote`) whose methods call the engine's. `Release` and `Close` unload them;
  a session that ends unloads its models. Loads and runs of all the sessions take turns. The
  client's `HWNPU_MODEL_ALLOWLIST` narrows hwnpud's policy, as `LoadRequest` does.
- So the preparing, the compiling (`Run`'s loads and `Warm`, and `CompileOnLoad`'s first load)
  and the models run under the name `hwnpud`: Core AI's cache is
  `coreai-cache/<OS>/hwnpud/` alone, and the store's `compiled_by` is `hwnpud` for every
  program. A model prepared for a role under one use's name is ready for another use of the
  same model, and compiled when hwnpud compiled either (hwnpu-audio's `whisper-large-v3-turbo`
  is the interpreter's `transcribe`).
- Every role runs through hwnpud: depth, faces, label, pose, detect, segment, ocr, image and
  image-edit, transcribe, speak, translate, chat, embed, decide, music, notes, pitch and sfx
  (each registered, `airole.Register`, with its client side, `Role.Remote`). The stream methods
  that are made of a model's single calls (`TranscribeStream`, `SpeakStream`, `DetectStream`,
  `CompleteStream`, `Labeler.Label`, `Segmenter.Segment`, the nodes) run in the client over
  those calls.
- What a model's call leaves in the engine: a picture's encoding (`Segmenter.Encode`'s
  `*PictureEncoding`, the image encoder's output that `Masks` decodes) stays there behind a
  handle (`airpc.RegisterRef`): the client's value has its exported fields (the times) and the
  handle, `Masks` given it is the engine's own again, and the engine's is let go when the
  client's is garbage collected (or the session ends). A label's candidates and the embeddings
  (`Candidates`, `Embedding`) are plain data and cross as they are; `Rank` and `Probability`
  still run in the engine, with the model. A result a stream fills in as it goes
  (`MusicGenerator.Generate`'s `*MusicStats`) is sent again with the stream's end, so that it
  is whole when the loop over the stream ends, as here; a loop that stops early waits for the
  producer to stop, as a loop over an iterator does (the producer is a few chunks ahead: the
  stream has no flow control). Variadic methods (`MusicGenerator.Style`) send their slice.
- The commands' and samples' model work all goes through `ai`. What stays in the program's own
  process: `hwnpu-image convert` (Anima's conversion, no NPU), `hwnpu-dev` (the
  developer's process), `hwnpu issue`, and the logs of the Neural Engine's memory
  (`mem.NPUWired`, the machine's, so a command still tells how much more it holds). A command's
  `--backend` sets this process's `HWNPU_BACKEND`, which hwnpud does not take (it says that its
  own is used).
- `hwnpu-chat chat` runs its model through `ai` too, as its other commands do: the flags of how a
  model is converted and run become an app's definition (`ai.Definition`) that hwnpud prepares,
  instead of being refused, and the model's files are known to the cache's commands. The
  engine's chat routes (`/hwnpud/v1/load`, `/v1/chat/completions`) stay, for the OpenAI API's
  clients. What it gives up: a model the route loaded stayed in the engine until it was idle,
  so the next run answered at once (0.6 s); through `ai` the model is unloaded when the last
  client lets it go, and the next run loads it again from Apple's compiler's cache (Qwen3-0.6B:
  0.5 s, the run 1.2 s from start to end).
- One copy of a model for every client: the engine's Runtimes (one a session) share a model
  loaded alike, the same role, model, files and settings of how it runs (`LoadSettings`,
  `AllowOffNPU`): the first `Get` loads it, the next ones take it, and it is unloaded when the
  last Runtime lets it go (`Release`, `Close`, a session that ends). Its calls take turns as
  its type makes them (each runs one request at a time). An app's `InTurns` still releases its
  own share of the others of the group. The model's log goes to the latest client's that holds
  it. `hwnpu service status` shows each model held and the clients holding it.
- The cache's commands run in the engine (`internal/cachecmd`, `cachecmd.Host` on the same
  session): `hwnpu cache` (info, ls, prune, clear, clean) and `hwnpu memory`, with the engine's
  settings (its `HWNPU_CACHE` and sources, not the shell's), their output and their question
  ("remove them?") at the client. `hwnpu status` and the apps' `status` show the engine's cache.
  So hwnpud is the only process that removes from the cache: `ls` marks the units the models it
  holds are loaded from ("in use"), `clear` and `prune` keep them ("kept: hwnpud holds
  Qwen3-0.6B loaded, for 2 clients"), and the engine's own trimming to make room skips them
  (`cache.SetInUse`: the engine runs for long, so what it once touched is not kept for good, as
  a command's process keeps it). What removes waits for the loads and runs going on and keeps
  the next from starting until it is done. With `HWNPU_DAEMON=off` they run here, as before.
- A pointer to a zero value crosses as one (gob sends it as nil): `ChatOptions.Think` of false
  is no thinking, not the model's own choice (`--no-think` thought anyway before).
- Not yet: the models bundled with an app (`Store.Models.Bundle`), and an app's own
  `Store.Models` in the status (it shows the engine's cache).
- M3 (macOS 27, a hwnpud of the worktree on its own socket, the NAS's exports): `hwnpu-image depth` of a
  picture took 60 s the first time (hwnpud fetched and unpacked DA3-BASE's programs, 35 s, then
  compiled the picture's size at its first call, 24 s) and 0.2 s the next. `hwnpu-audio
  transcribe` of 6 s of Japanese: 75 s the first time (Whisper's load compiling, 57 s), the
  speech in 0.7 s. `interpreter models` then showed `transcribe` compiled (hwnpu-audio's
  `whisper-large-v3-turbo`), and `models prepare` made PLaMo 3 Translate 2B and LFM3-Audio-2B-TTS
  ready in 3m10s; `interpreter run` of the recording loaded its three models in 0.3, 1.3 and
  0.6 s and took 7.5 s in all. `hwnpu-audio tts` after it loaded the interpreter's speech model
  in 0.2 s, compiling nothing; `parallax models` and `layers models` showed the depth and face
  models hwnpu-image had run as compiled. Of Core AI's cache only `hwnpud/` grew (18 entries,
  2.5 GB, to 28, 5.5 GB); `hwnpu-image/`, `hwnpu-audio/` and `interpreter/` did not change, and
  no `parallax/` or `layers/` appeared. `hwnpu-chat` through the same engine answered in 1.3 s
  from start to end.
- M3, the other roles (the first time each, the models fetched from the NAS): `hwnpu-image label`
  (SigLIP 2 large) 67 s, `detect` (RTMDet-m) 27 s, `pose` (RTMO-s) 46 s, `ocr` (PP-OCRv6 small)
  14 s, `segment` (EfficientSAM-S, a point: its encoding kept in the engine for the masks) 35 s;
  `hwnpu-audio pitch` and `notes` 11 s each, `generate` of 5 s of music (Magenta RealTime 2,
  its stats whole after the stream) 31 s; `hwnpu-chat search` (PLaMo-Embedding-1B) 61 s, then
  0.3 s; `hwnpu-chat systemone` (d1-2B) 3m34s, then 0.45 s; `voice-agent models prepare`
  (Whisper, LFM3-1.7B, Supertonic 3) 2m49s. Core AI's `hwnpud/` grew from 18 entries to 54
  (2.5 to 8.9 GB); no other program's directory changed.
- M3, one copy and the cache: two `hwnpu-chat -m Qwen3-0.6B` open at once held the model once
  (`hwnpu service status`: "Qwen3-0.6B (chat): loaded once, for 2 clients"; the second `Get` took the
  first's); `hwnpu cache ls Qwen3-0.6B` marked its two units in use and `clear -y` kept them;
  after both ended, the same `clear` would remove them.

On Linux with AMD's NPU, the same supervisor and engine:

- `hwnpu service install` writes systemd's user units `~/.config/systemd/user/hwnpud.socket`
  (`ListenStream=$XDG_RUNTIME_DIR/hwnpud.sock`, mode 0600, `FileDescriptorName=hwnpud`) and
  `hwnpud.service` (`ExecStart=… serve`, `KillMode=mixed`: the supervisor stops the engine
  itself), then `systemctl --user enable --now hwnpud.socket`. The supervisor takes the socket
  from `LISTEN_FDS` (the variables are not passed to the engine) and checks the peer's uid with
  `SO_PEERCRED`. The log is the journal (`journalctl --user -u hwnpud`). systemd's user manager
  gives the service `~/.config/environment.d`'s variables, so `install` warns only of those
  neither it nor a `.env` file sets.
- The engine's chat routes (`/hwnpud/v1/load`, `/v1/chat/completions`, kept for the OpenAI API's
  clients) hold one chat model at a time on AMD's NPU: a load of another closes the one held,
  once its requests have ended (no new request gets it meanwhile). The requests of a model take
  turns on it (`chat.Engine` runs one generation at a time), in the order they came. The models
  of the `ai` sessions (step 3, below) are not among them: the engine holds those side by side,
  chat models too, and nothing closes one for another yet (step 4's budget).
- The NPU lock (`neural-engine.lock`) is taken on AMD too, when a chat, embedding or decision
  model's IRON export loads (in hwnpud's engine and in a program run with `HWNPU_DAEMON=off`), so
  such a program prints `waiting for the NPU: process N (hwnpud) is using it` and waits for the
  engine's idle exit. In the engine every role's models share one process; a program with
  `HWNPU_DAEMON=off` that runs Whisper, speech or a picture model opens the NPU without the lock
  (step 4).
- `hwnpu service status` shows the engine's NPU buffers, which are pinned (`drm-total-memory` of
  `/proc/self/fdinfo`, `mem.DRMMemory`).
- Ryzen AI 9 365 (`hwnpu service install`'s units, and `systemd-socket-activate` for a second socket):
  LFM3-1.7B's export loads in 2.6 s the first time and answers at 17 tok/s; the next run takes
  3.1 s from start to end with no load (1.62 GiB pinned). Three clients at once are served one
  after the other (12 s, 24 s, 43 s). A load of Qwen3.5-0.8B closes LFM3-1.7B first (1.02 GiB
  pinned after). `hwnpu service stop` ends the service; the next connection starts it again.

Step 3 on Linux with AMD's NPU: the same code as on the Mac; nothing in it is the Mac's own.

- `HWNPU_DAEMON` is `on` by default here too: with no hwnpud answering, every command of the
  `ai` package fails and says what to do (`hwnpu service install`, or `HWNPU_DAEMON=off`).
- The engine holds the models of every session side by side, whatever their roles (two chat
  models, a chat and Whisper, ...). Their commands interleave on the NPU, a call of one between
  two calls of another; the barriers of the models that need one (`stepBarrier`,
  `GraphOptions.Barrier`) hold, each graph running its own.
- Ryzen AI 9 365 (2026-11-23, `hwnpud serve` of the worktree on its own socket, the cache's
  exports): through hwnpud ran `hwnpu-chat` (LFM3-1.7B loaded in 1.9 s, 16.9 tok/s as in its own
  process; Qwen3.5-0.8B and 4B), `hwnpu-audio tts` and `transcribe` (the spoken sentence
  transcribed back exactly, Whisper loaded in 0.5 s), `hwnpu-image generate` (FLUX.2 klein, 28 s),
  `depth`, `segment`, `detect`, `label`, `face-detect`, `hwnpu-chat search` (PLaMo-Embedding-1B),
  `hwnpu cache info` and `hwnpu status`. With LFM3-1.7B generating, three `transcribe`s of 2.6 s
  of speech took 2.2 s each instead of 1.0 s, the chat's 16.6 tok/s unchanged. Two chat models
  generating at once (LFM3-1.7B and Qwen3.5-4B, 5.35 GiB pinned) went from 16.6 and 7.7 tok/s
  alone to 6.1 and 5.7 tok/s, both ending in 3.5 min, about as one after the other would. The
  kernel logged nothing. Two large models side by side (Qwen3.6-35B-A3B) were not tried.
- Found on the way, in the programs' own processes as well (`HWNPU_DAEMON=off`), and fixed:
  SigLIP 2's text graph hung its second call (ERT_CMD_STATE_TIMEOUT: `label` embeds its 444
  labels 8 a call), which `GraphOptions.Barrier` cures as for the image models; and a YuNet
  prepared for the CPU while ONNX Runtime was there failed to load once it was gone, where
  `airole.Role.Usable` now makes the store count it as not prepared, and preparing again puts
  it on hwnpu's own executor (`milcpu`, made several times faster for it: [vision-models](vision-models.md)).

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
  for each new engine (`envfile.Environ`). Every 5 seconds, and at `POST /hwnpud/v1/reload`
  (`hwnpu config set`), it compares all of hwnpu's, Hugging Face's and AWS's variables (not
  `HWNPU_DAEMON*`, read when it starts) with those the engine started with: when they changed and
  no request goes on it stops the engine, and the next request starts one with them (so does a
  request that finds no other going on). A model loaded from the old cache is not kept. `hello`
  gives the settings the next engine gets (credentials only as set, passwords taken out of URLs,
  the model policy too), the files read, and `pending`, the names the running engine has other
  values of.
- **hwnpu service, not hwnpud.** hwnpud has only what the service manager runs (`serve`, and
  `engine` for the supervisor); the user looks after it with `hwnpu service
  install|uninstall|status|start|stop|restart` (`internal/servicecmd`), as `tailscale` does
  `tailscaled`. Installed, `start` starts the supervisor (by connecting: the service manager
  starts it) and the engine (`POST /hwnpud/v1/start`), `stop` quits it (the next connection
  starts it again), and `restart` asks it (`POST /hwnpud/v1/restart`) to exit and waits for the
  service manager to start a new one (a new executable too), then starts the engine. Not
  installed, `start` runs `hwnpud serve` in its place, in the foreground (`exec`, with the
  environment it was started with and not what the files set, to read them itself); `stop` and
  `restart` from another terminal work the same, and `restart` of a supervisor run by hand makes
  it run its executable again in its own process (`Run` returns `ErrRestart`, `serve` execs
  itself): the terminal keeps its log. `hello`'s `started` tells the new from the old, of the
  same pid.
- **A client of other settings is told.** A client compares its own settings (its shell and its
  `.env` files, with a project's) with `hello`'s. With `HWNPU_DAEMON=on` it goes through and logs
  that hwnpud's settings are used and which files hwnpud reads; with `auto` it runs in its own
  process with its own settings. `hwnpu service status` shows the settings and what differs in the
  shell it runs in; `hwnpu service install` warns of the variables only the shell sets.
- **The policy narrows, never widens.** A client sends its `HWNPU_MODEL_ALLOWLIST`
  (`LoadRequest.allowlist`, `X-Hwnpu-Model-Allowlist`). hwnpud runs a model when its own policy
  allows it and the definition files' rules with the client's items do too
  (`chat.AllowedAlso`): a client can refuse more, and its `+nsfw` or `+license` does not
  override hwnpud's `-`. An administrator's rules go in the machine's `models.yaml` (`policy:`),
  which every process reads. A family is picked by hwnpud's policy alone so far: when the client
  refuses the pick it is an error, not the next model, until the Accept-like pick (below).
- **A command to change them**, by keys as git config (`envfile.Keys`): the users name settings
  by keys, not by the variables. `hwnpu config set|get|remove KEY` for a single value
  (`cache`, `memory`, ...), `hwnpu config add|get|remove KEY` for a list, an item at a time:
  `get` numbers the items and `remove N` takes the number (or the item). `model-source` is the
  sources in the order they are looked in, kept as `HWNPU_MODEL_SOURCE`, `_1`, `_2`, ... and
  numbered again when one is removed (`add --at 1` puts one first). `model-allowlist` and
  `model-denylist` are the `+` and the `-` items of `HWNPU_MODEL_ALLOWLIST`, written without the
  sign (`hwnpu config add model-allowlist nsfw`, `hwnpu config add model-denylist
  license:non-commercial`). `hwnpu config` lists what is in effect and where it comes from,
  `hwnpu config keys` the keys. They are kept in `~/.config/hwnpu/.env` (`--file` another), its
  comments and other lines kept, which every process reads; a change is applied by hwnpud at once
  (`reload`), and the command tells what wins over the file (the shell, a file read before).
  Credentials in the OS's store (`--secret`, below) are not built yet.

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
  the Mac as now, and the same lock on Linux (taken by the chat, embedding and decision models' IRON exports so far). It releases the lock when it
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

- `HWNPU_DAEMON=on|off|auto`, `on` unless it is set (decided 2026-11-23): every program, the `ai`
  package's apps and hwnpu's commands, goes through hwnpud and fails when it does not answer;
  `auto` takes it when it answers with a matching protocol, else runs in its own process; `off`
  runs in its own process, as a test does. `go run ./cmd/hwnpu-chat` in a worktree runs that
  worktree's code with `HWNPU_DAEMON=off` (the `base/hwnpu-chat` / `new/hwnpu-chat`
  comparisons set it), or against a hwnpud of the worktree on its own socket
  (`HWNPU_DAEMON_SOCKET`).
- `HWNPU_DAEMON_SOCKET` points to another socket, to try a hwnpud built from a worktree beside the
  installed one.
- The first message (`hello`) gives the protocol version, the build, the hash of the model
  definitions and recipes built in (`definitions`: models.yaml and model-recipes.yaml, which decide
  the exports and how the models run) and the settings each role takes (`roles`, the names of
  `ai.LoadSettings`; `airole.Role.LoadSettings`). Added 2026-11-27, after an installed hwnpud of
  11-24 ran newer clients' models by its own definitions and dropped a setting it did not know
  (`reserve_moe_memory`) without a word.
  - Another protocol: the client refuses, as before.
  - No `definitions` (a hwnpud from before), or others than the client's: the client says so in one
    line (`go install ./cmd/hwnpud && hwnpu service restart`), and so does `hwnpu service status`,
    and runs through it. An app built against another version of the module meets the same line;
    it is a note, not an error, as its hwnpud is the machine's.
  - A setting a role of hwnpud's does not take: the client stops before opening (with the names
    the role takes, and the version when it differs). The engine refuses one too (`Open`,
    `CheckSettings`), and so does a process of its own (`HWNPU_DAEMON=off`): a misspelled name is
    an error everywhere.
- The supervisor watches its executable: once it is replaced (`go install`) and no engine runs, it
  exits, and the service manager starts the new one at the next connection. A hwnpud whose file
  was never replaced keeps running: the line above is what tells.
- The socket is mode 0600 and the daemon checks the peer's uid.

## Plan: clients without the engine (not built yet)

Written 2026-11-27. Every program that uses hwnpu links the whole engine: `hwnpu-chat` depends on
88 of hwnpu's packages, hwnpud on 123, and an app of the `ai` package on as many. With
`HWNPU_DAEMON=off` a client runs the models itself. That was the way before hwnpud and stays the
way of the tests, but it has costs beyond turning the sharing off:

- Two paths. The same machine may run a model differently through hwnpud and in a process of its
  own; an investigation done with `off` (Anima's VAE on the M2, 2026-11-27) says nothing of the
  hwnpud path.
- Versions that do not show. The engine's code and definitions decide the exports and how the
  models run; a client carries its own besides hwnpud's (`definitions` above now tells them apart).
- Settings in two places: a client of its own reads its shell; hwnpud reads the .env files.
- aned's compile cache is keyed by the signing identity, `a.out` for every Go build: the more
  binaries that run the Neural Engine, the more often one reuses another's program (a bad prebuilt
  VAE came back in a fresh build, 2026-11-27).
- Every app is as large as the engine, with the shims embedded (Core AI's dylib).

The plan, in steps:

1. The client side alone. Each role package (`ai/text`, `ai/imagegen`, ...) keeps its types, its
   `Remote` and its uses, and moves `Prepare` and `Load` (and what they import) into a package of
   the engine that registers them (`airole.Register`), imported by hwnpud (`internal/aihost`),
   hwnpu-dev and the tests. `internal/daemon`'s client (`Client`, `Use`, `Hello`) moves apart from
   the supervisor and the engine, which import the backends today. A test checks that
   `cmd/hwnpu-chat` and an app of `ai` import none of `internal/llm`, `internal/backend` and
   `internal/coreai`.
2. The commands go through hwnpud only. `HWNPU_DAEMON=off` stays where the engine is linked
   (hwnpud itself, `hwnpu-dev`, `go test`); elsewhere it is an error that says how to run a
   hwnpud of the worktree.
3. Development: a worktree's code runs through a hwnpud of the worktree on its own socket
   (`HWNPU_DAEMON_SOCKET`), which a command of `hwnpu-dev` builds and starts, so that the path
   tried is the path shipped.

To find out first:

- hwnpud runs on the Mac (launchd) and Linux (systemd); Windows has no service yet
  (`install_other.go`): Intel's NPU on Windows needs one before step 2.
- Whether every path of the commands works through the engine as with `off`: the CPU checks
  (`--cpu`, hwnpu's CPU interpreter) and the first preparing (`CompileOnLoad`) in particular.
- `hwnpu-dev` (convert, prebuild, models package) stays a process of its own: it makes the
  programs, it does not serve them.

## Installing and distributing

### Now: built with Go

- `go install ./cmd/hwnpud` (or `go build`) names the executable `hwnpud`, which on the Mac is the
  only key of Core AI's entries. `go run ./cmd/hwnpud` runs under the same name from a temporary
  folder and hits the same cache.
- Go's linker signs ad-hoc as `a.out` on the Mac, as every development build is, so aned's code
  stays shared. No `codesign` step.
- `hwnpu service install` writes the service files pointing at its own path (`os.Executable`, usually
  `~/go/bin/hwnpud`) and starts them (`launchctl bootstrap gui/$UID`, `systemctl --user enable
  --now hwnpud.socket`). `hwnpu service uninstall` undoes it. After `go install` replaces the binary, the
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
| `hwnpu service install` or the `ai` cli's `daemon install`, downloading with Go and checking the sha256 | No | Runs |
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
   `chunkBarrier` still holds). Measured without a gate (Ryzen AI 9 365, step 3 above): a chat
   and Whisper, and two chat models, ran side by side in the engine, their calls interleaved and
   their results right; two chat models shared the NPU's time about evenly. Not yet: models
   whose contexts take more than the NPU has free beside another's.

## Steps

1. hwnpud with the supervisor, the engine, socket start, idle exit, `install` / `uninstall` on the
   Mac and Linux; chat models only, with `hwnpu-chat` as a client (`HWNPU_DAEMON=on`)
2. The model management in hwnpud: list, status, plan, prepare with progress, licenses, gc; the
   commands' `list` / `status` and `hwnpu cache` as clients; `hwnpu config`; the policy in the pick
3. The remote side of `ai.Runtime`, with `HWNPU_DAEMON=on` the default; samples interpreter, layers
   and parallax through the daemon, with the Core AI cache in `coreai-cache/<OS>/hwnpud/` only.
   Built (above)
4. Images and audio; the machine-wide scheduler (program budget, unloading, priority); one copy of
   a model for every client is built (above)
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
