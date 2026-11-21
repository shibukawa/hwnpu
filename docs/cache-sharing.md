# Model cache and sharing it

[日本語](cache-sharing.ja.md)

hwnpu never asks you to convert a model by hand. The first time you use a model it is downloaded, converted for your NPU and kept in a **cache**. This guide explains that cache from four sides:

1. [How the cache works on one machine](#1-the-cache-on-one-machine) (everyone)
2. [Sharing it at home or in a company](#2-sharing-the-cache-with-a-nas-or-s3) with a NAS, a web server or S3
3. [Reference](#3-user-reference): variables, URLs, commands, folder layouts
4. [How each kind of NPU uses it](#4-what-is-shared-on-each-npu) and [how it is built inside](#5-internal-architecture)

If you only want to run models, read part 1. If several machines should not each wait for the same download and conversion, read part 2.

> The history of the design, with every measurement and decision, is kept in [cache-sharing-design.md](cache-sharing-design.md) (Japanese). References to "section 7" and the like in other documents and in code comments point there.

---

## 1. The cache on one machine

### 1.1 What happens the first time

```
hwnpu-chat -m LFM3-1.7B
   │
   ├─ checkpoint (the original weights)   → downloaded from Hugging Face
   ├─ conversion for your NPU             → made on this machine (minutes)
   └─ answers
```

Both the downloaded files and the conversion result stay in the cache. The second run starts in seconds, and other hwnpu commands that need the same model use the same files.

### 1.2 Where it lives

One folder holds everything hwnpu writes:

| OS | Cache root |
|---|---|
| macOS | `~/Library/Caches/hwnpu` (not synced to iCloud) |
| Linux | `$XDG_CACHE_HOME/hwnpu` (`~/.cache/hwnpu`) |
| Windows | `%LOCALAPPDATA%\hwnpu` |

Move it, for example to an external disk, with `HWNPU_CACHE=/mnt/big/hwnpu`. Your own settings (`models.yaml`, `licenses.json`, `run.json`) are not in the cache, so clearing it never loses them.

### 1.3 What is inside

| Kind | What it is | Made when |
|---|---|---|
| original | The model's files as on Hugging Face (`original/OWNER/NAME/`) | First download |
| converted | Programs and exports for your NPU (`models/`, `entries/`, `converted/`) | First use on this chip |
| machine-bound | Things compiled for this exact machine (Core ML `.mlmodelc`, Level Zero blobs, placement results) | First use |
| IRON kernels | AMD's JIT compile cache (`iron/`) | While exporting |
| others | prompts, history, locks, work in progress | Always |

The cache is not unbounded. Before a download hwnpu checks the free space of the disk (default: keep 10 GB free) and, if you set a limit, removes the least recently used model first.

### 1.4 Looking after it: `hwnpu cache`

```
hwnpu cache                          root, free space, limits, sizes by kind
hwnpu cache ls [-s]                  every unit with size and last use (-s: biggest first)
hwnpu cache prune --max-size 200GB   remove the least recently used until it fits
hwnpu cache clear Qwen3-1.7B         remove one model (name, OWNER/NAME or a glob)
```

`prune` and `clear` take `-n` to show what would go without removing it. Whatever was used within the last hour, or is waiting to be uploaded, is kept. After removing something, the next use simply makes it again.

Two settings for the limits (put in the environment or a `.env` file, see 3.2):

```
HWNPU_CACHE_MIN_FREE=10GB     refuse a download that would leave less free (0 to disable, 5% also works)
HWNPU_CACHE_MAX_SIZE=200GB    keep the cache under this size, removing the oldest first
```

### 1.5 Choosing how much memory a model uses

A model can have a smaller form for machines with little memory. To run a model beside other programs on a big machine, tell hwnpu to treat it as a smaller machine:

```
hwnpu memory                      memory of this machine and your choices
hwnpu memory LFM3-1.7B 8          run it the 8 GB way in every command
hwnpu memory LFM3-1.7B --forget   back to this machine's memory
```

The choice is stored in your settings folder (`~/.config/hwnpu/run.json`), not in the cache, so `prune` does not remove it. Switching the form leaves the old form's programs in the cache until they are the oldest; `hwnpu cache clear NAME` removes them at once.

---

## 2. Sharing the cache with a NAS or S3

### 2.1 The idea

Without sharing, every machine downloads the same checkpoint and converts it again. Conversion is the slow part (on an AMD NPU the first export takes minutes; some models take hours). With sharing, **the first machine does the work once and puts the result in a shared place; every other machine only downloads the result.**

You do this by listing **sources**: places hwnpu looks in, in order, before it goes to Hugging Face.

```
1. this machine's cache                always first, always written
2. HWNPU_MODEL_SOURCE, _1, _2 ...      your shared places, in the order of their numbers
3. models installed with the app       read only (for apps that bundle models)
4. Hugging Face                        last; HWNPU_MODEL_HUGGING_FACE=false turns it off
   └─ nothing found → made on this machine
```

A source with `?writeback=true` is also **written to**: whatever this machine downloads or converts is put there, so the next machine finds it. A source without it is only read. A source that cannot be reached (NAS asleep, VPN down) is skipped for the rest of the run with one line on stderr; the next source is tried and your work goes on.

A machine that only runs models needs no checkpoint and no converter. It downloads the converted files from the source and answers. On an AMD NPU that also means IRON, Peano and Python are not needed on that machine.

### 2.2 Setting a source up

Sources are plain URLs in environment variables, so the same setting works in a shell, a service, a container or a `.env` file.

**A folder** (a NAS mounted over NFS or SMB, an external disk):

```
HWNPU_MODEL_SOURCE=file:///mnt/nas/hwnpu?writeback=true
```

Windows: `file:///C:/models` or `file://server/share`.

**WebDAV** (Synology, UGREEN, Nextcloud, nginx with `dav_methods`):

```
HWNPU_MODEL_SOURCE=https://USER:PASSWORD@nas.local:5006/hwnpu?writeback=true
```

If the user and password are not in the URL, hwnpu looks in `~/.netrc` and then at `HWNPU_WEBDAV_USER` and `HWNPU_WEBDAV_PASSWORD`. A NAS usually has a self-signed certificate; trust exactly that one with `?tls_sha256=` (get the value with `openssl s_client -connect HOST:PORT </dev/null | openssl x509 -noout -fingerprint -sha256`). For a quick try on a trusted LAN only, `?insecure=true` skips the check.

**Plain HTTP** (any static web server, such as nginx, serving the same folder): read only.

```
HWNPU_MODEL_SOURCE=http://models.corp.example/hwnpu
```

**S3** (AWS, Cloudflare R2, RustFS and other S3-compatible servers):

```
HWNPU_MODEL_SOURCE=s3://my-bucket/hwnpu?region=ap-northeast-1&writeback=true
HWNPU_MODEL_SOURCE=s3://models/hwnpu?endpoint=http://rustfs.lab:9000&path_style=true&profile=rustfs
```

Keys come from the usual AWS variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) or a `~/.aws` profile (`profile=NAME`; static keys only). If you use SSO, export the keys first:
`eval "$(aws configure export-credentials --profile X --format env)"`. A public bucket needs `no_sign=true`. Cloudflare R2 uses `region=auto`.

Check what is in effect:

```
hwnpu-dev models sources      the root, each source, whether it is written back to, whether Hugging Face is read
hwnpu-dev models ls           what the sources hold
```

### 2.3 Where to put the settings

Put the variables in a `.env` file. hwnpu reads `.env.local` and `.env` at start, and a variable set in an earlier place wins:

1. the process environment
2. the current directory
3. the user's folder: `~/.config/hwnpu/` (Windows: `%LOCALAPPDATA%\hwnpu\`)
4. the machine's folder, put by an administrator: `/etc/hwnpu/` (macOS: `/Library/Application Support/hwnpu/`, Windows: `%ProgramData%\hwnpu\`)

Only names starting with `HWNPU_`, `HF_` and `AWS_` are read, so another project's `.env` does no harm. Put passwords and keys in `.env.local`, which by convention is not committed. Logs and `hwnpu-dev models sources` never show the user and password of a URL.

### 2.4 Three set-ups

**At home: one NAS.** Mount the NAS (or use its WebDAV) and give every machine the same line with `?writeback=true`. The first machine that downloads or converts a model fills it; the others only read. The Mac and the Ryzen AI mini PC share the NAS, though each stores its own converted form (see part 4).

**In a company: one writer, many readers.** Give one machine or CI job (the *builder*) a source with `?writeback=true` and write access. Give everyone else the same source read only (a plain `http://` URL is enough, or an S3 bucket with a read-only key) and `HWNPU_MODEL_HUGGING_FACE=false` if the office network must not reach the internet. Put the variables in `/etc/hwnpu/.env` so nobody has to set them up. Readers cannot spoil the share, and the people who can write are the people you already trust with a binary: an exported program runs on the NPU.

**Several levels.** Sources are tried in order and the one that answers first wins. A source found further back is also copied into the nearer writable ones. So a fast local NAS in front of a remote S3 bucket works as a two-level cache:

```
HWNPU_MODEL_SOURCE=file:///mnt/nas/hwnpu?writeback=true        # fast, in the room
HWNPU_MODEL_SOURCE_1=s3://corp-models/hwnpu?writeback=true     # shared by all sites
```

**Offline.** `HWNPU_MODEL_HUGGING_FACE=false` never contacts Hugging Face. A model that no source holds is an error, not a download. Fill the source from a connected machine first.

### 2.5 What a machine does with a shared model

- **The checkpoint** (original weights) is downloaded through the sources first, from Hugging Face last. Files are checked against the SHA-256 recorded for them; a file that does not match (a bit flip in a big download happened once) makes hwnpu skip that source and try the next.
- **The converted result** is looked up by a key that can be computed before downloading anything, so a machine that finds the result never downloads the checkpoint.
- Some programs are bound to the machine and are not shared; each machine makes them once (see part 4).
- Gated models and models with a license prompt can be written to a shared place too. The license prompt is still answered on each machine that runs the model.

### 2.6 Keeping a shared source tidy

A shared source is a folder tree with no server logic, so tidy it with commands:

```
hwnpu-dev models update [MODEL...]   move a model to Hugging Face's newest commit (only changed files are downloaded)
hwnpu-dev models verify DIR --sums   check conversions against their entry.json
hwnpu-dev models gc 1 --older-than 90 --apply   remove conversions in source 1 made over 90 days ago
                                    (also --model, --kind, --key; without --apply it only shows them)
hwnpu cache clean                   remove what a cut-off upload left: half-written folders, temporary files,
                                    in S3 multipart uploads older than an hour
```

A model's files stay at the Hugging Face commit recorded in `hwnpu-lock.json` until you run `update`, so the whole team runs the same bytes; nobody is moved to a new revision by accident. For S3 it also helps to add a bucket lifecycle rule that aborts incomplete multipart uploads after a few days.

Machines that crash or lose the network in the middle of an upload leave nothing broken for readers: a conversion is visible only when its `entry.json` exists, and a file only when the lock names it.

---

## 3. User reference

### 3.1 Environment variables

| Variable | Meaning |
|---|---|
| `HWNPU_MODEL_SOURCE`, `_1`, `_2`, … | Source URLs, tried in the order of their numbers (3.3) |
| `HWNPU_MODEL_HUGGING_FACE` | `false`: never contact Hugging Face. Default `true` |
| `HWNPU_CACHE` | This machine's cache root |
| `HWNPU_CACHE_MIN_FREE` | Free space to keep. Default `10GB`; `0` off; `5%` |
| `HWNPU_CACHE_MAX_SIZE` | Upper limit of the cache; least recently used removed first. Default none |
| `HWNPU_WEBDAV_USER`, `HWNPU_WEBDAV_PASSWORD` | WebDAV login when the URL and `~/.netrc` have none |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_PROFILE` | S3 keys |
| `HF_TOKEN` | Gated Hugging Face repositories (CI; on a desktop sign in once instead) |
| `HWNPU_MEMORY` | Treat the machine as this many GB |
| `HWNPU_COREAI_PREBUILT` | `off`: do not use the precompiled Core AI parts (4.1) |
| `HWNPU_MODELS`, `HWNPU_ACCEPT_LICENSE`, `HWNPU_BACKEND`, `HWNPU_DEVICE` | Your own model definitions, license prompts, which NPU |

### 3.2 `.env` files

`NAME=VALUE`, one per line; a leading `export ` is allowed; `#` starts a comment; `'...'` is literal; `"..."` understands `\n` and `\"`. Order of precedence is in 2.3. Tests (`go test`) do not read them.

### 3.3 Source URLs

| URL | Kind | Written back |
|---|---|---|
| `file:///path` | Folder | Plain file operations |
| `http(s)://[USER:PASSWORD@]host[:port]/path` | WebDAV, or any HTTP server read only | PUT, MOVE, MKCOL |
| `s3://BUCKET/PREFIX` | S3 | PUT and multipart upload (parts of 64 MiB) |

Parameters, in the query string:

| Parameter | For | Meaning |
|---|---|---|
| `writeback=true` | all | Put there what this machine downloads or converts, and what was read from a later source |
| `tls_sha256=HEX` | https | Trust exactly this certificate (self-signed NAS) |
| `insecure=true` | https | Skip certificate and host name checks. Trusted LAN only |
| `endpoint=URL` | s3 | S3-compatible server |
| `region=R` | s3 | Region (`auto` for R2) |
| `profile=P` | s3 | A profile of `~/.aws` (static keys) |
| `path_style=true` | s3 | Bucket in the path instead of the host name. Default for non-AWS hosts |
| `no_sign=true` | s3 | Public bucket, no signature |

### 3.4 Commands

`hwnpu cache` (for users):

| Command | Does |
|---|---|
| `hwnpu cache` | Root, free space, limits, sizes by kind, sources |
| `ls [-k KIND] [-s] [PATTERN]` | Units with size and last use |
| `prune [--max-size S] [--older-than 30d]` | Remove the least recently used |
| `clear PATTERN [--kind K] [--all]` | Remove named units (asks first in a terminal; `-y`) |
| `clean [-n] [-y]` | Remove leftovers of cut-off uploads from the sources written back to |
| `hwnpu memory [MODEL [GB\|--forget]]` | How much memory a model runs as |

`hwnpu-dev models` (for who runs a shared source):

| Command | Does |
|---|---|
| `sources` | The sources in effect |
| `ls [N] [--original]` | What source N holds (conversions, or the repositories' locks) |
| `push` | Put exports made here (AMD) into the writable sources |
| `export MODEL...` | Make the AMD exports of chat models with IRON on the CPU alone (no NPU), unless a source has them, put them in the writable sources and print their sizes |
| `import DIR...` | Put folders made by hand (`~/models`) into a source: checkpoints that match their Hugging Face repository, and exports |
| `update [MODEL...]` | Move repositories to Hugging Face's newest commit |
| `verify [--sums]` | Check against `entry.json` |
| `gc N [--model M] [--kind K] [--key K] [--older-than DAYS] [--target T] [--apply]` | Remove conversions from source N (source numbers are those `sources` prints; 0 is this machine's cache) |
| `migrate N` | Move an older layout (`blobs/`, `upstream/`, `entries/`) by hard links |

### 3.5 What a source looks like inside

```
original/OWNER/NAME/config.json …     the model's files, as on Hugging Face
original/OWNER/NAME/hwnpu-lock.json   the commit and each file's SHA-256 and size
converted/MODEL/TARGET/EXPORT/VERSION/…   a conversion; entry.json is written last
```

`TARGET` is the kind of NPU (`amd-xdna2`, `coreai`, `any` for things like tables). `VERSION` is `vN` for a model with a version in its definition, `vN-HASH` for a model in development, `v0-HASH` otherwise. The folder can be served as it is by any web server, or browsed with a file manager.

### 3.6 Using it from your own program

```go
s, _ := modelstore.Open(modelstore.Options{}) // root and sources from the environment
p, _ := s.File(ctx, "LiquidAI/LFM3-1.7B", "config.json")
w, _ := s.Weights(ctx, "LiquidAI/LFM3-1.7B")
```

See [USAGE_LIBRARY.md](../USAGE_LIBRARY.md). The same lookup runs behind every hwnpu command.

---

## 4. What is shared on each NPU

What can be shared depends on what the NPU's toolchain produces. The checkpoint (the original weights) is shared on every chip. The converted result differs:

| | Apple (Neural Engine) | AMD (XDNA 2) | Intel (NPU) |
|---|---|---|---|
| Converted form | Core AI `.aimodel` (and Core ML `.mlpackage`) | IRON export: `xclbin`, ELF instructions, weight banks | OpenVINO IR |
| Shared through sources | Core AI exports and tables: yes. Core ML programs: not yet | Chat models, the first-run exports of image models, tables: yes | Checkpoints only; programs not yet |
| Bound to the machine | The Neural Engine compile (per Mac) | Nothing; IRON's JIT cache is local | Level Zero blob (driver, device) |
| Needs a toolchain to make | Xcode 27 + Metal Toolchain for precompiling | IRON, mlir-aie, Peano, Python | OpenVINO |
| A reader needs | Just hwnpu | Just hwnpu (no IRON) | Checkpoint (until programs are shared) |

### 4.1 Apple

- **Core AI** (`target coreai`): the `.aimodel` is a device-independent bytecode, the same for every Mac. The first Mac that converts puts it in the source; any other Mac downloads it and needs no checkpoint. The tables the commands read rows from (`tables.safetensors`) are shared the same way.
- **The first compile is per Mac.** Core AI compiles each part for the Neural Engine on first load (seconds to minutes per part). That result lives in Core AI's own cache, tied to the OS build, and cannot be shared.
- **Precompiled parts** shorten that. On a developer machine with Xcode 27, `hwnpu-dev models prebuild --arch all EXPORT` compiles the parts for each chip generation (`h13g` = M1, `h14g` = M2, `h15g` = M3 …), strips the weights (about 99% of the size; they are rebuilt from the `.aimodel`) and puts the result next to the export, 1–2% of the export per generation. A Mac fetches the one for its chip and finishes in a second or a minute instead of compiling for tens of minutes. It helps only the models whose parts the Neural Engine accepts, so it is chosen per model. `HWNPU_COREAI_PREBUILT=off` skips it. If the disk fills, the machine's copy of those compiled parts is removed first, because it can be rebuilt in seconds.
- **Core ML** programs for the Neural Engine are still made on each Mac. The compiled `.mlmodelc` depends on the OS version and the e5 bundle on the machine.

### 4.2 AMD XDNA 2 (Linux)

- Models are converted to an **IRON export**: a folder with `manifest.json`, the NPU programs (`xclbin`, instruction ELFs) and weights, about 0.5–2 GB per chat model. It is stored in `entries/amd-xdna2/` and in sources as `converted/MODEL/amd-xdna2/EXPORT/vN/`.
- In a source an export is **one packed file**, `files.tar.zst` (tar and zstd), beside `entry.json`, which lists the files inside with their sizes and SHA-256. It is downloaded (resuming if cut off), unpacked and checked file by file; only the unpacked export is kept here. Packing saves 16% over the 13 chat models (65.7 to 55.2 GB) and one upload per file on WebDAV. Exports put loose by an older hwnpu are still read file by file.
- The key of an export is a fingerprint of: the model, the checkpoint commit, the SHA-256 of the definition, the export's options (`amd_export`) and the hash of the bundled IRON tools that export reads: `iron_export_lm.py` and its common modules, the modules of the model's architecture (a table in `tools/amd/lmdeps.go`, followed through their imports), and every kernel. A change to one model's exporter (Gemma 4's per-layer table) or to an image model's leaves the other chat models' exports found. A change that alters the output needs the key (or the version) to change, so a stale export is never matched.
- A machine without IRON (the common case in an office) downloads the export and runs on the NPU after "preparing" for a few seconds.
- Image models that build their programs on first use (`detect`, `pose`, `segment`, FLUX.2 klein) are shared the same way: found in a source, else made with IRON and put back.
- Programs made **by hand** (some image and audio models, depth, face, labels, models without `amd_export`) are kept in `entries/amd-xdna2/` or `~/models`. Use `hwnpu-dev models import ~/models` once to put them in a source; machines that need one fetch it by name and keep a local copy, and the first machine that uses a hand-made one writes it back automatically.
- The IRON compile cache (`iron/`, tens of GB on a developer machine) is never shared. It is a build artifact of the exporter.

### 4.3 Intel

- The checkpoint is shared. OpenVINO programs are converted and compiled on each machine; making them shareable (converted IR as an entry, the Level Zero blob as a machine-bound entry) is planned.
- The key's structure already has room for it: the target `intel-npu3` / `intel-npu4` is part of the entry's path.

### 4.4 Adding a new chip

A chip's backend decides what an entry contains and when it is machine-bound. To share it: write the converted result under a fresh folder, add an `entry.json` last that lists the files with sizes and SHA-256 and the key's recipe, choose a `target` name from the hierarchy (a broad name such as `amd`, then the generation `amd-xdna2`), and register the kind so `hwnpu-dev models put` and the readers know the files. The generic parts (lookup, copying, verification, writing back) need no change.

---

## 5. Internal architecture

### 5.1 Layers and the lookup path

Each layer is a *store*: this machine's cache, then each source, then bundled models, then Hugging Face. They all use the same tree (3.5), so the code that reads a file or an entry does not care which layer it talks to.

```
chat.Fetch / modelstore.File
  → for each layer in order: has the file the lock names? (HEAD, or PROPFIND depth 0, or stat)
     found → download to this machine's original/OWNER/NAME/FILE (4 files at once)
           → check SHA-256 against the lock; on mismatch skip this layer
           → if a nearer layer is writable, queue a copy there
  → not found anywhere → Hugging Face at the lock's commit (resolve/<commit>/)
           → tee to writable sources while downloading
```

The commit is decided once per repository: the definition's pin; else the first source's lock; else this machine's lock; else Hugging Face's `main`. Files come from that commit whichever layer serves them, so two machines never run different bytes of one model. Moving to a newer commit happens only with `hwnpu-dev models update`.

### 5.2 Entries and keys

Everything a conversion produces is an **entry**: a folder tree plus `entry.json`:

```json
{ "key": "<fingerprint>", "kind": "iron-lm", "scope": "portable",
  "model": "LFM3-1.7B", "target": "amd-xdna2", "version": 2,
  "recipe": {...}, "files": [{"path": "decode.elf", "size": 123, "sha256": "..."}],
  "made_by": {...}, "checked": {...} }
```

- The **key** is computed before anything is downloaded, from fixed values in the model's definition (the commit and the files' SHA-256), the options and the versions of the exporter and its tools. That is why a machine can look up a conversion without the checkpoint.
- **Versions.** `format_version` (per vendor) changes when the loader's format changes; the model's `version` (per model and target, in `models.yaml` under `targets`) is raised by hand when the output changes. A source path contains `vN`; a model still in development uses `vN-HASH` where the hash is the key, so each change is its own folder.
- **Scope** says how far an entry may travel: `portable` (programs, exports, tables: any machine), `host` (bound to an OS version, a driver or a chip: only with a machine fingerprint in the key), `local` (prompt cache, history, license answers: never shared).
- All paths in an entry are relative, so a copied folder works anywhere.

### 5.3 Writing back

Writes never show a half-written state to readers:

- A file is written to a temporary name (`.put-…`) in its folder and renamed (S3 shows an object only when the PUT finishes). A conversion becomes visible when `entry.json` is written last; a repository file is visible when `hwnpu-lock.json` names it.
- **Tee** while downloading: bytes from Hugging Face are sent to a writable source as they arrive and committed only if the size and SHA-256 match. If the source falls 64 MiB behind, or the download resumes from a `.part`, the tee stops and the file is queued after the download. A download is never slowed or stopped by a slow source.
- **Convert and upload at the same time**: while a local conversion runs, the export folder is watched; files untouched for 3 seconds are uploaded (4 at a time), so most of the data is already uploaded when the conversion ends. Only the files that changed afterwards and `entry.json` are uploaded at the end.
- Uploads run in a **queue** (4 streams). Whatever is not done before the command exits is waited for; if the process is interrupted, the queue is kept in `work/queue` and finished at the next start. A failure is logged in `work/queue.log` and never fails the command.
- A folder being written carries a mark `.writing-ID` (machine, process, last-write time), refreshed every 5 minutes. `hwnpu cache clean` leaves folders whose mark or files changed within the last hour, so it never removes what another machine is uploading.
- Locks are read, extended and written back; two machines adding files to one repository at the same time may lose one of the additions, which the next machine to read the file adds again.

### 5.4 Transports

- **Folder**: ordinary file operations.
- **WebDAV** (`internal/store/webdav.go`, standard library only): `GET` with `Range` to resume, `PUT` to a temporary name then `MOVE`, `MKCOL` for parents, `PROPFIND` depth 1 to list. Works with servers that do not return `Content-Length` on `HEAD` or that answer 405 where 409 is correct. Retries 3 times on timeouts and 5xx; an unreachable server is switched off for the process with one message.
- **S3** via the `storage/s3` package of tinygodriver (no AWS SDK): resumable `GetRange`, 64 MiB multipart parts with abort on failure, `ListMultipartUploads` for `clean`, retries on 5xx and `SlowDown`. Checked against RustFS and Cloudflare R2.
- **Plain HTTP** is the WebDAV client with write operations disabled.

### 5.5 The local cache

`internal/cache` owns the root and the rules:

- **Space**: a free-space guard before every download (`ErrNoRoom`) and, if a maximum is set, least-recently-used removal of whole units (a repository, a conversion, an IRON kernel). Last use is the newest time of the folder and its files, touched when the unit is read (atime is not trusted). Units used in the last hour, by this process, or waiting in the upload queue are never removed. The entry file is removed first, so readers never see a half-removed unit.
- **Locks** per download, conversion and export (`flock`/`LockFileEx`) so two processes never write one `.part`. Resumed downloads send `If-Range`.
- **Hard links and reflinks** share identical files between models and between the cache and bundled models without copying.
- The machine-bound keys include the chip, so Core ML placement results for an M1 are not used on an M3.

### 5.6 What is not built yet

- A Hugging Face organization (`hwnpu`) of ready-converted models, built and checked by CI and read as the last layer before converting locally. Today a conversion is shared through *your* sources.
- Entries for Core ML and OpenVINO programs, and machine-bound (`host`) entries in shared sources.
- A single `converted/` folder in the local cache for every kind (some still live in `models/` and `entries/`).
- Distinguishing the AMD and Intel NPU generations in the target name.

The design record, [cache-sharing-design.md](cache-sharing-design.md), has the details of each, including the planned CI and fingerprint-lock scheme.
