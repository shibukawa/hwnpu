# Model distribution and the cache

[日本語](distribution.ja.md)

How hwnpu finds, downloads, converts, shares and removes models. This file is the design summary and the decisions. Details live in:

- [../cache-sharing.md](../cache-sharing.md): the user guide and reference (variables, URLs, commands, layouts, per-NPU sharing, internals)
- [../cache-sharing-design.md](../cache-sharing-design.md): the design record (Japanese)
- [../model-table.md](../model-table.md): the models.yaml design; [../../models.yaml](../../models.yaml)'s header documents every field
- [../prebuilt.md](../prebuilt.md): the design for uploading converted models from a source to Hugging Face (`hwnpu/<model>`) and reading them back (not built)

## models.yaml

### Files and overrides

- The built-in [models.yaml](../../models.yaml) is embedded in hwnpu. Files read over it, in order: the machine's (`/etc/hwnpu/models.yaml`; macOS `/Library/Application Support/hwnpu/models.yaml`; Windows `%ProgramData%\hwnpu\models.yaml`), the user's (`~/.config/hwnpu/models.yaml`; Windows `%LOCALAPPDATA%\hwnpu\models.yaml`), `HWNPU_MODELS` (comma separated), `--models FILE` (`-models` outside hwnpu-chat). Loaded by `chat.LoadModels` (`internal/chat/table.go`).
- A later file changes only the fields it writes. A vendor table (`device`, `memory`, `bytes`) changes only the vendors it names. `variants` it names are replaced. `remove: true` drops a model or a use's column. A model override with no `from` or `like` for an unknown name is skipped with a warning.
- `like: OTHER` copies another model's fields except `tag` and `sha256` (the files may be of another repository). So a `like` model must pin its own `sha256`, or it has no recipe key and is never shared (MOSS-SoundEffect-v2.0-lut8 had this).

### Selection

| Way of naming | Picks |
|---|---|
| `-m NAME` | that model |
| `FAMILY:TAG` (`plamo3translate:2b`) | that tag |
| `FAMILY` | the tag of the largest `machine` up to this machine's memory; a memory tag (`image:8g`, `8G`, `8 GB`, or `12g`, not a column) picks as a machine of that memory |
| a use (`chat`, `image`, `depth`, …; `categories:` in models.yaml) | the `picks` column for this machine's memory (`machine: 8/16/24`); a use's steps must match its family's |
| `aliases:` | NAME runs OTHER wherever NAME is picked |
| a Hugging Face repository | that repository (its license is unknown to the policy) |

- `HWNPU_MEMORY` treats the machine as that many GB; `hwnpu memory MODEL GB` stores a per-model choice in `~/.config/hwnpu/run.json`.
- `not_on: [M1, …]`: a family or a use skips the model on that chip and picks the next; the name still runs it.
- `tier`, `prefill`, `format`, `context`, `paged`: how hwnpu-chat runs a chat model when the flags do not say (see [llm-engine.md](llm-engine.md)). `variants: {16: {...}}` changes them, `memory` and `amd_export` on bigger machines.
- `hwnpu-chat list` and `hwnpu-image list [image|depth]` print one table (`chat.ListHave`) with this NPU's DOWNLOAD, MEMORY and RUNS FROM.

### Values measured per NPU

- `bytes` (the files preparation downloads, GB = 10⁹), `memory` (the peak of a whole run, NPU and process together), `machine` (least memory the family recommends it for). How to measure: [new-model-checklist.md](new-model-checklist.md).
- `memory` and `bytes` take one number or a vendor table (`memory: {amd: 3.5, apple: 4.1}`); a vendor left out takes the largest value. Apple's chips may be named too (`{amd: 2.2, apple: 3.1, M1: 1.7}`).
- `device: {amd: cpu}`: where a model runs per vendor (`amd`, `apple`, `intel`; `npu` by default). Order: `--device`, `HWNPU_DEVICE`, models.yaml (`chat.ChooseDevice`, which returns why). Reason: AMD's NPU loses on small models (YuNet 320×256 on Ryzen AI: CPU 3.96 ms / 10 mJ, NPU 30.5 ms / 137 mJ), but Apple's and Intel's run whole graphs, so one vendor's result is not carried to another. A model with no NPU form for this vendor falls back to the CPU and says so; only an explicit `--device npu` is an error.
- `sha256: {FILE: …}` pins files. A file pinned by another model with the same SHA-256 and already downloaded is hard-linked, not downloaded (Qwen3-1.7B as MOSS-SoundEffect's text encoder).

### Licenses and the model policy

- Every model has `license: {id, url, class, terms, accept}` (`TestBuiltinLicenses`). `class`: `open`, `conditional` (a revenue cap, a registration), `non-commercial` (the weights, even when outputs may be sold). Sources of each license: [../model-provenance.md](../model-provenance.md).
- `content: [nsfw]` only on models that make NSFW pictures unasked (Anima, Chroma), not on models that can only if asked.
- Policy (`chat.Policy`, `internal/chat/policy.go`): `HWNPU_MODEL_ALLOWLIST` (also from `.env`) and `policy: [...]` in a user's file (not in the built-in one). Items: `+license:X`, `-license:X` (X = id or class), `+nsfw`.
  - `-` wins over `+`. Any `+license` turns the licenses into an allow-list: an unmatched or unknown license is refused.
  - `nsfw` models are refused without `+nsfw`. `+nsfw` only allows them; it changes no generation behaviour.
  - A use skips refused models; if none is left for this machine, it adds the allowed model of the same kind and features with the most open class. Refused models and why follow the pick's reason.
  - A model named directly (`-m`, a tag, a repository, an API request) is never swapped: `*chat.PolicyError` with the reason. A repository's license is unknown: `-license` lets it pass, `+license` refuses it.
  - Lists show refused models and why at the end; a column with everything refused shows `-`.
  - The `image` use picks FLUX.2 klein 4B from 8 GB. Anima and Chroma are not picked by the use even with `+nsfw`; named, they run with `+nsfw`.
  - Examples (`TestPolicyPicks`): `-license:conditional` → chat 8 GB Qwen3-1.7B (LFM3 has a revenue cap), chat 16 GB gemma-4-E2B-it; `-license:lfm-open-1.0` → speech is an error (no alternative); `+license:open` → translate is an error (PLaMo 3 Translate is conditional).
- License confirmation (`chat.AcceptLicense`): a model with `accept: true` (CircleStone Labs' Anima, for example) asks once on a terminal at first use by name and stores the answer in `~/.config/hwnpu/licenses.json`. Without a terminal it stops and asks for `--accept-license` or `HWNPU_ACCEPT_LICENSE=ID` (this run only). The answer is local and never shared; each machine answers.

## The cache root

- `cache.Root`: `hwnpu` under the OS's non-roaming cache folder (macOS `~/Library/Caches/hwnpu`, Linux `$XDG_CACHE_HOME/hwnpu`, Windows `%LOCALAPPDATA%\hwnpu`). `HWNPU_CACHE` moves it; code uses `modelstore.Options.Root` or `cache.SetRoot`. hwnpu writes nowhere else. User settings (`models.yaml`, `licenses.json`, `run.json`) are in the config folder, not the cache.
- `go test` without `HWNPU_CACHE` uses a per-process temporary root; chat tests use one root per test (`cacheHere`).
- Layout (full table in [../cache-sharing.md](../cache-sharing.md)): `original/OWNER/NAME/` with `hwnpu-lock.json` (same as a source), `models/NAME/` (converted programs, Core AI exports `NAME.coreai`, tables), `entries/amd-xdna2/NAME.amd-npu2*` (AMD exports), `KEY.mlmodelc` / `.blob` / `.json` at the top (machine-bound compile results), `work/` (`.part`, queue, packing; on the same file system for rename and reflink), `tools/` (bundled IRON tools), `iron/` (IRON JIT cache, passed as `NPU_CACHE_HOME`), `locks/`, `prompts/`, `history/`.
- `~/models` is a read-only legacy layer for AMD exports (`cache.ExportRoots`); nothing is written there.
- Completion marks are relative paths (`cache.WriteDone`, `ReadDone`), so a copied root still works.
- Locks (`cache.Lock`, flock / `LockFileEx` in `locks/`): per download file, per chat conversion program, per IRON export. The waiter finds what the other process made.

## Sources

### Order of layers

```
0. this machine's cache (cache.Root)      always written, used in place
1. HWNPU_MODEL_SOURCE, _1, _2, ...         in number order; a source in error is skipped
2. models bundled with the app            read only, used in place
3. the converted models on Hugging Face   models.yaml's hub (repository and commit by target); HWNPU_MODEL_HUB=false skips it
4. Hugging Face                           HWNPU_MODEL_HUGGING_FACE=false never contacts it (3 neither)
   none has it -> made on this machine
```

- A source is a URL: `file:///path` (mounted folder), `http(s)://[USER:PASS@]host/path` (WebDAV; read only it needs only GET and HEAD, so any HTTP server), `s3://BUCKET/PREFIX?endpoint=…`. Parameters (`writeback`, `tls_sha256`, `insecure`, `region`, `profile`, `path_style`, `no_sign`): [../cache-sharing.md](../cache-sharing.md). Parsed in `internal/store/source.go`.
- `?writeback=true` puts there both what this machine makes and what it read from a later layer (gated models too). A source found further back is copied into the nearer writable ones, so a NAS in front of S3 works as two levels.
- `.env` and `.env.local` are read from the current directory, the user's config folder, then the machine's (`internal/envfile`, imported blank by `internal/console` and `internal/cache` so they load first). Only `HWNPU_`, `HF_`, `AWS_` names, only when not already set. Not read under `go test`. URL passwords are hidden in output.
- Bundled models (`store.SetBundles`, `store.Dir.InPlace`: hard link, else symlink; the lock is trusted): the installer places them in `configdir`'s resource folders (`share/hwnpu/models`), an app sets `modelstore.Options.Bundle`. No environment variable.
- Every command uses the same lookup, also public as `modelstore` ([api.md](api.md)). Commands have no `--dir`; the root is `HWNPU_CACHE`.

### Layout in a source

```
original/OWNER/NAME/FILE              as on Hugging Face
original/OWNER/NAME/hwnpu-lock.json   commit, and each file's sha256 and size
converted/MODEL/TARGET/EXPORT/VERSION/   a conversion; entry.json written last
```

- A file the lock does not name is not read. A repository copied by hand without a lock (a gated checkpoint on a NAS) is not found, and the fetch goes on to Hugging Face.
- The commit is decided once per repository: the models.yaml pin (`from: hf:OWNER/NAME@REV`), else the first source's lock, else the local lock, else Hugging Face's `main`. Every file comes from that commit whichever layer serves it. Only `hwnpu-dev models update` moves a lock (re-downloads changed files only). When a source's lock moves, the local one follows and files with the same sha256 stay (`store.Record`).
- Fast path: when the local lock names the file, its size matches and the commit is the decided one, no network and no source is asked.
- Every original file, from any layer, lands in local `original/OWNER/NAME/FILE` (`chat.RepoDir`); `chat.Fetch(ctx, repo, name, log)` and `chat.FetchWeights` return the path. Parts shared by models (Chroma's T5-XXL and VAE) are in each repository's folder, hard-linked by sha256.
- `TARGET`: `amd-xdna2`, `coreai`, `coreai-ARCH` (prebuilt), `any` (tables, GPTQ codes). `VERSION`: `vN` when models.yaml has `targets.<target>.version: N` (matched if the recipe without the tools hash is the same), `vN-<8 hex>` with `develop: true`, `v0-<8 hex>` with no version.
- A published version (`vN`) no layer has is taken from `v0-HASH` of the same key (the same bytes, made before the version was written) and written back as `vN` (`chat.fetchFrom`).
- A folder is either whole or not: a conversion is whole when its `entry.json` exists; a repository holds only what its lock names. Readers never see half-written data.

### Keys (computed before any download)

| Kind | Key |
|---|---|
| AMD chat export (`chat.ExportRecipe`, `chat.SharedExport`) | model, checkpoint commit, models.yaml sha256, `amd_export` arguments, hash of the bundled IRON tools it reads |
| AMD first-run exports (detect, pose, segment, FLUX.2 klein; `chat.PinnedRecipe`) | models.yaml sha256 of every input, tool name and hash, a Go-side version (`detect.ExportVersion`, …, `AMDExportVersion`) raised by hand when the arguments change. No pins → no key → not shared |
| AMD export made by hand (`iron-import`, `chat.AMDNamed`) | its name (`NAME.amd-npu2`, `NAME-1024.amd-npu2`, `NAME-SIZE.amd-npu2`, `NAME.amd-npu2-q4`) |
| Core AI export (`coreAIRecipe`, `kind: coreai`) | model, repository, sha256 (else the files' commit), repositories read with it, export name (holds version and format), `ailower.FormatVersion` |
| tables (`kind: tables`, `chat.Tables`) | model part, architecture, table name, `tablesVersion` |
| GPTQ codes (`kind: codes`, `converted/MODEL/any/codes-HASH`) | model part and the index.json hash |
| Core AI prebuilt (`coreai.ProgramsDigest`) | each `.aimodel`'s path and `main.hash` sha256, architecture, export name |

- A model with neither sha256 nor commit known is not shared. Raise a version by hand when what is made changes (`ailower.FormatVersion` 2 changed every Core AI key; all exports had to be remade).
- GPTQ codes are needed only to convert, so they go to the sources too (`chat.putCodes`, `fetchCodes`); without them nobody can remake an export after a key change.

### What goes through the sources

- Originals (checkpoints), with sha256 checked; a mismatch skips that layer.
- AMD exports: chat models, first-run image exports, exports made by hand (fetched by name into `entries/amd-xdna2`, then used locally; a hand-made one without `entry.json` is described and written back on first use, `chat.ShareMadeByHand`). Not fetched on a machine without AMD's NPU.
- Core AI exports (`llm.FetchExport`, `llm.ShareExport`): one `.aimodel` set for every Mac (chip compilation stays in each Mac's Core AI cache). The check runs before the checkpoint download, so a hit downloads no checkpoint. Split-attention models are written back only after all attention programs are added (`shareMade`). depth and MoGe-2 share per size subfolder.
- Tables and GPTQ codes. With exports and tables in a source, a machine that only runs a model downloads no checkpoint (only config, tokenizer, index).
- Not shared: Core ML programs, OpenVINO / Level Zero programs, the IRON JIT cache, compile results bound to a machine, license answers, prompts, history.

### Writing back

- Order: data first, `entry.json` (or the lock entry) last. File writes go to `.put-…` then MOVE (S3 shows an object only when complete).
- Tee: a Hugging Face download is streamed to folder and WebDAV sources as it comes, committed only if size and sha256 match. A source 64 MiB behind (`store.TeeBuffer`), S3, or a resumed `.part` drops out and is queued after.
- Non-packed conversions are put while being made (`store.StartEarly`: files idle 3 s, `EarlySettle`); `PutEntry` puts the rest. Packed ones (AMD, Core AI) are put once whole.
- The queue (`store.PutLater`, `work/queue/`, 4 streams) finishes before exit, resumes at the next start, logs failures to `work/queue.log`, and never fails the command. `hwnpu-dev models flush` waits for it.
- A folder being written carries `.writing-ID` (JSON: machine, process, start, last write), rewritten every 5 minutes, removed at the end (`store.MarkWriting`). Every writing path sets it (tee and early puts too).
- Packing (`store.Packs`, `files.tar.zst` beside `entry.json`): AMD and Core AI exports go as one tar+zstd file. The gain is fewer requests (one PUT+MOVE on WebDAV), size 1–5% for Core AI (LFM3-1.7B 2.62 → 2.59 GB), 16% over 13 AMD chat exports (65.7 → 55.2 GB). Download resumes, sha256 of the pack, then each file checked against `entry.json`; only the unpacked export is kept. Loose exports from older hwnpu are still read; `hwnpu-dev models repack` packs them.

### Transports

- WebDAV (`internal/store/webdav.go`) uses the standard library only; plain HTTP is the same client without writes. S3 (`store.S3`) uses tinygodriver's `storage/s3`, no AWS SDK or CLI; static keys only (SSO via `aws configure export-credentials --format env`); checked on RustFS 1.0.5 and Cloudflare R2. Details: [../cache-sharing.md](../cache-sharing.md).
- Pitfalls that still apply:
  - A NAS's self-signed certificate (no host name) fails Go's check and the source is skipped: use `tls_sha256=`.
  - Every response body must be read and closed, or Go opens a new connection per request; with mDNS (`.local`) each lookup cost 5.01 s. `cachedDialer` resolves a host once. After the fix, 1.5 GB in 95 files plus 0.51 GB of tables uploaded in 60 s; a 96 MB download went 45 s → 2.3 s (UGREEN NAS, M2).
  - Throughput Wi-Fi ↔ UGREEN NAS: 5 MB/s with one stream, 9 MB/s with four.
- Round trip (Qwen3.5-0.8B, Core AI, empty `HWNPU_CACHE`, WebDAV NAS, M2 16 GB): NAS empty 10 min 41 s to ready; NAS full 45 s (2.2 GB of exports plus 12 MB of config, tokenizer, index; no checkpoint). Results from the source are byte-identical to the converting machine's.

## Downloads

- Only the files the converter reads, never the whole repository. Files of one family are downloaded once and shared.
- Before each download `cache.Room` checks space (see below).
- Several files at once: 4 streams (`store.FetchStreams` for converted entries, largest first; `chat.FetchFiles` for originals of image, audio and split chat checkpoints). Every progress bar shows the remaining total of the plan (`chat.Expect`).
- Resume: Hugging Face downloads keep the ETag in `NAME.part.etag` and resume with `If-Range` only on the same version. `X-Repo-Commit` (also on the response before a redirect) is the commit recorded when putting in a source.
- Lookup in sources before Hugging Face: by the models.yaml sha256, else `X-Linked-Etag` from a HEAD (LFS files), else the commit's lock.
- A model that already has its conversion locally or in a source checks only small files: e.g. `mtpArch` reads a split checkpoint's `model.safetensors.index.json` (`weight_map`) instead of downloading the weights.

### Gated repositories (`internal/chat/hfauth.go`)

- Token order: `HF_TOKEN`; the huggingface_hub token file (`HF_TOKEN_PATH`, `$HF_HOME/token`, `~/.cache/huggingface/token`, read only); a sign-in made in this process.
- On 401, if the API (`/api/models/OWNER/NAME`, no token) says `gated` and a terminal is present, sign in with OAuth 2.0 device flow (RFC 8628): hwnpu's public OAuth app (`hfClientID`, no secret; `HWNPU_HF_CLIENT_ID` replaces it), scopes `openid` and `gated-repos` only (reads public gated repositories the account accepted; no private repositories, no profile). The user enters the code at https://hf.co/oauth/device. Works over SSH and on machines without a browser.
- Tokens are kept in process memory only. The access token lives 15 minutes; the refresh token (also memory only) renews it silently on 401; if that fails, sign in again.
- 403 (account has not accepted): show the repository page, retry on Enter.
- No terminal (`hwnpu-chat serve`, scripts): an error with the page and how to make `HF_TOKEN`.
- A 401 the API does not call gated (unknown or private name) is a plain error, no sign-in.
- `HF_TOKEN` is recommended only for CI.

## Core AI prebuilt (per chip)

Details and Apple specifics: [apple-coreai.md](apple-coreai.md), [../cache-sharing.md](../cache-sharing.md).

- What: `coreai-build` output (`X.ARCH.aimodelc`) per chip, without the weights. `coreai.Pack` maps `resources.bin` (99% of the size) to `resources.bin.map.json` against the blobs of `main.mlirb` (a 231 MB part → 2.97 MB, rebuilt byte for byte). `coreai.Bundle` makes `prebuilt.tar.zst` per chip (one chip only is fetched, so not one archive for all).
- Fetch: when `Compile` loads a part with no Core AI cache entry and no local prebuilt, once per export per process, `coreai.FetchPrebuilt` asks the sources and stores in `EXPORT.prebuilt/ARCH/`. `coreai.Assemble` rebuilds in a temporary folder (small files hard-linked, weights read from `main.mlirb`, size and sha256 checked; never from a different bytecode), Core AI opens it, the folder is removed. `HWNPU_COREAI_PREBUILT=off` skips it.
- Put: `hwnpu-dev models prebuild --arch LIST EXPORT` (`--packed` packs without putting, `--no-put` leaves the programs whole beside their sources). Build, pack and drop each part's full `.aimodelc` one at a time (Sana's 27 parts × 12 chips whole would be 50 GB). Build one chip at a time: 12 at once put none. Watch disk: `$TMPDIR/com.apple.MetalPerformanceShadersGraphHost` kept 12 GB after `coreai-build`.
- The `.aimodel` export is needed even where a prebuilt exists (the weights come from it).
- Which models get one: only where it pays.

| Model | Native first run | With prebuilt | Machine |
|---|---|---|---|
| FLUX.2 klein 4B 1024² | 52 min 16 s | 16 min 55 s | M2 16 GB |
| FLUX.2 klein 4B 1024² | 40 min 08 s | 7 min 48 s | M1 16 GB |
| Anima-Turbo 1024² | 45 min 01 s | 37 min 50 s | M2 16 GB |
| Chroma1-Flash 1024² DiT, one 4-block part | 219 s | 66 s | M2 16 GB |
| LFM3-1.7B (chat) | 2 min 17 s | 8 min 12 s | M2 16 GB |
| plamo-3-translate-8b (chat) | 1 min 13 s | 7 min 08 s | M2 16 GB |

  - Image models with many slow parts win (FLUX.2 klein, Chroma1-Flash, Anima: put with `--arch all`). Chat models lose: the first prebuilt part of a process costs a fixed 3–6 minutes in `ANELargeModelCompilerService`; Qwen3.5-4B loaded but did not answer for 1 h 40 min. Chat models have no prebuilt.
  - Steps per second are the same. FLUX.2 klein's picture differs slightly with prebuilt (PSNR 28.9 dB); Chroma and Anima match byte for byte.
- Constraints:
  - Only 12 chips work (h13g, h14c/g, h15c/g, h16c/g/s, h17c/d/g/s). On M1 Pro / Max / Ultra, M2 Pro / Ultra, M3 Pro / Ultra, h15m and A18 `coreai-build` crashes in `ANEHostCompiler` (SIGSEGV, any program). M6 (h18g) fails with `redefinition of symbol named 'hwnpu_N'` and falls to GPU. Reported to Apple.
  - ANE code only with `--preferred-compute neural-engine`.
  - Device-side compiled results (aned's `/Library/Caches/com.apple.aned`, inside SIP) cannot be exported, so only `coreai-build` output is distributable.
  - Low disk breaks the first run: with 12–22 GB free, macOS's `deleted` purged freshly finished programs, aned reported `Model load failed … isPreCompiledModel=1`, and Core AI silently compiled on the device (first step 5–15 min). Keep 25 GB or more free when measuring a first run.
  - Image exports hold every aspect ratio in one export, so the prebuilt covers all sizes at once and must end with `0 failed` ([new-model-checklist.md](new-model-checklist.md)).

## Sharing identical files on disk

Compression does not help (bf16 checkpoints: lz4 100%, zstd −3 77%; lut6 weights: zstd 87–90%; and mapped files need an unpacked copy anyway). The waste is duplicates: identical `weight.bin` in Q=8 and Q=64 programs, the `.mlmodelc` copy of the package's weights, the same banks in several AMD exports. Fix: share blocks with copy-on-write clones, which stay ordinary files.

| OS | `cache.Clone` uses | Shares on | No-op on |
|---|---|---|---|
| macOS | `clonefile(2)` | APFS | — |
| Linux | ioctl `FICLONE` (plain `syscall`, no x/sys) | Btrfs, XFS with reflink, bcachefs, ZFS with block cloning | ext4, tmpfs |
| Windows (`clone_windows.go`) | `FSCTL_DUPLICATE_EXTENTS_TO_FILE` | ReFS, Dev Drive | NTFS |

- Clone first, then compare the clone with dst: a file system that refuses costs no reads, and writes to src during the compare cannot leak into dst. dst keeps its mode and mtime.
- `cache.ClonePrefix`: shares the common leading blocks of same-named files of different size (FICLONERANGE on Linux, same call on Windows; nothing on macOS). Example: a text-only Gemma 4 bank (1.67 GiB) is the prefix of the multimodal bank (2.64 GiB). Common parts under 1 MiB are left.
- Already shared? Linux: `sharedPrefix` compares FIEMAP extents (0.5 ms per export; never over-reports). macOS and Windows have no cheap query: compare once, then a mark.
- Where it runs:
  - conversion: `llm.convertPart` clones identical program weights in the same folder;
  - compile: `coreml` clones the `.mlmodelc` weights from the package;
  - existing files: `llm.ShareWeights` once per model folder in `chat.Prepare` (mark `.weights-shared`);
  - AMD exports: `iron.ShareExport` / `cache.ShareFiles` before loading in hwnpu-chat, hwnpu-agent `-iron` and hwnpu-image: files ≥ 1 MiB that match a sibling export (has `manifest.json` or `.amd-npu` in its name) or an earlier file. Only the folder being loaded is changed. Mark `.files-shared`; newer files trigger a re-check. Symlinks are not followed.
  - originals: a file pinned by sha256 in another repository is hard-linked.
- Measured: M3, model folders' 24.6 GiB of weights fully shared; weights on disk ~49.6 → ~20 GiB. Btrfs, two Gemma 4 E2B AMD exports (7.0 + 8.2 GB): 15.2 → 8.2 GiB, 4 s; outputs unchanged.
- Pitfall: the first load after replacing existing files with clones took 20–50 s per program (the Neural Engine's system cache sees a new file and recompiles); later loads ~2 s. New conversions are shared from the start, so this happens once.
- Sizes count hard links once (dev/ino). `df` is useless for these comparisons on macOS (swap and system caches move it by tens of GB); APFS `ATTR_CMNEXT_PRIVATESIZE` gives the unshared size.

## Local cache size and cleaning

- `cache.Room(need, path, log)` runs before every fetch from a source (`store.FetchOriginal`), every converted fetch (`store.materialize`) and every Hugging Face download (`chat.fetchHF`, by Content-Length).
  - `HWNPU_CACHE_MIN_FREE` (default 10 GB; `0` off; a percentage allowed): refuse to start below it.
  - `HWNPU_CACHE_MAX_SIZE` (default none): when over it, or short of free space, remove least recently used units.
  - Not enough → `ErrNoRoom`; `chat.fetch` stops and does not fall through to Hugging Face.
  - Why no default maximum: some units (AMD exports) take long to remake, so automatic removal happens only when the user sets a limit. The free-space floor is on by default so the disk never fills.
- Units: an original repository, `models/NAME`, `converted/.../vN`, `entries/TARGET/NAME`, `iron/KEY`, a root compile result per key, a `prebuilt` unit (`EXPORT.coreai.prebuilt`, not counted as model size). Last use = folder mtime advanced by `cache.Touch` (atime is not trusted), called by the fast path of `chat.fetch`, `chat.ModelDir` and `cache.FindExport`; listing (`Downloaded`) does not touch.
- Never removed: used in the last hour (`cache.KeepRecent`), used by this process, waiting in the write-back queue (`cache.SetBusy`). The scan runs once per process.
- Short of space with prebuilt available, Core AI cache entries of parts that have a prebuilt (`coreai-cache/<OS>/hwnpu*/<hash>`, a copy of the weights) go first: they rebuild in seconds. The prebuilt itself stays.
- The machine-bound compile cache (`cache.Prune`, called after a chat load): entries whose source package (`<key>.src`) is gone, entries unused for 30 days, temporary folders older than a day.
- Commands: `hwnpu cache [ls|prune|clear|clean]` for users, `hwnpu-dev models …` for whoever runs a source (source 0 is the local cache). Flags: [../cache-sharing.md](../cache-sharing.md).
- `modelstore` exposes the same (`Options.Limits`, `Usage`, `PlanPrune`, `Prune`, `Remove`, `PlanClean`, `Clean`, `ErrNoRoom`).

### Cleaning a source (`hwnpu cache clean [-n] [-y]`, `store.PlanClean`)

- `converted/…/VERSION` without `entry.json`: everything may go. With it: only `.put-` leftovers go.
- `original/OWNER/NAME`: files the lock does not name (old commits, `.put-`) may go; no lock → everything.
- Outside `converted/` and `original/` (a NAS's `.DS_Store`): untouched.
- A folder is "being written", and kept, if it has a `.writing-ID` mark whose last write is within an hour (`store.WritingStale`), or any file time (WebDAV `getlastmodified`, mtime, S3 LastModified) within an hour. A mark alone was not enough: an older hwnpu without marks started writing between plan and apply. Each folder is re-checked just before removal; the mark goes last.
- S3 multipart uploads never completed or aborted (billed, not listed) under `converted/` and `original/` older than `WritingStale` are aborted (`ListMultipartUploads`, tinygodriver v1.3.10).
- `hwnpu-dev models gc N … --orphans` uses the same rules.

## Open

- Hugging Face distribution of converted models (organization `hwnpu`, one repository per model, uploaded by hand from a source with `hwnpu-dev models hf push`, read as a layer before the original repository): the layer, models.yaml's `hub` pins and `hwnpu-dev models hf plan / push / check` are built; nothing is uploaded yet. Left: a device flow to write, splitting exports over 20 GB, stopping when a gated repository refuses ([../prebuilt.md](../prebuilt.md)). Today sharing goes through the user's sources.
- Local layout: move `models/NAME` programs and `entries/` to `converted/MODEL/TARGET/EXPORT/vN/` with `entry.json` as the only completion mark (now `NAME.done`, `NAME.ext.done` and `entry.json` coexist); migrate `~/models` once and stop reading it; `modelstore` API to make conversions (`Make`); use bundled conversions in place (now copied).
- Entries for Core ML and OpenVINO programs; machine-bound (`host`) entries in sources; AMD / Intel NPU generations in the target name.
- Hand-made image exports whose tool arguments Go does not own (Anima): only by name today.
- `hwnpu-dev models put` does not know the first-run FLUX.2 klein export.
- Not checked on hardware: AWS S3, Synology WebDAV, Windows locks and ReFS / Dev Drive cloning, bcachefs, ZFS, NFS cloning; disk-space reads on Windows and macOS during a real download.
- Core AI chat exports of the 8 GB column (LFM3-1.7B, Qwen3.5-0.8B, Qwen3-1.7B) not yet remade and put after `FormatVersion` 2. Anima-Turbo's 1024² Core AI export not yet in the sources. Qwen2.5-Coder-7B's GPTQ codes must be regenerated (`tools/ref/qwen2_gptq.py`) and converted once.
- The fixed first-part cost of prebuilt finishing, the 9 crashing chips, M6's name clash and FLUX's numeric difference wait on Apple (a new Metal Toolchain is needed to rebuild).
- Storing the HF refresh token (OS credential store) and exchanging Actions tokens (Trusted Publishers, RFC 8693).
- Checkpoints are kept open at run time for embedding tables; writing only the tables at conversion would shrink LFM3 3.5 → 0.27 GB and Gemma 4 10.2 → ~5.5 GB (needs re-download to reconvert).

## Not adopted

- Compressing checkpoints or weights on disk: lz4 ~100%, zstd 71–90%, and mapped files need an unpacked copy.
- Sources in models.yaml (`sources`, `search`, `cache.layers`, `from: SOURCE:NAME`), `HWNPU_CACHE_LAYERS`, `HWNPU_CACHE_WRITE`: replaced by source URLs in the environment; a file that still writes them gets an error with how to move.
- Content-addressed layout (`blobs/sha256/…`, `upstream/`, `entries/KIND/KEY.json`) and its dedup: unreadable for people and not servable by a plain HTTP server. `hwnpu-dev models migrate` moves old data by hard links.
- `HWNPU_OFFLINE` (use `HWNPU_MODEL_HUGGING_FACE=false`) and `HWNPU_PREBUILT` (a prebuilt is used if a source has it).
- Saving the Hugging Face token to disk: memory only.
- Fetching the ONNX Runtime library automatically: native library distribution is the user's choice (`HWNPU_ONNXRUNTIME` or the system's).
- A default cache size limit: remaking some units takes long.
- Fetching prebuilt as one archive for all chips (36.98 → 1.33 MB with zstd −19 --long): each Mac needs only its own.
- Prebuilt for chat models: slower first load, and Qwen3.5-4B hung.
- Exporting device-compiled Core AI results: inside SIP, per signing identity.
- Sana (every model) and Qwen-Image-2.1 (Turbo and base, with the Qwen Research license prompt): removed from models.yaml. Qwen-Image has too many parameters for the NPU; Sana went with it, and its Core AI prebuilt did not reach the ANE for most parts anyway.
- MinIO as a tested S3 server: no longer maintained.
- Writing into `~/models`: read-only legacy layer.
- Large models in Cloudflare R2's free tier: it holds 10 GB, so only small ones go there.
