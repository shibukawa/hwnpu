# Reproducing the reports

The programs and scripts that reproduce each report, how to make them, and what they showed on 2026-11-25 (MacBook Air M2 16 GB, macOS 27.2.1 26B131, coreai-build 3605.6.3).

Tools (`tools/coreai/fbrepro`):

- `fbrepro`: writes the small programs: `sin`, `clipew`, `prescale`, `snake [-clip] [-square]`, `attn [-rows] [-keys] [-chunk] [-head]`. They are fp16 with constant random weights and need no checkpoint.
- `offline.sh ARCH PROGRAM...`: compiles each program with `coreai-build` for ARCH and says whether `manifest.plist` has `mps.fullyPlacedOnANE`.
- `load.sh PROGRAM[#FUNCTION]`: loads it on this Mac's Neural Engine with no cache for it (a copy of `aicheck` signed with a new identifier) and samples the free disk.
- `purge.sh PROGRAM [KEEP_GB]`: loads, fills the disk to KEEP_GB for 5 minutes, then loads again.

```sh
go build -o /tmp/fb/fbrepro ./tools/coreai/fbrepro
go build -o /tmp/fb/aicheck ./tools/coreai/aicheck
cd /tmp/fb
./fbrepro sin sin.aimodel
./fbrepro snake snake.aimodel
./fbrepro snake -square square.aimodel
./fbrepro attn -chunk 1024 attn-1536-c1024.aimodel   # a 512-row chunk is left
./fbrepro attn -chunk 768 attn-1536-c768.aimodel     # the workaround
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  $REPO/tools/coreai/fbrepro/offline.sh h13g sin.aimodel snake.aimodel square.aimodel
AICHECK=./aicheck $REPO/tools/coreai/fbrepro/load.sh sin.aimodel
```

`aicheck run` prints `ane=true` when the program runs on the Neural Engine.

## compile-disk-and-time.md (new)

- Program: `vae_s6.aimodel` of Anima-Turbo's Core AI export (`anima-turbo-v1.1-8sizes-ffe2f7fe-int8-int8-g4-q512-v5.coreai`, 832 KB, 8 size functions), from `hwnpu-dev models fetch Anima-Turbo/coreai`.
- `load.sh "vae_s6.aimodel#s1024x1024"` on an M2 with 70 GB or more free.
- Note: Core AI compiles every function of a program together, so this load compiles all 8 sizes.

## purged-programs-silent-recompile.md (new, known-issues 8)

- `purge.sh PROGRAM 11` on a test machine. Use a program that takes minutes to compile (`vae_s6.aimodel#s1024x1024`), so that the recompile after the purge is easy to see.
- What it showed on 2026-11-25 in normal use: [the report](purged-programs-silent-recompile.md). The script itself was not run on the M2 used day to day.

## sin-off-ane-m1-m2.md (new, known-issues 10)

| program | h13g offline | h14g offline | h15g offline | M2 on the device |
|---|---|---|---|---|
| `sin` | not on ANE | not on ANE | ANE | not on ANE |
| `snake` (with sin) | not on ANE | not on ANE | ANE | not on ANE |
| `snake -square` (y*y for sin) | ANE | ANE | ANE | ANE |
| `clipew` | ANE | ANE | ANE | ANE |
| `prescale` | ANE | ANE | ANE | ANE |

`coreai-build` exits 0 in every case, also when the program is not on the Neural Engine.

- `sin` reproduces on M1 and on M2 (h14g), offline and on the device; M3 (h15g) takes it.
- The fused clip ("NEFUSED_EW: EW: MAX") and the constant-scaled subtraction ("KernelPreScales") do not reproduce with these small programs offline for h13g or on the M2. They were seen with the M1's device compiler in the full DAC stage: check `clipew` and `prescale` on an M1 with `load.sh`, and if they pass, attach the DAC stage instead (below).

## offline-stricter-than-device.md (new, known-issues 11)

- `snake -clip` passes offline for h15g, so this small program does not reproduce it.
- It needs the DAC's stage 4 as hwnpu wrote it before r3 (Snake with `sin` and a clip): a MOSS-SoundEffect export from a commit before `DACRevision` 3, then `aicheck merge` with `HWNPU_DEV=ailower.cut=47` for the first 47 operations (known-issues 11).
- Compile it with `offline.sh h15g`, and load it on an M3 with `load.sh`.

## h15g-dma-cache-hint.md (new, known-issues 12)

- `attn-1536-c1024.aimodel` (a chunk of 1024 query rows and one of 512, over 1536 keys) and `attn-1536-c768.aimodel` (two chunks of 768). Both are all on the Neural Engine offline for h13g, h14g and h15g, and on the M2's device.
- The failure is in the M3's finishing of the h15g `.aimodelc`. Make them with `coreai-build compile ... --architecture h15g` (60 KB each), then on an M3 load the `.aimodelc` (`load.sh attn-1536-c1024.h15g.aimodelc`) while `log stream --predicate 'process == "ANELargeModelCompilerService"'` runs. Expect "cache_hint" for c1024 and none for c768.
- Not checked on an M3 yet with these programs.

## h13g-attention-8192-keys.md (new, known-issues 13)

- The program needs FLUX.2 klein's weights: `flux2klein`'s `TestWriteSingleProbe` with `HWNPU_TEST_KLEIN_PROBE_OUT`, `HWNPU_TEST_KLEIN_DIR` (the checkpoint), `HWNPU_TEST_KLEIN_PROBE_REFS=1`, `HWNPU_TEST_KLEIN_PROBE_GROUP=0,2`, `HWNPU_TEST_KLEIN_PROBE_REF=...mlpackage` (known-issues 13).
- `hwnpu-dev prebuild -arch h13g`, then on an M1 `go run ./tools/coreai/aicheck run -cpu -ref REF OUT#s1024x1024_ref1` and look at the cosine.
- Not reproduced on 2026-11-25 (it needs an M1).

## Additions

- FB25624129: the measurement of [add-FB25624129-finish-slower.md](add-FB25624129-finish-slower.md) was made with hwnpu-chat / hwnpu-audio copies signed with new identifiers. Each model was loaded once compiling on the device (`HWNPU_COREAI_PREBUILT=off`) and once from `hwnpu-dev prebuild -arch h14g` of all its programs.
- FB25624259: start `load.sh "vae_s6.aimodel#s1024x1024"`, stop it after a minute, and start `load.sh sin.aimodel`: the second waits until the first compile ends.
