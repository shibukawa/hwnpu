# Neural Engine compiler takes 15 minutes and about 50 GB of disk for an 832 KB VAE stage that uses the `silu` op; offline it fails

Filed as FB25745042 (2026-11-27).

- **Area**: Core AI / Neural Engine (ANECompilerService), Xcode / coreai-build
- **Type**: Performance / Resource usage

## Description

A Core AI program that is one stage of an image VAE decoder (Anima's `vae_s6`: convolutions, group norm, `silu`, upsampling; 832 KB of weights; 8 functions, one per picture size) has two problems when its activations use the MIL `silu` op:

1. **On the device**, `ANECompilerService` compiles it for 880 seconds on an M2, single-threaded, and writes about 51 GB of temporary files while it does. The disk use goes back to normal when the compile ends. With less disk free than that, the compile fails with "Compiler internal error: Error: Failed to allocate memory for file backing" and Core AI places the program on the GPU. Before it fails, the disk use pushes free space below the level where `deleted` starts purging, and other compiled Neural Engine programs are purged (FB25724246).
2. **Offline**, `coreai-build` for h14g and h15g fails its Neural Engine compile for every function ("Procedure compilation phase failed", `ANECCompileOffline() failed ... CompilationFailure`). It writes a GPU-only `.aimodelc` and exits 0 (as in FB25724867).

The only change between the two attached programs is how `silu(x)` is written: the second uses `x / (1 + exp(-x))`. That one compiles on the device in 527 seconds without using the disk, and compiles onto the Neural Engine offline. Across Anima's ten VAE stages, the native `silu` made coreai-build fail 8 of 10 stages for h13g, h14g and h15g, and the exp form fixed all of them. On an M3 the device compile of the ten stages at 512×512 went from 3552 s to 482 s.

So the `silu` op in this program, not the program's size, makes the compiler take minutes and tens of GB, and makes the offline compiler give up.

## Steps to reproduce

1. On an M2 with 16 GB of memory and 70 GB or more of free disk, with no Core AI or aned cache for the program (a binary with a new code-signing identifier), load `vae_s6-v5.aimodel` with its `s1024x1024` function on the Neural Engine (`AIModel(contentsOf:)` with the Neural Engine preferred). Core AI compiles all 8 functions on the first load.
2. Sample free disk space every few seconds (`df`) while it loads.
3. Repeat with `vae_s6-v6.aimodel` (the same stage, silu in the exp form).
4. `xcrun coreai-build compile vae_s6-v5.aimodel --platform macOS --architecture h15g --preferred-compute neural-engine --output out/`, and the same for v6. Check `manifest.plist` for `mps.fullyPlacedOnANE`.

## Expected results

The program with `silu` compiles about as fast as the one with the exp form, its temporary disk use is small compared with the program, and coreai-build places it on the Neural Engine.

## Actual results (2026-11-27, MacBook Air M2 16 GB)

| program | device compile | free disk during it | coreai-build h14g / h15g |
|---|---|---|---|
| `vae_s6-v5.aimodel` (`silu` op) | 880 s | 139.9 GB → 89.0 GB (51 GB) | GPU only, exit 0 |
| `vae_s6-v6.aimodel` (exp form) | 527 s | unchanged | Neural Engine |

Both ran on the Neural Engine once compiled (163 ms and 145 ms a call). The v5 result was the same on 2026-11-25 (871 s, 99.1 GB → 48.2 GB).

On 2026-11-25 the app compiled the whole decoder (native `silu`), one stage at a time: 2 to 15 minutes a stage, about 70 minutes in all. A run that started with 28 GB free ended at 2.3 GB free in another stage with:

```
Error Domain=com.apple.appleneuralengine.compiler Code=1 "_ANECompiler : ANECCompile() FAILED"
... "Compiler internal error: Error: Failed to allocate memory for file backing"
```

and the program was placed on the GPU.

The exp form still takes 527 s for an 832 KB program, because Core AI compiles all 8 size functions at the first load.

## Environment

MacBook Air M2 16 GB, macOS 27.2.1 (26B131), coreai-build 3605.6.3 (Metal Toolchain 27.2.47.0). The M3 numbers are from the same macOS.

## Attachments

- `vae_s6-v5.aimodel` (`silu` op, 832 KB) and `vae_s6-v6.aimodel` (exp form, 788 KB)
- `load-v5.txt`, `load-v6.txt`: the device loads (time, placement, free disk)
- `df-samples-v5.txt`, `df-samples-v6.txt`: free disk every 5 s during each load
- `system-log-v5.txt`, `system-log-v6.txt`: aned's and ANECompilerService's log over each compile
- `offline-h14g-h15g.txt`: coreai-build's results for both programs
- `README.txt`: what each file is
