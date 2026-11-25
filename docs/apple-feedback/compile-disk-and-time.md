# Neural Engine compiler takes up to 15 minutes and about 50 GB of disk per program for a 1024×1024 VAE decoder stage

- **Area**: Core AI / Neural Engine (ANECompilerService)
- **Type**: Performance / Resource usage

## Description

Loading a Core AI program that is one stage of an image VAE decoder at 1024×1024 (convolutions and upsampling, fp16 activations, int8 weights) makes `ANECompilerService` compile it for 1 to 15 minutes, single-threaded, and write up to about 50 GB of temporary files while it does. The ten stages of the decoder take about 70 minutes in total on an M2. The disk use goes back to normal when each compile ends.

With less disk free than that, the compile fails with "Compiler internal error: Error: Failed to allocate memory for file backing" and Core AI runs the program on the GPU instead. Before it fails, the disk use pushes free space below the level where `deleted` starts purging, and the purge removes other compiled Neural Engine programs (reported separately: *Compiled Neural Engine programs are purged under storage pressure and Core AI silently recompiles*).

The same decoder of another model (FLUX.2 klein's, smaller channel counts) compiles its ten stages in 104 s at the same size.

## Steps to reproduce

1. On an M2 with 16 GB of memory and 70 GB or more of free disk, with no Core AI or aned cache for the program (a binary with a new code-signing identifier), load the attached `vae_s6.aimodel` with its `s1024x1024` function on the Neural Engine (`AIModel(contentsOf:)` with the Neural Engine preferred). The program is 832 KB, and it holds 8 functions, one for each picture size. Core AI compiles all of them on the first load.
2. Sample free disk space every few seconds (`df`) while it loads.

## Expected results

The program compiles in a few minutes with temporary disk use small compared with the program (832 KB, eight size functions).

## Actual results

`vae_s6.aimodel#s1024x1024` alone, as in the steps (2026-11-25, M2 16 GB, `tools/coreai/fbrepro/load.sh`): the load took **871 s**, and free disk went from **99.1 GB to 48.2 GB** (51 GB) during the compile and came back to 103 GB after it. It ran on the Neural Engine afterwards (162 ms a call).

The whole decoder in the app, one stage at a time, nothing else using the Neural Engine:

| stage | compile |
|---|---|
| vae_s0 | 2 min 10 s |
| vae_s2 | 5 min 00 s |
| vae_s3 | 3 min 51 s |
| vae_s4 | 11 min 03 s |
| vae_s6 | 14 min 21 s |
| vae_s7 | 14 min 32 s |
| vae_s8 | 14 min 59 s |
| vae_s9 | 2 min 16 s |

Free disk went from 67 GB to 10 GB during one stage and came back after it. A run that started with 28 GB free ended at 2.3 GB free in `vae_s2` with:

```
Error Domain=com.apple.appleneuralengine.compiler Code=1 "_ANECompiler : ANECCompile() FAILED"
... "Compiler internal error: Error: Failed to allocate memory for file backing"
```

and the program was placed on the GPU. The compile's input in the temporary folder (`.../com.apple.MetalPerformanceShadersGraph/mpsgraph-PID-.../*_ANE_region_0_0.bc.mlir`) is a few hundred MB; the rest of the disk use is the compiler's.

## Environment

MacBook Air M2 16 GB, macOS 27.2.1 (26B131).

## Attachments

- `vae_s6.aimodel` (832 KB, the stage with the longest compile)
- the `df` samples and `log show --info` of ANECompilerService over one stage
