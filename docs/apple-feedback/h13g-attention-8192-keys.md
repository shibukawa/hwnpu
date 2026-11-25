# h13g .aimodelc with attention over 8192 or more keys computes noise on an M1, or crashes ANELargeModelCompilerService

- **Area**: Xcode / coreai-build, Core AI (ANELargeModelCompilerService) on M1
- **Type**: Incorrect results / Crash

## Description

A function of an image-model block with attention over 8704 keys (FLUX.2 klein's edit of a 1024×1024 picture from one 1024×1024 reference, `s1024x1024_ref1`; the keys in pieces whose softmax is combined by log-sum-exp) compiled by `coreai-build` for h13g is placed entirely on the Neural Engine offline, but on an M1:

- **single-stream block program**: it finishes and runs on the Neural Engine, but its output is noise (cosine −0.0005 with the CPU, 37,924 non-finite values). The picture is noise.
- **double-stream block program**: ANELargeModelCompilerService crashes near the end of that function's conversion: SIGSEGV in `ZinAneHloConverter::ParseInstructions` (2nd of 13 functions, task 37,838 of 38,060). Core AI then compiles the whole program on the device: 54 minutes for one 13-function program, during which the compiler wrote about 137 GB.

The same program compiled on the M1 itself is correct (cosine 0.999999), and so is the CPU (0.999996). Functions with fewer keys from the same build are correct:

| function | keys | compile | cosine with CPU |
|---|---|---|---|
| `s1024x1024` | 4608 | coreai-build h13g | 0.999999 |
| `s1024x1024_refh1` | 5632 | coreai-build h13g | 0.999999 |
| `s1024x1024_refh2` | 6656 | coreai-build h13g | 0.999999 |
| `s1024x1024_ref1` | 8704 | coreai-build h13g | −0.0005 |
| `s1024x1024_ref1` | 8704 | on the M1 | 0.999999 |

The h14g and h15g `.aimodelc` of the same program are correct on an M2 and an M3.

## Steps to reproduce

1. `xcrun coreai-build compile single-2blocks-ref1.aimodel --platform macOS --architecture h13g --preferred-compute neural-engine --output out/` (attached: two single-stream blocks, one function `s1024x1024_ref1`).
2. On an M1, with no Core AI entry for the program, run the `.aimodelc` with the attached input and compare with the CPU run of the same `.aimodel` (`AIModel` with CPU only).
3. For the crash, the same with `double-0.aimodel` (13 functions).

## Expected results

The h13g program computes the same as on the CPU, as the device-compiled program and the h14g/h15g programs do.

## Actual results

Noise (non-finite values) from the single-stream program; SIGSEGV of ANELargeModelCompilerService for the double-stream program, followed by a 54-minute on-device compile that wrote about 137 GB.

## Environment

M1 16 GB (h13g), macOS 27.2.1 (26B131), coreai-build 3605.6.3 (Metal Toolchain 27.2.47.0).

## Attachments

- `single-2blocks-ref1.aimodel`, its h13g `.aimodelc`, input and the CPU's output
- the crash report of ANELargeModelCompilerService
