# coreai-build places on the GPU a program the device's compiler places entirely on the Neural Engine (a clip before sin), and exits 0

Filed as FB25724867 (2026-11-25).

- **Area**: Xcode / coreai-build (Metal Toolchain)
- **Type**: Incorrect/Unexpected behavior

## Description

A stage of an audio codec decoder (MOSS-SoundEffect's DAC, stage 4) uses Snake activations written as `sin(clip(alpha * x, -1500, 1500))`. When Core AI compiles it on an M3, the whole stage loads on the Neural Engine. `coreai-build` for h15g, h16g and h17g fails its Neural Engine compile ("Procedure compilation phase failed", `ANECCompileOffline() failed ... CompilationFailure`, "Falling back to full compile on GPU"), writes a GPU-only `.aimodelc` and exits with status 0. When this was first seen, the compiler's log named the layer: "Unsupported NE layer "NEFUSED_EW": EW: MAX" (the clip fused into an elementwise op).

The same stage passes offline when the only change is to drop the clip (`sin(alpha * x)`), or to compute sin by a polynomial. Earlier bisection of the stage: its first 46 operations pass offline, adding the 47th (convolution, reshape, per-channel multiply, clip) fails; those four alone pass. So it is the clip together with the earlier `sin` that fails offline. A small Snake block with the same clip before `sin` passes offline, so the attachment is the decoder stage itself.

Two problems:
1. The offline compiler is stricter than the device's for the same architecture.
2. It reports success when the result is not on the Neural Engine. Only the missing `mps.fullyPlacedOnANE` in `manifest.plist` tells.

## Steps to reproduce

1. `xcrun coreai-build compile dac4-native-clip.aimodel --platform macOS --architecture h15g --preferred-compute neural-engine --output out/` (attached, 432 KB, fp16 weights).
2. Check `out/.../manifest.plist` for `mps.fullyPlacedOnANE`, and the exit status.
3. The same with `dac4-native.aimodel` (the same stage without the clip).
4. On an M3, load `dac4-native-clip.aimodel` with `AIModel(contentsOf:)` and the Neural Engine preferred, and check where it runs.

## Expected results

coreai-build places the program on the Neural Engine as the device does, or exits with an error when it cannot.

## Actual results (2026-11-25, coreai-build 3605.6.3)

| program | h15g | h16g | h17g |
|---|---|---|---|
| `dac4-native-clip.aimodel` (clip before sin) | GPU, exit 0 | GPU, exit 0 | GPU, exit 0 |
| `dac4-native.aimodel` (no clip) | Neural Engine | Neural Engine | Neural Engine |

On an M3 (macOS 27.2.1), the device compiled the clipped stage onto the Neural Engine (seen in hwnpu's DAC "r2", before 2026-11-23).

## Environment

coreai-build 3605.6.3 (Metal Toolchain 27.2.47.0) on a MacBook Air M2, macOS 27.2.1 (26B131). Device behaviour on an M3, macOS 27.2.1.

## Attachments

- `dac4-native-clip.aimodel` (fails offline) and `dac4-native.aimodel` (passes)
- `coreai-build-h15g.log` (stdout and stderr of step 1)
- (before filing: load `dac4-native-clip.aimodel` on an M3 once more to confirm it is on the Neural Engine there)
