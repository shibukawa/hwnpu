# coreai-build places on the GPU a program the device's compiler places entirely on the Neural Engine ("NEFUSED_EW: EW: MAX"), and exits 0

- **Area**: Xcode / coreai-build (Metal Toolchain)
- **Type**: Incorrect/Unexpected behavior

## Description

A stage of an audio codec decoder (MOSS-SoundEffect's DAC, "r2": Snake activations with `sin` and a clip) loads entirely on the Neural Engine of an M3 when Core AI compiles it on the device. `coreai-build` for h15g, h16g and h17g rejects it with "Unsupported NE layer "NEFUSED_EW": EW: MAX", writes a GPU-only `.aimodelc` and exits with status 0.

The first 46 operations of the stage pass offline; adding the 47th (convolution → reshape → per-channel multiply → clip) fails. Those four operations alone pass, and the stage with `sin` replaced by `mul(y, y)` passes, so it is the clip together with the earlier `sin` that fails offline.

Two problems: the offline compiler is stricter than the device's for the same architecture, and it reports success when the result is not on the Neural Engine (only `manifest.plist`'s missing `mps.fullyPlacedOnANE` tells).

## Steps to reproduce

1. `xcrun coreai-build compile dac-r2-stage4.aimodel --platform macOS --architecture h15g --preferred-compute neural-engine --output out/`
2. Check `out/.../manifest.plist` for `mps.fullyPlacedOnANE`, and the exit status.
3. Load the same `dac-r2-stage4.aimodel` on an M3 with the Neural Engine preferred and check where it runs.

## Expected results

coreai-build places the program on the Neural Engine as the device does, or exits with an error when it cannot.

## Actual results

coreai-build: exit 0, no `mps.fullyPlacedOnANE`, os_log "Unsupported NE layer "NEFUSED_EW": EW: MAX". On the M3: all on the Neural Engine.

## Environment

M3 (h15g), macOS 27.2.1 (26B131), coreai-build 3605.6.3 (Metal Toolchain 27.2.47.0).

## Attachments

(Before filing: a small Snake block with a clip, `fbrepro snake -clip`, passes offline for h15g, so the attachment has to be the DAC stage itself. See repro.md.)


- `dac-r2-stage4.aimodel` and its first 47 operations (`stage4-cut47.aimodel`)
