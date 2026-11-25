# M1 and M2 (h13g, h14g) Neural Engine compilers place a program with `sin` on the GPU, where M3 runs it on the Neural Engine; coreai-build exits 0

Filed as FB25724316 (2026-11-25), for sin and cos. `cos` measured the same way afterwards: not on the Neural Engine for h13g and h14g (offline, and on the M2's device), on it for h15g.

- **Area**: Core AI / Neural Engine compiler, Xcode / coreai-build
- **Type**: Incorrect/Unexpected behavior

## Description

A program that contains `sin` does not run on the Neural Engine of an M1 or an M2: Core AI places the whole program on the GPU. The same program runs on the Neural Engine of an M3. The offline compiler agrees with each device: `coreai-build` for h13g and h14g writes a `.aimodelc` without `mps.fullyPlacedOnANE` and exits with status 0. For h15g it places the program on the Neural Engine.

In a larger program the `sin` splits the Neural Engine regions. An audio codec decoder stage (Snake activations, MOSS-SoundEffect's DAC) compiles into 8 or 9 Neural Engine regions for h13g, h14g and h14c, with `sin` off the Neural Engine between them, and into one region for h15g and later.

The workaround in hwnpu is to compute sin by a range reduction and a degree-9 polynomial on M1 and M2.

On the M1's device compiler we also saw a clip fused into an elementwise op ("Unsupported NE layer NEFUSED_EW: EW: MAX") and the subtraction of two constant-scaled inputs ("NEFUSED_EW: KernelPreScales") rejected in that decoder stage, and the stage failed with "unexpected failure" at 4998 frames a call (4500 compiled). Small programs of those patterns (attached) compile onto the Neural Engine offline for h13g and on an M2's device, so they are not shown by the attachments yet.

## Steps to reproduce

1. `xcrun coreai-build compile sin.aimodel --platform macOS --architecture h14g --preferred-compute neural-engine --output out/` (attached `sin.aimodel`: `sin` of a (1, 256, 1, 4096) fp16 input, 20 KB).
2. Look for `mps.fullyPlacedOnANE` in `out/.../manifest.plist`, and at the exit status.
3. Repeat for h13g and h15g.
4. On an M2, load `sin.aimodel` with `AIModel(contentsOf:)` and the Neural Engine preferred, and check where it runs.
5. The same with `snake.aimodel` (a Snake block: 1×1 convolution, `sin`, square, 1×1 convolution) and `square.aimodel` (the same with `y*y` in place of `sin`).

## Expected results

`sin` runs on the Neural Engine as it does on h15g, or only the `sin` leaves the Neural Engine and the rest of the program stays on it. Also, `coreai-build` should not report success when the program is not on the Neural Engine.

## Actual results (2026-11-25)

| program | h13g offline | h14g offline | h15g offline | M2 device |
|---|---|---|---|---|
| `sin.aimodel` | not on ANE | not on ANE | ANE | GPU |
| `snake.aimodel` | not on ANE | not on ANE | ANE | GPU |
| `square.aimodel` | ANE | ANE | ANE | ANE |

`coreai-build` exited 0 for every one.

## Environment

MacBook Air M2 16 GB, macOS 27.2.1 (26B131), coreai-build 3605.6.3 (Metal Toolchain 27.2.47.0). Earlier on an M1 16 GB with the same macOS, and an M3 for comparison.

## Attachments

- `sin.aimodel`, `snake.aimodel`, `square.aimodel` (written by `tools/coreai/fbrepro`)
