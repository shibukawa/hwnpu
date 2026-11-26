# h14g .aimodelc of convolution stages runs without error on an M2 but writes wrong, run-to-run varying rows; the same program compiled on the device is right

- **Area**: Xcode / coreai-build, Core AI (ANELargeModelCompilerService) on M2
- **Type**: Incorrect results

## Description

The stages of an image VAE decoder (Anima's, 3×3 convolutions with an RMS norm over the channels and SiLU; one program per stage, 8 size functions each), compiled by `coreai-build` for h14g, are placed entirely on the Neural Engine offline (`mps.fullyPlacedOnANE`). On an M2 they load, finish and run with no error, and Core AI reports nothing. But some functions of some stages write wrong values in bands of rows that repeat at a fixed interval, like the seams of tiles. The picture gets a band of noise every 160 rows.

Each stage was run alone on the Neural Engine and compared with a float64 reference computed from that run's own input, so a wrong stage shows by itself (PSNR against the reference; a band is a row whose error is over 20 times the median row's):

| function | stage | `coreai-build` h14g, NAS copy (built on an M3) | `coreai-build` h14g, built on the M2 | compiled on the M2 |
|---|---|---|---|---|
| `s512x512` | 1, 2, 5 | 77–88 dB | 77–88 dB | 77–88 dB |
| `s512x512` | 3, 4 | 30.2, 26.6 dB, bands | 30.2, 26.6 dB, same rows | 81.1, 80.2 dB |
| `s512x512` | 6, 7, 8 | 31.4, 26.3, 35.1 dB, bands | 31.4, 26.3, 35.3 dB, same rows | 79.2, 74.3, 75.8 dB |
| `s512x512` | 9 (norm, SiLU, 3×3 conv to 3 channels) | 21.3 dB, rows 160–167, 320–327, 480–487 | 21.2 dB, same rows | 75.8 dB |
| `s1024x1024` | 1–8 | 75–89 dB | | 75–89 dB |
| `s1024x1024` | 9 | 16.2 dB, rows 160–167, 320–327, … every 160 | | 76.2 dB |
| `s832x1248` | 9 | bands | | right |
| `s880x1168` | 3, 4, 6–9 | bands | | right |

- **The wrong values change from run to run.** Three runs of the same `.aimodelc` with the same input, each from a binary signed with a new identifier (no Core AI or aned entry for it): stages 0–2 are identical byte for byte, stages 3 and later differ every time (39–64 dB between runs). It looks like reading memory that was not written, or written by another tile.
- **Where it was built does not matter.** The `.aimodelc` built on an M3 and the one built on the M2 with the same `coreai-build` are wrong in the same stages and the same rows. Their ANE regions (`*_ANE_region_0_0.mlir.bc`) are identical for stages 0 and 2 and differ for the others, including stage 1, which is right.
- **Only h14g.** The h13g `.aimodelc` of the same export draws the picture with no bands on an M1. The same programs compiled on the M2 itself (no Core AI or aned entry) are right in every stage and function.
- Nothing tells the app: the load succeeds, `prediction` succeeds, and there is no log line the app can see. A check that the program loads (what we did before shipping) passes.
- A smaller case of the same kind: FLUX.2 klein's VAE from `coreai-build` drew a band of noise every 112 rows at 512×512 on an M1 and an M2, while the 1024×1024 function of the same `.aimodel` was right.

## Steps to reproduce

1. `xcrun coreai-build compile vae_s9.aimodel --platform macOS --architecture h14g --preferred-compute neural-engine --output out/` (attached: `vae_s9.aimodel`, 288 KB: an RMS norm over 96 channels, SiLU as `x / (1 + exp(-x))`, a 3×3 convolution to 3 channels; 8 size functions).
2. On an M2, from a binary signed with an identifier that has never run this program (`codesign -f -s - -i NEW_ID`), load function `s512x512` from the `.aimodelc` and run it on the attached input (`vae_s8` output, fp16, 1×96×512×512).
3. Compare with the attached reference output, or with the same `.aimodel` compiled on the device (another new identifier, no `.aimodelc`).
4. Run step 2 again with a third new identifier, and compare the two runs.

## Expected results

The h14g program computes the same as the program compiled on the device (75.8 dB against the float64 reference, no bands), every run.

## Actual results

21.2 dB against the reference, with bands of 8 rows at rows 160, 320 and 480; the values in the bands change from run to run. No error anywhere.

## Environment

MacBook Air M2 16 GB (h14g), macOS 27.2.1 (26B131), Xcode 27, Metal Toolchain 27.2.47.0, coreai-build 3605.6.3. `.aimodelc` built on the M2 and on an M3 with the same toolchain.

## Attachments

- `vae_s9.aimodel` (288 KB) and its h14g `.aimodelc` (0.2 MB of compiled regions)
- the input (`vae_s8` output at 512×512) and the output of the device-compiled program, as raw fp16
- the outputs of two runs of the h14g `.aimodelc`
- a sysdiagnose and `log show --info --predicate 'process == "aned" OR process == "ANECompilerService" OR process == "ANELargeModelCompilerService"'` over the runs
