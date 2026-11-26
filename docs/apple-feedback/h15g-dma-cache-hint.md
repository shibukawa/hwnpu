# h15g .aimodelc from coreai-build fails the device's validation ("cache_hint != ...drop_v8") and the whole program falls back to an on-device compile

Filed as FB25744357 (2026-11-27).

- **Area**: Xcode / coreai-build, Core AI (ANELargeModelCompilerService)
- **Type**: Incorrect/Unexpected behavior

## Description

`.aimodelc` files that `coreai-build` makes for h15g, for image-model blocks with several size functions (FLUX.2 klein's DiT, 8 functions), fail to finish on an M3: Core AI reports `failedToSpecialize` and compiles the whole program on the device (3 to 19 minutes a program), because one function failed. Only the 512×512 function failed. ANELargeModelCompilerService logs:

```
tile_dma_src.dma_config2.cache_hint != ane_tile_dma_src_dma_config2_cache_hint_drop_v8
Instruction corresponding to oplayer ... failed validation test
Invalid TD exists
```

The shape that triggers it: attention with a block of 512 query rows over 1536 keys (1536 rows split into blocks of 1024, leaving 512), with the keys cut in parts (1024 + 512) under one softmax: the largest of the parts' maxima, the sum of their sums, and the probabilities as `exp(x - max) * (1 / sum)`. It reproduces in an attention-only program of one head (attached). The same program with the keys in one tensor passes, and so do 768-row blocks, 1536 rows at once, and 512 rows alone over 1536 or 1024 keys. In the FLUX.2 programs, 1280 and 512 rows over 1792 keys failed too. Which shapes fail depends on the offline compiler's allocation.

The failing `.aimodelc`'s LLIR (`binary_0.llir.bundle/.../*.mlir.bc`) has a planar-engine multiply that reads its source through the tile DMA (`l2_access_mode` 2), transposed and broadcast, with `cache_hint` 4. The programs that load have no such instruction. In FLUX.2 it was the multiply by the reciprocal of the sum. Dividing by the sum instead removed it, and those programs loaded.

`.aimodelc` files made on an M1 and on the M3 are identical but for dates and statistics files, and both fail. `--min-deployment-version 27.2` and `--expect-frequent-reshapes` do not change it.

## Steps to reproduce

1. `xcrun coreai-build compile attn-1536-c1024.aimodel --platform macOS --architecture h15g --preferred-compute neural-engine --output .`. The attached program is one head's attention, 1536 query rows over 1536 keys, head 128, fp16, inputs q, k and v. The keys are in two parts under one softmax, and the queries in chunks of 1024, so a chunk of 512 is left. Its manifest has `mps.fullyPlacedOnANE`. The attached `.aimodelc` files were made this way.
2. Build the attached loader: `swiftc -sdk "$(xcrun --sdk macosx --show-sdk-path)" -target arm64-apple-macos27.0 -O load.swift -o aiload`. It loads each path with `AIModel(contentsOf:)` on the Neural Engine.
3. On an M3: `./aiload attn-1536-c1024.h15g.aimodelc attn-1536-c768.h15g.aimodelc` while `log stream --predicate 'process == "ANELargeModelCompilerService"'` runs. `attn-1536-c768` is the same program with chunks of 768.

## Expected results

A `.aimodelc` that coreai-build makes for h15g, with `mps.fullyPlacedOnANE`, finishes on an M3.

## Actual results

On the M3, 2026-11-27:

```
attn-1536-c1024.h15g.aimodelc: FAILED after 0.15 s: failedToSpecialize
attn-1536-c768.h15g.aimodelc: loaded in 10.19 s
```

ANELargeModelCompilerService logs the three lines above for c1024. When the `.aimodel` is loaded with its `.aimodelc` beside it, Core AI tries the `.aimodelc`, fails the same way and compiles the program on the device (10.6 s here; 3 to 19 minutes for each FLUX.2 program). With 768-row blocks (the workaround in hwnpu), the same model's programs finish from prebuilt on the M3.

## Environment

M3 (h15g), macOS 27.2.1 (26B131), coreai-build 3605.6.3 (Metal Toolchain 27.2.47.0).

## Attachments

`h15g-dma-cache-hint.zip` (72 KB, made with `tools/coreai/fbrepro attn -parts`; see [repro.md](repro.md)):

- `attn-1536-c1024.aimodel` and `attn-1536-c1024.h15g.aimodelc` (fail)
- `attn-1536-c768.aimodel` and `attn-1536-c768.h15g.aimodelc` (pass)
- `load.swift` (the loader), `README.txt`
- `run.txt` and `ane.log` (`log show --info` of aned, ANECompilerService and ANELargeModelCompilerService over the run)
- and a sysdiagnose taken right after reproducing
