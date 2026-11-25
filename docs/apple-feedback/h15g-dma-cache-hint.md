# h15g .aimodelc from coreai-build fails the device's validation ("cache_hint != ...drop_v8") and the whole program falls back to an on-device compile

- **Area**: Xcode / coreai-build, Core AI (ANELargeModelCompilerService)
- **Type**: Incorrect/Unexpected behavior

## Description

`.aimodelc` files that `coreai-build` makes for h15g, for image-model blocks with several size functions (FLUX.2 klein's DiT, 8 functions), fail to finish on an M3: Core AI reports `failedToSpecialize` and compiles the whole program on the device (3 to 19 minutes a program), because one function failed. Only the 512×512 function failed. ANELargeModelCompilerService logs:

```
tile_dma_src.dma_config2.cache_hint != ane_tile_dma_src_dma_config2_cache_hint_drop_v8
Instruction corresponding to oplayer ... failed validation test
Invalid TD exists
```

The shape that triggers it: attention with a block of 512 query rows over 1536 keys (1536 rows split into blocks of 1024, leaving 512). It reproduces in an attention-only program (one head, keys not split). 768-row blocks, 1536 rows at once, 512 rows over 1024 keys and 512 rows over 3072 keys pass; 1280 and 512 rows over 1792 keys fail. Which shapes fail depends on the offline compiler's allocation.

`.aimodelc` files made on an M1 and on the M3 are identical but for dates and statistics files, and both fail. `--min-deployment-version 27.2` and `--expect-frequent-reshapes` do not change it.

## Steps to reproduce

1. `xcrun coreai-build compile attn-1536-c1024.aimodel --platform macOS --architecture h15g --preferred-compute neural-engine --output out/`. The attached program is one head's attention, 1536 query rows over 1536 keys, head 128, with the queries in chunks of 1024, so a chunk of 512 is left. It is 64 KB. Offline, it is all on the Neural Engine for h13g, h14g and h15g.
2. On an M3, load `out/attn-1536-c1024.h15g.aimodelc` with `AIModel(contentsOf:)` while `log stream --predicate 'process == "ANELargeModelCompilerService"'` runs.
3. The same with `attn-1536-c768.aimodel` (chunks of 768).

## Expected results

A `.aimodelc` that coreai-build makes for h15g, with `mps.fullyPlacedOnANE`, finishes on an M3.

## Actual results

`failedToSpecialize`; the log lines above. With 768-row blocks (the workaround in hwnpu), the same model's programs finish from prebuilt on the M3.

## Environment

M3 (h15g), macOS 27.2.1 (26B131), coreai-build 3605.6.3 (Metal Toolchain 27.2.47.0).

## Attachments

- `attn-1536-c1024.aimodel` and its h15g `.aimodelc`
- `attn-1536-c768.aimodel` and its h15g `.aimodelc` (expected to pass)
- (to confirm on an M3 before filing: these small programs were not loaded on an M3 yet)
