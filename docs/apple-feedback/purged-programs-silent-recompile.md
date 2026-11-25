# Compiled Neural Engine programs are purged under storage pressure while Core AI's cache entry stays, and Core AI silently recompiles

- **Area**: Core AI / Neural Engine (aned), Storage (CacheDelete)
- **Type**: Incorrect/Unexpected behavior

## Description

When free disk falls below about 15–22 GB, `deleted` purges purgeable files in chunks of about 2 GB. Among them are aned's compiled Neural Engine programs (`/Library/Caches/com.apple.aned`). Core AI's own cache entry for the same program (`~/Library/Caches/coreai-cache/<OS build>/<executable>/<hash>`) is left in place.

On the next load, Core AI finds its entry and asks aned for the program; aned logs `Model load failed. ... isPreCompiledModel=1` and Core AI compiles the program from its source on the device, without any error or signal to the app. For a program the app had loaded from a `coreai-build` `.aimodelc` (finished by ANELargeModelCompilerService in under a minute), this means a full compile instead: 22 minutes instead of 45 seconds for one image-model block on an M2.

The app cannot tell this happened, and cannot avoid it: whether aned still holds the program is not visible (the cache is protected by SIP), and Core AI's entry says the program is ready. hwnpu offers the prebuilt `.aimodelc` only when Core AI has no entry, so after a purge it is never used again.

## Steps to reproduce

1. On an M2, load a large program from a prebuilt `.aimodelc` (`AIModel(contentsOf:)` on the `.aimodelc`) so that Core AI's entry and aned's program are made. It finishes in under a minute (ANELargeModelCompilerService).
2. Fill the disk until free space is about 11–12 GB and keep it there for a few minutes (e.g. a large file copy, or another Neural Engine compile; see the separate report on compile disk use). `log show --predicate 'process == "deleted"'` shows `Purge: goal ... freespace ...` and `com.apple.deleted_helper ... purged 2055802880 bytes`.
3. Free the disk again, and load the same program (from the `.aimodel` or the `.aimodelc`).

## Expected results

Either aned's program and Core AI's entry are purged together (so the next load sees no entry and can use the prebuilt program), or the load reports that the program has to be compiled again, or aned's compiled programs are not purgeable while Core AI's entry refers to them.

## Actual results

Log of one occurrence (2026-11-25, M2): purges of 2.4 GB and 1.9 GB at 14:20–14:22 while the disk was full, then on the next load:

```
14:34:39 aned: Model load failed. model=<private> : csIdentity=<private> : isPreCompiledModel=1 : lErr=(null)
```

followed by a full compile in ANECompilerService (no ANELargeModelCompilerService), 21–23 minutes per program, where the first load from the prebuilt had taken 41–61 s per program. Over 09:30–12:10 the same day, `deleted` purged more than 120 GB in chunks of about 2 GB while the disk stayed near 11.8 GB free, and every later load of those programs compiled again. Removing Core AI's entries by hand made the next load use the prebuilt programs again (44–56 s each).

## Environment

MacBook Air M2 16 GB, macOS 27.2.1 (26B131). Seen also on an M1 (first step 5–15 minutes, with 12–22 GB free).

## Attachments

- `README.txt`: the two episodes of 2026-11-25 in order, and which file shows each step
- `system-log-1125-1305.txt`, `system-log-1408-1500.txt`: `log show --info` of deleted's purges and its "Purge: goal ... freespace ...", aned's `isPreCompiledModel=1` failures, its connections to ANELargeModelCompilerService, and ANECompilerService's "End of compilation of network"
- `app-log-1212-rerun-after-purge.txt`: the load after the purges (DiT blocks 22 min 53 s and 21 min 14 s)
- `app-log-1410-prebuilt-entries-removed.txt`: the load with Core AI's entries removed (DiT blocks 44–56 s from prebuilt)
- `app-log-1434-after-purge.txt`: the next load after the second purge
- `free-space-from-deleted.txt`, `df-samples-from-1434.txt`: free disk over time
- a sysdiagnose taken within a few days of 2026-11-25 (its log archive holds the full logs of both episodes)
