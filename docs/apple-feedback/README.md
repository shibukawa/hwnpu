# Drafts of reports to Apple (Feedback Assistant)

Drafts written 2026-11-25 for the problems of [known-issues.md](../known-issues.md) not reported yet, and additions to two reports already filed. Each file is one report: title, area, description, steps, expected and actual results, environment and what to attach. When one is filed, write its FB number here and in known-issues.md.

| File | Kind | known-issues.md | FB |
|---|---|---|---|
| [compile-disk-and-time.md](compile-disk-and-time.md) | new | 14 | FB25745042 |
| [purged-programs-silent-recompile.md](purged-programs-silent-recompile.md) | new | 8 | FB25724246 |
| [sin-off-ane-m1-m2.md](sin-off-ane-m1-m2.md) | new | 10 | FB25724316 |
| [offline-stricter-than-device.md](offline-stricter-than-device.md) | new | 11 | FB25724867 |
| [h15g-dma-cache-hint.md](h15g-dma-cache-hint.md) | new | 12 | FB25744357 |
| [h13g-attention-8192-keys.md](h13g-attention-8192-keys.md) | new | 13 | |
| [finisher-crash-throttle.md](finisher-crash-throttle.md) | new (written 2026-11-27) | 15, 17 | |
| [h13g-kernel-overrun.md](h13g-kernel-overrun.md) | new (written 2026-11-27; the cause of 13 too, so file it before or with h13g-attention-8192-keys.md) | 16, 17, 18, 13 | |
| [add-FB25624129-finish-slower.md](add-FB25624129-finish-slower.md) | addition | 1, 9 | FB25624129 |
| [add-FB25624259-no-cancel.md](add-FB25624259-no-cancel.md) | addition | 5 | FB25624259 |

Environment unless a file says otherwise: macOS 27.2.1 (26B131), Xcode 27 with Metal Toolchain 27.2.47.0, coreai-build 3605.6.3. Machines: MacBook Air M2 16 GB (h14g), M1 16 GB (h13g), M3 (h15g).

Attach to every report: a sysdiagnose taken right after reproducing, and `log show --info --predicate 'process == "aned" OR process == "ANECompilerService" OR process == "ANELargeModelCompilerService"'` over the minutes of the reproduction. Programs (`.aimodel`) that reproduce are made by hwnpu from public checkpoints; attach the smallest one that shows the problem, or offer it on request when too large to attach.
