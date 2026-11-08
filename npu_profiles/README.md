# NPU profiles

Reports written by `cmd/npuprobe` and `cmd/powerbench` on real machines, named `<tool>-<chip>[-<what>]`. [PROBE.md](../PROBE.md) ([日本語](../PROBE.ja.md)) explains how to run them and reads the results.

| chip | reports |
|---|---|
| Apple M1 (MacBook Pro 13", 16 GB, macOS 27.2) | `npuprobe-m1` (bench, limit) |
| Apple M2 (MacBook Air, 16 GB, macOS 27.2) | `npuprobe-m2.md` (bench, limit), `npuprobe-m2-split-96`, `-100`, `-104`, `-128` (the total of parts run in turn) |
| Apple M3 (MacBook Air, 16 GB, macOS 27.0.1; `-macos27.2` on 27.2) | `npuprobe-m3-macos27.2` (bench), `npuprobe-m3-split`, `-functions`, `-kv`, `-kv-mha`, `-kv-int8`, `-kv-limit`, `-kv-chunk`, `-kv-chunk-long`, `-attn`, `-seam`; `powerbench-m3`, `-cpu64`, `-rate100` |

The `.json` next to a `.md` is the same run in machine readable form. A new machine's report goes here as `npuprobe-<chip>.md`, with a section in PROBE.md.
