# npuinfo

[日本語](README.ja.md)

Reads what the operating system tells about NPUs: the devices, which processes use them, how busy they are, and what the drivers and firmware report. Activity Monitor on macOS and the usual Linux tools show none of this; Windows' Task Manager shows some.

For now it collects what each OS and driver gives, as it is, in OS-specific structs (`Info.Linux`, `Info.Windows`, `Info.Darwin`). A common API comes later, once it is known what each OS can give. It imports only the standard library and nothing else from hwnpu, so that it can move to its own module and other modules (gopsutil, for example) can use it without new dependencies. The types build on every OS, so a viewer can decode JSON from another machine.

```bash
go run ./npuinfo/cmd/npuinfo              # once, indented JSON
go run ./npuinfo/cmd/npuinfo -n 10 -i 1s  # ten times, one JSON per line
go run ./npuinfo/cmd/npuinfo -wake        # also query a sleeping NPU (Linux; wakes it)
```

```go
info, err := npuinfo.GetWithContext(ctx, npuinfo.Options{})
```

Reading must not change what it reads. amdxdna's queries power up a sleeping NPU and keep it up for the driver's autosuspend delay (5 s), so a monitor that polls every second would keep it awake. They run only while the NPU is already active (`power/runtime_status`), unless `Options.Wake` is set. Opening `/dev/accel/accel0` and reading sysfs do not wake it.

## What each OS gives

| | Linux (amdxdna, checked on a Ryzen AI 9 365) | Linux (intel_vpu, not checked) | Windows (not checked) | macOS (not checked) |
|---|---|---|---|---|
| Device | sysfs: PCI ids, `fw_version`, `vbnv` (RyzenAI-npu4), link, `power_state`; ioctl: AIE version, array shape (8 columns × 6 rows), firmware, clocks (MP-NPU 1267 MHz, H 1800 MHz), power mode, TOPS and task capacity | sysfs: `npu_busy_time_us`, `npu_current_frequency_mhz`, `npu_max_frequency_mhz`, `sched_mode` (dumped with all sysfs attributes) | PnP: ComputeAccelerator devices, driver version and date; D3DKMT: adapters including compute-only (MCDM), adapter type, PCI ids, nodes (engines) | I/O Registry: the ANE classes (`H11ANEIn` and others), all properties; sysctl: model, CPU |
| Processes using it | ioctl `GET_ARRAY(HW_CONTEXT_ALL)`: per hardware context PID, columns, commands submitted/completed, state (no root needed); `/proc/*/fd`: holders of `/dev/accel/*` | `/proc/*/fd` and fdinfo (`drm-*` keys if the driver has them) | performance counters with `pid_N` instances (`GPU Engine`, `GPU Process Memory`, and NPU sets on builds that have them) | user clients under the ANE (`IOUserClientCreator`: "pid N, name") |
| How busy | gpu_metrics: `average_ipu_activity` (%), command counts over time | `npu_busy_time_us` over time | `Utilization Percentage`, `Running Time` per engine and process | powermetrics (root only); IOReport (private, no root) not done yet |
| Power | gpu_metrics `average_ipu_power` (mW), socket power; amdxdna's sensor query returns ENODEV on this BIOS | — | D3DKMT adapter perf data, if the driver fills it | powermetrics `ane_power` (root only) |
| Memory | `/proc/PID/maps` entries of `/dev/accel/accel0`: the NPU buffers, which RSS does not show (LFM3 q4: 1635 MiB = the 1.64 GiB measured before); gpu_metrics NPU and DRAM reads/writes (MB/s) | fdinfo `drm-total-*`, `drm-resident-*` | `GPU Process Memory`, `GPU Adapter Memory` counters; segment sizes | — |
| Inside the NPU | AIE status registers per column (raw, 504 bytes × 8), firmware telemetry (raw), last async error | — | — | — |
| Caches | — | — | — | `com.apple.e5rt.e5bundlecache` under `~/Library/Caches` and `DARWIN_USER_CACHE_DIR`: size, files, OS builds |

What was not available: amdxdna in Linux 7.0 adds no `drm-*` keys to fdinfo, and the context fields `gops`, `fps` and `heap_usage` stay 0 with this firmware (1.1.2.64). On Windows, Task Manager has had NPU columns per process since builds 26100.8524 (24H2) and 26200.8524 (25H2). Which counter sets it reads is not documented, so every counter set whose name contains GPU, NPU or Neural is dumped. Run `npuinfo` there to see which ones exist.

## Next

- Decode the raw parts: AIE status per tile (core state, PC, locks, DMA), telemetry
- Run on Windows (Intel and AMD) and on a Mac, then fill the "not checked" columns
- IOReport on macOS for the ANE's power without root (needs cgo or purego)
- A common API across OSes, then a browser viewer
