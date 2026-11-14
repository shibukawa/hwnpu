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

On macOS, IOReport and the `proc_*` calls are C functions. They are called without cgo, the way golang.org/x/sys/unix does it: only `dlopen` and `dlsym` are imported from libSystem (`cgo_import_dynamic` with assembly trampolines), and the rest is looked up at run time, so a missing private library is an error in `Info.Errors`, not a binary that does not start. Reading IOReport does not wake the ANE (the power state stayed 0 over 40 reads while idle). It blocks for `Options.SampleInterval` (500 ms); a negative one leaves it out.

## What each OS gives

| | Linux (amdxdna, checked on a Ryzen AI 9 365) | Linux (intel_vpu, not checked) | Windows (not checked) | macOS (checked on an M1, macOS 27.2.1) |
|---|---|---|---|---|
| Device | sysfs: PCI ids, `fw_version`, `vbnv` (RyzenAI-npu4), link, `power_state`; ioctl: AIE version, array shape (8 columns × 6 rows), firmware, clocks (MP-NPU 1267 MHz, H 1800 MHz), power mode, TOPS and task capacity | sysfs: `npu_busy_time_us`, `npu_current_frequency_mhz`, `npu_max_frequency_mhz`, `sched_mode` (dumped with all sysfs attributes) | PnP: ComputeAccelerator devices, driver version and date; D3DKMT: adapters including compute-only (MCDM), adapter type, PCI ids, nodes (engines) | I/O Registry: the ANE classes (`H11ANEIn` and others), all properties; typed: architecture (`h13g`), cores (16), version, `ane,t8020`, whether the firmware is loaded, power state (0 idle, 1 while running); the loaded ANE kext and its version, AppleNeuralEngine.framework's version.plist; sysctl: model, CPU |
| Processes using it | ioctl `GET_ARRAY(HW_CONTEXT_ALL)`: per hardware context PID, columns, commands submitted/completed, state (no root needed); `/proc/*/fd`: holders of `/dev/accel/*` | `/proc/*/fd` and fdinfo (`drm-*` keys if the driver has them) | performance counters with `pid_N` instances (`GPU Engine`, `GPU Process Memory`, and NPU sets on builds that have them) | user clients under the ANE (`IOUserClientCreator`: "pid N, name"); per process (`proc_pid_rusage`): the ANE memory, which also finds the processes that go through `aned` (`mediaanalysisd`), path, the number of clients; the system's ANE processes (`aned`, `aneuserd`, `ANECompilerService`) |
| How busy | gpu_metrics: `average_ipu_activity` (%), command counts over time | `npu_busy_time_us` over time | `Utilization Percentage`, `Running Time` per engine and process | IOReport (no root): the ANE's interrupts (about 3 per program call), the power state |
| Power | gpu_metrics `average_ipu_power` (mW), socket power; amdxdna's sensor query returns ENODEV on this BIOS | — | D3DKMT adapter perf data, if the driver fills it | IOReport (no root): `PMP`/`Energy Counters`/`ANE` (mJ) → W (2.0 W running a 0.23 GiB lut4 program, 6.9 W under `hwnpu-image`); every other power domain beside it (CPU, GPU, DRAM, display…: `soc_power_w`); thermal pressure and Low Power Mode (notify); powermetrics `ane_power` as root |
| Memory | `/proc/PID/maps` entries of `/dev/accel/accel0`: the NPU buffers, which RSS does not show (LFM3 q4: 1635 MiB = the 1.64 GiB measured before); gpu_metrics NPU and DRAM reads/writes (MB/s) | fdinfo `drm-total-*`, `drm-resident-*` | `GPU Process Memory`, `GPU Adapter Memory` counters; segment sizes | `ri_neural_footprint` per process, now and the most ever (234 MiB for the 0.23 GiB program; not in Activity Monitor's memory); IOReport: the ANE's DRAM reads and writes (`AMC Stats`, 44 GB/s for that program = its weights × 175 calls/s), and every other client's (`dram_bytes_per_sec`) |
| Inside the NPU | AIE status registers per column (raw, 504 bytes × 8), firmware telemetry (raw), last async error | — | — | — |
| Caches | — | — | — | `com.apple.e5rt.e5bundlecache` under `~/Library/Caches` and `DARWIN_USER_CACHE_DIR`: size, files, OS builds |

What was not available: amdxdna in Linux 7.0 adds no `drm-*` keys to fdinfo, and the context fields `gops`, `fps` and `heap_usage` stay 0 with this firmware (1.1.2.64). On Windows, Task Manager has had NPU columns per process since builds 26100.8524 (24H2) and 26200.8524 (25H2). Which counter sets it reads is not documented, so every counter set whose name contains GPU, NPU or Neural is dumped. Run `npuinfo` there to see which ones exist.

## Next

- Decode the raw parts: AIE status per tile (core state, PC, locks, DMA), telemetry
- Run on Windows (Intel and AMD), then fill the "not checked" columns; check the IOReport channel names on M2 and later (`Energy Model`/`ANE…` is read too)
- A common API across OSes, then a browser viewer
