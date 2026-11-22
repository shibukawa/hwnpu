# AMD XDNA

[日本語](amd.ja.md)

AMD's Ryzen AI NPU: the driver path (`internal/xdna`, `internal/backend/amd`, Linux), the XRT path (`internal/backend/amdxrt`, Windows), program formats (`internal/xclbin`, `internal/aieelf`, `internal/ert`), IRON as the build-time compiler, shared kernels, pinned memory and power. Model work lives elsewhere: [amd-llm](amd-llm.md) (language models), [image-amd](image-amd.md) (image models), [moe](moe.md) (Qwen3.6, gpt-oss). The longer design notes stay in [../amd-npu-models.md](../amd-npu-models.md); data types and raw throughput in [../amd-xdna2-bench.md](../amd-xdna2-bench.md).

Measured machine (all numbers below unless noted): Ryzen AI 9 365 (Strix Point, PCI 0x17f0, `npu4` = NPU2, 8 columns, AIE 1.1, 4 core rows), Ubuntu 26.04, stock Linux 7.0 amdxdna (no DKMS), firmware 1.1.2.64 (`npu_7.sbin`). No XRT installed.

## Device and driver (Linux, amdxdna)

- Discovery: `/sys/class/accel/accel*/device/driver` → `amdxdna` (AMD) or `intel_vpu` (Intel) (`internal/accelfs`). Node `/dev/accel/accelN`, `O_RDWR|O_CLOEXEC`. Needs the `render` group; the error says how.
- ioctls: DRM `_IOC` encoding, `DRM_COMMAND_BASE + 0..10` (CREATE_HWCTX, DESTROY_HWCTX, CONFIG_HWCTX, CREATE_BO, GET_BO_INFO, SYNC_BO, EXEC_CMD, GET_INFO, SET_STATE, WAIT_CMD, GET_ARRAY). Structs copy `amdxdna_accel.h` (all `__u32`/`__u64`, no padding). Checked against `testdata/abi/xdna_abi.json` (gcc on the kernel UAPI and XRT headers: ioctl numbers, layouts, xclbin structs, ERT bits).
- Pointers inside ioctl structs are `unsafe.Pointer`, not integers (a goroutine stack move could hand the kernel a stale address).
- Retry only `EINTR`. amdxdna returns `EAGAIN` for "heap not ready", which would loop forever.
- `internal/xdna/xdnasim` is a Go fake driver that checks the raw ioctl structs with the same rules as Linux 7.0 (`xdnasim.NoSensors` = no power sensors).

Buffers:

| Kind | Use | Address the device sees |
|---|---|---|
| `BO_DEV_HEAP` | One per open device, before any `BO_DEV`. Multiple of 64 MiB (`AIE2_DEVM_SIZE`); npu1 fixed at 64 MiB, npu4+ up to 512 MiB. Mapped right after creation (else `BO_DEV` gives `EAGAIN`) | `xdna_addr` (from `0x4000000`) |
| `BO_DEV` | PDIs, control code, instruction streams (`BufferInstructions`, XRT's cacheable). Not mappable; CPU address = heap map + offset | `xdna_addr` |
| `BO_SHARE` | Inputs, outputs, weights | Linux 7.0 leaves `xdna_addr` invalid; under PASID the device uses the user address = `GET_BO_INFO.vaddr` after mmap. `xdna.BO.DeviceAddr()` picks `xdna_addr` if valid, else `vaddr` |
| `BO_CMD` | ERT packet (4096 B) | as BO_SHARE |

Submit and wait:

- `EXEC_CMD`: with `cmd_count = 1`, `cmd_handles` holds the handle value itself; `args` = handles of every BO the command touches.
- AIE2 does not implement `WAIT_CMD`. Wait with `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` on the hardware context's syncobj at the seq `EXEC_CMD` returned (`WAIT_FOR_SUBMIT`, absolute CLOCK_MONOTONIC). On `EINTR` (Go runtime signals) wait again with the same deadline. The thread sleeps, so the CPU is free.
- Result: the state field of the CMD BO's first word: `COMPLETED (4)`; `ERROR (5)`, `TIMEOUT (8)` … become `DriverError`.
- `CREATE_HWCTX` right after another process's context ended sometimes returns `EINVAL`: retried for up to 5 s.
- **The device heap (64 MiB) must sit at an address that is a multiple of its size**: otherwise the
  firmware refuses `MAP_HOST_BUFFER` (opcode 0x106, status 0x4000003) and `CREATE_HWCTX` fails with
  `EINVAL`. The kernel's `mmap` returns aligned addresses while the address space is empty, but not
  always for a second device in one process or after large mappings (3 models in one process failed on
  the 2nd or 3rd). `internal/xdna.MmapAligned` reserves size + alignment and maps the heap with
  `MAP_FIXED` at the aligned address; `internal/backend/amd.Open` uses it (12 of 12 opens pass, 5 of 12
  before). Taken to be the same cause as the rare failure in [../amd-xdna2-bench.md](../amd-xdna2-bench.md)
  (about 1 in 125 process starts); that case was not reproduced.

Cache coherency (x86):

- Sync is user-space `CLFLUSH` + `MFENCE` (`xdna.FlushCPU`, Go assembly), as XRT does. Linux 7.0's `SYNC_BO` flushes through the kernel mapping and does nothing for BOs whose pages are not pinned yet. Other architectures use `SYNC_BO`.
- When `SYNC_BO` is used it is always `TO_DEVICE`: in 7.0 `FROM_DEVICE` syncs a "debug BO" and returns `EINVAL` for ordinary BOs; clflush writes back and invalidates, so it serves both directions.
- New pages are zeroed through the CPU cache on first touch; those dirty lines, written back after the NPU wrote DRAM, erased the first command's output. `allocMapped` writes every page and flushes right after mapping; the device heap is flushed after mapping too.

## Program formats

### xclbin flow (`ERT_START_CU`)

- `internal/xclbin`: `IP_LAYOUT` (`IP_PS_KERNEL` entries: name, functional, kernel_id), `AIE_PARTITION` (columns, ops per cycle, PDIs; offsets relative to the section), `EMBEDDED_METADATA` (XML: argument order, offsets, sizes; `extended-data dpu_kernel_id` matched against the CDO groups' `dpu_kernel_ids`, as XRT's shim).
- PDI → `BO_DEV`; `CREATE_HWCTX{max_opc, num_tiles = ncol × core rows}`; `CONFIG_HWCTX(CONFIG_CU)` with `cu_bo` = the PDI's BO_DEV (cannot be reconfigured).
- ERT packet (`internal/ert`): header `state[3:0] … count[22:12] opcode[27:23] type[31:28]`, CU mask, then the register map from the metadata's offsets (64-bit values low word first; buffers = device addresses). IRON (mlir-aie v1.4.4) puts `bo0`–`bo4` in every signature; arguments not set are sent as 0.
- Loading an xclbin (context + PDI) 0.5 ms; passthrough 16 KiB median 0.075 ms.

### full ELF (`ERT_START_NPU_PREEMPT_ELF`, opcode 22)

The form every model uses on NPU2: the whole graph call in one hardware context and one command.

- `internal/aieelf` reads aiecc's full ELF: group → instance (`main:sequence`) and control code; `.dynsym` arguments and PDIs; `.rela.dyn` relocations; `.note.xrt.configuration` column count. Relocation schemes: 3 scalar32, 5 shim DMA 48-bit (+0x80000000 aperture), 8 address64. Scheme 4 (control packet), preemption and `.ctrldata` are errors. `Patch` computes from the unpatched words, so re-patching with new arguments is idempotent.
- `internal/backend/amd/elf.go`: `ProgramSource.ELF` (or an ELF in `Xclbin` / `Path`). PDIs in one `BO_DEV` shared by the program, control code a `BO_DEV` per run, rewritten and synced only when arguments change. Context with `CREATE_HWCTX` only (no CU). Firmware needs mailbox protocol major 7 (`AIE2_NPU_COMMAND`, `AIE2_PREEMPT`); Linux 7.0 builds `EXEC_NPU_TYPE_ELF` slots (`force_cmdlist`).
- Checked by recording XRT's ioctls under `LD_PRELOAD` (XRT unpacked without root) and comparing patched control code and packets word for word.
- IRON ReLU → AXPY 2048 elements: bit-identical to `graph.reference()` without IRON or XRT at run time; load 1.0 ms, a call 0.27 ms.

### Per-call values (scratchpad) and carried values

- Control code TXN (`internal/aieelf/txn.go`, major 0): `CREATE_SCRATCHPAD` (op 10) writes the `scratch-pad-ctrl` BO_DEV address (scheme 8); `UPDATE_REG` (op 12, `{slot, func, FuncArg, RegOff}`) adds `table[slot]` (INCR) or `× FuncArg` (MUL) to a register. Core parameters are zeroed by WRITE then INCR'd; shim BD lengths get MUL after the BD write (mlir-aie `AIEUtils.cpp`). `core` values are shifted left 2 bits (firmware drops the low bits).
- A scratchpad word is an affine form of the call's values (IRON `Word.form`: `clamp(((v·scale + bias) >> down)·mul + add, lo, hi) % mod + post`), written into the manifest by `tools/amd/iron_export.py` and computed in Go. `Run.Scratchpad()` exposes the buffer.
- Carried values (IRON `Carried`), called once per call from the host: before a call the current values go to stage 1 of the `carry` buffer; afterwards an affine next value (`steps + 1`) is computed on the host, a computed one (`node`) is read from stage 2. Walk (64-entry successor table) × 12 calls matched, 0.18 ms a call.
- Every per-call word must be affine in one per-call value: forms such as `ceil(rows / 8)` or `start + rows` are refused at export (`words without a form`). Use fixed chunks and one per-call `start`.
- Not supported: several commands in one ELF (`Plan.commands`); words depending on buffer addresses (`recording.addresses`) error at export.

### One xclbin per step (NPU1)

- NPU1 (Phoenix, Hawk Point) firmware has no full ELF (no AIE2_PREEMPT). IRON then makes each version "one xclbin per step" (`PerStep`): each operator is an xclbin kernel (opcode 3, instructions, length, buffers) submitted once. Configurations are shared: Gemma 4 E2B decode 1095 steps on 2 xclbins, a prompt chunk 1096 steps on 4.
- No scratchpad: per-call values become dispatch-time scalars and the instruction stream is rebuilt per call. mlir-aie generates C++ (`dispatch_gen.cpp`, Linux-only shared library) for that; the export turns it into an instruction program (JSON, `tools/amd/iron_stream.py`) that a Go interpreter (`internal/llm/iron/stream.go`) runs with C's conversions (sign extension, truncation, usual arithmetic conversions). Unknown lines are refused at export. The export compares every program with the generated library on sample scalars; `check/streams.json` (64 programs) also passes in Go (`TestExportStreams`).
- Placement: resident data (weights, KV) in arena banks; temporaries in a separate pool per version that counts from 0 again → argument `pool` (`pool_1`, …), not on top of the weights.
- Run (`internal/llm/iron/steps.go`): each xclbin loaded once per model; a run per step made at load; buffers bound as parts of a bank (`Run.SetBufferAt`: xdna adds to the address, XRT makes a sub-buffer). Commands in one context run in order, so steps are queued without waiting (a wait every 16, and before the xclbin changes). Window 1: 12 % slower; 64: no change.
- Instruction buffers: one static and one per-call buffer per (version, xclbin), each step a 64-byte-aligned slot (a buffer per step, 2191 of them, ran out of the 64 MiB heap). Per xclbin because XRT puts instruction arguments in a memory group per xclbin (65537, 131073, …) and copies a buffer bound to several groups whole. XRT cannot resubmit a run not waited for (`bad command state`): wait first.
- On NPU2 with `--per-step`: Gemma 4 E2B correct on xdna and XRT, decode 2.6 tok/s (full ELF 6.1), 14-token prompt 1.3 s. NPU1 export (`--device npu1 --cols 4`, 6 xclbins) loads and builds every step's stream on a simulated 4-column NPU (`TestExportLoads`). Towers (image, audio) are out of scope: IRON does not run them on NPU1.

## Windows: xrtshim (`internal/backend/amdxrt`, backend `xrt`)

- Windows' AMD NPU driver is usable only through XRT's C++ library (`xrt_coreutil.dll`). `native/xrtshim` is a small C++ library exposing a C ABI; Go loads it with purego (`internal/dl`) at run time. Still no cgo for users.
- API: devices (count, open, name, BDF, platform/electrical JSON, XRT version), buffers (host_only, cacheable, memory group, map, address, sync), hardware contexts (xclbin via `register_xclbin`, full ELF via `xrt::elf`), kernels (`xrt::kernel`, `xrt::ext::kernel`, `group_id`), runs (args, `get_ctrl_scratchpad_bo`, start syncs the scratchpad, wait, state). Errors come back as malloc'd strings that Go copies and frees (thread-local last-error does not work with goroutines moving threads).
- Selection: on Windows always; on Linux only with `HWNPU_BACKEND=xrt`. Registered before `amd` (xdna), so `hwnpu.Open(WithVendor(AMD))` picks it.
- `xrt_coreutil.dll` is outside the search path (`System32\AMD`, the driver store): it is loaded first by absolute path (or `HWNPU_XRT_CORE`) so xrtshim's imports bind to it. `HWNPU_XRTSHIM` names the shim. XRT headers 2.21.75 (Apache-2.0) are in `native/xrtshim/include`; `make-importlib.ps1` makes the import library from the DLL's exports.
- XRT 2.21 pitfalls: (1) waiting again on a command whose `run.wait(timeout)` timed out → `qds_device::wait() unexpected command state`: waits are unbounded. (2) An xclbin kernel's instruction argument lives in its own memory group (`group_id(1)`); a group-0 cacheable buffer fails making a "local copy". Instruction buffers are Go memory, copied into an XRT buffer of that group when bound. (3) A full ELF kernel keeps its control code in the 64 MiB heap while open (an audio tower, 39.8 MiB, then got `ENOSPC`): kernels are closed when no run uses them.
- Power: `power_consumption_watts` of the electrical JSON, read every 50 ms and integrated (`power.Integrator`). Linux 7.0's stock driver does not report it.
- Checked on Linux with XRT 2.21.75 unpacked without root: passthrough (xclbin) PASS; relu-axpy, bounded-relu (scratchpad), walk (carried) bit-identical; Gemma 4 E2B 6.1 tok/s (as xdna); multimodal hwnpu-chat image and audio answers right. Compiles with zig's x86_64-windows-gnu.

## IRON as the build-time compiler

- AMD publishes no graph compiler, and an xclbin per op costs 0.46 ms a switch (`tools/amd/ctxbench`: 3 contexts alternating 2.960 ms vs 1.571 ms alone; batching does not help) — 46 ms a token for 100 ops. So [amd/IRON](https://github.com/amd/IRON) (Apache-2.0) and mlir-aie compile at export time, each model call becomes one full ELF, and hwnpu runs it in Go. This is the one exception to "conversion in Go only"; users' machines need neither Python nor XRT.
- Export tools: `tools/amd/iron_export.py` (an IRON `CompiledGraph` → ELF + manifest: commands, argument order and sizes, buffer layout, inputs, outputs, weights, params, `carried`, `images` = initial bank images `arg_<kind>.bin`; `--check` writes random inputs and `graph.reference()` outputs, or `check/calls.json` for carried graphs). `iron_export_lm.py` (language models), `iron_graph.py` + `iron.Graph` (other graphs, one ELF may hold several weight banks `variants`). `hwnpu-dev xclbin` builds IRON designs without XRT's xclbinutil.
- Run-side tools: `tools/amd/ironcheck` (an export against its check), `irongraph` (`-runs N`, `-swap -variant`), `ironlm` (language models), `gemvbench`, `ctxbench`, `elfconfigs` (reconfigurations per ELF), `HWNPU_DEV=iron.timing=1` (per-command times), `--command-per-step` (one command per step, for timing only).
- Build environment: mlir-aie wheels need Python 3.12 (uv). IRON pins an mlir-aie nightly. Device by `aie_utils.set_current_device(from_name("npu2", n_cols=8))`. Big exports (Qwen3.6: 30 min, peak RSS 53.8 GB of 60): not beside other big jobs, under `~/models` (not `/tmp`, which is tmpfs), long ones with `systemd-run --user`.
- Export names: `~/models/NAME.amd-npu2-q4` (or `NAME.amd-npu2`); hwnpu-chat picks it for `-m NAME` on an AMD-only machine.

## Cost model (what bounds speed)

| Cost | Value | Source |
|---|---|---|
| DRAM → NPU weight reads (GEMV, bf16) | ~50 GB/s ceiling, ~60 % of what the CPU reads (55 GB/s 1 thread, 79 GB/s 8). DMA read 55–57, write 48 GB/s | `gemvbench`, [../amd-xdna2-bench.md](../amd-xdna2-bench.md) |
| A submit | ~0.06–0.07 ms fixed; a full-ELF op 0.13–0.15 ms (bounds tiny models' tokens) | |
| A command boundary | ~0.1 ms; a command chain (ERT_CMD_CHAIN) does not remove it; the scratchpad is read once a command | |
| Switching xclbin (context) | 0.46 ms | `ctxbench` |
| Reconfiguring the array inside a command (an operator change) | ≈ 40 µs + 2.3 µs/KB of configuration (control-code writes at ~430 MB/s); 70–280 µs measured for 6–136 KB configs | `iron_moeprobe.py --switch`, `elfconfigs` |
| Zero blocks in a configuration (.bss) | ~1.1 µs per KB | |

- Inside a full ELF each operator change emits `load_pdi` (TXN op 8) and then the array setup (16 KB program memory × cores used, data memory init, locks, DMA, switches) as control-code writes. The PDI itself is a 368 B shell. The same operator back to back is configured once.
- Dense-model decode ≈ weights / 50 GB/s + reconfigurations. LFM3-1.7B q4: 92.6 ms measured vs 33 + 65 = 98 ms estimated; Gemma 4 E2B q4: 137 vs 35 + 103 = 138 ms. So decode is bound by the count and size of operators, not bandwidth. What helps: fewer operators per layer, smaller configurations (QGEMV 1 core per column), or one resident array per step (gpt-oss's `GptOssStep`, [moe](moe.md)).
- `elfconfigs`' "reorder … removes at most" ignores dependencies; meaningful only for one layer per command. Skipping repeated program writes would save 0–20 % of program bytes, and the cost hardly depends on bytes.

## Shared kernels and operator choices

hwnpu swaps its own kernels into IRON's operators at export without changing IRON (`tools/amd/iron_q4.py`, `tools/amd/kernels/`):

- Quantized GEMV `QGEMV` (q4nx, q6, q8: `q4_mv.cc`, `q6_mv.cc`, `q8_mv.cc`), `QGEMVAdd` (bias and residual added as the output leaves, `kernels/mv_add.h`), `QGEMVGlu` (gate and up as one table and one operator), `QDequant` / `q4_dequant` for prompts and DiTs. Formats and per-model choices: [quantization](quantization.md).
- QGEMV is 1 core per column (`iron_q4.q_linear`, cores 0 = 1 up to 16384 out-features, 2 above; `HWNPU_Q_CORES`, `HWNPU_Q_CORES_UP_TO`): config 92 → 36 KB (q8, 2048 out), same speed, bit-identical. `QGEMVGlu` stays 2 cores (`HWNPU_GLU_CORES`; 1 core is slower with two accumulations). LFM3-1.7B decode 15.6 → 17.2–17.3 tok/s (with sandwich), Gemma 4 E2B 9.3 → 9.7, Qwen3.5-0.8B 18.5 → 18.7.
- Sandwich inside the next GEMV (`iron_q4.sandwich_operator`, `sandwich_linear`, `Q_SANDWICH`): each core computes y = r + rms(h)·w_post and n = rms(y)·w_pre, then n's GEMV (same sums and rounding as SandwichAdd). Used for LFM3 (`HWNPU_LFM3_SANDWICH_GEMV`); alone it changes nothing (15.5 tok/s).
- The GEMV takes only as many columns as divide its 64-row blocks: pad to whole blocks on 8 columns (`iron_gdn.COLUMN_ALIGN` 64). Quantized projections take multiples of 64 out by 512 in.
- Cores round to nearest even in hwnpu's kernels. AIE cores start with truncation; narrowing to bf16 / bfp16 then pulls sums of inputs with a mean downward (IRON's `DequantBFP` too: cosine 0.968 vs `QDequant` bf16 0.99994 on mean-1 inputs). Language-model prompts keep `DequantBFP` (bf16 there is 5–9 % slower and changes nothing).
- `iron_silu.SiLUF32` (`kernels/silu_f32.cc`): sigmoid in float32; IRON's bf16-table SiLU is exactly 0 from x = −8 and 1 % off over [−6, 2].
- `exp2_mid.h`: the softmax's exp on one bf16 limb, 102.7 → 83.2 ms on a 4608² × 24-head attention at the same error (default for image models except Anima). Attention sinks (`iron_sinks.py`) and windowed attention (`iron_window.py`) are in [moe](moe.md) / [amd-llm](amd-llm.md).

## Pinned memory

- NPU buffers (amdxdna `BO_SHARE`) have every page pinned and do not show in RSS: Gemma 4 E2B had RSS 0.43 GB while `Unevictable` grew 9.3 GB. Measure with `drm-total-memory` in `/proc/PID/fdinfo` of the accel fd, or the size of the process's `/dev/accel/accel0` mappings in `smaps` (neither mixes in other processes). Allocating pins and zeroes pages at 0.33 s/GiB; freeing 0.11 s/GiB.
- Ways to pin less (details in the model files): host tables for PLE and embeddings (`iron.HostTable`, [amd-llm](amd-llm.md)); towers loaded on first use (`growBanks`); image stages loaded only while they run, DiT groups in one swapped bank (`iron.GraphOptions.SwapVariants`), VAE stages sharing buffers (`GraphOptions.Buffers`) ([image-amd](image-amd.md)); 8-bit DiT weights. Pinning more does not speed up (35 GB pinned for MoE changed nothing).
- Idle release (`hwnpu-chat -idle`, default 5m; 0 = never): after no request for that long, `chat.Engine` copies the conversation to the host (`putAside`), `iron.Model.Release` frees all device buffers and instruction streams and closes the device; the next request `Reload`s the same export and continues from the copy or the prompt store. The model keeps its manifest, host tables and tower preprocessing, so `Images`, `Context` and media token counts still answer; runs while released fail with `errReleased`. E2B q4 1775 MiB → 0, reload 0.8 s; E2B with image and audio 2818 MiB → 0, reload 2.0 s. For a MoE with experts on the CPU, `-idle` keeps only the core ones wired. Only IRON: Neural Engine and Intel programs take tens of seconds to minutes to reload.
- Exports side by side share identical files and bank prefixes on disk (`iron.ShareExport`, [distribution](distribution.md)).

## Power

- `SET_STATE(SET_POWER_MODE)` is `DRM_ROOT_ONLY` and device-wide: hwnpu does not touch it. The power policy maps to the context's QoS priority (LOW 0x280, NORMAL 0x200, HIGH 0x180; 0 is HIGH in the driver).
- Meter: the NPU sensor (`GET_INFO(QUERY_SENSORS)`, mW, `Scope = npu`) when present — only the DKMS driver has it; stock Linux 7.0 does not. Otherwise amdgpu's hwmon `power1_input` (PPT, the whole APU package, µW, no root; `Scope = package`). Either gives `CapPowerMeter`. Both are instantaneous: a background reader samples every 50 ms from the first `PowerSample` and integrates by trapezoids (`power.Integrator`, shared with XRT's electrical JSON); it stops when the device closes. `Provider.Hwmon` swaps the search path in tests. RAPL is the other option for package power.
- Querying amdxdna (GET_INFO, GET_ARRAY, opening the device) wakes a sleeping NPU and keeps it up for the 5 s autosuspend delay; monitors query only when `power/runtime_status` is active ([memory-and-power](memory-and-power.md)).
- Numbers (package, idle 4.7 W after cooling): GEMV 12.3 W, 3–5 mJ per 32 MiB read (CPU only reading: 18–28 W, 7–12 mJ). Gemma 4 E2B generating: 11.8 W average, 1.93 J a token (idle included).

## Pitfalls (AMD-wide)

| Pitfall | What to do |
|---|---|
| A command over 2 s ends in `ERT_CMD_STATE_TIMEOUT` | Split at export (`SCRATCHPAD_WORDS`: 24 for dense models; Gemma 4's tail export 32) |
| A command's control code must fit the 64 MiB device heap with the others open (`CREATE_BO: invalid argument`, `ENOSPC`) | Split commands; on `ENOSPC` drop the decode / verify run and retry (as IRON); close kernels no run uses |
| More than 3 transfers queued on one shim channel (the task queue is 4) hang the command | Wait for the oldest before queuing more; queue at most 2 fills ahead when a later fill gates them |
| Two hangs in a row, or polling the firmware twice a second while a model creates contexts, left the SMU dead (`aie2_smu_init: Access power failed`, `failed to init smu`, mailbox timeout); `modprobe -r` hangs; only a reboot helps | Avoid hangs (above); do not poll `hwnpu status` during loads; read `drm-total-memory` from fdinfo instead |
| A graph's second run, or the next chunk's first command, hangs | An empty command in a hardware context of its own before each run (`iron.GraphOptions.Barrier`, `stepBarrier`, `chunkBarrier`, ~0.3–0.5 ms). An empty command in the same context does not help. Check graphs with `irongraph -runs 3`. Cause unknown |
| Rewriting `load_pdi` or program writes in an ELF to measure | Breaking the load sequence stopped the NPU and the SMU until reboot: do not |
| What a host step reads can be overwritten between commands (IRON's arena does not count a host read as a use) | Copy it into a state; check host-step exports at full depth |
| A DMA run is at most 1023 words; a DMA loop counts at most 64; a descriptor takes no stride back in a middle dimension | Lay tables out in the order operators read them |
| Program memory is 16 KB a core | One noinline function instead of template copies |
| `filter_even` / `filter_odd` on long vectors: ~290 K cycles a call | `interleave_unzip` on 32-element vectors |
| llvm-aie fails some small loops with an early return ("LoopStart/LoopBody: insufficient padding") | Unroll or remove the branch |
| bf16 `mmul` | Native as `mac_4x8_8x8` (256 products/instruction vs 16 for vector mac); AIE2P's native float matrix instruction is bfp16 `mac_8x8_8x8T` |
| IRON's compile cache keys on an operator's parameters, not its code | Give a changed operator a parameter of its form (`form`, `sink_form`) |
| The installed kernel headers differ from an mlir-aie checkout | Read the headers IRON compiles |
| An `array=True` parameter that differs per layer makes an array per layer (reloaded each command) | Keep sizes out of array params; read command times after export changes |
| IRON's reference copies its kernels (tanh-GELU table, no MEDIUM bfp16, no epilogue SiLU error: AIE2P `vtanh` returns x for \|x\| ≤ 0.5) | Check each operator against float32 with real activations, not only IRON's reference |
| IRON's MHA keeps scores in bf16: large scores with small differences lose them | Subtract the first key from every key, or check the largest score against float32 |
| A constant "ones" channel carrying biases drifts with activation and rounding | Pick a value activation and bf16 give back exactly (16 for SiLU) |
| Random weights magnify bf16 / bfp16 rounding (flat logits: top-1 margin 0.23 vs 1.51 real); cosines 0.88–0.99 against IRON's reference are not porting errors | Check against HF with real weights |

## Open

- NPU1 on real hardware (only simulated 4 columns, and the same export shape on NPU2). Its 6 xclbins use all 6 hardware contexts (none if another process uses the NPU): exports past that need context swapping.
- xrtshim on Windows hardware; building with MSVC.
- The NPU's own power on stock Linux (no sensor; numbers are package PPT).
- `--load-pdi-to-ctrl-pkt` (configurations streamed as control packets by DMA; biggest expected cut of the 2.3 µs/KB part): aiecc stops at `npu_materialized.mlir` ("operand #0 does not dominate this use", `aie.lock`), and `internal/aieelf` does not support scheme 4 yet.
- GDN kernels' static arrays as IRON Worker local buffers (no zero blocks): ~8 ms a token on Qwen3.6, ~5 ms on Qwen3.5; not started.
- One resident array per layer for dense models (LFM3-1.7B: ~60 ms a token expected).
- Several commands in one ELF; address-dependent scratchpad words.
- ELF flow on UMQ-generation hardware; device heap growth.
- aiecc's warning about reusing a BD after `dma_free_task` (thought harmless, not proven).

## Not adopted

- An xclbin per operator on NPU2: 0.46 ms a switch (46 ms a token for 100 ops).
- Waiting with `WAIT_CMD`: AIE2 does not implement it.
- Retrying `EAGAIN`: means "heap not ready", loops forever.
- `SYNC_BO(FROM_DEVICE)`: `EINVAL` on Linux 7.0 for ordinary BOs.
- Per-word fitting of NPU1 dispatch scalars: breaks on ternaries (`s > 1 ? 511 : 0`); the generated C++ library is Linux-only.
- For Windows, XRT's legacy C API through purego (`xrtPLKernelOpen`, `xrtRunSetArg`: deprecated, not sure to work on the NPU, varargs) and ONNX Runtime's VitisAI EP (no kernel runs, a large dependency): xrtshim was chosen.
- Timed waits in XRT (`run.wait(timeout)`): re-waiting breaks the command state.
- `--reconfig-teardown`: small configs 95 → 79 µs, large ones 236 → 274 µs, more control code (483 → 597 KB).
- Reordering operators to share configurations: at most 40 reconfigurations a token on Qwen3.6 (~4 ms, 2 %), dependencies block it on dense models.
- `QGEMVGlu` at 1 core: 16.3 tok/s against 17.2 at 2 (two accumulations on one core).
- Merging small configurations (RMSNorm, add): does not pay.
- `-idle` for Neural Engine and Intel: reload takes tens of seconds to minutes.
- Pinning more memory for speed: no effect.
