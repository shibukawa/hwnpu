# Setting up an AMD Ryzen AI NPU for hwnpu

This guide prepares a Linux machine with an AMD Ryzen AI processor (XDNA NPU: Phoenix, Hawk Point, Strix Point, Krackan Point and later) so that hwnpu can use its NPU, builds a small test design, and shows how to check the result.

> **Status.** Checked on a Ryzen AI 9 365 (Strix Point, npu2 designs) with Ubuntu 26.04 and its in-tree driver: `cmd/report` runs the IRON passthrough design and the output matches. Phoenix and Hawk Point (npu designs) and AMD's DKMS driver have not been tested; reports from those machines are the most useful help right now.

Japanese: [INSTALL_AMD.ja.md](INSTALL_AMD.ja.md)

## Supported platforms

| Platform | Supported | Notes |
|---|---|---|
| Linux (x86-64) | Yes | Kernel with the `amdxdna` driver (step 1). |
| Windows | **No** | Not supported yet. AMD's Windows driver is only reachable through XRT's C++ library (`xrt_coreutil.dll`), which hwnpu cannot call without cgo. |
| WSL2 | **No** | Windows is not supported, and WSL2 does not pass the NPU to Linux (there is no `/dev/accel` inside WSL2). Use a native Linux installation. |
| macOS | Not applicable | Macs have no AMD NPU. |

## What hwnpu needs

hwnpu talks to the Linux **amdxdna** kernel driver directly through `ioctl` and `mmap`. It does not use XRT at run time. Windows is not supported yet.

| Component | Needed by hwnpu | Notes |
|---|---|---|
| Linux kernel with `amdxdna` | yes | Mainline since 6.14 (`CONFIG_DRM_ACCEL_AMDXDNA`). Newer is better: AMD's tools ask for 6.17 or later. |
| NPU firmware (`/usr/lib/firmware/amdnpu/`) | yes | From `linux-firmware`, or from the `amdxdna-dkms` / XRT plugin packages. |
| Read/write on `/dev/accel/accelN` | yes | `render` group. |
| XRT (`xrt-smi`) | no | Useful to validate the installation. |
| ROCm | no | ROCm is the GPU stack (`amdgpu`, HIP, ROCr). The NPU has its own driver (`amdxdna`), runtime (XRT) and firmware, and works without ROCm. |
| IRON / mlir-aie and Peano | no | Needed to **build** kernels (xclbin and instruction stream). |
| Go | yes | 1.27 or later (see `go.mod`), `CGO_ENABLED=0` works. |

Unlike Intel, the AMD NPU does not take models. It runs kernels built with IRON / mlir-aie: an **xclbin** (the AIE array configuration and kernel metadata) plus an **instruction stream** (`insts.bin`).

## 1. Driver and firmware

### Ubuntu 26.04

Ubuntu 26.04 ships a kernel (7.0) with the `amdxdna` driver and the NPU firmware in `linux-firmware`, so nothing needs to be installed for hwnpu. Check that the device is there:

```bash
ls -l /dev/accel/
```

```bash
sudo dmesg | grep -i amdxdna
```

On a desktop session, systemd-logind gives the logged-in user access to `/dev/accel/accel0`. For SSH-only use, add the user to the `render` group as described below. The in-tree driver does not report the NPU power sensor, so `cmd/report` skips that check. Continue with step 2.

### Ubuntu 24.04

Use the latest BIOS and make sure the NPU (sometimes called IPU) is enabled in the BIOS. The mlir-aie instructions also recommend disabling Secure Boot, so that the DKMS driver module can load.

Check the kernel version:

```bash
uname -r
```

If it is older than 6.17, install the HWE kernel and reboot:

```bash
sudo apt update && sudo apt install --install-recommends linux-generic-hwe-24.04
```

The simplest way to get a current driver, firmware and XRT is AMD's PPA, as recommended by [mlir-aie](https://github.com/Xilinx/mlir-aie) (checked on 2026-10-28):

```bash
sudo add-apt-repository ppa:amd-team/xrt
```

```bash
sudo apt update && sudo apt install libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu amdxdna-dkms
```

Reboot after the installation. hwnpu itself only needs the kernel driver and firmware; with a recent kernel and `linux-firmware` the in-tree driver is enough for hwnpu, but the PPA keeps driver and firmware versions matched.

Give your user access to the device, then log out and back in:

```bash
sudo usermod -aG render $USER
```

The driver pins buffer memory. If allocations fail, raise the locked memory limit (AMD's recommendation), then log out and back in:

```bash
printf '* soft memlock unlimited\n* hard memlock unlimited\n' | sudo tee /etc/security/limits.d/99-amdxdna.conf
```

Check the device:

```bash
ls -l /dev/accel/
```

```bash
xrt-smi examine
```

`xrt-smi` comes from the XRT packages and is optional for hwnpu.

## 2. Build the test design

`cmd/report` runs the IRON example `passthrough_dmas`, which copies a buffer through the NPU's DMAs and uses no AIE core. aiecc still needs Peano (llvm-aie), which `env_install.sh` installs. Set up IRON as described in the mlir-aie README:

```bash
git clone https://github.com/Xilinx/mlir-aie.git
```

```bash
cd mlir-aie
```

```bash
sudo apt install build-essential clang lld cmake ninja-build python3-venv python3-pip uuid-dev
```

```bash
source utils/env_install.sh
```

```bash
source utils/env_setup.sh
```

Build only the xclbin and the instruction stream. Use `--dev npu` for Phoenix and Hawk Point, `--dev npu2` for Strix Point and later. The flag names come from the example's documentation; check `--help` if they differ in your version:

```bash
cd programming_examples/basic/passthrough_dmas
```

```bash
python3 passthrough_dmas.py --dev npu --xclbin-path final.xclbin --insts-path insts.bin
```

The design copies 4096 int32 values (16384 bytes), which is the default of `cmd/report`. If you build it with another `-n`, pass `-bytes` to the report.

### Without XRT

aiecc packages the xclbin with `xclbinutil`, which comes with XRT. Without XRT (for example on Ubuntu 26.04 with the in-tree driver) the build stops with `tool 'xclbinutil' not found`. hwnpu has a stand-in that takes the same arguments; build it from the hwnpu checkout (step 3) and point aiecc at it:

```bash
go build -o ~/.local/bin/hwnpu-xclbin ./cmd/hwnpu-xclbin
```

```bash
AIE_XCLBINUTIL=~/.local/bin/hwnpu-xclbin python3 passthrough_dmas.py --dev npu2 --xclbin-path final.xclbin --insts-path insts.bin
```

The result holds what hwnpu reads (kernel metadata, compute units, AIE partition and PDIs) and is not meant for XRT.

If `python3 -m venv` fails because `ensurepip` is missing, install `python3-venv` (`sudo apt install python3-venv`).

## 3. Get hwnpu

```bash
git clone https://github.com/shibukawa/hwnpu.git
```

```bash
cd hwnpu
```

## 4. Check the installation

```bash
go run ./cmd/hwnpuinfo
```

An AMD NPU appears with `backend: xdna`, its generation, the number of AIE columns, the AIE version and the firmware version.

Run the report with the design from step 2:

```bash
go run ./cmd/report -xclbin /path/to/final.xclbin -insts /path/to/insts.bin -o report.md
```

It checks buffer allocation and cache synchronization, an instruction buffer in the device heap, the NPU power sensor, and then loads the design, runs it, checks that the output equals the input and measures latency and NPU energy. Without `-xclbin` the kernel check is skipped. If the kernel in your xclbin is not named `MLIR_AIE`, pass `-kernel NAME`.

To compare power policies (hardware context priority):

```bash
go run ./cmd/report -xclbin final.xclbin -insts insts.bin -power low -o low.md
```

```bash
go run ./cmd/report -xclbin final.xclbin -insts insts.bin -power performance -o performance.md
```

Benchmarks of the data types and throughput on Strix Point, and the scripts behind them: [docs/amd-xdna2-bench.md](docs/amd-xdna2-bench.md) (Japanese), [tools/iron](tools/iron/README.md).

## 5. Using your own kernels

```go
dev, _ := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
kr, _ := dev.Kernel()
prog, _ := kr.Load(ctx, hwnpu.ProgramSource{Path: "final.xclbin"})
k, _ := prog.Kernel("MLIR_AIE")
for _, a := range k.Args() { fmt.Println(a.Index, a.Name, a.Type, a.Buffer) }
```

- Put the instruction stream in `dev.AllocBuffer(n, hwnpu.WithBufferKind(hwnpu.BufferInstructions))`; it must live in the device heap.
- The NPU is not cache coherent: call `SyncToDevice` after writing inputs and `SyncFromDevice` before reading outputs.
- Argument order and sizes come from the xclbin metadata. IRON designs use `opcode (3), instr, ninstr, buffers...`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| No AMD device in `hwnpuinfo` | Check `ls /dev/accel/` and `sudo dmesg \| grep -i amdxdna`. The driver needs a kernel with `amdxdna` and matching firmware in `/usr/lib/firmware/amdnpu/`. |
| `permission denied ... add the user to the render group` | `sudo usermod -aG render $USER`, then log in again. |
| `create device heap` fails | Another process may hold the NPU's resources, or the firmware failed to load (see `dmesg`). Raising the memlock limit (step 1) helps with `ENOMEM`. |
| `no valid DPU kernel found in the xclbin` | The xclbin has no PDI for its kernel id; rebuild it for the right `--dev`. |
| `the xclbin needs N columns, the device has M` | The design was built for a larger NPU (`npu2` designs on an `npu1` device). |
| `command finished with ERT_CMD_STATE_ERROR` or `TIMEOUT` | The firmware rejected or aborted the run. Check `dmesg`. Driver and firmware version mismatches cause aborts: reinstall them from one source. |
| Changing the device power mode | `xrt-smi` or the driver's `SET_STATE` ioctl can do it as root; hwnpu does not change it because it affects every process. |
