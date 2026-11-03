# Setting up an AMD Ryzen AI NPU for hwnpu

This guide prepares a Linux machine with an AMD Ryzen AI processor (XDNA NPU: Phoenix, Hawk Point, Strix Point, Krackan Point and later) so that hwnpu can use its NPU, builds a small test design, and shows how to check the result.

> **Status.** On a Ryzen AI 9 365 (Strix Point, `npu4`, 8 columns) with the amdxdna driver of the stock Linux 7.0 kernel of Ubuntu 26.04 and firmware 1.1.2.64, `cmd/report` (step 4) passes every check, including the passthrough design built with mlir-aie v1.4.4 (2026-11-03). Phoenix and Hawk Point (`npu1`) are untested. Reports from other machines are welcome.

Japanese: [INSTALL_AMD.ja.md](INSTALL_AMD.ja.md)

## Supported platforms

| Platform | Supported | Notes |
|---|---|---|
| Linux (x86-64) | Yes | Kernel with the `amdxdna` driver (step 1). |
| Windows (x86-64) | **Untested** | Through XRT and xrtshim, a small C library of hwnpu's own (section 6). Tested on Linux against XRT 2.21, not yet on Windows. |
| WSL2 | **No** | WSL2 does not pass the NPU to Linux (there is no `/dev/accel` inside WSL2). Use Windows itself (section 6) or a native Linux installation. |
| macOS | Not applicable | Macs have no AMD NPU. |

## What hwnpu needs

On Linux, hwnpu talks to the **amdxdna** kernel driver directly through `ioctl` and `mmap`. It does not use XRT at run time. On Windows it goes through XRT (section 6).

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

## 1. Driver and firmware (Ubuntu 24.04)

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

## 2. Build a test design

`cmd/report` runs the IRON example `passthrough_dmas`. It only copies a buffer with the NPU's DMA, so it needs no compiler for the AIE cores.

IRON requires **Python 3.12**. If your distribution ships another version, [uv](https://docs.astral.sh/uv/) installs it without root:

```bash
uv python install 3.12
```

Check out a release tag. Without one, the script finds no prebuilt wheel and stops.

```bash
git clone --depth 1 --branch v1.4.4 https://github.com/Xilinx/mlir-aie.git
```

```bash
cd mlir-aie
```

```bash
PATH=$(dirname $(uv python find 3.12)):$PATH source utils/env_install.sh --no-pre-commit ironenv
```

```bash
source utils/env_setup.sh
```

It ends with `xrt-smi not found`. hwnpu does not use XRT, so ignore it.

The xclbin is packaged by `xclbinutil`, an XRT tool. If you installed the XRT packages (`libxrt-utils` from the PPA of step 1, or from Ubuntu 26.04 universe), it is already there. Without root, unpack the packages instead (Ubuntu 26.04):

```bash
mkdir -p ~/xrt-debs && cd ~/xrt-debs
```

```bash
apt-get download libxrt-utils libxrt2 libboost-filesystem1.90.0 libboost-program-options1.90.0
```

```bash
for d in *.deb; do dpkg -x $d root; done
```

```bash
printf '#!/bin/sh\nLD_LIBRARY_PATH=%s/root/usr/lib/x86_64-linux-gnu exec %s/root/usr/bin/xclbinutil "$@"\n' $PWD $PWD > xclbinutil && chmod +x xclbinutil
```

```bash
export AIE_XCLBINUTIL=~/xrt-debs/xclbinutil
```

Build only the xclbin and the instruction stream. Use `-d npu` for Phoenix and Hawk Point, `-d npu2` for Strix Point and later:

```bash
cd programming_examples/basic/passthrough_dmas
```

```bash
python3 passthrough_dmas.py -d npu2 -n 4096 --xclbin-path=build/final.xclbin --insts-path=build/insts.bin
```

The design copies 4096 int32 values (16384 bytes), which is the default of `cmd/report`. If you build it with another `-n`, pass `-bytes` to the report.

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

It checks buffer allocation and cache synchronization, an instruction buffer in the device heap, the power meter (the NPU's sensor, `QUERY_SENSORS`, exists only in AMD's DKMS driver; on the stock Linux 7.0 kernel hwnpu reads the package power the amdgpu driver reports, PPT, which needs no root and covers the whole APU), and then loads the design, runs it, checks that the output equals the input and measures latency and NPU energy. Without `-xclbin` the kernel check is skipped. If the kernel in your xclbin is not named `MLIR_AIE`, pass `-kernel NAME`.

To compare power policies (hardware context priority):

```bash
go run ./cmd/report -xclbin final.xclbin -insts insts.bin -power low -o low.md
```

```bash
go run ./cmd/report -xclbin final.xclbin -insts insts.bin -power performance -o performance.md
```

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
- Argument order and sizes come from the xclbin metadata. IRON designs use `opcode (3), instr, ninstr, buffers...`. mlir-aie v1.4.4 lists unused buffers in the signature too (`bo0` to `bo4`). Arguments you leave unset are sent as zero, as in XRT.

## 6. Windows (untested)

AMD's NPU driver for Windows is reached only through XRT's C++ library, `xrt_coreutil.dll`, which the driver installs. hwnpu has no cgo, so it calls XRT through **xrtshim** (`native/xrtshim`), a C library of its own, loaded at run time (the `xrt` backend, `internal/backend/amdxrt`). The same backend runs on Linux against the XRT packages, which is how it is tested: on the Ryzen AI 9 365 with XRT 2.21.75, it passes `cmd/report`'s passthrough design (an xclbin), IRON's full ELFs with per-call values, and Gemma 4 E2B with images and audio at the speed of the Linux driver (2026-11-04). On Windows it has not run yet: reports are welcome.

1. Install the NPU driver (Ryzen AI driver, from AMD). `C:\Windows\System32\AMD\xrt-smi.exe examine` should list the NPU. A full ELF (Gemma 4 E2B) needs a recent driver, whose XRT creates a hardware context from an ELF (`xrt::hw_context(device, xrt::elf)`; XRT 2.21 does).
2. Build xrtshim with Visual Studio 2022 (C++ workload) and CMake, in a "Developer PowerShell for VS":

   ```powershell
   cd native\xrtshim
   .\make-importlib.ps1          # xrt_coreutil.lib, from the driver's xrt_coreutil.dll
   cmake -B build -DXRT_LIBRARY=xrt_coreutil.lib
   cmake --build build --config Release
   ```

   The XRT headers (2.21) are in `native/xrtshim/include`. If the driver's XRT is much older or newer, its C++ ABI may differ: take headers of its version from XRT's sources (`-DXRT_INCLUDE_DIR`).
3. Put `build\Release\xrtshim.dll` next to hwnpu's executables (or set `HWNPU_XRTSHIM` to it). hwnpu finds `xrt_coreutil.dll` in `System32\AMD` or the driver store; `HWNPU_XRT_CORE` names another.
4. Build hwnpu for Windows (`GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build ./cmd/...`) and run `hwnpuinfo`: the NPU appears with the backend `xrt`.

Models are exported on Linux (IRON and mlir-aie, see the README), and the exported directory is copied over: `hwnpu-chat -iron DIR` runs it the same way. What differs from Linux:

- The power meter reads XRT's electrical info, when the driver reports the NPU's power there.
- XRT allocates the control code. On Linux 7.0 its device heap holds 64 MiB, as hwnpu's own driver path does; Windows' may differ.

On Linux, the `xrt` backend lists the NPU only with `HWNPU_BACKEND=xrt` (build the shim with `native/xrtshim/build-linux.sh`, and point `HWNPU_XRTSHIM` at it).

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
