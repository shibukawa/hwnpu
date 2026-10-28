# Setting up an Intel NPU for hwnpu

This guide prepares an Intel Core Ultra machine (Meteor Lake, Arrow Lake, Lunar Lake, Panther Lake, Wildcat Lake) so that hwnpu can use its NPU, and shows how to check the result.

> **Status.** The Intel backend has been tested against a simulated Level Zero driver only. Running `cmd/report` on real hardware (step 4) and sending the report back is the most useful help right now.

Japanese: [INSTALL_INTEL.ja.md](INSTALL_INTEL.ja.md)

## Supported platforms

| Platform | Supported | Notes |
|---|---|---|
| Linux (x86-64) | Yes | Ubuntu 24.04 / 26.04 with Intel's NPU driver packages (step 1). |
| Windows 11 (x86-64) | Yes | With Intel's NPU driver (step 2). |
| WSL2 | **No** | WSL2 passes only the GPU to Linux (`/dev/dxg`). The NPU is not visible inside WSL2, so use Windows directly. |
| macOS | Not applicable | Macs have no Intel NPU. On Apple Silicon hwnpu uses Core ML instead. |

## What hwnpu needs

hwnpu talks to the NPU through **Level Zero** and the NPU graph extension, loaded at run time with purego. No cgo, no OpenVINO runtime.

| Component | Linux | Windows |
|---|---|---|
| Kernel driver | `intel_vpu` (in the mainline kernel) | Intel NPU driver |
| Firmware | `intel-fw-npu` package | included in the driver |
| User mode driver | `intel-level-zero-npu` (`libze_intel_npu.so.1`) | included in the driver |
| Driver compiler (to compile OpenVINO IR) | `intel-driver-compiler-npu` | included in the driver |
| Level Zero loader | `libze1` (`libze_loader.so.1`) | `ze_loader.dll` |
| Device access | read/write on `/dev/accel/accel0` (`render` group) | none |
| Go | 1.27 or later (see `go.mod`), `CGO_ENABLED=0` works | same |

The driver, driver compiler and firmware must come from the **same release**. A mismatch shows up as compile errors such as `MAPPED_INFERENCE_VERSION is NOT compatible with the ELF`.

## 1. Linux (Ubuntu 24.04 / 26.04)

These steps follow the instructions of [intel/linux-npu-driver](https://github.com/intel/linux-npu-driver/releases). The commands below are for **v1.38.0** (checked on 2026-10-28); take the file names from the latest release when you install.

Remove old packages:

```bash
sudo dpkg --purge --force-remove-reinstreq intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu intel-level-zero-npu-dbgsym
```

Download and extract the release (Ubuntu 24.04 shown; use the `ubuntu2604` archive on 26.04):

```bash
wget https://github.com/intel/linux-npu-driver/releases/download/v1.38.0/linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz
```

```bash
tar -xf linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz
```

Install the dependency of the driver compiler, then the packages:

```bash
sudo apt update && sudo apt install libtbb12
```

```bash
sudo dpkg -i *.deb
```

Install the Level Zero loader. The release page links the `libze1` package it was validated with (v1.32.0 for v1.38.0):

```bash
wget https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260830T100000Z/pool/main/l/level-zero-loader/libze1_1.32.0-1~24.04~ppa1_amd64.deb
```

```bash
sudo dpkg -i libze1_*.deb
```

If `libze1` conflicts with an older Level Zero installation, remove it first:

```bash
sudo dpkg --purge --force-remove-reinstreq level-zero level-zero-devel
```

Give your user access to the device, then log out and back in (or reboot):

```bash
sudo gpasswd -a ${USER} render
```

Check that the device node exists and is accessible:

```bash
ls -l /dev/accel/accel0
```

It should be owned by group `render` with mode `crw-rw----`. If the group is `root`, fix it until the next reboot with:

```bash
sudo chown root:render /dev/accel/accel0 && sudo chmod g+rw /dev/accel/accel0
```

## 2. Windows 11

1. Install the latest **Intel NPU driver** for Windows from Intel's download center, or through Windows Update. Task Manager shows an **NPU** entry under Performance when the driver works.
2. Check that the Level Zero loader is present:

   ```powershell
   where.exe ze_loader.dll
   ```

   hwnpu loads `ze_loader.dll` through the normal DLL search path. Where the NPU driver installs it is **not verified yet**. If `where.exe` finds nothing, look under `C:\Windows\System32` and the driver's folder, or take `ze_loader.dll` from a [Level Zero release](https://github.com/oneapi-src/level-zero/releases), and point hwnpu at it:

   ```powershell
   $env:HWNPU_ZE_LOADER = "C:\path\to\ze_loader.dll"
   ```

WSL2 cannot be used: it exposes the GPU (`/dev/dxg`) but not the NPU.

## 3. Get hwnpu

```bash
git clone https://github.com/shibukawa/hwnpu.git
```

```bash
cd hwnpu
```

On a machine without Go you can cross-compile the report on another machine and copy the single executable; the test models are embedded:

```bash
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o report.exe ./cmd/report
```

## 4. Check the installation

```bash
go run ./cmd/hwnpuinfo
```

An Intel NPU appears with `backend: levelzero`. The `driver` line lists the Level Zero version, the driver version, the graph extension, the compiler version and whether the NPU command queue extension (turbo, workload type) is available.

Then run the report and keep the Markdown file:

```bash
go run ./cmd/report -o report.md
```

For each Intel NPU it compiles two OpenVINO IR models in strict mode, compares the results with fp32 references (the NPU computes in fp16, so small differences are expected), and measures latency.

To compare power policies, run it twice. On Linux the package energy (RAPL) is readable by root only, so run these with `sudo` to get the energy columns:

```bash
go run ./cmd/report -power low -o low.md
```

```bash
go run ./cmd/report -power performance -o performance.md
```

## 5. Preparing models

hwnpu takes models in two forms:

- **OpenVINO IR** (`model.xml` and `model.bin`): compiled by the driver compiler when loaded. Convert from ONNX, PyTorch or TensorFlow with OpenVINO's `ovc` tool or `openvino.convert_model` (this needs OpenVINO only on the machine that converts). Use static shapes: the NPU does not run dynamic shapes.
- **Compiled blob**: the output of a previous compilation for the same driver and compiler. With `hwnpu.WithCacheDir(dir)` hwnpu stores and reuses blobs itself.

```go
dev, _ := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorIntel), hwnpu.WithCacheDir(cacheDir))
gr, _ := dev.Graph()
g, err := gr.Compile(ctx, hwnpu.GraphSource{Path: "model.xml"})
```

A model with an operation the NPU compiler does not support fails with `hwnpu.ErrNotOnNPU`; the error lists the operations and includes the compiler log.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `runtime library not found ... libze_loader.so.1` | Install `libze1` (step 1), or set `HWNPU_ZE_LOADER` to the loader's path. |
| `found /dev/accel/accel0 but cannot use it` | The device exists but Level Zero failed; the message carries the reason. |
| `permission denied` on `/dev/accel/accel0` | Add the user to `render` and log in again (step 1). |
| No device, `/dev/accel/accel0` missing | Check `sudo dmesg \| grep -i vpu` for firmware load errors; reinstall `intel-fw-npu` from the same release as the driver. |
| `MAPPED_INFERENCE_VERSION is NOT compatible` | Driver, compiler and firmware are from different releases. Reinstall all packages of one release. |
| `the driver has no compiler for OpenVINO IR` | `intel-driver-compiler-npu` is missing. Install it, or load precompiled blobs. |
| `ZE_extension_graph version ... is not supported` | The driver is too old for hwnpu (graph extension 1.5 or later is needed). Update the driver. |
| Need more detail from Level Zero | Run with `ZE_ENABLE_LOADER_DEBUG_TRACE=1`. |
