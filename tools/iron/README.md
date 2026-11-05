# IRON benchmarks for the AMD NPU

Builds mlir-aie (IRON) example designs for npu2 (Strix Point and later), runs them through hwnpu's public kernel API and checks the output against IRON's own references. Results and their reading: [docs/amd-xdna2-bench.md](../../docs/amd-xdna2-bench.md).

| file | what it does |
|---|---|
| `ironbench/` | Go runner: loads an xclbin, binds `opcode, instr, ninstr, bo0..` and times the runs (JSON on stdout) |
| `bench.sh` | build + run + check, one JSON line per design |
| `gemm.py` | `basic/matrix_multiplication/whole_array` (i8, i16, bf16; bf16 emulated with bfp16) |
| `gemm_bfp.py` | `ml/block_datatypes/matrix_multiplication/whole_array` (bfp16ebs8 in and out) |
| `mk_memcpy.py` | `getting_started/00_memcpy` (8 columns x 2 DMA channels) |
| `dma_bw.py` | one-way DMA: DDR -> cores (read) or cores -> DDR (write), cores only acquire and release |
| `q4deq.py` | q4nx (4-bit, scale and minimum per 32) -> bfp16 with `kernels.quant.q4nx_dequant`, one core per DMA channel, checked byte for byte |
| `q4gemv.py`, `q4gemv.cc` | decode GEMV y = W x with q4nx weights (FastFlowLM's `q4nx_accumulate`), bf16 x, f32 y, on up to 8 x 4 cores; weights split and x broadcast through the memtiles |
| `lfm3conv.py`, `lfm3conv.cc` | one decode step of an LFM3 convolution layer (both norms, in_proj, the gated convolution, out_proj, SwiGLU, the residuals) in one NPU call, 8 x 2 cores, q4nx weights; checked against a numpy model of the cores' arithmetic with the checkpoint's layer 0 |
| `npu_clock.py` | prints the NPU clocks (`GET_INFO(QUERY_CLOCK_METADATA)`) |
| `ironcompile.py` | loads a design file and compiles it for npu2 |

Setup (mlir-aie v1.4.4, Python 3.14; see [INSTALL_AMD.md](../../INSTALL_AMD.md) step 2). The environment of IRON's pinned mlir-aie (1.4.5.dev119, [docs/amd-npu-models.md](../../docs/amd-npu-models.md) section 10) works too (`dma` and `gemm` checked on 2026-11-06); set `PEANO_INSTALL_DIR` to its `site-packages/llvm-aie` and `MLIR_AIE_SRC` to an mlir-aie checkout.

```bash
python3 -m venv iron && iron/bin/pip install https://github.com/Xilinx/mlir-aie/releases/download/v1.4.4/mlir_aie-1.4.4-cp314-cp314-manylinux_2_35_x86_64.whl
```

```bash
git clone --depth 1 --branch v1.4.4 https://github.com/Xilinx/mlir-aie.git
```

```bash
iron/bin/pip install -r mlir-aie/utils/peano-requirements.txt
```

```bash
go build -o ironbench ./tools/iron/ironbench && go build -o hwnpu-xclbin ./cmd/hwnpu-xclbin
```

Environment:

```bash
export PATH=$PWD/iron/bin:$PATH PEANO_INSTALL_DIR=$(iron/bin/pip show llvm-aie | awk '/^Location:/ {print $2 "/llvm-aie"}') MLIR_AIE_SRC=$PWD/mlir-aie AIE_XCLBINUTIL=$PWD/hwnpu-xclbin IRONBENCH=$PWD/ironbench OUT=$PWD/iron-out
```

Examples:

```bash
tools/iron/bench.sh memcpy 64M 16777216
```

```bash
tools/iron/bench.sh gemm i8i32_4096 i8 i32 4096 4096 4096 64 64 64 8 0
```

```bash
tools/iron/bench.sh bfp 2048 2048 2048 2048 64 64 64 8
```

```bash
tools/iron/bench.sh dma read64M read 16777216 4 2 1024
```

```bash
tools/iron/bench.sh q4deq 8192 8192 4 2
```

```bash
tools/iron/bench.sh q4gemv 16384x4096 16384 4096 8 4
```

The LFM3 layer reads the checkpoint (config.json, model.safetensors) and a reference of `tools/ref/ref_logits.py` for its input:

```bash
tools/iron/bench.sh lfm3conv layer0 ~/.cache/hwnpu/models/LFM3-1.7B ref/lfm3-1.7b
```

`gemm` takes `NAME DTIN DTOUT M K N m k n COLS EMU`: M must be a multiple of 4m, N of n x COLS, m of 2r where (r, s, t) is the MMUL shape (i8 8x8x8, i16 4x4x8, bf16 4x8x8, bf16 emulated 8x8x8). Checking a 4096³ result in numpy takes minutes.
