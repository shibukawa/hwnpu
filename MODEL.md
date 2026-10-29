# Models and data formats

What each NPU computes in, what hwnpu accepts, and how to prepare models. Facts from vendor documentation are linked; observations from hwnpu's own tests are marked as such.

Japanese: [MODEL.ja.md](MODEL.ja.md)

## Summary

| | Intel NPU | Apple Neural Engine | AMD XDNA NPU |
|---|---|---|---|
| hwnpu backend | `levelzero` | `coreml` | `xdna` |
| What hwnpu loads | OpenVINO IR (`.xml` + `.bin`), or a blob compiled by the same driver | Core ML model (`.mlpackage`, `.mlmodel`, `.mlmodelc`) | xclbin + instruction stream built with IRON / mlir-aie (kernels, not models) |
| Native compute | fp16; int8 for quantized models | fp16; int8-int8 on newer chips (A17 Pro, M4) | int8, int16, bf16; XDNA2 adds block floating point (BFP16) |
| fp32 models | Accepted, executed in fp16 | Placed on the CPU (strict mode rejects them) | Converted to bf16 by AMD's tools; fp32 is emulated, not native |
| Shapes | Static only | Static recommended; hwnpu uses the default shape | Fixed by the kernel |
| Who compiles for the device | Driver compiler, at load time (cached with `WithCacheDir`) | Core ML, at load time (cached with `WithCacheDir`) | Offline, with mlir-aie and Peano |
| Placement check | Compiler result; unsupported ops are listed | `MLComputePlan` (estimate) | Always NPU |

## Intel NPU

**Accepted formats**

- OpenVINO IR v11. hwnpu reads the `.xml` and the `.bin` next to it, and passes them to the driver compiler like the OpenVINO NPU plugin does.
- A compiled blob from the same driver and compiler version. Blobs are not portable across driver versions, so ship IR and let `WithCacheDir` keep the blobs.

**Data types**

- The NPU computes in fp16. OpenVINO's NPU plugin accepts fp32, fp16 and quantized u8 models, and quantized models may be int8 or mixed fp16-int8 ([OpenVINO NPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)).
- Use int8 quantization (NNCF) for the best speed and power. For LLMs, OpenVINO GenAI documents low-bit weight compression for the NPU; check its NPU guide for the formats your driver supports.
- `ArgInfo.DType` is the type the buffer must hold (device precision). `ArgInfo.ModelDType` is the type declared by the model. They differ when the device converts precision.

| ArgInfo.DType | Size | Notes |
|---|---|---|
| `F32`, `F16`, `BF16`, `F64` | 4, 2, 2, 8 | Use `hwnpu.EncodeF16` / `DecodeF16` for fp16. |
| `U8`, `I8`, `I16`, `I32`, `I64` | 1, 1, 2, 4, 8 | |

**Constraints**

- Shapes must be static. hwnpu rejects IR with dynamic dimensions.
- The model's opset must be supported by the driver compiler. `hwnpuinfo` shows the compiler version.
- An operation the compiler does not support fails the load with `ErrNotOnNPU`, listing the operation. The NPU never falls back to the CPU.

**Converting**

- From ONNX, PyTorch or TensorFlow: OpenVINO's `ovc` command or `openvino.convert_model`, then `openvino.save_model` (fp16 compression is the default when saving).
- Fix input shapes during conversion (`--input "x[1,3,224,224]"` for `ovc`).
- Quantize with NNCF post-training quantization.

## Apple Neural Engine (Core ML)

**Accepted formats**

- `.mlpackage` (ML Program), `.mlmodel`, or a compiled `.mlmodelc`. Use the **ML Program** format: placement can only be verified for ML Programs, and strict mode rejects models it cannot verify.

**Data types**

- The Neural Engine computes in fp16. In hwnpu's tests, an fp32 ML Program was placed entirely on the CPU, so strict mode rejects fp32 models.
- Weights can be compressed: Core ML supports int8 and int4 linear quantization and palettization to 1 to 8 bits for weights, and int8 for activations. W8A8 uses a faster int8-int8 path on newer hardware such as A17 Pro and M4 ([coremltools optimization overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)).
- hwnpu supports multi-array inputs and outputs only (no image or dictionary features):

| Core ML multi-array type | ArgInfo.DType |
|---|---|
| Float16 | `F16` |
| Float32 | `F32` |
| Float64 | `F64` |
| Int32 | `I32` |
| Int8 | `I8` |

**Constraints**

- hwnpu uses each input's default shape. Flexible shapes are not supported.
- Observed in hwnpu's tests (macOS 27): small models are placed on the CPU even in fp16. A linear layer with a (64, 512) input and 512 outputs ran on the Neural Engine; (32, 256) and smaller ran on the CPU, as did 1x1 convolutions up to 256 channels at 32x32. Batch small inputs together if they must run on the Neural Engine.
- Placement is an estimate from `MLComputePlan` (macOS 14.4 or later). Core ML does not report where a prediction actually ran.

**Padding small models to reach the Neural Engine**

Core ML decides placement from the whole model's cost, not from a fixed threshold. In hwnpu's tests (macOS 27):

| Model (fp16) | Placement |
|---|---|
| Linear 256 to 256, batch 1 to 40 | CPU |
| Linear 256 to 256, batch 48 or more | Neural Engine (batch 56 split: linear on the Neural Engine, relu on the CPU) |
| Linear 64 to 64, any batch up to 4096 | CPU |
| Linear, batch 1, width up to 4096 | CPU (matrix times vector) |
| 3 layer MLP, width 128, batch 1 to 48 | CPU |
| 3 layer MLP, width 128, batch 64 or more | Neural Engine |
| 3x3 convolution, 32 channels, 32x32, batch 1 or 8 | CPU; batch 64: Neural Engine |

Growing the **batch** (first) dimension is what moves a model to the Neural Engine. Padding the batch with zero rows does not change the results of models that process rows independently, which covers most inference models. Padding the **feature** dimensions with zero weights (like embedding a small matrix in a larger affine transform) keeps linear and relu chains exact, but changes normalizations such as softmax or layer norm, and did not move any tested model to the Neural Engine.

`tools/coreml/ane_pad.py` rebuilds an ML Program with a larger batch, tries batch sizes from small to large, and keeps the smallest one at which every operation is on the Neural Engine:

```bash
python tools/coreml/ane_pad.py model.mlpackage model_ane.mlpackage --verify
```

It records the original and padded batch in the model's metadata. hwnpu reads them: `ArgInfo.Dims` and `Bytes` keep the original batch, `ArgInfo.PaddedDims` shows the shape that runs, and hwnpu copies the caller's rows into zero padded buffers and copies the results back.

**Measure before you adopt padding.** For the 3 layer MLP above, one row took 73 µs on the CPU and 280 µs when padded to 64 rows on the Neural Engine; 64 rows on the CPU took 62 µs (measured through hwnpu by cmd/powerbench). Starting the Neural Engine costs more than this model's work, so Core ML's choice of the CPU was right for latency. Slower is still worth it if the Neural Engine uses less energy, so measure energy, not only latency:

```bash
go run ./cmd/powerbench
```

It asks for the sudo password once, runs powermetrics (CPU, GPU and ANE rails) while each model runs, subtracts idle power, and prints the energy per call (mJ/call = net power / calls per second). It measures the built-in `mlp_small` pair by default; pass `-model a.mlpackage -model b.mlpackage` for your own models, and `-rate 100` to compare at a fixed request rate, which is closer to how a small model runs in practice. `-json` and `-raw` keep the results and the raw powermetrics output. powermetrics values are estimates, so compare models on the same Mac only.

On an M3 MacBook Air the padded `mlp_small` did not pay off. Back-to-back, the CPU used 0.32 mJ per call for 1 row and 0.33 mJ for 64 rows, while the Neural Engine used 0.78 mJ for 64 rows (3.3 W of it on the CPU, 1.5 W on the ANE). At 100 calls per second the ANE rail alone drew 0.6 mJ per call, because the ANE starts and stops for every call. Core ML's fixed cost per call (about 55 µs) is larger than this model's work, so the CPU wins on energy as well as latency. Padding is worth trying only for models whose work outweighs that cost; measure them with `-model`. Padding pays off when the padded rows carry real work: collect requests and fill the batch instead of zeros.

**Converting**

- From PyTorch (via `torch.export` or tracing) or from the MIL builder with `coremltools.convert(..., convert_to="mlprogram", compute_precision=coremltools.precision.FLOAT16)`.
- Give fixed input shapes and fp16 inputs and outputs where possible.
- Compress weights with `coremltools.optimize`.
- coremltools has no maintained ONNX converter; convert from the original PyTorch model, or from ONNX through PyTorch.

### Where the Neural Engine pays off

Measured on an M3 MacBook Air (macOS 27) with `cmd/powerbench`: each model ran back-to-back, once with Core ML limited to the CPU and once with the CPU and the Neural Engine, and energy is the power above idle divided by calls per second (10 s per phase, 2 rounds).

| model | work per call | CPU only: latency / energy | Neural Engine: latency / energy | NPU energy vs CPU |
|---|---|---|---|---|
| encoder d128, seq 32, 1 layer | 13 MFLOP | 80 µs / 0.41 mJ | 173 µs / 0.91 mJ | 2.2x |
| encoder d256, seq 64, 2 layers | 210 MFLOP | 218 µs / 1.58 mJ | 206 µs / 1.49 mJ | 0.94x |
| all-MiniLM-L6-v2, seq 32 | 0.69 GFLOP | 790 µs / 4.5 mJ | 527 µs / 2.6 mJ | 0.57x |
| encoder d384, seq 128, 2 layers | 0.96 GFLOP | 579 µs / 5.7 mJ | 305 µs / 2.2 mJ | 0.38x |
| all-MiniLM-L6-v2, seq 128 | 2.9 GFLOP | 1.90 ms / 16.5 mJ | 0.74 ms / 4.3 mJ | 0.26x |
| encoder d768, seq 128, 4 layers | 7.4 GFLOP | 3.46 ms / 31.6 mJ | 1.15 ms / 7.4 mJ | 0.23x |
| encoder d1024, seq 256, 6 layers | 40 GFLOP | 16.7 ms / 164 mJ | 5.9 ms / 41 mJ | 0.25x |

The break-even point is around 200 MFLOP per call (a 1.6 M parameter, 2 layer encoder over 64 tokens). Below it, Core ML's fixed cost per call dominates and the CPU wins; above it, the Neural Engine is 2 to 3 times faster and uses a quarter of the energy, about 1 TFLOP per joule against 0.25 on the CPU. The Neural Engine is not low power in absolute terms (it drew up to 6 W here), but it finishes the same work with much less energy.

all-MiniLM-L6-v2, a real embedding model, behaves like the synthetic encoder of the same size even though Core ML keeps the token embedding lookup, the first layer norm and the pooling on the CPU (148 of 170 operations on the Neural Engine). hwnpu's strict mode rejects such a split model; open it with `WithPlacement(PlacementPrefer)`.

Idle power varied between 0.45 W and 1.06 W during the run, so results near the break-even point (the d256 row) are within noise.

Reproduce with:

```bash
python tools/coreml/gen_sweep.py build/sweep
python tools/coreml/convert_minilm.py build/minilm/minilm_s128_b1.mlpackage --seq 128
go run ./cmd/powerbench -model 'build/sweep/*.mlpackage' -model 'build/minilm/*.mlpackage' -json build/power-sweep.json
python tools/coreml/gen_sweep.py build/sweep --report build/power-sweep.json
```

`gen_sweep.py` writes BERT style encoders of increasing size with random weights (fp16, fixed shape). `convert_minilm.py` converts sentence-transformers/all-MiniLM-L6-v2 to an ML Program with a fixed sequence length (input_ids and attention_mask in, normalized 384 wide embedding out) and checks it against PyTorch (cosine similarity 0.99998).

## AMD XDNA NPU

**Accepted formats**

- An xclbin with an AIE partition and DPU kernel metadata, plus the instruction stream, as produced by IRON / mlir-aie (`aiecc`). hwnpu runs **kernels**: it does not take ONNX or other model files, because AMD's model compiler (Vitis AI, used by ONNX Runtime) is not a public library.

**Data types**

- The AIE-ML cores compute natively in int8, int16 and bf16. XDNA2 (Strix Point and later) adds a block floating point format (BFP16) in which eight values share an exponent ([GEMM optimization across Ryzen AI generations](https://arxiv.org/pdf/2512.13282)). fp32 is emulated. Native fp16 support was not found in the documentation.
- Buffers are raw bytes: the kernel defines their layout and type. Scalars are set with `SetUint32`, `SetUint64` or `SetBytes`, in the sizes listed by `Kernel.Args()`.
- hwnpu has fp16 helpers only. Convert bf16 in your code (the upper 16 bits of an fp32 value, with rounding).

**Constraints**

- Shapes, tiling and data types are fixed when the kernel is built.
- The design must fit the device: `npu1` (Phoenix, Hawk Point) has 4 columns, `npu4` (Strix Point) 8.
- Instruction streams must be allocated with `WithBufferKind(BufferInstructions)`, and buffers must be synchronized explicitly (`SyncToDevice`, `SyncFromDevice`).

**Running models on AMD**

- For ONNX models, AMD's supported route is ONNX Runtime with the Vitis AI execution provider (Ryzen AI Software). AMD's documentation describes int8 quantization for CNNs and bf16 for transformers there, with fp32 models converted to bf16. hwnpu does not cover this route yet.

## Preparing models: convert once, in Go, before distribution

hwnpu cannot convert models at load time without heavy dependencies (OpenVINO's C++ frontends, coremltools in Python, AMD's closed compiler), and converting on every machine would spend time and power against hwnpu's goal. So models are converted once by the developer, with one artifact per vendor in the precision that vendor's NPU reads fastest, and distributed (S3, NFS, a private Hugging Face repository). At load time the application picks the artifact for the device it finds and lets the vendor compile it; `WithCacheDir` keeps that result per machine.

The converter needs no Python: LLMs share a handful of block structures (norm, attention or convolution, gated MLP), so `cmd/hwnpu-convert` builds the graph in Go from `config.json` and reads the weights from the safetensors or GGUF checkpoint. The supported architectures are a whitelist (`-archs` lists them), so far Liquid AI's LFM3 and BERT encoders (all-MiniLM-L6-v2, E5, BGE, GTE and the like), detected from `config.json`. It writes Core ML packages; the Intel IR and AMD writers are not implemented yet.

```bash
go run ./cmd/hwnpu-convert -config build/lfm3/config.json -weights build/lfm3/LFM3-1.7B.safetensors -o lfm3.mlpackage -seq 32
```

The package takes `x`, the fp16 token embeddings `(1, seq, hidden)`, and returns `y`, the logits `(1, seq, vocab)`. The embedding lookup stays on the host because Core ML always places `gather` on the CPU. BERT packages take a second input, `mask` (fp16 `(1, seq)`, 1 for tokens and 0 for padding), and return the last hidden state `(1, seq, hidden)`; pooling (mean or CLS) and normalization into a sentence embedding are left to the host. The weights are 4 bit lookup tables (`-format lut4`), one table of 16 fp16 values per 16 output rows, which is the only compressed layout the Neural Engine runs at full speed (see below). `-format fp16` keeps the weights as they are, for checking the graph.

**Splitting a large model**: the Neural Engine takes a model only when its weight file is under a limit that depends on the chip; a larger model is placed on the CPU as a whole. The limit is per model, so `hwnpu-convert` splits a model over it into the fewest parts under it, balanced by size. `-target` picks the chip (the default, `any`, is the smallest limit of the known chips).

| `-target` | limit per model | source |
|---|---|---|
| `m1` | 1 GiB | assumed, from a report of half the M4's; to be measured with `npuprobe limit` |
| `m2` | 1 GiB | assumed the same as the M3's; to be measured with `npuprobe limit` |
| `m3` | 1 GiB | `npuprobe limit`: 0.99 GiB on the Neural Engine, 1.02 GiB on the CPU, in every format ([PROBE.md](PROBE.md)) |
| `m4` | 2 GB | measured: a 4 B model in lut4 (2 GB) ran on the Neural Engine |

The estimated weights of each part fill at most 97% of the limit, and after writing, the actual weight file of each part is checked to be under it. A split model is written as `out.part1.mlpackage`, `out.part2.mlpackage`, … with `out.parts.json`, which lists the parts in order with their inputs and outputs. Each part's input `x` is the hidden state entering its first layer (the token embeddings for the first part); other inputs such as `mask` go to every part. Run the parts in order with the buffer of each part's output `y` bound as the next part's `x`; the last part's `y` is the model's output. On the M3 the parts ran as fast as one model would (`npuprobe split` in PROBE.md). On the M4 too, 4 parts of 0.91 GiB ran as fast as 2 of 1.81 GiB, with the same error (58.8 and 58.3 ms per token). Every part is loaded at once, but the peak memory of the process while loading is about the weights of one part, so finer parts help a machine short of memory (on the M3, 1.81 GiB in 2 parts peaked at 1.03 GiB, in 4 at 0.55 GiB). The default `any`, one set of packages for every chip, is the one to use. Writing the parts as the functions of one package does not work: the limit counts the whole package, and every function falls to the CPU. `-parts N` sets the number of parts, and `-target none` turns splitting off. `-archs` lists the per chip limits after the architectures.

```bash
go run ./cmd/hwnpu-convert -config config.json -weights model.safetensors -o model.mlpackage -seq 32 -target m3
```

A conversion can be checked without a Mac. `tools/ref/ref_logits.py` saves reference activations from the Hugging Face implementation, and `cmd/hwnpu-check` runs the graph on the CPU in fp32 and compares it layer by layer. Each layer is fed the reference's own input, so errors do not pile up and the first wrong layer stands out; tensors in the checkpoint that the conversion does not read are reported too. With `-package` it compares a written package's logits (lut4, for example) with the reference.

```bash
python tools/ref/ref_logits.py build/lfm3 build/lfm3/ref --seq 32
go run ./cmd/hwnpu-check -config build/lfm3/config.json -weights build/lfm3/LFM3-1.7B.safetensors -ref build/lfm3/ref
go run ./cmd/hwnpu-check -package lfm3.mlpackage -ref build/lfm3/ref
```

A split model or a single one generates text with `cmd/hwnpu-generate`.`hwnpu-generate` builds a chat prompt with a tokenizer written in Go (the byte-level BPE of `tokenizer.json`), looks up the embeddings in the checkpoint and generates on the Neural Engine one token at a time.

Long text needs a KV cache. Converted with `-cache N`, the program takes `-seq` tokens per call (1 for generating), reads the keys and values of the tokens before them (and LFM3's convolution state) as inputs, and outputs the new ones. The cache lives in host buffers that `hwnpu-generate` appends to after every call. Each part holds only its own layers' cache, so it combines with splitting. Give `-package` the `.mlpackage` of a single model or the `.parts.json` of a split one.

```bash
go run ./cmd/hwnpu-convert -config build/lfm3/config.json -weights build/lfm3/LFM3-1.7B.safetensors -o build/lfm3/lfm3-kv.mlpackage -seq 1 -cache 4096 -quant kmeans-minmax -parts 2
go run ./cmd/hwnpu-generate -package build/lfm3/lfm3-kv.parts.json -config build/lfm3/config.json -weights build/lfm3/LFM3-1.7B.safetensors -n 1500 -prompt "Write an internal memo on why we should use Go."
```

The cache of a sliding attention layer (5 of LFM3's 24 layers, window 256) is a ring buffer of the window's length, so only the 6 full attention layers need the long cache. LFM3-1.7B (lut4, 2 parts) on an M3 takes 19.1 ms per token with a cache of 1024 tokens, 27.7 ms with 4096 and 38.5 ms with 8192: the cost follows the cache's capacity, not how much of it is used, so pick the length you need. The Neural Engine takes tensor dimensions up to 16384, so with `-cache` plus `-seq` over 16384 the attention runs on the CPU and is very slow (the converter warns). A model with a cache takes 15 to 20 seconds to load even with its compiled cache (about 1 second without a KV cache; not investigated yet). A model converted without a cache computes the whole `-seq` window for every token and drops the oldest tokens when it is full (about 34 ms at seq 128).

### Choosing the source checkpoint

Convert from the official bf16 checkpoint. Use a published quantized checkpoint only when it can be repacked without loss into the target's format, which is a one way move to a finer or wider scheme: symmetric int4 (GPTQ without reordering, compressed-tensors W4A16) fits Intel's symmetric int4, AWQ fits AMD's asymmetric uint4 with groups of 128. No published 4 bit format fits the Neural Engine's lookup tables, and re-quantizing a 4 bit checkpoint into another 4 bit format compounds the error: for LFM3-1.7B, bf16 to lut4 gives KL 0.95 nats against the bf16 model, the official QAD Q4_0 GGUF re-palettized gives 0.94 even though it starts closer (its own deviation is 0.35). Always measure the result: `tools/ref/ref_logits.py` saves the PyTorch hidden states and logits, `cmd/hwnpu-check`, `tools/ref/eval_logits.py` and the Go tests compare the package with them layer by layer.

### What the Neural Engine accepts (M3, macOS 27.0.1)

- Lookup tables must cover whole rows, 16 output channels per table (`per_grouped_channel` in coremltools). Tables with column blocks (16 x 64) move the whole model to the CPU; one table per row compiles for 7 minutes and returns garbage.
- Linear int4 (`constexpr_blockwise_shift_scale`) runs on the Neural Engine only with one scale per row; scales per 32 or 128 inputs (llama.cpp's Q4_0 layout) go to the CPU, so Q4_0 cannot be repacked for it.
- Matrix products have an absolute error floor of about 2.5e-4 on their outputs (2048 long dot products, weights of rms 0.02): inputs of rms 1 come back 0.03% off, inputs of rms 0.03 come back 0.95% off, the same absolute error. LFM3's gated projections (`out_proj`, `o_proj`, `w2`) produce outputs of rms 0.008 and lost 4 to 12% per layer. Scaling the input up and the weights down does not help, since the products stay the same. The converter instead multiplies those weights by 64 (`-post-scale`) and undoes it with an exact elementwise multiply inside the following sandwich norm, where the compiler cannot fold it back; the per layer error drops to 1.4 to 2.7%.
- Names of program values must not contain dots: the loader crashes.
- Elementwise operations, the depthwise convolution and the norms are accurate; the loss is in the matrix products.

### Choosing the table values

The table of a 4 bit palette can be fitted in several ways, and the one with the lowest weight error is not the best for the model. On LFM3-1.7B (24 token prompt, KL against the bf16 model on the CPU, 16 x 64 blocks):

| fitting | weight rel RMS error | KL (nats) |
|---|---|---|
| k-means from quantiles (lowest squared error) | 0.113 | 3.68 |
| uniform levels between min and max | 0.138 | 1.07 |
| k-means from the uniform levels (`-quant kmeans-minmax`, default) | 0.117 | 0.54 |

Squared error treats the many small weights and the few large ones alike; the network depends on the large ones, which the uniform start keeps while the k-means passes refine the bulk. With the whole row tables the Neural Engine needs, the default fitting gives KL 0.95 on the CPU and 0.89 on the Neural Engine with `-post-scale 64`. That is still coarse (argmax agreement 9 of 24); keeping either the MLP or the attention/convolution projections in fp16 halves it, so the loss is spread over the model rather than caused by one tensor. Activation aware quantization or 6 bit tables are the next candidates.
