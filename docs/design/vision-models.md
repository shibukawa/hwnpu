# Vision models

[日本語](vision-models.ja.md)

Models that read images (not image generation, see [image-models.md](image-models.md)): detection, pose, OCR, segmentation, depth, faces, labels and embeddings. All run from `hwnpu-image`. Most are distributed only as ONNX and go through one shared path (`internal/onnx`).

## Models and commands

| Command | Use (models.yaml) | Models (pick by machine memory) | Source | License |
|---|---|---|---|---|
| `detect` | `detect` | RTMDet-tiny, RTMDet-s (8 GB), RTMDet-m (16 GB) | ONNX, `bukuroo/RTMDet-ONNX` (MMDeploy export of MMDetection's COCO weights; box AP 41.1 / 44.6 / 49.1) | Apache-2.0 |
| `pose` | `pose` | RTMO-s (8 GB) | ONNX, `Xenova/RTMO-s` (9.9M, 17 COCO keypoints, no person detector) | Apache-2.0 |
| `ocr` | `ocr` | PP-OCRv6-small (8 GB) | ONNX, `PaddlePaddle/PP-OCRv6_small_rec_onnx` + detector in `with: {detector: ...}` (50 languages, 18708 characters with kana) | Apache-2.0 |
| `face-detect` | `face-detect` | YuNet (8 GB) | ONNX, `opencv/face_detection_yunet` (2023mar) | MIT |
| `segment` | `segment` | EfficientSAM-Ti (8 GB), EfficientSAM-S (16 GB) | the authors' ONNX encoder and decoder, `yunyangx/EfficientSAM` | Apache-2.0 |
| `depth` | `depth` | DA3-SMALL (8 GB), DA3-BASE (16 GB), DA3MONO-LARGE (24 GB); also DA3METRIC-LARGE, moge-2-vits / vitb / vitl | safetensors / `model.pt` checkpoints | Apache-2.0 (DA3), MIT (MoGe-2) |
| `label`, `embed` | `label` | siglip2-base-patch16-224 (8 GB), siglip2-large-patch16-256 (16 GB) | safetensors checkpoint | Apache-2.0 |

- Every model is fetched by name on first use, pinned by sha256 in [models.yaml](../../models.yaml). `with:` lets one model read a second repository (`chat.Model.With`, hashes under `detector/FILE`).
- Output: JSON lines (boxes, labels, text, keypoints). `--draw` (face-detect) writes a picture; `--repeat` times it. One picture at a time, no streaming.
- `depth` writes `NAME.depth.png` (8-bit inverse depth at the input's size); `--raw` float32 (MoGe-2 also points), `--sky` sky mask / confidence (DA3-SMALL/BASE) / mask (MoGe-2), `--normal` (MoGe-2), `--focal` / `--fov` (meters for DA3METRIC; MoGe-2 with `--fov` solves only the z shift), `--export-size`, `--ref` and `--ply` for multi-view exports.
- `label`: zero-shot over candidates in any language (`-l`), else a built-in list of about 440 labels. Each label is an independent sigmoid, not a softmax. `embed --compare` compares pictures and texts.

## Where each runs

| | Apple (Neural Engine) | AMD (XDNA 2) | Intel NPU | CPU |
|---|---|---|---|---|
| RTMDet-tiny / s | Core AI | CPU (`device: {amd: cpu}`) | OpenVINO IR | ONNX Runtime, else milcpu |
| RTMDet-m | Core AI | NPU, IRON export made on first run | OpenVINO IR | same |
| RTMO-s | Core AI | NPU, IRON export made on first run | OpenVINO IR | same |
| PP-OCRv6-small | Core AI | CPU (`device: {amd: cpu}`; no IRON form) | OpenVINO IR | same |
| YuNet | Core AI | CPU by default; `--device npu` runs `iron_yunet.py`'s export | OpenVINO IR | ONNX Runtime, else milcpu |
| EfficientSAM encoder | Core AI | NPU, IRON export made on first run | OpenVINO IR | ONNX Runtime, else hwnpu's evaluator |
| EfficientSAM decoder | host evaluator | host evaluator | host evaluator | host evaluator |
| DA3, MoGe-2 | Core AI | NPU (`iron_da3.py`, `iron_moge.py` exports) | OpenVINO IR | — |
| SigLIP 2 | Core AI | NPU (`iron_siglip2.py` export) | no form yet | — |

- **Apple: Core AI only.** `hwnpu-image` on a Mac needs macOS 27; without Core AI it fails (`errNoCoreAI`), no fallback to Core ML. `--backend coreml` and `HWNPU_BACKEND=coreml` are read as Core AI. Programs are written straight as `.aimodel` from ONNX or the checkpoint. See [apple-coreai.md](apple-coreai.md).
- `--cpu` (detect, pose, ocr; label/embed too) runs the Core ML `.mlpackage` on the CPU interpreter (`milcpu`). It checks a conversion; it is slow.
- **Intel** (checked only through OpenVINO's CPU plugin, `HWNPU_BACKEND=openvino`, on a Ryzen machine): pose, detect, ocr, face-detect, depth (DA3, MoGe-2), segment give the same results as the CPU or AMD paths. Whether the NPU compiler accepts DepthToSpace, ConvolutionBackpropData, Clamp, MaxPool and EfficientSAM's 4096-token attention is not checked. See [intel.md](intel.md).
- **Placement**: models.yaml `device` per vendor (`{amd: cpu}`; an unnamed vendor means npu). Override order: `--device`, `HWNPU_DEVICE`, models.yaml (`chat.ChooseDevice`, which also returns the reason). `chat.ConvertedDevice` puts ONNX models with no NPU form on the CPU. A model with no form for this NPU falls back and says so; only an explicit `--device npu` is an error. `hwnpu-image list` shows RUNS FROM (`CPU (ONNX Runtime)` or the export).
- `hwnpu-image` holds the Neural Engine lock (`llm.OpenDevice`); placement is strict (it says why when it cannot verify). `aicheck run` and `probe` take the lock too (`llm.LockNPU`). See [apple-neural-engine.md](apple-neural-engine.md).

### CPU path

- `internal/onnxrt`: ONNX Runtime's shared library through purego (no cgo). Needs OrtApi 16 or newer, calls by table index. 1 thread by default, no spinning (`session.*.allow_spinning = 0`). Library from `HWNPU_ONNXRUNTIME` or the system; never downloaded (the user decides how native libraries are shipped). `HWNPU_ONNXRUNTIME=none` disables it.
- `onnxrt.Cut` cuts the ONNX graph between two values; `onnxrt.Program` returns the same inputs/outputs as `onnx.Program` (rows for `Flat`), so the host code (box decoding, keypoints, CTC) is unchanged.
- Without ONNX Runtime: milcpu with `internal/sgemm` (GotoBLAS-style float32 GEMM; AVX2 / AVX-512 on amd64, NEON 6×16 on arm64, 106 GFLOPS on one M3 thread; im2col packed while filling B) and SIMD exp / sigmoid / SiLU / erf (`internal/f32`). `--threads` (default 1: the least CPU time and energy).

| 1 thread, Ryzen AI 9 365 | ONNX Runtime | milcpu |
|---|---|---|
| RTMDet-m (640²) | 1.37 s | 2.25 s (4 threads 1.0 s) |
| RTMDet-tiny | — | 0.70 s |
| RTMO-s | 0.45 s | 0.89 s |
| PP-OCRv6 small (3 lines) | 1.8 s | 4.5 s (4 threads 2.6 s) |

## ONNX-only models (`internal/onnx`)

One tool instead of porting each model by hand. Graphs mix a static part (CNN, encoder) with a dynamic tail (TopK, NMS, per-person classifiers).

- Reads ONNX without dependencies (protowire: nodes, attributes, initializers, I/O).
- Host evaluator (float64, numpy broadcasting): shape ops, TopK, NonMaxSuppression, Conv, Pad (constant, reflect), MatMul, Softmax, Reduce*, If, ScatterND, ConvTranspose, InstanceNormalization, LayerNormalization, Resize (nearest, linear, cubic), OneHot. Large matmuls (≥ 2^22 MACs) use sgemm float32.
- `onnx.binop` loops without index tables when one side has the output's shape and the other repeats along its trailing axes (`TestBinopRepeats`).
- Lowering to MIL (`Program`): values fixed by shapes and constants (Shape / Gather / Concat chains) are folded on the host; only input-dependent values become program ops. Pow 2 → mul, Neg → mul −1, Clip → maximum/minimum, MatMul by a constant (+ Add) → one linear with bias. HardSigmoid → clip, BatchNormalization → per-channel mul/add, GlobalAveragePool → reduce_mean, AveragePool → depthwise conv of weight 1/(kh kw), ConvTranspose with kernel = stride → 1×1 conv + `pixel_shuffle`, `auto_pad` SAME_UPPER/LOWER, Resize to its own size, LayerNormalization.
- `Graph.Cut`: the named dynamic ops and everything after them become the tail; the static values the tail reads become program outputs. `EvalShapes` evaluates the tail on the host from those outputs.
- Outputs of different shapes are joined as `(1, rows, width)` with rows = the common divisor of all sizes, so no axis passes 16384 (`Flat`; `FlatRows` / `SplitFlat` split it without the graph). Concatenating on the last axis broke outputs on the Neural Engine.
- Max pools of window 9 and 13 (stride 1) become chains of window 5 (same max); the large ones left the Neural Engine on M1.
- Initializer names starting with a digit (YuNet's `420`) get a `w_` prefix; Core ML rejects them.
- `onnx.Program.ExpSigmoid`: write Sigmoid as `1 / (1 + exp(-x))` and SiLU as `x / (1 + exp(-x))` (see pitfalls). On for RTMDet and PP-OCR; off for RTMO, YuNet (and Basic Pitch).

## ONNX → IRON converter on AMD (`tools/amd/iron_onnxcnn.py`)

Turns an ONNX CNN (CSPNeXt, YOLOX-type) into one IRON graph. Switching operators costs about 250 µs on AMD's NPU, so one convolution must be one GEMM. Details in [../amd-npu-models.md](../amd-npu-models.md); shared IRON parts in [amd.md](amd.md).

- Tensors: channels-last grid padded by 2 pixels with zeros (`G`), channels to multiples of 64. The last two channels are "ones" (16 inside the picture, 0 in the pad); biases ride them, SiLU is the GEMM epilogue.
- Ops: 1×1 (one GEMM over the grid), k×k stride 1 (`iron_gridgemm.GridGEMM` when k·Cp ≤ 512, else im2col copies), stride 2 (3 im2col copies), depthwise (block-diagonal weights), channel attention (GEMM with a ones row, GEMV fc, clamp, row multiply), SPPF max pool (windows 2/4/8 combined; max(a, b) = b + ReLU(a − b), pad −1), 2× nearest upsampling (4 phase copies), add, concat, Focus (as a 6×6 stride-2 conv), Split, grouped conv, AIFI transformer layer (32-dim heads padded to 64 for IRON's MHA, scale 1/√32, LayerNorm, erf GELU, HIGHEST).
- The box head's exp and strides stay on the host (`detect.NPUOutputs`, kept in `heads.json`). Grid state is reused by the next grid of the same shape (565 MiB pinned instead of 1451).
- Export I/O and run are shared by detect and pose: `internal/vision/ironcnn`. Made on first run with `chat.RunIRON` (`~/models/rtmdet-m.amd-npu2`, `rtmo-s.amd-npu2`); without IRON, the CPU.
- Rules learned (kept in the exporter):
  - The ones channel must be exactly 16: silu(16) is 16 in bf16 and rounding brings small drifts back. A value of 1 (silu(z) = 1) lost an ulp a layer (stage 2 at cos 0.904).
  - Max pool pad −1, not −30000: a − b then rounds to steps of 128 in bf16 and breaks the right and bottom edges.
  - Depthwise channels sorted by size so the 8 channels that share a bfp16 exponent are alike.
  - Biases split into high and low bf16 on two ones channels.

## Per model

### RTMDet (`hwnpu-image detect`)

- Program: up to the head convolutions (class logits and 4 box distances per level). The graph then transposes, sigmoids, decodes, takes top 5000 and runs per-class NMS on the host. The host decodes only cells whose logit beats the min score's logit (sigmoid is monotonic). The Neural Engine has no transpose. Input fit into 640×640.
- `detect.Open` gets the class count from output sizes and strides (no second ONNX read; m's weights were 529 MB as float64).
- Apple (M1): tiny 13–15 ms a picture (network 7.0 ms), s 15–17 ms (9.3), m 24–26 ms (18.2); resize 5–7 ms; footprint 56 / 73 / 141 MB, Neural Engine 11 / 21 / 56 MB.
- Accuracy (s vs ONNX Runtime): score diff 0.063 / 0.042, boxes 5.5 / 6.7 px (score ≥ 0.1); objects ≥ 0.3 the same.
- AMD: RTMDet-m 0.33 s a picture, 1.86 J; CPU ONNX Runtime 1 thread 0.54 s and 8.2–10.7 J (balanced), milcpu 1.0 s and 16 J. Same objects as ONNX Runtime, scores within 0.013, boxes 1.3–7.6 px. 565 MiB pinned, 98 MiB process. tiny and s stay on the CPU (Not adopted).

### RTMO-s (`hwnpu-image pose`)

- RGB 0–255 stretched to 640×640. All 370 ops up to 2000 anchors' scores, boxes, pose features and visibility on the NPU; TopK, NMS and the per-person coordinate classifier (DCC, GAU, sin/cos embedding) on the host evaluator.
- Apple: M1 about 100 ms a picture (network 9–100 ms, host 36–43 ms), footprint 232 MB; M3 Core AI 87–88 ms, of which 70 ms host. Versus the ONNX reference: boxes 1.6 px, visible keypoints 3.4 px (as much as fp16 weights alone).
- AMD: up to the head convs on the NPU (`ironcnn.LastConvs`), 8 outputs to the host. HIGH with SiLU as an op: keypoints within 2 px, scores within 0.03, 0.22 s a picture. 327 MiB pinned, 215 MiB process. MEDIUM (0.15 s) moved keypoint scores by 0.13 and low-confidence boxes by 97 px.

### PP-OCRv6 small (`hwnpu-image ocr`)

- Detector at 960×960, 1280×736 or 736×1280 (the one that shrinks least); image top-left, rest zero after normalization. 13 ms on M1. Versus PaddleOCR 3's default (no shrinking, multiples of 32): 94 of 98 lines of a paper page the same (CER 0.12 %).
- Recognizer: logits (8 lines × 160 steps × 18710 classes) are the output; the softmax is computed by `Decode` only near the max. The last linear is split into two of 9355 classes stacked as (…, 2, 9355) (same order as (…, 18710)); its bias is folded in. All on the NPU.
- Output read as fp16 (`runner.F16`): max by comparing fp16 bits as integers (`sortKey`), only steps inside the line's width, probabilities only for values within 16 of the max. Decoding runs during the next call on another goroutine.
- **The recognizer's reading depends on the zero padding after a line** (SVTR's attention sees the whole width): "SUB TOTAL" (230 px) reads right at width 320, as "SUBTOTAL" at 640. Like PaddleOCR, lines are batched with the widest line's width. Widths `ocr.Widths` = 320, 480, 640, 960, 1280, 1920, 2560 (each ≤ 1.5× the last); up to 8 lines and 5120 columns a call (1280: 4 lines, 2560: 2). Each width's program is converted on first use (5–20 s compile).
- M1: 4 Japanese lines 58 ms; a paper page (1224×1584, 98 lines) 488 ms (detector 13, boxes 16, crops 34, recognition 342 ms in 30 calls); a receipt (13 lines) 41 ms. Footprint up to 425 MB (7 widths and their 12–24 MB outputs).
- Versus PaddleOCR on ONNX Runtime (`tools/ref/ref_ppocr.py --fit 960x960`, `tools/ref/compare_ocr.py`): paper 93/98, receipt 13/13, Japanese 3/4 lines identical (CER 0.20 %, 0, 3.9 %). The NPU and the CPU interpreter give the same lines.
- Line order is PaddleOCR's `sorted_boxes` (top down, left first within 10 px), so two-column pages interleave. "〜" (U+301C) is in the dictionary but never read (the model, PaddleOCR too).
- AMD: CPU (ONNX Runtime 1.8 s for 3 lines).

### YuNet (`hwnpu-image face-detect`)

- Apple / Intel: the whole ONNX graph as one `internal/onnx` program (12 outputs joined). Apple: 640×480 in 3.4 ms (M1), M3 Core AI 4.2–4.4 ms. Boxes and landmarks within 0.3 px of OpenCV's FaceDetectorYN.
- AMD (`tools/amd/iron_yunet.py`): 1×1 + depthwise 3×3 fused into one 3×3 conv, biases on a ones channel, ReLU as ProjectionGEMM's clamp (`ReluGEMM`); pixels divided by 255 and the ones moved to im2col's last 8 columns (bfp16 shared exponent erased them: cos 0.69). Opened with `iron.GraphOptions.KeepRuns` (each command's run kept: host CPU 1.6 → 1.1 ms; its control code stays in the 64 MiB device heap, so not for large graphs). Matches FaceDetectorYN.
- AMD energy at 320×256: NPU 29 ms and 127 mJ, CPU (ONNX Runtime 1 thread) 2.7 ms and 7.7 mJ. Operator switches (~250 µs each) dominate. Hence `device: {amd: cpu}`. This is about AMD's IRON only; Apple and Intel submit the whole graph, so they keep the NPU. Do not carry one vendor's result to another.
- Without ONNX Runtime the CPU runs the ONNX graph converted to Core ML on `milcpu` (it went to AMD's NPU before). The store records which (`runtime: hwnpu`), and the role's `Usable` prepares it again when ONNX Runtime comes or goes. Ryzen AI 9 365, 320×256, `--repeat 2000`, 1 thread, package power from `gpu_metrics` less idle: `milcpu` 2.32 ms and about 30 mJ a picture (12.58 ms and 243 mJ before the kernels below), ONNX Runtime 1.32 2.15 ms and about 20 mJ, the NPU 30.7 ms and about 184 mJ. Same faces as ONNX Runtime (score 0.934 both). This way of measuring gives ONNX Runtime 20 mJ, not the 7.7 above: compare within one method.
- `milcpu` for these CNNs (`internal/milcpu`, `internal/f32`): a run's buffers are kept for the next values (`pool.go`: the collector had 40 % of the CPU and woke every core); depthwise 3×3 of stride 1 from three padded rows a call (`f32.Depthwise3`); other small convolutions (taps ≤ 64: YuNet's first, stride 2) as weighted sums of input rows cut into phases (output channels in blocks of 8, then 4, then single ones: `f32.WeightedRows8`, `WeightedRows4`, `WeightedRows`); 1×1 convolutions (≤ 256 input channels) the same way on blocks of packed columns instead of sgemm (planes of 4 KB multiples put every row's column in one set of L1: packed rows 16 floats apart); a conv followed only by a ReLU computes it; 2×2 max pool, ReLU, nearest 2× upsampling in their own loops. YuNet's program 11.6 → 1.85 ms (1 thread). The kernels are AVX2/FMA on amd64 and NEON on arm64 (with `f32.Axpy`, and exp, sigmoid, SiLU and `f32.GLU` in `f32_arm64.s`: Cephes' expf in the AVX2 loop's steps). On NEON a block of 8 is eight outputs of 8 columns in 16 of the 32 registers (a row's two vectors and the tap's two of weights read for 16 FMAs). On amd64 it is two passes of 4: eight outputs a pass (8 columns of AVX2, whose 16 registers it fills, or 32 of AVX-512) were no faster on Zen 5, where `WeightedRows4` already runs FMAs at the core's peak (about 145 GFLOPS); amd64's results are bit-identical to the 4-channel blocks'. The NEON kernels were checked under qemu-user (the tests, YuNet's faces, 22 Core ML packages against the Go loops) and on an M2 (this Mac's 15 packages: within 7e-6 of the largest value against the Go loops). M2, 1 thread: YuNet 320×256 24.0 → 2.7 ms, 640×480 89.6 → 9.0 ms (`face-detect --device cpu` 92.7 → 10.6 ms a picture), SwiftF0 1.09 → 0.62 s, Supertonic's encoder 105 → 24 ms. `dw3NEON` starts each chain with an FMUL and adds the bias last (dw3AVX2's order): begun from a copy of the bias or a zeroed register, the first FMLA waited for it, and YuNet 640×480 took 10.0 ms instead of 9.0. The 8-channel blocks are what make the 1×1 convolutions faster than 4 on the M2 (`wrows8NEON` 1.66 s against `wrows4NEON` 1.99 s of a profile).

### EfficientSAM (`hwnpu-image segment`)

- Encoder (1024×1024 ViT + neck) on the NPU; prompt decoder (point/box encoding, mask decoding) on the host evaluator per prompt (If, ScatterND, shapes that change with the number of points). Host resizes to 1024×1024 with ONNX's bilinear (half_pixel) first (`segment.OpenProgram`).
- AMD (`tools/amd/iron_efficientsam.py`; 64-dim heads fit IRON's MHA): encoder Ti 0.34 s, S 0.48 s (CPU ONNX Runtime 1.9 s, 4.5 s). NPU vs CPU mask IoU 0.93–1.0; the top mask written by default ≥ 0.989. Ti 118 MiB pinned / 274 MiB process, S 158 / 267 MiB.
- Apple: encoder on the Neural Engine by default (`--device cpu` or `HWNPU_DEVICE=cpu` for the CPU). M3: S 0.35 s (Core ML) / 0.43 s (Core AI), Ti 0.15 / 0.19 s; all on the Neural Engine; mask IoU vs CPU 0.9994 (S), 0.9986 (Ti). First compile Ti 13 s, S 22 s. Footprint 0.22–0.27 GB. A picture 0.72 s on M3 (decoder 0.28 s).
- Decoder on the host: 0.28 s on M3, 0.6–0.7 s on AMD. Matches ONNX's reference evaluator (IoU prediction 1e-6, mask sign on every pixel).
- CPU encoder: ONNX Runtime, else hwnpu's evaluator (Ti 16–30 s, S 33 s; no external dependency).

### Depth Anything 3 (`hwnpu-image depth`)

- DA3MONO-LARGE (0.33B, DINOv2 ViT-L + DPT; relative depth and sky), DA3METRIC-LARGE (same; meters = depth × focal px / 300, needs `--focal` or `--fov`), DA3-SMALL / BASE (multi-view models run as one view; depth and confidence).
- Sizes: the official sizes (long side 504, short side a multiple of 14) for 9 aspect ratios; the host uses the picture's official size if present, else the nearest aspect ratio (`depth.Official`, `Choose`). Stretching to a near size cost 6–12 % against the official output, more than the NPU's 1 %.
- Host (`internal/vision/depth`): resize (bicubic), patches, exp of depth, sky (≥ 0.3) filled with the 99th percentile of the rest, confidence exp + 1, bilinear back to the input size.
- AMD (`iron_da3.py`): `Grid` with rows padded to 16, stride-2 3×3, transposed conv (GEMM + Copy to pixels), bilinear as two GEMMs with interpolation matrices, 2-D RoPE by permuting q/k dims, head channels padded to 64. 504×378: SMALL 0.26 s, LARGE 0.7 s; pinned 0.49–1.39 GiB; depth cosine 0.9993–0.9999 vs the official CPU fp32.
- Multi-view (`iron_da3.py --views N`, AMD only): packed view tokens, windowed in-view attention, cross-view attention over all rows; reference view choice and camera MLP on the host (`depth.SaddleBalanced`); writes `cameras.json` and `--ply`. 3 views 1.1 s, cosine 0.9998. Near-identical pictures can pick a different reference view than the official code (`--ref`).
- Apple (`internal/vision/depth/mil`): transposed conv as 1×1 + `pixel_shuffle`, bilinear as matrix products. One export, a `WxH/` folder per size converted the first time a picture needs it (seconds). M3 Core AI: DA3-BASE 504² ViT 106 ms, head 16 ms; depth cosine 1.000000 against Core ML. DA3-SMALL 1.000000, DA3MONO-LARGE 0.999993 vs the official module (sky 0.9998).

### Depth models compared (Ryzen AI 9 365, through `ai/vision`)

Three pictures (one 1024×672, two 1024×1024), while another session used the NPU. No ground truth: the
look is from grey maps side by side.

| Model | A picture (from the 2nd) | First | NPU memory | Process | Depth size | Export | Look |
|---|---|---|---|---|---|---|---|
| DA3-SMALL | 0.31 s | 0.57 s | 0.48 GiB | 0.17 GiB | 504 | 1.0 GB | soft edges; thin branches and hair vanish |
| DA3-BASE | 0.43 s | 0.86 s | 0.74 GiB | 0.17 GiB | 504 | 2.3 GB | a little sharper than SMALL |
| DA3MONO-LARGE | 0.85 s | 1.63 s | 1.36 GiB | 0.21 GiB | 504 | 6.2 GB | sharp edges, background layers apart |
| DA3METRIC-LARGE | 0.85 s | 1.50 s | 1.36 GiB | 0.21 GiB | 504 | 6.2 GB | as MONO; fine grain in sky or flowers on some pictures |
| moge-2-vits | 2.52 s | 3.15 s | 1.94 GiB | 0.44 GiB | 1024 | 2.4 GB | fine edges, background layers close together |
| moge-2-vitb | 2.68 s | 3.26 s | 2.10 GiB | 0.42 GiB | 1024 | 3.6 GB | fine edges, background layers apart |
| moge-2-vitl | 3.41 s | 4.17 s | 2.62 GiB | 0.50 GiB | 1024 | 7.7 GB | finest; cuts the sky out as farthest |

- The `depth` use stays: SMALL from 8 GB, BASE from 16 GB, MONO-LARGE from 24 GB.
- Nearness (the inverse of depth) leaves everything behind a close object near 0; `samples/parallax`
  spreads it with log(1 + k·n) / log(1 + k), k = 30 by default.

### MoGe-2 (`hwnpu-image depth -m moge`)

- Metric depth, points, normals and a mask without knowing the focal length (DA3METRIC needs it). DINOv2 + ConvStack neck + 3 heads (points, normals, mask). No normalization (`res_block_*_norm` none). All 3×3 pad by replicating edges.
- `model.pt` is read without torch (`internal/torchzip`; only dict and tensor reconstruction from the zip's pickle). Post-processing in Go (`internal/vision/moge`): focal length and z shift by one-variable Levenberg–Marquardt on a 64×64 subsample (`TestRecoverFocalShift`).
- AMD (`iron_moge.py`): replicate pad by `Grid(rep=True)` and `fill_pads`; the 3 heads are one program with 3 weight banks (`SwapVariants`): keeping all 3 loaded would pin 3.9 GiB. 812×602: vits ~2.7 s, vitl ~3.6 s; depth cosine 0.9999, mean relative error 0.7–1.2 %. Most time is the 16× resolution heads.
- Apple (`internal/vision/moge/mil`): ViT from depth/mil; neck and heads one program (replicate as `pad` + valid conv; Core ML wants `pad` even with `pad_type` valid). ViT-S: head outputs cosine ≥ 0.99993, scale logit 1.6885 vs 1.6853; 0.3–0.5 s a picture.
- The mask head's last `pixel_shuffle` (256 ch → 64×400×400) on a 700×700 `dense` left the Neural Engine on M1 Core AI (`empty reset layer candidates in SS cost model`, sent to GPU). It is split into row bands (`mil.deconv2`, `shuffleRows` up to 256 rows); same values.

### SigLIP 2 (`hwnpu-image label`, `embed`)

- Host (`internal/vision/siglip`): resize (PIL bilinear) to [−1, 1], patches; text lower-cased (trained so; Transformers' tokenizer does not), Gemma tokenizer, cut to 63 tokens + `<eos>`, padded to 64, no mask; template `this is a photo of {}.`. Embeddings normalized; probability = sigmoid(cos × exp(logit_scale) + bias).
- Label embeddings are cached in the user cache (`hwnpu/siglip/NAME-HASH.f32`); a label list is computed once.
- Text tower: 8 texts a call; rows token-major so 8 texts become one attention of 8 × heads heads. Missing texts are filled with copies of the first (zero rows make fp16 LayerNorm 0/0).
- AMD (`iron_siglip2.py`, `--precision HIGHEST`, `--gelu erf`): exact GELU instead of IRON's table tanh GELU (5× bf16 rounding, invisible against IRON's reference). Image cosine 0.9998–0.9999 vs Transformers float32. Ryzen AI 9 365: base 84 ms a picture, 131 ms for 8 texts, 760 MiB; large 353 ms / 491 ms, 1579 MiB.
- Apple (`internal/vision/siglip/mil`, `siglip.Prepare`): converted in Go on first run from `config.json`, `tokenizer.json`, `model.safetensors` to `NAME-v1` (vision and text programs, `text/embed.bin` fp16 token table read by the host, `model.json`); base 1.9 s, large 3.5 s. Both towers fp16. LayerNorm divides each row by its max first (`ScaledLayerNorm`); tanh GELU written with exp. The probe query is a constant placed on 8 rows (a one-row part leaves the Neural Engine).
- M3: image cosine ≥ 0.99986, text ≥ 0.999992 (Core AI); label probability diff ≤ 0.007. base 7.3–7.9 ms a picture, 444 default labels 1.7–2.0 s once; large 33–36 ms, 4.0–4.8 s. Neural Engine: base 357 MB (Core ML) / 476 MB (Core AI), large 1240 / 1556 MB. models.yaml memory: Apple 0.9 / 3.3 GB, AMD 0.8 / 1.7 GB.

## Neural Engine pitfalls for these models

| Pitfall | Effect | Fix |
|---|---|---|
| The Neural Engine's `sigmoid` and `tanh` drift through a deep CNN (80 SiLUs in RTMDet-s) | box 98 px, score 0.35 off, an object lost (CPU fp16: 7.5 px) | `ExpSigmoid`: 5.5 px, score 0.063. Costs 2 ops: s 6.6 → 9.3 ms, m 12.2 → 18.2 ms; accuracy wins. `0.5 x (1 + tanh(x/2))` also drifts |
| Any op on a tensor wider than 16384 on an axis, even the output identity | PP-OCR's 18710-class softmax, bias add and output on the CPU | logits as output, linear split in two, bias folded |
| Core AI on the Neural Engine computes `divide_v1` by a constant wrong (×√2 instead of ÷√2, plus a per-channel offset) | PP-OCR detector cosine 0.05, recognizer −0.42 | `ailower` writes `x × (1/c)`; only ONNX graphs write such divides, so other programs' bytes are unchanged. `HWNPU_DEV=ailower.divide=1` restores the divide for checks |
| Core AI's CPU sums `reduce_mean` in fp32 then stores fp16 before 1/n | −inf where \|mean\| > 65504/n (an SE block of 57600 elements) | only matters on Core AI's CPU (`HWNPU_DEV=coreai.units=cpu`); not fixed, it would change every program's hash |
| Core AI `max_pool` | did not lower | `pad_v1` (−inf) + `primitive.max_pool_2d_v1`, as coreai-torch writes it |
| Core AI `conv_transpose` with MIL's pads (may be negative) | — | full output with padding 0, `slice_v1`, `pad_v1` with 0, then bias |
| `pad` replicate / reflect | — | `#aicode.padding_mode_v1<replicate>` / `<reflect>` |

- Core AI and Core ML match when both caches are warm (M3, three new processes each: Core AI 0.01–0.05 s faster per command). Core ML's cache (~4 GiB) evicts; Core AI keeps its programs until the disk runs low. See [apple-coreai.md](apple-coreai.md).
- Chips checked on Core AI: M3, every part on the Neural Engine, cosine vs milcpu ≥ 0.9997; M1, every command with its defaults gives the same results as Core ML (after the MoGe-2 `dense` fix above).
- Per call on M3, Core AI vs Core ML: PP-OCR detector 13.0 / 9.7 ms, recognizer 8×320 5.1 / 4.0 ms, RTMDet-m 10.2 / 13.5 ms, RTMO-s 8.0 / 4.9 ms, YuNet 1.2 / 0.9 ms, DA3-BASE ViT 106 / 103 ms. Core AI first compile 1.6–38 s per part (RTMO 36–38 s). Neural Engine memory is higher on Core AI (OCR 27 → 40 MB, RTMDet 56 → 89 MB).
- Checking tools: `aicheck run -cpu PKG` (Core AI) and `-coreml` vs milcpu; `HWNPU_DEV=ailower.cut=N` cuts after op N and compares that value; `AICHECK_WORST=1` prints the worst elements.
- Host resize (`lfm3vl.ResizeRGB`, PIL bilinear) is row-parallel without per-pixel closures: 5–7 ms instead of 37–55 ms, same bytes (`TestResizeRGBSameBytes`). Rounding to a byte uses the built-in `min` / `max`: `math.Max` and `math.Min` are calls (for NaN and signed zeros, which the sums are not) and took a third of the CPU. `lfm3vl.RGB` drops alpha a row at a time on the cores. M2, 512² to 640²: resize 4.2 → 1.2 ms, RTMDet-m's `Pixels` 5.4 → 2.2 ms; bytes the same as before on 140 resizes (bilinear and bicubic, RGBA and NRGBA, up and down).

## Exports on AMD and sharing

- detect, pose, segment (`amdFirstRunExport`): local export used as is, else from the source layer, else made with IRON and written back (`chat.SharedExport`). A hand-made export (no `entry.json`) is described with `ExportKind(tool)` and `PinnedRecipe(m, tool, version)` and placed in the source on first use (`chat.ShareMadeHere`). Not placed: exports with `entry.json`, outside `AMDRoot` (`~/models`), or without a key (definitions without sha256). See [distribution.md](distribution.md).
- depth (`~/models/NAME.amd-npu2`, lower case) and SigLIP 2 exports come from `tools/amd/iron_da3.py`, `iron_moge.py`, `iron_siglip2.py`.

## Open

- Power not measured: Apple's vision models (`hwnpu-dev powerbench` needs sudo), RTMO and EfficientSAM on AMD, SigLIP 2.
- Intel's real NPU for every model; SigLIP 2 has no Intel form (same MIL could be lowered to IR, unchecked).
- M2 / M4 on Core AI (docs/handoff-coreai-vision.md). On M1, EfficientSAM's encoder attention (4096 × 4096 × heads) passes 64 MiB.
- Why the Core AI divide error does not reproduce in a 3-op program; whether it differs by chip.
- Core AI CPU `reduce_mean` overflow.
- RTMDet-tiny and m checked against ONNX Runtime only through s; RTMO-m / l; PP-OCRv6 medium; 180°-rotated lines (no PP-LCNet orientation classifier); vertical text; layout analysis (PP-StructureV3).
- `ExpSigmoid` for RTMO and YuNet (likely better: RTMO is a SiLU CSPNeXt).
- DA3METRIC-LARGE and DA3-BASE not run on Apple's Neural Engine against the official module; MoGe-2 ViT-B / L not compared on Apple. Multi-view DA3 is AMD only; ray pose head (`use_ray_pose`) and given cameras (`cam_enc`) unsupported.
- MoGe-2's heads are heavy on AMD (16× resolution, three heads each over the same neck output).
- SigLIP 2: so400m (72-dim heads do not fit IRON's 64-multiple MHA), NaFlex, SafeSearch-like thresholds, images in `/v1/embeddings`, dense patch features.

## Not adopted

- RTMDet-tiny / s on AMD's NPU: BN-folded weights and biases cancel to outputs ~50× smaller; MEDIUM's bfp16 (A and B), the epilogue SiLU (vtanh returns x for |x| ≤ 0.5) and near-constant channels push tiny's scores 0.2 off and make s lose objects. All-HIGH tiny is 0.25–0.30 s, slower than the CPU (0.21 s). CPU there.
- Channel reordering for bfp16 in `iron_onnxcnn.py`: helped tiny (0.2 → 0.07) but made RTMDet-m lose objects.
- Splitting weights into high and low bf16: no effect (the error is activations' bfp16 and SiLU).
- YuNet on AMD's NPU by default: 16× the CPU's energy.
- Text tower batch of 16 on AMD: 13.8 ms a text vs 16.4 ms at 8, but worse for few candidates; 8 kept.
- IRON's tanh GELU and MEDIUM matmuls for SigLIP 2: image cosine 0.9988; exact GELU + HIGHEST costs 9–22 % time.
- Running EfficientSAM's decoder on the Neural Engine: dynamic shapes (If, ScatterND, point count); the host was made faster instead.
- Sharing SigLIP label embeddings through the source layer: a few hundred KB, rebuilt in seconds.
- DA3NESTED, DA3-LARGE, DA3-GIANT: CC BY-NC.
- BiRefNet for cutouts: deformable conv; EfficientSAM (plain ViT) instead.
- A vision-language model (LFM3-VL) for boxes: tens of tokens per box, seconds a picture, misses small objects, coordinates drift. A one-stage detector gives all boxes in one pass; a VLM fits as a later step describing crops.
- Core ML for `hwnpu-image` on a Mac: Core AI only (Core ML kept for `--cpu` exports).
