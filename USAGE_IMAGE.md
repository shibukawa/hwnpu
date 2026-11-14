# Image commands (hwnpu-image)

[日本語](USAGE_IMAGE.ja.md)

`hwnpu-image` runs image generation (`generate`), depth estimation (`depth`), faces (`face-detect`), objects (`detect`), human poses (`pose`), text (`ocr`), labels and embeddings (`label`, `embed`) and cutouts (`segment`) on the NPU. `hwnpu-image list` shows the models to choose from.

## Image generation (Anima)

`go run ./cmd/hwnpu-image` draws a picture from a prompt with Anima (circlestone-labs/Anima: the DiT of Cosmos-Predict2 2B, Qwen3-0.6B and the VAE of Qwen-Image). Encoding the text, the DiT steps and decoding with the VAE are all IRON graphs on the NPU. The host does the tokenization, picking the embedding rows, reordering the patches, the Euler steps and the PNG. An export is made per image size and diffusion model, by hwnpu's `tools/amd/iron_anima.py` (4.0 GB at 1024 × 1024, with the DiT blocks' weights in 8 bits).

```bash
python tools/amd/iron_anima.py ~/models/Anima ~/models/anima-turbo-1024.amd-npu2 --size 1024
go run ./cmd/hwnpu-image generate "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms"
```

Anima has two diffusion models. Without `-m` it is Anima-Turbo (`anima-turbo`), which draws in 8 steps without CFG. `-m anima` is Anima-Aesthetic (tuned on high quality pictures), with CFG 4 (`--guidance`; the model card recommends 4 to 5). By default it runs 25 steps of DPM-Solver++ 2M, with CFG on the 17 noisiest steps only (`--sampler euler --steps 30 --cfg-sigmas 0,1` puts CFG on all 30 steps, as the model card does). It runs the DiT 42 times a picture, so it takes about 4.5 times as long as Turbo. The negative prompt is `--negative` (empty by default; the model card recommends `"worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"`, with no score tags for Aesthetic). `--negative` on Turbo without `--guidance` runs a weak CFG of 1.5 (the DiT twice a step). The export picks the diffusion model with `--dit`.

```bash
python tools/amd/iron_anima.py ~/models/Anima ~/models/anima-aesthetic-1024.amd-npu2 --size 1024 --dit anima-aesthetic-v1.1.safetensors
go run ./cmd/hwnpu-image generate -m anima "masterpiece, best quality, safe, 1girl, cherry blossoms" --negative "worst quality, low quality"
```

On AMD's NPU, the name of `-m` uses `~/models/NAME.amd-npu2` (`anima-turbo-1024.amd-npu2`, `anima-aesthetic-1024.amd-npu2` and so on). The model can be a name in models.yaml, a family (`anima-turbo`, `anima`, `sana`, `sana:1.6b`) or an export directory. With a family alone, the machine's memory picks the size (`sana` is SANA 1.5 1.6B from 16 GB, Sana 600M below that). The memory can be given, as in `-m sana:8g` (also `8G`, `8GB`, or `12g`, which is not a column of the table), and picks what a machine of that memory would. `hwnpu-image list` (`list image`, `list depth`, `list face-detect`) shows a table of the models, whether an AMD export is there, and what each family and use picks for each memory. It shows the same table when nothing matches `-m`. A model picked by name looks for `~/models/NAME.amd-npu2` (in lower case, and with `-1024` too). Without `-o` it writes `image.png` in the current directory (`image (2).png`, `image (3).png`, … if that exists) and opens it in the OS viewer (open on macOS, the file association on Windows, xdg-open on Linux).

On a Ryzen AI 9 365, a 1024 × 1024 picture takes about 44 seconds with Anima-Turbo's 8 steps (4.6 seconds a DiT step), and about 3 minutes 20 seconds with Anima-Aesthetic (CFG 4) (3.8 kJ a picture; Turbo is 715 J). Anima-Aesthetic runs 25 steps of DPM-Solver++ 2M instead of the model card's 30 Euler steps, and puts CFG on the noisy steps only (the picture is the same; 30 steps took 4 minutes 40 seconds and 4.8 kJ. `--sampler euler --steps 30 --cfg-sigmas 0,1` brings that back). The memory pinned to the NPU is at most 2.4 GiB (the VAE stage), the same for every model. The layout of `~/models/Anima` and the accuracy measurements are in section 11 of [docs/amd-npu-models.md](docs/amd-npu-models.md).

On the Mac's Neural Engine and Intel's NPU, as with hwnpu-chat, the first run downloads the checkpoint from Hugging Face (about 5.4 GB) and exports it to Core AI programs (on a Mac, macOS 27) or OpenVINO IR in Go alone (no Python is needed). It goes in `hwnpu/models/Anima` in the same user cache directory as hwnpu-chat (`--dir`), and an export is made per image size (`--size`) and weight format. The first run also compiles for the NPU while loading, which takes a few minutes at 1024 × 1024.

```bash
go run ./cmd/hwnpu-image generate "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms" -o out.png
```

On the Neural Engine of an M3 (16 GB), a 1024 × 1024 picture takes about 40 seconds in 8 steps, with at most 3.8 GB of process memory (3 seconds to load, 4.2 to 4.4 seconds a DiT step, 3.7 seconds for the VAE; the default weights are int8 for both the text and the DiT). On an M1 (16 GB), Anima-Turbo takes about 48 seconds a picture and Anima-Aesthetic about 3 minutes 55 seconds (5.6 seconds a DiT run; 2.6 GiB on the Neural Engine and at most 3.9 GB for the process as measured before 16.201). The records are in sections 16.85, 16.152, 16.160, 16.201 and 16.202 of [docs/design.md](docs/design.md). Intel's NPU is untested ([docs/handoff-anima-mac-intel.md](docs/handoff-anima-mac-intel.md)). To export by hand, use `hwnpu-image convert DIR --model OUT` (DIR laid out as `~/models/Anima`).

## Image generation (Sana)

`hwnpu-image generate -m sana` draws with NVlabs' Sana (a DiT with linear attention, Gemma 2 2B and the DC-AE VAE): Sana 600M and SANA 1.5 1.6B, with 20 steps and CFG (on the 14 noisiest steps only). It is meant for English prompts. On the Mac's Neural Engine it is the fastest of the image generation models (on an M1, about 17 seconds for Sana 600M and about 31 seconds for SANA 1.5, against about 58 seconds for Anima-Turbo), and it needs less memory on the Neural Engine (0.6 / 1.6 GiB; Anima-Turbo needs 2.6 GiB).

On AMD's NPU, give it an export. `tools/amd/iron_sana.py` makes the export from Efficient-Large-Model's diffusers repositories. Sana-Sprint, distilled to 2 steps, is left out, as its pictures are much worse than Sana's (section 16.148 of [docs/design.md](docs/design.md)).

```bash
python tools/amd/iron_sana.py ~/models/Sana_600M_1024px_diffusers ~/models/sana-600m-1024.amd-npu2
go run ./cmd/hwnpu-image generate -m sana:0.6b "a watercolor painting of a red fox in a snowy birch forest"
```

On a Ryzen AI 9 365, a 1024 × 1024 picture takes about 60 seconds with Sana 600M and about 2 minutes with SANA 1.5 1.6B. The memory pinned to the NPU is at most 2.3 GiB (Gemma 2's 8 bit weights). It expects English prompts. `-m sana` picks Sana 600M or SANA 1.5 1.6B by memory. `--negative` adds a negative prompt. For pictures that are not square, export at one of Sana's bin sizes (`iron_sana.py --width 1152 --height 896` and so on). The steps and the accuracy are in section 14 of [docs/amd-npu-models.md](docs/amd-npu-models.md). The licenses are Apache-2.0 and MIT for the DiT and the VAE, and the Gemma Terms of Use for Gemma 2.

On Apple's Neural Engine, Sana 600M and SANA 1.5 1.6B run with `hwnpu-image generate -m sana "..."` and the like. As with Anima, the first run downloads the checkpoint to `--dir/Sana`, converts it to Core AI programs (12 to 20 seconds; one export holds 512 x 512 and 1024 x 1024, and `--size` picks) and compiles each program for the Neural Engine (about 21 minutes the first time at 1024 x 1024, 512 x 512 compiled with it). On an M1 (16 GB), from the second run on, a picture takes about 6 seconds at 512 x 512 and 16 at 1024 x 1024 with Sana 600M, about 9 and 28 with SANA 1.5, with at most 3.9 GB of process memory. The cosine of the latents against diffusers in fp32 from the same noise is 0.998 for Sana 600M and 0.984 for SANA 1.5 (with an fp16 DiT and CFG on every step). Gemma 2 is int8; the DiT is int8 for Sana 600M and lut8 for SANA 1.5; the VAE is fp16; only the time embedding is computed on the host. CFG is on the 14 noisiest steps only (`--cfg-sigmas 0,1` for every step). The cross-attention's keys and values are computed once a prompt. The records are in sections 16.145, 16.160 and 16.189 of [docs/design.md](docs/design.md). To export by hand, use `hwnpu-image convert ~/models/Sana_600M_1024px_diffusers -m OUT` (`--text` for where Gemma 2 is). On Intel's NPU (untested on the hardware), the same steps convert to OpenVINO IR (`--backend intel`). That path does not use lookup-table weights, so SANA 1.5's DiT is fp16 there.

## Image generation (Chroma1)

`hwnpu-image generate -m chroma` draws with lodestones' Chroma1 (the 8.9B DiT of FLUX.1 schnell, T5-XXL and the FLUX VAE, Apache-2.0). `chroma` is Chroma1-Flash (8 Heun steps, no CFG), and `chroma:1-hd` is Chroma1-HD, not distilled (CFG 3, 25 steps of DPM-Solver++ 2M, with CFG on the noisy steps only).

On AMD's NPU it uses the export of `tools/amd/iron_chroma.py` ([docs/amd-npu-models.md](docs/amd-npu-models.md)). On Apple's Neural Engine, the first run downloads the checkpoint (27.5 GB) to `--dir/Chroma`, converts it to Core AI programs in Go alone (about 1 minute) and compiles it for the Neural Engine. T5-XXL is lut8 (fp16 only for the FF output projection, which gives outliers) and is read and closed 3 layers at a time; the DiT is lut8 and the VAE fp16, and all of it runs on the Neural Engine. On an M1 (16 GB), from the second run on, a picture takes about 2 minutes with Chroma1-Flash at 512 × 512, about 9 and a half minutes at 1024 × 1024, and about 22 minutes with Chroma1-HD at 1024 × 1024 (the first run compiles for the Neural Engine for about an hour and a half; HD shares T5 and the VAE with the Flash export). It keeps the DiT's weights (8.3 GiB) pinned on the Neural Engine while it runs, so it needs a Mac of 16 GB or more. The cosine of the latents against float32 on the CPU from the same noise is 0.987 to 0.997 (3 prompts × 2 kinds, 512²), and the pictures are the same. The records are in section 16.170 of [docs/design.md](docs/design.md). To export by hand, use `hwnpu-image convert ~/models/Chroma1-Flash -m OUT --tokenizer T5_TOKENIZER.json` (`--text` if T5 and the VAE are somewhere else).

## Depth estimation (Depth Anything 3, MoGe-2)

`go run ./cmd/hwnpu-image depth` estimates the depth of a photo on AMD's NPU or Apple's Neural Engine (Intel's NPU too, untested on the hardware). The models are Depth Anything 3 (Apache-2.0): DA3MONO-LARGE (the default, relative depth), DA3METRIC-LARGE (meters, with `--fov` or `--focal`) and DA3-SMALL / BASE (lighter versions, depth and confidence). For each image it writes `NAME.depth.png` (brighter is nearer); `--raw` also writes the depth as float32, and `--sky` an image of the sky (the confidence for SMALL / BASE). An export holds a size for each aspect ratio, and the one nearest the image is used. As with generate, `-m` is a name, a family (`da3:small`, `da3metric`, `moge`, `moge:vits`) or an export directory. The default use, `depth`, picks by memory: DA3-SMALL on 8 GB, DA3-BASE on 16 GB and DA3MONO-LARGE from 24 GB (`depth:24g` or `-m DA3MONO-LARGE` for the larger one).

```bash
python tools/amd/iron_da3.py ~/models/DA3MONO-LARGE ~/models/da3mono-large.amd-npu2
go run ./cmd/hwnpu-image depth photo.jpg
```

On a Ryzen AI 9 365 a picture takes 0.7 seconds (LARGE, 504 × 378), and SMALL 0.26 seconds. The memory pinned to the NPU is at most 1.4 GiB (0.5 GiB for SMALL). DA3-SMALL / BASE exported with `--views 3` estimate 3 pictures of the same scene at once, and give depth on one scale, the camera positions (`cameras.json`) and a point cloud (`--ply`). The steps and the accuracy are in section 15 of [docs/amd-npu-models.md](docs/amd-npu-models.md).

On Apple's Neural Engine, the name alone is enough (`hwnpu-image depth -m da3:small photo.jpg`, `-m moge:vits`). The first run fetches the checkpoint, and the size for each aspect ratio is converted when it is first used (a few seconds, and it is compiled on the first load of that size). On an M1 a 504 × 504 picture takes 0.2 seconds with DA3-SMALL, 0.5 seconds with DA3MONO-LARGE and 0.5 seconds with MoGe-2 ViT-S (700 × 700), and the cosine of the depth against the official module is 0.99999 or more. `--views` (several views) is AMD only. The records are in section 16.145 of [docs/design.md](docs/design.md).

Given an export of MoGe-2 (Ruicheng/moge-2-vit{s,b,l}-normal, MIT) made by `tools/amd/iron_moge.py`, it gives depth in meters without knowing the focal length, a point cloud (`--raw`), normals (`--normal`) and a mask of the valid pixels (`--sky`). A picture takes 2.7 to 3.6 seconds at 812 × 602 (section 16).

```bash
python tools/amd/iron_moge.py ~/models/moge-2-vits-normal ~/models/moge-2-vits.amd-npu2
go run ./cmd/hwnpu-image depth -m moge:vits --normal photo.jpg
```

## Face detection (YuNet)

`go run ./cmd/hwnpu-image face-detect` finds the faces in a photo (YuNet, OpenCV Zoo, MIT). For each image it gives one line of JSON with each face's box (x, y, w, h), score and 5 points (right eye, left eye, tip of the nose, right and left corners of the mouth). `--draw` also writes `NAME.faces.png` with the boxes and points drawn. The sizes are 320×256 and 640×480, and the one that shrinks the image least is used (`--size` fixes it). The results match OpenCV's FaceDetectorYN.

Where it runs is set for each NPU vendor by `device` in models.yaml. YuNet runs on the CPU on AMD and on the Neural Engine on Apple (the ONNX graph converted to Core ML as is, 3.4 ms a picture at 640×480). On AMD's NPU, switching between the ops of this small model is costly: 320×256 takes 30 ms and 137 mJ a picture, against 4 ms and 10 mJ on the CPU (docs/amd-npu-models.md section 17). On the CPU it uses ONNX Runtime's shared library. Put `lib/libonnxruntime.so` from a [release](https://github.com/microsoft/onnxruntime/releases) somewhere and set `HWNPU_ONNXRUNTIME` to its path (one in the system is used if there is one). The ONNX file is fetched by itself. `--device npu|cpu` or `HWNPU_DEVICE` overrides where it runs (`--device npu` to keep the CPU free, for games and the like). The RUNS FROM column of `hwnpu-image list` shows where it runs on this machine.

```bash
python tools/amd/iron_yunet.py ~/models/face_detection_yunet ~/models/yunet.amd-npu2   # hf:opencv/face_detection_yunet
go run ./cmd/hwnpu-image face-detect --draw photo.jpg               # AMD: the CPU (ONNX Runtime)
go run ./cmd/hwnpu-image face-detect --device npu photo.jpg          # AMD's NPU: the export above
```

## Objects, poses and text (RTMDet, RTMO, PP-OCRv6)

`go run ./cmd/hwnpu-image detect` finds the objects in a photo (RTMDet, OpenMMLab MMDetection, Apache-2.0). For the 80 COCO classes (person, car, bicycle, traffic light, dog, cup…), it gives one line of JSON for each image with the boxes (x1, y1, x2, y2), labels and scores. `--draw` writes `NAME.objects.png` with the boxes drawn, and `--class person,car` narrows the classes. `go run ./cmd/hwnpu-image pose photo.jpg` gives the people in an image and their 17 keypoints with RTMO. The machine's memory picks RTMDet-s (8 GB) or RTMDet-m (16 GB or more) (`-m rtmdet:tiny` to choose one). On an M1's Neural Engine a picture takes 15 to 26 ms (7 to 18 ms for the network, all on the NPU).

`go run ./cmd/hwnpu-image ocr` reads the text in an image (PaddleOCR's PP-OCRv6 small, Apache-2.0; 50 languages, including Japanese, Chinese and English). For each line it gives the four corners, the text and a score as JSON. `--text` gives the text alone, and `--draw` also writes `NAME.text.png` with the lines' boxes drawn. The two networks, text detection and recognition, run on the NPU: on an M1, 13 lines of a receipt take 40 ms, and a page of a paper (98 lines) 0.5 seconds. Against PaddleOCR's pipeline run on ONNX Runtime, 93 of the page's 98 lines were the same character for character (a character error rate of 0.2 %). The detector's size is chosen from 960×960, 1280×736 and 736×1280 (`--size` changes it; larger picks up smaller text).

```bash
go run ./cmd/hwnpu-image detect --draw photo.jpg                     # JSON lines, photo.objects.png
go run ./cmd/hwnpu-image ocr --text scan.png                         # the lines of text
go run ./cmd/hwnpu-image pose photo.jpg                              # people and their 17 keypoints
```

On Apple's Neural Engine (and Intel's NPU, untested on the hardware), both fetch and convert the model on the first run (the OCR recognizer has a program for each line width, converted when that width is first used, each taking 5 to 20 seconds to compile the first time). On AMD machines, RTMDet-m and `pose`'s RTMO-s run on the NPU (exported with IRON on the first run; 0.33 seconds and 0.22 seconds a picture; docs/amd-npu-models.md sections 21 and 22). RTMDet's tiny and s, and OCR, run on the CPU (as do `hwnpu-audio pitch` and `notes`). With nothing installed they run on hwnpu's CPU executor (on 1 thread, 2.3 seconds a picture for RTMDet-m and 4.5 seconds for 3 lines of OCR), and if ONNX Runtime's shared library is there they use it (`HWNPU_ONNXRUNTIME` set to its path; 1.4 seconds and 1.8 seconds). `--threads` adds threads (docs/amd-npu-models.md section 19). When you need coordinates and kinds of objects, these are faster than a general vision language model (LFM3-VL, `hwnpu-chat --image`) (tens of ms a picture against seconds), pick up every object, and give exact coordinates.

## Labels and image embeddings (SigLIP 2)

`go run ./cmd/hwnpu-image label` labels a photo with Google's SigLIP 2 (Apache-2.0). It puts each candidate (`-l "猫,犬,a red car"`, `--labels-file`, any language) in `this is a photo of {}.` and gives, for each label, the probability that the sentence describes the photo (zero shot; they do not add up to 1). Without candidates it chooses from about 440 built-in objects, places and kinds of image. `hwnpu-image embed` turns photos and texts (`-t`) into embeddings in the same space, and `--compare` gives their cosines and probabilities (for searching images, or finding similar ones). It runs on AMD's NPU and Apple's Neural Engine, and the machine's memory picks Base (8 GB, 224 × 224) or Large (16 GB or more, 256 × 256). On a Ryzen AI 9 365 a picture takes 84 ms (Base) and 8 texts 131 ms, and the cosine of the embeddings against Transformers (float32) is 0.9998 or more. On Apple the first run fetches the checkpoint and converts it in Go (Base 1.5 GB, Large 3.6 GB; the conversion takes a few seconds, and the first load compiles for 13 to 33 seconds). On an M3, Base takes 7 ms a picture and 16 ms for 8 texts (embedding the 444 default labels takes 2 seconds, the first time only), and Large 35 ms and 53 ms. The cosine of the embeddings against Transformers is 0.9998 or more for images and 0.99999 or more for texts. The label embeddings go in the user cache, and the same list is computed only once.

```bash
python tools/amd/iron_siglip2.py ~/models/siglip2-base-patch16-224 ~/models/siglip2-base-patch16-224.amd-npu2   # AMD only. hf:google/siglip2-base-patch16-224
go run ./cmd/hwnpu-image label photo.jpg                     # JSON lines: the likeliest labels
go run ./cmd/hwnpu-image embed --compare photo.jpg -t "a cat on a sofa"
```

AMD's steps and accuracy are in section 18 of [docs/amd-npu-models.md](docs/amd-npu-models.md), and Apple's records in section 16.183 of [docs/design.md](docs/design.md).

## Cutouts from points and boxes (EfficientSAM)

`go run ./cmd/hwnpu-image segment` makes the mask of the object pointed at by points (`--point x,y`; `--negative` for points to leave out) or a box (`--box x1,y1,x2,y2`), with Meta's EfficientSAM (Apache-2.0, Segment Anything's prompts). It writes `NAME.mask.png` (the object in white) and `NAME.cutout.png` (transparent outside the object), and gives the score (the predicted IoU), the area and the box as JSON. Of the decoder's 3 candidates it writes the one with the highest score; `--all` writes all 3. The image encoder (a 1024 × 1024 ViT) runs on AMD's NPU (exported with IRON on the first run), Apple's Neural Engine and Intel's NPU (converted from ONNX on the first run; Intel is untested on the hardware). On other machines, or with `--device cpu`, it runs on the CPU (ONNX Runtime if it is there, otherwise hwnpu's evaluator, which is slow). The decoder runs on the host for each prompt. The machine's memory picks Ti (8 GB) or S (16 GB or more). On a Ryzen AI 9 365 the encoder takes 0.34 seconds for Ti and 0.48 seconds for S, and the decoder 0.6 to 0.7 seconds. On an M3 S's encoder takes 0.4 seconds and the decoder 0.3 seconds, and the mask has an IoU of 0.9994 with the CPU's.

```bash
go run ./cmd/hwnpu-image segment --box 280,40,500,385 photo.jpg   # photo.mask.png, photo.cutout.png
go run ./cmd/hwnpu-image segment --point 390,200 --all photo.jpg
```

The steps and the accuracy are in section 23 of [docs/amd-npu-models.md](docs/amd-npu-models.md).
