# What hwnpu covers of the cloud AI APIs

[日本語](CLOUD_AI_COMPATIBILITY.ja.md)

As of 2026-11-11. This page maps what cloud APIs do to the hwnpu commands and models that do it on the machine's NPU. It also gives each NPU's state, what has no counterpart yet, and the candidates for it. The APIs compared:

- Google Cloud (Vision, Speech-to-Text, Text-to-Speech, Natural Language, Translation, Document AI, Vertex AI)
- AWS (Rekognition, Transcribe, Polly, Comprehend, Textract)
- Azure AI (Vision, Speech, Language, Document Intelligence, Content Safety)
- OpenAI's API
- Apple's on-device frameworks (Vision, SoundAnalysis)

The order of the work: images, then audio, then language, then the API endpoints.

## Marks

| Mark | Meaning |
|---|---|
| ◎ | Run on the device, checked against a reference and measured |
| ○ | The path is there, not run on the device (all of Intel so far; README's table) |
| CPU | No NPU form; runs on the CPU (`--cpu`'s interpreter, or `device: {amd: cpu}` in models.yaml) |
| — | Does not run on that NPU |

The NPUs:

- **Apple**: the Neural Engine of the M1 to M4 (Core ML)
- **Intel**: Core Ultra's NPU (OpenVINO IR, Level Zero)
- **AMD**: Ryzen AI's NPU (IRON, xdna)

## 1. What there is

| Feature | In the cloud | hwnpu | Models | Apple | Intel | AMD |
|---|---|---|---|---|---|---|
| Chat, coding | OpenAI Chat Completions, Gemini, Bedrock | `hwnpu-chat`, `agent`, `serve` (`/v1/chat/completions`) | LFM3, Gemma 4, Qwen3.5 and others | ◎ | ○ | ◎ (models.yaml's `amd_export`) |
| Questions about images | GPT's image input, Gemini | `hwnpu-chat -m vision` | LFM3-VL-2B, Gemma 4 E2B with images and audio | ◎ | ○ | ◎ |
| Questions about audio | GPT's audio input, Gemini | `hwnpu-chat -m audio` | LFM3-Audio / Omni, Gemma 4 E2B | ◎ | ○ | ◎ |
| Translation | Cloud Translation, Azure Translator, DeepL | `hwnpu-chat -m translate` | PLaMo 3 Translate 0.8B / 2B | ◎ | ○ | ◎ |
| Text embeddings | OpenAI Embeddings, Vertex text embeddings | `hwnpu-chat search`, `/v1/embeddings` | PLaMo-Embedding-1B | ◎ | ○ | — (no export) |
| Transcription | Speech-to-Text, Transcribe, Azure Speech, OpenAI `/v1/audio/transcriptions` | `hwnpu-audio transcribe` (no endpoint yet) | Whisper large-v3-turbo | ◎ | — (writes Core ML packages only) | — |
| Speech | Text-to-Speech, Polly, Azure TTS, OpenAI `/v1/audio/speech` | `hwnpu-audio tts` (no endpoint yet) | LFM3-Audio-2B | ◎ | — | — |
| Decisions about a state | (llama.cpp's System One API) | `hwnpu-chat systemone` | d1-2B, d1-Omni-2B | ◎ | — | — |
| Image generation | OpenAI Images, Imagen | `hwnpu-image generate` (no endpoint yet) | Anima-Turbo | ◎ | ○ | ◎ |
| 〃 | 〃 | 〃 | Sana / Sana-Sprint | ◎ | — | ◎ |
| Depth, points, normals | (no common cloud API) | `hwnpu-image depth` | Depth Anything 3, MoGe-2 | ○ (converted, not measured) | — | ◎ |
| Faces (boxes and 5 landmarks) | Cloud Vision face detection, Rekognition DetectFaces, Azure Face, Apple Vision | `hwnpu-image face-detect` | YuNet | ○ (not measured) | — | CPU (it runs on the NPU, at 16–40× the CPU's energy; docs/amd-npu-models.md 17) |
| People's poses (17 points) | Apple Vision's body pose, Video Intelligence's person detection | `hwnpu-image pose` | RTMO-s | ◎ | ○ | CPU |
| Objects (boxes and labels) | Cloud Vision object localization, Rekognition DetectLabels, Azure object detection | `hwnpu-image detect` | RTMDet-tiny / s / m | ◎ | ○ | CPU |
| Text (OCR) | Cloud Vision TEXT / DOCUMENT_TEXT_DETECTION, Textract, Azure Read, Apple Vision | `hwnpu-image ocr` | PP-OCRv6 small | ◎ | ○ | CPU |
| Pitch (F0) | (none) | `hwnpu-audio pitch` | SwiftF0 | ◎ | ○ | CPU |
| Notes (MIDI) | (none) | `hwnpu-audio notes` | Basic Pitch | ◎ | ○ | CPU |
| Music | Vertex's Lyria | `hwnpu-audio generate` | Magenta RealTime 2 small | ◎ | — | — (`--cpu` only) |
| Sound effects | ElevenLabs Sound Effects | `hwnpu-audio generate -m sfx` | MOSS-SoundEffect v2.0 | ◎ | — | — (`--cpu` only) |

The five on AMD's CPU (poses, objects, OCR, pitch, notes) all come through `internal/onnx`, which lowers ONNX to MIL. That path writes Core ML and OpenVINO IR, not an AMD export. A general path from ONNX to an IRON graph would put all five on AMD at once. `tools/amd/iron_yunet.py` reads YuNet's ONNX, but lays its layers out by hand.

## 2. What there is not yet

### 2.1 Images (next)

| Feature | In the cloud | Candidates (license) | hwnpu (a plan) | On AMD |
|---|---|---|---|---|
| **Labels (zero-shot) and image embeddings** | Cloud Vision LABEL_DETECTION, Rekognition DetectLabels, Vertex's and Azure's multimodal embeddings, Apple Vision's classification and feature prints | SigLIP 2 (Apache-2.0, many languages), LINE's Japanese CLIP (Apache-2.0) | `hwnpu-image label`, `hwnpu-image embed`; later `/v1/embeddings` takes images | ◎ fits: a ViT, mostly GEMMs; DA3's and MoGe-2's ViT parts carry over |
| **Object masks** (instance segmentation) | Apple Vision's foreground instance masks, Vertex AutoML | RTMDet-Ins (Apache-2.0) | `hwnpu-image detect --mask` | △ a small CNN of many ops, which may lose to the CPU as YuNet does; measure after the general ONNX → IRON path |
| **Background removal**, subject cutouts | Apple's subject lifting, remove.bg | BiRefNet / BiRefNet_lite (MIT); RMBG-2.0 is non-commercial and out | `hwnpu-image cutout` (a transparent PNG) | ○ the Swin backbone is mostly GEMMs; how to run the decoder's deformable conv is still to check |
| Cutouts from a point or a box | Meta's SAM | SAM 2, EfficientSAM (Apache-2.0) | `hwnpu-image segment --point x,y` / `--box` | ○ the image encoder is a ViT; the light decoder can run on the CPU per prompt |
| **Document layout, tables, page and line orientation** | Document AI, Textract AnalyzeDocument, Azure Document Intelligence Layout | PP-DocLayout, SLANet, PP-LCNet orientation (Apache-2.0) | `hwnpu-image ocr --layout` / `--table`; orientation fixed by itself | △ CNNs; after the general path, as the masks |
| Face comparison (the same person?) | Rekognition CompareFaces / SearchFaces, Azure Face Verify | SFace (OpenCV Zoo, Apache-2.0); insightface's weights are non-commercial and out | `hwnpu-image face-detect --embed`, `face-compare` | CPU, likely (a small CNN) |
| Unsafe images | Cloud Vision SafeSearch, Rekognition DetectModerationLabels, Azure Content Safety | Falconsai/nsfw_image_detection (Apache-2.0, accuracy to check), or SigLIP 2 zero-shot | `hwnpu-image moderate` | ◎ fits (a ViT) |
| Upscaling | Vertex Imagen's upscale | Real-ESRGAN (BSD-3) | `hwnpu-image upscale` | ○ convolutions over large maps, large im2col GEMMs |
| Erasing (inpainting) | Vertex Imagen's editing | LaMa (Apache-2.0) | `hwnpu-image erase --mask` | △ it has FFTs, to be written as DFT matmuls (on Apple too) |
| Barcodes, QR | Apple Vision, ML Kit | gozxing (Apache-2.0, Go; not deep learning) | `hwnpu-image barcode` | Any (the host) |
| Tracking in video | Video Intelligence, Rekognition Video | ByteTrack (MIT; no new model) | `hwnpu-image detect --track` (numbered frames) | Wherever detect runs |

Left out:

- Logos, landmarks, celebrities and web matches need large databases, not on-device models.
- Faces' emotion, age and gender: biased, and Azure has withdrawn them.

### 2.2 Audio

| Feature | In the cloud | Candidates (license) | hwnpu (a plan) |
|---|---|---|---|
| **Voice activity (VAD)** | Inside Speech-to-Text and Transcribe, WebRTC VAD | Silero VAD (MIT) | `hwnpu-audio vad`; cuts the spans before transcription |
| **Speaker diarization** (who spoke) | Speech-to-Text diarization, Transcribe speaker labels, Azure Speech | pyannote segmentation-3.0 (MIT, downloads behind an agreement), WeSpeaker / 3D-Speaker (Apache-2.0) | `hwnpu-audio transcribe --speakers` |
| Sound classification (applause, dogs, sirens) | Apple's SoundAnalysis | YAMNet (Apache-2.0, TFLite: `internal/tflite`) | `hwnpu-audio classify` |
| Source separation (voice and backing) | (specialist services) | Demucs (MIT, size to check) | `hwnpu-audio separate` |
| Noise suppression | (NVIDIA Maxine and others) | DeepFilterNet (MIT / Apache-2.0) | `hwnpu-audio denoise` |

### 2.3 Language

| Feature | In the cloud | Candidates (license) | hwnpu (a plan) |
|---|---|---|---|
| **Reranking** | Cohere Rerank, Vertex's Ranking API, Bedrock's rerank | ruri's reranker (to check), bge-reranker-v2-m3 (Apache-2.0) | `hwnpu-chat search --rerank`, `/v1/rerank` |
| Entities, sentiment, classification | Cloud Natural Language, Comprehend, Azure Language | The chat models do it; GLiNER (Apache-2.0, per version to check) to be small and fast | Not urgent |
| Unsafe text | OpenAI Moderations, Azure Content Safety | A chat model asked to judge | Not urgent |

### 2.4 API endpoints

`hwnpu-chat serve` has three: `/v1/models`, `/v1/chat/completions` and `/v1/embeddings` (`internal/chat/server.go`). Some features have the model but no endpoint:

- `/v1/audio/transcriptions` (Whisper), `/v1/audio/speech` (LFM3-Audio), `/v1/images/generations` (Anima, Sana)
- An image analysis endpoint (as Cloud Vision's `images:annotate`: objects, text, faces, poses and labels in one call)
- `/v1/rerank` (after the reranker of 2.3)

## 3. Starting from AMD

AMD has no public graph compiler; each model's IRON graph is built in Python (`tools/amd/iron_*.py`; docs/amd-npu-models.md 1 and 2.3). What decides the speed, as measured (ibid. 17.2):

- Each switch between a GEMM and another op takes about 250 µs and about 1 mJ. Runs of the same op take 5–30 µs each.
- So small CNNs of many ops lose to the CPU. YuNet (about 70 ops) took 127 mJ a picture on the NPU, 7.7 mJ on one CPU thread.
- ViTs, transformers and large convolutions (im2col GEMMs) fit. DA3, MoGe-2, Sana and Anima run this way.

The order suggested for the image features on AMD:

1. **SigLIP 2** (labels, image embeddings, unsafe images): a ViT, so DA3's parts carry over. This one model fills three rows of 2.1.
2. **A SAM family encoder and BiRefNet** (cutouts): heavy backbones, mostly GEMMs. Look at BiRefNet's deformable conv first.
3. **A general ONNX → IRON path**: this puts the five of section 1 (poses, objects, OCR, pitch, notes) on AMD, and RTMDet-Ins and the document layout with them. Compare each model's mJ on the NPU and the CPU, and leave the ones that lose at `device: {amd: cpu}` in models.yaml, as YuNet is.
4. **Barcodes and tracking**: any NPU, any time.

## 4. Not checked

- The candidates' licenses and sizes are from what was written when surveyed, and from memory. Check each against its weight files and model card before bringing it in (docs/new-model-checklist.md 1).
- Intel's ○ means only that the conversion path is there. None of it has run on the device.
- Apple's depth and face detection are converted but not measured for speed or energy on the M1.
