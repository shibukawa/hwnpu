# Recommended models

Which model hwnpu runs best for each use, by how much memory the machine has. The numbers are those of main on 2026-11-06; Apple's chat models were measured again with Core AI (macOS 27) on an M1 on 2026-11-15, its image models on 2026-11-16. What each NPU computes in and how models are converted are in [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

Japanese: [MODEL.ja.md](MODEL.ja.md)

## By use and memory

The columns are the memory installed in the machine. The second header row is roughly what is left for models after the OS and everyday applications. NPU buffers are pinned and cannot be swapped out: going over that amount pushes other applications to swap.

hwnpu-chat and hwnpu-chat agent pick from this table themselves: the name under each use (`-m translate`, `-m chat`) runs the model of the machine's column, downloading only that one. A family picks the same way (`-m plamo3translate`, `-m gemma4`), and a tag chooses one whatever the memory, like a Docker image's: a model's (`-m plamo3translate:2b`, `-m chat:e4b`) or a column's (`-m chat:24g`, `-m plamo3translate:8g`: what a machine of that memory runs). `hwnpu-chat list` shows what each name picks on the machine; `HWNPU_MEMORY=24` picks as a machine of 24 GB would. hwnpu-audio transcribe (`-m transcribe`), hwnpu-audio tts (`-m speech`) and hwnpu-chat systemone (`-m decision`) pick the same way. The table comes from [models.yaml](models.yaml) (its `categories` and the models' `machine`), which hwnpu embeds; a file of your own adds models, replaces a column's model or points a model to another (aliases), written as differences only: see [docs/model-table.md](docs/model-table.md).

<table>
<thead>
<tr>
<th rowspan="2">Use</th>
<th>8 GB</th>
<th>16 GB</th>
<th>24 GB</th>
<th>32 GB</th>
<th>48 GB</th>
<th>64 GB or more</th>
</tr>
<tr>
<th>about 3 GB</th>
<th>about 9 GB</th>
<th>about 15 GB</th>
<th>about 22 GB</th>
<th>about 35 GB</th>
<th>about 50 GB</th>
</tr>
</thead>
<tbody>
<tr>
<th>Chat, coding<br><sub>hwnpu-chat, hwnpu-chat agent <code>-m chat</code></sub></th>
<td><b>LFM3-1.7B</b> (1.7 GB on Apple, 2.2 GB on AMD)<br>fast, for agents<br><b>Gemma 4 E2B</b> (2.4 GB on AMD, 2.3 GB on Apple)<br>a size larger, 5B with its per-layer embeddings<br><b>Qwen3.5-2B</b> (<code>-m qwen3.5</code>, 2.2 to 2.4 GB)<br>writes code too<br><b>Qwen3-1.7B</b> (<code>-m qwen3</code>, 2.1 GB; Apple only)<br>in one program, slower to read long prompts (<code>--prefill on</code> for faster, +0.3 GB)<br><b>Qwen3-0.6B</b> (<code>-m qwen3:0.6b</code>, 1.1 GB on an M1, 1.6 GB on an M2 or later, 1.5 GB download)<br>the smallest download, weak at facts</td>
<td><b>Gemma 4 E2B</b> (2.4 GB on AMD, 2.3 GB on Apple)<br>with a long context too (+0.8 GB at 131072)<br><b>Gemma 4 E4B</b> (4.3 GB on AMD, 4.0 GB on Apple)<br>for short conversations<br><b>Qwen3-1.7B</b> (<code>-m qwen3</code>, 2.3 GB)<br>with a program of its own to read prompts, 5 times faster at it</td>
<td colspan="2"><b>Gemma 4 E4B</b> (4.3 GB on AMD, 4.0 GB on Apple)<br>with a long context too (+2.1 GB at 131072)</td>
<td colspan="2"><b>Gemma 4 E4B</b> (4.3 GB on AMD, 4.0 GB on Apple)<br><i>planned: Qwen3.6-35B-A3B (MoE, about 25 GB at 4 bits)</i></td>
</tr>
<tr>
<th>Reading images<br><sub>hwnpu-chat <code>-m vision</code> (<code>image_url</code>, <code>/image</code>)</sub></th>
<td><b>LFM3-VL-2B</b> (2.6 to 2.7 GB)</td>
<td colspan="5"><b>Gemma 4 E2B with images and audio</b> (3.4 GB)<br><b>LFM3-VL-2B</b> (2.6 to 2.7 GB)</td>
</tr>
<tr>
<th>Transcription<br><sub>hwnpu-audio transcribe</sub></th>
<td colspan="6"><b>Whisper large-v3-turbo</b> (about 1.9 GB)<br>99 languages, with timestamps, 30 seconds read in 0.6 s</td>
</tr>
<tr>
<th>Pitch (F0)<br><sub>hwnpu-audio pitch</sub></th>
<td colspan="6"><b>SwiftF0</b> (about 0.06 GB)<br>the pitch of one voice or instrument every 16 ms, with a voicing confidence; 41 seconds in 0.24 s</td>
</tr>
<tr>
<th>Notes (MIDI)<br><sub>hwnpu-audio notes</sub></th>
<td colspan="6"><b>Basic Pitch</b> (about 0.07 GB)<br>any instrument, chords too, with pitch bends, as MIDI; 35 times real time</td>
</tr>
<tr>
<th>People's poses<br><sub>hwnpu-image pose</sub></th>
<td colspan="6"><b>RTMO-s</b> (about 0.23 GB)<br>the people in an image and their 17 keypoints in one pass (no person detector); about 0.1 s an image</td>
</tr>
<tr>
<th>Objects<br><sub>hwnpu-image detect</sub></th>
<td><b>RTMDet-s</b> (about 0.1 GB)<br>boxes and COCO's 80 labels; 9 ms an image on the M1's Neural Engine</td>
<td colspan="5"><b>RTMDet-m</b> (about 0.2 GB)<br>more accurate (49.1 box AP, s 44.6); 18 ms an image</td>
</tr>
<tr>
<th>Text (OCR)<br><sub>hwnpu-image ocr</sub></th>
<td colspan="6"><b>PP-OCRv6 small</b> (about 0.45 GB)<br>lines of text, their boxes and what they say, 50 languages (Japanese, Chinese, English, ...); a page of 98 lines in 0.5 s on an M1</td>
</tr>
<tr>
<th>Music generation<br><sub>hwnpu-audio generate</sub></th>
<td colspan="6"><b>Magenta RealTime 2 small</b> (about 1.0 GB)<br>48 kHz stereo music in the style of texts or recordings, playing a MIDI file's notes or going on from a clip, streamed as it comes, about 1.4 times faster than real time; weights CC-BY 4.0 (credit "Magenta RealTime 2 by Google DeepMind" where the music is used)</td>
</tr>
<tr>
<th>Sound effects<br><sub>hwnpu-audio generate -m sfx</sub></th>
<td><b>MOSS-SoundEffect v2.0</b> (about 2.2 GB; the text encoder in lut8 with 5 layers in fp16)</td>
<td colspan="5"><b>MOSS-SoundEffect v2.0</b> (about 2.3 GB; the text encoder in fp16)<br>up to 30 s of 48 kHz mono from an English or Chinese description; 10 s in about 4 minutes at the card's 100 steps on an M1 (<code>-steps 50</code>: half); Apache-2.0</td>
</tr>
<tr>
<th>Reading audio (questions about it)<br><sub>hwnpu-chat <code>-m audio</code> (<code>input_audio</code>, <code>/audio</code>)</sub></th>
<td><b>LFM3-Audio-2B</b> (2.6 GB)</td>
<td colspan="5"><b>Gemma 4 E2B with images and audio</b> (3.4 GB)<br><b>LFM3-Omni-2B</b> (3.0 to 3.1 GB, images too)</td>
</tr>
<tr>
<th>Translation<br><sub>hwnpu-chat <code>-m translate</code></sub></th>
<td><b>PLaMo 3 Translate 0.8B</b> (1.0 GB)<br>3 times as fast as the 2B, translations differing from it by rewordings</td>
<td colspan="5"><b>PLaMo 3 Translate 8B</b> (8.2 GB on Apple, about 6.5 GB on AMD)<br>more apt wording than the 2B; 10 to 11 tokens/s on an M3 (loading in 4 s), 9 to 10 on an M2 (35 to 41 s every start)<br>on an M1 <b>PLaMo 3 Translate 2B</b> (3.2 GB) instead: the 8B is 0.8 tokens/s there (<code>-m plamo3translate:8b</code> still runs it)</td>
</tr>
<tr>
<th>Image generation<br><sub>hwnpu-image</sub></th>
<td colspan="6"><b>FLUX.2 klein 4B</b> 1024 × 1024 (3.9 GB on AMD, 4.9 GB on Apple; at <code>--size 512</code> 1.9 GB on AMD, 3.1 GB on Apple)<br>the main model: Apache-2.0, 4 steps, 41 s on an M1 (11 s at 512 × 512), 29 s on AMD; edits and composes from reference images on AMD (<code>--image</code>)<br>Opt-in (<code>HWNPU_MODEL_ALLOWLIST=+nsfw</code>, named with <code>-m</code>): <b>Anima-Turbo</b> (4.8 GB on Apple, 48 s on an M1; Danbooru tags), <b>Anima-Aesthetic</b> (<code>-m anima</code>, 25 steps with guidance, about 4.5 times as long)</td>
</tr>
<tr>
<th>Speech<br><sub>hwnpu-audio tts</sub></th>
<td colspan="6"><b>LFM3-Audio-2B</b> (about 2.4 GB)</td>
</tr>
<tr>
<th>Sentence embeddings, search<br><sub>hwnpu-chat <code>-m embedding</code> (<code>/v1/embeddings</code>)</sub></th>
<td colspan="6"><b>PLaMo-Embedding-1B</b> (1.2 to 1.3 GB)</td>
</tr>
<tr>
<th>Decisions on a state<br><sub>hwnpu-chat systemone</sub></th>
<td colspan="6"><b>d1-2B</b> (about 2.4 GB, images too)<br><b>d1-Omni-2B</b> (about 2.8 GB, images and audio)</td>
</tr>
<tr>
<th>Several at once<br><sub>hwnpu-chat <code>serve -m A,B,…</code></sub></th>
<td>LFM3-1.7B + PLaMo 3 Translate 0.8B (2.7 GB on Apple, 3.2 GB on AMD)</td>
<td>Gemma 4 E2B + PLaMo-Embedding-1B (3.5 to 3.6 GB)<br>+ PLaMo 3 Translate 2B (6.3 GB on AMD, 6.7 GB on Apple)</td>
<td>Gemma 4 E4B + PLaMo-Embedding-1B + PLaMo 3 Translate 2B (8.0 GB on AMD, 8.4 GB on Apple)</td>
<td colspan="3">and Gemma 4 E2B with images and audio, and FLUX.2 klein 4B in another process (about 16 to 17 GB)</td>
</tr>
</tbody>
</table>

- Image generation (`-m image`, the default of hwnpu-image generate) is FLUX.2 klein 4B from 8 GB. Anima and Chroma, which may draw NSFW pictures unasked, run only when you opt in with `HWNPU_MODEL_ALLOWLIST=+nsfw` and name them (`-m anima-turbo`, `-m chroma`): the use does not pick them even then. The same variable narrows the models by license ([docs/model-provenance.md](docs/model-provenance.md))
- The figure in parentheses is the expected memory after starting and a short conversation (next section). Long conversations add the KV cache
- Past 32 GB there is no larger model yet. A machine with more memory runs the combinations above at once, or long contexts
- A machine of 8 GB runs the models the way that takes less memory (LFM3-1.7B without its whole program of short conversations, Qwen3-1.7B without a program of its own to read prompts). To run one beside other programs on a machine of more memory, `hwnpu memory LFM3-1.7B 8` (or `hwnpu-chat --memory 8`) runs it the 8 GB way in every command ([docs/cache-sharing-design.md](docs/cache-sharing-design.md) 7.8)
- On a small machine, a shorter `--context` cuts the KV cache. hwnpu-chat shrinks the context itself when the KV cache would be over a quarter of the physical memory

## Memory per model

"At start" is the NPU's buffers (weights, work areas, the KV cache of the default context) plus the process itself (tokenizer, host buffers, about 0.4 GB). NPU buffers do not show in the process's RSS, so this is more than a task manager or `top` shows.

| Model | Use | At start | Growth with the conversation (KV cache) | Basis |
|---|---|---|---|---|
| LFM3-1.7B | Chat, coding | 2.2 GB on AMD. On Apple 1.7 GB (1.36 GiB on the Neural Engine and 0.23 GiB of the process); 3.1 GB on an M2 or later from 16 GB (the program of short conversations holds the weights once more: 2.64 GiB and 0.21 GiB) | 12 KiB/token (3 GiB at 262144) | measured (AMD and an M1; an M2's programs loaded on the M1 with `--tier 2048`) |
| LFM2-8B-A1B | Chat, tool calls (`-m lfm2`; Apple's M2 and later, not an M1) | 8.0 GB on an M2 (the experts' 7.23 GiB in the process's memory, `HWNPU_MOE_WIRE=1` wires them when memory is short); 12.9 GB at the first run's conversion | 13.5 KiB/token with the means of 8 positions (1.65 GiB at 128000) | measured (an M2; speed on an M3 too) |
| Gemma 4 E2B | Chat, coding | 2.4 GB on AMD (2026-11-17), 2.3 GB on Apple (an M1 with Core AI: 1.66 GiB on the Neural Engine; the process not measured with Core AI) | 6 KiB/token (0.75 GiB at 131072) | measured |
| Gemma 4 E2B with images and audio | Images, audio (one image and 30 seconds a prompt) | 3.4 GB (2.4 GB until an image or audio arrives) | 6 KiB/token | measured |
| Gemma 4 E4B | Chat, coding | 4.3 GB on AMD (2026-11-17), 4.0 GB on Apple | 16 KiB/token (2 GiB at 131072) | measured |
| LFM3-VL-2B | Images (512 × 512) | 2.6 GB on AMD, 2.7 GB on Apple (M1: 2.24 GiB on the Neural Engine and 0.25 GiB of the process describing a picture; 2.6 GB with text alone, the vision encoder loaded from the start) | 12 KiB/token (1.46 GiB at 128000) | measured (AMD and an M1) |
| SwiftF0 | Pitch (F0) | about 0.06 GB (M1: a footprint of 58 MB; 14K weights) | none (30 seconds at a time) | measured on an M1 |
| Basic Pitch | Notes (MIDI) | about 0.07 GB (M1: a footprint of 75 MB; 36K weights) | none (2 second windows) | measured on an M1 |
| RTMO-s | People's poses | about 0.23 GB (M1: a footprint of 232 MB; 9.9M weights in fp16) | none (an image at a time) | measured on an M1 |
| RTMDet-tiny / s / m | Objects | about 0.07 / 0.1 / 0.2 GB (M1: footprints of 56 / 73 / 141 MB, and 11 / 21 / 56 MB on the Neural Engine) | none (an image at a time) | measured on an M1 |
| PP-OCRv6 small | Text (OCR) | about 0.45 GB (M1: a footprint of 425 MB at most, the recognizer's programs of seven widths loaded) | none (an image at a time) | measured on an M1 |
| Magenta RealTime 2 small | Music generation | about 1.0 GB (M1 footprints: a text style 0.97 GB, a recording's style 0.55 GB, going on from a clip 0.96 GB, 0.42 GB with the codes given by `-codes`) | none (a window of the last 41 frames) | measured on an M1 |
| MOSS-SoundEffect v2.0 | Sound effects | about 2.3 GB (M1: a peak footprint of 2.34 GB with the text encoder in fp16, 2.17 GB in lut8; the text encoder's programs, 2.8 GB in fp16 or 1.7 GB in lut8, run first and are let go, then the DiT's 1.4 GB in lut8 and the decoder's 0.2 GB) | none (30 seconds of latents at a time) | measured on an M1 |
| Whisper large-v3-turbo | Transcription (any length, 30 seconds at a time) | about 1.9 GB (M3: 0.78 GiB on the Neural Engine and 1.1 GB of the process) | none (up to 448 tokens every 30 seconds) | measured on an M3 |
| LFM3-Audio-2B | Audio (clips up to 30 seconds) | 2.6 GB (M1: 2.18 GiB on the Neural Engine and 0.21 GiB of the process answering about a clip; 2.4 GB with text alone) | 12 KiB/token | measured (AMD and an M1) |
| LFM3-Omni-2B | Images and audio | 3.0 GB on AMD (2.2 GB until an image or audio arrives), 3.1 GB on Apple (M1: 2.63 GiB on the Neural Engine and 0.25 GiB of the process with a picture and a clip; both encoders are loaded from the start, 2.8 GB with text alone) | 12 KiB/token | measured (AMD and an M1) |
| PLaMo 3 Translate 2B | Translation | 2.7 GB on AMD, 3.2 GB on Apple (M1: 2.74 GiB on the Neural Engine and 0.24 GiB of the process) | 7 KiB/token (1.75 GiB at 262144) | measured (AMD and an M1) |
| PLaMo 3 Translate 8B | Translation | about 8.2 GB on Apple (M1: 3.81 GiB on the Neural Engine in 4 bit tables and the process's 4.1 GB at its peak); about 6.5 GB on AMD (q4, estimated from the weights) | 9 KiB/token (2.25 GiB at 262144) | measured on an M1, M2 and M3 (Apple); AMD not run |
| PLaMo 3 Translate 0.8B | Translation | 1.0 GB (M1: 0.82 GiB on the Neural Engine and 0.12 GiB of the process) | 1 KiB/token (0.28 GiB at 262144) | measured (AMD and an M1) |
| PLaMo-Embedding-1B | Sentence embeddings | 1.3 GB on AMD, 1.2 GB on Apple (M1: 1.05 GiB on the Neural Engine and 0.03 GiB of the process) | none | AMD estimated from the weights; Apple measured on an M1 |
| MoGe-2 ViT-S / B / L (AMD) | Metric depth, points, normals (hwnpu-image depth) | 1.9 GB (S), 2.1 GB (B), 2.6 GB (L) | none | measured |
| Depth Anything 3 Mono / Metric Large, Small, Base (AMD) | Depth estimation (hwnpu-image depth) | 1.4 GB (Large), 0.75 GB (Base), 0.5 GB (Small) | none | measured |
| FLUX.2 klein 4B (1024 × 1024, AMD) | Image generation, editing reference images (hwnpu-image generate `-m flux2-klein`, `--image`) | 3.9 GB (the NPU's 3.4 GiB peak, in the VAE's stages, and the process's 0.2 GiB; the same for an edit; 1.9 GB at 512 × 512; the text encoder and the DiT are read a group at a time) | none | measured |
| FLUX.2 klein 4B (1024 × 1024, Apple) | Image generation (hwnpu-image generate `-m flux2-klein`; editing runs on AMD) | 4.9 GB (M1: in the VAE's stages 3.8 GiB of the process and 0.8 GiB on the Neural Engine; 2.3 GiB on the Neural Engine at most, in the DiT; 3.1 GB at 512 × 512) | none | measured (M1) |
| FLUX.3 klein (1024 × 1024, AMD) | Image generation (hwnpu-image generate `-m flux3-klein`; text to image) | 3.9 GB (the NPU's buffers and the process's 3.6 GiB at the peak; 2.7 GB at 512 × 512; the text encoder and the DiT are read a group at a time) | none | measured |
| FLUX.3 klein (1024 × 1024, Apple) | Image generation (hwnpu-image generate `-m flux3-klein`; text to image) | 4.1 GB of process memory at most (M3; the Neural Engine's own not measured) | none | measured (M3) |
| DA3-SMALL, DA3MONO-LARGE, MoGe-2 ViT-S (Apple) | Depth estimation (hwnpu-image depth) | the programs of one size (fp16), converted for each aspect on first use | none | not measured |
| SigLIP 2 Base / Large (Apple) | Labels, embeddings of pictures and texts (hwnpu-image label, embed) | About 0.9 / 3.3 GB (M3: a footprint of 0.45 / 1.8 GB at most, and 0.36 / 1.24 GB on the Neural Engine; 0.48 / 1.56 GB with Core AI) | none (a picture, 8 texts at a time) | Measured on an M3 |
| EfficientSAM-Ti / S (Apple) | The cutout a point or a box picks (hwnpu-image segment) | A footprint of about 0.22 / 0.25 GB (M3; the encoder on the Neural Engine, the decoder on the CPU) | none (a picture at a time) | Measured on an M3 |
| Anima-Turbo, Anima-Aesthetic (1024 × 1024) | Image generation (opt-in, `+nsfw`) | 3.5 GB on AMD (the NPU's 2.4 GiB peak in the VAE's stages and the process's 0.9 GiB); 4.8 GB on Apple (M1: 3.1 GiB of the process and 1.4 GiB on the Neural Engine) | none | measured (AMD and an M1) |
| LFM3-Audio-2B (speech) | Speech | about 2.4 GB | 12 KiB/token | estimated from the weights |
| d1-2B, d1-Omni-2B | Decisions on a state | about 2.4 GB, about 2.8 GB | short inputs only | estimated from the weights |
| Qwen3-0.6B | Chat (weak at facts) | 1.1 GB on an M1 (0.65 GiB on the Neural Engine and 0.35 GiB of the process), 1.6 GB on an M2 or later (with the program of short conversations 1.23 GiB and 0.26 GiB) | 126 KiB/token (4.9 GiB at its context of 40960; a machine of 16 GB cuts it to 32768, 3.9 GiB) | measured on an M1 (an M2's programs with `--tier 1024`) |
| Qwen3-1.7B | Chat (weak at Japanese facts) | 2.1 GB below 16 GB, 2.3 GB from 16 GB (M1: one program of 8 bit tables, 1.64 GiB on the Neural Engine and 0.23 GiB of the process; from 16 GB with the function that reads prompts, 1.80 GiB and 0.36 GiB) | 126 KiB/token (4.9 GiB at its context of 40960; a machine of 16 GB cuts it to 32768, 3.9 GiB) | measured on an M1 |
| Qwen3.5-0.8B | Chat (weak at Japanese facts) | 1.5 GB (AMD with a 6-bit MLP; Apple in 8 bit tables, 0.99 GiB on the Neural Engine and 0.39 GiB of the process); 2.3 GB on an M2 or later from 16 GB (with the program of short conversations 1.84 GiB and 0.31 GiB) | 14 KiB/token (3.38 GiB at 262144) | measured (AMD and an M1; an M2's programs with `--tier 2048`) |
| Qwen3.5-2B | Chat, code | 2.4 GB on AMD (6-bit MLP), 2.2 GB on Apple (M1: 1.67 GiB on the Neural Engine in 6 bit tables and 0.39 GiB of the process) | 14 KiB/token (3.38 GiB at 262144) | measured (AMD and an M1) |
| Qwen3.5-4B | Chat, code | about 4.2 GB (3.9 GiB of 8 bit weights) | 32 KiB/token (8 GiB at 262144; 114688 tokens on 16 GB) | runs on an M3's Neural Engine; memory estimated from the weights |
| Qwen2.5-Coder-7B-Instruct (Apple) | Code, chat | about 4.4 GB (M1: 4.0 GB on the Neural Engine in 4 bit tables chosen by GPTQ and the process's 0.35 GB) | 63 KiB/token (1.97 GiB at 32768) | measured on an M1 (0.8 tokens/s there, over its Neural Engine's 2.8 GiB: not picked on an M1); the M2 and later not measured |
| gpt-oss-20b | Chat, code, reasoning | AMD: 14.5 GB (13.0 GiB pinned for the NPU: MXFP4 experts as they are, 8-bit attention, 4-bit head). Apple (Core AI, M2 and later): 12.2 GB (the experts' 10.2 GB mapped from the checkpoint and run on the CPU, 1.44 GiB on the Neural Engine in 8 and 6 bit tables, 0.5 GB of the process); 17–19 tokens/s on a 16 GB M3 | AMD 48 KiB/token (0.38 GiB at its 8192); Apple 27 KiB/token (3.38 GiB at 131072) | measured on a Ryzen AI 9 365 and an M3 (16 GB) |
| Qwen3.6-35B-A3B (planned) | Chat, code | about 25 GB | — | estimated from 4-bit weights; not ported |

- The weights mix 4 and 8 bits (the MLP and output matrices at 4 bits, the attention projections at 8; for image generation, the DiT at 8 bits). Qwen3.5-0.8B and 2B fall apart on long English text at 4 bits, so their MLP and output matrices are at 6 bits
- "Measured" is the memory pinned for the NPU, measured on a Ryzen AI 9 365, plus the process's own. It follows from the size of the weights
- Apple's Neural Engine (Core AI, macOS 27) has its weights in 4, 6 and 8 bit tables (LFM3-1.7B, Gemma 4 and Qwen3.5-2B in 6 bits; LFM3's VL, Audio, Omni and d1 checkpoints, PLaMo 3 Translate 2B and 0.8B, Qwen3 and the other Qwen3.5 in 8; PLaMo 3 Translate 8B in 4). The program that generates and the one that reads prompts are functions of the same `.aimodel`, so the Neural Engine holds the weights once. With Core ML each of the two had its own copy: on an M3 LFM3-1.7B held 2.43 GiB on the Neural Engine and PLaMo 3 Translate 0.8B took about 2.4 GB in all, and on an M1 Qwen3-0.6B about 3.9 GB
- On an M2 or later, a model whose weights twice are within 2.5 GiB (LFM3-1.7B, Qwen3-0.6B, Qwen3.5-0.8B) also loads one whole program for short conversations (`--tier`), which holds the weights once more. An M1 does not load it (the split programs are faster there). On a machine of 8 GB LFM3-1.7B and Qwen3.5-0.8B do not load it either (models.yaml's `tier: none` and `variants`: LFM3-1.7B's 3.1 GB would be over what such a machine has for models). `--tier auto` loads it there, and `--tier none` leaves it out on a machine of 16 GB or more
- The values marked M1 were measured on an M1 of 16 GB on 2026-11-15 (docs/design.md 16.221): from the second start on, the peak of the process's footprint and the Neural Engine's share together, sampled every 0.1 s while it answers 200 tokens. The first start converts the model, and the process is larger for a while (at most 2.0 GiB for Qwen3-0.6B, 2.3 for PLaMo 3 Translate 0.8B, 2.7 for the 2B, 2.8 to 2.9 for LFM3's VL, Audio and Omni, 3.0 for Qwen3.5-0.8B and 6.0 GiB for Qwen3.5-2B). Gemma 4 E4B, Qwen3.5-4B and PLaMo 3 Translate 8B were not measured again with Core AI
- The image models marked M1 were measured the same way on 2026-11-16 (docs/design.md 16.228), drawing one picture from the second start on. Their peak is in the VAE's last stages at 1024 × 1024. models.yaml's values of Chroma1 are not measured again (Core ML's, the process alone)
- With Core AI the cache of a conversation also keeps the means of every 8 positions (the two-level attention of long conversations), so it grows by nine eighths of the table (13.5 KiB/token for LFM3-1.7B, 3.4 GiB at 262144; 6.75 KiB for Gemma 4 E2B)
- The Neural Engine's share is kernel memory of its driver, in no process's RSS or footprint (Activity Monitor and `top` do not show it). hwnpu-chat says how much it is after loading; a program compiled just then comes onto the Neural Engine at its first call
- Qwen3-0.6B and Qwen3-1.7B are split at their attention with Core AI as the other chat models are (all 28 of their layers attend; on an M1 Qwen3-0.6B generates 32 tokens/s and Qwen3-1.7B 22). Their context is 40960, the cache growing with the conversation, and cut to what a quarter of the memory holds (32768 on 16 GB). With Core ML they were one program with a cache of 4096 inside, about 3.9 GB for Qwen3-0.6B and 4.5 GB for Qwen3-1.7B (7.6 GB from 16 GB). Below 16 GB of memory Qwen3-1.7B runs as one program, the one that generates reading the prompts too, 8 tokens per call (models.yaml's `prefill: off` and its `variants`): with Core ML a new prompt of 1822 tokens took 16.8 s, 3.4 s with the program that reads prompts (`--prefill on` to have it anyway). With Core AI that program is a function over the same weights: it adds 0.16 GiB on the Neural Engine, 0.3 GB in all. AMD's NPU has no export for them yet
- Not counted: Gemma 4's per-layer embedding table (2.4 GB for E2B in int8, 4.7 GB in bf16 as converted before 2026-11-17; docs/improvements.md 12.7) and the token embedding tables (0.5 to 0.75 GB) stay files, and only the rows used are read. They go into the OS's page cache, which drops them when memory runs short
- Several exports of one model (text only, and with images and audio, for example) can share their files on the disk (Btrfs, XFS, APFS, ReFS). Memory is not shared: run at once, they count separately
