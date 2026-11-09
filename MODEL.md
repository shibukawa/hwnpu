# Recommended models

Which model hwnpu runs best for each use, by how much memory the machine has. The numbers are those of main on 2026-11-06. What each NPU computes in and how models are converted are in [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

Japanese: [MODEL.ja.md](MODEL.ja.md)

## By use and memory

The columns are the memory installed in the machine. The second header row is roughly what is left for models after the OS and everyday applications. NPU buffers are pinned and cannot be swapped out: going over that amount pushes other applications to swap.

hwnpu-chat and hwnpu-agent pick from this table themselves: the name under each use (`-use translate`, `-use chat`) runs the model of the machine's column, downloading only that one. A family picks the same way (`-use plamo3translate`, `-use gemma4`), and a tag chooses one whatever the memory, like a Docker image's: a model's (`-use plamo3translate:2b`, `-use chat:e4b`) or a column's (`-use chat:24g`, `-use plamo3translate:8g`: what a machine of that memory runs). `hwnpu-chat -list` shows what each name picks on the machine; `HWNPU_MEMORY=24` picks as a machine of 24 GB would. hwnpu-transcribe (`-use transcribe`), hwnpu-tts (`-use speech`) and hwnpu-systemone (`-use decision`) pick the same way. The table comes from [models.yaml](models.yaml) (its `categories` and the models' `machine`), which hwnpu embeds; a file of your own adds models, replaces a column's model or points a model to another (aliases), written as differences only: see [docs/model-table.md](docs/model-table.md).

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
<th>Chat, coding<br><sub>hwnpu-chat, hwnpu-agent <code>-use chat</code></sub></th>
<td><b>LFM3-1.7B</b> (2.2 GB)<br>fast, for agents<br><b>Gemma 4 E2B</b> (2.3 GB)<br>a size larger, 5B with its per-layer embeddings</td>
<td><b>Gemma 4 E2B</b> (2.3 GB)<br>with a long context too (+0.8 GB at 131072)<br><b>Gemma 4 E4B</b> (4.0 GB)<br>for short conversations</td>
<td colspan="2"><b>Gemma 4 E4B</b> (4.0 GB)<br>with a long context too (+2.1 GB at 131072)</td>
<td colspan="2"><b>Gemma 4 E4B</b> (4.0 GB)<br><i>planned: Qwen3.6-35B-A3B (MoE, about 25 GB at 4 bits)</i></td>
</tr>
<tr>
<th>Reading images<br><sub>hwnpu-chat <code>-use vision</code> (<code>image_url</code>, <code>/image</code>)</sub></th>
<td><b>LFM3-VL-2B</b> (2.6 GB)</td>
<td colspan="5"><b>Gemma 4 E2B with images and audio</b> (3.4 GB)<br><b>LFM3-VL-2B</b> (2.6 GB)</td>
</tr>
<tr>
<th>Transcription<br><sub>hwnpu-transcribe</sub></th>
<td colspan="6"><b>Whisper large-v3-turbo</b> (about 1.9 GB)<br>99 languages, with timestamps, 30 seconds read in 0.6 s</td>
</tr>
<tr>
<th>Reading audio (questions about it)<br><sub>hwnpu-chat <code>-use audio</code> (<code>input_audio</code>, <code>/audio</code>)</sub></th>
<td><b>LFM3-Audio-2B</b> (2.6 GB)</td>
<td colspan="5"><b>Gemma 4 E2B with images and audio</b> (3.4 GB)<br><b>LFM3-Omni-2B</b> (3.0 GB, images too)</td>
</tr>
<tr>
<th>Translation<br><sub>hwnpu-chat <code>-use translate</code></sub></th>
<td><b>PLaMo 3 Translate 0.8B</b> (about 2.4 GB)<br>3 times as fast as the 2B, translations differing from it by rewordings</td>
<td colspan="5"><b>PLaMo 3 Translate 2B</b> (2.7 GB)</td>
</tr>
<tr>
<th>Image generation<br><sub>hwnpu-image</sub></th>
<td>△ <b>Anima-Turbo</b> (4.1 GB)<br>with other applications closed</td>
<td colspan="5"><b>Anima-Turbo</b> 1024 × 1024 (4.1 GB)<br><b>Sana-Sprint 0.6B</b> (2.3 GB, AMD only): lighter and faster, English prompts</td>
</tr>
<tr>
<th>Speech<br><sub>hwnpu-tts</sub></th>
<td colspan="6"><b>LFM3-Audio-2B</b> (about 2.4 GB)</td>
</tr>
<tr>
<th>Sentence embeddings, search<br><sub>hwnpu-chat <code>-use embedding</code> (<code>/v1/embeddings</code>)</sub></th>
<td colspan="6"><b>PLaMo-Embedding-1B</b> (1.3 GB)</td>
</tr>
<tr>
<th>Decisions on a state<br><sub>hwnpu-systemone</sub></th>
<td colspan="6"><b>d1-2B</b> (about 2.4 GB, images too)<br><b>d1-Omni-2B</b> (about 2.8 GB, images and audio)</td>
</tr>
<tr>
<th>Several at once<br><sub>hwnpu-chat <code>-listen -use A,B,…</code></sub></th>
<td>one at a time</td>
<td>Gemma 4 E2B + PLaMo-Embedding-1B (3.6 GB)<br>+ PLaMo 3 Translate 2B (6.3 GB)</td>
<td>Gemma 4 E4B + PLaMo-Embedding-1B + PLaMo 3 Translate 2B (8.0 GB)</td>
<td colspan="3">and Gemma 4 E2B with images and audio, and Anima-Turbo in another process (about 15.5 GB)</td>
</tr>
</tbody>
</table>

- The figure in parentheses is the expected memory after starting and a short conversation (next section). Long conversations add the KV cache
- Past 32 GB there is no larger model yet. A machine with more memory runs the combinations above at once, or long contexts
- On a small machine, a shorter `-context` cuts the KV cache. hwnpu-chat shrinks the context itself when the KV cache would be over a quarter of the physical memory

## Memory per model

"At start" is the NPU's buffers (weights, work areas, the KV cache of the default context) plus the process itself (tokenizer, host buffers, about 0.4 GB). NPU buffers do not show in the process's RSS, so this is more than a task manager or `top` shows.

| Model | Use | At start | Growth with the conversation (KV cache) | Basis |
|---|---|---|---|---|
| LFM3-1.7B | Chat, coding | 2.2 GB (Apple's Neural Engine: about 3.0–3.3 GB) | 12 KiB/token (3 GiB at 262144) | measured (AMD and an M3) |
| Gemma 4 E2B | Chat, coding | 2.3 GB | 6 KiB/token (0.75 GiB at 131072) | measured |
| Gemma 4 E2B with images and audio | Images, audio (one image and 30 seconds a prompt) | 3.4 GB (2.4 GB until an image or audio arrives) | 6 KiB/token | measured |
| Gemma 4 E4B | Chat, coding | 4.0 GB | 16 KiB/token (2 GiB at 131072) | measured |
| LFM3-VL-2B | Images (512 × 512) | 2.6 GB | 12 KiB/token (1.46 GiB at 128000) | measured |
| Whisper large-v3-turbo | Transcription (any length, 30 seconds at a time) | about 1.9 GB (M3: 0.78 GiB on the Neural Engine and 1.1 GB of the process) | none (up to 448 tokens every 30 seconds) | measured on an M3 |
| LFM3-Audio-2B | Audio (clips up to 30 seconds) | 2.6 GB | 12 KiB/token | measured |
| LFM3-Omni-2B | Images and audio | 3.0 GB (2.2 GB until an image or audio arrives) | 12 KiB/token | measured |
| PLaMo 3 Translate 2B | Translation | 2.7 GB | 7 KiB/token (1.75 GiB at 262144) | measured |
| PLaMo 3 Translate 0.8B | Translation | 1.0 GB (M3: 1.38 GiB on the Neural Engine and 1.0 GB of the process, about 2.4 GB) | 1 KiB/token (0.28 GiB at 262144) | measured (AMD and M3) |
| PLaMo-Embedding-1B | Sentence embeddings | 1.3 GB | none | estimated from the weights |
| MoGe-2 ViT-S / B / L (AMD) | Metric depth, points, normals (hwnpu-image depth) | 1.9 GB (S), 2.1 GB (B), 2.6 GB (L) | none | measured |
| Depth Anything 3 Mono / Metric Large, Small, Base (AMD) | Depth estimation (hwnpu-image depth) | 1.4 GB (Large), 0.75 GB (Base), 0.5 GB (Small) | none | measured |
| Sana-Sprint 0.6B, Sana 600M, Sana-Sprint 1.6B, SANA 1.5 1.6B (1024 × 1024, AMD) | Image generation | 2.3 GB (the text graph, Gemma 2 in 8 bits; the DiT a group at a time, the VAE a stage at a time) | none | measured |
| Anima-Turbo (1024 × 1024) | Image generation | 1.9 GB (the VAE's largest stage; the DiT reads its blocks' weights a group at a time) | none | measured |
| LFM3-Audio-2B (speech) | Speech | about 2.4 GB | 12 KiB/token | estimated from the weights |
| d1-2B, d1-Omni-2B | Decisions on a state | about 2.4 GB, about 2.8 GB | short inputs only | estimated from the weights |
| Qwen3.5-0.8B | Chat (weak at Japanese facts) | 1.5 GB (6-bit MLP; M3: 1.47 GiB on the Neural Engine and 1.1 GB of the process, about 2.6 GB) | 14 KiB/token (3.38 GiB at 262144) | measured (AMD and M3) |
| Qwen3.5-2B | Chat, code | 2.4 GB (6-bit MLP; M3: 2.67 GiB on the Neural Engine in two programs of 6 bit tables and 1.5 GB of the process, about 4.2 GB; 1.32 GiB with `-prefill off`) | 14 KiB/token (3.38 GiB at 262144) | measured (AMD and M3) |
| Qwen3.5-4B | Chat, code | about 4.2 GB (3.9 GiB of 8 bit weights) | 32 KiB/token (8 GiB at 262144; 114688 tokens on 16 GB) | runs on an M3's Neural Engine; memory estimated from the weights |
| Qwen3.6-35B-A3B (planned) | Chat, code | about 25 GB | — | estimated from 4-bit weights; not ported |

- The weights mix 4 and 8 bits (the MLP and output matrices at 4 bits, the attention projections at 8; for image generation, the DiT at 8 bits). Qwen3.5-0.8B and 2B fall apart on long English text at 4 bits, so their MLP and output matrices are at 6 bits
- "Measured" is the memory pinned for the NPU, measured on a Ryzen AI 9 365, plus the process's own. It follows from the size of the weights
- Apple's Neural Engine holds more: its weights are 6 and 8 bit tables (LFM3-1.7B and Qwen3.5-2B in 6 bits; LFM3's VL, Audio, Omni and d1 checkpoints, PLaMo 3 and the other Qwen3.5 in 8), and hwnpu-chat loads two programs, the one that generates and the one that reads prompts, each with its own copy of the weights (one for Gemma 4 E4B, PLaMo 3 Translate 2B and Qwen3.5-4B). On an M3 (2026-11-07), LFM3-1.7B held 2.43 GiB on the Neural Engine (two programs of 1.21 GiB) plus the process's 0.5–0.9 GB, and PLaMo 3 Translate 0.8B 1.38 GiB in its two programs plus the process's 1.0 GB. The other models are not measured on Apple yet; expect them above the table as well
- The Neural Engine's share is kernel memory of its driver, in no process's RSS or footprint (Activity Monitor and `top` do not show it). hwnpu-chat says how much it is after loading; a program compiled just then comes onto the Neural Engine at its first call
- Not counted: Gemma 4's per-layer embedding table (4.4 GB for E2B) and the token embedding tables (0.5 to 0.75 GB) stay files, and only the rows used are read. They go into the OS's page cache, which drops them when memory runs short
- Several exports of one model (text only, and with images and audio, for example) can share their files on the disk (Btrfs, XFS, APFS, ReFS). Memory is not shared: run at once, they count separately
