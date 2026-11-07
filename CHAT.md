# Chat, embeddings, an OpenAI compatible API and OpenCode

[日本語](CHAT.ja.md)

Run LFM3-1.7B and PLaMo-Embedding-1B on the Apple Neural Engine alone, and use them in the terminal or as an OpenAI compatible API from clients such as OpenCode. No Python is needed.

The same commands run on Intel's NPU (Core Ultra), where the models are converted to OpenVINO IR and compiled by the NPU driver; this is not yet tried on the hardware. See [INSTALL_INTEL.md](INSTALL_INTEL.md) section 6.

`-use` picks the models by name, and `-list` shows the names and what each model is for:

```bash
go run ./cmd/hwnpu-chat -list
```

| Name | Kind | Features | in the terminal | with `-listen` (section 3) |
|---|---|---|---|---|
| `LFM3-1.7B` (default) | chat | `tools`, `thinking` | talk with it (section 1) | `/v1/chat/completions` |
| `LFM3-VL-2B` | chat | `tools`, `thinking`, `vision` | talk with it, with images (section 1) | `/v1/chat/completions` |
| `LFM3-Audio-2B` | chat | `tools`, `thinking`, `audio` | talk with it, with audio (section 1) | `/v1/chat/completions` |
| `LFM3-Omni-2B` | chat | `tools`, `thinking`, `vision`, `audio` | talk with it, with images and audio (section 1) | `/v1/chat/completions` |
| `gemma-4-E2B-it` | chat | `tools`, `thinking` | talk with it (section 1) | `/v1/chat/completions` |
| `gemma-4-E4B-it` | chat | `tools`, `thinking` | talk with it (section 1) | `/v1/chat/completions` |
| `Qwen3.5-4B` | chat | `tools`, `thinking` | talk with it (section 1) | `/v1/chat/completions` |
| `plamo-embedding-1b` | embedding | `query` | search texts by their embeddings (section 4) | `/v1/embeddings` |

The features say what a model can do besides its kind (names as in Ollama's capabilities):

| Feature | |
|---|---|
| `tools` | calls the tools a request offers (`tools`, `tool_calls`), so that agents such as OpenCode can drive it (section 6). LFM3-1.7B is text only, and small and fast enough to run behind an agent |
| `thinking` | thinks before it answers (`reasoning_content`); `-think` and `reasoning_effort` turn it on and off |
| `query` | embeds search queries and the documents searched differently (`input_type`, section 4) |
| `vision` | reads the images of user messages (`image_url`); `/image FILE` in the terminal |
| `audio` | reads the audio of user messages (`input_audio`, WAV); `/audio FILE.wav` in the terminal |

- The terminal runs one model, chosen by `-use`; what it does follows the model's kind. The server runs any number, `-use LFM3-1.7B,plamo-embedding-1b` for example
- Names are matched ignoring case, and the catalog's repository names work too. A Hugging Face repository outside the catalog, `-use OWNER/NAME`, works when its architecture is one hwnpu-chat runs (LFM3 for chat, PLaMo's encoder for embeddings): its `config.json` tells which. Set `HF_TOKEN` for gated repositories

## 1. Chat

```bash
go run ./cmd/hwnpu-chat
```

No arguments are needed. The first run does the following by itself (about 4 minutes on an M3, besides the download: a little over two minutes of conversion and two of compiling). Later runs only load the compiled programs and start in a few seconds to a minute, showing how far loading is.

1. It downloads `config.json`, `tokenizer.json` and `model.safetensors` (3.5 GB) of [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) from Hugging Face to `~/Library/Caches/hwnpu/models/LFM3-1.7B`. A download cut off goes on from where it stopped on the next run
2. It converts the model for the Neural Engine, with 6 bit tables and the grouped attention, split at its full attention layers (`lfm3-lut6-grouped-p8192-c262144-q8.parts.json` and `-q128-last`, 7 parts of about 1.2 GiB together each and a program without weights for the attention): one generates, 8 tokens per call (a token and 7 guesses checked at once, section 3), the other reads prompts 128 tokens per call, computing the logits of the last row alone. It reads about a fifth faster than the `-q64` (64 tokens per call) of versions until November 6, 2026, which is no longer read and can be deleted (section 16.82 of docs/design.md). The two share one KV cache. hwnpu-chat runs the full attention layers' attention itself, over the cache it holds, in chunks of 8192 tokens (`-paged`, section 2): a step then costs in proportion to the conversation, and the cache's memory grows with it, so the context is the model's longest, 262144 tokens. The default until November 4, 2026 (`lfm3-lut6-grouped-c16320-q8.mlpackage`, a context of 16320, every step costing in proportion to it) is no longer read and can be deleted, as can the 4 bit `lfm3-c16320-q8.mlpackage` of before. Against the model in bf16, lut6 is six times closer than lut4 was (KL 0.23 nats to 1.39), a coding task takes 6 rounds where it took 13 to 27, and with the grouped attention a call is still 38% faster than before (section 16.41 of [docs/design.md](docs/design.md))
3. It compiles them for the Neural Engine (1 to 2 minutes). The result stays in `~/Library/Caches/hwnpu`. Files of the same weights (the programs that generate and that read prompts, and what they compile to) share one copy on the disk on macOS (APFS clones, ordinary files still), and at start the compiled programs whose source is deleted or that went unused for 30 days are removed (section 16.64 of docs/design.md)

Then type after the `>`.

- The model thinks first (`<think>`, shown dim) and then answers. `/think off` makes it answer at once, `/think on` brings thinking back
- `/reset` clears the conversation, `/exit` or Ctrl-D quits, Ctrl-C stops an answer. A line ending in `\` continues on the next line
- The input is edited as in a shell: ↑ ↓ (Ctrl-P, Ctrl-N) bring back what was typed before (up to 1000 entries kept in `~/Library/Caches/hwnpu/history/chat`, so also after a restart), and ← →, Home, End, Ctrl-A, Ctrl-E, Ctrl-K, Ctrl-U and Ctrl-W work. Japanese and other wide characters take two columns. A pasted text of several lines comes in as one input, its line breaks shown as `⏎`. Ctrl-C while typing clears the line; on an empty line it quits
- Loading and help are shown in gray, with `all on the NPU` in green when everything is on the Neural Engine and `NOT all on the NPU` in red when not. No colors when not on a terminal or with `NO_COLOR` set (hwnpu-agent the same)
- From the second turn on, only what is new since the last answer is read
- After each answer it shows the tokens read and the speed of generation (about 15 tokens/s on an M3)

### Images and audio (LFM3-VL-2B, LFM3-Audio-2B, LFM3-Omni-2B)

`-use LFM3-VL-2B`, `-use LFM3-Audio-2B` or `-use LFM3-Omni-2B` runs one of Liquid AI's multimodal checkpoints. The first run downloads it (3.8, 4.0 and 4.3 GB) and converts its text model the same way, but with 8 bit tables (lut8, 2 parts each): with 4 bit tables these models describe pictures wrongly and mishear speech (docs/design.md, 16.31). It also converts the encoders the checkpoint has, in fp16: the vision encoder for images of 512x512 (an image is stretched to it) and the audio encoder for clips of up to 30 seconds.

```bash
go run ./cmd/hwnpu-chat -use LFM3-VL-2B
```

- In the terminal, `/image FILE question` sends an image and `/audio FILE.wav question` a clip. The server reads the `image_url` (data URLs) and `input_audio` (base64 WAV) parts of user messages
- About 36 ms per token on the M3 with the grouped attention (the default since November 2026; 41 ms before), and about 70 ms while it describes a picture, where few of the speculative drafts come true. An image prompt of about 300 tokens reads in 0.7 seconds. Like LFM3-1.7B these models get no 4096 tier (section 2, `-tier`)
- The text model is split at its full attention layers as LFM3-1.7B's is (`lfm3-lut8-grouped-p8192-c128000-q8.parts.json` and `-q128-last`, 7 parts each; the `-q64` of versions until November 6 can be deleted), with the model's longest context, 128000 tokens; its cache grows with the conversation by 12 KiB a token, to 1.46 GiB at 128000. On the M3 it is all on the Neural Engine, a second start loads in 1.4 and 4.2 seconds, and its answers on a picture, a sum and Mount Fuji's height were the whole model's (a context of 16320) token for token, at 24.0, 33.2 and 36.0 tokens/s (the whole model: 11.9, 26.7 and 29.4). It recalled a passphrase at the start of prompts of 16K, 64K and 120K tokens, read in 20, 171 and 497 seconds, generating at 25.2, 58.6 and 68.1 ms a token at those lengths. LFM3-Audio-2B and LFM3-Omni-2B run the same way (transcribing a clip, describing a picture) and recalled the passphrase at 120K (read in 525 and 512 seconds, a process footprint of 1.7 GB). 2026-11-04, sections 16.63 and 16.65 of docs/design.md. That was reading the whole cache; with the two-level attention (`-sparse`, the default) 120K reads in 431 seconds for VL, 516 for Audio and 501 for Omni (VL reading the whole cache the same day: 583), a call at that length takes 111 to 120 ms (179), and all three recalled the passphrase (2026-11-06, 16.65). The `lfm3-lut8-grouped-c16320-*` files of earlier versions are no longer read and can be deleted
- Answers are text; LFM3-Audio's speech output is `hwnpu-tts` (section 5)
- Other image sizes or clip lengths: convert the encoders yourself and give them with `-package`, `-vision` and `-audio` (section 5)

### Gemma 4 E2B and E4B

`-use gemma-4-E2B-it` runs Google's Gemma 4 E2B (instruction tuned) instead. It goes the same way: `model.safetensors` (10.2 GB, with the audio and vision parts, of which only the text model is converted) is downloaded to `~/Library/Caches/hwnpu/models/gemma-4-E2B-it`, and the programs of 8 and 64 tokens per call are converted with 6 bit tables (lut6), split at the full attention layers as LFM3-1.7B's are (`gemma4-lut6-p8192-c131072-q8.parts.json` and `-q128-last`, 8 parts each, about 1.6 GiB, and a program without weights for the attention; prompts are read 128 tokens per call and the last 20 layers, which share earlier keys and values, compute the last token's row alone, so 16K reads in 13 seconds rather than 27.8 and 64K in 95 rather than 205 (sections 16.82 and 16.83), and the `-q64` of versions until November 6 can be deleted). The first conversion takes about 3 minutes on an M3 and the compilation about 3 more; later starts take about 2 seconds (the whole model in 2 parts did not stay in the Neural Engine's compile cache and took 50 to 100 seconds at every start: sections 16.42, 16.43 and 16.63 of docs/design.md). The `gemma4-lut6-c16320-*` files of versions until 2026-11-04 and the earlier `gemma4-c16320-*` are no longer read and can be deleted.

```bash
go run ./cmd/hwnpu-chat -use gemma-4-E2B-it
```

- The chat format is Gemma 4's `chat_template.jinja` in Go, with thinking (`<|think|>`, returned as `reasoning_content`) and tool calls
- The per-layer embedding table (4.7 GB) stays on the host, which reads one row (17.5 KB) per token
- The context is the model's longest, 131072 tokens. Of its 7 full attention layers, 4 use layer 14's keys and values, so the cache hwnpu-chat holds is 3 layers' and grows with the conversation by 6 KiB a token (0.75 GiB at 131072). Measured on the M3 (section 16.63 of docs/design.md): 23.6 to 36.0 tokens/s answering short questions (the whole model with a context of 16320: the same answers at 13.3 to 28.2); a prompt of 81,691 tokens read in 299 seconds (273 tokens/s), the passphrase at its start recalled, 96.7 ms a token at that length, with a process footprint of 1.1 GB. At 16K, 64K and 125K it recalled the passphrase too, reading in 27, 224 and 709 seconds and generating at 23.9, 82.4 and 119.5 ms a token, with 0.7, 1.2 and 2.4 GB of memory, reading the whole cache; with the two-level attention (`-sparse`, the default) 64K and 125K read in 204 and 521 seconds and generate at 73.9 and 92.4 ms a token, recalling the passphrase too (section 16.65 of docs/design.md)
- 4 bit tables (lut4) cost Gemma 4 much more than LFM3 (it got Mount Fuji's height and the author of Kokoro wrong), so it converts with 6 bit tables by default; `hwnpu-convert` does too when `-format` is left out. See sections 16.25 and 16.26 of docs/design.md
- `-use gemma-4-E4B-it` runs the larger E4B (16.0 GB download, 42 layers of width 2560) the same way, slower than E2B. Its context is 131072 too, its cache 16 KiB a token (2 GiB at 131072). Its two programs (generating and reading prompts, 6.5 GiB) would not stay in the Neural Engine's compile cache (about 4 GiB), so only the one that generates (8 tokens per call) is converted and loaded, and it reads the prompts too (`-prefill auto`). On the M3 later starts take about 4 seconds and short questions generate at 13 to 16 tokens/s, but a long prompt reads 4.4 times slower than with two (16K in 183 seconds against 42). `-prefill on` gives it the two, at about 80 seconds every start and 1 to 3 seconds of switching every question. It recalled the passphrase at 125K (with two), but swap reached 14 GB (section 16.65 of docs/design.md). With one and the two-level attention (the default) it recalled the passphrase at 64K and 125K too, a call taking 217 and 276 ms at those lengths, but the prompts read in 21 and 56 minutes. On a 16 GB machine, E2B is the better choice. The larger Gemma 4 sizes (12B and up) are another architecture, which hwnpu-chat does not run

### Translating with PLaMo 3 Translate

Preferred Networks' translation model PLaMo 3 Translate 2B is `-use plamo-3-translate-2b`. As the other models, the first run downloads the checkpoint (5.8 GB) and converts it: 8 bit tables (lut8) for the Neural Engine, int8 for Intel's NPU, split at its 3 full attention layers (7, 15 and 23; a context of 262144 tokens, the cache growing with the conversation by 7 KiB a token). On AMD's NPU, the IRON export runs with `-iron` (the README's AMD section). Each text typed is translated, and the earlier source texts and translations go into the prompt as its translation history (keeping terms and style consistent). `-to` sets the target language (by default English for a text with Japanese characters, else Japanese); `/to LANG` changes it and clears the history. Languages are lowercase codes: `en`, `ja`, `zh-hans`, `ko`, `ja(easy)` (plain Japanese) and so on. `-from` sets the source language, `-form plain` or `polite` the style of Japanese, and `-system` becomes the reference context. The model neither thinks nor calls tools.

`-use plamo-3-translate-0.8b` is the same model in a smaller size (download 1.8 GB). On an M3 it generates about 3 times as fast as the 2B (53 against 17 tokens/s), and its two programs (8 and 128 tokens per call, 1.38 GiB together) fit the Neural Engine's compile cache, so it reads a document of 1600 tokens in 1.2 seconds. Of 20 sentences (English to Japanese and Japanese to English), 8 came out word for word as the 2B's and the other 12 were correct rewordings.

```bash
go run ./cmd/hwnpu-chat -use plamo-3-translate-2b
go run ./cmd/hwnpu-chat -use plamo-3-translate-2b -to ja -form plain
go run ./cmd/hwnpu-chat -iron ~/models/plamo-3-translate-2b.amd-npu2-q4
```

The weight formats (design.md 16.71): on an M3's Neural Engine, the next token is fp16's at 98% of the positions with lut8, 89% with lut6; a call takes 57.8 ms with lut8, 53.7 with lut6. lut4 is unusable (KL 1.91 nats on the CPU; Intel's int8 0.0025). The two programs of lut8 (4.9 GiB) are over the Neural Engine's compile cache (about 4 GiB), so the one that generates (8 tokens per call) reads the prompts too: short texts hardly feel it, a document of thousands of tokens is read about ten times slower. `-format lut6` favours speed, with two programs.

The server takes `target_lang`, `source_lang`, `form`, `context` and `glossary` (`[{"source": "NPU", "target": "NPU"}]`) in `chat_template_kwargs`, as vLLM does. `messages` alternate source texts (user) and translations (assistant), ending with the text to translate.

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"The meeting has been rescheduled."}],"chat_template_kwargs":{"target_lang":"ja"},"temperature":0}'
```

## 2. Options

| Flag | Default | |
|---|---|---|
| `-use NAMES` | `LFM3-1.7B` | the models to run, comma separated: names from `-list` or repositories `OWNER/NAME`; one in the terminal |
| `-list` | | list the models, what each is for, and whether it is downloaded |
| `-listen ADDR` | none | serve the OpenAI compatible API at `ADDR` (e.g. `127.0.0.1:8080`) instead of the terminal (section 3) |
| `-think` | `true` | let the model think before it answers |
| `-think-budget N` | `1024` | most tokens the model thinks before it is made to answer (0: no limit) |
| `-tier A,B` | `auto` | caches shorter than `-context` to run on while a conversation is short: it starts on the shortest that holds its prompt and moves up as it grows (a step costs in proportion to the cache, whatever it holds); empty for none. `auto` is `4096` for a model in 4 bit tables and none for 6 or 8 bits: the Neural Engine keeps its compiled programs in a cache of its own of about 4 GiB, and four programs of LFM3-1.7B in lut6 (4.8 GiB) push each other out of it, so every start compiles some of them again: four starts in a row took 110, 92, 77 and 62 seconds, while two programs (2.4 GiB) settle at 3 seconds a start |
| `-draft N` | `8` | tokens per call of the program that generates: a token and N-1 guesses checked at once (speculative decoding, section 3); `0` or `1` generates one token per call |
| `-prefill auto` | `auto` | whether a program of 64 tokens per call reads the prompts beside the one that generates. `auto` has one unless the two programs' weights would be over the Neural Engine's compile cache (about 4 GiB; Gemma 4 E4B), and then the one that generates reads the prompts (slower for long ones). `on` always, `off` never |
| `-prompt-cache DIR` | `prompts` in the compile cache | keep the prompts read on disk, so that a prompt beginning like an earlier one (an agent's system message and tools) is not read again after a restart (section 3); `off` for none |
| `-system TEXT` | none | the chat's system message (a translation model's reference context) |
| `-to LANG` | none | the language a translation model (PLaMo 3 Translate) translates into; none: `en` for a text with Japanese characters, else `ja` |
| `-from LANG` | none | the source language of a translation model (none: the model finds it) |
| `-form plain\|polite` | none | the style of a translation into Japanese |
| `-n N` | `0` | most tokens of an answer (0: until the context is full) |
| `-temp`, `-top-k`, `-top-p`, `-rep` | `0.1`, `50`, `0`, `1.05` | sampling (temperature, top k, top p, repetition penalty within the answer) |
| `-context N` | `0` | most tokens of a prompt and its answer together. `0`: the model's longest when it is split at its attention (`-paged`; 262144 for LFM3-1.7B and Qwen3.5, 128000 for LFM3-VL, Audio and Omni, 131072 for Gemma 4), else 16320. Another value converts the model again for it; a model not split holds its cache in chunks over 16320 |
| `-paged N` | `0` | split the model at its full attention layers and attend over the cache hwnpu-chat holds in chunks of N tokens: a step costs in proportion to the conversation, not to `-context`, and the cache's memory grows with it (12 KiB a token for LFM3-1.7B, 3 GiB at 262144; the context is cut when that would be over a quarter of the memory). `0` is 8192 where the model allows it (LFM3 and its VL, Audio and Omni checkpoints, Gemma 4, Qwen3.5, on the Neural Engine and on Intel's NPU), `-1` never (then `-tier` applies) |
| `-sparse` | `true` | past about 48K tokens of a model split at its attention (`-paged`), read in each call 16384 keys exactly (32768 while reading a prompt: those of the 8-token buckets that weigh most for the call, the last 512 and the first 4) and every other bucket as the mean of its keys and values: a few times fewer calls at 128K and over ([docs/design.md](docs/design.md) 16.61). `false` reads the whole cache on every call |
| `-dir DIR` | `~/Library/Caches/hwnpu/models` | where the models are kept, each in `DIR/NAME` with its checkpoint and converted programs |
| `-package A,B` | none | chat programs you converted with `hwnpu-convert` (section 5) |
| `-dump DIR` | none | with the server, save each request, its prompt and the generated text |
| `-embed-seq A,B` | `128,1024` | lengths an embedding model is converted for; a text runs on the shortest that holds it, and the longest is the most tokens a text can have |
| `-embed-package A,B` | none | embedding programs you converted with `hwnpu-convert` (encoders take `-seq` only), with `config.json` and `model.safetensors` in the model's directory |

The flags from `-think` to `-context`, `-package`, `-config`, `-weights` and `-tokenizer` are for chat models, the `-embed-` flags for embedding models. `-package` and its companions (and `-embed-package`) take one model of their kind in `-use`.

- With LFM3-1.7B split at its attention (`-paged` 8192, the default), a step costs in proportion to the conversation: a context of 262144 does not slow a short one. Measured on an M3 (16 GB) (sections 16.58 and 16.59 of [docs/design.md](docs/design.md)); a short conversation generates 1.3 to 1.7 times as fast as with the model not split (`-paged -1 -context 16320`), with the same answers word for word:

| conversation | generation (M3) | reading the prompt | memory of the process | `-sparse` (the default): generation | reading | memory |
|---|---|---|---|---|---|---|
| a few hundred | 28 to 33 ms/token (with the drafts accepted) | — | — | (the same) | — | — |
| 16K | about 43 ms/token | 19 seconds | about 0.4 GB | (the same) | (the same) | about 0.5 GB |
| 64K | about 93 ms/token | 2.5 minutes | about 1.0 GB | about 78 ms/token | 2.9 minutes | about 1.6 GB |
| 128K | about 165 ms/token | 8.6 minutes | about 1.8 GB | about 90 ms/token | 7.8 minutes | about 2.5 GB |
| 250K | about 320 ms/token | 29 minutes | about 3.2 GB | about 126 ms/token | 20 minutes | about 4.1 GB |

- The cache grows by 12 KiB a token, to 3 GiB at 262144. At every length the model recalled a passphrase put at the start of the prompt
- A prompt of 250K tokens read at once takes about half an hour. Prompts read are kept on disk (`-prompt-cache`), so the second time reads nothing
- Past about 48K tokens the two-level attention (`-sparse`, the default) reads fewer keys: at 128K a call of the attention layers is 7 calls over chunks of 8192 instead of 16, at 256K 11 instead of 32. The `-sparse` columns of the table above are measured on the M3 (2026-11-06, section 16.61 of docs/design.md), the three before them reading the whole cache (`-sparse=false`) on November 4. Against the whole cache measured again the same day, a call takes about half as long at 128K and 0.42 times at 250K, and a prompt reads in 0.76 and 0.56 times the time; only at 64K is the reading a tenth slower. Memory grows by the summaries and the copies of the keys read exactly, the cache by 14 KiB a token (3.38 GiB at 262144). Checked on the CPU with LFM3-1.7B at 128K (pass phrases at 10%, 50% and 90% of the context all recalled, the next-token loss up by 0.007 to 0.009 nats); on the M3 too it recalled the three in a document of 128K. Past 128K, LFM3-1.7B itself misses passphrases far back in a long document (in the CPU check at 256K it recalled only the one at 90%), with or without `-sparse`
- A model not split (`-paged -1`) costs in proportion to its cache whatever it holds: the longer the context, the slower each token, and with `-tier 4096` (the default for 4 bit tables) a conversation runs on a cache of 4096 while it fits, about twice as fast, and moves up to `-context` as it grows; the programs of each tier are converted and loaded too. OpenCode needs 16K or more

| `-context` | generation (M3) | reading a prompt | KV cache |
|---|---|---|---|
| 4096 | about 28 ms/token | — | about 50 MiB |
| 16320 (default) | about 64 ms/token | about 570 tokens/s | about 190 MiB |
| 32768 | about 138 ms/token | about 300 tokens/s | about 390 MiB |
| 65536 | about 363 ms/token | about 150 tokens/s | about 770 MiB |

- Over 16320, the cache is held in chunks of 16384 tokens, to stay within the Neural Engine's 16384 per tensor dimension. With 32768 and 65536 the model recalled a passphrase put at the start of prompts of 25,288 and 53,736 tokens
- The server keeps up to two other conversations as copies of the part of the cache each used (of the whole cache until November 4, 2026)

## 3. OpenAI compatible server

```bash
go run ./cmd/hwnpu-chat -listen 127.0.0.1:8080
```

```bash
go run ./cmd/hwnpu-chat -listen 127.0.0.1:8080 -use LFM3-1.7B,plamo-embedding-1b
```

| Endpoint | |
|---|---|
| `GET /v1/models` | the names of the models loaded (`LFM3-1.7B`, `plamo-embedding-1b`), their `kind` (`chat` or `embedding`), `capabilities` (the features, for a model of `-list`), `context_length`, and `dimensions` of an embedding model |
| `POST /v1/chat/completions` | `stream` (SSE), `tools` and `tool_calls`, `reasoning_content`, `max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop`, `stream_options.include_usage` |
| `POST /v1/embeddings` | `input` (a string or a list), `encoding_format` (`float` or `base64`), `dimensions` (only the model's 2048), and besides OpenAI's: `input_type` (`query` or `document`, section 4) and `truncate` |

- `-think` (default true) decides whether the model thinks; a request changes it with `reasoning_effort` (none, minimal and low turn it off, medium and high on) or `chat_template_kwargs.enable_thinking`
- `-think-budget` (default 1024) is the most tokens of thinking: past it, the thinking is closed with a sentence saying so and the model answers from what it has thought. A small model otherwise thinks for thousands of tokens (minutes at 64 ms a token), and an agent waits meanwhile; the log says `thinking cut` when it happened. `0` leaves the thinking unbounded
- Requests run one at a time. The beginning a prompt shares with an earlier one is not read again, and `usage.prompt_tokens_details.cached_tokens` says how much. When another conversation comes in between, up to two long conversations are kept aside and come back
- The prompts read are also kept on disk (`-prompt-cache`, `~/Library/Caches/hwnpu/prompts` by default). After a restart, a prompt that begins like one of them comes back from disk at the last mark inside the shared beginning (a mark every 512 tokens and at the end of each prompt) and only the rest is read; the log says `N tokens from the prompt store`. Up to 4 sequences of 1024 tokens or more are kept per model and conversion, about 19 KiB per token for LFM3-1.7B (12 KiB of keys and values of the full attention layers per token, and 3.5 MB of rings, windows and logits per mark; 145 MB for the 7.9K tokens of OpenCode's first request), the least recently used going first. On an M3, that request reads in 14.6 seconds cold and comes back from the store in 0.2 seconds. `-prompt-cache off` turns it off
- Generation stops when the model starts going round in circles: one short pattern over and over, or the same lines (or runs of tokens) coming back a third time and more, which a small model asked what it does not know falls into ("- Gifu? Already." "- Mie? Already." ...). An answer that writes out its draft once more is not stopped. In the thinking, the thinking is closed as at `-think-budget` and the model answers; the log says `stopped repeating itself` or `thinking cut going round in circles`
- `-dump DIR` saves each request, its prompt and the generated text, for finding out what a client sends
- A request goes to the model its `model` names (ignoring case) among the endpoint's kind. When only one model of that kind is loaded, it answers whatever the name; with several, an unknown name answers 404. A model of the other kind (`plamo-embedding-1b` on `/v1/chat/completions`) is an error, and so is an endpoint with no model of its kind loaded

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"lfm3","messages":[{"role":"user","content":"What is the capital of Japan?"}]}'
```

## 4. Embeddings (PLaMo-Embedding-1B)

```bash
go run ./cmd/hwnpu-chat -use plamo-embedding-1b
```

[pfnet/plamo-embedding-1b](https://huggingface.co/pfnet/plamo-embedding-1b) turns a Japanese (or English) text into a vector of 2048 numbers; texts of similar meaning get vectors pointing the same way. The first run downloads `config.json`, `tokenizer.model` and `model.safetensors` (2.1 GB) to `~/Library/Caches/hwnpu/models/plamo-embedding-1b` and converts the model once for each length of `-embed-seq`, like the chat model.

In the terminal, each line typed is a document kept for searching, and a line starting with `?` is a query: the documents are listed by their cosine similarity to it, most similar first. `/list` shows the documents, `/reset` clears them, `/exit` or Ctrl-D quits.

```text
> 東京は日本の首都です。
> 最近は随分と暖かくなりましたね。
> ? 日本の首都はどこ？
```

Over the API (with `-listen` and `plamo-embedding-1b` in `-use`):

```bash
curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' -d '{"model":"plamo-embedding-1b","input":["東京は日本の首都です。","最近は随分と暖かくなりましたね。"]}'
```

```bash
curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' -d '{"model":"plamo-embedding-1b","input":"日本の首都はどこ？","input_type":"query"}'
```

- PLaMo-Embedding embeds a search query and what is searched differently: a query gets the instruction `次の文章に対して、関連する文章を検索してください: ` before it (the model's `encode_query`), a document nothing (`encode_document`). On the API, `input_type: "query"` (or `search_query`) makes the inputs queries; the default is `document`, which is also the one for texts that are not about search (clustering, classification). OpenAI's API has no such field, so a client that cannot send it gets documents
- The embedding is the mean of the last hidden state over the text's own tokens (not the BOS, not the instruction), scaled to length 1, so the dot product of two embeddings is their cosine similarity
- A text runs on the shortest converted length that holds it, padded. A text over the longest (1024 tokens by default) is an error (`context_length_exceeded`) unless the request says `"truncate": true`; in the terminal it is cut, with a note. The model reads up to 4096 tokens: `-embed-seq 128,1024,4096` converts that length too (the attention over 4096 tokens has not been tried on the Neural Engine)
- Requests run one at a time, one text after another. The chat and the embedding model run independently of each other

## 5. Programs you converted

To try another quantization or cache length, convert twice with `hwnpu-convert` with the same cache length, and list both in `-package`.

```bash
go run ./cmd/hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -quant kmeans-minmax -seq 1 -cache 16320 -parts 2 -o DIR/lfm3-16k-q1.mlpackage
```

```bash
go run ./cmd/hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -quant kmeans-minmax -seq 64 -cache 16320 -o DIR/lfm3-16k-q64.mlpackage
```

```bash
go run ./cmd/hwnpu-chat -package DIR/lfm3-16k-q1.parts.json,DIR/lfm3-16k-q64.mlpackage -config DIR/config.json -weights DIR/model.safetensors
```

### Images (LFM3-VL-2B)

`-use LFM3-VL-2B` (section 1) does all of this by itself. For other image sizes, convert the text model as above and the vision encoder with `-arch lfm3-vision`, once per image size (the Neural Engine runs fixed shapes). List them in `-vision`, comma separated, and the `image_url` parts (data URLs) of user messages are read. An image is stretched to the size of the program with the closest aspect ratio. In the terminal, `/image FILE question` sends an image (docs/design.md, 16.28 and 16.31).

```bash
go run ./cmd/hwnpu-convert -arch lfm3-vision -image 512x512 -format fp16 -config DIR/config.json -weights DIR/model.safetensors -o DIR/vision-512x512.mlpackage
```

```bash
go run ./cmd/hwnpu-chat -package DIR/lfm3vl-q1.parts.json,DIR/lfm3vl-q64.mlpackage -vision DIR/vision-512x512.mlpackage -config DIR/config.json -weights DIR/model.safetensors
```

### Audio (LFM3-Audio-2B)

`-use LFM3-Audio-2B` (section 1) does all of this by itself. For other clip lengths, convert the text model as above and the audio encoder with `-arch lfm3-audio-encoder` (`-frames` is the longest clip in mel frames of 10 ms; the default 3000 is 30 seconds). List the packages in `-audio`, comma separated, and the `input_audio` parts (base64 WAV) of user messages are read. In the terminal, `/audio FILE.wav question` sends a clip. Only WAV is read; other sample rates than 16 kHz are converted. Answers are text; `hwnpu-tts` below speaks a text (docs/design.md, 16.29 and 16.31).

```bash
go run ./cmd/hwnpu-convert -arch lfm3-audio-encoder -frames 3000 -format fp16 -config DIR/config.json -weights DIR/model.safetensors -o DIR/audio-30s.mlpackage
```

### Speech (LFM3-Audio-2B TTS)

`hwnpu-tts` speaks a text into a WAV file (24 kHz). Convert the text model with a KV cache and `-hidden-out 1` (it then outputs its final hidden state as well), the depthformer (`-arch lfm3-depthformer`) and the codec decoder of `audio_tokenizer/` (`-codec-frames`). Spoken chat answers are not supported yet. Not yet checked on the real checkpoint (docs/design.md, 16.30).

```bash
go run ./cmd/hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -seq 1 -cache 2048 -hidden-out 1 -o DIR/tts-q1.mlpackage
```

```bash
go run ./cmd/hwnpu-convert -arch lfm3-depthformer -format fp16 -config DIR/config.json -weights DIR/model.safetensors -o DIR/depth.mlpackage
```

```bash
go run ./cmd/hwnpu-convert -codec-frames 64 -format fp16 -config DIR/audio_tokenizer/config.json -weights DIR/audio_tokenizer/model.safetensors -o DIR/codec.mlpackage
```

```bash
go run ./cmd/hwnpu-tts -model DIR -text DIR/tts-q1.mlpackage -depth DIR/depth.mlpackage -codec DIR/codec.mlpackage -lang en -o out.wav "Hello from the Neural Engine."
```

## 6. Use it from OpenCode

Started like this in the project directory, it runs the server and OpenCode together.

```bash
go run ./cmd/hwnpu-chat -opencode
```

- After loading the model it serves on a free port from 30000 to 49151 (below 49152, where macOS takes ports for outgoing connections), writes an OpenCode config for that server to `~/Library/Caches/hwnpu/opencode/opencode.json` and hands it to OpenCode with the `OPENCODE_CONFIG` environment variable. Nothing is written into the project
- OpenCode gets this terminal's standard input, output and error as they are, so its screen, mouse and keys work as when it is started alone. The server's log goes to `~/Library/Caches/hwnpu/opencode/server.log`, off the screen
- When OpenCode exits, the server stops, and hwnpu-chat ends with OpenCode's exit code
- The arguments after the flags go to OpenCode; put OpenCode options that start with `-` after `--` (e.g. `hwnpu-chat -opencode -- --continue`, `hwnpu-chat -opencode run "fix the bug in main.go"`). `-listen` serves at that address instead, and `-opencode-bin` names another OpenCode command
- `opencode run` waits for its standard input to close; from a script, add `< /dev/null`
- OpenCode puts a project's `opencode.json` over `OPENCODE_CONFIG`: a provider `hwnpu` written there wins, with its address

To set it up by hand, put an `opencode.json` in the project directory. The model's name is the one `GET /v1/models` returns.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "hwnpu": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "hwnpu (Neural Engine)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "LFM3-1.7B": {
          "name": "LFM3 1.7B on the Neural Engine",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 262144, "output": 4096 }
        }
      }
    }
  },
  "model": "hwnpu/LFM3-1.7B",
  "permission": {
    "bash": { "*": "ask", "go *": "allow", "ls*": "allow", "cat *": "allow", "pwd": "allow" }
  }
}
```

- Set `limit.context` to the server's context (`ready: context N tokens` at the start). OpenCode's requests take about 7.4K tokens for the system prompt and the tools alone, so with a context of 8K it starts compacting after one turn
- A 1.7B model sometimes writes commands that make no sense. Allowing only some commands for `bash` is safer
- An example on an M3: asked to fix the bug in `add` of main.go, it read the file, changed `a - b` to `a + b` and checked with `go run`, in 12 requests. The first request read 7.4K tokens in 13 seconds; the later ones reused 99% of the prompt and read the rest in 0.1 to 0.5 seconds. Generation takes about 64 ms a token (about 15 tokens/s)

## 7. hwnpu-agent: a coding agent on the engine itself

`hwnpu-agent` is a coding agent with four tools, read, edit, write and bash, in the loop of [Pi](https://pi.dev) written in Go on the same engine as hwnpu-chat, with no HTTP in between. Where OpenCode's first request takes about 7.4K tokens for its system prompt and ten tools, hwnpu-agent's system prompt and four declarations take about 630 tokens, so nearly all of the context (262144 for LFM3-1.7B) is left for the task, and it makes no side requests (OpenCode asks for a title first), so the task starts at once.

```bash
go run ./cmd/hwnpu-agent "fix the bug in the add function of main.go"
go run ./cmd/hwnpu-agent -C ~/src/demo -yes "add a mul function and print mul(2, 3)"
go run ./cmd/hwnpu-agent
```

- With a task on the command line it runs that task and exits (0 when the model answered, 1 otherwise); without one it takes tasks in the terminal, in one conversation (`/reset` starts over, `/exit` quits, Ctrl-C stops the task at hand)
- It works in `-C` (the current directory): paths are relative to it, commands run in it, and it writes no file outside it. `AGENTS.md` there, if any, goes into the system prompt (up to 4000 characters)
- Before a command or a change to a file it asks on the terminal; `-yes` skips the question, and `-allow "go *,ls*,cat *"` lets commands matching the patterns run without asking (each part of a command joined with `&&` or `;` must match). Without a terminal and without `-yes` it runs nothing else, so that a script cannot let a 1.7B model loose by accident
- `-tools search` offers a fifth tool, search, and alone makes an assistant that looks facts up instead of recalling them, which a 1.7B model does badly (at lut4 it put Mount Fuji at about 4,000 m and named the wrong author of a novel; with a search result in front of it, it copies the number and names the source). `-search` picks what is searched: `wikipedia` (no key; an article is found by the name of its subject, so a mountain, a river, a novel or a person, not a question in words), `brave` (the Brave Search API, the key in `BRAVE_API_KEY`) or the URL of a [SearXNG](https://searxng.org) instance such as `http://localhost:8888` for the web (DuckDuckGo refuses requests that are not a browser's, so it is not offered). A small model does not decide to search reliably, and a one-sentence change to the prompt flips it, so `-search-first` makes the first round of each task a search: when the model answers from memory instead, that round is dropped and a search for the task's first line takes its place. It also searches on with the answer in front of it, so a task gets four searches, and at the fifth it is asked to answer from what it has. `-tools search,fetch` adds fetch, which reads a page of the results as text (the first 6000 characters, tags and scripts stripped) when a snippet does not answer. With fetch, thinking is worth its time: on an M3 with Wikipedia, five questions took 190 seconds with thinking and 201 without. Thinking, the model searched in English, fetched the English articles and got the height of Mount Fuji (3,776.24 m), the longest river (信濃川) and a city's population right with their sources, and computed 17 × 23; only the author of 『こころ』 failed, because Wikipedia's search does not find the novel for 『こころ』 作者. Without thinking it got two of the five, and the page of Mount Fuji sent it into a long muddled answer. Search alone (no fetch, no thinking) got one: Mount Fuji from the article's opening after one search in about 3 seconds. With a SearXNG instance for the web (`-search http://localhost:8888`, thinking on, 324 seconds) four of the five came right, 『こころ』 among them, and the fifth, Mount Fuji, got muddled with another mountain's height from the article's lower sections. The web search finds what Wikipedia's own search misses; what is left is the 1.7B model: it romanizes Japanese names into queries that find nothing, and picks a wrong number out of a long page. Section 16.40 of [docs/design.md](docs/design.md) has the details
- `-tools read,bash` offers fewer tools (for a question about the code), `-system FILE` replaces the opening of the system prompt, `-think=false` answers without thinking, `-draft` sets the speculative decoding as in hwnpu-chat (8), `-tier` the shorter caches to run on while the conversation is short (4096), `-keep-thinking` keeps the thinking of earlier rounds of a task in the conversation, `-steps` limits the rounds (30), `-n` the tokens of an answer (2048), and `-dump DIR` writes each request, its prompt and the generated text as JSON, to compare prompts
- The conversation is kept within the context without asking the model for a summary: when it would leave fewer than `-n` tokens for the answer, old tool results are folded to their first line and a count, the oldest first, the latest four untouched
- A 1.7B model needs guards more than rules, as little-coder's work on Pi shows. `edit` and `write` on a file it has not read in the conversation are refused with a hint to read it first. Its thinking is closed at `-think-budget` tokens (1024, as in hwnpu-chat) so that it answers from what it has thought. When it writes its final answer and adds a call anyway, round after round, the second such answer ends the task; a call made twice in a row is not run but answered with a hint to stop, and a third time, like the last round, is followed by one request without tools for the summary. Paths like `/main.go` that it writes for `main.go` are taken as relative, `cd` to a directory it has not seen is refused with the rest of the command to run, and the directory's long absolute path is `.` in what it sees
- An example on an M3, "fix the bug in the add function of main.go" (`a - b` to `a + b`): 8 requests in 53 seconds, with the model compiled already (loading it takes about 15 seconds then, and about 3 minutes the first time, while Core ML compiles it). The first request read 629 tokens in 1.3 seconds, the later ones only the new messages (0.1 to 0.4 seconds), and it generated at 15 to 16 tokens/s. It read the file, changed the line with edit, read it again, tried `cd /tmp` (refused), ran `go run main.go`, saw 5, and answered with a summary. Earlier forms of the prompt show how little a 1.7B model forgives: with the directory's absolute path in the system prompt it copied the path wrong for 30 rounds, refusing `/main.go` as outside the directory cost another 30, and it kept adding `echo done` after its final answer until repeated answers ended the task. Section 16.35 of [docs/design.md](docs/design.md) has the details

## 8. Limits

- OpenCode first asks for a title of the conversation. This model thinks before answering even when told not to, and the main request waits meanwhile; `-think-budget` (default 1024 tokens, about a minute) bounds the wait
- With 1.7B parameters and 4 bit weights, factual answers can be wrong. Letting it think helps
- A longer conversation generates more slowly (on an M3, about 0.1 s a token at 64K and 0.3 s at 250K), and a prompt of 250K takes about half an hour to read
- PLaMo-Embedding-1B is converted with 8 bit tables (lut8): in 4 bits its embeddings point elsewhere (cosine 0.76 with fp16's). On an M3 its embeddings have a cosine of 0.997 to 0.999 with the Hugging Face implementation's (`tools/ref/embed_golden.py`, `TestPlamoEmbeddingGolden` in `internal/chat`), as on the CPU. The last layer's MLP produces values past fp16's range, so the converter computes the last layer divided by 64 (`head-scale`, section 16.34 of docs/design.md). Programs converted before, `embed-s128.mlpackage` and `embed-s1024.mlpackage` in lut4, are no longer used and can be deleted
- The implementation notes are in sections 16.17, 16.23 and 16.35 of [docs/design.md](docs/design.md)
