# Chat, embeddings, an OpenAI compatible API and OpenCode

[日本語](CHAT.ja.md)

Run LFM3-1.7B and PLaMo-Embedding-1B on the Apple Neural Engine alone, and use them in the terminal or as an OpenAI compatible API from clients such as OpenCode. No Python is needed.

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

No arguments are needed. The first run does the following by itself (about 3.5 minutes on an M3, besides the download). Later runs only load the compiled programs and start in a few seconds to a minute, showing how far loading is.

1. It downloads `config.json`, `tokenizer.json` and `model.safetensors` (3.5 GB) of [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) from Hugging Face to `~/Library/Caches/hwnpu/models/LFM3-1.7B`. A download cut off goes on from where it stopped on the next run
2. It converts the model twice for the Neural Engine (about a minute each): one generates, 8 tokens per call (a token and 7 guesses checked at once, section 3), the other reads prompts 64 tokens per call, about 20 times faster. The two share one KV cache
3. It compiles them for the Neural Engine (1 to 2 minutes). The result stays in `~/Library/Caches/hwnpu`

Then type after the `>`.

- The model thinks first (`<think>`, shown dim) and then answers. `/think off` makes it answer at once, `/think on` brings thinking back
- `/reset` clears the conversation, `/exit` or Ctrl-D quits, Ctrl-C stops an answer. A line ending in `\` continues on the next line
- From the second turn on, only what is new since the last answer is read
- After each answer it shows the tokens read and the speed of generation (about 15 tokens/s on an M3)

### Images and audio (LFM3-VL-2B, LFM3-Audio-2B, LFM3-Omni-2B)

`-use LFM3-VL-2B`, `-use LFM3-Audio-2B` or `-use LFM3-Omni-2B` runs one of Liquid AI's multimodal checkpoints. The first run downloads it (3.8, 4.0 and 4.3 GB) and converts its text model the same way, but with 8 bit tables (lut8, 2 parts each): with 4 bit tables these models describe pictures wrongly and mishear speech (docs/design.md, 16.31). It also converts the encoders the checkpoint has, in fp16: the vision encoder for images of 512x512 (an image is stretched to it) and the audio encoder for clips of up to 30 seconds.

```bash
go run ./cmd/hwnpu-chat -use LFM3-VL-2B
```

- In the terminal, `/image FILE question` sends an image and `/audio FILE.wav question` a clip. The server reads the `image_url` (data URLs) and `input_audio` (base64 WAV) parts of user messages
- About 41 ms per token on the M3 (LFM3-1.7B in lut4: 19 ms)
- Answers are text; LFM3-Audio's speech output is `hwnpu-tts` (section 5)
- Other image sizes or clip lengths: convert the encoders yourself and give them with `-package`, `-vision` and `-audio` (section 5)

### Gemma 4 E2B and E4B

`-use gemma-4-E2B-it` runs Google's Gemma 4 E2B (instruction tuned) instead. It goes the same way: `model.safetensors` (10.2 GB, with the audio and vision parts, of which only the text model is converted) is downloaded to `~/Library/Caches/hwnpu/models/gemma-4-E2B-it`, and the programs of 1 and 64 tokens per call are converted with 6 bit tables (lut6) in 2 parts each (about 0.8 + 0.8 GiB).

```bash
go run ./cmd/hwnpu-chat -use gemma-4-E2B-it
```

- The chat format is Gemma 4's `chat_template.jinja` in Go, with thinking (`<|think|>`, returned as `reasoning_content`) and tool calls
- The per-layer embedding table (4.7 GB) stays on the host, which reads one row (17.5 KB) per token
- Contexts up to 16320 (holding the cache in chunks past that is LFM3's only). About 31 ms per token on the M3 with a cache of 1024, 55 ms with the context of 16320 (LFM3-1.7B: 19 and 63 to 66 ms)
- 4 bit tables (lut4) cost Gemma 4 much more than LFM3 (it got Mount Fuji's height and the author of Kokoro wrong), so it converts with 6 bit tables by default; `hwnpu-convert` does too when `-format` is left out. See sections 16.25 and 16.26 of docs/design.md
- `-use gemma-4-E4B-it` runs the larger E4B (16.0 GB download, 42 layers of width 2560) the same way, slower than E2B. It is converted by the same code but not yet checked on the M3. The larger Gemma 4 sizes (12B and up) are another architecture, which hwnpu-chat does not run

## 2. Options

| Flag | Default | |
|---|---|---|
| `-use NAMES` | `LFM3-1.7B` | the models to run, comma separated: names from `-list` or repositories `OWNER/NAME`; one in the terminal |
| `-list` | | list the models, what each is for, and whether it is downloaded |
| `-listen ADDR` | none | serve the OpenAI compatible API at `ADDR` (e.g. `127.0.0.1:8080`) instead of the terminal (section 3) |
| `-think` | `true` | let the model think before it answers |
| `-think-budget N` | `1024` | most tokens the model thinks before it is made to answer (0: no limit) |
| `-draft N` | `8` | tokens per call of the program that generates: a token and N-1 guesses checked at once (speculative decoding, section 3); `0` or `1` generates one token per call |
| `-system TEXT` | none | the chat's system message |
| `-n N` | `0` | most tokens of an answer (0: until the context is full) |
| `-temp`, `-top-k`, `-top-p`, `-rep` | `0.1`, `50`, `0`, `1.05` | sampling (temperature, top k, top p, repetition penalty within the answer) |
| `-context N` | `16320` | most tokens of a prompt and its answer together; another value converts the model again for it; over 16320 the cache is held in chunks |
| `-dir DIR` | `~/Library/Caches/hwnpu/models` | where the models are kept, each in `DIR/NAME` with its checkpoint and converted programs |
| `-package A,B` | none | chat programs you converted with `hwnpu-convert` (section 5) |
| `-dump DIR` | none | with the server, save each request, its prompt and the generated text |
| `-embed-seq A,B` | `128,1024` | lengths an embedding model is converted for; a text runs on the shortest that holds it, and the longest is the most tokens a text can have |
| `-embed-package A,B` | none | embedding programs you converted with `hwnpu-convert` (encoders take `-seq` only), with `config.json` and `model.safetensors` in the model's directory |

The flags from `-think` to `-context`, `-package`, `-config`, `-weights` and `-tokenizer` are for chat models, the `-embed-` flags for embedding models. `-package` and its companions (and `-embed-package`) take one model of their kind in `-use`.

- The longer the context, the slower each token. OpenCode needs 16K or more. For chat alone, `-context 4096` is faster

| `-context` | generation (M3) | reading a prompt | KV cache |
|---|---|---|---|
| 4096 | about 28 ms/token | — | about 50 MiB |
| 16320 (default) | about 64 ms/token | about 570 tokens/s | about 190 MiB |
| 32768 | about 138 ms/token | about 300 tokens/s | about 390 MiB |
| 65536 | about 363 ms/token | about 150 tokens/s | about 770 MiB |

- Over 16320, the cache is held in chunks of 16384 tokens, to stay within the Neural Engine's 16384 per tensor dimension. With 32768 and 65536 the model recalled a passphrase put at the start of prompts of 25,288 and 53,736 tokens
- The server keeps up to two other conversations as copies of the whole cache, so memory goes up to three times the KV cache above

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
- Generation stops when the model starts repeating one pattern
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
          "limit": { "context": 16320, "output": 4096 }
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

- Set `limit.context` to the server's cache length. OpenCode's requests take about 7.4K tokens for the system prompt and the tools alone, so with an 8K cache it starts compacting after one turn
- A 1.7B model sometimes writes commands that make no sense. Allowing only some commands for `bash` is safer
- An example on an M3: asked to fix the bug in `add` of main.go, it read the file, changed `a - b` to `a + b` and checked with `go run`, in 12 requests. The first request read 7.4K tokens in 13 seconds; the later ones reused 99% of the prompt and read the rest in 0.1 to 0.5 seconds. Generation takes about 64 ms a token (about 15 tokens/s)

## 7. hwnpu-agent: a coding agent on the engine itself

`hwnpu-agent` is a coding agent with four tools, read, edit, write and bash, in the loop of [Pi](https://pi.dev) written in Go on the same engine as hwnpu-chat, with no HTTP in between. Where OpenCode's first request takes about 7.4K tokens for its system prompt and ten tools, hwnpu-agent's system prompt and four declarations take about 630 tokens, so nearly all of the 16K context is left for the task, and it makes no side requests (OpenCode asks for a title first), so the task starts at once.

```bash
go run ./cmd/hwnpu-agent "fix the bug in the add function of main.go"
go run ./cmd/hwnpu-agent -C ~/src/demo -yes "add a mul function and print mul(2, 3)"
go run ./cmd/hwnpu-agent
```

- With a task on the command line it runs that task and exits (0 when the model answered, 1 otherwise); without one it takes tasks in the terminal, in one conversation (`/reset` starts over, `/exit` quits, Ctrl-C stops the task at hand)
- It works in `-C` (the current directory): paths are relative to it, commands run in it, and it writes no file outside it. `AGENTS.md` there, if any, goes into the system prompt (up to 4000 characters)
- Before a command or a change to a file it asks on the terminal; `-yes` skips the question, and `-allow "go *,ls*,cat *"` lets commands matching the patterns run without asking (each part of a command joined with `&&` or `;` must match). Without a terminal and without `-yes` it runs nothing else, so that a script cannot let a 1.7B model loose by accident
- `-tools read,bash` offers fewer tools (for a question about the code), `-system FILE` replaces the opening of the system prompt, `-think=false` answers without thinking, `-draft` sets the speculative decoding as in hwnpu-chat (8), `-keep-thinking` keeps the thinking of earlier rounds of a task in the conversation, `-steps` limits the rounds (30), `-n` the tokens of an answer (2048), and `-dump DIR` writes each request, its prompt and the generated text as JSON, to compare prompts
- The conversation is kept within the context without asking the model for a summary: when it would leave fewer than `-n` tokens for the answer, old tool results are folded to their first line and a count, the oldest first, the latest four untouched
- A 1.7B model needs guards more than rules, as little-coder's work on Pi shows. `edit` and `write` on a file it has not read in the conversation are refused with a hint to read it first. Its thinking is closed at `-think-budget` tokens (1024, as in hwnpu-chat) so that it answers from what it has thought. When it writes its final answer and adds a call anyway, round after round, the second such answer ends the task; a call made twice in a row is not run but answered with a hint to stop, and a third time, like the last round, is followed by one request without tools for the summary. Paths like `/main.go` that it writes for `main.go` are taken as relative, `cd` to a directory it has not seen is refused with the rest of the command to run, and the directory's long absolute path is `.` in what it sees
- An example on an M3, "fix the bug in the add function of main.go" (`a - b` to `a + b`): 8 requests in 53 seconds, with the model compiled already (loading it takes about 15 seconds then, and about 3 minutes the first time, while Core ML compiles it). The first request read 629 tokens in 1.3 seconds, the later ones only the new messages (0.1 to 0.4 seconds), and it generated at 15 to 16 tokens/s. It read the file, changed the line with edit, read it again, tried `cd /tmp` (refused), ran `go run main.go`, saw 5, and answered with a summary. Earlier forms of the prompt show how little a 1.7B model forgives: with the directory's absolute path in the system prompt it copied the path wrong for 30 rounds, refusing `/main.go` as outside the directory cost another 30, and it kept adding `echo done` after its final answer until repeated answers ended the task. Section 16.35 of [docs/design.md](docs/design.md) has the details

## 8. Limits

- OpenCode first asks for a title of the conversation. This model thinks before answering even when told not to, and the main request waits meanwhile; `-think-budget` (default 1024 tokens, about a minute) bounds the wait
- With 1.7B parameters and 4 bit weights, factual answers can be wrong. Letting it think helps
- A longer context generates more slowly (about 0.36 s a token with 65536). Contexts up to 65536 were tried
- PLaMo-Embedding-1B is converted with 8 bit tables (lut8): in 4 bits its embeddings point elsewhere (cosine 0.76 with fp16's). On an M3 its embeddings have a cosine of 0.997 to 0.999 with the Hugging Face implementation's (`tools/ref/embed_golden.py`, `TestPlamoEmbeddingGolden` in `internal/chat`), as on the CPU. The last layer's MLP produces values past fp16's range, so the converter computes the last layer divided by 64 (`head-scale`, section 16.34 of docs/design.md). Programs converted before, `embed-s128.mlpackage` and `embed-s1024.mlpackage` in lut4, are no longer used and can be deleted
- The implementation notes are in sections 16.17, 16.23 and 16.35 of [docs/design.md](docs/design.md)
