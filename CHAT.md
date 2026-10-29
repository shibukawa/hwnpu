# Chat, an OpenAI compatible API and OpenCode

[日本語](CHAT.ja.md)

Run LFM3-1.7B on the Apple Neural Engine alone, and talk with it in the terminal or use it as an OpenAI compatible API from clients such as OpenCode. No Python is needed.

## 1. Chat

```bash
go run ./cmd/hwnpu-chat
```

No arguments are needed. The first run does the following by itself (about 3.5 minutes on an M3, besides the download). Later runs only load the compiled programs and start in about a minute.

1. It downloads `config.json`, `tokenizer.json` and `model.safetensors` (3.5 GB) of [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) from Hugging Face to `~/Library/Caches/hwnpu/models/LFM3-1.7B`. A download cut off goes on from where it stopped on the next run
2. It converts the model twice for the Neural Engine (about a minute each): one generates 1 token per call, the other reads prompts 64 tokens per call, about 20 times faster. The two share one KV cache
3. It compiles them for the Neural Engine (1 to 2 minutes). The result stays in `~/Library/Caches/hwnpu`

Then type after the `>`.

- The model thinks first (`<think>`, shown dim) and then answers. `/think off` makes it answer at once, `/think on` brings thinking back
- `/reset` clears the conversation, `/exit` or Ctrl-D quits, Ctrl-C stops an answer. A line ending in `\` continues on the next line
- From the second turn on, only what is new since the last answer is read
- After each answer it shows the tokens read and the speed of generation (about 15 tokens/s on an M3)

## 2. Options

| Flag | Default | |
|---|---|---|
| `-listen ADDR` | none | serve the OpenAI compatible API at `ADDR` (e.g. `127.0.0.1:8080`) instead of chatting (section 3) |
| `-think` | `true` | let the model think before it answers |
| `-system TEXT` | none | the chat's system message |
| `-n N` | `0` | most tokens of an answer (0: until the context is full) |
| `-temp`, `-top-k`, `-top-p`, `-rep` | `0.1`, `50`, `0`, `1.05` | sampling (temperature, top k, top p, repetition penalty within the answer) |
| `-context N` | `16320` | most tokens of a prompt and its answer together; another value converts the model again for it |
| `-model DIR` | `~/Library/Caches/hwnpu/models/LFM3-1.7B` | where the checkpoint and the converted programs are |
| `-repo NAME` | `LiquidAI/LFM3-1.7B` | the Hugging Face repository of the checkpoint; set `HF_TOKEN` for gated ones |
| `-package A,B` | none | use programs you converted with `hwnpu-convert` (section 4) |
| `-dump DIR` | none | with the server, save each request, its prompt and the generated text |

- A context of 16320 is the most the Neural Engine takes (a tensor dimension goes up to 16384, and the key axis is the context plus the tokens per call). OpenCode needs 16K. For chat alone, `-context 4096` generates faster (about 28 ms a token on an M3, against about 64 ms with 16K)

## 3. OpenAI compatible server

```bash
go run ./cmd/hwnpu-chat -listen 127.0.0.1:8080
```

| Endpoint | |
|---|---|
| `GET /v1/models` | the model's name (`LFM3-1.7B`; with `-package`, the first package's) and `context_length` |
| `POST /v1/chat/completions` | `stream` (SSE), `tools` and `tool_calls`, `reasoning_content`, `max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop`, `stream_options.include_usage` |

- `-think` (default true) decides whether the model thinks; a request changes it with `reasoning_effort` (none, minimal and low turn it off, medium and high on) or `chat_template_kwargs.enable_thinking`
- Requests run one at a time. The beginning a prompt shares with an earlier one is not read again, and `usage.prompt_tokens_details.cached_tokens` says how much. When another conversation comes in between, up to two long conversations are kept aside and come back
- Generation stops when the model starts repeating one pattern
- `-dump DIR` saves each request, its prompt and the generated text, for finding out what a client sends
- The request's `model` is not looked at: any name gets the same model

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"lfm3","messages":[{"role":"user","content":"What is the capital of Japan?"}]}'
```

## 4. Programs you converted

To try another quantization or cache length, convert twice with `hwnpu-convert` with the same cache length, and list both in `-package`.

```bash
go run ./cmd/hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -quant kmeans-minmax -seq 1 -cache 16320 -parts 2 -o DIR/lfm3-16k-q1.mlpackage
```

```bash
go run ./cmd/hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -quant kmeans-minmax -seq 64 -cache 16320 -o DIR/lfm3-16k-q64.mlpackage
```

```bash
go run ./cmd/hwnpu-chat -package DIR/lfm3-16k-q1.parts.json,DIR/lfm3-16k-q64.mlpackage -model DIR
```

## 5. Use it from OpenCode

Put an `opencode.json` in the project directory. The model's name is the one `GET /v1/models` returns.

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

## 6. Limits

- OpenCode first asks for a title of the conversation. This model thinks before answering even when told not to, which takes 1 to 2 minutes, and the main request waits meanwhile
- With 1.7B parameters and 4 bit weights, factual answers can be wrong. Letting it think helps
- The context is at most 16320 tokens. Longer contexts need the attention computed over the cache in chunks ([docs/kv-cache.md](docs/kv-cache.md))
- The implementation notes are in section 16.17 of [docs/design.md](docs/design.md)
