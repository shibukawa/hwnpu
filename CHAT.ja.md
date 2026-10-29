# チャット、OpenAI 互換 API、OpenCode

[English](CHAT.md)

LFM3-1.7B を Apple の Neural Engine だけで動かし、ターミナルで対話したり、OpenAI 互換の API として OpenCode などのクライアントから使ったりします。Python は要りません。

## 1. モデルを変換する

Hugging Face の [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) から `config.json`、`tokenizer.json`、`model.safetensors` を 1 つのディレクトリに置きます。同じキャッシュの長さで、1 回 1 トークンのものと 1 回 64 トークンのものを変換します。

```bash
go run ./cmd/hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -quant kmeans-minmax -seq 1 -cache 16320 -parts 2 -o DIR/lfm3-16k-q1.mlpackage
```

```bash
go run ./cmd/hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -quant kmeans-minmax -seq 64 -cache 16320 -o DIR/lfm3-16k-q64.mlpackage
```

- 1 つ目は 2 つの部分に分けて `DIR/lfm3-16k-q1.parts.json` を作ります (動作を確かめた構成。分けずに 1 つでも Neural Engine に載ります)
- 1 回 64 トークンのものはプロンプトを約 20 倍速く読みますが、1 トークンの生成は遅いので、生成には 1 回 1 トークンのものを使います。2 つは 1 つのキャッシュを共有します
- キャッシュ 16320 は Neural Engine に載る最大です (テンソルの 1 次元は 16384 まで、鍵の軸はキャッシュ + 1 回のトークン数)。OpenCode には 16K が要ります。対話だけなら `-cache 4096` のほうが生成は速くなります (M3 で 1 トークン約 28 ms、16K では約 64 ms)

## 2. 対話する

```bash
go run ./cmd/hwnpu-chat -package DIR/lfm3-16k-q1.parts.json,DIR/lfm3-16k-q64.mlpackage -config DIR/config.json -weights DIR/model.safetensors
```

- 最初の読み込みは Neural Engine 向けのコンパイルで 1〜2 分かかります。2 回目からはコンパイル済みのキャッシュを使います
- モデルはまず考えて (`<think>`、薄い色で表示) から答えます。`/think off` で考えずに答えます。`/reset` で会話を消し、`/exit` か Ctrl-D で終わります。Ctrl-C は答えを止めます。行末の `\` で次の行に続けます
- 会話の 2 回目以降は、前回からの差分だけを読みます

## 3. OpenAI 互換のサーバー

```bash
go run ./cmd/hwnpu-chat -listen 127.0.0.1:8080 -package DIR/lfm3-16k-q1.parts.json,DIR/lfm3-16k-q64.mlpackage -config DIR/config.json -weights DIR/model.safetensors
```

| エンドポイント | |
|---|---|
| `GET /v1/models` | モデル名 (1 つ目のパッケージの名前、例 `lfm3-16k-q1`) と `context_length` |
| `POST /v1/chat/completions` | `stream` (SSE)、`tools` と `tool_calls`、`reasoning_content`、`max_tokens`、`temperature`、`top_p`、`top_k`、`repetition_penalty`、`seed`、`stop`、`stream_options.include_usage` |

- 考えるかどうかは `-think` (既定 true) で決め、要求ごとに `reasoning_effort` (none、minimal、low で考えない、medium、high で考える) か `chat_template_kwargs.enable_thinking` で変えられます
- 要求は 1 つずつ処理します。前の要求と共通の先頭は読み直さず、`usage.prompt_tokens_details.cached_tokens` にその数が入ります。別の会話が割り込んでも、長い会話は 2 本まで取っておいて戻ります
- 同じ並びを繰り返し始めたら、そこで止めます
- `-dump DIR` で、要求、プロンプト、生成した文を要求ごとに保存します。クライアントが何を送っているかを調べるときに使います

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"lfm3","messages":[{"role":"user","content":"日本の首都は？"}]}'
```

## 4. OpenCode から使う

プロジェクトのディレクトリに `opencode.json` を置きます。モデル名は `GET /v1/models` が返す名前にします。

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "hwnpu": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "hwnpu (Neural Engine)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "lfm3-16k-q1": {
          "name": "LFM3 1.7B on the Neural Engine",
          "tool_call": true,
          "reasoning": true,
          "limit": { "context": 16320, "output": 4096 }
        }
      }
    }
  },
  "model": "hwnpu/lfm3-16k-q1",
  "permission": {
    "bash": { "*": "ask", "go *": "allow", "ls*": "allow", "cat *": "allow", "pwd": "allow" }
  }
}
```

- `limit.context` はサーバーのキャッシュの長さに合わせます。OpenCode の要求は system と道具の説明だけで約 7.4K トークンあるので、8K のキャッシュでは 1 往復で要約が始まってしまいます
- 1.7B のモデルは見当違いのコマンドを作ることがあります。`bash` は許すコマンドを絞っておくと安全です
- M3 での例: 「main.go の add 関数のバグを直して」で、ファイルを読み、`a - b` を `a + b` に直し、`go run` で確かめるまでを 12 往復で行いました。最初の要求は 7.4K トークンを 13 秒で読み、2 回目以降はプロンプトの 99% を再利用して 0.1〜0.5 秒で読みます。生成は 1 トークン約 64 ms (約 15 tokens/s) です

## 5. 限界

- OpenCode は最初に会話の題名を頼みます。このモデルは考えないよう指示しても考えてから答えるので 1〜2 分かかり、その間は本題の要求が待ちます
- 1.7B で 4 ビットの重みなので、知識の問いは誤ることがあります。考えさせたほうが正確です
- 文脈は 16320 トークンまでです。これより長い文脈には、キャッシュを塊に分けて注意を計算する方式が要ります ([docs/kv-cache.md](docs/kv-cache.md))
- 実装の記録は [docs/design.md](docs/design.md) の 16.17 にあります
