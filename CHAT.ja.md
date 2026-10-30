# チャット、OpenAI 互換 API、OpenCode

[English](CHAT.md)

LFM3-1.7B を Apple の Neural Engine だけで動かし、ターミナルで対話したり、OpenAI 互換の API として OpenCode などのクライアントから使ったりします。Python は要りません。

## 1. 対話する

```bash
go run ./cmd/hwnpu-chat
```

引数は要りません。最初の起動では次のことを自動で行います (M3 で、落とす時間を除いて約 3 分半)。2 回目からはコンパイル済みのものを読み込むだけで、約 1 分で始まります。

1. Hugging Face の [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) から `config.json`、`tokenizer.json`、`model.safetensors` (3.5 GB) を `~/Library/Caches/hwnpu/models/LFM3-1.7B` に落とします。途中で切れても、もう一度起動すれば続きから落とします
2. Neural Engine 用に 2 回変換します (それぞれ 1 分ほど)。1 回 1 トークンで生成するものと、1 回 64 トークンでプロンプトを約 20 倍速く読むものです。2 つは 1 つの KV キャッシュを共有します
3. Neural Engine 向けにコンパイルします (1〜2 分)。結果は `~/Library/Caches/hwnpu` に残ります

起動したら、`>` に続けて話しかけます。

- モデルはまず考えて (`<think>`、薄い色で表示) から答えます。`/think off` で考えずに答え、`/think on` で戻ります
- `/reset` で会話を消し、`/exit` か Ctrl-D で終わります。Ctrl-C は答えを止めます。行末の `\` で次の行に続けます
- 会話の 2 回目以降は、前回からの差分だけを読みます
- 答えのあとに、読んだトークン数と生成の速さ (M3 で約 15 tokens/s) を出します

### Gemma 4 E2B を使う

`-repo google/gemma-4-E2B-it` で Google の Gemma 4 E2B (指示調整版) を使えます。手順は同じで、`~/Library/Caches/hwnpu/models/gemma-4-E2B-it` に `model.safetensors` (10.2 GB。音声と画像の部分も入っているが、変換するのはテキストだけ) を落とし、1 トークンと 64 トークンのプログラムを 6 ビットの表 (lut6) で 2 部分ずつ (約 0.8 + 0.8 GiB) に変換します。

```bash
go run ./cmd/hwnpu-chat -repo google/gemma-4-E2B-it
```

- チャットの書式は Gemma 4 の `chat_template.jinja` を Go に移したもので、思考 (`<|think|>`、`reasoning_content` に返す) と道具の呼び出しに対応します
- 層ごとの埋め込みの表 (4.7 GB) はホストに置き、トークンごとに 1 行 (17.5 KB) を読みます
- M3 で 1 トークン約 31 ms (キャッシュ 1024)、文脈 16320 では 55 ms (LFM3-1.7B は 19 ms と 63〜66 ms)
- Gemma 4 は 4 ビットの表 (lut4) では損失が大きく (富士山の高さや『こころ』の作者を誤る)、6 ビットを既定にしています。`-package` で自分で変換するときは `hwnpu-convert` の `-format` を省けば lut6 になります。詳細は docs/design.md の 16.21 と 16.22

## 2. オプション

| フラグ | 既定 | |
|---|---|---|
| `-listen ADDR` | なし | 対話の代わりに OpenAI 互換のサーバーを `ADDR` (例 `127.0.0.1:8080`) で開く (3 節) |
| `-think` | `true` | 答える前に考えさせる |
| `-system TEXT` | なし | 対話の system メッセージ |
| `-n N` | `0` | 1 つの答えの最大トークン数 (0 は文脈が尽きるまで) |
| `-temp`、`-top-k`、`-top-p`、`-rep` | `0.1`、`50`、`0`、`1.05` | サンプリング (温度、上位 k、上位 p、答えの中の繰り返しの罰則) |
| `-context N` | `16320` | プロンプトと答えを合わせた最大トークン数。変えると、その長さで変換し直す |
| `-model DIR` | `~/Library/Caches/hwnpu/models/NAME` (`-repo` の名前) | チェックポイントと変換したプログラムの置き場 |
| `-repo NAME` | `LiquidAI/LFM3-1.7B` | チェックポイントを落とす Hugging Face のリポジトリ (`LiquidAI/LFM3-1.7B` か `google/gemma-4-E2B-it`)。認証の要るものは環境変数 `HF_TOKEN` |
| `-package A,B` | なし | `hwnpu-convert` で自分で変換したプログラムを使う (4 節) |
| `-dump DIR` | なし | サーバーで、要求、プロンプト、生成した文を要求ごとに保存する |

- 文脈 16320 は Neural Engine に載る最大です (テンソルの 1 次元は 16384 まで、鍵の軸は文脈 + 1 回のトークン数)。OpenCode には 16K が要ります。対話だけなら `-context 4096` のほうが生成は速くなります (M3 で 1 トークン約 28 ms、16K では約 64 ms)

## 3. OpenAI 互換のサーバー

```bash
go run ./cmd/hwnpu-chat -listen 127.0.0.1:8080
```

| エンドポイント | |
|---|---|
| `GET /v1/models` | モデル名 (`LFM3-1.7B`。`-package` のときは 1 つ目のパッケージの名前) と `context_length` |
| `POST /v1/chat/completions` | `stream` (SSE)、`tools` と `tool_calls`、`reasoning_content`、`max_tokens`、`temperature`、`top_p`、`top_k`、`repetition_penalty`、`seed`、`stop`、`stream_options.include_usage` |

- 考えるかどうかは `-think` (既定 true) で決め、要求ごとに `reasoning_effort` (none、minimal、low で考えない、medium、high で考える) か `chat_template_kwargs.enable_thinking` で変えられます
- 要求は 1 つずつ処理します。前の要求と共通の先頭は読み直さず、`usage.prompt_tokens_details.cached_tokens` にその数が入ります。別の会話が割り込んでも、長い会話は 2 本まで取っておいて戻ります
- 同じ並びを繰り返し始めたら、そこで止めます
- `-dump DIR` で、要求、プロンプト、生成した文を要求ごとに保存します。クライアントが何を送っているかを調べるときに使います
- リクエストの `model` は見ません。どの名前でも同じモデルが答えます

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"lfm3","messages":[{"role":"user","content":"日本の首都は？"}]}'
```

## 4. 自分で変換したプログラムを使う

ほかの量子化やキャッシュの長さを試すときは、`hwnpu-convert` で同じキャッシュの長さのものを 2 つ変換し、`-package` にカンマで並べます。

```bash
go run ./cmd/hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -quant kmeans-minmax -seq 1 -cache 16320 -parts 2 -o DIR/lfm3-16k-q1.mlpackage
```

```bash
go run ./cmd/hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -quant kmeans-minmax -seq 64 -cache 16320 -o DIR/lfm3-16k-q64.mlpackage
```

```bash
go run ./cmd/hwnpu-chat -package DIR/lfm3-16k-q1.parts.json,DIR/lfm3-16k-q64.mlpackage -model DIR
```

## 5. OpenCode から使う

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

- `limit.context` はサーバーのキャッシュの長さに合わせます。OpenCode の要求は system と道具の説明だけで約 7.4K トークンあるので、8K のキャッシュでは 1 往復で要約が始まってしまいます
- 1.7B のモデルは見当違いのコマンドを作ることがあります。`bash` は許すコマンドを絞っておくと安全です
- M3 での例: 「main.go の add 関数のバグを直して」で、ファイルを読み、`a - b` を `a + b` に直し、`go run` で確かめるまでを 12 往復で行いました。最初の要求は 7.4K トークンを 13 秒で読み、2 回目以降はプロンプトの 99% を再利用して 0.1〜0.5 秒で読みます。生成は 1 トークン約 64 ms (約 15 tokens/s) です

## 6. 限界

- OpenCode は最初に会話の題名を頼みます。このモデルは考えないよう指示しても考えてから答えるので 1〜2 分かかり、その間は本題の要求が待ちます
- 1.7B で 4 ビットの重みなので、知識の問いは誤ることがあります。考えさせたほうが正確です
- 文脈は 16320 トークンまでです。これより長い文脈には、キャッシュを塊に分けて注意を計算する方式が要ります ([docs/kv-cache.md](docs/kv-cache.md))
- 実装の記録は [docs/design.md](docs/design.md) の 16.17 にあります
