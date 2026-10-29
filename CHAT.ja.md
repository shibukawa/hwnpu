# チャット、埋め込み、OpenAI 互換 API、OpenCode

[English](CHAT.md)

LFM3-1.7B と PLaMo-Embedding-1B を Apple の Neural Engine だけで動かし、ターミナルで使ったり、OpenAI 互換の API として OpenCode などのクライアントから使ったりします。Python は要りません。

使うモデルは `-use` に名前で指定します。選べるモデルの名前と用途は `-list` で出ます。

```bash
go run ./cmd/hwnpu-chat -list
```

| 名前 | 用途 | 機能 | ターミナル | `-listen` のとき (3 節) |
|---|---|---|---|---|
| `LFM3-1.7B` (既定) | chat | `tools`、`thinking` | 対話する (1 節) | `/v1/chat/completions` |
| `plamo-embedding-1b` | embedding | `query` | 埋め込みで文を検索する (4 節) | `/v1/embeddings` |

機能は、用途のほかにそのモデルができることです (名前は Ollama の capabilities に合わせています)。

| 機能 | |
|---|---|
| `tools` | リクエストが示す道具を呼び出す (`tools`、`tool_calls`)。OpenCode のようなエージェントから動かせる (6 節)。LFM3-1.7B はテキストだけだが、小さく速いのでエージェントの裏で動かしやすい |
| `thinking` | 答える前に考える (`reasoning_content`)。`-think` と `reasoning_effort` で切り替える |
| `query` | 検索の問いと検索される文書とを別々に埋め込む (`input_type`、4 節) |

- ターミナルで動かすのは `-use` に指定した 1 つのモデルで、何をするかはモデルの用途で決まります。サーバーはいくつでも動かせます (例 `-use LFM3-1.7B,plamo-embedding-1b`)
- 名前の大文字と小文字は区別せず、一覧のリポジトリ名でも指定できます。一覧にない Hugging Face のリポジトリも `-use OWNER/NAME` で指定でき、hwnpu-chat が動かせる構造 (チャットは LFM3、埋め込みは PLaMo のエンコーダ) なら `config.json` から用途を判断します。認証の要るリポジトリは環境変数 `HF_TOKEN`

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

## 2. オプション

| フラグ | 既定 | |
|---|---|---|
| `-use NAMES` | `LFM3-1.7B` | 動かすモデルをカンマ区切りで: `-list` の名前かリポジトリ `OWNER/NAME`。ターミナルでは 1 つ |
| `-list` | | モデルの一覧 (用途、大きさ、落としてあるか) を出して終わる |
| `-listen ADDR` | なし | ターミナルの代わりに OpenAI 互換のサーバーを `ADDR` (例 `127.0.0.1:8080`) で開く (3 節) |
| `-think` | `true` | 答える前に考えさせる |
| `-system TEXT` | なし | 対話の system メッセージ |
| `-n N` | `0` | 1 つの答えの最大トークン数 (0 は文脈が尽きるまで) |
| `-temp`、`-top-k`、`-top-p`、`-rep` | `0.1`、`50`、`0`、`1.05` | サンプリング (温度、上位 k、上位 p、答えの中の繰り返しの罰則) |
| `-context N` | `16320` | プロンプトと答えを合わせた最大トークン数。変えると、その長さで変換し直す |
| `-dir DIR` | `~/Library/Caches/hwnpu/models` | モデルの置き場。モデルごとに `DIR/名前` にチェックポイントと変換したプログラムを置く |
| `-package A,B` | なし | `hwnpu-convert` で自分で変換したチャットのプログラムを使う (5 節) |
| `-dump DIR` | なし | サーバーで、要求、プロンプト、生成した文を要求ごとに保存する |
| `-embed-seq A,B` | `128,1024` | 埋め込みモデルを変換する長さ。文はそれが収まる最も短いもので動き、最も長いものが 1 つの文の最大トークン数になる |
| `-embed-package A,B` | なし | `hwnpu-convert` で自分で変換した埋め込みのプログラムを使う (エンコーダは `-seq` だけ)。`config.json` と `model.safetensors` はモデルの置き場に置く |

`-think` から `-context` まで、`-package`、`-config`、`-weights`、`-tokenizer` はチャットのモデルの、`-embed-` で始まるものは埋め込みモデルのフラグです。`-package` とその仲間 (と `-embed-package`) は、`-use` にその用途のモデルが 1 つのときだけ使えます。

- 文脈 16320 は Neural Engine に載る最大です (テンソルの 1 次元は 16384 まで、鍵の軸は文脈 + 1 回のトークン数)。OpenCode には 16K が要ります。対話だけなら `-context 4096` のほうが生成は速くなります (M3 で 1 トークン約 28 ms、16K では約 64 ms)

## 3. OpenAI 互換のサーバー

```bash
go run ./cmd/hwnpu-chat -listen 127.0.0.1:8080
```

```bash
go run ./cmd/hwnpu-chat -listen 127.0.0.1:8080 -use LFM3-1.7B,plamo-embedding-1b
```

| エンドポイント | |
|---|---|
| `GET /v1/models` | 読み込んだモデルの名前 (`LFM3-1.7B`、`plamo-embedding-1b`)、用途 `kind` (`chat` か `embedding`)、機能 `capabilities` (`-list` にあるモデルのみ)、`context_length`、埋め込みモデルは `dimensions` も |
| `POST /v1/chat/completions` | `stream` (SSE)、`tools` と `tool_calls`、`reasoning_content`、`max_tokens`、`temperature`、`top_p`、`top_k`、`repetition_penalty`、`seed`、`stop`、`stream_options.include_usage` |
| `POST /v1/embeddings` | `input` (文字列かその配列)、`encoding_format` (`float` か `base64`)、`dimensions` (モデルの 2048 のみ)、OpenAI にないものとして `input_type` (`query` か `document`、4 節) と `truncate` |

- 考えるかどうかは `-think` (既定 true) で決め、要求ごとに `reasoning_effort` (none、minimal、low で考えない、medium、high で考える) か `chat_template_kwargs.enable_thinking` で変えられます
- 要求は 1 つずつ処理します。前の要求と共通の先頭は読み直さず、`usage.prompt_tokens_details.cached_tokens` にその数が入ります。別の会話が割り込んでも、長い会話は 2 本まで取っておいて戻ります
- 同じ並びを繰り返し始めたら、そこで止めます
- `-dump DIR` で、要求、プロンプト、生成した文を要求ごとに保存します。クライアントが何を送っているかを調べるときに使います
- 要求は、エンドポイントの用途のモデルのうち `model` の名前 (大文字小文字は区別しない) のものに行きます。その用途のモデルが 1 つだけなら、名前にかかわらずそれが答えます。複数あるときに知らない名前なら 404 です。用途の違うモデルの名前 (`/v1/chat/completions` に `plamo-embedding-1b`) や、その用途のモデルを読み込んでいないエンドポイントはエラーになります

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"lfm3","messages":[{"role":"user","content":"日本の首都は？"}]}'
```

## 4. 埋め込み (PLaMo-Embedding-1B)

```bash
go run ./cmd/hwnpu-chat -use plamo-embedding-1b
```

[pfnet/plamo-embedding-1b](https://huggingface.co/pfnet/plamo-embedding-1b) は日本語 (と英語) の文を 2048 個の数のベクトルにします。意味の近い文ほど同じ向きのベクトルになります。最初の起動で `config.json`、`tokenizer.model`、`model.safetensors` (2.1 GB) を `~/Library/Caches/hwnpu/models/plamo-embedding-1b` に落とし、チャットのモデルと同じく `-embed-seq` の長さごとに 1 回ずつ変換します。

ターミナルでは、入力した 1 行ずつが検索の対象の文書として溜まり、`?` (全角の `？` も可) で始まる行が検索の問い (クエリ) になります。文書をクエリとのコサイン類似度の高い順に並べます。`/list` で文書の一覧、`/reset` で文書を消し、`/exit` か Ctrl-D で終わります。

```text
> 東京は日本の首都です。
> 最近は随分と暖かくなりましたね。
> ? 日本の首都はどこ？
```

API では (`-listen` を付け、`-use` に `plamo-embedding-1b` を入れて) 次のようにします。

```bash
curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' -d '{"model":"plamo-embedding-1b","input":["東京は日本の首都です。","最近は随分と暖かくなりましたね。"]}'
```

```bash
curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' -d '{"model":"plamo-embedding-1b","input":"日本の首都はどこ？","input_type":"query"}'
```

- PLaMo-Embedding は、検索の問いと検索される文書とで埋め込み方が違います。クエリには前置き `次の文章に対して、関連する文章を検索してください: ` を付け (モデルの `encode_query`)、文書には何も付けません (`encode_document`)。API では `input_type: "query"` (か `search_query`) で入力をクエリとして扱います。既定は `document` で、検索以外 (クラスタリング、分類など) の文もこちらです。OpenAI の API にはこの項目がないので、送れないクライアントでは文書として扱われます
- 埋め込みは、文そのもののトークン (BOS と前置きを除く) の最終隠れ状態の平均を、長さ 1 に揃えたものです。2 つの埋め込みの内積がそのままコサイン類似度になります
- 文は、変換した長さのうち収まる最も短いものに詰め物をして動かします。最も長いもの (既定 1024 トークン) を超える文は、リクエストに `"truncate": true` がなければエラー (`context_length_exceeded`) にします。ターミナルでは切り詰めて、その旨を表示します。モデル自体は 4096 トークンまで読めます。`-embed-seq 128,1024,4096` でその長さも変換できます (4096 トークンの注意を Neural Engine で動かすのは未確認)
- 要求は 1 つずつ、文も 1 つずつ処理します。チャットのモデルと埋め込みモデルは互いに独立に動きます

## 5. 自分で変換したプログラムを使う

ほかの量子化やキャッシュの長さを試すときは、`hwnpu-convert` で同じキャッシュの長さのものを 2 つ変換し、`-package` にカンマで並べます。

```bash
go run ./cmd/hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -quant kmeans-minmax -seq 1 -cache 16320 -parts 2 -o DIR/lfm3-16k-q1.mlpackage
```

```bash
go run ./cmd/hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -quant kmeans-minmax -seq 64 -cache 16320 -o DIR/lfm3-16k-q64.mlpackage
```

```bash
go run ./cmd/hwnpu-chat -package DIR/lfm3-16k-q1.parts.json,DIR/lfm3-16k-q64.mlpackage -config DIR/config.json -weights DIR/model.safetensors
```

## 6. OpenCode から使う

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

## 7. 限界

- OpenCode は最初に会話の題名を頼みます。このモデルは考えないよう指示しても考えてから答えるので 1〜2 分かかり、その間は本題の要求が待ちます
- 1.7B で 4 ビットの重みなので、知識の問いは誤ることがあります。考えさせたほうが正確です
- 文脈は 16320 トークンまでです。これより長い文脈には、キャッシュを塊に分けて注意を計算する方式が要ります ([docs/kv-cache.md](docs/kv-cache.md))
- Neural Engine 上の PLaMo-Embedding-1B は、まだ Hugging Face の実装と照合していません。`tools/ref/embed_golden.py` と `internal/chat` の `TestPlamoEmbeddingGolden` で比べられます (docs/design.md の 16.21)
- 実装の記録は [docs/design.md](docs/design.md) の 16.17 と 16.21 にあります
