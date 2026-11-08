# チャット、埋め込み、OpenAI 互換 API、OpenCode

[English](CHAT.md)

LFM3-1.7B と PLaMo-Embedding-1B を Apple の Neural Engine だけで動かし、ターミナルで使ったり、OpenAI 互換の API として OpenCode などのクライアントから使ったりします。Python は要りません。

同じコマンドが Intel の NPU (Core Ultra) でも動きます。モデルは OpenVINO IR に変換され、NPU のドライバがコンパイルします。実機ではまだ試していません。[INSTALL_INTEL.ja.md](INSTALL_INTEL.ja.md) の 6 節を見てください。

使うモデルは `-use` に名前で指定します。選べるモデルの名前と用途は `-list` で出ます。

```bash
go run ./cmd/hwnpu-chat -list
```

| 名前 | 用途 | 機能 | ターミナル | `-listen` のとき (3 節) |
|---|---|---|---|---|
| `LFM3-1.7B` (既定) | chat | `tools`、`thinking` | 対話する (1 節) | `/v1/chat/completions` |
| `LFM3-VL-2B` | chat | `tools`、`thinking`、`vision` | 画像を交えて対話する (1 節) | `/v1/chat/completions` |
| `LFM3-Audio-2B` | chat | `tools`、`thinking`、`audio` | 音声を交えて対話する (1 節) | `/v1/chat/completions` |
| `LFM3-Omni-2B` | chat | `tools`、`thinking`、`vision`、`audio` | 画像と音声を交えて対話する (1 節) | `/v1/chat/completions` |
| `gemma-4-E2B-it` | chat | `tools`、`thinking` | 対話する (1 節) | `/v1/chat/completions` |
| `gemma-4-E4B-it` | chat | `tools`、`thinking` | 対話する (1 節) | `/v1/chat/completions` |
| `Qwen3.5-4B` | chat | `tools`、`thinking` | 対話する (1 節) | `/v1/chat/completions` |
| `plamo-embedding-1b` | embedding | `query` | 埋め込みで文を検索する (4 節) | `/v1/embeddings` |

機能は、用途のほかにそのモデルができることです (名前は Ollama の capabilities に合わせています)。

| 機能 | |
|---|---|
| `tools` | リクエストが示す道具を呼び出す (`tools`、`tool_calls`)。OpenCode のようなエージェントから動かせる (6 節)。LFM3-1.7B はテキストだけだが、小さく速いのでエージェントの裏で動かしやすい |
| `thinking` | 答える前に考える (`reasoning_content`)。`-think` と `reasoning_effort` で切り替える |
| `query` | 検索の問いと検索される文書とを別々に埋め込む (`input_type`、4 節) |
| `vision` | ユーザーのメッセージの画像 (`image_url`) を読む。対話では `/image FILE` |
| `audio` | ユーザーのメッセージの音声 (`input_audio`、WAV) を読む。対話では `/audio FILE.wav` |

- ターミナルで動かすのは `-use` に指定した 1 つのモデルで、何をするかはモデルの用途で決まります。サーバーはいくつでも動かせます (例 `-use LFM3-1.7B,plamo-embedding-1b`)
- 名前の大文字と小文字は区別せず、一覧のリポジトリ名でも指定できます。一覧にない Hugging Face のリポジトリも `-use OWNER/NAME` で指定でき、hwnpu-chat が動かせる構造 (チャットは LFM3、埋め込みは PLaMo のエンコーダ) なら `config.json` から用途を判断します。認証の要るリポジトリは環境変数 `HF_TOKEN`

## 1. 対話する

```bash
go run ./cmd/hwnpu-chat
```

引数は要りません。最初の起動では次のことを自動で行います (M3 で、落とす時間を除いて約 4 分: 変換に 2 分少々、コンパイルに 2 分)。2 回目からはコンパイル済みのものを読み込むだけで、数秒から 1 分ほどで始まります。読み込みの間は進み具合を表示します。

1. Hugging Face の [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) から `config.json`、`tokenizer.json`、`model.safetensors` (3.5 GB) を `~/Library/Caches/hwnpu/models/LFM3-1.7B` に落とします。途中で切れても、もう一度起動すれば続きから落とします
2. Neural Engine 用に、6 ビットの表と grouped 注意で、全注意の層で分けて変換します (`lfm3-lut6-grouped-p8192-c262144-q8.parts.json` と `-q128-last`。それぞれ 7 部分で合わせて約 1.2 GiB と、重みのない注意のプログラム 1 つ)。生成するもの (1 回 8 トークン: 本物 1 つと推測 7 つをまとめて確かめる。3 節) と、1 回 128 トークンでプロンプトを読むものです (ロジットは最後の行だけ計算する)。2026 年 11 月 6 日までの `-q64` (1 回 64 トークン) より 2 割ほど速く読み、`-q64` はもう読まれないので消して構いません (design.md の 16.82)。2 つは 1 つの KV キャッシュを共有します。全注意の層の注意は、hwnpu-chat が自分の持つキャッシュに対して 8192 トークンの塊ごとに計算します (`-paged`、2 節)。このため 1 ステップの費用は会話の長さに比例し、キャッシュのメモリも会話とともに増えるので、文脈はモデルの上限の 262144 トークンです。2026 年 11 月 4 日までの既定 (`lfm3-lut6-grouped-c16320-q8.mlpackage`、文脈 16320、費用は容量に比例) のファイルはもう読まれないので消して構いません。2026 年 11 月までの既定は 4 ビットの表と tile 注意 (`lfm3-c16320-q8.mlpackage`) で、そのファイルはもう読まれないので消して構いません。bf16 のモデルとの近さは lut4 の 6 倍 (KL 1.39 → 0.23 nats)、コーディングの課題は 13〜27 往復が 6 往復になり、grouped 注意のおかげで 1 回の呼び出しは以前より 38% 速いままです ([docs/design.md](docs/design.md) の 16.41)
3. Neural Engine 向けにコンパイルします (1〜2 分)。結果は `~/Library/Caches/hwnpu` に残ります。同じ重みのファイル (生成用と読み込み用のプログラム、そのコンパイル結果) は macOS ではディスクの上で 1 つ分を共有し (APFS の clone。普通のファイルのまま)、プログラムを消したものと 30 日使わなかったもののコンパイル結果は起動時に消します (docs/design.md の 16.64)

起動したら、`>` に続けて話しかけます。

- モデルはまず考えて (`<think>`、薄い色で表示) から答えます。`/think off` で考えずに答え、`/think on` で戻ります
- `/reset` で会話を消し、`/exit` か Ctrl-D で終わります。Ctrl-C は答えを止めます。行末の `\` で次の行に続けます
- 入力はシェルのように編集できます: ↑ ↓ (Ctrl-P、Ctrl-N) で前に打ったものを出し (`~/Library/Caches/hwnpu/history/chat` に 1000 件まで残り、次の起動でも出せる)、← → と Home、End、Ctrl-A、Ctrl-E、Ctrl-K、Ctrl-U、Ctrl-W が使えます。日本語は 2 桁幅で扱います。貼り付けた複数行はそのまま 1 つの入力になり、改行は `⏎` と表示します。入力中の Ctrl-C は行を消し、空の行での Ctrl-C は終了です
- 読み込みの表示や説明は灰色で出し、Neural Engine に全部載ったら `all on the NPU` を緑、載らなかったら `NOT all on the NPU` を赤で出します。端末でないときと `NO_COLOR` を設定したときは色を付けません (hwnpu-agent も同じ)
- 会話の 2 回目以降は、前回からの差分だけを読みます
- 答えのあとに、読んだトークン数と生成の速さ (M3 で約 15 tokens/s) を出します

### 画像と音声を読む (LFM3-VL-2B、LFM3-Audio-2B、LFM3-Omni-2B)

`-use LFM3-VL-2B`、`-use LFM3-Audio-2B`、`-use LFM3-Omni-2B` で Liquid AI のマルチモーダルのチェックポイントを使えます。初回はそれを落とし (3.8、4.0、4.3 GB)、テキスト部を同じ手順で、ただし 8 ビットの表 (lut8、2 部分ずつ) で変換します。4 ビットの表では画像の説明を誤り、音声を聞き違えるためです (docs/design.md の 16.31)。チェックポイントにあるエンコーダも fp16 で変換します。視覚エンコーダは 512x512 の画像用 (画像はこの大きさに引き伸ばす)、音声エンコーダは 30 秒までのクリップ用です。

```bash
go run ./cmd/hwnpu-chat -use LFM3-VL-2B
```

- 対話では `/image FILE 質問` で画像を、`/audio FILE.wav 質問` で音声を送れます。サーバーはユーザーのメッセージの `image_url` (data URL) と `input_audio` (base64 の WAV) を読みます
- M3 で grouped 注意 (2026 年 11 月からの既定) なら 1 トークン約 36 ms (以前は 41 ms)。画像の説明中は約 70 ms (投機的デコードの推測が当たらない)。約 300 トークンの画像つきプロンプトの読み込みは 0.7 秒。LFM3-1.7B と同じく 4096 の段階は付きません (2 節の `-tier`)
- テキスト部は LFM3-1.7B と同じく全注意の層で分けて変換し (`lfm3-lut8-grouped-p8192-c128000-q8.parts.json` と `-q128-last`、7 部分ずつ。11 月 6 日までの `-q64` は消して構いません)、文脈はモデルの上限の 128000 トークンです。キャッシュは会話とともに 1 トークン 12 KiB ずつ増え、128000 で 1.46 GiB。M3 では全部が Neural Engine に載り、2 回目の起動は 1.4 秒と 4.2 秒で、画像の説明、計算、富士山の高さの答えは分けない形 (文脈 16320) と 1 トークンも違わず、生成は 24.0、33.2、36.0 tokens/s (分けない形は 11.9、26.7、29.4)。プロンプトの先頭に置いた合言葉は 16K、64K、120K トークンで正しく答え、読み込みは 20 秒、171 秒、497 秒、その長さでの生成は 25.2、58.6、68.1 ms/token。LFM3-Audio-2B と LFM3-Omni-2B も同じ形で動き (音声の書き起こし、画像の説明)、120K の合言葉を正しく答えました (読み込み 525 秒と 512 秒、プロセスのメモリ 1.7 GB)。2026 年 11 月 4 日、design.md の 16.63 と 16.65。これは全部の鍵を読んだときで、2 段の注意 (`-sparse`、既定) では 120K の読み込みが VL 431 秒、Audio 516 秒、Omni 501 秒 (同じ日の VL の全部を読む形は 583 秒)、その長さの 1 回の呼び出しは 111〜120 ms (同じく 179 ms) で、3 つとも合言葉を答えました (11 月 6 日、16.65)。それまでの `lfm3-lut8-grouped-c16320-*` はもう読まれないので消して構いません
- 返答は文字です。LFM3-Audio の音声出力は `hwnpu-tts` (5 節)
- ほかの画像の大きさやクリップの長さを使うには、エンコーダを自分で変換して `-package`、`-vision`、`-audio` で渡します (5 節)

### Gemma 4 E2B と E4B を使う

`-use gemma-4-E2B-it` で Google の Gemma 4 E2B (指示調整版) を使えます。手順は同じで、`~/Library/Caches/hwnpu/models/gemma-4-E2B-it` に `model.safetensors` (10.2 GB。音声と画像の部分も入っているが、変換するのはテキストだけ) を落とし、8 トークンと 64 トークンのプログラムを 6 ビットの表 (lut6) で、LFM3-1.7B と同じく全注意の層の前後で分けて変換します (`gemma4-lut6-p8192-c131072-q8.parts.json` と `-q128-last`。それぞれ 8 部分で約 1.6 GiB と、重みのない注意のプログラム 1 つ。プロンプトは 1 回 128 トークンで読み、KV を共有する後ろの 20 層は最後のトークンの行だけを計算するので、16K で 27.8 → 13 秒、64K で 205 → 95 秒 (design.md の 16.82、16.83)。11 月 6 日までの `-q64` は消して構いません)。初回の変換は M3 で約 3 分、コンパイルも約 3 分で、2 回目からの起動は 2 秒ほどです (分けない 2 部分のプログラムは、Neural Engine のコンパイルのキャッシュに残らず毎回 50〜100 秒かかっていました。docs/design.md の 16.42、16.43、16.63)。2026 年 11 月 4 日までの `gemma4-lut6-c16320-*` と、それより前の `gemma4-c16320-*` はもう読まれないので消して構いません。

```bash
go run ./cmd/hwnpu-chat -use gemma-4-E2B-it
```

- チャットの書式は Gemma 4 の `chat_template.jinja` を Go に移したもので、思考 (`<|think|>`、`reasoning_content` に返す) と道具の呼び出しに対応します
- 層ごとの埋め込みの表 (4.7 GB) はホストに置き、トークンごとに 1 行 (17.5 KB) を読みます
- 文脈はモデルの上限の 131072 トークンです。全注意の 7 層のうち 4 層は層 14 の鍵と値を使うので、hwnpu-chat が持つキャッシュは 3 層分で、1 トークン 6 KiB (131072 で 0.75 GiB) ずつ会話とともに増えます。M3 の実測 (design.md の 16.63): 短い質問の生成は 23.6〜36.0 tokens/s (分けない文脈 16320 のプログラムは同じ答えで 13.3〜28.2)。81,691 トークンのプロンプトを 299 秒 (273 tokens/s) で読み、先頭に置いた合言葉を正しく答え、その長さで 1 トークン 96.7 ms。そのときのプロセスのメモリ (footprint) は 1.1 GB。16K、64K、125K でも合言葉を正しく答え、読み込みは 27 秒、224 秒、709 秒、その長さでの生成は 23.9、82.4、119.5 ms/token、メモリは 0.7、1.2、2.4 GB。これは全部の鍵を読んだときで、2 段の注意 (`-sparse`、既定) では 64K と 125K の読み込みが 204 秒と 521 秒、生成が 73.9 と 92.4 ms/token で、どちらも正しく答えました (design.md の 16.65)
- Gemma 4 は 4 ビットの表 (lut4) では損失が大きく (富士山の高さや『こころ』の作者を誤る)、6 ビットを既定にしています。`-package` で自分で変換するときは `hwnpu-convert` の `-format` を省けば lut6 になります。詳細は docs/design.md の 16.25 と 16.26
- `-use gemma-4-E4B-it` で一回り大きい E4B (ダウンロード 16.0 GB、幅 2560 の 42 層) を同じ手順で使えます。E2B より遅くなります。文脈は同じく 131072 で、キャッシュは 1 トークン 16 KiB (131072 で 2 GiB)。生成用と読み込み用のプログラム 2 本 (6.5 GiB) は Neural Engine のコンパイルのキャッシュ (約 4 GiB) に収まらないので、生成用の 1 本 (1 回 8 トークン) だけを変換して読み込み、プロンプトもそれで読みます (`-prefill auto`)。M3 で 2 回目からの起動は 4 秒ほど、短い質問の生成は 13〜16 tokens/s。代わりに長いプロンプトの読み込みは 2 本のときの 4.4 倍かかります (16K で 183 秒、2 本なら 42 秒)。`-prefill on` で 2 本にできますが、起動は毎回 80 秒前後、質問ごとに 1〜3 秒の切り替えがかかります。125K の合言葉は (2 本で) 正しく答えましたが、スワップが 14 GB に達しました (design.md の 16.65)。1 本で 2 段の注意 (既定) のときも 64K と 125K の合言葉を答え、その長さの 1 回の呼び出しは 217 ms と 276 ms ですが、読み込みは 21 分と 56 分かかります。16 GB の機械では E2B を勧めます。12B 以上の Gemma 4 は別のアーキテクチャで、hwnpu-chat では動きません

### PLaMo 3 Translate で訳す

Preferred Networks の翻訳モデル PLaMo 3 Translate 2B は `-use plamo-3-translate-2b` で使えます。ほかのモデルと同じく、初回にチェックポイント (5.8 GB) を落として変換します。Neural Engine には 8 ビットの表 (lut8)、Intel の NPU には int8 で、全注意の 3 層 (7、15、23) の前後で分けます (文脈は 262144 トークン、キャッシュは会話とともに 1 トークン 7 KiB ずつ増える)。AMD の NPU では IRON で書き出したものを `-iron` で動かします (README の AMD の節)。打った文がそのまま訳され、それまでの原文と訳文の組が対訳の履歴としてプロンプトに入ります (訳語や文体がそろう)。訳す先は `-to` (既定は、日本語の文字を含む文は英語、ほかは日本語)、`/to LANG` で途中で変えると履歴は消えます。言語は小文字のコードで、`en`、`ja`、`zh-hans`、`ko`、やさしい日本語の `ja(easy)` などです。`-from` で原文の言語、`-form plain` か `polite` で日本語の常体と敬体を指定でき、`-system` は参考文脈になります。考える部分や道具の呼び出しはありません。

`-use plamo-3-translate-0.8b` は同じモデルの小さい版です (ダウンロード 1.8 GB)。M3 では生成が 2B の約 3 倍速く (1 秒に 53 トークン、2B は 17)、2 本のプログラム (1 回 8 トークンと 128 トークン、合わせて 1.38 GiB) が Neural Engine のコンパイルのキャッシュに収まるので、1600 トークンの文書を 1.2 秒で読みます。20 文 (英 → 日と日 → 英) のうち 8 文は 2B と一字一句同じ訳で、残りの 12 文も正しい言い換えでした。

```bash
go run ./cmd/hwnpu-chat -use plamo-3-translate-2b
go run ./cmd/hwnpu-chat -use plamo-3-translate-2b -to ja -form plain
go run ./cmd/hwnpu-chat -iron ~/models/plamo-3-translate-2b.amd-npu2-q4
```

重みの形式 (design.md の 16.71): M3 の Neural Engine で、次のトークンが fp16 と同じになるのは lut8 で 98%、lut6 で 89%。1 回の呼び出しは lut8 が 57.8 ms、lut6 が 53.7 ms。lut4 は使えません (CPU で KL 1.91 nats。Intel の int8 は 0.0025)。lut8 の 2 本のプログラム (4.9 GiB) は Neural Engine のコンパイルのキャッシュ (約 4 GiB) を超えるので、生成用の 1 本 (1 回 8 トークン) だけでプロンプトも読みます。短い文ではほとんど変わりませんが、数千トークンの文書は読み込みが 10 倍ほど遅くなります。速さを優先するなら `-format lut6` で、2 本になります。

サーバーでは、`chat_template_kwargs` に `target_lang`、`source_lang`、`form`、`context`、`glossary` (`[{"source": "NPU", "target": "NPU"}]`) を渡します (vLLM と同じ)。`messages` は原文 (user) と訳文 (assistant) の交互で、最後が今回の原文です。

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"The meeting has been rescheduled."}],"chat_template_kwargs":{"target_lang":"ja"},"temperature":0}'
```

## 2. オプション

| フラグ | 既定 | |
|---|---|---|
| `-use NAMES` | `LFM3-1.7B` | 動かすモデルをカンマ区切りで: `-list` の名前かリポジトリ `OWNER/NAME`。ターミナルでは 1 つ |
| `-list` | | モデルの一覧 (用途、大きさ、落としてあるか) を出して終わる |
| `-listen ADDR` | なし | ターミナルの代わりに OpenAI 互換のサーバーを `ADDR` (例 `127.0.0.1:8080`) で開く (3 節) |
| `-think` | `true` | 答える前に考えさせる |
| `-think-budget N` | `1024` | 考えるトークン数の上限。超えたら答えに移らせる (0 は無制限) |
| `-tier A,B` | `auto` | `-context` より短いキャッシュ。会話はプロンプトが収まる最も短いもので始まり、伸びたら長いものへ移る (1 ステップの費用は、使った量ではなくキャッシュの容量に比例する)。空で無し。`auto` は 4 ビットの表のモデルなら `4096`、6 と 8 ビットなら無し: Neural Engine はコンパイルしたプログラムを約 4 GiB の独自のキャッシュに持ち、lut6 の LFM3-1.7B のプログラム 4 本 (4.8 GiB) は互いに追い出し合って起動のたびにどれかをコンパイルし直す (4 回続けて起動して 110、92、77、62 秒)。2 本の 2.4 GiB なら残り、起動は 3 秒 |
| `-draft N` | `8` | 生成するプログラムの 1 回のトークン数: 本物 1 つと推測 N-1 個をまとめて確かめる (投機的デコード、3 節)。`0` か `1` で 1 回 1 トークン。`-iron` (AMD) では書き出しで決まる: `--verify 3` で書き出したものは 1 回に推測 2 つを確かめる (今は Qwen3.5)。`HWNPU_IRON_DRAFTS=0` で使わない |
| `-mtp auto\|off` | `auto` | `auto`: チェックポイントに複数トークン予測の層 (Qwen3.5 の `mtp.`) があれば、その層の推測も使う (多くて 2 つ、`HWNPU_MTP_DRAFTS` で変えられる)。`off`: プロンプトからの推測だけ |
| `-prefill auto` | `auto` | 生成用のほかに、1 回 64 トークンでプロンプトを読むプログラムを持つか。`auto` は持つが、2 本の重みが Neural Engine のコンパイルのキャッシュ (約 4 GiB) を超えるなら持たず (Gemma 4 E4B)、生成用のプログラムがプロンプトも読む (長いプロンプトは遅い)。`on` で常に持ち、`off` で持たない |
| `-prompt-cache DIR` | コンパイルキャッシュの `prompts` | 読んだプロンプトをディスクに残し、起動し直した後も、先頭が同じプロンプト (エージェントの system と道具) を読み直さない (3 節)。`off` で無し |
| `-system TEXT` | なし | 対話の system メッセージ (翻訳モデルでは参考文脈) |
| `-to LANG` | なし | 翻訳モデル (PLaMo 3 Translate) が訳す先の言語。なしなら日本語を含む文は `en`、ほかは `ja` |
| `-from LANG` | なし | 翻訳モデルの原文の言語 (なしならモデルが判断する) |
| `-form plain\|polite` | なし | 日本語へ訳すときの常体と敬体 |
| `-n N` | `0` | 1 つの答えの最大トークン数 (0 は文脈が尽きるまで) |
| `-temp`、`-top-k`、`-top-p`、`-rep` | `0.1`、`50`、`0`、`1.05` | サンプリング (温度、上位 k、上位 p、答えの中の繰り返しの罰則) |
| `-context N` | `0` | プロンプトと答えを合わせた最大トークン数。`0` は、注意で分けたモデル (`-paged`) ならモデルの上限 (LFM3-1.7B と Qwen3.5 は 262144、LFM3-VL、Audio、Omni は 128000、Gemma 4 は 131072)、それ以外は 16320。変えると、その長さで変換し直す。分けないモデルで 16320 を超えるとキャッシュを塊に分ける |
| `-paged N` | `0` | 全注意の層でモデルを分け、hwnpu-chat が持つキャッシュに対して N トークンの塊ごとに注意を計算する。1 ステップの費用が `-context` ではなく会話の長さに比例し、キャッシュのメモリも会話とともに増える (LFM3-1.7B で 1 トークン 12 KiB、262144 で 3 GiB。メモリの 4 分の 1 を超えるなら文脈を縮める)。`0` はモデルが許すとき 8192 (LFM3 とその VL、Audio、Omni、Gemma 4、Qwen3.5。Neural Engine でも Intel の NPU でも)、`-1` で分けない (そのとき `-tier` が効く) |
| `-sparse` | `true` | 注意で分けたモデル (`-paged`) で会話が約 48K を超えたら、1 回の呼び出しで鍵 16384 (プロンプトを読むときは 32768) だけを正確に読み (その呼び出しにとって重い 8 トークンのバケット、最後の 512、最初の 4)、ほかのバケットは鍵と値の平均 1 つで読む。128K 以上で呼び出しが数分の 1 になる ([docs/design.md](docs/design.md) の 16.61)。`false` で毎回キャッシュ全体を読む |
| `-dir DIR` | `~/Library/Caches/hwnpu/models` | モデルの置き場。モデルごとに `DIR/名前` にチェックポイントと変換したプログラムを置く |
| `-package A,B` | なし | `hwnpu-convert` で自分で変換したチャットのプログラムを使う (5 節) |
| `-dump DIR` | なし | サーバーで、要求、プロンプト、生成した文を要求ごとに保存する |
| `-embed-seq A,B` | `128,1024` | 埋め込みモデルを変換する長さ。文はそれが収まる最も短いもので動き、最も長いものが 1 つの文の最大トークン数になる |
| `-embed-package A,B` | なし | `hwnpu-convert` で自分で変換した埋め込みのプログラムを使う (エンコーダは `-seq` だけ)。`config.json` と `model.safetensors` はモデルの置き場に置く |

`-think` から `-context` まで、`-package`、`-config`、`-weights`、`-tokenizer` はチャットのモデルの、`-embed-` で始まるものは埋め込みモデルのフラグです。`-package` とその仲間 (と `-embed-package`) は、`-use` にその用途のモデルが 1 つのときだけ使えます。

- LFM3-1.7B (`-paged` 8192、既定) では、1 ステップの費用は会話の長さに比例します。文脈を 262144 にしても、短い会話は遅くなりません。M3 (16 GB) での実測は次のとおりです ([docs/design.md](docs/design.md) の 16.58、16.59)。短い会話の生成は分けない形 (`-paged -1 -context 16320`) の 1.3〜1.7 倍の速さで、答えは一字一句同じでした

| 会話の長さ | 生成 (M3) | プロンプトを読む時間 | プロセスのメモリ | `-sparse` (既定) の生成 | 読む時間 | メモリ |
|---|---|---|---|---|---|---|
| 数百 | 28〜33 ms/token (下書きの採用込み) | — | — | (同じ) | — | — |
| 16K | 約 43 ms/token | 19 秒 | 約 0.4 GB | (同じ) | (同じ) | 約 0.5 GB |
| 64K | 約 93 ms/token | 2.5 分 | 約 1.0 GB | 約 78 ms/token | 2.9 分 | 約 1.6 GB |
| 128K | 約 165 ms/token | 8.6 分 | 約 1.8 GB | 約 90 ms/token | 7.8 分 | 約 2.5 GB |
| 250K | 約 320 ms/token | 29 分 | 約 3.2 GB | 約 126 ms/token | 20 分 | 約 4.1 GB |

- キャッシュは 1 トークン 12 KiB ずつ増え、262144 で 3 GiB です。どの長さでも、先頭に置いた合言葉を正しく答えました
- 250K のプロンプトを一度に読むと 30 分ほどかかります。読んだプロンプトはディスクに残るので (`-prompt-cache`)、2 回目からは読み直しません
- 会話が約 48K を超えると、2 段の注意 (`-sparse`、既定) が読む鍵を減らします。全注意の層の 1 回は、128K で塊 8192 の呼び出し 16 回が 7 回に、256K で 32 回が 11 回になります。上の表の `-sparse` の列は M3 での実測で (2026 年 11 月 6 日、design.md の 16.61)、左の 3 列は全部を読む形 (`-sparse=false`) の 11 月 4 日の値です。同じ日に測り直した全部を読む形と比べると、1 回の呼び出しは 128K で約半分、250K で 0.42 倍、プロンプトの読みは 0.76 倍と 0.56 倍で、64K の読みだけは 1 割遅くなります。メモリは要約と正確に読む鍵の写しの分だけ増え、キャッシュは 1 トークン 14 KiB (262144 で 3.38 GiB) です。CPU で LFM3-1.7B を 128K で確かめました (文脈の 10%、50%、90% に置いた合言葉を 3 つとも答え、次トークンの損失の増えは 0.007〜0.009 nats)。M3 でも 128K の文書の 3 つの合言葉を答えました。128K を超えると、`-sparse` の有無によらず、LFM3-1.7B 自身が長い文書の遠くの合言葉を外します (CPU での 256K の確認では 90% のものだけ答えた)
- 分けないモデル (`-paged -1`) では、1 ステップの費用は使った量ではなくキャッシュの容量に比例します。文脈を長くするほど 1 トークンの生成は遅くなり、`-tier 4096` (4 ビットの表のときの既定) では会話が収まるうちは 4096 のキャッシュで約 2 倍速く動き、伸びたら `-context` のものへ移ります。段階ごとのプログラムも変換して読み込みます。OpenCode には 16K 以上が要ります

| `-context` | 生成 (M3) | プロンプトを読む速さ | KV キャッシュ |
|---|---|---|---|
| 4096 | 約 28 ms/token | — | 約 50 MiB |
| 16320 (既定) | 約 64 ms/token | 約 570 tokens/s | 約 190 MiB |
| 32768 | 約 138 ms/token | 約 300 tokens/s | 約 390 MiB |
| 65536 | 約 363 ms/token | 約 150 tokens/s | 約 770 MiB |

- 16320 を超える文脈では、Neural Engine の 1 次元 16384 の上限に収まるよう、キャッシュを 16384 トークンずつの塊に分けて持ちます。32768 では 25,288 トークン、65536 では 53,736 トークンのプロンプトの先頭に置いた合言葉を正しく答えました
- サーバーはほかの会話を、その会話が使った分のキャッシュの写しで 2 本まで取っておきます (2026 年 11 月 4 日まではキャッシュ全体の写しでした)

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
- `-think-budget` (既定 1024) は考えるトークン数の上限です。超えたら「もう時間がない」という一文で思考を閉じ、そこまでの考えで答えさせます。小さなモデルは放っておくと数千トークン考え (1 トークン 64 ms なら数分)、エージェントはその間待たされます。切ったときはログに `thinking cut` と出ます。`0` で無制限
- 要求は 1 つずつ処理します。前の要求と共通の先頭は読み直さず、`usage.prompt_tokens_details.cached_tokens` にその数が入ります。別の会話が割り込んでも、長い会話は 2 本まで取っておいて戻ります
- 読んだプロンプトはディスクにも残します (`-prompt-cache`、既定は `~/Library/Caches/hwnpu/prompts`)。起動し直した後、先頭が同じプロンプトは、共通の先頭の中の最後の印 (512 トークンごとと各プロンプトの終わりに取る) からディスクで戻り、残りだけを読みます。ログに `N tokens from the prompt store` と出ます。モデルと変換ごとに 1024 トークン以上の系列を 4 本まで残し、LFM3-1.7B で 1 トークンあたり約 19 KiB (全注意の層の K/V が 1 トークン 12 KiB、印ごとのリングと窓とロジットが 3.5 MB。OpenCode の最初の要求 7.9K トークンで 145 MB)、使われていないものから消します。M3 では、その要求を冷えた状態で読むと 14.6 秒、ストアから戻すと 0.2 秒です。`-prompt-cache off` で止まります
- 堂々めぐりを始めたら止めます。短い同じ並びの繰り返しのほか、同じ行 (やトークンの並び) が 3 回目以降も戻ってくる繰り返しです。小さなモデルは知らないことを聞かれるとこれに落ちます (「- Gifu? Already.」「- Mie? Already.」…)。下書きをもう 1 回清書する答えは止めません。思考の中なら `-think-budget` に届いたときと同じく思考を閉じて答えさせます。ログに `stopped repeating itself` か `thinking cut going round in circles` と出ます
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

### 画像を読む（LFM3-VL-2B）

`-use LFM3-VL-2B` (1 節) なら以下を自動で行います。ほかの画像の大きさを使うときは、テキスト部を上と同じに変換し、視覚エンコーダを `-arch lfm3-vision` で画像の大きさごとに変換します（Neural Engine は固定の形で動くため）。`-vision` にカンマで並べると、ユーザーのメッセージの `image_url`（data URL）を読みます。画像は、縦横比がいちばん近いプログラムの大きさに引き伸ばされます。対話では `/image FILE 質問` で画像を送れます（docs/design.md の 16.28 と 16.31）。

```bash
go run ./cmd/hwnpu-convert -arch lfm3-vision -image 512x512 -format fp16 -config DIR/config.json -weights DIR/model.safetensors -o DIR/vision-512x512.mlpackage
```

```bash
go run ./cmd/hwnpu-chat -package DIR/lfm3vl-q1.parts.json,DIR/lfm3vl-q64.mlpackage -vision DIR/vision-512x512.mlpackage -config DIR/config.json -weights DIR/model.safetensors
```

### 音声を読む（LFM3-Audio-2B）

`-use LFM3-Audio-2B` (1 節) なら以下を自動で行います。ほかのクリップの長さを使うときは、テキスト部を上と同じに変換し、音声エンコーダを `-arch lfm3-audio-encoder` で変換します（`-frames` は受け付ける最長のメルフレーム数で、10 ms ごと。既定の 3000 は 30 秒）。`-audio` にカンマで並べると、ユーザーのメッセージの `input_audio`（base64 の WAV）を読みます。対話では `/audio FILE.wav 質問` で送れます。読めるのは WAV だけで、16 kHz 以外は変換します。返答は文字です。読み上げは下の `hwnpu-tts` で行います（docs/design.md の 16.29 と 16.31）。

```bash
go run ./cmd/hwnpu-convert -arch lfm3-audio-encoder -frames 3000 -format fp16 -config DIR/config.json -weights DIR/model.safetensors -o DIR/audio-30s.mlpackage
```

### 音声で話す（LFM3-Audio-2B の TTS）

`hwnpu-tts` が文章を読み上げて WAV（24 kHz）に書きます。テキスト部を KV キャッシュ付きで `-hidden-out 1` を付けて変換し（最終の隠れ状態も出力する）、depthformer（`-arch lfm3-depthformer`）と、`audio_tokenizer/` のコーデックのデコーダ（`-codec-frames`）を別に変換します。チャットの返答を音声にする機能はまだありません。実物での確認はまだです（docs/design.md の 16.30）。

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
go run ./cmd/hwnpu-tts -model DIR -text DIR/tts-q1.mlpackage -depth DIR/depth.mlpackage -codec DIR/codec.mlpackage -lang ja -o out.wav こんにちは
```

## 6. OpenCode から使う

プロジェクトのディレクトリで次のように起動すると、サーバーと OpenCode をまとめて動かします。

```bash
go run ./cmd/hwnpu-chat -opencode
```

- モデルを読み込んだあと、30000〜49151 番の空いたポート (macOS が外向きの接続に使う 49152 番以降は避けます) でサーバーを開き、そのサーバーを使う OpenCode の設定を `~/Library/Caches/hwnpu/opencode/opencode.json` に書いて、環境変数 `OPENCODE_CONFIG` で OpenCode に渡します。プロジェクトのファイルには何も書きません
- OpenCode はこの端末の標準入力、出力、エラーをそのまま受け継ぐので、画面の描画、マウス、キー操作は単独で起動したときと同じです。サーバーのログは `~/Library/Caches/hwnpu/opencode/server.log` に書き、画面を乱しません
- OpenCode を終えるとサーバーも止まり、OpenCode の終了コードで終わります
- フラグのあとの引数は OpenCode に渡します。`-` で始まる OpenCode のオプションは `--` のあとに書きます (例: `hwnpu-chat -opencode -- --continue`、`hwnpu-chat -opencode run "main.go のバグを直して"`)。`-listen` を付けるとそのアドレスで開き、`-opencode-bin` で OpenCode のコマンドを変えられます
- `opencode run` は標準入力が閉じるまで待つので、スクリプトから使うときは `< /dev/null` を付けます
- OpenCode はプロジェクトの `opencode.json` を `OPENCODE_CONFIG` より優先します。プロジェクトに別の `hwnpu` プロバイダーを書いてあると、そちらのアドレスが使われます

手で設定するときは、プロジェクトのディレクトリに `opencode.json` を置きます。モデル名は `GET /v1/models` が返す名前にします。

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

- `limit.context` はサーバーの文脈の長さ (起動時の `ready: context N tokens`) に合わせます。OpenCode の要求は system と道具の説明だけで約 7.4K トークンあるので、8K の文脈では 1 往復で要約が始まってしまいます
- 1.7B のモデルは見当違いのコマンドを作ることがあります。`bash` は許すコマンドを絞っておくと安全です
- M3 での例: 「main.go の add 関数のバグを直して」で、ファイルを読み、`a - b` を `a + b` に直し、`go run` で確かめるまでを 12 往復で行いました。最初の要求は 7.4K トークンを 13 秒で読み、2 回目以降はプロンプトの 99% を再利用して 0.1〜0.5 秒で読みます。生成は 1 トークン約 64 ms (約 15 tokens/s) です

## 7. hwnpu-agent: エンジンを直接呼ぶコーディングエージェント

`hwnpu-agent` は、read、edit、write、bash の 4 つの道具を持つコーディングエージェントです。[Pi](https://pi.dev) のループを Go で書き、hwnpu-chat と同じエンジンを HTTP を挟まずに呼びます。OpenCode の最初の要求は system prompt と 10 個の道具で約 7.4K トークンですが、hwnpu-agent の system prompt と 4 つの宣言は約 630 トークンで、文脈 (LFM3-1.7B は 262144) のほぼ全部を課題に使えます。脇の要求 (OpenCode は最初に題名を頼みます) もないので、課題がすぐ始まります。

```bash
go run ./cmd/hwnpu-agent "main.go の add 関数のバグを直して"
go run ./cmd/hwnpu-agent -C ~/src/demo -yes "mul 関数を足して mul(2, 3) を表示して"
go run ./cmd/hwnpu-agent
```

- コマンドラインに課題を書くとそれを行って終わります (モデルが答えたら 0、それ以外は 1)。書かなければ端末で課題を受け付け、1 つの会話として続きます (`/reset` でやり直し、`/exit` で終了、Ctrl-C で今の課題を止めます)
- `-C` (既定はカレントディレクトリ) の中で働きます。パスはそこからの相対、コマンドはそこで実行し、その外のファイルには書きません。そこに `AGENTS.md` があれば system prompt に入れます (4000 文字まで)
- コマンドの実行とファイルの変更の前に端末で確認します。`-yes` なら確認せず、`-allow "go *,ls*,cat *"` ならパターンに合うコマンドは確認なしで実行します (`&&` や `;` でつないだ各部分が合うこと)。端末がなく `-yes` もないときはそれ以外を実行しません (スクリプトからうっかり 1.7B のモデルを野放しにしないため)
- `-tools search` で 5 つ目の道具 search を出し、それだけにすると、知識をモデルに頼らず調べて答えるアシスタントになります。1.7B のモデルは事実の記憶が弱く (lut4 では富士山を「約 4,000 m」、『こころ』の作者を別人と答えました)、検索結果が前にあれば数値を写して出典を挙げます。`-search` で検索先を選びます。`wikipedia` (鍵なし。記事は主題の名前で見つかるので、山・川・小説・人物の名前であって文の質問ではない)、`brave` (Brave Search API、鍵は `BRAVE_API_KEY`)、ウェブなら [SearXNG](https://searxng.org) の URL (`http://localhost:8888` など。DuckDuckGo はブラウザ以外の要求を拒むので入れていません)。小さなモデルは検索するかどうかを安定して決められず、プロンプトの一文で変わるので、`-search-first` で各課題の最初の往復を検索にします。モデルが記憶から答えたらその往復は捨て、課題の 1 行目で検索したものに置き換えます。答えが前にあっても検索し続けるので、1 課題の検索は 4 回までで、5 回目には手元の結果から答えるよう求めます。`-tools search,fetch` なら fetch も出し、結果の抜粋に答えが無いときにページを文章として読みます (先頭 6000 文字、タグとスクリプトは除く)。fetch があれば思考は時間に見合います。M3 で Wikipedia を使った 5 問は、思考ありで 190 秒、なしで 201 秒。思考ありでは英語で検索して英語版の記事を fetch し、富士山の高さ (3,776.24 m)、日本一長い川 (信濃川)、市の人口を出典つきで当て、17 × 23 は計算し、『こころ』の作者だけ Wikipedia の検索が「『こころ』 作者」で小説を出さず外しました。思考なしは 5 問中 2 問で、富士山のページを読むと長い混乱した答えになりました。search だけ (fetch も思考もなし) では 1 問: 富士山を記事の冒頭から 1 回の検索 (約 3 秒) で。SearXNG でウェブを検索すると (`-search http://localhost:8888`、思考あり、324 秒) 5 問中 4 問が正解で『こころ』も当たり、残る富士山は記事の後ろの節にある別の山の高さを混ぜて崩れました。ウェブ検索は Wikipedia 自身の検索が外すものを見つけ、残る弱さは 1.7B のモデルの側です (日本語の名前をローマ字にして検索して外す、長いページから違う数字を拾う)。詳しくは [docs/design.md](docs/design.md) の 16.40
- `-tools read,bash` で道具を減らし (コードについての質問など)、`-system FILE` で system prompt の冒頭を差し替え、`-think=false` で考えずに答え、`-draft` で hwnpu-chat と同じ投機的デコードの幅 (8) を変え、`-tier` で会話が短いうちに使う短いキャッシュ (4096) を変え、`-keep-thinking` で課題の途中の考えを会話に残し、`-steps` で往復の上限 (30)、`-n` で答えの長さ (2048) を変え、`-dump DIR` で要求、プロンプト、生成文を JSON に書き出して prompt を比べられます
- 会話が文脈に収まらなくなるときも、モデルに要約は頼みません。答えのために `-n` トークンを残せなくなったら、古い道具の結果から順に先頭 1 行と行数に畳み、最新 4 つはそのまま残します
- 1.7B のモデルには規則より強制が効きます (Pi を小さなモデル向けに調整した little-coder の知見)。会話の中で read していないファイルへの `edit` と `write` は断り、先に読むよう促します。思考は `-think-budget` トークン (1024、hwnpu-chat と同じ) で閉じ、考えた分から答えさせます。最終回答を書いたあとに道具の呼び出しを付けることを繰り返すときは 2 回目の回答で課題を終え、同じ呼び出しが 2 回続いたら実行せずに止めるよう促し、3 回目と往復切れのときは道具なしの要求を 1 回出して要約を書かせます。`main.go` のつもりで書く `/main.go` のようなパスは相対として扱い、見たことのないディレクトリへの `cd` は断って残りのコマンドを示し、作業ディレクトリの長い絶対パスは `.` にして見せます
- M3 での例、「main.go の add 関数のバグを直して」(`a - b` → `a + b`): 8 要求、53 秒 (モデルがコンパイル済みのとき。読み込みはそのとき約 15 秒、初回は Core ML のコンパイルで約 3 分)。最初の要求は 629 トークンを 1.3 秒で読み、以後は新しいメッセージの分だけ (0.1〜0.4 秒)、生成は 15〜16 tokens/s でした。ファイルを読み、edit で 1 行を直し、読み直し、`cd /tmp` を試して断られ、`go run main.go` で 5 を見て、要約を書いて終わりました。前の形の prompt では 1.7B のモデルの弱さが見えます。system prompt に作業ディレクトリの絶対パスを書くと 30 往復写し間違え続け、`/main.go` を「外」と断るとまた 30 往復、最終回答のあとに `echo done` を付け続けて止まりませんでした。詳しくは [docs/design.md](docs/design.md) の 16.35 にあります

## 8. 限界

- OpenCode は最初に会話の題名を頼みます。このモデルは考えないよう指示しても考えてから答え、その間は本題の要求が待ちます。`-think-budget` (既定 1024 トークン、1 分前後) で待ちに上限が付きます
- 1.7B で 4 ビットの重みなので、知識の問いは誤ることがあります。考えさせたほうが正確です
- 会話が長くなると生成が遅くなります (M3 で 64K で 1 トークン約 0.1 秒、250K で約 0.3 秒)。250K のプロンプトを読むのに 30 分ほどかかります
- PLaMo-Embedding-1B は 8 ビットの表 (lut8) で変換します。4 ビットでは埋め込みの向きがずれます (fp16 とのコサイン類似度 0.76)。M3 での埋め込みは、Hugging Face の実装とのコサイン類似度が 0.997〜0.999 で、CPU で動かしたときと同じです (`tools/ref/embed_golden.py` と `internal/chat` の `TestPlamoEmbeddingGolden`)。最終層の MLP が fp16 の範囲を超える値を作るので、変換では最終層を 1/64 倍で計算します (`head-scale`。docs/design.md の 16.34)。以前に変換した lut4 の `embed-s128.mlpackage` と `embed-s1024.mlpackage` はもう使わないので、消してかまいません
- 実装の記録は [docs/design.md](docs/design.md) の 16.17、16.23、16.35 にあります
