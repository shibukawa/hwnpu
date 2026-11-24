# コマンド

[English](cli.md)

コマンドの設計: 家族の分け方、引数の解析、端末の表示、Ctrl+C、手入れ用の `bonsaiai` コマンド。使い方は
[CHAT.ja.md](../../CHAT.ja.md)、[USAGE_IMAGE.ja.md](../../USAGE_IMAGE.ja.md)、[USAGE_AUDIO.ja.md](../../USAGE_AUDIO.ja.md)、
[USAGE_DEV.ja.md](../../USAGE_DEV.ja.md)、[USAGE_LLM_AMD.ja.md](../../USAGE_LLM_AMD.ja.md)、[PROBE.ja.md](../../PROBE.ja.md) を見る。

## コマンドの家族

コマンドは用途でまとめる。1 つの家族が 1 つの実行ファイルで、サブコマンドを持つ。用途ごとだった古い実行ファイルは
ラッパーを残さずに消した。

| 実行ファイル | サブコマンド | 備考 |
|---|---|---|
| `bonsaiai` | `status`、`issue`、`cache` (`ls`、`prune`、`clear`、`clean`)、`memory` | 機械の手入れ: NPU、キャッシュ、メモリの区分、不具合の報告 |
| `bonsaiai-chat` | `chat` (既定)、`agent`、`search`、`serve`、`opencode`、`systemone`、`list` | 言語モデル: チャット、コーディングエージェント、埋め込み、OpenAI 互換 API、d1 の判定 |
| `bonsaiai-image` | `generate`、`depth`、`face-detect`、`pose`、`detect`、`ocr`、`label`、`embed`、`segment`、`convert`、`list` | 画像の生成と視覚のモデル |
| `bonsaiai-audio` | `generate` (音楽。`-m sfx` で効果音)、`tts`、`transcribe`、`pitch`、`notes`、`list` | `list` は music、sfx、speech、transcribe、pitch、notes |
| `bonsaiaid` | `serve` (既定)、`install`、`uninstall`、`status`、`stop` | NPU とモデルを持つユーザーごとのプロセス。`bonsaiai-chat` と `ai` のパッケージを使うコマンドとアプリは既定でこれを通る (`BONSAIAI_DAEMON=off` で自分のプロセス) ([bonsaiaid](bonsaiaid.ja.md)) |
| `bonsaiai-dev` | `check`、`convert`、`generate`、`formats`、`irdiff`、`info`、`probe`、`powerbench`、`report`、`mxfp4`、`xclbin`、`models` (`fetch`、`package`、`push`、`hf`)、`sources` (`list`、`ls`、`import`、`update`、`verify`、`gc`) | 開発用。利用者向けではない |

- 音楽と効果音は同じ `bonsaiai-audio generate`: 違うのはモデルだけで、出力 (48 kHz ステレオの WAV、スピーカー) も
  表示も同じ。音楽だけのフラグ (`--codes`、`--continue`、`--notes`) を効果音のモデルに付けるとエラー
- 8 GB 向けの形はモデルのセレクター (models.yaml の項目かメモリのタグ) で選び、環境変数では選ばない。
  [memory-and-power.ja.md](memory-and-power.ja.md) を見る

## 引数の解析 (kong)

- 利用者向けのコマンドはどれも `github.com/alecthomas/kong`: サブコマンドごとに構造体 1 つ、フィールドとタグで書き、
  `--help` が付く。フラグは 2 本の `-` (`--model`/`-m`、`--out`/`-o`)。プロンプトとファイルは位置引数
- 共通の決まり: `-m/--model` は models.yaml の名前、ファミリー、用途 (`chat`、`translate`、`plamo3translate:2b`。
  ファミリーだけなら機械のメモリで選ぶ)、Hugging Face の `OWNER/NAME`、書き出しのディレクトリ (`/` を含むか、
  先頭が `.` か `~` の値) を受ける。`-list` のフラグの代わりに家族ごとの `list`
- 手で変換したプログラムを渡すフラグは、ヘルプの "Converted by hand" の組にまとめる。チェックポイントの
  ディレクトリは `--checkpoint DIR` (`-model` は `-m` とぶつかる)。音声の符号器は `--audio-encoder`
- モデルの定義に任せるだけのフラグはポインタにする (`Tier *string`、`Think *bool`)。与えていないことと空の値を分ける
- `bonsaiai-dev` は `passthrough:"all"` の kong の殻: 道具ごとに `cmd/bonsaiai-dev/NAME` のパッケージで `Main` を持ち、
  標準の `flag` のフラグのまま。`os.Args` と `flag.CommandLine` を差し替えて `Main` を呼ぶ。`probe` は自分を
  子プロセスで起動するので、`probe.ChildArgs` (`["probe"]`) を前に付ける。`bonsaiai-dev models` は普通の kong の
  サブコマンド
- `bonsaiai-dev models` は、手元、取得元、Hugging Face にあるモデルの変換を扱うものをまとめる。コマンドとフラグはどの
  OS でも同じで、裏で動く処理が違う。変換は取得元のフォルダ (`converted/MODEL/TARGET`) と同じく `MODEL/TARGET`
  で呼ぶ: `amd-xdna2`、`coreai` (汎用の `.aimodel` の書き出し)、`coreai-ARCH` (Neural Engine の世代 `h14g` などに
  コンパイルしたもの。`coreai-all` で全部)。`fetch` が持ってきて、`package` が作り、`push` は手で作ったものを取得元に
  置く (`package` が作ったものは自分で置かれる)。その OS にないものが要るコマンドは、そう言って止まる
  (`coreai-ARCH` は macOS の `coreai-build` が要る)。`package MODEL[/TARGET]` は、その対象でのアプリの初回と同じに
  モデルを用意する (`airole.Role.Prepare`。ロールはモデルの種類から `airole.For` で探す)。どの機械でもよい:
  プログラムは変換するか取得元から取り、読み込まない。作ったものは書き戻す取得元に置く。Core AI の書き出しは
  M1 だけがチップで違うので、`coreai` は M1 として、また M2 として作る (`chat.ExportChips`、`chat.ActAsChip`)。
  `--chip` と `--memory` で絞る。`coreai-ARCH` は、そうして作った書き出しを続けてコンパイルする (`coreai.Build`)。
  Core AI の書き出しのフォルダ (`DIR[/coreai-ARCH]`) を渡すと、それだけをコンパイルする。`fetch MODEL[/TARGET]` は同じ準備を、何も変換せずに走らせる
  (`llm.WithFetchOnly`)。手元にも取得元にもないものは、変換を始めるところ (`llm.MayConvert`。`StartExport` /
  `StartProgram`、chat の `convert` と `convertShared`、表、`SharedExportOr`、画像の書き出し) で止めて一覧に出す。
  チェックポイントも持ってくる (取得元のロックが挙げるファイル。なければチャットのモデルの config、トークナイザ、
  重み。`--no-checkpoint` で外す)。あとでそのモデルで変換を試せるように。
- `bonsaiai-dev sources` は取得元の保守: `list` (引数なしでもこれ。根と効いている取得元)、`ls`、`import` (手で作った
  フォルダ)、`update` (ロックを新しいコミットへ)、`verify`、`gc`。書き戻しが切れて残したものは `bonsaiai cache clean`。
  前の名前の `bonsaiai-dev cache` と `bonsaiai-dev prebuild` は 2026-11-22 に外した。`flush` (どのコマンドもキューを待つ)、
  `gc --orphans`、`migrate` (NAS は移し終えた) も
- `bonsaiai-dev report` は利用者に issue のために頼む実機の確認: テスト用モデルを `internal/fixtures` から埋め込む
  (Core ML が実ファイルを要するので一時ディレクトリに書き出す)。同じ計算 (64×512 · 512×512 + バイアス + ReLU) を
  Core ML と OpenVINO IR で持ち、ベンダー間で比べる。NPU の許容誤差は絶対値 0.05 と相対値 2 % (fp16 の誤差は
  0.028)。確認項目は列挙、厳格モードでのコンパイル、結果、コンパイルと実行の時間、fp32 の Core ML モデルが
  厳格モードで拒否されること。Level Zero loader の読み込み結果をいつも載せる (Windows には `/dev/accel` が無く、
  Intel の NPU の有無はこれでしか分からない)。`-power` と `-cache` で、電力計のある機械では 1 回あたりの電力量も
- `aiecc` は `AIE_XCLBINUTIL` の実行ファイルをサブコマンドなしで呼ぶので、実行ファイルの名前が `bonsaiai-xclbin`
  か `xclbinutil` なら `xclbin` として動く (`go build -o ~/.local/bin/bonsaiai-xclbin ./cmd/bonsaiai-dev`)

## bonsaiai-chat

### モデルの選び方

- `-m` にモデルの名前を書く。用途 (チャット、埋め込み) と機能はモデルが持つ。`list` は一覧 (名前、用途、
  リポジトリ、大きさ、メモリ、機能、落としてあるか) を出す
- 機能の名前は Ollama の capabilities に合わせる: `tools`、`thinking`、`vision`、`query` (クエリと文書を別々に
  埋め込む)。機能には `chat.Features` に 1 行の説明が要る (`TestLookup` が説明のないものを拒む)。`/v1/models` は
  `capabilities` として返す
- 一覧にないリポジトリ: `config.json` だけ先に落とし、構造で用途を決める (因果 LM ならチャット、`plamo-bi` なら
  埋め込み、それ以外は拒否)。機能は分からないので出さない

### 端末

- `chat` はプロンプトがあれば 1 回答えて標準出力に書いて終わり、なければ対話。思考は標準エラーに出る
  (生成文を比べるときは標準エラーを見る)
- `search`: 打った行は文書として溜め、`?` で始まる行をクエリとして、文書をコサイン類似度の順に並べる

### OpenAI 互換のサーバー (`serve`)

- `GET /v1/models` (全部、`kind` と `capabilities` 付き)、`POST /v1/chat/completions` (SSE、`reasoning_content`、
  `tool_calls`、`stop`、`stream_options.include_usage`、`usage.prompt_tokens_details.cached_tokens`)、
  `POST /v1/embeddings`
- モデルはいくつでも。要求は `model` で振り分ける。その用途のモデルが 1 つなら名前は問わない (`"model":"lfm3"`)。
  複数で知らない名前は 404。用途の違うモデルの名前 (`/v1/chat/completions` に埋め込みモデル) はエラーにする。
  クライアントが埋め込みモデルを選んでしまったときに気づけるように
- 要求は 1 つずつ処理する
- `reasoning_effort` が none、minimal、low なら考えさせない (題名づくりで 2000 トークン以上考えたことがある)
- 小さなモデルが繰り返し始めたら止める (直近 64 トークンが周期 16 以下の繰り返し)
- `--dump DIR` で要求、プロンプト、生成文を保存する
- 埋め込み: 長すぎる文は `context_length_exceeded`、`truncate: true` (text-embeddings-inference と同じ名前) で
  切り詰め。クエリか文書かは Cohere や Voyage と同じ `input_type` (`query`/`search_query`、
  `document`/`search_document`/`passage`。既定は文書)。`encoding_format: base64` は float32 のリトルエンディアン
- 要求をまたいだプロンプトの再利用 (共通の先頭、並行する会話のための系列の写し) は [llm-engine.ja.md](llm-engine.ja.md)。
  チャットの書式と道具の呼び出しの XML は [llm-models.ja.md](llm-models.ja.md)

### OpenCode (`opencode`)

- モデルを読み込んでから、127.0.0.1 の 30000〜49151 番 (IANA と macOS の一時ポート 49152 番以降を避ける) で、
  乱数で選んだ位置から空きを探して開く。見つけた listener のまま `http.Server.Serve` に渡すので、その間に
  ほかに取られない。`--listen` で指定もできる
- OpenCode の設定 (プロバイダー `bonsaiai`、`@ai-sdk/openai-compatible`、チャットのモデルごとの文脈と出力の上限) を
  キャッシュの `opencode/opencode.json` に書き、`OPENCODE_CONFIG` で渡す。カスタム設定はグローバルより強く、
  プロジェクトの `opencode.json` より弱い
- モデルは `-m bonsaiai/NAME` で選ぶ: TUI では先頭に、`opencode run` では `run` の後ろに。ほかのサブコマンドは `-m` を
  受け付けないので付けず、設定の `model` に任せる。`-m` に複数書くとチャットのモデルを全部設定に並べ、最初を使う
- OpenCode は `os/exec` で起動し、標準入力、出力、エラーにこのプロセスのものをそのまま渡す (パイプを挟まないので、
  raw モード、代替画面、マウス、端末への問い合わせが通る)。サーバーのログはファイルに。SIGINT はこのプロセスで
  捕まえて捨てる (無視にすると子に継がれる。捕まえた信号は exec で既定に戻る)。OpenCode が終わったらサーバーを
  閉じ、その終了コードで終わる
- `opencode run` は標準入力が閉じるまで待つ (端末なら問題ない。スクリプトでは `< /dev/null`)
- 計測 (M3、LFM3-1.7B、キャッシュ 16320): OpenCode の最初の要求 (system と道具 10 個) は 7.4K トークンで、
  読むのに 13 秒。バグ修正は 12 往復、2 回目以降の要求はプロンプトの 99 % を再利用、63〜68 ms/token

### コーディングエージェント (`agent`)

Pi (pi.dev) のループを Go で書き、エンジンを `ai/text` の `Chatter` から HTTP なしで直接呼ぶ (`ai/agent/coding`)。合う Go の
エージェントのライブラリはなかった。system prompt と 4 つの道具の宣言で 629 トークン (OpenCode の最初の要求は
7,516)、脇の要求もない (OpenCode は最初に題名を頼み、小さなモデルでは 1〜2 分かかる)。

- `chat.Engine.Complete` / `Count`: Render → Tokens → Generate → Parse を 1 回に。`agent.Model` (`Complete`、
  `Count`、`Context`) を `*chat.Engine` が満たし、テストは台本どおりに答える偽物
- `Session.Run`: user を足し、`Complete`、道具の呼び出しを順に実行して `tool` の役 (`tool_call_id` 付き) を足す、
  を答えに呼び出しがなくなるまで繰り返す
- 前の往復の思考は会話に残さない (`--keep-thinking` で残す。16K の文脈を食う)
- 文脈の予算: 要求の前に `Count` を取り、`Context() - Reserve` (`-n`、既定 2048) を超えたら古い道具の結果から
  先頭 1 行 + `[... N more lines omitted]` に畳み、最新 4 つは残す。要約の生成はしない。それでも収まらなければ
  `ErrContext`
- system prompt は 3 層: 固定の短い核、出した道具ごとの規則 (道具を減らせば規則も減る)、`AGENTS.md` (4000 文字まで)。
  課題ごとのものは system に置かない: エンジンは共通の先頭を読み直さないので、変わるものは user メッセージに置く。
  `--system FILE` で冒頭を差し替え、`--dump DIR` で同じ課題の prompt を比べる
- 道具の引数は string と integer だけ (LFM3 の `<parameter=k>値</parameter>` は文字列が安全で、配列や入れ子は小さな
  モデルには書けない)

| 道具 | 振る舞い |
|---|---|
| read | 1 回 200 行 / 8 KB で切り、`[Lines a-b of n. Use offset=b+1 to read on.]`。行番号は付けない (edit の `old` に写される) |
| edit | `old` がちょうど 1 回一致したら置き換え。0 回は「read して正確に写せ」、2 回以上は「周りの行を足せ」。`old` の前後の空行は無視 (パーサーが値の前後の改行を 1 つ落とす) |
| write | 親ディレクトリを作る |
| bash | `sh -c`、プロセスグループごと kill、既定 60 秒、末尾 100 行 / 8 KB (Pi の 2000 行 / 50 KB は 16K には大きすぎる) |
| search | `Searcher`: Wikipedia (MediaWiki API、鍵なし、上位の記事の冒頭 2 文)、SearXNG (JSON)、Brave (`BRAVE_API_KEY`)。`--search` で選ぶ |
| fetch | ページを文章にして先頭 `FetchChars` (6000) 文字。`<main>` (なければ `<article>`) の中だけ、script、style、head、nav、footer、aside、コメントを落とす |

ガード (1.7B のモデルには規則よりガードが効く):

| ガード | 理由 |
|---|---|
| 作業ディレクトリのパスを prompt に書かず、ファイルの一覧と「相対パスを使う」にする | 100 文字を超える絶対パスを写し間違えて空回りした |
| ディレクトリの外を指す絶対パスでも、その名前のファイルが中にあればそれとする (`resolve`)。なければファイルの一覧を返す | `/main.go` と書く (OpenCode の癖)。断ると空回りした |
| 外への `cd` は断って残りを示す (`Run: go run main.go`)。出力中の作業ディレクトリの絶対パスは `.` に | `cd /tmp && go run` を繰り返した |
| 同じ呼び出し (名前と引数) が 2 回なら実行せず促す。3 回目で `ErrLoop` | 修正の後に同じ呼び出しを繰り返した |
| 20 語以上で前の答えと 70 % 以上同じ語なら最終の答えとし、呼び出しを捨てる (`similar`) | 要約の末尾に `echo "done"` を付け続けた |
| 繰り返しか往復の上限 (`--steps`、30) では、道具なしの要求を 1 回出して短い要約を書かせる (`close`) | 要約なしで終わっていた |
| read していないファイルへの edit と write を断る (`Session.seen`。write したものは既知、`/reset` で忘れる) | little-coder から |
| 思考は `--think-budget` (1024、0 で無制限) まで: エンジンが `</think>` を差し込む (`chat.ThinkLimit`) | 小さなモデルは数千トークン考える |
| write、edit、bash の前に `Config.Allow`: 端末で y/N、`--yes` で省略、端末も `--yes` もなければ全部断る。`--allow "go *,ls*"`: `*` は任意の文字列、`&&`、`\|\|`、`;`、`\|`、改行で分けた各部分が合うこと | 安全のため |

検索を先に (`--tools search[,fetch] --search-first`):

- 検索するかの判断をモデルに任せない: system prompt の「Answer in the language of the question」を「Always answer in
  Japanese」に変えただけで検索を呼ばなくなり、作者を捏造した (温度 0 で再現する)
- 最初の往復は道具を宣言したまま普通に行う (そこでの検索語は良い)。search を呼ばなければその往復を捨て、課題の
  1 行目で検索した呼び出しと結果を置く。検索語だけを書かせると質問の写しが出た
- 1 課題の検索は `Searches` (4) 回まで: その後は「手元の結果から答えよ」を返し、道具なしの要求を 1 回出す
  (`ErrSearches`)。上限なしでは答えが前にあっても検索を続けた (30 往復、14K トークン)
- search だけを出すと `SearchSystem` (255 トークン、ファイルの規則なし)
- 検索語と URL の整え: Wikipedia には `『』「」` を外す。0 件なら最初の語だけで引き直す。URL は百分率符号化を
  戻して渡す (`readable`。符号化した URL はモデルが写し間違える)。規則は「数値 (measurements) は検索、計算は検索
  しない、名前は書かれたまま (翻訳もローマ字化もしない)」
- 計測 (M3、lut4 の LFM3-1.7B、思考あり、SearXNG、search と fetch): 事実の 5 問中 4 問が正解。Wikipedia 自身の
  検索は当たりが悪い (『こころ』を外した)。残る誤りはモデルの側: 日本語の名前のローマ字化、長いページの関係ない
  節の数値を混ぜる

### systemone

`bonsaiai-chat systemone` は llama.cpp の `POST /v1/systemone` (TypeSafe 互換の System One API) に d1 のモデルで
答える: ファイルか標準入力の要求 1 つ、または `--listen ADDR` で `/v1/systemone` と `/v1/models`
(`output_modalities: ["decisions"]`) を出す。質問ごとに 1 回、プロンプトが入る最短の `--seq` のプログラムで動かす。
`--cpu` は確認用に `milcpu` で動かす。モデルの詳細は [llm-models.ja.md](llm-models.ja.md)。

## 端末の表示 (`internal/console`)

- ログ (ダウンロード、変換、読み込み) は標準エラーに灰色で (`console.Log`)。「all on the NPU」は緑、「NOT all on
  the NPU」は赤、「on the CPU interpreter」は普通。`MLComputePlan` が終わらなかったとき (Neural Engine がコンパイルで
  忙しい) は「placement not known, going on」と出し、NPU から外れた扱いにはしない。利用者が求めたもの (答え、書いた
  ファイル、サーバーのアドレス) は普通の色。`console.Notef` はバーの行に付ける灰色の注記 (例: 「prebuilt programs …
  finished on the device」)
- `console.Bar`: 端末ではラベル、スピナー、バー、割合、経過時間、詳しい情報を 1 秒に 10 回描き直し、`Done` で結果の
  1 行を残す。端末でなければ `Done` の 1 行だけで、エスケープシーケンスも出さない
- 描き直す行は端末の幅 (TIOCGWINSZ) から 1 引いた幅に切り、先にラベルを `…` で縮める。描くあいだは自動の折り返しを
  止める (DECAWM、`\x1b[?7l` … `\x1b[?7h`)。幅の数え違い (東アジアの曖昧な幅の `█` `░`、後から狭めた窓) でも折り返して
  前の行が残ることがない。`Done` の行は切らない
- ダウンロード (`chat.Fetch`、全コマンドで共通): 大きさ、速さ、残り時間。ファイルは 1 つずつ落とす: 並列のバーは
  それぞれの分け前の速さで見積もるので待ち時間を多く見せる。`chat.Expect` が先に 8 本並列の HEAD で大きさを集める:
  Hugging Face の `resolve` は LFS や xet のファイルに CDN へのリダイレクトを返し、その `Content-Length` はリダイレクトの
  本文の長さなので、リダイレクトは追わず `X-Linked-Size` を読む。手元の取得元は stat、S3 とサインインの要るリポジトリは
  合計から外し、`.part` は済んだ分に数える。バーは `1.20 / 4.00 GB, all 3.10 / 9.40 GB 85.3 MB/s, 1m26s left`
  (全体の残り ÷ 今の速さ)。別のファイルを読んで初めて決まるもの (config.json、index.json) と小さいファイルは一覧に
  入れない
- 読み込みのバー (`console.NewLoadBar`): 読み込みは中の見えない 1 回の呼び出し (Core AI、Core ML、Level Zero) で、
  コンパイルの時間は大きさに比例しない (M1、Anima Turbo の 20 本: キャッシュに当たると合計 0.16 秒、相関 0.98。Core AI
  のキャッシュなしで Neural Engine は当たると 72 秒、0.86。Neural Engine の分が押し出されると 320 秒、0.29 で、同じ
  288 MB の DiT の塊が 0.1 秒と 177 秒)。そこで 1 秒以上かかった読み込みを覚え (キャッシュに当たったものは何も教えない)、
  次は大きさが一番近いもの (対数で。同じなら最後のもの) の時間を大きさの比で伸ばして見積もる。1 MiB 未満は 1 MiB と
  する。見積もりが 1 秒未満か、何も覚えていなければスピナーだけ。進みは経過 ÷ 見積もりで、0.8 までは比例、その先は
  `1 - 0.04 / (x - 0.6)` (見積もりちょうどで 90 %、2 倍で 97 %)、99 % を超えない。`(estimated)` と出し、残り時間は
  出さない。分けたモデルは `console.LoadingPart(bytes)` で部分を知らせ、失敗した読み込み (`Bar.Failed`) は覚えない。
  見積もりはプロセスの中だけ。大きさは `llm.ProgramBytes`、`llm.PartBytes`
- bonsaiai-chat の読み込みの行: `[1/2] Qwen3-0.6B 15s, 8 tokens per call, 1 part, all on the NPU`、続けて
  `loaded … in 34s, 2 programs, all on the NPU` と `ready after 35s: converting, loading and compiling 35s`
  (`console.Prepared`)。ファイル名は `BONSAIAI_PROFILE=1` のときだけ
- bonsaiai-image: 書き出しと読み込みは部品ごとの数えとスピナー、ステップのバー (1 ステップの時間と残り時間)、VAE の段の
  バーは `Pipeline.Stage` (`StageFunc`) で受ける。最後の要約は、最初のステップがほかの平均より 5 秒以上長いとき分けて
  出す (コンパイルと最初の呼び出しを含む): `8 steps in 5m25s (3.7 s a step; the first 4m59s, with loading and compiling)`
- 最後の行 (`console.Times`): `took 3m12s: downloading 2m40s, converting, loading and compiling 25s, running 6.2s`。
  `console.Running()` を呼んだコマンドだけが出す (会話とサーバーは出さない)。処理の途中の読み込みは準備に数える
  (`console.Loading()`。`iron.Graph.Use` が読み直す重みの組は `console.Reading()`)。途中で別のモデルを読むコマンドは
  `console.Preparing()` で準備に戻す。ダウンロードの時間は `fetch` の中で測る (重なりは 1 回、サインインの待ちと
  ハッシュの確認を含む)。Core ML が最初の呼び出しで行うコンパイルと `--play` は処理の側。`BONSAIAI_ENERGY` があれば
  段階ごとの電力量が続く ([memory-and-power.ja.md](memory-and-power.ja.md))
- `BONSAIAI_PROFILE=1` (全コマンド、`console.Profile`): ダウンロード 1 件ずつ (時間、バイト数、速さ、`REPO@REV/FILE`)、
  大きさを集めた時間、プログラムの読み込み 1 件ずつ (中の読み込みは字下げ、`runner.Open` の中の同じパスは 1 回)、
  重みの組の読み直しを 1 行、`decode.WriteProfile`。ダウンロードのバーは最初のバイトから始まるので、プロファイルの
  時間は Hub と CDN の待ちを含む (swift-f0: 1.68 秒対 0.2 秒)

## Ctrl+C

- `console.Interrupt()` は `signal.NotifyContext` と同じ形で文脈と `stop` を返す。Ctrl+C で文脈を取り消し、
  `InterruptGrace` (0.1 秒) たっても終わらないか 2 回目の Ctrl+C が来たら、SIGINT の扱いを既定に戻して自分に SIGINT を
  送る (Unix。ほかは終了コード 130)。0.1 秒は、文脈を見ているもの (ダウンロード、子プロセス、書きかけのファイルの
  削除) が自分で戻るための時間
- 理由: 実行中の NPU のコマンドは取り消せない (`run.Close` が終わりを待つ。1024² の VAE の 1 段は最大 2.7 秒)。
  ピン留めバッファの確保 (`mmap` 1.4 GB) と重みの読み込み (`io.ReadFull` 750 MB) は文脈を見ない。どこで SIGKILL
  しても安全だった: カーネルがコンテキストとバッファを 0.001〜0.34 秒で片付け、ドライバのエラーはなく、次の実行は
  同じ PNG を出した
- `console.EndInterrupted(err)`: Ctrl+C の後にエラーで戻っても SIGINT で終わる (終了コード 1 にしない)。シェルの
  ループやスクリプトも止まる。端末では先に改行を 1 つ出す
- `stop()` の後はコマンドが自分で扱う: チャットの 1 ターンの Ctrl+C は答えだけを止める。OpenCode の端末。`serve` と
  `systemone --listen` は Ctrl+C が通常の終わり方: `stop()`、モデルを閉じ、終了コード 0 (閉じるのに約 0.2 秒で、猶予より
  長い)。読み込み中の Ctrl+C はすぐ終わる
- 使うのは `bonsaiai`、`bonsaiai-chat`、`bonsaiai-image`、`bonsaiai-audio`。止めても結果を出す `bonsaiai-dev probe`、`generate`、
  `formats` は自分の扱いのまま
- 計測 (AMD、Ctrl+C から終わるまで): Anima Turbo 1024² 0.11〜0.27 秒、1024² の VAE の段 0.42〜0.87 秒、gpt-oss-20b の
  読み込み中 0.72〜1.47 秒。残るのはカーネルの片付けで、ピン留めしたメモリが多いほど長い (12 GiB で 1.5 秒)

## bonsaiai status

- 出すもの: BonsaiAI の版 (モジュールとコミット)、OS、CPU、メモリ、コマンドが動かす NPU (`chat.HereVendor`、
  `BONSAIAI_BACKEND`)、デバイス、NPU の今 (`npuinfo` を OS ごとに要約: Linux は amdxdna の状態、ファームウェア、タイル、
  クロック、文脈、gpu_metrics の忙しさと電力。macOS は ANE の起きているか、IOReport の電力と読み出し。Windows は
  計算専用のアダプターとプロセスごとの使用率)、NPU を使っているプロセスとその NPU のメモリ、キャッシュと層。
  `--json` で全部
- 読むことで NPU を起こさない: 既定では `bonsaiai.Enumerate` を呼ばない (デバイスを開くと amdxdna は起きる)。NPU の有無と
  種類は OS から決める (Linux はドライバ名 `amdxdna`/`intel_vpu`、macOS は I/O Registry の ANE、Windows は計算専用の
  アダプターの PCI ベンダー、または `BONSAIAI_BACKEND`)。`--wake` でデバイスを開き、寝ている NPU にも問い合わせる。
  `--wake` なしで `runtime_active_time` が増えないことを確かめた。macOS では「device: not opened」を出さない (開いても
  起こすものも分かることも増えない)
- 端末での `-w 2s` は top のように: 代替画面 (`\x1b[?1049h`)。固定の部分 (`writeFixed`: 版、機械、NPU、デバイス、
  キャッシュ) は 1 度だけ書き、書き換える部分 (`writeLive`: 時刻の帯、負荷、NPU の状態、プロセス) をその場で書き直す。
  行は幅で切り (折り返すと下がずれる)、画面の高さを超える行は出さない (スクロールすると固定の部分が消える)。大きさが
  変わったら全部を書き直し、Ctrl-C で元の画面に戻す。間隔は読む時間を含む。端末でない、`--json`、Windows (VT を有効に
  していない) は繰り返して書く。Enumerate とキャッシュの走査は 1 回だけ
- 負荷 (`internal/sysload`、root なし):

| | CPU / メモリ | GPU | NPU |
|---|---|---|---|
| Linux | `/proc/stat` の差分。MemTotal − MemAvailable | amdgpu の sysfs (`gpu_busy_percent`、`mem_info_vram_*`、`mem_info_gtt_*`) | gpu_metrics の IPU の忙しさと電力。プロセスの写像の合計 |
| macOS | `host_statistics` の tick。`host_statistics64` の「使用済みメモリ」(アプリ + 固定 + 圧縮、アクティビティモニタと同じ) | IOAccelerator の PerformanceStatistics (`Device Utilization %`、`In use system memory`) | IOReport の電力。ANE の driver の wired メモリ (`mem.NPUWired`)。忙しさ (下) |
| Windows | まだ読まない | まだ読まない | 性能カウンター |

- プロセスの表 (`sysload.ProcSampler`、% は差分から): PID、名前 (16 文字)、CPU%、メモリ、GPU%、GPU のメモリ、NPU%、
  NPU のメモリ、コマンドライン (`-w` と 1 回の表示だけ。`issue` には載せない)。`-` は分からないもの。Linux:
  `/proc/PID/stat`、`statm`、DRM の fdinfo (`drm-engine-*`、`drm-resident-*`、古い `drm-memory-*`、`drm-client-id` で
  重ねない) で amdgpu を GPU に、amdxdna と intel_vpu を NPU に。macOS: `proc_pid_rusage` (時間は `mach_timebase_info`
  で。メモリは footprint + wired: mlock したファイルのページは footprint に入らない。JSON に `wired_bytes`。ほかの
  ユーザーのプロセスは root が要る) と、IOAccelerator の user client の `accumulatedGPUTime` (プロセスごとの GPU の
  メモリは出ない)
- macOS の NPU の読み方 (M3、macOS 27.2):
  - IOReport のエネルギーは root でないと全部 0 (ANE も CPU も)。CPU のエネルギーも 0 なら 0.00 W ではなく「power not
    readable without root (sudo powermetrics …)」(`DarwinIOReport.ANEPowerKnown`)
  - NPU% は `PMP`/`AF BW`/`ANE0 RD` (ANE の読み出しの帯域の分布、1〜32 GB/s の 32 区分、M3 で約 0.23 ms に 1 回) が
    いちばん下の区分の外にいた時間の割合 (`ANEBusyPercent`。生成中 39〜51 %)。ANE にはプロセスごとの時間がないので、
    プロセスの NPU のメモリの 9 割以上を持つプロセスにまとめて付ける (`npuHolder`。日本語入力や Siri も数 MB 持つ)
  - 割り込みは `ane N` (M3 は `ane 0` でなく `ane 4`)。`FirmwareLoaded` はあるときだけ出す (`FirmwareKnown`。M3 の
    `H11ANE` にはない)。`ANE`/`IOP State` は読み込まれている間ずっと `Running` で、負荷には使えない。`AMC Stats` の
    `ANE RD` は M3 にない

## bonsaiai issue

- 実行の記録: `bonsaiai-chat`、`bonsaiai-image`、`bonsaiai-audio` は引数 (`console.RecordRun`)、`console.Log` の行 (進み具合の
  行は最後の状態、色なし)、終わり方 (`console.EndInterrupted`) を、キャッシュの下の `logs/<prog>-<time>-<pid>.log` に
  書く。コマンドごとに最後の 10 回 (`console.RunsKept`)。`BONSAIAI_NO_RUN_LOG` で止まる。端末かどうかの判定は変えない
  (`teeRun` も `Terminal()` を持つ)
- `bonsaiai issue` (`--list`、`--run N`、`--lines` (80)、`--print`、`--title`、`-y`、`--wake`): 直近 1 週間で失敗した最後の
  実行 (なければ最後の実行) の、コマンド、エラー、ログの末尾、status を Markdown にする
- 伏せるもの: ホーム (`~`)、ユーザー名と本名、ホスト名、`BONSAIAI_MODEL_SOURCE*` のホストとフォルダ、公開のもの
  (Hugging Face、GitHub) 以外の URL のホスト、URL のパスワード、トークン (`token=`、`hf_`、`AKIA`、Bearer)、メール
  アドレス。空白を含む 40 文字を超える引数 (プロンプト) は文字数に。プロセスのコマンドラインは載せない
- 送り方: `gh` は使わない。`https://github.com/shibukawa/bonsaiai/issues/new?title=...&body=...` を `open` (macOS)、
  `xdg-open` (Linux)、`rundll32 url.dll,FileProtocolHandler` (Windows。`start` では cmd が `&` を読む) で開く。画面の
  ボタンを押すまで送られない。7500 バイトを超えるならログを 2/3 ずつ削り、それでも長ければ `logs/issue-TIME.md` を
  貼るよう案内する (下書きはいつもそこに残す)

## bonsaiai cache、bonsaiai memory

- `bonsaiai cache` (情報、`ls`、`prune`、`clear`、`clean`): [distribution.ja.md](distribution.ja.md) を見る
- `bonsaiai memory [MODEL [GB]] [--forget]`: モデルをどのメモリの区分で動かすか。`run.json` に残す。
  [memory-and-power.ja.md](memory-and-power.ja.md) を見る

## 採らなかったもの

- 古い単独の実行ファイル (`bonsaiai-depth`、`bonsaiai-music`、`bonsaiai-sfx`、`bonsaiai-tts`、`bonsaiai-transcribe`、`bonsaiai-pitch`、
  `bonsaiai-notes`、`bonsaiai-systemone`、`bonsaiai-pose`、`bonsaiai-agent`、`bonsaiai-check`、`bonsaiai-convert`、`bonsaiaiinfo`、`npuprobe`、
  `powerbench`、`report` など): 家族にまとめ、ラッパーは残さない
- チャットのモデルを用途で選ぶ (`-use chat|embed`): 同じ用途のモデルは増えるので名前で選ぶ
- 並列のダウンロード: 速さの配分が変わるとファイルごとの残り時間が狂う (4 GB と 1 GB を 100 MB/s で並列に落とすと
  80 秒と 20 秒と出るが、どちらも 50 秒で終わり、1 つずつと同じ)
- 読み込みのバーの残り時間: 見積もりが当てにならない
- 読み込みの見積もりをディスクに残す
- issue に `gh` を使う: 利用者がブラウザで確かめて送る
- エージェント: 既存の Go のエージェントのライブラリ (kfet/agent は Anthropic の型でプロバイダーなし、pigo と og は
  internal に閉じた CLI)。Pi の TypeScript の拡張の仕組み (`agent.Config` の関数、AGENTS.md、`--system`、`--tools` で
  足りる)。サブエージェント、plan mode、ターンごとの skill の注入 (1.7B には使いこなせない)。要約による compaction
  (畳むことで代える)。DuckDuckGo の html と lite (この機械からは 202 の anomaly)。search-first でモデルに検索語だけを
  書かせる (質問の写しになった)
- OpenCode の親で SIGINT を無視する: 無視は OpenCode に継がれる

## 未解決

- エージェント: 崩れた道具の呼び出しの修復 (output-parser)。1 度 `<parameter=command` と崩れ、bash の「command is
  required」で次の往復に直った。1 つの答えに複数の tool_calls (順に実行するが、出たのを見ていない)。Gemma 4 E2B/E4B を
  エージェントに。`--no-think` との比較
- 検索: `fetch` に `query` を付けて該当の段落を切り出す (長いページは前半だけが届き、関係ない数値を拾う)。ja で 0 件か
  外れのとき en も引く。モデルの検索語に日本語が無く課題にあるときは課題の文で引く
- 埋め込みのサーバー: 複数の文をまとめて流す (今は 1 文ずつ)
- `bonsaiai status`: macOS と Windows の要約は初めクロスビルドだけ。Windows の CPU と GPU の負荷は読まない
- `bonsaiai issue`: GitHub が URL で受け付ける長さをブラウザで確かめていない
- `bonsaiai-audio` の Ctrl+C は実機で確かめていない
- 見積もりのバーの、言語モデルの parts.json の鎖、Intel、AMD での確認 (ビルドだけ)
- 小さなモデルでは、OpenCode の題名の要求で最初の答えが 1〜2 分遅れる
