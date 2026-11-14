# アプリに組み込む API: モデルの準備と実行を分ける (計画)

作成: 2026-11-11。状態: `ai` パッケージの v0 を実装 (2026-11-14、11 節)。`ai/server` から先は未実装。

hwnpu は NPU の機能を提供し、それをアプリに組み込む仕組みを提供する。`hwnpu-chat`、`hwnpu-image`、`hwnpu-audio` は、個々の機能で何ができるかを見せるショーケースとして残す。組み込みの例は `samples/` に置き、最初は同時通訳 (逐次通訳) にする。その後、多レイヤー分割と擬似立体 (顔の位置と深度による視差) を作る。

## 1. 今の状態と足りないもの

先行の設計案: [design.md](design.md) の 16.94「モデルを部品として組み合わせる」。そこでは、ライブラリ、常駐のプロセス、CLI の 3 層に分ける。また、Device の常駐の管理 (メモリと ANE のキャッシュの予算、追い出し、投入の順番) と、受け渡すデータの外枠 (一括と流れ、時刻、制御、出どころ) を決める。この計画はその 1 層目と 2 層目を具体にしたもの。16.94 は「2 つ目の組み合わせが動くまで公開しない (API を早く固めすぎない)」としている。そこで `ai` は、最初は v0 として出し、同時通訳と擬似立体の 2 つのサンプルが動いてから形を固める

- 公開されている Go の API はグラフを動かす層だけで、`Device`、`Graph`、`Buffer`、`Request` がある (ルートの `hwnpu` パッケージ)
- 機能はどれも `internal/` にあり、外のアプリからは import できない。`transcribe.Transcriber`、`speech.Synth`、`chat.Engine`、`face.Detector`、`depth.Estimator` などがそれにあたる
- モデルの準備は各コマンドの中にあり、`Prepare` が 1 つの関数で取得と変換を両方行う (`cmd/hwnpu-audio/transcribe.go`、`internal/transcribe/prepare.go`)。models.yaml からモデルを選ぶ処理もコマンドの中にある
- HTTP のサーバーは `/v1/models`、`/v1/chat/completions`、`/v1/embeddings` の 3 つだけ (`internal/chat/server.go`)
- 音声やカメラの映像を流し込むストリーミングの口がない

## 2. 方針: 登録 → 計画 → 準備 → 実行

**実行の API はネットワークに触らず、変換もしない。** 準備済みの置き場からだけ読み込む。

```go
// 登録: アプリが使うモデルを宣言する (組み込みの models.yaml と、アプリ独自の項目)
reg := ai.NewRegistry()
reg.Require("transcribe", "translate", "speech", "vad")

// 計画: 何が足りないか、何を何バイト落とすか、変換が要るか、ライセンスを調べる。まだ何も落とさない
plan, err := reg.Plan(ctx, ai.Store{Dir: dataDir, Bundles: []string{resources}}, ai.HuggingFace())
plan.DownloadBytes()
plan.Licenses() // 同意の画面や、表示が要るもの (CC-BY など)

// 準備: 取得 → sha256 の照合 → 変換 → 試しに 1 回読み込む (Core ML のコンパイル)。進み具合を返す
err = plan.Run(ctx, func(p ai.Progress) { ... })

// 実行: 準備されていなければ ErrNotPrepared。ネットワークにも変換にも触らない
rt, err := ai.Open(ctx, ai.Store{Dir: dataDir, Bundles: []string{resources}})
stt, err := rt.Transcriber("transcribe")
```

- **用途の名前は計画のときに決める**: `translate` のように、マシンのメモリで選ぶモデルが変わる名前がある (`internal/chat/tags.go`)。計画の段階で選んだモデルを置き場の目録 (`store.json`) に書き、実行時はそれを読む。準備したモデルと実行時に選ぶモデルが食い違わないようにするため
- **試しの読み込み**: Apple の ANE のコンパイル結果は、その機械と OS のものなので配れない (prebuilt.md 1 節)。Gemma 4 のように初回が 100 秒かかるものもある。準備の最後に 1 回読み込んでおき、初めて実行したときに待たせない
- **CLI は今の動きを残す**: ショーケースなので、計画と準備を自動で行ってから動かす。中身はこの API を呼ぶだけにする

## 3. 取得元

| 取得元 | 中身 | 変換 |
|---|---|---|
| `ai.HuggingFace()` | 今の実装 (`chat.Fetch`)。途中から再開でき、同意の要るリポジトリはデバイスフローでサインインする (design.md 16.153) | する |
| `ai.Mirror(baseURL)` | HF と同じファイルの並びを、HTTPS の基準の URL の下に置く。公開の S3、CloudFront、署名付き URL はこれで足りる | する |
| `ai.Prebuilt(...)` | 変換済みの配布 (prebuilt.md)。HF の `hwnpu/NAME`、またはミラー | しない |
| `ai.Source` (インタフェース) | `Fetch(ctx, file) (io.ReadCloser, int64, error)`。非公開の S3 などは、アプリが自分の SDK で実装する。hwnpu は AWS の SDK に依存しない | 中身による |
| バンドル (`Store.Bundles`) | アプリに同梱した、読み取り専用のディレクトリ (4 節) | **しない** |

- どの取得元でも、models.yaml の `sha256` で照合する。今は一部のファイルにしか付いていない (Qwen3-1.7B、Sana、Anima など) ので、全ファイルに広げる。変換済みの配布は、目録 (`manifest.json`) の sha256 で照合する
- ライセンス: ミラーや同梱が許されるかどうかは、モデルごとに異なる。Gemma の規約、同意の要るリポジトリ、CC-BY の表示 (Magenta RealTime 2) などがある。`plan.Licenses()` は models.yaml の項目から、再配布が許されるかと、表示する文を返す

## 4. バンドル: NPU の種類ごとに作り、実行時には変換しない

**アプリを配布するときは、NPU の種類ごとにビルドする。** 同梱するのは変換済みのプログラムで、実行時の変換はない。

### 4.1 対象の名前

prebuilt.md 3 節と同じ名前を使う。

| 対象 | 同梱するもの | 1 つのバンドルで動く範囲 |
|---|---|---|
| `apple` | `.mlpackage` か `.mlmodelc` (Core ML でコンパイル済みのもの。`internal/backend/coreml` はそのまま読める) | M1〜M4 で同じファイル。M1 は一度に読み込むプログラムを少なくするだけ (`internal/chat/model.go` の `m1HoldBytes`)。macOS の最低の版を目録に書く |
| `amd-npu2` | IRON の書き出し (ELF、xclbin、バンク、manifest) | XDNA 2 (Strix 以降) |
| `amd-npu1` | `--per-step` の書き出し | XDNA (Phoenix、Hawk Point)。実機で未確認 |
| `intel` | OpenVINO IR (`.xml`、`.bin`) | Core Ultra。NPU の blob は、ドライバが読み込むときにコンパイルする |

### 4.2 中身と目録

```
bundle/
├── hwnpu-bundle.json    目録: format_version、対象 (apple など)、作った hwnpu のコミット、
│                        macOS の最低の版、各モデルの名前、メモリの段、ファイルと sha256、ライセンス
├── whisper-large-v3-turbo/   トークナイザー、埋め込みの表、プログラム (prebuilt.md 4 節の組と同じ並び)
├── plamo-3-translate-0.8b/
└── LFM3-Audio-2B-TTS/
```

- **メモリの段**: 用途の名前はメモリで選ぶモデルが変わる。バンドルを作るときに、入れる段を決める (例: 8 GB の段と 16 GB の段)。実行時は、**入っている段のうち**マシンのメモリに合うものを選ぶ。合う段がなければ `ErrBundleTarget` を返し、変換にも取得にも逃げない
- **実行時の照合**: `ai.Open` は、目録の対象とその機械の NPU (`hwnpu.Target()`) を比べる。違えば、何が違うかを書いた `ErrBundleTarget` を返す。format_version が読めない範囲のときも同じ
- 中身は prebuilt.md の組とファイルの並びを同じにする。変換済みの配布を、そのままバンドルにもミラーにも使えるようにするため

### 4.3 作り方: `hwnpu-dev bundle`

```
hwnpu-dev bundle -target apple -use transcribe,translate,speech,vad -memory 8,16 -o dist/bundle
```

- 計画と準備 (2 節) を、その対象とメモリの段で行い、置き場から目録とファイルを集める。変換済みの配布があればそれを使い、なければ手元で変換する
- **`apple` と `intel` は、どの OS の機械でも作れる**: 変換は Go だけで、Python も Mac も要らない (README の `hwnpu-dev convert`)。ただし `.mlmodelc` にするには Mac の Core ML のコンパイルが要る。Mac 以外で作るときは `.mlpackage` を入れ、初回に端末でコンパイルする
- `amd-*` は IRON の環境が要る。aix1 で作るか、変換済みの配布を使う
- バンドルに入れるモデルのライセンスを確かめ、再配布できないものは失敗にする (`-allow-license` で明示したときだけ通す)

## 5. ビルドタグ

**タグを付けないビルドは、今と同じ動きにする**: その OS のバックエンドを全部入れ、変換も入れる。開発と CLI のショーケース用。

| タグ | 意味 |
|---|---|
| `hwnpu_apple` | Core ML のバックエンドだけを入れる。darwin のみ |
| `hwnpu_amd` | AMD のバックエンドだけを入れる (Linux の `amd`、Windows の `amdxrt`) |
| `hwnpu_intel` | Level Zero のバックエンドだけを入れる |
| `hwnpu_noconvert` | 変換を入れない。チェックポイントの変換、ONNX の書き換え、MIL や IR の書き出し、IRON の書き出しの起動が入らない。準備はバンドルと変換済みの配布だけになり、変換の要る計画は `ErrNoConvert` で失敗する |

- 対象のタグは複数付けてよい (`hwnpu_amd,hwnpu_intel` で Windows の両方)。1 つも付けなければ全部入れる
- OpenVINO の CPU のバックエンドは、Intel の経路を NPU なしで確かめるためのもの (`HWNPU_BACKEND=openvino`)。タグを付けないビルドにだけ入れる
- 間違った組み合わせはコンパイルで止める。たとえば `hwnpu_apple` を darwin 以外で付けたら、わざとコンパイルが失敗するファイルを置く (`backend_apple_other.go`)
- `hwnpu.Target()` (その機械の対象の名前) と `hwnpu.Built()` (入っているバックエンドと、変換が入っているか) を公開し、アプリが自分のビルドを確かめられるようにする
- ネットワークを入れないタグ (`hwnpu_nofetch`) は、今は作らない。取得は `plan.Run` を呼んだときだけ起きるので、バンドルだけのアプリは呼ばなければよい

配布するときのビルドの例:

| 配布物 | GOOS / GOARCH | タグ | 同梱 |
|---|---|---|---|
| Mac 版 | darwin / arm64 | `hwnpu_apple,hwnpu_noconvert` | `apple` のバンドル |
| Linux の Ryzen AI 版 | linux / amd64 | `hwnpu_amd,hwnpu_noconvert` | `amd-npu2` (と `amd-npu1`) のバンドル |
| Windows の Ryzen AI 版 | windows / amd64 | `hwnpu_amd,hwnpu_noconvert` | `amd-npu2` のバンドル |
| Core Ultra 版 | linux、windows / amd64 | `hwnpu_intel,hwnpu_noconvert` | `intel` のバンドル |

`hwnpu_noconvert` を入れるには、今の `Prepare` を「取得」と「変換」に分け、変換のファイルにタグを付ける必要がある。`chat.Prepare`、`transcribe.Prepare`、`speech.Prepare`、`internal/onnx` を使うものが対象になる。手間が大きいので、7 節の 4 番目に回す。

## 6. 公開のパッケージ (`github.com/shibukawa/hwnpu/ai`)

`internal` の機能を薄く包む。準備 (2 節) と実行の型を同じパッケージに置く。

```go
rt, err := ai.Open(ctx, store, ai.WithLog(w)) // NPU を 1 回だけ開き、モデルどうしで共有する
defer rt.Close()

stt, _ := rt.Transcriber("transcribe")
text, _ := stt.Transcribe(ctx, pcm16k, ai.TranscribeOptions{Language: "ja"})

tr, _ := rt.Translator("translate")
en, _ := tr.Translate(ctx, text, ai.TranslateOptions{To: "en", History: hist}) // 前の発話を文脈にする

tts, _ := rt.Speaker("speech")
pcm, rate, _ := tts.Speak(ctx, en, ai.SpeakOptions{Language: "en", Voice: "female"})

vad, _ := rt.VAD("vad")
```

- 各機能は `OnNPU()` で、全部が NPU に載ったかどうかを返す

### 6.1 データの流し方: 一括と、塊の流れ

どの機能も 2 つの形を持つ (16.94 の「一括と流れ」)。

- **一括**: 値を 1 つ渡し、結果を 1 つ受け取る (上の `Transcribe`、`Translate`、`Speak`)。ファイルの処理や HTTP の 1 回の要求に使う
- **流れ**: ある程度の大きさの塊を順に渡し、結果も塊の列で受け取る (RxJS の Observable に近い)。マイク、カメラ、WebSocket、機能どうしのつなぎに使う

```go
// 塊: 中身、共通の時計の上の位置と長さ、区切り (発話や文の終わり)
type Chunk[T any] struct {
	Data  T
	Start time.Duration // 流れの始まりから (音声は標本の数から計算し、ずれない)
	Len   time.Duration
	Final bool          // 区切りの最後の塊 (発話の終わり、文の終わり)
}

// 流れは Go の iter.Seq2。range で読め、break と ctx で止まる
type Stream[T any] = iter.Seq2[Chunk[T], error]

voice := mic.Stream(ctx)                          // Stream[[]float32]: 20 ms ごとでも 100 ms ごとでもよい
speech := vad.Detect(ctx, voice)                  // 発話のところだけ。発話の終わりに Final
texts := stt.TranscribeStream(ctx, speech, opts)  // Stream[ai.Text]: 仮の字幕と確定 (Final) の列
en := tr.TranslateStream(ctx, ai.Finals(texts), topts) // 確定した文だけを訳し、訳を文ごとの塊で
audio := tts.SpeakStream(ctx, en, sopts)          // Stream[[]float32]: 24 kHz の音の塊
for c, err := range audio { play(c.Data) }
```

- **塊の大きさは、受け取る側が決め直す**: NPU のプログラムは形が固定なので、それぞれの機能が自分に合う大きさに組み直す。入る側の塊の大きさは自由にする
  - VAD: 0.5 秒 (Silero を 16 ステップ展開したプログラム、8 節)
  - 文字起こし: 読み直しの間隔 1〜2 秒 (app-api-tasks.md の B5)
  - 読み上げの出力: コーデックの復号のプログラムが 1 回に読むフレームの数 (`speech.DecodeCodec` の `fr`)
  - 読み上げの入力: 文や句の区切り (app-api-tasks.md の B6)
- **一括は、塊が 1 つの流れ**: `Final` の付いた塊を 1 つ流せば一括と同じ。ただし、30 秒より長い音声の文字起こし (タイムスタンプで窓を進める) のように、一括で別のやり方をとるものは一括の実装を残す
- **段を重ねて動かす**: `ai.Async(stream, n)` は前の段を別の goroutine で回し、n 個の塊までためる。読み上げが 1 文目の音を作る間に、翻訳が 2 文目を訳せる。NPU の仕事は Device が順に出す (16.94 の投入の順番。音声を先に)
- **塊をまとめる道具** (RxJS の bufferTime、bufferCount に当たる): `ai.Rechunk` (標本の数や時間で切り直す)、`ai.Sentences` (テキストを文で区切る)、`ai.Finals` (確定したものだけ)、`ai.Tee` (同じ流れを 2 つに分ける。字幕と翻訳など)。数は少なく保ち、必要になったものから足す
- **制御はデータと別にする**: 取り消しは ctx で行う。割り込み (相手が話し始めたら読み上げを止める) は、VAD の流れを読み上げの `ai.Interrupt(vadEvents)` に渡す形にする
- **時計は標本の数**: 音声の流れの時刻は、壁時計ではなく流れの始まりからの標本の数で数える。マイクの遅れやネットワークの揺れでずれないようにするため。16.94 で決めていなかった「時計の基準」はこれにする
- **サーバーとの対応**: WebSocket の 1 つのメッセージが 1 つの塊になる。HTTP の一括の要求は、塊が 1 つの流れ。テキストの流れは SSE でも返せる (`/v1/chat/completions` の `stream: true` と同じ形)
- **同時に置くモデルのメモリ**: 同時通訳では 3 つのモデルを同時に置く。models.yaml の `memory` では、whisper-large-v3-turbo が 1.9 GB、PLaMo 3 Translate 0.8B が 2.4 GB、LFM3-Audio-2B-TTS が 2.4 GB で、合わせて約 6.7 GB になる。8 GB では厳しいので、計画の段階で合計を出し、足りなければ必要なときに読み込み直す動き (`ai.WithReload`) を選べるようにする。M1 では ANE が一度に持てる量 (2.8 GiB) も効く
- 画像の機能 (`FaceDetector`、`DepthEstimator` など) は、擬似立体のサンプルのときに足す

## 7. ローカルサーバー (`github.com/shibukawa/hwnpu/ai/server`)

`http.Handler` として作り、アプリが自分のサーバーに組み込める。`hwnpu-chat serve` もこれを使う。**既存のサービスの API との互換は、サービスごとに下のパッケージに分ける**。共通の部分 (仕事の置き場、認証、読み書きしてよいディレクトリ、ストリーム) は `ai/server` に置き、互換のパッケージはそれぞれの形の要求を `ai.Runtime` の呼び出しに直す薄い層にする。

```
ai/server               共通: 仕事の置き場と待ち行列、トークン、許すディレクトリ、/v1/capabilities、
                        独自のストリーム (/v1/audio/stream)、下の互換の口をまとめる Handler
ai/server/openai        OpenAI 互換
ai/server/googlecloud   Google Cloud の REST 互換
ai/server/a2a           A2A (7.1 節)
ai/server/mcp           MCP (7.1 節。hwnpu-mcp はこれを stdio で動かす)
```

```go
h := server.New(rt, server.WithToken(tok), server.WithDirs(out),
	openai.Mount(),      // /v1/chat/completions、/v1/audio/...
	googlecloud.Mount(), // /v1/speech:recognize、/language/translate/v2 ...
	a2a.Mount())         // /.well-known/agent-card.json、JSON-RPC
http.ListenAndServe("127.0.0.1:8080", h)
```

- **`openai`** (最初): 今の `/v1/models`、`/v1/chat/completions` (翻訳もここ、`chat_template_kwargs`)、`/v1/embeddings` を `internal/chat/server.go` から移す。足すもの: `POST /v1/audio/transcriptions`、`/v1/audio/translations` (英語への書き起こし)、`/v1/audio/speech`。あとで `/v1/images/generations`。OpenAI の SDK は `base_url` を変えるだけで向けられるので、効果が一番大きい。モデルの名前は用途に読み替える (`whisper-1` → `transcribe`、`tts-1` → `speech`、それ以外は models.yaml の名前)
- **`googlecloud`** (次): REST の形で、Speech-to-Text v1 `POST /v1/speech:recognize`、Text-to-Speech v1 `POST /v1/text:synthesize`、Translation v2 `POST /language/translate/v2`、Vision `POST /v1/images:annotate` (顔、ラベル、OCR、物体)。OpenAI に文の翻訳と画像の解析の API がないところを埋める。注意: Google の公式のクライアントは gRPC が既定のものが多い。REST で投げるコード、または REST の転送とエンドポイントの差し替えができるクライアント (認証はダミー) からだけ使える。gRPC は依存が増えるので作らない
- 候補 (今は作らない): DeepL 互換 (`/v2/translate`。翻訳を使うアプリが多い)、Ollama 互換 (`/api/chat`。手元の道具が多く話す)。AWS は SigV4 の署名と `X-Amz-Target` の形で手間が大きいので外す
- 独自のもの: `GET /v1/audio/stream` (WebSocket)。16 kHz の PCM を流し込むと、VAD の区切り、文字起こし、指定すれば翻訳と読み上げの音声が返る。どのサービスにも同じものがないので `ai/server` に置く (OpenAI の Realtime API は会話のモデルを前提にしていて形が合わない)
- WebSocket は標準ライブラリにない。依存を増やさないため、`internal/ws` に RFC 6455 の最小限のもの (テキストとバイナリのフレーム、ping、close) を書く
- 互換の範囲は、各パッケージの文書に「対応する項目」と「受け付けて無視する項目」と「断る項目」で書く。知らない項目は黙って無視せず、400 で名前を返す (結果が違うのに気づけないため)

### 7.1 エージェント向けの口: 重い仕事を下請けする

**考え方**: 計画を立てる頭脳は、クラウドの高性能なモデルか、ツール呼び出しのできる手元のモデル (`hwnpu-chat agent`) が受け持つ。hwnpu はその下請けとして、NPU で安く、手元で、電力を抑えて済む仕事を引き受ける。たとえば次のようなもの。

- 会議の録音の書き起こし、フォルダの画像の OCR やラベル付け、写真の埋め込みと検索
- 画像の生成、レイヤー分割、読み上げ、長い文書の翻訳
- 小さなモデルでの分類や抜き出し (d1 の型付きの質問、要約の map の段)

利点は 3 つある。

- **元のデータを手元に残せる**: 頭脳に渡すのは結果 (文字、ラベル、ファイルの場所) だけでよい
- **頭脳の文脈を使わない**: 大きな出力 (画像、音声、PSD) はファイルに書き、場所と短い要約だけを返す
- **計画に使える見積もりを返せる**: 1 件あたりの時間と電力量 (`PowerSample` で測れる) を返す

**仕事の API (HTTP、`ai/server` の中)**: MCP のツールの呼び出しは、要求が返るまで待つ作りが多く、数分かかる仕事には向かない。そこで核は非同期の仕事の API にする。

| | 中身 |
|---|---|
| `POST /v1/jobs` | 種類 (`transcribe`、`ocr`、`label`、`image.generate`、`layers` など)、入力 (ファイル、フォルダと glob、URL)、設定、優先度 (`interactive`、`normal`、`background`)、電源の方針。返すのは `202` と仕事の ID |
| `GET /v1/jobs/{id}` | 状態 (待ち、準備中 (モデルの取得や変換)、実行中、完了、失敗、取り消し)、進み具合 (件数、段階)、残りの見積もり、ここまでの電力量 |
| `GET /v1/jobs/{id}/events` | SSE。進み具合と途中の結果 (6.1 節の塊)。多数の件を処理する仕事は、1 件ずつの結果を JSONL で |
| `GET /v1/jobs/{id}/result` | 小さな JSON。大きな出力は出力のディレクトリのファイルの場所 |
| `DELETE /v1/jobs/{id}` | 取り消す |
| `GET /v1/capabilities` | この機械でできること、準備済みのモデル、NPU の対象、仕事の種類ごとの 1 件の時間と電力量の目安。頭脳が手元で頼むかクラウドでやるかを決めるのに使う |
| `GET /v1/tools` | 仕事の種類ごとの入力の JSON Schema。どのエージェントの枠組みにも、関数として登録できる |

- 仕事は置き場のディレクトリに書いておき、サーバーを起動し直しても状態を失わない
- **優先度は Device の投入の順番に載せる** (16.94): 音声の対話 (`interactive`) が来たら、背景の仕事は塊の切れ目で待つ。`background` の仕事は低い電源の方針で回す
- サーバーを NPU を持つ唯一のプロセスにする (app-api-tasks.md の B4)。サーバーが動いているときは、CLI もサーバーに頼む
- 安全: 既定では 127.0.0.1 だけで待ち受ける。外から受けるときはトークンを求める。エージェントは任意のファイルの場所を渡せるので、読み書きしてよいディレクトリを設定で限る。URL の取得も、許した先だけにする

**エージェント向けの主な口は A2A にする**: 頭脳から下請けへ仕事を渡す、という形は、MCP (道具を呼ぶ) より A2A (Agent2Agent、Linux Foundation、1.0 は 2026-03) の作りに近い。A2A は、仕事 (task) の状態の移り変わり、途中の知らせの流れ、終わったときの知らせ (push notification)、成果物 (artifact) を、はじめから持つ。上の仕事の API との対応は次のとおり。

| A2A | hwnpu |
|---|---|
| Agent Card (`/.well-known/agent-card.json`) | `/v1/capabilities` と `/v1/tools`: 技能 (skill) ごとに、入出力の種類、例、時間と電力量の目安 |
| `message/send` (待たない設定で) | `POST /v1/jobs`。技能の ID と、JSON Schema に沿った引数を DataPart で受ける |
| `message/stream`、仕事の購読 | `GET /v1/jobs/{id}/events` (SSE)。状態の変化と、成果物の途中の塊 |
| `tasks/get`、`tasks/cancel` | `GET`、`DELETE /v1/jobs/{id}` |
| push notification の設定 | 終わったときに、指定の URL に知らせる (webhook) |
| 成果物 (TextPart、FilePart、DataPart) | 書き起こしの文字、出力ファイルの URI、構造のある結果 (ラベル、枠、JSON) |
| 状態 `input-required`、`auth-required` | 同意の要るモデル (gated) のサインイン、ライセンスへの同意、足りない引数 |
| `contextId` | 続きのある仕事の文脈 (翻訳の前の発話、会話) |

- 送り方は JSON-RPC over HTTP と SSE にする。gRPC は依存が増えるので作らない。A2A の Go の SDK にも頼らず、JSON-RPC を自分で書く。相互に通じるかは、公式の Python の SDK のクライアントで確かめる
- A2A は、相手が自然言語を読むエージェントであることを前提にしている。しかし hwnpu の技能は、ほとんどが決まった入出力の関数である。そこで技能ごとの引数を DataPart で受け、自然言語だけのメッセージは断る (`rejected`)。手元の小さなモデルで自然言語を技能に振り分けるのは後で考える
- 版: 1.0 は 0.3 から壊れる変更が多く、その後の小さな版の番号も情報源によって違う。1.0 に固定し、`A2A-Version` のヘッダーで照合する
- 呼べる頭脳 (A2A のクライアント): Google ADK、Microsoft Agent Framework (プレビュー)、LangGraph など。Claude Code のように MCP しか話さないものもあるので、次の MCP の口も残す

**MCP の口 (`hwnpu-mcp`、補助)**: 仕事の API を呼ぶだけの薄いサーバー (stdio)。MCP しか話さない頭脳 (Claude Code、エディタ) のため。

- ツールは `capabilities`、`submit_job`、`get_job`、`wait_job`、`cancel_job`。`wait_job` は最大 N 秒 (既定 50 秒) だけ待ち、終わっていなければ状態を返す。エージェントはもう一度呼べばよく、MCP の呼び出しの時間切れに当たらない
- 結果のファイルは MCP の resource (ファイルの URI) として返す
- MCP の Tasks (2025-11-25 の仕様で実験的に入った非同期の仕事の仕組み) は、クライアントが対応していると申し出たときだけ使う。Tasks は、その後の草案で本体から拡張 (SEP-2663) へ移る話がある。形が固まるまでは、上の `wait_job` の方式を基本にする

A2A と MCP のどちらも、同じ仕事の置き場と待ち行列の上の薄い層にする。どちらから頼んだ仕事も、同じ順番で NPU に載る。

**手元のモデルを頭脳にするとき**: `hwnpu-chat agent` は、同じ仕事の種類を HTTP なしで道具として呼ぶ (同じ JSON Schema)。

**前提として足りないもの**: 決まった形で答える生成 (JSON Schema に沿うよう文法で縛ったデコード、`response_format`)。16.94 にもあるとおり、今の `chat` にはない。小さなモデルに分類や抜き出しを頼むとき、壊れた JSON を返さないために要る。

## 8. 同時通訳のサンプル (`samples/interpreter`)

- Go のプログラムが Web ページを配る。ブラウザのマイク (getUserMedia) の音声を WebSocket で受け、VAD → Whisper → PLaMo 3 Translate (前の発話を文脈にする) → LFM3-Audio TTS と処理し、訳した音声をブラウザで鳴らす
- `ai` パッケージを直接使う例と、HTTP と WebSocket で使う例を兼ねる
- 発話が終わってから訳す**逐次通訳**になる。遅れは 1 発話 2〜4 秒の見込み。Whisper は M3 で 30 秒の窓を 0.6 秒で読む。TTS の速さはまだ測っていない
- **VAD**: 最初は Go で書く音量の VAD (区切りのしきい値と、終わりを決めるまでの待ち時間) で流れを作る。その後 Silero VAD v5 (MIT、約 2 MB) に置き換える。Silero には `internal/onnx` にない LSTM と If がある。LSTM は MatMul と sigmoid / tanh に分解し、If は 16 kHz の側の分岐に固定する。32 ms ごとに NPU を呼ぶと呼び出しのコストが勝つので、0.5 秒分 (16 ステップ) を 1 つのプログラムに展開して呼ぶ

## 9. 実装の順

手分けのための作業の一覧 (依存、機械、合格の条件) は [app-api-tasks.md](app-api-tasks.md)。大きな流れは次のとおり。

0. 前提: Core ML のパッケージを決定的に書き、変換し直すかどうかを入力のハッシュで決める。今は `Manifest.json` の UUID が乱数で、`proto.Marshal` も決定的なモードなしなので、同じ入力で変換し直してもコンパイルのキャッシュが外れる (handoff-ane-cache.md 4 節)。直さないと、バンドルや配布の sha256 が安定せず、アプリを更新するたびに利用者の端末で ANE のコンパイルがやり直しになる
1. `ai` パッケージ: 登録、計画、準備、置き場の目録、取得元 (HF とミラー)。transcribe、translate、speech と音量の VAD を入れ、`hwnpu-audio` の transcribe と tts を、これを使う形に置き換える
2. `ai/server` と `internal/ws`: 音声の 2 つのエンドポイントとストリーム。`hwnpu-chat serve` をこれに載せ替える
3. `samples/interpreter`
4. ビルドタグ、`Prepare` を取得と変換に分ける作業、`hwnpu_noconvert`、バンドルと `hwnpu-dev bundle`
5. Silero VAD

その後: 擬似立体 (顔と深度、深度からメッシュ)、多レイヤー分割 (EfficientSAM と穴埋め。CLOUD_AI_COMPATIBILITY.md 2.1 節)。

## 10. 決めていないこと

- `ai` というパッケージの名前 (公開するとあとから変えにくい)
- 変換済みの配布 (prebuilt.md) と、この計画のどちらを先に作るか。バンドルの中身は配布の組と同じ並びにするので、4 番目の作業は配布の作業と一緒に進められる
- LFM3-Audio の日本語の読み上げの品質と速さ (未測定)
- 8 GB の機械で 3 つのモデルを置けるか (未測定)

## 11. v0 で実装したもの (2026-11-14)

`ai/` にある。2 節と 6 節の形から変えたところと、まだないものを書く。

| ファイル | 中身 |
|---|---|
| `stream.go` | `Chunk`、`Stream`、`Text`、`Of`、`Collect`、`Finals`、`Rechunk`、`SplitSentences`、`Sentences`、`Tee`、`Async` |
| `vad.go` | 音量の VAD (`NewVAD`、`Detect`、`Segments`)。背景の雑音の高さを追い、そこから 12 dB 上を発話とする。発話の終わりの待ち 600 ms、最短 250 ms、前後の余白 200 ms、最長 25 秒 |
| `plan.go` | `Registry` (`Require`、`Memory`)、`Plan` (`Items`、`Ready`、`DownloadBytes`、`MemoryGB`、`Licenses`、`Run`)、`Progress`、`Prepare` (1 回で済ませる CLI 向け)、`Definitions` |
| `store.go` | `Store` (`Dir`、`Bundles`、`Cache`)、置き場の目録 `store.json`、`ErrNotPrepared`、`ErrBundleTarget` |
| `runtime.go` | `Open`、`WithLog`、`WithCPU`、`Runtime` (`Transcriber`、`Translator`、`Speaker`、`VAD`、`Uses`、`Close`) |
| `transcribe.go`、`translate.go`、`speak.go` | 一括 (`Transcribe`、`Translate`、`Speak`) と流れ (`TranscribeStream`、`TranslateStream`、`SpeakStream`) |

- **準備は今の `Prepare` を呼ぶ**: `transcribe.Prepare`、`speech.Prepare`、`chat.Prepare` (AMD では `chat.AMDExport`) を、`hwnpu-audio` と `hwnpu-chat` と同じ設定で呼ぶ。プログラムの名前が同じになるので、CLI で準備したものはそのまま使える (`DefaultStore` は CLI の置き場)。準備の結果 (プログラムの道筋) を `store.json` に、置き場からの相対の道筋で書く。`Open` はそれだけを読み、`Prepare` を呼ばない
- **バンドルの最初の形**: 準備した置き場をそのままコピーしたものがバンドルになる (`store.json` の道筋が相対なので)。`store.json` に対象 (`apple`、`intel`、`amd-npu2`) を書き、違う機械では `ErrBundleTarget`。`hwnpu-bundle.json` (4 節) の目録、メモリの段、`hwnpu-dev bundle` はまだない (E3)
- **計画はファイルを数えない**: 「準備済みか」は `store.json` に同じモデルで書かれていて、ファイルが全部あるかで決める。落とす量は、チェックポイントがなければ models.yaml の `bytes`。変換が要るかどうか、どのファイルを何バイト落とすかの正確な一覧は、`Prepare` を取得と変換に分けてから (A3、B1 の残り)
- **取得元**: `ai.HuggingFace()`、`ai.Mirror()`、`ai.Source` はまだない。今は models.yaml の `sources` と `search` (フォルダと S3) が効き、アプリは `ai.Definitions` で自分の定義のファイルを読ませる。HTTPS のミラーは `internal/chat` の取得元に種類を 1 つ足す形にする
- **進み具合**: 今は `internal` が書くログの行 (`Progress.Message`) と段階 (`prepare`、`warmup`、`done`) だけ。バイト数の入った構造のある値は、取得を分けてから
- **試しの読み込み**: NPU のある機械では、準備の最後に 1 回読み込んで閉じる (`Plan.NoWarmUp` で外せる)
- **6 節の形から変えたところ**: `rt.Transcriber(ctx, use)` のように、読み込みの関数は ctx を取る (初めの呼び出しでプログラムを読み込み、数十秒かかることがあるため)。`Plan` は取得元を引数に取らない
- **流れの形の制限**: `TranscribeStream` は区切り (`Final`) ごとにまとめて読む (話している途中の仮の字幕は B5)。`SpeakStream` は文ごとに全部作ってから返す (できた音から返すのは B6)。`TranslateStream` は直前の 4 組を履歴にする
- **NPU で動くかの報告**: `Transcriber.OnNPU`、`Translator.OnNPU` (足した `chat.Engine.OnNPU`)。`Speaker` はテキスト部のセッションの置き場所を外から見られないので、まだない
- **確かめたこと**: CPU で動くテスト (流れの道具、VAD の一括と流れの一致、計画 → 準備 → 2 回目の計画で何も要らない、バンドルへのコピー、対象の違い)。`-race` で 20 回。Linux と Windows へのビルド。NPU での確認 (`HWNPU_AI_STORE`、`HWNPU_AI_USES` を付けた `TestOnDevice`) はまだ回していない
