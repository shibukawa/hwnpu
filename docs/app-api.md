# アプリに組み込むライブラリ: `ai` のパッケージ群

作成: 2026-11-11。書き直し: 2026-11-21 (v0 を main に載せ直し、構成を変えた。design.md 16.281)。作業の一覧は [app-api-tasks.md](app-api-tasks.md)。使い方と、制約とルール (NPU は 1 つの待ち行列、実時間で回るかの見積もり、メモリ)、API の設計の論点 (D1〜D21) は [developer-guide.ja.md](developer-guide.ja.md) ([English](developer-guide.md))。

hwnpu は、NPU で AI のモデルを動かす Go のライブラリの集まりにする。`hwnpu-chat`、`hwnpu-image`、`hwnpu-audio` は、その機能を 1 つずつ試すためのデモで、中身はライブラリを呼ぶだけにしていく (コマンドの名前と場所は変えない)。アプリは複数のモデルをつないで使う。例:

- 同時通訳: 文字起こし → 翻訳 → 読み上げ (`samples/interpreter`、動く)
- 擬似立体: 画像の生成 → 深度の推定 → カメラの映像から頭の位置を追い、視差をつけて描く
- 声で絵を直す: 絵の構成要素の検出と、文字起こしと、画像の編集

## 1. パッケージの構成

| パッケージ | 中身 | 状態 |
|---|---|---|
| `hwnpu` (ルート) | グラフを動かす層: `Device`、`Graph`、`Buffer`、`Request` | ある |
| `hwnpu/modelstore` | モデルのファイルの置き場と取得元 (手元のキャッシュ、`HWNPU_MODEL_SOURCE*`、同梱、HF) | ある |
| `hwnpu/npuinfo` | OS が NPU について教えること | ある |
| `hwnpu/ai` | アプリが使うモデルの登録 (`App`、`Use`)、計画と準備 (`Plan`)、実行 (`Runtime`) | v0 |
| `hwnpu/ai/flow` | モデルの間のデータの渡し方: 塊の流れ (`Stream`)、非同期、メトロノーム | v0 |
| `hwnpu/ai/speech` | 文字起こし (`Transcribe`)、読み上げ (`Speak`)、音量の VAD、WAV の読み書き | v0 |
| `hwnpu/ai/text` | 翻訳 (`Translate`)、チャット (`Chat`) | v0 |
| `hwnpu/ai/aicli` | アプリのコマンドラインに足す kong のサブコマンド: `models` (一覧、ダウンロード、コンパイル)、`status` | v0 |
| `hwnpu/ai/vision` | 深度、顔、物体、姿勢、切り抜き、OCR、ラベル | まだ |
| `hwnpu/ai/imagegen` | 画像の生成 (Anima、FLUX.2 / FLUX.3 klein、Chroma) | まだ |
| `hwnpu/ai/sound` | 音楽、効果音、音高、採譜 | まだ |

- **機能ごとにパッケージを分ける**: 文字起こしだけを使うアプリに、画像の生成のコードを入れないため。`ai` は機能のパッケージを import しない。機能のパッケージが、自分のモデルの準備と読み込みのしかた (`internal/airole` の `Role`) を持ち、`ai.Define` で型のついた `Use` を作る
- **`internal/airole` は機能のパッケージと `ai` の間の約束**: 外のモジュールは `Role` を作れないので、`Use` を作れるのは hwnpu の機能のパッケージだけ
- 16.94「モデルを部品として組み合わせる」の 3 層 (ライブラリ、常駐のプロセス、CLI) のうち、これは 1 層目。常駐の管理 (メモリの予算、投入の順番) はまだない (10 節)

## 2. 登録 → 計画 → 準備 → 実行

**実行はネットワークに触らず、変換もしない。** 準備済みの置き場からだけ読み込む (`WithAutoPrepare` を付けたときだけ、CLI のように初回に準備する)。

```go
// 登録: アプリが使うモデルを、機能のパッケージの Use で宣言する。名前は models.yaml のモデル、ファミリー、用途
var (
	stt = speech.Transcribe("transcribe")
	tr  = text.Translate("translate")
	tts = speech.Speak("speech")

	app = ai.NewApp("interpreter", stt, tr, tts)
)

// 計画: 何が足りないか、何バイト落とすか、動かすときのメモリ、ライセンスを調べる。まだ何も落とさない
plan, err := app.Plan(ctx)
plan.DownloadBytes(); plan.MemoryGB(); plan.Licenses()

// 準備: 層か取得元から取る → 変換 (層にあれば取るだけ) → 1 回読み込む (NPU 向けのコンパイル)
plan.NoCompile = true          // ダウンロードと変換だけ (NPU は要らない)
err = plan.Run(ctx, func(p ai.Progress) { ... })

// 実行: 準備されていなければ ErrNotPrepared
rt, err := app.Open(ctx)
defer rt.Close()
t, err := stt.Get(ctx, rt)     // *speech.Transcriber。最初の Get で読み込み、Release か Close まで持つ
text, err := t.Transcribe(ctx, pcm16k, speech.TranscribeOptions{Language: "ja"})
```

### 2.1 機械のメモリごとのモデルの構成

用途の名前 (`translate`) は、1 つだけ動かすときにその機械で動く一番大きなモデルを選ぶ (16 GB から PLaMo 3 Translate 8B、8.2 GB)。いくつも同時に動かすアプリでは、主になるモデルを大きく、補助のモデルを小さく速くしたい。アプリは登録のときに、用途ごとに機械のメモリごとの構成を書く。

```go
var (
	stt = speech.Transcribe("transcribe")
	tr  = text.Translate("translate").      //  8 GB: 定義の選ぶもの (0.8B)
		At(16, "plamo3translate:2b").       // 16 GB から: 文字起こしと読み上げの横なので 2B
		At(32, "plamo3translate:8b")        // 32 GB から: 8B
	tts = speech.Speak("speech")
	aux = text.Chat("chat").As(8)           // どの機械でも、8 GB の機械として選ぶ (補助のモデル)
)
```

| 口 | 意味 |
|---|---|
| `use.At(gb, name)` | メモリが gb GB 以上の機械では、この用途を `name` (モデル、タグつきのファミリー、用途) にする。いくつも書けて、機械のメモリ以下で一番大きい段が効く。どの段にも届かない機械は元の名前 |
| `use.As(gb)` | この用途を、gb GB の機械として選ぶ (モデルの選び方と、models.yaml の `variants` によるプログラムの選び方の両方)。補助のモデルをどの機械でも小さく保つ |
| `app.Memory(gb)` | アプリ全体を gb GB の機械として計画する (小さい機械向けに準備するとき)。`At` の段もそのメモリで決まる |

- 選んだ理由は `Item.Why` に入り、`APP models` が出す (「plamo3translate:2b on machines of 16 GB and up, by the app」)。`models` は、同時に動かす合計と機械のメモリも並べて出し、合計が機械のメモリを超えるときはそう書く
- 利用者の側の調整はコマンドと同じに効く: `HWNPU_MEMORY` (機械のメモリを言い換える)、`hwnpu memory MODEL GB` (モデルごと)、`HWNPU_MODEL_ALLOWLIST`
- **置き場の記録と今の選び方が違えば、準備されていない扱い**: 目録にあるのが別のモデル (アプリの構成が変わった、機械のメモリの設定が変わった) や別のメモリ向けのプログラムなら `ErrNotPrepared` になり、前のものを黙って読み込まない。準備し直すと記録が入れ替わる (チェックポイントや書き出しは残っているので、作り直すのは足りないものだけ)

### 2.2 そのほか

- **`Use[T]` は型のついた取っ手**: `stt.Get` は `*speech.Transcriber` を返す。用途の名前を文字列で引いて型を合わせる必要がない。登録していない `Use` の `Get` はエラー
- **用途の名前は、計画でも実行でも同じ決まりで引く**: 定義 (models.yaml とアプリの定義のファイル) とアプリの構成 (2.1 節) から、その機械のモデルが決まる。実行時もネットワークなしで同じように引き、置き場の目録 (`store.json`) の記録と比べる
- **準備は 2 段**: (1) ダウンロードと変換 (`NoCompile`)。NPU が要らないので、ビルド用の機械でも、NPU をほかが使っているときでもできる。(2) 1 回読み込む。Apple の ANE のコンパイル結果はその機械と OS のものなので配れない (prebuilt.md 1 節)。Gemma 4 のように初回が 100 秒かかるものもあるので、準備の最後に済ませておく。目録にはコンパイル済みかどうかも書く
- **1 つずつ使う**: `use.Release(rt)` でモデルを下ろす。画像の生成のあとに深度、のように順に使うアプリは、同時に置くメモリが一番大きいモデルの分で済む
- **ライセンス**: 同意の要るライセンス (`License.Accept`) のモデルは、`plan.AcceptLicenses` を付けないと `*ai.LicenseError`。アプリが画面で同意を取ってから付ける
- **CLI と同じ置き場**: `ai.Store` のゼロ値はコマンドと同じ (環境変数と `.env`)。`hwnpu-audio` や `hwnpu-chat` が準備したものはそのまま使え、アプリが準備したものをコマンドも使う

## 3. アプリのコマンドライン (`ai/aicli`)

アプリは kong で自分のコマンドを作り、モデルの世話をするサブコマンドを `aicli` から足す (kong に決め打ち)。

```go
var cli struct {
	Run    RunCmd       `cmd:"" default:"withargs" help:"..."`
	Models aicli.Models `cmd:"" help:"The models interpreter uses: list, download, prepare."`
	Status aicli.Status `cmd:"" help:"This machine's NPU and what uses it."`
}

k := kong.Parse(&cli, aicli.Bind(ctx, app)) // アプリ自身のコマンドも Run(ctx context.Context, app *ai.App) で受け取れる
k.FatalIfErrorf(k.Run())
```

| コマンド | 中身 |
|---|---|
| `APP models` (`list`) | 使うモデルの一覧: 状態 (missing、ready, not compiled、compiled、bundled)、落とす量、メモリ、ライセンス、同時に動かす合計と機械のメモリ、アプリが構成で選んだ理由。`--json` |
| `APP models download` | 足りないものを落として変換する。読み込まない (NPU は要らない)。この後はネットワークなしで起動できる |
| `APP models prepare` (`compile`) | download のあと、各モデルを 1 回読み込んで NPU 向けにコンパイルする |
| `APP status` | `hwnpu status` と同じ (デバイス、ドライバ、負荷、電力、使っているプロセス、キャッシュ)。`-w`、`--json`、`--wake` |

- 同意の要るライセンスは、端末なら聞き、`--accept-license` で聞かずに通す
- `hwnpu status` の中身は `internal/npustatus` に移した (`cmd/hwnpu` と `aicli` が同じものを呼ぶ)。アプリの `status` はアプリの置き場 (`app.Store`) のキャッシュと層を出す

## 4. モデルの間のデータの渡し方 (`ai/flow`)

次の段をいつ始められるかで、つなぎ方は 3 つある。どれも `flow.Stream[T]` (Go の `iter.Seq2[Chunk[T], error]`) の上にあり、段は「流れを受けて流れを返す関数」なので、前の段の結果を次の段に渡すだけでつながる。止めるのは `break`、エラー、ctx の取り消し。

```go
// 塊: 中身、流れの時計の上の位置と長さ、区切り (発話や文の終わり、でき上がった絵)
type Chunk[T any] struct {
	Data       T
	Start, Len time.Duration // 流れの始まりから。音声は標本の数から計算し、壁時計でずれない
	Final      bool
}
```

| つなぎ方 | いつ使う | 道具 |
|---|---|---|
| **前が全部終わってから** | 次の段が結果の全体を要る (絵ができてから深度) | 一括の呼び出し (`Generate` → `Estimate`)、`flow.Last` (流れを最後まで読む)、`flow.Of` (全体を 1 つの塊として渡す)。大きなモデルは `Use.Release` で下ろしてから次を読む |
| **できたものから** | 前の段が次を作っている間に、できた塊で始める (文ごとに訳して読む、カメラの映像を追う) | `flow.Async` (段を重ねる。塊は落とさない)、`flow.Latest` (読む側が遅いとき古い塊を捨て、いつも最新を読む)、`flow.Hold` (最新を持っておき、欲しいときに読む)、`flow.Tee` (1 つの流れを複数の読み手に) |
| **拍に合わせて** | 決まった時刻に出力が要る (再生に間に合わせる音、画面に出すフレーム) | `flow.Metronome` (拍ごとの tick。`Lead` だけ前に渡す。`Skip` で過ぎた拍を捨てる)、`flow.Pace` (塊をその時刻に渡す。再生側)、`flow.Live` (録音をマイクのように実時間で流す) |

```go
// できたものから: 発話ごとに書き起こし、訳し、読む。読み上げは訳の 4 つ先まで待たせずに進める
speech := speech.NewVAD(speech.VADOptions{}).Detect(ctx, mic)
texts := stt.TranscribeStream(ctx, speech, sopts)
en := tr.TranslateStream(ctx, texts, topts)
audio := tts.SpeakStream(ctx, flow.Async(ctx, en, 4), popts)
for c, err := range audio { play(c.Data) }

// 拍に合わせて: 1 秒に 60 回描く。頭の位置は検出器が出した最新のもの (検出はカメラの最新のフレームだけを読む)
head := flow.Hold(ctx, faces.DetectStream(ctx, flow.Latest(ctx, camera)))
for tick := range (flow.Metronome{Period: time.Second / 60, Skip: true}).Ticks(ctx) {
	pos, _ := head.Latest()
	draw(mesh, pos.Data, tick.Data.At)
}
```

- **どの機能も 2 つの形を持つ**: 一括 (値を 1 つ渡して 1 つ受け取る) と流れ (`...Stream`)。一括は塊が 1 つの流れと同じ
- **塊の大きさは受け取る側が決め直す**: NPU のプログラムは形が固定なので、入る側の塊の大きさは自由にし、段が自分に合う大きさに組み直す (`flow.Rechunk`)
- **時計は標本の数**: 音声の流れの時刻は、流れの始まりからの標本の数で数える。壁時計を見るのは `Metronome`、`Pace`、`Live` だけ
- **拍に遅れたとき**: `Metronome` は遅れを `Tick.Late` で知らせる。穴を開けられない出力 (音) は全部の拍を遅れて受け取り、今を出す出力 (描画) は `Skip` で過ぎた拍を捨てる (捨てた数は `Tick.Skipped`)
- **制御はデータと別**: 取り消しは ctx。割り込み (相手が話し始めたら読み上げを止める) はまだない
- **テスト**: 時刻に関わるものは `testing/synctest` の偽の時計で、待たずに決定的に確かめる
- 道具の数は少なく保ち、必要になったものから足す。段の間で NPU の仕事の順番を決める仕組み (音声を先に、など) はまだない: 各モデルは自分の中で 1 回に 1 つの要求を通す

## 5. 取得元

取得元は `ai` の型にせず、コマンドと同じ層 (公開の `modelstore`、design.md 16.212、16.213、cache-sharing.md 7 節) をそのまま使う。`ai.Store.Models` は `modelstore.Options` で、ゼロ値はコマンドと同じ (環境変数と `.env`)。

| 層 | 中身 | 変換 |
|---|---|---|
| 0. この機械のキャッシュ (`Options.Root`、`HWNPU_CACHE`) | 元のファイル (`original/OWNER/NAME` と `hwnpu-lock.json`)、変換したもの (`models/NAME`、`converted/`)。いつも書き戻す。使わないものから消える (LRU、`Options.Limits`) | — |
| 1. 取得元 (`Options.Sources`、`HWNPU_MODEL_SOURCE*`) | `file://`、`http(s)://` (WebDAV、読むだけの HTTP)、`s3://`。`?writeback=true` で後ろから読んだものと作ったものを置く。Core AI の書き出しなど変換済みのものもここから取れる | あればしない |
| 2. 同梱のモデル (`Options.Bundle`) | アプリの資源フォルダ (configdir) の `models/` に、取得元と同じ並びで置いたもの。読むだけ、その場で使う | あればしない |
| 3. Hugging Face (`Options.HuggingFace`) | `chat.Fetch`。途中から再開でき、同意の要るリポジトリはデバイスフローでサインインする (design.md 16.153) | する |
| 準備済みの同梱 (`ai.Store.Prepared`) | 準備したキャッシュの一部を写したもの (`store.json` と、それが名指すファイル)。`Run` を通らずに読む (6 節、10 節) | **しない** |

- 定義の重ねはコマンドと同じ: 組み込み → 機械 (`/etc/hwnpu/models.yaml`、`/Library/Application Support/hwnpu/models.yaml`、`%ProgramData%\hwnpu\models.yaml`) → 利用者 (`~/.config/hwnpu/models.yaml`) → `HWNPU_MODELS` → アプリのファイル (`modelstore.Options.Files`、足した欄)。モデルを何 GB の機械として動かすかの選択 (`hwnpu memory`、`run.json`) と、モデルの方針 (`HWNPU_MODEL_ALLOWLIST`) も効く
- 層はプロセスの設定 (`modelstore.Open`)。`App.Plan` と `App.Open` はそのたびに `Store.Models` で設定し直す
- 前に考えていた `ai.HuggingFace()`、`ai.Mirror(baseURL)`、`ai.Source` (アプリが SDK で実装する取得元) は作らない。HTTPS のミラー、WebDAV、S3 は層 1 の URL で足りる
- どの取得元でも、models.yaml の `sha256` と `hwnpu-lock.json` で照合する。同梱のモデルは lock を信頼して計り直さない
- ライセンス: ミラーや同梱が許されるかどうかは、モデルごとに異なる。Gemma の規約、同意の要るリポジトリ、CC-BY の表示 (Magenta RealTime 2) などがある。`plan.Licenses()` は models.yaml の `license` (id、class、terms、url、accept) を返す

## 6. バンドル: NPU の種類ごとに作り、実行時には変換しない

> この節は計画で、まだ実装していない。今あるのは「準備済みの同梱」(`Store.Prepared`、10 節) だけ。

**アプリを配布するときは、NPU の種類ごとにビルドする。** 同梱するのは変換済みのプログラムで、実行時の変換はない。

### 6.1 対象の名前

cache-sharing.md 3.1 節と同じ名前を使う (層のエントリの `target` と同じ)。

| 対象 | 同梱するもの | 1 つのバンドルで動く範囲 |
|---|---|---|
| `coreai` | Core AI の書き出し (`NAME.coreai`、端末によらない `.aimodel`)。Mac の Neural Engine は Core AI だけで動かす (macOS 27、design.md 16.185、16.196)。端末で専用化したものは Core AI 自身のキャッシュにあり、配れない | M1〜M4 で同じファイル。macOS の最低の版を目録に書く |
| `amd-xdna2` | IRON の書き出し (ELF、xclbin、バンク、manifest) | XDNA 2 (Strix 以降) |
| `amd-xdna1` | `--per-step` の書き出し | XDNA (Phoenix、Hawk Point)。実機で未確認 |
| `intel` | OpenVINO IR (`.xml`、`.bin`) | Core Ultra。NPU の blob は、ドライバが読み込むときにコンパイルする |

### 6.2 中身と目録

```
bundle/
├── hwnpu-bundle.json    目録: format_version、対象 (coreai など)、作った hwnpu のコミット、
│                        macOS の最低の版、各モデルの名前、メモリの段、ファイルと sha256、ライセンス
├── whisper-large-v3-turbo/   トークナイザー、埋め込みの表、プログラム (prebuilt.md 4 節の組と同じ並び)
├── plamo-3-translate-0.8b/
└── LFM3-Audio-2B-TTS/
```

- **メモリの段**: 用途の名前はメモリで選ぶモデルが変わる。バンドルを作るときに、入れる段を決める (例: 8 GB の段と 16 GB の段)。実行時は、**入っている段のうち**マシンのメモリに合うものを選ぶ。合う段がなければ `ErrBundleTarget` を返し、変換にも取得にも逃げない
- **実行時の照合**: `App.Open` は、目録の対象とその機械の NPU (`hwnpu.Target()`) を比べる。違えば、何が違うかを書いた `ErrBundleTarget` を返す。format_version が読めない範囲のときも同じ
- 中身は prebuilt.md の組とファイルの並びを同じにする。変換済みの配布を、そのままバンドルにもミラーにも使えるようにするため

### 6.3 作り方: `hwnpu-dev bundle`

```
hwnpu-dev bundle -target coreai -use transcribe,translate,speech,vad -memory 8,16 -o dist/bundle
```

- 計画と準備 (2 節) を、その対象とメモリの段で行い、置き場から目録とファイルを集める。変換済みの配布があればそれを使い、なければ手元で変換する
- **`coreai` と `intel` は、どの OS の機械でも作れる**: 変換は Go だけで、Python も Mac も要らない (README の `hwnpu-dev convert`)。Neural Engine のコンパイルは初回に端末で行う (準備の試しの読み込み)
- `amd-*` は IRON の環境が要る。aix1 で作るか、変換済みの配布を使う
- バンドルに入れるモデルのライセンスを確かめ、再配布できないものは失敗にする (`-allow-license` で明示したときだけ通す)

## 7. ビルドタグ

> この節は計画で、まだ実装していない。

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
| Mac 版 | darwin / arm64 | `hwnpu_apple,hwnpu_noconvert` | `coreai` のバンドル |
| Linux の Ryzen AI 版 | linux / amd64 | `hwnpu_amd,hwnpu_noconvert` | `amd-xdna2` (と `amd-xdna1`) のバンドル |
| Windows の Ryzen AI 版 | windows / amd64 | `hwnpu_amd,hwnpu_noconvert` | `amd-xdna2` のバンドル |
| Core Ultra 版 | linux、windows / amd64 | `hwnpu_intel,hwnpu_noconvert` | `intel` のバンドル |

`hwnpu_noconvert` を入れるには、今の `Prepare` を「取得」と「変換」に分け、変換のファイルにタグを付ける必要がある。`chat.Prepare`、`transcribe.Prepare`、`speech.Prepare`、`internal/onnx` を使うものが対象になる。手間が大きいので、あとに回す (app-api-tasks.md の E2)。

## 8. ローカルサーバー (`github.com/shibukawa/hwnpu/ai/server`)

> この節は計画で、まだ実装していない。

`http.Handler` として作り、アプリが自分のサーバーに組み込める。`hwnpu-chat serve` もこれを使う。**既存のサービスの API との互換は、サービスごとに下のパッケージに分ける**。共通の部分 (仕事の置き場、認証、読み書きしてよいディレクトリ、ストリーム) は `ai/server` に置き、互換のパッケージはそれぞれの形の要求を `ai` の呼び出しに直す薄い層にする。

```
ai/server               共通: 仕事の置き場と待ち行列、トークン、許すディレクトリ、/v1/capabilities、
                        独自のストリーム (/v1/audio/stream)、下の互換の口をまとめる Handler
ai/server/openai        OpenAI 互換
ai/server/googlecloud   Google Cloud の REST 互換
ai/server/a2a           A2A (8.1 節)
ai/server/mcp           MCP (8.1 節。hwnpu-mcp はこれを stdio で動かす)
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

### 8.1 エージェント向けの口: 重い仕事を下請けする

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
| `GET /v1/jobs/{id}/events` | SSE。進み具合と途中の結果 (4 節の塊)。多数の件を処理する仕事は、1 件ずつの結果を JSONL で |
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

## 9. サンプル

### 9.1 同時通訳 (`samples/interpreter`、動く)

マイクか録音 (WAV) の音声を、VAD → Whisper → PLaMo 3 Translate (前の発話を文脈にする) → LFM3-Audio の読み上げとつなぎ、訳した音声の WAV を書く。

```
interpreter models                    # 使うモデルと、要るもの
interpreter models prepare            # ダウンロード、変換、コンパイル
interpreter --mic -o english.wav      # マイクから。Ctrl-C で止める (interpreter mics が入力を並べる)
interpreter talk.wav -o english.wav   # 録音から: 日本語の音声を英語の音声に (逆も)
interpreter talk.wav --live           # 録音をマイクのように実時間で流す
interpreter status
```

- **サンプルは別のモジュール** (`samples/go.mod`、`replace github.com/shibukawa/hwnpu => ../`): サンプルだけが要る依存を、ライブラリ本体の `go.mod` に足さないため。別のモジュールなので `internal` は import できず、公開のパッケージだけで書けることの確かめにもなる
- **マイクとカメラは [hwcamera](https://github.com/shibukawa/hwcamera)** から取る (cgo なし。Linux は PulseAudio / PipeWire と V4L2、macOS は AudioQueue と AVFoundation、Windows は WASAPI と Media Foundation)。流れに直すコードは `samples/capture`。入力は録音のファイルでもよく、後ろの段は同じ
- **生の入力はためる**: 流れは読む側が引くので、文字起こしと翻訳が動いている数秒の間、誰もマイクを読まない。そのままだとその間の音が失われる (実機で 2 つ目の発話の頭が欠け、3 つ目が消えた)。`capture.Mic` は別の goroutine で読んで 1 分ぶんまでためる。映像は逆に `flow.Latest` で古いフレームを捨てる

- 発話が終わってから訳す**逐次通訳**。話している途中の仮の字幕 (B5) と、できた音から返す読み上げ (B6) はまだ
- 次の形: ブラウザのマイクの音声を WebSocket で受ける (`ai/server`、8 節) ものは D1 に残る

### 9.2 擬似立体 (まだ)

画像の生成 (Anima: アニメの絵は前景と背景がはっきり分かれ、視差と相性がよい) → 深度 (前が全部終わってから) → カメラの映像から顔 (`Latest`) → 頭の位置を持っておき (`Hold`)、メトロノームで描く。要るもの: `ai/vision` (深度、顔)、`ai/imagegen`、深度からメッシュと描画 (Go)、カメラの入力 (hwcamera。フレームは Go のメモリの画像にしてから流れに載せる。録画や画像のファイルでも動くようにする)。

## 10. v0 の実装 (2026-11-21)

| ファイル | 中身 |
|---|---|
| `ai/app.go` | `App` (`NewApp`、`Use`、`Memory`)、`Use[T]` (`Get`、`Release`、`At`、`As`)、`Define` |
| `ai/plan.go` | `Plan` (`Items`、`Ready`、`Compiled`、`DownloadBytes`、`MemoryGB`、`MachineGB`、`Licenses`、`Run`)、`Progress`、`LicenseError`、`App.Prepare` |
| `ai/store.go` | `Store` (`Models`: `modelstore.Options`、`Prepared`)、置き場の目録 `store.json`、`ErrNotPrepared`、`ErrBundleTarget` |
| `ai/runtime.go` | `App.Open`、`Runtime` (`Loaded`、`Close`)、`WithLog`、`WithAutoPrepare`、`WithAcceptedLicenses` |
| `ai/flow` | 4 節の道具と、`Collect`、`Finals`、`Map`、`From`、`Sentences` |
| `ai/speech` | `Transcribe`、`Speak`、`NewVAD` (音量の VAD: 雑音の床から 12 dB、終わりの待ち 600 ms)、`ReadWAV`、`WriteWAV`、`Chunks` |
| `ai/text` | `Translate` (前の 4 組を履歴に)、`Chat` (`Complete`、`CompleteStream`) |
| `ai/aicli` | `Models`、`Status`、`Bind` |
| `internal/airole` | `Role` (準備と読み込み)、`Files`、この機械の NPU (`Here`: `coreai`、`intel`、`amd-xdna2`) |
| `internal/npustatus` | `hwnpu status` の中身 (`cmd/hwnpu/status.go` から移した) |

- **準備は今の `Prepare` を呼ぶ**: コマンドと同じ関数を同じ設定で呼ぶ。AMD では `transcribe.PrepareAMD`、`speech.PrepareAMD`、`chat.AMDExport` (IRON の書き出しを手元、取得元、なければ IRON で)、ほかでは `transcribe.Prepare`、`speech.Prepare`、`chat.Prepare`。プログラムの置き場所も名前も同じなので、CLI が準備したものはそのまま使える
- **`store.json`**: キャッシュの根 (`cache.Root`) にあり、道筋は根からの相対。NPU の対象 (`coreai`、`intel`、`amd-xdna2`) ごとに用途の記録を持つ (1 つのキャッシュを違う NPU の機械で共有しても互いを消さない)。記録は、モデル、定義の `run` (別のメモリを選べば準備し直す)、ファイル、準備した時刻、コンパイルした時刻。書き込みは `cache.Lock` の下
- **準備済みの同梱** (`Store.Prepared`): 準備したキャッシュから `store.json` と名指すファイルを写したもの。`Run` を通らずに読める。別の対象のものだけなら `ErrBundleTarget`。`hwnpu-bundle.json` の目録と `hwnpu-dev bundle` (6 節) はまだない
- **計画はファイルを数えない**: 「準備済みか」は `store.json` に同じモデルで書かれ、ファイルが全部あるかで決める。落とす量は、チェックポイントがなければ models.yaml の `bytes` (上限。層に変換済みのものがあれば落とさない)
- **進み具合**: `internal` が書くログの行 (`Progress.Message`) と段階 (`prepare`、`compile`、`done`)。バイト数の入った構造のある値は、取得を分けてから (B1 の残り)
- **AMD の NPU に複数のモデルを同居させるための直し**: 1 つのプロセスが 2 つ目のデバイスを開くと `CREATE_HWCTX` が `EINVAL` になった。デバイスヒープを 64 MiB の境界の番地に写すようにして直った (design.md 16.281)
- **確かめたこと**:
  - 機械のメモリごとの構成: `HWNPU_MEMORY=16` では翻訳が plamo-3-translate-2b になり (3 つで 7.8 GB、読み込み 3.1 秒、全体 25.5 秒)、設定なし (59 GB) では 8B (11.6 GB)。8B で準備した置き場を 16 GB として開くと `ErrNotPrepared` (前は黙って 8B を読み込んでいたので直した)
  - `go test -race ./ai/...`: 流れの道具 (偽の時計で `Async` の重なり、`Latest` の間引き、`Hold`、`Metronome` の遅れと `Skip`、`Pace`、`Live`)、計画 → ダウンロード → コンパイル → 2 回目の計画で何も要らない、`Get` / `Release` / `Close`、`WithAutoPrepare`、役割と種類の食い違い、メモリごとの構成 (`At`、`As`) と別のモデルで準備された置き場、アプリの定義のファイル、ライセンスの同意、準備済みの同梱と対象の違い、`aicli` のコマンド
  - Ryzen AI 9 365 (XDNA 2) で `samples/interpreter`: 13.3 秒の日本語の音声 (2 発話、3 文) を、whisper-large-v3-turbo、plamo-3-translate-8b、LFM3-Audio-2B-TTS の 3 つを 1 プロセスに同居させて英語の音声 8.5 秒に。読み込み 0.5、5.1、2.6 秒、全体 31.6 秒。書き起こしと訳は正しく、出力の音声を Whisper に戻すと訳文と一字一句同じ
- **確かめていないこと**: Mac (Core AI) と Intel での実機 (準備と読み込みはコマンドと同じ関数だが、`ai` を通しては動かしていない)。`Chat` の実機。8 GB の機械での 3 つの同居

## 11. まだないもの、決めていないこと

- **コマンドの載せ替え** (B3): `hwnpu-audio transcribe` / `tts`、`hwnpu-chat`、`hwnpu-image` を `ai` の上に載せる。コマンドにだけある設定 (重みの形式、`--cpu`、手で変換したプログラム、窓ごとの時間) をライブラリの口に出してから
- **メモリの合計から自動で構成を選ぶこと**: 今はアプリが段を書く (2.1 節)。合計が機械のメモリに収まるように小さいモデルへ自動で落とすことはしない (どれを主にするかはアプリが知っている)
- **常駐の管理** (16.94): メモリの予算、使っていないモデルの追い出し、NPU への投入の順番 (音声を先に)。今は `Release` を呼ぶアプリ任せで、モデルは自分の中でだけ要求を 1 つずつ通す
- **デバイスの共有**: 今はモデルごとにデバイスを開く (内部の `Load` がそうなっている)。1 回だけ開いて共有する形にするかは、常駐の管理と一緒に決める
- **Mac の Neural Engine のロック** (B4): プロセスが生きている間ロックを持つので、常駐するアプリがほかの hwnpu のコマンドを待たせる
- 流しながらの文字起こし (B5) と読み上げ (B6)、Silero VAD (F2)
- `ai/server` (8 節)、バンドル (6 節)、ビルドタグ (7 節)
- `ai` というパッケージの名前 (公開するとあとから変えにくい)。2 つ目のサンプル (擬似立体) が動くまで v0 とし、形を変えてよいものとする
