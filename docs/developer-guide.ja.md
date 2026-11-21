# 開発者向けガイド: hwnpu でアプリを作る (`ai` のパッケージ群)

[English](developer-guide.md)

> **v0**: この API は最初の版で、形はまだ変わります。この文書は、使い方の説明と、API の設計を今の段階で見直すための材料を兼ねています。設計の論点は [10 節](#10-設計の論点) に番号をつけて並べました (「D4 は案 B で」のように指してください)。

hwnpu は、NPU で AI のモデルを動かす Go のライブラリの集まりです。`hwnpu-chat`、`hwnpu-image`、`hwnpu-audio` は、その機能を 1 つずつ試すためのデモという位置付けです。アプリは複数のモデルを登録し、前もって準備し、つないで動かします。

- 今あるもの: 文字起こし、翻訳、チャット、読み上げ、音量の VAD、深度の推定、顔の検出と頭の位置、画像の生成と編集 (FLUX.2 klein)、モデルの間のつなぎ方、アプリに足すサブコマンド
- まだないもの: FLUX.2 klein のほかの画像の生成 (Anima、Chroma)、物体・姿勢・切り抜き・OCR、音楽と効果音、常駐の管理、ローカルサーバー ([11 節](#11-まだないもの))
- 実機で動かしたのは AMD の NPU (Ryzen AI 9 365) だけです。Mac と Intel は、コマンドと同じ準備と読み込みを呼びますが、このライブラリを通しては動かしていません
- 計画の全体は [app-api.md](app-api.md)、作業の一覧は [app-api-tasks.md](app-api-tasks.md) (どちらも日本語)

## 1. 全体像

```
登録                     計画                 準備                        実行
ai.NewApp(uses...)  →  app.Plan(ctx)   →   plan.Run(ctx, fn)     →    app.Open(ctx) → use.Get(ctx, rt)
アプリが使うモデル      何が足りないか        ダウンロード、変換、          準備済みの置き場からだけ
                       落とす量、メモリ、     NPU 向けのコンパイル          読み込む (ネットワークなし)
                       ライセンス
```

| パッケージ | 役割 |
|---|---|
| `github.com/shibukawa/hwnpu/ai` | `App` (登録)、`Use[T]` (型のついた取っ手)、`Plan` (計画と準備)、`Runtime` (実行)、`Store` (置き場) |
| `.../ai/speech` | 文字起こし `Transcribe`、読み上げ `Speak`、音量の VAD、WAV の読み書き |
| `.../ai/text` | 翻訳 `Translate`、チャット `Chat` |
| `.../ai/vision` | 深度の推定 `Depth`、顔の検出 `Faces`、顔から頭の位置 `HeadOf` |
| `.../ai/imagegen` | 画像の生成と編集 `Generate` (文から、文と参照の絵から) |
| `.../ai/flow` | モデルの間のデータの渡し方: `Stream` と道具、ネットワーク (節と `To`)、呼び出し (`Call`) |
| `.../ai/cli` | アプリのコマンドラインに足すサブコマンド: `models`、`status`。コマンドラインのライブラリに依存しない |
| `.../ai/cli/konghelper` | そのサブコマンドの kong 用のアダプタ |

**用語**

- **用途 (use)**: `transcribe` や `translate` のような、目的の名前。models.yaml の `categories` にあり、機械のメモリに合うモデルを選びます。`hwnpu-chat list` で見られます
- **ファミリーとタグ**: `plamo3translate:2b` のように、モデルの系列とその中の 1 つ
- **役割 (role)**: モデルが何として動くか (`transcribe`、`translate`、`chat`、`speak`)。機能のパッケージの関数 (`speech.Transcribe` など) が決めます
- **置き場 (store)**: モデルのファイルとプログラムのある場所。既定はコマンドと同じキャッシュ (`~/.cache/hwnpu` など) です
- **対象 (target)**: プログラムがどの NPU 向けか。`coreai` (Apple)、`intel`、`amd-xdna2`

## 2. 最小の例

WAV を読んで文字にするプログラムです。

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/speech"
)

// 使うモデルは、パッケージの変数にしておく
var (
	stt = speech.Transcribe("transcribe")
	app = ai.NewApp("dictation", stt)
)

func main() {
	ctx := context.Background()

	// 準備: 足りないものを落とし、変換し、NPU 向けにコンパイルする (2 回目からは何もしない)
	err := app.Prepare(ctx, func(p ai.Progress) {
		fmt.Fprintln(os.Stderr, p.Use, p.Stage, p.Message)
	})
	if err != nil {
		log.Fatal(err)
	}

	// 実行
	rt, err := app.Open(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer rt.Close()
	t, err := stt.Get(ctx, rt) // *speech.Transcriber
	if err != nil {
		log.Fatal(err)
	}
	raw, err := os.ReadFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	pcm, err := speech.ReadWAV(raw, speech.TranscribeRate) // モノラル、16 kHz
	if err != nil {
		log.Fatal(err)
	}
	res, err := t.Transcribe(ctx, pcm, speech.TranscribeOptions{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Text)
}
```

複数のモデルをつなぐ動く例は [samples/interpreter](../samples/interpreter) にあります (マイクか録音の音声を、別の言語の音声に訳す)。サンプルは別のモジュール (`samples/go.mod`) で、マイクとカメラは [hwcamera](https://github.com/shibukawa/hwcamera) から取ります。hwcamera はライブラリ本体の依存には入りません ([6.5 節](#65-生の入力-マイクとカメラ))。

| サンプル | 中身 |
|---|---|
| `samples/interpreter` | 同時通訳: マイクか録音 → 文字起こし → 翻訳 → 読み上げ。音声のコマンド。ネットワークと呼び出し |
| `samples/parallax` | 擬似立体: 絵の深度を推定し、カメラで頭の位置を追い、その視点から見た絵を描く (Ebitengine、Kage のシェーダー)。窓の横で絵を層に分け、途中経過を窓に出す |
| `samples/layers` | レイヤー分割: 絵を深度で層に分け、各層が隠していた部分を画像のモデルで描き足して、Photoshop の文書 (PSD) に書く。2 つのモデルを交互に |

```
cd samples
go run ./interpreter models prepare
go run ./interpreter --mic -o english.wav      # マイクから。Ctrl-C か「終わり」で止める。「変換、スペイン語」で訳す先を変える
go run ./interpreter talk.wav -o english.wav   # 録音から
```

## 3. モデルを登録する

### 3.1 `Use` と `App`

```go
var (
	stt = speech.Transcribe("transcribe")          // ai.Use[*speech.Transcriber]
	tr  = text.Translate("plamo3translate:2b")     // ai.Use[*text.Translator]
	tts = speech.Speak("speech")                   // ai.Use[*speech.Speaker]

	app = ai.NewApp("interpreter", stt, tr, tts)
)
```

- 名前は、models.yaml のモデル (`whisper-large-v3-turbo`)、タグつきのファミリー (`plamo3translate:2b`)、ファミリー (`plamo3translate`)、用途 (`translate`) のどれか
- `Use[T]` は値です。`Get` が返す型 `T` を持つので、取り出すときに型を合わせる必要がありません
- モデルを並べる代わりに、ネットワーク (6.6 節、`app.From(source).To(node)...`) や呼び出し (6.7 節) を書くと、その節のモデルがアプリのモデルになります
- `app.Use(more...)` であとから足せます。同じ名前と役割の `Use` を 2 度登録すると、最初のものが残ります
- 役割に合わない名前 (`speech.Transcribe("chat")`) は、`Plan` と `Get` でエラーになります

### 3.2 機械のメモリごとの構成

用途の名前は、**1 つだけ動かすとき**にその機械で動く一番大きなモデルを選びます (`translate` は 16 GB の機械から PLaMo 3 Translate 8B、8.2 GB)。いくつも同時に動かすアプリでは、主になるモデルを大きく、補助のモデルを小さく速くしたいので、登録のときに構成を書きます。

```go
tr := text.Translate("translate").      //  8 GB: 定義の選ぶもの (0.8B)
	At(16, "plamo3translate:2b").       // 16 GB 以上: 2B
	At(32, "plamo3translate:8b")        // 32 GB 以上: 8B

helper := text.Chat("chat").As(8)       // どの機械でも、8 GB の機械として選ぶ
```

| 口 | 意味 |
|---|---|
| `use.At(gb, name)` | メモリが gb GB 以上の機械では、この用途を `name` にする。いくつも書け、機械のメモリ以下で一番大きい段が効く。どの段にも届かない機械は元の名前 |
| `use.As(gb)` | gb GB の機械として選ぶ: モデルの選び方と、そのモデルのプログラムの選び方 (models.yaml の `variants`) の両方 |
| `app.Memory(gb)` | アプリ全体を gb GB の機械として計画する (小さい機械向けに準備するとき)。`At` の段もそのメモリで決まる |

- `At` と `As` は新しい `Use` を返します。元の値は変わりません
- 利用者の側の調整も、コマンドと同じに効きます: `HWNPU_MEMORY` (機械のメモリを言い換える)、`hwnpu memory MODEL GB` (モデルごと)、`HWNPU_MODEL_ALLOWLIST` (使ってよいモデル)
- 選ばれたモデルと理由は `plan.Items[i].Model` と `.Why` に入ります。`plan.MemoryGB()` は同時に動かす合計、`plan.MachineGB()` は機械のメモリです

### 3.3 アプリ独自の定義

アプリは、自分の定義のファイル (models.yaml と同じ形式) を足せます。自分のモデル、自分の用途 (メモリごとの `picks`)、別名を書けます。

```go
app.Store = ai.Store{Models: modelstore.Options{Files: []string{"myapp-models.yaml"}}}
```

```yaml
version: 1
categories:
  interpreter-translate:
    picks:
      - {machine: 8, model: plamo-3-translate-0.8b}
      - {machine: 16, model: plamo-3-translate-2b}
```

3.2 節の `At` とは、同じことをコードに書くかファイルに書くかの違いです ([D5](#10-設計の論点))。

## 4. 準備する

### 4.1 計画

```go
plan, err := app.Plan(ctx)   // 何も落とさない
for _, it := range plan.Items {
	fmt.Println(it.Use, it.Model, it.Ready, it.Compiled, it.Download, it.MemoryGB, it.Why)
}
plan.DownloadBytes()   // 落とす量 (上限)
plan.MemoryGB()        // 同時に動かすときのメモリの合計
plan.MachineGB()       // 機械のメモリ
plan.Licenses()        // モデルごとのライセンス (ID、種類、URL、同意が要るか、同意済みか)
plan.Target()          // coreai、intel、amd-xdna2。"" は、モデルを動かせる NPU がない
```

- `Ready`: ファイルとプログラムが置き場にあり、ネットワークなしで読み込める
- `Compiled`: この機械の NPU で 1 回読み込んである (プログラムがコンパイル済み)
- `Download` は models.yaml の `bytes` で、上限です。取得元に変換済みのものがあれば、チェックポイントは落としません

### 4.2 準備は 2 段

```go
plan.NoCompile = true                       // ダウンロードと変換だけ
err = plan.Run(ctx, func(p ai.Progress) {   // p.Stage: prepare、compile、done
	...
})
```

| 段 | 中身 | NPU |
|---|---|---|
| ダウンロードと変換 | 取得元 (手元のキャッシュ、`HWNPU_MODEL_SOURCE*`、同梱、Hugging Face) から取り、プログラムに変換する。取得元に変換済みのものがあれば取るだけ | 要らない |
| コンパイル | 各モデルを 1 回読み込む。Apple の Neural Engine のコンパイルは初回に数秒から数分かかり (Gemma 4 で 100 秒)、結果はその機械のもの | 要る |

- `plan.Run` は途中で切れても、もう一度呼べば続きから進みます
- `Progress.Message` は、ダウンロードや変換のログの 1 行です。バイト数の入った値はまだありません ([D15](#10-設計の論点))
- `app.Prepare(ctx, fn)` は `Plan` と `Run` を続けて呼びます
- 作ったものは、書き戻し先の取得元 (`?writeback=true`) にも置かれます。`Run` はそれが終わるのを待ってから返ります

### 4.3 ライセンス

同意の要るライセンス (`License.Accept`) のモデルがあるとき、`plan.Run` は `*ai.LicenseError` を返します。アプリは `plan.Licenses()` を見せて同意を取り、`plan.AcceptLicenses = true` にしてから `Run` を呼びます。同意は利用者の設定のフォルダ (`licenses.json`) に残り、コマンドと共有されます。

### 4.4 置き場と、準備済みのものを配る

```go
app.Store = ai.Store{
	Models:   modelstore.Options{Root: cacheDir, Sources: []string{"https://models.example.com/hwnpu"}},
	Prepared: []string{filepath.Join(resources, "prepared")},
}
```

- `Store` のゼロ値はコマンドと同じ置き場です。コマンドが準備したものをアプリが使い、アプリが準備したものをコマンドが使います
- `Store.Models` は [`modelstore.Options`](../USAGE_LIBRARY.ja.md) (キャッシュの場所、取得元、同梱のモデル、大きさの上限、アプリの定義のファイル)
- 準備の結果は、キャッシュの根の `store.json` に書かれます (NPU の対象ごと、`役割:名前` ごと)
- `Store.Prepared` は、準備したキャッシュから `store.json` と名指すファイルを写したフォルダです。読むだけで、その場で使います。ここにある用途は、準備なしで読み込めます
- 置き場の設定 (キャッシュの場所、取得元、定義) は、今はプロセスに 1 つです。1 つのプロセスで違う `Store` の `App` を 2 つ同時には使えません ([D8](#10-設計の論点))

## 5. 実行する

```go
rt, err := app.Open(ctx)            // ai.WithLog(w)、ai.WithAutoPrepare(fn)、ai.WithAcceptedLicenses()
defer rt.Close()

t, err := stt.Get(ctx, rt)          // 最初の呼び出しで読み込む (1 秒未満から数十秒)。2 回目からは同じものを返す
...
err = stt.Release(rt)               // 下ろしてメモリを空ける。次の Get でまた読み込む
rt.Loaded()                         // 今持っているモデル: 用途、モデル、メモリ、読み込みにかかった時間
```

- **ネットワークにも変換にも触りません。** 準備されていない用途は `ai.ErrNotPrepared` です。`ai.WithAutoPrepare(fn)` を付けると、最初の `Get` で準備します (コマンドと同じ動き。ギガバイトのダウンロードになることがあります)
- **置き場の記録が今の選び方と違えば、準備されていない扱いです。** 別のモデルで準備してある (アプリの構成が変わった、機械のメモリの設定が変わった) とき、前のものを黙って読み込みません
- **読み込みは 1 つずつ**進みます。モデルは、それぞれ 1 回に 1 つの要求を通します (ほかは待ちます)。別のモデルどうしは同時に呼べます
- **順に使うアプリ**は `Release` で下ろします。画像の生成のあとに深度、のような使い方で、同時に載るメモリが一番大きいモデルの分で済みます。自動で下ろす仕組みはまだありません ([D9](#10-設計の論点))。画像の生成が中で行う段の入れ替えとは別の層です ([9.5 節](#95-release-と画像の生成の段の入れ替え))
- **同時に呼べますが、NPU は 1 つの待ち行列です。** 実時間で回り続けるかどうかは、各段の時間の合計で決まります ([9 節](#9-制約とルール))
- 各モデルの `OnNPU()` は、プログラムが NPU に載ったことを確かめられたかを返します (`Speaker` にはまだありません)

### 5.1 エラー

| エラー | いつ | どうする |
|---|---|---|
| `ai.ErrNotPrepared` | 用途が準備されていない、別のモデルで準備されている、ファイルが消えた | `plan.Run`、または `WithAutoPrepare` |
| `ai.ErrBundleTarget` | `Store.Prepared` のものが別の NPU 向けだけ、または読めない形式 | その NPU 向けの準備済みのものを配る、または手元で準備する |
| `*ai.LicenseError` | 同意の要るライセンスに同意していない | 同意を取り、`plan.AcceptLicenses` |
| `flow.ErrEmpty` | 塊が 1 つも来ないまま流れが終わった (`Held.Wait`) | — |
| 登録していない `Use` の `Get` | `app` に登録していない | `ai.NewApp` か `app.Use` に足す |

`errors.Is` と `errors.As` で調べます。

## 6. モデルをつなぐ (`ai/flow`)

### 6.1 塊と流れ

```go
type Chunk[T any] struct {
	Data       T
	Start, Len time.Duration // 流れの始まりからの位置と長さ
	Final      bool          // 区切りの最後の塊
}

type Stream[T any] = iter.Seq2[Chunk[T], error]

for c, err := range stream {
	if err != nil { return err }
	use(c.Data)
}
```

- 流れは Go の反復子です。`range` で読み、`break`、エラー、ctx の取り消しで止まります。エラーのあとは何も来ません
- **段は「流れを受けて流れを返す関数」**です。前の段の結果を次の段に渡すだけでつながります。読む側が読まなければ、前の段は進みません (`flow.Async` を挟まない限り)
- **時刻は流れ自身のもの**です。音声は標本の数から計算するので、マイクの遅れやネットワークの揺れでずれません。壁時計を見るのは `Metronome`、`Pace`、`Live` だけです
- **`Final`** は区切りの終わりです: 発話の最後の音の塊、確定した文、でき上がった絵。テキストの流れでは、`Final` の前の塊は「ここまでの仮の文」で、次の塊が前のものを置き換えます (話している途中の字幕、生成中の答え)
- 音声は `[]float32` (モノラル、−1〜1)、テキストは `flow.Text{Text, Lang}` です。音声の標本化周波数は塊に入っていません: 文字起こしと VAD は 16 kHz (`speech.TranscribeRate`)、読み上げの出力は `Speaker.Rate()` (24 kHz) です ([D11](#10-設計の論点))

### 6.2 3 つのつなぎ方

次の段をいつ始められるかで、3 つあります。

**(1) 前が全部終わってから。** 次の段が結果の全体を要るとき (絵ができてから深度)。

```go
img, err := gen.Generate(ctx, prompt)      // 一括の呼び出し: 値を 1 つ渡し、1 つ受け取る
if err := genUse.Release(rt); err != nil { ... }   // 大きなモデルは下ろしてから
est, err := depthUse.Get(ctx, rt)
d, err := est.Estimate(ctx, img)
```

流れを返す段の結果を全部待つなら `flow.Last(stream)`、全体のある値を流れとして渡すなら `flow.Of(value, length)` です。(`gen`、`est` は、まだないパッケージの想像図です。)

**(2) できたものから。** 前の段が次を作っている間に、できた塊で始めるとき。

```go
utterances := speech.NewVAD(speech.VADOptions{}).Detect(ctx, mic)   // 発話のところだけ
texts := stt.TranscribeStream(ctx, utterances, speech.TranscribeOptions{})
en := tr.TranslateStream(ctx, texts, text.TranslateOptions{To: "en"})
audio := tts.SpeakStream(ctx, flow.Async(ctx, en, 4), speech.SpeakOptions{})
for c, err := range audio { ... }
```

| 道具 | 働き | 塊を落とすか |
|---|---|---|
| `flow.Async(ctx, s, n)` | `s` を別の goroutine で読み、n 個まで先に進める。段が重なって動く (1 文目を読んでいる間に 2 文目を訳す) | 落とさない。n 個たまったら前の段が待つ |
| `flow.Latest(ctx, s)` | `s` を別の goroutine で読み、読む側には最新の塊だけを渡す。読む側が遅い間に来た塊は捨てる (カメラのフレームを、1 フレームより時間のかかる検出器へ) | 落とす (最後の塊とエラーは落とさない) |
| `flow.Hold(ctx, s)` | `s` を別の goroutine で読み、最新の塊を持っておく。`Latest()`、`Wait(ctx)`、`Stop()` | 上書きする |
| `flow.Tee(s, n)` | 1 つの流れを n 個の読み手に (字幕と翻訳に同じ文字起こしを) | 落とさない。遅い読み手の分はメモリで待つ |

**(3) 拍に合わせて。** 決まった時刻に出力が要るとき (再生に間に合わせる音、画面に出すフレーム)。

```go
// 1 秒に 60 回描く。頭の位置は、検出器が出した最新のもの
head := flow.Hold(ctx, faces.DetectStream(ctx, flow.Latest(ctx, camera)))
defer head.Stop()
for tick, err := range (flow.Metronome{Period: time.Second / 60, Skip: true}).Ticks(ctx) {
	if err != nil { break }
	pos, _ := head.Latest()
	draw(mesh, pos.Data, tick.Data.At)
}

// 2 秒ごとに音の塊を出す。拍の 500 ms 前に tick を受け取り、その間に作る
for tick, err := range (flow.Metronome{Period: 2 * time.Second, Lead: 500 * time.Millisecond}).Ticks(ctx) {
	if err != nil { break }
	chunk := generate(tick.Data.N)          // tick.Data.Late > 0 なら、間に合っていない
	player.At(tick.Data.At, chunk)
}
```

(`faces`、`camera`、`draw`、`generate`、`player` は想像図です。メトロノームと `Hold`、`Latest` は今あります。)

| 道具 | 働き |
|---|---|
| `flow.Metronome{Period, Lead, Skip}.Ticks(ctx)` | 拍ごとの `Tick{N, At, Late, Skipped}` の流れ。最初の tick はすぐ、その拍は `Lead` 後。読む側が遅れたとき: `Skip` なしは全部の拍を遅れて渡す (穴を開けられない出力)、`Skip` ありは過ぎた拍を捨てて今の拍を渡す (今を出す出力) |
| `flow.BPM(bpm)` | 1 分に bpm 拍の周期 |
| `flow.Pace(ctx, s, ahead)` | 塊を、その `Start` の時刻に渡す (最初の塊はすぐ)。`ahead` は先に渡してよい長さ (再生側のバッファ)。遅れて来た塊はすぐ渡し、落とさない |
| `flow.Live(ctx, s)` | 録音の流れを、マイクのように実時間で渡す (塊の終わりの時刻に)。ファイルでマイクやカメラの代わりをするとき |

### 6.3 そのほかの道具

| 道具 | 働き |
|---|---|
| `flow.Of(v, len)`、`flow.From(vs...)` | 値から流れを作る |
| `flow.Collect(s)`、`flow.Last(s)` | 流れを最後まで読む: 全部の塊、最後の塊の中身 |
| `flow.Map(s, fn)` | 関数を段にする (時刻と `Final` はそのまま) |
| `flow.Finals(s)` | `Final` の塊だけ (確定した文だけ) |
| `flow.Rechunk(s, rate, n)` | 音声を n 標本ずつの塊に切り直す |
| `flow.Sentences(s)`、`flow.SplitSentences(text)` | テキストを文で区切る (日本語の句点、英語の文末) |
| `flow.Samples(n, rate)` | n 標本の長さ |
| `speech.Chunks(pcm, rate, n)` | 音声の配列を、n 標本ずつの塊の流れに |

### 6.4 決まり

- `Async`、`Latest`、`Hold` に渡す流れは、ctx が取り消されたら止まるものにしてください。goroutine は、返した流れが終わる (読み切る、`break`、取り消し) 前に終わります。元の流れが次の塊を待っている間は、その塊が来るまで終わりません
- 自分で段を書くときは、エラーの塊を受け取ったらそのまま渡して終わり、`yield` が false を返したらすぐ戻ります

```go
func Upper(s flow.Stream[flow.Text]) flow.Stream[flow.Text] {
	return func(yield func(flow.Chunk[flow.Text], error) bool) {
		for c, err := range s {
			if err != nil {
				yield(c, err)
				return
			}
			c.Data.Text = strings.ToUpper(c.Data.Text)
			if !yield(c, nil) {
				return
			}
		}
	}
}
```

- 取り消しは ctx です。割り込み (相手が話し始めたら読み上げを止める) の口はまだありません ([D13](#10-設計の論点))

### 6.5 生の入力: マイクとカメラ

`ai` のパッケージ群は、マイクもカメラもスピーカーも持ちません。入力は `flow.Stream` であれば何でもよく、ファイルから作っても、デバイスから作っても、後ろの段は同じです。サンプルは [hwcamera](https://github.com/shibukawa/hwcamera) (cgo なしで OS のカメラと音声入力の API を呼ぶ) を使い、流れに直す小さなコードを `samples/internal/capture` に置いています。

```go
// samples/internal/capture: マイク (hwcamera/audio) を流れに
mic := capture.Mic(ctx, capture.MicOptions{Rate: speech.TranscribeRate}) // flow.Stream[[]float32]

// 録音から。後ろの段は同じ
mic := speech.Chunks(pcm, speech.TranscribeRate, speech.TranscribeRate/10)
mic = flow.Live(ctx, mic) // マイクの速さで流すなら
```

**生の入力は待ってくれません。** 流れは読む側が引く形なので、後ろの段が忙しい間は誰も入力を読まず、その間の音やフレームはデバイスの側で失われます。入力のすぐ後ろで、どうするかを決めます。

| 入力 | すること |
|---|---|
| 音 (全部聞く必要がある) | 別の goroutine で読んでためる: `flow.Async(ctx, mic, n)`。`capture.Mic` はこれを中で行い、それでも失われたら `MicOptions.Lost` で知らせる。ためる上限は `MicOptions.Backlog` (既定で 1 分ぶん)。**遅延ではありません**: 読んだ音はすぐ次へ渡し、後ろの段が追いついている間は何もたまりません |
| 映像 (古いものは要らない) | 最新だけを渡す: `flow.Latest(ctx, camera)` |

- 実機で確かめたこと: ためずにつなぐと、1 つ目の発話を書き起こして訳している数秒の間の音が失われ、2 つ目の発話の頭が欠け、3 つ目が消えました。ためると 3 つとも通りました
- `flow.Live` (録音をマイクの速さで流す) は、読む側が遅れても待つだけで失いません。この違いは録音では再現できません
- マイクを止めるのと、後ろの段を止めるのは別です。`capture.Mic` は ctx が取り消されると (エラーなしで) 終わり、後ろの段は聞いたぶんを最後まで処理します。そのため後ろの段には別の ctx を渡します (`samples/interpreter` は、1 度目の Ctrl-C でマイクを止め、2 度目で終了します)
- カメラ (`capture.Camera`): hwcamera のフレームは、ドライバのメモリにあるうちは `Release` が要るので、Go のメモリの画像に写してから流れに載せます (`flow.Stream[image.Image]`)。後ろの段へは `flow.Newest()` の辺でつなぎます。実物のカメラでは、まだ確かめていません
- 出力 (スピーカー、画面) も同じ考えで、ライブラリは持ちません ([D19](#10-設計の論点))

### 6.6 ネットワークを宣言する

ここまでの書き方は、流れを関数でつなぎます。つなぎ方はコードの中にあり、ライブラリは知りません。**ネットワーク**は、同じつなぎ方を構造として書くものです。節 (ノード) を `To` でつなぎ、アプリに渡すと、ライブラリがモデルを読み込んで走らせます。

```go
var (
	stt = speech.Transcribe("transcribe")
	tr  = text.Translate("translate").At(16, "plamo3translate:2b").At(32, "plamo3translate:8b")
	tts = speech.Speak("speech")
)

mic   := flow.Source[[]float32]("mic")                             // 入力: 流れはあとで渡す
vad   := speech.VADNode(speech.VADOptions{})                       // モデルのない節
hear  := speech.TranscribeNode(stt, speech.TranscribeOptions{})    // モデルの節
trans := text.TranslateNode(tr, text.TranslateOptions{To: "en"})
say   := speech.SpeakNode(tts, speech.SpeakOptions{})
out   := flow.Sink[[]float32]("out").Do(play)                      // 出力

// プログラムの節: 通るものを見て、加工し、流すかどうかを決める
heard := flow.Tap("heard", func(c flow.Chunk[flow.Text]) { log.Println(c.Data.Text) })
commands := flow.Each("commands", func(c flow.Chunk[flow.Text], emit func(flow.Chunk[flow.Text])) error {
	switch c.Data.Text {
	case "終わり":
		return flow.Stop                                           // ここで終える: 前の節は止まり、後ろは残りを片付ける
	case "変換スペイン語":
		trans.Update(func(o *text.TranslateOptions) { o.To = "es" }) // ほかの節の設定を変える。後ろには流さない
		return nil
	}
	emit(c)                                                        // そのまま流す
	return nil
})

app := ai.NewApp("interpreter")     // モデルは、ネットワークの節から引かれる
app.From(mic).To(vad).To(hear).To(heard).To(commands).To(trans).To(say, flow.Keep(4)).To(out)

mic.From(capture.Mic(ctx, capture.MicOptions{}))   // 実行のときに入力の流れを渡す
err := app.Run(ctx)                                 // モデルを読み込んでから、入力を読み始める
```

**節の種類**

| 節 | 作り方 | 中身 |
|---|---|---|
| 入力 | `flow.Source[T](name)`、`.From(stream)` | 流れ。宣言のときは名前だけで、実行の前に流れを渡す |
| モデル | `speech.TranscribeNode(use, opts)`、`speech.SpeakNode`、`text.TranslateNode` | そのモデルの流れの形 (`...Stream`)。設定を持ち、動いている間も変えられる (`node.Update(func(o *Options))`) |
| モデル (自分で) | `ai.Node(use, fn)`、`ai.NewModelNode(use, opts, fn)` | `fn(ctx, model, in)` が流れを返す。機能のパッケージにない使い方をするとき |
| プログラム | `flow.Each(name, fn, opts...)` | 塊が来るたびに `fn(c, emit)` が呼ばれる。**後ろをいつ動かすかは、この関数が決める**: `emit` を呼んだぶんだけ後ろに流れる (0 個でも、いくつでも) |
| プログラム (見るだけ) | `flow.Tap(name, fn)` | 全部そのまま流し、`fn` に見せる (ログ、画面の字幕、計数) |
| プログラム (振り分け) | `flow.Route(name, fn, opts...)` | `fn(c, emit)` が `emit(i, c)` で、i 番目の出力 (`To` を呼んだ順、0 から) にだけ流す |
| プログラム (合流) | `flow.Both(name, fn)`、`flow.With(name, fn)`、`flow.Join2(name, fn)` | 入力が 2 つ (`node.In1`、`node.In2`)。下の「合流」 |
| プログラム (流れごと) | `flow.Func(name, fn)` | `fn(ctx, in)` が流れを返す。6.4 節の「自分で書く段」をそのまま節にする |
| モデルなしの VAD | `speech.VADNode(opts)` | 発話の検出 |
| 出力 | `flow.Sink[T](name)`、`.Do(fn)` | 塊ごとに `fn` |

**つなぎ方**

- `a.To(b)` は「a の出力を b に渡す」で、**続きを書ける場所を返します**。`app.From(mic).To(vad).To(hear)` と続けて書けます
- **型の確かめ**: 節の `To` (`hear.To(trans)`) は、型が合わないとコンパイルできません。続けて書いた 2 つ目からの `To` は、呼ばれたとき (ネットワークを組み立てるとき) に確かめて、合わなければ panic します (「vad gives []float32 and translate takes flow.Text」)。Go のメソッドは型を続きに運べないためです。どちらも、何かが動く前に分かります
- **分岐**: 同じ場所から `To` を 2 度呼ぶと、両方が全部の塊を受け取ります。遅い読み手がほかを待たせません

```go
texts := app.From(mic).To(vad).To(hear)
texts.To(captions)                                  // 字幕へ
texts.To(trans).To(say, flow.Keep(4)).To(out)       // 翻訳して読み上げへ
```

- 辺ごとの渡し方を、`To` の 2 つ目の引数で書きます

| 渡し方 | 意味 | 使うところ |
|---|---|---|
| (なし) | 後ろの節が読むときに渡す。後ろが忙しい間、前は待つ | ふつうの辺 |
| `flow.Keep(n)` | 別の goroutine で読み、n 個までためる。前は待たずに進む。落とさない | 読み上げの前 (読んでいる間に次を訳す) |
| `flow.Newest()` | 最新の塊だけを渡す。後ろが忙しい間に来たものは捨てる | カメラのフレーム → 検出 |

**後ろをいつ動かすかを、プログラムの節が決める**

「前が全部終わってから次へ」「区切りまでたまってから流す」は、辺の種類ではなく、プログラムの節の中で決めます。来たものをためておき、流してよくなったら `emit` を呼びます。

```go
// 文の区切りまでためてから訳す。区切りが来ないまましばらく止まったら、たまったぶんを流す
var kept string
flush := func(emit func(flow.Chunk[flow.Text])) error {
	if kept != "" {
		emit(flow.Chunk[flow.Text]{Data: flow.Text{Text: kept}, Final: true})
		kept = ""
	}
	return nil
}
sentences := flow.Each("sentences", func(c flow.Chunk[flow.Text], emit func(flow.Chunk[flow.Text])) error {
	kept += c.Data.Text
	if i := strings.LastIndexAny(kept, "。！？"); i >= 0 {          // 区切りの前までを先に流し、後ろは残す
		i += len("。")
		emit(flow.Chunk[flow.Text]{Data: flow.Text{Text: kept[:i]}, Final: true})
		kept = kept[i:]
	}
	return nil
},
	flow.WhenIdle(800*time.Millisecond, flush),   // 800 ms 何も来なければ、待ち続けずに流す
	flow.AtEnd(flush))                            // 入力が終わったら、残りを流す
```

| オプション | いつ呼ばれるか |
|---|---|
| `flow.WhenIdle(wait, fn)` | 最後の塊から `wait` のあいだ何も来なかったとき (止まるたびに 1 回)。ためたものを抱えたまま待ち続けないため |
| `flow.AtEnd(fn)` | 入力が終わったとき |

**振り分けと合流**

```go
// 振り分け: コマンドは 1 番目の出力へ、訳す文は 2 番目へ
sort := flow.Route("sort", func(c flow.Chunk[flow.Text], emit func(to int, c flow.Chunk[flow.Text])) error {
	if isCommand(c.Data.Text) {
		emit(0, c)
	} else {
		emit(1, c)
	}
	return nil
})
hear.To(sort)
sort.To(commandHandler)   // 出力 0
sort.To(trans)            // 出力 1

// 合流: 1 つ目の入力の拍で動き、2 つ目の入力の最新を持っていく (描画と、頭の位置)
render := flow.With("render", func(tick flow.Chunk[flow.Tick], head flow.Chunk[Pose], ok bool, emit func(flow.Chunk[image.Image])) error {
	emit(flow.Chunk[image.Image]{Data: draw(mesh, head.Data)})
	return nil
})
clock.To(render.In1)
faces.To(render.In2, flow.Newest())
render.To(screen)
```

| 合流の節 | 動き方 |
|---|---|
| `flow.Both(name, fn)` | 両方の入力から 1 つずつ来るのを待って `fn(a, b, emit)`。1 つ目どうし、2 つ目どうしの組。どちらかが終われば終わる |
| `flow.With(name, fn)` | 1 つ目の入力が来るたびに `fn(a, b, ok, emit)`。`b` は 2 つ目の入力の最新 (まだ来ていなければ `ok` が false)。2 つ目は来るままに読み、待たない |
| `flow.Join2(name, fn)` | `fn(ctx, a, b)` が 2 本の流れを受けて流れを返す。読み方を全部自分で決める |

- 入力が 2 つの節は、両方に何かをつながないと走りません (`Run` がエラーを返します)
- 入力が 2 つあるネットワークは、入力 (`flow.Source`) を両方アプリに渡します (`app.From(clock)`、`app.From(camera)`)
- 入力が 3 つ以上の節、名前のついた出力をいくつも持つ節は、まだありません

**走らせ方**

- `app.From(source)` は、入力をアプリに渡し、そこから続きを書ける場所を返します。節どうしの `To` は、実行の前ならいつでも足せます
- **アプリのモデルは、ネットワークの節から引かれます。** `ai.NewApp(name, uses...)` に並べなくても、`app.Plan` と `APP models` が見つけます。`APP models` はネットワークも出します
- `app.Run(ctx)` は、`Open`、モデルの読み込み、ネットワークの実行、`Close` を続けて行います。開いた `Runtime` の上で走らせるなら `rt.Run(ctx)`
- **モデルは、入力を読み始める前に全部読み込みます。** 読み込みの数秒の間に、マイクの音が失われません
- 終わり方: 入力が全部終わって最後の塊まで通ると `nil`、節が失敗するとその節の名前のついたエラー (`*flow.NodeError`)、ctx の取り消しでそのエラー
- **節が中から終える**: `flow.Each` の関数が `flow.Stop` を返すと、その節はそこで終わります。前の節は止まり (マイクは閉じます)、後ろの節は受け取ったぶんを最後まで処理して、`Run` は `nil` を返します
- 同じモデルを 2 か所で使うときは、節を 2 つ作ります (`speech.TranscribeNode(stt, optsL)` と `speech.TranscribeNode(stt, optsR)`)。モデルは 1 つで、呼び出しは順に通ります (9.3 節)

**流れの関数との関係**: 捨てていません。節の中身は流れの関数で、辺の渡し方は `flow.Async` と `flow.Latest` です。ネットワークは、それを組み立てて走らせる層です。

### 6.7 要求ごとに 1 回動かす (`Call`)

ネットワークは、入力が来る限り動き続ける構成です。もう 1 つの構成が**呼び出し (`Call`)** で、要求 1 つにつき 1 回、値を 1 つ受け取り、節を順に通して、値を 1 つ返します。サーバーの応答、一括の処理の 1 段、「絵を作ってから深度を出す」がこの形です。

```go
// 宣言: 文 → 翻訳 → 読み上げ。節はネットワークと同じもの (ここ用に作る)
say := ai.NewCall[flow.Text, []float32](app, "say").
	Then(text.TranslateNode(tr, text.TranslateOptions{To: "en"})).
	Then(speech.SpeakNode(tts, speech.SpeakOptions{}))

// 要求ごとに
rt, err := app.Open(ctx)
pcm, err := say.Do(ctx, rt, flow.Text{Text: "こんにちは"})
```

| | ネットワーク (`From` / `To`) | 呼び出し (`NewCall` / `Then`) |
|---|---|---|
| 動き方 | 入力が来る限り動き続ける | 要求 1 つにつき 1 回 |
| 入力 | `flow.Source` の流れ | `Do(ctx, rt, in)` の値 1 つ |
| 出力 | `flow.Sink` | `Do` の戻り値 (最後の塊)。`All` は出てきた全部 |
| 走らせる | `app.Run(ctx)` | `call.Do(ctx, rt, in)` を要求のたびに |
| モデルの読み込み | 走らせる前に全部 | 初めて使うとき |

- `Then` の型は、`To` を続けて書いたときと同じに、組み立てのときに確かめます。最後の節の出力が `Call` の出力の型と違えば、最初の `Do` がエラーを返します
- 呼び出しの節も、プログラムの節 (`Each`、`Tap`、`Route`) を使えます。アプリのモデルは、呼び出しの節からも引かれます
- **節は、1 つのネットワークか 1 つの呼び出しのものです。** ネットワークと呼び出しの両方で同じモデルを使うなら、節をそれぞれに作ります。モデルは 1 つで、1 回だけ読み込まれます
- 呼び出しは同時にいくつも走らせられます。それぞれが自分の流れを持ちます (モデルの節は 1 回に 1 つを通します)
- `Use.Get` でモデルを取り出して自分で呼ぶ書き方 (5 節) も、そのまま使えます。`Call` は、同じことを構造として書き、アプリのモデルと `APP models` に載せるためのものです
- `samples/interpreter say TEXT` がこの形です

### 6.8 例: 擬似立体 (`samples/parallax`)

3 つのつなぎ方が 1 つのアプリに出てくる例です。絵の深度は 1 回だけ要り、頭の位置は来るたびに要り、描画は自分の拍で動きます。下はサンプルの抜粋で、`smooth` などの関数の中身は省いています。

```go
depth := vision.Depth("depth")
artist := imagegen.Edit("image") // 参照から描く
faces := vision.Faces("face-detect")
app := ai.NewApp("parallax")

// 呼び出し 2 つ: 絵 → 深度、塗った絵と文 → 描き足した絵。交互に動かし、終わったらモデルを下ろす
split := layering.Register(app, depth, artist)

// ネットワーク: カメラ → 顔 (最新のフレームだけ) → 頭の位置 → ならす → 視点
camera := flow.Source[image.Image]("camera")
steady := flow.Each("steady", smooth, flow.WhenIdle(1500*time.Millisecond, recentre)) // 顔が見えなくなったら中央へ戻す
view := flow.Sink[View]("view").Do(keepNewest)
app.From(camera).To(vision.FacesNode(faces), flow.Newest()).To(vision.HeadNode(vision.HeadOptions{})).To(steady).To(view)

rt, _ := app.Open(ctx)
near, _, _ := split.Near(ctx, rt, picture)      // 前が全部終わってから: 深度を 1 回、モデルは下ろす

camera.From(capture.Camera(ctx, capture.CameraOptions{}))
go rt.Run(ctx)                                  // できたものから (検出は最新のフレームだけ)
go layering.Split(ctx, picture, near, split.Models(rt), opts) // 窓の横で層に分ける。段ごとに opts.Progress へ
runWindow(picture, near, latestView, progress)  // 拍に合わせて: 窓は 1 秒に 60 回、最後に届いた視点と層で描く
```

- **描画の拍は Ebitengine のもの**です。窓の `Update` / `Draw` が 1 秒に 60 回動き、ネットワークの最後の節が置いた最新の視点を読みます (`flow.Hold` と同じ考え)。検出が 1 秒に 15 回でも、窓は検出を待ちません
- **視差**: 絵の各点を、近さに比例して動かします (奥は少し、手前は大きく)。画面の画素ごとに、目からの視線を手前の層から奥へたどり、絵がその層以上に近い最初の点を描きます。手前のものが奥を隠します。Kage のシェーダー (`parallax.kage`) と、同じ計算を Go で書いたもの (`Scene.Render`) があり、8 つの視点で画素を比べて一致を確かめています
- 近さは深度の逆数なので、目のすぐ前に草があると、その奥の橋も家も山もほぼ一緒に動きます (目で見たとおりですが、奥行きは伝わりません)。そこで奥の層を引き離す曲線をかけます (`Layers`、`--far`、既定 30: 近さ n を log(1 + 30n) / log(31) に。0 でかけない)。向かい合う視点の間の動きは、山 0、森 0→4、家 1→7、橋 3→17、手前の杭 43→57 画素 (1024 画素の絵) になりました
- 深度の輪郭は実際の輪郭より甘いので、近いものを数画素ぶん広げてから使います (`Spread`)。広げないと、輪郭が裂けて筋になります
- 絵の見えていなかった部分 (手前のものが動いて現れる背景) は、絵にないので引き伸ばされます。動きを大きくしすぎると目立ちます (`--strength`)
- **層に分けて描く**: 窓は、深度が出たらすぐ (1〜2 秒) 絵を 1 枚の深度で見せ、その横で絵を層に分けていきます (`--layers`、既定 3。[6.9 節](#69-例-深度でレイヤーに分ける-sampleslayers) と同じ処理で、`samples/internal/layering` を両方のサンプルが使います)。層が 1 つ切れるたびに窓の絵がその層つきのものに入れ替わり、奥の層から 1 層ずつ、その層の近さで描いて重ねます。層の中は今までどおり細かい起伏で動き、手前の層がずれて見えた所には、引き伸ばしの代わりに、描き足された裏の絵がその絵の近さで出ます
- **途中経過を見せる**: 分ける処理は 1 層に 1 分ほどかかるので、窓の下に今の段 (「層 3 / 3: 裏を描いている、40 %」)、全体の進み具合のバー、隅に今の段の絵 (層の形を 1 色で塗った絵、次にモデルが描いた絵) を出します。処理は別の goroutine で呼び出しを回し、段ごとに `layering.Options.Progress` で受けたものを窓が毎フレーム読むだけです (頭の追跡と描画は止まりません)。`--save-layers DIR` で層を書き出し、`parallax DIR` でモデルなしにもう一度見られます。`--layers 1` で層に分けません。画像のモデルが準備されていない機械や、参照からの生成がまだ動かない NPU では、1 枚の深度のままになります
- 頭の代わりにマウスで動かす (`--head pointer`)、勝手に回す (`--head sweep`)、窓なしで GIF に書く (`parallax render`)、カメラなしで絵をフレームとして流す (`parallax track --frames`) ことができます
- 窓は Ebitengine で、Linux では X11 と OpenGL の開発用のパッケージが要ります。ない機械では `-tags headless` で窓以外をビルドできます

### 6.9 例: 深度でレイヤーに分ける (`samples/layers`)

絵を深度で層に分け、Photoshop の文書 (PSD) に書きます。各層が隠していた部分は画像のモデルが描き足すので、層を動かしたり消したりしても、その裏に絵があります。2 つのモデルを**交互に**使う例です。分ける処理そのものは `samples/internal/layering` にあり (モデルは関数として受け取ります)、このサンプルと擬似立体のサンプルが使います。下は、その中でしていることの抜粋です。

```go
var (
	depth  = vision.Depth("depth")
	artist = imagegen.Edit("image") // AMD は Generate でも同じ。Apple では編集のプログラムも準備する
)

app := ai.NewApp("layers")
depthOf := ai.NewCall[image.Image, *vision.DepthMap](app, "depth of the picture").Then(vision.DepthNode(depth))
drawing := imagegen.GenerateNode(artist, imagegen.Options{})
behind := ai.NewCall[imagegen.Request, image.Image](app, "what a layer hid").Then(drawing)

// 手前の層から順に (抜粋)
d, _ := depthOf.Do(ctx, rt, picture)
depth.Release(rt)                         // 深度のモデルを下ろしてから
out, _ := behind.Do(ctx, rt, imagegen.Request{
	Prompt:     "Remove the green shape from image 1. Where it was, draw the scenery that is behind it, ...",
	References: []image.Image{painted},   // 手前の層を 1 色で塗った絵
})
artist.Release(rt)                        // 画像のモデルを下ろしてから、次の深度へ
```

```
絵 → 深度 → 一番手前の層を切り出す → その形を 1 色で塗る → 裏を描かせる
 ↑                                                              |
 └──────────────── その層のない絵 ←──────────────────────────────┘
```

- **交互に使う**: 深度 (1.4 GB) と画像の生成 (3.9 GB) は同時に要らないので、使うたびに `Release` で下ろします ([9.5 節](#95-release-と画像の生成の段の入れ替え))。読み込み直しは深度が 0.1 秒、画像のモデルが 4 秒ほどで、1 枚描く 1 分に比べて小さい
- **切る場所**: 近さ (深度の逆数に、奥を引き離す曲線をかけたもの) の分布から、層の中の近さが一番そろう切り方を求めます (`layers plan` が、層ごとに色をつけた絵と切る値を出します。`--cuts` で指定もできます)
- **裏を描かせる**: 編集のモデルにマスクの入力はないので、消したい部分を絵にない色 1 色で塗り、「その色の形を取り除いて、裏の景色を描く」と頼みます。描いた絵は層のあった所だけ使い、ほかは元の絵のままにします (層を重ね直すとほぼ元の絵に戻ります)。深度が拾い損ねた層の細い部分 (草の先、髪の束) は奥の層に残り、層を動かすと取り残されます。`--clean PIXELS` で層の縁の少し先まで描いた絵を使うとそれは消えますが、描いた絵が周りときれいにつながらない所では、層の周りに縁取りのように見えます
- **描き損じを見分ける**: モデルは、塗った形をそのまま (塗った色や、暗い影として) 残すことがあります。そのときはマスクの縁の全周に色の段差が出るので、縁の内と外の色の差 (`seam`) で見分け、別の種で描き直します (`--tries`、既定 3 回)。どの回もだめなら、その層は切り出さずに絵に残し、そこで分けるのをやめます (層が 1 つ減ります。変な絵を裏に入れるよりよい)。頼み方でも変わり、「この領域を背景で置き換える」より「この形を取り除く」のほうが残りにくい。写真では、大きく塗った形の所に人の群れを描いたことがあり、「人や動物を足さない」の一文を入れてあります
- **次の層**: 描き足した絵の深度をもう一度推定し、両方に写っている部分で最初の深度の目盛りに合わせてから、次の層を切ります (深度は絵ごとの目盛りなので、そのままでは比べられません)
- **層ごとの深度**: 色だけでなく、近さも層ごとに持ちます (`--png DIR` の `near-N.png`。16 ビットの灰色、全部の層で同じ目盛り)。描き足した部分の近さは、描き足した絵から推定したものです。層の近さは、層の縁の外へも縁の値のまま続けてあります (縁で奥の層の値に落ちると、描くときに縁が引き伸ばされるため)。`parallax DIR` がこれを読みます ([6.8 節](#68-例-擬似立体-samplesparallax))
- 3 層で画像を 2 回描き、1024×1024 で 2 分から 3 分半 (Ryzen AI 9 365)。PSD は層が文書と同じ大きさの 8 ビット RGB、PackBits で詰めたもの。Mac (M3) では、編集のプログラムのコンパイルが済めば 1 層が 512×512 で 39 秒、1024×1024 で 89 秒です

## 7. 機能の一覧

どの機能も、一括の形と流れの形を持ちます。

### 7.1 `ai/speech`

| | 一括 | 流れ |
|---|---|---|
| `Transcriber` (Whisper) | `Transcribe(ctx, pcm, TranscribeOptions) (Transcript, error)`。30 秒より長い音声は窓を進めて読む | `TranscribeStream(ctx, Stream[[]float32], opts) Stream[Text]`。`Final` の塊までをまとめて読み、文を `Final` の塊で返す |
| `Speaker` (LFM3-Audio) | `Speak(ctx, text, SpeakOptions) (Audio, error)` | `SpeakStream(ctx, Stream[Text], opts) Stream[[]float32]`。`Final` の文ごとに読み、音の塊を返す。時刻は出力の標本の数 |
| `VAD` (音量) | `Segments(pcm) []Span` | `Detect(ctx, Stream[[]float32]) Stream[[]float32]`。発話のところだけを通し、発話の終わりに `Final` |

- `TranscribeOptions{Language, Translate, MaxTokens}`: 言語を空にすると 30 秒ごとに判定します。`Translate` は英語に書き起こします
- `SpeakOptions{Language, Voice, Seed, MaxFrames}`: `SpeakStream` は言語が空なら塊の `Text.Lang` を使い、流れ全体で 1 つの種を使います (声がそろう)
- `VADOptions`: 雑音の床から 12 dB、発話の終わりの待ち 600 ms、最短 250 ms、前後の余白 200 ms、最長 25 秒 (どれも変えられます)。モデルは使いません (`speech.NewVAD`。登録は要りません)
- まだない: 話している途中の仮の字幕、できた音から返す読み上げ、モデルによる VAD
- `speech.ReadWAV(data, rate)`、`speech.WriteWAV(w, pcm, rate)`

### 7.2 `ai/text`

| | 一括 | 流れ |
|---|---|---|
| `Translator` (PLaMo 3 Translate) | `Translate(ctx, text, TranslateOptions) (string, error)` | `TranslateStream(ctx, Stream[Text], opts) Stream[Text]`。`Final` の文ごとに訳し、直前の 4 組を履歴にする |
| `Chatter` (チャットのモデル) | `Complete(ctx, []Message, ChatOptions) (string, error)` | `CompleteStream(ctx, msgs, opts) Stream[Text]`。トークンごとに「ここまでの答え」を仮の塊で、最後に全文を `Final` で |

- `TranslateOptions{To, From, Form, Context, History, Glossary, MaxTokens}`: `To` が空なら、日本語の文字があれば英語へ、なければ日本語へ
- `ChatOptions{MaxTokens, Think, ThinkBudget, Temperature}`
- まだない: 道具の呼び出し、画像や音声の入力、決まった形 (JSON Schema) で答える生成、埋め込み

### 7.3 `ai/vision`

| | 一括 | 流れ、節 |
|---|---|---|
| `DepthEstimator` (Depth Anything 3、MoGe-2) | `Estimate(ctx, img) (*DepthMap, error)` | `EstimateStream`、`vision.DepthNode(use)` (絵 → 深度) |
| `FaceDetector` (YuNet) | `Detect(ctx, img) ([]Face, error)` | `DetectStream`、`vision.FacesNode(use)` (フレーム → `Found{Frame, Faces}`) |
| 頭の位置 (モデルなし) | `vision.HeadOf(face, frame, opts)`、`vision.NearestHead(faces, frame, opts)` | `vision.HeadNode(opts)` (`Found` → `Head`。顔のないフレームは流さない) |

- 絵は `image.Image`、カメラの映像はその流れです
- **`DepthMap`**: `Depth` は遠さ (大きいほど遠い。MoGe-2 はメートル、DA3 は絵ごとの単位)。大きさはモデルの入力の大きさで、絵の大きさではありません。`Near(w, h)` が、絵の大きさに直した「近さ」(1 が一番近く、0 が一番遠い。深度の逆数に比例) を返します。視差のシェーダーが読むのはこれです。`NearImage(w, h)` は灰色の絵
- **`Face`**: 箱、確かさ、目・鼻・口の端の位置 (絵の画素)。`RightEye` は本人の右目 (絵では左側)
- **`Head`**: カメラの前の頭の位置。`X`、`Y` はカメラの視野の中の −1〜1 (`X` は本人から見た右が正。フレームとは左右が逆)、`Z` はカメラからの距離 (メートル)。距離は、フレームの中の両目の間隔から、ピンホールのカメラとして出します (`HeadOptions.FOV` が水平の画角、既定は 60 度。`Eyes` が両目の間隔、既定は 63 mm)。カメラを向いた顔での目安です
- **YuNet は実写の顔のモデル**です。アニメの絵の顔は見つけません (Anima で作った人物の絵で 0 件でした)
- **どこで動くか**: 深度は NPU。YuNet は、AMD の NPU の機械では定義 (models.yaml の `device`) が CPU に置きます (小さなグラフで、CPU のほうが 1 枚の時間も電力も 10 分の 1)。CPU では ONNX Runtime が要ります (`HWNPU_ONNXRUNTIME` にライブラリの場所)。`FaceDetector.Device()` が `npu` か `cpu` を返します
- Apple の Neural Engine と Intel の NPU では、深度のプログラムは絵の形ごとに、その形の絵が初めて来たときに変換します (数秒。最初の読み込みでコンパイル)
- **深度のモデルの選び方**: `vision.Depth("depth")` は機械のメモリで選びます (8 GB から DA3-SMALL、16 GB から DA3-BASE、24 GB から DA3MONO-LARGE)。名前で決めることもできます (`vision.Depth("moge-2-vitl")`)。AMD の NPU (Ryzen AI 9 365) で 3 枚の絵 (1024×672 が 1 枚、1024×1024 が 2 枚) にかけた値です

  | モデル | 1 枚 (2 枚目から) | 最初の 1 枚 | NPU のメモリ | 深度の大きさ | 輪郭と層 |
  |---|---|---|---|---|---|
  | DA3-SMALL | 0.31 秒 | 0.57 秒 | 0.48 GiB | 504 | 輪郭がぼやけ、細い枝や髪は消える。層の順は合う |
  | DA3-BASE | 0.43 秒 | 0.86 秒 | 0.74 GiB | 504 | SMALL より少しはっきり |
  | DA3MONO-LARGE | 0.85 秒 | 1.63 秒 | 1.36 GiB | 504 | 輪郭がはっきり (髪の束、枝)。背景の層 (花畑、庭) も分かれる |
  | DA3METRIC-LARGE | 0.85 秒 | 1.50 秒 | 1.36 GiB | 504 | MONO と同じくらい。空や花の中に細かい粒が出る絵がある |
  | moge-2-vits | 2.52 秒 | 3.15 秒 | 1.94 GiB | 1024 | 輪郭は細かいが、背景の層の差が小さい |
  | moge-2-vitb | 2.68 秒 | 3.26 秒 | 2.10 GiB | 1024 | 輪郭が細かく、背景の層も分かれる |
  | moge-2-vitl | 3.41 秒 | 4.17 秒 | 2.62 GiB | 1024 | 一番細かい。空は「一番遠い」に切り分ける |

  ほかにプロセスのメモリが 0.2 GiB (MoGe-2 は 0.4〜0.5 GiB)。Depth Anything 3 の深度は長辺 504 画素、MoGe-2 は 1024 画素です。1 枚の絵に 1 回だけ推定するなら DA3MONO-LARGE (止まった絵の視差には十分) か、輪郭を細かく取りたいとき moge-2-vitb / vitl。映像に続けてかけるなら DA3-SMALL (1 秒に 3 枚) です。MoGe-2 はメートルの深度も返します (`DepthMap.Metric`)
- まだない: 物体、姿勢、切り抜き、OCR、ラベル

### 7.4 `ai/imagegen`

| | 一括 | 節 |
|---|---|---|
| `Generator` (FLUX.2 klein 4B) | `Generate(ctx, prompt, opts) (*image.RGBA, error)` | `imagegen.GenerateNode(use, opts)` (`Request{Prompt, References}` → 絵。`Update` で設定を変える) |

- **文から**: `g.Generate(ctx, "a red fox in the snow", imagegen.Options{})`
- **編集と合成**: `Options.References` (4 枚まで) に絵を渡し、文の中で `image 1`、`image 2` と呼びます。参照は、絵の中央を出力の形に切って出力の大きさにします
- **大きさ**: モデルが描ける大きさは決まっています (`Sizes()`: AMD は 1024×1024、512×512、3:2、16:9 など 8 つ、Apple は 1024×1024 と 512×512)。`Options.Width`、`Height` に一番近い形の大きさで描きます (指定がなければ 1 枚目の参照の形、それもなければ正方形)。`SizeFor(w, h)` がその大きさを返します。形を変えたくない絵は、その大きさに伸ばしてから渡し、結果を伸ばし戻します
- **そのほか**: `Seed` (0 で毎回違う)、`Steps` (0 でモデルの既定。FLUX.2 klein は 4)、`Progress` (段ごとに呼ばれる)
- **時間とメモリ**: 1024×1024 で、文から約 30 秒、参照 1 枚の編集で約 1 分 (Ryzen AI 9 365)。動く間は最大 3.9 GB (AMD)。段の入れ替え ([9.5 節](#95-release-と画像の生成の段の入れ替え)) はパイプラインが自分でします
- **どこで動くか**: AMD の NPU (IRON の書き出し 1 つに全部の大きさ) と Apple の Neural Engine (512×512 と 1024×1024 の書き出し。`Sizes()` が返します)。Apple で参照から描くには編集のプログラム (別の書き出し) が要ります。`imagegen.Edit` が準備し (チェックポイント 16 GB をダウンロードして変換)、最初の編集でコンパイルします。`imagegen.Generate` はそれを用意せず、Apple では参照を断ります。M3 で編集 1 枚が 512×512 で 39 秒、1024×1024 で 89 秒
- まだない: Anima、Chroma、FLUX.3 klein (コマンド `hwnpu-image generate` にはあります)、まとめて描く (`--batch`)、参照を半分の大きさで読む (`@half`)

## 8. アプリのコマンドライン (`ai/cli`)

モデルの世話をするサブコマンド (一覧、ダウンロード、コンパイル、NPU の状態) は `ai/cli` にあります。**コマンドラインのライブラリには依存しません。** 使い方は 3 つです。

**(1) 関数を呼ぶ。** 自分のコマンドの中から呼びます。どのコマンドラインのライブラリでも、GUI のボタンからでも使えます。

```go
cli.List(ctx, app, cli.ListOptions{})                             // 一覧 (JSON: true で JSON)
cli.Download(ctx, app, cli.PrepareOptions{})                      // ダウンロードと変換
cli.Prepare(ctx, app, cli.PrepareOptions{AcceptLicenses: true})   // さらにコンパイル
cli.Status(ctx, app, cli.StatusOptions{Watch: 2 * time.Second})   // NPU の状態
```

**(2) 引数を渡す。** コマンドラインを自分で読むアプリ (標準の `flag` だけ、など) は、引数をそのまま渡します。

```go
func main() {
	ctx := context.Background()
	if handled, err := cli.Run(ctx, app, os.Args[1:]); handled {   // "models ..." か "status ..." だった
		if err != nil {
			log.Fatal(err)
		}
		return
	}
	// ここからアプリ自身の引数。ヘルプには cli.Usage を足せる
}
```

**(3) kong のアダプタ (`ai/cli/konghelper`)。** kong を使うアプリは、構造体を自分のコマンドの横に置きます。

```go
var cmds struct {
	Run    RunCmd            `cmd:"" default:"withargs" help:"..."`
	Models konghelper.Models `cmd:"" help:"The models the app uses: list, download, prepare."`
	Status konghelper.Status `cmd:"" help:"This machine's NPU and what uses it."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	k := kong.Parse(&cmds, konghelper.Bind(ctx, app))
	k.FatalIfErrorf(k.Run())
}

// アプリ自身のコマンドも、ctx と app を受け取れる
func (c *RunCmd) Run(ctx context.Context, app *ai.App) error { ... }
```

| コマンド | 関数 | 中身 |
|---|---|---|
| `APP models` | `cli.List` | 使うモデルの一覧: 状態 (missing、ready, not compiled、compiled、bundled)、落とす量、メモリ、ライセンス、同時に動かす合計とこの機械でモデルに使える量、アプリが構成で選んだ理由、ネットワーク。`--json` |
| `APP models download` | `cli.Download` | ダウンロードと変換。NPU は要りません。`--accept-license`、`-q` |
| `APP models prepare` (`compile`) | `cli.Prepare` | download のあと、各モデルを 1 回読み込んでコンパイル |
| `APP status` | `cli.Status` | `hwnpu status` と同じ: デバイス、ドライバ、負荷、電力、使っているプロセスと NPU のメモリ、キャッシュ。`-w 2s`、`--json`、`--wake` |

```
$ interpreter models
interpreter's models, for this machine's NPU (amd-xdna2)

USE         ROLE        MODEL                   STATE     DOWNLOAD  MEMORY  LICENSE
transcribe  transcribe  whisper-large-v3-turbo  compiled  -         2.2 GB  mit
translate   translate   plamo-3-translate-2b    missing   -         2.7 GB  plamo-community-license (conditional)
speech      speak       LFM3-Audio-2B-TTS       compiled  -         2.9 GB  lfm-open-1.0 (conditional)

Their checkpoints are here or come from the sources; about 7.8 GB running together (a machine of 16 GB has about 9 GB for models). ...
translate: plamo3translate:2b on machines of 16 GB and up, by the app; ...

The network:
  mic → vad → transcribe → heard → commands → translate → said → speech (keep 4) → out
  say → translate → speech
```

- ほかのコマンドラインのライブラリ (cobra、urfave/cli など) のアダプタは置いていません。(1) の関数を、そのライブラリのコマンドから呼んでください
- `cli.Status` は標準出力に書きます (`hwnpu status` と同じ表示のため)。`cli.List` と `cli.Download` / `cli.Prepare` は、書く先を `Out` と `Log` で変えられます

## 9. 制約とルール

NPU は 1 つで、メモリは固定され、プログラムの形は決まっています。アプリの設計は、ここから決まる部分が大きいです。数値は出どころを書きました。**「実測」とあるのは Ryzen AI 9 365 (AMD、XDNA 2) で 2026-11-21 にこのライブラリを通して測った値**で、ほかは [docs/design/README.ja.md](design/README.ja.md) と MODEL.md の記録です。Mac と Intel では、このライブラリを通した測定はまだありません。

### 9.1 NPU は 1 つの待ち行列

**別々のモデルを同時に呼べます。ただし速くはなりません。** 合計の時間は、1 つずつ動かした時間の和になります。

| 呼び方 (実測、AMD) | 単独 | 同時に呼んだとき |
|---|---|---|
| 文字起こし (13.3 秒の音声) | 1.4 秒 | 読み上げと同時で 3.1 秒 |
| 翻訳 (8B、3 文) | 3.3 秒 | 読み上げと同時で 3.8 秒 |
| 読み上げ (3.9 秒ぶんの音声) | 10.1 秒 | 文字起こしと同時で 11.4 秒、翻訳と同時で 13.4 秒 |
| 3 つ同時 | 和は 14.8 秒 | 全部が終わるまで 14.7 秒 |

- 呼び出しは細かく入れ替わりながら進みます (上の文字起こしは、読み上げの途中の 3.1 秒で終わっています)。待たされるのは相手の呼び出し全体ではありません
- **順番を決める仕組みはまだありません。** 音声を先に通す、背景の仕事を待たせる、といった優先度は書けません ([D10](#10-設計の論点))。重い仕事 (画像の生成の 1 ステップは 1〜数秒) の途中に短い呼び出しがどれだけ待つかは、測っていません
- AMD: モデルのプログラムごとにハードウェアコンテキストができ、ファームウェアが時分割します ([docs/design/architecture.ja.md](design/architecture.ja.md))。切り替えは 1 回約 0.5 ms
- Apple: プログラムの切り替えに 1 回 15〜35 ms かかります ([docs/design/llm-decoding.ja.md](design/llm-decoding.ja.md))。1 フレームに何度もプログラムを呼ぶモデル (読み上げ) では、これが効きます。1 プロセスの中で複数のモデルを同時に呼んだときの振る舞いは、測っていません
- Intel: 同時に動かしたときの振る舞いは未測定です ([docs/design/architecture.ja.md](design/architecture.ja.md))

### 9.2 リアルタイムに連携できるか

「同時に動く」ことと「実時間で回り続ける」ことは別です。NPU が 1 つの待ち行列なので、**各段が NPU を使う時間の合計が、実時間の 1 秒あたり 1 秒を下回るとき**だけ、流れは遅れずに回り続けます。上回ると、待ちが積み上がります。

| 段 (実測、AMD) | NPU の時間 |
|---|---|
| 文字起こし (whisper-large-v3-turbo) | 発話 1 つに約 1〜1.4 秒 (30 秒までの発話。窓が 30 秒で固定なので、短い発話でもあまり減らない) |
| 翻訳 (plamo-3-translate-8b / 2b) | 1〜2 文の発話 1 つに約 2.7 秒 / 約 1.1 秒 |
| 読み上げ (LFM3-Audio-2B-TTS) | 音声 1 秒ぶんに約 2.6 秒 (実時間の約 0.4 倍) |

例: 4 秒話して 1 秒休む人を訳し続ける場合 (5 秒ごとに発話 1 つ)。

| 構成 | 発話 1 つの NPU の時間 | 5 秒に対して | 結果 |
|---|---|---|---|
| 字幕だけ (文字起こし + 翻訳 2B) | 約 2.1 秒 | 0.4 | 回り続ける。遅れは発話の終わりから 2〜3 秒 |
| 字幕だけ (翻訳 8B) | 約 3.7 秒 | 0.7 | 回り続けるが余裕は少ない |
| 訳した音声まで | 約 13.6 秒 (読み上げが 10 秒) | 2.7 | 追いつかない。遅れが発話ごとに伸びる |

実測の遅れ (録音をマイクの速さで流し、発話の終わりから数えた時間。`samples/interpreter --live` が出す `+N.Ns`):

| | 1 つ目の発話 | 3 つ目の発話 |
|---|---|---|
| 文字 | +1.6 秒 | +5.4〜5.7 秒 |
| 訳 (8B / 2B) | +4.3 / +2.7 秒 | +8.7 / +7.9 秒 |
| 訳した音声 (8B / 2B) | +20 / +14 秒 | +24 / +19 秒 |

- 文字の +1.6 秒の内訳は、発話の終わりを決める無音の待ち 0.6 秒 (`VADOptions.Hangover`) と、文字起こしの約 1 秒です
- 後の発話ほど遅れるのは、前の発話の読み上げと NPU を取り合うからです
- Mac の記録 (単独): Whisper は M3 で 30 秒の窓を 0.6 秒、PLaMo 3 Translate は M3 で 0.8B が 53、2B が 17、8B が 10〜11 トークン/秒、読み上げは M3 の lut8 で実時間の約 0.5 倍 ([docs/design/llm-models.ja.md](design/llm-models.ja.md)、古い測定)。読み上げが実時間に届かないのは AMD と同じです

**合計が 1 を超えるときの選び方**

| 状況 | 設計 |
|---|---|
| 古い入力は要らない (カメラのフレーム → 検出) | 捨てる: `flow.Latest`。検出は「今」だけを見て、間引かれる |
| 出力の時刻が決まっている (描画) | 拍に合わせる: `flow.Metronome` と `flow.Hold`。モデルの結果は来たときに差し替わる |
| 全部要るが、遅れてよい (録音の書き起こし、あとで読む訳) | ためる: `flow.Async`。待ちは伸びるが、失わない |
| 全部要り、遅れも困る (会話) | **順番に話す形**にする (話し終わるまで次を聞かない、押している間だけ聞く)。または軽くする: 小さいモデル (`use.At`、`use.As`)、短い出力、段を減らす (音声をやめて字幕に) |

- 合計が 1 を下回っていても、遅れは「発話が終わるまで」+ 各段の時間です。話している途中の仮の字幕や、できた音から鳴らす読み上げは、まだありません
- その機械で合計がいくつになるかを、アプリが測る口はまだありません ([D20](#10-設計の論点))。今は上のように単独の時間を測って足します

### 9.3 1 つのモデルに複数の入力

- **モデルは 1 回に 1 つの要求を通します。** 同時に呼ぶと、順に通ります
- **呼び出しどうしは独立です。** `Transcribe`、`Translate`、`Speak`、`Complete` は、前の呼び出しの結果に依りません (翻訳の履歴も、会話のメッセージも、呼ぶ側が毎回渡します)。だから 1 つのモデルに複数の入力を交互に通せます
- **L と R の音声**: 入力はモノラルです (文字起こしと VAD は 16 kHz のモノラル)。ステレオは 2 本のモノラルの流れに分け、それぞれに VAD をかけ、同じ文字起こしのモデルに通します。実測: 13.3 秒の音声 2 本を同時に通すと 2.8 秒 (1 本 1.4 秒の 2 倍)。9.2 節の合計には 2 本ぶんが入ります
- **同じモデルを 2 つ読み込んでも速くなりません**: NPU が 1 つの待ち行列なので、合計は同じで、メモリだけ倍になります (2 つ読み込む形は測っていません)
- **まとめて 1 回で処理する入力 (バッチ) は、ほとんどありません。** プログラムの形は固定で、文字起こしは 30 秒の窓 1 本、チャットと翻訳は 1 系列、埋め込みは 1 文ずつです。例外は画像で、画像の生成は数枚をまとめて描け (`hwnpu-image --batch`)、ラベル付けの文は 8 つずつです
- チャットのモデルは、直前の会話 (と、その前の長い会話 2 つまで) の続きなら読み直しを省きます (結果は変わりません)。それより多くの会話を 1 つのモデルに交互に通すと、その省略が効かなくなります (遅くなるだけ)

### 9.4 メモリ

**まず、アプリの `models` サブコマンドを見ます。** その機械で、登録したモデルが同時に動かすといくつになり、モデルに使えるのがいくつで、収まるかどうかを出します (8 節。アプリに `cli` のサブコマンドを足してあれば使えます)。

```
$ interpreter models
...
translate   translate   plamo-3-translate-2b    missing   -         2.7 GB  ...
...
about 7.8 GB running together (a machine of 16 GB has about 9 GB for models).
```

- 収まらないときは、その下に注意が出ます (「Together they take more than this machine has for models ...」)
- ほかの機械ではどうなるかは、`HWNPU_MEMORY=8 interpreter models` のように、機械のメモリを言い換えて見られます
- プログラムからは `interpreter models --json` (`memory_gb`、`machine_gb`、`usable_gb`、`fits`、モデルごとの `memory_gb`)、または `plan.MemoryGB()`、`plan.MachineGB()`、`plan.UsableGB()`、`plan.Fits()`
- **実際に使っている量**は `APP status` が出します: NPU を使っているプロセスごとの、プロセスのメモリと NPU のメモリ (`-w 2s` で見続けられます)

**数字の意味**

- **NPU のバッファは固定されたメモリで、スワップに逃がせません。** 超えると、ほかのアプリがスワップに追い出されます。同時に読み込んでいるモデルのメモリは足し算です (共有されません)
- 「モデルに使える量」は、OS とふだんのアプリの分を引いた目安です (MODEL.md)。機械のメモリが 8 GB で約 3 GB、16 GB で約 9 GB、24 GB で約 15 GB、32 GB で約 22 GB、48 GB で約 35 GB、64 GB で約 50 GB
- モデルごとのメモリは models.yaml の値で、プロセスのぶんと NPU のぶんを合わせたものです。Apple では Neural Engine のぶんがプロセスのメモリ (footprint) に出ず、AMD でも NPU のバッファはプロセスの RSS に出ないので、OS のプロセスの一覧だけでは分かりません
- 例 (同時通訳、AMD の値): 文字起こし 2.2 GB + 読み上げ 2.9 GB に、翻訳が 0.8B なら 1.0、2B なら 2.7、8B なら 6.5 GB。16 GB の機械 (約 9 GB) に 8B は入りません

**機種ごとに分かっている上限** (サブコマンドには、まだ出ません)

| 機種 | 制約 | 出どころ |
|---|---|---|
| M1 (Pro、Max、Ultra も同じとみなす) | Neural Engine に同時に載せるプログラムが 2.8 GiB を超えると、どの呼び出しも約 12 倍遅くなる。大きいモデル (PLaMo 3 Translate 8B など) は用途から選ばれない (`not_on: [M1]`) | `internal/chat/model.go` の `m1HoldBytes`、PROBE.md |
| M2 以降 | 上の制限は見つかっていない (M2 で 3.65 GiB まで全速) | 同上 |
| Apple 全般 | Neural Engine のコンパイルのキャッシュは約 4 GiB。超えるぶんは起動のたびにコンパイルし直す (数分)。コンパイルは機械全体で 1 本の直列 | [docs/design/llm-qwen.ja.md](design/llm-qwen.ja.md)、[docs/design/apple-coreai.ja.md](design/apple-coreai.ja.md) |
| AMD (XDNA 2) | NPU 専用のメモリの上限は見つかっていない (バッファはホストのメモリ)。DRAM の帯域 (約 55 GB/s) を取り合うので、大きな重みを読むモデルどうしは互いに遅くなる | [docs/design/architecture.ja.md](design/architecture.ja.md) |
| Intel | Core Ultra 5 135U で、Anima の fp16 が約 4.6 GB で `OUT_OF_HOST_MEMORY`。プログラム 1 本の重みの上限は分かっていない | [docs/design/intel.ja.md](design/intel.ja.md)、intel-npu-models.md |

### 9.5 `Release` と、画像の生成の段の入れ替え

どちらも「同時にメモリに載せない」ための仕組みですが、**層が違います**。

| | 画像の生成の段の入れ替え | `use.Release(rt)` |
|---|---|---|
| 単位 | 1 つのモデルの中の段 (文の符号器 → DiT → VAE) と、DiT の中のブロックの組 | 用途 (モデル) まるごと |
| 誰がするか | パイプラインが自分で (既定で入っている) | アプリが呼ぶ |
| 次に使うとき | 段ごとに自動で読み込む | 次の `Get` が全部を読み込み直す |
| 今どこにあるか | `internal/imagegen` (コマンドと `ai/imagegen` が使う) | `ai` |

画像の生成の入れ替えは、機種ごとに作りが違います (記録の値)。

- **AMD**: 3 つの段は同時に動かないので、動く間だけ NPU に置きます。Anima の 1024² で、最大が 14.2 → 5.0 GiB。読み込みは 1 枚に数秒。DiT は 1 つのバンクに組ごとの重みを読み込みます (固定メモリが 7 × 464 → 464 MiB。`--resident-dit` で全部置く)
- **Apple (Core AI)**: DiT の組を、順番が来たら読み込み、回したら閉じ、次の組を先に読みます。Anima-Turbo で、Neural Engine が抱える量が 2.53 → 0.57 GiB、1 枚の時間は同じ 49 秒 (M1)。Chroma は 8.3 → 2.2〜2.6 GiB で、入れ替えるほうが速い
- **Intel**: 入るだけ持ち、入らなくなったら使っていないものを閉じて読み直します

- `ai/imagegen` (FLUX.2 klein) は、この入れ替えを既定のまま持っています。アプリは生成のあと `Release` を呼んで残り (小さい部品) も下ろし、深度などの次のモデルに進みます ([6.9 節](#69-例-深度でレイヤーに分ける-sampleslayers) のサンプルがそうしています)
- 3 つ目の仕組みとして、AMD のチャットのモデルは、しばらく使わないと固定メモリを手放し、次の要求で 1〜2 秒かけて読み直します (`hwnpu-chat --idle`)。`ai` からはまだ使えません
- この 3 つを 1 つの「常駐の管理」(メモリの予算、追い出してよいもの) にまとめるのが [D9](#10-設計の論点) です
- 読み直しの時間 (実測、AMD): 文字起こし 0.5〜1.2 秒、翻訳 8B 5〜9 秒、2B 約 3 秒、読み上げ 2.3〜3.8 秒。Apple では、コンパイル済みなら数秒、初回は数十秒から数分です

### 9.6 プロセス

- **Mac: Neural Engine を使う hwnpu のプロセスは 1 つずつ**です。2 つ同時に使うと M1 の OS が固まったので、最初にデバイスを開いたプロセスが、終わるまでロックを持ちます ([docs/design/apple-neural-engine.ja.md](design/apple-neural-engine.ja.md))。常駐するアプリが動いている間、`hwnpu-chat` などのコマンドは待ちます。アプリが NPU を手放す口 (`Suspend`) はまだありません
- AMD と Intel にはこのロックがありません。AMD では別々のプロセスが同じ NPU を使えますが、時間を分け合います (ほかのプロセスが NPU を使っていて、コマンドが最大 50 倍遅くなった記録があります。[docs/design/new-model-checklist.ja.md](design/new-model-checklist.ja.md))。Intel は試していません
- Mac では macOS 自身のサービス (写真の解析、Apple Intelligence) も Neural Engine を使います。それが動いている間、チャットが 0.7 トークン/秒に落ちた記録があります ([docs/design/llm-engine.ja.md](design/llm-engine.ja.md))。アプリからは止められません
- **1 プロセスに `App` は 1 つ**と考えてください。置き場の設定がプロセスに 1 つだからです ([D8](#10-設計の論点))
- AMD で 1 プロセスに複数のモデルを読み込めるのは、デバイスヒープの番地の直し ([docs/design/api.ja.md](design/api.ja.md)) が入った版からです

### 9.7 入力と出力の形

| | 形 |
|---|---|
| 文字起こし、VAD の入力 | モノラル、16 kHz、float32 (−1〜1)。30 秒の窓。VAD は発話を 25 秒で切る (`VADOptions.MaxSpeech`) |
| 読み上げの出力 | モノラル、24 kHz。文 1 つをまとめて返す (途中からは鳴らせない) |
| 翻訳、チャット | 文を 1 つずつ。チャットの文脈の長さはモデルの定義で決まる |
| 画像 (予定) | 大きさは書き出しにあるものだけ (アスペクト比ごとに決まった大きさ) |

- 標本化周波数やチャンネルの数が違う音は、流す前に合わせます (`speech.ReadWAV(data, rate)` はファイルを合わせて読みます。hwcamera のマイクは OS が合わせます)

## 10. 設計の論点

今の形と、そうした理由、ほかの案を並べます。**決めていないもの**には印をつけました。

### 登録と取り出し

**D1. 型のついた取っ手 `Use[T]`** — 今: `stt := speech.Transcribe("transcribe")`、`stt.Get(ctx, rt)` が `*speech.Transcriber` を返す。

- 理由: 型の合わせ間違いがコンパイルで分かる。`ai` が機能のパッケージを import しないで済む (使う機能のコードだけがアプリに入る)
- 別の案 A: `rt.Transcriber(ctx, "transcribe")` (前のブランチ)。短いが、`ai` が全部の機能を import し、名前を 2 か所に書く
- 別の案 B: `ai.Get(ctx, rt, stt)` の関数。Go はメソッドに型引数を持てないので、`rt.Get(stt)` とは書けない
- **未定**: `stt.Get(ctx, rt)` の語順 (取っ手が主語) でよいか。`Use` をパッケージの変数に置く書き方でよいか

**D2. 関数の名前** — 今: `speech.Transcribe` (動詞) が `Use` を返し、型は `Transcriber`。`text.Chat` の型は `Chatter` (関数と名前が当たるため)。

- **未定**: 取っ手を作る関数を `speech.TranscribeUse("...")` や `speech.Transcriber.Use("...")` にするか。`Chatter` という名前

**D3. 名前を必ず書く** — 今: `speech.Transcribe("transcribe")`。既定の名前はない。

- 別の案: 引数なしで用途の既定 (`speech.Transcribe()` が `transcribe`)。短いが、同じ役割の用途を 2 つ使うときに区別がつきにくい

**D4. 同じ役割のモデルを 2 つ** — 今: 名前が違えば別の用途 (`text.Chat("chat")` と `text.Chat("code")`)。置き場の鍵は `役割:名前`。

- 同じ名前を 2 つの構成で使うこと (`translate` を大きいものと小さいもので 2 つ) はできない。アプリの定義のファイルで別の用途の名前を作れば書ける
- **未定**: `Use` に自分の名前をつける口 (`text.Translate("translate").Named("subtitle")`) が要るか

### モデルの選び方

**D5. メモリごとの構成の書き方** — 今: 用途ごとに `At(gb, name)` と `As(gb)`。

- 別の案 A: アプリの表 `app.LineUp(16, tr.Is("plamo3translate:2b"), ...)`。機械のメモリごとに構成全体が 1 か所で読めるが、口が増える
- 別の案 B: アプリの定義のファイルに用途を書く (3.3 節、今もできる)。コードを変えずに配布後も差し替えられるが、型のついた取っ手との対応が名前だけになる
- 別の案 C: 合計が機械のメモリに収まるよう自動で小さいモデルに落とす。どれを主にするかはアプリが知っているので、優先度 (`use.Priority(n)`) を書かせることになる
- **未定**: A か C を足すか。`At` の段は「機械のメモリ」で切っているが、「空いているメモリ」で切るべき場面があるか

**D6. 構成が変わったときの置き場** — 今: 記録のモデルが今の選び方と違えば `ErrNotPrepared`。記録は用途ごとに 1 つなので、メモリの設定を行き来すると、そのたびに準備し直す (ファイルは残っているので、作り直すのは足りないものだけ)。

- 別の案: 用途ごとに複数の記録を持つ。行き来が速いが、目録が増える

### 準備

**D7. 準備の 2 段の口** — 今: `plan.NoCompile = true` と `plan.Run`。

- 別の案: `plan.Download(ctx, fn)` と `plan.Compile(ctx, fn)` に分ける。否定の旗がなくなり、読みやすい
- **未定**: どちらにするか。`AcceptLicenses` を旗にするか、`plan.Accept(licenses...)` にするか

**D8. 置き場の設定がプロセスに 1 つ** — 今: `App.Plan` と `App.Open` が `modelstore.Open` を呼び、プロセス全体の設定 (キャッシュの場所、取得元、定義) を置き換える。

- 中の仕組み (`internal/cache`、`internal/chat` の定義の表) がプロセスに 1 つなのを、そのまま出している
- **未定**: 1 プロセスに `App` は 1 つ、と決めてしまうか。置き場を値として持ち回る形に直すか (中の直しが大きい)

### 実行

**D9. メモリの管理はアプリ任せ** — 今: `Get` で載せ、`Release` で下ろす。予算も追い出しもない。

- 別の案: `Runtime` に予算を持たせ、足りなければ使っていないモデルを下ろす ([docs/design/architecture.ja.md](design/architecture.ja.md))。下ろしてよいかを `Use` に書く (`use.Resident()`、`use.Evictable()`)。取り出したモデルの値が下ろされても壊れないよう、値を間接にする必要がある
- **未定**: いつ入れるか。入れると `Get` が返す値の寿命の決まりが変わる

**D10. 並行** — 今: モデルごとに 1 回に 1 つ。モデルどうしの順番は決めない (NPU のドライバ任せ)。モデルごとにデバイスを開く。

- **未定**: 音声を先に通す、背景の仕事を待たせる、といった優先度の口。Mac では Neural Engine のロックをプロセスが生きている間持つので、常駐するアプリがほかの hwnpu のコマンドを待たせる

### 流れ

**D11. 流れは `iter.Seq2`、音声は `[]float32`** — 今: 引く形の反復子。音声の標本化周波数は塊に入れず、段が決める。

- 理由: `range` と `break` と ctx だけで書け、段が関数になる。goroutine は要るところ (`Async`、`Latest`、`Hold`) だけ
- 別の案 A: チャネル。`select` で複数の流れを待てるが、閉じ方とエラーの渡し方を毎回決めることになる
- 別の案 B: 音声を `flow.Audio{Samples, Rate, Channels}` にし、つなぐときに周波数を合わせる ([architecture](design/architecture.ja.md) の「ポートが型と制約を宣言する」)。間違ったつなぎ方が実行時に分かる
- **未定**: B を入れるか。画像の型 (`image.Image` のままか、時刻つきのフレームか)。ステレオ

**D12. `Final` の 2 つの意味** — 今: 音声では「発話の終わり」、テキストでは「確定した文」(その前は仮の文で、次が置き換える)。

- **未定**: 仮のものと区切りを別の欄にするか (`Chunk.Partial` と `Chunk.End`)。仮のテキストを、全文の置き換えではなく差分にするか

**D13. 制御** — 今: 取り消しは ctx だけ。

- **未定**: 割り込み (読み上げを途中で止めて次へ)、一時停止、流れの途中での設定の変更 (訳す先の言語を変える) の口。別の流れとして渡すか (`tts.SpeakStream(ctx, in, opts, flow.Interrupt(events))`)

**D14. 時計** — 今: 流れごとに自分の時刻。`Metronome` は壁時計で、流れどうしをそろえる共通の時計はない。

- **未定**: 音と映像をそろえるときの基準 (再生の時計に合わせるか)。`Metronome` の拍を音声の標本の数で数える版

### そのほか

**D15. 進み具合** — 今: 段階と、ログの 1 行 (`Progress.Message`)。バイト数や割合はない (中の `Prepare` が取得と変換を 1 つの関数で行い、ログを書くだけなので)。

**D16. 設定の渡し方** — 今: 実行の設定は構造体 (`TranscribeOptions{...}`)、`Open` の設定は関数 (`ai.WithLog(w)`)。変換の設定 (重みの形式、文脈の長さ) は models.yaml の定義で、コードからは変えられない。

- **未定**: 登録のときに変換の設定を書けるようにするか (`text.Chat("lfm3").Context(8192)`)。コマンドを載せ替えるには要る

**D17. サブコマンドとコマンドラインのライブラリ** — 今 (2026-11-21 に決めた): サブコマンドの中身は `ai/cli` の関数で、コマンドラインのライブラリに依存しない。kong 用は `ai/cli/konghelper` のアダプタ。引数をそのまま渡す `cli.Run(ctx, app, args)` もある (8 節)。

- 前の形: `ai/aicli` が kong の構造体だけを出していた
- **未定**: ほかのライブラリ (cobra など) のアダプタを置くか。置くと本体のモジュールの依存が増えるので、置くなら別のモジュールにする

**D18. パッケージの名前と分け方** — 今: `ai`、`ai/speech`、`ai/text`、`ai/flow`、`ai/cli`、`ai/cli/konghelper`。`ai/vision`、`ai/imagegen`。予定: `ai/sound`。

- **未定**: `ai` という名前 (公開するとあとから変えにくい)。`speech` に文字起こしと読み上げの両方を置く分け方。`text` という名前 (標準ライブラリの `text/...` と並ぶと紛らわしい)

**D19. 入力と出力のデバイス** — 今: ライブラリは持たない。流れを受け取り、流れを返すだけ。サンプルが hwcamera でマイク (とカメラ) を流れに直す (`samples/internal/capture`)。

- 理由: ライブラリ本体の依存を増やさない。入力はファイルでもネットワークでも同じ形になる
- 別の案: hwcamera を流れに直す部分を、hwnpu の別のモジュール (本体の依存にならない) として出す。アプリごとに同じコードを書かずに済み、「ためる」「最新だけ」の決まりを 1 か所に置ける
- **未定**: その別のモジュールを作るか。音の出力 (再生) の口をどこに置くか

**D20. 実時間で回るかを測る口と、順番を決める口** — 今: どちらもない。アプリは単独の時間を自分で測って足す (9.2 節)。

- 別の案 A: `Runtime` がモデルごとの呼び出しの回数と時間を数え、`rt.Loaded()` で返す。アプリは「この機械で合計がいくつか」を見て、構成を変えたり、利用者に知らせたりできる
- 別の案 B: `APP models bench` のようなサブコマンドで、登録したモデルを決まった入力で測り、その機械の目安を出す
- 別の案 C: 用途に優先度を書く (`use.Priority(ai.Interactive)`)。音声を先に通し、背景の仕事は切れ目で待たせる。NPU への投入を 1 か所に集める必要がある (D10)
- **未定**: どれを入れるか。A は小さく、C は常駐の管理 (D9) と一緒になる

**D21. ネットワークと呼び出しを宣言する** — 今 (2026-11-21 に v0 を入れた、6.6 節と 6.7 節): つなぎ方を構造として書く。動き続ける構成は `app.From(source).To(node).To(node)`、要求ごとに 1 回の構成は `ai.NewCall[In, Out](app, name).Then(node).Then(node)`。

- そうした理由: ライブラリがつなぎ方を知ると、実時間で回るかの見積もり (9.2 節) や辺ごとの計測 (D20)、辺ごとの渡し方、メモリの見積もり (D9)、ネットワークの表示ができる
- 決めたこと:
  - **モデルと、ネットワークの中の場所を分けた**: `Use` がモデル、`speech.TranscribeNode(use, opts)` がその場所。同じモデルを 2 か所に置ける
  - **続けて書ける**: `To` と `Then` は、続きを書ける場所を返す。型は、節の `To` はコンパイルで、続きの `To` は組み立てのときに確かめる (Go のメソッドは型引数を持てないので、続きに型を運べない)
  - **後ろをいつ動かすかは、プログラムの節が決める**: 「前が全部終わってから次へ」「区切りまでたまってから」を辺の種類にはせず、`flow.Each` の関数が `emit` を呼ぶかどうかで決める。待ち続けないための `flow.WhenIdle` と、終わりの `flow.AtEnd`
  - **振り分けと合流もプログラムの節**: `flow.Route` (出力を選ぶ)、`flow.Both` (両方を待つ)、`flow.With` (片方の最新を持っていく)、`flow.Join2` (全部自分で)
  - **設定は節が持つ**: `node.Update(func(o *Options))` で、動いている間も変えられる
  - **並べるだけの登録も残した**: `ai.NewApp(name, uses...)` と `Get`
  - **流れの関数は下の層として残した**
- **未定**:
  - **名前**: 今は、動き続ける構成が `From` / `To`、要求ごとが `NewCall` / `Then` / `Do`。ほかの案: `To` の代わりに `Connect` (向きが読みにくい) や `Then` (時間の順に読める)、`Call` の代わりに `Request`、`Procedure`、`Once`。`To` と `Then` を別の語にしたのは、「流れ続ける」と「1 回通す」を書き分けるため
  - **型の確かめを全部コンパイルに寄せるか**: 続けて書く形をやめて `flow.Join(a, b)` のような関数だけにすれば、全部コンパイルで確かめられる。今は書きやすさを取っている
  - **入力が 3 つ以上の節、名前のついた出力をいくつも持つ節** (仮の字幕と確定した文、深度と信頼度)
  - **モデルを下ろすタイミング**: 呼び出しの中で、絵の生成が終わったらそのモデルを下ろしてから深度を読み込む、をライブラリに任せるか (D9)。今は `Release` をアプリが呼ぶ
  - **辺ごとの計測**: 通った塊の数、待っている数、遅れ (D20)
  - **節の名前**: 今はモデルの節の名前が用途の名前で、同じモデルの節が 2 つあると同じ名前になる
  - **走っている間のつなぎ替え**: 今はできない

## 11. まだないもの

- 画像: 物体、姿勢、切り抜き、OCR、ラベル (`ai/vision` に足す)、FLUX.2 klein のほかの画像のモデル (Anima、Chroma、FLUX.3 klein を `ai/imagegen` に。今はコマンド `hwnpu-image generate` で作る)
- 音: 音楽、効果音、音高、採譜 (`ai/sound`)
- コマンドの載せ替え: `hwnpu-audio`、`hwnpu-chat`、`hwnpu-image` を、このライブラリを呼ぶだけにする
- 常駐の管理 (D9、D10)、ローカルサーバー (OpenAI 互換など)、配布用のバンドル、ビルドタグ
- 順番と中身は [app-api-tasks.md](app-api-tasks.md)

## 12. hwnpu の開発者向け: 機能のパッケージを足す

機能のパッケージは、`internal/airole` の `Role` を 1 つ持ち、`ai.Define` で `Use` を作ります。`ai` はそれ以外を知りません。

```go
// ai/vision/depth.go (想像図)
package vision

func Depth(name string) ai.Use[*DepthEstimator] {
	return ai.Define[*DepthEstimator](depthRole, name)
}

var depthRole = &airole.Role{
	Name:  "depth",
	Kinds: []chat.Kind{chat.KindDepth},
	// 準備: コマンドが初回にするのと同じこと。Load が読むファイルを返す
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		dir, err := ...(e.Ctx, e.Model, e.NPU, e.Log)
		f.Set("dir", dir)
		return f, err
	},
	// 読み込み: ファイルだけから。ネットワークにも変換にも触らない
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		...
		return &DepthEstimator{...}, nil
	},
}
```

- **`Prepare` と `Load` を分ける**: `Prepare` は取得と変換、`Load` は `Files` に書いた道筋だけを読みます。`Files` の道筋は、置き場の目録にキャッシュの根からの相対で書かれます (準備済みのものを別の場所に写して配れるように)
- **コマンドと同じ関数を同じ設定で呼ぶ**: プログラムの名前と置き場所がコマンドと同じになり、互いに準備したものを使えます。コマンドの中 (`cmd/...`) にある手順は、先に `internal` に移します
- **NPU ごとの分かれ道**は `e.NPU` で見ます: `e.NPU.AMD()` なら IRON の書き出し、ほかは `e.NPU.Format` (`coreai`、`intel`) のプログラム
- **一括と流れの両方**を出します。流れの形は、区切り (`Final`) ごとに一括を呼ぶところから始めて構いません
- **モデルの値は `io.Closer`** で、1 回に 1 つの要求を通すようにします (中で排他する)
- **テスト**は、偽の `Role` で計画から読み込みまでを NPU なしで回せます (`ai/ai_test.go`)。時刻に関わるものは `testing/synctest` を使います
- 外のモジュールは `internal/airole` を import できないので、`Use` を作れるのは hwnpu の中のパッケージだけです
