# 開発者向けガイド: hwnpu でアプリを作る (`ai` のパッケージ群)

[English](developer-guide.md)

> **v0**: この API は最初の版で、形はまだ変わります。この文書は、使い方の説明と、API の設計を今の段階で見直すための材料を兼ねています。設計の論点は [9 節](#9-設計の論点) に番号をつけて並べました (「D4 は案 B で」のように指してください)。

hwnpu は、NPU で AI のモデルを動かす Go のライブラリの集まりです。`hwnpu-chat`、`hwnpu-image`、`hwnpu-audio` は、その機能を 1 つずつ試すためのデモという位置付けです。アプリは複数のモデルを登録し、前もって準備し、つないで動かします。

- 今あるもの: 文字起こし、翻訳、チャット、読み上げ、音量の VAD、モデルの間のつなぎ方、アプリに足すサブコマンド
- まだないもの: 画像 (深度、顔、物体、生成)、音楽と効果音、常駐の管理、ローカルサーバー ([10 節](#10-まだないもの))
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
| `.../ai/flow` | モデルの間のデータの渡し方: `Stream` と道具 |
| `.../ai/aicli` | アプリのコマンドラインに足す kong のサブコマンド: `models`、`status` |

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

複数のモデルをつなぐ動く例は [samples/interpreter](../samples/interpreter) にあります (録音の音声を別の言語の音声に訳す)。

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

3.2 節の `At` とは、同じことをコードに書くかファイルに書くかの違いです ([D5](#9-設計の論点))。

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
- `Progress.Message` は、ダウンロードや変換のログの 1 行です。バイト数の入った値はまだありません ([D15](#9-設計の論点))
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
- 置き場の設定 (キャッシュの場所、取得元、定義) は、今はプロセスに 1 つです。1 つのプロセスで違う `Store` の `App` を 2 つ同時には使えません ([D8](#9-設計の論点))

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
- **順に使うアプリ**は `Release` で下ろします。画像の生成のあとに深度、のような使い方で、同時に載るメモリが一番大きいモデルの分で済みます。自動で下ろす仕組みはまだありません ([D9](#9-設計の論点))
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
- 音声は `[]float32` (モノラル、−1〜1)、テキストは `flow.Text{Text, Lang}` です。音声の標本化周波数は塊に入っていません: 文字起こしと VAD は 16 kHz (`speech.TranscribeRate`)、読み上げの出力は `Speaker.Rate()` (24 kHz) です ([D11](#9-設計の論点))

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

- 取り消しは ctx です。割り込み (相手が話し始めたら読み上げを止める) の口はまだありません ([D13](#9-設計の論点))

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

## 8. アプリのコマンドライン (`ai/aicli`)

```go
var cli struct {
	Run    RunCmd       `cmd:"" default:"withargs" help:"..."`
	Models aicli.Models `cmd:"" help:"The models the app uses: list, download, prepare."`
	Status aicli.Status `cmd:"" help:"This machine's NPU and what uses it."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	k := kong.Parse(&cli, aicli.Bind(ctx, app))
	k.FatalIfErrorf(k.Run())
}

// アプリ自身のコマンドも、ctx と app を受け取れる
func (c *RunCmd) Run(ctx context.Context, app *ai.App) error { ... }
```

| コマンド | 中身 |
|---|---|
| `APP models` | 使うモデルの一覧: 状態 (missing、ready, not compiled、compiled、bundled)、落とす量、メモリ、ライセンス、同時に動かす合計と機械のメモリ、アプリが構成で選んだ理由。`--json` |
| `APP models download` | ダウンロードと変換。NPU は要りません。`--accept-license`、`-q` |
| `APP models prepare` (`compile`) | download のあと、各モデルを 1 回読み込んでコンパイル |
| `APP status` | `hwnpu status` と同じ: デバイス、ドライバ、負荷、電力、使っているプロセス、キャッシュ。`-w 2s`、`--json`、`--wake` |

```
$ interpreter models
interpreter's models, for this machine's NPU (amd-xdna2)

USE         ROLE        MODEL                   STATE     DOWNLOAD  MEMORY  LICENSE
transcribe  transcribe  whisper-large-v3-turbo  compiled  -         2.2 GB  mit
translate   translate   plamo-3-translate-2b    missing   -         2.7 GB  plamo-community-license (conditional)
speech      speak       LFM3-Audio-2B-TTS       compiled  -         2.9 GB  lfm-open-1.0 (conditional)

Their checkpoints are here or come from the sources; about 7.8 GB running together, of this machine's 16 GB. ...
translate: plamo3translate:2b on machines of 16 GB and up, by the app; ...
```

## 9. 設計の論点

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

- 別の案: `Runtime` に予算を持たせ、足りなければ使っていないモデルを下ろす (design.md 16.94)。下ろしてよいかを `Use` に書く (`use.Resident()`、`use.Evictable()`)。取り出したモデルの値が下ろされても壊れないよう、値を間接にする必要がある
- **未定**: いつ入れるか。入れると `Get` が返す値の寿命の決まりが変わる

**D10. 並行** — 今: モデルごとに 1 回に 1 つ。モデルどうしの順番は決めない (NPU のドライバ任せ)。モデルごとにデバイスを開く。

- **未定**: 音声を先に通す、背景の仕事を待たせる、といった優先度の口。Mac では Neural Engine のロックをプロセスが生きている間持つので、常駐するアプリがほかの hwnpu のコマンドを待たせる

### 流れ

**D11. 流れは `iter.Seq2`、音声は `[]float32`** — 今: 引く形の反復子。音声の標本化周波数は塊に入れず、段が決める。

- 理由: `range` と `break` と ctx だけで書け、段が関数になる。goroutine は要るところ (`Async`、`Latest`、`Hold`) だけ
- 別の案 A: チャネル。`select` で複数の流れを待てるが、閉じ方とエラーの渡し方を毎回決めることになる
- 別の案 B: 音声を `flow.Audio{Samples, Rate, Channels}` にし、つなぐときに周波数を合わせる (16.94 の「ポートが型と制約を宣言する」)。間違ったつなぎ方が実行時に分かる
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

**D17. kong に決め打ち** — 今: `aicli` は kong の構造体。

- 別の案: 同じ中身を関数でも出す (`aicli.WriteModels(w, plan)`、`aicli.RunPrepare(ctx, app, opts)`)。ほかの CLI のライブラリや GUI から呼べる
- **未定**: 関数も出すか

**D18. パッケージの名前と分け方** — 今: `ai`、`ai/speech`、`ai/text`、`ai/flow`、`ai/aicli`。予定: `ai/vision`、`ai/imagegen`、`ai/sound`。

- **未定**: `ai` という名前 (公開するとあとから変えにくい)。`speech` に文字起こしと読み上げの両方を置く分け方。`text` という名前 (標準ライブラリの `text/...` と並ぶと紛らわしい)

## 10. まだないもの

- 画像: 深度、顔、物体、姿勢、切り抜き、OCR、ラベル (`ai/vision`)、画像の生成 (`ai/imagegen`)
- 音: 音楽、効果音、音高、採譜 (`ai/sound`)
- コマンドの載せ替え: `hwnpu-audio`、`hwnpu-chat`、`hwnpu-image` を、このライブラリを呼ぶだけにする
- 常駐の管理 (D9、D10)、ローカルサーバー (OpenAI 互換など)、配布用のバンドル、ビルドタグ
- 順番と中身は [app-api-tasks.md](app-api-tasks.md)

## 11. hwnpu の開発者向け: 機能のパッケージを足す

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
