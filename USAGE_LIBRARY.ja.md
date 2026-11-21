# hwnpu を Go のライブラリとして使う (planned)

[English](USAGE_LIBRARY.md)

> **planned**: Go のライブラリとしての API は予定の段階です。名前や引数は予告なく変わります。

hwnpu の Go の API の使い方と、NPU ごとの注意です。コマンド (hwnpu-chat、hwnpu-image、hwnpu-audio、hwnpu-dev) の使い方は [README.ja.md](README.ja.md) から辿ってください。

## 基本

```go
dev, err := hwnpu.Open(ctx)
if err != nil { ... }
defer dev.Close()

gr, _ := dev.Graph()
g, err := gr.Compile(ctx, hwnpu.GraphSource{Path: "model.mlpackage"})
if errors.Is(err, hwnpu.ErrNotOnNPU) {
	// err は *hwnpu.PlacementError。NPU 以外に置かれた演算が Report.Ops に入っている
}
defer g.Close()

in, _ := g.Arg("x")
out, _ := g.Arg("y")
x, _ := dev.AllocBuffer(in.Bytes)
y, _ := dev.AllocBuffer(out.Bytes)
hwnpu.EncodeF16(x.Bytes(), input)

req, _ := g.NewRequest()
req.BindName("x", x)
req.BindName("y", y)
err = req.Run(ctx)
hwnpu.DecodeF16(result, y.Bytes())
```

## 電力とキャッシュ

```go
dev, err := hwnpu.Open(ctx,
	hwnpu.WithPowerPolicy(hwnpu.PowerLow), // Intel: 低優先度 + バックグラウンド
	hwnpu.WithCacheDir(cacheDir),          // コンパイル結果を保存して再利用
)

before, err := dev.PowerSample() // 電力量を読めないデバイスでは ErrUnsupported
// ... 推論 ...
after, _ := dev.PowerSample()
fmt.Println(hwnpu.AveragePower(before, after), "W", after.Scope) // "package" は CPU パッケージ全体
```

- 電力ポリシーは、`CapPowerControl` のあるデバイスでだけ効きます。Core ML には電力の設定がありません
- Intel の電力量は Linux の RAPL で読みます。NPU 単体ではなく CPU パッケージ全体の値で、通常は root 権限が必要です
- キャッシュのキーにはモデル、ビルドフラグ、ドライバとコンパイラ (Apple は macOS) の版が入るので、どれかが変わると作り直します

## モデルのファイル (modelstore)

`github.com/shibukawa/hwnpu/modelstore` は、hwnpu のコマンドと同じ順でモデルのファイルを探して取ります: この機械のキャッシュ、取得元 (`HWNPU_MODEL_SOURCE`: フォルダ、WebDAV と HTTP、S3)、アプリが一緒に入れたモデル、Hugging Face の順です (docs/cache-sharing-design.md 7 節)。

```go
app := configdir.New("", "myapp") // github.com/shibukawa/configdir v1.1.0
s, err := modelstore.Open(modelstore.Options{
	Bundle: &app, // インストーラがアプリの資源フォルダの models/ に置いたもの (nil: なし)
	// Root: "" (HWNPU_CACHE、なければユーザーのキャッシュの hwnpu)、Sources: nil (環境のもの)
})
config, err := s.File(ctx, "LiquidAI/LFM3-1.7B", "config.json") // 手元のパス
weights, err := s.Weights(ctx, "LiquidAI/LFM3-1.7B")            // model.safetensors (分割ならその索引も)
```

- 設定はプロセスのもの: `Open` はプロセス全体に効きます
- `Options.Files` は、アプリ自身の定義のファイル (models.yaml の形式: モデル、用途、別名) です。hwnpu のもの、機械のもの、利用者のものの上に読みます
- 同梱のモデルは取得元と同じ並び (`original/OWNER/NAME/` と `hwnpu-lock.json`) で置き、その場で使います (コピーせずリンク)
- `Options.Limits` (nil: `HWNPU_CACHE_MAX_SIZE`、`HWNPU_CACHE_MIN_FREE`) でキャッシュの大きさを抑えます: ディスクの空きが `MinFree` (指定がなければ 10 GB) を切るダウンロードは `modelstore.ErrNoRoom` で失敗し、`MaxSize` があれば先に最近使っていない単位から消して場所を空けます。`s.Usage()`、`s.PlanPrune(o)`、`s.Prune(o)`、`s.Remove(units)` は `hwnpu cache` と同じ処理です

```go
s, err := modelstore.Open(modelstore.Options{Limits: &modelstore.Limits{MaxSize: 100e9, MinFree: 20e9}})
u, err := s.Usage()                                                     // u.Units: 種類、大きさ、最後に使った時刻
gone, err := s.Prune(modelstore.PruneOptions{OlderThan: 30 * 24 * time.Hour}) // 30 日使っていないもの
```

## モデルを組み合わせるアプリ (ai)

`github.com/shibukawa/hwnpu/ai` は、アプリが使うモデルを登録し、前もってダウンロードとコンパイルを済ませ、NPU で動かすためのパッケージ群です (v0: 形はまだ変わります。設計は [docs/app-api.md](docs/app-api.md))。hwnpu-chat、hwnpu-image、hwnpu-audio が初回にするのと同じ準備を同じ置き場に行うので、コマンドで準備したモデルはそのまま使えます。

```go
// 使うモデルを登録する。名前は models.yaml のモデル、ファミリー、用途 (hwnpu-chat list)
var (
	stt = speech.Transcribe("transcribe")
	tr  = text.Translate("translate").
		At(16, "plamo3translate:2b"). // 16 GB 以上の機械では 2B
		At(32, "plamo3translate:8b")  // 32 GB 以上の機械では 8B
	tts = speech.Speak("speech")

	app = ai.NewApp("interpreter", stt, tr, tts)
)

// 準備: 計画 (何も落とさない) を見てから、ダウンロード、変換、NPU 向けのコンパイル
plan, err := app.Plan(ctx)
fmt.Println(plan.DownloadBytes(), plan.MemoryGB(), plan.Licenses())
err = plan.Run(ctx, func(p ai.Progress) { fmt.Println(p.Use, p.Stage, p.Message) })

// 実行: ネットワークにも変換にも触らない。準備されていなければ ai.ErrNotPrepared
rt, err := app.Open(ctx)
defer rt.Close()
t, err := stt.Get(ctx, rt) // *speech.Transcriber
res, err := t.Transcribe(ctx, pcm16k, speech.TranscribeOptions{})
```

| パッケージ | 中身 |
|---|---|
| `ai` | `App` (登録)、`Use` (型のついた取っ手: `Get`、`Release`、機械のメモリごとの構成 `At` と `As`)、`Plan` (計画と準備)、`Runtime` |
| `ai/speech` | 文字起こし (Whisper)、読み上げ (LFM3-Audio)、音量の VAD、WAV の読み書き |
| `ai/text` | 翻訳 (PLaMo 3 Translate)、チャット |
| `ai/flow` | モデルの間のデータの渡し方: 塊の流れ `Stream` と、その道具 |
| `ai/aicli` | アプリのコマンドラインに足す kong のサブコマンド: `models` (一覧、`download`、`prepare`)、`status` |

- **機械のメモリごとの構成**: 用途の名前は、1 つだけ動かすときにその機械で動く一番大きなモデルを選びます。いくつも同時に動かすアプリは、`At(gb, name)` (そのメモリ以上の機械では別のモデル) と `As(gb)` (その機械として選ぶ) で、主になるモデルを大きく、補助のモデルを小さくできます。`APP models` が、選ばれたモデル、同時に動かす合計、機械のメモリを出します
- **1 つずつ動かす**: `use.Release(rt)` でモデルを下ろすと、続けて使う大きなモデル (画像の生成のあとに深度など) が同時にメモリに載りません
- **初回に自動で準備する**: `app.Open(ctx, ai.WithAutoPrepare(fn))` は、準備されていないモデルを最初の `Get` で準備します (コマンドと同じ動き)。ギガバイトのダウンロードになることがあります
- **ライセンス**: 同意の要るライセンスのモデルは、`plan.AcceptLicenses` を付けないと `*ai.LicenseError` になります。`plan.Licenses()` を見せて同意を取ってから付けてください

### モデルをつなぐ (ai/flow)

どの機能も、一括の形 (値を 1 つ渡して 1 つ受け取る) と流れの形 (`...Stream`) を持ちます。流れは `iter.Seq2` で、`range` で読み、`break`、エラー、ctx の取り消しで止まります。つなぎ方は、次の段をいつ始められるかで 3 つあります。

| つなぎ方 | 道具 |
|---|---|
| 前が全部終わってから (絵ができてから深度) | 一括の呼び出し、`flow.Last`、`flow.Of` |
| できたものから (文ごとに訳して読む、カメラの映像を追う) | `flow.Async` (段を重ねる)、`flow.Latest` (古い塊を捨てて最新だけ)、`flow.Hold` (最新を持っておく)、`flow.Tee` |
| 拍に合わせて (再生に間に合わせる音、画面に出すフレーム) | `flow.Metronome` (拍ごとの tick)、`flow.Pace` (塊をその時刻に渡す)、`flow.Live` (録音をマイクのように流す) |

```go
utterances := speech.NewVAD(speech.VADOptions{}).Detect(ctx, mic) // mic: flow.Stream[[]float32]、16 kHz
texts := t.TranscribeStream(ctx, utterances, speech.TranscribeOptions{})
en := translator.TranslateStream(ctx, texts, text.TranslateOptions{To: "en"})
audio := speaker.SpeakStream(ctx, flow.Async(ctx, en, 4), speech.SpeakOptions{})
for c, err := range audio { ... } // c.Data: 24 kHz の音声
```

### アプリのコマンドライン (ai/aicli)

```go
var cli struct {
	Run    RunCmd       `cmd:"" default:"withargs" help:"..."`
	Models aicli.Models `cmd:"" help:"The models the app uses: list, download, prepare."`
	Status aicli.Status `cmd:"" help:"This machine's NPU and what uses it."`
}
k := kong.Parse(&cli, aicli.Bind(ctx, app))
k.FatalIfErrorf(k.Run())
```

`APP models` は使うモデルと要るものの一覧、`APP models download` はダウンロードと変換 (NPU は要りません)、`APP models prepare` はさらに 1 回読み込んで NPU 向けにコンパイルします。`APP status` は `hwnpu status` と同じです。動く例は [samples/interpreter](samples/interpreter) (録音の音声を別の言語の音声に訳す) にあります。

- 今ある機能は音声と文だけです。画像 (深度、顔、生成など) は、まだコマンドからだけ使えます
- 実機で確かめたのは AMD の NPU (Ryzen AI 9 365) だけです。Mac と Intel では、同じ準備と読み込みをコマンドが行っていますが、`ai` を通しては動かしていません

## AMD (カーネルを直接動かす)

AMD の NPU は、モデルではなく IRON / mlir-aie でビルドしたカーネル (xclbin と命令列) を実行します。

```go
kr, err := dev.Kernel() // AMD 以外では ErrUnsupported
prog, err := kr.Load(ctx, hwnpu.ProgramSource{Path: "final.xclbin"})
k, err := prog.Kernel("MLIR_AIE")

instr, _ := dev.AllocBuffer(len(insts), hwnpu.WithBufferKind(hwnpu.BufferInstructions))
in, _ := dev.AllocBuffer(n)
out, _ := dev.AllocBuffer(n)
copy(instr.Bytes(), insts)
instr.SyncToDevice(0, len(insts)) // AMD の NPU はキャッシュ非コヒーレント
in.SyncToDevice(0, n)

run, _ := k.NewRun()
err = run.SetUint64(0, 3).SetBuffer(1, instr).SetUint32(2, uint32(len(insts)/4)).
	SetBuffer(3, in).SetBuffer(4, tmp).SetBuffer(5, out).
	Run(ctx)
out.SyncFromDevice(0, n)
```

- 引数の番号、オフセット、型は xclbin のメタデータから読みます (`k.Args()`)
- `/dev/accel/accelN` に読み書きできる権限 (`render` グループ) が必要です。Linux 6.14 以降のカーネル (amdxdna) を前提にしています
- 電力ポリシーはハードウェアコンテキストの優先度 (QoS) に反映します。デバイス全体の電力モードは root 専用なので変更しません
- 電力は NPU のセンサー値 (瞬間値) を、なければ amdgpu ドライバが出すパッケージ全体の電力 (PPT、root 不要) を読みます。最初の `PowerSample` から 50 ms ごとに読んで積分し、電力量にします。NPU のセンサーは AMD の DKMS 版のドライバにしかなく、Linux 7.0 の標準カーネルではパッケージ全体の値になります (`Scope` が `package`)

Strix Point 以降 (XDNA2) では、IRON がビルドした full ELF (`design.elf`、グラフ全体を 1 つのハードウェアコンテキストと 1 つのコマンドで流すもの) も読み込めます。カーネル名は `main:sequence` の形で、引数はすべてバッファです。呼び出しごとの値は `run.Scratchpad()` に書きます。

## Apple での注意

- Core ML は「Neural Engine だけ」を指定できません。hwnpu は CPU と Neural Engine だけを許可し、`MLComputePlan` (macOS 14.4 以降) の推定で全演算が Neural Engine に置かれることを確認します。推定なので、実行時の配置までは保証されません
- Neural Engine は fp16 前提です。fp32 のモデルや小さすぎるモデルは CPU に置かれます。coremltools で fp16 の ML Program に変換してください
- 小さすぎるモデルは、`tools/coreml/ane_pad.py` でバッチ方向にパディングすると Neural Engine に載せられます。ただし小さなモデルでは遅くなることがあるので、詳しくは [MODEL_ARCHITECTURE.ja.md](MODEL_ARCHITECTURE.ja.md) を読んでから使ってください
- `hwnpu-chat` と `hwnpu-image` は Mac では Core AI だけで動きます (macOS 27 が要ります。Core ML は使いません。`hwnpu-image --cpu` は確かめ用に Core ML の書き出しを CPU のインタプリタで動かします)。ほかのコマンドも macOS 27 では同じ変換済みのモデルを Core AI で動かせます (`--backend coreai` か `HWNPU_BACKEND=coreai`。既定は Core ML)。結果と速さは同じで、違いはコンパイルした結果をディスクが減るまで持ち続けることです (Core ML は約 4 GiB を超えると 2 回目以降もコンパイルし直します)。使えるコマンドと費用は [INSTALL_APPLE.ja.md](INSTALL_APPLE.ja.md) の 5 節にあります

## Intel での注意

- Intel NPU ドライバ (`intel-level-zero-npu`、`intel-driver-compiler-npu`、`intel-fw-npu`) と Level Zero loader (`libze1`) が必要です。loader の場所は環境変数 `HWNPU_ZE_LOADER` で指定することもできます
- Linux では `/dev/accel/accel0` に読み書きできる権限 (`render` グループ) が必要です
- `GraphSource{Path: "model.xml"}` で OpenVINO IR (隣の `.bin` も読みます) を、`GraphSource{Blob: ...}` でコンパイル済み blob を渡せます。コンパイラに渡す追加のオプションは `BuildFlags` に `--config KEY="VALUE"` の形で書きます
- NPU コンパイラが対応しない演算を含むモデルは、配置方針にかかわらず `ErrNotOnNPU` になり、原因の演算とビルドログが返ります
- Intel の NPU に向けて OpenVINO IR に変換するのは、hwnpu-chat のモデル (LFM3 とその VL・Audio・d1、Gemma 4、Qwen3.5、Qwen3、PLaMo 3、PLaMo-Embedding)、Anima、`hwnpu-audio` の transcribe (Whisper)、tts、generate (Magenta RealTime 2、MOSS-SoundEffect)、pitch、notes、`hwnpu-image` の depth (DA3、MoGe-2)、face-detect (YuNet)、segment (EfficientSAM のエンコーダ)、detect、pose、ocr です。Chroma、FLUX.2 klein、SigLIP 2 (label) は AMD の NPU と Apple の Neural Engine だけです (Intel はまだ)。実機で動かしたのは LFM3、PLaMo 3 Translate、PLaMo-Embedding、Anima だけで、ほかは OpenVINO の CPU (`HWNPU_BACKEND=openvino`、[INSTALL_INTEL.ja.md](INSTALL_INTEL.ja.md) の「OpenVINO で切り分ける」) で変換から結果までを確かめた段階です ([docs/design.md](docs/design.md) 16.166)

## 診断

```bash
go run ./cmd/hwnpu-dev info -model internal/fixtures/coreml/linear_relu_fp16.mlpackage
```

## テスト

```bash
CGO_ENABLED=0 go test ./...
```

- C コンパイラ (`cc`) があれば、テストは偽の Level Zero ドライバ (`testdata/zemock`) をビルドして Intel バックエンドを確かめます
- テスト用モデルは `internal/fixtures` にあり、`gen_models.py` で再生成できます (numpy と、Core ML 分は coremltools が必要です)
- Level Zero の構造体の基準値は `testdata/abi/levelzero_abi.c`、amdxdna と xclbin と ERT の基準値は `testdata/abi/xdna_abi.c` で再生成できます
- AMD のバックエンドは、Linux の amdxdna ドライバの振る舞いを模した Go 製の偽ドライバ (`internal/xdna/xdnasim`) でテストします
