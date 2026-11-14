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
- `hwnpu-image` は Mac では Core AI だけで動きます (macOS 27 が要ります。Core ML は使いません。`--cpu` は確かめ用に Core ML の書き出しを CPU のインタプリタで動かします)。ほかのコマンドも macOS 27 では同じ変換済みのモデルを Core AI で動かせます (`--backend coreai` か `HWNPU_BACKEND=coreai`。既定は Core ML)。結果と速さは同じで、違いはコンパイルした結果をディスクが減るまで持ち続けることです (Core ML は約 4 GiB を超えると 2 回目以降もコンパイルし直します)。使えるコマンドと費用は [INSTALL_APPLE.ja.md](INSTALL_APPLE.ja.md) の 5 節にあります

## Intel での注意

- Intel NPU ドライバ (`intel-level-zero-npu`、`intel-driver-compiler-npu`、`intel-fw-npu`) と Level Zero loader (`libze1`) が必要です。loader の場所は環境変数 `HWNPU_ZE_LOADER` で指定することもできます
- Linux では `/dev/accel/accel0` に読み書きできる権限 (`render` グループ) が必要です
- `GraphSource{Path: "model.xml"}` で OpenVINO IR (隣の `.bin` も読みます) を、`GraphSource{Blob: ...}` でコンパイル済み blob を渡せます。コンパイラに渡す追加のオプションは `BuildFlags` に `--config KEY="VALUE"` の形で書きます
- NPU コンパイラが対応しない演算を含むモデルは、配置方針にかかわらず `ErrNotOnNPU` になり、原因の演算とビルドログが返ります
- Intel の NPU に向けて OpenVINO IR に変換するのは、hwnpu-chat のモデル (LFM3 とその VL・Audio・d1、Gemma 4、Qwen3.5、Qwen3、PLaMo 3、PLaMo-Embedding)、Anima、Sana、`hwnpu-audio` の transcribe (Whisper)、tts、generate (Magenta RealTime 2、MOSS-SoundEffect)、pitch、notes、`hwnpu-image` の depth (DA3、MoGe-2)、face-detect (YuNet)、segment (EfficientSAM のエンコーダ)、detect、pose、ocr です。Chroma、SigLIP 2 (label)、Qwen-Image は AMD の NPU と Apple の Neural Engine だけです (Intel はまだ)。実機で動かしたのは LFM3、PLaMo 3 Translate、PLaMo-Embedding、Anima だけで、ほかは OpenVINO の CPU (`HWNPU_BACKEND=openvino`、[INSTALL_INTEL.ja.md](INSTALL_INTEL.ja.md) の「OpenVINO で切り分ける」) で変換から結果までを確かめた段階です ([docs/design.md](docs/design.md) 16.166)

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
