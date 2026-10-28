# hwnpu

CPU 内蔵の NPU を cgo なしで Go から使うためのライブラリです。軽量モデルを低消費電力で動かすことを目的にしています。

- 既定の厳格モードでは、NPU だけで完結しないモデルを実行前にエラーにします。CPU や GPU への黙ったフォールバックはしません
- 依存は [purego](https://github.com/ebitengine/purego) と golang.org/x/sys だけです

| プラットフォーム | バックエンド | 状態 |
|---|---|---|
| macOS (Apple Silicon) | Core ML (GPU を除外し、MLComputePlan で ANE 配置を検証) | M1 実装済み |
| Linux / Windows (Intel Core Ultra) | Level Zero (OpenVINO IR のドライバ内コンパイル、コンパイル済み blob) | M2 実装済み。実機では未検証 |
| Linux (AMD Ryzen AI) | amdxdna ioctl (IRON / mlir-aie の xclbin を実行) | M4 実装済み。実機では未検証 |
| Windows (AMD Ryzen AI) | なし | 非対応 |
| WSL2 | なし | 非対応。WSL2 からは Intel と AMD のどちらの NPU も見えない |

各 NPU が扱えるデータ形式とモデルの用意の仕方は [MODEL.ja.md](MODEL.ja.md)（[English](MODEL.md)）にあります。

導入手順は、Apple が [INSTALL_APPLE.ja.md](INSTALL_APPLE.ja.md)（[English](INSTALL_APPLE.md)）、Intel が [INSTALL_INTEL.ja.md](INSTALL_INTEL.ja.md)（[English](INSTALL_INTEL.md)）、AMD が [INSTALL_AMD.ja.md](INSTALL_AMD.ja.md)（[English](INSTALL_AMD.md)）にあります。

設計は [docs/design.md](docs/design.md) にあります。

## 使い方

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

## AMD での使い方

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
- 電力は NPU のセンサー値 (瞬間値) を読み、`PowerSample` の呼び出し間で積分して電力量にします

## Apple での注意

- Core ML は「Neural Engine だけ」を指定できません。hwnpu は CPU と Neural Engine だけを許可し、`MLComputePlan` (macOS 14.4 以降) の推定で全演算が Neural Engine に置かれることを確認します。推定なので、実行時の配置までは保証されません
- Neural Engine は fp16 前提です。fp32 のモデルや小さすぎるモデルは CPU に置かれます。coremltools で fp16 の ML Program に変換してください
- 小さすぎるモデルは、`tools/coreml/ane_pad.py` でバッチ方向にパディングすると Neural Engine に載せられます。ただし小さなモデルでは遅くなることがあるので、詳しくは [MODEL.ja.md](MODEL.ja.md) を読んでから使ってください

## 動作確認レポート

手元の NPU で hwnpu が動くかを確かめ、結果をそのまま貼り付けられる Markdown で出力します。テスト用モデルはプログラムに埋め込んであるので、リポジトリのどこから実行しても動きます。

```bash
go run ./cmd/report -o report.md
```

- 見つかったデバイスごとに、そのバックエンド向けのモデルを厳格モードでコンパイルし、推論結果を fp32 の参照値と比べ、実行時間を測ります
- Level Zero loader の読み込み結果も載るので、Intel NPU が見つからないときの原因が分かります
- `-json report.json` で JSON も出力します。`-n` で計測の回数を、`-power low` などで電力ポリシーを、`-cache DIR` でコンパイルキャッシュを指定できます
- AMD では `-xclbin final.xclbin -insts insts.bin` を渡すと、mlir-aie の `programming_examples/basic/passthrough_dmas` をビルドした設計を実行して出力を照合します (引数の並びは opcode, instr, ninstr, 入力, 未使用, 出力)
- 失敗があると終了コード 2、NPU が見つからないと 1 で終わります

macOS では `go run ./cmd/powerbench` で、Core ML モデルの呼び出し 1 回あたりのエネルギーを powermetrics で測れます。最初に sudo のパスワードを一度だけ聞きます。詳しくは [MODEL.ja.md](MODEL.ja.md) を参照してください。

`go run ./cmd/hwnpu-convert` は、Hugging Face のチェックポイント (safetensors か GGUF) を Python なしで Neural Engine 用の `.mlpackage` に変換します。今は LFM3 のみ対応です。元モデルの選び方と Neural Engine の制約は [MODEL.ja.md](MODEL.ja.md) にまとめてあります。

`go run ./cmd/npuprobe` は、生成したモデルで NPU を測ります。LLM が 1 トークンを生成するのと同じ形のモデルを Go で作り、まず 0.5B と 1B の重みで形式ごと (Apple は fp16、int8、lut4、int4、Intel は fp16、int8、int4) の速さを測ります (`bench`)。次にモデルを大きくしながら NPU から外れる境目を二分探索で絞り、境目のすぐ下のモデルを複数同時に読み込んで、上限がモデルごとか合計かを確かめます (`limit`)。Python もモデルのファイルも root も要らず、クローンしなくても次の 1 行で動きます。結果は `npuprobe-<機種>-<日時>.md` に書き出します。

```bash
go run github.com/shibukawa/hwnpu/cmd/npuprobe@llm-convert
```

`bench` か `limit` を付けると片方だけを行います。Apple で 1 トークンあたりのエネルギーも測るには `go run ./cmd/powerbench -decode 8,16 -gpu` を使います (sudo が要ります)。

Go の入っていない Windows 機で試すときは、Go のある機械でビルドした実行ファイルを持っていきます。

```bash
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o report.exe ./cmd/report
```

## 診断

```bash
go run ./cmd/hwnpuinfo -model internal/fixtures/coreml/linear_relu_fp16.mlpackage
```

## Intel での注意

- Intel NPU ドライバ (`intel-level-zero-npu`、`intel-driver-compiler-npu`、`intel-fw-npu`) と Level Zero loader (`libze1`) が必要です。loader の場所は環境変数 `HWNPU_ZE_LOADER` で指定することもできます
- Linux では `/dev/accel/accel0` に読み書きできる権限 (`render` グループ) が必要です
- `GraphSource{Path: "model.xml"}` で OpenVINO IR (隣の `.bin` も読みます) を、`GraphSource{Blob: ...}` でコンパイル済み blob を渡せます。コンパイラに渡す追加のオプションは `BuildFlags` に `--config KEY="VALUE"` の形で書きます
- NPU コンパイラが対応しない演算を含むモデルは、配置方針にかかわらず `ErrNotOnNPU` になり、原因の演算とビルドログが返ります

## テスト

```bash
CGO_ENABLED=0 go test ./...
```

- C コンパイラ (`cc`) があれば、テストは偽の Level Zero ドライバ (`testdata/zemock`) をビルドして Intel バックエンドを確かめます
- テスト用モデルは `internal/fixtures` にあり、`gen_models.py` で再生成できます (numpy と、Core ML 分は coremltools が必要です)
- Level Zero の構造体の基準値は `testdata/abi/levelzero_abi.c`、amdxdna と xclbin と ERT の基準値は `testdata/abi/xdna_abi.c` で再生成できます
- AMD のバックエンドは、Linux の amdxdna ドライバの振る舞いを模した Go 製の偽ドライバ (`internal/xdna/xdnasim`) でテストします
