# Apple の Neural Engine を hwnpu で使うための準備

Apple シリコンの Mac で、hwnpu から Neural Engine を使う手順です。機械の確認から、動作確認レポートの実行、モデルの用意までを扱います。

> **現状。** Apple バックエンドは、hwnpu の開発と計測に使ってきたものです（M3、macOS 27）。[MODEL.ja.md](MODEL.ja.md) の数値はここで測りました。

English: [INSTALL_APPLE.md](INSTALL_APPLE.md)

## 対応プラットフォーム

| プラットフォーム | 対応 | 補足 |
|---|---|---|
| Apple シリコンの macOS 14.4 以降 | 対応 | Core ML の `MLComputePlan` で、モデルが Neural Engine で動くことを hwnpu が確認できる。 |
| Apple シリコンの macOS 13〜14.3 | 一部 | モデルの読み込みと実行はできるが、配置を確認できない（`PlacementReport.Verified` が false）。厳格モードでは拒否される。 |
| Intel の Mac | 該当なし | Neural Engine がない。Core ML はすべて CPU で動かす。 |
| iOS、iPadOS | **非対応** | hwnpu は macOS 用の Go ライブラリで、iOS 向けにはビルドしない。 |
| Linux、Windows | 該当なし | Neural Engine は macOS の Core ML からしか使えない。 |

## hwnpu が必要とするもの

hwnpu は purego と Objective-C ランタイムを通して Core ML を呼びます。追加で入れるものはありません。Core ML は macOS の一部です。

| 部品 | hwnpu に必要か | 補足 |
|---|---|---|
| Apple シリコンの Mac | 必要 | Apple シリコンにはすべて Neural Engine がある。 |
| macOS 14.4 以降 | 必要 | 配置の確認に必要。読み込みだけなら 13 以降。 |
| Go | 必要 | 1.27 以降（`go.mod` を参照）。`CGO_ENABLED=0` で動くので、Xcode も C のツールチェーンも要らない。 |
| Xcode または Command Line Tools | 不要 | |
| Python と coremltools | `hwnpu-convert` が知らないモデルにだけ必要 | 対応する LLM のチェックポイント（今は LFM3）は `cmd/hwnpu-convert` で Python なしに変換できる（手順 4）。 |
| `sudo` | 電力の計測にだけ必要 | `cmd/powerbench` が使う `powermetrics` に root が要る。 |

Intel や AMD と違い、Core ML には「Neural Engine だけ」を指定できません。hwnpu は CPU と Neural Engine だけを許可して GPU を使わず、計算計画を確認して CPU に置かれる演算を報告します。

## 1. 機械の確認

```bash
sysctl hw.model && sw_vers
```

Apple のチップ（M1 以降）を載せた Mac なら使えます。すべての確認を行うには macOS 14.4 以降が必要です。

## 2. hwnpu の入手

```bash
go get github.com/shibukawa/hwnpu
```

以下のツールを動かすにはリポジトリをクローンしてください。デバイスの一覧を出します。

```bash
go run ./cmd/hwnpuinfo
```

Neural Engine は `backend: coreml` として表示されます。

## 3. 動作確認

レポートは組み込みのテスト用モデルを実行し、配置を確認して時間を測ります。

```bash
go run ./cmd/report -o report.md
```

fp16 のモデルは、すべての演算が Neural Engine に置かれているはずです。2 つのモデルは拒否されるために入っています。fp32 のものと、バッチ 1 の小さな MLP（Core ML が CPU に置く）です。レポートは、厳格な配置（API の `WithPlacement(hwnpu.PlacementStrict)`、既定）がこれらを拒否することを確かめます。黙って CPU に落ちては困るアプリケーションではこの既定のままにしてください。同じ MLP をバッチ 64 に広げたものは Neural Engine で動きます。

呼び出し 1 回あたりのエネルギーを測るには、`powermetrics` のために root が要ります。

```bash
go run ./cmd/powerbench -model 'path/to/*.mlpackage' -o power.md
```

最初に sudo のパスワードを一度だけ聞き、Neural Engine と CPU（`-gpu` を付ければ GPU も）を比べます。M3 での結果と読み方は [MODEL.ja.md](MODEL.ja.md) にあります。

生成したモデルで Neural Engine を測るには、npuprobe を使います。LLM の生成と同じ形のモデルを Go で作り、0.5B と 1B の重みで fp16、int8、lut4、int4 の速さを Neural Engine、CPU、GPU で測ります (`bench`)。次に形式ごとにモデルを大きくしながら Neural Engine から外れる境目を二分探索で絞り、境目のすぐ下のモデルを複数同時に読み込んで、上限がモデル 1 つごとか合計かを見分けます (`limit`)。Python もモデルのファイルも root も要りません。40 分ほどかかり、最大のモデルの 2 倍ほどの空きディスクが要ります (既定ではメモリの 4 分の 1 まで試し、`-max-gb` で変えられます)。結果は `npuprobe-<機種>-<日時>.md` に書き出します。クローンしなくても動きます。

```bash
go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest
```

1 トークンあたりのエネルギーは、`go run ./cmd/powerbench -decode 8,16 -gpu` が同じモデルを powermetrics で測ります。結果の読み方とこれまでの結果は [PROBE.ja.md](PROBE.ja.md) にあります。

## 4. モデルの用意

Core ML は `.mlpackage`（ML Program）、`.mlmodel`、コンパイル済みの `.mlmodelc` を受け付けます。`.mlpackage` は最初の読み込み時に `.mlmodelc` へコンパイルされます。`WithCacheDir` を指定するとその結果が保存され、次回からはコンパイルを省けます。

Neural Engine がモデルに求めること:

- **fp16 であること。** fp32 の演算は CPU で動きます。`compute_precision=FLOAT16` で変換してください。
- **形状が固定であること。** 動的な形状は CPU に落ちます。系列長やバッチの大きさごとに 1 つずつモデルを書き出してください。
- **1 回の呼び出しに十分な計算量があること。** 小さなモデルは fp16 でも CPU に置かれます。`tools/coreml/ane_pad.py` はバッチ次元を広げて Neural Engine に移るまで探します。ただし、得になるのは呼び出し 1 回あたり約 200 MFLOP 以上なので、先に [MODEL.ja.md](MODEL.ja.md) の電力の結果を読んでください。
- **重みがモデル 1 つあたり 1 GiB 未満であること。** M3 では、境目は fp16、int8、lut4 のどれでも保存された重みで 0.99〜1.02 GiB の間にあり（`cmd/npuprobe limit`）、それを超えるモデルは丸ごと CPU に置かれました。上限はモデルごとで、0.91 GiB のモデルを 3 つ同時に読み込んでもすべて Neural Engine に載り、速さも変わりませんでした。大きなネットワークは複数のモデルに分ければ載せられます。全速で動くのは 4 ビットの表引き方式だけです。

本線は、bf16 のチェックポイントから Go の `hwnpu-convert` で変換することです。coremltools は、hwnpu-convert が構造を知らないモデルのための回避路です。

- **LLM のチェックポイントは hwnpu-convert（Go だけ、Python 不要）で。** `go run ./cmd/hwnpu-convert -config config.json -weights model.safetensors -o out.mlpackage -seq 32` が、既知の構造について `config.json` からグラフを組み立て、重みを 4 ビットの表引きにします。今は LFM3 に対応しています。Llama や Qwen 系は `internal/llm` に構造の定義を足せば対応できます。元のチェックポイントの選び方と Neural Engine が受け付ける形式は [MODEL.ja.md](MODEL.ja.md) にまとめてあります。
- **それ以外は coremltools（Python）で。** どんなグラフでも受ける汎用の変換器は hwnpu の範囲外なので、対応する LLM の構造ではないモデル（MiniLM のようなエンコーダ、画像モデルなど）は、PyTorch から `ct.convert(..., convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14)` と固定の入力形状で変換します。`tools/coreml/convert_minilm.py` と `tools/coreml/gen_sweep.py` が、ベンチマーク用に書いた実例です。

## 困ったとき

| 症状 | 原因と対処 |
|---|---|
| `PlacementReport` に CPU の演算が並ぶ | fp32 の演算、動的な形状、`gather`（埋め込みの表引きは必ず CPU）、または Neural Engine には小さすぎるモデル。fp16 と固定形状で変換し直し、埋め込みの表引きはホスト側で行う。 |
| `Verified` が false | macOS が 14.4 より古いか、モデルが ML Program ではなく古い `neuralnetwork` 形式。macOS を更新し、ML Program として変換し直す。 |
| 読み込みがエラーを返さずに落ちる | 手書きの ML Program で値の名前にドットがあるか、パッケージが壊れている。coremltools と `hwnpu-convert` が書く名前は問題ない。 |
| fp16 なのにモデル全体が CPU で動く | 重みが大きすぎる（モデル 1 つあたり 1 GiB 以上。複数のモデルに分ける）か、Neural Engine が受け付けない並びの重み圧縮（表引きは行全体を 16 行ごとに 1 表、ブロックごとの int4 は不可）。 |
| 結果が PyTorch と数パーセントずれる | Neural Engine の行列積には約 2.5e-4 の絶対誤差の床があり、出力の小さい層が精度を失う。LFM3 では `hwnpu-convert -post-scale` が回避する。[MODEL.ja.md](MODEL.ja.md) を参照。 |
| CPU や GPU との比較がしたい | 読み込み時に `HWNPU_COREML_UNITS=cpu` か `gpu` を設定する（計測専用で、API ではない）。 |
| `powerbench` が `Password:` で止まる | `powermetrics` に root が要る。sudo のパスワードを一度入れれば、あとはツールが資格を保つ。 |
