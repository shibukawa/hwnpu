# モデルとデータ形式

各 NPU が何の型で計算するか、hwnpu が何を受け取るか、モデルをどう用意するかをまとめます。各社の資料に基づく事実にはリンクを付けました。hwnpu 自身のテストで観測したことは、その旨を明記しています。

English: [MODEL.md](MODEL.md)

## 一覧

| | Intel NPU | Apple Neural Engine | AMD XDNA NPU |
|---|---|---|---|
| hwnpu のバックエンド | `levelzero` | `coreml` | `xdna` |
| hwnpu が読み込むもの | OpenVINO IR（`.xml` と `.bin`）、または同じドライバでコンパイルした blob | Core ML モデル（`.mlpackage`、`.mlmodel`、`.mlmodelc`） | IRON / mlir-aie でビルドした xclbin と命令列（モデルではなくカーネル） |
| ネイティブの計算精度 | fp16。量子化モデルなら int8 | fp16。新しいチップ（A17 Pro、M4）では int8 同士の計算も | int8、int16、bf16。XDNA2 ではブロック浮動小数点（BFP16）も |
| fp32 のモデル | 受け付けるが fp16 で実行される | CPU に置かれる（厳格モードでは拒否） | AMD のツールが bf16 に変換する。fp32 はネイティブではなく模擬 |
| 形状 | 固定のみ | 固定を推奨。hwnpu は既定の形状を使う | カーネルの作成時に決まる |
| デバイス向けのコンパイル | 読み込み時にドライバ内コンパイラが行う（`WithCacheDir` で保存） | 読み込み時に Core ML が行う（`WithCacheDir` で保存） | 事前に mlir-aie と Peano で行う |
| 配置の確認 | コンパイル結果で判定し、未対応の演算を列挙 | `MLComputePlan`（推定） | 常に NPU |

## Intel NPU

**受け付ける形式**

- OpenVINO IR v11。hwnpu は `.xml` と隣の `.bin` を読み、OpenVINO の NPU プラグインと同じ形でドライバ内コンパイラに渡します。
- 同じ版のドライバとコンパイラでコンパイルした blob。blob はドライバの版が変わると使えないので、配布するのは IR にして、blob は `WithCacheDir` に保存させてください。

**データ型**

- NPU は fp16 で計算します。OpenVINO の NPU プラグインは fp32、fp16、量子化済みの u8 のモデルを受け付け、量子化モデルは int8 か fp16 と int8 の混在です（[OpenVINO NPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)）。
- 速度と電力の面では、NNCF による int8 量子化が最も有利です。LLM 向けには、OpenVINO GenAI が NPU 用の低ビットの重み圧縮を案内しています。お使いのドライバが対応する形式は、その NPU ガイドで確かめてください。
- `ArgInfo.DType` は、バッファに入れるべき型（デバイス上の精度）です。`ArgInfo.ModelDType` はモデルが宣言した型で、デバイスが精度を変換するときは両者が異なります。

| ArgInfo.DType | バイト数 | 補足 |
|---|---|---|
| `F32`、`F16`、`BF16`、`F64` | 4、2、2、8 | fp16 は `hwnpu.EncodeF16` / `DecodeF16` で変換できる。 |
| `U8`、`I8`、`I16`、`I32`、`I64` | 1、1、2、4、8 | |

**制約**

- 形状は固定でなければなりません。動的な次元を含む IR は hwnpu が拒否します。
- モデルの opset は、ドライバ内コンパイラが対応している必要があります。コンパイラの版は `hwnpuinfo` で確かめられます。
- コンパイラが対応しない演算があると、読み込みは `ErrNotOnNPU` で失敗し、その演算が列挙されます。CPU に回ることはありません。

**変換の仕方**

- ONNX、PyTorch、TensorFlow からは、OpenVINO の `ovc` コマンドか `openvino.convert_model` で変換し、`openvino.save_model` で保存します（保存時の既定は fp16 への圧縮です）。
- 入力の形状は変換時に固定します（`ovc` なら `--input "x[1,3,224,224]"` のように指定）。
- 量子化には NNCF の学習後量子化を使います。

## Apple Neural Engine（Core ML）

**受け付ける形式**

- `.mlpackage`（ML Program）、`.mlmodel`、コンパイル済みの `.mlmodelc`。**ML Program 形式**を使ってください。配置を確認できるのは ML Program だけで、確認できないモデルは厳格モードで拒否されます。

**データ型**

- Neural Engine は fp16 で計算します。hwnpu のテストでは、fp32 の ML Program はすべて CPU に置かれました。そのため、fp32 のモデルは厳格モードで拒否されます。
- 重みは圧縮できます。Core ML は、重みについては int8 と int4 の線形量子化と、1〜8 ビットのパレット化に対応しています。活性化は int8 に対応しています。重みと活性化をともに 8 ビットにする W8A8 は、A17 Pro や M4 などの新しいハードウェアで、より速い int8 同士の計算経路を使います（[coremltools の最適化の概要](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)）。
- hwnpu が扱える入出力は多次元配列だけです（画像や辞書の特徴量は扱えません）。

| Core ML の多次元配列の型 | ArgInfo.DType |
|---|---|
| Float16 | `F16` |
| Float32 | `F32` |
| Float64 | `F64` |
| Int32 | `I32` |
| Int8 | `I8` |

**制約**

- hwnpu は各入力の既定の形状を使います。可変の形状には対応していません。
- hwnpu のテスト（macOS 27）で観測したこととして、小さなモデルは fp16 でも CPU に置かれます。入力 (64, 512)、出力 512 の全結合層は Neural Engine で動きました。(32, 256) 以下は CPU に置かれ、32×32 で最大 256 チャネルの 1×1 畳み込みも CPU でした。小さな入力を Neural Engine で動かしたいときは、まとめてバッチにしてください。
- 配置は `MLComputePlan`（macOS 14.4 以降）による推定です。予測が実際にどこで動いたかを、Core ML は知らせません。

**小さなモデルをパディングして Neural Engine に載せる**

Core ML は、固定の閾値ではなく、モデル全体の計算量から配置を決めます。hwnpu のテスト（macOS 27）では、次のようになりました。

| モデル（fp16） | 配置 |
|---|---|
| 全結合 256→256、バッチ 1〜40 | CPU |
| 全結合 256→256、バッチ 48 以上 | Neural Engine（バッチ 56 では全結合が Neural Engine、ReLU が CPU に分かれた） |
| 全結合 64→64、バッチ 4096 まで | すべて CPU |
| 全結合、バッチ 1、幅 4096 まで | CPU（行列とベクトルの積になるため） |
| 3 層 MLP、幅 128、バッチ 1〜48 | CPU |
| 3 層 MLP、幅 128、バッチ 64 以上 | Neural Engine |
| 3×3 畳み込み、32 チャネル、32×32、バッチ 1 と 8 | CPU。バッチ 64 なら Neural Engine |

モデルを Neural Engine に移すのは、**バッチ**（先頭の次元）を大きくすることです。バッチをゼロの行で埋めても、各行を独立に処理するモデルなら結果は変わりません。推論用のモデルの多くはこれに当たります。**特徴量**の次元を、重みをゼロで埋めて広げる方法もあります（小さな行列を大きなアフィン変換に埋め込むのと同じ考え方です）。全結合と ReLU の連なりなら結果は変わりませんが、softmax や LayerNorm のような正規化の結果は変わります。また、試したモデルでは、この方法で Neural Engine に移ったものはありませんでした。

`tools/coreml/ane_pad.py` は、ML Program をより大きなバッチで作り直します。小さいバッチから順に試し、全演算が Neural Engine に載った最小のバッチを採用します。

```bash
python tools/coreml/ane_pad.py model.mlpackage model_ane.mlpackage --verify
```

元のバッチとパディング後のバッチは、モデルのメタデータに記録されます。hwnpu はそれを読み、`ArgInfo.Dims` と `Bytes` は元のバッチのまま見せます。実際に動く形状は `ArgInfo.PaddedDims` に入ります。利用者の行はゼロ埋めしたバッファに書き写して実行し、結果は元の行だけを書き戻します。

**採用する前に測ってください。** 上の 3 層 MLP では、1 行の計算が CPU で 73 マイクロ秒、64 行にパディングして Neural Engine で動かすと 280 マイクロ秒でした。64 行を CPU で計算しても 62 マイクロ秒です（cmd/powerbench で hwnpu 経由で測定）。このモデルでは Neural Engine を起動する固定費が計算そのものより大きく、遅延の面では Core ML が CPU を選んだのが正解でした。遅くても、Neural Engine のほうが消費するエネルギーが少なければ価値があります。遅延だけでなく、エネルギーを測ってください。

```bash
go run ./cmd/powerbench
```

最初に sudo のパスワードを一度だけ聞きます。そのあと、各モデルを動かしている間に powermetrics で CPU・GPU・ANE の電力を測り、アイドル時の電力を差し引いて、呼び出し 1 回あたりのエネルギー（mJ/call ＝ 正味の電力 ÷ 毎秒の呼び出し回数）を表示します。既定では組み込みの `mlp_small` の 2 モデルを比べます。自分のモデルは `-model a.mlpackage -model b.mlpackage` で渡せます。`-rate 100` を付けると、一定の頻度（毎秒 100 回）で呼び出して比べます。小さなモデルの実際の使われ方に近いのはこちらです。`-json` と `-raw` で、結果と powermetrics の生の出力を残せます。powermetrics の値は推定値なので、比べるのは同じ Mac の上だけにしてください。

M3 の MacBook Air で測ったところ、`mlp_small` のパディングは得になりませんでした。呼び出しを連続させた場合、CPU は 1 行で 0.32 mJ、64 行でも 0.33 mJ でした。Neural Engine は 64 行で 0.78 mJ で、その内訳は CPU が 3.3 W、ANE が 1.5 W です。毎秒 100 回の場合、ANE の電力だけで呼び出し 1 回あたり 0.6 mJ になりました。呼び出しのたびに ANE が起動と停止を繰り返すためです。このモデルでは、Core ML の呼び出し 1 回にかかる固定費（約 55 マイクロ秒）が計算そのものより大きく、CPU は遅延だけでなくエネルギーでも有利です。パディングを試す価値があるのは、計算がこの固定費を上回るモデルだけです。そうしたモデルは `-model` で測ってください。

パディングが本当に得になるのは、埋めた行に実際の仕事を載せる場合です。ゼロで埋める代わりに、複数の要求を集めてバッチを満たしてください。

**変換の仕方**

- PyTorch から（`torch.export` かトレース経由）、または MIL のビルダーから `coremltools.convert(..., convert_to="mlprogram", compute_precision=coremltools.precision.FLOAT16)` で変換します。
- 入力の形状は固定にし、入出力はできるだけ fp16 にします。
- 重みの圧縮には `coremltools.optimize` を使います。
- coremltools には保守されている ONNX 変換器がありません。元の PyTorch モデルから変換するか、ONNX を PyTorch 経由で変換してください。

### Neural Engine が得になるモデルの大きさ

M3 の MacBook Air（macOS 27）で `cmd/powerbench` を使って測りました。各モデルを間を空けずに呼び続け、Core ML を CPU だけに制限した場合と、CPU と Neural Engine を使う場合を比べています。エネルギーは、アイドル時を超えた分の電力を毎秒の呼び出し回数で割った値です（各計測 10 秒、2 周）。

| モデル | 1 回の計算量 | CPU のみ: 遅延 / エネルギー | Neural Engine: 遅延 / エネルギー | CPU に対する NPU のエネルギー |
|---|---|---|---|---|
| エンコーダー d128、系列長 32、1 層 | 13 MFLOP | 80 µs / 0.41 mJ | 173 µs / 0.91 mJ | 2.2 倍 |
| エンコーダー d256、系列長 64、2 層 | 210 MFLOP | 218 µs / 1.58 mJ | 206 µs / 1.49 mJ | 0.94 倍 |
| all-MiniLM-L6-v2、系列長 32 | 0.69 GFLOP | 790 µs / 4.5 mJ | 527 µs / 2.6 mJ | 0.57 倍 |
| エンコーダー d384、系列長 128、2 層 | 0.96 GFLOP | 579 µs / 5.7 mJ | 305 µs / 2.2 mJ | 0.38 倍 |
| all-MiniLM-L6-v2、系列長 128 | 2.9 GFLOP | 1.90 ms / 16.5 mJ | 0.74 ms / 4.3 mJ | 0.26 倍 |
| エンコーダー d768、系列長 128、4 層 | 7.4 GFLOP | 3.46 ms / 31.6 mJ | 1.15 ms / 7.4 mJ | 0.23 倍 |
| エンコーダー d1024、系列長 256、6 層 | 40 GFLOP | 16.7 ms / 164 mJ | 5.9 ms / 41 mJ | 0.25 倍 |

損益分岐点は、呼び出し 1 回あたり約 200 MFLOP です（パラメータ 160 万個、2 層のエンコーダーに 64 トークンを通す程度）。これより小さいと Core ML の呼び出しの固定費が勝ち、CPU が有利です。これより大きいと、Neural Engine は 2〜3 倍速く、エネルギーは 4 分の 1 で済みます。1 ジュールあたりの計算量は、Neural Engine が約 1 TFLOP、CPU が約 0.25 TFLOP でした。Neural Engine の電力そのものは小さくありません（ここでは最大 6 W）。同じ仕事を、ずっと少ないエネルギーで終えるということです。

実際の埋め込みモデルである all-MiniLM-L6-v2 も、同じ大きさの合成エンコーダーとほぼ同じ結果でした。このモデルでは、単語の埋め込みの表引き、最初の LayerNorm、最後のプーリングが CPU に残ります（170 演算のうち 148 が Neural Engine）。hwnpu の厳格モードはこのように分かれたモデルを拒否するので、`WithPlacement(PlacementPrefer)` で開いてください。

計測中のアイドル時の電力は 0.45〜1.06 W の範囲で揺れていました。損益分岐点に近い結果（d256 の行）は、誤差の範囲内です。

再現する手順:

```bash
python tools/coreml/gen_sweep.py build/sweep
python tools/coreml/convert_minilm.py build/minilm/minilm_s128_b1.mlpackage --seq 128
go run ./cmd/powerbench -model 'build/sweep/*.mlpackage' -model 'build/minilm/*.mlpackage' -json build/power-sweep.json
python tools/coreml/gen_sweep.py build/sweep --report build/power-sweep.json
```

`gen_sweep.py` は、大きさを段階的に変えた BERT 型のエンコーダーを、ランダムな重み（fp16、固定の形）で作ります。`convert_minilm.py` は sentence-transformers/all-MiniLM-L6-v2 を、系列長を固定した ML Program に変換します。入力は input_ids と attention_mask、出力は正規化した 384 次元の埋め込みです。変換後に PyTorch の出力と照合します（コサイン類似度 0.99998）。

## AMD XDNA NPU

**受け付ける形式**

- IRON / mlir-aie（`aiecc`）が出力する、AIE パーティションと DPU カーネルの情報を含む xclbin と、命令列。hwnpu が実行するのは**カーネル**です。ONNX などのモデルファイルは受け付けません。AMD のモデル用コンパイラ（ONNX Runtime が使う Vitis AI）が、公開されたライブラリではないためです。

**データ型**

- AIE-ML のコアがネイティブに計算できるのは int8、int16、bf16 です。XDNA2（Strix Point 以降）では、8 個の値で指数部を共有するブロック浮動小数点（BFP16）も加わります（[Ryzen AI の世代間での GEMM 最適化](https://arxiv.org/pdf/2512.13282)）。fp32 は模擬で処理されます。fp16 のネイティブ対応は、資料では見つかりませんでした。
- バッファは生のバイト列で、配置と型はカーネルが決めます。スカラー引数は、`Kernel.Args()` に示される大きさに合わせて、`SetUint32`、`SetUint64`、`SetBytes` で渡します。
- hwnpu が用意している変換関数は fp16 用だけです。bf16 は利用者のコードで変換してください（fp32 の上位 16 ビットを、丸めて取り出します）。

**制約**

- 形状、タイル分割、データ型は、カーネルをビルドする時点で決まります。
- 設計はデバイスに収まる必要があります。`npu1`（Phoenix、Hawk Point）は 4 列、`npu4`（Strix Point）は 8 列です。
- 命令列は `WithBufferKind(BufferInstructions)` で確保する必要があります。バッファの同期（`SyncToDevice`、`SyncFromDevice`）も明示的に行います。

**AMD でモデルを動かすには**

- ONNX モデルを AMD の NPU で動かす公式の経路は、ONNX Runtime と Vitis AI 実行プロバイダ（Ryzen AI Software）です。AMD の資料によれば、この経路では CNN を int8 に量子化し、Transformer は bf16 で扱います。fp32 のモデルは bf16 に変換されます。hwnpu はこの経路にまだ対応していません。

## モデルの用意の仕方：配布前に Go で一度だけ変換する

hwnpu は読み込み時にモデルを変換できません。変換には OpenVINO の C++ フロントエンド、Python の coremltools、AMD の非公開コンパイラが要り、機械ごとに変換すると時間と電力を使ってしまうからです。そこで、開発者が一度だけ変換し、実行先ごとに（その NPU が最も速く読める精度で）1 つずつ成果物を作って配ります（S3、NFS、Hugging Face の非公開リポジトリなど）。アプリケーションは見つけたデバイス用の成果物を選び、実行時のコンパイルだけをベンダーに任せます。`WithCacheDir` がその結果を機械ごとに保存します。

変換に Python は要りません。LLM の構造は数種類のブロック（正規化、注意機構または畳み込み、ゲート付き MLP）の繰り返しなので、`cmd/hwnpu-convert` は `config.json` からグラフを Go で組み立て、重みは safetensors か GGUF から読みます。対応するアーキテクチャはホワイトリストで管理していて（`-archs` で一覧を表示）、今のところ Liquid AI の LFM3（LFM3-VL のチェックポイントのテキスト部と、d1-2B などの判定モデルを含む）、LFM3-VL の視覚エンコーダ（`-arch lfm3-vision`）、LFM3-Audio と LFM3-Omni の音声エンコーダ（`-arch lfm3-audio-encoder`）と音声出力（depthformer の `-arch lfm3-depthformer` と、`model_type` が liquid_codec のコーデックのデコーダ）、Google の Gemma 4（gemma-4-E2B-it のテキスト部分）、BERT 系エンコーダ（all-MiniLM-L6-v2、E5、BGE、GTE など）、PLaMo の双方向エンコーダ（PLaMo-Embedding-1B）です。アーキテクチャは `config.json` から判別します。書き出すのは Core ML のパッケージです。`-backend intel` を付けると、Intel の NPU 向けに OpenVINO IR（`out.xml`、`out.bin`、`out.parts.json`）を書き出します。プログラムは Core ML と同じように組み立て、`internal/ovlower` が IR に訳します。重みは `-format int4`、`int4-g128`、`int8` のどれかで、既定は lut4 の代わりに int4、lut6 と lut8 の代わりに int8 です。4 ビットでは文埋め込みが崩れる BERT と PLaMo-Embedding は int8 になります（[docs/intel-npu-models.md](docs/intel-npu-models.md)）。今のところ IR は、BERT と PLaMo-Embedding について OpenVINO の CPU で確かめただけで、Intel の NPU 実機ではまだ動かしていません。AMD 向けの書き出しは未実装です。

```bash
go run ./cmd/hwnpu-convert -config build/lfm3/config.json -weights build/lfm3/LFM3-1.7B.safetensors -o lfm3.mlpackage -seq 32
```

できたパッケージの入力は `x`（fp16 の埋め込み `(1, seq, hidden)`）、出力は `y`（ロジット `(1, seq, vocab)`）です。Core ML は `gather` を必ず CPU に置くので、埋め込みの表引きはホスト側で行います。BERT 系と PLaMo-Embedding のパッケージは、2 つ目の入力 `mask`（fp16 の `(1, seq)`。トークンは 1、パディングは 0）を取り、最後の隠れ状態 `(1, seq, hidden)` を返します。文埋め込みにするためのプーリング（平均か CLS）と正規化は、ホスト側で行います。Gemma 4 にはもう 1 つの埋め込み表、層ごとの埋め込み（Per-Layer Embeddings。E2B では 262144 x 8960、bf16 で 4.7 GB）があり、これもホストに置きます。パッケージはトークンの行を `ple`（fp16 の `(1, seq, 8960)`）で受け取ります。d1 の判定モデル（d1-2B）も `mask` を取り、最終の隠れ状態を返します。ホストはそこから、プロンプトの最後のトークンでの答えのラベルのロジットを読みます（`cmd/hwnpu-systemone`。llama.cpp のサーバーの System One API）。LFM3-VL の視覚エンコーダは画像の大きさ 1 つごとに変換します（`-image 幅x高さ`）。入力は画像の正規化したパッチ `(1, パッチ数, 3*パッチ*パッチ)`、出力はテキスト側の入力になる埋め込み `(1, トークン数, hidden)` で、ホストが `<|vision_start|>` と `<|vision_end|>` の間に入れます（`internal/vision`）。音声エンコーダは、受け付ける最長のメルフレーム数ごとに変換します（`-frames`）。入力はクリップの対数メルフレーム `(1, フレーム数, 128)`（後ろはゼロ）とその `mask`、出力は `(1, フレーム数/8, hidden)` で、先頭がクリップの分です（`internal/audio`）。重みは 4 ビットの表引き方式（`-format lut4`）で、出力 16 行ごとに fp16 16 個の表を持ちます。これが Neural Engine が全速で読める唯一の圧縮形式です（後述）。`-format fp16` は重みをそのまま書き出し、グラフの確認に使います。

**大きなモデルの分割**：Neural Engine が受け付けるのは、重みのファイルがチップごとの上限より小さいモデルだけです。それより大きいモデルは丸ごと CPU に置かれます。上限はモデル 1 つごとなので、`hwnpu-convert` は上限を超えるモデルを、上限に収まる最少の数の部分に、大きさがそろうように分けます。`-target` でチップを選びます（既定の `any` は、知っているチップの上限のうち最小のもの）。

| `-target` | 1 モデルの上限 | 出典 |
|---|---|---|
| `m1` | 1 GiB | 仮。M4 の半分という報告による。`npuprobe limit` で要計測 |
| `m2` | 1 GiB | 仮。M3 と同じと仮定。`npuprobe limit` で要計測 |
| `m3` | 1 GiB | `npuprobe limit`：0.99 GiB は Neural Engine、1.02 GiB は CPU（どの形式でも）。[PROBE.ja.md](PROBE.ja.md) |
| `m4` | 2 GB | 4 B のモデルを lut4 で（2 GB）Neural Engine で動かせたという計測 |

部分ごとの重みの見積もりは上限の 97% までに収め、書き出した後に実際の重みのファイルが上限より小さいことを確かめます。分けたときは `out.part1.mlpackage`、`out.part2.mlpackage` … と、順番と各部分の入出力を書いた `out.parts.json` ができます。各部分の入力 `x` は、その部分の最初の層に入る隠れ状態です（最初の部分ではトークンの埋め込み）。`mask` などほかの入力は全部の部分が取ります。前の部分の出力 `y` のバッファを次の部分の `x` にそのまま渡して順に動かし、最後の部分の `y` がモデルの出力です。M3 では、分けて動かしても 1 つのモデルと同じ速さでした（PROBE.ja.md の `npuprobe split`）。M4 でも、0.91 GiB の 4 つと 1.81 GiB の 2 つで速さも誤差も変わりませんでした（58.8 ms と 58.3 ms）。全部の部分を同時に読み込みますが、読み込むときのプロセスの最大メモリはおよそ 1 部分の重みの大きさになるので、メモリの少ない機械では細かく分ける意味があります（M3 で 1.81 GiB を 2 つにして 1.03 GiB、4 つにして 0.55 GiB）。1 つのパッケージに部分を関数として入れる方法は使えません。上限はパッケージ全体で数えられ、全部 CPU に置かれます。そのため、全部のチップに 1 組の成果物を配れる既定の `any` を勧めます。`-parts N` で部分の数を指定でき、`-target none` で分割をやめます。`-archs` は対応するアーキテクチャに続けて、チップごとの上限を表示します。

```bash
go run ./cmd/hwnpu-convert -config config.json -weights model.safetensors -o model.mlpackage -seq 32 -target m3
```

変換が正しいかは、Mac がなくても確かめられます。`tools/ref/ref_logits.py` で Hugging Face の実装から参照の活性を保存し、`cmd/hwnpu-check` がグラフを CPU で fp32 のまま実行して層ごとに照合します。各層には参照側の入力を与えるので、誤差が後ろの層に積み重ならず、最初に間違っている層がそのまま分かります。チェックポイントにあって変換が読まないテンソルも報告します。`-package` を付けると、書き出したパッケージ（lut4 など）のロジットを参照と比べます。

```bash
python tools/ref/ref_logits.py build/lfm3 build/lfm3/ref --seq 32
go run ./cmd/hwnpu-check -config build/lfm3/config.json -weights build/lfm3/LFM3-1.7B.safetensors -ref build/lfm3/ref
go run ./cmd/hwnpu-check -package lfm3.mlpackage -ref build/lfm3/ref
```

分けたモデルも 1 つのモデルも、`cmd/hwnpu-generate` で文章を生成できます。`hwnpu-generate` は、Go で書いたトークナイザー（`tokenizer.json` の byte-level BPE）でチャットの書式のプロンプトを作り、埋め込みをチェックポイントから引いて、Neural Engine で 1 トークンずつ生成します。

長い文章には KV キャッシュを使います。`-cache N` を付けて変換すると、プログラムは 1 回に `-seq` トークン（生成には 1）を受け取り、それまでのトークンのキーと値（と LFM3 の畳み込みの直前の値）を入力として読み、新しい分を出力します。キャッシュはホストのバッファに置かれ、`hwnpu-generate` が毎回書き足します。各部分は自分の層の分だけを持つので、分割と組み合わせられます。1 つのモデルなら `-package` に `.mlpackage` を、分けたモデルなら `.parts.json` を渡します。

```bash
go run ./cmd/hwnpu-convert -config build/lfm3/config.json -weights build/lfm3/LFM3-1.7B.safetensors -o build/lfm3/lfm3-kv.mlpackage -seq 1 -cache 4096 -quant kmeans-minmax -parts 2
go run ./cmd/hwnpu-generate -package build/lfm3/lfm3-kv.parts.json -config build/lfm3/config.json -weights build/lfm3/LFM3-1.7B.safetensors -n 1500 -prompt "社内向けに、Go言語を採用する理由をまとめた文書を書いてください。"
```

スライディング注意の層（LFM3 では 24 層のうち 5 層、窓 256）のキャッシュは、窓の長さのリングバッファになります。長い容量が要るのは全注意の 6 層だけです。M3 の LFM3-1.7B（lut4、2 分割）では、キャッシュが 1024 トークンなら 1 トークン 19.1 ms、4096 なら 27.7 ms、8192 なら 38.5 ms でした。遅さは使った量ではなくキャッシュの容量に比例するので、必要な長さに合わせて選んでください。Neural Engine はテンソルの 1 次元の長さが 16384 までなので、`-cache` と `-seq` の合計が 16384 を超えると注意の演算が CPU に置かれ、極端に遅くなります（変換時に警告します）。

費用は容量に比例するので、短い文脈のうちは短いキャッシュで動かすほうが速くなります。`-cache 1024,4096,8192` のように複数の長さを増える順に書くと、長さごとの部分（`out.c1024.part1.mlpackage` …）を 1 つの `out.parts.json` に段階として並べます。`hwnpu-generate` は既定で全部の段階を先に読み込み（`-preload all`）、最も短い段階から始めて、満杯になったらキャッシュを写して次の段階に移ります。出力は最初から長い段階で動かした場合と同じです。段階の読み込みには 1 つ 6〜30 秒かかり、裏で読み込みながら生成すると生成が半分以下の速さに落ちるので、`-preload 0.5` のような裏での先読みは、読み込みの遅さを受け入れられるときだけ使ってください。重みは段階ごとに重複して書かれ（LFM3-1.7B の lut4 で 1 段階 0.8 GiB）、全部の段階がメモリに載ります。キャッシュ付きのモデルは、コンパイル済みのキャッシュがあっても読み込みに 15〜20 秒ほどかかります（キャッシュなしのモデルは 1 秒ほど。原因は未調査）。キャッシュなしで変換したモデルは、毎トークン `-seq` の長さの窓全体を計算し、窓が満ちると古いトークンを捨てます（seq 128 で 34 ms 前後）。

### 元のチェックポイントの選び方

公式の bf16 から変換します。公開されている量子化済みのものは、対象の形式へ劣化なしに並べ替えられるときだけ使います。劣化なしの変換は「より細かい単位か、より広い範囲の方式」への片方向だけです。対称 int4（並べ替えなしの GPTQ、compressed-tensors の W4A16）は Intel の対称 int4 に、AWQ は AMD の非対称 uint4（128 個ごと）にそのまま入ります。Neural Engine の表引き方式に劣化なしで入る公開形式はなく、4 ビットを別の 4 ビットに量子化し直すと誤差が二重に乗ります。LFM3-1.7B では、bf16 から lut4 にすると bf16 モデルに対する KL が 0.95 nat、公式の QAD Q4_0 GGUF から作り直すと 0.94 でした（QAD 自体の差は 0.35 なので、出発点は近いのに結果は良くなりません）。結果は必ず測ります。`tools/ref/ref_logits.py` が PyTorch の隠れ状態とロジットを保存し、`cmd/hwnpu-check`、`tools/ref/eval_logits.py`、Go のテストがパッケージを層ごとに照合します。

### Neural Engine が受け付ける形式（M3、macOS 27.0.1）

- 表引きの表は行全体を覆う必要があり、出力 16 チャンネルごとに 1 表です（coremltools の `per_grouped_channel`）。列方向にもブロック化した表（16 × 64）はモデル全体が CPU に落ちます。行ごとに 1 表にするとコンパイルに 7 分かかり、出力は壊れます。64 値と 256 値の表（6 ビットと 8 ビットの添字、`-format lut6` と `lut8`）も動き、速さはバイト数に従います。Gemma 4 E2B の大きさの 6 層で、lut4 が 1.81 ms、lut6 が 2.65 ms、lut8 が 3.49 ms でした（fp16 は 6.52 ms）。
- 線形の int4（`constexpr_blockwise_shift_scale`）は行ごとに 1 つのスケールのときだけ Neural Engine で動きます。入力 32 個や 128 個ごとのスケール（llama.cpp の Q4_0 の並び）は CPU に落ちるので、Q4_0 をそのまま並べ替えることはできません。
- 行列積の出力には約 2.5e-4 の絶対誤差の床があります（長さ 2048 の内積、重みの rms 0.02）。rms 1 の入力では 0.03% の誤差、rms 0.03 の入力では 0.95% で、絶対誤差は同じです。LFM3 のゲート付き射影（`out_proj`、`o_proj`、`w2`）の出力は rms 0.008 しかなく、層ごとに 4〜12% を失っていました。入力を大きくして重みを小さくしても積が変わらないので効きません。そこで変換器はこれらの重みを 64 倍し（`-post-scale`）、直後のサンドイッチ正規化の中で正確な要素積で元に戻します。この場所ならコンパイラが折りたたんで打ち消すことができず、層ごとの誤差は 1.4〜2.7% になりました。
- プログラム中の値の名前にドットを含めてはいけません。ローダが落ちます。
- 要素ごとの演算、深さ方向の畳み込み、正規化は正確です。損失は行列積にあります。

### 表の値の決め方

4 ビットの表の決め方は何通りかあり、重みの誤差が最も小さい方法がモデルにとって最良とは限りません。LFM3-1.7B（24 トークンのプロンプト、CPU で bf16 モデルに対する KL、16 × 64 ブロック）では次の通りです。

| 決め方 | 重みの相対 RMS 誤差 | KL (nat) |
|---|---|---|
| 分位点から始める k-means（二乗誤差が最小） | 0.113 | 3.68 |
| 最小値と最大値の間を等間隔 | 0.138 | 1.07 |
| 等間隔の値から始める k-means（`-quant kmeans-minmax`、既定） | 0.117 | 0.54 |

二乗誤差は多数の小さい重みと少数の大きい重みを同じに扱いますが、ネットワークは大きい重みに依存しています。等間隔の初期値はそれを残し、k-means の反復が本体を細かくします。Neural Engine が必要とする行全体の表では、既定の決め方で CPU の KL が 0.95、`-post-scale 64` を付けた Neural Engine で 0.89 でした。まだ粗く（argmax の一致は 24 中 9）、MLP か注意機構・畳み込みの射影のどちらかを fp16 に残すと半分になるので、損失は特定のテンソルではなくモデル全体に広がっています。活性を見た量子化や 6 ビットの表が次の候補です。Gemma 4 E2B は lut4 の損失がずっと大きく（KL 2.54。層が残差の大半を作り直すので、重みの誤差がそのまま出ます）、既定で lut6 に変換します。バイト数 1.5 倍で KL 0.25、M3 で 1 トークン 31 ms（lut4 は 24 ms。lut8 は KL 0.05、41 ms）です。Liquid AI のマルチモーダルと d1 のチェックポイント（LFM3-VL、LFM3-Audio、d1-2B、d1-Omni-2B）のテキスト部は、既定で lut8 に変換します。lut4 では d1-2B の答えが fp16 からずれ、LFM3-VL は画像を取り違え、LFM3-Audio の音声は聞き取れなくなりますが、lut8 は fp16 と同じに答えます（2 部分、M3 で 1 トークン 41 ms）。LFM3-1.7B は lut4 のままです。いちばん崩れるのは BERT 系と PLaMo-Embedding-1B の文埋め込みで、fp16 のプログラムとのコサイン類似度は lut4 で 0.23（all-MiniLM-L6-v2）と 0.76（PLaMo、`kmeans-minmax`）、lut8 で 0.999 と 0.998 でした。そのため、これらも既定で lut8 に変換します（PLaMo-Embedding-1B も 1 部分に収まります）。PLaMo-Embedding-1B の最終層の MLP は最大 5e4 の値を出し、Neural Engine では無限大になります。この層の出力は最終の正規化しか読まないので、変換では最終層を 1/64 倍で計算します（`-head-scale`）。これで M3 での埋め込みは、Hugging Face の実装とのコサイン類似度が 0.997〜0.999 になりました。
