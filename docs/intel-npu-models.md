# 変換したモデルを Intel NPU で動かす: 設計

作成: 2026-10-30。対象は `cmd/hwnpu-convert` が扱う全アーキテクチャ (LFM3 とその VL・Audio・d1、Gemma 4、BERT、PLaMo-Embedding) と、それを動かす `internal/llm/{chain,decode,runner}`、`internal/chat`、`internal/systemone`、`internal/speech`。

## 1. 現状

Mac で確かめてきたモデルは、どれも Intel NPU では動かない。足りないのは変換と実行の両方。

| 層 | 今の状態 | Intel で足りないもの |
|---|---|---|
| 構造の記述 (`internal/llm/<arch>`、`blocks`) | `mlpackage.Builder` に MIL の演算を直接積む | IR を書く経路がない |
| 重み形式 (`blocks.Graph.matrix`) | `fp16`、`lut4/6/8` (`constexpr_lut_to_dense`) | Intel NPU が読む対称 int4/int8 (Convert → Multiply の形) |
| 出力 (`llm.ConvertTiers`、`WritePart`) | `.mlpackage` と `out.parts.json` | `.xml` + `.bin` |
| メタデータ (arch、params、format) | パッケージの `userDefined` に書き、`runner` が `mlpackage.ReadModel` で読む | IR から読む手段 |
| 分割 (`llm.Targets`) | M1〜M4 の重みファイルの上限 | Intel の上限は未測定 |
| 実行 (`chain`、`decode`) | `hwnpu.Graph` と名前付きの fp16 バッファだけを使う | **ほぼ不要**。バックエンドに依存していない |
| 実行の入口 | `WithBackend("coreml")` が 6 か所に固定 (`chat/engine.go`、`chat/embed.go`、`systemone/model.go`、`speech/speech.go`、`hwnpu-generate`、`powerbench`) | 端末によるバックエンドの選択 |
| Level Zero バックエンド | IR のコンパイル、実行、ブロブのキャッシュがある。ただし動作確認は模擬ドライバだけ | 実機での確認 |

実行側がバックエンドに依存していないのは大きい。KV キャッシュをホストのバッファに置く方式 (design.md 16.14) は、Level Zero のホストメモリ (`zeMemAllocHost`、`CapHostCoherent`) でそのまま成り立つ。重い作業は変換側に集まる。

## 2. 方針: MIL を中間表現にして IR へ下ろす

### 2.1 候補

**(A) MIL → OpenVINO IR の変換パスを足す。** アーキテクチャのコードは今のまま `mlpackage.Builder` に積み、できた MIL の関数を 1 演算ずつ IR に訳す。

**(B) バックエンドに依存しないグラフ API を作り、Core ML と IR の両方へ書く。** `blocks.Graph` の `*mlpackage.Var` を抽象型に置き換え、7 つのアーキテクチャと blocks (テストを除いて約 4500 行) を書き直す。

### 2.2 選択: (A)

- 使っている MIL の演算は **26 種類しかない** (3 節)。どれも OpenVINO の opset に素直な対応がある。
- アーキテクチャのコードに手を入れずに、全モデルが一度に対象になる。移植 (`port-architecture` スキル) の手順も、合格条件も変わらない。
- MIL 側には CPU インタープリタ `internal/milcpu` がある。同じプログラムを milcpu と OpenVINO の CPU で動かして比べれば、変換パスの誤りだけを切り分けられる。
- (B) の抽象化は、(A) で Intel 向けに形を変える必要が実際に出てから、必要な箇所だけに入れればよい (6 節)。

弱点: MIL の演算名と入力の辞書 (`map[string]any`) を介すので、型の検査は実行時になる。変換パスは、知らない演算や入力に出会ったら必ずエラーで止め、黙って落とさない。

### 2.3 流れ

```
config.json + safetensors/GGUF
        │  internal/llm/<arch>.Build / BuildCached (変更なし)
        ▼
mlpackage.Builder (MIL の演算列 + 重みの blob)
        ├── -backend coreml: そのまま .mlpackage (今まで通り)
        └── -backend intel : internal/ovlower で IR へ
                              → out.xml + out.bin (+ out.parts.json)
```

`WritePart` は今、`mlpackage.Create` でパッケージを作り、Builder の定数をその `weight.bin` に書きながら構築する。Intel の場合は次のようにする。

1. 定数の書き先は、出力ディレクトリの一時ファイル (`out.milblob.tmp`) にする。`BlobWriter` はファイルに順に書くので、数 GB のモデルでもメモリを食わない。
2. `b.Function(outputs...)` で `milspec.Function` を得て、`ovlower.Lower(fn, blobReader, ins, outs)` に渡す。
3. `ovlower` が `out.xml` を書き、定数を `out.bin` に順に書き出す。一時ファイルは最後に消す。

`ovir.Builder` は今、`.bin` を全部メモリに持つ (`bin []byte`)。プローブ用の小さなモデルなら問題ないが、LFM3-1.7B の int8 なら 1.7 GB になる。`io.Writer` に書き出す形に直す。

## 3. 演算の対応

`internal/llm` 以下が使う演算は、全アーキテクチャを合わせて次のとおり (2026-10-30 時点、テストを除く)。

| MIL | 使う所 | OpenVINO IR (opset) | 注意 |
|---|---|---|---|
| `mul` `add` `sub` `real_div` `maximum` | 全部 | `Multiply` `Add` `Subtract` `Divide` `Maximum` (opset1、`auto_broadcast=numpy`) | MIL のスカラー (形 nil) は形 `[]` の `Const` |
| `reshape` | 全部 | `Reshape` (opset1、`special_zero=false`) | 形の定数は int32 から i64 に直す |
| `transpose` | 全部 | `Transpose` (opset1) | `perm` も i64 |
| `slice_by_index` | LFM3、Gemma 4、VL、Audio、gen、RoPE | `StridedSlice` (opset1) | `begin_mask`/`end_mask` の意味 (true で無視) は同じ。`stride`、`squeeze_mask` は使っていれば対応、なければエラー |
| `identity` | 全部 (出力に名前を付けるため) | 演算は作らない。入力を作った層の名前を出力名にする | 4.2 を参照 |
| `matmul` | SDPA、gen | `MatMul` (opset1、`transpose_a/b`) | |
| `linear` | 全部 (`blocks.Linear`) | `MatMul(transpose_b=true)` (+ `Add` で bias) | 重みは 5 節の形 |
| `pad` | LFM3 の畳み込み、VL、Audio、gen | `Pad` (opset12) | MIL の `pad` は軸ごとに (前, 後) が並ぶ。`pads_begin` と `pads_end` に分ける。`mode` は `constant` だけ使っている |
| `rsqrt` | RMSNorm | `Add`(epsilon) → `Sqrt` → `Divide(1, ·)` | MIL の rsqrt は `1/sqrt(x+eps)`。`Power(x, -0.5)` より、Intel のコンパイラが RMSNorm の形を認識しやすい書き方を実機で選ぶ |
| `reduce_mean` `reduce_max` | RMSNorm、ChunkedSDPA | `ReduceMean` `ReduceMax` (opset1、`keep_dims`) | |
| `concat` | RoPE、LFM3、Gemma 4 | `Concat` (opset1) | `interleave=true` は使っていない。来たらエラー |
| `tile` | GQA の K/V の複製 | `Tile` (opset1) | |
| `silu` `sigmoid` `tanh` `exp` | 活性化 | `Swish` (opset4)、`Sigmoid`、`Tanh`、`Exp` | |
| `softmax` | SDPA | `SoftMax` (opset8) | |
| `gelu` | BERT、VL | `Gelu` (opset7)、`EXACT` → `erf`、`TANH_APPROXIMATION` → `tanh` | |
| `layer_norm` | BERT、VL | `MVN` (opset6、`eps_mode=inside_sqrt`) → `Multiply`(gamma) → `Add`(beta) | |
| `conv` | LFM3 (`ConvMode=conv`)、Audio | `GroupConvolution` (opset1。`groups=1` なら `Convolution`) | MIL の重み (out, in/g, k) を (g, out/g, in/g, k) に組み直す |
| `const` | 全部 | `Const` | |
| `constexpr_blockwise_shift_scale` | 新しい重み形式 (5 節) | `Const(i4/i8)` → `Convert(f16)` → `Multiply(scale)` | |
| `constexpr_lut_to_dense` | `lut4/6/8` | **対応しない** | 「Intel では `-format int4` か `int8` を使う」というエラーにする |
| `read_state` `write_state` | LLM では未使用 | 対応しない | |

入力 (`Parameter`) と出力の型は fp16 のまま。KV キャッシュの入出力 (`kv_NAME`、`kv_NAME_new`、`win_NAME`) も同じ規則で訳す。

## 4. 出力物

### 4.1 ファイル

| | Core ML | Intel |
|---|---|---|
| 1 つのプログラム | `out.mlpackage` | `out.xml` + `out.bin` |
| 分割 | `out.partN.mlpackage` | `out.partN.xml` + `.bin` |
| 段階 | `out.cN[.partK].mlpackage` | `out.cN[.partK].xml` |
| 一覧 | `out.parts.json` | `out.parts.json` (同じ形式。`backend: "intel"` を足す) |

`chain` は `GraphSource{Path}` で読み込み、Level Zero バックエンドは拡張子 `.xml` で IR と判断する (`device.go` の `Compile`)。したがって `chain` は変更なしで両方を読める。

### 4.2 入出力の名前

`chain` と `decode` は、引数を名前で結ぶ (`x`、`y`、`mask`、`kv_*`、`win_*`、`ple`、`h`)。Level Zero が返す引数の名前は、IR の `Parameter` の名前と、`Result` に入る層の名前 (`ovir.Parse` の `producer`) になる。そこで変換パスは次のようにする。

- `Parameter` の層名と tensor 名は、MIL の入力名をそのまま使う。
- MIL で `identity` に名前を付けて出力にしている所 (`NamedOp("identity", "y", …)`) は、identity を消し、入力を作った層の名前を `y` にする。
- 1 つの層が 2 つの出力を兼ねる場合 (同じ値を `y` と `h` に出すなど) に備え、名前が衝突したら `Convert(f16→f16)` などの無害な層を 1 つ挟む。実機で名前が期待どおりに返るかは、模擬ドライバ (`testdata/zemock`) と実機の両方で確かめる。

### 4.3 メタデータ

`runner.Open` などは、アーキテクチャ名・パラメータ・形式を `mlpackage.ReadModel(...).UserDefined` から読んでいる。これを IR でも読めるように、次の 2 か所に同じものを書く。

- `out.parts.json` の `meta` (新設)。Intel では 1 つのプログラムでも `out.parts.json` を必ず書く。
- IR の `<rt_info>` (`hwnpu_arch` など)。`.xml` だけを渡された場合のため。

`runner` は manifest の `meta` を先に見て、なければ従来どおり mlpackage から読む。既存の Core ML 変換結果はそのまま使える。

## 5. 重みの形式

### 5.1 Intel NPU が速く読む形

OpenVINO の NPU プラグインと NNCF は、重みの圧縮を `Const(i4|i8|u4|u8)` → `Convert(f16)` → [`Subtract`(零点)] → `Multiply`(スケール) → `MatMul` という形で表す。NPU のコンパイラはこの形を認識し、重みを圧縮したまま読む。`npuprobe` の `BuildIR` も、対称・行ごとのスケールでこの形を書いている。

OpenVINO GenAI の NPU 向けの推奨は「int4、対称、チャネルごと (行ごと)」。グループ量子化 (128 列ごと) は NPU4 (Lunar Lake) 以降で動くが遅い、というのが公開情報。**どれも要実測** (8 節)。

### 5.2 新しい形式名

| `-format` | 中身 | MIL での表現 | IR での表現 |
|---|---|---|---|
| `int4` | 対称 int4、行ごとに fp16 のスケール 1 つ | `Builder.Int4Scaled`、block `[1, cols]` | `Const(i4)` → `Convert` → `Multiply(scale (rows,1))` |
| `int4-g128` | 対称 int4、128 列ごとにスケール | block `[1, 128]` | `Const(i4 (rows, cols/128, 128))` → `Convert` → `Multiply(scale (rows, cols/128, 1))` → `Reshape (rows, cols)` |
| `int8` | 対称 int8、行ごと | `Builder.Int8Scaled`、block `[1, cols]` | `Const(i8)` → `Convert` → `Multiply` |
| `fp16` | そのまま | | `Const(f16)` |

`mlpackage.Builder` には `Int4Scaled`/`Int8Scaled` (MIL の `constexpr_blockwise_shift_scale`) が既にある。`blocks.Graph.matrix` に上の 3 形式を足すだけで、全アーキテクチャで使える。

この形式は Core ML でも動く (Neural Engine での速さは lut より劣る。MODEL.md)。そのため、**Intel 向けの量子化の精度を、Mac の上で今までと同じ手順** (`tools/ref/ref_logits.py` → `hwnpu-check`、`eval_logits.py`) で測れる。

スケールは、ブロックの最大値の 1〜0.5 倍の 11 通りを切り捨ての位置として試し、二乗誤差が最小のものを選ぶ (`-quant mse`、既定)。最大値をそのまま 7 (int4) か 127 (int8) に当てる min-max は `-quant uniform`。all-MiniLM-L6-v2 の int4 (行ごと) で、文埋め込みのコサイン類似度が min-max の 0.909 から 0.934 に上がったので、mse を既定にした (11 節)。

### 5.3 既定の形式

アーキテクチャの `Format` (と `Formatter`) は Core ML の形式名で書かれている。Intel では次の表で読み替える。

| Core ML の既定 | Intel の既定 | 該当 |
|---|---|---|
| `lut4` | `int4` | LFM3-1.7B |
| `lut4` | `int8` (`Arch.IntelFormat`) | BERT、PLaMo-Embedding (int4 では文埋め込みが崩れる。11 節) |
| `lut6`、`lut8` | `int8` | Gemma 4 E2B (lut6)、d1、LFM3-VL、LFM3-Audio/TTS (M3 で lut4 が劣化した系統。精度を優先) |
| `fp16` | `fp16` | |

`KeepFP16`、`FormatFor` はそのまま効く。`FormatFor` に `lut*` が書かれていたら、同じ表で読み替える。

## 6. Neural Engine 向けの工夫の扱い

アーキテクチャのコードには、Neural Engine のために選んだ書き方が入っている。どれも数学的には同じ計算なので、**第 1 段階ではそのまま IR に訳す**。Intel で遅い、または精度が悪いと実測で分かったものだけ、`llm.Options` に足す `Backend` を見て形を変える。

| 工夫 | 場所 | Intel での扱い |
|---|---|---|
| 1 次元 16384 の上限のため、KV キャッシュを塊に分ける (`CacheChunkOver`、`ChunkedSDPA`) | `llm`、`blocks` | 上限は Intel では未知。最初は同じ値のまま。`npuprobe kv` を Intel で回して決める |
| 塊の合流で `reduce_sum` を避け、1 の列との `matmul` で和を取る | `ChunkedSDPA` | 同じ計算なので残す |
| `LinearS` と RMSNorm の `inScale` (fp16 の桁あふれ対策) | `blocks` | Intel も fp16 で計算するので残す |
| LFM3 の短い畳み込みを `taps` (pad と積和) で書く。`conv` は ANE で誤差 5〜12% | `lfm3.ConvMode` | Intel では `conv` (GroupConvolution) の精度を測り、良ければ既定にする |
| GQA の K/V を `tile` で複製 (`SDPA`) と、複製しない `SDPAGrouped` | `blocks` | Intel では `SDPAGrouped` の形が NPU に載るかを測る |
| スライディング注意のリングキャッシュ | `lfm3`、`gemma4` | ホスト側の処理なのでそのまま使える |
| 重みファイルの上限による分割 (`llm.Targets`) | `split.go` | 6.1 |

### 6.1 分割と段階

Intel NPU の重みの上限は分かっていない。Lunar Lake 以降はシステムメモリを共有するので、ANE の 1 GiB のような固い上限はない見込みだが、2 つの別の理由で分割は要る。

- **コンパイル時間。** ドライバ内のコンパイラは、大きなグラフに数分かかる (OpenVINO の NPUW が層を「同じ関数の繰り返し」として 1 回だけコンパイルするのはこのため)。分割すると、各部分のコンパイル結果を `WithCacheDir` で別々に再利用できる。
- **静的な形。** プロンプトを読む `-seq 64` と生成の `-seq 1` は別のグラフになる。Core ML では別パッケージの重みは別に読まれ、Intel でもグラフごとに重みを持つ。**メモリが倍になる**。第 1 段階では受け入れ、design.md 5.3 の「コンパイラの重み分離 (6.3 以降、要検証)」を後で調べる。

`llm.Targets` に `intel-mtl`、`intel-lnl`、`intel-arl`、`intel-ptl` を足す。`MaxBytes` は最初は仮の値 (たとえば 2 GiB) にして `Measured: false` とし、`-target intel` は「既知の Intel の最小値」とする。実機で `npuprobe limit` と `npuprobe split` を回して決める。

## 7. 実行側

### 7.1 バックエンドの選択

固定の `WithBackend("coreml")` を、次の関数 1 つに置き換える。

```go
// llm.OpenDevice は、この端末の NPU を開き、それが読む変換結果の種類を返す。
func OpenDevice(ctx context.Context, opts ...hwnpu.Option) (*hwnpu.Device, Backend, error)
```

macOS では `coreml`、それ以外では `levelzero` を選ぶ。`HWNPU_BACKEND` 環境変数と `-backend` フラグで上書きできるようにする。`PlacementPrefer` は Core ML だけの話なので、Intel では付けない (Intel のコンパイラは他の装置に逃がさないので、コンパイルが通れば全演算が NPU に載る)。

### 7.2 hwnpu-chat の初回変換

`hwnpu-chat` は、初回にチェックポイントを取ってきて、その場で変換する (`chat/embed.go` の `embed-s%d.mlpackage` など)。この変換を `Backend` に合わせて `.xml` で書くようにする。変換器は Go だけで書かれているので、Intel 機 (Linux、Windows) でもそのまま動く。

変換結果の置き場所はバックエンドごとに分ける (`<model>/coreml/`、`<model>/intel/`)。同じディレクトリを Mac と Intel 機で共有しても衝突しない。

### 7.3 変更の要らない所

- `chain`: 部分の出力バッファを次の部分の入力に結ぶ処理は、`hwnpu.Graph` の引数の名前だけを使う。
- `decode`: fp16 のバイト列を直接読み書きする。Level Zero のホストメモリは CPU から直接読めて、Sync は何もしない。
- `tokenizer`、`vision`、`audio`、`speech` の前処理と後処理。

## 8. 確かめ方

Intel の実機がなくても、グラフの正しさと量子化の精度は Mac の上で確かめられる。OpenVINO は macOS (arm64) でも pip で入り、CPU で IR を実行できる。

| 段階 | 何を確かめるか | どこで | 道具 |
|---|---|---|---|
| L0 演算 | 3 節の 1 演算ずつの訳が正しいか | Mac / Linux | 小さな MIL を作り、`milcpu` の結果と、IR を OpenVINO CPU で動かした結果を比べる。Go のテストから `tools/ov/run_ir.py` を呼ぶ (OpenVINO がなければ skip) |
| L1 モデル | モデル全体の IR が PyTorch と合うか | Mac / Linux | `tools/ref/ref_logits.py` の結果と、IR を OpenVINO CPU で動かした結果を `eval_logits.py` で比べる。fp16 と int4/int8 の両方 |
| L1' 量子化 | int4/int8 の精度 | Mac | `-backend coreml -format int4` の結果を `hwnpu-check` で。L1 と同じ数字になるはず |
| L2 コンパイル | NPU のコンパイラが受け付けるか、全演算が載るか | Intel 実機 | `hwnpu-check`、Level Zero の `SupportedLayers` (載らない層の名前が出る) |
| L3 精度と速さ | fp16 の NPU での誤差、1 トークンの時間、電力 | Intel 実機 | `hwnpu-check`、`hwnpu-generate`、`npuprobe kv/limit/split`、`powerbench` |

L0 と L1 は CI に載せられる (Linux の runner に `pip install openvino`)。L2 以降は実機が要る。リポジトリに `npuprobe.exe` があるので、Windows の Intel 機で回す想定。INSTALL_INTEL.md の `cmd/report` の結果もここで集める。

## 9. 進め方

| 段階 | 内容 | 終わりの条件 |
|---|---|---|
| **P1 変換パスとキャッシュなしのモデル** | `ovir.Builder` の `.bin` の逐次書き出し。`internal/ovlower` (3 節の全演算、4.2 の名前の規則)。`-backend intel`。`int4`/`int4-g128`/`int8` 形式。manifest の `meta` | BERT (all-MiniLM-L6-v2) と PLaMo-Embedding-1B が L0・L1 に合格 |
| **P2 KV キャッシュのある LLM** | LFM3-1.7B (畳み込み、リング、段階、塊)。`llm.OpenDevice` と、6 か所の `WithBackend` の置き換え。`runner` の meta の読み方 | LFM3-1.7B が L1 に合格。`hwnpu-generate -backend intel` が OpenVINO CPU の結果と同じトークンを出す (模擬ドライバでは配線だけ確かめる) |
| **P3 残りのモデル** | Gemma 4 (`ple` 入力、共有 KV)、LFM3-VL の画像エンコーダ、Audio エンコーダ、depthformer と codec、d1 | 全アーキテクチャが L1 に合格 |
| **P4 実機** | L2・L3。Intel の上限の実測と `llm.Targets`。6 節の形の切り替え (`Options.Backend`)。重み分離の調査 | 各モデルが Intel NPU で動き、MODEL.md に Intel の欄が増える |

P1〜P3 は Mac だけで進められる。P4 には Intel 機が要る。

## 10. 未確定のこと

- Level Zero が返す出力の引数名が、4.2 の規則で期待どおりになるか (模擬ドライバと実機)。
- int4 の行ごととグループごとの、NPU 世代 (Meteor Lake / Lunar Lake / Arrow Lake / Panther Lake) ごとの速さと精度。Meteor Lake (NPU 3720) で int4 が速いか。
- Intel NPU の 1 次元の上限、グラフの大きさの上限、コンパイル時間。KV キャッシュの塊の長さ (`CacheChunkLen`) をそれに合わせて変えるか。
- RMSNorm を分解した形 (`mul`、`reduce_mean`、`rsqrt`、`mul`) を、コンパイラが 1 つの演算にまとめるか。まとめないなら、どの書き方ならまとまるか。
- `-seq 64` と `-seq 1` のグラフで重みを共有する方法 (コンパイラの重み分離、または 1 つのグラフで両方を持つ)。
- Windows で、数 GB の `.bin` をコンパイラに渡すときのメモリの使い方 (今の `compileIR` は `.xml` と `.bin` を全部メモリに読んで `ovir.Serialize` に渡す)。

## 11. 進み具合

### P1 (2026-10-30)

できたこと:

- `internal/ovlower`: 3 節の全演算の MIL → IR 変換。出力名の規則 (4.2) を含む。lut の重みは int4/int8 を指すエラーで止める。
- `internal/ovir`: `.bin` を逐次書き出す `NewStreamBuilder`、層に名前を付ける `NamedOp`、`rt_info` の書き込み。
- `-format int4`、`int4-gN`、`int8` (`palette.QuantizeLinear`、`Builder.Int4Scaled/Int8Scaled`)。Core ML にも同じ形で書ける。
- `hwnpu-convert -backend intel -o out.xml`: `out.xml`、`out.bin`、`out.parts.json` (`backend` と `meta` 付き) を書く。既定の形式は 5.3 の読み替え。`-target` の一覧に `intel` (2 GiB、仮の値) を足した。
- `internal/ovcpu` と `tools/ov/run_ir.py`: IR を OpenVINO の CPU で動かす。Go のテストは `HWNPU_OV_PYTHON` に openvino の入った Python を指定すると動き、なければ skip する。

確かめたこと (Mac、OpenVINO 2026.4 の CPU、推論の精度 f32):

| テスト | 内容 | 結果 |
|---|---|---|
| `ovlower` の 5 テスト (L0) | 全演算を milcpu と比べる | 相対誤差 3e-4 以下 (IR の出力が fp16 なので、その丸めの分)。形の操作は完全に一致 |
| `bert`、`plamo` の `TestIntelIR` (L1) | 小さな合成モデルを、Go で書いた参照実装と比べる | fp16 で一致。PLaMo は int4/int8 でも、同じ量子化をした参照と一致 |
| `bert` の `TestIntelIRCheckpoint` (L1) | all-MiniLM-L6-v2 の実物。最後の隠れ状態の平均 (文埋め込み) のコサイン類似度を、fp16 の MIL を milcpu で動かした結果と比べる | 下の表 |

| 形式 | コサイン類似度 | 隠れ状態の相対誤差 |
|---|---|---|
| fp16 | 1.00000 | 0.0002 |
| int8 | 0.99946 | 0.033 |
| int4 (行ごと、mse) | 0.934 | 0.33 |
| int4 (行ごと、uniform) | 0.909 | 0.40 |
| int4-g32 (mse) | 0.961 | 0.25 |
| 参考: lut4 (milcpu) | 0.229 | 1.20 |
| 参考: lut8 (milcpu) | 0.9988 | 0.046 |

all-MiniLM-L6-v2 (隠れ層 384、6 層) は 4 ビットに弱い。int4 でも 0.93 で、文埋め込みに使うなら int8 にしたほうがよい。**Mac の既定 (lut4) では 0.23 で、ほとんど使えない**。BERT の既定の形式は見直しが要る (Intel 向けの作業とは別の問題)。

PLaMo-Embedding-1B の実物 (`plamo` の `TestIntelIRCheckpoint`、`HWNPU_PLAMO_DIR`): 3 つの文 (日本語 2、英語 1、`-seq 48`) の文埋め込み (BOS の後の行の平均) を、fp16 の MIL を milcpu で動かした結果と比べ、3 つのうち最も低いコサイン類似度を取った。

| 形式 | 最も低いコサイン類似度 |
|---|---|
| fp16 | 1.00000 |
| int8 | 0.99959 |
| int4 (行ごと) | 0.912 |
| int4-g128 | 0.919 |
| int4-g32 | 0.936 |
| 参考: lut4 uniform (milcpu) | 0.198 |
| 参考: lut4 kmeans-minmax (milcpu、hwnpu-chat の設定) | 0.756 |
| 参考: lut8 kmeans-minmax (milcpu) | 0.998 |

埋め込みモデルは、MiniLM も PLaMo も 4 ビットでは崩れ、8 ビットなら保たれる。そこで Intel の既定を BERT と PLaMo-Embedding では int8 にした (`Arch.IntelFormat`)。

**Mac 側の問題**: hwnpu-chat は PLaMo-Embedding を lut4 (kmeans-minmax) で変換しており (`chat/model.go`)、fp16 とのコサイン類似度は 0.76 しかない。`TestPlamoEmbeddingGolden` の合格条件 (HF の結果と 0.98 以上) には届かないはず。lut8 なら 0.998。

P1 は以上で終わり。

