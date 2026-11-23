# 画像を読むモデル

[English](vision-models.md)

画像を読むモデル (画像生成ではない。[image-models.md](image-models.md)): 物の検出、姿勢、OCR、切り抜き、深度、顔、ラベルと埋め込み。どれも `hwnpu-image` で動かす。多くは ONNX でしか配られず、共通の道 (`internal/onnx`) を通る。

## モデルとコマンド

| コマンド | 用途 (models.yaml) | モデル (機械のメモリで選ぶ) | 取得元 | ライセンス |
|---|---|---|---|---|
| `detect` | `detect` | RTMDet-tiny、RTMDet-s (8 GB)、RTMDet-m (16 GB) | ONNX、`bukuroo/RTMDet-ONNX` (MMDetection の COCO の重みを MMDeploy で書き出したもの。box AP 41.1 / 44.6 / 49.1) | Apache-2.0 |
| `pose` | `pose` | RTMO-s (8 GB) | ONNX、`Xenova/RTMO-s` (9.9M、COCO の 17 関節、人の検出器なし) | Apache-2.0 |
| `ocr` | `ocr` | PP-OCRv6-small (8 GB) | ONNX、`PaddlePaddle/PP-OCRv6_small_rec_onnx` と、`with: {detector: ...}` の検出器 (50 言語、仮名を含む 18708 字) | Apache-2.0 |
| `face-detect` | `face-detect` | YuNet (8 GB) | ONNX、`opencv/face_detection_yunet` (2023mar) | MIT |
| `segment` | `segment` | EfficientSAM-Ti (8 GB)、EfficientSAM-S (16 GB) | 作者の ONNX の符号器と復号器、`yunyangx/EfficientSAM` | Apache-2.0 |
| `depth` | `depth` | DA3-SMALL (8 GB)、DA3-BASE (16 GB)、DA3MONO-LARGE (24 GB)。ほかに DA3METRIC-LARGE、moge-2-vits / vitb / vitl | safetensors / `model.pt` のチェックポイント | Apache-2.0 (DA3)、MIT (MoGe-2) |
| `label`、`embed` | `label` | siglip2-base-patch16-224 (8 GB)、siglip2-large-patch16-256 (16 GB) | safetensors のチェックポイント | Apache-2.0 |

- どのモデルも初めて使うときに名前で取得し、[models.yaml](../../models.yaml) の sha256 で固定する。`with:` で 2 つ目のリポジトリを読む (`chat.Model.With`、ハッシュは `detector/FILE`)
- 出力は JSON 行 (箱、ラベル、文字列、関節)。`--draw` (face-detect) で描いた画像、`--repeat` で時間。1 枚ずつで、流れ (streaming) はない
- `depth` は `NAME.depth.png` (逆深度を 8 ビットに、入力の大きさで) を書く。`--raw` で float32 (MoGe-2 は点も)、`--sky` で空 / 確信度 (DA3-SMALL/BASE) / マスク (MoGe-2)、`--normal` (MoGe-2)、`--focal` / `--fov` (DA3METRIC のメートル。MoGe-2 は `--fov` を与えると z のずれだけを解く)、`--export-size`、複数視点の書き出しは `--ref` と `--ply`
- `label`: 候補 (`-l`、どの言語でも) のゼロショット。なければ組み込みの約 440 ラベル。ラベルごとに独立の sigmoid で、softmax ではない。`embed --compare` は絵と文を比べる

## どこで動くか

| | Apple (Neural Engine) | AMD (XDNA 2) | Intel の NPU | CPU |
|---|---|---|---|---|
| RTMDet-tiny / s | Core AI | CPU (`device: {amd: cpu}`) | OpenVINO IR | ONNX Runtime、なければ milcpu |
| RTMDet-m | Core AI | NPU、初回に IRON で書き出す | OpenVINO IR | 同じ |
| RTMO-s | Core AI | NPU、初回に IRON で書き出す | OpenVINO IR | 同じ |
| PP-OCRv6-small | Core AI | CPU (IRON の形なし) | OpenVINO IR | 同じ |
| YuNet | Core AI | 既定は CPU。`--device npu` で `iron_yunet.py` の書き出し | OpenVINO IR | ONNX Runtime、なければ milcpu |
| EfficientSAM の符号器 | Core AI | NPU、初回に IRON で書き出す | OpenVINO IR | ONNX Runtime、なければ hwnpu の評価器 |
| EfficientSAM の復号器 | ホストの評価器 | ホストの評価器 | ホストの評価器 | ホストの評価器 |
| DA3、MoGe-2 | Core AI | NPU (`iron_da3.py`、`iron_moge.py` の書き出し) | OpenVINO IR | — |
| SigLIP 2 | Core AI | NPU (`iron_siglip2.py` の書き出し) | 形はまだない | — |

- **Apple は Core AI だけ**。Mac の `hwnpu-image` は macOS 27 が要り、Core AI がなければエラー (`errNoCoreAI`)。Core ML へは落とさない。`--backend coreml` と `HWNPU_BACKEND=coreml` は Core AI に読み替える。プログラムは ONNX やチェックポイントから直接 `.aimodel` に書く。[apple-coreai.md](apple-coreai.md)
- `--cpu` (detect、pose、ocr。label/embed も) は Core ML の `.mlpackage` を CPU インタプリタ (`milcpu`) で動かす。変換の確かめ用で、遅い
- **Intel** (OpenVINO の CPU プラグインだけで確かめた。`HWNPU_BACKEND=openvino`、Ryzen の機械): pose、detect、ocr、face-detect、depth (DA3、MoGe-2)、segment は CPU や AMD の経路と同じ結果。NPU のコンパイラが DepthToSpace、ConvolutionBackpropData、Clamp、MaxPool、EfficientSAM の 4096 トークンの注意を受け付けるかは未確認。[intel.md](intel.md)
- **置き場所**: models.yaml の `device` をベンダーごとに (`{amd: cpu}`。書かないベンダーは npu)。上書きの順は `--device`、`HWNPU_DEVICE`、models.yaml (`chat.ChooseDevice`、理由も返す)。`chat.ConvertedDevice` は NPU の形のない ONNX のモデルを CPU に置く。この NPU の形がないモデルは動ける所に落ちてその旨を出し、`--device npu` を明示したときだけエラー。`hwnpu-image list` の RUNS FROM に置き場所 (`CPU (ONNX Runtime)` か書き出し)
- `hwnpu-image` は Neural Engine のロックを取る (`llm.OpenDevice`)。配置は厳密 (確かめられないときは理由を出す)。`aicheck run` と `probe` もロックを取る (`llm.LockNPU`)。[apple-neural-engine.ja.md](apple-neural-engine.ja.md)

### CPU の経路

- `internal/onnxrt`: ONNX Runtime の共有ライブラリを purego で呼ぶ (cgo なし)。OrtApi 16 以上を求め、関数は表の位置で呼ぶ。既定 1 スレッド、スピンなし (`session.*.allow_spinning = 0`)。ライブラリは `HWNPU_ONNXRUNTIME` かシステムのもの。自動で取ってこない (ネイティブのライブラリの配布は利用者が決める)。`HWNPU_ONNXRUNTIME=none` で使わない
- `onnxrt.Cut` は ONNX のグラフを 2 つの値の間で切る。`onnxrt.Program` は `onnx.Program` と同じ入出力 (`Flat` なら行) を返すので、ホストの処理 (箱の復号、関節、CTC) は変わらない
- ONNX Runtime がなければ milcpu。`internal/sgemm` (GotoBLAS の形の float32 の行列積。amd64 は AVX2 / AVX-512、arm64 は NEON の 6 × 16 で M3 の 1 スレッド 106 GFLOPS。im2col は B を詰めながら) と SIMD の exp / sigmoid / SiLU / erf (`internal/f32`)。`--threads` (既定 1: CPU 時間とエネルギーが最小)

| 1 スレッド、Ryzen AI 9 365 | ONNX Runtime | milcpu |
|---|---|---|
| RTMDet-m (640²) | 1.37 秒 | 2.25 秒 (4 スレッドで 1.0 秒) |
| RTMDet-tiny | — | 0.70 秒 |
| RTMO-s | 0.45 秒 | 0.89 秒 |
| PP-OCRv6 small (3 行) | 1.8 秒 | 4.5 秒 (4 スレッドで 2.6 秒) |

## ONNX だけで配られるモデル (`internal/onnx`)

モデルごとに手で移す代わりの共通の道具。グラフには静的な部分 (CNN、encoder) と動的な尾部 (TopK、NMS、人ごとの分類器) が混ざる。

- 依存なしで ONNX を読む (protowire。ノード、属性、初期値、入出力)
- ホストの評価器 (float64、numpy の放送): 形の計算、TopK、NonMaxSuppression、Conv、Pad (定数、反射)、MatMul、Softmax、Reduce*、If、ScatterND、ConvTranspose、InstanceNormalization、LayerNormalization、Resize (nearest、linear、cubic)、OneHot。大きな行列積 (積和 2^22 以上) は sgemm の float32
- `onnx.binop` は、片方が出力と同じ形でもう片方がその末尾の軸の繰り返しなら、添字の表なしで回す (`TestBinopRepeats`)
- MIL への下ろし (`Program`): 形と定数だけで決まる値 (Shape / Gather / Concat の連鎖) はホストで畳み込み、入力から計算する値だけをプログラムにする。Pow 2 → mul、Neg → mul −1、Clip → maximum と minimum、定数の行列との MatMul (と Add) → bias つきの 1 つの linear。HardSigmoid → clip、BatchNormalization → チャネルごとの積と和、GlobalAveragePool → reduce_mean、AveragePool → 重み 1/(kh kw) の深さ方向の畳み込み、核 = ストライドの ConvTranspose → 1 × 1 の畳み込みと `pixel_shuffle`、`auto_pad` の SAME_UPPER/LOWER、自分の大きさへの Resize、LayerNormalization
- `Graph.Cut`: 指定した動的な演算とそれに依存するノードを尾部にし、尾部が読む静的な側の値をプログラムの出力にする。尾部は `EvalShapes` がその出力からホストで評価する
- 形の違う出力は `(1, 行, 幅)` で連結する。行は全出力の大きさの公約数で、どの軸も 16384 を超えない (`Flat`。`FlatRows` / `SplitFlat` はグラフなしで分ける)。最後の軸での連結は Neural Engine で出力が崩れた
- 窓 9 と 13 の max pool (ストライド 1) は窓 5 の連続に (同じ最大値)。大きな窓は M1 で Neural Engine から外れた
- 数字で始まる初期値の名前 (YuNet の `420`) は Core ML が受けないので `w_` を前に
- `onnx.Program.ExpSigmoid`: Sigmoid を `1 / (1 + exp(-x))`、SiLU を `x / (1 + exp(-x))` と書く (落とし穴の表)。RTMDet と PP-OCR で入、RTMO、YuNet (と Basic Pitch) は切

## AMD の ONNX → IRON 変換器 (`tools/amd/iron_onnxcnn.py`)

ONNX の CNN (CSPNeXt、YOLOX 型) を IRON のグラフ 1 つにする。AMD の NPU は演算の切り替えに約 250 µs かかるので、畳み込み 1 つを行列積 1 回にする。詳しくは [../amd-npu-models.md](../amd-npu-models.md)、共通の IRON の部品は [amd.md](amd.md)。

- テンソル: 2 画素の 0 で囲んだチャネル後置きの格子 (`G`)、チャネルは 64 の倍数。最後の 2 チャネルは「1 のチャネル」(画像の中で 16、パッドで 0)。バイアスはここに乗り、SiLU は行列積の後段
- 演算: 1 × 1 (格子全体に行列積 1 回)、k × k のストライド 1 (k·Cp ≤ 512 なら `iron_gridgemm.GridGEMM`、ほかは im2col の写し)、ストライド 2 (im2col の写し 3 回)、深さ方向 (ブロック対角の重み)、チャネル注意 (1 の行との行列積、GEMV の fc、clamp、行ごとの積)、SPPF の max pool (窓 2/4/8 の組み合わせ。max(a, b) = b + ReLU(a − b)、パッド −1)、最近傍の 2 倍 (位相 4 つの写し)、足し算、連結、Focus (6 × 6、ストライド 2 の畳み込みに)、Split、group の畳み込み、AIFI の Transformer の層 (32 次元の頭を IRON の MHA 向けに 64 に埋める、尺度 1/√32、LayerNorm、erf の GELU、HIGHEST)
- 箱の頭の exp とストライドはホスト (`detect.NPUOutputs`、`heads.json` に残す)。格子の状態は同じ形の次の格子に回す (固定 1451 → 565 MiB)
- 書き出しの読み書きと実行は detect と pose で共有 (`internal/vision/ironcnn`)。初回に `chat.RunIRON` で作る (`~/models/rtmdet-m.amd-npu2`、`rtmo-s.amd-npu2`)。IRON がなければ CPU
- 分かった決まり (書き出しに入っている):
  - 1 のチャネルはちょうど 16 に。silu(16) は bf16 で 16 で、小さなずれは丸めで戻る。1 (silu(z) = 1) だと層ごとに 1 ulp 落ちた (stage 2 で cos 0.904)
  - max pool のパッドは −1。−30000 だと a − b が bf16 で 128 刻みになり、右端と下端が壊れる
  - 深さ方向のチャネルを大きさの順に並べ、bfp16 の指数を共有する 8 チャネルを揃える
  - バイアスを上位と下位の bf16 に分け、2 本の 1 のチャネルに乗せる

## モデルごと

### RTMDet (`hwnpu-image detect`)

- プログラムは頭の畳み込み (段ごとのクラスの logit と箱の 4 つの距離) まで。グラフはその後で並べ替え、sigmoid、復号、上位 5000、クラスごとの NMS をホストでする。ホストは logit が最小スコアの logit を超えるセルだけを復号する (sigmoid は単調)。Neural Engine には並べ替えがない。入力は 640 × 640 に収める
- `detect.Open` は出力の大きさとストライドからクラスの数を出す (ONNX を読み直さない。m の重みは float64 で 529 MB だった)
- Apple (M1): tiny 1 枚 13〜15 ms (ネットワーク 7.0 ms)、s 15〜17 ms (9.3)、m 24〜26 ms (18.2)。縮小 5〜7 ms。footprint 56 / 73 / 141 MB、Neural Engine 11 / 21 / 56 MB
- 精度 (s、ONNX Runtime と): スコアの差 0.063 / 0.042、箱 5.5 / 6.7 px (スコア 0.1 以上)。スコア 0.3 以上の物は同じ
- AMD: RTMDet-m 1 枚 0.33 秒・1.86 J。CPU の ONNX Runtime 1 スレッド 0.54 秒・8.2〜10.7 J (balanced)、milcpu 1.0 秒・16 J。ONNX Runtime と同じ物、スコアの差 0.013 以内、箱 1.3〜7.6 px。固定 565 MiB、プロセス 98 MiB。tiny と s は CPU (採らなかったもの)

### RTMO-s (`hwnpu-image pose`)

- 0〜255 の RGB を 640 × 640 に引き伸ばす。2000 アンカーのスコア、箱、姿勢の特徴、見え方までの 370 演算すべてが NPU。TopK、NMS、人ごとの座標分類器 (DCC、GAU、sin/cos の位置埋め込み) はホストの評価器
- Apple: M1 で 1 枚約 100 ms (ネットワーク 9〜100 ms、ホスト 36〜43 ms)、footprint 232 MB。M3 の Core AI で 87〜88 ms (うちホスト 70 ms)。ONNX の参照と: 箱 1.6 px、見える関節 3.4 px (fp16 の重みの丸めと同じ)
- AMD: 頭の畳み込みまで NPU (`ironcnn.LastConvs`)、出力 8 つをホストへ。HIGH と SiLU の演算で、関節 2 px 以内、スコア 0.03 以内、1 枚 0.22 秒。固定 327 MiB、プロセス 215 MiB。MEDIUM (0.15 秒) は関節のスコアが 0.13、確信の低い枠が 97 px ずれた

### PP-OCRv6 small (`hwnpu-image ocr`)

- 検出器は 960 × 960、1280 × 736、736 × 1280 のうち縮める割合が最も小さいもの。画像は左上、残りは正規化の後 0。M1 で 13 ms。PaddleOCR 3 の既定 (縮めずに 32 の倍数) と比べ、論文 1 ページの 98 行のうち 94 行が同じ (CER 0.12 %)
- 認識器: logit (8 行 × 160 ステップ × 18710 クラス) を出力にし、softmax は `Decode` が最大の周りだけで計算する。最後の linear を 9355 クラスずつ 2 つに分けて (…, 2, 9355) に積む ((…, 18710) と同じ並び)。bias も linear に入れる。全部 NPU
- 出力は fp16 のまま受け取る (`runner.F16`)。最大は fp16 のビットを整数として比べ (`sortKey`)、行の幅の中のステップだけを読み、確率は最大から 16 以内の値だけ。読み取りは次の呼び出しの間に別の goroutine で
- **認識器は行の後ろの 0 の幅で読み方が変わる** (SVTR の注意が幅全体を見る): 230 px の「SUB TOTAL」は 320 幅で正しく、640 幅では「SUBTOTAL」。PaddleOCR と同じく、1 回の行をその中で最も広い行の幅にそろえる。幅は `ocr.Widths` = 320、480、640、960、1280、1920、2560 (前の 1.5 倍まで)。1 回に最大 8 行、5120 列 (1280 で 4 行、2560 で 2 行)。幅ごとのプログラムは初めて使うときに変換 (コンパイル 5〜20 秒)
- M1: 日本語の 4 行 58 ms。論文 1 ページ (1224 × 1584、98 行) 488 ms (検出 13、箱 16、切り出し 34、認識 342 ms を 30 回)。レシート (13 行) 41 ms。footprint は最大 425 MB (7 つの幅と 12〜24 MB の出力)
- ONNX Runtime の PaddleOCR (`tools/ref/ref_ppocr.py --fit 960x960`、`tools/ref/compare_ocr.py`) と比べ、論文 93/98、レシート 13/13、日本語 3/4 行が一字一句同じ (CER 0.20 %、0、3.9 %)。NPU と CPU インタプリタは同じ行を出す
- 行の順は PaddleOCR の `sorted_boxes` (上から、10 px 以内は左から) で、2 段組みは左右交互。「〜」(U+301C) は辞書にあるが読まれない (モデルの読み方。PaddleOCR も)
- AMD: CPU (ONNX Runtime で 3 行 1.8 秒)

### YuNet (`hwnpu-image face-detect`)

- Apple / Intel: ONNX のグラフを丸ごと `internal/onnx` のプログラム 1 つに (12 の出力を 1 つに)。Apple: 640 × 480 で 3.4 ms (M1)、M3 の Core AI で 4.2〜4.4 ms。OpenCV の FaceDetectorYN と箱と目印が 0.3 px 以内
- AMD (`tools/amd/iron_yunet.py`): 1 × 1 と深さ方向の 3 × 3 を 1 つの 3 × 3 に、バイアスは 1 のチャネルに、ReLU は ProjectionGEMM の clamp (`ReluGEMM`)。画素を 255 で割り、1 を im2col の最後の 8 列に離す (bfp16 の共有の指数で消えた。cos 0.69)。`iron.GraphOptions.KeepRuns` で開く (コマンドの run を持ち続ける。ホストの CPU 1.6 → 1.1 ms。制御コードがデバイスのヒープ 64 MiB に残るので大きなグラフには使わない)。FaceDetectorYN と一致
- AMD の 320 × 256 のエネルギー: NPU 29 ms・127 mJ、CPU (ONNX Runtime 1 スレッド) 2.7 ms・7.7 mJ。演算の切り替え (1 回約 250 µs) が支配的。だから `device: {amd: cpu}`。これは AMD の IRON の話で、Apple と Intel はグラフをまとめて投入するので NPU のまま。あるベンダーの結果をほかへ持ち込まない
- ONNX Runtime がなければ、CPU は ONNX のグラフを Core ML に変換したものを `milcpu` で動かす (前は AMD の NPU に回していた)。どちらかはストアに書き (`runtime: hwnpu`)、役の `Usable` が ONNX Runtime の出入りで準備し直させる。Ryzen AI 9 365、320 × 256、`--repeat 2000`、1 スレッド、`gpu_metrics` のパッケージ電力からアイドルを引く: `milcpu` 1 枚 2.32 ms・約 30 mJ (下の速くする前は 12.58 ms・243 mJ)、ONNX Runtime 1.32 は 2.15 ms・約 20 mJ、NPU は 30.7 ms・約 184 mJ。顔は ONNX Runtime と同じ (どちらも score 0.934)。この測り方では ONNX Runtime は上の 7.7 mJ でなく 20 mJ になる: 比べるのは同じ測り方どうしで
- この種の CNN のための `milcpu` (`internal/milcpu`、`internal/f32`): 実行の値のバッファを次の値に使い回す (`pool.go`。GC が CPU の 4 割を取り、全部のコアを起こしていた)。stride 1 の深さ方向の 3 × 3 は、詰め物をした 3 行から 1 行を (`f32.Depthwise3`)。ほかの小さな畳み込み (タップ 64 以下: YuNet の最初の stride 2) は、位相に分けた入力の行の重み付きの和で (出力のチャネルを 8 つずつ、残りを 4 つ、1 つずつ: `f32.WeightedRows8`、`WeightedRows4`、`WeightedRows`)。1 × 1 の畳み込み (入力 256 チャネル以下) も同じく、詰めた列の塊で sgemm の代わりに (4 KB の倍数の面はどの行の同じ列も L1 の同じ組に入れた: 詰めた行を 16 個ずつずらす)。ReLU だけが読む畳み込みは ReLU も自分で計算。2 × 2 の max pool、ReLU、最近傍の 2 倍は専用のループ。YuNet のプログラム 11.6 → 1.85 ms (1 スレッド)。カーネルは amd64 では AVX2/FMA、arm64 では NEON (`f32.Axpy` も。exp、sigmoid、SiLU、`f32.GLU` は `f32_arm64.s`: Cephes の expf を AVX2 のループと同じ手順で)。NEON の 8 つの塊は、8 つの出力の 8 列を 32 本のレジスタのうち 16 本に持つ (行の 2 ベクトルとタップの重みの 2 ベクトルを読んで FMA 16 回)。amd64 では 4 つずつ 2 回: 1 回で 8 つ (AVX2 の 8 列で 16 本のレジスタを使い切る形、AVX-512 の 32 列) は Zen 5 で速くならなかった。`WeightedRows4` がすでにコアの FMA の上限 (約 145 GFLOPS) で動いている。amd64 の結果は 4 つずつの塊とビット単位で同じ。NEON のカーネルは qemu-user (テスト、YuNet の顔、22 の Core ML のパッケージを Go のループと比べて) と M2 (このマシンの 15 のパッケージ: Go のループとの差は最大値の 7e-6 以内) で確かめた。M2 の 1 スレッド: YuNet 320×256 24.0 → 2.7 ms、640×480 89.6 → 9.0 ms (`face-detect --device cpu` 1 枚 92.7 → 10.6 ms)、SwiftF0 1.09 → 0.62 秒、Supertonic のエンコーダ 105 → 24 ms。`dw3NEON` はどの鎖も FMUL で始め、バイアスは最後に足す (dw3AVX2 の順): バイアスのコピーや 0 にしたレジスタから始めると最初の FMLA がそれを待ち、YuNet 640×480 が 9.0 ms でなく 10.0 ms だった。1×1 の畳み込みが M2 で 4 つずつより速いのは 8 つの塊のおかげ (プロファイルで `wrows8NEON` 1.66 秒、`wrows4NEON` 1.99 秒)

### EfficientSAM (`hwnpu-image segment`)

- 符号器 (1024 × 1024 の ViT と首) は NPU、プロンプトの復号器 (点や箱の符号化、mask の復号) はプロンプトごとにホストの評価器 (If、ScatterND、点の数で形が変わる)。ホストが ONNX と同じ双線形 (half_pixel) で先に 1024 × 1024 にする (`segment.OpenProgram`)
- AMD (`tools/amd/iron_efficientsam.py`。64 次元の頭で IRON の MHA にそのまま載る): 符号器 Ti 0.34 秒、S 0.48 秒 (CPU の ONNX Runtime 1.9 秒、4.5 秒)。NPU と CPU の mask の IoU 0.93〜1.0、既定で書く一番の mask は 0.989 以上。Ti 固定 118 MiB / プロセス 274 MiB、S 158 / 267 MiB
- Apple: 符号器は既定で Neural Engine (`--device cpu` か `HWNPU_DEVICE=cpu` で CPU)。M3: S 0.35 秒 (Core ML) / 0.43 秒 (Core AI)、Ti 0.15 / 0.19 秒。全部 Neural Engine。CPU との mask の IoU 0.9994 (S)、0.9986 (Ti)。初回のコンパイル Ti 13 秒、S 22 秒。footprint 0.22〜0.27 GB。M3 で 1 枚 0.72 秒 (復号器 0.28 秒)
- ホストの復号器: M3 で 0.28 秒、AMD で 0.6〜0.7 秒。ONNX の参照評価器と一致 (IoU の予測 1e-6、mask の符号は全画素)
- CPU の符号器: ONNX Runtime、なければ hwnpu の評価器 (Ti 16〜30 秒、S 33 秒。外部依存なし)

### Depth Anything 3 (`hwnpu-image depth`)

- DA3MONO-LARGE (0.33B、DINOv2 ViT-L と DPT。相対の深度と空)、DA3METRIC-LARGE (同じ構造。メートル = 深度 × 焦点距離 (画素) ÷ 300、`--focal` か `--fov` が要る)、DA3-SMALL / BASE (複数視点のモデルを 1 枚として。深度と確信度)
- 大きさ: 公式の大きさ (長辺 504、短辺は 14 の倍数) で 9 つの縦横比。画像の公式の大きさがあればそれを、なければ最も近い縦横比 (`depth.Official`、`Choose`)。近い大きさへの引き伸ばしは公式の出力と 6〜12 % ずれ、NPU の誤差 1 % より大きかった
- ホスト (`internal/vision/depth`): 縮小 (双三次)、パッチ、深度の exp、空 (0.3 以上) を残りの 99 パーセンタイルで埋める、確信度は exp + 1、入力の大きさへ双線形で戻す
- AMD (`iron_da3.py`): 行を 16 の倍数に足した `Grid`、ストライド 2 の 3 × 3、転置畳み込み (GEMM と画素への Copy)、補間の行列の GEMM 2 回の双線形、q と k の次元の並べ替えによる 2 次元の RoPE、ヘッドのチャネルを 64 の倍数に。504 × 378 で SMALL 0.26 秒、LARGE 0.7 秒。固定 0.49〜1.39 GiB。公式の CPU fp32 と深度のコサイン 0.9993〜0.9999
- 複数視点 (`iron_da3.py --views N`、AMD のみ): 視点のトークンを詰めて並べ、視点の中は窓の注意、視点をまたぐ注意は全部の行。参照する視点の選択とカメラの MLP はホスト (`depth.SaddleBalanced`)。`cameras.json` と `--ply` を書く。3 枚で 1.1 秒、コサイン 0.9998。ほぼ同じ画像どうしでは公式と違う参照の視点を選ぶことがある (`--ref`)
- Apple (`internal/vision/depth/mil`): 転置畳み込みは 1 × 1 と `pixel_shuffle`、双線形は行列の積。書き出しは 1 つで、大きさごとの `WxH/` を初めて使うときに変換 (数秒)。M3 の Core AI: DA3-BASE 504² の ViT 106 ms、頭 16 ms。Core ML と深度のコサイン 1.000000。公式のモジュールと DA3-SMALL 1.000000、DA3MONO-LARGE 0.999993 (空 0.9998)

### 深度のモデルの比較 (Ryzen AI 9 365、`ai/vision` を通して)

絵 3 枚 (1024×672 が 1 枚、1024×1024 が 2 枚)、ほかのセッションが NPU を使っている間。正解の深度がないので、見た目は灰色の絵を並べて目で比べた。

| モデル | 1 枚 (2 枚目から) | 最初の 1 枚 | NPU のメモリ | プロセス | 深度の大きさ | 書き出し | 見た目 |
|---|---|---|---|---|---|---|---|
| DA3-SMALL | 0.31 秒 | 0.57 秒 | 0.48 GiB | 0.17 GiB | 504 | 1.0 GB | 輪郭がぼやけ、細い枝や髪は消える |
| DA3-BASE | 0.43 秒 | 0.86 秒 | 0.74 GiB | 0.17 GiB | 504 | 2.3 GB | SMALL より少しはっきり |
| DA3MONO-LARGE | 0.85 秒 | 1.63 秒 | 1.36 GiB | 0.21 GiB | 504 | 6.2 GB | 輪郭がはっきり、背景の層も分かれる |
| DA3METRIC-LARGE | 0.85 秒 | 1.50 秒 | 1.36 GiB | 0.21 GiB | 504 | 6.2 GB | MONO と同じくらい。空や花に細かい粒が出る絵がある |
| moge-2-vits | 2.52 秒 | 3.15 秒 | 1.94 GiB | 0.44 GiB | 1024 | 2.4 GB | 輪郭は細かいが、背景の層の差が小さい |
| moge-2-vitb | 2.68 秒 | 3.26 秒 | 2.10 GiB | 0.42 GiB | 1024 | 3.6 GB | 輪郭が細かく、背景の層も分かれる |
| moge-2-vitl | 3.41 秒 | 4.17 秒 | 2.62 GiB | 0.50 GiB | 1024 | 7.7 GB | 一番細かい。空を「一番遠い」に切り分ける |

- 用途 `depth` の構成はそのまま: 8 GB から SMALL、16 GB から BASE、24 GB から MONO-LARGE。
- 近さ (深度の逆数) は、近いもののある絵ではその奥が全部 0 近くになる。`samples/parallax` は log(1 + k·n) / log(1 + k) (既定 k = 30) で広げる。

### MoGe-2 (`hwnpu-image depth -m moge`)

- 焦点距離を知らなくてもメートル単位の深度、点群、法線、マスク (DA3METRIC は焦点距離が要る)。DINOv2、ConvStack の neck、3 つのヘッド (点、法線、マスク)。正規化なし (`res_block_*_norm` が none)。3 × 3 はすべて端の画素を複製して詰める
- `model.pt` は torch なしで読む (`internal/torchzip`。zip の pickle から辞書とテンソルの組み立てだけを許す)。後処理は Go (`internal/vision/moge`): 焦点距離と z のずれを、64 × 64 に間引いた点で 1 変数の Levenberg–Marquardt (`TestRecoverFocalShift`)
- AMD (`iron_moge.py`): 複製の詰めは `Grid(rep=True)` と `fill_pads`。3 つのヘッドは 1 つのプログラムに重みのバンク 3 組 (`SwapVariants`)。3 つとも載せたままだと固定 3.9 GiB。812 × 602 で vits 約 2.7 秒、vitl 約 3.6 秒。深度のコサイン 0.9999、相対誤差の平均 0.7〜1.2 %。時間の大半は 16 倍の解像度のヘッド
- Apple (`internal/vision/moge/mil`): ViT は depth/mil のもの。neck とヘッドは 1 つのプログラム (複製は `pad` と詰めなしの畳み込み。Core ML は `pad_type` valid でも `pad` を求める)。ViT-S: ヘッドの出力のコサイン 0.99993 以上、尺度の logit 1.6885 (公式 1.6853)。1 枚 0.3〜0.5 秒
- 700 × 700 の `dense` で、マスクのヘッドの最後の `pixel_shuffle` (256 チャネル → 64 × 400 × 400) が M1 の Core AI で Neural Engine から外れた (`empty reset layer candidates in SS cost model`、GPU に回る)。出力の行の帯に分ける (`mil.deconv2`、`shuffleRows` 256 行まで)。値は同じ

### SigLIP 2 (`hwnpu-image label`、`embed`)

- ホスト (`internal/vision/siglip`): 縮小 (PIL の双線形) して [−1, 1]、パッチ。文は小文字に (そう学習した。Transformers のトークナイザは小文字にしない)、Gemma のトークナイザで 63 トークンに切って `<eos>`、64 まで埋め、マスクなし。文の型は `this is a photo of {}.`。埋め込みを正規化し、確率 = sigmoid(コサイン × exp(logit_scale) + bias)
- ラベルの埋め込みはユーザーのキャッシュ (`hwnpu/siglip/NAME-HASH.f32`) に置き、同じ一覧は 1 回だけ計算する
- 文の塔: 1 回に 8 つ。行はトークンが先で、8 つの文が 8 × heads 個の頭の 1 回の注意になる。足りない分は最初の文の写しで埋める (0 の行は fp16 の LayerNorm で 0/0 になる)
- AMD (`iron_siglip2.py`、`--precision HIGHEST`、`--gelu erf`): IRON の表を引く tanh の GELU (bf16 の丸めの 5 倍、IRON の参照との照合では見えない) をやめ、exact の GELU。Transformers の float32 と画像のコサイン 0.9998〜0.9999。Ryzen AI 9 365: base 1 枚 84 ms、8 文 131 ms、760 MiB。large 353 ms / 491 ms、1579 MiB
- Apple (`internal/vision/siglip/mil`、`siglip.Prepare`): 初回に `config.json`、`tokenizer.json`、`model.safetensors` から Go で `NAME-v1` に変換 (vision と text のプログラム、ホストが引く fp16 のトークンの表 `text/embed.bin`、`model.json`)。base 1.9 秒、large 3.5 秒。どちらの塔も fp16。LayerNorm は行を最大値で割ってから (`ScaledLayerNorm`)、tanh の GELU は exp で書く。probe の問い合わせは定数にして 8 行に並べる (1 行だけの部品は Neural Engine から外れる)
- M3: 画像のコサイン 0.99986 以上、文 0.999992 以上 (Core AI)。ラベルの確率の差 0.007 以下。base 1 枚 7.3〜7.9 ms、既定の 444 ラベルは 1 回 1.7〜2.0 秒。large 33〜36 ms、4.0〜4.8 秒。Neural Engine: base 357 MB (Core ML) / 476 MB (Core AI)、large 1240 / 1556 MB。models.yaml の memory: Apple 0.9 / 3.3 GB、AMD 0.8 / 1.7 GB

## Neural Engine の落とし穴 (これらのモデルで)

| 落とし穴 | 影響 | 対処 |
|---|---|---|
| Neural Engine の `sigmoid` と `tanh` は深い CNN でずれる (RTMDet-s の SiLU 80 個) | 箱 98 px、スコア 0.35 ずれ、物が 1 つ減る (CPU の fp16 は 7.5 px) | `ExpSigmoid`: 5.5 px、スコア 0.063。演算が 2 つ増え s 6.6 → 9.3 ms、m 12.2 → 18.2 ms。精度を取る。`0.5 x (1 + tanh(x/2))` もずれる |
| 1 つの軸が 16384 を超えるテンソルの演算は、出力の identity でも | PP-OCR の 18710 クラスの softmax、bias の足し算、出力が CPU に | logit を出力に、linear を 2 つに分ける、bias を畳み込む |
| Core AI の Neural Engine は定数による `divide_v1` を誤る (√2 で割らずに √2 倍、チャネルごとのずれも) | PP-OCR の検出器のコサイン 0.05、認識器 −0.42 | `ailower` が `x × (1/c)` と書く。定数で割る MIL は ONNX のグラフだけなので、ほかのプログラムのバイトは変わらない。確かめるときは `HWNPU_DEV=ailower.divide=1` で割り算に戻す |
| Core AI の CPU は `reduce_mean` の和を fp32 でとり、1/n の前に fp16 に戻す | 平均の絶対値が 65504/n を超えると −inf (57600 要素の SE ブロック) | Core AI の CPU (`HWNPU_COREAI_UNITS=cpu`) だけの問題。全部のプログラムのハッシュが変わるので直さない |
| Core AI の `max_pool` | 下ろせなかった | coreai-torch の書き方どおり `pad_v1` (−inf) と `primitive.max_pool_2d_v1` |
| Core AI の `conv_transpose` と MIL の pad (負もある) | — | padding 0 で全体を出し、`slice_v1`、0 で `pad_v1`、そのあと bias |
| `pad` の replicate / reflect | — | `#aicode.padding_mode_v1<replicate>` / `<reflect>` |

- 両方のキャッシュが当たっていれば Core AI と Core ML の起動は同じ (M3、新しいプロセスで 3 回ずつ。Core AI が 1 コマンド 0.01〜0.05 秒短い程度)。Core ML のキャッシュ (約 4 GiB) は外れ、Core AI はディスクが減るまで持つ。[apple-coreai.md](apple-coreai.md)
- M3 の 1 回の呼び出し、Core AI / Core ML: PP-OCR 検出器 13.0 / 9.7 ms、認識器 8×320 5.1 / 4.0 ms、RTMDet-m 10.2 / 13.5 ms、RTMO-s 8.0 / 4.9 ms、YuNet 1.2 / 0.9 ms、DA3-BASE の ViT 106 / 103 ms。Core AI の初回のコンパイルは部品ごとに 1.6〜38 秒 (RTMO 36〜38 秒)。Neural Engine の持つ量は Core AI が多い (OCR 27 → 40 MB、RTMDet 56 → 89 MB)
- 確かめる道具: `aicheck run -cpu PKG` (Core AI) と `-coreml` で milcpu と比べる。`HWNPU_DEV=ailower.cut=N` で N 番目の演算の後で切ってその値を比べる。`AICHECK_WORST=1` で差の大きい要素を出す
- ホストの縮小 (`lfm3vl.ResizeRGB`、PIL の双線形) は画素ごとのクロージャなしで行を並列に: 37〜55 ms から 5〜7 ms、バイトは同じ (`TestResizeRGBSameBytes`)。バイトへの丸めは組み込みの `min` / `max`: `math.Max` と `math.Min` は呼び出しで (NaN と符号つきの 0 のため。和はどちらでもない)、CPU の 3 分の 1 を取っていた。`lfm3vl.RGB` はアルファを落とすのを行ごとに全部のコアで。M2 で 512² から 640²: 拡大縮小 4.2 → 1.2 ms、RTMDet-m の `Pixels` 5.4 → 2.2 ms。140 通り (双線形と双三次、RGBA と NRGBA、拡大と縮小) で前とバイトが同じ

## AMD の書き出しと共有

- detect、pose、segment (`amdFirstRunExport`): 手元の書き出しはそのまま使い、なければ取得元の層から、それもなければ IRON で作って書き戻す (`chat.SharedExport`)。手で作った書き出し (`entry.json` なし) は `ExportKind(tool)` と `PinnedRecipe(m, tool, version)` で記述し、初めて使ったときに取得元に置く (`chat.ShareMadeHere`)。置かないもの: `entry.json` のあるもの、`AMDRoot` の外 (`~/models`)、鍵のないもの (definitions が sha256 を持たない)。[distribution.md](distribution.md)
- depth (`~/models/NAME.amd-npu2`、小文字) と SigLIP 2 の書き出しは `tools/amd/iron_da3.py`、`iron_moge.py`、`iron_siglip2.py` で作る

## 未解決

- 電力量を測っていない: Apple の画像を読むモデル (`hwnpu-dev powerbench` は sudo が要る)、AMD の RTMO と EfficientSAM、SigLIP 2
- Intel の実機の NPU (全モデル)。SigLIP 2 は Intel の形がない (同じ MIL から IR に下ろせるが未確認)
- Core AI の M1 / M2 / M4 (docs/handoff-coreai-vision.md)。M1 では EfficientSAM の符号器の注意 (4096 × 4096 × 頭の数) が 64 MiB を超える
- Core AI の割り算の誤りが 3 つの演算だけのプログラムで再現しない理由。チップで違うか
- Core AI の CPU の `reduce_mean` のあふれ
- ONNX Runtime との照合は RTMDet-s だけ (tiny と m は未)。RTMO-m / l。PP-OCRv6 medium。180 度回った行 (文字の向きの分類器 PP-LCNet なし)。縦書き。段組みの解析 (PP-StructureV3)
- RTMO と YuNet の `ExpSigmoid` (RTMO も SiLU の CSPNeXt なので良くなる見込み)
- Apple の Neural Engine で DA3METRIC-LARGE と DA3-BASE を公式のモジュールと比べていない。MoGe-2 の ViT-B / L も Apple で比べていない。複数視点の DA3 は AMD だけ。光線のヘッド (`use_ray_pose`) とカメラを与える使い方 (`cam_enc`) は未対応
- AMD の MoGe-2 のヘッドが重い (16 倍の解像度、3 つのヘッドが同じ neck の出力を別々に処理)
- SigLIP 2: so400m (72 次元の頭が IRON の 64 の倍数の MHA に合わない)、NaFlex、SafeSearch にあたる閾値、`/v1/embeddings` で画像、パッチごとの特徴

## 採らなかったもの

- AMD の NPU の RTMDet-tiny / s: BN を畳み込んだ重みとバイアスが打ち消し合って出力が約 1/50。MEDIUM の bfp16 (A と B)、後段の SiLU (vtanh は |x| ≤ 0.5 で x を返す)、ほぼ定数のチャネルで、tiny のスコアは 0.2 ずれ、s は物が消える。全部 HIGH の tiny は 0.25〜0.30 秒で CPU (0.21 秒) より遅い。CPU に置く
- `iron_onnxcnn.py` で bfp16 向けにチャネルを並べ替える: tiny は良くなった (0.2 → 0.07) が RTMDet-m で物が消えた
- 重みを上位と下位の bf16 に分ける: 効かない (誤差は活性の bfp16 と SiLU)
- YuNet を AMD で既定の NPU に: CPU の 16 倍のエネルギー
- AMD の文の塔を 1 回 16 文に: 1 文 13.8 ms (8 文なら 16.4 ms) だが、候補が少ないと損。8 のまま
- SigLIP 2 の IRON の tanh の GELU と MEDIUM の行列積: 画像のコサイン 0.9988。exact の GELU と HIGHEST は時間が 9〜22 % 増える
- EfficientSAM の復号器を Neural Engine に: 形が動く (If、ScatterND、点の数)。ホストを速くした
- SigLIP のラベルの埋め込みを取得元の層で共有する: 数百 KB で、数秒で作り直せる
- DA3NESTED、DA3-LARGE、DA3-GIANT: CC BY-NC
- 切り抜きに BiRefNet: deformable conv がある。素の ViT の EfficientSAM にした
- 箱に視覚言語モデル (LFM3-VL): 箱ごとに数十トークン、1 枚数秒、小さな物を落とし、座標がずれる。1 段の検出器は 1 回で全部の箱を出す。視覚言語モデルは切り出しを説明させる後段に向く
- Mac の `hwnpu-image` で Core ML: Core AI だけ (Core ML の書き出しは `--cpu` のために残す)
