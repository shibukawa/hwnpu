# 画像のコマンド (hwnpu-image)

[English](USAGE_IMAGE.md)

`hwnpu-image` は、画像生成 (`generate`)、深度推定 (`depth`)、顔 (`face-detect`)、物 (`detect`)、人の姿勢 (`pose`)、文字 (`ocr`)、ラベル付けと埋め込み (`label`、`embed`)、切り抜き (`segment`) を NPU で行います。選べるモデルは `hwnpu-image list` で出ます。

## 画像生成 (Anima)

`go run ./cmd/hwnpu-image` は、Anima (circlestone-labs/Anima、Cosmos-Predict2 2B の DiT と Qwen3-0.6B、Qwen-Image の VAE) でプロンプトから絵を描きます。テキストの符号化、DiT のステップ、VAE の復号はすべて NPU の上の IRON のグラフで、ホストはトークン化、埋め込みの行の取り出し、パッチの並べ替え、Euler のステップ、PNG を受け持ちます。書き出しは画像の大きさと拡散モデルごとで、hwnpu の `tools/amd/iron_anima.py` が作ります (1024 × 1024 で 4.0 GB。DiT のブロックの重みは 8 ビット)。

```bash
python tools/amd/iron_anima.py ~/models/Anima ~/models/anima-turbo-1024.amd-npu2 --size 1024
go run ./cmd/hwnpu-image generate "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms"
```

Anima の拡散モデルは 2 つです。`-m` を省くと Anima-Turbo (`anima-turbo`) で、8 ステップ、CFG なしで描きます。`-m anima` は Anima-Aesthetic (質の高い絵で調整したもの) で、CFG 4 (`--guidance`、モデルカードの勧めは 4〜5) です。既定は DPM-Solver++ 2M の 25 ステップ、CFG は雑音の多い 17 ステップだけです (`--sampler euler --steps 30 --cfg-sigmas 0,1` でモデルカードのとおり 30 ステップ全部に CFG)。DiT を 1 枚に 42 回走らせるので Turbo の約 4.5 倍の時間がかかります。負のプロンプトは `--negative` です (既定は空。モデルカードの勧めは `"worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"` で、Aesthetic では score タグを入れない)。Turbo に `--guidance` なしで `--negative` を付けると、弱い CFG 1.5 で走ります (1 ステップに DiT を 2 回)。書き出しは `--dit` で拡散モデルを選びます。

```bash
python tools/amd/iron_anima.py ~/models/Anima ~/models/anima-aesthetic-1024.amd-npu2 --size 1024 --dit anima-aesthetic-v1.1.safetensors
go run ./cmd/hwnpu-image generate -m anima "masterpiece, best quality, safe, 1girl, cherry blossoms" --negative "worst quality, low quality"
```

AMD の NPU では、`-m` の名前で `~/models/NAME.amd-npu2` (`anima-turbo-1024.amd-npu2`、`anima-aesthetic-1024.amd-npu2` など) を使います。モデルは models.yaml の名前、ファミリー (`anima-turbo`、`anima`、`sana`、`sana:1.6b`)、書き出しのディレクトリのどれでも指定できます。ファミリーだけのときは、マシンのメモリで大きさを選びます (`sana` は 16 GB 以上で SANA 1.5 1.6B、それ未満で Sana 600M)。メモリは `-m sana:8g` (`8G`、`8GB`、表の列にない `12g` も可) のように明示でき、そのメモリのマシンで選ばれるものになります。`hwnpu-image list` (`list image`、`list depth`、`list face-detect`) が、モデル、AMD の書き出しの有無、ファミリーと用途がメモリごとに選ぶものを表で出します。`-m` に合うものがないときも同じ表を出します。名前で選んだモデルは `~/models/NAME.amd-npu2` (小文字、`-1024` 付きも) を探します。`-o` を省くと、今のディレクトリの `image.png` (あれば `image (2).png`、`image (3).png`、…) に書き、OS のビューア (macOS は open、Windows は関連付け、Linux は xdg-open) で開きます。

Ryzen AI 9 365 で、1024 × 1024 の 1 枚が Anima-Turbo の 8 ステップで約 44 秒 (DiT 1 ステップ 4.6 秒)、Anima-Aesthetic (CFG 4) で約 3 分 20 秒 (1 枚 3.8 kJ。Turbo は 715 J) です。Anima-Aesthetic は、モデルカードのオイラー法 30 ステップの代わりに DPM-Solver++ 2M の 25 ステップで、CFG は雑音の多いステップだけにかけます (絵は同じで、30 ステップのときは 4 分 40 秒、4.8 kJ。`--sampler euler --steps 30 --cfg-sigmas 0,1` で戻せます)。NPU に固定されるメモリは最大 2.4 GiB (VAE の段) で、どのモデルでも同じです。`~/models/Anima` の置き方と、精度の測定は [docs/amd-npu-models.md](docs/amd-npu-models.md) の 11 節にあります。

Mac の Neural Engine と Intel の NPU では、hwnpu-chat と同じく、初回の起動でチェックポイントを Hugging Face から落とし (約 5.4 GB)、Go だけで Core ML のパッケージか OpenVINO IR に書き出します (Python は要りません)。置き場所は hwnpu-chat と同じユーザーのキャッシュディレクトリの `hwnpu/models/Anima` (`--dir`) で、書き出しは画像の大きさ (`--size`) と重みの形式ごとです。初回は読み込みのときに NPU 向けのコンパイルもあり、1024 × 1024 で数分かかります。

```bash
go run ./cmd/hwnpu-image generate "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms" -o out.png
```

M3 (16 GB) の Neural Engine で、1024 × 1024 の 1 枚が 8 ステップで約 40 秒、プロセスのメモリは最大 3.8 GB です (読み込み 3 秒、DiT 1 ステップ 4.2〜4.4 秒、VAE 3.7 秒。既定の重みはテキストも DiT も int8)。M1 (16 GB) では Anima-Turbo が 1 枚約 58 秒、Anima-Aesthetic が約 4 分 30 秒 (DiT 1 回 6.3 秒、Neural Engine に 2.6 GiB、プロセスの最大 3.9 GB) です。記録は [docs/design.md](docs/design.md) の 16.85、16.152、16.160 にあります。Intel の NPU では未確認です ([docs/handoff-anima-mac-intel.md](docs/handoff-anima-mac-intel.md))。手で書き出すときは `hwnpu-image convert DIR --model OUT` (DIR は `~/models/Anima` と同じ置き方) を使います。

## 画像生成 (Sana)

`hwnpu-image generate -m sana` は NVlabs の Sana (線形注意の DiT、Gemma 2 2B、DC-AE の VAE) で描きます。Sana 600M と SANA 1.5 1.6B で、20 ステップと CFG (雑音の多い 14 ステップだけ) です。英語のプロンプト向けで、Mac の Neural Engine では画像生成のモデルでいちばん速く (M1 で Sana 600M 約 17 秒、SANA 1.5 約 31 秒、Anima-Turbo は約 58 秒)、Neural Engine が抱えるメモリも少なくて済みます (0.6 / 1.6 GiB、Anima-Turbo は 2.6 GiB)。

AMD の NPU では書き出しを渡します。書き出しは `tools/amd/iron_sana.py` が、Efficient-Large-Model の diffusers のリポジトリから作ります。2 ステップに蒸留した Sana-Sprint は、絵の質が Sana よりかなり低いので外しました ([docs/design.md](docs/design.md) の 16.148)。

```bash
python tools/amd/iron_sana.py ~/models/Sana_600M_1024px_diffusers ~/models/sana-600m-1024.amd-npu2
go run ./cmd/hwnpu-image generate -m sana:0.6b "a watercolor painting of a red fox in a snowy birch forest"
```

Ryzen AI 9 365 で、1024 × 1024 の 1 枚が Sana 600M で約 60 秒、SANA 1.5 1.6B で約 2 分です。NPU に固定されるメモリは最大 2.3 GiB です (Gemma 2 の 8 ビットの重み)。英語のプロンプトが前提です。`-m sana` はメモリで Sana 600M か SANA 1.5 1.6B を選びます。`--negative` で負のプロンプトを付けられます。正方形以外は Sana のビンの大きさ (`iron_sana.py --width 1152 --height 896` など) で書き出します。手順と精度は [docs/amd-npu-models.md](docs/amd-npu-models.md) の 14 節にあります。ライセンスは DiT と VAE が Apache-2.0 と MIT、Gemma 2 は Gemma の利用規約です。

Apple の Neural Engine では、`hwnpu-image generate -m sana "..."` などで Sana 600M と SANA 1.5 1.6B が動きます。Anima と同じく、初回にチェックポイントを `--dir/Sana` に落として Core ML のパッケージに変換し (10 秒)、各プログラムを Neural Engine 用にコンパイルします (初回は全体で約 8 分半)。M1 (16 GB) で、キャッシュのある 2 回目以降は 1 枚 Sana 600M が約 17 秒、SANA 1.5 が約 31 秒、プロセスのメモリは最大 3.9 GB です。diffusers の fp32 と同じノイズからの潜在のコサインは Sana 600M で 0.998、SANA 1.5 で 0.984 です (fp16 の DiT で全ステップ CFG のとき)。Gemma 2 は int8、DiT は Sana 600M が int8、SANA 1.5 が lut8、VAE は fp16 で、時刻の埋め込みだけホストで計算します。CFG は雑音の多い 14 ステップだけにかけます (`--cfg-sigmas 0,1` で全ステップ)。記録は [docs/design.md](docs/design.md) の 16.145 と 16.160 にあります。手で書き出すときは `hwnpu-image convert ~/models/Sana_600M_1024px_diffusers -m OUT` (`--text` で Gemma 2 の置き場所) を使います。 Intel の NPU (実機では未確認) では同じ手順で OpenVINO IR に変換します (`--backend intel`)。表を読まないので、SANA 1.5 の DiT は fp16 です。

## 画像生成 (Chroma1)

`hwnpu-image generate -m chroma` は lodestones の Chroma1 (FLUX.1 schnell の 8.9B の DiT、T5-XXL、FLUX の VAE、Apache-2.0) で描きます。`chroma` は Chroma1-Flash (Heun 8 ステップ、CFG なし)、`chroma:1-hd` は蒸留していない Chroma1-HD (CFG 3、DPM-Solver++ 2M 25 ステップで、CFG は雑音の多いステップだけ) です。

AMD の NPU では `tools/amd/iron_chroma.py` の書き出しを使います ([docs/amd-npu-models.md](docs/amd-npu-models.md))。Apple の Neural Engine では、初回にチェックポイント (27.5 GB) を `--dir/Chroma` に落とし、Go だけで Core ML のパッケージに変換して (約 1 分)、Neural Engine 用にコンパイルします。T5-XXL は lut8 (外れ値を出す FF の出力の射影だけ fp16) で 3 層ずつ読んで閉じ、DiT は lut8、VAE は fp16 で、全部が Neural Engine に載ります。M1 (16 GB) で、2 回目からの 1 枚は Chroma1-Flash の 512 × 512 で約 2 分、1024 × 1024 で約 9 分半、Chroma1-HD の 1024 × 1024 で約 22 分です (初回は Neural Engine 用のコンパイルが約 1 時間半。HD は Flash の書き出しから T5 と VAE を共有します)。DiT の重み (8.3 GiB) を Neural Engine に固定したまま回すので、16 GB 以上の Mac が要ります。CPU の float32 と同じノイズからの潜在のコサインは 0.987〜0.997 (3 プロンプト × 2 種、512²) で、絵は同じです。記録は [docs/design.md](docs/design.md) の 16.170 にあります。手で書き出すときは `hwnpu-image convert ~/models/Chroma1-Flash -m OUT --tokenizer T5_TOKENIZER.json` (T5 と VAE が別の場所なら `--text`) を使います。

## 深度推定 (Depth Anything 3、MoGe-2)

`go run ./cmd/hwnpu-image depth` は、AMD の NPU か Apple の Neural Engine (Intel の NPU も。実機では未確認) で写真の深度を推定します。モデルは Depth Anything 3 (Apache-2.0) の DA3MONO-LARGE (既定、相対の深度)、DA3METRIC-LARGE (`--fov` か `--focal` でメートル)、DA3-SMALL / BASE (軽い版、深度と確信度) です。画像ごとに `NAME.depth.png` (近いほど明るい) を書き、`--raw` で深度の float32、`--sky` で空 (SMALL / BASE は確信度) の画像も書きます。書き出しは縦横比ごとの大きさを持ち、画像に近いものを使います。`-m` は generate と同じく名前かファミリー (`da3:small`、`da3metric`、`moge`、`moge:vits`) か書き出しのディレクトリです。既定の用途 `depth` はメモリで選び、8 GB は DA3-SMALL、16 GB は DA3-BASE、24 GB 以上は DA3MONO-LARGE です (`depth:24g` や `-m DA3MONO-LARGE` で大きい方も)。

```bash
python tools/amd/iron_da3.py ~/models/DA3MONO-LARGE ~/models/da3mono-large.amd-npu2
go run ./cmd/hwnpu-image depth photo.jpg
```

Ryzen AI 9 365 で 1 枚 0.7 秒 (LARGE、504 × 378)、SMALL は 0.26 秒、NPU に固定されるメモリは最大 1.4 GiB (SMALL 0.5 GiB) です。DA3-SMALL / BASE を `--views 3` で書き出すと、同じ場面の 3 枚を同時に推定して、1 つの尺度の深度とカメラの位置 (`cameras.json`)、点群 (`--ply`) を出します。手順と精度は [docs/amd-npu-models.md](docs/amd-npu-models.md) の 15 節にあります。

Apple の Neural Engine では、名前だけで動きます (`hwnpu-image depth -m da3:small photo.jpg`、`-m moge:vits`)。初回にチェックポイントを取ってきて、縦横比ごとの大きさは初めて使うときに変換します (数秒、その大きさの最初の読み込みでコンパイル)。M1 で 504 × 504 の 1 枚が DA3-SMALL 0.2 秒、DA3MONO-LARGE 0.5 秒、MoGe-2 ViT-S (700 × 700) 0.5 秒で、公式のモジュールとの深度のコサインは 0.99999 以上です。`--views` (複数視点) は AMD だけです。記録は [docs/design.md](docs/design.md) の 16.145 にあります。

MoGe-2 (Ruicheng/moge-2-vit{s,b,l}-normal、MIT) の書き出し (`tools/amd/iron_moge.py`) を渡すと、焦点距離を知らなくてもメートル単位の深度、点群 (`--raw`)、法線 (`--normal`)、有効な画素のマスク (`--sky`) を出します。812 × 602 で 1 枚 2.7〜3.6 秒です (16 節)。

```bash
python tools/amd/iron_moge.py ~/models/moge-2-vits-normal ~/models/moge-2-vits.amd-npu2
go run ./cmd/hwnpu-image depth -m moge:vits --normal photo.jpg
```

## 顔の検出 (YuNet)

`go run ./cmd/hwnpu-image face-detect` は、写真の顔を探します (YuNet、OpenCV Zoo、MIT)。画像ごとに 1 行の JSON で、顔の枠 (x, y, w, h)、スコア、5 点 (右目、左目、鼻先、口の右端と左端) を出します。`--draw` で枠と点を描いた `NAME.faces.png` も書きます。大きさは 320×256 と 640×480 で、画像を縮める割合が最も小さいものを使います (`--size` で固定)。結果は OpenCV の FaceDetectorYN と一致します。

どこで動かすかは、models.yaml の `device` が NPU のベンダーごとに決めます。YuNet は AMD では CPU、Apple では Neural Engine です (ONNX のグラフをそのまま Core ML に変換、640×480 で 1 枚 3.4 ms)。AMD の NPU では、この小さなモデルの演算ごとの切り替えが重く、320×256 で 1 枚 30 ms・137 mJ かかるのに対し、CPU では 4 ms・10 mJ だからです (docs/amd-npu-models.md 17 節)。CPU では ONNX Runtime の共有ライブラリを使います。[リリース](https://github.com/microsoft/onnxruntime/releases)の `lib/libonnxruntime.so` を置いて、`HWNPU_ONNXRUNTIME` にそのパスを入れてください (システムにあればそれを使います)。ONNX のファイルは自動で取ってきます。`--device npu|cpu` か `HWNPU_DEVICE` で、置き場所を上書きできます (ゲームなどで CPU を空けたいときは `--device npu`)。`hwnpu-image list` の RUNS FROM の列に、このマシンでの置き場所が出ます。

```bash
python tools/amd/iron_yunet.py ~/models/face_detection_yunet ~/models/yunet.amd-npu2   # hf:opencv/face_detection_yunet
go run ./cmd/hwnpu-image face-detect --draw photo.jpg               # AMD: the CPU (ONNX Runtime)
go run ./cmd/hwnpu-image face-detect --device npu photo.jpg          # AMD's NPU: the export above
```

## 物の検出、姿勢、文字 (RTMDet、RTMO、PP-OCRv6)

`go run ./cmd/hwnpu-image detect` は、写真の物を探します (RTMDet、OpenMMLab MMDetection、Apache-2.0)。COCO の 80 種類 (人、車、自転車、信号、犬、カップ…) について、画像ごとに 1 行の JSON で、枠 (x1, y1, x2, y2)、ラベル、スコアを出します。`--draw` で枠を描いた `NAME.objects.png`、`--class person,car` で種類を絞れます。`go run ./cmd/hwnpu-image pose photo.jpg` は RTMO で画像の人と 17 の関節点を出します。マシンのメモリで RTMDet-s (8 GB) か RTMDet-m (16 GB 以上) を選びます (`-m rtmdet:tiny` で明示)。M1 の Neural Engine で 1 枚 15〜26 ms (ネットワーク 7〜18 ms、全部 NPU)。

`go run ./cmd/hwnpu-image ocr` は、画像の文字を読みます (PaddleOCR の PP-OCRv6 small、Apache-2.0。日本語、中国語、英語など 50 言語)。行ごとに、四隅の座標、文字、スコアを JSON で出します。`--text` で文字だけ、`--draw` で行の枠を描いた `NAME.text.png` も書きます。文字の検出と認識の 2 つのネットワークが NPU で動き、M1 でレシート 13 行が 40 ms、論文 1 ページ (98 行) が 0.5 秒です。PaddleOCR の手順を ONNX Runtime で動かしたものと、ページの 98 行のうち 93 行が一字一句同じでした (文字誤り率 0.2 %)。検出器の大きさは 960×960、1280×736、736×1280 から選びます (`--size` で変えられ、大きいほど小さな文字を拾います)。

```bash
go run ./cmd/hwnpu-image detect --draw photo.jpg                     # JSON lines, photo.objects.png
go run ./cmd/hwnpu-image ocr --text scan.png                         # the lines of text
go run ./cmd/hwnpu-image pose photo.jpg                              # people and their 17 keypoints
```

どちらも Apple の Neural Engine (と Intel の NPU、実機では未確認) で、初回にモデルを取ってきて変換します (OCR の認識器は行の幅ごとのプログラムを、その幅を初めて使うときに変換し、それぞれ初回のコンパイルに 5〜20 秒)。AMD の機械では、RTMDet-m と `pose` の RTMO-s は NPU で動きます (初回に IRON で書き出し、1 枚 0.33 秒と 0.22 秒。docs/amd-npu-models.md 21、22 節)。RTMDet の tiny と s、OCR は CPU で動きます (`hwnpu-audio pitch`、`notes` も同じ)。何も入れなくても hwnpu の CPU 実行器で動き (1 スレッドで RTMDet-m 1 枚 2.3 秒、3 行の OCR 4.5 秒)、ONNX Runtime の共有ライブラリがあればそちらを使います (`HWNPU_ONNXRUNTIME` にパス。1.4 秒と 1.8 秒)。`--threads` で増やせます (docs/amd-npu-models.md 19 節)。座標と物の種類が要るときは、汎用の視覚言語モデル (LFM3-VL、`hwnpu-chat --image`) より、こちらが速く (1 枚数十 ms と数秒)、全部の物を拾い、座標が正確です。

## ラベル付けと画像の埋め込み (SigLIP 2)

`go run ./cmd/hwnpu-image label` は、Google の SigLIP 2 (Apache-2.0) で写真にラベルを付けます。候補 (`-l "猫,犬,a red car"`、`--labels-file`、どの言語でも) を `this is a photo of {}.` に入れ、その文が写真を表す確率をラベルごとに出します (ゼロショット。合計は 1 になりません)。候補を省くと、組み込みの約 440 の物、場所、画像の種類から選びます。`hwnpu-image embed` は写真と文 (`-t`) を同じ空間の埋め込みにし、`--compare` でそのコサインと確率を出します (画像の検索や似た画像の検索に)。AMD の NPU と Apple の Neural Engine で動き、マシンのメモリで Base (8 GB、224 × 224) か Large (16 GB 以上、256 × 256) を選びます。Ryzen AI 9 365 で 1 枚 84 ms (Base)、文 8 つで 131 ms、Transformers (float32) との埋め込みのコサインは 0.9998 以上です。Apple では初回にチェックポイントを取ってきて Go で変換します (Base 1.5 GB、Large 3.6 GB。変換は数秒、最初の読み込みでコンパイルに 13〜33 秒)。M3 で Base は 1 枚 7 ms、文 8 つで 16 ms (既定の 444 ラベルの埋め込みは最初の 1 回だけ 2 秒)、Large は 35 ms と 53 ms で、Transformers との埋め込みのコサインは画像 0.9998 以上、文 0.99999 以上です。ラベルの埋め込みはユーザーのキャッシュに置き、同じ一覧は 1 回だけ計算します。

```bash
python tools/amd/iron_siglip2.py ~/models/siglip2-base-patch16-224 ~/models/siglip2-base-patch16-224.amd-npu2   # AMD だけ。hf:google/siglip2-base-patch16-224
go run ./cmd/hwnpu-image label photo.jpg                     # JSON lines: the likeliest labels
go run ./cmd/hwnpu-image embed --compare photo.jpg -t "a cat on a sofa"
```

AMD の手順と精度は [docs/amd-npu-models.md](docs/amd-npu-models.md) の 18 節、Apple の記録は [docs/design.md](docs/design.md) の 16.183 にあります。

## 点や箱で指定する切り抜き (EfficientSAM)

`go run ./cmd/hwnpu-image segment` は、Meta の EfficientSAM (Apache-2.0、Segment Anything のプロンプト) で、点 (`--point x,y`、外す点は `--negative`) か箱 (`--box x1,y1,x2,y2`) で指した物の mask を作ります。`NAME.mask.png` (物が白) と `NAME.cutout.png` (物の外が透明) を書き、スコア (予測した IoU)、面積、枠を JSON で出します。復号器の候補 3 つのうち一番スコアの高いものを、`--all` で 3 つとも書きます。画像の符号器 (1024 × 1024 の ViT) は AMD の NPU (初回に IRON で書き出す)、Apple の Neural Engine、Intel の NPU (初回に ONNX から変換。Intel は実機では未確認) で動き、それ以外の機械や `--device cpu` では CPU です (ONNX Runtime があればそれ、なければ hwnpu の評価器で遅い)。復号器はプロンプトごとにホストで動きます。マシンのメモリで Ti (8 GB) か S (16 GB 以上) を選びます。Ryzen AI 9 365 で、符号器が Ti 0.34 秒、S 0.48 秒、復号器が 0.6〜0.7 秒です。M3 では S の符号器が 0.4 秒、復号器が 0.3 秒で、mask は CPU のものと IoU 0.9994 です。

```bash
go run ./cmd/hwnpu-image segment --box 280,40,500,385 photo.jpg   # photo.mask.png, photo.cutout.png
go run ./cmd/hwnpu-image segment --point 390,200 --all photo.jpg
```

手順と精度は [docs/amd-npu-models.md](docs/amd-npu-models.md) の 23 節にあります。
