# 量子化

[English](quantization.md)

NPU ごとの重みの形式、形式と校正の方法の選び方、モデルの系統ごとの既定、落とし穴。形式は 1 トークンに読むバイト数を決め、それが速さと電力を決める。校正は選ぶ符号を変えるだけなので、実行時の費用は 0。

## 形式

### Apple の Neural Engine (Core ML と Core AI)

| 形式 | 並び | 注記 |
|---|---|---|
| `lut4`、`lut6`、`lut8` | 16 行の組ごとに 2^bits 個の fp16 の表 1 つ (`Options.Group`。16 が Neural Engine の速く読む形)。添字は詰める (uint6 は下位ビットから) | `mlpackage.Builder.LUT`、`palette.QuantizeBits`。主な形式 |
| `lut4r`、`lut6r`、`lut8r` | 各行を行の scale で割って lutN にし、出力に fp16 の行の scale を掛け戻す (`blocks.linearRows`) | 行の大きさが大きく違う行列に。gpt-oss の注意は行の大きさが最大 6600 倍違い、Neural Engine で KL 0.28 (lut8)、0.069 (lut8r)、0.092 (int8)、0.033 (fp16)。M3 で 1 ステップ lut8 と lut8r 57 ms、int8 73、fp16 79 |
| `int8`、`int4` | 対称、行ごとに scale 1 つ。`-quant mse` (既定) はクリップ点を探し、`uniform` は最大の重みで決める | Neural Engine で正確。ただし下の int8 のあふれ |
| `int4-g128`、`int8-g32` … | 列のブロックごとの scale | Core AI の Neural Engine では誤る (下)。変換が断る。Core ML は int8-gN を CPU に置く |
| `fp16` | | 参照。エンコーダ、VAE、小さな視覚のモデルは fp16 のまま |

混ぜ方: `-format-for SUBSTRING=FORMAT` (`Options.FormatFor`、最後に合うものが勝つ)。`-keep-fp16` がすべてに勝つ。

速さは重みのバイト数に従う。M2、Core AI、定数の 4096 × 4096 の linear (`aicheck limits -only formats`):

| | fp16 | int8 (行) | lut8 | lut6 | int4 (行) | lut4 |
|---|---|---|---|---|---|---|
| 1 行 | 742 µs | 466 | 449 | 387 | 329 | 351 |
| 64 行 | 808 | 506 | 515 | 477 | 379 | 382 |

呼び出しの下限 約 160 µs を引くと、fp16 は約 58 GB/s、int8 と lut8 は約 55 GB/s。lut4 は約 90 G 要素/秒で頭打ちになり、lut8 の 1.3〜1.4 倍にとどまる。M3 (Core ML、E2B の大きさの層 6 つ) では lut4 / lut6 / lut8 / fp16 が 1.81 / 2.65 / 3.49 / 6.52 ms。`hwnpu-dev probe` の速さはモデルの速さではない: M1 で probe は int8 を 62 GB/s で読むが、Gemma 4 E2B の部品は約 48 GB/s で、int8 は lut6 より遅かった。実際の部品を `HWNPU_PROFILE=1` で測る。

形式は Neural Engine のコンパイルのキャッシュ (約 3.5〜4 GiB) に入るプログラムの数も変える。LFM3 の lut6 の 4 本は入らず起動のたびにコンパイルし直したので、`-tier auto` は 6 ビットと 8 ビットの形式で段階を持たない。[Neural Engine](apple-neural-engine.ja.md) を参照。

### Intel の NPU

`int4` (行ごと)、`int4-g128`、`int8`、`fp16`。lutN はない (`checkIntelFormats` が断る)。`llm.IntelFormat` が Neural Engine の形式を読み替える: lut4 → int4、lut6 と lut8 → int8。アーキテクチャが自分で決めることもできる (`Arch.IntelFormat`、`IntelFormatter`)。多くのモデルの既定は int8: 次のトークンが fp16 と 52 中 51 一致、int4 は 22 (重み 1.6 GiB、int4 の 2 倍)。PLaMo 3 2B: int4 KL 0.72、int4-g128 0.32、int8 0.0025。[Intel](intel.md) を参照。

### AMD の XDNA (IRON)

| 形式 | 1 重みのビット | 並び | 使う所 |
|---|---|---|---|
| `q4nx` | 5 | FastFlowLM の形式: 32 個ごとの scale と min で一様な 16 段。当てはめは 9 つの候補から誤差の最小を選ぶ (重みの誤差 8.2 → 7.3 %) | `--q4` の MLP と出力層 |
| `q6` | 6.5 | q8 の並び (ブロックは出力 32 × 入力 256、入力 32 ごとに bf16 の scale)、符号 −31〜31 を 2 つの面に (下位のニブルは q4nx と同じ、上位 2 ビットは 128 個ごとに 32 バイト) | 小さなモデルの `--mlp q6 --head q6` |
| `q8` | 8.5 | int8 と 32 個ごとの scale | 注意と短い畳み込み |
| `q8x` | 8.5 | q8 の符号を mx4 のブロックの大きさ (出力 32 × 入力 160、bias つき) に。デコードとプロンプトが読む | gpt-oss |
| `mx4` | 4.3 | MXFP4 のまま: e2m1 のニブルをカーネルで戻し、指数のバイトを scale に | gpt-oss の専門家 |
| `bf16` | 16 | | router、参照 |
| `bfp16` | 約 9 | 8 値が指数を共有、8 ビットの仮数 | プロンプトの GEMM (q4nx はこれに戻す。gpt-oss のプロンプトの専門家) |

8192 × 2048 の GEMV: q4 0.30〜0.35 ms、q6 0.42〜0.47、q8 0.43〜0.50。q6 は q8 とほぼ同じ時間 (上位ビットを取り出す計算で律速)。得るのはメモリ (q8 の 76 %) と精度 (相対 RMS q4 0.071、q6 0.022、q8 0.005)。精度に効く数値の点:

- q8 と q6 の GEMV は、32 項の部分和を scale を掛ける前に 2 つの bf16 (上位と残り) に分ける: 1 つの bf16 ではコサイン 0.9998、2 つで 0.999992
- コアは切り捨ての丸めで始まる。hwnpu は最近接偶数の丸めにする (Anima の DiT の速度誤差 q8 0.088 → 0.031)。書き出し直すと効く

[AMD の LLM](amd-llm.md) と [AMD](amd.md) を参照。

### CPU: MXFP4 と int8 の専門家

Mac の専門家混合のモデルは専門家を CPU で回す ([MoE](moe.md) を参照):

- gpt-oss-20b: チェックポイントの MXFP4 (e2m1、32 個ごとに 2 の冪の scale) のまま。丸め直さない
- Qwen3.6-35B-A3B: bf16 の専門家を gpt-oss の形の MXFP4 に派生させる (`hwnpu-dev mxfp4`、`qwen35.DeriveMXFP4`、専門家 17.1 GB、全体 22.0 GB)。`mxfp4.Encode` は OCP の scale (最大値の指数 − 2) とその周りの 3 つを試して二乗誤差の小さいものを取り、値は最も近い e2m1 に (同じ距離は偶数)。正規分布の値で相対誤差 0.113。KL は測っていない
- LFM2-8B-A1B: 専門家を int8 の実行時の重みとして Neural Engine で (Core AI、M2 以降)

## モデルの系統ごとの既定

コード (`Arch.Format`、`Config.Format()`、`defaultFormats`) と [models.yaml](../../models.yaml) から。hwnpu-chat は lut の表を `kmeans-minmax` で当てはめる。変換器だけの既定は `uniform`。

| モデル | Apple (Neural Engine) | Intel | AMD (`amd_export`) |
|---|---|---|---|
| LFM3-1.7B | lut6 (grouped 注意と) | int8 | `--q4` (MLP と出力層 q4nx、注意と畳み込み q8) |
| LFM3-VL / Audio / Omni、d1 (テキスト部) | lut8 (`Config.Multimodal`) | int8 | `--q4` |
| Gemma 4 E2B / E4B | lut6 | int8 | `--q4` |
| Qwen3 | lut8 | int8 | — |
| Qwen3.5-0.8B、4B | lut8 | int8 | 0.8B `--q4 --mlp q6 --head q6`、4B `--q4` |
| Qwen3.5-2B | lut6 (hidden 2048) | int8 | `--q4 --mlp q6 --head q6` |
| Qwen2.5-Coder-7B | 行全体の表の lut4 と GPTQ の codes (`codes: faace07a`) | — | — |
| PLaMo 3 / Translate 0.8B、2B | lut8 | int8 | `--q4` |
| PLaMo 3 Translate 8B | lut4 (hidden ≥ 4096) | int4 | `--q4` |
| PLaMo-Embedding-1B、MiniLM (`plamo-bi`、`bert`) | lut8 (PLaMo は head-scale も) | int8 | — |
| Whisper | lut8 | int8 (読み替え) | [音](audio-models.md) を参照 |
| gpt-oss-20b | lut8。`self_attn=lut8r`、`lm_head=lut6r`。専門家は CPU の MXFP4 | — | 専門家 mx4、注意 q8x、出力層 q4、router bf16 |
| Qwen3.6-35B-A3B | token mixer と共有の専門家 lut8、`lm_head=lut6r`、経路のゲート fp16。専門家は CPU の MXFP4 | — | `--q4` (専門家と出力層 q4、mixer q8) |
| LFM2-8B-A1B | lut8、専門家 int8 | — | — |
| 画像のモデル | テキストのエンコーダと DiT は lut8 (Chroma の T5 は `wo=fp16`。FLUX.3 klein の DiT は活性に合わせた表の lut8)。Anima はテキストも DiT も int8 (DiT は fp16 と同じ速さで lut8 より fp16 に近い) | Anima fp16 | Anima の DiT q8 |

画像と音の詳細は [画像のモデル](image-models.md)、[Apple の画像](image-apple.md)、[AMD の画像](image-amd.md)、[音](audio-models.md)。

LLM の既定の理由 (断りがなければ float の参照に対する KL):

- LFM3 の lut6: KL 0.232、lut4 は 1.387。コーディングの課題が 6 往復、lut4 は 13〜27。grouped 注意で 1 回の呼び出しが以前の lut4 + tile より 38 % 速い
- マルチモーダルと d1 の lut8: lut4 では d1 が問い合わせを違う担当に回し、LFM3-VL が画像を誤って説明し、LFM3-Audio の TTS が聞き取れなくなった
- Gemma 4 の lut6: lut4 KL 2.54 (argmax 13/31)、lut6 0.25 (25/31)、lut8 0.048 (27/31) だが 3 部分で 41 ms (lut6 は 31.5)
- Qwen3.5-2B の lut6: lut8 の次のトークンと 283 中 260 一致、12 問の正解は同じ 10、1.32 GiB (lut8 は 1.78)。0.8B の lut6 は 242/283 で 17 × 23 を誤る。4B は落ちすぎる
- Qwen2.5-Coder-7B: 丸めの lut4 は KL 0.90、行全体の表の GPTQ の codes で 0.131 (lut8 は 0.0012 で 7.1 GB、lut4 は 3.5)
- PLaMo 3: 2B の lut4 は KL 1.91、lut6 は次のトークンの一致 89 %、lut8 は 98 % (M3)。8B の lut4 は 20 文のうち 13 を lut8 と一字一句同じに、残りを正しい言い換えに訳し、メモリは半分
- 埋め込み: lut4 のコサイン 0.20〜0.76、lut8 0.998。PLaMo-Embedding は head-scale も要る (下)

## 測り方

- **KL(参照 ‖ 量子化)**: 400 位置 (256 トークンをプロンプトとして読み、その後 1 つずつ) の平均、中央値、95 パーセンタイルと最上位の一致。**BOS をつける** (LFM3 は BOS なしだと 4 倍悪く見える)
- 5 つの文は `~/models/calib`: README、GPL-3 (校正に使わない。合わせすぎを見るため)、wiki40b-ja (評価分)、wikitext-2 (試験分)、Go のコード。変わり続けるものは `testdata/texts` に固定した写しがある
- 参照: CPU の fp32 (HF) か、同じ NPU の bf16。NPU の量子化の KL は CPU の見積もりと合うこと (LFM3 0.036 と 0.035)。合わなければまず NPU の演算を疑う
- Neural Engine: `hwnpu-dev generate -package P.parts.json -score REF,...` がキャッシュなしのプログラム (`hwnpu-dev convert -seq 128`) を参照のトークン (`ref_logits.py`、`ref_gemma4.py`) で流す。文の先頭のトークンは重い (KL 最大 8〜13) ので、設定の比較は同じ文どうしでだけ
- 量子化を疑う前に演算の誤差を除く: 同じプログラムを Neural Engine と Core ML の CPU で比べる。ANE の `silu` だけで LFM3 の lut8 の損の大半だった (`blocks.SiLU` で KL 0.058 → 0.022)
- `hwnpu-dev formats` (fp16 が部品に入らないときは `-ref lut8`): 次のトークンの一致と短い問い。短い文では量子化した方が perplexity が低いことがある。一致で比べる
- 画像の生成: 参照の軌跡の上の速度の相対 RMS 誤差 (`chroma.TestCallsOnDevice`) と、複数の種で自由に回した最後の潜在のコサイン。1 ステップの誤差は自由に回した結果を予測しない (FLUX.3: 速度誤差 2.33 → 2.29 % で、潜在のコサインは 0.964 → 0.983)
- 音声と画像のエンコーダ: WER / CER で (Whisper の lut6 は WER 4.43 → 4.87 % なので lut8)

## 形式を選ぶ

1. 部分ごとのビットを CPU で振る (注意の射影、MLP、出力層、畳み込み、塔。`q4_gptq.py --method rtn` に `--q8`、`--head`、`--fmt-*`。Apple の lut は `tools/quant/rotkl`)
2. 目安で決める:

| 測った値 | 判断 |
|---|---|
| KL < 0.05 (CPU、HF に対して) | 4 ビットを既定にできる |
| KL 0.05〜0.2 | 答えで決める (LFM3 の lut6 はここ) |
| KL ≥ 1 | 使えない |
| `hwnpu-dev formats` で 90 % 以上、正解の数も同じ | 小さい方を取る |
| 85〜90 %、または算数や事実の新しい誤り | 1 つ上の形式のまま |
| 80 % 未満、または答えが繰り返す | 使えない (Qwen3.5 の lut4: 60〜76 %) |

振って分かったこと:

- **注意の射影は 8 ビット**: 重みの 1 割ほどだが 4 ビットでは誤差の大半 (AMD の E2B の KL 0.097 → `--attn q4` で 0.229。bf16 は 0.091)。+0.12 GB
- AMD では MLP と出力層は 4 ビット。出力層の 4 ビットはほとんど効かない。短い畳み込みは 8 ビット
- サンドイッチ norm の層 (Gemma 4、PLaMo 3) は 4 ビットに弱い。Neural Engine では lut6 以上
- **同じ構造の大きなサイズは少ないビットで済むことがある** (PLaMo 3 8B の lut4)。チェックポイントごとに決める (`llm.Formatter`)。小さなサイズの判断を大きなものに持ち込まない
- **小さなモデル (2B 以下) は MLP が 4 ビットで崩れることがある**: Qwen3.5-0.8B は README 0.055 だが GPL 0.34 (GPTQ で 0.30)、AMD の q6 で 0.044。いつも GPL を見る
- **すでに 4 ビットのチェックポイントは自分の形式のまま**: gpt-oss の MXFP4 を q4nx に丸め直すと重みの誤差が 6.6 % 増え、KL は README 0.052 → 0.135、wiki-ja 0.043 → 0.107。MXFP4 は q6 にそのまま入り、mx4 ならそのまま
- 蒸留した 8 ステップの DiT は 4 ビットに弱い (Anima の q4nx は潜在のコサイン 0.777、q8 は 0.961)。大きな DiT は持ちこたえることがある (Qwen-Image-2.1 20B の lut4 + kmeans-minmax で 0.935〜0.981)。大きな DiT は MLP を 6 ビット、注意を 8 ビットに (Qwen-Image-2.1 の 7B の DiT の MLP q6: −1.21 GB、絵は同じ)
- プロンプトに 1 回だけ回すもの (テキストのエンコーダ、Chroma の approximator) は大きな形式にできる: ステップのあいだ持たない。ただし読み込み、ディスク、コンパイルのキャッシュの場所は使うので、要る重みだけ上げる (Chroma の T5: lut8 で `wo=fp16`、5.5 GB、fp16 は 9.5)
- 校正の対象に入っていない重みも基の形式で丸められる。小さなものは 1 段上に (Gemma 4 の `per_layer` の射影 41M を `-format-for per_layer=lut8`、+0.01 GiB)

## 校正

形式はそのまま、符号の選び方を良くする。この順で試す:

| 順 | 方法 | 道具 | 結果 |
|---|---|---|---|
| 1 | 丸め (RTN) | `iron_q4.quantize_fmt`、`palette.QuantizeBits` | 基準。重みの誤差の最小は出力の誤差の最小ではない |
| 1b | **ブロックの最小と最大を残す k-means** (データ不要) | `-quant kmeans-minmax` (`palette.KMeansMinMax`) | Qwen-Image-2.1 の DiT、4 ブロックを fp16 と比べて: 一様 4.9 % → 3.2 % (ただの k-means は 12.6 %: 外れ値を切る)。hwnpu-chat と音楽のモデルの既定 |
| 2 | **入力のチャネルごとの E[x²] で重みづけした表** (一様な段から Lloyd) | `q4_gptq.py --lut-fit wkmeans`、`palette.QuantizeWeightedMinMax`、`tools/coreai/lutfit` | LFM3 の lut6 0.050 → 0.038、lut4 0.66 → 0.33 (GPTQ と)。FLUX.3 の DiT の lut8 は潜在のコサイン 0.964 → 0.998 |
| 3 | **GPTQ** (各列の誤差を後の列に送る、H = E[xxᵀ]) | `tools/amd/q4_gptq.py`、`qwen2_gptq.py`、`gptoss_gptq.py`、`anima_gptq.py` … | AMD の wiki-ja: E2B 0.221 → 0.081、E4B 0.083 → 0.049、LFM3 0.071 → 0.038、PLaMo 3 0.058 → 0.040。Apple の LFM3 の lut6 0.074 → 0.050 |
| 4 | lut の前に回転 (下) | `llm.Rotate`、`tools/quant/rotkl` | Qwen3.5 でどの形式も KL が約半分 |
| 5 | AWQ 風の scale を前の演算に畳む | 書いていない | lut4 をさらに下げるとき |
| — | 形式を 1 つ上に (AMD の q6) | `--mlp q6`、`q4_gptq.py --fmt-q4 q6` | 校正で直らないとき |

効く GPTQ の設定:

- H を **対角の平均の 0.1 倍** で減衰させる。0.01 は校正の文に合わせすぎ、GPL が丸めより悪くなった
- 一般の文で校正する: wikitext-2 と wiki40b-ja を 64 × 512 トークン (`tools/amd/calib_data.py`)。repo の文書だけでは合わせすぎる。BOS の位置は H から外す
- 前の層を量子化してから次の層の H を集める (1 回に数層)
- Apple の lutN では表を先に当てはめて (一様か wkmeans) 固定し、GPTQ は各重みの添字を選ぶ。q4nx では GPTQ がその群に着いたときの重みに群の scale と min を合わせ直す
- 注意と畳み込みの射影も校正するには `--q8 gptq` が要る (LFM3: なしで 143 のうち 73、ありで 143。英文の KL 0.255 → 0.247)
- マルチモーダルのモデルのテキスト部は学習し直されている: それぞれ校正する (LFM3-VL / Audio / Omni)。基のモデルの codes を使い回さない
- 採るのは、校正に使わない GPL が丸めより悪くならないときだけ。LFM3-Audio は GPL が 0.154 → 0.161 だが、残りの 4 つの文が半分になったので採った
- DiT は別の種で自由に回して判断する (採らなかったもの)

費用: CPU の fp32 で、2B のモデルは 25〜60 分、E4B は 2.4 時間 (30 GB を超える)。Linux では `systemd-run --user` で回す。16 GB の Mac では、Gemma 4 E2B は埋め込みの表をファイルから読む (`q4_gptq.py` の `lean`、約 9 GB、M1 で 87 分)。lean のモデルはまず transformers のロジットと合わせる (KL ≤ 0.003、最上位が同じ)。`qwen2_gptq.py` は 7B を 1 層ずつ組む (M1 で 77 分)。Qwen3.6 (`qwen36_gptq.py`) は 4.3 時間。

codes:

- `palette.Codes` (`index.json`、重みごとの `.idx` と `.tab`。Go からは `palette.CodesWriter`) に書く。チェックポイントの重みの形と bf16 の値のハッシュで引く (`iron_q4.weight_key` = `palette.WeightKey`) ので、モデルごとの名前の対応は要らない。書き出しが重みを並べ替えたりつないだりすると当たらない。ログの「N weights took the chosen codes」で数を確かめる。1 つとして量子化する融合した重み (PLaMo 3 の `gate_up_proj`) は当たる
- AMD: `iron_export_lm.py --q4-codes DIR`。Apple: `hwnpu-dev convert -codes DIR`、`hwnpu-chat --codes DIR` (lutN だけ。プログラムの名前に codes のハッシュが入る。例 `gemma4-r1-lut6-gptq-27b3b448-per_layer.lut8-...`)
- models.yaml の `codes: HASH` があると codes で変換する。codes は書き出しと一緒に取得元に置き、取得元から取る (`chat.putCodes`、`fetchCodes`、`converted/MODEL/any/codes-HASH`)。書き出しの鍵が変わったとき、codes がないと誰も作り直せないため。[配布](distribution.md) を参照
- GPTQ が既定なのは Qwen2.5-Coder-7B だけ。ほかは使うときに指定する。`amd_export` は校正しない

## lut の前に回転

`llm.Rotate` (`llm.Rotation`、`Rotator`) は直交行列 Q = diag(s) H (ランダムな符号と、最大の 2 の冪の Walsh–Hadamard。割り切れなければ区分対角) を残差の流れに通す。変わるのは重みだけ: 埋め込み E Q、残差を読む行列 W diag(1 + w) Q (norm の重みは 0 に)、残差に書く行列 Qᵀ W、出力層 W diag(1 + w_final) Q。

- 残差を読むものが全部 norm の後の linear で、書くものが全部そのまま足されるモデルでだけ通る。Mac のチャットのモデルでは Qwen3.5 だけ (`qwen35.Config.Rotation`)。LFM3 (サンドイッチ norm)、Gemma 4、PLaMo 3 は書く側に重みつきの norm がある。gpt-oss は MXFP4 の専門家を回して量子化し直すことになる
- `rotkl` を CPU で (milcpu、float32、256 トークン、kmeans-minmax): Qwen3.5-2B の lut8 0.0047 → 0.0033、lut6 0.0560 → 0.0267、lut4 0.6026 → 0.3664。0.8B の lut6 0.0749 → 0.0388。fp16 に丸めても範囲は崩れない
- lut4 は回しても使えず、今の lut6 に届く lut4 の組み合わせもない (一番近い出力層 lut4 で 0.062)。回転で得るのは lut6 の精度で、速さではない
- Neural Engine で (`rotkl -ane -per-part 6`、M2、2B、128 トークン、CPU の fp16 と比べる): lut6 0.0537 → **0.0288**。lut8 は 0.0079 → 0.0091 (NE 自身の fp16 の誤差に隠れる)
- **Apple の Qwen3.5 の lut6 の書き出し (2B) は既定で回す**: `chat.Prepare` は回せるモデルを、Core AI と Core ML で lut6 以下のときだけ回す (書き出しの名前に `-rot`。`HWNPU_DEV=rotate=0` で回さない、`=1` で lut8 も)。AMD と Intel の書き出しは回さない。Qwen3.6 も回さない (専門家は CPU の MXFP4 で、量子化し直すことになる)。結果と MTP、表の扱いは [Qwen](llm-qwen.ja.md)

## 失敗の形

- **行ごとの int8 は Neural Engine で大きな出力があふれる**: 整数の積を足してから最後に行の scale を掛けるらしく、途中が fp16 を超える。Qwen3-VL のテキストのエンコーダは attention sink で NaN (層 6 の `down_proj` の行 0 で 13120)、Chroma の T5 の `wo` と DiT は無限大、SANA 1.5 の int8 の DiT は崩れた。lut8 (表を fp16 に戻してから足す) は問題ない。大きなモデルは lut8 に
- **linear と conv の出力は 2^15 未満に**、途中の和も (M2: 最後の和が 0 でも途中が 32768 で Inf)。[Core AI](apple-coreai.md) を参照
- **列のブロックごとの表と scale を Neural Engine が黙って誤る** (Core AI、macOS 27.2、M1): 行の組がどの列にも最初の列のブロックの表か scale を使う。全部載り、速さも同じで、値が違う (int4-g32 のコサイン 0.966〜0.984。32 列ごとの表の GPTQ の 7B は答えが壊れた)。Core AI の CPU は正しい。`blocks.Graph.rowsOnly` が Core AI で断る。行全体の表だけ。行列を分けても (`MatrixSlice`) 表は行列全体のものなので、codes もそのまま使える
- **隣り合う lut の表は違わなければならない**: `coreai-build` (オフラインの ANE のコンパイラ) は、256 行の区切りの先頭で隣の 16 行の表が同じだと 32 行で共有する表とみなして落ちる (`EmitGlobalBackedCoalescedLutTensor: ... not content-uniform`)。os_log にしか出ない。一様な表はよく重なる。`mlpackage.DistinctNeighbours` (`Builder.LUT` から、すべての lut の書き出し) が重なった表の絶対値の最大の値を fp16 で 1 単位 0 に寄せる。`tools/coreai/lutscan -tile 16` で探し、`tools/coreai/lutfake` で再現する。Apple に報告済み (FB25624151)。端末のコンパイラには関係ない
- **分位点から始めた k-means は大きな重みを潰す**: 一番上の段が 99.8 % 点に来る (FLUX.3 の DiT の lut8 で潜在のコサイン 0.56)。一様な (最小と最大の) 段から始める。wkmeans も、重みの誤差の最小の k-means から始めると KL が一番悪い
- 最終層の出力が fp16 を超える (PLaMo-Embedding の `down_proj` が最大 5e4 で −Inf、コサイン 0.9745): head-scale が最後の残差と `down_proj` の入力を 1/64 にする (2 の冪で正確。最終の RMSNorm は倍率に依らない) → 0.99986。重みの方を縮めると Intel の int8 の scale が非正規化数になった
- AMD: 切り捨ての丸め (上)。bfp16 は GEMM には良い (誤差が平均される) が、softmax が続く所には使わない
- ANE の `silu` / `softplus` の誤差は量子化の損に見える。[Neural Engine](apple-neural-engine.ja.md) と [Core AI](apple-coreai.ja.md) を参照

## 採らなかったもの

- Neural Engine で Q4_0 (Gemma の QAT の格子): grouped conv は 12.3 ms で lut4 の 1.81 の 7 倍、バッチの matmul は CPU に落ちる (42 ms)
- Apple の LLM と大きな DiT の行ごとの int8: lut8 より速くなく、大きな出力であふれる (上)。Anima は int8 のまま
- 丸めの lut4 を LFM3、Gemma 4、PLaMo 3 の 2B 以下、Qwen3.5、マルチモーダルの部分、埋め込み、Qwen2.5-Coder に: KL 0.9〜2.7 か答えが崩れる
- 回転した lut4 (Qwen3.5-2B で 0.37)。回した lut6 に出力層だけ lut4: +3.4 % 速い (33.5 対 32.4 tok/s) が KL 0.062 で今の 0.056 より悪い
- Qwen3.5 の lut8 の書き出し (0.8B、4B) を回す: 答えは同じで 0〜3.5 % しか速くならず、書き出し直しに 10〜27 分かかる
- Gemma 4 の lut6 で MLP だけ lut8: 1.99 GiB で 2 部分に入らない。注意だけでは誤差の大半が残る
- Core AI の Neural Engine で列のブロックごとの表と scale: 値が誤る (上)。(Qwen2.5-Coder の lut4 で 32 列ごとの表の GPTQ なら KL 0.060、行全体は 0.131)
- Apple の LFM3 と Gemma 4 で GPTQ を既定に: 同じ速さで KL は下がる (LFM3 英 −23 %、日 −35 %。E2B は `per_layer` lut8 と合わせて中央値 0.25 → 0.08) が、12 問の差は多くて 1 問 (LFM3 では GPTQ が富士山を誤った)。codes 1.6〜2.1 GB の配り方も決まっていない。`--codes` で使える
- DiT の GPTQ (Qwen-Image-2.1 の MLP q4): 校正にないプロンプトで教師強制の速度誤差は −13〜15 % だが、自由に回すと 3 つの種のうち 2 つで丸めより悪い。Anima の DiT は q4 + GPTQ ではなく q8 (潜在のコサイン 0.939、q8 は 0.962)
- gpt-oss の AMD の注意の q8 を GPTQ で: wiki-ja 0.027 → 0.066 (3 位置で KL 1.6〜2.9)。校正するのは出力層の q4 だけで、codes を渡したときだけ
- gpt-oss の AMD の注意を q6 に: KL 2〜3 倍 (README 0.133)
- MXFP4 の専門家を q4nx に丸め直す (上)
- gpt-oss のデコードの注意のスコアを bfp16 に: 出力が最大値の 3.6 % ずれる (コサイン 0.99987)。bf16 のまま
- norm の重みを q8 の表の列に畳む (gpt-oss の AMD): 範囲が広がり、32 トークンで最上位が変わった
- lut4 の DiT にただの k-means (外れ値を切る、12.6 %)。分位点から始めた k-means (上)
- H の減衰 0.01。repo の文書だけで校正 (GPL 0.105 → 0.129、丸めより悪い)
- Apple の Anima の DiT を fp16 に: 潜在 +0.006〜0.025 に 8 % 遅く 1.8 GB 増える。int8 のまま
- Whisper の lut6: WER 4.43 → 4.87 %
- Chroma の DiT の MLP を lut4 に (潜在のコサイン 0.87〜0.93、lut8 は 0.9965)。lut6 は絵を保ったが 1024² で速くならない

## 未解決

- AWQ 風の scale (書いていない)
- 回した書き出しを `hwnpu-chat --package` で直接渡すと、表は回していない `tables.safetensors` を選ぶ
- Neural Engine の GPTQ をほかのモデルに: E4B (lean でも 16 GB の Mac に大きすぎる)、`per_layer` を校正に入れる、lut4 + GPTQ のチャット、問いを増やした比較と hwnpu-agent の課題、M2 と M3
- Qwen3.6 の派生の MXFP4 の専門家の精度 (bf16 との KL、MXFP4 の GPTQ)。AMD の GPTQ の codes は書き出しに入っていない
- Qwen2.5-Coder 14B (変換も校正もしていない)。Qwen2.5-Coder の AMD の形
- gpt-oss の AMD の注意を GPTQ と合わせて q6 に (注意は層の転送の 36 %)
- ANE のコンパイルのキャッシュの容量 (3.2〜4.8 GiB の間に見えた) は aned の側で確かめていない
