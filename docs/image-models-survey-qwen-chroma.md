# Qwen-Image-2.1 と Chroma1 を AMD の NPU へ入れるための下調べ

2026-11-11。移植はまだしていない。[new-model-checklist.md](new-model-checklist.md) の 1 節 (紙の上の見積もり) と、最初から入れる最適化 (4.1、5、6、7.1、11) を、二つのモデルの構造に当てはめた。時間は Anima の実測 (DiT 17 TFLOP が 1 パス 4.55 s、約 3.75 TFLOPS) からの見積もり。Chroma の DiT は FLUX.1 schnell の試作 (worktree `t3-flux`、`tools/amd/iron_flux.py`) で測った 1 パス約 20 s と合っている。

## 1. 二つのモデル

| | Qwen-Image-2.1 (+ Turbo) | Chroma1-HD (+ Flash) |
|---|---|---|
| リポジトリ | `Qwen/Qwen-Image-2.1`、`Qwen/Qwen-Image-2.1-Turbo` | `lodestones/Chroma1-HD`、`lodestones/Chroma1-Flash` |
| ライセンス | **Qwen Research License (非商用のみ)**、2026-09-20 | Apache-2.0 |
| DiT | 7.1B。シングルストリーム 32 ブロック、幅 4096 (32 頭 × 128)、SwiGLU 12288、バイアスなし | 8.9B。FLUX.1 schnell の形: ダブル 19 + シングル 38、幅 3072 (24 頭 × 128)、MLP 12288 |
| 変調 (adaLN) | 全ブロックで共有する線形 1 つ (4096 → 4 × 4096)。scale と gate だけ (shift なし)、gate は tanh | ブロックごとの変調の線形 (FLUX の 3.3B) を、時刻だけから 344 本の変調ベクトルを作る FFN (approximator、5 層 × 5120、約 0.25B) に置き換え |
| テキスト | **Qwen3-VL-8B-Instruct そのもの** (重みの一部を比べて同じ)。最終の RMSNorm の前の最終層の隠れ状態。システムプロンプトの分は捨てる | T5-XXL のエンコーダ (4.76B)、最大 512 トークン、パディングにマスク (パディングを 1 つだけ残す) |
| 画像トークン | VAE が 1/16、パッチなし: 1024² で 64 × 64 = 4096 | VAE が 1/8 + 2 × 2 パッチ: 1024² で 4096 + テキスト 512 |
| 注意 | ブロック因果: テキストは因果、画像ブロックの中は双方向。生成する画像はテキスト全部と自分を見る | 結合の双方向 (テキストのパディングはキーから外す) |
| VAE | 新しい `AutoencoderKLQwenImage21` (Wan 2.2 系、RMS 正規化、z 64 チャンネル、RGBA の 4 チャンネル)。デコーダの幅 144 × [8, 8, 8, 4, 2, 1] | FLUX の AutoencoderKL (16 チャンネル、GroupNorm) |
| 既定のサンプリング | Base: 40 ステップ、CFG なし (`true_cfg_scale` 1、負のプロンプトを渡したときだけ)。Turbo: 8 ステップ、シグマの表がチェックポイントにある、CFG なし | HD: 40 ステップ、CFG 3.0、README の負のプロンプト。Flash: CFG を焼き込んだ版 (README に推奨の設定がない。要確認) |
| 公式の解像度 | 2048² 前後 (4 MP、7 つの縦横比) | 1024² 前後 |

公式の高速化が Qwen-Image-2.1 にもう入っている: `causal_condition` でテキストのトークンは t = 0 で変調され、ステップに依らない。最初のステップでテキストの K/V を層ごとに取り (`kv_cache_mode="extract"`)、以降のステップは生成する画像の 4096 行だけを計算する。これは checklist 7.1 の「ステップごとに同じ仕事をループの外へ」(Anima では未着手) を、モデルの側が保証している形。

## 2. ダウンロードと共有 (checklist 11.1)

| | 新たに落とすもの | 共有できるもの |
|---|---|---|
| Qwen-Image-2.1-Turbo | DiT 14.2 GB (bf16)、テキスト 17.5 GB、VAE 0.68 GB (bf16)、processor 約 16 MB: **約 32.4 GB** | テキストは Qwen3-VL-8B-Instruct と同じ重み (層 10 の down_proj、層 35 の o_proj、埋め込み、視覚の層 0 の先頭 4 MB の sha256 が三つのリポジトリで一致)。Base とも共有 |
| Qwen-Image-2.1 (Base) | DiT 14.2 GB だけ (Turbo の DiT とは別の重み) | テキスト、VAE (Base の VAE は fp32 1.35 GB。Turbo の bf16 と値が同じかは未確認) |
| Chroma1-HD | DiT 17.8 GB (`transformer/` の 2 つのシャード) だけ。`Chroma1-HD.safetensors` (同じ 17.8 GB の 1 ファイル版) は落とさない | **T5 と VAE は FLUX.1 schnell のものとバイト単位で同じ** (sha256 を手元の `~/models/FLUX.1-schnell` と照合済み)。Flash とも同じ |
| Chroma1-Flash | DiT 17.8 GB | 同上 |

テキストの視覚の塔 (約 0.6B) と lm_head は文字からの画像生成では読まない。読み込みは遅らせる (lazy towers、16.81) か、エクスポートで読まない。

## 3. 紙の上の見積もり (AMD、1 パス)

| | 512² | 768² | 1024² |
|---|---|---|---|
| Qwen-Image-2.1 の DiT | 15 TFLOP、約 4 s | 35 TFLOP、約 9 s | 67 TFLOP、約 18 s |
| Chroma の DiT | 22 TFLOP、約 6 s | 42 TFLOP、約 11 s | 74 TFLOP、約 20 s |

一度だけの仕事:

- Qwen3-VL-8B のテキスト (256 トークン): 3.6 TFLOP、約 1 s。DiT のテキストの K/V (256 行 × 32 層): 3.6 TFLOP、約 1 s
- T5-XXL (512 トークン): 4.9 TFLOP、約 1.3 s
- VAE のデコーダ 1024²: Qwen-Image-2.1 は約 17 TFLOP (Anima の VAE の約 3 倍。Anima は 5.2 s なので **約 15 s**)。FLUX の VAE は未測定 (GroupNorm が IRON にない、t3-flux のメモ)
- Chroma の approximator: 344 行 × 0.25B で 1 ステップ 0.18 TFLOP。時刻だけで決まる

1 枚の時間 (1024²、ImageMHA と VAE の最適化を入れる前提の目安):

| 設定 | DiT のパス数 | 見積もり |
|---|---|---|
| Qwen-Image-2.1-Turbo、8 ステップ | 8 | 8 × 18 + VAE 15 + テキスト 2 ≈ **2.7 分** (Anima-Turbo の 44 s の約 3.6 倍) |
| 同、768² | 8 | 約 1.5 分 |
| Qwen-Image-2.1 Base、DPM-Solver++ 2M 25 ステップ、CFG なし | 25 | 約 8 分 |
| Chroma1-HD、公式 (40 ステップ、CFG) | 80 | 約 27 分 |
| Chroma1-HD、DPM-Solver++ 2M 25 ステップ + CFG を σ ≥ 0.6 だけ | 約 42 (shift 3 なのでステップの 2/3 が σ ≥ 0.6) | 約 14 分 |
| Chroma1-Flash (CFG なし、ステップ数は要確認) | ステップ数 | 16 ステップなら約 5.5 分 |

ImageMHA は FLUX の試作で注意が 1 ブロックの 191 / 342 ms だったところを 2.5 倍ほど速くするので、両方の DiT は上の 0.7 倍ほどになりうる (要測定)。

**向いている順**: Qwen-Image-2.1-Turbo (パスあたりの仕事が少なく、テキストの分を毎ステップ計算しない、8 ステップで CFG なし) > Chroma1-Flash > Qwen-Image-2.1 Base > Chroma1-HD。

## 4. メモリ (checklist 6、10)

| 部品 | 重み | 固定するメモリ (AMD) の考え方 |
|---|---|---|
| Qwen-Image-2.1 の DiT | q8 で 7.1 GB、q6 で 5.8 GB、q4 (q4nx) で約 4.4 GB | 32 ブロックを 4 ブロックずつ 8 グループにし、一つのバンクに順に読む (`SwapVariants`、16.106): バンク約 0.9 GB (q8) + 活性 (MLP の中間 4096 × 12288 bf16 = 100 MB) + テキストの K/V (32 層 × 256 行 × 8192 × bf16 = 134 MB、CFG ならその 2 倍)。ホストは 7.1 GB のグループの像を持つ |
| Qwen3-VL-8B のテキスト | 36 層で 6.95B (埋め込みの 0.6B はホストの表) | 一度しか走らないので **これも SwapVariants で層のグループを順に読む**。読み直しの損がない (どのみち一度読む)。q8 で丸ごと固定すると 7 GB |
| Chroma の DiT | q8 で 8.9 GB | 同じくグループに分けて順に読む |
| T5-XXL | q8 で 4.8 GB | q4 はトークンを壊す (t3-flux: cos −0.16)、q8 は良い (0.992 以上)。q6 は未確認。SwapVariants |
| VAE | Qwen 0.34 GB、FLUX 0.08 GB | 活性が主: Qwen のデコーダの最後の段は 1024² × 144 チャンネル (bf16 で 300 MB) |

段は順に走る (テキスト → DiT → VAE) ので、段ごとに下ろす (stage-unload、npu-residency-policy)。8 GB の機械は無理で、16 GB 以上 (`machine: 16` か 24) になる見込み。`memory` と `machine` は実測で書く (11.2)。

## 5. 最初から入れる最適化

checklist で効いたものを、それぞれのモデルのどこに当てるか。

### 5.1 両方に共通

| 方法 (checklist の節) | 当てる所 |
|---|---|
| 画像の注意を 64 キーのブロック、QK^T を bfp16 (`ImageMHA`、7.1、16.147) | Qwen: 4096 行のクエリ × (テキスト + 4096) キー。Chroma: 4608 × 4608 |
| プロンプトの MHA を 8 パイプライン (4.1、16.107) | `iron_mha.attention` は行が 512 の倍数でないと 4 パイプラインに落ちる (t3-flux)。Chroma の 4608 = 9 × 512、Qwen はテキストを 256 行単位に詰め、4096 + 256k にする |
| CFG の二つのパスをグループごとに続けて (7.1、16.148) | SwapVariants で読む回数を 1 ステップ 8 グループのままにする |
| DPM-Solver++ 2M (`--sampler dpm2m`、16.154) | Base と HD。Turbo はチェックポイントのシグマの表に従う (他の表は公式に評価されていない) |
| CFG を雑音の多いステップだけ (`--cfg-sigmas 0.6,1`) | Chroma1-HD (CFG が既定)。Qwen Base は既定で CFG なし |
| 蒸留した版を既定に (7.1) | `qwen-image` は Turbo、`chroma` は Flash を既定の候補に (Flash の品質と設定を見てから決める) |
| 固定の負のプロンプトを定数に (16.148 の `empty_cond.bin`) | Chroma: README の負のプロンプト (または空) の T5 出力をエクスポートで計算して持つ。Qwen は CFG を使うときだけ |
| VAE の 3 × 3 畳み込みを格子の上の GEMM (`GridGEMM`)、最近傍 2 倍 + 畳み込みを位相の 4 つの 2 × 2 (4.1、16.139) | Qwen の VAE は `nearest-exact` の 2 倍 + 3 × 3 の 4 段 (+ DupUp の近道) なので、そのまま当てはまる。同じ形の段を一つのグラフに |
| コアの丸めを最近接偶数に (16.78) | 平均のある和を引き下げない。DiT とテキストの両方 |
| コマンドを 2 s 未満、shim のチャネルに 3 つまで (4.1) | 1024² の 1 ブロック (Qwen 218M 重み) は 1 コマンドに収まる見込み。テキストは `SCRATCHPAD_WORDS` で切る |
| 重みの形式は DiT q8 を既定、q6 と GPTQ q4 を試す (5.2、5.3) | 蒸留した 8 ステップの DiT は 4 ビットに弱い (Anima q4nx cos 0.777)。速度ベクトルの相対 RMS 誤差 (`anima_gptq.py`) で比べる。テキストは q8 / q6 を行の誤差で (Sana は q6) |
| 演算子ごとに float32 と実の活性で比べる (4.1、SigLIP 2) | tanh の GELU (Qwen の `txt_in`) は表の誤差が bf16 の 5 倍。厳密な GELU と HIGHEST を候補に |

### 5.2 Qwen-Image-2.1 だけ

| 方法 | 内容 |
|---|---|
| **テキストの K/V を一度だけ** (公式の prefix KV cache) | プロンプトごとに一度、DiT のテキストの行 (t = 0 の変調、因果の注意) を流し、32 層の K/V をバンクに残す。毎ステップのグラフは 4096 行の画像だけで、キーは [テキストの K/V (パディングはマスク)、画像] |
| 変調を全ステップ分まとめて | 共有の変調の線形 (67M) は時刻だけで決まる。全ステップの時刻と t = 0 の行を最初に一度に計算し、ステップごとのグラフの入力にする。ブロックごとに変調の重みを読まない |
| テキストの決まった前置きを定数に (4.1 の `iron_sana.Prefix`、16.138) | `<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n` は毎回同じ。Qwen3-VL-8B の因果の注意なので、その K/V をエクスポートで float32 で計算して持ち、残りを固定のチャンクと `start` で計算する。DiT 側でも、捨てない前置き (`<|im_start|>user\n`) の K/V は定数にできる |
| テキストの塔は使うときだけ | 文字からの生成では視覚の塔を読まない (16.81)。編集 (参照画像) は後回し |
| テキストは Qwen3 の層 | `iron_anima.py` の `Qwen3Layer` (Qwen3-0.6B) が同じ形 (Q/K の RMSNorm、GQA)。文字だけなら mrope の 3 軸は同じ位置になり、ふつうの RoPE と同じ。8B では `iron_export_lm.py` の量子化の GEMM を使う |
| RoPE の 3 軸 | 複素数の隣り合う組 (`use_real=False`)。t3-flux の `head_perm` (q / k の行を頭ごとに並べ替えて組を作る) がそのまま使える |

### 5.3 Chroma だけ

| 方法 | 内容 |
|---|---|
| 変調を全ステップ分まとめて | approximator の入力は時刻と固定の表だけ (guidance は 0 固定)。エクスポートの解像度とサンプラーとステップ数が決まれば、全ステップの 344 本は**定数**。サンプラーの組ごとにエクスポートで計算して持つか、画像の最初に 344 × ステップ数の行を一度に計算する。FLUX の 3.3B の変調の重みはもうない |
| T5 と VAE は FLUX.1 schnell と共通 | T5 のグラフ (相対位置のバイアスを明示の加算で、低ランクにはならない) と GroupNorm の VAE は、FLUX にも使える作業として一度だけ作る |
| テキストの長さ | 公式は最大 512 でパディングをマスク。DiT の行の 11 % がテキスト。256 行のエクスポートを既定にし、512 は形を分ける (形が違えばプログラムも別、11.3) かを測って決める。マスクはキーだけに (パディングのクエリの行は捨てるだけ。全部マスクした行の NaN を出さない) |
| ダブルブロックの二つの流れ | テキストの流れも毎ステップの時刻で変調されるので、Qwen のようには K/V を残せない |

## 6. 正しさの確かめ (checklist 2、9)

- 参照: ref-venv の diffusers 0.42.1 に `QwenImage21Pipeline` と `ChromaPipeline` がある。`tools/ref/ref_anima.py` の形で、CPU の float32 で潜在とデコードを `~/models/*-test/ref` に残す
- **Qwen のテキストの隠れ状態は最終の RMSNorm の前**。transformers 5.x (ref-venv は 5.22.1) は `hidden_states[-1]` を正規化した後のものにする。パイプラインはフックで戻しているが、自前の参照スクリプトでも同じにしないと、信号の 3 分の 1 が違うものと比べることになる
- Qwen のテキストは BOS がない。空のプロンプトは " " に置き換える (パイプラインの通り)
- IRON の MHA は点数を bf16 で持つ: Sana 600M のクロス注意で点数が約 1750 になり CFG がずれた (キーの行 0 を引いて直した、16.119)。Qwen の [テキスト + 画像] のキーでも点数の大きさを先に見る
- 1024² は Qwen-Image-2.1 の公式の解像度 (2048² 前後) の 4 分の 1。Base の動的な shift (mu) はトークン数で変わる。1024² と 768² の絵を CPU の float32 で公式の 2048² と並べ、使える解像度を先に決める
- 3 プロンプト × 2 種の絵を参照と並べる (9)。Turbo は 8 ステップの蒸留なので、潜在の cos より絵の並びで判断する (Anima-Turbo 0.876–0.948)

## 7. Apple と Intel

DiT の fp16 は 14–18 GB で、Neural Engine のコンパイルキャッシュ (実効 3.5 GiB、3.1) と 16 GB の Mac には大きすぎる。lut4 でも DiT だけで 3.6–4.5 GB。まず AMD だけで入れ、Mac は lut4 の分割で入るかを別に見積もる。

2026-11-12: Qwen-Image-2.1-Turbo を Mac の Neural Engine に入れた (DiT の画像のブロック lut4、文の符号器と DiT の文のパス lut8。512² で 1 ステップ 2.2 秒、1024² で 16 秒。時間の大半はキャッシュを超えるプログラムの読み込み)。[design.md](design.md) 16.165。

## 8. 決めること

1. **Qwen-Image-2.1 のライセンス**: Qwen Research License で非商用だけ。model-table.md 13 節は「研究用のライセンスのモデルは足さない」としている (ただし Anima も CircleStone の非商用ライセンスで入っている)。入れるなら `license` に書き、配布 (prebuilt.md) は外す。Apache-2.0 の Qwen-Image (20B) と Qwen-Image-2512 (20B) は DiT が約 3 倍で、1024² の 1 パスが 1 分近くになる
2. 解像度: Qwen は 768² と 1024² のどちらを既定にするか (6 節の比較の後)
3. Chroma は HD と Flash のどちらから入れるか (Flash の推奨のステップ数とサンプラーを調べてから)

## 9. 進める順 (案)

1. Qwen-Image-2.1-Turbo の参照 (CPU float32、512² / 768² / 1024²) と、テキストの K/V の一度だけの計算が公式と同じになることの確認
2. DiT のブロックを IRON に (t3-flux のブロックの形 + ImageMHA + 共有の変調)。1 ブロックを float32 と比べる
3. テキスト (Qwen3-VL-8B を q8 / q6、前置きを定数、SwapVariants) と VAE (GridGEMM、位相の上げ)
4. Go の側 (`internal/imagegen/qwenimage`)、models.yaml、実測の `memory` と `bytes`
5. Chroma: T5 と GroupNorm の VAE を FLUX と共通で、DiT は t3-flux から変調の部分を approximator の定数に置き換える
