# AMD の NPU の画像のモデル

[English](image-amd.md)

画像のモデルの DiT、VAE、文の符号器を、IRON で AMD XDNA 2 に載せる形。モデル、解き方、ガイダンス、大きさ、書き出しの決まりは [image-models.ja.md](image-models.ja.md)。IRON の基盤、ドライバ、言語モデルと共有のカーネルは [amd.ja.md](amd.ja.md)。数字は断りがなければ Ryzen AI 9 365。

## グラフの書き出し

- 画像のモデルは、入力から出力への IRON のグラフとして書き出す (`tools/amd/iron_graph.py`、Go の `iron.Graph`、`internal/llm/iron/graph.go`)。言語モデルの decode と prompt のプログラムとは別。書き出しの道具: `tools/amd/iron_anima.py`、`iron_flux2klein.py`、`iron_flux3klein.py`。共通の部分は `iron_dit.py` (`Checkpoint`、`qparam`、`head_perm`、`rope_angles`、`share_banks`、`VaeStage`、`timestep_sinusoid`、`text_linear`) と、FLUX.1 の VAE の `iron_chroma_parts.py` (`FluxVaeStage`。残りの T5 は外した Chroma1 のもの: [not-adopted/chroma1.md](../not-adopted/chroma1.md))。書き出しのディレクトリは `NAME.amd-npu2` で、`model.json` とマニフェストを持つ。
- **重みのバンク** (`variants`): 1 つの ELF がバンクを何組も持つ。DiT のブロックは数ブロックのグラフ 1 つと、組ごとのバンク (Anima: 28 ブロック = 4 ブロックのグラフ × 7 組)。
- **`SwapVariants`** (既定): バンクを 1 組だけ確保し、組を回す前にその組の像をバンクに読む (arena は重みを先に、作業領域を後ろに置き、1 回の呼び出しの後に作業領域に残すものはない)。Anima 1024²: 固定の最大 3445 → 1236 MiB、時間 +2 %、バイト単位で同じ。`--resident-dit` で全部の組を常駐。文の符号器 (Qwen3、T5、Qwen3-VL) も同じく組にする。
- **先読み** (`iron.GraphOptions.Prefetch`): 2 つ目のバンクを持ち、`Use(i)` の後に組 i+1 を裏で読み、次の `Use` はバンクを入れ替えるだけ。DiT の段に 2 つ目のバンクと 384 MiB の余裕 (`anima.prefetchMargin`: プログラムと制御コードも固定する) を足した量が、VAE のピーク以下か `anima.PrefetchFloor` (2 GiB) 以下のときだけ。`BONSAIAI_DEV=iron.prefetch=0` で切る。Anima は 1 ステップ 0.5 秒の読み直しのうち 0.1 秒しか隠れない: 読み込みと NPU が同じメモリ帯域を使う。
- 先読みしない読み込みは 8 スレッド (`iron.readPrefix`): 5〜7 → 15 GB/s。
- **`share_banks`**: 前のグラフと同じ重みが同じ順に並ぶグラフは自分のバンクの像を消し、`../GRAPH/` を指す (バイト単位で照らす)。そのため重みの配置は大きさにも行数にも依ってはいけない: **RoPE の表は入力** (`angles`、行 × head_dim、Go の `ropeAngles` がホストで作る、`TestRopeAngles`) にし、バンクに焼かない。`angles` のない古い書き出しも動く。
- **`iron.GraphOptions.Barrier`**: 同じグラフの 2 回目の最初のコマンドが `ERT_CMD_STATE_TIMEOUT` になることがある (呼び出しごとの位置に K/V を書くグラフ、6 コマンドの Chroma1 の approximator。外した)。同じ ELF を別のハードウェアコンテキストに読み、空のコマンドを先に回すと止まらない (irongraph `-barrier`)。
- グラフは行数ごとにコンパイルする。1 ステップの行は **512** の倍数 (注意の 8 本のパイプライン × 64 行) でなければならず、そうでない大きさは詰める (下)。

## 注意

`tools/amd/iron_mha.py` の `ImageMHA` (どの DiT の自己注意と交差注意も):

- **キーを 64 ずつの塊に**: mha.cc の速い softmax (`partial_softmax_rows`) は B_kv = 64 のときだけ通る。d = 128 では IRON が 32 × 32 を選び、softmax のコアが律速だった。因果でも窓でもなければ、キーの塊は問い合わせの塊と同じでなくてよい: B_q 32、B_kv 64、P·V を 2 コア (`pv_cores=2`)。4096² × 16 × 128 で 146 → 103 ms。
- **QK^T を bfp16 に** (`qk`): AIE2P のネイティブ bf16 の mmul は 1 サイクル約 24 積和。bfp16 (8 要素で指数を共有、8 ビットの仮数) が行列演算器のネイティブの形。103 → 59 ms (コサイン 0.999936)。**P·V も bfp16** (`qkpv`) で 54 ms。
- **ソフトマックスの指数を bf16 の limb 1 本で** (`kernels/exp2_mid.h`、`mha_mid.cc`、`exp=mid`): IRON の正確な exp2 は 3 次式を float32 の精度で計算する (積ごとに bf16 の limb 2 本: 32 個に mac 10 回と limb の分割 6 回)。小数部と Horner の各段を limb 1 本 (mac 4 回) にしても誤差は 0.2 % 以内で、結果の bf16 の丸めと同じ大きさ。FLUX.2 の注意だけで 102.7 → 83.2 ms、コサイン 0.999650 → 0.999649。IRON の `linalg/mha.cc` をこのヘッダの後に含めて `exp2_bf16` を差し替える。
- qk / qkpv などは宣言したカーネルのパラメータ (`param(array=True)`) にする: IRON のコンパイルのキャッシュ (`~/.npu/cache`) は演算子のパラメータで引くので、外からメソッドを差し替えただけでは前の ELF が黙って使われる。
- **呼び出しごとのキーの数 `nk`**: `nk` より後ろのキー (文のパディング、詰めた画像の行) を切る。64 の倍数でなくてよい (4608 行を `nk = 4568` で: コサイン 0.99995)。
- **0 のキーを softmax の和に数える** (`ImageMHA(zero_keys=N)`、`kernels/mha_zero_keys.cc`、Anima の `--dit-cross trim`): Anima の条件づけはプロンプトより後ろの行が 0 で、交差注意の k と v にバイアスがないので、そのキーのスコアは 0 で、和に e^{0−m} ずつ足すだけ。`dit_blocks` は呼び出しごとに `nt` (プロンプトの行の数) を取って `k[:nt]`、`v[:nt]` を読み、P·V のコアが割り算の前に (N − nt)·2^{−m s} を足す (gpt-oss の sink と同じ場所)。問い合わせ (4096 行) がキー (512) より多くてよくなり、4096 パッチを 8 × 512 回でなく 1 回で。交差注意だけで 12.7 → 4.6 ms、Anima-Turbo 1024² は 1 ステップ 3.7 → 3.2 秒、Anima-Aesthetic 1024² は −11 %。`nt = 512` は前の分けた形とビット単位で同じ。空の負のプロンプトは `nt = 1`。新しい書き出しには新しい BonsaiAI が要る (古いものは `nt = 0` を渡し、キーの塊 0 個の呼び出しで NPU が止まりうる)。
- IRON の MHA はスコアを bf16 で持つ: スコアが大きく (約 1750) 行の中の差が小さいと、条件つきの出力だけがずれて CFG で崩れる。外したモデルでは、キーから先頭のキーを引いて (softmax は変わらない) 直した。Anima の交差注意で同じことが起きていないかは見ていない。
- max は今も引く。q と k の norm の重みから上限 (√128·max|w_q|·max|w_k|: Anima で 15〜45、ブロック 25 だけ 221) が出せるので省けるが、自己注意は P·V (bf16、1 呼び出し約 6000 サイクル) と softmax (1 塊約 6000) が釣り合っていて縮まない見込み。試していない。

| モデル | `mha` | `exp` |
|---|---|---|
| Anima | qk | exact (`--dit-exp`。mid は 3 % のわりに絵が動く: 潜在 0.976 / 19.6 dB) |
| FLUX.2 klein、FLUX.3 klein | qkpv | mid |

## DiT のブロック

Anima の 4 ブロックのグラフ (1024²、下の変更の前) の内訳: 注意 48 %、GEMM 29 %、要素ごとの演算 30 % (ほぼ帯域、約 58 GB/s)、adaLN 1 %。要素ごとの演算は 4096 × 2048 で 1 つ 0.6〜0.8 ms で、IRON の厳密な GELU の 5.2 ms が目立った。

- **重みを bfp16 でホストで詰める** (`iron_q4.bfp16_pack`、`bfp16_linear`、形式 `bfp16`。どの DiT のブロックも既定): MEDIUM の GEMM は bf16 の重みを毎回 bfp16 に丸め、q8 はさらに毎ステップ展開する。書き出しのときに `flm.ProjectionGEMM` が読む `bfp16ebs8` の並びにしておけば、どちらもなくなる。MLP 1 つで q8 25.9、bf16 23.0、bfp16 18.5 ms、誤差は bf16 と同じ。1 重み 9 ビット (q8 は 8.5)。詰める側と GEMM に同じ `pack_b` の tile_n 64、k のタイル 512 (K が倍数でなければ 384、256、128)、ct_k 128 を渡す。違うと黙って並びが狂う。block 型の重みとして登録する (`bfp16_operand`)。払うもの: 書き出しの時間 (Anima の DiT 45 秒 → 3 分 18 秒、`bfp.encode` が直列)、`graph.reference()` は詰めた B を読めないので `--check` は `--proj bf16` の書き出しで。埋め込み、adaLN、最後の層は bf16。
- **GELU** (`iron_gelu.GELUEven`、`kernels/gelu_even.cc`、`--dit-gelu even`): IRON の tanh の表で、引数を float32 で作り (x² は正確、係数は bf16 の limb 2 本)、最近接偶数で丸める。厳密なカーネルの 5.2 ms に対して 1.76 ms。Chroma1 (tanh 近似の GELU で学習。外した) も使っていた。
- **LayerNorm に変調の掛け算を** (`F.layer_norm(x, weight=scale + 1)`、IRON の WeightedRowwise が 2 つ目のコアで): 1.23 → 0.61 ms、ビット単位で同じ。adaLN の全部に。
- **ヘッドごとの RMSNorm** (`iron_headnorm.py`、`kernels/head_rms.cc`): q と k の正規化を (行、ヘッド × 128) のまま 1 回で、重みはヘッドごとに並べる。1.4 → 0.83 ms、ビット単位で同じ。
- **SiLU を float32 で** (`iron_silu.SiLUF32`、`kernels/silu_f32.cc`): IRON の SiLU は x に bf16 の表のシグモイドを掛け、[−6, 2] で平均 1 % (最大 2.3 %) ずれ、x ≤ −8 でちょうど 0。言語モデルの残差には値が数十〜数百のチャネルがあり、SwiGLU の up の枝がそこで大きいので、誤差が MLP の出力に出る。x / (1 + exp(−x)) を IRON の `exp_f32` と float の逆数で計算し、bf16 へ 1 回だけ丸める: 平均誤差 0.14 %、512 × 9728 で 12 ms。FLUX.2 klein の Qwen3 の文の符号器で使う (最後の潜在 0.916 → 0.977)。DiT の SwiGLU は表のまま (ゲートが大きな負にならない: 1 ステップ目の速度 0.99999)。ほかの SwiGLU の言語モデルの書き出しにも当てはまりうる。未確認。

## 文の符号器

| モデル | 符号器 | 形 |
|---|---|---|
| Anima | Qwen3-0.6B と LLM アダプタ | GEMM は `HIGHEST`: MEDIUM (bfp16) は文で誤差を広げる (HIGHEST で潜在 0.938 → 0.966、時間は同じ) |
| FLUX.2 klein | Qwen3-4B、3 層ずつ 9 組、bf16、HIGHEST | 層 9、18、27 の後の隠れ状態 (7680 列)。`ctx` (context_embedder) は GEMM 3 つの和 (K 7680 は HIGHEST の GEMM に入らない) |
| FLUX.3 klein | Qwen3-VL-4B、32 層を 4 層ずつ 8 バンク、bf16 | FLUX.2 の `Qwen3Layer`。bf16 なので文の流れの 9.7e6 も縮めずに持てる |

- **T5 の相対位置のバイアス**: IRON の MHA にバイアスはない。8 頭ずつ、ブロック対角の K^T と V^T (射影の重みを GEMM の左に置いて頭ごとに連続に出す) で 1 回の GEMM に 8 頭の点数を並べ、バイアスは float32 の Softmax の足し算に定数 (全層で共有、64 MiB) で。パディングのマスクは点数の GEMM の縮約に入れる (K^T の 1 行に (mask − 1) × 16384、q の 1 列を 1)。1 回 3.2 秒、transformers の float32 とコサイン 0.9997。
- **Qwen3 の符号器のパッドの行 (FLUX)**: パイプラインはプロンプトを 512 トークンに埋め、DiT はパッドの行も読む。transformers のマスクでは、パッドの行はプロンプトのキーだけを見て (自分のキーも見ない)、プロンプトの行は因果。そこで注意を 2 回走らせる (全部の行の因果の注意と、最初の `nk` 個のキーへのマスクなしの注意) 。入力 `sel` (プロンプトの行で 1) で行ごとに選ぶ (`op + (oc − op)·sel`)。512 行を 1 回で (64 行ずつだとパッドの行の自分のキーが因果の注意に入る)。
- パッドの行は重みの丸めに敏感 (層 18 のパッドの行: q6 0.904、q8 0.995、bf16 0.9995): FLUX.2 の文の符号器は bf16 のまま (埋め込みの表 0.8 GB を含めて 5.9 GB、1 層 61 → 52 ms)。
- FLUX.3: 文は 512 行の枠 (`text_slot`) に持ち、プロンプトの R 行 (トークン数を 80 の倍数に切り上げ、80〜480) だけがキー (`nk`)。パッドの位置は 1。512 行で回して先頭 R 行を使う (80 行で回したものと同じ)。

## VAE

- **格子のままの畳み込み** (`iron_gridgemm.GridGEMM`、`--vae-conv grid`): 3 × 3 の畳み込みを im2col の写しなしで (3C が 512 以下なら GEMM 1 つ、超えればカーネルの行ごとに 3 つを足す)。拡大は 4 つの出力の位相をそれぞれ低い解像度の 2 × 2 の畳み込みで (`PHASE_TAPS`)。Anima の VAE 9.3 → 5.2 秒、im2col の形と PSNR 52.7 dB。
- `iron_anima.gemm_tile_n`: 8 列が埋まる tile (128 チャネルで 16、256 で 32。既定の 64 は 2 列しか使わなかった)。畳み込みだけなら 46 → 38 ms、段では 1.37 → 1.30 秒だけ: 畳み込みは A の転送 (画素あたり 9C) が律速。
- **バッファの使い回し** (`iron.GraphOptions.Buffers`、`anima.VAE.run`): デバイスのバッファの確保はページを固定して 0 で埋める (1 GiB に 0.33 秒、解放 0.11 秒)。FLUX.2 の復号器の 10 段は 17.6 GiB を確保し、14 秒のうち 7.6 秒だった。今はバンクの種類ごとに 1 つ (いちばん大きい段の大きさ) を全部の段で使い、像を読んだ後ろを 0 にする (複数スレッドで)。段の出力のバッファはそのまま次の段の入力 (コピーなし、ホストに写さない)。FLUX.2 1024² の復号 14 → 8.5 秒、符号化 7.3 → 4.8 秒、プロセスの RSS 1.17 → 0.16 GiB、ピーク 4.9 → 3.8 GB、バイト単位で同じ。Anima-Turbo の復号 5.2 → 4.0 秒。(VAE の段の先読みは確保の時間を隠していただけなので、なくした。)
- **GroupNorm** (`iron_groupnorm.py`、`kernels/chan_stats.cc`、`kernels/row_affine.cc`、`anima.HostNorm`、`--host-norm`、既定。FLUX.2 / FLUX.3 klein の FLUX の VAE): IRON には GroupNorm も逆数の平方根もない。
  - `ChannelStats`: bf16 の画素の行から、チャネルごとの和と二乗和 (float32) を 1 回読みで出す。
  - ホストの段 (`iron.Graph.After`、`anima.hostNorms`): グラフの状態に名前を付け (`gn0.stats`、`gn0.row`) マニフェストの `buffers` から場所を引く。コマンドが終わるたびに、次の norm の統計が書かれていれば (NaN の印が消えていれば)、float64 で群の平均と分散を出し、a = γ/√(var + eps)、b = β − 平均·a を書く。統計の演算の直後でコマンドを切る。
  - `RowAffine`: y = x·a + b。a と b は bf16 の limb 2 本、計算は float32、bf16 へ 1 回だけ丸める (x·a と b は結果の何倍にもなる: この復号器では |平均|/標準偏差が 6.3 まで)。
  - 格子のまま計算する (0 のパッドは和に入らない。本当の画素数で割る)。norm と SiLU の後はパッドが 0 でなくなるので、画像の画素だけを新しい格子に写す。
  - 1024² の 128 チャネルで約 157 ms → 数 ms。復号 512² 2.6 → 1.8 秒、1024² 8.5 → 7.0 秒、float32 との PSNR 52.6 dB。`graph.reference()` はホストの段を知らないので照合は NPU で (`BONSAIAI_DEV=klein.decode=1` で `--noise` の潜在を復号、`--dump`)。
  - 前の形 (GEMM と RMSNorm の約 20 演算。1/√(var + eps) は群の行と定数の行の RMSNorm で) はホストが要らず、`--no-host-norm` で残る。
- **潜在が 16 の倍数でない** (長方形。832 × 1248 で 104 × 156。`iron_anima.GridShape`、Go の `anima.GridShape`): 行の幅 `Wp` は W と詰め物 16 を 16 の倍数に。畳み込みは高さを 16 に切り上げた `Ha` を 16 行ずつ回り、画像の行だけを写す。中ほどの注意は画素を 256 の倍数に詰め、詰めたキーの点数に −30000 を足す (マスクは 1 の列とマスクの行の GEMM で 1 回だけ作る: 16384 列の行ごとの足し算は DMA の記述子が 16 を超える) (`attn_masked`)。正方形は前と同じ計算。
- DMA の繰り返しは 64 まで (列の塊の数を抑える)。100 万画素の転置の GEMM は終わらないので 262144 画素ずつのコマンドに (Chroma1 1024²。外した)。
- ストライド 2 の畳み込み (FLUX.2 の符号器): 同じ大きさの 3 × 3 の畳み込みのあと、1 つおきの画素を半分の格子へ写す (出力 (Y, X) は右と下に 1 画素詰めた畳み込みの (2Y + 1, 2X + 1))。
- VAE は `--batch` でも 1 枚ずつ ([image-models.ja.md](image-models.ja.md))。

## 行を 512 の倍数に詰める

| モデル | 行の並び | 詰め方 |
|---|---|---|
| Anima | [パッチ] | パッチの数を 512 の倍数に切り上げ (832 × 1248: 4056 → 4608)。自己注意のキーは `nk` = パッチの数で切る。詰め物の行の角度は位置 0。出力は先頭のパッチの数だけ取る。グラフは行数で名づける (`dit_blocks_n1024`。4096 行のものは名前なし) |
| FLUX.2 klein | [画像、参照]、512 の倍数でなければ [文 512、画像、参照、0] | double ブロックは文と画像で重みが違い、その境が並び方で変わるので、文が先のときは `doubles_tROWS` (バンクは同じ)。singles、`dit_in`、`dit_out` は行数だけで決まり共有する。1024² を文が先で回すといつもの並びと PSNR 36.9 dB (キーの塊の順の丸め) |
| FLUX.3 klein | singles: [画像 P、文の枠 512]。画像の mode ブロック: P 行 | 画像の行を 0 で P まで詰める (`image_rows`、大きい 7 つはどれも 4096)。キーは [画像 N 行、文 R 行] に詰め直す (`parallel(keys=...)`) ので、1 つのグラフがどのプロンプトの長さにも効く (文を先にすると鍵でない行がキーの間に挟まる)。`dit_in`、画像の mode ブロック、`dit_out` は P だけで決まり、なければ最初の大きさのものを使う (`flux3klein.ironPart`)。`singles` と VAE は大きさごと |

詰め物は何にも効かない: FLUX.2 の 2:3 を 4608 行と 5120 行で回すとバイト単位で同じ。大きい大きさはどれも 4608 行 (1024² と同じ) なので、時間もメモリも 1024² と同じ。

## モデルごと

### Anima-Turbo、Anima-Aesthetic

- DiT: 28 ブロック、4 ブロックずつ 7 組。重みは bfp16、`qk`、正確な指数、交差注意は `trim`、GELUEven。`dit_in` は 1 ステップ 1 回、CFG の 2 回は組ごと。空の負のプロンプトは `text/empty_cond.bin` (`iron_anima.empty_cond`、`anima.EmptyCond`)。
- 8 つの大きさの書き出し (`--size 1024 512 832x1248 880x1168 752x1360 1248x832 1168x880 1360x752`、Aesthetic は `--dit anima-aesthetic-v1.1.safetensors`): 6 分 38 秒、最大 RSS 7.1 GB、5.8 GB。DiT だけ作り直せる (`--only dit`)。

| | 時間 | 最大 (NPU と RSS) |
|---|---|---|
| Turbo 1:1 1024²、8 ステップ | 32 秒 (1 ステップ 3.2 秒) | 2.00 GiB (NPU 1.21) |
| Turbo 1:1s | 9.9 秒 | 1.74 GiB |
| Turbo 長方形 | 31〜32 秒 | 2.04〜2.11 GiB |
| Aesthetic 1024²、25 ステップ、CFG | 2 分 22〜23 秒、2.71 kJ | 2.14 GiB |

- 電力量: Turbo 1024² 572〜615 J (パッケージ全体、`BONSAIAI_ENERGY`)。精度 (Turbo 1024²、diffusers の fp32): 潜在のコサイン約 0.975〜0.98。
- models.yaml の AMD のメモリは 3.5 GB (NPU の最大 2.4 GiB は VAE の段、プロセス 0.9 GiB)。

### Chroma1-Flash、Chroma1-HD (外した)

2026-11-24 に外した。DiT、文のマスク、書き出しと数字は [not-adopted/chroma1.md](../not-adopted/chroma1.md) にある。

### FLUX.2 klein 4B

| グラフ | 中身 | 呼ぶ回数 |
|---|---|---|
| text | Qwen3-4B の 3 層 × 9 組、512 行を 1 回で | プロンプトごとに 9 組 |
| ctx | context_embedder | プロンプトごとに 1 回 |
| mods | 時刻の埋め込みから変調の行 (17 × 3072) | ステップの前に全ステップ分 (0.3 GB を持ち続けない) |
| dit_in | x_embedder (絵と参照の行) | 1 ステップ 1 回 |
| doubles、singles | double 1 つ × 5 組、single 2 つ × 10 組 | 1 ステップ 15 組 |
| dit_out | norm_out と proj_out (絵の行) | 1 ステップ 1 回 |
| vae/s0..s9 | 復号器 (`iron_chroma_parts.FluxVaeStage`。s0 の 1 × 1 が post_quant_conv) | 1 枚 1 回 |
| vae_enc/s0..s5 | 符号器。conv_out と quant_conv は 1 つの畳み込みに畳み、平均の 32 チャネルだけ出す | 参照 1 枚に 1 回 |

- DiT: 幅 3072、24 ヘッド × 128、バイアスなし、MLP は SwiGLU (3 倍)、変調の 3 つの線形はどのブロックも同じ行を使う。RoPE は 4 軸 × 32 (参照の番号、行、列、文の位置)、theta 2000。
- ブロックのグラフは 1 ステップの行数だけで決まり (位置は入力)、行数で名づける (`_nROWS`。最初の大きさのものは名前なし)。1024² の等倍 1 枚と半分 4 枚はどちらも 8704 行で同じ ELF。符号器は辺の長さで探す: 最初の大きさの `vae_enc`、その辺の 1/D の `vae_enc_dD`、なければ `sizes/WxH/vae_enc`。
- 既定の参照の組み合わせ (`default_refs`、`--refs N`、`NdD`、`A+BdD`): 等倍と半分の 4 枚までで、行数が 512 の倍数、12800 行までのもの。それより多い行も名前を書けばコンパイルでき動く (`--refs 3 4`)。行数 1 つで約 12 MB、コンパイル約 70 秒。
- **1 ステップの時間** ≈ 0.70 ms·n + 71 ns·n² (n 行。射影は 1 次、注意は 2 次): 4608 行 4.75 秒、12800 行 21 秒。12800 行までは VAE がピーク (3.9 GB) で、その先は DiT のバンク (double で 1 行 0.088 MiB) が増える (20992 行で 5.6 GB)。
- 8 つの書き出し: 12 分 58 秒、最大 RSS 19.1 GB、16.0 GB (長方形の VAE は 16 行ずつのまとまりで制御コードが増え、大きい)。

| | 時間 | 最大 |
|---|---|---|
| 1:1 1024²、4 ステップ | 28 秒 | 3.60 GiB |
| 1:1s 512² | 10 秒 | 1.90 GiB |
| 長方形 | 27〜28 秒 | 3.53〜3.64 GiB |
| 1024² の編集、等倍の参照 1 枚 | 59 秒 | 3.9 GB |
| 1024² の編集、半分の参照 1 枚 | 37 秒 | 3.8 GB |
| 1024²、等倍の参照 2 枚 | 1 分 41 秒 | 3.9 GB |
| 512²、等倍の参照 2 枚 | 20 秒 | 1.6 GB |

- CPU の float32 との精度: 1024² の生成で潜在 0.961、512² で 0.975。編集 0.996〜0.999 (等倍 2 枚 0.9989 / 40.0 dB)。残りの差は文のパッドの行から来る。
- DiT の形式を CPU で 1 つずつ模擬した誤差 (1 ステップ目の速度): 重みの bfp16 0.99992、線形の入力の bfp16 0.99989、q と k の bfp16 0.99989、3 つとも 0.99964。

### FLUX.3 klein

- 書き出しの道具は、diffusers の配置がなければ単一ファイルの DiT `model.safetensors` (BFL の名前、11.1 GB) を読む (`AMDFiles`。名前は diffusers 0.43.0 の `convert_flux3_transformer_checkpoint_to_diffusers` の置き換えを写し、`norm_out` の行は scale, shift に入れ替える)。変調 (14 行) はホスト (`time.safetensors`)。DiT の重みは bfp16、`qkpv`、指数は mid。
- 8 つの書き出し: 手で 10 分 45 秒、`PrepareAMD` から 11 分 27 秒、最大 RSS 16.8 GB、15 GB (足した 7 つの大きさは `singles` と `vae` だけで 0.2〜0.5 GB ずつ)。

| | 時間 | 最大 |
|---|---|---|
| 1024²、16 ステップ | 1 分 43 秒 (文 2.8 秒、1 ステップ 5.9 秒、VAE 6.5 秒) | 3.59 GiB |
| 512² | 32 秒 (1 ステップ 1.7 秒) | 2.49 GiB |
| 3:4 | 1 分 43 秒 (1 ステップ 5.7 秒) | 3.61 GiB |

- 精度: 512² で潜在 0.9985 / 39.4 dB (91 トークンのプロンプトで 0.984 / 32.2 dB: 自由に回すと細部の軌道が分かれる)。3:4 で diffusers の bf16 と 0.990 / 34.1 dB。

## 測り方

- 最大メモリ: 0.1 秒ごとに `/proc/PID/fdinfo` の accel の fd の `drm-total-memory` (amdxdna のバッファ) と `RssAnon + RssShmem` を読み、和の最大。
- 電力量: `BONSAIAI_ENERGY=10 bonsaiai-image …` (段ごとのパッケージの J、待機を引いたものも)。[memory-and-power.ja.md](memory-and-power.ja.md)。
- `/dev/accel` を使うほかのものがないこと (別のセッションで 1 層 52 → 500 ms になった)。冷えた起動を測るのでなければ、ページキャッシュに載った状態で。

## 未解決

- Anima の mid の指数を float32 と比べる (どちらが近いかわかっていない)。
- Anima: 交差注意の K/V をステップの外へ (演算の 0.7 %)、同じ形の VAE の段を 1 つのグラフに (`up_blocks.3` の ResBlock 3 つ。VAE は Turbo の約 12 %)。
- 自己注意: P·V (bf16) と softmax の釣り合い。その後に max の固定の上限。
- 射影と MLP の GEMM と正規化 (ブロックの約 6 割): カーネルの作業。
- `bfp.encode` を並列に (書き出しの時間)。
- FLUX.2 の参照: 形を変えずに切り出す読み方、`@quarter` (実行は長さで判定する。符号器と指定がまだ)。
- FLUX.2 と FLUX.3 klein の 1 枚の電力量。FLUX.3 klein の編集。
- 空のカーネルのキャッシュから 8 つの大きさを書き出す時間。

## 採らなかったもの

| もの | 理由 |
|---|---|
| AIE2P の exp2 命令 | 速さは limb 1 本と同じで誤差が 1.6 倍 (FLUX.2 の潜在 0.9645 → 0.9629、27.1 → 25.7 dB)。Anima の MHA では 4 倍 |
| IRON の tanh の GELU | 引数を bf16 の積 (それぞれ 0.4 %) で作るので、試験の絵で服が変わった |
| gpt-oss の `gl_exp32` で x·sigmoid(2u) の GELU | 2e-4 以内で正確だが、厳密なカーネルと同じ遅さ (5.5 ms) |
| Anima の `qkpv` | 1024² で時間 −5 %、電力量 −12 % だが、細部が参照から離れた (胸元の形が変わる)。全部の書き出しを作り直すことになる |
| gate と残差 (`x + y·g`) を 1 つの演算子に | 入力 3 つがコアの入力 DMA チャネル 2 本に入らない。2 コアのパイプラインでも約 1 % |
| adaLN を `dit_in` にまとめる | 約 1 % |
| 常駐の配列 (gpt-oss の GptOssStep の形) | 演算子 1 つが数 ms で、切り替えは 1024² で約 5 % |
| Anima の DiT を q6 / GPTQ の q4 に | q6 は潜在の中央値 0.912 (q8 0.957)。q4 は構図が変わる (騎士が 1 人から 2 人に) |
| FLUX.2 の文の符号器を bfp16 (MEDIUM) に | パッドの行が層 18 で 0.983。DiT はそれに敏感 |
| FLUX.2 の文の符号器を 64 行ずつ | パッドの行の自分のキーが因果の注意に入る |
| 最後の single ブロックを絵の行だけに | 編集の 1 ステップの 2 %。グラフがもう 1 つとバンクの共有の扱いが増える |
| GroupNorm の統計を bf16 のベクトルで 1 回読みの二乗和から | 40 倍の桁落ち。和は float32 で持ち、引き算はホストの float64 で |
| FLUX.3 の短い文の枠 (80〜240 行) | MHA の 8 本のパイプラインは 512 行の倍数しか取らない。見込みは 1 枚 1.8 秒 |
| VAE を絵をまたいでまとめる | 段の出力をホストに写し (1 枚 0.5〜0.8 秒)、ピークが上がる |
| VAE の次の段の先読み | バッファの確保を隠していただけで、使い回しで確保がなくなった |
| 畳み込みの 1 回の行数を 64 → 128、256 | 38.0 → 36.5 ms。切り替えの固定費ではない |
