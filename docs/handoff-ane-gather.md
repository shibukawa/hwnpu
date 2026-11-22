# 引き継ぎ: Neural Engine で MoE を動かすための、gather と実行時の重みの調査

作成: 2026-11-18、整理: 2026-11-21。M1 (16 GB、macOS 27、ビルド 26B131) で調べ、M3 と M2 で続けた。コードは main に入っている (2 節)。

**済んだこと**: M1 と M3 の調査 (1 節、1b 節)、LFM2-8B-A1B を hwnpu-chat に組み込み (1b の「hwnpu に組み込んだ」)、M2 での LFM2-8B-A1B と MoE の 1 層 (1c 節、[docs/design/apple-neural-engine.md](design/apple-neural-engine.md))。gpt-oss-20b の専門家を Neural Engine に置く道 (A) は止め、CPU の MXFP4 にした (1b の最後、[docs/design/apple-coreai.md](design/apple-coreai.md))。**残りは 3 節** (M4 での配置、ANE のコンパイラが外す理由)。

目的: Apple 向けの MoE (gpt-oss-20b、Qwen3.6-35B-A3B など) が無いのは、専門家の選択 (gather) が Neural Engine に載らないため ([docs/design/moe.md](design/moe.md) の gpt-oss の節、[model-fit.md](model-fit.md) 4 節)。この判断は Core ML 時代のもので、Core AI では試していなかった。ANE のコンパイラの中身を調べ、流用できるものを探した。

## 1. 分かったこと (M1)

### ANE のコンパイラには gather の部品がある

dyld の共有キャッシュ (`/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e*`) の文字列から:

- ANE 用の MLIR 方言 `anec` の演算: `anec.gather_nd`、`anec.dynamic_input_view`、`anec.input_view`、`anec.top_k`、`anec.sort`、`anec.arg_min_max`、`anec.param_linear`、`anec.param_convolution`、`anec.if`、`anec.state`、`anec.ring_buffer_reader` / `writer`、`anec.mx_dequant` / `mx_quant` ほか
- 内部の部品: `ZinIrGatherUnit`、`ZinIrDynamicSliceUnit`、`ZinIrDynamicGOCUnit`。検証関数 `ANECValidateGatherLayer`、`ANECValidateDynamicSliceLayer`
- gather の条件の文言: 「Dynamic axes in gather is not supported」「Data tensor of gather cannot have multi user mode」「Index/Data tensor dimension %s should be equal to 1」「Gather axes contain isolation axis」「Invalid interleave factor for gather layer」
- 実行時の位置での状態への書き込み: 「With dynamic offset, slice_update's begin must be produced by a concat」「Cannot have multiple slice_update per write_state」(KV キャッシュの書き込みに使えるかもしれない。未調査)
- ANE のコンパイラが受け取る MPS の演算: `mps.Gather`、`mps.GatherAlongAxis`、`mps.GatherND`、`mps.TopK`、`mps.Sort`、`mps.ReductionArgMax`、`mps.Slice`、`mps.StridedSlice`、`mps.Select`、`scf.If` ほか

Core AI の基本演算 (`aicode.primitive.*`) にもある。引数の数は C++ の型名 (`NOperands<N>`) から:

| 演算 | 引数 |
|---|---|
| `gather_nd_v1` | 2 (data、indices) |
| `gather_along_axis_v1` | 3 (input、indices、axis。axis は定数) |
| `slice_v1` | 4 (x、begin、end、stride。stride は定数、begin は実行時でもよい) |
| `slice_update_v1` | 5 (input、update、begin、end、stride) |
| `argsort_v1` | 4 (input、dim、descending、stable。後ろ 3 つは定数) |
| `reduce_index_v1` | 4 と region 1 つ |
| `batch_matmul_v1` | 2 |

流れ: Core AI の `aicode.primitive.*` → (CPU) BNNS / (ANE、GPU) MPS 方言 → MPSGraph が ANE の領域を切り出し、ANE のコンパイラに検証させる。

### M1 で載ったもの、載らなかったもの

| 演算 | Core AI の配置 | Core ML の MLComputePlan の `supportedComputeDevices` に ANE があるか |
|---|---|---|
| `gather_nd` (表が定数でも入力でも) | GPU | ない |
| `gather_along_axis` | GPU | ない |
| begin が実行時の `slice_by_index` | GPU | ない |
| `reduce_argmax` | (ailower 未対応) | ない |
| 重みが入力の `matmul` (fp16) | **全部 ANE** | ある |
| int8 の入力を `cast` で fp16 にし、拡大して `matmul` | **全部 ANE** | (Core ML は仕様 10 以上が要り、試していない) |

Core AI のログ: 「setNDXANEProcedureAttrs: entry '…' has 0 ANE regions」。MPSGraph が gather の部分を ANE に回していない。

### ANE の検証が外す理由 (M1)

MPSGraph は ANE の領域の候補を `ANEValidateNetworkCreate` に渡し、演算ごとに `support` と `errorMsg` を受け取る。この辞書は `log stream --level debug` にだけ出る (`log show` には残らない。読み方は 3.3)。索引の形を変えながら追った:

| 試したこと (probe) | 返事 |
|---|---|
| 索引が int32 (`gnd_const` ほか全部) | Incompatible element type for ANE: expected fp16, f8E4M3, si8, ui8, si16, or ui16 |
| 索引を int16 に、行を丸ごと取る形 (`gnd16_const`、`gnd16_mm`、`gaa16_*`) | Invalid gather axes size 1, must be 3 |
| 3 座標の組、索引 (1024, 3) (`gnd16_pts`) | Invalid gather index tensor width size 1024, must be 1 |
| 索引 (32, 32, 1, 3) (`gnd16_pts_hw`) | Invalid gather index tensor channel size 96, must be 3 |
| 索引 (1024, 1, 3)、(1, 1024, 1, 3) (`gnd16_pts_w1`、`gnd16_pts_4d`): 条件を全部満たす形 | **Cannot decompose layer on this architecture** |
| begin を int16 にした動的切り出し (`dslice16_*`) | Core AI → MPS の段で「'mps.strided_slice' op operand #1 must be … mps index type」(int32 か int64 しか取らず、ANE は int32 を取らない)。プロセスが落ちる |

ANE の gather は、索引が si16 などで、3 つの軸の座標の組 (チャネル 3、幅 1) の、要素ごとの取り出しだけ。M1 (h13) ではそれも「分解できない」で止まる。**M1 では gather は使えない**。新しい世代では最後の段を通るかもしれない (3.1)。通っても要素ごとなので、専門家の重み (数百万要素) を取るには索引が大きすぎ、埋め込みの行などの小さい表向き。

### 実行時の重みの速さ (M1)

8 行 × 2880 × 2880 (gpt-oss の幅)、200 回の平均。入力はプログラム自身の配列 (`INPLACE=1`、呼ぶたびのコピーなし)。1 回の呼び出しの固定費は約 0.2 ms ([docs/design/apple-coreai.md](design/apple-coreai.md))。

| 重み | 時間 | CPU (milcpu) との照合 |
|---|---|---|
| 定数 fp16 (`lin_const_big`) | 0.52 ms | cos 1.000000 |
| 定数 int8 行ごとの拡大 (`lin_i8_big`) | 0.35 ms | cos 1.000000 |
| 入力 fp16 (`mm_dyn_big`) | 0.61 ms | cos 1.000000 |
| 入力 int8 + cast + 拡大 (`mm_dyn_i8_big`) | 0.54 ms | **未確認** (milcpu が int8 の入力を作れない) |

呼ぶたびに入力をコピーすると (`INPLACE` なし)、fp16 1.16 ms、int8 0.72 ms。M1 のコピーは約 29 GB/s。

### MoE への当てはめ (見込み、未実装)

- ルーターは ANE、専門家の選択 (top-k) はホスト。層ごとにルーターのロジットをホストに戻す
- 専門家の重みは専門家ごとの Core AI の入力配列に int8 で置き、選んだものを層の形ごとに 1 つのプログラムに結び付ける。専門家ごとのプログラムは要らない (ANE のプログラムは機械全体で約 128 まで)
- 往復は層ごとに 1 回増える。gpt-oss の 24 層で 1 トークン 10 ms ほどの見積もり
- gpt-oss-20b の専門家は int8 で約 19 GB。16 GB の機械には載らない。32 GB 以上か、小さい MoE (LFM2-8B-A1B など) が対象

### MoE の 1 層を実行時の重みで (M1、`aicheck moe`)

専門家の SwiGLU (gate と up を 1 つの積、down、ルーティングの重みで足す) を、専門家ごとの int8 の配列 (と列ごとの fp16 の拡大) を入力にして組み、同じ計算を定数の int8 の重みで組んだもの (同じ大きさの密なモデルの費用) と比べた。1 行 (デコード)。照合は Go の float32 で cos 1.000000 (int8 の入力の正しさはこれで確認)。

形: gpt-oss-20b (幅 2880、専門家 2880、32 個、top-4、1 個 24.9 MB) と Qwen3-30B-A3B 型 (2048、768、128 個、top-8、1 個 4.7 MB)。

**1 プログラムに入る専門家は 5 個まで** (実行時の重み行列 10 枚)。6 個から急に遅い:

| top-k | gpt-oss | Qwen 型 |
|---|---|---|
| 4 | 3.75 ms | 1.44 ms |
| 5 | 5.00 ms | 1.79 ms |
| 6 | **21.53 ms** | **7.78 ms** |
| 8 | — | 10.70 ms |

入力の数でも大きさでもない (8 KB の入力 32 個、1 MiB の入力 16 個では段差なし)。top-8 は 4 個ずつ 2 回呼ぶ (`-split 4`)。

**結び付け直しの費用**: Core AI は、一度 ANE に割り当てた配列を数百 MB ぶん覚えている。超えると結び付けるたびに約 5 GB/s 相当の割り当て直しが起きる。1 層 (3 回の中央値):

| | gpt-oss (1 プログラム) | Qwen 型 (4 個 × 2 回) |
|---|---|---|
| 定数の重み (基準) | 2.14 ms | 1.64 ms |
| 実行時の重み、同じ専門家 | 3.99 ms | 3.22 ms |
| 回す専門家 少 (gpt-oss 4〜16、Qwen 8〜32) | 4.23〜4.43 ms | 3.86〜4.07 ms |
| 1 層の全部 (gpt-oss 32 個 796 MB、Qwen 128 個 604 MB) | 5.22 ms | 5.83 ms |

実際のモデルは全層の専門家を回す。4 層ぶん持たせて層もばらばらに選ぶと (gpt-oss 3.2 GB、Qwen 2.4 GB):

| 1 層 (ルーター 0.2 ms と top-k を含む) | gpt-oss | Qwen 型 |
|---|---|---|
| 定数の重み (基準) | 2.18 ms | 1.64 ms |
| 選んだ配列を結び付ける | 20.83 ms (+854%) | 13.67 ms (+735%) |
| **K 個の枠を結び付けたまま、選んだ専門家を CPU で枠へコピー** (`-copy`、8 スレッド) | **8.93 ms (+309%)**、うちコピー 4.30 ms | **6.16 ms (+277%)**、うちコピー 1.66 ms |

コピー方式は照合 cos 1.000000 (書き換えた中身は呼ぶたびに読み直される)。コピーは約 23 GB/s。

1 トークンの見込み (専門家の部分だけ): gpt-oss 24 層 × 8.9 ms ≈ 214 ms (密な同等 52 ms、AMD は全体で 20 tok/s)、Qwen3-30B-A3B 48 層 × 6.2 ms ≈ 296 ms。どちらも int8 で 16 GB に載らない (gpt-oss 約 19 GB、Qwen3-30B 約 29 GB)。16 GB に載るのは LFM2-8B-A1B のような小さい MoE。

まだ詰められること: 1 専門家ずつの呼び出しにして、次の専門家のコピーを今の専門家の計算と重ねる (呼び出し 1 回 0.2 ms と引き換え)。IOSurface や Metal のバッファの入力 (`tools/coreai/moebind`、Swift) は、NDArray より速くならなかった (gpt-oss の 32 専門家で 6.5〜8.0 ms、NDArray 5.4〜8.4)。moebind は入力 41 個 (Qwen 型 top-8) で ANE の要求が失敗する (`0xe00002bc`)。

## 1b. M3 で分かったこと (2026-11-18)

M3 (16 GB、macOS 27.2.1、ビルド 26B131)、ブランチ `t3/moe-m3-ane-gather-check` (`t3/ane-gather-probe` から)。

### gather は M3 の Neural Engine に載り、値も正しい

3.1 の probe: int32 の索引 (`gnd_*`、`gaa_*`) と動的切り出し (`dslice_*`) は M1 と同じく GPU。Core ML の計画でも CPU。**int16 の索引の probe は全部 `ane=true`** (`gnd16_const`、`gnd16_mm`、`gnd16_pts*`、`gaa16_*`)。M1 で「Invalid gather axes size 1, must be 3」だった行を丸ごと取る形 (`gnd16_const`) も載る。

`aicheck gather` (新規) で値を確かめた: 定数の表 (E, out, in) から、int16 の専門家の番号 (入力) で gather_nd して x と積を取り、Go の float32 と比べる。どの選び方でも cos 1.000000、出力はどれも選んだ専門家と一致。表の大きさは時間に効かない (E 16 と 64 で同じ。選んだ分だけ読む)。

| 形 (1 専門家 1024 × 1024) | 配置 | 1 回 |
|---|---|---|
| 定数 fp16 の K 枚と積 (基準、K 4) | ANE | 0.33 ms |
| fp16 の表、gather_nd 1 つで K 個 (K 1 / 2 / 4) | ANE | 0.55 / 2.9 / 6〜7 ms (K に比例しない) |
| fp16 の表、1 個ずつ gather_nd (`-each`、K 2 / 4 / 8) | ANE | 0.89 / 1.53 / 2.80 ms |
| int8 の定数 (constexpr ではない `I8s`) を gather → cast → 行の拡大 (`-int8 -each`、K 4) | ANE | 1.66 ms |
| 同じ int8 を gather_nd 1 つで K 4 | **GPU** (ANECCompile FAILED) | 0.77〜0.99 ms |
| constexpr の int8 の 3 次元の表 (`-int8 -raw8=false`) | GPU (gather の無い基準も GPU。3 次元の constexpr と batch の matmul のせい) | |

ANE の gather は**要素あたり約 3 G 要素/s** (fp16 でも int8 でも 1 M 要素で約 0.33〜0.36 ms)。定数の重みを読む速さ (約 35〜50 GB/s) の 1/6 ほど。専門家の重みを表から gather すると、LFM2-8B-A1B の 1 層 (4 個 × 11 M 要素) で約 15 ms になり、下の結び付けより遅い。**gather は小さい表 (専門家ごとの拡大、埋め込みの行など) に使う**。

### MoE の 1 層 (M3、`aicheck moe`)

M1 との違い:

- **1 プログラム 5 個までの段差は無い**。Qwen 型の top-5 / 6 / 8 が 1.13 / 1.34 / 1.76 ms (1 プログラム)。`-split` は要らない
- 結び付け直しがずっと安い。4 層 (gpt-oss 3.2 GB、Qwen 2.4 GB) で層もばらばらに選んで: gpt-oss 4.38 ms (M1 20.8)、Qwen 型 4.69 ms (M1 13.7)。gpt-oss はコピー方式 (6.19 ms) より速い
- コピー方式: gpt-oss 6.19 ms、Qwen 型 3.86 ms。コピーは約 16〜23 GB/s

結び付けの費用は、配列の**数**と**全体の大きさ**の両方で増える。LFM2 型 (1 層 352 MB、128 配列) で層の数を変えて: 1 層 1.81、4 層 2.11、8 層 2.40、11 層 2.59、14 層 2.70、18 層 3.54、22 層 (全部、7.75 GB) 5.32 ms。Granite 1B 型 (1 層 50 MB) では 8 層 3.70、16 層 4.22、24 層 4.53 ms (配列が 32 個/回と多い)。

### 実在の小さい MoE を全層ぶん持たせて (M3、16 GB)

config は Hugging Face のもの。MoE の層の専門家を全部、Core AI の入力配列に int8 で持たせ、毎回ランダムな層とランダムな専門家を選ぶ。1 層あたり (ルーター 0.15〜0.18 ms は別)。どれも照合 cos 1.000000。

| モデル (MoE 層 × 専門家、top-k) | 専門家の int8 | 定数 (同じ大きさの密) | 4 配列を結び付け (`moe`) | コピー方式 (`moe -copy`) | 2 配列 + 拡大は gather (`moepack -two`) | 1 配列 + 拡大は gather (`moepack`) |
|---|---|---|---|---|---|---|
| Granite 3.1 1B-A400M (24 × 32、8) | 1.2 GB | 0.73 ms | 4.53 | **2.2** | 3.07 | 2.56 |
| Granite 3.1 3B-A800M (32 × 40、8) | 3.0 GB | 0.72〜0.85 ms | 4.77 | **3.0** | 3.53 | 3.37 |
| OLMoE-1B-7B (16 × 64、8) | 6.4 GB | 1.22〜1.31 ms | 5.70 | **4.2** | 5.05 | 7.89 |
| LFM2-8B-A1B (22 × 32、4) | 7.75 GB | 0.97〜1.03 ms | 5.32 | 4.66 | **4.35** | 7.05 |

(コピー方式の列はコピー込みの専門家の呼び出し。`moepack` の 1 配列は (H, 3I) に gate|up と down の転置を並べたもの。実行時の重みを切り出すぶん、大きい専門家では計算が遅くなる。)

1 トークンの見込み (専門家とルーターだけ。注意・conv・頭は別): Granite 1B 約 55 ms、Granite 3B 約 100 ms、OLMoE 約 70 ms、LFM2-8B-A1B 約 100 ms。どれも同じ大きさの密 (定数の重み) の 3〜5 倍。**M3 の 16 GB で、8B 級までの MoE は Neural Engine で動かせる見込み** (LFM2-8B-A1B で 1 トークン 120 ms 前後、8 tok/s ほど)。

### 実際の LFM2-8B-A1B で生成 (M3、`aicheck moeserve` + `tools/ref/lfm2moe_ane.py`)

本物のチェックポイント (bf16、16.7 GB、`~/Library/Caches/hwnpu/original/LiquidAI/LFM2-8B-A1B`) で、MoE の 22 層を Neural Engine に、残り (conv、注意、密な 2 層、埋め込み、頭) を transformers の float32 (CPU) にして貪欲に 48 トークン生成した。専門家は行ごとの int8 (absmax / 127、拡大は fp16)。gate|up (H, 2I) と down (H, I) の 2 配列で、拡大は層ごとの表から ANE の gather で取る。全層の専門家 (7.75 GB) を持たせたまま、ルーターも ANE で回す (ルーターの重みは層ごとの入力配列)。選択 (sigmoid + expert_bias の top-4、正規化) はホストで行う。基準は同じモデルの専門家を bf16 → float32 で CPU で計算したもの。

| | 結果 |
|---|---|
| 生成した文 | 基準と 48 トークンとも同じ (「The sky appears blue because Earth's atmosphere scatters …」) |
| 基準の生成した列での KL(ref ‖ ane) | 平均 0.0019、最大 0.043、top-1 一致 100% |
| 選んだ専門家の一致 | 99.0% (層ごとに 97〜100%) |
| 1 行 1 層 (ANE、デコード中) | ルーター 0.33 ms + 専門家 2.7〜3.0 ms |
| 専門家の部分の 1 トークン | 22 層 × 約 3.2 ms ≈ 70 ms |

全体の 820 ms/トークンは、CPU の PyTorch (conv の参照実装) とパイプの往復が大半。ANE の部分だけなら、注意・conv を ANE に載せて (lfm3 と同じ形) 1 トークン 90〜100 ms、10 tok/s 前後の見込み。

### hwnpu に組み込んだ (M3、2026-11-19)

LFM2-8B-A1B が hwnpu-chat (`-m LiquidAI/LFM2-8B-A1B --backend coreai`) で動く。専門家は Neural Engine、選択はホスト。

- `internal/llm/lfm2moe` (新しいアーキテクチャ、`lfm2_moe`): LFM2 の conv と注意 (`out_proj`、`q_layernorm`、gate と sandwich norm なし)、密な 2 層、ルーターまで。lfm3 は触っていない
- `llm.Routed` (arch.go) と `llm.PlanRouted` (split.go): MoE の層ごとに部品を切る。部品は `moe_x_N` (ffn_norm の出力) と `moe_logits_N` (fp16 のルーター) を出して終わり、次の部品は `moe_o_N` を足して始まる。ConvertAI は専門家のプログラム (`model.experts.aimodel`) と行ごと int8 の専門家 (`model.experts.safetensors`、7.22 GiB) を書き、manifest の `experts` に記す。Core AI だけ、paged と tiers なし
- `internal/llm/experts`: 専門家のプログラム (2 配列 + 拡大の gather、silu は exp/real_div)、選択 (`Routing.Choose`)、重みファイルの書き出し
- `internal/llm/chain/moe.go`: `moe_logits_N` を出す部品の後で、行ごとに上位 k 個を選んで専門家のプログラムを呼ぶ。入力 `last` があれば有効な行だけ。全層の専門家はプログラム自身の入力配列に一度だけ読み込み、`Share` した chain と共有する
- chat: MoE のモデルは cache tiers なし (export ごとに専門家のファイルが付くため)、draft なし (1 回 8 行だと専門家も 8 行: 2.3 tok/s)。LFM2 の会話形式 (`chat.LFM2`、think なし)
- `hwnpu-dev generate -score` を KV キャッシュのあるモデルにも (1 トークンずつ)

| LFM2-8B-A1B (M3 16 GB、lut8 + 専門家 int8) | 結果 |
|---|---|
| 変換 (hwnpu-dev convert) | 21 s、23 部品 + 専門家 |
| 読み込み | 2.7 s (コンパイル済み)、全部品 NPU |
| 生成 (hwnpu-dev generate、1 トークン 1 回) | 15.4 tok/s (64.7 ms/トークン) |
| 生成 (hwnpu-chat) | 14.1 tok/s |
| 基準 (bf16 専門家、float32) の生成した 48 トークンでの KL | キャッシュなし 0.0062 (argmax 48/48)、キャッシュあり 1 トークンずつ 0.0056 (47/48) |

外れた 1 か所は基準でも ' through' 53% と ' than' 43% の僅差。そこでキャッシュあり生成の文が分かれる。

回帰: LFM3-1.7B は変更前 (`t3/ane-gather-probe`) と同じ文、同じ 57.4 tok/s。変換し直しもない (プログラムが同じ)。

プロンプト (2026-11-19): 呼び出しの全部の行を専門家 1 つずつで回すプログラム (`experts.BuildRows`、`model.experts.q128.aimodel`、出力を次の呼び出しの acc に) と、MoE のモデルのプロンプトのプログラムを 1 回 128 に。1503 トークンのプロンプト 61.7 s → 15.6 s (64) → 10.9 s (128)。KL 0.0057 (48/48)。生成の 1 トークン 70 ms (文脈 4096) のうち専門家 57.5 ms (1 層 2.6 ms、呼び出しそのもの)。

長い文脈と道具 (2026-11-19): MoE の切れ目と注意の分割を合わせた (`llm.PlanRouted` + `Paged`、29 部品、文脈 128000)。16.9K トークンの passkey を見つけた。LFM2 の道具の形式 (`chat.LFM2`、テンプレートと 9 会話一致)、API で呼び出しと応答が往復する。agent の課題はモデルが edit の `old` を外す。

残り:生成の専門家の呼び出しを次の部品に入れても 4 % ほど。4 ビットの専門家 (2026-11-19): 精度は 64 列ごとの scale で KL 0.023 (専門家だけ、int8 は 0.0019、行ごと 0.071、`moeserve -bits 4 -group 64`)。速さはどれも int8 (1 層 1.7〜2.6 ms) に負ける: int8 に 2 つ詰めて `floor_div` で取り出す 8.7〜11 ms (値も誤り)、Core AI の int4 配列 (`NDArray.ScalarType.int4`、1 バイトに 2 つ、下位の 4 ビットが先) を `blockwise_shift_scale` で戻す形は正しいが GPU に載り 6.4〜9.8 ms、int4 配列を cast する形は Neural Engine で正しいが 85 ms (`aicheck moe4`)。int4 の入力 (`types.I4`、ailower の `sint4`、shim) と `floor_div`、`mod`、`round` の lowering は残した。

### gpt-oss-20b の見込み (M3、2026-11-19)

専門家は 24 層 × 32 × (2880 × 5760 + 2880 × 2880) = 19.1 B。int8 で 19.1 GB (16 GB に載らない)、4 ビットで 9.6 GB (注意 0.64 B と出力層 0.58 B を lut8 で足して約 11.5 GB)。

4 ビットの専門家を int8 に 2 つ詰め、`+128` で負をなくして `floor_div` で取り出す形 (`aicheck moe4`):

| 形 (4 専門家、1 層) | LFM2 型 | gpt-oss 型 |
|---|---|---|
| 行ごとの scale | 8.7 ms | 17.3 ms |
| 64 列ごとの scale | 10.5 ms | 21.9 ms |

gpt-oss は専門家だけで 24 層 × 17〜22 ms = 0.4〜0.5 s/トークン (2 tok/s ほど)。時間の大半は重みの要素ごとの演算 (cast、+128、floor_div、×16、−)。int8 の 1 層 1.7 ms も cast が律速と見られる (Neural Engine の要素ごとの演算は約 16〜26 G 要素/s)。

値: Core AI の CPU (`HWNPU_COREAI_UNITS=cpu`) では正しい (cos 0.9998) が、Neural Engine では NaN。取り出しだけのプログラムは Neural Engine で全部の int8 について正しい (`aicheck fdiv`)。`mod` を入れたプログラムは GPU に回る。どこで無限大になるかは未調査 (`HWNPU_DEV=ailower.cut` で切り分ける)。

### gpt-oss-20b の専門家を 4 ビットで Neural Engine に (A の道、M3、2026-11-19、`aicheck mx`)

| 形 | 置き場所 | 値 | 時間 |
|---|---|---|---|
| MX の定数 (`float4E2M1FN` の符号と `float8E8M0FN` のスケール、`blockwise_shift_scale`、ailower は `mlpackage.Builder.MXFP4` から) | GPU。coreai-build で h15g、h16g、h17g 向けにコンパイルしても Neural Engine の領域なし (同じ行列の lut4 にはある) | cos 1.000000 | gate と up (5760 × 2880) 0.73 ms |
| MX の入力 | MPSGraph の `copy_with_constraints` が fp4 の配列を通さない | — | — |
| チェックポイントの符号を uint4 の入力に、e2m1 の 16 値の表で引き (`lut_to_dense`)、ブロックのスケールを掛ける | Neural Engine | cos 1.000000 | 1 専門家 49 ms (表引きだけで 47) |
| int4 の入力と `blockwise_shift_scale` (`linear`) | GPU | — | — |
| 参考: lut4 の定数 (`linear`) | Neural Engine | (スケールなし) | 0.33 ms |

`matmul` で書くと lut4 の定数でも GPU に回る。`linear` なら Neural Engine。どの 4 ビットの形も CPU の 1 層 1.0 ms (B の道、[docs/design/apple-coreai.md](design/apple-coreai.md)) に届かないので A はここで止めた。M3 の Neural Engine の検証の返事は `<private>` で読めていない (3.3 のプロファイルが要る)。

## 1c. M2 で分かったこと (2026-11-19〜20)

M2 でも LFM2-8B-A1B (専門家を Neural Engine、拡大は gather_nd) が動いた。

| LFM2-8B-A1B (hwnpu-chat) | M2 | M3 |
| --- | --- | --- |
| 配置 | 全部 NPU | 全部 NPU |
| 生成 | 12.0〜13.1 tok/s | 14.1 tok/s |
| 専門家 (1 層 1 行) | 3.83 ms | 2.7〜3.0 ms |
| プロンプト (M2 約 1620、M3 1503 トークン) | 14.4〜43.3 s (37〜112 tok/s、回ごとに違う) | 10.9 s (138 tok/s) |
| 初回 (取得・変換・コンパイル) | 17.5 分、最大 12.9 GB | — |
| 2 回目以降の読み込み | 3.1〜3.2 s、8.0 GB | — |

M2 のプロンプトの時間のばらつきは**メモリ不足**による (M2 での `HWNPU_DEV=moe.time` と `vm_stat`)。呼び出しの数は同じ (36,410 層行) で、遅い回ほど専門家の 1 層行が長く (0.42 → 0.54 → 1.21 ms)、ページの展開が多い (0.5 M → 1.53 M)。同じ配列のまま 2 度目を呼ぶと 1.26〜1.32 ms で変わらず、延びるのは結び付けた最初の呼び出しだけ (2.32 → 3.97 ms)。スワップ 5.5〜7.75 GB 使用、圧縮に約 6.6 GB。熱 (`pmset -g therm` に記録なし) とほかの Neural Engine の利用 (2 度目も遅くなるはず) ではない。`memory_pressure` は空き 68〜75% と出て判断に使えない。

- `HWNPU_MOE_WIRE=1`: 専門家の配列を mlock で固定する (既定は off。16 GB で 7.2 GB 固定すると他のアプリが押し出される。メモリが足りない 16 GB の機械で勧める)。M2 (1,624 トークン、スワップ 7.9 GB 使用) で 3 回ずつ交互に: WIRE 13.2 / 13.1 / 12.7 s (展開 0.07〜0.31 M)、なし 14.4 / 13.9 / 13.6 s (0.42〜0.65 M)。以前の mlock の試しも 13.4〜14.8 s と遅い回が出なかった。M3 では 9.2〜9.6 → 8.7 s、生成 15.3 → 16.1 tok/s
- 結び付けは費用ではない: 層と専門家ごとに request を持って配列を一度だけ結び付けても (試して外した)、メモリに余裕があれば M2 も M3 も同じ (M2 14.5 s、0.38〜0.39 ms の層行。M3 9.2〜9.3 s)。「最初の呼び出し 2.32 ms、2 度目 1.26 ms」は遅い回でだけ開く差で、圧縮されたページの展開だった。M2 の最良 14.5 s と M3 の 8.7〜9.2 s の差はチップの差 (生成と同じ割合)
- やらないこと: プロンプトを層ごとに全部の塊を回す順に変え、専門家を層ごとに 1 度だけ結び付ける。結び付けがほぼ無料なので速い回は速くならない
- MoE の 1 層 (`aicheck moe`、[docs/design/apple-neural-engine.md](design/apple-neural-engine.md)): M1 の「1 プログラムに専門家 5 個まで」の段は M2 にない (M3 と同じ)。実行時の重みは定数の重みの半分の速さ (gpt-oss の専門家で 26 GB/s と 50〜52 GB/s)

## 2. コード

| ファイル | 変更 |
|---|---|
| `internal/coreai/ailower/ops.go` | `gather_nd`、`gather_along_axis` を Core AI の基本演算に。begin が定数でない `slice_by_index` を `slice_v1` に |
| `internal/coreai/ailower/ailower.go` | `isConst`。int8、int16 の入力 (`sint8` / `i8`、`sint16` / `i16`)。`cast` の `int16`、`int8` は ops.go |
| `internal/backend/coreai/coreai_darwin.go` | int8、int16 の入出力 (`types.I8`、`types.I16`) |
| `internal/backend/coreml/plan_darwin.go` | MLComputePlan の `supportedComputeDevices` に ANE があれば、装置の名前に「(the Neural Engine supports it)」と付ける (実験用) |
| `internal/imagegen/milpart/milpart.go` | `InputI32`、`InputI16`、`InputI8` |
| `internal/mlpackage/program.go` | `Int16` |
| `tools/coreai/aicheck/probe_darwin.go` | probe の追加 (下)。`COREML=1` で Core ML、出力に理由と演算ごとの装置 |
| `tools/coreai/aicheck/run_darwin.go` | `INPLACE=1` でプログラム自身の入力配列 (`backend.InputAllocator`) を使う |
| `tools/coreai/aicheck/moe_darwin.go` | `aicheck moe` (MoE の 1 層の計測。`-model gptoss\|qwen3moe\|small`、`-layers`、`-topk`、`-split`、`-copy`、`-fp16`、`-scale-out`)、`aicheck locked CMD` (NPU のロックを取って CMD を回す) |
| `tools/coreai/aicheck/mx_darwin.go` (M3) | `aicheck mx`: gpt-oss の専門家を MX の定数 (`-const`、`-form mx\|lut4`、`-only`)、uint4 の符号と表 (`-form lut`)、int4 (`-form int4`) で。`-write` で .aimodel を書いて止まる (coreai-build で他のチップ向けに) |
| `tools/coreai/aicheck/gather_darwin.go` | `aicheck gather` (M3): 定数の表から int16 の番号で専門家を gather_nd して積。Go の float32 と照合、密な定数と時間を比べる。`-e`、`-out`、`-in`、`-k`、`-each`、`-int8`、`-raw8` |
| `tools/coreai/aicheck/moepack_darwin.go` | `aicheck moepack` (M3): 結び付ける配列を減らした専門家の層。`-scales gather` で拡大を層ごとの 1 配列から ANE の gather で取る。`-two` で gate|up と down の 2 配列 |
| `tools/coreai/aicheck/moeserve_darwin.go` | `aicheck moeserve CKPT`: 実際の LFM2-MoE の専門家とルーターを ANE で回すサーバー (標準入出力の 2 進) |
| `tools/ref/lfm2moe_ane.py` | LFM2-MoE の残りを transformers で回し、MoE を基準 (`run ref`) か moeserve (`run ane`) にして生成・比較 (`compare`) |
| `internal/mlpackage/program.go` | `I8s` (constexpr ではない int8 の定数。gather の表) |
| `tools/coreai/aicheck/moe_darwin.go` (M3) | `-model lfm2moe\|olmoe\|granite3b\|granite1b` |
| `tools/coreai/moebind/main.swift` | NDArray、Metal のバッファ、IOSurface の入力で結び付けの費用を比べる (Swift) |

probe (int16 の索引は fp16 の入力をプログラムの中で int16 に変換する `idx16`。Core ML のモデルの記述に int16 の入力がないため): `gnd16_const`、`gnd16_mm`、`gnd16_pts`、`gnd16_pts_in`、`gnd16_pts_w1`、`gnd16_pts_4d`、`gnd16_pts_hw`、`gaa16_const`、`gaa16_input`、`dslice16_const`、`dslice16_mm` (この 2 つは Core AI の段でプロセスが落ちる)、`gnd_const`、`gnd_input`、`gnd_mm`、`gaa_input`、`gaa_const`、`dslice_const`、`dslice_input`、`dslice_mm`、`mm_dyn`、`mm_dyn_big`、`mm_dyn_i8`、`mm_dyn_i8_big`、`lin_const_big`、`lin_i8_big`、`argmax`。

`plan_darwin.go` の文言は実験用。`aicheck run` の int8 の入力の値はでたらめ。

## 3. 確かめること

M2 と M3 は済んだ。残りは M4 と、ANE のコンパイラの返事を読むこと。測る前に、ほかの hwnpu のプロセスが Neural Engine を使っていないことを確かめる (`ps -ax | grep -E "hwnpu|aicheck"`)。aicheck は NPU のロックを取るので順番待ちになるが、時間はずれる。

### 3.1 M4 での配置

M1 は gather が載らず、M2 と M3 は int16 の索引で載る。M4 (手元にあれば) で同じ probe を回す。

```sh
go build -o /tmp/aicheck ./tools/coreai/aicheck
sysctl -n machdep.cpu.brand_string
rm -rf ~/Library/Caches/coreai-cache/*/aicheck     # aicheck の probe のキャッシュだけ
/tmp/aicheck probe linear gnd_const gnd_input gnd_mm gaa_input gaa_const dslice_const dslice_input dslice_mm mm_dyn mm_dyn_big mm_dyn_i8 mm_dyn_i8_big
/tmp/aicheck probe gnd16_const gnd16_mm gnd16_pts gnd16_pts_w1 gnd16_pts_4d gnd16_pts_hw gaa16_const gaa16_input
COREML=1 /tmp/aicheck probe linear gnd_const gnd_input gnd_mm gaa_input gaa_const dslice_const dslice_input argmax
```

`ane=true` なら全部 Neural Engine。**とくに見たいのは `gnd16_pts_w1` と `gnd16_pts_4d`** (M1 では「Cannot decompose layer on this architecture」)。新しい世代でここが通るかと、通ったときの次の返事 (3.3 の `log stream` で)。Core ML の行は、演算ごとに優先される装置と、ANE が対応するか (「(the Neural Engine supports it)」) を出す。**新しい世代で gather_nd か dslice が ANE に載れば、MoE の形はずっと簡単になる** (表は定数のまま、専門家の番号だけ入力)。

### 3.2 M4 での MoE の 1 層 (あれば)

M2 と M3 は済み (1b、1c)。M4 では `aicheck moe -model gptoss -layers 4 -calls 200 -copy`、`aicheck moe -model qwen3moe -layers 1 -calls 100 -topk 6` (段差が無いか) と、LFM2-8B-A1B の hwnpu-chat (`-m LiquidAI/LFM2-8B-A1B`) の生成とプロンプトの速さ。

### 3.3 ANE のコンパイラが gather を外す理由 (sudo)

ログの理由は `<private>` で伏せられる。macOS 27 では `sudo log config --mode "private_data:on"` は「Invalid Modes 'private_data:on'」で効かない。構成プロファイル [tools/coreai/logging/private-data.mobileconfig](../tools/coreai/logging/private-data.mobileconfig) (`com.apple.system.logging` の `System` → `Enable-Private-Data`) を入れる:

```sh
open tools/coreai/logging/private-data.mobileconfig
# システム設定 → 一般 → デバイス管理 (または プライバシーとセキュリティ → プロファイル) で
# 「Unified log private data (hwnpu)」をインストール (管理者のパスワード)
/usr/bin/log show --last 1m --info | grep -c "<private>"    # 減ったか確かめる
```

`log` は zsh の組み込みに化けるので `/usr/bin/log` と書く。

検証の返事は debug の段で、`log show` には残らない。probe の間 `log stream` で拾う:

```sh
rm -rf ~/Library/Caches/coreai-cache/*/aicheck
/usr/bin/log stream --level debug --style compact --predicate 'process == "aicheck"' > /tmp/anec_stream.log &
sleep 2; /tmp/aicheck probe gnd16_pts_w1 gnd16_pts_4d gnd16_const; sleep 2; kill %1
grep -B2 -A12 "outValidationParams" /tmp/anec_stream.log      # 演算ごとの support と errorMsg
grep errorMsg /tmp/anec_stream.log | grep -v '""' | sort | uniq -c
```

終わったら戻す:

システム設定の同じ画面でプロファイルを削除する。

### 3.4 残っている小さなこと

- `anec.mx_dequant` (MXFP4) が新しい世代で使えれば、gpt-oss の専門家を 4 bit のまま入力にできるかもしれない (M3 は MX の定数も入力も Neural Engine に載らなかった。1b の最後)
- 4 ビットを int8 に 2 つ詰めて `floor_div` で取り出す形が Neural Engine で NaN になる所 (`HWNPU_DEV=ailower.cut` で切り分ける。速さで負けるので急がない)
- 結び付けの費用が層の数で増える理由 (Core AI の覚えている配列の上限か、16 GB の機械のメモリ圧か)。24 GB 以上の機械で比べたい

## 4. 注意

- 幅の広い one-hot (128 × 16000 など) は M1 の Neural Engine を全プロセスで止めたことがある。probe の軸は 16384 以下に保つ
- 調べた文字列のダンプは 1.5 GB になる (`strings -a` を共有キャッシュの全部に)。ディスクの空きに注意
- `MPSGRAPH_PRINT_ANE_PLACEMENT_ANALYSIS`、`MPSGRAPH_DUMP_MODULE`、`MPSGRAPH_FORCE_DEVICE_PLACEMENT` などの環境変数はあるが、M1 では何も出なかった (社内向けの設定が要るのかもしれない)
