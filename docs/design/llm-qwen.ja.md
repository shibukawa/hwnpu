# Qwen のモデル

[English](llm-qwen.md)

密な Qwen のチャットのモデル: Qwen3 (0.6B、1.7B)、Qwen3.5 (0.8B、2B、4B。Gated DeltaNet)、
Qwen2.5-Coder (Qwen2)。MoE の Qwen3.6-35B-A3B は [moe.md](moe.md)。下書き (プロンプトからの推測、
Qwen3.5 の MTP の層) は [llm-decoding.md](llm-decoding.ja.md)。KV キャッシュ、注意で分ける形、段は
[llm-engine.md](llm-engine.ja.md)。AMD の書き出しは [amd-llm.md](amd-llm.ja.md)。

## モデル

| モデル (models.yaml) | アーキテクチャ | Apple の形式 | AMD | machine | メモリ (Apple) | 備考 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (`qwen3:0.6b`) | `qwen3` | lut8 | 書き出しなし | 8 | 1.6 GB (M1 1.1) | ダウンロードが最小 (1.5 GB)、事実に弱い |
| Qwen3-1.7B (`qwen3`、`qwen3:1.7b`) | `qwen3` | lut8 | 書き出しなし | 8 | 2.1 GB (16 GB 未満)、2.3 GB (16 GB 以上) | 16 GB 未満は `prefill: off` |
| Qwen3.5-0.8B (`qwen3.5:0.8b`) | `qwen35` | lut8 | `--q4 --mlp q6 --head q6 --verify 3` | 8 | 1.5 GB (8 GB では `tier: none`。16 GB 以上の M2 以降は 2.3 GB) | `no_think` |
| Qwen3.5-2B (8 GB では `qwen3.5`) | `qwen35` | lut6 | 0.8B と同じ | 8 | 2.2 GB (AMD 2.4) | `no_think`。コードも書く |
| Qwen3.5-4B (`qwen3.5:4b`) | `qwen35` | lut8 | `--q4` | 24 | 4.2 GB | 大きさのわりにコードに最も強い |
| Qwen2.5-Coder-7B-Instruct | `qwen2` | lut4 + GPTQ の codes (`codes: faace07a`) | 書き出しなし | 16、`not_on: [M1]` | 4.4 GB | 考えない、文脈 32768 |

- Mac の `hwnpu-chat` は Core AI だけで動く。Core AI ではどのモデルも全部の部品が Neural Engine に
  載る。生成のプログラムとプロンプトを読むプログラムは 1 つの `.aimodel` の 2 つの関数なので、重みは
  1 回分だけ持つ ([Core AI](apple-coreai.ja.md))
- Intel: IR は作れる (`hwnpu-convert -backend intel`。アーキテクチャの lut8 は int8 になる)。NPU で
  は測っていない
- 事実: ここの小さな Qwen はどれも日本語の事実 (富士山の標高、『こころ』の作者、日本一長い川) を誤る。
  HF の fp32 でも同じように誤るので、移植や量子化ではなくモデルの限界。MODEL.md のチャットのおすすめ
  には入れない

## Qwen3 (`internal/llm/qwen3`)

構造: RMS 正規化、q と k のヘッドごとの正規化の後に RoPE (θ = 1e6)、GQA (16 / 8 ヘッド、head_dim 128)、
SwiGLU。出力の行列は埋め込みと共有 (`tie_word_embeddings`。チェックポイントの `lm_head.weight` は写し
なので読まない)。0.6B と 1.7B は同じコード (1.7B: hidden 2048、28 層、MLP 6144)。

- 断る config: `rope_scaling`、窓、`attention_bias`、`full_attention` 以外の `layer_types`。
  transformers 5 の `rope_parameters.rope_theta` も読む
- fp16: 残差は最大 8141 (0.6B、層 2 から)、14573 (1.7B) になる。正規化は Gemma 4 と同じく入力を 1/64
  にしてから二乗する。**ただし層 0 の入口の正規化は縮めない**: 入力が埋め込み (最大 0.1 程度) で、
  1/64 にすると二乗が fp16 の下に落ちて NaN になる
- 28 層すべてが注意。注意で分けるため、最初の部分は層を持たない (`llm.NoLayers`): 層 0 の正規化と射影
  だけで、`attn_q_0` と K / V を出す
- 分けるかどうか (`llm.Unpaged`、`pagedChunk`):
  - Core AI: 既定で注意で分け、短い塊を使う。文脈 40960 をメモリの 4 分の 1 まで (16 GB で 32768)。
    M2 以降では短い会話用の分けない 1 本を前に置く (`--tier`、Qwen3-0.6B は 1024。
    [llm-engine.md](llm-engine.ja.md))。M3、Qwen3-0.6B lut8: 分けて 36.1 トークン/秒 (分けない 1 本
    19.1)、2149 トークンのプロンプト 1.7 秒 (3.6 秒)
  - ほかのバックエンド: キャッシュを中に持つ分けない 1 本 (`UnpagedContext` 4096)。`--paged N` を
    与えたときだけ分ける (`--paged 1024` で 32768)
  - 落とし穴: 分けない形の 8192 と 16320 のキャッシュは、コンパイルと読み込みは通るが Core ML の予測が
    エラーになる (K/V の入力 56 本、0.94 / 1.9 GB)
- チャット (`chat.Qwen3`、`internal/chat/qwen3.go`): ChatML、Hermes 式の道具 (`<tools>` の JSON、
  `<tool_call>{"name": …, "arguments": {…}}</tool_call>`)、思考は最後の質問より後の回答だけに残す。
  引数は Python の `json.dumps` と同じ (`1e20` → `1e+20`)。`Render` は `<think>\n` まで書き、考えて
  いる途中から流す (Qwen3.5 と同じ)。jinja2 の 9 会話と一字一句一致 (`template_golden.py
  --text-content`: 部品の列の content を先に文字列にする。vLLM も同じ)
- 照合: fp64 の正解と 1e-4 (共有あり・なし)、1 層ずつ、キャッシュつき、分けた形。実物の重み
  (`hwnpu-dev check`、32 トークン):

  | | fp32 / fp16 | lut8 | lut6 |
  |---|---|---|---|
  | 0.6B | KL 0.0000、32/32 | KL 0.0139、31/32 | 0.2425、23/32 |
  | 1.7B | KL 0.0000、32/32 | KL 0.0181、32/32 | 0.3148、25/32 |

- Qwen3-1.7B は MOSS-SoundEffect v2.0 の `text_encoder/` とバイト単位で同じ。models.yaml にリビジョン
  と 2 つのファイルの `sha256` を書き、`fetch` は同じ SHA-256 のファイルをほかのモデルのディレクトリ
  から探してハードリンク (できなければ写し) にし、確かめる (`sameFile`、`TestSameFile`)。
  [distribution.md](distribution.ja.md)
- 16 GB 未満の Qwen3-1.7B は 1 本 (`prefill: off`)。生成のプログラムが 1 回 8 トークンずつプロンプトを
  読む。`--prefill on` でプロンプトの関数を足す (Core AI で +0.3 GB)
- 速さ (Core AI、M1): Qwen3-0.6B 32 トークン/秒、Qwen3-1.7B 22 トークン/秒

## Qwen2 / Qwen2.5-Coder (`qwen2`、`internal/llm/qwen3` の中)

Qwen3 の層から q と k のヘッドごとの正規化を除き、q、k、v に bias を足したもの。config に `head_dim`
はない (hidden / heads = 128)。`Qwen2Arch`、`ParseQwen2Config`。分ける形、キャッシュ、`LastRow` は
Qwen3 のもの。

- チャット: `chat.Qwen25`。Qwen3 の考えない版に、system がないときの Qwen2.5 の決まった一文。
  `apply_chat_template` の 9 件 (道具、呼び出し、応答) と一字一句一致
- 照合: 合成の Qwen2 (bias、正規化なし、head_dim なし、共有あり・なし) が fp64 と 1.6e-5。bias を外すと
  0.70 で落ちる。実物の 7B の層 0〜2 と出力層を、1 層ずつ float32 で回す参照 (`tools/ref/ref_qwen2.py`。
  7B の float32 は 30 GB で 16 GB に入らない) と比べて、層ごと 7e-5〜6.6e-4、ロジット 4e-7
- Neural Engine の軸の上限 16384 を超える MLP (7B: 18944) は 32 列の倍数で分ける: gate と up は行、
  down は列 (`blocks.Graph.MatrixSlice`。表と索引は行列全体のものなので、値も codes も変わらない)。
  1 本のままだと部品が全部 Neural Engine から外れた (0.1 トークン/秒)。分けた MLP の部品は 2.94 ms
  (125 MiB、44 GB/s)。`Qwen2Arch.Revision` 1。14B の 13824 は分けない

### 重みの形式

アーキテクチャの既定は lut8。models.yaml の 7B は `format: lut4` と GPTQ の codes。CPU で全 28 層を
つなぎ、64 トークンの会話で float32 の参照のロジットとの KL (`TestRealChained`):

| 形式 | KL の平均 | 中央値 | 最上位の一致 | 7B の重み |
|---|---|---|---|---|
| lut4 一様 | 0.90 | 0.41 | 36/64 | 3.5 GB |
| lut4 k-means | 0.47 | 0.20 | 45/64 | 3.5 GB |
| lut4 k-means を入力の E[x²] で重みづけ (`palette.QuantizeWeighted`) | 0.23 | 0.078 | 51/64 | 3.5 GB |
| **lut4 行全体の表 + GPTQ (既定。Neural Engine で使える)** | **0.131** | **0.051** | **54/64** | 3.5 GB |
| 同じく `--format-for down_proj=lut8` | 0.078 | 0.035 | 57/64 | +0.95 GB |
| 同じく注意だけ lut8 | 0.105 | 0.043 | 55/64 | +0.44 GB |
| lut4、32 列ごとの表 + GPTQ (NE では誤る。下) | 0.060 | 0.038 | 62/64 | 4.5 ビット |
| lut6 k-means | 0.18 | 0.078 | 50/64 | 5.3 GB |
| lut8 | 0.0012 | 0.0008 | 63/64 | 7.1 GB |

- 校正: リポジトリの Go、Python、README の英語と日本語を 512 トークンずつ (評価の会話とは別)。入力の
  E[x²] は `ref_qwen2.py --stats`
- GPTQ (`tools/ref/qwen2_gptq.py`): 1 層ずつ、校正の文を量子化した層に通し、層の中は q/k/v → o →
  gate/up → down の順。表は先に重みつきの Lloyd 法で固定し、GPTQ は各重みの表の位置だけを選ぶ。down の
  H (18944²) は float32 で 1.4 GB。M1 で 7B 全体 77 分
- codes は `palette.Codes` (`index.json`、`.idx`、`.tab`。`palette.CodesWriter`) に書く。
  `hwnpu-chat --codes DIR` で変換に使う。codes のハッシュがプログラムの名前に入るので、変換済みのもの
  は取得元から取れる。変換するには `models/NAME/codes-HASH` か `--codes` が要る
- **Neural Engine は列のブロックごとの表とスケールを黙って誤る** (Core AI、macOS 27.2、M1): どの列にも
  最初の列のブロックの表を使う。int4-g32 / g128、int8-g32 も誤る (コサイン 0.966〜0.984)。行ごとの
  int4 と行全体の lut4 は正確。Core AI の CPU では正しいので、NE へのコンパイルの問題。Core AI の
  プログラムでは列のブロックを変換で断る (`blocks.Graph.rowsOnly`)。Core ML (int8-gN を CPU に置く) と
  Intel はそのまま。[apple-coreai.md](apple-coreai.ja.md)、[quantization.md](quantization.ja.md)

### 実機 (M1 16 GB、Core AI、`qwen2-r1-lut4-gptq-faace07a`)

- 変換 2 分 35 秒 (29 プログラム)、初回の読み込み 1 分 15 秒、2 回目から 0.5 秒。全部 Neural Engine。
  答えは正しい (繰り返しで書いた Go のフィボナッチ、goroutine を 2 文で)
- 0.8〜0.9 トークン/秒: NE の 4.0 GB が M1 の全速で持てる 2.8 GiB を超え、呼び出しのたびに載せ直す。
  それで `not_on: [M1]`。全部を持てる機械では 1 トークン約 0.1 秒の見込み (測っていない)。M2 は
  3.65 GiB まで全速 (Core ML で測った値)
- メモリ: NE 4.0 GB、プロセス 0.35 GB。初回の変換中は 6.3 GB

## Qwen3.5 (`internal/llm/qwen35`)

`modeling_qwen3_5.py` (transformers 5.22) から移した。チェックポイントは画像とテキストのモデル
(`Qwen3_5ForConditionalGeneration`、テンソルは `model.language_model.`)。テキストだけのもの
(`qwen3_5_text`、`model.`) も読む。視覚の塔 (`model.visual.`) は変換しない。分割された safetensors
(`model.safetensors.index.json`) を読む (`llm.OpenWeights`)。

### 構造

- 3 層の Gated DeltaNet (GDN) に 1 層の全注意の繰り返し。4B: 32 層 (GDN 24、注意 8)、hidden 2560。
  0.8B / 2B: 24 層 (注意は 3、7、…、23)。語彙 248320、出力は埋め込みと共有
- RMS 正規化は 0 中心 (`1 + w`)。GDN の最後の正規化 (`Qwen3_5RMSNormGated`) だけは `w` をそのまま
  掛け、`silu(z)` でゲートする。`1 + w` は fp16 に 1 回だけ丸める (w を先に丸めると 3% ほどの要素が
  ずれる)
- 全注意: `q_proj` が各ヘッドの問い合わせと出力ゲートを出す (256 + 256)、QK 正規化、先頭 64 次元だけの
  RoPE (`rotate_half` はその中で)、GQA (4B: 16 ヘッド、KV 4)、出力にチャネルごとに `sigmoid(gate)`。
  テキストでは mrope の 3 つの位置が同じなので普通の RoPE
- GDN: `in_proj_qkv` → 幅 4 の深さ方向の因果的な畳み込み + silu。q と k は L2 正規化 (q はさらに
  1/√128。L2 正規化は eps 1e-6/n の `RMSNormWith` に 1/√n)。鍵ヘッドを値ヘッドで共有
  (`repeat_interleave`。4B 16 → 32、0.8B / 2B 16 = 16)。減衰 `exp(-exp(A_log) · softplus(a + dt_bias))`、
  書き込みの強さ `sigmoid(b)`、値ヘッドごとの状態 S (128 × 128): `S ← S·decay; S ← S + kᵀ((v − kS)·β);
  o = qS`
- MTP の層 (`mtp.`): 下書きに使う。[llm-decoding.md](llm-decoding.ja.md)

### 状態

- **丸ごと置き換える状態 `st_`** (`llm.Cached`): GDN の状態は位置ごとに書けない。プログラムは
  `st_rec_N` (1, ヘッド, 128, 128) を取り `st_rec_N_new` を出し、ホストが入力をそれで置き換える。
  畳み込みの窓は LFM3 と同じ `win_conv_N` (直前の 3 行)
- 詰め物は状態を変えてはいけない: `StepInputs` が `step_valid` (トークン 1、詰め物 0) を渡し、詰め物では
  β と g を 0 にする
- `Mark`、`Rewind`、`Resume`、`Migrate`、`chain.Share` は `st_` を `win_` と同じに扱う
  (`llm.CacheInput`、`llm.HeldInput`)
- ホストは状態を fp16 で持つ。3276 トークンで損失は育たない (位置 2048〜4096 で 3.356 / 3.310。
  PyTorch の fp32 と比べて)
- 下書きを途中まで採るとき: `state-rows` がホストで残す行を進める。なければ `Speculate` が回し直す。
  [llm-decoding.md](llm-decoding.ja.md)

### Gated delta rule を塊の形で (`delta=chunk`、既定)

1 行ずつ (`delta=rows`) では、状態 (4B: 32 × 128 × 128 の fp16 で 1 層 1 MiB) を 1 行に 5 回読み書き
する: 24 層で 1 ステップ約 1.3 GB、重み 4.25 GB の 3 割。`deltaRuleChunks` (flash-linear-attention の
`chunk_gated_delta_rule`、WY 表現) は 8 行の塊で、状態を `k S0` と `q S0` で読み、1 回書く。
G_t = g_1 + … + g_t (減衰の対数)、Γ_t = exp(G_t) として:

- (I + N) U = diag(β)(V − diag(Γ) K S0)、N_tj = β_t exp(G_t − G_j) k_t·k_j (j < t)
- O = diag(Γ) Q S0 + (Q Kᵀ ⊙ exp(G_t − G_j)[j ≤ t]) U
- 対角より上は min(·, 0) の exp を取ってからマスクで消すので溢れない。詰め物の行 (β 0、g 0) は状態を
  変えない

Neural Engine の落とし穴:
- (8 × 8)·(8 × 8) の行列積があると部品全体が NE から外れる (内側 128 の k·kᵀ と (8 × 8)·(8 × 128) は
  載る)。そこで逆行列を作らず、N が冪零なことを使い U = B − N(B − N(B − …)) を (8 × 8)·(8 × 128) の
  積 7 回で解く
- 累積和 G と差 G_t − G_j は、g の行への定数の `linear`
- 小さな試しのプログラム (45 演算) は形によらず CPU に置かれるので、配置は実物の大きさのプログラムの
  中で調べる
- NE は内側の次元が 1 の行列積を誤る (M3)。1 行の塊 (Q=1) は `mul` で書く (`Revision` 2)

結果 (M2、Core AI、4B): GDN の部品 9.51 → 8.23 ms (約 56 GB/s で天井 62 GB/s に近い)、その初回の
コンパイル 15.8 → 7.0 秒、モデルの 1 ステップ (Q=8、9 部品) 96 → 83〜89 ms、128 トークンのプロンプト
511 → 304〜344 ms。部品ごとの誤差は同じ。貪欲法の文は僅差のところで変わることがある (足し算の順)

### Neural Engine の数値

ANE の `silu` と `softplus` が長いプロンプトを壊していた (lut8 の位置ごとの KL が日本語 0.33、英文
0.76。Core ML の CPU では 0.003 前後)。今の `builder` の手当て:

| 問題 | 手当て |
|---|---|
| ANE の `softplus` がある入力で 0 を返す (減衰が 0.003〜0.13 のところで 1.0) | `relu(x) + log(1 + exp(−|x|))` |
| ANE の `silu` が 1.4% ずれ、小さな裾ほどずれる。鍵の L2 正規化がそれを拡大する (大きさ 0.001〜0.01 の行で 30%) | `x / (1 + exp(−x))` (x < −11 では 0)。ゲートと MLP も |
| o = qS の項が 1e-5 前後で fp16 の正規の最小値 (6e-5) を下回る: o が 6%、ゲートつきの正規化で 17% ずれる | q を `oScale` (1024) 倍、正規化の前に別の `mul` で 1/`oNormIn` (64)、eps を (1024/64)² 倍 |

- 割り算は正規化の `inScale` にしない: 重みを 1/64 にして fp16 の非正規数にしてしまう
- `TestNoNeuralEngineActivations` がプログラムに `silu` と `softplus` がないことを見る。
  `internal/milcpu` と `internal/ovlower` に `abs` と `log` を足した
- 直した後 (M3、lut8): 位置ごとの KL 0.0073 (日本語、27/27)、0.0052 (英文、29/31)。3276 トークンの
  損失 3.362 (PyTorch 3.354)。合言葉 3,276、12,895、59,797 トークンで 3/3
- ほかのモデルも `blocks.SiLU` を使う ([apple-neural-engine.ja.md](apple-neural-engine.ja.md))

### 形式

| 大きさ | 既定 | 理由 |
|---|---|---|
| 0.8B | lut8 | lut6 は fp16 との次のトークンの一致 242/283 (lut8 274)、17 × 23 を誤る |
| 2B | lut6 (`Config.Format`、hidden 2048) | lut8 との一致 260/283、perplexity 3.75 対 3.83、12 問の正解が同じ 10。1.32 GiB 対 1.78 |
| 4B | lut8 | lut6 は KL 0.031〜0.135、26〜30 トークン中 21〜24。答えが崩れる |

- 実物の重みと HF / PyTorch の fp32 (`hwnpu-dev check`): 0.8B の fp32 で各層 0.0002 以下、argmax 32/32、
  KL 0.0000。lut8 は KL 0.0076。4B の 32 層全部 (層を 1 つずつ読む参照 `tools/ref/ref_qwen35_stream.py`。
  活性は最大 121 で fp16 はあふれない): fp16 KL 0.0000〜0.0001、lut8 0.0015〜0.0050
- 2B の fp16 は作れない: 最後の層と出力層で 1.04 GiB になり、1 部分の上限を超える
- lut4 は 0.8B と 2B で同じ句を繰り返すか数を誤る
- MTP の層は `Format` を受け継ぐ (2B では lut6)
- **2B の lut6 の書き出しは回す** (`-rot`、[量子化](quantization.ja.md)): 重みに直交行列 Q を通し、ホストの表は `tables-rot.safetensors` (E Q を fp16 で、1.02 GB。取得元の鍵は `qwen35-rot`)。MTP の層は回さず、その 2 つの入力 (トークンの埋め込みと最後の隠れ状態) をホストで元に戻し (`mtpDrafter.plain`、`llm.Unrotate`)、最後の norm の 1 + w を掛ける。M2、2B lut6、温度 0、MTP ありで 200 トークン: 45.1〜45.4 → **51.4〜52.4 トークン/秒**、通った下書き 95/220 → 108/215。12 問 9 → 10。NE での KL 0.054 → 0.029。変更後の初回は書き出し直しとコンパイルがある (2B で約 5 分)
- 0.8B と 4B (lut8) は回さない: 0.8B は 61.6〜62.1 → 63.6〜64.5 トークン/秒、4B は 22.8 → 22.9〜23.1 で、12 問の答えは同じ
- AMD: 0.8B / 2B は q4 に MLP と出力層を q6 (4 ビットでは長い英文で崩れる)。[amd-llm.md](amd-llm.ja.md)

### 注意で分ける形、文脈、最後の行

- 全注意の層の前後で分ける (`llm.Paged`)。注意はホストが塊ごとに重みなしのプログラムで読む。分け目の
  前の部分は `attn_q_N`、`kv_k_N_new`、`kv_v_N_new` と出力ゲート `attn_gate_N` (1, seq, 16, 256。
  チャネルごと。LFM3 はヘッドごと) を出す。GDN の状態と畳み込みの窓は各部分の入力のまま。4B は 9 部分、
  0.8B / 2B は 7
- 文脈 262144 (`max_position_embeddings`)。KV は 1 トークン 4B 32 KiB (262144 で 8 GiB。16 GB では
  114688)、0.8B / 2B 14 KiB (3.38 GiB)
- `LastRow`: プロンプトを読むプログラムは最後の行だけで出力層を計算する (最後の正規化の前で `last` の
  one-hot を掛ける)。しないと 128 × 248320 の出力が 60.6 MiB になる

### トークナイザーとチャット

- tokenizer.json は NFC と、GPT-4 のものと違う分割の規則 (`[\p{L}\p{M}]+`、数字は 1 つずつ、`\p{M}` は
  記号から外す) を使う。`golang.org/x/text/unicode/norm` を依存に持つ。ID は HF の tokenizers と一致
  (`TestQwen35`、`TestSplitQwen35`)。0.8B / 2B / 4B のトークナイザーは同じ
- `chat.Qwen35` は chat_template.jinja を移したもの: 道具の宣言は Qwen 自身の説明つきで system に、思考は
  最後の質問より後の回答だけに残す。引数はオブジェクトと配列は JSON、ほかは Python の `str()` (True、
  None)。描いた 9 会話と一致。回答の分解は `chat.Parse` (LFM3 と同じ XML)
- 思考: 0.8B と 2B は `enable_thinking` が true のときだけ考える (テンプレートの既定) ので、models.yaml
  に `no_think: true`。考えさせると 0.8B は 4 問とも予算 1024 に届くまで同じ言葉を繰り返した。
  [llm-decoding.md](llm-decoding.ja.md)

### 測った値

| | 機械 | 生成 | プロンプト | Neural Engine | 起動 |
|---|---|---|---|---|---|
| 4B lut8 + MTP | M2 16 GB、Core AI | 20.5〜21.9 トークン/秒 | 2041 トークン 4.0 秒 | 4.05 GiB | 初回 14 分 39 秒 (コンパイル 13 分)、2 回目から 0.7〜0.8 秒 |
| 2B lut6 | M1、Core AI | 20.0〜21.0 トークン/秒 | | 1.67 GiB + プロセス 0.39 GiB | 初回 1 分 20 秒、2 回目から 1.6 秒 |
| 0.8B lut8 | M1、Core AI | 25.3〜27.0 トークン/秒 | | 0.99 GiB + プロセス 0.39 GiB | 初回 5 分 9 秒、2 回目から 2.3 秒 |

- M1 の 4B: lut8 の 3.96 GiB が M1 の NE の全速で持てる 2.86 GiB を超え、約 1 トークン/秒。M1 では 2B
- 初回の変換でプロセスは 2B で 6.0 GiB、0.8B で 3.0 GiB になる。8 GB の機械では変換済みのプログラムを
  取ってくる前提 ([distribution.md](distribution.ja.md))
- Core ML は Qwen3.5 の 128 トークンの GDN の部品のコンパイルがとても遅かった (0.8B で 15〜26 分。
  約 3.5〜4 GiB のキャッシュから追い出されると毎回)。Core AI は残る (2 回目 0.7 秒)

## 採らなかったもの

- Qwen3.5-0.8B / 4B の lut6、Qwen3.5 の lut4: 精度 (「形式」)
- Qwen3 の lut6: KL 0.24 / 0.31
- Core AI での列のブロックごとの表やスケール: NE が最初のブロックの表を使う (答えが壊れ、7B の部品で
  コサイン 0.56〜0.84)。代わりに行全体の表 + GPTQ
- GDN のゲートつきの正規化を行の最大値で割ってから二乗する形: ANE でかえって悪い (17% → 40%)。原因は
  o の大きさだった
- (I + N) の逆行列を (I − N)(I + N²)(I + N⁴) で作る: 8 × 8 の積が NE から外れる
- Qwen3.5 の Q=1 と Q=64 の組: コンパイルが 24 分終わらなかった (Core ML)
- Anima のテキストエンコーダを Qwen3-0.6B と共有: Qwen3-0.6B-Base で別のチェックポイント
- Sana の Gemma 2 のチャットの型 `gemma2`: 同じ大きさの Gemma 4 E2B がある
- M1 の Qwen2.5-Coder: 0.8 トークン/秒 (`not_on: [M1]`)

## 未解決

- 16 GB の機械の Qwen3.5-4B: models.yaml は `machine: 24`。上の M3 / M2 の値はモデル 1 つだけのとき
- M2 以降の Qwen2.5-Coder-7B (速さ、4.0 GB が全速か)。14B (4 ビットで 7.0 GB、チェックポイント 29.5 GB、
  変換していない)。別の評価の文での KL。長い会話とエージェントの課題。AMD の書き出し (q4nx の 32 列
  ごとのスケールなら NE の表より細かい)
- Qwen3 / Qwen2 の AMD (書き出しなし) と Intel の NPU (測っていない)
- Qwen3.5 の Intel の NPU。2B の lut6 と HF の bf16 との KL。Qwen3.5 の GPTQ の lut6。0.8B / 2B の長い
  文脈の合言葉とエージェントの課題
- M1 での Qwen3.5-4B の lut4 (全速に収まり 10.9 トークン/秒) には KL と合言葉の照合が要る
