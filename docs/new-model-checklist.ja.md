# 新しいモデルを入れるときの検証の手順

[English](new-model-checklist.md)

作成: 2026-11-07。LFM3 (VL / Audio / Omni、d1)、Gemma 4 E2B / E4B、Qwen3.5-4B、PLaMo 3 Translate、PLaMo-Embedding、Anima を入れてきた間に、後になってから見つかった問題と、効いた手法を、**最初に確かめる順**に並べたもの。始めからこの順に確かめれば、作り直さずに、少ないメモリで速い形から始められる。

数値の出典は [design.md](design.md) の 16 章 (カッコ内の番号)、[improvements.md](improvements.md)、[model-fit.md](model-fit.md)、[amd-npu-models.md](amd-npu-models.md)。構造を Go に移す作業そのものの手順と合格条件は `.claude/skills/port-architecture` にあるので、ここでは繰り返さない。

| 段 | 何を決めるか | どこで |
|---|---|---|
| 1 | 机上で入るか、速いか (選ぶかどうか) | どこでも |
| 2 | 構造が正しいか (fp64、HF との照合、トークナイザー、チャットの型) | CPU |
| 3 | Mac の Neural Engine に収まるか、全部が NPU に載るか | Mac |
| 4 | AMD / Intel の形 | Linux / Windows |
| 5 | **量子化の形式と校正の方法** | CPU で見積もり、NPU で確認 |
| 6 | メモリ (常駐、固定、ディスク) | 各 NPU |
| 7 | 速さ (生成、プロンプト、起動) | 各 NPU |
| 8 | 長い文脈 | Mac (Intel) |
| 9 | 用途での確認と記録 | 各 NPU |

各項目の「合格」は目安で、満たさないときは理由を design.md に書いてから先に進む。

## 1. 机上の見積もり (移植する前に)

[model-fit.md](model-fit.md) の 2 節と 4 節の式と基準で、config.json だけから見積もる。

- **1 トークンの時間 (ANE)** ≈ 重みのバイト ÷ 65 GB/s + 長い容量に対して注意を計算する層の数 × 0.22 ms (`SDPAGrouped` の形) × 容量 / 1K + 3 ms
  - KV の共有は注意の計算の回数を減らさない (メモリだけ)。窓、短い畳み込み、Mamba2、GDN の層は容量に依らない
- **重み**: lut4 0.5 B、lut6 0.75 B、lut8 1 B / 重み。M3 (16 GB) で常駐は 3 GiB までが目安。AMD は 4 ビット (q4nx 5 ビット + 注意 q8) で bf16 の 1/2〜1/5 の固定
- **語彙**: 出力層は語彙 × 隠れ。262144 の軸は ANE で CPU に落ちるので (16, 16384) に畳む手間が要る (Gemma 4、16.25)
- **使えない構造**: MoE のルーティング (`gather`)、学習済みの疎な注意、可変の形。AMD の MoE は部品を測っただけ (16.75)
- **ライセンスと言語** (hwnpu-chat の目録に載せるか)

**合格**: 1K で 50 ms 以下、16K で 100 ms 以下の見込み。M3 で常駐 3 GiB 以内に入る形式がある。

## 2. 構造の正しさ (CPU)

`port-architecture` スキルの合格条件 (fp64 の参照と 1e-4、1 層ずつとまとめての一致、壊して検出、`hwnpu-check` の層ごとの照合) に加え、今までに後から見つかったもの:

| 確かめること | 道具 | 教訓 |
|---|---|---|
| **BOS を付けて**比べる | `tools/ref/ref_logits.py`、`hwnpu-check`、`ironlm -forced -bos` | LFM3 は BOS なしだと KL が 4 倍に見え、0 層目の活性が外れ値になる (amd-npu-models.md 13 節)。チャットの型で実際に付くトークンで測る |
| 実物の重みで照合する | `hwnpu-check`、`iron_plamo3_check.py` | 乱数の重みでは NPU と CPU の参照が食い違うことがある (Gemma 4、PLaMo 3)。合否は実物の重みで |
| トークナイザーが HF と一致する | `tools/ref/tokenizer_golden.py` | 固定の文 + 数千の乱数の文で 0 件の食い違い。同点の扱い (PLaMo)、NFC と結合文字 (Qwen3.5)、byte fallback (Gemma 4) を逆にしたらテストが落ちることも確かめる |
| チャットの型が jinja2 と一字一句同じ | `tools/ref/template_golden.py` | 道具の呼び出し、考える部分の扱い、過去の考えの削除 |
| 多モーダルの前処理 | `resize_golden.py`、`mel_golden.py` | PIL との ±1、librosa との 1e-4 |
| 埋め込みモデル | `embed_golden.py` | HF との cosine 0.997 以上 |
| 画像生成 | `tools/ref/ref_anima.py` | diffusers の fp32 と潜在の cosine、VAE の PSNR |

## 3. Mac の Neural Engine に収まるか

**最初に確かめる**: Mac がないと何も決まらない項目。引き継ぎの文書 (`docs/handoff-*.md`) を書いて Mac で測る。

### 3.1 上限

| 上限 | 対処 | 節 |
|---|---|---|
| 1 パッケージ 1 GiB (1 部分の既定は 0.9 GiB) | 層で分ける (別のパッケージ。multifunction では越えられない) | 16.12–13 |
| 1 次元 16384 | 注意で分けた塊 (`-paged`)、ソフトキャップを (16, 16384) に畳む | 16.21, 16.25 |
| コンパイルのキャッシュ約 4 GiB (LRU) | 2 本 (Q=8 と Q=64) の合計が超えるなら 1 本 (`-prefill auto`)。段階 (`-tier`) は lut4 だけ | 16.41, 16.65, 16.71 |
| 表は 16 行ごと | 形式の選び方は 5 節 | |
| `gather` は CPU | 埋め込みの行はホストで引く (PLE も) | 16.25 |
| 注意のスコアは 1 テンソル 64 MiB 未満 | 超えると 2〜3 倍遅い。問い合わせを塊に (`mil.DitQChunk`) | 16.85 |

### 3.2 全部が NPU に載るか

`hwnpuinfo -model PART.mlpackage` で各部分が `all on the NPU` か。今までに CPU に落ちたもの:

- `tile` で K/V を写す注意の形、問い合わせを行列にした形 → `SDPAGrouped` (16.25, 16.39)。順番だけで配置が変わる
- 最後の部分の `transpose` / `reshape` → 出力をトークン先に取る (`attn_ot_N`、PLaMo 3)。Gemma 4 の最後の部分の `reshape` は未解決 (improvements.md 11.1)
- 1 行の部品、単独のプログラムの部品 → 8 行にする、隣の群に入れる (Anima、16.85)
- `reduce_sum` → 行列積で和を取る (塊の注意、30 回中 30 回 ANE)

### 3.3 fp16 であふれないか

`hwnpu-check -f16` で層ごとの相対 RMS が 0.01 未満か。効いた手当て:

- 正規化の入力を 1/64 にする (Gemma 4: KL 3.17 → 0.0004、PLaMo 3 も)
- 最後の層を head-scale で縮める (PLaMo-Embedding)、`-post-scale` (LFM3、1 層 1.4〜2.7 %)
- 残差を 1/32 で持ち、正規化は行ごとの最大で割ってから (Anima。固定の倍率だと非正規数になる)
- 詰め物の行を最後のトークンで埋める (Gemma 4 の eps が fp16 で 0 になって NaN)
- ANE の `softplus` と `silu` は不正確 (softplus がある入力で 0、silu は裾で 1.4 % ずれる)。exp と log で書いた `builder.softplus`、`builder.silu` を使う。fp16 の正規の最小値 (6e-5) を下回る積は ANE で桁を失うので、大きくしてから後で割る (Qwen3.5 の GDN の出力、16.88)。CPU の `hwnpu-check` ではこれは見えないので、ANE と Core ML の CPU (`HWNPU_COREML_UNITS=cpu`) で位置ごとの KL を比べる

### 3.4 合格

- 全部の部分が `all on the NPU`
- 2 回目の起動が数秒 (コンパイルのキャッシュに残る)。分けた形 (`-paged`) で Gemma 4 は 98 s → 1.9 s
- `hwnpu-check -package` の KL と argmax を記録 (形式ごと、5 節)

## 4. AMD (IRON) と Intel の形

### 4.1 AMD

| 確かめること | 教訓 | 節 |
|---|---|---|
| IRON に同じアプリがあるか。なければ IRON の部品から書く | LFM3 (`iron_lfm3.py`)、PLaMo 3 (`iron_plamo3.py`) は部品から書いた | 16.66, 16.70 |
| HF と top-1 で照合 (`iron_export_lm.py --check`) | 長いプロンプト (2000 トークン超) でも | 16.76 |
| full ELF で 1 本に | xclbin の切り替えは 1 回 0.46 ms | 16.45–50 |
| プロンプトの塊 1024 | 2048 より 20 % 速い。256 は窓の層がリングになり巻き戻せない | 16.54 |
| 1 コマンドを 2 s 未満に | 長いと `ERT_CMD_STATE_TIMEOUT`。書き出しで分ける (`SCRATCHPAD_WORDS`) | 16.65 |
| 窓の層は窓の中だけ読む (`iron_window.py`) | プロンプトが −12 %。塊の前に空のコマンド (`chunkBarrier`) がないと 2 つ目の塊で止まる。キャッシュの行数は `cache_rows` で書き出す | 16.84, 16.87 |
| KV を共有する層はプロンプトの最後の行だけ | Gemma 4 E2B のプロンプトが 2.3 倍速い | 16.80 |
| 表はホストに (PLE、埋め込み) | E2B 9.08 → 4.60 GB の固定。出力は同じ | 16.72–73 |
| 塔は使うときだけ読む (`growBanks`) | テキストだけの書き出しが要らない | 16.81 |
| `-idle` で固定を返す | 次の要求に 0.8〜2 s | 16.74 |
| コアの丸めを最近接偶数に | 床関数の丸めで平均のある和が下に偏る (Anima の DiT) | 16.78 |
| MHA の head_dim は 64 の倍数 | 96 は 128 に詰める (LFM3-VL の塔) | 16.65 |
| 画像と音声を最後の書き出しの形で通す | 塔を初めて使うときに重みのバッファを取り替えるので、古いバッファに結び付いた塊の前の空のコマンドが失敗していた (ENOENT、2026-11-07 に修正)。書き出しを変えたら `hwnpu-chat` で画像と音声を 1 回ずつ通す | 16.81, 16.87 |
| 状態のある層 (畳み込み) は塊の境目で写す | 印への巻き戻しとプロンプトの保存のため | 16.66 |
| 投機的デコードは**しない** | 下書きの確認が 4.5 ステップ分の費用 | |

### 4.2 Intel

- `ovlower` で IR に。既定は int8 (int4 は fp16 と次のトークンが 22/52 しか合わない)。今は OpenVINO の CPU で精度だけ確かめられる (実機の速さとメモリは未計測)

## 5. 量子化: 形式と校正の方法を選ぶ

**一番手間がかかり、一番効く段**。モデルによって 4 ビットの落ち方が 10 倍違う (AMD の q4 の KL: PLaMo 3 の訳文 0.009、LFM3 0.036、E4B 0.049、E2B 0.097)。次の順に決める。

### 5.1 測り方を揃える

- **KL(参照 ‖ 量子化)** の平均、中央値、95 %点、top-1 の一致。400 位置 (前の 256 トークンをプロンプトとして読ませ、その後を 1 つずつ)、**BOS あり**
- 文は 5 種: README (日英の混在)、GPL-3 (**校正に使わない**。合いすぎの検出用)、wiki40b-ja の評価分、wikitext-2 の試験分、Go のコード。置き場は `~/models/calib`
- 参照は fp32 の CPU (HF) か、同じ NPU の bf16。NPU の量子化の KL が CPU の見積もりと同じかを確かめる (LFM3 0.036 と 0.035、LFM3-VL も同じ傾向)。違えば NPU の計算 (丸め、bfp16) を疑う
- 画像生成は KL の代わりに、参照の軌跡の上での速度の相対 RMS と、自由に走らせた最後の潜在の cosine (`tools/amd/anima_gptq.py`)

### 5.2 部分ごとのビット数を CPU で掃く

部分 (注意の射影、MLP、出力の行列、畳み込み、塔) ごとに 4 / 8 / bf16 を切り替えて CPU で KL を測る (`q4_gptq.py --method rtn` の `--q8`、`--head`、`--fmt-*` で組み合わせる)。今までの結論:

- **注意の射影は 8 ビット**: 重みの 1 割余りなのに 4 ビットにすると誤差の大半 (E2B 0.097 → 0.229)。固定は +0.12 GB だけ
- MLP と出力の行列は 4 ビット。出力の行列は 4 ビットでもほとんど効かない
- 短い畳み込みは 8 ビット (4 ビットで +0.01)
- サンドイッチ正規化の層 (Gemma 4、PLaMo 3) は 4 ビットに弱い。ANE の表なら lut6 以上 (Gemma 4 の lut4 は KL 2.54、PLaMo 3 は 1.91)
- 多モーダルの文の部分、TTS、d1 の判定、埋め込みは lut8 (lut4 で崩れた)
- 蒸留した 8 ステップの DiT は 4 ビットに弱い (q4nx で潜在の cosine 0.777)。q8 が既定

### 5.3 校正の方法を順に試す

同じ形式のまま、どの符号 (段) を選ぶかで精度を上げる。形式が変わらないので速さとメモリは同じ。

| 順 | 方法 | 道具 | 結果の例 | 注意 |
|---|---|---|---|---|
| 1 | 丸め (RTN)。q4nx は 9 つの候補から最小の誤差を選ぶ | `iron_q4.quantize_fmt`、`palette.QuantizeBits` | 重みの誤差 8.2 → 7.3 % | 基準。重みの誤差が最小でも出力の誤差が最小とは限らない |
| 2 | **GPTQ** (H = E[xxᵀ] で誤差を後ろの列に送る) | `tools/amd/q4_gptq.py` → AMD は `iron_export_lm.py --q4-codes`、Apple は `hwnpu-convert -codes` | NPU の wiki-ja: E2B 0.221 → 0.081、E4B 0.083 → 0.049、LFM3 0.071 → 0.038、LFM3-VL 0.067 → 0.034、PLaMo 3 0.058 → 0.040。Anima の DiT の q4: 潜在の cosine 0.868 → 0.939 (q8 0.962) | 減衰は H の対角の平均 × **0.1** (0.01 は校正の文に合いすぎ、GPL が丸めより悪くなった)。校正の文は一般の文 (wikitext-2 と wiki40b-ja、64 × 512 トークン)。リポジトリの文書だけでも合いすぎる。BOS の位置は H から外す。前の層を量子化してから次の層を測る (数層ずつのパス) |
| 3 | **表の当てはめ** (Apple の lutN): E[x²] で重み付けした k-means | `q4_gptq.py --lut-fit wkmeans` | LFM3 lut6 0.050 → 0.038、lut4 0.66 → 0.33 (Go の変換) | 一様な表から始める (重みの誤差が最小の k-means から始めると KL は最悪)。ANE での確認は handoff-m3-gptq.md |
| 4 | AWQ 式のスケールの折り込み | 未実装 | | model-fit.md T8 の 2。lut4 をさらに下げたいとき |

**採るかどうか**: 校正に使っていない GPL でも丸めより悪くならないこと。LFM3-Audio は GPL だけ GPTQ で変わらない (0.154 → 0.161) が、他の 4 文が半分以下なので採った。多モーダルのモデルの文の部分は学習し直してあることが多い (LFM3-VL / Audio / Omni) ので、元のモデルの符号を使い回さず、それぞれ校正する。悪くなる文があれば、減衰を上げる (0.3 で試した)、校正の文を足す。

**校正の費用**: CPU の fp32 で 2B 級 25〜60 分、E4B 2.4 時間 (fp32 で 30 GB 以上使う。並べて流すと OOM になる)。Linux では `systemd-run --user` で流す (セッションと一緒に殺されない)。

**符号の引き当て**: 符号はチェックポイントの重みの形と bf16 の値のハッシュで引く (`iron_q4.weight_key` = `palette.WeightKey`)。書き出しが重みを並べ替えたり、つないだりすると当たらない。書き出しのログの「N weights took the chosen codes」で数を確かめる (LFM3-VL: 73 個。q8 の部分は既定で丸めるので含まない)。PLaMo 3 の `gate_up_proj` のように 1 つの重みのまま量子化するものは当たる。

### 5.4 形式の決め方

| 目安 | 判断 |
|---|---|
| KL 0.05 未満 (CPU の HF との比較) | port-architecture の目安。4 ビットを既定にしてよい |
| KL 0.05〜0.2 | 答えの質 (9 節) を見て決める。ANE の lut6 (LFM3 0.232) はここ |
| KL 1 以上 | 使えない (lut4 の LFM3 は事実の質問 4 問中 3 問を誤った) |

速さとの引き換えは、重みのバイトどおり (ANE の lut4 / lut6 / lut8 で 1.81 / 2.65 / 3.49 ms)。lut6 の遅さは `SDPAGrouped` で取り戻した (16.41)。

## 6. メモリ

| 測るもの | 道具 | 教訓 |
|---|---|---|
| AMD の固定 (`Unevictable` の増え) | `/proc/meminfo` (RSS に出ない) | 大半は重みと作業域。KV は 5 % 未満 |
| Mac の `phys_footprint` | `npuprobe split` | ANE 側 (aned) は未計測 (model-fit.md T10) |
| 会話の印の大きさ | | 使っているリングの行だけ写す (`ringRows`、578 → 183 MB) |
| 変換の直後 | | `debug.FreeOSMemory` (2.4 GB → 154 MB) |
| KV の大きさと文脈 | | 物理メモリの 1/4 を超えたら塊単位で縮める |
| ディスク | `cache.Clone` | 同じファイル (Q=8 と Q=64 の `weight.bin`、書き出しの同じ始まり) を共有。ext4 では共有されない |

**合格**: [improvements.md](improvements.md) の 5.1 の表に行を足せる数値が揃う。

## 7. 速さ

| 測るもの | 道具 |
|---|---|
| 1 回の呼び出し (Q=8、Q=64) | `TestCallTimes` (`HWNPU_CALL_PKGS`) |
| 生成の tokens/s、プロンプト、起動 (2 回目) | `hwnpu-generate`、`hwnpu-chat` のログ |
| 形式ごとの比較 | `hwnpu-formats` |
| AMD の GEMV、電力 | `tools/amd/gemvbench`、`powerbench` |

続けて入れるもの (どのモデルにも効いた):

- 2 本のプログラム (生成と読み込み) で 1 つのキャッシュ、前のプロンプトの再利用、プロンプトの保存 (ディスク)
- 投機的デコード (プロンプトの n-gram、Apple / Intel)。考える予算 (`-think-budget`)、繰り返しの検出

同じ条件で続けて測った組だけを比べる (M3 はスワップ、AMD は電源の状態で 1.4 倍ぶれる)。

## 8. 長い文脈

- 注意で分ける (`-paged`): 費用が使った長さに比例。既定 262144
- 2 段の注意 (`-sparse`): 約 48K から。合言葉の試験 (128K、250K、文書の 10 / 50 / 90 % の 3 つ)
- 状態型の層 (GDN) は 1 トークンずつと Q=64 の塊の一致 (`TestCachedMatchesFull` の形)

## 9. 用途での確認と記録

KL だけで決めない。モデルの用途で bf16 (または fp16) と並べる:

| 用途 | 確かめ方 |
|---|---|
| チャット | 16.26 の 4 問 (富士山、17 × 23、『こころ』、信濃川)、hwnpu-agent の課題 (`go test` が通るか) |
| 翻訳 | **訳文のトークンだけ**で KL と一致を測る (原文の側は予測しないので高く出る)。44 文で bf16 と一字一句同じ数 |
| 画像の説明、音声の書き起こし、TTS | 同じ画像と音声で答えを並べる。TTS はもう一度 ASR に通す |
| 埋め込み | HF との cosine |
| 画像生成 | 同じ雑音で画像を並べる。種によって構図が変わることがある (4 つに 1 つ) |

**記録**:

- design.md の 16 章に節を足す (数値、採ったもの、採らなかったものと理由)
- [improvements.md](improvements.md) と [improvements.ja.md](improvements.ja.md) の表 (速さ、メモリ、精度、モデルの一覧 10 節、未確認 11 節)
- Mac で測れなかったことは `docs/handoff-*.md` に、持っていくもの、コマンド、期待する値を書く

## 10. 試して効かなかったもの (繰り返さない)

[improvements.md](improvements.md) の 8 節の要点: 小さなモデルを詰め物で ANE に (CPU より遅い)、multifunction で 1 GiB を越える (全部 CPU)、MLState の KV、int8 の KV、次の段階を裏で読み込む、Q=16 の 1 本、E4B の lut4、モデルの圧縮、上位の鍵だけの疎な注意、AMD の注意の 4 ビット、GEMV の 1 列 1 コア、MoE の専門家の写し、DiT の `HIGH`、DiT の q4 の既定化、q4nx のプロンプトを bf16 で、RMSNorm の固定の倍率。

## 11. 今進めているもの (2026-11-07)

- AMD は済み: E2B、E4B、LFM3 (と VL / Audio / Omni)、PLaMo 3 の GPTQ の符号を NPU で測った (amd-npu-models.md 13 節の「続き」)。bf16 との比較: 画像と音声つきの E2B は文、画像、音声とも丸めと同じく正しい答え。PLaMo 3 の訳文が bf16 と一字一句同じなのは丸め 14/20、GPTQ 17/20
- Apple: GPTQ の lut6 / lut4 の表の ANE での確認 ([handoff-m3-gptq.md](handoff-m3-gptq.md))。hwnpu-chat はまだ `-codes` を取らない
- Anima: GPTQ の 4 ビットの DiT (CPU の 512×512 で潜在の cosine 0.939) を NPU の 1024×1024 の画像で確かめていない

結果が出たら 5.3 節の表と、この節を更新する。
