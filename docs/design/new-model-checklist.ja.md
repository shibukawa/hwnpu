# 新しいモデルを入れるときの検証の手順

[English](new-model-checklist.md)

LFM3 (VL / Audio / Omni、d1)、Gemma 4、Qwen3.5、PLaMo 3、PLaMo-Embedding、Whisper、MoE のモデル、画像のモデルを入れたときに後から見つかった問題と、効いた手法を、**先に確かめる順**に並べる。最初からこの順で確かめれば、やり直さずに、メモリが少なく速い形になる。構造を Go に移す作業そのものの手順と合格条件は `.claude/skills/port-architecture` にあるので、ここでは繰り返さない。

## 方針: 速さより消費電力

hwnpu は NPU だけで動かして消費電力を下げる。手法は次の順に考える。

- **合計の時間が同じなら、NPU に載せる。** CPU に移して少し速くなる形 (短い文脈の注意を CPU で、など) は採らない。GPU は使わない
- **読む重みのバイトを減らす。** 1 トークンの電力の大半は重みの読み出し。ビット数、プログラムに要らない層を持たせない、MoE は速さと電力の両方に効く
- **NPU の上での無駄を減らす。** NPU のメモリに収まらず載せ直すプログラム、1 つの演算だけ CPU に落ちて NPU と CPU を往復する形は、速さが同じでも電力を使う
- 電力は `hwnpu-dev powerbench` (1 トークンの mJ。Mac は sudo が要る) か `HWNPU_ENERGY` で測る ([memory-and-power.md](memory-and-power.md))。測れないときは、読む重みのバイト、`mem.NPUWired`、CPU の使用率で比べる

| 段 | 何を決めるか | どこで |
|---|---|---|
| 1 | 机上で入るか、速いか | どこでも |
| 2 | 構造が正しいか (fp64、HF との照合、トークナイザー、チャットの型) | CPU |
| 3 | Mac の Neural Engine に収まり、全部がそこで動くか | Mac |
| 4 | AMD / Intel の形 | Linux / Windows |
| 5 | **量子化の形式と校正** | CPU で見積もり、NPU で確認 |
| 6 | メモリ (常駐、固定、ディスク) | 各 NPU |
| 7 | 速さ (生成、プロンプト、起動) | 各 NPU |
| 8 | 長い文脈 | Mac (Intel) |
| 9 | 用途での確認と記録 | 各 NPU |
| 10 | 部品として組み合わせるための値 | 各 NPU |
| 11 | どの NPU でも名前で動く (models.yaml、取得元) | 各 NPU |

各「合格」は目安。満たさないときは理由をそのモデルの topic の文書に書いてから先に進む。

## 1. 机上の見積もり (移植する前に)

[model-fit.md](../model-fit.md) (2 節と 4 節) で config.json だけから見積もる。

- **1 トークンの時間 (ANE)** ≈ 重みのバイト ÷ 62〜65 GB/s + (長い容量に注意を計算する層の数) × 0.22 ms × 容量 / 1K + 3 ms
  - KV の共有は注意の回数を減らさない (メモリだけ)。窓、短い畳み込み、Mamba2、GDN の層は容量に依らない
- **1 トークンの時間 (AMD)**: 重みのバイトでなく演算子の切り替えの数で決まる。切り替えのたびに配列の設定を書き直し、1 回約 85〜100 µs (`tools/amd/elfconfigs` で数える)。今 (Ryzen AI 9 365): Qwen3.5-0.8B q6 18.7 tok/s、LFM3-1.7B q4 約 17、Gemma 4 E2B q4 9.7、Qwen3.6-35B-A3B 8.9、gpt-oss-20b 20.2 ([amd-llm.md](amd-llm.md))。層の数と入れ済みのモデルから見積もる
- **重み**: lut4 0.5 B、lut6 0.75 B、lut8 1 B / 重み。M3 16 GB で常駐は 3 GiB まで。AMD は 4 ビット (q4nx 5 ビット + 注意 q8) で bf16 の 1/2〜1/5 の固定。**M1 16 GB では、順に回す部分の重みのうち Neural Engine が抱える合計を約 2.9 GiB 未満に**: 超えると 1 回ごとに約 12 倍遅い (Qwen3.5-4B は 2.97 GiB で 1.1 tok/s、1.99 GiB で 10.9)。M2 と M4 は 3.62 GiB でも全速 ([apple-neural-engine.ja.md](apple-neural-engine.ja.md))
- **語彙**: 出力層は語彙 × 隠れ次元。linear は ANE で 262144 の軸を取れるが、その後の要素ごとの演算は CPU に落ちる。単調な演算 (ソフトキャップ) はホストに任せる
- **使えない構造**: 学習した疎な注意、可変の形。MoE: AMD は経路の選択をデバイス (かホスト) で。Mac は Core AI でホストが専門家を選んで回す (`llm.Routed`) か、CPU で MXFP4 で回す (`llm.HostRouted`) ([moe.md](moe.md))。全層の専門家を Mac のメモリと比べ、経路つきの層は動く重みの約 3 倍の時間と見る
- **チェックポイントが持つ下書き用の層** (Qwen3.5 の `mtp.`): 変換で捨てない。下書きとしてプロンプトの検索に勝つ (Qwen3.5-0.8B、M1: コードで 26.1 → 43.8 tok/s) ([llm-decoding.md](llm-decoding.md))
- **ライセンスと言語** (カタログに入れるか) ([distribution.md](distribution.md))

**合格**: 1K で 50 ms 以下、16K で 100 ms 以下、M3 で常駐が 3 GiB に収まる形式がある。

## 2. 構造の正しさ (CPU)

`port-architecture` の合格条件 (fp64 と 1e-4、1 層ずつと全体が一致、壊してテストが落ちるのを見る、`hwnpu-dev check` で層ごと) の上に:

| 確かめること | 道具 | 教訓 |
|---|---|---|
| ダウンロードが壊れていない | HF の `x-linked-etag` と `sha256sum` | 大きなダウンロードでまばらなビット化けがあった (LFM3-Audio / Omni: 2〜6 バイトが約 10 か所)。乱数の重みの確認も値の範囲も見つけない |
| **BOS つきで**比べる | `tools/ref/ref_logits.py`、`hwnpu-dev check`、`ironlm -forced -bos` | BOS なしだと LFM3 の KL が 4 倍悪く見え、層 0 の活性が外れ値になる。チャットの型が実際に先頭に置くトークンで |
| 本物の重み | `hwnpu-dev check`、AMD は `ironlm -forced` + `tools/ref/forced_logits.py` | 乱数の重みでは NPU と CPU が食い違う (平らなロジット、bf16 / bfp16 の丸めが広がる)。本物の重みで決める |
| **全部の大きさ**の本物の重み | `hwnpu-dev check` | 大きさが違うと config で別の道を通る (Qwen3.5-0.8B は値のヘッドとキーのヘッドが同数)。カタログに入れる大きさは全部 |
| トークナイザーが HF と一致 | `tools/ref/tokenizer_golden.py` | 決めた文と数千の乱数の文で不一致 0。同点 (PLaMo)、NFC と結合文字 (Qwen3.5)、バイトへの退避 (Gemma 4) を逆に扱うとテストが落ちることも確かめる |
| チャットの型が jinja2 と 1 文字まで一致 | `tools/ref/template_golden.py` | ツール呼び出し、思考、前の思考の削除 |
| 大きさごとのチャットの既定 | テンプレート、モデルカード | Qwen3.5-0.8B / 2B は既定で考えずに答え、4B は考える。小さいものに考えさせると予算まで回った。カタログの `no_think` は既定に従う |
| 多モーダルの前処理 | `resize_golden.py`、`mel_golden.py`、`whisper_mel_golden.py` | PIL と ±1、librosa と 1e-4 |
| エンコーダ・デコーダ (音声) | `tools/ref/ref_whisper.py`、`hwnpu-dev check` (`REF.x.bin`) | エンコーダ、交差注意のキーと値、デコーダを別々に。30 秒を超える音声は HF の窓の取り方 (次の窓はタイムスタンプから) に従う。30 秒ごとに切ると文字が落ちる |
| 埋め込みのモデル | `embed_golden.py` | HF とコサイン 0.997 以上 |
| 画像生成 | `tools/ref/compare_image.py`、モデルの `ref_*.py` | diffusers の fp32 との潜在のコサイン、VAE の PSNR。16 GB の Mac に fp32 で入らないモデルは、繰り返す層の重みを層の直前に読む (`tools/ref/stream_weights.py`) か、diffusers のブロックを meta デバイスに作って重みをブロックごとに渡す (`ref_chroma_lean.py`: M1 で 512² の DiT 1 回 98 秒。`--stats` でブロックごとの最大、`--dump` でブロックの入力) |
| **参照がどうマスクするか** | `tools/ref/ref_chroma.py --mask keys` | diffusers の ChromaPipeline は 0/1 の文のマスクをスコアに足す: Chroma は詰め物を隠して学習したのに、隠れない。NPU の正しいマスクが誤りに見えた (0.9886) が、参照でキーを隠すと 0.99985。直すのは狙った部分だけ (同じ差し替えが T5 の bias のマスクも壊した)。NPU を疑う前に参照のマスクを読む |
| Go の型なしの定数 | 尺度を読む | 型なしの整数どうしの `T5FFScale/T5Scale` は `1/64 = 0`。`float32(a)/b` と書く |

## 3. Mac の Neural Engine に収まるか

Mac がないと何も決まらない。ないときは引き継ぎ (`docs/handoff-*.md`) を書く。Mac のコマンドは Core AI だけで動く (macOS 27): モデルを MIL (`.mlpackage`) に変換し、`.aimodel` に下ろす ([apple-coreai.md](apple-coreai.md))。Core ML のパッケージは `--cpu` と開発の道具が読む ([apple-coreml.ja.md](apple-coreml.ja.md))。

### 3.1 上限

| 上限 | 対処 |
|---|---|
| 1 パッケージ 1 GiB (1 部分の既定は 0.9 GiB) | 層で別のパッケージに分ける (multifunction では越えられない) |
| 1 次元 16384 | 注意を塊に (`--paged`)、ソフトキャップを (16, 16384) に畳む |
| Core ML のコンパイルのキャッシュ約 4 GiB (LRU)、実効約 3.5 GiB | Core ML だけ: 超えると起動のたびにコンパイルし直し (Qwen3.5-2B の 3.56 GiB で 4〜5 分)。Core AI はディスクが減るまでプログラムを持つ。起動は新しいプロセスで 3 回、ほかを止めて測る |
| 表は 16 行ごと | 行が 16 で割り切れない出力層 (Whisper の 51866) は 0 の行で詰めて後で切る (`blocks.LinearPadded`)。詰めないと lut の変換が落ちる |
| `gather` は CPU | 埋め込みの行 (PLE も) はホストで引く |
| 注意のスコアは 1 テンソル 64 MiB 未満 | 超えると 2〜3 倍遅い: 問い合わせを塊に (`maxScoreBytes`)。必ずではない: Whisper のエンコーダ (90 MB) は塊にしても速くならなかった。プログラムがキャッシュを超えるときはコンパイルの時間も比べる。分ける前に測る |
| 1 プロセスの Neural Engine のプログラム 112 本 | 113 本目は GPU に。Core AI のデバイスが上限の近くで知らせる ([apple-coreai.md](apple-coreai.md)) |

### 3.2 全部が NPU に載るか

各部分を `hwnpu-dev info -model PART.mlpackage` (Core ML の配置) と Core AI の配置 (3.5) で確かめる。今までに外れたもの:

- `tile` で K/V を写す注意、問い合わせを行列にした形 → `SDPAGrouped`。順番だけで配置が変わる
- 最後の部分の `transpose` / `reshape` → 出力をトークン先に取る。出力層の後の要素ごとの演算 → ホストのサンプリングで上位の値だけに (`llm.LogitSoftcap`)
- 1 行の部品、小さなプログラム → 8 行にする、隣の群に入れる。小さなプログラム (約 45 演算) は中身によらず CPU に行く
- `reduce_sum` → 行列積 (塊の注意)
- 16384 を超える軸の演算はどれも、y を名づける identity でも: softmax の前のロジットを取り、バイアスを linear に畳み、幅の大きな出力の linear は分けて積む (`internal/onnx`)
- 4096 チャネルを超える深さ方向の畳み込み: チャネルを分ける (`maxDepthwise`)
- `int8-g32` (グループつきの int8) は linear が CPU: 行ごとの int8 に
- 8 × 8 の行列どうしの積はプログラムを NE から外す ((8 × 8)·(8 × 128) と k·kᵀ は大丈夫)
- softmax の和で割るのを P·V の後にした交差注意 (`Deferred`) は、問い合わせの数が 8 の倍数でないと M2 以降で GPU に落ちる: 問い合わせを 8 の倍数に詰める (`milpart.DeferredRows`)

### 3.3 fp16 であふれないか

`hwnpu-dev check -f16` で層ごとの相対 RMS が 0.01 未満か。1 層ずつ通っても全体で崩れることがある。次も見る:

- 全体の `-f16` と**位置ごと**のコサイン。プロンプトや音声の位置と詰め物の位置を分けて
- 外れて大きい活性 (注意の吸い込み先) がどこで生まれるか: HF の隠れ状態 (`REF.hidden.bin`) の層ごとの最大
- **原因を追う前に、HF を fp16 で動かして比べる。** 同じだけずれるなら移植は正しい (Whisper: どちらも 0.9978)
- 最初の N 層のプログラム (`hwnpu-dev convert -layers N`) を ANE と Core ML の CPU で HF と比べると、ずれの始まりがわかる

効いた手当て:

- 正規化の入力を 1/64 にする (Gemma 4: KL 3.17 → 0.0004、PLaMo 3)。値の小さな入力は除く: 二乗が fp16 の下に落ちて NaN (Qwen3-0.6B の埋め込みは約 0.1)。層 0 は `-f16` で必ず見る
- 最後の層を縮める: head-scale (PLaMo-Embedding)、`-post-scale` (LFM3)
- 残差を 1/32 で持ち、正規化の前に行ごとの最大で割る (Anima。固定の倍率は非正規数を作った)
- 残差の倍率は試して決めず、参照の層ごとの最大から決める (fp32 の参照が出す)
- 詰め物の行を最後のトークンで埋める (Gemma 4 の eps が fp16 で 0 になる)
- ANE の `sigmoid`、`tanh`、`softplus`、`silu`、`gelu` は不正確: exp で書く (`blocks.SiLU`、`onnx.Program.ExpSigmoid`、GELU は `x / (1 + exp(-1.5958 (x + 0.044715 x³)))`)。CPU の `hwnpu-dev check` では見えない: ANE と Core ML の CPU (`HWNPU_COREML_UNITS=cpu`) で位置ごとの KL を比べ、ANE と CPU インタプリタで途中の値を比べて探す (`TestBisect`)
- 後の全層が受け取る 1 行の計算 (拡散の時刻の埋め込み) は ANE で 1 % ずれた: ホストの fp32 で
- 線形注意の和は残差が収まっても fp16 を超える: 各ヘッドの q の行をその最大で割り、キーを和の前に縮め、分母の下を切る
- linear の出力の 1 行が fp16 の上限に近いと ANE でだけ NaN: 直後が正規化なら重みとバイアスを 1/64 に。部品を 1 つずつ CPU に移して探す
- **行ごとの int8 の重みは大きな出力であふれる**: NE は int8 の符号の和を取ってから行の尺度を掛け、注意の吸い込みの行の 13000 の出力が NaN になった (Core ML の CPU は正しい)。lut8 は積の前に fp16 に戻すので正しい。層ごとに非有限の値を数えて探す
- **重みに畳んだ尺度で積が fp16 の正規化の範囲を下回る** (小さなチャネルが消える)。射影の出力に掛ける。Core ML は linear の隣の定数倍をどのみち重みに畳む
- **外れ値のチャネルのある行の norm**: 2 乗の平均は最大で割った写しから取り、x そのものに rsqrt / max を掛ける (`chroma/mil.t5Norm`)
- **NE の規則** (`aicheck rules`、M2 で測った。Core AI の CPU ではどれも起きない。[apple-coreai.md](apple-coreai.md)):
  - linear と畳み込みの和は**途中の和も** 2^15 未満 (reduce_sum は全範囲)
  - 最後の軸の slice で開始を 32 で割った余りが 1〜14 だと 4094 を超える値が Inf: `aicheck lint` (か `HWNPU_AILOWER_NOTES=1`)
  - fp16 の正規化の範囲を下回る積は matmul の中で 0 になる。要素ごとでは残る
  - softplus と内側の次元が 1 の matmul は `ailower` が書き換える

### 3.4 合格

- 全部の部分が NPU に載る
- 2 回目と **3 回目**の起動が数秒 (2 回目だけでは追い出し合う 2 本を見逃す)。部分ごとの初回のコンパイルの時間を記録する
- プログラムが NE の抱えられる量を超えるモデルは、毎ステップ回すものを最小のバイトに、ほかの部分は 1 回の実行に 1 回、プロンプトはまとめて (`--batch`)。読み込みの時間を記録する
- M1 では `HWNPU_PROFILE=1` の部分の時間が呼び出しごとに揃う。順に回す部分が約 2.9 GiB を超えると、同じ部分の時間が揺れ、10 倍以上遅くなる (載せ直し)。順番を誤った回し方でも起きる (条件ごとに DiT の全部の組を通す)。2.9 GiB を超えても載せ直さないモデルもある: 載せ直しの仮説は、重みの大きさを変えて確かめてから動く
- **ディスクを空けておく**: 空きが少ないと macOS がコンパイルのキャッシュを消し (空き 11〜22 GB で見た。初回を測るときは 25 GB 空ける)、次の実行で部分ごとに数分かけてコンパイルし直す。テストのバイナリは自分の名前のキャッシュを持つ: コマンドの名前でディレクトリに作る (`go test -c -o DIR/hwnpu-image`) ([apple-coreai.md](apple-coreai.md))
- **`mem.NPUWired` は機械全体の値。** 測るプロセスは NE のロック (`llm.LockNPU`) を取るので、始まる前の値はほぼ 0。ロックの後は値が落ち着くまで待つ (終わったプロセスのプログラムが 0.1〜7 秒残る)。失敗をプログラムのせいにする前に、ほかに NE を使うものがないか見る: 別のプロセスのコンパイルで失敗した ("Unable to compute the prediction"、`Request cancelled`)
- コンパイルの間は CPU を空けておく (横の CPU の参照で配置の確認が時間切れになった。`planTimeout` は 30 分)
- **プログラムの名前を変えずにグラフを変えたら `llm.Arch.Revision` を上げる** (名前に `-rN`。画像のモデルは書き出しの版)。上げないと古いプログラムが使われる
- 形式ごとに `hwnpu-dev check -package` の KL と argmax を記録する (5 節)

### 3.5 Core AI (macOS 27)

3.1〜3.4 を通った部分も Core AI で落ちることがある。M1 でも確かめる (M3 のコンパイラが通して M1 のが落とす部分がある)。詳しくは [apple-coreai.md](apple-coreai.md)。

- **どの演算にも下ろし方が要る。** `aicheck lower OUT PKG` は最初の欠けで止まる。新しいものは coreai-torch と同じに書く: 小さな torch のモジュールを coreai-torch で変換し、プログラムを出し、バイトコードを `tools/coreai/bcdump` で読む
- **各部分を Core AI と Core ML の CPU インタプリタと比べる**: `aicheck run -cpu PKG...` と `aicheck run -cpu -coreml PKG...`。全部 NE に載る部分が Core AI でだけ誤ることがある (PP-OCR の検出器 0.05 に対し 0.9999)
- **`HWNPU_AILOWER_CUT=N aicheck run -cpu PKG` で演算を探す** (`AICHECK_WORST=1`)。誤差の大きさや最大の場所だけでなく、何の誤りかを見る (入力と出力の間に直線を当てると、NE が割るべきところで掛けていた)。小さな写しでなく本物の部分で確かめる
- **下ろし方の誤りと NE の誤りを `HWNPU_COREAI_UNITS=cpu` で分ける。** 罠: Core AI の CPU は和が fp16 を超える `reduce_mean` であふれる。CPU インタプリタは実装したものしか参照にならない (`elu` は alpha 1 だけ)。`runner` は `HWNPU_CPU_F16=1` でなければ fp32、aicheck は fp16 の丸めつき: 参照の精度を先に決める
- **NE の落とし穴**: 定数での `divide_v1` を誤る (ailower は逆数の積に書く)。`(v + c) - c` を c が入力でも `v` に畳む。幅 1 より大きな核で 512 チャネル以上、入力を別の演算が計算する幅方向の畳み込みが誤る (高さ方向に回す)。膨張つきの畳み込みは長さによって誤る (膨張の行にフレームを並べる)。畳み込みの前の明示の `pad` が誤った。(1×1)·(1×N) の matmul は M3 で誤る (mul で)
- **1 本の `.aimodel` の関数はまとめてコンパイルされる**: コンパイラが 1 つの関数を落とすと全部が NE から外れる。先に大きさごとの部分を単独で見て、`aicheck probe` で小さな書き換えを試す (`tile` のチャネルの繰り返し → 0/1 の重みの 1 × 1 の畳み込み)。隣り合う lut の表を同じにしない (`coreai-build` が部分を NE から外した)
- **起動は両方のキャッシュを温めて比べる** (各バックエンドを 1 回回した直後に、新しいプロセスで 3 回ずつ)
- **`HWNPU_PROFILE` は段で読む** (呼び出しのトークン数が同じプログラムごとに分けて足す)。部分の時間で動く前に、部分が約 62 GB/s で読んでいるか見る
- **行ごとに触る再帰の状態は重みと同じだけかかる**: 活性と状態が動かすバイトも数える (Qwen3.5 の塊の GDN)
- **語彙の一部の下書きの出力層** (`llm.Options.HeadRows`): 覆う割合でなく、採られた下書きで決める
- `aicheck run` / `probe` は NE のロックを取る: 2 つのプロセスが同時に NE を使うと Mac が固まったことがある

### 3.6 MoE

詳しくは [moe.md](moe.md)。

- **専門家の形が `internal/llm/experts` に合うか**: バイアスなしの SwiGLU、sigmoid か softmax のスコア、選択のバイアス、正規化して倍率を掛けた上位 k。共有の専門家、バイアス、クランプ (gpt-oss) はプログラムを広げる
- **先に層の形を時間で測る**: `aicheck moe -model NAME -layers L -copy` と `aicheck moepack -two`。`aicheck probe gnd16_const gnd16_pts_w1` で、そのチップが尺度の gather を取るかがわかる (M2 と M3 は取る、M1 は取らない)
- **専門家の誤差と移植の誤差を分ける**: 残りを transformers で、MoE のブロックを `aicheck moeserve` で (`tools/ref/lfm2moe_ane.py`)。次にキャッシュなしとありで `hwnpu-dev generate -score`
- **アーキテクチャで**: `llm.Routed` を実装する。MoE の層の始まりで部分を切り (`moe_x_N`、fp16 の `moe_logits_N`)、次の部分は `moe_o_N` を足して始める。`llm.PlanRouted` で切る
- **行を数える**: MoE のモデルは下書きを切り、段は 1 つ。プロンプトの呼び出しは選ばれた専門家をその行の分 1 回ずつ回す (`experts.BuildRows`)、1 回 128 トークン
- **どの専門家が回るかを、多くの種類の文で数える** (`tools/moeusage`: `collect.sh MODEL OUT`、`analyze.py OUT`)。どの種類でも回る専門家は models.yaml の `experts_core` に。出す置き場の数で核が得か確かめる (AMD の Qwen3.6 は核なし)
- **ホストが回す専門家 (`llm.HostRouted`)**: 読むのと計算を分けて測る (`HWNPU_MOE_DEBUG=1`)。先に SSD を測る。同じ質問の繰り返しで設定を比べない (ページキャッシュが温まる): 分野と言語の違う質問の一覧で (`--prompt-cache off`)。CPU のプロファイルを見る (`mincore` とカーネルで 66 %)。層の間に NE の間を入れて (`HWNPU_QWEN36_GAP`)、長いプロンプト (20 ステップ以上) でも測る。MXFP4 のチェックポイントを作る前 (`hwnpu-dev mxfp4`) にディスクの空きを見る
- **生成した文は標準エラーで比べる**: `hwnpu-chat` は思考をそこに書くので、短い実行の標準出力は空。測るバイナリの名前は変えない (Core AI のキャッシュはプロセスの名前ごと、1 モデル 1.8 GB)
- **AMD で入りきらない専門家は置き場に** (`iron_export_lm.py --expert-places P`): `HWNPU_MOE_DEBUG=check` と、全部を置いた書き出しとの比較で確かめる。16 GB の機械には `variants` を。モデルの名前で動かす (名前でだけ核と回数が渡る)
- **MoE でないモデルが変わらないことを確かめる** (同じ文、同じ速さ、変換し直さない)

## 4. AMD (IRON) と Intel の形

### 4.1 AMD

詳しくは [amd.md](amd.md)、[amd-llm.md](amd-llm.md)、[image-amd.md](image-amd.md)。

**正しさ**

| 確かめること | 教訓 |
|---|---|
| IRON に同じアプリケーションがあるか。なければ IRON の部品で書く | LFM3 と PLaMo 3 はそうした |
| HF と: `ironlm -forced` の 400 行を `tools/ref/forced_logits.py` で | bf16 で KL 0.001〜0.014、top-1 96〜99 %。長いプロンプトも |
| **プロンプトを塊の大きさそのもので** (`iron_qwen35_check.py ref --tokens N --stream`) | 32 トークンでは 1 回の道しか通らない。Qwen3.6 は塊を何回かに分けて回すと壊れた |
| 各演算子を本物の活性で fp32 と | IRON の参照はカーネルを写す (tanh GELU の表、MEDIUM の bfp16 はない): HF の 1 層の活性を 1 つの演算子に通す |
| IRON の MHA はスコアを bf16 で持つ | 大きくて差の小さなスコアは差を失う: 全部のキーから最初のキーを引く。各注意の最大のスコアを見る |
| バイアスを運ぶ定数のチャネル ("ones") | 活性と丸めを通して正確に一定に (16: silu(16) = 16) |
| MEDIUM は A と B を bfp16 に丸める。AIE2P の vtanh は \|x\| ≤ 0.5 で x を返す | 層が打ち消し合うところは、恒等の GEMM で NPU の丸めを測り、移植の前に numpy で真似る |
| bf16 の max pool の詰め物 | -30000 でなく、入力の最小のすぐ下で |
| 2 回目の実行で止まるグラフ | どのグラフも `irongraph -runs 3` で。実行の前に自分の文脈で空のコマンド (`GraphOptions.Barrier`) |
| ホストの段が読むものはコマンドの間に上書きされうる | ホストが読むものは状態に写す。ホストの段のある書き出しはモデル全体で確かめる |
| IRON のコンパイルのキャッシュは演算子の引数で引く | 作りを変える演算子には形の引数を |
| IRON がコンパイルするカーネルのヘッダを読む (mlir-aie の checkout でなく) | 違う |
| コアの丸めは最近接の偶数 | 切り捨ては和を下に引く |
| 画像と音声は最終の書き出しの形で | 書き出しを変えたら `hwnpu-chat` に画像と音声を通す |
| 状態のある層は塊の境で写す | 戻る印とプロンプトの保存のため。輪のキャッシュは戻りと書き出し / 読み込みをビット単位で (`TestKVRingRewindOnDevice`) |

**上限**

| 上限 | 対処 |
|---|---|
| 1 コマンド 2 秒未満 | 書き出しで分ける (`SCRATCHPAD_WORDS`)。超えると `ERT_CMD_STATE_TIMEOUT` |
| 制御コードはデバイスのヒープ (64 MiB) に | 密なモデルは `SCRATCHPAD_WORDS` 24 で切る。確認とデコードの ELF が両方残るか、呼び出しごとに作る |
| shim のチャネルに待たせる転送は 3 つまで | 16 で止まった。2 回続けて止まると再起動が要る |
| DMA の 1 回は 1023 語まで、記述子の繰り返しは 64 まで | 表は演算子が読む順に並べる |
| プログラムメモリは 1 コア 16 KB | noinline の関数 1 つ。カーネルを足したら確かめる |
| 量子化の射影は出力 64 の倍数、入力 512 の倍数 | まとめた射影は列ごとに詰める (`iron_gdn.column_widths`) |
| GEMV の 64 行のブロックが 8 列で割り切れる | でないと少ない列で回る (97 ブロック: 1 列)。`num_aie_columns` を見て詰める (`COLUMN_ALIGN` 64) |
| MHA の head_dim は 64 の倍数 | 96 を 128 に詰める |
| 8 列で丸ごとのブロックにならない隠れ次元 (2880) | 残差を 0 で詰めて運び、norm の重みと eps を直す |
| 呼び出しごとの行の範囲 | 呼び出しごとの語はどれも 1 つの値の一次式に: 決めた塊と 1 つの `start`。`ProjectionGEMM` は 256 行の倍数だけ |
| HIGHEST の GEMM は N が 64 の倍数、RMSNorm の重みは 4096 まで、DMA のループは 64 まで | HIGH / MEDIUM、重みを次の linear に畳む、262144 行のコマンド |
| モデルの読み込み中に `hwnpu status` を問い合わせ続けない | メールボックスの時間切れと、再起動まで戻らない SMU が続いた。`/proc/PID/fdinfo` を読む |
| 大きな書き出し | Qwen3.6: 30 分、最大 RSS 54 GB。/tmp は RAM: `~/models` の下に書く。長いものは `systemd-run --user` で |

**速さ** (生成は演算子の切り替えで決まる。プロンプトも)

| 手法 | 教訓 |
|---|---|
| 先にデコードの演算子と切り替えを数える | `model.record().steps`、`HWNPU_EXPORT_STEPS=1`、`elfconfigs`。`--command-per-step` と `HWNPU_IRON_TIMING` で 1 つずつ測る (1 コマンドに約 0.1 ms)。その書き出しでは時間だけを読む |
| **デコードの演算子を減らす** | バイアスと残差を GEMV で (`into_linear`)、gate と up を 1 つの表に (`QGEMVGlu`)、sandwich norm を 1 つに (`SandwichAdd`)、q と k の RoPE を 1 回に、短い畳み込みを 1 つに (`ShortConvStep`)、状態の枠を直接 (`slots=`)、QGEMV を 1 列 1 コアに |
| **プロンプトの塊の演算子を減らす** | 短い畳み込みを 1 つの演算子に (`ShortConvChunk`、`GdnConvChunk`)。小さな設定 (RMSNorm、和) をまとめても効かない |
| デコードの 1 歩を 1 つの常駐の配列に (`GptOssStep`) | 載せ直しがなくなる (gpt-oss 79 → 49 ms)。代わりにコアのプログラムを 16 KB に収め、毎歩すべての operand を宣言する。層で違う `array=True` の引数は層ごとの配列になる: 書き出しを変えたらコマンドの時間を読む |
| プロンプトの MHA を複数のパイプラインで (`--mha-pipelines`、既定 8) | 塊が 2〜5 倍速い。カーネルの名前の `num_pipelines` を見る |
| 塊の大きさ | 注意のモデルは 1024、GDN のモデルは 512、256 未満にはしない |
| 演算子が繰り返す仕事を外に出す。ホストの段はトークンで CPU に分ける | GDN の SiLU を 1 回に: 30 → 10 ms |
| KV を共有する層はプロンプトの最後の行だけ、窓の層は窓だけ読む (`chunkBarrier` が要る) | Gemma 4 E2B のプロンプト 2.3 倍、−12 % |
| 表をホストに (PLE、埋め込み)、塔は使うときに、`-idle` | E2B の固定 9.08 → 4.60 GB |
| 注意の sink | sink の項を MHA の途中の和に、最後の割り算の前に足す。sink 6 以上でも試す |
| 指数 | コアの指数の回数と幅を数える。bf16 の 3 次でなく 2 limb の 4 次 |
| 段を速くする前に、カーネルを途中で止めて時間を測る | 行列積はただだった。指数と和はそうでなかった |
| 下書きを 1 回で確かめる (`--verify 3`) | どの演算も数行の形が要る (`QGEMVRows` は q8 / q6)。合格: `ironlm -verify` がデコードの 1 歩とビット単位で同じ |
| ホストの分も測る (pprof) | 201088 のロジットのループが 1.33 ms |
| 同じブロックのグラフで重みの bank を 1 つに (`share_banks`) | 2 つ目を置く前に bank の像を比べる |
| 画像の大きな部分 | 文の符号器と DiT を組にして 1 つの bank で (`SwapVariants`)。組の切り替えを確かめる (`irongraph -swap -variant 1`) |
| 測る間はほかに `/dev/accel` を使うものがない | ほかの利用でコマンドが最大 50 倍遅くなった |

書き出しの名前: `~/models/NAME.amd-npu2-q4` (か `NAME.amd-npu2`)。AMD の NPU だけの機械では、hwnpu-chat が `-m NAME` にそれを選ぶ。

### 4.2 Intel

`ovlower` で IR に。既定は int8 (int4 は次のトークンが fp16 と 52 中 22 しか合わない)。今は OpenVINO の CPU で精度だけを確かめられる ([intel.md](intel.md))。

## 5. 量子化: 形式と校正

**いちばん手間がかかり、いちばん効く段。** 4 ビットで失うものはモデルで 10 倍違う (AMD の q4 の KL: PLaMo 3 の訳 0.009、LFM3 0.036、E4B 0.049、E2B 0.097)。詳しさとモデルごとの選択は [quantization.md](quantization.md)。

### 5.1 測り方を揃える

- **KL(参照 ‖ 量子化)**: 400 位置 (256 をプロンプトとして読み、後は 1 つずつ) の平均、中央値、95 パーセンタイル、top-1。**BOS つき**で
- `~/models/calib` の 5 つの文: README (`testdata/texts` に `GPL-3`、`model.go` と止めてある)、GPL-3 (**校正には使わない**。過学習を見つけるため)、wiki40b-ja (評価の部分)、wikitext-2 (テストの部分)、Go のコード
- 参照は CPU の HF の fp32 か、同じ NPU の bf16。NPU での量子化の KL は CPU の見積もりと合うはず。合わなければ NPU の演算を疑う
- Apple: キャッシュのないプログラム (`hwnpu-dev convert -seq 128`) で `hwnpu-dev generate -package P.parts.json -score REF,...`。最初のトークンの重みが大きい (KL が 8〜13 まで): 設定は同じ文どうしで比べる
- 小さな KL の差は 5 つの文では決められない: 約 128 位置の平均は全体の値の 0.6〜1.4 倍に散り、0.01 以下の差の設定は 34〜42 % の割合で逆順になった。僅差なら 512 トークンの塊を 24〜36 個 (256 をプロンプト、残り 256 を採点) 使い、塊ごとの平均の対の差に 95 % 区間を付ける。0 を含めば差なしとする。36 塊では区間は約 ±15 %。
- チャットのモデルには、user の発話の中でなくアシスタントの答えとして文を読ませる: 発話の中では各位置で `<|im_end|>` を予測するため分布が変わる (Qwen3.6 のパープレキシティは発話中 106〜582、答えの形で 5.7〜10.9)。gpt-oss は wikitext-2 の書式を自分では書かず、答えの形でも 131、自分の答えでは 1.4。経路の規則を変えたときの KL は wikitext 調の文で 7 倍高かった。生成に効く変更はモデル自身の答えでも測る。
- 画像生成: 参照の軌道に沿った速度の相対 RMS (参照の DiT の入力を 1 つずつ NPU に通す、`chroma.TestCallsOnDevice`) と、自由に回した最後の潜在。ステップとともに育つ誤差は軌道に沿ってでないと見えない

### 5.2 部分ごとのビット数を CPU で掃く

各部分 (注意の射影、MLP、出力、畳み込み、塔) を 4 ビット、8 ビット、bf16 で切り替える (`q4_gptq.py --method rtn` に `--q8`、`--head`、`--fmt-*`)。今までに:

- **注意の射影は 8 ビット**: 重みの約 10 % だが 4 ビットの誤差の大半 (E2B 0.097 → 0.229)
- MLP と出力の行列は 4 ビット、短い畳み込みは 8 ビット
- sandwich norm のモデル (Gemma 4、PLaMo 3) は 4 ビットに弱い: ANE では lut6 以上
- **同じ構造の大きいものは少ないビットで済むことがある** (PLaMo 3 Translate 8B は lut4 で持ち、2B は崩れた): 大きさごとに測る (`Formatter`)
- 多モーダルの文の部分、TTS、d1 の判断、埋め込み、M3 での LFM3 の系統は lut8
- **小さなモデル (2B 以下) の MLP は 4 ビットで崩れることがある**: Qwen3.5-0.8B は README 0.055 なのに GPL 0.34。AMD の q6 で 0.044。GPL を必ず見る
- **すでに 4 ビットのチェックポイント (MXFP4) はその形式のまま**: q4nx に丸め直すと KL が 2〜3 倍
- DiT: 行ごとの int8 と lut8 の両方を試して絵を見る (int8 があふれるものがある)。蒸留した少ないステップの DiT は 4 ビットに弱い。大きな DiT は k-means-minmax の表で lut4 に持つことがある。MLP 6 ビット、注意 8 ビット。プロンプトごとに 1 回のパスはただで形式を上げられる
- 1 枚に 1 回で閉じる部分 (文の符号器) は DiT ほどバイトを削らなくてよいが、fp16 が要る重みだけを fp16 に (`--text-format-for wo=fp16`)

### 5.3 校正の方法を順に

形式は変わらないので、速さとメモリも変わらない。

| 順 | 方法 | 道具 | 注意 |
|---|---|---|---|
| 1 | 丸め (RTN)。q4nx は 9 つの候補から誤差の最小を選ぶ | `iron_q4.quantize_fmt`、`palette.QuantizeBits` | 基準。重みの誤差の最小は出力の誤差の最小ではない |
| 1b | **ブロックの最小と最大を残す k-means の表** (lut4、データなし) | `-quant kmeans-minmax` | lut4 の既定。ただの k-means は外れ値を切る |
| 2 | **GPTQ** | `tools/amd/q4_gptq.py` → AMD `--q4-codes`、Apple `hwnpu-dev convert -codes`、`hwnpu-chat --codes`、models.yaml の `codes:` | 減衰は H の対角の平均の 0.1 倍 (0.01 は過学習)。一般の文で校正 (wikitext-2 + wiki40b-ja、64 × 512)。H から BOS を除く。前の層から量子化する。全部の射影を校正するには `--q8 gptq` |
| 3 | Apple の lut の表を合わせる: E[x²] で重みづけた k-means | `q4_gptq.py --lut-fit wkmeans` | 一様な表から始める |
| 4 | lut の前の回転 (`llm.Rotate`) | `tools/quant/rotkl` | Qwen3.5 の KL が半分 (NE の lut6 で 0.054 → 0.029)。lut6 の書き出しの既定 |
| — | 形式を 1 つ上げる (AMD の q6) | `--mlp q6` | 校正で直らないとき |

- **採るのは** GPL が丸めより悪くならないときだけ。多モーダルのモデルの文の部分は学習し直されていることが多い: それぞれ校正し、元のモデルの codes を使い回さない
- **DiT の GPTQ は、別の種で自由に回して判断する**: 教師強制の誤差は減ったのに自由に回すと悪くなった (DiT には採らない)
- **校正が残す重み**も丸められる: 一覧の外の小さな射影を探して 1 段上に (Gemma 4 の `per_layer`: `-format-for per_layer=lut8`)
- 16 GB の Mac 向けの軽い読み込みは、まず transformers のロジットと合わせる (KL 0.003 以下)
- **codes の照合**は形と bf16 の値のハッシュ (`palette.WeightKey`): 書き出しのログの "N weights took the chosen codes" を見る。codes は取得元にも置く (`converted/MODEL/any/codes-HASH`)。でないと書き出しの鍵が変わったとき誰も変換し直せない
- 費用: fp32 で 2B のモデル 25〜60 分、E4B 2.4 時間 (2 つ並べるとメモリが足りない)。16 GB の Mac は E2B を軽く読んで 87 分。Linux では `systemd-run --user` で

### 5.4 形式の決め方

| 目安 | 決め方 |
|---|---|
| KL 0.05 未満 (CPU の HF と) | 4 ビットを既定にしてよい |
| KL 0.05〜0.2 | 答えで決める (9 節) |
| KL 1 以上 | 使えない |
| `hwnpu-dev formats`: 参照の次のトークンの 90 % 以上、短い問いの正答も同じだけ | 小さい形式を取る |
| 85〜90 %、または計算や事実の新しい誤り | 1 つ大きい形式のまま |
| 80 % 未満、または繰り返す答え | 使えない |
| 音声と画像のエンコーダ | WER / CER で決める (Whisper は lut8 のまま) |

- 速さは重みのバイトに従う (ANE の lut4 / lut6 / lut8 の 1 回: 1.81 / 2.65 / 3.49 ms)。probe の形式ごとの読む速さはモデルの速さではない: 本物の部分を `HWNPU_PROFILE=1` で測る
- 参照は fp16。部分に入らないときは lut8 (`-ref lut8`) にし、lut8 を HF と比べる
- パープレキシティでなく参照との一致で比べる (短い文では量子化のほうが低いことがある)
- 形式は大きさで違うことがある: チェックポイントごとに決める (`Formatter`)

## 6. メモリ

詳しくは [memory-and-power.md](memory-and-power.md)。

| 測るもの | 道具 | 教訓 |
|---|---|---|
| AMD の固定のメモリ | `/proc/meminfo` の `Unevictable`、`manifest.json` の bank | 大半は重みと作業域。KV は 5 % 未満。GDN の層は塊の境ごとに枠を持つので、固定は文脈とともに増える |
| Mac の footprint と NE のメモリ | `hwnpu-dev probe split`、`HWNPU_PROFILE=1`、`mem.NPUWired` | NE の重みはカーネルの wired のメモリ。読み込みでなく最初の呼び出しで載る: 1 回呼んでから測る |
| 1 回の実行の最大 | `go run ./tools/memwatch [-csv FILE] CMD...` (0.1 秒ごと) | mlock したファイルのページは footprint に入らない。内訳は `footprint PID`、`vmmap -summary PID` |
| 会話の印 | | 使っている輪の行だけ写す (`ringRows`) |
| 変換の後 | | `debug.FreeOSMemory` |
| KV と文脈 | | 物理メモリの 1/4 を超えるなら塊単位で縮める |
| ディスク | `cache.Clone`、`LinkFrom` | 同じファイルを共有する (ext4 では何も共有されない)。大きさによらない部分は別の書き出しからハードリンク |
| VAE の段の間のホストのメモリ | `debug.FreeOSMemory` | 大きな段の fp32 の入出力が Go のヒープに残る。ステップの前だけに要るホストの重みも手放す |
| AMD の VAE の先読み | `anima.VAE.PrefetchLimit` | 2 段が 3 GiB に収まるときだけ先読み |

**合格**: models.yaml の `memory` と [performance.md](performance.md) の 1 行に足りる数値。

## 7. 速さ

| 測るもの | 道具 |
|---|---|
| 1 回の呼び出し | `TestCallTimes` (`HWNPU_CALL_PKGS`) |
| 生成、プロンプト、起動 (2 回目) | `hwnpu-dev generate`、`hwnpu-chat` のログ |
| 形式の比較 | `hwnpu-dev formats` |
| AMD の GEMV、電力 | `tools/amd/gemvbench`、`hwnpu-dev powerbench` |
| 1 回の実行の段階ごとの電力量 | `HWNPU_ENERGY=10 hwnpu-image ...` (chat、audio も) |
| 1 歩の内訳 | `HWNPU_PROFILE=1` |

最初から入れるもの ([llm-engine.md](llm-engine.md)、[llm-decoding.md](llm-decoding.md)):

- 1 つのキャッシュに 2 本のプログラム (生成とプロンプト)、前のプロンプトの使い回し、プロンプトの保存
- **プロンプトのプログラムの出力層は最後の行だけ** (`LastRow`)
- 下書き (プロンプトの検索。MTP は GDN の状態を `state-rows` からホストで作り直す。AMD は `--verify`)、考える予算、繰り返しの検出。MTP の下書きは 1、2、3 個をコードと文で測る (3 個目は日本語で負けた)
- キャッシュを書く最後の層で止まるプロンプトのプログラム (`-kv-only`、`llm.CacheWriters`)
- 出力層の後の単調な演算はホストで (`llm.LogitSoftcap`)

比べるのは、続けて同じ条件で測った組だけ (M3 のスワップ、AMD の電源の状態で最大 1.4 倍揺れる)。先に `top` を見る: Mac では `mediaanalysisd` と Apple Intelligence が NE を取る (どの形も 0.7 tok/s に落ちた)。

### 7.1 画像生成

1 枚の時間は DiT のパス: ステップ × 1 ステップのパス (CFG で 2) × 1 パス。まずパスを減らし、次に 1 パスの時間。1 パスの FLOP を NPU と比べて残りの余地を見る。どの変更も同じ雑音から CPU の fp32 の参照と、種の格子で確かめる。詳しさとモデルごとの状況は [image-models.md](image-models.md)、[image-apple.md](image-apple.md)、[image-amd.md](image-amd.md)。

新しい画像のモデルをそれぞれに揃える (当てはまらなければ理由を書く):

- CFG の 2 パスを重みの組ごとに続けて (`velocity`)、プロンプトによらない埋め込みは 1 回
- 空の否定のプロンプトは書き出しで計算した定数
- 2 次の解法 (DPM-Solver++ 2M)。多くの Euler のステップと比べて判断
- CFG は雑音の多いステップだけ (`--cfg-sigmas 0.6,1`)
- ステップによらないキーと値はプロンプトごとに 1 回 (交差注意、t = 0 で変調する文の行)。変調は sigma ごとに 1 回、ステップの前に閉じる
- 決まった指示の前置きは定数で持つ (塊より長いときだけ)
- 否定は同じ文の呼び出しで読む (`EncodeAll`)
- 複数枚をまとめて (`--batch`): 部分を 1 回読み、N 枚回す
- Core AI: 注意のスケールを q の norm に、絶対値は `abs` で、注意をヘッドごとにスコアを (1, 問い合わせ, 1, キー) に、キーを 2 のべき乗の塊に (`milpart.KeyParts`)。次のプログラムを先に読む (`runner.InTurn`)。DiT の組を入れ替える
- AMD: 画像の注意を 64 キーのブロックで QKᵀ を bfp16 に (`ImageMHA`)、VAE の畳み込みを格子の上で (`GridGEMM`)、同じ形の VAE の段を 1 つのグラフに、DiT の重みを bfp16 に、文の符号器を組に (`SwapVariants`)
- DiT の重みのバイトを減らす (絵で判断)、M1 での `mem.NPUWired`、1 枚の電力量、diffusers の fp32 と 3 プロンプト × 2 種

採る前に測るもの: 初回のコンパイルのための小さなプログラム、文の行を減らす、1 ステップのプログラムを減らす (`aicheck cycle`: 大きな組が別の組に続くと 50〜70 ms)。

## 8. 長い文脈

- 注意で分ける (`--paged`): 費用は使った長さに従う。分けると注意の層ごとに NE を 2 回呼ぶ: 全部の層が注意のモデル (Qwen3-0.6B) は 1 本のほうが 2〜4 倍速いことがある (`llm.Unpaged`)。注意の層を数えて両方測る
- 2 段の注意 (`-sparse`) は約 48K から。パスキーの試験 (128K、250K、文書の 10 / 50 / 90 % に 3 つ)
- 状態のある層 (GDN): 1 トークンずつと塊が一致する (`TestCachedMatchesFull`)

## 9. 用途での確認と記録

KL だけで決めない。用途での答えを bf16 / fp16 と並べる:

| 用途 | 確かめ方 |
|---|---|
| チャット | 4 つの問い (富士山、17 × 23、『こころ』、信濃川) と `hwnpu-chat agent` の課題 (`go test` が通るか)。小さなモデルが誤るときは fp16 に同じことを聞く: 同じに誤るならモデルの限界 |
| 翻訳 | **訳のトークンだけ**で KL と一致。44 の文のうち bf16 と一字一句同じ数。`hwnpu-dev formats` のパープレキシティは意味がない |
| 画像の説明、TTS | 同じ画像と音声で。TTS の出力は ASR に戻す |
| 書き起こし | LibriSpeech の検証の 73 本、`say` の 4 つの声で日本語 12 文。HF の fp32 と同じ本数、WER / CER。30 秒を超える音声も |
| 埋め込み | HF とのコサイン |
| 検出、OCR | ONNX Runtime と、重みだけ fp16 にしたグラフ (`tools/ref/onnx_f16_weights.py`) と物体を比べる。OCR は配布元の流れ全体を参照に (`ref_ppocr.py`、`compare_ocr.py`)。詰め物は配布元と同じに |
| 画像生成 | 同じ雑音からの絵。3 プロンプト × 2 種を参照と並べる。蒸留した少ないステップのモデルは条件の小さな変化を広げる: 部分はそれ自身の fp32 との誤差と絵の格子で判断する。どのアスペクト比も一番下の機械で 1 回描いて見る |

実際には: 比較用の fp16 のプログラムでディスクが倍になる (比べたら消す)。Core ML と Core AI はプロセスの名前ごとにキャッシュを作る (それぞれ数 GB)。止まったダウンロードは時間切れにならない。`hwnpu-dev formats` は分けていない 1 本の重みを 0.00 GiB と出す。

**記録**: モデルの事実、数値、採ったもの、採らなかったもの (と理由) は `docs/design/` のそのモデルの topic の文書に。[performance.md](performance.md) に 1 行。models.yaml の測った値 (11 節)。測れなかったものは `docs/handoff-*.md` に、持っていくもの、コマンド、期待する値とともに。

## 10. 部品として組み合わせるための値

モデルはつなげて (音声認識 → LLM → TTS) 1 つのプロセスで一緒に動かす予定 ([api.md](api.md))。モデルを入れるときに記録する:

| 何を | どう測る | 何のため |
|---|---|---|
| 部品ごとの固定の大きさ | AMD: `smaps` の `/dev/accel/accel0` の割り付け。Mac: 重みと `phys_footprint` | メモリの予算 |
| 部品ごとの読み直しの時間 | AMD `Release` / `Reload`、Mac はキャッシュが当たったとき | 追い出してよいか |
| 一緒に回る部品と順に回る部品 | 処理の流れ | 段ごとに読めるのは順に回る部品だけ |
| AMD: いちばん長いコマンド | `hwnpu-chat` / `irongraph` のログ | ほかがそれを待つ |
| Mac: 部品の重みの合計と切り替えの時間 | `TestCallTimes` を交互に | 切り替えは 15〜35 ms |
| 入出力の種類と条件 | モデルカード、前処理 | つなぐモデルとの突き合わせ |
| 流せるか、最初の出力までの時間 | 塊を入れる | 音声の連鎖の遅れ |

- モデルの間は文で渡す (トークナイザーが同じならトークン ID)。ベクトルを直接つなぐのは一緒に学習した部品の間だけ
- 音声を音声のまま扱うモデル (Audio、Omni) は、音声を入れて音声を出す 1 つの部品

## 11. どの NPU でも名前で動くようにする (models.yaml と準備)

名前、ファミリー、用途から、手作業なしにどの NPU でも準備して動くまでは入ったことにしない。詳しくは [distribution.md](distribution.md)。

### 11.1 準備の順

1. **手元の書き出し**で `NAME.done` (最後に書く) のあるもの。途中で切れたものは作り直す
2. **取得元の書き出し** (`HWNPU_MODEL_SOURCE`): 同じレシピの項目、なければ人が名前で置いたもの (`hwnpu-dev sources import`)
3. **手元での変換**: 変換器が読むファイルだけを落とし、sha256 を確かめ、変換し、`.done` を書く。Apple と Intel は Go の変換器 (`chat.Prepare`、`anima.Prepare`)。AMD は hwnpu に入った IRON の書き出し器 (`chat.PrepareAMD`、`amd_export`、`HWNPU_IRON_ENV`)。IRON がなければ止めて、入れるものを言う (INSTALL_AMD.md)
4. **使えない**: 止めて、何が足りないかを言う

- リポジトリを丸ごと落とさない (Sana のものは 16.5 GB で、要るのは 8.9)。ファミリーで共通のファイルは共有する
- この NPU に形のないモデルは動ける場所 (CPU) に落ち、そう言う。`--device npu` を明示したときだけ誤り

### 11.2 models.yaml の値 (見積もりでなく測った値)

| 欄 | 何 | どう測る | 落とし穴 |
|---|---|---|---|
| `bytes` | ダウンロード | 取るファイルの HF の `lfs.size` の和。AMD の NPU で、準備がチェックポイントでなく取得元の書き出しを取るものは、取得元に置かれた書き出しの大きさを NPU の種類ごとの表に (`{amd: 12.50e9, apple: 1.6e10}`)。エントリが圧縮されていれば (`files.tar.zst`) `entry.json` の `pack.size`、そうでなければ `files` の和 (`hwnpu-dev sources ls` が出すのは展開後の大きさ) | GB は 10⁹。`du -h` は GiB を出す。ディスクの上の書き出しはダウンロードの量ではない: 圧縮されたものはもっと小さい (FLUX.2 klein はディスクで 16.0 GB、圧縮で 12.5 GB)。ファイルのまま置かれた画像の書き出しは、大きさごとのプログラムが重みごと入り、チェックポイントの数倍になる (DA3MONO-LARGE は 1.4 GB に対して 6.59 GB)。表にない種類は一番大きい値になる |
| `disk` | 動かすのにディスクに残る量 | 展開した変換結果と表の和を NPU の種類ごとに (`{amd: 2.28e9}`)。エントリの `files` の和。`hwnpu-dev models hf sizes --write models.yaml` が取得元から書く (AMD はそのモデル自身の書き出しで、変種のものではない。Core AI はモデルの版のエントリだけなので、Mac で一度動かしたあと) | チェックポイント (変換するときだけ読む) と、Mac で Core AI が作るコンパイル結果 (hwnpu のキャッシュの外) は入れない。表にない種類は不明 (`hwnpu-chat list` で `-`) で、一番大きい値にはならない。準備のあいだは `bytes` と `disk` を同時に使う |
| `memory` | 動いている間の最大 | 1 回の実行の全体を 0.1 秒ごとに (`tools/memwatch`。AMD は accel の割り付け + RSS) | 読み込みの直後でなく実行の最大 (文の段、VAE) |
| `machine` | ファミリーがそのモデルを勧める最小の機械のメモリ | `memory` + OS とアプリの余裕 | 同じ大きさのモデルは揃える |
| `file` | 1 つのリポジトリに何本もあるときのチェックポイントのファイル | ファイルの一覧 | テンソルの名前の頭が違うことがある |
| `device` | 会社ごとの動く場所 | NPU と CPU で 1 回の電力量 | 小さなモデルは AMD の NPU で負ける (YuNet 137 mJ に対し 10 mJ) |

Apple の画像モデルの Core AI の `bytes` には、いちばん大きいチップ向けのプリビルトを足す。

- **受け入れの要るライセンス** (`license: {id, url, terms, accept: true}`): 端末で 1 回聞き、`~/.config/hwnpu/licenses.json` に残す。端末がなければ `--accept-license` / `HWNPU_ACCEPT_LICENSE`
- **NPU で違う値**: 会社ごとの表 (`memory: {amd: 3.5, apple: 4.1}`)。チップの名前も書ける (`M1: 1.7`)。表にない会社は最大の値
- **用途はファミリーと同じくメモリで選ぶ**: 用途の `picks` にファミリーと同じ段を
- **確かめる**: `HWNPU_MEMORY=8`、`16`、`24` で `hwnpu-image list` / `hwnpu-chat list`。各 NPU で大きさ、メモリ、RUNS FROM が正しい

測る台本の例 (AMD):

```bash
"$@" & pid=$!
while kill -0 $pid 2>/dev/null; do
  awk '/accel/{a=1;next} /^[0-9a-f]+-/{a=0} a&&/^Size:/{s+=$2} END{printf "%d ", s/1024}' /proc/$pid/smaps
  awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status    # NPU と RSS (MiB)。memory はその和の最大
  sleep 0.1
done
```

### 11.3 AMD の書き出しと Go

- IRON のコンパイラ (mlir-aie、Peano) は安く Go に移せない。コンパイルしたプログラムは形だけに依り、重みは bank にある。**同じ形なら同じプログラム** (Sana の書き出しでバイト単位で確かめた): 形ごとにプログラムを配り、Go で bank を書けば、利用者の機械に Python は要らない
- 形に入るもの: 次元、絵の大きさ、文脈の長さと塊の大きさ、部分ごとの形式、NPU の世代と列の数、書き出しの形式の版
- Python なしのコンパイルは、新しい形を作る人と CI だけの話。優先は低い ([amd.md](amd.md))

### 11.4 画像の書き出し: 全部のアスペクト比を 1 つに、取得元に

**決まり (どの画像のモデルも、どの NPU も)**: モデルと NPU の会社ごとに 1 つの書き出しが models.yaml の**全部**のアスペクト比を持ち、重みは 1 回だけ置く。大きさごとの書き出しも、一部の大きさだけのものも作らない。`--size` を変えた利用者にダウンロードし直させない。例外には、測った理由をモデルの topic の文書と models.yaml の項目に書く。

- **大きさは models.yaml のもの** (`npu.amd.sizes`、`npu.apple.sizes`: `--size` が取る 8 つの比、`internal/imagegen/aspect`)。コマンドはそれで `--size` を読み、書き出しを作る
- **Apple (Core AI)**: 大きさでグラフが変わる部分は、大きさごとの関数 (`s{W}x{H}`、`llm.WriteAIFunctions`) を持つ 1 本の `.aimodel`。大きさによらない部分は関数 1 つ。manifest の `sizes` に大きさごとの部分を書き (`ProgramManifest.At`)、名前に大きさを入れる (`aspect.Tag`)。部分の数は大きさで変えない
- **Apple での費用**: Core AI は `.aimodel` の関数をまとめてコンパイルし、コンパイラが 1 つの大きさを落とすとその部分は全部の大きさで NPU から外れる。取得元に置く前に、大きさを 1 つずつ、prebuilt を作るチップごとにオフラインでコンパイルする (`coreai-build compile … --architecture h13g/h14g/h15g/h16g`、`/usr/bin/log` で `Falling back` を探す)。その後の `hwnpu-dev models package MODEL/coreai-ARCH --arch ARCH EXPORT` は `0 failed` で終わること。VAE には prebuilt を作らない (`coreai.LeftToDevice`)
- **AMD (IRON)**: 最初の大きさのグラフが bank を持ち、ほかの大きさのグラフは同じ bank の上で `sizes/WxH/` に (`share_banks`)。512 行の丸ごとにならない大きさは詰めて回し、キーを `nk` で区切る。詰めた行だけに依るグラフは共有する
- **鍵**: `chat.PinnedRecipe(model, tool, version)` と書き出し器の引数。models.yaml に読む全部のファイルの `sha256:` が要る。入力か出力が変わったら書き出しの版を手で上げる
- **最初の実行が取得元に置く** (`chat.SharedExportOr`、`?writeback=true`)。手で作ったものは `hwnpu-dev sources import DIR` か、最初に使ったとき
- **反対側から確かめる**: `hwnpu-dev sources ls N`。手元の書き出しを脇に置いて名前で動かす: 取得元から来て、同じバイトを描くこと
- **取得元のチェックポイント**は `hwnpu-lock.json` があるときだけ読む
- **どの大きさの絵も**一番下の機械で見る (壊れた関数はそこにしか出ない)
- **書き出しの時間、大きさ、最大 RSS を記録する** (models.yaml と利用者へのメッセージに)

## 採らなかったもの

測って、繰り返さないもの (1 行ずつ。ほかは [performance.md](performance.md) と topic の文書に):

- 小さなモデルを詰めて ANE でまとめて回す: CPU 73 µs → ANE 280 µs
- 1 GiB を越えるための multifunction: 上限はパッケージ全体で数え、全部 CPU に行った
- MLState の KV キャッシュ: 速さは同じ (電力のための選択肢として残す)。int8 の KV キャッシュ: メモリ半分で 10 % 遅い
- 次の段の裏での読み込み: 生成が 2 倍以上遅い
- 生成とプロンプトに Q=16 のプログラム 1 本: +15 %
- E4B の lut4: 20 % 速いだけで、次のトークンの一致 90 % → 64 %
- モデルのファイルの圧縮 (lz4、zstd): 71〜100 %、割り付けたファイルを展開することになる
- 上位 k の単独のキーの疎な注意: 128K で重みの 53 % (2 段にした)
- AMD の注意の射影を 4 ビットに: 固定 −0.12 GB、E2B の KL 0.097 → 0.229
- AMD の MoE: 専門家を写してから GEMV (26 GB/s の写しはその場で読むより遅い)
- 小さな Qwen3.5 の 4 ビットの MLP を GPTQ で直す: GPL は崩れたまま (q6 に)。q6 から速い GEMV: q8 の時間
- AMD の DiT を `HIGH` に: 0.966 → 0.976 に 7.2 → 10.9 秒 / ステップ。DiT を既定で q4 (潜在 0.777)。DiT の GPTQ (自由に回すと悪い)
- q4nx のプロンプトを bf16 を通して (`QDequant`): KL は同じで 5〜9 % 遅い
- fp16 の固定の倍率の RMSNorm: 非正規数 (行の最大で割る)
- Whisper のエンコーダの LayerNorm を行の最大で割る (変わらない)、その注意を問い合わせの塊に (速くならない)
- NE の MoE: 定数の表から専門家を gather (正しいが 6 分の 1 の速さ)、専門家を 1 つの配列に (幅の広い専門家で遅い)、4 ビットの専門家 (int8 より速い形がない)
- gpt-oss の出力層を lut4 / lut6 に (KL 0.0088 → 0.068 / 0.0136。lut8 の 0.55 GiB に対し 0.27 / 0.41 GiB)
- ホストが回す MoE の専門家にプロンプトの検索の下書き: 7 つの下書きの確認で 60 ms の 1 歩に約 90 ms 足される
- 重みつきの `.aimodelc` を配る。チャットのモデルの prebuilt (初回の読み込みが遅い: LFM3-1.7B 2:17 → 8:12)。VAE の prebuilt (縞、得もない)
- 絵の大きさごとの書き出し (ダウンロードし直しになる。11.4)
