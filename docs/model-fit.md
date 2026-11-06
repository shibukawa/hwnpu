# Neural Engine に合うモデルの条件と、少ないメモリで速く賢くするための手法 (調査と提案)

作成: 2026-11-01。対象は hwnpu のチャットのエンジン (`cmd/hwnpu-chat`、`internal/chat`、`internal/llm/decode`) と、今動いている LFM3-1.7B と Gemma 4 E2B。数値は design.md 16 章と [kv-cache.md](kv-cache.md) の M3 (16 GB、macOS 27) の実測から取り、見込みはその実測の係数で計算した。候補モデルの構成は Hugging Face に届かない環境で調べたので、config.json を直接見ていない値には **要確認** を付けた (二次情報の出典は 7 節)。

## 1. 要約

Neural Engine で 1 トークンにかかる時間は、実測ではほぼ次の 2 項で決まる (2 節)。

> 1 トークン ≈ **重みのバイト数 ÷ 65 GB/s** + **長い容量に対する注意の計算の回数 × (0.22〜0.47 ms) × 容量 / 1K** + 約 3 ms

常駐のメモリは **読み込んだプログラムの数 × 重みのバイト数** が大半で、KV キャッシュはその 1 割に満たない (LFM3 の 16K で 190 MiB)。

ここから、やることの順番は次になる。

| 順 | 手法 | 何に効くか | 見込み (M3) | 変更の量 |
|---|---|---|---|---|
| 1 | LFM3 を lut6 にする (3 節 T7) | 賢さ | lut4 の KL 0.89 を Gemma の例 (2.54 → 0.25) から 0.2〜0.3 に。1 トークン +6 ms、重み +0.4 GiB | 変更なし。`hwnpu-check` で KL を測って既定を変えるだけ |
| 2 | LFM3 の注意を `SDPAGrouped` の形にする (T1) | 速さ | 16K で 64 → 45〜50 ms。Gemma 4 の係数 (0.22 ms) と LFM3 の係数 (0.47 ms) の差がこの形の差 | 数十行 |
| 3 | 重みなしの注意プログラムに分け、使っている塊だけ動かす (T2) | 速さとメモリ | 費用が容量でなく長さに比例。段階 (tiers) が要らなくなり、段階ごとの重みの重複 (0.5〜0.7 GiB) が消える | 数百行。先に境目の費用を `npuprobe` で測る |
| 4 | 生成と prefill のプログラムを 1 本にする (T3) | メモリ | 常駐 2.3 GiB → 1.5 GiB 前後 (16320 のみ)。要求ごとの切り替え 35 ms も消える | 小。Q=16 の費用を `TestCallTimes` で測る |
| 5 | 退避した系列と system プロンプトの K/V をディスクに置き、埋め込み表を変換時に書き出す (T4、T5) | メモリとディスク | 写し 105 MB × 2 と埋め込みのキャッシュ (最大 256 MB) がプロセスから消える。OpenCode の初回 prefill 13〜17 秒が消える。実行時に 3.5〜10 GB のチェックポイントが要らなくなる | 小〜中 |
| 6 | 活性を見た表の当てはめ (T8) | 賢さ | lut4 のまま KL を下げる。「重みの誤差が最小の表が最良ではない」(MODEL_ARCHITECTURE.md) の解決策。GPTQ で段を選ぶ部分は入れた (lut6 の KL が半分、16.87)。表そのものの当てはめは残り | 中 |
| 7 | 新しいアーキテクチャ (4〜5 節) | 賢さと長い文脈 | 第一候補は Qwen3.5-2B (Gated DeltaNet 18 層 + 全注意 6 層、Apache 2.0)。次に Mamba2 の Granite 4.0 H-Micro か Nemotron 3 Nano 4B (注意 4 層) | 大。状態型の層 (GDN、Mamba2) の移植 |

モデルを選ぶ基準 (4 節) は、KV キャッシュの*メモリ*ではなく、**長い容量に対して注意を計算する層の数** と **重みのバイト数** と **語彙の大きさ** で、MoE と動的な索引は使えない。「隠れ次元が多い (幅広く浅い)」が良いという直感は正しく、理由は同じバイト数なら層が減って注意の回数と演算の数が減るからで、行列積の効率のためではない (1 トークンの生成は行列とベクトルの積で、幅に依らず帯域で決まる)。

## 2. 費用とメモリの式 (実測から)

**時間**。LFM3-1.7B (lut4 0.87 GB、全注意 6 層、`tile` で K/V を 4 → 16 ヘッドに複製) と Gemma 4 E2B (lut6 1.72 GB、全注意 7 層 (自分の K/V を持つ 3 層と、共有の K/V を読む 4 層)、`SDPAGrouped` で複製なし) の実測を、式に当てはめる。

| | 重み ÷ 65 GB/s | 注意の回数 × 係数 × 容量 | 固定 | 式 | 実測 |
|---|---|---|---|---|---|
| LFM3-1.7B、容量 1K | 13.4 ms | 6 × 0.47 × 1 = 2.8 | 3 | 19.2 | 19.1 ms (16.16) |
| LFM3-1.7B、容量 16K | 13.4 | 6 × 0.47 × 16.3 = 46 | 3 | 62 | 63〜66 ms (16.17) |
| Gemma 4 E2B、容量 1K | 26.5 | 7 × 0.22 × 1 = 1.5 | 3 | 31 | 31.5 ms (16.26) |
| Gemma 4 E2B、容量 16K | 26.5 | 7 × 0.22 × 16.3 = 25 | 3 | 55 | 55 ms (16.26) |

- 重みの項は帯域で決まる (16.12 の 65 GB/s)。lut4 → lut6 → lut8 → fp16 でバイト数どおりに遅くなる (16.26 の表)。
- 注意の項は**容量** (使った量ではない) に比例し、KV の読み出しの帯域でも量子化でも減らない (kv-cache.md 3 節)。係数は注意の形で違う: LFM3 の `tile` の形が 0.47 ms、Gemma 4 の `SDPAGrouped` の形 (問い合わせのヘッドを行に積み、KV ヘッドごとに 1 回の行列積) が 0.22 ms。ヘッドの数や head_dim (LFM3 は 16 × 128、Gemma は 8 × 512) の差より形の差が大きい。
- **KV の共有は注意の回数を減らさない**。Gemma 4 E2B の層 15〜34 は層 13、14 の K/V を読むが、注意の計算は層ごとに行う。共有で減るのは KV のメモリと K/V 射影の重みだけ。
- スライディング注意の層はリング (256〜512) なので容量に依らず、短い畳み込みや SSM の状態も同じ。
- 固定の 3 ms は Core ML の呼び出し (1 回 55 µs) と 1600 前後の演算の分で、小さい。

**メモリ** (16.37 の実測: LFM3-1.7B lut4、2 本で最大 RSS 2.29 GiB、段階 4096 を足した 4 本で 2.73〜2.96 GiB)。

| 何が | 大きさ | 備考 |
|---|---|---|
| プログラム (生成 Q=8 と prefill Q=64) × 段階の数 | 重み 0.81 GiB × 本数 (ファイルの写像なので RSS は倍にはならないが、ANE 側にも載る) | 最大の項。段階 1 つにつき 2 本 |
| KV キャッシュ | 16320 で 190 MiB (LFM3)、32K で 390 MiB、64K で 770 MiB | 全注意の層だけ。リングは固定 |
| 退避した系列の写し (`Session.Save`) | 105 MB × 最大 2 | 並行する会話のため |
| 埋め込みの行のキャッシュ (`chat.Engine.emb`) | float32 で最大 2^26 値 = 256 MB | 行はチェックポイントから `ReadAt` で引く |
| チェックポイント (safetensors) のページキャッシュ | 3.5 GB (LFM3)、10.2 GB (Gemma 4 E2B、音声と画像を含む) | 埋め込みの行のためだけに実行時も開く |
| Core ML / ANE のシステム側 (aned、ANECompilerService) | 未計測 | RSS に入らない。16.12 で計測を予定したまま |

測定中の M3 はスワップが 9〜11 GB に達していた (16.20、16.33)。プロセスの RSS だけでは半分しか見えていないので、システム側の分を測るのが先 (T10)。

## 3. 手法: 少ないメモリで速く、賢く

### T1. LFM3 の注意を `SDPAGrouped` の形にする (速さ)

- 今の LFM3 は `blocks.SDPA` で、K/V を `tile` で 4 ヘッドから 16 ヘッドに複製してから注意を計算する。16K では層ごとに K と V それぞれ 64 MB の写しを毎ステップ作る (6 層で 768 MB)。Gemma 4 は `blocks.SDPAGrouped` (問い合わせのヘッドを系列の軸に積む) で、この形にしたとき 16320 の 1 トークンが 90〜150 ms から 47 ms になった (16.25)。
- 2 節の係数の差 (0.47 対 0.22) がそのままなら、LFM3 の 16K は 13.4 + 6 × 0.22 × 16.3 + 3 ≈ 38 ms。写しの分だけでも 10 ms 前後は減る見込み。Q=8 (投機的デコード) では行が 4 × 8 = 32 になるので、行列積の効率も上がる。
- 16.15 で CPU に落ちた「問い合わせを (kv ヘッド, グループ) の行列にする」形は `npuprobe` の試作で、`SDPAGrouped` (問い合わせのヘッドを系列の軸に積む) はそれとは別の並べ方。Gemma 4 では全演算が ANE に載っている (16.25)。LFM3 (16 Q / 4 KV) で同じになるかは配置の確認が要る。
- やること: `lfm3/build.go` の `m.SDPA` を `m.SDPAGrouped(q, k, v, 1/sqrt(hd), mask)` に (`ChunkedSDPA` にも同じ並べ方を足す)。`TestPlacementVariants` で配置、`TestCallTimes` で費用、`TestCachedMatchesFull` で一致を確かめる。
- 2026-11-02: `-attn grouped` で試した (design.md 16.39)。M3 の c16320 で Q=8 の 1 回が 73.5 → 40.7 ms、NPU に載り、貪欲法の答えは tile と同じ。OpenCode の 7.9K の prefill は 14.6 → 10.9 秒、生成は 60 → 34 ms/token。既定にするかは未決
- 2026-11-02 (lut6 と組み合わせ): lut6 でも効く。Q=8 84.1 → 42.8 ms、Q=64 109.7 → 90.9 ms (design.md 16.41)

### T2. 重みなしの注意プログラムに分け、使っている塊だけ動かす (速さとメモリ)

**実装済み** (2026-11-04、design.md 16.59): `hwnpu-convert -paged 8192`、hwnpu-chat と hwnpu-agent の LFM3 の既定。文脈は 262144。Mac での全体の計測はこれから。

**Gemma 4 と Intel の NPU にも** (2026-11-04、design.md 16.63): Gemma 4 E2B は 8 部分、鍵と値を共有する層は元の層のキャッシュを読むので 1 トークン 6 KiB、文脈 131072。M3 で 81,691 トークンの合言葉を答え、その長さで 96.7 ms/token。分けない形と同じ答えで短い会話も速く、起動は毎回の 50〜100 秒から 2 秒に。LFM3-VL は文脈 128000。`-backend intel` も分けた IR を書く。

kv-cache.md 4 節 (b) の設計。層を「射影と MLP と畳み込み (重みあり、容量に依らない)」と「キャッシュに対する注意 (重みなし、塊ごと)」に分ける。

- **速さ**: 塊 (たとえば 1024) のプログラムを 1 つコンパイルして全部の塊に使い回し、使っている塊だけ呼ぶ。合流 (塊ごとの最大値、指数の和、分子) はホストで行う (1 トークンあたり 16 ヘッド × 128 の小さな計算。Q=8 でも同じ)。費用は容量でなく長さに比例し、16K の容量で 2K しか使っていなければ 1K の段階と同じ速さになる。prefill も同じで、7.4K のプロンプトが 16K 容量の費用 (13 秒) を払わなくなる。
- **メモリ**: 段階 (`-tier`) が要らなくなる。段階は重みを段階ごとに重複させる (0.5〜0.7 GiB、2 本ずつ) ので、その分が消える。16.37 の「中間の段階 12288」も不要。
- **費用**: 呼び出しが増える。全注意 6 層 × (使っている塊の数 + 1) 回 × 55 µs で、8K 使用・塊 1024 なら 6 × 9 × 55 µs ≈ 3 ms。1K では今より遅いが、4K を超えると得。
- **先に測ること**: プログラムの切り替えの費用。重みのある 2 本 (Q=1 と Q=8、各 0.81 GiB) を交互に呼ぶと 1 回 35 ms かかった (16.36) が、鎖の部分 (0.4 GiB × 2) を続けて呼ぶ境目は 0.1 ms (16.12)。前者は ANE に載る重みの量の上限で載せ直しが起きていると見ている。重みなしのプログラムを間に挟んだときにどちらになるかを `npuprobe` で先に測る (合格条件: 境目 1 つ 0.2 ms 以下)。載せ直しが起きるなら、重みなしの注意を「重みありの部分の関数」として同じパッケージに入れる形 (16.13 の multifunction は重みの合計が上限を超える問題だったので、重みなしの関数なら上限に足さない見込み) を試す。
- 重みなしのプログラムはコンパイルも読み込みも速いはず (16.20 の 6〜30 秒は ANE 向けのコンパイルで、重みの処理と見ている。未確認)。
- 2026-11-04 (design.md 16.58、`npuprobe attn`): 重みなしの注意だけを測った。コンパイルは 0.1〜0.3 秒。Q=8 は塊 4096 から ANE (それより小さいと Core ML が CPU に置く)。塊 8192 で 6 層の注意は 1K あたり 1.09 ms、使った長さに比例し、塊ごとにバッファを付け替えても費用は増えず、入力は複製されない (262144 で RSS 3.1 GiB)。重みありとの切り替え (`npuprobe seam`、1.13 GiB の 7 部分と 6 層の注意を同時に読み込み) では載せ直しは起きず、境目の固定費はない (8192 使用で交互と和の差が −0.03 ms)。65536 使用では交互にすると注意が 1〜7% 遅くなる。Q=8 の 1 ステップは 8192 使用で 29.0 ms、65536 で 95.3 ms

### T3. 生成と prefill のプログラムを 1 本にする (メモリ)

- 今は段階ごとに Q=8 (生成) と Q=64 (prefill) の 2 本で、重みが 2 回 ANE に載り、要求ごとに切り替え 35 ms を払う。
- 実測 (16.37、c16320): Q=8 68.5 ms、Q=64 100.8 ms (635 tokens/s)。Q=8 だけで prefill すると 8 ÷ 0.0685 ≈ 117 tokens/s で、OpenCode の 7.4K が 12 秒 → 63 秒になるので、Q=8 だけでは足りない。
- Q=16 は未計測。Q=1 → Q=8 が +15%、Q=8 → Q=64 が +47% なので、Q=16 は +5〜10% の見込みで、prefill は 230 tokens/s 前後 (7.4K が 32 秒)。T2 で prefill が長さに比例するようになれば、7.4K は 16K 容量の費用を払わないので、この見込みはさらに縮む。
- やること: `hwnpu-convert -seq 16` で c16320 を作り、`HWNPU_CALL_PKGS` で `TestCallTimes` を回す。生成の費用が Q=8 の +10% 以内なら、`chat.Prepare` の 2 本を 1 本にする (`-draft 16`)。下書きの幅 15 は n-gram の下書きでは当たりが続かないので (16.37: 7 個と 3 個の差は 10%)、幅は 8 のままで Q=16 の残りを prefill にだけ使う。
- 2026-11-02: Q=16 を計測した (design.md 16.39): 73.4 ms で Q=8 (69.2 ms) の +6%。1 本化は費用の面で成り立つ。grouped の後で測り直す
- 2026-11-02 (grouped): `-attn grouped` の 3 本を `TestCallTimes` で: Q=8 41.4 ms、Q=16 47.8 ms (+15%)、Q=64 82.4 ms。交互の Q=8/Q=64 は 159.5 ms (切り替え約 18 ms)。grouped では Q=16 の 1 本化は見合わない (design.md 16.40 の追記)

### T4. 退避した系列と system プロンプトの K/V をディスクに置く (メモリ、初回の速さ)

- `Session.Save` の写し (16K で 105 MB、最大 2 本) はプロセスに持たなくてよい。`~/Library/Caches/hwnpu` に書き、`Rewind` のときに読む (105 MB は SSD から 0.1 秒以下)。
- 同じ仕組みで system プロンプト (OpenCode は system と道具だけで 7.4K) の K/V を保存すれば、16.32 と 16.37 が挙げた「初回の prefill 13〜17 秒」が消える。鍵はプロンプトのトークン列のハッシュとモデルの名前と容量。
- 保存するときに int8 (KIVI: K は列ごと、V はトークンごとのスケール) にすれば半分。生きているキャッシュを int8 にすると 1 割遅くなる (kv-cache.md 3 節) が、写しには費用がない。
- 2026-11-01: 読んだプロンプトの K/V をディスクに置き、再起動後に共通の先頭から戻す仕組みを実装した (design.md 16.38、`-prompt-cache`)。退避した系列をプロセスから外す分と int8 は未着手。

### T5. 埋め込み表を変換時に書き出す (メモリとディスク)

- 実行時に埋め込みの行をチェックポイントから `ReadAt` で引く (`llm.TokenRow`) ために、3.5 GB (LFM3) や 10.2 GB (Gemma 4 E2B) の safetensors を開いたままにしている。ページキャッシュがほかのものを追い出す。
- `hwnpu-convert` が `embed.bin` (fp16 の行。LFM3 で 65536 × 2048 × 2 B = 256 MB、Gemma 4 E2B で 768 MB、PLE の表 262144 × 8960 は 4.5 GB → 行ごとのスケール付き int8 で 2.3 GB) を書き、`chat.Prepare` は変換が終わったらチェックポイントを消してよいことにする。
- `chat.Engine.emb` の行のキャッシュは float32 (上限 2^26 値 = 256 MB)。fp16 のまま持てば半分、`embed.bin` を写像すればキャッシュ自体が要らない。

### T6. prefill も使っている塊だけ (速さ)

T2 と同じ仕組みで、プロンプトの前半は前半の塊しか見ない。独立の項目ではなく T2 の一部。

### T7. LFM3 を lut6 にする (賢さ。コードの変更なし)

- LFM3-1.7B の lut4 は bf16 に対して KL 0.89 nats、argmax 9/24 (MODEL_ARCHITECTURE.md) で、「富士山 4,076 m」(16.17) や「OpenCode の指示の取り違え」の有力な原因。Gemma 4 E2B は lut4 2.54 → lut6 0.25 → lut8 0.048 で、lut6 で富士山と信濃川に正答した (16.26)。
- 費用: 重み 0.81 → 1.2 GiB (2 部分のまま)、1 トークン +6 ms (1K で 19 → 25 ms、16K で 64 → 70 ms)。常駐は 2 本で +0.8 GiB。
- 混ぜる形: `-format-for` で感度の高い群だけ lut8 にする。候補は出力の rms が 0.008 と小さい `out_proj`、`o_proj`、`w2` (MODEL_ARCHITECTURE.md の誤差の床の話) と出力層 (65536 × 2048、lut4 64 MB → lut8 128 MB、+1 ms)。
- やること: `hwnpu-convert -format lut6` と `hwnpu-check -package` で KL と argmax を測り、`lfm3.Arch.Format` を変える。`chat.Prepare` は形式を名前に入れる (16.34 の `embed-lut8-s128` と同じ) ので古いパッケージと混ざらない。
- 2026-11-02: lut6 を M3 で試した (design.md 16.39): 富士山が 3,776 m になり、ほかは変わらず (『こころ』と川は lut6 でも誤り)。12〜15 tokens/s (lut4 は 18〜24)。KL は参照ロジット (LFM3 のリモートコードが要る) 待ち
- 2026-11-02 (KL): `hwnpu-check -package` で bf16 の参照 (32 トークンの英文) に対し、lut6 は KL 0.232 nats、argmax 23/32、cosine 0.963。lut4 は KL 1.387、argmax 16/32、cosine 0.816。見込みの 0.2〜0.3 どおりで、賢さの面では lut6 に変える価値がある。速さ (12〜15 対 18〜24 tokens/s) と重み (+0.4 GiB) との引き換えは未決 (design.md 16.39)
- 2026-11-02 (grouped と組み合わせ): lut6 + grouped の Q=8 は 42.8 ms で、今の既定 lut4 + tile の 69.2 ms より 38% 速く、lut4 + grouped (41.4) との差は 3%。コーディングの課題は 6 往復で終わる (lut4 は 13〜27)。既定にした: `lfm3.Config.Format` が lut6、`attn` が grouped、パッケージ名は `lfm3-lut6-grouped-c16320-q8` (design.md 16.41)

### T8. 活性を見た表の当てはめ (賢さ)

MODEL_ARCHITECTURE.md の「表の値の選び方」の表は、重みの誤差が最小の表 (k-means) が KL では最悪 (3.68) で、一様な表から始めた k-means が最良 (0.54) だった。重みの誤差ではなく**出力の誤差**を最小にする当てはめにする。どれも変換時だけの費用で、実行時の形は変わらない。

1. **重み付き k-means**: 列 j の重みを E[x_j²] (その入力チャンネルの活性の二乗平均) にして、Σ_j E[x_j²] (w_ij − q_ij)² を最小にする。GPTQ や OBQ のヘッセ行列の対角に当たる。`palette.QuantizeBits` に重みの引数を足すだけで、校正データは `tools/ref/ref_logits.py` の隠れ状態か、`milcpu` で数十文を流して集める。
2. **AWQ 式のスケールの折り込み**: 重要な入力チャンネル j を s_j 倍してから量子化し、s_j の逆数を前の演算に折り込む (q/k/v/gate/up の射影なら直前の RMSNorm の重み、`w2` なら `up_proj` の行、`o_proj` なら `v_proj` の行)。数学的に同じで実行時の費用は 0。ANE の表は 16 行ごとなので、列を伸ばすと行の範囲が広がる取引があり、AWQ と同じく s^α (α を 0〜1 で探索) で決める。
3. **GPTQ 式の誤差の補償**: 列を順に量子化し、誤差をまだ量子化していない列に送る。表引き形式でも使える (GPTVQ)。1 と 2 の後で足りなければ。
- 2026-11-06 (design.md 16.87): 3 を先に入れた。`tools/amd/q4_gptq.py` が CPU で H を集めて GPTQ で各重みの段を選び (表は今の Uniform のまま)、`hwnpu-convert -codes DIR` が読む。LFM3 の lut6 は CPU の見積もりで KL が半分 (README 0.070 → 0.027、wiki40b-ja 0.112 → 0.068)、Go の変換で 0.074 → 0.050。lut4 は 2.7 → 0.66 で、まだ使えない。AMD の q4nx にも同じ道で入れた (E2B の KL 0.120 → 0.092、日本語 0.221 → 0.081)。1 (重み付き k-means で表そのものを合わせる) と 2 (AWQ 式のスケール) は試していない。lut4 を使える水準に近づけるなら次はこれら

### T9. 下書きモデルによる投機的デコード (速さ。優先度は低い)

- Q=8 の検証は Q=1 の +15% でほぼ無料 (16.36) なので、下書きの採用率が n-gram の 2〜3 割から 6〜8 割に上がれば生成は 2 倍を超える。同じトークナイザーの小型が要る: Qwen3.5-0.8B → 2B/4B、Gemma 4 E2B → E4B、Granite 4.0 H-350M → H-Micro。LFM3 には小型がない。
- 懸念は、下書きモデルの重みが主モデルと ANE の載る量を取り合うこと (T2 の 35 ms の切り替えと同じ現象)。M3 では主モデルが 0.8 GiB のときに限って試す価値があり、M4 (上限 2 GB) 向け。

### T10. ANE 側のメモリの計測 (観測)

`npuprobe split` に aned と ANECompilerService の `phys_footprint` を足す (16.12 の未確定事項)。T2 と T3 の効果はこれがないと半分しか見えない。

- 2026-11-02: `footprint` で hwnpu-chat だけ測った (design.md 16.39): 読み込み直後に 2 本 1,968 MB、4 本 2,807 MB。aned は sudo が要り未計測

### 採らないもの

- KV の量子化を生きているキャッシュに使う: メモリは半分になるが 1 割遅く (kv-cache.md 3 節)、KV は常駐の 1 割に満たない。写し (T4) にだけ使う。
- MLState: 速さは同じで CPU の使用率 0.15 コアが減るだけ (16.15)。電力のためなら価値がある。
- 中間の段階 (12288): T2 で不要になる。
- PagedAttention、動的な形: `gather` が CPU に置かれ、形が固定でない (kv-cache.md 2 節)。

## 4. モデルを選ぶ基準

2 節の式から、候補を次の順に見る。**要確認** の値は 7 節の二次情報。

| 基準 | 理由 | 目安 |
|---|---|---|
| **長い容量に対して注意を計算する層の数** | 注意の項は層の数 × 容量に比例し、KV の共有では減らない。T2 (使っている塊だけ) を入れれば「長さ × 層の数」になり、重要さは下がるが消えない | LFM3 6、Gemma 4 E2B 7 (共有を含む)。4 以下なら 16K でも注意の項が 15〜30 ms に収まる |
| **全注意でない層の種類** | スライディング窓 (リング)、短い畳み込み (LFM3)、Mamba2、Gated DeltaNet、RWKV の状態は固定の大きさで、容量に依らない。移植の作業はこの種類で決まる (5 節) | 窓と畳み込みは実装済み。Mamba2 と GDN は要素ごとの演算と小さな行列積で書けるので ANE 向き。Mamba-1 は塊分けが重い |
| **重みのバイト数 (ANE に載る分)** | 65 GB/s。表の形式は lut4 (0.5 B/重み)、lut6 (0.75 B)、lut8 (1 B)。M3 は 1 部分 1 GiB、16 GB の機械なら常駐 (本数 × バイト) を 3 GiB 以内に | 1.5 GB (lut6 で 2B) なら 23 ms。4B 級は lut6 で 3 GB、M3 では 1 本でも 46 ms で、2 本常駐は 6 GB |
| **lut の耐性** | サンドイッチ正規化の層は出力の大半を層の中で作り直すので、4 ビットの誤差がそのまま出る (Gemma 4: lut4 KL 2.54)。QAT の Q4_0 の格子は表せない (16.25) | LFM3 系は lut4 で動く (ただし T7)。Gemma 系は lut6。埋め込みモデルは lut8 |
| **語彙の大きさ** | 出力層のバイト数が語彙 × 隠れ。262144 × 1536 の lut6 は 300 MB で 1 トークン 4.6 ms、248320 × 2048 は 380 MB で 6 ms。262144 の軸はソフトキャップが CPU に置かれ、(16, 16384) に畳む手間がある (16.25) | 65536 (LFM3) が最も安い。128K で +2 ms、250K で +6 ms |
| **幅広く浅い** | 同じバイト数なら層が減り、注意の回数と演算の数 (1 層 70 前後) が減る。prefill (Q=64) の行列積の効率も上がる。1 トークンの生成は行列とベクトルの積で幅に依らない | 24 層 × 2048 は 36 層 × 1536 より有利。MobileLLM 型の「深く細い」設計は ANE には不利 |
| **KV ヘッドの数と head_dim** | KV のメモリと `tile` の写しの量。`SDPAGrouped` の形なら少ないほど問い合わせの行が増えて効率が上がる | 1〜2 KV ヘッドが最良。16384 の軸の上限に head_dim は関係ない |
| **使えない構造** | 形が固定で、動的な索引は CPU 行き | MoE (ルーティングは `gather`。全専門家を密に計算する形は重みのバイト数どおりに遅い)、学習済みの疎な注意 (MiniCPM4 の InfLLM v2、DeepSeek の DSA)、MTP の頭、画像の可変の形 |
| **ライセンスと言語** | hwnpu-chat の目録に載せるため | Apache 2.0 (Qwen、Granite、Jamba、RWKV)、MIT (Phi)、各社の条件 (Gemma、LFM、Nemotron)。日本語は Gemma 4、Qwen3.5、PLaMo 3 が強い |

Apple の on-device モデル (Foundation Models、約 3B) は後半 37.5% の層の K/V 射影を外して前半のを使い、2 ビットの QAT で、まさにこの基準で作られている (kv-cache.md 6 節)。重みは公開されていないので、公開物で最も近いのは Gemma 4 E2B (PLE と KV の共有)。

## 5. 候補

見込みの計算: 重み (lut6) ÷ 65 GB/s + 注意の回数 × (0.22〜0.47 ms) × 容量 / 1K + 3 ms。0.22 は `SDPAGrouped` の形、0.47 は `tile` の形 (2 節)。T2 を入れると 16K の列は使っている長さの分だけになり、1K の列に近づく。パラメータ数は ANE に載る分 (埋め込みの表は除き、出力層は含む) の概算。

### 5.1 合う候補

| モデル | 長い容量の注意 / 窓 / 状態型 | KV (1 トークン) | ANE の重み (lut6) | 1K / 16K の見込み (M3) | 移植 | 評価 |
|---|---|---|---|---|---|---|
| **LFM3-1.7B** (実装済み) | 6 / 5 (窓 256) / 畳み込み 13 | 12 KiB | lut4 0.87 GB (lut6 1.29 GB) | 実測 19 / 64 (lut6 で 25 / 70) | — | 基準。T1 で 16K が 40 ms 台、T7 で賢く |
| **Gemma 4 E2B** (実装済み) | 7 (自分 3 + 共有 4) / 12 (窓 512) / 共有 20 | 6 KiB | 1.72 GB | 実測 31 / 55 | — | 共有で KV は最小。PLE の表 4.5 GB をホストに置く (T5) |
| Gemma 4 E4B (変換済み、未計測) | 約 7〜8 / 約 20 / 共有 18 (**要確認**) | 6〜8 KiB | 約 3.1 GB、4 部分 (**要確認**) | 約 53 / 76〜112 | — | 2 本常駐で 6 GB。M3 16 GB では窮屈で、M4 (2 GB の部分) と 24 GB 以上向け |
| **Qwen3.5-2B** | 6 (全注意、8 Q / 2 KV × 256) / 0 / Gated DeltaNet 18 (**要確認**: 24 層、hidden 2048、vocab 248320、GDN の値ヘッド 16) | 12 KiB + GDN の状態 9 MiB | 約 1.3 GB (本体 1.2B + 出力層 0.5B) | 24〜26 / 44〜68 | GDN (5.3 節)。全注意は QK-norm、部分 RoPE、出力ゲート (要素ごと) | **第一候補**。公称では同級で最上位、Apache 2.0、文脈 262K、画像も読める (テキストだけ移す)。語彙 248K の出力層が 6 ms。0.8B と 4B が同じトークナイザー (T9) |
| Qwen3.5-4B (移植済み、CPU で照合、ANE は未計測。design.md 16.62) | 8 (16 Q / 4 KV × 256) / 0 / GDN 24 (config で確認: 32 層、hidden 2560、GDN の鍵ヘッド 16 と値ヘッド 32 × 128) | 32 KiB + GDN の状態 24 MiB | ANE に載る分 4.2B (出力層 0.64B を含む): lut6 約 3.2 GB、4 部分 (lut4 約 2.1 GB) | 51 前後 / 80〜110 | 同上 | 賢さの上限が高いが M3 では 1 本 51 ms、2 本常駐 6.3 GB。M4 向け |
| Qwen3.5-0.8B | 6 / 0 / GDN 18 (**要確認**) | 6〜12 KiB | 約 0.45 GB (出力層が半分) | 11〜13 / 31〜56 | 同上 | 単独では弱い。4B の下書き (T9) か、判定の用途 |
| **Granite 4.0 H-Micro** (3B) | 4 / 0 / Mamba2 36 (40 層、hidden 2048、vocab 100352、位置埋め込みなし (NoPE)、**要確認**) | 約 8 KiB + Mamba2 の状態 36 MiB | 約 2.25 GB、3 部分 (lut4 1.5 GB、2 部分) | 39〜40 / 52〜68 | Mamba2 (5.3 節)。NoPE なら RoPE の表が要らない | 注意 4 層は候補中で最少級。Apache 2.0、企業向けの調整 (道具の呼び出し)。日本語は弱め |
| Granite 4.0 H-1B (約 1.5B) | 4 (**要確認**) / 0 / Mamba2 | 8 KiB | 約 1.1 GB | 21〜22 / 34〜50 | 同上 | H-Micro と同じ構造の小型。H-350M は下書きに |
| **Nemotron 3 Nano 4B** | 4 / 0 / Mamba2 21 + MLP 17 (42 層、vocab 131072、2026-03) | 約 16 KiB (**要確認**) + 状態 21 MiB | 約 3.0 GB、4 部分 (lut4 2 GB) | 50〜51 / 64〜80 | Mamba2 | 注意 4 層で 4B 級。NVIDIA Open Model License。M3 では重く、M4 向け |
| Jamba Reasoning 3B | 2 / 0 / Mamba 26 (28 層、**要確認**: Mamba-1 か 2 か) | 約 8 KiB | 約 2.25 GB | 38〜39 / 45〜53 | Mamba-1 なら 5.3 節の重さ | 注意 2 層で 16K がほぼ平ら。Apache 2.0。Mamba の版を確かめてから |
| PLaMo 3 NICT 2B | 交互の窓付きと全注意 (比率と窓は **要確認**) / — / 0 | 要確認 | 約 1.5 GB | 27〜32 / 40〜57 (全注意 4 層なら)、69〜118 (12 層なら) | 窓と全注意だけなので `blocks` で足りる。新しい arch パッケージのみ | 日本語の base モデル (指示調整版の有無は要確認)。移植は最も軽い |
| Hymba-1.5B | 3 / 29 (窓 1024) / Mamba-1 32 (並列ヘッド、KV の共有、meta トークン 128、**要確認**) | 約 3 KiB | 約 1.1 GB | 21 / 31〜43 | Mamba-1 × 32 層 (5.3 節) | 設計はこの基準の見本 (KV は Llama 3.2 1B の 1/11.7)。2024 年の質。NVIDIA のライセンス |
| RWKV-7 G1 2.9B | 0 / 0 / RWKV 32 | 0 + 状態 10 MiB | 約 2.2 GB | 37 / 37 | 時間混合 (delta rule、5.3 節と同型) と token shift | 文脈に依らず平ら。Apache 2.0。質は同級の Transformer より下と見る (要実測) |

### 5.2 合わない例 (全注意の Transformer と、毎層に注意のある並列ハイブリッド)

| モデル | 注意の回数 | KV (1 トークン) | ANE の重み (lut6) | 1K / 16K の見込み | 理由 |
|---|---|---|---|---|---|
| Qwen3-1.7B | 28 | 112 KiB | 1.3 GB | 29〜36 / 123〜237 | 全層が全注意。4K を超えると注意の項が重みの項を超える |
| Llama 3.2 1B / 3B | 16 / 28 | 32 / 112 KiB | 0.93 / 2.4 GB | 21〜25 / 74〜140 (1B) | 同上。1B は層が少ない分まし |
| SmolLM3-3B | 36 | 72 KiB | 2.3 GB | 46〜55 / 167〜314 | 深く細い (36 層 × 2048) |
| Phi-4-mini (3.8B) | 32 | 128 KiB | 2.85 GB | 54〜62 / 162〜292 | 同上。語彙 200K |
| Phi-4-mini-flash-reasoning (3.8B、SambaY) | 約 9 (全注意 1 + 共有の K/V を読む cross-attention 約 8) | 1 層分 (最小級) | 2.85 GB | 49〜51 / 79〜116 | KV の*メモリ*は最小だが、注意の*計算*は 9 回残る。Mamba-1。数学向けの調整 |
| Falcon-H1-1.5B-deep / 3B | 66 / 32 (毎層に注意 6〜10 Q / 2 KV と Mamba2 が並列) | 66 / 32 KiB | 1.16 / 2.4 GB | 36〜52 / 258〜527 (1.5B-deep)、47〜55 / 155〜285 (3B) | 並列ハイブリッドは毎層が注意。2K までなら良い |
| Gemma 3 1B | 4 (26 層の 5:1、4 Q / 1 KV × 256) | 4 KiB | 0.75 GB | 15〜16 / 29〜45 | 合うが弱い。Gemma 4 E2B が上位互換 |
| MoE (Gemma 4 26B-A4B、Qwen3.5-35B-A3B、LFM2-8B-A1B、Granite 4.0 H-Tiny、gpt-oss-20b) | — | — | 全専門家 | — | ルーティングが `gather`。密に計算すれば重みのバイト数どおり (8B で 4 GB、60 ms 超) |

### 5.3 移植の見積もり

**Gated DeltaNet** (Qwen3.5、Qwen3-Next、Kimi Linear)。1 トークンの更新は、ヘッドごとの状態 S (d_k × d_v、Qwen3.5-2B で 128 × 128 と見る) に対して

> S ← S · exp(g) · (I − β k kᵀ) + β v kᵀ、o = Sᵀ q

で、g は負の減衰 (−exp(A_log) · softplus(a_proj(x) + dt_bias))、β = sigmoid(b_proj(x))、k は L2 正規化。すべて要素ごとの演算と (ヘッド, 128, 128) の行列積で、形は固定。q/k/v の前の短い畳み込み (幅 4) は LFM3 の `win_*` と同じ窓で持てる。出力は z でゲートした RMSNorm と射影。状態は層ごとに 16 × 128 × 128 × 2 B = 512 KiB で、`kv_*` と同じくホストの入出力のバッファに置く。

prefill (Q=64) は塊の形が要る。塊の中は (I + tril(β K Kᵀ))⁻¹ の三角行列が要り、strictly lower triangular な N に対して (I + N)⁻¹ = (I − N)(I + N²)(I + N⁴)…(I + N³²) なので、64 × 64 の行列積 6 回で書ける (N⁶⁴ = 0)。塊の間は状態の受け渡しで、行列積だけ。要素ごとの演算と行列積に閉じるので ANE に載る見込みが高い。64 ステップを展開する形 (64 × 8 演算 × 18 層 ≈ 9000 演算) は演算の数が多く、第二案。

**Mamba2 (SSD)** (Granite 4.0、Nemotron 3 Nano、Falcon-H1)。1 トークンの更新はヘッドごとの状態 h (head_dim × d_state、64 × 128) に対して h ← exp(Δ A) h + Δ (x ⊗ B)、y = h C + D x で、Δ = softplus(dt + bias) はヘッドごとのスカラー。要素ごとの演算と小さな行列積。prefill の塊は、減衰の累積和 (三角の 1 行列との行列積) から L = exp(segsum) を作って tril で隠し、Q Kᵀ の形の注意と同じ流れで書ける。fp16 の範囲に気を付ける (累積和の指数は最大値を引いてから取る。塊 32 でも可)。B と C はグループ (GQA と同じ) で、`tile` でなく `SDPAGrouped` と同じ並べ方にする。

**Mamba-1** (Jamba の一部、Hymba、Phi-4-mini-flash の self-decoder)。減衰が (チャンネル, 状態) ごとに違うので、塊の形は d_inner × d_state × 64 × 64 の中間 (4096 × 16 で 512 MB) を作るか、64 ステップを展開するかになり、重い。Mamba2 より後回し。

**RWKV-7**。時間混合は一般化 delta rule で GDN と同じ型 (状態 64 × 64 × ヘッド)、token shift は `win_*` の窓。チャンネル混合は MLP。GDN の塊の形がそのまま使える。

**YOCO / GMU** (Phi-4-mini-flash)。cross-attention は外から K/V を受ける普通の `SDPA` で、鎖が部分の間で共有の K/V を渡す仕組み (Gemma 4 の `shared_k_*`、16.25) がそのまま使える。GMU は要素ごと。

**交互の窓付きと全注意** (PLaMo 3)。`blocks` にすべて揃っており、arch パッケージ (config の読み方と重みの名前) だけ。`port-architecture` スキルの手順で、`hwnpu-check` の照合まで 1 日の作業。

### 5.4 推奨

1. **今のモデルで**: T7 (lut6)、T1 (`SDPAGrouped`)、T2 (塊だけ) の順。これだけで LFM3-1.7B は 16K で 40 ms 前後、4K 使用なら 25 ms 前後、常駐は 2 本分になる見込み。
2. **次のアーキテクチャは Qwen3.5-2B**。Gated DeltaNet の移植 (5.3 節) は要素ごとの演算と固定の形の行列積に閉じ、ANE 向きで、同じ移植が Qwen3.5 の 0.8B〜9B と Qwen3-Next、Kimi Linear、RWKV-7 に効く。全注意 6 層は LFM3 と同じなので、長い文脈の速さは T1 と T2 が決め、賢さと文脈 262K とライセンスで上回る。語彙 248K の出力層 (6 ms) が代償。
3. **その次は Mamba2** (Granite 4.0 H-Micro、Nemotron 3 Nano 4B)。注意 4 層で 16K でも注意の項が 15〜30 ms。Granite は Apache 2.0 で 3B が M3 に載り、Nemotron は 4B で M4 向け。
4. **日本語を優先するなら PLaMo 3 NICT 2B** を先に試す。移植が最も軽い。指示調整版と層の比率を確かめてから。
5. Gemma 4 E4B は変換の仕組みが揃っているので M4 の利用者に計測を頼む。M3 では E2B に留める。

## 6. 確かめ方

- 時間の式の係数: `TestCallTimes` (`HWNPU_CALL_PKGS`) で 1 本ごとの 1 回の費用を測り、2 節の表に行を足す。T1 と T3 の判断はこれで決まる。
- メモリ: `npuprobe split` にシステム側の `phys_footprint` を足す (T10)。スワップが 1 GB 未満の機械で測る (16.36 の条件)。
- 賢さ: `hwnpu-check -package` の KL と argmax、`tools/ref/eval_logits.py`。チャットの体感は 16.26 の 4 問 (富士山、17 × 23、『こころ』、信濃川) を揃えて比べる。
- 新しいアーキテクチャ: `.claude/skills/port-architecture` の合格条件 (fp64 の参照との一致、改変の検出、`hwnpu-check` の層ごとの照合、ANE の配置)。GDN と Mamba2 の塊の形は、`TestCachedMatchesFull` と同じく「1 トークンずつ」と「Q=64 の塊」の一致で確かめる。

## 7. 参照

実測は design.md の 16.12〜16.37 と [kv-cache.md](kv-cache.md)。候補モデルの構成は次の二次情報による (Hugging Face に届かない環境で調べた。config.json で確かめること)。

- Qwen3.5: [Qwen/Qwen3.5-2B の model card](https://huggingface.co/Qwen/Qwen3.5-2B) (6 × (3 × GDN → 1 × 全注意))、[Qwen3.5: Nobody Agrees on Attention Anymore](https://medium.com/@mlabonne/qwen3-5-nobody-agrees-on-attention-anymore-4709e1bd014b)、[Qwen 3.5: Architecture and Model Selection](https://www.overshoot.ai/blogs/qwen3.5-on-overshoot) (3:1 の比率)、派生の config の写し ([RinggAI/Transcript-Analytics-Qwen3.5-2B](https://huggingface.co/RinggAI/Transcript-Analytics-Qwen3.5-2B/blob/main/config.json): hidden 2048、24 層、8 Q / 2 KV × 256、vocab 248320、linear_num_value_heads 16、full_attention_interval 4)、[Small LLMs That Fit in 8GB (2026)](https://pinggy.io/blog/small_llms_that_fit_in_8gb_memory/) (4B の構成)
- Gemma 4: [Gemma 4 Technical Report](https://arxiv.org/pdf/2607.02770) (E2B 20/35、E4B 18/42 の KV 共有)、[fairseq2 の gemma4](https://facebookresearch.github.io/fairseq2/nightly/reference/fairseq2.models.gemma4.html) (窓 512、E2B 4:1、E4B 5:1)
- Granite 4.0: [IBM の発表](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models) (Mamba2 9:1、H-Micro 3B dense)、[granite-4.0-h-micro の一覧](https://en.immers.cloud/ai/ibm-granite/granite-4.0-h-micro) (40 層、全注意 4)、[Granite 4.0 Nano](https://www.llmreference.com/model-family/granite-4)
- Nemotron 3 Nano 4B: [model card (PDF)](https://developer.download.nvidia.com/assets/ace/model_card/Nemotron_3_Nano_4B.pdf)、[NVIDIA-Nemotron-3-Nano-4B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8) (Mamba-2 と MLP に注意 4 層)、[Nemotron 3 の vLLM の記事](https://blog.vllm.ai/2025/12/15/run-nvidia-nemotron-3-nano.html)
- Jamba Reasoning 3B: [model card](https://huggingface.co/ai21labs/AI21-Jamba-Reasoning-3B)、[比較記事](https://tinyweights.dev/posts/jamba-reasoning-3b-vs-smollm3-3b/) (28 層、Mamba 26 + 注意 2)
- Phi-4-mini-flash-reasoning: [Azure の記事](https://azure.microsoft.com/en-us/blog/reasoning-reimagined-introducing-phi-4-mini-flash-reasoning/)、[model card](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning) (共有 KV と全注意 1 層、GMU)
- Falcon-H1: [技術報告](https://arxiv.org/abs/2507.22448) (1.5B-deep 66 層、6 Q / 2 KV、3B 32 層、10 Q / 2 KV)
- Hymba: [論文](https://arxiv.org/abs/2411.13676)
- PLaMo 3: [pfnet の plamo-3 コレクション](https://huggingface.co/collections/pfnet/plamo-3) (NICT 2B / 8B / 31B base)、[vLLM の plamo3](https://docs.vllm.ai/en/v0.15.0/api/vllm/model_executor/models/plamo3/) (layer_types に sliding と full)、[PLaMo 2 Technical Report](https://arxiv.org/pdf/2509.04897) (PLaMo 2 は Samba 型)
- RWKV-7: [RWKV Evals](https://wiki.rwkv.com/basic/RWKV-Evals.html)
- LFM3: [6 モデルの発表](https://aiweekly.co/alerts/liquid-ai-opens-six-model-lfm3-family-at-256k-context) (すべて 1.7B base から。小型や大型はない)
- 変換で混成にする研究: [Morphing into Hybrid Attention Models (FlashMorph)](https://arxiv.org/pdf/2606.30562) (学習済みの Transformer の層を選んで線形注意に置き換える。TransMLA と同じく追加学習が要り、Go の変換器の範囲外。将来、全注意の Transformer を候補に戻す道)
- 手法: [AWQ](https://arxiv.org/abs/2306.00978)、[GPTQ](https://arxiv.org/abs/2210.17323)、[GPTVQ](https://arxiv.org/abs/2402.15319)、[Gated Delta Networks](https://arxiv.org/abs/2412.06464)、[Transformers are SSMs (Mamba2、SSD)](https://arxiv.org/abs/2405.21060)、[YOCO](https://arxiv.org/abs/2405.05254)、[Apple Intelligence Foundation Language Models 2025](https://arxiv.org/abs/2507.13575)
