# 参考にした NPU の論文と、BonsaiAI で確かめた結果

[English](reference-paper.md)

作成: 2026-11-21。2026-11-20 からの論文の調査 ([handoff-amd-papers.md](handoff-amd-papers.md)) で拾った論文について、何を言っていて、BonsaiAI の実機で当てはまったかをまとめる。細かい数字と試し方は [docs/design/README.ja.md](design/README.ja.md) の各節にある。ここは一覧と、そこへの道案内。

- AMD の NPU (XDNA2、Ryzen AI 9 365) の最適化の候補が中心。MoE の論文と、同じ時期に M2 で確かめた Apple の Neural Engine の論文も載せる
- MoE の論文の全体の表は [docs/design/performance.ja.md](design/performance.ja.md)にある。ここでは AMD に関係するものだけ

**論文の確かめた度合い** (handoff-amd-papers.md と同じ記号)
- ✓: 本文 (HTML) を調べた
- (a): 要旨だけで見た
- †: 一般に知られた手法で、本文を読んでいない

2026 年のものは査読前のプレプリントが多い。

**結果の欄の言葉**
- **当てはまった**: 論文の主張が BonsaiAI の実機でも成り立ち、それを使って速くしたか、直した
- **すでに同じ**: BonsaiAI がもともと同じことをしていた
- **一部**: 当てはまったところと、違ったところがある
- **未着手**: まだ試していない

## 一覧

| 論文 | 言っていること | 結果 | BonsaiAI の節 |
|---|---|---|---|
| HeteroMosaic 2607.12839 ✓ | データパスの設定のし直し (PDI の読み直し) は高い。DMA の切り替えはほぼ無料 | **当てはまった**。1 回 40 µs + 2.3 µs/KB。密なモデルのデコードの時間の 7 割前後がこれ。演算をまとめて LFM3 11.2 → 16.9 tok/s ほか | [amd](design/amd.ja.md)、[amd-llm](design/amd-llm.ja.md) |
| 2606.07586 ✓ (XDNA2 の Llama-3.2-1B) | XRT のコンテキストを使い回す。BO をゼロコピーで回す。NPU だけが使うバッファの同期を省く | **すでに同じ**。IRON の経路は使う範囲だけを同期していた | [amd](design/amd.ja.md) の最後 |
| 2512.13282 ✓ (Ryzen AI の 2 世代の GEMM) | 多次元の DMA の番地で並べ替え、ホストで重みをタイルにしない。出力の C のバッファを 1 つに | **未着手**。1 次元 1023 語の制限は、[amd-llm](design/amd-llm.ja.md) でも当たった | [moe](design/moe.ja.md)、[amd-llm](design/amd-llm.ja.md) |
| Asymmetric Tile Buffering 2511.16041 (a) | 入力と出力でタイルの大きさを変える (XDNA2 の BFP16 の GEMM で 4.54×) | **未着手** | |
| 2609.21264 ✓ (IRON と MLIR-AIR の FlashAttention) | QKᵀ をコアのメモリに置いて融合。XDNA2 で効く | **未着手**。融合で設定のし直しが減る点は [amd](design/amd.ja.md) と同じ向き | |
| STEEL 2607.09385 (a) | 因果マスクの疎を考えた FlashAttention の段の配置 | **未着手** | |
| Zen-Attention 2508.17593 (a) | 注意を畳み、DRAM の帯域を減らす | **未着手** | |
| MR-GPTQ 2509.23202 (a)、QuaRot / SpinQuant † | 量子化の前にアダマールで回して外れ値をならす | AMD は**未着手**。Apple の lut で試して KL が約半分 (製品には未投入) | [apple-neural-engine](design/apple-neural-engine.ja.md) |
| TileFuse 2606.11357 ✓ | BFP16 のカーネルは LLM の端から端までには精度が足りない | **確認だけ**。BonsaiAI は bfp16 の所を KL かコサインで確かめている | [moe](design/moe.ja.md)、[image-amd](design/image-amd.ja.md) |
| ENEC 2604.03298 (a, Ascend) | 重みの可逆圧縮で転送の律速を外す | **未着手**。今のデコードは帯域より設定のし直しで決まるので、後回し | |
| 2505.16056 (Not All Models Suit Expert Offloading) | 活性の専門家の約 2 倍のキャッシュで足りる。LFU が LRU より良い | **一部**。AMD の置き場 (`--expert-places`) を使用回数で選ぶ形に入った | [moe](design/moe.ja.md)、[moe](design/moe.ja.md) |
| EStream 2609.06551 (Hexagon) | 専門家を銀行 × 枠の置き場に。番地は 先頭 + 枠 × 大きさ。O_DIRECT で読む | **当てはまった**。AMD の置き場の設計の下敷き。gpt-oss-120b は O_DIRECT で読む | [moe](design/moe.ja.md)、handoff-amd-moe.md |
| Fate 2502.12224 | 次の層のゲートを今の層の入力で予測して先読み | **当てはまった**。Apple の gpt-oss-20b で上位 4 個に 86〜87 %、AMD の gpt-oss-120b で外れ 9.8 → 3.1 % | [docs/design/performance.ja.md](design/performance.ja.md)、handoff-amd-moe.md 4 |
| NPUMoE 2604.18788 | プロンプトだけ Neural Engine に。経路と集めは CPU | **すでに同じ**。Neural Engine に gather を載せない判断 ([moe](design/moe.ja.md)) と合う | [moe](design/moe.ja.md) |
| Pre-gated MoE 2308.12066 ほか | 次のゲートを 1 層前に | 背景として読んだだけ | [docs/design/performance.ja.md](design/performance.ja.md) |
| 2606.22283 (Neural Engine のリバースエンジニアリング) | 出力の 2^15、幅の slice、softplus などの数値の規則 | **一部**。規則はおおむね M2 でも成り立つ。幅の slice は条件が論文より狭い | [apple-neural-engine](design/apple-neural-engine.ja.md) |
| 2611.19101 (Neural Engine の roofline) | 遅い活性の経路 (約 14 GB/s)、実行時の重みは定数の半分の速さ | **当てはまった**。M2 で要素ごとの演算 16〜20 GB/s、実行時の重み 26 GB/s と 50〜52 GB/s | [apple-neural-engine](design/apple-neural-engine.ja.md) |
| ANEForge 2606.17090 | 直接の呼び出しで約 70 µs | **一部**。Core AI を通すと約 160〜175 µs | [apple-neural-engine](design/apple-neural-engine.ja.md) |

## AMD の NPU

### HeteroMosaic 2607.12839 ✓ — 当てはまった

**言っていること**
- データパスの設定のし直し (PDI の読み直し) は高い。DMA の設定の切り替えはほぼ無料
- 同じ xclbin の操作をまとめる。NPU を扱うスレッドは 1 本にする

**確かめたこと** ([amd](design/amd.ja.md))
- BonsaiAI の full ELF では、演算子が変わるたびに `load_pdi` と配列の設定 (コアのプログラムメモリを含む) が制御コードに入る
- 1 回の費用は 40 µs + 2.3 µs/KB。[moe](design/moe.ja.md) の「違う種類の演算子 1 個 約 80 µs」の中身はこれだった
- 見積もりは実機のデコードと 1 割以内で合った (LFM3-1.7B 92.6 ms に対して 98 ms、Gemma 4 E2B 137 ms に対して 138 ms)
- gpt-oss-20b のデコードは常駐の 1 演算 (`GptOssStep`) なので当てはまらない

**使ったこと**: 1 層の演算の数を減らした。並べ替えはほとんど効かない (依存があるので、Qwen3.6 で 1 トークン 2 % まで)。

| 節 | やったこと | 結果 |
|---|---|---|
| [amd-llm](design/amd-llm.ja.md) | GDN の .bss の配列を Worker の局所バッファに (0 の塊を消す) | Qwen3.5-0.8B 13.2 → 14.1 tok/s、ロジットはビット単位で同じ |
| [amd-llm](design/amd-llm.ja.md) | LFM3 のデコードに `ShortConvStep`、`SandwichAdd`、`QGEMVGlu` | LFM3-1.7B 11.2 → 15.6 tok/s |
| [amd-llm](design/amd-llm.ja.md) | Gemma 4 の per-layer input をまとめて、`SandwichAdd` | Gemma 4 E2B 7.7 → 9.3 tok/s |
| [moe](design/moe.ja.md) | t3/qwen36-integrate の取り込み (GDN の状態の枠を直接) | Qwen3.5-0.8B 13.8 → 18.7、Qwen3.6 8.5 → 8.9 tok/s |
| [amd](design/amd.ja.md) | QGEMV を 1 列 1 コアに (設定がほぼ半分)、sandwich を GEMV の中で | LFM3-1.7B 15.6 → 17.2、Gemma 4 E2B 9.3 → 9.7 tok/s |
| [amd-llm](design/amd-llm.ja.md) | プロンプトの短い畳み込みを 1 演算子に (`ShortConvChunk`、`GdnConvChunk`)。LFM3 のデコードの状態の Copy をなくす | LFM3 16 トークンのプロンプト −13 %、Qwen3.5-0.8B −25 %。LFM3 デコード 16.5 → 16.9 tok/s |

**当てはまらなかったこと、使えなかったこと**
- `--reconfig-teardown`: 小さな設定では速いが、大きな設定 (GEMM) で遅くなる
- `--load-pdi-to-ctrl-pkt` (設定を DMA で流す): aiecc が「operand #0 does not dominate this use」で止まる
- 「NPU を扱うスレッドを 1 本」は試していない

**残り**: デコードの残りの設定のし直しを大きく減らすには、層を 1 つの配列にする (`GptOssStep` の形)。Qwen3.6 は GDN の演算子の設定 374 KB × 30 層が大きい。

### 2606.07586 ✓ — すでに同じ

**言っていること**: XDNA2 で Llama-3.2-1B のデコード 10.8 tok/s。XRT のコンテキストを使い回し、層ごとの BO をゼロコピーで回す。NPU だけが読み書きするバッファは、ホストとの同期を省く。

**確かめたこと** ([amd](design/amd.ja.md) の最後): IRON の経路 (`internal/llm/iron` の `readRegion`、`writeRegion`、`writeCopies`、`rows.go`) は、すでに使う範囲だけを同期していた。バッファ全体の同期が残るのは、読み込みのときと IRON でない経路 (`internal/llm/chain`、小さなバッファ) だけ。何も変えていない。

### 2512.13282 ✓ (Ryzen AI の 2 世代の GEMM) — 未着手

**言っていること**
- XDNA2 で int8 38 TOPS、bf16 14.7 TOPS。bf16 は bfp16 のデータパスに通す
- 多次元の DMA の番地で並べ替え、ホストで重みを前もってタイルにしない
- 出力の転送が償却できるとき、C のバッファを 1 つにする (+13〜18 %)

**BonsaiAI との関係**
- bfp16 のデータパスは、DiT と gpt-oss のプロンプトの専門家ですでに使っている
- DMA の 1 次元は 1023 語までで、重みを消費順に並べ替える 4 次元の TAP が通らない ([moe](design/moe.ja.md))。そのため専門家の表をホストで並べ直している
- [amd-llm](design/amd-llm.ja.md) の `ShortConvChunk` も、行の数で DMA を切る `length_parameter` がこの制限に当たり、塊の全部の行を動かしている
- 1023 語を超える 1 次元を 2 つの次元に割れるかは、まだ試していない

### Asymmetric Tile Buffering 2511.16041 (a) — 未着手

**言っていること**: タイルを入力と出力で違う大きさで持つ。XDNA2 の BFP16-BF16 の GEMM で 24.6 TFLOPS、4.54×。

**対象の候補**: 画像の DiT の GEMM (Anima、FLUX)、gpt-oss のプロンプトの専門家。[amd-llm](design/amd-llm.ja.md) の「IRON の GEMM は呼び出しの行に関係なく重みを全部の行の単位に流す」(短いプロンプトの 6 割) とも関係するかもしれない。

### 注意の融合: 2609.21264 ✓、STEEL 2607.09385 (a)、Zen-Attention 2508.17593 (a) — 未着手

**言っていること**
- 2609.21264: IRON と MLIR-AIR の FlashAttention、128K まで。融合版は XDNA2 で 3.62 TFLOP/s。QKᵀ をコアのメモリに置き、cascade の配線で集める。融合は XDNA2 で効き、XDNA1 ではほとんど効かない
- STEEL: 因果マスクの疎を考えて、パイプラインの段を置く
- Zen-Attention: 注意を畳み、タイルを決めて DRAM の帯域を減らす。注意 最大 4×、端から端まで 最大 32 %

**BonsaiAI との関係**: プロンプトの因果な注意と、DiT の注意が対象。融合は設定のし直しの回数も減らすので、[amd](design/amd.ja.md) の結果とも向きが合う。2609.21264 は IRON で書いているので、コードが公開されていれば今の `iron_mha.py` と比べるのが早い。

### 回転: MR-GPTQ 2509.23202 (a)、QuaRot / SpinQuant † — AMD は未着手、Apple で一部

**言っていること**: 量子化の前に残差の次元をアダマール変換で回し、外れ値をならす。重みに畳み込むので、実行時の演算は増えない。

**確かめたこと** (Apple、[apple-neural-engine](design/apple-neural-engine.ja.md)): Qwen3.5 の lut で KL が約半分。
- Qwen3.5-2B の lut6: 0.056 → 0.027
- Qwen3.5-0.8B の lut6: 0.075 → 0.039
- lut4 は回しても使えない (2B で 0.60 → 0.37)

**AMD に持ってくるときの制約** ([apple-neural-engine](design/apple-neural-engine.ja.md) で分かった)
- 回せるのは、残差を読むものが全部 norm の後の linear で、書くものが全部そのまま足されるモデルだけ
- チャットのモデルでは Qwen3.5 系だけが当たる
- LFM3、Gemma 4、PLaMo 3 は、書く側に重みつきの norm (sandwich) があって回せない
- gpt-oss は専門家も回して量子化し直すことになる

AMD の q4 (Qwen3.6 の HF とのコサイン 0.986〜0.996 の差の大半が q4) に試す価値はあるが、まだ試していない。

### TileFuse 2606.11357 ✓ — 確認だけ

**言っていること**: XDNA2 の W4A16/W8A16 の融合カーネル。「BFP16 のカーネルは LLM の端から端までには精度が足りない」。

**BonsaiAI との関係**: BonsaiAI は bfp16 を、DiT の注意と MLP (潜在のコサインで確かめた、[image-amd](design/image-amd.ja.md)) と、gpt-oss の専門家のプロンプト (KL を測った、[moe](design/moe.ja.md)) に使っている。新しい LLM で bfp16 を使うときは、同じ KL の掃引を通す。測り直してはいない。

### ENEC 2604.03298 (a, Ascend) — 未着手

**言っていること**: 長さ固定のブロックの符号と分岐なしの復号で、重みの転送の律速を外す。端から端まで最大 6.3×。

**BonsaiAI との関係**: [amd](design/amd.ja.md) のとおり、密なモデルのデコードは帯域より設定のし直しで決まっている。効くのはそれを減らした後。先に q4nx の符号とスケールのエントロピーを測る (2.5 ビット/符号を切らないなら見合わない見込み)。

## MoE

全体の表は [docs/design/performance.ja.md](design/performance.ja.md)。AMD に関係するものだけ。

### 2505.16056 (Not All Models Suit Expert Offloading) — 一部

**言っていること**: 20 個の MoE の経路の局所的な一貫性を測った。活性の専門家の約 2 倍のキャッシュで足りる。LFU が LRU より 3〜5 ポイント上。

**BonsaiAI との関係**
- AMD の専門家の置き場 (`--expert-places`、[moe](design/moe.ja.md)) は、校正した使用回数で置く専門家を選ぶ
- Qwen3.6-35B-A3B は 16 GB の機械で 7.5 GiB、5.8〜6.0 tok/s ([moe](design/moe.ja.md))
- 置き場の大きさは「活性の 2 倍」より大きく取っている (16 GB の機械の gpt-oss-20b は約 5 倍、1 層 32 個のうち 20)

### EStream 2609.06551 (Hexagon) — 当てはまった

**言っていること**
- 1 つのコンパイル済みの専門家グラフを全部の専門家で使う
- 重みは銀行 × 枠の DMA-BUF の置き場にあり、番地は 先頭 + 枠 × 間隔。経路の記述は呼び出しのときに渡す
- 専門家は O_DIRECT で読む (同時に 2 つで帯域の 95 %)

**BonsaiAI との関係**
- `MoERoute` がスクラッチパッドに書く位置 ([moe](design/moe.ja.md)) は、この論文の「専門家の仮想化」と同じ考え方。一部の専門家だけを持つ AMD の置き場 ([moe](design/moe.ja.md)) の下敷きになった
- gpt-oss-120b は、専門家のファイルを O_DIRECT で読む (handoff-amd-moe.md 4)

### Fate 2502.12224 — 当てはまった

**言っていること**: 次の層のゲートを今の層の入力で予測する。生成の予測は 79〜84 %。

**BonsaiAI との関係**
- Apple の gpt-oss-20b: 次の層の専門家の予測が上位 4 個に 86〜87 %。先読みで、どの予算でも 3〜6 % 速くなった
- AMD の gpt-oss-120b: 次の層の先読み (上位 4 個) で外れが 9.8 → 3.1 %、9.2 → 9.7〜9.9 tok/s (handoff-amd-moe.md 4)

### NPUMoE 2604.18788 — すでに同じ

**言っていること**: プロンプトだけを Core ML で Neural Engine に。経路、集め、書き戻しは CPU。

**BonsaiAI との関係**: Neural Engine に gather を載せない BonsaiAI の判断 ([moe](design/moe.ja.md)) と合う。生成は扱っていない。

## Apple の Neural Engine (M2 で確かめた、[apple-neural-engine](design/apple-neural-engine.ja.md))

AMD の調査と同じ時期に、Neural Engine の論文の規則と上限を M2 で確かめた。速さは上がっていない。入ったのは、正しさの直し、上限の検出、測る道具 (`tools/coreai/aicheck`)。

### 2606.22283 (リバースエンジニアリング) — 一部

| 論文の規則 | M2 で | BonsaiAI の扱い |
|---|---|---|
| linear / conv の出力は 2^15 未満 | 成り立つ。**途中の和でも** 32768 に届くと Inf | もともと範囲を抑えている。新しいモデルの点検の項目に |
| 幅の軸の slice で、開始が 0 でないと ×16 | 開始を 32 で割った余りが 1〜14 のときだけ | `aicheck lint` で探す。手元の書き出しで影響のある所はない |
| softplus(+inf) は 0 | x > 10.4 で 0 | ailower の softplus の形を変えた (最大の相対誤差 1 → 0.0014) |
| matmul の中の非正規化数は 0 (論文は M1) | M2 も同じ | 範囲の点検で |
| 1 プロセスのプログラムは約 128 個 | **112 個**まで。それを越えると Core AI が黙って GPU で動かす | 上限の近くで落ちたときにそう言うようにした |
| 1 プログラムに専門家 5 個までの段 (M1) | **M2 にはない** (top-k に比例) | |

### 2611.19101 (roofline) — 当てはまった

- 遅い活性の経路 (M2 Pro で約 14 GB/s): M2 で要素ごとの演算が 16〜20 GB/s で頭打ち
- 2 本の重みの経路: 実行時の重みは 26 GB/s、定数の重みは 50〜52 GB/s で、約半分
- lut8 は fp16 の 0.6 倍の時間: 論文の M2 の表と同じ

### ANEForge 2606.17090 — 一部

直接の呼び出しは約 70 µs。Core AI を通すと約 160〜175 µs で、Core AI とドライバの分が約 100 µs ある。

## 論文から始めて、別のことが分かったもの

- **GDN の 0 の塊** ([amd](design/amd.ja.md)、[amd-llm](design/amd-llm.ja.md)): 設定のし直しを数える道具 (`tools/amd/elfconfigs`) で、GDN のカーネルの .bss の配列に aiecc が毎回 0 を書いていると分かった。論文にはない
- **QGEMV の設定の大きさ** ([amd](design/amd.ja.md)): 設定の費用が大きさで増える (2.3 µs/KB) ので、QGEMV を 1 列 1 コアにして設定をほぼ半分にした。GEMV の速さは帯域で決まるので変わらない
- **IRON の GEMM の限界** ([amd-llm](design/amd-llm.ja.md)): 短いプロンプトの時間の 6 割は、行に関係なく重みを全部流す GEMM と重みの展開。IRON の GEMM の作りの限界で、論文の範囲外
- **Qwen3.6 の書き出しの遅さの原因** ([amd-llm](design/amd-llm.ja.md)、[moe](design/moe.ja.md)): 取得元の書き出しが、main に入っていないブランチ (t3/qwen36-integrate) から作られていた

## まだ試していない候補 (優先の順)

handoff-amd-papers.md の 3〜9 節に、試し方を書いてある。

1. 多次元の DMA で 1023 語の制限を避ける (2512.13282)。ホストの並べ直しが減るほか、プロンプトの演算子で行の数で DMA を切れるようになる
2. 注意の融合 (2609.21264)。設定のし直しも減る
3. タイルを入力と出力で違う大きさで持つ (2511.16041)。画像の DiT と長いプロンプト
4. q4 の前に回転 (QuaRot の形)。Qwen3.5 / 3.6 の AMD の q4 の精度
5. 重みの可逆圧縮 (ENEC)。設定のし直しを減らした後
