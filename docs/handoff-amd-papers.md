# 引き継ぎ: 論文から拾った AMD の NPU の最適化の候補

作成: 2026-11-20、整理: 2026-11-21。arXiv の「NPU」の検索 (505 件) と、AMD XDNA、端末の LLM の論文の調査から、hwnpu の AMD の経路 (IRON、`internal/llm/iron`、`internal/backend/amd`) に入れられそうなものを選んだ。全体の一覧 (`docs/npu-optimization-papers.md`: MoE 以外、`docs/moe-papers.md`: MoE) は、このリポジトリにはまだ入っていない。

論文ごとの結果 (当てはまったか) の一覧は [reference-paper.ja.md](reference-paper.ja.md)。

対象の機械は Ryzen AI 9 365 (XDNA2、NPU4 系)。書いてあるのは仮説と、それを確かめる手順。

**論文の確かめた度合い**
- ✓: 本文 (HTML) を調べた
- (a): 要旨だけで見た
- †: 一般に知られた手法で、本文を読んでいない

2026 年のものは査読前のプレプリントが多い。数字を設計の根拠にする前に本文を読む。

## 進み具合 (2026-11-21)

1 と 2 は済み (下に結果と残りだけ)。3〜9 は手を付けていない。

道具: `go run ./tools/amd/elfconfigs [-seq] ELF...` (設定のし直しの数とバイト)、`iron_moeprobe.py --switch` (切り替えの費用)。

## 0. 今の数字 (比べる基準)

2026-11-20 の値。このあと 1 で LFM3 系、Gemma 4、Qwen3.5 / 3.6 のデコードは速くなった (1 節)。

| もの | 値 | 出典 |
|---|---|---|
| gpt-oss-20b のデコード | 49.4 ms/トークン (20.2 tok/s)。層のコマンド 1.80 ms | [docs/design/moe.md](design/moe.md) |
| gpt-oss-20b のプロンプト | 256 トークン 1.8〜1.9 s | [moe](design/moe.md) |
| Qwen3.6-35B-A3B のデコード | 172〜192 ms (5.2〜5.8 tok/s) | [moe](design/moe.md)、[moe](design/moe.md) |
| Qwen3.6 のプロンプト | 183 トークン 4.6 s。GDN の層の A が塊の時間の半分近く。専門家の B は重みを読む速さ (約 27 GB/s) が下限 | [moe](design/moe.md) |
| q4nx の GEMV | 47〜54 GB/s (帯域どおり) | [moe](design/moe.md) |
| 違う種類の演算子 1 個の上乗せ | 約 80 µs。同じ演算子が続くと 20 µs 未満 | [moe](design/moe.md) |
| コマンド 1 個 | 62 µs | [moe](design/moe.md) |
| Anima-Aesthetic 1024² (DPM 25) | 2 分 37 秒、2.84 kJ | [image-amd](design/image-amd.md) |

測り方:
- `HWNPU_IRON_TIMING=1`: コマンドとホストの手順ごとの時間 (コマンドごとに待つので、プロンプトの時間は数倍にふくらむ)
- `iron_export_lm.py --command-per-step`: プロンプトを手順ごとのコマンドに切って内訳を見る ([amd](design/amd.md))
- `HWNPU_ENERGY`: 電力量 (package)
- `go run ./tools/amd/irongraph`、`tools/amd/ironlm`: 照合

## 1. 演算子の切り替えの上乗せを減らす — 済み ([docs/design/amd.md](design/amd.md)、[amd-llm](design/amd-llm.md)、[moe](design/moe.md), [amd](design/amd.md), [amd-llm](design/amd-llm.md))

出典は HeteroMosaic 2607.12839 ✓ (データパスの再設定は高く、DMA の設定の切り替えはほぼ無料) と 2606.07586 ✓。仮説どおり、演算子が変わるたびに `load_pdi` と配列の設定が制御コードに入る (約 40 µs + 2.3 µs/KB)。1 層の演算をまとめて、LFM3 系 11.2 → 17.2 トークン/秒、Gemma 4 E2B 7.7 → 9.3、Qwen3.5-0.8B 13.8 → 18.7、Qwen3.6 8.5 → 8.9 (どれも KL かロジットは同じ)。

残り:

- デコードの層の中の演算の切り替え: 減らすには gpt-oss の `GptOssStep` (iron_gptoss_layer.py) のような 1 配列の常駐の形を密なモデル (LFM3-1.7B か Qwen3.5) に作る。比べる基準は `elfconfigs` の数とデコードの 1 トークンの時間、合格条件はロジットのコサインが変わらないこと (`ironlm` の閾値)
- Qwen3.6 の GDN の演算子の設定 (374 KB × 30 層)
- プロンプト: IRON の GEMM が行に関係なく重みを全部の行の単位に流す
- `--load-pdi-to-ctrl-pkt` は aiecc が「operand #0 does not dominate this use」で止まる (`--ctrlpkt-adaptive-overlay=false`、teardown なしでも)。上流 (Xilinx/mlir-aie #3928 の計画、#3934・#3935 はマージ済み) にこの不具合の報告はない。複数列の設計は #3488 (実機で止まる、open) にも当たりうる。`--reconfig-teardown` は大きな設定で遅くなる

## 2. 要らない同期を外す — 済み (すること無し、16.266 の最後)

2606.07586 ✓ の「NPU だけが読み書きするバッファはホストとの同期を省く」は、IRON の経路 (`internal/llm/iron` の `readRegion`/`writeRegion`/`writeCopies`、`rows.go`、`graph.go`、`moe_route`、`moe_combine`、`moeHostRouteMX4`) ではすでに使う範囲だけを同期している。全体の同期が残るのは読み込みと、IRON でない経路 (`internal/llm/chain/moe.go`、`chain.go`。Intel と Apple でも使う) だけ。同期を減らすときは、ホストが読む領域の位置を変えない (IRON の割り当てで使用に数えられず、あとの演算に上書きされることがある。[moe](design/moe.md)、[moe](design/moe.md))

## 3. 多次元の DMA の番地で、1023 語の制限と事前の並べ替えを避ける (優先: 中)

**出典**: 2512.13282 ✓ (Ryzen AI の 2 世代の GEMM、XDNA2 で int8 38 TOPS、bf16 14.7 TOPS)。XDNA2 では bf16 を bfp16 のデータパスに通す。多次元の DMA の番地で並べ替え、ホストで重みを前もってタイルにしない。shim あたり 16 を超える BD は、同期してから使い回す。

**hwnpu で困っていること**: 「DMA の 1 次元の大きさは 1023 語まで: 重みを消費順に並べ替える 4 次元の TAP は通らないので、表の順のまま流す」([moe](design/moe.md))。そのため次のことが起きている。
- 専門家の表を「[組][k ブロック][gate/up][コア]」の並びに作り直している
- プロンプトの MoE で x をホストが 2 回、4 回と並べ直している

**やること**
- 2512.13282 の本文の DMA の書き方 (次元ごとの大きさと stride の割り方) を読み、1023 語を超える 1 次元を、2 つの次元に割って同じ並びを作れるかを、`iron_moep.py` の小さな形 (32 専門家、16 トークン) で試す
- 通れば、ホストでの x の並べ直し (`moe_route`) の時間を減らせる

## 4. タイルを入力と出力で違う大きさで持つ (優先: 中、画像と長いプロンプト)

**出典**
- 2511.16041 (a): Asymmetric Tile Buffering。XDNA2 の BFP16-BF16 の GEMM で 24.6 TFLOPS、4.54×
- 2512.13282 ✓: 出力の転送が償却できるとき、C のバッファを 1 つに (+13〜18%)

**対象**
- 画像の DiT の GEMM (Anima、FLUX.2/3 klein、Chroma の bfp16 の MLP と注意。[image-amd](design/image-amd.md)、[image-amd](design/image-amd.md))
- gpt-oss のプロンプトの専門家 (`MX4GroupGateUp`、bf16 の `mac_4x8_8x8`)

**やること**: 本文を読んで、今の `iron_gridgemm.py` と DiT の GEMM のタイルの大きさ、L1 と L2 の二重バッファの数と比べる。1 つの形 (Anima の DiT の MLP) で試し、1 ステップの時間を比べる

## 5. 注意の融合と、因果の三角を均す配置 (優先: 中)

**出典**
- 2609.21264 ✓: IRON と MLIR-AIR で FlashAttention、128K まで。融合版は XDNA2 で 3.62 TFLOP/s
  - QKᵀ のスコアをコアのメモリに置き、cascade の配線で集める
  - 融合は XDNA2 で効き、XDNA1 (ridge point が低い) ではほとんど効かない
  - メモリの段 (shim、MemTile、コア) ごとに、命令から数えた天井で roofline を描く
- STEEL 2607.09385 (a): XDNA の FlashAttention を公開。因果マスクの疎を考えて、パイプラインの段を置く
- Zen-Attention 2508.17593 (a): 注意を畳み、タイルを決めて DRAM の帯域を減らす。注意 最大 4×、端から端まで 最大 32%

**対象**
- プロンプトの塊の因果な注意 ([llm-engine](design/llm-engine.md), [amd-llm](design/amd-llm.md))。三角の後ろの行ほど仕事が多いので、列ごとの仕事が偏っていないかを先に測る
- DiT の注意 ([image-amd](design/image-amd.md) の 64 キーの塊、QKᵀ を bfp16)

**やること**: 2609.21264 は IRON で書いているので、コードが公開されていれば今の `iron_mha.py` と同じ形で比べるのが早い。融合は設定のし直しの回数も減らす ([amd](design/amd.md))

## 6. q4nx の前に回転 (アダマール) で外れ値をならす (優先: 中、精度)

**出典**: MR-GPTQ 2509.23202 (a, MXFP4 とブロックのアダマール変換)、QuaRot / SpinQuant †

**動機**
- Qwen3.6 の 40 層の q4 は、HF とのコサインが 0.986〜0.996 で、差の大半は q4 ([moe](design/moe.md))
- gpt-oss の専門家は q4nx に丸め直すと重みの誤差 6.6%。KL は README 0.052 → 0.135 ([moe](design/moe.md))

**やること**
1. `tools/amd/iron_q4.py` で量子化する前に、残差の次元を回す。RMSNorm の重みを次の射影に畳み込み、W を W·Hᵀ に書き換える。実行時の演算は増えない
2. CPU の KL の掃引 (`ref_gptoss.py sweep`、`qwen36_gptq.py` の評価) で、回転なしの q4 と GPTQ の q4 と比べる
3. 効けば、GPTQ と組み合わせる (先に回して、それから GPTQ)

**注意**
- RMSNorm の重みが `1 +` の形 (Qwen3.5 系、`RMSNorm.plus_one`) と、そうでない形 (gpt-oss) の両方がある
- 回転は残差の流れ全体にかかる。埋め込み、出力層、共有の専門家、router のすべてに同じ H を入れる

## 7. bfp16 の精度を測り直す (優先: 低、確認だけ)

**出典**: TileFuse 2606.11357 ✓ (XDNA2 の W4A16/W8A16 の融合カーネル)。「BFP16 のカーネルは LLM の端から端までには精度が足りない」と書いている。

**対象**: hwnpu は bfp16 を DiT の注意と MLP ([image-amd](design/image-amd.md))、gpt-oss の専門家のプロンプト ([moe](design/moe.md)) に使っている。DiT は潜在のコサインで確かめ済みで、gpt-oss は KL を測っている。新しい LLM で bfp16 を使うときは、[moe](design/moe.md) と同じ KL の掃引を通す。

## 8. 重みの可逆圧縮でデコードの帯域を稼ぐ (優先: 低、研究寄り)

**出典**: ENEC 2604.03298 (a, Ascend)。長さ固定のブロックの符号と分岐なしの復号で転送の律速を外す。端から端まで最大 6.3×。

**前提を先に測る**: q4nx の符号 (4 ビット) とスケールのエントロピー。2.5 ビット/符号を切らないなら、復号の費用に見合わない見込み。`tools/amd` に小さなスクリプトを書いて、Qwen3.6 と gpt-oss の表で測るだけでよい。[amd](design/amd.md) のとおり密なモデルのデコードは帯域より設定のし直しで決まっているので、効くのはそれを減らした後。

## 9. MoE

MoE の論文と順位つきの候補は別の資料 (`docs/moe-papers.md`、未収録)。AMD に直接関係する上位 3 つ:

1. 専門家の常駐を、校正した頻度で決める。層ごとに活性の約 2 倍 (2505.16056、NPUMoE、Fate)。[handoff-amd-moe.md](handoff-amd-moe.md) と同じ方向
2. `MoERoute` がスクラッチパッドに書く「表の中の位置」を「arena の枠の位置」に読み替える (EStream 2609.06551)。今の形 ([moe](design/moe.md)) はこの論文の「専門家の仮想化」と同じ考え方
3. 層をまたいだゲートの先読み (Fate、Pre-gated)

## 10. 気をつけること (前の作業の落とし穴)

- **shim のチャネルのタスクキューは 4 段**: 待たずに積むと止まる (`ERT_CMD_STATE_TIMEOUT`)。続けて 2 回止めると SMU が戻らず、再起動が要る ([moe](design/moe.md))。DMA の形を変える試し (3 と 4) では、チャネルごとに 3 本までにする
- **IRON のコンパイルのキャッシュは演算子の引数で引く**: build の中を変えても古い設計が動く。形を変えたら、形の引数を足す ([moe](design/moe.md) の `sink_form`)
- **ホストが読む領域**は、ホストの手順の直前の演算の出力か、状態に写したものだけ ([moe](design/moe.md))
- **書き出しは重い**: Qwen3.6 で 29 分、最大 RSS 50 GB。他の重い仕事と並べない ([moe](design/moe.md))
- **NPU を他のセッションと分け合う**: 2026-11-20 は別のセッションが gpt-oss-120b の hwnpu-chat を続けて回していて、小さな試しの時間が 2〜5 倍に揺れた。比べるものは同じ回の中で交互に回し、`/dev/accel/accel0` を開いているプロセスを先に見る
- 結果は docs/design/ の話題のファイル ([design/README.md](design/README.md)) に今の状態として書く (測った表、採らなかったもの、試し方)
