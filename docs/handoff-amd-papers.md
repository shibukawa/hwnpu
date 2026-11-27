# 引き継ぎ: 論文から拾った AMD の NPU の最適化の候補

残り: 論文から拾った AMD の NPU の最適化の候補のうち、手を付けていない 3〜9 節。機械は Ryzen AI 9 365 (XDNA2、NPU4 系)。書いてあるのは仮説と、それを確かめる手順。

作成: 2026-11-20、整理: 2026-11-21、2026-11-27。arXiv の「NPU」の検索 (505 件) と、AMD XDNA、端末の LLM の論文の調査から、hwnpu の AMD の経路 (IRON、`internal/llm/iron`、`internal/backend/amd`) に入れられそうなものを選んだ。全体の一覧 (`docs/npu-optimization-papers.md`: MoE 以外、`docs/moe-papers.md`: MoE) は、このリポジトリにはまだ入っていない。論文ごとの結果 (当てはまったか) の一覧は [reference-paper.ja.md](reference-paper.ja.md)。

済んだ 0〜2 節 (基準の数字、演算子の切り替えの上乗せ、要らない同期) は外した。結果は [docs/design/amd.md](design/amd.md)、[amd-llm](design/amd-llm.md)、[moe](design/moe.md)。今の数字と測り方もそこにある。

**論文の確かめた度合い**
- ✓: 本文 (HTML) を調べた
- (a): 要旨だけで見た
- †: 一般に知られた手法で、本文を読んでいない

2026 年のものは査読前のプレプリントが多い。数字を設計の根拠にする前に本文を読む。

道具: `go run ./tools/amd/elfconfigs [-seq] ELF...` (設定のし直しの数とバイト)、`iron_moeprobe.py --switch` (切り替えの費用)。落とし穴は [moe](design/moe.md) と [amd](design/amd.md) の表にある。

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
- 画像の DiT の GEMM (Anima、FLUX.2/3 klein の bfp16 の MLP と注意。[image-amd](design/image-amd.md)、[image-amd](design/image-amd.md))
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

1. 専門家の常駐を、校正した頻度で決める。層ごとに活性の約 2 倍 (2505.16056、NPUMoE、Fate)。[moe](design/moe.md) と同じ方向
2. `MoERoute` がスクラッチパッドに書く「表の中の位置」を「arena の枠の位置」に読み替える (EStream 2609.06551)。今の形 ([moe](design/moe.md)) はこの論文の「専門家の仮想化」と同じ考え方
3. 層をまたいだゲートの先読み (Fate、Pre-gated)

## 10. 気をつけること

shim のチャネルのキュー、IRON のコンパイルのキャッシュ、ホストが読む領域、書き出しの重さは [moe](design/moe.md) と [amd](design/amd.md) の落とし穴の表。ここにはこの調査に固有のことだけ。

- DMA の形を変える試し (3 と 4) では、チャネルごとに 3 本までにする (タスクキューは 4 段。続けて 2 回止めると再起動が要る)
- **NPU を他のセッションと分け合う**: 2026-11-20 は別のセッションが gpt-oss-120b の hwnpu-chat を続けて回していて、小さな試しの時間が 2〜5 倍に揺れた。比べるものは同じ回の中で交互に回し、`/dev/accel/accel0` を開いているプロセスを先に見る
- 結果は docs/design/ の話題のファイル ([design/README.md](design/README.md)) に今の状態として書く
