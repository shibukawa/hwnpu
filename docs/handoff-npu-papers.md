# 引き継ぎ: 2026-11-21 の arXiv の調査から試す候補

作成: 2026-11-25。2026-11-21 の arXiv の調査 (端末の NPU で LLM を速くする論文、約 50 本。リポジトリには入っていない) を hwnpu の実装と突き合わせた。すでに実装していたもの (16 行) は [reference-paper.ja.md](reference-paper.ja.md) の「後から見た論文」に書いた。ここは、まだ試していない候補を順位の順に並べ、始め方を書く。

順位は今の律速で決めた。
- AMD の密なモデルのデコードは、演算子の切り替え (設定のし直し) で決まる ([amd](design/amd.ja.md))
- Neural Engine の長い文脈は、使った長さに比例する計算で決まる ([kv-cache](kv-cache.md))
- MoE は、外れた専門家の読み込みで決まる ([moe](design/moe.ja.md))

論文は要旨だけで見た程度 (Gemma 3 の XDNA2 の論文だけ本文)。数字を設計の根拠にする前に本文を読む。

## 1. Gemma 3 を XDNA2 に (2602.06063) のデコードの層の組み方

**動機**: q4nx と融合した dequant はすでに FastFlowLM のカーネルで使っている。残る価値は、デコードの 1 層を少ない演算子に組む形と、注意を流す FlowKV。密なモデルの 1 層を 1 つの常駐の配列にする仕事 ([reference-paper](reference-paper.ja.md) の HeteroMosaic の「残り」) の手本になる。対象のモデルの大きさ (Gemma 3 1B/4B、Ryzen AI 7 350) も同じ帯。

**やること**
1. 本文のデコードの節を読み、1 層の演算子の数と、配列に何を常駐させているかを書き出す
2. gpt-oss の `GptOssStep` (`tools/amd/iron_gptoss_layer.py`) と比べ、密なモデル (LFM3-1.7B か Qwen3.5-0.8B) に同じ形を作れるかを決める
3. 比べる基準は `go run ./tools/amd/elfconfigs` の設定のし直しの数と、デコードの 1 トークンの時間。合格条件はロジットのコサインが変わらないこと (`tools/amd/ironlm` の閾値)

## 2. AMD で、外れた専門家を CPU で計算する (Fiddler 2402.07033)

**動機**: Apple は専門家を CPU の MXFP4 で回していて実績がある (`llm.HostRouted`)。AMD は置き場 (`--expert-places`) に無い専門家を運んでいて、「コピーせずホストで計算」は未着手 ([moe](design/moe.ja.md) の未解決)。

**やること**
1. 置き場の小さい Qwen3.6-35B-A3B (16 GB の機械、置き場 32) で、外れの率と 1 回の運ぶ時間を測る
2. 同じ専門家をホストで計算する時間を Ryzen の CPU で測る (`internal/llm/iron/moe.go` のホストの経路 `moeHostRouteMX4` と、`internal/mxfp4` を使う)
3. 専門家とトークンの数ごとに、運ぶか CPU で回すかを費用の見積もりで選ぶ。合格条件は出力が変わらないこと (計算の場所が変わるだけ)

## 3. 下書きを確かめる費用を減らす (sd.npu 2510.15312)

**動機**: AMD の 3 行の確かめる版はデコードの 1.67〜1.96 ステップ分かかり、`chat.draftPolicy` が多くの下書きを確かめずに捨てている ([amd-llm](design/amd-llm.ja.md))。sd.npu は、グラフの切り替えがデコードの最大 29 % になる問題を、切り替えを隠すことと、下書きを伸ばして NPU を埋めることで解いている。

**やること**
1. 確かめる版の ELF を `elfconfigs` で数え、1.67〜1.96 ステップの内訳 (設定のし直しか、行の数か) を出す
2. 設定のし直しが大半なら、デコードの版と確かめる版で演算子をそろえる。行の数が大半なら、下書きを伸ばしても費用が増えないかを測る

## 4. AWQ 風の scale を前の演算に畳む (AWQ 2306.00978)

**動機**: [quantization](design/quantization.ja.md) の「符号の選び方」の 5 番目で、書いていない。手間が小さく、回転、GPTQ、重みづけ k-means と組み合わせられる。Neural Engine で lut6 を lut4 に下げられれば、読むバイトが減って速くなる。

**やること**: LFM3-1.7B の lut4 で、回転と GPTQ に AWQ の scale を足したときの KL を、足さないときと比べる (`tools/coreai/lutfit`、`tools/amd/q4_gptq.py`)。scale は RMSNorm の重みか前の linear に畳み、実行時の演算を増やさない。

## 5. 長い文脈の KV を縮める (SnapKV 2404.14469、StreamingLLM の sink)

**動機**: Neural Engine のデコードの費用は使った長さに比例する。SnapKV はプロンプトの読みからデコードへ移るところで 1 回だけキャッシュを縮めるので、ホストの塊の形に合う。StreamingLLM の sink は [kv-cache](kv-cache.md) で「実装が簡単」としたまま。

**やること**: 残す行はホストが選び、マスクで示す (Neural Engine 側は変えない)。出力が変わるので、選べる設定にする。3.3 万トークンの文書の合言葉の試し ([llm-engine](design/llm-engine.ja.md)) で、縮めても答えが変わらないかを見る。

## 6. 外れた専門家を低い精度で読む (HOBBIT 2411.01433)

**動機**: 置き場の小さい Qwen3.6 で、外れたときに転送するバイトを減らす。2 と同じ所を触るので、2 の後に。

**注意**: 出力が変わる。[moe](design/moe.ja.md) の表のとおり、KL の確認の後ろでだけ使う。

## 見て、採らなかったもの

- KIVI などの 2〜3 ビットの KV: Neural Engine で int8 の KV を測ったら、メモリは半分、速さは 1 割遅い ([kv-cache](kv-cache.md) 3 節)。100K を超える文脈まで要らない
- EAGLE-3、Medusa: モデルごとに下書きのヘッドの学習が要る。MTP の層を持つモデルはすでに使っている
- TEAL (活性の疎で重みの列を飛ばす): 固定の DMA と合わない
- PowerInfer-2、LLM in a flash、ActiveFlow: 今のモデルはメモリに収まる
- Orion (Neural Engine の非公開 API): 製品に入れにくい。制約の一覧だけ、Neural Engine を調べるときに読む
- GLM-5.4 (Z.ai、2026-11-23 の告知、ループの Transformer): 約 750B で重みも出ていない。ループのモデルは Ouro を AMD の NPU に移した (ブランチ t3/add-loop-models-small、未マージ) (重みの置き場は減るが、層を通る回数ぶんの時間はそのまま)。約 30B の試しのモデルが重みつきで出たら見直す
