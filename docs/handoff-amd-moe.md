# 引き継ぎ: AMD で MoE の専門家を一部だけ置く (使用統計、固定する専門家、gpt-oss-120b)

作成: 2026-11-20、整理: 2026-11-21。Mac (Core AI) の gpt-oss-20b で作った「専門家ごとに、よく使うものだけを常駐させる」仕組みと使用統計の道具を、AMD の NPU (Ryzen AI 9 365、XDNA 2、50 TOPS) に持っていき、メモリに入りきらない MoE (gpt-oss-120b) を動かすためのメモ。Mac 側の実装は [docs/handoff-m2-moe.md](handoff-m2-moe.md) の 1 節、AMD の gpt-oss-20b は [docs/design/moe.md](design/moe.md)。

## 1. Mac で作ったもの (ブランチ `t3/gpt-oss-20b`、PR #243)

- **使用統計** (`internal/llm/experts/usage.go`): 層ごと専門家ごとの使用回数。ファイルは `"HWEU"`、版 1、層の数、専門家の数 (uint32 little endian)、層 × 専門家の uint32。合わせるときに上限 (2^30) を超えたら全部を半分に
- **固定の配置** (`internal/llm/experts/placement.go`、`experts.Placement`): 芯 (どの文でも使われる専門家) を先に、次にアプリの統計の点数の順で予算まで。30 秒ごとに使われ方で入れ替え、使わない間は芯だけ。アプリごとの統計は `~/Library/Caches/hwnpu/models/<モデル>/experts-usage/<キー>.bin` (hwnpu-chat の `--experts-key`、既定はコマンド名、`off` で無し)。無いときは同じ所のほかの統計の合計
- **models.yaml の `experts_core`**: 層ごとの芯の番号 (`chat.Model.ExpertsCore` → `chat.Config.Experts` → `chain.OpenWith` → `experts.OpenHostWith`)
- **道具** (`tools/moeusage/`): `prompts.tsv` (19 分野: 数学、物理、化学、生物、コンピュータ、機械、歴史、地理、ビジネス、個人の相談、ゲーム、ボードゲーム、冗談、文学、法律、医学、料理、旅行、各国の言葉。約 25 言語、110 個)、`collect.sh MODEL OUT` (hwnpu-chat で回して分野ごとに `OUT/分野.bin`)、`analyze.py OUT` (偏り、ほかの分野から選んだときの当たり、芯の YAML)
- **gpt-oss-20b の結果** (Mac、1 層 21 万回の選択): 768 個すべて 147 回以上使われた。分野の中では上位 16 個 (半分) で 74〜90 %、全分野を均等に混ぜると 81 %。ほかの分野で選んだ 16 個はコンピュータ 59 %、数学 63 % しか当たらない (アプリごとの統計が要る)。芯 (どの分野でも層の中で均等な割合 1/32 以上) は 141 個 (1.9 GB) で 31〜41 %
- **Mac での効果**: 16 GB の M3 で 6.4 GB 固定 17.4〜18.3 tok/s (全部固定 18.8〜19.1、当たり 76〜83 %)、M2 で 10.1 (全部 13.2、固定なし 7.9)。どちらもほかのアプリに約 3 GB 返す

## 2. 残り

使用回数、置き場、gpt-oss-120b、Qwen3.6-35B-A3B の置き場は済んだ (4 節と 5 節)。前の計画の節 (AMD の元の形、やること) は外した。残っているもの:

- **120b の書き出しを取得元 (NAS) へ** (`hwnpu-dev models put`、70 GB)
- **置き場を増やす**: プロンプトのスロットは 64 個 (`iron_gptoss_moe.COUNT_WORDS`) までなので 56 が上限。カウントの物を広げれば 64 GB の機械でもう少し置ける (固定 32 GiB で空き 19 GiB)
- **外れの費用を減らす** (120b は 1 トークン約 14 個 × 2.5 ms、ディスクの速さ): 層ごとに置き場の数を変える (配列の鍵が同じ大きさを要るので今は一様)
- **外れた専門家を CPU で計算する形 (B)**: していない。`internal/mxfp4` は arm64 の NEON だけで、amd64 は Go の遅い道 (`mxfp4_other.go`)。AVX-512 VNNI (VPDPBUSD) のカーネルが要り、NPU の配列と CPU を層ごとに往復する
- **Qwen3.6-35B-A3B**: プロンプトの速さ (P = 32 で全部の 2.2 倍)、デコードの先読み (ホストの経路にする)
- **AMD の記事の基準を同じ機械で**: ONNX Runtime GenAI の「Accelerating GPT-OSS-20B on AMD Ryzen AI NPUs」(全部常駐 13.4 tok/s・13.7 GB、24 層動的 2.4 tok/s・2.7 GB)。環境が無く測っていない

## 3. 気をつけること

- 統計のファイルの形と `Placement` の決め方は Mac と同じにしておく (models.yaml の `experts_core` も、道具も共通)
- 固定の量の測り方: AMD は smaps の `/dev/accel/accel0` の写像の大きさ。`hwnpu status` は Linux では resident を出す
- 結果が出たら: [docs/design/performance.md](design/performance.md) (と .ja.md)、[docs/design/new-model-checklist.md](design/new-model-checklist.md) (と .ja.md) の MoE の節、models.yaml、MODEL.md

## 4. 結果 (2026-11-20、[docs/design/moe.md](design/moe.md))

- **使用回数**: できた (`iron.Model.CountExperts`)。デコードは次のコマンドのスクラッチパッドの位置の語を読み戻す (費用は測れない)。20b の 110 プロンプトは Mac とほぼ同じ (芯 140 個と 141 個、131 個が共通)。models.yaml の `experts_core` はそのまま
- **一部だけ常駐**: 決まった数の置き場 (スロット) を持ち、配列は「スロット番号 → 置き場の位置」を実行時の値で受け取る形 (EStream 2609.06551 の考え方)。カーネルは変えず、デコードは 1 コマンドずつ待ち、ホストが外れをコピーして位置の語を書き直す (待つ費用 1 トークン約 1.2 ms)。20b: 置き場 16 で固定 6.90 GiB、17.2 tok/s (全部 12.22 GiB、20.2)、置き場 24 で 9.44 GiB、19.0。答えは全部の書き出しと 1 文字ずつ同じ。B (CPU で外れを計算) はしていない
- **gpt-oss-120b**: 64 GB の機械 (OS から 59 GiB) で置き場 56 (`--expert-places 56`)、デコードのルーティングはホスト (`--host-route`: router のカーネルは 32 個用のまま)。固定 32.05 GiB。専門家のファイルは O_DIRECT で読む (EStream と同じ。中継のバッファから)。次の層の先読み (上位 4 個) で外れ 9.8 → 3.1 %、9.2 → 9.7〜9.9 tok/s (SSD の帯域で決まる)
- **120b の HF との照合**: 32 トークンと 3 ステップでコサイン 0.9978〜0.9992、最上位 3/4 (外れた 1 つは参照の 1 位と 2 位が p 0.052 と 0.036 の僅差で、2 位を選んだ)
- **120b の使用統計**: 110 プロンプトで集めた (`~/.cache/hwnpu/moeusage/gpt-oss-120b-amd`)。128 個のうち 0 回の専門家がある (5 個、10 回未満は 39 個)。よく使う 56 個で選択の約 84 % (20b の 16/32 の 81 % より散らばる)。芯 515 個 (1 層 14.3 個) を `experts_core` として渡すとコピー 3.1 → 2.8 %、10.0 tok/s
- **120b は models.yaml に載せない** (検証だけ): NPU に 32 GiB を固定し、10 tok/s はディスクの帯域で決まる。このサイズは GPU 向き。80 TOPS を超える NPU (Snapdragon X2、M6 など) とメモリが安くなってから考える
- **AMD の記事の基準** (同じ機械では測っていない: ONNX Runtime GenAI の環境が無い): 20b の全部常駐 13.4 tok/s・13.7 GB に対し hwnpu 20.2・12.2 GiB、24 層動的 2.4 tok/s・2.7 GB に対し置き場 16 で 17.2・6.9 GiB

## 5. Qwen3.6-35B-A3B (2026-11-21、[docs/design/moe.md](design/moe.md))

- 置き場を `qwen35moe` にも (`--expert-places P`、P は 256 まで)。デコードは経路の選択がデバイスのまま、[moe](design/moe.md) の fed の書き直しがそのまま動く。プロンプトは Qwen3.6 の数と x (専門家の番号の順) を置き場の順にした `poolRowPasses`
- 40 層 (main 3202fb7 で作り直し): P = 32 で 7.2〜7.6 GiB、5.8〜6.0 tok/s (コピー 33〜38 %)、128 で 14.9〜16.0 GiB、7.5〜7.6 (6〜7 %)、全部 24.4〜25.2 GiB、8.8〜8.9。ロジットは全部の書き出しとビット単位で同じ。models.yaml: 16 GB は 32、24 GB から 128、48 GB から全部
- 芯は入れない (置き場 32 では悪くなる)。AMD の hwnpu-chat が `experts_core` とアプリの統計を置き場に渡していなかった不具合を直した (gpt-oss も)
- 残りは 2 節。Apple は [moe](design/moe.md)、[moe](design/moe.md) (Core AI、専門家は CPU)
