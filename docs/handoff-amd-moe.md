# 引き継ぎ: AMD で MoE の専門家を一部だけ置く (使用統計、固定する専門家、gpt-oss-120b)

作成: 2026-11-20。Mac (Core AI) の gpt-oss-20b で作った「専門家ごとに、よく使うものだけを常駐させる」仕組みと使用統計の道具を、AMD の NPU (Ryzen AI 9 365、XDNA 2、50 TOPS) に持っていき、メモリに入りきらない MoE (gpt-oss-120b) を動かすためのメモ。Mac 側の実装は [docs/handoff-m2-moe.md](handoff-m2-moe.md) の 1 節、AMD の gpt-oss-20b は [docs/design.md](design.md) の 16.179。

## 1. Mac で作ったもの (ブランチ `t3/gpt-oss-20b`、PR #243)

- **使用統計** (`internal/llm/experts/usage.go`): 層ごと専門家ごとの使用回数。ファイルは `"HWEU"`、版 1、層の数、専門家の数 (uint32 little endian)、層 × 専門家の uint32。合わせるときに上限 (2^30) を超えたら全部を半分に
- **固定の配置** (`internal/llm/experts/placement.go`、`experts.Placement`): 芯 (どの文でも使われる専門家) を先に、次にアプリの統計の点数の順で予算まで。30 秒ごとに使われ方で入れ替え、使わない間は芯だけ。アプリごとの統計は `~/Library/Caches/hwnpu/models/<モデル>/experts-usage/<キー>.bin` (hwnpu-chat の `--experts-key`、既定はコマンド名、`off` で無し)。無いときは同じ所のほかの統計の合計
- **models.yaml の `experts_core`**: 層ごとの芯の番号 (`chat.Model.ExpertsCore` → `chat.Config.Experts` → `chain.OpenWith` → `experts.OpenHostWith`)
- **道具** (`tools/moeusage/`): `prompts.tsv` (19 分野: 数学、物理、化学、生物、コンピュータ、機械、歴史、地理、ビジネス、個人の相談、ゲーム、ボードゲーム、冗談、文学、法律、医学、料理、旅行、各国の言葉。約 25 言語、110 個)、`collect.sh MODEL OUT` (hwnpu-chat で回して分野ごとに `OUT/分野.bin`)、`analyze.py OUT` (偏り、ほかの分野から選んだときの当たり、芯の YAML)
- **gpt-oss-20b の結果** (Mac、1 層 21 万回の選択): 768 個すべて 147 回以上使われた。分野の中では上位 16 個 (半分) で 74〜90 %、全分野を均等に混ぜると 81 %。ほかの分野で選んだ 16 個はコンピュータ 59 %、数学 63 % しか当たらない (アプリごとの統計が要る)。芯 (どの分野でも層の中で均等な割合 1/32 以上) は 141 個 (1.9 GB) で 31〜41 %
- **Mac での効果**: 16 GB の M3 で 6.4 GB 固定 17.4〜18.3 tok/s (全部固定 18.8〜19.1、当たり 76〜83 %)、M2 で 10.1 (全部 13.2、固定なし 7.9)。どちらもほかのアプリに約 3 GB 返す

## 2. AMD の今の形 (gpt-oss-20b、16.179)

- 専門家は MXFP4 のまま mx4 (1 重み 4.55 ビット) で、**全部を NPU が読めるバッファに置いて固定**している。固定の合計 12.22 GiB (`/dev/accel/accel0` の写像の大きさ、smaps)
- デコードは常駐の 1 配列 (`GptOssStep`): ルーター、選択 (routing)、専門家、注意が段になって NPU の上で回る。19.7 tok/s (50.6 ms/トークン)。1 トークンで読むのは 2.53 GB (専門家 4 個 57 MB × 24 層 + 注意 + 出力層 415 MB)、55 GB/s で 46 ms が下限
- プロンプトは塊ごとにホストが選ぶ (`internal/llm/iron/moe.go` の `moeHostRouteMX4`、`HostStep` の `moe_route_mx4`): ここではホストが選ばれた専門家を知っている。`Slots` / `SlotValues` は 1 回の塊で 1 つの専門家に行が溢れたときの置き場で、重みは全部固定のまま
- 道具: `tools/amd/iron_gptoss.py` (書き出し)、`iron_gptoss_moe.py` (専門家)、`iron_gptoss_layer.py` (層、`check --first` / `--one` で段の時間)、`kernels/moe_mx4.cc`、`kernels/gptoss_layer.cc`

## 3. やること

### 3.1 AMD でも使用回数を数える (20b)

- プロンプト: `moeHostRouteMX4` が選んだ専門家を `experts.Usage` に足すだけ
- デコード: 選択は NPU の上。選ばれた 4 個は配列の中の語 (routing の段が書く)。1 トークンに 1 回読み出せるか、読み出しの費用を測る (16.179 の記録: 1 トークン 1 コマンドの形では routed の語をホストが読めない。mlir-aie は scratchpad を 1 並びに 1 回しか読まない)。読めなければ、何トークンかに 1 回だけ、別の小さなコマンドで読む
- 数えたら Mac の統計と比べる。注意が q8、専門家が mx4 で同じ値なので、選ばれ方は Mac とほぼ同じはず。同じなら models.yaml の `experts_core` (モデルの属性) をそのまま使える
- `tools/moeusage/collect.sh` は hwnpu-chat の `HWNPU_MOE_DEBUG=usage=FILE` を使う。AMD の経路でも同じ名前で書けるようにすると道具がそのまま回る

### 3.2 一部だけ常駐させる形を 20b で試す

先にある 2 つ (improvements.md 13 節):

- AMD の技術記事「Accelerating GPT-OSS-20B on AMD Ryzen AI NPUs」(ONNX Runtime GenAI): 経路は CPU、選んだ専門家の行列を NPU で、INT4 に量子化し直し。専門家を「OS の mmap と固定で動的に」読み書きする層の数を 0〜24 で選ぶ: 0 層で 13.4 tok/s・13.7 GB、24 層で 2.4 tok/s・2.7 GB。**超えるべき基準** (同じ機械で測る)
- EStream (arXiv 2609.06551、Hexagon、プロンプトだけ): 銀行 × 枠の置き場 (DMA-BUF)、番地は 先頭 + 枠 × 専門家の大きさ、経路の記述は呼び出しのときに渡す。専門家はオフラインの並べ替えでまとめて 1 回の読みに、O_DIRECT で同時に 2 つ (帯域の 95 %)、銀行は最後の呼び出しの後で使い回す。予算の中で枠と銀行の数を費用のモデル (max-plus) で選ぶ。下の A の設計の下敷きになる

NPU が読めるのは固定したバッファだけなので、Mac のように「固定していない専門家はページキャッシュから」はできない。2 つの形:

- **A. 専門家の置き場 (スロット) を決まった数だけ持つ**: 配列は「スロット番号 → 置き場の位置」を実行時の値で受け取る (プログラムは作り直さない。今のプロンプトの `SlotValues` が実行時の値で専門家を選ぶのと同じ考え)。よく使う専門家は置き場に入れたまま、外れたらホストがチェックポイントの写像から置き場へコピーする。AMD では前に「専門家をコピーしてから GEMV」を試して、コピーが 26 GB/s で GEMV より遅かった (improvements.md 8 節「AMD MoE: copying the experts, then GEMV」)。外れの割合 × 1 専門家 (mx4 で約 14 MB) / 26 GB/s が 1 トークンに足される。デコードでは選択が NPU の上なので、外れたときに配列を止めてホストに戻る道が要る (段の間のバリアか、選択だけホストに戻す)
- **B. 外れた専門家は CPU で**: Mac と同じく、写像したチェックポイントから CPU (Zen 5) で計算する。`internal/mxfp4` は arm64 の NEON だけで、amd64 は Go の遅い道 (`mxfp4_other.go`)。AVX-512 VNNI (VPDPBUSD) のカーネルが要る。NPU の配列と CPU を段の間で行き来するので、1 層ごとに往復が入る
- まず 20b で、芯と統計の上位だけを置いて (例えば半分) 外れの費用を測る。比べるのは今の全部固定 (19.7 tok/s、12.22 GiB)

### 3.3 gpt-oss-120b

- 形 (要確認: config.json を読む): 117B、トークンあたり 5.1B、36 層、128 専門家から 4 個、hidden 2880。チェックポイント約 65 GB の大半が MXFP4 の専門家。1 専門家は 20b と同じ大きさ (約 13 MB、mx4 で約 14 MB)、1 トークンで 36 × 4 = 144 個、約 1.8〜2 GB を読む
- 全部が NPU から読めれば、55 GB/s で専門家だけ 1 トークン約 35 ms、注意と出力層を足して 20 tok/s 前後が上限の見込み (計算は 50 TOPS に対して軽い: 1 トークン約 5 GMAC)
- 機械のメモリで決まる: 64 GB 以上なら今の形 (全部固定) のまま置ける。32 GB なら 3.2 の A か B が要る。**まず Ryzen 機のメモリとディスクの空きを確かめる** (チェックポイント 65 GB、書き出し)
- 統計は 20b のものは使えない (別のモデル)。`collect.sh` で取り直す。128 専門家なので、すべての専門家が十分使われるかを `analyze.py` の「the fewest uses」で確かめる (20b は 110 プロンプト 400 トークンで最少 147 回)。足りなければプロンプトを増やすかトークンを長くする

## 4. 気をつけること

- 統計のファイルの形と `Placement` の決め方は Mac と同じにしておく (models.yaml の `experts_core` も、道具も共通)
- 固定の量の測り方: AMD は smaps の `/dev/accel/accel0` の写像の大きさ。`hwnpu status` は Linux では resident を出す
- 結果が出たら: improvements.md (と .ja.md)、new-model-checklist.md (と .ja.md) の MoE の節、models.yaml、MODEL.md

## 5. 結果 (2026-11-20、design.md 16.261)

- **3.1 使用回数**: できた (`iron.Model.CountExperts`)。デコードは次のコマンドのスクラッチパッドの位置の語を読み戻す (費用は測れない)。20b の 110 プロンプトは Mac とほぼ同じ (芯 140 個と 141 個、131 個が共通)。models.yaml の `experts_core` はそのまま
- **3.2 一部だけ常駐**: A の形 (置き場) で。カーネルは変えず、デコードは 1 コマンドずつ待ち、ホストが外れをコピーして位置の語を書き直す (待つ費用 1 トークン約 1.2 ms)。20b: 置き場 16 で固定 6.90 GiB、17.2 tok/s (全部 12.22 GiB、20.2)、置き場 24 で 9.44 GiB、19.0。答えは全部の書き出しと 1 文字ずつ同じ。B (CPU で外れを計算) はしていない
- **3.3 gpt-oss-120b**: 64 GB の機械 (OS から 59 GiB) で置き場 56 (`--expert-places 56`)、デコードのルーティングはホスト (`--host-route`: router のカーネルは 32 個用のまま)。9.2〜9.3 tok/s、固定 32.05 GiB。専門家のファイルは O_DIRECT で読む

### 残り

- 120b の HF との照合 (`tools/ref/ref_gptoss.py ref`。この機械に torch が無い) と KL
- 120b の使用統計 (`~/.cache/hwnpu/moeusage/gpt-oss-120b-amd`) から models.yaml の `experts_core`、128 個すべてが十分使われるか (analyze.py の the fewest uses)
- 120b の書き出しを取得元 (NAS) へ (`hwnpu-dev cache push`、70 GB)
- 置き場を増やす: プロンプトのスロットは 64 個 (`iron_gptoss_moe.COUNT_WORDS`) までなので 56 が上限。カウントの物を広げれば 64 GB の機械でもう少し置ける (固定 32 GiB で空き 19 GiB)
- 外れの費用: 120b は 1 トークン約 14 個 × 2.5 ms (ディスクの速さ)。外れを減らす置き方 (層ごとに置き場の数を変える: 配列の鍵が同じ大きさを要るので今は一様) や、次の層の専門家を先読みする形は試していない
