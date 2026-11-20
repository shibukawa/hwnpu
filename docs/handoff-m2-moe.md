# 引き継ぎ: M2 (16 GB) での gpt-oss-20b の専門家を速く

作成: 2026-11-20。M3 (16 GB) で作った gpt-oss-20b の Core AI の経路 (注意と出力層は Neural Engine、専門家は CPU で MXFP4 のまま) と、専門家ごとの固定 (`experts.Placement`) を、M2 の MacBook Air (16 GB、macOS 27.2.1) で速くするためのメモ。設計と M3 での記録は [docs/improvements.md](improvements.md) の 7.5 節の gpt-oss の行、[docs/design.md](design.md) の 16.260 (`hwnpu status`)。

## 1. ここまでの状態

- ブランチ: `t3/gpt-oss-20b` (PR #243)
- 専門家は CPU で、チェックポイントを写像したまま (`safetensors.File.Map`、10.2 GB、コピーしない) `internal/mxfp4` の NEON のカーネル (e2m1 を表で引く、行は 32 個ごとに int8、SMMLA / SDOT) で計算する。`internal/llm/experts/host.go`
- 専門家の固定 (`internal/llm/experts/placement.go`): メモリの半分に収まらないとき (16 GB の機械) は、芯 (models.yaml の `experts_core`、141 個 1.9 GB) を先に、次にアプリの使用統計 (`~/Library/Caches/hwnpu/models/gpt-oss-20b/experts-usage/<キー>.bin`) の順に、予算 (既定はメモリの 3/8 = 6.4 GB) まで専門家ごとに mlock する。30 秒ごと (初回は 5 秒後) に使われ方で入れ替え、`--idle` (5 分) 使わなければ芯以外を外す
- `HWNPU_MOE_WIRE`: `0` 固定しない、`1` 全部、`6GB` のような大きさは予算
- `HWNPU_MOE_DEBUG`: 項目の並び。`1` (time) で専門家の時間、固定の様子、固定した専門家に当たった割合。`usage=FILE` で使用回数をファイルにも、`prefault=0|wait`
- 使用統計の道具: `tools/moeusage/` (`prompts.tsv` 19 分野 約 25 言語 110 個、`collect.sh`、`analyze.py`)

## 2. 測った値

生成は「Explain how a CPU cache works, with examples.」に 600 トークン (`--seed 1 --prompt-cache off`)。

| | M2 (MacBook Air 16 GB) | M3 (16 GB) |
|---|---|---|
| 全部固定 (9.4〜9.6 GB) | **13.2 tok/s** (1 ステップ 75 ms、専門家 1 層 1.85 ms) | 18.8〜19.1 tok/s (専門家 約 0.9 ms) |
| 6.4 GB (今の既定)、統計あり | 10.1 tok/s (専門家 2.38 ms) ※ENOMEM の直し (237e28c) の前 | 17.4〜18.3 tok/s (0.86〜1.21 ms、当たり 76〜83 %) |
| 6.4 GB、初回 (統計なし) | 6.5 tok/s (4.72 ms) ※同上: 芯が 0.4 GB しか固定できず次の入れ替えまで待った | 17.7〜18.1 |
| 固定なし | 7.9 tok/s (126 ms/ステップ、部品 1 つ 9〜11 ms、128 の塊 7.5 s) / 電力を測った回は約 13 tok/s (ページキャッシュに残っていた) | 17.2〜17.9 (メモリに余裕) |
| Neural Engine の部品 (1 つ) | 約 4.8 ms (固定あり) | 約 3.5 ms |
| 電力 (生成中、powermetrics) | 固定なし 7.0 W、全部固定 7.7 W (CPU 5.3〜5.9、ANE 1.7〜1.8) | 全部固定 8.96 W (CPU 7.22、ANE 1.74)、0.52 J/トークン |

- 1 トークンの内訳 (M2、全部固定): 専門家 24 層 × 1.85 ≈ 44 ms、Neural Engine の部品 約 31 ms。M3 は 21 ms と 36 ms
- M2 の専門家は M3 の 1.85 倍。クロック (P コア 3.5 対 4.05 GHz) の差 1.16 倍より大きい。帯域でも説明できない: 1 層 1 トークンは 4 専門家 × 13.3 MB = 53 MB で、M2 の 1.85 ms は 29 GB/s、M3 の 0.9 ms は 59 GB/s (どちらもメモリは約 100 GB/s)

## 3. やること (効く順の見込み)

### 3.1 専門家のカーネル (余地がいちばん大きい: 1.85 → 1.0 ms なら全部固定で約 17 tok/s)

カーネルは M3 で合わせた。M2 で同じとは限らない。

- まず CPU だけで測る (Neural Engine もロックも要らない):

  ```sh
  HWNPU_GPTOSS_CKPT=~/Library/Caches/hwnpu/original/openai/gpt-oss-20b \
    go test ./internal/llm/experts -run 'TestHostTokenTime|TestHostRowsTime' -v -count=1
  ```

  `TestHostTokenTime` は 1 行 (生成) の 24 層、`HWNPU_GPTOSS_GAP=1ms` で層の間に間を入れる (hwnpu-chat では Neural Engine の部品が間に入り、眠ったワーカーの起き上がりが遅い)。`TestHostRowsTime` は 128 行 (プロンプト)。M3 の値: 1 行 0.83〜0.93 ms、間 1 ms で 1.07〜1.25、128 行 32 ms/層 (`internal/mxfp4` の MOVI の直しの後)
- 疑うところ:
  - **累積の形**: 11c6385 で M3 に合わせて `mm28NEON` の累積を 1 つに (2 つ目の累積と ADD4S を外した) した。M3 では `VEOR v,v,v` が 0 にする慣用句として扱われず、累積を使い回すと鎖が直列になっていたのが本当の原因で、`VMOVI $0` に替えて 2 倍になった (記憶メモ「NEON: zero with MOVI」)。M2 (Avalanche/Blizzard) の SMMLA の待ち時間と並びの数は測っていない。2 つの累積、ループの展開、命令の並べ方を M2 で比べる
  - **命令ごとの速さ**: M3 は SMMLA が約 6 サイクル待ち、1 サイクル 1.5 個 (1 コア 約 190 GMAC/s)、SDOT が約 3 サイクル、3.5 個。命令の種類を 1 つずつ抜いて時間を比べると、どこが詰まっているか分かる (M3 ではこれで EOR を見つけた)。`internal/mxfp4/mxfp4_test.go` の `BenchmarkDot`、`BenchmarkRunVectors`
  - **SDOT の道との比較**: `HWNPU_MXFP4_I8MM=0` で SMMLA を使わない。M2 で SDOT の方が速ければ、M2 だけ切り替える
  - **ワーカーの数と塊**: `internal/mxfp4/matvec.go` の `Run` は `GOMAXPROCS` (M2 も 8 = P 4 + E 4) のワーカーが 1 つのカウンタから塊 (`chunk` 行) を取り合う。E コアは遅いが取り合いなので最後の塊のぶんしか待たない。それでも `GOMAXPROCS=4`、`6` と塊の大きさを比べる。眠りと起き上がり: `HWNPU_MXFP4_SPIN`、`HWNPU_MXFP4_SPINNERS` (M3 では hwnpu-chat の中で差がなかった)
- 変えたら: `go test ./internal/mxfp4 ./internal/llm/experts` (値は `TestDot`、`TestRunVectors` が見る)、hwnpu-chat で同じ文になること、`hwnpu-dev generate -score` の KL が変わらないこと。M3 でも遅くならないこと (M3 で測れないなら、CPU の種類で切り替える)

### 3.2 固定していない専門家の外れ (6.4 GB の 10.1 tok/s と全部固定の 13.2 の差)

- まず ENOMEM の直し (237e28c) の後で測り直す (下の 4 節)。初回から 6.4 GB 固定できているか (`experts: wired 6.4 GB … (budget 6.4 …)` が最初の行に出る)、当たりの割合 (`experts: NN% of … runs on wired experts`)
- 外れは 1 層あたり 0.5 ms ほど (M2: 2.38 − 1.85)。Neural Engine が部品を回す 3.5〜5 ms の間、CPU と SSD は空いている。次の層で使う専門家を先に読む:
  - 簡単な形: 固定していない専門家のうち点数の高いものを `madvise(MADV_WILLNEED)` かページに触れて温めておく (`placer` の点数が使える)
  - 予測する形: 層 l の隠れ状態に層 l+1 のルーターを掛けて次の 4 つを当てる (MoE のオフロードの論文でよく使われる)。gpt-oss のルーターは 1 層 2880 × 32 + bias で CPU でも軽い。ルーターの重みは今 Neural Engine の部品の中なので、チェックポイントから読む。当たりの割合を先に測る
- 予算の曲線: `HWNPU_MOE_WIRE=4GB`、`6GB`、`8GB`、`1` で速さとメモリ。メモリを減らすことにも価値があるので、曲線のまま記録する

### 3.3 Neural Engine の部品 (M2 は 1 つ 約 4.8 ms、M3 は 3.5)

- 余地は小さい。部品をまとめて呼び出しを減らす (1 トークン 25 回)、注意の重みを小さくする (lut6r は KL が悪くなる、improvements.md の lut8r の行)

## 4. M2 で最初に測ること (同じ状態で交互に)

固定なしの速さはページキャッシュの残り方で 7.9〜13 tok/s と揺れるので、3 つの固定を同じ状態で交互に回し、`memory_pressure` も残す。ほかのアプリはふだんどおり開けておく (それが 16 GB の機械の普通の状態)。

```sh
go build -o hwnpu-chat ./cmd/hwnpu-chat
for r in 1 2; do for w in 0 6GB 1; do
  echo "== $w, round $r"; memory_pressure | tail -1
  HWNPU_MOE_WIRE=$w HWNPU_MOE_DEBUG=1 ./hwnpu-chat chat -m gpt-oss-20b -n 600 --seed 1 --prompt-cache off \
    "Explain how a CPU cache works, with examples." 2>&1 >/dev/null | grep -oE "experts: .*|\[prompt.*"
done; done
```

長いプロンプト (2662 トークン、M3 の固定ありで 21.1 s) も同じ 3 つで。`hwnpu status -w` を別の端末で開くと、hwnpu-chat のメモリ (footprint + wired)、NPU% (ANE が読んでいた時間の割合) が見える。電力は root が要る (`sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000`)。

## 5. 気をつけること

- Neural Engine のロックはほかのセッションが長く持つことがある。迂回しない (CPU だけの 3.1 はロックが要らない)
- Core AI のコンパイル済みのキャッシュ (`~/Library/Caches/<プロセス>/com.apple.e5rt.e5bundlecache`) が消えると最初に 40 秒ほどコンパイルし直し、その回の速さは比べられない。速さは 2 回目から
- ディスク: 最初の取得はチェックポイント 13 GB と変換、`HWNPU_CACHE_MIN_FREE` (10 GB) の空きも要る
- `hwnpu status` の NPU% は推定 (PMP の ANE の読み出しの帯域の分布で、いちばん下の区分の外にいた時間)。M2 では確かめていない (チャンネルの名前が違うかもしれない)。電力は macOS 27 では root でないと 0 しか読めないので「読めない」と出す
- 速くなったら: improvements.md (と .ja.md) の gpt-oss の行、MODEL.md、models.yaml の `memory`
