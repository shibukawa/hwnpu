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
- 外れは 1 層あたり 0.5 ms ほど (M2: 2.38 − 1.85)
- **次の層の専門家の先読みは M3 で入れた** (`internal/llm/experts/prefetch.go`、既定で上位 4 個、`HWNPU_MOE_DEBUG=prefetch=N` で変える、0 で切る)。moe_x_l / w_l · w_{l+1} に層 l+1 のルーター (チェックポイントから読む) を掛けて予測し、固定していない専門家を Neural Engine の部品の間に `madvise(MADV_WILLNEED)` とページへの接触で読み込む。予測は上位 4 個に 86〜87 %、上位 8 個に 97〜98 %。M3 では、ほかのアプリが忙しいとき 6 GB で 14.1〜14.3 → 15.1〜15.2 tok/s、3 GB の先読みあり (14.7〜15.0) が 6 GB の先読みなしより速い。静かでページキャッシュに全部あるときは差なし。12 GB の真似 (乱数 4 GB) では SSD の帯域で決まり (3 GB で 4〜5 tok/s)、先読みは効かない (improvements.md 7.5)
- M2 で確かめること: `HWNPU_MOE_DEBUG=1` の最後の行 (`the next layer's guess had …`、`… brought in before`) と、`prefetch=0` と既定の交互の比較 (4 節のスクリプトに `HWNPU_MOE_DEBUG=time,prefetch=0` を足す)
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

## 6. M2 で測った結果 (2026-11-20、MacBook Air M2 16 GB、0c459eb の先読み入り)

まとめ: **M2 のカーネルは遅くない。遅さは熱と、メモリの取り合いと、層の間で眠ったコア**。先読みがあれば予算 3〜6 GB はどれも 12.4〜12.6 tok/s。

### 6.1 測り方 (これをしないと設定の差が揺れに埋もれる)

- ファンのない Air は 1〜2 回続けて回すと `hwnpu status` の `thermal pressure` が heavy になり (nominal に戻るまで約 60 s)、同じ設定が 12.8 → 8.7 tok/s と落ちる。毎回 nominal に戻ってから 45〜60 s 待つ
- `vm_stat` の Pageins を前後で取る。Safari (WebKit 多数) を開いたままだと、生成中もシステム全体で 100〜450 MB/s 読み、1 回 7〜36 GB
- 使ったスクリプト (4 節のものに冷ますのと page-ins を足した):

```sh
therm(){ ./hwnpu status 2>&1 | grep -oE "thermal pressure [a-z]+"; }
cool(){ until [ "$(therm)" = "thermal pressure nominal" ]; do sleep 10; done; sleep 45; }
pi(){ vm_stat | awk '/Pageins/{gsub("\\.","",$2);print $2}'; }
run(){ cool; echo "== $*"; a=$(pi); env "$@" ./hwnpu-chat chat -m gpt-oss-20b -n 600 --seed 1 --prompt-cache off \
  "Explain how a CPU cache works, with examples." 2>&1 >/dev/null | grep -E "^experts:|\[prompt|chain: experts"; b=$(pi); echo "pageins $(( (b-a)*16/1024 )) MB; end: $(therm)"; }
for r in 1 2 3; do for w in 6GB 4GB; do for p in 4 0; do run HWNPU_MOE_WIRE=$w HWNPU_MOE_DEBUG=time,prefetch=$p; done; done; done
```

### 6.2 カーネル (3.1): 詰める余地は小さい

- `internal/mxfp4` の `BenchmarkGenLayer` (1 層の生成: 4 専門家の gate_up と down、1 ベクトル、53 MB、318 MB を回してキャッシュの外): **1.05 ms (約 50 GB/s)**、M3 の 0.9 ms とほぼ同じ。`GOMAXPROCS` 8、6、4 で差なし (1.01〜1.10 ms)
- 同じ機械で 8 スレッドの単純な読み出しは約 61 GB/s。カーネルはその 83 %: 命令を詰めても上がるのは最大 15 % ほど。生成は 1 行なので `dotNEON` (SDOT) で、SMMLA (`HWNPU_MXFP4_I8MM`) は関係しない
- 実機のチェックポイントでの `TestHostTokenTime` はページの読み込みで 8.8〜23 ms/層になり、カーネルの比較には使えない (メモリが空いていないと)
- 層の間の間: `HWNPU_GPTOSS_GAP=1ms` で 1.09 ms (変わらない)、`4ms` で 1.5〜2.2 ms (コアが眠る)。`HWNPU_MXFP4_SPINNERS=2` で 4 ms の間でも 1.0〜1.07 ms に戻る
- ところが hwnpu-chat の中では spinners は効かない: 4 GB・先読みありで spinners 0 は 12.5 / 12.4 / 12.4 tok/s、2 は 11.3 / 12.5 / 11.6。既定 (0) のまま
- 「M2 の専門家は M3 の 1.85 倍」(2 節) は、カーネルではなく、ページの読み込みと眠ったコアと熱だった。chat の中の専門家は今も 1.64〜1.76 ms/層 (合成の 1.05 ms との差が残り)

### 6.3 先読みと予算 (3.2、4 節)

先読みが入る前 (d12b662、冷まして 2 回ずつ、ほかのアプリはふだんどおり):

| `HWNPU_MOE_WIRE` | tok/s | page-ins / 回 |
|---|---|---|
| 0 | 12.4 / 11.0 | 8.6 / 12.2 GB |
| 4GB | 7.5 / 11.4 | 36.6 / 12.6 GB |
| 6GB | 9.4 / 9.7 / 12.8 | 22.1 / 19.3 GB / — |
| 8GB | 11.3 / 11.0 | 12.9 / 14.5 GB |
| 1 (全部、9.1〜9.4 of 10.2 GB で EAGAIN) | 11.0 / 9.9 | 17.3 / 18.9 GB |

遅い回は、30 秒ごとの入れ替えで 1.2 GB を読み直したとき、またはメモリの状態が悪いとき。全部固定しても、Neural Engine の 1.44 GiB と合わせてほかのアプリに残らず、page-ins は減らない。

先読みあり (0c459eb、どの回も終わりまで nominal、3 回ずつ交互):

| 予算 | 先読み | tok/s | 専門家 ms/層 | 固定に当たった | 外れのうち先に読めた |
|---|---|---|---|---|---|
| 6GB | 4 (既定) | 12.2 / 12.5 / 12.5 | 2.11 / 1.68 / 1.69 | 82〜87 % | 75〜77 % |
| 6GB | 0 | 12.1 / 12.3 / 12.1 | 1.79 / 1.74 / 1.79 | 85〜86 % | — |
| 4GB | 4 | 12.4 / 12.5 / 12.5 (別の回 12.5 / 12.4 / 12.4) | 1.69〜1.76 | 66〜69 % | 77〜78 % |
| 4GB | 0 | 11.8 / 12.0 / 12.2 | 1.76〜1.84 | 68〜69 % | — |
| 3GB | 4 | 12.4 / 12.6 / 12.4 | 1.65〜1.74 | 57〜58 % | 77 % |

- 予測は上位 4 個に 87.3 %、上位 8 個に 97.7 % (M3 と同じ)
- 先読みで 6 GB +2 %、4 GB +4 %。**先読みがあれば 3 GB でも 6 GB と同じ速さ** (M3 でも 3 GB の先読みありが 6 GB の先読みなしより速かった)。先読みは外れの予測のぶん page-ins を 1 回 2〜3 GB 増やす
- 提案 (まだ変えていない): M2 では既定の予算 (メモリの 3/8 = 6.4 GB) を下げても速さは落ちず、ほかのアプリに 3 GB ほど返せる。M3 の 3 GB (14.7〜15.0) と 6 GB (15.1〜15.2) の差は小さいので、既定を 4 GB 前後 (メモリの 1/4) にするか、両方の機械で決める

### 6.4 chat の中の 1.7 ms と合成の 1.05 ms の差の内訳

`HWNPU_MOE_DEBUG=time` に、生成の 1 層ごとの専門家の時間を、選んだ専門家が「全部固定」「固定していないものは先読み済み」「先読みしていないものがある」に分けた分布 (`experts: a row's layer, …`) と、次の層の予測の時間を足した。4 GB (メモリの状態が悪めの回、10.0〜10.7 tok/s、page-ins 12〜27 GB):

| 1 層の専門家 | 割合 | 中央値 | p90 | p99 | 時間の割合 |
|---|---|---|---|---|---|
| 全部固定 | 28 % | 1.32〜1.37 ms | 1.42〜1.70 | 2.1〜3.6 | 17 % |
| 固定していないものは先読み済み | 49 % | 1.39〜1.40 | 2.5〜3.6 | 14〜16 | 45 % |
| 先読みしていないものがある | 23 % | 1.46 | 13〜16 | 27 | 38 % |

- **差のほとんどは外れ**: 中央値はどれもほぼ同じで、固定していない層がときどき SSD を待って 14〜27 ms かかる。その裾が時間の 8 割以上。先読み済みでも p99 は 14〜16 ms (読み終わる前に使われたか、追い出された)。メモリが苦しい回は先読みが page-ins を 2 倍にし (12 → 27 GB)、速さは変わらなかった (10.0 / 10.5 と 10.7 / 10.3)
- **次の層の予測 0.12〜0.13 ms/層**: 専門家の前に同期して回っていた (`prefetch=0` でも)。goroutine に移した (`prefetcher.guessLoop`。遅れた予測は数えて捨てる、`guesses too late`): 全部固定の層の中央値 1.35 → 1.23〜1.25 ms、4 GB で 10.6〜10.8 → 10.8〜11.6 tok/s (3 回ずつ交互)。遅れた予測は 0
- **残り約 0.2 ms** (1.25 と 1.05): 全部固定でも残る。眠ったコアの起き上がりと活性化 (spinners では縮まなかった)

### 6.5 先読みの読み方

メモリに無い専門家 1 つ (13.3 MB) を読む速さ (`TestColdRead`、`mincore` でメモリに無いものを選ぶ):

| 読み方 | GB/s | 1 専門家 |
|---|---|---|
| ページに触れる (1 goroutine、16 KB ずつの割り込み) | 0.5〜0.6 | 20〜27 ms |
| `madvise(WILLNEED)` してから触れる (それまでの先読み) | 1.7〜1.8 | 7.1〜8.1 ms |
| 同じく 4 goroutine で触れる、1 MB ずつ WILLNEED | 1.7〜1.9 | 7.0〜7.8 ms |
| ファイルを `pread` (1 MB ずつ、4〜8 スレッド) | 2.7〜2.8 | 4.7〜4.8 ms |

先読みを、1 MB ずつに分けて 4 goroutine で、メモリに無いもの (`mincore`) だけ `pread` してから触れる形に替えた (メモリにあるものを pread すると写すぶん帯域を使う)。4 GB、3 回ずつ交互 (メモリが苦しめの回、page-ins 22〜30 GB): 9.6〜9.8 → 9.8〜10.2 tok/s。先読み済みの層の平均 2.37〜2.46 → 1.93〜1.98 ms、p99 17〜18 → 10 ms。全部固定の層の中央値は 1.27〜1.30 → 1.34〜1.36 ms (読み手が帯域を使う)

それでも 1 専門家 4.8 ms は Neural Engine の部品の間 (約 1.5 ms) より長い。SSD から読むかぎり先読みで外れは隠しきれない

### 6.6 残り

- 先読みしていない外れ (層の 22 %、p99 30 ms): 2 層先を予測して早く読む、上位 8 個を読む (予測の 97.7 %。ただし読む量が 2 倍)
- メモリが苦しいときに先読みの数を減らす (間違った予測が要るページを追い出すか)
- 1 ステップ 80 ms のうち専門家 24 × 1.7 ≈ 41 ms、残り約 39 ms が Neural Engine の部品 (3.3)
- 電力 (`sudo powermetrics`) は今回測っていない。熱で絞られる Air では、1 トークンあたりの電力が持続する速さを決める
