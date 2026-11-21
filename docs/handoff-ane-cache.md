# 引き継ぎ: Neural Engine のコンパイル済みキャッシュの限界を M1〜M4 で測る

作成: 2026-11-10、整理: 2026-11-21。Sana の Mac 対応 (#99、[design.md](design.md) 16.145) で、hwnpu のキャッシュがあっても読み込みに 1〜2 分かかる原因を調べた。その続きとして、システム側のキャッシュが何をどれだけ持つかを測るためのメモ。

**今の位置づけ**: これは **Core ML** の話。その後 Mac のコマンドは全部 Core AI になり (hwnpu-image 16.190、hwnpu-chat 16.191、hwnpu-audio も)、Core ML は `--cpu` の CPU のインタプリタでしか使わない。Sana もやめた (16.244)。Core AI のキャッシュは別物で (`~/Library/Caches/coreai-cache/<OS>/<実行ファイル名>/`)、Core ML の約 4 GiB を超える 17 GB の書き出しでも残る (design.md 16.169)。なので下の 2 節の計測は急がない。前の版の「結果で決まること」(Sana の常駐や読む順) は外した。

測る道具は `hwnpu-dev probe cache` (旧 `cmd/npuprobe`。下の 4 節に動かし方)。

## 1. 分かっていること

### 1.1 「キャッシュがあるのに遅い」の正体

hwnpu のキャッシュ (`--cache`、既定は `~/Library/Caches/hwnpu`) が持つのは次の 2 つ:

- コンパイル済みの `.mlmodelc`
- 配置の確認 (`MLComputePlan`) の結果

これらは当たっている。遅いのはその先で、`MLModel` を読むときに Neural Engine 向けのコンパイル (`ANECompilerService`) がまた走っている。これがシステム側のキャッシュの外れ。

M1 (16 GB、macOS 27.2) で、Sana-Sprint 0.6B の書き出しから 3 本 (`vae_s1`、`dit_blocks_0`、`vae_s0`) を、別々のプロセスで 3 回読んだ。それぞれ新しいプロセスで、`runner.Open` の時間を測った:

| | vae_s1 | dit_blocks_0 | vae_s0 |
|---|---|---|---|
| 1 回目 | 129.2 秒 | 14.7 秒 | 10.6 秒 |
| 2 回目 | 0.1 秒 | 0.1 秒 | 0.1 秒 |
| 3 回目 | 0.1 秒 | 0.1 秒 | 0.1 秒 |

当たれば 0.1 秒、外れればプログラムの大きさに応じて数十秒〜2 分。

### 1.2 Sana のパイプライン (やめたモデル)

1 枚に 26 本のプログラム (約 4 GB) を順に読んで閉じる Sana で、2 回目も毎回 1〜2 分外れて見えた。ただしこれは、同時に動いていたコンパイルと書き出しの作り直しによるものだった (1.7)。

### 1.3 前の記録 (design.md 16.41、M3、LFM3)

- 2 本 (2.4 GiB) は 2 回目から全部当たり (起動 3 秒)
- 4 本 (4.8 GiB) は 4 回続けて起動しても毎回 3〜6 部分が外れた
- lut4 の 4 本 (3.2 GiB) は落ち着くと全部当たり
- lut4 の 2 部分 (0.8 GiB) は 1 回目 6.7 秒、2 回目 0.06 秒

ここから「容量 3.2〜4.8 GiB の LRU (4 GiB か)。部分の数ではなく容量で決まる」と見ている。ただし M3 の LFM3 だけでの見立てで、ほかのチップとの違いは測っていない。

### 1.4 キャッシュの置き場所 (分かった範囲)

- `~/Library/Caches/<実行ファイル名>/com.apple.e5rt.e5bundlecache/<OS のビルド>/<ハッシュ>/`: 実行ファイルごとにある
  - `hwnpu-image` には 16 GB 溜まっていた。書き出しを何度も作り直した残りで、削除済み
- 1.1 の計測で使ったテスト (`zzload.test`) の同じ場所は 36 KB しかなかった。それでも 2 回目は 0.1 秒で当たったので、当たりを持っているのはこのディレクトリではない
- `~/Library/Caches/com.apple.e5rt.e5bundlecache`: 11 MB
- 16.41 の見立てでは、本体は `aned` 側 (sudo が要り未確認)。実行ファイル名で分かれるのか、プロセスをまたいで共有されるのかも、ちゃんと調べていない

### 1.5 測るときの注意

- **書き出しを作り直すと別物になる**: `internal/mlpackage` が Manifest.json に乱数の UUID を書くため、パッケージのバイトが変わり、hwnpu のキャッシュも Neural Engine のキャッシュも当たらなくなる。測るときは同じパッケージを使い回すこと
- **ほかのプロセスに注意する**: ほかのプロセスが Neural Engine でコンパイルしていると、配置の確認が数分かかることがある (M1 で 2 分を超えた)。Neural Engine のロック (16.132、`llm.OpenDevice`) を取らないテストや他のツールが動いていないか確かめること
- **実行ファイル名で分かれる可能性がある**: キャッシュが実行ファイル名で分かれるなら、`go test` (`<pkg>.test`) と `hwnpu-dev` と `hwnpu-image` で結果が違うかもしれない。1.4 の置き場所の謎と合わせて確かめる

### 1.6 容量の内側でも当たらない: Gemma 4 (design.md 16.42〜16.43、M3)

容量だけでは説明できない外れがある。Gemma 4 E2B は、合計が容量より小さくても当たらない。

- 起動は毎回 98 秒。4 部分とも `MLModel` の読み込みが 22〜25 秒で、4 回続けて起動しても一度も当たらない
- hwnpu 側のコンパイル済みキャッシュは毎回当たっている (0.7〜0.8 秒)。外れているのは Neural Engine 側
- 次のことは、どれも原因ではなかった:
  - CPU に落ちる演算: 最後の `reshape` を無くし (`-logits folded`)、全部 NPU に載せても外れる
  - 部分の大きさ: 0.51〜0.55 GiB の 3 部分 (`-parts 3`) に分けても、3 回とも 16〜28 秒の外れ
- 形式で変わる。同じ Q=8 を 3 回読んだ (`TestCallTimes`):

| | 部分 | 1 回目 | 2 回目 | 3 回目 |
|---|---|---|---|---|
| Gemma 4 lut4 | 2 部分 (0.54 + 0.52 GiB) | 外れ | 外れ | **当たり** (0.07 + 0.04 秒) |
| Gemma 4 lut6 | 2 部分 | 外れ (29 + 25 秒) | 外れ | 外れ |
| Gemma 4 lut8 | 3 部分 (0.69〜0.74 GiB) | 外れ (8〜10 秒) | 外れ | 外れ |

- 同じ大きさの LFM3 は当たる。lut6 (0.60 GiB の部分) は 2 回目から、lut8 (VL、0.81 GiB) は 3 回目から
- Sana の Gemma 2 (int8、6 層ずつ 5 本、計 2.2 GB) は M1 で当たる (0.6B の 1 枚 7〜8 秒、design.md 16.145)。同じ Gemma の系列でも、int8 では起きていない
- Gemma 4 の lut6 / lut8 のプログラムだけが Neural Engine 側に残らない。グラフの何か (形、演算、パレットの並べ方など) が効いていそうだが、分かっていない
- lut4 は答えが悪い (16.26) ので既定にできない。今の Gemma 4 は起動に約 100 秒かかったまま

### 1.7 M2 の結果と、M1 の SANA 1.5 (2026-11-10〜11)

- M2 (Mac14,2、macOS 27.2、16 GB) の `hwnpu-dev probe cache` (容量、2026-11-10):
  - fp16 の 0.45 GiB は 16 本 (7.25 GiB) まで当たり、20 本 (9.06 GiB) で外れる
  - 1.02 GiB は 1 本は当たり、2〜3 本は 3 回目から当たり、4 本以上は外れる。システムのキャッシュは 1.02 GiB で重みの約 3 倍、0.91 GiB までは約 1 倍
  - int8 は合計約 3 GiB、lut4 は合計約 1 GiB まで
  - 合計の GiB ではなく、合成の層の本数 (fp16 約 72、int8 約 50、lut4 約 36) で決まるように見えた
- M1 の SANA 1.5 (design.md 16.152) は、この見方に合わない: Gemma 2 int8 約 2.2 GB と DiT fp16 約 3.2 GB と VAE で、ほかのプロセスが Neural Engine を使っていなければ 2 回目のプロセスから全部当たる (1 枚 1m12s)。1.2 の「毎回外れる」は、同時に動いていたコンパイルと書き出しの作り直しによるものだった。M1 と M2 の違いか、合成プログラムと実際のモデルの違いかは分かっていない

## 2. 残り (急がない、Core ML を Neural Engine でまた使うときに)

M2 の容量は済み (1.7)。残りは `hwnpu-dev probe cache` の 4 節のコマンドで、結果は `npu_profiles/npuprobe-<chip>-cache.md` と `.json` に:

- **容量** (M1、M3、M4): 大きさ S のプログラム N 本の合計を 1〜8 GB に振り、全部当たり続ける最大の合計。バイト数で決まるのか本数で決まるのか (M2 は合成の層の本数で決まるように見えた)
- **載せたまま / 閉じる**、**押し出され方** (同じ順と逆順)、**プロセスとの関係** (プロセスの中の読み直し、実行ファイル名、再起動の後): 1 台でよい
- **容量の内側の外れ** (1.6 の Gemma 4): `-cache-format` に lut6 と lut8 を足して、合成でも外れるか (形式のせい) / 当たるか (Gemma 4 のグラフのせい) を分ける。書き出し済みのパッケージを直接読むモード (`-cache-packages DIR...`) を足して、Gemma 4 と LFM3 の lut6 の部分を同じ手順で比べる。`e5bundlecache` の中身で、外れ続けるプログラムだけ書き込まれていないか
- 書き出しのたびにバイトが変わる元 (`internal/mlpackage/package.go` の `uuid()`) を決まった値にすれば、作り直してもキャッシュが当たるはず (別の改修の候補)

## 3. 関係するコード

- `internal/llm/runner`: `runner.Open` (1 本の読み込み)
- `internal/llm/chain`: 部品の読み込みと配置の確認
- `internal/backend/coreml/graph_darwin.go`: コンパイル済みキャッシュ (`compiledModel`)、配置のキャッシュ
  - `HWNPU_COREML_TRACE=1` で読み込みの内訳が出る
- `internal/backend/coreml/plan_darwin.go`: `MLComputePlan`
  - 待ちは 10 分 (#99 で 2 分から延ばした)
  - 内部の失敗は 1 回だけやり直す
- `internal/mlpackage/package.go`: `uuid()` (書き出しのたびにバイトが変わる元)
- `cmd/hwnpu-dev/probe` (`cache.go`)、`internal/probe`: 合成プログラム、子プロセスでの計測 (`child.go`)、報告 (`report.go`)

## 4. 動かし方 (`hwnpu-dev probe cache`)

1 つの構成は「S GiB のプログラム N 本」。プログラムは構成ごとに別物で、一度だけ書いて使い回す。回ごとに新しいプロセスがそれを順に読み、1 回ずつ実行する。hwnpu のコンパイルのキャッシュ (`.mlmodelc` と配置) は構成ごとのディレクトリに残すので、読み込みが長ければ Neural Engine のコンパイルのやり直し。読み込みと最初の呼び出しが `-cache-hit` (既定 1 秒) 以下なら当たり。実行中は hwnpu の Neural Engine のロックを持つ (子プロセスはロックなしで開く)。

```bash
# 容量 (全チップ): fp16 の 0.5 GiB と 1 GiB、合計 1〜8 GiB、各 3 回。1 時間以上かかる
go run ./cmd/hwnpu-dev probe cache -o npu_profiles/npuprobe-m1-cache.md

# 本数で決まるかを見るなら、小さいプログラムを足す (0.25 GiB の 16 本 = 4 GiB など)
go run ./cmd/hwnpu-dev probe cache -cache-sizes 0.25,0.5,1,2 -cache-totals 2,3,4,5,6

# 載せたまま / 閉じる
go run ./cmd/hwnpu-dev probe cache -cache-sizes 0.5 -cache-totals 3,4,5,6 -cache-hold close,open

# 押し出され方: 容量を少し超える合計で、同じ順と 1 回おきの逆順。回を増やす
go run ./cmd/hwnpu-dev probe cache -cache-sizes 0.5 -cache-totals 5 -cache-order same,reverse -cache-rounds 5

# プロセスの中での読み直しと実行ファイル名
go run ./cmd/hwnpu-dev probe cache -cache-sizes 0.5 -cache-totals 2,5 -cache-passes 2 -cache-names hwnpu-image,other

# 再起動の後: プログラムを -dir に残し、再起動してから同じ指定でもう一度 (1 回目から当たるか)
go run ./cmd/hwnpu-dev probe cache -cache-sizes 0.5 -cache-totals 2 -dir ~/probe-cache -cache-keep
```

報告の見方:

- 構成ごとの表に「各回の当たりの数」(例 `0/8, 8/8, 8/8`) と warm (2 回目以降が全部当たり) が出る。`-cache-names` の回は名前つきで出て、warm の判定には入れない
- 「Largest total that stays warm」の行が、大きさ・載せ方・順ごとの容量 (warm の最大の合計と、外れた最小の合計)
- 最後の表が各読み込みの秒数 (`*` が外れ、`[番号]` がプログラム) と、各回の後の `~/Library/Caches/<実行ファイル名>/com.apple.e5rt.e5bundlecache` と `~/Library/Caches/com.apple.e5rt.e5bundlecache` の大きさ

注意:

- `go run` の実行ファイル名は `hwnpu-dev`。`go build -o` で別の名前にすると、名前で分かれるかの比較にもなる
- 既定の -dir は一時ディレクトリ。`-cache-keep` なしでは構成が終わるたびに消す。最大で合計 (8 GiB) の 2 倍ほどの空きが要る (パッケージと `.mlmodelc`)
- `-cache-hold open` は合計の分だけメモリを使う。16 GB の機械では 8 GiB の構成を避ける (`-cache-totals` で絞る)
- ほかに Neural Engine を使うもの (Xcode のプレビュー、写真の解析など) を止めてから。hwnpu の別のコマンドはロックで待つ
