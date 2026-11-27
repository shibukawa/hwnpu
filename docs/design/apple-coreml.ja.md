# Core ML のバックエンド

[English](apple-coreml.md)

Core ML のバックエンド (`internal/backend/coreml`、macOS arm64) は**ベストエフォート**。コマンドは [Core AI](apple-coreai.ja.md) (macOS 27) で動く。Core ML を残すのは、ライブラリといくつかの道具が使い、モデルを CPU で動かせるため。このファイルには上限といくつかの数値だけを置き、すべては網羅しない。Neural Engine そのものの事実は [Neural Engine](apple-neural-engine.ja.md)。

## 使っているところ

| どこ | どう |
|---|---|
| 公開のライブラリ (`hwnpu.Open`) | `backends_darwin.go` で最初に登録され、Apple の既定のデバイス。Core AI は名前で (`WithBackend("coreai")`、`HWNPU_BACKEND=coreai`) |
| CPU で動かす | `GraphSource.Units` / `HWNPU_DEV=coreml.units=cpu` でパッケージを Core ML の CPU で動かす (参照、ANE のほうが悪いモデル) |
| バックエンドを指定しない `llm.OpenDevice` | `coreml` を優先。`hwnpu-dev generate` と `formats` が使う |
| `hwnpu-dev info`、`powerbench`、`probe`、`report` | Core ML のデバイスと配置の報告 (`MLComputePlan` が演算ごとのデバイスを示す) |
| `aicheck run -coreml` | Core AI の結果が怪しいときの比べる相手 |

`hwnpu-chat`、`hwnpu-image`、`hwnpu-audio` は `coreml` を Core AI と読む。プログラムの形式 `.mlpackage` (MIL) は今も多くのモデルの変換先で、Core AI がそれを下げ、`milcpu` が動かす。

## 仕組み

- `purego/objc` で Core ML を呼ぶ (cgo なし)。計算ユニットは `CPUAndNeuralEngine` (既定で GPU は使わない)。入力は `Buffer` を写さずに包み、出力は `outputBackings` に書かせる (写し 0 回)。
- `GraphSource.Function` はマルチファンクションのパッケージの 1 つの関数を読む。
- 配置: `MLComputePlan` (macOS 14.4 以降) で各演算が ANE に置かれること。`PlacementStrict` は `ErrNotOnNPU` で失敗し、`PlacementPrefer` は動かして外れた演算を並べる。計画は見積もり (`Verified` は証明ではない)。最大 30 分待ち (`planTimeout`)、結果はキャッシュする (起動 42 → 16 秒)。

## 上限

| 上限 | 値 | hwnpu のすること |
|---|---|---|
| 1 つのパッケージの重み | 0.99 GiB は ANE で動き、1.02 GiB は CPU に落ちる (M1〜M3、どの形式も)。M4 は 2 GB | 変換のときに層ごとに分ける (下) |
| 仕事が少なすぎる | 小さなモデルは fp16 でも CPU に残る ((32, 256)→256 は CPU、(64, 512)→512 は ANE。1 行の部品) | 8 行に。小さな部品をまとめる。バッチの詰め物 (`tools/coreml/ane_pad.py`) |
| ANE のコンパイルのキャッシュ | 約 4 GiB ([Neural Engine](apple-neural-engine.ja.md))。測った値は下 | 少なく小さなプログラムに |
| hwnpu のキャッシュ | `WithCacheDir` (コマンドは `~/Library/Caches/hwnpu`) の `.mlmodelc` と配置の結果。鍵はパッケージの木のダイジェスト | `cache.Prune`。変換し直すと古い項目が残る |
| e5 bundle のキャッシュ | `~/Library/Caches/<プロセス名>/com.apple.e5rt.e5bundlecache`。上限は見えない (49 GB) | テストのバイナリはコマンドの名前でビルドする |

Core ML が CPU に置く演算と代わりの形は [Neural Engine](apple-neural-engine.ja.md) にある。部品は `hwnpu-dev info -model PART.mlpackage` で確かめる (`all on the NPU`)。

### ANE のコンパイルのキャッシュ

hwnpu のキャッシュ (`.mlmodelc` と配置) は当たっている。そこからの読み込みが遅いのは、システムのキャッシュの外れで、`MLModel` が `ANECompilerService` をまた走らせる。Core AI のキャッシュは別物で (`~/Library/Caches/coreai-cache/<OS>/<実行ファイル名>/`)、17 GB の書き出しでも残る ([Core AI](apple-coreai.ja.md))。

M1 16 GB、Sana-Sprint 0.6B の書き出しの 3 本を、それぞれ新しいプロセスで読んだ (`runner.Open`):

| | vae_s1 | dit_blocks_0 | vae_s0 |
|---|---|---|---|
| 1 回目 | 129.2 秒 | 14.7 秒 | 10.6 秒 |
| 2 回目 | 0.1 秒 | 0.1 秒 | 0.1 秒 |
| 3 回目 | 0.1 秒 | 0.1 秒 | 0.1 秒 |

容量の内側の外れ: Gemma 4 E2B (M3) は起動のたびに外れる (98 秒。部分ごとの `MLModel` の読み込みが 22〜25 秒)。hwnpu のキャッシュは当たっている (0.7〜0.8 秒)。CPU に落ちる演算をなくしても、部分を小さくしても (0.51〜0.55 GiB の 3 部分) 変わらない。形式では変わる (Q=8 を 3 回読んだ、`TestCallTimes`):

| | 部分 | 1 回目 | 2 回目 | 3 回目 |
|---|---|---|---|---|
| Gemma 4 lut4 | 2 部分 (0.54 + 0.52 GiB) | 外れ | 外れ | **当たり** (0.07 + 0.04 秒) |
| Gemma 4 lut6 | 2 部分 | 外れ (29 + 25 秒) | 外れ | 外れ |
| Gemma 4 lut8 | 3 部分 (0.69〜0.74 GiB) | 外れ (8〜10 秒) | 外れ | 外れ |

同じ大きさの LFM3 は当たり (lut6 は 2 回目から、lut8 の VL は 3 回目から)、Sana の Gemma 2 の int8 も M1 で当たる。Gemma 4 の lut6 と lut8 のグラフの何かが残らない。原因は分かっていない。

M2 (16 GB、macOS 27.2、2026-11-10) の `hwnpu-dev probe cache`:

- fp16 の 0.45 GiB のプログラムは 16 本 (7.25 GiB) まで当たり、20 本 (9.06 GiB) で外れる。
- 1.02 GiB: 1 本は当たり、2〜3 本は 3 回目から当たり、4 本以上は外れる。
- int8 は合計約 3 GiB まで、lut4 は合計約 1 GiB まで当たる。
- 合計の GiB ではなく、層の本数 (fp16 約 72、int8 約 50、lut4 約 36) で決まるように見えた。
- M1 の SANA 1.5 はこれに合わない: Gemma 2 int8 (約 2.2 GB)、fp16 の DiT (約 3.2 GB)、VAE が、ほかのプロセスが ANE を使っていなければ 2 回目のプロセスから全部当たる。M1 の違いか、合成のプログラムの違いかは分かっていない。

測るとき: 同じパッケージを使い回す (変換し直すと乱数の UUID が変わり、どちらのキャッシュも外れる)。ほかのプロセスが ANE 向けにコンパイルしていないことを確かめる。probe の動かし方は [PROBE](../../PROBE.ja.md)。

### 層ごとに分ける

- `hwnpu-dev convert -target m1|m2|m3|m4|any|none` (既定 `any`) と `-parts N`。`llm.PlanSplit` が見積もったバイト数で釣り合わせる。出力は `out.partN.mlpackage` と `out.parts.json`。`internal/llm/chain` が部分 i の出力を部分 i+1 の入力につなぐ (写しなし)。
- 切れ目をまたぐ値は参照と同じ尺度の残差の流れでなければならない (port-architecture のスキルに書いてある)。そうでないと切れ目が見つからない。

## 数値

| 何 | 結果 |
|---|---|
| 分割、M3 16 GB、lut4、64 層 (1.81 GiB) | 1 つのパッケージは CPU。2 部分 30.0 ms/token、4 部分 30.2、どれも全部 ANE (約 65 GB/s) |
| 分割、M4 (3.6 GiB) | 2/4/8/16 部分で 58.3 / 58.8 / 59.4 / 60.6 ms/token |
| 切れ目の費用 | 0.1 ms (M3)〜0.16 ms (M4)。誤差は 2 部分と 4 部分で同じ |
| 同じ層の形式ごと、M3 | lut4 / lut6 / lut8 で 1.81 / 2.65 / 3.49 ms。Q4_0 の並びは 7 倍遅いか CPU |
| バッチの詰め物、3 層の MLP (幅 128) | 詰めた 64 行は全部 ANE で 280 µs。CPU は 1 行 73 µs、64 行 104 µs (選んだときだけ使う) |

## 採らなかったもの

- 1 GiB を超えるためのマルチファンクションのパッケージ: 上限はパッケージ全体で数える (2 関数は全部 CPU で 12549 ms/token、2 パッケージは 30.6 ms)。
- 分割の数ごとの書き出し。重みを入力に。使うたびに部分を読み直す。GPU (非目標)。
- 配った 1 つのパッケージを実行時に分ける (`MemoryBudget`、`Reshape`): 分け方は変換のときに決める。
- すべてのモデルを Core ML で網羅する: コマンドは Core AI に移ったので、Core ML ではモデルごとの調整も測定もしない。

## 未解決

- LFM3-1.7B の 8 部分の分割だけ、キャッシュからの読み込みが遅い (23.7 秒)。
- `MLState` の KV キャッシュが 1 GiB の上限に数えられるか。

ANE のコンパイルのキャッシュ。急がない (コマンドが Core ML を使うのは `--cpu` だけ)。`hwnpu-dev probe cache` で:

- M1、M3、M4 の容量: 当たり続ける最大の合計と、バイト数で決まるか本数で決まるか。
- プログラムを開いたままか閉じるか、押し出される順、プロセスとの関係 (1 つのプロセスの中の読み直し、実行ファイル名、再起動の後)。1 台でよい。
- Gemma 4 の容量の内側の外れ: `-cache-format` に lut6 と lut8 を足す。書き出し済みのパッケージを読むモード。外れ続けるプログラムについて `e5bundlecache` に何があるか。
- `uuid()` を決まった値に: `internal/mlpackage/package.go` は乱数の UUID (`rand.Read`) を書くので、変換のたびにパッケージのバイトが変わる。決まった値なら、変換し直してもどちらのキャッシュも当たるはず。
