# Core ML のバックエンド

[English](apple-coreml.md)

Core ML のバックエンド (`internal/backend/coreml`、macOS arm64) は**ベストエフォート**。コマンドは [Core AI](apple-coreai.ja.md) (macOS 27) で動く。Core ML を残すのは、ライブラリといくつかの道具が使い、モデルを CPU で動かせるため。このファイルには上限といくつかの数値だけを置き、すべては網羅しない。Neural Engine そのものの事実は [Neural Engine](apple-neural-engine.ja.md)。

## 使っているところ

| どこ | どう |
|---|---|
| 公開のライブラリ (`bonsaiai.Open`) | `backends_darwin.go` で最初に登録され、Apple の既定のデバイス。Core AI は名前で (`WithBackend("coreai")`、`BONSAIAI_BACKEND=coreai`) |
| CPU で動かす | `GraphSource.Units` / `BONSAIAI_COREML_UNITS=cpu` でパッケージを Core ML の CPU で動かす (参照、ANE のほうが悪いモデル) |
| バックエンドを指定しない `llm.OpenDevice` | `coreml` を優先。`bonsaiai-dev generate` と `formats` が使う |
| `bonsaiai-dev info`、`powerbench`、`probe`、`report` | Core ML のデバイスと配置の報告 (`MLComputePlan` が演算ごとのデバイスを示す) |
| `aicheck run -coreml` | Core AI の結果が怪しいときの比べる相手 |

`bonsaiai-chat`、`bonsaiai-image`、`bonsaiai-audio` は `coreml` を Core AI と読む。プログラムの形式 `.mlpackage` (MIL) は今も多くのモデルの変換先で、Core AI がそれを下げ、`milcpu` が動かす。

## 仕組み

- `purego/objc` で Core ML を呼ぶ (cgo なし)。計算ユニットは `CPUAndNeuralEngine` (既定で GPU は使わない)。入力は `Buffer` を写さずに包み、出力は `outputBackings` に書かせる (写し 0 回)。
- `GraphSource.Function` はマルチファンクションのパッケージの 1 つの関数を読む。
- 配置: `MLComputePlan` (macOS 14.4 以降) で各演算が ANE に置かれること。`PlacementStrict` は `ErrNotOnNPU` で失敗し、`PlacementPrefer` は動かして外れた演算を並べる。計画は見積もり (`Verified` は証明ではない)。最大 30 分待ち (`planTimeout`)、結果はキャッシュする (起動 42 → 16 秒)。

## 上限

| 上限 | 値 | BonsaiAI のすること |
|---|---|---|
| 1 つのパッケージの重み | 0.99 GiB は ANE で動き、1.02 GiB は CPU に落ちる (M1〜M3、どの形式も)。M4 は 2 GB | 変換のときに層ごとに分ける (下) |
| 仕事が少なすぎる | 小さなモデルは fp16 でも CPU に残る ((32, 256)→256 は CPU、(64, 512)→512 は ANE。1 行の部品) | 8 行に。小さな部品をまとめる。バッチの詰め物 (`tools/coreml/ane_pad.py`) |
| ANE のコンパイルのキャッシュ | 約 4 GiB ([Neural Engine](apple-neural-engine.ja.md)) | 少なく小さなプログラムに |
| BonsaiAI のキャッシュ | `WithCacheDir` (コマンドは `~/Library/Caches/bonsaiai`) の `.mlmodelc` と配置の結果。鍵はパッケージの木のダイジェスト | `cache.Prune`。変換し直すと古い項目が残る |
| e5 bundle のキャッシュ | `~/Library/Caches/<プロセス名>/com.apple.e5rt.e5bundlecache`。上限は見えない (49 GB) | テストのバイナリはコマンドの名前でビルドする |

Core ML が CPU に置く演算と代わりの形は [Neural Engine](apple-neural-engine.ja.md) にある。部品は `bonsaiai-dev info -model PART.mlpackage` で確かめる (`all on the NPU`)。

### 層ごとに分ける

- `bonsaiai-dev convert -target m1|m2|m3|m4|any|none` (既定 `any`) と `-parts N`。`llm.PlanSplit` が見積もったバイト数で釣り合わせる。出力は `out.partN.mlpackage` と `out.parts.json`。`internal/llm/chain` が部分 i の出力を部分 i+1 の入力につなぐ (写しなし)。
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
