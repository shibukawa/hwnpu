# Apple の Neural Engine を hwnpu で使うための準備

Apple シリコンの Mac で、hwnpu から Neural Engine を使う手順です。機械の確認から、動作確認レポートの実行、モデルの用意までを扱います。

> **現状。** Apple バックエンドは、hwnpu の開発と計測に使ってきたものです（M3、macOS 27）。[MODEL_ARCHITECTURE.ja.md](MODEL_ARCHITECTURE.ja.md) の数値はここで測りました。

English: [INSTALL_APPLE.md](INSTALL_APPLE.md)

## 対応プラットフォーム

| プラットフォーム | 対応 | 補足 |
|---|---|---|
| Apple シリコンの macOS 14.4 以降 | 対応 | Core ML の `MLComputePlan` で、モデルが Neural Engine で動くことを hwnpu が確認できる。 |
| Apple シリコンの macOS 13〜14.3 | 一部 | モデルの読み込みと実行はできるが、配置を確認できない（`PlacementReport.Verified` が false）。厳格モードでは拒否される。 |
| Intel の Mac | 該当なし | Neural Engine がない。Core ML はすべて CPU で動かす。 |
| iOS、iPadOS | **非対応** | hwnpu は macOS 用の Go ライブラリで、iOS 向けにはビルドしない。 |
| Linux、Windows | 該当なし | Neural Engine は macOS の Core ML からしか使えない。 |

## hwnpu が必要とするもの

hwnpu は purego と Objective-C ランタイムを通して Core ML を呼びます。追加で入れるものはありません。Core ML は macOS の一部です。

| 部品 | hwnpu に必要か | 補足 |
|---|---|---|
| Apple シリコンの Mac | 必要 | Apple シリコンにはすべて Neural Engine がある。 |
| macOS 14.4 以降 | 必要 | 配置の確認に必要。読み込みだけなら 13 以降。 |
| Go | 必要 | 1.27 以降（`go.mod` を参照）。`CGO_ENABLED=0` で動くので、Xcode も C のツールチェーンも要らない。 |
| Xcode または Command Line Tools | 不要 | |
| Python と coremltools | `hwnpu-dev convert` が知らないモデルにだけ必要 | 対応する LLM のチェックポイント（今は LFM3）は `hwnpu-dev convert` で Python なしに変換できる（手順 4）。 |
| `sudo` | 電力の計測にだけ必要 | `hwnpu-dev powerbench` が使う `powermetrics` に root が要る。 |

Intel や AMD と違い、Core ML には「Neural Engine だけ」を指定できません。hwnpu は CPU と Neural Engine だけを許可して GPU を使わず、計算計画を確認して CPU に置かれる演算を報告します。

## 1. 機械の確認

```bash
sysctl hw.model && sw_vers
```

Apple のチップ（M1 以降）を載せた Mac なら使えます。すべての確認を行うには macOS 14.4 以降が必要です。

## 2. hwnpu の入手

```bash
go get github.com/shibukawa/hwnpu
```

以下のツールを動かすにはリポジトリをクローンしてください。デバイスの一覧を出します。

```bash
go run ./cmd/hwnpu-dev info
```

Neural Engine は `backend: coreml` として表示されます。

## 3. 動作確認

レポートは組み込みのテスト用モデルを実行し、配置を確認して時間を測ります。

```bash
go run ./cmd/hwnpu-dev report -o report.md
```

fp16 のモデルは、すべての演算が Neural Engine に置かれているはずです。2 つのモデルは拒否されるために入っています。fp32 のものと、バッチ 1 の小さな MLP（Core ML が CPU に置く）です。レポートは、厳格な配置（API の `WithPlacement(hwnpu.PlacementStrict)`、既定）がこれらを拒否することを確かめます。黙って CPU に落ちては困るアプリケーションではこの既定のままにしてください。同じ MLP をバッチ 64 に広げたものは Neural Engine で動きます。

呼び出し 1 回あたりのエネルギーを測るには、`powermetrics` のために root が要ります。

```bash
go run ./cmd/hwnpu-dev powerbench -model 'path/to/*.mlpackage' -o power.md
```

最初に sudo のパスワードを一度だけ聞き、Neural Engine と CPU（`-gpu` を付ければ GPU も）を比べます。M3 での結果と読み方は [MODEL_ARCHITECTURE.ja.md](MODEL_ARCHITECTURE.ja.md) にあります。

生成したモデルで Neural Engine を測るには、hwnpu-dev probe を使います。LLM の生成と同じ形のモデルを Go で作り、0.5B と 1B の重みで fp16、int8、lut4、int4 の速さを Neural Engine、CPU、GPU で測ります (`bench`)。次に形式ごとにモデルを大きくしながら Neural Engine から外れる境目を二分探索で絞り、境目のすぐ下のモデルを複数同時に読み込んで、上限がモデル 1 つごとか合計かを見分けます (`limit`)。Python もモデルのファイルも root も要りません。40 分ほどかかり、最大のモデルの 2 倍ほどの空きディスクが要ります (既定ではメモリの 4 分の 1 まで試し、`-max-gb` で変えられます)。結果は `npuprobe-<機種>-<日時>.md` に書き出します。クローンしなくても動きます。

```bash
go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe
```

1 トークンあたりのエネルギーは、`go run ./cmd/hwnpu-dev powerbench -decode 8,16 -gpu` が同じモデルを powermetrics で測ります。結果の読み方とこれまでの結果は [PROBE.ja.md](PROBE.ja.md) にあります。

## 4. モデルの用意

Core ML は `.mlpackage`（ML Program）、`.mlmodel`、コンパイル済みの `.mlmodelc` を受け付けます。`.mlpackage` は最初の読み込み時に `.mlmodelc` へコンパイルされます。`WithCacheDir` を指定するとその結果が保存され、次回からはコンパイルを省けます。

Neural Engine がモデルに求めること:

- **fp16 であること。** fp32 の演算は CPU で動きます。`compute_precision=FLOAT16` で変換してください。
- **形状が固定であること。** 動的な形状は CPU に落ちます。系列長やバッチの大きさごとに 1 つずつモデルを書き出してください。
- **1 回の呼び出しに十分な計算量があること。** 小さなモデルは fp16 でも CPU に置かれます。`tools/coreml/ane_pad.py` はバッチ次元を広げて Neural Engine に移るまで探します。ただし、得になるのは呼び出し 1 回あたり約 200 MFLOP 以上なので、先に [MODEL_ARCHITECTURE.ja.md](MODEL_ARCHITECTURE.ja.md) の電力の結果を読んでください。
- **重みがモデル 1 つあたり 1 GiB 未満であること。** M3 では、境目は fp16、int8、lut4 のどれでも保存された重みで 0.99〜1.02 GiB の間にあり（`hwnpu-dev probe limit`）、それを超えるモデルは丸ごと CPU に置かれました。上限はモデルごとで、0.91 GiB のモデルを 3 つ同時に読み込んでもすべて Neural Engine に載り、速さも変わりませんでした。大きなネットワークは複数のモデルに分ければ載せられます。全速で動くのは 4 ビットの表引き方式だけです。

本線は、bf16 のチェックポイントから Go の `hwnpu-dev convert` で変換することです。coremltools は、hwnpu-dev convert が構造を知らないモデルのための回避路です。

- **LLM のチェックポイントは hwnpu-dev convert（Go だけ、Python 不要）で。** `go run ./cmd/hwnpu-dev convert -config config.json -weights model.safetensors -o out.mlpackage -seq 32` が、既知の構造について `config.json` からグラフを組み立て、重みを 4 ビットの表引きにします。今は LFM3 に対応しています。Llama や Qwen 系は `internal/llm` に構造の定義を足せば対応できます。元のチェックポイントの選び方と Neural Engine が受け付ける形式は [MODEL_ARCHITECTURE.ja.md](MODEL_ARCHITECTURE.ja.md) にまとめてあります。
- **それ以外は coremltools（Python）で。** どんなグラフでも受ける汎用の変換器は hwnpu の範囲外なので、対応する LLM の構造ではないモデル（MiniLM のようなエンコーダ、画像モデルなど）は、PyTorch から `ct.convert(..., convert_to="mlprogram", compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS14)` と固定の入力形状で変換します。`tools/coreml/convert_minilm.py` と `tools/coreml/gen_sweep.py` が、ベンチマーク用に書いた実例です。

## 5. Core AI で動かす (macOS 27)

macOS 27 には Core ML のほかに Core AI があります。追加で入れるものはありません。

- **`hwnpu-chat` と `hwnpu-image` は Core AI だけで動き**、Mac では macOS 27 が要ります (`--backend` に `coreml` はありません。ほかのコマンドが受け取る `HWNPU_BACKEND=coreml` は、この 2 つでは Core AI の意味になります)。モデルは Core AI 向け (`.aimodel`、チェックポイントから Go で書き出す) に変換し、Core ML は使いません。`hwnpu-image --cpu` は確かめ用に、これまでどおり Core ML の書き出しを CPU のインタプリタで動かします。macOS 26 以前の Mac では、そう書いたエラーになります
- **`hwnpu-audio` も Core AI だけで動きます** (macOS 27)。音楽 (Magenta RealTime 2)、効果音 (MOSS-SoundEffect)、tts (LFM3-Audio-2B)、transcribe (Whisper)、pitch、notes を、チェックポイントや ONNX から Core AI 向け (`.aimodel`) に変換します。`--cpu` と、ONNX Runtime の無い機械で CPU で動かす pitch と notes は、Core ML の書き出しを CPU のインタプリタで読みます
- **置き場所**: どのコマンドも、モデルの Core AI のプログラムを同じ形で、モデルのディレクトリのチェックポイントの横に書き出し `NAME.coreai` (`.aimodel` と要るものを入れたディレクトリ) として置き、できあがったら横に `NAME.coreai.done` を置きます。プログラムが 1 本のモデルは `NAME.coreai/model.aimodel`、別々に変換し直す部分はそれぞれの書き出しです (音楽: 言語モデル、SpectroStream の復号と符号化、MusicCoCa の文と音。効果音: 文の符号器、DiT、DAC)。前の版の `coreai` ディレクトリと `NAME.aimodel` は、置き換える書き出しを書いたときに消します (docs/design.md 16.198)
- **変換の共有**: 定義に共有の層 (NAS のフォルダや S3 のバケット、`cache.layers`、docs/cache-sharing.md 7.4 節) があれば、書き出しは変換する前にそこを探し、手元で変換したらそこへ置きます。コマンドが行を引く表も同じです。ほかの機械が変換したものを動かす Mac は、書き出しを落とすだけでチェックポイントは要りません。Neural Engine の初回のコンパイルはそれぞれの Mac で要ります (docs/design.md 16.199)
- **変換**はモデルの部品 (プログラム) を、CPU のコア数と空きメモリが許すだけ並べて書きます (`HWNPU_JOBS=N` で数を決める。`1` で 1 つずつ)。端末では全体のバーと書いている部品ごとの行を出し、全部終わったら消します
- **ほかのコマンド**は Core ML が既定のままで、`--backend coreai` か `HWNPU_BACKEND=coreai` で、変換済みのモデル (`.mlpackage`、Go で Core AI のプログラムに下ろす) を Core AI で動かします

- **使えるコマンド**: `hwnpu-chat` の全部、`hwnpu-image` の generate (Anima、Chroma、Qwen-Image、Sana。Sana の書き出しは 512 × 512 と 1024 × 1024 を 1 つに持つ)、ocr、detect、pose、face-detect、depth、segment、label、embed。`hwnpu-audio` の全部 (docs/design.md 16.196、16.197)
- **結果と速さ**: Core ML と同じです (同じ文、同じ物や文字、埋め込みや深さのコサイン 0.9999 以上、1 回の呼び出しは ±10 % 以内)
- **違い**: コンパイルした結果の持ち方です。Core ML のキャッシュは約 4 GiB で、それを超えると 2 回目以降もコンパイルし直します (大きな画像生成のモデルで数分〜数十分)。Core AI は自分のキャッシュ (`~/Library/Caches/coreai-cache`) に、ディスクが減るまで持ち続けます。両方のキャッシュが当たっていれば起動は同じです
- **費用**: 重みがディスクに 3 回分置かれます (変換した `.mlpackage`、下ろした `.aimodel`、Core AI のキャッシュ)。空きが約 15 GB を切ると macOS がキャッシュを消し、次の起動でコンパイルし直します
- **hwnpu-chat** は、チェックポイントから Core AI のプログラムを直接書きます (画像のモデルと同じく、モデルのディレクトリの書き出し `NAME.coreai` に `.aimodel`、`.mlpackage` は作りません)。1 回に 8 トークンと 128 トークンのプログラムは 1 本の関数 2 つなので、重みはディスク、Core AI のキャッシュ、Neural Engine のどれにも 1 回分です (LFM3-1.7B: 書き出し 1.2 GB、Neural Engine 1.32 GiB、M3 で 2 回目の起動 0.4 秒。docs/design.md 16.185)。Qwen3.5-4B と gemma-4-E4B-it は M2 で 2K トークンのプロンプトを約 2 秒で読みます (16.191)
- 記録は [docs/design.md](docs/design.md) の 16.169〜16.197、まとめは [docs/improvements.ja.md](docs/improvements.ja.md) の 7.5 節にあります

## 困ったとき

| 症状 | 原因と対処 |
|---|---|
| `PlacementReport` に CPU の演算が並ぶ | fp32 の演算、動的な形状、`gather`（埋め込みの表引きは必ず CPU）、または Neural Engine には小さすぎるモデル。fp16 と固定形状で変換し直し、埋め込みの表引きはホスト側で行う。 |
| `Verified` が false | macOS が 14.4 より古いか、モデルが ML Program ではなく古い `neuralnetwork` 形式。macOS を更新し、ML Program として変換し直す。 |
| 読み込みがエラーを返さずに落ちる | 手書きの ML Program で値の名前にドットがあるか、パッケージが壊れている。coremltools と `hwnpu-dev convert` が書く名前は問題ない。 |
| fp16 なのにモデル全体が CPU で動く | 重みが大きすぎる（モデル 1 つあたり 1 GiB 以上。複数のモデルに分ける）か、Neural Engine が受け付けない並びの重み圧縮（表引きは行全体を 16 行ごとに 1 表、ブロックごとの int4 は不可）。 |
| 結果が PyTorch と数パーセントずれる | Neural Engine の行列積には約 2.5e-4 の絶対誤差の床があり、出力の小さい層が精度を失う。LFM3 では `hwnpu-dev convert -post-scale` が回避する。[MODEL_ARCHITECTURE.ja.md](MODEL_ARCHITECTURE.ja.md) を参照。 |
| CPU や GPU との比較がしたい | 読み込み時に `HWNPU_COREML_UNITS=cpu` か `gpu` を設定する（計測専用で、API ではない）。 |
| `hwnpu-dev powerbench` が `Password:` で止まる | `powermetrics` に root が要る。sudo のパスワードを一度入れれば、あとはツールが資格を保つ。 |
