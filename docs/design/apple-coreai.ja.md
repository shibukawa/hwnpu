# Apple: Core AI のバックエンド

[English](apple-coreai.md)

Core AI (CoreAI.framework、macOS 27) は、hwnpu が Apple の Neural Engine を動かす道である。hwnpu はチェックポイントから Go で Core AI のプログラム (`.aimodel`) を書き、purego で読む小さな Swift の shim を通して動かす。Core ML は CPU のインタプリタ (`--cpu`) と参照のためだけに残す。[Core ML](apple-coreml.ja.md) を見よ。

Core AI にした理由:

- Core AI は端末で専用化した結果を、プログラムのハッシュを鍵に自分のディスクのキャッシュに持つ。Core ML のコンパイルのキャッシュのような約 4 GiB の上限がないので、2 回目の起動でコンパイルし直さない (Core ML はキャッシュから追い出された部品をコンパイルし直し、M1 で同じチャットのモデルが 4 秒〜13 分だった)
- 1 本のプログラムは M3 で 2.5〜3 GiB まで持てる (fp16 で 2.5 GiB は全部 Neural Engine、3 GiB は GPU に外れる)。Core ML は 1 GiB
- 1 本の `.aimodel` の関数どうしは、ディスクでも、Core AI のキャッシュでも、Neural Engine のメモリでも重みを共有する。チャットのモデルはどの機械でもプロンプトを読むプログラムを持てる
- Core ML で一部が Neural Engine から外れたプログラム (Qwen3.5-2B の本体、MTP の部品) が全部載る

| 2 回目の起動、M1 16 GB | Core ML | Core AI |
|---|---|---|
| Qwen3.5-0.8B | 12 分 54 秒 | 2.3 秒 |
| LFM3-1.7B | 57 秒 | 4.5 秒 |
| Qwen3.5-4B (M2 16 GB) | 5.8 秒、生成 14.5 トークン/秒、2041 トークンのプロンプト 29.5 秒 | 0.7〜0.8 秒、20.5〜21.9 トークン/秒、4.0 秒 |

それ以外の生成の速さは ±10 % 以内で同じ。M3 ではチャットのモデルが Core AI のキャッシュから 0.2〜0.5 秒で立ち上がる。

## どのコマンドが使うか

| コマンド | Mac では |
|---|---|
| `hwnpu-chat` (チャット、埋め込み、`systemone`) | Core AI だけ。`--backend` は `coreai`、`intel`、`levelzero`、`openvino`。`coreml` は選べない |
| `hwnpu-image` (生成と画像を読むモデル) | Core AI だけ。`convert` の既定は `coreai`。`convert --backend coreml` は `--cpu` が読むものを書くために残す |
| `hwnpu-audio` (music、sfx、transcribe、tts、pitch、notes) | Core AI だけ |
| `hwnpu-dev models package MODEL/coreai-ARCH` | `coreai-build` で prebuilt を作る (下) |
| `hwnpu-dev convert` | 既定は `coreml`。`llm.Convert` は `coreai` も取る |

- Core AI がない (macOS 26 以前) とき、これらのコマンドは「Core AI が要る、macOS 27」というエラーで止まる (`errNoCoreAI`)。Core ML には落とさない
- ほかの道具のために残っていることのある `HWNPU_BACKEND=coreml` は `coreai` に読み替える
- `--cpu` と、ONNX Runtime のない機械の detect / pose / ocr / pitch / notes の CPU の経路は、Core ML の書き出し (`.mlpackage`) を `milcpu` で読む。どれも小さなモデルで、Core ML の Neural Engine の問題は関わらない
- `llm.NativeFormat` はデバイスが Core AI なら `coreai` を返す。`llm.ExportDevice` は `coreai` の書き出しを Core AI で開く。前の Core ML の書き出しも、読み込むときに下ろして Core AI で動かせる

## Go で `.aimodel` を書く

`.aimodel` はディレクトリ:

| ファイル | 中身 |
|---|---|
| `main.mlirb` | `aicode` 方言の MLIR バイトコード v6。256 バイトを超える定数は resource (重み) |
| `main.hash` | `main.mlirb` の sha256。Core AI のキャッシュの鍵 |
| `metadata.json` | Core AI のメタデータ |
| `hwnpu-io.json` / `hwnpu-io-NAME.json` | 入出力。関数ごとに 1 つ |
| `hwnpu-meta.json` | 説明などのメタデータ (`llm.MetaFile`、`ReadMeta`) |

- `internal/coreai/mlirbc` がバイトコードを書いて読み、`internal/coreai/ailower` が MIL をそれに下ろす。coreai-core の出力を `tools/coreai/bcdump` で読み戻して合わせた。属性と型は方言のテキストの形で書き、op の properties は方言の並び (必須は属性の番号、省略できるものは `番号 << 1 | 1`、無ければ 0)
- Python も Xcode の道具も要らないので、書き出しはどの OS (Linux も) でも作れる
- `llm.WritePart` の `BackendCoreAI` は MIL を組んで `.aimodel` を直接書く。間に `.mlpackage` を置かない。バイト列は `.mlpackage` を下ろしたものと同じ (`TestCoreAIPartMatchesPackage`、fp16 と lut4) で、2 つの道は Core AI のキャッシュの項目を共有する
- `ailower.FormatVersion` (今は 2) は共有する書き出しの鍵の一部。同じ名前の書き出しのプログラムが変わる変更では手で上げる
- 書くときのメモリ: `mlirbc.Writer.Spill` は blob を `ailower.SpillKeep` (512 MiB) までメモリに、残りを `NAME.aimodel.blobs.tmp` に持ち、`WriteTo` がファイルの最後の resource の節へ流す。部品を下ろし終えたら呼び出し側が `.milblob.tmp` を消す (消さないと部品 1 つがディスクに 3 つ分を持つ)。全部を一時ファイルにすると、ほぼ埋まったディスクで遅くなった。512 MiB がいちばんよかった
- 部品は並べて書く (`console.Jobs`): コア数と、使えるメモリ (Linux は MemAvailable、ほかは物理メモリの半分) の 3/4 を部品 1 つの見積もり (Anima 1 GiB、Sana / Chroma / Qwen-Image 2 GiB、`llm.ConvertAI` 4 GiB) で割った数の小さい方。`HWNPU_JOBS=N` で決める。出力は 1 つずつ書いたものとバイト単位で同じ (`metadata.json` の creationDate を除く)。Ryzen AI (20 コア、59 GB)、Qwen-Image-2.1 Turbo lut6 で 2 分 16 秒 → 50 秒

### 下ろし方に Core AI が求めること

| 求めること | ailower の書き方 | そうしないと |
|---|---|---|
| 位置 | Core AI 自身の `#aicode.debuginfo.location_v1`。ブロックの引数にも | builtin の `loc` は「versioned IR」にできない |
| 縮約 | 複合の `reduce_max_v1` / `reduce_sum_v1` の中に `primitive.reduce_v1` (形の決まらない型) | — |
| 行列の積 | 複合の `broadcasting_batch_matmul_v1` の中に `broadcast_to` と `batch_matmul` | 基本の演算を外に並べると、int8 の重み (`blockwise_shift_scale_v2`) があるときプログラム全体が Neural Engine から外れる。重みの reshape では MPSGraph が形を誤る |
| softmax | `exp(x − max) × reciprocal(sum)` | `exp / sum` や softmax は matmul と注意の演算にまとめられ、Neural Engine から外れる |
| 2 項演算 | 同じ形どうしだけ。広げるのは `broadcast_in_dims` | — |
| 定数での割り算 | 逆数との積 (`HWNPU_DEV=ailower.divide=1` は調べるために割り算のまま) | Neural Engine が割る代わりに掛ける (PP-OCR の検出器でコサイン 0.05) |
| 内側の次元が 1 の matmul | 掛け算 (外積) で書く | (1×1)·(1×N) は Neural Engine で誤る (M = 1 で相対誤差 0.28) |
| softplus | `relu(x) + log(1 + exp(−|x|))` | Core AI が `log(1 + exp(x))` をまとめ、x > 10.4 で 0 になる |
| 隣り合う lut の表 | 同じにしない (`mlpackage.DistinctNeighbours`: 値を 1 つ 0 の方へ 1 ulp) | `coreai-build` の ANE のコンパイラが落ち (「per-cout group … not content-uniform」)、部品を GPU に回す |

ほかに `max_pool`、`conv_transpose` (MIL の pads に切り詰め)、`pad` の replicate / reflect、`elu`、LUT (lut4〜lut8) などの下ろし方がある。新しいものは coreai-torch で小さな torch のモジュールを変換し (`uv run --offline --prerelease=allow --python 3.13 --with coreai-torch==0.4.3 --with torch`)、プログラムを出して `bcdump` でバイトコードを読む。演算の operand の順は coreai-core の `_coreai_ops_gen.py` にある。

`ailower` は下ろしながら、知られている Neural Engine の罠を記録する (`HWNPU_DEV=ailower.notes=1` で出す。`aicheck lint` は書き出しを探す)。今あるのは、最後の軸の slice の開始を 32 で割った余りが 1〜14 のもの (4094 を超える値が Inf になる)。

## 書き出し: `NAME.coreai`

どのコマンドも Core AI のプログラムを同じ形に置く (`internal/llm/export.go`):

- モデルのディレクトリのチェックポイントの横に `NAME.coreai/`、できあがったら `NAME.coreai.done` (相対パス、`cache.WriteDone`)。書き出しは一度に変換する単位で、途中で切れたものは空にして書き直す (`StartExport`、`FinishExport`)。`ExportDone` は前の空の `.done` も読む
- 1 本のモデルは `NAME.coreai/model.aimodel` (`ProgramPath`)。分けたモデルは関数を持つ `model.partK.aimodel`、プログラムごとに `PATH#関数` を並べる `NAME.parts.json`、注意のプログラム
- 版が別々に変わるものは別の書き出し (music、sfx、transcribe はいくつかずつ持つ。[音声のモデル](audio-models.ja.md))
- 前の置き方を置き換える書き出しを書き終えたら、前の `coreai/` ディレクトリと `NAME.aimodel` / `.done` を消す
- Core ML の書き出し (`--cpu` 用) と Intel の OpenVINO IR は自分の置き場所のまま
- 書き出しは詰めて取得元で共有する: [配布](distribution.ja.md)

## 1 本の `.aimodel` に関数を複数

`ailower.LowerFunctions` は 1 本の `.aimodel` に `graph_v1` の関数を複数書き、同じバイトの blob は 1 回だけ書く。shim は `PROGRAM.aimodel#NAME` を開く。Core AI はその資産に 1 つの項目を作り、関数ごとに別の `AIModel` で開いても Neural Engine は重みを 1 回分しか持たない。

- LFM3-1.7B (M1、8 トークンと 128 トークンを関数 2 つ): `.aimodel` 2.43 → 1.22 GiB、Core AI のキャッシュ 2.53 → 1.32 GiB、`mem.NPUWired` 2.56 → 1.34 GiB、生成 +13 %。footprint は変わらない (重みはドライバのメモリにある)
- `llm.ConvertAI` (hwnpu-chat): 1 つのモデルのプログラム (1 回のトークン数、最後の行のロジット、`KVOnly`、`state-rows`) をいつもどおり分け (`PlanPaged`、`PlanSplit`)、同じ層から始まる部品を 1 本の `.aimodel` の関数にする (`WriteAIFunctions`。例 `base.partK.aimodel#q8` と `#q128-last`)。Gemma 4 の `KVOnly` の部品は同じ層から始まる生成側の部品に入る。層のない部品 (`NoLayers`) は別。重みのない `attn` / `mass` はプログラムごと。`.aimodel` の名前は中のプログラムを全部表す (`…-c262144-q8-q128-last.part1.aimodel`) ので、`--draft` を変えてもほかの組が指すファイルを上書きしない。MTP の層と符号器は 1 本ずつ。`PrepareEmbedder` は 128 と 1024 を 1 本の関数 2 つにする
- Core AI では重みを 1 回分と数える (`sharesWeights`、`heldBytes`) ので、`chat.Prepare` はどのメモリの機械でもプロンプトを読むプログラムを持ち、Core ML のコンパイルのキャッシュの上限 (`compileCacheBytes`) も当てない。M1 で Qwen3.5-2B、LFM3-VL / Audio / Omni-2B、PLaMo 3 Translate 2B のプロンプトの読みが 2.3〜3.2 倍速く、Neural Engine のメモリは 0.1〜0.14 GiB 増えるだけ
- 関数の名前は `p` + 中身の sha256 の 16 桁。どの機械でも同じ
- Core AI は資産の関数を全部一緒にコンパイルする。コンパイラが 1 つの関数で失敗すると、ほかの関数も Neural Engine から外れる (M1 で Qwen-Image の VAE の 1024² の段が 512² の段も GPU に連れていった。30 → 109 ms)。大きさごとの部品を先に単独で確かめる

読み込むときに `.mlpackage` を下ろす場合の自動のまとめ (`internal/backend/coreai/group_darwin.go`):

- 兄弟: 同じフォルダの `.mlpackage`、または隣のフォルダの同じ名前のもの (別の大きさの画像の書き出し) のうち、重みの大きい順 3 つの blob (`groupBlobs`) が同じバイトのもの。読むのはメタデータとその blob だけ。1 MiB 未満の重み (`groupMin`) はまとめない
- 部品ごとの記録 `KEY.aifn` が資産と関数を書くので、2 回目からは自分の部品だけをハッシュする。`groupRule` (今は 3) が違う記録は見直す。兄弟が増えると (3 つ目の大きさ) 全員でまとめ直し、小さいまとまりは消す。前の単体の資産とその Core AI の項目は消す (`cai_forget`)。同じフォルダを 2 つのプロセスがまとめないよう `cache.Lock` を取る
- 関数が Neural Engine に全部載らないまとまりはばらし、単体のままにする (`Alone`)
- 落としてきた `.aimodel` はそのまま開く。ハッシュもまとめ直しも消すこともしない

## 実行: shim

- `internal/backend/coreai/shim/shim.swift` を `tools/coreai/build-shim.sh` で `libcaishim.dylib` にし、go:embed して purego で読む。`CGO_ENABLED=0` のまま
- `Compile` は Neural Engine を指定してプログラムを開く。配置は Core AI のキャッシュの MPSGraph の manifest の `mps.fullyPlacedOnANE` (関数ごとには `Entry Function Attributes` の下) で見る。delegate のフォルダは関数によらず `main-this-delegates`
- 1 本のプログラムの呼び出しは 1 つずつ (`graph.mu`)
- 入力: `cai_input_array(h, i)` は入力 i の形の `NDArray` を作り、その中身 (密な行優先) の番地を返す。呼び出し側はそこへ直接書き (`backend.InputAllocator`、`hwnpu.Graph.AllocInput`)、`cai_run` はその配列を写さずに渡す。配列は同じ形のどの入力にも、ほかのプログラムにも束ねられる (生成と読み取りのプログラムがキャッシュを共有し、注意のプログラムが塊を順に受け取る)。形の違うバッファは写す。`chain` はキャッシュ、リング、窓、状態、マスク、注意の塊をこれで取る。画像と音声のプログラムは今も写す。M1 での効果: LFM3-1.7B の 3101 トークンのプロンプトの後 39.8〜40.6 → 43.0〜43.1 トークン/秒 (注意 1 回 1.34 → 1.0 ms)、PLaMo 3 Translate 2B 12.1 → 12.6〜13.1 トークン/秒。PLaMo 3 2B のリングの持ち分 252 → 84 MiB。写しは塊とともに増える: 塊 8192 で注意 1 回 2.07 ms のうち 0.56 ms
- M1 の写しは約 29 GB/s。何もしないプログラムの 1 回は 0.21 ms (M1)、呼び出しの下限は 160〜175 µs (M2)
- Neural Engine に載るプログラムは 1 プロセス 112 個まで (M2)。113 個目は読み込みに失敗し (「no ANE resources」)、Core AI は黙って GPU で動かす。デバイスは読み込んだプログラムを数え (`aneLive`)、上限の近くで失敗したらそう言う。gpt-oss-20b の書き出しは約 84 個、ほかのチャットのモデルは 40 個以下。2 つのモデルを 1 プロセスに持つと越えうる
- Core AI は Core ML の `GraphSource.Units` のように計算するユニットを選べない。`HWNPU_COREAI_UNITS=cpu` は全部を Core AI の CPU で動かす (Neural Engine なしのテスト用)
- Core AI のキャッシュからの読み込みは 0.2〜0.3 秒なので、1 本を動かす間に次を読める (`runner.InTurn`、`runner.Ring`)。[Apple の画像のモデル](image-apple.ja.md)

## キャッシュとディスク

モデルの重みがある場所:

| 写し | 場所 | 鍵 / 注 |
|---|---|---|
| 書き出し | `NAME.coreai/…/main.mlirb` | 既定で残す |
| Core AI の項目 | `~/Library/Caches/coreai-cache/<OS のビルド>/<実行ファイル名>/<main.hash>/<ID>/model.aimodelx` | 鍵は `main.hash` と全項目共通の ID だけ。別の場所に写した資産も、ビルドし直した同じ名前の実行ファイルも 0.02 秒で当たる。大きさは重みとほぼ同じ |
| aned のコンパイル済みの ANE のコード | `/Library/Caches/com.apple.aned` (SIP。sudo でも読めない) | 鍵は署名の識別子。Go の実行ファイルは ad-hoc 署名の `a.out` なので開発ビルドはどれも共有する。正規の識別子で署名したリリースは共有しない。purgeable。0.8 GB の項目に約 1.6 GB |
| MPSGraph の作業 | `$TMPDIR/com.apple.MetalPerformanceShadersGraph` (GPU に外れた関数ごとに 0.6 GB)、`…GraphHost` (`coreai-build` の作業。12 GB 残っていたことがある) | 落ちたプロセスの分は残る |

ここから決まること:

- Core AI のキャッシュは実行ファイル名ごと。名前を変えたテストのバイナリは丸ごと 1 回分 (1 モデル 1.8 GB) 増やす。名前はそのままにしてディレクトリで分ける (`base/hwnpu-chat`、`new/hwnpu-chat`)。`hwnpu-image` の名前で作った `aicheck` はそのコマンドのキャッシュに当たる
- OS を更新すると、全部を `main.mlirb` からコンパイルし直す
- ディスクの空きが少ないと、macOS が Core AI の項目、aned のコード、ときに `~/Library/Caches` の下の hwnpu のキャッシュの根まで消す (既定の方針は `storagePressure` で消し、使っていない順ではない)。M2 では空き約 15 GiB を切ると始まり、空き 22 GB でも仕上げたばかりの prebuilt を消した。初回は 25 GB 以上空ける
- aned のコードが消えて項目が残ると、`specialize` / `loadFunction` がすぐ失敗する (`_GenericObjCError.nilError`、`_GenericObjCError 0`)。shim は項目を消して読み直す。ディスクに空きがなければ Neural Engine のコンパイラが失敗し、GPU に回る。`HWNPU_DEV=coreai.debug=1` は「the cached entry failed (…), compiled again」を出し、GPU に落ちた項目を消し、空きが 10 GiB 未満なら警告する
- Core AI に項目ができたら `main.mlirb` は読まれない。空にしても資産は読める (`main.hash` は残す)
- `hwnpu cache` が空きの下限まで切るときは、prebuilt のある部品の Core AI の項目を先に消す。prebuilt から数秒で戻る
- Neural Engine の重みはカーネルの wired メモリ (`mem.NPUWired` で読む) で、プロセスの footprint ではない: [メモリと電力](memory-and-power.ja.md)

## 端末でのコンパイル

- 時間は全部 `ANECompilerService`: root の 1 インスタンス、1 スレッド、機械全体で 1 本の直列の列。ほかのプロセスやスレッドもその後ろに並ぶ (10 プロセス × 10 部品が 67 秒で一斉に終わる。順に投げて 79 秒)。`SpecializationOptions` に調整はない
- コンパイルの時間は部品のブロック数に超線形 (Chroma の DiT、1024²、M2: 1 ブロック 30 秒、2 ブロック 76 秒、4 ブロック 219 秒)。呼び出しは線形 (360 → 1428 ms)
- 常駐するプログラムは最初の呼び出しで wired メモリを取る: 約 0.36 GB の固定 + lut8 の重みの 1.2 倍 (Chroma の 1024²)。収まる間は切り替えは無料、物理メモリを超えるとカーネルが毎回ワイヤし直す。だから 16 GB では DiT を今より細かく分けない ([Apple の画像のモデル](image-apple.ja.md))
- `AIModelCache.model(for:options:)` は項目の有無を 1〜20 ms で答える (aned のコードの有無はわからない)
- Core AI の中で変換に失敗するとプロセスごと落ちる (LLVM)。下ろし方の誤りは先に `aicheck` で見つける
- Neural Engine のコンパイルの失敗はエラーにならず、プログラムは GPU で動く。配置を確かめる

## prebuilt (`coreai-build`)

Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`) に `coreai-build` がある:

```
xcrun coreai-build compile X.aimodel --platform macOS --architecture h14g --preferred-compute neural-engine --output out/
```

- `X.h14g.aimodelc` を書く: `main.hash` (元と同じ)、`metadata.json` (`sourceHash`)、`main-ARCH.mlirb` (1 KB、重みなし)、MPSGraph のパッケージ (`manifest.plist`、`specialized_model_1.mpsgraph`、`original_model_0.mpsgraph`、ANE の LLIR (`binary_0.llir.bundle/…/*.mlir.bc`、約 1 %)、`resources.bin` (重み、99 %))
- ANE のコードが出るのは `--preferred-compute neural-engine` のときだけ。既定の `none` は GPU だけ。1 プロセス 1 スレッド。`--architecture` を省くと 22 種類を順に作り、最初に落ちたところで止まるので、1 種類ずつ別のプロセスで作る
- オフラインのコンパイラが乗せられない部品は、stderr に `ANECCompileOffline() failed … CompilationFailure`、LLIR なし、`ANERegionsHash` の後半が空の sha256 で、終了コードは 0。本当の理由は stderr でなく os_log (プロセス `coreai-build`、`ANEHostCompiler`) に出る

読み込み (`coreai.Prebuilt`):

- `.aimodelc` を `AIModel(contentsOf:)` に渡すと、ANE のコンパイラの代わりに `ANELargeModelCompilerService` (CPU 25 %、1 スレッド) が LLIR を仕上げる。項目はネイティブのコンパイルと同じハッシュと ID に入り、`resources.bin` はバイト単位で同じ。次からは `.aimodel` の道で 0.01 秒で読む
- 規則: `hasEntry` が偽で、この種類の同じハッシュの prebuilt があり、その manifest に `mps.fullyPlacedOnANE` があれば、それを開く。`HWNPU_COREAI_PREBUILT=off` で端末でコンパイルする。shim は `PATH.aimodelc#NAME` も分ける。仕上げに失敗した prebuilt (`failedToSpecialize`) は端末でコンパイルする
- `coreai.LeftToDevice`: 名前が `text` で始まるプログラム (と `_device` で終わるもの) はいつも端末でコンパイルし、`Build` も外す。Chroma の T5 は `Request cancelled` で落ち、FLUX.2 klein の Qwen3 は prebuilt のほうが遅かった (112 秒対 81 秒)。VAE はレシピで決める (`model-recipes.yaml` の apple の `prebuild`、`do: device`): FLUX.2 と FLUX.3 klein はいまはどのチップでも端末 (M1 では prebuilt の VAE が 512² で 112 行ごとに縞を描き、1024 × 768 の VAE は `Request cancelled` で落ちた。M2 では 512² に同じような帯が、prebuilt でも端末のコンパイルでも出た)、Anima は M2 で端末 (書き出し v5 までは `coreai-build` がどのチップでも 10 段のうち 8 段を落とした。v6 は全段が通る。M2 の h14g の prebuilt は 160 ピクセルごとの帯を描き、M1 の h13g は正しかった)。M3 では FLUX.2 klein の復号器の 10 段のコンパイルに 2799 秒かかり、prebuilt の仕上げは 0.9 秒で、出力はバイトまで同じ (2026-11-25、[image-apple.ja.md](image-apple.ja.md)): それを取るのは FLUX の別の変更。M1 で prebuilt の VAE ですでに描いた機械はその項目を持ち続けるので、`~/Library/Caches/coreai-cache/<OS のビルド>/hwnpu-image/` を消す
- ほかの種類の `.aimodelc` は読めない (`incompatibleCompiledAssetArchitecture`)。名前と `deviceDescriptor` を書き換えても読めない
- prebuilt から仕上げたプログラムの呼び出しは遅い: 3〜5 % (Chroma、M2)、8〜16 % (M1)
- 端末で aned がコンパイルしたコードは持ち出せない (SIP、署名の識別子ごと)。配れるのは `coreai-build` の出力だけ

### 機種

`coreai-build inspect X.ARCH.aimodelc` の `Supported chips` による名前: h13 が M1、h14 が M2、h15 が M3、h16 が M4、h17 が M5、h18 が M6。`g` が無印、`s` が Pro、`c` が Max、`d` が Ultra。h15m はもう 1 つの M3 Max、h17p は A18 / A18 Pro、h17d は名前が出ない。`--platform` が違えば並びも違うので、macOS の 22 種類は Mac のもの。

| 結果 | 種類 | Mac |
|---|---|---|
| 作れて ANE に乗る (12) | h13g、h14c、h14g、h15c、h15g、h16c、h16g、h16s、h17c、h17d、h17g、h17s | M1、M2、M2 Max、M3、M3 Max (h15c)、M4 の全部、M5、M5 Pro、M5 Max (と h17d) |
| `coreai-build` が落ちる (`ZinIrCompilerParameters::PropogateGlobalOptionsToPerProcedureOptions` の SIGSEGV、プログラムによらない) (9) | h13c、h13d、h13s、h14d、h14s、h15d、h15m、h15s、h17p | M1 Pro / Max / Ultra、M2 Pro / Ultra、M3 Pro / Ultra、M3 Max (h15m)、A18: 端末でコンパイル |
| ANE のコードが出ない (`redefinition of symbol named 'hwnpu_N'`)、全部 GPU | h18g | M6: 作らない。端末のコンパイルでどうなるかは実機がなくわからない |

`--preferred-compute none` (GPU) なら 22 種類とも通る。12 種類の間で `resources.bin`、`original_model_0.mpsgraph`、`main.hash`、LLIR の `Manifest.json` は同じ。ANE の LLIR は種類ごとに違う (11 通り。同じなのは h17c と h17d だけ)。`manifest.plist` (`deviceDescriptor` に ANE の世代と LLIR のハッシュ)、`specialized_model_1.mpsgraph`、`main-ARCH.mlirb`、`metadata.json` (1〜40 KB) は違う。

### 詰めて配る

- `resources.bin` は `main.mlirb` の resource の blob を並べ替えたバイト。`coreai.Pack` はそれを `resources.bin.map.json` (16 バイトの頭と 64 KiB ごとの位置の索引、合わない数バイトは literal) に置き換える: Chroma の 1 ブロックの部品で 231 MB → 2.97 MB。中身の多くが `main.mlirb` にないファイルはそのまま置く
- `coreai.Assemble` は小さなファイルをハードリンクし、`resources.bin` を `main.mlirb` から書いて大きさと sha256 を照らす。違うバイトコードからは作らない。バックエンドは一時フォルダに組み立てて Core AI に開かせ、開いたら消す (項目が重みを持つ)
- `coreai.Bundle` は種類ごとに 1 つの `prebuilt.tar.zst` (zstd。機械は自分の種類だけを取る) を作る: Chroma の 1 ブロックで 3.02 MB → 1.22 MB
- 鍵: `coreai.ProgramsDigest` (各 `.aimodel` の場所と `main.hash`)、種類、書き出しの名前。`converted/MODEL/coreai-ARCH/EXPORT/v0-HASH`。モデルは書き出しの `entry.json` のもの
- `Compile` は項目も手元の prebuilt もない部品に出会うと、書き出しごと・プロセスごとに 1 回取得元に聞き (`coreai.FetchPrebuilt`、chat が設定)、`EXPORT.prebuilt/ARCH/` に置く。`EXPORT.coreai.prebuilt` は種類 `prebuilt` のキャッシュの単位で、モデルの量に数えず、切るときも残す。prebuilt のある機械にも `.aimodel` の書き出しは要る (重みはそこから)。書き出しの取得元と詰め方: [配布](distribution.ja.md)

`hwnpu-dev models package MODEL/coreai-ARCH|EXPORT[/coreai-ARCH] [-j N] [--force] [-n] [--packed] [--no-push]` (`coreai.Build`):

- 部品 × 種類ごとに `coreai-build` を 1 プロセスずつ、`-j` 個並べて走らせる (既定: 6 GB を超えた 4 GB ごとに 1、コアの半分まで。16 GB で 2)。`.prebuild-*` に書いてから名前を付け替える。部品ごとに作って詰め、`.aimodelc` 全体は消す (Sana の 27 本 × 12 種類を全体で持つと 50 GB)。詰めて種類ごとの束を書き戻す取得元に置く。`--packed` は書き出しの横に詰めるだけ、`--no-push` は全体のままソースの横に残す。モデル名を渡すと、先に coreai 向けに用意し (`hwnpu-dev models package MODEL/coreai`: 取得元から、なければ変換) その書き出しをコンパイルする
- オフラインのコンパイラが失敗した部品は失敗とし (出力は残さない)、ほかが終わってから単独でもう 1 度試す
- 1 回に 1 種類ずつ作る (`--arch ARCH`)。12 種類を一度に作ると、ディスクが埋まるまで何も置かれなかった。ディスクの空きと `$TMPDIR/com.apple.MetalPerformanceShadersGraphHost` を見る
- Chroma1-Flash の 1024²: 29 本を 9 分 (1 プロセスのピーク 2.3 GB)。ANE のコードは合計 140 MB、重みは 15.6 GB

### prebuilt が効くところ

初回、空の Core AI と aned のキャッシュ (署名の識別子を変えたコピー)、1024²:

| モデル、機械 | 端末 | prebuilt から | 絵 |
|---|---|---|---|
| FLUX.2 klein 4B、M1 16 GB | 40 分 08 秒 | 7 分 48 秒 | バイト単位で同じ |
| FLUX.2 klein 4B、M2 16 GB | 52 分 16 秒 | 16 分 55 秒 | 構図は同じ、細部が違う (PSNR 28.9 dB) |
| Chroma1-Flash、M2 16 GB | DiT のコンパイル約 46 分 | 読み込みと仕上げ全部で 1 分 07 秒、1 枚 7 分 24 秒 | バイト単位で同じ |
| Anima-Turbo、M2 16 GB | 45 分 01 秒 | 37 分 50 秒 (−16 %) | バイト単位で同じ |
| LFM3-1.7B、M2 | 2 分 17 秒 | 8 分 12 秒 | 答えも速さも同じ |
| plamo-3-translate-8b、M2 | 1 分 13 秒 | 7 分 08 秒 | 同じ |
| Qwen3.5-4B、M2 | 12 分 26 秒 | 7 分 07 秒で読めたが、1 時間 40 分答えなかった | — |

- 効くのは 1 本のコンパイルに数分かかる大きな部品 (画像の DiT: FLUX の `dit_single` が約 7 分 → 33 秒)。小さな部品では負ける: 仕上げは 1 本約 10 秒かかり (サービスが 1 本ごとに終わり、launchd が起動を遅らせる)、プロセスの最初の prebuilt に数分の決まった費用があるように見える
- そこで prebuilt は FLUX.2 klein と Anima (外した Chroma1-Flash にも作っていた。DiT など、文と VAE 以外の部品) に作り、チャットのモデルには作らない
- 1 ステップの時間はどちらも同じ

Core AI と `coreai-build` の未解決の問題、Apple への報告の番号、直ったときにすること: [既知の問題](../known-issues.md)。

## Core AI でモデルを確かめる

`tools/coreai` の道具:

| 道具 | 使い方 |
|---|---|
| `aicheck run -cpu PKG…` | 部品を Core AI で動かし、乱数の入力で `milcpu` と比べる。`-coreml` は同じ部品を Core ML で (誤りが Core AI のものか)。`-calls N` は 2 回目以降の平均。`-ref` は参照のパッケージと比べる (なければ `X.aimodel` は隣の `X.mlpackage` と比べる)。`AICHECK_WORST=1` は差の大きい要素を出す。`AICHECK_CONST=name=v,…` は定数やマスクの入力に本当の値を与える |
| `HWNPU_DEV=ailower.cut=N` | 先頭の N 個の演算だけを下ろし、その演算を `milcpu` とコサインで比べる。誤る演算、Neural Engine から外れる演算を二分探索する |
| `HWNPU_DEV=ailower.trace` | op_id (Core AI のエラーが示す) と演算の対応 |
| `aicheck probe` | 小さな MIL (SDPA、RoPE、RMSNorm、int8 の線形、書き換え) が Neural Engine に載るか |
| `aicheck lower OUT PKG` | 下ろし方のない最初の演算で止まる |
| `aicheck merge OUT.aimodel f8=PKG f128=PKG` | 複数のパッケージを 1 本の関数に |
| `aicheck cycle [-rounds N] PKG…` | ほかのプログラムの後に呼ぶときの費用 |
| `aicheck rules`、`limits`、`lint` | Neural Engine の数値の規則 (float32 と比べる)、上限 (呼び出しの下限、プログラムの数、形式)、書き出しの lint |
| `caiload` (Swift) | Core AI だけで specialize、実行、`-cycle` で切り替えの費用 |
| `lutfake`、`lutscan -tile 16`、`kleinonly`、`bcdump` | 合成の lut の層、隣り合う同じ表、FLUX.2 klein の準備と置き、バイトコードの読み出し |
| `llirtasks [-task N [-around K]] X.mlir.bc` | prebuilt の LLIR (`X.aimodelc/…/binary_0.llir.bundle/…/X.mlir.bc`) の Neural Engine のタスクを順に並べ、それぞれの計算と読むカーネルを出す。端末の仕上げのサービスが失敗したときのタスク (os_log の `Convert 1/1 procedures of ANEHLO task N/M`) が何かを引く。`bcdump` も LLIR を読む |
| `AICHECK_SENTINEL=PROGRAM aicheck run …` | 動くと分かっているプログラムを最初に動かし、失敗したプログラムの後はそれが動くまで待つ。Neural Engine をタイムアウトさせる prebuilt の後は、しばらく次のプログラムも失敗するので、その間は判定しない |
| `aicheck locked CMD…` | Neural Engine のロックを持ったまま CMD を動かす。CMD が起動する hwnpu のプロセスは同じロックの下で動く (`HWNPU_NPU_LOCK_FD`) ので、その間にほかのセッションが Neural Engine を取らない。CMD が残したプロセスもロックを持ち続けるので、補助のプロセスには記述子 3 を閉じさせる |
| `partprobe -dir CHECKPOINT -o OUT.aimodel NAME=start:layers:seq:paged[:key=value,…] …` | チャットのモデルの部品を、1 本のプログラムの関数として書く。関数ごとに設定 (1 回の行数、attention で分けるか、アーキテクチャのパラメータ) を変えられる。prebuilt の仕上げ 1 回で、どの部品が失敗するかを見分ける。NAME が 1 つなら書き出しの部品そのもの (同じ `main.hash`) になる |

`aicheck run` と `probe` はコマンドと同じく Neural Engine のロックを取る (`llm.LockNPU`): 2 つのプロセスが同時に使うと Mac が固まったことがあり、ほかのプロセスのコンパイルは時間を乱す。

新しいモデルで見ること ([Neural Engine の項目](apple-neural-engine.ja.md) と[チェックリスト](new-model-checklist.ja.md)に加えて):

- Core ML で通っても Core AI で、M1 でも確かめる: MoGe-2 の dense の頭と Sana の VAE は、M3 や Core ML では通り、M1 の Core AI のコンパイラで落ちた
- 下ろし方の誤りと Neural Engine の誤りを `HWNPU_COREAI_UNITS=cpu` で分ける
- 誤りの大きさだけでなく中身を見る (PP-OCR の誤りは割り算の代わりの掛け算で、チャンネルごとに直線を当てて分かった)。疑う演算だけの小さなプログラムは正しいことがある。本物の部品を切り詰めて確かめる
- 小さなプログラム (約 45 演算) は中身によらず CPU に行く。Neural Engine から外す原因は、載ると分かっている本物の大きさのプログラムに足して探す
- 大きさを 1 本の関数にするときは、切り詰める前に大きさごとの部品を単独で見る
- 起動の時間は両方のキャッシュを温めて比べる (それぞれ動かした直後に 3 回ずつ): 追い出された Core ML のキャッシュを Core AI の速さと読み違えたことがある
- 参照の精度を先に決める: `runner` を通す CPU のインタプリタは `HWNPU_DEV=cpu.f16=1` でなければ fp32。`aicheck` は fp16 の丸めで動かす

## Core AI とその下の Neural Engine の罠

Neural Engine に共通の数値 (fp16 の範囲、不正確な silu、int8 のあふれ…) は [Neural Engine](apple-neural-engine.ja.md) にある。Core AI の道で見つけたもの:

| 罠 | 回避 |
|---|---|
| 定数での `divide_v1` を Neural Engine が積として計算する | ailower が逆数との積で書く |
| Core AI の CPU は和が fp16 を超える `reduce_mean` であふれる (1/n の前に和が fp16 に戻る。平均 −1.37 の 57600 個で −inf)。Neural Engine と GPU は正しい | `HWNPU_COREAI_UNITS=cpu` を使うときに覚えておく |
| c が入力でも `(v + c) - c` を `v` にまとめる (魔法数の丸めが壊れる) | Neural Engine ではその丸めの形を使わない |
| 幅の軸の畳み込みで、幅 1 の核、512 以上のチャンネル、計算された入力だと誤る | 高さの軸で: x を (1, C, T, 1)、重みを (co, ci, k, 1) |
| 膨張つきの畳み込みが長さによって誤るか GPU に行く | フレームを膨張の行に並べ、膨張なしで畳み込む |
| 畳み込みの前の明示の `pad` が誤った | — |
| 内側の次元が 1 の matmul | 掛け算で書く |
| softplus が x > 10.4 で 0 | 安定な形 (上) |
| 最後の軸の slice の開始を 32 で割った余りが 1〜14: 4094 を超える値が Inf | `HWNPU_DEV=ailower.notes`、`aicheck lint` (当たるのは LFM2-8B-A1B のルーティングの重みだけで、値は 1 以下) |
| linear / conv の出力が、途中の和でも 2^15 に届くと Inf (M2) | 範囲を抑える (MaxScaled、`mil.BlockRanges`) |
| matmul の中の非正規化数は 0 | 範囲の点検 |
| 「empty reset layer candidates in SS cost model」: 単独なら通る演算の組で落ちる (Sana の VAE の `tile` + 1×1 の畳み込み、MoGe-2 の 1×1 の逆畳み込みの後の `pixel_shuffle`) | 値の同じ書き方に: チャンネルの繰り返しを 0/1 の重みの 1×1 の畳み込みで、シャッフルを行の帯に分けて (`mil.deconv2`、256 行まで) |
| 8 × 8 の行列どうしの積でプログラムが Neural Engine から外れる ((8 × 8)·(8 × 128) と k·kᵀ は外れない) | — |
| kv ヘッドごとの注意のプログラムは kv ヘッドが 1 つだと Neural Engine から外れる | そこは grouped のまま ([LLM のエンジン](llm-engine.ja.md)) |
| M1 のコンパイラは `sin`、要素ごとの演算にまとめた clip (「NEFUSED_EW: EW: MAX」)、定数倍した入力どうしの引き算を受け付けない | 避けて書く (MOSS の DAC の snake を級数で): [音声のモデル](audio-models.ja.md)、[既知の問題](../known-issues.md) |
| `coreai-build` は端末のコンパイラより厳しい | オフラインでも通る形で書く |
| 1 つの関数の失敗で資産の関数が全部 GPU に行く | 関数を単独で確かめる。ばらす |
| 1 プロセス 112 個を超えたプログラムは黙って GPU | プログラムを数える (`aneLive`) |
| Core AI 自身の変換の失敗でプロセスが落ちる | 先に `aicheck` で確かめる |

## ほかのファイルにある Core AI の選択

| 選択 | ファイル |
|---|---|
| 短い会話は分けない 1 本で、伸びたら注意で分けた形に (`wholeTiers`、`decode.Session.Migrate`。M1 ではなし: `chat.wholeSlower`)、Core AI では Qwen3 も注意で分ける、kv ヘッドごとの注意 (`paged.Heads`)、長い窓のリングの段 | [LLM のエンジン](llm-engine.ja.md) |
| DiT の組を呼び出しごとに入れ替える、先読み、まとめて描く、注意の並べ方、メモリに応じたブロック数 | [Apple の画像のモデル](image-apple.ja.md) |
| 部品の間でホストが専門家を選ぶ | [MoE](moe.ja.md) |
| 取得元の書き出し、`files.tar.zst`、prebuilt の束 | [配布](distribution.ja.md) |

## 採らなかったもの

- **Core AI のない Mac で Core ML に落とす**: コマンドは macOS 27 を求め、そう言う
- **Core ML を中間の形式にする** (読み込むときに `.mlpackage` を下ろす): 書き出しはチェックポイントから直接書く。ディスクに重みが 3 回 (書き出し、下ろした `.aimodel`、Core AI の項目) 置かれ、変換の時間もかかった
- **`.aimodelc` を丸ごと配る**: 99 % は `.aimodel` がすでに持つ重み。`.aimodel` と、種類ごとの重みのない詰めた prebuilt を配る
- **チャットのモデルの prebuilt**: 初回が遅くなり (LFM3-1.7B 2 分 17 秒 → 8 分 12 秒)、Qwen3.5-4B は答えなかった
- **文の符号器の prebuilt** (`LeftToDevice`) **と M1・M2 の VAE の prebuilt** (レシピ): 絵が誤る / `Request cancelled`、得もない
- **Sana の prebuilt**: 試したとき `coreai-build` が ANE に乗せたのは 27 本のうち 8 本で、それも端末のコンパイルより仕上げが遅かった。Qwen-Image も対象にしない
- **h18g と落ちる 9 種類**: 作らない (GPU だけの prebuilt は M6 を ANE から外す)
- **Core AI の項目ができたら `main.mlirb` を空にする** (`HWNPU_DEV=coreai.drop-bytecode`、2026-11-26 に削除): 効くのは `.mlpackage` を変換キャッシュに下ろした資産だけで、主な経路のダウンロードした書き出しには効かなかった (M1 の YuNet: 前者は空になり、後者はそのまま)。ダウンロードした書き出しに広げると、項目が消えたとき (OS の更新、ディスクの空き不足、実行ファイル名の変更) に落とし直しになる。ディスクを空けるのは `hwnpu` コマンドのキャッシュの整理の役目で、ライブラリは `main.mlirb` を常に残す。空の `main.mlirb` を Core AI が解析するとプロセスごと落ちる (LLVM: `cannot unwrap empty odiec_module_t`)
- **コンパイルを並べる**: 機械に ANE のコンパイラの列は 1 本しかない
- **呼び出しごとに新しい `NDArray`、呼び出し側のバッファの view** (`NDArray.RawView`、IOSurface): 10.0 / 10.7 / 8.3 ms に対し、使い回す配列は 7.2 ms (PLaMo 3 0.8B の part1、M1)。今は呼び出し側が Core AI の配列に書く
- **分けない 1 本と分けた形を 1 本の `.aimodel` に**: Qwen3-0.6B で 60 関数になり、読み込み 2 分 28 秒、一部が GPU に外れ、GPU の関数ごとに MPSGraph の作業が 0.6 GB でディスクが尽きた
- **分けない 1 本を 2 段 (256 と 1024) にして 1 本の `.aimodel` の関数に**: 1024 の方が半分の速さ (29.4 対 59〜60 トークン/秒)。段は 1 つ、キャッシュの長さごとに別の `.aimodel`
- **下書きが外れたら 1 トークンの関数に切り替える**: 1 回 8 トークンは 1 トークンより 4 % 高いだけ (LFM3-1.7B、M1)
- **gate と up を 1 つの線形に** (AMD のように): LFM3-1.7B の部品で 3.31 と 3.30 ms。部品は重みのバイト (約 51 GB/s) で決まり、Core AI は 2 つの線形の間に呼び出しの境目を作らない
- **hwnpu-audio の `--decoder-on gpu|cpu`**: 外した。Core AI はユニットを選べない
- **16 GB で DiT の組を細かく** (Chroma の `--group 2`): コンパイルは半分になるが、1 ステップが 60 % 遅い (wired メモリのワイヤし直し)

## 未解決

- Core AI の `.persistent` のキャッシュの方針 (`AIModelCache.specialize(…cachePolicy:)`、`~/Library/Application Support/coreai-cache/` に置く): 掃除によるコンパイルし直しはなくなるが、項目は hwnpu が自分で消すことになり (`deleteEntries(for:)`)、aned のコードは消されるまま。決めていない
- 同じ `.aimodel` が OS の版を越えて読めるか (キャッシュのフォルダは OS のビルドごと)
- プロセスの最初の prebuilt の仕上げにかかる数分、Anima の最初のステップの 13 分 (prebuilt によらない): 調べていない
- VAE が h13g と h14g の 512² で縞を描く理由 (M2 では端末のコンパイルでも)、T5 が cancelled になる理由。直ったらレシピの決まりと `LeftToDevice` を外す。取得元の古い FLUX.2 klein の prebuilt (今のコードが読まない書き出しのもの) はまだ VAE と文の部品を含む。v6 の書き出しの h13g のものは含まない
- `coreai-build` は 9 種類で落ち、h18g で ANE のコードを出さない (Apple に報告済み)。M6 が端末でコンパイルできるかはわからない
- aned のディスクの量は直接測っていない
- 入力の配列に直接書く形: M2 / M3 では測っていない。画像と音声のプログラムは今も写す
- macOS 26 でのエラーと Intel の Mac は実機で確かめていない
