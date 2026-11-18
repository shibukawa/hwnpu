# 引き継ぎ: Neural Engine で MoE を動かすための、gather と実行時の重みの調査

作成: 2026-11-18。M1 (16 GB、macOS 27、ビルド 26B131) で調べた。ブランチ `t3/ane-gather-probe` (PR は出していない)。

目的: Apple 向けの MoE (gpt-oss-20b、Qwen3.6-35B-A3B など) が無いのは、専門家の選択 (gather) が Neural Engine に載らないため ([design.md](design.md) 16.179 の gpt-oss の節、[model-fit.md](model-fit.md) 4 節)。この判断は Core ML 時代のもので、Core AI では試していなかった。ANE のコンパイラの中身を調べ、流用できるものを探した。

## 1. 分かったこと (M1)

### ANE のコンパイラには gather の部品がある

dyld の共有キャッシュ (`/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld/dyld_shared_cache_arm64e*`) の文字列から:

- ANE 用の MLIR 方言 `anec` の演算: `anec.gather_nd`、`anec.dynamic_input_view`、`anec.input_view`、`anec.top_k`、`anec.sort`、`anec.arg_min_max`、`anec.param_linear`、`anec.param_convolution`、`anec.if`、`anec.state`、`anec.ring_buffer_reader` / `writer`、`anec.mx_dequant` / `mx_quant` ほか
- 内部の部品: `ZinIrGatherUnit`、`ZinIrDynamicSliceUnit`、`ZinIrDynamicGOCUnit`。検証関数 `ANECValidateGatherLayer`、`ANECValidateDynamicSliceLayer`
- gather の条件の文言: 「Dynamic axes in gather is not supported」「Data tensor of gather cannot have multi user mode」「Index/Data tensor dimension %s should be equal to 1」「Gather axes contain isolation axis」「Invalid interleave factor for gather layer」
- 実行時の位置での状態への書き込み: 「With dynamic offset, slice_update's begin must be produced by a concat」「Cannot have multiple slice_update per write_state」(KV キャッシュの書き込みに使えるかもしれない。未調査)
- ANE のコンパイラが受け取る MPS の演算: `mps.Gather`、`mps.GatherAlongAxis`、`mps.GatherND`、`mps.TopK`、`mps.Sort`、`mps.ReductionArgMax`、`mps.Slice`、`mps.StridedSlice`、`mps.Select`、`scf.If` ほか

Core AI の基本演算 (`aicode.primitive.*`) にもある。引数の数は C++ の型名 (`NOperands<N>`) から:

| 演算 | 引数 |
|---|---|
| `gather_nd_v1` | 2 (data、indices) |
| `gather_along_axis_v1` | 3 (input、indices、axis。axis は定数) |
| `slice_v1` | 4 (x、begin、end、stride。stride は定数、begin は実行時でもよい) |
| `slice_update_v1` | 5 (input、update、begin、end、stride) |
| `argsort_v1` | 4 (input、dim、descending、stable。後ろ 3 つは定数) |
| `reduce_index_v1` | 4 と region 1 つ |
| `batch_matmul_v1` | 2 |

流れ: Core AI の `aicode.primitive.*` → (CPU) BNNS / (ANE、GPU) MPS 方言 → MPSGraph が ANE の領域を切り出し、ANE のコンパイラに検証させる。

### M1 で載ったもの、載らなかったもの

| 演算 | Core AI の配置 | Core ML の MLComputePlan の `supportedComputeDevices` に ANE があるか |
|---|---|---|
| `gather_nd` (表が定数でも入力でも) | GPU | ない |
| `gather_along_axis` | GPU | ない |
| begin が実行時の `slice_by_index` | GPU | ない |
| `reduce_argmax` | (ailower 未対応) | ない |
| 重みが入力の `matmul` (fp16) | **全部 ANE** | ある |
| int8 の入力を `cast` で fp16 にし、拡大して `matmul` | **全部 ANE** | (Core ML は仕様 10 以上が要り、試していない) |

Core AI のログ: 「setNDXANEProcedureAttrs: entry '…' has 0 ANE regions」。MPSGraph が gather の部分を ANE に回していない。理由はログで `<private>` (3 章)。

### 実行時の重みの速さ (M1)

8 行 × 2880 × 2880 (gpt-oss の幅)、200 回の平均。入力はプログラム自身の配列 (`INPLACE=1`、呼ぶたびのコピーなし)。1 回の呼び出しの固定費は約 0.2 ms ([design.md](design.md) 16.218)。

| 重み | 時間 | CPU (milcpu) との照合 |
|---|---|---|
| 定数 fp16 (`lin_const_big`) | 0.52 ms | cos 1.000000 |
| 定数 int8 行ごとの拡大 (`lin_i8_big`) | 0.35 ms | cos 1.000000 |
| 入力 fp16 (`mm_dyn_big`) | 0.61 ms | cos 1.000000 |
| 入力 int8 + cast + 拡大 (`mm_dyn_i8_big`) | 0.54 ms | **未確認** (milcpu が int8 の入力を作れない) |

呼ぶたびに入力をコピーすると (`INPLACE` なし)、fp16 1.16 ms、int8 0.72 ms。M1 のコピーは約 29 GB/s。

### MoE への当てはめ (見込み、未実装)

- ルーターは ANE、専門家の選択 (top-k) はホスト。層ごとにルーターのロジットをホストに戻す
- 専門家の重みは専門家ごとの Core AI の入力配列に int8 で置き、選んだものを層の形ごとに 1 つのプログラムに結び付ける。専門家ごとのプログラムは要らない (ANE のプログラムは機械全体で約 128 まで)
- 往復は層ごとに 1 回増える。gpt-oss の 24 層で 1 トークン 10 ms ほどの見積もり
- gpt-oss-20b の専門家は int8 で約 19 GB。16 GB の機械には載らない。32 GB 以上か、小さい MoE (LFM2-8B-A1B など) が対象

## 2. コード (このブランチ)

| ファイル | 変更 |
|---|---|
| `internal/coreai/ailower/ops.go` | `gather_nd`、`gather_along_axis` を Core AI の基本演算に。begin が定数でない `slice_by_index` を `slice_v1` に |
| `internal/coreai/ailower/ailower.go` | `isConst`。int8 の入力 (`sint8`、短い名前 `i8`) |
| `internal/backend/coreai/coreai_darwin.go` | int8 の入出力 (`types.I8`) |
| `internal/backend/coreml/plan_darwin.go` | MLComputePlan の `supportedComputeDevices` に ANE があれば、装置の名前に「(the Neural Engine supports it)」と付ける (実験用) |
| `internal/imagegen/milpart/milpart.go` | `InputI32`、`InputI8` |
| `tools/coreai/aicheck/probe_darwin.go` | probe の追加 (下)。`COREML=1` で Core ML、出力に理由と演算ごとの装置 |
| `tools/coreai/aicheck/run_darwin.go` | `INPLACE=1` でプログラム自身の入力配列 (`backend.InputAllocator`) を使う |

probe: `gnd_const`、`gnd_input`、`gnd_mm`、`gaa_input`、`gaa_const`、`dslice_const`、`dslice_input`、`dslice_mm`、`mm_dyn`、`mm_dyn_big`、`mm_dyn_i8`、`mm_dyn_i8_big`、`lin_const_big`、`lin_i8_big`、`argmax`。

実験用の変更で、main に入れる形にはしていない (`plan_darwin.go` の文言、`run` の int8 の入力の値はでたらめ)。

## 3. 確かめること

測る前に、ほかの hwnpu のプロセスが Neural Engine を使っていないことを確かめる (`ps -ax | grep -E "hwnpu|aicheck"`)。aicheck は NPU のロックを取るので順番待ちになるが、時間はずれる。

### 3.1 ほかのチップでの配置 (M2、M3、M4)

```sh
git fetch origin && git switch t3/ane-gather-probe
go build -o /tmp/aicheck ./tools/coreai/aicheck
sysctl -n machdep.cpu.brand_string
rm -rf ~/Library/Caches/coreai-cache/*/aicheck     # aicheck の probe のキャッシュだけ
/tmp/aicheck probe linear gnd_const gnd_input gnd_mm gaa_input gaa_const dslice_const dslice_input dslice_mm mm_dyn mm_dyn_big mm_dyn_i8 mm_dyn_i8_big
COREML=1 /tmp/aicheck probe linear gnd_const gnd_input gnd_mm gaa_input gaa_const dslice_const dslice_input argmax
```

`ane=true` なら全部 Neural Engine。Core ML の行は、演算ごとに優先される装置と、ANE が対応するか (「(the Neural Engine supports it)」) を出す。**新しい世代で gather_nd か dslice が ANE に載れば、MoE の形はずっと簡単になる** (表は定数のまま、専門家の番号だけ入力)。

### 3.2 実行時の重みの速さ

```sh
KEEP=1 /tmp/aicheck probe mm_dyn_big mm_dyn_i8_big lin_const_big lin_i8_big   # 最後の行に packages in DIR
INPLACE=1 /tmp/aicheck run -calls 200 DIR/lin_const_big.mlpackage DIR/lin_i8_big.mlpackage DIR/mm_dyn_big.mlpackage DIR/mm_dyn_i8_big.mlpackage
INPLACE=1 /tmp/aicheck run -calls 50 -cpu DIR/mm_dyn_big.mlpackage
```

`call A/B` の B が 1 回の平均。

### 3.3 ANE のコンパイラが gather を外す理由 (sudo)

ログの理由は `<private>` で伏せられる。macOS 27 では `sudo log config --mode "private_data:on"` は「Invalid Modes 'private_data:on'」で効かない。構成プロファイル [tools/coreai/logging/private-data.mobileconfig](../tools/coreai/logging/private-data.mobileconfig) (`com.apple.system.logging` の `System` → `Enable-Private-Data`) を入れる:

```sh
open tools/coreai/logging/private-data.mobileconfig
# システム設定 → 一般 → デバイス管理 (または プライバシーとセキュリティ → プロファイル) で
# 「Unified log private data (hwnpu)」をインストール (管理者のパスワード)
/usr/bin/log show --last 1m --info | grep -c "<private>"    # 減ったか確かめる
```

`log` は zsh の組み込みに化けるので `/usr/bin/log` と書く。

```sh
rm -rf ~/Library/Caches/coreai-cache/*/aicheck
/tmp/aicheck probe gnd_const gaa_const dslice_const mm_dyn
/usr/bin/log show --last 3m --info --debug --style compact \
  --predicate 'process == "aicheck" OR process == "ANECompilerService"' > /tmp/anec_private.log
grep -i -E "gather|slice|reject|unsupported|not supported|validation|ANE region|placement" /tmp/anec_private.log | head -100
```

見たいもの: `com.apple.anec:validation` の Start と End の間の行、`ANECValidateGatherLayer rejected …` の status、MPSGraph の配置の判断。終わったら戻す:

システム設定の同じ画面でプロファイルを削除する。

### 3.4 次の段 (実装)

- int8 の入力の正しさ: milcpu に int8 の入力を足すか、`run` で int8 の値を作って比べる
- 小さい MoE の 1 層 (ルーター、ホストの top-k、選んだ専門家の配列の結び付け) を組んで、1 トークンの時間を測る
- 配列を数百個持ったときの挙動 (IOSurface の数、常駐するメモリ)
- `anec.mx_dequant` (MXFP4) が新しい世代で使えれば、gpt-oss の専門家を 4 bit のまま入力にできるかもしれない

## 4. 注意

- 幅の広い one-hot (128 × 16000 など) は M1 の Neural Engine を全プロセスで止めたことがある。probe の軸は 16384 以下に保つ
- 調べた文字列のダンプは 1.5 GB になる (`strings -a` を共有キャッシュの全部に)。ディスクの空きに注意
- `MPSGRAPH_PRINT_ANE_PLACEMENT_ANALYSIS`、`MPSGRAPH_DUMP_MODULE`、`MPSGRAPH_FORCE_DEVICE_PLACEMENT` などの環境変数はあるが、M1 では何も出なかった (社内向けの設定が要るのかもしれない)
