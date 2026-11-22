# アーキテクチャ

[English](architecture.md)

hwnpu が乗るベンダーのスタック、コードの分け方、cgo を使わないための規約、テストのしかた、モデルの
変換のしかた。公開 API は [api.ja.md](api.ja.md)、各バックエンドはそれぞれのファイル
([Neural Engine](apple-neural-engine.ja.md)、[Core AI](apple-coreai.ja.md)、[Core ML](apple-coreml.ja.md)、[Intel](intel.ja.md)、[AMD](amd.ja.md))。

## ベンダーのソフトウェアスタック

| ベンダー | OS | hwnpu の経路 | バックエンド名 |
|---|---|---|---|
| Intel NPU (Core Ultra、Meteor Lake 以降) | Linux、Windows | Level Zero のローダ (`libze_loader.so.1` / `ze_loader.dll`) + NPU のグラフ拡張を purego で | `levelzero` |
| Intel (NPU なしでの確認) | どれでも | OpenVINO IR を OpenVINO 自身で (CPU プラグイン、または `HWNPU_DEV=ov.device=NPU`) | `openvino` (`HWNPU_BACKEND=openvino` のときだけ) |
| AMD XDNA (Ryzen AI、Phoenix 以降) | Linux | カーネルドライバ `amdxdna` の ioctl と mmap を `golang.org/x/sys/unix` で (purego すら不要) | `xdna` |
| AMD XDNA | Windows (試験は Linux) | XRT (`xrt_coreutil.dll`) を、実行時に読む小さな C のライブラリ `native/xrtshim` 経由で | `xrt` (Linux では `HWNPU_BACKEND=xrt` のときだけ) |
| Apple Neural Engine | macOS | Core ML を `purego/objc` で | `coreml` (既定) |
| Apple Neural Engine | macOS 27 | Core AI を Swift の shim (`libcaishim.dylib`、埋め込み、purego で読む) で | `coreai` (`WithBackend("coreai")`、`HWNPU_BACKEND=coreai`) |

### Intel

```
アプリ (Go, purego)
  └─ libze_loader.so.1 / ze_loader.dll            Level Zero ローダ (MIT)
       └─ libze_intel_npu.so.1 / (Windows の DLL) UMD、intel/linux-npu-driver (MIT)
            ├─ libnpu_driver_compiler.so          ドライバ内コンパイラ: OpenVINO IR → NPU blob
            └─ /dev/accel/accelN                  KMD intel_vpu (drivers/accel/ivpu、GPL)
```

- NPU 拡張 (`intel/level-zero-npu-extensions`) は全て C ABI で、`ze_graph_ext.h` の版付き DDI テーブルから
  使う。拡張名は古いドライバでは版付き (`ZE_extension_graph_1_2`)、新しいドライバでは版なし
  (`ZE_extension_graph`)。どちらも受ける。
- ファームウェアへのジョブの形式は UMD の内部で、`ivpu_accel.h` の UAPI に無い。ioctl 直叩きは現実的で
  なく、Level Zero が唯一の筋のよい道。
- `zeGraphCreate` はコンパイル済み blob (`ZE_GRAPH_FORMAT_NATIVE`) と OpenVINO IR
  (`ZE_GRAPH_FORMAT_NGRAPH_LITE`、ドライバ内でコンパイル) の両方を受ける。IR の入出力の記述は OpenVINO の
  `serializeIOInfo` の移植。`--config` は `GraphSource.BuildFlags` で渡す。
- ドライバ、コンパイラ、ファームウェアの版が揃わないと "MAPPED_INFERENCE_VERSION is NOT compatible with
  the ELF" で失敗する。Linux では `/dev/accel/accel0` に `render` グループが要る。

### AMD

```
アプリ (Go)
  └─ golang.org/x/sys/unix で ioctl / mmap
       └─ /dev/accel/accelN    KMD amdxdna (drivers/accel/amdxdna、Linux 6.14 で upstream)

参考: 公式の経路  C++ アプリ ─ libxrt_coreutil.so.2 + xrt_plugin.*-amdxdna (shim、Apache-2.0)
```

- UAPI (`amdxdna_accel.h`) は小さく安定している: `CREATE_HWCTX / DESTROY_HWCTX / CONFIG_HWCTX /
  CREATE_BO / GET_BO_INFO / SYNC_BO / EXEC_CMD / WAIT_CMD / GET_INFO / SET_STATE / GET_ARRAY`。
- 仕事はハードウェアコンテキスト (列の区画) に、CMD BO の `ert_start_kernel_cmd` パケットとして投げる。
  プログラムは xclbin (AXLF。`AIE_PARTITION` に PDI = overlay) と制御コード (IRON の `insts.bin`)、
  または full ELF (AIE2P、Strix Point 以降)。
- XRT の C API は purego から使えない: ハードウェアコンテキストとモジュールは C++ 専用で、残る C の呼び出しは
  可変長引数。Linux はドライバを直接叩き、XRT の shim (Apache-2.0) を移植した。KMD のインターフェースが
  非公開の Windows は、xrtshim から XRT の C++ ライブラリを使う。詳しくは [amd.ja.md](amd.ja.md)。

### Apple

- ANE 自身のフレームワーク (`AppleNeuralEngine.framework`、`_ANEClient`) は非公開。公開の経路は Core ML
  (と macOS 27 の Core AI)。
- `MLComputeUnitsCPUAndNeuralEngine` (macOS 13 以降) で GPU を除ける。「Neural Engine だけ」は指定できない。
- `MLComputePlan` (macOS 14.4 以降) で、演算ごとの推奨デバイスと対応デバイスを実行前に取れる。厳格モードに
  使う。推定であり、実行時の配置の保証ではない。
- Core AI は配置を自分のコンパイルキャッシュから知らせる (MPSGraph の delegate のマニフェスト、
  `mps.fullyPlacedOnANE`)。[apple-coreai.ja.md](apple-coreai.ja.md) を参照。

### Qualcomm (未着手)

Snapdragon X の Hexagon NPU は QNN SDK (`QnnHtp.dll` / `libQnnHtp.so`、`QnnInterface_getProviders`) の
C API で purego 向きで、`GraphRunner` に収まる。今は範囲外。

## 設計方針

### 共通にするもの、しないもの

| レイヤ | 共通化 | 理由 |
|---|---|---|
| デバイスの列挙とオープン | する | sysfs / ローダ / DLL の違いを吸収するのは安い |
| デバイスから見えるメモリ (`Buffer`) | する | 「ゼロコピーの `[]byte` ビュー + 明示的な Sync」はどのバックエンドにも合う |
| 完了待ち (`Fence`) | する | `context.Context` で取り消せる `Wait` |
| 仕事の投入 | しない (能力別) | グラフの推論 (`GraphRunner`: Intel、Apple) とカーネルの実行 (`KernelRunner`: AMD) は引数も失敗のしかたも違う |
| モデルの形式 | しない | OpenVINO IR、xclbin / ELF、mlpackage / aimodel は互いに変換できない |

Intel はグラフ実行機 (ネットワーク全体をドライバがコンパイル、ユーザーカーネルの公開 API なし)、AMD は
カーネル実行機 (IRON / mlir-aie / Peano のカーネルを AIE タイル配列に流す、公開のドライバ内コンパイラなし)。
API はこの違いを隠さない。

層: ドライバ → hwnpu の中核 (モデルとカーネルの実行) → 用途別のヘルパ (チャット、埋め込み、画像、音声、
`internal/` の中)。

### バックエンドの選定の理由

- **Intel = Level Zero (purego)。** 唯一の公開経路。C ABI で構造体は全てポインタ渡し。
- **AMD Linux = ioctl 直叩き。** 依存なし、純粋な Go でビルド。代償は shim の移植とコマンド形式への追従。
- **AMD Windows = xrtshim 経由の XRT。** KMD のインターフェースは非公開で、公開の入口は C++ の
  `xrt_coreutil.dll` だけ。
- **Apple = Core ML (purego/objc)、macOS 27 では Core AI。** GPU を除き、配置を先に確かめる。開発機が
  Mac なので API の形はそこで決める。

## パッケージ構成

モジュールは `github.com/shibukawa/hwnpu`。

| パス | 内容 |
|---|---|
| `hwnpu.go`、`device.go`、`graph.go`、`kernel.go`、`options.go`、`float16.go` | 公開 API ([api.ja.md](api.ja.md)) |
| `backends.go`、`backends_darwin.go` | 組み込むバックエンドの登録 (Level Zero、xrt、xdna、OpenVINO はどこでも。Core ML、Core AI は darwin) |
| `models.go` | `ModelsYAML`: 埋め込んだ `models.yaml` |
| `modelstore/`、`npuinfo/` | 公開パッケージ ([api.ja.md](api.ja.md)) |
| `internal/types` | 公開 API とバックエンドが共有する値の型 |
| `internal/backend` | バックエンドのインターフェースと登録。`levelzero`、`openvino`、`amd` (amdxdna)、`amdxrt` (XRT)、`coreml`、`coreai` |
| `internal/dl`、`internal/ze` | purego / LoadLibrary の薄いラッパ。Level Zero と NPU 拡張のバインディング |
| `internal/xdna`、`internal/xclbin`、`internal/aieelf`、`internal/ert`、`internal/accelfs` | amdxdna の UAPI (`xdnasim`: 偽のドライバ)、AXLF の読み、full ELF の読み、ERT パケット、`/dev/accel` の走査 |
| `internal/mlpackage`、`internal/milspec`、`internal/mlspec` | `.mlpackage` の書き出し (MIL の protobuf、MILBlob の `weight.bin`)。coremltools の .proto からの生成物 |
| `internal/coreai/ailower`、`internal/coreai/mlirbc` | MIL → Core AI のプログラム。MLIR bytecode |
| `internal/ovlower`、`internal/ovir`、`internal/ovcpu` | MIL → OpenVINO IR。IR の読み。OpenVINO の CPU プラグインで IR を実行 |
| `internal/milcpu`、`internal/onnxrt`、`internal/sgemm`、`internal/f32`、`internal/mxfp4`、`internal/vecops` | CPU の実行系と内側のループ |
| `internal/safetensors`、`internal/gguf`、`internal/torchzip`、`internal/onnx`、`internal/tflite` | チェックポイントの読み |
| `internal/palette` | LUT の重みの当てはめ |
| `internal/llm` (+ `blocks`、`archs`、アーキテクチャごとのパッケージ、`decode`、`chain`、`runner`、`paged`、`experts`、`iron`、`check`) | 言語モデルなどの変換器と実行器 |
| `internal/chat`、`internal/imagegen/*`、`internal/speech`、`internal/transcribe`、`internal/vision`、`internal/audio` … | コマンドの裏の用途別のエンジン |
| `internal/cache`、`internal/store`、`internal/power`、`internal/mem`、`internal/sysload` | コンパイルキャッシュとモデルの置き場、エネルギーのカウンタ、ページ境界のメモリと NPU のメモリ、負荷 |
| `cmd/hwnpu`、`cmd/hwnpu-chat`、`cmd/hwnpu-image`、`cmd/hwnpu-audio`、`cmd/hwnpu-dev` | コマンド ([cli.ja.md](cli.ja.md)) |
| `native/xrtshim`、`internal/backend/coreai/shim` | 実行時に読む shim (XRT 用の C、Core AI 用の Swift) |
| `tools/` | Python の参照 (`tools/ref`)、IRON の書き出し (`tools/amd`)、Core ML / Core AI / OpenVINO の補助 |

ルートのパッケージの依存: Linux と Windows では `github.com/ebitengine/purego` と `golang.org/x/sys`。
darwin ではさらに `google.golang.org/protobuf`、`klauspost/compress` (zstd)、`golang.org/x/text`。
`npuinfo` は標準ライブラリだけ。コマンドはもっと使う (kong、go-yaml、oto …)。

## purego / ABI 規約

```go
package dl
func Open(candidates ...string) (*Library, error)      // 順に試し、全て失敗したら試した一覧つきの ErrLibrary
func (l *Library) Bind(fptr any, symbol string) error  // RegisterLibFunc の panic を error に
func (l *Library) Sym(symbol string) (uintptr, error)
func BindAddr(fptr any, addr uintptr)                  // DDI テーブルの関数ポインタ用
```

- C 側に Go のメモリを持たせない。デバイスから見えるメモリはドライバが確保する (`zeMemAllocHost`、BO の
  mmap) か、Go のヒープの外 (`internal/mem`)。
- 構造体は値渡ししない。Go の struct は C のレイアウトを手で再現し、パディングも書く。
- コールバックを使わない: Level Zero のイベントはポーリング、amdxdna は ioctl (かコンテキストの syncobj) で待つ。
- 文字列は NUL を付けた `[]byte` で渡し、呼び出しが終わるまで `runtime.KeepAlive`。purego の自動コピーに頼らない。
- 浮動小数点の引数は使わない。型付きの `RegisterFunc` だけで、`purego.SyscallN` は使わない。
- 64 ビットだけ (`amd64 || arm64`)。
- Linux: `purego.Dlopen(name, RTLD_NOW|RTLD_LOCAL)`。Windows: `windows.LoadLibraryEx(…,
  LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)` + `GetProcAddress`。
- Linux で `CGO_ENABLED=0` のとき、purego の fakecgo が `dlopen` を動かす。完全な静的リンクでは動かない。
- purego を使わないとき (darwin の `npuinfo`): `golang.org/x/sys/unix` と同じく、`//go:cgo_import_dynamic` と
  アセンブリのトランポリンで libSystem の `dlopen`/`dlsym`/`dlerror` だけを取り込み、残りは実行時に引く。
  非公開のライブラリが無くてもバイナリは起動し、エラーになるだけ。

## テストと診断

| 種別 | 内容 | ハードウェア |
|---|---|---|
| ABI golden | `unsafe.Sizeof` / `Offsetof` を、C ヘッダから一度生成した値と比べる (`testdata/abi/levelzero_abi.c`、`testdata/abi/xdna_abi.c`: `ze_*`、`amdxdna_*`、`axlf`、`ert_*`) | 不要 |
| ioctl 番号 | `iowr(…)` を既知の定数と比べる (例 `DRM_IOCTL_AMDXDNA_CREATE_BO = 0xC0206443`) | 不要 |
| xclbin / ELF の読み | mlir-aie のサンプルを fixture に | 不要 |
| 偽のドライバ | `testdata/zemock` (偽の Level Zero、`cc` があればビルド)、`internal/xdna/xdnasim` (Go で書いた偽の amdxdna) | 不要 |
| fixture | `internal/fixtures` の小さなモデル (`gen_models.py` で作り直す) | 不要 |
| 変換器 | 合成の重み (`llm.NewSynthetic`)、プログラムと `weight.bin` のダイジェスト、`modeling_*.py` から書いた fp64 の forward、`milcpu` と PyTorch の照合 | 不要 |
| 実機 | Mac での Core ML / Core AI の試験。各 NPU での `hwnpu-dev report` (passthrough の設計、サンプルのモデル)。書き出し済みのプログラムは `HWNPU_TEST_PREBUILT` | 要 |
| 電力 | 同じモデルを `PowerLow` と `PowerPerformance` で、`hwnpu-dev powerbench` (mJ/トークン) | 要 |

試験は全部 `CGO_ENABLED=0 go test ./...` で動く。

診断: `hwnpu-dev info [-model M]` (デバイス、版、ライブラリの解決先、モデルの配置)、`hwnpu status` と
`hwnpu issue` (機械、`npuinfo` による NPU の状態、プロセス、キャッシュ。`--wake` なしでは NPU を起こさない)、
`go run ./npuinfo/cmd/npuinfo` (生の JSON)。[cli.ja.md](cli.ja.md) を参照。

## Go での変換の流れ

- モデルは Hugging Face の公式チェックポイント (bf16、`config.json` + safetensors か GGUF) から各 NPU の
  形式へ直接変換する。間に ONNX を置かない。量子化のしかたが実行先ごとに違うので、書き出しは実行先ごとに
  作る。量子化済みの公開物は劣化なしに並べ替えられるときだけ使い、量子化済みのものから量子化し直さない
  ([量子化](quantization.ja.md))。
- 変換は一度だけ: 書き出し済みを配るか、コマンドが初回に変換してキャッシュに置く
  ([配布](distribution.ja.md))。利用者はダウンロードし、実行時のコンパイル (Core ML の mlmodelc、Intel の
  ドライバ、Core AI の特殊化) だけを行う。
- 書き出し: `internal/mlpackage` (Core ML の MIL)、`internal/coreai/ailower` (同じ MIL を Core AI へ)、
  `internal/ovlower` (同じ MIL を Intel 向けの OpenVINO IR へ)。AMD のプログラムは IRON で作る
  (`tools/amd`、`internal/llm/iron`。[amd-llm.ja.md](amd-llm.ja.md))。
- 読み: `internal/safetensors`、`internal/gguf` (Q4_0、Q4_1、Q8_0、Q4_K、Q5_K、Q6_K を逆量子化)、
  `internal/torchzip`、`internal/onnx`、`internal/tflite`。

### コマンド

| コマンド | すること |
|---|---|
| `hwnpu-dev convert` | モデルを組み立てる: アーキテクチャは `config.json` から判別 (`-arch` で指定)、`-backend coreml` (`.mlpackage`) か `intel` (OpenVINO IR + `out.parts.json`)、`-target` (大きさの上限で分割を決めるチップ。`none` は分けない)、`-layers N` (最初の N 層だけ)、`-archs` (ホワイトリストの一覧)。アーキテクチャ固有のパラメータはフラグになる (LFM3: `-post-scale`、`-conv`、`-attn`、`-hidden-out`) |
| `hwnpu-dev check` | 参照と照合する: 埋め込み、各層単独 (参照の入力を与える)、最終 norm と出力層、モデル全体。層の相対 RMS 誤差が `-tol` (既定 0.01) を超えると終了コード 1 で最初の層を示す。`-package` は既存の `.mlpackage` のロジットを照合 |
| `tools/ref/ref_logits.py` | PyTorch fp32 の隠れ状態とロジットを書く (多くの実装で `output_hidden_states` の最後は最終 norm の後なので、デコーダ層への hook で取る。`hidden[0]` は `get_input_embeddings()` への hook)、`REF.json`、エンコーダは `--encoder` |

### NPU なしの確認

- `internal/milcpu` は MIL のプログラムを CPU で fp32 のまま実行する (linear、matmul、要素ごとの演算、
  reduce、softmax、reshape/transpose/slice/concat/pad、1 次元の conv、`gelu`、`layer_norm`、LUT と
  blockwise の constexpr …)。目的は「グラフが PyTorch と同じ計算をしているか」。`RoundF16` は演算ごとの
  結果を fp16 に丸めて NE の保存精度を近似する (NE の演算精度ではない)。速さは二の次 (行列積だけ並列)。
- 同じ関数の別のグラフ (taps と conv、post-scale の有無、層ごととまとめた実行) は一致しなければならない。
  わざと壊した層を照合が見つけることも試験で確かめる。
- NE 自身の数値の特性は、Mac で NE と Core ML の CPU を演算ごとに比べて確かめる
  ([apple-neural-engine.ja.md](apple-neural-engine.ja.md))。

## モデルのホワイトリスト

llama.cpp と同じく、対応アーキテクチャはホワイトリストで持つ。どれも Hugging Face の `modeling_*.py` の
forward を手で移したもので、`nn.Module` 1 つを Go のメソッド 1 つにする。理由: NPU 向けの書き換え
(post-scale、畳み込みのタップ分解、gather を避ける) を入れやすく、汎用のものを信用しなくて済む。手順と合格条件は
`.claude/skills/port-architecture` と [新しいモデルのチェックリスト](new-model-checklist.ja.md)。

- `internal/llm`: アーキテクチャに依らない部分。`Weights` (safetensors / GGUF を HF のテンソル名で)、
  `Options` (共通の設定。`NoHead` は最終 norm の前の隠れ状態を出す、`Start` を層の数にすると最終 norm と
  出力層だけ)、`Arch` (名前、`model_type` / `architectures`、GGUF の名前の対応、`Params`、Intel の形式 …)、
  `Register` / `Detect` / `LoadConfig`。
- `Model.Tensors()` は組み立てが読むテンソルの一覧。`llm.NewSynthetic` はそこから決定的な乱数の重みを作り、
  `llm.Inventory` はチェックポイントと突き合わせて、足りない、shape の違う、読まれないテンソルを出す
  (読まれないテンソルは移し漏れの手がかり)。
- `Model.Output()` (幅とロジットかどうか)、`Model.Embedding()`、`Build` が返す入力の一覧: エンコーダの
  ための口。`Start == 0` のプログラムは埋め込み段も含む。
- `internal/llm/blocks`: 共通の層 (重み形式を選べる linear、`LinearBias`、`LinearPadded`、RMSNorm、
  `LayerNorm`、SwiGLU、RoPE、マスク、`SDPA` / `Attention` / `SDPAGrouped`、`SiLU`、`Activation`)。
  アーキテクチャは `blocks.Graph` を埋め込み、固有の層だけを書く。
- `internal/llm/archs`: その import の一覧がホワイトリスト。

| パッケージ | Hugging Face のアーキテクチャ |
|---|---|
| `lfm3` | LFM3 (`lfm3`、`lfm3_vl`、`lfm3_omni`、`lfm3_audio` のテキストモデル) |
| `lfm3vl`、`lfm3audio`、`lfm3gen` | LFM3-VL の画像エンコーダ、LFM3-Audio のエンコーダ、LFM3-Audio / Omni の音声出力 |
| `lfm2moe` | LFM2 MoE |
| `gemma4` | Gemma 4 のテキストモデル |
| `qwen3`、`qwen35` | Qwen3 の密なモデル。Qwen3.5 のテキストモデル |
| `gptoss` | gpt-oss |
| `plamo`、`plamo3` | PLaMo の双方向エンコーダ (PLaMo-Embedding)。PLaMo 3 |
| `bert` | BERT 系エンコーダ (all-MiniLM-L6-v2 など) |
| `whisper` | Whisper |

モデルの一族は [llm-models.ja.md](llm-models.ja.md)、[llm-qwen.ja.md](llm-qwen.ja.md)、
[moe.ja.md](moe.ja.md)、[audio-models.ja.md](audio-models.ja.md)。画像と視覚のモデルは別の変換器を持つ
([image-models.ja.md](image-models.ja.md)、[vision-models.ja.md](vision-models.ja.md))。

## 残るマイルストーン

デバイスの列挙、厳格モードつきの Core ML、Intel (native blob と IR)、電力ポリシーとコンパイルキャッシュ、
Linux の AMD (xclbin と full ELF)、全 NPU での小さな LLM は済んでいる。残り:

| 項目 | 状態 |
|---|---|
| Windows、Intel | 対応。ドライバが `ze_loader.dll` をどこに置くかは未確認 |
| Windows、AMD | `xrt` バックエンドは Linux で XRT 2.21 に対して確認済み。Windows ではまだ動かしていない |
| AMD のユーザーモードキュー (UMQ) | 未着手 |
| Qualcomm QNN | 未着手 |
| 仮想化デバイス (VM、クラウド) | 未着手 (未解決を参照) |

## 未解決

- **Intel の電力。** エネルギーは RAPL から (`ScopePackage`、CPU パッケージ全体、たいてい root が要る)。
  メトリックストリーマーで NPU だけの値が取れるかは確かめていない。
- **仮想化。** クライアントの NPU は SoC ごとに 1 つの PCIe デバイスなので、今は VM やクラウドから使えない。
  AMD の shim に `virtio` のディレクトリがあり、UAPI に PF のデバイス種別があるので、準パススルーが準備中の
  可能性がある (推測)。`Device` の下に新しいバックエンドを足すだけで済むよう、バックエンドの境界を保つ。
- **glibc 以外での purego fakecgo。** musl / Alpine で `CGO_ENABLED=0` のまま `dlopen` できるかは未確認。
  静的リンクでは動かないことは分かっている。
- **MLComputePlan は推定。** Core ML の `Verified` は実行時の配置の保証ではない。

## 参考資料

- Intel Level Zero NPU 拡張: https://github.com/intel/level-zero-npu-extensions (`ze_graph_ext.h`、`ze_command_queue_npu_ext.h`)
- Intel NPU UMD: https://github.com/intel/linux-npu-driver (`docs/overview.md`)。KMD の UAPI `include/uapi/drm/ivpu_accel.h`
- OpenVINO NPU プラグイン: `src/plugins/intel_npu/src/compiler_adapter/src/{model_serializer,ze_graph_ext_wrappers,driver_compiler_adapter}.cpp`
- AMD KMD の UAPI `include/uapi/drm/amdxdna_accel.h`、`Documentation/accel/amdxdna/amdnpu.rst`
- AMD XRT shim: https://github.com/amd/xdna-driver (`src/shim/{hwctx,hwq,buffer,device,pcidev,pcidrv}.cpp`、`src/shim/kmq/`)
- XRT ヘッダ: `xrt/detail/ert.h`、`xrt/detail/xclbin.h`、`xrt/xrt_kernel.h`、`xrt/xrt_hw_context.h`、`xrt/experimental/xrt_module.h`
- mlir-aie の Windows ホストビルド: https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWin.md
- Ryzen AI ドキュメント (Windows の xrt-smi): https://ryzenai.docs.amd.com/en/latest/xrt_smi.html
- purego: https://github.com/ebitengine/purego

## 採らなかったもの

| もの | 理由 |
|---|---|
| 変換器での汎用の取り込み (ONNX など) | 取り込んだグラフには NPU 向けの書き換えを入れにくい。LLM は構造のパターンが少なく、手で移す方が安い |
| 中間形式としての ONNX | `config.json` + 重みから直接書く |
| Intel での ioctl 直叩き | ファームウェアのジョブ形式が UMD の内部 |
| purego からの XRT の C API | コンテキストは C++ 専用、カーネルを開く API は deprecated、可変長引数 |
| ONNX Runtime の EP をバックエンドに | EP はグラフを分けて黙って CPU に回すので厳格モードが成り立たない。ONNX Runtime は `models.yaml` が CPU に置くモデルの CPU の実行系 (`internal/onnxrt`、float32、CPU の EP) としてだけ使う |
| 別モジュールの `contrib/`、`cmd/hwnpuinfo`、公開の `xdna/` と `levelzero/` のヘルパ | 不要だった: CPU の実行系は `internal/onnxrt`、診断は `hwnpu-dev info` と `hwnpu status` |
| 量子化済みの公開物の量子化し直し | 実行先ごとに bf16 から量子化する。量子化済みのものは劣化なしに並べ替えられる形式のときだけ使う |
| 32 ビット | C の構造体は全て 64 ビット前提 |
