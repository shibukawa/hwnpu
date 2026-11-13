# hwnpu 設計書: CPU内蔵NPUをGoから直接使うライブラリ

- 対象: Intel Core Ultra 内蔵 NPU (Meteor Lake 以降)、AMD Ryzen AI 内蔵 NPU (XDNA / Phoenix 以降)
- 制約: cgo を使わない (purego または純粋な syscall)
- Apple: Core ML 経由で同じ API を提供する。ANE (Apple Neural Engine) を直接は使えないが、GPU を除外し、ANE で動くかを実行前に検証する
- 目的: 軽量モデルを低消費電力で動かすこと。GPU は対象外
- 作成日: 2026-10-11 / 改訂: 2026-10-27 (NPU 専用・省電力の方針、Core ML を本体に格上げ)

---

## 1. 要約

| ベンダー | OS | 採用経路 | cgo なしで可能か | 備考 |
|---|---|---|---|---|
| Intel NPU | Linux / Windows | Level Zero (`libze_loader.so.1` / `ze_loader.dll`) + `ZE_extension_graph` を purego で呼ぶ | 可 | ドライバ内コンパイラ (OpenVINO IR → blob) も同じ経路で使える |
| AMD XDNA | Linux | カーネルドライバ `amdxdna` の ioctl を `golang.org/x/sys/unix` で直接叩く | 可 (purego すら不要) | XRT の C API は NPU 向けに実質使えないため、XRT shim (Apache-2.0, C++) を Go に移植する |
| AMD XDNA | Windows | `xrt_coreutil.dll` は C++ ABI 中心。第1段階では非対応。旧 C API の動作検証後に判断 | 要検証 | 代替は contrib の ONNX Runtime (VitisAI EP) |
| Apple ANE | macOS | Core ML を `purego/objc` で呼ぶ | 可 | `CPUAndNeuralEngine` で GPU を除外し、`MLComputePlan` で ANE 配置を事前検証する |

設計上の最重要判断は「**統一抽象はデバイス・バッファ・同期までに留め、計算の投入は能力 (capability) 別の 2 つのインターフェイスに分ける**」ことです。

- `GraphRunner`: コンパイル済みニューラルネットグラフの推論 (Intel、Apple。将来 AMD は ONNX Runtime 経由で)
- `KernelRunner`: ユーザーが用意したカーネルバイナリ (xclbin / ELF) と命令列の実行 (AMD)

Intel と AMD はプログラミングモデルが根本的に違います。Intel はネットワーク全体をドライバ内コンパイラでコンパイルして実行する「グラフ実行機」で、ユーザーカーネルの公開 API はありません。AMD は AIE タイル配列に IRON / mlir-aie / Peano でビルドしたカーネル (overlay + ctrlcode) を流し込む「カーネル実行機」で、公開されたドライバ内コンパイラはありません。この非対称を無理に隠すと両方で使いにくい API になるので、隠しません。

### 1.1 目標と非目標

目標:

- **NPU で動くことの保証。** 既存環境 (ONNX Runtime の EP、OpenVINO の AUTO) は未対応の演算を黙って CPU/GPU に回す。hwnpu は既定で厳格モードとし、NPU だけで完結しないモデルは実行前にエラーにして、原因の演算を返す
- **電力の制御と計測。** 電力ポリシーを第一級の設定にし、可能なバックエンドでは NPU の消費電力を読めるようにする
- **軽量モデルへの備え。** 形状違いのコンパイル結果の共存、重み共有、コンパイル結果の永続キャッシュ
- **cgo なし、依存最小。** 本体の依存は purego と x/sys のみ

非目標:

- GPU (NVIDIA Tensor Core、Intel XMX、AMD RDNA を含む)。必要なら ONNX Runtime など既存のエンジンを使う
- 自前のグラフ IR、モデル変換器、ONNX リーダー。入力はベンダー固有の成果物に限る
- データセンター向け NPU (AWS Inferentia、Google TPU など)

---

## 2. 前提調査: ベンダーごとのソフトウェアスタック

### 2.1 Intel NPU

```
アプリ (Go, purego)
  └─ libze_loader.so.1 / ze_loader.dll        (oneAPI Level Zero loader, MIT)
       └─ libze_intel_npu.so.1 / ze_intel_npu64.dll(要検証)   (UMD, intel/linux-npu-driver, MIT)
            ├─ libnpu_driver_compiler.so   (compiler-in-driver, OpenVINO IR → NPU blob)
            └─ /dev/accel/accelN           (KMD: intel_vpu / drivers/accel/ivpu, GPL)
```

- Level Zero の NPU 拡張 (`intel/level-zero-npu-extensions`, MIT) は全て C ABI。`ze_graph_ext.h` のバージョン付き DDI テーブル (`ze_graph_dditable_ext_t`, 現在 1.20) で全機能にアクセスする。
- ファームウェアへのジョブ投入フォーマットは UMD 内部実装で、`ivpu_accel.h` の UAPI には書かれていない。したがって **ioctl 直叩きは現実的でなく、Level Zero 経由が唯一の筋のよい道**。
- コンパイル済み blob (`ZE_GRAPH_FORMAT_NATIVE`) と OpenVINO IR (`ZE_GRAPH_FORMAT_NGRAPH_LITE`) の両方を `zeGraphCreate` に渡せる。後者はドライバ内コンパイラが処理する。
- ドライバ / コンパイラ / ファームウェアはバージョンが揃っていないと "MAPPED_INFERENCE_VERSION is NOT compatible with the ELF" で失敗する。ライブラリ側で検出して分かりやすく報告する必要がある。
- Linux では `/dev/accel/accel0` の権限 (`render` グループ) が必要。

### 2.2 AMD XDNA (Ryzen AI)

```
アプリ (Go)
  └─ (本設計) golang.org/x/sys/unix で ioctl / mmap
       └─ /dev/accel/accelN   (KMD: amdxdna / drivers/accel/amdxdna, Linux 6.14 で upstream 化)

参考: 公式経路
アプリ (C++) ─ libxrt_coreutil.so.2 + xrt_plugin.*-amdxdna (shim, github.com/amd/xdna-driver, Apache-2.0)
```

- カーネル UAPI (`include/uapi/drm/amdxdna_accel.h`) は小さく安定している: `CREATE_HWCTX / DESTROY_HWCTX / CONFIG_HWCTX / CREATE_BO / GET_BO_INFO / SYNC_BO / EXEC_CMD / WAIT_CMD / GET_INFO / SET_STATE / GET_ARRAY` の 11 個。
- 実行単位は「ハードウェアコンテキスト」(列単位の空間パーティション) に対して、`ert_start_kernel_cmd` 形式のコマンドパケットを CMD BO で投げる。
- カーネルバイナリは xclbin (AXLF コンテナ。`AIE_PARTITION` セクションに PDI = overlay が入る) と ctrlcode (IRON が出力する `insts.bin`)。新しい世代では ELF ベースの流れ (`xrt::elf` / `xrt::module`、`ERT_START_NPU_PREEMPT_ELF`) に移行中。
- XRT の C API は使えない: `xrt_hw_context.h` と `xrt_module.h` は C++ 専用、`xrt::kernel(device, uuid, name)` は deprecated、残る C API (`xrtPLKernelOpen`, `xrtRunSetArg(...)`) は可変長引数で purego と相性が悪い。**AMD/Linux は ioctl 直叩きを第一候補にする**。shim が Apache-2.0 で公開されているので移植元として使える。

### 2.3 Apple

- ANE (`AppleNeuralEngine.framework`, `_ANEClient`) は非公開 API。Core ML のみが公開経路。
- `purego/objc` で `MLModel` / `MLDictionaryFeatureProvider` / `MLMultiArray` を呼べるため、cgo なしで `GraphRunner` を実装できる。purego の一部なので追加依存は無く、本体モジュールに入れる。
- 「ANE のみ」は指定できないが、`MLComputeUnitsCPUAndNeuralEngine` (macOS 13 以降) で GPU は除外できる。
- `MLComputePlan` (macOS 14.4 以降) で、演算ごとの推奨実行デバイスと対応デバイスを実行前に取得できる。これを厳格モードの検証に使う。ただし計画は推定であり、実行時の配置を保証するものではない。

### 2.4 Qualcomm (将来)

- Snapdragon X の Hexagon NPU は QNN SDK (`QnnHtp.dll` / `libQnnHtp.so`) の C API (`QnnInterface_getProviders`) で、purego 向き。本設計の `GraphRunner` にそのまま収まるが、v1 の範囲外とする。

---

## 3. 設計方針

### 3.1 抽象の境界

| レイヤ | 共通化する | 理由 |
|---|---|---|
| デバイス列挙・オープン | する | sysfs / loader / dll の違いを吸収するのは安価 |
| デバイス可視メモリ (`Buffer`) | する | 「ゼロコピーの `[]byte` ビュー + 明示的 Sync」はどちらにも自然に対応する |
| 完了待ち (`Fence`) | する | `context.Context` でキャンセルできる Wait が欲しい |
| 計算の投入 | しない (能力別) | グラフ推論とカーネル実行は引数の型も失敗の仕方も違う |
| モデル形式 | しない | OpenVINO IR / xclbin / mlmodelc は相互変換できない |

### 3.2 バックエンド選定の根拠

- **Intel = Level Zero (purego)**: 唯一の公開経路。C ABI で、構造体は全てポインタ渡しなので purego と相性がよい。
- **AMD Linux = ioctl 直叩き**: 依存ゼロ、ビルドは純粋な Go。代償は shim の移植と、コマンドフォーマットの追従コスト (後述のリスク)。
- **AMD Windows = 保留**: Windows の KMD インターフェイスは非公開で、公開されているのは `xrt_coreutil.dll` (NPU ドライバインストーラが `C:\Windows\System32[\AMD]` に配置) だけ。第 8 章で選択肢を整理する。
- **Apple = Core ML (purego/objc)**: NPU を直接は使えないが、GPU 除外と事前検証で「ほぼ NPU」を担保できる。開発機 (arm64 Mac) で API を確かめられる価値も大きいので本体に入れる。

### 3.3 cgo 禁止から導かれる規約

- C 側が Go のメモリを保持する状況を作らない。デバイス可視メモリは必ずドライバ側で確保する (`zeMemAllocHost` / BO の mmap)。
- 構造体は値渡ししない。全てポインタで渡し、Go 側の struct は C のレイアウトを手で再現する (パディングも明示)。
- コールバックは使わない (Level Zero のイベントはポーリング、amdxdna は ioctl で待つ)。
- Windows では purego の `Dlopen` が無いので `golang.org/x/sys/windows.LoadLibraryEx` + `purego.RegisterFunc` を使う。
- Linux で `CGO_ENABLED=0` の場合、purego は内部の fakecgo で `dlopen` を動かす。静的リンク (`-linkmode external -extldflags -static` や musl 環境) では動かないので README に明記する。

---

## 4. パッケージ構成

モジュールパスは `github.com/shibukawa/hwnpu` とする。

```
hwnpu/
├── hwnpu.go            公開 API (Device, Buffer, Fence, GraphRunner, KernelRunner, errors)
├── enumerate.go        デバイス列挙 (OS ごとに build tag)
├── options.go
├── internal/
│   ├── dl/             purego / LoadLibrary の薄いラッパ。ライブラリ探索順とエラー整形
│   ├── ze/             Level Zero core + NPU 拡張の Go バインディング (構造体定義、DDI テーブル)
│   ├── backend/
│   │   ├── backend.go  バックエンドが実装する内部インターフェイス
│   │   ├── levelzero/  Intel: ze を使って Device/Buffer/Graph を実装
│   │   ├── coreml/     Apple: purego/objc で MLModel / MLComputePlan を呼ぶ (darwin のみ)
│   │   └── xdna/       AMD Linux: ioctl 実装。uapi.go (構造体と ioctl 番号), hwctx.go, bo.go, exec.go
│   ├── xclbin/         AXLF パーサ (IP_LAYOUT, AIE_PARTITION, EMBEDDED_METADATA)
│   ├── ert/            ERT コマンドパケットのビルダ
│   └── accelfs/        /sys/class/accel 走査 (Linux 共通)
├── xdna/               AMD 固有の公開ヘルパ (insts.txt の読み込み、QoS 型など)
├── levelzero/          Intel 固有の公開ヘルパ (ビルドフラグのビルダ、Unwrap)
├── cmd/hwnpuinfo/      診断 CLI (vulkaninfo 相当)
├── contrib/
│   └── onnxrt/         別 go.mod。ONNX Runtime C API。NPU で完結しないモデルの逃げ道 (任意)
└── docs/
```

依存は本体では `github.com/ebitengine/purego` と `golang.org/x/sys` のみ。

---

## 5. 公開 API

### 5.1 デバイス

```go
package hwnpu

type Vendor uint8

const (
	VendorUnknown Vendor = iota
	VendorIntel          // Level Zero
	VendorAMD            // amdxdna
	VendorApple          // Core ML
)

type Capability uint32

const (
	CapGraph        Capability = 1 << iota // GraphRunner が使える
	CapCompile                             // ドライバ内コンパイラがある (IR → blob)
	CapKernel                              // KernelRunner が使える
	CapHostCoherent                        // Buffer.Sync* が不要 (no-op)
	CapStrictVerify                        // NPU 配置を実行前に検証できる
	CapPowerControl                        // PowerPolicy が効く
	CapPowerMeter                          // NPU の消費電力を読める
)

type DeviceInfo struct {
	Index    int
	Vendor   Vendor
	Backend  string // "levelzero" | "xdna" | "coreml"
	Name     string // "Intel NPU (Lunar Lake)", "AMD NPU (4x8 columns)" など
	Path     string // "/dev/accel/accel0" など。Windows では空
	Caps     Capability
	Driver   string // UMD / KMD / 拡張バージョン
	Firmware string
}

// Enumerate はライブラリをロードできる範囲で全デバイスを列挙する。
// ロードに失敗したバックエンドは errors.Join で返し、列挙自体は継続する。
func Enumerate(ctx context.Context) ([]DeviceInfo, error)

// Open は最初に見つかった使えるデバイスを開く。WithVendor などで絞れる。
func Open(ctx context.Context, opts ...Option) (*Device, error)
func OpenDevice(ctx context.Context, info DeviceInfo, opts ...Option) (*Device, error)

type Device struct{ /* unexported */ }

func (d *Device) Info() DeviceInfo
func (d *Device) Close() error
func (d *Device) AllocBuffer(size int, opts ...BufferOption) (*Buffer, error)
func (d *Device) Graph() (GraphRunner, error)   // 非対応なら ErrUnsupported
func (d *Device) Kernel() (KernelRunner, error) // 非対応なら ErrUnsupported
```

### 5.2 バッファと同期

```go
// Buffer はデバイスから見えるメモリ。Bytes() はゼロコピーのビューを返す。
type Buffer struct{ /* unexported */ }

func (b *Buffer) Bytes() []byte
func (b *Buffer) Size() int
func (b *Buffer) SyncToDevice(off, n int) error   // CPU 書き込み後、投入前に呼ぶ
func (b *Buffer) SyncFromDevice(off, n int) error // 完了後、CPU 読み出し前に呼ぶ
func (b *Buffer) Close() error

type Fence interface {
	Wait(ctx context.Context) error // ctx のキャンセルで ctx.Err() を返す
	Done() (bool, error)
}
```

Intel は `CapHostCoherent` を立てて Sync を no-op にする。AMD は SHMEM BO がキャッシュ可能なので Sync が必須 (ioctl `SYNC_BO`。x86 では Go アセンブリの `CLFLUSH` + `MFENCE` に置き換える最適化を後で入れられる)。

### 5.3 GraphRunner (Intel、Apple、contrib の ONNX Runtime)

```go
type GraphFormat uint8

const (
	GraphFormatNative     GraphFormat = iota // ベンダー固有のコンパイル済み blob
	GraphFormatOpenVINOIR                    // xml + bin。ドライバ内コンパイラが必要
)

type GraphSource struct {
	Format     GraphFormat
	Path       string // ファイルまたはディレクトリ (Core ML: .mlpackage / .mlmodel / .mlmodelc)
	Blob       []byte // Native
	XML, Bin   []byte // OpenVINO IR
	BuildFlags string // 例: `--inputs_precisions="0:fp16" --config NPU_PLATFORM="4000"`
}

type GraphRunner interface {
	Compile(ctx context.Context, src GraphSource) (*Graph, error)
}

type ArgKind uint8 // ArgInput, ArgOutput

type ArgInfo struct {
	Index  int
	Kind   ArgKind
	Name   string
	Dims   []int
	DType  DType  // F32, F16, BF16, I8, U8, I32, ... (networkPrecision 由来)
	Layout string // "NCHW" など
	Bytes  int    // 必要なバッファサイズ
}

type Graph struct{ /* unexported */ }

func (g *Graph) Args() []ArgInfo                 // 入力が先、出力が後。各グループ内の順序はバックエンドが決める (Core ML は名前順)
func (g *Graph) Arg(name string) (ArgInfo, bool)
func (g *Graph) Placement() PlacementReport
func (g *Graph) NativeBlob() ([]byte, error)     // アプリ側キャッシュ用 (M3 で追加予定)
func (g *Graph) NewRequest() (*Request, error)
func (g *Graph) Close() error

// Request は 1 本のコマンドリストに対応し、同時に 1 つの goroutine からだけ使う。
type Request struct{ /* unexported */ }

func (r *Request) Bind(index int, b *Buffer) error
func (r *Request) BindName(name string, b *Buffer) error
func (r *Request) Submit() (Fence, error)
func (r *Request) Run(ctx context.Context) error // Submit + Wait
func (r *Request) Close() error
```

### 5.4 KernelRunner (AMD)

```go
type ProgramSource struct {
	Xclbin []byte      // 現行の流れ
	ELF    []byte      // 将来の ELF 流れ (xrt::elf 相当)
	QoS    xdna.QoS    // gops, fps, latency, priority など。ゼロ値で既定
}

type KernelRunner interface {
	Load(ctx context.Context, src ProgramSource) (*Program, error)
}

type Program struct{ /* unexported */ } // = ハードウェアコンテキスト 1 つ

func (p *Program) Kernels() []string
func (p *Program) Kernel(name string) (*Kernel, error)
func (p *Program) Close() error

type Kernel struct{ /* unexported */ }

func (k *Kernel) NewRun() *Run

// Run は 1 つの CMD BO に対応する。引数の順序と型は xclbin のカーネル署名に従う。
type Run struct{ /* unexported */ }

func (r *Run) SetBuffer(i int, b *Buffer) *Run
func (r *Run) SetUint32(i int, v uint32) *Run
func (r *Run) SetUint64(i int, v uint64) *Run
func (r *Run) Submit() (Fence, error)
func (r *Run) Run(ctx context.Context) error
func (r *Run) Close() error
```

IRON の典型的なホストコードは `kernel(opcode, bo_instr, instr_count, bo_in, bo_out)` という呼び出しなので、Go では次の形になる。

```go
prog, _ := kr.Load(ctx, hwnpu.ProgramSource{Xclbin: xclbinBytes})
k, _ := prog.Kernel("MLIR_AIE")
instr, _ := dev.AllocBuffer(len(insts))        // insts.bin の内容
copy(instr.Bytes(), insts)
instr.SyncToDevice(0, len(insts))
err := k.NewRun().
	SetUint64(0, 3).                           // opcode (IRON の慣習)
	SetBuffer(1, instr).SetUint32(2, uint32(len(insts)/4)).
	SetBuffer(3, in).SetBuffer(4, out).
	Run(ctx)
out.SyncFromDevice(0, out.Size())
```

### 5.5 厳格モードと電力

```go
// Placement はコンパイル時の NPU 配置の検証方針。
type Placement uint8

const (
	PlacementStrict   Placement = iota // 既定。NPU 以外に回る演算があれば ErrNotOnNPU
	PlacementPrefer                    // 回る演算を Report に記録して続行
)

type PlacementReport struct {
	Verified bool          // バックエンドが検証できたか (できなければ false)
	Method   string        // 検証手段。例 "coreml-compute-plan"
	Reason   string        // 検証できなかった理由
	NPUOps   int           // NPU に配置された演算の数
	Ops      []OpPlacement // NPU 以外に回る演算の一覧
}

func (r PlacementReport) OnNPU() bool // Verified かつ Ops が空

// 厳格モードの Compile は *PlacementError を返す。errors.Is(err, ErrNotOnNPU) が真になる。
type PlacementError struct{ Report PlacementReport }

type OpPlacement struct{ Name, Type, Device string }

func (g *Graph) Placement() PlacementReport

// PowerPolicy はバックエンドごとの設定に写像される。
type PowerPolicy uint8

const (
	PowerDefault     PowerPolicy = iota
	PowerLow                     // Intel: workload=BACKGROUND, turbo off / AMD: power mode LOW / Apple: 効果なし
	PowerBalanced                // AMD: MEDIUM
	PowerPerformance             // Intel: turbo on / AMD: HIGH または TURBO
)

type PowerSample struct {
	Time   time.Time
	Watts  float64
	Source string // "amdxdna-sensor", "ivpu-metric" など
}

func (d *Device) PowerSample() (PowerSample, error) // CapPowerMeter が無ければ ErrUnsupported
```

バックエンドごとの検証手段:

| バックエンド | 厳格モードの検証 | 電力制御 | 電力計測 |
|---|---|---|---|
| Intel | 対応レイヤー問い合わせ (`QueryNetworkCreate2` / `GetSupportedLayers`)。NPU プラグインはフォールバックしないのでコンパイル成功で NPU 実行が確定 | コマンドキュー拡張の workload type と turbo | メトリックストリーマー (要検証: 電力値が取れるか) |
| AMD | xclbin / ELF は NPU 専用なので常に満たす | `SET_STATE(SET_POWER_MODE)`、hwctx の QoS | `GET_INFO(QUERY_SENSORS)` の POWER |
| Apple | `MLComputePlan` で全演算の推奨デバイスが Neural Engine か確認。推定なので `Verified` は true でも保証ではない旨を文書化 | なし (compute units の選択のみ) | 公開 API なし。`powermetrics` (root 必須) の ANE Power を外部で使う |

### 5.6 軽量モデル向けの機能

- **形状違いの共存。** NPU は静的形状を好むため、LLM ではプリフィル用とデコード用など形状の違う複数のグラフを切り替える。`CompileSet` で複数の `GraphSource` を一度にコンパイルし、重みを共有させる (Intel はコンパイラの重み分離機能、要検証: コンパイラ 6.3 以降)
- **コンパイルキャッシュ。** コンパイル自体が電力を消費するので、起動ごとに再コンパイルしない。`WithCacheDir(path)` で、Intel は `NativeBlob`、Apple はコンパイル済み `.mlmodelc` を保存する。キーはモデルのハッシュ、ビルドフラグ、ドライバとコンパイラのバージョン

### 5.7 エラー

```go
var (
	ErrNoDevice    = errors.New("hwnpu: no NPU device found")
	ErrUnsupported = errors.New("hwnpu: not supported by this device/backend")
	ErrLibrary     = errors.New("hwnpu: runtime library not found") // 探したパス一覧を Wrap する
	ErrNotOnNPU    = errors.New("hwnpu: model does not run entirely on the NPU") // PlacementReport を伴う
)

type DriverError struct { Backend string; Op string; Code int64; Msg string }   // ze_result_t / errno
type BuildError  struct { Log string; Err error }                              // コンパイル失敗時のビルドログ
type VersionError struct { Component string; Have, Need string }               // 拡張バージョン不足など
```

### 5.8 オプション

- `WithVendor(Vendor)`, `WithBackend(string)`, `WithIndex(int)`
- `WithLibraryPath(name, path string)`: `ze_loader` などの明示指定。環境変数 `HWNPU_ZE_LOADER`, `HWNPU_ACCEL_DEVICE` でも上書き可
- `WithPowerPolicy(PowerPolicy)`: 共通の電力ポリシー。下の個別設定が優先する
- `WithPlacement(Placement)`: 既定は `PlacementStrict`
- `WithCacheDir(string)`: コンパイルキャッシュ
- `WithTurbo(bool)`, `WithWorkloadType(Default|Background)`: Intel のコマンドキュー NPU 拡張に対応
- `WithQoS(xdna.QoS)`: AMD のハードウェアコンテキスト QoS

---

## 6. Intel バックエンド (`internal/backend/levelzero`)

### 6.1 ライブラリ解決

1. `HWNPU_ZE_LOADER` → 2. `libze_loader.so.1` / `ze_loader.dll` (通常の検索パス) → 3. 既知のインストール先 (Linux: `/usr/lib/x86_64-linux-gnu`, `/usr/local/lib`; Windows: `%SystemRoot%\System32`, oneAPI の `bin`)。
失敗時は試したパスを `ErrLibrary` に含める。

Windows で Level Zero loader が同梱されない構成がありうる。回避策として、ドライバ DLL を直接 `LoadLibrary` して `zeGetDriverProcAddrTable` などの DDI テーブルを引く「loader バイパスモード」を v2 の候補にする (NPU UMD はこれらをエクスポートしている)。

### 6.2 初期化とデバイス選択

```
zeInit(ZE_INIT_FLAG_VPU_ONLY = 0x2)
zeDriverGet → 各 driver で zeDeviceGet → zeDeviceGetProperties.type == ZE_DEVICE_TYPE_VPU (5)
zeDriverGetExtensionProperties → 名前が "ZE_extension_graph" か "ZE_extension_graph_<数字>..." のものを選ぶ (OpenVINO と同じ規則。16.2 参照)
zeDriverGetExtensionFunctionAddress(driver, name, &table)                  → ze_graph_dditable_ext_t*
同様に "ZE_extension_command_queue_npu" → ze_command_queue_npu_dditable_ext_t* (任意)
zeContextCreate
graph.pfnDeviceGetGraphProperties → graphExtensionVersion, compilerVersion, graphFormatsSupported
```

`Caps` は `graphFormatsSupported` に `NGRAPH_LITE` が含まれるときだけ `CapCompile` を立てる。

### 6.3 DDI テーブル

`ze_graph_dditable_ext_t` は関数ポインタの配列で、拡張バージョンごとに末尾に追加される。Go では `[N]uintptr` として受け取り、交渉したバージョン以下のエントリだけを `purego.RegisterFunc` で Go 関数に束縛する。

| 版 | 追加されるエントリ (順序どおり) |
|---|---|
| 1.0 | Create, Destroy, GetProperties, GetArgumentProperties, SetArgumentValue, AppendGraphInitialize, AppendGraphExecute, GetNativeBinary, DeviceGetGraphProperties |
| 1.1 | GraphGetArgumentMetadata, GetArgumentProperties2 |
| 1.2 | GetArgumentProperties3 |
| 1.3 | QueryNetworkCreate, QueryNetworkDestroy, QueryNetworkGetSupportedLayers |
| 1.4 | BuildLogGetString |
| 1.5 | Create2, QueryNetworkCreate2, QueryContextMemory |
| 1.6 | DeviceGetGraphProperties2 |
| 1.7 | GetNativeBinary2 |
| 1.8 | GetProperties2, GraphInitialize |
| 1.11 | CompilerGetSupportedOptions, CompilerIsOptionSupported |
| 1.12 | Create3, GetProperties3, BuildLogGetString2, BuildLogDestroy |
| 1.15 | SetArgumentValue2 |
| 1.16 | Evict |
| 1.20 | GetArgumentProperties4, GetArgumentNames |

最低要求は 1.5 (`Create2` と `ze_graph_desc_2_t`) とし、それ未満は `VersionError` にする。1.8 以上では `GetProperties2` の `initStageRequired` を見て `GraphInitialize` を直接呼ぶ (OpenVINO と同じ分岐)。

### 6.4 グラフ生成

- `GraphFormatNative`: `ze_graph_desc_2_t{format: NATIVE, pInput: blob, inputSize, pBuildFlags: "", flags}` をそのまま渡す。blob がページ境界揃いなら `ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT` を立ててコピーを避けられる (要検証: 1.13 以降)。
- `GraphFormatOpenVINOIR`: OpenVINO の `model_serializer.cpp` (ALL_WEIGHTS_COPY 形式) と同じバイト列を組み立てる。

```
compilerVersion (ze_graph_compiler_version_info_t, DeviceGetGraphProperties で得た値をそのまま)
uint32 numberOfInputData = 2
uint64 xmlSize, xml bytes
uint64 weightsSize, weights bytes
```

`pBuildFlags` は OpenVINO の `serializeIOInfo` / `serializeConfig` が生成する文字列形式 (`--inputs_precisions="..." --inputs_layouts="..." --outputs_precisions="..." --outputs_layouts="..." --config KEY="VALUE" ...`) に合わせる。`levelzero.BuildFlags` ビルダを用意し、利用者が直接文字列を書かなくて済むようにする。失敗時は `BuildLogGetString` (1.4 以上) か、1.12 以上なら `Create3` で得たビルドログハンドルから `BuildError.Log` を埋める。

### 6.5 引数とメモリ

- `AllocBuffer` は `zeMemAllocHost(ctx, {flags: 0}, size, align: 4096)` を使い、返ったポインタを `unsafe.Slice` で `[]byte` にする。解放は `zeMemFree`。
- `SetArgumentValue(graph, i, ptr)` に渡せるのは Level Zero が確保したメモリだけなので、`Bind` は `*Buffer` しか受け付けない (Go のスライスは受け付けない。これが cgo ポインタ規則上も正しい)。
- 引数のメタデータは `GetArgumentProperties3` (1.2 以上) から `ArgInfo` に写す。`dims_count` と `dims` から形状、`networkPrecision` から `DType`、`type` から入出力を決める。

### 6.6 実行

```
zeCommandQueueCreate(ctx, dev, {ordinal: 0, mode: DEFAULT, priority: NORMAL, pNext: ze_command_queue_desc_npu_ext_2_t{options: TURBO?}})
zeCommandListCreate(ctx, dev, {commandQueueGroupOrdinal: 0})
(初回のみ) GraphInitialize(graph) または AppendGraphInitialize(cl, graph, nil, 0, nil)
AppendGraphExecute(cl, graph, nil, nil, 0, nil)
zeCommandListClose(cl)
zeFenceCreate(queue) → zeCommandQueueExecuteCommandLists(queue, 1, &cl, fence)
```

`Fence.Wait` は `zeFenceHostSynchronize(fence, 50ms)` を `ctx.Done()` を見ながら繰り返す。`Request` は `Submit` ごとに `zeFenceReset` + `zeCommandListReset` して再記録する。引数差し替えのたびに再記録するコストが問題になったら `zeMutableCommandList` 拡張 (ドライバ v1.6.0 以降) を導入する。

### 6.7 構造体レイアウト

`ze_graph_desc_2_t`, `ze_graph_argument_properties_3_t`, `ze_device_graph_properties_t`, Level Zero core の各 desc は Go の struct で C と同じレイアウトを書く。`stype` は `uint32` + 4 バイトパディング + `pNext uintptr` という並びになるので、`unsafe.Sizeof` / `Offsetof` を C の `sizeof` / `offsetof` と突き合わせる golden テストを置く (第 11 章)。

---

## 7. AMD バックエンド (`internal/backend/xdna`, Linux)

### 7.1 デバイス発見

`/sys/class/accel/accel*/device/` を走査し、`driver` シンボリックリンクの先が `amdxdna` なら AMD、`intel_vpu` なら Intel と判定する (PCI ID のハードコードは最小限にする)。ノードは `/dev/accel/accelN` を `O_RDWR|O_CLOEXEC` で開く。Intel もここで検出して、Level Zero のロード前に「デバイスはあるがライブラリが無い」と「デバイスが無い」を区別する。

### 7.2 ioctl の符号化

```go
// _IOC(dir, type, nr, size): dir<<30 | size<<16 | type<<8 | nr
const drmIoctlBase = 'd'    // 0x64
const drmCommandBase = 0x40

func iowr(nr uintptr, size uintptr) uintptr { return 3<<30 | size<<16 | drmIoctlBase<<8 | nr }
func iow(nr uintptr, size uintptr) uintptr  { return 1<<30 | size<<16 | drmIoctlBase<<8 | nr }

var (
	ioctlCreateHwctx  = iowr(drmCommandBase+0, unsafe.Sizeof(createHwctx{}))
	ioctlDestroyHwctx = iowr(drmCommandBase+1, unsafe.Sizeof(destroyHwctx{}))
	ioctlConfigHwctx  = iowr(drmCommandBase+2, unsafe.Sizeof(configHwctx{}))
	ioctlCreateBO     = iowr(drmCommandBase+3, unsafe.Sizeof(createBO{}))
	ioctlGetBOInfo    = iowr(drmCommandBase+4, unsafe.Sizeof(getBOInfo{}))
	ioctlSyncBO       = iowr(drmCommandBase+5, unsafe.Sizeof(syncBO{}))
	ioctlExecCmd      = iowr(drmCommandBase+6, unsafe.Sizeof(execCmd{}))
	ioctlGetInfo      = iowr(drmCommandBase+7, unsafe.Sizeof(getInfo{}))
	ioctlSetState     = iowr(drmCommandBase+8, unsafe.Sizeof(setState{}))
	ioctlWaitCmd      = iow (drmCommandBase+9, unsafe.Sizeof(waitCmd{}))
	ioctlGetArray     = iowr(drmCommandBase+10, unsafe.Sizeof(getArray{}))
)
```

構造体は `amdxdna_accel.h` をそのまま写す (全て `__u32`/`__u64` で自然に揃うので手動パディングは不要)。

```go
type createHwctx struct {
	Ext, ExtFlags, QoSPtr          uint64
	UMQBO, LogBufBO, MaxOPC        uint32
	NumTiles, MemSize, UMQDoorbell uint32
	Handle, SyncobjHandle          uint32
}
type createBO struct { Flags, VAddr, Size uint64; Type, Handle uint32 }
type getBOInfo struct { Ext, ExtFlags uint64; Handle, Pad uint32; MapOffset, VAddr, XdnaAddr uint64 }
type syncBO struct { Handle, Direction uint32; Offset, Size uint64 }
type execCmd struct { Ext, ExtFlags uint64; Hwctx, Type uint32; CmdHandles, Args uint64; CmdCount, ArgCount uint32; Seq uint64 }
type waitCmd struct { Hwctx, Timeout uint32; Seq uint64 }
```

### 7.3 実行までの流れ

```
1. GET_INFO(QUERY_AIE_METADATA)      → cols, rows, core.row_count, version
   GET_INFO(QUERY_FIRMWARE_VERSION)  → DeviceInfo.Firmware
2. CREATE_BO{type: DEV_HEAP, size: HEAP}  (デバイス open ごとに 1 回。DEV BO より先。要検証: サイズ。カーネル文書の
   「命令バッファ 64 MB」と shim の実装を突き合わせる)
   GET_BO_INFO → map_offset, xdna_addr → mmap(fd, map_offset, MAP_SHARED|MAP_LOCKED)
3. xclbin を解析 (internal/xclbin):
   - IP_LAYOUT: m_type == IP_PS_KERNEL(7) のエントリ → カーネル名、ps_kernel ビットフィールドの functional / kernel_id
   - AIE_PARTITION: ncol (info.start_columns / column_width), operations_per_cycle, aie_pdi[] → cdo_groups[].dpu_kernel_ids で
     kernel_id に対応する PDI バイト列を選ぶ
   - EMBEDDED_METADATA (XML): カーネル引数の並びと型 (SetBuffer/SetUint* の検証に使う)
4. PDI ごとに CREATE_BO{type: DEV, size: len(pdi)} → heap 上の仮想アドレス (heap_vaddr + xdna_addr - heap_xdna_addr) に書き込み → SYNC_BO(TO_DEVICE)
5. CREATE_HWCTX{qos_p: &qos, umq_bo: INVALID, log_buf_bo: INVALID, max_opc: operations_per_cycle, num_tiles: ncol * core.row_count}
   → handle, syncobj_handle
6. CONFIG_HWCTX{handle, param_type: CONFIG_CU(0), param_val: &{num_cus, cu_configs: [{cu_bo: pdiBO, cu_func: functional}]}, param_val_size}
7. 入出力: CREATE_BO{type: SHMEM} → GET_BO_INFO → mmap。xdna_addr をカーネル引数に使う
8. コマンド: CREATE_BO{type: CMD, size: 4096} → mmap → ert_start_kernel_cmd を書く
9. EXEC_CMD{hwctx, type: SUBMIT_EXEC_BUF(0), cmd_handles: &cmdHandle, cmd_count: 1, args: &argHandles[], arg_count} → seq
10. WAIT_CMD{hwctx, timeout_ms, seq}  (ctx を見るため短いタイムアウトで繰り返す)
    代替: DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT{handles: [syncobj_handle], points: [seq]}。shim は syncobj があればそちらを使う
11. CMD BO 先頭の state が ERT_CMD_STATE_COMPLETED(4) か確認。ERROR(5) / TIMEOUT(8) などは DriverError にする
12. 出力の SYNC_BO(FROM_DEVICE)
```

### 7.4 ERT パケット (`internal/ert`)

```go
// header: state[3:0] stat_enabled[4] unused[9:5] extra_cu_masks[11:10] count[22:12] opcode[27:23] type[31:28]
func header(state, extraMasks, count, opcode, typ uint32) uint32 {
	return state&0xF | (extraMasks&0x3)<<10 | (count&0x7FF)<<12 | (opcode&0x1F)<<23 | (typ&0xF)<<28
}
// 続いて cu_mask (uint32)、その後に data[] が count-1 ワード
```

- xclbin 流れのカーネル: `opcode = ERT_START_CU(0)`, `type = ERT_CU(3)`, `cu_mask = 1 << cuIndex`, `data[]` はカーネル引数を 32 bit ワード単位で並べたもの (64 bit 値は下位・上位の順)。引数の各オフセットは EMBEDDED_METADATA のレジスタマップに従う。
- ELF 流れ: `ERT_START_NPU(20)` + `ert_npu_data{instruction_buffer u64, instruction_buffer_size u32, instruction_prop_count u32}`、プリエンプション対応は `ERT_START_NPU_PREEMPT(21)` / `ERT_START_NPU_PREEMPT_ELF(22)`。v1 では xclbin 流れのみ実装し、ELF は `ProgramSource.ELF` を予約するに留める。

正確なペイロードは XRT 本体 (`xrt_kernel.cpp`) が組み立てており、shim には現れない。検証手段として、shim のデバッグ機能 (`/tmp/BO_DUMPS.<pid>/exec_buf.<handle>` にコマンド BO と引数 BO をダンプする) を使い、公式 C++ ホストコードで動かした結果と Go 側が生成したバイト列を突き合わせる golden テストを作る。

### 7.5 リスクと手当て

| リスク | 手当て |
|---|---|
| UMQ (ユーザー管理キュー) 世代では `umq_bo` / doorbell を使う別経路になる | `hwq` を内部インターフェイスにし、v1 は KMQ だけ実装。`CREATE_HWCTX` の `umq_doorbell` が返るかで判定 |
| ELF 流れへの移行で xclbin が廃れる | `ProgramSource` にフィールドを予約。`ert` パッケージは opcode ごとのペイロードを分離 |
| ファームウェアとコマンド形式の追従 | `hwnpuinfo` でカーネル / FW バージョンを出し、動作確認済みの組み合わせを README に表で持つ |
| 権限 (`render` グループ) | エラーメッセージに `chmod` / `usermod` の手順を含める |

---

## 8. Windows 対応

### 8.1 Intel

Level Zero loader (`ze_loader.dll`) が PATH にあれば Linux と同じコードで動く。loader の有無と NPU ドライバ DLL の名前は要検証 (`ZE_ENABLE_LOADER_DEBUG_TRACE=1` でロード経路を確認できる)。デバイス発見は sysfs が無いので `zeDeviceGet` の結果だけで行う。

### 8.2 AMD

公開経路が `xrt_coreutil.dll` (C++ ABI) しかないため、次の 3 案を比較して決める。v1 では未対応とし、`Enumerate` では「ドライバはあるが未対応」を `DeviceInfo` として返す。

| 案 | 内容 | 利点 | 欠点 |
|---|---|---|---|
| A | DLL がエクスポートする旧 C API (`xrtPLKernelOpen`, `xrtRunOpen`, `xrtRunSetArg`...) を purego で呼ぶ | 追加バイナリ不要 | `xrt::kernel(device, uuid, name)` は deprecated で NPU で動く保証がない。`xrtRunSetArg` は可変長引数 |
| B | 自前の小さな C シム DLL (XRT C++ API を薄く C に包む) をビルドして同梱し、purego でロード | 確実に動く | ビルド済みバイナリの配布と署名の運用が必要。利用者のビルドは cgo 不要のまま |
| C | contrib/onnxrt (ONNX Runtime + VitisAI EP) で `GraphRunner` として提供 | Windows の公式推奨経路 | カーネル実行はできない。依存が大きい |

まず A を `dumpbin /exports` と小さな実験で検証し、だめなら B。C は A/B と独立に用意できる。

---

## 9. Apple バックエンド (`internal/backend/coreml`, macOS arm64)

### 9.1 実行経路

- `purego/objc` で `MLModel compileModelAtURL:error:` → `modelWithContentsOfURL:configuration:error:`。`MLModelConfiguration.computeUnits = MLComputeUnitsCPUAndNeuralEngine` を既定にし、GPU を使わせない。
- `GraphSource` は `GraphFormatNative` に `.mlpackage` / `.mlmodel` / `.mlmodelc` のパスを取る。コンパイル結果の `.mlmodelc` は `WithCacheDir` に移して再利用する (既定では一時ディレクトリに作られ消える)。
- 入力は `MLMultiArray initWithDataPointer:shape:dataType:strides:deallocator:error:` で `Buffer` をゼロコピーで包み、`MLDictionaryFeatureProvider` に入れる (独自クラス登録を避ける)。`Buffer` はページ境界に揃えた mmap 領域で確保する。
- 実行は `predictionFromFeatures:error:` を goroutine 上で同期呼び出しし、`Fence` で包む。Objective-C の自動解放は呼び出しごとに `objc_autoreleasePoolPush` / `Pop` で囲む。

### 9.2 NPU 配置の検証

- `MLComputePlan loadContentsOfURL:configuration:completionHandler:` (macOS 14.4 以降) で計画を得て、プログラムの各演算について `computeDeviceUsageForMLProgramOperation:` の `preferredComputeDevice` が `MLNeuralEngineComputeDevice` かを確認する。
- 完了ハンドラは purego の `NewCallback` (上限あり、解放されない) を避けたいので、ブロックを 1 つだけ作って使い回す。要検証: purego/objc のブロック対応で足りるか。
- ANE は fp16 前提で、fp32 の演算や動的形状は CPU に回りやすい。coremltools での fp16 変換と形状固定を README で案内し、`PlacementReport` に原因の演算を載せる。
- macOS 14.4 未満、または neuralnetwork 形式 (ML Program 以外) のモデルでは検証できないので `Verified = false` とし、厳格モードではエラーにする。

## 10. contrib: ONNX Runtime (`contrib/onnxrt`, 任意)

- `OrtGetApiBase()` が返す関数ポインタ表は純粋な C ABI で purego 向き。EP は OpenVINO (Intel) / VitisAI (AMD) / CoreML (Apple) / QNN (Qualcomm)。
- NPU で完結しないモデルを動かすための逃げ道。EP はグラフを分割して未対応の演算を CPU に回すため、hwnpu の厳格モードの保証は及ばない。`Placement()` は `Verified = false` を返す。
- 2026-11-10: CPU の実行系として `internal/onnxrt` を作った (CPU の EP だけ、float32 のテンソルだけ)。models.yaml の `device: {VENDOR: cpu}` が置くモデル (小さなモデルで、NPU では演算ごとの費用が計算より重いもの) に使う (16.126)。

---

## 11. purego / ABI 規約 (`internal/dl`, `internal/ze`)

```go
package dl

type Library struct{ handle uintptr }

func Open(candidates ...string) (*Library, error)       // 順に試し、全て失敗したら試した一覧つきの ErrLibrary
func (l *Library) Bind(fptr any, symbol string) error   // purego.RegisterLibFunc の panic を error に変換
func (l *Library) Sym(symbol string) (uintptr, error)
func BindAddr(fptr any, addr uintptr)                   // DDI テーブルの関数ポインタ用 (purego.RegisterFunc)
```

- Linux/macOS: `purego.Dlopen(name, RTLD_NOW|RTLD_LOCAL)`。Windows: `windows.LoadLibraryEx(name, 0, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)` + `GetProcAddress`。
- 文字列は `[]byte` に NUL を付けて渡し、呼び出しが終わるまで `runtime.KeepAlive`。purego の自動コピーに頼らない (長期保持される `pBuildFlags` などに備える)。
- 浮動小数点引数は使わない (Level Zero / ioctl ともに不要)。
- `purego.SyscallN` は使わず、型付きの `RegisterFunc` に統一する。
- 全ての C 構造体は `//go:build amd64 || arm64` の下で 64 bit 前提に書く。32 bit は非対応と明記する。

---

## 12. テストと診断

| 種別 | 内容 | ハードウェア |
|---|---|---|
| ABI golden | Go の `unsafe.Sizeof` / `Offsetof` を、C ヘッダから一度生成して JSON で保存した値と比較 (`ze_*`, `amdxdna_*`, `axlf`, `ert_*`) | 不要 |
| ioctl 番号 | `iowr(...)` の結果を既知の定数 (例 `DRM_IOCTL_AMDXDNA_CREATE_BO = 0xC0206443`) と比較 | 不要 |
| xclbin パーサ | mlir-aie のサンプル xclbin を fixture にし、カーネル名・PDI 数・ncol を検証 | 不要 |
| ERT golden | shim の BO ダンプと Go 生成パケットのバイト比較 | ダンプ採取時のみ |
| バックエンド単体 | `internal/backend` のインターフェイス越しにフェイクを差し、公開 API の状態遷移を検証 | 不要 |
| 電力 | `PowerLow` と `PowerPerformance` で同じモデルを回し、`PowerSample` (Apple は `powermetrics`) の差を記録する。回帰の監視用 | 実機 |
| 統合 | build tag `hwtest` + 環境変数 `HWNPU_TEST_VENDOR=intel|amd|apple`。Apple: 小さな fp16 の ML Program を厳格モードで。Intel: 小さな IR (matmul) の NATIVE / IR 両経路。AMD: IRON の `vector_scalar_mul` 相当 | セルフホストランナー |

`cmd/hwnpuinfo` は列挙結果、拡張バージョン、コンパイラ / FW バージョン、対応フォーマット、ライブラリの解決先を表示する。バグ報告のテンプレートに貼ってもらう前提で作る。

---

## 13. マイルストーン

| M | 内容 | 完了条件 |
|---|---|---|
| M0 | 骨格、`internal/dl`、`accelfs`、`Enumerate`、`hwnpuinfo` | Linux で Intel/AMD、macOS で ANE の存在を表示できる |
| M1 | Apple: Core ML の推論と `MLComputePlan` による厳格モード | 手元の Mac で公開 API の形を確定できる |
| M2 | Intel: NATIVE blob と OpenVINO IR の推論、対応レイヤー検証、ビルドログ | IR から直接動き、未対応の演算が報告される |
| M3 | 電力: `PowerPolicy`、`PowerSample`、コンパイルキャッシュ | 3 バックエンドで電力ポリシーの差が計測できる |
| M4 | AMD Linux: xclbin ロード、BO、ERT、`Run` | IRON サンプルが Go だけで動く |
| M5 | 軽量 LLM: `CompileSet`、重み共有、プリフィルとデコードの切り替え | 小さな LLM が NPU だけで動く |
| M6 | Windows: Intel 動作確認、AMD 案 A の検証と判断 | 判断記録を docs に残す |
| M7 | AMD ELF 流れ / UMQ、Qualcomm QNN、仮想化デバイスの検討 | 設計更新 |

---

## 14. 未確定事項と検証方法

1. ~~**Level Zero graph 拡張名の交渉**~~ 解決済み (16.2)。新しいドライバは "ZE_extension_graph"、古いドライバは "ZE_extension_graph_1_2" のような版付きの名前を返す。
2. ~~**OpenVINO ビルドフラグ文字列**~~ 解決済み (16.2)。`serializeIOInfo` を移植した。`--config` は利用者が `GraphSource.BuildFlags` で渡す。
3. **DEV_HEAP の大きさと必須性**: shim の `device.cpp` / `pcidev.cpp` の `on_first_open` 相当を読む。カーネル文書は「コンテキストごとに 64 MB のホスト常駐命令バッファ」と記述。
4. **ERT ペイロードの正確な並び**: shim のダンプ機能で採取して golden にする。`xrt_kernel.cpp` の `ERT_START_CU` / `ERT_START_NPU` の選択条件も確認する。
5. **`WAIT_CMD` と syncobj の使い分け**: 両方実装し、`syncobj_handle` が有効ならそちらを優先する。
6. **Windows の Intel ドライバ DLL 名と loader 同梱の有無**: 実機で `ZE_ENABLE_LOADER_DEBUG_TRACE=1`。
7. **Windows の `xrt_coreutil.dll` の C エクスポート**: `dumpbin /exports` と NPU での `xrtPLKernelOpen` 実験。
8. **Apple の検証 API の Go からの呼び方**: `MLComputePlan` は完了ハンドラ (ブロック) を取る。purego/objc で 1 個のブロックを使い回せるか確認する。
9. **Intel の電力計測**: メトリックストリーマーで NPU の電力が取れるか、取れなければ sysfs や RAPL で代替できるか。
10. **仮想化とクラウド**: クライアント NPU は SoC ごとに 1 つの PCIe デバイスで全コア共有のため、現状 VM やクラウドから使う口は無い。AMD の shim に `virtio` ディレクトリがあり、UAPI に物理ファンクション (PF) のデバイス種別があるので、VM 向けの準パススルーが準備中の可能性がある (推測)。対応が出たら `Device` の下に新しいバックエンドを足すだけで済むよう、バックエンドの境界を保つ。
11. **purego fakecgo の動作条件**: glibc 以外 (musl / Alpine) で `CGO_ENABLED=0` のまま `dlopen` できるか。できなければ README に制約として書く。

---

## 15. 参考資料

- Intel Level Zero NPU 拡張ヘッダ: https://github.com/intel/level-zero-npu-extensions (`ze_graph_ext.h`, `ze_command_queue_npu_ext.h`)
- Intel NPU UMD: https://github.com/intel/linux-npu-driver (`docs/overview.md`: ライブラリ名、パッケージ、権限、バージョン整合)
- Intel KMD UAPI: `include/uapi/drm/ivpu_accel.h`
- OpenVINO NPU プラグイン (IR 直列化、ビルドフラグ、DDI 利用例): `src/plugins/intel_npu/src/compiler_adapter/src/{model_serializer,ze_graph_ext_wrappers,driver_compiler_adapter}.cpp`
- AMD KMD UAPI: `include/uapi/drm/amdxdna_accel.h`、カーネル文書 `Documentation/accel/amdxdna/amdnpu.rst`
- AMD XRT shim: https://github.com/amd/xdna-driver (`src/shim/{hwctx,hwq,buffer,device,pcidev,pcidrv}.cpp`, `src/shim/kmq/`)
- XRT ヘッダ: `xrt/detail/ert.h` (コマンドパケット), `xrt/detail/xclbin.h` (AXLF), `xrt/xrt_kernel.h`, `xrt/xrt_hw_context.h`, `xrt/experimental/xrt_module.h`
- mlir-aie Windows ホストビルド: https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWin.md (`xrt_coreutil.dll` の出所)
- Ryzen AI ドキュメント (Windows の xrt-smi 配置): https://ryzenai.docs.amd.com/en/latest/xrt_smi.html
- purego: https://github.com/ebitengine/purego, https://pkg.go.dev/github.com/ebitengine/purego

---

## 16. 実装記録

### 16.1 M1 (Apple / Core ML) — 2026-10-27

実装したもの:

- 公開 API: `Enumerate` / `Open` / `OpenDevice` / `Device` / `Buffer` / `GraphRunner` / `Graph` / `Request` / `Fence`、`Placement` (`PlacementStrict` 既定、`PlacementPrefer`)、`PlacementError`、fp16 変換 (`EncodeF16` / `DecodeF16`)
- `internal/backend/coreml`: purego/objc による Core ML 呼び出し。`MLComputeUnitsCPUAndNeuralEngine` で GPU を除外し、`MLComputePlan` で配置を検証する
- `internal/mem`: Go ヒープ外のページ境界メモリ (mmap)。`MLMultiArray initWithDataPointer:` でゼロコピーで包む。出力は `MLPredictionOptions.outputBackings` に渡し、使われなかった場合だけコピーする
- `cmd/hwnpuinfo`: デバイス一覧と、`-model` 指定時の配置レポート
- `testdata/coreml` (現在は `internal/fixtures/coreml`): fixture と生成スクリプト (`gen_models.py`、coremltools 9.0 で生成)

検証結果 (macOS 27.0.1, Apple Neural Engine 16 cores, `CGO_ENABLED=0`):

| テスト | 結果 |
|---|---|
| fp16 モデルが全演算 ANE 配置で推論でき、参照値と一致 | 合格。最大絶対誤差 0.0014 |
| fp32 モデルが厳格モードで `ErrNotOnNPU`、原因として `linear` と `relu` が CPU と報告 | 合格 |
| `PlacementPrefer` では fp32 モデルが実行でき、配置レポートに CPU 演算が載る | 合格。誤差 0 |
| 小さすぎるバッファの Bind と未バインドの Submit を拒否 | 合格 |
| `Fence.Wait` が context の期限で抜ける | 合格 |
| `CGO_ENABLED=0` で linux/amd64, linux/arm64, windows/amd64, darwin/amd64, darwin/arm64 をビルド | 合格 |
| `go test -race -count=3` | 合格 |

実験で分かったこと:

- **purego/objc のブロックで `MLComputePlan` の完了ハンドラを受けられる。** `objc.NewBlock` が使え、`CGO_ENABLED=0` でも別スレッドからのコールバックが動いた
- **Core ML は小さなモデルを fp16 でも CPU に置く。** compute plan での観測は次のとおり。軽量モデルを ANE に載せるには、ある程度の計算量が要る。軽量モデル向けの設計 (5.6) では、バッチをまとめて大きくする工夫が必要になる可能性がある

| モデル (fp16、linear + relu) | 配置 |
|---|---|
| 入力 (1, 64) → 32 | CPU |
| 入力 (8, 256) → 256、(16, 128) → 128、(32, 256) → 256 | CPU |
| 入力 (64, 512) → 512、(256, 1024) → 1024 | Neural Engine |
| 1x1 conv (1, 64..256, 8..32, 8..32) | CPU |
| 同じ linear を fp32 で | CPU |

- **compute plan は推定。** 実行時に ANE が実際に使われたかは公開 API で確認できない。`PlacementReport.Method = "coreml-compute-plan"` で検証手段を明示している
- **出力はゼロコピーで書き込まれる。** `outputBackings` に渡した `MLMultiArray` に Core ML が直接書き込み、コピーへのフォールバックは fp16 と fp32 のどちらでも 0 回だった (`TestOutputBackingZeroCopy`)

### 16.2 M2 (Intel / Level Zero) — 2026-10-28

開発機に Intel NPU が無いため、実機での確認は未実施。代わりに次の三段で確かめた。

1. **構造体の配置**: 公式ヘッダ (oneapi-src/level-zero, intel/level-zero-npu-extensions) から C で求めたサイズとオフセット (`testdata/abi/levelzero_abi.json`) と Go の構造体を突き合わせる。macOS arm64 の clang と Linux amd64 の gcc で生成した値は同一だった
2. **偽のドライバ**: `testdata/zemock/zemock.c` は、hwnpu が依存する Level Zero と graph 拡張の振る舞いだけを真似る共有ライブラリ。テスト開始時に `cc` でビルドし、`HWNPU_ZE_LOADER` で読み込ませる
3. **Linux での実行**: Docker の golang:1.27 イメージ (linux/arm64 と linux/amd64) で `CGO_ENABLED=0 go test ./...` が合格

実装したもの:

- `internal/dl`: 共有ライブラリの読み込み (Unix は purego.Dlopen、Windows は syscall.LoadLibrary)
- `internal/ze`: Level Zero 本体と graph 拡張のバインディング。拡張は OpenVINO と同じ規則で名前と版を選び (目標 1.20、最低 1.5)、版に応じた数だけ関数表を読む
- `internal/ovir`: OpenVINO IR の読み取り、ドライバ内コンパイラ向けの直列化 (ALL_WEIGHTS_COPY 形式)、ビルドフラグの生成、`is_new_api` と `use_indices_for_io_metadata` の実行時情報の差し込み
- `internal/accelfs`: `/sys/class/accel` の走査 (`intel_vpu` / `amdxdna` の判定)
- `internal/backend/levelzero`: デバイス列挙、`zeMemAllocHost` のバッファ、グラフ作成、初期化 (`GraphInitialize` とコマンドリスト経由の両方)、実行要求、完了待ち
- 公開 API の追加: `ArgInfo.ModelDType` (モデル上の精度。Intel では fp32 モデルが fp16 で実行される場合などに `DType` と異なる)、`DeviceInfo.ID`、`BuildError` / `DriverError` / `VersionError`、`PlacementError.Cause`

Intel での配置検証の扱い:

| 状況 | 結果 |
|---|---|
| コンパイル済み blob | `Verified`、`Method = "levelzero-native-blob"`。blob は NPU 用のコードなので常に NPU 上 |
| IR のコンパイル成功 | `Verified`、`Method = "levelzero-compiler"`。NPU コンパイラは他デバイスに回さないので全演算が NPU 上 |
| IR のコンパイル失敗 | 対応レイヤーを問い合わせ、未対応の演算があれば配置方針にかかわらず `PlacementError` (`Device = "unsupported"`、`Cause` にビルドログ付き `BuildError`)。無ければ `BuildError` |

OpenVINO から移植した細部:

- 対応レイヤーの問い合わせ結果は `<名前0><名前1>...` 形式。問い合わせには `--config` 部分だけを渡す
- ビルドログはグラフのハンドルを NULL にして `pfnBuildLogGetString` で取る
- 引数の値はコマンドリストに積んだ時点で取り込まれるので、実行要求ごとに `SetArgumentValue` から積み直す (グラフ単位のロックで保護)

未確認 (実機が必要):

- 実際のドライバと NPU コンパイラで IR がコンパイルでき、結果が正しいこと
- 差し込んだ実行時情報とビルドフラグを実際のコンパイラが受け付けること
- Windows の `ze_loader.dll` 経由での動作

### 16.3 動作確認レポート (`cmd/report`) — 2026-10-28

実機での確認を利用者に頼めるよう、`go run ./cmd/report` で動く確認プログラムを追加した。

- テスト用モデルを `internal/fixtures` に移し、`embed` でプログラムに埋め込んだ。実行時に一時ディレクトリへ書き出す (Core ML が実ファイルを要するため)
- 同じ計算 (64x512 の入力に 512x512 の重みを掛け、バイアスを足して ReLU) を Core ML と OpenVINO IR の両方で用意し、ベンダー間で結果と速度を比べられるようにした。IR は OpenVINO 2026 の CPU 推論 (fp32 指定) で参照値との誤差 2.4e-6 を確認した。fp16 で計算した場合の誤差は 0.028 だったので、NPU の許容誤差は絶対値 0.05 と相対値 2% にした
- 確認項目: デバイスの列挙とオープン、厳格モードでのコンパイルと配置、推論結果の比較、コンパイル時間と初回・中央値・最小の実行時間、Core ML の fp32 モデルが厳格モードで拒否されること
- Level Zero loader の読み込み結果を必ず載せる。Windows では `/dev/accel` が無く、loader が無いと Intel NPU の有無も分からないため

### 16.4 M3 (電力ポリシー、電力量の計測、コンパイルキャッシュ) — 2026-10-28

AMD バックエンドは未実装なので、Intel (偽のドライバで検証) と Apple (実機で検証) が対象。

公開 API の追加:

- `WithPowerPolicy(PowerDefault | PowerLow | PowerBalanced | PowerPerformance)`、`Device.PowerPolicy()`
- `Device.PowerSample()` は積算の電力量 (ジュール) を返す。平均電力は `AveragePower(a, b)` で求める。`Scope` は `"npu"` か `"package"` (CPU パッケージ全体)
- `WithCacheDir(dir)`、`Graph.Cached()`

バックエンドごとの対応:

| | Intel (Level Zero) | Apple (Core ML) |
|---|---|---|
| PowerLow | キュー優先度 LOW、ワークロード種別 BACKGROUND (NPU キュー拡張) | 効果なし (`CapPowerControl` なし) |
| PowerBalanced | キュー優先度 NORMAL | 同上 |
| PowerPerformance | キュー優先度 HIGH、ターボ (NPU キュー拡張 1.0 は `turbo`、1.1 は `options`) | 同上 |
| 電力量 | Linux の RAPL (`/sys/class/powercap/intel-rapl:N`)。CPU パッケージ全体で、NPU 単体ではない。通常 root が必要 | 公開 API なし |
| キャッシュ | IR のコンパイル結果を `GetNativeBinary` で取り出して保存し、次回は blob として読み込む。キャッシュ使用時はドライバ側キャッシュを止める (`ZE_GRAPH_FLAG_DISABLE_CACHING`、OpenVINO と同じ) | `compileModelAtURL` の結果 (`.mlmodelc`) を保存する |
| キャッシュのキー | 直列化した IR、ビルドフラグ、ドライバ・graph 拡張・コンパイラの版、デバイス名と ID | モデルの全ファイルの内容、macOS の版 |

あわせて直した不具合: コマンドキューとコマンドリストを、グループ番号 0 ではなく計算用のグループ (`zeDeviceGetCommandQueueGroupProperties` で探す。OpenVINO と同じ規則) に作るようにした。偽のドライバは計算用グループを 1 番にし、0 番を使うとエラーにして検証している。

検証結果:

| テスト | 結果 |
|---|---|
| 電力ポリシーごとのキュー優先度・ターボ・ワークロード種別 (偽のドライバ) | 合格 |
| Intel のキャッシュ: 2 回目は IR をコンパイルせず blob を読み込む。ビルドフラグが変わると別の項目になる (偽のドライバ) | 合格 |
| Apple のキャッシュ: 2 回目はコンパイルを省略 (158 ms から 27 ms)。別のモデルは別の項目。キャッシュから読んだモデルの推論結果も正しい | 合格 |
| RAPL の読み取りとカウンタの巻き戻り (偽の sysfs) | 合格 |
| linux/arm64 と linux/amd64 のコンテナで `CGO_ENABLED=0 go test ./...` | 合格 |

`cmd/report` に `-power` と `-cache` を追加し、電力量を読めるデバイスでは 1 回あたりの電力量と平均電力を表に載せる。

未確認・未決:

- Intel 実機で、ポリシーの違いが実際に電力に表れるか (`go run ./cmd/report -power low` と `-power performance` を比べる)
- NPU 単体の電力を読む手段 (Intel、Windows、Apple)

### 16.5 M4 (AMD Ryzen AI / amdxdna) — 2026-10-28

開発機に Ryzen AI 機が無いため、実機での確認は未実施。Linux カーネル (drivers/accel/amdxdna、include/uapi/drm/amdxdna_accel.h) と XRT (xrt_kernel.cpp、xclbin_parser.cpp、ert.h、xclbin.h)、XRT shim (amd/xdna-driver src/shim) のソースから仕様を読み取り、次の三段で確かめた。

1. **ABI の基準値**: カーネルの UAPI ヘッダと XRT のヘッダを Linux amd64 の gcc でコンパイルし、ioctl 番号、構造体の配置、xclbin の構造体、ERT ヘッダのビット配置を `testdata/abi/xdna_abi.json` に出力。Go 側と突き合わせて全て一致
2. **ドライバの模擬**: `internal/xdna/xdnasim` はソースで確認した規則をそのまま検査する Go 製の偽ドライバ。ioctl の構造体を生のまま受け取るので、hwnpu 側の詰め方も検証される
3. **Linux での実行**: golang:1.27 の linux/arm64 と linux/amd64 で全テスト (レース検出込み) に合格

ソースから読み取った要点:

| 項目 | 内容 |
|---|---|
| デバイスヒープ | `BO_DEV_HEAP` は `AIE2_DEVM_SIZE` (64 MiB) の倍数。npu1 は拡張不可 (最大 64 MiB)、npu4 以降は 512 MiB まで。BO_DEV を作る前にマップが必要 (無いと `EAGAIN`)。BO_DEV の CPU アドレスはヒープのマップ先からの差で決まり、BO_DEV 自体はマップできない |
| デバイスアドレス | BO_DEV はヒープ内 (`0x4000000` 起点)。BO_SHARE は PASID 有効時はマップ後のユーザーアドレス。hwnpu はマップ後に `GET_BO_INFO` を取り直す |
| CU の設定 | `CONFIG_HWCTX(CONFIG_CU)` の `cu_bo` は BO_DEV (PDI を置く)。再設定は不可 |
| PDI の選択 | xclbin の EMBEDDED_METADATA の `extended-data dpu_kernel_id` を AIE_PARTITION の CDO グループの `dpu_kernel_ids` と照合 (XRT shim と同じ)。AIE_PARTITION 内のオフセットはセクション先頭からの相対 |
| コマンド | xclbin 経路の DPU カーネルは `ERT_START_CU`、`type = ERT_CU`、CU マスク 1 語のあとに、メタデータの `offset` / `size` どおりに引数を並べたレジスタマップ。バッファ引数は BO のデバイスアドレス |
| 投入 | `EXEC_CMD` は `cmd_count = 1` のとき `cmd_handles` に **ハンドル値そのもの** を入れる。`args` は参照する全 BO のハンドル配列 |
| 完了待ち | AIE2 は `cmd_wait` を実装していないので `WAIT_CMD` は使えない。ハードウェアコンテキストの syncobj に対し、`EXEC_CMD` が返した seq を点として `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` (`WAIT_FOR_SUBMIT`、CLOCK_MONOTONIC の絶対時刻) で待つ。結果は CMD BO の先頭語の状態欄で判定 |
| 電力 | `SET_STATE(SET_POWER_MODE)` は `DRM_ROOT_ONLY` でデバイス全体に効くので hwnpu は触らない。電力ポリシーはコンテキストごとの QoS priority (LOW 0x280 / NORMAL 0x200 / HIGH 0x180) に写像。既定 0 はドライバ内で HIGH 扱い |
| 電力計測 | `GET_INFO(QUERY_SENSORS)` の POWER センサー (mW, `unitm = -3`)。瞬間値なので、hwnpu は `PowerSample` の呼び出し間で台形積分して `EnergyJ` を作る (`Scope = "npu"`) |

あわせて直したもの (実機で問題になる種類):

- ioctl の `EAGAIN` を再試行していた。amdxdna は「ヒープ未準備」を `EAGAIN` で返すので無限ループになりうる。再試行は `EINTR` のみに
- 構造体内のポインタを整数で持っていた。ゴルーチンのスタック移動で古いアドレスがカーネルに渡りうるので `unsafe.Pointer` 型に

公開 API の追加:

- `Device.Kernel()` → `KernelRunner.Load(ProgramSource{Xclbin | Path})` → `Program.Kernel(name)` → `Kernel.NewRun()` → `Run.SetUint64 / SetUint32 / SetBuffer / SetBytes` (連鎖可、最初のエラーを保持) → `Run.Run(ctx)` / `Submit()`
- `AllocBuffer(size, WithBufferKind(BufferInstructions))`: 命令列用。AMD ではヒープ上の BO_DEV (XRT の cacheable に相当)
- `PowerSample.Watts`

`cmd/report` は AMD デバイスでバッファの往復、命令列用バッファ、電力センサーを確かめ、`-xclbin` と `-insts` を与えると IRON の passthrough 設計を実行して出力を照合する。

未確認・未決:

- 実機での動作全般 (Ryzen AI の Linux 機で `go run ./cmd/report -xclbin ... -insts ...`)
- 本物の xclbin での確認。mlir-aie の wheel で passthrough 設計をビルドしようとしたが、外部の大きな wheel の導入と実行がこの環境の権限で拒否されたため未実施。合成した xclbin は C ヘッダ由来の配置で検証済み
- 実際の XML の引数オフセット (合成した xclbin では 0x00, 0x08, 0x10, 0x14, ... と仮定。hwnpu はメタデータを読むので値には依存しない)
- ELF 経路 (`ERT_START_NPU*`) と UMQ 世代、ヒープの拡張

### 16.6 Apple: 小さなモデルのバッチパディング — 2026-10-28

Core ML は小さなモデルを fp16 でも CPU に置く。配置はモデル全体の計算量で決まり、単純な閾値では表せない (MODEL_ARCHITECTURE.md に測定表)。ANE に移すのはバッチ次元の拡大で、各行が独立なモデルならゼロ埋めで結果は変わらない。

- `tools/coreml/ane_pad.py`: ML Program を MIL に読み戻し (coremltools の milproto ローダー)、入力のバッチ次元を書き換えて型推論をやり直し、変換し直す。候補のバッチを小さい順に試して `MLComputePlan` で全演算が ANE に載る最小のものを採用し、メタデータ `hwnpu.logical_batch` / `hwnpu.padded_batch` を書く。`--verify` で元のモデルと出力を比べる。coremltools は開いたパッケージの Manifest.json を書き換えるので、入力は一時ディレクトリに複製してから開く
- Core ML バックエンド: メタデータ (`MLModelCreatorDefinedKey`) を読み、`ArgInfo.Dims` / `Bytes` は元のバッチ、`ArgInfo.PaddedDims` に実際の形状を入れる。実行要求ごとにパディング済みのバッファを持ち、入力の行を書き写し (余白はゼロのまま)、出力の行を書き戻す
- 検証: 3 層 MLP (幅 128、バッチ 1) は CPU 配置で厳格モードに拒否され、バッチ 64 にパディングしたものは全 6 演算が ANE、参照値との差 0.0002 (`TestCoreMLBatchPadding`)

測定 (Python から Core ML、中央値): 1 行を CPU で 73 µs、64 行にパディングして ANE で 280 µs、64 行を CPU で 104 µs。このモデルでは ANE の起動の固定費が計算より大きく、遅延は悪化する。電力は root が必要で未測定。パディングは明示的に選んだときだけ効く仕組みとし、埋めた行に実際の要求を載せるマイクロバッチ化と組み合わせるのが本筋。


### 16.7 Apple: 電力の計測と Neural Engine の損益分岐点 — 2026-10-28

- `cmd/powerbench` (darwin): sudo のパスワードを最初に一度だけ聞き、`sudo -n powermetrics -f plist` を計測ごとに起動する。各モデルを別スレッドで呼び続けたまま (`-rate` で一定頻度も可) CPU / GPU / ANE の電力を取り、アイドル時を差し引いた正味の電力と、呼び出し 1 回あたりのエネルギー (mJ/call) を出す。NPU に 1 演算でも載るモデルは、CPU だけの場合も測る
- CPU や GPU との比較用に、Core ML バックエンドは環境変数 `HWNPU_COREML_UNITS=cpu` (`MLComputeUnitsCPUOnly`) と `gpu` (`MLComputeUnitsCPUAndGPU`) で読み込む (計測専用で、公開 API にはしない)
- `tools/coreml/gen_sweep.py` (大きさを段階的に変えた合成エンコーダー) と `tools/coreml/convert_minilm.py` (all-MiniLM-L6-v2 を系列長固定で変換し、PyTorch と照合) を追加
- 結果 (M3、詳細は MODEL_ARCHITECTURE.ja.md): 損益分岐点は呼び出し 1 回あたり約 200 MFLOP。それより大きいと ANE は 2〜3 倍速く、エネルギーは約 4 分の 1 (約 1 TFLOP/J 対 CPU 0.25 TFLOP/J)。それより小さいと Core ML の呼び出しの固定費 (約 55 µs) と ANE の起動の費用が勝ち、CPU が有利。16.6 のゼロ埋めパディングは、このモデルの大きさでは得にならなかった

### 16.8 M5: Go だけの LLM 変換 (LFM3-1.7B → Core ML) — 2026-10-28

- 方針: モデルは Hugging Face の公式 bf16 から配布前に一度だけ各 NPU の形式へ変換し、S3/NFS などで配る。中間形式 (ONNX) は置かず、`config.json` と safetensors/GGUF から Go で直接組み立てる。量子化済みの公開物は劣化なしに並べ替えられるときだけ使い、量子化し直しはしない (決定の詳細は `.knowledge/concepts/decision/model-source-selection.yaml`)
- 実装: `internal/mlpackage` (MIL protobuf、MILBlob 形式の weight.bin、Manifest.json)、`internal/milspec`/`internal/mlspec` (coremltools 9.0 の .proto から protoc-gen-go で生成。`google.golang.org/protobuf` を依存に追加)、`internal/safetensors`、`internal/gguf` (Q4_0/Q8_0/Q6_K の逆量子化)、`internal/palette` (4 ビット LUT の当てはめ)、`internal/llm/lfm3` (構造の定義)、`cmd/hwnpu-convert`
- 検証の流れ: `tools/lfm3/ref_logits.py` (現在は `tools/ref/ref_logits.py`) で PyTorch fp32 の各層の隠れ状態とロジットを保存し、`-layers N` で切り詰めたモデルを層ごとに照合する (`TestAgainstReference`、`TestPerLayer`)。層の中は `TestConvPieces` で演算ごとに照合し、ANE の数値特性は `TestActivationRange`、配置は `TestPlacementVariants`/`TestOpVariant` で調べた
- 結果 (M3): bf16 → fp16 のグラフは CPU で KL 0.020 (argmax 21/24) と正しい。lut4 (出力 16 行ごとの表、`kmeans-minmax`) は 24 層 1627 演算がすべて ANE に載り、KL 0.89。損失のほぼ全部が 4 ビット量子化によるもので、ANE の演算誤差は `-post-scale` で層ごと 1.4〜2.7% に抑えた
- ANE について分かったこと (MODEL_ARCHITECTURE.ja.md にも記載): 行列積の出力に約 2.5e-4 の絶対誤差の床がある。LUT は 16 行ごとの表だけが ANE で動き、列方向のブロック化や行ごとの表、入力ブロックごとのスケールを持つ int4 (Q4_0 相当) は使えない。gather は常に CPU。値の名前にドットがあるとローダが落ちる
- 未着手: KV キャッシュと畳み込み状態 (Core ML の MLState) による生成、埋め込み表の配布形式、Llama/Qwen 系の構造、Intel IR と AMD 向けの書き出し、精度改善 (活性を見た量子化、6 ビット表)

### 16.9 対応モデルのホワイトリストと共通ブロック — 2026-10-29

- 方針: llama.cpp と同じく、対応するアーキテクチャをホワイトリストで管理する。各アーキテクチャの構造は、Hugging Face の `modeling_*.py` の forward を手で Go に移したもので、nn.Module 1 つを Go のメソッド 1 つにする。ONNX などからの汎用の取り込みは置かない (ANE 向けの書き換え、たとえば `-post-scale` や畳み込みのタップ分解を入れにくくなるため)。ホワイトリスト外のモデルは、これまで通り `.mlpackage` / OpenVINO IR を持ち込む
- `internal/llm`: アーキテクチャに依らない部分。`Weights` (safetensors / GGUF を Hugging Face のテンソル名で読む)、`Options` (共通の設定と、アーキテクチャ固有の `Params`)、`Arch` (名前、`config.json` の `model_type` / `architectures`、GGUF の名前の対応表、固有の設定) と登録の仕組み (`Register`、`Detect`、`LoadConfig`)
- `internal/llm/blocks`: 複数のアーキテクチャで使う層 (重み形式を選べる linear、RMSNorm、SwiGLU、RoPE の表と回転、因果マスク、GQA の注意)。アーキテクチャのパッケージは `blocks.Graph` を埋め込み、固有の層 (LFM3 の短い畳み込みなど) だけを書く
- `internal/llm/archs`: すべてのアーキテクチャを import するだけのパッケージ。この import の一覧がホワイトリスト。`hwnpu-convert` はここから登録を読み、`-archs` で一覧を出す。アーキテクチャは `config.json` から判別し、`-arch` で指定もできる。固有の設定 (`-post-scale`、`-conv`) は登録された `Param` からフラグを作る
- 検証: `TestBuildTiny` は、小さな合成の LFM3 を 5 通りの設定で組み立てる。Neural Engine なしで動き、プログラムと weight.bin のダイジェストをログに出す。今回の切り出しでは、前後でダイジェストがバイト単位で一致することを確かめた。`TestArch` は、`hwnpu-convert` と同じ経路 (判別とパラメータ) で組み立てた結果が直接の `Build` と同じことを確かめる
- `Model.Tensors()` は変換が読むテンソルの名前と shape の一覧。`llm.NewSynthetic` はそこから決定的な乱数の重みを作り (実物の重みなしのテスト用)、`llm.Inventory` はチェックポイントと突き合わせて、足りないテンソル、shape の違い、読まれないテンソルを出す。読まれないテンソルは、構造の移し漏れの手がかりになる
- `llm.Options.NoHead` は、最後の層だけを切り出したときも最終 norm の前の隠れ状態を出力させる。`Start` を層の数にすると、最終 norm と出力層だけを組み立てる

### 16.10 Mac なしの検証: CPU インタプリタと hwnpu-check — 2026-10-29

- `internal/milcpu`: `mlpackage.Builder` が出す演算 (linear、matmul、要素ごとの演算、reduce_mean、rsqrt、softmax、reshape、transpose、tile、slice_by_index、pad、concat、1 次元の conv、constexpr_lut_to_dense、constexpr_blockwise_shift_scale) を CPU で fp32 のまま実行する。目的は「グラフが PyTorch と同じ計算をしているか」の確認で、速さは二の次 (行列積だけ並列。4 コアで約 12 GFLOP/s)。`RoundF16` で演算ごとの結果を fp16 に丸め、Neural Engine の保存精度を近似する (演算の精度ではない)。各演算は手で計算した値と照合し、小さな LFM3 では同じ関数を表す別のグラフ (畳み込みを taps と conv で、PostScale の有無、層ごとの実行とまとめた実行) が一致することを確かめた
- `cmd/hwnpu-check`: 参照 (`tools/ref/ref_logits.py`) に対して、埋め込み、各層 (参照の入力を与えて単独で)、最終 norm と出力層、モデル全体の順に照合する。層の相対 RMS 誤差が `-tol` (既定 0.01) を超えると終了コード 1 で、最初の層を示す。既定の形式は fp16 (構造の確認)。`-package` は既存のパッケージのロジットを照合する。テストでは、1 つの層の重みだけを変えた参照に対して、その層だけが検出されることを確かめた
- `tools/ref/ref_logits.py`: `tools/lfm3` から移して汎用にした。隠れ状態は `output_hidden_states` ではなくデコーダ層への hook で取る (多くの実装で `output_hidden_states` の最後は最終 norm の後のため)。`REF.json` にモデルの種類と大きさを書く

### 16.11 2 つ目のアーキテクチャ: BERT 系エンコーダと移植スキル — 2026-10-29

- `internal/llm/bert`: transformers の `modeling_bert.py` を移した。埋め込み (単語、位置、token type の和と LayerNorm)、post-norm のエンコーダ層 (双方向の注意と GELU の MLP)。出力は最後の隠れ状態で、プーリングと正規化はホストに任せる。位置と token type 0 の埋め込みは、系列長が固定なので 1 つの定数にまとめた。2 つ目の入力 `mask` から加算用のバイアス (パディングに -10000、transformers と同じ) を作る。タスク用のチェックポイント (`BertForMaskedLM` など) の `bert.` 接頭辞は `architectures` から判断する。相対位置埋め込みと、GELU、ReLU 以外の活性化は拒否する
- エンコーダを載せるために `llm.Model` を広げた: `Output()` (出力の幅とロジットかどうか)、`Embedding()` (埋め込み表の名前)、`Build` は入力の一覧を返す。`Start == 0` のプログラムは、表引きと層 0 の間の処理 (埋め込み段) も含む。参照の `hidden[0]` は、どのモデルでも埋め込み表を引いただけの値 (`tools/ref` は `get_input_embeddings()` への hook で取る) なので、層 0 の照合には埋め込み段も含まれる
- `blocks` に `SDPA` (マスクを外から渡す注意。`Attention` はこれに因果マスクを渡す)、`LinearBias`、`LayerNorm`、`Activation` を追加し、`milcpu` に `gelu` と `layer_norm` を追加した。LFM3 のダイジェストは変わっていない
- 検証: `bert_test.go` に、`modeling_bert.py` をループで直接書いた fp64 の forward を正解として置き、パディングの有無、`gelu_new`、`bert.` 接頭辞の 4 通りで相対誤差 1e-4 以内に一致することを確かめた。マスクを外す、残差の足し先を変える、という 2 つの改変をテストが検出することも確かめた。実物のチェックポイント (all-MiniLM-L6-v2 など) での `hwnpu-check` と、Neural Engine での配置はまだ確かめていない
- `.claude/skills/port-architecture/SKILL.md`: LFM3 と BERT の経験から、移植の手順、合格条件、間違えやすい点をまとめた。`tools/ref/ref_logits.py` に `--encoder` (AutoModel で読み、最後の隠れ状態を出力として書く) を追加した
- 次: トークナイザー (byte-level BPE、WordPiece) と、実物のチェックポイントでの照合

### 16.12 Apple: 1 GiB の上限を層ごとの分割で超える (実測と設計) — 2026-10-29

**実測** (M3 16GB、macOS 27.0.1、`npuprobe split`、lut4 64 層 1.81 GiB。レポートは `npu_profiles/npuprobe-m3-split.md`)

| 分割 | 各部分 | 配置 | ms/token | 各部分の単独の合計 | 重み GB/s | GPU との誤差 (各部分のあと) |
|---|---|---|---|---|---|---|
| 2 | 32 層 0.91 GiB | すべて ANE | 30.0 | 30.5 | 65 | 3.32%、4.91% |
| 4 | 16 層 0.45 GiB | すべて ANE | 30.2 | 30.4 | 64 | 2.31%、3.32%、4.21%、4.91% |

- 1 つのモデルにすると CPU に落ちる 1.81 GiB が、層で分ければ全部分 ANE に載り、16 層単体 (7.5 ms、65 GB/s) と同じ帯域で動く。上限はモデルごとで、全部分を同時に読み込んだままでよい
- 遅延: 全部分を続けて動かす時間は各部分の単独時間の合計以下。部分を切り替えても重みを読み直していない。境目 1 つの費用は 0.1 ms 程度 (4 分割は 2 分割より 0.2 ms 遅いだけ)。前の部分の出力バッファを次の部分の入力に `Bind` するだけで、コピーもない
- 劣化: 32 層目と 64 層目のあとの誤差が 2 分割と 4 分割で全桁一致した。境目は何も失わず、誤差は ANE の演算誤差が深さとともに積もったもの
- コンパイル: 部分が小さいほど合計が短い (16 層は 1 つ 8 秒、32 層は 1 つ 16〜30 秒)

これで 1 GiB は「モデルの大きさの上限」ではなく「1 パッケージの大きさの上限」になる。lut4 で 3〜8B 級のモデル (8B は約 4 GiB で 5 部分、1 トークン約 60 ms の見込み。未計測)、6 ビット LUT (LFM3-1.7B で約 1.2 GiB)、感度の高いテンソルを fp16 に残す構成が Mac の ANE で扱えるようになる。

**M4** (利用者の計測、2026-10-29。3.6 GiB を 2/4/8/16 分割): 58.3 / 58.8 / 59.4 / 60.6 ms/token。どの分け方でも誤差はなく、分割が細かいほど消費メモリが大きく減った (指標と数値は未記録)。境目 1 つの費用は約 0.16 ms で、16 分割でも 2 分割より 4% 遅いだけ。1.8 GiB の部分が動いたので、M4 の上限は 1 モデルあたり 1.8 GiB 以上。

分割の細かさは、メモリを減らす代わりに境目 1 つあたり 0.1〜0.16 ms を払う取引で、速さの差は小さい。機械のメモリに合わせて分割数を変えたいので、分割は配布物ではなく実行時に決める。

**設計 (変換時に分ける段階だけ実装。16.13)**: 決定の記録は `.knowledge/concepts/decision/coreml-layer-split.yaml`。

- 配布物: 分けない 1 つの `.mlpackage` (1 GiB を超えてよい)。チップの上限やメモリの量によらず 1 種類で、manifest と sha256 も 1 つ。`hwnpu-convert` は今まで通り 1 つのパッケージを書き、1 GiB を超えるときは「そのままでは ANE で動かないので分割して使う」と表示する。変換器もアーキテクチャのパッケージも変えない
- 分割器 (`internal/mlpackage`): `model.mlmodel` を読み、main 関数の演算列 (トポロジカル順) で、前後をまたぐ値が 1 つ (と関数の入力) だけになる位置を境目の候補にする。定数 (`const`、`constexpr_*`) はまたぐ値に数えず、使う部分に複製する (RoPE の表や因果マスクのような共有の小さな定数は各部分に入る)。LLM では各層の後ろが候補になる。候補ごとの重みのバイト数は blob の範囲から正確に数える。`Split(src, cuts, dst)` は部分ごとに `.mlpackage` を書き、`weight.bin` は使う blob をそのまま写す (量子化し直さないので重みはビット単位で同じ)。部分の入力は `x`、`mask` などの関数の入力は使う部分で宣言し直し、出力は `y`。重みはメモリに丸ごと読まず blob 単位で流す。層の境目を優先させたいときだけ、`Build` が層の出力の値の名前をメタデータ (`com.github.shibukawa.hwnpu.cuts`) に残す (演算は足さないのでダイジェストは変わらない)
- 分割数の決め方 (Policy): `Parts(n)` は n 個に、`MaxPartBytes(b)` は 1 部分 b 以下で最少の数に、`MemoryBudget(b)` は消費メモリの見込みが b 以下になる最少の数にする。どれも 1 部分あたりのチップの上限を超えない (既定は 0.9 GiB。M4 などは実測が揃った機種から上げる)。部分の数を決めたうえで、最大の部分が最小になるよう候補の中から境目を選ぶ。`MemoryBudget` の見込みの式は計測から作る (未確定)
- キャッシュ: 部分ごとのコンパイル済み `.mlmodelc` を既存のコンパイルキャッシュ (`WithCacheDir`) に置く。キーは元パッケージのダイジェストと部分の範囲。同じ分け方なら 2 回目からは切り分けもコンパイルもしない。分割数を変えると最初の 1 回だけ切り分けとコンパイルが要る (M3 で 16 層 1 部分 8 秒)。分け方ごとに増えるので、使っていないものは古い順に消せるようにする
- 実行 (新しい公開パッケージ、仮に `github.com/shibukawa/hwnpu/chain`): `Open(ctx, runner, path, policy)` が切り分け、コンパイル、全部分の読み込みを行い、部分 i の出力バッファを部分 i+1 の `x` に、追加の入力は 1 つのバッファを全部分に `Bind` する (`npuprobe split` と同じで、コピーなし)。`Chain` は 1 つのモデルのように `x`、追加の入力、`y` を見せ、`Run` は部分を順に動かす。`Parts()` が今の分け方と各部分の配置を返す。どの部分も厳格モードで全演算 ANE を確かめる
- 動いている途中の変更: `Chain.Reshape(ctx, policy)`。メモリを減らす向きでは古い部分を閉じてから新しい部分を読み込む (その間は止まる)。増やす (速くする) 向きでは新しい部分を読み込み終えてから入れ替える。キャッシュにない分け方は、別の goroutine で切り分けとコンパイルをしてから入れ替える。最初の版は生成の要求と要求の間だけで行い、KV キャッシュを持ったままの入れ替えは扱わない
- 確認: 分割器と鎖は `milcpu` でも動くので Mac なしで確かめられる。`hwnpu-check` に `-parts n` を足し、切り分けた部分を順に実行して、まとめた実行とビット単位で一致すること、境目の隠れ状態が参照の `hidden` と合うことを見る
- 計測: `npuprobe split` に消費メモリの列を足す (子プロセスの最大 RSS、全部分を読み込んだあとと実行中の `phys_footprint`、コンパイル中の ANECompilerService の量)。M3 と M4 で分割数ごとに測り、`MemoryBudget` の見込みの式を作る
- 移植の約束事: 層の境目で受け渡すのは残差の流れそのもの (参照と同じスケール) でなければならない。ANE 向けの書き換えは層の中で閉じていること (LFM3 の `-post-scale` は直後のサンドイッチ正規化の中で戻すので満たす)。閉じていないと境目の候補が見つからないだけで、黙って壊れることはない。port-architecture スキルに書いた
- 生成への伸ばし方: KV キャッシュと畳み込みの状態は層ごとなので、その層を持つ部分が自分の状態を持つ。prefill と decode の 2 つの形と重み共有も部分ごとに作る
- 採らない案: 分割数ごとに配布物を作る (容量が増え、上限が分かるたびに作り直しになる)。変換時の固定の分割で配る (機械ごとに変えられない)。細かい部分で配って実行時につなぐ (切り分けと同じ書き換えで、1 つで配るほうが単純)。重みを入力として渡す。一部の部分だけを読み込み、使うたびに読み込み直す (1 トークンごとに読み込みが入る)。GPU に回す (非目標)

段階: (1) `npuprobe split` のメモリの計測 → (2) 分割器と `milcpu` での一致の確認、`hwnpu-check -parts` → (3) `chain` パッケージの `Parts` と `MaxPartBytes`、キャッシュ → (4) `MemoryBudget` と `Reshape`。

合格条件: (1) 小さな合成の LFM3 と BERT を fp16 と lut4 で書き出し、1〜層の数の部分に切り分けて `milcpu` で順に実行した結果がまとめた実行とビット単位で一致し、部分の blob は元と同じバイト列。(2) 各層の後ろがすべて境目の候補になり、2 つの値がまたぐ位置は候補にならない。(3) `Parts` と `MaxPartBytes` が上限を守り、部分の数が最少で、最大の部分が最小になる。(4) `TestBuildTiny` のダイジェストが変わらない。(5) Mac で、1 GiB を超える実物のモデルを 2/4/8 分割で開いて全部分が ANE に載り、分けないとき (`milcpu`) と同じ精度になる。`Reshape` しても出力が変わらない。(6) `npuprobe split` が分割数ごとの消費メモリを出し、M3 と M4 で記録される。

未確定: 消費メモリが分割数でどう変わるか (常駐か、コンパイル時の山か) と `MemoryBudget` の式。M4 以降の部分の上限の既定 (sysctl でチップを判別するか、初回に探索するか)。Core ML の状態 (MLState) の KV キャッシュが上限やメモリの見込みに数えられるか。生成の途中での `Reshape` (層ごとの状態を新しい部分へ移す)。出力層だけで上限を超えるモデル (語彙 128k × 隠れ 4096 の fp16 は 1 GiB) は語彙方向に分ける必要がある (それまでは計画のエラーにする)。Intel NPU にも同じ上限があるか。

### 16.13 分割の最初の段階: パーツごとのパッケージ、トークナイザー、生成 — 2026-10-29

**multifunction では上限を超えられない** (M3、`npuprobe split -layout functions,files`、lut4 64 層 1.81 GiB。レポートは `npu_profiles/npuprobe-m3-functions.md`)。Core ML の ML Program は関数を複数持て、`MLModelConfiguration.functionName` で 1 つずつ別のモデルとして読み込める (macOS 15)。関数は weight.bin を共有するので、1 つのファイルのまま分けられるかを試した。

| 分け方 | 配置 | ms/token | プロセスの最大 RSS | GPU との誤差 (最後) |
|---|---|---|---|---|
| 関数 2 つ (32 層ずつ) | 全部 CPU | 12549 | 5.26 GiB | 10.9% |
| 別パッケージ 2 つ | 全部 ANE | 30.6 | 1.03 GiB | 4.91% |
| 関数 4 つ (16 層ずつ) | 全部 CPU | 12362 | 6.42 GiB | — |
| 別パッケージ 4 つ | 全部 ANE | 31.3 | 0.55 GiB | 4.91% |

1 GiB の上限は読み込む関数ではなくパッケージ全体で数えられる。分けるならパーツごとに別のパッケージにする。別パッケージでは、プロセスの最大 RSS がおよそ 1 部分の重みの大きさになる (システム側の Core ML と ANE のプロセスは含まない)。速さは 2 分割と 4 分割でほぼ同じ。`GraphSource.Function` (関数の選択) と `mlpackage.FinishFunctions` は、prefill と decode のように関数を分ける用途のために残した。

**実装** (16.12 の設計のうち、変換時に分ける段階。実行時の切り分けは未実装)

- 変換時の分割は、main に並行して入った `hwnpu-convert -target` (チップごとの上限、推定サイズで均等に分ける `llm.PlanSplit`、`out.partN.mlpackage` と `out.parts.json`) に合わせた。こちらで書いた層の数で等分する `-parts` とディレクトリ形式は、マージのときに捨てた。`out.parts.json` の型 (`llm.Manifest`) と 1 部分の書き出し (`llm.WritePart`) を `internal/llm` に移し、`chain` とテストからも使う
- `internal/llm/chain`: `out.parts.json` か 1 つのパッケージを開き、全部分を読み込んで、部分 i の出力バッファを部分 i+1 の `x` に、ほかの入力は 1 つのバッファを全部分に `Bind` する。`Run` は部分を順に動かす
- `internal/tokenizer`: `tokenizer.json` の byte-level BPE。追加トークンを先に切り出し、GPT-4 型の前処理の正規表現 (Go の regexp にない先読み `\s+(?!\S)` を含む) を手で書いた走査で分け、マージの順位で結合する。対応しない形式 (WordPiece、byte fallback、別の前処理) は `Load` でエラーにする。LFM3 では、Hugging Face が参照のプロンプトに出した 24 トークンと一致した
- `cmd/hwnpu-generate`: チャットの書式 (`<|startoftext|><|im_start|>user ... <|im_end|>\n<|im_start|>assistant\n`) でプロンプトを作り、埋め込みはチェックポイントからホストで引き、貪欲法かサンプリング (温度、top-k、繰り返しの罰則) で生成する。KV キャッシュはなく、毎トークン固定長の窓全体を動かして最後の位置のロジットを読む。モデルは因果的なので、まだ使っていない位置のゼロは結果を変えない。窓が満ちたら、最初のトークンを残して古い 4 分の 1 を捨てる
- テスト: `TestPartsMatchWhole` (小さな LFM3 を fp16 と lut4 で 1〜3 分割し、`milcpu` で順に動かした結果が 1 部分とビット単位で一致し、1 部分はまとめたプログラムと一致する)。トークナイザーは前処理の分け方、BPE の結合、往復、LFM3 の実物 (`HWNPU_LFM3_DIR`)
- `npuprobe split` に `-layout` (files / functions) と、プロセスの最大 RSS の列を足した

**LFM3-1.7B** (M3 16GB、lut4 `kmeans-minmax`、0.81 GiB、seq 128、`hwnpu-generate`、64 トークン、貪欲法)

| 分割 | 配置 | ms/token | プロセスの最大 RSS | 読み込み (初回のコンパイル / キャッシュから) |
|---|---|---|---|---|
| 1 | 全部 ANE | 34.2 | 1.01 GiB | 29.5 s / 0.8 s |
| 2 | 全部 ANE | 34.2 | 0.93 GiB | 15.9 s / 1.1 s |
| 4 | 全部 ANE | 36.7 | 0.72 GiB | 27.2 s / 4.0 s |
| 8 | 全部 ANE | 33.6 | 0.84 GiB | 27.9 s / 23.7 s |

- 分割による速さの差は誤差の範囲。RSS にはモデル以外 (トークナイザー、ロジットのバッファなど) の分が入るので、分割数に比例しては減らない。8 分割だけキャッシュからの読み込みが遅い理由は調べていない
- 日本語の質問に、まず英語で考え (`<think>`) てから答える。128 トークンの窓は考える途中で満ちるので、文書を書かせるには窓が足りない
- 未着手 (当時): KV キャッシュ (16.14 で実装)、実行時の切り分け (16.12)、`hwnpu-check` の部分のディレクトリへの対応、埋め込み表の配布形式

### 16.14 KV キャッシュ: ホストのバッファを入出力に渡す — 2026-10-29

- 方式: キャッシュはホストが持ち、プログラムの入出力として渡す (MLState は使わない)。`Bind` したバッファはコピーなしで Core ML に渡るので、キャッシュはプログラムの入力バッファそのものに置き、各ステップの後で新しい分を書き込む。名前で役割を決める (`llm.Cached`):
  - `kv_NAME` (1, kv_heads, cache, head_dim) と出力 `kv_NAME_new` (1, kv_heads, Q, head_dim): ホストがステップのトークンの位置に書き込む。RoPE を掛けた後の K を持つ
  - `win_NAME` (1, W, 幅) と出力 `win_NAME_new` (1, W+Q, 幅): LFM3 の短い畳み込みの、直前 6 行の B*x。プログラムはゼロの左詰めの代わりにこれを前に付ける。ホストは最後の本物のトークンで終わる W 行を残す
  - それ以外の入力はモデルが位置から作る (`StepInputs`): 注意の型ごとのマスク (1, 1, Q, cache+Q) と RoPE の表 (1, Q, 1, head_dim)。gather (CPU 行き) を使わずに位置を渡せる。スライディング注意の窓はマスクで表す
  - 1 ステップは最大 Q トークン。足りない分はゼロで埋め、そこの K/V は書き込まない。prefill は同じプログラムを Q トークンずつ回す
- 実装: `lfm3.BuildCached` (`Options.Cache`)、`internal/llm/decode` (`Session`: x と `StepInputs` を書き、実行し、K/V と窓を書き戻す。バッファはただの fp16 のバイト列なので、Core ML の鎖とテストの CPU インタプリタで同じコードが動く)、`chain.Program`、`hwnpu-convert -cache N` (-seq は 1 回のトークン数。`out.parts.json` に cache を書き、各部分のキャッシュの出力を `state_outputs` に並べる)、`hwnpu-generate` のキャッシュ付きの経路。部分ごとに自分の層のキャッシュを持つので、分割と組み合わさる。`blocks.SDPA` はキーの長さが問い合わせと違ってよいようにし、`blocks.RopeValues` を切り出した (キャッシュなしのダイジェストは変わらない)
- 検証: `TestCachedMatchesFull` は小さな LFM3 (スライディング窓 3、畳み込み幅 3) の 8 トークンを、Q=1/3/8、キャッシュ 8〜10、1〜3 分割で 1 ステップずつ動かし、各位置のロジットがキャッシュなしのプログラムと相対誤差 6e-4 以内 (キャッシュの fp16 の分)。スライディング窓を外す、K/V の書き込み位置をずらす、畳み込みの窓を取り違える、RoPE の位置をずらす、の 4 つの改変を検出することを確かめた
- 結果 (M3、LFM3-1.7B lut4 kmeans-minmax、2 分割、全部 ANE、貪欲法):

| キャッシュ | Q | 生成 ms/token | プロンプト 23 トークン | プロセスの最大 RSS |
|---|---|---|---|---|
| なし (窓 128 を毎回) | 128 | 34.2 | — | 0.93 GiB |
| 256 | 1 | 16.8 | 402 ms | 1.01 GiB |
| 1024 | 1 | 20.6 | 551 ms | 1.06 GiB |
| 4096 | 1 | 37.6 | 1010 ms | 1.07 GiB |
| 1024 | 8 | 23.3 | 577 ms | 0.70 GiB |

- 出力はキャッシュなしの貪欲法と一字一句同じ書き出しになった。キャッシュ 4096 で、見出し付きの日本語の社内文書を 1500 トークン、26.7 tokens/s で書いた
- 遅さはキャッシュの使用量ではなく容量に比例する (形が固定なので毎回全体を計算する)。1000 トークン分の容量につき約 5.4 ms。4096 で増えた分は KV の読み出しに直すと 4.6 GB/s しかなく、Core ML が入力の配列を毎回コピーしている可能性がある (推定、未確認)
- 読み込み: キャッシュ付きのモデルは、コンパイル済みのキャッシュがあっても 2 回目以降の読み込みに 15〜22 秒かかる (キャッシュなしは 1 秒前後)。原因は調べていない (ANE 向けのコンパイルが毎回やり直されている可能性)
- マージ後の変換器 (`-target any -parts 2`、重みで均等に分けて 0.40 / 0.41 GiB) でも、キャッシュ 1024 で全部 ANE、22.7 ms/token
- 次: MLState との比較 (コピーがなくなるか、状態の演算が ANE に載るか)。容量の違うプログラムを用意し、文脈が伸びたら大きいものへキャッシュを移す。prefill 用の Q の大きいプログラム (長いプロンプトで効くかを測る)

### 16.15 KV キャッシュ: ホスト側バッファと MLState の比較 — 2026-10-29

- 実装: `mlpackage.Builder.State` (状態の入力) と `UpdateState` (write_state の後に read_state で読み直す。coremltools の coreml_update_state と同じ形)、`Package.States`。Core ML バックエンドは状態のあるモデルの要求ごとに `newState` で MLState を作り、`predictionFromFeatures:usingState:options:error:` で動かす。状態は引数 (`Args`) には出ない。`Request.ResetState` は `getMultiArrayForStateNamed:handler:` で全状態を 0 にする (新しい要求も 0 から始める。Core ML は新しい状態の中身を約束しないため)。`TestState` で、状態が呼び出しの間で引き継がれ、リセットで 0 に戻ることを確かめた
- `npuprobe kv`: bench と同じ大きさの層 (lut4) を 11 層 (LFM3-1.7B の注意層の数) にし、本物の注意を入れて 1 トークンを通す。`host` はキャッシュを入力に取り新しい K/V を出力する (LFM3 の方式)、`state` はキャッシュを層ごとの MLState に置き、one-hot の列 (1, 1, cache, 1) で `cache + onehot*(k - cache)` と書き込む (要素ごとの演算だけ。`slice_update` は CPU に置かれる)。レポートは `npu_profiles/npuprobe-m3-kv.md`
- 結果 (M3、全部 ANE):

| 容量 | KV | host ms (CPU use) | state ms (CPU use) |
|---|---|---|---|
| 256 | 5.5 MiB | 7.1 (0.16) | 6.6 (0.06) |
| 1024 | 22 MiB | 11.0 (0.18) | 10.0 (0.04) |
| 4096 | 88 MiB | 27.0 (0.16) | 26.8 (0.02) |
| 8192 | 176 MiB | 49.5 (0.15) | 49.3 (0.02) |

- 速さは変わらない。MLState で下がるのは CPU use だけで、ホスト側バッファでは Core ML が CPU で入力を扱っている分 (0.15 コアほど) がなくなる。容量に比例する費用は両方に残るので、原因は入力のコピーではなく、固定の長さのキャッシュ全体に対する ANE の計算 (連結または更新、GQA の tile、注意)
- ANE の制約: 状態に書いた値をそのまま後ろで使うと、ANE 向けのコンパイルが `error code: -14` で失敗する (CPU と GPU なら読み込める)。書いた後に読み直した値を使えば載る。GQA で K/V を tile で複製せず、問い合わせを (kv ヘッド, グループ) の行列にしてまとめて読む形 (同じ計算) は、host では全演算 CPU (8192 で 85.7 ms)、state ではコンパイル失敗だった
- 費用の正体 (`npuprobe kv -kv-modes host,host-mha`、レポート `npu_profiles/npuprobe-m3-kv-mha.md`): K/V を `tile` で複製せず 16 の問い合わせヘッドがそれぞれ K/V を持つ形 (キャッシュ 4 倍) は、1024 / 4096 / 8192 で 14.2 / 36.6 / 63.5 ms と host の 1.3 倍にしかならない。帯域でも複製でもなく、容量全体に対する問い合わせ 1 行の注意の計算 (層ごとに 16 ヘッド × 容量、1K トークンで約 8 GFLOP/s) が ANE を使い切れていない。手法の調査と当てはめは `docs/kv-cache.md`
- int8 のキャッシュ (`npuprobe kv -kv-modes host-int8`、レポート `npu_profiles/npuprobe-m3-kv-int8.md`): キャッシュを int8 の入力 (Core ML の仕様の版 10、macOS 26 以降) とトークンごとの fp16 のスケールで渡し、`cast` と `mul` で戻す形は全演算 ANE に載る。メモリは半分で、速さは同じ実行の fp16 より 1 割遅い (1024/4096/8192 で 11.9/29.5/50.8 対 10.8/27.6/46.5 ms)。量子化はメモリのための手法で、ANE の速さには効かない。`mlpackage.Int8` の入力の型を足した
- 判断: LFM3 はホスト側バッファのままにする。CPU インタプリタ (`milcpu`) で確かめられ、容量の違うプログラムへの移し替えや分割と扱いやすい。CPU を空けたいとき (電力) は MLState に移す価値がある。状態の仕組みは `mlpackage` とバックエンドに残した
- 次: 容量の費用を減らす。LFM3 のスライディング注意の層 (11 のうち 5、窓 256) は 256 トークンのリングバッファで足りるので、全注意の層だけを長い容量にする

### 16.16 KV キャッシュ: スライディング注意の層をリングバッファに — 2026-10-30

- 方針: LFM3 のスライディング注意の層 (24 層のうち 5 層、窓 256) は直近 256 トークンしか見ないので、キャッシュは 256 で足りる。`Config.cacheLen` が層の型と `Options.Cache` から長さを決め (窓がキャッシュより短いときだけ窓)、`BuildCached` はその長さの `kv_k_N`/`kv_v_N` とマスク `(1, 1, Q, 256+Q)` を作る。全注意の層 (6 層) は今までどおり `Cache` の長さ
- ホスト側 (`decode.Session`): キャッシュの長さが最大のもの (全注意の層) は順に埋め、短いもの (リング) は `pos mod 長さ` の行に書く。追加の約束は要らず、長さの違いだけで見分ける。`StepInputs` のリングのマスクは、スロット j が最後に書かれた位置 `pos-1-((pos-1-j) mod W)` を求め、それが 0 以上で窓の内側なら見えるとする。K には RoPE を掛けてから保存しているので、リングの中の並びは結果に効かない (softmax は鍵の順に依らない)
- 検証: `TestCachedMatchesFull` (小さな LFM3、窓 3、キャッシュ 8〜10、Q=1/3/8、1〜3 分割) がリングの経路を通り、キャッシュなしのプログラムと相対誤差 6e-4 以内。変異 (書き込み位置を 1 ずらす、リングの書き込みを外す、マスクの位置を 1 ずらす) はすべて検出し、リングを無効にした対照は通る。実物の LFM3-1.7B では、1024 のリング版の貪欲法の出力が以前のリングなしの版と一字一句同じ
- 結果 (M3、LFM3-1.7B lut4 2 分割、`hwnpu-generate`、200 トークン、貪欲法):

| キャッシュ | リングなし (16.14) | リングあり | 配置 | KV のメモリ (fp16) |
|---|---|---|---|---|
| 1024 | 20.6 ms | 19.1 ms | 全部 ANE | 22 MiB → 14.5 MiB |
| 4096 | 37.6 ms | 27.7 ms | 全部 ANE | 88 MiB → 50.5 MiB |
| 8192 | (約 60 ms の見込み) | 38.5 ms | 全部 ANE | 176 MiB → 98.5 MiB |
| 16384 | — | 375.5 ms | 注意の演算が CPU (下記) | — |

- 1K トークンの容量あたりの費用は 5.4〜5.6 ms から 2.8 ms に下がった。全注意の層の割合 6/11 どおり。4K で 26% 速く、8K で 1 トークン 38.5 ms (26 tokens/s)
- **ANE の 1 次元あたりの上限は 16384**: キャッシュ 16384 (鍵の軸 16385) では全注意の層の注意の演算 (2 部分で 42 と 86 演算) が CPU に置かれ、1 トークン 375 ms、読み込み 140 秒、プロセスの最大 RSS 4.5 GiB になった。`npuprobe kv -kv-layers 2 -kv-cache 16383,16384` で境目を確かめた: 鍵 16384 は全部 ANE で 16.6 ms、16385 は注意の 39 演算が CPU で 117 ms (`npu_profiles/npuprobe-m3-kv-limit.md`)。`llm.NeuralEngineMaxDim` に記録し、`hwnpu-convert` は `-cache + -seq` がこれを超えると警告する。今の方式で ANE に載る文脈の上限は 16383 トークン (Q=1)
- 次: 容量の段階 (1K/4K/8K/16K) と移し替え、プロンプトのキャッシュ、塊分けの注意 (16K を超えるにはこれが要る)

### 16.17 チャット、OpenAI 互換 API、OpenCode — 2026-10-30

- 目的: LFM3-1.7B を Neural Engine だけで動かし、対話、OpenAI 互換の API、コーディングエージェント (OpenCode) から使えるようにする
- **2 つのプログラムで 1 つのキャッシュ**: 1 回 64 トークン (-seq 64) のプログラムはプロンプトを速く読むが、1 トークンの生成は遅い (キャッシュ 8192、M3):

| -seq | プロンプト 860 トークン | 生成 ms/token |
|---|---|---|
| 1 (2 分割) | 35.4 s | 48.0 |
| 64 (1 部分) | 1.6 s | 77.0 |

  そこで同じキャッシュ長で両方を変換し、両方を読み込む。`chain.Share` は 2 つ目の鎖のキャッシュの入力 (`kv_*`、`win_*`) を 1 つ目の鎖のバッファに `Bind` し、`decode.Session.Add` はそれを確かめて (同じメモリか) プログラムを足す。`Step` はトークンの数を受け取れる最も小さいプログラムを選ぶ。重みは 2 回読み込まれる (ANE に 0.81 GiB × 2、読み込みは 2 回目以降で計 60 秒前後)
- **文脈 16320**: OpenCode の要求は system と 10 個の道具だけで 7.4K トークンあり、8K では 1 往復で要約 (compaction) が始まる。キャッシュ 16320 なら -seq 64 の鍵の軸がちょうど 16384 (16.16 の上限) で、両方とも全部 ANE に載った。16K で生成は 63〜66 ms/token、7.4K トークンのプロンプトを 13 秒で読む
- **プロンプトのキャッシュ** (`internal/chat.Engine`): 前の要求と共通の先頭は読み直さない。全注意の層は位置を戻すだけでよい (位置より先の行はマスクで見えず、後のステップは位置より先にしか書かない) が、スライディング層のリングと畳み込みの窓はその場で上書きされるので、プロンプトを読み終えるたびに `decode.Session.Mark` (リングと窓の写し、約 3 MB、最後のロジット付き) を取り、共通の先頭の中で最も後ろの印に `Rewind` する。チャットの書式は前の応答の考え (`<think>`) を消して描き直すので、共通の先頭はちょうど前のプロンプトの終わりになることが多い。並行する別の会話 (OpenCode は本題と並べて題名や要約を頼む) で本題を読み直さないよう、長い系列はキャッシュ全体の写し (`Session.Save`、16K で約 105 MB) で 2 本まで取っておき、共通の先頭が最も長いものに入れ替える。`TestEngineReuse` (CPU インタプリタ、窓 3) は、続き、途中での分岐、別の会話との往復で、最初から読んだときとロジットが一致し、再利用したトークン数が期待どおりであることを確かめる。入れ替えでキャッシュを戻さない、取っておく系列のスライスを共有する、の 2 つの改変を検出する
- **チャットの書式** (`internal/chat.Render`): LiquidAI/LFM3-1.7B の `chat_template.jinja` を Go に移した。道具は system の末尾に `<tools>` と JSON (キーの順を保ち、Python の `json.dumps` と同じ区切り)、呼び出しは `<tool_call><function=名前><parameter=キー>値</parameter>...` の XML、結果は `tool` の役の `<tool_response>`。最後の user より前の応答の考えは消す。`testdata/template.json` は jinja2 で transformers と同じ設定で描いた期待値で、6 つの会話 (道具、呼び出し、結果の結合、部品の配列の内容、null) が一字一句一致する。生成した XML は `Parse` で OpenAI の `tool_calls` に戻す (値は道具の型が string なら文字列、ほかは JSON として読めればその値)
- **サーバー** (`hwnpu-chat -listen`): `GET /v1/models`、`POST /v1/chat/completions` (SSE のストリーム、`reasoning_content`、`tool_calls`、`stop`、`stream_options.include_usage`、`usage.prompt_tokens_details.cached_tokens`)。要求は 1 つずつ処理する。`reasoning_effort` が none、minimal、low なら考えずに答えさせる (題名づくりで 2000 トークン以上考え続けたため)。小さなモデルが同じ並びを繰り返し始めたら止める (直近 64 トークンが周期 16 以下の繰り返し)。`-dump DIR` で要求、プロンプト、生成文を保存する
- **OpenCode 1.18.25** (M3、キャッシュ 16320、`build/opencode-demo`): 「main.go の add 関数のバグを直して」で、read、edit (`a - b` → `a + b`)、`go run` で 5 を確かめて終わった。12 往復、2 回目以降の要求はプロンプトの 99% を再利用 (読むのは 0.1〜0.5 秒)、生成は 63〜68 ms/token。「mul 関数を追加して」も 6 往復で編集と `go run` まで通った (ただし add の表示を mul に置き換えた。指示の取り違えはモデルの力の限界)
- 残る問題: OpenCode は最初に題名を頼み、モデルは考えないよう指示しても考えてから答えるので 1〜2 分かかり、その間本題の要求は待つ。思考なしの短い答えではモデルの知識が怪しい (富士山を 4,076 m と答えた)。lut4 の量子化の影響か、モデルの大きさの限界かは切り分けていない
- 引数なしの起動 (`chat.Prepare`): `-model` (既定は `~/Library/Caches/hwnpu/models/LFM3-1.7B`) にチェックポイントがなければ Hugging Face から落とし (途中で切れたら `Range` で続きから、`HF_TOKEN` に対応)、文脈 16320 の 1 トークンと 64 トークンのプログラムがなければ変換する (変換は `llm.Convert` に移し、`hwnpu-convert` と共有)。変換の終わりに `NAME.done` を書き、途中で止まった変換はやり直す。M3 で初回は落とす時間を除いて 3 分 23 秒 (変換 22+25+53 秒、コンパイルと読み込み 80 秒)、2 回目は読み込みだけで 59 秒

### 16.18 3 つ目のアーキテクチャ: PLaMo-Embedding-1B (PlamoBiModel) — 2026-10-30

- `internal/llm/plamo` (アーキテクチャ名 `plamo-bi`): pfnet/plamo-embedding-1b が同梱する `modeling_plamo.py` の `PlamoBiModel` を移した。PLaMo のデコーダの注意を双方向にした日本語の埋め込みモデルで、pre-norm の 16 層 (RMSNorm、注意、SwiGLU、それぞれ残差) と最終 RMSNorm。注意は融合した `qkv_proj` (2304 = 2048+128+128)、16 ヘッドに KV 1 ヘッド (MQA)、head_dim 128、ヘッドごとの重みを持つ QK-norm (重みなしの RMSNorm を head_dim で取り、`q_weight` (16, 128) と `k_weight` (1, 128) を掛ける。eps は 1e-6 に固定)、`rotate_half` の RoPE (base 10000、config に `rope_theta` はない)。MLP の `gate_up_proj` (16384, 2048) は前半が gate、後半が up。出力は最終 norm 後の隠れ状態で、プーリングはホストに任せる
- 判別: `model_type` の `plamo` は PLaMo の因果 LM (`PlamoForCausalLM`) と同じなので、`architectures` の `PlamoBiModel` で選ぶ (`Arch.ModelTypes` は空)。ほかの `architectures`、MoE (`n_expert`、`sparse_step`)、`eval_*_n_bit` は拒否する。`linear_type: fp8` と `group_size` は学習用の設定で推論のコードは読まないので受け付ける
- 入力は BERT と同じく `x` と `mask`。位置はパディングに関わらず 0..seq-1。HF の `encode` は左にパディングし、位置 ID をずらさないが、双方向でマスクがあり RoPE は距離にしか依らないので、右パディングでも同じ結果になる (テストの `left padding` で確認)。パディングのバイアスは BERT の `maskBias` を `blocks.MaskBias` に移して共有した
- `blocks` に `WeightRows` と `LinearRows` を追加: 融合した行列の行の範囲を 1 つの linear にする。定数の名前は `NAME_begin_end`。`gate_up_proj` を 1 つの linear にすると出力の軸がちょうど ANE の上限 16384 になるので、8192 ずつの 2 つにした。lut4 の表は 16 行ごとで、切れ目 (2048、2176、8192) は 16 の倍数なので、部分ごとに量子化しても行列全体を量子化したのと同じ値になる。`Weight` は中身を `matrix` に分けただけで、LFM3 (`TestBuildTiny`) と BERT のダイジェストは変わっていない
- 検証 (`plamo_test.go`): `modeling_plamo.py` をループで直接書いた fp64 の forward を正解とし、MQA のパディングなし、右パディング、左パディング、GQA (KV 2 ヘッド) の 4 通りで相対誤差 1e-4 以内。1 層ずつ (`NoHead`) と最終 norm だけ (`Start == Layers()`) をつないでも同じ。合成の重みは 0.1 以下で注意がほぼ一様になり、位置やマスクの誤りが見えないので、norm と QK-norm の重みを 1 前後にずらした。正解側は、プログラムが fp16 で持つもの (重み、RoPE の表、入力の埋め込み) を fp16 に丸めて使う (入力を丸めないと 6e-4 ずれた)。lut4 は、行列を丸ごと量子化して戻した重みの正解と 1e-4 以内。実物の大きさ (hidden 2048、16:1 の MQA、1 層) でも 6.0e-5。改変 (マスクを外す、残差の足し先を正規化後にする、gate と up を入れ替える、鍵の RoPE を外す、q_weight をヘッド 0 のものにする、鍵を値の行から取る、最終 norm を外す) はすべて検出した。マスクを外したときに通るのはパディングなしだけ
- 実物の config.json から、テンソル 130 個の名前と形が safetensors のヘッダと一致すること、1 GiB の上限で lut4 は 1 部分、fp16 は 2 部分になることをテストした (`TestPlamoEmbedding1B`)。パラメータは 10.5 億 (埋め込み表 1.03 億はホスト)
- `tools/ref/ref_logits.py` に `--attn` を追加した。`PlamoBiModel` の層は `config._attn_implementation` で `{"sdpa": ModifiedAttention}` を引くので、既定の eager では読み込めない
- 未確認: 実物のチェックポイントでの `hwnpu-check` (この Mac に PyTorch がない)、Neural Engine での配置と精度。手順は次のとおり。`modeling_plamo.py` は transformers 4.47 向けなので、新しい版で読めなければ 4.47〜4.5x を使う。既定の英文は 32 トークンに足りないことがあるので、日本語の文を渡す

```bash
python tools/ref/ref_logits.py MODEL_DIR build/plamo/ref --seq 32 --encoder --attn sdpa --text "PLaMo-Embedding-1Bは、Preferred Networks, Inc. によって開発された日本語テキスト埋め込みモデルです。検索、分類、クラスタリングに使えます。"
go run ./cmd/hwnpu-check -config MODEL_DIR/config.json -weights MODEL_DIR/model.safetensors -ref build/plamo/ref -f16
```

- **fp16 のあふれに注意**: 参照の RMSNorm は fp32 で計算するが、プログラムは fp16 で `x*x` を取るので、残差の値が絶対値 256 を超えると inf になる。`ref_logits.py` が表示する `max |hidden|` と、`hwnpu-check -f16` で確かめる。超えるなら、二乗の前に縮める書き換え (`Params` の設定) を入れる
- 次: SentencePiece BPE のトークナイザー (`tokenizer.model`: BPE、byte fallback、正規化 identity、先頭に空白を補う、空白の連続と改行が user-defined)、埋め込みを実行するコマンド (表引き、実行、プーリング。プーリングは本文のトークンだけの平均で、BOS とクエリの前置き「次の文章に対して、関連する文章を検索してください: 」は含めない)

### 16.19 PLaMo-Embedding のトークナイザー: SentencePiece BPE — 2026-10-30

- `internal/tokenizer` に `LoadSentencePiece` を追加した。`tokenizer.model` (SentencePiece の ModelProto) を、依存を足さずに protobuf の符号を手で読む。`tokenizer.json` の `Load` はそのまま。返す `SentencePiece` は `Encode`、`Decode`、`ID`、`VocabSize` を持ち、BOS は付けない (PLaMo の BOS は `<|startoftext|>` = 1。HF は `add_bos_token` で付ける)
- 符号化は sentencepiece の BPE と同じ手順にした。正規化は identity (空白を ▁ にし、先頭に ▁ を補う。壊れた UTF-8 は 1 バイトずつ U+FFFD)。user-defined の記号 (▁ の 2/4/8/16 個の連続、タブの 1/2/4 個、改行) は最長一致で 1 記号として切り出し、結合しない。残りは文字に分け、隣り合う 2 記号をつないだものが語彙にあれば候補にして、ピースのスコアが高い順 (同点なら左) に結合する。マージの一覧ではなくスコアで決まる。語彙にない記号は UTF-8 のバイト (`<0xNN>`) にする
- HF の PlamoTokenizer は slow tokenizer で、特殊トークン (`<|unknown|>`、`<|startoftext|>`、`<|endoftext|>`、`<|pad|>`) を先に切り出し、その間ごとに sentencepiece を呼ぶ。そのため特殊トークンの後ろにも ▁ が補われる (`before<|endoftext|>after` は `▁before`、`<|endoftext|>`、`▁after`)。`Encode` も同じにした。`Decode(ids, false)` は `Encode` の逆 (区切りごとに先頭の ▁ を落とす。HF の decode は特殊トークンの前に空白を足すので往復しない)。`skipSpecial` のときは、HF の `skip_special_tokens=True` と同じく区切りをつないでから戻す
- 対応しない設定は読み込みでエラーにする: Unigram、正規化の規則表 (nmt_nfkc など)、`remove_extra_whitespaces`、空白を後ろに付ける形、unused のピース、byte fallback のない BPE、バイトのピースの欠け
- 正解: `tools/ref/tokenizer_golden.py` が HF の tokenizer (特殊トークンなし) で id を出し、特殊トークンを含まない文は sentencepiece 単体の結果とも一致することを確かめる。`internal/tokenizer/testdata/plamo-embedding-1b.json` の 32 文 (日本語、英語、混在、空白の連続、タブ、改行、`\r\n`、絵文字と ZWJ、𠮷 などの珍しい字、結合文字、制御文字、▁ そのもの、特殊トークン入り、空文字列) の 599 トークンがすべて一致し、▁ そのものを含む 1 文 (decode で空白に戻る。sentencepiece も同じ) を除いて往復した。`TestPlamoEmbedding` は `HWNPU_PLAMO_DIR` (`tokenizer.model` のある場所) が要る。別に乱数で作った 3032 文 (88072 トークン) でも、符号化と sentencepiece の decode との不一致は 0
- 小さなモデルのテスト (`TestSentencePiece`): ModelProto をテストの中で組み立て、スコアの順、同点の左優先、user-defined を結合しないこと、バイトへの退避、区切りごとの ▁、往復を確かめる。改変 (同点を右優先、スコアの低い順、user-defined も結合、先頭の ▁ なし、特殊トークンを切り出さない、バイトの代わりに unknown、decode で区切りごとに ▁ を落とさない) はすべてどれかのテストが検出した。実物の語彙ではスコアの同点も user-defined をまたぐ結合も起きないので、この 2 つは小さなモデルでしか見えない
- 正解を作る環境: sentencepiece 0.2.2 は `SentencePieceProcessor.Init` がなく、`tokenization_plamo.py` が読み込めない。0.2.0 は Python 3.15 用の配布物がないので、Python 3.12 で `pip install sentencepiece==0.2.0 protobuf "transformers<5"` とした (torch は要らない)

```bash
python tools/ref/tokenizer_golden.py MODEL_DIR internal/tokenizer/testdata/plamo-embedding-1b.json
HWNPU_PLAMO_DIR=MODEL_DIR go test -run PlamoEmbedding ./internal/tokenizer/
```

- HF の `encode` は `pad_token` を `eos_token` (この tokenizer_config.json では `<|startoftext|>`) にして左に詰める。マスクで隠れるので、パディングの id は結果に効かない
- 次: 埋め込みを実行するコマンド (BOS と前置きを付けて符号化、表引き、実行、本文のトークンだけの平均プーリング)
### 16.20 KV キャッシュの容量の段階 (tiers) と、Core ML の読み込みの時間 — 2026-10-30

- 方針: 費用は容量に比例するので、短い文脈は短いキャッシュで動かす。`hwnpu-convert -cache 1024,4096,8192` が長さごとの部分 (`out.c1024.part1.mlpackage` …) を 1 つの `out.parts.json` に段階 (`Manifest.Tiers`) として並べる。重みは段階ごとに重複する (LFM3-1.7B の lut4 で 1 段階 0.8 GiB)
- 実行: `chain.Open` は最初の段階だけを読み込む。`chain.Grow` が次の段階に切り替え、`decode.Session.Migrate` がキャッシュを写す (全注意の層は行をそのまま、スライディング層のリングは長さが同じなのでそのまま。短い段階への移動は拒否)。次の段階は `chain.Preload` で裏で読み込むか、`chain.LoadAll` で先に全部読み込む。`hwnpu-generate -preload` は既定 `all`、`0〜1` で裏の先読みを始める満たし方
- 検証: `TestMigrateTiers` (小さな LFM3、キャッシュ 5 → 12、q=1) が、満杯で移した系列のロジットがキャッシュなしのプログラムと相対誤差 6e-4 以内で一致することを確かめる。ヘッドの位置ずれ (誤差 1.1)、行を写さない (0.99)、リングを写さない (0.58) の 3 つの改変をすべて検出。実物の LFM3-1.7B では、1024 → 4096 と切り替えた 1500 トークンの貪欲法の出力が固定 4096 と一字一句同じ
- **Core ML の読み込みの内訳** (`HWNPU_COREML_TRACE=1`、c4096 の 2 部分、コンパイル済みキャッシュあり): コンパイル済みモデルの参照 0.3 秒、`MLModel` の読み込み 6〜22 秒、`MLComputePlan` 5〜21 秒。合計 42 秒。配置の検証の結果は同じコンパイル済みモデル・関数・OS で同じなので、キャッシュディレクトリに `.json` で置くようにした (`hwnpu-coreml-placement-v1`)。2 回目からは 0.01 秒で、合計 16 秒。残る `MLModel` の読み込みは ANE 向けのコンパイルで、システム側のキャッシュに当たれば 1 秒台 (全 3 段階で 2.9 秒のことも)、外れると 6〜30 秒。当たり外れは制御できない (多数のモデルを試すと追い出される)
- **裏での先読みは生成と競合する**: 次の段階を裏で読み込みながら生成すると、生成が 2 倍以上遅くなる (1500 トークンで 80〜83 秒、53〜56 ms/token。固定 4096 は 42 秒)。切り替えの待ちは 0 秒になるが、合計では損。そのため既定を「先に全部読み込む」にした。常駐するサービスなら起動時に 1 回払うだけで済む
- **全段階を常駐させても動いている段階は遅くならない**: 3 段階を読み込んだ状態で 1024 の段階だけで終わる 900 トークンは 19.2 ms/token で、固定 1024 と同じ
- **端から端までの利得は今回は示せなかった**: 同じ条件で続けて測った組 (固定 4096 / 段階あり) は 57.4 / 56.7 と 33.1 / 32.4 ms/token で、段階ありが 1〜2% 速いだけ (見込みは 1024 の 1024 トークンが 19 ms、残りが 27.7 ms で約 20%)。計測の間、機械はスワップを 8.7〜9.7 GB 使っていた (別の `hwnpu-chat` が 2 GB、ほかにアプリ多数) ので、どの値も信用しきれない。全段階を読み込んだ直後の 1 本は 71.5 ms/token、同じ構成の 2 本目は 19.2 ms。空いた機械で測り直す
- 次: 空いた機械での測り直し。`MLModel` の読み込みを速くする手立て (システムの ANE キャッシュの挙動の調査)。塊分けの注意 (16K を超えるため)

### 16.21 KV キャッシュ: 塊に分けて 16K を超える — 2026-10-30

- 問い: ANE の 1 次元の上限 16384 (16.16) を、キャッシュを 16384 以下の塊に分けることで超えられるか。Flash-Decoding の合流 (塊ごとの最大値、指数の和、出力) を、ホストではなく 1 つのプログラムの中で行えば、ホスト側の仕組み (`decode`、プロンプトのキャッシュ) はほとんど変わらない
- `npuprobe kv -kv-modes host-chunk` (`internal/probe.splitAttention`): 塊 J ごとに別の入力 `kv_k_N_J`、`kv_v_N_J` (1, 4, 長さ, 128) と `mask_J` を取り、各塊は分けないキャッシュと同じ演算で注意を計算し、塊どうしは要素ごとの演算だけで合わせる。全体の最大値に ln 4n を足した値から指数を取るので、全部の塊の指数の和は 長さ/4 以下、重み付きの値の和は 長さ/4 × 値の最大 以下で、値が 16 未満なら fp16 に収まる。`TestKVChunkMatchesHost` (CPU インタプリタ、キャッシュ 256 を 64 ずつ 4 塊、一部と全部が隠れた塊を含む) で分けないキャッシュと相対誤差 8.4e-7。自分のトークンの項を外す改変を検出する。`milcpu` に `reduce_sum` と `reduce_max` を足した
- 結果 (M3、6 層 lut4、LFM3-1.7B の全注意の層の数。レポートは `npu_profiles/npuprobe-m3-kv-chunk.md`、`npu_profiles/npuprobe-m3-kv-chunk-long.md`):

| キャッシュ | host (分けない) | host-chunk (16384 ずつ) |
|---|---|---|
| 4096 | 14.5 ms | 13.2 ms |
| 16320 | 49.1 ms | 43.4 ms |
| 32768 | (CPU) | 84.0 ms、全部 ANE |
| 65536 | (CPU) | 201.6 ms、全部 ANE |

- 2 塊以上で全演算が CPU に置かれた形 (キャッシュ 512 を 256 ずつ 2 塊、1 層で切り分け。1 塊なら同じ形でも ANE): 塊を 1 つのテンソルの軸に並べて軸方向に縮約する形 (軸 1 でも軸 0 でも)、各塊をその和の割合 (`和_J / 全体の和`) で重み付けする形、和に 1/n を掛けてから全体の最大値と組み合わせる形。和を先に足し、最後に 1 回だけ全体の和で割る形は載った。理由は分からない (Core ML のコンパイラの中)
- LFM3-1.7B の見込み: 16K で 1 トークン約 64 ms (16.17) に、注意の増えた分 (上の表の 16K からの差) を足して、32K で約 100 ms、64K で約 220 ms。KV のメモリは全注意の層で 1 トークン 12 KiB、64K で 768 MiB (ほかの会話の写しを取っておく分は別)
- 次: `lfm3.BuildCached` で、全注意の層のキャッシュを塊の入力にする。`decode.Session` は位置から塊と行を決めて書き込み、`StepInputs` は塊ごとのマスクを作る。64 トークンのプログラムは鍵の軸が塊の長さ + 64 になるので、塊の長さは 16320

### 16.22 KV キャッシュ: LFM3 の文脈を 16K の先へ — 2026-10-30

- 規則 (`llm.CacheChunks`): 16320 (`CacheChunkOver`) を超えるキャッシュは 16384 (`CacheChunkLen`) ずつの塊 (最後は短い) に分け、塊ごとに別の入力 `kv_k_N_cJ` を取る。16320 以下はこれまでどおり 1 つの入力で、変換済みのモデルはそのまま動く。規則はキャッシュの長さだけで決まるので、1 トークンと 64 トークンのプログラムが同じバッファを共有できる。スライディング層のリング (256) は分けない
- `blocks.ChunkedSDPA`: 塊ごとと、ステップのトークン自身 (`mask_TYPE_self`) の注意を、要素ごとの演算だけで合わせる。全体の最大値に ln 4n を足した値から指数を取り、指数の和と重み付きの値は行列積 (`e @ 1` と `e @ V`) で求め、全部を足して最後に 1 回割る。`lfm3.BuildCached` は全注意の層でキャッシュが塊のときこれを使い、`StepInputs` はマスクを塊ごとと自分の分に分ける。`decode.Session` は `kv_NAME_cJ` を 1 つのキャッシュにまとめ、位置から塊と行を決めて書く。`Mark` はリングと窓だけを写す (塊はリングでない)
- 64 トークンのプログラムの配置: 16.21 の形 (指数の和を `reduce_sum`、塊ごとに正規化してから和で重み付け) は、1 トークンでは常に ANE だが、64 トークンでは同じグラフでもコンパイルごとに ANE と CPU が入れ替わった (注意だけのグラフで、塊 2048〜16384、2〜8 塊。同じ条件を 2 回で結果が変わる)。実物の 32K で CPU に置かれ、読み込み 198 秒、プロンプトの読みが CPU 1 コアの速さになった。和を行列積にした形は 30 回のコンパイルすべてで ANE (`TestChunkedSDPAOnNeuralEngine`)
- 検証: `TestCachedChunksMatchFull` (CPU インタプリタ、塊を 4 ずつに縮め、キャッシュ 8〜12 を 2〜3 塊) で、キャッシュなしのプログラムと相対誤差 6.3e-4 以内、共有と巻き戻しも同じ。塊の中の行の位置を 1 ずらす、マスクの塊の境目を 1 ずらす、の改変を検出する
- 結果 (M3、LFM3-1.7B lut4、`hwnpu-chat -context N`、どちらのプログラムも全部 ANE):

| 文脈 | 塊 | 読み込み (初回、64 トークンのプログラム) | プロンプト | 生成 | 先頭の合言葉 |
|---|---|---|---|---|---|
| 32768 | 2 | 39.6 s | 25,288 トークンを 83.7 s | 138 ms/token | 正答 |
| 65536 | 4 | 49.5 s | 53,736 トークンを 360.8 s | 363 ms/token | 正答 |

- 生成の 1 トークンは、16.21 の 6 層だけの計測から見込んだ値 (32K 約 100 ms、64K 約 220 ms) より遅い。トークン数が少ない (19 と 9) 測り方なので、長い生成での値は測り直す
- 残り: 使っている塊だけを動かす (今は容量全体を毎回計算する)。プロンプトの読みも文脈に比例して遅くなる

### 16.23 hwnpu-chat でチャットと埋め込みを使い分ける (`-use` にモデル名) — 2026-10-30

- `hwnpu-chat` の `-use` にモデルの名前を書く (既定 `LFM3-1.7B`)。用途 (chat、embedding) はモデルが持つ。最初は `-use chat|embed` と用途で選ぶ形にしたが、同じ用途のモデルは今後増えるので名前にした。モデルの一覧 (`chat.Catalog`: 名前、用途、リポジトリ、大きさ、説明) は `-list` で表にして出す。落としてあるかも出す
- 名前は大文字小文字を区別せず、一覧のリポジトリ名でもよい。一覧にないリポジトリは `-use OWNER/NAME` で、`config.json` だけ先に落として構造で用途を決める (`llm.Cached` ならチャット、`plamo-bi` なら埋め込み、それ以外は拒否)。モデルは `-dir/名前` (既定 `~/Library/Caches/hwnpu/models/名前`) に置く。これで `-model`、`-repo` と、いったん足した `-embed-model`、`-embed-repo` は要らなくなった。置き場は以前と同じ名前なので、落としたものはそのまま使える
- ターミナルは 1 つのモデルだけを動かし、用途で動きが決まる: チャットは従来の対話、埋め込みは入力を文書として溜め、`?` で始まる行をクエリとして文書をコサイン類似度の順に並べる。サーバーはいくつでも動かせ、チャットのモデルは `/v1/chat/completions`、埋め込みモデルは `/v1/embeddings` で出し、`GET /v1/models` は全部を `kind` 付きで並べる。要求は `model` の名前で振り分ける (`route`)。その用途のモデルが 1 つなら名前は問わない (従来どおり `"model":"lfm3"` でも答える)。複数で知らない名前は 404。用途の違うモデルの名前はエラーにする (OpenCode などがモデル一覧から埋め込みモデルを選んでしまったときに気づけるように)
- モデルは用途のほかに機能 (`Features`) を持つ: LFM3-1.7B は `tools` (ツールコール) と `thinking`、PLaMo-Embedding は `query` (クエリと文書を別々に埋め込む)。名前は Ollama の capabilities に合わせた。同じチャットでも、テキストだけでツールコールのできる LFM3 と、マルチモーダルのモデル (別に進めている Gemma 4 E2B なら `vision` など) を選び分けられるように、`-list` に機能の列とその説明を出し、`/v1/models` にも `capabilities` として返す。機能を足すときは `chat.Features` に説明を 1 行足す (`TestLookup` が説明のない機能を拒む)。一覧にないリポジトリの機能は分からないので出さない
- `-package`、`-config`、`-weights`、`-tokenizer` はチャットのモデルが 1 つ、`-embed-package` は埋め込みモデルが 1 つのときだけ使える。`-context` はチャットのモデル全部に、`-embed-seq` は埋め込みモデル全部に効く
- 埋め込みモデルの用意はチャットと同じ (`PrepareEmbedder`): `config.json`、`tokenizer.model`、`model.safetensors` を落とし、`-embed-seq` (既定 `128,1024`) の長さごとに `embed-sN.mlpackage` に変換する (lut4、kmeans-minmax。LFM3 と同じ設定で、PLaMo での比較はしていない)。lut4 で 1 部分に収まる (16.18)。フラグは `-embed-seq` と `-embed-package`
- 実行 (`chat.Embedder`): BOS (`<|startoftext|>`) を付けて符号化し、収まる最も短いプログラムに右詰めの詰め物とマスクで流し、最終隠れ状態を本文のトークンだけで平均して長さ 1 に揃える。クエリは `QueryPrefix + text` を 1 つの文として符号化し、BOS と `Encode(QueryPrefix)` の個数分を平均から外す。前置きの末尾の空白は単独の `▁` になり、本文の先頭の `▁` は前置き側に入る (小さなトークナイザーで確認。実物の金データ `…ください: 東京の天気は晴れです。` でも、前置きの直後に文頭と同じ id 35913 が来るので、同じと見ている)。双方向なので詰め物の位置は結果に効かない (16.18)。プログラムを長さで選ぶので、短い文は短いプログラムで安く済む
- 長すぎる文は API では既定でエラー (`context_length_exceeded`)、`truncate: true` で切り詰め (text-embeddings-inference と同じ名前)。ターミナルでは切り詰めて表示する。クエリか文書かは OpenAI の API にないので、Cohere や Voyage と同じ `input_type` (`query`/`search_query`、`document`/`search_document`/`passage`) で受け、既定は文書。`encoding_format: base64` (OpenAI の Python クライアントの既定) は float32 のリトルエンディアン
- テスト: 小さな PLaMo (hidden 32、2 層、8 と 128 トークンのプログラム) を CPU のインタープリタで動かし、詰め物をした埋め込みが、ちょうどの長さのプログラムで本文だけを平均したものと一致すること (`TestEmbedderPadding`)、BOS と前置きの扱い、空文字列、長すぎる文の切り詰めとエラー (`TestEmbed`)、API (`TestEmbeddingsAPI`)、名前での振り分け (`TestRoute`)、一覧の引き方と一覧にないリポジトリの用途の判定 (`TestLookup`、`TestResolve`) を確かめる。改変 (マスクを外す、平均に BOS を含める、BOS を付けない) はすべて検出した。`HWNPU_PLAMO_DIR` があれば、実物のトークナイザーで、クエリの平均が本文のちょうど先頭から始まることも確かめる (`TestPlamoQueryTokens`)
- 未確認: この環境は Hugging Face にも Neural Engine にも届かないので、実物での確認はしていない。とくに HF の `encode_query`/`encode_document` が前置きの長さをどう数えるか、平均を正規化するか (コサイン類似度には効かない)、`add_eos_token` の有無は、16.18 の記録からの推測。`tools/ref/embed_golden.py` が HF の `encode_query`/`encode_document` の結果とトークン列を書き、Mac で `TestPlamoEmbeddingGolden` がトークン列の一致、コサイン類似度 0.98 以上、各クエリの最も近い文書の一致を確かめる。fp16 のあふれ (16.18) もここで NaN/Inf として見える

```bash
python tools/ref/embed_golden.py MODEL_DIR MODEL_DIR/embed_golden.json
HWNPU_PLAMO_DIR=MODEL_DIR go test -run 'PlamoQueryTokens|PlamoEmbeddingGolden' -v ./internal/chat/
```

- 次: 実物での照合と速さの計測 (128 と 1024 の 1 文あたりの時間、読み込みの時間)。複数の文をまとめて流すバッチ (今は 1 文ずつ)

### 16.24 hwnpu-chat -opencode — 2026-10-30

- サーバーと OpenCode をまとめて起動する。モデルを読み込んだあと、127.0.0.1 の 30000〜49151 番 (IANA と macOS の一時ポート 49152 番以降を避ける) から乱数で選んだ位置から空きを探して `net.Listen` し、その listener のまま `http.Server.Serve` に渡す (探してから開くまでの間に取られない)。OpenCode の設定 (プロバイダー `hwnpu`、`@ai-sdk/openai-compatible`、モデルの文脈と出力の上限) をキャッシュディレクトリの `opencode/opencode.json` に書き、`OPENCODE_CONFIG` で渡す。カスタム設定はグローバルより強く、プロジェクトの `opencode.json` より弱い。モデルは `-m hwnpu/NAME` で選ぶ (画面を出すときは先頭に、`run` のときはその後ろに置く。ほかのサブコマンドは `-m` を受け付けず失敗するので付けず、設定の `model` に任せる)。`-use` に複数のモデルを書くと、チャットのモデルを全部設定に並べ、最初のものを使う (埋め込みのモデルもサーバーには出る)
- OpenCode は `os/exec` で起動し、標準入力、出力、エラーにこのプロセスのファイルをそのまま渡す (パイプを挟まないので、端末の raw モード、代替画面、マウス、端末への問い合わせがそのまま通る)。サーバーのログはファイルに回す。SIGINT はこのプロセスでは捕まえて捨てる (無視にすると子に継がれる。捕まえた信号は exec で既定に戻る)。OpenCode が終わったらサーバーを閉じ、その終了コードで終わる
- 検証 (M3、OpenCode 1.19.5): `hwnpu-chat -opencode run "…" < /dev/null` で、ポート 42846 に開き、7 往復で `a - b` を `a + b` に直して終了コード 0、サーバーも止まった。疑似端末で TUI を起動し (`-- --prompt`)、代替画面とマウスが有効になり、5 往復の要求が届き、Ctrl-C 1 回で OpenCode とともに終了コード 0 で終わった。`opencode run` は標準入力が閉じるまで待つ (端末なら問題ない)

### 16.25 4 つ目のアーキテクチャ: Gemma 4 (gemma-4-E2B-it のテキスト部分) — 2026-10-30

- 対象は `google/gemma-4-E2B-it` (指示調整版) だけ。ベースの `gemma-4-E2B` と config.json は `eos_token_id` しか違わない (同じ構造で、-it は `chat_template.jinja` を持つ)。チェックポイントは `Gemma4ForConditionalGeneration` (テキスト 4.65B、音声塔 0.30B、画像塔 0.17B) で、テキストだけを移した。音声と画像のテンソル 1411 個は `unused`
- `internal/llm/gemma4` (アーキテクチャ名 `gemma4`): `modeling_gemma4.py` の `Gemma4TextModel` を移した。35 層、hidden 1536、8 ヘッドに KV 1 ヘッド。層の型は sliding (窓 512、head_dim 256) 4 層に full (head_dim 512) 1 層の繰り返し。各層はサンドイッチ norm の注意と GeGLU (`gelu_pytorch_tanh`) の MLP、そのあと層ごとの埋め込みの枝、最後に残差ごと `layer_scalar` を掛ける。注意のスコアは縮めない (`scaling = 1.0`。QK-norm が代わり)。値にも重みなしの RMSNorm。RMSNorm は `w * x` (Gemma 3 までの `(1 + w) * x` ではない)。埋め込みに `sqrt(hidden)` (float32 の参照に合わせて丸めない値)、ロジットに softcap 30 (tanh)。full 層の RoPE は proportional: head_dim 512 のうち先頭 64 周波数だけを回し、残りは周波数 0 (LFM3 の部分回転と同じ並び)
- **層ごとの埋め込み (PLE)**: 2 つ目の埋め込み表 `embed_tokens_per_layer` (262144 x 8960、bf16 で 4.7 GB) はホストに置く。`llm.TokenInputs` を足し、表の行を `x` と同じようにホストが引いて入力 `ple` (1, seq, 8960) にする (`llm.TokenRow` は x の行の後ろにこれを並べ、`decode.Session.Step` が分けて書く)。層 0 から始まるプログラムが `(norm(per_layer_model_projection(埋め込み)) + ple * 16) / sqrt(2)` を計算し、層 i はその i 番目の 256 を使う。分割したときは出力 `per_layer_inputs` を後ろの部分が同名の入力で受け取る。`per_layer_model_projection` は層の外だが最初の部分に入るので、`llm.Prologue` で分割の見積もりに知らせる。表はホストに置くので見積もりから外す
- **KV の共有**: 層 15〜34 は自分の K/V を持たず、sliding は層 13、full は層 14 の K/V (キャッシュありならキャッシュとそのステップの分をつないだもの) を使う。キャッシュを持つのは 15 層だけ (full 3 層、sliding 12 層はリング)。共有の元と使う層が別の部分に分かれたときは、元の部分が `shared_k_sliding` などを出力し、後ろの部分が同名の入力で受け取る。そのため `chain` に「前の部分の出力と同じ名前の入力はその出力のバッファに結ぶ」を足した (`passed` は同期しない)。`llm.WritePart` は `Cached` なモデルならキャッシュなしでも `BuildCached` を使い、この出力を落とさない
- `blocks.SDPAScale` を足した (スコアの倍率を指定。1 なら掛け算を置かない)。`SDPA` はこれを `1/sqrt(head_dim)` で呼ぶだけで、LFM3 (`TestBuildTiny`)、PLaMo、BERT のダイジェストは変わっていない。`blocks.SDPAGrouped` も足した: KV ヘッドを共有する問い合わせのヘッドを系列の軸に積み、KV ヘッドごとに 1 回の行列積にする。`SDPA` の `tile` は KV を問い合わせのヘッドの数だけ複製するので、E2B (8 ヘッドに KV 1、全注意の head_dim 512) でキャッシュ 16320 だと、全注意の 7 層で毎ステップ K と V を 8 倍 (1 つ 128 MB) に写していた。gemma4 だけが使う (LFM3 は今のまま)
- **fp16 のあふれ**: 実物の層 0 は input_layernorm の重みが最大 270 あり、正規化した入力が 269 になる。それを射影した q/k/v を QK-norm でまた正規化するときに二乗が fp16 を超え、注意の出力が 45% ずれた。残差も layer_scalar (0.018) を掛ける前は 1900 に達する。RMSNorm は入力の倍率に依らないので、gemma4 の norm はすべて `x/64` を正規化する (eps を 64² で割る。数学的に同じ)。`blocks.RMSNorm` の inScale (重みを 64 で割る) は使わない: E2B の norm の重みは小さいものが 0.16 で、割ると fp16 の非正規化数になる (小さなモデルで、重みを割る形は 2.5e-4 ずれた)。PLE の射影の norm だけは入力が小さく (RMS 0.3)、縮めると二乗が非正規化数になって 0.04% → 0.5% に悪化したので縮めない。LFM3 の `post-scale` に当たる書き換えは、Neural Engine で効くか確かめていないので入れていない
- テスト (`gemma4_test.go`、`cache_test.go`、`config_test.go`): 小さな合成モデル (6 層、sliding/full、GQA、proportional RoPE、PLE、層 3〜5 が層 1・2 の K/V を共有、double wide MLP、softcap) で、`modeling_gemma4.py` をループで直接書いた fp64 の forward と相対誤差 1.6e-6。正解側は Config のメソッド (共有の元、MLP の幅、head_dim) を使わず、HF のコードと同じ判定を自前で書いた。1 層ずつ (参照の `per_layer_inputs` と共有 K/V を与える) と最終段だけでも 1e-4 以内。2〜4 部分に分けても同じ。キャッシュありで 1〜3 部分、q=1/3/4/11 を順に流し、キャッシュなしと 5e-3 以内 (この小さなモデルは縮めないスコアが 30 前後になり fp16 に敏感で、全演算を fp16 に丸めると 0.2〜3% 動く。K/V を fp16 で持つだけの差は 0.2〜0.3%)。改変 15 個 (値の norm を外す、共有の元を最初の層にする、PLE のスライスをずらす、layer_scalar を外す、窓を 1 広げる、リングのスロットを 1 ずらす、proportional を全周波数にする、スコアを 1/sqrt(d) で縮める、埋め込みを縮めない、PLE の行を 16 倍しない、共有の出力を落とす、キャッシュありの共有に今のステップを含めない、softcap を外す、MLP を double wide にしない、ほか) はすべて検出した。実物の config.json から形と分割の見積もり (lut4 で 2 部分) を確かめる。対応しない設定 (MoE、`attention_k_eq_v`、`attention_bias`、双方向、tie しない出力層、ほかの rope、層の型、PLE の語彙の違い、共有の元がない) はエラー
- 参照: `tools/ref/ref_gemma4.py`。チェックポイント全体を float32 にすると 20 GB で 16 GB の Mac に載らないので、text config から `Gemma4ForCausalLM` を float32 で組み、言語モデルのテンソルだけを 1 つずつ読み、PLE の表はトークンの行だけをチェックポイントから読むモジュールに差し替える (9 GB)。forward は transformers のもの。層の入力の hook で `per_layer_inputs`、層 15 の注意の hook で共有 K/V を取り、`REF.NAME.bin` に書く。`hwnpu-check` は x 以外の入力に、引く表の行 (`ple`)、参照の値 (`check.Ref.Extra`)、どちらもなければ 1 を与える。`ref_logits.py` は埋め込みの `embed_scale` を割って書くようにした (0 番は表の行そのもの、の約束どおり)
- トークナイザー: Gemma の tokenizer.json は byte fallback の BPE (空白を ▁ に置き換え、間で切らず、マージは順位、語彙にない文字は UTF-8 のバイト)。`tokenizer.Load` が読む (`fallback.go`)。HF の tokenizers と、35 文 (`testdata/gemma-4-E2B-it.json`、`TestGemma4` は `HWNPU_GEMMA4_DIR` が要る) と乱数の 300 文 (計 7026 トークン) で一致。この tokenizer は BOS を付けない (チャットの書式が `<bos>` を書く)
- チャットの書式: `chat.Template` を足し、LFM3 と Gemma 4 を `TemplateFor(アーキテクチャ名)` で選ぶ。`chat.Gemma4` は chat_template.jinja (2026-07-09 版) の移植で、ターン `<|turn>ROLE`、思考 `<|think|>` と `<|channel>thought ... <channel|>`、道具の宣言・呼び出し・応答の独自の記法 (文字列は `<|"|>`、キーは dictsort) を書く。`Render` は「プロンプトが思考の中で終わるか」も返す (道具の応答の後、思考ありなら `<|channel>thought\n` で終わる)。transformers で描いた 9 会話 (`testdata/gemma4_template.json`、`tools/ref/template_golden.py`) と一字一句一致。生成の終わりは `<turn|>`、`<eos>`、`<|tool_response>`
- **実物の照合** (`hwnpu-check`、`<bos>` から始まる 31 トークン): `tensors: 0 missing, 0 mismatched, 1471 unused` (音声と画像 1411 個、共有の層の k_proj/v_proj/k_norm 60 個)。fp16 で各層の相対 RMS 誤差 1e-4 以下、最終段とモデル全体は argmax 31/31、KL 0.0000。全演算を fp16 に丸めても (`-f16`) 最悪の層 0.0098 (層 13)、全体は 31/31、KL 0.0004。norm を縮める前は `-f16` で層 0 が 0.30、全体が 16/30、KL 3.17 だった。参照の hidden の最大は 132 (層 30) で、層の境目は fp16 に収まる
- **lut4 は粗い**: `-format lut4 -quant kmeans-minmax` で各層 5〜31%、全体 argmax 13/31、KL 2.54 (LFM3 は 0.95)。sandwich norm のため層の出力の大半が層の中で作り直され、4 ビットの重みの誤差がほぼそのまま層の出力に出る。重みの誤差も LFM3 より大きい (層 11 で 0.17〜0.22、LFM3 は 0.117)。1 つの群を fp16 に残しても (MLP、注意、PLE の枝、出力層) 半分程度にしかならず、表の当て方 (uniform、kmeans) や群の大きさ (4 行、1 行) でも 11〜21% より下がらない。出力層だけは lut4 でも KL 0.07 で、fp16 に残すと 0
- **QAT 版**: `google/gemma-4-E2B-it-qat-q4_0-unquantized` (同じ形式の bf16) の重みは 32 個ごとに 16 値以下で、Q4_0 の格子にちょうど乗る。ただし ANE の lut4 は 16 行 x 行全体で 1 つの表なので、32 列ごとの scale は表せず、層 11 の重みの誤差は 0.16〜0.20 にしか下がらない。Q4_0 をそのまま使うには、列のブロックごとの部分積をバッチの行列積で求めて scale を掛けて足す形 (整数 -8..7 の 1 つの表で正確) が要る。未着手
- **Neural Engine** (M3、`-seq 1`、lut4 kmeans-minmax、`-target m3` で 2 部分 0.54 + 0.52 GiB (層 0〜23 と 24〜34)、変換 80 秒、300 トークンの貪欲法): キャッシュ 1024 で 1 トークン 23.6〜25.8 ms (約 42 tokens/s)、16320 で 47 ms (21 tokens/s)。LFM3-1.7B (19 ms、63〜66 ms) より短い文脈では遅く、長い文脈では速い (キャッシュを持つ層が 15、全注意は 3 層だけ)。`SDPAGrouped` の前 (`tile`) は 1024 で 31〜35 ms、16320 で 90〜150 ms だった。スワップが 10 GB を超えた状態では 1024 でも 72〜80 ms になり、測り直すと戻った (同じ条件で交互に測った)。ソフトキャップ (mul、tanh、mul) は 262144 の軸のままだと CPU に置かれたので、(16, 16384) に畳んでから掛ける。最後に (1, seq, 262144) に戻す reshape だけは CPU に置かれる。畳んだまま出力すると全部 ANE に載るが、1 トークン 137 ms と 4 倍遅くなったので戻した。出力層の linear (出力 262144) 自体は ANE に載る
- 生成の例 (貪欲法): 英語の説明やコードは自然。「日本の首都は」に「東京」。「富士山の高さは」には 2,400 m 前後と誤った (lut4 の損失。LFM3 も 4,076 m と答えた)
- **hwnpu-chat** (`-use gemma-4-E2B-it`、文脈 16320、1 トークンと 64 トークンのプログラム): OpenAI 互換 API で、思考なしの日本語の回答、思考あり (`reasoning_content` に分かれる)、道具の呼び出し (`get_weather{"city":"Osaka"}`、`finish_reason: tool_calls`) が通った。読み込みは 1 つ目 133 秒、2 つ目 113 秒 (初回のコンパイル)
- 未確認: 64 トークンのプログラムの読み込み速度の計測、OpenCode からの利用、分割の境目を層 13 より前にして共有 K/V を部分の間で渡さない形の速さ。lut4 の粗さは 16.26 で 6 ビットの表にして解いた

### 16.26 Gemma 4 の重み: 6 ビットの表 (lut6) を既定に — 2026-10-30

- 問題: lut4 の Gemma 4 E2B は KL 2.54 nats (argmax 13/31)。M3 の Neural Engine で「富士山の高さ」に 3778 と答え、『こころ』の作者や日本一長い川に答えなかった (16.25)
- **ANE に載る形の候補を測った** (`npuprobe` と同じ作りの一時的なプログラム、E2B の大きさの線形層 6 層、2.08 億パラメータ、トークン 1 つ、M3):

| 形 | 大きさ | 1 回 | 配置 |
|---|---|---|---|
| lut4 (16 行ごとに 16 値の表) | 104 MB | 1.81 ms | NPU |
| lut6 (64 値の表) | 157 MB | 2.65 ms | NPU |
| lut8 (256 値の表) | 211 MB | 3.49 ms | NPU |
| Q4_0 を 32 列ずつの grouped conv で (整数 -8..7 の表 1 つ + ブロックごとの scale を掛けて足す) | 117 MB | 12.3 ms | NPU |
| Q4_0 を 32 列ずつのバッチの matmul で | 117 MB | 42 ms | すべて CPU |
| fp16 | 415 MB | 6.52 ms | NPU |

  lut6 と lut8 は ANE に載り、速さはほぼ重みのバイト数に比例する。QAT 版 (Q4_0 の格子に乗る) をそのまま使う形は、grouped conv なら ANE に載るが 7 倍遅く、バッチの matmul は CPU に落ちるので見送った。int8 (行ごとの scale) は PROBE.md で lut4 の 5.5 倍遅いと分かっている
- 実装: `mlpackage.Builder.LUT` (4/6/8 ビット。uint6 は下位ビットから詰める、blob の型 13、uint8 は 1 バイトずつ)、`palette.QuantizeBits` (表の大きさ 2^bits、4 ビットは `Quantize` と同じ値)、`milcpu` の uint6/uint8 の読み出し、`-format lut6|lut8`。`TestConstexprWideTables` が CPU の展開と `Dequantize` の一致を、`TestQuantizeBits` が詰め方と誤差 (ガウスで 0.134 → 0.037 → 0.009) を確かめる。LFM3、PLaMo、BERT のダイジェストは変わっていない
- 精度 (`hwnpu-check`、`<bos>` から 31 トークン、kmeans-minmax):

| 形式 | 層の誤差の平均 / 最悪 | 全体の argmax | KL (nats) | M3 の 2 部分か | 1 トークン (キャッシュ 1024) |
|---|---|---|---|---|---|
| lut4 | 15% / 31% | 13/31 | 2.54 | 2 部分 1.06 GiB | 23〜24 ms |
| lut6 | 4.7% / 11% | 25/31 | 0.25 | 2 部分 1.60 GiB | 31.5 ms |
| lut6、全演算を fp16 に丸める | 4.7% / 11% | 24/31 | 0.26 | | |
| lut8 | 1.2% / 3.0% | 27/31 | 0.048 | 3 部分 2.16 GiB | 41 ms |
| fp16 | 0.01% | 31/31 | 0.0000 | (4.6 GB、5 部分) | |

- 同じ質問への答え (M3、貪欲法): 富士山の高さ lut4 3778 / lut6 3776 / lut8 3776。17×23 はどれも 391。『こころ』の作者は lut4 が答えず、lut6 と lut8 は夏目漱石。日本一長い川は lut4 が「明確な定義はない」、lut6 と lut8 は信濃川
- **既定を lut6 にした**: `llm.Arch.Format` (変換の既定の形式) を足し、Gemma 4 は `lut6`。`hwnpu-convert` は `-format` を省くとこれを使い、`hwnpu-chat` の変換も同じ。LFM3 は今のまま lut4
- `-format-for SUBSTRING=FORMAT` (`Options.FormatFor`) を足した: 名前に SUBSTRING を含むテンソルだけ別の形式にする (最後に合うものが勝ち、`-keep-fp16` がすべてに勝つ)。lut6 のまま一部を lut8 にする混ぜ方は、MLP だけで 1.99 GiB と 2 部分に収まらず、注意だけでは層の誤差の大半 (MLP) が残るので、今回は既定にしていない
- **hwnpu-chat** (`-use gemma-4-E2B-it`、lut6、文脈 16320、2 部分 0.81 + 0.79 GiB): 1 トークン 55 ms (lut4 は 47 ms)。思考なしで「富士山の高さは3,776メートル」「信濃川」と答え、思考ありの 465 トークンも `stop` で終わり、道具の呼び出しも通った。読み込みは 96 秒と 104 秒 (初回)

### 16.27 d1-2B: LFM3 の判定モデルと System One API — 2026-10-30

LiquidAI/d1-2B / d1-Omni-2B / LFM3-VL-2B / LFM3-Audio-2B への対応の検討 (2026-10-29) の結論は、テキスト部がどれも LFM3 と同じ構造で、hwnpu は埋め込みの表引きをホストで行う (入力が `x`) ので、画像や音声のエンコーダの出力を `x` に差し込めばテキスト側のプログラムは変わらない、というもの。順に (1) d1-2B のテキストのみ、(2) LFM3-VL の視覚エンコーダ、(3) LFM3-Audio の音声入力 (d1-Omni-2B もこれで揃う)、(4) LFM3-Audio の音声出力、とする。d1-omni-600M (LFM2 系、SigLIP2 と FastConformer) は対象外。この節は (1)。

Hugging Face にこの環境から届かなかったため、構造は llama.cpp (2026-10-29 の master) の `conversion/lfm3.py`、`conversion/lfm2.py`、`src/models/lfm3.cpp`、`tools/server/server-decision.cpp` から読んだ。config.json の値とテンソル名は実物で確かめていない。

- **d1-2B の正体**: LFM3-VL のチェックポイント (`model_type: lfm3_vl`、LFM3 の設定は `text_config` の下) で `architectures: ["D1Model"]`。重みは元のまま (llama.cpp は「the stock weights, the d1 prompt read at its last token」)。違いは読み方だけで、プロンプト全体を 1 回で読み、full_attention の層は双方向、sliding_attention の層と短い畳み込みは因果のまま (`LLAMA_NON_CAUSAL_TYPE_DENSE_ONLY`)。KV キャッシュは使わない。答えは最後のトークンのロジットのうち、選択肢のラベルのトークンだけを使う
- **`internal/llm/lfm3`**: `ParseConfig` が `text_config` の入れ子を読む (`model_type` は `lfm3` か `lfm3_vl`、`text_config.model_type` は `lfm3`)。テキスト部のテンソル名の接頭辞 `Config.Prefix` は `model.language_model.` (HF の複合モデルの並び。llama.cpp は `language_model.` を取り除いて読む)。出力層は `lm_head.weight` のまま。最上位の `tie_word_embeddings` が `text_config` のものより優先する。`D1Model` なら `Config.D1` で、`d1_codes` が 2、`d1_attention` が `set` (llama.cpp のサーバーが前提にする値) でなければエラー。d1 のプログラムは `x` と `mask` (1, seq) を取り、full_attention の層はパディングだけを隠す (BERT と同じ `MaskBias`)。出力は最終 norm の後の隠れ状態 (`Output()` は隠れ状態の幅)。出力層 (約 1 億 3 千万パラメータ) はプログラムに入れず、ホストがラベルの行だけを読む。KV キャッシュ付きの変換はエラー。GGUF の名前は接頭辞を `model.` に戻して引く。`TestBuildTiny` のダイジェストは変わっていない
- **`internal/systemone`**: llama.cpp の `POST /v1/systemone` (TypeSafe 互換の System One API) を d1 用に移した。要求の解釈 (`state`、`questions` の `choice` / `score` / `noul`、キーの順を保つ)、プロンプト (llama.cpp が LFM3 d1 の GGUF に書く systemone テンプレート = LFM2 の d1 のプロンプトと `<think></think>`)、ラベル (`d1_labels`: choice は選択肢のキーがすべて 1 文字の英字ならそれ、でなければ A、B…、1 トークンでないものは予備の表から。空白付きの形も同じ選択肢として最大を取る。noul は yes/Yes/YES と no/No/NO、score は数字)、答え (`calibration.json` の `default` の温度で softmax。choice は最大、score は累積確率が半分に達する段階 (中央値)、noul は true の確率。confidence は答えの確率。d1 の api.py に合わせた llama.cpp のとおり)。要求の文字列に control トークンの綴り (`<|im_end|>` など) があれば、最初の文字の後にゼロ幅空白を入れる (`d1_break`)。質問ごとにプロンプトが別なので 1 問 1 回の実行で、`-seq` の違う複数のパッケージのうち、プロンプトが入る最短のものを使う
- **`internal/pyjson`**: chat の順序付き JSON 値を切り出して共有にし、`json.dumps(indent=2)` の形 (`WriteIndent`) と Jinja の真偽 (`Truthy`) を足した。数値は読んだときの文字列のまま書く (Python は自分の形で書くので、`1.50` などは違う)
- **`cmd/hwnpu-systemone`**: 要求を 1 つ読んで答えを出すか、`-listen` で `/v1/systemone` と `/v1/models` (`output_modalities: ["decisions"]`) を提供する。`-cpu` は `milcpu` で動かす (Neural Engine のない機械で答えを確かめる用。遅い)
- **検証**: `reference_test.go` に LFM3 の fp64 の forward (サンドイッチ norm、Lfm2ShortConv、QK norm と部分 RoPE、repeat_kv、ヘッドごとのゲート、SwiGLU。d1 は full_attention だけ双方向) を書いた。今まで LFM3 には独立した正解がなかったので、因果 LM のロジットも照合する。相対誤差は因果 LM 2.9e-7、d1 (パディング 3) 2.8e-7。1 層ずつと最終 norm だけの実行、post-scale 64 も 1e-4 以内。参照側は、プログラムが fp16 で持つ定数 (RoPE の表、eps、注意のスケール) と入力 `x` を fp16 に丸める (丸めないと 1〜3e-4 の差が出て、構造の誤りと区別できない)。改変 (d1 の全注意を因果にする: 0.78、パディングのマスクを外す: 0.76、post_ffn_norm を飛ばす: 0.85) はすべて検出した
- プロンプトの正解は `tools/ref/systemone_golden.py` が llama.cpp のテンプレートを jinja2 で描画して作る (`internal/systemone/testdata/prompts.json`、7 通り: 説明あり/なし/偽の値の choice、条件あり/なしの noul、score、control トークンの綴り、日本語と入れ子の JSON)。`tojson` は llama.cpp の jinja と同じく ASCII に逃がさず、キーの順を保つ。`TestModel` は小さな d1 を `milcpu` で動かし、パディングのある長いプログラムの答えが、プロンプトの長さのプログラムの出力から独立に計算した確率と一致することを確かめる。最後の行でなく先頭の行を読む、パディングを隠さない、ラベルの組で最大を取らない、エスケープの位置を変える、中央値を別の分位にする、の改変を検出した
- 未確認: 実物の config.json の値と、テンソル名の接頭辞 (`model.language_model.` は推測)。`hwnpu-check` での照合。参照は d1 のリモートコードの読み方が分からないので、まず llama.cpp と答えを比べるのが確実 (同じ要求を `llama-server --model d1-2B.gguf` の `/v1/systemone` と `hwnpu-systemone -cpu` に送り、確率を比べる)。Neural Engine での配置と精度、速さ (Mac がない)。画像の入力 (次の段階)

```bash
go run ./cmd/hwnpu-convert -config d1-2B/config.json -weights d1-2B/model.safetensors -seq 512 -o d1-512.mlpackage
go run ./cmd/hwnpu-systemone -model d1-2B -package d1-512.mlpackage request.json
go run ./cmd/hwnpu-check -config d1-2B/config.json -weights d1-2B/model.safetensors -ref build/d1/ref   # 参照が作れたら
```

### 16.28 LFM3-VL の視覚エンコーダと画像の入力 — 2026-10-30

16.27 の段階 (2)。LFM3-VL-2B のテキスト部は 16.27 で変換できるようになったので、視覚エンコーダと、画像を d1-2B (`/v1/systemone`) とチャットに渡す部分を足した。構造は llama.cpp の `tools/mtmd/models/lfm3vl.cpp`、`conversion/lfm3.py`、`tools/mtmd/mtmd-image.cpp` (前処理)、`tools/mtmd/clip.cpp` (位置、トークン数) から読んだ。HF のリモートコードと config の実物は見ていない。

- **`internal/llm/lfm3vl`** (アーキテクチャ名 `lfm3-vision`): `model_type` は `lfm3_vl` で、これはテキスト部 (`lfm3`) が持つので、視覚エンコーダは `-arch lfm3-vision` で選ぶ。構造は LFM 型で、patchifier (パッチの画素への linear。重みは (d, 3, t, p, p) で、静止画は時間方向の全スライスに同じ画像を入れるので、変換時に t で足し合わせる)、埋め込みの RMSNorm、pre-norm の層 (畳み込みの層: in_proj を B, C, x に分け、B*x にパッチの格子上で 3×3 の depthwise 畳み込み (ゼロ埋め) をかけ C を掛ける。注意の層: QK の RMSNorm、2D RoPE、GQA、双方向)、どちらも SwiGLU。最終 norm はない。projector はパッチごとの LayerNorm (eps 1e-6)、2×2 のパッチの連結 (上の行の左右、下の行の左右)、linear、GELU (erf)、linear
- **2D RoPE**: ggml の `GGML_ROPE_TYPE_VISION` (`n_dims = head_dim/2`、組は (k, k+head_dim/2))。k が head_dim/4 未満は行、以上は列で回り、どちらの軸も head_dim/2 幅の RoPE の周波数 (`theta^(-2j/(head_dim/2))`)。表を作り直せば `blocks.ApplyRope` で回せる
- **3×3 の畳み込み**は LFM3 の `taps` と同じく、(行, 列, チャネル) の並びで 9 回のずらした積和にした (`milcpu` で動き、ANE の conv の誤差も避ける)。2×2 の連結は 5 次元の reshape と transpose
- **固定の大きさ**: Neural Engine は固定の形なので、プログラムは画像の大きさ 1 つ (`-image 幅x高さ`、2×パッチの倍数) ごとに変換する。`-seq` は使わない。`hwnpu-convert` は、パッケージのメタデータにアーキテクチャ名と固有の設定の実効値 (`com.github.shibukawa.hwnpu.arch`、`...param.image`) を書くようにした。実行側はこれで画像の大きさを知る
- **前処理** (`lfm3vl.Processor`): `preprocessor_config.json` (と `processor_config.json` の `image_processor`) の平均、標準偏差、`resample` (2 = bilinear、3 = bicubic)、`size` (最小と最大の画素数)。`SmartResize` は llama.cpp の静止画の smart_resize (各辺を 2×パッチの倍数に偶数丸め、画素数が範囲を外れたら縦横比を保って縮小か拡大)。`Resize` は PIL / torchvision のアンチエイリアス付きの縮小拡大 (横、縦の順に、各段で 8 ビットに丸める) を浮動小数点で書いたもので、PIL の結果と ±1 以内で一致する (`tools/ref/resize_golden.py`、4 つの大きさ × 2 つのフィルタ)
- **`internal/llm/runner`**: KV キャッシュのないプログラム (パッケージか out.parts.json) を NPU (`chain`) か CPU (`milcpu`) で動かす共通部分。`systemone` もこれを使うようにした
- **`internal/vision`**: 視覚エンコーダのプログラムを大きさごとに読み込み、画像を、処理系が選ぶ大きさ (smart_resize) に縦横比がいちばん近いプログラムの大きさに引き伸ばして埋め込みにする。大きさが同じなら参照と同じ入力、違えば近い格子への引き伸ばしで近似。画像はプロンプトの中で `<|vision_start|>`、埋め込みの行、`<|vision_end|>` (llama.cpp の mtmd と同じ)。動画は扱わない
- **d1-2B の画像** (`systemone`): `files` / `images` (data URL) と、チャットのメッセージの形の state の `image_url` 部分 (state から取り除く) を、この順に最大 8 枚。llama.cpp のとおり、要求の文字列のメディアの印 `<__media__>` を空白に置き換え、印を画像の数だけユーザーの番の先頭に置き、そこを画像のトークンにする。画像は 1 要求に 1 回だけ埋め込みにして全質問で使う。音声と動画の部分は 501
- **チャットの画像** (`chat`): `image_url` 部分は、llama.cpp のサーバーと同じく、文字の中のその位置にメディアの印 (起動ごとの乱数入り) を置き、`Engine.Tokens` が画像のトークンにする。画像の行は負のトークン id で表し、同じ画像 (data URL の SHA-256) は同じ id になるので、前のプロンプトのキャッシュがそのまま使える。保持は最近の 16 枚で、追い出した画像が再び来れば別の id で埋め込み直す。`/v1/models` は画像を読めるとき `input_modalities` に `image` を出す。`hwnpu-chat -vision`、対話では `/image FILE 質問`
- **検証**: `lfm3vl_test.go` に llama.cpp のグラフから書いた fp64 の正解 (patchifier は時間方向に同じ画像を入れて計算、畳み込みは PyTorch の相互相関の定義、RoPE は ggml の式そのまま、2×2 の連結は ggml の reshape と permute をたどった並び) を置き、6×4 パッチの横長の画像で相対誤差 4.5e-7、1 層ずつ (`NoHead`) と projector だけの実行も 1e-4 以内。合成の重み (±0.1) のままでは、層の後ろに norm がないため層の寄与が埋め込みの数 % しかなく、注意もほぼ一様で、層を壊しても誤差 4e-4 程度しか出なかった。そこでテストの重みを学習済みに近い形 (行列は 5 倍、norm は 1 前後、QK の norm は 3 前後) にした。改変 (2×2 の連結の順、RoPE の行と列、周波数の分母、RoPE を外す、畳み込みのタップの転置、B と C の取り違え、patchifier の和を上書き) はすべて 0.05〜1.1 の誤差で検出した。`vision` のテストは縦横比によるプログラムの選択と、出力がそのプログラムの出力と一致することを、`systemone` のテストは画像の埋め込みがユーザーの番の先頭に入ることを (プログラムの出力から独立に計算した確率と 1e-6 以内)、`chat` のテストは画像のトークンの並び、同じ画像の id の再利用、追い出しと埋め込み直しを確かめる
- 未確認: 実物の config (vision_config の `layer_types` の有無と値、`projector_hidden_size`、patchifier の重みの形が 2 次元か 5 次元か)、テンソル名、実物での参照との照合 (llama.cpp の `llama-mtmd-cli` の埋め込みとの比較が確実)。LFM3-VL のチャットテンプレートの画像の置き方 (今はメッセージの文字の中の位置にそのまま置く)。Neural Engine での配置と精度、速さ

```bash
go run ./cmd/hwnpu-convert -arch lfm3-vision -image 512x512 -format fp16 -config LFM3-VL-2B/config.json -weights LFM3-VL-2B/model.safetensors -o vision-512x512.mlpackage
go run ./cmd/hwnpu-convert -arch lfm3-vision -image 768x384 -format fp16 -config LFM3-VL-2B/config.json -weights LFM3-VL-2B/model.safetensors -o vision-768x384.mlpackage
go run ./cmd/hwnpu-systemone -model d1-2B -package d1-1024.mlpackage -vision vision-512x512.mlpackage,vision-768x384.mlpackage request.json
```

### 16.29 LFM3-Audio の音声入力と d1-Omni-2B — 2026-10-30

16.27 の段階 (3)。構造は llama.cpp の `tools/mtmd/models/lfm3a.cpp`、`conversion/lfm3.py`、`tools/mtmd/mtmd-audio.cpp` (前処理)、`tools/mtmd/clip.cpp` (位置、マスク) から読んだ。HF のリモートコードと config の実物は見ていない。

- **テキスト部** (`internal/llm/lfm3`): `lfm3_audio` は LFM3 の設定が `lfm` キーの下にあり、テンソルは `lfm.model.` (llama.cpp は `lfm.` を取り除いて読む)、出力層は `lfm.lm_head.weight`。`lfm3_omni` は `text_config` の下で、テンソルは `model.language_model.`。d1-Omni-2B (`lfm3_omni` で `D1Model`) も 16.27 の d1 モードで変換できる。視覚エンコーダ (`lfm3vl`) は `lfm3_omni` の config も読む (テンソル名は LFM3-VL と同じ)
- **`internal/llm/lfm3audio`** (アーキテクチャ名 `lfm3-audio-encoder`、`-arch` で選ぶ): LFM3-Audio (`encoder.lfm` が LFM2 の設定、`dim_in`、`sliding_window`。テンソルは `encoder.downsample.`、`encoder.lfm.lfm.`、projector は `mlp.model.0/1/3`) と LFM3-Omni (`audio_encoder_config.text_config`、`num_mel_bins`。テンソルは `model.audio_encoder.downsample.`、`model.audio_encoder.lfm.`、`model.audio_projector.layer_norm/linear_1/linear_2`) の両方。構造は、(時間, メル) 上のストライド 2 の畳み込み 3 つで時間を 1/8 にする部分 (3×3 → SiLU、depthwise 3×3 → pointwise → SiLU を 2 回。カーネルは ggml の読み方で [出力][入力][メル][時間])、チャネル優先で平らにして linear、因果的な LFM2 の層 (短い畳み込みと、スライディング窓 (`sliding_window` 個のキー) の注意。QK norm、head 全体の rotate_half の RoPE、θ = 1e6)、最終 RMSNorm、projector (LayerNorm eps 1e-5、linear、GELU、linear)。FFN の幅は LFM2 の計算 (`block_auto_adjust_ff_dim` など)
- **固定の長さ**: プログラムは最長のメルフレーム数 (`-frames`、既定 3000 = 30 秒) ごとに変換する。入力は `x` (1, frames, 128) と `mask` (1, frames)。llama.cpp は畳み込みのたびに入力の後ろをゼロで埋めるので、各段の出力のうちクリップより後ろ (出力 t は入力 2t があるときだけ存在する) をマスクを半分ずつ間引いたもので 0 にする。後ろの層は因果的なので、クリップの分の出力は長いプログラムで動かしても同じ。出力の先頭 `ceil(ceil(ceil(frames/2)/2)/2)` 行がクリップの埋め込み
- **前処理** (`lfm3audio.Features`): llama.cpp の `mtmd_audio_preprocessor_lfm3a` と同じ。16 kHz、最後のフレームが最後の標本を含むよう右をゼロで埋め、400 標本の周期的 Hann 窓を 160 ごとに、パワースペクトル、Slaney のメルフィルタ 128 本 (0〜8 kHz、面積正規化)、log10 (下限 1e-12)、(v + 4) / 4。librosa (1.0.0) の STFT とフィルタで作った正解 (`tools/ref/mel_golden.py`) と 1e-4 以内で一致する。WAV (8/16/24/32 ビット PCM、32 ビット浮動小数点、チャネルは平均) を読み、16 kHz 以外は Hann 窓の sinc で変換する (llama.cpp は miniaudio で読むので、16 kHz 以外は近似)
- **`internal/audio`**: 長さごとのプログラムを読み込み、クリップが入る最短のもので動かす。どれにも入らなければ `ErrTooLong`。LFM3-Audio はプロンプトに埋め込みをそのまま置き、LFM3-Omni は `<|vision_start|>` と `<|vision_end|>` で囲む (llama.cpp の mtmd のとおり)
- **チャット**: `input_audio` 部分 (base64 の WAV) を画像と同じくメディアの印で受け、`Engine.Tokens` が埋め込みにする。`Content.Images` は画像と音声の両方を持つ `Content.Media` にした。`hwnpu-chat -audio`、対話では `/audio FILE.wav 質問`。LFM3-Audio の生の ASR (システムプロンプト "Perform live ASR." でアシスタントの番に音声とテキストを交互に置く llama.cpp の流し読み) と音声の出力は扱わない
- **d1-Omni-2B** (`systemone`): `files` の `data:audio/` URL と state の `input_audio` 部分は、画像と同じくユーザーの番の先頭の印になる。話された質問 (`instructions` が `{"type": "input_audio", ...}`) は、llama.cpp のとおり質問文の位置の印になり、その音声は要求のメディアの後に足す。いずれも `<|vision_start|>` と `<|vision_end|>` で囲む。音声エンコーダがなければ 501
- **検証**: `lfm3audio_test.go` に llama.cpp のグラフから書いた fp64 の正解 (畳み込みは PyTorch のストライド 2、パディング 1 の定義、窓は直近 `sliding_window` 個) を置き、37 フレームのクリップを 50 フレームのプログラムで (後ろをゼロにしてマスク) 動かした先頭 5 行が、LFM3-Audio と LFM3-Omni の両方の並びで相対誤差 1e-6 未満。1 層ずつと projector だけの実行も 1e-4 以内。改変 (カーネルの時間とメルの取り違え、マスクで 0 にしない: 2e-3、平らにする順、窓なし: 0.11、SiLU を外す、B と x の取り違え、LayerNorm を外す) をすべて検出した。`audio` のテストは、短いプログラムと長いプログラムで同じ埋め込みになることと長すぎるクリップの拒否を、`chat` のテストは LFM3-Audio と LFM3-Omni での音声のトークンの並びを、`systemone` のテストは話された質問の位置 (`TestSpokenTokens`) と、jinja2 で描画した正解 (`spoken question`) を確かめる
- 未確認: 実物の config (encoder の LFM2 の設定、`subsampling_conv_channels` は NeMo の既定 256 を仮定、projector の幅は `adapter_hidden_size` がなければテキストの幅を仮定)、テンソル名、`lin_out` の重みの形、実物での照合 (llama.cpp の `llama-mtmd-cli` の音声の埋め込みとの比較が確実)。LFM3-Audio のチャットテンプレートでの音声の置き方。Neural Engine での配置と精度、速さ

```bash
go run ./cmd/hwnpu-convert -arch lfm3-audio-encoder -frames 3000 -format fp16 -config d1-Omni-2B/config.json -weights d1-Omni-2B/model.safetensors -o audio-30s.mlpackage
go run ./cmd/hwnpu-systemone -model d1-Omni-2B -package d1-1024.mlpackage -vision vision-512x512.mlpackage -audio audio-30s.mlpackage request.json
```

### 16.30 LFM3-Audio の音声出力 (TTS) — 2026-10-30

16.27 の段階 (4)。構造と生成の手順は llama.cpp の `tools/mtmd/models/lfm3a-gen.cpp` (depthformer とコーデックのデコーダ)、`tools/mtmd/mtmd-helper-gen.cpp` (`lfm3a_gen_audio_pipeline`)、`tools/mtmd/mtmd.cpp` と `mtmd-audio.cpp` (極座標から ISTFT)、`conversion/lfm3.py` (テンソル名、TTS のテンプレート、サンプリングの既定値) から読んだ。

- **流れ**: テキスト部が TTS のプロンプト (`<|startoftext|><|im_start|>system\nPerform TTS in {言語} using a {female|male} voice.<|im_end|>\n<|im_start|>user\n{文}<|im_end|>\n<|im_start|>assistant\n<|audio_start|>`、LFM3-Omni は `<|audio_start|>` の前に `<think></think>`) を読み、最終 norm の後の隠れ状態 h を出す。1 フレーム (80 ms) ごとに、depthformer がコードブックを乱数の順に MaskGIT で決め (LFM3-Audio は 12 個を 4 個ずつ 3 回)、各コードを温度、top-k (k 番目と同点は残す)、繰り返しの罰則 (直近の窓でそのコードブックが使ったコード) で選ぶ。いずれかが終端コード (語彙の最後) なら終わり。そうでなければフレームのコードのフィードバック埋め込み (`codebook_embedding` の `コード + コードブック × 語彙` 行の和) をテキスト部の次の入力にして、次の h を得る。最後に、コーデックのデコーダがコードを STFT のフレーム (1 フレームあたり 6) にし、ホストの ISTFT が 24 kHz の標本にする
- **テキスト部**: `lfm3` に `-hidden-out 1` を足した。KV キャッシュ付きのプログラムがロジット y に加えて h (最終 norm の後) を出力し、`decode.Session.Hidden` がそれを読む。`TestBuildTiny` のダイジェストは変わらない
- **`internal/llm/lfm3gen`**: `lfm3-depthformer` は MaskGIT の 1 回分のプログラム。入力は x (コードブックごとに、確定したコードの埋め込みかマスクの埋め込み。ホストが表を引く) と h、出力は全コードブックのロジット。中身は h の射影 (`depth_linear`) と x の和、双方向の LFM3 の注意層 (QK norm、全ヘッドの RoPE θ=1e6、ヘッドごとのゲート、サンドイッチなし)、最終 RMSNorm、コードブックごとの RMSNorm、コードブックごとのコード表 (出力層と共有) との積 (バッチの matmul)。`lfm3-codec` はコーデックのデコーダ (`audio_tokenizer/`、`model_type` liquid_codec で自動判別)。入力は各フレームの FSQ のコードの値 (水準の表はホスト)、2 つの射影、各フレームを 6 位置に複製、8 層の因果的な LFM3 の層 (サンドイッチ norm、窓 30 の注意、短い畳み込み)、出力 norm、対数振幅と位相への射影。層の種類と幅は config にないので、重みの形から読む (llama.cpp と同じ)
- **固定の長さとチャンク**: コーデックのデコーダは `-codec-frames` (既定 64 フレーム = 5.1 秒) ごとに呼ぶ。llama.cpp は状態 (K/V の窓、畳み込みの窓、位置) を持ち越して 1 フレームずつ流すが、ここでは全フレームを生成した後に、デコーダの遡り (注意の層ごとに窓-1、畳み込みの層ごとに核-1 の位置) を覆うだけ重ねたチャンクで動かし、各チャンクの先頭 (ゼロの状態から計算した分) を捨てる。因果的で RoPE は相対位置にしか依らないので、流した場合と同じ結果になる (`TestDecodeCodecChunks` で、5 フレームずつのチャンクと 16 フレーム 1 回が 1e-4 以内で一致)
- **ISTFT** (`speech.ISTFT`): llama.cpp の `mtmd_audio_streaming_istft` と同じ。振幅 exp(対数振幅) と位相から複素数にし、負の周波数を共役で埋めて逆 DFT (1/N)、周期的 Hann 窓をかけて hop ごとに重ねて足し、窓の二乗の和で割る。先頭の (n_fft-hop)/2 と、最後の flush の後ろの同じ数を捨てる。同じ窓と詰め物で取った STFT から元の信号に 1e-4 以内で戻ることを確かめた
- **サンプリングの既定**: LFM3-Audio は温度 1.0、top-k 128、罰則 1.1、窓 4096 フレーム。LFM3-Omni は 0.4、top-k なし、1.2、64。`generation_config.json` の `audio_*` が上書きする。乱数は Go の PCG で、llama.cpp (mt19937) と同じ種でも同じ音にはならない
- **`internal/llm/runner.OpenSession`**: KV キャッシュ付きのプログラムを NPU (`chain`) か CPU (`milcpu`) で開き、`decode.Session` にする (CPU でもキャッシュのバッファを共有する)
- **`cmd/hwnpu-tts`**: 文を読み上げて 16 ビットの WAV に書く。`-lang` は model card の言語コード (en、ja など) か言語名、`-voice` は female か male
- **`hwnpu-convert` の設定名**: 設定のフラグは全アーキテクチャで共有されるので、コーデックのフレーム数は音声エンコーダの `-frames` と別の `-codec-frames` にした。同じ名前の設定の既定値が食い違わないことをテストする (`internal/llm/archs` の `TestParamNames`)
- **検証**: `lfm3gen_test.go` に llama.cpp のグラフから書いた fp64 の LFM3 の層 (ゲート付き注意、窓付きの因果注意か双方向、サンドイッチ norm、短い畳み込み) と、depthformer (LFM3-Audio と LFM3-Omni の両方の並び) とコーデックのデコーダの正解を置いた。相対誤差は 2〜8e-7。改変 (ゲートを外す: 0.52、窓を外す: 0.48、post_ffn_norm を外す: 1.4、コードブックの norm を外す: 0.029、コード表の転置: 1.1、複製の向き: 1.4) をすべて検出した。`speech_test.go` は、小さなテキスト部と depthformer を CPU で動かすフレームの生成 (最初のフレームのコードが手で計算した貪欲法の結果と一致する、確定したコードだけが埋め込みで入る、次のフレームはフィードバック後の h を読む、終端コードで止まる) をスパイで確かめ、確定したコードを戻さない、h を更新しないなどの改変を検出した
- **扱わないもの**: チャットの返答を音声にすること (llama.cpp はテキストと音声を 6 トークンと 12 フレームで交互に出す `<|mixed_start|>` の形と、`<|audio_start|>` で音声に切り替える形を持つ)、流し出し (フレームごとにすぐ音を出す)、LFM3-Audio の生の ASR
- 未確認: 実物の config (`depthformer` の項目、`codebooks`、`audio_vocab_size`、コーデックの `config.json`)、テンソル名 (llama.cpp の変換が読む名前に合わせた)、Neural Engine での配置と精度と速さ (1 フレームに depthformer 3 回とテキスト部 1 回。実時間に要るのは 12.5 フレーム/秒)、聞いて分かる音になるか。参照との照合は、llama.cpp のサーバー (LFM3-Audio の mmproj を読み込んだもの) の TTS と、同じコードからデコードした音の比較が確実

```bash
go run ./cmd/hwnpu-tts -model LFM3-Audio-2B -text tts-q1.mlpackage -depth depth.mlpackage -codec codec.mlpackage -lang ja -o out.wav こんにちは
```

### 16.31 LiquidAI の 4 モデルの実機での確認と、テキスト部の lut8 — 2026-10-30

16.27〜16.30 は Mac も Hugging Face もない環境で書いたので、M3 (16 GB) で実物の d1-2B、d1-Omni-2B、LFM3-VL-2B、LFM3-Audio-2B を動かした。

- **config とテンソル名**: 4 つの config.json と safetensors のヘッダーで、11 のプログラム (d1、視覚、音声エンコーダ、テキスト部、depthformer、コーデック) がすべて変換できた。推測だった接頭辞 (`model.language_model.`、`lfm.model.`)、音声エンコーダの `conv_channels` 256 と `lin_out` (768, 4096, 1)、projector の 768 → 2048 は実物どおり。変換が読まないテンソルは、ホストが引く埋め込みの表と `codebook_embedding`、`mask_embedding`、d1-Omni の depthformer、コーデックのエンコーダ側、それと `depth_embeddings.N.to_logits.weight` だけ。最後のものは HF のコードでも埋め込みと共有で、読み込み時に無視される (`_keys_to_ignore_on_load_unexpected`)
- **配置**: どのプログラムも全部 Neural Engine に載った
- **lut4 の損失**: d1-2B の README の例 (二重請求と未着の返金) で、lut4 は team を shipping (0.54) と答えた。fp16 (NPU でも CPU のインタプリタでも)、lut8 は README どおり billing (0.57〜0.62)、refund 1.00、urgency medium。lut6 は billing 0.52 対 shipping 0.42 で境目。LFM3-VL-2B のチャットは lut4 だと赤い円と青い四角の画像を「1000pink and blue icon」と言い、日本語が崩れた。lut8 では「白い背景に赤い円と青い四角形」と左右も正しい。LFM3-Audio-2B の TTS は、lut4 の音声を LFM3-Audio 自身の ASR ("Perform ASR.") に戻すと崩れ (「The heck to the other thing.」)、lut8 では「Please send the report to the finance team before Friday.」が一字一句一致し、日本語もほぼ正しい。そこでマルチモーダルと d1 のチェックポイントのテキスト部 (`lfm3.Config.Multimodal`) は既定を lut8 にした (`llm.Formatter`、`Arch.DefaultFormat(model)`)。lut8 は 1 GiB の上限を超えるので 2 部分になる。LFM3-1.7B はチャットで問題が出ていないので lut4 のまま
- **d1-2B** (lut8): 日本語の問い合わせ (ログインでクラッシュ) は technical 0.998。合成画像 (赤い円、青い四角) の色と左右をすべて当て、猫は 0.003。NPU と CPU のインタプリタの確率の差は 1e-3 程度。2 回目以降は文章 4 問で 0.13 秒、画像 4 問で 0.9 秒
- **d1-Omni-2B** (lut8): `say` で作った音声の話題、言及された都市、日本語の質問、話された質問 (「Is the customer happy with the delivery?」に対し、苦情の state で 0.05、感謝の state で 0.91) をすべて当てた。README の例は refund 0.94 が一致し、team は billing 0.45 対 shipping 0.52 で README (billing) と逆。fp16 を CPU で動かしても同じなので、量子化と Neural Engine の誤差ではない。HF の実装は質問を最大 `d1_set_size` (16) 問まとめて読み、llama.cpp に合わせたここでは 1 問ずつ読むので、その違いと見ているが、HF の実装とは比べていない (torch がない)
- **LFM3-Audio-2B**: 音声入力は "Perform ASR." で `say` の英語を一字一句、日本語を 1 字違いで書き起こした。HF の処理系も音声を `<|audio|>` の並びとしてその位置に置くので、16.29 の置き方で合っている。思考ありで温度 0 だと思考が繰り返して答えに届かないことがある。TTS は lut4 で実時間の約 1.4 倍、lut8 (2 部分) で約 0.5 倍の速さ
- **LFM3-VL-2B** (lut8): 1 トークン 41 ms、画像を含むプロンプト (約 300 トークン) の読み込み 0.3 秒
- **ディスク**: 読み込むたびに Core ML のコンパイル結果がキャッシュ (`~/Library/Caches/hwnpu`) にパッケージと同じくらい増える。試しの変換を何度も読み込むときは `-cache` で別の場所にするとよい

### 16.32 hwnpu-chat: 思考の上限 (-think-budget) — 2026-10-30

- 動機: OpenCode で「全く応答が返らない」ときのサーバーのログ (M3、LFM3-1.7B、文脈 16320) は、題名づくりの要求 (587 トークン、道具なし) が 3440 トークン考えて 226 秒、本題 (8136 トークン、道具 11 個) がその後ろで待ってから 4080 トークン生成して OpenCode の出力の上限 (`output: 4080`) で切れていた (`finish_reason: length`)。1 トークン 64 ms は設計どおりで、時間の大半は考えすぎと要求の直列化
- 実装 (`chat.Request.Think`、`chat.ThinkLimit`): 思考の中 (テンプレートの開く印と閉じる印の間、または `Render` が「思考の中で終わる」と言ったプロンプトの続き) で `Budget` トークン生成しても閉じないとき、`End` (「もう時間がない」の一文と閉じる印。`Template.Think` が与える。LFM3 は `</think>`、Gemma 4 は `<channel|>`) をモデルが生成したものとして系列に流し込み、答えに移らせる。流し込みは `step` を Q トークンずつ回す (prefill と同じ)。流したトークンは `emit` にも渡るので、ストリームは `</think>` を見る。考えないモードで (`<think></think>` の後に) モデルが勝手に `<think>` を開いた場合も、開く印から数えて同じ上限がかかる。`Stats.ThinkCut` で記録し、サーバーのログとターミナルの統計に `thinking cut at N tokens` と出す
- `-think-budget` (既定 1024) をサーバーとターミナルの両方に。`0` で無制限
- `TestThinkBudget` (CPU インタプリタ、小さな LFM3、予算 3): 3 トークン目の後に `End` が流れること、流したトークンが `emit` にも届くこと、終わったあとのロジットが prompt と生成列を最初から読んだときと相対誤差 2e-3 以内で一致することを確かめる
- 調べて問題でなかったもの: サーバーの 1 トークンごとのホスト処理 (全トークンの decode、Parse、ストリームの差分) は 4000 トークンの時点で 0.11 ms。Core ML のバッファの `SyncToDevice`/`SyncFromDevice` は no-op。サンプリングも 1 ms 未満。ホスト側に削るものはない
- 次の候補 (未着手): 投機的デコード (prompt lookup の下書きを Q=8 のプログラムで検証。1 回の呼び出しの費用はトークン数にほとんど依らないので、`docs/kv-cache.md` 3 節 (c) のとおり効く見込み)、題名づくりなど道具なしの短い要求の思考を切って上限を付ける、使っている塊だけ計算する注意 (`docs/kv-cache.md` 4 節 b)、system プロンプトの K/V をディスクに保存して初回の prefill (7.4K トークンで 13〜17 秒) を省く

### 16.33 hwnpu-chat: 投機的デコード (prompt lookup の下書きを 1 回で確かめる) — 2026-10-30

- 動機: ANE の 1 回の呼び出しの費用はトークン数にほとんど依らない (16.14 の Q=8 は Q=1 の +13%)。`docs/kv-cache.md` 3 節 (c) の「問い合わせの行を増やす」の実装。下書きは追加のモデルなしに、プロンプトと答えの中から作る (prompt lookup decoding: 直近 3 (なければ 2) トークンの並びが前に出てきた最後の場所の続きを、最大 k 個)。エージェントは読んだものを書くものに写すので、答えがプロンプトを長く繰り返す
- **1 回の呼び出しの費用** (`TestCallTimes`、M3、LFM3-1.7B lut4、キャッシュ 16320、`HWNPU_LFM3_DIR` で動く): Q=1 (2 部分) 65.2 ms、Q=8 (1 部分) 75.0 ms、Q=64 115.1 ms。Q=8 は Q=1 の +15% で、1 回に平均 0.15 個の下書きが当たれば元が取れる
- **プログラムの切り替えは高い**: Q=1 と Q=8 を交互に呼ぶと 1 組 404.6 ms (続けて呼ぶと 70.4 + 120.6 ms。この計測はスワップ 11 GB の状態で、単独の値も最初の計測より遅い)。Q=1 と Q=64 も 414.9 ms。ANE が重みを載せ直していると見ている。したがって下書きの有無で Q=1 と Q=8 を使い分ける (最初の実装) と、下書きがないときの呼び出しが 2 倍以上になる (本文 400 トークンで 371 回の呼び出しが 50 秒、1 回 136 ms)。**生成は 1 つのプログラムで行う**: `-draft 8` (既定) のときは Q=1 の代わりに Q=8 を変換して読み込み、1 トークンのステップもそれで動かす。読み込むプログラムは従来どおり 2 本 (生成と prefill)。prefill の Q=64 から生成のプログラムへの切り替えは今までどおり要求ごとに 1 回
- 実装:
  - `decode.Session.Speculate(emb, accept)`: `Step` を `run` (入力を書いて動かす) と `keep(v')` (先頭 v' 個だけキャッシュに書く) に分け、`accept` が各位置のロジット (`row(i)`、求められたときに読む) を見て何個残すかを返す。全注意層のキャッシュは位置より先がマスクで見えないので捨てる行は放置でよく、リングは i < v' だけ書き、畳み込みの窓は出力 `win_*_new` (W+Q 行) の v'..v'+W 行を取る。コピーは要らない
  - `chat.Engine`: `Draft()` は生成プログラムの Q−1 (Q=1 なら 0)。`LookupDraft` が下書きを作る。`Generate` は「本物 1 つ + 下書き k 個」を `speculate` に渡し、位置 i の下書きは位置 i−1 のロジットからサンプリングした値と同じなら残す (貪欲法では厳密に同じ出力。温度ありでも分布は変わらない: 残るのは下書きがサンプリングした値と一致したときだけで、位置 i のトークンはいつもその位置でサンプリングした値そのものになる。棄却サンプリングのように「違っても確率で残す」ことはしないので、当たりは少ないが補正は要らない。16.33 の最初の記述とマージ済みの PR #26 の本文で「近似」と書いたのは誤り。16.36 で訂正)。分かれた位置でサンプリングした値、または全部当たったときの最後の行の値が次の本物になる。残した下書きは 1 つずつ `take` (停止トークン、思考の上限の計数、`emit`、繰り返しの検出、上限) を通す。下書きの長さは答えの上限、キャッシュの残り、思考の予算の残りで切る
  - `Stats.Drafted / Accepted / Calls`。サーバーのログに `(a of d drafts accepted, c calls)`
  - `chat.Prepare` は `Setup.Draft > 1` で生成プログラムを `-seq Draft` で変換する (`lfm3-c16320-q8.mlpackage`、1 部分)。`hwnpu-chat -draft N` (既定 8、0 か 1 で従来の Q=1)
- 検証: `TestSpeculate` (CPU インタプリタ、小さな LFM3、Q=4 と 8 のプログラム): 答えを知っていて 3 回に 1 回わざと外す下書き係で、生成列が 1 トークンずつの生成と一字一句同じ、19 個の下書きのうち 13 個が残り 20 トークンを 7 回の呼び出しで生成、終了後のロジットが最初から読んだときと相対誤差 2e-3 以内。`TestLookupDraft` は 3-gram と 2-gram、一致なし、最後の出現を取ること
- **端から端の計測** (M3、同じ 2 つの要求を続けて。スワップ 11 GB の状態なので絶対値は当てにならず、同じ条件で続けて測った組だけ比べる):
  - 下書きあり (Q=8 のみ)、main.go (約 40 行) を直して全文を書き直す要求 (考えない): 900 トークンを 67.3 秒、483 回の呼び出し、1024 個の下書きのうち 417 個が残った (1 回 1.86 トークン、呼び出し 1 回 139 ms)
  - 下書きあり、「空はなぜ青いか」を考えさせる要求: 400 トークンを 50.3 秒、371 回、340 個のうち 28 個 (本文では下書きがほとんど当たらない)
  - 下書きなし (Q=1): 計測中。結果はこの節に追記する

### 16.34 埋め込みモデルも lut8 に、PLaMo-Embedding-1B の実機での照合と head-scale — 2026-10-30

- きっかけ: Intel 向けの検証 ([intel-npu-models.md](intel-npu-models.md) §11) で、文埋め込みは 4 ビットで崩れると分かった。fp16 の MIL プログラムとのコサイン類似度 (milcpu) は、PLaMo-Embedding-1B の lut4 が `uniform` で 0.20、hwnpu-chat の `kmeans-minmax` で 0.76、lut8 で 0.998。all-MiniLM-L6-v2 は lut4 で 0.23、lut8 で 0.9988
- 変更: `bert` と `plamo-bi` の `Arch.Format` を lut8 にした (hwnpu-convert の既定と hwnpu-chat の変換の両方)。PLaMo-Embedding-1B は lut8 でも 1 部分 (0.89 GiB) に収まる。hwnpu-chat の埋め込みのプログラムは名前に形式を入れた (`embed-lut8-s128.mlpackage`)。前の `embed-s128.mlpackage` (lut4) を読み込まず、変換し直させるため。古いものは消さないので、要らなければ手で消す
- **実機での照合** (M3、`tools/ref/embed_golden.py` の HF fp32 の `encode_query`/`encode_document` と `TestPlamoEmbeddingGolden`): 128 と 1024 トークンのプログラムは全演算が ANE に載り、トークン列は 8 文すべて一致。lut8 にしただけではコサイン類似度が 0.971〜0.989 で、2 文 (「日本の首都はどこですか？」0.978、吾輩は猫である 50 トークン 0.971) が合格条件の 0.98 に届かなかった。同じ 4 文を milcpu (fp32) で動かすと HF との類似度は fp16 の重みで 0.9986〜1.0000、lut8 で 0.9973〜0.9987 なので、グラフと lut8 は正しく、差は Neural Engine 側
- **層ごとの比較** (`TestANEPrecision`、`precision_darwin_test.go`): 16 層の残差の流れをすべて出力にした lut8 のプログラムを ANE と milcpu (fp32) で動かして比べた。吾輩は猫である (50 トークン、seq 64)。
  - 層 0〜2 の注意と MLP の出力は rms 0.01〜0.1 と小さく、ANE ではそれぞれ 3〜5% ずれる (行列積の絶対誤差の床、MODEL_ARCHITECTURE.md)。残差の流れの誤差は 2〜3% で頭打ちになり、平均した埋め込みにはほとんど残らない
  - 層 3 の MLP が BOS の 1 チャンネルに 1.2e3 の値を作り、以後の残差に残る (BOS の外では最大 1e2〜1e3)。RMSNorm の `x*x` は fp16 を超える値だが、ANE は崩れなかった (二乗の和を広い範囲で計算しているらしい)。milcpu の `RoundF16` は二乗で inf になり、層 4 以降は役に立たない
  - **最終層の MLP の出力が fp16 を超える**: `down_proj` の出力は fp32 で最大 5e4 (rms 2570) になり、ANE はトークン 1 のチャンネル 1731 などで -Inf を返した。その手前の silu(gate)·up は最大 1.9e4 で収まっている。埋め込みの ANE と CPU のコサイン類似度は 0.9745
- **head-scale** (`plamo-bi` の `Params`、既定 64): 最終層の出力は最終の RMSNorm にしか入らず、RMSNorm は入力の倍率に依らない (eps を除く)。そこで最終層で、注意を足した残差を 1/64 倍し、`down_proj` の入力 (silu(gate)·up) も 1/64 倍してから足し、最終 norm の eps を 64² で割る。2 のべき乗の掛け算なので数学的には同じ。重みを 1/64 倍する形も ANE では同じ結果だったが、小さな重みと int8 のスケールが fp16 の非正規化数になり、Intel 向けの int8 の IR が小さなモデルで許容誤差 1e-3 をわずかに超えた (0.00101) ので、入力側にした。最終 norm がないプログラム (分割の途中、`NoHead`) は残差の流れをそのまま返す決まりなので、倍率を掛けない
- 結果: 埋め込みの ANE と CPU のコサイン類似度は 0.9745 → 0.99986。HF とは 8 文すべて 0.9969〜0.9987 で `TestPlamoEmbeddingGolden` に合格 (milcpu の lut8 と同じ水準)。全演算 ANE。1 文は重みを縮めた版で 37〜41 ms、入力側の版で 72〜84 ms だったが、後者の回は読み込みも 1.5〜2 倍遅かったので、マシンの負荷とみている (未確認)
- 正解を作る環境: Python 3.12 に torch 2.15、transformers 4.47.1、sentencepiece 0.2.0、protobuf。`modeling_plamo.py` と `tokenization_plamo.py` を置いたディレクトリにチェックポイントをリンクして `HF_HUB_OFFLINE=1` で動かした (fp32 で約 1 分)
- 次: 長い文 (1024 トークンのプログラム) での最大値。最終層の 5e4 は文によってはさらに大きいかもしれない。`TestANEPrecision` で文を変えて確かめる

### 16.35 hwnpu-agent: Pi を手本にした 4 道具のコーディングエージェント — 2026-10-30

- 目的: OpenCode の最初の要求は system と道具 10 個で 7.4K トークン (16.17) と 16K の文脈の半分を使い、題名づくりの脇の要求で最初に 1〜2 分待つ。Go で組み込める成熟したエージェントのライブラリはない (kfet/agent は Anthropic の型でプロバイダーなし、pigo と og は internal に閉じた CLI) ので、Pi (pi.dev、4 道具、prompt 約 400〜500 語) のループを Go で書き、`internal/chat.Engine` を HTTP なしで直接呼ぶ `cmd/hwnpu-agent` にした。system prompt と 4 つの宣言で 629 トークン (OpenCode の最初の要求は 7,516)
- **`chat.Engine.Complete` / `Count`** (`internal/chat/complete.go`): Render → Tokens → Generate → Parse を 1 回にまとめた。サーバーの completions は stop 文字列と media の扱いが絡むのでそのまま。`TestComplete` は `TestEngineReuse` と同じ CPU のインタプリタの小さなモデルで `Generate` まで通す (Generate を通すテストはこれが初)
- **`internal/agent`**: `Model` インタフェース (`Complete`、`Count`、`Context`) を `*chat.Engine` が満たし、テストは台本どおりに答える偽物。`Session.Run` は user を足し、`Complete` → tool_calls を順に実行して `tool` の役 (`tool_call_id` 付き) を足す、を答えに呼び出しがなくなるまで繰り返す。assistant の考えは既定で会話に残さない (template.go は最後の user より後の考えを残すので、残すと 16K を思考で食う。`-keep-thinking` で残す)。文脈の予算は要求の前に `Count` を取り、`Context() - Reserve` (Reserve は `-n`、既定 2048) を超えたら古い `tool` の結果から順に先頭 1 行 + `[... N more lines omitted]` に畳み、最新 4 つは残す。要約のための生成はしない
- **道具**: 引数は string と integer だけ (LFM3 の `<parameter=k>値</parameter>` は文字列が安全で、配列や入れ子は小さなモデルには書けない)。read は 200 行/8KB で切って `[Lines a-b of n. Use offset=b+1 to read on.]`、行番号は付けない (edit の old に写されると壊れる)。edit は old が 1 回だけ一致したとき置き換え、0 回なら「read して正確に写せ」、2 回以上なら「周りの行を足して一意に」。old の前後の空行だけが違うときは除いて比べる (parse.go は値の前後の改行を 1 つ落とす)。write は親ディレクトリを作る。bash は `sh -c`、プロセスグループごと kill、既定 60 秒、末尾 100 行/8KB (Pi の 2000 行/50KB は 16K には大きすぎる)。write/edit/bash の前に `Config.Allow` (端末は y/N、`-yes` で省略、端末がなければ全部断る)
- **system prompt の方針**: 固定の短い核 + 道具ごとの規則 (道具を減らせば規則も減る) + 任意の AGENTS.md (4000 文字) の 3 層。課題ごとに書き換えず、変わるものは user メッセージに置く (Engine は共通の先頭を読み直さない。little-coder も skill の注入を会話の末尾に移している)。`-system FILE` と `-dump DIR` で prompt を差し替えて同じ課題の往復数とトークンで比べる
- **1.7B のモデルで学んだこと** (M3、lut4、「fix the bug in the add function of main.go」、`-yes`、30 往復まで):
  1. 作業ディレクトリの絶対パス (scratchpad の 100 文字超) を system に書くと、モデルはそれを `e08aada8-7f0f?...` のように写し間違えて 30 往復空回りした。パスを消して「Files in the current directory: go.mod, main.go」の一覧と「相対パスを使う」規則にした
  2. それでも `read` に `/main.go`、edit に `/main.go` と書く (OpenCode の絶対パスの癖)。「作業ディレクトリの外」と断ると 30 往復空回りしたので、外を指す絶対パスでもその名前のファイルがディレクトリにあればそれとし (`resolve`)、見つからなければファイル一覧を返す。これで read → edit → read → bash `go run main.go` → `5` まで 5 往復
  3. そのあと最終回答 (要約) を書いた末尾に `echo "done"` のような呼び出しを付けることを毎回繰り返し、止まらなかった (思考では「もう呼ばずに答える」と言っている)。内容が前の回答と 70% 以上同じ語で 20 語以上なら最終回答として呼び出しを捨てて終える (`similar`)。同じ呼び出し (名前と引数) が 2 回続いたら実行せず「同じ呼び出し。終わったなら道具を呼ばずに答えよ」を返し、3 回目で `ErrLoop`
  4. 次は `cd /tmp && go run main.go`、`cd / && ...` と見たことのないディレクトリへ移ろうとして `stat main.go: no such file` を繰り返した (23 往復、`ErrLoop` で終了、修正と確認は 8 往復目までに済んでいた)。外への `cd` で始まるコマンドは断って残り (`Run: go run main.go`) を示し、出力中の作業ディレクトリの絶対パスを `.` に置き換え、bash の規則に「do not cd」を足した
  5. 次は同じ `go run main.go` を 3 回繰り返し、要約なしで終わった (11 往復、修正と確認は 6 往復目まで)。繰り返しと往復切れのときは道具なしの要求を 1 回出して要約を書かせる (`close`: user に「Stop using tools. Answer now with a short summary…」を足す。道具の宣言がないので呼べない)。6 回目は 8 往復、53 秒で read → edit → read → (`cd /tmp` を断る) → `go run main.go` → `5` → 要約と、自分から終わった
- **little-coder から取り入れた 4 つのガード** (Pi の拡張約 90 個とコミュニティの拡張集を見た。人気はサブエージェント、許可ゲート、plan mode、git のチェックポイント、表示で、1.7B に効くのは little-coder の 27 個のガードの側。拡張の仕組み自体は TypeScript のプラグインで Go には合わず、`agent.Config` の関数と AGENTS.md、`-system`、`-tools` で足りる): (a) read していないファイルへの edit と write を断る (`Session.seen`、write したファイルは既知。`/reset` で忘れる)。(b) 思考の上限 `ThinkBudget` (既定 1024、`-think-budget`、0 で無制限): 16.32 の `chat.ThinkLimit` を `Complete` に渡し、エンジンが `</think>` を差し込んで答えさせる (`Stats.ThinkCut` を画面に出す)。最初は emit で生成を止めて思考なしで問い直す形で書いたが、main を取り込んだときに 16.32 の方式に合わせた (問い直しがなく、考えた分を答えに使える)。生成のプログラムも hwnpu-chat と同じ `-draft` (16.33) のものを使う。(c) bash の許可リスト `AllowList` (`-allow "go *,ls*"`): `*` が任意の文字列、`&&`、`||`、`;`、`|`、改行で分けた各部分が合うときだけ確認なし (`Allowed`、`glob`)。(d) output-parser (崩れた呼び出しの修復) は、LFM3 の XML が崩れた例がまだないので見送り。サブエージェント、plan mode、ターンごとの skill 注入は 1.7B には使いこなせないので見送り
- **検証** (M3、最終の形): 8 要求、最初の要求 629 トークンを 1.3 秒で読み、以後は新しい分だけ (0.1〜0.4 秒、`Reused` が前の要求のプロンプト全体)、生成 15〜16 tokens/s、全体 53 秒 (コンパイル済みの読み込み 2.4 秒を含む。キャッシュが冷えていると読み込み 14 秒、初回は Core ML のコンパイルで 100 + 65 秒)。OpenCode (16.17) は 12 往復、最初の要求 7.4K トークンを 13 秒。prompt と道具の変更は上の 1〜5 を 1 つずつ入れ、毎回同じ課題を `-dump` 付きで回して往復数と中身で比べた。同じ課題でも結果は揺れる (温度 0.1) ので、1 回ずつの比較は傾向として読む。4 つのガードを入れた後の 7 回目も 8 往復で通った (read → edit → read → `go run` → `go vet` → 要約。思考は最大 378 トークンで上限 1024 には触れず、最後は要約の繰り返しの検出で終わった)。main (16.32〜16.34) を取り込んだ後、`-draft 8` の投機的デコードありで 8 回目を回すと 19 往復で通った (修正と `go run` の確認は 8 往復目まで。そのあと `cd /tmp` や `cat /main.go` で迷い、要約の繰り返しで終了。温度 0.1 でも回ごとの揺れはこのくらいある)。生成は下書きが当たる要約の場面で 41 tokens/s、下書きの採用率は 2〜6 割。1 回だけ引数名が `<parameter=command` と崩れた呼び出しが出た (bash が「command is required」を返して次の往復で直った) ので、output-parser 相当の修復は候補に残る
- テスト (`internal/agent`、モデル不要): 台本の 4 往復 (tool の `tool_call_id` と内容、考えを残さない、ファイルの変更)、edit の 0 回/2 回/空行の違い/外/欠け、write の作成と上書きと外、read の offset/limit/バイト上限/binary/directory/欠け、`/main.go` の解決、bash の末尾切り詰め/終了コード/timeout/`pwd` が `.`/外への cd、予算の畳み (最新 1 つは残る、畳んでも収まらなければ `ErrContext`)、知らない道具、壊れた引数、`-steps`、`ErrCut`、繰り返す回答、繰り返す呼び出し、`Allow` の拒否 (実行されない)、キャンセルで課題の痕跡が残らない、`-dump`。`prompt_test` は 4 道具で 1500 文字以下、`-tools read,bash` で edit の規則が消える、AGENTS.md が 4000 文字で切れる、冒頭の差し替え
- 未確認: Gemma 4 E2B/E4B での動作 (テンプレートは tools に対応している)、`-think=false` との比較、Pi にある要約方式の compaction、崩れた呼び出しの修復 (output-parser)、同時に 2 つ以上の tool_calls を出したときの振る舞い (順に実行する実装は入っている)

### 16.36 hwnpu-agent で投機的デコードを確かめる (空いた機械での測り直し) — 2026-10-30

- 目的: 16.33 の端から端の計測はスワップ 11 GB の状態で、絶対値が当てにならなかった。16.35 の hwnpu-agent (Pi を手本にした 4 道具のエージェント) で、同じ課題「fix the bug in the add function of main.go」(`a - b` → `a + b`、約 40 行の main.go、`-yes`、思考あり、予算 1024、温度 0.1) を、下書きあり (`-draft 8`) → なし (`-draft 0`) → ありの順に、空いた機械 (スワップ 0.9 GB) で続けて回した。各回は `-dump` で要求ごとの `Stats` を残した
- **1 回の呼び出しの費用の測り直し** (`TestCallTimes`、同じ状態): Q=1 (2 部分) 61.5 ms、Q=8 72.0 ms (+17%)、Q=64 103.2 ms。Q=1 と Q=8 を交互に呼ぶと 1 組 202.9 ms (続けて呼ぶ 133.5 ms より 69 ms 多い: 切り替え 1 回 35 ms。16.33 のスワップ下では 1 回 100 ms 超だった)。生成を 1 つのプログラムで行う判断は変わらない
- **結果** (3 回とも add を直し、`go run` で確かめて、要約で終わった。往復の数は温度 0.1 の揺れで 11〜28 回と大きく違うので、課題全体の時間ではなく生成の速さで比べる):

| | 往復 | 生成トークン | 生成の速さ | 呼び出し | 1 回あたり | 下書きの採用 |
|---|---|---|---|---|---|---|
| 下書きあり 1 回目 | 28 | 5690 | 42.2 ms/token (23.7 tokens/s) | 3279 | 1.74 トークン | 2445 / 7338 (33%) |
| 下書きなし (Q=1) | 17 | 2385 | 62.7 ms/token (15.9 tokens/s) | 2385 | 1.00 | — |
| 下書きあり 2 回目 | 11 | 1609 | 52.8 ms/token (19.0 tokens/s) | 1215 | 1.32 トークン | 412 / 1986 (21%) |

  - 課題全体では下書きありが 1.2〜1.5 倍速い。最初の往復 (read を呼ぶまでの思考だけで、写すものがない) は 75.1 / 66.0 / 71.9 ms/token と、下書きありが Q=8 の 1 回の費用の分 (+15% 前後) だけ遅い。read した本文を edit に写す往復は 23.8 tokens/s (763 個のうち 283 個が当たり) 対 15.9。要約を書く往復は 27〜34 tokens/s
  - 下書きの採用率は往復ごとに 1 割から 5 割の幅。道具の呼び出しの XML (`<tool_call><function=bash><parameter=command>`) と `go run main.go` のような繰り返しの命令、前の回答を言い直す要約で当たる
  - 思考の上限 1024 には 3 回とも触れなかった (最も長い思考でも数百トークン)
  - 28 往復の回は、修正と確認を 8 往復目までに終えたあと `bash` を 17 回 (同じような `go run` と `cat`) 続けて要約の繰り返しで終わった。16.35 の 5 と同じモデルの振る舞いで、下書きとは関係ない (温度ありでも分布は変わらないので、下書きが繰り返しを増やすことはない。上の訂正)
- 結論: `-draft 8` の既定は妥当。写す場面で 1.5〜2 倍、課題全体で 1.2〜1.5 倍、写すものがない思考だけの場面で 1 割強の損。損を減らすには Q=4 (1 回の費用は Q=1 と Q=8 の間と見込む) か、下書きがないときに Q=1 のプログラムに戻す (切り替え 35 ms なので、下書きの有無が続けて切り替わる場面では今より遅い) のどちらかで、どちらも未計測
- 次: Q=4 での費用と採用率。OpenCode の往復 (system と道具で 7.4K トークン、本文の XML が長い) での採用率。温度 0 (貪欲法) との比較 (当たりが増えるはず)

### 16.37 KV キャッシュの容量の段階をチャットのエンジンに (-tier)、下書きの幅と n-gram の見直し — 2026-10-30

- 動機: 1 ステップの費用はキャッシュの容量に比例し、使った量には依らない (16.14〜16.16、`docs/kv-cache.md` 3 節)。hwnpu-agent の会話は 635 トークンで始まり課題の終わりで 4〜5K なので、16320 の容量で払っている分が大半。段階 (`hwnpu-convert -cache a,b`、`chain.Grow`、`decode.Session.Migrate`、16.20) は `hwnpu-generate` にはあったが、チャットのエンジン (2 つのプログラムが 1 つのキャッシュを共有し、印と退避で前のプロンプトを使い回す) には入っていなかった
- **1 回の呼び出しの費用** (`TestCallTimes`、`HWNPU_CALL_PKGS` で任意の組を測れるようにした。M3、空いた機械):

| プログラム | 1 回 | 読み込み (コンパイル済みキャッシュなし / あり) |
|---|---|---|
| c4096 Q=4 | 28.5 ms | 33 s / — |
| c4096 Q=8 | 28.3〜29.4 ms | 20〜35 s / — |
| c4096 Q=64 | 36.9 ms (1730 tokens/s) | 36 s |
| c16320 Q=4 | 66.9 ms | 30 s |
| c16320 Q=8 | 68.5〜69.1 ms | 16 s / 1.5 s |
| c16320 Q=64 | 100.8 ms (635 tokens/s) | 19 s |

  - 段階の切り替え (c4096 Q=8 ↔ c16320 Q=8 を交互): 1 組 126.8 ms、続けて呼ぶ 96.8 ms より 30 ms 多い (切り替え 1 回 15 ms)。会話が伸びて段階を上がるときに 1 回払うだけ
  - **常駐のメモリ**: 2 プログラム (c16320 Q=8 と Q=64) で最大 RSS 2.29 GiB、4 プログラム (c4096 の 2 つを足す) で 2.73〜2.96 GiB。重みはファイルの写像なので倍にはならない。計測の間にスワップは 0.9 GB から 1.4〜1.7 GB に増えた (16 GB の M3、ほかのアプリ多数)
- **Q=4 は見送り**: Q=4 の 1 回は Q=8 より 2 ms (3%) 安いだけ。dump の再生 (下記) で下書き 3 個は 7 個より 1 回あたり 10% 少ないトークンしか得られないので、Q=8 のまま
- **実装** (`chat.Config.Tiers`、`chat.Setup.Tiers`、`-tier` (既定 `4096`、空で無し)):
  - `chat.Prepare` は段階ごとに生成 (Q=8) と prefill (Q=64) のプログラムを変換する (`lfm3-c4096-q8.mlpackage` など)。`Load` は段階ごとに 2 つの鎖 (`chain.Open` と `Share`) を開き、全部を常駐させる (16.20: 裏で読み込むと生成が半分以下に落ちる)
  - 系列は、読み直しになったとき (共通の先頭がない、または初回) にプロンプトが収まる最も短い段階で始める (`tierFor`、`use`: その段階のプログラムで `decode.New` + `Add` した新しい `Session`)。prefill と生成の各ステップの前に `room(n)` が容量を確かめ、足りなければ次の段階へ `Migrate` (全注意層の行を写す。4096 行で数十 ms) し、prefill のプログラムを `Add` し直す。印 (リングと窓だけ) は段階をまたいでも有効 (`Migrate` が行を同じ位置に写すため)。退避した系列 (`Save`) はキャッシュ全体の写しなので段階を記録し (`saved.tier`)、戻すときにその段階の `Session` を作ってから `Rewind` する。`Context()` は最も長い段階の容量。`Stats.Cache` に生成が終わった段階を入れ、エージェントとサーバーのログに `cache N` と出す
  - `TestEngineTiers` (CPU インタプリタ、小さな LFM3、12 / 24 / 48 の 3 段階、Q=1 と 4): 短いプロンプトは最初の段階で始まり、生成で容量を超えると上がり、続きの要求は上の段階でキャッシュを使い回し、さらに上がり、新しい短い会話は下に戻り、退避した長い会話は自分の段階に戻る。各点のロジットは 48 で最初から読んだときと相対誤差 2e-3 以内。どの段階にも収まらないプロンプトは `ErrContext`
- **hwnpu-agent での結果** (16.36 と同じ課題、`-draft 8`、段階あり → なし → ありの順、空いた機械。3 回とも add を直した。なしの回は 30 往復の上限で終わった (`-steps`、修正と確認は済んでいた)):

| | 往復 | 生成トークン | 生成の速さ | prefill 合計 | 下書きの採用 |
|---|---|---|---|---|---|
| 段階あり (4096 → 16320) 1 回目 | 25 | 5257 | 26.5 ms/token (37.8 tokens/s) | 6.7 s | 32% |
| 段階なし (16320) | 31 | 6385 | 51.6 ms/token (19.4 tokens/s) | 15.2 s | 26% |
| 段階あり 2 回目 | 26 | 6650 | 26.8 ms/token (37.4 tokens/s) | 6.0 s | 19% |

  - 課題全体で **1.95 倍**。4096 の段階にいる間は往復ごとに 35〜80 tokens/s (写す往復で 77〜80)、最初の往復 (思考だけ) も 28 対 14 tokens/s。約 3800 トークンで 16320 に上がり、以後は段階なしと同じ 18〜27 tokens/s。prefill も 2 倍以上速い (最初の 635 トークンが 0.7 秒)
  - 段階は投機的デコードと独立に効く (呼び出し 1 回の費用そのものが下がる)。16.33 の推定どおり
- **prompt lookup の n-gram** (dump の再生、56 要求 9684 トークン、k=7。再生は 16.36 の実測 (3279 回、1.74 トークン/回) と一致する): 3..2-gram (最後の出現) 1.552 トークン/回 → 3..1-gram 1.670 (+7.6%)、4..1-gram 1.673。4..2 と 5..2 は 1.554〜1.556、最初の出現を取る形は 1.550。下書きの検証は Q=8 のプログラムでは無料なので、当たりの少ない 1-gram の推測でも出したほうが得。`LookupDraft` を 4..1 にした (`TestLookupDraft` に 1-gram の例)
- 見送ったこと: 段階を 1 つのパッケージの複数の関数にして重みを共有する形。`BlobWriter` は定数を重複なく書く仕組みを持たず、2 段階で重みが 2 回入って 1 GiB の上限 (16.13) を超える。定数をハッシュで束ねれば試せるが、ANE が関数の切り替えで重みを載せ直すかどうかも未知。段階ごとの重みの重複 (RSS +0.5〜0.7 GiB) が問題になるときの候補
- 次: 12288 などの中間の段階 (OpenCode の 8K のプロンプト向け。費用は容量に比例するので 25% の見込み。常駐のメモリとの取引)。使っている塊だけ計算する注意 (`docs/kv-cache.md` 4 節 b) は、塊の数ごとのプログラムが要るので段階と同じ重みの重複を伴う。system プロンプトの K/V をディスクに保存して初回の prefill を省く (OpenCode の 7.4K で 13〜17 秒)


### 16.38 プロンプトのキャッシュをディスクに (prompt store、`-prompt-cache`) — 2026-11-01

- 動機: 16.32 と 16.37 が挙げた「system プロンプトの K/V をディスクに保存して初回の prefill を省く」。OpenCode の system と道具 (7.4K トークン) は hwnpu-chat を起動し直すたびに読み直していた (M3 で 13〜17 秒)。退避した系列 (`Save`、16K で 105 MB × 2) をプロセスの外に出す道の土台にもなる (今回は `Save` のまま)
- `decode.Session`: 全注意の層の行は位置ごとで、段階 (容量) や塊に依らない (`Migrate` が頼っている性質)。`ExportRows(from, to)` は位置順に、長い (リングでない) キャッシュを名前順に、ヘッドごとの行を fp16 で詰めて返し、`ImportRows` はどの段階のセッションにも同じ位置に戻す (`RowBytes` は 1 位置のバイト数)。リングと畳み込みの窓は行ではなく `Mark` が持つので、`Mark.Bufs` と `NewMark` を足した。`Resume(rows, mark)` は、印の名前がそのセッションのリングと窓と過不足なく合い大きさも同じこと、行の長さが印の位置とちょうど合うことを確かめてから、空にしたセッションに置く
- `chat.store` (`internal/chat/store.go`): モデルごとのディレクトリ (`storeKey`: 名前、最も長い段階のパッケージ名と更新時刻、各部分の重みのバイト数。変換し直すと別になる) に、系列ごとのディレクトリ (`meta.json` にトークン列、行の数、印の位置、最後に使った時刻。`rows.bin` に全位置の行。`mark-POS.bin` にリングと窓とロジットを gob で)。`find` は新しいプロンプトと共通の先頭が最も長い系列の、その先頭の中の最後の印。`load` は印までの行と印。`put` は、系列が項目の続きなら行と印を追記 (1024 位置ずつ書き出すのでメモリに全部は持たない)、項目の途中で分かれたら新しい項目 (共通部分の印は写す)、項目の先頭だけなら足りない印だけ足す。1024 トークン未満の系列は保存しない (1 秒で読める)。モデルごと 4 系列で、最も使われていないものから消す。書き込みは一時名からの改名で、途中の状態は見えない
- `chat.Engine.prefill`: メモリ内の再利用 (印と退避) より先へ進めるときだけストアから戻す。プロンプトが収まる最も短い段階の空のセッションに `Resume` し、`cur` をその系列の先頭に、`base` をその項目にする。読みながら 512 トークンごと (`markEvery`) と終わりに印を取り、読み終えたら `put`。読めない項目 (ファイルが壊れている、印の形が合わない) はその場で消して読み直す。`Stats.Restored` に戻したトークン数 (`Reused` に含まれる)。サーバーのログは `(N tokens from the prompt store)`、`usage.cached_tokens` には含まれる
- `-prompt-cache DIR` (hwnpu-chat と hwnpu-agent。既定はコンパイルキャッシュの `prompts`、`off` で無し)。`chat.Config.PromptDir`
- 検証 (CPU インタプリタ、小さな LFM3): `TestEngineStore` (印 8 トークンごと、保存は 8 トークン以上、4 項目で 2 本まで): 再起動後に 20 トークン共通のプロンプトが印 16 から戻って残りを読む。分かれた系列は自分の項目を持ち、共通部分の印 (8、16) を引き継ぐ。続きで行と印が増える。再起動後の同じプロンプトは何も読まない。3 本目で最も古い項目が消え、消えた項目の先頭は分かれた系列の項目から戻る。短いプロンプトは保存しない。各点のロジットは最初から読んだときと相対誤差 2e-3 以内。`TestResume`: 別の系列の行では誤差が 1e-2 を超え、行の不足、印の大きさの違い、印の欠けはエラー
- 計測 (M3 16 GB、LFM3-1.7B lut4、文脈 16320、段階 4096/16320、2026-11-01): OpenCode 1.19.5 の最初の要求 (system と道具 10 個と 1 行の依頼で 7,854 トークン) を `-dump` で取り、`-listen` のサーバーに再生して測った。冷えた読み込みは `-prompt-cache off` で 14.6 秒、空のストアに書きながらで 15.1 秒 (書き込みの上乗せは 0.5 秒以内。OpenCode から直接のときは 16.0 秒)。再起動後の同じ要求は 7,854 トークンすべてがストアから戻って 0.2 秒 (ストアの読み出し自体は 96 MB で 17〜36 ms、残りは ANE のバッファへの書き戻し)。依頼の文を変えた要求は印 7680 から戻って 177 トークンを読み 0.7 秒。貪欲法 (temperature 0) の 64 トークンは、冷えた読み込みと復元後で完全に一致した。大きさは、行が 1 トークン 12,288 バイト (見込みどおり)、印が 1 つ 3.45 MB (見込みの 2〜3 MB より大きい。23 のバッファとロジット 128,000 個)、7,857 トークンの項目が 145 MB (印 16)、1 往復進んだ 8,218 トークンの項目が 164 MB (印 17)。512 ごとの印で約 18.7 KiB/トークン、ほかに要求ごとの終わりの印が 3.45 MB ずつ増える (20 往復で 70 MB)。OpenCode は会話ごとに依頼の文で分かれるので会話ごとに 145 MB の項目ができ、4 本で約 600 MB。hwnpu-agent は system と道具で 617 トークン、1 往復の会話が 682 トークンで 1024 に届かず、保存も復元もしない (読み込みは 0.8〜0.9 秒)。Gemma 4 E2B は未計測
- 残り: 退避した系列 (`Save`) もストアに置いてプロセスから外す。要求ごとの終わりの印は古いものを間引く (512 ごとの印があれば足りる)。int8 で半分にする (写しには費用がない)

### 16.39 M3 での裏取り: プロンプトストアの段階と壊れたファイル、Q=16、grouped 注意、lut6、メモリ — 2026-11-02

- 目的: 16.38 と `docs/model-fit.md` はクラウドで書き、実機の数値はすべて見込みだった。そのチェックリスト (ブランチとテスト、ストアの端から端、壊れたファイル、Q=16 の費用、lut6、`SDPAGrouped`、メモリ) を M3 16 GB で通した。ストアの端から端の数値は 16.38 の計測の行、それ以外はここ
- ストアの段階と分岐: 1,524 トークンの要求 (OpenCode の system の先頭 7,000 文字と 1 行) は、冷えた初回でも 7,854 トークンの項目と共通する先頭の印 1024 から戻り (16320 で読んだ行が 4096 の段階に入る)、残り 500 を読んで 0.5 秒。再起動後は 1,524 すべてが戻って 0.0 秒。貪欲法の 48 トークンは冷えた読み込み、復元、`-prompt-cache off` の 3 通りで一致
- 壊れたファイル: 印ファイルを消した項目は最初の使用で `(dropped)` と出て消える。rows.bin を切り詰めた項目は起動時に数えられなくなる。ここで 2 つ直した。(1) 落とした後に次善の項目を試さず、印 1024 を持つ別の項目があっても全部読み直していたので、`prefill` は見つからなくなるまで繰り返す。(2) 読めない項目のディレクトリ (64 MB) がディスクに残り続け、4 本の上限にも数えられなかったので、`openStore` が消す。meta.json の無いディレクトリは別のプロセスが書いている途中かもしれない (`put` は meta.json を最後に書く) ので、1 時間 (`staleAfter`) より古いものだけ消す。`TestEngineStore` に両方のケースを足した (修正なしでは落ちる)
- Q=16 (model-fit.md T3): `-seq 16 -cache 16320` を変換して `TestCallTimes`: Q=8 69.2 ms、Q=16 73.4 ms (+6%)、Q=64 109.0 ms。基準の +10% 以内なので、生成と prefill を Q=16 の 1 本にする案は呼び出しの費用の面では成り立つ (prefill は 16 ÷ 0.0734 ≈ 218 tokens/s で 7.9K が 36 秒。今の Q=64 は 14.6 秒)。交互の呼び出しは Q=8/Q=64 で 224 ms (単独の和 178 ms、切り替え 23 ms)、Q=8/Q=16 で 253 ms
- grouped 注意 (T1): `lfm3/build.go` のキャッシュ付きの注意に `-attn grouped` (`blocks.SDPAGrouped`。既定は `tile`) を足した。`TestCachedMatchesFull` に grouped の場合を足し、CPU で fp16 と 2e-3 以内。M3 の c16320: Q=8 の 1 回が 73.5 → 40.7 ms (−45%、見込みは 38〜50 ms) で「all on the NPU」のまま。chat では 4 問の貪欲法の答えが tile と一字一句同じで 18〜24 → 27〜39 tokens/s。OpenCode の 7,854 トークンの prefill (Q=64 も grouped) は 14.6 → 10.9 秒、その後の生成は 60 → 34 ms/token、64 トークンの文は tile と一致。`ChunkedSDPA` (16320 超) は未対応。既定にするかは未決 (パッケージの名前が同じなので、変えるなら名前に入れる)
- lut6 (T7): `-format lut6` の c16320 (Q=8 と Q=64、各 2 部分 1.21 GiB) は NPU に載り、初回の読み込みは 64 + 112 秒。4 問 (貪欲法、思考なし、質問ごとに `/reset`): 富士山の高さは lut4「約 4,000 m」/ lut6「3,776 m」。17 × 23 はどちらも 391。『こころ』の作者は lut4「村上春見」/ lut6「宮本明」。日本一長い川は lut4 答えず / lut6「利根川」。4 問中 1 問が直った。速さは lut4 18〜24 tokens/s、lut6 12〜15 tokens/s (下書きの採用率込み)。KL (ユーザーが測定。`hf download` でリモートコードを取り、`ref_logits.py --seq 32` に長い英文を渡し、`hwnpu-check -package` を分割なしの 2 本に): lut6 は rel RMS 0.274、cosine 0.963、argmax 23/32、平均 KL 0.232 nats。lut4 は rel RMS 0.617、cosine 0.816、argmax 16/32、KL 1.387 nats。lut6 は見込みの 0.2〜0.3 に収まり、lut4 の 1/6。lut4 の 1.39 は MODEL_ARCHITECTURE.md の 0.89 (24 トークン、別の文) より大きく、文と長さで変わる。`ref_logits.py` には LFM3 の `modeling_lfm3.py` (リポジトリのリモートコード。hwnpu のダウンローダは取らない) が要る。既定の文は LFM3 のトークナイザで 30 トークンなので、`--seq 32` には `--text` で長い文を渡す。`hwnpu-check -package` は分割なしの 1 つの .mlpackage を CPU インタプリタで動かすので、`-target none` で変換したものを渡す
- メモリ (T10、`footprint`、静かな機械、読み込み直後): 2 本 (`-tier ""`) 1,968 MB、4 本 (既定) 2,807 MB。RSS は 1.08 と 2.36 GiB。7.9K の要求の後は footprint が 791 と 847 MB に落ちる一方 RSS は 1.57 と 2.03 GiB で、重みの載り方 (Core ML の mmap と共有) は footprint では追えない。aned の footprint は sudo が要り未計測
- テスト: `go test ./internal/llm/... ./internal/chat/ ./cmd/...` と `go vet ./...` は通る。hwnpu-agent の 2 回目はストアに触らない (16.38)
- 残り: lut6 を既定にするか (KL は 6 分の 1、速さは 12〜15 対 18〜24 tokens/s、重み +0.4 GiB。既定を変えるなら `chat.Prepare` の名前に形式を入れる)。grouped を既定にする (名前に入れる、`ChunkedSDPA`、Gemma 4 と同じ扱い)。Q=16 の 1 本化は grouped の後で測り直す (切り替えの費用も変わる)。OpenCode の `run` は初期化に 45 秒かかってから最初の要求を送る (hwnpu の外)

### 16.40 検索の道具と、最初の往復を検索にする強制 (hwnpu-agent `-tools search -search-first`) — 2026-11-02

16.39 で lut4 の LFM3-1.7B は 4 問中 3 問 (富士山の高さ、『こころ』の作者、日本一長い川) を誤り、lut6 でも 1 問しか直らなかった。知識をモデルの中で解決せず、検索の道具で外に出す案を M3 で試した。

- 下調べ (hwnpu-chat のサーバーに `web_search` を宣言し、結果は手で与える): 事実の 3 問はどれも妥当な検索語 (「富士山 高さ」「『こころ』 作家」「日本 最長 川」) で呼び、17 × 23 と Go の関数は呼ばない。結果を返すと 3 問とも正解で出典の URL も付け、答えの無い結果 (渋川市の人口) には「見つからなかった」と言う。思考なしで 1 往復目 0.7〜3 秒、2 往復目 1〜4 秒、結果は約 200 トークン。思考ありは 10〜23 秒で、『こころ』は結果に夏目漱石とあるのにローマ字化に迷って再検索した。system prompt の「Answer in the language of the question」を「Always answer in Japanese」に変えただけで『こころ』と渋川市で検索を呼ばなくなり、『こころ』は「村上 平作」と捏造した (温度 0 で再現する)。検索の判断をモデルに任せてはいけない
- 道具 (`internal/agent/search.go`、`tools.go`): `search(query)` は `Searcher` の裏側に `Wikipedia` (MediaWiki API、鍵なし。検索の上位の記事の冒頭 2 文を extracts で取る)、`SearXNG` (JSON)、`Brave` (API、`BRAVE_API_KEY`)。DuckDuckGo の html と lite はこの機械からの GET も POST も 202 の anomaly (ブラウザの検査) で、入れていない。SearXNG の docker は取得をこのセッションの権限が拒んだので httptest だけで確かめた。search だけを出すと system prompt は `SearchSystem` (調べて答えるアシスタント) になり、ディレクトリの一覧やパスの規則は入らない (255 トークン)
- 強制 (`Config.SearchFirst`): 最初の案は「モデルに検索語を書かせてから検索し、その呼び出しと結果を会話の先頭に置く」だったが、思考なしで検索語だけを書かせると質問の写しや解説が出て (例つきでも同じ)、100 文字で切った検索語では何も見つからなかった。道具を宣言した普通の往復では検索語は良い (6 問中 4 問が呼び、17 × 23 は呼ばず、呼ばなかったのは 1 問) ので、最初の往復はそのまま行い、search を呼ばなかったらその往復を捨てて課題の 1 行目で検索した呼び出しと結果を置く形にした。モデルが同じ検索語で呼び直すと nudge で止まる
- 検索の上限 (`Config.Searches`、既定 4): 上限なしでは答えが前にあっても検索を続け (富士山は正解の記事が 1 回目にあったのに 3 回)、『こころ』は 30 往復で 14K トークン、5 問で 10 分かかった。上限を越えた呼び出しには「もう検索はない、手元の結果から答えよ」を返し、その往復の後に道具なしの要求で答えを求める (`ErrSearches`)。上限ありで 5 問 146 秒 (読み込み込み)
- M3 の結果 (lut4、思考なし、Wikipedia、`-search-first`): 富士山は 1 回の検索で「約 3,775.56 m (最高地点 3,776.12 m)」と出典つき (約 3 秒)。『こころ』は Wikipedia の検索が「『こころ』 作者」「心 作家」で小説を返さず、4 回で「見つからなかった」。日本一長い川は「日本 最長の川」にドナウ川が返り (生の質問文「日本一長い川は?」なら「長さ順の日本の川一覧」が出る)、4 回使い切り。渋川市は名前を「Shigyū」とローマ字化して検索し見つからず、答えられないと言った。17 × 23 は検索してから 391。つまり道具の枠組みは動き、残る弱さは Wikipedia の検索の当たりの悪さで、ウェブの検索 (SearXNG か Brave) で測り直すのが次
- テスト: `TestWikipedia`、`TestSearXNGAndBrave` (httptest)、`TestSearchTool`、`TestSearchFirst` (置き換え、nudge、道具が無いときは検索しない、上限)。`go vet ./...`、`go build ./...`、`go test ./internal/agent/ ./cmd/...` は通る
- 残り: SearXNG か Brave での測り直し (鍵か docker が要る)。Wikipedia の検索語の整形 (括弧を外す、生の質問文も試す)。思考ありの答えは内容がきれいだが遅い: 思考なしだと「The user is asking...」が本文に出る。ページ本文を読む `fetch` は未実装 (1〜3K トークン、lut4 の prefill 540 tok/s で 2〜6 秒の見込み)

#### 16.40 の追記: fetch、Wikipedia の検索語、思考あり/なし、grouped の Q=16 — 2026-11-02

- 試せなかったもの: SearXNG の docker (`docker run searxng/searxng`) と LFM3 のリモートコード (`hf_hub_download` で `modeling_lfm3.py` など) は、どちらもこのセッションの自動モードの分類器が拒んだ (Code from External、Auto-Mode Bypass)。ウェブ検索の実走と lut6 の KL はユーザーの手で
- `fetch(url)` (`internal/agent/fetch.go`): ページを文章にして先頭 `Config.FetchChars` (6000) 文字を返す。script、style、head、コメントを落とし、ブロック要素を改行に、タグを除き、実体を戻し、空行を捨てる。Wikipedia の日本語のページは 6000 文字で約 4,200 トークン (prefill 3.8〜4.0 秒)。規則は「抜粋に答えが無ければ fetch で読む」
- Wikipedia の検索語: モデルは『』「」で題名を囲み、検索はそれを文字通りに取るので外す。語を並べて 0 件なら最初の語 (名前のことが多い) だけで引き直す。結果の URL は百分率符号化せず題名のまま (符号化した URL をモデルが写すと崩れ、`E7%A6%9F5B8%82` のような検索語になった)
- M3、lut4、Wikipedia、`-tools search,fetch -search-first`、同じ 5 問。思考なし 201 秒: 富士山は 1 回目の検索で記事の冒頭に標高があったのに fetch し、4.6K トークンのページから「富山県・静岡県」「3,876 m」と「3,775.56 m」を行き来する 1,500 トークンの答え (最後は 3,775.56 m)。『こころ』は見つからず、川は中国語で「答えられない」、渋川市はページを読んで数字を出さず、17 × 23 は 391。思考あり 190 秒: 「Mount Fuji height」と英語で検索し英語版を fetch して 3,776.24 m と出典、『こころ』は検索を使い切り (`ErrSearches`)、川は「longest river in Japan」→ Shinano River の記事を fetch して信濃川、渋川市は日本語版を fetch して 69,173 人 (ページの推計人口。ただしページの日付は 2026 年 10 月 1 日で、質問の 2025 年の値として答えた)、17 × 23 は 391。思考ありが 5 問中 4 問 (日付の取り違えを含む)、なしが 2 問、search だけ (前項) が 1 問。思考ありで英語版に行くのは、検索語を英語で書くと `Wikipedia.Lang` の判定 (かなと漢字で ja) が en を選ぶため。英語版の検索は語の並びに強い
- grouped の Q=16 (16.39 の残り): `-seq 16 -attn grouped` を変換し `TestCallTimes` (`HWNPU_CALL_PKGS` に grouped の 3 本): Q=8 41.4 ms、Q=16 47.8 ms (+15%)、Q=64 82.4 ms。交互は Q=8/Q=64 159.5 ms (和 123.8 ms、切り替え 1 回あたり約 18 ms)、Q=8/Q=16 220.1 ms (和 89.2 ms。Q=16 は直前にコンパイルしたばかりで、測り直しが要る)、Q=16/Q=64 166.9 ms。Q=16 の 1 本化は tile の +6% から +15% に広がり、prefill は 16 ÷ 0.0478 ≈ 335 tokens/s で 7.9K が 23.5 秒 (Q=64 grouped は 10.9 秒)。切り替えの 18 ms × 2 を消しても生成の 1 回が 6.4 ms 増えるので、T3 は grouped では見合わない
- 残り: ウェブ検索 (SearXNG か Brave) での測り直し。Wikipedia は ja の検索が弱いので、ja で 0 件か外れのとき en も引く手。fetch の 6000 文字は答えの箇所が後ろにあると届かない (渋川市の人口は infobox にあり届いた)。思考なしで長い答えが崩れるのは fetch の 4K トークンの後で、`-think-budget` と `-n` の調整は未検証

#### 16.40 の追記 2: SearXNG でのウェブ検索 — 2026-11-02

- ユーザーが SearXNG (docker、`search.formats` に json) を立て、`-tools search,fetch -search-first -search http://localhost:8888` (思考あり) で同じ 5 問を実行。富士山は「Mount Fuji height meters」→ 英語版と日本語版の Wikipedia が出て、両方を fetch (4.6K と 2.3K トークン) し「約 3,776 m」と両方の出典。『こころ』は「『こころ』 作家」で日本語版の「こゝろ」が 1 位に出て (Wikipedia 自身の検索では出なかった)、fetch して夏目漱石。日本一長い川は「Japan longest river」→ Shinano River を fetch して信濃川 (答えは英語)。渋川市は「Shibagawa City population 2025」とローマ字で検索して品川区が出てしまい、品川区の 426,005 人を答えた (誤り。ローマ字化の問題は 16.39 の lut4 の実験から一貫)。17 × 23 は規則の「numbers」を見て検索し、4 回使い切ってから 391。5 問中 4 問
- fetch の問題: Wikipedia のページは先頭の 6000 文字がほぼメニュー (「コンテンツにスキップ」「メインメニュー」...) で、4K トークンを使って本文はわずか。`Text` は `<main>` (無ければ `<article>`) があればその中だけを取り、`<nav>`、`<footer>`、`<aside>` を落とすようにした
- 規則の変更: 「数値」を「measurements」に変えて「Arithmetic needs no search」を足し、名前は「質問の書き方のまま、翻訳もローマ字化もしない」と足した
- 上の 2 つを入れて同じ 5 問を実行 (思考あり、324 秒。ユーザーの実行は時間を取っていない): 富士山は「富士山 高さ 数メートル」→ 日本語版を fetch (本文だけで 271 行、4.6K → 3.8K トークン) したが、本文の「世界の山」の節のエベレスト 8,848.86 m を混ぜて「約 3,776 m（または 3,999 m）…最新の測量では 8,848.86 m」と崩れた (思考の上限で切れている)。『こころ』は抜粋に「夏目漱石の長編小説」とあり正解。fetch しようとした Wikipedia の URL は百分率符号化を写し間違えて 404。日本一長い川は英語で検索し Shinano River を fetch して信濃川。渋川市は最初「Shigak city」とローマ字 (規則で止まらない) で外れ、2 回目「渋川市 人口 2025」で市の公式ページと統計サイトが出て、抜粋の「2025年人口72,090人」から 72,090 人 (日付は質問のものを当てはめた。市のページ自体には数字が無い)。17 × 23 は `-search-first` が最初の往復を検索にするので 3 回検索してから 391。5 問中 4 問に、富士山の崩れ。ウェブ検索は Wikipedia の検索より明らかに当たりが良く (『こころ』、渋川市)、残る弱さは lut4 のモデル側 (ローマ字化、長い本文からの取り違え、符号化 URL の写し間違え)。URL は `Format` で百分率符号化を戻して渡すようにした (`readable`)
- 残り: 富士山のような長い記事では本文の前半だけが届き、関係ない節の数値を拾う。fetch に `query` を付けて該当の段落を切り出す手がある。ローマ字化は規則では止まらず、`-search-first` の置き換えの検索語 (課題の 1 行目) の方が当たるので、最初の検索語に日本語が無く課題にあれば課題で引く手も

### 16.41 lut6 と grouped 注意の組み合わせ: 速さを失わずに賢さを上げる — 2026-11-02

16.39 の lut6 は賢さ (KL 0.232 対 1.387) と引き換えに 3 割遅く、16.39 の grouped は 1 回の呼び出しを 45% 速くした。2 つを組み合わせれば今の既定 (lut4、tile) より速いまま lut6 になるはず、という見込みを M3 で確かめた。`-format lut6 -quant kmeans-minmax -attn grouped -target m3` で c16320 の Q=8 と Q=64 を変換 (各 2 部分、1.21 GiB、50〜75 秒)。

- 1 回の呼び出し (`TestCallTimes`、4 本同時に読み込み、RSS 1.65 GiB): lut6 の Q=8 は tile 84.1 ms → grouped 42.8 ms、Q=64 は 109.7 → 90.9 ms。lut4 は tile 69.2 → grouped 41.4 ms (16.39、16.40)。つまり lut6 + grouped (42.8) は今の既定の lut4 + tile (69.2) より 38% 速く、lut4 + grouped との差は 3%。交互 (Q=8 と Q=64、grouped どうし) は 142.7 ms の組で、和 133.7 ms との差 9 ms が切り替え
- コーディングの課題 (CHAT.md 7 節の `a - b` → `a + b`、`-yes`、`-package` で 2 本だけ、段階なし。壁時計は読み込み込み): lut6 + grouped 90 秒で 6 要求、lut4 + grouped 82 秒で 13 要求、lut6 + tile 103 秒で 6 要求、lut4 + tile 345 秒で 27 要求。読み込みは lut6 が約 75 秒 (2 部分 × 2 本)、lut4 が約 35 秒なので、課題そのものは lut6 + grouped 約 15 秒、lut4 + grouped 約 50 秒、lut6 + tile 約 30 秒、lut4 + tile 約 310 秒。4 つとも直せたが、lut6 の 2 つは read、edit、(cd /tmp の heredoc、断られる)、go run、read、答え、の 6 往復で終わり、lut4 は `/main.go` や `cd /workspace` を繰り返して 13 と 27 往復 (16.35 で見た lut4 の癖そのもの)。生成は lut6 + grouped 22〜43 tokens/s、lut4 + tile 13〜26
- 5 問 (search と fetch、SearXNG、思考あり、`-package` の 2 本): lut6 + grouped 275 秒で 4 問 (富士山 約 3,776 m、『こころ』は fetch なしで抜粋から夏目漱石、信濃川 367 km と出典、17 × 23。渋川市は抜粋に「令和7年10月1日 人口 69,880 人」を見つけながら PDF を読みに行き、4 回使い切り)。lut4 + grouped 354 秒で 4 問 (渋川市は「Shibagawa」→ 品川区 426,403 人)。lut6 + tile 448 秒で 5 問 (渋川市 70,265 人、2025 年国勢調査を引くサイトから)。lut4 + tile 351 秒で 4 問 (渋川市は思考で 70,265 を見ながら「約 4,256 人」と答えた)。時間は検索の往復の数で揺れるので、速さの比較は呼び出しの費用とコーディングの課題で見る
- 結論: 見込みどおり。lut6 + grouped は今の既定より呼び出しが 38% 速く、課題の往復が 27 → 6 に減り、KL は 6 分の 1。払うのは重み +0.4 GiB (2 部分) と初回の読み込み (Core ML のコンパイル後は 1 本 37〜39 秒、lut4 の 1 本 16〜19 秒)。既定にする価値がある。変えるには `lfm3.Arch` の既定の形式と注意、`chat.Prepare` のパッケージ名 (形式と注意を名前に入れ、既存の `lfm3-c16320-q8` と衝突させない)、カタログの大きさの記述、16320 超の `ChunkedSDPA` の grouped 対応 (今は tile に落ちる) が要る
- 既定の切り替え (同日): `lfm3.Config.Format` が LFM3 本体に lut6 を返し (マルチモーダルは lut8 のまま)、`attn` の既定を grouped にした。16320 超の塊のキャッシュには `blocks.ChunkedSDPAGrouped` を足した (ChunkedSDPA と同じ式で、クエリの頭を kv の頭ごとに積み、マスクと最大値の計算は頭ごとの配置で行う)。`TestChunkedSDPAOnNeuralEngine` に grouped の 4 例を足し、1 と 8 と 64 トークン、2 と 4 塊のどれも NPU に載る。`TestCachedChunksMatchFull` は既定が grouped になったので塊の grouped を CPU で検算する。`chat.Prepare` のパッケージ名は構造・形式・注意・容量・Q になり (`lfm3-lut6-grouped-c16320-q8.mlpackage`)、以前の `lfm3-c16320-q8.mlpackage` は読まれなくなる (消してよい)。Gemma 4 も `gemma4-lut6-c16320-q8` に変わり一度変換し直す。カタログの LFM3-1.7B の説明に「6 bit tables」。`hwnpu-convert` の LFM3 の既定も lut6 と grouped になる
- 残り: 渋川市の問いは lut6 でも揺れる: 抜粋に答えがあっても PDF を読みに行き、思考が 1024 で切れる。`-think-budget` を増やすか、fetch が PDF を断るか
- 既定で動かした結果 (M3): 変換は 1 部分 44〜67 秒 (8 部分で約 7 分半)、Core ML のコンパイルは 1 本 74〜78 秒。4 問 (思考なし) は富士山 3,776 m、17 × 23 は 391、『こころ』は宮本明、川は利根川で、16.39 の lut6 と同じ 2 問。生成は 28〜45 tokens/s (lut4 + tile は 18〜24)。コーディングの課題は 6 要求で修正
- 起動が 2 分半になった: コンパイル済みなのに 1 本 35〜43 秒。`HWNPU_COREML_TRACE=1` では hwnpu のコンパイル済みキャッシュは当たり (0.3〜0.5 秒)、`MLModel` の読み込み (ANE 向けのコンパイル。16.12 の「システム側のキャッシュ」) が 1 部分 17〜19 秒で外れている。同じものを読み直すと当たり外れが部分ごとに入れ替わり、2 本 (4 部分 2.4 GiB) だけなら 2 回目は全部当たり (0.05〜0.5 秒、起動 3 秒)、4 本 (8 部分 4.8 GiB) は 4 回続けて起動しても毎回 3〜6 部分が外れ (110、92、77、62 秒)、直後に 2 本に戻すと 3 秒。以前の既定 (lut4 の 4 本 3.2 GiB) は何も挟まず 3 回起動すると 48、33、3 秒で落ち着く。lut4 の 4 本 (3.2 GiB) は朝の実行で全部当たり (1.1〜1.4 秒)。つまり ANE のキャッシュは容量が 3.2〜4.8 GiB の間 (4 GiB か) の LRU で、lut6 の 4 本は入らない。lut4 の 2 部分 (0.8 GiB) も 1 回目 6.7 秒 → 2 回目 0.06 秒なので、部分の数ではなく容量。`~/Library/Caches/com.apple.e5rt.e5bundlecache` は 11 MB で、本体は aned 側 (sudo が要り未確認)
- 対応: `-tier` の既定を `auto` にし (`chat.TierAuto`)、`Prepare` が形式で決める: lut4 (int4) なら 4096、6 と 8 ビットなら無し (その旨を 1 行出す)。lut6 の既定は c16320 の 2 本 (2.4 GiB) になり、短い会話は c4096 の段階 (grouped で 1 回 27〜28 ms) ではなく c16320 (42.8 ms) で動く。それでも以前の既定 (lut4 + tile、c4096 で 29.9 ms、c16320 で 69.2 ms) と比べ、短い会話で 4 割遅く、長い会話で 4 割速い。4 本にしたければ `-tier 4096` (起動のたびに 2 分半)。初回の起動は変換 4 部分 約 4 分 + コンパイル 2 本 約 2 分半
- 残り: 段階を lut6 でも持つには、ANE のキャッシュに収まる形が要る: Q=8 と Q=64 で重みを共有する 1 パッケージ 2 関数 (`npu_profiles/npuprobe-m3-functions.md` の方向) か、段階の方を Q=8 だけにして 8 トークンずつ読む (4K のプロンプトで 14 秒) か。ANE のキャッシュの容量と場所は aned の側で未確認 (sudo)

### 16.42 既定の切り替えの波及の確認: マルチモーダル、OpenCode、見つかった画像の不具合 — 2026-11-02

16.41 で `attn` の既定を grouped にしたのは LFM3 の全チェックポイントに及ぶ (VL、Audio、Omni、d1 のテキスト部は lut8 のまま grouped になり、名前も変わる)。手元にあるのは LFM3-1.7B だけだったので、LFM3-VL-2B を落として確かめた (3.84 GB、変換は 1 部分 58〜62 秒、視覚エンコーダ 1 秒)。

- `runner.Open` の不具合 (既定の切り替えとは別、cd92cbb の `-opencode` から): `chain.Program(ctx)` は渡された context を以後の全実行に使うのに、`runner.Open` は開いたときの context で取っていた。hwnpu-chat は `signal.NotifyContext` の context でモデルを読み込み、読み込み後に `stop()` を呼ぶ (Ctrl-C を答えの中断に使うため) が、`NotifyContext` の stop は context を取り消すので、以後の視覚エンコーダの実行は全部 `chain: part 0: context canceled` になっていた。`/image` が端末で動かず、サーバーの `image_url` も同じ道。テキストのチェーンは `chat.Engine` が `context.Background()` で取っているので無事だった。`runner.Open` と `OpenSession` の program を `context.Background()` で取るようにした (実行ごとの context は `Run` で確かめる)。audio、speech (hwnpu-tts)、systemone (d1) も `runner` を通るので同じ直し
- OpenCode の要求 (7,854 トークン、`replay.sh`、新しい既定の 2 本): prefill 10.9 秒 (lut4 + tile 14.6 秒、lut4 + grouped 10.9 秒)、生成 43 ms/token、道具の呼び出しは `read /main.go` で lut4 と同じ文。再起動後はプロンプトストアから 0.1 秒 (ストアの鍵はパッケージ名で変わるので新しい項目から)。読み込みは 33.5 + 37.9 秒 (ANE のキャッシュが外れたとき) と 17.6 + 61.6 秒で、16.41 の 3 秒は何も挟まず起動を重ねたときの値
- LFM3-VL-2B の `-tier auto`: lut8 なので段階なし (`no shorter cache tier for LFM3-VL-2B: its programs are lut8`)。`lfm3-lut8-grouped-c16320-q8` と `-q64` (各 2 部分 1.63 GiB) は NPU に載り、読み込み 43〜47 秒 (外れ)。視覚エンコーダは 7.1 秒
- 直した後の LFM3-VL-2B (grouped、lut8): `/image resize.png この画像には何が写っていますか?` に「緑、青、紫、ピンクなどの色が混ざった多色のパターン…全体的にぼやけた外観」(その画像は 16 × 16 の色のグラデーションを拡大した検査用で、そのとおり)。画像つきのプロンプト 278 トークンの読み込み 0.7 秒 (16.31 は約 300 トークンで 0.3 秒。2 部分の lut8 で、視覚エンコーダの実行込み)、生成は画像の答えが 13.9 tokens/s (下書きの採用 1/33: 画像の説明は推測が当たらない)、テキストの質問 (富士山 3,776 m) が 28.1 tokens/s (16.31 の tile は 41 ms/token = 24 tokens/s)。ANE に載り、出力の筋も通るので、grouped の既定はマルチモーダルのテキスト部でも成り立つ
- 未確認のまま: LFM3-Audio-2B と LFM3-Omni-2B (同じテキスト部なので同じ経路だが実行していない。ディスクが 13 GiB)、d1-2B の systemone と hwnpu-tts (runner の直しが効く側)、Gemma 4 (名前が変わる一度の変換し直しと、lut6 なので `-tier auto` が段階なしを選ぶこと。10.2 GB の取得が要る)
- Gemma 4 E2B (以前の lut4 のパッケージ 9 GB を消して 10.25 GB を取得): 名前は `gemma4-lut6-c16320-q8` と `-q64` (各 2 部分 0.81 + 0.79 GiB、変換 1 部分 44〜62 秒)、lut6 なので `-tier auto` は段階なし。4 問 (思考なし) は富士山 約 3,776 m、17 × 23 は 391、『こころ』は「どの作品か分かりかねます」、川は信濃川 (ただし「信濃川・長野川のどちらを指すかで変わる」と続く)。16.30 の答え (富士山 3,776 m、信濃川) と同じ範囲。生成 14〜28 tokens/s (16.30 は 55 ms/token = 18 tokens/s)。配置は「NOT all on the NPU」で、part1 の `reshape` 1 つが CPU (配置の記録)。gemma4 の構築は今日触っていないので以前からと思われる
- Gemma 4 の起動は毎回 98 秒: 4 部分とも `MLModel` の読み込みが 22〜25 秒で、4 回続けて起動しても一度も当たらない (LFM3 の lut6 の 2 本は 2 回目から当たって 3 秒)。CPU に落ちる演算がある部分は ANE のキャッシュに入らないのかもしれない (仮説。LFM3 の部分は全部 NPU)。CHAT.md の「読み込みは数秒から 1 分」は Gemma には当てはまらず、16.30 も初回の 96 と 104 秒しか記録していない。gemma4 の `reshape` を NPU に載せれば起動も直る可能性がある: 残り

### 16.43 Gemma 4 の起動 98 秒の追跡: CPU の reshape でも部分の大きさでもない — 2026-11-03

16.42 の仮説 (part1 の `reshape` が CPU にあるから ANE のキャッシュに残らない) を確かめた。

- `-logits folded` (gemma4 の新しいパラメータ、`Options.FoldedLogits`): y を (1, seq, 16, 16384) のまま出し、最後の reshape を無くす。`decode` は 4 次元の y も読む (`logitsShape`、`TestCachedFoldedLogits`)。結果は「all on the NPU」で、4 問の答えは flat と同じ、生成は 12.8〜24.8 tokens/s (flat 13.7〜28.1) と 5〜12% 遅い (build.go の注釈にあった「4 倍遅い (137 ms)」は再現しない)。しかし起動は 109 秒のままで、4 部分とも 25〜29 秒の外れ。仮説は否定
- 部分を小さく (`-parts 3`、0.51〜0.55 GiB): 3 回続けて起動しても 16〜28 秒の外れが続く (3 回目に 1 部分だけ当たり)。大きさでもない
- 同じ手順で LFM3-VL-2B は 57、33、7 秒と 3 回目で落ち着く (4 部分 + 視覚エンコーダが全部当たり)。LFM3-1.7B の lut6 も 2 回目から 3 秒。つまり Gemma 4 のプログラムだけが ANE 側のキャッシュに残らない。hwnpu 側のコンパイル済みキャッシュは毎回当たっている (0.7〜0.8 秒)
- 形式を変えて同じ Q=8 を 3 回読む (`TestCallTimes`、`-count=1`): lut4 (2 部分 0.54 + 0.52 GiB) は 10.7 + 9.4 秒、10.4 + 9.2 秒、3 回目 0.07 + 0.04 秒で当たる。lut8 (3 部分 0.69〜0.74 GiB) は 3 回とも 8〜10 秒の外れ。lut6 (2 部分) は 29 + 25 秒の外れが続く。LFM3 は lut6 (0.60 GiB の部分) も lut8 (VL、0.81 GiB) も当たるので、大きさの閾値ではない。Gemma 4 の 6 と 8 ビットの表のプログラムだけが ANE 側のキャッシュに残らない、というところまで。理由は ANE 側 (aned の挙動、sudo が要る) で、ここからは追えない。外れたときのコンパイルは lut6 が 1 部分 25〜30 秒、lut8 が 9〜10 秒、lut4 が 10 秒
- RSS (2 本読み込み): Gemma lut6 1.67 GiB、LFM3 lut6 2.07 GiB (重みはどちらも Core ML の mmap で RSS に全部は出ない)。1 回の呼び出しは Gemma lut6 の Q=8 61.5 ms、Q=64 137.8 ms (LFM3 の lut6 grouped は 41.9 と 78.8 ms)
- 結論: Gemma 4 の起動約 100 秒は今の形式 (lut6) では避けられず、`-logits folded` も `-parts` も効かない。lut4 なら 3 回目から 1 秒だが、16.26 のとおり答えが悪い。`-logits folded` はパラメータとして残す (全部 NPU に載り、5〜12% 遅い)。残り: aned のキャッシュの方針の確認。Gemma を lut4 の既定に戻すことはしない

### 16.44 AMD: 実機での最初の確認 (Ryzen AI 9 365) — 2026-11-03

- 機械: Ryzen AI 9 365 (Strix Point、PCI 0x17f0、`npu4` で 8 列、AIE 1.1、4 コア行)、Ubuntu 26.04 の Linux 7.0.0-42 標準カーネルの amdxdna (DKMS なし)、ファームウェア 1.1.2.64。XRT は入れていない
- 設計: mlir-aie v1.4.4 の wheel (Python 3.12 必須。uv で用意) で `passthrough_dmas` を `-d npu2` でビルド。xclbin をまとめる `xclbinutil` は XRT の道具なので、Ubuntu universe の `libxrt-utils` と boost の `.deb` を root なしで展開し、`AIE_XCLBINUTIL` で渡した
- 結果: `cmd/report` の全項目が合格 (電力センサーは SKIP)。passthrough 16 KiB の 1 回は中央値 0.075 ms、最小 0.068 ms、初回 0.15 ms。xclbin の読み込み (ハードウェアコンテキストの作成と PDI の設定) は 0.5 ms
- 偽のドライバ (16.5) では見えなかった食い違いが 3 つあった。どれも Linux 7.0 のソース (drivers/accel/amdxdna) で確かめ、`xdnasim` を同じ振る舞いに直した

| 症状 | 原因 | 直し方 |
|---|---|---|
| `SyncFromDevice` が `SYNC_BO: invalid argument` | 7.0 の `amdxdna_drm_sync_bo_ioctl` は、向きに関係なく範囲を clflush したあと、`FROM_DEVICE` ならその BO を「デバッグ用の BO」として同期する (`amdxdna_hwctx_sync_debug_bo`)。普通の BO はハードウェアコンテキストに割り当てられていないので `EINVAL` | ioctl には常に `TO_DEVICE` を渡す。x86 の clflush は書き戻しと無効化の両方なので、どちらの向きにも足りる |
| 実行は終わるが出力が入力と違う | 7.0 では host の BO (`BO_SHARE`、`BO_CMD`) の `xdna_addr` は設定されず `AMDXDNA_INVALID_ADDR` のまま。PASID のもとでは、デバイスは mmap 後のユーザーアドレスを使い、`GET_BO_INFO` の `vaddr` がそれを返す。hwnpu は全ビット 1 のアドレスを引数に入れていた | `xdna.BO.DeviceAddr()`: `xdna_addr` が有効ならそれ (BO_DEV とヒープ)、無効なら `vaddr`。マップ後も無ければエラー |
| `argument 6 (bo3) is not set` | v1.4.4 の IRON は、3 バッファの設計でも `bo0`〜`bo4` を署名に並べる。ホストのコードは使わない引数を設定せず、XRT はレジスタマップを 0 のまま送る | 未設定の引数は 0 で送る (検査をやめた)。`xdnasim` の passthrough も 8 引数にし、`bo3`、`bo4` が 0 であることを確かめる |

- 電力: upstream の 7.0 には `GET_INFO(QUERY_SENSORS)` の実装がない (DKMS 版だけ)。NPU 単体の電力は読めないので、測るなら RAPL (CPU パッケージ全体) など別の手段が要る
- 残り: Phoenix / Hawk Point (`npu1`、ヒープ 64 MiB 固定) での確認。ELF 流れと UMQ。LLM に向けた次の段階は、GEMV のカーネル 1 本で速度を測ること (16.45)

### 16.45 AMD: GEMV で NPU のデコードの速さと電力を測る — 2026-11-03

LLM のデコード (1 トークンずつ) は行列ベクトル積 (GEMV) が重みを読む速さで決まる。カーネルを自前で書く前に、NPU が DRAM から重みをどれだけ速く、どれだけの電力で読めるかを測った。

- 設計: [amd/IRON](https://github.com/amd/IRON) (Apache-2.0。AMD の NPU 向けの演算子とモデルの集まり) の `iron.operators.GEMV`。mlir-aie の bf16 の `mv` カーネルを 8 列に並べたもの。IRON の固定した mlir-aie の nightly (1.4.5.dev119) で、`op.compile()` だけを呼んで xclbin と命令列を作った (実行には XRT を使わない)。スクリプトは `aie_utils.set_current_device(from_name("npu2", n_cols=8))` で装置を指定する
- 実行: `tools/amd/gemvbench` (hwnpu の KernelRunner で `kernel(3, instr, ninstr, A, B, C)` を回し、CPU の float64 と照合)。どの形でも誤差は行の大きさの 2.4% 以内で、全行が合格
- 電力: upstream のドライバには NPU のセンサーがないので、amdgpu の hwmon の `PPT` (APU のパッケージ全体、µW、root 不要) を 50 ms ごとに平均した

| 形 (M×K、bf16) | 重み | 中央値 | 重みを読む速さ |
|---|---|---|---|
| 8192×2048 (FFN の上り) | 32 MiB | 0.686 ms | 48.9 GB/s |
| 8192×2048、2 チャネル | 32 MiB | 0.676 ms | 49.6 GB/s |
| 2048×8192 (FFN の下り) | 32 MiB | 0.687 ms | 48.8 GB/s |
| 1536×12288 (Gemma 4 E2B の下り) | 36 MiB | 0.760 ms | 49.6 GB/s |
| 6144×1536 | 18 MiB | 0.405 ms | 46.6 GB/s |
| 2048×2048 | 8 MiB | 0.239 ms | 35.1 GB/s |

- 天井は約 50 GB/s で、タイルの大きさやチャネル数ではほとんど変わらない。1 回の投入の固定費は約 0.07 ms (8 MiB で効率が落ちる分)
- CPU との比較 (同じ機械、Go で 2 GiB を読むだけの合計): 1 スレッド 55 GB/s、8 スレッド 79 GB/s。NPU は CPU が読める最大の約 6 割
- 電力 (8 秒回し続けたときのパッケージ全体、冷ましてから測ったアイドル 4.7 W): NPU の GEMV は 12.3 W で、32 MiB の 1 回あたり 3〜5 mJ。CPU は読むだけで 18〜28 W、32 MiB あたり 7〜12 mJ (計算を含めればさらに増える)。待つ間 hwnpu はスレッドを眠らせる (syncobj の待ち) ので CPU は空く
- 見積もり: 4 ビットの重み (q4nx。0.56 バイト/パラメータ前後) なら、LFM3-1.7B は 1 トークン約 1 GB で 50 GB/s なら 40 トークン/秒前後が上限。bf16 なら約 14 トークン/秒

**設計 (演算ごとに xclbin) の切り替え費用** (`tools/amd/ctxbench`): 3 つの GEMV の xclbin を同時に読み込み (ハードウェアコンテキストが 3 つ)、交互に投入すると、単独の合計 1.571 ms に対して 2.960 ms。**切り替え 1 回で約 0.46 ms**。まとめて投入しても縮まない (3.16 ms)。演算が 1 トークンに 100 個あるモデルを演算ごとの xclbin で回すと、切り替えだけで 46 ms になる

- 結論: デコードを NPU に任せる価値はある (CPU の 6 割の速さを、半分以下の電力で、CPU を空けたまま)。ただし演算ごとに xclbin を切り替える方式は使えない。IRON が NPU2 で使う「full ELF」(グラフ全体を 1 つのハードウェアコンテキスト、1 つのコマンドで流す) に対応する必要がある
- full ELF の投入: XRT は xclbin なしで ELF からハードウェアコンテキストを作り (`xrt::hw_context(device, elf)`)、コマンドは `ERT_START_NPU_PREEMPT_ELF`。Linux 7.0 は `force_cmdlist` (既定 true) の経路で `EXEC_NPU_TYPE_ELF` のスロットを作る。ファームウェアの機能表 (npu4) では、メールボックスのプロトコルの主版 7 で `AIE2_NPU_COMMAND` と `AIE2_PREEMPT` が有効になる。この機械のファームウェアは `npu_7.sbin` (1.1.2.64) なので使える

### 16.46 AMD で LLM を動かす設計 — 2026-11-03

[amd-npu-models.md](amd-npu-models.md) にまとめた。要点:

- AMD に公開されたグラフ用コンパイラはなく、演算ごとの xclbin は切り替え費用 (16.45) で使えない。[amd/IRON](https://github.com/amd/IRON) をビルド時のコンパイラとして使い、モデルの 1 回の呼び出しを 1 つの full ELF にしたものを、hwnpu が Go で実行する。「変換は Go だけで」の例外で、利用者の機械には Python も XRT も要らない
- full ELF の投入 (CREATE_HWCTX のみ、制御コードと PDI を BO_DEV に、再配置で引数のアドレスを書き、`ERT_START_NPU_PREEMPT_ELF`) は、XRT を root なしで展開して IRON のグラフを動かし、ioctl を `LD_PRELOAD` で記録して確かめた。再配置の種類 5 (shim_dma_48、+0x80000000) と 8 (address_64) は、書き換え後の制御コードと一致
- 最初のモデルは IRON に既にある Gemma 4 E2B (bf16、IRON の CI で 5.6 トークン/秒)。次に LFM3-1.7B の IRON のアプリケーションを書き、4 ビット (q4nx の一般化) で速くする

### 16.47 AMD: full ELF の実行 (段階 A1) — 2026-11-03

- `internal/aieelf`: aiecc の full ELF を読む。グループからインスタンス (`main:sequence`) と制御コード、`.dynsym` から引数と PDI、`.rela.dyn` から再配置 (種類 3、5、8。知らない種類と、制御パケット、プリエンプション、`.ctrldata` はエラー)、`.note.xrt.configuration` から列の数。`Patch` は書き換え前の語から計算するので、引数を変えて何度書き直しても同じになる。テスト用の `Build`
- `internal/backend/amd/elf.go`: `ProgramSource.ELF` (または中身が ELF の `Xclbin` / `Path`) で読み込む。PDI はプログラムで共有する `BO_DEV`、制御コードは run ごとの `BO_DEV`。引数が変わったときだけ書き直して同期する。`ctrl scratchpad` があれば run ごとに `BO_DEV` を作ってアドレスを書き、`hwnpu.Run.Scratchpad()` で見せる (中身の意味は段階 A2)
- `internal/ert`: `StartNPUPreemptELF`。XRT が書いたパケット (記録) と語ごとに一致することをテストで確かめる
- `xdnasim`: opcode 22 を受け、CU のないコンテキスト、ヒープ上の制御コード、プリエンプションなしを確かめ、`RunELF` に制御コードを渡す
- `tools/amd/iron_export.py`: IRON の `CompiledGraph` から ELF と manifest (コマンド、引数の順と大きさ、バッファの配置、入出力、重み、パラメータ) を書き、`--check` で乱数の入力と `graph.reference()` の出力を書く。`tools/amd/ironcheck` はそれを hwnpu だけで動かして照合する
- 実機 (Ryzen AI 9 365): IRON の ReLU → AXPY (2048 要素) の full ELF が、IRON も XRT も使わずに動き、出力は `graph.reference()` とビット単位で一致。読み込み 1.0 ms、1 回 0.27 ms

### 16.48 AMD: 呼び出しごとの値 (スクラッチパッド) と、キャッシュの一貫性 — 2026-11-03

- IRON の `Scratchpad` の値は、制御コードの TXN の命令で入る: `CREATE_SCRATCHPAD` (op 10。`scratch-pad-ctrl` の BO_DEV のアドレスを種類 8 で書く) が状態表を読み、`UPDATE_REG` (op 12、`{スロット, 関数, FuncArg, RegOff}`) がレジスタに `状態表[スロット]` を足す (INCR) か `× FuncArg` を足す (MUL)。コアが読むパラメータ (種類 `core`) は WRITE で 0 にしてから INCR で入れ、shim の BD の長さは BD の書き込みのあとに MUL で足す (mlir-aie の `AIEUtils.cpp`)。`core` の値は 2 ビット左にずらして書く (ファームウェアが下位ビットを落とすため)
- スクラッチパッドの語は、グラフの呼び出しごとの値の Affine 変換 (IRON の `Word.form`: `clamp(((v·scale + bias) >> down)·mul + add, lo, hi) % mod + post`)。IRON の bounded ReLU (4096 要素) の `n` はコアあたりの塊の数 `ceil(n / 2048)` (0〜2) になる。`tools/amd/iron_export.py` が式を manifest に書き、`tools/amd/ironcheck` が Go で計算する。`n` = 100、2048、3000、4096 で `graph.reference()` と一致 (デバイスは塊の単位で書くので、`n` を塊に切り上げた所まで書く)
- `internal/aieelf/txn.go`: 制御コードの TXN (主版 0 の形式) の解読器
- **キャッシュの一貫性 (実機で見つけた不具合)**: 新しい出力バッファに NPU が書いた結果が、最初のコマンドだけ CPU から 0 に見えた。原因は CPU の汚れたキャッシュ行。新しいページはカーネルが最初に触れたときにキャッシュを通して 0 で埋めるので、その汚れた行が、NPU が DRAM に書いたあとで書き戻され、結果を上書きする。2 回目からはページが既にきれいで起きない。スクラッチパッドが最初の呼び出しだけ効かないように見えたのも、これだった (切り分けの途中で、空のコマンドを先に流す、CREATE_SCRATCHPAD や LOADPDI だけを先に流す、を試したが関係なかった)
  - 直し方: `allocMapped` でマップした直後に全ページに書いてから全体をフラッシュし、汚れた行のない状態で渡す。デバイスヒープもマップ直後にフラッシュする
  - 同期は XRT と同じく、x86 ではユーザー空間の CLFLUSH + MFENCE (`xdna.FlushCPU`、Go のアセンブリ) にした。Linux 7.0 の `SYNC_BO` はカーネル側の写像でフラッシュするので、ページがまだ固定されていない BO では何もしない経路がある (`Can not get flush memory`)。amd64 以外は今までどおり `SYNC_BO`
- `xdna.Dev.Wait`: 完了待ちの `SYNCOBJ_TIMELINE_WAIT` が Go のランタイムのシグナルで `EINTR` になると、エラーにしていた。絶対時刻の期限のまま待ち直す
- 未対応: 前のプロセスの直後に `CREATE_HWCTX` が一時的に `EINVAL` になることがある (前のコンテキストの後始末と重なる)。再試行を足す

### 16.49 AMD: 引き継ぐ値 (段階 A2 の残り) — 2026-11-03

- IRON の `Carried` の値は、ホストから 1 回ずつ呼ぶとき (hwnpu の使い方) は、呼び出しごとの値と同じくスクラッチパッドの語になる (Walk の `node` と `steps` は DMA のオフセット、種類 `addr` でずらさずに書く)。IRON の `_write_values` と `_next_values` に合わせ、呼ぶ前に現在の値を `carry` バッファの 1 段目に書き、呼んだ後は、次の値が Affine の式 (`steps + 1`) ならホストで計算し、計算された値 (`node`) は 2 段目から読む
- `tools/amd/iron_export.py`: manifest に `carried` (carry の配置と名前の順、次の値の求め方) と `images` (重みを置いた引数の初期像 `arg_<種類>.bin`。状態は 0) を書く。`--check` は引き継ぐ値のあるグラフでは `graph.reference()` で呼び出しの列を作る (`check/calls.json`)
- 実機: IRON の section-5 の Walk (64 要素の後続表を引く) をホストから 12 回呼び、次の値が全て参照と一致。1 回 0.18 ms
- 残り: 1 つの ELF に複数のコマンドがある場合 (`Plan.commands`)。バッファのアドレスに依る語 (`recording.addresses`) は manifest に書けないのでエラーにしている。LLM のグラフで出てきたら対応する

### 16.50 AMD: Gemma 4 E2B が hwnpu で動く (段階 A3) — 2026-11-03

- `tools/amd/iron_export_lm.py`: IRON の `CausalLM` (今は Gemma 4) を、デコードの 1 ステップとプロンプトの塊 (2048 トークン) の 2 つの版にコンパイルし (IRON の `load_versions` と同じく 1 つの arena に置く)、ELF、arena の bank の初期像 (重みを置いたもの、3.5 GiB × 2 + 1.9 GiB)、manifest を書く。manifest には版ごとに、コマンド (プロンプトはスクラッチパッドが 32 語に収まらず 15 コマンド)、引数、コマンドごとのスクラッチパッドの語、引き継ぐ値 (token と position)、ロジットの出力、全バッファの配置、プロンプトの埋め込みの encoder を書く
  - 語の形: Affine (16.48)、Linear (`ceil(v·比 + オフセット)`、比とオフセットは分数)、rows の表 (Gather と RowwiseMul / RMSNorm の行数)、MHA の派生値 (演算子の定数と、範囲 `valid` = rows、`kv_valid` = position + 1 の式から、`kv_blocks`、`s_q`、`q_start`、`q_offset` などを Go で計算)。IRON が計算した語を `check/words.json` に書き、`TestWordsMatchIRON` でデコード 5 通り、プロンプト 4 通りの呼び出しの全ての語が一致することを確かめた
  - encoder: プロンプトのトークン ID を、埋め込みの表 (2 つ: 通常と per-layer) から行を引く DMA の制御語に変える (IRON の `Gather.control_words`: 雛形の語の位置に、表のデバイスアドレス + 0x80000000 + ID × 行のバイト数)。`hwnpu.Buffer.DeviceAddress()` を足した
  - 版によっては同じ名前で別の重みを別の場所に持つ (プロンプトの版の w1 など 4 つ)。名前で重複を除かず、版ごとに全部書く
- `internal/llm/iron`: 実行器。bank を `BO_SHARE` に確保して像を読み込み (9 GB、ページキャッシュから 2〜3 秒)、`Step(token, position)` と `Prompt(tokens, start)` がロジットを返す (ホストで標本を引く。デバイスが引いたトークンは使わない)。プロンプトの 15 コマンドは、Linux 7.0 のデバイスヒープ (npu4 でも 64 MiB 固定。512 MiB に広げられるのは DKMS 版だけ) に制御コード (合計 89 MiB) が載らないので、呼び出しのたびに run を作って閉じる。デコード (21 MiB) は常駐
- **再配置の addend は符号なし**: arena の bank の 2 GiB を超える位置への addend (10615 個) を符号付きで読んでいたため、DMA が bank の手前を指し、デコードの 1 ステップが自分の重み (第 26 層の per-layer 埋め込みの表) に 1 MiB の NaN を書いていた
- 結果 (Ryzen AI 9 365、実際の重み、greedy): 「日本の首都はどこですか？」に「日本の首都は東京です。」、Go の関数の作成、README (3505 トークン) の 3 点の要約、どれも筋の通った答え。デコード 5.5〜6.1 トークン/秒 (IRON の CI の Krackan で 5.6)。プロンプトは 25 トークンで 0.58 秒、3505 トークン (2 塊) で 19.5 秒 (180 トークン/秒) と、IRON の Llama 3.2 1B (1024 トークンで 0.47 秒) より遅い。要調査
- 乱数の重みでの IRON の CPU 参照 (`OnHost`) との照合は、コサイン 0.55〜0.83 で一致しなかった。乱数の重みは値の尺度が整っておらず 35 層で丸めの差が増幅されるためと考えるが、未確認。実際の重みでの Hugging Face のロジットとの照合は、torch のない環境なので未実施

### 16.51 AMD: hwnpu-chat と hwnpu-agent (段階 A4) — 2026-11-03

- `chat.Engine` の `sess` を interface (`chat.session`) にした。Core ML の `decode.Session` と、IRON の実行器を包む `ironSession` が満たす。IRON の版はトークン ID を受け取る (埋め込みはデバイスで引く) ので、エンジンの `step` はセッションが `tokenSession` ならトークンを渡す
- `ironSession`: 1 回に読むのは最大 2048 トークン (プロンプトの塊)、生成は 1 トークンずつ (下書きの確認なし)。Mark と Rewind は位置だけ: IRON の KV キャッシュは位置で書くので、巻き戻した先から上書きすればよい (Gemma 4 のスライディング窓の層は、`3 × リング < max_seq_len` のときだけリングになり、4096 ではならない。リングや畳み込みの状態は `restart` が塊の境に制約されるので、そういうモデルはまだ扱わない)。会話の退避 (`Save`)、prompt store、段階の切り替えは Core ML だけ
- `chat.Config.IRON` と、`hwnpu-chat -iron DIR`、`hwnpu-agent -iron DIR`。トークナイザーとチャットの型は書き出したディレクトリから (エクスポートが `tokenizer.json` などを写す)
- 実機 (Ryzen AI 9 365): `hwnpu-chat -iron -listen` の OpenAI 互換 API が答え、2 回目の要求でプロンプトの先頭 28 トークンを再利用 (1 トークン 163 ms)。`hwnpu-agent -iron` が `calc.go` を読む道具を使い「エクスポートされた関数は Add と Mul」と答えた (717 トークンのプロンプト、5.9 トークン/秒)
- プロンプトの遅さ (16.50) はデバイスの計算: 1953 トークンの塊の 15 コマンドが 190〜850 ms ずつ (計 7.6 秒)、run を作って閉じる費用は 1 コマンド 2 ms 以下

### 16.52 AMD: prompt store と会話の退避 — 2026-11-03

- IRON の KV キャッシュ (Gemma 4 E2B は 15 層 × keys / values の 30 本、(4096, G, D) の bf16) は、ホストから読み書きできる arena の bank の中に位置ごとに連続して並び、2 つの版で配置が同じ。`iron.Model.ExportRows` / `ImportRows` が位置の範囲を写す (1 位置 18 KiB、位置ごとにキャッシュを名前順)。KV キャッシュが状態の全てなので (リングや再帰の状態のあるモデルはまだ書き出さない)、行が系列の全てになる
- `ironSession`: `RowBytes`、`ExportRows`、`Resume` (prompt store から戻す) と、`Save` (系列の行を写した印) / `Rewind` (退避した印なら行を書き戻す)。store は `decode.Session` ではなく `RowBytes` と `ExportRows` だけを求める interface (`rowSource`) を受ける。store のキーはモデルの名前、manifest の更新時刻、bank の大きさ
- 実機 (README の 6000 文字を system に入れた 2736 トークンの要求、greedy): 初回 13.2 秒。サーバーを再起動して同じ要求は store から 2736 トークンを戻して 0.0 秒、答えは同じ。同じ system で別の質問は store の 2048 の印から戻して残りを読み 6.4 秒。別の文書の会話 (3258 トークン) を挟んだあと元の会話は退避から 2735 トークンを戻して 0.0 秒。store は 1 項目 51 MB

### 16.53 AMD: パッケージ全体の電力 (PPT) — 2026-11-03

- Linux 7.0 の標準カーネルには NPU の電力センサー (`QUERY_SENSORS`) がない。amdgpu の hwmon の `power1_input` (PPT: APU のパッケージ全体、µW、root 不要) を代わりに読み、`Scope` を `package` にする。NPU のセンサーがあればそちらを使う。どちらかがあれば `CapPowerMeter`
- どちらも瞬間値なので、最初の `PowerSample` から 50 ms ごとに読んで台形で積分する裏の処理を回す (デバイスを閉じると止める)。前は呼び出しの間を 1 本の台形にしていたので、長い区間では 2 点の平均になっていた
- `Provider.Hwmon` で探す場所を差し替えられる (テスト)。`xdnasim.NoSensors` で Linux 7.0 と同じくセンサーなしにできる
- 実機: `hwnpuinfo` に `power-meter`、`cmd/report` の電力の項目が通る。Gemma 4 E2B の生成中はパッケージ全体で平均 11.8 W、1 トークン 1.93 J (アイドルの 5〜7 W を含む)

### 16.54 AMD: 投機的デコードの見積もりと、プロンプトの塊の大きさ — 2026-11-03

下書きを k トークンまとめて確かめる呼び出しの費用を、プロンプトの版で k 行を流して測った (`ironlm -bench`: 500 トークンのあと、位置 500 で中央値)。

| プロンプトの塊 | デコード 1 ステップ | 2 / 4 / 8 トークンの呼び出し | 2683 トークンのプロンプト | 10 トークンのプロンプト |
|---|---|---|---|---|
| 2048 (IRON の既定) | 166 ms | 1325 ms (8.0 ステップ分) | 13.0 秒 | 0.56 秒 |
| 1024 | 166 ms | 820 ms (5.0 ステップ分) | 10.2 秒 | 0.43 秒 |
| 256 | 167 ms | 752 ms (4.5 ステップ分) | 10.2 秒 | 0.43 秒 |
| 64、8 | — | — | — | — |

- 64 と 8 は作れない (MHA の seq_pad は 64 の倍数、flm.ProjectionGEMM の M は 256 の倍数)
- 確かめる呼び出しは行数にほとんど依らず、デコード 4.5 ステップ分以上の固定費がかかる (15 コマンドで多くの設計を構成し直す)。下書きの確認が得になるには 1 回に平均 4.5 トークン以上受け入れる必要があり、prompt lookup の下書き (M3 の LFM3 で 1 回 1.07〜1.3 トークン、16.36〜16.37) では届かない。**今の IRON の版では投機的デコードはしない**。するなら、デコードと同じ GEMV の演算で k 行を 1 回の重みの読み出しで処理する確認用の版を、IRON の側に作る必要がある
- 副産物: 塊は 1024 の方が速い (長いプロンプトで 2 割、短いもので 2 割)。256 も同じ速さだが、4096 トークンの文脈では Gemma 4 のスライディング窓の層がリングになる (`3 × リング < max_seq_len`。リング = 窓 + 塊 − 1 以上の 2 のべき) ので、位置だけの巻き戻しと行の写しが使えない。1024 ならリングにならない。エクスポートに `--prefill-chunk` を足し、manifest に `restartable` (IRON の `CausalLM.recurrent` が空か) を書く。チャットのエンジンは `restartable` でないものを読み込まない
- hwnpu-agent の 2 回目の短いプロンプト (84 トークン) は 1.8 秒から 1.1 秒に

### 16.55 AMD: Gemma 4 E2B の画像と音声 — 2026-11-03

- IRON の `Gemma4Multimodal` は 1 つのグラフで 5 つの版を持つ: デコード、プロンプト、視覚の塔 (入力は画像の画素の行)、音声の塔 (入力は 16 kHz の波形)、`placed` (目印のトークンの位置に塔の出力を差し込むプロンプトの塊)。塔は soft token を共有の状態 `merged` に書き、`placed` の版は ID の列と、目印ごとの「その媒体の何番目から」(`before`) を受け取る。塔ごとに `merged` は 1 つなので、1 回のプロンプトに画像 1 枚と音声 1 本まで (IRON の制約)
- エクスポート (`tools/amd/iron_export_lm.py gemma4mm`): 5 つの版と、Go の前処理に要る設定 (目印のトークン、視覚の pooling と patch と予算 280、音声の hop と frame など) を書く
  - **塔の語は表にした**: 視覚の塔には、形のない派生値が 2〜4 つの値に依るもの (Resample の `image_count` と `patch_count`、ResampleTaps の `count`) があり、式の移植は誤りやすい。hwnpu は画像をホストで処理系の大きさ (48 ピクセルの塊 hk × wk、hk·wk ≤ 280) に縮めてから渡すので、塔の縮小は等倍になる。そこで取りうる (hk, wk) の 1600 通りすべてについて、エクスポート時に IRON の計算で全ての語を評価して表にした。音声の塔は値がどれもフレーム数で決まるので、フレーム数 (1〜3000) の表。バッファのアドレスに依る語だけは表にせず Affine のまま
  - バッファのアドレスに依る値 (`recording.addresses`: 中間の ID から行を引く表の `base_lo` / `base_hi`) を manifest に書き、Go が実行時に計算する
- 実行器 (`internal/llm/iron`): `RunVision` (`ImageSize` は IRON の `VisionConfig.size` を移植し、PIL 互換の bicubic でホストで縮め、行を 4096 バイトの塊に揃えて詰める)、`RunAudio` (`AudioTokens`、hop の分の 0 を前に置いた波形)、`PromptPlaced` (`placed` の入力には encoder の制御語ではなくトークン ID そのものを書く: デバイスが写して自分で行を引く)。音声の塔の制御コード (39.8 MiB) は常駐のデコード (21 MiB) と並ばないので、塔を流すときはデコードの run を外し、次のデコードで作り直す。PDI は像の本体 (`main`) の分をプログラムの読み込み時に確保する (run ごとに遅れて確保すると、ヒープの途中に散らばって 40 MiB の連続領域が残らなかった)
- エンジン: IRON の多モーダルのモデルでは、媒体ごとに soft token の数を数え、`<|image>` + 媒体ごとの負の ID × N + `<image|>` (音声は `<|audio>` / `<audio|>`) にする。`ironSession` は負の ID を含む塊を、媒体の境で分け、その媒体の塔を流し (別の媒体が `merged` に入っているときだけ)、負の ID を目印のトークンに戻し、`before` を付けて `placed` の版で読む。`Content` の `parts` に媒体の目印も順に入れた (Gemma 4 の型は `parts` を書くので、画像が落ちていた)
- prompt store: 媒体の負の ID はプロセスごとに振り直すので、再起動後に別の画像が同じ ID になりうる。store の照合は最初の負の ID より前だけにし、負の ID を含む系列は書かない (LFM3-VL にもあった危険)
- 実機 (Ryzen AI 9 365、greedy、思考なし): 白地に赤い円と青い四角の 640×480 の PNG に「左に赤い円、右に青い四角、背景は白」(290 トークン、3.3 秒)、続けて「どちらが大きいか」に「赤い円」(311 トークン再利用、0.8 秒)。ALSA の試験音に「front left」、speech-dispatcher の案内 (約 18 秒) をほぼ正確に書き起こし (756 トークン、2.2 秒)。`ImageSize` と `AudioTokens` は IRON の計算と 6 通り / 5 通りで一致 (テスト)

### 16.56 AMD: Windows のための XRT 経由のバックエンド (xrtshim) — 2026-11-04

- 動機: Windows の AMD の NPU ドライバは XRT の C++ ライブラリ (`xrt_coreutil.dll`) を通してしか使えない。hwnpu は cgo を使わないので、C の ABI を持つ小さなライブラリ `native/xrtshim` (C++ で XRT を呼び、C の関数を出す) を作り、purego (`internal/dl`) で実行時に読む。バックエンドは `xrt` (`internal/backend/amdxrt`)
- 中継の API: デバイス (数、開く、名前・BDF・platform と electrical の JSON・XRT の版)、バッファ (host_only と cacheable、メモリグループ、map、address、sync)、ハードウェアコンテキスト (xclbin から `register_xclbin`、full ELF から `xrt::elf`)、カーネル (`xrt::kernel` と `xrt::ext::kernel`、`group_id`)、ラン (引数、`get_ctrl_scratchpad_bo`、start で scratchpad を同期、wait、state)。失敗は malloc した文字列を返し、Go 側が写して解放する (スレッドローカルの最後のエラーはゴルーチンがスレッドを移るので使えない)
- XRT 2.21 で見つかったこと: (1) `run.wait(timeout)` が時間切れになった命令をもう一度待つと `qds_device::wait() unexpected command state` になる (デコード 1 回の 166 ms が区切りの 100 ms を超えた)。待ちは時間を区切らない。(2) xclbin のカーネルの命令列の引数はそれ専用のメモリグループ (`group_id(1)`) にあり、グループ 0 の cacheable のバッファは「local copy」を作ろうとして失敗する。命令列のバッファは Go のメモリに持ち、引数に結んだときにそのグループで XRT のバッファを作って写す (同期で全部に写す)。(3) full ELF のカーネルは開いている間、制御コードを 64 MiB のデバイスヒープに置く。カーネルを開いたままにすると音声の塔 (39.8 MiB) が `ENOSPC` になったので、カーネルは使うランがなくなったら閉じる (iron の塔の前にデコードのランを捨てる 16.55 の手当てがそのまま効く)
- 選び方: Windows では常に、Linux では `HWNPU_BACKEND=xrt` のときだけデバイスを出す。`amd` (xdna) より先に登録するので、`hwnpu.Open(WithVendor(AMD))` を使う iron やツールはそのまま `xrt` を選ぶ
- Windows で `xrt_coreutil.dll` は検索パスの外 (`System32\AMD`、ドライバストア) にあるので、先に絶対パスで読み込んでおき、xrtshim の import をそのモジュールに解決させる (`HWNPU_XRT_CORE` で指定もできる)。XRT のヘッダ 2.21.75 (Apache-2.0) を `native/xrtshim/include` に同梱し、`make-importlib.ps1` が DLL の export から import library を作る。MSVC でのビルドは未確認 (zig の x86_64-windows-gnu でコンパイルが通ることだけ確かめた)
- 電力: XRT の electrical の JSON の `power_consumption_watts` を 50 ms ごとに読んで積分する (`power.Integrator` に amd の PPT と共通化)。Linux 7.0 の標準ドライバでは出ない
- 検証 (Ryzen AI 9 365、Linux 7.0、XRT 2.21.75 を root なしで展開): `cmd/report` の passthrough (xclbin) が PASS、IRON の relu-axpy・bounded-relu (スクラッチパッド)・walk (引き継ぐ値) がビット単位で一致、`ironlm` の Gemma 4 E2B が 6.1 tok/s (xdna と同じ)、`hwnpu-chat -iron` の多モーダル版で画像 (「左に赤い円、右に青い四角」) と音声 (「Front, left」) が正しい。Windows の実機では未検証

### 16.57 AMD: NPU1 のための「1 ステップごとの xclbin」— 2026-11-04

- 動機: NPU1 (Phoenix、Hawk Point) のファームウェアには full ELF の投入 (AIE2_PREEMPT) がない。IRON は NPU1 では各版を「1 ステップごとの xclbin」にする (`PerStep`)。演算 (ステップ) ごとに、xclbin のカーネル (opcode 3、命令列、その長さ、バッファ) を XRT で 1 回ずつ投入する。設定を共有するので xclbin は少なく、Gemma 4 E2B のデコードは 1095 ステップで xclbin 2 個、プロンプトの塊は 1096 ステップで 4 個
- 呼び出しごとの値: xclbin にはスクラッチパッドがないので、値は「ディスパッチ時のスカラー」になり、命令列が呼び出しごとに作り直される。mlir-aie は設計ごとに、スカラーから TXN を組み立てる C++ (`dispatch_gen.cpp`、型付きの代入、算術、三項演算子、断る条件、`aie_runtime::txn_append_*`) を生成して共有ライブラリにする。これは Linux 専用で Windows では使えず、近似 (語ごとのあてはめ) は三項演算子 (`s > 1 ? 511 : 0`) で崩れる。そこで書き出し時にこの C++ を命令列プログラム (JSON の命令の並び、`tools/amd/iron_stream.py`) に直し、hwnpu は Go の解釈器 (`internal/llm/iron/stream.go`) で命令列を作る。C の型変換 (符号拡張、切り捨て、通常の算術変換) を写す。知らない行は書き出しで断る。スカラー自体は full ELF と同じ語の式 (Affine、Linear、MHA の導出) で、Go の既存の計算をそのまま使う
- 照合: 書き出しは各プログラムを、生成ライブラリと同じスカラー (デコードの 13 位置、プロンプトの 8 通りの開始と行数) で比べ、違えば止まる。最初の版で 2 つ形が見つかった (配列の初期化子、シフトの幅を守る三項演算子)。キャッシュにある 64 本の生成 C++ はすべて変換でき、Gemma の 49 本 (NPU2 向け) と NPU1 向けのプログラムはすべて一致した。`check/streams.json` の 64 本は Go の解釈器でも一致する (`TestExportStreams`)
- 置き場の違い: PerStep では、常駐するもの (重み、KV キャッシュ) はアリーナのバンクにあるが、一時的なものは版ごとの別のプールにあり、位置が 0 から数え直される。どちらも `scratch` という種類なので、最初の書き出しは一時的なものを重みの上に置き、出力は `<pad>` ばかりになった。一時的なものは引数 `pool` (、`pool_1`…) に置く
- 実行 (`internal/llm/iron/steps.go`): xclbin はモデルで 1 回ずつ読み込む (ハードウェアコンテキスト)。各ステップのランは読み込み時に作り、バッファはバンクの一部として結ぶ (`Run.SetBufferAt` を足した。xdna はアドレスに足し、XRT はサブバッファを作る)。同じコンテキストのコマンドは順に走るので、待たずに続けて投入し (16 個ごとに待つ)、xclbin が替わる前に直前の 1 つを待つ。窓を 1 にすると 12% 遅く、64 にしても変わらない
- 命令列の置き場: ステップごとにバッファを作ると 2191 個で 64 MiB のデバイスヒープが尽きた。版と xclbin の組ごとに、静的な命令列のバッファと呼び出しごとの命令列のバッファを 1 つずつ持ち、各ステップは 64 バイト境界のスロットを使う。xclbin ごとに分けるのは、XRT が命令列の引数を xclbin ごとのメモリグループ (65537、131073… 上位 16 ビットが xclbin の番号) に置き、複数のグループに結んだバッファを丸ごと複製するため。XRT は待っていないランを、終わっていても再投入できない (`bad command state`) ので、再投入の前に待つ
- 検証 (Ryzen AI 9 365): 同じ形の書き出しを NPU2 向けに作り (`--per-step`)、xdna と XRT の両方で Gemma 4 E2B が正しく答えた (「red, yellow, and blue」、hwnpu-chat で「Tokyo」「Paris」)。デコード 2.6 tok/s (full ELF は 6.1)、プロンプト 14 トークンで 1.3 秒。NPU1 向けの書き出し (`--device npu1 --cols 4`、xclbin 6 個) は、4 列の模擬 NPU で読み込みと全ステップの命令列の作成が通る (`TestExportLoads`)。NPU1 の実機では未検証
- 残り: NPU1 は版の xclbin が計 6 個で、ハードウェアコンテキストの上限 6 を使い切る (他のプロセスが NPU を使っていると作れない)。超える書き出しには、コンテキストを入れ替える仕組みが要る。塔 (画像と音声) は IRON が NPU1 で動かさないので対象外

### 16.58 塊ごとの注意を測る: 256K を既定の文脈にできるか (`npuprobe attn`) — 2026-11-04

- 動機: LFM3-1.7B の文脈の上限 262144 を、使わない会話を遅くせずに常に既定で出したい。今の方式は 1 ステップの費用が容量に比例し (16.16、kv-cache.md 3 節)、64K で 363 ms/token (16.22)。model-fit.md T2 (層を注意の前後で分け、重みなしの注意を使っている塊だけ呼ぶ) が前提になるので、その注意のプログラムを先に測った。Prefrontal-AI の Latent Sparse Attention (鍵を潜在空間でクラスタに分け、問い合わせは重心を採点して選んだ塊だけ読む) も、この「塊ごとに呼ぶ」形の上に載る
- プローブ (`internal/probe/attn.go`、`cmd/npuprobe/attn.go`): 重みなしで 1 塊に対する注意を計算し、塊の中で正規化した出力 o、行ごとの最大値 m、指数の和 l を返す。指数の和は 1 との行列積 (16.22 の `reduce_sum` の配置の揺れを避ける)。ホストは `probe.ChunkMerge` で float32 のまま合わせる (Flash-Decoding の合流を逐次に)。1 ステップは、6 層それぞれの使っている塊を別々のバッファから付け替えて呼ぶ。`TestChunkAttnMergeMatchesFull` (CPU インタプリタ、4 塊、一部と全部が隠れた塊、Q=3 の因果マスク) は 1 回の softmax と相対誤差 1.4e-4。最大値が変わったときの掛け直しを外す改変と、ヘッドの並びを取り違える改変を検出する
- 見つかった不具合: Core ML の `request.Bind` は、一度でも投入したリクエストを、終わっていても「実行中」として付け替えを断っていた。今までの生成はバッファを最初に 1 回付けるだけなので当たらなかった。`Submit` と Level Zero と同じく、終わっていれば付け替えるようにした (`TestBindAfterSubmissionFinished`)
- 配置 (M3、macOS 27.0.1。grouped): Core ML は仕事の少ない形を CPU に置く。Q=1 は塊 8192 まで CPU、16384 で ANE。Q=8 は 2048 まで CPU、4096 から ANE。Q=64 は 1024 から ANE。tile (K/V を問い合わせの頭ごとに複製) は Q=1 でも 4096 から ANE で、計算が多い形ほど ANE に置かれる
- 結果 (`npu_profiles/npuprobe-m3-attn.md`。6 層分の注意、1 ステップの ms。合わせる時間を含む):

| Q | 塊 | 配置 | 4096 使用 | 16384 | 65536 | 262144 | 1K あたり (16K〜64K) | うち合わせる時間 (262144) |
|---|---|---|---|---|---|---|---|---|
| 1 | 8192 | CPU | 3.4 | 6.6 | 26.4 | 115.8 | 0.41 | 0.8 |
| 1 | 16384 | ANE | 12.7 | 12.7 | 51.0 | 217.5 | 0.80 | 0.4 |
| 8 | 2048 | CPU | 4.8 | 18.9 | 75.2 | 325.8 | 1.18 | 24.5 |
| 8 | 4096 | ANE | 5.1 | 20.1 | 80.7 | 392.2 | 1.26 | 12.0 |
| 8 | 8192 | ANE | 8.8 | 17.4 | 69.5 | 295.7 | 1.09 | 5.7 |
| 64 | 2048 | ANE | 16.8 | 66.3 | 264.4 | 1044.7 | 4.13 | 180.6 |
| 64 | 8192 | ANE | 28.4 | 55.6 | 224.3 | 906.6 | 3.50 | 62.6 |

- 分かったこと
  - **付け替えに費用はない**: ステップの中の 1 回 (合わせる時間込み) は、1 塊だけを付けた 1 回とほぼ同じ (Q=8、塊 8192 で 1.45 対 1.38 ms。差の大半は合わせる時間)。塊ごとに別のバッファを持ってよい
  - **費用は使った長さに比例する**: 1K あたりの時間は 16K から 64K で一定 (Q=8、塊 8192 で 1.087、1.086 ms)。262144 で 5〜50% 増える組がある (Q=64 の塊 4096 は 1 回 2.27 → 3.57 ms、塊 16384 は 9.3 → 13.4 ms。塊 8192 は増えない)。16 GB の機械で 3 GiB を持つことによるメモリの圧迫か、ANE が写すバッファの数かは未確認
  - **入力の複製はない**: 最大 RSS は KV のバイト数 + 0.1 GiB 前後 (262144 で KV 3.0 GiB、RSS 3.05〜3.17 GiB)。Core ML は `initWithDataPointer` のバッファをそのまま使っている
  - **Q=1 では CPU が ANE の 2 倍速い** (0.41 対 0.80 ms/1K)。行列とベクトルの積は ANE が苦手で (kv-cache.md 3 節)、Core ML の配置の判断は速さの面では正しい。Q=8 では CPU (塊 2048) と ANE (塊 8192) がほぼ同じ (1.18 対 1.09)。生成は下書きの確認で Q=8 なので、ANE のままで損はない
  - **合わせる時間**: Q=1 と Q=8 では 2% 以下 (塊 8192)。Q=64 は行が 1024 あるので、塊 2048 で 17%、8192 で 7%。prefill では塊を大きくするか、行ごとに並列にする
- 見込み (lut6 + grouped の今の既定。重みの部分を 42.8 ms (16.41、容量 16320) から注意の分を引いた約 25 ms とし、Q=8、塊 8192 の注意を足す。層を分ける呼び出しの増え (約 7 回 × 55 µs) と、重みありと重みなしのプログラムの切り替え (未測定) は含まない):

| 使った長さ | 今 (容量で払う) | T2 の見込み |
|---|---|---|
| 4K | 42.8 ms (容量 16320) | 約 34 ms (塊 4096 なら約 30 ms) |
| 16K | 42.8 ms | 約 42 ms |
| 64K | 363 ms (lut4 + tile、16.22。grouped の塊は未測定) | 約 95 ms |
| 256K | (動かしていない。容量の比例で 1 秒超) | 約 320 ms (約 3 tokens/s) |

  プロンプトの読みは Q=64、塊 8192 で 1K あたり 3.5 ms なので、256K を一度に読むと注意だけで約 30 分 (平均 128K × 3.5 ms × 4096 回)。prompt store と退避が前提になる
- 結論: T2 の注意の側は合格。256K の容量を常に持っても短い会話は今より遅くならず、長い会話は数倍速くなる見込み。メモリは使った分だけ (塊を使い始めたときに確保すれば、1 トークン 12 KiB、256K で 3 GiB)。塊は 8192 (Q=8 と Q=64 の両方で ANE、Q=8 で最速、合わせる時間も小さい) を第一候補にする
- M4 でも配置の境目はほぼ同じだった (ユーザーの計測。Q=1 は塊 16384 から、Q=8 は 4096 から ANE)。境目は機種ではなく Core ML (macOS 27) の判断で決まっているとみる。塊は 16384 を超えられない (1 次元の上限、16.16) が、1 回の時間は鍵の数にほぼ比例し、呼び出しごとの固定費は塊 16384 で数 % なので、大きくしても得はない
- 切り替えの費用 (`npuprobe seam`、`npu_profiles/npuprobe-m3-seam.md`): lut4 の 40 層 (1.13 GiB、lut6 の LFM3-1.7B と同じくらい) を 7 部分にし、その間に 6 層分の塊ごとの注意 (grouped、塊 8192) を挟んで全部を同時に読み込み、1 ステップを 3 通りで測った (それぞれ 2 回のうち速いほう):

| Q | 使用 | 重みだけ | 注意だけ | 交互 | 交互 − 和 | 境目 1 つ (12 回) |
|---|---|---|---|---|---|---|
| 8 | 8192 | 20.21 | 8.83 | 29.01 | −0.03 | −0.003 |
| 8 | 65536 | 20.11 | 70.57 | 95.33 | +4.65 | +0.387 |
| 64 | 8192 | 26.84 | 28.72 | 49.91 | −5.65 | −0.470 |
| 64 | 65536 | 27.08 | 210.04 | 239.78 | +2.66 | +0.221 |

  - **載せ直しは起きない**: 起きれば境目 1 つで数十 ms (16.36 の 35 ms)、1 ステップで 400 ms を超える。8192 使用では交互と和の差は 0 か負で、境目の固定費はない
  - 65536 での増え (Q=8 で +4.65 ms、注意の 6.6%。Q=64 で +2.66 ms、1.3%) は使った長さとともに増えるので、境目の固定費ではなく、重みと交互にしたときに注意が少し遅くなる分とみる (768 MiB の KV と重みの読み出しが交互になること、あるいは揺れ)。16.58 の注意だけの 262144 での遅くなりと合わせて、本実装で測り直す
  - 見込みの確かめ: Q=8 の交互は 8192 使用で 29.0 ms、65536 で 95.3 ms。上の見込み (約 34 ms と約 95 ms) に合う。今の lut6 + grouped は容量 16320 で 42.8 ms
  - 結論: T2 の合格条件 (境目 1 つ 0.2 ms 以下) は、固定費としては満たす。本実装に進む
- 次: (1) `lfm3.BuildCached` を注意の前後で分け、`decode.Session` が塊を使い始めたときに確保し、塊ごとに呼んで合わせる。(2) `Session.Save` を使った行だけの写しにする (今は丸ごと複製で、256K では 3 GiB ずつ)。(3) 262144 での遅くなりと、交互にしたときの注意の数 % の遅くなりの原因

### 16.59 LFM3 を注意の前後で分け、文脈 262144 を既定に (`-paged`) — 2026-11-04

- 動機: 16.58 で、塊ごとの重みなしの注意 (費用は使った長さに比例、付け替えの費用も入力の複製もない) と、重みのある部分との境目に載せ直しがないことを確かめた。これを本実装にし、LFM3-1.7B の文脈の上限 262144 を既定にする
- 変換 (`hwnpu-convert -paged N`、`llm.Options.Paged`): `llm.Paged` のモデル (LFM3) を、全注意の層の前後で部分に分ける (`llm.PlanPaged`。LFM3-1.7B は層 2、6、10、14、18、22 で 7 部分。重みの予算を超える区間は、その中の別の層でさらに分ける)。全注意の層 N の前で終わる部分は、層 N の問い合わせ attn_q_N (RoPE 済み)、そのステップの鍵と値 kv_k_N_new と kv_v_N_new、ゲート attn_gate_N を出し、y はその層に入る隠れ状態。次の部分は、注意の出力 attn_o_N とその隠れ状態から層 N の残り (ゲート、o_proj、MLP) を計算する。ゲートは名前で前の部分の出力に結ぶ (`internal/llm/chain` の既存の規則)。スライディング層のリングと畳み込みの窓は今までどおり部分の中。`out.attn.mlpackage` は塊 1 つに対する重みなしの注意 (`internal/llm/paged`、16.58 のプローブと同じ形)。manifest に `paged` と `attention` を足した
- 実行 (`decode.Session`): 分けたプログラム (`Program.RunPart`、`Attend`、`Attn`、`Context`) を段ごとに動かし、段の間で、そのステップの鍵と値を自分が持つキャッシュに書き (注意の前に書くので、そのステップのトークン自身も塊の中で見える)、使っている塊ごとに注意を呼んで float32 で合わせ (`paged.Merge`)、attn_o_N に書く。塊は位置が届いたときに確保する (`Attention.NewBuf`、Core ML ではデバイスのバッファ)。マスクが要るのはそのステップのトークンが入る塊 (多くて 2 つ) だけで、ステップごとに 1 回作り、それより前の塊にはゼロのマスクを使う。`Mark` はこれまでどおりリングと窓だけ、`Save` は位置までの行 (`ExportRows`) だけを写す (分けないモデルも同じ。これまではキャッシュ全体の複製)。`Limit` で上限を下げられ、`Close` で塊を返す
- チャット (`chat.Prepare`): `llm.Paged` のモデルを Neural Engine 向けに変換するときは、既定で塊 8192 で分ける (`-paged`、`-1` で分けない)。`-context 0` (新しい既定) はそのときモデルの `max_position_embeddings` (LFM3-1.7B は 262144)、分けないモデル (Gemma 4、Intel の NPU) は 16320。分けたモデルには段階 (`-tier`) がない。名前は `lfm3-lut6-grouped-p8192-c262144-q8.parts.json`。エンジンは、キャッシュが文脈いっぱいで機械のメモリの 4 分の 1 を超えるなら塊単位で文脈を縮め (`internal/mem.Physical`)、ready の行に増え方を出す (`its cache growing with the conversation by 12 KiB a token to 3.00 GiB`)。使わなくなったセッションの塊は返す
- CPU のインタプリタの runner は、部分に分けたモデルと注意で分けたモデルを動かせるようにした (エンジンのテストと、NPU のない機械の `hwnpu-generate` のため)
- 検証 (Linux、CPU のインタプリタ):
  - `TestPagedMatchesFull` (小さな LFM3: 畳み込み、全注意 2 層、スライディング 1 層): 1〜4 トークンのステップ、塊の境をまたぐステップ、容量が塊の倍数でない場合、grouped と tile で、キャッシュなしのプログラムとロジットの相対誤差 8.6e-4 以内。塊は位置が届いた分だけ確保される
  - `TestPagedShareRewindResume`: 8 トークンと 1 トークンのプログラムが 1 つの系列を続け、印、`Save`、書き出した行からの `Resume` で戻っても同じ。`TestEngineReusePaged`: エンジンのプロンプトの再利用 (印、退避) が分けたモデルでも読み直しと同じロジット
  - 改変の検出: マスクが自分の鍵を隠す、最初の部分的な塊のマスクを付けない、1 ステップに 1 行しか書かない、`Save` が行を持たない、のどれも上のテストが落ちる
  - 実物の LFM3-1.7B (`TestPagedCheckpoint`、`HWNPU_LFM3_DIR` と `HWNPU_LFM3_PAGED_CPU=1`、fp16、文脈 64、塊 16、1 回 8 トークン): 分けたもの (7 部分) と分けないものを同じ重みで CPU で動かし比べた。45 トークン (塊 3 つにまたがる) の 6 ステップすべてで次のトークンが一致し、ロジットの相対誤差は 3.6e-4〜1.7e-3 (どちらもキャッシュは fp16)。分けたほうが持ったキャッシュは 576 KiB = 6 層 × 鍵と値 × 塊 3 つ × 4 ヘッド × 16 × 128 × 2 バイトで、使った塊の分だけ。変換は 2 つで 28 秒、1 ステップは CPU でどちらも約 10 秒
- Mac で確かめること (下の「M3 での確認」で実施): (1) `go test ./internal/llm/lfm3 ./internal/llm/paged ./internal/chat` (2) `hwnpu-chat` の初回の変換と読み込み (7 部分 × 2 と注意 2 本が全部 Neural Engine に載るか、読み込みの時間、Neural Engine のコンパイルのキャッシュに残るか)。(3) 短い会話と長い会話での 1 トークンの時間 (見込みは 4K で 30〜34 ms、16K で 42 ms、64K で 95 ms、262144 で 320 ms) と、メモリ (`footprint`)。(4) 長い文脈の先頭に置いた合言葉を答えるか (16.22 と同じ試験を 128K と 256K で)
- **M3 での確認** (MacBook Air M3 16 GB、macOS 27.0.1、lut6 + grouped、塊 8192、`-draft 8`、2026-11-04):
  - テスト: 引き継ぎの 7 パッケージは通る。`TestChunkAttnOnNeuralEngine` の配置は 16.58 と同じ (grouped の Q=1 は 4096 まで CPU で 16384 は ANE、Q=8 の 1024 は CPU、Q=64 は 1024 から ANE)
  - **見つかった不具合**: 端末の hwnpu-chat の答えが空 (0 トークン) になり、次の質問で `attention of layer 2 over chunk 0: coreml: cannot bind while a submission is running`、終了時に SIGSEGV (`Session.Close` がバッファを解放した後に予測が走っていた)。`chain` の注意のプログラムが読み込みの文脈を持ち続け、それで `Run` を待っていた。hwnpu-chat は `signal.NotifyContext` の下で読み込み、モデルが上がると止めるので、注意の呼び出しはどれも投入した直後に `context canceled` を返し、予測は裏で走り続けた (端末は `Canceled` を表示しない)。部分のほうは `Program(context.Background())` の文脈で動いていたので当たらず、CPU の runner にも同じ注意書きがある (16.42 の `runner.Open` と同じ形の取り違え)。注意のプログラムも段の `Program` の文脈で動かすようにした。`TestPagedOnNeuralEngine` (`internal/chat`、darwin): 小さな LFM3 を分けて Neural Engine に読み込み、読み込みの文脈を取り消してから塊 3 つにわたって CPU と比べる。相対誤差 0.004。修正を外すとクラッシュする
  - 変換と読み込み: 初回の変換は q8 と q64 がそれぞれ 66 秒と 67 秒 (各 7 部分 0.09〜0.27 GiB、計 1.21 GiB と注意のプログラム)。初回のコンパイルと読み込みは 62.0 + 60.2 秒で、14 部分と注意 2 本すべて `all on the NPU`。ready の行は見込みどおり `context 262144 tokens, its cache growing with the conversation by 12 KiB a token to 3.00 GiB (attention in chunks of 8192)`
  - コンパイルのキャッシュ: 2 回目の起動は 21.8 + 24.5 秒、3 回目と 4 回目は 1.0 + 1.2 秒 (起動から 4 問の終わりまで 16 秒)。部分が 2 本から 14 本 + 注意 2 本に増えても、2 つのプログラム (2.4 GiB) は Neural Engine のキャッシュに残る。ただし分けない形 (`-paged -1 -context 16320`) を一度起動すると、それは 2 回とも 33〜40 秒かけてコンパイルし直し、その後の分けた形も 32 秒かかった (4 本 4.8 GiB で追い出し合う、16.39 と同じ)。その後のサーバーの起動も 27 + 17 秒、次の起動から 1.1 + 1.1 秒に戻った
  - 短い会話 (16.39 の 4 問、貪欲法、思考なし、質問ごとに `/reset`): 答えは分けない形と一字一句同じ (富士山 3,776 m、17 × 23 = 391、『こころ』は「宮本明」、日本一長い川は「利根川」。16.39 の lut6 と同じ誤り) で、下書きの採用数も同じ。速さ (下書きの採用込み、分けた形は 4 回、分けない形は 2 回の起動の幅):

| 問 | 分けた形 (p8192-c262144) | 分けない形 (c16320) |
|---|---|---|
| 富士山の高さ (41 トークン) | 30.8〜35.3 tokens/s | 18.9〜20.8 |
| 17 × 23 (135) | 34.8〜36.0 | 26.5〜27.6 |
| 『こころ』の作者 (70) | 31.4〜32.1 | 23.5〜25.1 |
| 日本一長い川 (212) | 34.2〜34.4 | 25.3〜26.8 |

    1 トークン 28〜33 ms で、見込み (4K で 30〜34 ms) の内側。分けない形の 1.3〜1.7 倍の速さ
  - 長い会話 (サーバー、`-prompt-cache off -think=false`、先頭に合言葉を置いた 16.22 の試験。大きさごとにサーバーを起動し直す。思考ありでは 40 トークンをすべて考えに使い、答えが空になるので思考なしにした。考えの中には合言葉が出ていた): 4 つとも `BLUE-OTTER-42` と答えた。続けて同じ会話で「浜辺の晴れた日を 100 語ほどで」と頼み (最大 128 トークン、下書きはほぼ採られない)、その生成の時間を測った。メモリは読み込みの後の `footprint` の phys_footprint:

| プロンプト | 読む時間 | 読む速さ | 生成 | 1 回 (Q=8) | 見込み | phys_footprint |
|---|---|---|---|---|---|---|
| 16,017 | 18.7 s | 856 tokens/s | 42.7 ms/token | 42.9 ms | 約 42 ms | 416 MB |
| 63,878 | 147.2 s | 434 tokens/s | 92.8 ms/token | 97.4 ms | 約 95 ms | 996 MB |
| 127,737 | 513.7 s (8.6 分) | 249 tokens/s | 165.3 ms/token | 174 ms | — | 1,771 MB |
| 250,743 | 1752.2 s (29.2 分) | 143 tokens/s | 320.3 ms/token | 350 ms | 約 320 ms | 3,214 MB |

  - 分かったこと
    - **生成はほぼ見込みどおり**: 16.58 の見込み (重み約 25 ms + Q=8、塊 8192 の注意 1.09 ms/1K) に対し、1 回の時間は 16K で 42.9 対 42.0 ms、64K で 97.4 対 93.0、128K で 174 対 161、256K で 350 対 292 (16.58 の表の 262144 の 295.7 ms から立てた見込み 320 に対しては +9%)。1K あたりの傾きは 16K〜64K で 1.14 ms、64K〜128K で 1.20、128K〜256K で 1.43 と、長いほど大きくなる。64K までの 5% 増しは 16.58 の seam の「重みと交互にしたときに注意が少し遅くなる分」、128K を超えての増えは 16.58 の注意だけの計測にもあった 262144 での遅くなり (この組では注意だけでは出なかった) か、3 GiB を持つことのメモリの圧迫とみる。原因は確かめていない。下書きの採用 (256K で 235 のうち 10) を込めた 1 トークンは 320 ms で、見込みの値に並ぶ
    - **プロンプトの読みも見込みどおり**: 256K で 29 分 (見込みは注意だけで約 30 分)。2 回目からはプロンプトの保存 (`-prompt-cache`) が前提
    - **メモリは使った分だけ**: phys_footprint は 16K から 256K まで 2.8 GB 増え、235K トークン × 12 KiB = 2.7 GiB とほぼ同じ。スワップの使用量は全体で 5.5〜6.3 GB の間で揺れただけで (ほかのアプリの分)、256K で大きく増えてはいない。16 GB の機械の 4 分の 1 (4 GiB) に 3 GiB は収まるので、文脈は縮められない
    - 長い読み込みの直後の答えの最初の数回は遅い (256K で 8 回 10.3 秒、1 回 1.29 秒。続く会話では 1 回 350 ms。128K でも 260 対 174 ms)。読み終えた直後の Q=64 から Q=8 への切り替えか、読みの間に押し出されたページの戻りとみるが、確かめていない

### 16.60 堂々めぐりを見つける: 周期の長い繰り返しと、少しずつ変わる繰り返し (`chat.repeats`) — 2026-11-04

- 動機: M3 で hwnpu-chat に「東京の23区教えて」と聞くと、目に見えてループした。分けたモデル (16.59) のせいではない: 温度 0 で分けない形 (`-paged -1 -context 16320`) も同じところで同じように迷い、ほぼ同点の 1 か所で分かれるだけ。量子化のせいでもない (`-format fp16` も同じく迷い、「ソト区」「新川区」のような区を作る)。英語 ("List the 23 special wards of Tokyo.") でも同じ。答えを知っている問い (富士山の高さ、17 × 23、都道府県の数) は思考ありで 109〜242 トークンで終わる。1.7B のモデルは 23 区を覚えておらず、思考の中で一覧を書いては「Sumida is 足立? No…」と疑って最初から書き直し、1024 トークンの予算で切られ、作った表を答える (LFM3 の README も知識の要る仕事には検索を勧める)。`-rep 1.1` (HF の推奨) は字面の繰り返しを弱めるだけ、思考なしは迷わないが 14 個の誤った一覧。答えの中でも書き直しが続き、47 都道府県の表では答えが「- Gifu? Already.」「- Mie? Already.」を名前を変えながら何十行も繰り返し、`-n 4000` に届いた。既定の `-n 0` と文脈 262144 (16.59) では、そうした答えが何時間も続きうる
- 既定で答えの長さに上限を置くのはやめた: 長い答えが要るときに途中で切れる。繰り返しを見つける。それまでの `looping` は直近 64 トークンが周期 16 以下の同じ並びのときだけで、周期の長いものや少しずつ変わるものは素通りする
- 区別のつけ方 (実物の出力で測った): 本当の答えも繰り返す。下書きを清書する (思考で書いた 1〜120 の並びを答えに書く。思考の中でコードを書き直す)、テストが同じ準備の行を持つ、JSON の鍵、表の行。堂々めぐりとの違いは、同じものに戻る回数: 清書は 2 回目、堂々めぐりは 3 回目以降が続く。「前に 1 回出た」の割合は 23 区の思考で 0.50〜0.92、Python のクラスを書く正常な思考で 0.58 と重なり分けられないが、「前に 2 回以上出た」に絞ると正常な 24 の思考と答え (コード、表、JSON、計算、論理パズル、手順) で行は最大 7/16、8 トークンの並びは最大 50/256
- `chat.repeats` (repeat.go): 行と、8 トークンの並びの 2 つで数える
  - 行: 単語 (字を含むトークン) が 2 つ以上の行を、トークンの並びのハッシュで数え、直近 16 行のうち 12 行が 3 回目以降なら繰り返し。単語のない行 (`}`、`---`、数だけの表の行) は数えない。数字は区別したまま: 数字だけが違う行は、ループと同じくらい表のことが多い (数字をまとめると「| # | # |」や日付と天気の表で止まる)
  - 並び: 直近 256 位置のうち 230 位置で、そこで終わる 8 トークンが 3 回目以降なら繰り返し (改行のないループ。周期は問わない)
  - 思考と答えは別に数える (`<think>`、`</think>` で数え直す)。思考で 2 回書いたものを答えでもう 1 回書いてよい
  - 見つけたとき: 答えの中なら今までどおり止める (`Stats.Looped`、`stopped repeating itself`)。思考の中なら、予算に届いたときと同じく閉じて答えさせる (`Stats.ThinkLooped`、`thinking cut going round in circles`)。`looping` も思考の中ではこちらにした (今までは思考の途中で止まり、答えが空になった)
  - トークンの分類 (改行を含むか、字を含むか) は最初の生成でトークナイザの語彙から作る。トークナイザのないエンジン (テスト) は並びだけで数える
- 確かめたこと: これまでのログを同じトークナイザでトークンに戻して流すと、正常な 24 の思考と答えでは一度も見つけず、23 区の思考 2 つ (一覧の書き直し) で見つけた (898、680 トークン目)。実機 (M3、`-n 0`) で 47 都道府県 2 回と 23 区 2 回: 都道府県は答えが書き直しの後に「山梨県庁所在地は「山梨県庁所在地は「…」に入って止まった (1377、5115 トークン。これは `looping` でも止まる形)。23 区の 1 回目は、答えが思考のように「- 千代田区」「- 港区」「- 江東区」「- 江東? No.」を散文をはさんで 15〜17 回書いたところを行で見つけて止まった (4715 トークン。`looping` では止まらない形)。2 回目は自分で終わった (3442 トークン)。見つけるまでの長さは控えめにした分だけかかる (23 区の 1 回目は答えが約 3700 トークン)。正常な答えを切らないことを優先した
- テスト (`repeat_test.go`): 20 の名前を少しずつ順を変えて回す行は 2 周より後に見つけ、40 行の下書きの清書、同じ準備の行を持つテスト 20 個、数字だけが違う 120 行は見つけない。改行のない周期 100 の並びは 200 トークンより後に見つける。数え直しの後は前の回数を持ち越さない。`TestThinkLoop`: 合成のモデルの貪欲法は思考の中で周期 4 に落ち、思考が予算 300 より前に閉じて答えが始まり、答えでまた繰り返して止まる

### 16.61 2 段の注意: 128K 以上で読む鍵を減らす (`decode.Sparse`、`-sparse`) — 2026-11-04

- 動機: 16.59 で LFM3-1.7B の文脈 262144 が既定になったが、1 トークンは 128K で 165 ms、250K で 320 ms と長さに比例する (M3)。Prefrontal-AI の Latent Sparse Attention (arXiv 2610.42147: 鍵を RoPE のない次元で 64 次元に射影し、4 つずつ束ねてバケットに分け、問い合わせに近いバケットだけを正確に読む。蒸留した射影は Qwen3.8-27B と GLM-5.3-Flash のみ) の考えを、蒸留なしで LFM3 に使えるかを CPU で確かめてから実装した。LFM3 の全注意の層は RoPE を半分の次元 (32〜63、96〜127) にしか掛けないので、LSA の前提に合う
- **CPU での確認** (aix1、PyTorch bf16、実物の LFM3-1.7B。スクリプトと結果は `~/.cache/hwnpu/dev/results/lsa-2026-11-04/`):
  - 最後の 64 トークンだけを疎にした試験 (プリフィルは全部を読む、深さ 10% に合言葉 1 つ): 鍵を 1 つずつ選ぶ (top-k) のはだめ。理想の top-4096 (本物のスコアで選ぶ) でも 128K で注意の重みの 53% しか拾えず、KL 0.58。RoPE のない次元での k-means (LSA の形) は 8 トークンずつ区切ったバケットと同じで、Quest (ページの最小と最大の上界) は悪い。**2 段** (8 トークンのバケットの平均の鍵で重さを測り、重いバケットの鍵 16384 を正確に、残りのバケットは平均の鍵と値 1 つずつに log 8 を足して読む。最初の 4 つと最後の 512 は正確) は 64K で KL 0.018、128K で 0.052、答えも保った。top-k で同じくらいにするには 32K の鍵が要る
  - 128K、合言葉 3 つ (深さ 10%、50%、90%)、文書はこのリポジトリの docs (日本語と英語) と Go のソース。合言葉は貪欲法で一字一句出るか、損失はプリフィルの最後の 8K トークンの次トークン:

| プリフィルとデコード | 損失の増分 | 通常の注意との一致 | 10% | 50% | 90% |
|---|---|---|---|---|---|
| 通常の注意 | 0 | — | ○ | ○ | ○ |
| プリフィルは全部、質問だけ 2 段 | 0 | — | ○ (KL 0.06〜0.07) | ○ | ○ |
| 2 段、1 回 256 トークン | +0.022 | 81% | × | ○ | ○ |
| 2 段、1 回 64 トークン | +0.0094 | 84% | ○ (KL 0.105) | ○ | ○ |
| 2 段、プリフィルだけ正確な鍵 32K | +0.0093 | 87% | ○ (KL 0.081) | ○ | ○ |
| 2 段、64 トークンと 32K の両方 | +0.0068 | 89% | ○ (KL 0.071) | ○ (0.062) | ○ (0.068) |
| 2 段、先頭 4K を常に正確に | −0.027 | 82% | × | ○ | ○ |
| 2 段、16 トークンのバケット | +0.053 | — | ○ | ○ | ○ |
| 2 段、正確な鍵 4096 | — | — | × | ○ | ○ |

  - 別の文書 (英語の mlir-aie と IRON のドキュメント 43 ファイル、16 万トークンから 128K。合言葉と質問は同じ日本語):

| プリフィルとデコード | 損失の増分 | 通常の注意との一致 | 10% | 50% | 90% |
|---|---|---|---|---|---|
| 通常の注意 | 0 | — | ○ (log p −0.48) | × (−1.85) | ○ (−0.38) |
| 2 段、1 回 64 トークン | −0.136 | 87% | ○ (KL 0.344) | ○ (0.385) | ○ (0.352) |
| 2 段、64 トークンと 32K の両方 | −0.112 | 90% | ○ (KL 0.279) | ○ (0.321) | ○ (0.282) |

  - この文書では通常の注意が 50% の合言葉を外し、2 段のほうが損失が低く 3 つとも当たった。ただし通常の注意からは離れている (質問の KL 0.28〜0.39、日本語の文書の 4 倍)。128K で通常の注意自身が遠くに気を散らしていて、2 段がそれを絞っているとみるが、確かめていない (先頭 4K を正確にした形の損失が下がったのも同じ向き)。どちらの文書でも、プリフィルの正確な鍵を 32K にすると通常の注意に近づく (一致 +3〜5 ポイント、KL −20〜30%) ので、既定にした

  - 誤差はプリフィルで積もる (256 トークンを 1 つの選択で読むと遠い合言葉を落とす)。NPU のプリフィルと同じ 64 トークンずつなら保つ
  - 256K では通常の注意でも 10% と 50% の合言葉を外した (log p −3.64、−2.20。90% は −0.37 で当たる)。2 段のデコードは通常の注意と同じ当たり外れで、KL 0.16〜0.19。LFM3-1.7B が確かに使えるのは 128K くらいまでで、256K はモデル自身の限界
- 設計 (Neural Engine の重みなしの注意のプログラムの上に、16.59 の形のまま):
  - 要約: 全注意の層ごと、kv ヘッドごとに、窓より前の 8 位置ずつのバケットの鍵の平均と値の平均。塊 8192 と同じ形のデバイスのバッファ (1, kv_heads, 8192, head_dim) に、バケットが窓から出るたびに足す (ホスト、fp32 で平均して fp16 に)
  - 重さ: 問い合わせの各行の、全部の要約に対する softmax を、kv ヘッドの行と問い合わせのヘッドで足したもの (CPU の確認と同じ)。正規化には全部の要約の log-sum-exp が要るので、2 回に分ける。(1) 全部の要約への注意 (今までの注意のプログラム、マスクは数える要約に log 8、ほかは −1e4)。これは要約の分の注意の出力にもなる。(2) mass のプログラム (`paged.BuildMass`): q、要約の鍵、同じマスク、各行の −log-sum-exp (`Merge.Norm`) から、要約ごとの重さ (1, kv_heads, 1, 8192) を出す。行の和は 1 との行列積 (reduce_sum は配置が揺れたため、16.58)
  - 選択: kv ヘッドごとに重い順に、正確に読む塊 (`Exact`、既定 2 = 16384 の場所) に窓と呼び出し自身が収まる数 (1975 バケット + 最初のバケット)。ホストで選択アルゴリズム (並べ替えなし)。前の選択と同じバケットはそのまま置き、変わった分だけ写す
  - 正確に読む塊: kv ヘッドごとに違うバケットを同じ場所に並べ、その後ろに窓と呼び出し自身 (どのヘッドも同じ位置)。マスクはヘッドに共通 (バケットは全部見え、後ろは因果)。今までの注意のプログラムで 2 回
  - 正確に読んだバケットの要約: (1) で数えた分を `Merge.Sub` で引く (1 塊、1 回)。ヘッドごとのマスクを持つプログラムを作らずに済む。引いた残りの誤差は fp16 の丸め程度で、全体の和に対する割合で効く (負になれば 0 にする)
  - 1 呼び出しの回数: 要約の塊 × 2 + 1 + 2。全部を読むよりこれが少ないときだけ 2 段にする (塊 8192 で約 48K から)。128K で 16 → 7、256K で 32 → 11
  - プリフィル (64 トークンのプログラム) の正確な塊は `Prefill` で別に決める。既定は 4 (32768、上の確認で最もよかった形)。呼び出しは要約の塊 × 2 + 1 + 4 になり、約 57K から 2 段で読む
  - `Refresh` で選び直しを数回に 1 回にできる (mass の回数が減る)。実物では確かめていないので既定は毎回
- 実装:
  - `internal/llm/paged/mass.go`: `BuildMass`、`WriteMass`、`Merge.Norm`、`Sub`、`AddMerge`
  - `internal/llm/decode/sparse.go`: `Sparse`、`DefaultSparse`、`Session.SetSparse`、`attendSparse` (`runPaged` から層ごと)、`summarize`、`choose`、マスクの使い回し (変わる列だけ書く)。キャッシュに書くとき (`run`、`ImportRows`、`Reset`) にその位置から先の要約を捨てる (`sparseTouch`)。`HeldBytes` と `Close` に要約と写しのバッファを含める
  - 変換: `convertPaged` が `out.mass.mlpackage` を書き、manifest に `mass`。それ以前の変換は `llm.AddMass` で書き足す (重みがないので数秒。`chat.Prepare` が呼ぶので変換し直しは要らない)
  - 読み込み: `chain.loadMass`、`decode.Attention.Mass`。`OnNPU` は mass のプログラムも見る。CPU の runner も読む
  - チャット: `Config.Dense` (`-sparse=false`) でなく、manifest に mass があれば `DefaultSparse` で有効。メモリの上限 (`fitMemory`) に要約の分 (1/8) を足す。ready の行に `past about 48K tokens, 16384 keys exactly and the rest as means of 8`
- テスト (Linux):
  - `TestSparseMatchesDirect` など (`internal/llm/decode`): 1 層の偽のモデルと Go の注意で、2 段の注意を直接計算したもの (CPU の確認の形) とセッションの出力を比べる。セッションが選んだバケットで計算したものと相対誤差 1e-2 以内、選んだバケットの重みは最も重いものの 99% 以上 (fp16 の丸めで同じくらいの重さのものが入れ替わる)。2 段で読まない呼び出しは全部を読むものと一致。1 トークンと 4 トークンの呼び出し、`Refresh` 4 (選び直しは 4 回に 1 回)、`Mark` に戻って別のトークンを書いたとき、`Reset`、`Save` の行を戻したとき、8 トークンのプリフィルから 1 トークンのデコードに移るとき
  - `TestMassMatchesSoftmax`、`TestMergeSubAndAddMerge` (`internal/llm/paged`、CPU のインタプリタ)
  - `TestPagedSparse` (`internal/llm/lfm3`): 小さな LFM3 を変換し (mass のプログラムも)、2 段の注意を 3296 回呼んで、ロジットは全部を読むものと相対誤差 1.4%。`AddMass` が mass のない manifest に書き足す
- 見込み (16.59 の M3 の実測から。計測なし): 1 トークンは 128K で約 90 ms (今 165)、250K で約 140 ms (今 320)。128K のプロンプトを読む時間は約 7 分 (今 8.6)、250K は約 19 分 (今 29)。プリフィルは約 57K までは今と同じで、要約への 2 回の注意と mass と 4 つの正確な塊の分だけ縮みが小さい。M3 での確認は [handoff-m3-sparse.md](handoff-m3-sparse.md)
- **M3 での確認** (MacBook Air M3 16 GB、macOS 27.0.1、LFM3-1.7B の lut6 + grouped、塊 8192、`-draft 8`、2026-11-06):
  - テスト: 引き継ぎの 6 パッケージは通る (`internal/chat` の `TestPagedOnNeuralEngine` の 2 段の版を含む)
  - 読み込み: 初回の起動で `chat.Prepare` が既存の `lfm3-lut6-grouped-p8192-c262144-q8.parts.json` と `-q64` に mass のプログラムを書き足し、読み込みは 33.2 + 34.2 秒 (mass の 2 本のコンパイル込み)。**2 本とも `all on the NPU`** (`OnNPU` は mass も見る)。ready の行は見込みどおり `... (attention in chunks of 8192; past about 48K tokens, 16384 keys exactly and the rest as means of 8)` で、キャッシュは 1 トークン 12 → 14 KiB (262144 で 3.38 GiB、要約の分)。2 回目からは 1.0〜1.3 + 2.2〜9.8 秒で、プログラムが 2 本増えてもコンパイルのキャッシュに残る
  - 速さ (16.59 と同じ形の試験: サーバー、`-prompt-cache off -think=false`、温度 0、先頭に合言葉を置いた繰り返しの文、大きさごとにサーバーを起動し直す。続けて同じ会話で英語で「浜辺の晴れた日を 100 語ほどで」、最大 128 トークン)。生成の ms/token は下書きの当たりで揺れるので、その 100 語の生成の 1 回の呼び出しの時間 (生成の時間 ÷ 呼び出しの数) で比べる。全部を読む形 (`-sparse=false`) も同じ日に測った:

| プロンプト | 形 | 読む時間 | 生成 | 1 回 | footprint |
|---|---|---|---|---|---|
| 64,011 | 2 段 | 173.4 s | 78.4 ms/token | 84 ms | 1,596 MB |
| 64,011 | 全部 | 155.4 s | 89.3 ms/token | 101 ms | 1,009 MB |
| 127,699 | 2 段 | 468.8 s | 90.2 ms/token | 105 ms | 2,490 MB |
| 127,699 | 全部 | 616.0 s | 208.0 ms/token | 221 ms | 1,781 MB |
| 250,705 | 2 段 | 1217.1 s | 125.7 ms/token | 144 ms | 4,088 MB |
| 250,705 | 全部 | 2185.9 s | 277.2 ms/token | 345 ms | 3,279 MB |

  - 6 つとも合言葉を正しく答えた
  - 分かったこと
    - **生成は見込みどおり**: 2 段の 1 トークンは 128K で 90.2 ms (見込み約 90)、250K で 126 ms (約 140)。1 回の呼び出しは同じ日の全部を読む形の 128K で 0.48 倍、250K で 0.42 倍。64K では 0.83 倍 (塊 8 つのうち 2 段で 5 回)
    - **プロンプトの読みは長いほど縮む**: 128K で 0.76 倍、250K で 0.56 倍 (見込みは約 430 秒と約 1120 秒)。64K では 2 段のほうが 1 割遅い。プリフィルは約 57K から 2 段で読み、64K ではその分が短いので、要約への注意と mass、正確な塊 4 つの呼び出しと、ホストの選択の分が勝つ。境目は上げてよいかもしれないが、試していない
    - **この日の全部を読む形は 16.59 より遅い**: 128K で 1 回 221 対 174 ms、読み 616 対 514 秒、250K で読み 2186 対 1752 秒 (1 回は 345 対 350 で同じ)。64K は 101 対 97 ms で同じ。この日のスワップは 9〜10 GB (16.59 は 5.5〜6.3 GB) 使われていた。比べは同じ日の行どうしで
    - **メモリは要約と正確に読む塊の分だけ増える**: 250K で 3.28 → 4.09 GB。要約は 1 トークン 2 KiB (250K で 0.48 GiB)、残りは層ごとの正確に読む塊 (プリフィルの 4 つと窓の分) の写し
  - ホスト側の時間 (pprof。試験のために hwnpu-chat に `net/http/pprof` を足したビルドで、128K の 2 段): 生成の 10.0 秒 (95 回) の間に Go の CPU は 4.8 秒で、そのうち Core ML の呼び出し (`coreml.(*request).predict` の cgo) が 2.3 秒、**`decode.(*Session).choose` が 1.24 秒 (1 回あたり約 13 ms、1 回の 12%)**。`choose` のほとんどは memmove で、選び直しで入れ替わったバケットの鍵と値を正確に読む塊へ写す分 (`summarize`、`tailMask`、`topK` は 0.1 秒未満)。入れ替わるバケットが多いということで、写しを減らすには `Refresh` (下) か、入れ替わりの少ない選び方。プリフィルの 60 秒の間では `choose` は 2.9 秒 (5%)
  - `Refresh` の精度 (環境変数で `Sparse.Refresh` を変えた試験用のビルド): 128K の文書 (このリポジトリの docs と Go のソース、126K トークン、16.61 の CPU の確認と同じ形) の 10%、50%、90% に合言葉 3 つを置き、日本語で 3 つを尋ねる (温度 0)。文書の並びは docs が先と、Go のソースが先 (逆) の 2 通り。続けて 100 語の生成の 1 回の時間:

| 形 | 3 つの合言葉 (docs が先) | 1 回 | 読む時間 | 3 つの合言葉 (逆) | 1 回 | 読む時間 |
|---|---|---|---|---|---|---|
| 全部を読む | ○○○ | 181 ms | 591 s | ○○○ | 179 ms | 633 s |
| 2 段、`Refresh` 1 (既定) | ○○○ (2 と 3 の番号を取り違え) | 111 ms | 464 s | ○○○ | 105 ms | 513 s |
| 2 段、`Refresh` 4 | ○○○ (2 と 3 の番号を取り違え) | 81 ms | 407 s | ○○○ | 90 ms | 431 s |
| 2 段、`Refresh` 16 | ○○○ | 76 ms | 414 s | ○○○ | 83 ms | 479 s |

  - どの形も 3 つの合言葉を一字一句答えた。逆の文書は答えが長く、最初の 64 トークンでは 3 つ目まで届かなかったので、最大 256 トークンで測り直した (表はその値)。docs が先の文書で `Refresh` 1 と 4 が 2 と 3 の番号を取り違えた (文字列は 3 つとも出る) のは、`Refresh` 16 と全部を読む形にはなく、選び直しを減らしたせいではない
  - 選び直しを減らすと速くなる: 1 回は `Refresh` 1 に対して 4 で 0.73〜0.86 倍、16 で 0.68〜0.79 倍 (mass の呼び出しと上の `choose` の写しが減る)。読みは 0.84〜0.93 倍
  - 既定は毎回 (`Refresh` 1) のまま。この試験で見えるのは合言葉の想起だけで、2 つの文書、温度 0 の 1 回ずつ。損失や KL のような細かい差は Neural Engine では測っていない (CPU の確認で 1 回 256 トークンを 1 つの選択で読むと遠い合言葉を落としたのに対し、ここでの `Refresh` 16 のプリフィルは 1 回 64 トークンで 16 回 = 1024 トークンを同じ選択で読んでも落とさなかった。呼び出しごとに正確な塊の後ろの窓と呼び出し自身の鍵は読み直すため)。4 を既定にする候補だが、損失での確認の後に

### 16.62 Qwen3.5 を移す: Gated DeltaNet と、丸ごと置き換える状態 (`internal/llm/qwen35`) — 2026-11-04

- 動機: model-fit.md 7 節の第一候補。コーディング向けでは、ANE に載る大きさで Qwen3.5-4B が最も強い (LiveCodeBench v6 で 55.8。Gemma 4 E4B は 52.0、E2B は 44.0。どれもモデルカードの自己申告)。9B (65.6) は Qwen3-30B-A3B-Thinking (66.0) と同じくらいで、同じコードで変換できる
- 構造 (modeling_qwen3_5.py、transformers 5.22): 3 層の Gated DeltaNet に 1 層の全注意の繰り返し。4B は 32 層 (GDN 24、全注意 8)、hidden 2560、語彙 248320 (出力層は埋め込みと共有)
  - RMS 正規化は 0 中心 (`(1 + w)` を掛ける)。GDN の最後の正規化 (`Qwen3_5RMSNormGated`) だけは `w` をそのまま掛け、`silu(z)` でゲートする
  - 全注意: `q_proj` が各ヘッドの問い合わせと出力ゲートを並べて出す (ヘッドごとに 256 + 256)。QK 正規化、各ヘッドの先頭 64 次元だけの RoPE (`rotate_half` はその 64 の中で)、GQA (16 ヘッドに kv 4)、出力に `sigmoid(gate)` を要素ごとに掛ける。テキストでは多次元の RoPE (mrope) の 3 つの位置が同じなので、普通の RoPE と同じ
  - GDN: `in_proj_qkv` (8192 = 問い合わせ 2048 + 鍵 2048 + 値 4096) に幅 4 の深さ方向の因果的な畳み込みと silu。問い合わせと鍵は L2 正規化 (問い合わせはさらに 1/√128)、鍵ヘッド 16 を値ヘッド 32 に 2 つずつ使う (`repeat_interleave`)。減衰 `exp(-exp(A_log) · softplus(a + dt_bias))` と書き込みの強さ `sigmoid(b)` で、値ヘッドごとの状態 S (128 × 128) を 1 トークンずつ `S ← S·decay; S ← S + kᵀ((v − kS)·β); o = qS` と更新する
  - チェックポイントは画像とテキストのモデル (`Qwen3_5ForConditionalGeneration`、テンソルは `model.language_model.`)。視覚の塔 (`model.visual.`、297 個) と、次の次のトークンを当てる層 (`mtp.`、15 個。投機的デコードの下書きに使える) は変換しない。テキストだけのチェックポイント (`qwen3_5_text`、`model.`) も読む
- 実装:
  - nn.Module 1 つにメソッド 1 つ (`decoderLayer`、`attention`、`gatedDeltaNet`、`causalConv`、`deltaRule`)。GDN の更新は 1 トークンずつ書き下した形 (Q=64 では 64 回)。L2 正規化は「epsilon を 1e-6/n にした RMS 正規化に 1/√n を掛ける」と同じなので `RMSNormWith` で書く。`softplus` を CPU のインタプリタに足した (PyTorch の値と照合)
  - 0 中心の正規化のため、`blocks.RMSNorm` から重みの値を渡す `RMSNormWith` を切り出した (演算の順序は同じ。LFM3 と Gemma 4 のダイジェストは変わらない)
  - **丸ごと置き換える状態 `st_`** (`llm.Cached` に追加): GDN の状態は KV と違い位置ごとに書けない。プログラムは入力 `st_rec_N` (1, 32, 128, 128) を取り、ステップの後の状態を `st_rec_N_new` に出し、ホストはそれで入力を置き換える。詰め物のトークンは状態を変えてはいけないので、`StepInputs` が `step_valid` (トークンは 1、詰め物は 0) を渡し、詰め物では β と g を 0 にする (状態はそのまま)。畳み込みの状態は LFM3 と同じ `win_conv_N` (直前の 3 行)
  - 投機的デコードで下書きの一部だけを採るときは、ステップの後の状態が全部のトークンの後のものなので使えない。入力のバッファにはステップの前の状態が残っているので、`decode.Session.Speculate` は採ったトークンだけでもう一度走らせる (状態を持つモデルのときだけ)。`Mark` と `Rewind`、`Resume`、`Migrate`、`chain.Share` は `st_` を `win_` と同じに扱う (`llm.CacheInput`、`llm.HeldInput` にまとめた)
  - 分割された safetensors (`model.safetensors.index.json`) を読めるようにした (`llm.OpenWeights`。`model.safetensors` を渡して索引しかないときも索引で開く)。Qwen3.5-4B は 2 ファイル
- 合格条件 (Linux、CPU のインタプリタ):
  - `TestAgainstFP64`: modeling_qwen3_5.py から直接書いた fp64 の forward と相対誤差 3.7e-7 (小さな合成モデル: 2 種類の層、GQA、部分 RoPE、鍵ヘッドの共有)。最初は 4e-4〜1e-3 ずれた。正解側で `1 + w` の w を先に fp16 に丸めていたためで、プログラムは `1 + w` を一度だけ丸める (二重に丸めると 3% ほどの要素が隣の fp16 にずれる)
  - `TestLayerByLayer`: 1 層ずつ (Start、Layers、NoHead、最後の正規化と出力層だけ) が正解の各層と、つないだものが全体と一致
  - `TestCachedMatchesFull`、`TestShareRewindSpeculate`: キャッシュ付きで 1、3、4、10 トークンずつ (最初のステップは詰め物あり) 回したロジットが全体のプログラムと相対誤差 4e-4 以内 (ホストが状態を fp16 で持つ分)。2 つのプログラム (Q=1 と Q=4) が状態を共有し、寄り道の後に `Rewind` で戻り、4 トークンのうち 2 つだけ採る `Speculate` の後も一致
  - 壊して落ちることを確かめた: 鍵ヘッドの繰り返しの軸、詰め物の β と g、`Speculate` のやり直し、減衰と書き込みの順、出力ゲート、RoPE を頭全体に、キャッシュ内の因果マスク、0 中心でない正規化。どれも少なくとも 1 つのテストが落ちる
  - `TestParseConfig`: MoE (num_experts)、silu 以外、attention_bias、出力ゲートなし、未知の層の種類、層の数の食い違い、rope_type が default 以外、mrope_section が回転の対を覆わないもの、値ヘッドが鍵ヘッドの倍数でないものを拒む
- 実物の重みでの照合 (`hwnpu-check`、PyTorch 2.15 と transformers 5.22 の fp32 が参照): この機械の回線が遅く (約 0.4 MB/s) 9.3 GB を落とせなかったので、Hugging Face から範囲読みで層 0 (GDN) と層 3 (全注意) と、照合する 32 トークンの埋め込みの行だけを取り、2 層のテキストだけのチェックポイント (`Qwen3_5ForCausalLM`、ほかの埋め込みの行は 0) にして照合した。`A_log` は F32 で、ほかは BF16
  - `tensors: 0 missing, 0 mismatched, 0 unused`
  - fp16: 層 0 (GDN) の相対 RMS 誤差 0.0008、層 1 (全注意) 0.0001、最後の正規化と出力層 0.0000、全体 0.0011 で argmax 32/32、KL 0.0000
  - fp16 に丸めて (`-f16`): 層 0.0011 と 0.0013、全体 0.0020、argmax 32/32
  - lut6: 層 0.114 と 0.122、全体 0.21 で argmax 32/32、KL 0.016 (2 層だけの値で、32 層の損失はわからない)
  - 32 層全部での照合は未実施。回線のよい機械で `python tools/ref/ref_logits.py ~/models/Qwen3.5-4B build/qwen35/ref --seq 32` のあと `go run ./cmd/hwnpu-check -config ~/models/Qwen3.5-4B/config.json -weights ~/models/Qwen3.5-4B/model.safetensors.index.json -ref build/qwen35/ref` (深い層の大きな活性で fp16 があふれないかも、ここで見る)
- 文脈 262144 (`llm.Paged`、LFM3 の 16.59 と同じ形): 全注意の 8 層の前後でモデルを分け、注意はホストが塊ごとに重みなしのプログラムで読む。分け目の前の部分は `attn_q_N`、`kv_k_N_new`、`kv_v_N_new` に加えて出力ゲート `attn_gate_N` (1, seq, 16, 256) を出し、後の部分がそれを取る (LFM3 のゲートはヘッドごと、Qwen3.5 はチャネルごと)。GDN の状態と畳み込みの窓は各部分の入力のまま。4B は 9 部分 (lut6 で 2.95 GiB、M3 の 1 部分 1 GiB に収まる。最後の部分が出力層を含み 0.52 GiB)。hwnpu-chat は `llm.Paged` のモデルを自動で分け、文脈を `max_position_embeddings` (262144) にする。KV は 1 トークン 32 KiB で、262144 まで使うと 8 GiB (LFM3-1.7B は 3 GiB)
  - `TestPagedMatchesFull`: 3 部分に分けたものが全体のプログラムと相対誤差 6e-4 以内 (詰め物のあるステップ、2 つのプログラムの共有、塊の境目をまたぐ寄り道からの `Rewind`、2 つだけ採る `Speculate`、`Save`、書き出した行からの `Resume`)。詰め物の扱いとゲートの受け渡しを壊すと落ちる
  - `TestEngineReuseQwen35` (internal/chat): エンジンが前のプロンプトを再利用しても、先頭から 4 トークンずつ読んだものと相対誤差 3e-3 以内。LFM3 の 2e-3 より緩めたのは、比べる相手 (4 トークンずつ、状態を fp16 で持つ) 自体が全体のプログラムから最大 1.7e-3 ずれるため (エンジンの側は 5.9e-4)。`TestPrepareQwen35`: 2 ファイルに分かれたチェックポイントを索引から取り、分けた形 (3 部分と注意のプログラム) に変換する
- トークナイザー: Qwen3.5 の tokenizer.json は NFC の正規化と、GPT-4 のものと少し違う分割の規則 (`[\p{L}\p{M}]+` で結合文字を文字に含め、数字は 1 つずつ。`\p{M}` は記号の側から外す) を使う。`golang.org/x/text/unicode/norm` を依存に足し (このリポジトリの 3 つ目)、`internal/tokenizer` が両方の規則を読む。HF の tokenizers 0.23 と、日本語、ヒンディー語、結合文字、全角、特殊トークンを含む 2 文で ID が一致 (`TestQwen35`、`HWNPU_QWEN35_DIR`)。分割は 15 例で HF の分割と一致 (`TestSplitQwen35`)
- チャット: `chat.Qwen35` が chat_template.jinja を Go に移したもの。道具の宣言は Qwen 自身の説明つきで system に置き、思考は最後の質問より後の回答だけに残す。道具の引数は、オブジェクトと配列は JSON、ほかは Python の `str()` (True、None)。`tools/ref/template_golden.py` を Qwen3.5-4B で描かせた 9 会話と一字一句一致 (`TestQwen35RenderMatchesTemplate`)。回答の分解は LFM3 と同じ XML なので `chat.Parse` を使い、思考の前後の改行を落とす。目録に `Qwen3.5-4B` を足し、hwnpu-chat は分割された safetensors (索引と各ファイル) を落とす (`fetchWeights`)
- 残り:
  - **Neural Engine での配置と精度は未確認** (この機械は Linux)。気になる点: GDN の状態を fp16 で持ち続けること (PyTorch は fp32。長い会話で誤差が積もるか)、Q=64 のプログラムで更新を 64 回書き下すこと (GDN 24 層で約 1.8 万演算。コンパイルの時間と ANE への載り方)。後者は塊の形 (FLA の WY 表現。塊の中の下三角の逆行列は `(I − A)(I + A²)(I + A⁴)…` の 6 回の行列積で書ける) を `Params` で選べるようにするのが次の候補
  - M3 16 GB では、2 つのプログラム (Q=8 と Q=64) の重み 5.9 GiB が ANE のキャッシュ (16.39 で 4 GiB 前後とみた) に収まらず、起動のたびにコンパイルし直すおそれがある。そのときは lut4 (2 本で 3.9 GiB) か、Q=8 だけにする
  - 分けない形で 16383 を超えるキャッシュ (`llm.CacheChunks` の塊) は未実装 (分けた形を使うので要らない)。2 段の注意 (16.61) が Qwen3.5 でも保つかは未確認
  - Qwen3.5 の層 `mtp.` を投機的デコードの下書きに使うこと (model-fit.md T9)

### 16.63 Gemma 4 と LFM3 の VL も注意の前後で分け、Intel の NPU でも分ける — 2026-11-04

- 動機: 16.59 で LFM3-1.7B の文脈 262144 が既定になった。ほかのチャットのモデル (LFM3-VL、Audio、Omni、Gemma 4 E2B と E4B) も、少ないメモリでより長い文脈を持てるようにする。LFM3 の VL、Audio、Omni はテキスト部が `lfm3` なので 16.59 のままで分かれる (`chat.Prepare` が `llm.Paged` なら分ける)。残りは Gemma 4 と、Neural Engine に限っていた分割を Intel の NPU に広げること
- Gemma 4 の分け方 (`internal/llm/gemma4`): 全注意の層 (E2B は 4、9、14、19、24、29、34) の前後で切る。E2B は 8 部分。全注意の層 N の前で終わる部分は、問い合わせ attn_q_N (norm と比例 RoPE の後) と、自分の鍵と値を持つ層なら kv_k_N_new と kv_v_N_new (k_norm と RoPE、v は重みなしの norm の後) を出す。次の部分は attn_o_N から o_proj、post_attention_layernorm、MLP、層ごとの入力、layer_scalar を計算する。スライディングの層はリングを部分の中に持ち、共有の鍵と値 (shared_k_sliding、層 13 の分) と per_layer_inputs は前の部分の出力を名前で結ぶ (`internal/llm/chain` の既存の規則。後ろの部分のどれもが同じ出力を読む)
- 鍵と値の共有 (`llm.Paged.PagedKV`): 層 19、24、29、34 は層 14 の鍵と値を読む (num_kv_shared_layers 20)。分けた形ではこれらの層は attn_q_N だけを出し、`decode.Session` は層 14 のキャッシュに対して注意を呼ぶ (そのステップの行は層 14 のときに書いてあるので、書き直さない)。ホストが持つキャッシュは 3 層分で 1 トークン 2 × 3 × 1 ヘッド × 512 × 2 バイト = 6 KiB (131072 で 0.75 GiB)。E4B は自分のを持つ全注意の層が 4 つ、KV ヘッド 2 で 1 トークン 16 KiB (131072 で 2 GiB)。LFM3 は自分の層を返す
- 注意の倍率 (`paged.Shape.Scale`): Gemma 4 は点数を 1/√d で割らない (q_norm と k_norm がその役)。注意のプログラムに倍率を持たせ、0 なら今までどおり 1/√d。`llm.Paged.PagedShape` は `paged.Shape` を返すようにした
- 文脈: `max_position_embeddings` を読み、`MaxContext` に返す (E2B と E4B は 131072)。全注意のマスクは `PagedStepInputs` から外し、回転の表だけを残す
- Intel の NPU (`llm.convertPaged`): `-backend intel` で部分を OpenVINO IR (`out.partN.xml`) に、注意のプログラムも MIL で組んで `ovlower` で IR (`out.attn.xml`) に下ろす (`llm.writeIR` を部分と共用)。Level Zero と OpenVINO の `Bind` は終わった要求の付け替えを許すので、実行側は変えていない。`chat.pagedChunk` は NPU を問わず分けるようにした (Intel の既定は `lfm3-int8-grouped-p8192-c262144-q8.parts.json`)
- 検証 (CPU のインタプリタ):
  - `gemma4` の `TestPagedMatchesFull` (小さな Gemma 4: スライディング 4 層、全注意 2 層のうち 1 層が層 2 の鍵と値を共有、層ごとの埋め込み): 1、3、4 トークンのステップ、塊の境をまたぐステップ、塊の倍数でない容量で、キャッシュなしのプログラムとロジットの相対誤差 2.4e-3〜3.2e-3 (分けない KV キャッシュ付きの `TestCachedMatchesFull` と同じ 2.6e-3〜3.2e-3)。持つキャッシュは層 2 の分だけ。`TestPagedShareRewind`: 1 トークンと 4 トークンのプログラムが 1 つの系列を続け、印と `Save` で戻っても同じ
  - 改変の検出: 倍率を 1/√d に戻すと相対誤差 0.45 で落ちる。`paged` の `TestMergeMatchesFull` に倍率 1、KV ヘッド 1 の形を足した
  - Intel (OpenVINO 2026.4 の CPU、`HWNPU_OV_PYTHON`。Mac でも pip の openvino で動く): `gemma4` の `TestIntelIRPaged` (4 トークンと 1 トークンの IR が `chain` で 1 つのキャッシュを共有) で 2.8e-3、`lfm3` の `TestIntelIRPaged` (ゲートを部分の間で渡す) で 5.7e-4。`chat` の `TestPreparePaged` は Intel でも分けた IR を書く
- M3 (16 GB、macOS 27.0.1) の Gemma 4 E2B (lut6、`-draft 8`):
  - 変換: 部分は 0.12〜0.33 GiB で 1 本 1.61 GiB、Q=8 と Q=64 で計約 3 分。初回のコンパイルは 92 秒と 96 秒。**2 回目からの読み込みは 1.8〜1.9 秒**。分けない 2 部分の形 (`gemma4-lut6-c16320`) は同じ日に 49 秒と 50 秒で、起動のたびにコンパイルし直していた (16.42、16.43 の「lut6 の部分がコンパイルのキャッシュに残らない」)。部分を小さくすると残る
  - 配置: 部分は最後の 1 つの平らなロジットの reshape (16.43、`-logits folded` で外せる) を除いて全部 Neural Engine。注意のプログラム (head_dim 512、KV ヘッド 1、Q=8 と Q=64) は 17 演算とも Neural Engine
  - 答え (温度 0、思考なし、4 問): 富士山、17 × 23、川、『こころ』の答えは分けない形と 1 トークンも違わなかった (富士山の 2,360 m の誤りもそのまま。モデルの側)。生成は 23.6、36.0、26.3、26.2 tokens/s で、分けない形 (容量 16320) の 13.3、28.2、20.6、20.9 より速い
  - 長い文脈 (16.22 の試験、サーバー、`-prompt-cache off`): 81,691 トークンのプロンプトを 299.1 秒 (273 tokens/s) で読み、先頭の合言葉 BLUE-OTTER-42 を正しく答えた。その長さでの生成は 96.7 ms/token (171 トークン)。読み終えた後のプロセスの footprint は 1118 MB (キャッシュは 11 塊 × 8192 × 6 KiB = 528 MiB)
- M3 の LFM3-VL-2B (lut8): 7 部分ずつ、全部 Neural Engine。文脈 128000、キャッシュは 1 トークン 12 KiB で最大 1.46 GiB。初回のコンパイル 32 秒と 54 秒、2 回目は 1.4 秒と 4.2 秒。画像 (青地に円と右上の黄色い正方形) を正しく説明し (13.9 tokens/s、279 トークンのプロンプトを 0.6 秒)、17 × 23 = 391 (33.7 tokens/s)
- 確かめていないこと: LFM3-Audio と Omni (config.json だけ見た。全注意の層は VL と同じ 2、6、…、22、文脈 128000) と Gemma 4 E4B (config.json だけ。全注意 5、11、…、41 のうち 29、35、41 が層 23 を共有) の実機。Intel の NPU の実機 (ドライバのコンパイラが注意のグラフを受け付けるか、付け替えの費用)
- AMD の NPU は対象外: IRON の書き出しが `--max-seq-len` (既定 4096) で文脈を固定し、KV キャッシュはデバイスの bank の中にある。長くすると Gemma 4 のスライディングの層がリングになり (`3 × リング < max_seq_len`、16.54)、エンジンが読み込まない形になる。塊ごとの注意は IRON の側の仕事になる

### 16.64 同じ重みをディスクの上で共有し、コンパイルのキャッシュを掃除する — 2026-11-04

- 動機: M3 のディスクの空きが 12 GB まで減った。`~/Library/Caches/hwnpu` は 66 GB で、うちモデルが 41 GB、コンパイルのキャッシュ (`.mlmodelc`) が 24.6 GB
- 圧縮は効かない (実測): bf16 のチェックポイントは lz4 で 100%、zstd -3 で 77%、-19 で 76%、上下のバイトを分けて zstd -3 で 71%。lut6 のプログラムの重みは lz4 で 99%、zstd で 87〜90% (表の添字はほぼ一様)。しかも実行中に写像するもの (プログラムの重み、埋め込みを引くチェックポイント) は展開した写しが要るので減らない
- 無駄は重複だった: Q=8 と Q=64 のプログラムの `weight.bin` はバイト単位で同じで、Core ML のコンパイル結果 (`.mlmodelc/weights/weight.bin`) もパッケージのものと同じバイトの別のファイル。Gemma 4 E2B なら 1.6 GiB の重みが 4 つあった
- 共有 (`cache.Clone`): 同じバイトのファイルを APFS の clonefile(2) の写しに置き換える。ブロックを共有するだけで普通のファイルのまま (写像も Core ML の読み込みも変わらない。片方に書けばその分だけ別になる)。macOS 以外では何もしない (16.79 で Linux の FICLONE と Windows の ReFS も)。使うところは 3 つ: 変換 (`llm.convertPart` が書いたプログラムの重みを、同じディレクトリの同じバイトのものの写しにする)、コンパイル (`coreml` がキャッシュに置いた `.mlmodelc` の重みをパッケージの写しにする)、それ以前のファイル (`chat.Prepare` がモデルのディレクトリごとに 1 回 `llm.ShareWeights`、印は `.weights-shared`)
- 掃除 (`cache.Prune`): `Lookup` が項目の時刻を今にし、`Store` の後に元のパッケージを `<key>.src` に記録する (以前の項目は次に使われたときに記録する)。チャットのエンジンは読み込みの後で、元が消えた項目と 30 日使われていない項目と、途中で止まった一時ディレクトリ (1 日) を消す
- 結果 (M3、LFM3-1.7B、LFM3-VL-2B、Gemma 4 E2B を 1 回ずつ起動した後。APFS のファイルごとの共有されていない大きさ `ATTR_CMNEXT_PRIVATESIZE` で測った): モデルのディレクトリの重みのファイル 24.6 GiB は全部が他のファイルと共有された (共有されていない分 0)。コンパイルのキャッシュ 25.0 GiB のうち共有されていないのは 8.2 GiB で、今回読み込まなかったプログラム (分けない以前の形、fp16 の試験、プローブ) の分。重みのファイルが実際に使うディスクは約 49.6 GiB から約 20 GiB になった見込み。`df` の空きはスワップ (8〜9 GB) とシステム側のキャッシュで数十 GB 揺れるので、この比べには使えなかった。clonefile が空きを返すことは 2 GiB のファイルで確かめた (+2.00 GiB)
- 副作用: 共有に置き換えた後の最初の起動では、プログラム 1 本ごとに 20〜50 秒かかった (Neural Engine のシステム側のキャッシュが、ファイルが置き換わったのを見てコンパイルし直す)。2 回目からは元どおり 2 秒前後。新しく変換するものは最初から共有なので、これは今あるファイルの 1 回だけ
- 残り: チェックポイントは実行時も埋め込みの表のために開いている (model-fit.md T5)。表だけを変換時に書き出せば LFM3 は 3.5 GB → 0.27 GB、Gemma 4 は 10.2 GB → 約 5.5 GB (層ごとの埋め込みを int8 にすれば約 3 GB) になる。再変換にはダウンロードし直しが要る

### 16.65 分けた形の残りのモデルの確認と、長い文脈の計測 (M3) — 2026-11-04

16.63 で動かしていなかったモデル (LFM3-Audio-2B、LFM3-Omni-2B、Gemma 4 E4B) を M3 (16 GB、macOS 27.0.1) で動かし、LFM3-VL-2B と Gemma 4 E2B は分けない形との比べ合わせと長い文脈を測った。どれも温度 0。長い文脈は 16.22 の合言葉の試験で、サーバー (`-prompt-cache off`)、要求ごとに合言葉を変え (前の要求の先頭を使い回さない)、プロンプトの長さはモデルのトークナイザーで合わせた。メモリはサーバーのプロセスの footprint で、読み終えた後に測った (同じサーバーの前の会話の写し、最大 2 本を含む)

- LFM3-VL-2B (lut8): 画像 (青地に円と右上の黄色い正方形)、17 × 23、富士山の 3 問の答えが分けない形 (文脈 16320) と 1 トークンも違わない。生成は 24.0、33.2、36.0 tokens/s で、分けない形は 11.9、26.7、29.4
- LFM3-Audio-2B (lut8): 初回の変換と音声エンコーダの変換の後、読み込み 47 秒と 57 秒 (初回)、全部 Neural Engine。`say` で作った英語と日本語の質問の音声 (16 kHz) と、17 × 23、富士山の 4 問の答えは分けない形と同じで (日本語の音声への答えは終わりの数語が違った)、生成は 18.8、23.1、28.6、28.1 tokens/s (分けない形は 16.3、18.2、22.9、23.4)。音声への答え自体はどちらも的外れ (英語の「フランスの首都」を "Capital Steel of France" と聞き、日本語の質問には聞き返す)。書き起こしを頼むと英語は "Please tell me the Capital Steel of France."、日本語は「日本で一番高い山」と内容は取れていたので、聞き取りの経路は動いていて、答え方はモデルの側
- LFM3-Omni-2B (lut8): 画像の説明 (赤い円と黄色の正方形、「3 つの図形」と言う誤りあり)、英語の音声の書き起こし ("Please tell me the Capital Stew France.")、391、富士山 3,776.2 m。分けない形との比べは省いた (テキスト部の分け方は VL と Audio と同じ)
- Gemma 4 E4B (lut6): 部分は 0.36〜0.53 GiB、1 本 3.26 GiB で 8 部分。初回のコンパイルは 183 秒と 169 秒。答えは正しい (富士山 3,776 m、391、信濃川、『こころ』の語り手「私」と先生)。ただし遅い: **2 回目からの起動も 77 秒と 87 秒** (2 本で 6.5 GiB は Neural Engine のコンパイルのキャッシュ、約 4 GiB に収まらない。16.41)。14〜17 トークンのプロンプトの読み込みに 1〜3 秒、生成は 10〜13 tokens/s、1 つ目の答えは 1.2〜1.4 tokens/s。Q=64 と Q=8 のプログラムを切り替えるたびに重みを載せ直しているとみる (16.36)。分けない形との比べはディスクが足りず省いた (分ける前から 2 本で 6.5 GiB なので、この遅さは分けたことによるものではない見込み)
- 長い文脈 (どれも合言葉を正しく答えた。VL の 16K と 64K は思考の中で、ほかは思考なしの答えで):

| モデル | プロンプト | 読み込み | その長さの生成 | メモリ |
|---|---|---|---|---|
| LFM3-VL-2B | 15,998 | 20.1 秒 | 25.2 ms/token | 434 MB |
| LFM3-VL-2B | 64,011 | 171.0 秒 | 58.6 ms/token | 1,395 MB |
| LFM3-VL-2B | 120,005 | 496.9 秒 | 68.1 ms/token | 3,615 MB (前の 2 つの会話の写しを含む) |
| LFM3-Audio-2B | 120,005 | 525.2 秒 | 116.6 ms/token | 1,731 MB |
| LFM3-Omni-2B | 120,005 | 511.8 秒 | 89.2 ms/token | 1,731 MB |
| Gemma 4 E2B | 16,003 | 26.8 秒 | 23.9 ms/token | 708 MB |
| Gemma 4 E2B | 64,009 | 223.6 秒 | 82.4 ms/token | 1,216 MB |
| Gemma 4 E2B | 125,011 | 709.2 秒 | 119.5 ms/token | 2,395 MB |
| Gemma 4 E4B | 16,003 | 41.6 秒 | 45.8 ms/token | 976 MB |
| Gemma 4 E4B | 64,009 | 268.2 秒 | 435.2 ms/token | 2,335 MB |
| Gemma 4 E4B | 125,011 | 844.9 秒 | 752.2 ms/token | 5,487 MB (前の 2 つの会話の写しを含む) |

  生成の時間は答えが 18〜19 トークン (思考なし) と短く、投機的デコードの当たり方で揺れる (同じ 120K の LFM3 の 3 つで 68〜117 ms)。読み込みは 120K で 230〜250 tokens/s (LFM3)、125K で 176 tokens/s (Gemma 4 E2B)
- E4B の長い文脈の生成の時間は答えが 13 トークン (5 回の呼び出し) なので、Q=64 から Q=8 への切り替えで重みを載せ直す数秒が大半を占める。E4B の 125K ではスワップが 14.2 GB に達し、16 GB の機械では限界に近い
- 2 段の注意 (16.61、`-sparse`、既定) を通した Gemma 4 E2B (2026-11-05、main に取り込んだ後): 前に変換したプログラムには `chat.Prepare` が倍率 1 の mass のプログラムを足した (`llm.AddMass`)。64,009 トークンは 203.9 秒で読み 73.9 ms/token (全部を読む形は 223.6 秒、82.4)、125,011 トークンは 521.3 秒、92.4 ms/token (709.2 秒、119.5)。どちらも合言葉を正しく答えた。footprint は 1,692 MB と 2,955 MB (要約の分、1 トークン 6 → 7 KiB)。CPU では `gemma4` の `TestPagedSparse` (層 2 の鍵と値を共有する層を含む小さな Gemma 4) が全部を読む形と相対誤差 0.068。ノルムの重みを 1 前後にした (点数が 30 に届く、注意がほぼ 1 つの鍵に集まる) 重みでは 1.3〜1.6 で、共有の有無によらない。要約は尖った注意を近似しにくい
- 上の LFM3-VL、Audio、Omni と Gemma 4 E4B の長い文脈は、2 段の注意を取り込む前の形 (全部を読む) で測った
- 測定中、M3 のスワップは 8〜9 GB (E4B の 125K の後は 14.2 GB) 使われていた。測定の間、別の端末の hwnpu-chat (`go run`) が待ったまま残っていた (RSS 3 MB)
- E4B を速くする 2 つの案を測った (2026-11-05):
  - lut4 (`hwnpu-formats -use gemma-4-E4B-it -ref lut8 -formats lut6,lut4`、容量 4096、1 回 8 トークン、日本語と英語のコードの会話 291 位置): lut8 (4.37 GiB) を基準に、lut6 (3.26 GiB) は次のトークンの一致 263/291 (90%)、ロジットの相対誤差 0.107 (最悪 0.535)、パープレキシティ 7.92 (lut8 は 8.44)、1 回 84.6 ms。lut4 (2.17 GiB) は 186/291 (64%)、0.412 (最悪 2.675)、パープレキシティ 12.16、1 回 67.4 ms。E2B (16.26) と同じく 4 ビットは崩れが大きく、速さは 2 割しか変わらない。2 本で 4.3 GiB になり、コンパイルのキャッシュ (約 4 GiB) に収まるかも際どい。採らない
  - 1 本 (Q=8 のプログラムだけで生成もプロンプトも読む。`-package` に q8 の parts.json だけを渡す): 2 回目からの起動は 3.7 秒 (2 本では 77〜87 秒)、答えは 2 本のときと同じで、生成は 13.4〜16.4 tokens/s (2 本では 10〜13)、短いプロンプトの読み込みは 0.2〜0.3 秒 (1〜3 秒)。代わりに長いプロンプトは遅い: 16K の読み込みが 182.6 秒 (2 本では 41.6 秒、4.4 倍)
- 1 本を既定に (`chat.Prepare`、`-prefill auto`): Neural Engine 向けで、生成用と読み込み用の 2 本の重みの見積もり (`llm.EstimateBytes`。LFM3-1.7B 1.21、VL・Audio・Omni 1.63、E2B 1.60、E4B 3.26 GiB で実物と一致) の合計が約 4 GiB (`compileCacheBytes`) を超えるなら、生成用 (`-draft`、既定 8) だけを変換して読み込み、プロンプトもそれで読む。今のモデルでは E4B だけ (2 本で 6.51 GiB)。`-prefill on` で常に 2 本、`off` で常に 1 本。`-draft` が 1 以下 (1 回 1 トークン) と Intel の NPU では 2 本のまま
- 確認 (M3): E4B は「one program …」と出して q8 の 1 本だけを読み込み、起動は 5.5 秒と 4.1 秒 (format の比べで追い出された直後の 1 回は 109 秒)。答えは 2 本のときと同じ。E2B は 2 本のまま。このときの生成の速さ (E4B 4.9〜13.0、E2B 10.1〜18.2 tokens/s) は、直前の長い試験でスワップが 9.8 GB (10.2 GB 中) 残っていて揺れたので使えない。きれいな状態の値は上の 1 本の試験 (13.4〜16.4 tokens/s)
- 2 段の注意を通した LFM3-VL、Audio、Omni と Gemma 4 E4B の長い文脈 (2026-11-06。試験は 16.61 の「M3 での確認」と同じで、合言葉の後に同じ会話で 100 語を頼み、その 1 回の呼び出しの時間で比べる。この日のスワップは 9〜12 GB): 初回の起動で `chat.Prepare` が mass のプログラムを書き足し (LFM3 系の 3 つは 16〜26 秒、E4B は 86 秒で読み込み)、mass と注意のプログラムはどれも Neural Engine。どれも合言葉を正しく答えた:

| モデル | プロンプト | 形 | 読む時間 | 生成 | 1 回 | footprint |
|---|---|---|---|---|---|---|
| LFM3-VL-2B | 64,011 | 2 段 | 166.0 s | 86.3 ms/token | 93 ms | 1,654 MB |
| LFM3-VL-2B | 119,985 | 2 段 | 430.5 s | 101.0 ms/token | 111 ms | 2,371 MB |
| LFM3-VL-2B | 119,985 | 全部 | 583.0 s | 159.5 ms/token | 179 ms | 1,691 MB |
| LFM3-Audio-2B | 119,985 | 2 段 | 515.8 s | 115.0 ms/token | 120 ms | 2,409 MB |
| LFM3-Omni-2B | 119,985 | 2 段 | 500.8 s | 115.9 ms/token | 118 ms | 2,409 MB |
| Gemma 4 E4B (1 本) | 64,008 | 2 段 | 1244.1 s | 215.9 ms/token | 217 ms | 2,315 MB |
| Gemma 4 E4B (1 本) | 125,010 | 2 段 | 3340.9 s | 273.8 ms/token | 276 ms | 3,472 MB |

  - VL の 120K は同じ日の全部を読む形に対して 1 回 0.62 倍、読み 0.74 倍。この日の全部を読む形の読みは 4 日の 496.9 秒より 17% 遅く、LFM3-1.7B (16.61) と同じく機械の状態の差。Audio と Omni の読みは 4 日の全部を読む形 (525、512 秒) とほぼ同じに見えるが、同じ日の VL の全部を読む形 (583 秒) と比べれば 0.86〜0.88 倍。エンコーダを読み込んでいる分、VL より 1 回が 1 割遅い
  - E4B は今の既定の 1 本 (`-prefill auto`、1 回 8 トークンでプロンプトも読む)。生成は 4 日の 2 本のとき (64K 435.2、125K 752.2 ms/token。答えが 13 トークンで、Q=64 から Q=8 への切り替えの載せ直しが大半) より速いが、読みは 64K で 4.6 倍、125K で 4.0 倍 (268.2、844.9 秒) かかる。125K の読みは 56 分。125K の後はスワップが 11.75 GB
  - E4B の `NOT all on the NPU` は 2 段の注意によらない (4 日の 2 本でも、E2B でも同じ表示)。`hwnpuinfo` で見ると、mass (10 演算) と注意 (17 演算) と部分 1 (559 演算) は全部 Neural Engine で、最後の部分 (層 41 の注意の後と出力層) の `reshape` 1 つ (53 演算のうち) が CPU。PLaMo 3 (16.69) の transpose と同じ所なので、`attn_ot_N` で受ける形が Gemma 4 にも効くかもしれない (未確認)

### 16.66 AMD: LFM3-1.7B が hwnpu-chat と hwnpu-agent で動く (段階 A5) — 2026-11-04

- IRON には LFM3 のアプリケーションがないので、hwnpu に置いた (`tools/amd/iron_lfm3.py`、`iron_export_lm.py lfm3`)。IRON の部品だけで書けた
  - 畳み込みの層: `in_proj` の行を B、C、x の順に分け (`Linear.split`)、B × x を `CausalConv1d` (幅 7、Granite 4.0-H と同じく状態をプロンプトの塊の境ごとに `Checkpoints` に残す) に通し、C を掛けて `out_proj`
  - 全注意の層: q と k をヘッドごとに RMSNorm、RoPE は周波数の後ろ半分を 0 にする (「zero-tail」。Gemma 4 の全体の層の proportional と同じ書き方)。出力のヘッドごとの sigmoid のゲートは、`gate_proj` の各行をヘッドの列数だけ繰り返した重みにして、要素ごとの積にした
  - スライディング窓の層: 窓 256、RoPE は全次元で θ = 10000
  - 層は sandwich norm (`SandwichLayer`)、MLP は SwiGLU (`w1`、`w3`、`w2` を名前で付け替え)、最後の norm は `embedding_norm`、頭は埋め込み
- 照合: IRON のグラフをホストで (`OnHost`) 動かし、Hugging Face (fp32) のロジットと比べた。22 トークンのプロンプトで KL 9e-7、コサイン 0.99995。塊 64 で塊の境をまたぐプロンプトと続くデコードのステップも KL 0.001〜0.0025。NPU では、87 トークンの文の 70 トークンを読んだあと 16 ステップすべてでコサイン 0.999 以上 (`ironlm` に Hugging Face のロジットを `check/` として渡した)
- **状態の巻き戻し**: IRON の `Checkpoints` は塊の境ごとの区画 (slot) を持ち、区画 k は位置 k × 塊までに走った後の状態。呼び出しは始まりの位置の区画を読み、終わりの位置の区画に書き、始まった塊の中で終わる。そこで:
  - エクスポートが manifest に状態のバッファの一覧 (`checkpoints`、名前は `*.slots`) とリングの数 (`rings`) を書く
  - `iron.Model.StateAt(pos)` / `SetStateAt` が、位置 pos から始まる呼び出しが読む区画だけを写す (13 層で 312 KiB)。pos より後の呼び出しが書き換えるのはその区画とそれより後だけなので、行 (KV キャッシュ) とこの写しで pos の系列が全部戻る
  - `Prompt` は状態のあるモデルで呼び出しを塊の境で切る
  - `ironSession` の印 (`Mark`) がその写しを持ち、`Rewind`、`Save`、prompt store (`Resume`) がそれを書き戻す。Core ML の印がリングと畳み込みの窓を持つのと同じ形。リングのあるものはまだ読み込まない (`iron.Model.Resumable`)
  - `TestStateRewindOnDevice` (`HWNPU_IRON_RECURRENT`): 塊の境の 1 つ手前、境の上、境の後ろの 3 か所に戻り、24 ステップが最初の通りとコサイン 0.9999 以上で一致。印からのプロンプトが境をまたいでも同じ。区画を書き戻さないと落ちる
- 文脈: スライディング窓の層は `3 × リング < max_seq_len` でリングになる (リングは窓 + 塊 − 1 以上の 2 のべき)。塊 2048 ならリングは 4096 で、12288 までならない。12288、塊 2048 で書き出した (bank 3.5 GiB + 0.2 GiB)
- 速さ (Ryzen AI 9 365、bf16): デコード 8.7 トークン/秒 (位置 2048 で 118 ms、12287 で 131 ms)。プロンプトの 2048 トークンの塊は、位置 0 から 2.4 秒、2048 から 4.6、4096 から 6.9、6144 から 8.7、8192 から 10.4、10240 から 12.7 秒 (`ironlm -depth`)。1 コマンドの 2 秒の打ち切りには当たらなかった。塊の時間は始まりの位置に比例して増える: IRON の注意はスライディング窓の層でも、リングにしないときは文脈全体をマスク付きで読むため (Gemma 4 も同じ)。窓の外を飛ばせばプロンプトは数倍速くなる
- 実機:
  - hwnpu-chat: 「日本の首都は」に東京、17 × 23 に 391 (8.7 トークン/秒)。2 問目は前の 49 トークンを再利用
  - サーバー: README の先頭 5000 文字を system にした 2331 トークンの要求は、再起動のあと prompt store から 2331 トークンを戻して答えが同じ。同じ system の別の質問は store の 2048 の印から戻して残りを読み、store なしで全部読んだ答えと同じ
  - hwnpu-agent: 円の面積の誤り (π·R) で `go test` が落ちる小さなパッケージで、ls、3 つのファイルを読み、`go test` を走らせて原因を述べ、直し、もう一度 `go test` で通ることを確かめた (1604 トークン)。`Mul` の誤りを直す課題では、直したあとに存在しないディレクトリへの `cd` を何度か試み (エンジンが止めた)、`go vet` が通ったと答えた (実際には走っていない。モデルの癖)
- ついでに直したもの: 別のプロセスが NPU を放した直後の起動で `amd: map device heap: interrupted system call` になった。デバイスのヒープの mmap をシグナルで割り込まれたとき、ioctl と同じくやり直す

### 16.67 AMD: LFM3-VL、LFM3-Audio、LFM3-Omni の塔 (IRON の Multimodal) — 2026-11-05

- IRON の `Multimodal` (Gemma 4 と同じ形: 塔が soft token を状態 `merged` に書き、placed のプロンプトの塊が目印の位置に差し込む) に、16.66 の LFM3 を本体として、hwnpu に置いた塔を載せた。どれも IRON の部品だけで書いた
- **LFM3-VL の視覚の塔** (`tools/amd/iron_lfm3vl.py`): 画像は 1 つの大きさ (512 × 512、Core ML の既定と同じ。ホストで引き伸ばす) で、入力は処理系で正規化したパッチ (raster 順、`lfm3vl.Processor.Patches`)
  - ゲート付きの 3 × 3 の depthwise の畳み込み: 格子を上下に 0 の行の帯を付けた状態に置き、9 通りにずらして読み、それぞれのタップを掛けて足す。横の折り返しは最初と最後の列を 0 にする 2 枚の重み (全層で共有) で消す。IRON の `Conv2d` は depthwise だと (k × 幅 × チャンネル) の帯行列を作るので 1024 チャンネルには使えない
  - 軸ごとの 2 次元の RoPE は表を重みに持つ。2 × 2 の塊は最後に 4 次元の DMA の並べ方で 1 回で読み出す (Hugging Face は塊の順のパッチを受け取るので、射影の前は reshape で済む)
- **LFM3-Audio の音声エンコーダ** (`tools/amd/iron_lfm3audio.py`): 入力はホストで計算したログメル (`lfm3audio.Features`)。最長 4096 フレーム (41 秒、512 トークン。短い畳み込みの塊は 2 のべきが要る)
  - 下ごしらえ: 最初の stride 2 の畳み込みは `Conv2d` (入力 1 チャンネル)。bias は 1 チャンネル分の行で足す (66 画素 × 256 の行を全行に足す放送は DMA の記述 16 個を使い切った)。境界の画素に入った bias は 0 で書き直す。残り 2 つの depthwise は 9 通りの strided な読み出し。各段の有効な行の直後の 1 行を 0 にする (チェックポイントがクリップの外を 0 にするのと同じ結果。次の段の最後の有効な出力はそこまでしか読まない)。`lin_out` の列は (メル, チャンネル) の並びに並べ替える
  - LFM2 の 17 層: 短い畳み込みは `CausalConv1d` (作業領域は `CausalLM` と同じく共有)。注意はヘッド 96 次元で、IRON の MHA は 64 の倍数しか取らないので、0 で 128 に埋める (内積と出力は変わらない。尺度は 1/√96 を渡し、RoPE は実際の 96 列だけ。ヘッドごとの RMSNorm は 0 が平均に入る分、重みと eps を 0.75 倍)
  - クリップの長さで決まる行数の語は、Gemma 4 の音声の塔と同じくフレーム数ごとの表にする
  - 音声の出力 (depth decoder とコーデック) は作っていない。音声を入れて文字で答える
- **LFM3-Omni** (`tools/amd/iron_lfm3omni.py`): 上の 2 つの塔を、Hugging Face の内部の名前 (`model.vision_tower`、`model.audio_encoder`) のまま読む。画像も音声も `<|vision_start|>` と `<|vision_end|>` の間
- 書き出し (`iron_export_lm.py lfm3vl / lfm3audio / lfm3omni`) は manifest の多モーダルの種類 (`kind`) を書き、hwnpu (`iron.Model.RunVision`、`RunAudio`) はそれで前処理を選ぶ。LFM3-Audio の音声の目印はない (エンジンは目印の番号が負なら囲まない)
- 照合 (乱数の重みで、視覚の塔は本物の形、テキストは 4 層に縮めたチェックポイントを Hugging Face の実装から作った):

| 塔 | ホスト (IRON の参照計算) | Ryzen AI 9 365 |
|---|---|---|
| LFM3-VL の視覚 | コサイン 0.9999 | 0.99956 (画像の後のロジット 0.99876) |
| LFM3-Audio の音声 | 0.9977 | 0.985 (下ごしらえの後は 0.99996、17 層で少しずつ下がる) |
| LFM3-Omni の視覚 / 音声 | 0.9999 / 0.9990 | 0.99959 / 0.99031 |

- 実機 (本物の重み、文脈 12288、塊 2048):
  - LFM3-VL-2B: 赤い円と青い四角の 640 × 480 の画像に「赤い楕円と青い四角形です。」(289 トークン、0.9 秒)、続けて「赤い図形は左右どちら」に「画像の左側です。」(302 トークンを再利用)。デコード 8.6 トークン/秒。hwnpu-agent は同じ課題 (16.66) でファイルを読み直すのを繰り返し、テストのほうを直す誤った提案で終わった (チャットの型は LFM3 と同じ)
  - LFM3-Audio-2B: `Front_Left.wav` を「Front, left.」、speech-dispatcher の案内の音声 (約 10 秒) を一字一句そのまま書き起こした (音声込み 384 トークンを 1.0 秒)。テキストだけの問いにも答える (「日本の首都は東京です。」)。デコード 8.8 トークン/秒。hwnpu-agent は同じ課題でファイルを読み、`go test` を走らせ、編集し、通ることを確かめて終えたが、直したのはテストの期待値のほうだった (2B の音声モデルの力。ツールの呼び出しとプロンプトの再利用は毎回働いた)。考えるモデルで (テンプレートに enable_thinking はないが、答えの頭が空なら自分から `<think>` を書く)、LFM3 と同じ形で考える・考えないを切り替えられる。考えない (`<think></think>` で閉じた) 答えの頭の改行は落とす
  - LFM3-Omni-2B: 同じ画像に「赤い円と青い四角」、`Rear_Right.wav` に「Rear, right.」、画像と `Front_Left.wav` に「音声の言う側の図形は」と聞いて「Red circle」。テキストだけ、hwnpu-agent も動く (エージェントは原因を取り違えて直せなかった)
  - 落とした LFM3-Audio と LFM3-Omni の safetensors は、大きさは合っていたが sha256 が Hugging Face の値と違った。16 MiB ずつ取り直して比べると、それぞれ 10 か所ほどで 2〜6 バイトのビットが反転していた (乱数の重みの照合や値の範囲では見つからない)。直してから書き出し直した。落としたチェックポイントは `x-linked-etag` の sha256 と照らすこと
- Gemma 4 E4B (`iron_export_lm.py gemma4`、文脈 4096、塊 1024): IRON の Gemma 4 のアプリケーションがそのまま書き出す (プロンプトは 35 コマンド、像は 15 GB)。Hugging Face の bf16 の CPU のロジットと、プロンプト 21 トークンとその後の 16 ステップで最上位がすべて一致 (コサイン 0.99993 以上、最大の差は最大のロジットの 3.6 %)。デコード 278 ms/トークン (3.6 トークン/秒)。塊 1024 は位置 0 から 3.4 秒、3072 から 8.2 秒で、どの 1 コマンドも 2 秒の打ち切りに届かない。hwnpu-chat は「日本の首都は東京です。」「391」(17 × 23)。hwnpu-agent は 16.66 と同じ課題で、テストを走らせて読み、`shapes.go` の `math.Pi * c.R` を `math.Pi * c.R * c.R` に直し、`go test` が通ることを確かめた (考える部分が長く、1 回に約 1100 トークン)

### 16.68 AMD: 画像生成 Anima を NPU だけで (`hwnpu-image`、汎用のグラフの書き出し) — 2026-11-05

- ComfyUI や Diffusers のような画像生成の道具は、UI とワークフロー (ノードのグラフ) と、モデルの実行 (テキストエンコーダ、拡散のデノイザー、VAE、サンプラー) に分かれる。hwnpu は後者を受け持つ。最初のモデルは Anima-Turbo (Cosmos-Predict2 2B の DiT、Qwen3-0.6B と LLM アダプタ、Qwen-Image の VAE。8 ステップ、CFG 1)
- 言語モデルの書き出し (版が decode と prompt) とは別に、入力から出力への IRON のグラフを書き出す道を足した: `tools/amd/iron_graph.py` と Go の `iron.Graph` (`internal/llm/iron/graph.go`)。同じ ELF に重みのバンクを何組も持てる (`variants`)。DiT の 28 ブロックは 4 ブロックのグラフ 1 つと 7 組のバンク
- 1024 × 1024、8 ステップで約 70 秒 (DiT 1 ステップ 7.2 秒、VAE 7 秒)。diffusers の fp32 の CPU (9 分 17 秒) と同じノイズで、潜在のコサイン 0.966 (CPU の bf16 は 0.980)。行列積の `MEDIUM` (bfp16) がテキストで誤差を広げていたので、テキストだけ `HIGHEST` にした (0.938 → 0.966、時間は変わらず)
- トークナイザー: Qwen3 の前分割 (数字を 1 桁ずつ) と、T5 の Unigram (sentencepiece の precompiled charsmap、Viterbi) を Go に足した。transformers と 11 の文 (全角、アクセント、日本語、絵文字) で一致
- 詳細、測定、書き出しの手順は [amd-npu-models.md](amd-npu-models.md) の 11 節

### 16.69 画像生成 Anima を Neural Engine と Intel の NPU にも (`hwnpu-image -convert`) — 2026-11-05

- AMD の IRON のグラフ (16.68) と同じ部品を、MIL の 1 つの記述 (`internal/imagegen/anima/mil`) から Core ML のパッケージ (`-backend coreml`) と OpenVINO IR (`-backend intel`、`ovlower`) に書く。テキスト、DiT の時刻とパッチと 4 ブロックずつ 7 つと最後の層、VAE の 10 段。変換は Go だけで、256 × 256 で 12 秒 (重み fp16 で 5.1 GB)
- 実行は `runner.Program` (`internal/imagegen/anima/programs.go`)。パイプラインは IRON と共通にし、部品の実行だけを `engine` で分けた。AMD の出力はこの作り替えの前とビット単位で同じ
- fp16 の手当て (実測: DiT の残差は 12〜20 ブロックで最大 2.7e5、ゲートつきの MLP の出力 2.3e5、ゲート 4682、Qwen3 の残差 7e3):
  - DiT の残差を 1/32 で持つ (パッチの埋め込みとゲートを 1/32 に)。正規化は入力の大きさに依らないので、残りはそのまま
  - 幅の RMS norm と LayerNorm は、行ごとの最大の絶対値で割ってからする (`maxScaled`、`reduce_max` と `real_div`)。固定の倍率 (`RMSNorm` の inScale) では、大きな値のない行の 2 乗が fp16 の非正規数になって精度が落ちる
  - VAE の値は 8 未満なので手当ては要らない
- 注意は 1024 クエリずつ (`-qchunk`)。VAE の中段は 1 ヘッド 384 次元、16384 画素
- CPU のインタプリタに 2 次元の `conv` を足した (`milcpu`)。runner の CPU の実行に `HWNPU_CPU_F16` (演算ごとに fp16 に丸める) を足した
- 確かめたこと (256 × 256、8 ステップ、diffusers の fp32 と同じノイズ): Core ML を CPU のインタプリタで、潜在のコサイン 0.99985、画像の PSNR 37.4 dB (fp16 に丸めて 0.99899、32.0 dB)。OpenVINO IR を OpenVINO 2026.4 の CPU で 0.99985、37.9 dB (f16 の推論で 0.99478、25.5 dB)。条件づけはコサイン 0.999999、ステップ 0 の速度は 0.999997
- M3 の Neural Engine での確認と手直しは 16.85。Core Ultra では未確認。手順と見るところは [handoff-anima-mac-intel.md](handoff-anima-mac-intel.md)

### 16.70 AMD: PLaMo 3 Translate 2B を hwnpu-chat で訳す (`iron_plamo3.py`、PLaMo のトークナイザーと PFML3) — 2026-11-05

- 対象は pfnet/plamo-3-translate-2b (`Plamo3ForCausalLM`、2026-10-27 公開)。IRON にアプリケーションがないので hwnpu に置いた (`tools/amd/iron_plamo3.py`、`iron_export_lm.py plamo3`)。IRON の部品だけで書けた
  - 層は sandwich norm (`SandwichLayer`)。PLaMo の RMSNorm は `(offset + w) * x` で、offset は norm ごとに違う (前の norm、q と k の norm、最後の norm は 1、注意の後は 1/5、MLP の後は 1/5^1.5)。読み込んだ後に重みへ足し込む (Gemma の `plus_one` と同じく float32 で足して bf16 に丸める)
  - `qkv_proj` (q、k、v の行) と `gate_up_proj` (gate、up の行) はまとめた重みのまま `Linear.split` で分ける。q と k はヘッドごとに RMSNorm、RoPE は rotate_half
  - 8 層ごとの 1 層 (7、15、23) が全体を、ほかは 2048 位置 (自分を含む) の窓を見る。窓の層の RoPE は θ = 10000、全体の層は θ = 10⁶ の YaRN (倍率 256、元の文脈 4096、beta 32 と 1。modeling_plamo.py の `rope_parameters` は config.json の `rope_parameters` を読まず、`rope_scaling_factor` と `initial_context_length` から作る)。YaRN の cos と sin に掛ける 0.1 ln 256 + 1 は `rope_attention_scaling`
  - 埋め込みの尺度なし、頭は `lm_head` (tie なし)。トークナイザーは tokenizer.json がないので、IRON 側では読まない (`from_checkpoint` を差し替え)
- 照合 (`tools/amd/iron_plamo3_check.py`。Hugging Face 側は torch の venv、IRON 側は IRON の venv):
  - modeling_plamo.py で作った乱数の小さな PLaMo 3 (4 層、2 層ごとに全体、窓 8、YaRN は本物と同じ) で、IRON のグラフをホストで (`OnHost`) 動かし、100 トークン (塊 64 の境をまたぐ) と 4 ステップで最上位 5/5、KL 5e-4 以下。わざと壊すと落ちる: 注意の後の offset を 1 に (KL 0.55)、窓をなくす (0.65)、YaRN の倍率をなくす (0.12)、YaRN の傾斜をなくす (0.011)、全体の層に窓の RoPE (0.22)
  - 本物の重みで、翻訳の 43 トークンのプロンプトと 16 ステップ: ホストの IRON は Hugging Face (fp32) と最上位 17/17、コサイン 0.9999 以上、KL 最大 3.8e-3。NPU (Ryzen AI 9 365) も 17/17、コサイン 0.9998 以上、最大の差は最大のロジットの 1.7 % (`ironlm` に Hugging Face のロジットを `check/` として渡した)。2596 トークン (塊の境 2 つと窓 2048 をまたぐ) と 8 ステップも NPU で 9/9、コサイン 0.9996 以上
  - 乱数の重み (実物の形) での NPU と `OnHost` の照合はコサイン 0.45〜0.8 で合わなかった。Gemma 4 と同じ (16.47)。実物の重みでは合うので、照合は実物で行う
- 速さ (文脈 8192、塊 1024。bank 3.5 GiB + 2.3 GiB、プロンプトは 3 コマンド): デコード 7.7 トークン/秒 (位置 1024 で 133 ms、8191 で 152 ms)。1024 トークンの塊は位置 0 から 2.2 秒、1024 から 4.8、2048 から 6.6、4096 から 8.2、7168 から 10.5 秒。どれも 2 秒の打ち切りに当たらなかった。窓 2047 と塊 1024 のリングは 4096 で、3 × 4096 ≥ 8192 なのでリングにならず、巻き戻しと prompt store が使える
- トークナイザー (`tokenizer.LoadPlamo`、`Load` が `.jsonl` で選ぶ): tokenizer.jsonl の点数つきの断片に対し、費用 (点数 × 10⁴ を Python の round で丸めたものの符号を反転) の和が最小の分け方を後ろから探す。tokenization_plamo.py は Aho-Corasick の表で同じ探索をするが、hwnpu は全断片の接頭辞の map で引く。同点は位置ごとに長い断片を、どの断片でも覆えない文字は UTF-8 のバイトの断片を、ほかより厳密に安いときだけ選ぶ (表の並びと同じ)。特殊トークン `<|plamo:...|>`、4 回以上続く同じ文字、2 個以上の空白の前後で切る (tokenizer_config.json の閾値)。Hugging Face の出力と、`tools/ref/tokenizer_golden.py` の 40 例 (testdata に置いた) と乱数の 3040 例ですべて一致。同点の扱いを逆にするとテストが落ちる
- チャットの書式 (`chat.Plamo3`): chat_template.jinja の PFML3 を Go に移した。`target_lang`、`source_lang`、`form`、`context`、`glossary` は `Options.Translation` (API では `chat_template_kwargs`)。テンプレートが拒む system メッセージは `context` にする。訳す先がなければ、日本語の文字を含む文は英語、ほかは日本語にし、履歴の訳文はそれぞれの原文から決めた言語で書く (テンプレートは 1 つの `target_lang` で全部を書くので、自動のときだけ違う。こうしないと次の文の言語で前の訳文の札が変わり、プロンプトの先頭が再利用できない)。`tools/ref/plamo3_template_golden.py` の 8 例でテンプレートと一致。終わりは `<|plamo:tag|>`、`<|plamo:bos|>` (generation_config の eos)
- 実機: hwnpu-chat で「The meeting has been rescheduled to next Monday.」に「会議は来週の月曜日に延期されました。」、続く日本語に英語、`/to zh-hans` で中国語。2 文目以降は前のプロンプトを再利用 (50 トークンのうち 17 だけ読む)。サーバーの `glossary` (NPU → ニューラル処理装置) と `form` も効いた。`form: plain` の 1 例で Hugging Face (fp32) と読点 1 つ違ったが、そこは Hugging Face 自身で 0.454 対 0.440 の拮抗だった (bf16 の差)
- Core ML と Intel の NPU は 16.71 で移した。0.8B、8B、31B は同じ構造なので同じアプリケーションで書き出せるはずだが、試していない

### 16.71 PLaMo 3 を移す: Core ML と Intel の NPU (`internal/llm/plamo3`) — 2026-11-05

- modeling_plamo.py の `Plamo3ForCausalLM` を Go に移し、ホワイトリストに入れた (`plamo3`)。形は Gemma 4 (16.25) から層ごとの埋め込み、KV の共有、ロジットの soft cap を除いたもの
  - norm は Gemma 4 と同じく入力を 1/64 にしてから二乗する (参照の最大 |hidden| は層 18 で 1089、1/64 なしでは fp16 の二乗があふれる)。重みには norm ごとの offset (1、1/5、1/5^1.5) を足して fp16 の定数にする
  - `qkv_proj` と `gate_up_proj` は `LinearRows` で行ごとに分ける (PLaMo-Embedding と同じ)。`lm_head` は語彙を 16 の倍数に詰めた行数を持つので、行が多ければ語彙の分だけ使う (2B は 107520 でちょうど)
  - YaRN の concentration (0.1 ln 256 + 1 = 1.5545) は cos と sin に掛けず、全注意の層の注意のスコアに二乗を掛ける (`scoreScale`、注意で分けたときは `PagedShape.Scale`)。キャッシュの鍵は concentration を含まない値になり、ホストの注意 (internal/llm/paged) はスケールを掛けるだけで済む
  - スライディング窓の層は窓 2048 のリング (Gemma 4 と同じ `StepInputs`)、全注意の 3 層 (7、15、23) で分ける (`llm.Paged`、文脈 262144)
- 合格条件 (port-architecture):
  - 小さな合成モデル (4 層、2 層ごとに全注意、窓 3、2B と同じ YaRN、語彙 40 に lm_head 48 行) を fp16 で CPU で動かし、modeling_plamo.py から書いた fp64 の forward (concentration は Hugging Face どおり cos と sin に掛ける) と相対誤差 6.5e-5。1 層ずつも 1.7e-6〜4.8e-5。KV キャッシュ付き (1、3、4、11 トークン/回、1〜3 部分)、注意で分けた形、印と巻き戻しも、キャッシュなしのプログラムと 5e-3 以内 (fp16 のキャッシュ)。OpenVINO の CPU で IR も同じ (`TestIntelIRCached`、`TestIntelIRPaged`、最大 8.4e-4)
  - わざと壊すとテストが落ちる: concentration をなくす、YaRN の傾斜をなくす、注意の後の offset を 1 に、窓をなくす (プログラムとステップのマスクそれぞれ)、全注意の層を 1 つずらす、残差の足し先を変える
  - `hwnpu-check` (31 トークン): `tensors: 0 missing, 0 mismatched, 0 unused`。fp32 で各層の相対 RMS 誤差 0.0002 以下、全体で argmax 31/31。`-f16` で各層 0.0006 以下、全体で KL 0.0001 nats、argmax 30/31
  - 重みの形式 (全体の KL、argmax): lut4 1.91 nats (15/31)、lut6 0.073 (28/31)、lut8 0.0066 (29/31)、int4 0.72 (23/31)、int4-g128 0.32 (23/31)、int8 0.0025 (30/31)。lut4 は使えない
  - M3 の Neural Engine (`hwnpu-formats -formats lut6,lut8 -ref fp16`、キャッシュ 4096、1 回 8 トークン): どれも全部 Neural Engine に載る。次のトークンが fp16 と同じなのは lut6 で 235/265 (89%)、lut8 で 259/265 (98%)。ロジットの相対誤差 (平均/最大) は 0.148/0.264 と 0.039/0.082。1 回の呼び出しは fp16 105.4 ms、lut6 53.7、lut8 57.8。重みは 4.85、1.82、2.45 GiB
  - 既定は lut8 にした (Intel は対応表で int8)。lut8 の 2 本 (4.9 GiB) はコンパイルのキャッシュ (約 4 GiB) を超えるので、hwnpu-chat は生成用の 1 本でプロンプトも読む (`-prefill auto`)。翻訳する文は短く、50 トークンで約 0.4 秒。数千トークンの文書は 64 トークンのプログラムの約 10 倍かかるので、`-format lut6` を CHAT に書いた
- hwnpu-chat: カタログに `plamo-3-translate-2b` (機能 `translation`)。トークナイザーのファイルはアーキテクチャで選ぶ (`tokenizer.jsonl` と `tokenizer_config.json`)。この Linux 機で `-use plamo-3-translate-2b -backend openvino` (OpenVINO の CPU) を実行し、int8 の IR に変換 (2 本で 4 部分ずつ、各 0.34〜0.72 GiB)、AMD と同じ 2 文を同じ訳文に訳した (2 文目は 50 トークンのうち 17 だけ読む)。Core ML の変換 (lut6、8 トークン/回) も Linux で済み、4 部分 0.25〜0.54 GiB (どのチップの 1 GiB にも収まる)
- M3 で: 注意で分けた形の 1 回 8 トークンの版は `NOT all on the NPU` だった。`hwnpuinfo` で調べると、ホストの注意 (`attn`、17 演算) と 2 段の注意 (`mass`、10 演算) と部分 1〜3 は全部 Neural Engine で、最後の部分 (層 23 の注意の後と出力層、35 演算) の `transpose` 1 つが CPU だった。注意の出力を (1, heads, seq, head_dim) で受けて token を先に並べ替える transpose で、部分 2 と 3 の頭の同じ transpose は Neural Engine に載る。そこで PLaMo 3 の部分は注意の出力を token を先にした `attn_ot_N` (1, seq, heads, head_dim) で受け、並べ替えはホストがする (`decode.putAttnOut`。ホストは塊ごとの結果をもともと自分で合わせているので、24 × 8 × 128 の写しが 1 つ増えるだけ)。`attn_o_N` を受けるほかのモデルは変わらない。並べ替えをやめると分けた形のテストが落ちる
- M3 で直したあと全部 Neural Engine に載った (`hwnpuinfo` で部分 4 も全部)
- メモリ: M3 で hwnpu-chat が起動直後 4 GB、使うたびに減ると報告された。この Linux 機 (OpenVINO の経路) で測ると 2 つあった
  - 変換した直後の起動は 2.4 GB (変換中の最大 3.2 GB)。変換で読んだチェックポイントの残りで、Go はそれを少しずつしか OS に返さない。`chat.Prepare` の終わりに `debug.FreeOSMemory` を呼び、154 MB になった (変換しない起動は 141 MB)
  - 翻訳 1 回ごとに生きたヒープが約 85 MB 増えた (5 回で 578 MB)。会話の印 (`decode.Session.Mark`、エンジンは 8 個まで持つ) が、スライディング窓の 21 層のリング (2048 位置 × 4 kv ヘッド × 128、鍵と値、計 88 MB) を毎回まるごと写していた。会話の始めはリングはほとんど 0 で、macOS はそれを圧縮するので、使うほど減って見えたと考える (未確認)。位置がリングの長さに届くまでは、位置より後ろのスロットはマスクで見えない (StepInputs: 位置 pos で スロット j ≥ pos の最新の位置は j − 長さ < 0) ので、印は各ヘッドの使っている先頭のスロットだけを持つようにした (`ringRows`、`putRing`。prompt store から戻すときも同じ形を受ける)。5 回で 183 MB (生きたヒープ 36 MB)。窓より手前で印を取り、リングが一周したあとに戻るテスト (`TestCachedMarkBeforeRingFull`) は、書き戻しの位置か持つスロットの数を誤ると落ちる。Gemma 4 と LFM3 も同じ規則のリングなので、短い会話の印は同じく小さくなる
- 未確認: 精度と速さの詳しい計測。Intel の NPU の実機 (OpenVINO の CPU のみ)。lut6 の 2 本 (約 3.6 GiB) が Neural Engine のコンパイルのキャッシュ (約 4 GiB) に収まるか

### 16.72 AMD: 固定されるメモリを減らす (Anima を段ごとに、Gemma 4 の PLE をホストに) — 2026-11-05

- NPU のバッファ (amdxdna の BO_SHARE) は全ページが固定され、プロセスの RSS に出ない。Gemma 4 E2B は RSS 0.43 GB で、`Unevictable` が 9.3 GB 増えていた。中身は重みと作業領域で、KV キャッシュは 5 % 未満
- Anima: テキスト、DiT、VAE の各段を走るときだけ読み込む。最大 14.2 → 5.0 GiB、絵はビット単位で同じ、合計 69.3 → 67.7 秒
- Gemma 4: 層ごとの埋め込みの表 (PLE、E2B で 4.4 GB) をファイルのままホストに置き、呼び出しの前にトークンの行をデバイスの小さなバッファ (18 MB) に書く (`tools/amd/iron_hostple.py`、`iron.HostTable`)。E2B は 9.08 → 4.60 GB、E4B は 14.7 → 9.25 GB、画像と音声の E2B は 10.1 → 5.6 GB で、出力は一字一句同じ、速さも同じ
- OpenVINO のバックエンドが `/dev/shm` に残していた共有バッファのディレクトリ (落ちたプロセスの分、1 日で 8.4 GB) を、開くときに片付ける
- 詳細と残り (4 ビットの重みなど) は [amd-npu-models.md](amd-npu-models.md) の 12 節

### 16.73 AMD: 4 ビットと 8 ビットの重み (q4nx、q8) — 2026-11-05

- `iron_export_lm.py gemma4 --q4`: MLP と出力の行列を q4nx (FastFlowLM の形式、1 重み 5 ビット)、注意の射影を q8 (hwnpu の形式、int8 と 32 個ごとの scale、8.5 ビット)。デコードは hwnpu の `QGEMV` (`tools/amd/iron_q4.py`、1 列 2 コアで 55 GB/s、bf16 の GEMV と同じ帯域)、プロンプトは q4nx を IRON の `flm.DequantBFP` + bfp16 の GEMM、q8 を hwnpu の `Q8Dequant` + bf16 の GEMM。埋め込みは出力の行列と共有なので PLE と同じくホストの表に
- E2B: 固定 9.08 → 1.78 GB、デコード 6.1 → 7.9 トークン/秒、2.31 → 1.46 J/トークン、bf16 との KL 0.097 nats。E4B: 14.7 → 3.53 GB、3.7 → 5.6 トークン/秒、KL 0.049。画像と音声の E2B: 10.1 → 2.82 GB
- 注意を 4 ビットにすると E2B の KL は 0.229 (`--attn q4`)、bf16 なら 0.091 (`--attn bf16`、2.02 GB)。注意は重みの 1 割強で、4 ビットでは誤差の大半、8 ビットならほぼ bf16
- q8 の GEMV は部分和を 2 つの bf16 (上位と残り) に分けて scale を掛ける。1 つに丸める q4nx のやり方では 8 ビットの精度が消える (コサイン 0.9998 → 0.999992)
- 詳細は [amd-npu-models.md](amd-npu-models.md) の 13 節

### 16.74 AMD: 待つ間は NPU のメモリを返す (`-idle`) — 2026-11-06

- IRON のモデルのバッファは固定されたメモリで、常駐する hwnpu-chat (`-listen`、OpenCode の相手) は使っていない間もそれを持ち続けていた。`-idle` (既定 5 分) の間要求がなければ、`chat.Engine` が会話をホストに写し (`putAside`)、`iron.Model.Release` でデバイスのバッファと命令列を全部放し、デバイスを閉じる。次の要求で `Reload` (同じ書き出しを読み直す) してから、写しか prompt store から続ける
- `iron.Model` はマニフェスト、ホストの表 (PLE と埋め込みのファイル)、塔の前処理を残すので、待つ間も `Images`、`Context` やメディアのトークン数はそのまま答えられる。放した間に呼ばれた実行は `errReleased` で止まる
- 測定 (プロセスの `/dev/accel/accel0` の写像の大きさ、`smaps`。ほかのプロセスが NPU を使っていても混ざらない): E2B の 4 ビットで 1775 MiB → 0、読み直し 0.8 秒、830 トークンを写しから戻して次の質問の 58 トークンだけ読んだ。画像と音声の E2B で 2818 MiB → 0、読み直し 2.0 秒、画像の後の質問に正しく答えた
- Neural Engine と Intel の NPU のプログラムは読み直しに数十秒から数分かかるので、対象は IRON だけ

### 16.75 AMD: MoE (Qwen3.6-35B-A3B) の部品を測る (`tools/amd/iron_moeprobe.py`) — 2026-11-05

Qwen3.6-35B-A3B (`qwen3_5_moe`: 40 層 = Gated DeltaNet 30 + 全注意 10、hidden 2048、専門家 256 × 中間 512、選ぶのは 8 個と共有 1 個、36B パラメータ、bf16 で 72 GB) を AMD の NPU に載せる前に、今の IRON で書ける形の MoE の層を、その大きさで測った。モデルの移植ではない (重みは乱数)。

- 形: 1 層を 1 コマンドにし、層の境に空のホストステップを置く (ホストが次の層の専門家を選ぶ場所)。選んだ 8 個の gate、up、down を、呼び出しごとの値 (スクラッチパッド `e<層>_<i>`) で引く `Copy` (`iron.weight(表)[r]`、`in_offset` を呼び出しごとに書く) で作業用の state に写し、gate と up は 8 個まとめて (4096 × 2048) の GEMV 1 回ずつ、SiLU × up × 経路の重み、down は 8 個を横に並べた (2048 × 4096) の GEMV 1 回。続けて共有の専門家 (512 幅の GEMV 3 回)、残差と RMS norm、次の層の経路 (256 × 2048)
- 照合: `irongraph` で IRON の参照とコサイン 0.9997 (12 層、RMS norm の近似の差。1 層では誤差 0)
- 実行: `go run ./tools/amd/irongraph [-pin GiB] DIR`。`-pin` は先に別の BO を確保して埋め、大きな固定メモリを試す (この節で足した)

測定 (Ryzen AI 9 365、bf16、1 層の各部分は `--split` でコマンドを分けて):

| 部分 | 量 | 時間 | 速さ |
|---|---|---|---|
| 専門家 8 個の gate と up を写す (`Copy` の既定 1 チャネル) | 32 MiB | 4.85 ms | 読み 6.9 GB/s |
| 同 (`num_channels` 4 / 8) | 32 MiB | 1.56 / 1.27 ms | 21.5 / 26.4 GB/s |
| down を写す (2048 行 × 1 KiB の飛び飛びの書き込み、4 / 8 チャネル) | 16 MiB | 1.11 / 1.41 ms | 15 / 12 GB/s |
| down を転置して連続に写す (8 チャネル) | 16 MiB | 0.69 ms | 24 GB/s |
| 写した専門家の GEMV 3 回と要素ごとの演算 | 48 MiB | 1.37 ms | 37 GB/s |
| 転置した down を 1 行 (32 行に埋めた) GEMM で | 48 MiB | 1.92 ms | GEMV より遅い。採らない |
| 共有の専門家、残差、RMS norm、経路 (演算 9 個) | 5 MiB | 0.90 ms | 演算 1 個あたり約 80 µs |
| コマンド 1 個 (小さな加算 1 個) | — | 62 µs | ホストの往復は 1 トークン 40 回で 2.5 ms |
| q4nx の GEMV (`flm.LMHead`) 4096 × 2048 を 8 個 | 5.2 MB × 8 | 0.90 ms | 47 GB/s (1 個 112 µs) |
| 同 32768 × 2048 を 4 個 | 42 MB × 4 | 3.22 ms | 52 GB/s |
| 同 248832 × 2048 (Qwen3.6 の出力層の大きさ) | 318 MB | 5.93 ms | 54 GB/s |

- **大きな固定メモリは問題ない**: 12 層 (表 19.4 GB、bank 6 本) を読み込み (12 秒)、さらに `-pin 16` で 16 GiB を先に固定した計 35 GB でも、1 層 4.9〜5.0 ms で変わらない。`ulimit -l` は unlimited の機械。4 ビットの Qwen3.6-35B-A3B (22〜25 GB) も、8 ビット相当 (36〜40 GB) も載る大きさ
- 一時、同じグラフの層が 1 層 170〜520 ms に落ちた (20:46〜20:50 の 3 回の起動)。その後の 8 回 (固定 0〜35 GB) は全て 5 ms 前後で、遅いときに他のプロセスが NPU を開いていたかは確かめられていない (その後の監視では他の利用者なし)。同じ時期に別のセッションがあったので NPU の共有を疑っているが、原因は未確定
- **写しは今の最大の無駄**: 専門家を一度写してから GEMV で読むので、選んだ専門家のバイトを 3 回動かす。`Copy` は 8 チャネルでも読み 26 GB/s で、GEMV (37〜54 GB/s) より遅い。GEMV の重みの読み込み先を呼び出しごとの値で決められれば (IRON では `Copy` だけが `in_offset` を呼び出しごとに持つ) 写しは要らない
- **4 ビットの GEMV は帯域どおり**: q4nx は 5 ビット/パラメータで 47〜54 GB/s。ただし今あるのは Gemma 4 の出力層の形 (RMS norm と softcap を含む `LMHead`) だけで、MoE の部品には一般の q4nx GEMV (段階 A6) が要る
- **演算の数が帯域より効く**: 種類の違う演算が並ぶと 1 個あたり約 80 µs かかる (同じ演算が続く q4nx の 8 個は 1 個あたりの上乗せ 20 µs 未満)

Qwen3.6-35B-A3B の 1 トークン (q4nx、今の部品のまま) の見積もり: 層ごとに、専門家の写し 0.6〜0.8 ms + その GEMV 0.35 ms + 専門家以外の重み (GDN または注意、共有の専門家、経路で約 24 MB) 0.5 ms + 演算 25 個前後の上乗せ約 2 ms ≒ 3.5〜3.8 ms。40 層で 140〜150 ms、出力層 6 ms、コマンドの往復 2.5 ms で **約 150 ms (6〜7 トークン/秒)**。読むバイトだけなら 1.8 GB で約 40 ms (25 トークン/秒) なので、時間の 6 割は演算ごとの上乗せ。速くする順は (1) 写しをやめる (専門家の GEMV が読み込み先を呼び出しごとに取る)、(2) 層の演算をまとめる (FastFlowLM の `DecodeLayer` のような 1 層 1 演算)、(3) 経路の選択をデバイスで。いずれも NPU の仕事 (動かすバイトと演算の数) を減らす手

- まだ測っていないもの: Gated DeltaNet の IRON の形 (状態の更新と、プロンプトの塊ごとの delta rule)、プロンプトの MoE (塊のトークンを専門家ごとに並べ替える GEMM)
- 試し方: `python tools/amd/iron_moeprobe.py OUT --layers 12 --channels 8 --check` (12 層で約 2.5 分、19 GB)、`--split` で部分ごと、`--down-t` で down を転置、`--tiny 8` でコマンドの費用、`--q4 ROWS --reps N` で q4nx の GEMV

#### 4 ビットの専門家をその場で読む (`--q4moe q4`) — 2026-11-06

16.73 の 4 ビットの GEMV (`iron_q4.py` の `QGEMV`) は hwnpu の演算なので、重みの読み込み先を呼び出しごとの値で決める形を足せた。これで上の速くする手の (1) 「写しをやめる」が今の部品で書ける。

- `ExpertQGEMV` (gate と up 用): 1 列 = 1 専門家。列 `l` は表 (専門家 256 個の q4nx のブロックを並べたもの) の `e_l` バイト目から、その専門家の (512, 2048) を読む。`e_l` は層ごとのスクラッチパッドの値 × 専門家 1 個のバイト数 (Affine) で、シム DMA の BD のアドレスに足される (要素の単位なので uint8 の表ならバイト)。出力は 8 個の専門家の 512 個ずつを並べた 4096
- `ExpertDownQGEMV` (down 用): 8 個を横に並べた (2048, 4096) の GEMV。列は出力の 256 行 (列ブロック 4 個) を受け持ち、専門家ごとに 1 回、その列ブロックの部分を呼び出しごとの位置から読む (列あたり 8 回の fill)。カーネルは専門家の順に回り、列ブロックごとの累算を 4 個持つ。合計は列の中で済むので、専門家ごとの出力を足す演算は要らない
- 共有の専門家は `QGEMV`、経路と RMS norm は bf16 のまま。写しの state も `Copy` もない
- 照合: 演算 1 個ずつは IRON の参照 (選んだ 8 個だけを逆量子化) とコサイン 0.99994 (gate)、0.99993 (down)。層を重ねると 1−cos が 2 層ごとに約 2 倍になる (4 層 0.9996、8 層 0.9985、12 層 0.994)。どこかの層だけが壊れているのではなく、`QGEMV` の部分和を bf16 に丸める差 (16.73 の Gemma 4 と同じ) が、乱数の重みの層で増幅される。bf16 の写しの版は 12 層 0.9997。実際の重みでの影響は KL で見る必要がある
- 参照 (`reference`) は呼び出しごとの値を名前付きの引数で受ける。`**kwargs` では渡されず、0 番の専門家と比べてしまう
- aiecc が「完了の保証のない `dma_free_task` の後の BD の再利用」を警告する (down の重みの fill の後に、同じシムで次の演算が BD を使うとき)。fill を読み終えないと出力は書けないので実害はないと見ているが、静的には証明されていない

測定 (12 層、他に NPU を開いたプロセスがないのを確かめて):

| | 1 層 | 備考 |
|---|---|---|
| bf16、写してから GEMV (上の形、8 チャネル) | 5.2〜5.7 ms | |
| **q4nx、その場で読む** | **2.3〜2.5 ms** | 層の重み 19 MB (専門家 15.7 MB) |
| うち `ExpertQGEMV` 1 回 (5.2 MB) | 0.245 ms | コマンド込み、21 GB/s |
| うち `ExpertDownQGEMV` 1 回 (5.2 MB) | 0.46 ms | 11 GB/s。列あたり 8 回の小さな fill が遅い |

- 専門家の部分は 3 回で約 0.95 ms。残りの約 1.4 ms は共有の専門家 (`QGEMV` 3 回) と要素ごとの演算、RMS norm、経路の約 10 個で、演算ごとの上乗せがそのまま残る
- 表は層あたり 503 MB (q4nx、gate と up と down)、40 層で 20.1 GB。12 層 (6.1 GB、bank 2 本) で試した。40 層は他のセッションが 40 GB 使っていたので試していない (bf16 の版で 35 GB の固定は確かめてある)
- 見積もりの更新: MoE の部分は 40 層で約 95 ms (写しの版の q4 見積もりは約 145 ms)。Gated DeltaNet と注意の部分 (未実装、射影を q8 として 1 層 1〜1.5 ms と見込む)、出力層 6 ms、往復 2.5 ms を足して 1 トークン 140〜160 ms、**6〜7 トークン/秒のまま**。写しは消えたが、時間の大半は演算の数。次は down を「1 列 = 1 専門家」にして足し算をまとめる形との比較、共有の専門家を `ExpertQGEMV` に入れる (9 個目の専門家)、層の要素ごとの演算をまとめる
- 試し方: `python tools/amd/iron_moeprobe.py OUT --layers 12 --q4moe q4 --check` (12 層で約 1 分、5.7 GB)、`go run ./tools/amd/irongraph -tol 0.05 OUT`

#### 層を 4 演算に (`--q4moe q4 --fused`、`kernels/moe_mv.cc`) — 2026-11-06

上の次の手を 3 つ試した: 共有の専門家を 9 個目として同じ演算に入れる、要素ごとの演算を GEMV の出口に入れる、down を「1 列 = 1 専門家」にする形との比較。

- `MoEGateUp`: 表は専門家 257 個 (最後が共有) の (1024, 2048) で、gate と up の列ブロック (64 行) が交互。列 `l` は選ばれた `e_l` の 8 列ブロック分と、共有の専門家の `l` 番目の列ブロックを読む (共有の 512 行は 8 列で 64 行ずつ)。コアは gate と up の累算を両方持ち、出口で `silu(g) * u * w[専門家]` を書く (`moe_glu`、SiLU は mlir-aie の `silu_vec`)。出力は 9 × 512
- `MoEDown`: 9 個を横に並べた (2048, 4608) の GEMV で、残差を出口で足す (`moe_end_add`)。列は専門家ごとに自分の 4 列ブロックを連続した 1 回の読み込みで受ける (前の `ExpertDownQGEMV` は 3 次元の飛び飛び)
- 層は `MoEGateUp`、`MoEDown`、RMS norm、経路の 4 演算 (前は約 15)
- コアの入力 DMA は 2 本 (重みと x) しかないので、経路の重み (16 個) と残差 (列の 256 行) は x の後ろに付けて 1 つのオブジェクトにする。memtile での join はシムの出力チャネルが足りない (列あたり 3 本が要る) ので、同じ ObjectFifo に x と経路の重み (または残差) を続けて流す。`MoEDown` は 1 列 11 回の転送で BD が 16 を超えるので `--reclaim-runtime-bds` (Copy と同じ)
- 照合: `MoEGateUp` はコサイン 0.99992、`MoEDown` は 0.99996。12 層で 0.997 (分けた形は 0.994。演算が減った分、丸めも減った)

測定 (NPU を他が使っていないとき。単体の演算は 1 回目と 2 回目で 1.4 倍ほど違う。電力の状態と見て、比べるのは同じ回の中):

| | 1 回目 | 2 回目 |
|---|---|---|
| **層 (12 層の平均)、分けた形 (`--q4moe q4`)** | 2.31 ms | 2.33 ms |
| **層、4 演算 (`--fused`)** | **1.27 ms** | **1.27 ms** |
| `ExpertQGEMV` (gate だけ、8 個) | 0.25 ms | 0.17 ms |
| `MoEGateUp` (gate と up、9 個、SiLU と積) | 0.46 ms | 0.34 ms |
| `ExpertDownQGEMV` (8 個、飛び飛び) | 0.47 ms | 0.31 ms |
| `MoEDown` (9 個、連続、残差込み) | 0.53 ms | 0.33 ms |
| down を「1 列 = 1 専門家」(GEMV だけ、8 × 2048 の出力) | 0.26 ms | 0.20 ms |
| 同、8 個の出力を足す加算 7 回込み | 0.88 ms | 0.56 ms |

- **「1 列 = 1 専門家」の down は GEMV だけなら速いが、8 本の出力を足すのに加算 7 回 (または還元の演算) が要り、合計では `MoEDown` の 1.7 倍**。列をまたいだ和はコアの中ではできない。`MoEDown` を採る
- 層は 2.3 → 1.27 ms。読むのは 1 層 18.7 MB (専門家 9 個 17.7 MB と経路)。`MoEGateUp` 単体は 35 GB/s、`MoEDown` は 18 GB/s で、down の方はまだ上乗せが大きい
- 見積もりの更新: MoE の部分は 40 層で約 51 ms。Gated DeltaNet と注意 (未実装、1 層 1〜1.5 ms と見込む) 40〜60 ms、出力層 6 ms、往復 2.5 ms で 1 トークン 100〜120 ms、**8〜10 トークン/秒**。注意の側も同じように演算をまとめれば、さらに縮む
- 試し方: `python tools/amd/iron_moeprobe.py OUT --layers 12 --q4moe q4 --fused --check` (12 層で約 1.5 分)、`go run ./tools/amd/irongraph -tol 0.05 OUT`

### 16.76 AMD: PLaMo 3 Translate 2B を 4 ビットに (`iron_export_lm.py plamo3 --q4`) — 2026-11-06

- 16.73 の形式を PLaMo 3 に使った: MLP と出力の行列は q4nx、注意の射影は q8、埋め込み (出力の行列と別の重み、0.55 GB) はホストの表。まとめた `qkv_proj` と `gate_up_proj` は 1 つの重みのまま量子化し、行の範囲を列ブロック (出力 64) のバイトの範囲として切る (`iron_q4.q_rows`、`Linear.split` の差し替え)。デコードは重み全体を `QGEMV` 1 回で掛けて出力を切る (分けるより 6 % 速い)
- 固定されたメモリ 5.96 → 2.16 GiB、デコード 7.3 → 11.6 トークン/秒 (位置 2600 付近)、プロンプトは同じ。Hugging Face (fp32) とは 2596 トークンと 8 ステップで最上位 9/9
- 精度: bf16 との KL は訳文の側で 0.009 (英 → 日) と 0.007 (日 → 英)、最上位の一致 98 %。44 文の訳のうち 36 文が bf16 と同じで、残りも正しい言い換え。翻訳では 4 ビットの損はほとんど見えないので、README の書き出しを `--q4` にした
- 速さは別のセッションの CPU の負荷の下で測った (単体の q4nx GEMV が 16.73 のときの 1.6 倍遅い)。静かな機械でもう一度測る。詳細は [amd-npu-models.md](amd-npu-models.md) の 13 節

### 16.77 AMD: LFM3 とその多モーダルを 4 ビットに (`iron_export_lm.py lfm3 --q4`) — 2026-11-06

- 16.73 の形式を LFM3 (`iron_lfm3.Lfm3`、`install_q4`) に使った: MLP と出力の行列は q4nx、注意 (ゲートの重みを含む) と短い畳み込みの射影は q8 (`--attn`、`--conv`)、埋め込み (出力の行列と共有) はホストの表。`in_proj` は 16.76 の `Linear.split` で 1 つの重みのまま切る。本体が同じなので LFM3-VL、Audio、Omni も `--q4` で同じになる (塔は bf16)
- どこを何ビットにするかを CPU で量子化して戻した重みで比べた: 誤差の大半は MLP の 4 ビット (KL 0.031)、注意と畳み込みは 8 ビットならほぼ無損失 (すべて q8 で 0.0004)。既定の組み合わせは 0.035。文の先頭に BOS を置かないと 0.139 になり、0 層目の MLP だけが効いて見えた (測るときは BOS を置く)
- LFM3-1.7B (Ryzen AI 9 365、文脈 12288、塊 2048): 固定 3.75 → 1.64 GiB、デコード 8.7 → 11.2 トークン/秒 (位置 2048 で 117 → 92 ms)、1.34 → 1.05 J/トークン、プロンプトは同じ。NPU の bf16 との KL 0.036 (Gemma 4 E2B の 3 分の 1)、Hugging Face の fp32 と 0.036 (bf16 は 0.0007)
- 実機で bf16 と同じ問いに同じ答え: 東京、391、3,776 m、画像の図形と左右、音声の書き起こし。hwnpu-agent は円の面積の誤りを直して `go test` を通した。VL、Audio、Omni の固定は 4.1〜4.5 → 2.0〜2.4 GiB。README の書き出しを `--q4` にした
- 詳細は [amd-npu-models.md](amd-npu-models.md) の 13 節

### 16.78 AMD: Anima の DiT を 8 ビットに、コアの丸めを最近接偶数に (`iron_anima.py --dit-attn`、`--dit-mlp`) — 2026-11-06

- 画像生成の Anima で 4 ビットを試した。DiT のブロックの射影 (DiT の重みの 9 割) を 16.73 の q4nx か q8 にする道を足し (`--dit-attn`、`--dit-mlp`)、既定を q8 にした。固定されたメモリの最大 5020 → 3445 MiB、書き出し 5.7 → 4.0 GB、1 枚 67.3 → 68.3 秒 (重みを戻す分 2 %)、電力量は同じ (約 1050 J)。diffusers の fp32 との潜在のコサインは bf16 の 0.966 に対し 0.961 (ほかの 2 枚でも 0.974 / 0.970、0.824 / 0.823) で、ほぼ同じ絵になる。軌跡が分かれやすい seed では構図が変わることがある
- q4nx (2710 MiB) は 0.777 で、顔つきや手が変わる。CPU で重みだけを量子化しても 0.753 なので、量子化そのもの。8 ステップに蒸留した DiT は 4 ビットの丸めを絵まで運ぶ。GPTQ (同じ形式) で速度の誤差は 0.114 → 0.070 になるが、q8 の 0.010 には遠い。q4 から q8 で増えるのは 735 MiB なので q8 を選んだ
- 最初の NPU の q8 と q4 は CPU の量子化より 3〜4 倍悪かった。AIE のコアは丸めが切り捨てのまま始まり、bf16 と bfp16 に狭めるたびに値が下に寄って、入力に平均がある DiT では入力の数だけ積み重なっていた (IRON の `DequantBFP` も同じ)。hwnpu のカーネルは最初に最近接偶数にし、q4nx をプロンプトで bf16 に戻す `q4_dequant` を足した。言語モデルでは差がほとんど見えず (書き出し直して Gemma 4 E2B の KL 0.097 → 0.093、LFM3 0.036 → 0.036)、bf16 の経路は塊が 5〜9 % 遅いので、言語モデルのプロンプトは `DequantBFP` のまま
- 詳細は [amd-npu-models.md](amd-npu-models.md) の 11 節と 13 節

### 16.79 Linux と Windows でも同じファイルをディスクの上で共有する (FICLONE、ReFS、AMD の書き出しどうし、ファイルの先頭) — 2026-11-06

- 動機: 16.64 の共有は macOS (APFS の clonefile) だけで、Linux では何もしなかった。AMD の書き出しは同じモデルの版 (画像と音声つき、4 ビット、オプション違い) ごとにディレクトリが分かれ、同じファイルを持つ。開発機 (Ryzen AI 9 365) の `~/models` (297 GB) を中身のハッシュで比べると、まるごと同じ 1 MiB 以上のファイルが 68.7 GiB あった (大半は試しの書き出し。Gemma 4 E2B の `ple.bin` 4.4 GiB が 5 つ、bank 3.5 GiB が 6 つなど)。利用者の機械でも Gemma 4 E2B の文だけと画像と音声つき (`gemma4` と `gemma4mm`) を両方置けば、`ple.bin` (4.4 GiB)、`embed.bin` (0.75 GiB)、`prompt.elf`、`decode.elf`、`tokenizer.json` が同じ。LFM3 とその VL、Audio、Omni の 4 ビットの 4 つでは `prompt.elf` と `decode.elf` などの約 0.3 GiB (bank は違う)
- `cache.Clone` の Linux 版: ioctl FICLONE で写す。Btrfs、reflink つきの XFS (xfsprogs 5.1 からの既定)、bcachefs、ブロッククローンを有効にした ZFS が共有し、ext4 と tmpfs は断るので何もしない。golang.org/x/sys は使わず syscall で呼ぶ
- 順序を変えた (macOS も): 先に写しを作ってから dst と比べる。写しはメタデータだけで安く、写せないファイルシステムでは 1 バイトも読まずに分かる (Linux の多くを占める ext4 で、同じ大きさのファイルを全部読んでから断られることがない)。比べる相手は写しなので、比べる間に src が書かれても dst には入らない。dst の権限と更新時刻は保つ
- すでに共有しているものは飛ばす (`sameBlocks`、Linux だけ): FS_IOC_FIEMAP でエクステントの物理位置を比べ、全部同じなら写しどうしなので読まない。共有済みの書き出しをもう一度見るのは 0.5 ms (以前は同じバイトを読み直していた)。macOS には安く問う手段がないので今までどおり読み比べる (`.weights-shared` で 1 回だけ)
- 使うところ: 16.64 の 3 か所はそのまま Linux でも効く (Intel の IR の変換で Q=8 と Q=64 の `.bin`、`chat.Prepare` の `ShareWeights`)。新しく AMD の書き出し (`iron.ShareExport`、`cache.ShareFiles`): hwnpu-chat と hwnpu-agent の `-iron`、hwnpu-image の IRON の書き出しを読む前に、そのディレクトリの 1 MiB 以上のファイルを、隣の書き出し (manifest.json があるか、名前に `.amd-npu` を含むディレクトリ) か自分の前のファイルに同じバイトのものがあればその写しにする。置き換えるのは読み込むディレクトリのファイルだけで、隣は読むだけ。印は `.files-shared` で、書き出し直して印より新しいファイルができたらもう一度見る
- ファイルの先頭の共有 (`cache.ClonePrefix`): 文だけの書き出しの bank (`bank_scratch.bin`、1.67 GiB) は、画像と音声つきの bank (2.64 GiB) の先頭とバイト単位で同じ (塔の重みが後ろに足してある)。大きさの違うファイルは `Clone` に掛からないので、同じ名前のファイルどうしで、共通の先頭をファイルシステムのブロック単位で共有する。手順は `Clone` と同じく写しが先: dst の丸ごとの写しを作り、その先頭に src のブロックを写し (FICLONERANGE)、写しと dst を比べて最初に違う位置を見つけ、そのブロックから後ろに dst 自身のブロックを写し戻して、dst に置き換える。違うのが途中からでも (大きさが同じでも) 共通の部分だけを共有する。1 MiB に満たない共通部分は置き換えない。macOS には範囲の写しがないので何もしない
- すでに共有しているかの見方を、全部が同じかどうか (`sameBlocks`) から「先頭の何バイトが同じブロックか」(`sharedPrefix`) に広げた。FIEMAP のエクステントを区間として突き合わせ、物理位置が続いて一致する長さを数える (区切り方が違っても合う。圧縮や inline のエクステントは数えない。少なめに言うことはあっても多めには言わない)。`ShareFiles` は、候補のどれかとすでに共有しているファイルを飛ばす (別の写しに付け替えても何も返らない)
- Windows (`clone_windows.go`): ReFS と Dev Drive のブロッククローン (FSCTL_DUPLICATE_EXTENTS_TO_FILE)。FSCTL_GET_INTEGRITY_INFORMATION に答えるのが ReFS だけなので、それで写せるかとクラスターの大きさを知る。写し先は元と同じ整合性ストリームと疎の属性にし (違うと断られる)、長さを合わせてから、1 回 1 GiB までに分けてクラスター単位で写す (最後のクラスターは末尾を越えて丸ごと)。範囲の写しも同じ呼び出し。共有済みかを安く問う手段はないので、書き出しを最初に読むときに読み比べる。NTFS では何もしない
- 結果 (ext4 の上のループの Btrfs 24 GiB に、Gemma 4 E2B の 4 ビットの書き出し 2 つ、文だけ 7.0 GB と画像と音声つき 8.2 GB を普通に写して): 画像と音声つきを先に読むと、まるごと同じ 5 ファイル 5.32 GiB を 1.6 秒で共有し (ページキャッシュから)、bank の先頭 1.67 GiB を 2.4 秒で共有した。使用量は 15,563 → 8,443 MiB (15.2 → 8.2 GiB)。どちらの書き出しも同じ問いに「Tokyo」と答え、bank は元とバイト単位で同じ。共有済みの書き出しを見直すのは 2〜11 ms
- hwnpu-image (Anima の bf16、q4、q8 の書き出しを同じ Btrfs に写して。q4 と q8 の `text/` と `vae/` はもともと bf16 へのシンボリックリンクで、たどらない): q4 を読むと 9 ファイル 0.06 GiB (`dit_in` と `dit_out` の bank がまるごと、DiT の 7 つの bank の先頭 4 MiB ずつ)、bf16 では VAE の段の同じ `graph.elf` などを共有し、3 つで 0.09 GiB 減った。同じ種 (`-seed 7`) の画像は、ext4 の元の書き出しと共有後の Btrfs で bf16 も q4 もバイト単位で同じ PNG (1 枚 69〜72 秒、共有は 0.2〜0.5 秒)
- テスト (`TestClone`、`TestClonePrefix`、`TestShareFiles`、`TestShareWeights`、`TestShareExport`) は Btrfs と XFS (reflink=1) で通り、ext4 と tmpfs では共有しないことだけを見る (`TMPDIR` で場所を選ぶ)。Windows は windows/amd64、386、arm64 へのコンパイルと vet だけ
- 残り: Windows の実機 (ReFS か Dev Drive) での確認。bcachefs、ZFS、NFS での確認。Anima の DiT の bank は 4 ビットと 8 ビットで先頭 4 MiB の後が違い、共有できるのはそれだけ。開発機の `/` は ext4 なので、ここの 68.7 GiB は減らない

### 16.80 AMD: Gemma 4 のプロンプトで KV を共有する層を最後の行だけにする (`iron_gemma4.py`) — 2026-11-06

- 動機: oMLX 0.7.4 の Prefill Accelerator (速い GPU の Mac に前半の層を任せ、2 台でパイプラインにする) を調べた。hwnpu は省電力も目標なので、GPU などを足して速くする道は採らない。同じリリースの DeepSeek の CED prefill (後半の層を、後ろに塊が続くプロンプトでは飛ばす) と同じく、計算そのものを減らす
- Gemma 4 E2B は 35 層のうち後ろの 20 層 (`num_kv_shared_layers`。E4B は 42 層のうち 18 層) が鍵と値を書かず、同じ種類の最後の前の層 (E2B は 13 と 14) のキャッシュに注意する。プロンプトの塊が後に残すのはキャッシュと最後の行のロジットだけなので、共有の層の出力が要るのは最後の行だけ。しかも E2B の共有の層は MLP が 2 倍幅 (`use_double_wide_mlp`) で重い
- `tools/amd/iron_gemma4.py` の `Gemma4Tail` (IRON の `Gemma4` の `forward` を差し替えた hwnpu のアプリケーション): 塊は全部の行をキャッシュを持つ層に通し、最後の行だけをデコードのステップと同じ形 (`Call(prompt=False)`: 1 行の GEMV、位置 `position` の 1 つの問いの MHA、RoPE の表のその行) で共有の層に通す。層ごとの入力 (PLE) の射影は最後の行の埋め込みの写しから取る。トークンの表の行は、16.72 のホストの表ならホストが塊の行ごとに書いたものの最後の行 (`_Tail.row`、`iron_hostple.layer`)、`--device-ple` なら呼び出しに渡す `token` (塊の最後のトークン) で取り出す。多モーダル (`Gemma4TailMultimodal`) の placed の塊は `from_rows` で `token` を `Call` に入れる
- `iron_export_lm.py gemma4` と `gemma4mm` はこの版を書き出す (`--q4` も同じ)。IRON のままの版は `gemma4full`、`gemma4mmfull`。manifest のモデル名は `Gemma4` のまま (`hwnpu_name`。チャットの書式はこれで決まる)。Go の側は変えていない
- コマンドの切れ目: ホストの表 (16.72) は 1 コマンドのスクラッチパッドのワードを 24 に下げ、IRON のままの版の長いコマンドを分けている。これを `Gemma4Tail` にそのまま使うと、共有の層 (デコードの形) が 2 つのコマンドに分かれ、2 つ目の塊の最初のコマンドが amdxdna の打ち切り (60 秒待って `ERT_CMD_STATE_TIMEOUT`) まで止まった。塊の間にデコードのステップ (別のハードウェアコンテキスト) を挟むか、別のプロセスが NPU を使っていると止まらない。ワードを IRON の既定の 32 にすると共有の層は 1 つのコマンドに収まり、止まらない。`Gemma4Tail` の書き出しは 32 にした (`iron_gemma4.SCRATCHPAD_WORDS`、`iron_hostple.install(words=)`)。E2B のプロンプトは 50 → 6 コマンド、E4B は 62 → 10 で、最も長いコマンドは位置 3072 で 0.87 秒 (E4B 0.71 秒)。止まる仕組みは分かっていない (書き出しで、完了を保証しない `dma_free_task` の後にバッファ記述子を使い直すという aiecc の警告が、デコードの形の設計に約 500 出る)
- 書き出し (文脈 4096、塊 1024): E2B 5 分 49 秒 (最大 RSS 19.3 GB)、E2B q4 7 分 22 秒、E4B 7 分 17 秒 (27.4 GB)、E2B 多モーダル 9 分 14 秒
- 照合 (Ryzen AI 9 365、`ironlm -record` を足した: その書き出し自身のロジットを check/ として書く): IRON のままの書き出しで、IRON の prompt.txt の 2600 トークン (塊 3 つ) と貪欲法の 3 ステップを記録し、`Gemma4Tail` の書き出しで照らした。プロンプトのロジットの差は、最後の行の共有の層を GEMM ではなく GEMV で計算するため。その後のデコードは 3 つとも差 0.000 で、書かれたキャッシュは IRON のままの版とビット単位で同じ

| 書き出し (ホストの表) | プロンプトのロジット | その後の 3 ステップ | 2600 トークン |
|---|---|---|---|
| E2B bf16 | 最上位一致、コサイン 0.99992、最大の差は最大のロジットの 4.9 % | コサイン 1.00000、差 0.000 | 10.1 → 4.48 秒 |
| E2B q4 (`--q4`) | 最上位一致、0.99985、6.3 % | 1.00000、0.000 | 10.4 → 4.64 秒 |
| E4B bf16 | 最上位一致、0.99957、5.2 % | 1.00000、0.000 | 12.9 → 7.84 秒 |

- hwnpu-chat (`-iron`、思考なし): E2B、E2B q4、E4B とも「日本の首都は東京です。」、続けて「391」(17 × 23。49 トークンのうち 22 だけ読む再利用も働いた)
- 速さと電力量 (`ironlm -depth`、文脈 4096 を 1024 の塊 4 つで。電力はパッケージ全体の積算。ほかに NPU と CPU を使うものがないときに測った。デコードのステップはどれも変わらない: E2B 167〜179 ms、q4 133〜141 ms、E4B 276〜291 ms):

| 1024 トークンの塊 | E2B: IRON のまま | E2B: 最後の行だけ | E2B q4: IRON のまま | E2B q4: 最後の行だけ | E4B: IRON のまま | E4B: 最後の行だけ |
|---|---|---|---|---|---|---|
| 位置 0 から | 2.60 秒、30.0 J | 1.10 秒、16.7 J | 2.75 秒、30.2 J | 1.22 秒、13.3 J | 3.41 秒、39.9 J | 2.10 秒、24.5 J |
| 1024 から | 4.06 秒、49.9 J | 1.73 秒、21.8 J | 4.16 秒、49.7 J | 1.83 秒、21.1 J | 5.07 秒、64.1 J | 3.04 秒、38.1 J |
| 2048 から | 5.23 秒、62.5 J | 2.24 秒、26.3 J | 5.35 秒、63.3 J | 2.35 秒、27.0 J | 6.51 秒、79.6 J | 3.85 秒、46.9 J |
| 3072 から | 6.45 秒、75.6 J | 2.76 秒、32.1 J | 6.55 秒、76.7 J | 2.86 秒、32.8 J | 7.92 秒、95.4 J | 4.66 秒、56.7 J |
| 4K 全体 | 18.3 秒、218 J | **7.8 秒、97 J (2.3 倍、−56 %)** | 18.8 秒、220 J | **8.3 秒、94 J (2.3 倍、−57 %)** | 22.9 秒、279 J | **13.7 秒、166 J (1.7 倍、−40 %)** |

  - 位置で伸びる分も半分以下になった。窓の外を飛ばす前でも、E2B の 35 個の注意のうち 20 個 (共有の層) が最後の 1 行だけになるため。E4B は共有が 18 / 42 層で、MLP も 2 倍幅ではないので、減り方は小さい
- 多モーダル (`gemma4mm`、ホストの表。プロンプトは 50 → 6 コマンド、placed は 50 → 7。ワードを 32 に戻したので、視覚と音声の塔も 2 → 1 コマンド): 白地に赤い円と青い四角の 640 × 480 の画像に「この画像には、左側に赤い円、右側に青い四角が描かれています。」、続けて「画像では、赤い円の方が青い四角よりも大きく見えます。」、ALSA の試験音に「Front, left.」。IRON のままの書き出しと一字一句同じ。ただし 1 回目の比べでは 2 問目が分かれた (IRON のまま「面積が大きいように見えます」、最後の行だけ「ほぼ同じに見えます」)。IRON のままの書き出しも走らせるたびにこの答えが変わる (2 回目は上の文) ので、この変更によるものではない。何が揺れるのかは調べていない
  - 読む時間 (チャットの表示): 画像つき 291 トークン 3.3 → 2.9 秒 (視覚の塔を含む)、2 問目の 20 トークン 0.7 → 0.4 秒、音声つき 58 トークン 0.8 → 0.6 秒。短いプロンプトでは差は小さい
- 未確認: ホストは塊の行ごとに全部の層の PLE を書くが、共有の層が読むのは最後の行だけ (1 行あたり 20 層 × 512 バイトの写しが無駄)。共有の層のワードの数で止まる理由

### 16.81 AMD: 塔は使うときだけ読む (バンクの文の分だけを確保する) — 2026-11-06

- 動機: 画像と音声つきの書き出しは文だけの質問にも答えられるが、塔の重みも常に固定される (Gemma 4 E2B の 4 ビットで 1.78 GB に対して 2.82 GB)。そのため文だけの書き出しも並べて置くことになり、ディスクの重複 (16.79) の大本になっていた
- 書き出しは手を入れずに済んだ: バンクは 1 つ (`scratch`) だが、IRON の並べ方で文の重みと作業領域が先頭にあり、各版の引数が使うバイト数を manifest に持つ。Gemma 4 E2B の画像と音声つきの 4 ビットでは decode 1.71 GB、prompt 1.79 GB (文だけの書き出しのバンクとバイト単位で同じ先頭)、vision 2.17 GB、audio 2.83 GB、placed (ソフトトークンを読むプロンプト) 2.83 GB
- `iron.Model` (`loadBanks`、`growBanks`、`Manifest.textBanks`): 塔のある書き出しは、decode と prompt が使う分だけバンクを確保して読む。塔か placed の塊が初めて走る前に、全体の大きさのバッファを確保し、文の分をデバイスが残したまま写し (重みと、続いている会話のキャッシュと状態)、塔の重みをバンクの画像のその位置から読み、古いバンクと、それに結んだ decode の run を放す (Step が作り直す)。会話はそのまま続く。Release まで全体のままで、`-idle` の後の Reload ではまた文の分だけになる。文の版のバッファが引数の言うバイト数を越えるとき、xclbin をステップごとに持つ書き出し (NPU1) では、今までどおり全体を読む
- 実機 (Ryzen AI 9 365。固定されるメモリはプロセスの `/dev/accel/accel0` の写像の大きさ。main と同じ問いを同じ順に、温度 0):

| | main | 今回: 文だけの間 | 今回: 塔の後 |
|---|---|---|---|
| Gemma 4 E2B 画像と音声つき | 2818 MiB | **1826 MiB** | 2818 MiB |
| LFM3-Omni-2B | 2448 MiB | **1679 MiB** | 2448 MiB |
| Gemma 4 E2B 画像と音声つき、bf16 とホストの表、16.80 の書き出し (バンク 2 つのうち 1 つを後から) | 5665 MiB | **4607 MiB** | 5665 MiB |

- 文だけの書き出し (Gemma 4 E2B、1775 MiB) との差は 51 MiB (塔の入出力のバッファ) だけで、画像と音声つきの書き出し 1 つで文だけの用途も足りる。塔を読み込むのは 0.7 秒 (Gemma 4 の 4 ビット、992 MiB)、0.6 秒 (LFM3-Omni、768 MiB)、0.5 秒 (16.80 の bf16 の書き出し、1058 MiB) で、その分だけ最初の画像の答えが遅い (8.5 → 9.0 秒、4.6 → 5.1 秒)
- 答えは main とすべて同じ: 「Tokyo」、画像の説明の文 (魔女の絵)、音声の書き起こし「Front, left.」。文の会話 (名前と好きな色) の続きに画像を送ると、塔を読む前のキャッシュの 31 トークンを使い回したまま、名前と色と帽子を答えた (main と同じ文)。`-idle` で放した後の読み直しはまた文の分だけ (1826 MiB) になり、そのあとの画像も同じ答え
- 塔を初めて読むときは一瞬、古いバンクと新しいバンクが両方固定される (Gemma 4 E2B の 4 ビットで 1.67 + 2.64 GiB)
- テスト: `TestLazyTowerBanks` (模擬の XDNA デバイスで、文の分だけを読み、デバイスが書いたキャッシュの行が育てた後も残り、塔の分は画像から、画像の末尾から先は 0 になること)

### 16.82 Neural Engine: プロンプトを 1 回 128 トークンで読み、ロジットは最後の行だけ (`pagedPrefillSeq`、`-last-row`) — 2026-11-06

- 動機: improvements.md の prefill の施策のうち Apple に入っていないものを洗った。AMD では塊の大きさを測って 2048 → 1024 にした (16.54) が、Neural Engine のプロンプトのプログラムは最初から 64 のまま測っていなかった。もう 1 つは、16.80 の「後に残るのはキャッシュと最後の行のロジットだけ」という見方。プロンプトのプログラムは 64 行すべてのロジット (LFM3 は語彙 128000、Gemma 4 は 262144) を計算して出していたが、ホストが読むのは最後の行だけ
- 部分ごとの時間 (M3、入力 0 のまま 20 回の中央値。`hwnpu.Graph` を直接回す使い捨ての計測):

| LFM3-1.7B lut6 | Q=8 | Q=64 | Q=128 |
|---|---|---|---|
| 層 4 つの部分 (part6) | 3.42 ms | 3.62 ms | 5.47 ms (1 トークンあたり 64 の −25 %) |
| 層 2 つ (part1) | | 2.19 ms | 3.30 ms |
| 層 2 つと出力の層 (part7) | 4.77 ms | 5.33 ms | 7.20 ms |
| 注意 (8192 の塊 1 つ) | 1.66 ms | 4.39 ms | 7.27 ms (同 −17 %) |
| mass | 1.03 ms | 2.71 ms | 4.59 ms |

  - 64 行でも重みを読む時間が大半で、8 行とほとんど変わらない。1 回に読む行を増やすほど 1 トークンあたりは安い
  - 出力の層は part7 − part1 で 1 回約 3 ms (64 行)。4K のプロンプトの 1 回 (約 73 ms) の 4 % ほど
  - Gemma 4 E2B の共有の層だけの部分 (part5、5 層、MLP は 2 倍幅) は Q=8 で 4.18 ms、Q=64 で 4.71 ms。16.80 のように共有の層を最後の行だけにしても、部分の中では 1 回 2〜3 ms しか減らない (下の「まだのこと」)
- プロンプトの読み (hwnpu-chat のサーバー、合言葉の試験、`-prompt-cache off -think=false`、スワップ 11〜12 GB。同じ長さを何度か測った):

| LFM3-1.7B | 4K | 16K | 64K |
|---|---|---|---|
| Q=64 (今までの既定) | 3.9〜5.4 s (3 回) | 19.3〜25.2 s (3 回) | 160.2 s |
| Q=128 | 3.3〜3.9 s (3 回) | 15.4〜16.1 s (3 回) | 136.3 s |
| Q=256 | 4.6 s | 16.9 s | 138.2 s |
| Q=128、最後の行だけ (今回の既定) | 3.7 s | 15.2 s | 145.5 s |

  - 128 で 2 割ほど速い。256 は 128 より速くならない (プログラムの読み込みも重い) ので 128 にした。64K の 145.5 s と 136.3 s の差は、日の中の揺れの範囲 (同じ Q=64 の 4K が 3.9〜5.4 s 揺れる)
  - 合言葉はすべて正しく、続けて頼んだ 100 語の文も書いた。生成の速さは変わらない (生成は 8 トークンのプログラム)

| 16K のプロンプト | Q=64 (今まで) | Q=128、最後の行だけ |
|---|---|---|
| Gemma 4 E2B | 31.2 s | **23.5 s (−25 %)** |
| LFM3-VL-2B | 21.6 s | **16.9 s (−22 %)** |

  - Gemma 4 は語彙 262144 で、出力の層の後の `reshape` (y) が CPU に置かれる (以前からの 1 つ。16.43)。そこを流れるのが 64 行から 1 行になる。どちらがどれだけ効いたかは分けて測っていない
- 実装:
  - `llm.Options.LastRow` (`hwnpu-convert -last-row`、`llm.Arch.LastRow`: LFM3 と Gemma 4): 入力 `last` (1, 1, Seq) の one-hot との行列積 (`blocks.Graph.LastRow`。gather は CPU に落ちるので) で最後の norm の前に 1 行を取り出し、y は (1, 1, vocab)。1 が 1 つの積なので、取り出した行はその行のまま
  - `decode`: y の行が 1 のプログラム (`rows`) では、`run` が `last` の v−1 番目に 1 を書き、`keep` は 0 行目を読む。下書きを確かめる `Speculate` はそういうプログラムを選ばない (行ごとのロジットが要る。8 トークンの生成のプログラムがある限り、そもそも選ばれない)
  - `chat.Prepare`: 注意で分けたモデル (`-paged`) の Neural Engine のプロンプトのプログラムは 1 回 `pagedPrefillSeq` = 128 (塊が小さいならその大きさ)、名前は `…-q128-last`。分けないモデルは、鍵の軸 (キャッシュ + 1 回のトークン) が 16384 を越えないように 64 のまま (`DefaultContext` = 16320 も変わらない)。Intel の NPU は実機で測っていないので 64 のまま、`LastRow` も付けない
  - 新しい名前なので、初回の起動で変換し直す (LFM3 で約 1 分、Gemma 4 E2B で約 2 分のコンパイルを含む)。今までの `…-q64` はもう読まれないので消してよい (LFM3 1.2 GiB、VL と E2B はそれぞれ約 1.6 GiB)
- テスト: `lfm3.TestPagedLastRow` (CPU のインタープリタで、最後の行だけのプログラムが 4、3、1、2 トークンの呼び出しで、キャッシュのないプログラムのロジットと一致すること。one-hot をプログラムの最後の行に置く改変は検出する)、`gemma4.TestPagedLastRow`、`chat.TestEngineReusePagedLastRow` (プロンプトの再利用、印、退避とその戻し)、`TestPreparePaged` と `TestPreparePrefill` (名前と 1 回のトークン数)
- まだのこと (16.83 で入れた): 16.80 を Neural Engine に移す (Gemma 4 の KV を共有する層を最後の行だけにする)。上のとおり部分の中の減りは小さい。効くのは、共有の層のうち全注意の 4 層 (E2B は 7 層のうち 19、24、29、34) のホストの注意を 1 行にする方で、16K で 1 回の 15 % ほどの見込み (128 行の注意は 8192 の塊 1 つで約 7 ms、8 行で約 1.7 ms)。decode の注意を層ごとに行の数を変えて回す (2 段の注意の選び直しと写しも) 必要がある

### 16.83 Neural Engine: Gemma 4 の KV を共有する層を最後の行だけにする (`-last-row`、decode の `attendOne`) — 2026-11-06

- 16.80 (AMD) を Neural Engine に移した。16.82 の「まだのこと」のとおり、部分の中を 1 行にするだけでは 1 回 2〜3 ms しか減らないので、共有の層の全注意 (E2B の 19、24、29、34) のホストの注意も 1 行にした
- 変換 (`gemma4`、`LastRow` で KV 共有のあるとき。`builder.lastFrom`): 最初の共有の層 (E2B は 15) に入る隠れ状態と層ごとの入力から、`last` の one-hot で最後の行を取り出し (`toLast`)、その後の層はすべて 1 行で組む。共有の窓の層は、マスクと RoPE の表もその行を取り出して使う (`lastRow4`。鍵は今までどおりキャッシュと 1 回のトークン全部)。ホストが注意を回す層は `attn_q_N` も `attn_o_N` も (1, heads, 1, head_dim)、それより後の部分の x は (1, 1, hidden)。E2B の q128 の 8 部分のうち、4 部分目 (14〜18) の途中から 1 行になる
- decode (`oneRow`、`attendOne`): `attn_q_N` が 1 行の層は、いちばん短いプログラム (生成の 8 トークンのもの) の注意のプログラムで、その行 0 を位置 pos+v−1 として回す。マスクはその位置のものを 1 回だけ作り (`stepMasks`)、読む塊はその位置までの分。2 段の注意も同じ位置で選び、読む (`attendSparse` と `tailMask` に位置を渡すようにした。今までは s.pos 決め打ち)。1 行の層の後にまた全部の行の層が来るプログラムは、`checkPaged` が断る
- 実機 (M3、hwnpu-chat のサーバー、合言葉の試験、スワップ 11〜12 GB):

| Gemma 4 E2B | main (Q=64) | 16.82 (Q=128、出力の層だけ最後の行) | 今回 |
|---|---|---|---|
| 16K のプロンプト | 27.8 s (16.82 の日は 31.2 s) | 23.5 s | **12.6〜13.2 s (2 回)** |
| 64K のプロンプト | 205.2 s | — | **94.6 s** |
| その後の 128 トークン | 65.2 / 93.3 ms/token (16K / 64K) | | 63.0〜65.2 / 90.8 ms/token |
| プロンプトの後の footprint | 865 / 1805 MB | | 578〜594 / 1392 MB |

  - 合言葉はどれも正しく、続けて頼んだ浜辺の文は main と同じ書き出し。生成の速さは変わらない (生成は 8 トークンのプログラムで、全部の行を計算する)
  - 配置: 1 行になった部分 (part4〜7) はすべて NPU。CPU に置かれるのは以前からの part8 の y の `reshape` 1 つだけで、そこを流れるのは 1 行
  - 2.1〜2.2 倍で、AMD (16.80、2.3 倍) に近い。共有の層の MLP (2 倍幅) と、全注意 7 層のうち 4 層の注意が 1 行になる
- E4B は既定で 1 本のプログラム (`-prefill auto`、16.65) なので、この形にはならない。`-prefill on` なら同じ変換 (共有の層は 42 層のうち 24〜41、その全注意は 29、35、41) になる。測った (同じ日、同じ版):

| Gemma 4 E4B | 既定 (1 本、8 トークンで読む) | `-prefill on` (2 本、128 トークン、共有の層は最後の行) |
|---|---|---|
| 16K のプロンプト | 184.3 s | **25.4 s、24.3 s (7.5 倍)** |
| 64K のプロンプト | 1382.8 s | **168.4 s (8.2 倍)** |
| 読んだ直後の最初の 12 トークン | 0.7 s (16K)、1.5 s (64K) | 3.0〜3.2 s、13.8 s |
| その後の 128 トークン | 112.0 ms/token (16K)、236.2 (64K) | 129.9〜136.5、210.9 |
| 起動 (2 回目から) | 86 s (1 本) | 2 本で 135〜165 s (82.6 + 82.6、78.1 + 57.3 s) |
| プロンプトの後の footprint | 742 MB (16K)、2422 MB (64K) | 847〜856 MB、2813 MB |

  - 合言葉はどれも正しい。配置は 2 本とも以前からの part8 の `reshape` 1 つだけが CPU
  - 2 本で 6.5 GiB は Neural Engine のコンパイルのキャッシュ (約 4 GiB) に入らないので、起動は毎回コンパイルし直し、プロンプトから生成に移った最初の呼び出しで重みを載せ直す (読んだ直後の数トークンが遅い。16.65 と同じ)。16K の生成は 2 割遅く、64K では逆に 1 割速い (64K では注意が大半で、重みの載せ直しの差が見えなくなると見ている。1 回ずつなので揺れの範囲かもしれない)
  - 16.65 の 2 本 (Q=64) は 16K を 41.6 s で読んだ。今回はその 0.6 倍
  - 既定 (`-prefill auto` が 1 本を選ぶ) は変えていない。数千トークン以上の文書を読むなら `-prefill on` が速い。どちらを既定にするかは、起動と短い会話の速さとの兼ね合いで未決 (improvements.md 11.1)
- 名前は 16.82 と同じ `…-q128-last`。16.82 の版で変換した Gemma 4 の `-q128-last` (共有の層も全部の行) は、この版でも読めて正しく動くが速くならない。消せば変換し直す
- テスト: `gemma4.TestPagedLastRow` (2 部分目の y が (1, 1, hidden)、4、3、1、2 トークンの呼び出しでキャッシュのないプログラムと一致)、`TestPagedLastRowSparse` (200 トークンを 4、1、3 ずつ、2 段の注意で全部を読む形との相対誤差 0.063)。改変 (問いの位置を s.pos にする、2 段の注意の位置を s.pos にする、マスクの最初の行を取る) はすべて検出する (相対誤差 0.47、0.12、0.79)

### 16.84 AMD: スライディング窓の層は窓の中のキーだけを読む (`iron_window.py`)、プロンプトの塊の前に空のコマンド — 2026-11-06

- 動機: IRON の `KVCache` は、スライディング窓の層をリングにするのは `3 × リング < max_seq_len` のときだけで、Gemma 4 (窓 512) の文脈 4096 ではリングにならない。そのとき `MHA` は問いのブロックごとに位置 0 から今の位置までのキーを全部流す。mha.cc は窓の外のブロックの計算を飛ばすが、読み込みとループは飛ばさない。位置 3072 からの 1024 トークンの塊は、窓に要るのが 1536 個なのに、問いのブロックごとに 4096 個のキーを読んでいた。16.80 の後でも、キャッシュを持つ 15 層のうち 12 層がスライディング窓
- `tools/amd/iron_window.py` (IRON の `KVCache` の `__init__` と `attend` を差し替える。`iron_export_lm.py` は既定で入れ、`--scan-window` で IRON のままにする):
  - プロンプトの塊: IRON のリングの形 (`KVCache._ring`) と同じ形にした。`start + chunk` で終わる `chunk + window` 個 (64 の倍数に切り上げ、E2B は 1536) のキーと値を、`Copy` (呼び出しごとのオフセット、キャッシュの書き込みと同じ仕組み) で作業用の状態 (層で共有) に写し、IRON の `MHA` をその写しに掛ける。問いは塊の `chunk` 行のまま境界を付けず、出力から呼び出しの行だけを切る。オフセットと長さは `start` だけの式。最後の塊が文脈の終わりから 1 塊以内で始まっても写しがキャッシュの中に収まるよう、窓の層のキャッシュは `chunk` 行多い (E2B で約 12 MB)
  - デコードのステップ: `position` で終わる `window + 1` 個 (E2B は 512) を、その場で読む。IRON の `MHA` に K と V の DMA のオフセット (`kv_offset`) を足した派生クラス (名前は `MHA` のまま: 書き出しと Go の `iron.Derived` が MHA の派生ワードをその名前で計算する)
  - MHA は読むキーに対する相対の位置でマスクするので (問いはキーの最後の行)、注意は同じ。キャッシュは位置で引くままなので、どの位置からでもやり直せる (リングにしない)
- 試すための道具: `tools/amd/iron_window_probe.py` (Gemma 4 の窓の層 1 つ分の注意のグラフ。`irongraph` で IRON の参照と照らす)。窓の層 1 つの注意 (Ryzen AI 9 365、位置 3072〜4095 の 1024 行): IRON のまま 112 ms、窓だけ 66〜67 ms、コサイン 0.99999。デコードの 1 ステップ (位置 4000): 0.32 → 0.15 ms
- 止まる問題と `chunkBarrier` (Go、`internal/llm/iron/model.go`): 窓の形を入れた書き出しは、2 つ目のプロンプトの塊の最初のコマンドが amdxdna の打ち切り (`ERT_CMD_STATE_TIMEOUT`、60 秒) まで止まった (7 回。毎回 NPU は戻った)。1 つ目の塊だけなら答えも合う。調べたこと:
  - 演算子は単体で正しい (上の probe)。書き出しを xclbin をステップごとに (`--per-step`) にすると、2600 トークンの照合が通る (最上位一致、コサイン 0.99993)。止まるのは full ELF のときだけ
  - 塊の間に 200 ms 待っても止まる。デコードのステップ (別のハードウェアコンテキスト) を挟むと止まらない (`ironlm -depth`)。16.80 の「次の塊の最初のコマンドが止まる」と同じ現象
  - 言葉 (ワード) の値は IRON のままの書き出しと同じ。コマンドの切り方 (IRON のままの書き出しに同じ切れ目を足しても通る)、バンクの中の位置、PDI の数、アドレスのワードの数、ワードの抜けではない。原因は分かっていない
  - 回避: Go がプロンプトの塊の前に毎回、デコードの ELF の空のインスタンス (mlir-aie の `aiecc_empty`、制御コード 16 バイト、PDI なし) をデコードのハードウェアコンテキストで走らせる (`chunkBarrier`、0.3〜0.5 ms。run は 1 つ作って持つ)。これで止まらない。空のインスタンスを持たない書き出し (xclbin をステップごとに) では何もしない。すべての書き出しで走る。IRON のままの Gemma 4 E2B (16.80 の書き出し) と LFM3-1.7B bf16 の照合はそのまま通り、LFM3 4 ビットは文脈 12288 まで塊を続けて読めた (速さは比べていない)
  - あわせて `iron_gemma4.install_tail_boundary`: 共有の層 (16.80 の最後の行) の始まりで必ずコマンドを切る。窓の形はワードを増やし、IRON の切れ目が動いて、頭の draw だけが別のコマンドになっていた。16.80 で止まった切れ方を避け、測った書き出しと同じく共有の層、頭、draw を 1 つのコマンドに置く
- 照合 (Ryzen AI 9 365、16.80 と同じ IRON の prompt.txt の 2600 トークンと貪欲法の 3 ステップを、IRON のままの書き出しの記録と):

| 書き出し | プロンプトのロジット | その後のステップ | 2600 トークン |
|---|---|---|---|
| Gemma 4 E2B bf16 (16.80 → 今回) | 最上位一致、コサイン 0.99992 → 0.99997 | 最上位一致、0.99999 | 4.51 → 4.36 秒 |
| Gemma 4 E4B bf16 | 最上位一致、0.99968 | 最上位一致、0.99993 以上 | 7.84 → 7.39 秒 |
| PLaMo 3 q4 (6000 トークン、今までの書き出しの記録と) | 最上位一致、0.99973 | 最上位一致、0.9989 以上 | 38.4 → 36.1 秒 |

- 速さと電力量 (`ironlm -depth`。デコードのステップを塊の間に挟む。ほかに NPU を使うものがないときに、2 つを続けて、順を入れ替えてもう一度測った。2 回目の数字):

| Gemma 4 E2B bf16、1024 トークンの塊 | 16.80 (最後の行だけ) | 今回 (+ 窓) |
|---|---|---|
| 位置 0 から | 1.11 秒、16.0 J | 1.11 秒、14.2 J |
| 1024 から | 1.73 秒、20.8 J | 1.63 秒、18.9 J |
| 2048 から | 2.24 秒、25.1 J | 1.92 秒、21.7 J |
| 3072 から | 2.76 秒、30.5 J | **2.21 秒、24.7 J (−20 %、−19 %)** |
| 4K 全体 | 7.84 秒、92 J | **6.87 秒、80 J (−12 %、−14 %)** |
| デコード (位置 4095) | 172.3 ms | 168.8 ms (−2 %) |

| PLaMo 3 q4 (文脈 8192、窓 2048)、1024 トークンの塊 | 今まで | 今回 |
|---|---|---|
| 位置 0 / 2048 から | 2.48 / 6.81 秒 | 2.46 / 6.79 秒 |
| 4096 / 6144 から | 8.39 / 9.97 秒 | 7.42 / 8.08 秒 |
| 7168 から | 10.75 秒 | **8.39 秒 (−22 %)** |
| 8K 全体 | 60.3 秒 | **53.1 秒 (−12 %)** |
| デコード (位置 8191) | 106.6 ms | 92.2 ms (−14 %) |

  - 窓より手前の塊は変わらず、窓を越えた分だけ速くなる。塊の時間がまだ位置で伸びるのは、全体を見る層 (E2B は 15 層のうち 3 層、PLaMo 3 は 8 層に 1 層) と、共有の層の最後の行 (E2B)
- 答え: Gemma 4 E2B は「日本の首都は東京です。」と「391」。画像と音声つき (`gemma4mm`) は白地に赤い円と青い四角の画像に「この画像には、左側に赤い円、右側に青い四角が描かれています。」「赤い円の方が青い四角よりも大きく見えます。」、ALSA の試験音に「Front, left.」(日本語で書き起こしを頼むと「フロント、レフト、右」。16.80 の書き出しも同じ文)
- 書き出し (文脈 4096、塊 1024): E2B 約 6 分、E2B 画像と音声つき 10 分 24 秒、E4B 8 分 14 秒、PLaMo 3 q4 4 分 27 秒
- 未確認: 4 ビットの Gemma 4 (E2B、E4B) の書き出し。プロンプトの形をその場で読むもの (デコードと同じ形) に `chunkBarrier` と合わせて戻すこと。止まる原因

### 16.85 Anima を M3 の Neural Engine で (`hwnpu-image`、Core ML) — 2026-11-06

- 16.69 の書き出しを M3 (16 GB) で初めて動かした。そのままでは 3 つの部品が Neural Engine に載らず、読み込みで止まった
  - `text` (fp16 で 1.0 GB): 1 つのプログラムの上限を超え、3202 演算がすべて CPU。lut8 (538 MB) か int8 なら全部 Neural Engine
  - `dit_time` (入力 (1, 1, 2048)): 15 演算がすべて CPU。8 行以上 (8、16、64 で確認) なら全部 Neural Engine。入力を `mil.TimeRows` (8) 行にし、ステップの正弦波を 0 行目に、残りを 0 にする。manifest の `time_rows`
  - `dit_patch` ((1, n, 64) から 2048 への linear 1 つ): 512 × 512 と 1024 × 1024 のどちらでも 2 演算とも CPU。最初のブロックの群 (`dit_blocks_0`) の頭に入れると 148 演算すべて Neural Engine。`dit_blocks_0` はパッチ (1, n, 64) を受ける。前の書き出し (manifest の `patch`、`time_rows` なし) もそのまま読む
- 起動のキャッシュ: `hwnpu-image` はコンパイルのキャッシュを指定していなかった。Core ML は毎回別の一時パスにパッケージをコンパイルするので Neural Engine のバンドルも使い回されず、2 回目以降も DiT 全体をコンパイルし直していた (512 × 512 で読み込み 118 秒。e5bundlecache に 1 回 4 GB ずつ溜まった)。`-cache` (既定は hwnpu-chat と同じユーザーのキャッシュ) を足し、118 → 4.4 秒
- 1 ブロック (1024 × 1024、fp16、単独のプログラム、4 回の最小) の内訳と、問い合わせの塊 (`-qchunk`):

| 形 | 1 回 |
|---|---|
| そのまま (塊 1024) | 284 ms |
| 自己注意なし / MLP なし / 交差注意なし | 127 / 236 / 270 ms |
| 尺度を q の norm の重みに畳む / norm の前の `maxScaled` なし | 285 / 279 ms |
| 塊 4096 / 2048 / 512 / 256 / 128 / 64 | 356 / 291 / **215** / 214 / 221 / 237 ms |
| 塊 512、lut8 | 214 ms |

  - 自己注意が 55 %。行列積の分 (約 34 GMAC) より、16 × 1024 × 4096 の点数の softmax などの要素ごとの演算が重い。塊を小さくすると点数の行列が小さくなり速い。既定を 512 にした (256 は同じ速さでコンパイルが長い)
  - 計算で詰まっているので、重みを 8 ビットにしても遅くならない
- VAE: 拡大を reshape と tile から `upsample_nearest_neighbor` にした。段 5 (192 チャンネルの 512² から 96 の 1024²) を単独で 1275 → 412 ms (相対誤差 4.8e-4)。CPU のインタプリタに同じ演算を足し、OpenVINO IR には今までと同じ reshape と tile で下ろす (`ovlower`。この機械に OpenVINO がないので IR を書くところまで)。サブピクセルの畳み込み (2 × 2 の位相に畳んで `depth_to_space`) も同じ速さだったが、重みの並べ方を合わせきれていない
- VAE の段は、Core ML でも走るときだけ読む (AMD の 16.72 と同じ)。起動時に全段を 1 回読んで配置を確かめ (キャッシュにも入る)、閉じる。パイプラインの中では段 5 が 9 秒、同じプロセスで 2 回目のデコードは全段が遅くなっていた (冷えていても 14.5 → 28 秒)。全部の部品を持つプロセスは最大 6.9 GB で、メモリの逼迫と見ている。段ごとに読むと VAE 全体が 3.2〜4.0 秒になった
- 重みの形式 (1024 × 1024、同じシード、テキストは lut8。fp16 の DiT の最後の潜在とのコサイン): lut8 0.858、int8 0.902 (lut8 と int8 どうしは 0.892)。絵はどれも構図が同じで、服や耳飾りの細部が違う。テキストの条件づけ (fp16 を CPU のインタプリタで、とのコサイン): lut8 0.9943、int8 0.9965。Core ML の既定をテキストも DiT も int8 にした。この機械に torch がないので、diffusers の fp32 との比較はしていない
- 結果 (M3、1024 × 1024、8 ステップ、キャッシュあり):

| | 手直しの前 (動く形にしただけ) | 後 (既定の int8) |
|---|---|---|
| 読み込み | 98〜191 秒 (毎回コンパイル) | 2.8 秒 |
| 1 ステップ | 6.5〜7.4 秒 | 4.2〜4.4 秒 |
| VAE | 14.5〜18.8 秒 | 3.7 秒 |
| 全体 | 267 秒 | **40.4 秒** |
| プロセスの最大 (footprint) | 6.9 GB | 3.8 GB |
| 書き出し | 4.6 GB (DiT fp16) | 2.8 GB |

  - 512 × 512 は 1 ステップ 0.8 秒、全体 13 秒 (塊 1024、DiT fp16 のとき)。AMD (16.68) は 1024 × 1024 で 1 ステップ 7.2 秒、全体約 70 秒
  - 1 回目の読み込み (コンパイル) は 1024 × 1024 で 3〜5 分。書き出しの変種をいくつもコンパイルすると Neural Engine のキャッシュから追い出され、2 回目が 14 秒になることがあった
- 自己注意の中身 (注意の計算だけのプログラム、q、k、v は (1, 16, 4096, 128)、同じ注意を 4 回つないで入出力を薄めた 1 回あたり):

| 形 | 1 回 |
|---|---|
| 今の形 (`SDPAScale`、問い合わせ 512 ずつ) | 101 ms |
| 問い合わせ 1024 / 256 ずつ | 169 / 100 ms |
| 尺度を点数でなく q に掛ける | 103 ms |
| Core ML の `scaled_dot_product_attention` (切らない / 512 / 1024 ずつ) | 192 / 110 / 172 ms |
| softmax を exp、`reduce_sum`、`real_div` に分ける (1024 / 512 ずつ) | 181 / 104 ms |
| ヘッドで切る (4 / 1 ずつ) | 118 / 113 ms |
| ヘッドと問い合わせで切る (4 × 1024、8 × 512、2 × 2048、4 × 512、1 × 512) | 96〜105 ms |
| ヘッド 2 × 問い合わせ 256 / 4 × 128 | **90** / 99 ms (コンパイルが注意 4 回で 84 / 141 秒) |
| チャンネルを先にした並べ方 ((1, 16, 128, n)、softmax を鍵の軸で。512 / 1024 ずつ) | 122 / 179 ms |
| 行列積 2 つだけ (softmax なし) / 点数の行列積と `reduce_max` だけ | 55 / 53 ms |
| softmax の行の最大を引かない | 87 ms (fp16 であふれる) |

  - 上の表の値は、1 回の実行の受け渡し (float32 と fp16 の変換、約 140 ms) を 4 で割った約 36 ms を含む。形どうしの差は正しい。中身だけの時間は、つなぐ回数を 4 と 12 にした差を 8 で割って測った (下の表)
  - 点数の行列の大きさ (16 × 4096 × 4096 = 2.7 億要素) で決まっている。行列積は計算で詰まっていない。点数の数に比例する速さ (q kᵀ で毎秒約 160 億、softmax で毎秒約 80 億) で決まる。問い合わせを 512 以下に切ると頭打ちになり、ヘッドでも切ると少しだけ (−11 %) 速くなるが、コンパイルの時間がプログラムの大きさとともに伸び、28 ブロックでは使えない
  - 行の最大を省けるかを重みから見積もった: RMS norm の後の q と k は |q·k|/√128 ≤ √128 · max|w_q| · max|w_k| (RoPE は長さを変えない)。自己注意で層ごとに 2〜221、交差注意で 2〜16。fp16 の exp があふれる 11 を多くの層で超えるので、上限からは省けない
- 点数の行列は Neural Engine の内部に収まらないから遅いのか (つなぐ回数の差で測った、中身だけの 1 回):

| q、k、v (1, 16, 4096, 128)、塊 512 | 中身 |
|---|---|
| 受け渡しだけ (`mul` 1 つ) | 0.65 ms |
| q kᵀ と `reduce_max` | 17.2 ms |
| それに softmax を足す | 50.2 ms |
| さらに v との積を足す (注意の全体) | 66.3 ms |

| q kᵀ の内側の次元 D | 32 | 128 | 512 |
|---|---|---|---|
| 中身 | 16.1 ms | 17.2 ms | 29.2 ms |

| 1 つの点数のテンソル | q kᵀ (点数 / 秒) | softmax まで (16 ヘッドあたり) |
|---|---|---|
| 0.5〜2 MB (1 ヘッド、塊 64〜256) | 誤差の範囲 | 49〜51 ms |
| 4〜64 MiB (1 ヘッド、塊 512〜4096、16 ヘッド、塊 256〜512) | 毎秒 150〜160 億 | 50〜65 ms |
| 65 MiB (8 ヘッド、鍵 4160、塊 1024) | 146 億 | |
| 68〜112 MiB | 50〜76 億 | |
| 128 MiB (16 × 1024、8 × 2048、4 × 4096 のどれも) | 46 億 | 130〜151 ms |

  - D を 32 から 128 にしても q kᵀ の時間はほぼ同じなので、D が 128 のときは行列積の計算ではなく点数の数で決まる (512 で初めて計算が効く)
  - 1 つの点数のテンソルが 0.5 MB でも 64 MiB でも、点数あたりの速さはほぼ同じ (softmax だけ 1 MB 以下で約 2 割速い)。数 MB の内部のメモリに収まるかどうかでの段差は見えないので、「内部に収まらないから遅い」という見立ては当たっていない
  - 段差は 65 MiB と 68 MiB の間に 1 つある。それを超えると、形 (ヘッドの数か塊の大きさか) によらず 2〜3 倍遅くなる。1 つのテンソルの大きさに上限 (64 MiB 前後) があり、超えると別の分け方になると見ている (中は見えない)。塊 1024 (128 MiB) が 512 (64 MiB) より遅かったのはこれ。既定の塊 512 は、1024 × 1024 でこの上限のすぐ下。512 × 512 (トークン 1024) なら塊 1024 でも 16 MiB
  - これを受けて、DiT の塊は、自己注意の点数のテンソル (16 × 塊 × パッチの数 × 2 バイト) が 64 MiB を超えないところまで、変換のときに半分ずつ小さくする (`mil.DitQChunk`)。1024 × 1024 は 512 のまま、1280 × 1280 は 256、2048 × 2048 は 128。VAE の注意 (1 ヘッド) は 1024 × 1024 でも塊 512 で 16 MiB なので変えない
  - Core ML の公開された演算の範囲では、自己注意の残りの改善は小さい。残っているのは、トークンの数そのものを減らす (近似になる) ことくらい
- 新しい形 (時刻の 8 行、パッチを `dit_blocks_0` に、`upsample_nearest_neighbor`) の書き出し (256 × 256、既定の int8) を、CPU のインタプリタ (fp32、`-cpu`) と Neural Engine で同じシードから: 最後の潜在のコサイン 0.934、絵は同じ構図 (CPU は 1 ステップ 49 秒、全体 445 秒)。同じ重みでの fp32 と Neural Engine の fp16 の差で、形の誤りなら構図から崩れる
- 書き出しは hwnpu-chat と同じく自動にした (`anima.Prepare`)。`-model` を省くと、見つけた NPU の形式 (`llm.ProgramFormat`) で、`-dir/Anima` (既定はユーザーのキャッシュディレクトリの `hwnpu/models/Anima`) に足りないチェックポイント (公式の split files の DiT と Qwen3-0.6B、Diffusers 版の VAE とその config、トークナイザー 2 つ、計約 5.4 GB) を `chat.Fetch` で落とし、足りない書き出しを作る。書き出しの名前には大きさ、形式、群、塊と書き出しの形の版 (`anima.ExportVersion`、今は 2) が入り (`anima-turbo-v1.1-1024x1024-int8-int8-g4-q512-v2.coreml`)、最後に `.done` を書く。形を変えたら版を上げれば、前の書き出しを読まずに作り直す。Apple と Intel の NPU が見つからず、`~/models/anima-turbo-1024.amd-npu2` があれば今までどおりそれを使う (AMD の書き出しは Python の `iron_anima.py`)。`-convert` は手で書き出す手段として残した
  - M3 で: 足りないトークナイザー 1 つを落とし、書き出し 19 秒、初回の読み込み (コンパイル) 199 秒、1 枚 238 秒。出力は手で書き出したものとビット単位で同じ。手で書き出したものも残していると、2 回目の読み込みが 58 秒だった (同じ中身でも別のパッケージとしてコンパイルされ、Neural Engine のキャッシュの約 4 GiB を超える)。それを消すと 2.7 秒、1 枚 40 秒
- 未確認: Intel の NPU の実機。新しい形を diffusers と照合し直すこと (この Mac に torch がない)

### 16.86 端末の表示: ダウンロード、書き出し、コンパイル、ステップの進み具合 (`console.Bar`) — 2026-11-07

- `console.Bar`: 端末では、ラベル、スピナー、バー、割合、経過時間、詳しい情報を 1 秒に 10 回描き直し、`Done` でラベルと結果の 1 行を残す。割合が分からないあいだはスピナーと経過時間だけ。端末でなければ (パイプ、ファイル) `Done` の 1 行だけを書き、描き直しもエスケープシーケンスも出さない。`console.Log` の灰色の中で使う (「all on the NPU」は緑、「NOT all on the NPU」は赤のまま)
- ダウンロード (`chat.Fetch`、hwnpu-chat と hwnpu-agent と hwnpu-image で共通): バーに大きさ、速さ、残り時間 (`downloading model.safetensors from LiquidAI/LFM3-1.7B ▕███░░░▏ 14% 15s 0.47 / 3.46 GB 31.4 MB/s, 1m35s left`)。終わると大きさ、時間、置き場所の 1 行。前は 1 秒ごとに `\r` で数字を書き直すだけで、パイプにもそれを書いていた
- hwnpu-chat の読み込み: 部品ごとの行に数え (`[3/9]`) を足した (前の `dots` を `Bar` に)
- hwnpu-image: ログを灰色に。書き出しと読み込み (初回は Neural Engine 向けのコンパイル) は部品ごとに数えとスピナー、読み込みは全部 Neural Engine に載ったかも。ノイズ除去はステップのバー (1 ステップの時間と残り時間)、VAE は段のバー。端末でなければ、今までどおりステップと段ごとの行。VAE の段の時間はログの行ではなく `Pipeline.Stage` (`StageFunc`) で受ける (AMD の `VAE.Stage` も)
- `anima.Open` は書き出しの部品ごとの行を書く先 (`log`) を取る
- 描き直しが行を残していた (利用者の報告): ダウンロードの行はファイル名とリポジトリ名だけで 90 桁を超え、端末の幅を超えると折り返す。`\r` は折り返した最後の行の先頭にしか戻らないので、描き直すたびに前の行が残っていた。描き直す行 (`Bar` と `Spin`) は端末の幅 (TIOCGWINSZ) から 1 桁引いた幅に切り、足りなければ先にラベルを `…` で縮めてバーと割合を残す (幅 60 で `downloading dif… ⠹ ▕████░░…▏  20% 0.2s  11`)。さらに描くあいだだけ端末の自動の折り返しを止める (DECAWM、`\x1b[?7l` … `\x1b[?7h`)。幅の数え違い (東アジアの曖昧な幅の `█` `░` を 2 桁に描く端末や、描いたあとに狭めた窓) でも、行は右端で切れて折り返さない。`Done` の結果の行は 1 回しか書かないので、縮めずにラベルの全部を残す。確認: 疑似端末を 60、100、200 桁にして、描き直しがどれも幅に収まること

### 16.87 4 ビットの符号を活性を見て選ぶ (GPTQ、`tools/amd/q4_gptq.py`) — 2026-11-06

- 動機: 4 ビットの精度の落ち方がモデルで大きく違う (AMD の q4 の KL は E2B 0.097、LFM3 0.036)。今の量子化は重みを 1 つずつ最も近い段に丸める (RTN) だけで、層の出力の誤差は見ていない (model-fit.md T8)。形式を変えずに符号の選び方だけを変えれば、実行時の費用は 0
- `tools/amd/q4_gptq.py`: Hugging Face のモデルを CPU (fp32) で動かし、量子化する射影の入力の H = E[x xᵀ] を校正の文 (512 トークン × 64 本) で集め、GPTQ で符号を選ぶ。q4nx の 32 個ごとの scale と min は、GPTQ がその群に着いたときの (誤差を受け取った後の) 重みに `iron_q4._fit` で合わせる。層を 8〜9 個ずつ順に校正し、後の層は前の層を量子化した入力で校正する。bf16 の重みとの KL を評価の文で測る
- 書き出しへの渡し方: 符号を重みごとに q4nx のブロックで書き、名前を重みの形と bf16 の値のハッシュにした。`iron_export_lm.py --q4-codes DIR` で `iron_q4.quantize_fmt` が量子化の直前にハッシュで引く。モデルごとの名前の対応 (HF の `feed_forward.w1` と IRON の `mlp.gate_proj`、融合した重み) を書かずに済み、Gemma 4、LFM3、PLaMo 3 の書き出しが同じ道を通る
- 失敗と手当て:
  - この repo の文書と Go だけ (8K トークン) で校正すると、LFM3 の README の KL は 0.031 → 0.017 だが、校正にない GPL-3 は 0.105 → 0.129 と RTN より悪くなった。文頭 (BOS、外れ値の活性) を H から外しても直らない (0.139)。一般の英語と日本語 (wikitext-2、wiki40b-ja) を足しても 0.115
  - H の減衰 (対角に足す量) を GPTQ の論文の平均の 0.01 倍から 0.1 倍にすると、GPL も 0.091 と RTN より良くなり、ほかの文も良くなった。0.01 は校正の文に合わせすぎる。既定を 0.1 に
- 結果 (amd-npu-models.md 13 節の GPTQ): NPU の bf16 との KL
  - Gemma 4 E2B: README 0.120 → 0.092、GPL 0.161 → 0.140、日本語の Wikipedia 0.221 → 0.081 (最上位の一致 79 → 88 %)
  - LFM3-1.7B: README 0.032 → 0.017、GPL 0.110 → 0.092、日本語の Wikipedia 0.071 → 0.038
  - NPU の値は CPU の見積もりと同じ。速さとメモリは変わらない (同じ形式)。E2B は bf16 と同じ答え (bf16 の誤りまで同じ)
- Apple の lutN (16 行ごとに 1 つの Uniform の表) にも同じ手法を: 表は今の当てはめのまま固定し、GPTQ で各重みが表のどの段を取るかを選ぶ (誤差を後の列に送る)。`q4_gptq.py --fmt-q4 lut6 --fmt-q8 lut6 --head lut6` が添字と表を書き、`hwnpu-convert -codes DIR` が重みのハッシュで引く (`internal/palette.Codes`、`blocks.Graph.matrix`)。LFM3-1.7B (すべての射影と出力の行列を lutN、CPU の fp32 で KL):

| | README | GPL-3 | wiki40b-ja | wikitext-2 | Go |
|---|---|---|---|---|---|
| lut6 (今の当てはめ) | 0.070 | 0.232 | 0.112 | 0.075 | 0.067 |
| **lut6 + GPTQ** | **0.027** | **0.152** | **0.068** | **0.036** | **0.024** |
| lut4 (今の当てはめ) | 2.73 | 1.84 | 2.64 | 2.02 | 2.38 |
| lut4 + GPTQ | 0.66 | 1.42 | 1.23 | 0.83 | 0.58 |

  - lut6 は GPTQ で誤差が半分になり、AMD の q4 を丸めたとき (0.031 / 0.105 / 0.063) に近づく。lut4 は 4 分の 1 になるが、まだ使える水準ではない (表が 16 行 × 2048 列に 16 段しかない)
  - Go の変換 (Linux、`hwnpu-check -package` の fp32 の CPU で、BOS つきの wikitext-2 の 128 トークン): lut6 の KL 0.074 → 0.050、最上位の一致 105 → 111/128。143 個の重みすべてに表が当たる (注意のヘッドごとのゲート、16 行は当てはめのまま)。M3 の ANE での確認と速さは未確認 (形式は同じなので速さは変わらないはず)
- あわせて直したもの (16.84 の残り): `iron_window` の窓の層の KV キャッシュは `max_seq_len + chunk` 行あり、Go の `caches()` は `max_seq_len` で割っていた。LFM3 (文脈 12288、塊 2048) の新しい書き出しは割り切れずに開けず、Gemma 4 (4096 + 1024 行) は割り切れて 1 位置の行を 640 バイト (正しくは 512) と誤る (会話の退避と戻しの行がずれる)。書き出しが `cache_rows` (バッファごとの行数) を書き、Go はそれで割る。16.84 の後に作った Gemma 4 の書き出しは書き出し直すまで古い計算のまま
- 続き (2026-11-07):
  - **表そのものの当てはめ** (model-fit.md T8 の 1): lutN の表を、入力のチャンネルごとの E[x²] (H の対角) で重みを付けた k-means (一様な表から Lloyd を 8 回) で当てはめ直してから、GPTQ で段を選ぶ (`q4_gptq.py --lut-fit wkmeans`)。LFM3-1.7B の CPU の KL は lut6 で README 0.027 → 0.021、GPL 0.152 → 0.118、lut4 で 0.66 → 0.17、1.42 → 0.58。Go の変換 (`hwnpu-check -package`) では lut6 0.050 → **0.038** (最上位の一致 112/128)、lut4 0.66 → **0.33** (94/128)。lut4 は lut6 の丸め (0.074) の 4 倍でまだ既定にはできない。ANE での確認は docs/handoff-m3-gptq.md
  - E4B、PLaMo 3、LFM3-VL / Audio / Omni を校正して NPU で測った (amd-npu-models.md 13 節の「続き」)。LFM3 の多モーダルは文の部分の重みが LFM3-1.7B と違い、それぞれ校正が要る。NPU の bf16 との wiki40b-ja の KL は E4B 0.083 → 0.049、VL 0.067 → 0.034、Audio 0.046 → 0.022、Omni 0.048 → 0.024、PLaMo 3 0.058 → 0.040。README と GPL もほぼ良くなる (Audio の GPL だけ 0.154 → 0.161)。PLaMo 3 は融合した `gate_up_proj` もハッシュで当たる
  - Anima の DiT: 校正の道具 `tools/amd/anima_gptq.py` (参照の軌跡の上で DiT のブロックの射影の入力を集める) と `iron_anima.py --q4-codes`。512×512 の CPU で、q4nx の丸めの速度の誤差 0.123、潜在の cosine 0.868 が GPTQ で 0.071、0.939 (q8 は 0.013、0.962)。既定は q8 のまま
  - ほかのモデルを入れるときの手順を docs/new-model-checklist.md (日本語は .ja.md) にまとめた

### 16.88 Qwen3.5-4B を M3 の Neural Engine で: lut8 にし、silu と softplus を書き直し、GDN の出力を大きくする — 2026-11-07

- 16.62 の Qwen3.5-4B を M3 (16 GB) で初めて動かした。回線が速くなり (約 20 MB/s)、9.3 GB を 4 分で落とせた
- 配置と起動: 注意で分けた 9 部分 (lut6 で 0.26〜0.50 GiB、lut8 で 0.34〜0.68 GiB) がすべて Neural Engine。`-prefill auto` は 2 本 (Q=8 と Q=64) だと重みが 5.9 GiB (lut8 で 7.9 GiB) で ANE のキャッシュ (約 4 GiB) を超えるとみて、Q=8 の 1 本で読む。初回のコンパイルは 2.5〜4 分、2 回目からの読み込みは 2.3〜4.5 秒 (キャッシュに残る)。形式を切り替えると前の形式が追い出され、戻すと 1.5〜2 分のコンパイルになる。Q=1 と Q=64 の組 (`-draft 0`、lut8 で 7.9 GiB) はコンパイルが 24 分終わらなかったので止めた
- 32 層全部の照合 (16.62 の残り): 16 GB では fp32 の 4B (約 17 GB) を読めないので、層を 1 つずつ fp32 で読んでは捨てる参照 (`tools/ref/ref_qwen35_stream.py`、PyTorch 2.15 と transformers 5.22、Qwen3_5ForCausalLM を meta で作り、層の前のフックで重みを入れ、後のフックで meta に戻す) を書いた。英文 30 トークンの forward は 12.5 秒。深い層の活性は最大 121 で、fp16 はあふれない。`hwnpu-check` (CPU のインタプリタ、`-f16`) の全体の比べ:

| 形式 | 日本語のチャット (26 トークン) | 英文 (30 トークン) |
|---|---|---|
| fp16 | KL 0.0000、argmax 26/26 | KL 0.0001、30/30 |
| lut8 | KL 0.0015、26/26 | KL 0.0050、29/30 |
| lut6 | KL 0.031、24/26 | KL 0.135、21/30 |

- lut6 は LFM3 の系列 (16.65) と同じく足りない: 富士山の質問に「富士山ではなく…」と崩れた。**既定を lut8 にした** (`qwen35.Arch.Format`)。Intel は今までどおり int8
- **Neural Engine の上の誤り**: lut8 でも、ANE で動かすと日本語の質問の書き出しは合うが、長いプロンプト (3K と 13K トークンの文書の 10%、50%、90% に合言葉) では 3 つのうち 1 つしか答えず、「原文には多くの誤字や乱码が含まれています」と言った。PyTorch の fp32 (同じ層ごとの参照で、キャッシュつきで生成) は 13K で 3 つとも答えた。トークナイザーは 3276 トークンが HF と一致
  - 各位置の次のトークンの損失 (3276 トークン): PyTorch 3.354 に対して ANE 3.675。短いプロンプトでも位置ごとの KL が 1.0 (位置 5)、5.3 (位置 23) に跳ね、平均で日本語 0.33、英文 0.76。同じプログラムを Core ML の CPU (`HWNPU_COREML_UNITS=cpu`) で動かすと 0.003 前後で、プログラムの組み方ではなく ANE の数値の問題。8 トークンずつでも 1 トークンずつでも同じ値 (詰め物は関係ない)
  - 部分 1 を coremltools で CPU と ANE に通して出力を比べ (空の状態から 8 トークン)、GDN の層 0 の状態 `st_rec_0_new` がもう 30% ずれていた (`in_proj_qkv` の出力は 0.5%)。層 1 つ (`hwnpu-convert -layers 1`) に途中の値を出力として足して追った:
    1. 減衰 exp(−exp(A_log) · softplus(a + dt_bias)) が、CPU で 0.003〜0.13 のところで ANE は 1.0。**ANE の `softplus` がある入力で 0 を返す**。`relu(x) + log(1 + exp(−|x|))` で書く (`builder.softplus`)。減衰のずれは 28% → 0.1%
    2. 畳み込みの後の **`silu` が ANE では 1.4% ずれる** (自分の入力に対する正確な silu との差。入力自身のずれは 0.4%)。裾 (大きな負の入力) の小さな値ほどずれ、鍵の L2 正規化がそれを単位の長さまで拡大する: 大きさ 0.001〜0.01 の行で 30%、0.01〜0.1 で 7%。`x / (1 + exp(−x))` で書く (`builder.silu`、x < −11 では exp が inf になり 0。真の値は 2e-4 未満)。鍵のずれは 7% → 0.5%、状態は 12% → 0.6%。ゲートの silu(z) と MLP の silu も同じにした
    3. GDN の出力 o = q S は、q が単位ベクトルの 1/√128 倍 (最大 0.09) なので積の項が 1e-5 前後になり、fp16 の正規の最小値 (6e-5) を下回る。ANE ではそこで桁を失い、o が 6% ずれ、ゲートつきの正規化 (行の大きさ 1e-3 前後、eps 1e-6 が効く) が 17% に広げた。ANE の正規化そのものは正しい (ANE の o を numpy で正規化すると ANE の出力と 1% で合う)。q を `oScale` (1024) 倍し、正規化の前に `oNormIn` (64) で割り、eps を (1024/64)² 倍にする (正規化は入力の大きさによらない)。o のずれは 6% → 0.3〜0.5%。割り算は正規化の `inScale` ではなく別の mul にする: `inScale` は重みを 1/64 にして fp16 に入れ、0.004 未満の重みが非正規数になる (合成の重みのテストで 5e-4 ずれた)
    - 行の最大値で割ってから 2 乗する正規化 (Anima の `maxScaled` に eps の補正を足したもの) も試したが、ANE ではかえって悪く (ゲートの後 17% → 40%)、原因は o の側だったので採らなかった
  - 層 1 つの出力のずれ: 25% → 2% (残りは ANE の行列積の累積の精度)
- 直した後の ANE (lut8):
  - 位置ごとの KL: 日本語 0.33 → **0.0073** (argmax 27/27)、英文 0.76 → **0.0052** (29/31)。CPU の lut8 の照合と同じ水準
  - 3276 トークンの損失: 3.675 → **3.363** (PyTorch 3.354)。位置が進んでもずれは育たない (2048〜4096 で 3.356 と 3.310)。GDN の状態を fp16 で持ち続けても、この長さでは積もらない
  - 富士山の答えが PyTorch の fp32 と一字一句同じ (「日本で一番高い山は、標高 3,776 メートルの富士山です。」)
  - 合言葉 (サーバー、`-prompt-cache off -think=false`、温度 0): 3,276 トークン 3/3 (49 秒で読む)、12,895 トークン 3/3 (193 秒、66 tokens/s。生成 77.5 ms/token)。59,797 トークン (約 48K から先は 2 段の注意、既定): 3/3 (1421 秒、42 tokens/s。生成 158.9 ms/token)
  - 速さ (Q=8 の 1 本、下書きあり): LRU キャッシュのコード 250 トークンで 8.5 tokens/s (直す前の lut8)、直した後の短い答えで 7.8 tokens/s
- 足した演算: CPU のインタプリタ (`internal/milcpu`) と OpenVINO IR への下ろし (`internal/ovlower`) に `abs` と `log` (`log(x + epsilon)`)。`ovlower` には前の `softplus` もなく、Qwen3.5 の Intel 向けの変換は通らなかった (16.62 では IR を作っていなかった)。今は `hwnpu-convert -backend intel` が通る。`TestNoNeuralEngineActivations` がプログラムに `silu` と `softplus` が無いことを見る
- 最終のコードで変換し直して: 9 部分とも Neural Engine、富士山と 17 × 23 = 391 と回文の関数を正しく答え (7.6〜8.9 tokens/s)、位置ごとの KL は日本語 0.0073、英文 0.0052、3276 トークンの損失 3.362
- 確かめていないこと: Q=64 のプログラム (2 本で ANE のキャッシュを超えるので既定では作らない。1 本の起動を保つ)。**ほかのモデルも ANE の silu を使っている** (LFM3 の畳み込みと MLP、Gemma 4 と PLaMo 3 の MLP。`blocks.SwiGLU`)。Qwen3.5 では鍵の L2 正規化が小さな値の誤差を拡大したので目立ったが、ほかのモデルの lut6 / lut8 の精度の損 (16.65 の LFM3 の系列) の一部がこれかもしれない

### 16.89 AMD: Gated DeltaNet の IRON の形と、MoE の経路の選択をデバイスで (Qwen3.5 と Qwen3.6-35B-A3B、`iron_gdn.py`、`iron_moe.py`、`iron_qwen35.py`) — 2026-11-06

16.75 の残り (Gated DeltaNet の IRON の形、経路の選択をデバイスで) を書き、Qwen3.5 と Qwen3.6-35B-A3B (`qwen3_5_moe`) を `iron_export_lm.py qwen35` / `qwen35moe` で書き出せるようにした。照合は実物の重みの 2 層 (層 0 = GDN と層 3 = 全注意。`~/models/dl/partial.py` で範囲を読んで作った切り詰めの checkpoint: Qwen3.5-4B と Qwen3.6-35B-A3B、後者は 3.2 GB)。40 層全部はまだ載せていない (重み 72 GB を読み込んでいない)。

**Gated DeltaNet (`tools/amd/iron_gdn.py`、`kernels/gdn.cc`)**

- 再帰そのものは mlir-aie に kernel がある (`ssm/gated_delta_rule.cc`、状態を bf16 の 3 本の limb で持ち、l2 norm、softplus、decay、delta rule、出力) が、IRON に演算子がない。`gdn.cc` はそれを写し、ブロックをコアの中で集める `gdn_gather` (q、k、v に silu も掛ける) と、(dt_bias, A) を状態の前ではなく実行時の語で受ける `gdn_rule` に変えた
- 形: 8 列 × 2 コア。コアは値のヘッド 2 つを持ち、16 列 (slice) ずつ回る。slice の状態 (128 × 16、float32) が入り、呼び出しの行を全部ブロックずつ通し、最後の有効な行の後の状態が出る。列の行はメモリタイルで 2 コアに配り、状態と y と最終状態は 2 コア分を分けて (まとめて) 1 本のシムのチャネルにする。コアの入出力の DMA チャネルは 2 本ずつ、メモリタイルの S2MM は 6 本で、列あたり 2 コアがちょうど収まる (4 コアにはチャネルが足りない)
- 射影は 1 本にした: `in_proj_qkv`、`in_proj_a`、`in_proj_b` の行を列の順 (`gdn_layout`: 列ごとにその列のキーのヘッドの q と k、値のヘッドの v、ヘッドごとの (a, b)) に並べ替え、1 本の GEMV (Qwen3.5-4B / 3.6 で 8256 行) と 1 本の短い畳み込みにする。a と b の畳み込みのタップは 0, 0, 0, 1 (そのまま通る)。silu はカーネルで q、k、v だけに掛ける。各列が読むのは行のうち自分の 1032 語の連続した部分だけ
- 状態は演算子の並び (列、コアのヘッド、slice、コア、128、16) で、列の分が連続した 1 本の転送になる。`Checkpoints` の slot ごとに持つ (Granite 4.0-H の Mamba2 と同じ)。dt_bias と A はヘッドごとに 16 ビット 2 つずつの実行時の語 (書き出し時に決まる resident) で渡す (IRON の `Rtp` はベクトルの値を扱わないので 1 語ずつの値にした)
- 出力: ヘッドごとの RMS norm (重みは `1 +` なし) × silu(z) と `out_proj`。ほかの RMSNorm は `1 + weight` (`RMSNorm.plus_one`)
- **シムのチャネルのタスクキューは 4 段**: プロンプトの形で、y の転送 16 本を待たずに積んだら止まった (`ERT_CMD_STATE_TIMEOUT`)。続けて 2 回止めると SMU が戻らなくなり (`aie2_smu_init: Access power failed`)、ドライバの読み直しでも直らず再起動が要った。チャネルごとに 3 本までにし、古いものを待ってから次を積む (IRON の Gather が `sync()` で抑えているのと同じ)
- 照合 (`python tools/amd/iron_gdn.py check OUT [--rows 64 --valid 50]`、`irongraph -tol 0.05`): デコード (1 行) は y の最大の差 1.5e-5 / 0.15、最終状態 2.4e-7 (コサイン 1.000000)。プロンプト (64 行、有効 50) も y 4.9e-4 / 0.19、状態 3.5e-3 / 1.7 (コサイン 1.000000)
- 速さ: デコード 1 回 0.26〜0.33 ms (32 ヘッド、状態 2 MiB の読み書き)、64 行 7.3 ms (1 行 0.11 ms: 1024 行の塊は 1 層約 117 ms)

**Qwen3.5 (`tools/amd/iron_qwen35.py`、`qwen35`)**

- 注意: `q_proj` はヘッドごとに q と gate (sigmoid を出力に掛ける) が並ぶ。q と k の RMS norm、RoPE は頭の 64 列 (`partial_rotary_factor` 0.25、テキストの位置なので mrope は普通の RoPE)。MLP は SwiGLU
- 照合 (Qwen3.5-4B の層 0 と 3、32 トークンのプロンプトと貪欲法の 3 ステップ、HF の float32 と): IRON のホストのグラフはコサイン 0.99982〜0.99991、KL ≤ 2.7e-5 (`iron_qwen35_check.py ref` / `iron`)。NPU (bf16) は 0.99975〜0.99986、q8 / q4 (`--q4`) は 0.99607〜0.99813。最上位のトークンは全部一致
- 4B 全体は未確認 (重み 9 GB を読み込んでいない)

**MoE の経路の選択をデバイスで (`tools/amd/iron_moe.py`、`kernels/moe_route.cc`、`qwen35moe`)**

- デコードのステップは層ごとに 2 コマンド: A (層の mixer、残差、post-attention norm、経路の GEMV (経路の 256 行と共有の専門家のゲート 1 行)、`MoERoute`) と B (16.75 の `MoEGateUp` と `MoEDown`)。間に空のホストステップを置いてコマンドを分け、最後の B の後に出力層のコマンドを置く
- `MoERoute` (1 コア): ロジットから上位 8 個 (同値は番号の小さい方)、選んだ 8 個だけの softmax (全体の softmax を選んだ分で正規化し直すのと同じ。transformers と同じく bf16 に丸める)、共有の専門家の sigmoid を MoEGateUp の経路の重みとして書き、さらに **コマンド B のスクラッチパッドの語** (選んだ専門家の表の中のバイト位置: `id × 専門家 1 個のバイト数`) を書く。語の位置は IRON の lowering が決めるので、どの語を書くかは重みの表 (`program`、語ごとに専門家の番号、倍率、足す数、シフト) にし、書き出しで lowering の結果 (`params.txt` の B の語の Affine) から埋める (`Qwen35Moe.program_rows`)
- その出力 (`image`) は版の feedback 引数のバッファで、hwnpu はコマンド B の ctrl スクラッチパッドをそこに置く (manifest の `fed`: B のコマンドの番号とバッファ)。ホストはロジットを読まず、B の語も書かない。IRON の `CarriedLoop` の Emit が次の run のスクラッチパッドに書くのと同じ考え方を、1 回の呼び出しのコマンドの間に使った
- hwnpu の側 (`internal/llm/iron`、`internal/backend/amd`): 複数コマンドのデコードを許し、コマンドごとの run を持っておく。`fed` のコマンドは `Run.UseScratchpad` で feedback のバッファを ctrl スクラッチパッドにする (その run は自分のスクラッチパッドを持たず、ホストから同期もしない)。全部のコマンドを待たずに積み、最後に待つ (ハードウェアコンテキストは順に実行する)
- **feedback はデバイスのヒープ (`BufferInstructions`) に置く**: ふつうのバッファ (BO_SHARE) をスクラッチパッドにすると B が `ERT_CMD_STATE_ERROR` になった。ヒープのバッファは、コマンドの処理系はデバイスのアドレス (XdnaAddr) で読むが、シムの DMA はプロセスの仮想アドレス (PASID) で読み書きする。引数の再配置にデバイスのアドレスを入れると A の書き込みが届かない (B は全部 0 番の専門家を読み、HF とのコサインが 0.975 に落ちた) ので、ヒープのバッファを引数にするときは仮想アドレスを入れる (`buffer.dmaAddr`)
- プロンプト: MoE の版はデコードのステップだけで、hwnpu はプロンプトを 1 トークンずつ通す (塊のトークンを専門家ごとに分ける MoE はまだ)
- ホストでの照合: IRON の参照には書くデバイスがないので、`MoERoute` の参照が選んだ専門家を `iron_moeprobe.PICKED` に置き、`MoEGateUp` と `MoEDown` の参照がそれを読む
- 照合 (Qwen3.6-35B-A3B の層 0 と 3、専門家は q4nx): ホストのグラフと HF でコサイン 0.99841〜0.99987、KL ≤ 1.8e-3。NPU とホストのグラフで 0.99997〜0.99998。NPU と HF で 0.99822〜0.99982 (q8 の射影と q4 の出力層 (`--q4`) で 0.99575〜0.99928)。最上位のトークンは全部一致。`MoERoute` 単体はデバイスと参照でビット単位で同じ
- 時間 (コマンドを 1 つずつ待って。`HWNPU_IRON_TIMING=1`): A (埋め込みと GDN の層) 4.7 ms (q8 で 5.1)、A (全注意の層) 2.9 ms (3.3)、B 1.0 ms、出力層 18.1 ms (q4 で 5.9)。40 層の見積もりは 30 × 4.5 + 10 × 3.3 + 40 × 1.0 + 6 ≈ 210 ms (約 5 トークン/秒)。16.75 の見積もり (8〜10 トークン/秒) より遅いのは A の演算の数のため: q8 にしても縮まない。IRON の短い畳み込み (`F.causal_conv1d`) がデコードで約 10 演算、ゲートつきの norm が 3、`MoERoute` が 0.24 ms (スカラーの選択)
- 次: 畳み込みとゲートつきの norm を GDN の演算子に入れる、`MoERoute` のベクトル化、40 層の実物 (重みの読み込み、20 GB の表)、プロンプトの MoE、Qwen3.6 のチャットの型の違い (道具の引数の数と真偽を JSON で書く、`preserve_thinking`)
- 試し方: `python tools/amd/iron_export_lm.py qwen35moe CKPT OUT --max-seq-len 1024 --prefill-chunk 256 [--q4]`、check/ に HF の参照 (`iron_qwen35_check.py ref`) を置いて `go run ./tools/amd/ironlm OUT`

### 16.90 AMD: 畳み込みを GDN の演算子の中へ (止まるので MoE には未適用)、`MoERoute` のベクトル化、Qwen3.6 のチャットの型 — 2026-11-06

16.89 の残りのうち、短い畳み込みの取り込み、`MoERoute` のベクトル化、Qwen3.6 のチャットの型の違いをやった。40 層の実物は重み (72 GB) の読み込み中。

**短い畳み込みを GatedDeltaRule の中で (`iron_gdn.gdn_operator(n, hc, conv=K)`、`kernels/gdn.cc` の `gdn_conv`)**

- デコードの形だけ。射影の行 (畳み込みの前) と、畳み込みのタップ (K, W) を同じチャネルで流し (x の後にタップ、1 つのオブジェクト)、コアが自分の列の 1032 チャネル全部の畳み込みを float32 で足して 1 回だけ bf16 に丸め (`u_row`)、以後の gather はそこから読む
- 呼び出しの前の K − 1 行は、状態の列ごとの run の先頭のペア 1 つ (2 コアの半分ずつに同じ写し、bf16 で詰めて 3 × 1032 語、残りは 0) に置く。最終状態の先頭のペアに、この呼び出しで終わる K − 1 行を書く。状態の Copy (読み書き 1 本ずつ) はそのままで、IRON の `F.causal_conv1d` の約 10 演算がなくなる (GDN の層の A は 26 → 16 ステップ)
- 照合: 演算子単体 (`iron_gdn.py check OUT --conv 4`、`irongraph`) で y と最終状態 (先頭のペアを含む) がコサイン 1.000000。ホストのグラフ (Qwen3.6 の層 0 と 3) は畳み込みの丸めが 4 回 → 1 回になったぶんだけ変わり、HF と 0.99788〜0.99983 (前は 0.99841〜0.99987)。4 回丸めを真似ると前と同じ数になる。密な Qwen3.5-4B (2 層) をデコードだけの形にして畳み込みを入れると NPU で HF と 0.99771〜0.99857 (畳み込みなしの q4 は 0.996〜0.998)、複数コマンドに分けても通る
- **MoE のモデルでは止まる**: Qwen3.6 (2 層) に入れると、2 回目のデコードのステップの最初のコマンドが `ERT_CMD_STATE_TIMEOUT` になる (毎回、7 回止めてもドライバは戻った)。切り分け:
  - 止まる組み合わせは「最後の層の専門家のコマンド (B1) の後のコマンド 0」。コマンド 0 → B1 → コマンド 0 だけでも止まる。B0 (同じ演算子で表の位置だけ違う) なら止まらない。B1 だけを繰り返しても、コマンド 0 だけを繰り返しても止まらない
  - GDN の演算子は無関係: コマンドを分けて GDN の演算子のコマンドを飛ばしても止まる。止まるコマンド 0 は先頭の 4 演算 (埋め込みの Copy、RoPE の Copy、RMSNorm、in_proj の QGEMV) だけでも止まり、その制御コードは前の版のコマンド 0 の先頭とバッファのアドレス以外バイト単位で同じ。前の版ではコマンド 0 → B1 → コマンド 0 で止まらない
  - アリーナの配置 (`--max-seq-len 2048`)、デバイスのヒープの位置 (192 KB ずらす)、B のスクラッチパッドをホストが書く形、どれでも止まる。演算子の配置後の MLIR も前の版と同じ。原因は分からない (16.79 の Gemma 4 の「コマンドの分け方を変えると次の塊の最初のコマンドが止まる」に似る)
  - なので MoE では畳み込みを入れない (`Qwen35MoeLayer.fold = False`)。密な Qwen3.5 はプロンプトの塊があり、その状態の置き場所が違うので、今は入れる先がない (プロンプトの MoE を作るときに塊の形にも同じ並びを入れる)
- hwnpu の側: デコードだけの書き出し (プロンプトの版なし) と複数コマンドのデコードを、`fed` がなくても読めるようにした。`irongraph` は int32 の出力も比べる

**`MoERoute` のベクトル化 (`kernels/moe_route.cc`)**

- 上位 8 個を 32 本のベクトルの max と最初に一致するレーンで選ぶ (選んだものは -inf にする。同値は番号の小さい方で、前と同じ)。作業の配列はスタック (1 KB) に置かず static に
- 単体 (`python tools/amd/iron_moe.py check OUT [--ties]`、`irongraph -tol 0`) で重みとスクラッチパッドの像がビット単位で一致。1 回 0.1 ms (前は 0.24 ms)
- Qwen3.6 の 2 層: HF とのコサインは前と同じ数 (0.99575〜0.99928)。A0 4.37 → 4.26 ms、A1 2.84 → 2.70 ms (1 層 約 0.13 ms、40 層で約 5 ms/トークン)

**Qwen3.6 のチャットの型 (`internal/chat/qwen35.go` の `Qwen36`)**

- 違いは 2 つ: 道具の呼び出しの引数で、文字列でない値を JSON で書く (3.5 は Python の `str()`: `True`、`None`。3.6 は `true`、`null`、`1e+20`)。`preserve_thinking` (前の答えの思考も残す) は hwnpu の `KeepThinking` と同じ
- どちらの型かはクラス名では分からない (Qwen3.5 にも MoE の大きさがある) ので、書き出しの `chat_template.jinja` に `preserve_thinking` があれば Qwen3.6 にする
- 照合: `tools/ref/template_golden.py` で Qwen3.6-35B-A3B の型から作った 9 件 (`testdata/qwen36_template.json`) と一致。3.5 の 9 件も作り直して同じ

**40 層**: 16.91

### 16.91 AMD: Qwen3.6-35B-A3B の 40 層を NPU で (q4、5.2 tokens/s) — 2026-11-07

実物の重み (72 GB、26 本) の 40 層全部を `iron_export_lm.py qwen35moe CKPT OUT --max-seq-len 1024 --prefill-chunk 256 --q4` で書き出し、NPU で動かした。プロンプトの MoE はまだないので、書き出しはデコードの版だけ (プロンプトも 1 トークンずつ)。

- **読み込み**: 本物の checkpoint は視覚の塔 (`model.visual.*`) と MTP の頭 (`mtp.*`) を持ち、IRON の `load_state_dict` は余りの重みを誤りにする。`Qwen35.__init__` で `renamed` の後の木から外す (2 層の切り詰めの checkpoint には無かった)
- **書き出し**: 29.6 分、最大 RSS 50 GB (60 GB の機械で通る。他の重い仕事と並べない)。バンク 23 GB (7 個)、`decode.elf` 53 MB、デコードは 81 コマンド。hwnpu の読み込み 6〜8 秒
- **HF の参照**: float32 の 35B は 140 GB で載らないので、`iron_qwen35_check.py ref --stream` はモデルを meta の装置に作り、層ごとに前のフックで重みを読み (`safe_open`)、後のフックで捨てる (メモリ 20 GB 未満、9 回の呼び出しで約 25 分)。2 層の切り詰めでは前の参照とビット単位で同じ
- **照合** (32 トークンのプロンプトと貪欲法の 8 ステップ、最上位のトークンは全部一致):
  - NPU 対 HF の float32: コサイン 0.98613〜0.99512
  - IRON のホストのグラフ (同じ q4、`iron_qwen35_check.py iron --moe --q4`) 対 HF: 0.99192〜0.99589、KL ≤ 6.9e-2。差の大半は q4 (専門家と頭の 4 ビット、40 層で積み重なる)
  - NPU 対 ホストの q4: 0.99343〜0.99842 (`ironlm` の閾値 0.99 で PASS)。残りは bf16 の活性 (2 層では 0.99997)
- **速さ**: デコード 1 ステップ 192 ms (5.2 tokens/s、16.89 の見積もりどおり)。プロンプトは 1 トークンずつなので 32 トークンで 6.1 秒
- `hwnpu-chat -iron OUT` で Qwen3.6 の型 (16.90、`chat_template.jinja` の `preserve_thinking` で選ぶ) のまま会話できる: 「日本の首都はどこですか？一文で答えてください。」に思考の後で「日本の首都は東京です。」(21 トークンのプロンプト 4.1 秒、209 トークン)
- 残り: プロンプトの MoE (塊のトークンを専門家ごとに)、A の演算の数 (畳み込みは 16.90 の止まる件、ゲートつきの norm)

### 16.92 AMD: GDN の演算子に畳み込みとゲートつきの norm を入れる (MoE の止まる件は別のハードウェアのコンテキストの空のコマンドで)、Qwen3.5-4B の 32 層 — 2026-11-07

**畳み込みを入れると MoE で止まる件 (16.90) の手当て (`internal/llm/iron` の `stepBarrier`)**

- 各デコードのステップの前に、デコードの ELF の空のインスタンス (`aiecc_empty`) を流す。ただし、デコードの ELF をもう一度読み込んだ別のハードウェアのコンテキストで流す。デコードと同じコンテキストで流しても止まった (16.84 の `chunkBarrier` が効くのも、別のコンテキストだから)。1 ステップ約 0.3 ms
- 止まっていた 2 層の書き出し (Qwen3.6、畳み込みあり) が、これで毎回通る (HF とコサイン 0.99548〜0.99927、66 tok/s)。40 層でも止まらない。`Qwen35MoeLayer.fold = True` を既定にした

**ゲートつきの norm も GDN の演算子の中で (`gdn_operator(n, hc, conv, norm=eps)`、`kernels/gdn.cc` の `gdn_rule_keep` と `gdn_norm`)**

- コアは value head の 8 スライスの y をコアの中 (`y_keep`) に貯め、最後のスライスの後に `y / rms(y) * silu(z)` を float32 で計算して 1 回だけ bf16 に丸める。norm の重みは `out_proj` の入力の列に掛けておく
- z は別の GEMV (`in_proj_z`) のまま計算し、x のチャネルで行の後に流す (タップと同じく、同じ fifo に別の In を結ぶ)。z を `in_proj` に融合して 1 本の GEMV (8256 → 12352 行) にすると、層の A が 3.90 → 4.60 ms と遅くなった
- コアのプログラムメモリ (16 KB) が 3 KB はみ出したので、`silu_row` を長さを引数にした 1 つのコピーにし、畳み込みの版では普通の gather を出さない。z はコピーせず、x のオブジェクトを最後のスライスまで持って読む (データメモリが 896 B はみ出していた)
- 照合: 演算子単体で y がコサイン 0.999996、状態が 1.000000 (`iron_gdn.py check OUT --conv 4 --norm`)。畳み込みだけの版とプロンプトの 64 行の版も、カーネルを変えた後で同じく通る。Qwen3.6 の 2 層: ホストのグラフ対 HF 0.99824〜0.99985、NPU 対 HF 0.99650〜0.99947
- 2 層の時間 (`HWNPU_IRON_TIMING`): GDN の層の A が 4.26 ms (前) → 3.90 ms (畳み込み) → 3.62 ms (畳み込みと norm)

**40 層 (Qwen3.6-35B-A3B、q4)**

| 形 | 1 ステップ | 速さ | HF とのコサイン |
|---|---|---|---|
| 前 (16.91) | 192 ms | 5.2 tok/s | 0.986〜0.995 |
| 畳み込みを中に | 181 ms | 5.5 tok/s | 0.987〜0.996 |
| 畳み込みと norm を中に | 172 ms | 5.8 tok/s | 0.986〜0.996 |

- 最上位のトークンはどれも 9/9 一致。書き出しは 28 分、最大 RSS 53.8 GB (60 GB の機械ではほかの重い仕事と並べない)。`hwnpu-chat` で「富士山の高さは**3,776メートル**です。」(5.8 tok/s)

**Qwen3.5-4B の 32 層 (`iron_export_lm.py qwen35 ... --q4`、畳み込みは演算子の外のまま: プロンプトの塊がある)**

- 書き出し 4.3 分、4.7 GB、最大 RSS 19 GB。デコード 172 ms (5.8 tok/s)、32 トークンのプロンプトの塊 1.3 秒
- NPU 対 HF の float32: 最上位のトークンは 8/9 (`<think>` の直後で `\n\n` と `\n`)、コサイン 0.956〜0.996。同じ q4 のホストのグラフ (`iron_qwen35_check.py iron --q4 --decode`) も同じ所で同じトークンを選び、コサインもほぼ同じ (0.958〜0.996、KL ≤ 3.8e-2)。違いは q4 から来る (コサインの低い呼び出しは KL 1e-5 で、確率の小さいトークンのロジットがずれている)

**プロンプトの MoE はまだ**: 専門家の位置をデバイスが渡せるのはコマンドのスクラッチパッド (64 B、16 語) だけで、256 トークンの塊には 2048 語が要る。専門家ごとにトークンを集める DMA もデータで決まる番号では組めないので、塊ごとに全部の専門家の重みを 1 回ずつ流して該当するトークンだけを計算する演算子を作る (見積もりは 256 トークンで約 1.2 秒)

### 16.93 AMD: Qwen3.6 のプロンプトの MoE (経路はホスト、専門家はデバイスで塊ごとに 1 回流す、`iron_moep.py`) — 2026-11-07

デコードの経路の選択 (16.89) は、専門家の位置をコマンドのスクラッチパッド (16 語) で渡すので、256 トークンの塊 (2048 語) には使えない。プロンプトの塊では、層を 2 つのコマンドとホストの手順 2 つにした。

- **A**: mixer、残差 h (状態 `moe_h` に写す)、norm した hn、router のロジット (C × 320)
- **ホストの route** (`moe_route`): 各トークンの上位 8 個 (同値は番号の小さい方、重みは選んだ中の softmax を bf16 に丸める、MoERoute と同じ)、行を専門家ごとに並べ (トークン順)、専門家ごとの行数 (512 語の表: 256 個の専門家の数と、256 番目に埋め草の行の数) と x の流れを書く
- **B**: 共有の専門家 (q4 の Linear、状態 `moe_s` に写す)、`MoEGroupGateUp`、`MoEGroupDown`
- **ホストの combine** (`moe_combine`): out = h + 共有のゲート × 共有の出力 + Σ 重み × 専門家の行 (float32 で足して bf16 に 1 回)。次の層はこの状態 `moe_out` を読む

**演算子 (`tools/amd/iron_moep.py`、`kernels/moe_rows.cc`)**

- 重みの流れは静的: 全部の専門家の重みが 1 回ずつ流れる (gate と up は列 c が列ブロック 2c と 2c+1、down は列ブロック 4 つずつ)。行の数はデータ (x のチャネルの最初の物): コアは専門家ごとにその数だけ回す (`range_` の上限にバッファから読んだ値)。全体の行は常に 8 C なので、流れの大きさも静的
- gate と up: コアが専門家の行 (最大 64、`CAP`) の 32 出力ずつの累算を持ち、k ブロックが通るたびに足す。x はホストが「専門家、gate か up、k ブロック、行」の順に並べる (重みの表の順に合わせ、x は 2 回)。down は列ブロックごとに全専門家を流し、x (A × 512) を 4 回読む
- 埋め草の行 (呼び出しの行数を超えた塊の残り) は専門家に割り当てず最後に置き、コアは x を読み流して 0 を書くだけ (32 トークンの塊で B が 67 → 18.5 ms)
- 2 行ずつ (`q4nx_accumulate2`、符号を bf16 に広げるのを 2 行で 1 回): 256 トークン 256 専門家で 1 層 68 → 45 ms
- DMA の 1 次元の大きさは 1023 語まで: 重みを消費順に並べ替える 4 次元の TAP は通らないので、表の順のまま (列ブロックを 2048 バイトずつ) 流す
- ある専門家に 64 行を超えるトークンが来る塊は、ホストが塊をいくつかの回に分け、B を回の数だけ流す (Go の `runMoECommand`)
- 照合: 単体 (32 専門家、16 トークン) コサイン 0.999807、実物の大きさ (256 専門家、128 トークン) 0.999815、34.5 ms

**IRON の落とし穴: ホストの手順の読み出しは、領域の割り当てで使用に数えられない**。最初の版はホストが読む hn と h が、あとの演算 (共有の専門家の積) に上書きされて NaN になった (コマンド 0 の後の hn の 32 行目以降に silu(g) u が入っていた)。ホストが読むのは、ホストの手順の直前の演算の出力か、状態に写したものだけにする

**畳み込みを中に入れた形と両立させる (`iron_gdn.gdn_operator(..., pairs=True)`、`internal/llm/iron/convpairs.go`)**

- デコードは 16.92 の形 (畳み込みとゲートつきの norm を GDN の演算子の中で、履歴は状態の列ごとの先頭の組)。プロンプトの塊は IRON の `CausalConv1d` と自分の状態で畳み込み、GDN の演算子は各列の先頭の組を読み飛ばして書かない (`pairs`)。norm は重み 1 で (重みは out_proj の中)
- hwnpu が塊の前に、塊が読む枠の GDN の状態の組から畳み込みの状態へ履歴を写し、塊の後に、塊が書いた枠の畳み込みの状態から GDN の状態の組 (2 コアの半分ずつ) へ写す (manifest の `conv_pairs`)。畳み込みの状態はプロンプトの版だけのバッファなので、デコードの版の一覧にも同じ置き場所で載せる (checkpoints に入れるため)

**hwnpu の側**: manifest にホストの手順 (`host`: どのコマンドの前か、種類、読み書きする領域) を書き出し (`iron_export_lm.py` の `host_steps`)、Go がプロンプトの塊のコマンドの間で実行する (`internal/llm/iron/moe.go`)

**結果**

| 40 層、q4 | 前 (1 トークンずつ) | 今 |
|---|---|---|
| 32 トークンのプロンプト | 5.5〜6.1 s | 2.21 s |
| 183 トークンのプロンプト | 約 35 s (見積もり) | 4.6 s (25 ms/トークン) |
| デコード | 172 ms | 172 ms (5.8 tok/s) |
| HF とのコサイン、最上位のトークン | 0.986〜0.996、9/9 | 0.985〜0.996、9/9 |

- 2 層: 32 トークンのプロンプトが 104 ms (1 トークンずつ 480 ms)、HF とコサイン 0.99702、後のデコードも 0.99906〜0.99938
- `hwnpu-chat` で 3 回続けて会話 (2 回目以降は 256 トークンを超えて塊が 2 つ): 「東京都の人口は約1400万人です。」と前の答えを踏まえて答えた。320 トークンのプロンプト 8.3 s
- 書き出し 29 分、最大 RSS 50 GB
- 残り: GDN の層の A (プロンプトの塊で 1 層 約 38 ms、射影の GEMM と GDN の演算子) が塊の時間の半分近く。専門家の B は重みを読む速さ (約 27 GB/s で 1 層 18 ms) が下限。会話の 2 回目に、毎回会話全体を読み直している (このモデルの状態の保存を使っていない)

### 16.94 設計案: モデルを部品として組み合わせる (ライブラリの形、常駐の管理、受け渡すデータの種類) — 2026-11-07

まだ実装していない。音声認識 → LLM → TTS のように、小さなモデルをつないで同時に動かす使い方に向けた方針をまとめる。

#### 動機と今の形

- 今のモデルは、どれもコマンドごとの独立した CLI (`hwnpu-chat`、`hwnpu-image`、`hwnpu-tts`) になっている。外から import できる公開 API はルートの `hwnpu` (デバイス、バッファ、グラフとカーネルの実行) だけで、モデルの処理は `internal/` (`chat`、`imagegen/anima`、`speech`、`llm/iron`) の中にある
- `chat.Server` は複数のチャットと埋め込みのモデルを 1 つのプロセスに持つが、「各モデルは 1 度に 1 要求」で、モデル同士の調整はしない。画像と音声は入っていない
- 載せ方はモデルごとに決め打ちになっている
  - Anima (AMD、16.72): テキストは条件付けを作る間だけ、DiT はステップの間ずっと、VAE は段を 1 つずつ載せる。ピークは DiT の分で済むが、1 枚ごとに数秒の読み込みがかかる
  - Anima (Core ML、Intel): テキストと DiT は常に載せ、VAE だけを段ごとに載せる
  - チャット (AMD、16.74): `-idle` の間要求がなければ全部を放す
- 画像の生成は、1 枚の中ではテキスト → DiT → VAE と前の出力を次が使うので、段ごとに放しても並列性は失わない。失うのは読み込みの時間と、複数の枚をずらして流すことだけ
- 音声の連鎖は逆で、小さなモデルを全部載せたまま、時間を重ねて動かす (TTS が 1 文目を作る間に LLM が 2 文目を作る、再生中も音声認識が聞く)。要求ごとに読み直すと、0.8 秒から数秒の遅れがそのまま会話に出る

#### 1 つのプロセスが NPU を持つ

メモリの予算と、NPU に出す仕事の順番は、NPU を使う全モデルを 1 つのプロセスがまとめて持たないと調整できない。CLI を別々に起動してパイプでつなぐ形では、互いのメモリも仕事も見えない。何を調整するかはバックエンドで違う。

- **メモリの予算**: AMD のバッファは固定されたメモリで、RSS に出ない (16.72)。モデルは部品ごとに大きさと種類を登録する。種類は「常駐させたい」(音声の連鎖、会話中のチャット) と「追い出してよい」(Anima のテキスト、DiT、VAE) の 2 つ。予算を超えたら、追い出してよいものから、次にしばらく使っていないものから放す。Anima の段ごとの解放は「足りないときだけ起きる追い出し」の 1 つになり、余裕があれば全部載ったままになる。`-idle` もこの仕組みの 1 つにまとめる
- **仕事の投入 (AMD)**: 1 つの DMA チャネルに 5 件以上の転送が溜まると止まる件 (2 回止まって再起動が要った) は、1 つのコマンドの命令列 (IRON のランタイムの列) の中の制約で、書き出す側が守る。モデルごとのハードウェアコンテキストはファームウェアが時分割するので、モデルが増えてもこの上限に近づくわけではない。中央で調整するのは、(1) 長いコマンド (DiT のグループなど) の間、ほかのモデルが待つこと (コマンドの途中で切り替わるかは未確認)、(2) 1 つのコマンドが止まると NPU 全体が止まり、ほかのモデルも巻き込むこと。投入を 1 か所で受け、優先度 (音声を先に) と止まったときの扱いをそろえる
- **仕事の投入 (Apple)**: DMA は Core ML の中にあって見えない。調整するのは、ANE のキャッシュ (約 4 GiB、16.88) を超えると追い出されて再コンパイルに分単位かかること、プログラムの切り替えが 1 回 15〜35 ms かかること (16.36)、裏での読み込みが生成を 2 倍以上遅くすること (16.20)。メモリの予算と同じ形で「ANE のキャッシュの予算」を持つ
- **仕事の投入 (Intel)**: Level Zero のコマンドキューをドライバが管理する。モデルを同時に動かしたときの振る舞いは未測定
- **帯域**: LLM の生成 (q4 の GEMV) は DRAM の上限 55 GB/s に当たっている (16.73)。TTS などを同時に動かすと生成は遅くなる。どれだけかは測る
- **列の分け方**: 別々のプロセスが同じ NPU を使えることは確認済み。各グラフが列を全部使うなら実際は時分割になる。本当に同時に動かすには、グラフごとに使う列を分ける必要がある (未検証)
- 遠回りになるが、まず Anima に `-resident all | stage | auto` を付ける。`auto` は書き出しのマニフェストから各段の大きさを足し、空きと比べて選ぶ。DiT を q8 にした後 (16.78) の全部載せの合計は、`/dev/accel/accel0` の写像の大きさで測ってから決める

#### 3 つの層

1. **ライブラリ**: 用途ごとのインターフェース (`Transcriber`、`Chat`、`Speaker`、`ImageGen` など、名前は仮)。どれも同じ `hwnpu.Device` の上で動き、Device の常駐の管理に部品を登録する。少しずつ流す使い方 (音声のチャンクを入れると文字が出る、1 文ずつ入れると音声が出る) を前提にする
2. **常駐のプロセス**: NPU を持つプロセスを 1 つにして HTTP で出す。`chat.Server` を広げて、画像、音声認識、TTS の窓口を足す形。Go 以外 (ComfyUI のノード、Python、エディタ) はここから組み合わせる。複数のクライアントが来ても調整は 1 か所で済む
3. **CLI**: 1 か 2 を呼ぶだけの薄いもの。試すための入口として残す

#### 受け渡すデータの種類

共通の外枠を先に決める。音声と口の形とモーションをそろえる、話の途中で割り込む、といったことは、種類ごとの定義ではなく外枠で決まる。

- **一括と流れ**: どの種類も「一括の値」と「チャンクの流れ」の両方を持つ。チャンクには続きがあるか最後かの印を付ける
- **時刻**: 共通の時計の上の開始時刻と長さ
- **制御**: 発話の終わり、割り込み (相手が話し始めたら TTS を止める)、取り消し。データとは別の信号として流す
- **出どころ**: 作ったモデル、トークナイザーの識別子。同じトークナイザーどうしで ID のまま渡す、専用のベクトルを正しい相手にだけ渡す、の判断に使う

| 種類 | 中身 | 決めておくこと |
|---|---|---|
| テキスト | 文字列 (トークン ID とトークナイザーの識別子を付けてもよい) | 言語。流すときは文やトークンの区切り |
| 構造化テキスト | JSON の値と JSON Schema | スキーマは受け口の条件。生成する側は文法で縛ったデコードで必ず満たす。ツール呼び出しもここに入る |
| 画像 | 幅、高さ、画素の形式、色空間 | 受けられる大きさ (Anima は書き出しごとに固定)。動画は時刻つきの画像の列として後で足す |
| 音声 | PCM の float32、サンプルレート、チャンネル数 | 受けられるレート (音声認識は 16 kHz、LFM3-Audio の TTS の出力は 24 kHz など)。変換は自動で挟む |
| モーション | 骨格の定義 (関節の木構造、例: VRM の人型)、フレームレート、フレームごとの回転と位置 | 骨格をどの標準に合わせるか。顔の表情 (ARKit の 52 の blendshape など) を別の種類にするか |
| 埋め込み | ベクトルとモデルの識別子 | 同じモデルどうしでしか比べられないことを型で示す |
| モデル専用のベクトル | 隠れ状態など | 学習で組になった相手にだけつなぐ |

- **受け口**: 部品は入口と出口ごとに「種類と条件」(サンプルレート、画像の大きさ、言語、スキーマ) を宣言する。つなぐときに照らし合わせ、合わなければ変換 (リサンプル、リサイズ、テキスト → ID) を挟むか、エラーにする。Omni 系のモデルは「入口: 音声、出口: 音声とテキスト」の 1 つの部品として、ほかと同じに扱う
- 今の実装には、構造化テキスト (スキーマに沿った生成、`response_format` など) がない。種類を決めるのとは別に、`chat` に文法で縛るデコードを作る必要がある

#### モデルの間をベクトルでつながない理由

- トークナイザーが同じでも、隠れ状態は互換にならない。トークナイザーが決めるのは文字列と ID の対応だけで、埋め込みの表から先はモデルごとに学習されている。同じ構造でも初期値が違えば座標系は回転や並べ替えの分ずれ、同じ系列でも幅が違う (Gemma 4 の E2B と E4B)。同じベースから微調整したモデルどうしは近いが、保証はない
- 同じトークナイザーで得られるのは、ID のまま渡せること (再トークン化は軽いので得は小さい)、投機的デコード、ロジットを比べたり混ぜたりできること (KL での品質の確認もこれ)
- ベクトルで直結しているのは、橋を一緒に学習したものだけ: LFM3-Audio の TTS (テキストモデルの最後の隠れ状態を depthformer が読む、`hwnpu-tts` の `-hidden-out 1`)、多モーダルの塔 (エンコーダの出力を学習した射影で LLM の埋め込みの空間へ)、Anima (テキストエンコーダのベクトルを学習したアダプタで DiT へ)。別々のモデルを後からつなぐ研究 (model stitching、KV キャッシュを渡す Cache-to-Cache など) は、組み合わせごとに学習が要る
- したがって部品の境目はテキスト (同じトークナイザーなら ID) にする。音声のまま抑揚や話者を保ちたいときは、別々のモデルをつなぐのではなく、音声を直接扱うモデル (LFM3-Audio、LFM3-Omni) を 1 つの部品として使う

#### 進め方

1. `internal/` のまま、Device の常駐の管理 (予算、追い出し、投入の順番) と、用途ごとのインターフェースを作る。最初の利用者は Anima の `-resident`。チャットの `-idle` もここに移す
2. データの種類の外枠 (一括と流れ、時刻、制御、出どころ) と、テキスト、構造化テキスト、画像、音声の 4 種類。どれも今のモデルで試せる。モーションは枠だけ空けておく。骨格の標準は実際のモデルが決まってから選ぶ
3. AMD に TTS か音声認識を持ってくる (`hwnpu-tts` は今は Core ML だけ)。2 つ目の組み合わせが動いてインターフェースの形に納得できたら、公開のパッケージに出す。早く公開すると API を早く固めすぎる
4. 常駐のプロセスは `chat.Server` を広げる形で、1〜3 と並行して進められる

未定のこと: 種類の型を Go でどう表すか (インターフェースか、タグつきの構造体か)、時計の基準 (音声のサンプル数か、単調な壁時計か)、列を分けて本当に同時に動かせるか、TTS と LLM を同時に動かしたときの生成の遅れ。

### 16.95 ほかのモデルの silu も exp と real_div で書く、Neural Engine のメモリを読む、GPTQ の表を M3 で確かめる — 2026-11-07

**silu (`blocks.SiLU`)**

- 16.88 の残り: Qwen3.5 で見つけた ANE の `silu` のずれ (1.4%、裾の小さな値ほど大きい) が、ほかのモデルの精度の損にも入っているか。`blocks.SwiGLU` と `Activation("silu")` を `x / (1 + exp(-x))` (`blocks.SiLU`) にし、LFM3 (VL、Audio、Omni、d1 の文の部分を含む)、LFM3 の視覚と音声の塔、TTS の depthformer、PLaMo 3、PLaMo-Embedding、Anima の文の部分 (Qwen3-0.6B) に効く。Anima の DiT の silu も (下)
- 測り方: 128 トークンの英文 (LFM3 の model card) と日本文 (富士山、『こころ』、信濃川の段落) の PyTorch fp32 の参照 (`ref_logits.py`、BOS あり) に対し、`hwnpu-convert -seq 128` (キャッシュなし、2〜3 部分) を coremltools で CPU_AND_NE と CPU_ONLY に通して最後の logits の KL を比べた。`tools/ref/eval_logits.py` を parts.json を順につなげるように広げた (前の部分の出力を同じ名前の入力に、Gemma 4 の `ple` はチェックポイントの表の行を `--table` で)
- 位置ごとの KL の平均 (英文 / 日本文)。どれも配置は全部 Neural Engine (`hwnpuinfo`):

| モデル (形式) | ANE の silu | `blocks.SiLU` | 同じプログラムを Core ML の CPU で |
|---|---|---|---|
| LFM3-1.7B (lut6) | 0.365 / 0.082 | 0.336 / 0.076 | 0.337 / 0.077 |
| LFM3-1.7B (lut8) | 0.058 / 0.012 | **0.022 / 0.0048** | 0.026 / 0.0058 |
| PLaMo 3 Translate 2B (lut8) | 0.0067 / 0.0069 | **0.0053 / 0.0042** | 0.0069 / 0.0065 |

  - ANE の上の損はほぼ silu だった。直すと Core ML の CPU (fp32 の計算) と同じ水準になる。lut6 では量子化の損が大きく、差は 1 割に満たない。lut8 では KL が 2.5 分の 1
  - `x · sigmoid(x)` は ANE の `silu` と値まで同じ (ANE の中では同じものらしい)。`sigmoid` では直らない
  - PLaMo-Embedding-1B (lut8 + head-scale、`TestANEPrecision`): 出力の相対誤差 0.0243 → 0.0170、CPU との埋め込みのコサイン 0.99986 → **0.99993**
  - Gemma 4 E2B (lut6) の MLP は `gelu` (tanh 近似)。同じく `x / (1 + exp(-1.5958 (x + 0.044715 x³)))` で書いても KL は 0.243 / 0.424 → 0.240 / 0.428 で変わらない (CPU 0.245 / 0.411)。ANE の `gelu` は十分正確なのでそのまま
- 費用 (LFM3 lut6、チャットの形: 注意で 7 部分、Q=8 と Q=128): 3790 トークンのプロンプトが 2.7 → 2.95 秒 (+9%、4 回ずつ 2 巡)、生成の 1 回の呼び出しが 33.9 → 35 ms 前後 (+3%、ばらつきに近い)。1 つの演算が 4 つになるため。温度 0 の要約の答えは見出しまで同じで、その先の言い回しは分かれた
- 変換し直し: プログラムの名前は形式と設定で決まり、silu を変えても同じ名前になる。`llm.Arch.Revision` を足し、hwnpu-chat の変換の名前に `-r1` を入れた (`lfm3-r1-lut6-grouped-p8192-...`、`lfm3-vision-r1-512x512`、`embed-r1-lut8-s128`)。lfm3、lfm3-vision、lfm3-audio-encoder、plamo3、plamo-bi が 1。Anima は `ExportVersion` を 3 に。古い名前のプログラムは消さないので、ディスクに残る
- 確かめていないこと: LFM3-VL / Audio / Omni / d1 と塔、TTS (どれも同じ書き換えで、変換し直しになる)

**Anima の画像 (M3、1024 × 1024、既定の int8 / int8)**

- diffusers の fp32 (`tools/ref/ref_anima.py`、CPU で 1 枚 5 分) の参照を 3 つのプロンプト (少女と桜、本を読む眼鏡の男、山と湖の夕焼け) で作り、同じノイズ (`-noise`) から書き出しを変えて比べた (条件づけのコサイン、最後の潜在のコサイン、画像の PSNR)。16.85 では torch がなく、hwnpu 自身の fp16 の書き出しとしか比べていなかった
- 試すために一時的な環境変数で silu を切り替え (コミットしない)、書き出しごとに 3 枚:

| 書き出し | 条件づけ | 潜在のコサイン | PSNR (dB) | 1 ステップ / VAE |
|---|---|---|---|---|
| A: 文も ANE の silu (前の版) | 0.9976 / 0.9975 / 0.9928 | 0.847 / 0.559 / 0.752 | 12.6 / 12.7 / 14.8 | 4.1 s / 3.2 s |
| B: 文だけ `blocks.SiLU` | 0.9976 / 0.9975 / 0.9924 | 0.846 / 0.555 / 0.751 | 12.6 / 12.7 / 14.7 | 4.1 s / 3.3〜3.4 s |
| C: B と DiT の silu | 同じ | **0.948 / 0.672 / 0.977** | **16.1 / 14.1 / 25.8** | 4.1〜4.2 s / 3.4〜3.5 s |
| D: C と VAE の silu | 同じ | C と同じ | 16.1 / 14.1 / 25.8 (+0.04 以内) | 4.2〜4.3 s / 4.0〜4.3 s |
| E: C で DiT を fp16 | 同じ | 0.960 / 0.697 / 0.983 | 17.1 / 14.5 / 26.8 | 4.5 s / 3.7 s |

  - 文の silu は効かない (条件づけの誤差は int8 の量子化で決まる)
  - **DiT の silu** (時刻の埋め込み `dit_time` と、各ブロックと最後の層の変調の入力の `act`) が、ANE の上の誤差の大半だった。小さなテンソル (8 行と 1 行) で、28 ブロックすべての shift、scale、gate を動かす。直すと風景はほぼ参照どおり、少女も参照に寄り、男は窓とカーテンが参照に寄った (姿勢はまだ違う)。時間は変わらない
  - VAE の silu は画像を変えず、VAE が 15〜20 % 遅くなるのでそのまま
  - DiT の fp16 は C より少し近い (潜在 +0.006〜0.025) が 8 % 遅く、書き出しが 1.8 GB 増えるので、既定は int8 のまま。16.85 の「int8 は fp16 との潜在のコサイン 0.902」は両方に ANE の silu の誤差が入っていた
- 採ったもの: DiT の 3 か所の silu を `blocks.SiLU` に (`anima/mil/dit.go`)。`ExportVersion` 3 (文と DiT)。最終のコードで書き出し直すと C と潜在がビット単位で同じ
- 残る差: 2 つ目のプロンプトは C でも 0.67 (軌跡の分かれやすい種)。参照に対する残りの誤差の内訳 (文の int8、DiT の int8、fp16 の計算) は分けていない

**Neural Engine のメモリ (`mem.NPUWired`、model-fit.md T10)**

- ANE に載せた重みはプロセスの RSS にも footprint にも aned にも出ず、カーネルの wired メモリの `com.apple.driver.AppleH11ANEInterface` に入る (2026-11-06 に zprint で確かめた)。`mach_memory_info` (root 不要) を purego で呼び、その名前の大きさを返す。vm のタグ番号は macOS 27.2 で 167 → 158 に変わっていたので名前で探す。何も載せていないときは 144〜192 KiB (27.0.1 で見えた 1.22 GiB の下地はなくなっていた)
- hwnpu-chat は読み込みの前後の差を出す: LFM3-1.7B lut6 の 2 本 (Q=8 と Q=128) で「the Neural Engine holds 2.43 GiB」(重み 1.21 GiB × 2。同じ重みでもプログラムが違えば 2 つ持つ)。終わると戻る
- `npuprobe split` の表に「Neural Engine」の列: lut4 の 32 層を 2 部分 (重み 0.91 GiB) で 0.46 GiB。npuprobe の部分は同じ種で作るので 2 つが同じ中身になり、ドライバーは 1 つだけ持つらしい

**GPTQ で選んだ lut6 の表 (16.87) を M3 で** (docs/handoff-m3-gptq.md)

- 符号は Linux の機械に届かず、M3 で作り直した: `q4_gptq.py lfm3 CKPT OUT --fmt-q4 lut6 --fmt-q8 lut6 --head lut6 --lut-fit wkmeans --q8 gptq` (校正の文 `~/models/calib` も作り直した。GPL-3 は Mac にないので評価から抜ける)。`--q8 gptq` がないと注意と畳み込みの射影は丸めのままで、符号は 143 のうち 73 (MLP と出力層) だけになる (引き継ぎの手順にはこの引数が抜けていた)。校正は 2141 秒 (4 層ずつ 7 回、1 回 4〜5 分。スワップが 9 GB を超えていると 1 回目が 45 分かかった)。符号は 1.6 GB
- CPU の見積もり (bf16 の重みに対する KL): README 0.0203、wiki40b-ja 0.0500、wikitext-2 0.0291、Go 0.0189
- `hwnpu-convert -codes` で 143 の重みが選んだ表を取り、2 部分とも全部 Neural Engine。上の英文と日本文 128 トークンの PyTorch fp32 に対する KL (平均、英文は中央値も):

| 変換 | ANE | Core ML の CPU |
|---|---|---|
| 変換器の丸め (`-quant` 既定) | 英 0.336 (中央値 0.179) / 日 0.076 | 0.334 / 0.077 |
| チャットの丸め (`kmeans-minmax`) | 英 0.322 (0.155) / 日 0.054 | |
| GPTQ、MLP と出力層だけ (73) | 英 0.255 (0.126) / 日 0.050 | 0.259 / 0.050 |
| GPTQ、全部 (143) + wkmeans | **英 0.247 (0.097) / 日 0.035** | 0.255 / 0.037 |

  - ANE でも CPU と同じ順で、ANE の上乗せはない (silu を直した後)。チャットの丸めに対して英文の平均 −23 %、中央値 −37 %、日本文 −35 %。英文の平均は markdown のリンクなど予測の難しい位置で決まり、どの変換でも大きい
- チャットの形 (注意で 7 部分、Q=8 と Q=128、`-quant kmeans-minmax -codes`) で hwnpu-chat と比べた (名前を合わせて変換したものを置いた)。速さは同じ: 3790 トークンを 2.9 秒、生成の 1 回の呼び出し 33〜36 ms
- 答え (温度 0、考えない): 事実の 12 問 (富士山の高さ、首都、『吾輩は猫である』の作者、水、光の速さ、琵琶湖、1 年の日数、一番近い恒星、英語の 4 問) で、丸めも GPTQ も 10/12。外したのは丸めが猫の作者と恒星、GPTQ が**富士山 (4,324.5 m と答えた)** と猫の作者。bf16 の PyTorch は富士山を 3,776 m と答え、「日本で一番長い川」には丸めも GPTQ も bf16 も崩れる (モデルの限界)
- 判断: KL は確かに下がるが、少数の問いでは答えの良し悪しは見分けられず、GPTQ で富士山が崩れた例もある。既定にするには、もっと多くの問い (と hwnpu-agent の課題) で比べてから。符号の配り方 (1.6 GB) も未定のまま。lut4 + wkmeans は試していない (16.87 の CPU で lut6 の丸めの 4 倍)

**hwnpu-chat の ANE のメモリの表示 (追記)**

- Core ML がその場でコンパイルしたプログラムは、読み込んだ直後には ANE に載らず (LFM3 で 0.27 GiB)、最初の呼び出しで載る (2.43 GiB)。キャッシュから読んだときは読み込んだ時点で 2.43 GiB。hwnpu-chat は差が重みの半分に満たないとき「so far: programs compiled just now come onto it at their first call」と書く。`npuprobe split` は 1 回動かした後に測る

### 16.96 会話の続きを読み直さない (再帰の状態を持つ IRON のモデル)、短い読みはデコードで、密な Qwen3.5 にも畳み込みと norm を — 2026-11-07

- **会話の 2 回目以降に会話全体を読み直していた**: Qwen3.6 の型は、前の答えを描き直すときに思考 (`<think>…</think>`) を外すので、新しいプロンプトと前の列が共通なのは、前のアシスタントのターンの始まりまで。再帰の状態は印 (mark) の位置にしか戻れず、印はプロンプトの最後 (`<|im_start|>assistant\n<think>\n\n</think>\n\n` の後) にしかなかったので、毎回 0 から読んでいた。prefill が最後のターンの始まり (`<|im_start|>` の前) でも読みを切って印を付ける (`Engine.markTurns`、再帰の状態を持ち Restartable でない IRON のモデルだけ)。3 回目の会話で 87 トークン中 44 だけ読む
- **短い読みはデコードのステップで** (`ironSession.StepTokens`): Qwen3.6 の 40 層はプロンプトの塊が短くても約 2.2 s (GDN の層の A と専門家の重みの読み)、デコードのステップは 0.17 s。セッションが両方の時間を覚え、トークン数 × ステップの時間が塊より短ければステップで読む。会話の 2 回目以降 4.4 → 3.5 s
- **密な Qwen3.5 にも 16.92 の畳み込みとゲートつきの norm** (`Qwen35Layer.fold = prompt = True`、プロンプトの塊とは 16.93 の pairs と conv_pairs で両立): Qwen3.5-4B の 32 層のデコード 172 → 157 ms (6.4 tok/s)、HF との照合は前と同じ (最上位のトークン 8/9、q4 のホストのグラフと同じ所で外れる)。会話の続きも 50 トークン中 37 だけ読む

### 16.97 AMD: プロンプトの塊の GDN の演算子は呼び出しの行のブロックだけ回す — 2026-11-07

- 内訳の測り方: プロンプトの版を手順ごとのコマンドに切って書き出し (IRON の `plan` を包み、`Plan.commands` を全部の手順に)、`HWNPU_IRON_TIMING` で各コマンドを測る。密な Qwen3.5 の 2 層、256 行の塊に 32 行で、GDN の層の約 38 ms のうち 27 ms が `GatedDeltaRule` だった
- 演算子は 256 行の 16 ブロックを毎回全部回していた (有効な行のないブロックも gather と rule を通す)。コアが有効な行を持つブロック (少なくとも 1 つ) だけで rule を回し、残りは x を読み流して y に 0 を書く (`gdn_zero_y`)。単体で 27 → 5.0 ms、照合は同じ
- 残りの内訳 (同じ 2 層、32 行): 射影の GEMM 17.7 ms (行が少なくても 256 行で計算する)、重みの展開 (q4 の `DequantBFP` 9.6 ms、q8 の `QDequant` 7.9 ms、塊ごとに全部)、出力層 7.3 ms、GDN の演算子 5.7 ms
- Qwen3.6 の 40 層: 32 トークンのプロンプト 2.21 → 1.60 s (ロジットは前と同じ)。会話のターンの印は、ターンの見出しの行 (`<|im_start|>assistant\n`) の後ろに置き、印の後をデコードで読むトークンを 7 → 4 にした。会話の 2 回目以降の応答 2.5〜3.7 s (256 トークンの境目をまたぐと 2 回の呼び出し)、1 回目は印のぶん 4.6 → 6.2 s

### 16.98 PLaMo 3 Translate 0.8B (`plamo-3-translate-0.8b`) — 2026-11-07

- pfnet/plamo-3-translate-0.8b (2026-10-27 公開、PLaMo Community License) は 2B と同じ `Plamo3ForCausalLM` で、違うのは大きさだけ (隠れ層 1536、中間 6144、16 層、注意のヘッド 16 に KV ヘッド **1**、語彙 107520、出力の行列は埋め込みと別)。トークナイザーとチャットのテンプレートは 2B とバイト単位で同じ。`internal/llm/plamo3` は config から形を決めるので、コードの変更はカタログの 1 行だけ
- new-model-checklist の既存の最適化は、どれもアーキテクチャの側にあるので、そのまま効く: `blocks.SiLU` (r1)、norm の入力を 1/64、注意の出力を token を先にした `attn_ot_N`、注意で分けた形 (全注意の 2 層 7 と 15、文脈 262144)、窓のリングの印は使っている行だけ (`ringRows`)、変換の後の `debug.FreeOSMemory`、既定の lut8
- 2B との違いは `onePrograms` の判断: lut8 の 2 本 (1 回 8 トークンと 128 トークン) の重みは 1.37 GiB で、コンパイルのキャッシュ (約 4 GiB) に収まるので、プロンプトを読む用のプログラムも持つ (2B は 4.9 GiB で 1 本)
- M3 (16 GB、macOS 27.2)、hwnpu-chat:
  - 変換 (2 本、3 部分ずつ 0.19〜0.27 GiB): 約 80 秒。初回の読み込み 31 秒 (Core ML のコンパイル)、2 回目は 2.6 秒。どの部分も `all on the NPU`
  - メモリ: Neural Engine に 1.38 GiB (2 本)、プロセスの peak footprint 1.0 GB。KV は 1 トークン 1 KiB (全注意 2 層 × KV ヘッド 1 × 128 × 鍵と値、fp16)
  - 速さ (同じ 20 文を貪欲に): 生成の中央値 1 秒に 52.9 トークン、2B (lut8、1 本) は 17.1 で 3.1 倍。1629 トークンの文書 (wiki40b-ja) を 1.6 秒で読んだ (16.99 の `LastRow` の後は 1.2 秒。2B は 1 回 8 トークンなので約 200 回、57.8 ms ずつで 12 秒ほどの見込み、測っていない)
- 形式 (`hwnpu-formats -formats lut6,lut8 -ref fp16`、キャッシュ 4096、1 回 8 トークン、M3): 次のトークンが fp16 と同じなのは lut6 231/265 (87%)、lut8 255/265 (96%)。ロジットの相対誤差 (平均/最大) は 0.161/0.459 と 0.045/0.167。1 回の呼び出しは fp16 30.9 ms、lut6 16.4、lut8 20.3。2B (16.71) の 89% と 98% より少し下がるが、同じ傾向
  - hwnpu-formats の固定の会話は問答なので、翻訳モデルではパープレキシティ (2.7 万〜2.9 万) に意味はない。表の「weights」も lut6 と lut8 で 0.00 GiB と出た (分けない 1 本の `.mlpackage` で `chain.Part.WeightBytes` が数えられていないらしい、未修正)
- 訳文で比べる (new-model-checklist 9 節。英 → 日 10 文と日 → 英 10 文、1 文ごとに `/reset`、`-temp 0`): 0.8B の lut8 は 0.8B の fp16 と **20/20** 一字一句同じ。lut6 は 16/20 (違う 4 文は「予定が入っている」→「予定がある」などの正しい言い換え)。0.8B の lut8 と 2B の lut8 では 8/20 が同じで、残り 12 文もどれも正しい訳の言い換え (「開館し」→「オープンし」、"take an umbrella with you" → "bring an umbrella" など)。既定は lut8 のまま
- MODEL.md: 8 GB の列の翻訳を 0.8B に、16 GB 以上は 2B と並べた
- 確かめていないこと: Hugging Face の fp32 との照合 (`hwnpu-check`。この Mac に torch がない。構造は 2B と同じで、2B は照合済み)、AMD (`iron_export_lm.py plamo3`) と Intel、GPTQ の符号、長い文脈の passkey
- **チップ** (M2 と M3 の結果を受けて): 16 GB で 8B を勧めると M1 の 16 GB でも選ばれて 0.8 トークン/秒になるので、models.yaml に `not_on` (チップの一覧: M1〜M5。M1 Pro などは M1) を足した。ファミリーと用途はそのチップではそのモデルを選ばず、ファミリーの次 (8B なら 2B) を選ぶ。用途の列が外されたモデルのときも、同じファミリーのその列までで一番大きいもの (`notOnHere`)。名前と `FAMILY:TAG` (`plamo3translate:8b`) は今までどおり動かす。チップは `machdep.cpu.brand_string` (`chipName`)。8B は `machine: 16`、`not_on: [M1]`、用途 translate の 16 GB の列を 8B に (24 GB の列はなくした)。M2 は起動のたびに 35〜41 秒かかるが、生成は M3 と同じなので外していない (`not_on: [M1, M2]` で外せる)
- 部品としての値 (16.94): 入力と出力は文 (PLaMo のトークナイザー、53 言語)、常駐は 1.38 GiB (ANE) で 2 回目の読み込み 2.6 秒、ストリーミングで出せる、最初のトークンまで 33 トークンのプロンプトで 0.1 秒

### 16.99 Qwen3.5-0.8B と 2B (`Qwen3.5-0.8B`、`Qwen3.5-2B`)、Qwen3.5 と PLaMo 3 のロジットを最後の行だけ (`LastRow`)、小さいモデルは既定で考えない (`Model.NoThink`) — 2026-11-07

- Qwen/Qwen3.5-0.8B と 2B (2026-02-28、Apache-2.0、画像と文の `Qwen3_5ForConditionalGeneration`) は 4B (16.88) と同じ構造で、`internal/llm/qwen35` が config からそのまま読む (24 層、3 層の Gated DeltaNet に 1 層の全注意、全注意は 3、7、…、23 の 6 層、語彙 248320 で出力の行列は埋め込みと共有)。4B と違うのは大きさと、GDN の値のヘッドが鍵のヘッドと同じ数 (16 と 16。4B は 32 と 16) なこと。トークナイザーは 4B とバイト単位で同じ。チャットのテンプレートは考える部分の既定だけが違う (0.8B と 2B は `enable_thinking` が true のときだけ考える。4B は false のときだけ考えない)
- **Hugging Face の fp32 との照合** (4B ではしていなかった。この Mac の scratchpad に torch 2.15 と transformers 5.22 の venv を作り `tools/ref/ref_logits.py --seq 32`、日本語と英語の 32 トークン): `hwnpu-check` で `tensors: 0 missing, 0 mismatched, 168 unused` (視覚の塔と MTP の層)。fp32 で各層の相対 RMS 誤差 0.0002 以下、全体で argmax 32/32、KL 0.0000 nats。`-f16` で全体 0.0014、argmax 32/32、KL 0.0000。lut8 は KL 0.0076 nats (argmax 32/32)、lut6 は 0.089 (24/32)。既定は 4B と同じ lut8
- M3 の Neural Engine (0.8B、lut8、注意で分けた 7 部分 0.07〜0.26 GiB): 2 本 (8 トークン/回と 128 トークン/回) とも `all on the NPU`。重みは 2 本で 1.44 GiB (`onePrograms` はキャッシュに収まるとみて 2 本にする)。Neural Engine に 1.04〜1.47 GiB、KV は 1 トークン 14 KiB (全注意 6 層 × KV ヘッド 2 × 256)
  - 速さ: 生成は 1 秒に 27〜30 トークン (4B は 7.6〜8.9)。2000 トークンのプロンプトを 2.6 秒で読む
  - **初回のコンパイルが重い**: 128 トークン/回のプログラムの Core ML のコンパイルが 15 分 46 秒 (8 トークン/回は 56 秒)。次の起動でも一度 4〜6 分のコンパイルがあり (バイナリを作り直した後にも)、その後は 0.7 秒と 1.6 秒で読める。4B で Q=1 と Q=64 の組が 24 分終わらなかった (16.88) のと同じく、GDN の層を大きな Q でコンパイルするのが重い。ロジットを最後の行だけにしても (下) 14 分 17 秒で、出力層のせいではない
- 答え (16.26 の 4 問、貪欲、考えない): 17 × 23 = 391 は正しい。富士山は 3,778 m、『こころ』の作者は「小野寺 明」、日本一長い川は「長谷川」と誤る。**fp16 でも同じ答え**で、HF との照合も合うので、量子化や移植ではなく 0.8B の知識の限界。日本語の事実を問うチャットには勧めない
- **考える部分**: 考えさせると 4 問とも考える予算 (1024 トークン) で打ち切られ、同じ言葉を繰り返した (「答え：391」を十数回)。テンプレートの既定どおり、0.8B と 2B は既定で考えないようにした。`chat.Model.NoThink` (カタログ) を `Engine.NoThink` に写し、`-think` をコマンド行で与えなければそれに従う (hwnpu-chat と hwnpu-agent)。API の `reasoning_effort` と `enable_thinking` はそれより優先する (`Server.thinks`、`TestServerThinks`。`NoThink` を見ないようにすると落ちる)
- **ロジットを最後の行だけ (`LastRow`、#56 の最適化) を Qwen3.5 と PLaMo 3 にも**: プロンプトを読むプログラムは最後のトークンのロジットしか使わないのに、Qwen3.5 と PLaMo 3 は 128 行すべてで出力層を計算していた (Qwen3.5 は 128 × 248320 で 60.6 MiB の出力、ANE のテンソルの目安の 64 MiB に近い)。LFM3 と同じく最後の norm の前で `last` (1, 1, seq) の one-hot を掛けて 1 行にする。`TestPagedLastRow` (qwen35、plamo3): 1 トークン/回のプログラムと状態を共有して 4、3、1、2、1 トークンずつ進め、キャッシュなしのプログラムと相対誤差 6.2e-4 と 5.1e-4。`LastRow` を通さないと出力の形の検査で落ちる。プログラムの名前に `-last` が付くので、0.8B (PLaMo 3、Qwen3.5) の 128 トークン/回のプログラムは変換し直しになる。PLaMo 3 Translate 2B と Qwen3.5-4B は 1 本 (`-prefill auto`) なので変わらない
  - M3 で: PLaMo 3 Translate 0.8B の 128 トークン/回のプログラムを `-last` で変換し直し、20 文の訳は前と **20/20** 一字一句同じ。1629 トークンの文書を読む時間が 1.6 → **1.2 秒** (−25%)。初回のコンパイルは 20 秒。Qwen3.5-0.8B は 2000 トークンを 2.6 秒で読む (前の形では測っていない)
- **Qwen3.5-2B** (M3、lut8、7 部分 0.22〜0.52 GiB、すべて `all on the NPU`):
  - 答え (同じ 4 問、考えない): 富士山 3,776 m、17 × 23 = 391、日本一長い川は信濃川 (「川」の定義の前置きつき)。『こころ』の作者は「松本零士」と誤る。0.8B (1/4) より良いが、LFM3-1.7B や Gemma 4 E2B と並べてはいない
  - 速さ: 生成は 1 秒に 16〜20 トークン
  - **2 本では起動のたびにコンパイル**: 2 本の重みは 3.56 GiB で `onePrograms` の目安 (4 GiB) より小さいので 2 本になったが、128 トークン/回のプログラムの初回のコンパイルが 13 分 24 秒、そのあとの起動でも毎回 4 分 15 秒〜5 分 10 秒かかった (ANE のキャッシュに残らない)。`-prefill off` の 1 本なら 2 回目から 1.5 秒で読み込め、2000 トークンを 11.9 秒で読む (2 本なら 3.0 秒)。目安 (`compileCacheBytes`) を 3.5 GiB に下げ、2B は 1 本にした。見積もった 2 本の重みは LFM3-1.7B 2.43、LFM3-VL / Audio / Omni 3.26、Gemma 4 E2B 3.20、Qwen3.5-0.8B 1.44、PLaMo 3 Translate 0.8B 1.37 GiB で、変わるのは Qwen3.5-2B だけ (3.26 と 3.20 の 2 本が毎回のコンパイルにならないかは確かめていない)
  - メモリ: Neural Engine に 1.78 GiB、プロセスの peak footprint 1.5 GB。0.8B は 2 本で 1.47 GiB と 1.1 GB
- MODEL.md: Qwen3.5-0.8B と 2B をメモリの表に足した。チャットの欄のおすすめには入れない (日本語の事実を誤る。LFM3-1.7B、Gemma 4 E2B との比較が先)
- 確かめていないこと: Qwen3.5-2B の HF との照合 (0.8B と同じコードで、0.8B は照合済み)、hwnpu-agent の課題、長い文脈の passkey、AMD と Intel、Qwen3.5 の GDN を 64 トークン/回などにしてコンパイルが軽くなるか
- 部品としての値 (16.94): 入力と出力は文 (Qwen のトークナイザー、多言語)、道具の呼び出しあり。常駐は 0.8B 1.47 GiB、2B 1.78 GiB (ANE)。2 回目の読み込みは 0.8B 2.3 秒 (2 本)、2B 1.5 秒 (1 本)。**0.8B の初回は 15 分**のコンパイルがあるので、ほかのモデルと組むときは先に一度起動しておく

### 16.100 Whisper (`whisper-encoder`、`whisper-cross`、`whisper`) と hwnpu-transcribe — 2026-11-08

- openai/whisper-large-v3-turbo (809M、MIT) を移した。modeling_whisper.py の `WhisperForConditionalGeneration` を 3 つのプログラムに分けた (`internal/llm/whisper`)
  - **`whisper-encoder`** (`-arch`): mel (1, 3000, 128) → 畳み込み 2 つ (kernel 3、2 つ目はストライド 2) と GELU → 位置埋め込み → pre-norm の 32 層 → LayerNorm → (1, 1500, 1280)。畳み込みは時間方向の 3 フレームを特徴に並べて Linear 1 つで掛ける (ストライド 2 は `reshape` で 2 行ずつまとめる)。注意のスコアは 20 ヘッド × 1500 × 1500 で 90 MB になるので、クエリを 500 行ずつ (`qchunk`、30 MB) に分けていたが、速くならなかったので既定は分けない (16.115)
  - **`whisper-cross`** (`-arch`): エンコーダの出力から、デコーダの各層のクロス注意の K と V を一度に (1, 160, 1500, 64)。ホストがそのまま各層の `xk_N` / `xv_N` に分ける
  - **`whisper`** (model_type whisper、`llm.Cached`): デコーダ 4 層。自分の KV キャッシュ (448)、クロスの K/V を入力 `xk_N` / `xv_N` に、学習済みの位置埋め込みを one-hot の入力 `pos` (1, q, 448) と表の行列積で (gather を避ける)。出力層は埋め込みと共有で、語彙 51866 は 16 で割り切れないので、lut の表のために 0 の行で 51872 に詰めて掛け、語彙の分を切り出す (`blocks.LinearPadded`、新しい部品で、ほかのモデルの出力は変わらない)
  - `decode.Session.SetInputF16`: キャッシュでも `StepInputs` でもない入力を、ホストが一度書いてステップ、印、巻き戻しの間そのままにする (クロスの K/V)
- 合格条件 (port-architecture):
  - 合成のモデル (幅 32、4 ヘッド、エンコーダ 2 層、デコーダ 2 層、6 位置) で modeling_whisper.py から書いた fp64 の forward と 1e-4 以内: エンコーダ (クエリを分けない、4 行ずつ、1 行ずつ、gelu_new)、デコーダ (gelu、gelu_new、scale_embedding)、1 層ずつと最後の層の後だけ。キャッシュつきの 1 トークン/回と 3 トークン/回を `decode.Session` で混ぜ、印と巻き戻しも、キャッシュなしと相対誤差 3e-4
  - わざと壊すと落ちる: エンコーダ (畳み込みのタップを逆に、位置埋め込みなし、クエリのスケールなし、残差の足し先)、デコーダ (因果マスクなし、別の層のクロス K、位置を 1 ずらす、マスクで次の位置を見せる、残差の足し先、定数の位置をずらす)
  - 実の重みで `hwnpu-check` (日本語 6.5 秒と英語 5.2 秒の `say` の音声。`tools/ref/ref_whisper.py` が参照を書く): エンコーダは fp32 で各層 0.0000、全体のコサイン 1.000000。デコーダは日本語 29 トークン、英語 22 トークンで argmax 全一致、KL 0.0000 nats (`-f16` でも)。クロスは 8 つとも 0.0000 (`-f16` で 0.0002)。`hwnpu-check` は参照に `REF.x.bin` があればそれを層 0 からのプログラムの入力にする (mel を読む音声エンコーダのため)
  - **エンコーダの `-f16` は全体でコサイン 0.9978** (相対誤差 0.066、0.999 未満の位置 172)。層 20 が音声の後の無音の位置 (538、564〜566、971、972、1455、1456) に 302 という値 (注意の吸い込み先) を作り、入力のわずかな誤差がそこで増幅されて、注意を通じてほかの位置 (とくに音声の直後) に広がる。**Hugging Face を fp16 で動かしても同じ** (相対誤差 0.0664、コサイン 0.997797、0.999 未満 176。bf16 では 0.970) で、fp16 で動かすこと自体の誤差。チェックポイントも fp16 で配られている。書き起こしは下のとおり fp32 と同じ
- ホストの側:
  - 特徴 (`whisper.Features`、`FeaturesLong`): WhisperFeatureExtractor と最大差 2.7e-6 (`tools/ref/whisper_mel_golden.py`、`testdata/mel.json`)、30 秒で 70 ms。長い音声は全体を 1 回で計算して、最大値からの下限も全体で取る (HF の長い音声と同じ。窓ごとに計算すると区切りが変わる)
  - トークナイザー: GPT-2 の前処理 (ByteLevel の `use_regex`) を足した。固定の 40 文とランダムな 3040 本で HF と 0 件の違い (`TestWhisper`、`HWNPU_WHISPER_DIR`)
  - `internal/transcribe`: 30 秒以下は `<|notimestamps|>` で 1 回。長い音声は transformers の長い音声の生成と同じく、タイムスタンプつきで窓ごとに読み、最後に閉じた区切りの終わりから次の窓を始める (`timestampRules` は `WhisperTimeStampLogitsProcessor`、`segments` は `_retrieve_segment` と同じ規則。`TestTimestampRules`、`TestSegments`、4 通りの壊し方で落ちる)。言語の指定がなければ `<|startoftranscript|>` の後のロジットで選ぶ (`detect_language`)
  - `cmd/hwnpu-transcribe`: 初回にダウンロードと 3 つの変換。`-language`、`-translate`、`-timestamps`、`-v`
- M3 (16 GB、macOS 27.2、lut8): 3 つとも全部 Neural Engine (エンコーダ 1075 演算、クロス 25、デコーダ 165)。変換は 45 秒、初回の読み込み 38 秒、2 回目から 0.6 秒。Neural Engine に 0.78 GiB、プロセスの peak footprint 1.1 GB
  - 速さ: 30 秒の窓ごとにエンコーダとクロスで 0.6 秒、生成は 1 秒に 160〜180 トークン。6.5 秒の音声を 0.86 秒 (実時間の 7.6 倍)、65.8 秒を 4.0 秒 (16 倍)
  - 短い音声は日本語も英語も HF の fp32 と一字一句同じ。65.8 秒の日本語 (富士山の段落) は、30 秒で機械的に切ると 2 つ目の窓が「山小屋」だけになった (HF でも同じ)。タイムスタンプの方式にして全文が取れ、区切りの時刻も HF とほぼ同じ (最初の終わりが 4.76 と 4.74)。違うのは「信仰の大将」(HF の長い音声の方式は「大賞」) の 1 か所。45.4 秒の英語は文は同じで、1 つ目の窓の区切り (hwnpu 29.76 秒、HF 25.98 秒) が分かれる
  - 形式: fp16 でも区切りは lut8 と同じで、書き起こしは同じか句読点が揺れる程度。fp16 は 2 部分 (1.2 GiB) で生成が半分ほどの速さ (56〜127 トークン/秒) なので、既定は lut8
- kotoba-tech/kotoba-whisper-v2.2 (日本語向けの蒸留、デコーダ 2 層、Apache-2.0、fp32 のチェックポイント 3.0 GB) も同じ実装で `-use kotoba-tech/kotoba-whisper-v2.2`。M3 で Neural Engine に 0.73 GiB、生成は 1 秒 270〜290 トークン (turbo の 1.6 倍)。6.5 秒の日本語は HF と一字一句同じ (「天気ですね、」と句点が読点)。65.8 秒の日本語は崩れる (「活か山」「世界文化産」) が、**HF の長い音声の方式でもほぼ同じ文に崩れる**ので、モデルの性質 (kotoba は自前のチャンク分割の pipeline で使う前提)。長い音声には turbo を勧める
- 確かめていないこと: 実際の録音 (雑音、話者の重なり)、AMD と Intel、温度を上げてのやり直し (OpenAI の圧縮率と平均対数確率の閾値)、前の窓の文を前置きにする (`condition_on_prev_tokens`。この 65 秒では HF でも変わらなかった)、hwnpu-chat の API (`/v1/audio/transcriptions`)
- 部品としての値 (16.94): 入力は 16 kHz モノラルの音声 (WAV はほかのレートとチャンネルも読む)、出力は文と区切りの時刻。常駐は 0.78 GiB (ANE)、2 回目の読み込み 0.6 秒。30 秒ごとに 0.6 秒のエンコーダが先に要るので、ストリーミングでは窓の長さだけ遅れる

### 16.101 AMD: Qwen3.6 のプロンプトの MoE のホストの手順を速く、専門家と出力の行列を GPTQ で (`qwen36_gptq.py`) — 2026-11-08

- プロンプトのホストの手順 (32 トークンのプロンプト 1.62 → 1.43 s、デコードは 172 ms/トークンのまま、出力は同じ):
  - 専門家の x は本物の行 (上位 8 × トークン) の分だけ書き、出力も本物の行だけ読む。埋め草の行はコアが読み流して 0 を出す
  - デバイスのバッファは少しずつ読むと遅いので、まとめて写してから読む (`readRegion` は写しを返す。畳み込みの履歴はその場で書き戻すので `viewRegion`)
  - 上位 8 個は 256 個の並べ替えでなく選び出し (`topK`、同じ値は小さい番号を先に。安定な並べ替えと同じ順を `TestTopK` で)
  - `HWNPU_IRON_TIMING=1` でホストの手順と MoE のコマンドの時間も出す
- 見送ったこと:
  - **本物の行だけで GEMM**: `ProjectionGEMM` の `valid` は 256 行単位 (M_TILE × 行 × m_chunk) なので 32 行でも速くならない (q_linear 8192×2048 で 32 行と 256 行が同じ 2.4 ms)。行ごとに GEMV を回すと 32 行では遅い (1 行 0.29〜0.38 ms)
  - **専門家のコマンド (B)** は重みを読む速さの下限: shim の MM2S は 2 チャネルで 1 つを x が使い、重みは列ごとに 1 チャネル約 4.4 GB/s、全体で約 36 GB/s
  - **塊ごとの重みの展開をやめる** (bfp16 を前もって持つ。`DequantBFP` は 1 回 1.4 ms で、`reference` がビット一致するので作れる): 固定のメモリが増えるので見送った
- GPTQ (docs/amd-npu-models.md 13 の「Qwen3.6-35B-A3B」): 1 層ずつ safetensors から読み、校正の文の隠れ状態だけを持って進める (fp32 の全体 140 GB は入らない)。専門家ごとに経路が送ったトークンで H を作り、32 個ずつまとめて GPTQ。40 層と出力の行列で 4.3 時間、符号 18 GB。書き出しは `--q4-codes` で 19,263 個が当たる
  - CPU の KL (bf16 の重みとの): 0.0243 → **0.0185** (中央値 0.0162 → 0.0109)、最上位の一致 91.6 → 93.1 %。3 つの評価の文 (wikitext-2、wiki40b-ja、README) のどれも良くなる
  - NPU の `ironlm`: HF とのコサイン 0.9850〜0.9959 → 0.9894〜0.9974、最上位は 9 回とも一致。基準 (すべて 0.99 以上) には最後の 1 回 (0.9894) が届かず FAIL のまま
- 確かめていないこと: 長い文脈、チャットの答えの比較、校正の足りない専門家 (後の層で最大 44 個) をほかの文で埋めること、mixer (q8) の GPTQ
- 部品としての値 (16.94): 変わらない (40 層 23 GB、デコード 5.8 tok/s)。書き出しの前に GPTQ の 4.3 時間 (一度だけ、符号は 18 GB) が要る

### 16.102 AMD: Hugging Face の fp32 との照合 (Gemma 4 E2B、PLaMo 3、4 ビット)、Anima の q8 を多くの種で — 2026-11-08

- 道具: `tools/ref/forced_logits.py MODEL CKPT RUN.bin...`。`ironlm -forced` が書いた NPU のロジット (`RUN.bin`) と同じトークン (`RUN.bin.ids`) を Hugging Face の fp32 で 1 回に流し、同じ行 (最初の 256 トークンの後と、その後の 1 トークンずつ) のロジットを `RUN.bin.hf` に書いて KL(HF || NPU) を出す。モデルの読み込みは `q4_gptq.py` の `load` (文の部分だけ、fp32)。11 月 6〜7 日に GPTQ の評価で書いた NPU のロジット (`~/models/q4-gptq/npu`、README / GPL-3 / wiki40b-ja の 400 行ずつ) をそのまま使った
- **Gemma 4 E2B** (11.3 節の「HF のロジットと照合していない」): bf16 の書き出しと HF の fp32 で KL 0.0059 / 0.0056 / 0.0038、最上位の一致 97.8 / 98.2 / 97.8 % (README / GPL / wiki)。LFM3 と E4B と同じく合う。16.50 で乱数の重みの照合が合わなかったのは移植の誤りではない (原因は 16.106)
- **4 ビットの KL を HF と** (11.3 節。これまでは NPU の bf16 との KL だけ):

| | README | GPL-3 | wiki40b-ja |
|---|---|---|---|
| E2B bf16 | 0.0059 | 0.0056 | 0.0038 |
| E2B q4 丸め | 0.123 | 0.171 | 0.220 |
| E2B q4 GPTQ | 0.091 | 0.145 | 0.081 |
| PLaMo 3 2B bf16 | 0.0040 | 0.0118 | 0.0022 |
| PLaMo 3 2B q4 丸め | 0.097 | 0.093 | 0.055 |
| PLaMo 3 2B q4 GPTQ | 0.044 | 0.088 | 0.040 |

| E4B bf16 | 0.0138 | 0.0040 | 0.0031 |
| E4B q4 丸め | 0.047 | 0.089 | 0.085 |
| E4B q4 GPTQ | 0.025 | 0.077 | 0.051 |

  bf16 の書き出しが HF に近いので、HF との KL は NPU の bf16 との KL (amd-npu-models.md 13 節) とほぼ同じになる。E4B の README だけは bf16 の書き出しと HF の差が数か所の外れで大きく (平均 0.0138、中央値 0.0003)、HF と比べると GPTQ は 0.047 → 0.025 で CPU の見積もり (0.057 → 0.016) に近い。13 節の「E4B の README は NPU で改善が小さい」は、NPU の bf16 を基準にしたためだった (E4B は fp32 で 30 GB を超えるので、ほかの校正が終わってから CPU 16 スレッドで測った)
- 丸めの修正 (16.78) の後の 4 ビットの書き出し (11.3 節): GPTQ の書き出し (#62) と E4B の `-q4-rtn2` はどれも 16.78 の後に作ったもの。古いのは多モーダルの丸めの書き出し (`-q4`) だけで、既定の GPTQ の書き出しが置き換えた
- **Anima の q8 を多くの種で** (16.78 の「4 つの種の 1 つで構図が変わった」): 1024 × 1024、3 つのプロンプト (EVAL と同じ) × 種 1〜6 の 18 組を NPU の bf16 と q8 で描いた。最後の潜在の cosine は 0.915〜0.986 (中央値 0.957)、8 × 8 に平均した低い周波数では 0.981 以上。並べて見て、18 組のどれも構図 (人物の数と位置、窓と猫、騎士の向き) は同じで、変わるのは服や髪の細部だけだった。16.78 の 1 件は diffusers の fp32 との比べ方 (軌跡が分かれやすい種) によるもので、NPU の bf16 と q8 の間では起きなかった

### 16.103 AMD: PLaMo 3 Translate 0.8B — 2026-11-08

- `iron_export_lm.py plamo3` がそのまま書き出す (KV ヘッド 1、出力の行列は埋め込みと別。16.98 と同じくアーキテクチャの側で済む)。文脈 8192、塊 1024。bf16 は固定 1814 MiB、`--q4` (MLP と出力の行列 q4nx、注意 q8、埋め込みはホスト) は 599 MiB
- HF の fp32 と (16.102 の道具): bf16 0.0046 / 0.0038 / 0.0031 (最上位 97〜98 %)、q4 の丸め 0.095 / 0.094 / 0.094、**GPTQ** (`q4_gptq.py plamo3`、CPU で 28 分、33 個が当たり 32 個は q8 の丸め) 0.043 / 0.061 / 0.072
- 訳文 (16.98 と同じ 20 文、貪欲): q4 の丸めは bf16 と 19/20 が一字一句同じ ("because of" と "due to")、**GPTQ は 20/20**。0.8B の bf16 と 2B の bf16 は 11/20 が同じで、残りも正しい言い換え
- 速さ (README の 256 トークンの後に 399 ステップ、`ironlm -forced`、ほかの校正が CPU を使う中): bf16 57 ms/ステップ (17.5 トークン/秒)、q4 44 ms (22.8)。2B の q4 GPTQ は 89 ms (11.2)、bf16 143 ms (7.0)
- 書き出し: `iron_export_lm.py plamo3 CKPT OUT --max-seq-len 8192 --prefill-chunk 1024 --q4 --q4-codes ~/models/q4-gptq/plamo08-gptq/codes`

### 16.104 AMD: Qwen3.5-0.8B と 2B (GDN の列の埋め、密なモデルのコマンドを 24 語で、埋め込みをホストに、6 ビットの MLP) — 2026-11-08

- **融合した射影が 64 行で割り切れない**: 0.8B と 2B は GDN の値のヘッドが鍵のヘッドと同じ数 (16 と 16) で、`gdn_layout` の 1 列の幅が 2 × 2 × 128 + 2 × 130 = 772、8 列で 6176 行になり、q8 の GEMV (出力 64 ずつの列ブロック) に載らなかった (4B は 1032 × 8 = 8256)。`iron_gdn.column_widths` が 1 列の幅を 8 の倍数に切り上げる (772 → 776、全体 6208)。埋めた行は重みが 0 の行 (`gdn_layout` が「最後の行の 1 つ先」を指し、`padded` が 0 の行を足す)、畳み込みのタップも 0。演算子は q、k、v、a、b の位置を列の中の先頭から数えるので、列の幅が変わるだけ。4B の並びは前と同じ (並びの配列が一致)
- **2B の q4 のプロンプトがデバイスのヒープに載らない**: 塊が 3 コマンドになり、2 つめの制御コードが 64 MiB を超えて `CREATE_BO: invalid argument`。IRON はスクラッチパッドの語 (32) でしかコマンドを切らない。PLaMo 3 と LFM3 と同じ 24 語にした (`iron_qwen35.install_dense`、密な Qwen3.5 だけ。Qwen3.6 の経路の選択は 32 語の前提 `iron_moe.SLOTS` なので変えない)。2B の q4 は 13 コマンドで通る
- HF の fp32 と (16.102 の道具、BOS の代わりに `<|endoftext|>` 248044 を先頭に): **bf16** 0.8B 0.0014 / 0.0056 / 0.0012、2B 0.0009 / 0.0017 / 0.0007 (最上位 96〜99 %)。移植は正しい
- **4 ビットでは GPL が崩れる**: 0.8B の q4 は README 0.055、wiki 0.065 だが GPL 0.336 (95 % 点 1.44)。位置によらず散らばる。出力の行列を q8 (`--head q8`) にしても 0.330、注意と GDN の射影を bf16 (`--attn bf16`) にしても 0.324 で、誤差は MLP の 4 ビット。GPTQ (`q4_gptq.py qwen35` を足した。CPU で約 30 分) でも CPU の見積もりで README 0.022、wiki 0.032 に対し GPL 0.300 のまま。2B の q4 も GPL 0.163
- **6 ビット (q6、16.105) の MLP**: `iron_export_lm.py qwen35 --q4 --mlp q6 --head q6`。CPU の見積もり (丸め) で 0.8B 0.0055 / 0.034 / 0.0068。NPU で HF と 0.8B **0.0079 / 0.044 / 0.0077** (最上位 93 %)、2B **0.0047 / 0.019 / 0.0062** (96〜97 %)。小さい Qwen3.5 は q6 を勧める
- **埋め込みをホストに** (`Qwen35.embed_on_host`、`--q4` のとき。PLaMo 3 と LFM3 と同じ `embed.bin` と `embed_rows`): 出力の行列は埋め込みと共有だが、`quantize` が量子化した写しを持つので、bf16 の表はデバイスで使われていなかった。ロジットは前とビット単位で同じ。固定のメモリ:

| | bf16 | q4 (前) | q6 + ホストの埋め込み |
|---|---|---|---|
| 0.8B | 1858 MiB | 1460 MiB | **1069 MiB** |
| 2B | 4017 MiB | 2730 MiB | **2014 MiB** |

  Qwen3.6 (`Qwen35Moe`) は経路の選択が埋め込みを読むので変えない (`host_tables = False`)。`moe_body` は `CausalLM.body` を直に呼ぶので、足した `body` も通らない
- 速さ (400 ステップ、文脈 256〜656): 0.8B は bf16、q4、q6 とも約 86 ms/ステップ (11.6 トークン/秒)。24 層の演算の数で決まり、重みの量は効かない。2B は bf16 131 ms、q4 110 ms、q6 123 ms。Apple の M3 (16.99: 0.8B 27〜30、2B 16〜20 トークン/秒) より遅い
- 答え (16.99 と同じ 4 問、q6、考えない): 0.8B は 391 が正しく、富士山を 3,778 m、日本一長い川を「長谷川」と誤る (16.99 の Apple の fp16 と同じ誤り)。2B は 391、3,776 m、信濃川 (長さは誤る)。`hwnpu-chat -iron` は書き出しのディレクトリを読むので、カタログの `NoThink` は効かない (要求で `enable_thinking: false` を渡した)
- 書き出し: `iron_export_lm.py qwen35 CKPT OUT --max-seq-len 4096 --prefill-chunk 256 --q4 --mlp q6 --head q6`

### 16.105 AMD: 6 ビットの重み (q6) — 2026-11-08

- q8 と同じ並び (ブロックは出力 32 × 入力 256、入力 32 ごとに bf16 の scale、符号は 16 出力ずつの帯で列優先) で、符号を 6 ビットにした。符号は −31〜31 を +32 して 2 つの面に置く: 下位 4 ビットは q4nx と同じニブル、上位 2 ビットは 128 個ごとに 32 バイト (128 個の中の 32 j + i 番目をバイト i のビット 2 j、2 j + 1 に)。1 ブロック 6656 バイト、1 つの重み 6.5 ビット (q8 8.5、q4nx 5)。scale は入力 32 個の最大の絶対値 / 31
- `kernels/q6_mv.cc`: q8_mv.cc の符号の読み方だけを変えた。128 個を、下位を `aie::unpack` (uint4 → uint8)、上位を 32 バイトから `downshift` と `bit_and` で 4 つ取り出して `concat`、`bit_or` で合わせ、32 を引いて int8 にする。部分和を 2 つの bf16 に分けて scale を掛けるのも q8 と同じ。プロンプトは `q6_dequant` で bf16 に戻す (`QDequant`)
- `iron_q4.py check --fmt q6`: GEMV (1536 × 1536、x の平均 1) コサイン 1.000000、プロンプトの経路 (256 行) 0.999956 (q8 も 0.999955)。`--valid` を付けると q8 も同じく合わない (照合の側の問題で、q6 とは別。直していない)
- 速さ (8192 × 2048 の GEMV): q4 0.30〜0.35 ms、q6 0.42〜0.47 ms、q8 0.43〜0.50 ms。q6 は q8 とほぼ同じ時間で、上位 2 ビットを取り出す計算で律速されている (32 GB/s)。得るのはメモリ (q8 の 76 %) と精度 (q4 の誤差の 3 分の 1: 相対 RMS q4 0.071、q6 0.022、q8 0.005)
- 使える所: `iron_export_lm.py qwen35 --mlp q6 --head q6` (16.104)、`iron_anima.py --dit-attn q6 --dit-mlp q6` (16.106)、`q4_gptq.py --fmt-q4 q6 --head q6` (CPU の見積もりと GPTQ。対称な形式の符号の範囲は `SYM`)

### 16.106 AMD: Anima の DiT を 1 つの bank で回す、DiT の q6 と GPTQ の q4 を 1024 × 1024 で、512 × 512 — 2026-11-08

- **DiT の 7 組を 1 つの bank で** (amd-npu-models.md 12 節の「DiT の作業領域を 7 組で 1 つに」より大きく減らす): DiT の 28 ブロックは 4 ブロックずつ 7 組の bank (`variants`) で、組ごとに 464 MiB (重みの像 304 MiB と作業領域) をすべて固定していた。bank の中は IRON の arena が重みを先に、作業領域を後ろに置き、1 回の呼び出しの後に作業領域に残すものはない。そこで bank を 1 組だけ確保し、組を切り替える (`Graph.Use`) たびにその組の像を bank の先頭に読む (`iron.GraphOptions.SwapVariants`、ページキャッシュから 304 MiB)。hwnpu-image の既定にし、`-resident-dit` で前の形にできる
- 1024 × 1024、q8: 固定されたメモリの最大 **3445 → 1236 MiB** (残りの最大は VAE の段)。ノイズ除去 59 → 60 秒 (+2 %)、最後の潜在はバイト単位で同じ。`TestGraphSwapVariants` (sim): 像を読む、作業領域を残す、読みに失敗したら次は読み直す
- **DiT の q6** (`iron_anima.py --dit-attn q6 --dit-mlp q6`): 16.102 と同じ 18 組で、NPU の bf16 との潜在の cosine の中央値 0.912 (q8 0.957)、低い周波数で 0.950 以上、構図は 18 組すべて bf16 と同じ。1 ステップの時間は q8 と同じ。固定 (1 つの bank の前、全組常駐で) 3445 → 3025 MiB。既定は q8 のまま
- **GPTQ の 4 ビットの DiT を 1024 × 1024 で** (16.87 の残り): `iron_anima.py --dit-attn q4 --dit-mlp q4 --q4-codes ~/models/q4-gptq/anima-gptq/codes` (280 個すべて当たる)。同じ 18 組で中央値 0.860、最小 0.686、騎士のプロンプトの種 1 で騎士が 1 人から 2 人に、種 2 で城の形が変わった。q4 は既定にしない (q6 か q8)
- **512 × 512** (`iron_anima.py --size 512`): 書き出し約 10 分。1 枚 20 秒 (ノイズ除去 1.6 秒/ステップ、VAE 3.1 秒)、固定の最大 1111 MiB。猫のプロンプトで正しい絵

### 16.107 AMD: 乱数の重みで IRON の参照と合わない理由、プロンプトが IRON の Llama より遅い理由 — 2026-11-08

- **乱数の重みでの不一致** (16.50、16.70。improvements の 11.4 節): PLaMo 3 Translate 0.8B を `--random-weights 1 --check 32` で書き出し、今の hwnpu で照合した。プロンプトの塊から読むと、最上位のロジットのコサインは 0.884〜0.903 (16.50 の Gemma 4 E2B は 0.55〜0.83 で、35 層あるのでさらに広がる)。`ironlm -decode-only` (プロンプトもデコードのステップで読む) では 0.978〜0.988 だった
  - CPU だけの比べ (`tools/ref/random_sensitivity.py`: 同じモデルを fp32 と、重みと計算をすべて bf16 にしたもので流す): 本物の重みはロジットのコサイン 0.99983、最上位 100 %。IRON と同じ乱数の重み (行列は標準偏差 1/√入力数、ベクトルは 1 + 正規/10) では、層ごとのコサインが 0.9998 から 0.9881 まで少しずつ下がり、ロジットのコサイン 0.988、最上位 63 %。乱数の重みではロジットが平らで、最上位と 2 位の差の中央値が 0.23 しかない (本物の重みは 1.51)
  - デコードの経路の差 (0.985) は、この bf16 の丸めの差と同じ大きさ。プロンプトの経路はさらに開く (0.89): IRON の言語モデルのプロンプトの GEMM はすべて bfp16 (`Precision.MEDIUM`、8 個で指数を共有) で、その丸めも拡大される。本物の重みでは、PLaMo 3 0.8B の README で HF との KL がプロンプトから読んで 0.0046、ほぼデコードだけで読んで (`-split 8`) 0.0049 と変わらない
  - 結論: 乱数の重みは丸めを拡大するので、照合には使えない。移植の誤りではない (本物の重みで HF と合う: 16.102)
- **プロンプトの内訳** (11.4 節の「IRON の Llama 3.2 1B より遅い」): Gemma 4 E2B の q4 GPTQ で、位置 0 からの 1024 トークンの塊は 1.16 秒 (Llama 1B は 0.47 秒)。ホストの時間は入力とホストの表 8 ms、塊の前の空のコマンド 0.3 ms、ロジット 0.4 ms で、ほぼすべてがデバイスの計算 (以前の 1.51 秒はほかの仕事と重なったもの。3 回とも 1.16〜1.17 秒)
  - 内訳の測り方: `iron_export_lm.py --command-per-step` (プロンプトの塊を手順ごとのコマンドにし、`steps.json` に各コマンドの演算を書く。ほかの差し替え (`install_tail_boundary` など) の後に入れる) で 1249 コマンドの書き出しを作り、`HWNPU_IRON_TIMING` で位置 0 の塊を 1 つだけ流した (16.80 の「次の塊の最初のコマンドが止まる」を避けて、2 つ目の塊は流していない)。分けても塊は 1.4 秒で、上乗せは小さい
  - 位置 0 の 1279 ms のうち **MHA が 767 ms (60 %)**: 窓の層 (d 256、窓 511、キー 1536) 1 つ 45〜48 ms × 12 層、全体を見る層 (d 512、キー 4096 の確保) 1 つ 69 ms × 3 層。続いて射影の GEMM 171 ms (13 %)、RMSNorm 76 ms (276 回)、`QGEMV` 52 ms (共有の層の最後の行と出力層)、`DequantBFP` 40 ms、要素ごとの演算と Copy が合わせて約 130 ms
  - どの MHA も `num_pipelines1` だった。IRON の Llama は `profiles/npu2.json` でプロンプトの MHA を 8 本のパイプラインにしているが、hwnpu の書き出し (IRON の Gemma 4 も、hwnpu の PLaMo 3、LFM3、Qwen3.5 も) はプロファイルを持たないので、MHA の既定の 1 本で動いていた (Anima は 16.68 で 8 本を渡している)
- **プロンプトの MHA を複数のパイプラインに** (`iron_export_lm.py --mha-pipelines`、既定 8): 書き出しで、プロンプトの塊の行数 (`seq_pad` = `prefill_chunk`) の MHA に `num_pipelines` を与えるプロファイルを付ける (`prompt_attention_profile`)。MHA はパイプラインごとに 64 行の整数回を取り、6 本を超えるときは偶数なので、塊に合う本数まで減らす (Qwen3.5 の塊 256 は 4 本)。デコードの MHA は変えない。パイプラインはクエリのブロックを分けるだけなので、**ロジットはビット単位で同じ** (E2B、PLaMo 3 0.8B、Qwen3.5-0.8B の README / GPL / wiki の 400 行と、E2B と LFM3 の 3000 トークンのプロンプト (2〜3 塊) の後の 20〜40 ステップ)。Qwen3.6 (`qwen35moe`) は測っていないので 1 本のまま
- 塊の時間 (Ryzen AI 9 365、`ironlm -depth`):

| | 塊 | 位置 0 | 文脈の終わり近く | 電力量 (位置 0 / 終わり近く) |
|---|---|---|---|---|
| Gemma 4 E2B q4 | 1024 | 1.16 → **0.50 秒** | 3072 から 2.53 → **0.64 秒** | 13.7 → 4.7 J / 49.6 → 8.6 J |
| PLaMo 3 Translate 0.8B q4 | 1024 | 1.12 → **0.46 秒** | 7168 から 3.73 → **0.77 秒** | |
| LFM3-1.7B q4 | 2048 | 2.73 → **1.30 秒** | 10240 から 11.39 → **2.39 秒** | |
| Qwen3.5-0.8B q6 | 256 (4 本) | 0.63 → 0.59 秒 | 3840 から 1.04 → 0.69 秒 | |
| Qwen3.5-2B q6 | 256 (4 本) | 0.81 → 0.75 秒 | 3840 から 1.22 → 0.85 秒 | |
| PLaMo 3 Translate 2B q4 GPTQ | 1024 | 2.56 → **0.96 秒** | 7168 から 8.39 (16.84) → **1.73 秒** | 位置 0 で 11.6 J |

  E2B の 3000 トークンのプロンプトは 5.1 → 1.75 秒。文脈とともに伸びる分 (16.84 の「全体を見る層」) の大半が 1 本のパイプラインだった。Qwen3.5 は全体を見る層が 4 層に 1 つで、残りは GDN の層なので効きが小さい。IRON の Llama 3.2 1B (1024 トークン 0.47 秒) との差は、E2B で 0.50 秒まで縮んだ
- 既にある書き出しは、書き出し直すと速くなる (重みも答えも同じ)。PLaMo 3 2B の 3000 トークンのプロンプトは 15.7 → 5.5 秒。手元の書き出しは 16.108 ですべて書き出し直した (Anima は 16.68 から 8 本)
- **PLaMo 3 2B の 4 ビットの速さを静かな機械で** (11.3 節、16.76): ほかの仕事のない機械で `ironlm -depth`、デコード 1 ステップは bf16 132〜138 ms (7.4 トークン/秒)、q4 (窓) 84〜90 ms、q4 GPTQ 81〜90 ms (11.5 トークン/秒)。16.76 の 7.3 → 11.6 トークン/秒と同じで、負荷の影響は出ていなかった

### 16.108 AMD: 新しいモデルの手順 (new-model-checklist.md) で各モデルを見直す — 2026-11-08

new-model-checklist.md の 4.1 節 (AMD) と 5〜10 節を、AMD の書き出しごとに照らし合わせた。

- **入っていたもの**: プロンプトの出力層は最後の行だけ (IRON の `CausalLM` が `x[call.rows - 1]` だけを出力層に通す。Apple の `LastRow` に当たる)、表はホスト (Gemma 4 の PLE と埋め込み、PLaMo 3、LFM3、Qwen3.5 の埋め込み)、窓の中だけの注意、塔は使うときだけ読む (画像と音声つきの E2B、LFM3-VL / Audio / Omni の書き出しはどれも `vision` / `audio` の版を持つ)、コアの丸め、状態のある層の塊の境目での写し
- **足りなかったもの**:
  - プロンプトの MHA のパイプライン (16.107): E4B、画像と音声つきの E2B、LFM3-VL / Audio / Omni は書き出し直していなかった
  - 塊の大きさ: 手順は「1024 (2048 より 20 % 速い)」だが、LFM3 系は 2048 のままだった。8 本のパイプラインの上で比べると、12288 トークン全体は 2048 で 11.25 秒、1024 で 11.53 秒と同じで、短いプロンプトのコマンドは 1024 が速い (30 トークン 536 → 409 ms、300 トークン 597 → 469 ms)。ロジットは 3 つの文で同じ。LFM3 系を 1024 にする
  - Qwen3.5 は 256 だった。0.8B q6 で比べた (コマンドの合計):

| 塊 | 30 トークン | 300 トークン | 長いプロンプト (tok/s) |
|---|---|---|---|
| 256 (4 本) | 321 ms | 911 ms | 317〜362 |
| **512** (8 本) | 354 ms | **664 ms** | 509〜565 |
| 1024 (8 本) | 416 ms | 750 ms | 575〜625 |

    GDN の演算子と射影の GEMM は塊の行数で決まるので、短いプロンプトは小さい塊が速く、長いプロンプトは大きい塊が速い。512 を既定にした
  - Qwen3.5-4B: 手元の書き出し (16.92) は文脈 1024、埋め込みがデバイスで固定 4613 MiB。4 ビットが小さいモデルのように崩れないかを HF の fp32 で確かめた: q4 0.023 / 0.042 / 0.046 (README / GPL / wiki)、q6 0.004 / 0.005 / 0.040。4B は q4 で崩れないので q4 のまま (固定 4149 MiB、ホストの埋め込みで −464 MiB)、q6 は 4653 MiB
  - **部品としての値** (10 節) を記録していなかった (2 回目の読み込み、一番長いコマンド、デコード): PLaMo 3 0.8B q4 GPTQ 0.5 秒、84 ms、40〜48 ms/トークン。Qwen3.5-0.8B q6 0.8 秒、85 ms、約 90 ms。Qwen3.5-2B q6 1.5 秒、109 ms、約 120 ms (ほかの書き出しが CPU を使う中)。どれも入口と出口は文 (PLaMo のトークナイザー / Qwen のトークナイザー)
  - **`CREATE_HWCTX` の再試行の長さ**: `ironlm` を続けて起こしたとき、前のプロセスの直後に `CREATE_HWCTX: invalid argument` で 1 度落ちた (1 秒の再試行を過ぎても)。前のプロセスはハードウェアコンテキストを 3 つ持つ (デコード、塊の前の空のコマンド、プロンプト)。再試行を 5 秒にした (`internal/xdna`)。カーネルのログの `aie2_cmd_submit: Grab job sem failed, ret -4` は EINTR で、ioctl は再試行している (害はない)
- **書き出し直し** (8 本のパイプライン、LFM3 系は塊 1024、Qwen3.5 は塊 512): どれも README の 400 行のロジットが前の書き出しとビット単位で同じ。画像と音声を受ける 4 つ (E2B の画像と音声つき、LFM3-VL / Audio / Omni) は、`hwnpu-chat` で文 4 問、画像 1 枚 (魔女の絵の説明)、音声 1 本 (`Front_Left.wav`) を新旧で比べ、答えはすべて同じ (VL は画像まで、Audio は文と音声)。塊の時間:

| | 前 | 後 |
|---|---|---|
| Gemma 4 E4B q4 GPTQ (1024) | 位置 0 2.07 秒 (22.1 J)、3072 から 3.60 秒 | 1.04 秒 (11.8 J)、1.24 秒 |
| E2B の画像と音声つき (1024) | 1.14 秒 (11.8 J)、3072 から 2.24 秒 | 0.49 秒 (4.9 J)、0.63 秒 |
| LFM3-VL / Omni (2048 → 1024) | 2048 の塊 2.57 秒 (29 J)、10240 から 11.3 秒 | 1024 の塊 0.75 秒 (7.9 J)、11264 から 1.35 秒 (トークンあたり 1.7〜4.2 倍) |
| Qwen3.5-0.8B q6 (256 → 512) | 423 → 362 トークン/秒 | 534 → 494 トークン/秒 |
| Qwen3.5-2B q6 (256 → 512) | 332 → 292 トークン/秒 | 444 → 409 トークン/秒、固定 2014 → 1862 MiB (GDN の状態の控えは塊ごとなので半分) |

- **Qwen3.5-4B** を `--q4 --prefill-chunk 512` で書き出した (`~/models/Qwen3.5-4B.amd-npu2-q4`、文脈 4096): 固定 3727 MiB (16.92 の書き出しは 4613 MiB、文脈 1024)、デコード 157〜162 ms (6.2 トークン/秒)、プロンプト 193〜214 トークン/秒。ロジットは塊 256 の q4 と同じ
- 書き出しを続けて走らせた間に、`ironlm` が 2 回、前のプロセスの直後に失敗した (古いバイナリ、再試行 1 秒)。作り直した 5 秒の再試行では起きていない

### 16.109 Qwen3.5 の複数トークン予測 (MTP) の層で投機的デコード (`qwen35-mtp`、`-mtp`)、GDN の状態をホストで戻す (`state-rows`) — 2026-11-08

Qwen3.8-Flash-Next (Qwen4 の構造の先行版、MTP 4B を持つ) の発表を見て、今の Qwen3.5 系がすでに持っていて捨てていた MTP の層 (`mtp.`、1 層) を下書きに使った (model-fit.md T9 の「下書きモデル」を、別のモデルでなくチェックポイントの層で)。

- **MTP の層** (`internal/llm/qwen35/mtp.go`、`-arch qwen35-mtp`): transformers には実装がないので vLLM の `qwen3_5_mtp.py` (`Qwen3_5MultiTokenPredictor`) から移した。位置 t で、トークン t+1 の埋め込みと本体の最終の隠れ状態 h_t (最後の norm の後。vLLM、SGLang とも) をそれぞれ zero centered の norm (`pre_fc_norm_embedding`、`pre_fc_norm_hidden`) に通し、埋め込みを先に連結して `fc` (2d → d)、全注意の層 `mtp.layers.0` (本体の全注意と同じ形)、`mtp.norm`、本体の出力層 (共有) で、トークン t+2 のロジット
  - llm.Model としては 2 層: 層 0 が前段 (norm、連結、fc)、層 1 が復号の層。層 0 の注意はホストで回せない (`PlanPaged`) ので、注意で分ける形 (`-paged`) のために前段を層にした。入力は `x` (埋め込み) と `hid` (本体の h、`llm.TokenInputs` の表のない入力)。キャッシュつきでは自分の最終の隠れ状態 `h` も出し、2 つ目からの下書きは本体の h の代わりにそれを読む (vLLM と同じ)
  - 本体の `hidden-out` (Arch の param): 全部の行の最終の h を出力 `h` に (LastRow のプログラムでも norm を全部の行に掛けてから最後の行を選ぶ)。`decode.Session.HiddenRows` が残したトークンの分を返す
  - 照合: fp64 の正解 (`mtpReference`、vLLM の forward をそのまま) と 1.3e-7 程度 (`TestMTPAgainstFP64`)、キャッシュつき (Q=1、4) と `Truncate`、注意で分けた形 (`TestMTPCached`、`TestMTPPaged`)。連結の順を入れ替えると 8 件落ちる
- **下書き** (`internal/chat/mtp.go`、`mtpDrafter`): 本体の各呼び出しの後に、残したトークンの h を受け取る (`Engine.recordHidden`)。下書きの前に、次のトークンが分かった位置 (前回の残りと、今選んだトークン) を MTP の層に読ませ (Q=8 で `LastRow`)、最後の行のロジットの argmax が 1 つ目、2 つ目からは 1 トークンずつ。終わったら本物の位置まで `decode.Session.Truncate` で戻す (状態は全部が位置ごとの KV なので印が要らない)
  - 位置は `base` から: RoPE の注意は相対位置しか見ないので、h が分かっているところから始められる。プロンプトの戻り (前の要求の印) や会話の入れ替えは、持っているトークン列と比べて共通の先頭まで残す (`sync`)。512 位置より遅れたら (長いプロンプト) そこからやり直す (`mtpStart`。1 回 8 位置なので 2000 トークンの読み込みに 250 回は払わない)
  - プロンプトからの推測 (`LookupDraft`) が MTP の 1 つ目と合っていて長ければそちら
  - 照合 (`TestMTPDrafts`、CPU): 下書きありの生成が 1 トークンずつと同じ。1 つ目の下書き 45 回がすべて、列の始めから読み直した MTP の層と同じ (戻り、次の会話、印に戻って別の続き)。`sync` で entries を切らないようにすると 14 件落ちる (別の続きの場合だけ。同じ続きでは古い entries が正しいままなので見逃す)
- **GDN の状態をホストで戻す** (`state-rows`、`llm.StateReplayer`、`qwen35.Config.ReplayState`): Gated DeltaNet の状態はステップで丸ごと置き換わるので、下書きの途中までしか残さないとき `Speculate` は残す分をもう一度本体で回していた (呼び出しが 2 倍)。下書きを確かめるプログラムに各行の k (L2 正規化済み)、v、beta、decay を `st_rec_N_rows` (1, 値ヘッド, 行, dk+dv+2) として出させ、ホストがステップ前の状態から残した行だけ delta rule を進める (fp32、ヘッドごとに並列)。0.8B で 1 層 66 KB の出力。プロンプトを読むプログラムは確かめないので付けない (名前 `-sr` は下書きのプログラムだけ)
  - `TestShareRewindSpeculate` の `state-rows` 版: 呼び出し 1 回で、誤差は再実行と同じ 4e-4。decay を 0.9 倍にすると落ちる
  - LFM3 の短い畳み込みの窓 (`win_conv_N`) は、もとから要らない: プログラムが前の窓とステップの行を連結した全体を `win_conv_N_new` に出し、`decode.Session.keep` が残した行の後ろから窓を切り出す。`lfm3.TestCachedShareAndRewind` に、4 トークンのうち 2 つを採る `Speculate` が 1 回の呼び出しで全体のプログラムと合うことを足した
- 変換 (`chat.Prepare`): `-draft` が 2 以上で MTP の層があると (`-mtp auto`)、本体は `hidden-out` (名前に `-h`)、下書きのプログラムはさらに `state-rows` (`-sr`)、MTP のプログラム `qwen35-mtp-lut8-p8192-c262144-q8-last` (0.8B で 2 部分、0.01 + 0.26 GiB。出力層が大半) を足す
- **M1 (16 GB) の Neural Engine、Qwen3.5-0.8B lut8、`-prefill off`、貪欲 (temp 0)、2 問 (英語でコード、日本語の説明)、生成 1 回の tokens/s:**

| 下書き | コード | 日本語 |
|---|---|---|
| プロンプトだけ、再実行 (前) | 26.1 (79/696 採用) | — |
| プロンプトだけ + `state-rows` | 32.1 (79/696) | 25.8 (11/323) |
| MTP 1 つ + `state-rows` | 40.9 (146/325) | 27.4 (49/167) |
| **MTP 2 つ + `state-rows` (既定)** | **43.8 (169/308)** | **30.3 (90/263)** |
| MTP 3 つ + `state-rows` | 40.1 (187/373) | 23.7 (82/323) |
| MTP 2 つ、再実行 | 38.1 (169/308) | — |

  - 既定は 2 つ (`mtpDrafts`、`HWNPU_MTP_DRAFTS`)。3 つ目は当たりが減るうえ、1 つごとに MTP の層の 1 回 (出力層 248320 × 1024 を含む) を払う
  - 文は貪欲でも途中から変わる (日本語は 30 トークンほどで「芽生え」→「芽生え始め」)。ANE の fp16 は一緒に回す行で丸めが変わり、ホストの状態の戻しは fp32 なので、ビット単位では同じにならない (分布は変わらない。CPU の照合では一致)。プロンプトだけの下書きでも `-sr` の有無で長さが 266 と 268 で違う
- **2B と 4B を M1 で** (16 GB、同じ 2 問、`-think=false`、`-prefill off`、temp 0。プロンプトだけの下書きも `state-rows` つき。2B は 16.115 で lut6 にする前の lut8。2B の lut6 と M3 は 16.115):

| 下書き | 2B lut8 コード | 2B lut8 日本語 | 4B コード | 4B 日本語 |
|---|---|---|---|---|
| プロンプトだけ (`-mtp off`) | 23.5 (60/639) | 19.3 (19/605) | 1.0 (70/872) | 0.8 (10/410) |
| MTP 1 つ | 29.6 (124/275) | 21.6 (96/293) | 1.3 (161/352) | 1.0 (72/173) |
| **MTP 2 つ (既定)** | **32.4 (152/277)** | **20.1 (106/358)** | **1.6 (198/336)** | **1.1 (106/275)** |
| MTP 3 つ | 33.9 (169/265) | 17.5 (118/569) | 2.0 (220/354) | 1.1 (87/343) |

  - 2B はコードで 1.4 倍、日本語はほぼ同じ (0.8B ほど当たらない)。3 つは日本語で遅くなるので既定は 2 のまま
  - 4B は M1 16 GB ではもとが 1 tokens/s (M3 では 1 本で約 8、16.62)。MTP で 1.6〜2 倍でも実用にならない。M1 では 2B を勧める
  - **4B が遅い理由は M1 の Neural Engine に常駐する重みの量** (`HWNPU_PROFILE`、`-mtp off`、Q=8 の 1 ステップ、同じ質問で 40 トークン)。スワップは前後で増えず (1179 MB のまま)、ホストの注意は部分あたり 7〜10 ms で、時間は ANE の部分そのもの。重みの合計が境目を超えると、同じ大きさの部分が 1 回ごとにばらついて 10 倍以上遅くなる (載せ直し。16.112 の Gemma 4 E2B の 2 本目と同じ見立て)。境目は `npuprobe split` で測った (PROBE.md の M1): 乱数の重みの lut4 を 8 部分で順に回すと、Neural Engine が持つ量 2.86 GiB までは単独の合計どおり、2.97 GiB から 12 倍。部分の大きさではなく合計で決まる (2.95 GiB を 4 部分でも同じ)。M4 は 3.62 GiB でも全速:

| 形式 | 重みの合計 (ANE が持つ量) | 部分 1 つ (ms) | 1 ステップ (ms) | tokens/s |
|---|---|---|---|---|
| 2B lut8 (比べ) | 1.81 GiB (1.79) | 5.7〜10.2 | 61 | 20.1 |
| 4B lut4 | 1.96 GiB (1.99) | 8.1〜9.3 | 109 | 10.9 |
| 4B lut6 (2 回目) | 2.95 GiB (2.97) | 75〜131 | 985 | 1.1 |
| 4B lut8 | 3.96 GiB (3.99) | 65〜196 | 1251 | 0.9 |

  - 4B lut4 は 2B lut8 の約半分の速さで、この短い答えは崩れていないが、lut4 は 0.8B と 2B で同じ句を繰り返す (16.115)。M1 で 4B を lut4 にするなら、16.88 の照合 (KL、合言葉) が要る。4B の lut6 (2.97 GiB) は境目をわずかに超えるだけなので、0.1〜0.2 GiB 削れば (出力層だけ lut4 など) 載る見込み
  - `npuprobe limit` の 1 本の上限は M2 と M3 と同じ 0.99 GiB (lut4 35 層は載り 36 層は CPU、int8 17 層と 18 層)。`llm.Targets` の m1 を測定済みにした (変換の「m1 の上限は推定」の注意が消える。m2 はまだ推定)。limit の複数のモデル (0.99 GiB × 3 = 2.97 GiB) は 1 つずつ続けて呼ぶので、合計の境目は出ない
- 確かめていないこと: M3 での 4B、長いプロンプトの後の最初の下書きまでの時間 (最大 64 回)、ロジットを fp16 のまま argmax して 248320 個の変換を省くこと、下書きの語彙を絞る (FR-Spec のように頻度の高いトークンだけで出力層を回す)、AMD (IRON) と Intel

### 16.110 AMD: Qwen3.6 のプロンプトの MoE を何回かに分けたとき、共有の専門家が壊れていた件と、専門家ごとの行の上限 (`MOE_CAP`) 64 → 128 — 2026-11-08

- **不具合**: プロンプトの塊で、ある専門家に `MOE_CAP` 行を超えるトークンが来ると、ホストが塊を何回かに分けて専門家のコマンド (B) を回す (16.93)。B は共有の専門家も含み、その入力の hn (A の norm の出力) を毎回読み直す。ところが IRON の領域の割り当ては、`MoEGroupDown` の出力 y (8 MiB) を hn (1 MiB) の上に置いていた (B の中では hn は共有の専門家のあとに使われないため)。2 回目からの共有の専門家は、前の回の y を hn として読んでいた
  - 見つけ方: 256 トークンの塊の HF との照合 (`iron_qwen35_check.py ref --stream --tokens 256`、README の先頭) が**コサイン 0.578** (最上位も違う)。32 トークンの check (`ironlm` の既定) は 1 回で済むので見えなかった。16.93 以来、256 トークンの塊のほとんどの層 (2〜4 回) が壊れていた
  - 直し方 (`internal/llm/iron/moe.go`): ホストは経路の選択のときに hn を読んでいるので、それを持っておき、2 回目からの B の前に書き戻す (`moeRoute.hn`)。書き出し直しは要らない
  - 直したあと (HF の fp32 とのコサイン、q4 と GPTQ の書き出し): 64 トークン 0.9958、128 トークン 0.9723、256 トークン **0.578 → 0.9892** (その後のデコード 0.926 → 0.978、0.973 → 0.993)。256 トークンの最上位は HF の 1 位と 2 位が 15.41 と 15.19 の僅差で、NPU は 2 位を選ぶ
- **行の上限を 128 に** (`iron_moep.CAP`、`kernels/moe_rows.cc` の `MOE_CAP`): コアは専門家の行の gate と up の累算を持つ (fp32、1 行 32 出力 × 2)。64 行で 16 KB、128 行で 32 KB (256 行は 64 KB でコアのメモリに入らない)。256 トークンの塊は 1 層 3〜4 回 (40 層で 131 回) → 1〜2 回 (76 回)
  - 単体: 8 専門家に 125 行ずつ (128 トークン) でコサイン 0.999818、256 専門家 256 トークンは 1 回 44.7 ms で 64 行のときと同じ
  - 40 層: ロジットは 64 行の書き出しとビット単位で同じ。NPU に他のものが動いていないとき、README の先頭 256 トークンの塊 (`ironlm -forced -split 256`、ロジット 1 行込み) で 64 行 5.3 s → 128 行 **4.5 s** (−15 %)、専門家のコマンド (B) の合計 3.33 s → 2.57 s、NPU のコマンドは 170 → 116 回。ホストの経路の選択と合わせは 0.45 → 0.43 s で変わらない。書き出し直しで効く (Go は manifest の `cap` を読むので、64 行の古い書き出しもそのまま動く)
- 長い文脈 (`--max-seq-len 4096` の書き出しで、既定は 1024 のまま): 塊 (256 トークン) は 5.65 s (512 から) → 6.80 s (3840 から)、デコードは 173 → 176.5 ms。2,834 トークンと 3,727 トークンの文の中ほどと先頭に置いた 5 桁の番号を答えた (上の不具合のある状態で)。GDN の状態の枠が塊の境目ごとに 67 MB (30 層) で、4096 では固定のメモリが 0.33 → 1.1 GB に増える

### 16.111 AMD: プロンプトの塊の GDN の SiLU を演算子の前で 1 回だけ (`GdnSiLU`) — 2026-11-08

- 内訳 (Qwen3.6 の 2 層、256 トークンの塊、手順ごとのコマンド): GDN の層の手順 47 ms のうち `GatedDeltaRule` が 31.2 ms。射影の GEMM と重みの展開は合わせて数 ms
- 演算子の中の内訳 (単体、256 行): 30.4 ms のうち、行を写すときの q、k、v の SiLU (`gdn_gather` の `silu_row`、float32 の `exp_f32` と逆数で、丸めは 1 回) が約 23 ms、再帰そのものが約 7 ms (SiLU を外すと 7.5 ms、写すのも外すと 7.8 ms)。コアは状態を値の列 16 本の組 (slice) ずつしか持てないので、各行を slice ごとに読み直す。Qwen3.6 は 1 コア 2 頭 × 8 slice で 16 回、そのたびに同じ q と k (2 頭は同じ鍵の頭) に SiLU をかけ直していた
- 直し方 (`iron_gdn.silu_operator`、`kernels/gdn.cc` の `gdn_silu`): 演算子の前で、各列の q、k、v に 1 回だけ SiLU をかける (a と b と列の埋めはそのまま、`silu_raw`)。計算は gather のものと同じ (float32、丸め 1 回)。列の 4 コアが行を等分して持つ (`silu_parts`)。`GatedDeltaRule` は `presilu` で行の部分を写すだけ (`GDR_PRESILU`)。プロンプトの塊だけ (デコードの 1 行は前のまま)。書き出し直しで効く (Go は変えていない)
- 単体 (256 行): 30.4 → 9.8 ms (SiLU 約 2 ms + 再帰 7.7 ms)、誤差は前と同じ値 (37 行だけ有効でも同じ)。H = KH = 16 (Qwen3.5-0.8B の形、列の埋めあり) も通る
- Qwen3.6 の 40 層 (q4 と GPTQ、行の上限 128): ロジットは前の書き出しとビット単位で同じ。README の先頭 256 トークンの塊 4.5 → **3.9 s**、GDN の層の A 43 → 21.7 ms。32 トークンのプロンプト 1.49 → 1.42 s。デコードは変わらない (172 ms)
- Qwen3.5-0.8B (q6、塊 512): ロジット (1024 行) は前の書き出しとビット単位で同じ。512 トークンの塊 0.9 → **0.5 s**
- 残り: Qwen3.6 の塊 3.9 s のうち専門家のコマンド (B) が 2.56 s (重みを読む速さが下限)、A が 0.7 s。`GatedDeltaRule` の 7.7 ms は、各コアが全部の行を 16 回 DMA で受けること (1 行ずつの object) と、状態の 3 つの bf16 の limb の計算

### 16.112 Gemma 4: プロンプトを読むプログラムをキャッシュを書く層だけに (`-kv-only`)、ステップの内訳 (`HWNPU_PROFILE`) — 2026-11-08

- 動機: M1 (MacBook Pro 13"、16 GB、macOS 27.2) で Gemma 4 E2B の生成の 1 ステップ (Q=8) の内訳を測ると、部分 7 つは 3.5〜5.3 ms なのに、層 29〜33 の部分だけが 7.9 ms だった。`-prefill off` (生成用の 1 本だけ) にすると 4.7 ms に戻り、生成は 17.5 → 20.0 tokens/s。2 本 (q8 と q128-last、重み 3.20 GiB) を読み込むと Neural Engine が抱えるメモリ (`mem.NPUWired`) は 1.96 GiB で、1 本 (1.63 GiB) のときより 0.33 GiB 多いだけ。2 本目の重みはほとんど載らず、生成の部分の 1 つが呼び出しのたびに載せ直されている (推定。中は見えない)
- 見方: Gemma 4 の層 15〜34 (E2B。E4B は 24〜41) は前の層の鍵と値を使い回し、キャッシュに書かない。プロンプトを読むプログラムに要るのは、キャッシュを書く層 0〜14 だけ。今の q128-last は層 15〜34 を最後の行だけで計算していた (16.83) が、その重みと出力層 (計 1.2 GiB) を持っていた。プロンプトの最後のトークンのロジットは、生成用のプログラムで読めば出る
- 変更:
  - `llm.Options.KVOnly` (`hwnpu-convert -kv-only`、`-paged` と組で): 注意で分けた計画 (`PlanPaged`) から、キャッシュを書く層 (`llm.CacheWriters.CacheLayers`、Gemma 4 は共有の層の前まで) の部分だけを残す (`kvOnlyParts`)。最後の部分は層 14 の注意の手前で終わり、その鍵と値 (`kv_k_14_new`) を出す。出力層はない。マニフェストに `kv_only`
  - `chain` と CPU の `runner`: 最後の部分が `attn_q_N` を出して終わる形を受ける (その層には注意しない)。`decode.Program.NoLogits`
  - `decode`: `NoLogits` のプログラムは、最後の部分の後で、注意しなかった層のキャッシュの行を書き (`writeUnattended`)、ロジットを返さない。最初のプログラム (生成用) には使えない
  - `chat`: プロンプトのうち最後の「生成用の 1 回の分」(Q=8 なら 8 トークン) を除いた部分を KV だけのプログラムで読み、最後は生成用で読む (`Engine.logitsQ`)。区切り (mark) とストアの区切りはロジットなしになりうる。`reach` はもともとロジットのない区切りをプロンプトの終わりには使わない。ストアから戻すときも、プロンプトの終わりにあるロジットなしの区切りは使わない (`errNoLogits`)
  - `hwnpu-chat`: 注意で分けたモデルで、キャッシュを書かない層を持つもの (Gemma 4 E2B と E4B) は、プロンプト用を `-q128-kv` で変換する (前は `-q128-last`)。`-prefill auto` の見積もり (`onePrograms`) も、プロンプト用の重みを層 0〜14 の分で数える
- テスト: `TestPagedKVOnly` (CPU のインタプリタ、小さな Gemma 4、共有の層 3〜5): KV だけのプログラム (部分 1 つ、層 0〜1 と層 2 の鍵と値) で 7 トークンを読み、残りの 4 トークンを 1 トークン/回のプログラムで読むと、キャッシュなしのプログラムのロジットと相対誤差 0.0038 (`TestPagedLastRow` は 0.0036)。注意しなかった層の行を書かない改変は落ちる
- M1 での結果 (E2B、lut6、温度 0。同じ機械の別の作業でロードアベレージ 4〜5 があり、値は ±5% 揺れる):

| | 前 (q128-last) | 後 (q128-kv) |
|---|---|---|
| プロンプト用の部分と重み | 8 部分 1.6 GiB、出力の `reshape` 1 つが CPU | 3 部分 0.38 GiB、全部 Neural Engine |
| 読み込む重みの合計 / Neural Engine が抱えるメモリ | 3.20 / 1.96 GiB | 1.98 / 1.73 GiB |
| 2 本目の読み込み (2 回目以降) | 2.6〜3.5 s | 0.4〜0.5 s |
| 4249 トークンのプロンプト | 4.2〜4.6 s | 2.4〜2.7 s |
| 生成 (その後の 150 トークン) | 18.5〜21.5 tokens/s | 21.6〜24.4 tokens/s |
| 生成のステップの層 29〜33 | 7.9 ms | 4.9〜5.7 ms (1 本だけのときと同じ) |

  - 答えは 150 トークンとも前と一字一句同じ。短い質問 (21 トークン) の生成は 1 本だけ (`-prefill off`) のときと同じ 58〜60 ms/ステップになった (干渉が消えた)
  - E4B は 2 本の重みが 6.51 → 4.88 GiB になるが、生成用だけで 3.26 GiB あり、コンパイルのキャッシュの目安 (3.5 GiB) を超えるので `-prefill auto` は今までどおり 1 本。`-prefill on` の起動と速さは測っていない
- ステップの内訳 (`HWNPU_PROFILE=1`、`decode.WriteProfile`): 注意で分けたプログラムの 1 ステップを、部分ごと、その後のホストの注意ごと、ロジットの行の読み出しに分けて、1 ステップあたりの ms を出す。hwnpu-chat は答えのたびに出力し、Neural Engine が抱えるメモリも添える。M1 の E2B (21 トークンの質問、1 本): 部分 3.5〜4.7 ms × 7 と出力層の部分 8.2 ms (計 38 ms、lut6 で約 46 GB/s)、全注意 7 層のホストの注意が 1 層 2.0 ms (計 14 ms)、行の読み出し 0.6 ms、計 53 ms
- 試して採らなかったもの (M1、E2B):
  - int8: 部分が 38.1 → 48.8 ms。M1 は npuprobe で int8 を 62 GB/s で読む (lut6 より速い見込みだった) が、実物では約 48 GB/s で、バイト数が 4/3 倍のぶん遅い。lut6 のまま
  - 注意の塊 4096 (`-paged 4096`): ホストの注意が 1 層 2.05 → 1.36 ms、ステップ 53.1 → 51.6 ms。1 回の呼び出しに約 0.67 ms の固定費があり、短い会話でも 7 層で約 4.7 ms は残る。長い文脈では 1K あたりが 1 割ほど悪くなる (16.58 の M3 で 1.09 → 1.26 ms) ので、既定は 8192 のまま。CPU で注意する案は、電力を優先して採らない (Neural Engine に載せる方針)
  - ソフトキャップをホストで (`-logits raw`、出力層の後の mul、tanh、mul と CPU の `reshape` を外す): 測る前に、macOS の `mediaanalysisd` (CPU 122%) と Apple Intelligence の推論サービスが Neural Engine を使い始め、どの形も 0.7 tokens/s に落ちて測れなかった。ホストの側 (サンプリングでのソフトキャップ) は書いていないので戻した
- 残り: E4B の `-prefill on` (2 本 4.88 GiB) で起動と生成の干渉を測る。M3 での確認。消費電力 (`powerbench` は sudo が要る)。LFM3 など全部の層がキャッシュを書くモデルでも、KV だけのプログラムは出力層 (LFM3 で約 0.19 GiB) を外せる (`CacheLayers` を全層にした形。今は CacheWriters を持つモデルだけ)

### 16.113 Gemma 4: ロジットのソフトキャップをホストで (`-logits host`、既定、gemma4 の版 1) — 2026-11-08

- 動機: new-model-checklist.md の 3.2 節で Mac に残っていた「Gemma 4 の最後の部分の `reshape`」。生成用の最後の部分 (出力層) は、ソフトキャップ (mul、tanh、mul) を (16, 16384) に畳んだロジットに掛け、(1, seq, 262144) に戻す `reshape` 1 つが CPU に置かれていた (16.43)。毎ステップ、Neural Engine と CPU が出力層で交代する。消費電力を優先する方針 (new-model-checklist.md の冒頭) では、CPU に落ちる演算をなくしたい
- 測ったこと (M1、E2B、1 本、Q=8、21 トークンの質問の後の 176 トークン、3 回ずつ交互): 出力層の部分 (part7) とすぐ前の部分 (part6、重みがほぼ同じ) の差が、プログラムの中でキャップを掛ける形で 3.9〜4.5 ms、キャップなしで 1.5〜1.9 ms。1 ステップ約 2.3 ms (4%) 短い。キャップなしは全部 Neural Engine
- 変更:
  - gemma4 の `logits` の既定を `host` に (`Options.HostSoftcap`)。出力層は語彙 262144 の linear だけ。`flat` (今までの形) と `folded` も残す。名前が変わるよう gemma4 の `Revision` を 1 にした (`gemma4-r1-lut6-...`。前の変換は使わず変換し直す)
  - `llm.LogitSoftcap` (モデルが、その設定でホストに残すキャップを返す)。変換はその値をパーツのメタデータ `com.github.shibukawa.hwnpu.softcap` に書く (`WritePart` が必ず書く。メタデータを書かない呼び出しでも印は付く)。ホストは `llm.HostSoftcap(meta)` で読み、`llm.Softcap` で c·tanh(y/c) を掛ける
  - `decode`: `Program.Meta` (chain と CPU の runner が `llm.ReadMeta` で入れる) からキャップを知り、返すロジットの行に掛ける (ツールとテストはそのまま正しい値を得る)。`SetRawLogits(true)` で掛ける前の値を返す。キャップの違う 2 本は `Add` が断る
  - `chat`: セッションをキャップの前の値で使い、サンプリング (`Sampling.softcap`) が使う値だけに掛ける。キャップは単調なので、温度 0 と上位 k の選び方は変わらない。確率は上位 k 個のキャップ後の値で計算する。繰り返しの罰はキャップ後の値に掛けてから、キャップ前に戻して書く (`uncapped`、atanh)。CPU の手間は 1 ステップに数十の tanh
  - `hwnpu-check` (組み立てと `-package`) と `hwnpu-generate` のキャッシュなしの経路も、メタデータのキャップを掛けてから比べる、選ぶ
- テスト: `TestPickSoftcap` (キャップ前の値をキャップつきで選ぶと、キャップ後の値を選ぶのと温度 0 で同じ、温度・top-k・top-p で 500 種のうち 99% 以上同じ。罰で順位が入れ替わる例 (100 と 80 に 1.05) で、キャップ後に掛けたときの答え 1 を選ぶ)。罰をキャップ前の値に掛ける改変は落ちる (43/500、0/500)。gemma4 の部分と全体、キャッシュ、注意で分けた形のテストは、メタデータのキャップを掛けてから比べる
- M1 での結果 (E2B、lut6、温度 0。16.112 の KV だけのプロンプト用と組み合わせて):
  - 配置: 2 本とも `all on the NPU` (前は生成用が `NOT all on the NPU`)
  - 21 トークンの質問の後の生成: 1 ステップ 53.3 ms、出力層の部分 6.4 ms (すぐ前の部分との差 1.1 ms)、20.1 tokens/s
  - 4249 トークンのプロンプト: 2.5 s、その後の生成 25.0 tokens/s
  - 答えは、短い質問の 176 トークン、要約の 150 トークンとも、プログラムの中でキャップを掛ける形と一字一句同じ
- 確かめていないこと: M3 と M2、Intel (OpenVINO の IR も同じ `host` の形になる。CPU での照合だけ)、温度ありの答えの分布を実物で比べること (単体のテストだけ)、消費電力 (`powerbench` に sudo が要る)

### 16.114 Gemma 4 E2B の lut6 を GPTQ で (M1 の 16 GB で校正)、hwnpu-chat の `-codes` と `-format-for`、Neural Engine の上の KL (`hwnpu-generate -score`) — 2026-11-08

- 動機: new-model-checklist.md 5.3 節の Mac に残っていたもの (hwnpu-chat が GPTQ の符号を取らない)。16.95 は LFM3 を M3 で確かめただけで、チャットの既定の Gemma 4 E2B は Apple の lut6 で校正していなかった
- **16 GB の Mac で校正する** (`q4_gptq.py` の `load_gemma4_lean`、`MODELS["gemma4"]` の `lean`): E2B の文の部分は fp32 で 4.6B (18.5 GB) に出力層の写し 1.6 GB で、16 GB に入らない。2 つの埋め込みの表 (PLE 2.35B、語彙 0.40B) は引くだけなので、チェックポイントを memmap して使う行だけ読む (`TableRows`)。モデルは語彙 1 で組み立ててから表を差し替え、残りの重みと各層の `layer_scalar` (バッファ。最初は読み漏らして答えが崩れた) をファイルから fp32 で入れる。約 9 GB。transformers の bf16 の `from_pretrained` とのロジットの KL は 0.003 以下で、最上位はすべて一致 (`tools/ref/ref_gemma4.py` と同じ考え)
  - 1 回分の校正は、その回で量子化する最後の層の後で打ち切る (`_Stop`)。後ろの層と出力層 (語彙 262144) を計算しない
  - 参照のロジットは採点する位置だけ持つ (1 文 0.7 GB → 0.4 GB)
  - 校正の文は `tools/amd/calib_data.py` で Hugging Face の parquet から作る (`~/models/calib`)。GPL-3 は Mac にないので評価から抜ける
  - PyTorch は Homebrew の `pytorch` (2.15、venv は `/opt/homebrew/opt/pytorch/libexec`)。ref-venv はその venv の Python で作り、site-packages を `.pth` で足し、transformers などを入れた。coremltools は Python 3.15 の native がなく、`eval_logits.py` は使えない
- 校正: `q4_gptq.py gemma4 CKPT OUT --fmt-q4 lut6 --fmt-q8 lut6 --head lut6 --lut-fit wkmeans --q8 gptq` (16.95 の LFM3 と同じ)。M1 の CPU 4 スレッドで **87 分**。後半はスワップ (最大 11.6 GB) で 1 回 7〜15 分に延びた。206 個の重み (MLP、注意、出力層。共有の層の k と v は持たない)、符号 2.1 GB。CPU の見積もり (bf16 の重みとの KL、後ろ 400 位置): README 0.088、wiki40b-ja 0.084、wikitext-2 0.080、Go 0.034
- **Neural Engine の上の KL** (`hwnpu-generate -score REF,...`、足した): キャッシュなしのプログラム (`hwnpu-convert -seq 128`) を参照のトークンで 1 回流し、全位置で PyTorch の fp32 (`ref_gemma4.py`) と比べる。文は校正にない wikitext-2 の試験分と wiki40b-ja の評価分の先頭 128 トークン (BOS あり)。どれも 2〜3 部分で全部 Neural Engine

| E2B (Neural Engine、128 トークン) | 英 KL 平均 / 中央値、最上位 | 日 KL 平均 / 中央値、最上位 | 重み |
|---|---|---|---|
| lut6 の丸め (チャットの `kmeans-minmax`) | 1.02 / 0.25、88/128 | 1.16 / 0.47、82/128 | 1.60 GiB |
| 同、`per_layer` を lut8 | 0.94 / 0.25、92 | 1.03 / 0.43、81 | 1.61 GiB |
| lut6 の GPTQ (`-codes`、206 個が当たる) | 0.74 / 0.13、94 | 1.02 / 0.34、90 | 1.60 GiB |
| **GPTQ、`per_layer` を lut8** | **0.56 / 0.08、97** | **0.94 / 0.20、92** | 1.61 GiB |
| lut8 の丸め (目安) | 0.10 / 0.02、118 | 0.10 / 0.02、113 | 2.16 GiB |

  - 文の先頭の数トークンは分布が平らで KL が大きく (最大 8〜13)、平均を押し上げる。16.95 の 0.24 / 0.42 (別の文) とは比べられない。CPU の見積もり (後ろ 400 位置) より大きいのも同じ理由と、fp16 の計算 (lut8 でも 0.10)
  - **`per_layer` の射影** (層ごとの `per_layer_input_gate` と `per_layer_projection`、`per_layer_model_projection`、合わせて 41M): 校正の対象 (MODELS の q4 / q8) に入っておらず、変換器が lut6 に丸めていた。lut8 にすると +0.01 GiB で、GPTQ と合わせたときに中央値が 4 割下がる (丸めだけでは小さい)。fp16 の 5 部分のプログラムは Core ML が `Unable to compute the prediction` で動かなかった
- **hwnpu-chat の `-codes DIR` と `-format-for SUBSTRING=FORMAT`** (hwnpu-agent も、`chat.Setup.Codes`、`FormatFor`): 変換に `llm.Options.Codes`、`FormatFor` を渡す。名前に符号の index.json のハッシュと規則を入れ (`gemma4-r1-lut6-gptq-27b3b448-per_layer.lut8-p8192-...`)、符号なしのものと混ざらない。lutN でない形式に `-codes` は誤り。変換ごとに当たった数を出す (生成用 206、KV だけのプロンプト用 101)。`TestPrepareCodes`
- チャットの形 (M1、注意で分けた 8 部分と KV だけの 3 部分、温度 0、考えない) で、16.95 の 12 問を 1 問ずつ:
  - 丸め 10/12 (『吾輩は猫である』の作者を「野獣橋」、一番大きい湖を洞爺湖)。GPTQ と GPTQ + `per_layer` lut8 は 11/12 (夏目漱石。湖は洞爺湖のまま)。残りの答えは同じか言い回しだけ
  - 速さは同じ: 長い答え (171〜199 トークン、プロンプト込み) 20.2〜21.9 → 20.5〜21.9 tokens/s。Neural Engine の重み 1.98 → 2.00 GiB (`per_layer` lut8)。全部 Neural Engine
- 判断: 電力の方針 (読むバイトは同じかほぼ同じ) のまま精度が上がるので、`-codes` で使えるようにした。既定にはしない: 符号 2.1 GB の配り方が決まっていない (16.95 と同じ)、12 問では 1 問の差
- 確かめていないこと: E4B (fp32 で 30 GB を超え、lean でも層だけで 15 GB ほどの見込みで、16 GB の Mac では難しい)、問いを増やした比較と hwnpu-agent の課題、M3 と M2、`per_layer` を GPTQ の対象に入れること、lut4 + GPTQ のチャット

### 16.115 Neural Engine の数値の確かめ方 (Whisper のエンコーダ)、`qchunk`、書き起こしと Qwen3.5 の形式 — 2026-11-08

- **Neural Engine と Core ML の CPU** (`internal/transcribe/device_test.go` の `TestEncoderOnDevice`。`HWNPU_WHISPER_REF` に `tools/ref/ref_whisper.py` の参照、`HWNPU_WHISPER_ENCODERS` にパッケージ、`HWNPU_WHISPER_LAYER` に層の数 (`hwnpu-convert -layers N` の途中までのエンコーダ))。英語 5.2 秒の音声で HF fp32 の出力とのコサイン:

  | パッケージ | Neural Engine | Core ML の CPU | 2 つの間 |
  |---|---|---|---|
  | fp16 | 0.958 | 0.971 | 0.971 |
  | lut8 | 0.954 | 0.964 | 0.971 |
  | lut6 | 0.916 | 0.917 | 0.967 |

  - 0.99 未満の位置は、音声の部分 (20 ms ずつ) で fp16 と lut8 が 260 のうち 0〜1 (英語)、327 のうち 2〜8 (日本語)。低いのは無音の部分。lut6 は音声の部分でも 201/260、234/327
  - 層ごとに切ると、Neural Engine で層 4 0.99873、層 12 0.99962、層 20 0.99959 (最悪の位置 0.93、無音の部分)、**層 22 で 0.992、位置 1457 が 0.155**。層 21 が無音の部分に 300 ほどの値 (行の二乗和 28.6 万。音声の位置の RMS は 0.1) の吸い込み先を作り、どの位置がそうなるかが計算のわずかな違いで入れ替わる。CPU も Neural Engine も HF からは同じくらい離れ、互いにも同じくらい離れる。16.100 の `-f16` (HF の fp16 でも同じ) と同じ現象で、Neural Engine の演算の誤りではない
  - LayerNorm を行の最大値で割ってから計算する (Anima の方法) 版 (`LayerNormMax`) を試したが、数値は変わらなかったので入れない (fp16 のあふれではない)
  - 書き起こしは下のとおり変わらない (音声の部分は合っている)
- **`qchunk`**: エンコーダの注意をクエリ 0 (1500 行を一度に、スコア 90 MB)、300、500、750 行ずつで 520、518、537、555 ms、出力は同じ。テンソル 64 MiB の目安を超えても遅くならないので、既定を 500 から 0 にした
- **書き起こしの形式** (whisper-large-v3-turbo、M3):
  - LibriSpeech の dummy (`hf-internal-testing/librispeech_asr_dummy`、73 本、481 秒、英語)。WER は HF fp32 4.43%、fp16 4.43%、lut8 4.43%、lut6 4.87%。HF fp32 と一字一句同じ本数は fp16 71、lut8 71、lut6 61。fp16 と lut8 の書き起こしは互いに同じで、違う 2 本は fp16 で動かすことの違い (HF の fp16 でも "Burkett")
  - 日本語 (`say` の 48 本)。CER は HF fp32 3.60%、fp16 3.60%、lut8 3.74%、lut6 3.94%。HF と同じ本数は fp16 46、lut8 47、lut6 43
  - **Whisper は lut8 のまま**。lut6 は主にエンコーダで落ちる (上のコサイン)
- **Qwen3.5 の形式** (`hwnpu-formats`、文脈 4096、8 トークン/回、283 トークン。加えて `hwnpu-chat -temp 0 -prefill off` で短い 12 問: 富士山の標高、17 × 23、『こころ』の作者、日本一長い川、光の速さ、水の化学式、フランスの首都、1〜10 の和、Go の Max、「猫に小判」、英訳、太陽系で一番大きい惑星):

  | モデル | 形式 | 重み | 参照と同じ次のトークン | ロジットの相対誤差 (平均 / 最悪) | perplexity | 12 問の正解 | 繰り返しで止めた答え |
  |---|---|---|---|---|---|---|---|
  | 0.8B | fp16 | 1.40 GiB | (参照) | | 4.35 | 6 | 0 |
  | 0.8B | lut8 | 約 0.74 GiB | 274/283 (97%) | 0.057 / 0.159 | 4.39 | 6 | 0 |
  | 0.8B | lut6 | 0.53 GiB | 242/283 (86%) | 0.205 / 0.503 | 4.28 | 5 (17 × 23 = 221) | 0 |
  | 0.8B | lut4 | 0.35 GiB | 171/283 (60%) | 0.560 / 1.257 | 7.67 | 3 | 6 |
  | 2B | lut8 | 1.78 GiB | (参照) | | 3.83 | 10 | 0 |
  | 2B | lut6 | 1.32 GiB | 260/283 (92%) | 0.222 / 0.560 | 3.75 | 10 | 0 |
  | 2B | lut4 | 0.88 GiB | 214/283 (76%) | 0.625 / 1.313 | 4.66 | 6 (1〜10 の和 = 52) | 5 |

  - 2B の fp16 は作れない: 最後の層と出力層 (語彙 248320 × 2048) で 1.04 GiB になり、1 部分の 0.97 GiB を超える。それで 2B の参照は lut8
  - **2B の既定を lut6 にした** (`qwen35.Config.Format`、hidden 2048 のとき)。次のトークンは lut8 と 92% 同じで、perplexity も上がらず、12 問の正解も同じ数 (違うのは lut8 が『こころ』を、lut6 が英訳をしなかったこと)。重みは 1.32 GiB (lut8 の 74%)。生成は 1 秒 33.5 トークン (lut8 は 19.5。別の処理と重なった測り方で、目安)
  - **0.8B は lut8 のまま**。lut6 は fp16 との一致が 86% に落ち、計算を誤る。もともと事実に弱い (fp16 でも 12 問中 6 問) ので、これ以上は削らない
  - **lut4 はどちらも使えない**: 答えが同じ句の繰り返しになる (`hwnpu-chat` が止める) か、数を誤る
  - perplexity はこの短い文では lut6 の方が低いことがある (0.8B 4.28、2B 3.75)。量子化の誤差がこの文に偶然合っただけで、参照との一致の方を見る
  - 2B を lut6 にすると 2 本 (8 トークン/回と 128 トークン/回) の重みが 2.64 GiB で、コンパイルのキャッシュ (約 3.5 GiB) に収まるので、`-prefill auto` は 2 本を読み込む (lut8 では 3.56 GiB で 1 本だった)。M3 で 128 トークン/回の初回のコンパイルが 13 分 49 秒、2 回目からの読み込みは 2 本で 3.1 秒。Neural Engine に 2.67 GiB (lut8 の 1 本は 1.78 GiB)、プロセスの peak footprint 1.43 GB。562 トークンのプロンプトを 0.9 秒で読む。MTP (16.109) を既定で入れた後も同じく 2 本 (`-h`、`-sr` つき) で、初回は 8 トークン/回 1 分 36 秒と 128 トークン/回 13 分 40 秒 (変換を含めて peak footprint 6.9 GB)、2 回目から 3.1 秒、Neural Engine に 2.67 GiB、peak footprint 1.50 GB。同じ 4 問の生成は MTP なしの 18.3 / 23.9 / 22.4 / 24.5 トークン/秒から 27.2 / 43.7 / 30.9 / 29.1 トークン/秒に (16.109 の確かめていなかった 2B と M3)。答えは 16.109 のとおり途中の文から変わるが、4 問とも正解のまま。MTP の層 (`qwen35.MTP` は `Config` を埋め込むので `Format` も受け継ぐ) も 2B では lut6 で、下書きの受け入れは 162 中 46〜75 中 51Neural Engine のメモリを減らしたいときは `-prefill off` で 1 本 (1.32 GiB)
- 部品としての値 (16.94) の更新: Qwen3.5-2B の常駐は 2.67 GiB (ANE、2 本)、2 回目の読み込み 3.1 秒、**初回は 14 分**のコンパイル
- `hwnpu-formats` の表で「重み」が 0.00 GiB になる行がある (コンパイル済みのキャッシュから読み込んだとき)。重みの大きさは `hwnpu-chat` の読み込みの表示の方が確か
- 確かめていないこと: 2B の lut6 と HF の bf16 との KL (fp16 の 2B が作れないので、ホストか別の分け方で)、長い文脈と hwnpu-agent の課題での lut6、GPTQ の lut6 (16.97) を Qwen3.5 に
### 16.116 AMD: Qwen3.6 のプロンプトの専門家の gate と up を列の 4 コアで、行の上限 128 → 256 (256 トークンの塊は常に 1 回) — 2026-11-08

- 専門家のコマンド (B) の内訳 (2 層、手順ごとのコマンド): `MoEGroupGateUp` 21.4 ms、`MoEGroupDown` 10.8 ms。どちらも重みを約 15.6 GB/s で読む速さで決まる (gate と up 335 MB、down 168 MB)。B は回数ごとに全部の専門家の重みを読み直すので、1 層の回数がそのまま効く
- 回数 (README の先頭 256 トークン、40 層、上限 128): 40 層のうち 36 層が 2 回。いちばん多い専門家は 1 層で 107〜227 行 (256 トークンの 9 割近くが同じ専門家を選ぶ層もある)。上限を塊の大きさ (256) にすれば常に 1 回で済む
- コアのメモリ: 上限 128 で、gate と up の累算が 1 行 32 出力ずつ (fp32、32 KB)。256 行では 64 KB で入らない。そこで、列の 2 つの block (32 出力ずつ) を 4 コアに配り、各コアは block の半分 (16 出力、q4nx の 16 行の strip 1 つ) だけを持つ (`moe_gu_*`、`q4nx_strip`)。256 行でも累算は 32 KB のまま。block は memtile から 2 コアずつに同じものを送る。計算は strip ごとに前の `q4nx_accumulate` と同じ順で、丸めも同じ
- `iron_moep.CAP` と `MOE_CAP` を 256 に。Go は manifest の `cap` で回数を決めるので変えていない (前の書き出しもそのまま動く)。書き出し直しで効く
- 40 層 (q4 と GPTQ): ロジットは上限 128 の書き出しとビット単位で同じ。README の先頭 256 トークンの塊 3.9 → **3.0 s**、B の合計 2.58 → 1.68 s、NPU のコマンド 116 → 81 回。32 トークンのプロンプト 1.45 → 1.39 s。デコードは変わらない (172 ms)
- 単体の照合 (`iron_moep.py check`) は 64 トークンからは aiecc が `AIEDMATasksToNPUPass` で落ちる (main の版でも同じ)。上限を超える行は 2 層の実物の重みで、Go の上限を変えて (64、128、256 で 4、2、1 回) HF と比べた: どれもコサイン 0.98163 で同じ
- 気をつけること: `--command-per-step` の書き出しでは、Qwen3.6 の 2 層の 256 トークンの塊のロジットが HF とコサイン 0.32 (同じ checkpoint の普通の書き出しは 0.98、原因は未確認)。手順ごとの時間は測れるが、ロジットは当てにならない。なお 2 層に切り詰めた checkpoint (`Qwen3.6-35B-A3B-L0L3`) は、HF でも層 0 のほとんどのトークンが同じ 8 専門家を選ぶ (40 層の層 0 とは違う)
- ホストの合わせ (`moeHostCombine`) をトークンごとに CPU に分ける (各トークンの和の順は同じ): 40 層の合計 336 → 73 ms、塊 3.0 → **2.7 s**、ロジットはビット単位で同じ。経路の選択 (1 層 約 2.8 ms) は x の流れの 16 MB の写しがほとんど
- B の中身 (2 層、手順ごと): 重みを流すだけ (カーネルの計算を外す) なら gate と up 8.4 ms、down 4.4 ms (約 40 GB/s、16.45 の GEMV の天井 50 GB/s の 8 割)。計算込みでは 19〜23 ms と 15〜17 ms で、**計算で決まる** (memtile と L1 の object の深さを 2 → 3、4 にしても変わらない)。カーネルは 16 要素のベクトルにスカラーを掛ける MAC で、行列の積の命令 (mmul) を使っていない
- down も列の 4 コアで (`moe_dn`、`moe_dn2`、strip ごとに前と同じ計算): B の合計 1.67 → 1.49 s、塊 2.7 → **2.6 s**、ロジットはビット単位で同じ
- 試して採らなかったもの: gate と up を AIE2P の bf16 の mmul (ネイティブは 8×8×4) で、専門家の行を 8 つずつ (残りは 0 で埋めた 4 行、2 行、1 行)。符号は 8 行に 1 回だけ展開し、32 入力ずつの和を mmul で出して、そのあとのスケールと最小値は前と同じ (単体の照合は誤差まで前と同じ値)。ただし 2 層の手順ごとの時間で gate と up は 23 → 25.4 ms (層 0)、18.7 → 24.4 ms (層 1) と速くならなかった。分かったこと: 符号の 4 出力ずつの組への並べ替えを 128 要素や 64 要素のベクトルの `filter_even` / `filter_odd` ですると 1 呼び出し約 29 万サイクルかかる (32 要素どうしの `interleave_unzip` 4 回なら軽い)。mmul そのものは軽いが、8 行それぞれにスケールと最小値を掛けて足す所と、行を A の形にする所で差が消える
- 残り (塊 3.0 s のとき): B 1.68 s (1 層 42 ms のうち gate と up と down の重みの読み 32 ms、共有の専門家など)、A 約 0.7 s、ホストの経路の選択と合わせ 0.43 s

### 16.117 AMD: 下書きを 1 回で確かめる版 (`--verify`、投機的デコード)、GDN の列を 64 にそろえる — 2026-11-08

- 目的: 16.109 (Apple の MTP) の投機的デコードを AMD でも。AMD のデコードはどのモデルも演算の数で時間が決まる (Qwen3.5-0.8B は bf16 / q4 / q6 で同じ 86 ms) ので、1 回の呼び出しで数行を確かめても時間はあまり増えないはず。まず全モデル共通の土台 (数行を確かめる版と状態の戻し) を作り、下書きはプロンプトから (`LookupDraft`) で測った
- **プロンプトの塊は使えない** (`ironlm -bench`): 2〜8 トークンでも塊の時間がかかる。Qwen3.5-0.8B q6 350 ms (デコード 87.6 ms の 4.0 倍)、PLaMo 3 0.8B q4 345 ms (40 ms の 8.6 倍)、Gemma 4 E2B q4 411 ms (128.5 ms の 3.2 倍)。塊を 256 にしても 330 ms (1 層約 13 ms が行の数によらない)、64 は `ProjectionGEMM` が 256 行の倍数しか取らず書き出せない
- **数行の GEMV** (`iron_q4.QGEMVRows`、`kernels/mv_rows.h` を q8_mv.cc と q6_mv.cc が読む): 重みを 1 回流し、ブロックのコードを 1 回 bf16 にして全部の行に掛ける。各行の和と丸めは 1 行の GEMV と同じ (ロジットがビット単位で同じになる)。x は 256 入力ずつ全部の行を流す (コアに x を丸ごと持たないので K の上限がない、K 9216 も通る)。出力は排出の TAP で (rows, N) に並べる。q8 3584 × 1024 (命令の負担込みの中央値): 1 行 159 µs、2 行 157、3 行 155〜162、4 行 188 (列に 2 コア。1 コアだと 283)、8 行 389。`q_linear` は 2〜8 行 (`ROWS_GEMV`) の q8 / q6 をこれで回す (q4 はまだ)
- **確かめる版** (`iron_verify.py`、`iron_export_lm.py --verify ROWS`): `shapes` に ROWS 個の id の版を足す。プロンプトの塊と同じく各行の鍵と値を書き、状態の枠は塊のもの (`Checkpoints`) を読み書きするが、全部の行のロジットを返す (最後の行から引いたトークンを運ぶ)。行は境界をつけず全部回す (埋めの行の結果は読まない)。数行の形がない演算は 1 行ずつ:
  - 埋め込み: ホストの行 (`embed_rows`) を ROWS 行
  - 短い畳み込み (`causal_conv1d_rows`): 前の行と呼び出しの行を塊の窓に置いて ROWS 行だけ
  - RoPE: 表の start から ROWS 行
  - 注意 (`KVCache.attend`): 行ごとに鍵と値を書き、デコードの `MHA` (問い 1 つ) で 1 行ずつ。出力は `rows_o` に集める (リングのキャッシュはまだ)
  - Gated DeltaNet (`iron_gdn.gdn_operator(..., steps=ROWS)`、`kernels/gdn.cc` の `GDR_STEPS`): デコードの演算子 (畳み込みとゲートつき norm を中に持つ) が 1 回の呼び出しで ROWS 行を順に回す。`gdn_rule` を 1 行 × ROWS ブロックで呼び、状態はコアの中で行から行へ持ち越す (前からある `GDR_BLOCKS` の仕組み)。各行のあとの状態 (畳み込みの前の行の組も) を出し、最後の行のものを枠に、残りを層の `rows_state` に写す。最初は演算子を 3 回呼んでいたが 1 回 1.0 ms × 54 回で、3 行の確認が 148 ms (1.73 ステップ) かかった
- **状態の戻し** (`iron.Model.Keep`): 行の途中までしか採用しないとき、ホストが `rows_state` の行 kept-1 を枠に写す (Qwen3.5-0.8B で 18 層 × 1.1 MB、1.5 ms)。鍵と値の余分な行は次の呼び出しが書き直すので触らない。本体を回し直さないので、#78 の `state-rows` (ホストで delta rule) も要らない。チャンクの境目をまたぐ確認はデコードのステップで (`SpeculateTokens`)
- **GDN の列を 64 にそろえる** (`iron_gdn.COLUMN_ALIGN` 8 → 64): 0.8B と 2B の融合した射影は 776 × 8 = 6208 行 = 97 個の 64 行の組で、97 は素数なので GEMV が 1 列でしか回らなかった (1 行で 571 µs、6656 行に埋めると 8 列で 217 µs)。4B と Qwen3.6 の 1032 (129 組) は 3 列。埋めは重みが 0 の行なので、ロジットは前の書き出しとビット単位で同じ (README の 256 トークンの塊と 63 ステップ)。デコード 85.6 → 82.5 ms
- 制御コード: 確かめる版の ELF は 30 MB (デコード 26 MB)。デバイスのヒープ (64 MiB) に両方を置いたまま回す。入らないときは呼び出しごとに作る (`perCall`、2 ms)。プロンプトの塊や塔がヒープに入らないときは両方を手放す (`dropDecode`)。演算子を 3 回呼んでいた版は 47 MB で入らなかった
- **照合** (`ironlm -verify`、500 トークンの後): 3 行のロジットはデコードのステップ 3 回とビット単位で同じ (コサイン 1.000000、誤差 0)。1 行または 2 行だけ採用して `Keep` したあとのデコードのステップも、その行までをデコードで回したものとビット単位で同じ
- 速さ (Qwen3.5-0.8B q6、3 行): デコード 82.2 ms、確かめる版 **104.5 ms (1.27 ステップ)**。`hwnpu-chat -iron` (temp 0、300 トークン、出力はどちらも同じ文):

| | 下書きなし | プロンプトからの下書き 2 つ |
|---|---|---|
| 英語でコード | 82.4 ms/token | **72.1** (271 個のうち 78 個採用、222 回) |
| 日本語の説明 | 82.5 ms/token | **61.7** (239 個のうち 109 個採用、190 回) |

- 確かめる版の内訳 (手順ごとのコマンドで、GDN を 3 回呼んでいたとき): 数行の GEMV 157 回、`Copy` 226 回 (状態 1.1 MB の写しが 1 回 0.3 ms)、RMSNorm と要素ごとの演算が 1 回 0.2 ms
- 使い方: `iron_export_lm.py qwen35 CKPT OUT --q4 --mlp q6 --head q6 --verify 3`。`hwnpu-chat -iron OUT` は版があれば 2 つの下書きを確かめる (`HWNPU_IRON_DRAFTS=0` で使わない)
- 残り: q4 の数行の GEMV (Gemma 4、PLaMo 3、LFM3 の 4 ビット)、Gemma 4 の共有の層 (最後の行だけの形) と窓の注意、MTP の層を下書きに (本体の隠れ状態を出し、MTP の層の版を足す)、注意を数行の MHA 1 回に、状態の写しを減らす

### 16.118 M1 だけ `-prefill auto` の予算を 2.8 GiB に (`neuralEngineHold`)、一緒に読み込むプログラムも数える — 2026-11-09

- 動機: M1 は、順に回す部分を Neural Engine が 2.86 GiB 持つまでは単独の合計どおりで、2.97 GiB から 1 回ごとに約 12 倍遅い (16.109、PROBE.md)。M2 は同じ大きさ (`npuprobe split -formats lut4 -parts 8 -split-layers 96,100,104,128`) で 3.65 GiB まで単独の合計どおり (46.8 / 48.6 / 50.5 / 62.2 ms、63 GB/s)。M3 も Qwen3.5-4B の lut8 (1 本 3.96 GiB) が見込みどおりの速さで、M4 は 3.62 GiB で全速。境目は M1 だけ
- 判断: 全部のチップを 2.8 GiB にそろえると、M2 以降で LFM3-VL / Audio / Omni がプロンプト用を失い、E4B と Qwen3.5-4B は形式を下げることになる。変換するプログラムはもともとチップによらず同じファイル (部分は `-target any` の 0.99 GiB、名前にチップは入らない) なので、M1 では読み込むプログラムを減らすだけにした。Hugging Face から変換済みのものを落とす形にしても、置くファイルの組は全チップで共通で、M1 はプロンプト用を使わない
- 変更 (`internal/chat/model.go`):
  - `neuralEngineHold`: `machdep.cpu.brand_string` が `Apple M1` で始まれば `m1HoldBytes` (2.8 GiB)、ほかは 0 (境目なし)。M1 Pro / Max / Ultra は同じ Neural Engine なので同じに扱う (測っていない)
  - `onePrograms`: 2 本と、一緒に読み込むプログラムの重み (`besideBytes`: MTP の層、塔は fp16) の合計が境目を超えると 1 本。コンパイルのキャッシュ (3.5 GiB) との比べは今までどおり 2 本だけで、M2 以降の判断は変わらない
  - 生成用 1 本でも境目を超えるとき (Qwen3.5-4B の lut8、E4B の lut6) は、ログに「expect it to be slow on this machine」と出す。変換と読み込みは止めない
  - 見積もり (`programBytes`、`besideBytes`) は実物のファイルと合う: Qwen3.5-2B の lut6 で生成用 1.321 GiB (ファイル 1.322)、MTP の層 0.400 GiB (0.400)。16.109 の 0.8B の MTP 0.26 GiB に対し、2B は 0.40 GiB
- テスト: `TestPreparePrefill` (M1 の境目より小さい・大きい模型で 2 本と 1 本、警告)、`TestOneProgramsBeside` (一緒に読み込む分を M1 でだけ数える)、`TestBesideBytesMTP`、`TestPrepareEncoders` (塔の重みを数える)。境目の比べから一緒に読み込む分を外す改変、`neuralEngineHold` を使わない改変はどちらも落ちる
- M1 (16 GB、macOS 27.2) で Qwen3.5-2B (lut6、MTP): 「2 programs would hold 2.64 GiB on the Neural Engine with the 0.40 GiB of the programs beside them, over the 2.8 GiB an M1 holds」で 1 本。全部 Neural Engine、Neural Engine が持つ量 0.89 GiB (読み込み直後。最初の呼び出しで載る分は含まない)、17 × 23 = 391 を 1 秒 34.5 トークン (114 の下書きのうち 62 採用)
- 確かめていないこと: LFM3-VL / Audio / Omni を M1 で 1 本にしたときの生成と長いプロンプトの読み (別に計測中)、M1 Pro / Max / Ultra の境目、Q=8 と Q=128 を 1 つのパッケージの関数にして重みを共有できるか (できれば、どのチップでも 2 本の分が半分になる)

### 16.119 AMD: Sana と Sana-Sprint (`iron_sana.py`、`internal/imagegen/sana`) — 2026-11-09

- 動機: Anima より軽い画像生成の枠 (ライセンスは Apache 系)。Sana-Sprint は 2 ステップ、潜在が 32 分の 1 で 1024 × 1024 でもトークン 1024 個、自己注意は線形
- 対応: Sana-Sprint 0.6B と 1.6B、Sana 600M、SANA 1.5 1.6B (1024 × 1024)。テキスト (Gemma 2 2B、q8)、caption、DiT、DC-AE の VAE の全部が NPU。CPU の参照 (`tools/ref/ref_sana.py`) と同じノイズから潜在のコサイン 0.997 / 0.963 / 0.983 / 0.984 (Sana 600M は CPU の bf16 でも 0.962)。1 枚 24 秒 / 70〜80 秒 / 32 秒 / 約 2 分、固定の最大 2.3 GiB。記録は amd-npu-models.md の 14 節
- IRON に割り算がないので、線形注意の `N / D` は値の行に定数 c を混ぜて `[N, cD, 0…]` の行を作り、64 列の RMS 正規化で割る (14.2)
- 見つけたこと: IRON の MHA はスコアを bf16 で持つ。Sana 600M の交差注意 (q と k の正規化なし) はスコアが約 1750 で行の中の差が数単位しかなく、条件つきの出力だけがずれて CFG で崩れた (潜在 0.79)。鍵から先頭の鍵を引いて (softmax は変わらない) 0.963。Anima の交差注意にも同じことが起きていないかは見ていない
- 共有の変更: `internal/tokenizer` のバイトフォールバックは、Gemma 2 の語彙に `<0x09>` がなく (タブは普通のトークン) 読めなかった。文字そのものが語彙にあるバイトは `<0x..>` がなくてよいことにした (encode は文字を先に引くので使われない)。`TestTokens` (`HWNPU_SANA_TOKENIZER`) が HF の id と 3 つのプロンプト (日本語、絵文字、タブ) で一致
- `hwnpu-image`: `-model` が Sana の書き出し (`model.json` と `caption/`) なら Sana で描く。`-steps` の既定はモデルごと (Anima 8、Sana-Sprint 2、Sana 20)、`-guidance` (既定 4.5)、`-noise` は Sana-Sprint ではステップ + 1 個

### 16.120 AMD: Depth Anything 3 の深度 (`iron_da3.py`、`hwnpu-depth`) — 2026-11-09

- 動機: Apache-2.0 の軽い画像の分析。DA3MONO-LARGE (0.33B、DINOv2 ViT-L と DPT) は 1 枚の深度に特化した版、DA3METRIC-LARGE は同じ構造でメートル、DA3-SMALL / BASE は複数視点の版を 1 枚として (深度と確信度)
- 504 × 378 で LARGE 0.7 秒、SMALL 0.26 秒 (推定だけ)。固定の最大 0.49〜1.39 GiB。公式の CPU (fp32) と深度のコサイン 0.9993〜0.9999。記録は amd-npu-models.md の 15 節
- 書き出しは縦横比ごとの大きさ (長辺 504 の 5 つ) を持ち、hwnpu-depth が近いものを選ぶ
- 新しく作った部品: 行を 16 の倍数に足した格子 (`Grid`)、ストライド 2 の 3 × 3、転置畳み込み (GEMM と画素への Copy)、双線形の拡大 (補間の行列の GEMM 2 回と転置の Copy)、2 次元の RoPE (q と k の次元の並べ替え)、ヘッドのチャネルの 64 の倍数への埋め
- `hwnpu-depth`: 画像ごとに NAME.depth.png (逆深度を 8 ビットに、元の大きさで)、`-raw` で float32、`-sky` で空か確信度、`-focal` / `-fov` でメートル (METRIC)。`internal/vision/depth` の `TestQuantile`、`TestSky`、`TestPatches`、`TestNearest`

### 16.121 AMD: MoGe-2 (`iron_moge.py`、`hwnpu-depth`) — 2026-11-09

- 動機: 焦点距離を知らなくてもメートル単位の深度、点群、法線が出る (MIT)。DA3METRIC は焦点距離が要る
- 構成は DINOv2 と ConvStack の neck と 3 つのヘッド。正規化は使われていない (設定の `res_block_*_norm` が none)。3 × 3 は端の画素を複製する詰め (`Grid(rep=True)` と `fill_pads`)、双線形の align_corners=False の行列、torch なしで `model.pt` を読む `TorchZip`
- 3 つのヘッドは 1 つのプログラムに重みのバンクを 3 組 (`SwapVariants`)。全部を載せたままだと固定 3.9 GiB (ヘッドの作業領域 773 MiB × 3 と neck 736 MiB)
- 812 × 602 で vits 約 2.7 秒、vitl 約 3.6 秒。公式の CPU と深度のコサイン 0.9999、相対誤差の平均 0.7〜1.2 %。記録は amd-npu-models.md の 16 節
- 後処理は Go (`internal/vision/moge`): 焦点距離と z のずれは 1 変数の Levenberg–Marquardt (`TestRecoverFocalShift` が既知の値を戻す)

### 16.122 Sana の負のプロンプトと正方形以外、VAE の段を先に読む、深度推定の大きさと複数の画像、MoGe-2 の画角 — 2026-11-09

- Sana: `-negative` (指示文なしの Gemma 2 をもう 1 回)、`iron_sana.py --width --height` (Sana のビン。トークンと VAE の密な行を 256 の倍数に 0 で埋め、線形注意の鍵から消す)。VAE は次の段を読み込みながら走らせ、出力を次の入力へ直接写す (Sana 15 → 13 秒、Anima 11 → 9.4 秒、画素は同じ)。Anima の交差注意のスコアは小さく (最大 8)、Sana 600M のような bf16 の問題はない
- 深度推定: 引き伸ばしの誤差 (公式の大きさとの比べで 6〜12 %) が NPU の誤差 (1 %) より大きかったので、書き出しを公式の大きさで 9 つにし、画像の公式の大きさを優先する
- DA3 の複数の画像 (`--views N`、`hwnpu-depth` が `cameras.json` と `-ply`): 視点のトークンを詰めて並べ、視点の中の注意は窓、視点をまたぐ注意は先頭の範囲。参照する視点の選択とカメラの MLP はホスト。3 枚で 1.1 秒、公式と深度 0.9998
- MoGe-2: `-fov` で画角を与える (z のずれだけを解く)
- 対象外: DA3NESTED / LARGE / GIANT は CC BY-NC
- 記録は amd-npu-models.md の 11、14〜16 節

### 16.123 hwnpu-image をサブコマンドに (generate、depth、convert)、引数は kong — 2026-11-10

- 画像のコマンドが増えてきたので、`hwnpu-depth` を `hwnpu-image depth` にまとめ、`hwnpu-image -convert` を `hwnpu-image convert` に、絵を描くのを `hwnpu-image generate "プロンプト"` にした。`hwnpu-depth` は消した
- 引数は標準の flag から `github.com/alecthomas/kong` に。サブコマンドごとに構造体のフィールドとタグで書き、`--help` が出る。フラグは `--model` (`-m`)、`--out` (`-o`) のように 2 本の `-`。プロンプトと画像は位置引数。generate と convert は Anima の書き出しの設定 (`--backend`、`--size`、形式など) の構造体を共有する
- 今回は hwnpu-image だけ。ほかのコマンド (hwnpu-chat、hwnpu-transcribe など) は標準の flag のまま

### 16.124 hwnpu-image のモデルを models.yaml から選ぶ、-o なしで開く — 2026-11-10

- models.yaml に種類 `image` (Anima-Turbo、Sana-Sprint 0.6B / 1.6B、Sana 600M、SANA 1.5 1.6B) と `depth` (DA3 の 4 つ、MoGe-2 の 3 つ) を足した。用途 `image` は Anima-Turbo、`depth` は da3 と同じくメモリで (8 GB は DA3-SMALL、16 GB は DA3-BASE、24 GB 以上は DA3MONO-LARGE)
- `hwnpu-image generate -m sana` はファミリーなので、hwnpu-chat と同じくマシンのメモリで選ぶ (Sana-Sprint 0.6B が 8 GB、1.6B が 16 GB から)。固定のメモリはどれも 2.3 GiB で同じなので、境目は速さ (1.6B は 1.6 倍) と質の釣り合いで決めた。20 ステップの Sana はファミリー `sana-full`。メモリはタグでも明示できる (`sana:8g`、`8G`、`8GB`、`8 g`、表の列にない `12g` も、そのメモリのマシンの選び方)
- `hwnpu-image list [image|depth]`: hwnpu-chat の -list と同じ表 (`chat.ListHave`) で、ダウンロードの列の代わりに AMD の書き出し (`~/models/NAME.amd-npu2`)。`-m` の名前が合わないとき (`chat.NoModelError`) も、その種類の表を出してからエラーにする
- 名前で選んだモデルは AMD の NPU では `~/models/NAME.amd-npu2` か `NAME-1024.amd-npu2` (小文字) を使う。ないときは、落とすリポジトリと `iron_*.py` のコマンドを示して止まる (書き出しを自動で作るのはまだ)。Mac / Intel の Anima は今までどおり自動で落として変換する。`/` や `.`、`~` を含む値は書き出しのディレクトリ
- `-o` を省くと `image.png` (あれば `image (2).png`、…) に書き、OS のビューアで開く (macOS は open、Windows は rundll32 の FileProtocolHandler、ほかは xdg-open。DISPLAY も WAYLAND_DISPLAY もないときは開かない)

### 16.125 AMD: YuNet の顔検出 (`iron_yunet.py`、`hwnpu-image face-detect`) と小さなモデルの NPU の費用 — 2026-11-10

- OpenCV Zoo の YuNet (MIT) を 1 つの IRON グラフに (docs/amd-npu-models.md 17 節)。1 × 1 と深さ方向の 3 × 3 を 1 つの畳み込みに、バイアスは「1 のチャンネル」に、ReLU は ProjectionGEMM の clamp (下限 0、`ReluGEMM`) に。結果は OpenCV の FaceDetectorYN と一致
- models.yaml に種類 `face-detect` と YuNet (用途 `face-detect`)。`hwnpu-image face-detect` は JSON 行を出し、`--draw` で描いた画像、`--repeat` で時間
- `iron.GraphOptions.KeepRuns`: コマンドの run を一度作って持ち続ける。毎フレーム呼ぶ小さなグラフで、ホストの CPU が 1.6 → 1.1 ms。制御コードがデバイスのヒープ (64 MiB) に残るので、大きなグラフには使わない
- 測定の結論: 320 × 256 で NPU 29 ms・127 mJ、CPU (onnxruntime 1 スレッド) 2.7 ms・7.7 mJ。NPU は CPU を約 2 ms 空けるが、エネルギーは 16 倍。演算の切り替え (GEMM と他で約 250 µs) が支配的で、小さなモデルでは「NPU に置けば省エネ」は成り立たない。置き場所の方針 (CPU を空けるか、エネルギーか) はこの数字で決める

### 16.126 置き場所をベンダーごとに (models.yaml の device)、ONNX Runtime の CPU 実行系 — 2026-11-10

- 16.125 の結論は AMD の NPU (IRON) の話で、NPU 一般の話ではない。Apple の Neural Engine と Intel の NPU はグラフをまとめて投入するので、小さなモデルでも NPU が得かもしれない。そこで置き場所をモデルごと・NPU のベンダーごとに書く: models.yaml の `device: {amd: cpu}` (amd、apple、intel に npu か cpu。書かないベンダーは npu)。ユーザーの models.yaml は名前を挙げたベンダーだけを変え、ほかを残す
- 上書きの順は、コマンドの `--device`、`HWNPU_DEVICE`、models.yaml (`chat.ChooseDevice`、理由も返す)。ゲームなどで CPU を空けたい人は `--device npu`
- `internal/onnxrt`: ONNX Runtime の共有ライブラリを purego で呼ぶ (cgo なし)。OrtApi は API 16 を求め (それより新しいライブラリならどれでも)、関数は表の位置で呼ぶ。1 スレッド既定、スピンなし (`session.*.allow_spinning = 0`)。ライブラリは `HWNPU_ONNXRUNTIME` かシステムのもの。自動で取ってくることはしない (ネイティブのライブラリの配布は利用者が決める)。`FreeShapes` は ONNX の入出力の形と value_info を protowire で外す
- face-detect: AMD では既定で CPU (YuNet の ONNX は models.yaml の sha256 で固定して自動で取得)、`--device npu` で export。Apple と Intel では YuNet の NPU のグラフがまだないので CPU に落とし、その旨を出す (`--device npu` を明示したときはエラー)
- `hwnpu-image list` の列を RUNS FROM に: このマシンで CPU に置くモデルは「CPU (ONNX Runtime)」、ほかは AMD の書き出し
- Go のコマンドで 320 × 256: CPU 3.96 ms・10 mJ、NPU 30.5 ms・137 mJ。前処理が 11 ms かかっていたので、lfm3vl.RGB に速い道を足した (結果は color.Convert と同じ。テストで確かめる)

### 16.127 Qwen3-0.6B: 密な Qwen3 (`qwen3`)、全層が注意のモデルは分けずに、チャットの型 `chat.Qwen3` — 2026-11-09

- 動機: 8 GB の機械のチャットに、一番小さなダウンロード (1.5 GB) のモデルを足す。Qwen3-0.6B (Apache-2.0) は Anima のテキストエンコーダとして動いていたので、構造とトークナイザーは確かめてあった (16.68)。docs/model-table.md 12.2 節の `decoder` 型の最初の一歩でもある (名前は GGUF の `general.architecture` に合わせて `qwen3`。`llama` と `mistral` を受けるときに見直す)
- 移植 (`internal/llm/qwen3`): RMS 正規化、Q と K のヘッドごとの正規化のあとに RoPE (θ = 1e6)、GQA (16 / 8 ヘッド、head_dim 128)、SwiGLU。出力の行列は埋め込みと共有 (`tie_word_embeddings`。チェックポイントの `lm_head.weight` は transformers が読み込みで置き換える写しなので読まない)。`rope_scaling`、窓、`attention_bias`、`full_attention` 以外の `layer_types` はエラー。transformers 5 の `rope_parameters` の `rope_theta` も読む
- 照合: fp64 の正解との 1e-4 (共有あり・なし)、1 層ずつ、キャッシュつきの段の一致、部分に分けた形、注意で分けた形 (`TestPagedMatchesFull` など)。k_norm を外すとテストが 5 つ落ちる。実物の重み (`hwnpu-check`、32 トークン、チャットの型の文): fp32 で全層 0.0012 以下、KL 0.0000、32/32。fp16 も KL 0.0000、32/32。lut8 は KL 0.0139 (31/32)、lut6 は 0.2425 (23/32) なので lut8 を既定にした
- fp16 の手当て: 残差は層 2 から最大 8141 になる (外れて大きい活性)。正規化は Gemma 4 と同じく入力を 1/64 にしてから二乗する。ただし層 0 の入口は埋め込み (最大 0.1 程度) なので、1/64 にすると二乗が fp16 の下に落ちて NaN になった (`hwnpu-check -f16` の層 0)。層 0 の入口の正規化だけ縮めない
- 層 0 の注意もホストで (`llm.NoLayers`): `PlanPaged` は層 0 の注意で分けるのを断っていた。全層が注意のモデルのために、層を持たない最初の部分 (層 0 の正規化と射影だけで、`attn_q_0` と K / V を出す) を作るようにした。ほかのモデルの計画は変わらない
- 分けない方が速い (`llm.Unpaged`): M1 (16 GB) で、8192 の塊に分けた形は 1 トークン約 400 ms (3 tokens/s)。内訳 (`HWNPU_PROFILE=1`) は部分 29 本がそれぞれ 3〜5 ms、ホストの注意 (塊 1 つを読むプログラム) が 28 回それぞれ約 10 ms。塊 1024 でも 5.9 tokens/s。分けない 1 本 (キャッシュをプログラムの中に) は 4096 で 11〜12 tokens/s で、答えは分けた形と一字一句同じ。ただし 8192 と 16320 のキャッシュはコンパイルと読み込みは通るが、予測で Core ML がエラーを返す (28 層の K と V で入力が 56 本、計 0.94 / 1.9 GB)。そこで `llm.Unpaged` (`UnpagedContext` 4096) を足し、hwnpu-chat は `-paged N` を与えたときだけ分ける。`-paged 1024` で文脈 32768
- チャットの型 (`internal/chat/qwen3.go`): ChatML、Hermes 式の道具 (`<tools>` の JSON、`<tool_call>{"name": …, "arguments": {…}}</tool_call>`)、思考は最後の質問より後の回答だけ。引数は Python の `json.dumps` と同じに書く (`1e20` は `1e+20`)。テンプレートは思考ありでも `<|im_start|>assistant\n` で終わりモデルに `<think>` を書かせるが、Render は `<think>\n` まで書いて考えている途中から始める (Qwen3.5 と同じ扱いで、流す答えが分かれる)。jinja2 の 9 会話と一字一句一致 (`template_golden.py --text-content`: 部品の列の content を文字列にしてから描く。Qwen3 のテンプレートは列を描かないか落ちるため。vLLM も同じく文字列にする)
- M1 での答え (温度 0、思考なし): 17 × 23 = 391 は正しく、富士山 (1,380 m)、『こころ』、信濃川は誤る。HF の fp32 でも同じように誤る (1,852 m、関山川) ので、移植や量子化ではなくモデルの限界。思考ありでは 17 × 23 を 802 トークン考えて 391
- メモリと速さ (M1、2 回目の起動): 2 本の読み込み 0.4 s と 1.4 s、Neural Engine 1.16 GiB、プロセスの footprint 2.75 GB、生成 9.5〜12.4 tokens/s。初回のコンパイルは 1 本 30〜50 s (分けた形は 4 分 42 秒と 6 分 39 秒)
- 用途の表 (models.yaml の `chat`) の 8 GB の列は LFM3-1.7B のまま。Qwen3-0.6B は `-use qwen3` (`qwen3:0.6b`) で選ぶ。MODEL.md の 8 GB の欄に並べた
- 確かめていないこと: M2 以降と M3 の速さ、6144 などのキャッシュが通るか、プロセスの 2.75 GB の内訳 (2 本のプログラムがそれぞれ持つキャッシュの入出力の写しと見ている。`-prefill off` で減るか)、Intel の NPU、AMD の書き出し (`tools/amd/iron_anima.py` の `Qwen3Layer` を `iron_export_lm.py` から使う)、hwnpu-agent の課題

### 16.128 SwiftF0 (音高、`internal/pitch`、`hwnpu-pitch`) — 2026-11-09

- 何か: lars76/swift-f0 0.3.0 (MIT)。16 kHz の声や楽器 1 つの音高 (46.875〜2093.75 Hz の 95 候補) と有声の確からしさを 16 ms ごとに出す、14,386 パラメータの CNN。docs/small-models-survey.md の最初の候補
- 配布の形: ONNX の 1 本だけ (GitHub と PyPI)。Hugging Face の `aaatmy/swift-f0-onnx` の `model.onnx` が公式の 0.3.0 とバイト単位で同じ (sha256 `6385e8c2…`) なので、models.yaml ではそのリビジョンと sha256 で固定し、`internal/pitch` も同じ sha256 を確かめる。ONNX は依存なしで読む (`onnx.go`: protowire で初期値だけ。グラフは読まず、層の並びは Go に書いた)
- 分け方: 前段の 3 つの STFT (窓 1024 / 2048 / 512、ホップ 256、周期的な Hann、フィルタバンクで 128 帯、log(+1e-8)) はホストで float64 (`Features`、基数 2 の FFT)。CNN (12 チャンネル、6 つの残差ブロック、帯の方向の拡張 17〜48) を NPU のプログラム 1 本 (1896 フレーム = 1875 + 前 11 + 後 10)。ロジットから音高を出す最後 (最良の候補と両隣の softmax で log 周波数を平均) はホスト (`decode`)。`SwiftF0.detect` と同じく 1875 フレームの窓で区切る
- 固定長のプログラムで短いクリップを正しく: グラフはどの畳み込みもクリップの端でゼロを詰める。プログラムは `mask` (クリップの外で 0) を各 ReLU の出力と入力に掛け、クリップの外を常にゼロにする。13 フレームのクリップを 24 フレームのプログラムで流しても、13 フレームのプログラムと同じ (`TestProgramMatchesReference`)。マスクを外すとテストが落ちる
- Neural Engine: 拡張 17、27、40、48 の畳み込み (カーネル 7 / 5 / 3 × 1) を Core ML が CPU に置いた (M1、12 演算)。拡張 2 は NPU。そこで拡張の大きな畳み込みを、帯をタップごとにずらした k 本 (ずれた分はゼロ) をチャンネルに重ねた 1 回の 1×1 の畳み込みにした (`shifted`、数学的に同じ)。全 115 演算が NPU (`Revision` 3)
- 照合 (`tools/ref/ref_swiftf0.py`: onnx の参照の評価器で公式のグラフを流す。onnxruntime は Python 3.15 の wheel がない。最後の GatherElements は評価器が受けないので numpy で書いた): 3 秒と 41.3 秒 (窓 2 つ、`fmin 80`、`fmax 1000`) の試験信号で、前段の差 0.0014 (振幅 e^-6 より上の帯。弱い帯ではグラフの float32 の DFT の丸めで 0.04)、ロジットの相対誤差 2e-4、有声 / 無声は全フレーム一致、音高の差は最大 0.11 cent (CPU の fp16)。NPU では 2581 フレームすべて一致、最大 0.23 cent、確からしさの差 0.0015
- 速さとメモリ (M1): 41.3 秒を 0.24 秒 (169 倍速。前段を疎なフィルタバンクと前計算の回転因子にする前は 0.66 秒)、CPU のインタプリタでは 3.1 秒。プロセスの footprint 58 MB、読み込み 1.9 秒 (コンパイル済み)
- 使い方: `hwnpu-pitch voice.wav` (CSV: 時刻、音高、確からしさ、有声、音量)、`-json`、`-fmin`、`-fmax`。models.yaml に種類 `pitch` (`hwnpu-pitch`) と用途 `pitch` (`f0`) を足した
- 確かめていないこと: 実際の歌声と楽器 (試験信号だけ)、Intel の NPU、AMD (書き出しなし)、ストリーミング (`PitchStream`) と音符への区切り (`segment_notes`)

### 16.129 ONNX だけで配られるモデル (`internal/onnx`)、RTMO (`hwnpu-pose`)、Basic Pitch (`hwnpu-notes`) — 2026-11-09

- 動機: docs/small-models-survey.md の候補のうち、RTMO (姿勢) と Basic Pitch (音符) は ONNX でしか配られていない。どちらもグラフに静的な部分 (CNN、Transformer の encoder) と動的な部分 (TopK と NMS、人ごとの分類器。窓ごとの CQT) が混ざる。モデルごとに名前で手で移すより、ONNX を読んで静的な部分を MIL に下ろす共通の道具を作った
- `internal/onnx`: 依存なしで ONNX を読む (protowire。ノード、属性、初期値、入出力)。ホストの評価器 (float64、numpy の放送。形の計算、TopK、NonMaxSuppression、Conv、Pad (定数と反射)、MatMul、Softmax、Reduce* など) と、静的な部分の MIL への下ろし (`Program`)。下ろすときは、形と定数だけから決まる値 (Shape、Gather、Concat の連鎖) をホストで畳み込み (Shape はプログラムの値の静的な形)、入力から計算する値だけをプログラムにする。Pow 2 は mul、Neg は mul -1、Clip は maximum と minimum、定数の行列との MatMul は linear。`Graph.Cut` は、指定した動的な演算とそれに依存するノードを尾部として、尾部が読む静的な側の値をプログラムの出力にする。尾部は `EvalShapes` で、プログラムの出力と畳み込んだ値と形から、グラフそのものをホストで評価する (元のグラフと同じ計算)
- Neural Engine 向けの書き換え (M1 で CPU に落ちたものから): 窓 9 と 13 の max pool (ストライド 1) を窓 5 の連続に (同じ最大値)。形の違う出力の連結は、1 次元が 16384 を超えないよう、全出力の大きさの公約数を行数にした (1, 行, 幅) で (`Flat`)。最後の軸での連結は Basic Pitch で出力が崩れた (note の活性 59% ずれ、1 つか 2 つなら正しい) ので、Basic Pitch も `Flat` にした
- milcpu に足した演算: erf、cos、sin、pow、max_pool、pad の reflect と replicate
- **RTMO-s** (`Xenova/RTMO-s`、Transformers.js 向けの書き出し、Apache-2.0、9.9M): 640×640 に引き伸ばした RGB (0〜255) から、2000 アンカーの人のスコア、箱、姿勢の特徴 (256)、関節の見え方 (17) までが NPU (370 演算すべて)。TopK、NMS、選んだ人ごとの座標分類器 (DCC、GAU と sin / cos の位置埋め込み) はホストでグラフを評価。照合 (onnx の参照評価器、football-match.jpg): CPU インタプリタの fp16 で 4 出力の相対誤差 0.0004〜0.005、人 5 人 (スコア 0.3 以上は 4 人) と箱の差 1.6 px、見える関節点の差 3.4 px。重みだけ fp16 に丸めて fp32 で計算しても 3.4 px (重みの丸めそのもの)。M1 で 1 枚約 100 ms (縮小 55 ms、ネットワーク 9〜100 ms、ホスト 36〜43 ms)、footprint 232 MB
- **Basic Pitch** (`nmp.onnx`、Apache-2.0、36K。Hugging Face のミラー `AEmotionStudio/basic-pitch-onnx-models` が公式とバイト単位で同じなので sha256 で固定): 22050 Hz の 2 秒の窓の CQT (サンプルの軸 43844 が 16384 を超える) と対数正規化はホストでグラフを評価、倍音の積み重ねと CNN を NPU。窓の切り方とつなぎ方 (30 フレームの重なり)、音符の組み立て (`output_to_notes_polyphonic`、推定した onset、melodia の手法、ピッチベンド) と MIDI の書き出しは basic-pitch の Python を Go に移した。照合 (`tools/ref/ref_basic_pitch.py`: 参照評価器と、basic-pitch の note_creation.py そのもの): 参照の活性から作る音符は完全に一致。CQT の相対誤差 1.0e-4、CPU の fp16 の活性 2.2e-4、5 秒の試験の旋律と和音の 10 音符が一致。Neural Engine では活性の相対誤差 2% で、音符は同じ 10 個 (誤検出の 1 つの終わりだけ違う)。5 秒を 0.14 秒 (35 倍速)、footprint 75 MB
- 確かめていないこと: Basic Pitch の倍音の積み重ねの後の (3, 39) の畳み込みと ReLU が CPU に残る (核を幅 16 以下に分けても CPU。原因未詳)。RTMO-m / l (切り口は自動で決まるので同じコードで動く見込み)、実際の録音と写真、Intel の NPU、AMD (書き出しなし)、RTMO の画像の縮小 (55 ms) の高速化

### 16.130 Magenta RealTime 2 small (音楽生成、`internal/music`、`internal/tflite`、`hwnpu-music`) — 2026-11-09

- 何か: Google DeepMind の Magenta RealTime 2 (コード Apache-2.0、重み CC-BY 4.0。使うときは「Magenta RealTime 2 by Google DeepMind」と表示)。small は 230M。文で指定した曲調 (MusicCoCa の 12 の符号) から、SpectroStream の 12 段の RVQ の符号を 1 秒 25 フレームで生成し、48 kHz ステレオに復号する。`google/magenta-realtime-2` のリビジョンで固定 (チェックポイント 1.13 GB と MusicCoCa の TFLite 3 本、spm.model。計 1.7 GB)
- 部品と置き場所:
  - 時間方向の言語モデル (12 層、1024 次元、8 頭 × 128、FFN 4096 で gelu の tanh 近似とバイアス、RMSNorm を前後に、per_dim_scale、注意ごとに 1 つのシンクのキーと値 (ロジットはスケール前の query で)、自分と条件への注意とも直前 41 + 今の窓、位置埋め込みなし) を NPU のプログラム 1 本。キーと値は外のリングに持ち、新しい 1 フレーム分を出力で返す
  - 深さ方向のデコーダ (2 層、768 次元、6 頭、1024→768 の写像、最後の LayerNorm、12294 のロジット、softcap 30) を NPU のプログラム 1 本。1 フレームに 12 回 (段ごと) 呼ぶ
  - サンプリング (top-k 40、温度 1.3、Gumbel、その段の符号の範囲だけ) と条件 (144 チャンネル、+7。CFG の強さはビン分けした条件の符号で渡すので、バッチを 2 倍にしない) はホスト
  - SpectroStream の復号 (時間に因果な 2 次元の畳み込みと転置畳み込み、ELU、チャンネルの分割は重み共有のバッチ 2、先読み 4 STFT フレーム) を NPU のプログラム 1 本 (25 フレーム + 前 16 フレーム)。iSTFT (窓 960、ホップ 480) はホスト
  - MusicCoCa (文 → 曲調): 小文字にして SentencePiece の Unigram (`tokenizer.LoadSentencePieceUnigram`: spm.model を直接読む。nmt_nfkc の charsmap、byte fallback)、TFLite の文の符号器 → 写像 (ノイズは numpy の `RandomState(0).randn` を Go で再現、`LegacyNormal`) → L2 正規化 → RVQ。TFLite は `internal/tflite` (FlatBuffers を依存なしで読み、`internal/onnx` のグラフにする。WHILE の部分グラフ、SLICE、SIN、COS、CUMSUM など) でホストで評価
- 照合 (MLX の参照を fp32 で流したもの、`mrt_lm_ref.py` と `mrt_ss_ref.py`、`mcc_ref.py`):
  - SpectroStream の復号: STFT の相対誤差 1.4e-4 (先頭 7 フレームをマスクしても同じ)、音声 7e-4
  - 言語モデル (CPU の fp16): 条件の埋め込み 7.8e-7、ロジットの最大誤差 1.2e-3、最も確からしい符号は 600/600 一致。bf16 の参照とは 5% ずれたので fp32 の参照にした
  - NPU (温度 1.3 の分布の KL): lut8 で 0.0064 (最大の符号の一致 531/600)、fp16 で 0.0002 (588/600)。既定は lut8
  - MusicCoCa: 文の符号器の相対誤差 2.8e-7、写像 1.3e-6、"disco funk" の 12 の符号が一致 (`TestStyle`)、トークンも 5 つの文 (日本語は byte fallback) で一致
- 速さとメモリ (M1 16 GB): プログラムはすべて NPU。言語モデル 1 フレーム 22 ms (lut8。fp16 は 28 ms。実時間は 40 ms)、復号は 10 秒を 1.5 秒、合わせて 10 秒を 6.9 秒。MusicCoCa はホストで約 4 秒。読み込みは初回 14〜16 秒、コンパイル済みなら 0.3 秒 (コンパイルのキャッシュを既定で使う)
- メモリ: 符号を `-style` で渡すと footprint 0.41 GB。文から作ると文の符号器の重み (196M 個) をホストの評価器が float64 で持つので 1.5 GB、Go の GC の既定 (GOGC 100) だと 3.4 GB まで膨らんだ。`Style` の間だけ GC を 10% にし、評価器が使い終えた中間値を捨てるようにして 2.0 GB。models.yaml の memory は 2.0
- 使い方: `hwnpu-music -o out.wav -duration 30s "disco funk"`、`-style` (12 の符号)、`-temperature`、`-top-k`、`-cfg-style`、`-seed`、`-format` (lut8 / fp16)。models.yaml に種類と用途 `music` を足した
- 確かめていないこと: 音の良し悪しは耳では確かめていない (スペクトログラムが MLX の参照と同じように見えること、符号と分布が一致することまで)。音声からの曲調 (MusicCoCa の音楽の符号器)、音符と打楽器の条件、続きの生成とストリーミング、MusicCoCa の文の符号器を NPU に (今は 4 秒と 2 GB)、Intel の NPU と AMD

### 16.131 Magenta RT2 の続き: 曲調を NPU で、音声から曲調、音符と打楽器、続きの生成、ストリーミング — 2026-11-09

- 16.130 の「確かめていないこと」のうち、歌声 (SoulX-Singer) 以外を入れた
- **MusicCoCa の文の符号器を NPU に** (`textEncoderCut`): TFLite のグラフ (669 演算、ループなし) を `internal/onnx` の下ろしでプログラムにした。埋め込みの表引き (16000 × 768) はホスト (準備のときに fp16 の表 24 MB を書き出す)、注意の条件 (パディングに -2.38e38 を掛けて定数と比べる真偽値) は 1 - パディングとして入力に (準備のときにグラフで確かめる)。下ろしに足したもの: `Gelu`、`Rsqrt`、`Where` (条件を 0 と 1 の入力にして cond·a + (1 - cond)·b。最初の b + cond·(a - b) は、b がマスクの -3e4 のとき a - b が fp16 の刻み 16 に丸められ、Neural Engine で相対誤差 0.32 になった)、定数を fp16 の範囲に丸める (-2.38e38 が -inf になり 0 を掛けて NaN)、先頭が 1 の 3 次元の定数行列を linear に。CPU インタプリタで相対誤差 2.5e-4、"disco funk" の符号が参照と一致。ホストの float64 の重み 1.5 GB が要らなくなった
- **音声から曲調** (`Styler.Audio`): 16 kHz モノラルを 10 秒ずつ、TFLite の前処理 (ホスト。足した演算: PAD、RFFT2D (基数 2 の FFT)、COMPLEX_ABS (複素数は F に実部と虚部を並べる型 Complex64)、計算した重みの FULLY_CONNECTED) と音楽の符号器 (NPU)、平均。音楽の符号器は WHILE で 12 層を回すので、`onnx.Graph.Unroll` を足した: 条件が定数の変数 (回数) だけで決まるループを本体の写しに展開し、回数を定数のまま伝えるので、層ごとの重みの切り出しが下ろすときに定数として畳まれる。入口のパッチ分け (6 次元の reshape。Core ML の reshape は 5 次元まで) はホストで、準備のときに arange を流して添字の表 (0.5 MB) を書き出す。CPU インタプリタで埋め込みの相対誤差 5.6e-4、8 秒の曲から 12 の符号が参照と完全に一致。ホストで TFLite を評価すると 10 秒あたり 19 秒かかっていた
- **曲調の混ぜ合わせ**: 文と音声の埋め込みを重み付きで平均してから量子化 (`Mix`)。`hwnpu-music "jazz@2" song.wav`
- **SpectroStream の符号器** (`Encoder`、続きの生成のため): ホストの STFT (窓 960、ホップ 480、周期的な Hann、reverse causal) と、符号器のプログラム (41 フレーム。左右のチャンネルを重み共有のバッチ 2 で encoder_0〜5、合わせて encoder_6、bottleneck、出力の 1×1)。semicausal の時間の詰め物 (k - s 前、s - 1 後)、平均プーリング (reshape と reduce_mean)、ストリーミング用の Delay は全体を流すときは何もしない。ホストで RVQ (12 段、最も近い符号を引いて次へ)。MLX の参照 (`mrt_enc_ref.py`) と、STFT の相対誤差 9e-7、埋め込み 2e-4、4 秒の 1200 の符号のうち 1187 が一致 (第 1 段は 100 / 100)。25 フレームずつ前 16 フレームを読む窓でも同じ 1187
- **続きの生成** (`Session.Prefill`): 符号を強制して言語モデルに読ませる (C++ の実行系と同じく曲調はマスク)。12 層 × 41 の窓で届く最後の 500 フレーム (20 秒) まで。出力は元の音声のあとに続き (`-new-only` で続きだけ)。つなぎ目は前のチャンクから復号して逆 STFT の重なりをつなぐ
- **音符と打楽器** (`PianoRoll`、`notes.ReadMIDI`): MIDI の音符をフレームごとのピアノロールの状態 (マスク、オフ、継続、始まり、オン) に。C++ の実行系の populate_condition_tokens と同じく、鳴っている音だけをマスクから外す (`-note-width` で周りの音高を「オフ」に)、`-onsets` で始まりを区別。打楽器のチャンネル 10 は除く。打楽器は `-drums auto|on|off`。CFG の強さ (`-cfg-notes` 既定 5、C++ の実行系の既定) は条件の符号として渡す (Python の generate と同じ。C++ の実行系はバッチを 3 倍にする本当の CFG)
- **ストリーミング** (`Session`): 25 フレーム (1 秒) ごとに復号して音を返す。`-o -` で標準出力に WAV (大きさは未定の印) を流す (ファイルにも 1 秒ごとに書くので、途中で失敗してもそこまでの WAV が残る)。復号の先読み (1 フレーム) のため最後のフレームの音は次のフレームを待つので、`-duration` に 1 フレーム足して生成する
- 直したこと: 曲の頭の 4 STFT フレーム (先読み) を、参照と同じく逆 STFT の前に落とすようにした (前は逆 STFT のあとで落としていたので、最初の 0.2 秒が参照と相対誤差 0.23 で違った。今は全体で 7.3e-4。`TestSessionAudioMatchesReference`)
- Neural Engine について: 表引きを one-hot (128 × 16000) の入力と表の積にした版は、CPU インタプリタでは正しかったが、Neural Engine では最初の予測が約 100 秒で失敗し、その後はどのプロセスのどのプログラム (hwnpu-pitch の小さなものも) の予測も約 100 秒で失敗するようになった (M1。再起動なしで約 20 分後に戻った)。表引きはホストに戻した。新しい形のプログラムを Neural Engine で試すときは単独で試し、小さな既知のプログラムが動くことを確かめる
- **resampy の kaiser_best** (`Resample`): magenta-rt は録音を resampy で 16 kHz (と 48 kHz) にする。Hann 窓の sinc (`lfm3audio.Resample`) とは 9.4% 違い、MusicCoCa の対数メルは 8 kHz 近くの差に敏感で、埋め込みが 42% 動いて符号が変わった。kaiser_best (零点 50、1 零点 8192 値、ロールオフ 0.91735、Kaiser の beta 12.98459。npz の表に 1.8e-12 で一致) と resampy の補間をそのまま移して、resampy との差 3e-10、録音の 12 の符号が参照と完全に一致
- **曲調の混ぜ合わせの長さ**: MusicCoCa の音声の埋め込みは長さ約 28、文の埋め込み (写像の後で正規化) は 1 で、量子化は長さで符号が変わる (音声の埋め込みを 0.1 倍すると 12 のうち 4 つ違う)。単純な平均では音声だけになるので、向きを重みで平均し、長さを重みで平均した長さにした。1 つだけなら元のまま
- Neural Engine (M1 16 GB) での実測: 文の符号器 4 ms (相対誤差 1.0e-3、"disco funk" の 12 の符号が参照と一致。文から符号まで写像と量子化をホストで 0.5 秒。前は 4.2 秒)、音楽の符号器 10 秒の録音あたり 23 ms (2.6e-3)、SpectroStream の符号器 4 秒で 1.85 秒 (うち STFT 0.45 秒と量子化がホスト。第 1 段の符号 99 / 100、全体 1002 / 1200 が参照と一致。CPU インタプリタは fp32 で計算するので 1187)。続きは 8 秒の録音を符号にして言語モデルに読ませるまで 7.5 秒、10 秒の生成は 6.8 秒 (実時間の 1.5 倍) で変わらない。footprint は符号を渡して 0.41 GB、録音の曲調 0.55 GB、文の曲調 0.96 GB (写像と量子化のホストの float64)、続き 0.95 GB。models.yaml の memory を 2.0 から 1.0 に
- 音符への追従: ハ長調の音階 (0.5 秒ずつ 15 音) と最後の和音 (3 音) の MIDI を `-notes` で与え ("solo piano")、出力を Basic Pitch (`hwnpu-notes`) で書き起こすと、18 音のうち 15 が同じ音高 (17 が同じ音名) で鳴っていた。同じ文と種で `-notes` なしは 7 (10)
- 確かめていないこと: 本当の CFG (バッチ 3 倍)、音の良し悪しを耳で、44.1 kHz などの録音の続き (resampy で 48 kHz にする道は通るが参照と並べていない)、Intel の NPU と AMD

### 16.132 Neural Engine を使うプロセスは 1 つずつ (`llm.OpenDevice` のロック) — 2026-11-09

- 起きたこと (M1 16 GB): Neural Engine を使う hwnpu のコマンドを 2 つ同時に起動すると OS が固まり、その後も hwnpu-chat が途中で止まり、Ctrl+C も kill -9 も効かないプロセスが残った。16.131 の one-hot の版でも、Neural Engine が固まると、どのプロセスの予測も返らず、プロセスは割り込めない状態のまま終わらなかった (約 20 分後に戻った)
- 対策: `llm.OpenDevice` が Core ML のとき、ユーザーのキャッシュの `hwnpu/neural-engine.lock` に排他の flock を取り、プロセスが終わるまで持つ (カーネルがプロセスの終わりに外す。同じプロセスが何度開いても 1 回)。2 つ目のプロセスは「waiting for the Neural Engine: process N (hwnpu-music) is using it」と出して待ち、Ctrl+C で止められる。どうしても並べたいときは `HWNPU_NPU_LOCK=off`。全部のコマンドは `llm.OpenDevice` を通るので、これで全部に効く。Windows (flock がない) では何もしない
- 確かめたこと: hwnpu-music の 10 秒の生成の途中で hwnpu-pitch を起動すると、待ってから実行した (どちらも正しい結果)。待っている間の Ctrl+C ですぐ終わり、hwnpu-music はそのまま終わった
- 確かめていないこと: 2 つ同時で固まる原因そのもの (Neural Engine のメモリか、Core ML の予測の取り合いか)。Intel と AMD の NPU では 2 つ同時を試していない (ロックも Core ML のときだけ)
- M2 の MacBook Air (`on the NPU: false`) で `hwnpu-music -seconds 60` が、言語モデルの 1500 / 1501 フレームのあと (最後にまとめて 60 秒を復号するところ) でマシン全体を重くし、Core ML の予測の失敗で終わった。M1 ではすべてのプログラムが NPU。手がかりを得るため、NPU に載らないプログラムごとに、NPU 以外に置かれた演算の種類と数を表示するようにした (`Generator.Placement`、`Styler.Placement`)。また復号を 1 秒ごとに生成と交互にし (前は `-stream` のときだけ)、WAV の大きさも書くたびに直すので、途中で失敗してもそこまでの WAV が残る。原因の演算はまだわからない
- `-seconds` を `-duration` (Go の time.Duration: `10s`、`2m`、`1m30s`) にした
- そのため、NPU に載らないプログラムがあると hwnpu-music はその内訳を出して止まるようにした (WAV を作る前)。`-allow-off-npu` で続ける。`-cpu` (CPU インタプリタ) は別
- M2 での内訳は SpectroStream の復号のプログラムの 4 演算 (84 演算は NPU): `upsample_nearest_neighbor` 2 つ、`reshape` と `transpose` が 1 つずつ、どれも CPU。マスクをフレームから STFT のフレームに広げる upsample と、入力の (1, T, 256) を (1, 256, T, 1) にする並べ替えと見て、どちらもホストで作って入力にした (マスクは 1、2、4 倍の 3 つ、x はチャンネルを先に。出力もそのまま (2, 2, 4T, 480))。符号器も同じ形に。`ssRevision` を 2 にして変換し直す。M1 では精度 (STFT 1.4e-4、音声 7.3e-4、符号 1187 / 1200) も速さも同じ。M2 で全部 NPU に載るかは未確認

### 16.133 SpectroStream を GPU の fp32 に (Neural Engine では復号が正しくなかった) — 2026-11-09 (16.134 で NPU に戻した)

- 見つけたこと: 参照の符号を復号すると、Neural Engine の復号の音声は参照と相対誤差 0.80 (M1)、GPU の fp16 では NaN だった。CPU インタプリタ (内部は fp32) は 7e-4 で、これまでの照合はこちらだけだった (16.130 の「スペクトログラムが参照と同じように見える」は目で見ただけ)。milcpu に `Options.Trace` (演算ごとの絶対値の最大) を足して見ると、復号の残差の値が 3.7e4 から 1.2e7 まで育つ (出力は 1.4e4)。fp16 の上限 65504 を超え、ELU があるので尺度を変えて逃がすこともできない。M2 の Neural Engine で復号が約 100 秒で失敗し、マシンが重くなり日本語入力まで落ちたのも、M2 の側の分析 (ANECompilerService が CPU 97% で 93 秒、2.1 GB を書いた) と合わせて、このプログラムが Neural Engine に合っていなかったためと見ている
- 対策: SpectroStream の復号と符号化のプログラムを、入出力は fp16 のまま中を fp32 に (入力を cast、重みは fp32、出力を fp16 に cast)。fp32 は Neural Engine では計算できないので、Core ML の GPU で動かす (`GraphSource.Units` を足し、`chain.OpenOn` でプログラムごとに。`HWNPU_COREML_UNITS` は全部に効くので使えない)。`-decoder-on gpu|cpu` (既定 GPU。CPU は少し遅く CPU の時間を 2 倍使う)。読み込みのときに無音で 1 回ずつ動かし、失敗を生成の途中ではなく最初に出す。`ssRevision` は 3。言語モデルと MusicCoCa の符号器はこれまでどおり Neural Engine (照合済み)
- 精度: GPU で参照の符号の音声が相対誤差 5.0e-3 (出力の fp16 への丸め。CPU インタプリタも同じ 5.5e-3)、符号器は GPU で 1192 / 1200 の符号が参照と一致 (Neural Engine では 1002)
- 速さ (M1): 復号は 10 秒あたり GPU 4.5 秒、CPU 5.3 秒。そのままでは生成が実時間の 0.9 倍になったので、復号 (GPU) を言語モデル (NPU) と並べて動かす (1 秒ごとに前の 1 秒を復号。`Session` の符号を mutex で守る。-race で確かめた)。30 秒を 21 秒 (実時間の 1.4 倍)
- footprint (M1): 符号を渡して 0.68 GB、録音の曲調 0.83 GB、文の曲調 1.24 GB、続き 1.29 GB。models.yaml の memory を 1.3 に
- 音符への追従を直した復号で測り直した: 18 音のうち 14 が同じ音高 (17 が同じ音名)、`-notes` なしは 5 (10)
- 確かめていないこと: M2 で GPU の復号が動くか (Neural Engine に置かないので M2 で失敗した道は通らない見込み)、耳で聴くこと

### 16.134 SpectroStream の復号を NPU に戻す: ブロックごとの尺度で fp16 に収める — 2026-11-09

- 方針: このプロジェクトの目的は NPU で動かして電力を下げることなので、16.133 の GPU の fp32 はやめ、復号を Neural Engine のまま正しくする。GPU と CPU は `-decoder-on gpu|cpu` で比べるためだけに残す
- 分析 (milcpu の `Options.Trace` に RMS を足した): 復号の値はブロックごとに最大 480 (入力から decoder_0)、9e3 (decoder_1)、1.1e5、2.8e5、8.3e5、1.3e7、1.4e7 (decoder_6) と育つが、1 つのブロックの中では最大が RMS の数百倍に収まる。fp16 は浮動小数点なので、ブロックごとに 2 のべき乗で割れば収まる
- やり方 (`dgraph.scale`、`decoderScales`): ブロックの値を尺度 s で割った u で計算する。畳み込みは線形なのでバイアスを s で割るだけで正確、ELU は elu(s·u)/s = relu(u) + (exp(clip(s·u, -30, 0)) - 1)/s (Neural Engine の演算だけ)、ブロックの切り替えは 2 のべき乗の掛け算 (正確)。出力は STFT を 64 で割ったもので、ホストで掛け戻す。ブロックの和は ±60000 で飽和させ、想定より大きな音で inf と NaN にしない
- 尺度の選び方: 大きすぎると、バイアスや ELU の負の側を割った値が fp16 の非正規数 (6.1e-5 未満) になる。Neural Engine はこれを 0 にするらしく、尺度を 8192 まで上げた最初の版は Neural Engine で参照と相対誤差 0.068 だった。milcpu に `FlushSubnormals` (RoundF16 と一緒に、非正規数を 0 に) を足して再現し (3.5e-2)、尺度を 1/8 にして (decoder_0 から 1、1、8、16、64、1024、1024、出力 64。参照の曲での最大は 1.8e4 で fp16 の上限の 3.7 分の 1) 5.7e-3 に。半分にすると最大 3.5e4、さらに半分で inf
- 結果 (M1): Neural Engine の復号が参照の符号で相対誤差 6.8e-3 (16.130 からの版は 0.805 だった)。同じ種の 30 秒を GPU の fp32 と比べて 0.32%。30 秒を 19 秒 (実時間の 1.6 倍)、10 秒の復号は 2.0 秒 (GPU は 4.5 秒)。footprint 0.42 GB (符号)、0.55 GB (録音の曲調)、0.97 GB (文の曲調)、0.96 GB (続き)。models.yaml の memory を 1.0 に戻した。音符への追従は 14 / 18 (なしは 5)
- 符号器: 値の最大が 1e4 を超えないので尺度は 1 のまま (Neural Engine の符号の一致 1002 / 1200 は細かい段の fp16 の丸めで、第 1 段は 99 / 100)
- `ssRevision` は 4。読み込みのときの無音での試し実行は残す
- 確かめていないこと: M2 で NPU の復号が動くか (16.132 の失敗はこの値のあふれと関係があると見ているが未確認)、参照の曲より大きな音 (飽和で歪む)、耳で聴くこと

### 16.135 hwnpu-music の表示とスピーカーでの再生 — 2026-11-09

- 表示を hwnpu-image と同じ `internal/console` に: 経過は灰色 (`console.Log`)、待ち (曲調を読む、プログラムの読み込み、続きの録音を読む、最後の再生) はスピナー、生成は進み具合の棒 (何秒のうち何秒、実時間の何倍)。読み込みの行の「all on the NPU」は緑、`NOT all on the NPU` は赤 (`console.Highlight`)。書いたファイルの行だけ通常の色。端末でなければ色も書き直しもしない
- `-play`: できたそばから既定の出力 (スピーカー) で鳴らす。`github.com/ebitengine/oto/v3` (v3.5.1。macOS は purego で cgo なし、Linux は PulseAudio を Go で、Windows も cgo なしでビルドできる)。1 秒たまってから鳴らし始め、生成は実時間の 1.5 倍なので途切れない。oto のミキサーは別のゴルーチンで読み、何もないときは 0 バイトを返す (1 ms 後にまた読む)。最後は鳴り終わるまで待ち、Ctrl+C で止まる。`-play` だけならファイルは書かない (`-o` を付ければ両方)。`-o` も `-play` もなければ `-play` (鳴らすだけ)
- 終わりが「ぶつっ」と切れたので、最後の `-fade` (既定 2 秒) を出さずに持っておき、生成が終わったら 0 まで余弦で絞って書く (ファイルにもスピーカーにも)。`-fade 0` で前のとおり。スピーカーには最後に 0.25 秒の無音を足し、プレーヤーが読み終えてからも 0.3 秒待って閉じる (機器のバッファの分が切れないように)

### 16.136 MOSS-SoundEffect v2.0 (効果音、`hwnpu-music generate -m sfx`) — 2026-11-10

- 選んだ理由: 効果音で Apache-2.0 か MIT のものを探し直した (small-models-survey.md の 1 節)
  - MMAudio は重みが MIT だが、文の特徴に使う Apple DFN5B-CLIP が研究専用の許諾 (Apple Machine Learning Research Model License) なので外した
  - MOSS-SoundEffect v2.0 (OpenMOSS、コードと重みが Apache-2.0) を 8 GB と 16 GB の両方に入れた。2 つの違いは文の符号器の形式だけ
- 構造 (OpenMOSS/MOSS-TTS の moss_soundeffect_v2):
  - 文の符号器は Qwen3-1.7B の最後の隠れ状態 (最後の norm の後)
  - DiT は Wan 2.1 の 1.3B (30 ブロック、幅 1536、12 ヘッド、FFN 8960、交差注意は 512 行の文)
  - VAE は DAC (潜在 128 チャネル、1 秒に 50、48 kHz モノラル。decoder_rates は 8 5 4 3 2)
  - 生成は flow matching のオイラー法 (shift 5、カードの既定は 100 ステップ、CFG 4)
  - 負のプロンプトは空文字なので、文脈はすべて 0 になる。文の符号器は要らない
  - いつも 30 秒 (1500 フレーム) の潜在を消して、欲しい秒数だけを切り出す。プロンプトには「 duration: 10.0s」を付ける
- PyTorch の .pth を Go で読む (`internal/torchzip`)
  - torch.save の zip と pickle を、torch.save が書く opcode だけで読む。Python のクラスは作らない
  - 対応したもの: `_rebuild_tensor_v2`、`_rebuild_parameter`、`OrderedDict`、ストライドとオフセット
  - DAC の .pth は state_dict と metadata の dict で、weight norm は `weight_g` と `weight_v`。変換のときに g·v/|v| にまとめる
- 参照は `tools/ref/ref_moss_sfx.py`。DAC、Wan DiT、スケジューラを torch と transformers だけで書き写した。サブコマンドは vae、text、run、hidden
- DAC の復号 (5 段、すべて NPU、4 秒の音を 0.7 秒で。参照との相対誤差 0.93 %)
  - Neural Engine は 1 次元 16384 までなので、段ごとに時間を区切って回す。区切りの前後には受容野より広い「のりしろ」を付け、ホストでつなぐ
  - 畳み込みのゼロ詰めは、各畳み込みの入力にマスク (クリップの外は 0) を掛けて再現する。区切っても一続きのときと同じ値になる
  - 最初は全段が NPU に載らなかった。分かった制約と対処:
    - `sin` (Snake1d) が CPU に落ちる。`cos`、`floor`、`round`、`ceil`、`mod`、`floor_div`、`sign`、`cast` も同じ
    - 代わりに、値 v の整数への丸めを fp16 で (v + 1536) - 1536 として作った。ただし 1536 は入力 `mg` で渡す必要がある (定数だと Core ML がこの足し引きを畳んで消す)
    - その丸めで y = αx を π の倍数分戻し (π は 3.125 と残りに分ける)、sin を r/(π/2) の 9 次の級数で計算する。sin² の周期は π
    - α が 1e-2 未満のチャネル (最初の DecoderBlock は最小 8e-6) は 1/α が fp16 を超えるので、αx²(1 - y²/3 + 2y⁴/45) で計算する
    - `conv_transpose` は、ストライド 5 と 7 が CPU に落ちた。2048→1024 で k16 s8 のものは長さ 1536 で CPU に落ち、1024 なら NPU に載った (1 段目を 1024 にした)
    - ストライド 8、4、3 の `conv_transpose` は NPU に載るが、値が参照から 22 %、45 %、4 % ずれた (M1。ストライド 2 は 0.3 %)
    - ストライド 2 以外は多相に変えた (カーネル 2 の畳み込みで s·Cout チャネルを作り、reshape、transpose、reshape で時間に並べ替えて切り出す)。すべて NPU に載り、各段 0.14〜0.29 %
    - これらの値のずれは、hwnpuinfo の配置を見ても、milcpu (`HWNPU_CPU_F16=1` でないと丸めの技が再現しない) でも分からない。段ごとに参照の入力を与え、実機で照合して見つけた
- 文の符号器 (Qwen3-1.7B、`internal/llm/qwen3` を NoHead で使い、最後の norm はホスト。PromptRows 256 行、右詰めのパッドは因果注意なので前の行を変えない)
  - 最初のトークンに、層 3 から最大 1.2e4 の massive activation が出る
  - int8 では層 2 の down_proj の出力が NPU で inf になった。int8 はスケールを積和の後に掛けるので、途中の値があふれるとみられる。lut8 は先に fp16 の重みに戻すので問題ない
  - 参照との相対誤差 (RMS) は次のとおり:

    | 形式 | 誤差 | トークンごとのコサイン | 大きさ |
    |---|---|---|---|
    | fp16 | 0.66 % | 1.000 | 2.8 GB、4 部分 |
    | lut8 | 10 % | 最初のトークン 0.943 | |
    | lut8 + 層 0、1、2、26、27 を fp16 (`TextFP16`) | 4.9 % | 0.989 以上 | 1.7 GB |
    | lut8 (kmeans-minmax) | 19 % | | |
    | lut4 | 127 % | | |

  - 16 GB 以上の機械は fp16、8 GB は lut8 + `TextFP16` にした (`sfx.Formats`)
- DiT (5 ブロックずつ 6 プログラム、すべて NPU)
  - 時刻の埋め込み、時刻の射影、各ブロックの scale_shift_table はホストが計算し、入力 `mod` (ブロックあたり 6 行と、頭の 2 行) で渡す
  - RoPE は Wan では対 (2j, 2j+1) を回す。q と k の射影の行 (バイアスと RMSNorm の重みも) を並べ替え、blocks.ApplyRope の対 (j, j+64) にした。スコアは変わらない
  - 残差の最大は 1.4e4 (最後のブロックの FFN) で、fp16 の上限の 4.6 分の 1 しかないので、1/4 に縮めて持つ (`DitScale`)
  - 落とし穴: `DitScale` を型のない整数の定数にしたら `1/DitScale` が 0 になり、パッチの埋め込みと交差注意の出力が消えていた (速度が参照と 14 倍ずれた)。4.0 にして直した
  - 参照の速度との相対誤差 (1 回の呼び出しの時間) は次のとおり。lut8 にした:

    | 形式 | 誤差 | 1 回 | 大きさ |
    |---|---|---|---|
    | fp16 | 2.1 % | 1.65 s | 2.7 GB |
    | lut8 | 2.7 % | 1.28 s | 1.4 GB |
    | lut6 | 12.5 % | 1.15 s | |

- 結果 (M1 16 GB)
  - 4 ステップで参照と同じ雑音から作った音の波形のコサイン: 0.992 (16 GB の構成)、0.990 (8 GB の構成)
  - 10 秒の音は 100 ステップで 4 分 16 秒 (1 ステップは 2 回の呼び出しで約 2.6 秒)、20 ステップで 52 秒
  - 起動 (コンパイル済み): 文の符号器 2.6 秒、DiT と復号 2.7 秒
  - 初回は変換が数秒、コンパイルが約 5 分
  - 2 回目は一部をコンパイルし直した (26 秒と 1 分 13 秒)。3 回目からは 4 回続けて約 5 秒
  - footprint の最大: 2.34 GB (文の符号器が fp16)、2.17 GB (lut8)
  - 文の符号器のプログラムを先に動かして手放してから、DiT を読み込む
- 確かめていないこと:
  - 耳で聴くこと
  - 8 GB の実機
  - M2 での DAC の多相の変換
  - ステップ数を減らしたときの質 (`-steps`)
  - ftfy の fix_text は省いた (平文なら変わらない)
  - Intel と AMD

### 16.137 hwnpu-music を hwnpu-image に合わせる: サブコマンドと kong、効果音は同じ generate でモデル違い、8 GB 向けはタグで — 2026-11-10

- 動機: 効果音は別のコマンド `hwnpu-sfx` にしていたが、音楽とはモデルが違うだけで、出力 (WAV、スピーカー) も表示も同じ。8 GB 向けの形式も環境変数 (`HWNPU_MEMORY`) ではなく、モデルのセレクターで選びたい
- `hwnpu-music generate [STYLE|DESCRIPTION...]` と `hwnpu-music list [music|sfx]` に分けた。引数は hwnpu-image (16.123) と同じ kong で、フラグは `--model` (`-m`)、`--out` (`-o`)、`--duration` など 2 本の `-`。ヘルプは共通、Music、Sound effects のグループに分けた。`hwnpu-sfx` は消した
- `-m sfx` (用途、別名 sound-effects) か `-m moss-soundeffect` (ファミリー) で効果音になる (`chat.ResolveForKinds` で music と sfx の両方を受け、種類で分ける)。音楽だけのフラグ (`--codes`、`--continue`、`--notes`) を効果音に付けるとエラー
- 8 GB と 16 GB は models.yaml の別のモデルにした: `MOSS-SoundEffect-v2.0` (タグ v2.0、machine 16、文の符号器 fp16) と `MOSS-SoundEffect-v2.0-lut8` (タグ v2.0-lut8、machine 8、`like` で同じリポジトリ)。形式は選ばれたモデルの machine から決める (`sfx.Setup.Machine`、`sfx.Formats`)。`moss-soundeffect:8g`、`:8GB`、`:8gb` は 8 GB のマシンの選び方 (今までのメモリのタグ)、タグなしはこのマシンのメモリ。`TestPickSoundEffect`
- 表示を hwnpu-image にそろえた: 経過は灰色 (`console.Log`)、配置は「all on the NPU」が緑、`NOT all on the NPU` が赤 (`console.Highlight`)、書いたファイルの行だけ普通の色。seed は省くと `rand.Uint64` で引き、灰色で出す (#93 と同じ)
- 効果音の WAV は音楽と同じ 48 kHz ステレオ (モノラルを両方のチャンネルに)、利得 1 (音楽の WAV は参照に合わせて 0.5)。終わりの `--fade` は、音楽は既定 2 秒、効果音は付けたときだけ
- 配置が分からないのは NPU から外れたのとは別に扱う: 8 GB 向けの文の符号器が、コンパイルし直した直後に `NOT all on the NPU` で止まり、次の起動では全部 NPU だった。hwnpuinfo でも全部 NPU。Neural Engine がコンパイルで忙しい間に MLComputePlan が 2 分で終わらず (`placement.Verified` が false)、外れたことにされていたとみる。`runner` に `Placement()` (分からないときは理由、外れたときは演算の内訳) を足し、分からないプログラムは「placement not known, going on」と出して続ける。音楽の `checkNPU` も同じ

### 16.138 Sana の Gemma 2: 指示文のキーと値を書き出しに持ち、続きを 64 行の塊で。射影は q6 — 2026-11-10

- 動機: Sana のテキスト (Gemma 2 2B) は 1 枚 2.6 秒 (読み込み 1.0、計算 1.6)。毎回 512 行を計算していたが、そのうち指示文 (208 トークン) は毎回同じで、因果的な注意なので結果も同じ
- 指示文を書き出しのときに numpy の float32 で通し (`iron_sana.Prefix`)、各層のキーと値の先頭 192 行をグラフの定数に、最後の隠れ状態を `prefix_out.bin` に持つ。グラフは 64 行の塊で、呼び出しごとの値は `start` だけ (キャッシュへの書き込み、RoPE の表、注意のキーの数がどれも `start` の一次式)。Go は指示文と一致する長さから `start` を決め (`sana.textStart`)、長いプロンプトは塊を続けて呼ぶ。否定のプロンプトは `start` 0 から同じグラフで
- 呼び出しごとの行の数 (`x[:rows]`) で縛る形は、RMSNorm の回数 `ceil(rows / 8)` が一次式にならず書き出せなかった。IRON の lm と同じく固定の塊にした。MHA は 1 パイプライン 64 行なので、64 行の塊ではパイプライン 1 本
- 精度: 残す行は CPU の fp32 (指示文とプロンプトを通して計算したもの) とコサイン 0.99983 以上 (q8)。前の形と同じかわずかによい (相対誤差 1.51 → 1.35 %)
- 形式: 64 行では時間は形式でほとんど変わらない (CLI で q8 0.72、bf16 0.68、q6 0.70 秒)。後段の DiT につないで、CPU の参照 6 枚と同じノイズで比べ、q6 を既定にした (行の誤差 3.5 %、構図は参照と同じ。読み込み 1.0 → 0.76 秒、固定 2.3 → 1.7 GB)。q4 の丸めは行の誤差 15 % で構図が変わる
- 評価の教訓: 2 ステップの Sana-Sprint は条件の小さな差を大きく広げる (キャプションの 0.23 % の差で潜在のコサイン 0.965)。1 枚の潜在のコサインでは形を比べられない。行の誤差 (CPU の fp32 に対する) と、数枚の絵の構図で決める
- 結果: テキスト 2.6 → 1.9 秒 (計算 1.58 → 0.70 秒)。否定のプロンプトつきの Sana 600M はテキスト 5.7 → 3.3 秒。固定の最大は VAE の段 (2.4 GiB) で変わらず、テキストの段は 2.34 → 1.87 GiB。1 枚の時間の残りの大半は VAE (13 秒)
- 残り: テキストのグラフを載せたままにする使い方 (サーバー) では、暖まった計算が bf16 で 257 ms (q6 494 ms) なので、形式を選び直す余地がある。否定のプロンプトがあるときにグラフを 2 回読み込む (1 回にまとめられる)

### 16.139 Sana の VAE: 3 × 3 の畳み込みをタップごとの行列積に、同じ形の段をまとめる — 2026-11-10

- 動機: Sana-Sprint 0.6B の 1 枚 20 秒のうち VAE が 13 秒。1024² × 128 チャネルの ResBlock の段 (1.15 秒) を演算ごとのコマンドに分けて測ると、im2col の写しが 63 % (6.8 GB/s。連続する長さ 768 バイトの細かい DMA)
- タップごとの行列積: 格子 (チャネルが最後、行に Wp 画素) では各タップの入力が連続した行の範囲なので、写さずに行列積 (K = C) を 9 回して足す。段 1.15 → 0.76 秒。足し算 8 回 (塊あたり 7 ms) が残る: IRON の行列積の出口処理は活性化だけ。縦の 3 タップずつ写す形も概算したが、16 行あたり約 6 ms で同じ。偶奇で重みを変える拡大は im2col のまま (`--vae-conv im2col` で前の形)
- 同じ形の段をまとめる (13 → 10 段): パイプラインでは次の段の読み込みと重なって実行が 0.76 → 1.1 秒になるので、読み込みの回数を減らす。格子の状態を使い回し (`free_grid`)、2 つ目以降の ResBlock は入力の格子に上書き (しないと固定の最大が格子 1 枚 260 MiB 分増えた)
- 前の段を閉じる処理 (50〜136 ms) を次の段の実行と重ねた (`sana.VAE.Decode`)。単独では差がほとんど見えなかった
- 拡大を低い解像度の 4 つの位相で (`--vae-up phase`): 最近傍の 2 倍の後の 3 × 3 を、出力の位相 (2y + py, 2x + px) ごとの低い解像度の 2 × 2 の畳み込み (核の行と列を位相ごとに足した重み) にし、出力の格子の 2 行おき 2 画素おきに写す。積は 4/9、拡大した格子 (1024² × 256 で 545 MB) が要らず、固定の最大が 2409 → 1960 MiB。偶奇で重みを変える拡大も位相の py ごとの重みで同じ形
- 格子を行列積が直接読む (`tools/amd/iron_gridgemm.py` の `GridGEMM`): AIE のカーネルを書かず、IRON の `flm.ProjectionGEMM` を継承して A の DMA の並びだけを差し替えた。im2col の 1 行は縦のタップごとに格子の連続した 3C 要素なので、論理上の行の刻み C、K のタイル (3C) の刻み Wp·C の並びで読める (密な A と次元の数が同じ)。`sequence` の間だけ `TensorAccessPattern.full` を A の 7 次元の形のときに差し替える。C ≤ 170 は 9 タップで 1 回、大きい C は縦のタップごとに 3 回。拡大の位相 (2 × 2) も同じ。1024² の ResBlock 3 つ 2.7 → 1.4 秒。参照と許容誤差は、論理上の A を組み立てて計算する
- VAE の線形注意を MEDIUM (bfp16) に: 指数を共有するのは K の方向の 8 要素だけで、定数の行 (c = 4096) は値の行と共有しない。絵は HIGHEST と 51.6 dB
- 結果: VAE 12.3 → 6.3 秒、Sana-Sprint 0.6B の 1 枚 20 → 12 秒 (16.138 のテキストを含む)、Sana 600M 67〜70 → 59 秒。画素は前の VAE と 50〜54 dB で、CPU の参照 8 枚との PSNR は同じ。固定の最大 2409 → 1936 MiB (プロセスを足して 2.27 GB。models.yaml の memory 2.9 → 2.3)
- 残り: EfficientViT の線形注意がヘッドの数の倍の計算をしている (ブロック対角を密な積とマスクで)。深さ方向の畳み込みは要素ごとの 49 演算 (チャネルごとの係数で掛けて足す演算が IRON にない)
- 否定のプロンプトつきでテキストのグラフを 1 回だけ読む (`sana.EncodeBoth`): Sana 600M のテキスト 3.3 → 2.5 秒、潜在はビット単位で同じ


### 16.140 コマンドを用途の家族にまとめる: hwnpu-audio、hwnpu-chat systemone、hwnpu-image pose — 2026-11-10

- hwnpu-music を hwnpu-audio に改名し、hwnpu-tts、hwnpu-transcribe、hwnpu-pitch、hwnpu-notes をそのサブコマンド (`tts`、`transcribe`、`pitch`、`notes`) にした。音楽と効果音は 16.137 のまま `generate` (`-m sfx`)。`list` は 6 種類 (music、sfx、speech、transcribe、pitch、notes)
- hwnpu-systemone は `hwnpu-chat systemone`、hwnpu-pose は `hwnpu-image pose` に。古い名前は残していない (ラッパーなし)
- フラグは kong の流儀に: `-use` → `-m/--model`、`-list` → 各家族の `list`、`-o` はそのまま、ほかは `--name`。手で変換したプログラムを渡すフラグは "Converted by hand" の組にまとめ、tts と systemone のチェックポイントの `-model DIR` は `--checkpoint DIR` に (`-m` とぶつかるため)。systemone の `-audio` は hwnpu-chat に合わせて `--audio-encoder`
- pose の `-draw FILE` は face-detect に合わせて `--draw` (NAME.pose.png、`-o DIR` で置き場所)。notes の `--midi FILE` はそのまま (1 入力)
- ログは灰色 (console.Log) で、読み込みの行に "all on the NPU" / "NOT all on the NPU" / "on the CPU interpreter"
- 確かめたこと (AMD の機械、--cpu): `hwnpu-audio pitch` が 440 Hz の正弦波で 439.5 Hz、`hwnpu-audio notes --midi` が A4 (69) の 1 音、`hwnpu-image pose --draw` が動く。tts、transcribe、systemone は Neural Engine 用で、ここではヘルプと引数の検査だけ
- 開発用で家族に入れていないもの: hwnpu-check (CPU のインタプリタで Hugging Face の参照と層ごとに照合)、hwnpu-formats (チャットモデルの重みの形式を NPU で比べる)、hwnpu-convert、hwnpu-generate、hwnpu-xclbin、hwnpuinfo、npuprobe、powerbench、report

### 16.141 開発用の道具を hwnpu-dev のサブコマンドに — 2026-11-10

- hwnpu-check、hwnpu-convert、hwnpu-generate、hwnpu-formats、hwnpuinfo、npuprobe、powerbench、report、hwnpu-xclbin を `hwnpu-dev check|convert|generate|formats|info|probe|powerbench|report|xclbin` にした。各道具は `cmd/hwnpu-dev/NAME` のパッケージ (`main` を `Main` に) で、フラグは前のまま (std flag)。hwnpu-dev は kong の `passthrough:"all"` で、名前の後ろの引数 (`-h` も) をそのまま渡し、`os.Args` と `flag.CommandLine` を差し替えて `Main` を呼ぶ
- npuprobe は自分を子プロセスで起動するので、`probe.ChildArgs` (`["probe"]`) を前に付ける
- aiecc は `AIE_XCLBINUTIL` の実行ファイルをサブコマンドなしで呼ぶので、実行ファイルの名前が hwnpu-xclbin か xclbinutil なら xclbin として動く (`go build -o ~/.local/bin/hwnpu-xclbin ./cmd/hwnpu-dev`)
- 文書の名前は書き換えた。この design.md、.knowledge、npu_profiles、docs/handoff-* は当時の記録なので古い名前のまま

### 16.142 `hwnpu-dev probe cache`: Neural Engine のコンパイル済みキャッシュを測る — 2026-11-10

- 動機: Sana の Mac (#99) で、hwnpu のキャッシュが当たっていても読み込みに 1〜2 分 (ANECompilerService)。システム側のキャッシュの容量、閉じ方、順、プロセス、実行ファイル名との関係を M1〜M4 で測る ([handoff-ane-cache.md](handoff-ane-cache.md))
- 構成 = S GiB の合成プログラム N 本 (`internal/probe`、構成ごとに別の種)。一度だけ書き (バイトが変わると別物になるため)、回ごとに新しい子プロセスが順に読んで 1 回実行する。hwnpu のコンパイルのキャッシュ (`WithCacheDir`) は構成のディレクトリに残す。読み込みと最初の呼び出しが `-cache-hit` (1 秒) 以下で当たり
- フラグ: `-cache-sizes` (0.5,1)、`-cache-totals` (1〜8)、`-cache-format` (fp16)、`-cache-rounds` (3)、`-cache-hold close,open`、`-cache-order same,reverse` (1 回おきに逆順)、`-cache-passes` (同じプロセスで読み直す)、`-cache-names` (実行ファイルのコピーの名前で 1 回、その後に元の名前で 1 回)、`-cache-keep` (`-dir` に残して再起動の後に使い回す)、`-cache-pause`
- 報告: 構成ごとの各回の当たりの数と warm (2 回目以降すべて当たり。別名の回は入れない)、大きさ・載せ方・順ごとの warm の最大の合計と外れた最小の合計、各読み込みの秒数、各回の後の `~/Library/Caches/<名前>/com.apple.e5rt.e5bundlecache` と共有の方の大きさ
- 親は `llm.LockNPU` (新しく公開) で Neural Engine のロックを持ち、ほかの hwnpu のプロセスを待たせる。子はロックなしで開く。子が自分を起動し直すときの実行ファイルを `prober.exe` で差し替えられるようにした
- 未実行: Linux (AMD) の機械で書いたので、Mac での計測はまだ。ビルド (darwin/arm64) と、当たりの数え方と報告の単体テストだけ

### 16.143 Intel の NPU の実機での一通りの確認と、Anima の部品が NPU のメモリに載りきらないとき — 2026-11-10

- 機械: HP EliteBook 630 G11、Core Ultra 5 135U (Meteor Lake、NPU 3720)、32 GB、Windows 11 25H2、NPU ドライバ 32.0.100.5540 (Level Zero 1.16、ZE_extension_graph 1.18、コンパイラ 8.3)。`tools/intel/hwnpu-intel-check.ps1` でまとめて動かした
- 通ったもの (全部 NPU、コンパイルはキャッシュから): report の 2 つ、LFM3-1.7B (`--paged=-1 --context 4096` と既定の 262144 の 7 部分の両方、17〜25 トークン/秒、下書きつき)、考える答え、`hwnpu-chat agent` の read と write (answer.txt に 7351)、PLaMo 3 Translate 0.8B (ja → en と en → ja、25〜31 トークン/秒)
- hwnpu-formats (LFM3-1.7B、キャッシュ 4096、1 回 8 トークン): fp16 112.1 ms、int8 68.5 ms (次のトークンが fp16 と 99 %、パープレキシティ 4.57 / 4.56)、int4-g128 84.4 ms (76 %、6.13)、int4 37.9 ms (63 %、10.32)。Meteor Lake ではグループごとの int4 は int8 より遅く、行ごとの int4 は速いが精度が足りない。int8 の既定のままでよい
- 失敗したのはネットワーク (社内のプロキシ) だけ: PLaMo-Embedding のダウンロードの切断、Gemma 4 と Anima の名前解決
- Anima (512 × 512、text と DiT が fp16) は `dit_blocks_24.xml` で `pfnGraphInitialize: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY`。その時点で載っていたのは text (約 1.2 GB) と DiT の 6 群 (各約 0.57 GB)、合わせて約 4.6 GB。グラフの重みはドライバが NPU から見えるメモリに置くので、その上限に当たったとみる (上限そのものは未測定。`hwnpu-dev probe` の limit の several で測れる)
- 対処 (`anima.progEngine`): 部品を `slot` にし、読み込みがメモリ不足 (`types.IsOutOfMemory`: Level Zero の OUT_OF_HOST_MEMORY と OUT_OF_DEVICE_MEMORY) で失敗したら、ほかの部品を閉じて読み直す。閉じる順は、プロンプトごとに 1 回しか使わない text、次に DiT の群で最後に使ったもの (MRU)、最後に時刻・パッチ・最後の層。DiT の群は毎ステップ同じ順に回るので、MRU なら 7 群に 4 か所でもステップあたり 4 群の読み直しで済む (LRU では 7 群全部)。この機械の fp16 は text を閉じるだけで収まり、読み直しは 1 枚に text と 1 群の 2 回の見込み。VAE の段も同じ規則で場所を作る
- 偽の NPU (`residency_test.go`) で確かめた: 全部載る、text を閉じれば載る、群が載りきらない、何も閉じられない (エラーに「no other program loaded」)。実機では未確認。int8 の書き出し (`--dit-format int8 --text-format int8`) は約半分なので、確認スクリプトに `image-512-int8` を足して fp16 と比べる

### 16.144 `hwnpu-dev probe resident`: NPU が一度に持てる量 (Intel の OUT_OF_HOST_MEMORY) — 2026-11-10

- 動機: Core Ultra 5 135U で Anima の fp16 が約 4.6 GB を持ったところで `ERROR_OUT_OF_HOST_MEMORY` (16.143)。読み込みの時間の問題 (Neural Engine のキャッシュ、16.142) とは別で、同時に持てる量の上限。16.143 の読み直し (閉じて読み直す) は、閉じればメモリが戻ることを当てにしている
- 0.25 GiB (fp16 の 2 層) のプログラムを 1 本ずつ読んで開いたまま 1 回ずつ実行し、読み込みが失敗するか `-resident-max-gb` (RAM の 60 %、最大 16) まで続ける。最後に全部をもう一度実行する。メモリ不足 (`types.IsOutOfMemory`) で止まったら、一番古いものを閉じて失敗したものを読み直す (戻るか)。`-resident-hold close` は 1 本ずつ閉じて同じ合計まで (持っている量の上限か、読んだ量の上限か)。`-resident-formats int8` で、上限が重みのバイト数で決まるか
- Windows ではプロセスの private メモリ (K32GetProcessMemoryInfo) を各読み込みの後に記録する。`maxRSS` も Windows で peak working set を返すようにした
- 子プロセスの流れ (`residentRun`) は読み込みを差し替えられる形にし、偽の NPU (容量で OUT_OF_HOST_MEMORY、閉じても戻らないもの) でテストした。close の回で合計の上限で止まらない不具合をこのテストで直した
- Intel の確認スクリプトに `resident` (standard、fp16、開いたままと閉じる、8 GiB まで) と `resident-int8` (full) を足し、probe の試験は stdout からもハイライトを拾う
- 実機の 1 回目: 開いたままの回の途中で、子プロセスが 0xc0000005 (アクセス違反。ドライバの中で落ちたらしい) で終わった。子が結果を最後にまとめて送る作りだったので、何本・何 GB まで載ったかが残らなかった
  - 子は 1 本ごとに `npuprobe-progress` の行で結果を出し、親は子が落ちてもそこまでの分を報告に残す (crashed)。各段階 (読み込み、実行、閉じる) を stderr に書き、落ちた段階がエラーの最後の行に出る
  - `-resident-run=false` は実行せずに読むだけ。スクリプトの `resident-load-only` (standard) で、実行で落ちるのか読み込みで落ちるのかを分ける
  - メモリ不足がエラーでなくプロセスの異常終了になるなら、16.143 の Anima の読み直し (エラーを受けて閉じる) は効かない。そのときは、落ちる前の合計より手前で自分から閉じる (上限を決めておく) 形が要る

### 16.145 Apple: Sana と Sana-Sprint、深度推定 (DA3、MoGe-2)、顔検出 (YuNet) を Neural Engine で (Core ML) — 2026-11-10

- AMD の 16.119 と同じ部品を MIL で書き、Core ML のパッケージにする (`hwnpu-image generate -m Sana-Sprint-0.6B`、`hwnpu-image convert REPO`)。Gemma 2 2B を 6 層ずつ 5 本 (int8、1 本 約 470 MB)、caption、DiT 4 ブロックずつ 7 本 (fp16)、final、VAE 13 段。時刻の埋め込みだけホスト (下)。初回に `sana.Prepare` がチェックポイントを `--dir/Sana` に落として変換し (10 秒)、Gemma 2 (`gemma-2-2b-it/`) と その書き出しは全 Sana で共有する (別の Sana の書き出しからハードリンク)
- パイプライン (`internal/imagegen/sana`) はエンジンを 2 つに: IRON のグラフ (`ironEngine`、前の処理を移しただけ。bf16 → float32 → bf16 は可逆なので値は同じ) と Core ML (`progEngine`)。Anima の MIL の共通部分は `internal/imagegen/milpart` に移した (Anima の DiT の群、final、時刻、VAE の段の CPU インタプリタの出力が前とビット単位で同じ)
- fp16 の幅: diffusers の fp32 (M1 の CPU、1024 × 1024、`a cyberpunk cat ...`) で中間の最大を測った。残差は Gemma 2 で 4059、DiT で 1534、VAE で 1662 で、どれも尺度をかけずに収まる。あふれるのは線形注意の和: DiT の VᵀK は 2.1e4、N は 1.6e5、VAE (EfficientViT) は 2.4e5 と 4.5e7。分母の最小は 0.03。各ヘッドの q の行をその最大で割り (N / D は変わらない)、鍵を DiT で 1/64、VAE で 1/256 にしてから和を取る。分母は 1e-4 で下を切る (q がすべて 0 の行は N も 0)
- M1 の Neural Engine で見つけたこと:
  - 深さ方向の畳み込みは 4096 チャネルまで Neural Engine、5760 チャネル (GLUMBConv の 2 × 2880) は CPU。値とゲートの半分ずつの 2 つにした
  - グループつきの int8 (`int8-g32`) は linear が CPU になる。Gemma 2 は行ごとの int8
  - 時刻の埋め込み (正弦波 → 2 層 → 6 つの変調) を Neural Engine で走らせると出力が最大 1 % ずれ (CPU インタプリタの fp16 丸めでは 0.05 %)、全ブロックの変調に効いて DiT 1 回の出力のコサインが 0.9998 → 0.983 になった。1 行の計算 (1 ステップ 1100 万積和、数 ms) なので、重みを `time.safetensors` (float32) に書き出してホストで計算する
  - Gemma 2 の GELU (tanh 近似) を Neural Engine の `gelu` で書くと、Sana が残す行が CPU より悪い (caption のコサイン 0.99962、CPU の fp32 は 0.99974)。silu と同じく `x / (1 + exp(-1.5958 (x + 0.044715 x³)))` と書くと 0.99973。fp16 の重みの Gemma 2 (3 層ずつ 9 本) は Neural Engine で 0.99925 で int8 より悪い
  - 1 つの DiT の群だけを Neural Engine と CPU (fp32) で比べるとコサイン 0.999999 以上。差は時刻と Gemma 2 から来ていた
- `MLComputePlan` の待ちを 2 分 → 10 分 (`internal/backend/coreml`)。ほかのプロセスが Neural Engine でコンパイルしている間、Gemma 2 の 1 本 (単独で 13 秒) が 2 分を超えて読み込みが失敗した
- 結果 (M1 16 GB、Sana-Sprint 0.6B、1024 × 1024、2 ステップ、diffusers の fp32 と同じノイズ): caption のコサイン 0.9997、DiT 1 ステップ目 0.991 (参照の caption を与えると 0.9998)、潜在 0.9855、画像の PSNR 26.0 dB (AMD は 0.997、31.2 dB)。絵は構図も細部も同じ。手直しの前は潜在 0.972、22.7 dB
- 時間 (キャッシュあり、ほかのセッションが同時に Neural Engine を使っている状態): 全体 3 分 9 秒、プロセスの最大 3.3 GB。計算は Gemma 2 約 5 秒、DiT 1 ステップ 0.4〜0.5 秒、VAE 6 秒で、残りはプログラムの読み込み (DiT 約 57 秒、VAE の段 0、1 で約 90 秒)。何もない状態から (変換とコンパイル) 8 分 29 秒、2 回目 3 分 53 秒、同じシードの PNG はビット単位で同じ
- 4 つの Sana の確かめ (diffusers の fp32 と同じノイズから。潜在のコサイン、画像の PSNR、括弧は AMD の 14.3):

| | ステップ | 潜在 | PSNR |
|---|---|---|---|
| Sana-Sprint 0.6B | 2 | 0.986 (0.997) | 26.0 dB (31.2) |
| Sana-Sprint 1.6B | 2 | 0.990 (0.983) | 27.0 dB (25.2) |
| Sana 600M | 20 (CFG 4.5) | 0.998 (0.963) | 32.7 dB (20.0) |
| SANA 1.5 1.6B | 20 (CFG 4.5) | 0.984 (0.984) | 22.6 dB (21.4) |

  - Sana-Sprint 1.6B: GLUMBConv の深さ方向の畳み込みが 5600 チャネルで CPU に置かれた。4096 チャネルずつに分ける (`maxDepthwise`)
  - Sana 600M: scheduler_config.json に Python の `-Infinity` があり Go の json が読めない。読む前に 0 に置き換える (読む項目ではない)
  - SANA 1.5 (q と k の正規化 + CFG): 条件なし (caption の先頭の行だけ) が 1 ステップ目でコサイン 0.970、20 ステップで潜在 0.62 に崩れた。CPU インタプリタは fp32 でも fp16 の丸めでも 1.00000 で、Neural Engine の計算の問題。caption だけを CPU にすると直り、caption の先頭の行 (Gemma の `<bos>`) が Neural Engine で NaN になっていた。fp32 で caption の 2 つ目の線形の出力がその行で 3.8 万 (fp16 の上限 6.5 万の手前、ほかの Sana も 1.8〜2.8 万)。直後の RMS 正規化は尺度によらないので、線形の重みとバイアスを 1/64 に。条件なしが 1.00000、潜在 0.984。注意の softmax を最大を引いて書き直すことも試したが変わらなかった (戻した)。書き出しは v4
  - `MLComputePlan` がときどき "Failed to construct compute plan, internal failure" を返し、次の試み (や次のプロセス) では作れる。1 回だけやり直す
- 深度推定と顔検出も Apple の Neural Engine で (名前で、初回に取得と変換):
  - Depth Anything 3 (`internal/vision/depth/mil`): ViT (DINOv2。any-view のカメラトークン、2 次元の RoPE、q と k の正規化、cat_token) と DPT / DualDPT の頭。転置畳み込み (核 = ストライド) は 1 x 1 と `pixel_shuffle` (milcpu に足した)、双線形の拡大は補間の行列の積。大きさ (iron_da3.py の 9 つ) は初めて使うときに変換 (数秒)。CPU インタプリタの fp32 で公式のモジュールと深度のコサイン 1.000000 (DA3MONO-LARGE は空の画素を除いて。空は DA3 と同じく遠い値で埋める)。Neural Engine で DA3-SMALL 1.000000 (ViT 0.07〜0.19 秒、頭 0.02 秒)、DA3MONO-LARGE 0.999993、空 0.9998 (ViT 0.43 秒、頭 0.06 秒)。DA3METRIC-LARGE と DA3-BASE は同じコードで走らせていない。複数視点 (`--views`) は AMD のみ
  - MoGe-2 (`internal/vision/moge/mil`): ViT は depth/mil のもの (重みの接頭辞を Config に)、層の投影の和とクラストークンの行に尺度の頭の logit。首と 3 つの頭は 1 つのプログラム (端を複製する詰めは `pad` の replicate と詰めなしの畳み込み。Core ML は `pad_type` valid でも `pad` を求める)。model.pt は `internal/torchzip`、model_config の整数の型は問わない。ViT-S で公式と頭の出力のコサイン 0.99993 以上、尺度の logit 1.6885 (公式 1.6853)、1 枚 0.5 秒。ViT-B と L は走らせていない
  - YuNet (`internal/vision/face`): ONNX のグラフを丸ごと `internal/onnx` の Program に (12 の出力を 1 つに)。640 × 480 で 1 枚 3.4 ms、OpenCV の FaceDetectorYN と箱と目印が 0.3 画素以内。`internal/onnx`: 数字で始まる初期値の名前 (YuNet の 420 など) を Core ML が受けないので `w_` を前に。Apple の既定は Neural Engine (models.yaml の device は AMD だけ cpu)
- Gemma 2 の重みの形式 (3 つのプロンプト、画像まで): int8 0.985〜0.998 / 23〜31 dB、lut8 0.963〜0.992、lut6 (一様) 0.915〜0.966 / 18〜22 dB (構図は同じで細部と文字が変わる)、lut4 0.80〜0.83 (別の絵)。k-means の表は lut6 で悪化 (0.86〜0.92)、lut4 でわずかに良い (0.83〜0.91)。温まっていれば Gemma 2 はどの形式も約 2 秒で、速さは変わらない。既定は int8 のまま (`ConvertOptions.TextQuant` で量子化器を選べる)
- hwnpu-image は Neural Engine のロック (16.132、`llm.OpenDevice`) を取る (Sana、depth、face-detect)。配置は厳密に (確かめられないときに理由を出す)
- 複数の絵を 1 つのプロセスで (`hwnpu-image generate` にプロンプトを複数、`--keep-dit` で絵の間も DiT を載せたまま。Gemma 2 と VAE は毎回読む) と DiT の int8 (M1、1024 × 1024、3 つのプロンプトを 1 プロセスで、別のプロセスで 2 回ずつ。1 枚の時間):

| | 合計 | --keep-dit (1 回目: 1, 2, 3 枚目 / 2 回目) | --no-keep-dit (1 回目 / 2 回目) | 潜在のコサイン、PSNR |
|---|---|---|---|---|
| Sprint 0.6B fp16 | 約 4 GB | 4m17s (コンパイル), 7.2, 7.0 s / 8.3, 6.8, 6.6 s | 8.1, 7.9, 8.1 s / 7.8, 8.0, 8.0 s | 0.985〜0.998、23〜31 dB |
| Sprint 0.6B int8 | 約 3.4 GB | (コンパイル), 6.8, 6.4 s / 7.4, 6.3, 6.3 s | 7.3, 7.3, 7.2 s / 7.2, 7.3, 7.2 s | 0.980〜0.997、22〜30 dB |
| Sprint 1.6B fp16 | 約 6 GB | 12m12s, 35 s, 9.2 s / 4m27s, 26 s, 9.2 s | 6m21s, 3m13s, 1m07s / 15, 16, 16 s | 0.990、26.9 dB |
| Sprint 1.6B int8 | 約 4.4 GB | 11m41s, 2m09s, 7.8 s / 5m01s, 1m46s, 7.4 s | 3m10s, 5m09s, 4m21s / 4m21s, 4m38s, 11 s | 0.987、25.5 dB |

  - 0.6B は、16 GB 溜まっていた古いコンパイル結果を消した後は、新しいプロセスでも 1 枚 7〜8 秒 (Gemma 2 約 2 秒、2 ステップ 1〜2.5 秒、VAE 約 3.5 秒)。前の 3〜4 分は、作り直しを重ねた古い版で Neural Engine のキャッシュが埋まっていたためと見ている。`--keep-dit` で 1 枚約 1 秒縮む
  - 1.6B は fp16 でも int8 でも外れが続く: 再コンパイルの場所が回ごとに変わる (LRU で 4 GB 前後を超える、16.41 と合う)。`--keep-dit` なら 3 枚目から 7〜9 秒。int8 の DiT (合計 4.4 GB) でも収まらず、速さはほぼ変わらない。画質は少し下がる (0.6B で PSNR −0.8〜−1.6 dB)。既定は fp16 のまま
  - 容量の計測は [handoff-ane-cache.md](handoff-ane-cache.md) で引き継いだ。1.6B は合計を 4 GB 未満に (Gemma 2 の lut6 と DiT の int8 で約 3.9 GB)、または常駐で 2 枚目以降を読まない形が要る
- 残り: 1.6B の読み込み (上)、Intel (OpenVINO IR) の Sana と深度、複数視点の DA3、変換し直すとパッケージのバイトが変わってコンパイルのキャッシュが効かないこと (`internal/mlpackage` の UUID、全モデル共通)

### 16.146 Anima の 1024 × 1024 を Intel の NPU で: 1 プログラムを DiT の 1 ブロックに (DEVICE_LOST) — 2026-11-10

- Core Ultra 5 135U で `hwnpu-image generate` (1024 × 1024、fp16、1 プログラムに 4 ブロック) を動かすと、読み込みでは 16.143 の読み直しが初めて働き (dit_blocks_24 のために text と dit_blocks_20 を閉じた)、全部が載った。ところが最初のステップの最初の群で `zeFenceHostSynchronize: ZE_RESULT_ERROR_DEVICE_LOST`
- 512 × 512 の 4 ブロックは 1 群 0.4 秒ほど (1 ステップ 2.8 秒 / 7 群)。1024 × 1024 はパッチが 4 倍 (4096)、注意は 16 倍なので、1 群で 2〜4 秒になる見込み。Windows は時間の上限 (TDR、既定 2 秒) を超えたジョブを止めてデバイスを失わせるので、それに当たったとみる (メモリがぎりぎりだったことが効いた可能性も残る)
- 対処: `--group` の既定を 0 (自動) にし、`anima.DefaultGroup` で決める。Core ML は 4 のまま。Intel は 4 から半分ずつにして「ブロック数 × パッチ数」を 4096 (512 × 512 の 4 ブロック) 以下に: 512 で 4、640 で 2、768 と 1024 で 1。書き出しの名前に g が入るので、前の g4 の書き出しは使われず、作り直す。DEVICE_LOST のときは、群が 2 以上なら `--group` を半分にする案内をエラーに付ける (`types.IsDeviceLost`)
- 確かめたこと (AMD の機械、OpenVINO 2026.4 の CPU): 512 × 512 int8 の g4 と g1 の書き出しで同じ種の絵を描き、見た目は同じ、潜在の差は相対 3.6 % (群の間の受け渡しの丸め)。1024 × 1024 の g1 の書き出しは 41 本で作れる。Intel の実機では未確認
- g1 にした後の実機: 読み込みの dit_blocks_20 で「bad allocation」(ドライバのコンパイラの C++ の std::bad_alloc)。text と 20 ブロック (約 1.2 + 20 × 0.14 GB) が載った後なので、16.143 の上限が別の形で出たとみる。`types.IsOutOfMemory` が "bad allocation" / "bad_alloc" もメモリ不足と見なし、ほかの部品を閉じて読み直すようにした。text を閉じれば 8 ブロック分ほど空くので、読み直しはプロンプトごとに text と数ブロックの見込み

### 16.147 Anima を AMD の NPU で 70 → 44 秒: DiT の注意 (64 キーの塊、QK^T を bfp16) と VAE の格子のままの畳み込み — 2026-11-10

1024 × 1024 の 1 枚が M2 の 45 秒、M3 の 40 秒に対して 70 秒 (ノイズ除去 58 秒、VAE 9.3 秒) だった。DiT の 1 ステップ 7.3 秒の大半が自己注意 (4096 × 4096、16 ヘッド × 128)。

**注意 (`tools/amd/iron_mha.py`、`ImageMHA`)**

- 単体 (4096 × 4096 × 16 × 128、因果なし、8 本) で 146 ms。B_q を 16 にすると 292 ms、32 で 146 ms で、時間は計算量でなく塊の大きさで決まっていた。B_q と B_kv を別々に動かすと (MHA の `key_block` を差し替えて) 時間は B_kv だけで決まる: 16 で約 280 ms、32 で 145 ms、64 で 103 ms (B_q 16〜64 で同じ)
- 原因 1: mha.cc の速い softmax (`partial_softmax_rows`、8 行ずつ) は B_kv = 64 のときだけ通り、ほかは 1 行ずつの遅い道。d = 128 では B_q = B_kv = 64 がコアのメモリに入らないので、IRON は 32 × 32 を選び、softmax のコアが律速だった (ハードウェアのトレースで softmax のコアだけがロック待ちなし)。因果でも窓でもなければ、キーの塊は問い合わせの塊と同じでなくてよい (比べるのは mha.cc の因果の飛ばしだけ) ので、`ImageMHA` はキーを 64 ずつにする。B_q 32、P·V を 2 コア (`pv_cores=2`) で 146 → 103 ms、出力は元と同じ (参照とのコサイン 0.999983)
- 原因 2: 次の律速は QK^T のコア。トレースで 1 呼び出し (32 × 128 × 64) 約 10 800 サイクル、1 サイクル 24 積和ほど。K の 8 × 8 のタイルの転置を内側のループから出して一度だけにしても同じだったので、転置ではなく AIE2P のネイティブ bf16 の mmul の速さ。bfp16 の模擬 (`emulate_bf16_mmul_with_bfp16`、`Precision.MEDIUM` の GEMM と同じ) では 1 400 サイクル。IRON の MHA は「P·V の bfp16 は 8 キーで指数を共有する。softmax が時間を決める」として積を bf16 にしているが、P·V を 2 コアにすると QK^T のオブジェクトは別になるので、QK^T だけを bfp16 にできる: 103 → 59 ms (コサイン 0.999936)。P·V も bfp16 (`pv_bfp16`) で 54 ms (0.999877)
- AIE2P の exp2 命令 (`accurate_exp2` を切る) は 10 ms 速いが誤差が 4 倍 (コサイン 0.9991) なので使わない
- IRON のコンパイルのキャッシュ (`~/.npu/cache`) は演算子のパラメータと生成関数で引くので、メソッドを外から差し替えただけでは前の ELF が使われる (同じ ELF が出て気づいた)。`qk_bfp16` と `pv_bfp16` は宣言したパラメータ (`param(array=True)`) にした
- トレースの取り方: `iron.common.tracing.dump_traces` (pyxrt は `iron314.sh` の XRT で動く)。MHA の 8 本で P·V を 2 コアにすると、トレースの経路が足りず配置できない (32 チャネルに 33)。4 本で 1 本目のコアを見た
- `iron_anima.py --dit-mha qk|qkpv|bf16` (既定 qk)。自己注意と交差注意 (4096 × 512 を 512 行ずつ) の両方

**VAE (`iron_anima.py --vae-conv grid`、既定)**

- Sana の 16.139 と同じ: 3 × 3 の畳み込みを `iron_gridgemm.GridGEMM` で格子のまま (im2col の写しなし)。3 C が 512 以下なら GEMM 1 つ、超えれば kernel の行ごとに 3 つを足す。拡大は 4 つの出力の位相をそれぞれ低い解像度の 2 × 2 の畳み込みで (`PHASE_TAPS`、拡大した格子を作らない)。格子の並びは Anima と Sana で同じなので、そのまま移せた
- 同じ潜在から、im2col の VAE と格子の VAE の画像の PSNR は 50.5 dB (格子だけ)、52.7 dB (位相も)。VAE 9.3 → 5.2 秒

**結果 (Ryzen AI 9 365、1024 × 1024、8 ステップ、`~/models/anima-turbo-1024.amd-npu2` は DiT の重み bf16)**

| | 前 | 後 |
|---|---|---|
| ノイズ除去 | 58 秒 (7.3 秒/ステップ) | 36 秒 (4.6 秒/ステップ) |
| VAE | 9.3 秒 | 5.2 秒 |
| 1 枚 | 約 70 秒 | 約 44 秒 |
| 1 枚の電力量 (amdgpu の PPT、パッケージ全体、0.1 秒ごとに積分) | 1003 J (平均 14.4 W) | 688 J (15.7 W) |
| 固定されたメモリの最大 | 2409 MiB | 2409 MiB |
| 512 × 512 | 20 秒 | 14 秒 (ノイズ除去 1.4 秒/ステップ、VAE 1.4 秒) |

diffusers の fp32 (CPU) の参照と同じノイズから (DiT の重み q8 で比べる):

| `--dit-mha` | 1 枚 | 最後の潜在のコサイン | 画像の PSNR |
|---|---|---|---|
| 前 (IRON の MHA) | 70 秒 | 0.961 | 18.6 dB |
| bf16 (64 キー、積は bf16) | 56 秒 | 0.958 | — |
| **qk (既定)** | **44 秒** | **0.947** | **17.6 dB** |
| qkpv | 約 42 秒 | 0.930 | 17.2 dB |

- qk はほかの 4 組 (少女、猫と窓辺、騎士、山と湖) でも構図、顔、背景は同じで、細部 (服のしわ、盾の模様) が違う程度。前の書き出し (DiT の重み bf16) との潜在のコサインは 0.954〜0.968 で、q8 の重みの差も含む。qkpv は胸元の形まで変わったので既定にしない
- 手元の 5 つの書き出し (bf16、q8、q6、q4 (GPTQ)、512) を作り直して置き換えた。DiT の書き出しは 40 秒前後、VAE は 1.5 分
- 残り: 1 ステップ 4.6 秒 (1 ブロック約 165 ms) のうち注意は約 70 ms (自己注意 59 ms と交差注意)。次の律速は P·V のコア (bf16、1 呼び出し約 6 000 サイクル) と softmax の多項式の exp2。DiT の射影と MLP (bfp16 の GEMM) と正規化がブロックの残りの約 6 割

### 16.148 Turbo でない Anima (Anima-Aesthetic、Anima-Base) と CFG、Sana-Sprint を外す、画像モデルの名前 — 2026-11-10

**名前 (models.yaml)**

- ファミリー `anima` は Turbo でない版にした: `anima` (`anima:aesthetic`) は Anima-Aesthetic v1.1、`anima:base` は Anima-Base v1.0。Turbo はファミリー `anima-turbo` (`Anima-Turbo`、タグ v1.1)。用途 `image` (`hwnpu-image generate` の既定) は速い Anima-Turbo のまま
- Anima-Base は `machine: 0` (名前を明示したときだけ。ファミリーだけでは Aesthetic)
- 新しい欄 `file`: 1 つのリポジトリにある複数のモデルのうち、そのモデルのチェックポイント。Anima の `split_files/diffusion_models/` の 3 つを選ぶ。hwnpu-image は `--dit` の既定にこれを使い、AMD の書き出しの案内にも `--dit` を出す
- Sana は Sana 600M と SANA 1.5 1.6B だけにし、ファミリー `sana-full` を `sana` にした (`sana` は 8 GB で 600M、16 GB から 1.6B)

**Turbo でない Anima** (amd-npu-models.md 11.6)

- 重みは Turbo と同じ構造、同じ形 (685 本)。Base だけ名前の頭が `net.` (`model.diffusion_model.` でない) なので、`iron_anima.py`、`ref_anima.py`、`mil.OpenWeights` で見分ける
- CFG: `anima.Pipeline.Denoise` に負のプロンプトの条件づけとガイダンスの強さ。既定は書き出しの `checkpoint` (IRON の `dit_blocks` の meta、Core ML と OpenVINO の manifest の `dit`) で決め、turbo なら 8 ステップ CFG 1、ほかは 30 ステップ CFG 4 (モデルカードの 30〜50 と 4〜5)。記録のない前の書き出しは Turbo とみなす
- エンジンの `velocity` は条件づけの列を取る。IRON は `dit_in` を 1 回、重みの組ごとに条件と無条件を続けて走らせる (重みの読み直しは 1 ステップ 7 回のまま)。512² で 1 ステップ 3.0 → 2.7 秒、潜在はビット単位で同じ。Core ML と OpenVINO も組ごとに両方を走らせる (メモリが足りずに読み直すプログラムを 1 ステップ 1 回に)
- 空の負のプロンプト: diffusers の条件づけは重みだけで決まる 1 行 (マスクした Qwen3 のトークン 1 つ、`</s>` だけの T5。交差注意はバイアスがなく 0 を足す)。書き出しの `text/empty_cond.bin` に持たせた (`iron_anima.empty_cond`、`anima.EmptyCond`、どちらも diffusers とコサイン 1.0000)
- `ref_anima.py --guidance --negative` (diffusers の ClassifierFreeGuidance、空のプロンプトの扱いも同じ)
- 結果 (Ryzen AI 9 365、q8): 1024² の 30 ステップ CFG 4 で 1 枚 4 分 40 秒、4.8 kJ (Turbo は 44 秒、715 J)。固定の最大は Turbo と同じ 2409 MiB。512² の CPU fp32 の参照と同じノイズから潜在のコサイン 0.993、PSNR 27.4 dB (Turbo の 1024² 8 ステップは 0.947、17.6 dB)。CFG と 30 ステップで NPU の誤差が広がることはなかった
- Core ML と OpenVINO の書き出しも `empty_cond.bin` と `dit` を書くが、Mac と Intel の実機では未確認

**Sana-Sprint を外す**

- Sana-Sprint 0.6B と 1.6B は、2 ステップで速い (1024² で 12〜32 秒) が、絵の質が Sana よりかなり低かった。models.yaml から外し、SCM の刻み (`SCMTimesteps`)、guidance の埋め込み (時刻の埋め込みの `guidance_embedder`、`dit_in` の `gproj`)、ステップごとのノイズを消した
- `iron_sana.py`、`ref_sana.py`、`hwnpu-image convert` は `guidance_embeds` のチェックポイントを、`sana.Open` は `sprint` の書き出しを、理由を添えて断る
- 過去の記録 (16.119、16.138〜139、16.145) の Sana-Sprint の数字はそのまま残す

### 16.149 物の検出 (RTMDet) と文字の読み取り (PP-OCRv6): `hwnpu-image detect`、`hwnpu-image ocr` — 2026-11-11

- 動機: Google Cloud Vision の物の位置 (object localization) と OCR にあたるもの。汎用の視覚言語モデル (LFM3-VL) でも座標を文で書かせられるが、箱ごとに数十トークンを生成するので 1 枚数秒かかり、小さな物や多くの物を落とし、座標がずれる。専用の 1 段の検出器は 1 回の推論で全部の箱を出す (NPU で数 ms)。視覚言語モデルは、検出した箱の切り出しを説明させる後段に向く
- モデル (どれも Apache-2.0、ONNX で配られるものを `internal/onnx` で):
  - RTMDet-tiny / s / m (OpenMMLab MMDetection の COCO の重みを MMDeploy で書き出したもの、`bukuroo/RTMDet-ONNX`。box AP 41.1 / 44.6 / 49.1)。RTMO と同じ CSPNeXt の家族。8 GB は s、16 GB 以上は m
  - PP-OCRv6 small (PaddlePaddle の `PP-OCRv6_small_det_onnx` と `PP-OCRv6_small_rec_onnx`)。認識器は 50 言語、辞書 18708 字 (仮名あり)。tiny の認識器は辞書 6904 字で仮名がない。2 つのリポジトリを読むので、models.yaml に `with: {detector: hf:...}` を足した (`chat.Model.With`、ハッシュは `detector/FILE`)
- `internal/onnx` に足したもの: HardSigmoid (clip)、BatchNormalization (チャネルごとの積と和)、GlobalAveragePool (reduce_mean)、AveragePool (重み 1/(kh kw) の深さ方向の畳み込み)、ConvTranspose (核 = ストライドのとき 1 × 1 の畳み込みと `pixel_shuffle`。DA3 と同じ)、`auto_pad` の SAME_UPPER / SAME_LOWER、MatMul と定数の Add を bias つきの 1 つの linear に、`FlatRows` / `SplitFlat` (グラフなしで Flat の出力を分ける)
- **Neural Engine の sigmoid と tanh はずれる**: RTMDet-s を CPU インタプリタ (fp32 でも fp16 の丸めでも) で動かすと ONNX Runtime と箱の差 7.5 px (スコア 0.1 以上のアンカー) で、重みだけ fp16 に丸めた ONNX Runtime と同じ。Neural Engine では 98 px、スコア 0.35 ずれ、物が 1 つ減った。途中の値を Neural Engine と CPU で比べると (`TestBisect`)、SiLU (Sigmoid × x、80 個) を通るたびに広がる: 最初の畳み込み 0.0011、3 つ目の SiLU の後 0.014、ステージ 2 の後 0.155、頭 0.108 (相対 RMS)。Sigmoid を `1 / (1 + exp(-x))`、SiLU を `x / (1 + exp(-x))` と書くと (`onnx.Program.ExpSigmoid`、blocks.SiLU と同じ形)、ステージ 2 の後 0.014、頭 0.014、箱 5.5 px、スコア 0.063 で、重みの丸めの分まで戻った。`0.5 x (1 + tanh(x / 2))` もずれる (スコア 0.30)。演算が 2 つ増えて、ネットワークは s で 6.6 → 9.3 ms、m で 12.2 → 18.2 ms。精度を取る。既定は切り (RTMO、YuNet、Basic Pitch のプログラムは変えない)、RTMDet と PP-OCR で入れた
- RTMDet の切り口: 頭の畳み込み (段ごとのクラスの logit と箱の 4 つの距離、チャネルが先) まで NPU。グラフはその後でアンカーごとに並べ替え (Transpose)、sigmoid、箱の復号、上位 5000、クラスごとの NMS をする。ホストは logit が最小スコアの logit を超えるセルだけを復号する (sigmoid は単調なので、残る物は同じ)。Neural Engine には並べ替えがない
- ホストの縮小 (`lfm3vl.ResizeRGB`、PIL と同じ双線形) が 1 枚 37〜55 ms でネットワークより重かった。画素ごとのクロージャをやめ、縦の段を行ごとに、行を CPU の数に分け、YCbCr の変換を直接 (`color.YCbCr.RGBA` と同じ整数の計算) にして 5〜7 ms。結果のバイトは前と同じ (`TestResizeRGBSameBytes` が前の書き方と比べる)。LFM3-VL、RTMO も速くなる
- RTMDet の結果 (M1、800 × 533 と 800 × 800 の写真、20 回の平均):

| | 1 枚 | ネットワーク | 縮小 | 箱 (ホスト) | footprint の最大 | Neural Engine |
|---|---|---|---|---|---|---|
| tiny | 13〜15 ms | 7.0 ms | 5〜7 ms | 0.8 ms | 56 MB | 11 MB |
| s | 15〜17 ms | 9.3 ms | 5〜7 ms | 0.7 ms | 73 MB | 21 MB |
| m | 24〜26 ms | 18.2 ms | 5〜7 ms | 0.9 ms | 141 MB | 56 MB |

  - 照合 (`TestRTMDet`、s、football-match.jpg と city-streets.jpg): 全アンカーのスコアの差 0.063 / 0.042、箱 5.5 / 6.7 px (スコア 0.1 以上)。スコア 0.3 以上の物は ONNX Runtime と同じ (0.33 未満の境目の 1 つは出入りする)。`detect.Open` は ONNX を読み直していた (重みを float64 で、m で 529 MB) のをやめ、出力の大きさとストライドからクラスの数を出す
- PP-OCR の形:
  - 検出器は 960 × 960、1280 × 736、736 × 1280 のうち縮める割合が最も小さいもの。画像は左上に置き、残りは 0 (正規化の後)。M1 で 13 ms。PaddleOCR 3 の既定 (縮めずに 32 の倍数) と比べ、960 に縮めても論文 1 ページの 98 行のうち 94 行が同じ (CER 0.12 %)
  - 認識器の出力 (8 行 × 160 ステップ × 18710 クラス) の softmax と、出力 y の identity が CPU に置かれた。18710 は Neural Engine の次元の上限 16384 を超え、その幅のテンソルを読む演算はどれも CPU になる。softmax の手前の logit を出力にし (確率は Decode が最大の値の周りだけで計算)、最後の linear を 9355 クラスずつ 2 つに分けて (…, 2, 9355) に積む (`onnx` の linear: プログラムの出力で幅が上限を超えるとき。並びは (…, 18710) と同じ)。bias の Add も linear に入れた。全部 NPU
  - 出力は 1 回 24 MB (fp16) あり、float32 に変えるだけで 9.6 ms かかった。`runner.F16` で fp16 のまま受け取り、最大は fp16 のビットを整数として比べ (`sortKey`)、行の幅の中のステップだけを読み、確率は最大から 16 以内の値だけ float32 にする。読み取りは次の呼び出しの間に (出力を写して別の goroutine で)
  - **認識器は行の後ろの 0 の幅で読み方が変わる**: SVTR の注意が幅全体を見るため。230 px の「SUB TOTAL」は 320 幅で正しく、640 幅では「SUBTOTAL」。PaddleOCR は 1 回の行をその中で最も広い行の幅にそろえる。初めは 320、640、1280 の 3 つの幅だった (1280 を超える行は縮め、レシートの 2 行が違った)。320、480、640、960、1280、1920、2560 (前の 1.5 倍まで) にし、幅でそろえた行を幅ごとに最大 8 行、5120 列まで 1 回に読む (1280 で 4 行、2560 で 2 行)。幅ごとのプログラムは初めて使うときに変換し、それぞれ初回のコンパイルに 5〜20 秒
- PP-OCR の結果 (M1、3 回目からの平均):

| | 行 | 前 (3 つの幅、softmax が CPU) | 後 | 内訳 (後) |
|---|---|---|---|---|
| 日本語の 4 行 (900 × 420) | 4 | 113 ms | 58 ms | 検出 13 ms、認識 30 ms (3 回) |
| 論文 1 ページ (1224 × 1584) | 98 | 1528 ms | 488 ms | 検出 13 ms、箱 16 ms、切り出し 34 ms、認識 342 ms (30 回)、読み取りは重なる |
| レシート (576 × 864) | 13 | 97 ms | 41 ms | 認識 18 ms (3 回) |

  - PaddleOCR の手順を ONNX Runtime で動かした参照 (`tools/ref/ref_ppocr.py --fit 960x960`: cv2 の輪郭と最小の回転矩形、pyclipper の unclip、`get_rotate_crop_image`、1 回 6 行で最も広い行の幅、`tools/ref/compare_ocr.py`) と比べて、論文 93 / 98 行、レシート 13 / 13 行、日本語 3 / 4 行が一字一句同じ (CER 0.20 %、0、3.9 %)。違う行は数式の行、−と–、{βi}、全角と半角のコロンで、参照より正しいものもある (Entropv → Entropy、营 → 営)。NPU と CPU インタプリタは同じ行を出す (違いはホストの切り出しと幅)
  - 「〜」(U+301C) は辞書にあるが、PaddleOCR の参照も読まない (モデルの読み方)
  - 2 段組みの文書は、行が左右交互に並ぶ。PaddleOCR の `sorted_boxes` (上から、y の差 10 px 以内は左から) と同じ。段組みの解析 (PP-StructureV3 など) はない
  - プロセスの footprint は最大 425 MB (7 つの幅のプログラムとその出力の 12〜24 MB の領域)
- 組み合わせの値 (10 節): 入力は JPEG / PNG (RTMDet は 640 × 640 に収める、PP-OCR は検出器の 3 つの大きさ)、出力は JSON (箱、ラベル、文字列)。1 枚ずつで、流れ (streaming) はない。部品は RTMDet が 1 つ、PP-OCR が検出器 1 つと認識器の幅ごとに 7 つ (一緒に動く)
- 確かめていないこと: 電力 (`hwnpu-dev powerbench` は sudo が要る。`sudo go run ./cmd/hwnpu-dev powerbench -one -model ~/Library/Caches/hwnpu/models/RTMDet-ONNX/rtmdet-rtmdet-s-coco-r1-fp16-640.mlpackage`)、Intel の NPU (OpenVINO IR に変換はするが実機は未確認)、AMD (書き出しなし。`--cpu` で CPU インタプリタ)、RTMDet-tiny と m の ONNX Runtime との照合 (s だけ。同じコードで物の数と箱は s とそろう)、PP-OCRv6 medium、180 度回った行 (文字の向きの分類器 PP-LCNet は入れていない)、縦書き、RTMO と YuNet に `ExpSigmoid` を入れたときの精度 (RTMO も SiLU の CSPNeXt なので良くなる見込み)

### 16.150 Qwen3-1.7B: MOSS-SoundEffect の文の符号器をチャットに — 2026-11-11

- 動機: ほかのモデルの中で動いていて、単独でも使えるモデルを探した。Anima のテキストエンコーダ (`text_encoders/qwen_3_06b_base.safetensors`) は Qwen/Qwen3-0.6B-Base とバイト単位で同じ (sha256 `cd2a5120…`) で、事前学習だけの base。チャットの Qwen3-0.6B (16.127) とは別のチェックポイントなので、共有はしない。MOSS-SoundEffect v2.0 の `text_encoder/` は Qwen/Qwen3-1.7B (post-trained) の 2 つのファイルとバイト単位で同じ (`169ad53e…`、`912becff…`) なので、それをチャットに足した。Sana の Gemma 2 は gemma-2-2b-it (`Gemma2Model`) だが、Sana のは 512 行を 1 回通す符号器だけ (注意の softcapping なし、窓なし、生成のループなし) で、チャットの型 `gemma2` はない。同じ大きさなら Gemma 4 E2B があるので移さない
- 変換: コードの変更なし。`qwen3` の型は大きさによらない (hidden 2048、28 層、16 / 8 ヘッド、head_dim 128、MLP 6144、埋め込みと共有)。models.yaml に `Qwen3-1.7B` (`qwen3:1.7b`) を足しただけ。KV は 0.6B と同じ形なので、キャッシュも 0.6B と同じ 112 KiB/トークン、分けない 1 本で文脈 4096 (`llm.Unpaged`)
- 照合 (`tools/ref/ref_logits.py --seq 32`、チャットの型の日本語の文、`hwnpu-dev check`): fp32 で KL 0.0000、32/32。`-f16` も KL 0.0000、32/32 (残差は層 3 から最大 14,573 で 0.6B の 8,141 より大きいが、正規化の 1/64 で足りる)。lut8 は KL 0.0181 (32/32)、lut6 は 0.3148 (25/32) なので lut8 のまま
- M3 (16 GB) で: 初回は変換 4 本で各 5〜6 s、Core ML のコンパイルが 2 本で 32 s と 35 s、全部 NPU。2 回目からは読み込み 1.2〜2.2 s。生成 13〜22 tokens/s (プロンプトからの下書きつき。思考ありの 139 トークンで 13.1、コードの 300 トークンで 21.9)
- メモリ (`hwnpu-chat serve` に 4 回問い、`footprint` と `mem.NPUWired` を外から): Neural Engine は読み込んだ時点で決まり、問いの長さや回数で変わらない (4096 のキャッシュはプログラムの中)。2 本 (既定) で 3.26 GiB、1 本 (`--prefill off`) で 1.63 GiB。プロセスの footprint は読み込みが最大で、答えると下がる: 2 本で 3.0 GB (serve) 〜4.1 GB (`chat` の 1 回の最大)、答えている間 1.5〜2.1 GB。1 本で 2.2〜2.8 GB、答えている間 1.0〜1.6 GB。合わせて最大 7.2〜7.6 GB と 4.0〜4.5 GB。最初の測定の「答えた後 3.96 GB」は再現しなかった (同じ問いで 1.47 GB)。読み込みで残ったものがまだ返されていなかったと見ている
- 1 本の損はプロンプトを読む速さだけ: 新しい 1822 トークンを 2 本で 3.4 s、1 本で 16.8 s (1 回 8 トークン)。生成は同じ (14〜15 tokens/s)。プロンプトストアにあるものは読み直さない
- そこで models.yaml にモデルの動かし方 (`prefill`、`format`、`context`、`paged`) と機械のメモリごとの差分 (`variants`) を足した (`chat.Run`、`chat.Variant`、`Model.On`。docs/model-table.md 6.2 節)。Qwen3-1.7B は `machine: 8`、`memory: 4.5`、`prefill: off`、`variants: {16: {memory: 7.6, prefill: auto}}`。hwnpu-chat の `--prefill` の既定は `auto` から「モデルの定義、なければ auto」に。`HWNPU_MEMORY=8` で 1 本 (1.63 GiB)、16 で 2 本 (3.26 GiB)、8 でも `--prefill on` なら 2 本になるのを M3 で確かめた。ファミリー `qwen3` は 8 GB でも 1.7B を選ぶ (同じ列では memory の大きい方: 0.6B の 3.9 GB と 1.7B の 4.5 GB)。0.6B は `qwen3:0.6b`
- MOSS-SoundEffect とダウンロードを共有する: リポジトリが違うので `Model.DirName` では共有されない。そこで、定義が `sha256` で固定したファイルを SHA-256 ごとにまとめ (`pinned`)、`fetch` は落とす前に同じ SHA-256 のファイルをほかのモデルのディレクトリに探して、ハードリンク (できなければ写し) にし、SHA-256 を確かめる (`sameFile`)。違えば消して取得元から。Qwen3-1.7B はリビジョン (`70d244cc…`) と 2 つのファイルの SHA-256 を、MOSS-SoundEffect は `text_encoder/` の 2 つを書いた。M3 で、Qwen3-1.7B を落としたあと MOSS-SoundEffect の `text_encoder/` を `chat.Fetch` で取ると 2 つともリンク (同じ inode、ディスクは増えない、SHA-256 の確認込みで 2.2 s)。逆向き (MOSS-SoundEffect だけがある) も同じ。テスト (`TestSameFile`) は、リンクして HF に問わないこと、別の SHA-256 は落とすこと、変えられた写しは使わず残すことを確かめ、`sameFile` を外すと落ちる
- 答え (温度 0、思考なし): 17 × 23 = 391 は正しい。Go の文字列の反転 (rune で) は正しい。道具の呼び出し (`serve`、get_weather) は `{"city":"Tokyo"}` を JSON で返す。富士山 (1,644 m)、『こころ』(森鷗外)、日本一長い川 (琵琶湖) は誤る。HF の fp32 (transformers、貪欲) でも 1,689 m、琵琶湖と同じように誤るので、モデルの限界
- 確かめていないこと: 読み込みのときのプロセスの最大の内訳、8 GB の実機 (`HWNPU_MEMORY=8` の M3 で確かめただけ)、Intel の NPU、AMD の書き出し、hwnpu-chat agent の課題

### 16.151 ラベル付けと画像の埋め込み (SigLIP 2) を AMD の NPU で、ONNX の 5 つのモデルを AMD の機械の CPU で — 2026-11-11

- 動機: CLOUD_AI_COMPATIBILITY.md の画像の順の 1 番目。Cloud Vision の LABEL_DETECTION とマルチモーダルの埋め込みにあたる。`hwnpu-image label` (候補のゼロショット、既定は組み込みの約 440 ラベル)、`hwnpu-image embed` (`--compare`)。models.yaml の用途 `label` (8 GB は siglip2-base-patch16-224、16 GB 以上は siglip2-large-patch16-256)
- 形と精度は docs/amd-npu-models.md 18 節。文の塔はトークンが先の行で、8 つの文の注意を 1 回の MHA (8 × heads 個の頭) に。IRON の tanh の GELU は表で引くので bf16 の丸めの 5 倍ずれ (IRON の参照もまねるので照合では見えない)、exact の GELU に替えた。行列積を HIGHEST にして、画像の埋め込みのコサインが 0.9988 → 0.9998 (HF の bf16 と同じ程度)、時間は 9〜22 % 増
- 同時に、`internal/onnx` で Core ML と OpenVINO に変換する 5 つ (RTMO、RTMDet、PP-OCRv6、SwiftF0、Basic Pitch) を、AMD の機械 (と NPU のない機械) では ONNX Runtime で CPU で動かす (`onnxrt.Cut`、`onnxrt.Program`、`chat.ConvertedDevice`)。今までは `--cpu` の CPU インタプリタだけで、RTMDet-m が 1 枚 8.5 秒だった (1.4 秒に)。docs/amd-npu-models.md 19 節
- 実行時の外部依存をなくすため、ONNX Runtime がなければ hwnpu の CPU 実行器 (milcpu) で動かす。milcpu に行列積 (`internal/sgemm`、AVX2 / AVX-512 のカーネル、im2col を詰めながら) と SIMD の exp、sigmoid、SiLU、erf (`internal/f32`) を入れ、RTMDet-m を 1 スレッド 2.25 秒に (前は 12 コアで 7.1 秒、ONNX Runtime 1.37 秒)。19.1 節

### 16.152 Sana と Anima の確認の残り: ダウンロードの sha256、SANA 1.5 の起動、複数の種、Mac の Anima-Turbo と diffusers — 2026-11-11

- **ダウンロードの sha256**: Sana (600M、SANA 1.5) と Anima (Turbo、Aesthetic、Base) が取るファイルをすべて models.yaml の `sha256:` に固定した。取得のときに `chat.Fetch` が確かめる (前からある手元のファイルは確かめ直さない)。Anima は 2 つのリポジトリを読むので `with: {diffusers: hf:circlestone-labs/Anima-Base-v1.0-Diffusers}` を足し、そちらのファイルは `diffusers/FILE` と書く (16.149 の `with` と同じ。`chat.Pinned` で引く)。取るファイルの一覧は `sana.Files` と `anima.Files` にまとめ、`TestFilesPinned` がどれも固定されているかを確かめる。Sana の Gemma 2 は最初の Sana のリポジトリから 1 回だけ取る。ほかのリポジトリの写しは `text_encoder/config.json` のメタデータ (`_name_or_path`、`transformers_version`) だけが違うので、ハッシュはリポジトリごとに持つ
- **SANA 1.5 の起動** (M1 16 GB、ほかのセッションが Neural Engine を使っていない状態、新しいプロセスで続けて 3 回):

| Gemma 2 | 1 回目 | 2 回目 | 3 回目 | Gemma 2 の時間 | footprint の最大 |
|---|---|---|---|---|---|
| int8 (既定、5 本) | 7m10s (再コンパイル) | 1m12s | 1m13s | 2.5 s | 3.55〜3.57 GB |
| fp16 (3 層ずつ 9 本) | 14m46s | 1m21s | 1m21s | 10.5〜11.2 s | 最大 3.85 GB |

  - 2 回目からは読み込みがどれも 2.4 秒以下で、Neural Engine のキャッシュに当たる。1 回目は直前に作った fp16 の書き出しが押し出したもの (DiT の塊が 1 つ 51〜53 秒、VAE の s1 が 1m20s)。16.145 の「1.6B は外れが続く」と 11/10 の 6m47s は、ほかのセッションが同時に Neural Engine でコンパイルしていたことと、書き出しを作り直して互いに押し出していたことによる。合計 (Gemma 2 int8 約 2.2 GB、DiT fp16 約 3.2 GB、VAE) は 4 GiB を大きく超えるが当たる。M2 の合成プログラムでの「本数で決まる容量」([handoff-ane-cache.md](handoff-ane-cache.md) 1.7) は、M1 の SANA 1.5 には当てはまらない
  - Gemma 2 の fp16 は 4 倍遅く、当たるかどうかも変わらないので採らない。int8 のまま
  - 1 ステップ 2.5〜3.4 秒、20 ステップで約 66 秒
- **複数の種** (M1、1024 × 1024、3 プロンプト × 種 1、2、diffusers の fp32 を CPU で作り、同じノイズから Neural Engine で):

| | 潜在のコサイン | PSNR (dB) | 1 枚 (2 回目から) | NPUWired の最大 | footprint の最大 |
|---|---|---|---|---|---|
| Sana-600M (既定、int8 / fp16) | 0.989〜0.9998 (中央値 0.998) | 22.5〜38.7 | 21〜22 s (20 ステップ 14 s) | 1.14 GiB | 3.87 GB |
| Anima-Turbo (int8 / int8) | 0.876〜0.948 (中央値 0.914) | 12.9〜20.4 | 58〜59 s (8 ステップ、1 ステップ 6.5 s) | 2.62 GiB | 3.92 GB |

  - 12 組とも構図 (人物の数と位置、窓と猫、騎士の向き、看板の文字) は参照と同じ。違うのは細部だけ (Anima の p2 種 1 で猫の毛色と姿勢が少し違う)。Sana のいちばん低い 1 組 (和室の水彩、種 2) も構図は同じ
  - Anima-Turbo は 16.85 の M3 の照合 (int8、0.672〜0.977、14.1〜25.8 dB) の範囲で、PR #109 の後も Mac の Anima-Turbo は diffusers から離れていない。8 ステップに蒸留したモデルは小さな違いを広げる (16.138 の Sana-Sprint と同じ)
  - 1 回目は書き出し (Sana は ExportVersion 4、Anima は 3) を作り直したので変換とコンパイルで Sana 6m47s、Anima 18m14s。2 回目からはどちらも全部のプログラムが all on the NPU で当たる
  - models.yaml の Apple のメモリを測った値に直した: Sana-600M 3.3 → 3.9、SANA1.5-1.6B 3.1 → 3.6 (Anima の 4.1 は 3.92 GB で足りる)
  - 道具: 参照は `tools/ref/ref_sana.py` (torch 2.5.1、diffusers 0.33.1) と `tools/ref/ref_anima.py` (diffusers 0.42 以上が要り、python 3.9 では入らない。Homebrew の pytorch と python 3.15 に diffusers 0.42.1、transformers 5.22.2)。CPU で 1 枚 Sana 7〜8 分、Anima 約 10 分。NPUWired は `mem.NPUWired` を 0.1 秒ごとに読んだ最大 (Neural Engine の machine 全体の固定メモリ)
- **低電力モードを切って測り直した** (M1、AC 電源、`pmset` の lowpowermode 0、新しいプロセスで 3 回、同じプロンプトと種): Sana-600M 19〜21 s (Gemma 2 2.2〜2.8 s、1 ステップ 0.65 s、VAE 3.4〜3.7 s)、Anima-Turbo 58〜59 s (1 ステップ 6.4〜6.5 s、VAE 2.6〜3.0 s)、SANA1.5-1.6B 1m10s〜1m12s (1 ステップ 2.5〜3.4 s、VAE 3.6〜3.7 s)。上の値と同じ。M3 では低電力モードで画像生成が半分の速さになったが、M1 のこの計測では差がない (上の計測のときに低電力モードが入っていたかは記録がない)
- 確かめていないこと: Anima-Base (Turbo でない方、PR #109 で入った) の照合と Intel の NPU (どちらも別に)

### 16.153 同意の要るリポジトリ (gated) のダウンロード: Hugging Face の OAuth のデバイスフロー — 2026-11-11

FLUX.1 [schnell] (Apache-2.0) は Hugging Face のページで利用条件に同意しないと落とせない (`gated: auto`、トークンなしの `resolve` は 401、同意していないアカウントのトークンは 403)。これまでの `fetchHF` は `HF_TOKEN` を送るだけで、401 はそのまま「downloading ...: 401」で止まっていた。

- トークンの順: `HF_TOKEN`、huggingface_hub のトークンファイル (`HF_TOKEN_PATH`、`$HF_HOME/token`、`~/.cache/huggingface/token`: `hf auth login` が書く場所、読むだけ)、このプロセスでサインインしたもの (`internal/chat/hfauth.go`)
- 401 で、API (`/api/models/OWNER/NAME`、トークンなし) が `gated` と言い、端末に人がいれば、OAuth 2.0 のデバイスフロー (RFC 8628) で Hugging Face にサインインする。hwnpu の公開 OAuth アプリ (`hfClientID`、秘密鍵なし、`HWNPU_HF_CLIENT_ID` で差し替え) のスコープは `openid` と `gated-repos` だけ: 同意済みの公開の同意制リポジトリを読めるだけで、非公開のリポジトリもプロフィールも読めない。表示したコードを https://hf.co/oauth/device に入れて承認すると、`/oauth/token` をポーリングしてトークンを得る。SSH 越しでも、ブラウザのない機械でも使える
- トークンは保存しない (プロセスのメモリだけ)。アプリのトークンの寿命は 15 分 (ファイルのダウンロードだけなので最短にした) で、34 GB を 6 MB/s で落とすと途中で切れる。サインインのときに一緒にもらった refresh token (これもメモリだけ) で、401 のときに黙って取り直す。取り直せなければもう一度サインイン
- 403 (アカウントが未同意): リポジトリのページを示し、同意して Enter で再試行
- 端末がないとき (`hwnpu-chat serve`、スクリプト): 対話せず、同意のページと `HF_TOKEN` の作り方を書いたエラー
- API が `gated` と言わない 401 (存在しない名前や非公開のリポジトリも、トークンなしでは 401) は今までどおりのエラーで、サインインしない
- 実機: FLUX.1-schnell の `model_index.json` でサインインからダウンロード、refresh token での取り直しまで確かめた (2026-11-11)。単体テストは偽のハブ (`hfauth_test.go`) で 401 → サインイン → 403 → Enter → ダウンロード、refresh、端末なし、同意制でない 401

### 16.154 Anima-Aesthetic を 4 分 40 秒 → 3 分 19 秒: DPM-Solver++ 2M の 25 ステップと、CFG を雑音の多いステップだけに — 2026-11-11

16.148 の Anima-Aesthetic と Anima-Base は、1 枚の時間のほとんどが DiT を回す回数 (オイラー法 30 ステップ × CFG で 2 回 = 60 回、1 回 4.55 秒) だった。回数を減らす手と 1 回を速くする手 ([new-model-checklist.md](new-model-checklist.md) 7.1 節) を NPU で試した。

**測り方**: 512²、Anima-Aesthetic v1.1、CFG 4、空の負のプロンプト、3 つのプロンプト (人物、夜の街、風景) × 2 つの種 = 6 枚。真値の代わりに同じ NPU のオイラー法 100 ステップ。最後の潜在のコサインと画像の PSNR (6 枚の平均)、絵を並べる。電源のプロファイルは power-saver (このマシンの設定のまま)。

| | 1 枚 (512²) | 100 ステップとのコサイン / PSNR | 30 ステップのオイラー法と | 絵 |
|---|---|---|---|---|
| オイラー法 30 ステップ (16.148) | 74 秒 | 0.982 / 25.2 dB | — | — |
| オイラー法 20 ステップ | 49 秒 | 0.961 / 21.5 dB | 0.980 | 同じ |
| DPM-Solver++ 2M 20 ステップ | 49 秒 | 0.970 / 22.2 dB | 0.983 | 同じ (1 枚が 0.924) |
| DPM-Solver++ 2M 25 ステップ | 62 秒 | 0.984 / 24.3 dB | 0.983 | 同じ |
| DPM-Solver++ 2M 15 ステップ | 37 秒 | 0.920 / 18.4 dB | 0.925 | 細部が崩れる |
| CFG は σ ≥ 0.22 だけ (オイラー法 30) | 72 秒 | 0.982 | 0.9998 | 同じ |
| CFG は σ ≥ 0.5 だけ | 67 秒 | 0.982 | 0.9989 | 同じ |
| CFG は σ ≥ 0.6 だけ | 64 秒 | 0.982 | 0.998 | 同じ |
| CFG は σ ≥ 0.8 だけ | 55 秒 | 0.974 | 0.988 | ほぼ同じ (少し淡い) |
| **DPM-Solver++ 2M 25 ステップ、CFG は σ ≥ 0.6 だけ (既定)** | **53 秒** | **0.984 / 24.0 dB** | 0.982 | 同じ |
| CFG は σ ≤ 0.85 だけ (雑音の多いステップで外す) | 62 秒 | 0.79 | 0.78 | 構図が変わり、色が淡い |
| ブロックの組の結果の使い回し、しきい値 0.02 | 65 秒 | 0.975 | 0.989 | 顔の細部が変わる |
| 同じ、しきい値 0.05 | 41 秒 | 0.918 | 0.936 | 人物が変わる |
| P·V も bfp16 (`--dit-mha qkpv`) | 74 秒 | 0.982 | 0.997 | 同じ |

1024² (人物の 1 枚、`bench.sh`: 時間、amdgpu の PPT の積分):

| | 1 枚 | ノイズ除去 | 電力量 | 平均電力 | オイラー法 30 とのコサイン |
|---|---|---|---|---|---|
| オイラー法 30 ステップ | 277 秒 | 9.0 秒/ステップ | 4753 J | 17.1 W | — |
| **DPM-Solver++ 2M 25、CFG σ ≥ 0.6 (17 ステップ)** | **199 秒 (−28 %)** | 7.6 秒/ステップ | **3821 J (−20 %)** | 19.2 W | 0.994 |
| DPM-Solver++ 2M 20、CFG σ ≥ 0.6 (14 ステップ) | 172 秒 (−38 %) | 8.2 秒/ステップ | 3736 J (−21 %) | 21.7 W | 0.993 |
| P·V も bfp16、オイラー法 30 | 264 秒 (−5 %) | 8.5 秒/ステップ | 4168 J (−12 %) | 15.8 W | 0.997 |

- **DPM-Solver++ 2M** (`anima.Sampling`、`--sampler dpm2m`): diffusers の DPMSolverMultistep をフローの σ で。データの予測 D = x − σ v、対数 SNR λ = log((1 − σ) / σ)、h = λ_t − λ_s として x_t = (σ_t / σ_s) x_s + (1 − σ_t)(1 − e^−h) D、2 次は前のステップの D と r = h_前 / h で (1 + 1/2r) D_0 − (1/2r) D_1。σ = 1 からの最初のステップ (h が無限) と 0 への最後のステップは 1 次で、どちらもオイラー法と同じ。DiT の呼び出しはオイラー法と同じ 1 ステップ 1 回なので、同じステップ数なら時間も同じ。正規分布のデータの厳密な速度場で、オイラー法より誤差が小さいことを `TestDPM2M` で確かめる
- **CFG を一部のステップだけに** (`--cfg-sigmas LO,HI`、Kynkäänniemi ほか 2024): 外したステップは DiT を条件つきで 1 回。論文は雑音の多い側と少ない側の両方を外すが、Anima では雑音の多い側で外すと構図が変わり色が淡くなった。少ない側は σ 0.6 まで外しても絵は同じ。既定は σ ≥ 0.6 (25 ステップのうち 17)
- **既定**: Anima-Aesthetic と Anima-Base は DPM-Solver++ 2M の 25 ステップ、CFG は σ ≥ 0.6 (`Pipeline.DefaultSampling`)。Anima-Turbo はオイラー法の 8 ステップのまま (CFG なし、蒸留はオイラー法で)。`--sampler euler --steps 30 --cfg-sigmas 0,1` でモデルカードのとおり
- **電力量は時間ほど減らない**: 平均電力が 17.1 → 19.2 W (20 ステップでは 21.7 W) に上がった。ただし下の電源の設定を変えた測り直しでは、どれも約 17 W にそろった。この上がり方は、その時のマシンの状態によるものだったらしい (原因は確かめていない)
- **電源の設定を変えて測り直した** (BIOS の power mode を silent → balance、Ubuntu を power-saver → balanced、CPU の EPP は `balance_performance`、2 回ずつ、2 回の差は 1 % 以内): 潜在は 4 つとも power-saver の時とビット単位で同じ

| | power-saver (上の表) | balanced |
|---|---|---|
| オイラー法 30 ステップ | 277 秒、4753 J | 271 秒 (−2 %)、4726 J、17.4 W |
| **DPM-Solver++ 2M 25、CFG σ ≥ 0.6 (既定)** | 199 秒、3821 J | **192 秒 (−3 %)、3334 J (−13 %)**、17.4 W |
| DPM-Solver++ 2M 20、CFG σ ≥ 0.6 | 172 秒、3736 J | 156 秒 (−9 %)、2713 J (−27 %)、17.4 W |
| P·V も bfp16、オイラー法 30 | 264 秒、4168 J | 257 秒 (−3 %)、4044 J (−3 %)、15.8 W |
| Anima-Turbo (8 ステップ) | — | 41 秒、711 J、17.3 W |

  - NPU のステップはほとんど速くならない (オイラー法 30 で −2 %)。NPU の速さは電源の設定にあまりよらない。power-saver で測ってきたこれまでの数字は大きくは変わらない
  - 既定の解き方は、オイラー法 30 と比べて時間 −29 %、電力量 −29 % になり、時間と同じだけ電力量も減った
- **採らなかったもの**:
  - ブロックの組の結果の使い回し (First-Block Cache を最初の組、4 ブロックで): 組の出力の差分 (残差) が、最後に全部を回したステップのものと相対差でしきい値より小さければ、残り 6 組を回さずその残差を足す。条件と無条件を別々に使い回すと、c − u のずれがスケール 4 で広がって色が抜けた (しきい値 0.05 で 0.906)。両方を揃えて使い回しても、0.02 で 5/30 ステップ (−12 %、0.989)、0.05 で人物が変わる。CFG を外すほうが同じ時間でずれが小さい。28 ブロックのうち最初の 4 ブロックでは、使い回してよいかの目安として粗い
  - CFG を雑音の多いステップで外す (上の表)
- **P·V の bfp16** (`--dit-mha qkpv`、16.147): 512² では注意が小さく速くならない。1024² で −5 %、電力量 −12 %、絵は同じ。既定 (`qk`) を変えるとすべての書き出しを作り直すことになり、16.147 では参照から細部が少し離れたので、まだ既定にしていない
- 残り: 交差注意のキーと値をステップの外へ (演算の 0.7 %)、Anima の VAE の同じ形の段をまとめる (Turbo の 12 %)

### 16.155 Qwen-Image-2.1-Turbo を AMD の NPU で、ライセンスを初回に確かめる仕組み — 2026-11-11

`Qwen/Qwen-Image-2.1-Turbo` (7.1B のシングルストリーム DiT、文を読むのは Qwen3-VL-8B、64 チャンネル・1/16 の RGBA の VAE、8 ステップ・CFG なし) を AMD の NPU に入れた。調べた中身は [image-models-survey-qwen-chroma.md](image-models-survey-qwen-chroma.md)。書き出しは `tools/amd/iron_qwenimage.py`、パイプラインは `internal/imagegen/qwenimage`、参照は `tools/ref/ref_qwenimage21.py`。詳しくは [amd-npu-models.md](amd-npu-models.md) 20 節。

- **文の K/V を一度だけ**: 文の行は t = 0 で変調される (`causal_condition`) ので、各ブロックの K/V はステップに依らない。プロンプトごとに `dit_text` で一度だけ計算し、各ステップは画像の 4096 行 (1024²) だけを回して [画像の K、文の K] に注意する。文のパディングはキーの数 `nk` (呼び出しごと) で切る (画像を先に並べるので末尾になる)
- **共有の変調を dit_in で**: 全ブロック共通の変調 (1 + scale、tanh gate) を 1 ステップ 1 回、tanh まで NPU で
- 文: Qwen3-VL-8B の 36 層 (q6、HIGHEST) を 64 行ずつ、最後の層の norm の前の出力に DiT の txt_in まで同じグラフで。システム文の 14 トークンは捨てる。システム文は 1 チャンク (64) より短いので、Sana の前置きの K/V を持つ手 (16.138) は効かない
- 正しさ (CPU の float32 の diffusers と同じ雑音から): 512² 狐 潜在のコサイン 0.9978 / PSNR 27.9 dB、512² 看板の文字 0.9995 / 31.3 dB (文字も同じ)、1024² 狐 0.9989 / 34.2 dB
- 速さ (1024²): 1 枚 1 分 52 秒 (文 5.1 秒、DiT の文 2.2 秒、ステップ 12.0 秒 × 8、VAE 約 9 秒)。CPU が別の作業で混んでいたときは 2 分 17 秒 (14.6 秒/ステップ)。見積もり (Anima の 3.75 TFLOPS から 17.7 秒/ステップ) より速い。512² は 1 ステップ 3.4 秒
- メモリ: ピーク 6.8 GB (NPU 6.1 GB、文の段の q6 の重み)
- **DiT の重みを 1 回だけ持つ**: 最初は `dit_text` と `dit_blocks` が同じ q8 の重み (7 GB) を別々に持ち、書き出しが 22 GB だった。バンクの中身は 1 バイト目から違った。違いは `dit_text` の RoPE の表 (呼び出しごとに切り出すので `iron.weight`): これが重みの並びに入って配置がずれていた。RoPE をホストが渡す入力 (チャンクの 256 行) にすると、2 つのグラフの重みは同じものが同じ順に並び、バンクの像がバイトまで同じになる。書き出しは `dit_blocks` を先に作り、`dit_text` の最初の像が同じなら自分の像を消して、どの組も `../dit_blocks/` の像を指す (`share_banks`。違えば今までどおり自分で持つ)。書き出し 22 → **15 GB**、書き出しの時間も 7 組分短い。潜在は前とコサイン 1.0 (2 つのプロンプト)、時間も同じ
- **2 回目の実行が止まる**: `dit_text` (呼び出しごとの位置に K/V を書く) と Chroma の approximator (6 コマンド) で、同じグラフの 2 回目の最初のコマンドが ERT_CMD_STATE_TIMEOUT。1 回目は正しい。言語モデルの stepBarrier と同じく、同じ ELF を別のハードウェアコンテキストに読んで空のコマンドを先に回すと止まらない (`iron.GraphOptions.Barrier`、irongraph `-barrier`)。qwenimage のグラフはすべてこれを使う
- VAE: Wan 2.2 系の DupUp3D の近道 (入力のチャンネルを画素へ) は、チャンネルが同じなら最近傍、違えば位相ごとの選択の行列 (MEDIUM: HIGHEST の GEMM は N が 64 の倍数でないと作れない) で、位相の GEMM に足す。中央の注意 (1 頭 1152) は MHA に入らないので、QK^T・行のソフトマックス・P V^T (V の転置は単位行列の GEMM) の GEMM で
- ライセンス: Qwen Research License は研究と評価だけ。models.yaml の `license` (`id`、`url`、`terms`、`accept`) で `accept: true` のモデルは、名前で選んだときに一度だけ端末で聞き、`~/.config/hwnpu/licenses.json` に残す。端末がなければ `--accept-license` か `HWNPU_ACCEPT_LICENSE=qwen-research` (後者はその回だけ) を求めて止まる (`chat.AcceptLicense`)
- 残り: 文の 64 行が 1.4 秒 (q6 を毎チャンク展開)、文の段を SwapVariants にしてピークを下げる、Apple と Intel

### 16.156 Chroma1-Flash を AMD の NPU で: T5 の相対位置のバイアス、GroupNorm を NPU で、diffusers のマスク — 2026-11-11

`lodestones/Chroma1-Flash` (FLUX.1 schnell の DiT 8.9B、ブロックの変調は時刻だけから 344 行を作る FFN、T5-XXL、FLUX の VAE、Apache-2.0) を AMD の NPU に入れた。DiT は `tools/amd/iron_chroma.py` (FLUX の試作 t3-flux のブロックから)、T5 と VAE は `tools/amd/iron_chroma_parts.py`、パイプラインは `internal/imagegen/chroma`、参照は `tools/ref/ref_chroma.py` と `ref_chroma_parts.py`。

- **T5 の相対位置のバイアス**: IRON の MHA にはバイアスがない (低ランクでもない)。8 頭ずつ、ブロック対角の K^T と V^T (射影の重みを GEMM の左に置いて頭ごとに連続に出す) で 1 回の GEMM に 8 頭の点数を並べ、バイアスは float32 の Softmax の足し算に定数 (全層で共有、64 MiB) で。パディングのマスクは点数の GEMM の縮約に入れる (K^T の 1 行に (mask − 1) × 16384、q の 1 列を 1)。q8、HIGHEST、GELU は erf (IRON の tanh GELU の表は最悪の行を 0.986 に)。NPU で 1 回 3.2 秒、プロンプトの行で float32 の transformers とコサイン 0.9997
- **GroupNorm を NPU で**: 画素のチャンネルごとの和を 1 の塊との GEMM で、グループの平均を定数の GEMM で、平方根の逆数は RMSNorm 1 回 (グループの行のノルムと定数 c の行、eps を合わせると c の出力が 1/√(var + eps) に比例)。ホストの計算はない。512² の VAE 10 段、diffusers の float32 と PSNR 53.6 dB (CPU の参照)、NPU で各段コサイン 0.99986 以上
- **diffusers の ChromaPipeline のマスク**: 文のマスクを float の 0/1 で渡し、ブロックが外積にして SDPA に渡す。float のマスクは点数に足されるので、プロンプトの組に +1 がつくだけでパディングにも注意が向く。Chroma はパディングを (1 つを残して) 隠して学習された (モデルカード) ので、NPU はキーの数 `nk` で切る。これと diffusers は最初の速度でコサイン 0.9886 (すべてのキーに注意させると 0.9984)。参照に同じマスク (`ref_chroma.py --mask keys`、SDPA をキーのマスクに直す) を使うと、最初の速度 0.99985、8 ステップの潜在 0.9988、PSNR 40.8 dB (512²、オイラー法)
- 変調: approximator (5 層、5120) は 1 呼び出し 512 行 (344 + 0)。HIGHEST は K = 5120 で作れず (バッファ記述子) HIGH。RMSNorm の重み (幅 5120 > 4096) は次の線形の列に畳む。2 回目の実行が止まる (16.155 の Barrier で直る)
- DiT: ダブル 19 は素数なので 1 ブロック 1 バンク、シングル 38 は 2 ブロック。行は [画像、文] の順 (文のパディングがキーの末尾に来る)。Heun 8 ステップ (DiT 15 回) が既定、`--sampler euler` で 8 回
- 速さ (512²、CPU は別の作業で混んでいた): DiT 1 回 5.4 秒、1 枚 Heun 1 分 42 秒、オイラー法 50 秒。T5 は FLUX.1 schnell のと同じファイル (sha256) を共有し、トークナイザは Anima の t5_tokenizer.json (FLUX のは正規化が Sequence で Go が読めない。語彙と charsmap は同じ)
- 1024²: VAE の段 6 (1024² × 256 チャンネルの GroupNorm) が 2 秒を超えた。DMA の繰り返しは 64 まで (列の塊の数を抑える)、転置の GEMM は N = 1M 画素で終わらない (262144 画素ずつのコマンドに)。1 枚 4 分 13 秒 (Heun 8 ステップ、29.9 秒/ステップ、VAE 8.9 秒)、ピーク 5.9 GB
- 残り: CFG (Chroma1-HD)、1024² の参照との比較

### 16.157 Qwen-Image-2.1 (蒸留なし) と Chroma1-HD の CFG、文の符号器を組ごとに、VAE の先読みに上限 — 2026-11-11

- **CFG**: `hwnpu-image generate` の `--guidance`、`--negative`、`--sampler`、`--cfg-sigmas` を Qwen-Image にも (Chroma にも)。1 ステップの条件つきと条件なしは重みの組ごとに続けて回す (組の重みを読むのは 1 ステップ 1 回、16.148)。プロンプトと負のプロンプトは 1 回の読み込みでまとめて符号化する (`EncodeAll`)。Qwen-Image は負のプロンプトごとに文の K/V と画像の RoPE (フレームは文の長さ) とキーの数を持つ
- 正しさ (512²、12 ステップ、CPU の float32 と同じ雑音): Qwen-Image-2.1 CFG 4、空の負のプロンプト 潜在のコサイン 0.9982 / PSNR 30.4 dB。Chroma1-HD CFG 3 0.9957 / 34.1 dB
- **参照の誤り**: 最初 Chroma1-HD は 0.71 と合わなかった。`ref_chroma.py --mask keys` の SDPA の差し替えを T5 を回す前に入れていたので、T5 の SDPA の float のマスク (相対位置のバイアスとマスクの和) まで 0/−inf のキーのマスクに変わり、バイアスが消えていた。1 トークンの負のプロンプトでは DiT が見るパディングの行が大きくずれる。差し替えを T5 の後にして直した。NPU の T5 は transformers の float32 と、そのような短いプロンプトでも行 0 で 0.999995、パディングの行で 0.993
- **解き方の既定** (NPU、512²、100 ステップのオイラー法と比べた潜在のコサイン / PSNR):

| | Qwen-Image-2.1 (CFG なし、狐 / 看板) | Chroma1-HD (CFG 3、狐) | 時間 (Chroma) |
|---|---|---|---|
| オイラー法 40 (カードの既定) | 0.990 / 24.4 dB、0.998 / 28.6 dB | 0.972 / 24.8 dB | 7 分 19 秒 |
| DPM-Solver++ 2M 25 | 0.973 / 20.2 dB、0.993 / 23.6 dB | 0.961 / 23.0 dB | 4 分 36 秒 |
| DPM-Solver++ 2M 20 | 0.961、0.989 | | |
| DPM-Solver++ 2M 25、CFG は σ ≥ 0.6 だけ | | 0.962 / 23.2 dB | **3 分 59 秒 (−45 %)** |

  - Qwen-Image-2.1 はオイラー法 40 を既定のまま (DPM 25 は構図は同じだが細部が Anima より動く)。Chroma1-HD は DPM 25 + CFG σ ≥ 0.6 を既定に (model.json の `sampler`、`steps`、`cfg_sigmas`)。絵は 4 つとも見分けられない。σ ≥ 0.6 だけにしても変わらない (0.9624 と 0.9608) のは Anima と同じ。プロンプト 1 つの判断
- **文の符号器を組ごとに読む**: Qwen3-VL-8B の 36 層を 4 層ずつ 9 組の 1 つの ELF にし、1 つのバンクに組を順に読む (SwapVariants)。文はプロンプトごとに 1 回なので重みを読むのは 1 回のまま。txt_in は別の小さなグラフ (`text_out`)、RoPE はホストの入力。5.6 GB の q6 を一度に固定しなくなり、512² のピーク 6.8 → **2.1 GB**、潜在は同じ (1.0)
- **VAE の先読みに上限** (`anima.VAE.PrefetchLimit`、Qwen-Image と Chroma は 3 GiB): 段を前の段の実行中に読むのは、2 つで 3 GiB 以下のときだけ。1024² の Qwen-Image は段 4 と段 5 (計 5.9 GiB) を同時に持っていた。ピーク 6.4 → **4.2 GB**、VAE は 8.5 → 12 秒、画素は同じ
### 16.158 Anima-Base を models.yaml から外す — 2026-11-11

- 16.148 で足した Anima-Base v1.0 (`anima:base`) を models.yaml から外した。`-m anima:base` は「family anima has no tag "base"」と断る (`TestImageMemoryTags`)
- 理由: Base の取り柄は、調整前の基盤モデルで LoRA の学習のもとになること。hwnpu は LoRA を読み込まない (`internal/imagegen/anima/mil` の `ditLoRA` は adaLN の低ランクの層の幅で、LoRA の読み込みではない) ので、Base を走らせる場面がない。絵を描くなら、同じ構造、同じ時間の Anima-Aesthetic で足りる。Base は 512² で NPU で描けることを確かめただけで、参照との照合もしていなかった (16.152)
- 残したもの: `circlestone-labs/Anima-Base-v1.0-Diffusers` のリポジトリ (VAE、トークナイザー、設定。Turbo と Aesthetic が使う)、名前の頭 `net.` の見分け (`iron_anima.py`、`ref_anima.py`、`mil.OpenWeights`。プレビューのチェックポイントも `net.`)。`iron_anima.py --dit anima-base-v1.0.safetensors` の書き出しはディレクトリを `-m` に渡せば今も走る
- 過去の記録 (16.148、16.152、16.154) の Anima-Base の記述はそのまま残す

### 16.159 ダウンロード全体の残り時間と、最後の行の時間の内訳 (ダウンロード、変換・ロード・コンパイル、処理) — 2026-11-11

ファイルは 1 つずつ落とす。並列にすると全部のバーが見えるが、各バーの残り時間は今の分け前の速さで計算するため、小さいファイルが終わって残りが速くなる分を見込めない。例: 4 GB と 1 GB を合わせて 100 MB/s で並列に落とすと、表示は 80 秒と 20 秒になる。実際は 50 秒で終わり、1 つずつ落としたときと同じになる。正しい待ち時間は、1 つずつのまま、全体の残りのバイト数と今の速さから出せる。

- **大きさを先に集める** (`chat.Expect`、`internal/chat/plan.go`): まだないファイルの大きさを、8 本並列の HEAD で集める。Hugging Face の `resolve` は LFS や xet のファイルに CDN へのリダイレクト (302) を返し、その `Content-Length` はリダイレクトの本文の長さになる。そのためリダイレクトは追わず、`X-Linked-Size` を読む (小さいファイルの 307 にも付く)。path の取得元は stat で読む。s3 と、サインインの要るリポジトリは大きさがわからないので、合計から外す。`.part` は済んだ分として数える
- **バー**: 一覧に入ったファイルは `1.20 / 4.00 GB, all 3.10 / 9.40 GB  85.3 MB/s, 1m26s left` と出る。残り時間は全体の残り ÷ 今のファイルの速さ。一覧のファイルが全部そろうと、次の `Expect` は 0 から数え直す
- **呼ぶところ**: `FetchWeights` (分割された checkpoint の shard: チャット、文字起こし、TTS、systemone)、Anima、Sana、Magenta RealTime、MOSS-SoundEffect。config.json や index.json のように中身を見て初めて次のファイルが決まるもの、トークナイザーのような小さいファイルは一覧に入れない
- **最後の行** (`console.Times`、`internal/console/phases.go`): `took 3m12s: downloading 2m40s, converting, loading and compiling 25s, running 6.2s`。ダウンロードがなければその項目は出さない
  - 処理を始めるところでコマンドが `console.Running()` を呼ぶ。最後の行は、それを呼んだコマンドだけが出す。会話やサーバー (`hwnpu-chat` の対話、`serve`、`systemone --listen`) は出さない
  - 処理の途中の変換・ロード・コンパイルは、準備の側に数える。そのため `iron.OpenGraphWith`、`runner.Open`、大きさごとに変換する深度推定 (DA3、MoGe-2)、OCR、YuNet、SigLIP の塔の読み込みが `console.Loading()` で囲んである。Sana と Anima が段ごとに読み込む時間もこれで準備の側に入る。別のモデルを読み込み直すコマンド (MOSS-SoundEffect、Magenta RealTime) は `console.Preparing()` で準備の側に戻す
  - ダウンロードの時間は `fetch` の中で測る (重なったダウンロードは 1 回に数える)。サインインを待つ時間とハッシュの確認の時間も含む
  - 準備と処理が重なる時間 (ロードと同時に走る処理) は、準備の側に数える。Core ML が最初の呼び出しで行うコンパイル、再生の時間 (`--play`) は処理の側に入る
- **hwnpu-chat の読み込みのログ**: 分割したプログラムのバーは `.parts.json` の名前を出していた。そのため、Core ML のコンパイルの時間 (plamo-3-translate-8b の 5 部分で 37 秒) が JSON の処理に見えた (中身は 32 KB の部分の一覧で、読むのは一瞬)。バーはモデル名、時間、1 回に読むトークン数、部分の数、配置にした (`[1/2] Qwen3-0.6B 15s, 8 tokens per call, 1 part, all on the NPU`)。全部読み終えたら、hwnpu-image と同じく `loaded Qwen3-0.6B in 34s, 2 programs, all on the NPU` を出す。続けて、会話やサーバーにも、それまでの内訳 `ready after 35s: converting, loading and compiling 35s` を出す (`console.Prepared`)。Qwen3-0.6B の 2 回目の読み込みは 2.7 秒で、初回の時間のほとんどは Core ML の最初のコンパイル
- **`HWNPU_PROFILE=1` をすべてのコマンドに** (`console.Profile`、`internal/console/profile.go`): これまでは hwnpu-chat が答えのたびに、デコードの 1 ステップの内訳と Neural Engine が抱えるメモリを出すだけだった。今は hwnpu-audio、hwnpu-image、hwnpu-chat、hwnpu-dev の最後に次を出す (`console.Done`、`console.WriteProfile`)
  - ダウンロード 1 件ずつの時間、バイト数、速さ、`REPO@REV/FILE`
  - 先に大きさを集めた時間
  - プログラムの読み込み 1 件ずつの時間とパス。中で読み込んだものは字下げして、始まった順に並べる。読み込みを記録するのは `iron.OpenGraphWith`、`runner.Open`、`chain.Open` / `OpenOn` / `Share`、深度推定・MoGe-2 の大きさ、YuNet、OCR、SigLIP の塔。`runner.Open` の中の同じパスの `chain.Open` は 1 回に数える
  - `decode.WriteProfile` (`console.OnProfile` で登録。チャット以外のコマンドの言語モデルにも出る)
  - 詳細を消さない: チャットのバーは、プロファイルのときだけファイル名も残す (`[1/2] Qwen3-0.6B (qwen3-lut8-c4096-q8.mlpackage) 0.5s, 8 tokens per call, ...`)
  - 実機: swift-f0 の初回の取得は、プロファイルで 1.68 秒、バーで 0.2 秒だった。差は最初のバイトまでの待ちで、curl でも Hub のリダイレクトまで 0.2 秒、CDN の最初のバイトまで 0.8 秒。バーは最初のバイトから始まるので、この待ちを含まない。DA3-BASE の深度推定では、処理の途中で 504x322 を読み込んだ 0.26 秒 (vit 0.19 秒、head 0.08 秒) が準備の側に入った
- テスト: `TestExpect` (偽のハブ、302 と `X-Linked-Size`、200 と長さ、`.part` の続き、1 つずつ落としたときの合計)、`TestPhases` (重なったダウンロード、入れ子の Loading)、`TestProfile` (一覧、字下げ、同じパスの 2 重、リンクは載せない)。実機: swift-f0 の初回 `took 3.9s: downloading 1.8s, converting, loading and compiling 2.1s, running 0.1s`、2 回目 `took 0.1s: converting, loading and compiling 0.0s, running 0.1s`。Hub の HEAD で Qwen3-0.6B の model.safetensors (1503300328) と config.json (726)、Qwen3-4B の shard の大きさが返った

### 16.160 Sana を new-model-checklist の 7.2 節にそろえる: CFG を雑音の多いステップだけに、Mac の CFG を組ごとに、DiT を 8 ビットに — 2026-11-11

[new-model-checklist.md](new-model-checklist.md) の 7.2 節で画像のモデルごとに済んだ手法を並べると、Sana (600M、SANA 1.5) には Anima に入れた手がいくつか入っていなかった。M1 (16 GB、AC 電源、低電力モードなし、ほかに Neural Engine を使うプロセスなし) でできるものを入れた。AMD の分は別の作業で入れる。

**測り方**: 1024 × 1024、3 プロンプト (人物、夜の街、風景) × 種 1、2 の 6 枚。比べる相手は同じ Neural Engine の、全部のステップで CFG をかけたもの (fp16 の DiT)。最後の潜在のコサインと画像の PSNR、絵を並べる。時間はキャッシュに当たる新しいプロセスで。NPUWired は `mem.NPUWired` を 0.1 秒ごとに読んだ最大、footprint は `/usr/bin/time -l` の peak memory footprint。

**CFG を雑音の多いステップだけに** (`sana.Pipeline.CFGMin` / `CFGMax`、`DefaultCFGSigmas`、`--cfg-sigmas`): Anima の 16.154 と同じ。外したステップは DiT を条件つきで 1 回。Sana の 20 ステップ (shift 3) では σ ≥ 0.6 が 14 ステップ、σ ≥ 0.5 が 15、σ ≥ 0.8 が 9。

| | ステップ (CFG のあるもの) | Sana 600M のノイズ除去 | 全ステップ CFG とのコサイン / PSNR (6 枚) | SANA 1.5 のコサイン | 絵 |
|---|---|---|---|---|---|
| 全ステップ (前) | 20 | 14 秒 | — | — | — |
| σ ≥ 0.5 | 15 | 13 秒 | 0.9973〜0.9989 / 29〜34 dB | 0.9949〜0.9971 | 同じ |
| **σ ≥ 0.6 (既定)** | 14 | 12 秒 | 0.9956〜0.9984 / 28〜33 dB | 0.9921〜0.9958 | 同じ |
| σ ≥ 0.8 | 9 | 10 秒 | 0.972〜0.990 / 21〜24 dB | — | 風景で右の小さな舟が消える |

- 既定は Anima と同じ σ ≥ 0.6。`--cfg-sigmas 0,1` で全ステップ。パイプラインは IRON と Core ML で共通なので AMD でも既定で効くが、AMD では未確認
- SANA 1.5 は CFG を減らしてもノイズ除去が 38 → 37 秒しか縮まなかった。下の載せ直しが律速だったため (DiT を 8 ビットにした後は 1 ステップ 1.43 秒 → CFG なしのステップ 0.72 秒で、回数どおりに効く)

**Mac でも CFG の 2 回を重みの組ごとに** (`progEngine.velocity`): 前は条件ごとに 7 群を全部回していた (1 ステップ 14 回の群の呼び出しが、条件なし → 条件つきで 7 群ずつ)。AMD (16.148) と Anima の Core ML と同じく、群ごとに 2 回続けて回す。

| SANA 1.5 (fp16 の DiT、全ステップ CFG) | ノイズ除去 | 1 ステップ | NPUWired |
|---|---|---|---|
| 前の順 | 64 秒 | 2.45〜3.21 秒 (ばらつく) | 2.96〜3.08 GiB |
| 群ごと | 38 秒 | 1.60〜1.84 秒 | 2.98〜3.06 GiB |

- 絵はビット単位で同じ (潜在のコサイン 1.0000、PSNR ∞)。Neural Engine が抱えるのは約 3.0 GiB で、M1 の約 2.9 GiB (16.109、16.118) を超え、群が呼び出しのたびに載せ直されていた。群ごとに回すと載せ直しが 1 ステップ 1 回になる。16.152 の「1 ステップ 2.5〜3.4 秒」はこれ
- Sana 600M (1.14 GiB) は変わらない (20 ステップ 14 秒)

**DiT を 8 ビットに** (`sana.Formats`、モデルの名前で既定を決める):

| | DiT | 1 枚 (ノイズ除去) | NPUWired | fp16 の DiT とのコサイン / PSNR | 絵 |
|---|---|---|---|---|---|
| Sana 600M | fp16 (前) | 18 秒 (12 秒) | 1.14 GiB | — | — |
| Sana 600M | **int8 (既定)** | 17 秒 (11 秒) | 0.60 GiB | 0.964〜0.9997 / 22〜37 dB | 同じ |
| SANA 1.5 | fp16 (前) | 43 秒 (37 秒) | 2.98 GiB | — | — |
| SANA 1.5 | int8 | 33 秒 (27 秒) | 1.53 GiB | 0.58〜0.79 / 12〜14 dB | 崩れる (筆の跡のような模様) |
| SANA 1.5 | **lut8 (既定)** | 31 秒 (25 秒) | 1.56 GiB | 0.991〜0.999 / 26〜35 dB | 同じ |

(どれも σ ≥ 0.6 の既定の CFG)

- SANA 1.5 の行ごとの int8 は全部の絵が崩れた。lut8 (256 段の表で量子化の段を重みに合わせる) は崩れない。原因は追っていない (外れ値の大きな行が int8 の刻みで潰れているとみる。q と k の正規化の前の射影など)。Sana-Sprint 1.6B の int8 (16.145) は 0.987 で崩れていなかったので、同じ 1.6B の構造でも SANA 1.5 の重みに限る
- SANA 1.5 は NPUWired が M1 の上限の下に入り、載せ直しがなくなった。footprint の最大は 3.43〜3.85 GB で、models.yaml の Apple のメモリ 3.6 → 3.9
- 書き出しの名前に形式が入る (`sana-600m-1024x1024-int8-int8-g4-v4`、`sana1.5-1.6b-1024x1024-int8-lut8-g4-v4`) ので、`ExportVersion` は上げない。初回は変換とコンパイル (600M 約 7 分、1.5 約 12 分)。前の fp16 の書き出し (`-int8-fp16-`) は使われなくなる
- 時間の合計: Sana 600M 22 → 17 秒、SANA 1.5 70 秒 (16.152) → 31 秒

**Anima-Aesthetic を M1 で** (16.148、16.154 の既定、AMD でしか測っていなかった): 1024 × 1024、int8 / int8 の書き出し (`anima-aesthetic-v1.1-1024x1024-int8-int8-g4-q512-v3`)、3 プロンプト × 2 つのノイズ。

| | 1 枚 (2 回目から) | ノイズ除去 | 1 ステップ | NPUWired | footprint の最大 |
|---|---|---|---|---|---|
| **既定** (DPM-Solver++ 2M 25 ステップ、CFG は σ ≥ 0.6 の 17 ステップ) | 4m30s (6 枚とも) | 4m23s | CFG あり 12.5 s、なし 6.4 s | 2.62〜2.64 GiB | 3.72〜3.89 GB |
| オイラー法 30 ステップ、全ステップ CFG (モデルカード) | 約 6m25s | 6m14s〜6m17s | 12.5 s | 2.61〜2.64 GiB | 3.82〜4.28 GB |

- 初回 (取得、変換、コンパイル) 21m44s。テキスト 0.2 秒、VAE 2.8〜3.0 秒。Anima-Turbo (16.152、58 秒) の約 4.6 倍で、DiT 1 回は Turbo と同じ 6.3 秒
- 既定とオイラー法 30 ステップの比べ (同じノイズ、4 組。残り 2 組は打ち切った): 少女と桜の 2 組は潜在のコサイン 0.981 / 0.987 で同じ絵。黒猫と窓辺の 2 組は 0.833 / 0.960 で、1 組は人物が、もう 1 組は猫の位置が変わった。AMD の 512² の 6 組 (オイラー法 30 と 0.982〜0.983) よりばらつくが、どちらが正しいかはこの比べでは決まらない (真値の代わりの多いステップのオイラー法や diffusers の fp32 とは比べていない)。既定はそのまま
- diffusers の fp32 との照合は 1024² のオイラー法 30 ステップと CFG で CPU 1 枚約 1 時間かかるので見送った (Mac の Anima の部品は Turbo で照合済み、16.152)

**Mac でできなかったこと**: 電力量 (`hwnpu-dev powerbench` は sudo が要る)。AMD の Sana の DiT の q8、交差注意のキーと値をステップの外へ、Anima の VAE の段をまとめる、AMD での CFG のステップの確認は別の作業で。
### 16.161 RTMDet-m を AMD の NPU で: ONNX の CNN を IRON のグラフにする変換器 — 2026-11-11

- 動機: 19 節 (amd-npu-models.md) で CPU に置いた ONNX の 5 つのモデルのうち、計算の大きい RTMDet-m (39 GMAC) は NPU で CPU に勝てる見込みがあった (YuNet は 0.1 GFLOP で負けた)。演算の切り替え (1 回約 250 µs) が時間を決めるので、畳み込み 1 つを行列積 1 回 (バイアスは 1 のチャネル、SiLU は行列積の後段) にする
- `tools/amd/iron_onnxcnn.py` (amd-npu-models.md 21 節): 畳み込み、深さ方向、チャネル注意、SPPF の max pool、2 倍の拡大、足し算、連結。RTMDet-m の頭まで 150 ステップ、511 演算、13 コマンド
- 結果: 1 枚 0.33 秒 (CPU の ONNX Runtime 1 スレッド 1.43 秒、hwnpu の CPU 実行器 2.25 秒)。検出は ONNX Runtime と同じ物、スコアの差 0.013 以内、枠の差 1.3〜7.6 px。固定されるメモリ 565 MiB
- `hwnpu-image detect` は AMD の機械で、models.yaml の device が npu (RTMDet-m) なら書き出し (`~/models/rtmdet-m.amd-npu2`) を使い、なければ初回に IRON で作る (`chat.RunIRON` を `PrepareAMD` から切り出した)。IRON がなければ CPU
- tiny と s は精度が足りない: BN を畳み込んだ重みとバイアスの和が出力の 50 倍ほどで打ち消し合う。NPU で 1 演算ずつ測ると、原因は MEDIUM の bfp16 (A と B の両方、8 個で指数を共有)、行列積の後段の SiLU (vtanh は |x| ≤ 0.5 で引数をそのまま返す)、s のほぼ定数のチャネル (bf16 の刻みより小さい変化)。チャネルの並べ替えなどで tiny のスコアの差は 0.2 → 0.07 になったが m が悪くなり、全部 HIGH の tiny は CPU より遅い (amd-npu-models.md 21.4)。models.yaml で AMD では CPU に
- 電力 (balanced、PPT からアイドルを引く): 1 枚 NPU 0.325 秒・1.86 J、CPU の ONNX Runtime 1 スレッド 0.54 秒・8.2〜10.7 J、hwnpu の CPU 実行器 1.0 秒・16 J。NPU はエネルギーで 4.4〜5.8 分の 1
- 直したことと残り: amd-npu-models.md 21.2、21.3

### 16.162 RTMO-s を AMD の NPU で: Focus、Split、Transformer の層 (AIFI) を変換器に — 2026-11-11

- 動機: CLOUD_AI_COMPATIBILITY.md の画像の順番で、RTMDet の次に ONNX の汎用の道に載るもの。RTMO-s の骨格は YOLOX 型で、首に Transformer の層が 1 つある
- `iron_onnxcnn.py` に Focus (6 × 6、stride 2 の畳み込みに組み替え)、Split、group の畳み込み、AIFI (32 次元の頭を 64 に広げた IRON の MHA、LayerNorm、erf の GELU) を足した。頭の畳み込み (`ironcnn.LastConvs`) までを NPU、残りはホストの評価器
- 精度: MEDIUM では関節のスコアが 0.13、確信の低い人の枠が 97 px ずれた。HIGH と SiLU の演算で、関節の位置 2 px、スコア 0.03 以内。1 枚 0.22 秒 (MEDIUM なら 0.15 秒)
- 書き出しの読み書きと実行を `internal/vision/ironcnn` にまとめ、detect と pose で共有。`hwnpu-image pose` は AMD の機械で初回に書き出す (models.yaml の `device: {amd: npu}`)
- 詳しくは amd-npu-models.md 22 節

### 16.163 点や箱で指定する切り抜き (EfficientSAM): `hwnpu-image segment` — 2026-11-11

- 動機: CLOUD_AI_COMPATIBILITY.md の画像の 2 番目。BiRefNet は deformable conv があるので、素の ViT の EfficientSAM から
- 符号器 (1024 × 1024 の ViT と首) を `tools/amd/iron_efficientsam.py` で IRON のグラフ 1 つに (64 次元の頭なので IRON の MHA にそのまま載る)。初回に書き出す (models.yaml の `device: {amd: npu}`、`internal/vision/segment`)
- 復号器 (プロンプトの符号化と mask の復号) は、作者の ONNX をホストの評価器で。評価器に If、ScatterND、ConvTranspose、InstanceNormalization、LayerNormalization、Resize (cubic まで) を足し、大きな行列積は sgemm の float32 に。ONNX の参照評価器と IoU の予測 1e-6、mask の符号は全画素で一致、1 回 0.6〜0.7 秒
- 符号器の CPU の経路: ONNX Runtime があればそれ、なければ hwnpu の評価器 (Ti で 30 秒、外部依存なし)
- 結果: Ti の符号器 0.34 秒、S 0.48 秒 (CPU の ONNX Runtime 1.9 秒と 4.5 秒)。NPU と CPU の mask の IoU 0.93〜1.0 (既定で書く一番スコアの高い mask は 0.989 以上)
- 詳しくは amd-npu-models.md 23 節

### 16.164 PLaMo 3 Translate 8B (`plamo-3-translate-8b`): 8B は 4 ビットの表で — 2026-11-11

- pfnet/plamo-3-translate-8b は 2B と同じ `Plamo3ForCausalLM` で、大きさだけが違う (隠れ層 4096、中間 16384、32 層、注意のヘッド 32 に KV ヘッド 4、語彙 107520、出力の行列は埋め込みと別、全注意は 7、15、23、31)。トークナイザーとテンプレートも同じなので、`internal/llm/plamo3` は config から形を決め、変換はそのまま通った。ダウンロードは 17.1 GB
- 見積もり (`llm.EstimateBytes`): Neural Engine の重みは lut8 7.58 GiB、lut6 5.66、lut4 3.77。どれも M1 が全速で動かせる 2.8 GiB (PROBE.md) を超える。2 本 (8 トークンと 128 トークン) は 2 倍になるので、hwnpu-chat は 1 本 (`onePrograms`)。読み込みの前に M1 での遅さを警告する (前からある `neuralEngineHold` の警告)
- M1 (16 GB、macOS 27.2) で hwnpu-chat、3 つの形式 (どれも 1 本、`all on the NPU`)。プロンプトは 25 トークン前後、生成は貪欲:

| 形式 | 部分 | 変換 | 初回のコンパイル | 2 回目以降の読み込み | Neural Engine | プロンプト | 生成 |
|---|---|---|---|---|---|---|---|
| lut8 | 9 (0.62〜0.92 GiB) | 約 1 分 | 1 分 54 秒 | 10 秒 | 7.72 GiB | 7〜10 秒 | 0.4 トークン/秒 |
| lut6 | 9 (0.35〜0.84 GiB) | 約 1 分 | 4 分 04 秒 | — | — | 平均 6.5 秒 | 0.6 |
| **lut4** | 5 (0.31〜0.89 GiB) | 約 1 分 | 1 分 16 秒 | **3.2〜5.1 秒** | **3.81 GiB** | 4.6〜5.6 秒 | **0.8** |

  - 7.58 GiB の lut8 でも 2 回目の読み込みは 10 秒で、コンパイルのキャッシュに残った (ほかのプロセスが Neural Engine を使っていない間)。16.41 の「約 4 GiB」は、M1 のこの形には当てはまらなかった。lut4 を読んだ後の lut8 は 25 秒 (一部が外れた)
  - 遅さは重みの量どおり: lut8 と lut4 で 2 倍。M1 で 2.8 GiB を超えると全部の呼び出しが遅くなる (16.109) ので、M1 では lut4 でも実用にならない。M2〜M4 は限度がない (PROBE.md) ので、lut4 の 3.77 GiB を 63 GB/s で読めば 1 トークン約 65 ms (15 トークン/秒) の見込み。未計測
  - プロセスの footprint の最大は lut8 で 3.3 GB、lut4 で 4.1 GB (読み込みの間)。変換を含む初回は 3.4 GB で、チェックポイント (17 GB) をメモリに読み切らない (`safetensors` はテンソルごとに読む)
- **M2** (MacBook Air M2 16 GB、lut4、2026-11-11。M1 と M3 も 16 GB): 変換は 1 部分 7〜8 秒、初回のコンパイル 1 分 09 秒。2 回目以降の読み込みは **35〜41 秒**で、M1 (3〜5 秒) と違いコンパイルのキャッシュに一部しか残らない。読み込み直後に Neural Engine が持つのは 2.10 GiB と 1.20 GiB で、それぞれ part3〜5 と part4〜5 の大きさにちょうど合う (残る部分の数が回ごとに変わる)。生成は **9.4〜10.1 トークン/秒** (M1 の 12 倍)、プロンプト 25 トークンは 0.8〜1.5 秒
  - `HWNPU_PROFILE=1` (q=8、12 回の平均、1 回 195.6 ms): part0 50.9 ms (0.79 GiB、16 GiB/s)、part1 48.0、part2 49.8 (0.89 GiB、18〜19 GiB/s)、part3 26.1 (0.89 GiB、34 GiB/s)、part4 7.0 (0.31 GiB、44 GiB/s)、注意 4 つで 13.8 ms。遅い 3 つはこの起動でコンパイルし直した部分で、最初の呼び出しで Neural Engine に載る時間が平均に入る。2 文続けて訳すと (プロフィールは起動からの累計なので差で)、2 文目の 12 回は part0 20.6 ms、part1 22.8、part2 23.1、part3 23.2、part4 6.3 で、1 回 105.7 ms。どの部分も 38 GiB/s 前後で、これが M2 の Neural Engine が重みを読む速さの上限とみる (3.77 GiB で 1 トークン約 100 ms。見込みの 63 GB/s には届かない)。生成の 9.1 トークン/秒はこの 1 回 1 トークンどおり: 載る時間は主にプロンプトを読む回にかかるので、生成の速さは 1 文目から変わらない。下書き (7 個) は 14 回で 1 つも当たらない (訳は入力と言語が違い、プロンプトから写せない)
- **M3** (MacBook Air M3 16 GB、lut4、2026-11-11): ダウンロード 17.1 GB に約 7 分、変換は 1 部分 4〜7 秒、初回のコンパイル 1 分 04 秒。2 回目と 3 回目の読み込みは **4.6 秒と 3.6 秒**で、Neural Engine は 3.78 GiB を全部持つ (キャッシュに全部残る)。生成は **10.4〜10.9 トークン/秒**、プロンプト 25 トークンは 2 回目以降 0.5 秒。訳は M1、M2 と同じ
  - キャッシュに残る量はメモリ (どれも 16 GB) でなくチップか OS で違う: M1 は lut8 の 7.58 GiB まで残り、M2 は 1.2〜2.1 GiB、M3 は lut4 の 3.78 GiB が全部 (16.39 では約 4 GiB)。速さは M2 と M3 でほぼ同じ (重みを読む速さ 38〜40 GiB/s 前後)
  - lut8 (9 部分 0.62〜0.92 GiB、変換は 1 部分 5〜7 秒、初回のコンパイル 1 分 52 秒、`all on the NPU`): 20 文で生成 **5.8〜6.5 トークン/秒** (1 トークン約 165 ms、7.58 GiB を 46 GiB/s 前後)、プロンプト 25 トークン 0.5〜0.7 秒。lut4 の 9.4 トークン/秒より 1.55 倍遅いだけで、重みの量 (2 倍) ほどは差がない: lut8 のほうが 1 バイトあたり速く読める (lut4 は 38 GiB/s 前後)。2 回目以降の読み込みは測っていない (7.58 GiB は M2 のキャッシュの 1.2〜2.1 GiB を大きく超える)
  - 20 文 (lut4 と lut8): 13/20 が同じで M1 と同じ数。違う 7 文も言い換えだけ (「できないんです」と「できません」、"rescheduled for" と "moved to"、"I will" と "I'll" など)。M2 でも lut4 で訳は保たれる
- **形式** (2B が崩れた lut4 を試したのは、大きいモデルほど丸めに強いと見たため): 16.98 と同じ形の 20 文 (英 → 日 10、日 → 英 10、1 文ごとに `/reset`、`-temp 0`) を、lut8 を基準に比べた (fp16 は 15 GiB で Neural Engine に載らない。2B の lut8 は fp16 と次のトークンが 98 % 同じ)
  - lut6: 18/20 が lut8 と一字一句同じ。違う 2 文は「お持ちください」→「持って行ってください」と助詞 1 つ
  - lut4: 13/20 が同じ。違う 7 文もすべて正しい言い換え (「休館です」→「休館しています」、"moved to" → "rescheduled for"、"I'll" → "I will" など)
  - 長めの段落 4 つ (3 文ずつ、英 → 日 2、日 → 英 2): lut4 と lut8 で日 → 英はほぼ同じ (大文字 1 か所)、英 → 日は言い換え (「公共公園」→「公立公園」、「更新」→「アップデート」)
  - 2B の lut4 は CPU で KL 1.91 nats (argmax 15/31、16.71) で使えなかった。8B の lut4 は訳を保つ。量子化の誤差が 4096 の幅で平均される分、2B (2560) より効きにくいと見ている (KL は測っていない: この Mac に torch がなく、8B の fp32 は 34 GB)
  - 8B の lut8 と 0.8B の lut8 は 6/20 が同じで、8B のほうが的確 ("build time" を「製作時間」でなく「ビルド時間」、"rescheduled" を「再設定」でなく「延期」、"gently" を「さっくりと」でなく「優しく」)
- 採ったもの: **8B の既定を lut4** に (`plamo3.Config.Format`: 隠れ層 4096 以上で lut4、それより小さい 2B と 0.8B は lut8 のまま)。プログラムの名前には形式が入るので、2B と 0.8B の変換はそのまま使われる。Intel は PLaMo 3 の 2B 以下を int8 に (`Arch.IntelFormat`。lut4 の読み替えの int4 は 2B で KL 0.72)。8B は下の Intel の項
- models.yaml: `machine: 16` と `not_on: [M1]` (下の「チップ」)、メモリ Apple 8.2 GB (3.81 GiB + 4.1 GB)、AMD 6.5 GB (q4 の重みからの見積もり)、`amd_export: plamo3 --max-seq-len 8192 --q4`
- Intel: int8 の 7.6 GiB は、Core Ultra 5 135U の NPU が Anima で `OUT_OF_HOST_MEMORY` になった約 4.6 GB (16.143) を超えるので、載らない見込み。さらに Intel では hwnpu-chat は 2 本を持つ (`onePrograms`) ので、2 倍になる
  - 入らないよりはよいので、**8B の Intel の既定を行ごとの int4** にした (`llm.IntelFormatter` を足し、`plamo3.Config.IntelFormat` が隠れ層 4096 以上で int4。2B と 0.8B は `Arch.IntelFormat` の int8 のまま)。models.yaml に `prefill: off` を書き、Intel でも 1 本 (3.8 GiB) にする。Neural Engine はもともと 1 本 (`onePrograms`) なので変わらない (M1 で同じ lut4 のプログラムを 4.3 秒で読み、同じ訳)
  - 行ごとの int4 は Meteor Lake で int8 より速い (LFM3-1.7B で 37.9 ms と 68.5 ms、16.143)。2B の int4 は KL 0.72 で使えなかったが、8B は 2B をもっと崩す lut4 (KL 1.91) でも訳を保った
  - 精度の見積もり (Intel の機械なしで): 同じ int4 (行ごとのスケール、`llm.LinearFormat`) は Core ML でも組めるので、M1 の Neural Engine で 20 文を訳した (`hwnpu-chat --format int4`)。重みの丸めは Intel と同じで、計算の精度 (fp16) だけが違う。lut8 と 12/20 が一字一句同じ (lut4 は 13/20)、違う 8 文もすべて言い換え ("rescheduled" を「日程変更」、「先約がありまして」を「すでに予定が入っております」、"to conserve battery life" を "to minimize battery consumption" など)。数字、否定、固有名詞の取り違えはない。8B の int4 は lut4 と同じ程度に訳を保つとみる。Intel の実機の確認は handoff の 4 節
  - `hwnpu-dev formats -ref lut8 -formats lut4,int4` (M1、キャッシュ 4096、1 回 8 トークン、固定の会話文 265 位置): 次のトークンが lut8 と同じなのは lut4 も int4 も 187/265 (71%)、ロジットの相対誤差 (平均/最大) は 0.518/1.015 と 0.479/1.041。1 回の呼び出しは lut8 2198.8 ms、lut4 1424.2、int4 1043.7 (どれも M1 の 2.8 GiB を超えて遅い)。5.4 節の目安 (80 % 未満は使えない) を下回るが、この文は翻訳モデルへの入力ではなく、lut8 のパープレキシティも 65.9 と分布が平らで、1 位が入れ替わりやすい (翻訳モデルではパープレキシティにも意味がない、16.98)。訳の 20 文では lut4 13/20、int4 12/20 が lut8 と同じで、崩れはない
  - **訳のトークンだけの KL** (チェックリスト 9 節。HF の fp32 は 34 GB で作れないので lut8 を基準に): `hwnpu-dev generate` に `-write-ref` を足した。キャッシュなしのプログラム (`convert -seq 64`) で lut8 が 20 文を貪欲に訳し、プロンプトと答え (終わりのトークンまで) と自分のロジットを参照として書く (`check.Meta.From` が答えの始まり)。`-score` はその参照で、全位置に加えて答えのトークンを予測する位置だけの KL と argmax を出す。lut4: 答えの 235 トークンで KL 平均 0.305、argmax 207/235 (88 %)。文ごとの KL の中央値は 0.0000〜0.25 (20 文中 17 文で 0.01 未満) で、平均は言い回しの分かれ目の数トークン (最大 6.7) が押し上げる。5.4 節の 0.2〜1 の帯で、訳の質で決める側 (20 文はすべて正しい訳)。プロンプトを含む全位置では KL 1.0〜3.0、argmax 5 割前後 (原文は予測しないので意味がない)
  - int4 の同じ採点はできなかった: 64 トークンの int4 は Neural Engine で 1 つ目の部分の予測が Core ML のエラー (`Unable to compute the prediction`。8 トークンの hwnpu-chat の int4 は動く)、Core ML の CPU (`HWNPU_COREML_UNITS=cpu`) ではロジットが NaN。Intel の int4 は 20 文の訳 (上) と、Intel の実機の確認 (handoff の 4 節) で判断する
  - Core ML の int4 では出力層 (幅 107520 の `ios18.linear`) だけが CPU に置かれる (ほかの 670 演算は Neural Engine)。lut4 の同じ層は Neural Engine。Mac で int4 を使うことはないので、そのままにした
- 確かめていないこと ([handoff-plamo3-8b.md](handoff-plamo3-8b.md)): M4 の速さとコンパイルのキャッシュ、M3 の lut8、M2 の lut8 の 2 回目以降の読み込み、AMD の書き出し (ピークのメモリ、コマンドの 2 秒、デバイスのヒープ)、HF の fp32 との照合 (KL。8B の実重みでの `hwnpu-dev check`、`-f16` の層ごとの誤差、`check -package` の KL も。訳が崩れないことしか見ていない)、GPTQ の符号と lut の表の当てはめ (`--lut-fit wkmeans`。8B の fp32 の校正は約 34 GB で 16 GB の Mac ではできない)、長い文脈 (パスキー、128K と 250K)、M2 と M3 のメモリの最大
- 部品としての値 (16.94): 入力と出力は文 (PLaMo のトークナイザー)、常駐は 3.81 GiB (ANE、lut4)、2 回目の読み込み 3〜5 秒、KV は会話とともに 1 トークン 9 KiB (hwnpu-chat の表示、262144 で 2.25 GiB) と、窓 2048 のリング 28 層


### 16.165 Qwen-Image-2.1-Turbo を Mac の Neural Engine で: DiT は lut4、文の符号器と DiT の文のパスは lut8 — 2026-11-12

16.155 で AMD に入れた Qwen-Image-2.1-Turbo を Apple の Neural Engine (Core ML) に。`internal/imagegen/qwenimage/mil` (MIL のプログラム) と `qwenimage.Convert` / `Prepare` (取得と変換)、`progEngine` (実行)。`hwnpu-image generate -m qwen-image` は Mac では `--dir/Qwen-Image` に取得して変換する。M3 16 GB、macOS 27.2 で測った。

**プログラム** (どれも全部 Neural Engine):

| 部品 | 中身 | 形式 | 大きさ | 読むとき |
|---|---|---|---|---|
| text_N (9) | Qwen3-VL-8B の 4 層ずつ、512 行、因果。最後の組は DiT の txt_in まで | lut8 | 6.7 GB | プロンプトごとに 1 つずつ |
| dit_text_N (8) | DiT の 4 ブロックずつを文の 512 行に、t = 0 の変調、因果。行と各ブロックの K/V を 1 つの出力に並べる | lut8 (k-means-minmax) | 6.6 GB | プロンプトごとに 1 つずつ |
| dit_blocks_N (8) | 画像の行だけの 4 ブロックずつ。[自分の K、文の K] に注意し、文のパディングは `kbias` で隠す。最初の組に img_in、最後の組に norm_out と proj_out (小さい部品だけのプログラムは CPU に落ちる) | lut4 (k-means-minmax) | 3.4 GB | ステップの間ずっと |
| vae_sN (7) | 中段、上げのブロック 4 つ、最後のブロックを 2 段。DupUp3D の近道は 0/1 の 1 × 1 畳み込み (+ pixel_shuffle) | fp16 | 0.5 GB | 段ごとに |

- 時刻の埋め込みと共有の変調 (100 M の積和) はホストの float32 (`time.safetensors`、bf16)。3.3 節の「1 行の計算は ANE でずれる」
- RoPE は Qwen-Image の組 (2i, 2i+1) を ApplyRope の (i, i+64) に合わせて q と k の行を頭の中で並べ替える (`headPerm`、AMD の head_perm と同じ)。画像の RoPE は文の長さで変わるのでホストの入力
- 残差は DiT を 1/8、文の符号器を 1/4 で持つ (`DitScale`、`TextScale`)。float32 の参照で DiT の最大は 9000 (文の行)、文の符号器は層 6 から 13500 (最初の `<|im_start|>` の行の注意の吸い込み)
- 画像の注意のクエリの塊: 1024² で 4 ブロック 128 クエリずつ (38 MB) はコンパイル 59 秒・1 回 1.49 秒、512 (151 MB) は 33 秒・1.57 秒、全部は 35 秒・2.2 秒。コンパイルは Neural Engine のキャッシュが落とすたびにかかるので 512 (`maxScoreBytes` 160 MiB)。512² は塊なし
- 書き出し 15 GB (512²)。文の 2 つの部品は画像の大きさに依らないので、別の大きさの書き出しはハードリンクで共有する (`sharedExport`、Sana の Gemma 2 と同じ)

**参照**: torch のない Mac で float32 の参照を作るため、`tools/ref/stream_weights.py` (meta デバイスで組み、繰り返しの層の重みを実行の直前に読んで後で捨てるフック) と `ref_qwenimage21.py --stream` を足した。7B と 8B の float32 が数 GB で回る (512² の 1 枚 約 5 分)。各層の出力の最大も出す。`tools/ref/compare_image.py` は潜在のコサインと PSNR。

**文の符号器の行ごとの int8 は注意の吸い込みで NaN**: 最初の形 (文 int8) は参照と潜在のコサイン 0.698 で、狐が灰色になった。CPU のインタプリタでは同じ int8 のプログラムが文の行 0.9995 なので、グラフでなく Neural Engine。組ごと、層ごとに CPU と比べると、層 6 だけ 1744 個の非有限の値 (行 0)。層 6 の down_proj は行 0 で 13120 を出す (入力は 6409 で fp16 に収まる)。Neural Engine は行ごとの int8 の重みで符号のままの積和を取って最後に行の尺度を掛けるとみられ、その途中が fp16 を超える。lut8 (表を fp16 の重みに戻してから積和) では非有限なし、各行 0.9996 以上。SANA 1.5 の int8 の DiT が崩れたこと (16.160、原因は追っていなかった) も同じかもしれない (未確認)

**形式** (512² の狐、同じノイズの diffusers の float32 との潜在のコサイン / PSNR):

| 文の符号器 | DiT の文のパス | DiT の画像 | コサイン / PSNR | 1 ステップ |
|---|---|---|---|---|
| int8 | lut4 (一様) | lut4 (一様) | 0.698 / 11.1 dB (色が違う) | 2.2 秒 |
| lut8 | lut4 (k-means-minmax) | lut4 (k-means-minmax) | 0.965 / 21.1 dB | 2.3 秒 |
| **lut8** | **lut8** | **lut4 (既定)** | **0.980 / 22.5 dB** | 2.3 秒 |
| lut8 | lut8 | lut6 | 0.987 / 23.0 dB | 3.2 秒 |

- 4 ブロックの組 1 つを fp16 と比べた相対 RMS 誤差 (最初のステップの実の入力): lut4 一様 4.9 %、k-means 12.6 %、**k-means-minmax 3.2 %**、lut6 1.2 %、lut8 0.5 %。一部の射影だけ lut8 にしても 4.1〜4.8 % (誤差は全部の射影に広がっている)
- DiT の lut4 の 0.969 (最初のステップの速度) は CPU のインタプリタでも 0.969 で、fp16 のずれではなく量子化
- 文のパスはプロンプトごとに 1 回でステップの間は持たないので lut8 にしても Neural Engine が持つ量は増えない。画像の lut6 は 5.1 GB で 40 % 遅く、コンパイルのキャッシュを超えるので `--dit-format lut6` に残した
- 3 プロンプト (狐、「OPEN 24 HOURS」の看板、雨の夜の街) × 2 種、512²: 0.935〜0.981 / 13.9〜23.8 dB。6 枚とも構図は参照と同じで、看板の文字も同じ。看板の 2 枚 (0.935、0.945) は看板の位置が少しずれた

**速さ**:

| | 512² | 1024² |
|---|---|---|
| 1 ステップ (計算) | 2.2 秒 | 16 秒 (4 ブロックの組 1.57 秒 × 8 + ホスト) |
| 初回 (変換とコンパイル込み) | 5〜6 分 (変換 68 秒) | 11〜12 分 |
| 2 回目から 1 枚 | 4 分 26 秒 | 7 分 08 秒 |
| footprint の最大 | 4.5 GB | 6.0 GB |

- **時間の大半は読み込み**: プログラムの合計 17 GB が Neural Engine のコンパイル済みキャッシュ (約 4 GiB、16.41) を超えるので、2 回目でもほぼ全部が外れる (1 本 7〜12 秒、1024² の DiT は 17〜32 秒)。512² の計算は 1 枚 約 20 秒
- **複数のプロンプトを 1 回で**: Core ML の書き出しでは、全部のプロンプトを先に符号化し (文の部品を読むのは 1 回)、DiT を持ったまま全部のステップ、VAE は段ごとに全部の絵 (`Pipeline.Batched`、`DecodeAll`)。2 枚が 5 分 02 秒 (別々なら 8 分 52 秒)
- 効かなかった: 次のプログラムを今のものの実行中に読む、DiT を 4 本同時に読む。単独の試験では 4 本同時が 71 → 56 秒だったが、パイプラインでは 62 → 76 秒、footprint 4.5 → 7.4 GB で戻した (Neural Engine のコンパイルはほぼ 1 本ずつ)

**Qwen-Image-2.1 (蒸留なし)** も同じ形で動く (DiT の設定は Turbo と同じで、違うのはスケジューラと VAE の重み)。512²、40 ステップ、M3:

| | 時間 | 1 ステップ | footprint の最大 |
|---|---|---|---|
| 初回 (取得 15.6 GB 約 5 分、変換 78 秒を含む) | 15 分 (生成は 8 分 31 秒) | 2.2〜2.3 秒 | 4.5 GB |
| CFG 4 (`--guidance 4 --negative ...`)、2 回目 | 5 分 45 秒 (文 81 秒、ステップ 4 分 15 秒) | 4.6 秒 | — |

- 同じノイズの diffusers の float32 (CFG なし) と潜在のコサイン 0.970、PSNR 22.3 dB。構図は同じで、毛の色が少し灰色寄り。Turbo の 6 枚 (0.935〜0.981) と同じ程度で、40 ステップでもずれは溜まらなかった。参照は層ごとに重みを読む形で 79 分
- **別のモデルとは文の符号器の組だけ共有する**: 最初の `sharedExport` は形式だけを見ていて、蒸留なしの書き出しに Turbo の DiT の文のパスを、ハードリンクしようとした (重みが違うので絵が狂う)。txt_in を持つ文の符号器の最後の組も DiT の重み。書き出しの名前からモデルを取り、同じモデルなら前のとおり全部、別のモデルなら最後の組の前までを共有する (`exportModel`、`TestShareText`)
- CFG の 2 回の試しは、プログラムの読み込み (Neural Engine のコンパイル) の途中で止まったまま 2 時間と 90 分たって打ち切られた (`dit_text_12`、`dit_blocks_24`)。3 回目は同じ設定で最後まで通った。再現せず原因は分かっていない (読み込みに上限の時間はない)

**残り**: 読み込みが止まる件、1024² の参照との照合 (ストリームの参照で 1 枚 30 分以上)、`mem.NPUWired` の値、電力量。ディスク: 取得 32 GB + 書き出し 15 GB + コンパイルのキャッシュ (`~/Library/Caches/hwnpu-image/com.apple.e5rt.e5bundlecache` が数 GB ずつ増える)

### 16.166 Intel の経路のなかったモデルを OpenVINO IR に: 変換と、OpenVINO の CPU での確認 — 2026-11-12

Intel の NPU に向けた変換 (OpenVINO IR、`internal/ovlower`) がない、または壊れていたモデルを洗い出し、Go の変換器があるものはすべて IR を書けるようにした。Intel の実機はないので、確かめたのは「変換が通る」「IR が MIL と同じ値を出す」「コマンドが変換から結果まで通り、結果が Apple / AMD / CPU の経路と合う」まで。NPU のコンパイラが受け付けるか、速さ、メモリは未確認 (`tools/intel/hwnpu-intel-check.ps1` の full に足した)。

- **確かめ方**: `HWNPU_BACKEND=openvino` で、コマンドは機械を Intel のものとして扱う (`chat.HereVendor`)。Intel 向けの IR を書き、OpenVINO の CPU プラグイン (`internal/backend/openvino`) で動かす。Intel の NPU の Level Zero との違いはコンパイラと実行の装置だけで、変換、引数の名前、ホストの処理は同じ。今までは `--backend` を持つ hwnpu-chat と Anima だけがこれで確かめられ、AMD の機械では hwnpu-image と hwnpu-audio が AMD の経路に入っていた
- **見つかった不具合**: `llm.OpenDevice` の 2 番目の戻り値はプログラムの形式 (`intel`) なのに、pose、detect、ocr、pitch、notes はそれを `llm.ProgramBackend` にもう一度通し、`""` になって「Core ML も IR も動かない装置」として止まっていた。この 5 つは Intel の NPU では一度も動かなかったはず。`ProgramBackend` が形式名も受けるようにした (`TestProgramBackend`)
- **ovlower に足した演算** (MIL → IR): `clip` → Clamp、`max_pool` → MaxPool (opset1、明示のパディング)、`pow` → Power、`pixel_shuffle` → DepthToSpace (`depth_first`、PyTorch の並び)、`conv_transpose` → ConvolutionBackpropData (MIL の切り落とし `pad` を pads_begin/end に、出力が全長より長い分は `output_padding` に。groups は milcpu と同じく 1 だけ)、`elu` → Elu、`cos`、`sin`、`erf`、`softplus`、`cast` → Convert。これで milcpu の演算は `read_state` (プローブだけ) を除いて全部下ろせる。`TestImageOps` が OpenVINO の CPU で milcpu と相対誤差 3e-4 以内
  - RTMDet (clip)、RTMO と PP-OCR (max_pool)、DA3 と MoGe-2 (pixel_shuffle)、Magenta RT2 (conv_transpose、elu、clip)、MOSS の DAC (conv_transpose)、YuNet (pow) がこれを使う。ONNX の変換 (`internal/onnx`) を通るモデルは、今まで IR に下ろすと最初の未対応の演算で止まっていた
- **新しく IR を書くようにしたもの**: どれも Core ML と同じ変換の関数に `Backend` を通しただけで、グラフは同じ。プログラムは `.xml` で、モデルのディレクトリの `intel/` (hwnpu-chat と同じ置き方) か、書き出しのディレクトリの拡張子 `.intel`
  - hwnpu-image: face-detect (YuNet。今まで「AMD と Apple だけ」と断っていた)、depth (DA3、MoGe-2。`depth.Setup.Backend`、書き出しの `format: intel`)、generate の Sana (`sana.ConvertOptions.Backend`。`hwnpu-image convert --backend intel` も)、segment (EfficientSAM のエンコーダ。下)
  - hwnpu-audio: transcribe (Whisper の 3 本)、tts (LFM3-Audio の文のモデル、depthformer、コーデック)、generate の Magenta RT2 (`music.WriteProgram` が IR も書く。`llm.WriteIR` を足した) と MOSS-SoundEffect
  - hwnpu-chat systemone (d1。`systemone.Setup.Backend`、`--backend` を Load まで通す)。d1 のチェックポイントがこの機械になく、動かしていない
- **重みの形式**: Intel の NPU は表 (lut) を読まないので、`llm.IntelFormat` で読み替える (lut8 → int8)。例外は SANA 1.5 1.6B の DiT: 行ごとの int8 で絵が崩れる (16.160) ので Intel では fp16 (`sana.FormatsFor`)。MOSS の文のモデルは lut8 と同じく 5 層を fp16 に残す (`sfx.TextOptions` が int8 も)
- **EfficientSAM のエンコーダ**: 今までは AMD の NPU (IRON) か CPU だけ。ONNX のエンコーダを `internal/onnx` で MIL にするには、自分の大きさへの Resize (入力を 1024 × 1024 にする最初の段。プログラムの入力がもう 1024 × 1024 なので素通し) と LayerNormalization (MIL の layer_norm) が足りなかったので足した。ホストが ONNX と同じ双線形 (half_pixel) で 1024 × 1024 に引き伸ばしてから渡す (`segment.OpenProgram`)。Intel では既定で NPU、Apple では Neural Engine で未計測なので `--device npu` のときだけ
- **新しい道具**: `hwnpu-dev irdiff A.mlpackage ...` は、Core ML のパッケージを IR に下ろし、同じ乱数の入力で milcpu と OpenVINO の CPU の出力を比べる。Core ML の経路しかないモデルの、下ろせない演算と値の食い違いを、Intel の経路を書く前に調べられる。OpenVINO の Python パッケージは `openvino_telemetry` が別プロセスで利用の記録を送り、そのプロセスが hwnpu の消した `serve_ir.py` を読みにいって失敗していた。`serve_ir.py` と `run_ir.py` はテレメトリのモジュールを読ませない (OpenVINO は自分のスタブに切り替える)
- **テスト** (OpenVINO の CPU、`HWNPU_OV_PYTHON`): `qwen3`、`qwen35` (GDN、fp16 と int8)、`whisper` (エンコーダ、クロス、KV キャッシュ付きのデコーダ) の `TestIntelIR*` を足した。どれも MIL のキャッシュなしのプログラムと相対誤差 1e-3 以内。今までの lfm3、gemma4、lfm3vl、lfm3audio、plamo3、bert、plamo もそのまま通る
- **実物のチェックポイントで、コマンドを `HWNPU_BACKEND=openvino` で** (Ryzen AI の機械の CPU。比べた相手は同じ入力の CPU の ONNX Runtime なし実行器、AMD の NPU、milcpu):

| コマンド | モデル | 結果 |
|---|---|---|
| pose | RTMO-s | 人 1、箱とスコア 0.7085 が CPU と同じ、関節点も同じ |
| detect | RTMDet-m | 物 2、箱が CPU と 0.1 ピクセル以内、スコア 0.3474 と 0.3476 |
| ocr | PP-OCRv6 small | 2 行 "Hello Intel NPU 2026"、"OpenVINO IR check" が CPU と同じ |
| pitch | SwiftF0 | 125 フレームの有声/無声が全部同じ、f0 の差 0.013 % 以内 |
| notes | Basic Pitch | 音符が CPU と一字一句同じ |
| face-detect | YuNet | 顔 3 と 1 が AMD の NPU と箱 ±1 ピクセル、スコア差 0.002 |
| depth | DA3-SMALL | AMD の NPU と深度の相関 0.9994 (相対 RMS 2.4 %) |
| depth | MoGe-2 ViT-S | AMD の NPU と深度の相関 0.9998、点 0.9995、視野 54.5° と 54.9° |
| segment | EfficientSAM-S | AMD の NPU と同じマスク (面積 431、箱も同じ)、IoU の予測 0.882 と 0.883 |
| chat | Qwen3.5-0.8B (int8) | 「日本の首都は？」に「東京です。」 |
| tts → transcribe | LFM3-Audio-2B → Whisper large-v3-turbo (どちらも int8) | 読み上げた "Hello, this is a test of the Intel path." を Whisper がそのまま書き起こした |
| generate | Magenta RT2 small | 文 "disco funk" のスタイルの 12 符号が CPU と同じ、4 秒の音楽 |
| generate | Sana 600M 1024 × 1024 (int8) | 同じ種で AMD の NPU とほぼ同じ絵 (りんごと木の机) |
| generate -m sfx | MOSS-SoundEffect v2.0 (文のモデルと DiT は int8) | 16 本を IR に変換 (11.5 GB のダウンロード)、20 ステップで 3 秒の音 (RMS 2400、NaN なし。1 ステップ 18 秒)。中身は耳で確かめていない |

- **できなかったもの**: Chroma、SigLIP 2 (label)、FLUX は AMD の IRON のグラフ (Python) しかなく、Go の MIL のグラフがない。Intel (と Apple) で動かすには、その構造を `internal/llm/blocks` で組み直す移植が要る。Qwen-Image は 16.165 で Go のグラフ (Core ML) ができたが、IR を書くのはこの変更に入れていない
- 未確認 (実機が要る): NPU のコンパイラがこれらの演算 (DepthToSpace、ConvolutionBackpropData、Clamp、MaxPool、Elu など) と、大きなグラフ (Sana の Gemma 2 の 6 層 int8 で約 0.5 GB、EfficientSAM の 4096 トークンの注意) を受け付けるか、速さ、Core Ultra 5 135U のメモリ (Anima で 4.6 GB の上限、16.143) に Sana と MOSS が収まるか
### 16.167 Qwen-Image-2.1 の DiT の MLP を q6 に、Qwen-Image と Chroma1 の 1024² の電力量 — 2026-11-12

- **部分ごとの形式を CPU で** (`tools/ref/qwenimage_quant.py`、new-model-checklist 5.1〜5.3): DiT を torch の float32 で 1 ブロックずつ (重みは pread で読み、次のブロックを先読み、無名のメモリ 2.5 GB 以下) 回し、iron_q4 の q8、q6、q4nx に丸めた重みで、8 ステップの教師強制の速度の誤差と、float32 と同じ雑音からの自由走行の潜在のコサインを測る。float の重みで `ref_qwenimage21.py` と 1e-6、q8 は NPU の q8 の潜在とコサイン 0.9997 / 0.9996 (NPU の誤差はほぼ q8 の丸めで説明できる)
- 結果 (重みはブロックの射影のバイト、GB = 10⁹。コサインは狐 / 看板と、追加の 3 つの種):

| 注意 / MLP | GB | 速度の誤差 | 自由走行のコサイン | 追加の種 |
| q8 / q8 (前の既定) | 7.42 | 0.0048 / 0.0035 | 0.9983 / 1.0000 | 0.9997、0.9996、1.0000 |
| q6 / q8 | 6.88 | 0.0103 / 0.0082 | 0.9942 / 0.9998 | 0.9996、0.9986、0.9951 |
| **q8 / q6** | **6.21** | 0.0102 / 0.0084 | 0.9980 / 0.9998 | 0.9999、0.9965、0.9992 |
| q8 / q4 | 5.30 | 0.0314 / 0.0265 | 0.9942 / 0.9988 | 0.9918、0.9964、0.9978 |
| q8 / q4 GPTQ (別の 4 つのプロンプトで校正) | 5.30 | 0.0267 / 0.0227 | 0.9950 / 0.9980 | 0.9918、0.9888、0.9911 |
| q4 / q4 | 4.36 | 0.0389 / 0.0349 | 0.9929 / 0.9973 | 0.9913、0.9719、0.9840 |

  - 注意は 1 バイトあたりで敏感: q6 の誤差は MLP の q6 と同じほどで、減るのは 0.54 GB (MLP は 1.21 GB)。q4 の注意は看板の種で崩れる (0.97〜0.99)
  - q4nx は Anima (0.777) よりずっと強いが、細部が動く (敏感な種で 18 dB、MLP の q4 でも 25〜29 dB)
  - GPTQ は校正にないプロンプトで教師強制の誤差を 13〜15 % 下げるが、自由走行は 3 つの種のうち 2 つで丸めより悪く、看板の文字と色が変わった。評価するプロンプトで校正するとそのプロンプトだけよくなる (校正に合わせすぎ)。採らない
  - 1 つの種のコサインはばらつく (狐の種 0 は q8 でも 0.9983)。教師強制の誤差と、いくつかの種で決めた
- **採ったもの: 注意 q8、MLP q6** (`iron_qwenimage.py` の `--dit-attn` / `--dit-mlp`、`--dit` は両方を同じに)。デバイスのコードは同じ (iron_q4 の q6 は QDequant で bf16 にしてから MEDIUM の GEMM)。書き出し 15 → 14 GB
- NPU (512²、Turbo、同じ雑音で 3 つのプロンプト × 2 つの種): float32 とのコサイン 0.9946〜0.9996 で 5 枚は q8 と同じ。京都の寺の種 1 だけ 0.9940 → 0.9762 で、寺の位置が少しずれるが構図、人物、紅葉は同じ。1 ステップ 3.5 秒は q8 と同じ。1024² の狐 0.9988 (q8 0.9989、q8 とは 0.9997)
- **1024² の 1 枚の電力量** (AMD、amdgpu の PPT をパッケージ全体で積分、待機 6.8 W 込み、狐のプロンプト、既定の設定):

| モデル | 設定 | 時間 | 電力量 | 平均電力 | 固定のメモリの増え |
| Qwen-Image-2.1-Turbo | 8 ステップ | 119 秒 | 2.0 kJ | 17.0 W | 3189 MiB |
| Qwen-Image-2.1 | オイラー法 40、CFG なし | 504 秒 | 9.3 kJ | 18.5 W | 3189 MiB |
| Chroma1-Flash | Heun 8 (DiT 15 回) | 269 秒 | 4.7 kJ | 17.5 W | 4956 MiB |
| Chroma1-HD | DPM 25、CFG 3 は σ ≥ 0.6 | 691 秒 | 12.6 kJ | 18.2 W | 4956 MiB |

  - Anima-Turbo (688 J、16.148) と Anima-Aesthetic (3.8 kJ、16.154) と比べると、Qwen-Image-2.1-Turbo は 3 倍、Chroma1-Flash は 7 倍
  - Chroma の固定のメモリの山は T5 (q8 の 5 GB を 1 つのグラフで) だった。下の項で組にした後の値: Chroma1-Flash 267 秒 4.6 kJ、Chroma1-HD 685 秒 12.1 kJ
- **Chroma の T5 を組に** (`iron_chroma_parts.py` の `T5Config.group` 4、`--t5-group`): 24 層を 4 層ずつ 6 組、1 つの ELF に組ごとのバンクの像、1 つのバンクに順に読む (SwapVariants、Qwen-Image の文の符号器と同じ)。どの組のグラフも残差と最後の RMS norm の 2 つを出し、ホストは残差を次の組に渡し、最後の組の norm を使う (norm も NPU のまま)。プロンプトと負のプロンプトは組ごとに続けて回す。IRON の参照は 1 つのグラフと同じ値 (float32 の transformers と 0.999754)、NPU の潜在は前の書き出しとビットまで同じ (2 つのプロンプト)。1024² のピーク (NPU の割り当てと RSS) 5.9 GB → **4643 MiB**、T5 は 7.5〜9.4 → 4.2〜5 秒 (512² と 1024²)。models.yaml の memory を 4.6 に。一組だけでなく組の切り替えまで確かめるため irongraph に `-swap` (組 0 を回してから `-variant` の組を読む)
- **Chroma の書き出しの手順**: `hwnpu-image` が示す `python tools/amd/iron_chroma.py ROOT OUT` は DiT しか書き出さず、T5 と VAE と tokenizer.json は手で足していた。iron_chroma.py が DiT の後に iron_chroma_parts.py (t5、vae) を別のプロセスで回すようにした (`--parts`)。Chroma1 のリポジトリには tokenizer.json がない (spiece.model だけ) ので、spiece.model (protobuf の wire 形式を直に読む。IRON の環境に sentencepiece と transformers はない) から Anima の t5_tokenizer.json と同じ形を書く (`t5_tokenizer_json`)。生成したものは Anima のものとすべての項目が同じ
- 途中で、この機械のメモリのビット反転がまた出た (16.155 の頃と同じ): Go のビルドキャッシュが壊れてリンカが落ち、Btrfs のチェックサムが 1 ビット違うと読めないファイルが出て、同じ起動中に書いた書き出しにチェックサムに引っかからない 1 ビットの違いがあった。再起動の後は scrub に誤りがなく、写しも同じ (ディスクでなくメモリ)。その起動で書いた書き出しは作り直した

### 16.168 1 回の実行の電力量を段階ごとに (`HWNPU_ENERGY`)、組の重みの読み直しを準備に数える — 2026-11-12

- **`HWNPU_ENERGY`** (値は待機を測る秒数、数でなければ 5): コマンドの最初に待機電力を測り、そのあと段階 (変換・読み込み・コンパイルと、実行) が変わるたびにパッケージの電力量を分ける。最後の行の時間 (`took ...`) の下に、各段階の電力量と、待機電力 × その段階の時間を引いた値を出す (`console.MeasureEnergy`、`energyLine`)。計は 50 ms ごとの台形の積分 (`power.Integrator`)、源は amdgpu の PPT (パッケージ全体: CPU、GPU、NPU。NPU だけのセンサーは Linux 7.0 の amdxdna にない)。待機を測る秒は時間に数えない。今は AMD の APU の Linux だけ (`power.PackagePower`)。hwnpu-image、hwnpu-chat、hwnpu-audio の main で呼ぶ
- **組の重みの読み直しを準備に**: SwapVariants で 1 つのバンクに組を順に読む `iron.Graph.Use` は、時間と電力量で準備に数える (`console.Reading`)。プロファイルには 1 行ずつでなく回数、バイト、時間を 1 行で (`profile: weights read again N times, ...`)。グラフの読み込み (`console.Loading`) と同じく、実行の途中の読み込みも準備
- Qwen-Image と Chroma のコマンドは `console.Running()` を呼んでいなかった (最後の `took ...` の行が出なかった)。ほかの画像のコマンドと同じく `loaded ...` の後に呼ぶ。文の符号器の行も `the text in X s: ...` にそろえた (Qwen-Image の AMD の `loading` は字句解析の時間で、いつも 0 だった)
- 1024² の 1 枚 (狐のプロンプト、種 1、既定の設定、各回の前に 20 秒休み、待機 10 秒):

| モデル | 時間 (準備 / 実行) | 電力量 (準備 / 実行) | 待機を引いた電力量 (準備 / 実行) | 組の読み直し |
|---|---|---|---|---|
| Anima-Turbo | 46 秒 (9.6 / 36) | 698 J (121 / 577) | 56 J / 329 J | 55 回 30.5 GB 4.2 秒 |
| Anima-Aesthetic | 3 分 18 秒 (11 / 186) | 3.23 kJ (158 / 3.07 k) | 79 J / 1.82 kJ | 174 回 55.5 GB 6.3 秒 |
| Qwen-Image-2.1-Turbo | 1 分 59 秒 (17 / 102) | 2.03 kJ (224 / 1.80 k) | 99 J / 1.07 kJ | 78 回 59.3 GB 9.6 秒 |
| Qwen-Image-2.1 | 8 分 22 秒 (36 / 466) | 8.99 kJ (567 / 8.43 k) | 316 J / 5.25 kJ | 334 回 258 GB 28 秒 |
| Chroma1-Flash | 4 分 22 秒 (23 / 238) | 4.45 kJ (345 / 4.10 k) | 182 J / 2.49 kJ | 573 回 142 GB 14 秒 |
| Chroma1-HD | 11 分 20 秒 (32 / 648) | 11.89 kJ (493 / 11.39 k) | 277 J / 7.02 kJ | 953 回 234 GB 22 秒 |

  - 待機は 6.7〜7.1 W。実行中の待機の分は全体の 4 割前後 (Qwen-Image-2.1-Turbo の実行 1.80 kJ のうち 0.73 kJ)
  - 準備は全体の 3〜8 %: グラフの読み込み 4.5〜8.8 秒と、組の重みの読み直し (ページキャッシュから、10〜18 GB/s)。読み直しはステップごとなので、ステップの多いモデルほど増える (Chroma1-HD 953 回)。ResidentDiT (組を全部置く) ならこれは消えるが、固定のメモリが組の数だけ増える (16.106)
  - 16.167 の表 (bench.sh) と同じ全体の値 (Qwen-Image-2.1-Turbo 2.02 と 2.03 kJ、Chroma1-HD 12.1 と 11.9 kJ)。ファイルはどれもページキャッシュにある状態 (冷えた状態は sudo で drop_caches が要る)

### 16.169 Core AI (macOS 27) のバックエンド: MIL を Go で Core AI のプログラムに、Anima-Turbo — 2026-11-13

Core ML の Neural Engine のキャッシュは約 4 GiB で、それを超えるモデル (Qwen-Image-2.1 の 17 GB、16.165) は 2 回目でもほぼ全部の部品のコンパイルが外れる。Core AI (macOS 27 の CoreAI.framework) は端末で専用化した結果を自分のキャッシュ (`~/Library/Caches/coreai-cache/<OS>/<実行ファイル>/<model hash>/`) に持ち、M3 で Qwen-Image-Turbo の 32 本 (17 GB、Python の coreai-torch で作った試作) を 1 枚の順に読んで、2 回目から 1 枚 46 秒 (Core ML 4 分 26 秒) だった。そこで Core ML と同じ `.mlpackage` (MIL) を Core AI のプログラムに下ろすバックエンドを足した。既定は Core ML のまま、`--backend coreai` か `HWNPU_BACKEND=coreai` で選ぶ。

- **Go だけで書く** (`internal/coreai/mlirbc`、`internal/coreai/ailower`): `.aimodel` は `main.mlirb` (MLIR バイトコード v6、`aicode` 方言)、`main.hash` (その sha256)、`metadata.json`。属性と型は方言のテキストの形で書き (読む側が方言のパーサで読む)、op の properties は方言の並び (必須は属性の番号、省略できるものは `番号 << 1 | 1`、無ければ 0)。coreai-core の出力を Go の読み手 (`tools/coreai/bcdump`) で解いて合わせた。Python も Xcode の Metal Toolchain (`coreai-build`、`.aimodelc` 用) も要らず、変換はどの OS でもできる
- **Core AI が求めたこと**: 位置は Core AI 自身の `#aicode.debuginfo.location_v1` (builtin の `loc` では「versioned IR」に変換できない)、ブロックの引数にも位置。縮約は複合の `reduce_max_v1` / `reduce_sum_v1` の中の `primitive.reduce_v1` (形の決まらない型)。行列の積は複合の `broadcasting_batch_matmul_v1` の中で `broadcast_to` と `batch_matmul` (基本の演算を一番外に並べると、int8 の重み `blockwise_shift_scale_v2` があるとプログラム全体が Neural Engine から外れ、重みの reshape では MPSGraph が形を誤る)。softmax は `exp(x - max) × reciprocal(sum)` (`exp / sum` や softmax は matmul と合わせて注意の演算にまとめられ、Neural Engine に載らない)。2 項演算は同じ形どうしだけで、広げるのは `broadcast_in_dims` (広げる軸と大きさだけ)
- **実行** (`internal/backend/coreai`): Swift の shim (`shim/shim.swift`、`tools/coreai/build-shim.sh` で `libcaishim.dylib`、go:embed して purego で読む。`CGO_ENABLED=0` のまま)。`Compile` は `.mlpackage` を下ろして hwnpu のキャッシュに `.aimodel` を置き (キーはパッケージのダイジェスト)、Core AI で Neural Engine を指定して読む。入出力はコピー (`NDArray(shape:scalarType:)` と strides に沿った memcpy)。Core AI のキャッシュのエントリーは壊れることがあり (読み込みは通って `loadFunction` が 0.01 秒で `_GenericObjCError 0`、作り直さない)、そのときはエントリーを消して読み直す。配置は Core AI のキャッシュの MPSGraph の manifest の `mps.fullyPlacedOnANE` で見る
- **確かめ方** (`tools/coreai/aicheck`): `run -cpu` は部品を Core AI で動かして `milcpu` と比べる。`probe` は小さな MIL (SDPA、RoPE、RMSNorm、int8 の線形…) が Neural Engine に載るか。`HWNPU_AILOWER_CUT=N` は先頭の N 個の演算だけを下ろし、外れる演算を二分探索で探す。`HWNPU_AILOWER_TRACE` は op_id と演算の対応 (Core AI のエラーは op_id で場所を示す)
- **Anima-Turbo** (M3、1024²、int8-int8): 22 本すべて Neural Engine に載り、`milcpu` とのコサイン 0.99987〜1.000000 (文の符号器 0.12 秒、DiT の 4 ブロックの組 0.60 秒、VAE の段 0.04〜0.35 秒)。1 枚は 1 回目 4 分 50 秒 (Core AI のコンパイル 4 分 13 秒)、2 回目 38 秒 (読み込み 2.5 秒、1 ステップ 4.1 秒)。Core ML の 2 回目は 37 秒 (読み込み 3.4 秒、4.0 秒)。Anima は 2.8 GB で Core ML のキャッシュに収まるので差はない。同じ種の Core ML との最後の潜在はコサイン 0.971、PSNR 24.7 dB で、構図は同じ (手元が少し違う)
- `.aimodelc` (`coreai-build compile`) は採らない: M3 で初回の読み込みが `.aimodel` の 5〜10 倍遅く (0.5 GiB で 26〜77 秒)、Core AI のキャッシュには重みがやはり複製され、アーキテクチャ (macOS で 22) ごとに重みを持つ。`.aimodel` は端末に依らない
- 1 本の上限は M3 で 2.5〜3 GiB (fp16 の合成で 2.5 GiB は全部 Neural Engine、3 GiB は外れて GPU)。Core ML の 1 GiB より大きい

残り: Core AI の変換に失敗すると LLVM が abort してプロセスごと落ちる (下ろし方の誤りは `aicheck` で先に見つける)。`ailower` と `mlirbc` の単体テスト。LUT (lut4、lut8)、拡大、`pad` などは書いてあるが Anima の部品でしか確かめていない。Qwen-Image を Core AI で (キャッシュの効果が出るモデル)。Core AI のキャッシュに重みが複製される分のディスク (Qwen-Image で 17 GB)。M3 以外のチップ。
### 16.170 Chroma1 (Flash と HD) を Mac の Neural Engine で: T5 と DiT の fp16 の落とし穴、DiT は lut8 — 2026-11-12

#123 で AMD に入れた Chroma1 (lodestones/Chroma1-Flash、Chroma1-HD) を Apple の Neural Engine (Core ML) に入れた。`hwnpu-image generate -m chroma` は Mac では `--dir/Chroma` に取得し、初回に Go だけで変換する (`chroma.Prepare`、`chroma.Convert`)。部品は `internal/imagegen/chroma/mil`、エンジンは `internal/imagegen/chroma/programs.go`。AMD のエンジン (`iron.go`) とパイプライン (トークン化、サンプラ、CFG) は共通。参照は 16 GB の Mac でも回る `tools/ref/ref_chroma_lean.py` (T5 は層ごと、DiT はブロックごとにチェックポイントから読む float32。diffusers のブロックをメタデバイスで作って重みを入れる。キーのマスクは `ref_chroma.py --mask keys` と同じ)。M1 (16 GB、macOS 27.2) で測った。

**プログラム** (1024² で 33 本、512² で 33 本)

| 部品 | 中身 | 形式 | 呼ぶ回数 |
|---|---|---|---|
| text_0〜text_21 | T5-XXL の 3 層ずつ 8 本 (既定の lut8 + wo fp16。fp16 なら 2 層ずつ 12 本)。最後は最終ノルムと DiT の context_embedder まで | lut8 (wo は fp16) | プロンプトごとに 1 回 (組を 1 本ずつ読んで閉じる) |
| mods | approximator (344 行) | fp16 | 1 枚に σ の数だけ (Flash の Heun 8 ステップで 8 回)。ステップの前にまとめて回して閉じる |
| dit_double_0〜16 | ダブルブロック 4 つずつ 5 本。最初は x_embedder も | lut8 | DiT 1 回に 1 回ずつ |
| dit_single_0〜31 | シングルブロック 8 つずつ 5 本。最後は norm_out と proj_out も | lut8 | 同上 |
| vae_s0〜s9 | FLUX の VAE (AMD と同じ 10 段) | fp16 | 1 枚に 1 回 (段ごとに読んで閉じる) |

- 行は [画像、文]。文のパディングはキーの加算マスク (`kbias`、プロンプトのトークンと 1 つのパディングだけ 0)。FLUX の RoPE の組 (2i, 2i+1) は、q と k の重みの行を頭の中で並べ替えて `blocks.ApplyRope` の (i, i+64) にする (`mil.RopePerm`)。cos と sin は定数
- 残差は T5 が 1/64 (`T5Scale`、残差は 2.8e5 に達する)、DiT が 1/32 (`DitScale`、文の行が 2.4e5) で持つ。参照の最大値は `ref_chroma_lean.py --stats`
- GroupNorm (VAE) は、グループの値を (1, 32, a, b) に並べ (どの軸も 16384 以下)、グループの最大値で割り、平均を引いてから 2 乗の平均を取る。512² の VAE は diffusers の float32 と PSNR 66.3 dB
- approximator の出力は時刻だけで決まるので、σ ごとに 1 回 (Heun のステップの終わりと次の始まり、CFG の 2 回で共有)。ステップの間は Neural Engine に持たない
- 空の負のプロンプト (Chroma1-HD の既定) は `</s>` 1 トークンなので、各行の注意のキーは 1 つしかない。T5 の 2 行 (`</s>` と最初のパディング) を変換のときホストの float32 で計算して `text/empty.bin` に置き、T5 を回さない
- CFG の 2 回は重みの組ごとに続けて回す (16.148、16.160)

**fp16 の落とし穴** (512²、狐のプロンプト、float32 の参照と同じノイズ)

1. **型なし定数の整数の割り算**: `T5FFScale/T5Scale` が `1/64 = 0` になり、T5 の FF の出力が消えていた (CPU でも合わない。層 0 だけのプログラムで見つけた)
2. **最大値で割った値の非正規化数**: T5 の行には他の 1000 倍の外れ値のチャンネルがある。norm の前に行の最大値で割ると (`milpart.ScaledRMSNorm`)、他のチャンネルが fp16 の非正規化数の近くになる。2 乗の平均は割った写しから取り、出力は元の x にスカラーを掛けて作る (`mil.t5Norm`)
3. **重みに畳んだ縮尺と小さな積**: 残差の 1/64 を o と wo の重みに畳むと、外れ値の大きい行で積が fp16 の正規化の範囲 (6e-5) を下回り、Neural Engine が桁を落とす (Qwen3.5 の GDN と同じ、16.88)。プロンプトの行のうち 3 行が、コサイン 0.93、0.75、0.36 になった。o の 1/64 は出力に掛け、wo は 1/8 を重みに、1/8 を出力に (wo の出力は 1.4e5 に達する)。context_embedder の 1/32 も出力に。さらに norm の出力を 16 倍して射影に入れ、射影の出力で割る (`t5Up`)
4. **int8 の積和のあふれ**: 行ごとの int8 は、Neural Engine で符号のままの積和を取ってから尺度を掛けるとみられ、大きな入力の線形で途中の値が fp16 を超える。T5 の層 4〜11 (FF の入力が 5e3) と、DiT の後半のダブルブロックの文の側 (σ = 0.25 で) に inf が出た。CPU のインタプリタでは同じ int8 のプログラムが正しい。lut8 (重みを表から fp16 に戻してから掛ける) では出ない。#125 の Qwen-Image も同じ
5. **Core ML が定数の倍率を重みに畳む**: DiT にも 3 と同じ手 (射影の前に 16 倍、出力の縮尺を半分ずつ) を入れたところ、ほとんどの組の出力が前と最後の桁まで同じだった。Core ML のコンパイラが線形の前後の定数倍を重みに畳み直しているとみる (メモリの「丸めの定数は畳まれる」と同じ)。そのうえ最後のシングルの組だけ壊れたので戻した。T5 で効いたのは、縮尺の後に足し算が続く形だったためとみる (確かめていない)

| T5 の形式と手当て | プロンプトの行と float32 のコサイン | いちばん悪い行 |
|---|---|---|
| int8 (1 を直した後) | 0.82 (CPU のインタプリタでは 0.9896) | |
| lut8 | 0.9976 | |
| fp16 | 0.9486 | |
| fp16 + 2 | 0.9613 | 0.36 |
| fp16 + 2 + 3 の o と wo | 0.9974 | 0.948 |
| fp16 + 2 + 3 (context_embedder まで) | 0.9979 | 0.949 |
| **fp16 + 2 + 3 + t5Up** | **0.99975** | 0.9961 |
| lut8 + 2 + 3 + t5Up | 0.9908 | 0.766 |
| **lut8、wo だけ fp16 + 2 + 3 + t5Up (既定)** | **0.99824** | 0.979 |

- T5 の既定は lut8 で、FF の出力の射影 wo (入力に外れ値のチャンネルを持つ、重みの 22 %) だけ fp16 (`ConvertOptions.TextFormatFor`、`--text-format-for`)。プロンプトの行は全部 fp16 より少し離れるが、絵は同じ: 512² の 3 プロンプト (種 0) で、参照の float32 との潜在のコサイン 0.9930 / 0.9965 / 0.9945 (全部 fp16 では 0.9965 / 0.9961 / 0.9931)、全部 fp16 の T5 との 0.9961〜0.9990。重みは 9.5 → 5.5 GB、プロンプトごとの T5 は 12.7 → 9.5 秒で、ディスクとコンパイルのキャッシュも 4 GB 減る。全部 lut8 は行が崩れる (上の表)。AMD の q8 の T5 は 0.99975 (16.156)

| DiT の形式 | σ = 1 の速度 | σ = 0.25 の速度 | 8 ステップの潜在 | PSNR |
|---|---|---|---|---|
| int8 | 0.999966 | 0.990 (相対 14 %) | 0.922 (縞とピンクのむら) | 23.7 dB |
| **lut8 (既定)** | 0.99990 | 0.99961 (相対 2.8 %) | **0.9965 (同じ絵)** | **37.9 dB** |

**DiT をもっと小さく** (512²、狐の種 0、参照の 15 回の呼び出しの最小のコサイン / 最後の潜在のコサイン / PSNR。注意は lut8 のまま、MLP (`ff.net`、`ff_context`、`proj_mlp`、`proj_out` の MLP の側) だけ変える、`--dit-format-for`、`--dit-quant`)

| MLP | DiT | 呼び出し | 潜在 / PSNR | 絵 |
|---|---|---|---|---|
| lut8 (既定) | 8.6 GB | 0.9992 | 0.9965 / 37.9 dB | 同じ |
| lut6 | 6.9 GB | 0.9947 | 0.9853 / 31.2 dB | 同じ |
| lut6 k-means-minmax | 6.9 GB | 0.9951 | 0.9843 / 31.2 dB | 同じ |
| lut4 k-means-minmax | 5.4 GB | 0.9777 | 0.9315 / 23.3 dB | 狐の姿勢と大きさが変わる |
| lut4 | 5.4 GB | 0.8896 | 0.8737 / 20.7 dB | 崩れる |

- 4 ビットは蒸留した DiT には足りない (AMD の q4 と同じ、5.2 節)。lut6 は絵は同じだが、1024² でも速くならなかった (上) ので、既定は lut8 のまま。lut6 で減るのはディスクとコンパイルのキャッシュの 1.7 GB だけ


- 速度は参照の各呼び出しの入力を通したもの (15 回とも 0.9993 以上)。同じ int8 のプログラムを CPU のインタプリタで回すと σ = 0.25 で相対 1.5 %: int8 の誤差は Neural Engine の積和 (4) による
- SANA 1.5 も行ごとの int8 で崩れ、lut8 で直った (16.160)。原因は同じとみる

**3 プロンプト × 2 種** (Chroma1-Flash、512²、Heun 8 ステップ、`ref_chroma_lean.py` の float32 と同じノイズ。狐、窓辺の老漁師、雨の夜のネオン街)

| | 種 0 | 種 1 |
|---|---|---|
| 狐 | 0.9965 / 37.9 dB | 0.9870 / 31.4 dB |
| 老漁師 | 0.9961 / 34.8 dB | 0.9972 / 34.4 dB |
| ネオン街 | 0.9931 / 29.9 dB | 0.9966 / 32.2 dB |

(潜在のコサイン / 画像の PSNR)。6 組とも構図、顔、看板の配置まで同じ。2 回目からの 1 枚は 1 分 52〜56 秒

**速さとメモリ** (M1、2 回目から、Chroma1-Flash の既定: Heun 8 ステップ、DiT 15 回)

| | 512² | 1024² |
|---|---|---|
| 1 枚 | 2 分 06 秒 | 約 9 分 30 秒 (計算 8 分 55 秒と読み込み約 25 秒の見積もり。測った 3 回目は組の再コンパイルが入って 13 分 34 秒) |
| T5 (12 本の読み込み 11 秒を含む) | 12.8 秒 | 同じほど (行は 512 のまま) |
| DiT 1 回 | 6.4 秒 | 約 35 秒 (Heun の 1 ステップ 69〜73 秒) |
| VAE | 1.4 秒 | 5.5 秒 |
| Neural Engine が抱えるメモリ (`mem.NPUWired` の最大) | 8.29 GiB | 6.92 GiB (5.11 GiB の回も) |
| footprint の最大 | 3.5 GB | 4.3 GB |
| 初回 (変換、コンパイル) | 変換 1 分、コンパイル込みで約 17 分 | 変換 1 分、コンパイル約 1 時間半 (シングルの組 1 本 19 分、ダブル 4〜6 分) |

- DiT の 10 本 (lut8、8.3 GiB) は全部 Neural Engine に固定されたまま。M1 の約 2.9 GiB (16.118) を超えるのに載せ直しはなく、DiT 1 回は組の計算時間の和 (約 6.1 秒) とほぼ同じ。組を 1 本ずつ読んで閉じると 1 本 1 秒ほどかかる (DiT 1 回に 10 秒) ので、持ったままにする
- 計算の見積もり: 512² の DiT 1 回は約 21 TFLOP (行 1536)。6.4 秒で 3.3 TFLOPS。1024² は約 74 TFLOP (行 4608、注意は 15 TFLOP)
- **1024² の DiT 1 回は 35 秒**: Chroma1-HD の CFG のあるステップ (DiT 2 回) は 58 秒、ないステップ (1 回) は 35 秒で、組ごとに続けて回す 2 回目は 23 秒。NPUWired は 4.4〜6.9 GiB で DiT の 8.3 GiB を下回る。初めはこの差を組の載せ直しとみたが、DiT を 6.9 GB にしても (MLP を lut6、下) 1 ステップ 67〜75 秒、計算 8 分 59 秒で lut8 と変わらず、NPUWired は 4.37 GiB に下がった。重みの量によらないので載せ直しではない。2 回目が速い理由は見えない (組の続けての呼び出しで何かが使い回される)
- **コンパイルのキャッシュ**: Core ML のキャッシュはプロセス名ごと (`~/Library/Caches/hwnpu-image/com.apple.e5rt.e5bundlecache`)。決まった上限は見えず (49 GB まで増えた)、ディスクの空きが約 15 GiB を切ると macOS が古いものから消し、DiT の組や T5 がコンパイルし直された (1 本 2〜19 分)。初めは約 25 GB で頭打ちとみていたが誤り。同時に `~/Library/Caches/hwnpu` には変換し直したパッケージの古いコンパイル済みモデルがたまる (2 日で 100 GB 以上)。項目はソースのパスを 1 つ持つが、ハードリンクした書き出しは同じ項目を使う。古い書き出しを消してから「ソースのない項目」を消すと、使っている新しい書き出しの項目と Neural Engine のコンパイル済みの束まで失い、DiT がコンパイルし直された。当たったときに記録されたソースがなければ今のパッケージを記録し直すようにした (`internal/backend/coreml`、`cache.Prune` も同じ判定)。ディスクを空けた後は、Flash の 1024² の 2 回目の読み込みは 22 秒 (全部キャッシュから)
- **同じ部品をハードリンクで** (`ConvertOptions.LinkFrom`、`chroma.Prepare` はここにある完成した書き出しを全部候補にする): T5 のプログラムは画像の大きさによらず、最後の組 (context_embedder) を除いて Flash と HD で同じ。VAE は大きさが同じなら同じ。Core ML のキャッシュはプログラムのバイトで引くが、変換は毎回バイトを変える (UUID) ので、別々に変換すると同じ T5 を 2 回コンパイルして 2 回持つ。HD の 1024² は T5 の 11 本と VAE の 10 段を Flash の書き出しからリンクし、変換 39 秒、ディスクとキャッシュは DiT と approximator と T5 の最後の 1 本の分だけ (約 10 GB) 増えた
- **シングルブロックを 4 つずつの組に** (初回のコンパイルを短くできるか): 1024² の 1 本は、裏で CPU の参照が回っていたため配置の確認 (MLComputePlan) が 10 分の上限を超えて失敗した。結論は出ていない
- **配置の確認の上限を 30 分に** (`internal/backend/coreml` の `planTimeout`): 1024² のシングルブロック 8 つの組に lut6 と lut8 の重みを混ぜると、CPU が空いていても確認が 10 分を超え、読み込みが「NPU だけでは動かない」で失敗した。lut8 だけの組も読み込みに最大 19 分かかっており、確認の時間は読み込みと同じほど。30 分にして 19〜27 分で通った
- **同じモデル、大きさ、DiT の形式の書き出しから DiT もリンク**: T5 の形式だけを変えて書き出し直すとき、approximator と DiT の組をリンクし、1024² の DiT の 1 時間半のコンパイルをしない
- **文の行を 256 に** (`--text-rows 256`、プロンプトが 255 トークン以下なら同じ結果): 512² の 1 枚 1 分 55 秒 → 1 分 47 秒 (ノイズ除去 1 分 53 秒 → 1 分 45 秒、T5 12.7 → 11.7 秒)。行は 17 % 減るが時間は 7 % しか減らない。512 行との潜在のコサイン 0.99978。長いプロンプトが切れるので既定にしない
- `~/Library/Caches/hwnpu` に、変換し直したパッケージの古いコンパイル済みモデルが 135 個 (100 GB 以上) 残っていた。キャッシュの項目はソースのパスを持つので、ソースがないか、ソースの方が新しいものを消した

**部品として組み合わせるときの値** (new-model-checklist の 10 節。M1、512²)

| 部品 | 重み | Neural Engine に持つ間 | 2 回目からの読み込み | 初回のコンパイル |
|---|---|---|---|---|
| T5-XXL (lut8 + wo fp16、3 層ずつ 8 本。fp16 なら 2 層ずつ 12 本) | 5.5 GB (fp16 9.5 GB) | 1 本 (0.7 GB) ずつ、回したら閉じる | fp16 は 1 本 0.85〜0.94 秒 (12 本で 11 秒) | fp16 は 1 本約 26 秒 |
| approximator (fp16) | 0.52 GB | ステップの前に σ の数だけ回して閉じる | 0.6 秒 | 11 秒 |
| DiT (lut8、10 本) | 8.3 GiB | 最初のステップから最後まで全部 (`--keep-dit` で次の絵まで) | 1 本 0.8〜3.3 秒 (10 本で 13 秒) | 512² で 1 本 1.5〜3 分、1024² で 4〜19 分 |
| VAE (fp16、10 段) | 0.1 GB | 1 段ずつ | 1 段 0.02〜0.1 秒 | 数秒 |

- **全部のプロンプトをまとめて** (new-model-checklist 7.1、Qwen-Image の 16.165 と同じ): 複数のプロンプトは T5 の 1 回の読み込みでまとめて符号化し (負のプロンプトも)、DiT は持ったまま (`--keep-dit`)。512² の 2 枚で、2 枚目は 1 分 39 秒 (1 枚だけのとき 2 分 06 秒。T5 の読み込み 11 秒と DiT の読み込み 13 秒がない)
- 部品はすべて順に回り、同時には回らない (T5 → approximator → DiT → VAE)。T5 と approximator は DiT の前に閉じるので、DiT と一緒に持つことはない
- 合計はコンパイルのキャッシュ (約 4 GiB、10 節) を大きく超えるが、ディスク上の Core ML のキャッシュに残っていれば再コンパイルはない。残らないとき (上の追い出し) は 1 本数分

**そのほか**

- `cache.TreeDigest` はシンボリックリンクのパッケージを辿らず、リンクしたパッケージが全部空のツリーの鍵になって、同じコンパイル済みのモデル (先にコンパイルした approximator) に化けていた。リンクを解いてから読む (`TestTreeDigestSymlink`)
- 他のプロセスが Neural Engine でコンパイルしている間は、VAE の段 0 が「Unable to compute the prediction」で落ちた。独りで回すと通った
- テスト: `internal/imagegen/chroma/device_test.go` (`HWNPU_CHROMA_EXPORT` と `HWNPU_CHROMA_REF`)。`TestVAEOnDevice`、`TestDiTOnDevice` (部品ごとに参照の入力から)、`TestCallsOnDevice` (参照の各呼び出し)、`TestTextGroupsDeviceVsCPU` (組ごとに Neural Engine と CPU のインタプリタ)

**Chroma1-HD** (M1、1024²、既定: DPM-Solver++ 2M 25 ステップ、CFG 3 は σ ≥ 0.6 の 17 ステップ、空の負のプロンプト): 書き出しは Flash から T5 と VAE をリンクして約 10 GB。2 回目から 1 枚約 22 分 (計算 21 分 32 秒: CFG のあるステップ 58 秒 × 17、ないステップ 35 秒 × 8、VAE 5.5 秒)、NPUWired 4.7〜5.4 GiB、footprint 4.3 GB。照合は CPU の float32 の参照 (`ref_chroma_lean.py --size 1024 --calls 4`、1 回 213 秒) の最初の 4 回 (CFG の 2 ステップ) の入力を Mac の DiT に通し、文の行は Mac の T5 (負は `text/empty.bin`) で: 速度のコサイン 0.99983〜0.99991

**既定の書き出しで** (`hwnpu-image generate -m chroma` と `-m chroma:1-hd`、1024²、T5 lut8 + wo fp16、DiT lut8): 書き出しを変えると、`chroma.Prepare` は前の書き出しから approximator、DiT、VAE をリンクし (Flash 21 本、HD 28 本)、変換は 13〜23 秒。新しいのは T5 の 8 本 (コンパイル 1 本 38 秒) と HD の T5 の最後の 1 本。ディスクの空き 38 GiB で Flash → HD → Flash と回すと、3 回ともコンパイルし直しはなかった (読み込み 22〜25 秒)

| | 1 枚 | 計算 | T5 | NPUWired の最大 | footprint の最大 |
|---|---|---|---|---|---|
| Chroma1-Flash | 9 分 17〜27 秒 | 8 分 53 秒〜9 分 02 秒 | 9.9〜10.2 秒 | 6.3〜7.1 GiB | 4.25 GB |
| Chroma1-HD | 21 分 58 秒 | 21 分 36 秒 | 9.2 秒 | 7.8 GiB | 4.33 GB |

**できていないこと**: 電力量 (powerbench は sudo)。1024² の DiT 1 回の 35 秒の中身 (CFG の 2 回目が 23 秒で済む理由)。

### 16.171 キャッシュの根を 1 つに、共有の層 (共有フォルダと S3) と AMD の書き出しの共有 — 2026-11-12

docs/cache-sharing.md の P0 と P1、P2 の大半 (同 0 節に一覧)。

- **根** (`cache.Root`): OS のローミングしないキャッシュのフォルダの下の `hwnpu` (`HWNPU_CACHE` で移せる)。各コマンドがばらばらに `os.UserCacheDir()` を引いていた 13 か所をこれに。AMD の書き出しは `entries/amd-xdna2/NAME.amd-npu2*` に作り、`~/models` は読むだけ (`cache.ExportRoots`)。組み込みの IRON の道具は `tools/`。IRON の JIT のキャッシュは `iron/` (`NPU_CACHE_HOME` を書き出しに渡す)。`~/.npu/cache` は最初の書き出しの前に `iron/` へ名前を変えて移し、元の場所にリンクを残す (この機械では 52 GB が一瞬で移った)
- **`.done` を相対パスに** (`cache.WriteDone`、`ReadDone`): チャットの変換、書き起こし、Anima、Sana、Qwen-Image、hwnpu-dev formats。前の hwnpu の絶対パスは、その場所がなければ印の隣の同じ名前を読む
- **ロック** (`cache.Lock`): `locks/` のファイルを flock (Windows は LockFileEx) で。ダウンロード (ファイルごと)、チャットの変換 (プログラムごと)、IRON の書き出し (書き出しごと)。待った側は、つかめたときに相手が作ったものを見つける
- **HF の続きのダウンロードに `If-Range`**: 落とし始めの ETag を `NAME.part.etag` に残し、続きはその版のときだけ。ETag のない前の hwnpu の `.part` は今までどおり続ける。応答の `X-Repo-Commit` (リダイレクトの前の応答も見る) を、層に置くときのコミットとして覚える
- **共有の層** (`internal/store`): `Store` (フォルダ `Dir` と `S3`) の上に、blob (`blobs/sha256/AB/SHA`)、エントリ (`entries/KIND/KEY.json`)、チェックポイントのファイルの索引 (`upstream/OWNER/NAME@COMMIT/FILE.json`)。blob が先、名指すものが最後。取るときは sha256 を確かめ、違えば次の層へ
  - 設定: 定義ファイルの `cache.layers` (`use`、`write`、`fill`、`fill_gated`、`scopes`。後のファイルが並びごと置き換える)、機械の定義ファイル (`/etc/hwnpu/models.yaml`、macOS は `/Library/Application Support/hwnpu/models.yaml`、Windows は `%ProgramData%\hwnpu\models.yaml`。組み込みとユーザーの間)、`HWNPU_CACHE_LAYERS` (並びを置き換える。`off`)、`HWNPU_CACHE_WRITE=0` (読むだけ)
  - **元のチェックポイントのフォワードプロキシ** (`fill`): `fetch` は HF の前に層を見る。定義の sha256、なければ HF への HEAD の `X-Linked-Etag` (LFS のファイル) で blob を、どちらもなければコミットの索引で探す。HF から落としたものは裏で層に置く (sha256 も裏で計算)。gated のリポジトリとライセンスを尋ねるモデルは `fill_gated` の層にだけ
  - **AMD の書き出しの共有** (`chat.SharedExport`): 鍵はモデル、元のコミット、定義の sha256、`amd_export` の引数、組み込みの道具の hash の指紋 (`chat.ExportRecipe`)。層にあれば IRON なしで取り、なければ作って `entry.json` を書き、`write` の層へ裏で置く。前の層で当たったものは手前の `fill` の層へ。コミットが分からない (固定しておらず HF に届かない) ときは層を使わない
  - 裏の仕事は `work/queue/` に書いてから行い、終われば消す。切れたら次の起動で続ける。失敗は `work/queue.log` に書き、実行は止めない
- **S3** (`store.S3`): tinygodriver の `storage/s3` に。取得元の `type: s3` も `aws` コマンドから移した。鍵は環境変数か、`profile` の `~/.aws/credentials` と `~/.aws/config` の静的な鍵だけ (SSO は `aws configure export-credentials --format env` で環境変数に)。64 MiB を超えるファイルは 64 MiB の部分で上げ、失敗したら中止を送る。5xx、`SlowDown`、切断は 3 回まで。`no_sign` は署名のない GET。`path_style` を足した
- **`hwnpu-dev cache`**: `layers` (根と層、何が決めたか)、`ls` (層のエントリ)、`push MODEL` (手元の書き出しを `write` の層へ。`entry.json` がなければ今の道具の指紋で作る)、`verify DIR [--sums]`、`flush` (裏の仕事を終える)
- 確かめたこと: テスト (フォルダと偽の S3 の読み書き、部分に分けた送り、続きからのダウンロード、署名なし、プロファイル、エントリの往復と変わった blob、コミットと sha256 での取得、裏の仕事の残り、層を通したダウンロードで HF からの GET が 0、gated を写さない、書き出しを 1 回作って別の根で作らずに取る)。この機械で: Qwen3.5-0.8B の q6 の書き出し (1.57 GB) を `push` で共有フォルダの層へ (3.5 秒)、HOME とキャッシュの根を空にし `HWNPU_IRON_ENV` を存在しない場所にした hwnpu-chat が、層から 3.5 秒で取って NPU で答えた (チェックポイントは落とさない)
- RustFS 1.0.5 (手元) で (2026-11-13): `TestS3Server` (`HWNPU_S3_TEST_ENDPOINT`) が部分に分けた送り、空のファイル、続き、一覧、削除で通った。空のファイルは tinygodriver v1.3.8 が長さ 0 の本体を chunked で送り 411 になった (v1.3.9 で直り、hwnpu の回避は外した)。Qwen3.5-0.8B の書き出しを RustFS の層に `push` (1.57 GB、10 秒)、空の根と IRON なしの hwnpu-chat が層から取って答えた (準備 4.4 秒)
- Cloudflare R2 (`region: auto`) でも `TestS3Server` が通り、push (1.57 GB、119 秒) と層からの chat (準備 54 秒) ができた。無料枠は 10 GB なので小さなものだけ置く
- 未確認: AWS の S3。Windows のロックの実機。MinIO は対象にしない (保守が終わった)
- **初回に作る画像の書き出しも層に** (2026-11-13、`amdFirstRunExport`): hwnpu-image detect、pose、segment (RTMDet、RTMO、EfficientSAM) の書き出しは、手元になければ層を見て、なければ IRON で作って `write` の層へ。層にあれば IRON のない機械でも NPU で動く。鍵は `chat.PinnedRecipe`: 定義の sha256 (ONNX のモデルはファイルを全部固定している)、道具の名前と hash、Go の側の版 (`detect.ExportVersion`、`pose.ExportVersion`、`segment.ExportVersion`)。道具に渡す引数は Go が ONNX から決めるので、引数そのものは鍵に入れず、渡し方を変えたら版を手で上げる (cache-sharing.md 3 節の方針)。エントリの種類は `iron-onnxcnn` と `iron-efficientsam` (`chat.ExportKind`)。`hwnpu-dev cache push` もこの 3 種を受ける。この機械で: RTMO-s、RTMDet-m、EfficientSAM-Ti を共有フォルダの層に push し、空の HOME とキャッシュの根、IRON なしで、3 つとも層から 0.1〜0.2 秒で取って NPU で動いた
- **`hwnpu-dev cache gc N`** (`store.PlanGC`): 層 N から、`--model`、`--kind`、`--key` (先頭)、`--older-than DAYS` で選んだエントリと、それだけが名指していた blob を消す。`--orphans` で、どのエントリにもチェックポイントの索引にも名指されない blob も (書き込み中の層では使わない: 置いている途中のものは名指すものが最後に書かれる)。`--apply` がなければ、消すものを出すだけ。エントリを先に消し、blob を後に。共有フォルダの層と RustFS で確かめた。途中で止まったマルチパートは tinygodriver に一覧の操作がないので、バケットのライフサイクルの規則 (「未完了のマルチパートを N 日で中止」) に任せる
- 残り: Anima などの手で作る画像の書き出し (道具の引数を Go が持たない。HF の配布物で配る)。Core ML と OpenVINO のプログラム (チャットの変換) のエントリ化と、計画と作るの分離 (cache-sharing.md 8 節)、HF の配布物を読む層と Actions (同 9 節) は次

### 16.172 PLaMo (3 Translate と Embedding) を Core AI で: 経路の直しと、入力の配列の使い回し — 2026-11-13

16.169 の Core AI のバックエンドで PLaMo 3 Translate (0.8B、2B) と PLaMo-Embedding-1B を動かした。下ろし方 (`ailower`) は手を入れずに通り、どの部品も Neural Engine に載った。直したのは 2 つ。

- **経路**: `hwnpu-chat` は `HWNPU_BACKEND=coreai` でも Core ML で動いていた。`chat.Prepare` と `PrepareEmbedder` が Config.Backend をプログラムの形式 (`coreml`) で埋め、`OpenDevice` がそれで Core ML の端末を開くため。`llm.DeviceBackend` (指定された端末のバックエンド) を先に見る。`HereVendor` も `coreai` を apple に。`--backend coreai` は `hwnpu-chat` のどのコマンドでも使え、変換されるのは Core ML と同じ `.mlpackage` (同じディレクトリを共有する)
- **1 回の呼び出しの手間** (shim): 入力の `NDArray` を毎回作ってコピーしていたので、0.8B の生成は Core ML の 35 トークン/s に対して 25 だった (M1)。Core AI は一度使った配列を速く動かす: 0.8B の part1 (入力 7 MiB) で、毎回新しい配列 10.0 ms、同じ配列 7.2 ms、呼び出し側のバッファの view (`NDArray.RawView(bytes:)`、`outputViews`) 10.7 ms、IOSurface の view 8.3 ms。そこで入力の配列はプログラムごとに初回に作り、毎回それに memcpy する (7.7 ms、Core ML 7.8 ms)。プログラムの呼び出しは 1 本ずつ (graph.mu) なので配列を共有してよい

**結果** (M1、macOS 27.2、2 回目以降)

| モデル | Core ML | Core AI | 一致 |
|---|---|---|---|
| PLaMo 3 Translate 0.8B (lut8) | 生成 35〜42 トークン/s、読み込み 1.5 秒 | 38〜44 トークン/s、1.3〜1.5 秒 | 訳は一字一句同じ (2 文、3 文の段落) |
| PLaMo 3 Translate 2B (lut8) | 11.7 トークン/s、9.4 秒 | 11.7 トークン/s、10 秒 | 一字一句同じ |
| PLaMo-Embedding-1B (lut8、128 と 1024) | 1 文 48 ms | 48 ms | Core ML とのコサイン 0.99995〜0.99999 (8 文) |

- 初回の読み込み (Core AI のコンパイル) は 0.8B が 33 秒、2B が 29 秒、Embedding が 1 分 20 秒で、Core ML (40 秒、38 秒、1 分 51 秒) より短い
- `aicheck run -cpu` で 0.8B の部品を `milcpu` と比べた: 8 トークンの部品はコサイン 0.9989 以上。128 トークンの part1 は 0.92 まで下がるが、Core ML でも同じ入力で同じ値 (0.9179 と 0.9175) で、乱数の入力 (マスクなど) のせい
- `aicheck run` に `-coreml` (同じ部品を Core ML で、誤りが Core AI のものかを分ける) と `-calls N` (2 回目以降の N 回の平均) を足した

- **8B (lut4)** も M1 で試した (自動では選ばれないので `--format lut4` で変換): 生成はどちらも 0.8 トークン/s、2 回目の読み込みは Core AI 4.3 秒、Core ML 24 秒。訳 (段落、1 文、4 文の英日・日英) は一字一句同じ。部品はすべて Neural Engine に載り、`milcpu` とのコサインの最小は 0.984 (part1 の kv_v_7) で Core ML も 0.9839 なので lut4 と乱数の入力のせい。遅いのは M1 の Neural Engine が常駐 2.8 GiB を超えると約 12 倍遅くなるためで Core AI でも同じ。lut4 より小さい形式はない (lut3 は MIL では書けるが未実装で、質も落ちる見込み)

残り: 8B の速さは M2 と M3 で。電力量。Core AI のキャッシュに重みが複製される分のディスク (`hwnpu-chat` の 0.8B と 2B で 3.9 GB、重みの 3.8 GiB とほぼ同じ)。そのキャッシュは実行ファイルの名前ごとなので、`aicheck` や `go test` (chat.test) の分は別に持つ。

### 16.173 チャットのモデル (Qwen3、Qwen3.5、Gemma 4、LFM3) を Core AI で — 2026-11-13

16.172 に続けて、`hwnpu-chat` のチャットのモデルを `--backend coreai` で動かした (Qwen3.5-4B と gemma-4-E4B-it はまだ)。変換は Core ML と同じで、HF から落としたチェックポイントを Go で `.mlpackage` にし、それを `ailower` で Core AI のプログラムに下ろす。`ailower` にも shim にも手を入れずに全部通り、どの部品も Neural Engine に載った。どのモデルの MIL も `ailower` の演算の範囲に収まっている (linear、matmul、softmax、gelu、constexpr_lut_to_dense など)。

直したのは 1 つで、Core AI とは関係がない。`hwnpu-chat` は `--vision` と `--audio-encoder` を指定しないと、Prepare が変換した符号器を空で上書きしていた (`cmd/hwnpu-chat/load.go`、#97 から)。そのため LFM3-VL、Audio、Omni は Core ML でも画像と音声を読めず、`this model was loaded without a vision encoder` になっていた。指定があるときだけ上書きする。

**結果** (M1 16 GB、macOS 27.2。貪欲法 `--temp 0`、生成の速さは 2 回目以降)

| モデル (形式) | 生成 Core ML / Core AI | 2 回目の準備 Core ML / Core AI | 1 回目の準備 Core ML / Core AI | 一致 |
|---|---|---|---|---|
| Qwen3-0.6B (lut8) | 8.6 / 8.4 トークン/s | 19 秒 / 2.3 秒 | 47 秒 / 45 秒 | 33 トークン目から分かれる |
| Qwen3.5-0.8B (lut8、128 の読み取り) | 24.6〜25.3 / 25.3〜27.0 | 12 分 54 秒 / 2.3 秒 | 27 分 45 秒 / 5 分 9 秒 | 20 トークン目あたりから分かれる |
| Qwen3.5-2B (lut6、1 本) | 20.9〜21.2 / 20.0〜21.0 | 59 秒 / 1.6 秒 | 2 分 15 秒 / 1 分 20 秒 | 一字一句同じ |
| gemma-4-E2B-it (lut6) | 24.1〜24.7 / 24.0〜24.5 | 1 分 8 秒 / 2.7 秒 | 2 分 11 秒 / 1 分 17 秒 | 同じ (考える部分を含む 400 トークン) |
| LFM3-1.7B (lut6) | 35.7〜36.3 / 35.2〜35.5 | 57 秒 / 4.5 秒 | 12 分 49 秒 / 1 分 24 秒 | 同じ (249 トークン) |
| LFM3-VL-2B (lut8、画像) | 28.4 / 26.2 | 5.5 秒 / 3.2 秒 | 53 秒 / 4 分 4 秒 | 同じ |
| LFM3-Audio-2B (lut8、音声) | 25.0 / 24.7 | 3.9 秒 / 3.8 秒 | 43 秒 / 51 秒 | 1 文目 (「日本の首都は東京」) は同じ、そのあと分かれる |
| LFM3-Omni-2B (lut8、画像と音声) | 22.5 / 22.4 | 8.6〜22 秒 / 4.5〜5.5 秒 | 1 分 27 秒 / 1 分 17 秒 | 画像も音声も同じ |

- **準備** は読み込みとコンパイル (`ready after`) の時間。1 回目は、Qwen3-0.6B と Qwen3.5-0.8B 以外は Core AI を先に動かしたので、Core AI の側に Go の変換 (数十秒) が入る。LFM3-VL-2B の Core ML の本体は、直す前の実行でコンパイル済みだった
- **2 回目の準備**: Core AI は 1.6〜5.5 秒でどのモデルもほぼ一定。Core ML は Neural Engine のコンパイルのキャッシュ (約 4 GiB、ディスクが少ないと消される) から外れた部品をコンパイルし直すので、何を前に動かしたかで 4 秒から 13 分まで変わる。Qwen3.5-0.8B の 128 トークンの部品は Core ML のコンパイルが 1 回 26 分かかり、2 回目も 12 分かかった (Core AI は初回 5 分、2 回目 0.7 秒)
- **配置**: Qwen3.5 の 2B の本体と、0.8B と 2B の multi-token prediction の部品は、Core ML では一部が Neural Engine から外れる (`NOT all on the NPU`)。Core AI では全部載る
- **生成の速さ**はほぼ同じで、違いは ±10 % 以内。LFM3-VL-2B だけ Core AI が 8 % 遅い
- **精度**: `aicheck run -cpu` (乱数の入力で `milcpu` と比べる) で Core AI と Core ML の誤差は同じだった。Qwen3-0.6B の 8 トークンの部品で、y のコサインは Core AI 0.99975、Core ML 0.99983。Qwen3.5-0.8B の部品と MTP はどちらも 0.999997 以上。LFM3-Audio の音声の符号器 (fp16) は Core AI 0.9947、Core ML 0.9942。貪欲法の文が途中で分かれるのは、fp16 の丸めの違いで僅差のトークンが入れ替わるため
- 画像は計算機のアイコン (512²)、音声は `say -v Kyoko` の「日本の首都はどこですか。一文で答えてください。」(16 kHz)

**ディスク**: Core AI の経路では、重みがディスクに 3 回置かれる。変換した `.mlpackage`、hwnpu のキャッシュの下ろした `.aimodel`、Core AI 自身のキャッシュ (`~/Library/Caches/coreai-cache/<OS>/hwnpu-chat/`) の 3 か所。今回の 8 モデルで Core AI のキャッシュは 7.6 GB から 23 GB に増えた。この M1 はディスクの空きが 12〜50 GB しかなく、試すために、試作の Core AI のキャッシュ、使わない古い変換、Chroma の重みを消した。

残り: Qwen3.5-4B と gemma-4-E4B-it (落とすのが 16 GB、ディスクが要る)。`.aimodel` は Core AI のキャッシュに入ったあとも要るのか (要らなければ重みが 1 回分減る)。M2 と M3。電力量。

### 16.174 `.aimodel` を配って動かすための調査: 容量、キャッシュの鍵と方針、重みの共有 — 2026-11-13

変換済みのモデルを Hugging Face に置き、Apple では `.aimodel` を落として動かす形 (docs/cache-sharing.md の 10 節) を考えて調べた。M1 (macOS 27.2)、題材は LFM3-1.7B (lut6、8 トークンと 128 トークンの 2 本、それぞれ 7 つの部品)。

**決めたこと**
- 配るのは、8 トークンと 128 トークンを関数 2 つにまとめた `.aimodel` (重みは 1 本分) と、実行に要る小さなファイル
- `.aimodel` (`main.mlirb`) は手元に残すのを既定にする。Core AI のキャッシュに入ったあとは消しても動くが、キャッシュが消えたとき (ディスクが足りないときの掃除、OS の更新) に落とし直しになるので、消すのは容量に困ったときの最終手段のオプションにする

#### 今の容量 (元のチェックポイントから手元で変換する形)

| | ファイル | 大きさ |
|---|---|---|
| HF から落とす | `model.safetensors` (変換し直すときだけ使う) | 3.46 GB |
| | `tokenizer.json`、`config.json` | 18 MB |
| 手元で変換する | `tables.safetensors` (埋め込みの表、CPU で引く) | 0.52 GB |
| | `.mlpackage` 8 トークン、128 トークン | 各 1.24 GiB (重みは同じバイトで APFS の複製、実質 1 本分) |
| 実行のとき (Core ML) | hwnpu のコンパイルのキャッシュ (`.mlmodelc`) | 2.43 GiB (ほかに OS の Neural Engine のキャッシュ、測っていない) |
| 実行のとき (Core AI) | hwnpu のキャッシュの下ろした `.aimodel` | 2.43 GiB |
| | Core AI のキャッシュ (`coreai-cache/26B123/hwnpu-chat/`) | 約 2.5 GiB |

Core AI で動かすと全部で約 10 GB で、重みを 5 回分持っている (`.mlpackage` で 1 回、`.aimodel` で 2 回、Core AI のキャッシュで 2 回)。変換し終えたら `model.safetensors` は消せる。

#### 実行のときに読んでいるもの

- NPU の部品の `.mlpackage`: Core AI でも毎回読む。(1) 中身を全部読んで sha256 をとり、hwnpu のキャッシュの `.aimodel` を探す鍵にする (`cache.TreeDigest`)。(2) 出力の形を読む (`runner.go` の `outputShapes`)。(3) 元が消えると、30 日の掃除 (`cache.Prune`) で対応する `.aimodel` も消える
- `attn` と `mass` の `.mlpackage` (28 KB ほど): ホストの CPU で `milcpu` が動かす (キャッシュの上の注意)
- `parts.json` (分け方と注意の層)、`tables.safetensors`、`tokenizer.json`、`config.json`
- (1) の sha256 は LFM3-1.7B の 18 個 (2.4 GiB) で 2.9〜3.9 秒かかり、2 回目の準備 4.5 秒のほとんどがこれ。`.aimodel` を直接読めば準備は 1 秒ほどになる見込み

#### Core AI のキャッシュの鍵

- `main.hash` (`main.mlirb` の sha256) だけで決まる。`.aimodel` を別の場所に写しても 0.02 秒で読める
- キャッシュに入ったあとは、`main.mlirb` を消しても、同じ大きさのゼロにしても読めて、答えも同じ (`aicheck run -cpu -ref`、`milcpu` とのコサイン 1.000000)。`main.hash` を変えると読めない
- キャッシュのフォルダは OS のビルド (`26B123`) と実行ファイルの名前ごと。OS を更新したときや別の実行ファイル (`aicheck`、`chat.test`) では、`main.mlirb` からもう一度コンパイルする

#### Core AI のキャッシュの方針 (`AIModelCache.Policy`、SDK の CoreAIDelegates)

| 方針 | 消す条件 | 置き場所 | 入れ方 |
|---|---|---|---|
| 既定 | `storagePressure` (1) と `sourceAssetChangedOrDeleted` (2) | `~/Library/Caches/coreai-cache/` | `AIModel(contentsOf:)` (今の shim) |
| `.persistent` | なし (0) | `~/Library/Application Support/coreai-cache/` | `AIModel.specialize(contentsOf:options:cache:cachePolicy:)` |

- `AIModel(contentsOf:)` はどちらのエントリも使う (`main.mlirb` のない写しでも 0.02 秒で読める)
- 16.173 で空きが 12 GiB まで減ったあと、LFM3-1.7B のエントリ (12 時台に作った分) の一部が消え、次の読み込みでコンパイルし直しになった (48 秒)。それより古い PLaMo のエントリは残っていたので、使っていない順ではない。`storagePressure` の掃除と見ている (ログには何も出ていなかった)
- 元の `.aimodel` を消しても、どちらの方針でも 30 秒の間はエントリが残った。`sourceAssetChangedOrDeleted` がいつ効くかはわからない
- コンパイルし直しの途中で、part6 が 1 回 `nilError` で失敗した (121 秒)。もう一度走らせたら通った
- `.persistent` にすれば掃除でのコンパイルし直しはなくなるが、モデルを消すときに hwnpu が Core AI のエントリも消す (`AIModelCache.deleteEntries(for:)`) ことになる。採るかはまだ決めていない

#### 重みの共有: 関数 2 つの `.aimodel`

`ailower.LowerFunctions` (`aicheck merge OUT.aimodel f8=PKG f128=PKG`) は、1 本の `.aimodel` に `graph_v1` を名前を変えて 2 つ入れ、同じバイトの重みの資源は 1 回だけ書く。LFM3-1.7B の part2 (8 トークンと 128 トークン、重みは同じ 179 MB) で:
- `.aimodel` は 176 MB で、1 本分と同じ (別々なら 352 MB)
- Core AI のキャッシュのエントリも 1 つで 192 MB (`resources.bin` が 1 つ。別々なら 176 MB と 191 MB)
- 両方の関数とも全部 Neural Engine に載り、`milcpu` とのコサイン 1.000000。初回のコンパイルは 2 つの関数で 7.6 秒
- shim は `PROGRAM.aimodel#NAME` で関数 NAME を開く。入出力は関数ごとに `hwnpu-io-NAME.json`。Core AI のキャッシュの delegate のフォルダは関数によらず `main-this-delegates` で、配置は manifest の `Entry Function Attributes` の、その関数のエントリの `mps.fullyPlacedOnANE` で見る
- 1 本だけ下ろすときのバイト列は前と同じ (`main.hash` が変わらず、作った `.aimodel` のキャッシュはそのまま使える)

#### 配る形 (LFM3-1.7B の見積もり)

| | 落とす量 | 手元に要る量 |
|---|---|---|
| 今 (元のチェックポイントから変換) | 3.5 GB | 約 10 GB |
| 関数 2 つの `.aimodel` を配る (既定) | 約 1.8 GB (`.aimodel` 1.24 GiB、表 0.52 GB、ほか) | 約 3.2 GB (落としたもの 1.9 GB と Core AI のキャッシュ 1.3 GB) |
| 同じで、`.aimodel` を消す (最終手段のオプション) | 同じ | 約 1.9 GB (表など 0.6 GB と Core AI のキャッシュ 1.3 GB)。キャッシュが消えたら `.aimodel` を落とし直す |

- 一緒に配るもの: `tables.safetensors`、`parts.json`、`attn` と `mass` の小さな `.mlpackage`、`tokenizer.json`、`config.json`
- 要る手直し: `parts.json` が `.aimodel#NAME` を指せるようにする。runner が出力の形を `.mlpackage` ではなく `hwnpu-io-*.json` から読む。`.aimodel` をモデルのフォルダに置く (hwnpu のキャッシュの掃除の対象にせず、sha256 もとらない)
- `.aimodel` を消すオプションは、Core AI のキャッシュにあることを確かめてから消し、読み込みに失敗したら落とし直して読み直す

残り: チャットのモデル全体 (7 つの部品 × 2) を関数 2 つの `.aimodel` で動かす。M2 と M3。OS の版を越えて同じ `.aimodel` が読めるか。`.persistent` を採るか。

### 16.175 `probe resident` で、エラーにならない追い出しも見る (順に回す時間、Neural Engine のメモリ) — 2026-11-13

- 動機: M2 (16 GB、macOS 27.2) で `probe resident` が、開いたままでも閉じても 0.23 GiB × 42 本 (9.52 GiB) まで失敗なしで `-resident-max-gb` に達した。M4 と Intel (Core Ultra 5 135U) も 16 GB までエラーなし。ただし Apple では、Core ML は載りきらないものをエラーなしで追い出し、次の呼び出しで載せ直す。M1 の境目 (順に回す部分が約 2.9 GiB を超えると 12 倍遅い、16.118) はエラーでなく遅さで出た。16.144 の作りでは、実行は読み込んだ直後の 1 回 (載っていて当然、M2 で 0.008 s 一定) と、最後の全部の 1 回の成否だけで、追い出しが見えない。`held` も重みの足し算で、実際に持っている量ではない。M2 は読み込みが 1 本 1.8〜5.3 秒で、中身の違う合成プログラムは毎回 Neural Engine 向けにコンパイルされていた
- 変更 (`cmd/hwnpu-dev/probe/resident.go`):
  - `-resident-cycle` (既定、開いたままのときだけ): 各読み込みの後に、持っている全部を順に 2 周実行し、2 周目の 1 回の平均と最長を記録する (`cycle_ms`、`cycle_max_ms`)。それまでの最速の平均 (`base_ms`) の 2 倍を超え、かつ 0.5 ms 以上遅ければ遅い (`slow`)。遅くなる前の最大の合計を `fast_bytes`、最初に遅かった合計を `slow_bytes` に。3 回続けて遅ければ順に回すのはやめ (毎回全部の載せ直しになるため)、読み込みは続ける
  - Apple では、最初の読み込みの前からの Neural Engine のドライバの wired メモリの増えた量 (`mem.NPUWired`) を、各読み込みと順に回した後に記録する (`npu_bytes`)
  - `-resident-size` はカンマ区切りで複数 (Intel の確認スクリプトに `resident-large`: Anima の DiT の群と text の大きさの 0.57,1.2、開いたまま、8 GiB まで)。Intel で 0.23 GiB を 16 GB まで積めたので、Anima の 4.6 GB での失敗は合計でなく、1 本の大きさか本物のグラフの形によるかを分ける
  - 報告の表に「full speed in turn」「slower from」、各読み込みに「in turn, ms a call」「slowest ms」「Neural Engine」の列
- テスト: `TestResidentCycle` (エラーを出さず、最後に回した 4 本だけを速く持つ偽の NPU で、4 本まで速く 5 本から遅い、7 本の後は順に回さない、Neural Engine の量)。既存の偽の NPU (呼び出しは一瞬) では遅くならないことも確かめる
- 確かめていないこと: 実機。M1 で 2.9 GiB あたりの `slower from` が出れば測り方が正しい。M2 と M4 の境目、Neural Engine の量が重みの足し算とどこで離れるか

### 16.176 Chroma1-Flash を Core AI で、`.aimodel` の重みを捨てるオプションと、空の `.aimodel` を安全に開く — 2026-11-13

16.169 の Core AI のバックエンドで、Chroma1-Flash (16.170 の Core ML の書き出しそのまま) を動かした。`hwnpu-image generate -m chroma --backend coreai` (または `HWNPU_BACKEND=coreai`)。これまで Chroma はいつも Core ML でデバイスを開いていたので、Anima と同じく `llm.DeviceBackend` で選ぶようにした。部品を回す側 (`runner`) はバックエンドを選ばない。M2 (16 GB、macOS 27.2、26B123) で、512²、狐のプロンプト、種 1 で測った。

- **全部 Neural Engine に**: 33 本 (T5 の 8 本、approximator、DiT の 10 本、VAE の 10 段) がどれも `mps.fullyPlacedOnANE`。lut8 の線形、T5 の縮尺の工夫 (16.170 の 2 と 3)、VAE の GroupNorm は、Core AI でも下ろし方を変えずに通った。DiT の最後の組 (`dit_single_31`) の `aicheck run -cpu` は milcpu とコサイン 0.999998
- **絵**: 同じ種の Core ML との最後の潜在はコサイン 0.9976、画像の PSNR は 35.4 dB で、同じ絵。Core AI の実行どうしは潜在がバイト単位で同じ
- **時間**: 初回 (コンパイル込み) は 1 枚 12 分 45 秒 (T5 の 8 本で 117 秒、DiT の組 1 本 28 秒〜1 分 11 秒、VAE の 1 段 3〜31 秒)。2 回目から 1 枚 **1 分 48 秒** (読み込みは 1 本 0.6〜1.7 秒で全部で 18 秒、T5 8.0 秒、Heun の 1 ステップ 9〜11 秒、VAE 1.0 秒)。Core ML の初回の実行の計算 (読み込みを除く) は 1 分 28 秒で、Core AI の 1 分 30 秒と同じほど。M2 の Core ML の 2 回目は測っていない (ディスクの空きが足りない、下)。M1 の Core ML の 2 回目は 2 分 06 秒 (16.170)
- DiT を持つ間の `mem.NPUWired` は 8.15 GiB (M1 の Core ML は 8.29 GiB)。ほかのプロセスでスワップが 5 GB を超えていた回は、ステップが 17〜41 秒にばらついた

**ディスク**: 16.174 の LFM3 と同じく、重みを 3 回持つ。書き出し (14 GB)、hwnpu のキャッシュの `.aimodel` (14 GB)、Core AI のキャッシュ (14〜17 GB)。チェックポイントの 27.5 GB を足すと約 73 GB。この M2 では空きが約 15 GiB を切ったところで macOS が `~/Library/Caches` を掃除し、Core AI のキャッシュだけでなく hwnpu のキャッシュの根 (モデルと書き出しごと) も消えた。空き 20 GB で終えた 1 回目の後には、Core AI の 33 項目のうち 3 つしか残らず、2 回目はほぼ全部コンパイルし直した (16.174 の `storagePressure` の掃除と同じとみる)。

- **`.aimodel` の重みを捨てる** (`HWNPU_COREAI_DROP_BYTECODE=1`、16.174 の「最終手段のオプション」を下ろした `.aimodel` に): 読み込めて Core AI のキャッシュに項目ができたら、hwnpu のキャッシュの `.aimodel` の `main.mlirb` を空のファイルに置き換える (切り詰めずに rename で置き換える。Core AI がまだ古いファイルを写像しているかもしれない)。既定は 16.174 のとおり残す。Chroma1-Flash の 512² で hwnpu のキャッシュは 28 → 15 GB、DiT の組の読み込みは重みを読まないので 6.5 → 0.5 秒 (`aicheck`)。今の形 (元のチェックポイントから手元で変換する) では、項目が消えてもパッケージから数秒で下ろし直せる。しかし `.aimodel` を Hugging Face に置いて落とす形 (16.174、cache-sharing.md の 10 節) では、手元にパッケージがなく、項目が消えたら `.aimodel` の落とし直しになる (Chroma1-Flash で 14 GB)。それで既定は残す。項目が消えたときの Core AI のコンパイルし直しは、`.aimodel` を残していても同じだけかかる
- **空の `.aimodel` を Core AI に解析させない** (オプションによらず常に): 空の `main.mlirb` を Core AI が解析すると、LLVM が abort してプロセスごと落ちる (`cannot unwrap empty odiec_module_t`)。起きるのは、項目がないときと、項目に `main-this.odix` がないとき (Core AI がコンパイルし直す)。そこで空の `.aimodel` は開く前に項目を確かめる (`hasEntry`: `coreai-cache/<kern.osversion>/<実行ファイル名>/<main.hash>/*/model.aimodelx/main-this.odix` が空でないファイル)。なければパッケージから下ろし直す。項目があっても読めないとき (`main-this-delegates` がないと `loadFunction` が失敗する) は、shim は項目を消して読み直さず (`cai_open` の kind の `0x100`。読み直すと空のバイトコードを解析する)、Go の側で下ろし直して最初と同じように読む。直接渡された空の `.aimodel` で項目がないときは、開かずにエラーにする。項目を消す、`odix` を消す、`delegates` を消す、の 3 通りで、落ちずに下ろし直して同じ値になった。判定に迷うときは「空にしない」「下ろし直す」に倒す。16.174 の配る形で `.aimodel` を消すオプションを作るときも、この確かめ方 (`hasEntry`、`kindNoRetry`) が要る
- 単体テスト: `TestCut` (偽の HOME に項目を作る)

**1024²** (M2、同じプロンプトと種): 33 本すべて Neural Engine。初回は 1 枚 59 分 44 秒 (DiT の組 1 本のコンパイル 2〜6 分で 10 本で約 41 分、VAE の 10 段で 12 分)。2 回目から 1 枚 **6 分 58 秒** (読み込み 19 秒、T5 7.9 秒、Heun の 1 ステップ 48〜50 秒で DiT 1 回約 25 秒、VAE 4.4 秒)。M1 の Core ML の 2 回目は 9 分 17〜27 秒 (1 ステップ 69〜73 秒、16.170)。2 回の潜在はバイト単位で同じで、絵は崩れていない。M2 の Core ML の 1024² は測っていない。直前に `aicheck` で同じ 1024² の組をコンパイルしていた `dit_double_0` だけは 13 秒で読めた (Core AI のキャッシュのフォルダは実行ファイルごとだが、その下のコンパイルは使い回されるらしい)

**512² と 1024² を関数 2 つの `.aimodel` に** (16.174 の `aicheck merge`、`dit_double_0` で): T5 の重みは 512² の書き出しからのハードリンクで同じファイル、VAE の `weight.bin` はバイト単位で同じ。DiT の `weight.bin` は 1024² の方が 1.57 MB 大きい (RoPE の cos と sin)。それでも線形の重みは同じバイトで、`f512=` と `f1024=` を 1 本にした `.aimodel` は 922.6 MB (512² だけで 874 MB、別々なら約 1.75 GB)。Core AI のキャッシュの項目も 1 つで 874 MB。両方とも全部 Neural Engine に載り、元の `.mlpackage` の milcpu とのコサインは 0.999991 (512²) と 0.999989 (1024²)。1 回の計算は 0.319 秒と 1.457 秒で、別々のプログラムの 0.321 秒と 1.489 秒と同じ。初回の読み込み (243 秒) で両方の関数をコンパイルし、もう一方は 0.14 秒で読めた。1 本にすると初回に使わない大きさのコンパイルも払う

**512² を下見に、同じ構図を 1024² で** (試し): 同じ種でも、`Pipeline.Noise` は乱数を潜在の先頭から順に詰めるので、512² と 1024² で初めのノイズの形が別物になり、構図は揃わない。1024² のノイズを (16, 128, 128) に戻して 2×2 の平均を 2 倍し (分散 1)、512² のノイズにして `--noise` で渡した (コードの変更なし)。狐のプロンプトの種 1〜3 で、種 2 は狐の位置、向き、左の茂み、光まで同じ、種 1 は狐が右で木立も似るが向きと大きさが違う、種 3 は 512² が大きな正面の狐で 1024² は小さな狐の別の絵。低い周波数を揃えるだけでは、構図が揃うのは 3 つに 1 つほど。1 枚は 512² で 1 分 36 秒、1024² で 6 分 28 秒。512² の最後の潜在を引き伸ばして途中の σ から 1024² で回す形 (hires fix) も試したが、潜在の双線形の引き伸ばしでトークンの間隔の格子が出た (引き伸ばさない 1024² の潜在からでは出ない)。下見の用途は要らないので採らない

**`.aimodel` を配るときの Chroma** (見積もり): 落とすのは、元の 27.5 GB でなく `.aimodel` の約 14 GB (T5 5.5 GB、DiT 8.3 GiB、approximator 0.5 GB、VAE 0.1 GB)。変換がなく、チェックポイントも持たないので、手元に要るのは `.aimodel` と Core AI のキャッシュの約 28 GB (今は約 73 GB)。DiT と VAE のプログラムは画像の大きさごとに別だが、重みは大きさによらず同じバイトのはず (lut8 の表は重みだけで決まる)。16.174 の関数 2 つの `.aimodel` (`ailower.LowerFunctions`) で 512² と 1024² を 1 本にすれば、重みは 1 回分で配れる (上で `dit_double_0` を確かめた)。T5 は Flash と HD で最後の組を除いて同じなので、組ごとの `.aimodel` を 2 つのモデルで共有できる。空にするオプションは、配る形では落とし直しの代わりに使うことになり、そのときも `hasEntry` と `kindNoRetry` の確かめ方が要る

512² と 1024² の両方を置くと、hwnpu のキャッシュ (書き出し 2 つと `.aimodel`) が 32 GB、Core AI のキャッシュが 26 GB で、空きは 21 GB まで減った。

**できていないこと**: Chroma1-HD。M2 の Core ML の 2 回目との比較。runner が関数 2 つの `.aimodel` を読む形 (16.174 の手直し)。`.persistent` (16.174) にすれば掃除でのコンパイルし直しはなくなるが、Chroma は Core AI のキャッシュだけで 14 GB 以上あり、消える方が空きを守る面もある。Core AI のキャッシュの 14 GB はこちらからは減らせない。

### 16.177 Core AI: 重みの同じ部品を自動で関数 2 つの `.aimodel` に — 2026-11-13

16.174 で LFM3-1.7B に試した重みの共有 (8 トークンと 128 トークンを 1 本の `.aimodel` の関数 2 つに) を、Core AI のバックエンドがどのモデルにも自動で行うようにした (`internal/backend/coreai/group_darwin.go`)。呼び出す側 (chain、chat、PLaMo-Embedding) には手を入れていないので、マージ済みのモデルも次に `--backend coreai` で読み込んだときに移る。

**位置づけ**: これは手元で変換する経路 (チェックポイントから `.mlpackage` に変換し、hwnpu のキャッシュに `.aimodel` を作る) の話で、主流は落としてきた `.aimodel` を動かす形になる (16.174、cache-sharing.md 10 節)。その形とは次のように分ける。
- 落としてきた `.aimodel` (`PATH.aimodel` や `PATH.aimodel#NAME`) は、そのまま開く。ハッシュもまとめ直しもせず、消しもしない。手元に残すのが既定で、Core AI のキャッシュが消されても落とし直さずにコンパイルし直すだけで済む。ストレージが厳しいときに消すのはオプションにする (未実装)。そのオプションでは Core AI のエントリは残す (`cai_forget` は呼ばない)
- 変換の経路でこの節が消すのは、hwnpu のキャッシュにある、前の hwnpu が部品ごとに作った `.aimodel` (と Core AI のそのエントリ) だけ。どれも手元の `.mlpackage` から作り直せる
- まとめた `.aimodel` は配る形と同じ (関数をまとめた 1 本)。配るものを作るときは、これをキャッシュから取り出せばよい。関数の名前は中身の sha256 からとる (`p` + 16 桁) ので、同じ変換なら機械によらず同じ。配る `parts.json` が `.aimodel#NAME` を指すとき、この名前のままにするか、`f8` や `f128` のような読める名前にするかは、配る形を作るときに決める

- **兄弟の見つけ方**: 部品を初めて下ろすとき、同じフォルダの `.mlpackage` のうち、自分の重みのファイル (`weight.bin`) の大きい順 3 つの blob と同じバイトの blob を持つものを兄弟とする。読むのはメタデータの記録と、その大きさの blob だけ。初めはファイル全体が同じものとしたが、PLaMo-Embedding の 128 と 1024 は系列の長さによる小さな表 (458 KB の差) があってファイルが違い、まとまらなかった。`LowerFunctions` は同じバイトの blob を 1 回だけ書くので、違う blob があってもそれぞれの分だけ増える。1 MiB 未満の重み (注意の `attn`、`mass`) はまとめない
- **記録** (`KEY.aifn`、KEY は前と同じ部品の中身の鍵): 部品ごとに、まとめた `.aimodel` と関数の名前 (`p` + 中身の sha256 の 16 桁) を書く。2 回目からは自分の部品をハッシュして記録を読むだけで、兄弟は読まない。まとめた `.aimodel` の鍵は兄弟の中身の鍵の並びで、ファイルの名前によらない。兄弟の見つけ方の版 (`rule`) が違う記録は 1 回見直す。単体と記録された部品も、後から兄弟ができればまとまる
- **前の単体のエントリ**: まとめたとき、前の hwnpu が部品ごとに作った `.aimodel` を消し、その前に Core AI のキャッシュのエントリも消す (shim の `cai_forget`、`AIModelCache.default.deleteEntries(for:)`)。まとめた `.aimodel` が `cache.Prune` で消えたら、記録の残った部品が次に読み込まれたときにまとめ直す
- 同じフォルダを 2 つのプロセスがまとめないように `cache.Lock` を取る

**結果** (M1 16 GB、macOS 27.2、`--backend coreai`、貪欲法)

| モデル | まとまった部品 | 減った `.aimodel` | 減った Core AI のキャッシュ |
|---|---|---|---|
| LFM3-1.7B (8 と 128、7 部品) | 14 | 1.22 GiB (2.43 → 1.22) | 1.21 GiB (2.53 → 1.32) |
| Qwen3.5-0.8B (8 と 128、7 部品) | 14 | 0.77 GiB | 0.76 GiB |
| PLaMo-Embedding-1B (128 と 1024) | 2 | 0.90 GiB | 0.92 GiB |
| PLaMo 3 Translate 0.8B (8 と 128、3 部品) | 6 | 0.69 GiB | 0.72 GiB |
| Qwen3-0.6B (8 と 64) | 2 | 0.58 GiB | 0.59 GiB |
| gemma-4-E2B-it (128 は 3 部品) | 6 | 0.39 GiB | 0.41 GiB |

LFM3-1.7B は前後を測った値、ほかは「まとめた `.aimodel` とキャッシュのエントリの大きさ × (関数の数 − 1)」の見積もり (LFM3-1.7B ではこの見積もりと測った値が合う)。合わせて約 9.3 GiB。Qwen3.5-2B、LFM3-VL、Audio、Omni、PLaMo 3 Translate 2B と 8B は、16 GB の M1 では 1 本のプログラムで動かすので (`onePrograms`) 兄弟がなく、変わらない。

- 答え: 12 モデル (上の 6 つと、変わらない 6 つ) の貪欲法の答えは、まとめる前と同じ。埋め込みの検索の点数も同じ (0.797、0.677、0.630)。どの部品も全部 Neural Engine に載った
- 2 回目からの準備: 1.3〜5.4 秒で、まとめる前 (16.173) と同じほど。生成の速さも変わらない
- まとめる初回は、Core AI がまとめた `.aimodel` をコンパイルし直す。LFM3-1.7B は 1 分 2 秒、Qwen3.5-0.8B は 16 分 (16.173 では単体で 5 分 9 秒。128 トークンの部品が重い)、Embedding は 1 分 11 秒
- 2 回目のバッチ (記録の見直し) でも LFM3-1.7B (32 秒) と Qwen3.5-0.8B (12 分) がコンパイルし直された。まとまり方も鍵も同じで、ディスクの空きが 11〜13 GiB のときに Core AI がエントリを消したため (16.174 と同じ `storagePressure`)

**Neural Engine のメモリ** (M1、まとめる前は origin/main の hwnpu-chat を別の名前と別のキャッシュで。約 300〜450 トークンのプロンプトで 128 トークン (Qwen3 は 64) のプログラムも使い、40 トークンを生成。2 回目と 3 回目の値)

| モデル | NPUWired まとめる前 / 後 | 生成 前 / 後 | footprint の最大 前 / 後 | 2 回目からの準備 前 / 後 |
|---|---|---|---|---|
| Qwen3-0.6B | 1.19 / 0.61〜0.72 GiB | 8.2 / 8.2〜8.4 トークン/s | 2.74 / 2.74 GB | 1.8〜2.6 / 1.8〜2.3 秒 |
| PLaMo 3 Translate 0.8B | 1.43 / 0.75 GiB | 34.8〜41.2 / 42.6〜43.5 | 0.94 / 0.94 GB | 1.2〜1.3 / 1.5 秒 |
| LFM3-1.7B | 2.56 / 1.34 GiB | 22.8〜22.9 / 25.8 | 1.01 / 1.01 GB | 3.5〜4.3 / 4.2〜4.3 秒 |

- 1 本の `.aimodel` を関数ごとに開いても (`AIModel` が 2 つ)、Neural Engine は重みを 1 回分しか持たない。NPUWired はまとめる前の約半分で、重みの 1 回分 (LFM3-1.7B は 2.43 GiB の半分) とほぼ同じ。まとめる前は 8 トークンと 128 トークンのプログラムがそれぞれ重みを持っていた
- プロセスの footprint は変わらない (重みは Neural Engine のドライバのメモリにあり、footprint に入らない)
- 生成は LFM3-1.7B で 13 %、PLaMo 0.8B で 5〜20 % 速くなった。LFM3-1.7B のまとめる前の 2.56 GiB は、M1 で遅くなり始める約 2.8 GiB (16.118) の近くにある
- 測る前の NPUWired はどれも 0.002〜0.12 GiB で、プロセスが終わると戻る
- まとめる前の 3 モデル分のキャッシュは hwnpu の 5.0 GB と Core AI の 5.1 GB で、測ったあとに消した

**Core AI ではプロンプトを読むプログラムも持つ** (`chat.Prepare`): Neural Engine が持つ量の見積もり (`onePrograms`) は 128 トークンのプログラムの重みも足していて、Core AI では重みを 2 回数えていた。そのため 16 GB の M1 では、Qwen3.5-2B、LFM3-VL、Audio、Omni、PLaMo 3 Translate 2B が 1 本のプログラムに絞られていた。端末が Core AI のとき (`sharesWeights`) は重みを 1 回分と数え (`heldBytes`)、プロンプトを読むプログラムを落とさない。Core AI がコンパイルしたものは自分のディスクのキャッシュに置くので、Neural Engine のコンパイルのキャッシュ (`compileCacheBytes`) の上限も当てない。`--prefill off` と、モデルの定義の `prefill: off` はそのまま効く。Core ML はこれまでどおり。

M1、約 300〜400 トークンのプロンプト、40 トークンの生成、2 回目と 3 回目の値 (前は 1 本のプログラムで、8 トークンずつ読む)

| モデル | プロンプトを読む 前 / 後 | 生成 前 / 後 | NPUWired 前 / 後 | 2 回目からの準備 前 / 後 |
|---|---|---|---|---|
| Qwen3.5-2B (lut6、375 トークン) | 2.4〜2.6 / 1.0 秒 | 21.0〜21.4 / 20.6〜21.7 トークン/s | 1.76 / 1.90 GiB | 1.6 / 2.6〜3.2 秒 |
| LFM3-VL-2B (lut8、403) | 2.5〜2.8 / 0.8〜0.9 秒 | 29.4〜29.6 / 28.9〜29.2 | 2.02 / 2.12 GiB | 5.1 / 6.2〜6.8 秒 |
| LFM3-Audio-2B (lut8、403) | 2.5〜2.7 / 0.8〜0.9 秒 | 34.0〜34.3 / 33.3〜33.6 | 1.93 / 2.03 GiB | 4.0 / 5.9 秒 |
| LFM3-Omni-2B (lut8、403) | 2.6 / 0.9 秒 | 36.6〜37.3 / 36.4〜36.6 | 2.28 / 2.39 GiB | 4.1 / 6.2〜6.3 秒 |
| PLaMo 3 Translate 2B (lut8、300) | 3.4〜3.6 / 1.1 秒 | 11.3〜12.0 / 10.7〜11.2 | 2.48 / 2.57 GiB | 2.2 / 5.4〜5.8 秒 |

- プロンプトを読むのは 2.3〜3.2 倍速い。生成の速さは変わらない (±5 %)。貪欲法の答えの書き出し (150 字) は前と同じ (全文は比べていない)
- Neural Engine が持つ量は 0.1〜0.14 GiB 増えるだけで、どれも M1 の約 2.8 GiB の下にある。前の見積もり (2 本で 3.26〜4.90 GiB) では、5 つとも上限を超えるとしていた
- 準備は 1〜3.6 秒長くなる (2 本目の読み込み)
- 初回: 128 トークンのプログラムの変換と、まとめた `.aimodel` の Core AI のコンパイル。LFM3 の 3 つは 53〜54 秒、PLaMo 2B は 1 分 15 秒、Qwen3.5-2B は 16 分 48 秒。変換にチェックポイントが要るので、消していた PLaMo 2B は落とし直した (6 分 18 秒)。初回の生成は遅く (7.6〜18 トークン/s)、NPUWired は 3.2〜5.0 GiB まで上がった。コンパイルしたばかりのプログラムが初めて呼ばれる回だけで、2 回目からは上の値

配る `.aimodel` (16.174) は 8 トークンと 128 トークンを関数 2 つにまとめるので、Core AI ではこの形がどのメモリの機械でも既定になる。

**画像のモデル: 隣の書き出しの部品もまとめる** (`candidates`、`rule` 3): 画像のモデルは大きさごとに別の書き出し (`anima-turbo-v1.1-1024x1024-…`、`…-768x768-…`) を同じ親フォルダに並べ、部品の名前は同じ。兄弟を探すのを、同じフォルダに加えて隣のフォルダの同じ名前の `.mlpackage` (`親/*/text.mlpackage` など) にも広げた。プログラムの形は大きさで違うが、線形の重みは同じチェックポイントなら同じバイトなので、大きさごとの関数を 1 本の `.aimodel` に入れる。兄弟が増えたら (3 つ目の大きさ)、全員でまとめ直し、前の小さいまとまりは消す。Anima、Chroma、Qwen-Image、Sana の書き出しはどれもこの並び。

Anima (M1、`--backend coreai`、8 ステップ、シード 1。Turbo 1024²、aesthetic 1024²、Turbo 768² の 3 つの書き出し)

| 部品 | まとまり | 関数 |
|---|---|---|
| `text.mlpackage` (0.53 GiB) | Turbo 1024²、aesthetic、Turbo 768² | 3 |
| DiT (`dit_blocks_*` 7 本 0.27 GiB ずつ、`dit_time`、`dit_final`) | Turbo の 1024² と 768² | 2 |
| DiT (aesthetic) | 単体 (チェックポイントが違い、重みが違う) | 1 |
| `vae_s0`〜`s4` | 3 つとも (VAE は共通) | 3 |
| `vae_s5`〜`s9` | 単体 (1 MiB 未満) | 1 |

- hwnpu のキャッシュの Anima の `.aimodel` は 4.32 GiB。単体なら約 7.36 GiB で、約 3.0 GiB 減った (見積もりは上と同じ)。Core AI のキャッシュ (`hwnpu-image`) は 4.2 GB
- 画像: 前の hwnpu (769b0ba) の Turbo 1024² と、まとめた後の Turbo 1024² (1 回目、2 回目、768² を足した後) の PNG はバイト単位で同じ。768² も 2 回とも同じ。最後の回で 20 部品が全部 Neural Engine に載った
- 2 回目からの読み込み: 3.5〜3.9 秒。768² を足した後の 1024² は 18 秒 (まとめ直した `.aimodel` を開くだけ)
- 初回のコンパイル: 768² の書き出しを足した回が 36 分 10 秒 (DiT 9 本を関数 2 つで、text を関数 3 つでコンパイルし直す)。aesthetic の初回 (DiT は単体) は 23 分 29 秒、前の hwnpu で部品ごとにコンパイルしたのは 24 分 22 秒
- 画像のモデルの Neural Engine のメモリはここでは測っていない。1 回の生成で開くのは 1 つの大きさだけで、同じ時に開く関数は 1 つなので、チャットのような NPUWired の減りはない。減るのはディスク

残り: Qwen-Image (1024² と 512²) と Chroma (16.176 で Core AI に。512² と 1024² の DiT を手で関数 2 つにして 922 MB) は、このマシンに大きさ 2 つの書き出しがなく、自動でのまとまり方を確かめていない (仕組みは Anima と同じ)。16.176 の重みを捨てるオプション (`HWNPU_COREAI_DROP_BYTECODE`) は、まとめた `.aimodel` にもそのまま効く。空にした `.aimodel` の Core AI の項目が消えていたら、記録を読むときに消してまとめ直す (`usable`)

### 16.178 gpt-oss-20b を AMD の NPU で: MXFP4 のままの専門家 (mx4)、注意の sink、プロンプトの専門家を行列演算器で — 2026-11-13

openai/gpt-oss-20b (`gpt_oss`、21B、トークンあたり 3.6B) を AMD の NPU (Ryzen AI 9 365) に入れた。docs/new-model-checklist.md の順に進めた。Mac の Neural Engine は MoE の経路 (gather) が載らないので形を作らない (3 章は対象外)。Intel も作らない。

**構造** (transformers 5 の modeling_gpt_oss.py、`tools/ref/ref_gptoss.py` で照合)

- 24 層。`sliding_attention` (自分を含めて 128 位置) と `full_attention` が交互。q は 64 頭、KV は 8 頭、頭の次元は 64。q、k、v、o に bias。YaRN (factor 32、theta 150000、cos と sin に attention_scaling 1.3466)。頭ごとに学習した sink のロジットが softmax に入り、出力からは落ちる
- RMSNorm は重みを掛けるだけ (`1 +` なし)、eps 1e-5
- 専門家は 32 個から 4 個。router は bias つきの線形、選んだ 4 個のロジットの softmax。gate と up はチェックポイントで行が交互。gate は 7 で上を切り、up は [-7, 7] に切る。(up + 1) × gate × σ(1.702 gate)。down にも bias
- 専門家の重みは MXFP4 (e2m1 の 4 ビットと、出力 1 つの入力 32 個ごとに 2 の冪のスケール)。ほかは bf16。語彙 201088、出力層は埋め込みと別

**参照** (`tools/ref/ref_gptoss.py`): float32 で 84 GB になり 60 GB の機械に入らないので、層を 1 つずつ読む (専門家は transformers の `convert_moe_packed_tensors` と同じ式で、丸めずに戻す)。transformers のモジュールに同じ重みを入れて比べると、2 層 160 トークン (窓 128 を越える) で層ごとの相対誤差 1.3e-7、ロジット 3.0e-7。窓を 1 つずらすと 4.9e-4 で FAIL になることを確かめた

**トークナイザ** (`internal/tokenizer`): o200k_harmony の前処理 (大文字の列のあとの小文字の列で切る、縮約を語に付ける、数字は 3 桁ずつ、記号の列のあとの `/`)。固定の 40 件と乱数の 20000 件で HF と不一致 0。上の文字の列を戻す処理か `/` を外すと失敗する

**チャットの形式** (`internal/chat/gptoss.go`、harmony): system の推論の強さ (`NoThink` で low、既定は medium)、system メッセージを developer の指示に、道具は TypeScript の名前空間に (テンプレートの `render_typescript_type` のとおり、入れ子の object の型の前に改行と 16 文字の空白が入るところも)。jinja2 で描いた 11 件 (道具、既定値、配列、呼び出しのあとの答え、日付を固定) と 1 文字ずつ一致。分析 (analysis) のチャネルを考えた部分に、final を答えに、`to=functions.NAME` の commentary を道具の呼び出しにする。止めるトークンは `<|return|>`、`<|call|>`

**専門家は MXFP4 のまま** (`tools/amd/iron_gptoss_moe.py`、`kernels/moe_mx4.cc`)

- q4nx (32 個ごとのスケールと最小値の一様な 16 段) に丸め直すと重みの誤差 6.6 %、最大値で決める q8 でも 0.55 %。e2m1 の 2 倍は -12〜12 の整数でスケールは 2 の冪なので、q6 にはそのまま入る
- mx4: q8 と同じブロック (出力 32 × 入力 320) に bf16 のスケール 2^(E-128) と e2m1 の 4 ビット、ブロックの bias 32 個 (最初の k ブロックのもの)。1 重み 4.55 ビット。カーネルで v = 符号 × max(m, 2m - 4, 4m - 16) に戻す (m は大きさの 3 ビット)。32 個の部分和は bf16 の 2 つに分けてからスケールを掛ける (q6 と同じ)
- 2880 = 9 × 320 なので入力は埋めない。gate と up の出力は 3072 に埋める (専門家ごとに 96 組、列 2 つ × 4 コアに 12 組ずつ)
- CPU で測った KL (5 つの文、400 位置、`ref_gptoss.py sweep`): 専門家を q4nx にすると README 0.052 → 0.135、GPL 0.018 → 0.031、wiki-ja 0.043 → 0.107、wikitext 0.025 → 0.057、Go 0.140 → 0.158

**注意の sink** (`tools/amd/iron_sinks.py`、`kernels/mha_sinks.cc`)

- IRON の MHA の P×V のコアが最後の割り算 (rescale_O) の前に、行ごとの和 l に sink の項 2^(sink log2 e - m × scale log2 e) を足す。m はスコアの最大値 (スケール前)。仕事もキャッシュも増えない。MHA の build の P×V の worker のプログラムだけを差し替える (IRON の build のクロージャから値を読む)
- 最初は softmax の状態を sink で始めた (m = sink、l = 1)。sink が大きい頭は、どのスコアの指数も 0 より大きく下になり、装置の exp2 がそこで最大 2 倍高く、出力が 1.5 倍になった。割り算の前に足す形に変えた
- IRON のコンパイルのキャッシュは演算子の引数で引くので、build を変えても古い設計が動き、同じ誤った数字が 2 度出た。形の引数 (`sink_form`) を足した。入っている `mha.cc` は softmax の状態を float32 で、最大値はスケール前で持つ (最初に読んだ mlir-aie のチェックアウトは bf16 でスケール後)
- 照合 (64 頭、8 KV 頭、512 キー): デコード (詰めた形) と 256 クエリのプロンプト、窓 127 のそれぞれで HF の式とコサイン 0.999988 以上。sink 6 でも 0.999987

**グラフ** (`tools/amd/iron_gptoss.py`、`iron_export_lm.py gptoss`)

- 残差を 3072 に埋める (2880 は 64 のブロック 45 個で 8 列に割り切れず、量子化した GEMV は入力が 512 の倍数)。norm の重みに sqrt(2880 / 3072)、eps に 2880 / 3072 を掛ける (同じ出力)。出力層の行は 201216 に (列ごとに 393 ブロック)、hwnpu は先頭の 201088 個のロジットを読む
- q、k、v を 1 つの GEMV と bias の加算 1 つに (プロンプトの塊は重みの行の区間ごとの GEMM)。注意の射影は q8、出力層は q4、router は bf16。埋め込みはホスト (1.2 GB を固定しない)。窓の層は窓の中だけ読む (`iron_window`)
- デコード: 層が 2 コマンド (A: 注意と router と MoERoute、B: MX4GateUp と MX4Down)、Qwen3.6 と同じく経路はデバイスで選ぶ (`iron_moe.MoERoute`、32 個から 4 個)。B のスクラッチパッドに専門家の位置を書く
- プロンプトの塊 (256): 経路はホスト (Go の `moeHostRouteMX4`、x を専門家ごと k ブロックごとの行の順に)、`MX4GroupGateUp` と `MX4GroupDown` が全部の専門家の重みを 1 回ずつ流し、専門家の行 (1 回に最大 128) を通す。表はデコードと同じもの (並べ方を [組][k ブロック][gate/up][コア] にして両方が順に読む。DMA の 1 次元は 1023 語までで、並べ替える読み出しは通らなかった)。行は 1 回に 8 行ずつ、AIE2P の bf16 の行列演算器 (`mac_4x8_8x8`) で: 符号を戻してスケールを掛け (2 の冪なので正確)、8 × 8 の B の塊 4 つに分け、4 行の A の塊がそれぞれに掛かる。ブロック全体を float32 で足す。ベクトルの mac の版 773 ms → 462 ms (1 回、32 専門家、約 2100 行)、平均の誤差 1.2e-3 → 5.5e-5。列ごとに 4 コアにして x を読み直す回数を半分に: 241 ms
- 1 層の専門家に 128 行を超える塊は回に分ける。トークンは、その 4 つの専門家に空きのある最初の回に入れる (順に詰めると 256 トークンで 3 回、詰め直すと 2 回)
- ホストが読む専門家の入力 hn は状態に写す。写さない版は 2 層で正しく 24 層で崩れた (コサイン 0.98): 領域があとの演算に hn を渡していた (16.110 と同じ落とし穴)
- llvm-aie: 早い return のある小さなループで「LoopStart/LoopBody: insufficient padding」で落ちた (分岐なしに書いた)。gate-up のコアのプログラムが 16 KB を越えた (行列演算の関数を noinline、最後のループを展開しない)

**結果** (Ryzen AI 9 365、balanced。書き出し `~/models/gpt-oss-20b.amd-npu2`、文脈 8192、塊 256、12.6 GB の像)

| | |
|---|---|
| HF fp32 との照合 (32 トークンと 3 ステップ) | 最上位 4/4、コサイン 0.9968〜0.9997 |
| 2 層で塊の大きさごと (64、128、256 トークン) | すべて最上位が一致、コサイン 0.9967〜0.9971 |
| KL (NPU、256 トークンを読んだあと 144 位置、HF fp32 に対して) | 出力層を GPTQ で: README 0.055、GPL 0.0065、wiki-ja 0.021、wikitext 0.023、Go 0.030 (最上位の一致 0.87〜0.98)。丸めだけなら 0.060、0.0074、0.027、0.027、0.032 |
| デコード | 54 ms/トークン (18.4〜18.6 トークン/秒、下の「デコードを常駐の 1 配列に」の後。文脈 2048 で 58 ms。その前は 79.2 ms、最初は 97 ms)。層のコマンド 1.80 ms (全体を見る層は文脈 2048 で 2.12)、最初のコマンド 3.4 ms、最後 (専門家と出力層) 9.3 ms |
| プロンプト | 256 トークン 1.8 s (7 ms/トークン: bfp16 の専門家、スロット、塊を 1 回戻す。bf16 の行列演算器で 5.4 s、最初は約 6 s)。1 トークンずつの版は 106 ms/トークン |
| 固定するメモリ | NPU のバッファ 14.1 GiB + RSS 0.5 GiB (常駐のデコードの表と KV の配列で +1.06 GiB) |
| 電力量 (`HWNPU_ENERGY`、package) | 90 トークンを読んで 600 トークンを書く 62 s で 779 J (12.6 W、1.3 J/トークン、待機を除くと 0.65〜0.85 J)。2 回とも同じ |

- hwnpu-chat: 「富士山の標高は 3,776 メートル」「17 × 23 は 391」、『こころ』のあらすじ 3 文、「日本で最も長い川は信濃川」(流域や 2 番目の川の説明は誤り。bf16 の CPU で同じ答えを作るのは 1 トークンに全層を読み直すので試していない)。考える部分は分析のチャネルに出る
- hwnpu-chat agent (16.66 の課題、`-y`): 2 つのファイルを読み、`go test` で落ちることを見て、`math.Pi * c.R` を `math.Pi * c.R * c.R` に直し、`go test` が通ることを確かめた。最初のプロンプト 560 トークン 14.3 s、2 回目からは新しいトークンだけ読む
- 書式の比較 (CPU、`ref_gptoss.py sweep`): 注意を q6 にすると KL が 2〜3 倍 (README 0.133) なので q8 のまま。出力層の q6 と q8 は q4 より README 0.052 → 0.045 / 0.044、wiki-ja 0.043 → 0.034 / 0.033

**GPTQ** (`tools/amd/gptoss_gptq.py`、層を 1 つずつ、wikitext-2 と wiki40b-ja の 64 × 512 トークン、damp 0.1、25 分)

- 注意の q8 (層ごとに q、k、v を重ねたものと o_proj) と出力層の q4 を書き出しと同じ形 (埋めて重ねた行列) で選び、`--q4-codes` で 49 個すべてが選んだ符号を使った
- 全部を GPTQ にすると README 0.060 → 0.038、wikitext 0.027 → 0.020、Go 0.032 → 0.026、GPL は同じだが、wiki-ja が 0.027 → 0.066 (中央値は 0.0115 → 0.0057 なのに、3 つの位置が KL 1.6〜2.9)。出力層だけにすると 5 つの文すべてで丸めより良い (上の表)。出力層だけを採った (注意の q8 は丸めのまま)
- models.yaml の書き出し (`amd_export`) は校正をしないので出力層も丸め。符号は `gptoss_gptq.py` で作り、`--q4-codes` で渡す

**部品として組むときの値** (チェックリスト 10 章): 固定は 13.0 GiB の 1 部品 (像 4 バンク)、読み込み 5.5〜8 s (2 回目、OS のキャッシュに像がある)。最長のコマンドはプロンプトの層の専門家 (1 回 約 0.25 s、回が 2 つなら 2 回) で 2 s の打ち切りに遠い。入出力は文 (o200k、日本語と英語)、道具の呼び出しは harmony の JSON。答えは流せる (最初の分析のトークンまでがプロンプトの時間)

**models.yaml**: `gpt-oss-20b` (AMD だけ、memory 14.5 GB、machine 32)。`chat` の用途の候補には入れない (Mac にも Intel にも形がない)

**デコードの演算をまとめる** (FastFlowLM の gpt-oss-20b が同じ世代の NPU で 19 tok/s。配布物 `FastFlowLM/gpt-oss-20b-NPU2` は 14.45 GB で、専門家は MXFP4 のまま (`dequant_mxfp4.xclbin`): 量子化は小さくなく、差は演算の数と DMA の使い方と見た)

- 注意の A は 1 層 15 演算 (130 µs ずつ)。`iron_q4.add_operator` (QGEMVAdd、`kernels/mv_add.h`): GEMV の出力がコアを出るときに 1 つか 2 つのベクトル (bias、残差) を足す。ベクトルの列の分は x のチャネルで x の後ろに載せる (コアの入力は 2 チャネル)。1 行の `F.linear` の bias はこれに (`FUSE_BIAS`)、層の残差は `into_linear` で最後の射影 (o_proj、out_proj、down_proj) に
- gpt-oss: q、k、v の bias と o_proj の bias と残差を GEMV に、q と k を 1 つの RoPE に (出力で隣り合う 72 行)、router の bias を MoERoute に (プログラムの語の後ろに bf16 で)。A は 15 → 10 演算、1.95 → 1.5〜1.6 ms。2 層で HF と同じコサイン
- QGEMVAdd の照合 (q8、5120 × 3072 に bias、3072 × 4096 に bias と残差): コサイン 1.000000

**Qwen3.5-0.8B に同じことを** (`iron_qwen35.Qwen35Layer.body`、`iron_q4.glu_operators`)

- デコードのステップは 437 演算: 残差の加算 48、MLP の gate の GEMV、up の GEMV、SiLU、積が 24 層
- 残差を GEMV で足す (48 演算減): 87 → 77.5 ms/トークン
- gate と up を 1 つの表に (`glu_param`: 64 行の列ブロックを gate、up と交互に。表は 1 つで重みは増えない)。デコードは `QGEMVGlu` 1 演算で silu(g) u まで (コアが gate と up の同じ出力 32 個の累算を持つ。silu は exp_f32)。プロンプトの塊と草稿の検証は、同じ表の gate か up の列ブロックだけを読む (DMA の 1 次元は 1023 語までなので、1 列ブロックを 832 語ずつの 3 次元で): 演算の数はそのまま。77.5 → 73 ms/トークン
- 合わせて 87 → 73 ms/トークン (11.5 → 13.7 トークン/秒、−16 %)。前の書き出しのロジットに対してコサイン 0.99987 以上、最上位はすべて一致、`ironlm -verify` も PASS (デコードは残差を丸める前に足すので、検証の行と完全には同じでない: コサイン 0.99993 以上)
- q4 の MLP (Qwen3.5-4B) はまとめない (プロンプトの DequantBFP に列ブロックを選ぶ形がない)。GPTQ の符号は gate と up の重みで引くので、まとめた表は丸め

**専門家の読み出し** (デコードのステップの B)

- 計算をしないカーネル (`MX4_NOCOMPUTE=1`) で測ると、転送だけで 2.05 ms、計算込み 2.16 ms: DMA で頭打ち。gate-up 41 MB が約 1.2 ms、down 20 MB が約 0.8 ms で、出力層の GEMV の 54 GB/s なら 0.76 ms と 0.37 ms: 演算 1 つに約 0.4 ms の固定の分 (流れの始まりと終わり)
- `MX4Experts`: gate-up と down を 1 つの演算に (コアのプログラム 1 つ、重みの流れ 1 つ: gate-up の表の半分、続けて down の列の分。x の流れ 1 つ: x、続けて活性と列の残差と経路の重み)。活性は出力 y の前半に書いて DRAM を回り、後半の段の x として読む (`fill` の `source`)。down の重みは gate-up の段の終わりを待たずに積む。2.16 → 1.84 ms (照合のグラフ)
- ブロックのスケールを bf16 から MXFP4 の指数のバイトのままに (1 ブロック 5824 → 5504 バイト、1 重み 4.55 → 4.3 ビット、カーネルで bf16 の指数の欄に入れる)。1.89 → 1.81 ms。像は 0.8 GB 小さく
- 列ごとに 2 コアなら転送だけは 1.78 ms と速いが、計算で 2.39 ms。memtile の物の数を 3〜4 にするとチャネルの記述子 (24 個) が足りない
- 合わせて、デコード 89.5 → 80.5 ms/トークン (B 2.0 → 1.7 ms)。照合 (32 トークンと 3 ステップ) は同じ

**プロンプトの専門家を bfp16 で、演算の切り替えの費用**

- AIE2P の bf16 の `mmul<4,8,8>` は API の中で放送 mac (vmac.f) に展開されていた (aie_api/detail/aie2p/mmul_bf16_bf16.hpp、`mac_4x8_8x8_bf16` は emulated_mmul_intrinsics)。浮動小数の行列命令は bfp16 だけ (`mac_8x8_8x8T`、1 命令 512 積)。符号を戻すのを省いた測定版 (`MX4_FASTDECODE=1`) でも 189 → 168 ms で、戻すのは 1 割しか占めていなかった
- `MX_BFP` (moe_mx4_rows.cc、既定。`MX4_BFP=0` で bf16 の版): k 刻みごとに 8 行 × 8 入力を bfp16ebs8 に (行の 8 入力が指数を共有。活性は 8 ビットの仮数に丸まる。AMD のプレフィル GEMM と同じ)、重みは 8 出力 × 8 入力の塊を直接組む: bfp16ebs8 は {v64int8 の仮数、v8int8 の指数} で値は仮数 × 2^(指数 − 133) (aie.utils.bfp) なので、仮数 = v = 2 e2m1、指数 = グループの E + 5。1 つの指数を共有する 8 値は 1 出力の 8 入力で、MXFP4 の 32 入力のグループの中: 重みは正確。q4nx_dequant.cc と同じ並び (16 出力の 8 入力を 1 回の 512 ビットの読み込み、偶数語が出力 0〜7) から 8 × 8 の転置
- 照合 (乱数の重み、0〜128 行): 189 → 124 ms (float を通して組む版) → 84 ms (直接組む) → 82.6 ms (int8 の upshift を加算に)。コサイン 0.99992。256 トークンの塊らしい行数 (`MX4_COUNTS=chunk`、平均 32 行): 113.5 → 56.8 ms。転送だけなら 37 ms
- 24 層: プロンプト 256 トークン 5.4 → 3.4 s (21 → 13 ms/トークン)。デコードは変わらず。照合 PASS (コサイン同じ)。KL (256 トークンの後の 144 位置): readme 0.055 → 0.066、gpl 0.0065 → 0.0075、wiki-ja 0.021 → 0.017、wikitext 0.023 → 0.030、go 0.030 → 0.031 (5 つの平均 0.027 → 0.030)。速さを採った
- 演算の切り替えの費用 (デコード): 同じ演算を続けて 2 回にすると 2 回目は 1.03 ms (57 MB を 55 GB/s、ほぼ上限)、1 回目は 1.78 ms。切り替えは 1 回あたり約 67 µs と、読み込む構成 (PDI) 1 KB あたり約 1.2 µs (専門家のカーネルに使わない 6 KB を足すと PDI +218 KB で +0.31 ms)。専門家の PDI は 311 KB (コア 32 個 × プログラム 7.8 KB と初期設定 55 KB)。デコードの 1 トークンで切り替えは約 30 ms (80 ms のうち)
- 採らなかったもの: 専門家のコアのプログラムを小さく (glu を 16 個ずつ 1 つのコピーに、累算を 1 つのバッファにして `range_` で回す): 7.8 → 7.6 KB、1.78 → 1.80 ms。IRON はループを ObjectFifo の深さで展開するので小さくならない。KV キャッシュへの書き込み (`cache[pos] = x` の Copy 2 つ) を GEMV と RoPE から直接に: 出力を 2 つに分ける必要があり、切り替え 1 回 (約 1.6 ms/トークン) のため見送り

**プロンプトの専門家のスロット、塊を 1 回戻す、デコードの QGEMVQKV**

- 転送だけの測定で CAP 128 → 64 は 37 → 21 ms: x と活性のバイトが床 (1 列で x と活性 141 MB、重み 57 MB)。塊の行は平均 32 なのに、x と活性は専門家ごとにいつも CAP 行を流していた
- スロット (`SLOTS = 40`、`CAP = 56`): スロット e < 32 は専門家 e の最初の CAP 行、残りの 8 は行の多い専門家の続きの行 (その重みの位置は呼び出しごとの値、グラフの `slot<i>`、hwnpu の `moeHostRouteMX4` が書く)。全部のスロットに値を持たせられないのは、コマンドのスクラッチパッドが 32 語まで (コマンドプロセッサの表) だから。8 つで足りない塊は 2 回に分ける (24 層で 1 層ほど)
- 重みの塊 (gate と up の組) を最初に 1 回だけ bfp16 のタイルに戻してコアに置き (`mx4_rows_dec2`、1 塊 10.5 KB、指数はグループごと)、W の物はすぐ返す。スロットの全部の 8 行の物がそれを使う。64 行では累算と戻した組がコアに入らず 56 行に
- ホスト: x は使う行だけ領域に直接書き (12.9 MB を作って全部書いていた)、出力はスロットの使う行だけ読む
- 照合の塊 (平均 32 行): 56.8 → 49.4 ms (スロット) → 35.4 ms (1 回戻す)、転送だけなら 25 ms。24 層のプロンプト 256 トークン 3.4 → 1.95 → 1.8 s。値は同じ (KL も同じ)
- デコードの q、k、v を 1 つの演算に (`QGEMVQKV`、iron_q4.py と mv_add.h の Q_QKV): 入力の RMSNorm を各コアが自分の x で (norm の重みは x のチャネルで、x を bf16 に丸める RMSNorm と同じ値)、bias、q と k の RoPE を出口で。q と k の行は頭ごとに対の並び (d と d + 32 が隣: q8 のスケールは行ごとなので符号は同じ。プロンプトも Meta 形の RoPE、`two_halves=False`)。RMSNorm と RoPE の 2 演算が減って A 1.53 → 1.48 ms、デコード 80.4 → 79.2 ms。KL 5 文平均 0.031 (bfp16 の版 0.030)
- 採らなかったもの: norm の重みを表の列に畳む (重みの範囲が広がり q8 の誤差が増えて、32 トークンの照合で最上位が変わる)。decode の glu を tanh で (PDI は 6 KB しか減らず、tanh が bf16 の精度)。KV キャッシュへの書き込みを QGEMVQKV から直接 (出力を 3 つに分け、呼び出しごとの位置のある出力の扱いが不明。切り替え 1 回分、約 1 ms/トークン)。router の GEMV を MoERoute に (196 KB を 1 コアで流すと切り替えと同じくらい)

**デコードを常駐の 1 配列に (GptOssStep、`GPTOSS_RESIDENT`、2026-11-14)**

- 前提の測定: デバイスに設計が 1 つだけの full ELF は読み込みが 1 回 (IRON は設計が複数のとき読み込みを展開して毎回配列を構成し直す。1 つなら展開せず、ファームウェアが持っている PDI の読み込みを飛ばす)。専門家だけのグラフはコマンドをまたいでも 1 回 1.15 ms (他の演算の後は 1.78 ms)。展開しない読み込みを複数の設計で使うと (`Composition.expands = False`)、別の PDI の読み込みは 1 回約 1.2 ms
- プロンプトの専門家は内訳を測って打ち切り: gate-up 24.7 ms = 重みの流れと復号 11.7 + x の転送 3.6 + 行列積 9.4。x を memtile に 1 回置いて再送 (`repeat_count`、非対称の物) は 26.1 ms、出力を 8 行の物に (受け渡しを 1/8 に) は 24.66 ms で、どちらも効かなかった (戻した)
- `tools/amd/iron_gptoss_layer.py` と `kernels/gptoss_layer.cc`: 層の演算をすべて 1 つの配列 (専門家の形: 列ごとに重みの流れを 4 コアに分け、x の流れを 4 コアへ、出力を join) の段に。グラフの 1 ステップがコアのループ 1 周で、`phases` (配列の鍵でない param) がどの段を走らせるかを決め、コアは Rtp の語で読む。ステップごとに全部のオペランドを宣言する (配列の lane は 1 回だけ束ねる)。1 コマンド = 前の層の専門家 (E) → q/k/v (Q: RMSNorm、bias、RoPE) → 注意 (A) → o_proj (O: bias と残差) → router (R) → 経路 (T: 上位 4、重み、次のコマンドのスクラッチパッドの語)。25 コマンド/トークン
- コアのプログラム (16 KB): 段ごとに fifo の取得を並べると main だけで約 9 KB (動的な objectfifo の lowering)。段の種類によらない骨組み 1 つ (x の物 → 出力ブロックのループ → 重みのブロック → 出力) にして、違いはカーネルの中で分ける (`gl_plan` が段の数を cnt に書く)。カーネルはコアの役割ごとに別の物 (注意はコア 0、1。router は列 0 のコア 2、3。経路は列 0 のコア 3) で、関数名に版の接頭辞。スカラーの float 演算はコードが長いので使わない (頭ごとの値は周期 4 の 16 レーンで持つ)。読み込むベクトルの配列は 64 バイト境界に (ずれると上位ビットの番地を読む: 注意の出力のコサイン 0.79)
- 重み: q/k/v と o は q8x (iron_q4 の q8 と同じ符号とスケールを、32 出力 × 160 入力と bias で 5504 バイト = mx4 のブロックと同じ大きさに。順は [列][出力ブロック][k ブロック][コア])。o の入力は注意のコアが出す順 (`att_order`)。router は bf16 の 32 × 80 のブロック
- 注意: KV はデコード用に 1 つの配列 (行 = 8 グループのキー、続けて値。`cache.kv`)。x の流れで 128 行ずつ、グループのキーと値を組で読む (4 次元の読み出し、物の数は呼び出しごとの繰り返し回数 `repeat_parameter`。記述子の繰り返しは 64 回まで: 8192 行)。コアは 4 頭ずつ、走らせる最大値・和・値の重みつき和で softmax、sink は最後に足す。8 トークンずつ bf16 の mmul (スコア [トークン][頭] = キー [トークン][k] × q [k][頭]、値の和 [頭][d] += p [頭][トークン] × 値 [トークン][d])。トークンと頭ごとの内積とレーンの和の版は物 1 つ 62 µs、これは 13〜18 µs (文脈 8000 で 1 層 +1.1 ms)。採らなかった形: キーと値を別の流れで (同じ行を 4 コアに配る読み出しは途中に戻りがあり、呼び出しごとの長さと 1 つの記述子に入らない)、1 つの配列をプロンプト側が半分ずつ見る (IRON の演算は飛び飛びの view を取らない: コピーだけ)
- プロンプトは今までどおりキーと値の配列に書く。hwnpu が塊の後にその行を kv に写し (`joinKV`)、デコードの行はプロンプトの塊と `ExportRows` の前に戻す (`splitKV`)。キャッシュの一覧は全部の版のバッファから作る (デコードはキーと値の名前を持たない)
- 経路の語: 経路 (T) が次のコマンドのスクラッチパッド (feedback の引数のバッファ) に専門家の位置 8 語を書く。そのコマンドは位置の語なども読むので、ホストは残りの語を同じバッファに書く (コマンドを出す前にまとめて: 語はデバイスの語と同じキャッシュラインにある)。8 語の場所は IRON が決める (コマンドが共有する語の数で 2〜5 番目から) ので、1 回コンパイルして場所を読み、デコードを組み直す (`Recompile`)。manifest の `fed` に `first`
- 展開しない読み込み (`unexpanded_loads`: GptOssStep を含む並びだけ): 層の 25 コマンドは読み込みなし。出力層の RMSNorm と QGEMV は別の設計のまま (1 トークンに読み込み 3 回)
- 照合 (乱数の重み、`iron_gptoss_layer.py check`): 全部の段がコサイン 1.000000 (専門家 0.999995)、全体を見る層・窓の層・物の境界・8448 行。[Q A O R T] 0.82〜0.93 ms、[E] 1.15 ms
- 24 層: デコード 79.2 → 54 ms/トークン (12.6 → 18.4〜18.6 トークン/秒)、文脈 2048 で 58 ms (前の版は 84 ms)。HF との照合 PASS (コサイン同じ)、KL 5 文平均 0.0313 → 0.0310、hwnpu-chat と agent の課題 (多ターン: デコードの後のプロンプト) も通る (18.0〜18.3 トークン/秒)。メモリ +1.06 GiB (q8x の表 0.66、kv 0.4)

**残り**

- デコード: 出力層を配列の段に (q4 のカーネルがコアのプログラムに入れば、読み込み 3 回、約 4 ms が消える)。段の間で次の重みの流れを先に積む (層のコマンド 1.80 ms、転送量からは 1.61 ms)。注意を 4 コアで
- メモリ: q8x の表はプロンプトの q8 の表と同じ重みの写し (プロンプトが q8x を読めば消える)、kv はキーと値の写し
- プロンプト: 注意の側のコマンドが 1 層 12 ms (256 トークン)、専門家 35 ms、ホスト 4 ms。専門家は重みの流れ・復号・行列積がほぼ均等
- 草稿の検証 (`--verify`) は MoE では行ごとに専門家が違うので作っていない
