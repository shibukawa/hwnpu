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

Core ML は小さなモデルを fp16 でも CPU に置く。配置はモデル全体の計算量で決まり、単純な閾値では表せない (MODEL.md に測定表)。ANE に移すのはバッチ次元の拡大で、各行が独立なモデルならゼロ埋めで結果は変わらない。

- `tools/coreml/ane_pad.py`: ML Program を MIL に読み戻し (coremltools の milproto ローダー)、入力のバッチ次元を書き換えて型推論をやり直し、変換し直す。候補のバッチを小さい順に試して `MLComputePlan` で全演算が ANE に載る最小のものを採用し、メタデータ `hwnpu.logical_batch` / `hwnpu.padded_batch` を書く。`--verify` で元のモデルと出力を比べる。coremltools は開いたパッケージの Manifest.json を書き換えるので、入力は一時ディレクトリに複製してから開く
- Core ML バックエンド: メタデータ (`MLModelCreatorDefinedKey`) を読み、`ArgInfo.Dims` / `Bytes` は元のバッチ、`ArgInfo.PaddedDims` に実際の形状を入れる。実行要求ごとにパディング済みのバッファを持ち、入力の行を書き写し (余白はゼロのまま)、出力の行を書き戻す
- 検証: 3 層 MLP (幅 128、バッチ 1) は CPU 配置で厳格モードに拒否され、バッチ 64 にパディングしたものは全 6 演算が ANE、参照値との差 0.0002 (`TestCoreMLBatchPadding`)

測定 (Python から Core ML、中央値): 1 行を CPU で 73 µs、64 行にパディングして ANE で 280 µs、64 行を CPU で 104 µs。このモデルでは ANE の起動の固定費が計算より大きく、遅延は悪化する。電力は root が必要で未測定。パディングは明示的に選んだときだけ効く仕組みとし、埋めた行に実際の要求を載せるマイクロバッチ化と組み合わせるのが本筋。


### 16.7 Apple: 電力の計測と Neural Engine の損益分岐点 — 2026-10-28

- `cmd/powerbench` (darwin): sudo のパスワードを最初に一度だけ聞き、`sudo -n powermetrics -f plist` を計測ごとに起動する。各モデルを別スレッドで呼び続けたまま (`-rate` で一定頻度も可) CPU / GPU / ANE の電力を取り、アイドル時を差し引いた正味の電力と、呼び出し 1 回あたりのエネルギー (mJ/call) を出す。NPU に 1 演算でも載るモデルは、CPU だけの場合も測る
- CPU や GPU との比較用に、Core ML バックエンドは環境変数 `HWNPU_COREML_UNITS=cpu` (`MLComputeUnitsCPUOnly`) と `gpu` (`MLComputeUnitsCPUAndGPU`) で読み込む (計測専用で、公開 API にはしない)
- `tools/coreml/gen_sweep.py` (大きさを段階的に変えた合成エンコーダー) と `tools/coreml/convert_minilm.py` (all-MiniLM-L6-v2 を系列長固定で変換し、PyTorch と照合) を追加
- 結果 (M3、詳細は MODEL.ja.md): 損益分岐点は呼び出し 1 回あたり約 200 MFLOP。それより大きいと ANE は 2〜3 倍速く、エネルギーは約 4 分の 1 (約 1 TFLOP/J 対 CPU 0.25 TFLOP/J)。それより小さいと Core ML の呼び出しの固定費 (約 55 µs) と ANE の起動の費用が勝ち、CPU が有利。16.6 のゼロ埋めパディングは、このモデルの大きさでは得にならなかった

### 16.8 M5: Go だけの LLM 変換 (LFM3-1.7B → Core ML) — 2026-10-28

- 方針: モデルは Hugging Face の公式 bf16 から配布前に一度だけ各 NPU の形式へ変換し、S3/NFS などで配る。中間形式 (ONNX) は置かず、`config.json` と safetensors/GGUF から Go で直接組み立てる。量子化済みの公開物は劣化なしに並べ替えられるときだけ使い、量子化し直しはしない (決定の詳細は `.knowledge/concepts/decision/model-source-selection.yaml`)
- 実装: `internal/mlpackage` (MIL protobuf、MILBlob 形式の weight.bin、Manifest.json)、`internal/milspec`/`internal/mlspec` (coremltools 9.0 の .proto から protoc-gen-go で生成。`google.golang.org/protobuf` を依存に追加)、`internal/safetensors`、`internal/gguf` (Q4_0/Q8_0/Q6_K の逆量子化)、`internal/palette` (4 ビット LUT の当てはめ)、`internal/llm/lfm3` (構造の定義)、`cmd/hwnpu-convert`
- 検証の流れ: `tools/lfm3/ref_logits.py` (現在は `tools/ref/ref_logits.py`) で PyTorch fp32 の各層の隠れ状態とロジットを保存し、`-layers N` で切り詰めたモデルを層ごとに照合する (`TestAgainstReference`、`TestPerLayer`)。層の中は `TestConvPieces` で演算ごとに照合し、ANE の数値特性は `TestActivationRange`、配置は `TestPlacementVariants`/`TestOpVariant` で調べた
- 結果 (M3): bf16 → fp16 のグラフは CPU で KL 0.020 (argmax 21/24) と正しい。lut4 (出力 16 行ごとの表、`kmeans-minmax`) は 24 層 1627 演算がすべて ANE に載り、KL 0.89。損失のほぼ全部が 4 ビット量子化によるもので、ANE の演算誤差は `-post-scale` で層ごと 1.4〜2.7% に抑えた
- ANE について分かったこと (MODEL.ja.md にも記載): 行列積の出力に約 2.5e-4 の絶対誤差の床がある。LUT は 16 行ごとの表だけが ANE で動き、列方向のブロック化や行ごとの表、入力ブロックごとのスケールを持つ int4 (Q4_0 相当) は使えない。gather は常に CPU。値の名前にドットがあるとローダが落ちる
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

**実測** (M3 16GB、macOS 27.0.1、`npuprobe split`、lut4 64 層 1.81 GiB。レポートは `npuprobe-m3-split.md`)

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

**multifunction では上限を超えられない** (M3、`npuprobe split -layout functions,files`、lut4 64 層 1.81 GiB。レポートは `npuprobe-m3-functions.md`)。Core ML の ML Program は関数を複数持て、`MLModelConfiguration.functionName` で 1 つずつ別のモデルとして読み込める (macOS 15)。関数は weight.bin を共有するので、1 つのファイルのまま分けられるかを試した。

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
- `npuprobe kv`: bench と同じ大きさの層 (lut4) を 11 層 (LFM3-1.7B の注意層の数) にし、本物の注意を入れて 1 トークンを通す。`host` はキャッシュを入力に取り新しい K/V を出力する (LFM3 の方式)、`state` はキャッシュを層ごとの MLState に置き、one-hot の列 (1, 1, cache, 1) で `cache + onehot*(k - cache)` と書き込む (要素ごとの演算だけ。`slice_update` は CPU に置かれる)。レポートは `npuprobe-m3-kv.md`
- 結果 (M3、全部 ANE):

| 容量 | KV | host ms (CPU use) | state ms (CPU use) |
|---|---|---|---|
| 256 | 5.5 MiB | 7.1 (0.16) | 6.6 (0.06) |
| 1024 | 22 MiB | 11.0 (0.18) | 10.0 (0.04) |
| 4096 | 88 MiB | 27.0 (0.16) | 26.8 (0.02) |
| 8192 | 176 MiB | 49.5 (0.15) | 49.3 (0.02) |

- 速さは変わらない。MLState で下がるのは CPU use だけで、ホスト側バッファでは Core ML が CPU で入力を扱っている分 (0.15 コアほど) がなくなる。容量に比例する費用は両方に残るので、原因は入力のコピーではなく、固定の長さのキャッシュ全体に対する ANE の計算 (連結または更新、GQA の tile、注意)
- ANE の制約: 状態に書いた値をそのまま後ろで使うと、ANE 向けのコンパイルが `error code: -14` で失敗する (CPU と GPU なら読み込める)。書いた後に読み直した値を使えば載る。GQA で K/V を tile で複製せず、問い合わせを (kv ヘッド, グループ) の行列にしてまとめて読む形 (同じ計算) は、host では全演算 CPU (8192 で 85.7 ms)、state ではコンパイル失敗だった
- 費用の正体 (`npuprobe kv -kv-modes host,host-mha`、レポート `npuprobe-m3-kv-mha.md`): K/V を `tile` で複製せず 16 の問い合わせヘッドがそれぞれ K/V を持つ形 (キャッシュ 4 倍) は、1024 / 4096 / 8192 で 14.2 / 36.6 / 63.5 ms と host の 1.3 倍にしかならない。帯域でも複製でもなく、容量全体に対する問い合わせ 1 行の注意の計算 (層ごとに 16 ヘッド × 容量、1K トークンで約 8 GFLOP/s) が ANE を使い切れていない。手法の調査と当てはめは `docs/kv-cache.md`
- int8 のキャッシュ (`npuprobe kv -kv-modes host-int8`、レポート `npuprobe-m3-kv-int8.md`): キャッシュを int8 の入力 (Core ML の仕様の版 10、macOS 26 以降) とトークンごとの fp16 のスケールで渡し、`cast` と `mul` で戻す形は全演算 ANE に載る。メモリは半分で、速さは同じ実行の fp16 より 1 割遅い (1024/4096/8192 で 11.9/29.5/50.8 対 10.8/27.6/46.5 ms)。量子化はメモリのための手法で、ANE の速さには効かない。`mlpackage.Int8` の入力の型を足した
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
- **ANE の 1 次元あたりの上限は 16384**: キャッシュ 16384 (鍵の軸 16385) では全注意の層の注意の演算 (2 部分で 42 と 86 演算) が CPU に置かれ、1 トークン 375 ms、読み込み 140 秒、プロセスの最大 RSS 4.5 GiB になった。`npuprobe kv -kv-layers 2 -kv-cache 16383,16384` で境目を確かめた: 鍵 16384 は全部 ANE で 16.6 ms、16385 は注意の 39 演算が CPU で 117 ms (`npuprobe-m3-kv-limit.md`)。`llm.NeuralEngineMaxDim` に記録し、`hwnpu-convert` は `-cache + -seq` がこれを超えると警告する。今の方式で ANE に載る文脈の上限は 16383 トークン (Q=1)
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
- 結果 (M3、6 層 lut4、LFM3-1.7B の全注意の層の数。レポートは `npuprobe-m3-kv-chunk.md`、`npuprobe-m3-kv-chunk-long.md`):

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
  - 層 0〜2 の注意と MLP の出力は rms 0.01〜0.1 と小さく、ANE ではそれぞれ 3〜5% ずれる (行列積の絶対誤差の床、MODEL.md)。残差の流れの誤差は 2〜3% で頭打ちになり、平均した埋め込みにはほとんど残らない
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

