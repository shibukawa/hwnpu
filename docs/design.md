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

### 16.17 3 つ目のアーキテクチャ: PLaMo-Embedding-1B (PlamoBiModel) — 2026-10-30

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
