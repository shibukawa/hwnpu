# 公開の Go API

[English](api.md)

ほかのモジュールが import できるパッケージ: `github.com/shibukawa/hwnpu` (デバイス、バッファ、グラフ、
カーネル)、`modelstore` (モデルのファイル)、`npuinfo` (OS が NPU について教えること)。API はまだ固まって
いない (`USAGE_LIBRARY.md` は「planned」)。モデルのエンジンは形が決まるまで `internal/` に置く。
使い方の例: [../../USAGE_LIBRARY.ja.md](../../USAGE_LIBRARY.ja.md)。

API を形づくる決定は 1 つ: デバイス、バッファ、完了待ちは共通にし、仕事の投入は能力別に分ける
(コンパイル済みグラフは `GraphRunner`、カーネルバイナリは `KernelRunner`)。[architecture.ja.md](architecture.ja.md) を参照。

## デバイス

```go
type Vendor uint8      // VendorUnknown, VendorIntel, VendorAMD, VendorApple
type Capability uint32 // CapGraph, CapCompile, CapKernel, CapHostCoherent, CapStrictVerify, CapPowerControl, CapPowerMeter

type DeviceInfo struct {
	Index    int
	Vendor   Vendor
	Backend  string // "levelzero", "openvino", "xdna", "xrt", "coreml", "coreai"
	ID       string // バックエンド固有。OpenDevice が使う
	Name     string
	Path     string // デバイスノード。無ければ空
	Caps     Capability
	Driver   string
	Firmware string
}

func Enumerate(ctx context.Context) ([]DeviceInfo, error)  // 組み込んだ全バックエンド。失敗は Join し、ほかは列挙を続ける
func Open(ctx context.Context, opts ...Option) (*Device, error) // 最初に合うもの。ErrNoDevice (列挙のエラーと Join)
func OpenDevice(ctx context.Context, info DeviceInfo, opts ...Option) (*Device, error)

func (d *Device) Info() DeviceInfo
func (d *Device) Close() error // 先にグラフ、リクエスト、バッファを閉じる
func (d *Device) AllocBuffer(size int, opts ...BufferOption) (*Buffer, error)
func (d *Device) Graph() (GraphRunner, error)   // CapGraph が無ければ ErrUnsupported
func (d *Device) Kernel() (KernelRunner, error) // CapKernel が無ければ ErrUnsupported
func (d *Device) PowerPolicy() PowerPolicy
func (d *Device) PowerSample() (PowerSample, error) // CapPowerMeter が無ければ ErrUnsupported
```

バックエンドごとの能力 (各 provider から):

| バックエンド | Caps |
|---|---|
| `coreml` | Graph、Compile、HostCoherent、StrictVerify (`MLComputePlan` があるとき) |
| `coreai` | Graph、Compile、HostCoherent、StrictVerify |
| `levelzero` | Graph、HostCoherent、StrictVerify、PowerControl。ドライバ内コンパイラがあれば Compile、RAPL が読めれば PowerMeter |
| `openvino` | Graph、Compile、HostCoherent |
| `xdna` | Kernel、PowerControl。ドライバのセンサーか APU のパッケージ電力 (amdgpu の hwmon) があれば PowerMeter |
| `xrt` | Kernel。取れれば PowerMeter |

登録の順: darwin では Core ML が Core AI より先 (ライブラリの既定は Core ML。Mac の `hwnpu-chat` は Core AI
だけを使う。[cli.ja.md](cli.ja.md) を参照)。`xrt` は `xdna` より先で、Windows 以外では `HWNPU_BACKEND=xrt`
のときだけ列挙する。`openvino` は `HWNPU_BACKEND=openvino` のときだけ。

## バッファと完了待ち

```go
func (b *Buffer) Bytes() []byte                  // ゼロコピーのビュー。Close 後は無効
func (b *Buffer) Size() int
func (b *Buffer) DeviceAddress() (uint64, bool)  // AMD: BO のアドレス (ホストのバッファは PASID によるユーザーの VA、命令のバッファはデバイスヒープ)
func (b *Buffer) SyncToDevice(off, n int) error  // CPU が書いた後、Submit の前。CapHostCoherent なら何もしない
func (b *Buffer) SyncFromDevice(off, n int) error // Wait の後、CPU が読む前
func (b *Buffer) Close() error

func WithBufferKind(k BufferKind) BufferOption   // BufferDefault、BufferInstructions (AMD の制御コード、デバイスヒープから)

type Fence interface {
	Wait(ctx context.Context) error // 取り消されても投入は走り続け、終わるまで再投入できない
	Done() (bool, error)
}
```

- Intel と Apple はホストとコヒーレントで、Sync は何もしない。
- AMD はコヒーレントでない。x86 ではユーザー空間でキャッシュラインを書き戻し (CLFLUSH、XRT と同じ)、
  それ以外ではドライバの `SYNC_BO` を使う。

## GraphRunner (Intel、Apple)

```go
type GraphSource struct {
	Format     GraphFormat // GraphFormatNative (ベンダーの blob) か GraphFormatOpenVINOIR (CapCompile が要る)
	Path       string      // ファイルかディレクトリ (Core ML: .mlpackage、.mlmodel、.mlmodelc)
	Blob       []byte      // メモリ上の native blob
	XML, Bin   []byte      // OpenVINO IR
	BuildFlags string      // 例: `--inputs_precisions="0:fp16" --config NPU_PLATFORM="4000"`
	Function   string      // 複数の関数を持つ ML Program の 1 つ (macOS 15 以降)。関数どうしは重みを共有する
	Units      string      // Core ML: "" Neural Engine (+CPU)、"cpu"、"gpu"。HWNPU_DEV=coreml.units が上書き
}

type GraphRunner interface {
	Compile(ctx context.Context, src GraphSource) (*Graph, error) // 厳格モードでは *PlacementError
}

type ArgInfo struct {
	Index      int
	Kind       ArgKind // ArgInput, ArgOutput
	Name       string
	Dims       []int
	DType      DType   // 結びつけるバッファの型: F16, F32, F64, BF16, I8, U8, I16, I32, I64
	ModelDType DType   // モデルが宣言した型。デバイスが精度を変えるときに違う (Intel で fp32 を fp16 で実行)
	Layout     string
	Bytes      int
	PaddedDims []int   // NPU に載せるために詰めたときの実際の形。呼ぶ側は Dims/Bytes を使う
}

func (g *Graph) Args() []ArgInfo          // 入力が先、出力が後。その順が Bind の添字
func (g *Graph) Arg(name string) (ArgInfo, bool)
func (g *Graph) AllocInput(index int) (*Buffer, error) // バックエンドが直接読むメモリ (Core AI)。ほかは AllocBuffer
func (g *Graph) Cached() bool             // WithCacheDir のキャッシュから読んだ
func (g *Graph) Placement() PlacementReport
func (g *Graph) NewRequest() (*Request, error)
func (g *Graph) Close() error

func (r *Request) Bind(index int, b *Buffer) error // ArgInfo.Bytes 以上、DType/Dims で詰めて並べる
func (r *Request) BindName(name string, b *Buffer) error
func (r *Request) Submit() (Fence, error)
func (r *Request) Run(ctx context.Context) error   // Submit + Wait
func (r *Request) ResetState() error                // モデルの状態を 0 に (Core ML の MLState: KV キャッシュ)
func (r *Request) Close() error
```

- 1 つのリクエストは同時に 1 つの goroutine からだけ使う。
- `AllocInput` は大きく、実行ごとにあまり変わらない入力 (LLM のキャッシュ) のため。Core AI では実行が何も
  コピーしなくなる。同じ形の引数なら、デバイスのどのグラフにも結びつけられる。
- 1 つのモデルの複数の形 (プロンプトとデコードのプログラム) は、1 つの ML Program の関数として重みを共有する
  (`Function`)。Intel では形ごとに別のグラフ。
- `EncodeF16` / `DecodeF16` は `[]float32` とリトルエンディアンの fp16 のバイト列を変換する。

## KernelRunner (AMD)

```go
type ProgramSource struct {
	Xclbin []byte // xclbin の中身
	ELF    []byte // または full ELF (AIE2P)。Strix Point 以降で IRON と aiecc が書く
	Path   string // またはどちらかのファイル (中身で見分ける)
}

type KernelRunner interface {
	Load(ctx context.Context, src ProgramSource) (*Program, error)
}

func (p *Program) Kernels() []string // Program = ハードウェアコンテキスト 1 つ
func (p *Program) Kernel(name string) (*Kernel, error)
func (p *Program) Close() error

func (k *Kernel) Name() string
func (k *Kernel) Args() []KernelArg // Index、Name、Type (C の型)、レジスタマップでの Offset/Size、Buffer
func (k *Kernel) NewRun() (*Run, error)

func (r *Run) SetBuffer(i int, b *Buffer) *Run
func (r *Run) SetBufferAt(i int, b *Buffer, off, n int) *Run // デバイスには b のアドレス + off
func (r *Run) SetUint32(i int, v uint32) *Run
func (r *Run) SetUint64(i int, v uint64) *Run
func (r *Run) SetBytes(i int, v []byte) *Run
func (r *Run) Err() error                 // setter の最初のエラー
func (r *Run) Submit() (Fence, error)     // 先に入力を Sync する
func (r *Run) Run(ctx context.Context) error
func (r *Run) Scratchpad() []byte         // full ELF: Submit の前に書く ctrl scratchpad。無ければ nil
func (r *Run) UseScratchpad(b *Buffer, off, n int) error // デバイスが書いたメモリからパラメータを読ませる
func (r *Run) Close() error
```

- Run は終わった後、同じ引数でも新しい引数でも再投入できる。
- IRON のいつもの呼び出し `kernel(opcode, bo_instr, instr_count, bo_in, bo_out)` は
  `k.NewRun()` → `SetUint64(0, 3)`、`SetBuffer(1, instr)` (`BufferInstructions` のバッファ)、
  `SetUint32(2, words)`、`SetBuffer(3, in)`、`SetBuffer(4, out)` → `Run(ctx)` になる。

## 厳格モード

```go
type Placement uint8 // PlacementStrict (既定), PlacementPrefer

type PlacementReport struct {
	Verified bool          // バックエンドが確かめられたか
	Method   string        // 例 "coreml-compute-plan"
	Reason   string        // 確かめられなかった理由
	NPUOps   int
	Ops      []OpPlacement // NPU 以外の演算: Name、Type、Device ("cpu"、"gpu"、"unknown")
}
func (r PlacementReport) OnNPU() bool // Verified && len(Ops) == 0

type PlacementError struct {
	Report PlacementReport
	Cause  error // 例: NPU がそもそもコンパイルできなかったときのコンパイラのエラー
}
```

`PlacementStrict` は、全部が NPU で動くと確かめられないグラフを拒む (配置を確かめられないものも含む)。
`errors.Is(err, ErrNotOnNPU)` が真になる。`PlacementPrefer` はグラフを残し、結果を `Graph.Placement` に記録する。

| バックエンド | 配置の確かめ方 |
|---|---|
| Core ML | `MLComputePlan`: 全演算の推奨デバイスが Neural Engine であること。推定で、実行時の保証ではない |
| Core AI | 自分のコンパイルキャッシュ: MPSGraph の delegate のマニフェスト (`mps.fullyPlacedOnANE`。method は `coreai-cache-manifest`) |
| Level Zero | 対応レイヤーの問い合わせ (`levelzero-query-network`)。NPU プラグインはフォールバックしないので、コンパイルが通れば NPU で動く |
| OpenVINO | `HWNPU_DEV=ov.device=NPU` のときだけ確かめた扱い |
| AMD | xclbin / ELF は NPU 専用なので常に満たす |

## 電力

```go
type PowerPolicy uint8 // PowerDefault, PowerLow, PowerBalanced, PowerPerformance
type PowerScope string // ScopeNPU ("npu"), ScopePackage ("package": コア、GPU、NPU)

type PowerSample struct {
	Time    time.Time
	EnergyJ float64 // 累積。1 つのメーターでは単調
	Watts   float64 // メーターが出すときの瞬時値 (AMD のセンサー)。無ければ 0
	Scope   PowerScope
	Source  string  // 例 "rapl:intel-rapl:0"
}
func AveragePower(a, b PowerSample) float64 // (b.EnergyJ - a.EnergyJ) / dt
```

| バックエンド | `PowerPolicy` の写像 | メーター |
|---|---|---|
| Level Zero | Low: キューの優先度を低く + workload type を "background"。Balanced: 通常の優先度。Performance: 高い優先度 + turbo (turbo と workload type は拡張が要る) | RAPL、パッケージ全体、たいてい root |
| xdna | ハードウェアコンテキストの QoS の優先度 (Low / Normal / High) | ドライバのセンサー、または amdgpu のパッケージ電力 |
| Core ML / Core AI | なし (CapPowerControl なし) | API にはない。`npuinfo` (IOReport) と `hwnpu-dev powerbench` を使う |

詳しくは [memory-and-power.ja.md](memory-and-power.ja.md)。

## エラー

```go
var (
	ErrNoDevice    // NPU が見つからない
	ErrUnsupported // このデバイス / バックエンドでは使えない
	ErrLibrary     // 実行時のライブラリが無い (探したパスを Wrap)
	ErrNotOnNPU    // モデルが NPU だけで完結しない (*PlacementError)
	ErrClosed      // もう閉じている
)
type DriverError  struct{ Backend, Op string; Code int64; Msg string } // ze_result_t / errno
type BuildError   struct{ Log string; Err error }                      // ビルドログつきのコンパイル失敗
type VersionError struct{ Component, Have, Need string }
```

## オプション

| オプション | 効果 |
|---|---|
| `WithVendor(Vendor)`、`WithBackend(string)`、`WithIndex(int)` | `Open` でデバイスを選ぶ |
| `WithPlacement(Placement)` | 既定は `PlacementStrict` |
| `WithPowerPolicy(PowerPolicy)` | CapPowerControl が無ければ無視 |
| `WithCacheDir(dir)` | コンパイル結果を保存して使い回す。キーはモデル、ビルドフラグ、ドライバ / コンパイラ (または OS) の版。コンパイルは電力を使うので起動ごとに繰り返さない |

環境変数: `HWNPU_BACKEND` (コマンドのバックエンド。`openvino` / `xrt` を列挙させる)、`HWNPU_ZE_LOADER`
(Level Zero のローダのパス)、`HWNPU_DEV=coreml.units` と `coreai.units` (プログラムを Neural Engine から
外す。例: `cpu` でエンジンの誤差とモデルの誤差を分ける)。

## modelstore

`github.com/shibukawa/hwnpu/modelstore` は、hwnpu のコマンドと同じやり方でモデルのファイルを探して取る。
層は次の順:

| # | 層 | 設定 |
|---|---|---|
| 0 | この機械のキャッシュ (常に書き戻す) | `Options.Root` (`HWNPU_CACHE`、無ければユーザーのキャッシュディレクトリの `hwnpu`) |
| 1 | 取得元: `file://`、`http(s)://` (WebDAV か、読むだけのサーバ)、`s3://`。`?writeback=true` で後の層から読んだものを置く | `Options.Sources` (`HWNPU_MODEL_SOURCE`、`HWNPU_MODEL_SOURCE_1` …) |
| 2 | 同梱のモデル。読むだけで、その場で使う (ハードリンク、できなければシンボリックリンク) | `Options.Bundle` (`configdir.ConfigDir`)、`BundleFolder` |
| 3 | Hugging Face | `Options.HuggingFace` が false でなければ (`HWNPU_MODEL_HUGGING_FACE`) |

- リポジトリのファイルは、その `hwnpu-lock.json` が挙げる 1 つのコミットのもの (モデルが固定したコミット、
  無ければ手元のキャッシュより後の最初の lock、無ければ手元、無ければ Hugging Face の `main`)。ファイルは
  ルートの `original/OWNER/NAME` に置く (`Store.Dir`)。
- 層はプロセスの設定: `Open` はプロセスの中の hwnpu のコード全部に効く。
- API: `Open`、`File(ctx, repo, name)`、`Weights(ctx, repo)`、`Dir`、`Layers`、`NotFound(err)`。ディスク:
  `Usage`、`Limits` (`Options.Limits`)、`DiskSpace`、`PlanPrune` / `Prune`、`Remove`、`PlanClean` / `Clean`
  (取得元)、`ErrNoRoom`。モデルごとのメモリの区分: `Memory`、`SetMemory`、`Memories`。

設計: [distribution.ja.md](distribution.ja.md)、[../cache-sharing-design.md](../cache-sharing-design.md)。

## npuinfo

`github.com/shibukawa/hwnpu/npuinfo` は、OS とドライバが NPU について教えること (デバイス、使っている
プロセス、忙しさ、電力、メモリ) を読む。アクティビティモニタも Linux のいつもの道具もこれを見せない。

```go
info, err := npuinfo.GetWithContext(ctx, npuinfo.Options{Wake: false, SampleInterval: 0, SkipCaches: false})
// info.Linux / info.Windows / info.Darwin (動いている OS の分だけ)、info.Errors (一部の失敗)
```

- 標準ライブラリだけを import し、hwnpu のほかのパッケージも使わない。別のモジュールに切り出せる
  (gopsutil など)。型はどの OS でもビルドでき、別の機械の JSON を読める。`go run ./npuinfo/cmd/npuinfo`
  が JSON を書く。OS ごとの項目は [../../npuinfo/README.ja.md](../../npuinfo/README.ja.md)。
- OS が返すものをそのまま OS ごとの構造体に入れる。共通の API は後で考える。
- **読むことで NPU を起こさない。** amdxdna の `GET_INFO` / `GET_ARRAY` は寝ている NPU を起こし、自動
  サスペンドの待ち 5 秒の間起こしたままにする。そこで `power/runtime_status` が active のときだけ問い合わせる
  (`Options.Wake` で変えられる)。デバイスを開くこと、sysfs (PCI の config 空間は読まない)、`/proc` では起きない。
  macOS では IOReport を 40 回読んでも ANE の電源の状態は 0 のままだった。
- `SampleInterval` (既定 500 ms) は、率のカウンタ (IOReport、Windows のカウンタ) の 2 回の読みの間。
  負にすると macOS では読まない。

読めるもの (Ryzen AI 9 365、Linux 7.0、ファームウェア 1.1.2.64 と、M1、macOS 27.2.1 で測った):

| | Linux (amdxdna) | macOS |
|---|---|---|
| プロセスごと | `GET_ARRAY(HW_CONTEXT_ALL)` が root なしで: PID、列、送ったコマンドと終わったコマンドの数、状態 (`gops`、`fps`、`heap_usage` は 0 のまま) | `proc_pid_rusage(RUSAGE_INFO_V6).ri_neural_footprint` (`phys_footprint` に入らない分)。ほかのユーザーのプロセスは root が要る |
| NPU のメモリ | `/proc/PID/maps` の `/dev/accel/accel0` の行の合計 (LFM3 q4: 1635 MiB。RSS にも VmPin にも出ない) | 上の footprint (0.23 GiB の lut4 のプログラムで 234 MiB) |
| 忙しさ、電力 | amdgpu の `gpu_metrics` v3: `average_ipu_activity` (デコード中 93 %)、`average_ipu_power` (1026 mW)、NPU の読み | IOReport (非公開、root 不要): `PMP/Energy Counters/ANE` (mJ)、`AMC Stats` の ANE RD/WR のバイト、ANE の割り込み。電力の区分 26 個全部を `soc_power_w`、DRAM の読み書き元全部を `dram_bytes_per_sec` に |
| その他 | AIE の版と形、クロック、電力モード、TOPS とタスクの上限、列ごとの AIE の状態 (生)、テレメトリ (生)、最後の非同期エラー | I/O Registry の ANE のクラス (`H11ANEIn`、ロードバランサのクライアント)、`CurrentPowerState`、ANE の kext とフレームワークの版、システムの ANE のプロセス、熱の圧力、低電力モード |
| 取れないもの | fdinfo の `drm-*` (amdxdna が実装していない)。センサーの問い合わせはその BIOS で ENODEV | M1 の `Energy Model` のチャンネルは全部 0 (出さない) |

Windows のコードはある (PnP の ComputeAccelerator のデバイス、計算専用も含む D3DKMT のアダプター、名前に
GPU、NPU、Neural を含むパフォーマンスカウンターの全部) が、クロスビルドしかしていない。

## アプリに組み込むライブラリ `ai` (v0)

hwnpu を NPU でモデルを動かす Go のライブラリの集まりにし、`hwnpu-chat`、`hwnpu-image`、`hwnpu-audio` はその機能を試すデモにしていく (名前は変えない)。設計と作業: [../app-api.md](../app-api.md)、[../app-api-tasks.md](../app-api-tasks.md)。使い方と決まり: [../developer-guide.ja.md](../developer-guide.ja.md)。

| パッケージ | 中身 | 状態 |
|---|---|---|
| `ai` | `App` がアプリの使うモデルを登録し、`Use[T]` は型のついた取っ手 (`stt.Get(ctx, rt)` が `*speech.Transcriber` を返す)。`Plan` (足りないもの、ダウンロードの量、メモリ、ライセンス。ダウンロードと変換、NPU 向けのコンパイルの 2 段)。`Runtime` は準備済みの置き場からだけ読む | v0 |
| `ai/speech`、`ai/text` | 文字起こし、読み上げ、音量の VAD、WAV。翻訳、チャット。機能のパッケージが自分のモデルの準備と読み込み (`internal/airole.Role`) を持ち、`ai` はそれらを import しないので、アプリには使う機能だけが入る | v0 |
| `ai/flow` | 塊の流れ (`Stream[T]`、`iter.Seq2`) とモデルの間のネットワーク | v0 |
| `ai/cli` (+ `konghelper`) | アプリに足すサブコマンド: `models`、`models download`、`models prepare`、`status` (`internal/npustatus` で `hwnpu status` と共有) | v0 |
| `ai/vision` | `Depth` (用途 `depth`: Depth Anything 3、MoGe-2。`DepthMap.Near` が近さを 0〜1 で)、`Faces` (`face-detect`: YuNet。AMD の機械では CPU)、`HeadOf` / `NearestHead` / `HeadNode` (両目の間隔 63 mm とカメラの画角 60 度から頭の位置。モデルなし) | v0 |
| `ai/imagegen` | `Generate` (用途 `image`): 文と参照の絵から絵を、`GenerateNode`。頼まれた形に一番近い書き出しの大きさ (`SizeFor`)。FLUX.2 klein だけ (Anima、FLUX.3 klein は準備のときに断る) | v0 |
| `ai/sound`、`ai/vision` の物体、姿勢、切り抜き、OCR、ラベル | 音楽、効果音、音高 | まだ |

- 準備はコマンドの初回と同じ関数 (`transcribe.Prepare`、`speech.Prepare`、`chat.Prepare` と AMD の版) を同じ置き場に呼ぶので、アプリとコマンドは準備したものを共有する。
- 機械のメモリごとの構成: 用途の名前は 1 つだけ動かすときの一番大きなモデルを選ぶ。いくつも動かすアプリは `use.At(gb, name)` / `use.As(gb)` を書く (`samples/interpreter`: 翻訳は 16 GB から 2B、32 GB から 8B)。実行時に置き場の記録が別のモデルや別のメモリ向けなら `ErrNotPrepared`。`Plan.UsableGB` / `Plan.Fits` が同時に読む合計とモデルに使えるメモリを比べる。
- モデルのつなぎ方は、次の段をいつ始められるかで: 前が全部終わってから (`Last`、大きなモデルの前に `Use.Release`)、できたものから (`Async`、`Latest`、`Hold`、`Tee`)、拍に合わせて (`Metronome`、`Pace`、`Live`)。ネットワークとして宣言する: `flow.Source` → 節 → `flow.Sink` を `a.To(b, 辺)` でつなぐ (辺: なし、`flow.Keep(n)`、`flow.Newest()`)。プログラムの節 `flow.Each` / `Route` / `Both` / `With`、モデルの節 (`speech.TranscribeNode`、`text.TranslateNode` など)、要求ごとに 1 回は `ai.NewCall(...).Then(...)`。時刻の道具は `testing/synctest` でテストする。
- 生の入力: サンプルのモジュール (`samples/`、別の `go.mod`) はマイクを [hwcamera](https://github.com/shibukawa/hwcamera) から取る。`capture.Mic` は別の goroutine でためる (ためないと、文字起こしの間の声が失われた)。
- サンプル: `samples/interpreter` (文字起こし → 翻訳 → 読み上げ)、`samples/parallax` (絵の深度を 1 回推定し、カメラで見る人の頭を追い、その視点から Ebitengine の Kage のシェーダーで描き直す。`samples/internal/layering` の層が切れるたびに入れ替わる)、`samples/layers` (絵を深度で層に分け、各層が隠していた部分を FLUX.2 klein が描き、PSD に書く)。コマンドと共有する準備の手順は `internal/chat` (`AMDPictureExport`) と `internal/imagegen/flux2klein` に移した。

**NPU は 1 つの待ち行列** (Ryzen AI 9 365、whisper-large-v3-turbo と plamo-3-translate-8b と LFM3-Audio-2B-TTS を 1 プロセスに): 呼び出しは入れ替わるが合計は和と同じ (文字起こし 1.38 秒と翻訳 3.34 秒は、同時で 4.67 秒。3 つ同時で 14.7 秒、和は 14.8)。実時間で回るかは各段の NPU の時間の和で決まる: 読み上げは音声 1 秒に 2.6 秒かかるので、訳した音声まで出すと話し続ける相手に追いつかない。字幕だけなら 1 発話に 2.1 秒 (2B) か 3.7 秒 (8B)。`samples/interpreter`: 13.3 秒の日本語を 31.6 秒で 8.5 秒の英語の音声に。出力を書き起こすと訳文と一字一句同じ。

**部品はベクトルではなく文でつなぐ**。隠れ状態は、同じトークナイザーでもモデルの間で互換がない。ベクトルでつなぐのは、橋を一緒に学習したところだけ (文のモデルの隠れ状態を読む LFM3-Audio の読み上げ、多モーダルの塔、Anima の文のアダプタ)。

各モデルは「部品としての値」(入力と出力、常駐の大きさ、2 回目の読み込み時間、一番長いコマンド、ストリーミング) を[新しいモデルのチェックリスト](new-model-checklist.ja.md)に沿って記録する。

## 未解決

- `ai`: 常駐の管理 (メモリの予算、投入の順番)、入力が 3 つ以上の節、モデルを下ろすタイミングをライブラリに任せること、辺ごとの計測、`ai/vision` の残り、`ai/imagegen` の Anima / FLUX.3 klein、コマンドを `ai` に載せ替えること。Intel で `ai` を動かしていない (Mac では 3 つのサンプル、カメラ、窓まで動いた)。`text.Chat` と実物のマイクも未確認。列を分けて本当に同時に動かせるか (AMD) も未確認。
- `npuinfo`: Windows (Intel と AMD) で動かす。M2 以降の Mac で IOReport のチャンネル名を確かめる。AIE の状態と
  テレメトリを読み解く。共通の API。ブラウザで見る道具。
- ライブラリの API は planned で、名前が変わりうる。

## 採らなかったもの

| もの | 理由 |
|---|---|
| グラフの実行とカーネルの実行を隠す 1 つのインターフェース | Intel と AMD は根本で違い、隠すと両方で使いにくくなる |
| `CompileSet` (複数のソースを一度にコンパイルし重みを共有) | 重みの共有は複数の関数を持つ ML Program (`GraphSource.Function`) で行う |
| アプリ側でキャッシュするための `Graph.NativeBlob` | `WithCacheDir` が hwnpu の中でコンパイル結果を持つ |
| `WithLibraryPath`、`HWNPU_ACCEL_DEVICE`、`WithTurbo`、`WithWorkloadType`、`WithQoS` | `WithPowerPolicy` が turbo、workload type、QoS に写る。ローダは `HWNPU_ZE_LOADER` で指定する |
| 別々に学習したモデルの間で隠れ状態を渡す | 互換でなく、組み合わせごとに学習が要る |
| 用途ごとのインターフェースを今公開する | 2 つ目の組み合わせが動くまで待ち、API を早く固めすぎない |
