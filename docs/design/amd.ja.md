# AMD XDNA

[English](amd.md)

AMD の Ryzen AI の NPU: ドライバの経路 (`internal/xdna`、`internal/backend/amd`、Linux)、XRT の経路 (`internal/backend/amdxrt`、Windows)、プログラムの形式 (`internal/xclbin`、`internal/aieelf`、`internal/ert`)、ビルド時のコンパイラとしての IRON、共有のカーネル、固定されるメモリ、電力。モデルの仕事は別のファイル: [amd-llm](amd-llm.md) (言語モデル)、[image-amd](image-amd.md) (画像のモデル)、[moe](moe.md) (Qwen3.6、gpt-oss)。長い設計の記録は [../amd-npu-models.md](../amd-npu-models.md)、データ型と素の速さは [../amd-xdna2-bench.md](../amd-xdna2-bench.md) に残す。

測った機械 (断りのない数字はすべてこれ): Ryzen AI 9 365 (Strix Point、PCI 0x17f0、`npu4` = NPU2、8 列、AIE 1.1、4 コア行)、Ubuntu 26.04、標準の Linux 7.0 の amdxdna (DKMS なし)、ファームウェア 1.1.2.64 (`npu_7.sbin`)。XRT は入れていない。

## デバイスとドライバ (Linux、amdxdna)

- 発見: `/sys/class/accel/accel*/device/driver` が `amdxdna` なら AMD、`intel_vpu` なら Intel (`internal/accelfs`)。ノードは `/dev/accel/accelN` を `O_RDWR|O_CLOEXEC` で開く。`render` グループが要り、エラーにその手順を書く。
- ioctl: DRM の `_IOC` の符号化、`DRM_COMMAND_BASE + 0..10` (CREATE_HWCTX、DESTROY_HWCTX、CONFIG_HWCTX、CREATE_BO、GET_BO_INFO、SYNC_BO、EXEC_CMD、GET_INFO、SET_STATE、WAIT_CMD、GET_ARRAY)。構造体は `amdxdna_accel.h` を写す (全部 `__u32`/`__u64`、パディングなし)。`testdata/abi/xdna_abi.json` (カーネルの UAPI と XRT のヘッダを gcc でコンパイル: ioctl 番号、配置、xclbin の構造体、ERT のビット) と突き合わせる。
- ioctl の構造体の中のポインタは整数でなく `unsafe.Pointer` (ゴルーチンのスタック移動で古いアドレスがカーネルに渡りうる)。
- 再試行は `EINTR` だけ。amdxdna は「ヒープ未準備」を `EAGAIN` で返すので、再試行すると無限ループになる。
- `internal/xdna/xdnasim` は Go の偽のドライバ。生の ioctl の構造体を Linux 7.0 と同じ規則で検査する (`xdnasim.NoSensors` で電力センサーなし)。

バッファ:

| 種類 | 用途 | デバイスが見るアドレス |
|---|---|---|
| `BO_DEV_HEAP` | デバイスを開くごとに 1 つ、どの `BO_DEV` より先に。64 MiB (`AIE2_DEVM_SIZE`) の倍数。npu1 は 64 MiB 固定、npu4 以降は 512 MiB まで。作った直後にマップする (しないと `BO_DEV` が `EAGAIN`) | `xdna_addr` (`0x4000000` 起点) |
| `BO_DEV` | PDI、制御コード、命令列 (`BufferInstructions`、XRT の cacheable)。マップできない。CPU のアドレスはヒープのマップ先 + 差 | `xdna_addr` |
| `BO_SHARE` | 入力、出力、重み | Linux 7.0 は `xdna_addr` を無効のまま残す。PASID のもとではデバイスは mmap 後のユーザーアドレス = `GET_BO_INFO.vaddr` を使う。`xdna.BO.DeviceAddr()` は `xdna_addr` が有効ならそれ、無効なら `vaddr` |
| `BO_CMD` | ERT のパケット (4096 B) | BO_SHARE と同じ |

投入と完了待ち:

- `EXEC_CMD`: `cmd_count = 1` のとき `cmd_handles` にはハンドルの値そのものを入れる。`args` はコマンドが触る全 BO のハンドル。
- AIE2 は `WAIT_CMD` を実装していない。ハードウェアコンテキストの syncobj に対し、`EXEC_CMD` が返した seq を点として `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` (`WAIT_FOR_SUBMIT`、CLOCK_MONOTONIC の絶対時刻) で待つ。`EINTR` (Go のランタイムのシグナル) なら同じ期限で待ち直す。スレッドは眠るので CPU は空く。
- 結果は CMD BO の先頭語の状態欄: `COMPLETED (4)`。`ERROR (5)`、`TIMEOUT (8)` などは `DriverError`。
- 別のプロセスのコンテキストが終わった直後の `CREATE_HWCTX` はときどき `EINVAL` になる。5 秒まで再試行する。
- **デバイスヒープ (64 MiB) は、その大きさの倍数の番地に置く**: そうでないとファームウェアが `MAP_HOST_BUFFER` を断り (opcode 0x106、状態 0x4000003)、`CREATE_HWCTX` が `EINVAL` になる。カーネルの `mmap` は番地の空間が空いている間は倍数の番地を返すが、1 プロセスの 2 つ目のデバイスや大きな写像のあとではそうとは限らない (3 つのモデルを 1 プロセスに読むと 2 つ目か 3 つ目で失敗した)。`internal/xdna.MmapAligned` が大きさと境界の分だけ番地を予約し、境界の番地に `MAP_FIXED` で写す。`internal/backend/amd.Open` がこれを使う (12 回中 12 回通る。前は 5 回)。[../amd-xdna2-bench.md](../amd-xdna2-bench.md) のまれな失敗 (プロセスの起動の約 125 回に 1 回) も同じ原因と見ているが、再現はしていない。

キャッシュの一貫性 (x86):

- 同期は XRT と同じくユーザー空間の `CLFLUSH` + `MFENCE` (`xdna.FlushCPU`、Go のアセンブリ)。Linux 7.0 の `SYNC_BO` はカーネルの写像でフラッシュするので、ページがまだ固定されていない BO では何もしない。amd64 以外は `SYNC_BO`。
- `SYNC_BO` を使うときは常に `TO_DEVICE`。7.0 の `FROM_DEVICE` は「デバッグ用の BO」を同期し、普通の BO では `EINVAL`。clflush は書き戻しと無効化の両方なので、どちらの向きにも足りる。
- 新しいページは最初に触れたときに CPU のキャッシュを通して 0 で埋まる。その汚れた行が NPU の DRAM への書き込みの後に書き戻され、最初のコマンドの出力を消していた。`allocMapped` はマップ直後に全ページに書いてフラッシュする。デバイスヒープもマップ直後にフラッシュする。

## プログラムの形式

### xclbin の流れ (`ERT_START_CU`)

- `internal/xclbin`: `IP_LAYOUT` (`IP_PS_KERNEL` のエントリ: 名前、functional、kernel_id)、`AIE_PARTITION` (列、1 サイクルの演算数、PDI。オフセットはセクション先頭から)、`EMBEDDED_METADATA` (XML: 引数の順、オフセット、大きさ。`extended-data dpu_kernel_id` を CDO グループの `dpu_kernel_ids` と照合する。XRT の shim と同じ)。
- PDI → `BO_DEV`。`CREATE_HWCTX{max_opc, num_tiles = 列 × コア行}`。`CONFIG_HWCTX(CONFIG_CU)` の `cu_bo` は PDI の BO_DEV (設定し直せない)。
- ERT のパケット (`internal/ert`): ヘッダ `state[3:0] … count[22:12] opcode[27:23] type[31:28]`、CU マスク、続いてメタデータのオフセットどおりのレジスタマップ (64 ビット値は下位語から。バッファはデバイスアドレス)。IRON (mlir-aie v1.4.4) はどの署名にも `bo0`〜`bo4` を並べる。設定しない引数は 0 で送る。
- xclbin の読み込み (コンテキストと PDI) 0.5 ms。passthrough 16 KiB の中央値 0.075 ms。

### full ELF (`ERT_START_NPU_PREEMPT_ELF`、opcode 22)

NPU2 でどのモデルも使う形。グラフの 1 回の呼び出しを 1 つのハードウェアコンテキスト、1 つのコマンドで流す。

- `internal/aieelf` が aiecc の full ELF を読む: グループからインスタンス (`main:sequence`) と制御コード、`.dynsym` から引数と PDI、`.rela.dyn` から再配置、`.note.xrt.configuration` から列の数。再配置の種類: 3 scalar32、5 shim DMA 48 ビット (+0x80000000 の窓)、8 address64。種類 4 (制御パケット)、プリエンプション、`.ctrldata` はエラー。`Patch` は書き換え前の語から計算するので、引数を変えて何度書き直しても同じ。
- `internal/backend/amd/elf.go`: `ProgramSource.ELF` (または中身が ELF の `Xclbin` / `Path`)。PDI はプログラムで共有する 1 つの `BO_DEV`、制御コードは run ごとの `BO_DEV` で、引数が変わったときだけ書き直して同期する。コンテキストは `CREATE_HWCTX` だけ (CU なし)。ファームウェアはメールボックスのプロトコルの主版 7 (`AIE2_NPU_COMMAND`、`AIE2_PREEMPT`) が要る。Linux 7.0 は `EXEC_NPU_TYPE_ELF` のスロットを作る (`force_cmdlist`)。
- XRT を root なしで展開し、`LD_PRELOAD` で ioctl を記録して、書き換えた制御コードとパケットを語ごとに突き合わせて確かめた。
- IRON の ReLU → AXPY (2048 要素): 実行時に IRON も XRT も使わず `graph.reference()` とビット単位で一致。読み込み 1.0 ms、1 回 0.27 ms。

### 呼び出しごとの値 (スクラッチパッド) と引き継ぐ値

- 制御コードの TXN (`internal/aieelf/txn.go`、主版 0): `CREATE_SCRATCHPAD` (op 10) が `scratch-pad-ctrl` の BO_DEV のアドレスを書く (種類 8)。`UPDATE_REG` (op 12、`{スロット, 関数, FuncArg, RegOff}`) がレジスタに `表[スロット]` を足す (INCR) か `× FuncArg` を足す (MUL)。コアのパラメータは WRITE で 0 にしてから INCR、shim の BD の長さは BD の書き込みの後に MUL (mlir-aie の `AIEUtils.cpp`)。`core` の値は 2 ビット左にずらす (ファームウェアが下位ビットを落とす)。
- スクラッチパッドの語は呼び出しの値の Affine の式 (IRON の `Word.form`: `clamp(((v·scale + bias) >> down)·mul + add, lo, hi) % mod + post`)。`tools/amd/iron_export.py` が manifest に書き、Go で計算する。`Run.Scratchpad()` がそのバッファを見せる。
- 引き継ぐ値 (IRON の `Carried`)、ホストから 1 回ずつ呼ぶとき: 呼ぶ前に今の値を `carry` バッファの 1 段目に書き、呼んだ後は Affine の次の値 (`steps + 1`) をホストで計算し、計算された値 (`node`) は 2 段目から読む。Walk (64 要素の後続表) を 12 回呼んで一致、1 回 0.18 ms。
- 呼び出しごとの語は、呼び出しごとの値 1 つの Affine でなければならない。`ceil(rows / 8)` や `start + rows` のような形は書き出しで断る (`words without a form`)。固定の塊と呼び出しごとの `start` 1 つを使う。
- 未対応: 1 つの ELF に複数のコマンド (`Plan.commands`)。バッファのアドレスに依る語 (`recording.addresses`) は書き出しでエラー。

### 1 ステップごとの xclbin (NPU1)

- NPU1 (Phoenix、Hawk Point) のファームウェアには full ELF (AIE2_PREEMPT) がない。IRON はそのとき各版を「1 ステップごとの xclbin」(`PerStep`) にする。演算ごとに xclbin のカーネル (opcode 3、命令列、長さ、バッファ) を 1 回ずつ投入する。設定は共有されるので、Gemma 4 E2B のデコードは 1095 ステップで xclbin 2 個、プロンプトの塊は 1096 ステップで 4 個。
- スクラッチパッドがないので、呼び出しごとの値はディスパッチ時のスカラーになり、命令列を呼び出しごとに作り直す。mlir-aie はそのための C++ (`dispatch_gen.cpp`、Linux 専用の共有ライブラリ) を生成する。書き出しがこれを命令列プログラム (JSON、`tools/amd/iron_stream.py`) に直し、Go の解釈器 (`internal/llm/iron/stream.go`) が C の型変換 (符号拡張、切り捨て、通常の算術変換) どおりに動かす。知らない行は書き出しで断る。書き出しはどのプログラムも生成ライブラリと見本のスカラーで比べる。`check/streams.json` (64 本) は Go でも一致する (`TestExportStreams`)。
- 置き場: 常駐するもの (重み、KV) はアリーナのバンク。一時的なものは版ごとの別のプールで 0 から数え直すので、引数 `pool` (`pool_1`…) に置く (重みの上に置かない)。
- 実行 (`internal/llm/iron/steps.go`): xclbin はモデルで 1 回ずつ読み込む。各ステップの run は読み込み時に作り、バッファはバンクの一部として結ぶ (`Run.SetBufferAt`: xdna はアドレスに足し、XRT はサブバッファを作る)。同じコンテキストのコマンドは順に走るので、待たずに続けて投入する (16 個ごとと、xclbin が替わる前に待つ)。窓 1 は 12 % 遅く、64 でも変わらない。
- 命令列のバッファ: (版, xclbin) の組ごとに静的なものと呼び出しごとのものを 1 つずつ持ち、各ステップは 64 バイト境界のスロットを使う (ステップごとに作ると 2191 個で 64 MiB のヒープが尽きた)。xclbin ごとに分けるのは、XRT が命令列の引数を xclbin ごとのメモリグループ (65537、131073…) に置き、複数のグループに結んだバッファを丸ごと複製するため。XRT は待っていない run を再投入できない (`bad command state`) ので、先に待つ。
- NPU2 で `--per-step`: Gemma 4 E2B が xdna と XRT の両方で正しく答え、デコード 2.6 tok/s (full ELF は 6.1)、14 トークンのプロンプト 1.3 秒。NPU1 向けの書き出し (`--device npu1 --cols 4`、xclbin 6 個) は、4 列の模擬 NPU で読み込みと全ステップの命令列の作成が通る (`TestExportLoads`)。塔 (画像、音声) は IRON が NPU1 で動かさないので対象外。

## Windows: xrtshim (`internal/backend/amdxrt`、バックエンド `xrt`)

- Windows の AMD の NPU ドライバは XRT の C++ ライブラリ (`xrt_coreutil.dll`) からしか使えない。`native/xrtshim` は C の ABI を出す小さな C++ のライブラリで、Go が purego (`internal/dl`) で実行時に読む。利用者側は cgo なしのまま。
- API: デバイス (数、開く、名前、BDF、platform / electrical の JSON、XRT の版)、バッファ (host_only、cacheable、メモリグループ、map、address、sync)、ハードウェアコンテキスト (xclbin は `register_xclbin`、full ELF は `xrt::elf`)、カーネル (`xrt::kernel`、`xrt::ext::kernel`、`group_id`)、run (引数、`get_ctrl_scratchpad_bo`、start でスクラッチパッドを同期、wait、state)。エラーは malloc した文字列で返し、Go が写して解放する (スレッドローカルの最後のエラーは、ゴルーチンがスレッドを移るので使えない)。
- 選び方: Windows では常に、Linux では `HWNPU_BACKEND=xrt` のときだけ。`amd` (xdna) より先に登録するので、`hwnpu.Open(WithVendor(AMD))` はこれを選ぶ。
- `xrt_coreutil.dll` は検索パスの外 (`System32\AMD`、ドライバストア) にあるので、先に絶対パスで (または `HWNPU_XRT_CORE` で) 読み込み、xrtshim の import をそれに解決させる。shim は `HWNPU_XRTSHIM` で指定できる。XRT のヘッダ 2.21.75 (Apache-2.0) を `native/xrtshim/include` に同梱し、`make-importlib.ps1` が DLL の export から import library を作る。
- XRT 2.21 の落とし穴: (1) `run.wait(timeout)` が時間切れになったコマンドをもう一度待つと `qds_device::wait() unexpected command state`。待ちは時間を区切らない。(2) xclbin のカーネルの命令列の引数は専用のメモリグループ (`group_id(1)`) にあり、グループ 0 の cacheable のバッファは「local copy」を作れずに失敗する。命令列のバッファは Go のメモリに持ち、結ぶときにそのグループの XRT のバッファに写す。(3) full ELF のカーネルは開いている間、制御コードを 64 MiB のヒープに置く (音声の塔 39.8 MiB が `ENOSPC` になった)。使う run がなくなったらカーネルを閉じる。
- 電力: electrical の JSON の `power_consumption_watts` を 50 ms ごとに読んで積分する (`power.Integrator`)。Linux 7.0 の標準ドライバでは出ない。
- Linux で XRT 2.21.75 を root なしで展開して確かめた: passthrough (xclbin) が PASS、relu-axpy・bounded-relu (スクラッチパッド)・walk (引き継ぐ値) がビット単位で一致、Gemma 4 E2B 6.1 tok/s (xdna と同じ)、多モーダルの hwnpu-chat で画像と音声の答えが正しい。zig の x86_64-windows-gnu でコンパイルが通る。

## ビルド時のコンパイラとしての IRON

- AMD に公開されたグラフのコンパイラはなく、演算ごとの xclbin は切り替え 1 回 0.46 ms かかる (`tools/amd/ctxbench`: 3 つのコンテキストを交互に 2.960 ms、単独の合計 1.571 ms。まとめて投入しても縮まない)。100 演算なら 1 トークン 46 ms。そこで [amd/IRON](https://github.com/amd/IRON) (Apache-2.0) と mlir-aie で書き出し時にコンパイルし、モデルの 1 回の呼び出しを 1 つの full ELF にして、hwnpu が Go で動かす。「変換は Go だけで」の唯一の例外で、利用者の機械には Python も XRT も要らない。
- 書き出しの道具: `tools/amd/iron_export.py` (IRON の `CompiledGraph` → ELF + manifest: コマンド、引数の順と大きさ、バッファの配置、入出力、重み、パラメータ、`carried`、`images` = バンクの初期像 `arg_<種類>.bin`。`--check` は乱数の入力と `graph.reference()` の出力、引き継ぐ値のあるグラフでは `check/calls.json` を書く)。`iron_export_lm.py` (言語モデル)、`iron_graph.py` + `iron.Graph` (それ以外のグラフ。1 つの ELF が複数の重みのバンク `variants` を持てる)。`hwnpu-dev xclbin` は XRT の xclbinutil なしで IRON の設計をビルドする。
- 実行側の道具: `tools/amd/ironcheck` (書き出しをその check と照合)、`irongraph` (`-runs N`、`-swap -variant`)、`ironlm` (言語モデル)、`gemvbench`、`ctxbench`、`elfconfigs` (ELF ごとの設定のし直し)、`HWNPU_IRON_TIMING=1` (コマンドごとの時間)、`--command-per-step` (ステップごとに 1 コマンド。時間を測るためだけ)。
- ビルド環境: mlir-aie の wheel は Python 3.12 が要る (uv)。IRON は mlir-aie の nightly を固定する。装置は `aie_utils.set_current_device(from_name("npu2", n_cols=8))`。大きな書き出し (Qwen3.6: 30 分、ピーク RSS 60 GB 中 53.8 GB) はほかの大きな仕事と並べない。置き場は `~/models` (`/tmp` は tmpfs)。長いものは `systemd-run --user` で。
- 書き出しの名前: `~/models/NAME.amd-npu2-q4` (または `NAME.amd-npu2`)。AMD の NPU だけの機械では、hwnpu-chat が `-m NAME` にこれを選ぶ。

## 費用のモデル (速さを決めるもの)

| 費用 | 値 | 出所 |
|---|---|---|
| DRAM → NPU の重みの読み込み (GEMV、bf16) | 天井は約 50 GB/s、CPU が読める量の約 6 割 (1 スレッド 55 GB/s、8 スレッド 79 GB/s)。DMA の読み 55〜57、書き 48 GB/s | `gemvbench`、[../amd-xdna2-bench.md](../amd-xdna2-bench.md) |
| 1 回の投入 | 固定費 約 0.06〜0.07 ms。full ELF の演算 1 つ 0.13〜0.15 ms (小さいモデルの 1 トークンはこれで決まる) | |
| コマンドの境目 | 約 0.1 ms。コマンドの鎖 (ERT_CMD_CHAIN) でも消えない。スクラッチパッドはコマンドごとに 1 回読まれる | |
| xclbin (コンテキスト) の切り替え | 0.46 ms | `ctxbench` |
| コマンドの中での配列の設定のし直し (演算子が変わる) | ≈ 40 µs + 設定 1 KB あたり 2.3 µs (制御コードの書き込みを約 430 MB/s)。6〜136 KB の設定で 70〜280 µs を測った | `iron_moeprobe.py --switch`、`elfconfigs` |
| 設定の中の 0 の塊 (.bss) | 1 KB あたり約 1.1 µs | |

- full ELF の中では演算子が変わるごとに `load_pdi` (TXN の op 8) が入り、続いて配列の設定 (プログラムメモリ 16 KB × 使うコア、データメモリの初期値、ロック、DMA、スイッチ) が制御コードの書き込みとして並ぶ。PDI 本体は 368 B の殻。同じ演算子が続けば設定は 1 回。
- 密なモデルのデコード ≈ 重み ÷ 50 GB/s + 設定のし直し。LFM3-1.7B q4: 測定 92.6 ms、見積もり 33 + 65 = 98 ms。Gemma 4 E2B q4: 137 ms、35 + 103 = 138 ms。デコードを決めるのは帯域でなく演算子の数と大きさ。効くのは、1 層の演算子を減らす、設定を小さくする (QGEMV を 1 列 1 コアに)、ステップを 1 つの常駐の配列にする (gpt-oss の `GptOssStep`、[moe](moe.md))。
- `elfconfigs` の「reorder … removes at most」は依存を見ないので、1 層 1 コマンドのときだけ意味がある。同じプログラムの書き直しを省いてもプログラムのバイトの 0〜20 % で、費用はバイトにほとんどよらない。

## 共有のカーネルと演算子の選び方

hwnpu は書き出し時に、IRON を変えずに自前のカーネルを IRON の演算子に差し込む (`tools/amd/iron_q4.py`、`tools/amd/kernels/`):

- 量子化した GEMV `QGEMV` (q4nx、q6、q8: `q4_mv.cc`、`q6_mv.cc`、`q8_mv.cc`)、`QGEMVAdd` (出力が出るときにバイアスと残差を足す、`kernels/mv_add.h`)、`QGEMVGlu` (gate と up を 1 つの表、1 つの演算子に)、プロンプトと DiT の `QDequant` / `q4_dequant`。形式とモデルごとの選択は [quantization](quantization.md)。
- QGEMV は 1 列 1 コア (`iron_q4.q_linear`、cores 0 = 16384 出力まで 1、越えると 2。`HWNPU_Q_CORES`、`HWNPU_Q_CORES_UP_TO`): 設定 92 → 36 KB (q8、2048 出力)、速さは同じ、ビット単位で同じ。`QGEMVGlu` は 2 コアのまま (`HWNPU_GLU_CORES`。1 コアでは 2 本の累算で遅い)。デコード: LFM3-1.7B 15.6 → 17.2〜17.3 tok/s (sandwich と合わせて)、Gemma 4 E2B 9.3 → 9.7、Qwen3.5-0.8B 18.5 → 18.7。
- sandwich を次の GEMV の中で (`iron_q4.sandwich_operator`、`sandwich_linear`、`Q_SANDWICH`): 各コアが y = r + rms(h)·w_post と n = rms(y)·w_pre を作ってから n の GEMV をする (SandwichAdd と同じ和と丸め)。LFM3 で使う (`HWNPU_LFM3_SANDWICH_GEMV`)。これだけでは変わらない (15.5 tok/s)。
- GEMV は 64 行のブロックを割り切る数の列しか使わない: 8 列でブロックが揃うように詰める (`iron_gdn.COLUMN_ALIGN` 64)。量子化した射影は出力 64 の倍数 × 入力 512 の倍数。
- hwnpu のカーネルはコアの丸めを最近接偶数にする。AIE のコアは切り捨てで始まり、bf16 / bfp16 に狭めるたびに、平均のある入力の和が下に寄る (IRON の `DequantBFP` も同じ: 平均 1 の入力でコサイン 0.968、`QDequant` の bf16 は 0.99994)。言語モデルのプロンプトは `DequantBFP` のまま (そこでは bf16 が 5〜9 % 遅く、何も変わらない)。
- `iron_silu.SiLUF32` (`kernels/silu_f32.cc`): シグモイドを float32 で。IRON の bf16 の表の SiLU は x = −8 から下でちょうど 0、[−6, 2] で 1 % ずれる。
- `exp2_mid.h`: softmax の exp を bf16 の 1 limb で。4608² × 24 ヘッドの注意で 102.7 → 83.2 ms、誤差は同じ (Anima 以外の画像のモデルの既定)。注意の sink (`iron_sinks.py`) と窓の注意 (`iron_window.py`) は [moe](moe.md) / [amd-llm](amd-llm.md)。

## 固定されるメモリ

- NPU のバッファ (amdxdna の `BO_SHARE`) は全ページが固定され、RSS に出ない: Gemma 4 E2B は RSS 0.43 GB で、`Unevictable` が 9.3 GB 増えた。測るのは accel の fd の `/proc/PID/fdinfo` の `drm-total-memory`、またはプロセスの `smaps` の `/dev/accel/accel0` の写像の大きさ (どちらもほかのプロセスが混ざらない)。確保はページを固定して 0 で埋め 0.33 秒/GiB、解放 0.11 秒/GiB。
- 固定を減らす手 (詳細はモデルのファイル): PLE と埋め込みをホストの表に (`iron.HostTable`、[amd-llm](amd-llm.md))。塔は最初に使うときに読む (`growBanks`)。画像の段は走る間だけ読み、DiT の群は 1 つのバンクで入れ替え (`iron.GraphOptions.SwapVariants`)、VAE の段はバッファを共有 (`GraphOptions.Buffers`) ([image-amd](image-amd.md))。DiT の重みを 8 ビットに。多く固定しても速くはならない (MoE で 35 GB 固定しても同じ)。
- 待つ間の解放 (`hwnpu-chat -idle`、既定 5m、0 で解放しない): その間要求がなければ、`chat.Engine` が会話をホストに写し (`putAside`)、`iron.Model.Release` がデバイスのバッファと命令列を全部放してデバイスを閉じる。次の要求で同じ書き出しを `Reload` し、写しか prompt store から続ける。モデルは manifest、ホストの表、塔の前処理を残すので、`Images`、`Context`、メディアのトークン数は答えられる。放した間の実行は `errReleased`。E2B q4 1775 MiB → 0、読み直し 0.8 秒。画像と音声の E2B 2818 MiB → 0、2.0 秒。専門家を CPU で動かす MoE では、`-idle` は中核のものだけを wired に残す。対象は IRON だけ: Neural Engine と Intel のプログラムは読み直しに数十秒から数分かかる。
- 並んだ書き出しは、同じファイルとバンクの同じ先頭をディスクで共有する (`iron.ShareExport`、[distribution](distribution.md))。

## 電力

- `SET_STATE(SET_POWER_MODE)` は `DRM_ROOT_ONLY` でデバイス全体に効くので、hwnpu は触らない。電力ポリシーはコンテキストの QoS priority に写す (LOW 0x280、NORMAL 0x200、HIGH 0x180。0 はドライバの中で HIGH)。
- 計測: NPU のセンサー (`GET_INFO(QUERY_SENSORS)`、mW、`Scope = npu`) があればそれ。DKMS 版のドライバだけが持ち、標準の Linux 7.0 にはない。なければ amdgpu の hwmon の `power1_input` (PPT、APU のパッケージ全体、µW、root 不要。`Scope = package`)。どちらかがあれば `CapPowerMeter`。どちらも瞬間値なので、最初の `PowerSample` から裏で 50 ms ごとに読んで台形で積分する (`power.Integrator`、XRT の electrical の JSON と共通)。デバイスを閉じると止まる。テストでは `Provider.Hwmon` で探す場所を差し替える。パッケージの電力のもう 1 つの手は RAPL。
- amdxdna への問い合わせ (GET_INFO、GET_ARRAY、デバイスを開くこと) は寝ている NPU を起こし、自動サスペンドの待ち 5 秒の間起こしたままにする。監視は `power/runtime_status` が active のときだけ問い合わせる ([memory-and-power](memory-and-power.md))。
- 数字 (パッケージ全体、冷ましてからのアイドル 4.7 W): GEMV 12.3 W、32 MiB を読んで 3〜5 mJ (CPU は読むだけで 18〜28 W、7〜12 mJ)。Gemma 4 E2B の生成中は平均 11.8 W、1 トークン 1.93 J (アイドル込み)。

## 落とし穴 (AMD 全体)

| 落とし穴 | 対処 |
|---|---|
| 2 秒を超えるコマンドは `ERT_CMD_STATE_TIMEOUT` で終わる | 書き出しで分ける (`SCRATCHPAD_WORDS`: 密なモデルは 24、Gemma 4 の tail の書き出しは 32) |
| コマンドの制御コードは、開いているほかのものと合わせて 64 MiB のデバイスヒープに収まらなければならない (`CREATE_BO: invalid argument`、`ENOSPC`) | コマンドを分ける。`ENOSPC` ならデコード / verify の run を捨てて再試行 (IRON と同じ)。使う run のないカーネルを閉じる |
| 1 つの shim のチャネルに 3 本を超える転送を積むと (タスクキューは 4 段) コマンドが止まる | 古いものを待ってから積む。後の転送が前のものを終わらせる形では、先に積むのは 2 本まで |
| 続けて 2 回止める、またはモデルがコンテキストを作る間にファームウェアへ毎秒 2 回問い合わせると、SMU が戻らなくなった (`aie2_smu_init: Access power failed`、`failed to init smu`、メールボックスのタイムアウト)。`modprobe -r` も戻らず、再起動だけが直す | 止めない (上)。読み込み中に `hwnpu status` を回さない。代わりに fdinfo の `drm-total-memory` を読む |
| グラフの 2 回目、または次の塊の最初のコマンドが止まる | 各回の前に、別のハードウェアコンテキストで空のコマンドを流す (`iron.GraphOptions.Barrier`、`stepBarrier`、`chunkBarrier`、約 0.3〜0.5 ms)。同じコンテキストの空のコマンドでは効かない。グラフは `irongraph -runs 3` で確かめる。原因は不明 |
| 測るために ELF の `load_pdi` やプログラムの書き込みを書き換える | 読み込みの手順を崩すと NPU と SMU が止まり、再起動が要った。しない |
| ホストの手順が読むものが、コマンドの間に上書きされうる (IRON のアリーナはホストの読みを使用と数えない) | 状態に写す。ホストの手順のある書き出しはモデル全体の深さで確かめる |
| DMA の 1 回は最大 1023 語、DMA のループは最大 64 回、記述子は中間の次元で戻る stride を取れない | 表を演算子が読む順に並べる |
| プログラムメモリはコアあたり 16 KB | テンプレートの複製でなく noinline の関数 1 つに |
| 長いベクトルの `filter_even` / `filter_odd`: 1 回約 290 K サイクル | 32 要素のベクトルで `interleave_unzip` |
| llvm-aie は早期 return のある小さなループで失敗することがある ("LoopStart/LoopBody: insufficient padding") | 展開するか分岐をなくす |
| bf16 の `mmul` | `mac_4x8_8x8` がネイティブ (1 命令 256 積、ベクトルの mac は 16)。AIE2P のネイティブの浮動小数の行列命令は bfp16 の `mac_8x8_8x8T` |
| IRON のコンパイルのキャッシュは演算子のパラメータで引き、コードを見ない | 変えた演算子には形のパラメータ (`form`、`sink_form`) を与える |
| 入っているカーネルのヘッダは mlir-aie のチェックアウトと違う | IRON がコンパイルするヘッダを読む |
| 層ごとに違う `array=True` のパラメータは層ごとの配列になる (コマンドごとに読み直し) | 大きさを配列のパラメータにしない。書き出しを変えたらコマンドの時間も見る |
| IRON の参照はカーネルを写している (tanh の GELU の表、MEDIUM の bfp16 なし、epilogue の SiLU の誤差なし: AIE2P の `vtanh` は \|x\| ≤ 0.5 で x を返す) | 各演算子を本物の活性で float32 と比べる。IRON の参照だけで済ませない |
| IRON の MHA は点数を bf16 で持つ: 大きくて差の小さい点数は差を失う | 全部のキーから最初のキーを引く。または最大の点数を float32 と比べる |
| バイアスを運ぶ定数の「1」のチャネルが、活性と丸めでずれる | 活性と bf16 がそのまま返す値を選ぶ (SiLU なら 16) |
| 乱数の重みは bf16 / bfp16 の丸めを拡大する (ロジットが平らで、最上位と 2 位の差 0.23。本物は 1.51)。IRON の参照とのコサイン 0.88〜0.99 は移植の誤りではない | 本物の重みで HF と比べる |

## 未解決

- NPU1 の実機 (4 列の模擬と、同じ書き出しの形を NPU2 でしか確かめていない)。xclbin 6 個でハードウェアコンテキストの上限 6 を使い切る (ほかのプロセスが NPU を使っていると作れない)。それを超える書き出しにはコンテキストの入れ替えが要る。
- Windows の実機での xrtshim、MSVC でのビルド。
- 標準の Linux での NPU 単体の電力 (センサーなし。数字はパッケージの PPT)。
- `--load-pdi-to-ctrl-pkt` (設定を制御パケットとして DMA で流す。2.3 µs/KB の部分を最も減らせる見込み): aiecc が `npu_materialized.mlir` で止まり ("operand #0 does not dominate this use"、`aie.lock`)、`internal/aieelf` も種類 4 にまだ対応していない。
- GDN のカーネルの静的な配列を IRON の Worker の局所バッファに (0 の塊をなくす): Qwen3.6 で 1 トークン約 8 ms、Qwen3.5 で約 5 ms。未着手。
- 密なモデルの 1 層を 1 つの常駐の配列に (LFM3-1.7B で 1 トークン約 60 ms の見込み)。
- 1 つの ELF に複数のコマンド。アドレスに依るスクラッチパッドの語。
- UMQ 世代での ELF の流れ。デバイスヒープの拡張。
- `dma_free_task` の後の BD の再利用についての aiecc の警告 (害はないとみるが、証明していない)。

## 採らなかったもの

- NPU2 での演算子ごとの xclbin: 切り替え 1 回 0.46 ms (100 演算で 1 トークン 46 ms)。
- `WAIT_CMD` での待ち: AIE2 は実装していない。
- `EAGAIN` の再試行: 「ヒープ未準備」の意味で、無限ループになる。
- `SYNC_BO(FROM_DEVICE)`: Linux 7.0 では普通の BO で `EINVAL`。
- NPU1 のディスパッチのスカラーを語ごとのあてはめで: 三項演算子 (`s > 1 ? 511 : 0`) で崩れる。生成された C++ のライブラリは Linux 専用。
- Windows に、XRT の旧 C API を purego で (`xrtPLKernelOpen`、`xrtRunSetArg`: deprecated で NPU で動く保証がなく、可変長引数) と、ONNX Runtime の VitisAI EP (カーネルを動かせず、依存が大きい): xrtshim を選んだ。
- XRT の時間を区切った待ち (`run.wait(timeout)`): 待ち直すとコマンドの状態が壊れる。
- `--reconfig-teardown`: 小さな設定は 95 → 79 µs、大きな設定は 236 → 274 µs、制御コードも増える (483 → 597 KB)。
- 設定を共有するための演算子の並べ替え: Qwen3.6 で 1 トークン多くて 40 回 (約 4 ms、2 %)、密なモデルは依存で並べられない。
- `QGEMVGlu` の 1 コア: 2 コアの 17.2 に対し 16.3 tok/s (1 コアで 2 本の累算)。
- 小さな設定 (RMSNorm、加算) をまとめること: 割に合わない。
- Neural Engine と Intel の `-idle`: 読み直しに数十秒から数分かかる。
- 速さのために多く固定すること: 効かない。
