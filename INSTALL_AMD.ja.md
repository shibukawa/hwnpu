# AMD Ryzen AI の NPU を hwnpu で使うための準備

AMD Ryzen AI プロセッサ（XDNA NPU を積んだ Phoenix、Hawk Point、Strix Point、Krackan Point 以降）を載せた Linux 機で、hwnpu から NPU を使えるようにする手順です。小さなテスト用の設計をビルドし、動作を確かめるところまで扱います。

> **現状。** AMD バックエンドは、カーネルと XRT のソースをもとに作った、amdxdna ドライバを模した偽のドライバでしか試していません。実機で手順 4 の `cmd/report` を実際の設計付きで実行し、結果の報告を送っていただけるのがいちばん助かります。

English: [INSTALL_AMD.md](INSTALL_AMD.md)

## 対応プラットフォーム

| プラットフォーム | 対応 | 補足 |
|---|---|---|
| Linux（x86-64） | 対応 | `amdxdna` ドライバを含むカーネル（手順 1）。 |
| Windows | **非対応** | まだ対応していない。AMD の Windows ドライバは XRT の C++ ライブラリ（`xrt_coreutil.dll`）経由でしか使えず、hwnpu は cgo なしではこれを呼べない。 |
| WSL2 | **非対応** | Windows に対応しておらず、WSL2 は NPU を Linux 側に渡さない（WSL2 の中に `/dev/accel` がない）。Linux を直接インストールした環境で使う。 |
| macOS | 該当なし | Mac には AMD の NPU がない。 |

## hwnpu が必要とするもの

hwnpu は、Linux のカーネルドライバ **amdxdna** を `ioctl` と `mmap` で直接呼びます。実行時に XRT は使いません。Windows にはまだ対応していません。

| 部品 | hwnpu に必要か | 補足 |
|---|---|---|
| `amdxdna` を含む Linux カーネル | 必要 | 6.14 から標準カーネルに入っている（`CONFIG_DRM_ACCEL_AMDXDNA`）。新しいほどよく、AMD のツールは 6.17 以降を求める。 |
| NPU のファームウェア（`/usr/lib/firmware/amdnpu/`） | 必要 | `linux-firmware`、または `amdxdna-dkms` や XRT プラグインのパッケージに含まれる。 |
| `/dev/accel/accelN` の読み書き | 必要 | `render` グループ。 |
| XRT（`xrt-smi`） | 不要 | 導入の確認に便利。 |
| ROCm | 不要 | ROCm は GPU 用のスタック（`amdgpu`、HIP、ROCr）。NPU はドライバ（`amdxdna`）、ランタイム（XRT）、ファームウェアが別で、ROCm なしで動く。 |
| IRON / mlir-aie と Peano | 不要 | カーネル（xclbin と命令列）を**ビルドする**ときに必要。 |
| Go | 必要 | 1.27 以降（`go.mod` を参照）。`CGO_ENABLED=0` で動く。 |

Intel と違い、AMD の NPU はモデルを受け取りません。IRON / mlir-aie でビルドしたカーネルを実行します。カーネルは、AIE アレイの構成とカーネルの情報をまとめた **xclbin** と、**命令列**（`insts.bin`）の組です。

## 1. ドライバとファームウェア（Ubuntu 24.04）

BIOS を最新にし、BIOS で NPU（IPU と表記されることもあります）が有効になっていることを確かめてください。mlir-aie の手順では、DKMS のドライバモジュールを読み込めるよう、セキュアブートを無効にすることも勧めています。

カーネルの版を確かめます。

```bash
uname -r
```

6.17 より古ければ、HWE カーネルを入れて再起動します。

```bash
sudo apt update && sudo apt install --install-recommends linux-generic-hwe-24.04
```

最新のドライバ、ファームウェア、XRT をいちばん簡単にそろえる方法は、[mlir-aie](https://github.com/Xilinx/mlir-aie) が勧めている AMD の PPA です（2026-10-28 に確認）。

```bash
sudo add-apt-repository ppa:amd-team/xrt
```

```bash
sudo apt update && sudo apt install libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu amdxdna-dkms
```

入れ終わったら再起動します。hwnpu 自体が必要とするのはカーネルドライバとファームウェアだけです。新しいカーネルと `linux-firmware` があれば標準カーネルのドライバで足りますが、PPA を使うとドライバとファームウェアの版がそろいます。

利用者をデバイスにアクセスできるグループに加え、ログインし直します。

```bash
sudo usermod -aG render $USER
```

ドライバはバッファのメモリを固定します。確保に失敗する場合は、AMD の推奨に従って固定できるメモリの上限を上げ、ログインし直します。

```bash
printf '* soft memlock unlimited\n* hard memlock unlimited\n' | sudo tee /etc/security/limits.d/99-amdxdna.conf
```

デバイスを確かめます。

```bash
ls -l /dev/accel/
```

```bash
xrt-smi examine
```

`xrt-smi` は XRT のパッケージに含まれます。hwnpu には必須ではありません。

## 2. テスト用の設計をビルドする

`cmd/report` は IRON のサンプル `passthrough_dmas` を実行します。このサンプルは NPU の DMA でバッファを写すだけなので、AIE コア用のコンパイラが要りません。mlir-aie の README のとおりに IRON を準備します。

```bash
git clone https://github.com/Xilinx/mlir-aie.git
```

```bash
cd mlir-aie
```

```bash
sudo apt install build-essential clang lld cmake ninja-build python3-venv python3-pip uuid-dev
```

```bash
source utils/env_install.sh
```

```bash
source utils/env_setup.sh
```

xclbin と命令列だけをビルドします。Phoenix と Hawk Point では `--dev npu` を、Strix Point 以降では `--dev npu2` を指定します。フラグの名前はサンプルの説明文から取ったものです。版によって違う場合は `--help` で確かめてください。

```bash
cd programming_examples/basic/passthrough_dmas
```

```bash
python3 passthrough_dmas.py --dev npu --xclbin-path final.xclbin --insts-path insts.bin
```

この設計は int32 を 4096 個（16384 バイト）写します。これは `cmd/report` の既定値と同じです。別の `-n` でビルドしたときは、報告プログラムに `-bytes` で大きさを渡してください。

## 3. hwnpu を取得する

```bash
git clone https://github.com/shibukawa/hwnpu.git
```

```bash
cd hwnpu
```

## 4. 動作を確かめる

```bash
go run ./cmd/hwnpuinfo
```

AMD の NPU が `backend: xdna` として表示されます。世代、AIE の列数、AIE の版、ファームウェアの版も表示されます。

手順 2 で作った設計を渡して、報告を作ります。

```bash
go run ./cmd/report -xclbin /path/to/final.xclbin -insts /path/to/insts.bin -o report.md
```

最初に、バッファの確保とキャッシュの同期、デバイスヒープ上の命令列用バッファ、NPU の電力センサーを確かめます。続いて設計を読み込んで実行し、出力が入力と一致するかを確かめ、実行時間と NPU の電力量を測ります。`-xclbin` を省くと、カーネルの確認は飛ばします。xclbin 内のカーネル名が `MLIR_AIE` でない場合は、`-kernel 名前` で指定してください。

電力ポリシー（ハードウェアコンテキストの優先度）の効果を比べるときは、2 回実行します。

```bash
go run ./cmd/report -xclbin final.xclbin -insts insts.bin -power low -o low.md
```

```bash
go run ./cmd/report -xclbin final.xclbin -insts insts.bin -power performance -o performance.md
```

## 5. 自分のカーネルを使う

```go
dev, _ := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
kr, _ := dev.Kernel()
prog, _ := kr.Load(ctx, hwnpu.ProgramSource{Path: "final.xclbin"})
k, _ := prog.Kernel("MLIR_AIE")
for _, a := range k.Args() { fmt.Println(a.Index, a.Name, a.Type, a.Buffer) }
```

- 命令列は `dev.AllocBuffer(n, hwnpu.WithBufferKind(hwnpu.BufferInstructions))` に置きます。命令列はデバイスヒープ上にある必要があるためです。
- NPU は CPU のキャッシュと自動では同期しません。入力を書いたら `SyncToDevice` を、出力を読む前に `SyncFromDevice` を呼んでください。
- 引数の並びと大きさは xclbin のメタデータから読みます。IRON の設計では `opcode（3）、instr、ninstr、各バッファ` の順です。

## うまくいかないとき

| 症状 | 原因と対処 |
|---|---|
| `hwnpuinfo` に AMD のデバイスが出ない | `ls /dev/accel/` と `sudo dmesg \| grep -i amdxdna` を確かめる。`amdxdna` を含むカーネルと、版の合ったファームウェア（`/usr/lib/firmware/amdnpu/`）が必要。 |
| `permission denied ... add the user to the render group` | `sudo usermod -aG render $USER` を実行し、ログインし直す。 |
| `create device heap` で失敗する | 別のプロセスが NPU の資源を使っているか、ファームウェアの読み込みに失敗している（`dmesg` を確認）。`ENOMEM` なら、固定できるメモリの上限を上げると直ることがある（手順 1）。 |
| `no valid DPU kernel found in the xclbin` | xclbin にカーネル ID に対応する PDI がない。正しい `--dev` でビルドし直す。 |
| `the xclbin needs N columns, the device has M` | 大きい NPU 向けにビルドした設計を使っている（`npu1` の機械で `npu2` 向けの設計など）。 |
| `command finished with ERT_CMD_STATE_ERROR` や `TIMEOUT` | ファームウェアが実行を拒否したか中断した。`dmesg` を確かめる。ドライバとファームウェアの版のずれは中断の原因になるので、同じ入手元から入れ直す。 |
| デバイスの電力モードを変えたい | root なら `xrt-smi` やドライバの `SET_STATE` ioctl で変えられる。他のプロセスにも影響するので、hwnpu は変更しない。 |
