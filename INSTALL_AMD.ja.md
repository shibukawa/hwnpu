# AMD Ryzen AI の NPU を hwnpu で使うための準備

AMD Ryzen AI プロセッサ（XDNA NPU を積んだ Phoenix、Hawk Point、Strix Point、Krackan Point 以降）を載せた Linux 機で、hwnpu から NPU を使えるようにする手順です。小さなテスト用の設計をビルドし、動作を確かめるところまで扱います。

> **現状。** Ryzen AI 9 365（Strix Point、`npu4`、8 列）、Ubuntu 26.04 の Linux 7.0 標準カーネルの amdxdna、ファームウェア 1.1.2.64 で、手順 4 の `hwnpu-dev report` が mlir-aie v1.4.4 の passthrough 設計を含めて全て通りました（2026-11-03）。Phoenix と Hawk Point（`npu1`）はまだ試していません。hwnpu は NPU1 向けに書き出したモデルを「1 ステップごとの xclbin」として動かします。同じ形で `npu2` 向けに書き出したもの（`--per-step`）は Ryzen AI 9 365 で動きましたが、NPU1 の実機では動かしていません。別の機械での報告を送っていただけると助かります。

English: [INSTALL_AMD.md](INSTALL_AMD.md)

## 対応プラットフォーム

| プラットフォーム | 対応 | 補足 |
|---|---|---|
| Linux（x86-64） | 対応 | `amdxdna` ドライバを含むカーネル（手順 1）。 |
| Windows（x86-64） | **未確認** | XRT と、hwnpu 自身の小さな C ライブラリ xrtshim を通して使う（手順 6）。Linux の XRT 2.21 では確かめたが、Windows ではまだ動かしていない。 |
| WSL2 | **非対応** | WSL2 は NPU を Linux 側に渡さない（WSL2 の中に `/dev/accel` がない）。Windows そのもの（手順 6）か、Linux を直接インストールした環境で使う。 |
| macOS | 該当なし | Mac には AMD の NPU がない。 |

## hwnpu が必要とするもの

Linux では、hwnpu はカーネルドライバ **amdxdna** を `ioctl` と `mmap` で直接呼びます。実行時に XRT は使いません。Windows では XRT を通します（手順 6）。

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

`hwnpu-dev report` は IRON のサンプル `passthrough_dmas` を実行します。このサンプルは NPU の DMA でバッファを写すだけなので、AIE コア用のコンパイラが要りません。

IRON は **Python 3.12** を必須にしています。ディストリビューションの Python が別の版なら、[uv](https://docs.astral.sh/uv/) で入れられます（root は不要）。

```bash
uv python install 3.12
```

リリースのタグを取得します。タグのない版では、ビルド済みの wheel が見つからずに止まります。

```bash
git clone --depth 1 --branch v1.4.4 https://github.com/Xilinx/mlir-aie.git
```

```bash
cd mlir-aie
```

```bash
PATH=$(dirname $(uv python find 3.12)):$PATH source utils/env_install.sh --no-pre-commit ironenv
```

```bash
source utils/env_setup.sh
```

最後に `xrt-smi not found` と出ますが、hwnpu は XRT を使わないので無視できます。

xclbin をまとめる `xclbinutil` は XRT の道具です。XRT のパッケージ（手順 1 の PPA の `libxrt-utils`、または Ubuntu 26.04 の universe の `libxrt-utils`）を入れていればそのまま使えます。root がなく XRT を入れられない場合は、パッケージを展開して使えます（Ubuntu 26.04 の例）。

```bash
mkdir -p ~/xrt-debs && cd ~/xrt-debs
```

```bash
apt-get download libxrt-utils libxrt2 libboost-filesystem1.90.0 libboost-program-options1.90.0
```

```bash
for d in *.deb; do dpkg -x $d root; done
```

```bash
printf '#!/bin/sh\nLD_LIBRARY_PATH=%s/root/usr/lib/x86_64-linux-gnu exec %s/root/usr/bin/xclbinutil "$@"\n' $PWD $PWD > xclbinutil && chmod +x xclbinutil
```

```bash
export AIE_XCLBINUTIL=~/xrt-debs/xclbinutil
```

XRT をまったく使わない方法もあります。hwnpu には、aiecc が `xclbinutil` に渡す引数を受け付ける代わりの道具 `hwnpu-dev xclbin` があります。hwnpu のチェックアウト（手順 3）でビルドし、aiecc に渡します。

```bash
go build -o ~/.local/bin/hwnpu-xclbin ./cmd/hwnpu-dev   # hwnpu-xclbin（か xclbinutil）という名前の hwnpu-dev は xclbin として動く
```

```bash
export AIE_XCLBINUTIL=~/.local/bin/hwnpu-xclbin
```

この xclbin には hwnpu が読むもの（カーネルの情報、計算ユニット、AIE のパーティションと PDI）だけが入り、XRT 向けではありません。IRON の演算子がビルドする xclbin もまとめられます（Strix Point の GEMV で確認。NPU1 は未確認）。

xclbin と命令列だけをビルドします。Phoenix と Hawk Point では `-d npu` を、Strix Point 以降では `-d npu2` を指定します。

```bash
cd programming_examples/basic/passthrough_dmas
```

```bash
python3 passthrough_dmas.py -d npu2 -n 4096 --xclbin-path=build/final.xclbin --insts-path=build/insts.bin
```

この設計は int32 を 4096 個（16384 バイト）写します。これは `hwnpu-dev report` の既定値と同じです。別の `-n` でビルドしたときは、報告プログラムに `-bytes` で大きさを渡してください。

## 3. hwnpu を取得する

```bash
git clone https://github.com/shibukawa/hwnpu.git
```

```bash
cd hwnpu
```

## 4. 動作を確かめる

```bash
go run ./cmd/hwnpu-dev info
```

AMD の NPU が `backend: xdna` として表示されます。世代、AIE の列数、AIE の版、ファームウェアの版も表示されます。

手順 2 で作った設計を渡して、報告を作ります。

```bash
go run ./cmd/hwnpu-dev report -xclbin /path/to/final.xclbin -insts /path/to/insts.bin -o report.md
```

最初に、バッファの確保とキャッシュの同期、デバイスヒープ上の命令列用バッファ、NPU の電力センサーを確かめます。NPU の電力センサー（`QUERY_SENSORS`）は AMD の DKMS 版のドライバにしかありません。Linux 7.0 の標準カーネルでは、amdgpu ドライバが出すパッケージ全体の電力（PPT、root 不要、APU 全体の値）を読みます。続いて設計を読み込んで実行し、出力が入力と一致するかを確かめ、実行時間と NPU の電力量を測ります。`-xclbin` を省くと、カーネルの確認は飛ばします。xclbin 内のカーネル名が `MLIR_AIE` でない場合は、`-kernel 名前` で指定してください。

電力ポリシー（ハードウェアコンテキストの優先度）の効果を比べるときは、2 回実行します。

```bash
go run ./cmd/hwnpu-dev report -xclbin final.xclbin -insts insts.bin -power low -o low.md
```

```bash
go run ./cmd/hwnpu-dev report -xclbin final.xclbin -insts insts.bin -power performance -o performance.md
```

Strix Point でのデータ型ごとの性能と DMA の帯域、それを測ったスクリプトは [docs/amd-xdna2-bench.md](docs/amd-xdna2-bench.md) と [tools/iron](tools/iron/README.md) にあります。

### hwnpu-chat のモデル

hwnpu-chat は `tools/amd/iron_export_lm.py` の書き出し (`~/models/NAME.amd-npu2-q4`) を動かします。モデルの書き出しがなければ、初回にチェックポイントを落とし、手順 2 の IRON の環境で、models.yaml の `amd_export` のとおりに書き出します。その環境は `HWNPU_IRON_ENV` で渡します: 仮想環境のディレクトリか、環境を整えるスクリプト (bash で source するもの。`source utils/env_setup.sh` と `AIE_XCLBINUTIL` の設定をするものなど)。なければ PATH の python3 を試し、IRON を import できなければ、何を入れればよいかを言って止まります。

```bash
HWNPU_IRON_ENV=~/mlir-aie/ironenv go run ./cmd/hwnpu-chat "こんにちは"
```

ある機械で作った書き出しは、同じ NPU のほかの機械でも動きます (ディレクトリを写すか、`-m` に渡す)。

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
- 引数の並びと大きさは xclbin のメタデータから読みます。IRON の設計では `opcode（3）、instr、ninstr、各バッファ` の順です。mlir-aie v1.4.4 は、使わないバッファも `bo0`〜`bo4` として署名に並べます。設定しなかった引数は、XRT と同じく 0 で送ります。

## 6. Windows（未確認）

AMD の Windows 用 NPU ドライバは、ドライバが入れる XRT の C++ ライブラリ `xrt_coreutil.dll` を通してしか使えません。hwnpu は cgo を使わないので、hwnpu 自身の C ライブラリ **xrtshim**（`native/xrtshim`）を実行時に読み込み、それを通して XRT を呼びます（`xrt` バックエンド、`internal/backend/amdxrt`）。同じバックエンドは Linux の XRT のパッケージでも動き、そこで確かめています。Ryzen AI 9 365 と XRT 2.21.75 で、`hwnpu-dev report` の passthrough 設計（xclbin）、呼び出しごとの値を持つ IRON の full ELF、画像と音声を含む Gemma 4 E2B が、Linux のドライバと同じ速さで動きました（2026-11-04）。Windows ではまだ動かしていません。報告を送っていただけると助かります。

1. NPU のドライバ（AMD の Ryzen AI ドライバ）を入れます。`C:\Windows\System32\AMD\xrt-smi.exe examine` で NPU が見えれば準備できています。full ELF（Gemma 4 E2B）には新しいドライバが要ります。ELF からハードウェアコンテキストを作れる XRT（`xrt::hw_context(device, xrt::elf)`。XRT 2.21 はできる）を含むものです。
2. Visual Studio 2022（C++ のワークロード）と CMake で xrtshim をビルドします。「Developer PowerShell for VS」で次を実行します。

   ```powershell
   cd native\xrtshim
   .\make-importlib.ps1          # ドライバの xrt_coreutil.dll から xrt_coreutil.lib を作る
   cmake -B build -DXRT_LIBRARY=xrt_coreutil.lib
   cmake --build build --config Release
   ```

   XRT のヘッダ（2.21）は `native/xrtshim/include` にあります。ドライバの XRT の版が大きく違うと C++ の ABI が合わないことがあるので、その版のヘッダを XRT のソースから取って `-DXRT_INCLUDE_DIR` で渡してください。
3. `build\Release\xrtshim.dll` を hwnpu の実行ファイルと同じ場所に置きます（または `HWNPU_XRTSHIM` で場所を指定します）。`xrt_coreutil.dll` は `System32\AMD` かドライバストアから探します。別の場所なら `HWNPU_XRT_CORE` で指定します。
4. hwnpu を Windows 向けにビルドし（`GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build ./cmd/...`）、`hwnpu-dev info` を実行します。バックエンド `xrt` の NPU が出れば使えます。

モデルの書き出しは Linux で行い（IRON と mlir-aie。README を参照）、書き出したディレクトリを持っていきます。`hwnpu-chat -m DIR` で同じように動きます。Linux との違いは次のとおりです。

- 電力は、ドライバが XRT の electrical の情報に NPU の電力を出していればそれを読みます。
- 制御コードの置き場は XRT が確保します。Linux 7.0 ではデバイスのヒープが 64 MiB で、hwnpu 自身のドライバ経路と同じです。Windows では違うかもしれません。

Linux では、`xrt` バックエンドは `HWNPU_BACKEND=xrt` のときだけ NPU を出します（xrtshim を `native/xrtshim/build-linux.sh` でビルドし、`HWNPU_XRTSHIM` で指定します）。

## うまくいかないとき

| 症状 | 原因と対処 |
|---|---|
| `hwnpu-dev info` に AMD のデバイスが出ない | `ls /dev/accel/` と `sudo dmesg \| grep -i amdxdna` を確かめる。`amdxdna` を含むカーネルと、版の合ったファームウェア（`/usr/lib/firmware/amdnpu/`）が必要。 |
| `permission denied ... add the user to the render group` | `sudo usermod -aG render $USER` を実行し、ログインし直す。 |
| `create device heap` で失敗する | 別のプロセスが NPU の資源を使っているか、ファームウェアの読み込みに失敗している（`dmesg` を確認）。`ENOMEM` なら、固定できるメモリの上限を上げると直ることがある（手順 1）。 |
| `no valid DPU kernel found in the xclbin` | xclbin にカーネル ID に対応する PDI がない。正しい `--dev` でビルドし直す。 |
| `the xclbin needs N columns, the device has M` | 大きい NPU 向けにビルドした設計を使っている（`npu1` の機械で `npu2` 向けの設計など）。 |
| `command finished with ERT_CMD_STATE_ERROR` や `TIMEOUT` | ファームウェアが実行を拒否したか中断した。`dmesg` を確かめる。ドライバとファームウェアの版のずれは中断の原因になるので、同じ入手元から入れ直す。 |
| デバイスの電力モードを変えたい | root なら `xrt-smi` やドライバの `SET_STATE` ioctl で変えられる。他のプロセスにも影響するので、hwnpu は変更しない。 |
