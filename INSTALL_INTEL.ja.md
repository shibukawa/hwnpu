# Intel NPU を hwnpu で使うための準備

Intel Core Ultra（Meteor Lake、Arrow Lake、Lunar Lake、Panther Lake、Wildcat Lake）の NPU を hwnpu から使えるようにし、動作を確かめる手順です。

> **現状。** Intel バックエンドは、Level Zero ドライバを模した偽のドライバでしか試していません。実機で手順 4 の `cmd/report` を実行し、結果の報告を送っていただけるのがいちばん助かります。

English: [INSTALL_INTEL.md](INSTALL_INTEL.md)

## 対応プラットフォーム

| プラットフォーム | 対応 | 補足 |
|---|---|---|
| Linux（x86-64） | 対応 | Ubuntu 24.04 / 26.04 と Intel の NPU ドライバパッケージ（手順 1）。 |
| Windows 11（x86-64） | 対応 | Intel の NPU ドライバ（手順 2）。 |
| WSL2 | **非対応** | WSL2 が Linux 側に渡すのは GPU（`/dev/dxg`）だけで、NPU は WSL2 の中から見えない。Windows で直接使う。 |
| macOS | 該当なし | Mac には Intel NPU がない。Apple Silicon では hwnpu は Core ML を使う。 |

## hwnpu が必要とするもの

hwnpu は **Level Zero** と NPU のグラフ拡張を、purego で実行時に読み込んで NPU を使います。cgo も OpenVINO のランタイムも使いません。

| 部品 | Linux | Windows |
|---|---|---|
| カーネルドライバ | `intel_vpu`（標準カーネルに同梱） | Intel NPU ドライバ |
| ファームウェア | `intel-fw-npu` パッケージ | ドライバに同梱 |
| ユーザーモードドライバ | `intel-level-zero-npu`（`libze_intel_npu.so.1`） | ドライバに同梱 |
| ドライバ内コンパイラ（OpenVINO IR のコンパイル用） | `intel-driver-compiler-npu` | ドライバに同梱 |
| Level Zero ローダー | `libze1`（`libze_loader.so.1`） | `ze_loader.dll` |
| デバイスの権限 | `/dev/accel/accel0` の読み書き（`render` グループ） | 不要 |
| Go | 1.27 以降（`go.mod` を参照）。`CGO_ENABLED=0` で動く | 同じ |

ドライバ、ドライバ内コンパイラ、ファームウェアは、**同じリリース**のものをそろえてください。ずれていると、`MAPPED_INFERENCE_VERSION is NOT compatible with the ELF` のようなコンパイルエラーになります。

## 1. Linux（Ubuntu 24.04 / 26.04）

[intel/linux-npu-driver](https://github.com/intel/linux-npu-driver/releases) の手順に沿っています。以下のコマンドは **v1.38.0** 用です（2026-10-28 に確認）。実際に入れるときは、最新リリースのファイル名に読み替えてください。

古いパッケージを削除します。

```bash
sudo dpkg --purge --force-remove-reinstreq intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu intel-level-zero-npu-dbgsym
```

リリースを取得して展開します（Ubuntu 24.04 の例です。26.04 では `ubuntu2604` のアーカイブを使います）。

```bash
wget https://github.com/intel/linux-npu-driver/releases/download/v1.38.0/linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz
```

```bash
tar -xf linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz
```

ドライバ内コンパイラが依存するライブラリを入れてから、パッケージを入れます。

```bash
sudo apt update && sudo apt install libtbb12
```

```bash
sudo dpkg -i *.deb
```

Level Zero ローダーを入れます。リリースのページには、検証に使った `libze1` へのリンクがあります（v1.38.0 では v1.32.0）。

```bash
wget https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260830T100000Z/pool/main/l/level-zero-loader/libze1_1.32.0-1~24.04~ppa1_amd64.deb
```

```bash
sudo dpkg -i libze1_*.deb
```

古い Level Zero と衝突する場合は、先にそれを削除します。

```bash
sudo dpkg --purge --force-remove-reinstreq level-zero level-zero-devel
```

利用者をデバイスにアクセスできるグループに加え、ログインし直すか再起動します。

```bash
sudo gpasswd -a ${USER} render
```

デバイスノードがあり、アクセスできるかを確かめます。

```bash
ls -l /dev/accel/accel0
```

グループが `render`、権限が `crw-rw----` なら正常です。グループが `root` になっている場合は、次の再起動まで有効な応急処置として、次を実行します。

```bash
sudo chown root:render /dev/accel/accel0 && sudo chmod g+rw /dev/accel/accel0
```

## 2. Windows 11

1. Intel のダウンロードセンターか Windows Update から、最新の **Intel NPU ドライバ**を入れます。ドライバが動いていれば、タスクマネージャーのパフォーマンスに **NPU** が表示されます。
2. Level Zero ローダーがあるかを確かめます。

   ```powershell
   where.exe ze_loader.dll
   ```

   hwnpu は、通常の DLL 検索パスで `ze_loader.dll` を読み込みます。NPU ドライバがこの DLL をどこに置くかは、**まだ確認できていません**。`where.exe` で見つからない場合は、`C:\Windows\System32` やドライバのフォルダを探してください。見つからなければ [Level Zero のリリース](https://github.com/oneapi-src/level-zero/releases) から `ze_loader.dll` を取得し、その場所を hwnpu に指定します。

   ```powershell
   $env:HWNPU_ZE_LOADER = "C:\path\to\ze_loader.dll"
   ```

WSL2 は使えません。WSL2 に渡されるのは GPU（`/dev/dxg`）だけで、NPU は見えないためです。

## 3. hwnpu を取得する

```bash
git clone https://github.com/shibukawa/hwnpu.git
```

```bash
cd hwnpu
```

Go のない機械で試すときは、別の機械でクロスビルドした実行ファイルを持っていきます。テスト用のモデルは埋め込まれているので、ファイルは 1 つで済みます。

```bash
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o report.exe ./cmd/report
```

## 4. 動作を確かめる

```bash
go run ./cmd/hwnpuinfo
```

Intel NPU が `backend: levelzero` として表示されます。`driver` の行には、Level Zero の版、ドライバの版、グラフ拡張、コンパイラの版、NPU 用コマンドキュー拡張（ターボやワークロード種別）の有無が出ます。

次に報告を作り、Markdown のファイルを保存します。

```bash
go run ./cmd/report -o report.md
```

Intel NPU ごとに、2 つの OpenVINO IR モデルを厳格モードでコンパイルし、結果を fp32 の参照値と比べ、実行時間を測ります。NPU は内部で fp16 を使うので、小さな誤差は想定どおりです。

電力ポリシーの効果を比べるときは、2 回実行します。Linux では CPU パッケージの電力量（RAPL）を root しか読めないので、電力量の欄を出すには `sudo` で実行してください。

```bash
go run ./cmd/report -power low -o low.md
```

```bash
go run ./cmd/report -power performance -o performance.md
```

生成したモデルで NPU を測るには、npuprobe を使います。LLM の生成と同じ形の OpenVINO IR のモデルを Go で作り、0.5B と 1B の重みで fp16、int8、int4 の速さを測ります (`bench`。RAPL を読めるとき、つまり Linux で root のときは 1 トークンあたりのエネルギーも)。次に形式ごとにモデルを大きくしながら、ドライバがコンパイルできなくなる境目を二分探索で絞り、境目のすぐ下のモデルを複数同時に読み込みます (`limit`)。Python もモデルのファイルも要りません。数十分から 1 時間ほどかかり、結果は `npuprobe-<機種>-<日時>.md` に書き出します。クローンしなくても動きます。

```bash
go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest
```

先に `cmd/report` を動かしてください。そこで小さなモデルが失敗するなら、探索も同じように失敗します。結果の読み方は [PROBE.ja.md](PROBE.ja.md) にあります。

## 5. モデルの用意

hwnpu が受け取るモデルの形式は 2 つです。

- **OpenVINO IR**（`model.xml` と `model.bin`）。読み込むときにドライバ内コンパイラがコンパイルします。ONNX、PyTorch、TensorFlow からは、OpenVINO の `ovc` コマンドか `openvino.convert_model` で変換します（OpenVINO が要るのは変換する機械だけです）。NPU は動的な形状を扱えないので、形状は固定してください。
- **コンパイル済みの blob**。同じドライバとコンパイラで以前にコンパイルした結果です。`hwnpu.WithCacheDir(dir)` を指定すれば、hwnpu が blob を保存して再利用します。

```go
dev, _ := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorIntel), hwnpu.WithCacheDir(cacheDir))
gr, _ := dev.Graph()
g, err := gr.Compile(ctx, hwnpu.GraphSource{Path: "model.xml"})
```

NPU のコンパイラが対応しない演算を含むモデルは、`hwnpu.ErrNotOnNPU` で失敗します。エラーには、原因の演算とコンパイラのログが含まれます。

## うまくいかないとき

| 症状 | 原因と対処 |
|---|---|
| `runtime library not found ... libze_loader.so.1` | `libze1` を入れる（手順 1）か、`HWNPU_ZE_LOADER` にローダーのパスを指定する。 |
| `found /dev/accel/accel0 but cannot use it` | デバイスはあるが Level Zero の初期化に失敗している。理由はメッセージに含まれる。 |
| `/dev/accel/accel0` で `permission denied` | 利用者を `render` グループに加え、ログインし直す（手順 1）。 |
| デバイスがなく `/dev/accel/accel0` もない | `sudo dmesg \| grep -i vpu` でファームウェアの読み込みエラーを確かめる。ドライバと同じリリースの `intel-fw-npu` を入れ直す。 |
| `MAPPED_INFERENCE_VERSION is NOT compatible` | ドライバ、コンパイラ、ファームウェアのリリースがずれている。1 つのリリースのパッケージをすべて入れ直す。 |
| `the driver has no compiler for OpenVINO IR` | `intel-driver-compiler-npu` が入っていない。入れるか、コンパイル済みの blob を使う。 |
| `ZE_extension_graph version ... is not supported` | ドライバが古い（グラフ拡張 1.16 以降が必要。グラフ拡張 1.5 の Windows ドライバでは OpenVINO IR のビルドに失敗した）。ドライバを最新版に更新する。 |
| Level Zero の詳しい情報がほしい | `ZE_ENABLE_LOADER_DEBUG_TRACE=1` を付けて実行する。 |
