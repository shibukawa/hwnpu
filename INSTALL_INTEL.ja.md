# Intel NPU を hwnpu で使うための準備

Intel Core Ultra（Meteor Lake、Arrow Lake、Lunar Lake、Panther Lake、Wildcat Lake）の NPU を hwnpu から使えるようにし、動作を確かめる手順です。

> **現状。** Intel バックエンドは、Level Zero ドライバを模した偽のドライバでしか試していません。hwnpu-chat と hwnpu-chat agent は、同じプログラムを OpenVINO の CPU で動かして確かめました。実機で手順 4 の `cmd/report` と手順 6 のチャットを実行し、結果を送っていただけるのがいちばん助かります。

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

## 6. hwnpu-chat と hwnpu-chat agent

Mac と同じコマンドで、チャットとエージェントが Intel NPU で動きます (実機での確認はまだです。下の「確かめてほしいこと」を参照)。

```bash
go run github.com/shibukawa/hwnpu/cmd/hwnpu-chat@latest              # 端末で対話
go run github.com/shibukawa/hwnpu/cmd/hwnpu-chat@latest agent "README を要約して"
```

初回は次のことが起きます。

1. Hugging Face からチェックポイントを取ってくる (LFM3-1.7B で 3.5 GB)。
2. OpenVINO IR に変換する。置き場所はモデルのディレクトリの `intel/`。2026 年 11 月 4 日からは、Neural Engine と同じく全注意の層の前後で分けて (`-paged`) 文脈 262144 で変換する (`lfm3-int8-grouped-p8192-c262144-q8.parts.json` と `-q64`。それぞれ 7 部分と、重みのない注意のグラフ 1 つ)。1 ステップの費用は会話の長さに比例し、KV キャッシュのメモリも会話とともに増える (1 トークン 12 KiB)。OpenVINO の CPU (`-backend openvino`) では確かめたが、NPU の実機では未確認。分けない以前の形 (キャッシュ 4096 と 16320 のそれぞれに生成用と読み込み用の 4 つ、`lfm3-int8-c4096-q8.xml` など) は `-paged -1` で使える。
3. ドライバ内コンパイラが NPU 向けにコンパイルする。大きなモデルでは **1 つに数分かかることがあります**。結果はコンパイルキャッシュ (`-cache`、既定はユーザーキャッシュの `hwnpu`) に残るので、2 回目からは速くなります。

| フラグ | 意味 |
|---|---|
| `-backend` | 使う NPU。`levelzero` (`intel` でも同じ)、`coreml`、`openvino`。既定は `HWNPU_BACKEND`、なければ見つかった NPU |
| `-format` | チャットモデルの重みの形式。Intel では `int8` (LFM3 の既定)、`int4-g128`、`int4`、`fp16` |
| `-paged -1` | 全注意の層で分けない。グラフが部分ごとではなく 1 つになり、文脈の容量に比例した費用になる (そのとき `-tier` が効く) |
| `-tier none` | (`-paged -1` のとき) 短いキャッシュの段を作らない。グラフが 4 つから 2 つになり、初回のコンパイルとメモリが半分になる |
| `-context 4096` | 会話の長さ (KV キャッシュ)。短いほどコンパイルも 1 トークンも速い |

最初は小さく始めると切り分けやすいです。

```bash
hwnpu-chat --paged -1 --tier none --context 4096
```

### 確かめてほしいこと

1. `hwnpu-chat --paged -1 --tier none --context 4096` が `ready:` まで進むか。止まったら、そのときのメッセージ (コンパイラのログを含む) をください。
2. 質問に正しく答えるか。トークン/秒 (答えの後に表示) はいくつか。
3. 既定の設定 (`hwnpu-chat` だけ) で読み込めるか、コンパイルにどれくらいかかるか、メモリは足りるか。
4. `hwnpu-chat agent` がツールを呼べるか。
5. `hwnpu-formats` で int8、int4-g128、int4 の精度と速さを比べる (下の節)。

### 重みの形式を比べる (hwnpu-formats)

`hwnpu-formats` は、チャットモデルを形式ごとに変換して NPU に載せ、決まった会話 (日本語と、英語とコード) を hwnpu-chat と同じ 1 回 8 トークンで読ませて、次の表を出します。

| 列 | 意味 |
|---|---|
| ms per call | 1 回の呼び出しの時間 (中央値)。生成の速さの目安 |
| next token as reference | 各位置で最も確からしい次のトークンが、基準の形式 (既定は fp16) と同じだった割合 |
| logits rel. err. | ロジットの基準からの相対誤差 |
| perplexity | 答えの部分のパープレキシティ。低いほどよい。基準がなくても比べられる |

```bash
go run ./cmd/hwnpu-formats                                    # int8、int4-g128、int4 を fp16 と比べる
go run ./cmd/hwnpu-formats -formats int4-g128 -ref int8       # fp16 が NPU に載らないとき
go run ./cmd/hwnpu-formats -json formats.json                 # 結果をファイルにも書く
```

基準の fp16 は重みが 3.2 GiB あり、NPU のコンパイルに時間がかかるか、失敗することがあります。そのときは `-ref int8` (fp16 とほぼ同じ) にするか、OpenVINO が入っていれば基準だけ CPU で計算します。

```bash
HWNPU_OV_PYTHON=~/ov/bin/python go run ./cmd/hwnpu-formats -ref-backend openvino
```

変換結果はモデルのディレクトリの `intel/formats/` に残り、2 回目からは変換もコンパイルも省かれます。Linux x86 の OpenVINO CPU (`-backend openvino`) では次のようになりました。NPU での数字を送ってもらえると、Intel の既定の形式を決められます。

| 形式 | 重み (GiB) | 次のトークンが fp16 と同じ | パープレキシティ |
|---|---|---|---|
| fp16 | 3.22 | (基準) | 3.81 |
| int8 | 1.61 | 98% | 3.84 |
| int4-g128 | 0.83 | 80% | 4.28 |
| int4 | 0.81 | 76% | 4.89 |

形式を決めたら、hwnpu-chat と hwnpu-chat agent では `--format int4-g128` のように指定します。

### OpenVINO で切り分ける

hwnpu の Level Zero の経路で失敗するとき、同じ IR を OpenVINO で動かすと、hwnpu の問題かモデル (IR) の問題かを切り分けられます。OpenVINO の入った Python を用意して:

```bash
python3 -m venv ~/ov && ~/ov/bin/pip install openvino numpy
HWNPU_OV_PYTHON=~/ov/bin/python HWNPU_OV_DEVICE=NPU hwnpu-chat --backend openvino --tier none --context 4096
```

`HWNPU_OV_DEVICE=NPU` は OpenVINO の NPU プラグイン、`CPU` (既定) は CPU で動かします。Windows では `HWNPU_OV_PYTHON` に `python.exe` のパスを指定します。

## うまくいかないとき

| 症状 | 原因と対処 |
|---|---|
| `runtime library not found ... libze_loader.so.1` | `libze1` を入れる（手順 1）か、`HWNPU_ZE_LOADER` にローダーのパスを指定する。 |
| `found /dev/accel/accel0 but cannot use it` | デバイスはあるが Level Zero の初期化に失敗している。理由はメッセージに含まれる。 |
| `/dev/accel/accel0` で `permission denied` | 利用者を `render` グループに加え、ログインし直す（手順 1）。 |
| デバイスがなく `/dev/accel/accel0` もない | `sudo dmesg \| grep -i vpu` でファームウェアの読み込みエラーを確かめる。ドライバと同じリリースの `intel-fw-npu` を入れ直す。 |
| `MAPPED_INFERENCE_VERSION is NOT compatible` | ドライバ、コンパイラ、ファームウェアのリリースがずれている。1 つのリリースのパッケージをすべて入れ直す。 |
| `the driver has no compiler for OpenVINO IR` | `intel-driver-compiler-npu` が入っていない。入れるか、コンパイル済みの blob を使う。 |
| `ZE_extension_graph version ... is not supported` | ドライバが古い（グラフ拡張 1.5 以降が必要）。ドライバを更新する。 |
| hwnpu-chat が `chain: part 0 has no output y` で止まる | ドライバが返す引数の名前が IR の名前と合わない。メッセージと `hwnpuinfo` の結果を送ってほしい。 |
| hwnpu-chat の読み込み中にメモリが足りなくなる | `--tier none` でグラフを半分にする。`--format int4-g128` で重みを半分にする。 |
| Level Zero の詳しい情報がほしい | `ZE_ENABLE_LOADER_DEBUG_TRACE=1` を付けて実行する。 |
