# Mac 1 台でモデルを作る (計画)

作成: 2026-11-19。状態: 計画 (未実装)。取得元、版、`files.tar.zst`、書き戻しの仕組みは [cache-sharing-design.md](cache-sharing-design.md) を正とし、ここでは「どの機械で作るか」だけを決め直す。[prebuilt.md](prebuilt.md) 5 節と cache-sharing-design.md 9 節の「aix1 を self-hosted runner に」を置き換える案を含む (6 節)。

## 0. 要点

- Mac の上の Go (`hwnpu-dev`) が全体を回す: 計画、取得元 (NAS、HF) からの取得、詰める (zstd)、書き戻し。Linux の道具 (IRON、mlir-aie、Peano) が要る AMD の書き出しだけを、**linux/amd64 のコンテナ** (Rosetta) で動かす
- つなぎ目は今の `HWNPU_IRON_ENV` で足りる。実行できるファイルを指せばそれを python として呼ぶので、`docker run ... python3 "$@"` をする小さなスクリプトを置けば、Go を変えずに `hwnpu-dev cache export` が Mac で動く (3.3)
- コンテナはネットワークなし、資格情報なし、計算だけ。ファイルはキャッシュの根を**同じパス**で見せる
- 調べて分かった壁が 3 つ: (1) mlir-aie v1.4.4 と llvm-aie (Peano) の wheel は Linux では x86_64 しかない (arm64 では動かない)、(2) FLUX.3 klein の書き出しは最大 RSS 16.8 GB で M1 (16 GB) を超える、(3) M1 の空きは 51 GB で、FLUX.3 は元 20 GB + 書き出し 15 GB + 詰めたもの 13.8 GB が同時に要る
- AMD の NPU の上での照合 (ironcheck、KL) は Mac ではできない。最初の確かめは「aix1 で作ったものとバイトまで同じか」で行う (5 節)

## 1. 目的と範囲

- 今は Mac (Core AI の書き出し、`coreai-build` の prebuilt) と aix1 (Linux。AMD の書き出し) の 2 台が要る。これを Mac 1 台で全部作れるようにする
- Mac は外せない: Core AI の変換は Go でどこでもできるが、`coreai-build` (Metal Toolchain、Xcode 27) と ANE での確かめは Mac にしかない
- Linux の側は計算だけ (IRON の書き出しは CPU だけで、NPU を使わない: cache-sharing-design.md 7.10 節の `hwnpu-dev cache export`)。だからコンテナに閉じ込められる
- 範囲: 元のチェックポイントの取得 (NAS、なければ HF)、AMD と Core AI の書き出し、画像のモデルの 8 つのアスペクト比、zstd で詰める、取得元への書き戻し、片付け
- 範囲外: AMD の NPU の上での照合 (aix1 か利用者の機械に残す)、Intel (OpenVINO の変換は Go なので、もともと Mac で作れる)

## 2. もう揃っているもの

| もの | どこ | このあとの使い方 |
|---|---|---|
| 取得元の並び (`HWNPU_MODEL_SOURCE`、`_1`…、`?writeback=true`、最後に HF) | `internal/store`、cache-sharing-design.md 7 節 | そのまま。NAS (WebDAV) を先頭に、HF を最後に |
| 元のチェックポイントのフォワードプロキシ | 7.1 節 | NAS になければ HF から同じコミットで取り、NAS に置く |
| 作り方の鍵 (`ExportRecipe`、`PinnedRecipe`) と「あれば作らない」 | `internal/chat/sharedexport.go` | 何度走らせても、取得元にあるものは飛ばす (冪等) |
| 詰める: tar を zstd (`klauspost/compress`、`SpeedDefault`) で `files.tar.zst` 1 つに | `internal/store/pack.go`、7.10 節 | そのまま。改める点は 4.4 |
| `hwnpu-dev cache export MODEL...` (chat の `amd_export` を CPU だけで作って書き戻す) | `cmd/hwnpu-dev/cachecmd` | コンテナでの書き出しの入口 |
| `hwnpu-dev prebuild -arch all -put EXPORT` (Core AI の 12 種類の prebuilt) | 7.9 節 | Mac でそのまま |
| `hwnpu-dev xclbin` (XRT の `xclbinutil` の代わり) | INSTALL_AMD.md 手順 2 | コンテナに XRT を入れずに済む |
| `HWNPU_IRON_ENV` が実行できるファイルを受ける (`.sh` でなく実行ビットがあれば python として呼ぶ) | `internal/chat/amdprep.go` の `ironPython` | コンテナへのつなぎ目 (3.3) |
| 画像のモデルの 8 つの大きさ (`npu.amd.sizes`、`npu.apple.sizes`)。AMD は 8 つで 1 つの書き出し | models.yaml、design.md 16.251 | 作る単位 (4.2) |

## 3. コンテナ

### 3.1 なぜ linux/amd64 (Rosetta) か

調べた結果 (2026-11-19、GitHub のリリース):

- mlir-aie v1.4.4: `manylinux_2_35_x86_64` と `win_amd64` だけ (cp311〜cp314)
- llvm-aie (Peano) の nightly: `manylinux_2_28_x86_64` と `win_amd64` だけ

arm64 の Linux で動かすにはどちらもソースから作ることになる (LLVM 2 本。16 GB の M1 では数時間で、版を上げるたびにやり直し)。まずは linux/amd64 のイメージを Rosetta で動かす。arm64 にするのは、Rosetta の遅さ (5 節で測る) が困るほどだったときだけ。

### 3.2 イメージ (`tools/container/iron/Dockerfile`)

```
FROM --platform=linux/amd64 ubuntu:24.04
uv と Python 3.12
mlir_aie==1.4.4 (wheel)、llvm_aie==<固定の nightly> (wheel)、amd/IRON (固定のコミット)
numpy、ml_dtypes、safetensors、torch (CPU 版)、transformers、diffusers、onnx、Pillow
/opt/hwnpu/xclbin  ← hwnpu-dev xclbin (GOOS=linux GOARCH=amd64 CGO_ENABLED=0 で作る)
ENV AIE_XCLBINUTIL=/opt/hwnpu/xclbin
```

- 版は全部固定する。tools/amd の `import` が要るものは今 `iron`、`aie`、`numpy`、`torch`、`transformers`、`ml_dtypes`、`safetensors`、`diffusers`、`PIL`、`onnx`、`depth_anything_3`
- **Peano の nightly は消える**。リリースの `nightly` に残るのは最近のものだけなので、使う wheel を NAS (`tools/wheels/`) に置き、イメージはそこから入れる。mlir-aie の wheel も同じく置く
- できたイメージも `docker save | zstd` で NAS に置く (`tools/images/iron-<版>.tar.zst`)。作り直せなくなっても戻せるように
- イメージの版 (`HWNPU_IRON_IMAGE`、例 `hwnpu-iron:1.4.4-p2026111801`) は作り方の鍵に入れる (4.5)

### 3.3 つなぎ目: `tools/container/iron-python`

`HWNPU_IRON_ENV` に置く実行できるスクリプト (`.sh` を付けない。付けると `source` される)。`RunIRON` は `python ARGS` を、作業ディレクトリを展開した `tools/` にして呼ぶ。スクリプトはそれをそのままコンテナに渡す。

```sh
#!/bin/sh
root="${HWNPU_CACHE:-$HOME/Library/Caches/hwnpu}"
exec docker run --rm -i --platform linux/amd64 --network none \
  --user "$(id -u):$(id -g)" \
  --memory "${HWNPU_IRON_MEMORY:-12g}" \
  -v "$root:$root" -w "$PWD" \
  -e PYTHONUNBUFFERED -e NPU_CACHE_HOME -e HOME=/tmp \
  "${HWNPU_IRON_IMAGE:-hwnpu-iron:latest}" python3 "$@"
```

- **同じパスで見せる** (`-v $root:$root`): Go が渡すパス (チェックポイント `original/OWNER/NAME`、出力の `NAME.part`、`tools/`、IRON のカーネルのキャッシュ `NPU_CACHE_HOME`) を読み替えずに済む。全部キャッシュの根の下にある
- `--network none`: 取得と書き戻しは Mac の Go がする。コンテナに NAS のパスワードも HF のトークンも渡らない。書き出しの道具がネットに出ようとしたら (HF の `from_pretrained` がファイルを探しに行くなど) 落ちるので、作り方の漏れが分かる
- `--user`: 書いたファイルの持ち主を Mac の利用者に
- ロック (`cache.Lock`) は Mac の Go が持つ。コンテナの中は 1 回の書き出しだけ
- `ironPython` の確かめ (`python -c "import iron, aie.iron"`) もこのスクリプトを通るので、イメージがなければ今と同じ言葉で止まる
- コンテナのエンジンは docker 互換の CLI ならどれでもよい (`HWNPU_IRON_DOCKER` で `docker`、`podman`、`container` を選べるようにする)。候補は 7 節

### 3.4 Go の側で足すもの (小さい)

- `ironPython` の確かめの時間 (2 分) はコンテナの初回の起動でも足りるはず。足りなければ伸ばす
- `-e` で渡す環境変数を増やすときは、スクリプトの側で足す (Go は `os.Environ()` を渡すだけで、コンテナには届かない)。今要るのは `NPU_CACHE_HOME` と `PYTHONUNBUFFERED`

## 4. 作る流れ: `hwnpu-dev build`

### 4.1 コマンド

今の `hwnpu-dev cache export` (chat だけ) と `prebuild` を 1 つの入口にまとめる。

```
hwnpu-dev build [MODEL...] [--target amd-xdna2,coreai,coreai-ARCH] [--sizes all] [--dry-run] [--keep]
```

- 引数なしは「models.yaml の `targets` に版のあるもの全部」。`--dry-run` は計画だけを出す (どれが取得元にあり、どれを作るか、要るディスクとメモリ)
- 機械の NPU を見ない。今の `OnlyAMD` のような「この機械で動かす形」の判定を通らずに、名指した対象のものを作る (今の画像の `PrepareAMD` は AMD の機械の `hwnpu-image` からしか呼ばれないので、呼べる形に出す)
- 1 つずつ順に (並べない)。メモリとディスクが 1 本分しかない (4.3)

### 4.2 作る単位 (行列)

| 種類 | 対象 | 1 つの単位 | 道具 | どこで |
|---|---|---|---|---|
| チャット (`amd_export`) | `amd-xdna2` | モデル | `iron_export_lm.py` | コンテナ |
| 画像の生成 (FLUX.2 klein、FLUX.3 klein) | `amd-xdna2` | モデル。**8 つの大きさで 1 つ** (`npu.amd.sizes`) | `iron_flux2klein.py`、`iron_flux3klein.py` | コンテナ |
| 検出、姿勢、切り抜き | `amd-xdna2` | モデル | `iron_onnxcnn.py`、`iron_efficientsam.py` | コンテナ |
| 手で作っている AMD の書き出し (Anima、Chroma、深さ、顔、ラベル) | `amd-xdna2` | 今は `iron-import` | 道具ごと | コンテナ (作り方を models.yaml に書いてから) |
| Core AI の書き出し | `coreai` | モデル × 書き出し。画像は `npu.apple.sizes` の 8 つ | Go (`ailower`) | Mac |
| Core AI の prebuilt | `coreai-ARCH` | 書き出し × 12 種類 | `coreai-build` | Mac (効くモデルだけ: 7.9 節) |
| 表 | `any` | モデル | Go | Mac |

画像の 8 つの大きさ:

- AMD は今のとおり 1 つの書き出しに 8 つ (512² 以外は 1024² の `dit_in`、mode ブロック、`dit_out` を使い、大きさごとは singles と VAE だけ)。だから作る単位はモデル 1 つで、大きさは鍵 (`--size` の並び) に入る
- Core AI は大きさごとに関数 (1 本のプログラムに関数を並べる、new-model-checklist の 3.5)。配るときは 8 つを全部作る (`--sizes all`)。手元の初回は今どおり頼まれた大きさだけ
- 大きさの並びは models.yaml の `npu.<会社>.sizes` だけが決める。並びを変えたら鍵が変わり、作り直しになる (版を上げるのと同じ扱い)
- prebuilt は大きさ × 12 種類で数が多い (Core AI の仕上げは機械全体で 10 秒に 1 回: design.md 16.233)。初回が長い大きさ (1024² の DiT、VAE) だけに絞る

### 4.3 1 つの単位の流れ

```
1. 計画     鍵を出す (4.5)。取得元にあれば飛ばす
2. 空き     要るディスク = 元 + 書き出し + 詰めたもの + 15 GB (macOS が purgeable を消し始める線)
            足りなければ、前の単位の残り (書き出し、元) を消してから。それでも足りなければ止める
3. 元       NAS の original/ から 4 本ずつ、なければ HF から同じコミットで (NAS に書き戻す)
4. 作る     AMD はコンテナ、Core AI は Mac で
5. 詰める   files.tar.zst (4.4)
6. 置く     書き戻す取得元へ。entry.json を最後に
7. 確かめ   置いたものの大きさと sha256 を取得元から読み直す (HEAD と entry.json)
8. 片付け   既定では手元の書き出しと詰めたものを消す (--keep で残す)。元は同じ元を使う次の単位が済むまで残す
```

- 同じ元を使う単位は続けて回す (FLUX.3 の `amd-xdna2` と `coreai`、LFM3 の 4 つ)。元を 1 回だけ落とす
- 途中で止まっても、次の実行は 1 の計画から。置き終えたものは飛ばし、作りかけ (`.part`) は作り直す (今の `RunIRON` と同じ)
- 書き戻しが切れたら (NAS が止まっている)、今は裏の列 (`work/queue`) に積む。`build` では裏に回さず、その単位を失敗にして次へ (片付けで消してしまわないように)

### 4.4 zstd

今の形 (`files.tar.zst`、zstd の既定の速さ) を変えない。測った縮み方は 7.10 節のとおり: ELF は 1/60、重みのバンクは 0.78、Core AI の書き出しは 1〜5 %。`-19` にしても縮むのはプログラムだけで、時間に見合わない (Core AI の prebuilt だけはもともと `-19`)。

改めるのは 2 つ:

- **ディスク**: 今は書き出しを `work/pack/` に詰めてから送るので、書き出しと詰めたものが同時に要る (FLUX.3 で 15 + 13.8 GB)。詰めながら送る形 (tar → zstd → sha256 → PUT を 1 本の流れに) にすると、詰めたものを手元に持たずに済む。WebDAV は `Transfer-Encoding: chunked` の PUT、S3 はマルチパート。sha256 と大きさは流しながら数え、entry.json に書く。切れたら最初から送り直す (続きからはできない)。手元に置く今の形も残し、空きで選ぶ
- **速さ**: エンコーダの並列 (`zstd.WithEncoderConcurrency`) を M1 の 4 つの P コアに。NAS への送りが 1 本 5〜40 MB/s なので、詰める側が遅れなければよい

### 4.5 鍵に足すもの

今の `ExportRecipe` の鍵は、モデル、コミット、道具の名前と引数、tools/amd の hash で、**道具の環境 (mlir-aie、Peano、IRON の版) が入っていない**。aix1 と Mac のコンテナで版が違えば、違うものが同じ鍵で置かれうる。

- 鍵に `toolchain` を足す: イメージの版 (3.2) か、aix1 では `pip show` から作った同じ形の文字列 (`mlir_aie=1.4.4 llvm_aie=22.0.0.2026111801 iron=<commit>`)
- 足すと今の取得元のものに当たらなくなる。足すのは 5 節で「同じ版なら aix1 と同じバイト」を確かめてから、aix1 の版と同じ文字列で始める (今のものを置き直さずに済むように、文字列がないものは aix1 の今の版とみなす)

## 5. 確かめ方

Mac では AMD の NPU を動かせない (`xdnasim` はドライバの真似で、計算はしない)。だから:

1. **同じバイトか** (最初の確かめ): aix1 で作って NAS にあるものと同じ鍵の書き出しを Mac のコンテナで作り、`entry.json` の `files` の sha256 を全部比べる。同じなら、aix1 の上での照合 (ironcheck、KL、絵) がそのまま Mac で作ったものにも言える。小さいもの (Qwen3.5-0.8B のチャット、1.18 GB) から始め、次に FLUX.2 klein (8 つの大きさ)
   - 違ったら: どのファイルが違うかを見る。バンクの並べ方 (numpy の浮動小数の順)、ELF の中の時刻やパス、Python の dict の順など、決定的でないところを道具の側で直す。Rosetta の浮動小数の違い (x87 や FMA) なら、その計算を整数か numpy の決まった順に寄せる
2. **新しいモデル**: 初めて作るものは比べる相手がない。entry.json に `checked: {npu: false}` を書いて置き、AMD の機械が最初に使うときに照合して書き戻す (`hwnpu-dev cache verify --npu`、新しく)。それまで HF の `main` には進めない (cache-sharing-design.md 9 節の `staging`)
3. Core AI の書き出しは Mac の ANE で今どおり確かめられる。ほかのセッションが ANE を使っていることがあるので、`neural-engine.lock` を取って回す

## 6. GitHub Actions との関係

prebuilt.md 5 節と cache-sharing-design.md 9 節は aix1 を runner にする案だった。Mac 1 台で作れるなら:

- runner は Mac 1 台 (ラベル `macos-ane`、`iron-container`)。aix1 は照合だけ (5 節の 2) をする 2 台目に下げるか、なくす
- 公開リポジトリの self-hosted の注意 (PR では動かさない、actor の制限、runner group、専用のユーザー) はそのまま。コンテナの `--network none` は、ここでも作り方の漏れの検査になる
- 専用のユーザーで動かすなら、キャッシュの根もそのユーザーのもの。開発のキャッシュ (`~/Library/Caches/hwnpu`) とは分ける。ディスクが足りないので、根は外付けか、`build` の既定の片付けで持たない
- まずは Actions なしで、手で `hwnpu-dev build` を回す。Actions にするのは HF に上げる段 (cache-sharing-design.md 9 節) から

## 7. 機械と環境 (決めること)

| 項目 | 候補 | 案 |
|---|---|---|
| コンテナのエンジン | OrbStack、colima (lima、`--vm-type vz --vz-rosetta`)、Docker Desktop、Apple の `container` | colima か OrbStack。どちらも Rosetta の amd64 が速い。Docker Desktop は会社の利用で有償。Apple の `container` はコンテナごとに VM で、メモリを 1 本ずつ決められるのは合うが、docker 互換の引数かを確かめる |
| VM のメモリ | M1 16 GB のうち | 12 GB。Mac の側 (Go、Finder) に 4 GB。**FLUX.3 (16.8 GB) は入らない** (下) |
| 書き出しのメモリ | | 大きい画像のモデルは道具の側で減らす: safetensors を `safe_open` で 1 つずつ読み、バンクを書いたら離す。FLUX.3 は 11.1 GB の DiT を丸ごと持っているはず。減らせるまでは、FLUX.3 と Chroma は aix1 か 32 GB 以上の Mac で |
| ディスク | 空き 51 GB (2026-11-19) | 4.3 の 2 の検査、4.4 の詰めながら送る形。イメージ (CPU 版 torch 込みで 6〜8 GB の見込み) と VM のディスクも空きから引かれる |
| 道具の版 | mlir-aie 1.4.4、Peano は aix1 と同じもの | aix1 の `pip show mlir_aie llvm_aie` と IRON のコミットを写して固定し、wheel を NAS に |

メモリの目安 (今分かっている最大 RSS): FLUX.3 klein の書き出し 16.8 GB (8 つ)、16 GB (2 つ)。チャットのモデル、FLUX.2 klein はまだ測っていない。最初の PoC で `docker stats` と `/usr/bin/time -v` で測り、models.yaml に書き出しのメモリの欄 (`build_memory`) を足して、`build` の計画が VM の大きさと比べて止める。

## 8. 進める順 (それぞれ 1 つの PR)

1. **PoC (コードなし)**: エンジンを入れ、`tools/container/iron/Dockerfile` を作り、`HWNPU_IRON_ENV=tools/container/iron-python hwnpu-dev cache export Qwen3.5-0.8B` を空の根で。かかった時間 (aix1 と比べた Rosetta の遅さ)、最大メモリ、aix1 の NAS のものと sha256 が同じかを記録する (design.md 16.x)
2. Dockerfile と `iron-python` をリポジトリに。wheel とイメージを NAS に置く手順。INSTALL_APPLE に「Mac で AMD の書き出しを作る」節
3. 鍵に `toolchain` (4.5)
4. `hwnpu-dev build`: 計画、空きの検査、順に回す、片付け (4.1、4.3)。最初はチャットと画像 (FLUX.2 klein、FLUX.3 klein) の `amd-xdna2` と `coreai`
5. 詰めながら送る (4.4)
6. 書き出しのメモリを減らす (FLUX.3 から) と `build_memory`
7. 手で作っている AMD の書き出しを models.yaml の作り方に (Anima、Chroma、深さ、顔、ラベル)
8. Actions (6 節) と HF (cache-sharing-design.md 9 節)

## 9. 未決

- 1 の結果で sha256 が合わなかったときに、どこまで道具を決定的にするか (合わないままなら、Mac で作ったものは全部 5 節の 2 の扱い)
- Rosetta の遅さが大きければ arm64 のイメージ (mlir-aie と llvm-aie をソースから) にするか
- aix1 を残すか (照合だけの機械に下げるか)
- 書き出しのメモリを減らせないモデルを、どの機械で作るか
