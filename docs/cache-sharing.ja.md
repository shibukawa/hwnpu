# モデルのキャッシュと、その共有

[English](cache-sharing.md)

hwnpu は、モデルを手で変換させません。初めて使うときに、ダウンロードして、NPU 向けに変換して、**キャッシュ**に置きます。この文書は、そのキャッシュを次の順に説明します。

1. [1 台の機械でのキャッシュ](#1-1-台の機械でのキャッシュ) (全員向け)
2. [家庭や社内で共有する](#2-nas-や-s3-でキャッシュを共有する): NAS、Web サーバー、S3
3. [ユーザーリファレンス](#3-ユーザーリファレンス): 変数、URL、コマンド、フォルダの並び
4. [NPU の種類ごとの実装](#4-npu-の種類ごとに何を共有するか)と、[内部のしくみ](#5-内部のしくみ)

モデルを動かすだけなら 1 だけ読めば足ります。複数の機械が同じダウンロードと変換を待たないようにしたいときは 2 へ。

> 設計の経緯と測定、決めたことの全部は [cache-sharing-design.md](cache-sharing-design.md) にあります。ほかの文書やコードの注釈にある「7 節」などの参照は、そちらを指します。

---

## 1. 1 台の機械でのキャッシュ

### 1.1 初めて使うとき

```
hwnpu-chat -m LFM3-1.7B
   │
   ├─ チェックポイント (元の重み)   → Hugging Face から取る
   ├─ NPU 向けの変換               → この機械で作る (数分)
   └─ 答える
```

ダウンロードしたファイルも、変換の結果も、キャッシュに残ります。2 回目は数秒で始まり、同じモデルを使うほかの hwnpu のコマンドも同じファイルを使います。

### 1.2 置き場所

hwnpu が書くものは、1 つのフォルダに収まります。

| OS | キャッシュの根 |
|---|---|
| macOS | `~/Library/Caches/hwnpu` (iCloud には同期されません) |
| Linux | `$XDG_CACHE_HOME/hwnpu` (`~/.cache/hwnpu`) |
| Windows | `%LOCALAPPDATA%\hwnpu` |

外付けのディスクなどへ移すときは `HWNPU_CACHE=/mnt/big/hwnpu`。自分の設定 (`models.yaml`、`licenses.json`、`run.json`) はキャッシュの外にあるので、キャッシュを消しても失いません。

### 1.3 中身

| 種類 | 何か | いつできるか |
|---|---|---|
| 元のモデル | Hugging Face と同じ並びのファイル (`original/OWNER/NAME/`) | 初めてのダウンロード |
| 変換したもの | NPU 向けのプログラムと書き出し (`models/`、`entries/`、`converted/`) | そのチップで初めて使うとき |
| 機械に縛られるもの | この機械向けにコンパイルしたもの (Core ML の `.mlmodelc`、Level Zero の blob、配置の判定) | 初めて使うとき |
| IRON のカーネル | AMD の JIT のキャッシュ (`iron/`) | 書き出しの間 |
| ほか | プロンプト、履歴、ロック、作業中のもの | いつも |

キャッシュは増え続けるわけではありません。ダウンロードの前にディスクの空きを確かめ (既定は 10 GB 残す)、上限を決めてあれば、最近使っていないモデルから消します。

### 1.4 手入れ: `hwnpu cache`

```
hwnpu cache                          根、ディスクの空き、上限、種類ごとの量
hwnpu cache ls [-s]                  単位ごとの大きさと最後に使った時刻 (-s: 大きい順)
hwnpu cache prune --max-size 200GB   最近使っていないものから消して収める
hwnpu cache clear Qwen3-1.7B         モデルを 1 つ消す (名前、OWNER/NAME、glob)
```

`prune` と `clear` は `-n` で、消さずに何が消えるかだけ見せます。最近 1 時間に使ったものと、アップロード待ちのものは残ります。消したものは、次に使うときにまた作ります。

上限の設定は 2 つです (環境変数か `.env`。3.2 節)。

```
HWNPU_CACHE_MIN_FREE=10GB     これより空きが減るダウンロードは始めない (0 でなし、5% のような割合も)
HWNPU_CACHE_MAX_SIZE=200GB    これを超えないよう、古いものから消す
```

### 1.5 モデルが使うメモリを選ぶ

モデルには、メモリの小さい機械向けの形があるものがあります。大きな機械で、ほかのプログラムと一緒に動かしたいときは、小さい機械として扱うよう指定します。

```
hwnpu memory                      この機械のメモリと、選んだもの
hwnpu memory LFM3-1.7B 8          どのコマンドでも 8 GB の形で動かす
hwnpu memory LFM3-1.7B --forget   この機械のメモリに戻す
```

選んだものは設定のフォルダ (`~/.config/hwnpu/run.json`) に残り、キャッシュにはないので `prune` で消えません。形を変えても前の形のプログラムはキャッシュに残ります。すぐ消すなら `hwnpu cache clear NAME`。

---

## 2. NAS や S3 でキャッシュを共有する

### 2.1 考え方

共有しないと、機械ごとに同じチェックポイントを落とし、同じ変換をやり直します。遅いのは変換です (AMD の NPU では初回の書き出しに数分、モデルによっては数時間)。共有すると、**最初の 1 台が作って共有の場所に置き、ほかの機械は結果を落とすだけ**になります。

そのために **取得元** (source) を並べます。hwnpu は Hugging Face に行く前に、これを順に探します。

```
1. この機械のキャッシュ             いつも最初。いつも書く
2. HWNPU_MODEL_SOURCE、_1、_2 ...   共有の場所。番号の順
3. アプリと一緒に入れたモデル        読むだけ (モデルを同梱するアプリ向け)
4. Hugging Face                     いちばん最後。HWNPU_MODEL_HUGGING_FACE=false で使わない
   └─ どこにもなければ、この機械で作る
```

`?writeback=true` を付けた取得元には**書き戻し**もします。この機械が落としたものと変換したものをそこへ置くので、次の機械が見つけられます。付けなければ読むだけです。つながらない取得元 (NAS が眠っている、VPN が切れた) は、そのプロセスの間は飛ばし、標準エラーに 1 行出すだけで、次の取得元を試して続けます。

モデルを動かすだけの機械には、チェックポイントも変換器も要りません。取得元から変換済みを落として答えます。AMD の NPU なら、その機械に IRON、Peano、Python も要りません。

### 2.2 取得元の書き方

取得元は環境変数の URL です。シェル、サービス、コンテナ、`.env` のどれでも同じ書き方で効きます。

**フォルダ** (NFS や SMB でマウントした NAS、外付けのディスク):

```
HWNPU_MODEL_SOURCE=file:///mnt/nas/hwnpu?writeback=true
```

Windows は `file:///C:/models` か `file://server/share`。

**WebDAV** (Synology、UGREEN、Nextcloud、`dav_methods` を入れた nginx):

```
HWNPU_MODEL_SOURCE=https://USER:PASSWORD@nas.local:5006/hwnpu?writeback=true
```

URL に利用者とパスワードがなければ、`~/.netrc`、次に `HWNPU_WEBDAV_USER` と `HWNPU_WEBDAV_PASSWORD` を見ます。NAS の証明書は自己署名のことが多いので、その証明書だけを信じるには `?tls_sha256=` を付けます (値は `openssl s_client -connect HOST:PORT </dev/null | openssl x509 -noout -fingerprint -sha256`)。信頼できる LAN で手早く試すだけなら `?insecure=true` で検証を省けます。

**HTTP** (同じフォルダを配る静的な Web サーバー。nginx など): 読むだけです。

```
HWNPU_MODEL_SOURCE=http://models.corp.example/hwnpu
```

**S3** (AWS、Cloudflare R2、RustFS などの S3 互換):

```
HWNPU_MODEL_SOURCE=s3://my-bucket/hwnpu?region=ap-northeast-1&writeback=true
HWNPU_MODEL_SOURCE_1=s3://models/hwnpu?endpoint=http://rustfs.lab:9000&path_style=true&profile=rustfs
```

鍵は AWS の環境変数 (`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`) か、`~/.aws` のプロファイル (`profile=NAME`。静的な鍵だけ)。SSO の人は先に環境変数へ出します: `eval "$(aws configure export-credentials --profile X --format env)"`。公開バケットは `no_sign=true`、Cloudflare R2 は `region=auto`。

効いているものは次で確かめます。

```
hwnpu-dev sources      根、取得元、書き戻すか、Hugging Face を読むか
hwnpu-dev sources ls           取得元の中身
```

### 2.3 設定を置く場所

変数は `.env` に書けます。hwnpu は起動したときに `.env.local` と `.env` を読み、前にあるものが勝ちます。

1. プロセスの環境変数
2. カレントディレクトリ
3. ユーザーのフォルダ: `~/.config/hwnpu/` (Windows は `%LOCALAPPDATA%\hwnpu\`)
4. 機械のフォルダ (管理者が置く): `/etc/hwnpu/` (macOS は `/Library/Application Support/hwnpu/`、Windows は `%ProgramData%\hwnpu\`)

読むのは `HWNPU_`、`HF_`、`AWS_` で始まる名前だけなので、ほかのプロジェクトの `.env` を拾っても困りません。パスワードや鍵は、git に入れない慣例の `.env.local` に。ログと `hwnpu-dev sources` は、URL の利用者とパスワードを出しません。

### 2.4 3 つの組み方

**家庭: NAS を 1 台。** NAS をマウントする (または WebDAV を使う) 設定を、全部の機械に `?writeback=true` 付きで同じに書きます。最初にダウンロードや変換をした機械が NAS を埋め、ほかは読むだけ。Mac と Ryzen AI のミニ PC は同じ NAS を共有できますが、変換した形はそれぞれの NPU 向けのものを置きます (4 節)。

**社内: 書く人 1 つ、読む人たくさん。** 1 台の機械か CI のジョブ (ビルド係) にだけ、`?writeback=true` と書き込みの権限を渡します。ほかの人には同じ取得元を読むだけで渡します (`http://` の URL で足りる。S3 なら読むだけの鍵)。社内のネットワークから外に出したくなければ `HWNPU_MODEL_HUGGING_FACE=false`。変数は `/etc/hwnpu/.env` に置けば、各人は何も設定しません。読む人は共有を壊せず、書ける人は元から信頼している人です (書き出したプログラムは NPU で動くので)。

**何段か重ねる。** 取得元は順に探し、最初に答えたものが勝ちます。後ろの取得元で見つかったものは、手前の書き戻す取得元にも置かれます。部屋の中の速い NAS を、遠い S3 の手前に置けば、2 段のキャッシュになります。

```
HWNPU_MODEL_SOURCE=file:///mnt/nas/hwnpu?writeback=true        # 速い。部屋の中
HWNPU_MODEL_SOURCE_1=s3://corp-models/hwnpu?writeback=true     # 全拠点で共有
```

**ネットに出ない。** `HWNPU_MODEL_HUGGING_FACE=false` なら Hugging Face に一切つながず、どの取得元にもないモデルはダウンロードでなくエラーになります。先に、つながる機械で取得元を埋めておきます。

### 2.5 共有されたモデルを機械はどう使うか

- **チェックポイント** (元の重み) は、取得元から先に、Hugging Face は最後に取ります。ファイルは記録された SHA-256 と照らし、合わなければ (大きなダウンロードでビットが反転したことがありました) その取得元を飛ばして次へ行きます。
- **変換した結果**は、何も落とさなくても計算できる鍵で探します。当たった機械は、チェックポイントを落としません。
- 機械に縛られるプログラムは共有されず、機械ごとに 1 回作ります (4 節)。
- gated のモデルとライセンスの確認があるモデルも共有の場所に置けます。ライセンスの確認は、動かす機械ごとに今までどおり行います。

### 2.6 共有の取得元を整える

取得元はサーバーの仕組みを持たないフォルダの木なので、コマンドで整えます。

```
hwnpu-dev sources update [MODEL...]   モデルを Hugging Face の新しいコミットに移す (変わったファイルだけ落とす)
hwnpu-dev sources verify DIR --sums   変換結果を entry.json と照らす
hwnpu-dev sources gc 1 --older-than 90 --apply   取得元 1 の、90 日より前に作った変換を消す
                                    (--model、--kind、--key も。--apply がなければ見せるだけ)
hwnpu cache clean                   書き戻しが切れて残したものを消す: 書きかけのフォルダ、一時ファイル、
                                    S3 の 1 時間を過ぎた分割アップロード
```

モデルのファイルは、`update` するまで `hwnpu-lock.json` のコミットに留まるので、チーム全体が同じバイトで動き、うっかり新しい版に移ることはありません。S3 は、バケットのライフサイクルの規則に「未完了の分割アップロードを数日で中止」を足しておくと安心です。

書いている途中で機械が落ちても、読む側には壊れたものが見えません。変換結果は `entry.json` があって初めて見え、元のファイルは lock が名指して初めて使われます。

---

## 3. ユーザーリファレンス

### 3.1 環境変数

| 変数 | 働き |
|---|---|
| `HWNPU_MODEL_SOURCE`、`_1`、`_2`、… | 取得元の URL。番号の順に探す (3.3) |
| `HWNPU_MODEL_HUGGING_FACE` | `false` で Hugging Face につながない。既定は `true` |
| `HWNPU_CACHE` | この機械のキャッシュの根 |
| `HWNPU_CACHE_MIN_FREE` | 残す空き。既定 `10GB`、`0` でなし、`5%` |
| `HWNPU_CACHE_MAX_SIZE` | キャッシュの上限。超えたら古いものから消す。既定はなし |
| `HWNPU_WEBDAV_USER`、`HWNPU_WEBDAV_PASSWORD` | URL にも `~/.netrc` にもないときの WebDAV の利用者 |
| `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_SESSION_TOKEN`、`AWS_PROFILE` | S3 の鍵 |
| `HF_TOKEN` | gated の Hugging Face のリポジトリ (CI 向け。普段はサインインを勧める) |
| `HWNPU_MEMORY` | その機械を何 GB として扱うか |
| `HWNPU_COREAI_PREBUILT` | `off` で Core AI のコンパイル済みの部品を使わない (4.1) |
| `HWNPU_MODELS`、`HWNPU_ACCEPT_LICENSE`、`HWNPU_BACKEND`、`HWNPU_DEVICE` | 自分のモデルの定義、ライセンスの確認、使う NPU |

### 3.2 `.env` ファイル

1 行に `NAME=VALUE`。前に `export ` を付けてもよく、`#` から後ろは注釈、`'...'` はそのまま、`"..."` は `\n` と `\"` が使えます。優先順位は 2.3 節。テスト (`go test`) では読みません。

### 3.3 取得元の URL

| URL | 種類 | 書き戻し |
|---|---|---|
| `file:///path` | フォルダ | 普通のファイルの操作 |
| `http(s)://[USER:PASSWORD@]host[:port]/path` | WebDAV。書き戻さないなら読むだけの HTTP でもよい | PUT、MOVE、MKCOL |
| `s3://BUCKET/PREFIX` | S3 | PUT と分割アップロード (64 MiB ずつ) |

URL の後ろに足せる設定:

| 設定 | 対象 | 意味 |
|---|---|---|
| `writeback=true` | 全部 | この機械が落としたもの、変換したもの、後ろの取得元から読んだものを置く |
| `tls_sha256=HEX` | https | この証明書だけを信じる (自己署名の NAS) |
| `insecure=true` | https | 証明書とホスト名を見ない。信頼できる LAN だけ |
| `endpoint=URL` | s3 | S3 互換のサーバー |
| `region=R` | s3 | リージョン (R2 は `auto`) |
| `profile=P` | s3 | `~/.aws` のプロファイル (静的な鍵) |
| `path_style=true` | s3 | バケットをホスト名でなくパスに。AWS 以外は既定でこちら |
| `no_sign=true` | s3 | 公開バケット。署名しない |

### 3.4 コマンド

`hwnpu cache` (利用者向け):

| コマンド | 働き |
|---|---|
| `hwnpu cache` | 根、空き、上限、種類ごとの量、取得元 |
| `ls [-k KIND] [-s] [PATTERN]` | 単位ごとの大きさと最後に使った時刻 |
| `prune [--max-size S] [--older-than 30d]` | 最近使っていないものから消す |
| `clear PATTERN [--kind K] [--all]` | 名前の単位を消す (端末では確かめる。`-y`) |
| `clean [-n] [-y]` | 書き戻す取得元から、切れた書き戻しの残りを消す |
| `hwnpu memory [MODEL [GB\|--forget]]` | モデルを何 GB の形で動かすか |

`hwnpu-dev models` (変換を作る人向け。`MODEL/TARGET` の TARGET は `amd-xdna2`、`coreai`、`coreai-ARCH`、`coreai-all`):

| コマンド | すること |
|---|---|
| `fetch MODEL[/TARGET]...` | チェックポイントと、取得元にある変換を、何も変換せずに手元へ。どこにもないものは一覧に出す |
| `package MODEL[/TARGET]...` | 変換をこの機械で作り (どの OS でも。`coreai-ARCH` は macOS の `coreai-build` でコンパイル)、取得元になければ書き戻す取得元へ置く |
| `push MODEL...` | ここで手で作った AMD の書き出しを、書き戻す取得元へ |
| `hf plan\|push\|check\|sizes` | 取得元の変換を Hugging Face に上げる ([prebuilt](prebuilt.md)) |

`hwnpu-dev sources` (共有の取得元を管理する人向け):

| コマンド | すること |
|---|---|
| `list` (コマンドなしでも) | 効いている取得元 |
| `ls [N] [--original]` | 取得元 N の中身 (変換結果、または各リポジトリの lock) |
| `import DIR...` | 手で置いたフォルダ (`~/models`) を取得元へ: Hugging Face と同じと確かめたチェックポイントと、書き出し |
| `update [MODEL...]` | リポジトリを Hugging Face の新しいコミットへ |
| `verify [--sums]` | `entry.json` と照らす |
| `gc N [--model M] [--kind K] [--key K] [--older-than 日数] [--target T] [--apply]` | 取得元 N の変換結果を消す (番号は `list` に出るもの。0 は手元のキャッシュ) |

### 3.5 取得元の中の並び

```
original/OWNER/NAME/config.json …        モデルのファイル。Hugging Face と同じ並び
original/OWNER/NAME/hwnpu-lock.json      コミットと、ファイルごとの SHA-256 と大きさ
converted/MODEL/TARGET/EXPORT/VERSION/…  変換結果。entry.json を最後に書く
```

`TARGET` は NPU の種類 (`amd-xdna2`、`coreai`。表のように対象によらないものは `any`)。`VERSION` は、定義に版のあるモデルは `vN`、開発中のモデルは `vN-HASH`、それ以外は `v0-HASH`。このフォルダは、どの Web サーバーでもそのまま配れ、ファイルマネージャーでも見られます。

### 3.6 自分のプログラムから使う

```go
s, _ := modelstore.Open(modelstore.Options{}) // 根と取得元は環境変数から
p, _ := s.File(ctx, "LiquidAI/LFM3-1.7B", "config.json")
w, _ := s.Weights(ctx, "LiquidAI/LFM3-1.7B")
```

[USAGE_LIBRARY.ja.md](../USAGE_LIBRARY.ja.md) を見てください。hwnpu のどのコマンドも、裏で同じ探し方をしています。

---

## 4. NPU の種類ごとに何を共有するか

何を共有できるかは、その NPU の道具が何を作るかで決まります。チェックポイント (元の重み) はどのチップでも共有されます。変換した結果は違います。

| | Apple (Neural Engine) | AMD (XDNA 2) | Intel (NPU) |
|---|---|---|---|
| 変換した形 | Core AI の `.aimodel` (と Core ML の `.mlpackage`) | IRON の書き出し: `xclbin`、ELF の命令、重みのバンク | OpenVINO の IR |
| 取得元で共有 | Core AI の書き出しと表: する。Core ML のプログラム: まだ | チャットのモデル、初回に作る画像のモデルの書き出し、表: する | チェックポイントだけ。プログラムはまだ |
| 機械に縛られる | Neural Engine のコンパイル (Mac ごと) | なし。IRON の JIT のキャッシュは手元だけ | Level Zero の blob (ドライバとデバイス) |
| 作るのに要る道具 | コンパイル済みの部品を作るなら Xcode 27 と Metal Toolchain | IRON、mlir-aie、Peano、Python | OpenVINO |
| 読むだけの機械に要るもの | hwnpu だけ | hwnpu だけ (IRON は要らない) | チェックポイント (プログラムを共有するまで) |

### 4.1 Apple

- **Core AI** (`target coreai`): `.aimodel` は機械によらないバイトコードで、どの Mac も同じです。最初に変換した Mac が取得元に置き、ほかの Mac は落とすだけで、チェックポイントも要りません。コマンドが行を引く表 (`tables.safetensors`) も同じに共有します。
- **初回のコンパイルは Mac ごと**です。Core AI は部品ごとに初めて読むとき、Neural Engine 向けにコンパイルします (部品ごとに数秒から数分)。その結果は Core AI 自身のキャッシュにあり、OS のビルドに縛られて共有できません。
- **コンパイル済みの部品**がそれを縮めます。Xcode 27 のある開発機で `hwnpu-dev models package EXPORT/coreai-all` を走らせると、チップの世代ごと (`h13g` = M1、`h14g` = M2、`h15g` = M3 ...) に部品をコンパイルし、重み (大きさの約 99 %。`.aimodel` から組み立て直す) を外して、書き出しの隣に置きます。1 世代で書き出しの 1〜2 % です。Mac は自分のチップのものを取り、数十分のコンパイルが 1 秒から 1 分で済みます。効くのは Neural Engine に乗る部品のあるモデルだけなので、モデルを選んで置きます。使わないなら `HWNPU_COREAI_PREBUILT=off`。ディスクが足りなくなったときは、数秒で作り直せるので、この機械のコンパイル済みの部品の写しから先に消します。
- **Core ML** の Neural Engine 向けのプログラムは、今も Mac ごとに作ります。コンパイル済みの `.mlmodelc` は OS の版に、e5 のバンドルは機械に縛られます。

### 4.2 AMD XDNA 2 (Linux)

- モデルは **IRON の書き出し**に変換します。`manifest.json`、NPU のプログラム (`xclbin`、命令の ELF)、重みのフォルダで、チャットのモデルで 0.5〜2 GB ほどです。手元では `entries/amd-xdna2/`、取得元では `converted/MODEL/amd-xdna2/EXPORT/vN/` に置かれます。
- 取得元では書き出しを **1 つに詰めたファイル** `files.tar.zst` (tar と zstd) と、中のファイルの大きさと SHA-256 を並べた `entry.json` だけにします。落とし (切れたら続きから)、展開しながらファイルごとに照らし、手元には展開したものだけを残します。チャットの 13 モデルで 65.7 GB が 55.2 GB (16% 減) になり、WebDAV でファイルごとの送りもなくなります。前の hwnpu がばらで置いたものも、今までどおりファイルごとに取れます。版を上げても中のファイルが前の版と同じ (`entry.json` の digest が同じ) ときは、詰め直して送らず、取得元の中で前の版の `files.tar.zst` を写します (WebDAV の COPY、フォルダはハードリンク。S3 は送ります)。
- 書き出しの鍵は、モデル、チェックポイントのコミット、定義の sha256、書き出しの設定 (model-recipes.yaml のレシピの args)、同梱した IRON の道具のうちその書き出しが読むものの hash の指紋です。読むものは `iron_export_lm.py` と共通のモジュール、モデルのアーキのモジュール (`tools/amd/lmdeps.go` の表から import を辿ったもの)、全部のカーネルです。1 つのモデルの書き出し (Gemma 4 の層ごとの表) や画像のモデルの道具を変えても、ほかのチャットのモデルの書き出しはそのまま当たります。出力の変わる直しは鍵 (か版) を変えるので、古い書き出しに当たりません。
- IRON のない機械 (社内ではこれがふつう) は、書き出しを落とし、数秒の「準備」だけで NPU で動きます。
- 初回に自分のプログラムを作る画像のモデル (`detect`、`pose`、`segment`、FLUX.2 klein) も同じ共有です。取得元にあればそれを使い、なければ IRON で作って書き戻します。
- **手で作った**プログラム (一部の画像や音のモデル、深さ、顔、ラベル、AMD のレシピのないモデル) は、`entries/amd-xdna2/` か `~/models` にあります。一度 `hwnpu-dev sources import ~/models` で取得元に置けば、要る機械が名前で取り、手元に写しを残します。手で作ったものを初めて使った機械が、自動で書き戻しもします。
- IRON のコンパイルのキャッシュ (`iron/`。開発機では数十 GB) は共有しません。書き出す道具の作業の跡です。

### 4.3 Intel

- チェックポイントは共有します。OpenVINO のプログラムは機械ごとに変換とコンパイルをしていて、共有できるようにする計画です (変換した IR をエントリに、Level Zero の blob を機械に縛られるエントリに)。
- 鍵の仕組みに場所は用意してあります。対象 (`intel-npu3`、`intel-npu4`) がエントリのパスに入ります。

### 4.4 新しいチップを足すとき

エントリに何を入れるか、どれが機械に縛られるかは、チップのバックエンドが決めます。共有するには、変換した結果を新しいフォルダに書き、最後にファイルの大きさと SHA-256 と鍵のレシピを書いた `entry.json` を置き、`target` の名前を世代の階層から選び (広い名前 `amd`、世代 `amd-xdna2`)、`hwnpu-dev models push` と読む側がファイルを分かるよう種類 (kind) を登録します。探す、写す、照らす、書き戻すの共通の部分は変えなくて済みます。

---

## 5. 内部のしくみ

### 5.1 層と探す道すじ

どの層も 1 つの*ストア*です。この機械のキャッシュ、各取得元、同梱のモデル、Hugging Face の順で、中は同じ木 (3.5) なので、ファイルやエントリを読むコードはどの層かを気にしません。

```
chat.Fetch / modelstore.File
  → 層を順に: lock が名指すファイルがあるか (HEAD、PROPFIND の深さ 0、stat)
     あった → この機械の original/OWNER/NAME/FILE に落とす (4 本ずつ)
            → SHA-256 を lock と照らし、違えばこの層を飛ばす
            → 手前に書き戻す層があれば、そこへ置く仕事をキューに入れる
  → どこにもない → lock のコミットで Hugging Face から (resolve/<コミット>/)
            → 落としながら、書き戻す取得元へ tee する
```

コミットはリポジトリごとに 1 回決めます。定義の固定、なければ最初の取得元の lock、なければこの機械の lock、なければ Hugging Face の `main`。ファイルはどの層から来ても、そのコミットのものです。だから 2 台が同じモデルの違うバイトで動くことはなく、新しいコミットへ移るのは `hwnpu-dev sources update` のときだけです。

### 5.2 エントリと鍵

変換が作るものは全部**エントリ**です。ファイルの木と `entry.json` で:

```json
{ "key": "<指紋>", "kind": "iron-lm", "scope": "portable",
  "model": "LFM3-1.7B", "target": "amd-xdna2", "version": 2,
  "recipe": {...}, "files": [{"path": "decode.elf", "size": 123, "sha256": "..."}],
  "made_by": {...}, "checked": {...} }
```

- **鍵**は何も落とす前に計算できます。モデルの定義の固定値 (コミットとファイルの SHA-256)、設定、書き出しとその道具の版から作るので、チェックポイントなしで変換結果を探せます。
- **版**: `format_version` (会社ごと) はローダーの読む形式が変わるとき、モデルの `version` (モデル × 対象、`models.yaml` の `targets`) は出力が変わるときに、手で上げます。取得元のパスには `vN` が入り、開発中のモデルは鍵の hash を使う `vN-HASH` なので、直すたびに別のフォルダになります。
- **scope** はどこまで持ち出せるか: `portable` (プログラム、書き出し、表。どの機械でも)、`host` (OS の版、ドライバ、チップに縛られる。鍵に機械の指紋が要る)、`local` (プロンプトのキャッシュ、履歴、ライセンスの答え。共有しない)。
- エントリの中のパスはどれも相対で、写したフォルダはどこでも動きます。

### 5.3 書き戻し

書いている途中のものを、読む側に見せません。

- ファイルは同じフォルダの一時の名前 (`.put-…`) に書いて名前を変えます (S3 は PUT が終わるまで見えない)。変換結果は最後に `entry.json` を書いて初めて見え、元のファイルは `hwnpu-lock.json` が名指して初めて使われます。
- **tee**: Hugging Face から落とすバイトを、落としながら書き戻す取得元へ送り、大きさと SHA-256 が合ったときだけ確定します。取得元が 64 MiB 遅れたとき、`.part` から続けたときは tee を外れ、落とし終えてからキューで置きます。遅い取得元がダウンロードを遅くしたり止めたりすることはありません。
- **変換しながら置く**: 手元の変換の間、書き出しのフォルダを見て、3 秒触られていないファイルから 4 本ずつ置きます。変換が終わるころには大半が置かれていて、最後は後から変わったファイルと `entry.json` だけです。
- アップロードは **キュー** (4 本) に入れて進めます。コマンドは終わる前に済むのを待ち、途中で止められたら `work/queue` に残して次の起動で続けます。失敗は `work/queue.log` に残し、コマンドは失敗にしません。
- 書いている途中のフォルダには `.writing-ID` の印 (機械、プロセス、最後に書いた時刻) を置き、5 分ごとに書き直します。`hwnpu cache clean` は、印かファイルが最近 1 時間に変わったフォルダを残すので、ほかの機械がアップロードしているものを消しません。
- lock は読んで足して書きます。2 台が同時に同じリポジトリへ足すと片方が消えることがありますが、消えたものは次に読む機械が足し直します。

### 5.4 通信

- **フォルダ**: 普通のファイルの操作。
- **WebDAV** (`internal/store/webdav.go`、標準ライブラリだけ): 続きは `Range` 付きの `GET`、書きは一時の名前へ `PUT` してから `MOVE`、親は `MKCOL`、一覧は深さ 1 の `PROPFIND`。`HEAD` で `Content-Length` を返さないサーバーや、409 のところで 405 を返すサーバーにも合わせてあります。時間切れと 5xx は 3 回まで再試行し、つながらないサーバーはそのプロセスの間は止めて 1 行出します。
- **S3**: tinygodriver の `storage/s3` (AWS SDK は使わない)。続きの `GetRange`、失敗したら中止する 64 MiB の分割アップロード、`clean` のための `ListMultipartUploads`、5xx と `SlowDown` の再試行。RustFS と Cloudflare R2 で確かめました。
- **HTTP** は、書く操作をやめた WebDAV のクライアントです。

### 5.5 手元のキャッシュ

`internal/cache` が根と規則を持ちます。

- **空き**: ダウンロードのたびに空きの下限を確かめ (`ErrNoRoom`)、空きが足りないか上限を超えるときは、まず IRON のグラフごとの組み上げ (`iron/KEY`。カーネルの `.o` とコアの ELF が残っていれば数秒で作り直せます) を消し、上限があれば、単位 (リポジトリ、変換結果、IRON のカーネル) を、最近使っていない順に消します。最後に使った時刻は、フォルダと中のファイルの新しい方で、読んだときに更新します (atime は当てにしません)。最近 1 時間に使ったもの、このプロセスが使ったもの、アップロード待ちのものは消しません。消すときは先にエントリの印を消すので、読む側に半分消えたものが見えません。
- **ロック**: ダウンロード、変換、書き出しごとに (`flock`/`LockFileEx`)。2 つのプロセスが同じ `.part` に書きません。続きのダウンロードは `If-Range` を付けます。
- **ハードリンクと reflink**: モデルの間、手元と同梱のモデルの間で、同じファイルをコピーせずに共有します。
- 機械に縛られるものの鍵にはチップが入るので、M1 の Core ML の配置の判定を M3 で使うことはありません。

### 5.6 まだないもの

- 組織 `hwnpu` の Hugging Face に、変換済みのモデルを CI で作って照合して置き、手元で変換する前の最後の層として読む仕組み。今は変換結果の共有は、*自分の*取得元を通してです。
- Core ML と OpenVINO のプログラムのエントリ化と、共有の取得元での `host` のエントリ。
- 手元のキャッシュの、全種類の書き出しを 1 つの `converted/` にまとめること (一部は今も `models/` と `entries/`)。
- AMD と Intel の NPU の世代を、対象の名前で見分けること。

それぞれの詳細 (CI と指紋の lock の計画も) は、設計の記録 [cache-sharing-design.md](cache-sharing-design.md) にあります。
