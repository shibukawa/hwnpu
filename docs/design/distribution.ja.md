# モデルの配布とキャッシュ

[English](distribution.md)

hwnpu がモデルを探し、落とし、変換し、共有し、消す仕組み。この文書は設計のまとめと決めたことで、細部は次にある:

- [../cache-sharing.md](../cache-sharing.md): 利用者の手引きと参照 (変数、URL、コマンド、並び、NPU ごとの共有、中の作り)
- [../cache-sharing-design.md](../cache-sharing-design.md): 設計の記録
- [../model-table.md](../model-table.md): models.yaml の設計。各フィールドは [../../models.yaml](../../models.yaml) の頭の注釈に書いてある
- [../prebuilt.md](../prebuilt.md): 取得元にある変換済みのモデルを Hugging Face (`hwnpu/<モデル>`) に上げ、読み戻す設計 (未実装)

## models.yaml

### ファイルと上書き

- 組み込みの [models.yaml](../../models.yaml) は hwnpu に埋め込む。その上に読むファイルは順に: 機械の定義 (`/etc/hwnpu/models.yaml`、macOS は `/Library/Application Support/hwnpu/models.yaml`、Windows は `%ProgramData%\hwnpu\models.yaml`)、利用者の定義 (`~/.config/hwnpu/models.yaml`、Windows は `%LOCALAPPDATA%\hwnpu\models.yaml`)、`HWNPU_MODELS` (カンマ区切り)、`--models FILE` (hwnpu-chat 以外は `-models`)。読むのは `chat.LoadModels` (`internal/chat/table.go`)
- 後のファイルは書いたフィールドだけを変える。ベンダーの表 (`device`、`memory`、`bytes`) は挙げたベンダーだけを変える。挙げた `variants` は置き換える。`remove: true` はモデルや用途の欄を消す。`from` も `like` もなく、知らない名前を上書きするものは警告して飛ばす
- `like: OTHER` は別のモデルのフィールドを `tag` と `sha256` を除いて引き継ぐ (ファイルが別のリポジトリのものかもしれないため)。だから `like` のモデルは自分の `sha256` を固定しないと、レシピの鍵がなく共有されない (MOSS-SoundEffect-v2.0-lut8 で起きた)

### 選び方

| 指し方 | 選ぶもの |
|---|---|
| `-m NAME` | そのモデル |
| `FAMILY:TAG` (`plamo3translate:2b`) | そのタグ |
| `FAMILY` | この機械のメモリまでで `machine` が最大のタグ。メモリのタグ (`image:8g`、`8G`、`8 GB`、欄にない `12g` も) はそのメモリの機械として選ぶ |
| 用途 (`chat`、`image`、`depth` など。models.yaml の `categories:`) | この機械のメモリの `picks` の欄 (`machine: 8/16/24`)。用途の段はファミリーの段と揃える |
| `aliases:` | NAME が選ばれるところでは OTHER を動かす |
| Hugging Face のリポジトリ | そのリポジトリ (指定にはライセンスが分からない) |

- `HWNPU_MEMORY` は機械をその GB として扱う。`hwnpu memory MODEL GB` はモデルごとの選択を `~/.config/hwnpu/run.json` に残す
- `not_on: [M1, …]`: ファミリーや用途はそのチップではこのモデルを飛ばし次を選ぶ。名前で指せば動く
- `tier`、`prefill`、`format`、`context`、`paged`: フラグがないときの hwnpu-chat のチャットのモデルの動かし方 ([llm-engine.md](llm-engine.md))。`variants: {16: {...}}` は大きな機械でこれらと `memory`、`amd_export` を変える
- `hwnpu-chat list` と `hwnpu-image list [image|depth]` は 1 つの表 (`chat.ListHave`) で、この NPU の DOWNLOAD、MEMORY、RUNS FROM を出す

### NPU ごとに測る値

- `bytes` (準備で落とすファイル、GB = 10⁹)、`memory` (実行の全体を通したピーク、NPU とプロセスの和)、`machine` (ファミリーが勧める機械の最小のメモリ)。測り方は [new-model-checklist.md](new-model-checklist.md)
- `memory` と `bytes` は 1 つの数か、ベンダーの表 (`memory: {amd: 3.5, apple: 4.1}`)。表にないベンダーは最大の値を取る。Apple のチップも書ける (`{amd: 2.2, apple: 3.1, M1: 1.7}`)
- `device: {amd: cpu}`: ベンダーごとの置き場所 (`amd`、`apple`、`intel`。既定は `npu`)。順は `--device`、`HWNPU_DEVICE`、models.yaml (`chat.ChooseDevice`、理由も返す)。理由: AMD の NPU は小さなモデルで負ける (YuNet 320×256、Ryzen AI で CPU 3.96 ms・10 mJ、NPU 30.5 ms・137 mJ) が、Apple と Intel はグラフをまとめて投入するので、あるベンダーの結果をほかに持ち込まない。このベンダーの NPU の形がないモデルは CPU に落とし、その旨を出す。`--device npu` を明示したときだけエラー
- `sha256: {FILE: …}` でファイルを固定する。別のモデルが同じ SHA-256 で固定し、すでに落としたファイルは、落とさずにハードリンクする (MOSS-SoundEffect の文のエンコーダーの Qwen3-1.7B)

### ライセンスとモデルの指定

- 全モデルに `license: {id, url, class, terms, accept}` がある (`TestBuiltinLicenses`)。`class` は `open`、`conditional` (年商の上限、登録)、`non-commercial` (重みが商用不可。作ったものは売れても)。各ライセンスの一次資料は [../model-provenance.md](../model-provenance.md)
- `content: [nsfw]` は頼まなくても NSFW の絵を描くモデル (Anima、Chroma) だけ。頼めば出しうるだけのモデルには付けない
- 指定 (`chat.Policy`、`internal/chat/policy.go`): `HWNPU_MODEL_ALLOWLIST` (`.env` も) と、利用者の定義のファイルの `policy: [...]` (組み込みには書けない)。項目は `+license:X`、`-license:X` (X は id か区分)、`+nsfw`
  - `-` が `+` に勝つ。`+license` が 1 つでもあれば許可リストになり、当たらないもの、ライセンスの分からないものを弾く
  - `nsfw` のモデルは `+nsfw` がなければ弾く。`+nsfw` は許すだけで、生成の振る舞いは変えない
  - 用途は弾かれたモデルを飛ばす。この機械に合うものが残らなければ、同じ種類と機能で許されたモデルのうち、いちばん開いた区分のものを足す。弾いたモデルと理由は選んだ理由の後ろに出す
  - 名前で指したモデル (`-m`、タグ、リポジトリ、API のリクエスト) は差し替えず、理由を添えた `*chat.PolicyError`。リポジトリはライセンスが分からないので、`-license` は通り、`+license` で弾かれる
  - 一覧は弾かれたモデルと理由を最後に出す。全部弾かれた欄は `-`
  - 用途 `image` は 8 GB から FLUX.2 klein 4B。Anima と Chroma は `+nsfw` でも用途では選ばない。名前で指せば `+nsfw` で動く
  - 例 (`TestPolicyPicks`): `-license:conditional` → chat 8 GB は Qwen3-1.7B (LFM3 は年商の上限つき)、chat 16 GB は gemma-4-E2B-it。`-license:lfm-open-1.0` → speech はエラー (代わりがない)。`+license:open` → translate はエラー (PLaMo 3 Translate は条件付き)
- ライセンスの確認 (`chat.AcceptLicense`): `accept: true` のモデル (CircleStone Labs の Anima など) は、名前で初めて使うときに端末で一度だけ聞き、`~/.config/hwnpu/licenses.json` に残す。端末がなければ `--accept-license` か `HWNPU_ACCEPT_LICENSE=ID` (その回だけ) を求めて止まる。答えは手元だけで共有しない。機械ごとに答える

## キャッシュの根

- `cache.Root`: OS のローミングしないキャッシュのフォルダの下の `hwnpu` (macOS `~/Library/Caches/hwnpu`、Linux `$XDG_CACHE_HOME/hwnpu`、Windows `%LOCALAPPDATA%\hwnpu`)。`HWNPU_CACHE` で移す。コードからは `modelstore.Options.Root` か `cache.SetRoot`。ほかの場所には書かない。利用者の設定 (`models.yaml`、`licenses.json`、`run.json`) は設定のフォルダにあり、キャッシュではない
- `go test` は `HWNPU_CACHE` がなければプロセスごとの一時フォルダを根にする。chat の試験は試験ごとに根を分ける (`cacheHere`)
- 並び (全体の表は [../cache-sharing.md](../cache-sharing.md)): `original/OWNER/NAME/` と `hwnpu-lock.json` (取得元と同じ)、`models/NAME/` (変換したプログラム、Core AI の書き出し `NAME.coreai`、表)、`entries/amd-xdna2/NAME.amd-npu2*` (AMD の書き出し)、根の直下の `KEY.mlmodelc` / `.blob` / `.json` (機械に縛られたコンパイルの結果)、`work/` (`.part`、queue、詰める作業。rename と reflink のため同じファイルシステム)、`tools/` (組み込みの IRON の道具)、`iron/` (IRON の JIT のキャッシュ、`NPU_CACHE_HOME` で渡す)、`locks/`、`prompts/`、`history/`
- `~/models` は AMD の書き出しの読むだけの古い層 (`cache.ExportRoots`)。書かない
- 完了の印は相対パス (`cache.WriteDone`、`ReadDone`) なので、写した根もそのまま使える
- ロック (`cache.Lock`、`locks/` のファイルを flock / `LockFileEx` で): ダウンロードはファイルごと、チャットの変換はプログラムごと、IRON の書き出しは書き出しごと。待った側は相手が作ったものを見つける

## 取得元

### 層の順

```
0. この機械のキャッシュ (cache.Root)   いつも書き戻す・その場で使う
1. HWNPU_MODEL_SOURCE, _1, _2, ...    番号の順。エラーの取得元は飛ばす
2. アプリ同梱のモデル                  読むだけ・その場で使う
3. Hugging Face の変換済みのモデル     models.yaml の hub (リポジトリと対象ごとのコミット)。HWNPU_MODEL_HUB=false で読まない
4. Hugging Face                       HWNPU_MODEL_HUGGING_FACE=false で出ない (3 も読まない)
   どこにもない -> この機械で作る
```

- 取得元は URL: `file:///path` (マウントしたフォルダ)、`http(s)://[USER:PASS@]host/path` (WebDAV。読むだけなら GET と HEAD だけなので、どの HTTP サーバーでも)、`s3://BUCKET/PREFIX?endpoint=…`。パラメータ (`writeback`、`tls_sha256`、`insecure`、`region`、`profile`、`path_style`、`no_sign`) は [../cache-sharing.md](../cache-sharing.md)。読むのは `internal/store/source.go`
- `?writeback=true` は、この機械が作ったものと後ろの層から読んだものの両方を置く (gated のモデルも)。後ろで見つかったものは手前の書き戻す取得元にも写すので、S3 の前に NAS を置けば 2 段になる
- `.env` と `.env.local` を、カレントディレクトリ、利用者の設定のフォルダ、機械の設定のフォルダの順に読む (`internal/envfile`。`internal/console` と `internal/cache` が空の import で読むので先に読まれる)。`HWNPU_`、`HF_`、`AWS_` の名前だけを、環境にないときだけ足す。`go test` では読まない。URL のパスワードは表示で伏せる
- 同梱のモデル (`store.SetBundles`、`store.Dir.InPlace`: ハードリンク、できなければシンボリックリンク。lock を信頼する): インストーラが `configdir` の資源フォルダ (`share/hwnpu/models`) に置き、アプリは `modelstore.Options.Bundle` で指す。環境変数にはしない
- どのコマンドも同じ探し方で、公開の `modelstore` でもある ([api.md](api.md))。コマンドに `--dir` はなく、根は `HWNPU_CACHE`

### 取得元の中の並び

```
original/OWNER/NAME/FILE              Hugging Face と同じ
original/OWNER/NAME/hwnpu-lock.json   コミット、ファイルごとの sha256 と大きさ
converted/MODEL/TARGET/EXPORT/VERSION/   変換結果。entry.json を最後に書く
```

- lock の名指さないファイルは読まない。lock なしで手で写したリポジトリ (NAS の gated のチェックポイント) は見つからず、Hugging Face に回る
- コミットはリポジトリごとに 1 回決める: models.yaml の固定 (`from: hf:OWNER/NAME@REV`)、なければ最初の取得元の lock、手元の lock、Hugging Face の `main` の順。どの層から取っても全ファイルがそのコミットのもの。lock を移すのは `hwnpu-dev sources update` だけ (変わったファイルだけ落とし直す)。取得元の lock が移れば手元も移り、sha256 が同じファイルは残す (`store.Record`)
- 速い道: 手元の lock がファイルを名指し、大きさが合い、コミットが決まったものと同じなら、ネットにも取得元にも聞かない
- 元のファイルはどの層から来ても手元の `original/OWNER/NAME/FILE` (`chat.RepoDir`) に置く。`chat.Fetch(ctx, repo, name, log)` と `chat.FetchWeights` がパスを返す。モデルの間で共有する部品 (Chroma の T5-XXL と VAE) は各リポジトリのフォルダにあり、sha256 でハードリンクする
- `TARGET`: `amd-xdna2`、`coreai`、`coreai-ARCH` (prebuilt)、`any` (表、GPTQ の codes)。`VERSION`: models.yaml に `targets.<対象>.version: N` があれば `vN` (道具の hash を除いたレシピが同じなら当たる)、`develop: true` なら `vN-<8 桁>`、版がなければ `v0-<8 桁>`
- どの層にもない版 (`vN`) は、同じ鍵の `v0-HASH` (版を書く前に作った同じバイト) から取り、`vN` として書き戻す (`chat.fetchFrom`)
- フォルダは揃っているか、揃っていないかの 2 つ: 変換結果は `entry.json` があれば揃っている。リポジトリは lock が名指すものだけを持つ。読む側に書きかけは見えない

### 鍵 (何も落とす前に計算する)

| 種類 | 鍵 |
|---|---|
| AMD のチャットの書き出し (`chat.ExportRecipe`、`chat.SharedExport`) | モデル、チェックポイントのコミット、models.yaml の sha256、`amd_export` の引数、読む組み込みの IRON の道具の hash |
| AMD の初回の書き出し (detect、pose、segment、FLUX.2 klein。`chat.PinnedRecipe`) | 読む全ファイルの models.yaml の sha256、道具の名前と hash、Go の側の版 (`detect.ExportVersion` など、`AMDExportVersion`。引数の渡し方を変えたら手で上げる)。固定がなければ鍵がなく共有しない |
| 手で作った AMD の書き出し (`iron-import`、`chat.AMDNamed`) | 名前 (`NAME.amd-npu2`、`NAME-1024.amd-npu2`、`NAME-SIZE.amd-npu2`、`NAME.amd-npu2-q4`) |
| Core AI の書き出し (`coreAIRecipe`、`kind: coreai`) | モデル、リポジトリ、sha256 (なければファイルのコミット)、一緒に読むリポジトリ、書き出しの名前 (版と形式を含む)、`ailower.FormatVersion` |
| 表 (`kind: tables`、`chat.Tables`) | モデルの部分、アーキテクチャ、表の名前、`tablesVersion` |
| GPTQ の codes (`kind: codes`、`converted/MODEL/any/codes-HASH`) | モデルの部分と index.json のハッシュ |
| Core AI の prebuilt (`coreai.ProgramsDigest`) | 各 `.aimodel` の場所と `main.hash` の sha256、アーキテクチャ、書き出しの名前 |

- sha256 もコミットも分からないモデルは共有しない。作るものが変わったら版を手で上げる (`ailower.FormatVersion` を 2 にして Core AI の鍵が全部変わり、全書き出しを作り直した)
- GPTQ の codes は変換にだけ要るが、取得元にも置く (`chat.putCodes`、`fetchCodes`)。なければ鍵が変わったとき誰も書き出しを作り直せない

### 取得元を通るもの

- 元のファイル (チェックポイント)。sha256 を確かめ、違えばその層を飛ばす
- AMD の書き出し: チャットのモデル、画像の初回の書き出し、手で作った書き出し (名前で `entries/amd-xdna2` に落とし、以後は手元のものを使う。`entry.json` のない手で作ったものは初めて使ったときに記述して書き戻す、`chat.ShareMadeByHand`)。AMD の NPU のない機械では取りに行かない
- Core AI の書き出し (`llm.FetchExport`、`llm.ShareExport`): どの Mac にも同じ `.aimodel` 一式 (チップ向けのコンパイルは各 Mac の Core AI のキャッシュ)。チェックポイントを落とす前に探すので、当たればチェックポイントを落とさない。注意を分けたモデルは、注意のプログラムを全部足してから書き戻す (`shareMade`)。depth と MoGe-2 は大きさごとのサブフォルダが単位
- 表と GPTQ の codes。書き出しと表が取得元にあれば、動かすだけの機械はチェックポイントを落とさない (config、tokenizer、索引だけ)
- 共有しないもの: Core ML のプログラム、OpenVINO / Level Zero のプログラム、IRON の JIT のキャッシュ、機械に縛られたコンパイルの結果、ライセンスの答え、プロンプト、履歴

### 書き戻し

- 順: データが先、`entry.json` (か lock の項目) が最後。ファイルは `.put-…` に書いて MOVE (S3 は PUT が終わって初めて見える)
- tee: Hugging Face からのダウンロードを、来るそばからフォルダと WebDAV の取得元に流し、大きさと sha256 が合ったときだけ確定する。64 MiB 遅れた取得元 (`store.TeeBuffer`)、S3、`.part` からの続きは外れて、ダウンロードの後で queue から置く
- 詰めない変換結果は作りながら置く (`store.StartEarly`: 3 秒触られないファイル、`EarlySettle`)。残りは `PutEntry`。詰めるもの (AMD、Core AI) は揃ってから置く
- queue (`store.PutLater`、`work/queue/`、4 本) は終わる前に待ち、切れたら次の起動で続け、失敗は `work/queue.log` に書き、コマンドを失敗させない
- 書いている間のフォルダには `.writing-ID` (JSON: 機械、プロセス、始めた時刻、最後に書いた時刻) を置き、5 分ごとに書き直し、終われば消す (`store.MarkWriting`)。書く道はどれも置く (tee と作りながら置くのも)
- 詰める (`store.Packs`、`entry.json` の隣の `files.tar.zst`): AMD と Core AI の書き出しは tar を zstd で詰めた 1 ファイル。効くのは要求の数 (WebDAV で PUT と MOVE が 1 回)。大きさは Core AI で 1〜5 % (LFM3-1.7B 2.62 → 2.59 GB)、AMD のチャットの 13 モデルで 16 % (65.7 → 55.2 GB)。取るときは続きから落とし、詰めたものの sha256、次に各ファイルを `entry.json` と照らす。手元には展開したものだけを残す。前の hwnpu がばらで置いたものも読める (NAS のものは 2026-11-21 に詰め、詰め直すコマンド `hwnpu-dev cache repack` は外した。今はどの置き方も詰める)

### 通信

- WebDAV (`internal/store/webdav.go`) は標準ライブラリだけ。読むだけの HTTP は書き込みを外した同じクライアント。S3 (`store.S3`) は tinygodriver の `storage/s3` で、AWS SDK も CLI も使わない。鍵は静的なものだけ (SSO は `aws configure export-credentials --format env`)。RustFS 1.0.5 と Cloudflare R2 で確かめた。細部は [../cache-sharing.md](../cache-sharing.md)
- 今も当てはまる落とし穴:
  - NAS の自己署名の証明書 (ホスト名なし) は Go の検証を通らず、取得元が飛ばされる: `tls_sha256=` を使う
  - 応答の本体は全部読んで閉じる。閉じないと Go は要求ごとに新しい接続を開く。mDNS (`.local`) では名前の解決が毎回 5.01 秒だった。`cachedDialer` でホストを 1 回だけ引く。直した後、95 ファイル 1.5 GB と表 0.51 GB の書き戻しが 60 秒、96 MB の取り出しが 45 秒 → 2.3 秒 (UGREEN の NAS、M2)
  - Wi-Fi と UGREEN の NAS の間: 1 本 5 MB/s、4 本で計 9 MB/s
- 往復 (Qwen3.5-0.8B、Core AI、空の `HWNPU_CACHE`、WebDAV の NAS、M2 16 GB): NAS が空なら準備まで 10 分 41 秒、NAS に全部あれば 45 秒 (書き出し 2.2 GB と config・tokenizer・索引 12 MB。チェックポイントは落とさない)。取得元から取った結果は変換した機械のものとバイト単位で同じ

## ダウンロード

- 変換が読むファイルだけを落とし、リポジトリを丸ごとは落とさない。ファミリーで同じファイルは 1 回だけ落として共有する
- 落とす前に `cache.Room` が場所を確かめる (下)
- 何本かを並べる: 4 本 (`store.FetchStreams` は変換結果を大きいものから、`chat.FetchFiles` は画像、音、分割したチャットのチェックポイントの元のファイル)。どのバーも計画の全体の残りを出す (`chat.Expect`)
- 続き: Hugging Face のダウンロードは ETag を `NAME.part.etag` に残し、同じ版のときだけ `If-Range` で続ける。`X-Repo-Commit` (リダイレクトの前の応答も見る) を、取得元に置くときのコミットにする
- Hugging Face の前に取得元を探す: models.yaml の sha256、なければ HEAD の `X-Linked-Etag` (LFS のファイル)、なければコミットの lock
- 変換結果が手元か取得元にあるモデルは小さなファイルだけを見る: たとえば `mtpArch` は分割のチェックポイントなら `model.safetensors.index.json` の `weight_map` を読み、重みを落とさない

### 同意の要るリポジトリ (gated、`internal/chat/hfauth.go`)

- トークンの順: `HF_TOKEN`、huggingface_hub のトークンファイル (`HF_TOKEN_PATH`、`$HF_HOME/token`、`~/.cache/huggingface/token`。読むだけ)、このプロセスでサインインしたもの
- 401 で、API (`/api/models/OWNER/NAME`、トークンなし) が `gated` と言い、端末に人がいれば、OAuth 2.0 のデバイスフロー (RFC 8628) でサインインする: hwnpu の公開 OAuth アプリ (`hfClientID`、秘密鍵なし、`HWNPU_HF_CLIENT_ID` で差し替え)、スコープは `openid` と `gated-repos` だけ (アカウントが同意した公開の同意制リポジトリを読めるだけ。非公開のリポジトリもプロフィールも読めない)。コードを https://hf.co/oauth/device に入れる。SSH 越しでもブラウザのない機械でも使える
- トークンはプロセスのメモリだけ。アクセストークンの寿命は 15 分で、refresh token (これもメモリだけ) で 401 のときに黙って取り直す。取り直せなければもう一度サインイン
- 403 (アカウントが未同意): リポジトリのページを示し、Enter で再試行
- 端末がないとき (`hwnpu-chat serve`、スクリプト): 同意のページと `HF_TOKEN` の作り方を書いたエラー
- API が gated と言わない 401 (存在しない名前、非公開) は普通のエラーで、サインインしない
- `HF_TOKEN` は CI のときだけ勧める

## Core AI の prebuilt (機種ごと)

細部と Apple 側のこと: [apple-coreai.md](apple-coreai.md)、[../cache-sharing.md](../cache-sharing.md)。

- 何か: 機種ごとの `coreai-build` の出力 (`X.ARCH.aimodelc`) から重みを除いたもの。`coreai.Pack` は `resources.bin` (大きさの 99 %) を `main.mlirb` の塊と突き合わせて `resources.bin.map.json` にする (231 MB の部品が 2.97 MB、組み立て直してバイト単位で同一)。`coreai.Bundle` で機種ごとに `prebuilt.tar.zst` (取るのは自分の機種だけなので、全機種で 1 つにはしない)
- 取る: Core AI の項目も手元の prebuilt もない部品を `Compile` が読むとき、書き出しごとにプロセスで 1 回、`coreai.FetchPrebuilt` が取得元に聞き、`EXPORT.prebuilt/ARCH/` に置く。`coreai.Assemble` が一時フォルダに組み立て (小さなファイルはハードリンク、重みは `main.mlirb` から、大きさと sha256 を照らす。違うバイトコードからは作らない)、Core AI に開かせてから消す。`HWNPU_COREAI_PREBUILT=off` で使わない
- 置く: `hwnpu-dev models package EXPORT/coreai-ARCH` (`--packed` は置かずに詰めるだけ、`--no-push` はソースの横に全体のまま残す)。部品ごとに作って詰め、全体の `.aimodelc` は消す (Sana の 27 本 × 12 種類を全体で持つと 50 GB だった)。機種は 1 つずつ作る: 12 種類を一度に作るとどれも置かれない。ディスクを見る: `coreai-build` の後に `$TMPDIR/com.apple.MetalPerformanceShadersGraphHost` に 12 GB が残った
- prebuilt のある機種にも `.aimodel` の書き出しは要る (重みはそこから)
- 置くモデル: 効くものだけ

| モデル | ネイティブの初回 | prebuilt あり | 機械 |
|---|---|---|---|
| FLUX.2 klein 4B 1024² | 52 分 16 秒 | 16 分 55 秒 | M2 16 GB |
| FLUX.2 klein 4B 1024² | 40 分 08 秒 | 7 分 48 秒 | M1 16 GB |
| Anima-Turbo 1024² | 45 分 01 秒 | 37 分 50 秒 | M2 16 GB |
| Chroma1-Flash 1024² の DiT、4 ブロックの部品 1 つ | 219 秒 | 66 秒 | M2 16 GB |
| LFM3-1.7B (チャット) | 2 分 17 秒 | 8 分 12 秒 | M2 16 GB |
| plamo-3-translate-8b (チャット) | 1 分 13 秒 | 7 分 08 秒 | M2 16 GB |

  - 遅い部品の多い画像のモデルが勝つ (FLUX.2 klein、Chroma1-Flash、Anima: `--arch all` で置く)。チャットのモデルは負ける: プロセスで最初の prebuilt の部品に `ANELargeModelCompilerService` で決まった 3〜6 分がかかる。Qwen3.5-4B は読めたが 1 時間 40 分答えなかった。チャットのモデルには prebuilt を置かない
  - 1 ステップの時間は同じ。FLUX.2 klein の絵は prebuilt で少し違う (PSNR 28.9 dB)。Chroma と Anima はバイト単位で同じ
- 制約:
  - 使えるのは 12 種類 (h13g、h14c/g、h15c/g、h16c/g/s、h17c/d/g/s)。M1 Pro / Max / Ultra、M2 Pro / Ultra、M3 Pro / Ultra、h15m、A18 では `coreai-build` が `ANEHostCompiler` で落ちる (SIGSEGV、プログラムに依らない)。M6 (h18g) は `redefinition of symbol named 'hwnpu_N'` で GPU に落ちる。Apple に報告した
  - ANE のコードが出るのは `--preferred-compute neural-engine` のときだけ
  - 端末でコンパイルしたもの (aned の `/Library/Caches/com.apple.aned`、SIP の中) は持ち出せないので、配れるのは `coreai-build` の出力だけ
  - ディスクが少ないと初回が崩れる: 空きが 12〜22 GB のとき、macOS の `deleted` が仕上げたばかりのプログラムを消し、aned が `Model load failed … isPreCompiledModel=1` を出し、Core AI が黙って端末でコンパイルした (最初のステップが 5〜15 分)。初回を測るときは 25 GB 以上空ける
  - 画像の書き出しは全部のアスペクト比を 1 つに持つので、prebuilt も全部の大きさを一度に覆い、`0 failed` で終わらなければならない ([new-model-checklist.md](new-model-checklist.md))

## 同じファイルをディスクの上で共有する

圧縮は効かない (bf16 のチェックポイントは lz4 で 100 %、zstd -3 で 77 %。lut6 の重みは zstd で 87〜90 %。しかも写像するものは展開した写しが要る)。無駄は重複: Q=8 と Q=64 のプログラムの同じ `weight.bin`、パッケージの重みの `.mlmodelc` の写し、いくつもの AMD の書き出しの同じ bank。直し方は copy-on-write の写しでブロックを共有すること。普通のファイルのまま。

| OS | `cache.Clone` の手段 | 共有する | 何もしない |
|---|---|---|---|
| macOS | `clonefile(2)` | APFS | — |
| Linux | ioctl `FICLONE` (`syscall` で。x/sys なし) | Btrfs、reflink つきの XFS、bcachefs、ブロッククローンを有効にした ZFS | ext4、tmpfs |
| Windows (`clone_windows.go`) | `FSCTL_DUPLICATE_EXTENTS_TO_FILE` | ReFS、Dev Drive | NTFS |

- 先に写しを作ってから dst と比べる: 写せないファイルシステムでは 1 バイトも読まず、比べる間に src が書かれても dst に入らない。dst の権限と更新時刻は保つ
- `cache.ClonePrefix`: 大きさの違う同じ名前のファイルの共通の先頭のブロックを共有する (Linux は FICLONERANGE、Windows は同じ呼び出し、macOS は何もしない)。例: 文だけの Gemma 4 の bank (1.67 GiB) は画像と音声つきの bank (2.64 GiB) の先頭。1 MiB に満たない共通部分は置き換えない
- 共有済みか: Linux は `sharedPrefix` で FIEMAP のエクステントを比べる (書き出し 1 つで 0.5 ms。多めには言わない)。macOS と Windows には安く問う手段がないので、一度読み比べて印を置く
- 使うところ:
  - 変換: `llm.convertPart` が同じフォルダの同じバイトのプログラムの重みを写しにする
  - コンパイル: `coreml` が `.mlmodelc` の重みをパッケージの写しにする
  - 前からあるファイル: `chat.Prepare` がモデルのフォルダごとに 1 回 `llm.ShareWeights` (印は `.weights-shared`)
  - AMD の書き出し: hwnpu-chat、hwnpu-agent の `-iron`、hwnpu-image が読む前に `iron.ShareExport` / `cache.ShareFiles`: 1 MiB 以上のファイルを、隣の書き出し (`manifest.json` があるか、名前に `.amd-npu` を含む) か前のファイルの同じバイトのものの写しにする。置き換えるのは読むフォルダだけ。印は `.files-shared` で、それより新しいファイルがあれば見直す。シンボリックリンクはたどらない
  - 元のファイル: 別のリポジトリで sha256 を固定したファイルはハードリンク
- 測った値: M3 でモデルのフォルダの重み 24.6 GiB が全部共有され、重みの実際のディスクは約 49.6 → 約 20 GiB。Btrfs で Gemma 4 E2B の AMD の書き出し 2 つ (7.0 + 8.2 GB) が 15.2 → 8.2 GiB、4 秒。出力は変わらない
- 落とし穴: 前からあるファイルを写しに置き換えた後の最初の読み込みは、プログラム 1 本ごとに 20〜50 秒 (Neural Engine のシステム側のキャッシュがファイルの置き換わりを見てコンパイルし直す)。2 回目からは 2 秒前後。新しく変換するものは最初から共有なので、1 回だけ
- 大きさはハードリンクを 1 回だけ数える (dev/ino)。macOS の `df` はこの比べには使えない (スワップとシステムのキャッシュで数十 GB 揺れる)。共有されていない大きさは APFS の `ATTR_CMNEXT_PRIVATESIZE`

## 手元のキャッシュの大きさと掃除

- `cache.Room(need, path, log)` を、取得元からの取得 (`store.FetchOriginal`)、変換結果の取得 (`store.materialize`)、Hugging Face からのダウンロード (`chat.fetchHF`、Content-Length で) の前に呼ぶ
  - `HWNPU_CACHE_MIN_FREE` (既定 10 GB、`0` で切る、割合も): これを切るなら始めない
  - `HWNPU_CACHE_MAX_SIZE` (既定なし): 超えるとき、と空きが足りないときに、最近使っていない単位から消す
  - 足りなければ `ErrNoRoom`。`chat.fetch` は止まり、Hugging Face に回らない
  - 上限を既定で持たない理由: 作り直しに時間のかかる単位 (AMD の書き出し) があるので、利用者が上限を決めたときだけ自動で消す。下限は既定であり、ディスクを埋めきらない
- 単位: 元のリポジトリ、`models/NAME`、`converted/.../vN`、`entries/TARGET/NAME`、`iron/KEY`、根のコンパイルの結果 (鍵ごと)、`prebuilt` の単位 (`EXPORT.coreai.prebuilt`、モデルの量に数えない)。最後に使った時刻はフォルダの mtime で、`cache.Touch` が進める (atime は当てにしない)。呼ぶのは `chat.fetch` の速い道、`chat.ModelDir`、`cache.FindExport`。一覧 (`Downloaded`) は時刻を変えない
- 消さないもの: 最近 1 時間に使ったもの (`cache.KeepRecent`)、このプロセスが使ったもの、書き戻しの queue にあるもの (`cache.SetBusy`)。走査はプロセスに 1 回
- 空きが足りず prebuilt があるときは、prebuilt のある部品の Core AI の項目 (`coreai-cache/<OS>/hwnpu*/<hash>`、重みの複製) を先に消す: 数秒で作り直せる。prebuilt そのものは残す
- 機械に縛られたコンパイルのキャッシュ (`cache.Prune`、チャットの読み込みの後): 元のパッケージ (`<key>.src`) が消えた項目、30 日使っていない項目、1 日たった一時フォルダを消す
- コマンド: 利用者は `hwnpu cache [ls|prune|clear|clean]`、取得元を運用する人は `hwnpu-dev models …` (取得元の 0 番が手元のキャッシュ)。フラグは [../cache-sharing.md](../cache-sharing.md)
- `modelstore` も同じものを出す (`Options.Limits`、`Usage`、`PlanPrune`、`Prune`、`Remove`、`PlanClean`、`Clean`、`ErrNoRoom`)

### 取得元の掃除 (`hwnpu cache clean [-n] [-y]`、`store.PlanClean`)

- `entry.json` のない `converted/…/VERSION`: 全部消してよい。あるもの: `.put-` の残りだけ消す
- `original/OWNER/NAME`: lock の名指さないファイル (前のコミットのもの、`.put-`) は消してよい。lock がなければ全部
- `converted/` と `original/` の外 (NAS の `.DS_Store` など): 触らない
- 最後に書いたのが 1 時間以内 (`store.WritingStale`) の `.writing-ID` の印があるか、ファイルの時刻 (WebDAV の `getlastmodified`、mtime、S3 の LastModified) のどれかが 1 時間以内のフォルダは、書いている途中として残す。印だけでは足りなかった: 印を置かない前の hwnpu が、計画と実行の間に書き始めた。消す直前にフォルダを見直す。印は最後に消す
- S3 の、Complete も Abort もされなかったマルチパート (一覧に出ないまま課金される) は、`converted/` と `original/` の下で `WritingStale` を過ぎたものを中止する (`ListMultipartUploads`、tinygodriver v1.3.10)

## 未解決

- 変換済みのモデルの Hugging Face での配布 (組織 `hwnpu`、1 モデル 1 リポジトリ、取得元から `hwnpu-dev models hf push` で手で上げ、元のリポジトリの前の層として読む): 読む層、`hub` の固定、`hwnpu-dev models hf plan / push / check` を実装。まだ何も上げていない。書くための device flow、20 GB を超える書き出しを切ること、gated で断られたら止めることが残り ([../prebuilt.md](../prebuilt.md))。今は利用者の取得元で共有する
- 手元の並び: `models/NAME` のプログラムと `entries/` を `converted/MODEL/TARGET/EXPORT/vN/` に移し、完了の印を `entry.json` 1 つに (今は `NAME.done`、`NAME.ext.done`、`entry.json` が混在)。`~/models` を一度移して読まなくする。`modelstore` に変換の API (`Make`)。同梱の変換結果をその場で使う (今はコピー)
- Core ML と OpenVINO のプログラムのエントリ、機械に縛られた (`host`) エントリの取得元での共有、対象の名前での AMD / Intel の NPU の世代の区別
- 道具の引数を Go が持たない手で作る画像の書き出し (Anima): 今は名前でだけ
- `hwnpu-dev models push` は初回の FLUX.2 klein の書き出しを知らない
- 実機で確かめていない: AWS の S3、Synology の WebDAV、Windows のロックと ReFS / Dev Drive の写し、bcachefs、ZFS、NFS での写し、Windows と macOS で実際のダウンロード中のディスクの空き
- `FormatVersion` 2 の後、8 GB の列の Core AI のチャットの書き出し (LFM3-1.7B、Qwen3.5-0.8B、Qwen3-1.7B) をまだ作り直して置いていない。Anima-Turbo の 1024² の Core AI の書き出しはまだ取得元にない。Qwen2.5-Coder-7B は GPTQ の codes を作り直し (`tools/ref/qwen2_gptq.py`)、一度変換する要がある
- prebuilt の仕上げの最初の部品の決まった費用、落ちる 9 種類、M6 の名前の衝突、FLUX の数値の違いは Apple 待ち (作り直すには新しい Metal Toolchain が要る)
- HF の refresh token の保存 (OS の資格情報の置き場) と、Actions のトークンの交換 (Trusted Publishers、RFC 8693)
- チェックポイントは実行中も埋め込みの表のために開いている。変換時に表だけを書き出せば LFM3 は 3.5 → 0.27 GB、Gemma 4 は 10.2 → 約 5.5 GB (再変換にはダウンロードし直しが要る)

## 採らなかったもの

- チェックポイントや重みのディスクの上での圧縮: lz4 で約 100 %、zstd で 71〜90 %。しかも写像するものは展開した写しが要る
- models.yaml の取得元 (`sources`、`search`、`cache.layers`、`from: SOURCE:NAME`)、`HWNPU_CACHE_LAYERS`、`HWNPU_CACHE_WRITE`: 環境変数の URL に置き換えた。まだ書いたファイルは移し方を書いたエラー
- 内容で番地を決める並び (`blobs/sha256/…`、`upstream/`、`entries/KIND/KEY.json`) とその重複除去: 人が見て分からず、普通の HTTP サーバーで配れない。前のデータはハードリンクで移した (`hwnpu-dev models migrate`。NAS を移し終えて 2026-11-22 に外した)
- `HWNPU_OFFLINE` (`HWNPU_MODEL_HUGGING_FACE=false` を使う) と `HWNPU_PREBUILT` (取得元にあれば使う)
- Hugging Face のトークンのディスクへの保存: メモリだけ
- ONNX Runtime のライブラリの自動の取得: ネイティブのライブラリの配布は利用者が決める (`HWNPU_ONNXRUNTIME` かシステムのもの)
- キャッシュの大きさの既定の上限: 作り直しに時間のかかる単位がある
- 全機種の prebuilt を 1 つの圧縮ファイルで取る (zstd -19 --long で 36.98 → 1.33 MB): 各 Mac は自分の機種のものだけ要る
- チャットのモデルの prebuilt: 初回が遅くなり、Qwen3.5-4B は止まった
- 端末でコンパイルした Core AI の結果の持ち出し: SIP の中、署名の識別子ごと
- Sana (全モデル) と Qwen-Image-2.1 (Turbo と無印、Qwen Research のライセンスの確認ごと): models.yaml から消した。Qwen-Image は NPU で使うにはパラメータが多すぎ、Sana も一緒にやめた (Core AI の prebuilt もほとんどの部品が ANE に乗らなかった)
- 確かめる S3 サーバーとしての MinIO: 保守が終わった
- `~/models` への書き込み: 読むだけの古い層
- Cloudflare R2 の無料枠に大きなモデル: 10 GB までなので小さなものだけ
