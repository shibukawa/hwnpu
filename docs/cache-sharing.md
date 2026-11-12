# 変換済みのモデルの配布と、キャッシュの共有 (設計案)

作成: 2026-11-12。状態: P0、P1 の層とフォルダ、P2 の S3、AMD のチャットのモデルの書き出しの共有を実装 (0 節、design.md 16.169)。ほかは設計案。docs/prebuilt.md (Hugging Face で配る計画) と docs/model-table.md の 5 節 (取得元)、11 節 (運用) を、1 つの仕組みにまとめ直したもの。食い違うところはこの文書を正とする。

## 0. 実装したもの (2026-11-12)

- 根を 1 つに (`cache.Root`、`HWNPU_CACHE`)。AMD の書き出しは `entries/amd-xdna2/` に作り、`~/models` は読むだけ。IRON の JIT のキャッシュは `iron/` (`~/.npu/cache` を最初の書き出しの前に移し、リンクを残す)。組み込みの IRON の道具は `tools/`。まだ移していないもの: `models/` (チェックポイントと Core ML、OpenVINO のプログラム) を `src/` と `entries/` に分けること
- `.done` を相対パスに、ダウンロード・変換・書き出しのロック、HF の続きのダウンロードの `If-Range`
- 層 (7 節): フォルダと S3 (tinygodriver、7.2 節)、`write` と `fill` と `fill_gated` と `scopes`、組織のファイル (`/etc/hwnpu/models.yaml` など)、`HWNPU_CACHE_LAYERS`、`HWNPU_CACHE_WRITE=0` (7.4 節)。後のファイルの `cache.layers` は並びごと置き換える (層ごとの差分ではない)
- 元のチェックポイントのフォワードプロキシ (7.1 節): 定義の sha256、HF の HEAD の `X-Linked-Etag`、コミットの索引の順で層を探す
- エントリ (6 節): `entry.json` と、AMD のチャットのモデルの書き出しの共有 (`chat.SharedExport`)。鍵は指紋 (モデル、コミット、定義の sha256、`amd_export`、組み込みの道具の hash)。版 (`version`、5 節の lock) はまだない
- `hwnpu-dev cache layers|ls|push|verify|flush`
- 次: 画像のモデルの書き出し、Core ML と OpenVINO のプログラムのエントリ化 (8 節の計画と作るの分離)、`host` の scope、HF の配布物 (4、5、9 節)、`cache gc`

## 1. 目的

- 利用者は、変換済みのものを落として読み込むだけにする。AMD の利用者に IRON、mlir-aie、Peano、Python の環境を入れさせない。GPTQ の校正 (CPU で数時間) も手元でさせない
- 変換済みのものは GitHub Actions (aix1 の self-hosted runner) で作り、照合してから Hugging Face に上げる。上げる経路はこれだけにする
- 配っていない組み合わせ (利用者が `builds` で変えたもの) や社内の機械の間でも、作ったものを透過的に使い回せるようにする。共有フォルダ (NFS、SMB、外付け) と S3 を層として重ねる
- 配布物が使えなければ、今までどおり手元で変換する (後退させない)

## 2. 今の状態と、共有を妨げる点

できているもの: 元のチェックポイントの取得元の切り替え (`sources`、`search`、path と s3、`internal/chat/sources.go`)、`sha256` の固定とモデルの間のハードリンク、同じバイトのファイルの reflink での共有 (`cache.Clone`、`ShareFiles`)、`cache.Store` の一時ディレクトリと rename。変換済みのものを取ってくる経路はない。

直すもの (7 節の P0):

1. 変換の印 `.done` に絶対パスが入り、ローダーがそれを返す (`internal/chat/model.go` の `convert`、書き起こし、Anima、Sana、Qwen-Image)。写したフォルダは元の機械のパスを指す
2. `cache.Prune` は `.src` の絶対パスで元の有無を見る。共有した先では元がないと見て消す
3. プログラムの名前は設定だけで、元のチェックポイントのコミットも hash も入らない。モデルのフォルダは `path.Base(repo)` で、`owner1/X` と `owner2/X` がぶつかる
4. ロックがない。2 つのプロセスが同じ `.part` に書きうる
5. HF からの続きのダウンロード (Range) に `If-Range` がない。版の違う `.part` に足しうる
6. Core ML の配置の判定 (placement の JSON) の鍵にチップが入らない
7. AMD の書き出しはキャッシュではなく `~/models` にある。版は `manifest.json` の `version: 1` だけ

## 3. 版

2 種類の版を持ち、どちらも手で明示的に上げる。

| 版 | 単位 | 上げるとき | 見るもの |
|---|---|---|---|
| `format_version` | 対象の会社ごと (`apple`、`amd`、`intel`) | ローダーが読む形式が変わったとき (IRON の `manifest.json`、`parts.json` の並びなど) | ローダー。バイナリごとに読める範囲をコードで持つ |
| モデルの版 `version` | モデル × 対象 | 出力が変わるとき全部: 形式やパラメータ、元のコミット、変換器や校正の修正、`format_version` | 配布と利用者。HF の上の場所を決める |

### 3.1 対象

対象は model-table.md 4.1 節の NPU の名前 (世代の階層) を使う。「広い名前の行が既定、狭い名前の行が例外」も同じ。

- 機械の名前を狭いものから並べ (M1: `apple-m1` → `apple`、Strix Point: `amd-xdna2` → `amd`)、定義のある最初の行を使う
- 版は行ごとに独立。`apple` の版を上げても `apple-m1` は作り直さない
- M1 で中身が同じものは M1 の行を書かない (`apple` を共有)。M1 で使わないものは `apple-m1: {remove: true}` (今の `not_on: [M1]` に当たる)。配布物を探さず、手元でも作らない
- 前提: 世代を見分けるコードは Apple にしかない。AMD (`amd-xdna1` / `amd-xdna2`) と Intel (`intel-npu3` / `intel-npu4`) を分けて配るには、`hwnpu.Enumerate` が世代を返すようにする
- `prebuilt.md` のサブフォルダ `amd-npu2/` は `amd-xdna2/` に揃える (書き出しの中身の IRON の `npu2` はそのまま)

## 4. 定義 (組み込みの `models.yaml`)

`prebuilt.md` 5.3 節の別ファイル (`prebuilt/models.yaml`) は持たず、モデルの定義に書く。バイナリに組み込まれるので、各リリースの hwnpu は自分の取る版を知っていて、HF に新しい版が上がっても古いバイナリは古い版を取り続ける (再現できる)。

```yaml
models:
  LFM3-1.7B:
    from: hf:LiquidAI/LFM3-1.7B@6c1f0e3a...     # 配るモデルはコミットの固定が必須 (40 桁)
    prebuilt:
      repo: hf:hwnpu/LFM3-1.7B
      targets:
        amd-xdna2: {version: 2, format: q4, codes: gptq}
        intel:     {version: 1, format: int8, publish: staging}
        # apple は 10 節が決まるまで書かない
```

- 行に書いたパラメータ全部が、その版の中身。実行時の設定 (フラグ、`builds`、`variants`) がこれと合えば配布物を使い、合わなければ手元で変換する (`-explain` に理由)
- `like: 行の名前` で、ほかの行の値を引き継いで違うところだけ書ける (`apple-m1: {like: apple, version: 2, prefill: off}`)
- 利用者のファイルで版やリポジトリを差し替えられる (社内で作り直したもの)
- 表 (`tables.safetensors`) のように対象によらないものは `targets.any` に書く

## 5. 指紋と lock ファイル (上げ忘れの検査)

`hwnpu-dev prebuilt fingerprint` が、モデル × 対象の行ごとに指紋 (sha256) を計算する。入力:

- 行のパラメータ (`like` を解いた後)、対象の名前
- `from` のコミットと `sha256`
- 変換器の版の定数 (`Arch.Revision`、`ExportVersion`、`MANIFEST_VERSION` など)
- `format_version`
- 道具の版 (mlir-aie、Peano。`tools/ci/aix1-setup.sh` に固定したもの)
- 校正の設定 (GPTQ の damp、データ、行数)

`prebuilt/lock.json` に `{model, target, version, fingerprint}` を記録する (`go.sum` に当たる)。PR の検査:

| 状態 | 結果 |
|---|---|
| 指紋が変わったのに `version` が同じ | 失敗 |
| `version` が下がった | 失敗 |
| lock を更新していない | 失敗 |
| 指紋が同じで `version` だけ上がった | 警告 (作り直しを意図したとき) |

- `like` で引き継いだ値が変われば引き継いだ行の指紋も変わるので、その行の版も上げることになる
- 捕まえられないのは、変換器を直して版の定数を上げなかったときだけ (今と同じ)。補いに、CI で作ったものの sha256 と KL を前の版と比べて記録し、「指紋は同じなのに出力が違う」を警告にする。比べるには変換が同じバイトを出すかを確かめる (未確認。決定的でないものは KL で比べる)
- 指紋は 6 節のエントリの鍵でもある。`版 → 指紋` は lock で引く。利用者が同じパラメータを組めば、配布物にもチームのキャッシュにも当たる

## 6. エントリ

再利用するものを全部、不変のエントリにする。エントリはファイルの木と `entry.json`:

```json
{
  "key": "<指紋>", "kind": "iron-lm", "scope": "portable",
  "model": "LFM3-1.7B", "target": "amd-xdna2", "version": 2, "format_version": 1,
  "recipe": {"...": "指紋の入力そのもの"},
  "files": [{"path": "decode.elf", "size": 123, "sha256": "..."}],
  "made_by": {"hwnpu": "<コミット>", "tools": "<amd-tools の hash>", "iron": "v1.4.4"},
  "checked": {"kl": 0.036, "top1": 0.97, "machine": "aix1", "placement": "all on the NPU"}
}
```

- 鍵は作る前に計算できる (models.yaml の固定値から。落としたファイルからではない)。だからチェックポイントを落とさずに探せる
- 中のパスは全部相対。`.done` は `entry.json` に置き換える (書き終えてから最後に置く)
- エントリの単位: プログラム 1 本 (`.mlpackage`、IR)、IRON の書き出し 1 つ、表、GPTQ の符号、SigLIP のラベルの埋め込み

共有できる範囲 (`scope`):

| scope | 例 | 置ける層 |
|---|---|---|
| `upstream` | HF のチェックポイントのファイル (鍵は sha256) | path、s3 (今のミラー) |
| `portable` | 変換済みのプログラム、IRON の書き出し、表、符号 | hf、path、s3 |
| `host` | `.mlmodelc` (OS の版)、Level Zero の blob (ドライバとデバイス)、placement (チップ) | path、s3。鍵に機械の指紋 |
| `local` | プロンプトのキャッシュ、e5 バンドルのキャッシュ、履歴、ライセンスの同意、ロック | 共有しない |

## 7. 層 (透過型のキャッシュ)

```
読み: 手元 → 層 1 → 層 2 → ... → hf → 手元で作る
書き: 作ったら手元に置き、write: true の層へ書き戻す (失敗しても続ける)
```

| 層 | 並び | 読み書き |
|---|---|---|
| local (`HWNPU_CACHE`、既定は `UserCacheDir/hwnpu`) | 今の並び + `entry.json` | 読み書き |
| path | `blobs/sha256/ab/<sha256>` と `entries/<kind>/<key>.json` (内容で番地を決める) | 読み書き |
| s3 | path と同じ。`github.com/shibukawa/tinygodriver/storage/s3` (7.2 節) | 読み書き |
| hf | 9 節の並び。Actions だけが書く | 読みだけ |

- 書くときは blob を全部上げてから最後にエントリの JSON。JSON があれば揃っている。2 台が同じ鍵を同時に作っても、blob は同じ番地に重なるだけ
- 重複はファイル単位で除かれる (Q=8 と Q=64 の `weight.bin`)。手元では sha256 の索引を引き、同じものがあれば reflink (`cache.Clone`)。先頭だけ同じものの共有 (`ClonePrefix`) は手元だけの最適化
- 落としながら sha256 を計算し、違えばその層を飛ばして次へ (ミラーと同じ規則。この機械では大きなダウンロードにビットの反転があった)。HF の LFS は ETag が sha256 なので落とす前にも分かる
- 信頼: 書ける層は書ける人を信頼する (xclbin や ELF は NPU で動く)。HF は名前空間を信頼する。署名 (ed25519 でエントリの JSON に) は後で足せる形にする
- `Prune` は手元の層だけを掃除する。共有の層の掃除は `hwnpu-dev cache gc` で明示的に

設定は今の `sources` の定義を使う:

```yaml
sources:
  team-nfs: {type: path, path: /mnt/share/hwnpu}
  lab-s3:   {type: s3, bucket: models, prefix: hwnpu/, profile: rustfs}
search: [team-nfs, lab-s3, hf]           # 元のチェックポイント (今と同じ)
cache:
  layers:                                # 変換済み
    - {use: team-nfs, write: true, scopes: [portable, host]}
    - {use: lab-s3, write: true}
  prebuilt: auto                         # auto | off (常に手元で) | only (手元で作らない)
```

組織や環境ごとの設定の置き方は 7.4 節。

### 7.1 層に置くもの: 書き戻しと、読んだものの写し (fill)

層ごとに、何を置くかを 2 つの欄で決める。どちらも既定は `false`。

| 欄 | 置くもの | 使い方 |
|---|---|---|
| `write` | 手元で作ったエントリ (8 節の 3) | バリエーションや配っていない組み合わせの共有 |
| `fill` | より後ろの層や上流から読んだものを、そのまま (フォワードプロキシの共有キャッシュ) | 社内の 1 台が一度落とせば、ほかの機械は HF に出ない |

```yaml
cache:
  layers:
    - {use: team-nfs, write: true, fill: true}
    - {use: lab-s3, fill: true, scopes: [upstream, portable]}
```

`fill` で置くもの:

- HF の配布物 (エントリ): 落として sha256 を確かめた後、同じ `entry.json` と blob をその層に置く。次の機械は手前の層で当たる
- 元のチェックポイント (`upstream`): 今のミラーの並び (`OWNER/NAME@COMMIT/FILE`、model-table 5.4 節) で置く。`search` にその層を入れておけば、今の `fetch` のままで当たる
- 手前の層からの写し: 層 2 で当たったものは層 1 にも置く (層 1 が速い NFS、層 2 が遠い S3 のとき)

規則:

- 置くのは確かめたものだけ: エントリは `entry.json` の sha256、チェックポイントは固定の `sha256`。固定のないファイルは HF の応答の ETag (LFS は sha256) と照らす。確かめられないもの (LFS でない小さなファイルで固定がないもの) は、取った時点のコミットの下にだけ置く
- 場所は必ずコミットで決める。`@main` のように固定していないモデルは、HF の応答の `X-Repo-Commit` で実際のコミットを取り、その下に置く。ブランチの名前の下には置かない (古いものを新しいと見せないため)
- 手元で揃ってから、裏で写す (落としながら 2 か所に書かない)。写し終わらずにプロセスが終われば、`work/fill-queue` に残して次の起動で続ける。blob が先、JSON が最後 (7 節) なので、途中で切れても壊れたものは見えない
- 同意の要るリポジトリ (gated、Gemma など) とライセンスの確認のあるモデル (16.155) は既定で写さない (`fill_gated: true` で写す)。写したものを読む機械でも、ライセンスの確認は手元で今どおりにする
- 置けなかったとき (書けない、容量がない) は警告を 1 行出して続ける。層への書き込みの失敗で実行を止めない

別案: HF の API (`/resolve/`) を真似るキャッシュのサーバー (`hwnpu-dev cache serve`) を 1 台に置き、`HF_ENDPOINT` で向ける。hwnpu 以外の道具 (huggingface_hub、IRON の環境) も使えるが、サーバーの運用が要る。まずはサーバーのない形 (NFS、S3 に各機械が fill) で始め、要れば後で足す。

### 7.2 S3 のクライアント

AWS SDK (aws-sdk-go-v2) は重いので使わない。`aws` CLI を呼ぶ今の形 (model-table 5 節、`internal/chat/sources.go` の `fetchS3`) もやめ、`github.com/shibukawa/tinygodriver` の `storage/s3` (署名は `cloud/aws`) にする。元のチェックポイントの取得元 (`type: s3`) と、キャッシュの層の両方をこれ 1 つで読む。

- 依存: 標準ライブラリと tinygodriver の中の小さなパッケージだけ (`cloud/aws`、`internal/cloudhttp`、`internal/hmacsha256`、`encoding/simdhex`)。モジュールは 1 つで go.mod の require は多いが、リンクされるのは使うパッケージだけ
- 使う操作: `Get` / `GetRange` (続きから落とす)、`Head` (エントリの有無)、`Put`、マルチパートの 4 つ、`List` (gc と ls)
- 設定の欄は `endpoint` (`WithEndpoint`)、`region` (`WithRegion`)、`path_style` (`WithPathStyle`。既定は amazonaws.com なら仮想ホスト形式、それ以外はパス形式なので、RustFS で `~/.aws/config` に `addressing_style` を書く手順は要らなくなる)。大きなファイルは `WithUnsignedPayload` で、本体を読み込まずに送る
- tinygodriver が持たないものを hwnpu の側で足す:
  - 認証: tinygodriver は静的な値と環境変数 (`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_SESSION_TOKEN`、`AWS_REGION`、`AWS_ENDPOINT_URL_S3`) だけ。`profile` は、hwnpu が `~/.aws/credentials` と `~/.aws/config` のその節の静的な鍵 (`aws_access_key_id`、`aws_secret_access_key`、`aws_session_token`、`region`、`endpoint_url`) だけを読んで渡す。SSO とインスタンスのメタデータは持たない。SSO の利用者は `eval "$(aws configure export-credentials --profile X --format env)"` で環境変数にしてから使う (docs に書く)
  - マルチパート: `Put` は 1 回の要求 (上限 5 GiB) で、分割は自分でする。64 MiB を超えるファイルは 64 MiB の部分に分けて上げ (IRON のバンクには数 GB のものがある)、失敗したら必ず `AbortMultipartUpload`。途中で止まった分は gc が `List` で探して消す
  - 再試行: 5xx、接続の切断、`SlowDown` は指数的に待って 3 回まで。ダウンロードは `GetRange` で続きから
  - `no_sign` (署名なしの公開バケット): tinygodriver で鍵なしの要求が送れるかを確かめる。送れなければ、公開バケットは `https://` の取得元 (HTTP の GET) として読む
- 確かめたこと (2026-11-13、RustFS 1.0.5 を手元で): 部分に分けた送り (5 MiB の部分で 12 MiB)、空のファイル (tinygodriver v1.3.8 は長さ 0 の本体を chunked で送り 411 になった。v1.3.9 で直った)、`GetRange` の続き、一覧、削除、パス形式 (`internal/store` の `TestS3Server`、`HWNPU_S3_TEST_ENDPOINT` を与えたときだけ)。Qwen3.5-0.8B の書き出し (1.57 GB) を `push` (10 秒) し、空の根で IRON なしに層から取って NPU で答えた (準備 4.4 秒)
- Cloudflare R2 で (2026-11-13): `region: auto` で署名が通り、`TestS3Server` が通った (`HWNPU_S3_TEST_BUCKET` に既にあるバケット、`HWNPU_S3_TEST_REGION=auto`)。Qwen3.5-0.8B の書き出しを `push` (1.57 GB、119 秒。この機械の上りは無線で約 13 MB/s)、空の根と IRON なしの hwnpu-chat が層から取って答えた (準備 54 秒、取るのは約 30 MB/s)。R2 の無料枠は保存 10 GB/月なので、試しには小さな書き出しだけを置く
- 確かめること: AWS の S3 で仮想ホスト形式と、ほかのリージョンへのリダイレクト
- MinIO は対象にしない (保守が終わった)。S3 互換のサーバーの確認は RustFS で

### 7.3 手元の置き場所

hwnpu が書くものは全部、OS のローミングしないキャッシュのフォルダの下の 1 つの根に収める。ほかの場所には書かない。

| OS | 根 | 備考 |
|---|---|---|
| macOS | `~/Library/Caches/hwnpu` | iCloud に同期されない。Time Machine も既定で除く |
| Linux | `$XDG_CACHE_HOME/hwnpu` (`~/.cache/hwnpu`) | |
| Windows | `%LOCALAPPDATA%\hwnpu` | `Roaming` (`%APPDATA%`) には置かない。ユーザーの `models.yaml` も同じ根にあるので、掃除は下のサブフォルダだけを触る |

どれも `os.UserCacheDir()` の下で、`HWNPU_CACHE` で根ごと移せる (外付けのディスク)。根の下の並び:

```
hwnpu/
  src/<owner>/<name>@<commit>/       元のチェックポイント (upstream)。owner とコミットで分けるので取り違えない
  entries/<target>/<model>/<kind>-<key12>/   変換済みのエントリ (entry.json)。人が見て分かる名前、本体は鍵
  host/<key>.mlmodelc|.blob|.json    機械ごとのもの (Core ML のコンパイル、Level Zero、placement)
  work/                              作業中 (.part、変換と書き出しの途中)。rename と reflink が効くよう同じファイルシステムに置く
  tools/amd-tools-<hash>/            組み込みの IRON の道具
  iron/                              IRON の JIT のキャッシュ (NPU_CACHE_HOME)
  prompts/  history/  locks/         local の scope
```

今その外に書いているものを移す:

- AMD の書き出し (`~/models/NAME.amd-npu2*`): `entries/amd-xdna2/...` へ。`~/models` は開発の手で作ったものの置き場として、読みだけの古い層として探し続ける (書かない)
- IRON の JIT のキャッシュ (`~/.npu/cache`、この機械で 52 GB): `RunIRON` が `NPU_CACHE_HOME=<根>/iron` を渡す (mlir-aie の `python/utils/compile` が読む)
- 作業中のものを `/tmp` に置かない (Linux では RAM の tmpfs のことがあり、GB 級の書き出しで遅くなる)
- Apple の e5 バンドルのキャッシュ (`~/Library/Caches/<プロセス>/com.apple.e5rt.e5bundlecache`) は OS が持つもので、すでにキャッシュのフォルダの下。場所は変えない
- ユーザーの設定 (`models.yaml`、`licenses.json`) は今の設定のフォルダのまま (キャッシュではない)

### 7.4 組織と環境ごとの設定

書き戻し (`write`) と写し (`fill`) の先は、チームの S3 や NFS のキャッシュの層 (7 節)。HF には書かない (9 節の Actions だけ)。どの層を使うかは、利用者ではなく組織や環境で決まることが多いので、設定を 3 段で重ねる (あとのものが勝つ):

| 段 | 場所 | 使い方 |
|---|---|---|
| 組織 (機械全体) | Linux `/etc/hwnpu/models.yaml`、macOS `/Library/Application Support/hwnpu/models.yaml`、Windows `%ProgramData%\hwnpu\models.yaml` | 管理者が配る (MDM、構成管理)。社内の `sources`、`search`、`cache.layers`、社内のモデルやエイリアス |
| 利用者 | 今のユーザーのファイル (model-table.md 3 節) | 個人の差分 (層を外す、手元の NFS を足す) |
| 環境 | 環境変数 (下の表)、`--models FILE` | CI、コンテナ、一時的な切り替え |

- 組織の段は、今の重ね方 (組み込み → ユーザー → `HWNPU_MODELS` → `--models`) の、組み込みとユーザーの間に入れる。形も重ね方も同じ models.yaml なので、新しい形式を作らない
- 鍵 (S3 の認証) は設定のファイルに書かない。環境変数か `~/.aws/credentials` のプロファイル (7.2 節)。組織のファイルには `profile: team` のような名前だけを書く
- 層の欄 `write`、`fill`、`fill_gated` も層ごとに組織のファイルで決める。利用者のファイルで `false` にはできるが、組織の層を勝手に増やす向きの上書き (`write: true`) は警告を出して従う (組織が書き込みの権限で止める)

環境変数:

| 変数 | 働き |
|---|---|
| `HWNPU_CACHE` | 手元の根 (7.3 節) |
| `HWNPU_CACHE_LAYERS` | 層の並びを丸ごと置き換える。`s3://bucket/prefix?write=1&fill=1&profile=team,/mnt/share/hwnpu?fill=1`。`off` で共有の層を全部外す |
| `HWNPU_CACHE_WRITE=0` | どの層にも書かない (読むだけ) |
| `HWNPU_OFFLINE=1` | ネットに出ない (model-table.md 11.3 節)。path の層だけを読む |
| `HWNPU_PREBUILT` | `auto`、`off`、`only` (7 節の `prebuilt`) |

`hwnpu-dev cache layers` で、今効いている層の並び、各層の `write` と `fill`、それをどのファイルや変数が決めたかを出す。

## 8. 準備の流れ: 計画と作るを分ける

今の `chat.Prepare` と `PrepareAMD` は、まずチェックポイントを落とす。これを次にする:

1. 計画: モデル、対象、設定から、要るエントリの一覧と鍵を出す (表、プログラム、MTP、エンコーダ...)
2. 探す: エントリごとに 7 節の順で
3. 作る: なかったものだけ。チェックポイントを落とすのはこのときだけ。作ったら書き戻す

エントリごとに当たり外れが決まるので、一部だけの当たりも効く。`-explain` で各エントリの出どころと作った理由を出す。画像、音声、ビジョンの各 `prepare.go` も同じ形に寄せる。

### 8.1 手元で作るとき: 元データも残す

配布物がない (配っていないモデル、配っていない対象、`-prebuilt off`) ときは、今と同じく手元で作る。このときは元のチェックポイント (`src/`) と変換したエントリ (`entries/`) を両方残す。

- 理由: 既存のモデルのパラメータを変えたバリエーション (文脈の長さ、形式、Q、MLP の q6 など) を作るたびに、元を落とし直さないため。model-table.md 11.4 節の「変換が揃ったらすぐ元を消す」は採らない
- 配布物だけで動いているモデルは元を落とさない (8 節の 3 に行かない)。`src/` はそもそもできない
- 元を消すのは 11.4 節の「長く使っていない」(`cache.unused_days`) と全体の上限 (`cache.max`) のときだけ。元の最後に使った日時は、それから作ったエントリを使った日時でも更新する (バリエーションを使っている間は元を残す)。`hwnpu-dev cache gc --sources` で明示的に消せる
- 消した後にバリエーションを作るときは、固定どおりのものを落とし直す (5 節の `sha256` で確かめる)

### 8.2 バリエーション

利用者のファイルに、既存のモデルを引き継いでパラメータを変えたモデルを書く。

```yaml
models:
  LFM3-1.7B-32k:
    like: LFM3-1.7B
    context: 32768
    prebuilt: {targets: {amd-xdna2: {like: LFM3-1.7B/amd-xdna2, mlp: q6}}}   # 配布の行を引き継いで変える
```

- 指紋は変えたパラメータで計算し直す (5 節)。配布の行と同じになれば配布物に当たり、違えば手元で作る。版は持たない (配らないので。lock も要らない)
- 作ったエントリは `write: true` の層 (チームの NFS や S3) に書き戻されるので、同じバリエーションを使う別の機械は作らずに済む
- 手元で作ったものと配布物は同じ `entries/` に、同じ鍵の規則で入る。名前で区別しない
- 元が同じなので、`src/` は元のモデルと共有する (`like` の `from` が同じ)

## 9. GitHub Actions と Hugging Face

安全の手当ては prebuilt.md 5.1 節のまま (PR では self-hosted を動かさない、actor の制限、runner group、専用のユーザー、`HF_TOKEN` は environment)。

```
PR (ホストランナー)
  models.yaml と lock の検査 (5 節)、go test。モデルは作らない

main への push (aix1)
  1. 前のコミットと比べ、version が上がった (または増えた) モデル × 対象を出す
  2. 元のチェックポイント (aix1 のキャッシュ、コミット固定)
  3. 校正 (GPTQ)。指紋の校正の部分が同じならキャッシュから
  4. 変換と書き出し
  5. 照合: milcpu の KL、AMD は NPU の上の ironcheck と KL、Intel は OpenVINO の CPU。上限を超えたら失敗
  6. entry.json を作り、HF の staging に上げる
  7. NPU の上で照合したもの (amd-xdna2) は、通れば自動で main に進める

workflow_dispatch (owner だけ)
  rebuild: 同じ version で作り直す (staging だけ上書き。main に進んだものは上書きしない)
  promote: 実機 (Mac、Intel) で確かめた後に staging から main へ
```

ブランチ: Actions は `staging` に上げ、確かめ終えたコミットを `main` に進める (`main` をそのコミットにする)。利用者の hwnpu は既定で `main` だけを読む (`--revision staging` で試せる)。照合に通らなかったものは `staging` に留まり、利用者に届かない。

トークン: 組織 `hwnpu` の fine-grained トークン (リポジトリへの書き込みだけ、組織 `hwnpu` に限る) を GitHub の environment `huggingface` の `HF_TOKEN` に置き、`main` からのジョブにだけ渡す。リポジトリはジョブが初回に作る (公開、gated の元モデルは 8 節のとおり同じ条件で gated に)。

HF のリポジトリ (組織 `hwnpu`、1 モデル 1 リポジトリ):

```
README.md         モデルカード (元のモデル、ライセンス、照合の結果)
hwnpu.json        全部の版の目録 (各 entry.json の写し)
amd-xdna2/v2/...  版ごとのフォルダ。古いバイナリのために前の版も残す
intel/v1/...
any/v1/...        表など対象によらないもの
```

- 版の間で変わらないファイルは HF の Xet が重複を除くはず。ダウンロードの量にも効くかを確かめ、効かなければ entry.json で前の版のファイルを参照する
- 古い版は、参照するリリースがなくなってから手で消す
- 上げるのは `hwnpu-dev cache push --to hf:hwnpu/LFM3-1.7B --revision staging` (手元のエントリをそのまま上げる。CI 専用の形を持たない)

## 10. Apple (白紙)

Apple の機械が使えるようになるまで、Apple の配布の形は決めない。`targets` に `apple` の行は書かない。

- 今の Core ML の形: 配れるのは `.mlpackage` まで。`.mlmodelc` は OS の版ごと (`host`)、Neural Engine のコンパイル (e5 バンドル) は機械ごとで、初回の重さ (Gemma 4 で 100 秒級) が残る
- Core AI の方が事前にビルドしたものを配れる見込みがあり、そうなら Apple でも初回のコンパイルをなくせる。確かめること: 何を事前にビルドできるか、それがチップの世代と OS の版のどちらに縛られるか、Neural Engine への配置を確かめられるか
- この仕組みの側で備えること: 対象の世代の階層 (3.1 節) にチップは入っている。OS の版に縛られる形なら、エントリに `os: {min, max}` を持たせ、選ぶときに見る (`host` にせず `portable` のまま、OS の範囲で選ぶ)

## 11. 進める順

1. P0 (手元だけ): 7.3 節の置き場所 (根を 1 つに、`src/` と `entries/`、`NPU_CACHE_HOME`、今の場所からの移し方)、`.done` を `entry.json` に (相対パス)、`source.json` とフォルダ名に owner、エントリごとのロック (flock / LockFileEx)、`If-Range`、placement の鍵にチップ、AMD の書き出しをキャッシュの下へ (`~/models` は読みだけの古い層として探す)、`Prune` を手元の層だけに、指紋の計算
2. P1: `Store` の interface、計画と作るの分離、path の層 (`write` と `fill`)、`hwnpu-dev cache ls|pull|push|verify|gc`。この機械と Intel の機械の間で試す
3. P2: s3 の層を tinygodriver で (元のチェックポイントの `type: s3` も `aws` CLI から移す)
4. P3: models.yaml の `prebuilt`、lock と PR の検査、aix1 の runner と Actions、HF を読む層。最初は LFM3-1.7B の `amd-xdna2`
5. P4: `host` の scope をチームの層へ。世代を見分けるコード (AMD、Intel)
6. Apple は 10 節が決まってから

## 12. 決めたこと (2026-11-12)

- 版は手で明示的に上げる (ソースの hash で自動にはしない)。データの形式の `format_version` と、モデル × 対象の `version`
- 対象は model-table.md 4.1 節の世代の名前。M1 と M2 以降のように分かれるものは行を分ける
- 上げるのは GitHub Actions だけ
- Apple は白紙 (10 節)
- S3 は AWS SDK も `aws` CLI も使わず、tinygodriver の `storage/s3` (7.2 節)
- AMD (amd-xdna2) は CI の NPU の照合が通れば自動で `main` に進める。Intel と、Apple が決まった後の Apple は `promote` を待つ
- 書き戻しと写しの先はチームの S3 や NFS の層。どの層を使うかは組織のファイル、利用者のファイル、環境変数の 3 段で決める (7.4 節)
- HF の名前空間は組織 `hwnpu` (https://huggingface.co/hwnpu、2026-11-12 に取得)。1 モデル 1 リポジトリ (`hwnpu/<モデル名>`)
- 置き場所は OS のローミングしないキャッシュのフォルダの下の 1 つの根だけ (7.3 節)
- gated のモデルとライセンスの確認のあるモデルは、既定では `fill` で写さない。組織のファイルで層ごとに `fill_gated: true` (7.1 節)
- 共有の層には、手元で作ったものの書き戻し (`write`) のほかに、読んだものをそのまま置く写し (`fill`、フォワードプロキシの共有キャッシュ) も持てる (7.1 節)
- 配布物がなければ今の仕組みで手元で作る。そのときは元データと変換後のデータを両方残し、パラメータを変えたバリエーションを元を落とし直さずに作れるようにする (8.1、8.2 節)

## 13. 未決

なし (2026-11-12)。
