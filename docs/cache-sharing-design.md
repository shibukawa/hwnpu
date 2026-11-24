# 変換済みのモデルの配布と、キャッシュの共有 (設計の記録)

> 読み物としての説明は [cache-sharing.md](cache-sharing.md) (English) と [cache-sharing.ja.md](cache-sharing.ja.md)。この文書は設計の経緯と細部の記録で、他の文書やコードの「7 節」などの参照はここを指す。

作成: 2026-11-12。状態: P0、P1 の層とフォルダと gc、P2 の S3、AMD のチャットのモデルと初回に作る画像のモデルの書き出しの共有を実装 (0 節、[docs/design/apple-coreai.md](design/apple-coreai.md))。2026-11-15 に取得元を環境変数の URL (フォルダ、WebDAV と HTTP、S3) と固定の並び (`original/` と `bonsaiai-lock.json`、`converted/.../vN`) に作り直した (7 節、[docs/design/distribution.md](design/distribution.md))。ほかは設計案。docs/prebuilt.md (Hugging Face で配る計画) と docs/model-table.md の 5 節 (取得元)、11 節 (運用) を、1 つの仕組みにまとめ直したもの。食い違うところはこの文書を正とする。

## 0. 実装したもの (2026-11-12)

- 根を 1 つに (`cache.Root`、`BONSAIAI_CACHE`)。AMD の書き出しは `entries/amd-xdna2/` に作り、`~/models` は読むだけ。IRON の JIT のキャッシュは `iron/` (`~/.npu/cache` を最初の書き出しの前に移し、リンクを残す)。組み込みの IRON の道具は `tools/`。まだ移していないもの: `models/` (チェックポイントと Core ML、OpenVINO のプログラム) を `src/` と `entries/` に分けること
- `.done` を相対パスに、ダウンロード・変換・書き出しのロック、HF の続きのダウンロードの `If-Range`
- 取得元 (7 節、2026-11-15 に作り直し): `BONSAIAI_MODEL_SOURCE`、`_1`、... の URL (`file://`、`http(s)://` の WebDAV と読むだけの HTTP、`s3://`)、`?writeback=true`、`BONSAIAI_MODEL_HUGGING_FACE`、`.env` (7.4 節)。元のモデルは `original/OWNER/NAME/` と `bonsaiai-lock.json`、変換後は `converted/MODEL/TARGET/EXPORT/VERSION/`。定義の `targets` (版) と `develop`。`bonsaiai-dev sources|ls|push|update|verify|flush|gc|migrate`
- 元のチェックポイントのフォワードプロキシ (7.1 節): lock のコミットのファイルを取得元から、なければ HF から同じコミットで取り、書き戻す取得元に置く
- エントリ (6 節): `entry.json` と、AMD のチャットのモデルの書き出しの共有 (`chat.SharedExport`)。鍵は指紋 (モデル、コミット、定義の sha256、レシピの args (model-recipes.yaml、2026-11-23 までは models.yaml の amd_export。中身は同じ)、組み込みの道具のうちその書き出しが読むものの hash: `amdtools.LMHash`)。読むものは `iron_export_lm.py` と共通の 3 つ、アーキ (args の最初の語) ごとの表 `lmArch` (MODELS のモジュールと、その枝が install するもの) と旗 (`--verify`、`--q4-codes`) のモジュールから import を辿ったもの (`iron_export_lm.py` へ戻る import は辿らない)、それと全部のカーネル (部品から名前を作って読むものがあるので)。表にないアーキは道具の全部 (`amdtools.Hash`)。MODELS の全部のモデルが表にあること、`iron_export_lm.py` が import するモジュールがどこかの根にあることは `TestLMArchCoversModels` が見る (2026-11-17)。版 (`version`、5 節の lock) はまだない
- bonsaiai-image detect、pose、segment が初回に作る AMD の書き出しも層に (鍵は定義の sha256、道具の hash、Go の側の手で上げる版: `chat.PinnedRecipe`)
- bonsaiai-image generate の FLUX.2 klein も、AMD では初回に層から取るか IRON で作って書き戻す (`kind: iron-flux2klein`、鍵は `chat.PinnedRecipe` と `--size`。2026-11-16、[docs/design/image-amd.md](design/image-amd.md))。層にそれがなければ、手で作って `bonsaiai-dev sources import` で置いた同じ名前の書き出し (`kind: iron-import`、レシピはモデル、対象、名前だけ: `chat.ImportedRecipe`) を取る (`chat.SharedExportOr`)。取り込んだものを書き戻し直すことはしない
- AMD のレシピのある bonsaiai-chat のモデルは、レシピの書き出し (動かすメモリのもの: gpt-oss-20b の 16 GB は `-p16`) が手元になければ、取得元から取るのを、別の名前の手で作ったもの (`NAME.amd-npu2-q4`、`NAME.amd-npu2`) より先にする。手で作ったものは取得元にないとき (答えないときも) に、作る代わりに動かす (`chat.AMDExportOf`、2026-11-25)。レシピと同じ名前のものが `~/models` に entry.json なしであれば、ピンより先にそれを動かし、そう書く (`move it away to take the pin`)
- `bonsaiai-dev models fetch MODEL/amd-xdna2` は、レシピの形のあとに、models.yaml がピンした書き出しで手元にないもの (手で作った DA3-SMALL の `da3-small-mv3.amd-npu2`) も名前で取る (`--memory` のないとき)
- ほかの手で作る AMD の書き出し (bonsaiai-image の画像生成、深さ、顔、ラベル、AMD のレシピのない bonsaiai-chat のモデル) も、手元になければ取得元の同じ名前の `iron-import` を取って手元の `entries/amd-xdna2` に (`chat.AMDNamed`、2026-11-17、[docs/design/distribution.md](design/distribution.md))。2 回目からはその写しで、取得元には聞かない。`entries/amd-xdna2` にある手で作ったもの (entry.json がない) は、初めて使ったときに記述して書き戻す取得元に置く (`chat.ShareMadeByHand`。同じ名前があればその取得元には置かない)。`~/models` のものは書かない (`bonsaiai-dev sources import`)。物体検出、姿勢、切り抜き (初回に書き出す道具のもの: `iron_onnxcnn.py`、`iron_efficientsam.py`) の手で作ったものも同じく初めて使ったときに置く。記述は `bonsaiai-dev sources import` と同じくその道具の種類と鍵 (`ExportKind`、`PinnedRecipe`) で、ほかの機械の初回がそのまま取る (`chat.ShareMadeHere`、2026-11-18、[docs/design/vision-models.md](design/vision-models.md))
- 取得は並べる: 変換結果のファイルを 4 本ずつ、大きいものから (`store.FetchStreams`)、元のファイルも 4 本ずつ (`chat.FetchFiles`: 画像、音、チャットのモデルの分割した重み)。この機械 (Wi-Fi) と UGREEN の NAS で 1 本 5 MB/s、4 本 9 MB/s
- `bonsaiai-dev sources gc N` は取得元のエントリのフォルダを選んで消す (`--apply` で実際に。`--target amd-xdna2` でその対象のものだけに絞る: 同じモデルの Core AI の書き出しを残す)。書き戻しが切れて残したもの (`original/` も) は `bonsaiai cache clean` で消す (7.1 節)。途中で止まった S3 のマルチパートも `bonsaiai cache clean` が中止する (始まって 1 時間を過ぎたもの)。バケットのライフサイクルの規則があれば、それも効く
- Core AI の書き出し (2026-11-14、[docs/design/distribution.md](design/distribution.md)): どのコマンドの `NAME.coreai` も、作る前に層を探し、手元で作ったら書ける層へ書き戻す (`kind: coreai`、鍵はモデルのファイルの sha256 かコミット、書き出しの名前、`ailower.FormatVersion`)。ホストが引く表 (`tables.safetensors`) も (`kind: tables`)。変換済みを動かすだけの機械はチェックポイントを落とさない。path (NAS) と S3 (RustFS) で確かめた
- 次: Anima などの手で作る画像の書き出し (HF の配布物で)、Core ML と OpenVINO のプログラムのエントリ化 (8 節の計画と作るの分離)、`host` の scope、HF の配布物 (4、5、9 節)

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
- 前提: 世代を見分けるコードは Apple にしかない。AMD (`amd-xdna1` / `amd-xdna2`) と Intel (`intel-npu3` / `intel-npu4`) を分けて配るには、`bonsaiai.Enumerate` が世代を返すようにする
- `prebuilt.md` のサブフォルダ `amd-npu2/` は `amd-xdna2/` に揃える (書き出しの中身の IRON の `npu2` はそのまま)

## 4. 定義 (組み込みの `models.yaml`)

`prebuilt.md` 5.3 節の別ファイル (`prebuilt/models.yaml`) は持たず、モデルの定義に書く。バイナリに組み込まれるので、各リリースの BonsaiAI は自分の取る版を知っていて、HF に新しい版が上がっても古いバイナリは古い版を取り続ける (再現できる)。

```yaml
models:
  LFM3-1.7B:
    from: hf:LiquidAI/LFM3-1.7B@6c1f0e3a...     # 配るモデルはコミットの固定が必須 (40 桁)
    targets:                                    # 2026-11-15 に読むようにした (版だけ)
      amd-xdna2: {version: 2}
      coreai:    {version: 1}
      # apple は 10 節が決まるまで書かない
  FLUX.2-klein-4B:
    develop: true                               # 開発中: 取得元には vN-HASH で置く
```

- 2026-11-15 の実装: 版はモデルの直下の `targets.<対象>.version` (`prebuilt:` の下ではない。取得元が環境変数の URL になり `prebuilt.repo` は要らない)。`format` や `codes` などの欄はまだ読まない。利用者のファイルは名指した対象だけを変える

- 行に書いたパラメータ全部が、その版の中身。実行時の設定 (フラグ、`builds`、`variants`) がこれと合えば配布物を使い、合わなければ手元で変換する (`-explain` に理由)
- `like: 行の名前` で、ほかの行の値を引き継いで違うところだけ書ける (`apple-m1: {like: apple, version: 2, prefill: off}`)
- 利用者のファイルで版やリポジトリを差し替えられる (社内で作り直したもの)
- 表 (`tables.safetensors`) のように対象によらないものは `targets.any` に書く

## 5. 指紋と lock ファイル (上げ忘れの検査)

`bonsaiai-dev prebuilt fingerprint` が、モデル × 対象の行ごとに指紋 (sha256) を計算する。入力:

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
  "made_by": {"bonsaiai": "<コミット>", "tools": "<amd-tools の hash>", "iron": "v1.4.4"},
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

## 7. 取得元 (透過型のキャッシュ)

2026-11-15 に作り直した ([docs/design/distribution.md](design/distribution.md))。それまでの「層」(定義の `sources`、`search`、`cache.layers`、`BONSAIAI_CACHE_LAYERS`、内容で番地を決める `blobs/sha256`) は、環境変数の URL の並びと、人が見て分かる固定の並びに置き換えた。

```
0. この機械のキャッシュ  cache.Root (BONSAIAI_CACHE)       いつも書き戻す・その場で使う
1. BONSAIAI_MODEL_SOURCE, _1, _2 ...                          URL ごとの設定 (?writeback=true)
2. アプリ同梱のモデル    アプリの資源フォルダの models/     読むだけ・その場で使う
3. Hugging Face          BONSAIAI_MODEL_HUGGING_FACE (既定 true)
→ どこにもなければ手元で作る

書き: 0 番へは必ず、?writeback=true の取得元へは裏で、手元で作ったものと後ろの層から読んだものを置く (失敗しても続ける)
```

2026-11-15 の 2 つ目の変更 ([docs/design/distribution.md](design/distribution.md)) で、手元のキャッシュと同梱のモデルも同じ仕組みの層にした。どの層も中は同じ並び (下) で、手元のキャッシュも `original/OWNER/NAME/` に `bonsaiai-lock.json` を持つ。どの層から来たファイルも、使うのは手元の `original/OWNER/NAME/FILE` (`chat.RepoDir`)。同梱のモデルはハードリンク、別のファイルシステムならシンボリックリンクで、コピーしない。

取得元は URL で書く。`BONSAIAI_MODEL_SOURCE` で始まる環境変数を全部集め、番号の順に並べる (番号のない `BONSAIAI_MODEL_SOURCE` が最初、抜けた番号は気にしない)。エラーになった取得元は飛ばして次へ:

| URL | 取得元 | 書き戻し |
|---|---|---|
| `file:///mnt/llm/bonsaiai` | フォルダ (マウントした NAS、手元のディスク)。Windows は `file:///C:/models`、`file://server/share` | 普通のファイルの操作 |
| `https://USER:PASSWORD@nas:5006/bonsaiai`、`http://...` | WebDAV (7.5 節)。読むのは GET と HEAD だけなので、書き戻さないならどの HTTP サーバー (nginx の静的配信) でもよい | WebDAV (PUT、MOVE、MKCOL) |
| `s3://BUCKET/PREFIX?endpoint=http://host:9100&region=R&profile=P&path_style=true&no_sign=true` | S3 (7.2 節) | PUT とマルチパート |

- `?writeback=true` (どの種類にも): 書き戻す。手元で変換したもの (8 節の 3) と、後ろの取得元や HF から読んだもの (フォワードプロキシ) の両方。gated のリポジトリとライセンスの確認のあるモデルも置く (読む機械でのライセンスの確認は手元で今どおり)
- `BONSAIAI_MODEL_HUGGING_FACE`: 最後の番兵。`true` (既定) で HF から取る。`false` でネットに出ない (HF への HEAD もしない)。取得元で解決できないものはエラー
- 取得元の中の並び:

```
original/OWNER/NAME/FILE...               HF と同じ並びのファイル。そのままモデルのフォルダとして使える
original/OWNER/NAME/bonsaiai-lock.json    コミットと、ファイルごとの sha256 と大きさ
converted/MODEL/TARGET/EXPORT/VERSION/... 変換したもの。entry.json を最後に書く (あれば揃っている)
```

- `bonsaiai-lock.json` (`store.Lock`): `{"format": 1, "repo": "OWNER/NAME", "commit": "<40 桁>", "files": {"config.json": {"sha256": "...", "size": 123}}, "updated": "..."}`。名指しのないファイルは読まない。コミットの違うファイルは書き戻さない (`ErrOtherCommit`、キューの記録に残す)
- コミットの決め方: 定義がコミットを固定していればそれ。`main` を追うモデルは 1 番以降で最初の lock のコミット (どの機械も同じファイルで動く)、なければ手元のキャッシュの lock のコミット (手元にあるものは HF に聞き直さない)、それもなければ HF の `main` (`X-Repo-Commit`)。手元のキャッシュは決まったコミットに従う: 取得元の lock が別のコミットに移ると、手元の lock も移し、ファイルは新しい lock の sha256 と同じなら残し、違えば取り直す (共有の取得元は `ErrOtherCommit` のまま触らない)。取得元にないファイルを HF から取るときも、lock のコミットのものを取る (`resolve/<コミット>/`)。`main` 以外のブランチやタグは、HF に聞いたコミットと lock を照らし、聞けなければ (HF に出られるときは) 取得元を使わない
- lock を新しいコミットに移すのは明示的なコマンドだけ: `bonsaiai-dev sources update [MODEL...]` (書き戻す取得元がなければ手元のキャッシュ、`--source 0`)。HF の今のコミットを聞き、sha256 の変わったファイルだけ落とし直して置き、最後に lock を書く (LFS のファイルは `X-Linked-Etag` で比べ、変わらなければ落とさない)
- `VERSION` (`store.VersionDir`): 定義の `targets.<対象>.version` が N のモデルは `vN`。`develop: true` なら `vN-<レシピの指紋の 8 桁>`。版のないモデル (今は全部) は `v0-<8 桁>`
- 当たりの条件: `vN` は版と、道具の hash を除いたレシピ (元のコミット、sha256、引数) が同じもの。道具の hash が違うだけの 2 台が同じフォルダを書き換え合わないように。出力が変わる直しをしたら版を上げる (3 節)。`vN-HASH` と `v0-HASH` は指紋がそのまま同じもの
- `EXPORT` は手元の書き出しの名前 (`Qwen3.5-0.8B.amd-npu2-q6`、`LFM3-1.7B.coreai`、`tables.safetensors`)。対象のない表は `TARGET` を `any` に
- ファイル単位の重複除去 (同じ `weight.bin` を 1 つに) はやめた。並びが分かりやすいこと、普通の HTTP サーバーで配れることを取った
- 落としたものは lock や entry.json の sha256 と照らし、違えばその取得元を飛ばして次へ (この機械では大きなダウンロードにビットの反転があった)
- 信頼: 書ける取得元は書ける人を信頼する (xclbin や ELF は NPU で動く)
- `Prune` は手元だけを掃除する。取得元の掃除は明示的に: `bonsaiai-dev sources gc N` (エントリのフォルダを entry.json から消す) と `bonsaiai cache clean` (書き戻しが切れて残したもの、7.1 節)
- 前の並び (`blobs/`、`upstream/`、`entries/`) のフォルダは `bonsaiai-dev models migrate N [--remove]` でハードリンクで移した (NAS を移し終えて 2026-11-22 に外した)。元のモデルはファイルの多いコミットを取る。変換後は名前の分かるもの (Core AI、表、AMD の言語モデルと初回の書き出し) だけ

`bonsaiai-dev sources` で、今効いている層の並び (0 が手元のキャッシュ)、書き戻すか、HF を読むか、読んだ `.env` を出す。`ls [N] [--original]` で中身。

同梱のモデル: インストーラがアプリの資源フォルダ (`github.com/shibukawa/configdir` v1.1.0 の `QueryResourceFolders`: 実行ファイルの隣、`X.app/Contents/Resources`、`PREFIX/share/APP`、`$XDG_DATA_DIRS/APP`) の `models/` に、取得元と同じ並びで置く。環境変数ではなくアプリのコードで決める: BonsaiAI のコマンドは BonsaiAI の資源フォルダ (`share/bonsaiai/models` など) を読み、BonsaiAI を組み込むアプリは `modelstore.Options.Bundle` で自分のものを渡す (7.6 節)。

### 7.1 書き戻し

- Hugging Face から落とす元のファイルは、落としながら書き戻す取得元にも書く (`store.Tee`、2026-11-16)。一時の名前 (`.put-`) に送り、落とし終えて大きさと sha256 が合ったときだけ正式な名前に移して lock に足す。対応はフォルダと WebDAV (S3 はキューで)。取得元が 64 MiB (`store.TeeBuffer`) 遅れたとき、失敗したとき、`.part` から続けたとき、lock が別のコミットのときは tee から外れ、落とし終えた後にキューで置く。ダウンロードは止めない。Qwen3-0.6B (M1、Wi-Fi、UGREEN の NAS) で答えた後の待ちが 35 秒から 6 秒に。Wi-Fi では受信と送信が電波を分け合い、ダウンロードは 40 秒 (curl だけ) から 54 秒になった。有線なら遅れないはず
- 手元で変換するものは、変換しながら置く (`store.StartEarly`、2026-11-16。AMD と Core AI の書き出しは詰めて置くので除く: 7.10。2026-11-18 に Core AI も詰めるようにしたので、今これで置くものはない)。書き戻す取得元で見つからなかったとき、その書き出しのフォルダを 2 秒ごとに見て、見始めてから書かれ、3 秒触られていないファイルから 4 本まで並べて置く (`.` で始まるもの、`.tmp`、`.part` は除く)。変換器には手を入れない。最後の `PutEntry` は、置いた後に変わったファイルと後から書かれたファイルだけを置き、置いたが entry にないもの (変換が消したもの) を消してから `entry.json` を書く。取得元のその場所に `entry.json` がすでにあれば、変換しながらは置かない (その entry のファイルを途中で書き換えないため)。変換が途中で止まれば、置いた分は `entry.json` のないまま残り、印が古くなってから `bonsaiai cache clean` が消す。Qwen3-1.7B (書き出し 1.75 GB と表 0.62 GB) で、答えた後の待ちが 35 秒から 12 秒に (変換中に 129 ファイル)
- ほかの取得元から読んだものは、手元で揃ってから裏で置く。コマンドは終わる前に、キューの仕事 (このプロセスのものと、前のプロセスが残したもの) が済むまで待つ (`store.Wait`、2026-11-16。それだけを待つ `bonsaiai-dev models flush` は、どのコマンドも待つので 2026-11-22 に外した)。書き戻すのは自分で変換する人だけで、配られたものを読むだけの機械はキューが空なのですぐ終わる。待つ間の Ctrl-C と、プロセスが落ちたときは `work/queue` に残し、次の起動で続ける。entry.json と lock が最後なので、途中で切れても壊れたものは見えない
- 後ろの取得元で当たったものは、それより前の書き戻す取得元にも置く (前が速い NAS、後ろが遠い S3 のとき)
- 置けなかったとき (書けない、容量がない) は `work/queue.log` に残して続ける。書き戻しの失敗で実行を止めない
- 書き戻しは 4 本まで同時に (`store.PutStreams`): キューの仕事どうしも、1 つの変換結果のファイルどうしも (`entry.json` は最後)。元のファイルはリポジトリの lock を取る前に置き、lock は読んで足して書くときだけ取るので、同じリポジトリのファイルも並ぶ。lock が別のコミットなら置かない。WebDAV のフォルダ作り (MKCOL) はプロセスで 1 本ずつ (同時の MKCOL は 423 Locked で断られる)。Qwen3-0.6B (元 1.5 GB、変換結果 0.93 GB) で、答えた後の待ちが 55 秒から 35 秒に (M1 と UGREEN の NAS、合わせて 50 MB/s ほどで頭打ち。2026-11-16)
- lock は読んで足して書く。2 台が同時に同じリポジトリを足すと片方の名指しが消えることがあるが、消えたファイルは次に読む機械が HF から取って足し直す
- フォルダは揃っているか、揃っていないかの 2 つだけ: 変換結果のフォルダ (`converted/MODEL/TARGET/EXPORT/VERSION`) は `entry.json` があれば揃っていて、なければ中身は全部使わない。リポジトリのフォルダ (`original/OWNER/NAME`) は `bonsaiai-lock.json` が名指すファイルだけを使う。1 つのファイルは同じフォルダの一時の名前 (`.put-`) に書いて名前を変える (S3 は PUT が終わるまで見えない)
- 書いている間は、そのフォルダに印 `.writing-ID` (機械、プロセス、始めた時刻、最後に書いた時刻) を置き、5 分ごとに書き直し (`store.WritingRefresh`)、終わったら消す (`store.MarkWriting`: `PutEntry`、`PutOriginal`、`bonsaiai-dev sources update`。落としながらの tee は始めてから lock に足すまで、変換しながら置くのは始めてから `PutEntry` が引き継ぐまで: 大きなファイルの tee や、IRON の長いコンパイルの間に置いたものは 1 時間以上触られないことがある)
- `bonsaiai cache clean` (`store.PlanClean`、`modelstore.Store.PlanClean`): 書き戻す取得元から、書き戻しが切れて残したものを消す。entry.json のない変換結果のフォルダは丸ごと (空になったフォルダも)、lock が名指さない元のファイル (lock がなければ全部)、`.put-` の一時ファイル、古い印。印か、ファイルのどれかが最近 1 時間に変わったフォルダ (`store.WritingStale`) は書いている途中とみなして残す。ファイルの時刻も見るのは、印を置かない前の BonsaiAI が書いているフォルダを消さないため (2026-11-16、UGREEN の NAS で Mac が LFM3-1.7B の重みを置いている最中に、印のない `.put-` を消しかけた。NAS が 423 Locked で断った)。消す直前にもフォルダを見直し、その間に書き始めたもの、揃ったものは残す。S3 では、`converted/` と `original/` の下で始まって 1 時間 (`store.WritingStale`) を過ぎたマルチパートを `ListMultipartUploads` で探し、`AbortMultipartUpload` で中止する (部分の大きさは一覧に出ないので、表の大きさは `?`)。`converted/` と `original/` の外は触らない。書き戻しが切れて残したものを消すのはこれだけ (`bonsaiai-dev sources gc --orphans` は重なるので 2026-11-22 に外した)

別案だったもの: HF の API (`/resolve/`) を真似るキャッシュのサーバー (`bonsaiai-dev models serve`) を 1 台に置き、`HF_ENDPOINT` で向ける。サーバーの運用が要るので入れていない。`original/` は HF と同じ並びなので、要れば後で足せる。

### 7.2 S3 のクライアント

AWS SDK (aws-sdk-go-v2) は重いので使わない。`aws` CLI を呼ぶ今の形 (model-table 5 節、`internal/chat/sources.go` の `fetchS3`) もやめ、`github.com/shibukawa/tinygodriver` の `storage/s3` (署名は `cloud/aws`) にする。元のチェックポイントの取得元 (`type: s3`) と、キャッシュの層の両方をこれ 1 つで読む。

- 依存: 標準ライブラリと tinygodriver の中の小さなパッケージだけ (`cloud/aws`、`internal/cloudhttp`、`internal/hmacsha256`、`encoding/simdhex`)。モジュールは 1 つで go.mod の require は多いが、リンクされるのは使うパッケージだけ
- 使う操作: `Get` / `GetRange` (続きから落とす)、`Head` (エントリの有無)、`Put`、マルチパートの 4 つ、`List` (gc と ls)、`ListMultipartUploads` (clean)
- 設定の欄は `endpoint` (`WithEndpoint`)、`region` (`WithRegion`)、`path_style` (`WithPathStyle`。既定は amazonaws.com なら仮想ホスト形式、それ以外はパス形式なので、RustFS で `~/.aws/config` に `addressing_style` を書く手順は要らなくなる)。大きなファイルは `WithUnsignedPayload` で、本体を読み込まずに送る
- tinygodriver が持たないものを BonsaiAI の側で足す:
  - 認証: tinygodriver は静的な値と環境変数 (`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_SESSION_TOKEN`、`AWS_REGION`、`AWS_ENDPOINT_URL_S3`) だけ。`profile` は、BonsaiAI が `~/.aws/credentials` と `~/.aws/config` のその節の静的な鍵 (`aws_access_key_id`、`aws_secret_access_key`、`aws_session_token`、`region`、`endpoint_url`) だけを読んで渡す。SSO とインスタンスのメタデータは持たない。SSO の利用者は `eval "$(aws configure export-credentials --profile X --format env)"` で環境変数にしてから使う (docs に書く)
  - マルチパート: `Put` は 1 回の要求 (上限 5 GiB) で、分割は自分でする。64 MiB を超えるファイルは 64 MiB の部分に分けて上げ (IRON のバンクには数 GB のものがある)、失敗したら必ず `AbortMultipartUpload`。途中で止まった分 (プロセスが落ちて Abort できなかったもの) は、`bonsaiai cache clean` が `ListMultipartUploads` (tinygodriver v1.3.10 から) で見つけて中止する。始まって 1 時間を過ぎたものだけで、ほかの機械が上げている途中のものは残す。バケットのライフサイクルの規則 (未完了のマルチパートを N 日で中止) を足しておけば、clean を走らせなくても消える
  - 再試行: 5xx、接続の切断、`SlowDown` は指数的に待って 3 回まで。ダウンロードは `GetRange` で続きから
  - `no_sign` (署名なしの公開バケット): tinygodriver で鍵なしの要求が送れるかを確かめる。送れなければ、公開バケットは `https://` の取得元 (HTTP の GET) として読む
- 確かめたこと (2026-11-13、RustFS 1.0.5 を手元で): 部分に分けた送り (5 MiB の部分で 12 MiB)、空のファイル (tinygodriver v1.3.8 は長さ 0 の本体を chunked で送り 411 になった。v1.3.9 で直った)、`GetRange` の続き、一覧、削除、パス形式 (`internal/store` の `TestS3Server`、`BONSAIAI_TEST_S3_ENDPOINT` を与えたときだけ)。Qwen3.5-0.8B の書き出し (1.57 GB) を `push` (10 秒) し、空の根で IRON なしに層から取って NPU で答えた (準備 4.4 秒)
- Cloudflare R2 で (2026-11-13): `region: auto` で署名が通り、`TestS3Server` が通った (`BONSAIAI_TEST_S3_BUCKET` に既にあるバケット、`BONSAIAI_TEST_S3_REGION=auto`)。Qwen3.5-0.8B の書き出しを `push` (1.57 GB、119 秒。この機械の上りは無線で約 13 MB/s)、空の根と IRON なしの bonsaiai-chat が層から取って答えた (準備 54 秒、取るのは約 30 MB/s)。R2 の無料枠は保存 10 GB/月なので、試しには小さな書き出しだけを置く
- 確かめること: AWS の S3 で仮想ホスト形式と、ほかのリージョンへのリダイレクト
- MinIO は対象にしない (保守が終わった)。S3 互換のサーバーの確認は RustFS で

### 7.3 手元の置き場所

BonsaiAI が書くものは全部、OS のローミングしないキャッシュのフォルダの下の 1 つの根に収める。ほかの場所には書かない。

| OS | 根 | 備考 |
|---|---|---|
| macOS | `~/Library/Caches/bonsaiai` | iCloud に同期されない。Time Machine も既定で除く |
| Linux | `$XDG_CACHE_HOME/bonsaiai` (`~/.cache/bonsaiai`) | |
| Windows | `%LOCALAPPDATA%\bonsaiai` | `Roaming` (`%APPDATA%`) には置かない。ユーザーの `models.yaml` も同じ根にあるので、掃除は下のサブフォルダだけを触る |

どれも `os.UserCacheDir()` の下で、`BONSAIAI_CACHE` (コードからは `modelstore.Options.Root`、`cache.SetRoot`) で根ごと移せる (外付けのディスク)。`go test` は `BONSAIAI_CACHE` がなければプロセスごとの一時フォルダを根にする (試験がこの機械のキャッシュに書かない)。根の下の並び:

```
bonsaiai/
  original/OWNER/NAME/FILE...        元のモデル (層 0)。取得元と同じ並び、bonsaiai-lock.json 付き (大きさの上限と LRU は 7.7)
  models/NAME/                       変換したプログラム (次の PR で converted/ に移す)
  entries/<target>/<name>/           AMD の書き出し (同上)
  KEY.mlmodelc|.blob|.json           機械ごとのもの (Core ML のコンパイル、Level Zero、placement)
  work/                              作業中 (.part、変換と書き出しの途中、queue)。rename と reflink が効くよう同じファイルシステムに置く
  tools/amd-tools-<hash>/            組み込みの IRON の道具
  iron/                              IRON の JIT のキャッシュ (NPU_CACHE_HOME)
  prompts/  history/  locks/         local の scope
```

前の BonsaiAI が `models/NAME/` に置いた元のファイルは、読まれたときに `original/OWNER/NAME/` へ移す (rename)。sha256 を、定義の固定、取得元の lock、HF の `X-Linked-Etag` のどれかと照らし、合えば移して手元の lock に足し、違えば取り直す。sha256 の分からない小さなファイル (LFS でない) は HF に出られるなら取り直し、出られないならそのまま使う。

今その外に書いているものを移す:

- AMD の書き出し (`~/models/NAME.amd-npu2*`): `entries/amd-xdna2/...` へ。`~/models` は今は読みだけの古い層として探す (書かない)。次の PR で `converted/` に移して読まなくなる (13 節)
- IRON の JIT のキャッシュ (`~/.npu/cache`、この機械で 52 GB): `RunIRON` が `NPU_CACHE_HOME=<根>/iron` を渡す (mlir-aie の `python/utils/compile` が読む)
- 作業中のものを `/tmp` に置かない (Linux では RAM の tmpfs のことがあり、GB 級の書き出しで遅くなる)
- Apple の e5 バンドルのキャッシュ (`~/Library/Caches/<プロセス>/com.apple.e5rt.e5bundlecache`) は OS が持つもので、すでにキャッシュのフォルダの下。場所は変えない
- ユーザーの設定 (`models.yaml`、`licenses.json`) は今の設定のフォルダのまま (キャッシュではない)

### 7.4 組織と環境ごとの設定: `.env`

取得元は環境変数で決める。BonsaiAI の各コマンドは起動したときに `.env.local` と `.env` を読み (`internal/envfile`)、環境にない変数だけを足す。前にあるものが勝つ:

1. プロセスの環境変数
2. カレントディレクトリの `.env.local`、`.env`
3. ユーザーの設定のフォルダ (`~/.config/bonsaiai/`、Windows は `%LOCALAPPDATA%\bonsaiai\`) の `.env.local`、`.env`
4. 機械の設定 (Linux `/etc/bonsaiai/`、macOS `/Library/Application Support/bonsaiai/`、Windows `%ProgramData%\bonsaiai\`) の `.env.local`、`.env`: 組織の管理者が配る

- 読むのは `BONSAIAI_`、`HF_`、`AWS_` で始まる名前だけ (ほかのプロジェクトの `.env` を拾っても困らないように)。テスト (`go test`) では読まない
- 書き方: 1 行に `NAME=VALUE`、`export ` を前に付けてもよい、`#` から後ろは注釈、`'...'` はそのまま、`"..."` は `\n`、`\"` が使える
- 秘密 (WebDAV のパスワード、S3 の鍵) は `.env.local` に (git に入れない慣例)。ログと `bonsaiai-dev sources` には URL の利用者とパスワードを出さない

| 変数 | 働き |
|---|---|
| `BONSAIAI_MODEL_SOURCE`、`BONSAIAI_MODEL_SOURCE_1`、... | 取得元の URL (7 節) |
| `BONSAIAI_MODEL_HUGGING_FACE` | `false` でネットに出ない。既定は `true` |
| `BONSAIAI_CACHE` | 手元の根 (7.3 節) |
| `BONSAIAI_WEBDAV_USER`、`BONSAIAI_WEBDAV_PASSWORD` | URL にも `~/.netrc` にもないときの WebDAV の利用者 (7.5 節) |
| `AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_PROFILE` | S3 の鍵 (7.2 節) |
| `HF_TOKEN` | gated のリポジトリ。CI 以外ではデバイスフローのサインイン ([docs/design/distribution.md](design/distribution.md)) を勧める |
| `BONSAIAI_MODELS`、`BONSAIAI_ACCEPT_LICENSE`、`BONSAIAI_BACKEND`、`BONSAIAI_DEVICE` | 自分の定義のファイル、ライセンスの確認、機械ごとの NPU の選び方 |

なくしたもの: 定義のファイルの `sources`、`search`、`cache` (書いてあるファイルは、移し方を書いたエラーで止める)、`from: SOURCE:NAME` と `-use SOURCE:NAME` (自分のモデルは取得元の `original/OWNER/NAME/` に lock と一緒に置き、`from: hf:OWNER/NAME` で名指す)、`BONSAIAI_CACHE_LAYERS`、`BONSAIAI_CACHE_WRITE` (`?writeback=true` を外す)。設計だけだった `BONSAIAI_OFFLINE` (`BONSAIAI_MODEL_HUGGING_FACE=false`) と `BONSAIAI_PREBUILT` (取得元にあるかないかで決まる) も作らない。

### 7.5 WebDAV

`internal/store/webdav.go`。標準ライブラリの `net/http` と `encoding/xml` だけで書いた (依存なし。テストだけが `golang.org/x/net/webdav` のサーバーを使う)。Synology (5005 番が http、5006 番が https)、Nextcloud (`https://HOST/remote.php/dav/files/USER/...`)、nginx の `dav_methods` などを想定。

- 読み: `GET` (`Range` で途中から続ける。範囲を無視したサーバーには頭から)。有無と大きさは、書き戻す取得元では `PROPFIND` の深さ 0 (`HEAD` の `Content-Length` を省くサーバーや、フォルダにも 200 を返すサーバーがあるため)、読むだけの取得元では `HEAD`
- 書き: 同じフォルダの一時の名前 (`.put-<乱数>`) に `PUT` してから `MOVE` (`Overwrite: T`) で置く。読む側が書きかけを見ない。親のフォルダがなければ (409、404) 根から `MKCOL` して送り直す。大きなファイルも 1 回の `PUT` (サーバーの上限に注意: nginx の `client_max_body_size`、IIS の `maxAllowedContentLength`)
- 一覧: `PROPFIND` の深さ 1 で辿る (深さ無限は切ってあるサーバーが多い)。`.put-` の書きかけは出さない。`gc` と `ls` が使う
- 認証: Basic。利用者とパスワードは URL のもの、なければ `~/.netrc` (`NETRC`、Windows は `_netrc`) のそのホストの `machine`、なければ `BONSAIAI_WEBDAV_USER` と `BONSAIAI_WEBDAV_PASSWORD`。401 のエラーにはその並びを書く
- 再試行: 5xx (501 を除く)、429、接続の切断は S3 と同じく 3 回まで
- 確かめたこと: `TestWebDAV` (3 MB の読み書き、途中からの続き、空のファイル、フォルダは有無で偽、一覧、削除、鍵の外れ)、`TestWebDAVSource` (エントリと lock を書き戻し、同じサーバーを読むだけの取得元として読む、読むだけへの書き込みはエラー)、`TestWebDAVSignIn` (netrc、パスワードの違い、URL のパスワードを表示しない)。Synology の実機は未確認
- 自己署名の証明書 (NAS のもの。ホスト名を持たないものは、OS の信頼に足しても Go の検証が通らない): URL に `?tls_sha256=<証明書の SHA-256>` を足すと、その証明書だけを信じる (ホスト名は見ない)。値は `openssl s_client -connect HOST:PORT </dev/null | openssl x509 -noout -fingerprint -sha256`。手早く試すなら `?insecure=true` (証明書もホスト名も見ない。同じ LAN の NAS でだけ)
- UGREEN の NAS で書き戻しと読み戻し (2026-11-16): `?writeback=true&insecure=true` で Qwen3-0.6B を空の根から。NAS になかった元のファイル (config、tokenizer、model.safetensors 1.5 GB、lock) と、この Mac で作った `converted/Qwen3-0.6B/coreai/...coreai` (0.62 GB) と `any/tables.safetensors` (0.31 GB) が置かれた (キューを待つ間を含めて 71 秒)。次に空の根、`BONSAIAI_MODEL_HUGGING_FACE=false` で、変換済みを NAS から 18 秒で取り、チェックポイントを落とさずに答えた。macOS で Go の `.local` の名前引きは 1 回 5 秒かかるので、引いた番地をプロセスの間覚える。IP (`https://192.168.1.127:5006/llm`) でもよい
- Linux (Ryzen AI 9 365、Ubuntu 26.10) で同じことを AMD の NPU で (2026-11-16): 空の取得元のフォルダ (`llm/zz-linux-test`) と空の根で Qwen3.5-0.8B を。チェックポイント 1.75 GB を HF から、IRON で書き出し (1 分 27 秒)、終わる前に元のファイル 8 つ (35 MB/s) と書き出し 1.65 GB (45 MB/s) を置いて終わった。次に空の根、IRON なし、`BONSAIAI_MODEL_HUGGING_FACE=false` で、書き出しを NAS から取って NPU で答えた。`bonsaiai cache clean` で、前に切れた書き戻しの `.put-` 4 つ (3.5 GB) を消した
- 止まった NAS (2026-11-16): つながらない (名前が引けない、接続を断られる、5 秒で答えない、証明書が信じられない) 取得元は、そのプロセスの間は設定がなかったものとして扱う。最初の 1 回で標準エラーに 1 行出し、その後の要求はすぐ失敗して次の取得元へ。再試行は時間切れ (mDNS の `.local` の名前引きはときどき時間切れになる) のときだけ 3 回まで、断られたときと名前がないときはすぐ (前は 1 ファイルごとに接続の待ち 30 秒 × 再試行が入れ子で重なった)。書き戻しはキューに残り、次の起動で続く
- UGREEN の NAS (2026-11-15): フォルダのない所への PUT と、親のない MKCOL が 409 でなく 405 を返す。PUT の前にフォルダを (プロセスで一度) 作り、405 でも作り直す
- UGREEN の NAS (2026-11-16): 自己署名の証明書なので `?tls_sha256=` が要る。`.local` の名前の解決が 1 回 5 秒かかったので、ホストの番地はプロセスで 1 回だけ引き (答えなければ引き直す)、答えの本体は必ず読み切って閉じて接続を使い回す ([docs/design/distribution.md](design/distribution.md))
- `bonsaiai-dev sources import DIR...`: 手で置いたフォルダ (`~/models`) を書き戻す取得元に置く。チェックポイントは Hugging Face のツリー (`/api/models/REPO/tree/COMMIT`) と照らし、大きさと中身 (LFS は sha256、ほかは git の blob の sha1) の同じファイルだけを `original/` に。書き出し (`NAME.amd-npu2*`) は `converted/`。名前が定義にないフォルダは `DIR=OWNER/NAME`

### 7.6 公開の API: `modelstore`

BonsaiAI をライブラリとして組み込むアプリのための入り口 (`github.com/shibukawa/bonsaiai/modelstore`)。BonsaiAI のコマンドも同じ仕組み (`chat.Fetch`、`chat.FetchWeights`、`chat.RepoDir`) を通る。

```go
cd := configdir.New("", "myapp")
s, err := modelstore.Open(modelstore.Options{
    Root:    "",       // "" = BONSAIAI_CACHE、なければユーザーのキャッシュの bonsaiai (層 0)
    Sources: nil,      // nil = BONSAIAI_MODEL_SOURCE* (.env も)、空 = なし
    Bundle:  &cd,      // 資源フォルダの models/ (層 2)。nil = なし
})
p, err := s.File(ctx, "LiquidAI/LFM3-1.7B", "config.json") // 手元のパス
w, err := s.Weights(ctx, "LiquidAI/LFM3-1.7B")             // model.safetensors (分割なら索引も)
dir := s.Dir("LiquidAI/LFM3-1.7B")                         // original/LiquidAI/LFM3-1.7B
```

層はプロセスの設定 (`Open` がプロセス全体に効く)。変換したものの取得と作成 (`Make`) は次の PR で足す。

### 7.7 手元のキャッシュの大きさ: 空きの下限と LRU、`bonsaiai cache`

手元のキャッシュ (層 0) は放っておくと増える一方なので、ダウンロードの前に場所を確かめ、使っていないものから消せるようにした (`internal/cache` の `Room`、`Scan`、`PlanPrune`。公開の入り口は `modelstore`)。

| 変数 | 既定 | 意味 |
|---|---|---|
| `BONSAIAI_CACHE_MIN_FREE` | `10GB` (`0` でなし、`5%` のようにディスクの割合も) | 根のあるディスクに残す空き。これを切るダウンロードは始めない (`ErrNoRoom`) |
| `BONSAIAI_CACHE_MAX_SIZE` | なし | キャッシュの上限。超えるダウンロードの前に、最近使っていない単位から消して収める。これがあるときだけ、空きの下限を切るときも消す |

コードからは `modelstore.Options.Limits` (`cache.SetLimits`)。`go test` では空きの下限を既定で 0 にする。

- 単位 (まとめて消すもの): `original/OWNER/NAME` (元のファイル)、`models/NAME` と `converted/MODEL/TARGET/EXPORT/vN` (変換したもの)、`models/NAME/EXPORT.coreai.prebuilt` (Core AI の prebuilt、7.9。`models/NAME` の量には数えない)、`entries/TARGET/NAME` (AMD の書き出し)、`iron/KEY` (IRON のカーネル)、根の `KEY.mlmodelc` などの同じ鍵のもの (この機械でコンパイルしたもの)。それ以外 (`prompts/`、`history/`、`locks/`、`work/`、`dev/`、道具) は数えるだけで消さない
- 単位を消すとき (`bonsaiai cache clear` / `prune`、空きを作るときの LRU)、中の Core AI の書き出し (`.aimodel`) の `main.hash` の項目も、根の外の Core AI のキャッシュ (`~/Library/Caches/coreai-cache/<OS>/<実行ファイル>/<hash>/`、すべての OS のビルドと実行ファイル名) から消す (`cache.CoreAIEntries`、2026-11-26)。項目は重みをもう一度持ち、書き出しと同じくらい大きい (FLUX.3-klein の書き出し 10.1 GB に項目 8.9 GB)。書き出しを消すと、項目を探す手がかりの `main.hash` もなくなるので、残すと消す手立てがない。落とし直した書き出しはコンパイルし直しになる (ANE のコンパイラで数分、prebuilt から数秒)。残す項目: 根に残る別の書き出しが同じ hash を持つもの、prebuilt の単位を消すとき (書き出しが項目を使う)、このプロセスが読んでいるもの (`cache.Use`)。`clear` / `prune` は消す単位の表の後に項目の量も出す。`main.mlirb` だけを空にして項目から読む形 (`BONSAIAI_DEV=coreai.drop-bytecode`) は、ダウンロードした書き出しには効かず、どのみち落とし直しになるのでやめた (apple-coreai.md)
- 空きの下限を切るとき: まず、根の外の Core AI のキャッシュ (`~/Library/Caches/coreai-cache/<OS>/bonsaiai*/<hash>/`) のうち、手元に prebuilt のある部品の項目を古い順に消す (`BONSAIAI_CACHE_MAX_SIZE` がなくても、`cache.SetRestorable`)。項目は重みをもう一度持つので書き出しと同じだけ大きく、次に読むときに prebuilt から数秒で作り直される (ANE のコンパイラの数分ではない)。このプロセスが読んだ項目と、最近 1 時間の項目は消さない。次に (上限を超えるときも)、IRON の設計ごとの組み上げ (`iron/KEY`、24 桁の 16 進。グラフの全体の ELF と aiecc の途中のファイルで、1 つ 10〜100 MB 余り、`iron/` のほとんど) を古い順に消す (`cache.IRONBuild`)。組み上げのもとの `iron/objects` (カーネルの `.o`、ソースごと) と `iron/devices` (aie.device ごとの配置とリンク済みのコア) は数十 MB で残す。それがあれば組み上げは数秒で作り直される (RTMDet-m の書き出しが、全部あれば 12 秒、objects と devices だけで 31 秒、空で 52 秒。2026-11-26 aix1)。それでも足りなければ、上限があるときだけ根の単位を古い順に消し、なければ `ErrNoRoom`。prebuilt (`EXPORT.coreai.prebuilt`、書き出しの 1〜2 %) は作り直しのもとなので先には消さず、ほかの単位と同じ LRU に並ぶ
- 最後に使った時刻: 単位のフォルダの時刻と中のファイルの時刻の新しい方。取得 (`chat.Fetch` は速い道でも)、`chat.ModelDir`、`cache.FindExport`、Core AI の prebuilt を組み立てるバックエンドが `cache.Touch` でフォルダの時刻を今にする (5 分に 1 回まで)。atime は noatime/relatime で当てにならないので使わない。IRON のカーネルは IRON が読むときに時刻が変わらないので、作った順になる
- 消さないもの: 最近 1 時間に使った単位 (ほかのプロセスが読んでいるかもしれない)、このプロセスが使った単位、書き戻しの queue に仕事がある単位 (`store` が `cache.SetBusy` で教える)、書いている途中の単位
- 消すときは、元のファイルなら `bonsaiai-lock.json`、変換したものなら `entry.json` を先に消す (読む側が途中のものを揃っているとみなさない)。空になった親 (`original/OWNER`) も消す
- 大きさはファイルの大きさの和。ハードリンク (Unix) は 1 回だけ数える (同じ sha256 を別のリポジトリと共有しているもの)。reflink で共有したブロックは区別しない
- 上限の判定のための走査はプロセスに 1 回 (この機械では IRON の 61 万ファイルで 2.7 秒)。あとは足した分を数える。空きの下限だけなら statfs (`GetDiskFreeSpaceEx`) 1 回

一般の利用者のためのコマンドとして `bonsaiai` を足した (`bonsaiai-dev models` は取得元の管理のまま)。

```
bonsaiai cache                          根、ディスクの空き、上限、種類ごとの量、層
bonsaiai cache ls [-k KIND] [-s] [PAT]  単位ごとの大きさと最後に使った時刻 (古い順、-s で大きい順)
bonsaiai cache prune --max-size 200GB   古い順に消して収める (--older-than 30d も。なしなら BONSAIAI_CACHE_MAX_SIZE)
bonsaiai cache clear Qwen3-1.7B         名前 (モデル名、OWNER/NAME、glob) の単位を消す。--kind、--all。端末では確かめる (-y で聞かない)
bonsaiai cache clean                    書き戻す取得元から、書き戻しが切れて残したものを消す (7.1 節)。-n、-y
```

`prune` と `clear` は `-n` (`--dry-run`) で消すものだけ見せる。`clear` は最近使ったものも消す (書き戻し待ちのものは残す)。

### 7.8 モデルを何 GB の機械として動かすか: `bonsaiai memory`

モデルをメモリいくつの機械でどう動かすか (どのプログラムを作るか、取ってくるか) は models.yaml の定義 (`variants`) が決め、使う人はモデルの特性を知らなくてよい。使う人が選ぶのは「この機械を何 GB の機械として扱うか」だけで、選んだものはモデル管理の設定として、どのコマンドも守る (同じモデルの形違いを何組も入れない)。

```
bonsaiai memory                         この機械のメモリと、メモリを選んだモデル
bonsaiai memory LFM3-1.7B               何 GB の機械として動くか、そのときの形 (tier none など) とメモリ
bonsaiai memory LFM3-1.7B 8             8 GB の機械として動かす (16 GB の Mac で、ほかのプログラムと一緒に)
bonsaiai memory LFM3-1.7B --forget      この機械のメモリに戻す
```

- 置き場所: 設定のフォルダ (`~/.config/bonsaiai/run.json`、`$XDG_CONFIG_HOME/bonsaiai`、Windows は `%LOCALAPPDATA%\bonsaiai`)。ユーザーの models.yaml の隣の別のファイルで、`{"models": {"LFM3-1.7B": {"memory": 8}}}`。キャッシュには置かない: `prune` で消えず、同じキャッシュ (`BONSAIAI_CACHE`) を読むほかの機械には移らない
- 守るところ: モデルの名前の解決 (`chat.Resolve`、`ResolveFor`、ファミリーや用途で選んだモデル)。bonsaiai-chat、bonsaiai-image、bonsaiai-audio、`modelstore` を使うアプリが同じ。今メモリごとの形を持つのはチャットのモデル (LFM3-1.7B、Qwen3.5-0.8B、Qwen3-1.7B) だけだが、ほかのモデルに `variants` を足せばそのまま効く
- 優先: その実行のメモリのタグ (`lfm3:8g`) または `bonsaiai-chat --memory`、選んだメモリ、`BONSAIAI_MEMORY`、機械のメモリ。`bonsaiai-chat --memory 8` で用意したモデルは、そのメモリが選ばれたものとして残る
- ファミリーや用途の名前 (`lfm3`) がどのモデルを選ぶかは、この機械のメモリで決まる。選ばれたモデルにメモリが選んであれば、その形で動く
- 公開の API: `modelstore.Store` の `Memory`、`SetMemory` (0 で忘れる)、`Memories`
- 形を変えても、前の形のプログラムはキャッシュに残る (LRU で消えるまで)。`bonsaiai cache clear NAME` で全部消せば、次は選んだ形のものだけを作る (取ってくる)

### 7.9 Core AI の prebuilt: 変換後と一緒に、機械の種類ごとのコンパイル済みを配る

Core AI の書き出し (`coreai`、`.aimodel`) は機械に依らないが、初めて読む機械では Neural Engine のコンパイラが部品ごとに数十秒〜数分かかる (機械全体で 1 本の直列の列、[docs/design/apple-coreai.md](design/apple-coreai.md))。Metal Toolchain の `coreai-build` は実機がなくても 22 種類のアーキテクチャ向けに先にコンパイルでき (`X.ARCH.aimodelc`)、それを読む機械はコンパイラの代わりに仕上げだけをする。そのコンパイル済みを、変換後と同じ取得元に置く。

```
converted/MODEL/coreai/EXPORT/VERSION/...          書き出し (.aimodel)。どの Mac も同じ
converted/MODEL/coreai-ARCH/EXPORT/VERSION/...     ARCH (h13g = M1、h14g = M2、h15g = M3 ...) 向けのコンパイル済み、重みなし
```

**置く方針 (2026-11-18)**: Core AI のモデルは、取得元に次の 2 つを両方置く。どちらも tar を zstd で詰めた 1 つのファイル。

| 置くもの | 置き場所 | 誰が使うか |
| --- | --- | --- |
| `.aimodel` の書き出し (重みを含む、機械に依らない) | `coreai/`、`files.tar.zst` (7.10) | **すべての Mac**。prebuilt のない機種 (`coreai-build` が落ちる M1 Pro / Max / Ultra、M2 Pro / Ultra、M3 Pro / Ultra と h15m の M3 Max、A18、ANE に乗らない M6) はこれを端末でネイティブにコンパイルする。prebuilt のある機種も、重みはここから組み立てる |
| 機種ごとの prebuilt (重みを除いた `.aimodelc`) | `coreai-ARCH/`、`prebuilt.tar.zst` | `coreai-build` で作れる 12 種類 (M1、M2、M2 Max、M3、M3 Max の h15c、M4 / Pro / Max、M5 / Pro / Max / Ultra) だけ。初回のネイティブのコンパイルの代わりに仕上げだけにする |

- prebuilt は `.aimodel` を置き換えない (重みを持たないので、それだけでは動かない)。`.aimodel` を置かずに prebuilt だけ置くことはしない
- どの機種に効くかの対応は [docs/design/apple-coreai.md](design/apple-coreai.md) (`coreai-build inspect` の `Supported chips`)。作れない機種の prebuilt は置かない (取りに来ても見つからず、ネイティブに落ちる)
- 作る順: `.aimodel` の書き出しを置き (変換した機械が自動で)、そのあと開発機が `bonsaiai-dev models package EXPORT/coreai-all` で 12 種類の prebuilt を足す
- prebuilt を作るかはモデルの部品次第 (下の「効くかはモデル次第」)。作らないモデルは `.aimodel` だけ
- **チャットのモデルには今は作らない** (2026-11-18、M2、[docs/design/apple-coreai.md](design/apple-coreai.md)): LFM3-1.7B は初回の読み込みが 2 分 17 秒 (ネイティブ) から 8 分 12 秒 (prebuilt) に、plamo-3-translate-8b は 1 分 13 秒から 7 分 08 秒に延びた。Qwen3.5-4B は 7 分 07 秒で読めた (ネイティブは 12 分 26 秒) が、最初の呼び出しで止まり、1 時間 40 分たっても答えなかった。速さ (tokens/s) と答えは同じ
- 画像のモデル (2026-11-18、M2、1024²、[docs/design/apple-coreai.md](design/apple-coreai.md)): FLUX.2 klein 4B は初回が 52 分 16 秒から 16 分 55 秒に縮む (絵は細部が違う)。Anima-Turbo は 45 分 01 秒から 37 分 50 秒 (絵は同じ)。プロセスの最初の prebuilt に 3〜6 分かかるので、部品が少ないモデルでは負ける

- 重みは配らない: `.aimodelc` の 99 % は `resources.bin` (重み) で、中身は `.aimodel` の `main.mlirb` のバイトを並べ替えたもの (アーキテクチャに依らない)。`coreai.Pack` はそれを `resources.bin.map.json` (`main.mlirb` のどこから何バイト、の並び。数 KB) に置き換える。1 種類あたり書き出しの 1〜2 % (ANE の LLIR と小さなファイル)
- 鍵は部品のバイトコード: `programs` (書き出しの中の各 `.aimodel` の場所と `main.hash` の sha256)、`arch`、`export`、`format`。どの機械が変換したかに依らず、同じバイトの書き出しなら当たる。モデルは書き出しの entry.json のもの
- 作る (開発機、Xcode 27 の Metal Toolchain): `bonsaiai-dev models package EXPORT/coreai-all`。部品 × アーキテクチャごとに `coreai-build` を走らせ、詰めて `EXPORT.prebuilt/ARCH/` に置き、アーキテクチャごとに 1 つのファイル (`prebuilt.tar.zst`、tar を zstd で) にして書き戻す取得元へ (`EXPORT.prebuilt/ARCH.shared/`)。WebDAV ではファイルごとに PUT と MOVE がかかるので 1 つに。zstd で約 1/3 (h14g の 1 部品 3.0 MB → 0.97 MB、`-19`)。`coreai-build` が落ちる種類と、ANE に乗せられない部品は抜けたまま置く (その機械がネイティブにコンパイルする)
- 置いた束の `entry.json` には、中の部品の名前を `holds` に書く (`store.Entry.Holds`、2026-11-24)。束のキーは書き出しのプログラムのバイト列で、中身ではないので、作り終わる前に置いた束 (部品が足りない) と、あとで作り終えた束が同じキーになる。前は同じキーならあとの束を置かなかったので、FLUX.2 klein の v4 の h13g、h14g、h15g の束は ctx と二重ブロック 1 つだけのまま残り、M3 は DiT の大半を端末でコンパイルし直していた。今は、同じキーでも、置いてある束にない部品を持つ束なら置き換える (`store.HoldsMore`)。端末も、手元の束にない部品をコンパイルしようとしたときに取得元の `holds` を見て、足りない部品があれば束を取り直す
- 使う (どの Mac も、道具は要らない): Core AI のキャッシュに項目がなく、手元に prebuilt もない部品を読むとき、バックエンドが書き出しごとに 1 回、この機械の種類 (`AIModel.deviceArchitecture`) のものを取得元に聞く (`coreai.FetchPrebuilt`)。取れたら `EXPORT.prebuilt/ARCH.shared/` に置いて `EXPORT.prebuilt/ARCH/` に展開し、部品ごとに重みを `main.mlirb` から組み立てて (sha256 を照らす) 一時フォルダの `.aimodelc` を Core AI に渡し、読み込んだら消す。Core AI の項目ができるので次からは `.aimodel` のまま 0.01 秒。読めなければ (道具の版が違う、壊れている) その部品はネイティブに
- 効くかはモデル次第 ([docs/design/distribution.md](design/distribution.md)): `coreai-build` が ANE に乗せられない部品は配られず、乗った小さな部品は仕上げのほうがネイティブより遅いこともある。Chroma1-Flash の 1024² の DiT は初回の約 46 分が約 1 分になる。Sana-600M は書き出しを `coreai-build` に通る形にした ([docs/design/image-apple.md](design/image-apple.md)) ので、VAE の大きな段 (端末で 1 段 3 分 40 秒) が数秒〜10 秒になる見込み。だから prebuild は自動でなく、効くモデルの書き出しを選んで `-put` する
- 手元の `EXPORT.prebuilt/` は残す: 読み込んだ部品は Core AI の項目に入るが、項目 (重みの複製、書き出しと同じ大きさ) が消えたとき (空きが足りない、OS の更新) にネットに出ずに数秒で作り直すもとになる。空きが足りないときは、prebuilt のある部品の Core AI の項目が先に消える (7.7)。`bonsaiai cache` では種類 `prebuilt` の単位 (`clear --kind prebuilt` で消せる。消してもまた取得元から取る)。組み立てた重み入りの `.aimodelc` は読み込んだ直後に消す
- 項目がある機械は何も聞かない (2 回目からの実行は取得元に触らない)。`BONSAIAI_COREAI_PREBUILT=off` で使わない

### 7.10 AMD と Core AI の書き出しは 1 つに詰めて置く: `files.tar.zst`

AMD の書き出し (`converted/MODEL/amd-xdna2/EXPORT/VERSION`、`store.Packs`) は、取得元ではファイルを並べず、全部を tar にして zstd で詰めた 1 つのファイル (`files.tar.zst`、`store.PackFile`) と `entry.json` だけにする (2026-11-17、[docs/design/image-models.md](design/image-models.md))。Core AI の prebuilt (7.9) と同じく、WebDAV でファイルごとにかかる PUT と MOVE を 1 回にし、送る量を減らす。

2026-11-18 から Core AI の書き出し (`converted/MODEL/coreai/EXPORT/VERSION`、`.aimodel` の集まり) も同じに詰める (`store.Packs` が `coreai` の対象も取る)。チャットのモデルの書き出しはファイルが 100 ほど (LFM3-1.7B で 103) あり、縮むのは 1〜5 % (zstd の既定の速さで LFM3-1.7B の 2.62 GB が 2.59 GB、Anima の 3.03 GB が 2.89 GB、Chroma1-Flash の 15.97 GB が 15.25 GB。重みの表はほぼ縮まない)。効くのはファイルの数。機械の種類ごとの prebuilt (`coreai-ARCH`) はもともと `prebuilt.tar.zst` の 1 ファイルで、表 (`any`、`tables.safetensors`) は 1 ファイルなので、どちらもそのまま。変換しながら置くこと (7.1) はなくなり、書き出しが揃ってから詰めて送る。

```
converted/MODEL/amd-xdna2/EXPORT/VERSION/files.tar.zst   書き出しの全部のファイル (tar、zstd)
converted/MODEL/amd-xdna2/EXPORT/VERSION/entry.json      files: 中のファイル (大きさ、sha256)、pack: files.tar.zst の大きさと sha256
```

- 置く (`PutEntry`): 手元の書き出しを `work/pack/` に詰め (zstd の既定の速さ)、置いてから、そのフォルダの `files.tar.zst` と `entry.json` と印のほかのもの (前の BonsaiAI が並べて置いたファイル) を消し、`entry.json` を最後に書く。詰めるのは揃ってからなので、変換しながら置く (`StartEarly`) のはしない
- 取る (`materialize`): `files.tar.zst` を `DST.part/` に落とし (切れたら続きから)、sha256 を照らしてから展開する。entry.json の `files` にあるものだけを、それぞれの大きさと sha256 を照らしながら書き、足りなくても余っても断る。展開したら詰めたものは消す。手元に残るのは展開した書き出しだけ (詰めたものは持たない)
- 縮み方: 重み (bank) は 1 GB が 0.78 GB、プログラム (ELF) は 1/60 ほど (Qwen3.6-35B-A3B の書き出しで zstd -3)
- `entry.json` の `files` は中身を並べたままなので、`bonsaiai-dev sources ls` の大きさは展開したもの。`pack` を知らない前の BonsaiAI は中のファイルを取れずに外れ、手元で作る
- 同じ中身の版は送り直さない (2026-11-26): 置く前に、取得元の同じモデルと対象 (`converted/MODEL/TARGET/`) の別の書き出しや版から、`entry.json` の `Digest` (中のファイルの path、大きさ、sha256) が同じで `files.tar.zst` のあるものを探す。あれば詰めずに、取得元の中で写す (`store.Copier`: WebDAV は一時名への `COPY` (Depth 0) と `MOVE`、フォルダはハードリンク)。レシピや `tools/amd` の hash が変わって版を上げても、IRON の出力はバイト単位で同じなので (RTMDet-m の空のキャッシュからの書き出しが NAS のものと同じ `graph.elf` と bank)、数 GB を Wi-Fi で送らずに済む。写したものは独立したファイルなので、前の版を gc で消しても残る。S3 (tinygodriver に CopyObject がない) と、1 ファイルでも違う版は、今までどおり詰めて送る。部品 (グラフのフォルダ、bank) ごとに共有する置き方は、取得元の形式を変えるのでしていない
- `pack` のない AMD と Core AI の書き出し (前の BonsaiAI が置いたもの) も、今までどおりファイルごとに取れる
- 前の BonsaiAI がばらのまま置いた書き出しを詰め直す `bonsaiai-dev cache repack` があった。2026-11-21 に NAS の残り (手で取り込んだ AMD の書き出し 12 件、38 GB) を詰め、コマンドは外した。今の `PutEntry` は取り込みも含めて必ず詰めて置く

- `bonsaiai-dev models package MODEL...`: AMD のレシピ (model-recipes.yaml) のあるモデルの書き出しを作って (取得元にあれば取って) 書き戻し、書き出しと詰めた大きさを出す。IRON の書き出しは CPU だけ (aiecc と Peano、`bind_device` は名前で選ぶだけ) なので、NPU をほかが使っている間にも回せる
- 2026-11-17、チャットの 13 モデルを今の道具で書き出して UGREEN の NAS に置いた。合わせて 65.7 GB が 55.2 GB。gemma-4-E2B-it 7.42 → 7.11 GB、E4B 10.86 → 10.24 GB と gemma は 4〜6% しか縮まず、ほかは 2〜3 割 (Qwen3.5-4B 5.39 → 4.46、gpt-oss-20b 14.39 → 11.79、plamo-3-translate-8b 7.45 → 6.41)。Qwen3.5-0.8B (1.18 GB) を空の根に取って展開して答えるまでを確かめた (取るのに 40 秒、展開 2 秒)
- 注意: 同じ機械の `work/queue` を前の BonsaiAI (pack を知らないもの) も処理すると、同じ entry をばらで置き直し、詰めたものが `.put-` のまま残ることがある (plamo-3-translate-2b で起きた)。その機械の BonsaiAI を揃えること

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
- 元を消すのは 11.4 節の「長く使っていない」(`cache.unused_days`) と全体の上限 (`cache.max`) のときだけ。元の最後に使った日時は、それから作ったエントリを使った日時でも更新する (バリエーションを使っている間は元を残す)。`bonsaiai-dev sources gc --sources` で明示的に消せる
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

ブランチ: Actions は `staging` に上げ、確かめ終えたコミットを `main` に進める (`main` をそのコミットにする)。利用者の BonsaiAI は既定で `main` だけを読む (`--revision staging` で試せる)。照合に通らなかったものは `staging` に留まり、利用者に届かない。

トークン: 組織 `bonsaiai` の fine-grained トークン (リポジトリへの書き込みだけ、組織 `bonsaiai` に限る) を GitHub の environment `huggingface` の `HF_TOKEN` に置き、`main` からのジョブにだけ渡す。リポジトリはジョブが初回に作る (公開、gated の元モデルは 8 節のとおり同じ条件で gated に)。

HF のリポジトリ (組織 `bonsaiai`、1 モデル 1 リポジトリ):

```
README.md         モデルカード (元のモデル、ライセンス、照合の結果)
bonsaiai.json     全部の版の目録 (各 entry.json の写し)
amd-xdna2/v2/...  版ごとのフォルダ。古いバイナリのために前の版も残す
intel/v1/...
any/v1/...        表など対象によらないもの
```

- 版の間で変わらないファイルは HF の Xet が重複を除くはず。ダウンロードの量にも効くかを確かめ、効かなければ entry.json で前の版のファイルを参照する
- 古い版は、参照するリリースがなくなってから手で消す
- 上げるのは `bonsaiai-dev models push --to hf:bonsaiai/LFM3-1.7B --revision staging` (手元のエントリをそのまま上げる。CI 専用の形を持たない)

## 10. Apple (白紙)

Apple の機械が使えるようになるまで、Apple の配布の形は決めない。`targets` に `apple` の行は書かない。チームの層 (NAS、S3) での共有だけは先に入れた ([distribution](design/distribution.md)): Core AI の書き出しは `target: coreai` で、どの Mac も同じものを読む (端末で専用化したものは Core AI 自身のキャッシュに残る)。

- 今の Core ML の形: 配れるのは `.mlpackage` まで。`.mlmodelc` は OS の版ごと (`host`)、Neural Engine のコンパイル (e5 バンドル) は機械ごとで、初回の重さ (Gemma 4 で 100 秒級) が残る
- Core AI の方が事前にビルドしたものを配れる見込みがあり、そうなら Apple でも初回のコンパイルをなくせる。確かめること: 何を事前にビルドできるか、それがチップの世代と OS の版のどちらに縛られるか、Neural Engine への配置を確かめられるか
- Core AI で確かめたこと ([docs/design/apple-coreai.md](design/apple-coreai.md)、M1、macOS 27.2): 配れるのは `.aimodel` (端末によらない MLIR バイトコード)。端末で専用化したもの (初回のコンパイルの結果) は Core AI 自身のキャッシュにあり、OS のビルド (`26B123`) と実行ファイルの名前ごとで配れない。キャッシュの鍵は `main.hash` (中身の sha256) だけで、場所によらない。キャッシュに入ったあとは `main.mlirb` (重み) がなくても読める。8 トークンと 128 トークンのプログラムは、関数 2 つの 1 本の `.aimodel` にすれば、重みを 1 回分で配れて、キャッシュも 1 回分になる。初回のコンパイル (LFM3-1.7B で 1 分半ほど) は Core AI でも残る。`.aimodel` は手元に残すのを既定にし、消すのは最終手段のオプションにする (キャッシュはディスクが足りないときや OS の更新で消え、そのときは落とし直しになる)
- この仕組みの側で備えること: 対象の世代の階層 (3.1 節) にチップは入っている。OS の版に縛られる形なら、エントリに `os: {min, max}` を持たせ、選ぶときに見る (`host` にせず `portable` のまま、OS の範囲で選ぶ)

## 11. 進める順

1. P0 (手元だけ): 7.3 節の置き場所 (根を 1 つに、`src/` と `entries/`、`NPU_CACHE_HOME`、今の場所からの移し方)、`.done` を `entry.json` に (相対パス)、`source.json` とフォルダ名に owner、エントリごとのロック (flock / LockFileEx)、`If-Range`、placement の鍵にチップ、AMD の書き出しをキャッシュの下へ (`~/models` は読みだけの古い層として探す)、`Prune` を手元の層だけに、指紋の計算
2. P1: `Store` の interface、計画と作るの分離、path の層 (`write` と `fill`)、`bonsaiai-dev sources ls|pull|push|verify|gc`。この機械と Intel の機械の間で試す
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
- 書き戻しと写しの先はチームの S3 や NFS の層。どの層を使うかは組織のファイル、利用者のファイル、環境変数の 3 段で決める (7.4 節)。2026-11-15 に 12.1 節で置き換えた
- HF の名前空間は組織 `bonsaiai` (https://huggingface.co/bonsaiai。最初は `hwnpu` (2026-11-12 に取得) だったが、名前を BonsaiAI に変えたので移した)。1 モデル 1 リポジトリ (`bonsaiai/<モデル名>`)
- 置き場所は OS のローミングしないキャッシュのフォルダの下の 1 つの根だけ (7.3 節)
- gated のモデルとライセンスの確認のあるモデルは、既定では `fill` で写さない。組織のファイルで層ごとに `fill_gated: true` (7.1 節)。2026-11-15 に 12.1 節で置き換えた
- 共有の層には、手元で作ったものの書き戻し (`write`) のほかに、読んだものをそのまま置く写し (`fill`、フォワードプロキシの共有キャッシュ) も持てる (7.1 節)。2026-11-15 に 12.1 節で置き換えた
- 配布物がなければ今の仕組みで手元で作る。そのときは元データと変換後のデータを両方残し、パラメータを変えたバリエーションを元を落とし直さずに作れるようにする (8.1、8.2 節)

## 12.1 決めたこと (2026-11-15)

- 取得元は定義のファイルでなく、環境変数の URL の並び (`BONSAIAI_MODEL_SOURCE`、`_1`、...、番号の順、エラーなら次)。最後に `BONSAIAI_MODEL_HUGGING_FACE` (true/false)。`.env` と `.env.local` をカレント、ユーザー、機械の順に読む (7、7.4 節)
- `http(s)://` は WebDAV、書き戻さないなら読むだけの HTTP。URL のパスワードを受け付け、表示では伏せる。S3 の鍵は今どおり AWS の環境変数かプロファイル
- `?writeback=true` は、手元で作ったものと読んだものの写しの両方。gated も写す
- 中の並びは固定: `original/OWNER/NAME/` (コミットのフォルダは作らず `bonsaiai-lock.json` で照らす) と `converted/MODEL/TARGET/EXPORT/vN`。lock を移すのは `bonsaiai-dev sources update` だけ
- 版は `targets.<対象>.version`。版ありは `vN`、`develop: true` は `vN-HASH`、版なしは `v0-HASH`
- `BONSAIAI_CACHE_LAYERS` は読み替えずに消す。`BONSAIAI_PREBUILT` は作らない (取得元にあるかで決まる)
- `HF_TOKEN` は CI 以外では勧めない。refresh token の保存 (OS の資格情報の置き場) と、Actions のトークンの交換 (Hugging Face の Trusted Publishers、RFC 8693) は別に
- 層を平らに ([docs/design/distribution.md](design/distribution.md)): 手元のキャッシュを 0 番 (いつも書き戻す)、同梱のモデルを取得元の後ろ (読むだけ、その場で)。手元の並びも取得元と同じ `original/` に。モデルの探索と取得は公開の API (`modelstore`) にし、各コマンドも同じものを使う。各コマンドの `--dir` は消す (根は `BONSAIAI_CACHE`)
- 同梱のモデルの場所はインストーラが置き、アプリのコードで決める (環境変数にしない)。資源フォルダは configdir v1.1.0 の `QueryResourceFolders`
- `~/models` は次の PR で `converted/` に移して読まなくなる

## 13. 未決

層を平らにする (2026-11-15) の残り (2 つ目の PR):

- 変換したもの (`models/NAME/` のプログラム、`entries/` の AMD の書き出し) を手元でも `converted/MODEL/TARGET/EXPORT/vN/` に。完了の印を `entry.json` 1 つに (今は `NAME.done`、`NAME.ext.done`、`entry.json` が混在)。Core AI と AMD 以外の書き出しも取得元で共有される
- `~/models` を `bonsaiai-dev sources import DIR` で一度取得元に取り込み、以後は読まない。手で作ったものも同じく取り込む (tools/amd の案内も合わせる)
- `modelstore` に変換したものの API (`Make`)
- 同梱のモデルの変換したものをその場で使う (今は `FetchEntry` がコピーする)

- Synology の WebDAV の実機の確認
- refresh token の保存と、Actions のトークンの交換 (12.1 節)
