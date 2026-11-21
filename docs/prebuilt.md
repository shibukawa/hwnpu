# 変換済みのモデルを Hugging Face で配る (設計)

作成: 2026-11-07。2026-11-21 に書き直した: CI で作って上げる計画をやめ、**書き戻す取得元 (`?writeback=true` の層。今の環境では NAS) にある書き出しを上げることだけ**に絞った。状態: 最初の実装あり (2026-11-21、下の「実装の状態」)。まだ何も上げていない。

取得元の仕組み (層、`converted/MODEL/TARGET/EXPORT/VERSION`、`entry.json`、鍵、版) は [design/distribution.md](design/distribution.md) と [cache-sharing-design.md](cache-sharing-design.md) のまま使う。この文書は、その上に足す 2 つだけを決める。

1. **上げる道具** `hwnpu-dev models hf push`: 層の並びにある、今の main が取りに行く書き出しを選んで、`hwnpu/<モデル>` に上げる (`hwnpu` は既定の名前空間で、コードには書かない。models.yaml の `hub.push` で変えられる: 4.6 節)
2. **読む層**: hwnpu が models.yaml の `hub` に書いたリポジトリ (既定は `hwnpu/<モデル>`) を、利用者の取得元と元の Hugging Face の間で読む (読むだけの取得元の 1 つ)

作るのは今までどおり手元 (`hwnpu-dev models export`: AMD も Core AI の汎用版も Linux でも Mac でも作れる。チップごとの prebuilt だけは Mac の `hwnpu-dev models prebuild`。各コマンドの初回でも作られる)。作って書き戻す取得元に置くところまでと、そこから HF に出すところを分ける。

## 実装の状態 (2026-11-21)

| 部分 | どこ | 状態 |
|---|---|---|
| models.yaml の `hub.push` と、モデルの `hub` (リポジトリ × 対象のコミット) | `internal/chat/table.go`、`hubpins.go` | 済み。後から読んだファイルのリポジトリが先、同じリポジトリは対象ごとに上書き、`like` は引き継がない |
| 読む層 (取得元と同梱の後、元の HF の前) | `internal/store/hub.go`、`store.Sources` | 済み。`HWNPU_MODEL_HUB=false` で止まる。`hub` のないモデルと `original/` にはネットに出ない |
| 断られたときの取り方 (gated は元の HF と同じ `gate`、非公開は `HF_TOKEN` か `hf auth login`) | `internal/chat/hubpins.go` (`hubAuth`) | 済み。gated でないリポジトリにはトークンを送らない。**断られたら手元の変換に進む** (エラーで止めるのはまだ) |
| `vN` がなければ同じ鍵の `v0-HASH` を読み、`vN` として書き戻す | `internal/chat/sharedexport.go` (`fetchFrom`) | 済み。版を書いても取得元の今の書き出しが無駄にならない |
| `hwnpu-dev models export MODEL[/TARGET]` (どの OS でも、その対象向けに用意して書き戻す) | `cmd/hwnpu-dev/modelscmd`、`airole.For` / `ForTarget`、`chat.ActAsChip` | 済み (2026-11-21)。アプリの初回と同じ `airole.Role.Prepare` を、対象の NPU として呼ぶ (読み込まない)。AMD のチャットのモデルは今までの IRON の書き出し (変種ごと)。Core AI は M1 と M2 (それ以降の代表) の 2 通り: M1 だけ書き出しが違う (`wholeSlower`、`neuralEngineHold`)。Linux で M1 として作った Qwen3-0.6B の鍵は、NAS にある Mac の書き出しと同じだった |
| `hwnpu-dev models fetch MODEL[/TARGET]` (変換せずに、チェックポイントと取得元にある変換を手元へ) | `modelscmd/fetch.go`、`llm.WithFetchOnly` / `MayConvert` | 済み (2026-11-21)。export と同じ準備を変換なしで走らせ、取得元にないものは一覧に出す。HF から落としたものは書き戻す取得元にも入る |
| `hwnpu-dev models hf plan / push / check` | `cmd/hwnpu-dev/hfcmd` | 済み (下の 4 節)。手元の 1 回の走りで、取得元から選ぶ、置き場所を作る、prebuilt を書き出しのプログラムで照らす、カード・LICENSE・`config.json` を作る、`hf upload-large-folder`、models.yaml の固定、上がったかの確かめ |
| 書く権限の確かめ (`whoami-v2`) とトークンを貼る | `hfcmd/api.go`、`console.ReadSecret` | 済み。**書くための device flow はまだ** (トークンを貼るか `HF_TOKEN` / `hf auth login`) |
| 20 GB を超える書き出しを切る | | まだ (今の最大は Qwen3.6 の 22.65 GB) |
| `hwnpu-dev models hf check --remake`、gated の同意の文と欄の写し (`extra_gated_fields`) | | まだ (同意の文は hwnpu が書いた一文) |

実装で決めたこと (設計に足したもの):

- **道具だけが違うエントリも `vN` として上げる** (`hwnpu-dev models hf push` だけ。`plan -v` に「made with other tools」と出す)。`vN` を読む hwnpu は道具の hash を見ないので、同じことになる。2026-11-21 の NAS の AMD の書き出しは、どれも今の `tools/amd` より前の道具で作られている (鍵は違うが、道具の hash のほかは同じ)。上げる前に、その書き出しで今のコードが動くかを確かめる。実行時のほうは、同じ鍵 (同じバイト) のときだけ `v0-HASH` を読む
- 上げるときの `entry.json` は取得元のものをそのまま置く。`v0-HASH` から上げるものだけは、版と場所を書き換えて書き直す (中のファイルと詰めたものは同じバイト)
- 候補の行は状態の順 (上げられるもの、上げ済み、足りないもの)、その中は名前の順
- **Core AI の書き出しは版 (`vN`) のエントリだけを上げる** (2026-11-21 に決めた)。取得元には、今のコードがもう作らない名前の書き出しが、今の鍵のまま残っている (Anima の大きさごとの書き出しと、全部の大きさを 1 つにした書き出し。LFM3-1.7B では合わせて 17 GB、FLUX.2 klein では 65 GB)。名前だけではどれが今のものか分からないので、版を書いたあと `hwnpu-dev models export MODEL/coreai` を実行する (Linux でも Mac でも。Mac でそのモデルを一度動かすのと同じ)。今のコードが探す名前だけが、同じ鍵の `v0-HASH` から取られて `vN` として取得元に書き戻される (作り直しは要らない)。AMD の書き出しはレシピから名前が決まるので、`v0-HASH` からも上げる
- **表 (`any`) は Apple の行にだけ付ける**。表を読むのは Core AI の準備で、AMD の書き出しは自分の中に表を持っている
- **models.yaml の `disk`** (動かすのにキャッシュに残る量): `hwnpu-dev models hf sizes [--write models.yaml]` が取得元から出す。AMD はそのモデル自身の書き出し (変種のものでない。`bytes` と同じ)、Core AI は版のエントリだけ。表にない種類は不明 (一番大きい値にしない)。2026-11-21 に AMD の 20 モデルを書いた。AMD のダウンロード量は 20 モデルとも `bytes` と一致した

## 1. 何を配るか

| 対象 (フォルダ) | 中身 | 使う機械 | 配るモデル |
|---|---|---|---|
| `amd-xdna2` | IRON の書き出し (ELF、xclbin、バンク、manifest)。`amd_export` の版 (`variants` で増える `-p16`、`-p128` など) ごと | AMD の XDNA2 (Strix Point ほか)。**Linux と Windows で同じもの** | `amd_export` か初回の書き出しのあるモデル |
| `coreai` | Core AI の `.aimodel` (端末によらない MLIR バイトコードと重み) | macOS 27 以降のどの Mac も。初回はその Mac で ANE 向けにコンパイルする | Core AI の書き出しのあるモデル全部 |
| `coreai-<チップ>` (`coreai-h14g` など) | `coreai-build` の結果。重みなし (`coreai` の `main.mlirb` から組み立てる)。1 つ数十 MB | そのチップの Mac。初回のコンパイルが短くなる | 効く画像のモデルだけ (FLUX.2 klein、Chroma1-Flash、Anima)。チャットは配らない (初回が遅くなる: distribution.md) |
| `any` | 埋め込みの表 (`tables`) | どれも | 表のあるモデル。これがあればチェックポイントを落とさずに済む |

- **AMD は OS で分けない。** Windows は XRT (xrtshim) を通るが、読むファイルは Linux と同じバイト ([design/amd.md](design/amd.md))。`linux-xdna2` のような名前にはせず、Windows はモデルカードに「未確認」と書く。世代を分ける日 (XDNA1) が来ても、対象の名前 (`amd-xdna1`) で分かれる
- **Apple の汎用版は `coreai`。** `coreai-build` が落ちるチップ (M1 Pro / Max / Ultra、M2 Pro / Ultra、M3 Pro / Ultra、h15m、A18) と M6 は、`coreai-<チップ>` がないので `coreai` だけを落としてその Mac でコンパイルする。今の仕組みが既にそう動く (`coreai.FetchPrebuilt` がなければ端末でコンパイル)。汎用版を別に作る必要はない
- フォルダの名前は**コードの対象の名前のまま**にする (`ane-h14g` ではなく `coreai-h14g`)。取得元の並びと 1 対 1 にしておけば、読む層は前に `converted/MODEL/` を足すだけで済む。人向けの対応表 (h14g = M2 など、[design/apple-coreai.md](design/apple-coreai.md) の表) はモデルカードに書く
- 配らないもの: 元のチェックポイント (トークナイザーや config などの小さいファイルは、今どおり元のリポジトリから取る。元のライセンスへの同意もそこで済む)、GPTQ の符号 (変換し直すときにしか要らない)、Core ML と OpenVINO のプログラム (今も取得元で共有していない)、機械に縛られるコンパイルの結果

## 2. リポジトリの形

1 モデル 1 リポジトリ (`<名前空間>/<models.yaml の名前>`、例 `hwnpu/Qwen3.5-0.8B`。名前空間は 4.6 節)。ライセンスがモデルごとに違い、HF のライセンスの欄、gated、`base_model` もリポジトリごとなので、まとめない。

**AMD と Apple を 1 つのリポジトリにまとめる** (対象ごとに `hwnpu/X-xdna2`、`hwnpu/X-coreai` と分けない)。まとめた方が楽なので:

- models.yaml の `hub` がモデルに 1 つ (リポジトリは 1 つ、その下に対象ごとのコミット: 5.1 節)
- カード、LICENSE、gated の設定、ライセンスの確認がモデルに 1 つ
- 元のモデルのページの model tree に並ぶのが 1 つ。ダウンロード数も 1 つに集まる (並ぶ順に効く)
- 表 (`any`) を AMD と Apple の両方が読む。分けるとどちらかに置くか、両方に置くことになる
- 取る側は自分の対象のフォルダしか落とさないので、まとめても量は増えない。手で落とす人も `hf download --include "amd-xdna2/*"` で分けられる
- AMD の分だけを上げ直してもコミットは進むが、Apple の分は同じバイトのままで、固定を進めても Apple の利用者が落とし直すことはない
- 分けて得られるのは、対象ごとのダウンロード数と、対象の名前での検索だけ。検索はカードの `tags` (`amd-xdna2`、`coreai`) で足りる

```
README.md                                  モデルカード (3 節)
LICENSE                                    元のライセンスの写し (NOTICE があればそれも)
config.json                                目録: 置いた entry の一覧。hwnpu が取る前に 1 回読む (名前は 3 節のダウンロード数のため)
amd-xdna2/Qwen3.5-0.8B.amd-npu2-q6/v1/
  entry.json                               取得元のものをそのまま
  files.tar.zst                            取得元のものをそのまま
coreai/Qwen3.5-0.8B.coreai/v1/{entry.json,files.tar.zst}
coreai-h14g/flux.2-klein-4b-...-v1.coreai/v0-1a2b3c4d/{entry.json,prebuilt.tar.zst}
any/tables.safetensors/v0-5e6f7a8b/{entry.json,tables.safetensors}
```

- 中は取得元の `converted/MODEL/` の下と同じ並び。HF のパス `TARGET/EXPORT/VERSION/FILE` = 取得元の `converted/MODEL/TARGET/EXPORT/VERSION/FILE`
- **取得元のエントリをバイトのまま置く。** AMD と Core AI の書き出しは取得元で既に 1 つに詰めてある (`files.tar.zst`、`store.Packs`)。prebuilt も 1 つ (`coreai.BundleFile`)、表は 1 ファイル。だから HF でも 1 エントリはほぼ「`entry.json` と 1 ファイル」になる
  - FLUX.2 klein の AMD の書き出しも、NAS では詰めてある (673 ファイル、16.0 GB → 12.5 GB の 1 ファイル)。ばらなのは手元のキャッシュ (使うために展開したもの) だけ
  - 上げるときに展開も書き直しもしない。読む側は取得元から読むのと同じ道 (詰めたものを落とし、sha256 を見て展開する) を通る
  - ファイルの数が少ないので、匿名のダウンロードの要求の数の制限に当たりにくい。上げるのも簡単
  - 失うもの: 版の間の Xet の重複除去 (zstd の出力は、中の 1 ファイルが変わるとその後ろが変わる)。公開のリポジトリは容量が無料なので受け入れる
- **大きさ**: 一番大きい詰めたものは Qwen3.6-35B-A3B の `-q4-p128` で 22.65 GB (中身 34.8 GB)。HF の 1 ファイルの上限は 50 GB で、推奨は 20 GB。20 GB を超えるものは、詰めたものを 16 GiB ごとに切って置く (`files.tar.zst.000`、`.001`、... `entry.json` の `pack` に部分の大きさと sha256 の並び)。読む側はつないで 1 本として読む (zstd の流れのまま展開できる)。取得元でも同じ形にしてよい (S3 の 1 オブジェクトの上限 5 TB には関係ないが、WebDAV の NAS の 1 ファイルの上限が低いものにも効く)
- **版のフォルダは一度置いたら変えない。** 同じ `vN` に別の中身を上げようとしたら、道具はエラーにする (版を上げる)
- **履歴を消さない。** 各リリースは HF のコミットを固定して読む (5.1 節) ので、古いコミットが残っていれば古いリリースは動き続ける。新しいコミットで古い版のフォルダを消してもよい (消したコミットより前を固定したリリースは読める)。`super_squash_history` (履歴をまとめて古いコミットを消す) は使わない
- ブランチは `main` だけ。staging は持たない (確かめてから上げるので。4.3 節)

## 3. モデルカード (`README.md`)

道具が models.yaml と entry.json から作る。手で書き足す欄 (確かめた結果の補足) は `<!-- hwnpu:keep -->` から下に書き、作り直しても残す。

```yaml
---
license: apache-2.0              # models.yaml の license.id。HF にない ID は other と license_name、license_link
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: quantized   # 元のモデルのページの「量子化」に並ぶ
library_name: hwnpu
tags: [hwnpu, npu, amd-xdna2, coreai, apple-neural-engine]
---
```

**元のモデルから「量子化」として見せる**: HF の model tree。カードの `base_model` と `base_model_relation: quantized` で、元のモデルのページの横の欄 (Model tree の Quantizations) に並ぶ。

- `base_model` は models.yaml の `from` のリポジトリ (コミットは付けない)。`hwnpu-dev models hf push` が書く。ないリポジトリの名前だと HF が受け付けないので、上げる前に API で確かめる
- 関係の種類は adapter、merge、finetune、quantized だけで、「別の形式に変換した」はない。hwnpu の書き出しは q4、q6、lut8 などに量子化しているので quantized にする (量子化しない形だけの書き出しでも、近いのは quantized)
- 部品を別のリポジトリから読むモデル (`with:`、Chroma の T5-XXL など) は、主のリポジトリだけを書く。`base_model` を並べると merge に見える
- 同じ元のリポジトリの名前違い (`LFM3-Audio-2B` と `LFM3-Audio-2B-TTS`) は、別のリポジトリにすると同じ元のページに 2 つ並ぶ。1 つにまとめるか (リポジトリの下のフォルダに分ける形) は、並び方を見て決める
- 並ぶ順はダウンロード数や人気で決まるので、数える仕組み (下) がここにも効く

**`library_name: hwnpu` とダウンロード数**: `library_name` には何でも書けるが、HF が知っているライブラリの一覧 (huggingface.js の model-libraries) にないと、ページのリンクや使い方の表示が出ない。ダウンロード数は、一覧にないライブラリでは `config.json` への GET と HEAD を数える。

- 一覧に足すのは huggingface.js への PR (名前、リポジトリと文書の URL、使い方の例、`countDownloads` の条件)。HF の人が見て決めるので、通るかは分からない。モデルがいくつか上がり、使い方の文書が揃ってから出す
- **目録の名前を `config.json` にする** (2026-11-21 に決めた)。hwnpu は取る前にリポジトリの `config.json` を 1 回だけ読む (手元にあるときの速い道ではネットに出ないので、数えるのは本当に取ったときだけ)。一覧に登録する前から既定の決まりで数えられ、登録するときも `countDownloads` を `path:"config.json"` にすれば、数えるファイルが同じなので途切れない。登録したときに前の分が遡って数えられるかは HF の文書にないので、`hwnpu.json` のような別の名前で始めると、登録までの分を失う見込み
- 詰めたもの (`files.tar.zst`) を数える条件にはしない。1 回の取得で対象やファイルの数だけ数えてしまう
- 中身は `{"hwnpu": {"format": 1, "model": ..., "entries": [...]}}` だけにし、`model_type` や `architectures` は書かない。transformers の設定ファイルと同じ名前なので、HF のページや `AutoConfig` が読みに来たときに妙に見えることがある。最初の 1 モデルで、HF のページの見え方 (タグ、ウィジェット) とダウンロード数が増えることを確かめる。困る見え方になったら名前を変える (数えるのを諦めるか、登録を急ぐ)

本文:

- 使い方: `hwnpu-chat -m Qwen3.5-0.8B` だけ (hwnpu が自分で探して落とす)。自分の分だけ手で落とすなら `hf download hwnpu/Qwen3.5-0.8B --include "amd-xdna2/*"`
- 対象ごとの表: フォルダ、形式 (q6、lut8)、大きさ、作った hwnpu のコミット、確かめた機械 (`entry.json` の `made_by` と、上げるときに付けた検証の結果)
- チップの対応表 (`coreai-h14g` = M2 など)。prebuilt のないチップは `coreai` で動くこと
- 元のモデル (リポジトリとコミット)、ライセンス、条件 (`license.terms`)。表示の義務のあるもの (CC-BY の Magenta RT 2 の帰属など)
- Windows (XRT) は未確認

## 4. 上げる道具: `hwnpu-dev models hf push`

```
hwnpu-dev models hf plan [SEL...]           何を上げるかを出すだけ (上げない)
hwnpu-dev models hf sizes [N:MODEL...] [--write models.yaml]   ダウンロード量と、ディスクに残る量 (models.yaml の disk)
hwnpu-dev models hf push [SEL...] [--dry-run] [--all]
hwnpu-dev models hf check [MODEL...] [--remake]

SEL = [N:]MODEL[/TARGET]
  plamo-3-translate-2b                全部の対象、それぞれ一番前の層から
  1:plamo-3-translate-2b              1 番の層のものだけ
  plamo-3-translate-2b/amd-xdna2      AMD の分だけ
  1:FLUX.2-klein-4B/apple             Apple の分 (coreai と coreai-<チップ> の全部)
  FLUX.2-klein-4B/coreai-h14g         そのチップの prebuilt だけ (汎用の coreai は含めない)
```

**どこから上げるか**: 読むときと同じ層の並び (`hwnpu-dev models sources` の番号: 0 がこの機械のキャッシュ、1 から `HWNPU_MODEL_SOURCE`、`_1`、...) を前から順に見て、place の entry を最初に持つ層から取る。今の `store.FetchEntry` の chain of responsibility をそのまま使い、HF 用の別の置き場所は持たない。層を決めたいときは `N:` を付ける (`--source` のフラグは持たない)。今の環境では 1 番の NAS だけ。

**対象の選び方**: `/TARGET` は `amd-xdna2`、`coreai` (汎用だけ)、`coreai-<チップ>`、`apple` (`coreai` と全部の `coreai-<チップ>`)。表 (`any`) はどの対象を選んでも付く (どちらの NPU も読むので)。`--target` のフラグは持たない。prebuilt は汎用の `coreai` がないと使えない (重みをそこから組み立てる) ので、`coreai-<チップ>` だけを選んで HF に汎用のものがなければエラーにする。

**名指さないとき**: 層の一覧と、候補を出す。行の頭の `N:MODEL/TARGET` を、そのまま `hwnpu-dev models hf push` に渡せる。

```
$ hwnpu-dev models hf plan
layers:
  0  /home/me/.cache/hwnpu                         this machine's cache
  1  https://***@192.168.1.127:5006/llm            writeback (NAS)
  -  hf:hwnpu (hub.push, models.yaml built in)     where it goes

candidate                           license         pinned  version  chips  hf          next
1:whisper-large-v3-turbo/apple      open            yes     v2       12/12  old (v1)    push
1:whisper-large-v3-turbo/amd-xdna2  open            yes     v1       -      up to date  -
1:FLUX.2-klein-4B/apple             open            no      -        12/12  -           pin from, write version
-:FLUX.2-klein-4B/amd-xdna2         open            no      -        -      -           give the AMD export a key
1:LFM3-Audio-2B-TTS/amd-xdna2       LFM             yes     -        -      -           write targets.amd-xdna2.version, remake
1:LFM3-Audio-2B-TTS/apple           LFM             yes     -        0/12   -           write targets.coreai.version, remake
0:plamo-3-translate-2b/amd-xdna2    PLaMo           no      -        -      -           pin from, write version, remake
-:plamo-3-translate-2b/apple        PLaMo           no      -        -      -           hwnpu-dev models export plamo-3-translate-2b/coreai
-:flux-3-image-klein                non-commercial  -       -        -      -           give the AMD export a key (gated on HF)
```

- 行はモデル × 対象 (`amd-xdna2` と `apple`)。Apple の行の `chips` は、取得元にある prebuilt のチップの数 / `coreai-build` の通るチップの数 (今は 12。[design/apple-coreai.md](design/apple-coreai.md) の表)。`0/12` は汎用の `coreai` だけ (チャットのモデルは prebuilt を作らないのでいつもこれ)。`-v` でチップの名前と機種 (`h14g` M2、...) を並べ、足りないチップも出す。prebuilt のないチップの Mac は汎用のもので動くので、`0/12` でも上げられる`N` はその行の place を全部持っている一番前の層で、どの層にもなければ `-:`、層によって分かれていれば `0+1:` (それぞれの place を一番前の層から取る)
- 並び: 上げられるもの (4.2 節が全部通り、HF が古いか、ない) が先、次に足りないものがあるもの、最後に配らないもの。その中は名前の順。名前は models.yaml の全部のモデルから。どれから上げるかは人が決める (7 節の順)
- 列: ライセンスの表 (6 節) の判断、元のコミットの固定、`targets` の版、HF の状態 (`hub` があれば 4.5 節の `hwnpu-dev models hf check` の結果)、次にすること (4.2 節の外れた条件から)
- `hwnpu-dev models hf push` を名指さずに走らせると、同じ表を出し、「上げられる」(4.2 節が全部通り、HF が古いか、ない) 行だけを並べて確かめる (端末で y/N)。端末がないときは上げずに終わる。全部を黙って上げることはしない (`--all` で確かめを飛ばす)
- `hwnpu-dev models hf check` を名指さないときは、`hub` のあるモデル全部

`hwnpu-dev models` の隣に置く (取得元 0..N の扱いは同じ)。上げる先とトークンは 4.6 節。上げるのは手元の機械 (aix1 か Mac) で、上げる人が手で走らせる。

### 4.1 選ぶ: 「取得元にあるもの」ではなく「今の main が取りに行くもの」

取得元には古い鍵のエントリが残っている (2026-11-21 の NAS で 352 エントリ。Anima-Turbo の 1024² の Core AI だけで `v0-` が 2 つ、AMD は書き出しの道具の hash が変わるたびに増えた)。`ls` して全部上げると、誰も読まないものを上げることになる。

道具は、モデルごとに **hwnpu が実行時に計算するのと同じ place (鍵とパス) を、作らずに並べ**、そのエントリを取得元から探す:

| 種類 | place の出し方 (今あるもの) |
|---|---|
| AMD のチャット | `amdForms(m)` の各形 × `chat.ExportRecipe` (`hwnpu-dev models export` と同じ) |
| AMD の初回の書き出し (detect、pose、segment) | `chat.PinnedRecipe` (`hwnpu-dev models put` と同じ) |
| Core AI | モデルが使う書き出しの名前 × `coreAIRecipe` |
| Core AI の prebuilt | 上の Core AI の entry の `.aimodel` から `coreai.ProgramsDigest` × 各チップ |
| 表 | `chat.Tables` |

- これを `chat.Plan(ctx, m) []Place` として 1 か所にまとめる (今は `PrepareAMD`、`llm.FetchExport` などが作りながら計算している)。読む層 (5 節) の確かめ、`hwnpu-dev models` の「取得元に何が足りないか」にも使える (今は使い捨てのテストで調べている: 記憶の pitfalls)
- どの層にもなければ「足りない: `hwnpu-dev models export M` で作る」と出して飛ばす。**道具は作らない**
- 鍵のない種類 (手で作った `iron-import`: Anima、Chroma、FLUX.2 klein、DA3、MoGe-2、SigLIP 2、YuNet) は上げない。名前だけで当たるので、中身の違う書き出しを別のリリースの hwnpu が読みかねない。上げる前に、初回の書き出しのように鍵 (`PinnedRecipe` と Go の側の版) を持たせる (7 節)

### 4.2 上げてよいかを確かめる

上げる前に、モデルと place ごとに次を全部見る。1 つでも外れたらそのモデルは上げない (`plan` に理由)。

1. **元のコミットが固定してある** (`from: hf:OWNER/NAME@<40 桁>` か、全部のファイルの `sha256`)。`main` を追うモデルは、元が更新されると鍵が変わり、上げたものに誰も当たらなくなる
2. **版がある** (`targets.amd-xdna2.version`、`targets.coreai.version`。`develop: true` でない)。`v0-HASH` は鍵そのもので、AMD では `tools/amd` を直すたびに変わる (道具の hash が入る)。`vN` なら道具の hash を除いて当たるので、出力の変わらない直しでは上げ直さなくてよい。中身で鍵の決まる prebuilt と表は `v0-HASH` のままでよい
3. **ライセンス** (6 節): `models.yaml` の `license.class` と、6 節の表の「配る」
4. **取得元のエントリが揃っている**: `entry.json` があり、`.writing-` の印がない
5. **HF に同じパスがないか、あっても同じ鍵**。同じ `vN` で鍵 (道具の hash を除いたレシピ) が違えばエラー (版の上げ忘れ)

### 4.3 確かめた結果

上げるのは確かめたものだけにする。道具は確かめない (NPU で走らせるのは別の作業) が、記録を読む:

- `hwnpu-dev models hf push` は `--checked FILE` (JSON: place のパスごとに、機械、日付、KL や一致の数字、`all on the NPU`) を受け、目録 (`config.json`) とカードに書く。ないものは「未確認」と書いて上げる (`--require-checked` で止める)
- 最初は手で: AMD は aix1 で `hwnpu-chat` を回して数字を写す。Apple は Mac で
- これがあるので staging のブランチは要らない (CI で作って後から確かめる流れではないので)

### 4.4 上げる

1. **置き場所を作る**: `cache.Root()/work/hf/<モデル>/`。取得元の `entry.json` と詰めたもの (`files.tar.zst`) を、そのまま落として置く (展開しない)。sha256 は `entry.json` の `pack` と照らす。`file://` の取得元ならハードリンク。20 GB を超えるものは 2 節のとおり切る
2. **`entry.json` はそのまま** (`Path` は取得元の中のパスで、HF では前に `converted/MODEL/` を付けたものと読む)。`files`、`recipe`、`made_by` が同じなので、読む側の `takes` が取得元と同じに働く
3. **README.md、LICENSE、目録 (`config.json`) を作る** (3 節)。LICENSE は元のリポジトリから取る (`license.url` が別のページならカードにリンクだけ)
4. **HF に上げる**: `hf upload-large-folder <上げる先> <置き場所> --repo-type model` (huggingface_hub と hf_xet。トークンは 4.6 節で得たものを子のプロセスの `HF_TOKEN` にだけ渡す)。リポジトリがなければ、確かめてから (端末で y/N) 公開で作る
   - Go で Xet のプロトコルを書かない。上げるのは開発者の機械だけなので、Python の `hf` に任せる (`uv tool install huggingface_hub`)。`hf` がなければ入れ方を出して止める
   - `upload-large-folder` は途中から再開でき、コミットを分け、既にあるチャンクは送らない
   - コミットの説明: `hwnpu <コミット>: amd-xdna2/…/v1, coreai/…/v1`
5. **models.yaml を書き換える** (`hwnpu-dev models hf push` の続きで必ず行う): 上がった HF のコミットを、そのモデルの `hub` の、上げた先のリポジトリの**上げた対象の鍵** (`amd-xdna2`、`apple`、`coreai-<チップ>` だけを選んだら その鍵) に書く (5.1 節)。上げなかった対象の鍵は変えない
   - 書くファイルは 4.6 節 (上げる先を決めたファイル。組み込みの `hwnpu` ならソースの木の `models.yaml`)。見つからなければ書かずに、書く行を出して 0 以外で終わる
   - `hub` の下にその鍵の行があればその行だけを置き換え、なければ `hub` の下に足す (`hub` がなければ `from:` の次に `hub:` と `repo:` ごと足す)。コメント、並び、ほかの行には触らない (YAML を読み直して書き出すことはしない)。書いたら読み直して、その鍵が今のコミットになったことを確かめる
   - 1 回の `hwnpu-dev models hf push` で複数のモデルを上げたら、モデルごとに書く。上げに失敗したモデルの行は変えない
   - 書き換えのあとに、PR に入れるもの (`git diff models.yaml`) と、`go test ./internal/chat` (ネットのいらない検査、4.5 節) を走らせるよう出す。コミットは人がする
   - 別の機械で別の対象を上げても (AMD を aix1、Apple を Mac)、書き換えるのは別の行なので、マージでぶつからない
   - `--dry-run` では書き換えず、書く行を出すだけ
6. **片付け**: 上がったら置き場所を消す。モデル 1 つずつ進める (Qwen3.6-35B-A3B の AMD の 3 形で詰めて 60 GB 級。全部を一度に置かない)。`cache.Room` で空きを見てから置く

量の見当 (2026-11-21 の NAS から、古い鍵を除いて今の鍵だけ数える前の上限): AMD のチャットの書き出し 167 GB、Core AI 316 GB (同じ書き出しの古い鍵と、解像度ごとの古い形を含む。今の鍵の分は `hwnpu-dev models hf plan` で出す)、prebuilt 5 GB、表 23 GB。家の上りの速さで何時間かかるかは、最初の 1 モデルで測る。

### 4.5 足りないものと上げ忘れを見つける: `hwnpu-dev models hf check`

```
hwnpu-dev models hf check [MODEL...] [--remake]
```

`hub` のあるモデルごとに `chat.Plan` (4.1 節) の place を出し、models.yaml が固定した HF のコミットで次を見る。1 つでもあれば 0 以外で終わる。リリースの前と、`hwnpu-dev models hf push` のあとに走らせる。

| 出すもの | 意味 | 直し方 |
|---|---|---|
| HF にない | 版を上げた (または対象を足した) のに上げていない | `hwnpu-dev models hf push` |
| 鍵が違う | 同じ `vN` で、道具の hash を除いたレシピが違う (元のコミットや引数を変えて版を上げていない) | 版を上げて作り直し、`hwnpu-dev models hf push` |
| 道具が変わった (警告) | 鍵は同じだが、レシピの道具の hash が今のものと違う。出力が変わる直しなら版の上げ忘れ | `--remake` で確かめる |
| 固定が古い | `main` の先に、その対象の固定したコミットにないエントリがある | `hwnpu-dev models hf push` のあとに固定を書き換え忘れた |
| 固定が足りない | HF にその対象の place があるのに、`hub` にその対象の鍵がない | 鍵を書く (`hwnpu-dev models hf push` は書くので、手で上げたときだけ) |
| 取得元にあって HF にない (情報) | 上げられるもの | |

- **`--remake`**: 今のコードで手元に作り直し (`hwnpu-dev models export` と同じ、取得元には書かない)、HF のエントリの `files` の sha256 と比べる。違えば版の上げ忘れ。AMD のチャットは 1 形で数分から 30 分かかるので、「道具が変わった」と出たものだけに使う。前提として書き出しが同じ入力で同じバイトになること (時刻や乱数が入らないこと。[app-api-tasks.md](app-api-tasks.md) の確かめ)
- **ネットのいらない検査** (`go test`、`internal/chat`): `hub` のあるモデルは、元のコミットが固定してある、上げる対象に版がある (`develop` でない)、`hub` に `repo` があり、対象の鍵が知っている対象の名前で、値が 40 桁。PR で落ちる
- 道具の hash の変化を PR の時点で知らせたいなら、cache-sharing-design.md 5 節の lock (版ごとの指紋を書いたファイル) を足す。最初は `hwnpu-dev models hf check` の警告で始め、上げ忘れが実際に起きたら入れる

### 4.6 上げる先と権限: 名前空間は models.yaml から

`hwnpu` はコードに書かない。上げる先は models.yaml で決め、その名前空間に書ける権限を、上げるときに取りに行く。ほかの人が自分の名前空間 (自分のアカウントや組織) に上げることもできる。

**上げる先の決め方**:

1. `hwnpu-dev models hf push --to NAMESPACE` (その回だけ)
2. models.yaml の `hub.push` (後から読んだファイルが勝つ。組み込みは `hwnpu`。利用者は `~/.config/hwnpu/models.yaml` に `hub: {push: myorg}` と書く)
3. リポジトリの名前は `<名前空間>/<models.yaml のモデルの名前>`

`plan` の層の一覧の最後の行に、上げる先とそれを決めたファイルを出す (`- hf:myorg (hub.push, ~/.config/hwnpu/models.yaml)`)。

**権限を取りに行く流れ** (`push` の最初、何かを落とす前):

1. **トークンを探す**: `HF_TOKEN` → huggingface_hub のトークンのファイル (`hf auth login` が書く) → このプロセスで得たもの (5.2 節と同じ順)
2. **書けるかを聞く**: HF の `whoami-v2` で、トークンの持ち主、属する組織と役割、fine-grained のトークンならその範囲 (どの名前空間にどの権限か) を読み、上げる先の名前空間にリポジトリを作れるか、書けるかを決める (権限の名前と判定は、実装のときに HF の API で確かめる)
3. **書けなければ取りに行く** (端末があるとき):
   ```
   You cannot write to myorg on Hugging Face with the token of ~/.cache/huggingface/token
   (it can read only / it is scoped to otheruser).
     Enter   sign in here and allow hwnpu to write to repositories (device flow)
     t       paste a token that can write to myorg (https://huggingface.co/settings/tokens)
   >
   ```
   - device flow は 5.2 節と同じ hwnpu の OAuth アプリで、読むときより広い範囲 (`write-repos`、リポジトリを作るなら `manage-repos`) を `hwnpu-dev models hf push` のときだけ求める。組織に書くには、同意の画面でその組織を許してもらう必要がある (device flow でも組織を選べるかは実装のときに確かめる。だめならトークンを貼る方だけにする)
   - 貼るトークンは、書く範囲がその名前空間に限られた fine-grained のトークンを勧める
   - 得たトークンも 2 でもう一度確かめる。トークンはメモリにだけ置き、`hf upload-large-folder` には子のプロセスの環境変数 (`HF_TOKEN`) で渡す。ファイルには書かない
   - 端末がないときは聞かずにエラー (何が足りないか、トークンの作り方、`hf auth login`)
4. **リポジトリがなければ作る**: 上げる先と公開になることを出して y/N (`--all` でも飛ばさない)。gated にするものは作るときに設定する (5.2 節)

**固定を書くファイル**: 上げる先を決めたファイル (`hub.push` を書いたもの) に書く。組み込みの `hwnpu` ならソースの木の `models.yaml`、利用者の名前空間なら `~/.config/hwnpu/models.yaml` (`--to` で決めたときは `--models FILE`、なければ利用者のファイル)。書くのは `hub` の下のそのリポジトリの行 (4.4 節の 5)

**ほかの人の名前空間で読む**: 上げた人の models.yaml に固定が書かれるので、そのファイル (チームなら `/etc/hwnpu/models.yaml` や `HWNPU_MODELS`) を配れば、ほかの機械もそのリポジトリを読む (5.1 節の読む順で、組み込みの `hwnpu` より先に)。非公開のリポジトリなら読むときもトークンが要る (5.2 節の流れで 401 になる。非公開は gated と違い HF の API が gated と言わないので、「トークンが要る」とだけ出す)

## 5. 読む層 (hwnpu の側)

```
0. この機械のキャッシュ
1. HWNPU_MODEL_SOURCE, _1, ...      チームの NAS など
2. 同梱のモデル
3. hwnpu の Hugging Face (新)        hub のあるモデルだけ。読むだけ
4. 元の Hugging Face                 チェックポイントと小さいファイル
   どこにもなければ手元で作る
```

- `models.yaml` に `hub` (リポジトリと対象ごとのコミット、5.1 節) を足す (上げたモデルだけ)。`hub` のないモデルは 3 番に聞かない (存在しないリポジトリへの 404 を毎回出さない。オフラインでも何が配られているか分かり、`hwnpu-chat list` に「変換済みあり」と出せる)。`hub` は鍵に入れない
- `store` に読むだけの取得元を 1 つ足す: `converted/MODEL/REST` → `https://huggingface.co/<hub>/resolve/<コミット>/REST`。中身は今の読むだけの HTTP (GET と HEAD) で足りる。HF は 302 で CDN に回し、Range も効く (今の再開がそのまま使える)。HEAD は 302 を追った先の大きさ (または `X-Linked-Size`) を使う
- 当たったものは、今の仕組みのとおり 0 番と `?writeback=true` の取得元に写る (チームの 2 台目以降は NAS から)
- 止めるとき: `HWNPU_MODEL_HUGGING_FACE=false` で 3 番も 4 番も読まない。開発で自分の変換を試すなら `HWNPU_MODEL_HUB=false` (3 番だけ止める)。`hwnpu-dev models sources` に 3 番を出す
- トークン: gated でないリポジトリには送らない。gated のもの (6 節) は 5.2 節

### 5.1 HF のコミットを固定する (これがロックファイル)

models.yaml の `hub` に、**リポジトリごと、対象ごとの** HF のコミット (40 桁) を書く。読む層は自分の対象のコミットで `resolve/<コミット>/` を読み、`main` は見ない。別のロックファイルは持たない。

```yaml
hub:
  push: hwnpu                     # hwnpu-dev models hf push の上げる名前空間 (4.6 節)

models:
  whisper-large-v3-turbo:
    from: hf:openai/whisper-large-v3-turbo@41f01f3f...
    hub:
      hwnpu/whisper-large-v3-turbo:
        amd-xdna2: 3f2a9c...      # AMD の機械が読むコミット
        apple: 9c1d47...          # Mac が読むコミット (coreai と coreai-<チップ>)
        # coreai-h18g: 5e0b...    # 狭い名前は例外: そのチップだけ別のコミット
```

- **固定はリポジトリに付く**: コミットはそのリポジトリの中でしか意味がないので、リポジトリの名前の下に書く。別の名前空間に上げた人の固定 (4.6 節) は、別のリポジトリの行として並ぶ。混ざらない
- **読む順**: モデルの `hub` に並んだリポジトリのうち、自分の対象の鍵のあるものを、後から読んだファイルのものから順に (利用者のファイル → 組み込み)。最初に当たったものを使う。自分のチームで変換したものを先に、なければ公式のもの、と重ねられる

- **鍵は対象の名前**: `amd-xdna2`、`apple`、`coreai-<チップ>`。のちの `amd-xdna1` や `intel-npu4` も同じ形で足す。Mac は `coreai-<自分のチップ>` があればそれ、なければ `apple`。広い名前が既定で狭い名前が例外、という対象の決まり ([cache-sharing-design.md](cache-sharing-design.md) 3.1 節) と同じ
- **対象を別々に進められる**: Apple のチップの prebuilt を後から足したら `apple` だけを進める。AMD の機械が読むコミットは変わらないので、AMD の利用者は何も取り直さない (中身が同じなので取り直さないのは 1 本の固定でも同じだが、AMD の分を確かめ直す必要もなくなる)。チップを 1 つだけ試しに足すなら `coreai-<チップ>` に書く
- **表 (`any`)**: その対象のコミットにあるものを読む。HF の履歴は 1 本なので、表を上げたあとのコミットにはどれも入っている。`hwnpu-dev models hf check` は、対象ごとのコミットにその対象と表の place が全部あることを見る
- 利用者のファイルは名指したリポジトリの名指した鍵だけを変える (`targets` と同じ)。リポジトリを足すだけなら組み込みの固定は残る
- `hub` のないモデル、自分の対象の鍵のないモデルは 3 番に聞かない

- **新旧のプログラム**: models.yaml はバイナリに組み込まれるので、各リリースは自分が上げたときのコミットを読む。新しい版を上げると `main` は進むが、古いリリースが読むのは固定した古いコミットの中の古い `vN` のまま。新しいリリースは新しいコミットの新しい `vN`。同じリポジトリで両方が動く (2 節で履歴を消さないのはこのため)
- **中身の固定**: git のコミットは木を、木は各ファイルのポインタ (LFS の sha256) を指すので、コミットを固定すれば `entry.json` も詰めたものも全部決まる。`hwnpu` 組織のトークンが漏れて新しいコミットを積まれても、リリースは固定したコミットしか読まない。落としたものは今どおり `entry.json` の sha256 と照らす (壊れたダウンロードを見つける)。HF 自身がコミットの中身を正しく返すことは信じる
- **順番**: 作る → 取得元に置く → `hwnpu-dev models hf push` (HF にコミットができ、続けて models.yaml の `hub` の、上げた対象の鍵を書き換える: 4.4 節の 5) → その変更を PR に → マージ → リリース
- **利用者の上書き**: 自分の models.yaml で `hub: {hwnpu/X: {apple: <新しいコミット>}}` のように書けば、リリースを待たずに新しいものを試せる (鍵が合うものだけ当たる。合わなければ手元で変換)。コミットの代わりに `main` は開発用に許し、`hwnpu-dev models hf check` は警告を出す
- 開発中の main のビルドも models.yaml の固定を読む。上げる前の PR のブランチでは当たらず、手元か取得元 (NAS) から取る

### 5.2 gated のリポジトリを取る

今の gated の手順 (`internal/chat/hfauth.go`) は 4 番 (元の Hugging Face、`chat.fetchHF`) にしか効かない。3 番は `store` の取得元なので、そのままでは 401 を「ない」と受け取り、手元の変換 (AMD なら IRON の環境と数時間) に落ちてしまう。3 番と 4 番が同じ手順を使うようにする。

**仕組みを 1 か所に**: トークンを探す順と gate (401、403 のときの扱い) を `internal/hfauth` に移し、`chat.fetchHF` と 3 番の取得元の両方が呼ぶ。`store` の HTTP の取得元には `Auth` (トークンを返す、401 と 403 で gate を呼ぶ) を渡せるようにし、渡すのは 3 番だけ (HF のトークンを NAS に送らない)。

**トークンを探す順** (今のまま): `HF_TOKEN` → huggingface_hub のトークンのファイル (`hf auth login` が書く。読むだけ) → このプロセスで得たもの。

**401 (トークンがない、効かない) で、HF の API がそのリポジトリを gated と言うとき**:

```
hwnpu/flux-3-image-klein on Hugging Face asks you to agree to its terms before downloading.
  Enter   sign in here (opens nothing: enter a code at https://hf.co/oauth/device)
  t       paste a token (https://huggingface.co/settings/tokens, read access)
>
```

- 既定は device flow (今の `signIn`: hwnpu の OAuth アプリ、`openid` と `gated-repos` だけ、ブラウザのない機械や SSH でも使える)。トークンは 15 分で、大きなダウンロードの途中で切れたら refresh token で黙って更新する (今のまま)
- **`t` でトークンを貼る** (新): 画面に出さずに読む (`console` のエコーなし)。device flow が使えない環境 (組織の方針で OAuth アプリを許していない、など) のため。貼ったものもプロセスのメモリにだけ置き、書かない。残したい人には `hf auth login` を勧める (そのファイルは 1 番目に読む)
- 貼ったトークンも 401 なら、もう 1 度だけ聞く。2 回目もだめならエラー
- 端末がないとき (`hwnpu-chat serve`、スクリプト): 聞かずにエラー。同意のページ、`HF_TOKEN` か `hf auth login`、手元で変換するなら `HWNPU_MODEL_HUB=false` を並べて出す
- アプリ (`modelstore`): `Options.Auth` にコールバック (device flow のコードを出す、トークンを聞く) を渡せるようにする。渡さなければ端末のないときと同じエラー

**403 (アカウントがまだ同意していない)**: 今のとおり、ページを出して Enter を待ち、やり直す。

**同意が 2 つになるとき**: 元も gated (flux-3-image-klein) なら、元のリポジトリ (トークナイザーなど小さいファイル) と `hwnpu/` のリポジトリの両方に同意が要る。サインインは 1 回で両方に効く (同じアカウント) が、同意はリポジトリごと。2 回止めないように、3 番を読む前に両方に HEAD を 1 つずつ送り、403 のものをまとめて 1 度に出す。

**落ちない**: gated で断られたら、手元の変換に黙って進まずエラーにする (理由とやり方を出す)。手元で変換したいなら `HWNPU_MODEL_HUB=false`。

**リダイレクト**: HF は `resolve/` を CDN (別のホスト) に 302 で回す。Go のクライアントは別のホストへのリダイレクトで `Authorization` を落とし、CDN の URL は署名付きなのでそれでよい。トークンを送るのは `huggingface.co` だけにする (テストで確かめる)。

**上げる側**: gated にするリポジトリは、カードの YAML に `extra_gated_prompt` (と `extra_gated_fields`) を元のカードから写し、作るときに huggingface_hub の `update_repo_settings(gated="auto")` を呼ぶ (`hwnpu-dev models hf push` が models.yaml とライセンスの表 (6 節) を見て決める)。

## 6. ライセンス

**方針: 元のモデルに乗る。** 配るのは「有名なモデルが NPU ですぐ使える」ためで、元のモデルの代わりや新しい名前の別物にするためではない。なので、できるだけ元のとおりにする:

- ライセンスは元のまま (写す、言い換えない、足さない)。条件 (`license.terms`) はカードの先頭に元の言葉へのリンクと一緒に
- 名前は元のまま (`<名前空間>/<元の名前>`)。名前に元の名前を残す決まり (PLaMo) にも合う
- gated は元と同じ (下)
- カードは元のリポジトリとコミットを最初に示し、`base_model` で元のページの「量子化」に並べる (3 節)。性能や使い方の説明は元のカードに任せ、hwnpu で動かすことに関わること (対象、大きさ、確かめた機械) だけ書く
- 元が取り下げたり、ライセンスを変えたりしたら合わせる (取り下げなら HF のリポジトリを非公開にし、`hub` を外したリリースを出す)

公開して配るので、モデルごとに決める。**gated かどうかは元のリポジトリに合わせる**: 元が gated なら同じ条件で gated (承認は auto、同意の文と LICENSE へのリンクも元から写す)、元が gated でなければ gated にしない。元より厳しくしても法的な意味はなく、取る手間 (同意が 2 回、`hwnpu-chat serve` のように端末のないときはトークンが要る) が増えるだけ。

gated は「落とす前に同意させる」仕組みで、再配布してよいかの確認の代わりにはならない。確かめるのはライセンスごとに 1 回 (今は 4 つ): 量子化した派生物を再配布してよいか、LICENSE の写しや表示の義務、名前の付け方の決まり。結果はこの表に書き、カードの生成もこれを読む。

元の状態 (2026-11-21、HF の API の `gated`):

| ライセンス | モデル | 元の gated | 配るとき | 再配布の確認 |
|---|---|---|---|---|
| Apache-2.0、MIT ほか (`open`、32) | Qwen3.5、Gemma 4、whisper ほか | なし | gated なし。LICENSE を写す | 不要 |
| LFM Open License 1.0 (`conditional`) | LFM3 系、LFM2-8B-A1B、d1 | なし | gated なし。LICENSE を写し、条件 (年 1,000 万ドル未満の商用) をカードの先頭に | 配る (2026-11-21: ほかの人も量子化したものを HF で再配布している) |
| PLaMo コミュニティライセンス (`conditional`) | plamo-3-translate (3) | なし | gated なし。LICENSE を同梱、カードに「Built with PLaMo」と商用の登録の要否、リポジトリの名前に PLaMo を残す (`hwnpu/plamo-3-translate-2b`) | 配る (2026-11-21: 3 つの条件で) |
| CircleStone Labs 非商用 (`non-commercial`、`accept`) | Anima-Turbo、Anima-Aesthetic | なし | gated なし。LICENSE を写し、非商用 (重みは非商用、生成した絵は商用可) をカードの先頭に | 配る (2026-11-21: ほかの人も量子化したものを再配布している) |
| FLUX Kommunity (`non-commercial`、`accept`) | flux-3-image-klein | auto | gated (auto、元の同意の文を写す)。LICENSE を写し、非商用と Qualifying User の条件をカードの先頭に | 配る (2026-11-21: ほかの人も量子化したものを再配布している) |

- hwnpu の側の同意 (`accept: true`: 起動前に一度聞く) は、HF から取っても今どおり聞く。HF の gated とは別
- 確認が済むまで、そのライセンスのモデルは `hub` を書かない (上げない)
- 変換したもの (量子化した重み) は元の重みの派生物。どれも元のライセンスで配る

## 7. 進める順 (samples の使うモデルから)

samples が既定で使うモデルを先に配る。samples を試す人がチェックポイントの取得と変換 (AMD なら IRON の環境) なしで動かせるようにするため。

| sample | 使い (`ai.Use`) | モデル (機械のメモリで選ぶ) | ライセンス | AMD の書き出し | 前に要ること |
|---|---|---|---|---|---|
| interpreter | transcribe | whisper-large-v3-turbo | MIT | `iron-whisper` (鍵あり) | なし |
| | translate | plamo-3-translate-0.8b (8 GB)、2b (16 GB)、8b (32 GB) | PLaMo | `iron-lm` | 0.8b と 2b は元のコミットの固定 |
| | speech | LFM3-Audio-2B-TTS | LFM | `iron-lfm3speech` | なし |
| parallax、layers | depth | DA3-SMALL (8 GB)、DA3-BASE (16 GB)、DA3MONO-LARGE (24 GB) | Apache-2.0 | `iron-import` (鍵なし) | 鍵を持たせる、固定 |
| | image | FLUX.2-klein-4B | Apache-2.0 | `iron-import` (鍵なし) | 鍵を持たせる (`PinnedRecipe` の形は distribution.md にあるが、NAS のものは手で取り込んだもの)、固定 |
| parallax | face-detect | YuNet | MIT | `iron-import` (鍵なし) | 鍵を持たせる、固定 |

1. **仕組み**: `chat.Plan` (4.1 節)、`hwnpu-dev models hf plan` / `push` / `check` と `go test` の検査 (4.5 節)、読む層と HF のコミットの固定 (5 節、5.1 節)
2. **最初の 1 モデル: whisper-large-v3-turbo**。MIT、元のコミットは固定済み、AMD の書き出しに鍵がある、Core AI と表もある。`targets` の版を書いて作り直し、上げ、固定を書く。空の `HWNPU_CACHE`、取得元なしで、aix1 と Mac で `hwnpu-audio transcribe` がチェックポイントを落とさずに動くこと、落とした量と時間、HF のページの見え方 (model tree、`config.json`) とダウンロード数を確かめる
3. **interpreter の残り**: LFM3-Audio-2B-TTS、plamo-3-translate の 3 つ。ここで interpreter が全部 HF から動く
4. **LFM3-1.7B** (8 GB の chat の既定)
5. **parallax と layers**: DA3 の 3 つ、FLUX.2-klein-4B、YuNet。先に `iron-import` の書き出しに鍵 (`PinnedRecipe` と Go の側の版) を持たせ、作り直して取得元に置く。FLUX.2 klein は Core AI の prebuilt (`coreai-<チップ>`) も上げる最初のモデル
6. gated の取り方を共通に (5.2 節)。gated にするモデル (flux-3-image-klein) を上げる前に
7. 残りのモデル

## 8. 決めたこと (2026-11-21)

- CI で作らない。層の並び (書き戻す取得元。今は NAS) にあるものを、開発者が手で `hwnpu-dev models hf push` で上げる。どの層から取るかは読むときと同じ順で、`N:MODEL` で決められる。対象は `MODEL/TARGET` (`amd-xdna2`、`apple`、`coreai`、`coreai-<チップ>`) で選ぶ。self-hosted runner の計画 (この文書の前の版の 5 節) はやめた
- 1 モデル 1 リポジトリ、中は対象のフォルダ (`amd-xdna2`、`coreai`、`coreai-<チップ>`、`any`)。名前はコードの対象のまま
- AMD は OS で分けない (Linux と Windows で同じファイル)
- Apple の汎用版は `coreai`。チップ別 (`coreai-<チップ>`) は重みなしの差分で、ないチップは汎用版で動く
- HF には取得元のエントリをバイトのまま (詰めた `files.tar.zst` のまま) 置く。20 GB を超えるものは切る。版のフォルダは変えない。ブランチは `main` だけ。履歴は消さない
- 各リリースは models.yaml の `hub` の、**対象ごとの** HF のコミットを固定して読む (`amd-xdna2`、`apple`、例外に `coreai-<チップ>`。のちの対象も同じ形で足す)。`hwnpu-dev models hf push` が上げたあとに、上げた対象の行まで書き換える。これがロックファイルで、新旧のリリースが同じリポジトリで動く
- `hwnpu-dev models hf check` (足りないもの、版の上げ忘れ、固定の書き忘れ) と、ネットのいらない `go test` の検査を持つ
- 上げるのは版 (`vN`) のあるものだけ (prebuilt と表は中身の鍵の `v0-HASH` でよい)。元のコミットの固定が必須
- 上げるのは Python の `hf upload-large-folder` に任せる (Go で Xet を書かない)。読むのは Go の今の HTTP の取得元
- 読む層は `models.yaml` の `hub` のあるモデルだけ
- 名前空間 `hwnpu` はコードに書かない。上げる先は `hub.push` (利用者のファイルで変えられる) か `--to`。固定はリポジトリごとに書くので、ほかの人の名前空間の固定と混ざらず、読むときは利用者のファイルのリポジトリが先。書ける権限は `hwnpu-dev models hf push` の最初に `whoami-v2` で確かめ、なければ device flow かトークンを貼って取る
- リポジトリの目録の名前は `config.json` (HF の既定のダウンロード数の数え方に乗り、ライブラリの登録のあとも途切れない)
- gated の取り方は 3 番と 4 番で共通 (`internal/hfauth`)。端末では device flow (既定) かトークンを貼るかを選ぶ。どちらもメモリにだけ置く。断られたら手元の変換に落ちずにエラー
- 方針は「元のモデルに乗る」: ライセンス、名前、gated を元のとおりにし、カードは元を示して hwnpu で動かすことだけを書く (6 節)
- gated かどうかは元のリポジトリに合わせる (元が gated なら同じ条件で、でなければ gated にしない)。再配布の可否はライセンスごとに別に確かめる
- Apple のフォルダの名前は `coreai` / `coreai-<チップ>` (コードの対象の名前のまま)

## 9. 未決

- 書くための device flow (今はトークンを貼るか `HF_TOKEN` / `hf auth login`)。hwnpu の OAuth アプリに `write-repos` を足すか
- gated で断られたときに止める (今は手元の変換に進む)
- 20 GB を超える書き出しを切る形 (`files.tar.zst.000` ...) と、それを読む側

- Intel (OpenVINO IR) を配るか。今は取得元でも共有していない
- 確かめた結果の形 (`--checked` の JSON) と、それを作る手順 (`hwnpu-dev check` の出力から作れるか)
- 目録 (`config.json`) の形 (hwnpu が取る前に 1 回読む。ダウンロード数の条件にも使う)
- huggingface.js の一覧に `hwnpu` を足す PR を出す時期 (3 節)
- 詰めたものを切る大きさ (16 GiB) と、取得元でも同じ形にするか
- 古い版を消す基準 (どのリリースまで支えるか)
