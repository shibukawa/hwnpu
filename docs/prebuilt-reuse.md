# Core AI の prebuilt を部品ごとに差分で作る (作る側)

書き出しを作り直したとき (版が上がる、一部の部品だけ変わるなど) に、変わっていない部品の `.aimodelc` は前の書き出しのものを流用し、変わった部品だけを `coreai-build` にかける。範囲は作る側 (`hwnpu-dev prebuild -packed` / `-put`、`hwnpu-dev models package MODEL[/coreai-ARCH]`) だけ。端末が部品ごとに取る仕組みは別の課題 (7節)。

## 実装の状態 (2026-11-26)

- 入れたもの: 記録 (`build.json`)、流用の判断 (4節の6つの条件)、待避場所 (`ARCH.prev`)、端末の結果の記録 (動かしたときに自動で、または `-mark`)、記録の前に作った prebuilt に記録を足す `-record` (取得元の束にも)、記録が変わった束を取得元で置き換える `revision`
- コード: `internal/backend/coreai/record.go` (記録の形、`MarkDevice`)、`reuse_darwin.go` (候補、判断、写す、待避、記録を書く)、`prebuild_darwin.go` の `Build`、`pack_darwin.go` の `checkPacked`
- テスト: `internal/backend/coreai/reuse_darwin_test.go` (9節の場合を全部)
- 実機での確かめ (M1、偽の新しい書き出しに FLUX.2 klein v5 の `ctx` と `dit_double_0` を写して): 記録を足した v5 から `dit_double_0` の h14g を流用 (0.6 秒、重みの写像を読み直して確かめた。写したものは v5 のものとバイトまで同じ)。記録がなければ流用せずコンパイルに回った。`-mark failedToSpecialize` のあとは作らずに止まり、レシピの決まりを提案した
- Qwen3.5-0.8B で、記録のある版上げと、ビルドの選択を変えた一部の作り直しと、そのあとの推論を確かめた (M1、h13g、2026-11-26)。どれも、Core AI のキャッシュが空のプロセスで、端末でコンパイルしたもの (`HWNPU_COREAI_PREBUILT=off`) と同じ seed で同じ文を出し、prebuilt の失敗はなかった (端末の印は17部品とも `ok`)
  - r1 を一から: 17部品 6分22秒
  - r1 → r2 (今のコードの書き出し): 大きい9部品 (`model.part1`〜`7`、`*.mass`) を r1 から流用し、中身の変わった注意の8部品だけを作って 10 秒。読み込みは端末のコンパイルの 22 秒に対して 1.8 秒
  - レシピに `sr-q8.attn*` の `do: device` を足すと: その4部品は `ARCH.prev` に移り (束は13部品)、端末でコンパイルされた (読み込み 7.0 秒)
  - その決まりを外すと: 4部品は `ARCH.prev` から戻った (端末の `ok` も引き継ぐ)
  - `-j 1 EXPORT/model.part3.aimodel`: 「2 at once, 1 now」で `model.part3` だけ作り直し (1分50秒)、そのあとの通常の作り方では17部品ともそのまま
- `-record` を手元のフォルダの取得元 (`file://`) で確かめた: origin/main のコードで置いた記録のない束 (今の NAS の束と同じ形) に記録を足して置き直し、鍵と部品の一覧はそのまま、束の中の部品は元とバイトまで同じ、2回目は何もしない
- 手元の FLUX.2 klein の書き出しを `-n` で見ると、v1 (512+1024) の `ctx` は v5 のものと同じ source で流用の候補になる。v5 と v6 の DiT は source が違う (ExportVersion で変わった) ので作り直しになる。流用が効くのは、部品のバイト列が変わらない版上げ (レシピの別の部品の変更、VAE だけの変更など)

## 1. なぜハッシュだけにしないか

`main.hash` が同じなら `coreai-build` の出力も同じになるはずだが、前回の出力が悪ければ悪いものを持ち込むことになる。実例:

- **`coreai-build` は通るのに端末で仕上がらない**: FLUX.2 klein v6 の `dit_single_*` (10本) の h15g の prebuilt は、`coreai-build` がエラーなしに作ったが、M3 では10本とも `failedToSpecialize` だった (2026-11-26)。512² の 1024+512 のクエリの分け方で DMA の `cache_hint` が "Invalid TD" になったことも前にある (2026-11-23、[known-issues](known-issues.md) 12節)
- **2本同時に作ると NE から外れることがある** (`model-recipes.yaml` の冒頭)。`onANE` で一部は捕まる
- **`coreai-build` の版で結果が変わる**: レシピの `tools.coreai-build` で版を固定している。`.aimodelc` の `metadata.json` の `producer` に版が残る
- **レシピの決まりが変わる**: 前は端末に任せた部品が、今回は prebuilt の対象になることがある
- **チップごとの書き換え** (`rewrite`) は書き出しの名前とバイト列に入るので `main.hash` に出る

## 2. 記録 (`NAME.coreai.prebuilt/ARCH/build.json`)

パックした prebuilt と同じフォルダ (`coreai.PackedRoot`) に置く。そのため束 (`prebuilt.tar.zst`) にも入り、取得元から取った前の版の prebuilt にも記録が付いてくる。`ProgramsDigest` (束の鍵) は `.aimodel` の `main.hash` から作るので変わらない。端末は読まない。

```json
{
  "format": 1,
  "arch": "h15g",
  "tool": "coreai-build-3605.6.3",
  "pack": 2,
  "model": "FLUX.2-klein-4B",
  "export": "flux.2-klein-4b-8sizes-4d02f40a-5edits-cacbebeb-lut8-lut8-v6.coreai",
  "programs": {
    "text_0": { "source": "e190…", "result": "left", "rule": "LeftToDevice" },
    "dit_double_0": { "source": "…", "result": "built", "tool": "coreai-build-3605.6.3",
                      "jobs": 3, "built": "2026-11-25T10:35:30Z", "seconds": 1212,
                      "device": { "status": "ok", "checked": "2026-11-26", "function": "s1024x1024", "by": "hwnpu-image" } },
    "dit_single_0": { "source": "…", "result": "reused", "from": "…-v5.coreai", "jobs": 3,
                      "device": { "status": "failedToSpecialize", "checked": "2026-11-26",
                                  "function": "s1024x1024", "by": "hwnpu-dev prebuild -mark" } }
  }
}
```

- 項目の名前は書き出しの中のパスから `.aimodel` を除いたもの (`coreai.ProgramKey`)
- `result`: `built` (コンパイルした)、`reused` (流用した。`from` が元の書き出し)、`failed` (コンパイルが落ちた、または同じ source が端末で落ちた)、`left` (端末に任せた。`rule` がレシピの決まりの `why, checked`、または `LeftToDevice`)
- `jobs`: そのとき同時に走ったコンパイルの数 (キューの長さと `-j` の小さい方。1本ずつ作り直したものは 1)
- `pack`: `coreai.PackFormat` (取得元の鍵の `prebuiltFormat` と同じ値。そちらはこれを指すように変えた)
- `legacy: true`: 記録ができる前に作られた prebuilt (どう作られたか分からない)。今ある prebuilt を、その場で「作ってある」と飛ばしたときや `-mark` で印を付けたときに、ファイルから分かること (`metadata.json` の `producer` と `creationDate`) だけで書く
- 記録は `Build` の終わりに一度書く (`-n` では書かない)。途中でプロセスが殺されると書かれない (その部品は次に `legacy` として拾われる)

### 2.1 記録の前の prebuilt に記録を足す (`-record`)

```sh
hwnpu-dev prebuild -record -n                  # 足すものを並べる
hwnpu-dev prebuild -record                     # 手元のキャッシュの全部と、書き戻す取得元 (設定があれば) の全部の束
hwnpu-dev prebuild -record EXPORT_DIR...       # 手元のその書き出しだけ
```

- 記録のない (または `legacy` の) 部品に、ファイルから分かることで項目を書く (`coreai.Backfill`): `metadata.json` の `producer` と `creationDate`、`result: built`、`jobs` は今のレシピの `jobs`、`backfilled` にその日。取得元の束は、ここまでどれもレシピで作って置いたもの (落ちた部品のある書き出しは置かない) なので、記録のあとに作ったものと同じに扱う
- 端末で落ちた印 (`-mark` で付けたもの) は残す
- 取得元の束は、entry に `revision` のない (記録のない) ものだけを取ってきて、記録を足して束ね直し、置き直す。`revision` が変わるので置き換わる (`store.PutEntry`)。端末は部品の一覧 (`Holds`) が同じなので取り直さない
- 手元のキャッシュの束は、取得元の束を置き換えても書き換わらない (端末は記録のために取り直さない)。開発機ごとに `-record` を一度打つ (取得元の分は、もう記録があるので何もしない)

## 3. 作る流れ

1. 部品ごとに、レシピの決まり (`do: device`) と `LeftToDevice` を先に見る。当たれば作らない (流用もしない)
2. `-force` でなければ、候補から同じ source のものを探す。候補は次のフォルダの `*.ARCH.aimodelc`:
   - 今の書き出しの `NAME.coreai.prebuilt/ARCH` (その場所にあるもの) と、その待避場所 `ARCH.prev`
   - 同じフォルダ (同じモデルの `HWNPU_CACHE/models/<dir>/`) のほかの書き出しの `*.coreai.prebuilt/ARCH` と `ARCH.prev`
3. 同じ source の候補のどれかに、端末で落ちた印 (`device.status` が `ok` と `unchecked` のほか) があれば、作らずにその部品を失敗にする。同じ入力からは同じものができるので、作り直しても同じになる。今までの失敗と同じ流れで、レシピの `do: device` の決まりを提案して止まり、その書き出しのそのチップの分は取得元に置かない。印のあとでも作りたいときは `-force`
4. 候補を、その場所のもの、端末で `ok` のもの、そのほかの順に見て、4節の条件を全部満たす最初のものを使う。その場所のものならそのまま (「prebuilt already」)、ほかのものなら一時フォルダに写して (APFS の clone) から置き換える (「taken from X」)
5. どれも使えなければ今までどおりコンパイルする。候補があって外れたときは、最初に外れた理由をログに出す (`dit_double_0.aimodel h15g: compiling (not the one of …-v5.coreai: made by coreai-build-3605.5.0, coreai-build-3605.6.3 here)`)
6. 今回レシピが `do: device` にした部品の prebuilt がその場所にあれば、`ARCH.prev` へ移す。束に入って端末がそれを使うことがないように。決まりを外したときは、ほかの条件を満たせば `ARCH.prev` から戻す
7. 置き換える部品の前のもの (条件を満たさなかったもの、`-force` で作り直すもの) は、消さずに `ARCH.prev/` へ移し、記録の項目も `ARCH.prev/build.json` へ移す。次に作るときの候補になる。移すのは新しいものができてから (コンパイルが落ちたときは前のものがその場所に残る)
8. 終わったら `build.json` を書き、今までどおり `-packed` / `-put` で束にする。記録だけが変わったとき (`-mark` のあとなど) も、`-put` で取得元の束が置き換わる (`revision`)

パックしない作り方 (`hwnpu-dev prebuild` で `-packed` も `-put` もないとき、`.aimodelc` を `.aimodel` の横に丸ごと置く) は今までのまま (同じ source で NE に乗っていれば飛ばす)。

## 4. 流用する条件 (全部満たすときだけ)

| # | 条件 | どこで見るか |
|---|---|---|
| 1 | 元のプログラムが同じ | 新しい `X.aimodel/main.hash` と候補の `main.hash` (索引の鍵)、候補の `metadata.json` の `sourceHash`、記録の `source` |
| 2 | チップが同じ | フォルダ (`ARCH`) と名前 (`X.ARCH.aimodelc`) |
| 3 | `coreai-build` の版が同じ | 候補の `metadata.json` の `producer` と、今の `coreai-build --version` (`hwnpu-dev prebuild` は版がレシピの `tools.coreai-build` と違えば `--any-tools` なしでは止まる)。版が分からないときは、その場所にあるもののほかは使わない |
| 4 | その部品に当たるレシピの決まりが同じ | 今回 `do: device` なら作らない (1)。記録の `result` が `left` のものは使わない。今のレシピの決まりは `do: device` しかないので、決まりの中身のダイジェストまでは比べていない (決まりの種類が増えたら足す) |
| 5 | 前回の結果が良い | 記録の `result` が `built` か `reused`、`jobs` が今の `-j` 以下 (前より多く同時に作ったものは使わない。1本ずつ作ったものは何本同時の作り方にも使える)、`onANE`、端末で落ちた印がない (3) |
| 6 | 詰めた形が同じ | 記録の `pack` が `PackFormat`。候補の `resources.bin.map.json` を新しい `main.mlirb` から作り直して、大きさと SHA-256 が写像の書くものと同じこと (`checkPacked`。重みを一度読む。FLUX.2 の DiT 1本で 1 秒ほど)。`-n` では読まない |

記録のない候補は、その場所のもの (今の書き出しのもの) だけ今までどおり使う。ほかの書き出しのものは、条件 4 と 5 が分からないので使わない (`-mark ok` で端末で通った印を付けたものは使う)。記録の前に作った prebuilt には、一度 `-record` で記録を足す (2.1節)。フラグで流用の判断が変わることはない。

## 5. 端末の結果を記録に戻す

- **手で** (`-mark`): 検証のあとに打つ。`-programs` はパターン (path.Match)、なければ全部。`-function` は落ちた関数
  ```sh
  hwnpu-dev prebuild -mark failedToSpecialize -arch h15g -programs 'dit_single_*' -function s1024x1024 EXPORT_DIR
  hwnpu-dev prebuild -mark ok -arch h15g EXPORT_DIR
  ```
- **動かしたときに自動で** (設定なし): `hwnpu-image` などで、パックした prebuilt (`NAME.coreai.prebuilt/ARCH` の中のもの) を Core AI が開けたら `ok`、`the prebuilt program failed` のメッセージに `failedToSpecialize` があれば `failedToSpecialize` を、その束の `build.json` に書く (`coreai.DeviceMark`、`markPrebuilt`)
  - 書くのは記録 (`build.json`) のある束だけ。記録のない束に新しく作ることはしない
  - ほかの失敗 ("Request cancelled" など) は書かない。ほかのプロセスが Neural Engine を使っていたせいのこともあり、書くと良い部品を作る側が断ることになる。`failedToSpecialize` は部品の側の失敗で、毎回同じになる
  - 書くのは記録が変わるときだけ (`Record.NeedsMark`: `ok` の上に `ok`、同じ関数の同じ失敗は書き直さない)。関数ごとにプロセスで1回。書けないとき (アプリの読み取り専用の置き場所など) は黙って続ける
  - 利用者の機械でも書くが、その機械の中だけのもの (取得元へは `-put` する開発機からしか上がらない。`-put` すれば `revision` が変わって置き換わる)
- 落ちた印は、別の関数の `ok` では消えない (どれかの関数が落ちれば、その部品は使えない)。関数を書かない `-mark ok` (人が部品全体を確かめた) は消す
- チップが違う機械では確かめられないので、印のない部品は `unchecked` と同じに扱う (流用してよい)

まだないもの: 作ったあとにそのチップの開発機で全関数を1回ずつ仕上げてみる「確かめ」の段。`.aimodelc` を `AIModel(contentsOf:)` で読む小さなローダーと、`dataflow.graph_v1` を1関数だけ残すやり方で作れる。入れれば `-mark` を自動で打つことになる。

## 6. 決めたこと (2026-11-26)

1. **`build.json` は束に入れる**。`PackedRoot` の中に置くので `coreai.Bundle` がそのまま入れる。ほかの開発機が前の版の束を取れば、記録も付いてくる。束のバイトは変わるが、鍵 (`ProgramsDigest`) と `Holds` (`.aimodelc` の一覧) は変わらない。束の entry の `revision` が記録の SHA-256 で、記録だけが変わった束 (端末の印、`-record`) も取得元で置き換わる (2.1節)
2. **端末の結果は、動かしたときに自動で書き (設定なし。`failedToSpecialize` と `ok` だけ)、手 (`-mark`) でも書ける**。はじめに入れた `HWNPU_DEV=coreai.mark-prebuilt` はなくした。確かめの段 (5節の最後) は次
3. **記録のない今の prebuilt には、マージのときに `-record` で記録を足す** (取得元の全部の束と、各開発機の手元のキャッシュ)。フラグで信用する道 (はじめに入れた `-trust-old`) はなくした。記録のないものは、その場所では今までどおり使い、ほかの書き出しへは流用しない
4. **前の版を取得元から自動では引かない**。前の書き出しの名前を作る方法 (版を下げる) はモデルによって脆く、取得元を一覧する API もない。前の版の prebuilt が手元になければ、`hwnpu-dev models fetch` で前の書き出しの `coreai-ARCH` を取ってから作る (同じモデルのフォルダに入るので候補になる)

## 7. 範囲の外 (別の課題)

- **配る側を部品ごとに**: 今は「書き出し × チップ」で1束で、1部品変わると全部の取り直しになる。取得元の鍵を「チップ + 部品の `main.hash` + `coreai-build` の版 + `prebuiltFormat`」にすれば、端末は持っていない部品だけを取れる。WebDAV のファイルの数が増える (部品ごとに1ファイル。PUT と MOVE の費用) のが課題。束の中に索引を置いて、部品単位の束と組み合わせる手もある
- **同じ Mac の中での Core AI のキャッシュの再利用**: `~/Library/Caches/coreai-cache/<OS のビルド>/<プロセス名>/<main.hash の16進>/` が部品ごとのフォルダで、別のプロセス名のフォルダへ写しても使えた (2026-11-26、FLUX.2 の `fbase-image` から `fnew-image` へ。VAE の項目だけ除いて写し、DiT と文の符号器はコンパイルし直されなかった。初回 40 秒)。改名したテスト用のバイナリでモデル1つあたり 1.8 GB をコンパイルし直す問題に効く可能性がある。他の Mac へは持ち出せない見込み ([apple-coreai](design/apple-coreai.md): 端末で aned がコンパイルしたコードは SIP と署名の識別子に縛られる)

## 8. 参考: 数字 (M3、2026-11-25〜26)

| 部品 | `coreai-build` (h15g) | 端末で仕上げ | 端末でコンパイル |
|---|---|---|---|
| FLUX.2 VAE 17本 | 51 秒 | 0.9 秒 (復号器10段) | 625〜2799 秒 (復号器10段) |
| FLUX.2 `dit_double` 3本 | 5〜26 分/本 | 27〜101 秒/本 | 未測定 |
| FLUX.2 `dit_single` 10本 | 16〜21 分/本 | failedToSpecialize | 約 13 分/本 |
| FLUX.3 VAE 10本 | 1 分 22 秒 | 測定中 | 測定中 |
| Anima VAE | 8/10 段が `CompilationFailure` (h13g/h14g/h15g) | ― | 約 70 分 (M2) |

M3 (h15g) の FLUX.2 klein v6 の DiT 13本は、全部を作り直して 1時間21分かかった (3本同時)。M1 (h13g) では1部品で1時間を超えるものもある。

## 9. テスト (`reuse_darwin_test.go`)

- 前の書き出しに同じ source で記録が `built` / 端末 `ok` (または印なし) → 流用
- 同じ source で `producer` が違う → 作る (理由をログに)
- 同じ source で `device.status: failedToSpecialize` → 作らず、errBadOnDevice で止まる
- 同じ source で、今のレシピが `do: device` → `left`
- 前回は `left` → 作る。記録が `failed` → 作る。前は3本同時、今は1本 → 作る。NE に乗っていない → 作る。`pack` が違う → 作る
- 待避場所 (`ARCH.prev`) からの流用
- source が違う → 作る (理由なし)
- 記録のない前の prebuilt → 作る。`-record` (`Backfill`) で記録を足したものは流用、その印 (`failedToSpecialize`) は残る。その場所のものは使う。`-mark ok` を付けたものは流用
- 今回レシピが `do: device` にした部品の、その場所の prebuilt は `ARCH.prev` へ移す (束で端末へ行かないように)。決まりを外すと `ARCH.prev` から戻す
- `Backfill`: ファイルから書く、端末の印を残す、記録のあるものは書き直さない
- 取得元 (`chat` の `TestPrebuiltShared`): 記録が変わった束は同じ部品の束を置き換え、部品を持つ機械は取り直さない。部品の少ない束は、記録が違っても置き換えない
- 実際に写す: 写したもの、記録 (`reused`、`from`、端末の結果を引き継ぐ、`LeftToDevice` の項目)、その場所にあった古いものの待避と `ARCH.prev/build.json`、もう一度作るとその場所のものを使う
- 写像が別のバイト列に合わなければ使わない (`checkPacked`)
- `-mark`: パターン、`legacy` の項目、別の関数の `ok` では消えない、関数なしの `ok` で消える
