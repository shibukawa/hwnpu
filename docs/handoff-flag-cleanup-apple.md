# 引き継ぎ: 切り替えの整理の後に Mac で確かめること

作成: 2026-11-26。Linux の機械で、使わなくなった切り替え (比べ合い用の `HWNPU_DEV` のキー、convert の
引数、`--resident-dit`、tools/amd の環境変数) を消し、M1 の制限を model-recipes.yaml に移した。書き出しが
変わらないことは Linux で確かめたが、Mac でしか確かめられないことが残っている。

## 1. 変えたこと (PR)

| PR | 中身 | Apple の書き出し |
|---|---|---|
| #344 | `HWNPU_DEV` の 13 キー (attention、rotate、anima.attention、coreai.copy、iron.prefetch、load-ahead、hold-dit、moe.read/many/wave/prefetch2、mxfp4.i8mm/pad)、Gemma 4 の `-logits`、LFM3 の `-conv`/`-attn` (と paged の `tile` の形)、Qwen3.5 の `-delta`、`--resident-dit` を消す。消した Params は `llm.Arch.Fixed` に 1 つの値で残し、名前とメタデータは前と同じ | 同じ (下の 2) |
| #346 | `aspect.All32` (Sana、Qwen-Image) | 同じ (Anima) |
| #347 | 文書の前の変数名 | 関係なし |
| #348 | `HWNPU_MOE_WIRE` を 1 か所で読む。Neural Engine の専門家 (chain) も `6GB` を予算として読む (前は全部を固定) | 同じ (下の 2) |
| #345 | M1 の 3 つの判断 (一緒に持てる重み 2867 MiB、分けないプログラムなし、gather_nd の専門家を断る) を model-recipes.yaml の `apple.chips` に。会話のモデルの指紋に入る | 同じ (下の 2) |
| #349 | tools/amd の測定用の環境変数 | 関係なし |

`apple-versions.txt` は版を上げずに `-update` だけした (docs/model-recipes.md 2 の「作り直してバイトまで同じなら」)。

## 2. Linux で確かめたこと

`tools/coreai/same-exports.sh 987628d9 MODEL` (987628d9 は変える前の main) を Linux で回した。変換は Go
だけなので Linux でも作れる (プリビルドはしない)。

- 同じだったもの: 下の 3.1 の 3 つを除く全部 (47)。M1 で、会話のモデルは M1 と M2 の両方で (M1 で作らない
  LFM2-8B-A1B と PLaMo 3 Translate 8B は M2 で)
- 結果の記録: Linux の機械の `~/.cache/flagclean-same/*.txt`

## 3. Mac でやること

### 3.1 Linux で確かめられなかった書き出し

```sh
WORK=/tmp/same tools/coreai/same-exports.sh 987628d9 flux-3-image-klein supertonic-3
WORK=/tmp/same CHIP=M2 tools/coreai/same-exports.sh 987628d9 Qwen2.5-Coder-7B-Instruct
```

上の PR は全部 main に入っている。main で回す (987628d9 の後に入ったほかの PR は、`apple.code` の外か、
`apple-versions.txt` を自分で確かめたもの)。

- **flux-3-image-klein**: Hugging Face の利用条件への同意 (と HF_TOKEN) が Linux の機械になく、元のモデルを
  取れなかった。FLUX.2 klein (コードの大半が同じ) は同じだった
- **supertonic-3**: Linux では `.mlpackage` で書き出され、同じ commit で作っても Manifest.json の ID と
  model.mlmodel が毎回変わるので比べられない。変えたコードのうち supertonic が通るのは
  `internal/llm/convertai.go` だけで、そこで変えたのは注意で分けたモデルの書き方 (tile の形) だけ
- **Qwen2.5-Coder-7B-Instruct**: M1 では作らない (not_on)。M2 では GPTQ の表 (models.yaml の codes
  faace07a) を取得元から取るが、same-exports.sh は取得元を外すので、表がない機械では作れない。表を
  `~/Library/Caches/hwnpu/models/Qwen2.5-Coder-7B-Instruct/codes-faace07a` に置いてから回す
- 違ったら: 版を上げるのではなく、違いの元 (消した分岐が既定の道だった) を直す

### 3.2 動かして確かめる (M1 と M2 以降)

消した切り替えは、どれも既定でない道だけを選ぶものだった。既定の道が今までどおり動くかを見る。

- 会話: `hwnpu-chat` で LFM3-1.7B、Qwen3.5-2B (lut6 で回す: `rotate` の既定)、gemma-4-E2B-it (softcap は
  ホスト)。速さが前と同じくらいか
  - M1: 分けないプログラムを作らない、生成のプログラムがプロンプトも読む (`apple.chips.M1`)。
    `hwnpu-chat` の準備で作られるプログラムの名前が前と同じか
  - M1: LFM2-8B-A1B が「M2 or later, not on an M1's Neural Engine」で断られるか
- 専門家 (M2 以降): gpt-oss-20b か Qwen3.6-35B-A3B。MXFP4 の SMMLA (`mxfp4.i8mm` と `mxfp4.pad` を消し、
  閾値は定数 6)、`moe.read`、`moe.many`、`moe.wave` を消した。速さが前と同じくらいか
- `HWNPU_MOE_WIRE=6GB` で LFM2-8B-A1B (M2 以降): 前は全部 (約 7.2 GB) を固定した。今は 6 GB まで。
  固定した量はアクティビティモニタの「固定」(Wired) で見る
- 絵: Anima と FLUX.2 klein を Core AI で (DiT の組の入れ替えと先読みは、もう切れない)
- `HWNPU_DEV` に消したキーを書くと「not a key hwnpu knows」と 1 回出て、無視される

### 3.3 残っていること

- `coreai.drop-bytecode` は Mac の側で消した (#342)
- 確かめたら、この文書を消し、結果を docs/design/ の話題のファイルに書く
