# 引き継ぎ: FLUX.2 klein の編集の参照を Apple と AMD で揃える

作成: 2026-11-24。Mac (Core AI) の FLUX.2 klein 4B で、1024 x 1024 の編集が受け取る参照の数と大きさを決め直した (PR #322、書き出し v5)。同じ呼び出し (参照 N 枚、`@half` の指定なし) で、AMD も同じ大きさの参照で描くように揃えるためのメモ。

## 1. 決めたこと (2026-11-24)

| 絵 | 参照 1 枚 | 参照 2 枚 | 参照 3 枚 | 参照 4 枚 |
|---|---|---|---|---|
| 1024 x 1024 (`1:1`) | 1024 (`1`) | 512 × 2 (`2@half`) | 512 × 3 (`3@half`) | なし |
| 512 x 512 (`1:1s`) | 512 (`1`) | 512 × 2 (`2`) | なし | なし |

- models.yaml の `npu.apple.edits` (`"1:1": ["1", "2@half", "3@half"]`, `"1:1s": ["1", "2"]`)。`@half` は絵の辺の半分の参照
- **参照の枚数は AMD に合わせた** (1〜3 枚)。解像度は Apple だけ半分に下げる。ユーザーの方針は「解像度は違っても、入力の本数は揃えて影響を小さく」「機種の上で再コンパイルが起きない範囲に止める」
- **なぜ 2 枚目からは半分か**: Apple の Neural Engine のコンパイラ (coreai-build、Metal Toolchain 27.2) は、DiT の 1 ステップが 9728 行以上 (1024 の参照と、ほかの参照を合わせたもの) になると単ブロックを Neural Engine に置けなかった。機種の上でのコンパイルにすると初回に何時間もかかる。8704 行 (1024 の参照 1 枚) までは、attention のキーを部分ごとの softmax にして log-sum-exp で合わせる形 (`milpart.PieceKeys`) で通った
- 行の数: 1 枚 4096 + 4096 + 512 = 8704、半分 2 枚 4096 + 2 × 1024 + 512 = 6656、半分 3 枚 4096 + 3 × 1024 + 512 = 7680 (512 はテキスト)

## 2. AMD の今の振る舞い (main、2026-11-24)

- **書き出し**: `tools/amd/iron_flux2klein.py` の `default_refs` が、1024 の参照と半分の参照の組み合わせ (4 枚まで、12800 行まで) をすべて作る。`1`、`2d2`、`3d2` (1024 x 1024) と `1`、`2` (512 x 512) はもう入っているはず (6656 と 7680 は 12800 以下)。**書き出し直しは要らない見込み**。確かめるのは 3 節の 1 番
- **参照の大きさの既定**: `ai/imagegen/klein.go` (`!asked && len(divs) > 0 && k.g.editRefs != nil`) と `ai/imagegen/generate.go` の `editFor` は、`editRefs` が nil (AMD) のとき参照を全部絵と同じ大きさにする。だから AMD では参照 2 枚 = 1024 × 2 (12800 行、1 分 41 秒)、Apple では 512 × 2 になり、同じ呼び出しで違う絵になる
- `editRefs` を埋めているのは Apple の書き出しのマニフェスト (`klein.go` 150〜165 行、`man.Edits`) だけ
- models.yaml の `npu.<vendor>.edits` の読み込み (`internal/chat/table.go` の `ne.Edits`、`spec.Edits`) は vendor を問わない。`npu.amd` に書いても読める

## 3. AMD でやること

1. **書き出しに組み合わせがあるか確かめる**: 今の AMD の書き出し (`flux.2-klein-4b.amd-npu2`、models.yaml の v1) に、1024 x 1024 の `2d2` と `3d2` の行の数 (6656、7680) のグラフがあるか。無ければ `--refs` で足して書き出し直し、`amd.version` を上げる (TestAMDVersions)
2. **models.yaml の `npu.amd` に `edits` を足す**: Apple と同じ `"1:1": ["1", "2@half", "3@half"]`, `"1:1s": ["1", "2"]`
3. **`ai/imagegen` で AMD の `editRefs` を models.yaml から埋める**: 今は Apple の書き出しのマニフェストからだけ。AMD では `chat.Model` の `NPU["amd"].Edits` から作る (`@half` → 2、無印 → 1)。`--image FILE@half` で大きさを指定したときは今まで通りそちらが勝つ (`asked`)
4. **4 枚の扱いを決める** (ユーザーに聞く): AMD は 4 枚まで描ける (BFL の仕様、`MAX_REFS = 4`)。表に無い枚数を AMD で
   - (a) エラーにする (Apple と同じ。`editFor` のエラーの文がそのまま出る)、または
   - (b) 今まで通り全部 1024 で描く (表に無い枚数だけ AMD の既定)
   Apple に 4 枚は無い。1024 x 1024 で半分 4 枚は 8704 行で、1024 の参照 1 枚と同じグラフ (docs/design/image-amd.md) なので、Apple でも足せる余地はある (未確認)
5. **試験**: `ai/imagegen` の参照の大きさを選ぶ試験に AMD の場合を足す (参照 2 枚 → `[2 2]`、`@half` の指定ありは指定のまま)
6. **確認の絵**: 1024 x 1024 で参照 1、2、3 枚の編集を描いて時間を測る。見込み (docs/design/image-amd.md の式 0.70 ms·n + 71 ns·n² から、DiT だけ 4 ステップ): 6656 行で 1 ステップ約 7.8 s、7680 行で約 9.6 s。今の 1024 × 2 (12800 行、1 ステップ約 21 s) より速くなる
7. **文書**: docs/design/image-amd.md の表 (半分 2 枚、3 枚の時間)、models.yaml の FLUX.2 の about と npu のコメント

## 4. Mac 側の状態 (同じ日)

- **PR #322** (draft、ブランチ `t3/attn-pieces`): 書き出し v5。8192 キー以上の attention を部分ごとの softmax と log-sum-exp で合わせ、1 節の表の編集を同じ書き出しのプログラムの関数に入れた (`-device` の書き出しは無くなった)
- **M3 (h15g)**: 8 プログラムすべてプリビルドして NAS に置き、M3 で動作を確認した (2026-11-24 18:21)
- **M2 (h14g)**: 単ブロックの 4 プログラムが Neural Engine に置けなかった。原因は半分 3 枚の関数 (`s1024x1024_refh3`) だけで、コンパイラの理由は `RegAlloc: failed to allocate tensors with fixed L2 hint`。1 プログラムの単ブロックを 5 個から 3 個に減らすと通る (1、2、3 個は通り、4、5 個は落ちる。オフラインのコンパイル)。model-recipes.yaml の `single_group` を 3 にする案をユーザーに出す予定。**Apple だけの変更で、AMD には関係しない**
- **M1 (h13g)**: 1024 の参照 1 枚の編集を機種の上でコンパイル中 (二重ブロックは 3 つとも全部 NPU)。単ブロック 3 個のプログラムで M1 もプリビルドできるかを試している。M1 で半分 3 枚が通らなければ、M1 だけ編集の組み合わせが変わる可能性がある (そのときはこの文書を直す)

## 5. 気をつけること

- 参照の大きさの表は、models.yaml の `npu.apple.edits` と `npu.amd.edits` で同じにしておく。片方だけ変えると、同じ呼び出しで違う絵になる
- `tools/amd/iron_chroma_parts.py` は FLUX の AMD の VAE が使っている。触ると TestAMDVersions のハッシュが変わる (Chroma を外したときに残した。docs/not-adopted/chroma1.md)
- FLUX.3 klein は今回の対象外。FLUX.2 が済んでから同じ決め方をするか考える
- 結果が出たら: この文書の 3 節、docs/design/image-amd.md、models.yaml のコメント
