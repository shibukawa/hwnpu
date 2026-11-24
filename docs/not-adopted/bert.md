# BERT の系列 (all-MiniLM-L6-v2 など)

## 状態

- **一覧から外した (コードあり)**。models.yaml に載せたことはありません。
- 符号器の移植は 2026-10-29 です (`0d0e2219`)。models.yaml ができたのはそのあとの 2026-11-09 です。

## モデル

- 確かめたのは `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0、22M、6 層) です。
- transformers の `BertModel` (`modeling_bert.py`) の形をしたものなら、同じコードで変換できます。タスク用のチェックポイントにつく `bert.` の接頭辞は、config.json の `architectures` から判断します。

## どの config で動かしていたか

- arch は `bert` です。models.yaml の項目はなく、`bonsaiai-dev convert` にリポジトリを直接渡していました。
- 形式の既定は Apple が lut8、Intel が int8 です (`Arch.IntelFormat`、`DefaultFormatFor`。`internal/llm/archs/archs_test.go` で固定)。
- 入力は `x` と `mask` (語が 1、埋め草が 0) です。出力は最後の隠れ状態で、プーリングはホストに任せます。

## 実装状況

| | 結果 |
|---|---|
| Apple | 変換できます。all-MiniLM-L6-v2 の fp16 と比べたコサインは、lut8 で 0.9988、lut4 で 0.23 (使えない) でした |
| Intel | 変換できます。int8 が既定で、int4 は 0.91〜0.94 でした (`0287e300` の測定) |
| AMD | していません |

- `bonsaiai-check` は符号器も扱えます。余分な入力には 1 を入れ、隠れ状態は argmax と KL なしで比べます。

## ロジックが残っている場所

main にそのまま残っています。

- `internal/llm/bert/` (`bert.go`、`bert_test.go`)。`internal/llm/archs` に登録されています。
- 共有の部品として、`blocks` の外からマスクを受ける SDPA、`LinearBias`、`LayerNorm`、`Activation`、milcpu の `gelu` と `layer_norm` を足しました。いまはほかのモデルも使っています。
- 記録: [llm-models.md](../design/llm-models.md) の「BERT family」の節。

## 問題と一覧に入れない理由

- **トークナイザーがない:** WordPiece のトークナイザーを作っていません。`tokenizer.Load` が断るので、bonsaiai-chat からは使えません。
- **用途が重なる:** 文の埋め込みは、PLaMo-Embedding-1B (`plamo-bi`) が一覧にあり、日本語も扱えます。

## 再開するときの手順と注意

1. `internal/tokenizer` に WordPiece を足します。HF の tokenizers と同じ id になることを、ほかのトークナイザーと同じように固定の文とランダムな文で確かめます。
2. models.yaml に `kind: embedding` の項目を足します。sha256 と、Apple lut8、Intel int8 の形式も書きます。
3. lut4 は使えないので、8 GB の枠でも lut8 にします。
