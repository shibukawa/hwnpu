# Ouro (ループ型の言語モデル)

## 状態

- **一覧から外した (コードあり)**。2026-11-22 に models.yaml から外しました (`9a4f0048`)。
- 同じ日に `c053f77e` で入れて、開発中 (`develop: true`) として一覧に載せていました。

## モデル

| モデル | リポジトリ | ライセンス | 大きさ |
|---|---|---|---|
| Ouro-1.4B-Thinking | `ByteDance/Ouro-1.4B-Thinking` (`@3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`) | Apache-2.0 | 1.4B、`model.safetensors` 2.87 GB |
| Ouro-2.6B-Thinking | `ByteDance/Ouro-2.6B-Thinking` | Apache-2.0 | 2.6B。CPU で照合しただけ |

- 同じ重みの 24 層 (2.6B は 48 層) を、1 トークンに 4 回 (`total_ut_steps`) 回します。
- 層は Llama 型で、norm を前後に挟むサンドイッチ型です。注意は普通の多頭注意 (16 ヘッド、GQA なし) です。
- ByteDance の論文 (arXiv 2510.25741) は、1.4B で 4B 級の推論の力があるとしています。BonsaiAI ではベンチマークまではしていません。

## どの config で動かしていたか

外す直前の項目は `git show a60bfd6a:models.yaml` の `Ouro-1.4B-Thinking:` にあります。

```yaml
Ouro-1.4B-Thinking:
  family: ouro
  tag: 1.4b
  kind: chat
  develop: true
  from: hf:ByteDance/Ouro-1.4B-Thinking@3aaa2224253a92ca45cf2e3d427c360e1ef9c93d
  license: {id: apache-2.0, class: open}
  machine: 8
  memory: 3.8
  bytes: 2.87e9
  features: [thinking]
  amd_export: ouro --max-seq-len 2048 --q4 --mlp q6 --head q6
```

- arch は `ouro` です。各回を層として扱うので、1.4B の `Layers()` は 24 × 4 = 96 になります (`llm.Looped`)。
- 形式の既定は Apple が lut8、Intel が int8 です。AMD は注意が 8 ビット、MLP と出力の行列が 6 ビットです。

## 実装状況 (外した時点)

- **CPU (HF の fp32 と比べた結果、30 トークン):**

| 形式 | KL | 1 位の一致 |
|---|---|---|
| lut8 | 0.023 | 25/30 |
| lut6 | 0.18 | 23/30 |
| lut4 | 5.1 (使えない) | 3/30 |
| int8 | 0.006 | 29/30 |

  - 2.6B は 192 回の実行になるので、lut8 でも KL 0.13 でした。
- **AMD (Ryzen AI 9 365):**
  - q6 で KL 0.026、1 位の一致 98.4 % でした。
  - デコードは 3.7〜2.8 トークン/秒、bonsaiai-chat では 3.6 トークン/秒でした。比べると、LFM3-1.7B は 16.8 トークン/秒です。
  - NPU に固定されるのが 2.79 GiB (そのうち KV キャッシュが 1.6 GB) でした。
- **Apple (Neural Engine)、Intel の NPU:** 変換はできますが、実機では試していません。

## ロジックが残っている場所

main にそのまま残っています。

- Go: `internal/llm/ouro/` (`arch.go`、`build.go`、`config.go`、テスト)。`internal/llm/archs` に登録されています。
- チャットの書式: `chat.Ouro` (`ouro_template.json`)。
- トークナイザー: `internal/tokenizer` の `gpt2Digits`。
- AMD: `tools/amd/iron_ouro.py` (`iron_export_lm.py ouro`)。
- 記録:
  - [model-estimation.md](../model-estimation.md) の「Ouro」の節 (評価と判断)
  - [llm-models.md](../design/llm-models.md) の「Ouro」の節 (構造と照合)
  - [amd-llm.md](../design/amd-llm.md) (AMD への移植の覚え書き)
- AMD の NPU で動いた版: `a60bfd6a`。

## 問題と外した理由

- **遅い:** 1 トークンの計算が 96 層のモデルと同じになります。NPU のデコードは重みを読む時間で決まるので、同じ重みの大きさのモデルの 3〜5 分の 1 の速さでした。
- **キャッシュが大きい:** 回ごとに K と V を持ち、GQA でもないので、1 トークン 768 KiB になります。24 層で K と V が 4 ヘッドのモデルの 16 倍です。
- **小さく丸められない:** 同じ重みを 4 回読むので、丸めの誤差が積み重なります。4 ビットにはできません。
- **NPU の手当てが要った:** IRON がコマンドを切る目安に届かず、プロンプトの命令列 (233 MB) がデバイスのヒープ (64 MiB) に入りませんでした。12 層ごとにコマンドを切って直しました。
- **参照が古い:** リモートコードは transformers 5 で動かず、4.54 が要ります。

## 再開するときの手順と注意

1. models.yaml に上の項目を戻します (`develop: true` から始めます)。
2. まだ見ていないことがあります。
   - Neural Engine と Intel の NPU での配置と速さ。プログラム全体で 96 個のキャッシュを持ちます。
   - 回の数を減らしたとき (`total_ut_steps` 3) の速さと品質。
3. ループ型が釣り合うのは、大きな重みのモデルで一部の層だけを回すときだと見ています。例は Inkling 2 Mini です ([model-estimation.md](../model-estimation.md) の「Inkling 2 Mini」)。
