---
name: port-architecture
description: Hugging Face のモデル構造（modeling_*.py）を Go に移し、hwnpu-convert のホワイトリスト（internal/llm/archs）に新しいアーキテクチャを追加するときの手順と合格条件。新しいモデル種別への対応、既存アーキテクチャの config の対応範囲の拡大、移植した構造の数値の食い違いの調査に使う。
---

# アーキテクチャの移植

hwnpu-convert が扱うモデルはホワイトリストで、各アーキテクチャは `modeling_*.py` の forward を手で Go に移したものです（llama.cpp と同じ方式。理由は docs/design.md の 16.9）。実例は 2 つあります。手順に迷ったらまずこれを読んでください。

- `internal/llm/lfm3`：因果 LM。畳み込み層と注意層が混在し、Neural Engine 向けの書き換え（`post-scale`、`conv`）が入っている
- `internal/llm/bert`：エンコーダ。埋め込み段、attention mask の入力、隠れ状態の出力がある。`bert_test.go` の fp64 の正解実装が、テストの書き方の手本

## 合格条件

すべて満たすまで完了ではありません。結果の数値は docs/design.md の新しい節に残します。

1. `go test ./...` が通る。`GOOS=darwin CGO_ENABLED=0 go vet ./...` も通る
2. 新しいパッケージに、少なくとも次のテストがある
   - 小さな合成モデル（`llm.NewSynthetic`）を fp16 で組み立てて CPU で実行し、**テスト内に直接書いた fp64 の forward** と相対誤差 1e-4 以内で一致する。正解側は modeling_*.py から書き、重みは fp16 に丸めてから使う
   - 1 層ずつ（`Start`/`Layers`、必要なら `NoHead`）実行した結果が、まとめて実行した結果と一致する
   - config のうち対応しない設定が、`ParseConfig` でエラーになる
3. **テストが誤りを検出できることを確かめる**。実装をわざと壊し（マスクを外す、残差の足し先を変えるなど）、テストが失敗することを確認してから元に戻す
4. 共有部分（`internal/llm/blocks`、`internal/milcpu`、`internal/llm` の型）に手を入れたときは、他のアーキテクチャの出力が変わっていないことを確かめる。LFM3 なら `go test -v -run TestBuildTiny ./internal/llm/lfm3/` が出すダイジェストを、変更の前後で比べる
5. 実物のチェックポイントで `hwnpu-check` を実行する（PyTorch と重みのある環境が要る。なければ、その旨と実行するコマンドを利用者に伝える）
   - `tensors: 0 missing, 0 mismatched`。`unused` は、読まない理由を説明できるもの（pooler、タスク用のヘッドなど）だけ。理由は `Tensors()` のコメントに書く
   - fp16（既定）で、各層の相対 RMS 誤差が 0.01 未満
   - モデル全体の出力：エンコーダは cosine 0.999 以上。因果 LM は argmax の一致率と KL を記録する（目安は KL 0.05 未満）
6. Neural Engine での配置と精度は Mac 上でしか確かめられない。Mac がなければ「未確認」と明記する

## 手順

### 1. 調べる

- `config.json` と `modeling_<name>.py` の forward を読む。nn.Module の木は `print(model)` で見る
- テンソルの一覧は、safetensors のヘッダか `hwnpu-check` の棚卸し（unused / missing）で確かめる
- GGUF に対応するなら、llama.cpp の `convert_hf_to_gguf.py` と `src/models/` で、テンソル名の対応と並べ替えを確かめる
- 新しい演算が要るかを洗い出す。要るものは手順 4 で `blocks` と `milcpu` に足す

### 2. パッケージを作る

`internal/llm/<name>/<name>.go` を作り、`internal/llm/archs/archs.go` に import を 1 行足す。

- `Config`：config.json のうち必要な項目だけを持つ。`ParseConfig` で既定値を入れ、**対応しない設定はエラーにする**（rope の種類、活性化関数、位置埋め込みの種類など）。黙って無視しない
- `Arch`：`Name` は GGUF の `general.architecture` と同じにする。`ModelTypes` / `Architectures` には config.json の値を入れる。固有の設定は `Params` に入れ、既定値を書く
- `llm.Model` のメソッド：`Hidden`、`Layers`、`Output`（ロジットか隠れ状態か）、`Embedding`（埋め込み表の名前）、`Tensors`（読むテンソルの名前と shape をすべて）、`Build`

### 3. Build の約束事

- 最初の入力は `x`：埋め込み表を引いただけの値で、形は (1, seq, hidden)。Core ML は gather を CPU に置くので、表引きはホストが行う
- `Start == 0` のときは、表引きと層 0 の間の処理（位置埋め込み、スケーリング、埋め込みの norm）もプログラムに含める
- `Start`/`Layers` で層を切り出す。`NoHead` が付いていれば、最後の層の後でも最終 norm の前の値を出力する。`Start == Layers()` なら、最後の層より後ろ（最終 norm と出力層）だけを組み立てる。`hwnpu-check` はこれらを使って層ごとに照合する
- 出力の名前は `y`。2 つ目以降の入力（mask など）は `x` の後に並べる。`hwnpu-check` は、x 以外の入力にすべて 1 を入れる
- エラーは `blocks.Run` の中で `Fail` を呼んで返す
- **nn.Module 1 つにつきメソッド 1 つ**にし、Python と並べて読める粒度にする。既存の `blocks` にある部品は使い回す

### 4. 新しい演算や部品を足す

- 複数のアーキテクチャで使う部品は `internal/llm/blocks` に置く。既存の部品が作る演算と定数の順序は変えない（変えると他のアーキテクチャのダイジェストが変わる）
- MIL の演算を新しく使うときは、`internal/milcpu/ops.go` にも実装し、`milcpu_test.go` に手計算か PyTorch の値と照合するテストを足す。CPU で実行できないプログラムは `hwnpu-check` で確かめられない

### 5. テストを書く

`internal/llm/bert/bert_test.go` を手本にする。

- 小さな config を JSON で書く。lut4 を試すなら、重みの行数を 16 の倍数にする
- 重みは `llm.NewSynthetic(cfg)` で作る
- 正解側は、modeling_*.py をループで直接書いた fp64 の forward。共有部品やインタプリタを使わない（独立していないと正解にならない）
- 合格条件 3 の「壊して失敗を確かめる」を行う

### 6. 実物で照合する

```bash
python tools/ref/ref_logits.py MODEL_DIR build/<name>/ref --seq 32            # 因果 LM
python tools/ref/ref_logits.py MODEL_DIR build/<name>/ref --seq 32 --encoder  # エンコーダ
go run ./cmd/hwnpu-check -config MODEL_DIR/config.json -weights MODEL_DIR/model.safetensors -ref build/<name>/ref
```

出力の読み方は次のとおりです。

- **`unused tensor`**：変換が読んでいない重み。移し漏れの手がかり（bias、追加の norm、ゲートなど）
- **`embeddings` がずれる**：埋め込み表の名前が違うか、取り違えている
- **最初に `<- over -tol` が付いた層**：その層の中に誤りがある。各層には参照側の入力を与えているので、前の層の誤りは混ざらない
- **層 0 だけがずれる**：埋め込み段（位置埋め込み、スケーリング、norm）を疑う
- **`after the last layer` だけがずれる**：最終 norm、出力層、tie_word_embeddings を疑う
- 原因を絞り込むには、`-layers 3` で 1 層だけ照合する。`-f16` を付けると、fp16 の保存精度で誤差が出るかが分かる

### 7. Neural Engine 向けの書き換えは最後に入れる

fp16 の素直なグラフが参照と合ってから入れます。書き換えは `Params` の設定にして、既定値を決めます。それを外すと素直なグラフに戻るようにしておき、書き換えの前後で数学的に同じ計算になることをテストで確かめます（LFM3 の `TestCPUVariants`）。Neural Engine の制約（値の名前にドットを使えない、gather は CPU、lut4 は出力 16 行ごとの表、行列積の誤差の床など）は MODEL.md にまとめてあります。

### 8. ドキュメント

- docs/design.md に節を足す：何を移したか、合格条件の数値、未確認の点
- MODEL.md と MODEL.ja.md の対応アーキテクチャ、README.md の一覧を更新する

## 間違えやすい点

- **RMSNorm**：`w * x` か `(1 + w) * x` か（Gemma 系は後者）。eps の config 名はモデルごとに違う（`rms_norm_eps`、`norm_eps`、`layer_norm_eps`）
- **RoPE の並び**：`rotate_half`（前半と後半を入れ替える）か、偶数と奇数の交互か。部分回転（`partial_rotary_factor`）があるか。`rope_scaling` があるか
- **GGUF の llama 系**：q/k の重みが RoPE の並びに合わせて並べ替えられている。safetensors と同じ扱いにすると壊れる
- **`nn.Linear` の重み**は (out, in) の向き。bias があるかは config（`attention_bias`、`mlp_bias` など）で変わる
- **GQA**：kv ヘッドは連続して繰り返す（`repeat_kv`。`blocks.SDPA` がこの順）
- **attention のスケール**：`1/sqrt(head_dim)` とは限らない（`query_pre_attn_scalar` など）。QK-norm があるか、logit の softcap があるか
- **埋め込み**：`sqrt(hidden)` を掛けるモデルがある（Gemma）。`tie_word_embeddings` かどうか。RoBERTa 系は位置 ID が `padding_idx + 1` から始まる
- **norm の位置**：pre-norm か post-norm か（BERT は post）。サンドイッチ norm か
- **活性化関数**：`gelu`（erf）と `gelu_new` / `gelu_pytorch_tanh`（tanh 近似）は別物
- **`output_hidden_states`**：多くの実装で、最後の要素は最終 norm の後になる。`tools/ref` は hook で層の出力を取るので、この違いに左右されない
- **チェックポイントの名前の接頭辞**：タスク用のモデル（`*ForMaskedLM` など）は `bert.` のような接頭辞が付く
