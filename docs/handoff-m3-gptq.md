# 引き継ぎ: GPTQ で選んだ lut6 / lut4 の表の M3 での確認

作成: 2026-11-07。Linux (aix1) で作って CPU で確かめた、活性を見て選んだ表と添字 (GPTQ と重み付きの表の当てはめ) を、Neural Engine のある Mac (M3、16 GB) で測るためのメモ。設計と CPU での結果は [docs/design.md](design.md) の 16.87 と [docs/amd-npu-models.md](amd-npu-models.md) の 13 節の「GPTQ」。

## 1. ここまでの状態

- ブランチ: `t3/q4-gptq` (PR #60)
- 何をするか: `hwnpu-convert -codes DIR` で、`tools/amd/q4_gptq.py` が選んだ表 (16 行ごと) と添字を lutN の重みに使う。重みはチェックポイントの値のハッシュ (`palette.WeightKey`) で引き当て、当たらない重みは今までどおり変換器が丸める。実行時の形 (表の大きさ、添字のビット数、16 行ごとの表) は変わらないので、Neural Engine での配置と速さは同じはず
- Linux で確かめたこと (milcpu、`hwnpu-check -package`、128 トークンの英文 `ref128`、BOS あり):

| 形式 | 変換器の丸め | GPTQ (一様な表) | GPTQ + 重み付きの表 (`--lut-fit wkmeans`) |
|---|---|---|---|
| lut6 | KL 0.074 | 0.050 | **0.038** (argmax 112/128、cosine 0.995) |
| lut4 | 2.7 | 0.66 | **0.33** (argmax 94/128、cosine 0.934) |

- **まだ確かめていないこと (この引き継ぎの目的)**: Neural Engine の上での KL (fp16 の計算が加わる)、配置 (`all on the NPU`)、速さが変わらないこと、答えの質

## 2. Linux から持ってくるもの

符号 (どれも 1.7 GB、LFM3-1.7B の 143 個の重み):

```sh
rsync -a aix1:models/q4-gptq/lfm3-lut6-wk/codes/   ~/models/q4-gptq/lfm3-lut6-wk/codes/
rsync -a aix1:models/q4-gptq/lfm3-lut4-wk/codes/   ~/models/q4-gptq/lfm3-lut4-wk/codes/
rsync -a aix1:models/q4-gptq/lfm3-lut6-codes/codes/ ~/models/q4-gptq/lfm3-lut6-gptq/codes/   # 一様な表 (比べる用)
rsync -a aix1:models/q4-gptq/apple/ref128.'*'      ~/models/q4-gptq/apple/
```

`ref128` は Mac でも `python tools/ref/ref_logits.py ~/.cache/hwnpu/models/LFM3-1.7B ~/models/q4-gptq/apple/ref128 --seq 128 --text "..."` で作れるが、同じ文で比べるため持ってくる。チェックポイントは hwnpu-chat が取ってきた `~/.cache/hwnpu/models/LFM3-1.7B` と同じもの (ハッシュが合わないと全部の重みが丸めに戻り、`-codes: 0 weights` と出る)。

## 3. 変換と KL

```sh
go test ./internal/palette ./internal/llm/...
M=~/.cache/hwnpu/models/LFM3-1.7B
cd ~/models/q4-gptq/apple
for v in rtn:: lut6-gptq:../lfm3-lut6-gptq/codes lut6-wk:../lfm3-lut6-wk/codes; do
  n=${v%%:*}; c=${v#*:}
  go run ./cmd/hwnpu-convert -config $M/config.json -weights $M/model.safetensors -o lut6-$n.mlpackage -seq 128 -format lut6 ${c:+-codes $c}
  go run ./cmd/hwnpu-check -package lut6-$n.mlpackage -ref ref128
done
go run ./cmd/hwnpu-convert -config $M/config.json -weights $M/model.safetensors -o lut4-wk.mlpackage -seq 128 -format lut4 -codes ../lfm3-lut4-wk/codes
go run ./cmd/hwnpu-check -package lut4-wk.mlpackage -ref ref128
```

(`go run` はリポジトリの根で。`-target` は既定の Neural Engine。Linux では `-target none` で milcpu だった。)

| 見るところ | 期待 |
|---|---|
| `-codes:` の行 | `143 weights took the chosen tables (143 in ...)` |
| `hwnpu-check` の KL | Linux の表に近い。lut6 の丸めで 0.07 前後、wkmeans で 0.04 前後。Neural Engine の fp16 の分だけ上がるのは構わないが、順位 (丸め > GPTQ > wkmeans) が保たれること |
| 配置 | 3 つとも `all on the NPU` |

## 4. 速さと答え

hwnpu-chat は今は `-codes` を取らない (チャットの変換は `internal/chat/model.go` の `prepare` で、既定の `kmeans-minmax`)。速さは `hwnpu-convert` の既定の形 (`lfm3-lut6-grouped-c16320-q8` と同じ `-seq`、`-attn grouped`) で、`-codes` あり/なしの 2 つを作り、`npuprobe` か `hwnpu-generate` で 1 トークンの時間を比べる。表の形が同じなので差は誤差のはず。

答えは、16.39 の質問 (富士山の高さ、『こころ』の作者、川) を `-codes` あり/なしで。lut4 + wkmeans は KL がまだ lut6 の丸めの 4 倍なので、答えが崩れないかだけを見る (lut4 の速さ 18〜24 tokens/s を取り戻せるかの判断材料)。

## 5. 結果に応じて

- lut6 + wkmeans の KL が Neural Engine でも 0.04 前後なら: チャットの変換に符号を通す (`chat.Prepare` に `Codes`、符号を Hugging Face から取る仕組み、パッケージ名に `-gptq` を入れて古いものと混ざらないように)。符号は 1 バイト/重みの添字で 1.7 GB あるので、配る形 (lut6 なら 6 ビットに詰める、または変換済みの mlpackage を配る) を決める
- lut4 + wkmeans の答えが lut6 の丸めと同じ程度なら: 既定を lut4 に戻せるか (速さ 1.5 倍) を 16.41 の表と並べて判断する
- 結果は design.md 16.87 の末尾に「M3 での確認」として書き足す
