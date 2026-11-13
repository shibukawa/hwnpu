# 引き継ぎ: `hwnpu-chat` の Core AI の書き出し (`coreai`) を残りのモデルで確かめる

作成: 2026-11-14。[design.md](design.md) の 16.185 で、`hwnpu-chat --backend coreai` がチェックポイントから `.aimodel` を直接書くようになった。生成のプログラムと読み取りのプログラムの同じ層の部品は、関数 2 つの 1 本にまとめる。M3 (16 GB、macOS 27.2) で確かめたのは LFM3-1.7B、Qwen3-0.6B、Qwen3.5-0.8B、PLaMo-Embedding-1B。全部 Neural Engine に載り、2 回目の準備は 0.5 秒以下。M3 はディスクの空きが 12 GB しかなく、残りはここに書いたものを、ディスクに余裕のある機械 (macOS 27) でやる。

確かめ終わったら、`--backend` の既定を Core AI にして、`hwnpu-chat` から Core ML を外す (別の PR)。

## 1. 準備

```sh
go build -o /tmp/hwnpu-chat ./cmd/hwnpu-chat   # Core AI のキャッシュは実行ファイルの名前ごと
df -h /                                         # 空きが 15 GB を切ると macOS がキャッシュを消す
```

ほかの hwnpu のプロセスが Neural Engine を使っていないことを確かめる (ロックを待つと、それも準備の時間に入る)。

## 2. 確かめるモデル

各モデルを 2 回ずつ動かす。1 回目は変換とコンパイル、2 回目はキャッシュが当たる。

```sh
for m in Qwen3.5-4B gemma-4-E4B-it gemma-4-E2B-it Qwen3.5-2B Qwen3-1.7B plamo-3-translate-0.8b plamo-3-translate-2b; do
  for i in 1 2; do
    echo "== $m $i"
    /tmp/hwnpu-chat -m $m --backend coreai --temp 0 -n 150 --think=false "富士山について3文で説明してください。"
  done
done
/tmp/hwnpu-chat -m LFM3-VL-2B --backend coreai --temp 0 --image photo.png "この画像を一文で説明してください。"
/tmp/hwnpu-chat -m LFM3-Audio-2B --backend coreai --temp 0 --audio question.wav "答えてください。"
/tmp/hwnpu-chat -m LFM3-Omni-2B --backend coreai --temp 0 --image photo.png "この画像を一文で説明してください。"
for m in d1-2B d1-Omni-2B; do /tmp/hwnpu-chat systemone -m $m --backend coreai < request.json; done   # Core ML (--backend coreml) と答えが同じか
```

- 期待: どれも `all on the NPU` で、`converting …-q8-q128-last` の行のあとに `2 programs in N Core AI programs` が出る。2 回目の `ready after` は数秒以内
- 期待: 書き出しはモデルのディレクトリの `coreai/` だけに増え、その大きさは重みの 1 回分ほど。Core AI のキャッシュ (`~/Library/Caches/coreai-cache/<OS>/hwnpu-chat/`) の増え方も 1 回分
- **gemma-4-E2B-it と E4B**: 読み取りのプログラムはキャッシュだけ書く (`-kv`、`q128-kv`)。E2B ではその 3 部品が生成側の最初の 3 部品と同じ層で、関数 2 つの 1 本になる。M3 では小さな Gemma 4 で Neural Engine で確かめた (`TestConvertAIKVOnlyOnNeuralEngine`)。実物で、長めのプロンプト (128 トークンを超える) を読ませて答えが崩れないかを見る
- **Qwen3.5-4B**: 8 ビットの表で、16 GB の機械では重い。生成の速さと、`the Neural Engine holds … GiB` を記録する
- Core ML (`--backend coreml`) と答えを比べられるなら比べる。16.173 では、貪欲法の文が数十トークン目から分かれることがあった (fp16 の丸めの違い)

## 3. 記録すること

モデルごとに: 書き出しの時間と大きさ、1 回目と 2 回目の準備の時間、Neural Engine のメモリ、生成の速さ、答えの一行。design.md の 16.185 の表に足す。
