# 開発者向けのコマンド (hwnpu-dev)

[English](USAGE_DEV.md)

動作確認、測定、モデルの変換と照合のコマンドです。

## 動作確認レポート

手元の NPU で hwnpu が動くかを確かめ、結果をそのまま貼り付けられる Markdown で出力します。テスト用モデルはプログラムに埋め込んであるので、リポジトリのどこから実行しても動きます。

```bash
go run ./cmd/hwnpu-dev report -o report.md
```

- 見つかったデバイスごとに、そのバックエンド向けのモデルを厳格モードでコンパイルし、推論結果を fp32 の参照値と比べ、実行時間を測ります
- Level Zero loader の読み込み結果も載るので、Intel NPU が見つからないときの原因が分かります
- `-json report.json` で JSON も出力します。`-n` で計測の回数を、`-power low` などで電力ポリシーを、`-cache DIR` でコンパイルキャッシュを指定できます
- AMD では `-xclbin final.xclbin -insts insts.bin` を渡すと、mlir-aie の `programming_examples/basic/passthrough_dmas` をビルドした設計を実行して出力を照合します (引数の並びは opcode, instr, ninstr, 入力, 未使用, 出力)
- 失敗があると終了コード 2、NPU が見つからないと 1 で終わります

macOS では `go run ./cmd/hwnpu-dev powerbench` で、Core ML モデルの呼び出し 1 回あたりのエネルギーを powermetrics で測れます。最初に sudo のパスワードを一度だけ聞きます。詳しくは [MODEL_ARCHITECTURE.ja.md](MODEL_ARCHITECTURE.ja.md) を参照してください。

## モデルの変換と照合

`go run ./cmd/hwnpu-dev convert` は、Hugging Face のチェックポイント (safetensors か GGUF) を Python なしで Neural Engine 用の `.mlpackage` に変換します。対応するのは次のものです。移植の手順は `.claude/skills/port-architecture` に、元モデルの選び方と Neural Engine の制約は [MODEL_ARCHITECTURE.ja.md](MODEL_ARCHITECTURE.ja.md) にあります。

- LFM3 (d1-2B などの判定モデルを含む)、LFM3-VL の視覚エンコーダ、LFM3-Audio / LFM3-Omni の音声エンコーダと音声出力
- Gemma 4 (gemma-4-E2B-it)
- Qwen3.5 (Qwen3.5-4B などのテキスト部分。CPU での照合のみ)、Qwen3 (Qwen3-0.6B と 1.7B。`hwnpu-chat -m qwen3`)
- BERT 系エンコーダ、PLaMo-Embedding-1B
- PLaMo 3 (PLaMo 3 Translate 2B。CPU と OpenVINO の CPU での照合のみ。`hwnpu-chat -m plamo-3-translate-2b` で翻訳)
- Whisper (whisper-large-v3-turbo。音声エンコーダは `-arch whisper-encoder`、クロス注意のキーと値は `-arch whisper-cross`)

ほかのコマンド:

- `hwnpu-dev check`: 変換したグラフを CPU で fp32 のまま実行し、Hugging Face の実装の参照 (`tools/ref/ref_logits.py`) と層ごとに照合します。Mac は要りません
- `hwnpu-dev generate`: 変換した因果 LM で文章を生成します (トークナイザーも Go)。`-cache N` で変換すると KV キャッシュ付きになり、長い文章を書けます (M3 の LFM3-1.7B で、キャッシュ 1024 なら 1 トークン 20.6 ms)
- `hwnpu-dev formats`: 重みの形式 (int8、int4-g128、int4) ごとの精度と速さを NPU の上で比べます

## NPU を測る (probe)

`go run ./cmd/hwnpu-dev probe` は、生成したモデルで NPU を測ります。LLM が 1 トークンを生成するのと同じ形のモデルを Go で作り、まず 0.5B と 1B の重みで形式ごと (Apple は fp16、int8、lut4、int4、Intel は fp16、int8、int4) の速さを測ります (`bench`)。次にモデルを大きくしながら NPU から外れる境目を二分探索で絞り、境目のすぐ下のモデルを複数同時に読み込んで、上限がモデルごとか合計かを確かめます (`limit`)。Python もモデルのファイルも root も要らず、クローンしなくても次の 1 行で動きます。結果は `npuprobe-<機種>-<日時>.md` に書き出します。

```bash
go run github.com/shibukawa/hwnpu/cmd/hwnpu-dev@latest probe
```

`bench` か `limit` を付けると片方だけを行います。Apple で 1 トークンあたりのエネルギーも測るには `go run ./cmd/hwnpu-dev powerbench -decode 8,16 -gpu` を使います (sudo が要ります)。機種ごとの手順、結果の読み方、これまでの結果は [PROBE.ja.md](PROBE.ja.md) にまとめてあります。

Go の入っていない Windows 機で試すときは、Go のある機械でビルドした実行ファイルを持っていきます。

```bash
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o hwnpu-dev.exe ./cmd/hwnpu-dev
```

## 診断

```bash
go run ./cmd/hwnpu-dev info -model internal/fixtures/coreml/linear_relu_fp16.mlpackage
```
