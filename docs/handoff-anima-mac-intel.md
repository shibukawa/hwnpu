# 引き継ぎ: Anima (`hwnpu-image`) を Mac の Neural Engine と Intel の NPU で確かめる

作成: 2026-11-05。Linux (aix1、AMD の NPU) で実装し、CPU で確かめた Core ML と OpenVINO IR の書き出しを、実機で動かすためのメモ。設計と CPU での確認は [docs/design.md](design.md) の 16.69。

## 1. ここまでの状態

- ブランチ: `t3/anima-image` (#40)
- 何をするか: `hwnpu-image -convert` が、Anima のチェックポイントから Go だけで Core ML のパッケージ (`-backend coreml`) か OpenVINO IR (`-backend intel`) を書く。部品は AMD の IRON のグラフと同じ分け方:

| 部品 | 入力 | 出力 |
|---|---|---|
| `text` (Qwen3-0.6B と LLM アダプタ) | `qx`、`tx` (1, 512, 1024)、`qbias`、`tbias` (1, 1, 1, 512) | (1, 512, 1024) |
| `dit_time` | 時刻の正弦波 (1, 1, 2048) | (1, 1, 8192) |
| `dit_patch` | パッチ (1, n, 64) | (1, n, 2048) |
| `dit_blocks_0`、`_4`、… `_24` (4 ブロックずつ) | `x` (1, n, 2048)、`t`、`ctx` | (1, n, 2048) |
| `dit_final` | `x`、`t` | (1, n, 64) |
| `vae_s0`〜`vae_s9` | (1, C, H, W) | 次の段の入力、最後は (1, 3, 1024, 1024) |

  n は 1024 × 1024 で 4096。重みは既定で fp16 (`-text-format`、`-dit-format` で lut8 や int8 を選べる)。VAE は畳み込み (`conv`) をそのまま使う
- fp16 のための手当て: DiT の残差は 1/32 で持つ (実測の最大 2.7e5 → 8.4e3)、ゲートも 1/32。幅の正規化は行ごとに最大の絶対値で割ってからする (2 乗があふれない)
- Linux で確かめたこと (256 × 256、8 ステップ、diffusers の fp32 の CPU と同じノイズ):

| | 潜在のコサイン | 画像の PSNR |
|---|---|---|
| Core ML を CPU のインタプリタで (fp32) | 0.99985 | 37.4 dB |
| 同じく演算ごとに fp16 に丸める (`HWNPU_CPU_F16=1`) | 0.99899 | 32.0 dB |
| OpenVINO IR を OpenVINO の CPU で (f32) | 0.99985 | 37.9 dB |
| 同じく f16 の推論 (`HWNPU_OV_PRECISION=f16`) | 0.99478 | 25.5 dB |

- **まだ確かめていないこと (この引き継ぎの目的)**: Neural Engine と Intel の NPU に全部が載るか (hwnpu は 1 つでも CPU に落ちる部品があれば止まる)、コンパイルの時間、速さ、メモリ、1024 × 1024 の VAE が収まるか

## 2. 準備

Linux で使った `~/models/Anima` と同じ置き方 (docs/amd-npu-models.md の 11.4)。公式の `split_files` の 2 つと、Anima-Base-v1.0-Diffusers の設定、トークナイザー、VAE。

```sh
git fetch origin && git checkout t3/anima-image
go test ./internal/imagegen/... ./internal/tokenizer/ ./internal/milcpu/
```

## 3. Mac (M3、16 GB)

**2026-11-06 に確かめた** ([design.md](design.md) の 16.85)。1024 × 1024 は全部 Neural Engine で 1 枚 40 秒。そのままでは載らなかった `text` (fp16)、`dit_time`、`dit_patch` は形を変えた。既定の書き出しはテキストも DiT も int8。下は最初の手順の記録。

まず 512 × 512 で、全部が Neural Engine に載るかを見る。

```sh
go run ./cmd/hwnpu-image -convert ~/models/Anima -backend coreml -size 512 -model ~/models/anima-turbo-512.coreml
time go run ./cmd/hwnpu-image -model ~/models/anima-turbo-512.coreml -prompt "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms" -seed 1 -o out512.png
```

| 見るところ | 期待と、外れたときの手 |
|---|---|
| 読み込み | 各部品が `does not run on the NPU alone` で止まらない。止まる部品の演算を `go run ./cmd/hwnpu-check` か `npuprobe` で調べる。疑わしいのは `reduce_max` と `real_div` の行ごとの正規化 (幅 2048)、VAE の `tile` (階数 4) と大きな `conv` |
| 1 回目の読み込みの時間 | DiT の部品 (fp16 で 1 つ 570 MB) のコンパイル。2 回目はキャッシュで短くなるか |
| `step` の行 | 1 ステップの時間。注意は 1024 クエリずつ (`-qchunk`) |
| 画像 | 破綻していないか。ノイズは AMD と同じ作り方なので、1024 × 1024 で同じプロンプトとシードなら AMD の絵と構図が近いはず |

次に 1024 × 1024。DiT の自己注意 (4096 × 4096、16 ヘッド) と、VAE の 1024 × 1024 × 96 の畳み込みが Neural Engine の大きさの上限に掛かるかを見る。メモリが足りなければ `-dit-format lut8` (DiT が 2 GB)。

## 4. Intel (Core Ultra)

```sh
go run ./cmd/hwnpu-image -convert ~/models/Anima -backend intel -size 512 -model ~/models/anima-turbo-512.intel
time go run ./cmd/hwnpu-image -model ~/models/anima-turbo-512.intel -prompt "..." -seed 1 -o out512.png
```

ドライバ内コンパイラが IR を受け付けるか、NPU のメモリに入るか、時間。`-dit-format int8` も試す。

## 5. 記録すること

部品ごとの配置とコンパイルの時間、1 ステップの時間、VAE の時間、全体の時間、メモリ (`/usr/bin/time -l`)、画像。design.md の 16.69 に書き足す。
