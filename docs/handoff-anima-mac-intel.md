# 引き継ぎ: Anima (`hwnpu-image`) を Intel の NPU で確かめる

作成: 2026-11-05、整理: 2026-11-21。設計と CPU での確認は [docs/design/image-apple.md](design/image-apple.md)。

**Mac は済み**: M3 の Neural Engine で 1024 × 1024 が全部載り 1 枚 40 秒 ([image-apple](design/image-apple.md))、M1 と M2 も ([image-apple](design/image-apple.md), [image-models](design/image-models.md))。Mac の手順はこの文書から外した。残りは Intel だけ。

## 1. ここまでの Intel (Core Ultra 5 135U)

- OpenVINO の CPU では照合済み (256 × 256、8 ステップ、diffusers の fp32 と同じノイズ): f32 で潜在のコサイン 0.99985、PSNR 37.9 dB。f16 の推論 (`HWNPU_DEV=ov.precision=f16`) では 0.99478、25.5 dB
- fp16 で約 4.6 GB を持ったところで `ERROR_OUT_OF_HOST_MEMORY` ([intel](design/intel.md))。閉じて読み直す形を入れた
- 1024 × 1024 は、4 ブロックの群で Windows の TDR (2 秒) に当たり `DEVICE_LOST` ([image-apple](design/image-apple.md))。群を自動 (`anima.DefaultGroup`、768 と 1024 で 1 ブロック) にした
- 1 ブロックの群にした後: 読み込みの dit_blocks_20 で「bad allocation」(ドライバのコンパイラの std::bad_alloc)。`types.IsOutOfMemory` がこれもメモリ不足とみなし、ほかの部品を閉じて読み直すようにした。**その後の実機の確認はしていない**

## 2. 確かめること

```sh
hwnpu-image generate --size 1:1s "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms" --seed 1 -o out512.png
hwnpu-image generate "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms" --seed 1 -o out1024.png
```

- 512 × 512 と 1024 × 1024 で、全部の部品が読み込めるか (読み直しが働くか)、`DEVICE_LOST` や「bad allocation」で止まらないか
- ドライバ内コンパイラの時間、1 ステップの時間、VAE の時間、全体の時間、NPU のメモリ
- `--dit-format int8` と `int4` も試す (Intel の既定は fp16。メモリが減って読み直しが要らなくなるか)
- 絵が破綻していないか (同じプロンプトと種の Mac / AMD の絵と構図が近いはず)

## 3. 記録すること

部品ごとの読み込みとコンパイルの時間、1 ステップ、VAE、全体の時間、メモリ、画像。docs/design/ の話題のファイル ([design/README.md](design/README.md)) に今の状態として書き、USAGE_IMAGE.md / .ja.md の「Intel の NPU では未確認」と [docs/design/performance.md](design/performance.md) / .ja.md の残りの表の Anima の行を直す。
