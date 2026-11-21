# 引き継ぎ: 画像を読むモデルの Core AI と SigLIP 2 を M2、M4 で確かめる

作成: 2026-11-14、整理: 2026-11-21。M3 (16 GB、macOS 27.2) で入れた 2 つ ([docs/design/vision-models.md](design/vision-models.md)) の、ほかのチップでの確認。全部で 30 分ほど。

- 画像を読むモデル (PP-OCR、RTMDet、RTMO、YuNet、DA3、MoGe-2、EfficientSAM) と SigLIP 2 (`hwnpu-image label`、`embed`) を Core AI で
- `hwnpu-image segment` の符号器は Apple では Neural Engine

**済んだこと**

- M3: どの部品も全部 Neural Engine、CPU のインタプリタ (`milcpu`) とのコサイン 0.9997 以上。SigLIP 2 は Transformers (float32) と画像 0.9998 以上、文 0.99999 以上。EfficientSAM-S の符号器は 0.36〜0.43 秒
- M1: `hwnpu-image` を Core AI だけにしたとき ([apple-coreai](design/apple-coreai.md))、全部のコマンドを既定のまま走らせ、main の Core ML と同じ結果。MoGe-2 の dense が GPU に回ったのは直した
- `hwnpu-image` から Core ML (`--backend coreml`) は外れたので、前の版の「Core ML と Core AI でコマンドの結果と起動の時間を比べる」はもうしない

残りは **M2 と M4** (と、[apple-coreai](design/apple-coreai.md) の残りの「M2 と M3 の初回」)。

## 1. コマンドが動くか

測る前に、ほかの hwnpu のプロセスや `mediaanalysisd` が Neural Engine を使っていないことを確かめる。絵は何でもよい (文字のある絵、人や物の写る絵、顔の写る絵を 1 枚ずつ)。

```sh
go run ./cmd/hwnpu-image ocr --text text.png
go run ./cmd/hwnpu-image detect street.jpg
go run ./cmd/hwnpu-image pose street.jpg
go run ./cmd/hwnpu-image face-detect --device npu face.jpg
go run ./cmd/hwnpu-image depth -m da3:base --raw -o out photo.jpg
go run ./cmd/hwnpu-image depth -m moge:vits --raw -o out photo.jpg
go run ./cmd/hwnpu-image segment -p X,Y -o out photo.jpg
go run ./cmd/hwnpu-image label -l "a fox,a cat,a dog,a sign" photo.jpg
```

- 期待: どれも `all on the NPU` (または `on the NPU`) で、エラーなし。初回のコンパイルは 1 部品 0.2〜38 秒 (M3)。初回の時間を記録する
- 2 回目からの起動 (新しいプロセスで 3 回ずつ、`time`) の M3 の値: ocr 0.2 秒、detect 0.2 秒、pose 0.2 秒、face-detect 0.06 秒、depth 0.3〜0.5 秒、segment 0.8〜0.9 秒、label 0.15 秒
- `segment` は `-m efficientsam:ti` (8 GB の機械の既定) も

## 2. 部品を CPU のインタプリタと比べる

```sh
go build -o /tmp/aicheck ./tools/coreai/aicheck
M=~/Library/Caches/hwnpu/models
P="$M/PP-OCRv6_small_det_onnx/*.mlpackage $M/PP-OCRv6_small_rec_onnx/*.mlpackage $M/RTMDet-ONNX/*.mlpackage $M/RTMO-s/*.mlpackage $M/YuNet/*.mlpackage $M/DA3-BASE/da3-base-v1.coreml/*/*.mlpackage"
/tmp/aicheck run -cpu -calls 10 $P
```

- 期待 (M3): どれも `ane=true`、`y cos` が 0.9997 以上 (検出器 0.99990〜0.99993、認識器 0.99974〜0.99988、RTMDet 0.99994、RTMO 0.999999、YuNet 0.999998、DA3 0.99999 以上)
- `.mlpackage` は `--cpu` 用に残る Core ML の書き出し。無ければ `hwnpu-image convert --backend coreml` で作るか、`.aimodel` を渡す

## 3. 割り算の誤りがそのチップでも出るか (記録だけ)

```sh
HWNPU_AILOWER_DIVIDE=1 AICHECK_WORST=1 /tmp/aicheck run -cpu $M/PP-OCRv6_small_det_onnx/det-r1-fp16-960x960.mlpackage
HWNPU_AILOWER_DIVIDE=1 HWNPU_AILOWER_CUT=133 AICHECK_WORST=1 /tmp/aicheck run -cpu $M/PP-OCRv6_small_det_onnx/det-r1-fp16-960x960.mlpackage
```

- M3: 1 行目は `y cos 0.05`。2 行目は `real_div_98 cos 0.984`、相対 RMS 1.12 で、値は全体が正しい値の約 2 倍 (√2 で割るはずが √2 倍になる。[docs/design/vision-models.md](design/vision-models.md))。M2 の `aicheck rules -only div` ([apple-neural-engine](design/apple-neural-engine.md)) も合わせて見る
- どちらでも回避 (`ailower` が逆数との積に書く) はそのまま。出方が違えば (別の倍率、正しい) docs/design/ の話題のファイル ([design/README.md](design/README.md)) に書く

## 4. SigLIP 2 を参照と比べる

```sh
go run ./cmd/hwnpu-image embed -m siglip2:base a.jpg b.jpg -t "a red fox in the snow" -t "雪の中のキツネ" > emb.json
python tools/ref/ref_siglip2.py ~/Library/Caches/hwnpu/models/siglip2-base-patch16-224 ref --images a.jpg b.jpg --texts "a red fox in the snow" "雪の中のキツネ"
```

- 参照には、チェックポイントのフォルダに `preprocessor_config.json`、`tokenizer_config.json`、`special_tokens_map.json` も要る (Hugging Face の同じリポジトリから落とす。hwnpu は使わない)
- 期待: `emb.json` の埋め込みと `ref.npz` の `image_embeds`、`text_embeds` のコサインが 0.999 以上 (M3: 画像 0.99996 以上、文 0.99999 以上)。`siglip2:large` も (16 GB の機械の既定)
- 8 GB の機械があれば、`label` の既定 (base) のメモリ (`the Neural Engine holds`、`/usr/bin/time -l` の peak memory footprint) を記録し、models.yaml の `memory` と比べる

## 5. 記録する場所

- 数値は [docs/design/vision-models.md](design/vision-models.md) に足す (チップごとの行)
- models.yaml の SigLIP 2 と EfficientSAM の `memory` が違えば直す (new-model-checklist 11.2)
