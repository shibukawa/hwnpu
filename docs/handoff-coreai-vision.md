# 引き継ぎ: 画像を読むモデルの Core AI と、SigLIP 2 の Mac の形を M1、M2、M4 で確かめる

作成: 2026-11-14。M3 (16 GB、macOS 27.2) で入れた 2 つ ([design.md](design.md) の 16.182 と 16.183) の、M3 以外での確認。全部で 30 分ほど。

- 画像を読むモデル (PP-OCR、RTMDet、RTMO、YuNet、DA3、MoGe-2、EfficientSAM) を Core AI で (`--backend coreai`)
- SigLIP 2 (`hwnpu-image label`、`embed`) を Apple の Neural Engine で (Core ML と Core AI)
- `hwnpu-image segment` の符号器の既定を、Apple では Neural Engine に

## 1. ここまでの状態 (M3)

- どの部品も Core ML と Core AI の両方で全部 Neural Engine に載り、CPU のインタプリタ (`milcpu`) とのコサインは 0.9997 以上
- PP-OCR の検出器と認識器は、定数による割り算 (`divide_v1`) のままだと Core AI の Neural Engine でだけ誤る (検出器のコサイン 0.05)。`ailower` が逆数との積に書くので、今は起きない。**チップによって出方が変わるかは分かっていない**
- 起動は、両方のキャッシュが当たっていれば Core ML と Core AI で同じ (1 コマンド 0.05〜0.9 秒)
- SigLIP 2 の埋め込みは、Transformers (float32) とのコサインが画像 0.9998 以上、文 0.99999 以上
- EfficientSAM-S の符号器は Neural Engine で 0.36〜0.43 秒 (hwnpu の評価器では 33 秒)、mask は CPU のものと IoU 0.9994

## 2. 確かめること

測る前に、ほかの hwnpu のプロセスや `mediaanalysisd` が Neural Engine を使っていないことを確かめる。絵は何でもよい (文字のある絵、人や物の写る絵、顔の写る絵を 1 枚ずつ)。

### 2.1 コマンドの結果が Core ML と Core AI で同じか

```sh
for b in coreml coreai; do
  go run ./cmd/hwnpu-image ocr --text text.png --backend $b
  go run ./cmd/hwnpu-image detect street.jpg --backend $b
  go run ./cmd/hwnpu-image pose street.jpg --backend $b
  go run ./cmd/hwnpu-image face-detect --device npu face.jpg --backend $b
  go run ./cmd/hwnpu-image depth -m da3:base --raw -o out-$b photo.jpg --backend $b
  go run ./cmd/hwnpu-image depth -m moge:vits --raw -o out-$b photo.jpg --backend $b
  go run ./cmd/hwnpu-image segment -p X,Y -o out-$b photo.jpg --backend $b
  go run ./cmd/hwnpu-image label -l "a fox,a cat,a dog,a sign" photo.jpg --backend $b
done
```

- 期待: どのコマンドも `all on the NPU` (または `on the NPU`) で、エラーなし。初回のコンパイルは 1 部品 0.2〜38 秒 (M3)
- 期待: OCR の文字は同じ。物、人、顔の数とラベルは同じで、スコアの差は 0.01 以下。深さ (`*.depth.f32`) のコサイン 0.99999 以上。mask の面積の差は 0.1 % 以下。ラベルの確率の差は 0.01 以下
- `segment` は `--device` なしで符号器が Neural Engine で動く。**M1 では、符号器の注意の行列 (4096 × 4096 × 頭の数) が 64 MiB を超えるので、時間を記録する** (M3: S 0.36〜0.43 秒、new-model-checklist 3.1)。`-m efficientsam:ti` (8 GB の機械の既定) も

### 2.2 部品を CPU のインタプリタと比べる

```sh
go build -o /tmp/aicheck ./tools/coreai/aicheck
M=~/Library/Caches/hwnpu/models
P="$M/PP-OCRv6_small_det_onnx/*.mlpackage $M/PP-OCRv6_small_rec_onnx/*.mlpackage $M/RTMDet-ONNX/*.mlpackage $M/RTMO-s/*.mlpackage $M/YuNet/*.mlpackage $M/DA3-BASE/da3-base-v1.coreml/*/*.mlpackage"
/tmp/aicheck run -cpu -calls 10 $P            # Core AI
/tmp/aicheck run -cpu -coreml -calls 10 $P    # Core ML
```

- 期待 (M3): どれも `ane=true`、`y cos` が 0.9997 以上で、Core AI と Core ML がほぼ同じ (検出器 0.99990〜0.99993、認識器 0.99974〜0.99988、RTMDet 0.99994、RTMO 0.999999、YuNet 0.999998、DA3 0.99999 以上)

### 2.3 割り算の誤りがそのチップでも出るか (記録だけ)

```sh
HWNPU_AILOWER_DIVIDE=1 AICHECK_WORST=1 /tmp/aicheck run -cpu $M/PP-OCRv6_small_det_onnx/det-r1-fp16-960x960.mlpackage
HWNPU_AILOWER_DIVIDE=1 HWNPU_AILOWER_CUT=133 AICHECK_WORST=1 /tmp/aicheck run -cpu $M/PP-OCRv6_small_det_onnx/det-r1-fp16-960x960.mlpackage
```

- M3: 1 行目は `y cos 0.05`。2 行目は `real_div_98 cos 0.984`、相対 RMS 1.12 で、値は全体が正しい値の約 2 倍 (√2 で割るはずが √2 倍になる。design.md 16.183)
- 出なければ、Neural Engine のコンパイラの版かチップによる。どちらでも回避 (積) はそのままにする。出方が違えば (別の倍率、正しい) design.md に書く

### 2.4 起動を 3 回ずつ

2.1 のコマンドを、それぞれのバックエンドで 1 回走らせた直後に、新しいプロセスで 3 回ずつ (`time`)。

- 期待: 2 つのバックエンドでほぼ同じ (M3: ocr 0.2 秒、detect 0.2 秒、pose 0.2 秒、face-detect 0.06 秒、depth 0.3〜0.5 秒、segment 0.8〜0.9 秒、label 0.15 秒)
- Core ML だけ数秒〜十数秒かかる回があれば、コンパイルのキャッシュ (約 4 GiB) から外れている。直前に何を動かしたかと一緒に記録

### 2.5 SigLIP 2 を参照と比べる

```sh
go run ./cmd/hwnpu-image embed -m siglip2:base a.jpg b.jpg -t "a red fox in the snow" -t "雪の中のキツネ" > emb.json
python tools/ref/ref_siglip2.py ~/Library/Caches/hwnpu/models/siglip2-base-patch16-224 ref --images a.jpg b.jpg --texts "a red fox in the snow" "雪の中のキツネ"
```

- 参照には、チェックポイントのフォルダに `preprocessor_config.json`、`tokenizer_config.json`、`special_tokens_map.json` も要る (Hugging Face の同じリポジトリから落とす。hwnpu は使わない)
- 期待: `emb.json` の埋め込みと `ref.npz` の `image_embeds`、`text_embeds` のコサインが 0.999 以上 (M3: 画像 0.99996 以上、文 0.99999 以上)。`siglip2:large` も (16 GB の機械の既定)
- 8 GB の機械で `label` の既定 (base) のメモリ (`the Neural Engine holds`、`/usr/bin/time -l` の peak memory footprint) を記録し、models.yaml の `memory` と比べる

## 3. 記録する場所

- 数値は design.md の 16.183 に足す (チップごとの行)
- models.yaml の SigLIP 2 と EfficientSAM の `memory` が違えば直す (new-model-checklist 11.2)
