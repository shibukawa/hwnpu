# Chroma1 (Chroma1-Flash、Chroma1-HD)

## 状態

- **削除済み (コードなし)**。2026-11-24 に、Apple と AMD の両方から外しました。
- 外した理由は「問題と外した理由」に書きました。M1 では Apple のコンパイラの不具合で DiT が仕上がりません。FLUX.2 klein 4B が同じ枠を、軽く速く埋めています。

## モデル

| モデル | リポジトリ | ライセンス | 中身 |
|---|---|---|---|
| Chroma1-Flash | `lodestones/Chroma1-Flash` | Apache-2.0 | FLUX.1 schnell の 8.9B の DiT。変調は小さな FFN (approximator)。文は T5-XXL、VAE は FLUX.1 のもの |
| Chroma1-HD | `lodestones/Chroma1-HD` | Apache-2.0 | 蒸留していない本体。Flash と同じ形で、重みだけが違う |

- 学習で検閲をしていません (モデルカードに "Fully uncensored" とあります)。models.yaml では `content: [nsfw]` にしていて、`image` の用途の既定には選ばれませんでした。
- T5-XXL と VAE は FLUX.1 schnell と同じバイト列です。

## どの config で動かしていたか

最後に models.yaml と model-recipes.yaml にあったのは `05c406c9` です。`git show 05c406c9:models.yaml` の `Chroma1-Flash:` と `Chroma1-HD:` を見てください。

| | Chroma1-Flash | Chroma1-HD |
|---|---|---|
| family / tag | `chroma` / `1-flash` | `like: Chroma1-Flash` / `1-hd` (`chroma:1-hd` で指定) |
| machine | 16 | 0 (家族の名前だけでは選ばれない) |
| memory (GB) | AMD 3.9、Apple 4.3 (Core ML のときの値) | Flash と同じ |
| bytes | AMD 16.24e9、Apple 2.75e10 | Flash と同じ |
| 描き方 | Heun 8 ステップ、ガイダンスなし | DPM-Solver++ 2M 25 ステップ、CFG 3 (σ ≥ 0.6 だけ) |
| AMD の書き出し (models.yaml `exports`) | `chroma1-flash.amd-npu2: v1 sha256:0f171d8a…` | `chroma1-hd.amd-npu2: v1 sha256:370f1bdb…` |

- **AMD のレシピ** (`model-recipes.yaml`、`05c406c9`):
  - `amd.version: 1`、`script: iron_chroma.py`、`args: ["{in}", "{out}", --size, "{sizes}"]` です。
  - 8 つのアスペクト比 (`aspect.Of(16)`) を 1 つの書き出しにまとめていました。HD は `*chroma-amd` で同じレシピです。
- **Apple の形式** (Core AI):
  - T5-XXL は lut8 で、FF の出力 `wo` だけ fp16 です (`TextFormatFor wo=fp16`)。全部 lut8 にすると、1 行が cosine 0.766 まで落ちました。
  - DiT は lut8 です。int8 は σ = 0.25 で崩れました。approximator と VAE は fp16 です。
  - プログラムは 33 本です。
    - `text_0`〜`text_21`: 3 層ずつ 8 本
    - `mods`
    - `dit_double_*`: 4 ブロックずつ 5 本
    - `dit_single_*`: 8 ブロックずつ 5 本
    - `vae_s0`〜`s9`
  - プリビルドの規則 (model-recipes.yaml `apple.prebuild`):
    - h13g (M1) の `dit_single_*` は端末でコンパイル (2026-11-23)。
    - h14g (M2) の `dit_single_0`、`8`、`16` も端末でコンパイル (2026-11-23)。
    - 8 サイズにした版 (#314) では、h13g の `dit_double_*` も端末に回す規則を足しました (2026-11-24)。

## 実装状況 (外した時点)

### Apple (Core AI)

- **main (`05c406c9`):** 1 サイズに 1 つの書き出しです (1024² と 512² が別)。
  - M1 の 1024² は 5 分 36 秒でした (T5 2.1 秒、DiT 1 回 21.7 秒、VAE 4.6 秒、NE 2.53 GiB)。
  - HD は Core AI では測っていません。Core ML では 3 分 39 秒でした。
- **8 サイズの版 (#314、下書きのまま閉じた):** 8 サイズを 1 つの書き出しにして、重みは 1 回分にしました。
  - 書き出し名は `chroma1-flash-8sizes-4d02f40a-lut8+wofp16-lut8-v3.coreai` で、19.3 GB です。HD も同じ大きさです。
  - 512² の attention は 768 行ずつにしました (`headChunk`)。分けないと FLUX.2 klein と同じ形で M3 のプリビルドが落ちるからです (#301)。
  - プリビルドは途中で止めました。止めた時点でできていたもの (Flash だけ。HD は未着手):

    | チップ | 二重ブロック | 単ブロック |
    |---|---|---|
    | h13g | 5 本 | なし |
    | h14g | 5 本 | なし |
    | h15g | 5 本 | `dit_single_0`、`16`、`24` |

  - 1 本のコンパイル時間は、h13g が 41 分〜1 時間 36 分、h14g が 26〜59 分、h15g が 15〜21 分でした。単ブロックは h15g で 57 分〜1 時間 8 分でした。
  - 絵は 8 サイズとも 1 枚も描けていません。M1 で `dit_double_0` と `dit_double_4` を端末でコンパイルしたところで止めました。2 時間 7 分と 2 時間 1 分かかりました。
- 精度 (Core ML、M1、512²、Flash、3 つの文 × 2 つの seed、fp32 の `ref_chroma_lean.py` と比べて):
  - 潜在の cosine は 0.987〜0.997、29.9〜37.9 dB でした。構図、顔、看板の位置は同じでした。
  - HD の 1024² では、DiT の最初の 4 回の速度の cosine が 0.99983〜0.99991 でした。

### AMD (XDNA 2)

- 8 サイズを 1 つの書き出しにしていました。Flash は 12 分 2 秒で、ピークの RSS は 26 GB、大きさは 18.85 GB です。HD は 9 分 19 秒で 18.8 GB です (T5 と VAE は Flash からハードリンク)。
- 描く時間とピークのメモリ:

  | | 1024² | 512² | 長方形 | ピーク (NPU + RSS) |
  |---|---|---|---|---|
  | Flash (Heun 8) | 3 分 14 秒 | 1 分 1 秒 | 3 分 12〜15 秒 | 3.47〜3.62 GiB |
  | HD (DPM 25、CFG 3) | 8 分 34 秒 | 2 分 32 秒 | 8 分 29〜37 秒 | 3.46〜3.62 GiB |

- 文の mask: diffusers の ChromaPipeline は pad も attention で見ます。Chroma は pad を (1 つを残して) 隠して学習されているので、NPU では `nk` でキーを切っていました。
- 精度: 最初の速度は 0.99985、8 ステップ後の潜在は 0.9988、40.8 dB でした (512²)。

## ロジックが残っているコミット

| 何 | コミット | 場所 |
|---|---|---|
| main で最後に Chroma があった版 | `05c406c9` | `internal/imagegen/chroma/` (Go の変換、描画、MIL の部品)、`ai/imagegen/chroma.go`、`tools/amd/iron_chroma.py`、`tools/ref/ref_chroma*.py`、`tools/coreai/chromafake/`、models.yaml、model-recipes.yaml |
| 8 サイズの書き出し、512² の 768 行の attention、M1 の二重ブロックの規則 | `3aa2977f` (ブランチ `t3/chroma-8sizes-v3`、PR #314。ブランチを消しても `refs/pull/314/head` に残る) | 同じ場所。`internal/prep/image.go` の 8 サイズ化も含む |
| Apple で最初に動いた版 | `900401c8` (2026-11-13) | |
| AMD で最初に動いた版 | `9c01f3fb` (2026-11-11、#123) | |
| AMD の 8 サイズ | `94cac92c` (2026-11-17、#213) | |

**消さずに残したもの (FLUX が使っている)**
- **FLUX.1 の VAE の MIL:** `internal/imagegen/chroma/mil` から `internal/imagegen/fluxvae` に移しました。FLUX.2 klein と FLUX.3 klein が使っています。コードは1行も変えていないので、書き出しのバイト列は変わりません。
- **AMD の部品:** `tools/amd/iron_chroma_parts.py` は残しています。FLUX.2 と FLUX.3 の AMD の書き出しが、`FluxVaeStage` と `HOST_NORM` を使っているからです。T5 の部分は Chroma 専用ですが、AMD 側の都合で手をつけていません。

## 問題と外した理由

1. **M1 で DiT が仕上がらない:** M1 (h13g) では、8 サイズの DiT のプリビルドを開くと、`ANELargeModelCompilerService` が毎回 SIGSEGV で落ちます。
   - 落ちる場所は二つで、`ZinCompilerCoreLLIROnline::AllocateMemoryPools` から `ZinIrTensor::GetParentTensors` に入ったところと、`AneHloToZinPass` のハッシュ表です。
   - Core AI は端末でのコンパイルに戻らず、失敗にします。hwnpu は自分で端末コンパイルに回しますが、1 部品 2 時間かかります。
   - 単ブロックはもともと端末に回しているので、M1 では DiT の 10 部品すべてが端末でのコンパイルになり、初回は 15〜20 時間かかる見込みでした。
   - Apple には Feedback Assistant で報告しました (2026-11-24)。
2. **M2 も重い:** h14g では単ブロックのうち 3 本が Neural Engine に載らず、1 本 1.5〜3 時間の端末コンパイルになります。
3. **大きい:** 書き出しは 1 モデル 19 GB あり、Flash と HD の 2 つで NAS を 38 GB 使います。プリビルドを作る時間も長く、FLUX.3 klein に移るのが遅れていました。
4. **代わりがある:** FLUX.2 klein 4B も Apache-2.0 です。M1 で 1024² が 41 秒で描け、編集もできます。Chroma にしかないのは「検閲なし」くらいで、hwnpu の既定の用途には向きません。

## 再開するときの手順と注意

1. `3aa2977f` (PR #314) から始めます。main の `05c406c9` は 1 サイズずつの書き出しの古い形です。
2. FLUX.1 の VAE は `internal/imagegen/fluxvae` に移っているので、`chroma/mil` の VAE の部分は fluxvae を使うように直します。
3. models.yaml と model-recipes.yaml に、`05c406c9` の Chroma の項目を戻します。8 サイズにするなら #314 の項目を戻します。Apple の版 (`apple.version`) と code ハッシュの仕組み (#315) に合わせて、`apple.version` と `code` も書きます。
4. M1 のクラッシュが macOS で直ったか、先に確かめます。h13g で `dit_double_0` のプリビルドを 1 本作り、`tools/coreai/caiload` で開くのがいちばん早い方法です。直っていなければ、M1 は対象外にするのが現実的です。
5. 元の重みは NAS の `original/lodestones/Chroma1-Flash` と `Chroma1-HD` に残しています。書き出しとプリビルドは 2026-11-24 に消しました。

## 設計文書から移した記録 (英語の原文)
Chroma を外したときに、設計文書 (`docs/design/`) から消した行です。表の見出しは元の文書 (`05c406c9`) にあります。日本語訳は同じコミットの `.ja.md` にあります。

### docs/design/image-apple.md

```text
| Chroma1-Flash, Chroma1-HD | yes | no | `internal/imagegen/chroma` |
| Chroma1 | one export per size (512² and 1024² apart) |
| Chroma | T5-XXL | 3 layers a program, 8 (`text_0`…`text_21`), the last with the final norm and the DiT's `context_embedder` | an empty negative prompt (`</s>` only, Chroma1-HD's default) is precomputed on the host in float32 at conversion (`text/empty.bin`); 512 rows, `--text-rows 256` (for prompts ≤ 255 tokens: −7 % time) is not default |
## Chroma1 (Flash, HD)

- Programs (33): `text_*` T5 (8, lut8 + wo fp16), `mods` (approximator, fp16, once per σ before the steps: Flash's Heun 8 steps = 8 calls; shared by Heun's two evaluations and CFG's two), `dit_double_*` (4 double blocks a program, 5; the first with x_embedder), `dit_single_*` (8 single blocks, 5; the last with norm_out and proj_out), `vae_s0`–`s9` (fp16).
- Rows are [image, text]. Text pad is masked by a key bias (`kbias`: only the prompt tokens and one pad at 0). FLUX's RoPE pairs (2i, 2i+1) become `blocks.ApplyRope`'s (i, i+64) by permuting the q and k weight rows (`mil.RopePerm`); cos/sin are constants.
- CFG's two calls run group by group back to back.
- `ExportVersion` 2 changed only the DiT groups: T5, approximator and VAE link from version 1 exports. T5 is the same for Flash and HD except the last group (context_embedder); the VAE is the same per size. HD 1024² export from Flash's: about 10 GB new.
- Flags (`convert`): `--text-format-for`, `--dit-format-for`, `--dit-quant` (uniform, kmeans, kmeans-minmax), `--text-rows`, `--keep-dit` (on Core AI only marks the groups checked).
- MLComputePlan placement check timeout is 30 min (`planTimeout`, Core ML): mixed lut6/lut8 1024² single groups took 19–27 min.
- Accuracy (Core ML, M1, 512², Flash, 3 prompts × 2 seeds vs fp32 `ref_chroma_lean.py`): latent 0.987–0.997, 29.9–37.9 dB, same compositions, faces, sign placement. HD 1024² DiT on the reference's first 4 calls: velocity cosine 0.99983–0.99991. The current attention form vs the heads-together form (512², 3 prompts): 0.999–0.9996, 40.5–41.2 dB.

| Chroma1-Flash (Heun 8 steps, 15 DiT calls) | M1 16 GB | 1 m 25 s (DiT call 5.3 s) | 5 m 36 s (T5 2.1 s, DiT call 21.7 s, VAE 4.6 s; NE 2.53 GiB) | models.yaml 4.3 GB is Core ML's process footprint, not re-measured |
| Chroma1-HD (DPM-Solver++ 2M 25 steps, CFG 3 at σ ≥ 0.6) | M1 16 GB | 3 m 39 s (CFG step 10.0 s, other 5.4 s; NE 2.49 GiB) | not run on Core AI | |
| Chroma: DiT MLP lut6 | the same pictures, not faster at 1024²; only 1.7 GB of disk/cache saved |
| Chroma: DiT int8, MLP lut4 | stripes / changes pose or collapses |
| Chroma: T5 fp16 | 9.5 GB against 5.5, 12.7 s a prompt against 9.5 (Core ML); lut8 + wo fp16 gives the same pictures |
| Chroma: T5 all lut8 | one prompt row at 0.766 |
| Chroma: `--text-rows 256` as default | −7 % time, cuts long prompts |
| Chroma: `t5Up` trick on the DiT | Core ML folded it; broke the last single group |
- Chroma: memory on Core AI (models.yaml still Core ML's), Chroma1-HD at 1024² on Core AI, comparison with the fp32 reference after the attention change, why the second CFG call at 1024² was cheaper on Core ML (23 vs 35 s).
```

### docs/design/image-apple.md (weight formats)

```text
| Chroma T5-XXL | lut8, `wo` fp16 (`TextFormatFor wo=fp16`) | `wo` takes the outlier channels; 5.5 GB against fp16's 9.5 GB, the same pictures (latent cosine 0.993–0.997 to fp32, 0.996–0.999 to fp16 T5); lut8 throughout put one row at 0.766 |
| Chroma DiT | lut8 | int8 drifts at σ = 0.25 (rel. 14 %, stripes); lut8 2.8 %, latent 0.9965, 37.9 dB |
```

### docs/design/image-models.md

```text
| Chroma1-Flash (`chroma:1-flash`) | FLUX.1 schnell's 8.9B DiT with an FFN for its modulations (approximator), T5-XXL, FLUX VAE | Heun 8 (15 DiT calls), none | Apache-2.0, nsfw ("fully uncensored") | 16 | 3.9 / 4.3 (+6.9 GiB on the Neural Engine) | AMD 1024² 3 min 14 s, 512² 1 min; Apple M1 1024² ~9.5 min |
| Chroma1-HD (`chroma:1-hd`) | the undistilled Chroma1 | DPM-Solver++ 2M 25, CFG 3 at σ ≥ 0.6 | as Flash | 0 (by name only) | as Flash | AMD 1024² 8 min 34 s |
- T5-XXL and the VAE of Chroma1 are FLUX.1 schnell's byte for byte, shared when present; Chroma1-HD's export reuses Flash's.
| Chroma1-Flash | Heun 8 | `--sampler euler` (8 calls) |
| Chroma1-HD | DPM-Solver++ 2M 25, CFG 3 at σ ≥ 0.6 | written in the export's `model.json` (`sampler`, `steps`, `cfg_sigmas`) |
| Chroma `--text-rows 256` as default | −7 % at 512², exact only for prompts ≤ 255 tokens |
```

### docs/design/image-amd.md

```text
| Chroma1 | qk | mid |
| Chroma1 | T5-XXL | q8, HIGHEST, erf GELU (IRON's tanh table took the worst row to 0.986). 6 groups of 4 layers in one bank (`--t5-group 4`): each group outputs the residual and the final RMSNorm; prompt and negative run back to back per group. Peak 5.9 → 4.6 GB at 1024², T5 4.2–5 s |
- Chroma's tokenizer: Chroma1 has only `spiece.model`; the exporter reads its protobuf directly (no sentencepiece in the IRON env) and writes Anima's `t5_tokenizer.json` form (`t5_tokenizer_json`; identical to Anima's). FLUX's own tokenizer.json uses a Sequence normalizer Go cannot read.
| Chroma1 | [image, text 512, pad] | to 512; the pad follows the text's pad, so `nk` (image + text rows) already cuts it. Double blocks' image stream runs at the image rows rounded to 256 (`img_rows`: `flm.ProjectionGEMM`'s M is a multiple of 256), the text stream at 512, each writing only its rows. `dit_in`, `dit_out`, `doubles` per size (`_WxH`); `singles` per row count |
### Chroma1-Flash, Chroma1-HD

- DiT: 19 double blocks (prime: 1 a bank) and 38 single (2 a bank). The approximator (5 layers × 5120, 512 rows a call: 344 + pad) gives every sigma's modulations before the steps; HIGH, not HIGHEST (K = 5120 breaks the buffer descriptor); RMSNorm weights wider than 4096 folded into the next linear's columns.
- The text mask: diffusers' ChromaPipeline passes a float 0/1 mask that gets added to the scores, so padding is still attended. Chroma was trained with padding masked (but one), so the NPU cuts keys at `nk`; the reference matches with `ref_chroma.py --mask keys` (applied after T5 runs, or T5's bias mask is lost). First velocity 0.99985, 8-step latent 0.9988, 40.8 dB (512²).
- Export of eight sizes: 12 min 02 s, peak RSS 26 GB, 18.85 GB; HD 9 min 19 s, 18.8 GB (T5 and VAE hard-linked to Flash's).

| | 1024² | 512² | rectangles | Peak (NPU + RSS) |
|---|---|---|---|---|
| Flash, Heun 8 | 3 min 14 s | 1 min 01 s | 3 min 12–15 s | 3.47–3.62 GiB (512² 1.50) |
| HD, DPM 25, CFG 3 | 8 min 34 s | 2 min 32 s | 8 min 29–37 s | 3.46–3.62 GiB (512² 1.53) |

- Sizes whose rows fit 4096 are faster (768 × 1152: 2 min 44 s, 3.0 GiB) but the sizes stay `aspect.All`'s.
```

### docs/design/memory-and-power.md (AMD, one 1024² image)

```text
| Model | Time (prep / run) | Energy (prep / run) | Minus idle (prep / run) | Bank re-reads |
|---|---|---|---|---|
| Chroma1-Flash | 4m22s (23 / 238) | 4.45 kJ (345 / 4.10 k) | 182 J / 2.49 kJ | 573×, 142 GB, 14 s |
| Chroma1-HD | 11m20s (32 / 648) | 11.89 kJ (493 / 11.39 k) | 277 J / 7.02 kJ | 953×, 234 GB, 22 s |
```
