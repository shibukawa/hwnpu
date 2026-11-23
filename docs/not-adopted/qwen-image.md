# Qwen-Image-2.1 (Turbo と蒸留なし)

## 状態

- **削除済み (コードなし)**。2026-11-18 に外しました (#222、`bb5da8d5`)。Sana も一緒に外しています。

## モデル

| モデル | リポジトリ | ライセンス | 大きさ |
|---|---|---|---|
| Qwen-Image-2.1-Turbo | `Qwen/Qwen-Image-2.1-Turbo` | Qwen Research License (研究と評価だけ、商用不可) | 取得 約 32.4 GB |
| Qwen-Image-2.1 | `Qwen/Qwen-Image-2.1` | 同じ | DiT 14.2 GB (Turbo とは別の重み) |

- DiT は 7.1B のシングルストリームです。
- 文を読むのは Qwen3-VL-8B-Instruct そのものです (17.5 GB)。重みの一部の sha256 が一致することを確かめています。
- VAE は Wan 2.2 系の `AutoencoderKLQwenImage21` で、64 チャンネル、1/16、RGBA の4チャンネルです。
- Turbo は 8 ステップで CFG なしです。蒸留なしの版は 40 ステップのオイラー法で、CFG は既定で使いません。
- 調べたときの記録は [image-models-survey-qwen-chroma.md](../image-models-survey-qwen-chroma.md) にあります。

## どの config で動かしていたか

最後に models.yaml にあったのは `2d5e46f6` です (`git show 2d5e46f6:models.yaml` の `Qwen-Image-2.1-Turbo:` と `Qwen-Image-2.1:`)。

- `family: qwen-image`。`tag` は `2.1-turbo` と `2.1` で、family だけで呼ぶと Turbo になります。
- `license` は `{id: qwen-research, class: non-commercial, accept: true}` です。名前で選んだときに一度だけ端末で同意を聞き、`~/.config/hwnpu/licenses.json` に残していました。ここで作ったライセンス確認の仕組みは、今も Anima などで使っています。
- `machine: 16` (Turbo)、`machine: 0` (蒸留なし)。
- `memory: {amd: 4.2, apple: 6.0}`、`bytes: 3.25e10`。
- 形式:
  - AMD: 文は q6。DiT は q8 で、MLP を q6 にする案を試しました (16.167)。
  - Apple (Core ML): 文の符号器 lut8、DiT の文のパス lut8、DiT の画像のパス lut4、VAE fp16。
  - Apple (Core AI): DiT の組 8 本 (lut4、関数 512² と 1024²)、VAE 7 段、文の 17 本。

## 実装状況 (外した時点)

**AMD (Ryzen AI 9 365)**
- 1024² で1枚 1 分 52 秒でした。内訳は、文 5.1 秒、DiT の文 2.2 秒、ステップ 12.0 秒 × 8、VAE 約 9 秒です。512² は1ステップ 3.4 秒でした。
- ピークは 6.8 GB (NPU 6.1 GB) でした。
- CPU の float32 の diffusers と比べた結果です。

| 条件 | 潜在のコサイン | PSNR |
|---|---|---|
| 512² の狐 | 0.9978 | 27.9 dB |
| 512² の看板の文字 | 0.9995 | 31.3 dB |
| 1024² の狐 | 0.9989 | 34.2 dB |

- 蒸留なしの版と、8 つのアスペクト比の書き出しは、残りの作業のままでした。

**Apple**
- Core ML (M3): プログラムは合わせて 17 GB です。ANE のコンパイル済みキャッシュ (約 4 GiB) を超えるので、2 回目でもほぼ全部が外れました。
  - 2 回目からの1枚は、512² で 4 分 26 秒、1024² で 7 分 08 秒でした。
  - footprint の最大は 4.5 GB と 6.0 GB でした。
- Core AI (M1): 512² と 1024² をまとめ、重みの共有で書き出しを 22.1 GB から 18.1 GB に減らしました。
  - 初回 (コンパイル込み) は 1 時間 13 分でした。
  - 2 回目からは、512² が1枚 1 分 04 秒、1024² が 2 分 48 秒でした。
  - footprint の最大は 4.6 GB と 5.5 GB でした。
- CFG の試しでは、プログラムの読み込みの途中で 2 時間止まったことがあります。再現はせず、原因は分かっていません。

**Intel**
- していません。

## ロジックが残っているコミット

- **最後にコードがあったコミット: `2d5e46f6d26bc8c073e9be07b7d650f6b899ac7c`** (#222 の親)。
  - Go: `internal/imagegen/qwenimage/` (変換、`progEngine`、VAE のタイル `runTiled`、DiT の入れ替え `swapDiT`)、`internal/imagegen/qwenimage/mil/`、`cmd/hwnpu-image/qwenimage.go`
  - AMD: `tools/amd/iron_qwenimage.py`
  - 参照: `tools/ref/ref_qwenimage21.py` (`--stream` で重みを流し読み)、`tools/ref/qwenimage_quant.py`、`tools/coreai/qwenimage_dit.py`、`tools/coreai/qlut/`
  - 記録: `docs/design.md` の 16.155、16.157、16.165、16.167、16.181、16.184、16.211、16.216
- AMD と Core AI の最初の版は #123 (`9c01f3fb`) です。
- **今も main に残っているもの:**
  - `tools/ref/stream_weights.py`: 繰り返しの層の重みを実行の直前に読む参照です。
  - キーを 2 のべき乗の塊に分けるヘッドごとの注意 (`milpart.KeyParts`、`HeadOptions.Split`)。Qwen-Image で作り、FLUX でも使っています。
  - IRON の共通部品は `tools/amd/iron_dit.py` に移しました。
  - models.yaml の `license` の `accept` と、初回に同意を確かめる仕組み。

## 問題と外した理由

- **外した理由:** NPU で使うにはパラメータが多すぎました。7B の DiT と 8B の文の符号器です。
  - Mac では、プログラム 17 GB が ANE のコンパイル済みキャッシュに入らず、毎回の読み込みが時間の大半になりました。
  - AMD でも 1024² で約 1.5〜2 分かかりました。
  - ライセンスも非商用でした。
- **見つかった問題** (今のモデルにも当てはまるものは [image-apple.md](../design/image-apple.md) の「Lessons kept from dropped models」に残っています):
  - VAE の 1152 → 1152 チャンネルの段は、64 × 64 以上で M1 の Core AI の ANE に載りませんでした (`Allocation failure for L2-dep pair`)。重なりのある窓 (48 × 48 以下) に分けて貼り合わせると、厳密に同じ値で ANE に載りました。
  - Core AI で DiT の 8 本 (1 本 0.44 GiB) を全部開くと、最初のステップで `Inference failed — request aborted (underlying=0x15)` になりました。組を入れ替えながら回して避けました。
  - 文の符号器の層 6 から、残差が 13500 になりました (最初の `<|im_start|>` の行の注意の吸い込み)。1/4 に縮めて持ちました (`TextScale`)。
  - AMD では、同じグラフの 2 回目の最初のコマンドが `ERT_CMD_STATE_TIMEOUT` になりました (`dit_text`)。
  - 注意を 4 ビットにすると、看板の種で崩れました。MLP を 6 ビットにするのが釣り合う点でした。GPTQ は校正したプロンプトに合わせすぎるので採りませんでした。

## 再開するときの手順と注意

1. 先にライセンスを確かめます。Qwen Research License のままなら非商用です。新しい版のライセンスが変わっていないかを見ます。
2. 大きさの壁は変わっていません。再開するなら、次のどちらかのときです。
   - 小さい版 (DiT が数 B 以下) が出たとき
   - ANE のコンパイル済みキャッシュが大きくなったとき
3. `2d5e46f6` から `internal/imagegen/qwenimage`、`cmd/hwnpu-image/qwenimage.go`、`tools/amd/iron_qwenimage.py`、参照の道具を戻します。そのうえで、今の仕組みに合わせて直します。
   - 8 サイズを1つの書き出しにまとめる
   - model-recipes.yaml の `apple` と `amd`
   - hwnpud がキャッシュを一手に書く仕組み
4. M1 の Core AI では、プリビルドと端末のコンパイルの問題に気をつけます ([chroma.md](chroma.md) の h13g の問題も参照)。
