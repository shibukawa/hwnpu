# LLM と符号器のモデル

[English](llm-models.md)

LFM3 の系列 (LFM3-1.7B、LFM3-VL、LFM3-Audio、LFM3-Omni、d1)、Gemma 4、PLaMo 3 Translate、Ouro、
PLaMo-Embedding-1B、BERT の系列のモデルごとの記録と、トークナイザー (`internal/tokenizer`) とチャットの書式
(`internal/chat`)。Qwen のモデルは [llm-qwen.md](llm-qwen.md)、専門家の混合 (LFM2-8B-A1B、Qwen3.6、gpt-oss) は
[moe.md](moe.md)、エンジンの仕組み (KV キャッシュ、リング、注意で分けた形、プロンプトストア、最後の行のロジット) は
[llm-engine.md](llm-engine.md)、重みの形式の詳細は [quantization.md](quantization.md)、AMD の形は
[amd-llm.md](amd-llm.md)。モデルと既定は [models.yaml](../../models.yaml) にある。

## 概要

「移植」= Hugging Face の `modeling_*.py` を Go (`internal/llm/<arch>`) に移すこと。
[new-model-checklist.md](new-model-checklist.md) と `port-architecture` スキルを参照。Mac では `hwnpu-chat` は
Core AI だけで動く ([apple-coreai.md](apple-coreai.md))。Go の変換器はまず `.mlpackage` を書き、それを下ろす。

| モデル (models.yaml) | アーキテクチャ | 種類 | Apple の既定 | Intel | AMD |
|---|---|---|---|---|---|
| LFM3-1.7B | `lfm3` | chat (tools、thinking) | lut6、grouped の注意、文脈 262144 | int8、変換できる、実機では動かしていない | q4 ([amd-llm.md](amd-llm.md)) |
| LFM3-VL-2B | `lfm3` + `lfm3-vision` | chat + 画像 512×512 | テキスト部 lut8、符号器 fp16 | 変換できる | q4 |
| LFM3-Audio-2B | `lfm3` + `lfm3-audio-encoder` | chat + 音声入力 (30 秒) | テキスト部 lut8、符号器 fp16 | 変換できる | q4 |
| LFM3-Omni-2B | 両方の塔 | chat + 画像 + 音声 | テキスト部 lut8 | 変換できる | q4 |
| LFM3-Audio-2B-TTS | `lfm3` + `lfm3-depthformer` + `lfm3-codec` | speech (`hwnpu-audio tts`) | テキスト部 lut8、depthformer とコーデック fp16 | — | [audio-models.md](audio-models.md) |
| d1-2B / d1-Omni-2B | d1 モードの `lfm3` (+ 塔) | decision (`hwnpu-chat systemone`) | lut8 | 変換できる | — |
| gemma-4-E2B-it / E4B-it | `gemma4` | chat (tools、thinking)、テキストのみ | lut6 | 変換できる | q4、画像と音声も |
| plamo-3-translate-0.8b / 2b / 8b | `plamo3` | chat (translation) | lut8。8B は lut4 | int8。8B は行ごとの int4 | q4 |
| Ouro-1.4B-Thinking / 2.6B-Thinking | `ouro` | chat (thinking、道具なし) | lut8。Mac では動かしていない | int8。変換はできるが実機では動かしていない | q4 で MLP と head は q6。HF と比べて KL 0.026 (1 位の一致 98.4 %)、3.6 トークン/秒 |
| plamo-embedding-1b | `plamo-bi` | embedding (query) | lut8 + head-scale | int8 | — |
| BERT の系列 (all-MiniLM-L6-v2 など) | `bert` | — (models.yaml にない) | 変換のみ、lut8 | 変換のみ、int8 | — |

既定は `llm.Arch.Format` / `IntelFormat`、チェックポイントごとには `Config.Format()` / `IntelFormat()`
(`llm.Formatter`、`llm.IntelFormatter`) から来る。`--format` と `--format-for SUBSTRING=FORMAT` (最後に合うものが勝ち、
`-keep-fp16` がすべてに勝つ) で変えられる。

## LFM3 の系列

### LFM3-1.7B (`internal/llm/lfm3`)

- 構造: 短い畳み込み、スライディング窓の注意 5 層 (窓 256、リング)、全注意 6 層、QK norm、部分 RoPE、ヘッドごとのゲート、
  サンドイッチ norm、SwiGLU。
- アーキテクチャの設定: `post-scale` (既定 64。`out_proj`、`o_proj`、`w2` に掛ける 2 のべき乗で、サンドイッチ norm が吸収する。
  ANE の誤差を 1 層 1.4〜2.7% に抑える)、`conv` (既定 `taps`: 短い畳み込みをずらした積和で書き、ANE の conv の誤差を避ける)、
  `attn` (既定 `grouped`、`blocks.SDPAGrouped`: 積んだ問い合わせのヘッドに KV ヘッドごとに 1 回の積。M3、キャッシュ 16320 で
  `tile` の 1 回 73.5 ms → 40.7 ms)、`hidden-out` (1: 最終の隠れ状態 h も出す。TTS が使う)。
- 既定は lut6: bf16 との KL 0.23 nats (lut4 は 1.39)。コードの課題は lut4 の 13〜27 往復に対し 6 往復。プログラム 1 本の重みは
  1.21 GiB (lut4 は 0.81)。
- 独立した fp64 の forward (`reference_test.go`、約 3e-7) と `hwnpu-dev check` で照合済み。
- 注意で 7 部分に分け、文脈 262144 ([llm-engine.md](llm-engine.md))。
- Core AI、M1 16 GB: 35.2〜35.5 トークン/秒、2 回目の準備 4.5 秒。メモリは models.yaml のとおり (1.7 GB。16 GB では短い文脈の段階込みで 3.1 GB)。
- Intel: int8。int4 では次のトークンが fp16 と同じなのは 52 位置中 22 (int4-g128 29、int4-g32 33、ANE の lut4 26)、int8 は 51。

### マルチモーダルのチェックポイント: テキスト部

- LFM3-VL、LFM3-Audio、LFM3-Omni、d1 のテキスト部は LFM3 と同じ構造。`ParseConfig` が入れ子の config を読み、
  `Config.Prefix` がテンソル名を決める:

| `model_type` | config の場所 | テンソルの接頭辞 | 出力層 |
|---|---|---|---|
| `lfm3` | 最上位 | `model.` | `lm_head.weight` |
| `lfm3_vl` (VL、d1-2B) | `text_config` | `model.language_model.` | `lm_head.weight` |
| `lfm3_omni` (Omni、d1-Omni) | `text_config` | `model.language_model.` | `lm_head.weight` |
| `lfm3_audio` | `lfm` | `lfm.model.` | `lfm.lm_head.weight` |

- 最上位の `tie_word_embeddings` が `text_config` のものより優先する。
- **テキスト部は lut8** (`lfm3.Config.Multimodal`)。lut4 では (M3): d1-2B が README の問い合わせを違う部署にした
  (shipping 0.54。fp16 と lut8 は billing 0.57〜0.62)、LFM3-VL は赤い円と青い四角を「1000pink and blue icon」と言い、
  LFM3-Audio の TTS は自身の ASR で聞き取れなくなった。lut8 は 3 つとも fp16 と同じ。lut6 は境目 (billing 0.52 対 shipping 0.42)。
- Core AI、M1: LFM3-VL 26.2 トークン/秒、Audio 24.7、Omni 22.4。2 回目の準備 3.2〜5.5 秒。メモリは models.yaml のとおり
  (Apple で 2.6〜3.1 GB)。
- Neural Engine では、どのプログラム (d1、視覚、音声の符号器、テキスト部、depthformer、コーデック) も全部 NPU に載る。
- 変換が読まないテンソル: ホストが引く埋め込みの表、`codebook_embedding`、`mask_embedding`、d1-Omni の depthformer、
  コーデックの符号化側、`depth_embeddings.N.to_logits.weight` (埋め込みと共有。HF も読み込みで無視する)。

### 視覚エンコーダ (`internal/llm/lfm3vl`、アーキテクチャ `lfm3-vision`)

- `-arch lfm3-vision` で選ぶ (`model_type` の `lfm3_vl` はテキスト部が持つ)。`lfm3_omni` の config も読む。
- 構造 (LFM 型): patchifier (パッチの画素への linear。重み (d, 3, t, p, p) は、静止画は全時間スライスに同じ画像が入るので変換時に
  t で足す)、埋め込みの RMSNorm、2 種類の pre-norm の層 (どちらも SwiGLU)、最終 norm なし:
  - 畳み込みの層: `in_proj` → B、C、x。B·x にパッチの格子上の 3×3 depthwise 畳み込み (ゼロ埋め) をかけ C を掛ける。
    (行, 列, チャネル) の並びで 9 回のずらした積和にした (`milcpu` で動き、ANE の conv の誤差も避ける)。
  - 注意の層: QK の RMSNorm、2D RoPE、GQA、双方向。
- 2D RoPE = ggml の `GGML_ROPE_TYPE_VISION`: 組は (k, k+head_dim/2)。k < head_dim/4 は行、残りは列で回り、どちらも head_dim/2 幅の
  周波数。表を作り直せば `blocks.ApplyRope` で回せる。
- projector: パッチごとの LayerNorm (eps 1e-6)、2×2 のパッチの連結 (上の左、上の右、下の左、下の右。5 次元の reshape と
  transpose)、linear、GELU (erf)、linear。
- 固定の形: 画像の大きさごとにプログラム 1 つ (`-image 幅x高さ`、2×パッチの倍数。既定 512x512)。`-seq` は使わない。
  パッケージのメタデータにアーキテクチャと設定の値 (`com.github.shibukawa.hwnpu.arch`、`...param.image`) を書き、実行側はこれで大きさを知る。
- 前処理 (`lfm3vl.Processor`): `preprocessor_config.json` / `processor_config.json` の平均、標準偏差、`resample` (2 bilinear、
  3 bicubic)、画素数の範囲。`SmartResize` = llama.cpp の静止画の smart_resize。`Resize` = PIL / torchvision のアンチエイリアス付きの
  縮小拡大を浮動小数点で書いたもの (横、縦の順、各段で 8 ビットに丸める)。PIL と ±1 以内。
- `internal/vision`: 大きさごとにプログラムを読み込み、smart_resize が選ぶ大きさに縦横比がいちばん近いプログラムに引き伸ばす。
  プロンプトでは `<|vision_start|>`、埋め込みの行、`<|vision_end|>` (llama.cpp の mtmd と同じ)。動画は扱わない。
- チャット: `image_url` 部分は文字の中のその位置にメディアの印 (起動ごとの乱数入り) になり、`Engine.Tokens` が画像の行にする。
  行は data URL の SHA-256 から決まる負のトークン id なので、同じ画像はプロンプトのキャッシュを再利用する。最近の 16 枚を持ち、
  追い出した画像は新しい id で埋め込み直す。`/v1/models` は `input_modalities` に `image` を出す。端末では `/image FILE 質問`。
- M3 16 GB、Core AI、lut8: 512 × 512 の絵と質問 (275 トークン) を符号器込みで 0.43 秒で読む。絵の説明は 32.5 トークン/秒、テキストは 38。初回は 2 分 16 秒 (変換とコンパイル)、NE の重みは 1.64 GiB。

### 音声エンコーダ (`internal/llm/lfm3audio`、アーキテクチャ `lfm3-audio-encoder`)

- 両方の並びを読む: LFM3-Audio (`encoder.lfm` が LFM2 の設定、`dim_in`、`sliding_window`。テンソルは `encoder.downsample.`、
  `encoder.lfm.lfm.`、projector は `mlp.model.0/1/3`) と LFM3-Omni (`audio_encoder_config.text_config`、`num_mel_bins`。
  `model.audio_encoder.*`、`model.audio_projector.*`)。
- 構造: (時間, メル) 上のストライド 2 の畳み込み 3 つで時間を 1/8 に (3×3 → SiLU、続いて depthwise 3×3 → pointwise → SiLU を 2 回。
  カーネルは ggml の [出力][入力][メル][時間] で読む)。チャネル優先で平らにして linear (`conv_channels` 256、`lin_out` (768, 4096, 1))。
  因果的な LFM2 の層 (短い畳み込み、`sliding_window` 個のキーのスライディング窓の注意、QK norm、head 全体の rotate_half の RoPE θ = 1e6)。
  最終 RMSNorm。projector (LayerNorm eps 1e-5、linear、GELU、linear。768 → 2048)。FFN の幅は LFM2 の `block_auto_adjust_ff_dim`。
- 固定の長さ: 最長のメルフレーム数ごとにプログラム 1 つ (`-frames`、既定 3000 = 30 秒)。入力は `x` (1, frames, 128) と
  `mask` (1, frames)。各畳み込みの段で、クリップより後ろの出力を半分ずつ間引いたマスクで 0 にする (llama.cpp は段ごとにゼロで埋める)。
  後ろの層は因果的なので、長いプログラムでもクリップの行は同じ。クリップは先頭 `ceil(ceil(ceil(frames/2)/2)/2)` 行。
- 特徴量 (`lfm3audio.Features`、= llama.cpp の `mtmd_audio_preprocessor_lfm3a`): 16 kHz、最後のフレームが最後の標本を含むよう右をゼロで
  埋める、周期的 Hann 窓 400 / hop 160、パワースペクトル、Slaney のメルフィルタ 128 本 (0〜8 kHz、面積正規化)、log10 (下限 1e-12)、
  (v + 4) / 4。librosa と 1e-4 以内。WAV: 8/16/24/32 ビット PCM、32 ビット浮動小数点、チャネルは平均、16 kHz 以外は Hann 窓の sinc で
  変換 (llama.cpp は miniaudio で読むので、16 kHz 以外は近似)。
- `internal/audio`: クリップが入る最短のプログラムで動かし、入らなければ `ErrTooLong`。LFM3-Audio は埋め込みをそのままプロンプトに置き、
  LFM3-Omni は `<|vision_start|>` … `<|vision_end|>` で囲む (llama.cpp の mtmd のとおり)。
- チャット: `input_audio` 部分 (base64 の WAV) を画像と同じくメディアの印で受ける (`Content.Media`)。端末では `/audio FILE.wav 質問`。
- M3: "Perform ASR." で `say` の英語を一字一句、日本語を 1 字違いで書き起こした。

### 音声出力 (`internal/llm/lfm3gen`、`internal/speech`)

モデルの側の部品。流れ、サンプリング、コマンドは [audio-models.md](audio-models.md)。

- テキスト部は `-hidden-out 1`: キャッシュ付きのプログラムがロジットと h を出し、`decode.Session.Hidden` が読む。
- `lfm3-depthformer`: MaskGIT の 1 回分。入力は x (コードブックごとに、確定したコードの埋め込みかマスクの埋め込み。表はホストが引く)
  と h。h の射影 (`depth_linear`) + x、双方向の LFM3 の注意層 (QK norm、全ヘッドの RoPE θ = 1e6、ヘッドごとのゲート、サンドイッチなし)、
  最終 RMSNorm、コードブックごとの RMSNorm、コードブックごとのコード表 (出力層と共有) との積 (バッチの matmul)。
- `lfm3-codec` (`audio_tokenizer/`、`model_type` liquid_codec): FSQ のコードの値 (水準の表はホスト)、2 つの射影、各フレームを 6 位置に複製、
  因果的な LFM3 の層 8 つ (サンドイッチ norm、窓 30 の注意、短い畳み込み)、出力 norm、対数振幅と位相への射影。層の種類と幅は重みの形から読む
  (llama.cpp と同じ)。1 回 `-codec-frames` (既定 64 = 5.1 秒)。長い音声はデコーダの遡りの分だけ重ねたチャンクで動かし、各チャンクの先頭を
  捨てる (因果的で RoPE は相対位置なので、流した場合と同じ)。
- ホストの ISTFT (`speech.ISTFT`、= llama.cpp の `mtmd_audio_streaming_istft`)、24 kHz。
- 設定の名前は全アーキテクチャで共有されるので、コーデックは `-frames` でなく `-codec-frames`。`TestParamNames` が同じ名前の設定の既定値の
  一致を確かめる。
- M3: テキスト部 lut4 は実時間の約 1.4 倍だが聞き取れない。lut8 (2 部分) は約 0.5 倍。

### d1-2B / d1-Omni-2B (System One API)

- d1-2B は `architectures: ["D1Model"]` の LFM3-VL のチェックポイントで、重みは元のまま。違うのは読み方だけ: プロンプト全体を 1 回で読み、
  全注意の層は双方向、スライディングの層と短い畳み込みは因果のまま (llama.cpp の `LLAMA_NON_CAUSAL_TYPE_DENSE_ONLY`)、KV キャッシュなし。
  答えは最後のトークンのロジットのうち、ラベルのトークンだけを使う。
- `Config.D1` は `d1_codes` が 2、`d1_attention` が `set` でなければエラー。プログラムの入力は `x` と `mask` (1, seq)。全注意の層は
  パディングだけを隠す (`blocks.MaskBias`)。出力は最終の隠れ状態で、出力層 (約 1 億 3 千万パラメータ) はホストに置き、ラベルの行だけ読む。
  キャッシュ付きの変換はエラー。
- `internal/systemone`: llama.cpp の `POST /v1/systemone` (TypeSafe 互換) の移植。
  - 要求: `state`、`questions` (種類 `choice` / `score` / `noul`)、キーの順を保つ (`internal/pyjson`: 順序付き JSON、
    `json.dumps(indent=2)` の形、Jinja の真偽。数値は読んだときの文字列のまま書く)。
  - プロンプト: llama.cpp の systemone テンプレート (LFM2 の d1 のプロンプト + `<think></think>`)。
  - ラベル (`d1_labels`): choice は選択肢のキーがすべて 1 文字の英字ならそれ、でなければ A、B…。1 トークンでないものは予備の表から。
    空白付きの形も同じ選択肢として最大を取る。noul は yes/Yes/YES と no/No/NO、score は数字。
  - 答え: `calibration.json` の `default` の温度で softmax。choice は最大、score は中央値の段階、noul は true の確率、confidence は答えの確率。
  - 要求の文字列の control トークンの綴り (`<|im_end|>` など) は、最初の文字の後にゼロ幅空白を入れる (`d1_break`)。
  - 質問ごとにプロンプトが違うので 1 問 1 回。`-seq` の違うプログラムのうち入る最短のもの。
  - メディア: `files` / `images` (data URL) と、チャットの形の state の `image_url` 部分を、この順に最大 8 枚。文字列の `<__media__>` は
    空白にし、画像ごとの印をユーザーの番の先頭に置く。画像は 1 要求に 1 回だけ埋め込む。d1-Omni: `data:audio/` の files と `input_audio`
    部分も同じ。話された質問 (`instructions` の種類 `input_audio`) は質問文の位置の印になる。いずれも `<|vision_start|>`/`<|vision_end|>`
    で囲む。符号器がなければ 501。
- コマンド: `hwnpu-chat systemone` (`--vision`、`--audio-encoder`。`/v1/systemone` と、`output_modalities: ["decisions"]` の `/v1/models`)。
  NPU のない機械向けに `milcpu` の CPU モード (遅い)。
- M3、lut8: NPU と CPU のインタプリタの確率の差は 1e-3 程度。2 回目以降、文章 4 問で 0.13 秒、画像 4 問で 0.9 秒。日本語の問い合わせ、
  合成画像 (色と左右)、d1-Omni の話された質問は、どれも期待どおりに答えた。

## Gemma 4 E2B / E4B (`internal/llm/gemma4`)

- 対象は `-it` のチェックポイント (`Gemma4ForConditionalGeneration`) で、テキストのモデルだけを移した。音声と画像の塔 (E2B で 1411 テンソル)
  は Apple と Intel では使わない。
- E2B のテキスト: 35 層、hidden 1536、8 ヘッドに KV 1 ヘッド。sliding (窓 512、head_dim 256) 4 層に full (head_dim 512) 1 層の繰り返し。
  各層はサンドイッチ norm の注意、GeGLU (`gelu_pytorch_tanh`)、層ごとの埋め込みの枝、最後に残差ごと `layer_scalar` を掛ける。
- 注意のスコアは縮めない (`scaling = 1.0`、QK-norm が代わり。`blocks.SDPAScale`)。値にも重みなしの RMSNorm。RMSNorm は `w * x`
  (Gemma 3 の `(1 + w) * x` ではない)。埋め込みに sqrt(hidden) (fp32 の参照に合わせて丸めない)。ロジットに softcap 30 (tanh)。全注意の層は
  proportional RoPE: 512 のうち先頭 64 周波数だけ回す。
- **層ごとの埋め込み (PLE)**: `embed_tokens_per_layer` (262144 × 8960、bf16 で 4.7 GB) はホストに置き、int8 の表 (E2B で 2.4 GB) にして使う行
  だけ読む (`llm.TokenInputs`、入力 `ple` (1, seq, 8960))。層 0 から始まるプログラムが `(norm(per_layer_model_projection(emb)) + ple * 16) / sqrt(2)`
  を計算し、層 i は i 番目の 256 を使う。分けたプログラムは `per_layer_inputs` を名前で渡す。`per_layer_model_projection` は最初の部分に数える
  (`llm.Prologue`)。
- **KV の共有**: E2B の層 15〜34 は自分の K/V を持たず、sliding は層 13、full は層 14 のものを読む。キャッシュを持つのは 15 層 (full 3、
  sliding 12 はリング)。部分をまたぐときは元の部分が `shared_k_sliding` などを出力し、`chain` が同じ名前の入力を前の部分の出力に結ぶ。
  `llm.WritePart` はキャッシュを持つモデルに `BuildCached` を使い、この出力を落とさない。
- **fp16 のあふれ**: 層 0 の `input_layernorm` の重みが最大 270 あり、QK-norm の二乗が fp16 を超える (注意の出力が 45% ずれる)。残差も
  `layer_scalar` の前は 1900 に達する。gemma4 の norm はすべて `x/64` を正規化する (eps を 64² で割る。数学的に同じ)。`-f16` の照合で KL
  3.17 → 0.0004。PLE の射影の norm だけは縮めない (入力の RMS 0.3。縮めると二乗が非正規化数になり 0.04% → 0.5%)。
- `-logits host` (既定、アーキテクチャの revision 1): プログラムは softcap の前のロジットを出し、ホストが掛ける (`llm.LogitSoftcap`)。
  `flat` / `folded` はプログラムの中に softcap を持つ: Core ML では 262144 の軸のままの softcap は (16, 16384) に畳まないと CPU に落ち、
  最後に戻す reshape も CPU に残った。
- `SDPAGrouped` (`tile` でない): `tile` は毎ステップ K と V を 8 倍に写していた (キャッシュ 16320 で 1 つ 128 MB)。M3 の lut4 で、キャッシュ
  1024 は 1 トークン 31〜35 → 23.6〜25.8 ms、16320 は 90〜150 → 47 ms。
- 全注意の層で分ける (`llm.Paged`、文脈 131072)。Core ML では分けない lut6 の部分は ANE のキャッシュに残らず (毎回約 100 秒)、注意で 8 部分に
  分けると 1.9 秒。
- 照合: 小さな合成モデルで `modeling_gemma4.py` をループで書いた fp64 の forward と 1.6e-6。改変 15 個をすべて検出。実物の E2B
  (`hwnpu-dev check`、`<bos>` から 31 トークン): 欠け・不一致 0、argmax 31/31、KL 0.0000 (fp16)。`-f16` で最悪の層 0.0098、KL 0.0004。参照の
  `tools/ref/ref_gemma4.py` は text config から `Gemma4ForCausalLM` を fp32 で組み、PLE の行を必要なときだけ読む (20 GB でなく 9 GB)。
- **重みは lut6** (`Arch.Format`):

| 形式 (E2B、CPU の照合、31 トークン) | 層の誤差 平均 / 最悪 | argmax | KL (nats) |
|---|---|---|---|
| lut4 | 15% / 31% | 13/31 | 2.54 |
| lut6 | 4.7% / 11% | 25/31 | 0.25 |
| lut8 | 1.2% / 3.0% | 27/31 | 0.048 |
| fp16 | 0.01% | 31/31 | 0.0000 |

  - lut4 が粗いのは、サンドイッチ norm のため層の出力の大半が層の中で作り直され、4 ビットの重みの誤差がほぼそのまま出るから。1 つの群を
    fp16 に残しても半分程度にしかならない。lut4 は富士山を 3778 m と答え、事実の質問に答えなかった。lut6 と lut8 は答える (3776 m、夏目漱石、信濃川)。
  - Neural Engine の上 (128 トークン、fp32 の参照): lut6 の丸めは KL 平均 1.02 (英) / 1.16 (日)、lut8 は 0.10 / 0.10 (2.16 GiB、lut6 は 1.60)。
- **GPTQ の符号 (任意)**: E2B で `hwnpu-chat --codes DIR --format-for per_layer=lut8`: ANE の KL 1.02 → 0.56 (英)、1.16 → 0.94 (日)、
  中央値 0.25 → 0.08。速さは同じで +0.01 GiB。事実の 12 問は 10/12 → 11/12。`per_layer` の射影 (41M) は校正の対象に入っていないので、GPTQ と
  合わせるときは lut8 にするのが効く。校正は `q4_gptq.py` の `load_gemma4_lean` (表を memmap、約 9 GB) で 16 GB の Mac に入る (M1 で 87 分)。
  既定にしない: 符号 2.1 GB の配り方が決まっていない。書き出しの名前に符号のハッシュと規則が入る。詳細は [quantization.md](quantization.md)。
- 速さとメモリ: Core AI の M1 で E2B 24.0〜24.5 トークン/秒、2 回目の準備 2.7 秒。Core AI の M2 で E4B (8 部分) 20.3〜20.7 トークン/秒、
  Neural Engine に 3.27 GiB、2149 トークンのプロンプト 1.8 秒、2 回目の準備 0.9〜1.0 秒。メモリは models.yaml のとおり (Apple で E2B 2.3 GB、
  E4B 4.0 GB。E4B は 24 GB の機械から)。
- トークナイザー: byte fallback の BPE。BOS は付けない (チャットの書式が `<bos>` を書く)。

## PLaMo 3 Translate (`internal/llm/plamo3`)

- `Plamo3ForCausalLM`。形は Gemma 4 から PLE、KV の共有、softcap を除いたもの。サンドイッチ norm。RMSNorm の offset は norm ごと
  (1、1/5、1/5^1.5) で、fp16 の重みに足しておく。norm の入力は 1/64 (2B の最大 |hidden| は層 18 で 1089)。
- `qkv_proj` と `gate_up_proj` は行の範囲で分ける (`blocks.LinearRows`)。`lm_head` の行は語彙を 16 の倍数に詰めた数で、余りの行は使わない。
- 8 層に 1 層が θ = 10⁶ の YaRN (factor 256) の全注意、残りは窓 2048 のスライディング (リング)。YaRN の concentration
  (0.1 ln 256 + 1 = 1.5545) は cos と sin に掛けず、全注意の層のスコアに二乗を掛ける (`scoreScale`、`PagedShape.Scale`)。キャッシュの鍵は
  concentration を含まず、ホストの注意はスケールを掛けるだけで済む。
- 全注意の層で分ける (`llm.Paged`)、文脈 262144。
- 注意の出力は token を先にした `attn_ot_N` (1, seq, heads, head_dim) で受け、並べ替えはホストがする (`decode.putAttnOut`)。
  (1, heads, seq, head_dim) で受けると最後の部分の `transpose` 1 つが CPU に残った。
- アーキテクチャは全サイズで共通で、形は config から決まる:

| | 0.8B | 2B | 8B |
|---|---|---|---|
| hidden / 層 / KV ヘッド | 1536 / 16 / 1 (16 ヘッド) | 2560 / 24 / 4 | 4096 / 32 / 4 (32 ヘッド) |
| 全注意の層 | 7、15 | 7、15、23 | 7、15、23、31 |
| 1 トークンの KV | 1 KiB | 7 KiB | 9 KiB |
| Apple の形式 | lut8 | lut8 | **lut4** (`Config.Format`、hidden ≥ 4096) |
| Intel の形式 | int8 | int8 | **行ごとの int4** (`Config.IntelFormat`) |
| プログラム (hwnpu-chat) | 2 本 | Core AI で 2 本 (1 つの `.aimodel` の関数) | 1 本 (`prefill: off`) |
| models.yaml の `machine` | 8 | 16 | 16、`not_on: [M1]` |

- 2B の形式 (M3 の ANE で fp16 と比べる、キャッシュ 4096、1 回 8 トークン): 次のトークンが同じなのは lut6 89%、lut8 98%。1 回は fp16
  105.4 ms、lut6 53.7、lut8 57.8。CPU の KL: lut4 1.91 (使えない)、lut6 0.073、lut8 0.0066、int4 0.72、int4-g128 0.32、int8 0.0025。
- 0.8B (M3、Core AI): lut8 で、短い 1 文は 72 トークン/秒。1214 トークンの日本語の文書を 0.52 秒で読み、118 トークン/秒で訳す (写すコードを prompt lookup の下書きが当てる)。訳は 20/20 が fp16 と同じ (lut6 は 16/20)。
- 2B (M3、Core AI): 短い 1 文 23 トークン/秒、同じ 1214 トークンの文書を 1.9 秒で読み 38.6 トークン/秒で訳す。NE の重み 2.47 GiB。
- 8B: Neural Engine の重みは lut8 7.58 GiB、lut6 5.66、lut4 3.77。lut4 は訳を保つ: 20 文中 13 文が lut8 と一字一句同じで、残りは正しい言い換え。
  lut8 に対する答えのトークンの KL 0.305、argmax 88% (`hwnpu-dev generate -write-ref` / `-score`)。速さ (lut4): M3、Core AI で短い 1 文 14.5 トークン/秒、
  1214 トークンの文書を 12.8 秒で読み (1 回 8 トークンのプログラム 1 本、prefill なし) 23 トークン/秒で訳す。2 回目からの起動 0.3〜0.4 秒 (初回は書き出し 4 GB のダウンロード込みで 2 分 22 秒)、NE に 3.80 GiB。Core ML の頃: M2 9.4〜10.1 トークン/秒だが毎回の起動に 35〜41 秒 (コンパイルのキャッシュに 1.2〜2.1 GiB しか残らない)。M1 は
  0.8 トークン/秒 (M1 の NE が全速で動かせる 2.8 GiB を超える) なので `not_on: [M1]` で、M1 では 2B を選ぶ。M3 の lut8 は 5.8〜6.5 トークン/秒 (Core ML)。
  メモリは models.yaml のとおり (Apple のピーク 8.2 GB)。
- Intel の 8B: int8 (7.6 GiB) は Core Ultra 5 135U の NPU が持てた約 4.6 GB を超える。行ごとの int4 (3.8 GiB、1 本)。同じ int4 の丸めを M1 の
  ANE で動かすと、20 文中 12 文が lut8 と同じで、残りは言い換え。
- 照合 (2B): 小さなモデルで fp64 と 6.5e-5。`hwnpu-dev check` で欠け・余り 0、argmax 31/31 (fp32)、`-f16` で KL 0.0001。0.8B と 8B は HF の fp32
  と照合していない (構造は 2B と同じ)。
- hwnpu-chat の翻訳: 前の組を履歴として持つ。`--to`、`--from`、`--form`、`/to LANG`。サーバーは `glossary` も受ける ([cli.md](cli.md))。
  下書きは当たらない (出力の言語がプロンプトと違う)。
- トークナイザーのファイルはアーキテクチャで選ぶ (`tokenizer.jsonl`、`tokenizer_config.json`)。

## Ouro (`internal/llm/ouro`、`tools/amd/iron_ouro.py`)

- ByteDance のループ型の言語モデル (`OuroForCausalLM`、リモートコードの `modeling_ouro.py`。Apache 2.0)。層は Llama のものに
  サンドイッチ norm を付けた形 (`input_layernorm`、注意、`input_layernorm_2`、残差。`post_attention_layernorm`、SwiGLU、
  `post_attention_layernorm_2`、残差)。注意は素の多頭注意 (16 ヘッド、GQA なし)、RoPE θ 10⁶、bias なし、ヘッドごとの norm なし、
  `lm_head` は埋め込みと別、語彙は 49152 (SmolLM のもの)。層の全体を 1 トークンにつき `total_ut_steps` (4) 回、同じ重みで回す。
  回の終わりごとに `model.norm` を通し、次の回はその値から始める。出力層は最後の回の値を読む。KV は回ごと・層ごとに持つ。
- hwnpu は回を層として扱う: `Layers()` は層 × 回 (1.4B は 24 × 4 = 96、2.6B は 48 × 4 = 192)。層 v は層 v mod 層数の重みを使い
  (`llm.Looped`)、キャッシュは回ごとの `kv_k_v`。これでキャッシュ、部分への分割、`hwnpu-dev check` が普通のデコーダーと同じに動く。
  1 つのプログラムは、層の回をいくつ組み立てても、その層の重みを 1 度だけ持つ (ビルダーが定数を名前で覚えている。
  `TestPassesShareWeights`)。`llm.EstimateBytes` は回ごとに数える (`llm.Looped`) ので、2 回以上を組み立てる部分では実際より大きく見積もる。
  KV キャッシュは 24 層のモデルの 4 倍で、fp16 で 1 トークンあたり 1.4B は 768 KiB、2.6B は 1.5 MiB。
- 早く抜けるゲート (`model.early_exit_gate`) は読まない: `early_exit_threshold` が 1.0 (チェックポイントの値) なら transformers は
  最後の回を使う。1 未満の threshold と `early_exit_step` はエラーにする (トークンごとに抜けると、後の回のキャッシュにそのトークンの
  鍵が入らない)。モデルカードにあるとおり、config.json の `total_ut_steps` を下げれば回を減らせる。
- norm: 残差は最大 212 (1.4B)、行の RMS は最大 4.7 なので、残差を読む norm (`input_layernorm`、`post_attention_layernorm`、
  `model.norm`) は入力を 1/4 にしてから二乗する。注意と MLP の後の norm は最大 54 と 25 だが、行の RMS が 0.0002 まで小さくなるので、
  入力をそのまま使う (Qwen3 と同じ 1/64 にすると、そうした行の二乗が fp16 の下限を割る)。最初の層の norm は埋め込みをそのまま読む。
- 照合 (1.4B-Thinking): 小さなモデル (3 層 × 3 回) で fp64 と 8e-7、1 層ずつ 6e-7、キャッシュ付きで部分に分けて 3.5e-4。回の間の norm を
  外す、層のキャッシュを回の間で共有する、MLP の後の norm を外す、のどれでもテストが失敗する。`hwnpu-dev check` (30 トークン): 欠け 0、
  形の違い 0、余り 2 (ゲート)。fp32 で最悪の層 0.0004、argmax 30/30、KL 0.0000。`-f16` で最悪の層 0.0012、argmax 30/30、KL 0.0000。
  参照値は transformers 4.54 での `tools/ref/ref_logits.py` (リモートコードは 5.x で動かない: `pad_token_id`、`create_causal_mask`)。
  各層を回ごとに記録し、回の間の norm も記録する (`hook_passes`)。2.6B-Thinking (192 層、残差は最大 161): `-f16` で最悪の層 0.0020、argmax 30/30、KL 0.0000。
- 形式 (CPU、`-f16`、モデル全体を HF の fp32 と比べる、30 トークン): lut8 KL 0.023 (argmax 25/30)、lut6 0.18 (23/30)、lut4 5.1 (3/30、
  使えない)、int8 0.006 (29/30)。既定は lut8 (Intel は int8)。1 トークンに 4 回読む重みは、丸めの誤差が積み重なる。48 層で 192 回になる 2.6B はさらに失う: lut8 で KL 0.13
  (argmax 27/30)、int8 で 0.067 (26/30)。models.yaml には入れていない。
- AMD (`iron_export_lm.py ouro`): IRON の `SandwichLayer` に Ouro の norm の名前を付け替えて使う。`Ouro.layer` が `model.layers` を
  回ごとに 1 度ずつ動かし (`self.layers` はそれを 4 回並べたもの)、2 回目以降には回ごとの `KVCache` を call 経由で渡し
  (`passes.P.layers.L.cache.*`。ランタイムは名前の `.cache.` でキャッシュと見なす)、回の間に `model.norm` を入れる。MLP と head は q6 (ホストで丸めたときの KL は q4 0.130、q6 0.013)。NPU では
  (`ironlm -forced` と `forced_logits.py`、64 行) q6 で KL 0.026 (1 位の一致 98.4 %)、bf16 で 0.0022 (98.4 %)。詳しくは [amd-llm.ja.md](amd-llm.ja.md)。
- トークナイザー: SmolLM のもの (1 字ずつの `Digits` の後に GPT-2 のパターンの `ByteLevel`)。`internal/tokenizer` の `gpt2Digits`。
  正解の id は tokenizers のライブラリで作る (`tokenizer_golden.py --tokenizers`): transformers 5 は GPT2Tokenizer 自身の
  pre-tokenizer をファイルのものと入れ替え、Digits の段を落とす (" ①" が 1 つの断片になる)。
- チャットの書式: `chat.Ouro` (下)。Thinking のモデルは既定で考える。ベースのモデル (Ouro-1.4B、2.6B) はチャットのテンプレートがない。
- models.yaml には入れていない: NPU ではループ型は大きさのわりに遅く、キャッシュの増え方も大きい (評価の記録と、LFM3-1.7B などとの比較:
  [looped-models.md](../looped-models.md))。

## PLaMo-Embedding-1B (`internal/llm/plamo`、アーキテクチャ `plamo-bi`)

- `PlamoBiModel`: PLaMo のデコーダの注意を双方向にしたもの。pre-norm の 16 層 + 最終 RMSNorm。融合した `qkv_proj` (2304 = 2048 + 128 + 128)、
  16 ヘッドに KV 1 ヘッド、head_dim 128、ヘッドごとの QK-norm (`q_weight` (16, 128)、`k_weight` (1, 128)、eps 1e-6)、rotate_half の RoPE
  (base 10000)。`gate_up_proj` (16384, 2048) は前半が gate。出力は最終 norm の後の隠れ状態で、プーリングはホスト。
- `architectures: PlamoBiModel` で選ぶ (`model_type` の `plamo` は因果 LM と同じ)。MoE (`n_expert`、`sparse_step`) と `eval_*_n_bit` は拒否し、
  `linear_type: fp8` と `group_size` (学習用) は受け付ける。
- `gate_up_proj` は 8192 行ずつの 2 つの linear にする (`blocks.WeightRows` / `LinearRows`、定数名 `NAME_begin_end`): 1 つだと出力がちょうど
  ANE の上限 16384 になる。切れ目が 16 の倍数なので、lut の表は行列全体を量子化したものと同じ。
- 入力は `x` と `mask`。位置はパディングに関わらず 0..seq-1。右詰めは HF の左詰めと同じ結果 (双方向 + マスク + 距離だけに依る RoPE)。
- パラメータ 10.5 億 (埋め込み表 1.03 億はホスト)。lut8 で 1 部分 0.89 GiB。
- **8 ビット**: fp16 とのコサイン類似度 (milcpu) は lut4 の `uniform` 0.20、`kmeans-minmax` 0.76、lut8 0.998。
- **head-scale** (設定、既定 64): 最終層の `down_proj` の出力が fp32 で 5e4 になり、ANE は -Inf を返した。最終層で注意を足した残差と
  `down_proj` の入力を 1/64 にし、最終 norm の eps を 64² で割る (最終 norm しか読まないので数学的に同じ)。最終 norm のないプログラム (`NoHead`、
  途中の部分) は縮めない。ANE と CPU のコサイン類似度 0.9745 → 0.99986、HF とは 8 文で 0.9969〜0.9987 (`TestPlamoEmbeddingGolden`、
  `tools/ref/embed_golden.py`)。全演算 ANE。
- ANE の注意点: 層 3 の MLP が BOS の 1 チャンネルに 1.2e3 を作り、残差に残る。RMSNorm の `x*x` は fp16 を超えるが、ANE は崩れなかった
  (二乗の和を広い範囲で計算しているらしい)。milcpu の `RoundF16` はそこで inf になるので、このモデルでは層 4 より後ろの比較に使えない。
- 埋め込みの API (`chat.Embedder`、`/v1/embeddings`、端末は `hwnpu-chat search`):
  - BOS (`<|startoftext|>`) を付けて符号化する。クエリは `QueryPrefix + text` (「次の文章に対して、関連する文章を検索してください: 」) で、
    BOS と前置きのトークンを平均から外す。本文のトークンだけを平均し、長さ 1 に揃える。
  - `--embed-seq` (既定 128、1024) の長さごとのプログラム。入る最短のものに右詰めの詰め物とマスクで流す。
  - 長すぎる文は `context_length_exceeded`、`truncate: true` で切り詰め。`input_type` (`query` / `search_query`、`document` / `search_document` /
    `passage`。既定は文書)。`encoding_format: base64` は float32 のリトルエンディアン。
  - プログラムの名前に形式が入るので、古い lut4 の書き出しは使われない。

## BERT の系列 (`internal/llm/bert`)

- 符号器は移植済みで、lut8 (Apple) か int8 (Intel) に変換できる。all-MiniLM-L6-v2 の fp16 とのコサイン類似度は lut4 で 0.23、lut8 で 0.9988。
- hwnpu-chat からは使えない: WordPiece のトークナイザーがない (`tokenizer.Load` が拒否する)。models.yaml にない。

## トークナイザー (`internal/tokenizer`)

依存は `golang.org/x/text` だけ。どれも HF の tokenizers と照合している。

| 種類 | モデル | 仕組み | 照合 |
|---|---|---|---|
| byte-level BPE (`Load`) | LFM3 の系列、Qwen | 追加トークンを先に切り出し、GPT-4 型の pre-tokenizer を手書きの走査で (Go の regexp にない先読み `\s+(?!\S)`)。マージは順位。Qwen3.5 は NFC、`[\p{L}\p{M}]+`、数字を 1 つずつ ([llm-qwen.md](llm-qwen.md))。Ouro は 1 字ずつの Digits の後、その間の文字列に GPT-2 のパターン (`gpt2Digits`) | HF と一致 (Ouro: 40 文 + ランダム 500、18,441 トークン) |
| byte fallback の BPE (`fallback.go`、`Load` 経由) | Gemma 4 | 空白を ▁ に、追加トークンの間を 1 片として順位でマージ、語彙にない文字は UTF-8 のバイト (`<0xE3>`)。BOS は付けない | 35 文 + 乱数の 300 文 (7,026 トークン) |
| SentencePiece BPE (`LoadSentencePiece`) | PLaMo-Embedding | `tokenizer.model` の protobuf を手で読む。正規化 identity、先頭に ▁。user-defined のピース (▁ の 2/4/8/16 個の連続、タブ 1/2/4 個、改行) は最長一致で結合しない。マージ一覧でなくピースのスコアで結合 (同点は左)。語彙にないものは `<0xNN>`。特殊トークンを先に切り出し、その後ろにも ▁ を補う (HF の slow tokenizer と同じ)。BOS は付けない | 32 文 (599 トークン) + 乱数の 3,032 文 (88,072 トークン) で不一致 0 |
| PLaMo (`LoadPlamo`、`.jsonl` なら `Load` が選ぶ) | PLaMo 3 | `tokenizer.jsonl` のスコア付きのピースから合計コスト最小の分け方を後ろから探す。コスト = −round(スコア × 10⁴) (Python の round)。Aho-Corasick の代わりに全ピースの接頭辞の map。同点は長いピース。どのピースにもない文字は UTF-8 のバイト。特殊トークン、同じ文字 4 個以上の連続、空白 2 個以上で切る | 40 例 + 乱数の 3,040 例 |
| Unigram (`LoadUnigram`) | T5 (画像のモデル) | precompiled charsmap で正規化し Viterbi で分ける。未知の文字は最低スコア − 10。byte fallback なし | 11 文 |
| WordPiece | BERT | **未実装**。`Load` はエラー | — |

- `LoadSentencePiece` が拒否するもの: Unigram、正規化の規則表 (nmt_nfkc など)、`remove_extra_whitespaces`、空白を後ろに付ける形、unused の
  ピース、byte fallback のない BPE、バイトのピースの欠け。
- PLaMo-Embedding の `Decode(ids, false)` は `Encode` の逆 (区切りごとに先頭の ▁ を落とす)。HF の decode は特殊トークンの前に空白を足すので
  往復しない。`skipSpecial` は HF の `skip_special_tokens=True` と同じく区切りをつないでから戻す。
- 正解: `tools/ref/tokenizer_golden.py` → `internal/tokenizer/testdata/*.json`。実物の語彙のテストは `HWNPU_PLAMO_DIR`、`HWNPU_GEMMA4_DIR` が要る。
  スコアの同点と user-defined をまたぐ結合は実物の語彙では起きないので、小さなモデルのテスト (`TestSentencePiece`) が受け持つ。
- PLaMo のトークナイザーの正解を作る環境: Python 3.12、`sentencepiece==0.2.0` (0.2.2 は `SentencePieceProcessor.Init` がない)、`protobuf`、
  `transformers<5`。
- AMD も書き出しのディレクトリから同じ Go のトークナイザーで読む。

## チャットの書式 (`internal/chat`)

どれも `chat_template.jinja` を Go に移したもの (`chat.Template`、`TemplateFor(arch)` で選ぶ) で、jinja2 / transformers が描いたものと一字一句
一致する (`tools/ref/template_golden.py`)。

| 書式 | アーキテクチャ | 照合 | 道具 | 思考 |
|---|---|---|---|---|
| LFM3 (`chat.LFM3`) | `lfm3` | 6 会話 | システムメッセージの末尾に `<tools>` の JSON (キーの順を保つ)。呼び出しは `<tool_call><function=name><parameter=key>value</parameter>` | `<think>…</think>`。前の返答の思考は落とす |
| Gemma 4 (`chat.Gemma4`) | `gemma4` | 9 会話 (`gemma4_template.json`) | 独自の記法 (文字列は `<\|"\|>`、キーは dictsort) | `<\|think\|>`、`<\|channel>thought … <channel\|>` |
| PLaMo 3 / PFML3 (`chat.Plamo3`) | `plamo3` | 8 例 (`plamo3_template_golden.py`) | なし。`target_lang`、`source_lang`、`form`、`context`、`glossary` (`Options.Translation`。API では vLLM と同じ `chat_template_kwargs`) | なし |
| Ouro (`chat.Ouro`) | `ouro` | 4 会話 (`ouro_template.json`、transformers の `apply_chat_template`) | なし (エラーにする) | `NoThink` でなければプロンプトの後に `<think>` を書く (テンプレートの `enable_thinking`)。考えないときは Qwen3 と同じ空の `<think>\n\n</think>\n\n` (テンプレートにはこの切り替えがない) |
| System One (`internal/systemone`) | d1 | 7 例 (`systemone_golden.py`) | d1 の choice / score / noul | `<think></think>` |
| LFM2、Qwen3、Qwen2.5、Qwen3.5、Qwen3.6、harmony | — | [llm-qwen.md](llm-qwen.md)、[moe.md](moe.md) | | |

- Gemma 4: ターンは `<|turn>ROLE`。`Render` はプロンプトが思考の中で終わるかも返す (思考ありで道具の応答の後は `<|channel>thought\n` で終わる)。
  終わりは `<turn|>`、`<eos>`、`<|tool_response>`。
- PLaMo 3: システムメッセージは `context` になる。訳す先の言語がなければ、日本語を含む文は英語へ、それ以外は日本語へ。過去の訳のラベルは原文から
  選ぶので、プロンプトの先頭を再利用できる。
- 思考の予算は書式の閉じる印 (`</think>`、`<channel|>`) を入れる ([llm-decoding.md](llm-decoding.md))。
- モデルの情報 (種類、機能 `tools` / `thinking` / `vision` / `audio` / `query` / `translation`) は models.yaml から来て、`/v1/models` が
  `capabilities` として返す。要求は `model` で振り分け、種類の違うモデルの名前はエラー ([cli.md](cli.md))。

## 採らなかったもの

- lut4: Gemma 4 (KL 2.54、事実の答えが誤る)、LFM3 のマルチモーダルと d1 のテキスト部 (部署の誤り、画像の説明の崩れ、聞き取れない音声)、
  PLaMo 3 の 0.8B と 2B (KL 1.91)、埋め込み (コサイン類似度 0.20〜0.76)。
- Gemma 4 の QAT 版の Q4_0 の重みをそのまま使う: ANE の lut4 は 16 行 × 行全体で 1 つの表なので、32 列ごとの scale は表せない (層の誤差は
  0.17 → 0.16〜0.20 にしか下がらない)。32 列ずつの grouped conv の Q4_0 は ANE に載るが 12.3 ms (lut4 は 1.81、7 倍)。バッチの matmul は
  CPU に落ちる (42 ms)。
- Gemma 4 の lut6 + MLP だけ lut8 (1.99 GiB で Core ML の 2 部分に収まらない) と、lut6 + 注意だけ lut8 (誤差の大半は MLP)。
- Gemma 4 の norm を重みを 64 で割って縮める (`blocks.RMSNorm` の inScale): 小さい重み (0.16) が fp16 の非正規化数になる (小さなモデルで 2.5e-4 ずれた)。
- Gemma 4 のロジットを畳んだまま出力する: 全部 ANE に載るが 1 トークン 137 ms (4 倍)。代わりにホストの softcap を既定にした。
- PLaMo-Embedding の head-scale を重みに掛ける: ANE では同じだが、小さな重みと int8 のスケールが非正規化数になり、Intel の int8 の IR が
  許容誤差 1e-3 をわずかに超えた (0.00101)。
- PLaMo 3 8B の Intel の int8 (7.6 GiB、NPU の約 4.6 GB を超える)。PLaMo 3 2B の Intel の int4 (KL 0.72)。
- PLaMo 3 8B の Core ML の int4 の出力層 (幅 107520 の linear が CPU に落ちる): Mac では int4 を使わない。
- Gemma 4 の GPTQ の符号を既定にする: 2.1 GB の符号の配り方がまだない。
- d1-omni-600M (LFM2 系、SigLIP2 と FastConformer): 対象外。
- LFM3-Audio の生の ASR ("Perform live ASR." の交互の形)、チャットの返答を音声にすること (`<|mixed_start|>`、`<|audio_start|>` での切り替え)、
  音声の流し出し: 扱わない。動画の入力: 扱わない。

## 未解決

- d1-Omni-2B: README の例の team が billing 0.45 対 shipping 0.52 (README は billing) で、CPU の fp16 でも同じ。HF は最大 `d1_set_size` (16) 問を
  まとめて読み、hwnpu は llama.cpp と同じく 1 問ずつ読む。HF とは比べていない (torch がない)。
- d1-2B / Omni は、HF とも llama.cpp の `/v1/systemone` の確率とも比べていない。
- LFM3-Audio: 思考ありで温度 0 だと思考が繰り返して答えに届かないことがある。
- PLaMo-Embedding: 1024 トークンの文での最終層の最大値 (5e4 より大きくなるかもしれない)。複数の文をまとめて流すこと (今は 1 文ずつ)。
- Gemma 4: E4B の GPTQ (16 GB では校正できない)、`per_layer` を GPTQ の対象に入れること、問いと agent の課題を増やした比較、M2 と M3。
- PLaMo 3 8B: HF の fp32 との KL、GPTQ の符号と `--lut-fit wkmeans` (fp32 の校正は約 34 GB)、長い文脈のパスキー、M4、Intel と AMD の実機、Core AI での値。
- PLaMo 3 0.8B: HF の fp32 との `hwnpu-dev check`、パスキー、AMD と Intel。
- BERT: WordPiece のトークナイザー。
- Ouro: Neural Engine (配置、速さ、プログラム全体で 96 / 192 個のキャッシュ。`UnpagedContext` の 1024 は見当)、Intel の NPU、lut6 と lut4
  の GPTQ の符号 (2.6B では lut8 にも)、AMD での 2.6B (q8)、回を減らす (`total_ut_steps` 3) 価値があるか。
