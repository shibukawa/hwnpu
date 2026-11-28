# 音のモデル

[English](audio-models.md)

音のモデルはすべて 1 つのコマンド `hwnpu-audio` (`generate`、`tts`、`transcribe`、`pitch`、`notes`、`list`) で動く。フラグとサブコマンドは [cli.md](cli.md)。チャットのモデル (LFM3-Audio、d1-Omni) への音声入力は [llm-models.md](llm-models.md) にあり、ここでは符号器だけをまとめる。

## 概要

| モデル (models.yaml) | 用途 | Apple (Core AI) | Intel | AMD (IRON) | メモリ (models.yaml) |
|---|---|---|---|---|---|
| `whisper-large-v3-turbo` (MIT) | 書き起こし | lut8、全部 NE | IR (int8)、OpenVINO の CPU でだけ確認 | 符号器と復号器のグラフ、復号器は q8 | apple 1.9 / amd 2.2 GB |
| `LFM3-Audio-2B-TTS` (LFM Open 1.0) | 読み上げ | 文のモデル lut8、depthformer と codec は fp16 | IR (int8)、CPU でだけ | 文のモデル q8、depthformer と codec はグラフ | apple 2.4 / amd 2.9 GB |
| `magenta-realtime-2-small` (CC-BY 4.0) | 音楽 | 全部 NE | IR、CPU でだけ | SpectroStream の復号器のグラフ。言語モデル、MusicCoCa、符号器は CPU | apple 1.0 / amd 2.7 GB |
| `MOSS-SoundEffect-v2.0` / `-lut8` (Apache-2.0) | 効果音 | 全部 NE (M1 と M3) | IR、CPU でだけ | 文の符号器、DiT、DAC のグラフ | apple 2.3 (2.2) / amd 5.0 GB |
| `swift-f0` (MIT) | 音高 (F0) | 全部 NE | IR、CPU でだけ | CPU (`device: {amd: cpu}`。ONNX Runtime か milcpu) | 0.06 GB |
| `basic-pitch` (Apache-2.0) | 音符、MIDI | 一部が NE の外 | IR、CPU でだけ | CPU (`device: {amd: cpu}`。ONNX Runtime か milcpu) | 0.07 GB |

- 「IR、CPU でだけ」: Intel のプログラムは書く (lut8 は `llm.IntelFormat` で int8 に読み替え) が、通しで確かめたのは OpenVINO の CPU (`HWNPU_BACKEND=openvino`) だけで、Intel の NPU の実機では動かしていない。[intel.md](intel.md)
- Magenta RT2 の重みは CC-BY 4.0。音楽を使うところに「Magenta RealTime 2 by Google DeepMind」と表示する
- SwiftF0 と Basic Pitch は ONNX でしか配られていない。公式とバイト単位で同じ Hugging Face のミラーから、リビジョンと sha256 で固定して取る (`internal/pitch` も sha256 を確かめる)

## バックエンド

### Apple: Core AI だけ

- Mac の `hwnpu-audio` は Core AI だけを使う (macOS 27)。Core AI がなければ `errNoCoreAI` (「macOS 27 が要る」) で止まる。`HWNPU_BACKEND` が空、`coreml`、`coreai` のどれでも Core AI (`audioDevice`、`cmd/hwnpu-audio/backend.go`)。[apple-coreai.md](apple-coreai.md)
- どのモデルもチェックポイントから Core AI のプログラムを直接書く (`.mlpackage` を経ない)。書き出しはチェックポイントの横の `NAME.coreai` (`llm.CoreAIExport`)、Intel の IR は `Dir/intel`
- Core ML は CPU のインタプリタのためだけに残す: `--cpu` と、ONNX Runtime のない機械の pitch と notes
- 読み上げの文のモデルの 1 トークンと 64 トークンのプログラムは、1 本の `.aimodel` の関数 2 つ (`chat.ConvertShared`)。重みはディスクにもキャッシュにも 1 回分 (2 部分で 0.81 + 0.82 GiB。Core ML では 4 本で 3.4 GB)
- Core AI は計算するユニットを選べないので `--decoder-on` はない
- Magenta では Core AI の方が Core ML の NE より CPU の参照に近い (言語モデルの KL 0.0000 と 0.0003 nats、一番ありそうなトークンの一致 300 のうち 297 と 295)

### AMD: モデルごとの IRON のグラフ

- 変換器は AMD のプログラムを書かないので、モデルごとに `tools/amd` の IRON の書き出しがある (`iron_whisper.py`、`iron_lfm3speech.py`、`iron_moss_sfx.py` と `iron_moss_dac.py`、`iron_magenta.py`、`iron_supertonic.py`)。詳細は `../amd-npu-models.md`、共通の IRON の部品は [amd.md](amd.md)
- AMD の NPU があり `HWNPU_BACKEND` が空か `amd`/`xdna`/`iron` なら AMD で動く (`onAMD`)。パッケージごとの `PrepareAMD` が、手元、取得元、なければ IRON で書き出して取得元へ書き戻す。鍵には models.yaml の sha256 の固定が要る (`chat.PinnedRecipe`)。`like` は sha256 を引き継がないので、`MOSS-SoundEffect-v2.0-lut8` にも同じものを書いた
- `iron.GraphProgram` がグラフの書き出しを `runner.Program` にする: meta の `program` がプログラムの形を言い、詰めた行と列は入れるときに 0 で埋め、出すときに外す。そのためホストの道 (抽選、MaskGIT、CFG、チャンク) は Apple と Intel と同じ
- AMD の書き出しのないモデル (pitch、notes) は CPU になる (`chat.ConvertedDevice`)
- 取得元にある書き出し: whisper 1.81 GB、LFM3-Audio の読み上げ 3.13 GB、MOSS 6.3 GB (fp16) / 5.3 GB (lut8)、Magenta の復号器 0.09 GB、Supertonic 3 0.27 GB。[distribution.md](distribution.md)

## Whisper (`internal/llm/whisper`、`internal/transcribe`)

- 既定は `openai/whisper-large-v3-turbo` (809M)。`kotoba-tech/kotoba-whisper-v2.2` (日本語向けの蒸留、復号器 2 層) もリポジトリ名で指定すれば同じコードで動く。models.yaml には入れていない。長い音声で崩れる (HF の長い音声の方式でも同じ) ので、長い音声には turbo ([not-adopted/kotoba-whisper.md](../not-adopted/kotoba-whisper.md))

### プログラム (Apple / Intel)

| arch | 中身 |
|---|---|
| `whisper-encoder` | mel (1, 3000, 128) → 畳み込み 2 つ (kernel 3、2 つ目はストライド 2) を 3 フレームを並べた Linear 1 つで (ストライドは `reshape`) → pre-norm の 32 層 → (1, 1500, 1280) |
| `whisper-cross` | 復号器の全層のクロスの K/V を一度に (1, 160, 1500, 64)。ホストが `xk_N`/`xv_N` に分ける |
| `whisper` | 4 層の復号器。自分の KV キャッシュ (448)、クロスの K/V は入力、学習済みの位置は one-hot の `pos` (1, q, 448) と表の積 (gather なし)。語彙 51866 を lut のため 51872 に詰める (`blocks.LinearPadded`) |

- `decode.Session.SetInputF16`: 一度書いてステップ、印、巻き戻しの間そのままの入力 (クロスの K/V)
- `qchunk` (符号器の注意をクエリで分ける) の既定は 0: 0/300/500/750 行で 520/518/537/555 ms、出力は同じ。90 MB のスコアでも遅くならない

### AMD (`iron_whisper.py`)

- グラフ 2 つ: `encoder` (ログメル、conv1 と conv2、32 層、復号器 4 層のクロスの K/V) と `decoder` (1 トークン。自己注意の KV キャッシュとクロスの `xk{i}`/`xv{i}` はグラフの状態。トークンと位置の埋め込みはホストが足す)
- conv2 のストライド 2: ホストが conv1 の im2col を偶数のフレームを前、奇数を後ろに並べる。フレーム t は偶数の t、奇数の t、奇数の t−1 を読む (最後は状態に 1 行ずらして書く)
- 復号器: 重みは q8 (`--decoder bf16` で bf16。q8 の GEMV を K が 256 の倍数まで広げた)、MHA の `num_pipelines` 4 (cross の MHA が 4 層で 3.9 → 2.0 ms)、QKV を 1 つの GEMV に、残差は `linear_add`。logits の表は 52224 に詰める
- 符号器の行列積は MEDIUM (bfp16): 1 窓 0.84 秒 (HIGHEST 1.71 秒)、文は同じ
- 復号器の 1 トークンは重みを読む時間でなく演算の数で決まる: 演算 1 つに約 0.13〜0.15 ms の固定費

### ホスト

- 特徴 (`whisper.Features`、`FeaturesLong`) は WhisperFeatureExtractor と 2.7e-6 以内。長い音声はメルと下限を全体で取る (HF と同じ)
- トークナイザー: GPT-2 の ByteLevel の前処理。固定の 40 文とランダムな 3040 本で HF と違い 0
- 30 秒以下は `<|notimestamps|>` で 1 回。長い音声は transformers の長い音声の生成と同じタイムスタンプつきの窓 (`timestampRules` = `WhisperTimeStampLogitsProcessor`、`segments` = `_retrieve_segment`)。次の窓は最後に閉じた区切りの終わりから。30 秒で機械的に切ると文が抜けた
- 言語の指定がなければ `<|startoftranscript|>` の後のロジットで選ぶ
- AMD では `transcribe.PrepareAMD`。`programs` の interface が変換したプログラムと IRON を隠す

### 精度

- 符号器の fp16 は fp32 とコサイン 0.9978。層 20〜21 が無音の位置に注意の吸い込み先 (約 300) を作り、入力のわずかな誤差でどの位置がそうなるかが入れ替わる。HF を fp16 で動かしても同じ (0.9978) なので、fp16 そのものの誤差で NE の誤りではない。音声の位置は合っていて書き起こしは変わらない
- NE の数値の確かめ方: `TestEncoderOnDevice` (`internal/transcribe/device_test.go`、`HWNPU_TEST_WHISPER_REF`、`HWNPU_TEST_WHISPER_ENCODERS`、`HWNPU_TEST_WHISPER_LAYER` と `-layers N`)。NE、Core ML の CPU、HF を比べる
- 形式 (turbo、M3): LibriSpeech の dummy (73 本、481 秒) の WER は HF fp32 4.43 %、fp16 4.43 %、lut8 4.43 %、lut6 4.87 %。日本語 (`say` の 48 本) の CER は 3.60 / 3.60 / 3.74 / 3.94 %。**既定は lut8**
- AMD: 乱数の入力で符号器と IRON の参照のコサインは 0.9999 (8 層) → 0.99 (32 層) と広がるが、実際の音声は正しく書き起こす。88.3 秒の英語を 3 つの窓で同じ文と切れ目

### 数字

| 機械 | 結果 |
|---|---|
| M3 16 GB、Core AI | 初回 1 分 44 秒 (ダウンロード 33 秒、変換とコンパイル 1 分 06 秒)、2 回目から読み込み 0.1 秒。30 秒の窓 (符号器と交差) 0.53〜0.56 秒。生成 1 秒 244〜258 トークン。英語の `say` の音声: 5.6 秒を 0.70 秒 (実時間の 8 倍)、71.8 秒を 2.95 秒 (24 倍)。NE に 0.78 GiB |
| Ryzen AI 9 365 | 復号器 1 トークン 13 ms (78 トークン/秒)。符号器 1 窓 0.84 秒。88.3 秒の英語を 6.8 秒。最大 2.0 GiB。IRON の書き出し 19 秒 |

## LFM3-Audio

### 音声入力 (符号器 `lfm3-audio-encoder`、`internal/llm/lfm3audio`、`internal/audio`)

- LFM3-Audio と LFM3-Omni の両方の並びを読む。(時間, メル) 上のストライド 2 の畳み込み 3 つで時間を 1/8 に、Linear、因果的な LFM2 の層 (短い畳み込み、スライディング窓の注意、QK norm、RoPE θ 1e6)、RMSNorm、projector (LayerNorm、Linear、GELU、Linear)
- プログラムは長さ固定 (`-frames`、既定 3000 = 30 秒)。`internal/audio` が入る最短のものを選び、なければ `ErrTooLong`。各段でクリップより後ろの出力を半分ずつ間引いたマスクで 0 にするので、短いクリップを長いプログラムで動かしても同じ埋め込み
- 前処理 (`lfm3audio.Features`) は llama.cpp と同じ: 16 kHz、400 標本の周期的 Hann を 160 ごと、Slaney のメル 128 本、log10、(v + 4) / 4。librosa と 1e-4 以内。WAV (8/16/24/32 ビット PCM、32 ビット浮動小数点、チャネルは平均) を読み、ほかのレートは Hann 窓の sinc で変換
- 生の ASR ("Perform live ASR.") は扱わない

### 読み上げ (`internal/speech`、`internal/llm/lfm3gen`)

- 流れ: 文のモデルが TTS のプロンプト (`Perform TTS in {言語} using a {female|male} voice.` … `<|audio_start|>`。Omni は `<think></think>` を足す) を読み、h (最終 norm の後、`-hidden-out 1`、`decode.Session.Hidden`) を出す。80 ms のフレームごとに depthformer が 12 の符号帳を乱数の順の MaskGIT で決め (4 個ずつ 3 回)、温度、top-k (同点は残す)、繰り返しの罰則で選ぶ。終端コードで止まる。フレームの feedback の埋め込み (`codebook_embedding` の行の和) が文のモデルの次の入力
- `lfm3-depthformer`: MaskGIT の 1 回。h の射影とコードかマスクの埋め込み (表引きはホスト)、双方向の LFM3 の注意層、符号帳ごとの norm とコード表の積
- `lfm3-codec`: コーデックの復号器 (`audio_tokenizer/`、liquid_codec)。FSQ の値、各フレームを 6 位置に複製、因果的な LFM3 の層 8 つ (窓 30 の注意、短い畳み込み)、対数振幅と位相。層の種類と幅は重みの形から読む
- codec は全フレームの生成の後に `-codec-frames` (既定 64 = 5.1 秒) ごとに、復号器の遡りの分を重ねたチャンクで動かし、各チャンクの先頭を捨てる。因果的で RoPE は相対位置なので流した場合と同じ
- ISTFT (`speech.ISTFT`) はホストで llama.cpp と同じ。出力は 24 kHz
- サンプリングの既定: LFM3-Audio は温度 1.0、top-k 128、罰則 1.1、窓 4096。Omni は 0.4、top-k なし、1.2、64。`generation_config.json` の `audio_*` が上書き。乱数は Go の PCG で、同じ種でも llama.cpp と同じ音にはならない
- 文のモデルは lut8: lut4 では読み上げを書き起こし直すと崩れた (M3)。lut8 は一字一句一致
- `hwnpu-dev convert` では `-frames` でなく `-codec-frames` (設定のフラグは全アーキテクチャで共有、`TestParamNames`)

AMD (`iron_lfm3speech.py`、書き出し `lfm3-audio-2b-tts.amd-npu2`):
- 文のモデルは `iron_lfm3tts.py` 経由の `iron_export_lm.py lfm3tts` (MLP と head は q8、`--q4`)。h を状態 `hwnpu_hidden` に書く。ホストの埋め込みの表 (`hwnpu_embed_rows`) で、ホストが feedback の行を書ける (`iron.Model.StepRow`)。`iron_export_lm.py` そのものは変えない (`amdtools.LMHash` とチャットの書き出しの鍵が変わらないように)
- depthformer: 12 の符号帳の行を MHA の 64 行に詰める。32 の頭を 64 に [x0..x15, 0×16, x16..x31, 0×16] と並べ、rotate_half の対を保つ。64 の上の RMSNorm は重みを √2 で割り eps を半分に
- codec: 1 回 64 フレーム。6 倍の複製はホスト。短い畳み込みはタップごとの積を状態に k 行ずらして書いて足す
- ホストの表は `tables.safetensors` (bf16)
- 落とし穴: チェックポイントの 4 GB の表を開いてから文のモデルの書き出しを読むと、2 回に 1 回ほど `CREATE_HWCTX` が `EINVAL` (64 MiB のデバイスヒープの登録)。原因はヒープの番地の境界で、直した ([AMD](amd.ja.md))。`Load` は今も文のモデルを先に読む

数字:
| 機械 | 結果 |
|---|---|
| M3 16 GB、Core AI | lut8 で実時間の 1.6〜1.7 倍 (8.3〜9.0 秒の声を 5.1〜5.3 秒。テキストのモデル 3.2 秒、depthformer 1.5 秒、コーデック 0.1 秒)。初回 1 分 32 秒 (変換とコンパイル 1 分 23 秒)、2 回目から読み込み 0.2 秒。英語と日本語の声を Whisper で書き起こすと元の文 |
| Ryzen AI 9 365 | 実時間の 0.43 倍: フレームごとに文のモデル 107 ms、depthformer 3 × 24 ms。codec は 64 フレームで 72 ms。最大 2.6 GiB。IRON の書き出し 52 秒。全部 CPU では 4.16 秒に 42.5 秒 |

## Magenta RealTime 2 small (`internal/music`、`internal/tflite`)

- 230M。曲調 (文や音声から MusicCoCa の 12 の符号) → SpectroStream の 12 段の RVQ の符号を 1 秒 25 フレーム → 48 kHz ステレオ。`google/magenta-realtime-2` のリビジョンで固定 (1.7 GB)

| 部品 | 置き場所 | 備考 |
|---|---|---|
| 時間方向の言語モデル (12 層、1024、8 頭、注意ごとに 1 つのシンクの K/V、直前 41 + 今) | NPU、1 本 | K/V のリングは外。新しいフレームの K/V を返す |
| 深さ方向のデコーダ (2 層、768、ロジット 12294、softcap 30) | NPU、1 本 | 1 フレームに 12 回 |
| サンプリング (top-k 40、温度 1.3、Gumbel)、条件 | ホスト | CFG の強さはビン分けした条件の符号で渡すので、バッチを 2 倍にしない |
| SpectroStream の復号器 (因果な 2 次元の畳み込みと転置畳み込み、ELU) | NPU、25 フレーム + 前 16 | iSTFT (960 / 480) はホスト |
| SpectroStream の符号器 (`--continue` のため) | NPU、41 フレーム | STFT と RVQ はホスト |
| MusicCoCa の文の符号器 | NPU (`textEncoderCut`) | `spm.model` の SentencePiece Unigram (`tokenizer.LoadSentencePieceUnigram`)。埋め込みの表引きはホスト (fp16 の表 24 MB) |
| MusicCoCa の音楽の符号器 | NPU | 16 kHz モノラルを 10 秒ずつ、前処理はホスト (TFLite の PAD、RFFT2D、COMPLEX_ABS) |

- TFLite は依存なしで読む (`internal/tflite`、FlatBuffers → `internal/onnx` のグラフ)。`onnx.Graph.Unroll` は回数が定数の WHILE (音楽の符号器の 12 層) を展開し、層ごとの重みの切り出しを定数に畳む。6 次元のパッチの reshape (Core ML は 5 次元まで) はホストの添字の表
- 写像のノイズは numpy の `RandomState(0).randn` を再現 (`LegacyNormal`)
- 再標本化は resampy の `kaiser_best` をそのまま移した (`Resample`、resampy との差 3e-10)。Hann 窓の sinc では MusicCoCa の埋め込みが 42 % 動き符号が変わった
- 曲調の混ぜ合わせ (`Mix`、"jazz@2" song.wav): 向きを重みで平均し、長さを重みで平均した長さに。音声の埋め込みは長さ約 28、文は 1 で、量子化は長さで変わるので、単純な平均では音声だけになる
- 続き (`Session.Prefill`): 符号を強制し曲調はマスク、最後の 500 フレーム (20 秒) まで。音符と打楽器 (`PianoRoll`、`notes.ReadMIDI`) は C++ の実行系の `populate_condition_tokens` と同じ、チャンネル 10 は除く。CFG は条件の符号 (Python の generate と同じ) で、C++ の実行系の 3 倍のバッチではない
- ストリーミング: 25 フレームごとに復号。`-o -` で標準出力に WAV、ファイルにも 1 秒ごとに書く。復号の先読みのため 1 フレーム多く生成する。曲の頭の 4 STFT フレームは逆 STFT の前に落とす (参照と同じ)

### fp16 の数値 (NE、Intel)

- SpectroStream の復号器の値はブロックごとに 1.4e7 まで育つ (fp16 の上限 65504 を超える)。1 つのブロックの中では最大が RMS の数百倍に収まるので、ブロックごとに u = 値 / s で計算する (`dgraph.scale`、`decoderScales`): 畳み込みのバイアスを s で割る、ELU は relu(u) + (exp(clip(s·u, −30, 0)) − 1)/s、ブロックの切り替えは 2 のべき乗、出力は STFT / 64 (ホストで掛け戻す)、ブロックの和は ±60000 で飽和
- 尺度が大きすぎるとバイアスや ELU の負の側が非正規数になり、NE は fp16 の非正規数 (6.1e-5 未満) を 0 にする。milcpu の `FlushSubnormals` で再現できる。今の尺度は 1、1、8、16、64、1024、1024、出力 64 (参照の曲での最大 1.8e4)
- 符号器は値が 1e4 を超えないので尺度 1
- 読み込みのときに SpectroStream のプログラムを無音で 1 回ずつ動かし、失敗を最初に出す
- `Where` は cond·a + (1 − cond)·b で下ろす: b + cond·(a − b) では b = −3e4 のとき a − b が刻み 16 に丸められた (NE で相対誤差 0.32)。定数は fp16 の範囲に丸める (−2.38e38 が −inf になり 0 を掛けて NaN)
- Core AI: `elu` は max(x, 0) + α(exp(min(x, 0)) − 1) で下ろす
- 埋め込みの表引きはホストのまま: one-hot (128 × 16000) と表の積の版は、その機械のどの NE の予測も約 100 秒で失敗する状態を約 20 分続けた (M1)。新しい形のプログラムは NE で単独で試し、小さな既知のプログラムが動くことを確かめる
- MusicCoCa の RVQ は後の段ほど小さな差で符号が変わる: "disco funk" で Core AI と Core ML が 3 つ目から違うが、埋め込みのコサインは 0.999997 以上。符号でなく埋め込みで比べる

### AMD (`iron_magenta.py`)

- NPU に置くのは SpectroStream の復号器だけ。言語モデル、MusicCoCa、符号器は Core ML の書き出しを CPU で (`dev` nil、`checkNPU` に掛けない)。CPU の部分のためにチェックポイントが要る
- 形: 行が (時間, 周波数)、列がチャンネル。畳み込みは余白つきの格子を IRON の Copy の tap で im2col し GEMM 1 つ。DMA の一番内側は 1023 ワードまでなので、1024 チャンネルの 3 タップはタップごとに列の区画へ。転置畳み込みは出力の位相ごとに入力上の普通の畳み込み (Flax の重みを反転) を tap で散らす
- マスクなし: ホストが曲の最初のフレームを行 0 に置き、格子の 0 の行が同じ 0 になる。窓は 48 フレーム
- ELU: float32 のカーネル `kernels/elu_f32.cc` (exp は −30 で切る)。bf16 ではブロックの尺度は要らない
- 出力の 7×7 の畳み込み (64 → 2): 時間の 7 タップだけ im2col (K 3136 → 448、写し 578 → 82 MB)、周波数のタップは GEMM の列に持ち、ずらして写して足す (`out_sum`)

### 数字

| 機械 | 結果 |
|---|---|
| M3 16 GB、Core AI | 実時間の 2.1 倍、全部 NE。初回 1 分 12 秒 (読み込み 58 秒)、2 回目から曲調 0.3 秒、読み込み 0.1 秒。2 回の出力はバイト単位で同じ |
| M1 16 GB (Core ML の頃) | 言語モデル 1 フレーム 22 ms (lut8。fp16 は 28 ms、実時間は 40 ms)。NE の復号器は参照と相対誤差 6.8e-3。音符への追従 18 音のうち 14 (`--notes` なしは 5) |
| Ryzen AI 9 365 | 復号器 1 秒分に 0.86 秒 (CPU 5.7 秒)。4 秒の音楽を 15.4 秒 (実時間の約 0.3 倍、全部 CPU は 0.14 倍)。最大 2.5 GiB (8 秒)。IRON の書き出し 13 秒 |

- 言語モデルの形式: 既定 lut8 (NE で温度 1.3 の KL 0.0064。fp16 は 0.0002)。`--format` で fp16 / lut6
- footprint (M1): 符号を渡して 0.42 GB、録音の曲調 0.55 GB、文の曲調 0.97 GB、続き 0.96 GB

## MOSS-SoundEffect v2.0 (`internal/sfx`、`hwnpu-audio generate -m sfx`)

- 構造 (OpenMOSS の MOSS-TTS の `moss_soundeffect_v2`): 文の符号器は Qwen3-1.7B の最後の隠れ状態 (最後の norm の後)、DiT は Wan 2.1 の 1.3B (30 ブロック、幅 1536、12 ヘッド、文の 512 行への交差注意)、VAE は DAC (潜在 128 チャネル、1 秒に 50、48 kHz モノラル、rates 8 5 4 3 2)。flow matching のオイラー法、shift 5、カードの既定は 100 ステップ、CFG 4
- 負のプロンプトは空なので文脈はすべて 0、文の符号器は回さない。いつも 30 秒 (1500 フレーム) を消して欲しい長さを切り出す。プロンプトに「 duration: 10.0s」を付ける
- `.pth` は Go で読む (`internal/torchzip`: torch.save の zip と pickle、torch.save が書く opcode だけ)。DAC の weight norm は変換のときに g·v/|v| に畳む
- models.yaml は 2 つ: `MOSS-SoundEffect-v2.0` (machine 16、文は fp16) と `-lut8` (machine 8、`moss-soundeffect:8g`)。`sfx.Formats`: 16 GB 以上は文 fp16 / DiT lut8、それ未満はどちらも lut8 で、文の符号器の層 0、1、2、26、27 を fp16 に (`TextFP16`)。Intel は int8 で同じ 5 層を fp16
- CPU が計算に使う表 (トークンの埋め込み、最後の norm、DiT の時刻の埋め込みなど 39 個) を `sfx-tables.safetensors` (0.69 GB、`chat.WrittenTables`) に切り出して取得元と共有するので、実行にチェックポイント (11.5 GB) は要らない。チェックポイントは変換と表を書くときだけ取る
- WAV: 48 kHz ステレオ (モノラルを両方に)、利得 1。フェードは付けたときだけ
- 文の符号器を先に動かして手放してから DiT を読み込む

### 文の符号器 (Qwen3-1.7B、`internal/llm/qwen3` の NoHead、256 行)

- 最初のトークンに層 3 から最大 1.2e4 の massive activation。int8 は NE で層 2 の down_proj が inf (スケールを積和の後に掛ける)。lut8 は先に重みを戻すので問題ない

| 形式 | 相対誤差 (RMS) | 大きさ |
|---|---|---|
| fp16 | 0.66 % | 2.8 GB、4 部分 |
| lut8 | 10 % (最初のトークンのコサイン 0.943) | |
| lut8 + `TextFP16` | 4.9 % (コサイン 0.989 以上) | 1.7 GB |
| lut8 (kmeans-minmax) | 19 % | |
| lut4 | 127 % | |

### DiT (5 ブロックずつ 6 プログラム)

- 時刻の埋め込み、その射影、各ブロックの scale_shift_table はホストで計算し、入力 `mod` で渡す
- Wan の RoPE は対 (2j, 2j+1) を回す。q と k の射影の行 (バイアスと RMSNorm の重みも) を `blocks.ApplyRope` の対 (j, j+64) に並べ替える。スコアは変わらない
- 残差の最大は 1.4e4 なので 1/4 で持つ (`DitScale`)。落とし穴: 型のない整数の `DitScale` で `1/DitScale` が 0 になり、パッチの埋め込みと交差注意の出力が消えた。4.0 にする
- 形式 (参照の速度との誤差): fp16 2.1 % (1 回 1.65 秒、2.7 GB)、**lut8 2.7 %** (1.28 秒、1.4 GB)、lut6 12.5 %

### Core AI の DAC の復号器 (`internal/sfx/dac.go`、`DACRevision` 3)

- NE は 1 次元 16384 までなので、段ごとに時間を区切り、受容野より広いのりしろを付ける。各畳み込みの入力にマスクを掛けてゼロ詰めを再現し、区切っても一続きと同じ値
- snake の sin: u = y / π、n = (u + mg) − mg2、w = 2(u − n)、sin(y) = sin(π/2 · w) を 9 次の級数で。丸めの定数は 2 つの入力 `mg`、`mg2` (どちらも 1536): 1 つだと Core AI が (u + mg) − mg を u に畳む。clip (±1500) は残す。α が 1e-2 未満のチャネルは αx²(1 − y²/3 + 2y⁴/45) (1/α が fp16 を超える)
- 畳み込みは全部高さ方向 (`conv1d`、x を (1, C, T, 1))。Core AI の幅方向の畳み込みは、カーネルが 1 より大きく 512 チャンネル以上で入力を別の演算が計算していると誤る (相対 RMS 1.26)
- 膨張は行に (`phaseConv`): T フレームを (1, C, T/d, d) に並べ、膨張のない k × 1 の高さ方向の畳み込みに。Core AI の膨張のある畳み込みと、畳み込みの前の明示の `pad` は誤った。1 回のフレーム数は出力が 9 の倍数になるよう 1026、3195、3996、4500、7992
- ストライド 2 以外の転置畳み込みは多相に (カーネル 2 の畳み込みで s·Cout チャネル、reshape と transpose で時間に)。Core ML はストライド 5 と 7 を CPU に置き、ストライド 8/4/3 は NE で 22/45/4 % ずれた (M1)
- M1 (h13) のための直し: `sin` は GPU に回る。要素ごとの演算にまとめた clip ("NEFUSED_EW: EW: MAX") と定数倍した入力どうしの引き算 ("KernelPreScales") は受け付けない。段 3 の 4998 フレームは "unexpected failure"、4500 は通る。M3 も同じ書き出し
- 配置をオフラインで確かめる: `xcrun coreai-build compile dac-N.aimodel --platform macOS --architecture h13g --preferred-compute neural-engine`、`manifest.plist` の `mps.fullyPlacedOnANE`。理由は `/usr/bin/log show --predicate 'process == "coreai-build"'` (zsh の `log` は組み込み)
- Core ML の書き出し (`--cpu` だけ) は `sin` のまま。Intel は前の fp16 の級数と、入力の定数での `(v + 1536) − 1536` の丸めのまま
- 参照の落とし穴: `runner` の CPU の経路は `HWNPU_DEV=cpu.f16=1` がないと fp32 で計算し、fp16 の丸めの技が効かない。比べる前に参照の精度を決める (aicheck は milcpu を `RoundF16` で動かす)

### AMD (`iron_moss_sfx.py`、`iron_moss_dac.py`、`sfx.AMDExportVersion` 2)

- 文の符号器: 28 層、256 行、4 層ずつ 1 コマンド。`--text bf16` (fp16 の版) か `q8` (lut8 の版。`TextFP16` の層は bf16)。最後の norm と埋め込みはホスト
- DiT: 残差を `DitScale` で割るのは Apple と同じ。q と k の RMSNorm は 1536 の全体、Wan の RoPE は `two_halves=False`、自己注意は `iron_anima.dit_attention` で鍵 1500。cross の K/V は 1536 行の状態の最初の 512 行に書いて `nt` = 512 (MHA の鍵の行は問い合わせの詰めた行以上)。GELU は exact。行列積は MEDIUM。1 ブロックが 1 コマンド
- DAC: Snake は float32 (`kernels/snake_f32.cc`): 範囲の縮小 (αx、v/π、nπ) は float32 の肢、級数 (r⁷ まで) は bf16 の積、sin と sin² だけ 2 つの肢。正確な値との誤差は平均 0.0013 (bf16 に丸めるだけで 0.0012)。小さな α の分岐は要らない。100 万値に 0.59 ms
- DAC の畳み込み: Copy の tap の im2col (膨張はストライドに) と GEMM。2048 チャネルは 1024 ずつの列の区画に。転置畳み込み (ストライド s、核 2s) は 3 タップの畳み込みで s × Cout。マスクなし: ホストが窓を曲の頭か halo の分だけ前に置く。出力 512 チャネル未満では重みを A、im2col を列優先の B に (stage 4 で 46 → 33 ms)
- DAC の GEMM は HIGHEST のまま: MEDIUM では CPU との SNR が 25.8 dB (HIGHEST 34.6 dB) に下がり、float32 の部分和でも同じで、時間も 2 割しか縮まない。チャネルごとの 2 のべき乗の尺度でも誤差は 5 % しか減らず、bfp16 を 2 つに分けると HIGHEST より遅い
- 段ごとの窓のフレーム数: 768、2048、2048、5120、8192 (行は 256 の倍数)

### 数字

| 機械 | 結果 |
|---|---|
| M3 16 GB、Core AI、`:8gb`、5 秒、30 ステップ | 全部 NE。文の符号器 1.6 秒、読み込み 0.1 秒、1 ステップ 1.8 秒、合わせて 55 秒 (初回は文の符号器 15 秒、読み込み 3 分 20 秒)。2 回の出力はバイト単位で同じ |
| M1、Core AI、5 秒、30 ステップ、表から | 全部 NE。1 分 31 秒 (fp16)、1 分 43 秒 (lut8)。DAC は 5 秒の潜在を 4.1 秒 (CPU と相対 RMS 0.0085) |
| Ryzen AI 9 365 | 速度 1 回 1.24 秒 (1 ブロック 41 ms)、CFG の 1 ステップ 2.8 秒。DAC は 5 秒に 1.5 秒 (CPU 5.7 秒、SNR 34.5 dB)。5 秒を 25 ステップで 1 分 06 秒。最大 4.71 (fp16) / 4.75 (lut8) GiB (10 秒)。IRON の書き出し 28 秒 |

- 精度: 参照と同じ雑音から 4 ステップの波形のコサイン 0.992 (16 GB の構成) / 0.990 (8 GB)。AMD と CPU の Core ML fp16: 文の出力のコサイン 0.99963、最初の速度 0.99978

## Supertonic 3 (`internal/supertonic`)

### AMD (`iron_supertonic.py`、`supertonic.AMDExportVersion` 1)

- flow (`estimator-tT-lL`、1 ステップ)、vocoder (`vocoder-lL`)、text_encoder (`encoder-tT`)、duration predictor の文の符号器 (`duration-tT`) は `Sizes` のそれぞれの IRON のグラフ。ONNX Runtime は要らない。文の 2 グラフの窓つき相対位置の注意 (VITS) は、コマンドの間にホストが計算する (`iron.on_host`、Go の `Graph.After`、sgemm の `relAttention`)。IRON の MHA には相対位置の帯がなく、bf16 の対角の Copy は DMA の 4 バイト単位に合わない。電力の大きい GEMM は NPUサイズで重みの像を 1 つにした (596 → 255 MB)
- ガイダンスの 2 つの枝を 1 回の呼び出しで: ストリームの行は条件つきの枝のフレームと条件なしの枝のフレーム、それぞれの前後に 0 の行が 16。条件なしの枝の文への注意は定数の行 (文はどの位置も同じトークン)。スタイルの鍵は定数。時刻の行、RoPE の表、更新 x + (4 v_c − 3 v_u) / total はホスト
- pwconv1 の bias と GELU は 1 つの演算子 (`iron_biasgelu.py`)。pwconv2 の bias は、ちょうど 16 になる GELU の列で重みの列から足す (ConvNeXt 1 つが 10 → 8 演算子、5〜8 % 速い)。GEMM を MEDIUM にして GELU を出力段に融合する形はさらに 9〜12 % 速かったが、1 ステップの精度が 25〜30 dB から 18〜21 dB に下がるので採らなかった
- 深さ方向の畳み込みは `iron_dwconv1d.py` (7 までの K タップ、dilation、bias を 1 つの演算子で。16 行 × 128 チャネルのタイル。行の窓のフィフォは BD が足りなかった)。左端は各枝の最初のフレームを写し、右はマスクした 0 (Mac の固定長のプログラムと同じ)
- 全部 bf16 (HIGHEST)。flow の 1 ステップの更新はパディングした ORT と 21〜27 dB。vocoder は潜在の unnormalize を埋め込みに畳んで 37 dB (ホストで bf16 にしたときは 29 dB)。MEDIUM でもグラフは速くならない: 演算子の再構成が時間を決める (ConvNeXt の GEMM の 1 対は行数によらず約 1 ms)
- CPU interpreter の `conv1d` を `conv2d` (sgemm、im2col) に通した: text_encoder が 1 スレッドの和で 1.5 秒、57 ms に (NPU に移す前)。sgemm は 1 つの goroutine に 8 Mflop 以上 (各 goroutine が A を全部詰める)

### 数字

| 機械 | 結果 |
|---|---|
| Ryzen AI 9 365 | flow 1 ステップ 70 / 77 / 91 ms (64 × 32 / 128 × 64 / 256 × 128)、vocoder 21 / 27 / 40 ms。英語 4.25 秒を 0.68〜0.70 秒 (6.2 倍、文 0.04 秒)。+7.4 W、音声 1 秒に 1.15 J (CPU の ONNX Runtime、4 スレッド: 7.6 倍、+41.9 W、5.5 J) |

## SwiftF0 (`internal/pitch`)

- lars76/swift-f0 0.3.0、14,386 パラメータ。16 kHz の音高 (95 候補、46.875〜2093.75 Hz) と有声の確からしさを 16 ms ごと。ONNX の初期値は protowire で読み (`onnx.go`)、層は Go に書いた
- ホスト: 3 つの STFT (1024/2048/512、ホップ 256) → 128 帯、log(+1e-8) を float64 で (疎なフィルタバンク、前計算の回転因子)。最後の復号 (最良の候補と両隣の softmax で log 周波数) もホスト。NPU は CNN のプログラム 1 本 (1896 フレーム = 1875 + 11 + 10)、`SwiftF0.detect` と同じく 1875 フレームの窓
- `mask` を各 ReLU の出力と入力に掛け、固定長のプログラムで短いクリップが自分の長さのプログラムと同じになる
- 大きな膨張 (17、27、40、48) は、帯をずらした k 本をチャンネルに重ねた 1×1 の畳み込み 1 回に (`shifted`)。そうしないと Core ML が CPU に置いた。全 115 演算が NE (`Revision` 3)
- 精度: ONNX の参照と最大 0.11 cent (CPU の fp16)、NE で 2581 フレーム最大 0.23 cent、有声 / 無声は全部一致。Core AI と Core ML: 全フレーム同じ音高
- M1 (Core ML): 41.3 秒を 0.24 秒、footprint 58 MB。M3 の Core AI: 初回 7.8 秒、2 回目から読み込み 0.1 秒

## Basic Pitch (`internal/notes`、`internal/onnx` 経由)

- `nmp.onnx` (36K)。`internal/onnx` がグラフを切る (`Graph.Cut`): 静的な部分は MIL に下ろし、動的な尾部はホストで評価 (`EvalShapes`)。CQT (22050 Hz、2 秒の窓。サンプルの軸 43844 が 16384 を超える) と対数正規化はホスト、倍音の積み重ねと CNN は NPU
- 出力は平らに連結 (`Flat`): 最後の軸での連結は note の活性が崩れた (59 % ずれ)
- 窓の切り方 (30 フレームの重なり)、音符の組み立て (`output_to_notes_polyphonic`、推定した onset、melodia の手法、ピッチベンド)、MIDI の書き出しは basic-pitch から移した。参照の活性からの音符は完全に一致
- 倍音の積み重ねの後の (3, 39) の畳み込みと ReLU は NE の外に残る (幅 16 以下に分けても)。Core ML でも Core AI でも
- M1 (Core ML): 5 秒を 0.14 秒、NE の活性の相対誤差 2 %、同じ 10 音符。M3 の Core AI: 2 回目から 0.2 秒、CPU の参照と同じ 8 音

## 採らなかったもの

- SpectroStream を GPU の fp32 で: 目的は NPU。ブロックの尺度で NE の復号器が正しく (相対 6.8e-3)、速くなった (10 秒の復号 2.0 秒、GPU 4.5 秒)。`--decoder-on` は外した
- `hwnpu-audio` で Core ML を Neural Engine に使うこと: Mac では Core AI だけ
- Whisper の lut6: WER 4.87 % (lut8 4.43 %)、主に符号器で落ちる。Whisper の fp16 を既定に: 2 部分で生成が半分ほどの速さ、書き起こしは同じ
- Whisper の符号器の `qchunk` 500: 速くならない。`LayerNormMax`: 変わらない (fp16 のあふれではない)
- MOSS の文の符号器の素の int8 (NE で inf)、lut4 (127 %)、kmeans-minmax の lut8 (19 %)。DiT の lut6 (12.5 %)
- Magenta の one-hot の表引きを NE で: 機械の NE 全体が止まった
- Magenta の言語モデルを AMD の NPU に (`iron_magenta_lm.py`): 1 フレームに temporal 34 ms と depth 58 ms で CPU と同じ速さ、しかも復号器と NPU を取り合う。書き出しは測るために残す
- Magenta の出力の層を FLUX の VAE の `GridGEMM` で: IRON で落ちた ("aiex.npu.rtp_write must be a Value")。原因は調べていない
- AMD で Snake を後ろの GEMM に融合: IRON の GEMM に後処理の口がなく、7 タップの im2col の前では 7 倍の計算。カーネルを速くした
- AMD の MOSS の DAC の GEMM を MEDIUM に (上)
- 効果音の MMAudio: 文の特徴の Apple DFN5B-CLIP が研究専用の許諾
- 別々のコマンド `hwnpu-transcribe`、`hwnpu-tts`、`hwnpu-pitch`、`hwnpu-notes`、`hwnpu-music`、`hwnpu-sfx`: `hwnpu-audio` にまとめた
- 扱わないもの: チャットの返答を音声に (`<|mixed_start|>`)、読み上げの流し出し、LFM3-Audio の生の ASR、LFM3-Omni の読み上げ (`model.depth_decoder`)、MOSS のプロンプトの ftfy の `fix_text`、Magenta の本当の CFG (3 倍のバッチ)

## 未解決

- 耳で聴くこと: 読み上げ、Magenta、MOSS のどれも耳では確かめていない (数値と書き起こし直しだけ)。読み上げを同じ符号から llama.cpp と比べる
- Intel の NPU の実機: 音のモデルはどれも OpenVINO の CPU でだけ確かめた (MOSS は 20 ステップ、1 ステップ 18 秒)。NPU のコンパイラが ConvolutionBackpropData、Elu などを受け付けるか
- M2: Core ML の NE での SpectroStream の復号が機械を重くしたことがある。ブロックの尺度や Core AI では確かめていない
- MOSS の DAC r3 の M3 の実機での速さと値 (オフラインのコンパイルでは乗る)
- macOS 26 での `hwnpu-audio` のエラーを実機で確かめていない
- Whisper: 実際の録音 (雑音、話者の重なり)、AMD での日本語、温度を上げてのやり直し、`condition_on_prev_tokens`。`-v` が英語の窓でも `ja` と出る、長い英語に大文字が多い (Core ML でも同じ)。`/v1/audio/transcriptions` の API はない
- AMD の Whisper: LayerNorm を GEMV に融合していない (約 1.9 ms。`QGEMVQKV` の正規化は RMS だけ)。IRON の演算 1 つの固定費 (約 0.13〜0.15 ms) が小さな復号器を決める
- AMD の読み上げ: 使わない head (0.27 GB) を外す、`depth_linear` をフレームに 1 回 (4 ms)、depthformer の重みを q8 に
- Magenta: 参照の曲より大きな音は飽和で歪む (±60000)。44.1 kHz の続きは参照と並べていない。AMD では MusicCoCa の音声の側と符号器は CPU
- Basic Pitch: (3, 39) の畳み込みが NE の外に残る理由
- SwiftF0: 実際の声と楽器、ストリーミング (`PitchStream`)、`segment_notes`
