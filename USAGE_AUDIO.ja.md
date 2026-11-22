# 音声のコマンド (hwnpu-audio)

[English](USAGE_AUDIO.md)

`hwnpu-audio` は、音楽と効果音の生成 (`generate`)、読み上げ (`tts`)、書き起こし (`transcribe`)、音高 (`pitch`)、音符 (`notes`) を NPU で行います。どれも初回にモデルを落として自分で変換します。選べるモデルは `hwnpu-audio list` で出ます。コマンドはアプリ向けのライブラリ `ai` のデモです ([docs/app-api.md](docs/app-api.md)。音楽、効果音、音高、音符は `ai/sound`、読み上げと書き起こしは `ai/speech`)。動かす前に足りないものだけを準備し、コンパイルしたプログラムは hwnpu の利用者のキャッシュに置きます (`--cache` はありません)。モデルの作り方や動かし方を変えるフラグ (`--format`、`--text-format`、`--dit-format`、`--cpu`、`--device`、`--threads`) は、元のモデルとは別に準備する、そのモデルだけのモデルになります。

Mac では Core AI を通して Neural Engine で動かすだけなので、macOS 27 が要ります (macOS 26 以前では、そう書いたエラーで止まります)。モデルは Core AI 向け (モデルのディレクトリの書き出し `NAME.coreai` に `.aimodel`、hwnpu-chat や hwnpu-image と同じ) に変換し、コンパイルした結果は Core AI が自分のキャッシュに持つので、2 回目からの読み込みは 0.1 秒です ([docs/design/audio-models.ja.md](docs/design/audio-models.ja.md))。`--cpu` は確かめ用に、Core ML の書き出しを CPU のインタプリタで動かします。

AMD の NPU (Ryzen AI 300 以降) では、どのモデルも IRON の書き出し (`tools/amd`) で動きます。初回は、取得元になければ自分で書き出し (IRON の環境は [INSTALL_AMD.ja.md](INSTALL_AMD.ja.md) のとおり)、書き戻す取得元に置きます。書き起こし、読み上げ、効果音は NPU で、音楽は復号器が NPU、言語モデルが CPU です。`pitch` と `notes` は CPU で動きます。

## 音楽 (Magenta RealTime 2)

`go run ./cmd/hwnpu-audio generate "disco funk"` は Magenta RealTime 2 small で、文で指定した曲調の 48 kHz ステレオの音楽を作り、そのままスピーカーで鳴らします (`-o out.wav` でファイルに)。M1 で実時間の約 1.4 倍の速さです。重みは CC-BY 4.0 なので、使うときは「Magenta RealTime 2 by Google DeepMind」と表示してください。

- 録音の曲調: `song.wav`
- 曲調の混ぜ合わせ: `"jazz@2" "bossa nova"`
- MIDI の音符: `--notes melody.mid`
- 録音の続き: `--continue intro.wav`
- 鳴らしながらファイルにも書く: `--play -o out.wav`、標準出力に流す: `-o -`

初回はモデルの全部を準備します。曲調の音声の側と SpectroStream の符号化器 (`--continue` 用) もいっしょなので、あとで録音の曲調や続きを使うときに、ダウンロードも変換も要りません。

AMD の NPU では、SpectroStream の復号器が `tools/amd/iron_magenta.py` のグラフで、言語モデルと MusicCoCa は CPU で動きます。初回は、取得元に書き出し (`magenta-realtime-2-small.amd-npu2`) がなければ IRON で復号器を書き出します (約 15 秒)。Ryzen AI 9 365 で実時間の約 0.3 倍 (4 秒を 13 秒。音楽の 1 秒あたり復号器 1.6 秒、言語モデル 1.6 秒。全部 CPU なら 28 秒)、約 2.7 GB です。

## 効果音 (MOSS-SoundEffect)

同じ `generate` に `-m sfx` を付けると MOSS-SoundEffect v2.0 (Apache-2.0) で、英語か中国語の説明から最長 30 秒の 48 kHz の効果音を作ります。文の符号器 (Qwen3-1.7B)、DiT (1.3B)、DAC の復号がすべて NPU で動き、M1 でカードの 100 ステップなら 10 秒の音に約 4 分です。マシンのメモリで、16 GB 以上は文の符号器が fp16 のモデル、8 GB は lut8 のモデルを選びます (`-m moss-soundeffect:8gb` や `:16G` で明示)。

```bash
go run ./cmd/hwnpu-audio generate -m sfx "A dog barking loudly in a park."
go run ./cmd/hwnpu-audio generate -m sfx --duration 20s --steps 50 "Rain on a tin roof."   # --steps 50 で半分の時間
```

AMD の NPU では、文の符号器、DiT、DAC の復号器が `tools/amd/iron_moss_sfx.py` と `iron_moss_dac.py` のグラフで動きます (DiT の行列積は bfp16、文の符号器は bf16、`moss-soundeffect:8g` では 8 ビット。復号器の Snake は、Intel の NPU の版と同じく sin を級数で計算する float32 のカーネル)。初回は、取得元に書き出し (`moss-soundeffect-v2.0.amd-npu2`) がなければ IRON で書き出します (IRON のカーネルが揃っていれば約 30 秒)。Ryzen AI 9 365 で 1 ステップ 2.8 秒、復号器は 5 秒の音に 1.7 秒 (5 秒の音を 25 ステップで 1 分 06 秒、カードの 100 ステップなら約 4 分 40 秒)、約 5 GB です。

## 読み上げ (LFM3-Audio-2B)

`go run ./cmd/hwnpu-audio tts --lang ja -o out.wav こんにちは` は LFM3-Audio-2B (`-m speech`) で文章を読み上げます。詳しくは [CHAT.ja.md](CHAT.ja.md) の 5 節にあります。

AMD の NPU では `tools/amd/iron_lfm3speech.py` の書き出しで動きます。文のモデル (8 ビット、最後の隠れ状態を出す)、depthformer の MaskGIT の 1 回、codec の復号器がそれぞれ NPU のプログラムで、符号の抽選と逆 STFT はホストです。初回は、取得元に書き出し (`lfm3-audio-2b-tts.amd-npu2`) がなければ IRON で書き出します (IRON のカーネルが揃っていれば約 1 分)。Ryzen AI 9 365 で実時間の約 0.43 倍 (日本語の 5.1 秒を 11.8 秒。80 ms のフレームごとに文のモデル 107 ms、depthformer の 3 回で 72 ms)、約 2.8 GB です。

## 書き起こし (Whisper)

`go run ./cmd/hwnpu-audio transcribe speech.wav` は Whisper large-v3-turbo で音声を書き起こします。初回にチェックポイントを落として変換します。M3 で 30 秒の窓を 0.6 秒で読み、65 秒の日本語を 4 秒で書き起こしました。`-l ja`、`--translate`、`--timestamps` が使えます。

AMD の NPU (Ryzen AI 300 以降) では、`tools/amd/iron_whisper.py` の 2 つのグラフ (窓ごとの符号器と、1 トークンずつの復号器。復号器の KV キャッシュは NPU に置きます) で動きます。初回は、取得元に書き出し (`whisper-large-v3-turbo.amd-npu2`) がなければ、チェックポイントを落として IRON で書き出します (IRON のカーネルが揃っていれば約 20 秒。IRON の環境は [INSTALL_AMD.ja.md](INSTALL_AMD.ja.md) のとおり、`HWNPU_IRON_ENV`)。Ryzen AI 9 365 で、30 秒の窓の符号器が 0.84 秒、復号が 1 秒に 51 トークンで、88 秒の英語を 9 秒で書き起こしました (NPU の固定メモリとプロセスで約 2.1 GB)。`-m DIR` で書き出したディレクトリを動かせます。

## 音高と音符 (SwiftF0、Basic Pitch)

`go run ./cmd/hwnpu-audio pitch voice.wav` は SwiftF0 で声や楽器の音高を 16 ms ごとに、`go run ./cmd/hwnpu-audio notes --midi song.mid song.wav` は Basic Pitch で音符を MIDI に出します。どちらも ONNX で配られるモデルで、`internal/onnx` が静的な部分を NPU のプログラムにします。AMD の機械では CPU で動きます ([USAGE_IMAGE.ja.md](USAGE_IMAGE.ja.md) の物の検出の節と同じく、ONNX Runtime があればそれを使います)。
