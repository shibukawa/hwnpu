# 音声のコマンド (hwnpu-audio)

[English](USAGE_AUDIO.md)

`hwnpu-audio` は、音楽と効果音の生成 (`generate`)、読み上げ (`tts`)、書き起こし (`transcribe`)、音高 (`pitch`)、音符 (`notes`) を NPU で行います。どれも初回にモデルを落として自分で変換します。選べるモデルは `hwnpu-audio list` で出ます。

Mac では Core AI を通して Neural Engine で動かすだけなので、macOS 27 が要ります (macOS 26 以前では、そう書いたエラーで止まります)。モデルは Core AI 向け (`.aimodel`、モデルのディレクトリの `coreai`) に変換し、コンパイルした結果は Core AI が自分のキャッシュに持つので、2 回目からの読み込みは 0.1 秒です (docs/design.md 16.191)。`--cpu` は確かめ用に、Core ML の書き出しを CPU のインタプリタで動かします。

## 音楽 (Magenta RealTime 2)

`go run ./cmd/hwnpu-audio generate "disco funk"` は Magenta RealTime 2 small で、文で指定した曲調の 48 kHz ステレオの音楽を作り、そのままスピーカーで鳴らします (`-o out.wav` でファイルに)。M1 で実時間の約 1.4 倍の速さです。重みは CC-BY 4.0 なので、使うときは「Magenta RealTime 2 by Google DeepMind」と表示してください。

- 録音の曲調: `song.wav`
- 曲調の混ぜ合わせ: `"jazz@2" "bossa nova"`
- MIDI の音符: `--notes melody.mid`
- 録音の続き: `--continue intro.wav`
- 鳴らしながらファイルにも書く: `--play -o out.wav`、標準出力に流す: `-o -`

## 効果音 (MOSS-SoundEffect)

同じ `generate` に `-m sfx` を付けると MOSS-SoundEffect v2.0 (Apache-2.0) で、英語か中国語の説明から最長 30 秒の 48 kHz の効果音を作ります。文の符号器 (Qwen3-1.7B)、DiT (1.3B)、DAC の復号がすべて NPU で動き、M1 でカードの 100 ステップなら 10 秒の音に約 4 分です。マシンのメモリで、16 GB 以上は文の符号器が fp16 のモデル、8 GB は lut8 のモデルを選びます (`-m moss-soundeffect:8gb` や `:16G` で明示)。

```bash
go run ./cmd/hwnpu-audio generate -m sfx "A dog barking loudly in a park."
go run ./cmd/hwnpu-audio generate -m sfx --duration 20s --steps 50 "Rain on a tin roof."   # --steps 50 で半分の時間
```

## 読み上げ (LFM3-Audio-2B)

`go run ./cmd/hwnpu-audio tts --lang ja -o out.wav こんにちは` は LFM3-Audio-2B (`-m speech`) で文章を読み上げます。詳しくは [CHAT.ja.md](CHAT.ja.md) の 5 節にあります。

## 書き起こし (Whisper)

`go run ./cmd/hwnpu-audio transcribe speech.wav` は Whisper large-v3-turbo で音声を書き起こします。初回にチェックポイントを落として変換します。M3 で 30 秒の窓を 0.6 秒で読み、65 秒の日本語を 4 秒で書き起こしました。`-l ja`、`--translate`、`--timestamps` が使えます。

## 音高と音符 (SwiftF0、Basic Pitch)

`go run ./cmd/hwnpu-audio pitch voice.wav` は SwiftF0 で声や楽器の音高を 16 ms ごとに、`go run ./cmd/hwnpu-audio notes --midi song.mid song.wav` は Basic Pitch で音符を MIDI に出します。どちらも ONNX で配られるモデルで、`internal/onnx` が静的な部分を NPU のプログラムにします。AMD の機械では CPU で動きます ([USAGE_IMAGE.ja.md](USAGE_IMAGE.ja.md) の物の検出の節と同じく、ONNX Runtime があればそれを使います)。
