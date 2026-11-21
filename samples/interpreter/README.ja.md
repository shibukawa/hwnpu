# interpreter

[English](README.md) · [サンプルの一覧](../README.ja.md)

音声を別の言語の音声に通訳します。入力はマイクか録音です。日本語の音声は英語に、ほかの言語は日本語に (指定すればその言語に) します。

```
mic → vad → transcribe → heard → commands → translate → said → speech → out
```

`transcribe`、`translate`、`speech` は NPU で動くモデルです (Whisper large-v3-turbo、PLaMo 3 Translate、LFM3-Audio の読み上げ)。ほかはプログラムの処理です。`vad` は音量で音を発話に区切り、`heard` と `said` は通るものを表示し、`commands` は話した命令を訳さずに受け取ります。読み上げはほかの後ろで動くので、1 つの発話を読み上げている間に、次の発話の文字起こしと翻訳が進みます。

## 動かし方

```bash
cd samples
go run ./interpreter models prepare            # モデルのダウンロード、変換、コンパイル (約 11 GB)
go run ./interpreter talk.wav                  # 録音: 日本語の音声を英語の音声にしてスピーカーで
go run ./interpreter --mic                     # マイク: Ctrl-C か「終わり」まで
go run ./interpreter say こんにちは              # 文を訳して読み上げる
go run ./interpreter talk.wav -o english.wav   # スピーカーの代わりに WAV のファイルに
```

通訳した音声はスピーカーで流します ([oto](https://github.com/ebitengine/oto) を使い、cgo なし)。発話ごとに読み上げができたらすぐ流し、その間に次の発話の文字起こしと翻訳が進みます。`-o FILE` を付けると、代わりに WAV のファイルに書きます。

| コマンド | 何をするか |
|---|---|
| `interpreter FILE.wav` | 録音を通訳する (WAV ならサンプリング周波数もチャネル数も問わない)。`--live` はマイクと同じ速さで渡し、各段の遅れを出す |
| `interpreter --mic` | マイクの音を通訳する。`--seconds 30s` でその時間で止める |
| `interpreter mics` | 音声の入力の一覧 (番号つき)。`--device 2` で 2 番目を使う |
| `interpreter say TEXT` | 翻訳と読み上げを 1 回呼ぶ |
| `interpreter models` | この機械向けに選んだモデル、要る量、ネットワーク |
| `interpreter status` | この機械の NPU と、それを使っているもの |

オプション: `--speed 1.3` (30 % 速く。声の高さは変わらない。0.5〜2)、`--to es` (通訳先の言語)、`-l ja` (話す言語。指定しなければ発話ごとに判定)、`--voice male`、`--no-commands`。

### 話す命令

| 言うこと | どうなるか |
|---|---|
| 「変換、スペイン語」 / "switch to Spanish" | 次の発話からスペイン語に通訳する |
| 「終わり」 / "that's all" | 聞いた分を読み上げてから終わる |

## ダウンロード

| モデル | Apple | AMD |
|---|---|---|
| Whisper large-v3-turbo (文字起こし) | 1.6 GB | 1.5 GB |
| PLaMo 3 Translate 2B (翻訳) | 5.8 GB | 2.4 GB |
| LFM3-Audio 2B TTS (読み上げ) | 4.0 GB | 2.8 GB |
| **合計** | **11.4 GB** | **6.6 GB** |

16 GB の機械の場合です。この機械の量は `interpreter models` が出します。Mac ではチェックポイントを落としてここで変換し、AMD では NPU 向けの書き出しを落とします。

## 要るもの

- メモリ: 一緒に動かして約 7.5 GB。翻訳は 16 GB 以上の機械で PLaMo 3 Translate 2B です (理由は `interpreter models` が出します)。
- 速さ: M3 で、5 秒の日本語の文が 5.3 秒で 4.9 秒の英語の音声になりました (録音から、モデルは読み込み済み)。
- Mac ではターミナルのアプリにマイクの許可が要ります。最初の `--mic` で macOS が聞いてきます。

ネットワークは [main.go](main.go) で `ai/flow` を使って宣言しています。[docs/developer-guide.ja.md](../../docs/developer-guide.ja.md) に解説があります。
