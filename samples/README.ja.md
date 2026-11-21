# ai パッケージのサンプル

[English](README.md)

hwnpu の [`ai`](../USAGE_LIBRARY.ja.md) パッケージで作った小さなアプリです。使うモデルを宣言し、この機械の NPU 向けに準備し、プログラム自身の処理とつなぎます。AMD の NPU (Ryzen AI) と Apple の Neural Engine (macOS 27、Core AI) で動きます。

| サンプル | 何をするか | モデル | ダウンロード (Apple / AMD) |
|---|---|---|---|
| [interpreter](interpreter/README.ja.md) | マイクか録音の音声を、別の言語の音声に通訳する | Whisper、PLaMo 3 Translate、LFM3-Audio の読み上げ | 11.4 GB / 6.6 GB |
| [parallax](parallax/README.ja.md) | 絵を奥行きがあるように見せる。カメラの前の頭の動きに合わせて向きが変わる | Depth Anything 3、YuNet、FLUX.2 klein | 16.5 GB / 14.9 GB |
| [layers](layers/README.ja.md) | 絵を深度で層に分け、各層が隠していた部分を描き足して、Photoshop の文書に書く | Depth Anything 3、FLUX.2 klein | 16.5 GB / 14.9 GB |

ダウンロード量は 16 GB の機械の場合です (モデルは機械のメモリで選ばれます)。1 回だけで、モデルは hwnpu のキャッシュに残り、サンプルとコマンドで共有します。この機械で落とす量は `go run ./SAMPLE models` が出します。parallax と layers は同じモデルを使うので、両方動かしてもダウンロードは 1 つ分です。

## 動かし方

サンプルは独立した Go のモジュールで、このチェックアウトの hwnpu を使ってビルドします ([go.mod](go.mod) の `replace`)。このディレクトリで実行します:

```bash
cd samples
go run ./interpreter models            # モデル、ダウンロードと実行に要る量、ネットワーク
go run ./interpreter models prepare    # 先にダウンロード、変換、コンパイル (実行時に --prepare でも)
go run ./interpreter talk.wav -o out.wav
```

どのサンプルにも同じ `models` (`list`、`download`、`prepare`) と `status` のコマンドがあります。モデルは hwnpu のキャッシュに置かれ、コマンド (`hwnpu-chat`、`hwnpu-image`、`hwnpu-audio`) と共有します。どれかが準備したものは、ほかも使います。

- モデルの初回はダウンロードと NPU 向けのコンパイルで数分かかります。Apple の FLUX.2 klein の編集のプログラムは最大 1 時間ほどです (prebuilt でなくす予定)。2 回目からは数秒で始まります。
- Apple の Core AI は、コンパイルした結果をプロセスの名前ごとに持ちます。`go run` と自分でビルドしたバイナリは別々にコンパイルされるので、1 度ビルドして (`go build -o bin/ ./...`) バイナリを動かすのがおすすめです。
- モデルはこの機械のメモリに合わせて選ばれます。一緒に動かしたときの量は `models` が出します。

## 必要なもの

- hwnpu が動かす NPU: AMD Ryzen AI (Linux) か、macOS 27 の Apple Silicon の Mac。[INSTALL_AMD.ja.md](../INSTALL_AMD.ja.md)、[INSTALL_APPLE.ja.md](../INSTALL_APPLE.ja.md) を見てください。
- マイクとカメラは [hwcamera](https://github.com/shibukawa/hwcamera) から取ります (cgo なし)。Mac ではターミナルのアプリにその許可が要ります (システム設定 › プライバシーとセキュリティ)。初めて使うときに聞かれます。
- parallax の窓は [Ebitengine](https://ebitengine.org) です。Linux では Ebitengine がビルドに使う X11/OpenGL のライブラリが要ります。画面のない機械では `-tags headless` でビルドします (`render` と `track` は動きます)。

## 構成

| ディレクトリ | 中身 |
|---|---|
| `interpreter/`、`parallax/`、`layers/` | サンプル (それぞれ `main` パッケージ) |
| `internal/capture` | マイクとカメラを `ai/flow` の流れに (hwcamera)、スピーカー (oto) |
| `internal/layering` | 絵を深度で層に分ける処理。layers と parallax が共有 |

hwcamera、oto (スピーカー)、Ebitengine はこのモジュールだけの依存で、hwnpu のライブラリには入りません。
