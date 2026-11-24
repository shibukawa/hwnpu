# BonsaiAI

[English](README.md)

CPU 内蔵の NPU を cgo なしで Go から使うためのコマンドです (Go のライブラリは予定)。軽量モデルを低消費電力で動かすことを目的にしています。

- 既定の厳格モードでは、NPU だけで完結しないモデルを実行前にエラーにします。CPU や GPU への黙ったフォールバックはしません
- 依存は [purego](https://github.com/ebitengine/purego) と golang.org/x/sys だけです。利用者の機械に Python は要りません

| プラットフォーム | バックエンド | 状態 |
|---|---|---|
| macOS (Apple Silicon) | Core ML (macOS 27 は Core AI も。bonsaiai-image と bonsaiai-audio は Core AI だけ) | 実装済み。M1 / M3 で確認 |
| Linux / Windows (Intel Core Ultra) | Level Zero (OpenVINO IR) | 実装済み。実機で動かしたのは一部のモデルだけ |
| Linux (AMD Ryzen AI) | amdxdna ioctl (IRON / mlir-aie の書き出し) | 実装済み。Ryzen AI 9 365 (Strix Point) で確認 |
| Windows (AMD Ryzen AI) | XRT (xrtshim) | 実装済み。Windows では未検証 |
| WSL2 | なし | 非対応 (NPU が見えない) |

## 試す

```bash
go run ./cmd/bonsaiai-chat "日本の首都は？"     # 初回にモデルを落として NPU 向けに変換する
go run ./cmd/bonsaiai-chat list                 # 選べるモデルと、このマシンで選ばれるもの
go run ./cmd/bonsaiai-dev report -o report.md   # この機械の NPU で bonsaiai が動くかを確かめる
```

導入の手順は [Apple](INSTALL_APPLE.ja.md)、[Intel](INSTALL_INTEL.ja.md)、[AMD](INSTALL_AMD.ja.md) にあります。

## コマンド

| コマンド | できること | 説明 |
|---|---|---|
| `bonsaiai-chat` | 対話、翻訳、コーディングエージェント (`agent`)、OpenAI 互換のサーバー (`serve`)、埋め込みの検索 (`search`)、型付きの判定 (`systemone`) | [CHAT.ja.md](CHAT.ja.md)、AMD は [USAGE_LLM_AMD.ja.md](USAGE_LLM_AMD.ja.md) |
| `bonsaiai-image` | 画像生成 (FLUX.2 klein、FLUX.3 klein、Anima) と編集 (FLUX.2 klein)、深度、顔、物、姿勢、文字 (OCR)、ラベル付け、切り抜き | [USAGE_IMAGE.ja.md](USAGE_IMAGE.ja.md) |
| `bonsaiai-audio` | 音楽と効果音の生成、読み上げ、書き起こし、音高、音符 | [USAGE_AUDIO.ja.md](USAGE_AUDIO.ja.md) |
| `bonsaiai` | NPU の状態と使っているもの (`status`)、失敗したコマンドの issue の下書き (`issue`)、モデルのキャッシュ: 中身、ディスクの空き、使っていないものを消す (`cache`、`cache prune`、`cache clear`)、モデルを何 GB の機械として動かすか (`memory`) | [docs/cache-sharing-design.md](docs/cache-sharing-design.md) 7.7、7.8 |
| `bonsaiai-dev` | 動作確認、NPU の測定 (`probe`)、モデルの変換と照合 | [USAGE_DEV.ja.md](USAGE_DEV.ja.md)、[PROBE.ja.md](PROBE.ja.md) |

モデルは名前 (`-m gemma-4-E2B-it`)、ファミリー (`-m gemma4`)、用途 (`-m translate`) で指定でき、ファミリーと用途はマシンのメモリに合うものを選びます。選べるモデルは組み込みの [models.yaml](models.yaml) から来ていて、自分のファイルで足したり差し替えたりできます ([docs/model-table.md](docs/model-table.md))。メモリの多い機械でモデルをほかのプログラムと一緒に動かすときは、`bonsaiai memory LFM3-1.7B 8` で、どのコマンドでもそのモデルを 8 GB の機械の形で動かせます。

コマンドが失敗したら、`go run ./cmd/bonsaiai issue` でその issue の下書きを作れます。コマンド、エラー、ログの末尾、この機械の NPU (`bonsaiai status`) が入ります。ホームフォルダ、ユーザー名とマシン名、取得元のアドレス、トークンは伏せ、コマンドラインのプロンプトは文字数に置き換えます。下書きを入れた GitHub の新しい issue のページを開くので、そこでボタンを押すまでは何も送られません。

## ライブラリとして使う (planned)

Go のライブラリとしての API は予定の段階で、まだ固まっていません。今の形は [USAGE_LIBRARY.ja.md](USAGE_LIBRARY.ja.md) にあります。

## 文書

- [MODEL.ja.md](MODEL.ja.md): 用途とメモリの量ごとのおすすめのモデル
- [MODEL_ARCHITECTURE.ja.md](MODEL_ARCHITECTURE.ja.md): 各 NPU が扱えるデータ形式とモデルの用意の仕方
- [CLOUD_AI_COMPATIBILITY.ja.md](CLOUD_AI_COMPATIBILITY.ja.md): クラウドの AI API の機能のどれを代わりに受け持てるか
- [docs/design/README.ja.md](docs/design/README.ja.md): 設計と測定の記録

## テスト

```bash
CGO_ENABLED=0 go test ./...
```

## ライセンス

[Apache-2.0](LICENSE)。モデルの重みはそれぞれのライセンスに従います。
