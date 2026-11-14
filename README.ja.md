# hwnpu

[English](README.md)

CPU 内蔵の NPU を cgo なしで Go から使うためのコマンドです (Go のライブラリは予定)。軽量モデルを低消費電力で動かすことを目的にしています。

- 既定の厳格モードでは、NPU だけで完結しないモデルを実行前にエラーにします。CPU や GPU への黙ったフォールバックはしません
- 依存は [purego](https://github.com/ebitengine/purego) と golang.org/x/sys だけです。利用者の機械に Python は要りません

| プラットフォーム | バックエンド | 状態 |
|---|---|---|
| macOS (Apple Silicon) | Core ML (macOS 27 は Core AI も。hwnpu-image は Core AI だけ) | 実装済み。M1 / M3 で確認 |
| Linux / Windows (Intel Core Ultra) | Level Zero (OpenVINO IR) | 実装済み。実機で動かしたのは一部のモデルだけ |
| Linux (AMD Ryzen AI) | amdxdna ioctl (IRON / mlir-aie の書き出し) | 実装済み。Ryzen AI 9 365 (Strix Point) で確認 |
| Windows (AMD Ryzen AI) | XRT (xrtshim) | 実装済み。Windows では未検証 |
| WSL2 | なし | 非対応 (NPU が見えない) |

## 試す

```bash
go run ./cmd/hwnpu-chat "日本の首都は？"     # 初回にモデルを落として NPU 向けに変換する
go run ./cmd/hwnpu-chat list                 # 選べるモデルと、このマシンで選ばれるもの
go run ./cmd/hwnpu-dev report -o report.md   # この機械の NPU で hwnpu が動くかを確かめる
```

導入の手順は [Apple](INSTALL_APPLE.ja.md)、[Intel](INSTALL_INTEL.ja.md)、[AMD](INSTALL_AMD.ja.md) にあります。

## コマンド

| コマンド | できること | 説明 |
|---|---|---|
| `hwnpu-chat` | 対話、翻訳、コーディングエージェント (`agent`)、OpenAI 互換のサーバー (`serve`)、埋め込みの検索 (`search`)、型付きの判定 (`systemone`) | [CHAT.ja.md](CHAT.ja.md)、AMD は [USAGE_LLM_AMD.ja.md](USAGE_LLM_AMD.ja.md) |
| `hwnpu-image` | 画像生成 (Anima、Sana、Chroma1)、深度、顔、物、姿勢、文字 (OCR)、ラベル付け、切り抜き | [USAGE_IMAGE.ja.md](USAGE_IMAGE.ja.md) |
| `hwnpu-audio` | 音楽と効果音の生成、読み上げ、書き起こし、音高、音符 | [USAGE_AUDIO.ja.md](USAGE_AUDIO.ja.md) |
| `hwnpu-dev` | 動作確認、NPU の測定 (`probe`)、モデルの変換と照合 | [USAGE_DEV.ja.md](USAGE_DEV.ja.md)、[PROBE.ja.md](PROBE.ja.md) |

モデルは名前 (`-m gemma-4-E2B-it`)、ファミリー (`-m gemma4`)、用途 (`-m translate`) で指定でき、ファミリーと用途はマシンのメモリに合うものを選びます。選べるモデルは組み込みの [models.yaml](models.yaml) から来ていて、自分のファイルで足したり差し替えたりできます ([docs/model-table.md](docs/model-table.md))。

## ライブラリとして使う (planned)

Go のライブラリとしての API は予定の段階で、まだ固まっていません。今の形は [USAGE_LIBRARY.ja.md](USAGE_LIBRARY.ja.md) にあります。

## 文書

- [MODEL.ja.md](MODEL.ja.md): 用途とメモリの量ごとのおすすめのモデル
- [MODEL_ARCHITECTURE.ja.md](MODEL_ARCHITECTURE.ja.md): 各 NPU が扱えるデータ形式とモデルの用意の仕方
- [CLOUD_AI_COMPATIBILITY.ja.md](CLOUD_AI_COMPATIBILITY.ja.md): クラウドの AI API の機能のどれを代わりに受け持てるか
- [docs/design.md](docs/design.md): 設計と測定の記録

## テスト

```bash
CGO_ENABLED=0 go test ./...
```

## ライセンス

[Apache-2.0](LICENSE)。モデルの重みはそれぞれのライセンスに従います。
