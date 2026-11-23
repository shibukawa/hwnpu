# kotoba-whisper v2.2

## 状態

- **一覧から外した (コードあり)**。models.yaml に載せたことはありません。
- Whisper を入れたのは 2026-11-08 です (#71、`e2957801`)。そのときに、同じ実装で動くことを確かめました。

## モデル

- `kotoba-tech/kotoba-whisper-v2.2` (Apache-2.0)。日本語向けに蒸留した Whisper で、デコーダは 2 層です。fp32 のチェックポイントは 3.0 GB です。
- 一覧にある書き起こしの既定は `openai/whisper-large-v3-turbo` (809M) です。

## どの config で動かしていたか

- models.yaml の項目はありません。リポジトリの名前で指定すると、書き起こしの役で読み込めます (`chat.ResolveFor("kotoba-tech/kotoba-whisper-v2.2@main", KindTranscribe, …)`。`internal/chat/table_test.go` で固定)。
- arch と形式は turbo と同じです (`whisper-encoder`、`whisper-cross`、`whisper`。lut8)。

## 実装状況

| | 結果 |
|---|---|
| Apple (M3、Core ML の時点) | Neural Engine に 0.73 GiB。生成は 1 秒 270〜290 トークン (turbo の 1.6 倍) |
| AMD、Intel | していません |

- 6.5 秒の日本語は、HF と一字一句同じでした。
- 65.8 秒の日本語は崩れました (「活か山」「世界文化産」)。HF の長い音声の方式でもほぼ同じ文に崩れるので、モデルの性質です。kotoba は、自前のチャンク分割の pipeline で使う前提のモデルです。

## ロジックが残っている場所

- Whisper の実装をそのまま使います。`internal/llm/whisper/`、`internal/transcribe/` です。
- 記録: [audio-models.md](../design/audio-models.md) の Whisper の節 (1 行)。2026-11-21 までの `docs/design.md` の 16.100 にも詳しい記録があります。

## 問題と一覧に入れない理由

- 長い音声で崩れます。長い音声には turbo を勧めています。
- 短い日本語なら速いので、用途の候補には残ります。

## 再開するときの手順と注意

1. 長い音声を、kotoba の pipeline と同じチャンク分割で読むかを決めます。今の `internal/transcribe` は transformers の長い音声の方式です。
2. models.yaml に `kind: transcribe` の項目を足します。sha256 も書きます。turbo と同じ形式で動きます。
3. AMD と Intel では、まだ確かめていません。
