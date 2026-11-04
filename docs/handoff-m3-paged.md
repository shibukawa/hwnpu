# 引き継ぎ: LFM3 を注意の前後で分けた形 (`-paged`) の M3 での確認

作成: 2026-11-04。Linux (aix1) での作業を、Neural Engine のある Mac (M3、16 GB) に引き継ぐためのメモ。

## 1. ここまでの状態

- ブランチ: `t3/2e10556d` (origin に push 済み)
- 主なコミット
  - `ed838e8` 変換 (`hwnpu-convert -paged N`)、`internal/llm/paged`、`decode.Session` の塊ごとの注意、`internal/llm/chain` の読み込み
  - `01d219f` hwnpu-chat と hwnpu-agent の既定化 (LFM3 は塊 8192 で分け、文脈 262144)、メモリの上限、CPU の runner
  - `61fafb9` 実物の LFM3-1.7B での CPU 確認 (`TestPagedCheckpoint`)、CHAT.md と MODEL.md、design.md 16.59
  - その前: `npuprobe attn` と `npuprobe seam` の計測 (design.md 16.58)、Core ML の `Bind` の修正 (`e8fa3fd`)
- Linux で確かめたこと
  - 小さなモデル: 分けたものと分けないもののロジットの相対誤差は 8.6e-4 以内 (`TestPagedMatchesFull`、`TestPagedShareRewindResume`、`TestEngineReusePaged`)
  - 実物の LFM3-1.7B (fp16、CPU のインタプリタ): 45 トークン、6 ステップすべてで次のトークンが一致、相対誤差 1.7e-3 以内
- **まだ確かめていないこと (この引き継ぎの目的)**: Neural Engine への配置、速さ、メモリ、読み込みの時間、コンパイルのキャッシュに残るか、長い文脈での想起

読む順: [docs/design.md](design.md) の 16.58 (計測) と 16.59 (実装)、[CHAT.ja.md](../CHAT.ja.md) の 2 節。

## 2. 準備

```sh
cd <hwnpu の checkout>
git fetch origin
git checkout t3/2e10556d
git pull
git log --oneline -1   # 61fafb9 以降であること
```

古い既定のプログラムはもう読まれません。場所を空けたいときだけ消します (`~/Library/Caches/hwnpu/models/LFM3-1.7B/lfm3-lut6-grouped-c16320-*`)。比較 (5 節) に使うなら残しておきます。

## 3. テスト (数分)

```sh
go test ./internal/llm/lfm3 ./internal/llm/paged ./internal/llm/decode ./internal/llm/chain ./internal/chat ./internal/backend/coreml ./internal/probe
```

- `internal/backend/coreml` の `TestBindAfterSubmissionFinished` (付け替えの修正) が通ること
- `internal/probe` の `TestChunkAttnOnNeuralEngine` は、塊の注意の配置をログに出す (Q=8 の塊 4096 以上と Q=64 は Neural Engine、Q=1 の 8192 以下は CPU になるはず)

失敗したら、その出力をそのまま記録する (7 節)。

## 4. hwnpu-chat の初回と 2 回目

```sh
time go run ./cmd/hwnpu-chat
```

初回は新しい名前で変換し直す (`lfm3-lut6-grouped-p8192-c262144-q8.parts.json` と `-q64`。それぞれ 7 部分と注意のプログラム 1 本)。lut6 の kmeans の変換で数分、コンパイルで数分かかる見込み。

確かめること:

| 項目 | 見るところ | 期待 |
|---|---|---|
| 配置 | 読み込みのログの各プログラムの行 | 2 本とも `all on the NPU` (注意のプログラムも含めて判定している) |
| 文脈 | `ready:` の行 | `context 262144 tokens, its cache growing with the conversation by 12 KiB a token to 3.00 GiB (attention in chunks of 8192)` |
| 初回の時間 | 変換、コンパイル、読み込みのそれぞれ | 記録する |
| 2 回目の起動 | 一度終えてもう一度 `time go run ./cmd/hwnpu-chat` | 部分が 2 本から 14 本 + 注意 2 本に増えたので、Neural Engine のコンパイルのキャッシュ (約 4 GiB) に残るかが心配。毎回コンパイルし直すなら起動が長い (16.41、16.43 と同じ現象) |

`NOT all on the NPU` と出たら、どのプログラムかを調べる。注意のプログラムだけなら `-draft` が 0 か 1 (1 回 1 トークン) になっていないか確認する (塊 8192 の Q=1 は Core ML が CPU に置く)。

## 5. 速さ

ターミナルの各答えの後の `[prompt N tokens, M new in X s; K tokens at T tokens/s; ...]` の T を記録する (ms/token = 1000 / T。投機的デコードの採用率で変わるので、同じ質問で比べる)。

1. 短い会話: 富士山の高さ、17 × 23 など、これまでの 4 問 (16.39、16.41) をそれぞれ `/reset` してから
2. 比較のため、分けない形でも同じ 4 問: `go run ./cmd/hwnpu-chat -paged -1 -context 16320` (古いファイルがあれば変換なしで読み込む)
3. 長い会話 (6 節の要求の後に続けて質問する): 16K、64K、128K、256K で 1 トークンの時間

見込み (注意だけの計測から、design.md 16.58):

| 会話の長さ | 見込み | 今までの形 |
|---|---|---|
| 4K | 30〜34 ms/token | 42.8 ms (容量 16320 で固定) |
| 16K | 約 42 ms/token | 42.8 ms |
| 64K | 約 95 ms/token | 363 ms (lut4 + tile、16.22) |
| 256K | 約 320 ms/token | なし |

## 6. 長い文脈での想起とメモリ

サーバーで、先頭に合言葉を置いた長いプロンプトを送る (16.22 と同じ試験)。

```sh
go run ./cmd/hwnpu-chat -listen 127.0.0.1:8080 -prompt-cache off
```

別の端末で:

```sh
python3 - <<'EOF' > /tmp/long.json
import json
n = 128_000          # 目安のトークン数。256K を試すときは 250_000
filler = "The weather report for the region said that the morning would be clear and the afternoon cloudy. "
words = filler * (n * 4 // len(filler))   # 英語は 1 トークン 4 文字ほど
prompt = "The passphrase is BLUE-OTTER-42. Remember it.\n\n" + words + "\n\nWhat is the passphrase given at the very beginning?"
print(json.dumps({"model": "LFM3-1.7B", "max_tokens": 40, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]}))
EOF
time curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/long.json | python3 -m json.tool
```

- 読み込みの時間 (curl の `real`) とサーバーのログのトークン数を記録する。128K で十数分、256K で 30 分以上の見込み
- 答えに `BLUE-OTTER-42` が入るか
- メモリ: 読み込みの後に `footprint $(pgrep -f hwnpu-chat | head -1)` (または アクティビティモニタのメモリ) と `sysctl vm.swapusage`。KV キャッシュは 1 トークン 12 KiB (128K で 1.5 GiB、256K で 3 GiB) が足される見込み
- 読み終えた後、同じ会話で短い質問を 1 つ続け、1 トークンの時間を記録する (5 節の 3)

16 GB の機械で 256K にしてスワップが大きく増えるなら、それも記録する (エンジンはメモリの 4 分の 1 までしかキャッシュを使わない設計: 16 GB なら 4 GiB なので 262144 は縮めない)。

## 7. 記録と、うまくいかなかったとき

結果は [docs/design.md](design.md) の 16.59 の末尾に「M3 での確認」として追記し、[CHAT.ja.md](../CHAT.ja.md) と [CHAT.md](../CHAT.md) の 2 節の表の「見込み」を実測に置き換える。npuprobe と同じく、数値と条件 (機種、macOS、形式、`-draft`) を書く。

想定される問題と手がかり:

| 症状 | 見るところ |
|---|---|
| 変換が失敗する | `llm.PlanPaged` (`internal/llm/split.go`) と `convertPaged` (`internal/llm/convert.go`)。部分が 1 GiB の上限を超えていないか |
| 読み込みでエラー | `internal/llm/chain/chain.go` の `loadPaged`: 部分と注意が交互になっているか、注意のプログラムの入出力 (q, k, v, mask, o, m, l) |
| 答えが崩れる | CPU では一致しているので、Neural Engine の精度か配置。`hwnpu-chat -paged -1` と比べる。`TestPagedCheckpoint` を Mac でも動かせる (`HWNPU_LFM3_DIR=~/Library/Caches/hwnpu/models/LFM3-1.7B HWNPU_LFM3_PAGED_CPU=1 go test ./internal/llm/lfm3 -run TestPagedCheckpoint -v`、数分) |
| 起動が毎回遅い | コンパイルのキャッシュからあふれている。部分の数を減らす (今は全注意の層ごとに切るので 7 部分) 方法を検討する |
| 長い文脈で遅すぎる | 262144 での遅くなり (16.58 で一部の組に 5〜50%) と、重みと交互にしたときの注意の数 % の遅くなり (16.58 の seam)。`npuprobe attn -attn-used 262144` を空いた機械で測り直す |
| 1 回 1 トークンの注意が CPU | 仕様どおり (Core ML の判断)。`-draft 8` (既定) を使う |

## 8. コードの地図

| 場所 | 役割 |
|---|---|
| `internal/llm/paged/paged.go` | 塊 1 つに対する重みなしの注意のプログラム (`Build`、`Write`) と、ホストでの合わせ方 (`Merge`) |
| `internal/llm/arch.go` の `Paged` | 注意で分けられるモデルの約束 (`PagedLayers`、`PagedShape`、`PagedStepInputs`、`MaxContext`) |
| `internal/llm/split.go` の `PlanPaged`、`internal/llm/convert.go` の `convertPaged` | 部分の切り方と変換、manifest の `paged` と `attention` |
| `internal/llm/lfm3/build.go` の `beforeAttention`、`afterAttention` | 全注意の層を前後に分けた半分ずつ |
| `internal/llm/decode/decode.go` の `runPaged`、`stepMasks`、`part`、`Save` | 段ごとの実行、塊の確保、マスク、使った行だけの退避 |
| `internal/llm/chain/chain.go` の `loadPaged`、`attention.program` | デバイスでの読み込み、段、注意のプログラムの付け替え |
| `internal/chat/model.go` の `Prepare`、`pagedChunk` | 既定で分けて変換する判断、名前 (`-p8192`)、文脈 |
| `internal/chat/engine.go` の `fitMemory` | メモリの 4 分の 1 に収める上限、ready の行 |
| `internal/llm/runner/runner.go` の `cachedCPU` | CPU のインタプリタでの実行 (テストと NPU のない機械) |
