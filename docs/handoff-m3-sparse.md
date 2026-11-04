# 引き継ぎ: 2 段の注意 (`-sparse`) の M3 での確認

作成: 2026-11-04。Linux (aix1) で実装し CPU で確かめた 2 段の注意を、Neural Engine のある Mac (M3、16 GB) で測るためのメモ。設計と CPU での確認は [docs/design.md](design.md) の 16.61。

## 1. ここまでの状態

- ブランチ: `t3/sparse-two-level` (main の `a4bdeed`、#35 の後から)
- 何をするか: 注意で分けた LFM3 (16.59、`-paged`) で、会話が約 48K を超えたら、全注意の層の 1 回の呼び出しが読む鍵を減らす。8 トークンのバケットの平均 (要約) を作り、呼び出しの問い合わせに対して重いバケット (kv ヘッドごと、生成で約 16K、プロンプトの読みで約 32K の鍵) と、窓 512、最初の 4 つ、呼び出し自身の鍵は正確に、残りのバケットは要約 1 つで読む
- Neural Engine の呼び出し (どれも塊 8192 の重みなしのプログラム):
  1. 全部の要約に対する注意 (今までの注意のプログラム。要約の塊の数だけ。256K で 4 回)
  2. 要約の重み (新しい mass のプログラム `*.mass.mlpackage`。1 と同じ回数)
  3. 正確に読む鍵を集めた塊 (今までの注意のプログラム、生成で 2 回、プロンプトの読みで 4 回)
  4. 正確に読んだバケットの要約 (1 で数えた分を引くため、1 回)
- 1 呼び出しの回数 (生成): 128K で 16 → 7、256K で 32 → 11。プロンプトの読みは 9 と 13
- 変換し直しは要らない: hwnpu-chat は既存の `lfm3-lut6-grouped-p8192-c262144-q8.parts.json` と `-q64` に mass のプログラムを書き足す (`llm.AddMass`、重みがないので数秒)。初回の読み込みで mass の 2 本をコンパイルする
- Linux で確かめたこと: `go test ./...` が通る。`internal/llm/decode` の `TestSparse*` (Go で直接計算した 2 段の注意と、相対誤差 1e-2 以内で一致、選んだバケットの重みが最も重いものの 99% 以上)、`internal/llm/paged` の `TestMassMatchesSoftmax`、`internal/llm/lfm3` の `TestPagedSparse` (変換したプログラムを CPU のインタプリタで、小さな LFM3 のロジットが全部を読む場合と相対誤差 1.4%)
- **まだ確かめていないこと (この引き継ぎの目的)**: Neural Engine での配置 (特に mass のプログラム)、速さ、ホスト側の処理の時間、長い文脈での想起

## 2. 準備とテスト

```sh
git fetch origin
git checkout t3/sparse-two-level
go test ./internal/llm/decode ./internal/llm/paged ./internal/llm/lfm3 ./internal/llm/chain ./internal/chat ./internal/backend/coreml
```

## 3. 起動

```sh
time go run ./cmd/hwnpu-chat
```

| 見るところ | 期待 |
|---|---|
| 読み込みのログの各プログラムの行 | 2 本とも `all on the NPU` (mass のプログラムも含めて判定する。`NOT all` なら mass の配置を疑う) |
| `ready:` の行 | `... (attention in chunks of 8192; past about 48K tokens, 16384 keys exactly and the rest as means of 8)` |
| 2 回目の起動 | コンパイルのキャッシュに残るか (プログラムが 2 本増えた) |

mass のプログラムが CPU に置かれるなら、Q=8 の形 (1, 4, 32, 8192) が小さすぎる可能性がある。結果を記録する。

## 4. 速さ (`-sparse` と `-sparse=false` を同じ条件で)

16.59 の「長い会話」と同じ手順 (サーバー、`-prompt-cache off -think=false`、先頭に合言葉、続けて 100 語ほどの生成)。64K、128K、250K で、`-sparse` (既定) と `-sparse=false` の両方。

| 長さ | 今の実測 (`-sparse=false`、16.59) | 見込み (`-sparse`、計測なし) |
|---|---|---|
| 64K | 生成 92.8 ms/token、読む 147 s | 生成 約 75 ms、読む 約 145 s |
| 128K | 165 ms/token、514 s | 約 90 ms、約 430 s |
| 250K | 320 ms/token、1752 s | 約 140 ms、約 1120 s |

見込みは 16.59 の実測から、重み 25 ms (Q=8) と 60 ms (Q=64)、塊 1 回 9〜10 ms (Q=8) と 24 ms (Q=64) で数えたもの。mass のプログラムは V の積がない分だけ軽いはずだが、同じとして数えた。ホストの処理 (要約、選択、写し) は入っていない。

ホストの時間が大きいときは、`go test -cpuprofile` の代わりに、hwnpu-chat を `-cpuprofile` 付きで (なければ pprof を足して) 測り、`decode.(*Session).attendSparse`、`choose`、`summarize`、`tailMask` の割合を見る。

## 5. 想起

- 16.59 の合言葉 (`BLUE-OTTER-42`) を 128K と 250K で。`-sparse` でも答えるか
- できれば、長い文書の途中 (10%、50%、90%) に置いた 3 つの合言葉 (16.61 の CPU の確認と同じ形)。文書はこのリポジトリの docs と Go のソースをつないで 128K に

## 6. 記録

[docs/design.md](design.md) の 16.61 の末尾に「M3 での確認」として追記し、[CHAT.ja.md](../CHAT.ja.md) と [CHAT.md](../CHAT.md) の 2 節の表に `-sparse` の列を足す。

## 7. コードの地図

| 場所 | 役割 |
|---|---|
| `internal/llm/paged/mass.go` | mass のプログラム (`BuildMass`、`WriteMass`) と、合わせ方の足し引き (`Merge.Norm`、`Sub`、`AddMerge`) |
| `internal/llm/decode/sparse.go` | 2 段の注意 (`Sparse`、`SetSparse`、`attendSparse`、`summarize`、`choose`) |
| `internal/llm/decode/decode.go` の `runPaged` | 層ごとに、2 段で読むか全部を読むかを呼び出しの回数で決める |
| `internal/llm/convert.go` の `writeMass`、`AddMass` | 変換と、既存の変換への書き足し |
| `internal/llm/chain/chain.go` の `loadMass`、`mass.program` | Neural Engine での読み込みと呼び出し |
| `internal/chat/engine.go` の `Config.Dense`、`tier.sparse`、`fitMemory` | 有効にする判断、ready の行、メモリの上限 (要約の分を足す) |
