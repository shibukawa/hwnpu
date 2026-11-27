# 引き継ぎ: Neural Engine の gather と実行時の重み (残り)

作成: 2026-11-18、整理: 2026-11-27。M1、M3、M2 の調査は済み、LFM2-8B-A1B は hwnpu-chat で動く。分かったことは [docs/design/apple-neural-engine.md](design/apple-neural-engine.md) の「Gather and run-time weights」、専門家の動かし方は [docs/design/moe.md](design/moe.md)。gpt-oss-20b の専門家は CPU の MXFP4 ([docs/design/apple-coreai.md](design/apple-coreai.md))。

**残り**: M4 での probe、ANE のコンパイラが外す理由 (非公開のログのプロファイル)、`mx_dequant` と `floor_div` の NaN、24 GB 以上の機械での結び付けの費用。

## 3. 確かめること

M2 と M3 は済んだ。残りは M4 と、ANE のコンパイラの返事を読むこと。測る前に、ほかの hwnpu のプロセスが Neural Engine を使っていないことを確かめる (`ps -ax | grep -E "hwnpu|aicheck"`)。aicheck は NPU のロックを取るので順番待ちになるが、時間はずれる。

### 3.1 M4 での配置

M1 は gather が載らず、M2 と M3 は int16 の索引で載る。M4 (手元にあれば) で同じ probe を回す。

```sh
go build -o /tmp/aicheck ./tools/coreai/aicheck
sysctl -n machdep.cpu.brand_string
rm -rf ~/Library/Caches/coreai-cache/*/aicheck     # aicheck の probe のキャッシュだけ
/tmp/aicheck probe linear gnd_const gnd_input gnd_mm gaa_input gaa_const dslice_const dslice_input dslice_mm mm_dyn mm_dyn_big mm_dyn_i8 mm_dyn_i8_big
/tmp/aicheck probe gnd16_const gnd16_mm gnd16_pts gnd16_pts_w1 gnd16_pts_4d gnd16_pts_hw gaa16_const gaa16_input
COREML=1 /tmp/aicheck probe linear gnd_const gnd_input gnd_mm gaa_input gaa_const dslice_const dslice_input argmax
```

`ane=true` なら全部 Neural Engine。**とくに見たいのは `gnd16_pts_w1` と `gnd16_pts_4d`** (M1 では「Cannot decompose layer on this architecture」)。新しい世代でここが通るかと、通ったときの次の返事 (3.3 の `log stream` で)。Core ML の行は、演算ごとに優先される装置と、ANE が対応するか (「(the Neural Engine supports it)」) を出す。**新しい世代で gather_nd か dslice が ANE に載れば、MoE の形はずっと簡単になる** (表は定数のまま、専門家の番号だけ入力)。

### 3.2 M4 での MoE の 1 層 (あれば)

M2 と M3 は済み ([apple-neural-engine](design/apple-neural-engine.md))。M4 では `aicheck moe -model gptoss -layers 4 -calls 200 -copy`、`aicheck moe -model qwen3moe -layers 1 -calls 100 -topk 6` (段差が無いか) と、LFM2-8B-A1B の hwnpu-chat (`-m LiquidAI/LFM2-8B-A1B`) の生成とプロンプトの速さ。

### 3.3 ANE のコンパイラが gather を外す理由 (sudo)

ログの理由は `<private>` で伏せられる。macOS 27 では `sudo log config --mode "private_data:on"` は「Invalid Modes 'private_data:on'」で効かない。構成プロファイル [tools/coreai/logging/private-data.mobileconfig](../tools/coreai/logging/private-data.mobileconfig) (`com.apple.system.logging` の `System` → `Enable-Private-Data`) を入れる:

```sh
open tools/coreai/logging/private-data.mobileconfig
# システム設定 → 一般 → デバイス管理 (または プライバシーとセキュリティ → プロファイル) で
# 「Unified log private data (hwnpu)」をインストール (管理者のパスワード)
/usr/bin/log show --last 1m --info | grep -c "<private>"    # 減ったか確かめる
```

`log` は zsh の組み込みに化けるので `/usr/bin/log` と書く。

検証の返事は debug の段で、`log show` には残らない。probe の間 `log stream` で拾う:

```sh
rm -rf ~/Library/Caches/coreai-cache/*/aicheck
/usr/bin/log stream --level debug --style compact --predicate 'process == "aicheck"' > /tmp/anec_stream.log &
sleep 2; /tmp/aicheck probe gnd16_pts_w1 gnd16_pts_4d gnd16_const; sleep 2; kill %1
grep -B2 -A12 "outValidationParams" /tmp/anec_stream.log      # 演算ごとの support と errorMsg
grep errorMsg /tmp/anec_stream.log | grep -v '""' | sort | uniq -c
```

終わったら戻す:

システム設定の同じ画面でプロファイルを削除する。

### 3.4 残っている小さなこと

- `anec.mx_dequant` (MXFP4) が新しい世代で使えれば、gpt-oss の専門家を 4 bit のまま入力にできるかもしれない (M3 は MX の定数も入力も Neural Engine に載らなかった。[apple-neural-engine](design/apple-neural-engine.md))
- 4 ビットを int8 に 2 つ詰めて `floor_div` で取り出す形が Neural Engine で NaN になる所 (`HWNPU_DEV=ailower.cut` で切り分ける。速さで負けるので急がない)
- 結び付けの費用が層の数で増える理由 (Core AI の覚えている配列の上限か、16 GB の機械のメモリ圧か)。24 GB 以上の機械で比べたい

## 4. 注意

- 幅の広い one-hot (128 × 16000 など) は M1 の Neural Engine を全プロセスで止めたことがある。probe の軸は 16384 以下に保つ
- 調べた文字列のダンプは 1.5 GB になる (`strings -a` を共有キャッシュの全部に)。ディスクの空きに注意
- `MPSGRAPH_PRINT_ANE_PLACEMENT_ANALYSIS`、`MPSGRAPH_DUMP_MODULE`、`MPSGRAPH_FORCE_DEVICE_PLACEMENT` などの環境変数はあるが、M1 では何も出なかった (社内向けの設定が要るのかもしれない)
