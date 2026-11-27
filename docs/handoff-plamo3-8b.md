# 引き継ぎ: PLaMo 3 Translate 8B を M4、AMD、Intel の NPU で確かめる

残り: PLaMo 3 Translate 8B (`plamo-3-translate-8b`) を **M4**、**AMD (Ryzen AI)**、**Intel (Core Ultra 5 135U)** で確かめること。M1、M2、M3 (lut4 と lut8) は済み。結果は [docs/design/llm-models.md](design/llm-models.md)。

## 1. M4 — 30 分ほど

### 1.1 M4 の速さと読み込み

```sh
for i in 1 2 3; do go run ./cmd/hwnpu-chat -m plamo3translate:8b --temp 0 "The meeting has been rescheduled to next Monday."; done   # 1 回目は 17.1 GB の取得と変換、コンパイル
```

- 期待: 2 回目と 3 回目の読み込み (`ready:` の行までの時間) が数秒 (M3 の Core AI で 0.3〜0.4 秒)。毎回 1 分以上ならコンパイルのキャッシュから外れている。その時間と `the Neural Engine holds` の値を記録
- 生成の速さ (統計行の `tokens at N tokens/s`。M3 の Core AI で短い 1 文 14.5)。`HWNPU_PROFILE=1` で部分ごとの ms
- 測る前に `top` で `mediaanalysisd` などが Neural Engine を使っていないことを確かめる (new-model-checklist 7 節)

### 1.2 訳 (lut4 と lut8)

lut8 を基準にする (fp16 は 15 GiB で載らない)。lut8 は M2 で約 6 トークン/秒、M3 で 5.8〜6.5。M4 で lut4 と lut8 の速さと訳を比べる。この 20 文は 2 節と 3 節の比較にも使う:

```sh
cat > /tmp/s.txt <<'X'
The meeting has been rescheduled to next Monday.
Please take an umbrella with you, as it may rain this afternoon.
The new library will open in the spring, with a café on the top floor.
I'm afraid I can't join you for dinner; I already have plans.
Our team reduced the build time from twelve minutes to three.
The museum is closed on Tuesdays except during national holidays.
She said the report would be ready by the end of the week.
If the train is delayed, we will start the presentation without you.
This software runs on the neural processing unit to save power.
The ingredients should be mixed gently until the batter is smooth.
明日の会議は午後三時からに変更になりました。
この先の道路は工事中のため、迂回してください。
駅前の新しいパン屋は朝七時から開いています。
申し訳ありませんが、その日は都合がつきません。
このアプリは電池の消費を抑えるために NPU で動きます。
桜の季節には、川沿いの道が多くの人でにぎわいます。
資料は金曜日までにメールでお送りします。
彼は十年前からこの町で小さな書店を営んでいる。
雨が降りそうなので、洗濯物を取り込んでおいてください。
新しい制度は来年の四月から導入される予定です。
X
awk '{print; print "/reset"}' /tmp/s.txt > /tmp/in.txt
for f in lut4 lut8; do go run ./cmd/hwnpu-chat -m plamo3translate:8b --format $f --temp 0 < /tmp/in.txt > /tmp/out-$f.txt 2> /tmp/err-$f.txt; done
diff <(grep '^> .' /tmp/out-lut4.txt) <(grep '^> .' /tmp/out-lut8.txt)
```

- M1、M2 の結果: 13/20 が同じ、違う 7 文は言い換え。違う文の数が大きく増えたり、誤訳 (数字、否定、固有名詞の取り違え) が出たら、docs/design/ の話題のファイル ([design/README.md](design/README.md)) に書いて lut6 を試す
- lut8 の変換は 7.6 GB 増える。比べたら `~/Library/Caches/hwnpu/models/plamo-3-translate-8b/plamo3-r1-lut8-*` を消す

## 2. AMD (Ryzen AI、IRON の環境) — 1〜2 時間

hwnpu-chat が初回に書き出す (`amd_export: plamo3 --max-seq-len 8192 --q4`、`chat.PrepareAMD`):

```sh
go run ./cmd/hwnpu-chat -m plamo3translate:8b "The meeting has been rescheduled to next Monday."
```

確かめること (new-model-checklist 4.1 節):

- **書き出しのピークのメモリ**: 17 GB の bf16 のチェックポイントから書き出す。`/usr/bin/time -v` の Maximum resident set size。16 GB の機械で足りるか (Qwen3.6 の 40 層は 53.8 GB だった)。足りなければ models.yaml の `machine` か書き出しを見直す
- **コマンドの 2 秒**: 2B は 1024 トークンの塊を 3 コマンドで、位置 7168 から 10.5 秒 ([amd-llm](design/amd-llm.md))。8B は計算が約 3.5 倍。`ERT_CMD_STATE_TIMEOUT` が出たら、書き出しの `SCRATCHPAD_WORDS` を下げてコマンドを分ける。`HWNPU_DEV=iron.timing=1` で一番長いコマンド
- **デバイスのヒープ (64 MiB)**: `CREATE_BO: invalid argument` か `ENOSPC` が出たら、制御コードが載っていない ([amd-llm](design/amd-llm.md))
- **固定メモリ**: プロセスの `/dev/accel/accel0` の写像 (`smaps`) + RSS の最大 (new-model-checklist 11.2 の例のスクリプト)。見積もりは約 6.5 GB。models.yaml の `memory.amd` を実測に
- **速さ**: デコードは 150〜200 ms の見込み (2B の q4 は 89 ms)。`ironlm -forced` の ms/ステップ
- **HF との照合**: `tools/ref/forced_logits.py` で HF fp32 と (README、GPL-3、wiki40b-ja の 400 位置)。8B の fp32 は約 34 GB 要るので、64 GB 級の機械で。bf16 の書き出しの KL が 0.001〜0.014 なら移植は正しい (2B と同じ)
- **訳**: 1.2 の 20 文を q4 で。2B の q4 は bf16 と 36/44 が同じだった ([amd-llm](design/amd-llm.md))
- GPTQ (`q4_gptq.py plamo3`) は fp32 で約 34 GB。丸めで訳が保たれるなら要らない

## 3. Intel (Meteor Lake、Core Ultra 5 135U) — 1 時間ほど

8B の Intel の既定は行ごとの int4 で、1 本 (models.yaml の `prefill: off`) で約 3.8 GiB。int8 (7.6 GiB) は、Anima で `OUT_OF_HOST_MEMORY` になった約 4.6 GB ([intel](design/intel.md)) を超えるので使わない:

```sh
hwnpu-chat -m plamo3translate:8b --temp 0 "The meeting has been rescheduled to next Monday."
```

- 行ごとの int4 は Meteor Lake で int8 より速い (LFM3-1.7B で 37.9 ms と 68.5 ms、[intel](design/intel.md) の hwnpu-formats)。グループごとの int4-g128 は int8 より遅いので使わない
- 精度: M1 の Neural Engine で同じ int4 の重みを使うと、20 文は lut8 と 12/20 が同じで、残りは言い換えだった ([llm-models](design/llm-models.md))。Intel では計算の精度が違うので、1.2 の 20 文 (`--format` は付けない) で、Mac の lut8 の訳と比べる。崩れるなら int4-g128 (遅いが精度は上) か、NPU から外すかを決める
- 記録: 読み込みの成否と時間 (Windows の TDR の 2 秒に触れないか)、tok/s、NPU のメモリ。NPU のメモリが尽きるとエラーでなくプロセスが 0xc0000005 で落ちることがある ([intel](design/intel.md))

## 4. 記録

- 結果は [docs/design/llm-models.md](design/llm-models.md) に足す。速さとメモリは MODEL.md / MODEL.ja.md の表、models.yaml の `memory`、[docs/design/performance.md](design/performance.md) / [.ja.md](design/performance.ja.md) のモデルの表
