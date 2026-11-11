# 引き継ぎ: PLaMo 3 Translate 8B を M2〜M4 と AMD の NPU で確かめる

作成: 2026-11-11。M1 (16 GB) で入れた PLaMo 3 Translate 8B (`plamo-3-translate-8b`、[design.md](design.md) の 16.155) の、M1 では測れなかったことの確認。

## 1. ここまでの状態

- 構造は 2B と同じ `plamo3` で、コードの変更は既定の形式だけ (8B は lut4、`plamo3.Config.Format`)。Intel は int8 (`plamo3.Arch.IntelFormat`)
- M1 で確かめたこと: 変換 (5 部分、約 1 分)、全部 `all on the NPU`、初回のコンパイル 1 分 16 秒、2 回目以降 3〜5 秒、Neural Engine 3.81 GiB とプロセスの最大 4.1 GB、20 文と段落 4 つの訳 (lut8 と比べて崩れない)
- M1 では 2.8 GiB を超えるので 0.8 トークン/秒。速さは測れていない
- M2 (16 GB) で確かめたこと (16.155): 生成 9.4〜10.1 トークン/秒、2 回目以降の読み込み 35〜41 秒 (キャッシュに残るのは 1.2〜2.1 GiB だけ)、20 文は lut8 と 13/20 が同じで残りは言い換え。全部載った後も 1 回 106 ms (38 GiB/s 前後) で、生成の速さは 9〜10 トークン/秒のまま。残り: lut8 の速さと読み込み
- 比べた文: 16.155 の 20 文 (英 → 日 10、日 → 英 10) と段落 4 つ。下の 2.2 にそのまま置いた

## 2. Mac (M2、M3、M4) — 30 分ほど

### 2.1 速さと読み込み

```sh
go run ./cmd/hwnpu-chat -m plamo3translate:8b --temp 0 "The meeting has been rescheduled to next Monday."   # 初回: 17.1 GB の取得と変換、コンパイル
go run ./cmd/hwnpu-chat -m plamo3translate:8b --temp 0 "The meeting has been rescheduled to next Monday."   # 2 回目
go run ./cmd/hwnpu-chat -m plamo3translate:8b --temp 0 "The meeting has been rescheduled to next Monday."   # 3 回目
```

- 期待: 2 回目と 3 回目の読み込み (`[1/1] ... parts.json Ns`) が数秒 (M1 は 3〜5 秒)。毎回 1 分以上ならコンパイルのキャッシュから外れている。その時間と `the Neural Engine holds` の値を記録
- 期待: 生成 15 トークン/秒前後 (3.77 GiB を 63 GB/s、1 トークン約 65 ms)。統計行の `tokens at N tokens/s` を記録。`HWNPU_PROFILE=1` で部分ごとの ms
- 測る前に `top` で `mediaanalysisd` などが Neural Engine を使っていないことを確かめる (new-model-checklist 7 節)

### 2.2 訳 (lut4 と lut8)

M1 では lut8 を基準にした。fp16 は 15 GiB で載らない。lut8 (7.58 GiB) が M2 以降で全速なら、lut4 と lut8 の速さと訳を比べ直す:

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

- M1 の結果: 13/20 が同じ、違う 7 文は言い換え。違う文の数が大きく増えたり、誤訳 (数字、否定、固有名詞の取り違え) が出たら、design.md に書いて lut6 を試す
- lut8 の変換は 7.6 GB 増える。比べたら `~/Library/Caches/hwnpu/models/plamo-3-translate-8b/plamo3-r1-lut8-*` を消す

### 2.3 16 GB の M2 以降で 8B を勧めるか

M2 以降で 15 トークン/秒前後が出て、メモリ (Neural Engine 3.8 GiB + プロセス 4 GB 前後) がほかのアプリと並べて足りるなら、`machine` を 16 にすることを考える。ただし M1 の 16 GB でも `-m translate` が 8B を選ぶようになる (チップで選ぶ仕組みはない) ので、その対策と一緒に決める。

## 3. AMD (Ryzen AI、IRON の環境) — 1〜2 時間

hwnpu-chat が初回に書き出す (`amd_export: plamo3 --max-seq-len 8192 --q4`、`chat.PrepareAMD`):

```sh
go run ./cmd/hwnpu-chat -m plamo3translate:8b "The meeting has been rescheduled to next Monday."
```

確かめること (new-model-checklist 4.1 節):

- **書き出しのピークのメモリ**: 17 GB の bf16 のチェックポイントから書き出す。`/usr/bin/time -v` の Maximum resident set size。16 GB の機械で足りるか (Qwen3.6 の 40 層は 53.8 GB だった)。足りなければ models.yaml の `machine` か書き出しを見直す
- **コマンドの 2 秒**: 2B は 1024 トークンの塊を 3 コマンドで、位置 7168 から 10.5 秒 (16.70)。8B は計算が約 3.5 倍。`ERT_CMD_STATE_TIMEOUT` が出たら、書き出しの `SCRATCHPAD_WORDS` を下げてコマンドを分ける。`HWNPU_IRON_TIMING=1` で一番長いコマンド
- **デバイスのヒープ (64 MiB)**: `CREATE_BO: invalid argument` か `ENOSPC` が出たら、制御コードが載っていない (16.104)
- **固定メモリ**: プロセスの `/dev/accel/accel0` の写像 (`smaps`) + RSS の最大 (new-model-checklist 11.2 の例のスクリプト)。見積もりは約 6.5 GB。models.yaml の `memory.amd` を実測に
- **速さ**: デコードは 150〜200 ms の見込み (2B の q4 は 89 ms)。`ironlm -forced` の ms/ステップ
- **HF との照合**: `tools/ref/forced_logits.py` で HF fp32 と (README、GPL-3、wiki40b-ja の 400 位置)。8B の fp32 は約 34 GB 要るので、64 GB 級の機械で。bf16 の書き出しの KL が 0.001〜0.014 なら移植は正しい (2B と同じ)
- **訳**: 2.2 の 20 文を q4 で。2B の q4 は bf16 と 36/44 が同じだった (16.76)
- GPTQ (`q4_gptq.py plamo3`) は fp32 で約 34 GB。丸めで訳が保たれるなら要らない

## 4. 記録

- 結果は design.md の 16.155 に足す。速さとメモリは MODEL.md / MODEL.ja.md の表、models.yaml の `memory`、improvements.md / improvements.ja.md のモデルの表
