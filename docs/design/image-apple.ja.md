# Apple と Intel の画像のモデル

[English](image-apple.md)

画像生成のモデルを Apple の Neural Engine と Intel の NPU で動かす形: モデルごとのプログラム、DiT の注意の形、fp16 の落とし穴、重みの形式、プログラムの読み込み方、今の数字。モデルの一覧、サンプラ、CFG、アスペクト比、残すモデルは [image-models.md](image-models.md)。AMD の形は [image-amd.md](image-amd.md)。Core AI の仕組み (`.aimodel`、キャッシュ、prebuilt、NE のコンパイル) は [apple-coreai.md](apple-coreai.md)、Core ML の上限は [apple-coreml.md](apple-coreml.md)、Level Zero と OpenVINO IR は [intel.md](intel.md)。

## バックエンドと、モデルごとに動くところ

| モデル | Apple (Core AI) | Intel の NPU | パッケージ |
|---|---|---|---|
| Anima-Turbo、Anima-Aesthetic | あり | あり (OpenVINO IR。実機の通しは未確認) | `internal/imagegen/anima` |
| FLUX.2 klein 4B | あり (文からの生成。編集は編集のプログラムで、下) | なし | `internal/imagegen/flux2klein` |
| FLUX.3 klein | あり (文からの生成) | なし | `internal/imagegen/flux3klein` |

- Mac の `hwnpu-image` は Core AI だけで動く (macOS 27)。`--backend coreml` や `HWNPU_BACKEND=coreml` も Core AI を開く (`cmd/hwnpu-image/backend.go` の `imageDevice`)。Core AI のない Mac には `hwnpu-image` のデバイスがない。
- Core ML の書き出しは `hwnpu-image convert --backend coreml` で今も書ける。`--cpu` が回すのはこれ: CPU のインタプリタ (`milcpu`) の float32 で、PyTorch のない機械の参照になる。`HWNPU_DEV=cpu.f16` は CPU で演算ごとに fp16 に丸める。
- モデルごとに MIL の記述が 1 つある (`internal/imagegen/<model>/mil`)。それを Core AI の `.aimodel`、Core ML のパッケージ、OpenVINO IR (`ovlower`、Anima だけ) に書く。パイプライン (トークナイザー、サンプラ、CFG、まとめ描き) は AMD と共通で、違うのはエンジン (`programs.go` の `progEngine`) だけ。
- 初回: `Prepare` が足りないチェックポイントのファイルを落とし (`chat.Fetch`、sha256 は [../../models.yaml](../../models.yaml) に固定)、Go だけで変換し、最後に `NAME.done` を書く。書き出しの名前にはプログラムを変えるもの (`ExportVersion` を含む) が入る。形が変われば名前が変わり、変換し直す。古い書き出しは手で消すまで残る。
- どのモデルも Apple の書き出しを開く前に Neural Engine のロックを取る (`llm.LockNPU`、「waiting for the Neural Engine: process N … is using it」)。取らないと、Neural Engine を使う別の hwnpu のプロセスがいるときに VAE の段が `Inference failed — request aborted (underlying=0x15)` で落ちた。[apple-coreml.md](apple-coreml.md) を参照。

## 書き出し: 1 つのモデルに 1 つで全部の大きさ

決まり ([image-models.md](image-models.md)、[new-model-checklist.md](new-model-checklist.md)): モデルと NPU の会社ごとに書き出しは 1 つで、models.yaml の全部のアスペクト比を持ち、重みは 1 回だけ置く。大きさごとの書き出しも、一部の大きさだけをまとめた書き出しも作らない。

- Apple の形: 大きさでグラフが変わる部品は、大きさごとの関数 `s{W}x{H}` を持つ 1 本の `.aimodel` にする (`llm.WriteAIFunctions`: 同じバイトの重みは 1 回だけ)。manifest の `sizes` に大きさごとの部品の名前を書き、`ProgramManifest.At` で選ぶ。名前の大きさの部分は `aspect.Tag` (3 つ以上は `8sizes-` と大きさの CRC-32)。機種ごとの prebuilt は書き出し 1 つにつき 1 つで、全部の大きさを持つ。
- 払うもの (ダウンロードの重複のほうが重いので受け入れた):
  - Core AI は 1 本の `.aimodel` の関数を全部まとめてコンパイルする。初回は全部の大きさをコンパイルする。VAE は prebuilt にしない (下) ので、その大きさは初回に端末でコンパイルする。
  - Neural Engine のコンパイラが 1 つの大きさを通さないと、その部品の全部の大きさが NE から落ちる。置く前にチップごと、大きさごとに `coreai-build` でオフラインで確かめる。
  - 大きさを足すと、書き出しと prebuilt を全部作り直す。
- 大きさによらない部品 (文の符号器) は関数 1 つ。モデルや書き出しの間で共有する部品はハードリンクにする (`ConvertOptions.LinkFrom`、`linkFrom`、`TextVersion`)。Core AI のキャッシュに当たり、コンパイルし直さない。キャッシュはプログラムのバイトで引き、変換は毎回パッケージに新しい UUID を書くので、別々に変換した同じ部品は 2 回コンパイルされ、2 回持たれる。

今の状態 (移す途中):

| モデル | 今の Apple の書き出し |
|---|---|
| Anima | 大きさごと (`anima-turbo-v1.1-WxH-int8-int8-g4-q512-v4.coreai`。パッチが 8 の倍数でない大きさは `-v5`) |
| FLUX.2 klein | 512² + 1024² で 1 つ (7.5 GB。別々なら 7.3 + 7.5 GB)。`npu.apple.sizes` のほかの 6 つは別 |
| FLUX.3 klein | 512² と 1024² × 文の 80 行と 160 行で 1 つ (`DefaultSizes`、`DefaultTextRows`) |

移す順: Anima-Turbo → Anima-Aesthetic → FLUX.2 klein (`npu.apple.sizes` の 8 つ) → FLUX.3 klein (Chroma1 は外した: [not-adopted/chroma1.md](../not-adopted/chroma1.md))。チップは M1、M2、M3 (h13g、h14g、h15g) から。

## prebuilt にしないもの

`coreai.LeftToDevice` は、名前が `vae` か `text` で始まるプログラムを端末でコンパイルさせる: `hwnpu-dev models package MODEL/coreai-ARCH` は外し (「left to the device」)、`coreai.Prebuilt` は取得元にすでにあるそれらの prebuilt を使わない。

| 部品 | 理由 |
|---|---|
| VAE の段 (FLUX.2 klein、Anima。Chroma1 も) | `coreai-build` の FLUX.2 klein の VAE は、512² で 112 行ごとに数行の色の縞を出した (同じ `.aimodel` の 1024² の関数は正しい)。1024 × 768 の `vae_s0` は `ANEProgramProcessRequestDirect() … Request cancelled` で落ちた。そもそも得がない: 端末が prebuilt を仕上げる時間はコンパイルと同じ (1024² の 10 段で 104 秒対 106 秒)。仕上げは 1 つごとに ANELargeModelCompilerService の起動を待つ (10 秒) |
| 文の符号器 (`text_*`) | prebuilt の Chroma1 (外した) の T5-XXL の `text_0` は `Request cancelled` で 4 回続けて落ちた。FLUX.2 klein の Qwen3 は prebuilt で動くが、端末でコンパイルしたものより遅い (112 秒対 81 秒) |

- prebuilt の VAE ですでに描いた機械では、仕上げた項目が Core AI のキャッシュに残り (`hasEntry` が prebuilt より先)、縞のままになる。`~/Library/Caches/coreai-cache/<OS のビルド>/hwnpu-image/` を消す (VAE の項目だけでもよい)。
- 未解決: `coreai-build` の何が VAE を誤らせるか (112 行の周期は段の中の行の区切りかもしれない)、何が T5 を cancel させるか。直ったら `LeftToDevice` を外す。Apple に報告する (FB)。

## モデルに共通の形

### DiT の注意 (Core AI)

1024² の DiT は 4096 の画像のトークンに注意を向ける。時間は積ではなく、スコアの上を通る要素ごとの演算 (スケール、max、引き算、exp、和、逆数) にある: Core AI では softmax を `exp(x − max) × reciprocal(sum)` と書く (Core AI の注意の演算は NE に載らない)。16 × 512 × 4096 のスコアを 1 回通ると約 11 ms。だから下の形は、通る回数を減らすか、並べ方を変える。

| 形 | 使うところ | 効果 (M1、1024²、1 ブロックの注意の中身) |
|---|---|---|
| 注意のスケール 1/√d を q の RMSNorm の重みに。`SDPAScale` が 1 なら積を置かない | どのモデルも (q を RMS norm する)。FLUX の文のパスも同じ重みを読む | Anima の自己注意 102 → 89 ms、4 ブロックの組 −4.7 %。milcpu との誤差は同じ (RoPE は線形) |
| `milpart.HeadAttention`: ヘッドを 1 つずつ、塊のスコアを (1, 問い合わせ, 1, キー) に reshape、softmax は最後の軸 | 画像の自己注意。問い合わせ 1024 ずつ (`HeadChunk`)、割り算は V との積の前 | Anima 90 → 72 ms。2 つの条件がどちらも要る: 同じ並べ方で全部のヘッドを 1 つのテンソルにすると前より遅く、塊を小さくするだけでも遅い |
| キーを 4096 以下の 2 のべき乗の塊に分け (`milpart.KeyParts`)、画像のキーと文のキーを別のスコアのテンソルに (`HeadOptions.Split`)。softmax は 1 つ (最大値は塊の最大値の最大、和は塊の和の和)。文のパディングの bias は文の塊だけに | FLUX.2 klein、FLUX.3 klein (文の行が同じ注意に入る)。最初は Chroma1 (外した) | 1 スコアあたり、キーがちょうど 4096 で 0.27 ns、2304〜8192 では 0.38〜0.47 ns。Chroma1 205 → 139 ms (24 ヘッド、4608 キー)。ダブルの組 1.68 → 1.40 秒、シングルの組 3.47 → 2.91 秒 |
| 交差注意: 全部の問い合わせを 1 度に、和で割るのを V との積の後に (`Deferred`) | Anima だけ (文のキー 512) | 14 → 10 ms。割る前の積が最大 8.7 なので安全。`TestBlockRanges` は余裕が 8 倍を切ると落ちる |
| `Deferred` の問い合わせを 0 の行で `milpart.DeferredRows` (8) の倍数に詰め、後で捨てる (`paddedDeferred`) | Anima | M2 からの NE のコンパイラ (h14g、h15g、h16g) は、問い合わせの数が 8 の倍数でない `Deferred` の注意を GPU に回す (`no compiled region`)。h13g は通す。直る大きさは 880 × 1168、752 × 1360 とその縦横の入れ替え (書き出し `-v5`、`PaddedExportVersion`)。ほかの大きさはバイトまで同じで `-v4` のまま。M2 の 3:4 は 1 ステップ 4.3 秒、全部 NE |

分かった決まり:
- 自己注意では割り算を後に回さない: Anima の最後のブロック (27) で積が fp16 の最大に届く。避けようと V の int8 の重みに 1/256 を畳むと絵が変わった (fp16 の正規化の範囲を下回る積を NE が落とす)。
- NE から外れた書き方: 2 つの塊を同じ和で割る形 (和の逆数を掛ける形にする)、幅の違う塊の最大値どうしの `maximum` (`Interleave should be matched for dual-input PEs`。最大値を連結してからもう 1 度 `reduce_max`)。Qwen-Image では問い合わせの塊 2048 も外れた。
- Intel はいつもヘッドをまとめた形 (`QChunkAttention`)。
- Core ML の NE では、1 つのスコアのテンソルが約 64 MiB を超えると 1 スコアあたり 2〜3 倍遅い (65 MiB と 68 MiB の間に段差)。そのため `mil.DitQChunk` は変換のときに、16 × 塊 × パッチの数 × 2 バイトが 64 MiB 以下になるまで問い合わせの塊を半分にする (Anima のヘッドをまとめた形: 1024² で 512、1280² で 256、2048² で 128)。内部のメモリの大きさが原因ではない: 1 スコアあたりの速さは 0.5 MB から 64 MiB まで平ら。

### fp16 の落とし穴と手当て

| 落とし穴 | 手当て | どこで |
|---|---|---|
| 残差が fp16 を超える (Anima の DiT 2.7e5、Chroma1 の T5 2.8e5 と DiT の文の行 2.4e5、FLUX.3 の文の流れ 9.7e6) | 縮めて持つ: Anima `DitScale` 1/32 (パッチの埋め込みとゲートを 1/32 に)。FLUX.2 klein の DiT 1/32、`TextScale` 1/8。FLUX.3 は画像 1/32、文 `TextStreamScale` 1/1024 で、ブロックの出力 (1/32) を文に足すゲートを 32/1024 倍 | DiT と文の符号器の全部 (外した Chroma1 は `T5Scale` 1/64、`DitScale` 1/32) |
| RMSNorm や LayerNorm の 2 乗があふれる。固定の倍率では小さな行が非正規化数になる | 先に行ごとの最大の絶対値で割る (`milpart.AbsScaled`、`AbsScaledLayerNorm`。`abs` は 1 演算、前の `MaxScaled` は `maximum(x, −x)`) | DiT の norm の全部。定数にはできない: Anima の q と k は 384 に達し、行の RMS は 1400 倍に開く (`mil.BlockRanges` で測った)。重みから決めた倍率でも 700 倍残る |
| 最大値で割ると、外れ値のないチャンネルが非正規化数の近くになる (T5 の行には他の 1000 倍のチャンネルがある) | 2 乗の平均は割った写しから取り、出力は元の x にスカラーを掛けて作る (Chroma1 の `mil.t5Norm`) | Chroma1 の T5 (Chroma1。外した: [not-adopted/chroma1.md](../not-adopted/chroma1.md)) |
| 重みに畳んだ縮尺で、積が fp16 の正規化の範囲 (6e-5) を下回り NE が落とす | 縮尺は出力の側に: o の 1/64 は積の後。wo は 1/8 を重みに、1/8 を後に。norm の出力を 16 倍して射影に入れ、射影の後で 16 で割る (`t5Up`) | Chroma1 の T5、外した (前はプロンプトの 3 行がコサイン 0.93、0.75、0.36) |
| Core ML が線形の前後の定数倍を重みに畳み直す | `t5Up` の手は縮尺の後に足し算が続く形でだけ効く。DiT では何も変わらず、最後のシングルの組が壊れた (戻した) | Chroma1 の DiT (外した) |
| 型なし定数の整数の割り算: `1/64 = 0`、`8/32 = 0` | 浮動小数の定数で書く。単体テストで見る (`TestCtx`) | Chroma1 (外した) の T5 の FF の出力が消えた。FLUX.2 klein の文の行が全部 0 |
| 行ごとの int8: NE は符号のままの積和を取って最後に行の尺度を掛けるとみられ、大きな入力で途中の値が fp16 を超える → inf / NaN | lut8 (表を fp16 の重みに戻してから掛ける)。同じ int8 のプログラムは CPU のインタプリタでは正しい | Chroma1 (外した) の T5 の層 4〜11 と後半のダブルブロック。Qwen3-VL の注意の吸い込み (NaN)。だから注意の吸い込みのある文の符号器と、FLUX の DiT は lut8 |
| 16384 を超える次元は CPU に回る | 分ける: FLUX.3 の `context_embedder` の入力 20480 を 2 つの入力と列の半分ずつの線形に。FFN の `linear_in` 18432 を `#gate` と `#up` に分けて読む。Sana の K/V の出力は層を軸 1 に置いた | |
| 1 行のプログラム (Anima の `dit_time`、入力 (1, 1, 2048)) や小さいプログラム (`dit_patch`、linear 1 つ) は CPU に回る | 8 行にする (`mil.TimeRows`、正弦波は 0 行目)。小さい部品は隣の組に入れる (パッチの埋め込みを `dit_blocks_0` に、norm_out と proj_out を最後の組に) | |
| 時刻の 1 行から作る変調は NE でずれる | ホストの float32 で計算する (FLUX.2 と FLUX.3 klein の時刻の埋め込みと共有の変調、`time.safetensors`、σ ごとに 1 回) | |

### Apple の重みの形式

| 部品 | 形式 | 理由 |
|---|---|---|
| Anima の文 (Qwen3-0.6B) と DiT | int8 / int8 | DiT の int8 は lut8 より fp16 に近い (潜在のコサイン 0.902 対 0.858、M3)。Core AI で、注意の中身を抜いたブロックが int8 97.4 ms、fp16 111.9、lut8 111.6 (NE は int8 を戻す計算を払わず、読むバイトが半分)。fp16 は NE が持つ量が倍 (2.62 → 5.2 GiB) |
| FLUX.2 klein の Qwen3-4B、DiT | lut8、lut8 | int8 では注意の吸い込み。lut6 は絵が変わる |
| FLUX.3 klein の Qwen3-VL、DiT | lut8。DiT の表は活性に合わせる (`lut8x2`) | 下 |
| context_embedder (FLUX.2 の `Ctx`、FLUX.3 の `ctx`)、VAE | fp16 | プロンプトか絵に 1 回だけ: ステップの間は持たないので、ビットを上げても只 |

- 速さ: Core AI で int8 と lut8 のどちらが速いかは形による。Anima は int8 が lut8 より 13〜17 % 速く、Qwen-Image (隠れ 4096) は速くならなかった (組 369 対 361 ms、毎秒 6.2 TFLOP で NE の上限)。速さのために形式を変える前に、そのモデルの組で測る。
- 蒸留した DiT には 4 ビットは足りない (Chroma1 の MLP を lut4 にすると姿勢が変わるか崩れた。外した)。
- 形式そのものは [quantization.md](quantization.md)。

### プログラムの読み込み: DiT の組の入れ替えと先読み

- Core AI では DiT の組を持ったままにしない。組を順番が来たときに読み込み、回して閉じる (`runner.Swap`。Anima の `progEngine.swap`、FLUX の `runner.Ring`)。組が回っている間に次を読む (順に回す部品は `runner.InTurn`、DiT は `runner.Ring`: 最後の組の後は次の呼び出しのために最初の組を読む。使わなかった読み込みは閉じる)。開くのは 2 本まで。先読みに失敗したら、順番が来たところで 1 回だけ読み直す (`TestRing`、`TestInTurn`)。
- 理由: Core AI のキャッシュからの読み込みは 0.02〜0.3 秒で、1 回の実行と同じほど。全部持っても速くならない。大きな組は別のプログラムの後に来るとどのみち切り替えの費用を払う (`aicheck cycle`: Anima の 4 ブロックの組が別の組の後で +50〜70 ms。小さいプログラム 2 つ、組と小さいもの 1 つでは +0)。たくさん持つとかえって遅いか失敗する: Chroma1-Flash の 1024² で 8.8 GiB を全部持つと 1 割遅い。Qwen-Image の lut4 の DiT を 5.2 GiB 持つと `underlying=0x15` で失敗した (Core AI は Core ML のように追い出さずにエラーにする)。
- 文の符号器はプロンプトの後に、approximator はステップの前に、最後のステップの間に次のために読んだ組は VAE の前に閉じる。VAE の段は 1 段ずつ読む (1 段 10〜20 ms、先読みは要らない)。
- Core ML と Intel は 1 本ずつ読む (Intel は入るだけ持ち、メモリが足りなければほかを閉じる)。
- 読み込んだ直後 (または別のプログラムの後) の 1 回目の呼び出しは遅い: VAE の段と文の符号器の組はいつも 1 回目。
- 複数のプロンプトをまとめて (`--batch`、`GenerateAll`、`DecodeAll`、`EncodeAll`): 文は全部のプロンプトで 1 回、DiT の組ごとに全部の絵、VAE の段ごとに全部の絵。[image-models.md](image-models.md) を参照。
- 1024² のメモリの最大は DiT ではなく VAE の最後の段 (プロセスの float32 のバッファ)。FLUX.2 klein は、全部の σ の行を作ったら時刻の埋め込みの重み (0.7 GB) を捨て、ステップのごみを VAE の前と段ごとに返す (`debug.FreeOSMemory`): footprint 5.8 → 4.1 GB。

| M1、1024² で測った値 (Core AI) | 全部持つ | 入れ替え、先読みなし | 入れ替え + 先読み (既定) |
|---|---|---|---|
| Chroma1-Flash (外した): 1 枚 / NE が抱える量 | 6 分 16 秒 / 8.80 GiB | 6 分 11 秒 / 1.80 GiB | 5 分 36 秒 / 2.53 GiB |
| Anima-Turbo: 1 枚 / NE が抱える量 / プロセスの最大 | 49 秒 / 2.53 GiB / 3.88 GB | — | 49 秒 / 0.57 GiB (組を 2 本持つ間 0.84) / 3.32 GB |

どの形でも絵はバイト単位で同じ。

### 文の符号器

| モデル | 符号器 | プログラム | 注 |
|---|---|---|---|
| Anima | Qwen3-0.6B | `text` (1 本) | fp16 の 1.0 GB は Core ML の 1 プログラムの上限を超えた。int8 |
| FLUX.2 klein | Qwen3-4B の先頭 27 層 | 3 層ずつ 9 本 (`mil.TextGroups`: 9、18、27 層の後の隠れ状態がそれぞれ 1 本の出力)。`Ctx` が 3 つを並べる | 512 行。プロンプトの行は因果的、パッドの行はプロンプトのキーだけを見る (`kbias` と因果のマスク)。DiT がパッドの行も読むので省けない |
| FLUX.3 klein | Qwen3-VL-4B の言語部分、36 層のうち 32 層 | 4、8、…、32 層の後の隠れ状態を並べる (20480) | 文の行は 80 の倍数に詰める (80 か 160)。パッドの位置 ID は 1。マスクは FLUX.2 klein と同じ (transformers 5.23 と照合) |

- 文の符号器はプロンプトに 1 回。残差の縮尺が注意の吸い込みを収める (FLUX.2 の ctx で Qwen3 が 21700、Qwen3-VL が 11264)。
- 決まった前置きのキーと値をプログラムの定数にする形 (`mil.TextFrom`、`mil.TextKV`) は Sana のために作った。今のモデルには使うものがない: FLUX.2 klein のチャットのテンプレートの前置きは 3 トークン。

### VAE

- 段ごとに回す (Anima 10 段、FLUX 10 段)。段ごとに読み、回したら閉じる。各段は前の段の出力を直接読む。
- 拡大は `upsample_nearest_neighbor` (Anima の段 5 を単独で 1275 → 412 ms、reshape と tile の 3 倍速い)。OpenVINO IR は reshape と tile のまま。
- FLUX の GroupNorm (`fluxvae.GroupNorm`): グループの値を (1, 32, a, b) に並べ (どの軸も 16384 以下)、グループの最大値で割り、2 乗の前に平均を引く (|平均| / 標準偏差は 6.3 に達する)。512² の VAE は diffusers の float32 と 66.3 dB。
- FLUX.2 と FLUX.3 klein: FLUX.1 のデコーダ (`internal/imagegen/fluxvae`。元は Chroma1 のもの) に `post_quant_conv` を足したもの (`fluxvae.Vae.PostQuant`)。バッチノルムの統計で戻すのと 2 × 2 の展開はホスト。
- Anima の VAE の値は 8 未満で、fp16 の手当ては要らない。
- prebuilt にしない (上)。
- 段の間を float32 でなく fp16 で渡しても効かなかった (0〜0.3 秒、Sana の 1024²)。

## Anima (Cosmos-Predict2 2B、Qwen3-0.6B)

- プログラム: `text`、`dit_time` (8 行)、DiT の 4 ブロックずつの組 (`dit_blocks_N`、28 ブロックで 7 本。パッチの埋め込みは `dit_blocks_0`)、最後の層、VAE の 10 段。Apple は int8 / int8 (`anima.Formats`)、Intel は fp16。
- 注意: 自己注意は `HeadAttention` を 1024 ずつ、割り算は先。交差注意は全部を 1 度に、割り算は後、問い合わせは 8 に詰める。スケールは q の norm の重みに。norm は `abs` で。`ExportVersion` 4、`PaddedExportVersion` 5。
- `--group` (既定の 0 は `anima.DefaultGroup`): Apple は 4。`--qchunk` 512 (Core ML と Intel のヘッドをまとめた形)。
- 1024² の 1 ブロックの内訳 (M1、Core AI、注意の形を変える前): 232 ms。MLP 54 ms (毎秒 5.1 TFLOP)、自己注意の中身 102 ms (毎秒 1.3 TFLOP)、交差注意の中身 14 ms。注意のほかの要素ごとの演算は 221 ms のブロックのうち約 24 ms (11 %): q と k の RMSNorm 11.3 (うち最大で割る分 4.8)、RoPE 3.4、LayerNorm の最大で割る分 3.7、厳密な GELU 5.8 (tanh と sigmoid の近似は遅い: +4.7、+12.5 ms)。
- CFG のステップはちょうど 1 回だけのステップの 2 倍 (11.1〜11.2 秒と 5.6〜5.7 秒): 組が小さく (0.28 GiB) 切り替えの費用を払わないので、組ごとにまとめて回しても速くならない。
- 精度: Anima-Turbo の 512² を、同じ int8 の重みを CPU の float32 で回したものと比べて、潜在のコサイン 0.987〜0.999。diffusers の float32 とは 1024² (M1、Core ML、int8) で 0.876〜0.948、構図は同じ。8 ステップに蒸留したモデルは小さな差を広げる。Anima-Aesthetic (小さなステップ 25 回) は離れない (注意の形どうしで 0.997〜0.9999)。

## Chroma1 (外した)

Chroma1 (Flash、HD) は 2026-11-24 に外した。プログラム、形式、数字は [not-adopted/chroma1.md](../not-adopted/chroma1.md) にある。

## FLUX.2 klein 4B

- プログラム: Qwen3-4B の 9 本 (lut8)、`Ctx` (fp16)、DiT はダブルブロック 2 つずつ 3 本とシングルブロック 5 つずつ 4 本 (lut8)、VAE。
- DiT は Chroma1 (外した) の形から: バイアスなし、SwiGLU、変調はブロックで共有 (ダブルの画像 6 行、文 6 行、シングル 3 行、norm_out 2 行) でホストが σ ごとに計算。シングルブロックの融合した射影はチェックポイントの部分の名前で読む (`to_qkv_mlp_proj.weight#q`、`mil.Weights`)。RoPE は 4 軸 × 32、θ 2000、文の行は 4 軸目の位置。注意は 1 ヘッドずつ、画像のキーと文のキーを別に、マスクなし (DiT はパッドも見る)。
- 512² + 1024² を 1 つの書き出しに。潜在は大きさごとの書き出しとビット単位で同じ。
- 精度 (512²、最初のステップの速度。同じプログラムを CPU の float32 で回したものと、`flux2klein.TestF32Check`): 相対 RMS 3.8 % = lut8 2.7 % + NE の fp16 2.8 %。組ごとには 0.2〜0.8 % で、25 ブロックで積み重なる。diffusers とは比べていない (その Mac に PyTorch がない)。
- 当てはまらないもの: CFG (蒸留済み)、文のキーと値をステップの外に (文の行はステップごとに変調される)、決まった前置き (3 トークン)、文の行を減らす (DiT がパッドも見る)。

### 編集: 同じ書き出しに (`edit.go`)

- 編集には、文からの生成にないプログラムが要る: 絵の行と参照の行を通す DiT の組 (`mil.Dit.Refs`。1024² で等倍の参照 1 枚なら 8704 行、文からは 4608 行) と、VAE の符号器。これを全部の大きさの**1 つの書き出しに入れる**: models.yaml の `npu.apple.edits` がアスペクト比ごとに編集を挙げ (`ConvertOptions.Edits`、`AppleOptions`)、DiT の各プログラムは大きさごとと編集のステップごとに関数を持つ (`EditFunction`、`s1024x1024_ref1h2`)。重みは 1 回。8 つの大きさ + 編集 5 つ = 13 関数 (2026-11-22)。ただしキーが 8192 以上の編集 (DiT の行。1024² で 1024 の参照があると 8704) はそこに入れない: `coreai-build` はそれだけのキーのアテンションをどのチップでも GPU に回し (Metal Toolchain 27.2、`NEFUSED_EW: KernelPreScales`。キーを別のテンソルに分けても、部分ごとに softmax を閉じて後で重みを掛けても同じ。8064 なら通る)、そういう関数が 1 つあるとプログラム全体の prebuilt がなくなる。そのステップは別のプログラム (関数 1 つずつ、`dit_double_0_device.aimodel`、`flux2klein.PrebuiltKeys`、`DeviceSuffix`) にし、`coreai.LeftToDevice` が端末のコンパイラ (こちらは通す) に任せる。最初のその編集でコンパイルする。組の重みをもう 1 度持つ (約 4 GB 増える)。主のプログラムは 10 関数 (8 つの大きさと 512² の編集 2 つ) で prebuilt があり、1024 の参照の 3 つの編集は端末でコンパイルする。書き出しの版は v2。編集は、1024² の絵に 1024 の参照 1 枚、1024 + 512、1024 + 512 × 2。512² の絵に 512 の参照 1 枚か 2 枚。符号器の段は別のプログラムで、1024² と 512² の関数を持つ (512 の参照はどちらの絵でも同じ関数)。マニフェストでは `edits` (`EditSize`)。名前に `-5edits-ハッシュ` が付く (`flux.2-klein-4b-8sizes-4d02f40a-5edits-fa7f0afe-lut8-lut8-v1.coreai`)。
- v5 (2026-11-24) から、8192 キー以上の attention はキーの部分ごとの softmax を log-sum-exp で合わせる (`milpart.PieceKeys`)。この形は coreai-build が Neural Engine 向けにコンパイルできるので、1024² を 1024 の参照 1 枚で編集するステップ (8704 キー) も同じプログラムの関数になり、`-device` の書き出しはなくなった。8 サイズ + 5 編集 = 13 関数で、M2 と M3 では全部プリビルト (v6: 単ブロックは 1 プログラムに 2 個)。
- **M1 は自分用の書き出しを持つ** (2026-11-25、models.yaml の `npu.apple.generation_edits`)。1024² の絵の参照が 1 枚のとき、参照を 512 で読む (`1@half`、5632 キー)。これで 13 関数のどれも 8192 キー未満になる。coreai-build が h13g 向けに作った 8704 キーの関数は、M1 で正しく動かない ([known-issues 13 節](../known-issues.md))。単ブロックのプログラムは仕上がるが雑音を計算し、二重ブロックのプログラムは仕上げのサービスが落ちる。Core AI はそのプログラムを丸ごと端末でコンパイルし直す (13 本のうち 1 本で 54 分)。同じ関数を M1 の上でコンパイルすれば正しく、coreai-build の 4608 キーと 6656 キーの関数も正しい。この編集だけを別のプログラム (v2 から v4 の `-device` の書き出し) に分ければ機能は残せるが、ダウンロードが 3.9 GB 増え、初めて使うときに 2 時間ほどコンパイルがかかる。M1 はメモリもストレージも一番少ない機械なので、分けずに参照を小さく読むことにした。`--image FILE` は、書き出しの表から参照の大きさを自分で選ぶ (`ai/imagegen`。ログに `reference 1 at 512x512` と出る)。出来上がる絵は 1024² のまま。ほかのチップの書き出しと AMD は変わらない。`hwnpu-dev models package` は、チップごとの書き出しを、それを使うチップの種類だけにプリビルドする (`chat.TakesExportsOf`)。
- それまでは編集を別の書き出しにしていた (`ConvertEdit`、`PrepareEdit`、`Pipeline.UseEdit`、`flux.2-klein-4b-edit-…`)。最初の編集で変換し、文からの生成ではどれもコンパイルしないため (Core AI は `.aimodel` の関数を全部まとめてコンパイルする)。取得元の prebuilt があれば初回もコンパイルしないので、2 つ目の書き出しは DiT の重みをもう 1 度 (4 GB 多く) と 2 つ目の prebuilt を持つだけになり、なくした。ほかの大きさの編集は断る (`References` が書き出しの動かせるものを挙げる)。
- 参照のあるステップの DiT: 画像の流れは絵の行、続いて参照の行 (変調は同じ行)。参照の RoPE の最初の軸は 10、20、…。`HeadAttention` の分け目は画像の流れの終わり。`proj_out` は絵の行だけ。`TestDitReferences` が diffusers のブロックを float64 で書いたものと比べる (相対 RMS 3e-4、参照なしと同じ)。
- 符号器 (`fluxvae.VaeEncStages`、7 段: 絵の大きさとその半分では残差ブロック 1 つずつ、ほかのダウンブロックはまとめて、中のブロックと頭)。ダウンサンプラーは (0, 1, 0, 1) を埋める (`milpart.Conv2DPad`)。`quant_conv` は `conv_out` の最初の 32 チャネル (平均) に畳み込む。diffusers と (`tools/ref/ref_flux2klein.py --encode`、狐の 512²): 相対 RMS 0.39 %、コサイン 0.99999。
- 数字 (M3 16 GB、キャッシュあり、前の別の書き出し): 512² で参照 1 枚の 1 ステップ 3.0 秒。`samples/layers` の 1 層の描き足しは 512² で 39 秒、1024² で 89 秒 (参照の符号化を含む)。

## FLUX.3 klein

- DiT 5.55B: 流れごとの mode ブロック (それぞれ 5 層、別の重み、注意は自分の流れの中だけ、注意と FFN を同じ変調の入力から並列に計算し `x + gate (attn + ff)`)、その後に [文, 画像] の single ブロック 28 層。FFN は SwiGLU で、`linear_in` の前半が gate。
- 変調はホスト (14 行): 画像は `silu(temb(t) + temb'(t'))`、文は `silu(temb(0) + temb'(t'))`。flow-map の目標時刻 t' が文にも入るので、文の mode ブロックも毎ステップ回す。`norm_out` は scale、shift の順。timestep は bf16 に丸めて 1000 倍し、もう 1 度丸める (パイプラインと同じ)。diffusers との差は相対 RMS 1e-7 (`TestModulation`)。
- RoPE: 4 軸 × 32、θ 10000、画像の行と列は 0.5 倍 (`image_rope_scale`)、文は (0, 0, 0, i)。`RopePerm`。
- 書き出し: 512² と 1024² × 文の 80 行と 160 行を 1 つに (文、`ctx`、文の mode ブロックは行数ごと、画像の mode ブロックと VAE は大きさごと、single ブロックは組ごと)。Qwen3-VL と DiT は lut8、`ctx` と VAE は fp16。変換 2 分 37 秒 (表の作成 1 分 23 秒)、9.4 GB。初回のコンパイルは約 50 分 (single ブロックの組 1 つ 6.5 分。512² の初回に 1024² もコンパイルする)。
- DiT の lut8 の表は活性に合わせる: 入力の列ごとの E[x²] を重みにした Lloyd を、一様 (最小と最大) の段から始める (`palette.QuantizeWeightedMinMax`。書き出しの名前に `lut8x2`)。統計は書き出しのフォルダの `flux3-klein.x2.safetensors`、なければ取得元の `original/hwnpu/flux-3-image-klein-calibration/` (Hugging Face にはない。SHA-256 はコードに固定、`StatsSHA256`)。どちらにもなければ一様で警告する。作るのは `ref_flux3klein.py --stats` (int8 の DiT、評価とは別の 4 プロンプト)。

| DiT の形式 (MPS の bf16 の参照、512²、最後の潜在のコサイン) | ポートレート | 8 プロンプトの平均 / 中央値 |
|---|---|---|
| 行ごとの int8 (クリップ点を探す) | 0.9967 | 0.988 / 0.996 |
| lut8 一様 | 0.964 | 0.956 (3 枚) |
| lut8 分位点からの k-means | 0.56 (壊れる) | — |
| lut8 重み付き分位点からの重み付き k-means | 0.93 | — |
| **lut8 一様からの重み付き Lloyd (E[x²]、校正 4 本)** | **0.998** | 0.981 / 0.997 |

- 分位点から始めると、表のいちばん上の段が 99.8 % 点になり、少数の大きな重みが潰れる。重みなしの k-means を一様から始めても一様と同じ。校正 12 本でも 4 本と同じ。1 ステップの速度の誤差は自由に回した結果を予測しない (single ブロックの注意だけ int8 にすると速度は 2.33 → 2.29 % でも、潜在は 0.964 → 0.983)。
- fp16: 画像の流れ 1/32、文 1/1024 (上)。パディングの行の文の状態が参照から 50 % ずれる (プロンプトの行は 3〜7 %): パディングの位置 1 は注意の吸い込みの 2 行目と同じで、Qwen3-VL の lut8 の誤差を拾うとみる。絵には目立たない。
- M3、512²、参照と同じノイズ: 37.0 dB、最初のステップの速度のコサイン 0.9997 (相対 RMS 2.5 %)、最後のステップで 0.993。

## 今の数字 (Core AI、2 回目の起動から、キャッシュあり)

| モデル | 機械 | 512² | 1024² | メモリの最大 (プロセス + NE) |
|---|---|---|---|---|
| FLUX.2 klein 4B (4 ステップ) | M1 16 GB | 10〜11 秒 (文 0.9、1 ステップ 2.2〜2.5 秒、VAE 1.2)。`--batch 4` で 1 枚 9.0 秒 | 41 秒 (1 ステップ 9.1〜9.2 秒、VAE 4.7) | 1024² で 4.93 GB (VAE で 3.83 + 0.76 GiB。NE の最大は DiT で 2.32 GiB)。512² で 3.09 GB |
| FLUX.3 klein (16 ステップ) | M3 16 GB | 29 秒 (1 ステップ 1.7 秒) | 2 分 02 秒 (文 1.0 秒、1 ステップ 7.4 秒、VAE 3.8 秒) | footprint 4.1 GB。NE は未計測 (models.yaml の 5.5 GB は見積もり) |
| Anima-Turbo (8 ステップ) | M1 16 GB | ステップ 7.5 秒 | 48〜49 秒 (1 ステップ 5.6〜5.7 秒、VAE 2.8〜3.2 秒) | 4.81 GB (3.06 + 1.42 GiB) |
| Anima-Turbo 3:4 (880 × 1168) | M2 | | 8 ステップ 34 秒 (1 ステップ 4.3 秒。M1 は 5.4 秒) | |
| Anima-Aesthetic (25 ステップ、CFG は 17) | M1 16 GB | | ステップ 3 分 51〜57 秒 (CFG のステップ 11.0〜11.2 秒、ほか 5.6〜5.7 秒) | Anima-Turbo と同じ |

- FLUX.2 klein v6 を M1 で、M1 用の書き出しのプリビルトから (2026-11-25、macOS 27.2.1。上の表の行は v1 から v5 で、端末でコンパイルした、単ブロック 5 個で 1 プログラムのもの):

  | | 時間 | 1 ステップ |
  |---|---|---|
  | 空の Core AI キャッシュからの初回 (512²) | 18 分 15 秒。DiT の 13 本はプリビルトから仕上がる (二重ブロックの組 1 分 35 秒、単ブロックの組 1 分 16 秒)。VAE は端末でコンパイル (1 分 21 秒) | |
  | 512² | ステップと VAE で 15 秒 | 3.5 秒 |
  | 1024² | 58 秒 | 12.3〜12.4 秒 |
  | 1280 × 704、704 × 1280 | 49 秒 | 10.0〜10.1 秒 |
  | 768 × 1024 | 44 秒 | 9.0 秒 |
  | 1024² の編集、参照 1 枚 (512 で読む) | 1 分 29 秒 (参照は最初に使うとき 16 秒。エンコーダーを端末でコンパイルする) | 16.0 秒 |
  | 1024² の編集、512 の参照 2 枚と 3 枚 | 1 分 29 秒、1 分 44 秒 | 19.9 秒、23.4 秒 |
  | 512² の編集、参照 1 枚と 2 枚 | 28 秒、43 秒 | 5.5 秒、9.0 秒 |

  プリビルトの失敗は 1 つもなく、DiT は端末で 1 本もコンパイルしなかった。1024² の 1 ステップは 12.4 秒で、端末でコンパイルした v5 (単ブロック 5 個の組) の 9.1 秒より遅い。1 ステップに呼ぶプログラムが 7 本から 13 本に増えたため (単ブロック 2 個の組は 4608 行で 1.0 秒、二重ブロックの組は 1.0〜1.3 秒)。M1 用の書き出しで単ブロックの組を大きくすれば、いくらか戻せる (3 個の組は h13g 向けにオフラインでコンパイルできるが、512 の参照 3 枚の編集だけは通らない)。
- FLUX.3 klein を M1 で、取得元の書き出しと h13g のプリビルトから (2026-11-25): 512² は 1 分 14 秒 (1 ステップ 4.0〜4.4 秒)、1024² は 4 分 45 秒 (1 ステップ 16.8〜17.2 秒、VAE 9.3 秒)。初回は 49 分 33 秒 (書き出しの取得 7 分 25 秒を含む。オフラインのコンパイル 3 本と同時に走らせた)。プリビルトの失敗はなかった。
- 初回 (変換と NE のコンパイル): FLUX.2 klein 40 分 (両方の大きさ)、FLUX.3 klein 約 50 分、Anima-Turbo の 1024² 約 21 分。初回のメモリの最大は大きい (FLUX.2 klein の 1024²: 7.00 GB)。
- 測る実行ファイルは `hwnpu-image` という名前にする: Core AI のキャッシュは実行ファイルの名前ごとで、別の名前では全部をコンパイルし直す ([apple-coreai.md](apple-coreai.md))。本番のキャッシュから部品 1 本の時間を測るには、`tools/coreai/aicheck` を `hwnpu-image` という名前でビルドし、`run -calls N 書き出し/部品.aimodel#関数`。
- メモリは `ri_phys_footprint + ri_neural_footprint` (`proc_pid_rusage`) の最大。`mem.NPUWired` は機械全体の値で、ほかのプロセスの分も入る。[memory-and-power.md](memory-and-power.md) を参照。
- M3 では低電力モードで画像生成が半分の速さになった。M1 では差がなかった。

## Intel の NPU (Anima)

- 同じ MIL から OpenVINO IR (`ovlower`)。重みは fp16、注意はヘッドをまとめた形 (`QChunkAttention`、塊は `DitQChunk`)、拡大は reshape と tile。
- 組の大きさ (`anima.DefaultGroup`): 4 から半分ずつにして、ブロックの数 × パッチの数を 4096 以下に (512² で 4、640² で 2、768² と 1024² で 1)。Windows の TDR (2 秒) は、それより長いジョブを `ZE_RESULT_ERROR_DEVICE_LOST` で止める (Core Ultra 5 135U、1024²、4 ブロック)。DEVICE_LOST のとき、組が 2 以上ならエラーに `--group` を半分にする案内を付ける (`types.IsDeviceLost`)。
- ドライバのコンパイラの「bad allocation」/「bad_alloc」はメモリ不足と見なす (`types.IsOutOfMemory`): ほかの部品を閉じて読み直す。1024² の g1 では、文と 20 ブロックを読んだ後の `dit_blocks_20` でこれに当たった。プロンプトごとに文と数ブロックを読み直す見込み。
- 確かめたこと: OpenVINO 2026.4 の CPU で、256²、8 ステップ、diffusers の float32 と潜在のコサイン 0.99985、37.9 dB。512² の int8 の g4 と g1: 同じ絵で、潜在の差は 3.6 % (組の間の受け渡しの丸め)。Core Ultra 5 135U の 512² の g4: 1 組約 0.4 秒、1 ステップ 2.8 秒。

## 外したモデル (Sana、Qwen-Image) から残す教訓

Sana と Qwen-Image はもう出していない ([image-models.md](image-models.md) の採らなかったもの。再開するときのメモは [not-adopted/sana.md](../not-adopted/sana.md) と [not-adopted/qwen-image.md](../not-adopted/qwen-image.md))。今も当てはまること:

- Core AI の NE のコンパイラは幅の広い畳み込みを通さないことがある: Qwen-Image の 1152 → 1152 チャンネルの VAE の段は 64 × 64 以上で (M1、`Allocation failure for L2-dep pair`) GPU に回った。同じパッケージは Core ML なら載った。畳み込みをチャンネルや行で分けても直らなかった。段の入力を重なりのある窓 (48 × 48 以下) に分けて貼り合わせると、窓の内側の端から段が届く距離より離れた画素だけを使えば厳密に一致する (重なりは 2 枚で 7、3 枚以上で 14)。FLUX の VAE (最大 512 チャンネル) には要らない。
- まとめた `.aimodel` の関数が 1 つ失敗すると、ファイル全体が GPU に回る: バックエンドが組をほどき (`ungroup`、`record.Alone`)、単体で下ろし直す ([apple-coreai.md](apple-coreai.md))。
- `coreai-build` の手当て (Sana): プログラムの末尾の `x × 1` で、定数に重みの名前がついた出力の融合を切る (`redefinition of symbol named 'hwnpu_N'`)。チャンネル方向の最大はチャンネルを最後の軸に移してから取る。VAE の段は 1 段に 1 ブロック。群の 1 × 1 の畳み込みはブロック対角の密な 1 × 1 に。チャンネルの繰り返し (`tile`) は 0 と 1 の重みの 1 × 1 の畳み込みに (`mil.repeatChannels`)。
- 128 行の文の関数は、NE を 1〜2 分止めてから `Request cancelled` で失敗し、その後しばらくどのプロセスも同じ失敗になった (何も動かさずに置くと戻る)。同じ作りの 64 行と 320 行は動いた。原因は分からない。
- 少ないステップに蒸留したモデル (Sana-Sprint、Anima-Turbo) は条件の小さな差を広げる: 形どうしは、1 枚の潜在のコサインでなく、float32 に対する行の誤差と数枚の絵の構図で比べる。
- メモリの少ない機械での参照: `tools/ref/stream_weights.py` は繰り返しの層の重みを使う直前に読む (8B の float32 が数 GB で回る)。

## 採らなかったもの

| もの | 理由 |
|---|---|
| `hwnpu-image` の Core ML | Mac では Core AI だけ。Core ML の書き出しは `--cpu` のために残す |
| VAE と文の符号器の prebuilt | 絵が誤る / `Request cancelled`、時間も減らない (上) |
| Anima: 7 ブロックの組 | 1 ステップ −1.2 % だけで、プログラムが大きく 8 GB の機械に載りにくい |
| Anima: q、k、v を 1 本の線形に (2048 → 6144) | 遅くなる (組 928 → 946 ms) |
| Anima: 自己注意の正規化を P·V の後に | 最後のブロックで fp16 があふれる。V に倍率を入れると絵が変わる |
| キーを先にしたスコア (1, 16, キー, 問い合わせ) | 89 ms に対して 128 ms |
| Anima: 交差注意の短いキーの塊 (厳密。sink と同じく最後の 0 の行に ln(n+1)) | キー 256 で組 −2.3 % だが、512 行の関数も要り、関数は全部一緒にコンパイルされる (初回のコンパイルが倍)。保留 |
| Anima: 交差注意のキーと値をプロンプトごとに 1 回 (Sana の形) | 組あたり計算が約 3.5 ms 減るが、呼び出しごとに入力が 16 MB 増える。減るのは NE の重みの 117 MB だけ |
| Anima: GELU の tanh と sigmoid の近似 | 厳密な erf より遅い |
| norm の前の最大で割る分を定数に | 範囲が 1400 倍に開く |
| 速さのために重みのビットを上げる (Anima の fp16 や lut8) | int8 より遅く、NE が持つ量も倍 |
| Core AI で DiT の組を全部持つ | 速くならず、1024² では最大 1 割遅く、NE のメモリが増え、失敗しうる |
| FLUX.2 klein の DiT を lut6 | 1 ステップ −9 % だが潜在 0.957 で絵が変わる |
| FLUX.2 klein の組を大きく (ダブル 3、シングル 7) | 1 ステップ 2.2 秒で変わらない (先読みが切り替えを隠す)。コンパイルが長い |
| FLUX.3 klein の lut8 を分位点からの k-means で | 絵が壊れる (0.56) |
| Core ML で次のプログラムを先に読む | Core ML では読み込みがコンパイルで、NE のコンパイルはほぼ 1 本ずつ。DiT を 4 本同時に読むと遅くなり 7.4 GB 使った |
| VAE の段の間を fp16 で渡す | 0〜0.3 秒、ばらつきの中 |
| Anima のサブピクセルの畳み込みでの拡大 (`depth_to_space`) | `upsample_nearest_neighbor` と同じ速さ。重みの並べ方を合わせきれていない |
| softmax の行の最大を引かない | 重みからの上限 (層ごとに 2〜221) が fp16 の exp の限界 11 を超える |

## 未解決

- Apple の書き出しを全部、全部の大きさを持つ 1 つに移す (順は上)。先にチップごとに各大きさを `coreai-build` で確かめる。
- `coreai-build` が VAE を誤り T5 を cancel させる理由。直ったら `LeftToDevice` を外す。
- Apple の FLUX.3 klein: Neural Engine のメモリ、編集、ほかの大きさ、161 トークン以上のプロンプト、パディングの行のずれ (Qwen3-VL の先の層を fp16 にするなど)、校正の統計の配り方。
- Apple の FLUX.2 klein の編集。
- Intel の NPU の Anima: 1024² の実機での通し、v4 と v5 の書き出しの Intel での確かめ。Anima-Aesthetic の範囲 (`TestBlockRanges`)。
- M2 と M3: ブロックの切り替えの費用 (`aicheck cycle`)、問い合わせの塊の大きさ、M3 と M4 での詰めた `-v5` の大きさ。1 枚の電力量 (powerbench は sudo が要る)。
- 1 プロセスの最初の部品の後、prebuilt の仕上げが 1 本約 10.2 秒の床になる件 ([apple-coreai.md](apple-coreai.md))。
