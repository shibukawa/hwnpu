# AMD の NPU の LLM

[English](amd-llm.md)

AMD XDNA 2 (Ryzen AI、NPU2) で IRON を通して動かす密な言語モデルと、その多モーダルの塔。ドライバ、実行時、full ELF、共有のカーネル、電力は [AMD](amd.ja.md)。専門家の混合 (Qwen3.6、gpt-oss) は [MoE](moe.ja.md)。重みの形式そのものは [量子化](quantization.ja.md)。モデルの構造、トークナイザー、チャットの書式は [LLM のモデル](llm-models.ja.md) と [Qwen](llm-qwen.ja.md)。

数字は断りがなければ Ryzen AI 9 365 (Strix Point) で測ったもの。

## モデル

| 書き出し (`iron_export_lm.py MODEL`) | アプリケーション | モデル |
|---|---|---|
| `gemma4` | `iron_gemma4.Gemma4Tail` (KV を共有する層を最後の行だけにした IRON の `Gemma4`) | Gemma 4 E2B、E4B |
| `gemma4mm` | `iron_gemma4.Gemma4TailMultimodal` | 画像と音声つきの Gemma 4 E2B |
| `gemma4full`、`gemma4mmfull` | IRON のままの `Gemma4`、`Gemma4Multimodal` | 照合用のみ |
| `lfm3` | `iron_lfm3.Lfm3` (hwnpu のもの。IRON にはない) | LFM3-1.7B |
| `lfm3vl`、`lfm3audio`、`lfm3omni` | `iron_lfm3vl`、`iron_lfm3audio`、`iron_lfm3omni` | LFM3-VL-2B、LFM3-Audio-2B、LFM3-Omni-2B |
| `plamo3` | `iron_plamo3.Plamo3` (hwnpu のもの) | PLaMo 3 Translate 0.8B、2B、8B |
| `d1`、`d1omni` | `iron_d1.D1Vl`、`iron_d1.D1Omni` (hwnpu の) | d1-2B、d1-Omni-2B (System One の判断モデル): LFM3-VL / Omni の full attention の層をプロンプトの塊で双方向にし、最終の隠れ状態を出す ([amd-npu-models.md](../amd-npu-models.md) 29 節) |
| `ouro` | `iron_ouro.Ouro` (hwnpu のもの。IRON にはない) | Ouro-1.4B-Thinking (評価のみで models.yaml にはない。[model-estimation.ja.md](../model-estimation.ja.md))。2.6B は書き出していない |
| `qwen35` | `iron_qwen35.Qwen35` (GDN は `iron_gdn.py`) | Qwen3.5-0.8B、2B、4B |

今の書き出しの指定は [model-recipes.yaml](../../model-recipes.yaml) の `iron_export_lm.py` の `args`:

| モデル | `iron_export_lm.py` の args (`{in}` `{out}` を除く) |
|---|---|
| LFM3-1.7B / VL / Audio / Omni | `lfm3 / lfm3vl / lfm3audio / lfm3omni --max-seq-len 12288 --q4` |
| Gemma 4 E2B、E4B | `gemma4 --max-seq-len 4096 --q4` (文だけ。カタログでは塔を書き出さない) |
| PLaMo 3 Translate 0.8B / 2B / 8B | `plamo3 --max-seq-len 8192 --q4` |
| d1-2B / d1-Omni-2B | `d1 / d1omni --max-seq-len 1024 --prefill-chunk 1024 --q4` (1 問が位置 0 からの塊 1 回) |
| Qwen3.5-0.8B、2B | `qwen35 --max-seq-len 4096 --q4 --mlp q6 --head q6 --verify 3` |
| Qwen3.5-4B | `qwen35 --max-seq-len 4096 --q4` |

書き出しの名前は `~/models/NAME.amd-npu2-q4` (または `NAME.amd-npu2`)。AMD の NPU だけの機械では `hwnpu-chat -m NAME` がそれを選ぶ。`hwnpu-chat -iron DIR` と `hwnpu-agent -iron DIR` は書き出しのディレクトリを読み、トークナイザー (`tokenizer.json`、PLaMo は `tokenizer.jsonl`) とチャットの書式もそこから取る。`NoThink` などカタログの指定は `-iron DIR` には効かない (要求で `enable_thinking: false` を渡す)。

IRON にアプリケーションのないモデルは IRON の部品 (`Linear.split`、`CausalConv1d`、`Checkpoints`、`MHA`、`SandwichLayer` …) で書く。LFM3 と PLaMo 3 はそれだけで書けた。

## 書き出し

`tools/amd/iron_export_lm.py MODEL CKPT OUT` は版を 1 つの arena にコンパイルし (IRON の `load_versions` と同じ)、ELF、重みを置いた bank の像、`manifest.json` を書く。

- 版: `decode` (1 トークン)、`prompt` (塊)、あれば `vision`、`audio`、`placed` (塔の出力を差し込むプロンプトの塊)、`verify` (数行)
- manifest には版ごとに、コマンド、引数、コマンドごとのスクラッチパッドの語、引き継ぐ値 (token、position)、ロジットの出力、全バッファの配置、プロンプトの埋め込みの encoder を書く
- 語の形: Affine、Linear (`ceil(v·比 + オフセット)`、分数)、行数の表、MHA の派生値 (演算子の定数から Go で計算、`iron.Derived`)。`TestWordsMatchIRON` が全部の語を IRON のものと照らす (`check/words.json`)
- 版によって同じ名前の重みが別の場所にある。名前で重複を除かず、版ごとに全部書く
- manifest の `restartable`: IRON の `CausalLM.recurrent` が空なら真。状態のあるモデルは `checkpoints` (状態のバッファ、`*.slots`) と `rings`
- 再配置の addend は符号なし (bank の 2 GiB を超える位置)

主な指定 (全部は `iron_export_lm.py --help`):

| 指定 | 既定 | 意味 |
|---|---|---|
| `--max-seq-len` | 4096 | 文脈 |
| `--prefill-chunk` | `qwen35` は 512、ほかの密なモデルは 1024 | プロンプトの塊 |
| `--q4` | なし | 量子化した重み ([重みの形式](#重みの形式)) |
| `--attn`、`--conv`、`--mlp`、`--head` | q8、q8、q4、q4 | `--q4` のときの部分ごとの形式 |
| `--q4-codes DIR` | なし | 丸めの代わりに `q4_gptq.py` の GPTQ の符号 |
| `--ple` | int8 | Gemma 4 の層ごとのトークンの表をホストに: int8 と fp16 の scale (E2B 4.4 → 2.2 GiB、bf16 との KL 0.00005) か bf16 |
| `--device-ple` | なし | Gemma 4 の PLE の表をデバイスに (IRON の gather) |
| `--mha-pipelines` | 8 | プロンプトの MHA のパイプライン |
| `--verify ROWS` | 0 | 下書きを確かめる版を足す |
| `--scan-window` | なし | IRON のままのスライディング窓の注意 (位置 0 から読む) |
| `--per-step` | なし | ステップごとの xclbin (full ELF のない NPU1) |
| `--command-per-step` | なし | 測るため: 1 手順 1 コマンド、`OUT/steps.json` |
| `--check N`、`--random-weights SEED` | | 照合を書く / 乱数の重み |

`HWNPU_EXPORT_STEPS=1` で版ごとの演算の並び (`<版>.steps.txt`) を書く。

書き出しの時間 (文脈 4096、塊 1024): E2B 約 6 分 (最大 RSS 19.3 GB)、画像と音声つきの E2B 10 分半、E4B 8 分 (27.4 GB)、PLaMo 3 q4 4 分半、LFM3-1.7B q4 2 分半、Qwen3.5-0.8B 2 分。

## 実行器 (`internal/llm/iron`)

- bank を `BO_SHARE` に確保し、bank の像から読み込む (ページキャッシュから、複数のスレッドで)。`Step(token, position)` と `Prompt(tokens, start)` がロジットを返し、ホストが標本を引く (デバイスが引いたトークンは使わない)
- デバイスのヒープ (Linux 7.0 の amdxdna では npu4 でも 64 MiB。512 MiB は DKMS 版だけ): デコードの run は常駐。ヒープに載らないプロンプトや塔の run は呼び出しごとに作って閉じ (約 2 ms)、デコードの run を手放し (`dropDecode`)、次のステップで作り直す。各コマンドはヒープと amdxdna の 2 秒の打ち切りに収める: 書き出しで `SCRATCHPAD_WORDS` で切る (下)
- PDI はプログラムの読み込み時に像の `main` の分を確保する (遅れて確保するとヒープに散らばり、40 MiB の連続領域が残らなかった)
- `chunkBarrier`: プロンプトの塊の前に毎回、デコードの ELF の空のインスタンス (`aiecc_empty`、制御コード 16 バイト、PDI なし) をデコードのハードウェアコンテキストで走らせる (0.3〜0.5 ms、run は 1 つ持つ)。ないと、窓の注意か Gemma 4 の tail の full ELF で、2 つ目の塊の最初のコマンドが `ERT_CMD_STATE_TIMEOUT` (60 秒) まで止まる。原因は分かっていない。`--per-step` の書き出しでは何もしない
- `stepBarrier`: 数コマンドのデコードのステップの前に、別のハードウェアコンテキストで空のコマンド ([AMD](amd.ja.md))
- `CREATE_HWCTX` は 5 秒やり直す (`internal/xdna`): 別のプロセスが終わった直後は 1 秒を超えて `invalid argument` になりうる (そのプロセスはコンテキストを 3 つ持っていた)。カーネルのログの `aie2_cmd_submit: Grab job sem failed, ret -4` は EINTR で、やり直している。シグナルで割り込まれたデバイスのヒープの mmap も ioctl と同じくやり直す
- `-idle` (`Release` / `Reload`) は固定したメモリを返し、会話をホストに残す ([メモリと電力](memory-and-power.ja.md))
- ディスク: 並べた書き出しはファイルを共有する (`iron.ShareExport`、[配布](distribution.ja.md))

### コマンドの大きさ

| モデル | `SCRATCHPAD_WORDS` | 理由 |
|---|---|---|
| PLaMo 3、LFM3 (`iron_hostple.SCRATCHPAD_WORDS`)、密な Qwen3.5 (`iron_qwen35.DENSE_SCRATCHPAD_WORDS`) | 24 | 32 では Qwen3.5-2B q4 のプロンプトの 2 つ目のコマンドが 64 MiB を超えた (`CREATE_BO: invalid argument`) |
| Gemma 4 の tail (`iron_gemma4.SCRATCHPAD_WORDS`) | 32 | 24 では IRON が共有の層を 2 つのコマンドに分け、次の塊の最初のコマンドが止まった。`install_tail_boundary` が共有の層の始まりで必ず切り、共有の層、頭、draw を 1 コマンドに置く |
| Qwen3.6 | 32 | 経路の選択がそれを前提にする (`iron_moe.SLOTS`、[MoE](moe.ja.md)) |

## プロンプトのプログラム

### 塊の大きさ

| モデル | 塊 | 理由 |
|---|---|---|
| 注意のモデル (Gemma 4、LFM3 系、PLaMo 3) | 1024 | 2048 は短いプロンプトで 2 割遅い (LFM3: 30 トークン 536 → 409 ms、300 トークン 597 → 469 ms。12288 トークン全体は 11.25 と 11.53 秒で同じ) |
| Qwen3.5 (GDN) | 512 | GDN と射影の GEMM は塊の行数で決まる。短いプロンプトは 256、長いものは 1024 が速い (0.8B q6、256 / 512 / 1024: 30 トークン 321 / 354 / 416 ms、300 トークン 911 / 664 / 750 ms、長いもの 317〜362 / 509〜565 / 575〜625 tok/s) |
| なし | 256 未満 | `flm.ProjectionGEMM` (MEDIUM) は 256 行の倍数だけ、MHA の `seq_pad` は 64 の倍数 |

Gemma 4 を 256 にするとスライディング窓の層がリングになり (IRON は `3 × リング < max_seq_len` のとき窓の層をリングにする。リング = 窓 + 塊 − 1 以上の 2 のべき)、位置だけの巻き戻しと行の写しが使えない。1024 / 4096 ならならない。LFM3 の 12288、PLaMo 3 の 8192 もならない。

### プロンプトの塊を速くするもの

| 変更 | どこを | 効果 |
|---|---|---|
| 出力層は最後の行だけ | IRON の `CausalLM` が `x[call.rows - 1]` を通す | Apple の `LastRow` に当たる |
| MHA を 8 本のパイプラインで | `--mha-pipelines` (既定 8、塊 256 では最大 4) | 塊 2〜5 倍速く、電力量 −66〜−83 %、ロジットはビット単位で同じ。書き出しのログの `prompt attention: N pipelines` とカーネルの名前の `num_pipelines` で確かめる |
| Gemma 4 の KV を共有する層を最後の行だけ (`Gemma4Tail`) | E2B は 35 層のうち後ろの 20 層 (E4B は 42 層のうち 18 層) が 13 / 14 層のキャッシュを読み、何も書かないので、塊が要るのは最後の行の出力だけ。全行はキャッシュを持つ層に通し、最後の行だけをデコードの形 (`Call(prompt=False)`: GEMV、問い 1 つの MHA、RoPE の行) で共有の層に通す。PLE の行は最後の行のもの (ホストの表) か塊の最後の `token` (`--device-ple`) | 4K のプロンプト: E2B 18.3 → 7.8 秒、218 → 97 J。E4B 22.9 → 13.7 秒。その後のステップはビット単位で同じ。プロンプトのロジットは少し違う (GEMV と GEMM: コサイン 0.99992) |
| 窓の層は窓の中だけを読む (`iron_window.py`、既定) | プロンプト: `start + chunk` で終わる `chunk + window` 個 (64 に切り上げ、E2B は 1536) の鍵と値を `Copy` で共有の作業用の状態に写し、IRON の `MHA` を掛ける。窓の層のキャッシュは `chunk` 行多い。デコード: `window + 1` 個をその場で読む (`kv_offset` を足した `MHA` の派生クラス、名前は `MHA` のまま)。キャッシュは位置で引くままなので、どの位置からでもやり直せる | 位置 3072 の窓の層の注意 112 → 66 ms。E2B 4K 7.84 → 6.87 秒 (−12 %)、PLaMo 3 q4 8K 60.3 → 53.1 秒。デコード −2 % (E2B)、−14 % (PLaMo 3、位置 8191) |
| GDN の rule は有効な行のあるブロックだけ (`gdn_zero_y`) | 有効な行のないブロックは x を読み流して 0 を書く | 演算子 27 → 5 ms (256 行に 32 行) |
| GDN の SiLU を rule の前で 1 回 (`GdnSiLU`、`iron_gdn.silu_operator`、`GDR_PRESILU`) | rule は値の列 16 本の slice ごとに行を読み直し、そのたびに q、k、v の SiLU をかけ直していた | 演算子 30.4 → 9.8 ms。Qwen3.5-0.8B の 512 の塊 0.9 → 0.5 秒。ロジットはビット単位で同じ |
| LFM3 の短い畳み込みを 1 演算子に (`ShortConvChunk`、`iron_shortconv.short_conv_chunk_operator`、`kernels/short_conv.cc`) | B·x、7 タップ、C· を 1 つに。8 列 1 コアずつ、8 行のタイル、前の 6 行を `hist` に、`valid` は RTP。in_proj は 6144 出力の GEMM 1 回。IRON の形は 1 層約 26 演算。本文の層だけ (LFM3-Audio のエンコーダは前のまま) | 参照とビット単位で同じ |
| Qwen3.5 の GDN の畳み込みと SiLU を 1 つに (`GdnConvChunk`) | 1 列 2 コア (4 コアは memtile の DMA チャネルが 10 本要り、6 本を超える)。呼び出しの行より後は 0 | 8704 チャネル: 3 行 3.99 → 0.95 ms |

今のプロンプトの時間 (`ironlm -record` の最初の呼び出し、NPU をほかが使っていないとき):

| モデル | 設定のし直し | 16 トークン | 長いもの |
|---|---|---|---|
| LFM3-1.7B q4 | 632 | 386 ms | 1010 トークン 583 ms |
| Qwen3.5-0.8B q6 | 702 | 259 ms | 300 トークン 439 ms |

パイプラインの後のほかの塊の時間: 画像と音声つきの E2B は位置 0 で 0.49 秒、3072 から 0.63 秒。E4B q4 1.04 秒 / 1.24 秒。LFM3-VL / Omni は位置 0 で 0.75 秒、11264 から 1.35 秒。Qwen3.5-0.8B q6 534〜494 tok/s、2B q6 444〜409 tok/s、4B q4 193〜214 tok/s。

短いプロンプトの残り (LFM3-1.7B、16 トークン、コマンドごと): `ProjectionGEMM` 219 ms と重みの展開 (q4 / q8 → bf16 / bfp16 を DDR に) 119 ms で 6 割。どちらも呼び出しの行に関係なく塊の全部の行で回る (`flm/gemm/op.py`: 数は memtile のロックの値にあり、呼び出しごとの語が届かない)。IRON の GEMM の作りの限界。

## デコードのプログラム

AMD のデコードは重みのバイト数ではなく演算の数で決まる: 演算が替わるたびに配列の設定を書き直す (約 40 µs + 2.3 µs/KB。密なモデルの 1 トークンの 7 割前後)。Qwen3.5-0.8B は bf16、q4、q6 で同じ時間。だから演算をまとめる。数は `HWNPU_EXPORT_STEPS=1` と `elfconfigs` で数える。

| まとめたもの | モデル | 既定 / 切り替え |
|---|---|---|
| `QGEMVAdd`: 射影の bias と残差を GEMV の出口で足す | Qwen3.5、gpt-oss | オン |
| `QGEMVGlu`: gate と up を 1 つの表で `silu(Gx)·(Ux)` | Qwen3.5 (q6 / q8)、LFM3 (q4) | オン |
| `QGEMVGlu` の GELU 版 (`Q_GLU_GELU`) | Gemma 4 | **オフ** (採らなかったもの) |
| `ShortConvStep` (`iron_shortconv.py`): デコードの短い畳み込みを 1 演算子に。状態の slot を呼び出しごとのオフセットで直接読み書きする (`short_conv_operator(slots=)`、前後の Copy なし)。in_proj は QGEMV 1 回 | LFM3 | オン |
| `SandwichAdd` (`iron_sandwich.py`、`kernels/sandwich_add.cc`): post norm、残差、次の norm を 1 コアで。二乗和は float32、bf16 への丸めは 1 回。Gemma の layer_scalar には `scale` (丸めてから積、IRON と同じ) | LFM3、Gemma 4 | オン |
| Gemma 4 の per-layer input を最初の層の前で 35 層分まとめて (35 × 256 行の GEMV 1 つ、RMSNorm 1 つ …) | Gemma 4 | オン。ロジットはビット単位で同じ |
| GDN のデコードの演算子に畳み込みとゲートつきの norm を入れる (`Qwen35Layer.fold`) | Qwen3.5 | オン |
| GDN の配列を Worker の局所バッファに (`gdn_bind`): 状態の 3 つの limb だけ (`s_hi`、`s_mid` / `s_lo` は定数の位置)。`blk`、`u_row`、`y_keep` は静的 (全部にするとプログラムが 16 KB を超えた) | Qwen3.5 / 3.6 | オン。GDN 1 回の 0 の塊 258 → 61 KB |
| GDN の列を 64 にそろえる (`iron_gdn.COLUMN_ALIGN`) | Qwen3.5 | 0.8B / 2B の融合した射影は 97 ブロック (素数) で 1 列でしか回らなかった: 571 → 217 µs |

今のデコード (断りがなければ q4 の書き出し):

| モデル | 設定のし直し / トークン | 速さ | 固定 |
|---|---|---|---|
| LFM3-1.7B q4 | 226 | 16.8〜16.9 tok/s | 1.64 GiB |
| LFM3-VL-2B q4 | 226 | 15.6 tok/s (slots の形の前) | 2.0〜2.4 GiB (VL / Audio / Omni) |
| Gemma 4 E2B q4 | 672 | 9.3 tok/s (107 ms) | 1775 MiB |
| Gemma 4 E4B q4 | 作り直していない | 5.6 tok/s | 3.53 GB |
| PLaMo 3 2B q4 GPTQ | | 89 ms (11.2 tok/s)、CPU の負荷の下 | 2.16 GiB |
| PLaMo 3 0.8B q4 | | 40〜48 ms (約 22 tok/s) | 599 MiB |
| Qwen3.5-0.8B q6 | | 70.9 ms (14.1 tok/s) | 1069 MiB |
| Qwen3.5-2B q6 | | 約 120 ms (GDN の 0 の塊の修正の前) | 1862 MiB |
| Qwen3.5-4B q4 | | 157〜162 ms (6.2 tok/s) | 3727 MiB |
| Ouro-1.4B-Thinking q6 | | 268〜357 ms (3.7〜2.8 tok/s。1 トークンに 96 層ぶん) | 2.79 GiB |

電力量: E2B q4 1.46 J/トークン、LFM3-1.7B q4 1.05 J/トークン (まとめる前)。

部品としての値 ([新しいモデルの手順](new-model-checklist.ja.md)): 2 回目の読み込み / 一番長いコマンド / デコード — PLaMo 3 0.8B q4 GPTQ 0.5 秒 / 84 ms / 40〜48 ms。Qwen3.5-0.8B q6 0.8 秒 / 85 ms / 約 90 ms。Qwen3.5-2B q6 1.5 秒 / 109 ms / 約 120 ms。

LFM3 のデコードの残り (226 回): QGEMV の設定は 1 つ 75〜97 KB (戻るたびに約 250 µs)。次は QGEMV の出口に後の演算を入れる形か、q・k の norm と RoPE をまとめる。Gemma 4 の残り: QGEMV 206、RMSNorm 102、ElementwiseMul 72、GEMV 71 (PLE の gate と projection、bf16)、SandwichAdd 70、GELU 70、RoPE 50。

## 会話の状態

- KV キャッシュはホストから読み書きできる arena の bank の中に位置ごとに並び、どの版でも配置が同じ (E2B: 15 層 × 鍵 / 値、1 位置 18 KiB)。`Mark` / `Rewind` は位置だけを動かす: 次の呼び出しが上書きする
- `iron.Model.ExportRows` / `ImportRows` が位置の範囲を写す。prompt store と `Save` は `rowSource` (`RowBytes`、`ExportRows`) を通して使う。store のキーはモデルの名前、manifest の更新時刻、bank の大きさ。2736 トークンで 1 項目 51 MB。再起動のあとの同じ要求は 13.2 秒 → 0.0 秒
- 状態のあるモデル (LFM3 の畳み込み、Qwen3.5 の GDN): IRON の `Checkpoints` は塊の境ごとに slot を持つ (slot k は位置 k × 塊までの後の状態)。`StateAt(pos)` / `SetStateAt` は pos から始まる呼び出しが読む slot だけを写し (LFM3 の 13 層で 312 KiB)、KV の行と合わせて pos の系列が全部戻る。`Prompt` は呼び出しを塊の境で切る。印、`Rewind`、`Save`、store がこの写しを持つ。`TestStateRewindOnDevice` (`HWNPU_TEST_IRON_RECURRENT_EXPORT`)
- `Resumable()` はリングのある (`Rings > 0`) か slot の合わない書き出しを拒む。エンジンは restartable でなくこれも満たさない書き出しを読み込まない
- `Engine.markTurns`: restartable でない状態のあるモデルでは、prefill が最後のターンの見出し (`<|im_start|>assistant\n`) の後にも印を付ける。チャットの型が前の思考を外すので、共通の先頭は前のアシスタントのターンまでだから
- `ironSession.StepTokens`: トークン数 × ステップの時間が塊より短ければ、短い読みをデコードのステップで読む (セッションが両方の時間を覚える)
- 媒体: 塔の soft token は負の ID で入る。store は最初の負の ID より前だけを照らし、負の ID を含む系列は書かない (ID はプロセスごとに振り直す)

## 多モーダルの塔

`--q4` でも塔は bf16。塔は soft token を共有の状態 `merged` に書き、`placed` の塊が ID と目印ごとの `before` を受けて差し込む。`merged` は塔ごとに 1 つなので、1 回のプロンプトに画像 1 枚と音声 1 本まで。

| 塔 | 形 |
|---|---|
| Gemma 4 の視覚 | 入力はホストで処理系の大きさに縮めたパッチ (48 ピクセルの塊 hk × wk、hk·wk ≤ 280。PIL 互換の bicubic、`ImageSize`)。塔の縮小は等倍になる。形のない語は取りうる (hk, wk) の 1600 通りすべてを書き出しで表にする |
| Gemma 4 の音声 | 16 kHz の波形、前に hop の分の 0 (`AudioTokens`)。語はフレーム数 (1〜3000) の表。制御コード (39.8 MiB) は常駐のデコードと並ばないので、流す間はデコードの run を外す |
| LFM3-VL の視覚 (`iron_lfm3vl.py`) | 1 つの大きさ 512 × 512 (ホストで引き伸ばす)。ゲート付きの 3 × 3 の depthwise の畳み込みは、上下に 0 の帯を付けた格子を 9 通りにずらして読み、端を消す 2 枚の重み (IRON の depthwise の `Conv2d` は k × 幅 × チャンネルの帯行列を作り、1024 チャンネルには大きすぎる)。2 次元の RoPE は表を重みに。2 × 2 の塊は 4 次元の DMA で 1 回で読む |
| LFM3-Audio のエンコーダ (`iron_lfm3audio.py`) | ホストでログメル (`lfm3audio.Features`)、最長 4096 フレーム。最初の畳み込みは `Conv2d`、残り 2 つの depthwise は 9 通りの strided な読み出し、各段の有効な行の直後の 1 行を 0 に。LFM2 の 17 層は `CausalConv1d`。ヘッド 96 次元を 0 で 128 に埋める (尺度 1/√96、RoPE は 96 列だけ、ヘッドの RMSNorm の重みと eps を 0.75 倍)。語はフレーム数の表。音声の出力はない (文字で答える) |
| LFM3-Omni (`iron_lfm3omni.py`) | 2 つの塔を HF の名前 (`model.vision_tower`、`model.audio_encoder`) で読む。画像も音声も `<|vision_start|>` と `<|vision_end|>` の間 |

- manifest の `kind` が前処理を選ぶ (`RunVision`、`RunAudio`、`PromptPlaced`)。LFM3-Audio には音声の目印がない
- バッファのアドレスに依る値 (`recording.addresses`) は manifest に書き、Go が計算する
- **塔は使うときだけ読む** (`loadBanks`、`growBanks`、`Manifest.textBanks`): bank の先頭に文の重みと作業領域があるので、decode と prompt はその分だけ確保する。塔か `placed` が初めて走る前に全体のバッファを確保し、文の分をデバイスのキャッシュと状態ごと写し、塔の分を像から読み、古い bank とそれに結んだデコードの run を放す。会話はそのまま続く。`Release` まで全体のまま、`-idle` の後の `Reload` はまた文の分だけ。NPU1 (`--per-step`) は全体を読む

| 固定 | 文だけの間 | 塔の後 |
|---|---|---|
| 画像と音声つきの Gemma 4 E2B q4 | 1826 MiB (文だけの書き出し 1775) | 2818 MiB |
| LFM3-Omni-2B | 1679 MiB | 2448 MiB |

塔の読み込みは 0.5〜0.7 秒。その間は一瞬、新旧の bank が両方固定される (E2B q4: 1.67 + 2.64 GiB)。テストは `TestLazyTowerBanks`。塔つきの書き出し 1 つで文だけの用途も足りるので、文だけの書き出しを並べない。

## 投機的デコードの確認

プロンプトの塊で 2〜8 トークンを確かめると塊 1 つ分 (デコード 3〜9 ステップ) かかる: Qwen3.5-0.8B q6 350 ms (デコード 87.6)、PLaMo 3 0.8B 345 (40)、E2B q4 411 (128.5)。そこで `--verify ROWS` (`iron_verify.py`) がデコードのステップの演算を ROWS 行で回す版を足す:

- `QGEMVRows` (`kernels/mv_rows.h`): 重みを 1 回流し、ブロックの符号を 1 回 bf16 にして全部の行に掛ける。行ごとの和と丸めは 1 行の GEMV と同じ (ビット単位で同じ)。q8 3584 × 1024: 1 行 159 µs、3 行 155〜162、4 行 188、8 行 389。q8 と q6 だけ、8 行まで (`ROWS_GEMV`)
- 埋め込みはホストの行、短い畳み込みは `causal_conv1d_rows`、RoPE は `start` からの行、注意は行ごとに鍵と値を書いてデコードの `MHA`
- GDN: デコードの演算子が 1 回の呼び出しで ROWS 行を順に回す (`gdn_operator(..., steps=ROWS)`、`GDR_STEPS`)。状態はコアの中で持ち越し、各行の後の状態を出し、最後の行を slot に、残りを `rows_state` に (3 回呼ぶと 1.0 ms × 54 回かかり、ELF も 47 MB でヒープに入らなかった)
- `iron.Model.Keep`: 一部の行だけを採るとき、ホストが `rows_state` の行 kept−1 を slot に写す (Qwen3.5-0.8B で 18 層 × 1.1 MB、1.5 ms)。余分な鍵と値の行は後で上書きされる。塊の境をまたぐ確認はデコードのステップで (`SpeculateTokens`)
- 制御コード: 確かめる版の ELF 30 MB とデコードの 26 MB を 64 MiB のヒープに両方置く。入らなければ呼び出しごとに作る (`perCall`)
- 合格: `ironlm -verify` — 3 行のロジットがデコード 3 ステップとビット単位で同じ、`Keep` の後のステップもビット単位で同じ
- Qwen3.5-0.8B q6 (デコード 82 ms のとき): 確かめる版 104.5 ms (1.27 ステップ)。プロンプトからの下書き 2 つ (`LookupDraft`) で、英語のコード 82.4 → 72.1 ms/トークン、日本語の説明 82.5 → 61.7。`hwnpu-chat -iron` は版があれば使い、`HWNPU_DEV=iron.drafts=0` で使わない。カタログで `--verify 3` を持つのは Qwen3.5-0.8B と 2B だけ

### 下書きの選択的な確認

3 行の確認は今、Qwen3.5-0.8B でデコード 1.96 ステップ分、2B で 1.67 ステップ分かかる。Ryzen AI 9 365 では全部の下書きを確かめると、散文は下書きなしより遅かった。AMD だけで使う `chat.draftPolicy` は、手前の一致の長さごとに採用率 q を数え、`1 + q + q²` が確認の費用 (既定はデコード 1.8 ステップ、3 % の余裕を加える) を超えるときだけ確かめる。確かめなかった下書きも、後のトークンと照らして q を更新する。`HWNPU_DEV=draft.policy=0` で全部確かめ、`HWNPU_DEV=draft.ratio` で費用を変え、`HWNPU_DEV=draft.trace=FILE` で判断を記録する。Apple と Intel は変わらない。

Ryzen AI 9 365 で 300 トークン、温度 0 のとき、0.8B の散文は全部確かめる 14.6 tok/s から採否を使うと 18.8 (下書きなしも 18.8)、2B は 9.6 → 11.5 (下書きなしは 11.2)。写しの多い答えの速さも保った: 0.8B のコードは 26.9 → 27.2 tok/s、2B の JSON は 14.2 → 13.6。温度 0 でも下書きの有無で文が変わることがある。原因は未調査。

下書きの側は [デコード](llm-decoding.ja.md)。

## 重みの形式

形式の定義は [量子化](quantization.ja.md)。ここでの使い方 (`--q4` のとき):

| 部分 | 形式 | デコード | プロンプト |
|---|---|---|---|
| MLP、出力の行列 | q4nx (FastFlowLM の形式、5 ビット) | `QGEMV`、1 列 2 コア、bf16 と同じ 55 GB/s | `flm.DequantBFP` + bfp16 の GEMM |
| 注意の射影 (LFM3 のゲートを含む)、LFM3 の畳み込みの射影 | q8 (int8、入力 32 ごとに bf16 の scale、8.5 ビット) | `QGEMV`、部分和を 2 つの bf16 (上位と残り) に | `Q8Dequant` / `QDequant` + bf16 の GEMM |
| Qwen3.5-0.8B / 2B の MLP と出力の行列 | q6 (6.5 ビット、q8 の並び、`kernels/q6_mv.cc`) | q8 とほぼ同じ時間 (上位 2 ビットの取り出しで律速) | `q6_dequant` |
| トークンの埋め込み | ホストの表 (`embed.bin`、`embed_rows`) | 呼び出しの前に行を書く | |
| Gemma 4 の PLE | ホストの表 (`ple.bin`、int8) | | |
| 塔 | bf16 | | |

- まとめた重み (`qkv_proj`、`gate_up_proj`、LFM3 の `in_proj`) は 1 つのまま量子化し、行の範囲を列ブロックのバイトの範囲として切る (`iron_q4.q_rows`)。デコードは QGEMV 1 回で掛けて出力を切る (分けるより 6 % 速い)
- 重みの相対 RMS 誤差: q4 0.071、q6 0.022、q8 0.005。GEMV 8192 × 2048: q4 0.30〜0.35、q6 0.42〜0.47、q8 0.43〜0.50 ms
- Qwen3.5 の出力の行列は埋め込みと共有だが量子化した写しを持つので、デコードの bf16 の表は使われておらず、ホストに移した: 0.8B −0.47 GiB、2B −0.95 GiB、ロジットはビット単位で同じ。Qwen3.6 はデバイスに残す (経路の選択が読む)
- GDN の融合した射影は列ごとに 0 の行で埋める (`iron_gdn.column_widths`: 772 → 776、さらに `COLUMN_ALIGN` 64)。量子化のブロック (出力 64 × 入力 512) に載せるため

`--q4` の効果:

| モデル | 固定 | デコード |
|---|---|---|
| Gemma 4 E2B | 9.08 → 1.78 GB | 6.1 → 7.9 tok/s (まとめる前) |
| Gemma 4 E4B | 14.7 → 3.53 GB | 3.7 → 5.6 tok/s |
| LFM3-1.7B | 3.75 → 1.64 GiB | 8.7 → 11.2 tok/s (まとめる前) |
| PLaMo 3 2B | 5.96 → 2.16 GiB | 7.3 → 11.6 tok/s |
| PLaMo 3 0.8B | 1814 → 599 MiB | 17.5 → 22.8 tok/s |
| Qwen3.5-0.8B (q6) | 1858 → 1069 MiB | 同じ (演算の数で決まる) |
| Qwen3.5-2B (q6) | 4017 → 1862 MiB (塊 512) | 131 → 123 ms |

### HF の fp32 との精度

KL(HF ‖ NPU)、README / GPL-3 / wiki40b-ja の 400 行ずつ (`ironlm -forced`、`tools/ref/forced_logits.py`):

| 書き出し | README | GPL-3 | wiki |
|---|---|---|---|
| Gemma 4 E2B bf16 | 0.0059 | 0.0056 | 0.0038 |
| E2B q4 丸め / GPTQ | 0.123 / 0.091 | 0.171 / 0.145 | 0.220 / 0.081 |
| Gemma 4 E4B bf16 | 0.0138 | 0.0040 | 0.0031 |
| E4B q4 丸め / GPTQ | 0.047 / 0.025 | 0.089 / 0.077 | 0.085 / 0.051 |
| PLaMo 3 2B bf16 | 0.0040 | 0.0118 | 0.0022 |
| PLaMo 3 2B q4 丸め / GPTQ | 0.097 / 0.044 | 0.093 / 0.088 | 0.055 / 0.040 |
| PLaMo 3 0.8B bf16 | 0.0046 | 0.0038 | 0.0031 |
| PLaMo 3 0.8B q4 丸め / GPTQ | 0.095 / 0.043 | 0.094 / 0.061 | 0.094 / 0.072 |
| Qwen3.5-0.8B bf16 / q6 | 0.0014 / 0.0079 | 0.0056 / 0.044 | 0.0012 / 0.0077 |
| Qwen3.5-2B bf16 / q6 | 0.0009 / 0.0047 | 0.0017 / 0.019 | 0.0007 / 0.0062 |
| Qwen3.5-4B q4 / q6 | 0.023 / 0.004 | 0.042 / 0.005 | 0.046 / 0.040 |

今のまとめた書き出し、256 トークンの後の 320 行の平均 / 中央値 / 95 % 点: LFM3-1.7B q4 0.0643 / 0.0494 / 0.171 (top-1 86.6 %)、LFM3-VL-2B q4 (文だけ) 0.1188 / 0.0589 / 0.511、Qwen3.5-0.8B q6 0.0084 / 0.0074 / 0.020 (94.1 %)、Gemma 4 E2B q4 0.1539 / 0.0684 / 0.600 (82.2 %)。

- PLaMo 3 の翻訳は q4 で訳文の側の KL 0.007〜0.009。44 文のうち 36 文が bf16 と同じで、残りも正しい言い換え。PLaMo 3 0.8B の GPTQ は 20 文すべて bf16 と同じ
- LFM3 の q4: 誤差の大半は MLP の 4 ビット (0.031)。注意と畳み込みは q8 ならほぼ無損失 (0.0004)
- Qwen3.5 の 0.8B / 2B は q4 で GPL が崩れる (0.8B 0.336、GPTQ でも 0.300。2B 0.163)。原因は MLP。q6 にする。4B は q4 で崩れない

## 移植の覚え書き

- **LFM3** (`iron_lfm3.py`): 畳み込みの層は `in_proj` を B、C、x に分け、B·x を `CausalConv1d` (幅 7、状態は `Checkpoints`) に通し、C を掛けて `out_proj`。全注意の層は q と k をヘッドごとに RMSNorm、RoPE は周波数の後ろ半分を 0 (「zero-tail」)、ヘッドごとの sigmoid のゲートは `gate_proj` の行をヘッドの列数だけ繰り返して要素ごとの積に。スライディング窓の層は窓 256、θ 10000。sandwich norm、SwiGLU、最後の `embedding_norm`、頭は埋め込み
- **PLaMo 3** (`iron_plamo3.py`、照合は `iron_plamo3_check.py`): RMSNorm は `(offset + w)·x` で offset は norm ごと (1、注意の後 1/5、MLP の後 1/5^1.5)。float32 で重みに足して bf16 に丸める。8 層に 1 層 (7、15、23) が全体、ほかは窓 2048。窓の RoPE は θ 10000、全体の層は θ 10⁶ の YaRN (倍率 256、元の文脈 4096、beta 32 と 1。config.json の `rope_parameters` ではなく `rope_scaling_factor` と `initial_context_length` から作る)、`rope_attention_scaling` は 0.1 ln 256 + 1。`lm_head` は tie なし、埋め込みの尺度なし。トークナイザーと PFML3 のチャットの書式は [LLM のモデル](llm-models.ja.md)
- **Ouro** (`iron_ouro.py`): ループ型のデコーダーで、24 層を 1 トークンにつき同じ重みで 4 回回す ([LLM のモデル](llm-models.ja.md))。IRON の `SandwichLayer` に Ouro の norm の名前を付け替える (`input_layernorm_2` → `post_attention_layernorm`、`post_attention_layernorm` → `pre_feedforward_layernorm`、`post_attention_layernorm_2` → `post_feedforward_layernorm`)。`self.layers` は `model.layers` を 4 回並べたもの。`Ouro.layer` が 2 回目以降の回に、その回の `KVCache` を call 経由で渡し (`call.ouro_cache`。バッファは `passes.P.layers.L.cache.keys` で、ランタイムは名前の `.cache.` で KV キャッシュと見なす)、回の最後の層の後に `model.norm` を通す。重みはバンクに 1 度だけ置かれる (IRON の `walk` は同じモジュールを 1 度しか取らない)。チェックポイントの `model.early_exit_gate` は木から外す (IRON は余った重みをエラーにする)。MLP と head は q6 で、gate と up を 1 つの表にする (`QGEMVGlu`、`HWNPU_OURO_FUSE_GLU=0` で外す)。注意は q8。ホストで丸めたときの KL (5 つの評価の文の平均、`q4_gptq.py ouro --method rtn`、transformers 4.54) は q4 0.130、q6 0.013、q8 0.0012。文脈 2048: キャッシュは 96 層 × 1 位置 8 KiB (2048 で 1.6 GB)、バンクは 2.86 GB。書き出しは 43〜48 秒、RSS のピーク 7.7 GB。プロンプトの塊のコマンドは 12 層ごとに切る (`install_cuts`、`HWNPU_OURO_LAYERS_PER_COMMAND`): 4 回とも同じ重みを読むので IRON の切れ目 (コマンドのスクラッチパッドの語の数による) が来ず、塊が 1 つのコマンドで命令列が 233 MB (bf16 で 160 MB) になり、64 MiB のデバイスのヒープに入らなかった (`control code buffer: CREATE_BO: resource temporarily unavailable`。ほかの書き出しで最大のコマンドは 37 MB)。いまは約 29 MB のコマンドが 8 つ。デコードは 1 つのまま (36 MB)。Ryzen AI 9 365 で (2026-11-22): Hugging Face と比べて (`ironlm -forced` で readme.md の 256 + 64 トークン、transformers 4.54 の `forced_logits.py`) q6 は KL 0.026 / 中央値 0.0036 / 95 パーセンタイル 0.108、1 位の一致 98.4 %。bf16 は KL 0.0022、98.4 %。q6 のデコードは 300 で 1 トークン 268 ms (3.6〜3.7 トークン/秒)、1024 で 312 ms、2047 で 357 ms。bf16 は 321 ms。1024 の塊は 0 から 2.50 秒、1024 から 3.11 秒 (409 / 329 トークン/秒。どのコマンドも 2 秒の制限より十分短い)。固定されるメモリ 2.79 GiB (`drm-total-memory`)、プロセス 0.82 GB、読み込み 1.1 秒。hwnpu-chat で動く (貪欲に 3.6 トークン/秒)。
- **Gemma 4**: IRON のアプリケーション。hwnpu はプロンプトの forward を差し替え (`Gemma4Tail`)、PLE と埋め込みをホストに移す (`iron_hostple`)。モデル名は `Gemma4` のまま (`hwnpu_name`。チャットの書式はこれで決まる)
- **密な Qwen3.5**: モデルは [Qwen](llm-qwen.ja.md)。GDN の演算子は Qwen3.6 と共通 ([MoE](moe.ja.md))

## 移植の照合

- 実物の重みで HF の fp32 と: `ironlm -forced` のあと `tools/ref/forced_logits.py MODEL CKPT RUN.bin`。bf16 の書き出しは KL 0.001〜0.014、最上位 96〜99 %
- 実物の形の乱数の重みでは、NPU と IRON の `OnHost` の参照が合わない (コサイン 0.45〜0.83、Gemma 4 と PLaMo 3)。照合は実物の重みで。層の少ない小さな乱数のモデルはホストでは合う
- KL は先頭に BOS を置いて測る (LFM3 は BOS なしで 0.139、ありで 0.035)。Qwen3.5 は `<|endoftext|>` 248044、LFM3 の BOS は 124894
- LFM3 の `forced_logits.py`: チェックポイントの `auto_map` で transformers がリモートコードを読もうとする。config.json から `auto_map` を消した置き場を渡す
- `ironlm -record` はその書き出し自身のロジットを `check/` として書き、別の書き出しをそれに照らす (作り直しの「ビット単位で同じ」はこれで確かめる)
- `ironlm -depth` は文脈まで塊の境ごとに塊とステップを測る。`-bench` は数トークンの呼び出し、`-steps N` はデコードの速さ。新旧を交互に、NPU をほかが使っていないときに
- 演算ごとの時間: `--command-per-step` と `HWNPU_DEV=iron.timing=1` (和は普通の書き出しより約 1 割多い)
- 書き出しを変えたら、`hwnpu-chat` で画像 1 枚と音声 1 本を通して塔を確かめる
- 落としたチェックポイントは `x-linked-etag` の sha256 と照らす: LFM3-Audio と Omni の safetensors に、乱数の重みの照合では見つからないビットの反転があった

## 採らなかったもの

- GPU や 2 台目の機械でプロンプトを速くする (oMLX の Prefill Accelerator): hwnpu は省電力が目標。代わりに計算を減らす (Gemma 4 の tail)
- プロンプトの塊で下書きを確かめる: デコード 3〜9 ステップ分かかり、prompt lookup の 1 回 1.07〜1.3 トークンの採用では取り返せない
- 出力層を 2 段にする (2〜3 ビットの写しで候補を選び、正確に採点): 貪欲な選択では Qwen3.5-0.8B の 34 %、Qwen3.5-2B / Qwen3.6 の 50 %、gpt-oss の 40 % を読むが、上位 64 個を残すと 56〜83 % かそれ以上。Gemma 4 E2B は写しの誤差が大きく使えない。出力層は 1 ステップの 5〜15 % なので、NPU の実装なしに得られるのは数 % と見積もる。
- 注意のモデルのプロンプトの塊 2048 (2 割遅い) や 256 (Gemma 4 の窓の層がリングになる)
- 注意を 4 ビットに (`--attn q4`): E2B の KL 0.229 (q8 は 0.097)。注意は重みの 1 割ほどだが 4 ビットの誤差の大半
- q8 の部分和を 1 つの bf16 に丸める (q4nx のやり方): コサイン 0.9998 (2 つなら 0.999992)
- Qwen3.5-0.8B / 2B の MLP を 4 ビットに: GPL が崩れる (上)。q6 にする
- Gemma 4 の `QGEMVGlu` の GELU 版を既定に: 9.3 → 9.9 tok/s だが KL 0.1539 / 0.0684 / 0.600 → 0.1580 / 0.0736 / 0.625、前とのコサイン 0.9897。プロンプトの塊が GLU の表を MEDIUM の GEMM で読む経路を疑っているが、確かめていない
- `SandwichRows` (プロンプトの RMSNorm、和、RMSNorm を 1 演算子に): 設定のし直し 632 → 538 回だが時間は同じ (RMSNorm と和の設定は約 5 KB)。コードは残していない
- GDN のカーネルを `-Os` / `-Oz` / `-O1` で: プログラムの大きさはほぼ同じ (1045 → 1037 KB)、`-Os` は上限を 32 バイト超えた。IRON の build cache は名前と引数で引くので、フラグだけでは作り直されない
- 多モーダルの書き出しの横に文だけの書き出しを置く: 塔は使うときだけ読む
- `gemma4full` / `gemma4mmfull` (IRON のままの forward) と `--scan-window`: 照合用に残すだけ

## 未解決

- `chunkBarrier` なしで full ELF の 2 つ目のプロンプトの塊が止まる理由 (書き出しで、デコードの形の設計に `dma_free_task` の後にバッファ記述子を使い直すという aiecc の警告が約 500 出る)
- 画像と音声つきの Gemma 4: 同じ書き出しでも走らせるたびに 1 つの問いの答えが変わった。調べていない
- ホストは塊の行ごとに全部の層の PLE を書くが、共有の層が読むのは最後の行だけ (1 行あたり 20 層 × 512 バイトが無駄)
- 短いプロンプトの GEMM と重みの展開 (16 トークンの 6 割): 呼び出しの行に従う GEMM か、量子化の重みを直接読む数行の GEMM が要る。16 行の `QGEMVRows` は今の GEMM と同じくらいの見積もり
- デコード: 今のまとめ方の先は gpt-oss の `GptOssStep` のような 1 配列の形 ([MoE](moe.ja.md))。最初の関門はコアのプログラムが 16 KB に収まるか。GDN を 1 列 1 コアに (設定がほぼ半分、1 トークン約 12 ms の見積もり) は、デコードとプロンプトの版と hwnpu の `conv_pairs` が共有する状態の並びを変える必要がある
- Gemma 4: PLE の gate と projection を q8 の QGEMV に。GELU と積を KL を悪くせずにまとめる
- デコードのまとめで作り直して測る: Gemma 4 E4B、`gemma4mm`、LFM3-Audio、LFM3-Omni
- 確かめる版: q4 の `QGEMVRows` (Gemma 4、PLaMo 3、LFM3)、Gemma 4 の共有の層と窓の注意、リングのキャッシュ、MTP の下書き、注意を数行の MHA 1 回に、状態の写しを減らす
- PLaMo 3 2B のデコードは別のセッションの CPU の負荷の下で測った。静かな機械で測り直す
- `iron_q4.py check --valid` は q8 でも合わない (照合の側の問題、直していない)
