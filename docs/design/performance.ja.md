# 性能

[English](performance.md)

改善ごとに何が得られたか、組み合わせるときに気をつけること、速さの上限とソフトウェアでまだ動かせるところ、まだ確かめていないこと。改善 1 つに 1 行。設計と測った値の全体はリンク先の topic の文書にある。

機械: Apple の数値は M1、M2、M3 (16 GB、macOS 27、Core AI) の Neural Engine で、書いたとおり。AMD の数値は Ryzen AI 9 365 (Strix Point、XDNA 2)。「生成」は 1 トークン、「プロンプト」はプロンプトを読む時間、「起動」は 2 回目以降の起動。GPU は使わない。

## ひと目で

### 言語モデル: Apple

| 変更 | 得られたもの | 代わりに払うもの | 詳しく |
|---|---|---|---|
| 層で分ける (1 パッケージ 1 GiB、1 部分 0.9 GiB) | ANE に収まる。RSS は約 1 部分 | | [apple-coreml.ja.md](apple-coreml.ja.md) |
| KV キャッシュ | 1 トークン 34 → 17〜21 ms (M3、LFM3-1.7B) | | [llm-engine.md](llm-engine.md) |
| 窓の層を輪に | 4K で −26 %。KV −43 % | | [llm-engine.md](llm-engine.md) |
| `SDPAGrouped` (まとめた注意) | 生成 −45 %、プロンプト −25 %。tile の写しなし | | [llm-engine.md](llm-engine.md) |
| 1 つのキャッシュに 2 本のプログラム (生成、プロンプト) | 860 トークンのプロンプト 35 秒 → 1.6 秒 | 重みが 1 回になるのは 1 本の `.aimodel` の関数にしたときだけ | [llm-engine.md](llm-engine.md) |
| 前のプロンプトをメモリで使い回す | 2 往復目から 0.1〜0.5 秒 | | [llm-engine.md](llm-engine.md) |
| ディスクのプロンプトの保存 | 7.9K トークン 14.6 秒 → 0.2 秒 | 1 本約 145 MB | [llm-engine.md](llm-engine.md) |
| 注意で分ける (`--paged`、既定) | 生成 1.3〜1.7 倍。文脈 262144。費用は使った長さに従う | 注意の層ごとに NE を 2 回呼ぶ: 全部が注意のモデル (Qwen3-0.6B) は分けないほうが 2〜4 倍速い (`llm.Unpaged`) | [llm-engine.md](llm-engine.md) |
| 注意の塊を kv ヘッドごとに (Core AI、`paged.Heads`) | M1: Qwen3-0.6B 30.3 → 32 tok/s。128 トークンの呼び出し 88〜91 → 85 ms | kv ヘッド 1 つのモデル (Gemma 4 E2B、PLaMo 3 0.8B) はまとめたまま | [apple-coreai.md](apple-coreai.md) |
| 2 段の注意 (`-sparse`) | 1 回の呼び出し 128K で −52 %、250K で −58 % (LFM3)。プロンプト −24 % / −44 % | 近似 (パスキーは保つ)。メモリ +1/8。64K で読むのが 1 割遅い | [llm-engine.md](llm-engine.md) |
| プロンプトを 1 回 128 トークン、ロジットは最後の行だけ (`LastRow`) | 16K で −20〜25 %。文書を読むのが −25 % | | [llm-engine.md](llm-engine.md) |
| Gemma 4 の KV を共有する層を最後の行だけ | E2B 16K 27.8 → 12.6〜13.2 秒、64K 205 → 94.6 秒 | | [llm-models.md](llm-models.md) |
| キャッシュを書く最後の層までのプロンプトのプログラム (`-kv-only`) | M1 の E2B: 4249 トークン 4.2〜4.6 → 2.4〜2.7 秒。重み 3.20 → 1.98 GiB | | [llm-models.md](llm-models.md) |
| ソフトキャップをホストで (`llm.LogitSoftcap`) | M1 で 1 歩 −2.3 ms (4 %)。出力層が全部 NE に | | [llm-models.md](llm-models.md) |
| `blocks.SiLU` (exp で書いた silu) | lut8 の KL が 2.5 分の 1 (LFM3 0.058 → 0.022)、PLaMo 3 −20〜40 % | 生成 +3 % | [apple-neural-engine.ja.md](apple-neural-engine.ja.md) |
| 重みの形式 lut4 / lut6 / lut8 | 1 回 1.81 / 2.65 / 3.49 ms。lut6 / lut8 の KL は lut4 の 1/6〜1/10 | バイト | [quantization.md](quantization.md) |
| lut の前の回転 (`llm.Rotate`、Qwen3.5 2B lut6、既定) | NE での KL 0.054 → 0.029。MTP ありで 45 → 52 tok/s (M2) | 初回に書き出し直し | [quantization.ja.md](quantization.ja.md) |
| プロンプトの検索の下書き | 課題全体で 1.2〜1.5 倍 (15.9 → 23.7 tok/s) | Q=8 の呼び出しは Q=1 より 13 % 高い | [llm-decoding.md](llm-decoding.md) |
| MTP の下書き、GDN の状態をホストで作り直す (`state-rows`) | Qwen3.5-0.8B (M1) コードで 26.1 → 43.8 tok/s、日本語 25.8 → 30.3 | NE に +0.27 GiB | [llm-decoding.md](llm-decoding.md) |
| 語彙の一部の MTP の出力層 (`HeadRows`、74272 行) | Qwen3.5-4B 17.4〜18.2 → 19.6〜20.1 tok/s。出力層 0.68 → 0.26 GiB | | [llm-decoding.md](llm-decoding.md) |
| 考える予算、繰り返しの検出 | 何分も回り続けるのを止める | | [llm-decoding.md](llm-decoding.md) |
| 塊の形の Gated DeltaNet (WY 形) | Qwen3.5-4B 1 歩 96 → 83〜89 ms。128 トークン 511 → 304〜344 ms。初回のコンパイル 15.8 → 7.0 秒 | | [llm-qwen.md](llm-qwen.md) |
| Mac のバックエンドを Core AI に | どのチャットのモデルも 2 回目の起動 0.7〜5.5 秒。Qwen3.5-4B 20.5〜21.9、E4B 20.3〜20.7 tok/s (M2)。2K のプロンプト 4.0 / 1.8 秒 | Mac ごとの初回のコンパイル | [apple-coreai.md](apple-coreai.md) |
| 重みの同じ部分を 1 本の `.aimodel` の関数に | NE が重みを 1 回だけ持つ (LFM3-1.7B 2.56 → 1.34 GiB)、+13 % | 初回のコンパイルは全部の関数の分 | [apple-coreai.md](apple-coreai.md) |
| 入力の配列を 1 回だけ作る、呼び出し側が直接書く | PLaMo 3 0.8B 25 → 38〜44 tok/s。LFM3-1.7B 3K トークンで 39.8 → 43.0 tok/s (M1) | | [apple-coreai.md](apple-coreai.md) |
| M1 では分けない 1 本を使わない (`chat.wholeSlower`) | M1: LFM3-1.7B 33.6 → 39.9、Qwen3-0.6B 20.2 → 30.3 tok/s | | [apple-coreai.md](apple-coreai.md) |
| 会話の短いうちは短い文脈のプログラム (`ringTiers`) | M1 の PLaMo 3 Translate 2B 12.9 → 19.0 tok/s | 初回のコンパイルが 2 倍 | [llm-models.md](llm-models.md) |
| PLE の表を層の切れ端ごとの int8 に | E2B の表 4.4 → 2.2 GiB。Mac のダウンロードは E2B 4.94 GB (チェックポイントは 10.2)。KL 0.00005 | なし: 速さと最大のメモリは同じ | [llm-models.md](llm-models.md) |

### 言語モデル: AMD

| 変更 | 得られたもの | 代わりに払うもの | 詳しく |
|---|---|---|---|
| 1 つの full ELF | 演算ごとの 0.46 ms の xclbin の切り替えがない | | [amd.md](amd.md) |
| デコードの演算子の切り替えを減らす (`QGEMVGlu`、`SandwichAdd`、`ShortConvStep`、状態の枠を直接、PLE をまとめる、QGEMV を 1 列 1 コアに) | LFM3-1.7B 11.2 → 17 tok/s、Gemma 4 E2B 7.7 → 9.7、E4B 5.5 → 6.4、Qwen3.5-0.8B 13.7 → 18.7、Qwen3.6 8.4 → 8.9 | KL はばらつきの中で同じ (LFM3 0.0642 → 0.0646) | [amd-llm.md](amd-llm.md) |
| デコードの 1 歩を 1 つの常駐の配列に (`GptOssStep`) | gpt-oss-20b 79.2 → 49.4 ms (20.2 tok/s)、読みの下限の 93 % | +1.06 GiB (0.67 は `Q8XDequant` で戻る) | [moe.md](moe.md) |
| プロンプトの塊の短い畳み込みを 1 つの演算子に (`ShortConvChunk`、`GdnConvChunk`) | LFM3-1.7B 16 トークン 443 → 386 ms。Qwen3.5-0.8B 343 → 259 ms | | [amd-llm.md](amd-llm.md) |
| プロンプトの MHA を 8 パイプラインで (`--mha-pipelines`) | 塊が 2〜5 倍速い。電力量 −66〜−83 % | 書き出し直し | [amd-llm.md](amd-llm.md) |
| プロンプトの塊 1024 (注意)、512 (GDN) | 2048 より −20 % | | [amd-llm.md](amd-llm.md) |
| 4 ビットの重み (`--q4`、注意は q8) | 生成 1.3〜1.6 倍。固定 1/2〜1/5 | KL はモデルで 0.009〜0.097 | [quantization.md](quantization.md) |
| 小さな Qwen3.5 の MLP を q6 に | 0.8B の GPL の KL 0.34 → 0.044。0.8B はホストの埋め込みと合わせて 1858 → 1069 MiB | | [llm-qwen.md](llm-qwen.md) |
| チェックポイントの MXFP4 のまま (mx4) | 1 重み 4.3 ビット。q4nx に丸め直すと KL 2〜3 倍だった | | [moe.md](moe.md) |
| PLE と埋め込みをホストに | E2B の固定 9.08 → 4.60 GB | | [amd-llm.md](amd-llm.md) |
| Gemma 4 の KV を共有する層をプロンプトの最後の行だけ | E2B のプロンプト 2.3 倍、E4B 1.7 倍 | | [amd-llm.md](amd-llm.md) |
| 窓の層は窓だけ読む | プロンプト −12 % | +12 MB。`chunkBarrier` が要る | [amd-llm.md](amd-llm.md) |
| 塔は使うときに読む | E2B の文だけのとき 2818 → 1826 MiB | 最初の画像 / 音声 +0.6〜0.7 秒 | [amd-llm.md](amd-llm.md) |
| `-idle` | 待つ間の固定 0 | 次の要求 0.8〜2 秒 | [amd.md](amd.md) |
| 下書きを 1 回で確かめる (`--verify 3`) | 3 行の確認がデコード 1.27 歩分。Qwen3.5-0.8B はプロンプトの検索で 1.15〜1.3 倍 | 制御コード +30 MB。q8 / q6 だけ | [llm-decoding.md](llm-decoding.md) |
| 専門家を置き場に (`--expert-places`) | gpt-oss-20b P=16: 固定 6.90 GiB、17.2 tok/s (全部: 12.22 GiB、20.2)。Qwen3.6 置き場 32 で 7.5 GiB、5.8〜6.0 tok/s (全部: 8.8〜8.9) | 置いていない専門家が選ばれたら写す | [moe.md](moe.md) |
| ホストのロジットと上位 k をベクトルのループで | gpt-oss の 201088 のロジット 1 トークン 1.33 → 0.23 ms | | [amd-llm.md](amd-llm.md) |

### Mac の MoE

| 変更 | 得られたもの | 詳しく |
|---|---|---|
| Core AI で経路つきの専門家 (`llm.Routed`) | LFM2-8B-A1B が動く。プロンプト 138 tok/s (1 行ずつなら 24) | [moe.md](moe.md) |
| 専門家を CPU で MXFP4 (`llm.HostRouted`)、丸ごと読む、プロンプトを部分の順に (wave) | M3 の Qwen3.6-35B-A3B: 11.0 tok/s、2602 トークンのプロンプト 21.5 秒、NE と合わせて 2.69 GB | [moe.md](moe.md) |
| gpt-oss-20b、専門家は CPU、予算まで固定 | M3 17.3 tok/s、1 トークン 0.52 J。M2 13.9 tok/s、0.50 J | [moe.md](moe.md) |

### 画像生成

| 変更 | 得られたもの | 代わりに払うもの | 詳しく |
|---|---|---|---|
| CFG の 2 パスを重みの組ごとに | Anima 512² 1 ステップ 3.0 → 2.7 秒 | | [image-models.md](image-models.md) |
| DPM-Solver++ 2M (`--sampler dpm2m`) | 25 ステップが Euler 30 より Euler 100 に近い | | [image-models.md](image-models.md) |
| CFG は雑音の多いステップだけ (`--cfg-sigmas 0.6,1`) | AMD の Anima-Aesthetic 1024² 4 分 40 秒 → 3 分 19 秒、4.8 → 3.8 kJ | | [image-models.md](image-models.md) |
| 変調は sigma ごとに 1 回、キーと値はプロンプトごとに 1 回 | ステップの間に持たない | | [image-models.md](image-models.md) |
| Core AI: 注意のスケールを q の norm に、絶対値は `abs`、ヘッドごと、キーを 2 のべき乗の塊に | Anima-Turbo 1024² 1 ステップ 7.1 → 5.6 秒 (M1)。Chroma1-Flash 1024² 1 枚 5 分 36 秒 (M1) | 書き出し直し | [image-apple.md](image-apple.md) |
| Core AI: 次のプログラムを先に読む、DiT の組を入れ替える | Chroma の DiT 1 回 7.2 → 5.4〜5.8 秒。NE が持つもの 8.3 → 2.2〜2.6 GiB | | [image-apple.md](image-apple.md) |
| Core AI の `--batch` | Chroma の DiT 1 回 1 枚 5.4 → 5.05 秒 | 1 枚ごとのホストのメモリ | [image-apple.md](image-apple.md) |
| 全部の大きさを 1 つの書き出しの関数に | 全部の大きさで 1 回のダウンロード (Anima の大きさごとの書き出しは 1 つ 2.9 GB だった) | 初回のコンパイルは全部の大きさの分 | [image-apple.md](image-apple.md) |
| チップごとの prebuilt (`hwnpu-dev models package MODEL/coreai-ARCH`) | M1 の FLUX.2 klein の初回 40 分 → 7 分 48 秒 | チャットのモデルと VAE には作らない | [apple-coreai.md](apple-coreai.md) |
| Core AI の FLUX.2 klein 4B | M1: 512² 11 秒、1024² 41 秒 / 枚 | | [image-apple.md](image-apple.md) |
| VAE の段の間にホストのメモリを返す | FLUX.2 klein 1024² の最大 5.8 → 4.1 GB | 復号 +0.4 秒 | [image-apple.md](image-apple.md) |
| AMD: DiT の重みをホストで bfp16 に | 1 ステップ: Anima-Turbo 1024² 4.1 → 3.6〜3.8 秒、Chroma1-Flash 29.9 → 22.8〜25.4 秒、Chroma1-HD 27.5 → 20.2 秒 | 1 重み 9 ビット。書き出しが 3〜4 倍長い | [image-amd.md](image-amd.md) |
| AMD: GELU の tanh の引数を fp32 で (`GELUEven`) | Anima 1024² 1 ステップ 4.6 → 4.2 秒 | | [image-amd.md](image-amd.md) |
| AMD: 交差注意をプロンプトの行だけに (`zero_keys`) | Anima-Turbo 1024² 1 ステップ 3.7 → 3.2 秒 | 絵が細部で変わる | [image-amd.md](image-amd.md) |
| AMD: 画像の注意を 64 キーのブロックで (`ImageMHA`) | 146 → 59 ms (4096² × 16 × 128) | | [image-amd.md](image-amd.md) |
| AMD: VAE を格子の上で (`GridGEMM`) | Anima の VAE 9.3 → 5.2 秒 | | [image-amd.md](image-amd.md) |
| AMD: DiT の組を 1 つの bank に (`SwapVariants`) | Anima の固定 3445 → 1236 MiB | +2 % | [image-amd.md](image-amd.md) |
| AMD: DiT を q8 に | 固定 5020 → 3445 MiB。潜在のコサイン 0.966 → 0.961 | +2 % | [image-amd.md](image-amd.md) |
| AMD: コアの丸めを最近接の偶数に | Anima の q8 のステップ 0 の誤差 0.088 → 0.031 | | [amd.md](amd.md) |
| AMD: 全部のアスペクト比を 1 つの書き出しに (`share_banks`) | FLUX.2 klein の 2 つの大きさ 22.68 → 11.77 GB。8 つで 16.23 GB | | [image-amd.md](image-amd.md) |
| AMD: FLUX.3 klein | 1024² 1 分 43 秒、512² 32 秒 | | [image-amd.md](image-amd.md) |

### 画像を読むモデルと音声

| 変更 | 得られたもの | 代わりに払うもの | 詳しく |
|---|---|---|---|
| ONNX のプログラムの sigmoid / SiLU を exp で (`ExpSigmoid`) | RTMDet-s の箱の誤差 98 → 5.5 px | 1 枚 6.6 → 9.3 ms | [vision-models.md](vision-models.md) |
| PP-OCR の認識器を NPU で | 98 行のページ 1528 → 488 ms | 幅ごとにプログラム | [vision-models.md](vision-models.md) |
| ホストの拡大縮小をクロージャなしで全部のコアに | 1 枚 37〜55 → 5〜7 ms | | [vision-models.md](vision-models.md) |
| NE の SigLIP 2 | base 1 枚 7.3〜7.9 ms (M3。AMD 84 ms) | | [vision-models.md](vision-models.md) |
| AMD: Whisper large-v3-turbo | 復号 78 tok/s。30 秒の窓のエンコーダ 0.84 秒 | | [audio-models.md](audio-models.md) |
| AMD: LFM3-Audio の読み上げ、MOSS-SoundEffect、Magenta RT2 の復号器 | 実時間の 0.43 倍、1 ステップ 2.8 秒、実時間の 0.3 倍 | | [audio-models.md](audio-models.md) |

### どこでも

| 変更 | 得られたもの | 詳しく |
|---|---|---|
| ディスクで同じファイルを共有 (`cache.Clone`) | M3 約 50 → 20 GiB。AMD の書き出し 2 つ 15.2 → 8.2 GiB (ext4 ではなし) | [distribution.md](distribution.md) |
| 取得を 4 本並びに | Wi-Fi と NAS で 5 → 9 MB/s | [distribution.md](distribution.md) |
| 会話の印は使っている輪の行だけ、変換の後にメモリを返す | 578 → 183 MB。2.4 GB → 154 MB | [memory-and-power.md](memory-and-power.md) |

## 組み合わせるときの注意

- **lut6 と段**: `--tier auto` は 4 ビットの表のときだけ短い文脈の段を作る。8 GB の機械では作らない (models.yaml の `tier`)
- **2 段の注意**は近似: 注意が 1 つのキーに集まると誤差が大きくなる。250K までパスキーは通る。64K あたりではプロンプトを読むのが 1 割遅い。`-sparse=false` で切る
- **長いプロンプト**: 256K を一度に読むと約 29 分 (LFM3)。プロンプトの保存が前提
- **ばらつき**: M3 のスワップ、AMD の電源の状態で数値が最大 1.4 倍動く。続けて測った組だけを比べる。AMD ではほかに `/dev/accel` を使うプロセスがないか、Mac では `top` (`mediaanalysisd`) を見る
- **AMD の 4 ビット**: 失うものはモデルで違う (LFM3 の KL 0.036、E2B 0.097)。bf16 なら `--q4` なしで書き出す
- **AMD の Anima の DiT の 8 ビット**: 軌道が分かれやすい種では構図が変わることがある (4 プロンプトに 1 つ)
- **カーネルの直しは書き出しの中にある**: AMD の書き出しは、作ったときのカーネル (丸め、まとめた演算子) を書き出し直すまで持つ。新しい書き出し器の書き出しを古い hwnpu が拒むことがある (gpt-oss の常駐の 1 歩)
- **`-idle`**: 次の要求が 0.8〜2 秒払う。**使うときに読む塔**: 最初の画像か音声が 0.5〜0.7 秒待ち、その間は古い bank と新しい bank の両方が固定される
- **AMD の barrier**: 塊の barrier は 1 塊 0.3〜0.5 ms、1 歩の barrier は 1 歩 0.3 ms (gpt-oss の常駐の 1 歩では配列を載せ直す 2.4 ms になるので飛ばす)
- **AMD の時間の測り方**: `HWNPU_IRON_TIMING=1` は各コマンドの往復を足す。`ironlm -forced` の 1 歩の時間を読む。層で違う `array=True` の引数はコマンドごとに配列を載せ直し、それでも確認は通る
- **ディスクの共有はファイルシステムによる**: ext4 は何も共有しない。Btrfs か XFS に (Windows は Dev Drive)
- **Core AI の初回**はどの部分も 1 回コンパイルし、関数の多い `.aimodel` は全部の関数をコンパイルする (Chroma1-Flash の 512² と 1024² で 1 時間 14 分)。キャッシュは OS のビルドとプロセスの名前ごと: OS の更新の後、`aicheck` や `go test` からはコンパイルし直す。ディスクが減ると macOS が消す (空き 11〜22 GB で見た)。変換にも空きが要る
- **書き出しの名前に配置の版**があると、1 回変換とコンパイルをし直す。絵は前の版とビット単位では同じでない (和の順が違う)。古い書き出しは消すまで残る
- **NE での数値の変更の判断**: 2 つの NPU の形が互いに 0.97 違っても、どちらも fp32 の参照に同じだけ近いことがある。それぞれを参照と比べ、絵を見る

## 速さの上限

### 上限は NPU が DRAM を読む速さ

`hwnpu-dev probe` ([PROBE.md](../../PROBE.md))、16 層 (0.97 B):

| | M1 | M2 | M3 | AMD (Ryzen AI 9 365) |
|---|---|---|---|---|
| NPU の公称 | 11 TOPS | 15.8 TOPS | 18 TOPS | 50 TOPS |
| DRAM の帯域 | 68 GB/s | 100 GB/s | 100 GB/s | 90〜120 GB/s |
| int8 の重みを読む速さ | 62 GB/s | 62 GB/s | 23 GB/s (M3 だけ) | — |
| 4 ビットの重みを読む速さ | 44 GB/s (lut4) | 63 GB/s | 64 GB/s | 53〜55 GB/s (q4、q8、bf16 を続けて) |
| 行列積の測った最良 | 約 5 TFLOPS (Core AI) | — | 13.7 TFLOPS (K=64、Core ML) | 15.2 TFLOPS (bfp16 の重み、4096 行) |
| 下書きの確認: 1 回 4 トークンの費用 | — | — | 1.0 歩 (64: 1.10) | 1.28 歩 (8: 2.83) |

- **ANE は測ったどの Apple のチップでも重みを約 62〜64 GB/s で読む**: 上限は DRAM でなく ANE 自身のメモリへの道。計算でなくバイトを動かすのが上限 (M2 で int8 と lut4 が同じに読む)。計算で決まるのは M1 の lut4 だけ (表の展開)
- **NE そのものの費用** (M2、`aicheck limits`): 1 回の呼び出しの下限 160〜175 µs。要素ごとの演算 16〜20 GB/s。4096 × 4096 を 1 行: fp16 742 µs、int8 466、lut8 449、lut6 387、lut4 351、int4 329 ([apple-coreai.md](apple-coreai.md))
- **AMD の DMA は 8 チャネルで 55〜57 GB/s が上限**。4 ビットの GEMV は 16 コアで届く。ただしチャットのモデルの書き出しの 1 歩はその 3 分の 1 ほど: 演算子の切り替えのたびに配列の設定を書き直す (85〜100 µs、`elfconfigs`)。gpt-oss-20b の常駐の 1 歩は下限の 93 % で回る (1 トークン 2.53 GB: 55 GB/s で 46 ms、実際 49.4)
- **計算は逆**: bfp16 の重みの AMD の GEMM は 15.2 TFLOPS で、M3 の NE より上、M1 の約 3 倍。計算で決まる仕事 (画像の DiT、プロンプト) は AMD が先 (FLUX.2 klein 1024² 約 29 秒、M1 は 41)
- TOPS が効くのはプロンプトと長い文脈の注意だけ。生成の 1 トークンは [model-fit.md](../model-fit.md) の 2 節: 重みのバイト ÷ 読む帯域 + 注意 + 約 3 ms

### 新しいチップで何が変わるか (見積もり、測っていない)

| | NPU の読みの見込み | M3 に対する生成 | hwnpu で動くか |
|---|---|---|---|
| Apple M6 (Neural Engine 2 つ、170 GB/s) | 1 つが 64 GB/s のままなら同じ。両方を同時に使えれば 2 倍まで | 1.1〜2 倍 | 動く |
| Snapdragon X2 Elite (80 TOPS、152 GB/s) | 75〜100 GB/s | 1.3〜1.6 倍 | バックエンドなし |
| Snapdragon X2 Elite Extreme (228 GB/s) | 115〜150 GB/s | 1.8〜2.2 倍 | バックエンドなし |

新しい機械で確かめる: `hwnpu-dev probe` の 16 層の lut4 の "weights GB/s"。約 64 なら生成は M3 と同じ。

### ソフトウェアでまだ動かせるところ

トークン/秒 ≈ 読む帯域 ÷ 1 トークンに読むバイト × 1 回の読みで進むトークン − 読んでいない時間。

| 項 | 手法 | 状態 |
|---|---|---|
| 1 トークンのバイト | ビットを減らす (lut3 と lut4 を混ぜる、表を合わせる、AWQ 風の尺度) | GPTQ と表の合わせは入った。AWQ は書いていない。精度との引き換え |
| 〃 | MoE (使う専門家だけ読む) | AMD と Mac に入った |
| 1 回の読みのトークン | プロンプトの検索、MTP の下書き | Apple に入った。AMD はプロンプトの検索を `--verify` で (q8 / q6)。AMD の MTP はまだ |
| 〃 | 1 回の読みで複数の要求 (batch) | していない |
| 〃 | 小さなモデルの下書き (2B / 8B に PLaMo 3 0.8B。訳ではプロンプトの検索が 113 に 1 回しか当たらない) | 測っていない: 一致 80 % で約 1.9 倍 |
| 読んでいない時間 | AMD: デコードの演算子の切り替え | どのモデルも減らした。1 つの常駐の配列は gpt-oss だけ (ほかのモデルは q4 で約 2.7 倍の余地) |
| 〃 | AMD のプロンプト: 重みを DiT のように bfp16 で持つ (呼び出しごとに展開しない) | していない: probe で 256 行の q4 / q8 は 86〜88 ms、bfp16 は 28 |
| 〃 | Apple: 呼び出しの固定の費用、部分の切り替え、ホストの仕事 | M1 の LFM3-1.7B の 1 歩 31.2 ms = 重み 26.8 (48 GB/s) + 注意 4.2 + ホスト 0.5: 残りは 1 割 |
| 長い文脈の注意 | `SDPAGrouped`、分ける形、2 段の注意 | 入った |

AMD の手法で Apple と Intel に持っていけそうなもの (試していない): gated MLP の gate と up を 1 つの linear に。Intel で尺度のバイトを減らす。注意の sink を 1 列足して。そこでは要らないもの: バイアスと残差を linear で、常駐の 1 歩 (Core ML / OpenVINO はまとめてプログラムを載せたままにする。ただし大きな Core AI のプログラムが別のものに続くと 50〜70 ms 払う)。

## 採らなかったもの

測って外した、モデルをまたぐ速さの案 (モデルを入れるときのものは [new-model-checklist.md](new-model-checklist.md)):

- 下書きの幅 Q=4: Q=8 より 3 % 安いだけで、当たりが 1 割減る
- 下書きを残さないとき用の 1 回 1 トークンの関数: 8 トークンの呼び出しから 4 % (M1)
- Gemma 4 の短い輪: 40.1 ms に対し 40.4 ms
- M1 だけ lut8 でなく int8: PLaMo 3 2B の 128 トークンの部分が NE で取り消された (status 0xf)。チップごとの書き出しも要る
- Anima の DiT をホストの段をはさんだコマンドに分ける (AMD): 最初のコマンドが時間切れ。4 ブロックの 1 つのグラフに
- 1 トークンのコマンドを 1 本の鎖に (`ERT_CMD_CHAIN`): 19.3〜19.6 tok/s に対し 19.6〜19.7。境の 0.1 ms はファームウェアがコマンドを始める時間
- 1 トークン 1 コマンド (gpt-oss): mlir-aie は scratchpad を 1 列に 1 回しか読まず、経路が壊れた (コサイン 0.877)
- デコードの注意のスコアに bfp16 の行列命令: 速くならず、出力が 3.6 % ずれる
- プロンプトの sandwich norm を 1 つの演算子に (`SandwichRows`): 速くならない (設定が小さい)。GDN のカーネルを `-Os` に: 小さくならない
- Gemma 4 の MLP の gate と up を GELU つきで 1 つの演算子に: 9.3 → 9.9 tok/s だが KL が上がり (0.154 → 0.158)、ロジットがずれる。既定はオフ (`HWNPU_GEMMA4_FUSE_GLU=1`)
- Mac で専門家の worker を回し続ける (`HWNPU_MXFP4_SPIN`)、スレッドを USER_INTERACTIVE に、層の間 worker を起こしておく: NE の呼び出しが遅くなるか得がない。コードは消した
- 16 GB で Qwen3.6 の専門家を予算まで固定: 固定なしの 10.6 に対し 8.6〜8.8 tok/s。メモリの 3/4 を超える専門家は固定しない
- 画像の DiT の Core AI: 重みのビットを増やす (NE では int8 が最速)、q / k / v を 1 つの linear に、7 ブロックの組、自己注意で softmax の和を P·V の後に割る (fp16 を超える)、値の int8 の重みに 1/256 (絵が変わった)、norm の行の最大の割り算を定数に
- Core AI: VAE の段の fp16 の出力をそのまま次へ、小さな部分の先読み: ばらつきの中
- AMD の画像の DiT: IRON の tanh GELU (プロンプトの服が変わった)、x sigmoid(2u) の GELU (速くならない)、ゲートつきの残差を 1 つの演算子に (入力の DMA チャネル 2 本に入らない)

## まだ確かめていないこと

### Apple

- 2 段の注意: Qwen3.5 のパスキー。`Refresh` を既定で 4 か 16 に (1 回 14〜32 % 速いが NE で KL を測っていない)。始める位置 (約 57K)。`choose` の写し (128K で 1 回の 12 %)
- 128K を超えてからの遅れ (1K あたり 1.14 → 1.43 ms)。256K のプロンプトの後の最初の数回 (1 回 1.29 秒)
- NE のメモリのうち重み以外 (KV、途中の値) を分けていない
- M4 以降: 上限 (`hwnpu-dev probe limit`、`probe split`) と、既定の部分を大きくして得か
- 塔と多モーダルのモデルの `blocks.SiLU` (VL の説明、TTS) を前と比べていない
- Apple での Gemma 4 / Qwen3.5 の `hwnpu-chat agent`、Gemma 4 とプロンプトの保存
- d1-Omni-2B のチームの答えが README と逆 (HF は 16 問をまとめて読む)。LFM3-Audio の TTS を llama.cpp の音と。PLaMo-Embedding の長い文。BERT の本物の重みを ANE で
- GPTQ の codes で答えが良くなるか (NE で KL は下がる。12 問では決まらない)。NE で lut4 + wkmeans
- Qwen3.5-2B の lut6 を HF の bf16 と (今は lut8 とだけ)。Whisper の形式を実際の録音で
- Core AI: OS の版をまたぐ `.aimodel`。写しがなくなった後の M2 / M3 での分けない 1 本。Qwen3.5 の GDN の状態を次の入力にそのまま渡す (1 歩 18〜24 MiB)。注意の塊を 1.5 倍刻みに (3〜6 % の見込み)。キーと値を int8 の入力に
- Anima: diffusers の fp32 との残りの誤差 (文の符号器の int8 か)、1024² より大きな絵、M3 / M4 での v5
- MoE: M1 (尺度の `gather_nd` がない) と M4。持つ専門家が増えると上がる束ねの費用 (メモリの圧迫か Core AI の上限か)。LFM2-8B-A1B の agent の課題。gpt-oss のプロンプトの専門家 (SMMLA が 1 コア 112 GMAC/s、単独なら約 190)、その GPTQ の codes、M3 が忙しいとき、M4 で固定しないとき

### Intel

**実機の NPU ではまだ何も動かしていない**。OpenVINO の CPU で精度だけ ([intel.md](intel.md))。残り: ドライバのコンパイラがモデルと分けた形を受けるか、引数の名前 (`argNames`)、int8 と int4-g128 の速さ、大きさの上限 (`-target intel` の 2 GiB は仮)、コマンドリストを記録し直す費用、段つきのメモリ、hwnpu-chat での本物のチェックポイント、Linux での IR の共有、実機での PLaMo 3 と Anima。

### AMD

- NPU1 (Phoenix、Hawk Point): 1 歩 1 xclbin は 4 列の模擬の NPU でだけ確かめた
- Windows: 実機の xrtshim、MSVC でのビルド。ReFS / Dev Drive の複製はコンパイルしただけ
- bcachefs、ZFS、NFS での共有
- NPU だけの電力: Linux 7.0 にはセンサーがない (パッケージの値だけ。gpt-oss 1 トークン 0.92 J)
- 長い文脈: 書き出しの文脈は固定 (4096 / 12288)
- gpt-oss: 1 歩の barrier なしで何時間も、常駐の配列での `-idle`、注意の射影を GPTQ で q8 未満に
- Qwen3.5-2B (q6) と 4B (q4 の数行の GEMV が要る) の下書き。`--verify` での Gemma 4 の共有の層と輪
- Qwen3.6 のプロンプトの注意はパイプライン 1 のまま (8 は測っていない)
- 短いプロンプト: `ProjectionGEMM` と展開が 16 トークンの塊の 6 割で、行の数に従わない (行に従う GEMM が要る)
- GDN を 1 列 1 コアに (Qwen3.6 で 1 トークン約 12 ms の見込み)
- q6 の GEMV の速い並び。aiecc の BD の使い回しの警告。64 トークンからの `iron_moep.py check`。`iron_q4.py check --valid`

### 原因のわからないもの

- 16384 の上限で CPU に落ちる形と残る形 (Core ML のコンパイラの中)
- 128K の英語の文書で、2 段の注意のほうが普通の注意より損失が低い (普通の注意は遠いトークンに気を取られる?)
- AMD: `chunkBarrier` なしで full ELF の 2 つ目の塊が止まる。ホストの段で分けた Anima の DiT が時間切れ。同じ演算が 1 回目と 2 回目で 1.4 倍違う (電源の状態?)。`--command-per-step` の Qwen3.6 のロジットが誤る (時間を測るだけに使う)
