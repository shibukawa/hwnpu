# LLM のエンジン

[English](llm-engine.md)

言語モデルを NPU で動かすうちの、モデルによらない部分: KV キャッシュとその並び、注意で分ける形、注意のプログラム、プロンプトの読み、メモリとディスクでのプロンプトの再利用、長い文脈。パッケージ: `internal/llm/decode` (`Session`)、`internal/llm/paged` (重みのない注意のプログラム)、`internal/llm/chain` (部分の鎖)、`internal/llm/runner` (CPU のインタプリタの runner)、`internal/chat` (`Engine`、`Prepare`、`store`)。AMD の NPU は IRON の別の経路 ([AMD の LLM](amd-llm.md))。KV キャッシュの手法の調査と Neural Engine への当てはめは [../kv-cache.md](../kv-cache.md) (そのまま残す)。

## キャッシュはホストが持つ

- キャッシュはホストのメモリで、プログラムの入出力として渡す (MLState は使わない)。`Bind` したバッファはコピーなしで Core ML に渡るので、キャッシュは入力のバッファそのもので、ホストが各ステップの後で新しい行を書く。バッファはただの fp16 のバイト列なので、同じ `decode` のコードが Core ML、Core AI、Intel、テストの CPU インタプリタ (`milcpu`) で動く。
- 役割は名前で決まる (`llm.Cached`):

| 入力 | 形 | 意味 |
|---|---|---|
| `kv_NAME` / 出力 `kv_NAME_new` | (1, kv_heads, cache, head_dim) / (1, kv_heads, Q, head_dim) | RoPE を掛けた後の鍵と値。ホストがステップの位置に新しい行を書く |
| `win_NAME` / `win_NAME_new` | (1, W, 幅) / (1, W+Q, 幅) | 短い畳み込みの直前の行 (LFM3 は B·x の 6 行)。ホストは最後の本物のトークンで終わる W 行を残す |
| `kv_NAME_cJ` | 塊ごと | 分けないプログラムの 16320 を超えるキャッシュの塊 (下) |
| ほかの入力 | | モデルが位置から作る (`StepInputs`): 注意の型ごとのマスク (1, 1, Q, cache+Q)、RoPE の表 (1, Q, 1, head_dim)。`gather` は使わない (CPU に落ちる) |

- 1 ステップは最大 Q トークン。足りない分はゼロで、その行は書かない。部分ごとに自分の層のキャッシュを持つので、部分への分割と組み合わさる。
- 鍵は RoPE の後で保存するので、リングの中の並びは結果に効かない (softmax は鍵の順に依らない)。

### スライディング注意の層はリング

- スライディング層のキャッシュは窓の長さ (`Config.cacheLen`: 窓がキャッシュより短いときだけ窓)。ホストは `pos mod W` の行に書く。リングと全注意のキャッシュは長さだけで見分ける。
- リングのマスク: スロット j が最後に書かれた位置は `pos-1-((pos-1-j) mod W)`。それが 0 以上で窓の内側なら見える。
- LFM3-1.7B は注意層 11 のうち 5 層がスライディング (窓 256)。リングで、分けないプログラムの容量 1K あたりの費用が 5.4 から 2.8 ms になった (M3)。

### テンソルの上限: 1 次元 16384

- Neural Engine は 1 次元 16384 まで (`llm.NeuralEngineMaxDim`)。鍵の軸が 16385 以上の注意は CPU に置かれる (M3: 2 層で 16.6 ms → 117 ms。キャッシュ 16384 で 1 トークン 375 ms、読み込み 140 秒)。
- そのため分けないプログラムはキャッシュ + Q を 16384 以下にする: 注意で分けないモデルの `chat.DefaultContext` は 16384 − 64 = 16320 で、プロンプトのプログラムは 64 トークン。`hwnpu-dev convert` は `-cache + -seq` が上限を超えると警告する。
- 16320 を超える分けないプログラムのキャッシュは、16384 ずつの入力の塊に分ける (`llm.CacheChunks`、`CacheChunkOver` = 16320、`CacheChunkLen` = 16384、最後は短い)。`blocks.ChunkedSDPA` が塊とステップ自身のトークン (`mask_TYPE_self`) を、プログラムの中で要素ごとの演算だけで合わせる: 指数は全体の最大値に ln 4n を足した値から取り (和を fp16 に収める)、指数の和と重み付きの値は行列積 (`e @ 1`、`e @ V`)、割るのは最後に 1 回。規則はキャッシュの長さだけで決まるので、1 トークンと 64 トークンのプログラムがバッファを共有できる。リングは分けない。この経路は注意で分ける形より 1 トークンが遅く、分けられるモデルはすべて分けるのが既定。

## 2 つのプログラムで 1 つのキャッシュ

- チャットのモデルは、生成するプログラム (Q = `--draft`、既定 8: 1 トークンと下書きを 1 回で確かめる。[デコード](llm-decoding.md)) と、プロンプトを読むプログラム (Q = 64、注意で分けたモデルは 128) を持つ。どちらも同じキャッシュで変換する。`chain.Share` が 2 つ目の鎖のキャッシュの入力を 1 つ目の鎖のバッファに結び、`decode.Session.Add` が同じメモリかを確かめる。`Step` はトークンの数を受け取れる最も小さいプログラムを選ぶ。
- 重みは 2 回読み込まれる。`--prefill` (既定 `auto`、`on`、`off`) が決める:
  - `auto` は、2 本が Neural Engine のコンパイルのキャッシュ (約 4 GiB。予算 `compileCacheBytes` = 3.5 GiB) に収まらないか、M1 で全速で抱えられる量 (`neuralEngineHold` = 2.8 GiB、一緒に読み込むプログラムも数える) を超えるなら 1 本にする (生成用がプロンプトも読む)。Gemma 4 E4B (2 × 3.26 GiB) は 1 本: 起動 3.7〜5.5 秒 (2 本では 77〜87 秒)、生成 13.4〜16.4 tokens/s (10〜13) だが、16K のプロンプトは 4.4 倍遅い (183 秒対 42 秒、M3)。
  - Intel と `--draft` 1 以下は常に 2 本。
- models.yaml でモデルと機械のメモリごとに `prefill`、`context`、`paged`、`tier` を決められる ([配布](distribution.md))。

## 注意で分ける (paged)

分けられるモデル (`llm.Paged`: LFM3 とその VL、Audio、Omni、Gemma 4、Qwen3.5、PLaMo 3。Qwen3 は Core AI だけ) はすべて分けるのが既定で、Core ML、Core AI、Intel で動く。

- 変換 (`hwnpu-dev convert -paged N`、`llm.Options.Paged`、`llm.PlanPaged`): 全注意の層 N の前で部分を終え、`attn_q_N` (RoPE や norm の後)、そのステップの `kv_k_N_new` と `kv_v_N_new` (LFM3 はゲート `attn_gate_N` も) を出す。次の部分は `attn_o_N` を受けて層 N の残りを計算する。リングと畳み込みの窓は部分の中のまま。部分の間の出力は名前で結ぶ (`chain`)。重みの予算を超える区間はさらに分ける。manifest に `paged` と `attention`。
- 注意のプログラム (`internal/llm/paged`、`out.attn.<ext>`) は重みがない: 塊 1 つへの注意で、塊の中で正規化した出力 o、行ごとの最大値 m、指数の和 l を返す。和は 1 との行列積 (Core ML の Q=64 では `reduce_sum` の配置がコンパイルごとに ANE と CPU で入れ替わる)。
- o は `matmul(e, v) / l` ではなく `matmul(e · (S / l), v) · (1/S)`、S = 1024 (`paged.Version` 1、`paged.OutScale`)。Neural Engine は行列積の中で約 2^-16 より小さい積を捨て、値の小さい層 (LFM3-1.7B の層 18、|v| ≤ 0.7) では重みの小さい多くの鍵が分子から消えた (塊の o が最大 79 % ずれ、変えた後は最大 0.4 %。3.3 万トークンの文書の合言葉 3 つ: 0 → 3)。|v| < 32 なら途中の和は NE の 2^15 未満。費用: 8192 の鍵への 128 行の呼び出しが 5.5 → 5.95 ms (M3)、8 行は変わらない。`Manifest.AttnVersion` に記録し、古い注意のプログラムは `RefreshAttention` が書き直す (重みがないので数秒)。`TestSmallValuesOnNeuralEngine` が実機で確かめる (前は 4.6 %、今は 0.3 %)。
- `decode.Session` が部分を順に動かす。部分の間で、そのステップの鍵と値を自分のキャッシュに書き (注意の前に書くので、ステップのトークンが自分を見る)、使っている塊ごとに注意を呼び、float32 で合わせ (`paged.Merge`、Flash-Decoding の合流)、`attn_o_N` に書く。マスクが要るのはステップのトークンが入る塊 (多くて 2 つ) だけで、それより前の塊はゼロのマスク。
- 塊は位置が届いたときに確保する (`Attention.NewBuf`、Core ML ではデバイスのバッファ)。`Limit` で文脈を下げ、`Close` で塊を返す。
- 塊の付け替えに費用はなく、入力の複製もなく (RSS ≈ KV のバイト数 + 0.1 GiB)、重みの部分と注意のプログラムの境目で載せ直しも起きない (`npuprobe seam`、M3)。
- 費用は容量ではなく使った長さに比例する。文脈の既定はモデルの最長 (`max_position_embeddings`: LFM3-1.7B と Qwen3.5 は 262144、Gemma 4 は 131072、LFM3-VL/Audio/Omni は 128000)。キャッシュが文脈いっぱいで物理メモリの 4 分の 1 を超えるなら、エンジンが塊単位で文脈を縮める (`fitMemory`、`memoryShare` = 4)。ready の行に 1 トークンあたりの増え方を出す。
- 鍵と値の共有 (`llm.Paged.PagedKV`): 別の層の鍵を読む Gemma 4 の層 (E2B は 19〜34 が 14 を読む) は `attn_q_N` だけを出し、セッションは持ち主の層のキャッシュに注意する。ホストのキャッシュは 1 トークン LFM3-1.7B 12 KiB、Gemma 4 E2B 6 KiB、E4B 16 KiB。
- 点数の倍率: `paged.Shape.Scale` (0 なら 1/√d。Gemma 4 は 1 で、q と k の norm がその役)。
- Qwen3 (`llm.Unpaged`、28 層すべてが注意) は既定では Core AI だけで分ける (`pagedChunk`)。Core ML と Intel では `--paged` を与えたときだけ。分けないときの文脈は 4096 (28 層の鍵と値を 1 本に持つとそれを超えて失敗する)。M3 の Core AI で分けて短い塊つき: 22 トークンのプロンプトで 19.1 → 36.1 tokens/s、2149 トークンで 18.0 → 21.6、プロンプト 3.6 → 1.7 秒。Core ML では差が小さく (M3 19.0 → 21.2)、M1 の Core ML では分けると遅かったので、選んだときだけ。
- 注意の層ごとにホストとの往復が 1 回ある (Core AI で 1 回約 0.4 ms の固定費)。注意の層の多いモデルは層の数だけ払う。短い会話では分けない 1 本の段 (下) がこれをなくす。

### フラグ (hwnpu-chat)

| フラグ | 既定 | 意味 |
|---|---|---|
| `--paged N` | 0: models.yaml、なければ分けられるモデルで 8192 (`paged.DefaultChunk`)。−1 で分けない | 注意の塊の長さ |
| `--shortest N` | 0: Core AI 1024、Core ML 4096、Intel なし。−1 でなし | 短い塊の注意のプログラム (`--paged` 未満の N、2N、…) |
| `--context N` | 0: models.yaml、なければ分けたモデルはモデルの最長、分けないモデルは 16320 (分けない Qwen3 は 4096) | プロンプトと答えのトークン数 |
| `--sparse` | true (`--no-sparse`) | 約 48K からの 2 段の注意 |
| `--tier` | models.yaml、なければ `auto` | 短い会話用の短いプログラム (下) |
| `--prefill` | models.yaml、なければ `auto` | プロンプト用のプログラムを別に持つか |
| `--draft` | 8 | 生成の 1 回のトークン数 |
| `--prompt-cache DIR` | コンパイルキャッシュの `prompts`。`off` でなし | ディスクのプロンプトストア |

### 会話を覆う一番短い塊

- 1 層の 1 回の費用は、固定の約 0.36 ms と 1K の鍵あたり約 0.14 ms (M3、Core AI、LFM3-1.7B、8 トークン: 1024 で 0.50、2048 で 0.63、4096 で 0.93、8192 で 1.49 ms)。n を覆う 1 つの塊は、長い塊 1 つより、短い塊いくつかより安い。
- `decode.Program.Smaller` が短い塊の注意のプログラムを持つ (それぞれ次と `Attn` を割り切る)。セッションはキャッシュを会話を覆う一番短い塊で持つ (`Session.fit`): ステップが塊を越える前に、行を同じ位置のまま次の長さへ移す (ヘッドごとに連続する行をまとめて、最大 4096 行)。`Reset`、`Rewind`、`Truncate` の後の次のステップで短い塊に戻す。`ImportRows` も先に合わせる。
- `llm.AddSmaller` が書き出し済みのモデルに変換し直さずに足す (`Manifest.Smaller`、`OUT.attnC.<ext>`)。重みのある部分は塊の長さによらない。`chain` は NPU に全部載らない短いプログラムを外し (`Chain.Chunks`)、hwnpu-chat は 1 つの段のプログラムがみな持つ塊だけ使う (`commonSmaller`)。
- Core ML は小さい注意を CPU に置く: grouped の Q=1 は塊 8192 まで、Q=8 は 2048 まで CPU、Q=64 は 1024 から ANE (M3 と M4 で同じ。機種でなく macOS 27 の Core ML の判断)。そのため Core ML は 4096 から。
- 結果 (M3、LFM3-1.7B、64 トークン): 150 トークンの会話で Core AI 44.1 → 53.7 tokens/s、Core ML 42.4 → 47.9。2183 トークンで Core AI 40.3 → 45.0 (プロンプト 1.6 → 1.1 秒)。

### 注意のプログラムの形

- `grouped` (`blocks.SDPAGrouped`): kv ヘッドごとに、重ねた問い合わせのヘッドとの積 1 回。K/V を `tile` で複製しない (`tile` の形、LFM3 の `attn=tile` は 2026-11-26 に削除)。
- `paged.Heads` (Core AI、kv ヘッド 2 つ以上。`llm.attentionForm`): kv ヘッド 1 つ分の問い合わせを (1, 1, 行, d) にし、スコアを (1, 行, 1, キー) に並べ、倍率は問い合わせに掛け、最大値と和は最後の軸で `reduce_sum`。1 回が 5〜26% 速い (M1)。kv ヘッドが 1 つ (Gemma 4 E2B、PLaMo 3 0.8B) だと Neural Engine から外れるので `grouped` のまま。
- `llm.RefreshAttention` が前の書き出しの注意のプログラムを書き直す (`Manifest.AttnForm`)。重みの部分はそのまま。`chat.Prepare` が `AddMass` の後に呼ぶ。
- mass のプログラム (`paged.BuildMass`) は `grouped` のまま。

## 段: 会話の短いうちは短いプログラムで

段はすべて起動時に読み込む (`chain.LoadAll`): 生成しながら次の段を裏で読み込むと、生成が 2 倍以上遅くなった。全段を常駐させても動いている段は遅くならない。セッションはプロンプトが収まる最も短い段で始め (`tierFor`)、`room(n)` が収まらないステップの前に上の段へ移す。段の切り替えは 1 回約 15 ms (Core ML)。

| 段の種類 | いつ | 移り方 |
|---|---|---|
| 分けないプログラムの容量の段 (`--tier 4096,…`) | 分けないモデル。`auto` は lut4 なら 4096、lut6/lut8 ならなし (lut6 の 4 本はコンパイルのキャッシュからあふれ、起動のたびにコンパイルし直す) | `Session.Migrate` が全注意の層の行を同じ位置に写す。リングは同じ長さ |
| 分けた形の前の分けない 1 本 (`wholeTiers`) | Core AI、分けたモデル、`auto`。キャッシュは 512/1024/2048/4096 のうち 1 層の K/V が 4 MiB 以下の最長 (LFM3-1.7B 2048、Qwen3-0.6B 1024)。重みの 2 倍が 2.5 GiB (`wholeHold`) を超えるならなし、M1 ではなし (`wholeSlower`) | `migratePaged`: 全位置の行をセッションの塊 (覆う一番短い塊) に移す。同じ長さのリング、窓、状態はそのまま写す |
| 短い文脈の分けたプログラム (`ringTiers`) | Core AI、`llm.Windowed` のモデル (PLaMo 3): 窓の 4 分の 1、256 以上なら (PLaMo 3 は 512) | `migrateSplit`: ホストの塊はそのまま。全位置を持っていたキャッシュはリング (`pos mod 長さ`) になる。`Session.Remark` が移る前の印を作り直す (リングが一周した後は nil) |

- 分けない 1 本は呼び出しのたびにキャッシュ全体を Core AI の配列にコピーする。費用は 1 層の K/V のバイト数で決まり、1 層 4 MiB あたりで分けた形と並ぶ (M3 Core AI の tokens/s、分けない 512/1024/2048/4096 対 分けた形: Qwen3-0.6B 68/53/35/20 対 34〜40、LFM3-1.7B 64/62/59/52 対 55)。段つきで、22 トークンの会話が Qwen3-0.6B 37.4 → 52.5、LFM3-1.7B 54.9 → 59.2 (M3)。
- M1 では分けない 1 本が、塊 1024 の分けた形より遅い (LFM3-1.7B 33.6〜37.0 対 39.9〜43.6 tokens/s) ので置かない。
- 分けない 1 本はキャッシュの長さごとに別の `.aimodel` で、重みは Neural Engine に 2 回載る。リングの段は長い文脈のプログラムと同じ `.aimodel` の関数 (`llm.ConvertAI` は同じ層から始まる部分を 1 本の `.aimodel` にする) で、重みは 1 回分。
- PLaMo 3 Translate 2B (M1、窓 2048 の 21 層で、1 回にリング 84 MiB): 78 トークンの段落が 12.9 → 18.9〜19.0 tokens/s (1 ステップ 77.3 → 52 ms、重みの床 49.7 ms の近く)。438 トークンのプロンプトと 469 トークンの訳で 21.7 → 26.0。Gemma 4 (窓 512、リング 14 MiB) は 256 の段で速くならないのでなし。
- models.yaml の `tier: none` が 8 GB の機械で分けない 1 本を外す (LFM3-1.7B、Qwen3-0.6B、Qwen3.5-0.8B)。
- 2 段の注意は最後の段だけに付ける。段を移ったプロンプトの読みの途中の印はストアに入れない。

## プロンプトを読む

- プロンプトのプログラムの Q: 1 本のモデル (鍵の軸を 16384 以下に) と Intel (実機で測っていない) は 64。Neural Engine の分けたモデルと、専門家を経路で選ぶ MoE のモデルは 128 (`pagedPrefillSeq`、塊が小さければその長さ)。64 行でも 1 回は重みを読む時間が大半。256 は 128 より速くない。LFM3-1.7B (M3) の Q=128: 4K 3.3〜3.9 秒、16K 15.2〜16.1 秒、64K 136〜146 秒 (Q=64: 3.9〜5.4、19.3〜25.2、160 秒)。Gemma 4 E2B の 16K 31.2 → 23.5 秒、LFM3-VL の 16K 21.6 → 16.9 秒。
- 最後の行だけのロジット (`llm.Options.LastRow`、`-last-row`、`llm.Arch.LastRow`: LFM3、Gemma 4): 入力 `last` (1, 1, Seq) の one-hot と隠れ状態の積 (`blocks.Graph.LastRow`。`gather` は CPU に落ちる) で最後の norm の前に 1 行を取り出し、y は (1, 1, vocab)。`decode` が `last` を書き、0 行目を読む。Intel では付けない。`Speculate` はこの形のプログラムを選ばない (行ごとのロジットが要る)。名前の終わりは `-q128-last`。
- KV だけのプロンプトのプログラム (`llm.Options.KVOnly`、`-paged` と組で `-kv-only`): 後ろの層がキャッシュを書かないモデル (`llm.CacheWriters`: Gemma 4 の共有の層) では、キャッシュを書く最後の層までの部分だけを残す。最後の部分はその層の注意の手前で終わり、その K/V を出す。出力層はない (`decode.Program.NoLogits`、`writeUnattended`)。エンジンはプロンプトのうち最後の生成 1 回分を除いた部分をこれで読み、残りを生成用で読む (`Engine.logitsQ`)。名前の終わりは `-q128-kv`。Gemma 4 E2B (M1): プロンプト用が 1.6 → 0.38 GiB で全部 Neural Engine、4249 トークンが 4.2〜4.6 → 2.4〜2.7 秒、その後の生成 18.5〜21.5 → 21.6〜24.4 tokens/s (2 本目が生成の部分を追い出さなくなった)。ロジットのない印はプロンプトの終わりには使わない (`errNoLogits`)。
- ホストでのロジットのソフトキャップ (Gemma 4): [LLM のモデル](llm-models.md)。`decode` は部分のメタデータのキャップを掛ける (`SetRawLogits(true)` で掛けない)。

## プロンプトの再利用

### メモリの中 (`chat.Engine`)

- 全注意の行は戻すのにコピーが要らない: 位置より先の行はマスクで見えず、後のステップが上書きする。リングと畳み込みの窓はその場で上書きされるので、エンジンは読みながら 512 トークンごとと各プロンプトの終わりに `Mark` (リング、窓、最後のロジット。LFM3 で約 3〜3.5 MB) を取り、共通の先頭の中の最後の印に `Rewind` する。
- チャットの書式は前の `<think>` を消すので、共通の先頭はたいてい前のプロンプトの終わりになる。
- ほかの系列を 2 本まで退避する (`maxSaved`)。`Session.Save` は位置までの行だけを写す (`ExportRows`)。並行する要求 (OpenCode の題名や要約) で本題を読み直さずに済む。退避した系列は自分の段を覚える。

### ディスク: プロンプトストア (`internal/chat/store.go`、`--prompt-cache`)

- 全注意の行は位置ごとで、段や塊に依らない。`ExportRows(from, to)` は位置順に (リングでないキャッシュを名前順、ヘッドごとの行、fp16) 詰め、`ImportRows` はどの段のセッションにも戻す。`Resume(rows, mark)` は、印のバッファがセッションのリングと窓と名前と大きさで合い、行が印の位置でちょうど終わることを確かめる。
- 置き方: モデルごとのディレクトリ (`storeKey`: 名前、パッケージの名前と時刻、重みの大きさ。変換し直すと別のキー) に、系列ごとの項目のディレクトリ: `meta.json` (トークン、行数、印の位置、最後の使用)、`rows.bin`、`mark-POS.bin` (リング、窓、ロジットの gob)。
- `find` は共通の先頭が最も長い項目と、その中の最後の印。`put` は、系列が続きなら項目に追記 (1024 位置ずつ書く)、分かれたら新しい項目 (共通部分の印は写す)、足りない印を足す。印は 512 トークンごと (`markEvery`)。1024 トークン未満の系列は保存しない (`minStored`: 1 秒で読める)。モデルごと 4 項目 (`maxStored`)、最も使われていないものから消す。書き込みは一時名からの改名で、`meta.json` を最後に書く。
- 読めない項目は使うときに消し、次善の項目を試す。`openStore` は `meta.json` のない 1 時間 (`staleAfter`) より古いディレクトリを消す。
- `Engine.prefill` はメモリ内の再利用より先へ進めるときだけストアを使い、プロンプトが収まる最も短い段で戻す。`Stats.Restored` が戻したトークン数 (`Reused` と `usage.cached_tokens` に含まれる)。サーバーのログは `(N tokens from the prompt store)`。
- 大きさ (LFM3-1.7B): 行は 1 トークン 12,288 バイト、印 1 つ 3.45 MB、印込みで 1 トークン約 18.7 KiB。7,857 トークンの項目が 145 MB。
- M3、LFM3-1.7B、OpenCode の最初の要求 (7,854 トークン): 冷えた読み 14.6 秒 → 再起動後 0.2 秒 (ストアの読み出し自体は 17〜36 ms、残りは ANE のバッファへの書き戻し)。依頼の文を変えても 0.7 秒 (印 7680 から)。書き込みの上乗せは 0.5 秒以内。貪欲法の出力は冷えた読みと一致。
- hwnpu-agent のプロンプト (system と道具で 617 トークン) は `minStored` 未満で保存しない。

## 2 段の注意 (`decode.Sparse`、`--sparse`)

分けたモデルで約 48K トークンから (塊 8192)。Latent Sparse Attention の考えを、蒸留した射影なしで使う。

- 要約: 全注意の層と kv ヘッドごとに、窓より前の 8 位置のバケットの鍵の平均と値の平均。塊と同じ形 (1, kv_heads, 8192, head_dim) のデバイスのバッファに、バケットが窓から出るたびにホストが足す (fp32 で平均し fp16 で持つ)。キャッシュが 8 分の 1 増える (LFM3 は 1 トークン 12 → 14 KiB、Gemma 4 E2B は 6 → 7 KiB)。
- 重さ: (1) 全部の要約への注意を今までの注意のプログラムで (数える要約のマスクは log 8、ほかは −1e4)。これが要約の分の注意の出力にもなる。(2) mass のプログラム (`paged.BuildMass`) が、q、要約の鍵、マスク、各行の −log-sum-exp (`Merge.Norm`) から要約ごとの重さを出す。
- 選択: kv ヘッドごとに、重い順に `Exact` 個の塊 (2 = 16384 の鍵) に窓と呼び出し自身と収まる数、と最初のバケット。ホストの選択アルゴリズム (並べ替えなし)。前と同じバケットはそのまま置き、変わった分だけ写す。
- 正確に読む塊: kv ヘッドごとのバケットを同じ場所に並べ、その後ろに窓と呼び出し自身 (どのヘッドも同じ位置)。マスクはヘッドに共通。それらのバケットの要約は (1) から `Merge.Sub` で引く (負なら 0)。
- 1 ステップの呼び出し: 要約の塊 × 2 + 1 + `Exact`。全部を読むより少ないときだけ (約 48K から)。プロンプトの呼び出しは `Prefill` = 4 塊 (32768 の鍵) で、約 57K から。`Refresh` (選び直しの間の呼び出し数) の既定は 1。
- `DefaultSparse()` = Bucket 8、Sinks 4、Window 512、Exact 2、Prefill 4、Refresh 1。書き込み (`run`、`ImportRows`、`Reset`) の位置から先の要約は捨てる (`sparseTouch`)。短い塊 (`Smaller`) では動かず、最も長い塊 (`Attn`) のときだけ。
- 書き出し: `convertPaged` が `out.mass.<ext>` を書く。前の書き出しには `llm.AddMass` が足す (数秒、`chat.Prepare` が呼ぶ)。
- ready の行: `past about 48K tokens, 16384 keys exactly and the rest as means of 8`。

CPU での確認 (LFM3-1.7B、PyTorch bf16、128K、10/50/90% に合言葉): 既定の形で次トークンの損失は +0.007 nats、3 つとも保った。プロンプトを 1 回 64 トークンで読むことが効く (256 では 10% の合言葉を落とした)。

## 長い文脈の数値 (M3 16 GB、Core AI、LFM3-1.7B lut6 grouped)

macOS 27.2.1。`hwnpu-chat serve` を既定のまま (`--paged 8192`、約 48K を超えたら 2 段の注意、下書き 7)、`--prompt-cache off --no-think`、温度 0、1 回ごとにサーバーを起動し直す。読み = 最初のトークンまでの時間。生成 = プロンプトの直後の答え (100〜256 トークン) の tokens/s (採られた下書きの数で動くので幅がある)。文書はこのリポジトリの英語の docs と Go のソース。

| プロンプト (トークン) | 読み、2 段 | 生成、2 段 | 読み、全部 (`--no-sparse`) | 生成、全部 |
|---|---|---|---|---|
| 短い (20) | 0.1 秒 | 44 tok/s (23 ms/token) | — | — |
| 17,322 | 13.3 秒 | 34 tok/s (29 ms/token) | — | — |
| 65,133 | 95〜97 秒 | 37〜53 tok/s (19〜27 ms/token) | 114 秒 | 27 tok/s (37 ms/token) |
| 130,522 | 264〜267 秒 | 17〜30 tok/s (34〜61 ms/token) | 318 秒 | 13 tok/s (77 ms/token) |
| 256,732 | 679 秒 | 16 tok/s (62 ms/token) | 測っていない | 測っていない |

- 繰り返しの文の先頭に置いた合言葉 1 つ (前からの試験) は、3.5 万と 7 万トークンで正しく答える。
- 文書の 10%、50%、90% に置いた合言葉 3 つをまとめて尋ねると、1.7 万で 3 つ中 2 つ、6.5 万と 13 万で 0〜1 つ。鍵を全部読んでも同じ (上の注意の直しの前に測った)。
- メモリ: サーバーの RSS は 1.7 万で 0.59 GB、6.5 万で 1.74 GB、13 万で 1.90 GB、25.7 万で 3.89 GB (キャッシュは会話とともに 1 トークン 14 KiB ずつ増える)。
- ほかのモデル、2 段、Core ML の頃の値 (Core AI では測り直していない): Gemma 4 E2B 64K 204 秒 / 73.9 ms/token、125K 521 秒 / 92.4 ms。LFM3-VL 120K 431 秒 / 101 ms。LFM3-Audio と Omni 120K 約 500〜516 秒 / 115 ms。Gemma 4 E4B (1 本) の 125K の読みは 56 分。
- LFM3-1.7B が確かに使えるのは 128K くらいまで。256K では通常の注意でも 10% と 50% の合言葉を外した (モデルの限界、CPU で確認)。

## バックエンドの違い

| | Core AI | Core ML | Intel の NPU |
|---|---|---|---|
| 注意で分ける | する、Qwen3 も | する (Qwen3 は `--paged` のときだけ) | する (部分と注意を OpenVINO IR に、`out.partN.xml`、`out.attn.xml`) |
| 一番短い塊 | 1024 | 4096 (短いと CPU) | なし |
| 注意の形 | kv ヘッド 2 つ以上なら `Heads` | `grouped` | `grouped` |
| 分けない 1 本の段 / リングの段 | あり (M1 は除く) / あり | なし | なし |
| プロンプトの Q、最後の行 | 128、最後の行 / KV だけ | 128、最後の行 / KV だけ | 64、最後の行なし |
| 分けないモデルの生成用 | 2 部分 | 2 部分 | 1 つのグラフ (部分なし) |
| `--prefill auto` | 予算 3.5 GiB (M1 は 2.8) | 同じ | 常に 2 本 |

Level Zero と OpenVINO はもともと終わった要求の付け替えを許す。Core ML の `request.Bind` も許すように直した (`TestBindAfterSubmissionFinished`)。AMD はこのエンジンの外: IRON の書き出しは `--max-seq-len` で文脈を固定し、キャッシュはデバイスの bank の中 ([AMD の LLM](amd-llm.md))。

## 落とし穴

- 注意のプログラムは、読み込みの文脈ではなく段の `Program` の文脈で動かす: hwnpu-chat は ready になると読み込みの文脈を取り消すので、その下の呼び出しは `context canceled` を返し、予測は裏で走り続けた (空の答え、`cannot bind while a submission is running`、終了時の SIGSEGV)。`TestPagedOnNeuralEngine` が見る。
- Core ML: 1 つのプログラムの中で塊を合わせる形のいくつかは CPU に落ちる (塊を 1 つの軸に並べて縮約、各塊を `和_J / 全体の和` で重み付け、和に先に 1/n を掛ける)。和を先に足し、最後に 1 回だけ割る。
- Core ML の Q=64: `reduce_sum` の配置がコンパイルごとに揺れる。1 との行列積を使う。
- MLState: `write_state` の値をそのまま使うと ANE のコンパイルが `error code: -14` で失敗する。読み直してから使う。
- とても長いプロンプトを読んだ直後の数回は遅い (256K で 1 回 1.29 秒、その後 350 ms)。原因は確かめていない。
- 2 段の注意は尖った注意を近似しにくい: ノルムの重みを 1 前後にした小さな Gemma 4 (点数が 30 に届く) で、全部を読む形との相対誤差 1.3〜1.6 (普通の重みでは 0.068)。
- `HWNPU_PROFILE=1` (`decode.WriteProfile`) は 1 ステップを部分、層ごとのホストの注意、ロジットの読み出しに分ける。hwnpu-chat は答えのたびに Neural Engine が抱えるメモリと一緒に出す。

## テスト

`TestCachedMatchesFull`、`TestCachedChunksMatchFull`、`TestMigrateTiers`、`TestEngineTiers`、`TestPagedMatchesFull`、`TestPagedShareRewindResume`、`TestEngineReusePaged`、`TestPagedLastRow`、`TestPagedKVOnly`、`TestSmallerChunks*`、`TestShorterChunks(OnNeuralEngine)`、`TestMigrateToPaged(OnNeuralEngine)`、`TestMigrateRings(OnNeuralEngine)`、`TestEngineRingTiers`、`TestWholeTiers`、`TestSparseMatchesDirect`、`TestPagedSparse`、`TestMassMatchesSoftmax`、`TestEngineStore`、`TestResume`、`TestMergeMatchesFull`、`TestRefreshAttention`。どれも CPU のインタプリタでキャッシュなしのプログラムのロジットと比べ、わざと壊した形 (行のずれ、リングの写し忘れ、マスクのずれ) が落ちることを確かめる。

## 採らなかったもの

- キャッシュを MLState に: ホストのバッファと速さが同じ (M3、256〜8192)。下がるのは CPU の使用 (約 0.15 コア) だけ。CPU のインタプリタで動き、段の間で移しやすいホストのバッファのままにする。状態の仕組みは `mlpackage` とバックエンドに残す。
- int8 のキャッシュの入力: 全部 ANE でメモリは半分だが 1 割遅い。メモリのための手法でしかない。
- 分けないプログラムで GQA を `tile` なしの (kv ヘッド, グループ) の行列で読む形: 全演算が CPU (後の `grouped` の形は載る)。
- 次の段を裏で読み込む: 生成が 2 倍以上遅くなる。先に全部読み込む。
- Q=4 の生成用: 1 回は Q=8 より 3% 安いだけで、1 回に採られるトークンが 1 割少ない。
- 段を 1 つの Core ML のパッケージの関数に: blob の書き出しが定数を重複なく書けず、重みが 2 倍で 1 GiB を超える。
- 分けない 1 本と分けた形を 1 本の `.aimodel` に (重みの共有): Qwen3-0.6B で 60 関数になり、読み込み 2 分 28 秒、一部の関数が GPU に外れ、MPSGraph が 1 つ 0.6 GB を `$TMPDIR/com.apple.MetalPerformanceShadersGraph` に書いてディスクが尽きた。
- 分けない 1 本を 2 段 (256、1024) にして 1 本の `.aimodel` の関数に: 1024 の 1 本が半分の速さ。
- 2 段の注意で鍵を 1 つずつ選ぶ (top-k): 理想の top-4096 でも 128K で注意の重みの 53% (KL 0.58)、2 段と並ぶには 32K の鍵が要る。Quest のページの上界は悪く、RoPE のない次元での k-means は 8 トークンの固定バケットと同じ、16 トークンのバケットは損失が増え、正確な鍵 4096 と先頭 4K を常に正確にする形は 10% の合言葉を落とした。
- Core AI の塊 4096 を `--paged` の既定に (M1 の Gemma 4 E2B): ホストの注意は 1 層 2.05 → 1.36 ms だが 1 回 0.67 ms の固定費が残り、長い文脈では 1K あたり 1 割悪くなる。短い場合は短い塊 (`--shortest`) が受け持つ。
- 注意を CPU で: 電力を優先して採らない (NPU に載せる)。
- Core AI でゲートと up を 1 つの線形に (AMD と同じ形): 部分の時間 3.31〜3.33 対 3.30〜3.32 ms。重みを読む速さで決まり、Core AI は 2 つの線形の間に呼び出しの境目を作らない。
- Gemma 4 E4B を 4 ビット (lut4) にして 2 本を収める: 次のトークンの一致が lut6 の 90% に対し 64%、速さは 2 割だけ。
- Gemma 4 のリングの段 (窓 512): 1 ステップ 40.1 対 40.4 ms。

## 未解決

- `Refresh` 4 や 16 は 128K で 1 回が 14〜32% 速く、合言葉も保った。Neural Engine で損失を確かめるまで既定は 1。
- 2 段のプロンプトは 64K で 1 割遅い (約 57K の始まりが低すぎるかもしれない。試していない)。
- 128K を超えると 1K あたりの費用が上がる (1.14 → 1.43 ms) ことと、長い読みの直後の遅い呼び出し: 原因は不明 (メモリの圧迫かバッファの数)。
- `choose` のホストの時間 (選び直したバケットの memmove) が 128K で 1 回約 13 ms (12%)。
- ストア: 退避した系列 (`Save`) もストアに置く、要求ごとの終わりの古い印を間引く、行を int8 に。
- 全部の層がキャッシュを書くモデルの KV だけのプロンプトのプログラム (LFM3 の出力層、約 0.19 GiB を外す)。
- 注意を直した後も、2.4 万〜3.2 万トークンでの遠くの想起は PyTorch に届かない: 合言葉 3 つで、hwnpu の lut6 は 2.5 万で 3 つ中 2 つ (PyTorch の fp16 は 3)、別の 3.3 万の文書で 0 (PyTorch は 2)。lut8 は lut6 が今は答える文書で 0。NE の道の残り (重みの部分、ほかの行列積) は同じやり方ではまだ見ていない。
- 同じ小さな積の取りこぼしは、`matmul(e または softmax, v)` で書いたほかの注意にもあるだろう: `blocks.ChunkedSDPA` / `ChunkedSDPAGrouped` (分けない長いキャッシュ。e を log(4·塊の数) だけ下げてもいる)、`blocks.SDPA` / `SDPAGrouped` (softmax のあと積)、画像のモデルの `milpart` の注意 (e·(1/sum))。直すと重みのあるプログラムを書き出し直す (アーキテクチャごとに `Revision`)。
- 上の長い文脈の表は注意を直す前に測った (今はプロンプトの読みが数 % 遅い)。
- 測っていない: M2/M3 での `Heads` の形、M2/M3 と PLaMo 3 0.8B / 8B のリングの段、サーバーでいくつもの会話が段を行き来するとき、Intel の NPU の実機での分けたモデル。
