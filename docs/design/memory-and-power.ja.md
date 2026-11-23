# メモリと電力

[English](memory-and-power.md)

電力量とメモリの測り方、NPU ごとの電力の制御、機械のメモリの区分に合わせたモデルの動かし方、NPU が一度に持てる量の
調べ方。Go の API (`WithPowerPolicy`、`PowerSample`、`AveragePower`) は [api.ja.md](api.ja.md)、Neural Engine に固有の
メモリの上限は [apple-neural-engine.ja.md](apple-neural-engine.ja.md)、KV キャッシュのメモリは [llm-engine.ja.md](llm-engine.ja.md)。

## 電力ポリシー

`WithPowerPolicy(PowerDefault | PowerLow | PowerBalanced | PowerPerformance)`。`CapPowerControl` のないデバイスは無視する。

| バックエンド | Low / Balanced / Performance |
|---|---|
| Intel (Level Zero) | キュー優先度 LOW / NORMAL / HIGH (いつも)。NPU キュー拡張があれば、Low でワークロード種別 BACKGROUND、Performance でターボ (拡張 1.0 は `turbo`、1.1 は `options`) |
| AMD (amdxdna) | ハードウェアコンテキストの QoS の優先度 (権限不要)。デバイス全体の電力モード (`SET_STATE`) は root が要り、全ユーザーに効くので触らない |
| Apple (Core ML、Core AI) | 効果なし (`CapPowerControl` なし。選べるのは compute units だけ) |

## 電力計

`Device.PowerSample()` は積算のジュールを返す。`Scope` は `npu` か `package` (CPU パッケージ全体)。

| バックエンド | 源 | 範囲 | 備考 |
|---|---|---|---|
| Intel、Linux | RAPL `/sys/class/powercap/intel-rapl:N` | package | 通常 root が要る。カウンタの巻き戻りに対応 |
| AMD、Linux | ドライバにあれば NPU のセンサー (`QUERY_SENSORS`、AMD の DKMS のドライバ)。なければ amdgpu の hwmon の `power1_input` (PPT、µW、root 不要) | npu / package | Linux 7.0 の標準の amdxdna には NPU のセンサーがない。どちらも瞬間値なので、最初の `PowerSample` から 50 ms ごとに読んで台形で積分する裏の処理を回す (`power.Integrator`)。閉じると止める |
| AMD、Windows (xrtshim) | XRT の electrical info (W) | — | 読めれば `CapPowerMeter` |
| Apple | API にはない | — | 計測は `hwnpu-dev powerbench` (root)。`hwnpu status` の IOReport の電力も root のときだけ |

- テスト: `Provider.Hwmon` で hwmon の場所を差し替える。`xdnasim.NoSensors` で Linux 7.0 と同じくセンサーなし
- macOS の IOReport のエネルギー (`Energy Model` の ANE と `CPU Energy`) は root でないと全部 0。`hwnpu status` は
  0.00 W ではなく「power not readable without root」と出す ([cli.ja.md](cli.ja.md))

### HWNPU_ENERGY (1 回の実行の段階ごとの電力量)

- `HWNPU_ENERGY=秒数` (数でなければ 5): `main` の最初の `console.MeasureEnergy` が、その秒数だけ待機電力を測り (実行の
  時間には数えない)、段階 (変換・読み込み・コンパイルと、実行) が変わるたびにパッケージの電力量を分ける。`took …` の行の
  下に、各段階の電力量と、待機電力 × その段階の時間を引いた値を出す (`energyLine`)
- 源は `power.PackagePower` (amdgpu の PPT) なので、今は Linux の AMD の APU だけ。ほかではそう言って何もしない。
  hwnpu-image、hwnpu-chat、hwnpu-audio で呼ぶ
- 段階の数え方は時間の行と同じ ([cli.ja.md](cli.ja.md)): 実行の途中の読み込みと重みの組の読み直し (`iron.Graph.Use`、
  `console.Reading`) は準備に数える
- 計測 (Linux の AMD の APU。1024² の 1 枚、ファイルはページキャッシュにある、待機 6.7〜7.1 W):

| モデル | 時間 (準備 / 実行) | 電力量 (準備 / 実行) | 待機を引いた電力量 (準備 / 実行) | 組の読み直し |
|---|---|---|---|---|
| Anima-Turbo | 46 秒 (9.6 / 36) | 698 J (121 / 577) | 56 / 329 J | 55 回、30.5 GB、4.2 秒 |
| Anima-Aesthetic | 3 分 18 秒 (11 / 186) | 3.23 kJ (158 / 3.07 k) | 79 J / 1.82 kJ | 174 回、55.5 GB、6.3 秒 |

  - 実行の電力量の約 4 割が待機の分
  - 準備は全体の 3〜8 %: グラフの読み込み 4.5〜8.8 秒と、ページキャッシュからの組の読み直し (10〜18 GB/s)。読み直しは
    ステップの数とともに増える。組を全部置けば消えるが、固定のメモリが組の数だけ増える ([image-amd.ja.md](image-amd.ja.md))
- AMD のほかの値: Gemma 4 E2B の生成中はパッケージ全体で平均 11.8 W、1 トークン 1.93 J (待機の 5〜7 W を含む)

### powerbench (Apple)

- `hwnpu-dev powerbench` (darwin): sudo のパスワードを最初に一度だけ聞き、計測ごとに `sudo -n powermetrics -f plist` を
  起動する。各モデルを別スレッドで呼び続け (`-rate` で一定頻度)、CPU / GPU / ANE の電力から待機を引いた値と、1 回あたりの
  mJ を出す。Neural Engine に載るモデルは CPU だけの場合も測る (`-cpu=false` で省く)。`-decode 8,16` は言語モデルの
  デコードの形のモデルを作る (1 トークンあたりの電力量)
- `HWNPU_COREML_UNITS=cpu|gpu` で Core ML を `MLComputeUnitsCPUOnly` / `CPUAndGPU` で読み込む。計測専用で、公開の API に
  はしない。大きさを変えた合成モデルは `tools/coreml/gen_sweep.py`
- 結果 (M3): 損益分岐点は 1 回あたり約 200 MFLOP。それより大きいと ANE は 2〜3 倍速く、電力量は約 4 分の 1 (約 1 TFLOP/J
  対 CPU 0.25 TFLOP/J)。小さいと Core ML の 1 回の固定費 (約 55 µs) と ANE の起動が勝ち、CPU が有利。詳しくは
  [MODEL_ARCHITECTURE.ja.md](../../MODEL_ARCHITECTURE.ja.md)

## メモリの測り方

| 何を | どう | 落とし穴 |
|---|---|---|
| Apple の Neural Engine の重み | `mem.NPUWired`: カーネルの wired メモリの `com.apple.driver.AppleH11ANEInterface` を `mach_memory_info` で読む (root 不要、purego) | RSS にも footprint にも `aned` にも出ない。機械全体の値 (ほかのプロセスが混ざる)。vm のタグ番号は macOS の版で変わったので名前で探す。何も載せていないときは 144〜192 KiB。Core ML がコンパイルしたばかりのプログラムは、読み込みでなく最初の呼び出しで載る (LFM3: 読み込み後 0.27 GiB、1 回呼んだ後 2.43 GiB): 1 回呼んでから測る |
| Apple のプロセスごと | `proc_pid_rusage` (flavor 6) の `ri_phys_footprint` + `ri_neural_footprint` を 0.1 秒ごとに読み、和の最大を取る | プロセスごとなので、ほかのプロセスの Neural Engine の分は混ざらない。mlock したファイルのページは footprint に入らないので wired を足す (専門家を 6.4 GB 固定した gpt-oss-20b: footprint 0.53、wired 6.41、resident 9.53 GB)。resident は固定していないページキャッシュと共有ライブラリを数え、スワップと圧縮は数えない |
| 1 回の実行の最大 | `go run ./tools/memwatch [-csv FILE] CMD …`: footprint、wired、`mem.NPUWired`、システムの空きの割合と逼迫を 0.1 秒ごと。最大と models.yaml の `memory` を出す | 内訳は `footprint PID`、`vmmap -summary PID` |
| AMD の固定メモリ | `/proc/meminfo` の `Unevictable` の増え。プロセスの分は smaps の `/dev/accel/accel0` の写像の大きさの和。書き出しの `manifest.json` の組 | 固定したページは RSS にもスワップにも出ない。ほとんどが重みと作業領域で、KV は 5 % 未満 |
| Windows | `K32GetProcessMemoryInfo` の private。`maxRSS` は peak working set | |
| 実行中 | `HWNPU_PROFILE=1`、hwnpu-chat の読み込みのログ (「the Neural Engine holds …」)、`hwnpu status` のプロセスの表 | |

- モデルの値は 2 回目 (変換とコンパイルが済んだ後) を取る。1 回目の最大は変換の分
- 変換の後は `debug.FreeOSMemory` を呼ぶ (Go がチェックポイントの分をなかなか返さなかった: 起動直後 2.4 GB → 154 MB)
- Core ML は、同じ重みでも 2 つのプログラムが持てば 2 回分持つ。Core AI の複数の関数を持つ `.aimodel` は 1 回分
  ([apple-coreai.ja.md](apple-coreai.ja.md))

### 今の値 (M1 16 GB、macOS 27.2.1、Core AI)

`hwnpu-chat -m NAME --temp 0 -n 200 "…"` の 2 回目。「合わせて」は Neural Engine + プロセスの和の最大を GB で。

| モデル | Neural Engine | プロセス | 合わせて | 変換のときのプロセスの最大 |
|---|---|---|---|---|
| LFM3-1.7B (lut6) | 1.36 GiB | 0.23 GiB | 1.71 GB | — |
| 同、M2 以降の形 (`--tier 2048`) | 2.64 | 0.21 | 3.05 GB | — |
| Qwen3-0.6B (lut8) | 0.65 | 0.35 | 1.07 GB | — |
| 同、M2 以降の形 (`--tier 1024`) | 1.23 | 0.26 | 1.60 GB | 2.0 GiB |
| Qwen3-1.7B (lut8、プロンプト用の関数つき) | 1.80 | 0.36 | 2.32 GB | 3.1 GiB |
| 同、`prefill: off` | 1.64 | 0.23 | 2.01 GB | 1.8 GiB |
| Qwen3.5-0.8B (lut8、MTP の層つき) | 0.99 | 0.39 | 1.48 GB | 3.0 GiB |
| 同、M2 以降の形 (`--tier 2048`) | 1.84 | 0.31 | 2.29 GB | 3.3 GiB |
| Qwen3.5-2B (lut6) | 1.67 | 0.39 | 2.20 GB | 6.0 GiB |
| LFM3-VL-2B (lut8) 文だけ / 画像 1 枚 | 2.14 / 2.24 | 0.23 / 0.25 | 2.55 / 2.67 GB | 2.9 GiB |
| LFM3-Audio-2B 文だけ / 音声 1 本 | 2.05 / 2.18 | 0.23 / 0.21 | 2.44 / 2.57 GB | 2.8 GiB |
| LFM3-Omni-2B 文だけ / 画像と音声 | 2.40 / 2.63 | 0.23 / 0.25 | 2.83 / 3.10 GB | 2.9 GiB |
| PLaMo 3 Translate 0.8B (lut8) | 0.82 | 0.12 | 1.01 GB | 2.3 GiB |
| PLaMo 3 Translate 2B (lut8) | 2.74 | 0.24 | 3.20 GB | 2.7 GiB |
| PLaMo-Embedding-1B (lut8) | 1.05 | 0.03 | 1.16 GB | 1.3 GiB |

- プロセスの定常は 0.03〜0.4 GiB (トークナイザー、キャッシュ、読んだ埋め込みの行)
- 「M2 以降の形」は短い会話用の分けない 1 本を足したもので、重みをもう 1 組持つ ([llm-engine.ja.md](llm-engine.ja.md))。
  M1 で `--tier` を与えて測った値で、M2 と M3 の実機では測っていない
- LFM3 の VL と Omni は符号器を初めから読み込む (Omni は Apple で画像か音声が来る前から 2.8 GB)
- 変換のあいだプロセスは 2〜3 GiB (Qwen3.5-2B は 6.0 GiB) で、まだ Neural Engine には載っていない。8 GB の機械で
  Qwen3.5-2B を変換するとスワップに入るので、8 GB の機械は変換済みのプログラムを取ってくるのが前提
  ([distribution.ja.md](distribution.ja.md))
- PLaMo 3 Translate 2B の Neural Engine の分 (2.74 GiB) は、M1 が全速で持てる 2.86 GiB の内
- AMD のモデルごとの固定メモリは [amd-llm.ja.md](amd-llm.ja.md)

## 機械のメモリの区分

どのメモリの機械でどう動かすかはモデルの定義 (models.yaml) が決めるもので、使う人がフラグごとに決めるものではない。
使う人が選ぶのは「この機械をメモリいくつの機械として扱うか」だけ。

### models.yaml の中

- `machine`: そのモデルを出す最小の区分。ファミリーや用途の名前 (`lfm3`、`qwen3.5`) は機械のメモリで選ぶ
- `memory`: モデルが使う GB。ベンダーごと、チップごと (`chat.PerVendor`): `memory: {amd: 2.2, apple: 3.1, M1: 1.7}`。
  Apple の NPU では、その Mac のチップの欄 (`chat.Chips`: M1〜M5、`not_on` と同じ名前) があればそれを取る。
  `hwnpu-chat list` が出す。同じファミリーの候補を比べる値 (`Model.Memory`) はチップを見ない
- `variants: {16: {…}}`: メモリごとの動かし方の差分 (`chat.Run`: `prefill`、`format`、`context`、`paged`、`tier`)。
  `tier` は `--tier` と同じ値 (`auto`、`none`、長さの並び)
- 8 GB の機械の目安は 1 モデル約 3 GB

| 決まり | モデル | 理由 |
|---|---|---|
| 8 GB では短い会話用の分けない 1 本を読まない (`tier: none`、`variants: {16: {tier: auto}}`) | LFM3-1.7B、Qwen3.5-0.8B | 重みをもう 1 組持つ: LFM3-1.7B は 1.7 → 3.1 GB で 8 GB の目安を越える |
| 変えない (8 GB でも 1 本を持つ) | Qwen3-0.6B | 1 本を足しても 1.6 GB で目安の中。M3 では約 2 倍速い (68 対 34〜40 トークン/秒)。`tier` はベンダーを分けないので、Intel の 4096 の段まで消してしまう |
| 16 GB 未満ではプロンプト用のプログラムなし (`prefill: off`) | Qwen3-1.7B | 8 GB では小さい方を既定に。Core AI で省けるのは今 0.3 GB。速さが要れば `--prefill on` |
| `machine: 8` | Qwen3.5-2B | 2.2 GB。`-m qwen3.5` は 8 GB で 2B を選ぶ (0.8B は `qwen3.5:0.8b`) |
| M1 は短い会話用の 1 本を読まない | 全部 | その場で見る (`chipName`) |

今の 8 GB の Apple: LFM3-1.7B 1.7 GB、Qwen3.5-0.8B 1.5 GB、`-m qwen3.5` は Qwen3.5-2B の 2.2 GB。8 GB でいくつも同時に
動かすなら LFM3-1.7B + PLaMo 3 Translate 0.8B (Apple 2.7 GB、AMD 3.2 GB)。

### 区分の選び方

- `hwnpu memory LFM3-1.7B 8` (引数なしで一覧、`--forget` で忘れる)。`hwnpu-chat --memory 8` やメモリのタグ (`lfm3:8g`、
  `:8GB`、`:8gb`) で用意したモデルも、用意が済んだらその選択を残す (`keepMemory`)。API は `modelstore.Store` の
  `Memory`、`SetMemory`、`Memories`
- 置き場所は設定のフォルダの `run.json` (`chat.RunFile`: ユーザーの models.yaml の隣、
  `{"models": {"LFM3-1.7B": {"memory": 8}}}`)。キャッシュには置かない (`hwnpu cache prune` で消え、`HWNPU_CACHE` を共有すると
  ほかの機械に移る)。models.yaml にも書かない (機械が書き戻すとコメントが崩れる)。`go test` は `XDG_CONFIG_HOME` がなければ
  読み書きしない
- 守るところは `lookupOn` (Resolve、ResolveFor、ResolveForKinds、`serve`、`modelstore`): モデルを解決するコマンドはどれも
  その区分で動かし、その区分のプログラムだけを変換するか取ってくる。起動時に 1 行:
  `LFM3-1.7B: runs as on a machine of 8 GB, as chosen (…/run.json; hwnpu memory LFM3-1.7B --forget forgets it)`
- 優先の順: その実行のメモリのタグか `--memory` → 選んだメモリ → `HWNPU_MEMORY` (実行全体の上書き、残さない) → 機械のメモリ
- ファミリーの名前がどのモデルを選ぶかは機械のメモリで決める (選択はモデルごとで、選ぶ前には分からない)。選んだモデルに
  区分が選んであれば、その区分で動く
- `--tier`、`--prefill`、`--format`、`--context`、`--paged` は残さない: その実行だけ。なければその区分の定義の値
- 前の区分のプログラムは LRU で消えるまでキャッシュに残る。`hwnpu memory` が `hwnpu cache clear NAME` を勧める
- `variants` を持つのは今チャットのモデルだけ。画像や音声のモデルに足してもそのまま効く
- 実行中の KV キャッシュは物理メモリの 4 分の 1 まで ([llm-engine.ja.md](llm-engine.ja.md))

## 一度に持てる量の調べ方 (`hwnpu-dev probe resident`)

NPU が一度に持てる量を測る。使い方は [PROBE.ja.md](../../PROBE.ja.md)。

- `-resident-size` GiB (カンマ区切りで複数。6100 万重みの層を単位に丸める。既定 0.25) のプログラムを 1 本ずつ読んで 1 回ずつ
  実行し、読み込みが失敗するか `-resident-max-gb` (RAM の 60 %、最大 16) まで続ける。最後に全部をもう一度実行する。メモリ
  不足 (`types.IsOutOfMemory`) なら一番古いものを閉じて、失敗したものを読み直す
- `-resident-hold open,close`: 全部開いたままか、1 本ずつ閉じるか (持っている量の上限か、読んだ量の上限か)。
  `-resident-formats int8`: 上限が重みのバイト数で決まるか。`-resident-run=false`: 読むだけ (落ちるのが読み込みか実行か)
- `-resident-cycle` (既定、開いたままのとき): 各読み込みの後に、持っている全部を順に 2 周実行し、2 周目の平均と最長を記録
  する (`cycle_ms`、`cycle_max_ms`)。最速の平均 (`base_ms`) の 2 倍を超え、かつ 0.5 ms 以上遅ければ遅い。遅くなる前の最大の
  合計 `fast_bytes` と `slow_bytes` を出す。3 回続けて遅ければ順に回すのはやめ (毎回全部の載せ直しになる)、読み込みは続ける。
  Apple は載りきらないものをエラーなしで追い出して次の呼び出しで載せ直すので、上限は失敗でなく遅さで出るため
- Apple では最初の読み込みの前からの `mem.NPUWired` の増え (`npu_bytes`) を、各読み込みと順に回した後に記録する。`held` は
  重みの足し算にすぎない
- 子は 1 本ごとに `npuprobe-progress` の行を出すので、子が落ちても親はそこまでを報告に残す (`crashed`)。各段階 (読み込み、
  実行、閉じる) を標準エラーに書き、落ちた段階がエラーの最後の行に出る
- Windows では各読み込みの後のプロセスの private メモリを記録する
- 偽の NPU (容量で OUT_OF_HOST_MEMORY、閉じても戻らない、最後の 4 本だけ速い) でテストする

分かっている結果:

| 機械 | 結果 |
|---|---|
| Intel Core Ultra 5 135U | Anima の fp16 が約 4.6 GB で `ERROR_OUT_OF_HOST_MEMORY`。一方 0.23 GiB のプログラムはエラーなしで 16 GB まで積めた。probe が 1 度ドライバの中で 0xc0000005 で落ちた (進み具合の行はこのため)。`resident-large` (0.57,1.2 GiB、開いたまま、8 GiB まで) で、1 本の大きさか合計かを分ける |
| M2 16 GB、macOS 27.2 | 0.23 GiB × 42 本 (9.52 GiB) まで、開いたままでも閉じてもエラーなし。中身の違う合成プログラムは毎回 Neural Engine 向けにコンパイル (1 本 1.8〜5.3 秒) |
| M4、Intel | 16 GB までエラーなし |
| M1 | 順に回す部分は、Neural Engine が 2.86 GiB 持つまで全速、2.97 GiB から 1 回ごとに約 12 倍遅い。M2 は 3.65 GiB で全速、M3 と M4 ではこの段差は見ていない ([apple-neural-engine.ja.md](apple-neural-engine.ja.md)) |

## 採らなかったもの

- AMD のデバイス全体の電力モード (`SET_STATE(SET_POWER_MODE)`): root が要り、全ユーザーの NPU を変える
- 8 GB 向けの形を使う人が環境変数 (`HWNPU_MEMORY`) で選ぶ: モデルのセレクター (タグ) と `hwnpu memory` で選ぶ。
  `HWNPU_MEMORY` は実行全体の上書きとしてだけ残す
- メモリの選択をキャッシュや models.yaml に残す (上のとおり)
- 使う人がメモリに合わせて `--tier` や `--prefill` をモデルごとに選ぶ: 区分ごとに定義が決める
- 8 GB で Qwen3-0.6B の短い会話用の 1 本を外す (目安に収まり、M3 で 2 倍速い)
- ANE の呼び出しの費用をならすためのゼロ埋めのパディング: M3 で測った大きさでは得にならなかった

## 未解決

- Intel: ポリシーの違いが実機で電力に表れるか (`hwnpu-dev report -power low` と `-power performance` を比べる)
- Intel、Windows、Apple で NPU 単体の電力を読む手段 (root なしの API がない)
- AMD の APU 以外での `HWNPU_ENERGY` (RAPL、Windows)
- 「M2 以降の形」のメモリと、8 GB で 1 本を読まないときの速さの M2 と M3 の実機での確認 (M3 では 1 本つきが 8 % 速かった)
- Core AI で測っていないもの: Gemma 4 E2B のプロセスの分、Gemma 4 E4B、Qwen3.5-4B、PLaMo 3 Translate 8B (Neural Engine
  3.81 GiB + プロセス 0.3 GiB 前後のはず)、d1
- 実機での `probe resident -resident-cycle`: M1 で 2.9 GiB あたりに `slower from` が出れば測り方が正しい。M2 と M4 の境目。
  `npu_bytes` が重みの足し算からどこで離れるか
- Intel でメモリ不足がエラーでなくプロセスの異常終了なら、エラーを受けて閉じる形は効かない: 落ちる前の合計より手前で自分から
  閉じる上限が要る
