# 変換したモデルを AMD の NPU で動かす: 設計

作成: 2026-11-03。**状態 (2026-11-04): 段階 A1〜A4 を実装し、Gemma 4 E2B が `hwnpu-chat -iron` と `hwnpu-agent -iron` で動いた (design.md 16.47〜16.51)。Windows 向けの XRT 経由のバックエンド (16.56) と、NPU1 向けの「1 ステップごとの xclbin」(16.57) も実装した。どちらも同じ経路をこの機械 (NPU2、Linux) で実機検証したが、Windows と NPU1 の実機では未検証。** 対象は `hwnpu-chat` と `hwnpu-agent` を AMD Ryzen AI の NPU (XDNA2、Strix Point 以降) で動かすこと。根拠は design.md 16.44 (実機での最初の確認) と 16.45 (GEMV と切り替えの測定)、それとこの文書の 2 節の測定。

## 1. 現状

| 層 | 今の状態 | AMD で足りないもの |
|---|---|---|
| ドライバの呼び出し (`internal/xdna`、`internal/backend/amd`) | xclbin と命令列の実行 (ERT_START_CU) が Ryzen AI 9 365 で動く | **full ELF** の実行 (3 節) |
| 演算の実装 | なし。AMD には公開されたグラフ用コンパイラがない | カーネルとその並べ方。自分では書かない (2.3) |
| 変換 (`hwnpu-convert`) | Core ML と OpenVINO IR を書く | AMD 向けの成果物 (5 節) |
| 実行 (`chain`、`decode`) | `hwnpu.GraphRunner` と、ホストに置いた KV キャッシュが前提 | IRON の像を回す実行器 (6 節) |
| チャット (`chat.Engine`) | `decode.Session` に直に依存 | Session の interface 化 (6.3) |
| 実行の入口 | `WithBackend("coreml")` が 6 か所に固定 | 機械による選択 (Intel と共通) |

## 2. 測定から決まったこと

### 2.1 速さと電力 (design.md 16.45)

- NPU が重みを読む速さは約 50 GB/s (bf16 の GEMV、8 列)。CPU が読むだけなら 55〜79 GB/s
- 電力 (パッケージ全体): NPU の GEMV はアイドルより約 5〜8 W 増。CPU は読むだけで 13〜24 W 増。同じ量の重みを読む電力量は NPU が半分以下で、待つ間 CPU は空く
- **ハードウェアコンテキストの切り替えは 1 回約 0.46 ms**。演算ごとに xclbin を投入する方式は、1 トークンに 100 演算なら切り替えだけで 46 ms になり使えない
- ハードウェアの上限 (IRON のサンプルで 2026-11-02〜03 に測定、[amd-xdna2-bench.md](amd-xdna2-bench.md)): DMA は読みだけで 55〜57 GB/s、書きだけで 48 GB/s、読み書き同時で計 64 GB/s (8 本で頭打ち、1 本 14 GB/s)。1 回の実行の固定費は最小 0.06 ms。4096³ の行列積は int8 7.3〜7.6 TOPS、bf16 2.7〜3.0 TFLOPS (bfp16 で代用すると 5.6) で、ピークの 1〜4 割 (コア内の計算が律速)。4 ビットの GEMV は 16 コア以上で読みの帯域に届く

### 2.2 IRON の実績 (amd/IRON の CI、Krackan = NPU2、2026-11-02)

| モデル | 形式 | デコード | 1024 トークンのプロンプトの初回 |
|---|---|---|---|
| Llama 3.2 1B | bf16、full ELF | 14.5 トークン/秒 | 0.47 秒 |
| Llama 3.2 1B | bf16、演算ごとに投入 | 5.4 トークン/秒 | 0.76 秒 |
| Llama 3.2 3B | bf16、full ELF | 6.4 トークン/秒 | 1.6 秒 |
| Gemma 4 E2B | bf16、full ELF | 5.6 トークン/秒 | — |
| Gemma 4 E2B | q4nx (FastFlowLM のエンジン) | 28 トークン/秒 | — |

full ELF と演算ごとの投入の差 (2.7 倍) は、2.1 の切り替え費用とおおむね合う。プロンプトの処理は 2000 トークン/秒前後で、M3 の Neural Engine (OpenCode の 7.4K トークンに 13〜17 秒) より 1 桁速い。デコードは bf16 だと遅く、4 ビットの重みで 5 倍になる。

### 2.3 方針の選択

| 案 | 内容 | 判断 |
|---|---|---|
| A | MIL のプログラムを AIE 向けに下ろす変換器とカーネルを自分で書く | しない。AIE の配置、DMA、カーネルは IRON と mlir-aie が何年もかけて作っている部分で、追いつけない |
| B | 演算ごとに xclbin を作り、Go から順に投入する | しない。切り替え費用 (2.1) |
| C | ONNX Runtime と Vitis AI | しない。Windows が中心、依存が大きく、Linux の NPU 単体では動かない |
| **D** | **IRON をビルド時のコンパイラとして使い、hwnpu は Go の実行器になる** | **採る** |

案 D の考え方は MODEL.md の「開発者が一度だけ変換し、NPU ごとの成果物を配る」と同じ。ただし AMD の成果物を作るのは Go ではなく IRON (Python) になる。利用者の機械には Python も XRT も要らない。これは「変換は Go だけで」という原則の例外で、AMD に公開されたグラフ用コンパイラがない以上ほかに道がない。IRON (Apache-2.0) は、AMD がモデル単位で精度と速さを毎晩確かめているので、移植の検証の手本にもなる。

## 3. full ELF の実行 (hwnpu 側)

IRON は NPU2 で、モデルの 1 回の呼び出し (デコードの 1 ステップ、またはプロンプトの 1 塊) を **1 つの ELF** にまとめる (`Graph.compile(full_elf=None)`)。全演算の PDI (タイルの構成) と制御コードが 1 つのハードウェアコンテキストに入り、1 つのコマンドで流れる。

### 3.1 投入の手順 (実機で XRT 2.21 の ioctl を記録して確かめた)

`LD_PRELOAD` で ioctl を横取りする道具 (`xtrace.so`) を挟み、IRON の 2 演算のグラフ (ReLU → AXPY) を pyxrt で動かして記録した。Linux 7.0 の標準ドライバ、ファームウェア 1.1.2.64。

1. デバイスヒープ (`BO_DEV_HEAP`、64 MiB) を作ってマップする (xclbin の流れと同じ)
2. 入出力の BO (`BO_SHARE`) を作ってマップする
3. `CREATE_HWCTX`: `num_tiles` = 列 × コア行 (8 × 4 = 32)。列の数は ELF の注記 `.note.xrt.configuration` (種類 6、値 8)。QoS は全て 0、`max_opc` 0。**`CONFIG_HWCTX` (CU の設定) は呼ばない**
4. カーネルのインスタンスの制御コード (`.ctrltext.N`) と、それが参照する PDI (`.pdi.N`) を、それぞれヒープ上の `BO_DEV` に写す。PDI の BO のデバイスアドレスは制御コードに書き込む (3.2 の種類 8)
5. 実行ごとに、引数の BO のアドレスを制御コードに書き込み (種類 5)、`SYNC_BO`
6. コマンドの BO (`BO_CMD`): 頭の語 `0x3b00d001` (状態 NEW、語数 13、opcode 22 = `ERT_START_NPU_PREEMPT_ELF`)。続けて CU マスク 1 語 (0)、`amdxdna_cmd_preempt_data` {inst_buf = 制御コードのデバイスアドレス、save_buf 0、restore_buf 0、inst_size、save_size 0、restore_size 0、inst_prop_cnt 0}、最後に 2 語 (最初の引数の BO のアドレス。ドライバは読まない)
7. `EXEC_CMD`: `args` は引数の BO のハンドル (制御コードと PDI の BO は含めない)。完了待ちは xclbin の流れと同じ syncobj

ドライバ側の裏付け: Linux 7.0 は `force_cmdlist` (既定 true) の経路で `aie2_cmdlist_fill_npu_elf` を通り、`EXEC_NPU_TYPE_ELF` のスロットを作る。npu4 のファームウェアの機能表で、メールボックスのプロトコルの主版 7 なら `AIE2_NPU_COMMAND` と `AIE2_PREEMPT` が有効 (`npu_7.sbin`)。

### 3.2 ELF の読み方

ELF32 (OS/ABI 0x45、ABI 版 0x21)。IRON のカーネル名 `main:sequence` は「カーネル `main` のインスタンス `sequence`」。

| 部分 | 意味 |
|---|---|
| `.symtab` の FUNC 記号 | カーネル。`_Z4mainPcPcPc` = `main(char*, char*, char*)` で、引数の数は名前から読む |
| `.symtab` の OBJECT 記号 | インスタンス (`sequence`、`aiecc_empty`)。`st_shndx` はカーネルの記号の番号 |
| COMDAT グループ (`.group.N`) | 署名がインスタンスの記号で、中身が `.ctrltext.N` |
| `.dynsym` | 名前が `"0"`、`"1"`… の記号は引数の番号、`.pdi.N` は PDI。`st_shndx` は書き込み先の `.ctrltext` |
| `.rela.dyn` | `r_offset` は記号の節の中の位置 |

再配置の種類 (XRT 2.21.75 の `xrt_module.cpp` の `patcher::patch_*` を読み、5 と 8 は記録した書き換え後の制御コードと突き合わせて確かめた。`w[i]` は `r_offset` からの i 語目、`A` は書き込むアドレス):

| 種類 | 名前 | 書き方 |
|---|---|---|
| 5 | shim_dma_48 | DMA のバッファ記述子 (TXN の BLOCKWRITE の BD) への 48 ビットのアドレス。`b = ((w[2] & 0xFFFF) << 32 \| w[1]) + A + addend + 0x80000000` (DDR の窓のずれ)、`w[1] = b & 0xFFFFFFFC`、`w[2] = w[2] & 0xFFFF0000 \| b >> 32` |
| 8 | address_64 | `w[0] = lo(A)`、`w[1] = hi(A)`。addend も窓のずれも足さない。PDI の BO のアドレスを TXN の LOADPDI に書く |
| 4 | control_packet_48 | 5 と同じ計算を `w[2]`、`w[3]` (`(w[3] & 0xFFF) << 32`) に |
| 3 | scalar_32 | `mask = st_size`、`w[0] = w[0] &^ mask \| v & mask` |

- ABI 版が 1 の ELF だけは、種類を `addend & 0xF`、addend を `addend >> 4` から読む。IRON の ELF は 0x21 (主版 2) で、種類は `ELF32_R_TYPE`
- `.dynsym` の特別な記号: `scratch-pad-ctrl` (制御コードが読むスクラッチパッドの BO。大きさは `st_size`)、`control-packet` (`.ctrldata`)、`ctrlpkt-pm-*`、`scratch-pad-mem` (プリエンプションの退避)。今回の 2 演算の ELF にはどれもない
- XRT は書き換え前の語を最初に覚えておき、引数が変わるたびにそこから書き直す。引数が変わらなければ、2 回目からの投入はコマンドの頭を書いて `EXEC_CMD` するだけ
- PDI の BO は XRT 2.21.75 ではモジュールごと (全ての run で共有)、XRT の master では run ごと。`BO_DEV` は 32 KiB 境界に置かれる
- コマンドの頭の語は XRT が書いたものを記録すると `0x3b00d001` (上位 4 ビットの種類欄が 3)。ドライバは opcode と語数だけを読む

未確認で、実装の中で確かめるもの:

- ctrl scratchpad (呼び出しごとの値)。IRON の `ParameterScratchpad` は `scratch-pad-ctrl` の BO に位置などを書く。LLM の ELF で、この記号と、制御コードへのアドレスの書き込み (種類 5 か 8) を確かめる
- 1 つの ELF に複数のコマンドがある場合 (`Plan.commands`。ホストの処理を挟むときや、スクラッチパッドの語数が 32 を超えるとき)
- 各 BD の BLOCKWRITE の後ろに TXN の DDR_PATCH (0x81、`arg_idx`、`arg_plus`) が残っている。ELF の流れでファームウェアがこれをどう扱うかはソースからは読めない。XRT と同じものを送るので、同じに振る舞うはず
- 知らない種類の再配置は必ずエラーにする

### 3.3 API

`ProgramSource.ELF` (design.md 5.4 で予約済み) と、インスタンス名つきのカーネル名を受け付ける。

```go
prog, err := kr.Load(ctx, hwnpu.ProgramSource{Path: "decode.elf"}) // 拡張子か ELF の頭で判別
k, err := prog.Kernel("main:sequence")
run, _ := k.NewRun()
run.SetBuffer(0, input).SetBuffer(1, output).SetBuffer(2, scratch)
```

- 引数の BO が変わったときだけ制御コードを書き換える (書き換えの位置は読み込み時に表にしておく)
- 複数の `Program` が 1 つのヒープを分け合う。ヒープは npu4 なら 512 MiB まで広げられる。制御コードは数 MB になりうる (Llama のデコードの全演算)
- 偽のドライバ (`xdnasim`) に ELF のコマンドを足し、制御コードの書き換えを検査する

## 4. 何を IRON の外から渡すか

IRON の `CompiledGraph` が実行時に使うものを、全てファイルに書き出す。

| もの | IRON での出どころ | hwnpu での使い道 |
|---|---|---|
| ELF (版ごと) | `compiled.artifacts.image` (`design.elf`) | 3 節で読み込む |
| 版の一覧 | `Model.shapes()`: デコードのステップ (入力なし) とプロンプトの塊 (`ids` の長さ) | 実行器が呼び分ける |
| 引数の割り当て | `FullELFCallable._argument_index`: `input`、`output`、`feedback`、scratch の bank | BO を作って番号どおりに渡す |
| バッファの配置 | `sequence.layout`: 名前 → (種類、オフセット、バイト数)。scratch は `BANK` (3.5 GiB) ごとの bank | 重み、KV キャッシュ、入出力の場所 |
| 重み | `recording.weights`: 名前 → (配列、handle)。読み込み時に `_copy_in` で配置どおりに写すだけ | 下の重みの像をそのまま bank に読み込む |
| 呼び出しごとの値 | `artifacts.parameter_table` (スクラッチパッドの語) と `Carried` (デバイスで引き継ぐ値) | 位置、行数、開始位置を書く |
| 出力 | `recording.returned` の handle の配置 | ロジット (またはデバイスで引いたトークン) を読む |

重みは「デバイス上の並びのまま」の像として書き出す。IRON は読み込み時に重みを変形する (RoPE の表、並べ替え) が、hwnpu はそれを移植せず、変形後のバイト列を bank ごとのファイルにする。実行時は読み込んで BO に写すだけになる (mmap からの写しで、16.38 の prompt store と同じ扱い)。

## 5. 成果物と変換の道具

```
gemma-4-E2B-it.amd-npu2/
  manifest.json      版、引数、配置、パラメータの表、出力、IRON と mlir-aie の版、元の重みのハッシュ
  decode.elf         デコードの 1 ステップ
  prompt-256.elf     256 トークンのプロンプトの塊 (塊の大きさは版ごと)
  bank0.bin ...      デバイスの並びの重み (bf16)。KV キャッシュなどの状態は含めない (実行時に 0 で作る)
```

- 道具: `tools/amd/iron_export.py` (開発者の機械で一度だけ)。IRON のアプリケーション (`iron.applications.gemma4` など) を、NPU のない機械でも `set_current_device(from_name("npu2", n_cols=8))` でコンパイルし、上の全てを書く。IRON と mlir-aie の版を固定し、manifest に記録する
- 装置ごと: NPU2 (8 列、Strix、Strix Halo、Krackan) は full ELF。NPU1 (Phoenix、Hawk Point、4 列) は full ELF がないので、IRON と同じく「1 ステップごとの xclbin」で書き出す (`--device npu1 --cols 4`、design.md 16.57)。呼び出しごとの値は、mlir-aie が生成する命令列の C++ を命令列プログラムに直して持つ
- `hwnpu-convert` には `-backend amd` を足さない (Go では作れないため)。`hwnpu-chat` のカタログに AMD の成果物の置き場所 (URL とハッシュ) を持つ

## 6. 実行側

### 6.1 IRON の像の実行器 (`internal/llm/iron`)

- `Open(dir)`: manifest を読み、bank と入出力の BO を作り、重みの像を読み込み、版ごとの ELF を読み込む (ハードウェアコンテキストは版ごとに 1 つ。プロンプトとデコードの切り替えは 1 回 0.46 ms で、トークンごとではないので許せる)
- `Prompt(ids, start)` / `Step(id, pos)`: 呼び出しごとの値を書き、1 コマンド (または manifest の数) を投入し、ロジットを読む
- KV キャッシュは IRON の配置どおり bank の中にあり、ホストから読み書きできる BO_SHARE なので、`ExportRows` / `ImportRows` (16.38 の prompt store) はバイト列の写しで実装できる。巻き戻し (`Rewind`) は位置を戻すだけで済む (IRON のキャッシュは位置で書く)

### 6.2 IRON の版の作り方 (エクスポート時に決める)

hwnpu のエンジンは、ロジットを受け取ってホストで標本を引く (温度、思考の上限 16.32、投機的デコード 16.33)。IRON の既定のモデルはデバイスでトークンを引いて次の呼び出しに引き継ぐ (`Sample` と `Carried`) ので、エクスポートでは次の版を作る:

- デコード: 1 トークンを入れて **ロジット** を返す
- 投機的デコードの確認: k トークンを入れて k 行のロジットを返す (プロンプトの塊を全行のロジットを返す形にしたもの)。**測った結果、プロンプトの版の流用では行数によらずデコード 4.5〜8 ステップ分かかり得にならない (design.md 16.54)。デコードと同じ演算で k 行を処理する版が IRON 側に要る**
- プロンプト: 塊を入れて最後の行のロジットを返す

デバイスでのサンプリング (ロジットをホストに読まない分速い) は、まず測ってから。ロジットは Gemma 4 なら 262144 × bf16 = 512 KiB で、50 GB/s の DRAM では読み出しは問題にならない。

### 6.3 `chat.Engine` との境目

`chat.Engine` が `decode.Session` に求めているもの (`Q`、`Qs`、`Cache`、`Pos`、`Reset`、`Mark`、`Save`、`Rewind`、`Step`、`Speculate`、`RowBytes`、`ExportRows`、`ImportRows`、`Resume`、`Migrate`) を interface にし、`decode.Session` と IRON の実行器の両方が満たす形にする。違い:

- IRON は埋め込みをデバイスで引くので、入力は埋め込みではなくトークン ID。interface の入力をトークン ID にし、Core ML 側は今までどおりホストで埋め込みを引く
- KV キャッシュの段階 (16.20、16.37) は IRON にはない (`max_seq_len` で 1 つ)。`Migrate` は何もしない
- 画像と音声: Gemma 4 E2B の塔は IRON の多モーダルのモデル (`gemma4mm`) で書き出して動く (design.md 16.55)。1 回のプロンプトに画像 1 枚と音声 1 本まで

### 6.4 バックエンドの選択

`WithBackend("coreml")` の 6 か所 (`chat/engine.go`、`chat/embed.go`、`systemone/model.go`、`speech/speech.go`、`hwnpu-generate`、`powerbench`) を、モデルの成果物の種類 (Core ML のパッケージ、OpenVINO IR、AMD の manifest) から決める。Intel の設計 (intel-npu-models.md 7.1) と同じ作業なので共通にする。

## 7. モデルの順番

1. **Gemma 4 E2B (テキスト部、bf16)**: IRON の既存のアプリケーション。AMD が毎晩精度を確かめていて、hwnpu も Apple で対応済み (チャットの型、トークナイザー)。実行器と full ELF の正しさを、移植の誤りと切り分けて確かめられる。重みは Hugging Face から認証なしで取れる。見込み 5〜6 トークン/秒
2. **LFM3-1.7B (bf16)** (2026-11-04 済み、[design.md](design.md) の 16.66。hwnpu の `tools/amd/iron_lfm3.py`、8.7 トークン/秒): IRON のアプリケーションを書く (Python)。部品は IRON に揃っている (RMSNorm、GEMV/GEMM、RoPE の部分回転、スライディング窓の注意、depthwise の 1 次元畳み込み、SwiGLU)。`port-architecture` スキルと同じく Hugging Face のロジットと照合する。重み 3.4 GB/トークンで 50 GB/s なら上限 14 トークン/秒、実際は 8〜10 と見込む
3. **4 ビットの重み** (2026-11-05、Gemma 4 で済み、13 節): q4nx (FastFlowLM の形式、32 個ごとの scale と min) の GEMV を hwnpu 側に書き (`tools/amd/iron_q4.py`、`kernels/q4_mv.cc`)、プロンプトは IRON の `flm.DequantBFP` で bfp16 に戻して GEMM

## 8. リスク

| リスク | 手当て |
|---|---|
| コマンドの制限時間。amdxdna は既定 `tdr_timeout_ms=2000` で、約 4 秒動かないコマンドを打ち切る。IRON は 150000 を勧めるが、変えるには root が要る | プロンプトの塊を小さくして 1 コマンドを 2 秒以内に保つ。深い文脈での塊の時間を測ってから塊の大きさを決める |
| IRON と mlir-aie の API が速く変わる | 版を固定し manifest に記録する。成果物を作り直すのは開発者だけ |
| ドライバとファームウェアの版の違い (DKMS 版と標準カーネル、プロトコルの版) | `hwnpuinfo` に機能 (ELF の可否) を出す。ELF が使えない機械では AMD の LLM を使わない |
| メモリの固定。重みと KV キャッシュの BO は実行中に固定される | プロセスの RSS には出ない (12 節)。`ulimit -l` の案内を INSTALL_AMD に書く (この機械は unlimited) |
| 書き出した重みの像が大きい | 元の bf16 と同じ大きさ。4 ビット (7 節の 3) で 4 分の 1 |

## 9. 段階と合格条件

| 段階 | 内容 | 合格条件 |
|---|---|---|
| A1 | full ELF の実行 (3 節) | IRON の ReLU → AXPY の `design.elf` を Go から実行し、CPU の計算と一致。`xdnasim` のテスト |
| A2 | スクラッチパッドと複数コマンド | IRON の section-3 (呼び出しごとの長さ) と section-5 (引き継ぐ値) の例が Go で一致 |
| A3 | エクスポートと実行器 | Gemma 4 E2B のデコードとプロンプトのロジットが、同じ機械の IRON (pyxrt) と許容誤差内で一致。トークン/秒を測る |
| A4 | チャット | `hwnpu-chat` と `hwnpu-agent` が AMD で Gemma 4 E2B を使って応答する (E4B も、16.67) |
| A5 | LFM3 | IRON のアプリケーションが Hugging Face のロジットと一致し、hwnpu-chat で動く (済み、16.66) |
| A5b | LFM3 の多モーダル | LFM3-VL、LFM3-Audio、LFM3-Omni の塔が Hugging Face の出力と一致し (乱数の重み)、本物の重みで画像と音声に答える (済み、16.67) |
| A6 | 4 ビット | q4nx の一般化とその精度 (KL) の測定 |
| A7 | 画像生成 (Anima) | テキスト、DiT、VAE が NPU で動き、diffusers の CPU の参照と同じ絵になる (済み、11 節) |

## 10. 開発環境 (root なし)

この文書の測定と記録に使った手順。利用者には要らない。

- mlir-aie と IRON: Python 3.12 (uv で用意) の venv に、IRON の `requirements.txt` が固定した wheel を入れる。NPU のない機械でもコンパイルできる
- XRT (記録と照合のためだけ): Ubuntu 26.04 universe の `libxrt2`、`libxrt-npu2`、`libxrt-utils`、`python3-xrt` と boost の `.deb` を `dpkg -x` で展開し、`XILINX_XRT=<展開先>/usr` と `LD_LIBRARY_PATH` を設定する。`pyxrt` は Python 3.14 用なので、IRON を 3.14 の venv にも入れる。この状態で IRON の full ELF のグラフが Linux 7.0 の標準ドライバで動いた
- ioctl の記録: `LD_PRELOAD` の小さな C の道具で、amdxdna の ioctl の構造体と、`EXEC_CMD` の時点の全 BO の中身を書き出す

## 11. 画像生成: Anima (`hwnpu-image`)

2026-11-05。Anima (circlestone-labs/Anima) を AMD の NPU だけで動かす。ComfyUI や Diffusers が担う「モデルの実行」の部分を hwnpu が受け持ち、UI やワークフローは持たない。

### 11.1 構成

| 部品 | 中身 | グラフ (`tools/amd/iron_anima.py`) |
|---|---|---|
| テキスト | Qwen3-0.6B (28 層、因果) の最終層の隠れ状態、その後 LLM アダプタ (6 層。T5 のトークンの埋め込みが自分自身と Qwen3 の隠れ状態に注意する) | `text` (1 つ、5 コマンド)。プロンプトは 512 トークンまで |
| DiT | Cosmos-Predict2 2B (28 ブロック、hidden 2048、16 ヘッド)。adaLN-zero の自己注意 (3 軸の RoPE)、テキストへの交差注意、GELU の MLP | `dit_in` (パッチの埋め込みと時刻の埋め込み)、`dit_blocks` (4 ブロック。1 つの ELF に重みのバンクが 7 組。ブロックの射影は 8 ビット (q8)、13 節)、`dit_out` |
| VAE | Qwen-Image の VAE (Wan 2.1 のもの) の復号器を 1 フレームで。因果 3D 畳み込みは時間方向の最後のタップだけ、時間方向の拡大は通らない | `vae/s0`〜`s9` (段ごとに 1 つ) |

ホスト (`internal/imagegen/anima`) がするのは、テンソルの計算でないもの: トークン化 (Qwen3 の BPE と T5 の Unigram、どちらも Go)、埋め込みの行の取り出し、時刻の正弦波、2 × 2 のパッチの並べ替え、flow matching の Euler のステップ (`x += (σ' − σ) v`)、ノイズ、PNG。Anima-Turbo は CFG 1 なので、負のプロンプトも 2 回目の DiT もない。

### 11.2 決めたこと

- **汎用のグラフの書き出し** (`tools/amd/iron_graph.py`、Go の `iron.Graph`): 言語モデルの版 (decode、prompt) を持たない、入力から出力への IRON のグラフ。manifest は `iron_export_lm.py` の版の形をそのまま使う。`variants` で、同じ ELF に別の重みのバンクを持てる (DiT の 7 組。コンパイルは 1 回)。バンクの像は末尾のゼロを切って書く (VAE で 8 GB → 0.2 GB、読み込み 10 秒 → 4 秒)
- **コマンドの大きさ**: 1 コマンドの制御コードはデバイスのヒープ (64 MiB) に、時間は amdxdna の 2 秒に収める。テキストは 7 層ごとに空のホストステップ (`iron.on_host`) でコマンドを分けた (80 MB の ELF が 5 コマンドに)。DiT では同じ分け方で最初のコマンドが `ERT_CMD_STATE_TIMEOUT` で止まった (原因は未調査) ので、4 ブロック (制御コード 20 MB、1.0 秒) を 1 コマンドにし、グループごとにグラフを呼ぶ。VAE は段ごとに別のグラフ
- **交差注意**: IRON の MHA はクエリより短いキーを取らない (クエリがキーの最後の行という前提)。4096 個のパッチを 512 行ずつに分け、それぞれ 512 個のテキストのキーに注意する
- **アダプタのマスク**: 注意のキーを有効な行に切る (`k[:n]`)。出力の有効でない行はホストがゼロにする (アダプタのマスクと 512 行への埋めと同じ)
- **VAE の畳み込み**: チャネルを最後に置いた、ゼロで囲んだ格子 (各行 W + 16 画素、上下に 1 行) にし、3 × 3 の畳み込みは Copy で近傍 9 画素を集めた im2col の行列と GEMM 1 回にする。結果の内側だけを次の格子に Copy で書き戻すので、囲みはずっとゼロのまま (RMS 正規化と SiLU はゼロをゼロのままにする)。行数は 256 の倍数に揃える (GEMM のタイルと、格子全体の写しが奇数の長さに分かれないため)。中段の注意は 1 ヘッド 384 次元で、MHA は `d=384` を取る (16384 行で 423 ms)
- **精度**: IRON の行列積の `MEDIUM` は A と B を bfp16 に丸める。DiT の射影で `HIGH` の 6.5 倍速い (4096 × 2048 × 2048 で 2.1 対 13.3 ms)。テキストは 1 枚に 1 回なので `HIGHEST` にした (時間は同じ 0.5 秒)。DiT と VAE は `MEDIUM` (`--dit-precision high` で選べる)

### 11.3 測定 (Ryzen AI 9 365、1024 × 1024、Anima-Turbo v1.1、8 ステップ)

| 段 | 時間 |
|---|---|
| 読み込み | 0.5 秒 (manifest だけ。各段は走るときに読む、12 節) |
| テキスト | 0.5〜0.9 秒 (読み込みを含む) |
| DiT 1 ステップ (`dit_in`、4 ブロック × 7、`dit_out`) | 7.2 秒 (4 ブロックが 1.0 秒。自己注意 4096 × 4096 が 1 ブロック 0.21 秒で、いちばん重い) |
| VAE (10 段) | 10 秒 (段ごとの読み込みを含む。読み込みなしでは 7 秒、1024 × 1024 の段は 1 つ 0.8〜0.9 秒) |
| 合計 | 約 68 秒 |
| 固定されたメモリの最大 | 3.4 GiB (DiT の重み q8。bf16 では 5.0 GiB、全部を読み込んだままだと 14.2 GiB) |

CPU (diffusers、fp32、9 分 17 秒) を参照に、同じノイズから:

| | 潜在のコサイン | 画像の PSNR |
|---|---|---|
| CPU の bf16 | 0.980 | 21.5 dB |
| NPU (テキスト HIGHEST、DiT MEDIUM、DiT の重み bf16) | 0.966 | 19.1 dB |
| **NPU (同じで DiT の重み q8。既定、13 節)** | **0.961** | **18.6 dB** |
| NPU (DiT の重み q4nx) | 0.777 | 12.9 dB |
| NPU (DiT HIGH。1 ステップ 10.9 秒) | 0.976 | 20.2 dB |
| NPU (テキストも MEDIUM) | 0.938 | 17.2 dB |

ステップ 0 の速度 (同じ入力) はコサイン 0.9997、条件づけは 0.99989 で、差はステップを重ねて軌跡が離れることで広がる。絵は構図も人物も同じで、細部 (口元、背景の枝) が違う。VAE だけを比べると、同じ潜在から NPU と CPU の画像の PSNR は 44.5 dB。

### 11.4 書き出しの手順

`~/models/Anima` に置くもの: 公式の `split_files` (`diffusion_models/anima-turbo-v1.1.safetensors`、`text_encoders/qwen_3_06b_base.safetensors`) と、circlestone-labs/Anima-Base-v1.0-Diffusers の設定、トークナイザー、VAE (`diffusers/{tokenizer,t5_tokenizer,text_encoder,transformer,vae,text_conditioner,scheduler}`、VAE は diffusers の名前の `diffusion_pytorch_model.safetensors`)。

```bash
python tools/amd/iron_anima.py ~/models/Anima ~/models/anima-turbo-1024.amd-npu2 --size 1024   # 約 5 分、4.0 GB (DiT の重み q8)
go run ./cmd/hwnpu-image -model ~/models/anima-turbo-1024.amd-npu2 -prompt "..." -seed 1 -o out.png
```

DiT のブロックの重みは `--dit-attn` と `--dit-mlp` (`q8` が既定、`q4`、`bf16`) で選ぶ (13 節)。確かめ方: `--check` で各グラフに乱数の入力と IRON の参照の出力を書き、`go run ./tools/amd/irongraph DIR` で NPU と比べる。`--random` は乱数の重み (チェックポイントの形だけ使う)。全体は `tools/ref/ref_anima.py` (diffusers で CPU) の `noise` を `-noise` に、`-dump` と `-latents` で条件づけ、各ステップの速度、最後の潜在を書いて比べる。

### 11.5 残り

- 速さ: 自己注意 (head_dim 128 で MHA の B_q が 64 まで) が DiT の 6 割。512 × 512 なら DiT は約 4 分の 1
- CFG (負のプロンプト) は 1 ステップに DiT が 2 回で、Turbo でない Anima (30〜50 ステップ、CFG 4〜5) は 1 枚 10 分を超える見込み
- ComfyUI のノードからの呼び出し (同じグラフを持つサーバー) は未実装
- DiT をホストステップで分けると止まる原因

## 12. メモリ: 固定されるバッファを減らす (2026-11-05)

IRON の arena (bank) とグラフのバッファは amdxdna の BO_SHARE (shmem) で、作ったときから全ページが固定される。プロセスの RSS には出ず (Gemma 4 E2B で 0.43 GB)、`/proc/meminfo` の `Shmem` と `Unevictable` が増え、`MemAvailable` がその分減る (E2B で 9.3 GB)。スワップにも出ない。測るときは `Unevictable` の増え方を見る。

中身 (manifest の配置から):

| | 計 | 内訳 |
|---|---|---|
| Gemma 4 E2B | 9.0 GB | 層ごとの埋め込みの表 (PLE) 4.4 GB、MLP 2.9 GB、embed_tokens (出力の行列と共有) 0.75 GB、KV キャッシュ 72 MB |
| Gemma 4 E4B | 14.3 GB | PLE 5.25 GB、MLP 6.2 GB、embed 1.25 GB |
| LFM3-1.7B | 3.7 GB | ほぼ重み。KV キャッシュ (12288) 264 MB。`--q4` で 1.64 GiB (13 節) |
| Anima (1024 × 1024) | 14.2 GB | テキスト 1.05、DiT 4.95 (重み 3.7、7 組それぞれに作業領域 160 MB)、VAE 8.2 (重み 0.16、残りは特徴マップ) |

したこと:

- **Anima: 段ごとに読み込む** (`internal/imagegen/anima`): テキスト、DiT、VAE の 10 段は同時には走らないので、走るときに読み込み、終わったら放す。最大は DiT の 5.0 GiB (DiT の重みを q8 にして 3.4 GiB、13 節)。絵はビット単位で同じ。読み込みはページキャッシュから 1 枚あたり数秒 (VAE の段は特徴マップが大半で、像のファイルは 0.16 GB)
- **Gemma 4: PLE をホストに** (`tools/amd/iron_hostple.py`、`iron.HostTable`): PLE は 1 トークンに 1 行 (E2B で 35 層 × 256 × bf16 = 17.5 KiB) しか読まないのに、表全体が固定されていた。表はファイル `ple.bin` (語彙 × 層 × 256、スケールと目印の行の置き換えは IRON の `load_state_dict` と同じ) のまま置き、デバイスには `hwnpu_ple_rows` (層 × 塊の行 × 256、E2B で 18 MB) だけを持つ。ホストは呼び出しの前にトークンの行を読み (`ReadAt`、ページキャッシュ) その順に書き、層はその行を順に読む。gather がなくなるだけで計算は同じ: E2B の生成は 19 トークンの質問と 2483 トークン (塊 3 つ) の要約で古い書き出しと一字一句同じ、速さも同じ (6.1 トークン/秒、塊は 3072 から 6.29 秒 / 古いもの 6.34 秒)
  - コマンドの分け方: IRON は 1 コマンドのスクラッチパッドの語が 32 を超えるところでコマンドを切る。PLE の gather は語を多く使っていたのでプロンプトは 15 コマンドだったが、なくすと 7 コマンドになり、1 つが 3072 から 3.7 秒 (amdxdna の打ち切りに近い) で、制御コードもデコードのものと並んでヒープに入らなかった (`ENOSPC`)。書き出しのときだけ上限を 24 語にして 50 コマンドにした (最長 381 ms)。ホストステップでは切れない (デコードが送り込むプロンプトの版はホストステップを持てない)
  - Go 側はヒープが足りないとき (`ENOSPC`) デコードのランを放してやり直す (IRON と同じ)
  - E4B: 863 トークンの要約で古い書き出しと一字一句同じ、デコード 3.7 トークン/秒も同じ。2483 トークン (塊 3 つ) は古い書き出しでは 2 つめの塊の最初のコマンドが 61 秒止まって `ERT_CMD_STATE_TIMEOUT` になる (2 回とも。原因は調べていない) が、新しい書き出し (62 コマンド) は通る。画像と音声の E2B (`gemma4mm`) は、画像 1 枚の説明と次の質問の答えが古い書き出しと同じ
  - `--device-ple` で前の書き出し (表をデバイスに) にできる。`host_tables` のない古い書き出しもそのまま動く

| | 前 | 後 |
|---|---|---|
| Gemma 4 E2B | 9.08 GB | 4.60 GB |
| Gemma 4 E4B | 14.7 GB | 9.25 GB |
| Gemma 4 E2B (画像と音声) | 10.1 GB | 5.6 GB |
| Anima | 14.2 GB | 5.0 GB |

残り (効く順):

- **4 ビットの重み**: 13 節で済み
- **常駐するサーバーで待つ間はモデルを放す** (2026-11-06 済み、`-idle`、既定 5 分): `iron.Model.Release` / `Reload` (マニフェストとホストの表は残し、デバイスのものを全部放す) と `chat.Engine` の時計 (`internal/chat/idle.go`)。放す前に会話をホストに写し (`putAside`、256 トークン以上)、次の要求で読み直してそこから続ける。E2B の 4 ビットで 1775 MiB → 0、読み直し 0.8 秒、830 トークンを写しから戻した。画像と音声の E2B で 2818 MiB → 0、読み直し 2.0 秒、画像の後の質問に正しく答えた。Neural Engine と Intel の NPU は読み直しに数十秒以上かかるので放さない
- **塔を使うときだけ読む**: E2B の画像と音声の書き出しは塔の分 (約 1 GB) が文だけの会話でも固定される
- **DiT の作業領域を 7 組で 1 つに**: 約 1 GB。重みと作業領域を別のバンクにする書き出しが要る
- **VAE をタイルに分ける**: VAE の段は特徴マップが大半。分ければ段の最大 1.2 GB が数分の 1 に (今は DiT が最大なので、DiT を減らしてから)
- **PLaMo の embed_tokens をホストに**: 13 節の PLaMo 3 の `--q4` で済み (0.55 GB)

## 13. 4 ビットと 8 ビットの重み (q4nx、q8) (2026-11-05)

`iron_export_lm.py gemma4 ... --q4` で、Gemma 4 の MLP (gate、up、down) と出力の行列を 4 ビット (q4nx)、注意の射影 (q、k、v、o) を 8 ビット (q8) にする。`--attn q4|bf16` で注意を 4 ビットか bf16 にもできる。

形式 (どちらもブロックは出力 32 × 入力 256、並びは `flm.DequantBFP` が読む順で、出力 64 ずつの列ブロックが連続。デコードとプロンプトは同じバイト列を読む):

- **q4nx**: FastFlowLM の Gemma 4 の形式 (mlir-aie の `flm_gemma4/q4nx.h`、IRON の `iron.operators.flm.q4nx`)。入力 32 個ごとに bf16 の scale と min、重みは `min + scale × code` (code は 0〜15)。1 重み 5 ビット (bf16 の 31 %)
- **q8**: hwnpu の形式。入力 32 個ごとに bf16 の scale、重みは `scale × code` (code は -127〜127、int8)。並びは q4nx と同じ (16 行の帯ごとに列優先)。1 重み 8.5 ビット (bf16 の 53 %)

演算 (`tools/amd/iron_q4.py`、カーネルは `tools/amd/kernels/q4_mv.cc` と `q8_mv.cc`):

- **デコード** (1 行): `QGEMV`。1 列に 2 コア。メモリタイルが列ブロックの 2 ブロック (出力 0〜31 と 32〜63) を 2 コアに分け、x を両方に配り、出力をつなぐ。q4nx は FastFlowLM の `q4nx_accumulate` (32 列の部分和を bf16 にしてから scale を掛け、min は x の 32 個ごとの和との積 1 回)。q8 は同じ形だが、部分和を bf16 1 つに丸めると 8 ビットの精度が消える (コサイン 0.9998) ので、上位と残りの 2 つの bf16 に分けて scale を掛ける (0.999992)
- **プロンプト** (複数行): q4nx は IRON の `flm.DequantBFP` でデバイス上で bfp16 に戻し、`flm.ProjectionGEMM` (bfp16 の B)。Gemma 4 はもともと `Precision.MEDIUM` (bfp16) で掛けているので、計算の精度は量子化の分だけ変わる。q8 は `QDequant` (hwnpu の演算。int8 を bf16 にして scale を掛け、16 行 × 8 列ずつ転置して書く) で bf16 の (N, K) に戻し、IRON の `F.linear` に渡す。`QDequant` は q4nx も戻せる (`iron_q4.PREFILL["q4"] = "bf16"`、Anima の DiT。下の「コアの丸めの向き」)
- **量子化** (q4nx): 32 個ごとに、[min, max] を少し縮めた候補 9 つから、code を決めたあと scale と min を最小 2 乗で合わせ直し、誤差の最も小さいものを選ぶ (llama.cpp の make_qkx と同じ考え)。単純な min-max より重みの誤差が 1 割小さい (E2B の 3 層目の gate_proj で 8.2 % → 7.3 %)。q8 は 32 個の最大の絶対値 / 127
- **トークンの埋め込み**: 出力の行列と共有 (tied) なので、出力側を 4 ビットにすると bf16 の行がなくなる。PLE と同じくホストの表 (`embed.bin`、12 節) にし、デバイスには塊の行数分 (3 MB) だけ置く
- IRON への手当て (hwnpu 側で、IRON は変えない): `F.linear` を差し替え、量子化した重みなら `q_linear` へ。`Operand.fits` が bfp16 の型で落ちる (bfp16 の B に呼び出しごとの行数があるとき) ので、バイトの並びで判定する。プロンプトの版の入力 `ids` は表を引かなくなっても読まれている必要があり、全体を 1 回写す (4 KiB)

単体 (`python tools/amd/iron_q4.py check OUT --fmt q4|q8 --n N --k K [--rows R] [--cores 1|2]`、`go run ./tools/amd/irongraph OUT`): NPU の出力と、量子化した重みでの float64 の積のコサインが q4nx で 0.99994 以上、q8 で 0.99999 (プロンプトの経路は bfp16 の GEMM で 0.99995)。速さ (1 行):

| 形 | bf16 の GEMV | q4nx 1 コア/列 | q4nx 2 コア/列 | q8 2 コア/列 |
|---|---|---|---|---|
| 6144 × 1536 (E2B の gate) | 0.421 ms | 0.254 ms | 0.171 ms | |
| 1536 × 12288 (down、double-wide) | 0.729 ms | 0.414 ms | 0.286 ms | |
| 262144 × 1536 (出力の行列) | 14.7 ms (55 GB/s) | 7.06 ms (36 GB/s) | 4.49 ms (56 GB/s) | |
| 2048 × 1536 (E2B の q) | | | | 0.150 ms |
| 1536 × 4096 (大域の層の o) | | | | 0.198 ms |

1 コア/列では計算で頭打ちだった。2 コア/列で bf16 の GEMV と同じ 55 GB/s に届き、ここが NPU から DRAM を読む上限に見える。q8 の戻し (プロンプト) は、1 要素ずつのスカラー版では 2048 × 1536 で 39 ms、ベクトル版で 1.2 ms (GEMM 込み)。

Gemma 4 E2B (Ryzen AI 9 365):

| | bf16 (元) | bf16 + PLE をホスト | q4 + 注意 bf16 | **q4 + 注意 q8 (既定)** | q4 + 注意 q4 |
|---|---|---|---|---|---|
| 固定されたメモリ | 9.08 GB | 4.60 GB | 2.02 GB | **1.78 GB** | 1.66 GB |
| デコード | 6.1 トークン/秒 | 6.1 | 7.7 | **7.9** | 7.7〜7.9 |
| 1 トークンの電力量 | 2.31 J | 2.20 J | 1.47 J | **1.46 J** | 1.44〜1.50 J |
| bf16 との KL (平均 / 中央値 / 95 %) | | 0 | 0.091 / 0.004 / 0.44 | **0.097 / 0.004 / 0.49** | 0.229 / 0.010 / 0.98 |
| 最上位の一致 | | 1 | 89.7 % | **90.0 %** | 84.8 % |

Gemma 4 E4B (`--q4`、注意 q8): 固定されたメモリ 14.7 GB (元) → 9.25 GB (PLE をホスト) → 3.53 GB (注意 bf16 では 4.30 GB)、デコード 3.7 → 5.6 トークン/秒、bf16 との KL 0.049 / 0.006 / 0.20 (注意 bf16 で 0.047)、最上位の一致 93.0 % (E2B より 4 ビットに強い)。画像と音声の E2B (`gemma4mm --q4`、塔は bf16): 10.1 → 5.6 → 2.82 GB、画像 1 枚の説明と次の質問に正しく答える。hwnpu-chat で E2B は「日本の首都は東京です。」「391」。

KL は、英語 (README) と日本語 (この文書) の 656 トークンの文を 256 トークンのプロンプトとして読み、続く 400 位置を 1 つずつ与えたときのロジット (`ironlm -forced`)。注意の射影は bf16 で 0.53 GB だが 4 ビットでは誤差の大半で、8 ビットならほぼ bf16 と同じ。出力の行列は 4 ビットでもほとんど効かない (出力だけ bf16 で 0.2245)。

デコードの残り約 125 ms のうち、重みを読む GEMV は 30 ms 余り。残りは注意 (IRON は 4096 の文脈全体を読む) と小さな演算の積み重ねで、ここからの速さは GEMV の外にある。

### PLaMo 3 Translate 2B (`iron_export_lm.py plamo3 --q4`、2026-11-06)

Gemma 4 と同じ形式で、MLP (`gate_up_proj`、`down_proj`) と出力の行列 (`lm_head`) を q4nx、注意の射影 (`qkv_proj`、`o_proj`) を q8 (`--attn q4|bf16` も同じ) にする。埋め込み (`embed_tokens`、0.55 GB) は出力の行列と別の重みで、PLE と同じくホストの表 (`embed.bin`、12 節) にする (埋め込みの尺度はない)。形はどれも出力 64 × 入力 512 で割り切れる (隠れ層 2560、中間 10240、q 3072、k と v 512、語彙 107520)。

- まとめた重み (`qkv_proj` の q、k、v の行、`gate_up_proj` の gate、up の行) は 1 つの重みのまま量子化する。列ブロック (出力 64) が連続なので、行の範囲は重みのバイトの範囲になる (`iron_q4.q_rows`)。`Linear.split` を差し替え、プロンプトは範囲ごとに戻して GEMM、デコードは重み全体で `QGEMV` を 1 回して出力を切る (分けた 3 回と 2 回より、単体で qkv 0.61 → 0.44 ms、gate_up 1.12 → 1.02 ms、デコード全体で 6 %)
- プロンプトは 36 コマンド (bf16 は 3)。gather がなくなるので Gemma 4 と同じく `SCRATCHPAD_WORDS` を 24 にした。速さは bf16 と同じ

Ryzen AI 9 365、文脈 8192、塊 1024 (別のセッションの PyTorch が 8 コアと DRAM を使っている間に測った。3 回ずつ交互に走らせた最良。単体の q4nx の GEMV も 12 節の表より 1.6 倍遅く、静かな機械ではもっと速いはず):

| | bf16 | **q4 + 注意 q8** |
|---|---|---|
| 固定されたメモリ (バンク) | 5.96 GiB (3.5 + 2.3) | **2.16 GiB** (埋め込み 0.51 GiB はホストのファイル) |
| 書き出し | 5.9 GB | 2.8 GB |
| デコード (位置 2600 付近) | 7.3 トークン/秒 | **11.6** |
| 2596 トークンのプロンプト | 11.2 秒 | 11.8 秒 |
| Hugging Face (fp32) との照合 (2596 トークン + 8 ステップ) | 9/9、コサイン 0.9996 以上 | 9/9、コサイン 0.991 以上 |

精度 (`ironlm -forced`、bf16 の NPU のロジットとの比較):

| 文 | 最上位の一致 | KL (平均 / 中央値 / 95 %) | bf16 の訳文のパープレキシティ (bf16 → q4) |
|---|---|---|---|
| 英 → 日の訳文 20 文 (PFML3 の対訳の履歴、訳文の 268 トークン) | 98.5 % | 0.0089 / 0.0001 / 0.047 | 1.24 → 1.23 |
| 日 → 英の訳文 20 文 (208 トークン) | 97.6 % | 0.0068 / 0.0000 / 0.040 | 1.30 → 1.32 |
| 同じ会話の原文の側 (442 + 382 トークン、モデルが予測するものではない) | 85〜88 % | 0.054〜0.099 | |
| README (日英の混じった 400 位置、Gemma 4 と同じ文) | 89.5 % | 0.128 / 0.029 / 0.30 | |

訳文 44 文 (英 → 日 20、日 → 英 20、中国語と韓国語 4、貪欲) のうち 36 文が bf16 と一字一句同じ。違う 8 文はどれも正しい訳の言い換え (「数千人もの人々が」→「数千人が」、「I'm already booked」→「I already have plans」など)。訳文の側の KL は Gemma 4 E2B の README での 0.097 の 10 分の 1 で、翻訳は 4 ビットに強い。PLaMo 3 は重みを読む時間がデコードの大半 (bf16 で 5.8 GB/トークン) なので、Gemma 4 E2B (6.1 → 7.9) より速さが伸びた。

### LFM3 とその多モーダル (`iron_export_lm.py lfm3 / lfm3vl / lfm3audio / lfm3omni --q4`、2026-11-06)

MLP (`w1`、`w3`、`w2`) と出力の行列を q4nx、全注意とスライディング窓の層の射影 (`q`、`k`、`v`、`o` と、ヘッドごとのゲートを列に繰り返した重み) と短い畳み込みの射影 (`in_proj`、`out_proj`) を q8 にする (`--attn`、`--conv` で q4 か bf16 にもできる)。出力の行列は埋め込みと共有なので、Gemma 4 と同じく埋め込みはホストの表 (`embed.bin`、0.49 GiB) にする (尺度はない)。`in_proj` は B、C、x の 3 つの行の範囲を 1 つの重みのまま量子化し、PLaMo 3 と同じ `Linear.split` の差し替えで切る。形はどれも出力 64 × 入力 512 で割り切れる (隠れ層 2048、中間 7680、k と v 512、語彙 128000)。LFM3-VL、Audio、Omni は本体 (`iron_lfm3.Lfm3`) が同じなので同じ設定がそのまま効き、塔は bf16 のまま。プロンプトは 18 コマンド (bf16 は 12。`SCRATCHPAD_WORDS` 24)。書き出しは約 3 分。

どこを何ビットにするか (CPU、Hugging Face の fp32 で重みだけを量子化して戻した、bf16 の重みとの KL。英語と日本語の 656 トークンの文の 400 位置。Gemma 4 と同じ文):

| 4 ビットにしたもの (ほかは bf16) | KL (平均 / 95 %) | 最上位の一致 |
|---|---|---|
| 出力の行列だけ | 0.005 / 0.013 | 97 % |
| 短い畳み込みだけ | 0.009 / 0.028 | 96.5 % |
| 注意だけ | 0.015 / 0.054 | 93 % |
| MLP だけ | 0.031 / 0.096 | 93 % |
| **MLP と出力 q4、注意と畳み込み q8 (既定)** | **0.035 / 0.101** | 94 % |
| 同じで畳み込みも q4 | 0.046 / 0.153 | 92 % |
| すべて q4 | 0.060 / 0.191 | 90 % |
| すべて q8 | 0.0004 / 0.002 | 99 % |

文の先頭に `<|startoftext|>` (124894) を置かないと同じ既定で 0.139 になり、0 層目の MLP だけで 0.098 を占めた (BOS がないと最初のトークンが注意の溜まり場になって 0 層目の活性が外れ値になるらしい)。hwnpu はチャットの型で必ず置くので BOS ありが実際の値。以前の CPU での LFM3 の 4 ビットの測定 (`hwnpu-check`、24 トークン) も BOS の有無を確かめてから比べること。

Ryzen AI 9 365、文脈 12288、塊 2048 (ほかに NPU を使うものがない状態):

| | bf16 | **q4 + 注意と畳み込み q8** |
|---|---|---|
| 固定されたメモリ (`Unevictable` の増え方) | 3.75 GiB | **1.64 GiB** |
| 書き出し | 3.6 GB | 2.2 GB (うち `embed.bin` 0.5 GB) |
| デコード (位置 2048 / 12287) | 117 / 131 ms (8.7 トークン/秒) | **92 / 103 ms (11.2)** |
| 1 トークンの電力量 (パッケージ、150 トークン) | 1.34 J (11.6 W) | **1.05 J** (11.9 W) |
| 2048 トークンの塊 (位置 0 / 10240 から) | 2.29 / 12.44 秒 | 2.38 / 12.53 秒 |
| Hugging Face (fp32) との KL、最上位の一致 (`ironlm -forced -bos 124894`、400 位置) | 0.0007、98.8 % | 0.036、92.7 % |
| NPU の bf16 との KL (平均 / 中央値 / 95 %) | | 0.036 / 0.026 / 0.11、92.0 % |

NPU の q4 の KL は CPU の見積もり (0.035) と同じで、NPU の計算 (q4nx の GEMV、プロンプトの bfp16) による上乗せは見えない。Gemma 4 E2B の既定 (0.097) の 3 分の 1 で、LFM3 は 4 ビットに強い。デコードの残り 90 ms のうち重み (約 1.3 GB) を読むのは 25 ms ほどで、残りは注意 (文脈の全体を読む) と小さな演算。

実機での答え (貪欲、bf16 と q4 で同じ問い):

- LFM3-1.7B: 「日本の首都は」に東京、17 × 23 に 391、富士山に 3,776 m、空が青い理由を 3 文で。どれも正しく、言い回しが少し違うだけ。hwnpu-agent は 16.66 の円の面積の課題で、ファイルを読み、`go test` を走らせ、`math.Pi * c.R * c.R` に直して通ることを確かめた (11 トークン/秒)
- LFM3-VL-2B: 赤い円と青い四角の画像に「赤い楕円と青い四角形が写っています。」、「赤い図形は左右どちら」に「左」(bf16 と同じ)。デコード 8.7 → 11.3 トークン/秒、1.35 → 1.03 J/トークン。固定 4.12 → 2.00 GiB
- LFM3-Audio-2B: `Front_Left.wav` に「front, left.」、`Rear_Right.wav` に「Rear, right」、speech-dispatcher の案内 (約 10 秒) の書き起こしは bf16 と読点 1 つと「/var/log」→「/var/logs」の 1 か所だけ違う。固定 4.14 → 2.03 GiB
- LFM3-Omni-2B: 画像に「赤い円と青い四角」、2 つの音声に「Front, left.」「Rear, right.」(bf16 と同じ)。固定 4.51 → 2.39 GiB

### Anima の DiT (`iron_anima.py --dit-attn`、`--dit-mlp`、2026-11-06)

DiT の 28 ブロックの射影 (自己注意と交差注意の q、k、v、o、MLP の 2 つ。DiT の重みの 9 割) を q8 か q4nx にする。adaLN の変調 (2 つ目の線形の入力が 256 で、q4nx のブロックにならない)、埋め込み、最後の層は bf16。既定は両方 q8。4 ブロックのバンク (重みと作業の領域) は 689 → 464 MiB (q8)、359 MiB (q4nx) で、7 組。

Ryzen AI 9 365、1024 × 1024、8 ステップ、同じノイズとプロンプト (ほかに NPU を使うものがない状態):

| DiT の重み | bf16 | **q8 (既定)** | q4nx |
|---|---|---|---|
| 固定されたメモリの最大 (`Unevictable` の増え方) | 5020 MiB | **3445 MiB** | 2710 MiB |
| 書き出し (テキストと VAE を含む) | 5.7 GB | **4.0 GB** | 3.2 GB |
| デノイズ 8 ステップ / 全体 (4 つのプロンプトの平均、ページキャッシュに載せて) | 56.3 / 67.3 秒 | **57.5 / 68.3 秒** | 57.3 / 68.0 秒 |
| 1 枚の電力量 (amdgpu の PPT、パッケージ全体) | 約 1050 J | **約 1050 J** | 約 1050 J |
| diffusers の fp32 (CPU) との潜在のコサイン / 画像の PSNR (11 節のプロンプト) | 0.966 / 19.1 dB | **0.961 / 18.6 dB** | 0.777 / 12.9 dB |
| 同じ (猫と窓辺、seed 13) | 0.974 / 21.9 dB | **0.970 / 21.8 dB** | 0.855 / 14.9 dB |
| 同じ (11 節のプロンプト、seed 10) | 0.824 / 13.2 dB | **0.823 / 13.2 dB** | 0.667 / 10.4 dB |

q8 は bf16 とほぼ同じ絵になる (構図、顔、背景の枝まで同じ)。重みを毎ステップ戻す分、デノイズが 2 % (1 枚 1 秒) 遅い。fp32 からの離れ方は NPU の bf16 (`MEDIUM` の計算) が大半で、seed 10 では bf16 も q8 も同じだけ離れる (0.824 と 0.823)。ただ軌跡が分かれやすいプロンプトと seed では、q8 の小さな差でも構図が変わることがある (猫と窓辺、hwnpu の seed 4: bf16 と q8 の PSNR 15.4 dB、手前の家具が違う)。q4nx は顔つきと手の位置が変わり、首飾りが現れる。CPU (fp32) で重みだけを q4nx にして戻しても 0.753 / 12.6 dB で、NPU の q4nx の差はほぼ量子化そのもの。8 ステップに蒸留した DiT は 4 ビットの丸めを絵まで運ぶ。

どこが 4 ビットに弱いか (CPU、fp32、参照の軌跡のステップ 0 と 4 での速度の相対誤差の平均。重みだけを量子化して戻す):

| 量子化したもの (ほかは bf16) | 速度の誤差 |
|---|---|
| すべて q8 | 0.010 |
| adaLN だけ q4nx | 0.0004 |
| 交差注意だけ q4nx | 0.024 |
| MLP の 2 つ目だけ q4nx | 0.061 |
| MLP の 1 つ目だけ q4nx | 0.064 |
| 自己注意だけ q4nx | 0.088 |
| すべて q4nx | 0.114 |
| すべて q4nx を GPTQ で (4 つのプロンプト × 4 ステップのヘッシアン、同じ形式) | 0.070 |
| GPTQ の q4nx、自己注意と MLP の 1 つ目は q8 (約 6.7 ビット) | 0.033 |

GPTQ は q4nx の誤差を 4 割減らし、CPU の絵は 0.753 → 0.868 / 14.8 dB まで戻るが、q8 (0.010) にはまだ遠く細部が変わる。q4nx から q8 で増える固定は 735 MiB なので、既定は q8 にした。GPTQ の符号を書き出しに読む道は作っていない。

### コアの丸めの向き (2026-11-06)

AIE のコアは丸めのモードが切り捨て (負の無限大へ) のまま始まる。float から bf16 や bfp16 に狭めるたびに値が少しずつ下に寄り、入力 K 個にわたって足されると、入力に平均があるとき (Anima の DiT の活性) 出力の 1 割近い偏りになる。平均 0 の乱数の単体の試験では打ち消し合って見えない。IRON の `flm.DequantBFP` の README にも切り捨てとある。

- hwnpu のカーネル (`q4_mv.cc`、`q8_mv.cc`) は呼ばれるたびに最初に最近接偶数 (`aie::set_rounding(conv_even)`) にする。1 命令で、速さは変わらない
- q4nx をプロンプトで bf16 に戻す `q4_dequant` (`QDequant` の q4nx 版) を足した。IRON の `DequantBFP` は変えない。量子化した DiT も `--check` の参照 (IRON の `F.linear` に戻した重み) と比べられる (512 × 512 の 4 ブロック、q8 でコサイン 0.999987)
- 単体 (`iron_q4.py check --x-mean 1`、2048 × 2048、4096 行): `DequantBFP` の経路はコサイン 0.968、`QDequant` (bf16) の経路は 0.99994、q8 は 0.99996
- Anima の DiT (diffusers の fp32 とのステップ 0 の速度の相対誤差): NPU の q8 は 0.088 → 0.031、q4nx は 0.244 → 0.087。NPU の bf16 は 0.025、CPU で量子化だけなら q8 0.007、q4nx 0.092 で、直したあとの差は量子化と NPU の bf16 の計算の分だけ。絵は上の表のとおり
- 言語モデル (正規化した、平均がほぼ 0 の入力) では差がほとんど見えない: 書き出し直した Gemma 4 E2B の q4 の NPU の bf16 との KL は 0.097 → 0.093 (最上位の一致 90.0 %、同じ)、LFM3-1.7B は 0.0364 → 0.0362 (92.0 → 93.0 %)。プロンプトの q4nx も `QDequant` (bf16) にすると KL は同じ (0.094、0.0368) で塊が 5〜9 % 遅くなる (LFM3 の 2048 で 628 → 684 ms、E2B の 1024 で 940 → 988 ms) ので、言語モデルの q4nx は `DequantBFP` のまま (`PREFILL["q4"] = "bfp16"` が既定)。デコード (`QGEMV`) は丸めだけが変わり、速さは同じ (LFM3 88 ms、E2B 128 ms)。今ある書き出しはそのまま動き、書き出し直すと丸めが直る

ほかのビット数: AMD の NPU (AIE2P) がそのまま掛けられるのは bf16、bfp16 (8 値で指数 1 つ、約 9 ビット)、int8 など。4 ビット、6 ビット、8 ビットのブロック形式は、コアの中で bf16 に戻しながら掛ける (q4nx、q8 と同じ) ので、形式ごとに GEMV のカーネルと、プロンプト用に bf16 か bfp16 へ戻す演算が要る。q8 は q4nx の形を元にしたので、6 ビットも同じ並びで足せる。
