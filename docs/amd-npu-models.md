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
2. **LFM3-1.7B (bf16)**: IRON のアプリケーションを書く (Python)。部品は IRON に揃っている (RMSNorm、GEMV/GEMM、RoPE の部分回転、スライディング窓の注意、depthwise の 1 次元畳み込み、SwiGLU)。`port-architecture` スキルと同じく Hugging Face のロジットと照合する。重み 3.4 GB/トークンで 50 GB/s なら上限 14 トークン/秒、実際は 8〜10 と見込む
3. **4 ビットの重み**: mlir-aie の `q4nx` (FastFlowLM の Gemma 4 から取り出した形式、32 個ごとの scale と min) は今 Gemma 4 E2B の形に固定されている。形を一般化した q4nx の GEMV を IRON に足す。デコードはおおむね 4 倍になる見込み

## 8. リスク

| リスク | 手当て |
|---|---|
| コマンドの制限時間。amdxdna は既定 `tdr_timeout_ms=2000` で、約 4 秒動かないコマンドを打ち切る。IRON は 150000 を勧めるが、変えるには root が要る | プロンプトの塊を小さくして 1 コマンドを 2 秒以内に保つ。深い文脈での塊の時間を測ってから塊の大きさを決める |
| IRON と mlir-aie の API が速く変わる | 版を固定し manifest に記録する。成果物を作り直すのは開発者だけ |
| ドライバとファームウェアの版の違い (DKMS 版と標準カーネル、プロトコルの版) | `hwnpuinfo` に機能 (ELF の可否) を出す。ELF が使えない機械では AMD の LLM を使わない |
| メモリの固定。重みと KV キャッシュの BO は実行中に固定される | Gemma 4 E2B は bf16 で数 GB。`ulimit -l` の案内を INSTALL_AMD に書く (この機械は unlimited) |
| 書き出した重みの像が大きい | 元の bf16 と同じ大きさ。4 ビット (7 節の 3) で 4 分の 1 |

## 9. 段階と合格条件

| 段階 | 内容 | 合格条件 |
|---|---|---|
| A1 | full ELF の実行 (3 節) | IRON の ReLU → AXPY の `design.elf` を Go から実行し、CPU の計算と一致。`xdnasim` のテスト |
| A2 | スクラッチパッドと複数コマンド | IRON の section-3 (呼び出しごとの長さ) と section-5 (引き継ぐ値) の例が Go で一致 |
| A3 | エクスポートと実行器 | Gemma 4 E2B のデコードとプロンプトのロジットが、同じ機械の IRON (pyxrt) と許容誤差内で一致。トークン/秒を測る |
| A4 | チャット | `hwnpu-chat` と `hwnpu-agent` が AMD で Gemma 4 E2B を使って応答する |
| A5 | LFM3 | IRON のアプリケーションが Hugging Face のロジットと一致し、hwnpu-chat で動く |
| A6 | 4 ビット | q4nx の一般化とその精度 (KL) の測定 |

## 10. 開発環境 (root なし)

この文書の測定と記録に使った手順。利用者には要らない。

- mlir-aie と IRON: Python 3.12 (uv で用意) の venv に、IRON の `requirements.txt` が固定した wheel を入れる。NPU のない機械でもコンパイルできる
- XRT (記録と照合のためだけ): Ubuntu 26.04 universe の `libxrt2`、`libxrt-npu2`、`libxrt-utils`、`python3-xrt` と boost の `.deb` を `dpkg -x` で展開し、`XILINX_XRT=<展開先>/usr` と `LD_LIBRARY_PATH` を設定する。`pyxrt` は Python 3.14 用なので、IRON を 3.14 の venv にも入れる。この状態で IRON の full ELF のグラフが Linux 7.0 の標準ドライバで動いた
- ioctl の記録: `LD_PRELOAD` の小さな C の道具で、amdxdna の ioctl の構造体と、`EXEC_CMD` の時点の全 BO の中身を書き出す
