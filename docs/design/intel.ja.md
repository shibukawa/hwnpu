# Intel の NPU

[English](intel.md)

Intel の NPU を Level Zero とその graph 拡張で使う (`internal/ze`、`internal/backend/levelzero`)。プログラムは Go が OpenVINO IR で書く (`internal/ovlower`、`internal/ovir`)。モデルごとの Intel の計画と演算の表は [../intel-npu-models.md](../intel-npu-models.md)。

## 現状

- 実機: HP EliteBook 630 G11、Core Ultra 5 135U (Meteor Lake、NPU 3720)、32 GB、Windows 11 25H2、NPU ドライバ 32.0.100.5540 (Level Zero 1.16、`ZE_extension_graph` 1.18、コンパイラ 8.3)。`tools/intel/hwnpu-intel-check.ps1` (standard / full の段階) で回す。
- この機械で通ったもの (全部 NPU): `report`、LFM3-1.7B (`--paged=-1 --context 4096` と、既定の文脈 262144 の 7 部分。下書きつきで 17〜25 tok/s)、考える答え、`hwnpu-chat agent` の read と write、PLaMo 3 Translate 0.8B (ja→en と en→ja、25〜31 tok/s)。
- それ以外 (画像・音・絵のコマンド、Gemma 4、Qwen3、Qwen3.5、Whisper) は OpenVINO の CPU でだけ確かめた (下)。NPU のコンパイラがそのグラフを受け付けるか、速さ、メモリは未解決。

## ライブラリの読み込み

1. `HWNPU_ZE_LOADER`、2. 通常の検索パスの `libze_loader.so.1` / `ze_loader.dll`、3. 既知の場所 (Linux `/usr/lib/x86_64-linux-gnu`、`/usr/local/lib`。Windows `%SystemRoot%\System32`、oneAPI の `bin`)。失敗したら `ErrLibrary` に試したパスを並べる。`internal/dl` が purego (Unix) か `syscall.LoadLibrary` (Windows) で読む。cgo は使わない。

## 初期化とデバイス

```
zeInit(ZE_INIT_FLAG_VPU_ONLY)
zeDriverGet → zeDeviceGet → properties.type == ZE_DEVICE_TYPE_VPU
zeDriverGetExtensionProperties → "ZE_extension_graph" か "ZE_extension_graph_<n>..." (OpenVINO と同じ規則)
zeDriverGetExtensionFunctionAddress → ze_graph_dditable_ext_t。同様に "ZE_extension_command_queue_npu" (任意)
zeContextCreate; pfnDeviceGetGraphProperties → graph 拡張の版、コンパイラの版、形式
```

- `CapCompile` は `graphFormatsSupported` に `NGRAPH_LITE` があるときだけ。
- DDI テーブルは `[N]uintptr` で受け、交渉した版までのエントリだけを `purego.RegisterFunc` で束縛する。目標 1.20、最低 1.5 (`Create2`、`ze_graph_desc_2_t`)。それ未満は `VersionError`。1.8 以上は `GetProperties2` の `initStageRequired` で `GraphInitialize` を直接呼ぶかを決める (OpenVINO と同じ)。
- Linux では `internal/accelfs` が `/sys/class/accel` を走査し (`intel_vpu` / `amdxdna`)、「デバイスはあるがライブラリがない」と「デバイスがない」を分ける。Windows には sysfs がないので `zeDeviceGet` の結果だけで決める。

| 拡張の版 | 足されるエントリ |
|---|---|
| 1.0 | Create, Destroy, GetProperties, GetArgumentProperties, SetArgumentValue, AppendGraphInitialize, AppendGraphExecute, GetNativeBinary, DeviceGetGraphProperties |
| 1.1 | GraphGetArgumentMetadata, GetArgumentProperties2 |
| 1.2 | GetArgumentProperties3 |
| 1.3 | QueryNetworkCreate, QueryNetworkDestroy, QueryNetworkGetSupportedLayers |
| 1.4 | BuildLogGetString |
| 1.5 | Create2, QueryNetworkCreate2, QueryContextMemory |
| 1.6 | DeviceGetGraphProperties2 |
| 1.7 | GetNativeBinary2 |
| 1.8 | GetProperties2, GraphInitialize |
| 1.11 | CompilerGetSupportedOptions, CompilerIsOptionSupported |
| 1.12 | Create3, GetProperties3, BuildLogGetString2, BuildLogDestroy |
| 1.15 | SetArgumentValue2 |
| 1.16 | Evict |
| 1.20 | GetArgumentProperties4, GetArgumentNames |

## グラフ

- ネイティブの blob: `ze_graph_desc_2_t{format: NATIVE}`。
- OpenVINO IR: OpenVINO の `model_serializer.cpp` (ALL_WEIGHTS_COPY) と同じバイト列。コンパイラの版、`numberOfInputData = 2`、xml の大きさ + xml、重みの大きさ + 重み。`internal/ovir` は実行時情報 `is_new_api` と `use_indices_for_io_metadata` も差し込む。
- ビルドフラグは OpenVINO の `serializeIOInfo` / `serializeConfig` の形 (`--inputs_precisions=… --config KEY="VALUE"`) で、`levelzero.BuildFlags` が組む。対応レイヤーの問い合わせには `--config` の部分だけを渡す。答えは `<名前0><名前1>…`。
- ビルドログはグラフのハンドルを NULL にして `pfnBuildLogGetString` で取り、`BuildError.Log` に入れる。
- コンパイルした blob はキャッシュする (`WithCacheDir`)。キャッシュからの読み込みはコンパイラを通らない。

Intel での配置の検証 (NPU のコンパイラは演算を他の装置に回さない):

| 状況 | 結果 |
|---|---|
| コンパイル済み blob | `Verified`、`Method = "levelzero-native-blob"` |
| IR のコンパイル成功 | `Verified`、`Method = "levelzero-compiler"` |
| IR のコンパイル失敗 | 対応レイヤーを問い合わせ、未対応があれば配置方針によらず `PlacementError` (`Device = "unsupported"`、`Cause` はログ付きの `BuildError`)。なければ `BuildError` |

## 引数、メモリ、実行

- `AllocBuffer` は `zeMemAllocHost(align 4096)` を `[]byte` として見せる。解放は `zeMemFree`。`SetArgumentValue` は Level Zero のメモリしか受けないので、`Bind` は `*Buffer` だけを受ける (cgo のポインタ規則でも正しい)。ホストが直接読める (`CapHostCoherent`)。Sync は何もしない。
- 引数のメタデータは `GetArgumentProperties3` から `ArgInfo` (形、`DType`、向き) に。`ArgInfo.ModelDType` はモデル上の精度 (fp32 のモデルが fp16 で動くことがある)。
- 実行: コマンドキュー (turbo のための NPU 拡張は任意) → コマンドリスト → 初回だけ `GraphInitialize` か `AppendGraphInitialize` → `AppendGraphExecute` → close → fence 付きで実行。`Fence.Wait` は `ctx.Done()` を見ながら `zeFenceHostSynchronize(50 ms)` を繰り返す。
- 引数の値はコマンドを積んだ時点で取り込まれるので、要求ごとに `SetArgumentValue` から積み直す (グラフごとのロック)。終わった要求への `Bind` は許す (注意の塊の鎖が付け替えを使う)。
- C の配置 (`ze_graph_desc_2_t`、`ze_graph_argument_properties_3_t` など) は Go の struct で書き、`testdata/abi/levelzero_abi.json` (公式ヘッダから求めたサイズとオフセット。macOS arm64 の clang と Linux amd64 の gcc で同じ) と golden テストで突き合わせる。
- `testdata/zemock/zemock.c` は偽のドライバ。テストの開始時に `cc` でビルドし、`HWNPU_ZE_LOADER` で読ませる。

## Intel 向けのプログラム

- 変換器は Apple と Intel で 1 つ。アーキテクチャが MIL を組み、`internal/ovlower` が 1 演算ずつ IR に下ろす (`out.xml` + `out.bin`、`out.partN.xml`、`meta` 付きの `out.parts.json`)。知らない演算や入力はエラーで止める。milcpu の演算は `read_state` (プローブだけ) を除いて全部下ろせる。演算の表は [../intel-npu-models.md](../intel-npu-models.md)。
- 置き場: `<model>/intel/` (hwnpu-chat)、書き出しのディレクトリでは拡張子 `.intel`。変換器は `Backend` を受ける (`-backend intel`、`hwnpu-image convert --backend intel`、`depth.Setup.Backend`、`sana.ConvertOptions.Backend`、`systemone.Setup.Backend`、`music.WriteProgram` / `llm.WriteIR`)。
- コマンドは `chat.HereVendor` で機械を Intel として扱う。`llm.ProgramBackend` はバックエンド名だけでなくプログラムの形式名 (`intel`) も受ける。
- 重みの形式: NPU は表を読めないので、`llm.IntelFormat` が `lut4` → `int4`、`lut6` / `lut8` → `int8` に読み替える。lut の重みはこの案内つきで断る。アーキテクチャごとの上書き (`Arch.IntelFormat`、`llm.IntelFormatter`) は、LFM3、BERT、PLaMo-Embedding、PLaMo 3 の 2B 以下を int8、PLaMo 3 8B (隠れ層 4096 以上) を行ごとの int4 にする。SANA 1.5 1.6B の DiT は fp16 のまま (`sana.FormatsFor`。行ごとの int8 で絵が崩れる)。MOSS の文のモデルは 5 層を fp16 に残す。詳細は [quantization](quantization.md)。
- LLM は注意の前後で分ける (`llm.convertPaged`): 部分 `out.partN.xml` と注意のプログラム `out.attn.xml` (MIL で組み `ovlower` で下ろす)。Intel の LFM3 の既定の形は `lfm3-int8-grouped-p8192-c262144-q8.parts.json`。実行側は [llm-engine](llm-engine.md)。
- Intel の hwnpu-chat はモデルごとに 2 本 (8 トークンと 64 トークン、`onePrograms` でない) を持つので、重みを 2 回持つ。models.yaml の `prefill: off` で 1 本にする (PLaMo 3 8B で使う。int4 で 3.8 GiB)。
- 8 トークンと 64 トークンのプログラムの `.bin` は、ファイルシステムが複製を共有できるところでディスクを共有する ([distribution](distribution.md))。

### Meteor Lake での重みの形式 (`hwnpu-formats`、LFM3-1.7B、キャッシュ 4096、1 回 8 トークン)

| 形式 | 1 回の ms | 次のトークンが fp16 と同じ | パープレキシティ |
|---|---|---|---|
| fp16 | 112.1 | — | 4.57 |
| int8 | 68.5 | 99 % | 4.56 |
| int4-g128 | 84.4 | 76 % | 6.13 |
| int4 (行ごと) | 37.9 | 63 % | 10.32 |

既定は int8 のまま。ここではグループごとの int4 は int8 より遅く、行ごとの int4 は速いが精度が足りない。

## NPU なしで確かめる (OpenVINO の CPU)

- `HWNPU_BACKEND=openvino` (または `--backend openvino`、装置は `HWNPU_DEV=ov.device`) で、Intel の IR を OpenVINO の CPU プラグインで動かす (`internal/backend/openvino`、Python は `HWNPU_OV_PYTHON`、`serve_ir.py` / `run_ir.py`)。変換、引数の名前、ホストの処理は NPU と同じで、違うのはコンパイラと装置だけ。pip の openvino は Mac でも動く。
- `serve_ir.py` と `run_ir.py` は `openvino_telemetry` のモジュールを読ませない (別プロセスが消したスクリプトを読みにいって失敗していた)。
- バックエンドを開くとき、落ちたプロセスが `/dev/shm` に残した共有バッファのディレクトリ (1 日で 8.4 GB) を片付ける。
- `hwnpu-dev irdiff A.mlpackage …`: Core ML のパッケージを IR に下ろし、同じ乱数の入力で milcpu と OpenVINO の CPU を比べる。Intel の経路を書く前に、下ろせない演算を見つけられる。
- テスト `TestIntelIR*` (lfm3、gemma4、lfm3vl、lfm3audio、plamo3、bert、plamo、qwen3、qwen35 の fp16 / int8、whisper。gemma4 と lfm3 は分けた形も): milcpu と相対誤差 1e-3 以内 (分けた形は gemma4 2.8e-3、lfm3 5.7e-4)。`TestImageOps`: 3e-4 以内。
- 実物のチェックポイントを `HWNPU_BACKEND=openvino` で Ryzen AI の機械の CPU で動かし、CPU、AMD の NPU、milcpu と合った: RTMO-s の pose、RTMDet-m の detect (箱 0.1 ピクセル以内)、PP-OCRv6 small、SwiftF0 (f0 0.013 % 以内)、Basic Pitch (音符が同じ)、YuNet、DA3-SMALL (深度の相関 0.9994)、MoGe-2 ViT-S (0.9998)、EfficientSAM-S (同じマスク)、Qwen3.5-0.8B int8 のチャット、LFM3-Audio-2B の TTS → Whisper large-v3-turbo の往復、Magenta RT2 small (12 符号が同じ)、Sana 600M 1024² int8、MOSS-SoundEffect v2.0 (16 本、1 ステップ 18 秒、耳では未確認)、Anima (f32 で潜在のコサイン 0.99985、PSNR 37.9 dB。f16 の推論では 25.5 dB)。

## メモリの上限と常駐

- ドライバはグラフの重みを NPU から見えるメモリに置く。135U では Anima 512² fp16 が、約 4.6 GB (text 1.2 GB + DiT の 6 群 × 0.57 GB) を持ったところで `dit_blocks_24.xml` が `pfnGraphInitialize: ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY` になった。一方 `hwnpu-dev probe resident` は 0.23 GiB のプログラムを 16 GB までエラーなしで積めるので、上限は単純な合計ではない。1 本の大きさか本物のグラフの形かはまだ分けていない (`resident-large`: 0.57 と 1.2 GiB のプログラム)。
- 読み込みはコンパイラの `std::bad_alloc` ("bad allocation") でも失敗する。`types.IsOutOfMemory` は Level Zero の `OUT_OF_HOST_MEMORY` / `OUT_OF_DEVICE_MEMORY` と "bad allocation" / "bad_alloc" をメモリ不足とみなす。
- Intel の Anima (`anima.progEngine`): 部品を slot にし、メモリ不足ならほかの部品を閉じて読み直す。閉じる順は text (プロンプトごとに 1 回)、次に DiT の群で最後に使ったもの (MRU。7 群に 4 か所でステップあたり 4 群の読み直し。LRU なら 7)、最後に時刻・パッチ・最後の層。VAE の段も同じ規則で場所を作る。偽の NPU (`residency_test.go`) で確かめた。実機では 1024² で読み直しが働いた。
- `hwnpu-dev probe resident`: 0.25 GiB のプログラムを 1 本ずつ読み (開いたまま、1 回ずつ実行)、読み込みが失敗するか `-resident-max-gb` (RAM の 60 %、最大 16) まで続け、最後に全部をもう一度実行する。メモリ不足なら一番古いものを閉じて読み直す。フラグは `-resident-hold open|close`、`-resident-formats int8`、`-resident-size a,b`、`-resident-run=false` (読むだけ)、`-resident-cycle`。Windows では各読み込みの後にプロセスの private メモリ (`K32GetProcessMemoryInfo`) を記録し、`maxRSS` は peak working set を返す。子プロセスは 1 本ごとに `npuprobe-progress` の行を、各段階を stderr に出すので、落ちてもそこまでの分が残る。
- 1 回、読み込み中に子がドライバの中で 0xc0000005 で落ちた。メモリ不足がエラーでなく異常終了になりうるなら、上の閉じて読み直す手は効かず、落ちる合計より手前に自分で上限を置く形が要る。

### Windows のジョブの時間の上限 (TDR)

- Windows は時間の上限 (TDR、既定 2 秒) を超えた GPU / NPU のジョブを止め、デバイスを失わせる (`zeFenceHostSynchronize: ZE_RESULT_ERROR_DEVICE_LOST`)。1 プログラムに DiT 4 ブロックの Anima 1024² がこれに当たった (1 群 2〜4 秒の見込み)。
- `--dit-group` の既定は 0 = 自動 (`anima.DefaultGroup`): Core ML と Core AI は 4。Intel は 4 から半分ずつにして「ブロック数 × パッチ数」を 4096 以下に (512²: 4、640²: 2、768² と 1024²: 1)。`g` は書き出しの名前に入る。DEVICE_LOST (`types.IsDeviceLost`) で群が 2 以上なら、エラーに `--group` を半分にする案内を付ける。
- g4 と g1 の書き出しは同じ絵を描く (潜在の差は相対 3.6 %、群の間の丸め)。

## Windows

- `ze_loader.dll` が PATH にあれば同じコードで動く (`ZE_ENABLE_LOADER_DEBUG_TRACE=1` で読み込みの経路が見える)。上の 135U で確かめた。
- IR のディスクの共有は、ReFS / Dev Drive のブロックの複製があれば使う ([distribution](distribution.md))。

## 未解決

- chat 以外のモデルのために足した演算 (DepthToSpace、ConvolutionBackpropData、Clamp、MaxPool、Elu など) と大きなグラフ (Sana の Gemma 2 の 6 層 int8 で約 0.5 GB、EfficientSAM の 4096 トークンの注意) を NPU のコンパイラが受け付けるか。速さ。Sana と MOSS が 135U のメモリに収まるか。
- 4.6 GB の失敗の原因 (合計でなく、1 本の大きさかグラフの形か)。0xc0000005 の異常終了がメモリ不足か。
- g1 にした後の NPU の Anima を 512² と 1024² で: 全部の部品が読めるか ("bad allocation" の後の読み直し。プロンプトごとに text と数ブロックの見込み)、`DEVICE_LOST` で止まらないか。ドライバ内コンパイラ、1 ステップ、VAE、全体の時間と NPU のメモリ。`--dit-format int8` と `int4` を既定の fp16 と比べる (メモリが減って読み直しが要らなくなるか)。同じプロンプトと種の Mac / AMD の絵と比べる。済んだら USAGE_IMAGE.md / .ja.md の「Intel の NPU では未確認」と [performance.ja.md](performance.ja.md) の Anima の行を直す。
  ```sh
  hwnpu-image generate --size 1:1s "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms" --seed 1 -o out512.png
  hwnpu-image generate "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms" --seed 1 -o out1024.png
  ```
- 塊ごとの注意でのコマンドリストの積み直しの費用 (毎ステップ、全注意の層 × 使う塊の数)。
- Gemma 4、LFM3-VL / Audio、Qwen3.5、Whisper、d1、PLaMo 3 8B int4 (2 本か 1 本か) の NPU の実機。
- Apple の 1 GiB のような大きさの上限。`-target intel` の 2 GiB は仮の値。大きなグラフのコンパイル時間。
- 8 トークンと 64 トークンのグラフで、コンパイラが重みを共有できるか (今はそれぞれが持つ)。
- Lunar Lake 以降の速さ (測ったのは Meteor Lake だけ)。グループごとの int4 がそこで違う振る舞いをするか。
- 2 つのモデルを同時に動かすとき (キューはドライバが管理する。未測定)。
- SigLIP 2 (label)、FLUX は IRON のグラフ (Python) しかなく、Go の MIL のグラフがない。Qwen-Image は Go のグラフがあるが、まだ IR を書かない。まず `internal/llm/blocks` への移植が要る。
- Windows に loader が入らない構成があれば、Level Zero の「loader バイパス」(ドライバの DLL を読み DDI テーブルを引く)。今のところ要っていない。
- ページ境界に揃った blob のコピーを省く `ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT`。積み直しの費用が見えたら `zeMutableCommandList`。

## 採らなかったもの

- int4 (行ごと、グループごと) を Intel の既定にすること: Meteor Lake の表のとおり (行ごとの int4 は次のトークンの一致 63 %。最初の CPU の確認で LFM3 は 52 中 22)。
- Intel での `-idle` の解放: Intel のプログラムの読み直しは数十秒から数分かかる。解放するのは AMD (IRON) だけ ([amd](amd.md))。
- Anima の DiT の群の LRU: MRU ならステップあたりの読み直しが 7 群でなく 4 群。
