# hwnpu

CPU 内蔵の NPU を cgo なしで Go から使うためのライブラリです。軽量モデルを低消費電力で動かすことを目的にしています。

- 既定の厳格モードでは、NPU だけで完結しないモデルを実行前にエラーにします。CPU や GPU への黙ったフォールバックはしません
- 依存は [purego](https://github.com/ebitengine/purego) と golang.org/x/sys だけです

| プラットフォーム | バックエンド | 状態 |
|---|---|---|
| macOS (Apple Silicon) | Core ML (GPU を除外し、MLComputePlan で ANE 配置を検証) | M1 実装済み |
| Linux / Windows (Intel Core Ultra) | Level Zero (OpenVINO IR のドライバ内コンパイル、コンパイル済み blob) | M2 実装済み。実機では未検証 |
| Linux (AMD Ryzen AI) | amdxdna ioctl (IRON / mlir-aie の xclbin を実行) | M4 実装済み。Ryzen AI 9 365 (Strix Point、Linux 7.0) で IRON の設計と full ELF の実行、Gemma 4 E2B の hwnpu-chat / hwnpu-agent を確認 |
| Windows (AMD Ryzen AI) | XRT (hwnpu の C ライブラリ xrtshim を通す) | 実装済み。Windows では未検証。同じ経路を Linux の XRT 2.21 で確かめた (Gemma 4 E2B の画像と音声まで) |
| WSL2 | なし | 非対応。WSL2 からは Intel と AMD のどちらの NPU も見えない |

各 NPU が扱えるデータ形式とモデルの用意の仕方は [MODEL.ja.md](MODEL.ja.md)（[English](MODEL.md)）にあります。

導入手順は、Apple が [INSTALL_APPLE.ja.md](INSTALL_APPLE.ja.md)（[English](INSTALL_APPLE.md)）、Intel が [INSTALL_INTEL.ja.md](INSTALL_INTEL.ja.md)（[English](INSTALL_INTEL.md)）、AMD が [INSTALL_AMD.ja.md](INSTALL_AMD.ja.md)（[English](INSTALL_AMD.md)）にあります。

設計は [docs/design.md](docs/design.md) にあります。

## 使い方

```go
dev, err := hwnpu.Open(ctx)
if err != nil { ... }
defer dev.Close()

gr, _ := dev.Graph()
g, err := gr.Compile(ctx, hwnpu.GraphSource{Path: "model.mlpackage"})
if errors.Is(err, hwnpu.ErrNotOnNPU) {
	// err は *hwnpu.PlacementError。NPU 以外に置かれた演算が Report.Ops に入っている
}
defer g.Close()

in, _ := g.Arg("x")
out, _ := g.Arg("y")
x, _ := dev.AllocBuffer(in.Bytes)
y, _ := dev.AllocBuffer(out.Bytes)
hwnpu.EncodeF16(x.Bytes(), input)

req, _ := g.NewRequest()
req.BindName("x", x)
req.BindName("y", y)
err = req.Run(ctx)
hwnpu.DecodeF16(result, y.Bytes())
```

## 電力とキャッシュ

```go
dev, err := hwnpu.Open(ctx,
	hwnpu.WithPowerPolicy(hwnpu.PowerLow), // Intel: 低優先度 + バックグラウンド
	hwnpu.WithCacheDir(cacheDir),          // コンパイル結果を保存して再利用
)

before, err := dev.PowerSample() // 電力量を読めないデバイスでは ErrUnsupported
// ... 推論 ...
after, _ := dev.PowerSample()
fmt.Println(hwnpu.AveragePower(before, after), "W", after.Scope) // "package" は CPU パッケージ全体
```

- 電力ポリシーは、`CapPowerControl` のあるデバイスでだけ効きます。Core ML には電力の設定がありません
- Intel の電力量は Linux の RAPL で読みます。NPU 単体ではなく CPU パッケージ全体の値で、通常は root 権限が必要です
- キャッシュのキーにはモデル、ビルドフラグ、ドライバとコンパイラ (Apple は macOS) の版が入るので、どれかが変わると作り直します

## AMD での使い方

AMD の NPU は、モデルではなく IRON / mlir-aie でビルドしたカーネル (xclbin と命令列) を実行します。

```go
kr, err := dev.Kernel() // AMD 以外では ErrUnsupported
prog, err := kr.Load(ctx, hwnpu.ProgramSource{Path: "final.xclbin"})
k, err := prog.Kernel("MLIR_AIE")

instr, _ := dev.AllocBuffer(len(insts), hwnpu.WithBufferKind(hwnpu.BufferInstructions))
in, _ := dev.AllocBuffer(n)
out, _ := dev.AllocBuffer(n)
copy(instr.Bytes(), insts)
instr.SyncToDevice(0, len(insts)) // AMD の NPU はキャッシュ非コヒーレント
in.SyncToDevice(0, n)

run, _ := k.NewRun()
err = run.SetUint64(0, 3).SetBuffer(1, instr).SetUint32(2, uint32(len(insts)/4)).
	SetBuffer(3, in).SetBuffer(4, tmp).SetBuffer(5, out).
	Run(ctx)
out.SyncFromDevice(0, n)
```

- 引数の番号、オフセット、型は xclbin のメタデータから読みます (`k.Args()`)
- `/dev/accel/accelN` に読み書きできる権限 (`render` グループ) が必要です。Linux 6.14 以降のカーネル (amdxdna) を前提にしています
- 電力ポリシーはハードウェアコンテキストの優先度 (QoS) に反映します。デバイス全体の電力モードは root 専用なので変更しません
- 電力は NPU のセンサー値 (瞬間値) を、なければ amdgpu ドライバが出すパッケージ全体の電力 (PPT、root 不要) を読みます。最初の `PowerSample` から 50 ms ごとに読んで積分し、電力量にします。NPU のセンサーは AMD の DKMS 版のドライバにしかなく、Linux 7.0 の標準カーネルではパッケージ全体の値になります (`Scope` が `package`)

Strix Point 以降 (XDNA2) では、IRON がビルドした full ELF (`design.elf`、グラフ全体を 1 つのハードウェアコンテキストと 1 つのコマンドで流すもの) も読み込めます。カーネル名は `main:sequence` の形で、引数はすべてバッファです。呼び出しごとの値は `run.Scratchpad()` に書きます。

### LLM (Gemma 4 E2B / E4B、LFM3 とその多モーダル、PLaMo 3 Translate) を AMD の NPU で

IRON (amd/IRON) を開発者の機械でコンパイラとして一度だけ使い、`tools/amd/iron_export_lm.py` で書き出したものを hwnpu が Go だけで動かします (利用者の機械に Python も XRT も要りません)。

```bash
python tools/amd/iron_export_lm.py gemma4 ~/models/gemma-4-E2B-it ~/models/gemma-4-E2B-it.amd-npu2 --max-seq-len 4096 --prefill-chunk 1024
go run ./cmd/hwnpu-chat -iron ~/models/gemma-4-E2B-it.amd-npu2
go run ./cmd/hwnpu-agent -iron ~/models/gemma-4-E2B-it.amd-npu2 "README.md を要約して"
```

Gemma 4 E4B (gemma-4-E4B-it) も同じ `gemma4` で書き出します (文脈 4096、塊 1024。書き出しは 15 GB)。デコードは 1 秒に 3.6 トークン (E2B の約 6 割) で、位置 3072 からの 1024 トークンを 8.2 秒で読みます。

LFM3-1.7B は IRON にアプリケーションがないので、hwnpu の `tools/amd/iron_lfm3.py` を使います (`lfm3`)。畳み込みの層の状態は、プロンプトの塊の境ごとに残り、hwnpu はその位置の分を印に写して巻き戻します。文脈 12288、塊 2048 まではスライディング窓の層がリングにならないので、巻き戻しと prompt store が使えます。Ryzen AI 9 365 で、デコードは 1 秒に 8.7 トークンです。プロンプトは、先頭の 2048 トークンを 2.4 秒、位置 10240 からの 2048 トークンを 12.7 秒で読みます (注意が窓の外も含めて文脈の全体を読むため)。

```bash
python tools/amd/iron_export_lm.py lfm3 ~/.cache/hwnpu/models/LFM3-1.7B ~/models/LFM3-1.7B.amd-npu2 --max-seq-len 12288 --prefill-chunk 2048
go run ./cmd/hwnpu-chat -iron ~/models/LFM3-1.7B.amd-npu2
go run ./cmd/hwnpu-agent -iron ~/models/LFM3-1.7B.amd-npu2 "go test を通して"
```

LFM3-VL-2B、LFM3-Audio-2B、LFM3-Omni-2B も同じく hwnpu の IRON アプリケーションで書き出します (`lfm3vl`、`lfm3audio`、`lfm3omni`)。LFM3-VL は画像を 512 × 512 に引き伸ばして読み (256 トークン)、LFM3-Audio は 41 秒までの音声を読みます (音声での返事は作りません)。LFM3-Omni はその両方です。

```bash
python tools/amd/iron_export_lm.py lfm3vl ~/models/LFM3-VL-2B ~/models/LFM3-VL-2B.amd-npu2 --max-seq-len 12288 --prefill-chunk 2048
python tools/amd/iron_export_lm.py lfm3audio ~/models/LFM3-Audio-2B ~/models/LFM3-Audio-2B.amd-npu2 --max-seq-len 12288 --prefill-chunk 2048
python tools/amd/iron_export_lm.py lfm3omni ~/models/LFM3-Omni-2B ~/models/LFM3-Omni-2B.amd-npu2 --max-seq-len 12288 --prefill-chunk 2048
go run ./cmd/hwnpu-chat -iron ~/models/LFM3-VL-2B.amd-npu2 -listen 127.0.0.1:8080
```

Gemma 4 E2B に画像と音声も読ませるときは、塔 (視覚と音声のエンコーダー) ごと書き出します (`gemma4mm`。1 回のプロンプトに画像 1 枚と音声 1 本まで)。OpenAI 互換 API の `image_url` (data URL) と `input_audio` (WAV) で渡します。

```bash
python tools/amd/iron_export_lm.py gemma4mm ~/models/gemma-4-E2B-it ~/models/gemma-4-E2B-it.mm.amd-npu2 --max-seq-len 4096 --prefill-chunk 1024
go run ./cmd/hwnpu-chat -iron ~/models/gemma-4-E2B-it.mm.amd-npu2 -listen 127.0.0.1:8080
```

NPU のバッファは固定されたメモリで、プロセスの RSS には出ません (Gemma 4 E2B の 4 ビットで 1.8 GB)。`hwnpu-chat` と `hwnpu-agent` は、`-iron` のモデルを要求が `-idle` (既定 5 分) 来ない間はデバイスから下ろしてメモリを返し、次の要求で読み直します (E2B で 1〜2 秒)。会話はホストに写しておいた続きから進みます。`-idle 0` で下ろしません。

Preferred Networks の翻訳モデル PLaMo 3 Translate 2B (pfnet/plamo-3-translate-2b) も、hwnpu の `tools/amd/iron_plamo3.py` で書き出します (`plamo3`)。hwnpu-chat は打った文を訳し (日本語を含む文は英語へ、ほかは日本語へ。`-to LANG` で訳す先、`-from` で原文の言語、`-form plain|polite` で日本語の文体)、それまでの対訳を履歴としてプロンプトに残します。OpenAI 互換 API では `chat_template_kwargs` の `target_lang`、`source_lang`、`form`、`context`、`glossary` (vLLM と同じ) で指定し、system メッセージは参考文脈 (`context`) になります。`--q4` で MLP と出力の行列を 4 ビット、注意を 8 ビット、埋め込みをホストに置くと、固定されるメモリが 5.96 から 2.16 GiB に、デコードが 1 秒に 7.3 から 11.6 トークンになり (Ryzen AI 9 365、位置 2600 付近)、訳文は 44 文中 36 文が bf16 と同じで、残りも正しい言い換えでした ([docs/amd-npu-models.md](docs/amd-npu-models.md) の 13 節)。bf16 のままなら `--q4` を外します (デコードは 1 秒に 7.7 トークン、位置 8191 で 6.6。1024 トークンの塊は位置 0 から 2.2 秒、7168 から 10.5 秒)。Gemma 4 の書き出しにも `--q4` があります。

```bash
python tools/amd/iron_export_lm.py plamo3 ~/models/plamo-3-translate-2b ~/models/plamo-3-translate-2b.amd-npu2-q4 --max-seq-len 8192 --prefill-chunk 1024 --q4
go run ./cmd/hwnpu-chat -iron ~/models/plamo-3-translate-2b.amd-npu2-q4
go run ./cmd/hwnpu-chat -iron ~/models/plamo-3-translate-2b.amd-npu2-q4 -to zh-hans
```

Phoenix と Hawk Point (NPU1) には full ELF がないので、`--device npu1 --cols 4` で書き出します。各版は IRON と同じく「1 ステップごとの xclbin」(Gemma 4 E2B のデコードで 1095 ステップ、xclbin は 2 個) になり、呼び出しごとの命令列は書き出しに含めたプログラムから hwnpu が作ります。NPU1 の実機では未検証です。同じ形の書き出しを `--per-step` で NPU2 向けに作ると Ryzen AI 9 365 で正しく動き、デコードは 1 秒に 2.6 トークンでした (full ELF の 6 の半分以下。1 トークンに 1095 回の投入が要るため)。NPU1 は、版ごとの xclbin (計 6 個) で、ハードウェアコンテキストの上限 6 を使い切ります。

```bash
python tools/amd/iron_export_lm.py gemma4 ~/models/gemma-4-E2B-it ~/models/gemma-4-E2B-it.amd-npu1 --max-seq-len 4096 --prefill-chunk 1024 --device npu1 --cols 4
```

Ryzen AI 9 365 で、デコードは 1 秒に 5.5〜6 トークン、プロンプトは 1 秒に 260 トークン前後です (bf16 の重み)。プロンプトの塊は 1024 トークンが速く (IRON の既定の 2048 より 2 割)、4096 トークンの文脈でもスライディング窓の層がリングにならないので、巻き戻しと prompt store が使えます。仕組みと制約は [docs/amd-npu-models.md](docs/amd-npu-models.md) にあります。

### 画像生成 (Anima) を AMD の NPU で

`go run ./cmd/hwnpu-image` は、Anima (circlestone-labs/Anima、Cosmos-Predict2 2B の DiT と Qwen3-0.6B、Qwen-Image の VAE) でプロンプトから絵を描きます。テキストの符号化、DiT のステップ、VAE の復号はすべて NPU の上の IRON のグラフで、ホストはトークン化、埋め込みの行の取り出し、パッチの並べ替え、Euler のステップ、PNG を受け持ちます。書き出しは画像の大きさごとで、hwnpu の `tools/amd/iron_anima.py` が作ります (Anima-Turbo、1024 × 1024 で 5.4 GB)。

```bash
python tools/amd/iron_anima.py ~/models/Anima ~/models/anima-turbo-1024.amd-npu2 --size 1024
go run ./cmd/hwnpu-image -model ~/models/anima-turbo-1024.amd-npu2 -prompt "masterpiece, best quality, score_7, safe, 1girl, cherry blossoms" -o out.png
```

Ryzen AI 9 365 で、1024 × 1024 の 1 枚が 8 ステップで約 70 秒です (読み込み 4 秒、テキスト 0.5 秒、DiT 1 ステップ 7.2 秒、VAE 7 秒)。`~/models/Anima` の置き方と、精度の測定は [docs/amd-npu-models.md](docs/amd-npu-models.md) の 11 節にあります。

Mac の Neural Engine と Intel の NPU には、`-convert` が同じ部品を Go だけで Core ML のパッケージか OpenVINO IR に書きます (Python は要りません)。CPU のインタプリタと OpenVINO の CPU では参照と一致しますが、実機では未確認です ([docs/handoff-anima-mac-intel.md](docs/handoff-anima-mac-intel.md))。

```bash
go run ./cmd/hwnpu-image -convert ~/models/Anima -backend coreml -size 1024 -model ~/models/anima-turbo-1024.coreml   # Intel: -backend intel
go run ./cmd/hwnpu-image -model ~/models/anima-turbo-1024.coreml -prompt "..." -o out.png
```

## Apple での注意

- Core ML は「Neural Engine だけ」を指定できません。hwnpu は CPU と Neural Engine だけを許可し、`MLComputePlan` (macOS 14.4 以降) の推定で全演算が Neural Engine に置かれることを確認します。推定なので、実行時の配置までは保証されません
- Neural Engine は fp16 前提です。fp32 のモデルや小さすぎるモデルは CPU に置かれます。coremltools で fp16 の ML Program に変換してください
- 小さすぎるモデルは、`tools/coreml/ane_pad.py` でバッチ方向にパディングすると Neural Engine に載せられます。ただし小さなモデルでは遅くなることがあるので、詳しくは [MODEL.ja.md](MODEL.ja.md) を読んでから使ってください

## 動作確認レポート

手元の NPU で hwnpu が動くかを確かめ、結果をそのまま貼り付けられる Markdown で出力します。テスト用モデルはプログラムに埋め込んであるので、リポジトリのどこから実行しても動きます。

```bash
go run ./cmd/report -o report.md
```

- 見つかったデバイスごとに、そのバックエンド向けのモデルを厳格モードでコンパイルし、推論結果を fp32 の参照値と比べ、実行時間を測ります
- Level Zero loader の読み込み結果も載るので、Intel NPU が見つからないときの原因が分かります
- `-json report.json` で JSON も出力します。`-n` で計測の回数を、`-power low` などで電力ポリシーを、`-cache DIR` でコンパイルキャッシュを指定できます
- AMD では `-xclbin final.xclbin -insts insts.bin` を渡すと、mlir-aie の `programming_examples/basic/passthrough_dmas` をビルドした設計を実行して出力を照合します (引数の並びは opcode, instr, ninstr, 入力, 未使用, 出力)
- 失敗があると終了コード 2、NPU が見つからないと 1 で終わります

macOS では `go run ./cmd/powerbench` で、Core ML モデルの呼び出し 1 回あたりのエネルギーを powermetrics で測れます。最初に sudo のパスワードを一度だけ聞きます。詳しくは [MODEL.ja.md](MODEL.ja.md) を参照してください。

`go run ./cmd/hwnpu-convert` は、Hugging Face のチェックポイント (safetensors か GGUF) を Python なしで Neural Engine 用の `.mlpackage` に変換します。対応するのは LFM3 (d1-2B などの判定モデルを含む)、LFM3-VL の視覚エンコーダ、LFM3-Audio / LFM3-Omni の音声エンコーダと音声出力、Gemma 4（gemma-4-E2B-it）、Qwen3.5（Qwen3.5-4B などのテキスト部分。CPU での照合のみ）、BERT 系エンコーダ、PLaMo-Embedding-1B、PLaMo 3（PLaMo 3 Translate 2B。CPU と OpenVINO の CPU での照合のみ。`hwnpu-chat -use plamo-3-translate-2b` で翻訳）で、移植の手順は `.claude/skills/port-architecture` にあります。元モデルの選び方と Neural Engine の制約は [MODEL.ja.md](MODEL.ja.md) にまとめてあります。`go run ./cmd/hwnpu-check` は、変換したグラフを CPU で fp32 のまま実行し、Hugging Face の実装の参照 (`tools/ref/ref_logits.py`) と層ごとに照合します。Mac は要りません。`go run ./cmd/hwnpu-generate` は、変換した因果 LM で文章を生成します (トークナイザーも Go)。`-cache N` で変換すると KV キャッシュ付きになり、長い文章を書けます (M3 の LFM3-1.7B で、キャッシュ 1024 なら 1 トークン 20.6 ms)。`go run ./cmd/hwnpu-chat` は、引数なしで LFM3-1.7B を落として変換し、対話を始めます。使うモデルは `-use` に名前で指定し (`-use plamo-embedding-1b` なら PLaMo-Embedding-1B で文を埋め込んで検索)、選べるモデルと用途は `-list` で出ます。`-listen` で OpenAI 互換の API サーバーになり、`-use LFM3-1.7B,plamo-embedding-1b` ならチャット (`/v1/chat/completions`) と埋め込み (`/v1/embeddings`) を 1 つのサーバーで出します。`-opencode` は、サーバーを 30000 番台以降の空いたポートで開き、その設定で OpenCode を起動します。OpenCode のようなコーディングエージェントからも使えます。`go run ./cmd/hwnpu-agent "main.go の add のバグを直して"` は、同じモデルを HTTP なしで直接呼ぶ 4 道具 (read、edit、write、bash) のコーディングエージェントで、system prompt と道具の宣言は約 600 トークン (OpenCode は 7.4K) です。`-tools search -search-first` なら、知識をモデルに頼らず検索 (Wikipedia、Brave、SearXNG) して答えるアシスタントになります。手順は [CHAT.ja.md](CHAT.ja.md)（[English](CHAT.md)）にあります。Intel の NPU (Core Ultra) では、同じ hwnpu-chat と hwnpu-agent がモデルを OpenVINO IR に変換して動かします (実機では未確認。[INSTALL_INTEL.ja.md](INSTALL_INTEL.ja.md) の 6 節)。`go run ./cmd/hwnpu-formats` は、重みの形式 (int8、int4-g128、int4) ごとの精度と速さを NPU の上で比べます。`go run ./cmd/hwnpu-systemone` は、d1-2B で状態についての型付きの質問 (選択、段階、真偽) に答えます。llama.cpp のサーバーと同じ `/v1/systemone` を提供します。`hwnpu-chat` と `hwnpu-systemone` は `-vision` で画像を、`-audio` で音声を読みます (LFM3-VL-2B、LFM3-Audio-2B、d1-2B、d1-Omni-2B)。`go run ./cmd/hwnpu-tts` は LFM3-Audio-2B で文章を読み上げます。

`go run ./cmd/npuprobe` は、生成したモデルで NPU を測ります。LLM が 1 トークンを生成するのと同じ形のモデルを Go で作り、まず 0.5B と 1B の重みで形式ごと (Apple は fp16、int8、lut4、int4、Intel は fp16、int8、int4) の速さを測ります (`bench`)。次にモデルを大きくしながら NPU から外れる境目を二分探索で絞り、境目のすぐ下のモデルを複数同時に読み込んで、上限がモデルごとか合計かを確かめます (`limit`)。Python もモデルのファイルも root も要らず、クローンしなくても次の 1 行で動きます。結果は `npuprobe-<機種>-<日時>.md` に書き出します。

```bash
go run github.com/shibukawa/hwnpu/cmd/npuprobe@latest
```

`bench` か `limit` を付けると片方だけを行います。Apple で 1 トークンあたりのエネルギーも測るには `go run ./cmd/powerbench -decode 8,16 -gpu` を使います (sudo が要ります)。機種ごとの手順、結果の読み方、これまでの結果は [PROBE.ja.md](PROBE.ja.md) にまとめてあります。

Go の入っていない Windows 機で試すときは、Go のある機械でビルドした実行ファイルを持っていきます。

```bash
GOOS=windows GOARCH=amd64 CGO_ENABLED=0 go build -o report.exe ./cmd/report
```

## 診断

```bash
go run ./cmd/hwnpuinfo -model internal/fixtures/coreml/linear_relu_fp16.mlpackage
```

## Intel での注意

- Intel NPU ドライバ (`intel-level-zero-npu`、`intel-driver-compiler-npu`、`intel-fw-npu`) と Level Zero loader (`libze1`) が必要です。loader の場所は環境変数 `HWNPU_ZE_LOADER` で指定することもできます
- Linux では `/dev/accel/accel0` に読み書きできる権限 (`render` グループ) が必要です
- `GraphSource{Path: "model.xml"}` で OpenVINO IR (隣の `.bin` も読みます) を、`GraphSource{Blob: ...}` でコンパイル済み blob を渡せます。コンパイラに渡す追加のオプションは `BuildFlags` に `--config KEY="VALUE"` の形で書きます
- NPU コンパイラが対応しない演算を含むモデルは、配置方針にかかわらず `ErrNotOnNPU` になり、原因の演算とビルドログが返ります

## テスト

```bash
CGO_ENABLED=0 go test ./...
```

- C コンパイラ (`cc`) があれば、テストは偽の Level Zero ドライバ (`testdata/zemock`) をビルドして Intel バックエンドを確かめます
- テスト用モデルは `internal/fixtures` にあり、`gen_models.py` で再生成できます (numpy と、Core ML 分は coremltools が必要です)
- Level Zero の構造体の基準値は `testdata/abi/levelzero_abi.c`、amdxdna と xclbin と ERT の基準値は `testdata/abi/xdna_abi.c` で再生成できます
- AMD のバックエンドは、Linux の amdxdna ドライバの振る舞いを模した Go 製の偽ドライバ (`internal/xdna/xdnasim`) でテストします
