# LLM を AMD の NPU で動かす

[English](USAGE_LLM_AMD.md)

Gemma 4 E2B / E4B、LFM3 とその多モーダル、PLaMo 3 Translate、Qwen3.5 を AMD の NPU で動かす手順です。コマンドの使い方そのもの (対話、サーバー、エージェント) は [CHAT.ja.md](CHAT.ja.md) に、IRON の環境の作り方は [INSTALL_AMD.ja.md](INSTALL_AMD.ja.md) にあります。

## 書き出しと実行

IRON (amd/IRON) を開発者の機械でコンパイラとして一度だけ使い、`tools/amd/iron_export_lm.py` で書き出したものを hwnpu が Go だけで動かします (利用者の機械に Python も XRT も要りません)。

```bash
python tools/amd/iron_export_lm.py gemma4 ~/models/gemma-4-E2B-it ~/models/gemma-4-E2B-it.amd-npu2 --max-seq-len 4096 --prefill-chunk 1024
go run ./cmd/hwnpu-chat -m ~/models/gemma-4-E2B-it.amd-npu2
go run ./cmd/hwnpu-chat agent -m ~/models/gemma-4-E2B-it.amd-npu2 "README.md を要約して"
```

Apple と Intel の NPU がなく AMD の NPU があるとき、`hwnpu-chat` (`chat`、`agent`、`serve`) は `-m` に書き出しのディレクトリを渡さなくても、`-m` のモデル (既定 LFM3-1.7B) の書き出しを `~/models/NAME.amd-npu2-q4`、なければ `~/models/NAME.amd-npu2` から選びます (`go run ./cmd/hwnpu-chat -m gemma-4-E2B-it`)。用途やファミリーもほかの NPU と同じくメモリで選びます (`-m translate`、`-m chat`)。読み込みでは書き出しの部分 (`decode`、`prompt`) ごとに `all on the NPU` (緑) か、ホストで動くステップがあれば `NOT all on the NPU` (赤) と出します。

書き出しがなければ、ほかの NPU で初回にモデルを落として変換するのと同じく、初回にチェックポイントを `~/.cache/hwnpu/models/NAME` に落とし (書き出しが読むファイルだけ)、IRON で `~/models/NAME.amd-npu2-q4` (Qwen3.5 の 0.8B と 2B は `-q6`) に書き出してから動かします。書き出し方は models.yaml の `amd_export` に書いてあります (`lfm3 --max-seq-len 12288 --q4` など)。書き出しには IRON の環境 (mlir-aie、Peano、Python 3.12。[INSTALL_AMD.ja.md](INSTALL_AMD.ja.md) の 2) が要り、`HWNPU_IRON_ENV` にその仮想環境か、それを整えるスクリプト (bash で source するもの) を渡します (なければ PATH の python3)。環境がなければ、何を入れればよいかを言って止まります。書き出しの出力は `~/models/NAME.amd-npu2-q4.log` に残ります。`amd_export` のないモデル (Qwen3-0.6B、埋め込みのモデル) は AMD の NPU では動きません (`hwnpu-chat list` の RUNS FROM の列)。

```bash
HWNPU_IRON_ENV=~/mlir-aie/ironenv go run ./cmd/hwnpu-chat -m translate "桜が満開です。"
```


## Gemma 4 E4B

Gemma 4 E4B (gemma-4-E4B-it) も同じ `gemma4` で書き出します (文脈 4096、塊 1024。書き出しは 15 GB)。デコードは 1 秒に 3.6 トークン (E2B の約 6 割) で、位置 3072 からの 1024 トークンを 8.2 秒で読みます。

## LFM3-1.7B

LFM3-1.7B は IRON にアプリケーションがないので、hwnpu の `tools/amd/iron_lfm3.py` を使います (`lfm3`)。畳み込みの層の状態は、プロンプトの塊の境ごとに残り、hwnpu はその位置の分を印に写して巻き戻します。文脈 12288、塊 2048 まではスライディング窓の層がリングにならないので、巻き戻しと prompt store が使えます。`--q4` で MLP と出力の行列を 4 ビット、注意と短い畳み込みの射影を 8 ビット、埋め込みをホストに置くと、Ryzen AI 9 365 で固定されるメモリが 3.75 から 1.64 GiB に、デコードが 1 秒に 8.7 から 11.2 トークン (1 トークン 1.34 → 1.05 J) になり、bf16 との KL は 0.036 nats (最上位の一致 92 %) でした ([docs/amd-npu-models.md](docs/amd-npu-models.md) の 13 節)。プロンプトの速さは変わらず、先頭の 2048 トークンを 2.4 秒、位置 10240 からの 2048 トークンを 12.5 秒で読みます (注意が窓の外も含めて文脈の全体を読むため)。bf16 のままなら `--q4` を外します。

```bash
python tools/amd/iron_export_lm.py lfm3 ~/.cache/hwnpu/models/LFM3-1.7B ~/models/LFM3-1.7B.amd-npu2-q4 --max-seq-len 12288 --prefill-chunk 2048 --q4
go run ./cmd/hwnpu-chat -m ~/models/LFM3-1.7B.amd-npu2-q4
go run ./cmd/hwnpu-chat agent -m ~/models/LFM3-1.7B.amd-npu2-q4 "go test を通して"
```

## LFM3-VL、LFM3-Audio、LFM3-Omni

LFM3-VL-2B、LFM3-Audio-2B、LFM3-Omni-2B も同じく hwnpu の IRON アプリケーションで書き出します (`lfm3vl`、`lfm3audio`、`lfm3omni`)。LFM3-VL は画像を 512 × 512 に引き伸ばして読み (256 トークン)、LFM3-Audio は 41 秒までの音声を読みます (音声での返事は作りません)。LFM3-Omni はその両方です。`--q4` は本体だけを 4 ビットと 8 ビットにし (塔は bf16)、固定されるメモリは VL で 4.12 → 2.00 GiB、Audio で 4.14 → 2.03、Omni で 4.51 → 2.39 GiB になります。

```bash
python tools/amd/iron_export_lm.py lfm3vl ~/models/LFM3-VL-2B ~/models/LFM3-VL-2B.amd-npu2-q4 --max-seq-len 12288 --prefill-chunk 2048 --q4
python tools/amd/iron_export_lm.py lfm3audio ~/models/LFM3-Audio-2B ~/models/LFM3-Audio-2B.amd-npu2-q4 --max-seq-len 12288 --prefill-chunk 2048 --q4
python tools/amd/iron_export_lm.py lfm3omni ~/models/LFM3-Omni-2B ~/models/LFM3-Omni-2B.amd-npu2-q4 --max-seq-len 12288 --prefill-chunk 2048 --q4
go run ./cmd/hwnpu-chat serve -m ~/models/LFM3-VL-2B.amd-npu2-q4 --listen 127.0.0.1:8080
```

## Gemma 4 E2B の画像と音声

Gemma 4 E2B に画像と音声も読ませるときは、塔 (視覚と音声のエンコーダー) ごと書き出します (`gemma4mm`。1 回のプロンプトに画像 1 枚と音声 1 本まで)。OpenAI 互換 API の `image_url` (data URL) と `input_audio` (WAV) で渡します。

```bash
python tools/amd/iron_export_lm.py gemma4mm ~/models/gemma-4-E2B-it ~/models/gemma-4-E2B-it.mm.amd-npu2 --max-seq-len 4096 --prefill-chunk 1024
go run ./cmd/hwnpu-chat serve -m ~/models/gemma-4-E2B-it.mm.amd-npu2 --listen 127.0.0.1:8080
```

## メモリとディスク

NPU のバッファは固定されたメモリで、プロセスの RSS には出ません (Gemma 4 E2B の 4 ビットで 1.8 GB)。`hwnpu-chat` (`chat`、`agent`、`serve`) は、AMD の NPU の書き出しを要求が `--idle` (既定 5 分) 来ない間はデバイスから下ろしてメモリを返し、次の要求で読み直します (E2B で 1〜2 秒)。会話はホストに写しておいた続きから進みます。`--idle 0` で下ろしません。画像と音声つきの書き出しは、塔の重みを最初の画像か音声が来るまで載せないので (Gemma 4 E2B の 4 ビットで 2.82 → 1.83 GB、文だけの書き出しとの差は 51 MiB。最初の画像か音声で 0.7 秒。[docs/design.md](docs/design.md) の 16.81)、文だけで使うときも同じ書き出しで済みます。

同じモデルの書き出しを複数置くと (文だけと画像と音声つき、4 ビットと bf16 など)、同じファイルを持ちます (Gemma 4 E2B なら `ple.bin` の 4.4 GiB など)。モデルを置くディスクが Btrfs や XFS (reflink)、Windows の ReFS や Dev Drive のように写しを共有できるファイルシステムなら、hwnpu は書き出しを最初に読むときに、隣の書き出しと同じバイトのファイルとファイルの先頭をその写し (reflink) に置き換え、ディスクを 1 つ分しか使いません (Gemma 4 E2B の 2 つで 15.2 → 8.2 GiB。普通のファイルのまま)。ext4 と NTFS では何もしません ([docs/design.md](docs/design.md) の 16.79。Windows は実機では未確認)。

## PLaMo 3 Translate

Preferred Networks の翻訳モデル PLaMo 3 Translate 2B (pfnet/plamo-3-translate-2b) も、hwnpu の `tools/amd/iron_plamo3.py` で書き出します (`plamo3`)。hwnpu-chat は打った文を訳し (日本語を含む文は英語へ、ほかは日本語へ。`--to LANG` で訳す先、`--from` で原文の言語、`--form plain|polite` で日本語の文体)、それまでの対訳を履歴としてプロンプトに残します。OpenAI 互換 API では `chat_template_kwargs` の `target_lang`、`source_lang`、`form`、`context`、`glossary` (vLLM と同じ) で指定し、system メッセージは参考文脈 (`context`) になります。`--q4` で MLP と出力の行列を 4 ビット、注意を 8 ビット、埋め込みをホストに置くと、固定されるメモリが 5.96 から 2.16 GiB に、デコードが 1 秒に 7.3 から 11.6 トークンになり (Ryzen AI 9 365、位置 2600 付近)、訳文は 44 文中 36 文が bf16 と同じで、残りも正しい言い換えでした ([docs/amd-npu-models.md](docs/amd-npu-models.md) の 13 節)。bf16 のままなら `--q4` を外します (デコードは 1 秒に 7.7 トークン、位置 8191 で 6.6。1024 トークンの塊は位置 0 から 2.2 秒、7168 から 10.5 秒)。Gemma 4 と LFM3 の書き出しにも `--q4` があります。

```bash
python tools/amd/iron_export_lm.py plamo3 ~/models/plamo-3-translate-2b ~/models/plamo-3-translate-2b.amd-npu2-q4 --max-seq-len 8192 --prefill-chunk 1024 --q4
go run ./cmd/hwnpu-chat -m ~/models/plamo-3-translate-2b.amd-npu2-q4
go run ./cmd/hwnpu-chat -m ~/models/plamo-3-translate-2b.amd-npu2-q4 --to zh-hans
```

## Qwen3.5

Qwen3.5 (0.8B、2B、4B) は hwnpu の `tools/amd/iron_qwen35.py` で書き出します (`qwen35`)。0.8B と 2B は 4 ビットでは長い英文で崩れるので、MLP と出力の行列を 6 ビットにします (`--mlp q6 --head q6`。注意と GDN の射影は 8 ビット、埋め込みはホスト。固定されるメモリは 0.8B 1.04 GiB、2B 1.97 GiB)。`--verify 3` を足すと、次のトークンと推測 2 つを 1 回で確かめる版も書き出し、hwnpu-chat はプロンプトと答えからの推測を確かめながら生成します (投機的デコード。ロジットは 1 トークンずつと同じなので、温度 0 なら文も同じ)。Ryzen AI 9 365 の 0.8B で 1 トークン 82 ms が、コードで 72 ms、日本語の説明で 62 ms になりました ([docs/design.md](docs/design.md) の 16.117)。`HWNPU_IRON_DRAFTS=0` で推測を使いません。

```bash
python tools/amd/iron_export_lm.py qwen35 ~/models/Qwen3.5-0.8B ~/models/Qwen3.5-0.8B.amd-npu2-q6 --q4 --mlp q6 --head q6 --verify 3
go run ./cmd/hwnpu-chat -m ~/models/Qwen3.5-0.8B.amd-npu2-q6
```

## Phoenix と Hawk Point (NPU1)

Phoenix と Hawk Point (NPU1) には full ELF がないので、`--device npu1 --cols 4` で書き出します。各版は IRON と同じく「1 ステップごとの xclbin」(Gemma 4 E2B のデコードで 1095 ステップ、xclbin は 2 個) になり、呼び出しごとの命令列は書き出しに含めたプログラムから hwnpu が作ります。NPU1 の実機では未検証です。同じ形の書き出しを `--per-step` で NPU2 向けに作ると Ryzen AI 9 365 で正しく動き、デコードは 1 秒に 2.6 トークンでした (full ELF の 6 の半分以下。1 トークンに 1095 回の投入が要るため)。NPU1 は、版ごとの xclbin (計 6 個) で、ハードウェアコンテキストの上限 6 を使い切ります。

```bash
python tools/amd/iron_export_lm.py gemma4 ~/models/gemma-4-E2B-it ~/models/gemma-4-E2B-it.amd-npu1 --max-seq-len 4096 --prefill-chunk 1024 --device npu1 --cols 4
```

## 速さ

Ryzen AI 9 365 で、デコードは 1 秒に 5.5〜6 トークン、プロンプトは 1 秒に 260 トークン前後です (bf16 の重み)。プロンプトの塊は 1024 トークンが速く (IRON の既定の 2048 より 2 割)、4096 トークンの文脈でもスライディング窓の層がリングにならないので、巻き戻しと prompt store が使えます。仕組みと制約は [docs/amd-npu-models.md](docs/amd-npu-models.md) にあります。
