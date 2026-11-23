# 画像生成のモデル

[English](image-models.md)

hwnpu が動かす画像生成のモデル (`hwnpu-image generate`) と、どの NPU でも共通のパイプラインの決まり: 解き方、ガイダンス、編集、まとめて描く、大きさ、書き出し。NPU ごとの動かし方は [AMD](image-amd.ja.md)、[Apple と Intel](image-apple.ja.md)。コマンドのフラグは [cli.ja.md](cli.ja.md)、取得と取得元は [distribution.ja.md](distribution.ja.md)。

## パイプライン

1 枚は 3 つの部品で、どれも NPU で動く: 文の符号器がプロンプトを読み、DiT が潜在を N ステップでノイズ除去し、VAE が復号する。ComfyUI や diffusers は UI とワークフローをモデルの実行と分けている。hwnpu が受け持つのはモデルの実行だけ。

- 1 枚の時間は DiT を回す回数で決まる: ステップ数 × 1 ステップの回数 (classifier-free guidance で 2) × 1 回。文と VAE は数秒。まず回数を減らし、次に 1 回を速くする。
- どの変更も、同じノイズから CPU の float32 の参照 (`tools/ref/ref_*.py`) と比べる: 最後の潜在のコサインと絵の PSNR を、いくつかのプロンプト × 種で。絵も見る。種 1 つのコサインは大きくばらつく。
- パッケージ: `internal/imagegen/anima`、`flux2klein`、`flux3klein`、`fluxvae` (FLUX.1 の VAE。FLUX.2 と FLUX.3 が使う)。`internal/imagegen/aspect` (大きさ)、`internal/imagegen/milpart` (Core AI の部品)。

## 一覧 (models.yaml)

| 名前 (family:tag) | 部品 | ステップ / ガイダンス | ライセンス、内容 | `machine` | メモリ AMD / Apple (GB) | NPU、時間 |
|---|---|---|---|---|---|---|
| **FLUX.2-klein-4B** (`flux2-klein:4b`) | FLUX の形の 3.9B の DiT (double 5、single 20)、Qwen3-4B、FLUX.2 の VAE | オイラー法 4、なし | Apache-2.0、安全 | 8 | 3.9 / 4.9 | AMD 1024² 約 29 秒、512² 約 10 秒。編集は AMD 1 分、参照を半分で 37 秒、Apple M3 1024² 89 秒、512² 39 秒 (編集のプログラム)。Apple M1 1024² 約 41 秒、512² 11 秒 |
| flux-3-image-klein (`flux3-klein`) | 5.5B の DiT (流れごとの mode ブロックのあと single ブロック)、Qwen3-VL-4B、FLUX.2 の VAE | flow-map 16 ステップ、なし | FLUX Kommunity (非商用、gated。初回に聞く) | 16 | 3.9 / 5.5 (Apple は見積もり) | AMD 1024² 約 1 分 45 秒、512² 32 秒。Apple M3 1024² 約 2 分 (lut8)。文からの生成だけ |
| Anima-Turbo (`anima-turbo:v1.1`) | Cosmos-Predict2 2B の DiT、Qwen3-0.6B と LLM アダプタ、Qwen-Image の VAE | オイラー法 8、なし (蒸留) | 重みは非商用 (絵は商用可。初回に聞く)、nsfw | 8 | 3.5 / 4.8 | AMD 1024² 約 32 秒。Apple M1 48 秒。Intel の形があるのはこれだけ |
| Anima-Aesthetic (`anima:aesthetic`、ファミリーの既定) | Anima-Turbo のプログラムに別の重み | DPM-Solver++ 2M 25、CFG 4 は σ ≥ 0.6 | Anima-Turbo と同じ、nsfw | 8 | 3.5 / 4.8 | Anima-Turbo の約 4.5 倍。AMD 1024² 2 分 22 秒、2.71 kJ |

- Anima の VAE、トークナイザー、設定は `circlestone-labs/Anima-Base-v1.0-Diffusers` から (`with: diffusers`)。欄 `file` はリポジトリのどのチェックポイントがそのモデルかを選ぶ (`--dit` の既定)。Anima-Base のチェックポイントは名前の頭が `net.` (`model.diffusion_model.` でない) で、`iron_anima.py`、`ref_anima.py`、`mil.OpenWeights` は今も見分ける。
- `accept: true` のライセンスは端末で一度だけ聞き、利用者の設定の隣の `licenses.json` に残す。端末がなければ `--accept-license` か `HWNPU_ACCEPT_LICENSE=ID` (その回だけ) (`chat.AcceptLicense`)。
- `content: [nsfw]` のモデル (Anima) は、名前で動かすのに `HWNPU_MODEL_ALLOWLIST` の `+nsfw` が要る。
- メモリ = NPU のメモリとプロセスのメモリの和の最大 (AMD: accel の fd の `drm-total-memory` + `RssAnon + RssShmem`。Apple: `ri_phys_footprint + ri_neural_footprint`)。1024² の 2 回目で測る。最大はたいてい VAE の最後の段。[memory-and-power.ja.md](memory-and-power.ja.md)。

## 主力: 用途 `image`

`image` (別名 `generate`、`hwnpu-image generate` の既定) は、どの NPU でも **8 GB から FLUX.2 klein 4B** を選ぶ。

- 理由: Apache-2.0、頼まなければ NSFW を描かない、AMD では編集もできる、Anima-Turbo とほぼ同じメモリ (M1 で 4.9 と 4.8 GB) で速い (41 と 48 秒)。
- 8 GB の機械 (モデルに使えるのは約 3 GB) では 1024² の最大がそれを超える。`--size 1:1s` (512²、Apple 3.1 GB、AMD 1.9 GB) なら入る。
- Anima は `+nsfw` があっても選ばない。名前で指せば動く。

## 解き方とステップ数

| モデル | 既定 | フラグ |
|---|---|---|
| Anima-Turbo | オイラー法 8、shift 3、ガイダンス 1 (`--negative` があれば 1.5) | 蒸留はオイラー法で |
| Anima-Aesthetic | DPM-Solver++ 2M 25、CFG 4 は σ ≥ 0.6 (25 のうち 17 ステップ) | モデルカードのとおり: `--sampler euler --steps 30 --cfg-sigmas 0,1` |
| FLUX.2 klein | オイラー法 4 | — |
| FLUX.3 klein | flow-map 16 ステップ (`flux3klein.Steps`) | — |

**DPM-Solver++ 2M** (`--sampler dpm2m`、`anima.Sampling`): diffusers の DPMSolverMultistep をフローの σ で。データの予測 D = x − σv、λ = log((1 − σ)/σ)、h = λ_t − λ_s として x_t = (σ_t/σ_s) x_s + (1 − σ_t)(1 − e^−h) D。2 次は前のステップの D と r = h_前/h で (1 + 1/2r) D_0 − (1/2r) D_1。最初のステップ (σ = 1 から) と最後 (0 へ) は 1 次で、オイラー法と同じ。DiT の呼び出しはオイラー法と同じ 1 ステップ 1 回。`TestDPM2M` が正規分布のデータの厳密な速度場で確かめる。

測ったこと (Anima-Aesthetic、512²、AMD、6 枚を 100 ステップのオイラー法と): オイラー法 30 で 0.982 / 25.2 dB、74 秒。DPM 25 で 0.984 / 24.3 dB。DPM 20 で 0.970。DPM 15 で 0.920 (細部が崩れる)。DPM 25 に CFG を σ ≥ 0.6 だけで 0.984、53 秒。1024² で全ステップ CFG のオイラー法 30 と比べて時間 −29 %、電力量 −29 % (電源のプロファイル balanced)。ステップ数はモデルカードの数だけでなく、多いステップのオイラー法と比べて決める。

## Classifier-free guidance

- `--guidance SCALE`、`--negative TEXT`、`--cfg-sigmas LO,HI` (σ が [LO, HI] のステップだけ CFG。`0,1` で全ステップ)。ガイダンスのあるモデルの既定は `0.6,1`: σ 0.6 未満のステップは DiT を 1 回。絵は同じ (Anima で全ステップ CFG との潜在のコサイン 0.998)。σ ≥ 0.8 では色が淡くなり、小さな物が消える。
- 条件つきと条件なしの 2 回は **重みの組ごとに続けて** 回す (エンジンの `velocity`、どのバックエンドも)。組の重みを読むのは 1 ステップ 2 回でなく 1 回。時刻とパッチの埋め込み (`dit_in`) は 1 回。
- プロンプトと負のプロンプトは文の符号器の 1 回の読み込みでまとめて符号化する (`EncodeAll`)。
- **空の負のプロンプトは定数**: diffusers は空のプロンプトをマスクしたトークンで回す。残るのは重みだけで決まる 1 行で、書き出しのときに計算する (Anima `text/empty_cond.bin`、diffusers とコサイン 1.0000)。

## 参照画像の編集 (FLUX.2 klein)

`hwnpu-image generate -m flux2-klein -i fox.png -i room.png@half "The fox from image 1 ... the living room from image 2."`

- `-i`/`--image` を 4 回まで (BFL の klein の上限)。プロンプトでは渡した順に image 1、image 2 と呼ぶ。参照は中央を絵の形に切り出し、絵の大きさに縮める。`FILE@half` はその参照を絵の半分の辺で読む (行は 4 分の 1。既定は `@full`)。`FILE@half` という名前のファイルがあれば、それをそのまま読む。
- 参照の潜在 (VAE の符号器の平均をパッチにし、バッチ正規化の統計で正規化) を絵の行の後ろに並べ、RoPE の最初の軸を 10、20、30、40 にする。変調は絵の行と同じ。最後に絵の行だけ読む。
- 半分の大きさは被写体の細部を変える (毛が平らに、顔が簡単に。CPU の float32 でも同じ) が、構図の参照は変わらない。被写体を等倍、部屋を半分にすると、等倍 2 枚と同じ細かさで 1 分 41 秒 → 1 分 8 秒。
- 1 ステップの時間は行数で決まる。書き出しは既定の参照の組み合わせを持つ ([image-amd.ja.md](image-amd.ja.md))。持っていない組み合わせは、動かせる一覧 (`References`) を添えて断る。
- Apple の Neural Engine では、編集は編集のプログラム (最初の編集で変換する別の書き出し、[image-apple.ja.md](image-apple.ja.md)) で動く。ほかのモデルは文からの生成だけ。

## 複数枚: `--batch`

複数のプロンプトで `--batch N` (既定 4) は N 枚をまとめて描く。`--batch 1` は 1 枚ずつ。絵は 1 枚ずつ描いたものとバイト単位で同じで、最後の 1 枚が描けたときに書く。

| | 文 | ステップ | VAE |
|---|---|---|---|
| Apple (Core AI): FLUX.2 klein | N 枚のプロンプトを 1 回の読み込みで | N 枚一緒に、重みの組は全部で 1 回 | 段ごとに N 枚 (`runner.Stages`) |
| AMD: FLUX.2 klein | 全部のプロンプトを 1 回の読み込みで | 組の重みを 1 回読んで全部の絵の行を通す | 1 枚ずつ |
| Anima | まとめない: プロンプト 1 つで、組を続けて回しても速くならない | | |

- 1 回読み込んで N 回回す部品は、遅い 1 回目の呼び出しを 1 回だけ払う。512² で大きく効き (M1 の Chroma1 (外した) 1 枚 1 分 24 秒 → 1 分 18 秒)、1024² では 1〜3 % (AMD の FLUX.2 klein: 512² 1 枚 8.1 → 7.3 秒、1024² 3 %)。
- `runner.Stages`: VAE の段を 1 回読み込み、全部の絵をその段に通してから次の段へ。段の間はホストが float32 で持つ (1024² で 1 枚最大 1.2 GB) ので、出力が `StageBudget` (1.25 GiB) を超える段からは入る枚数ずつ。全部の段を読み込んだまま順に呼ぶのはずっと遅い (13 段の VAE で 2.5 → 16.9 秒)。
- AMD の VAE は 1 枚ずつのまま: 段の出力をデバイスのまま次の段に渡しており、まとめると 1 枚 0.5〜0.8 秒のためにホストへ写してピークが上がる。
- 絵ごとの復号の前に `debug.FreeOSMemory` で、ピークを 1 枚分に保つ。

## 大きさとアスペクト比

`--size` は 8 つの名前 (`internal/imagegen/aspect`) を受ける: `1:1`、`1:1s` (小さい正方形)、`2:3`、`3:4`、`9:16`、`3:2`、`4:3`、`16:9`。`N` や `WxH` も。

| 名前 | `aspect.All` (16 の倍数) | FLUX.2 / FLUX.3 klein、AMD | FLUX.2 klein、Apple |
|---|---|---|---|
| 1:1 | 1024 × 1024 | 1024 × 1024 | 1024 × 1024 |
| 1:1s | 512 × 512 | 512 × 512 | 512 × 512 |
| 2:3 / 3:2 | 832 × 1248 | 832 × 1248 | 768 × 1152 |
| 3:4 / 4:3 | 880 × 1168 | 880 × 1168 | 768 × 1024 |
| 9:16 / 16:9 | 752 × 1360 | 768 × 1344 | 704 × 1280 |

- 1:1s 以外はどれも約 1.0 M 画素で、VAE の中ほどの注意は辺 16384 画素以下に収まる。
- 大きさは models.yaml の `npu.<会社>.sizes` (世代ごとに上書きできる) で、書き出しの道具の既定ではない。コマンドはこれを読んで (`npuAspects`) `--size` を解し、書き出しも作るので、`--size 16:9` は書き出しが持つものを描く。`npu.*.sizes` のないモデルは `aspect.All`。(`aspect.All32` は 32 画素の倍数のモデル向けに残るが、今の models.yaml にはない。)
- FLUX の 16:9 は 1344 × 768: 比 1.750 (−1.6 %)、1.03 M 画素、辺が 64 の倍数。Apple の FLUX.2 の大きさは Neural Engine がコンパイルできるもの (1344 × 768 と 832 × 1248 は失敗する。原因は Core AI の側: [image-apple.ja.md](image-apple.ja.md))。
- どの大きさも、いちばん低い機械で一度描いて絵を見る (縞、継ぎ目、ノイズ、真っ白)。1 つの大きさの関数の壊れはそこでしか出ない。

## 書き出し: 全部のアスペクト比で 1 つ

**決まり (どの画像のモデルも、どの NPU も)**: 書き出しはモデルと NPU の会社ごとに 1 つで、models.yaml の **全部の** アスペクト比を持ち、重みは 1 回だけ置く。大きさごとの書き出しも、一部の大きさ (`512+1024`) だけをまとめてほかを別にするものも作らない。`--size` を変えた利用者にモデルをもう一度落とさせない。足りない大きさは書き出しの道具の不具合として直す。例外には、測った理由を設計の文書とそのモデルの models.yaml の項に書く。

- **AMD**: 最初の大きさのグラフが重みのバンクを持ち、ほかの大きさのグラフ (`sizes/WxH/` か、行数の名前) は同じバンクを指す (`share_banks`)。[image-amd.ja.md](image-amd.ja.md)。
- **Apple (Core AI)**: 大きさでグラフが変わる部品 (DiT の組、最後の層、VAE の段) は、大きさごとの関数 (`s{W}x{H}`、`llm.WriteAIFunctions`) を持つ 1 本の `.aimodel`。変わらない部品 (文の符号器、時刻の埋め込み) は関数 1 つ。manifest の `sizes` に大きさごとの部品を書き、`ProgramManifest.At` が `--size` のものを選ぶ。部品の数は大きさに依ってはいけない (`VaeStages` を全部の大きさで確かめる)。[image-apple.ja.md](image-apple.ja.md)、[apple-coreai.ja.md](apple-coreai.ja.md)。

減るもの: Apple の Anima の大きさごとの書き出しは 1 つ 2.9 GB でどれにも重みが丸ごと入り、8 つで 23 GB、Turbo と Aesthetic で 46 GB だった。AMD の FLUX.2 klein は 1:1 と 1:1s で 22.68 → 11.77 GB、8 つで 16.0 GB。ページキャッシュが持つ量も半分。1 枚の時間は変わらない。

払うもの (ダウンロードの重複のほうが重いので受け入れた):

- Core AI は 1 本の `.aimodel` の関数を全部まとめてコンパイルする: 初回は全部の大きさをコンパイルする (M1 の FLUX.2 klein で 512² だけ 10 分、512² と 1024² で 40 分)。prebuilt があれば端末では仕上げだけだが、VAE は prebuilt を使わないので、その大きさはどの機械でも初回にコンパイルする。
- 1 つの大きさを Neural Engine のコンパイラが通さないと、その部品の **全部の** 大きさが NPU から落ちる。取得元に置く前に、チップごとに大きさを 1 つずつオフラインでコンパイルし (`coreai-build compile … --architecture h13g/h14g/h15g/h16g`、そのあと `/usr/bin/log` で `Falling back` を探す)、チップごとの prebuilt (`hwnpu-dev models package MODEL/coreai-ARCH --arch ARCH EXPORT`) は `0 failed` で終わること。
- 大きさを足すと、書き出しと prebuilt を全部作り直す。

### 名前と鍵

| | 名前 |
|---|---|
| AMD | `NAME.amd-npu2`、モデルの名前の小文字: `anima-turbo`、`anima-aesthetic`、`flux.2-klein-4b`、`flux-3-image-klein`。Anima (6.13 GB) は取得元に詰めて置く (`files.tar.zst`)。大きさの名前 (`-512`、`-aspects`) はもうない |
| Apple | 書き出しの名前に `aspect.Tag` で大きさが入る (`8sizes-` と大きさの CRC-32)。大きさの組が違えば別の書き出し |

- 鍵: `chat.PinnedRecipe(model, tool, version)` と書き出しの道具の引数 (`AMDRecipe`)。道具が読む全部のファイルの `sha256:` が models.yaml に要る (固定がなければ共有しない)。道具に渡すものか道具が書くもの (大きさ、形式、並べ方) が変わったら `AMDExportVersion` (FLUX.2 と FLUX.3 klein は 2) を手で上げる。組み込みの `tools/amd` のハッシュも鍵に入るので、そこが変わると作り方の項目は見つからなくなる。名前で取り込んだ書き出し (`iron-import`、`hwnpu-dev sources import`、`chat.ImportedRecipe`) は名前だけで探すので、中身を変えたら名前を変えるか、取得元の古いものを消す。
- 初回が書き出しを取得元に置く (`chat.SharedExportOr`、`?writeback=true` の取得元へ書き戻す)。同じ名前の手元の書き出しに頼まれた大きさがなければ、`NAME.amd-npu2.replaced` に退けて取得元の今のものを取り (なければ IRON で書き出す)、どちらもできなければ戻す。
- 初回にどこから来るか: AMD の FLUX.2 と FLUX.3 klein は 手元 → 取得元 → IRON で書き出す (`PrepareAMD`)。AMD の Anima は手元か取得元。Apple と Intel は取得して変換する。詳しくは [distribution.ja.md](distribution.ja.md)。
- Apple の大きさごとの書き出しは、この順に 1 つの形へ移している: Anima-Turbo → Anima-Aesthetic → FLUX.2 klein (`npu.apple.sizes` の 8 つ) → FLUX.3 klein (Chroma1 は外した)。チップは M1、M2、M3 (h13g、h14g、h15g) から。

## 未解決

- Apple の全部の大きさで 1 つの書き出しへの移行 (上の順)。Apple の FLUX.3 klein は 1024² だけ。
- Apple での編集、FLUX.3 klein の編集。
- 1 枚の電力量: AMD の FLUX.2 と FLUX.3 klein。Mac (`powerbench` は sudo が要る) と Intel の全部のモデル。
- Intel: 形があるのは Anima だけで、実機での実行は確かめていない。

## 採らなかったもの

| もの | 理由 |
|---|---|
| Sana 600M、SANA 1.5 1.6B (外した) | Qwen-Image と一緒にやめた。8 GB の枠は FLUX.2 klein に。`internal/imagegen/sana`、`iron_sana.py`、`iron_dwconv.py` は消し、Anima と FLUX (と、のちに外した Chroma1) が借りていた部分は `tools/amd/iron_dit.py` に移した。`kernels/dwconv3.cc` などのカーネルは残す: チャットの書き出しの鍵が全部のカーネルを含む (`amdtools.LMHash`) ([メモ](../not-adopted/sana.md)) |
| Chroma1-Flash、Chroma1-HD (外した) | M1 のコンパイラが DiT の 8 サイズのプリビルドで落ちる (そのたびに端末で 1 部品 2 時間ほどコンパイルする)。用途は FLUX.2 klein で足り、軽くて速い。`internal/imagegen/chroma`、`iron_chroma.py` は消し、FLUX.1 の VAE は `internal/imagegen/fluxvae` に移した。`iron_chroma_parts.py` は FLUX の AMD の VAE が使うので残す ([メモ](../not-adopted/chroma1.md)) |
| Qwen-Image-2.1、-Turbo (外した) | 7B の DiT と Qwen3-VL-8B: NPU で使うにはパラメータが多すぎる (プログラム 17 GB は Neural Engine のコンパイルのキャッシュを超える。AMD でも 1024² で速くて約 1.5 分) ([メモ](../not-adopted/qwen-image.md)) |
| Sana-Sprint 0.6B / 1.6B (外した) | 2 ステップで速い (1024² で 12〜32 秒) が、絵の質が Sana よりかなり低い。書き出しの道具は `guidance_embeds` のチェックポイントを断る ([メモ](../not-adopted/sana.md)) |
| Anima-Base v1.0 (外した) | 取り柄は LoRA の学習のもとになること。hwnpu は LoRA を読まない。同じ構造・同じ時間の Anima-Aesthetic で足りる。`-m anima:base` は断る (`TestImageMemoryTags`)。`iron_anima.py --dit anima-base-v1.0.safetensors` は今も書き出せる |
| 大きさごとの書き出し、一部の大きさだけのもの | ダウンロードの重複 (上) |
| LongCat-Image-Edit、OmniGen2 (編集のモデルとして) | LongCat は DiT 6.3B と文の符号器 16.6 GB。OmniGen2 は質が下 |
| 参照の K/V を 1 ステップ目から使い回す (FLUX.2 klein 9B-KV の方式を 4B の重みで) | 正しい編集との潜在のコサイン 0.46。その学習をしたモデルでないと使えない |
| First-Block Cache (後ろの組の残差の使い回し) | 条件つきと条件なしを別々にすると CFG がずれを広げ色が抜ける。揃えても 0.02 で 12 % 速く 0.989、0.05 で人物が変わる。CFG を雑音の多いステップだけにするほうが同じ時間でずれが小さい |
| CFG を雑音の多いステップで外す (論文は両端とも外す) | 構図が変わり、色が淡くなる (0.79) |
| 条件つきと条件なしを 1 回の呼び出しにまとめる | GEMM はもう 4096 行ある |
| DPM-Solver++ 15 ステップ | 細部が崩れる (0.920) |
| NPU の横で GPU | 方針: NPU だけ |
