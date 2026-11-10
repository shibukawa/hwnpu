# モデルの定義ファイル (models.yaml) (設計案)

作成: 2026-11-09。状態: 1 段、取得元の切り替え (3 段の一部: path、s3、全体の探索順、sha256)、hwnpu-tts / hwnpu-systemone の自動の準備 (4.5 段の一部) を実装。ほかは設計案。

## 0. 実装したもの (2026-11-09)

- 直下の `models.yaml` (`hwnpu.ModelsYAML` で組み込み) から `chat.Catalog`、`chat.Categories`、`chat.Aliases` を作る (`internal/chat/table.go`)。`models`、`categories`、`aliases` と、モデルの `family`、`tag`、`kind`、`from` (`hf:` だけ、`@REVISION` 可)、`arch`、`machine`、`memory`、`bytes`、`features`、`no_think`、`about`、`like`、`remove`
- ユーザーのファイル (`~/.config/hwnpu/models.yaml`、Windows は `%LOCALAPPDATA%\hwnpu\models.yaml`)、`HWNPU_MODELS`、`-models` を差分として重ねる。用途の `picks` は列ごと。版の扱いは 13 節のとおり (組み込みは厳しく、ユーザーの知らないモデルへの上書きと新しい版の知らないキーは警告)。`sources`、`npus`、`builds`、`cache` は、ユーザーのファイルにあれば「まだ読まない」と警告して飛ばす
- `hwnpu-chat list` と `-list` (hwnpu-transcribe、hwnpu-tts、hwnpu-systemone) に、そのコマンドの種類のモデル、エイリアス、読んだファイル
- hwnpu-transcribe、hwnpu-tts、hwnpu-systemone が `-use` (`transcribe`、`speech`、`decision`、ファミリー、タグ、`OWNER/NAME`) で選ぶ。hwnpu-tts と hwnpu-systemone は自動で落として変換する (`speech.Prepare`、`systemone.Prepare`)。手で変換したものを渡す今までの形も残す
- 3 つとも、変換が済んでいればチェックポイントを落とさず、表 (T5) を書いてそれで動く: hwnpu-transcribe は `tables.safetensors` (デコーダの単語の埋め込み)、hwnpu-tts は `speech-tables.safetensors` (テキスト部の埋め込み、depthformer の表、feedback、マスク)、hwnpu-systemone は `tables.safetensors` (埋め込みと、答えのラベルの出力の行)
- 取得元 (`internal/chat/sources.go`): ユーザーのファイルの `sources` (`type: path` の共有フォルダ、`type: s3` を aws コマンドで。`endpoint`、`region`、`profile`、`no_sign`) と、全体の探索順 `search` (既定 `[hf]`、5.4 節)。`from: SOURCE:NAME` でモデルごとに名指し。`-use SOURCE:NAME` も。ファイルごとに `search` の順に探し、見つかった取得元を次のファイルから先に見る。共有フォルダはハードリンクを試してから写す。リビジョンを固定したモデルは `OWNER/NAME@REV/` を先に探す。`sha256` を書いたファイルは、どの取得元から取っても確かめ、違えば消して次の取得元へ。組み込みのファイルは `sources` と `search` を持たない。まだないもの: `copy: false` (写さずに読む)、オフライン (11.3 節)、ダウンロードしながらの sha256 の計算 (今は取ったあとに読み直す)
  - 確かめたこと: テスト (共有フォルダを先に、なければ HF。`hf` のない探索順は HF に出ない。sha256 の違う写しを飛ばす。固定したリビジョンのフォルダを先に。`SOURCE:NAME` のモデル。偽の aws で S3 の URL、endpoint、profile、no_sign と、ないもの。aws がないときのエラー)。sha256 の確認と探索順を壊すとテストが落ちる。M3 で、whisper-large-v3-turbo の写しを置いたフォルダだけを探索順にして (`search: [share]`)、空のモデルのフォルダから取得、変換、表、書き起こしまで通った (HF への通信なし)
- 同じリポジトリのモデルはダウンロードを共有する (`Model.DirName`: リポジトリの名前。LFM3-Audio-2B と LFM3-Audio-2B-TTS)
- 2 つのリポジトリを読むモデル (OCR の検出器と認識器) は、`from` に主のリポジトリ、`with: {役割: hf:OWNER/NAME@REV}` にほかのリポジトリを書く (`Model.With`、2026-11-11)。`with` のリポジトリのファイルの `sha256` は `役割/ファイル名` のキーで書く。`like` は `with` を引き継ぐ
- 確かめたこと (M3、2026-11-09):
  - hwnpu-transcribe: 表を書いて書き起こし、チェックポイントを退けても同じ結果
  - hwnpu-tts: 初回にダウンロード (4 GB、6 分半)、変換 4 本、表 (0.67 GB) まで自動で通った。「こんにちは」(1.6 秒) と「今日はいい天気ですね。散歩に行きましょう。」(5.0 秒を 3.3 秒で生成) を hwnpu-transcribe で書き起こすと、どちらも元の文。チェックポイント (本体とコーデック) を退けても読み上げ、読み込み 3.2 秒
  - hwnpu-systemone: d1-2B の変換 (lut8、512 トークン、2 部分) と視覚エンコーダ、表 (0.52 GB) まで自動で通った。「二重に請求された、返事がない、怒っている」問い合わせに、担当 billing (0.98)、怒っているか 0.996、緊急度 1 (やや緊急、0.45)。チェックポイントを退けても同じ答え、読み込み 1.1 秒
- 画像を読む用途は `vision` だけにし、`image` は別名にしない (画像生成の用途 `image` のために空けておく)
- チャットのモデルの動かし方 (`prefill`、`format`、`context`、`paged`) と、機械のメモリごとの違い (`variants`) (2026-11-11、6.2 節)。hwnpu-chat のフラグ (`--prefill` など) を与えなければ、モデルのこの欄が既定になる


モデルの選び方、取得元、NPU ごとのビルド設定を、コードから 1 つの YAML に移す。既定のファイルはバイナリに組み込み、利用者はユーザーフォルダに置いたファイルで上書きする。

## 1. 目的

- **選び方をデータに**: 今は `chat.Categories` (用途ごとのおすすめ) と `chat.Catalog` (モデル、タグ、`Machine`) が Go のコード。MODEL.md の表と手で合わせている
- **取得元を差し替えられるように**: 今は Hugging Face だけ。社内の蒸留モデルに差し替える、共有フォルダや S3 (rustfs などの互換実装を含む) から取る、ができるようにする
- **変換済みを優先し、なければ今と同じく手元で変換する**: ネイティブの変換済み (docs/prebuilt.md) があればそれを落として読む。なければ、元のチェックポイント (Hugging Face のベース、独自のモデル) を落として手元で変換する。過渡期の今も、配布のない独自のモデルでも動くように、手元の変換は残す。変換したものを取得元に書き戻すことはしない (9 節)
- **ビルド設定を 1 か所に**: 今のプログラム名は 形式 × 文脈 × Q × last × 塊 × 段階 で、その既定はあちこちにある (2 節)。利用者が扱う軸は **NPU × モデル (ファミリーと大きさ) → 設定** だけにし、Q=128、最後の行だけ、`-kv`、`state-rows` のような導出されるものはコードに残す

## 2. 今の既定の置き場所 (移すもの)

| 設定 | 今の置き場所 | 今の値 |
|---|---|---|
| 用途ごとのおすすめ | `internal/chat/tags.go` `Categories` | chat: 8 GB LFM3-1.7B、16 GB Gemma 4 E2B、24 GB E4B など |
| モデル、タグ、推奨の機械 | `internal/chat/catalog.go` `Catalog` | `plamo3translate:0.8b` は 8 GB など |
| 重みの形式 | `llm.Arch.Format`、`Formatter` (チェックポイントで変わるもの)、`Arch.IntelFormat`、`llm.IntelFormat` | Gemma 4 lut6、PLaMo 3 lut8、Qwen3.5 lut8 (2B だけ lut6)、LFM3 lut6 (多モーダルは lut8)、無指定は lut4。Intel は lut4 → int4、lut6 / lut8 → int8、LFM3 は int8 |
| 文脈 | `Setup.Context` | 0: 分けられるモデルは学習時の最長、それ以外は 16320 |
| 塊 | `Setup.Paged` | 分けられるモデルは 8192 |
| 下書き | hwnpu-chat の `--draft` | 8 |
| プロンプト用のプログラム | `PrefillAuto`、`compileCacheBytes`、`neuralEngineHold` | コンパイルのキャッシュ 3.5 GiB を超えるなら持たない。M1 は 2.8 GiB |
| 段階 | `autoTiers` | lut4 か Intel なら 4096、それ以外はなし |
| MTP | `Setup.MTP` | auto |
| AMD の書き出し | `chat.AMDExport` | `~/models/NAME.amd-npu2-q4`、なければ `.amd-npu2` |

## 3. ファイルの置き場所と重ね方

読む順 (あとのものが前のものを上書き):

1. 組み込みの既定: リポジトリの直下の `models.yaml` (MODEL.md の隣。PR を出す人が見つけやすいように)。直下のパッケージ `hwnpu` が `go:embed` で組み込む
2. ユーザー: macOS と Linux は `~/.config/hwnpu/models.yaml` (`$XDG_CONFIG_HOME` があればその下。macOS でも同じ場所にする。`os.UserConfigDir` の `~/Library/Application Support` は手で置くには深い)、Windows は `%LOCALAPPDATA%\hwnpu\models.yaml` (共有フォルダのドライブ文字や `aws` のプロファイルなど機械ごとの値が入るので、ローミングしない方に置く)
3. 環境変数 `HWNPU_MODELS` (`:` 区切り、Windows では `;`) と、hwnpu-chat / hwnpu-chat agent の `--models FILE` (何度でも指定できる)

重ね方: ユーザーのファイルには**差分だけ**を書き、既定と合成する。既定のファイルを写して直す必要はない。

- `models` と `sources` と `npus`: 名前ごとに欄単位で上書き。`plamo-3-translate-0.8b: {from: ...}` と書けば、取得元だけが変わり、ほかの欄は既定のまま。`remove: true` で消す
- `aliases`: 名前ごとに足す (5.1 節)
- `categories`: `picks` は列 (`machine`) ごとに上書き。`{machine: 8, model: X}` と書けば 8 GB の列だけが変わる。`{machine: 8, remove: true}` でその列を消す
- `builds`: 足していく。既定のルールより具体的か、同じ具体さであとにあるので、書いた欄だけが勝つ (6 節)
- コマンドラインの `-format` などは、どのファイルよりも強い

ユーザーのファイルの例 (蒸留モデルに差し替え、M1 だけ設定を変える):

```yaml
version: 1
sources:
  lab-s3: {type: s3, bucket: models, endpoint: http://rustfs.lab:9000, profile: rustfs}
models:
  plamo-distill-0.8b: {like: plamo-3-translate-0.8b, from: lab-s3:plamo-distill-0.8b}
aliases:
  plamo-3-translate-0.8b: plamo-distill-0.8b
builds:
  - {npu: apple-m1, family: plamo3translate, prefill: off}
```

各ファイルの頭に `version: 1` を書く。知らない版、知らないキーはエラーにする (厳しく読む)。

## 4. ファイルの形

```yaml
version: 1

# 取得元。models の from は「取得元の名前:場所」と書く。hf は組み込み
sources:
  team-share:
    type: path
    path: /Volumes/share/models       # その下の NAME/ を読む
    copy: false                        # チェックポイントは写さずに読む (変換したものは手元のキャッシュへ)
  lab-s3:
    type: s3                           # aws コマンドで取る (5 節)
    bucket: models
    prefix: hwnpu/
    endpoint: http://rustfs.lab:9000   # 互換実装。省けば AWS
    region: us-east-1
    profile: rustfs                    # ~/.aws/config のプロファイル。省けば既定の認証
    # no_sign: true                    # 公開のバケット

# NPU ごとの性質 (ビルドの判断に使う上限)
npus:
  apple:    {compile_cache: 3.5GiB}
  apple-m1: {hold: 2.8GiB}             # これを超えて一緒に読むと 1 回ごとに約 12 倍遅い (PROBE.md)

models:
  plamo-3-translate-0.8b:
    family: plamo3translate
    tag: 0.8b
    size: 0.8                          # パラメータ数 (B)。省けば config.json から数える (6.1 節)
    kind: chat
    features: [translation]
    from: hf:pfnet/plamo-3-translate-0.8b
    bytes: 1.79e9
    memory: 2.4                        # 動かしたときのメモリ (GB)。MODEL.md の「開始時」
    about: "Preferred Networks' PLaMo 3 Translate 0.8B: ..."
  PP-OCRv6-small:
    family: ppocr
    tag: small
    kind: ocr
    from: hf:PaddlePaddle/PP-OCRv6_small_rec_onnx@b8f84f0b...   # 認識器
    with: {detector: "hf:PaddlePaddle/PP-OCRv6_small_det_onnx@28fe5895..."}   # ほかに読むリポジトリ (役割: hf:OWNER/NAME)
    sha256:
      inference.onnx: 5435fd74...
      detector/inference.onnx: d73e0058...   # with のリポジトリのファイルは 役割/ファイル名
  # ...

# 用途ごとのおすすめ (MODEL.md の行)。machine は MODEL.md の列 (GB)
categories:
  translate:
    aliases: [translation]
    about: translation
    picks:
      - {machine: 8,  model: plamo-3-translate-0.8b}
      - {machine: 16, model: plamo-3-translate-2b}

# ビルド設定: NPU × モデル → 設定。1 行 1 ルール
builds:
  - {npu: apple, format: lut4, paged: 8192, draft: 8, prefill: auto, tiers: auto, mtp: auto}
  - {npu: intel, format: int8, tiers: [4096]}
  - {npu: apple, family: lfm3, format: lut6}
  - {npu: apple, family: [lfm3-vl, lfm3-audio, lfm3-omni], format: lut8}   # lut4/6 で絵と音を取り違える (16.31)
  - {npu: apple, family: gemma4, format: lut6}
  - {npu: apple, family: [plamo3translate, qwen3.5], format: lut8}
  - {npu: apple, family: qwen3.5, size: 2, format: lut6}                    # 8 ビットと同じ質 (16.99)
  - {npu: apple, family: [plamo-embedding], format: lut8}                   # 4 ビットで崩れる
  - {npu: apple-m1, family: lfm3, prefill: off}                             # 例: M1 だけ変える
  - {npu: amd-xdna2, format: q4}
  - {npu: amd-xdna2, family: qwen3.5, size: "<=2", format: q6}
  - {npu: intel-npu3, format: int4}                                       # 例: 古い世代だけ変える
```

### 4.1 NPU の名前

製品名 (CPU の型番、シリーズ番号) ではなく、**NPU の世代**で呼ぶ。製品名は世代と揃っていないため:

- AMD: Ryzen 7040 / 8040 (Phoenix、Hawk Point) は XDNA、Ryzen AI 300 (Strix Point、Krackan Point) と Ryzen AI Max (Strix Halo) は XDNA 2。ところが 2025 年の Ryzen 200 は Hawk Point の名前の付け替えで、番号は新しくても XDNA
- Intel: Core Ultra シリーズ 1 (Meteor Lake) は NPU 3。シリーズ 2 は、Lunar Lake (200V) が NPU 4 なのに、Arrow Lake (200H、200U、200S) はシリーズ 1 と同じ NPU 3。シリーズ 3 (Panther Lake) は NPU 5

| 名前 | 当てはまるもの | 見分け方 |
|---|---|---|
| `apple` | Apple の Neural Engine 全部 | |
| `apple-m1` `apple-m2` `apple-m3` `apple-m4` … | その世代 (Pro、Max、Ultra を含む) | `machdep.cpu.brand_string` (`cpuBrand`) |
| `amd` | AMD の NPU 全部 | |
| `amd-xdna1` | XDNA: Phoenix、Hawk Point (Ryzen 7040、8040、200) | amdxdna の PCI ID (1502) |
| `amd-xdna2` | XDNA 2: Strix Point、Krackan Point、Strix Halo (Ryzen AI 300、Ryzen AI Max) | amdxdna の PCI ID (17f0) |
| `intel` | Intel の NPU 全部 | |
| `intel-npu3` | Meteor Lake、Arrow Lake (Core Ultra シリーズ 1、シリーズ 2 の 200H / U / S) | intel_vpu の PCI ID、OpenVINO の `DEVICE_ARCHITECTURE` (3720) |
| `intel-npu4` | Lunar Lake (Core Ultra 200V) | 同 (4000) |
| `intel-npu5` | Panther Lake (Core Ultra シリーズ 3) | 同 |

- 別名として、IRON の呼び方 `amd-npu1` / `amd-npu2` と、開発の名前 (`strix`、`lunarlake` など) も受け付ける。`-list` や `-explain` が出すのは上の名前だけ
- 世代を見分けるコードは、Apple のほかはまだない (`hwnpu.Enumerate` は会社しか返さない)。2 段で足す
- 配布 (docs/prebuilt.md) のサブフォルダの名前も、これに揃える (`amd-npu2/` → `amd-xdna2/`)。書き出しの中身 (IRON の `npu2` のデバイス) はそのまま

「LFM3 の M2 以降の既定はこれ、M1 はこれ」は、`npu: apple` の行と `npu: apple-m1` の行の 2 行で書く。広い名前の行が既定、狭い名前の行が例外になる (6 節)。

## 5. 取得元

| type | 書き方 | 取り方 |
|---|---|---|
| `hf` (組み込み) | `from: hf:OWNER/NAME@COMMIT` (5.3 節) | 今の `fetch`。`HF_TOKEN` を使う |
| `path` | `from: team-share:plamo-distill-0.8b` | `path` の下のフォルダ。`copy: false` なら写さずに読む (NFS や SMB の共有フォルダ) |
| `s3` | `from: lab-s3:plamo-distill-0.8b` | `aws s3 sync s3://BUCKET/PREFIX/plamo-distill-0.8b/ DIR` を呼ぶ |

S3 は `aws` コマンド (v2) に任せる。認証、リージョン、再試行、マルチパートを自前で持たないため。

- `endpoint` は `--endpoint-url`、`profile` は `--profile`、`region` は `--region`、`no_sign` は `--no-sign-request` で渡す。ほかは `aws` の通常の設定 (`AWS_*` の環境変数、`~/.aws/config`) がそのまま効く
- rustfs や MinIO のようにパス形式のアドレスが要る実装は、`~/.aws/config` のそのプロファイルに `s3 = addressing_style = path` を書く (`aws` にコマンドラインの指定がないため)。docs に手順を書く
- `aws` が見つからないときは、`from` に S3 を使っているモデルを選んだときだけ、入れ方と一緒にエラーを出す
- 取るのは、HF のリポジトリと同じ並び (`config.json`、トークナイザー、safetensors) か、変換済みの並び (`hwnpu.json` があるもの、docs/prebuilt.md 4 節)。`hwnpu.json` があれば変換済みとして読む

### 5.1 エイリアス (蒸留モデルへの差し替え)

2 つの書き方がある。

```yaml
# (a) 同じモデルのまま取得元だけ差し替える
models:
  plamo-3-translate-0.8b:
    from: lab-s3:plamo-distill-0.8b

# (b) 別のモデルを足し、エイリアスで差し替える
models:
  plamo-distill-0.8b:
    like: plamo-3-translate-0.8b       # tag と size 以外の欄 (family、kind、features、about ...) を引き継ぐ
    from: team-share:plamo-distill-0.8b
aliases:
  plamo-3-translate-0.8b: plamo-distill-0.8b
```

- `aliases` の左の名前が選ばれるところ (用途の `picks`、`plamo3translate:0.8b`、`-use plamo-3-translate-0.8b`、API の `model`) では、どこでも右のモデルを使う。`translate` の表を書き直さなくてよい
- エイリアスは 1 段だけ (右の名前をさらに置き換えない)。左右に知らない名前があればエラー
- `-list` と起動のログには「plamo-3-translate-0.8b → plamo-distill-0.8b (~/.config/hwnpu/models.yaml)」と出す
- 読み込むときにも 10.1 節のうち安いもの (アーキテクチャ、config、トークナイザー、会話の型、特別なトークンの id) を確かめ、合わなければ何が違うかを出して止める。黙って崩れた答えを返さないため
- `like` で足したモデルは `tag` を持たないので、ファミリーのタグの選択には出ない (エイリアスか名前で選ぶ)。`tag` を書けば出る

ファミリーと大きさが同じなら、`builds` のルールもそのまま当たる。

取得元が変わったときに、前の取得元で変換したプログラムを読まないように、モデルのフォルダに `source.json` (取得元、コミット、ファイルごとの sha256、取った日時) を書く。違えば変換し直す (5.3 節)。

### 5.2 どこから持ってくるか (読み込みの順)

モデルには取得元を 2 つ書ける。

```yaml
models:
  LFM3-1.7B:
    from: hf:LiquidAI/LFM3-1.7B          # 元のチェックポイント (変換の元)
    prebuilt:                            # ネイティブの変換済み (省けばなし)
      from: hf:hwnpu/LFM3-1.7B
      npus: [apple, amd-xdna2]           # 配っている NPU。選ぶときにネットを見ないで済むように書く
```

使う NPU と、6 節で決まった設定について、上から順に探す。

1. **手元にあるもの**: 前に落としたか変換したプログラムが、設定と合えばそれを使う (今の `.done` と同じ)
2. **ネイティブの変換済み**: `prebuilt` がその NPU を配っていて、`hwnpu.json` のその組の設定と `format_version`、元のチェックポイントのハッシュ (5.3 節) が合えば、その NPU のサブフォルダだけを落とす。元のチェックポイントは落とさない
3. **手元で変換**: `from` のチェックポイントを落として、今と同じに変換する。Apple と Intel は Go の変換器で。AMD は IRON の道具 (mlir-aie、Peano、Python) が要るので、入っていればそれで書き出し、なければ 4 へ
4. **使えない**: 何が足りないか (配っていない、道具がない) を出して止める

- 2 で合わないのは、ユーザーの `builds` で形式などを変えたとき、hwnpu が古くて `format_version` を読めないとき。`-explain` と起動のログに、変換済みを使わなかった理由を出す
- `-prebuilt auto` (既定) / `off` (常に手元で変換。変換器を確かめるとき) / `only` (手元で変換しない。変換に時間をかけたくない機械、CI)
- `from` に変換済みの並び (`hwnpu.json` があるもの) を書けば、`prebuilt` を書かなくてもそれを読む。元のチェックポイントを渡さない独自のモデル向け。この場合は 3 がない
- **選ぶとき (用途やファミリーのタグ) は、この NPU で使えるモデルだけを候補にする**: `prebuilt.npus` に入っているか、手元で変換できる (アーキテクチャがその NPU の変換に対応している、AMD なら道具がある) もの。8 GB の AMD の機械で 0.8B の書き出しも道具もなければ、次の候補に落ちる。`-list` には「この NPU では使えない」と理由を出す

### 5.3 リビジョンとハッシュで固定する

同じリポジトリでも、リビジョンが変われば構造まで変わることがある (`config.json` の層や語彙、トークナイザー、会話の型)。黙って違う重みで変換したり、前の重みで作った変換済みを新しい重みに当てたりしないよう、安全側に倒して**コミットとファイルのハッシュの両方**で固定する。

```yaml
models:
  LFM3-1.7B:
    from: hf:LiquidAI/LFM3-1.7B@6c1f0e3a9b...   # コミット (40 桁)。ブランチやタグは書かない
    sha256:                                     # 落とすファイル全部
      config.json: 9b2e...
      tokenizer.json: 41d0...
      model.safetensors: e7a5...
```

- **組み込みのファイル**: すべてのモデルにコミットと `sha256` を書く。`go test` で、ないもの、コミットが 40 桁でないものを落とす
- **確かめ方**: 落としながら sha256 を計算し (読み直さない)、合わなければファイルを消して止める。HF の LFS のファイルは ETag が sha256 なので、落とす前にも食い違いが分かる。S3 の ETag は分割アップロードで sha256 にならないので、こちらは落としながらの計算だけを使う
- **構造の変化**: `config.json`、トークナイザー、会話の型のハッシュが変われば、別のモデルとして扱う。変換済み、表 (`tables.safetensors`)、GPTQ の符号は、元のチェックポイントのハッシュと一緒に記録し、合わなければ使わない
- **ネイティブの変換済み**: `hwnpu.json` に、作った元のコミットとハッシュを書く (docs/prebuilt.md 4 節)。定義ファイルの固定と合わなければ使わない (5.2 節の 2 を飛ばして 3 へ)
- **固定を上げる**: 上流が更新されたら、ビルド・アップロードツール (9 節) か skill (10 節) が新しいコミットのハッシュを取り、量子化の KL を測り直してから、定義ファイルを直す PR を出す。配布のビルド番号もそこで上げる
- **ユーザーのファイル**: 固定しない `from: hf:OWNER/NAME` も書ける (手元で試すとき)。そのときは初めて落としたときのコミットとハッシュを `source.json` に書き、以後はそれを使い続ける。上流が更新されても勝手には変わらない。新しくするのは `-update NAME` で頼んだときだけ。起動のログに「固定していない」と出す
- **今ある手元のキャッシュ**: `source.json` のないモデルのフォルダは、初回に一度だけ手元のファイルの sha256 を計算し (10 GB で数十秒)、定義ファイルと合えばそのまま使う。合わなければ変換し直す
- **path と s3**: コミットはないので `sha256` だけで固定する。書いていなければ、ユーザーの HF と同じく初回の値を `source.json` に記録する

### 5.4 全体の探索順 (ミラー)

モデルごとの `from` を書き換えなくても、全部のモデルを先に共有フォルダや S3 で探し、なければ Hugging Face から取れるようにする。社内の共有フォルダや S3 に HF の写しを置いておく使い方向け。

```yaml
sources:
  team-share: {type: path, path: /Volumes/share/hf}
  lab-s3: {type: s3, bucket: models, prefix: hf/, endpoint: http://rustfs.lab:9000, profile: rustfs}
search: [team-share, lab-s3, hf]      # 全体の探索順。既定は [hf]
```

- `from: hf:OWNER/NAME` のモデルは、`search` の順に `OWNER/NAME/` を探す (共有フォルダなら `/Volumes/share/hf/OWNER/NAME/`、S3 なら `s3://models/hf/OWNER/NAME/`)。最初に見つかったところから取る。`hf` が並びにあれば、最後に Hugging Face から
- 置き方は HF のリポジトリと同じ並び (`config.json`、トークナイザー、safetensors)。リビジョンを固定したモデルは `OWNER/NAME@COMMIT/` を先に探す
- 安全: ミラーから取ったファイルも、定義ファイルの sha256 (5.3 節) で確かめる。合わなければそのミラーを飛ばして次を探し、ログに出す。ミラーが古い版を持っていても、違う重みで変換することはない
- ネイティブの変換済み (5.2 節) も同じ順に探す (`OWNER/NAME` の代わりに `prebuilt.from` の名前で)
- `-offline` (11.3 節) では、`search` のうち `path` のものだけを見る
- `search` から `hf` を外せば、Hugging Face に出ない (社内のミラーだけで動かす)
- 書くのはユーザーのファイル (差分)。組み込みの `models.yaml` は `search` を書かない (既定の `[hf]`)
- モデルごとの `from: team-share:NAME` (5 節) は、そのモデルだけ別のところから取るときに使い、`search` より優先する

## 6. ビルド設定の当てはめ方

ルールの条件:

- `npu`: 4.1 節の名前 (1 つか並び)
- `model`: モデルの名前
- `family`: ファミリー (1 つか並び)
- `size`: パラメータ数。`2`、`"<=2"`、`">4"`

設定:

- `format`
- `format_for`
- `codes`: GPTQ の符号の取得元。`from` と同じ書き方
- `context`
- `paged`
- `draft`
- `prefill`
- `tiers`
- `mtp`

条件に当たるルールを全部集め、**具体的なものから順に欄を埋める** (CSS の詳細度に近い)。

1. 具体さは、条件の欄ごとの点を足したもの: `model` 8、`size` 4、`family` 2、`npu` は狭い名前 (`apple-m1`) 2、広い名前 (`apple`) 1
2. 同じ具体さなら、あとに書いたもの (ユーザーのファイルは既定のあと) が勝つ
3. 欄ごとに決める。`apple-m1` の行が `prefill` だけを書けば、`format` は `apple` の行のまま

`hwnpu-chat explain NAME` (案) で、決まった設定と、それぞれがどのファイルのどの行から来たかを出す。

コードに残すもの: Q (8、64、128)、最後の行だけ、`-kv`、`state-rows`、`hidden-out`、エンコーダの fp16。これらはモデルの構造と上の設定から決まる。

### 6.1 大きさ (size)

重みの形式は大きさで変わる (Qwen3.5 は 0.8B と 4B が lut8、2B が lut6。小さいモデルほど 4 ビットで崩れやすい)。それで、ルールは大きさごとに書けるようにし、どのモデルにも大きさが決まるようにする。

- **数え方**: NPU に載せる重みの数。`config.json` の `llm.Model.Tensors()` の形を掛けて足し、CPU に残す表 (`llm.HostTables`: 単語の埋め込み、Gemma 4 の層ごとの埋め込み) を除く。単位は B (10 億)
  - `config.json` はチェックポイントより先に落とすので、数えるのにチェックポイントは要らない
  - 表を除くのは、形式が効くのが NPU の重みだけだから。Gemma 4 E2B は表を入れると約 5B だが、NPU の重みは約 2B で、名前の「E2B」とも合う
  - 視覚や音声のエンコーダは、それぞれのプログラムの大きさとして別に数える (今は fp16 で固定なので使わない)
- **書いた値が先**: `models` の `size` があればそれを使う。ルールを読む人が名前と同じ数で考えられるようにするため。既定のファイルには全モデルの `size` を書き、数えた値との差が 2 割を超えたら `go test` で落とす (書き間違いに気づくため)
- **書いていなければ数える**: ユーザーが足したモデル、`OWNER/NAME` で直接指定した HF のリポジトリ、蒸留モデル。`like` は `size` を引き継がない (蒸留で大きさが変わるため)
- **当てはめ**: `size: 2` は 2 の前後 1 割 (1.8〜2.2)。`"<=2"`、`">4"`、`"1-3"` は範囲。どれも数えた値と書いた値のどちらでも同じに扱う

コードにある大きさの判断 (`qwen35.Config.Format` の「hidden が 2048 なら lut6」) は、この `size` のルールに移す。

### 6.2 機械のメモリごとの違い (variants、実装済み)

`builds` は NPU とモデルで決める。それとは別に、同じモデルを**機械のメモリ**で動かし分けたいことがある。Qwen3-1.7B は、プロンプトを読むプログラムを別に持つと Neural Engine に 3.26 GiB とプロセスの最大 4.1 GB で約 7.6 GB になり、8 GB の機械には重い。生成のプログラムにプロンプトも読ませれば 1.63 GiB と 2.8 GB の約 4.5 GB で載る (新しいプロンプトを読むのは約 5 倍遅い)。そこで、モデルの欄に動かし方を書き、`variants` に、より大きなメモリの機械での差分を書く:

```yaml
  Qwen3-1.7B:
    machine: 8          # この欄は 8 GB の機械のもの
    memory: 4.5
    prefill: off
    variants:
      16: {memory: 7.6, prefill: auto}   # 16 GB 以上
```

- 動かし方の欄は `prefill` (auto、on、off)、`format` (重みの形式)、`context` (トークン)、`paged` (塊のトークン、-1 は分けない)。hwnpu-chat の同じ名前のフラグの既定になり、フラグを与えればフラグが勝つ。チャットのモデルだけが持てる
- `variants` のキーは機械のメモリ (GB)。機械が持つ最大のキーの差分を 1 つだけ、モデルの欄の上に重ねる (16 と 32 があれば、32 GB の機械では 32 の差分だけ)。差分が書けるのは `memory` と動かし方の 4 つで、書いた欄だけが変わる。キーは `machine` より大きくなければならない (`machine` の欄がモデル自身)
- どの機械の差分か: 名前で選んだとき (`-m Qwen3-1.7B`) はこの機械のメモリ (`HWNPU_MEMORY` で上書き)、ファミリーと用途では選ぶときの機械 (`qwen3:16g` なら 16 GB)。ファミリーで同じ列の候補を比べる `memory` も、その機械の差分のもの (8 GB で、Qwen3-0.6B の 3.9 GB と Qwen3-1.7B の 4.5 GB から 1.7B)
- `hwnpu-chat list` の MEMORY はこの機械の差分のもの。`chat.Lookup` と `chat.Catalog` は差分を当てる前のモデル、`Model.On(machine)` が当てたもの
- ユーザーのファイルは、名前を書いた差分だけを替え、ほかは残す (`device` と同じ)。`like` は動かし方と差分を引き継ぐ
- `builds` (6 節) を実装するときは、`builds` の欄がモデルの欄と差分より先に来る (NPU ごとの事情の方が具体的)。今は NPU で分ける必要がない (Qwen3-1.7B は Apple だけ)

## 7. ほかの部分とのつながり

- **ほかのコマンドも同じ表で選ぶ**: 用途に `transcribe`、`speech`、`decision`、`image` を足し、モデルの `kind` で、どのコマンドが読むかを分ける (`-use chat` を hwnpu-transcribe に渡せばエラー)。MODEL.md の表は全部の行が定義ファイルから来る。コマンドごとの今の形と、要る作業:

| コマンド | 今 | 要る作業 | 段 |
|---|---|---|---|
| hwnpu-chat、hwnpu-chat agent (埋め込みを含む) | `-m` で名前、用途、タグ。自動で落として変換 | 定義ファイルから読むだけ | 1 |
| hwnpu-transcribe | `-use OWNER/NAME` (既定 whisper-large-v3-turbo)。自動で落として変換 (`transcribe.Prepare`) | `-use` を同じ解決 (用途 `transcribe`、ファミリー `whisper`) に。kotoba-whisper (Apache-2.0) を日本語のタグに。表の書き出し (T5) と、チェックポイントを遅れて落とす仕組みを chat と揃える | 1 の続き |
| hwnpu-tts | `-model DIR` に手で変換したものを渡す (CHAT.md 7 節の手順) | 自動の準備 (`speech.Prepare`: テキスト部、depthformer、コーデックの変換) を作り、`-use` に。T5 (テキストの埋め込み、depth の表) も。用途 `speech` | 3 のあと |
| hwnpu-systemone | `-model DIR` に手で変換したもの | 自動の準備 (d1 の本体、視覚と音声のエンコーダ) と `-use`。T5 (埋め込み、ラベルの行)。用途 `decision` | 3 のあと |
| hwnpu-image | `-m` で名前、ファミリー (`sana`、`moge`)、用途 (`image`、`depth`)。AMD は `~/models/NAME.amd-npu2`、Mac / Intel の Anima は自動で落として変換 | AMD の書き出しを自動で作る (Python の iron_*.py) か取得する | 済み (2026-11-10、design.md 16.124) |
| hwnpu-dev formats | `-use` で chat の名前 | chat と同じ解決になる。候補の形式を `builds` から取る (10.2 節で skill が使う) | 2 |
| hwnpu-dev convert、hwnpu-dev check、hwnpu-dev generate | `-config`、`-weights`、`-package` でファイルを直に | 開発用なので、ファイルを直に渡す形は残す。hwnpu-dev convert には `-use NAME -npu NPU` を足し、`builds` で決まる設定で変換する (ビルドツールの土台) | 4 |

- hwnpu-tts と hwnpu-systemone は、今は実行時もチェックポイント (`model.safetensors`) を開く。T5 を入れるまでは、11.4 節のチェックポイントの削除の対象から外す
- **MODEL.md の表**: `categories` と `models` の `memory` から作れる。`go test` で、MODEL.md の表と定義ファイルが食い違っていないかを確かめる (表の生成は後回し)
- **hwnpu-dev convert**: 今は `Arch.Format` を既定にしている。同じ `builds` を読み、`-npu` で当てはめるようにする。`Arch.Format` と `Formatter` は最後に消す
- **配布 (docs/prebuilt.md)**: `prebuilt/models.yaml` の `targets` は、ここの `builds` で NPU ごとに決まる 1 組と同じものにする。`prebuilt/models.yaml` には、配るモデル、取得元のリビジョン、更新番号 `build` だけを残す
- **AMD**: `amd-xdna2` の `format` (q4、q6) から書き出しの名前 (今の `NAME.amd-npu2-q4`) を決める。配布が入れば、取得元から落とす

## 8. 実装の順

| 段 | 中身 | 変わること |
|---|---|---|
| 1 | `models.yaml` を組み込み、`models` と `categories` を移す。ユーザーのファイルと `-models` を差分として重ね、`aliases` を入れる。`-list` に出どころ | Go の `Catalog` と `Categories` が YAML から作られる |
| 2 | `builds` と `npus` を入れ、2 節の既定を移す。AMD と Intel の世代を見分ける。`-explain` | `DefaultFormatFor`、`autoTiers`、`neuralEngineHold`、`-draft` の既定がルールになる |
| 3 | `sources` (path、s3) と `like`、`source.json`、コミットと sha256 の固定 (5.3 節) | 蒸留モデル、共有フォルダ、S3。上流の更新で黙って変わらない |
| 3.5 | 変換済みの並び (`hwnpu.json`) と `prebuilt` を読み、5.2 節の順で探す。`-prebuilt` | docs/prebuilt.md の配布が始まれば、ダウンロードと読み込みだけになる |
| 4 | hwnpu-dev convert と配布の `targets` を同じルールに | 7 節 |
| 4.5 | ロック、オフライン、キャッシュの削除、Windows の置き場所 (11 節)。hwnpu-tts と hwnpu-systemone の自動の準備と `-use` (7 節) | 1 段のあとならいつでも。ロックは 3 段 (取得元を足す) より前に |
| 5 | ビルド・アップロードツール (`cmd/hwnpu-build`) | 9 節。aix1 の CI がこれを呼ぶ |
| 6 | 定義ファイルを作る skill | 10 節。`-explain` (2 段) と `like` (3 段) が要る |
| 7 | 素直なデコーダの型 `decoder` (12.2 節)。Anima の Qwen3 を取り出し、Qwen3-0.6B で確かめ、ChatML の会話の型を足す | 1 段のあとならいつでも。Llama 型の蒸留モデルが定義ファイルだけで使える |

依存: YAML は `github.com/goccy/go-yaml` (依存なし、保守されている。`gopkg.in/yaml.v3` はアーカイブ済み)。厳しく読む設定 (`yaml.Strict()`、`DisallowUnknownField`) で読む。

## 9. ロードの側に持たないもの

- **変換したものの書き戻し**: hwnpu-chat などのロードの側は、取得元を読むだけにする。手元の変換 (5.2 節の 3) は今と同じに残すが、その結果は手元のキャッシュに置くだけで、取得元には上げない。変換して配るのは別のビルド・アップロードツール (仮に `cmd/hwnpu-build`) の役目
  - 同じ `models.yaml` を読み、`builds` で NPU ごとに決まる 1 組を変換し、照合して、取得元 (Hugging Face、共有フォルダ、S3) に上げる
  - docs/prebuilt.md の aix1 の CI は、このツールを呼ぶ
  - 社内で蒸留モデルを配る人も、同じツールで共有フォルダや S3 に変換済みのものを置ける。ロードの側は `hwnpu.json` があれば変換済みとして読む (5 節)

## 10. 定義ファイルを作る支援 (skill)

蒸留モデルや別のチェックポイントを足すとき、`models.yaml` の差分を作るのを手伝う skill (`.claude/skills/tune-model/`、仮の名前) を置く。手順は docs/new-model-checklist.ja.md の 5 節 (量子化) と 6 節 (メモリ) を、1 つのモデルの差分を作る形にまとめたもの。KL などを見ながら、許容誤差とメモリの条件で形式を決める。

### 10.1 使えるかどうか (最初に確かめる)

移植済みのアーキテクチャとほぼ同じなら、そのまま使える。層の数、幅、ヘッドの数、語彙の大きさ、文脈の長さが違うだけなら、`config.json` から組み立てるので変換もそのまま通る。使えないのは、コードを足さないと動かないものだけ。

| 確かめるもの | どう確かめるか | 使えないとき |
|---|---|---|
| アーキテクチャ | `llm.Detect` (`model_type`、`architectures`) | 新しい構造。port-architecture の skill で移植する |
| config の設定 | `llm.LoadConfig` (知らない設定はエラーになる。例: 別の RoPE の拡張、別の注意の窓) | 設定を足すのも port-architecture の範囲 |
| テンソル | `hwnpu-dev check` の `tensors: 0 missing, 0 mismatched` | 名前の付け方が違う (別の学習の道具で保存した、など) |
| トークナイザー | `tokenizer.Load` が読めるか。今読めるのは BPE (GPT-2 式の ByteLevel、GPT-4 / Qwen3 / Qwen3.5 の区切り、byte fallback) と PLaMo の `tokenizer.jsonl` | SentencePiece の unigram、独自の正規化や前処理、Python のコードで実装したトークナイザー (`trust_remote_code`)。トークナイザーの実装を足す必要がある |
| 会話の型 | 元のモデルの `chat_template` と同じか (`tokenizer_config.json` か `chat_template.jinja` を比べる)。hwnpu はアーキテクチャごとの型を Go で持つ (`chat.TemplateFor`) | 型が違えば、答えが崩れるか、ツールの呼び出しが読めない。違いを報告して止める |
| 語彙 | 特別なトークン (BOS、EOS、`<think>`、ツールの呼び出しの印) が元のモデルと同じ id か | 違えば会話の型と同じ扱い |

### 10.2 形式を決める

1. 聞くこと: 使う NPU、載せたい機械のメモリ (MODEL.md の列)、許容誤差 (既定は 5.4 節の目安: KL 0.05 未満なら 4 ビット、参照と同じ次のトークン 90% 以上)、用途 (チャット、翻訳、埋め込み)
2. 大きさを数え (6.1 節)、既定の `builds` で何が選ばれるかを `-explain` で見る。元のモデルと同じファミリーと大きさなら、多くはそのまま当たる
3. 候補の形式を、軽いものから順に測る
   - CPU: `q4_gptq.py --method rtn` で部分ごとのビット数を掃く (5.2 節)
   - NPU: `hwnpu-dev formats` (参照との次のトークンの一致、KL、perplexity、1 回の時間)、Apple は `hwnpu-dev generate -score`
   - 5 種の文で、校正に使わない GPL を必ず見る。蒸留モデルは元のモデルより量子化に弱いことがある (小さいモデルほど 4 ビットで崩れる)
4. 届かなければ校正 (GPTQ、表の当てはめ) を試す (5.3 節)。符号の置き場所は `codes` に、取得元と同じ書き方で書く
5. メモリ: 決めた形式で読み込み、Neural Engine が抱える分とプロセスの分を測る (6 節)。M1 の 2.8 GiB、コンパイルのキャッシュ 3.5 GiB を超えないか。超えるなら `prefill` か形式を変える
6. 差分を書く: `models` (`like`、`from`、`size`、`memory`)、必要なら `aliases` と `builds`。**測った値をコメントに残す** (KL、一致率、メモリ、測った機械と日付)。`-explain` で、書いたとおりに当たるかを確かめる

### 10.3 書き先: 手元か、このプロジェクトへの PR か

skill は最後に、どちらにするかを聞く。

**手元だけ** (社内の蒸留モデル、共有フォルダや S3 の取得元): ユーザーのファイル (`~/.config/hwnpu/models.yaml`) か、指定されたファイルに差分を書く。既定のファイルは変えない。

**このプロジェクトに PR を出す** (誰でも Hugging Face から取れるモデル): 既定に入れるのに要るものを揃え、PR まで出す。

1. `models.yaml` に足す: `models` (`from` は `hf:` だけ。社内の取得元は入れない)、必要なら `builds` と `categories`
2. 記録: MODEL.md と MODEL.ja.md のメモリの表に行を足し、おすすめの表を変えたならそれも。docs/design.md に節を足し、10.2 節で測った値 (形式ごとの KL、一致率、メモリ、測った機械と OS) を残す。MODEL.md のように .ja.md と対になっている文書は両方を直す
3. 確かめる: `go test ./...`。`size` の値と数えた値の差、MODEL.md と定義ファイルの食い違い (7 節) のテストもこれに入る
4. ライセンス: 元のモデルのライセンスで再配布 (docs/prebuilt.md の配布) ができるかを PR の本文に書く。できなければ「配布しない、手元で変換する」と書く
5. PR を出す: ブランチを切ってコミットし、`gh pr create`。書く権限がなければ、フォークに push してから出す。本文に、測った値の要約、確かめていない NPU (Mac しかなければ AMD と Intel は「未確認」)、10.1 節で引っかかったところを書く
6. 配布のビルド (`prebuilt/models.yaml` の `build`) は PR では動かない (docs/prebuilt.md 5.1。公開リポジトリの self-hosted runner を守るため)。PR に `prebuilt/models.yaml` の行を足してもよく、ビルドは owner が見て main にマージしたあとに動く

コミットと PR は、利用者が頼んだときだけ出す。測った値が許容誤差に届かないモデルは、PR にせず、手元のファイルで使ってもらう。

変換したものを上げるのは、どちらでも skill の役目ではない (ビルド・アップロードツール、9 節)。

## 11. 運用

### 11.1 置き場所

| | macOS | Linux | Windows |
|---|---|---|---|
| ユーザーの定義ファイル | `~/.config/hwnpu/models.yaml` | `$XDG_CONFIG_HOME/hwnpu/models.yaml` (`~/.config`) | `%LOCALAPPDATA%\hwnpu\models.yaml` |
| モデルのキャッシュ | `~/Library/Caches/hwnpu/models` (今と同じ) | `$XDG_CACHE_HOME/hwnpu/models` (`~/.cache`) | `%LOCALAPPDATA%\hwnpu\models` (`os.UserCacheDir`) |

`HWNPU_CACHE` でキャッシュの場所を変えられる (外付けのディスク、共有の機械)。

### 11.2 ロック

同じモデルを 2 つのプロセスが同時に準備すると、ダウンロードと変換がぶつかる (今はロックがない)。

- モデルのフォルダごとに `.lock` を置き、ダウンロード、変換、削除の間だけ排他でつかむ。Unix は `flock`、Windows は `LockFileEx`。プロセスが落ちれば OS が外すので、古いロックが残らない
- 待つ側は「別のプロセスが LFM3-1.7B を準備中 (PID 1234)」と出して待ち、つかめたら手元にあるものから続ける (たいていはもう揃っている)
- 読み込んで動かしている間は共有でつかむ。削除 (11.4 節) は排他を取れないモデルを飛ばす。動いているモデルのファイルを消さないため
- ダウンロードは `NAME.partial` に書き、sha256 が合ってから名前を変える。途中で切れたら、次は続きから落とす (HTTP の Range)

### 11.3 オフライン

`-offline` か `HWNPU_OFFLINE=1` (`HF_HUB_OFFLINE=1` も同じに扱う) で、ネットに一切出ない。

- 手元のキャッシュと、`path` の取得元 (共有フォルダ) だけを使う。HF と S3 は見ない
- 選ぶとき (用途、ファミリーのタグ) は、手元に揃っているモデルだけを候補にする。揃っていなければ、何が足りないかと、つながる機械で準備するコマンドを出す
- 固定 (5.3 節) は `source.json` の記録で確かめる。ネットを見ないので、上流の更新は気にしない
- 準備だけをするコマンド: `hwnpu-chat prepare NAME,...` (案) (ダウンロードと変換をして終わる)。キャッシュのフォルダを写せば、つながらない機械でもそのまま動く (Apple の Neural Engine のコンパイルだけは向こうで起きる)
- ネットにつながらないときに自動でオフラインに切り替えることはしない。黙って古いものを使わないため。つながらなければ、`-offline` を示してエラーにする

### 11.4 キャッシュの削除 (積極的に)

ディスクを食うのは、元のチェックポイント (Gemma 4 E2B で 10.2 GB)、前の設定で作ったプログラム、Neural Engine のコンパイルのキャッシュ。使わないものは自動で消す。

モデルのフォルダに `inuse.json` (今の設定で使うファイルの一覧、最後に使った日時) を書き、それ以外を消す候補にする。

| 消すもの | いつ |
|---|---|
| 元のチェックポイント (safetensors) | 今の設定のプログラムと表 (`tables.safetensors`) が揃ったら、すぐ。`config.json`、トークナイザー、`source.json` は残す。変換し直すときは固定どおりのものを落とし直す |
| 前の設定、前の `format_version`、前のハッシュのプログラム | 準備の終わりに。今の `gemma4-lut6-c16320-*` のような古い名前もここで消える |
| 前に配布物を落とした分と、手元で変換した分の重複 | 配布物に切り替わったら、手元で変換した方を消す |
| 長く使っていないモデル | `cache.unused_days` (既定 30 日) 使っていなければ、フォルダごと消す。消しても次に頼めば落とし直す |
| キャッシュ全体が上限を超えたとき | `cache.max` (既定なし。例 `100GB`)。最後に使った日時が古い順に消す |
| Neural Engine のコンパイルのキャッシュ (`~/Library/Caches/<プロセス>/com.apple.e5rt.e5bundlecache`) | モデルを消したとき、そのプロセスの分を消す。どの項目がどのプログラムのものかは分からないので、コマンドごとに丸ごと消し、次の起動でコンパイルし直す |

```yaml
cache:
  unused_days: 30
  max: 100GB
  keep_checkpoint: false     # true: 元のチェックポイントを残す (形式を何度も変えて試すとき、skill の 10.2 節)
```

- 消す前に、何をどれだけ消すかを 1 行ずつログに出す
- `-keep-checkpoint` で、その実行だけ元のチェックポイントを残す。hwnpu-dev formats と skill は既定で残す
- `hwnpu-chat cache` で、モデルごとのディスクの使用量、最後に使った日時、消す候補を出す。`--clean` で今すぐ消す、`--rm NAME` でそのモデルを消す
- 消さないもの: `~/models` の AMD の書き出し (利用者が作ったもの)、`path` の取得元 (`copy: false` で読んでいる共有フォルダ)、`--codes` (hwnpu-chat) に渡された符号のフォルダ

## 12. モデル固有の情報: どこまで定義ファイルに来るか

線引きは「**何を選ぶか、どれだけか**はデータ、**どう計算するか**はコード」。既知のアーキテクチャ、トークナイザー、会話の型の組み合わせで済むモデルは、定義ファイルだけで足せる。新しい計算が要るものは、今と同じく Go に書く (port-architecture の skill)。

| 情報 | 今の置き場所 | 定義ファイルに来るか |
|---|---|---|
| 層の数、幅、ヘッド、語彙、文脈、MoE の専門家の数と top-k | `config.json` (アーキテクチャが読む) | 来ない。もともとデータで、チェックポイントについてくる。ハッシュで固定される (5.3 節) |
| 用途、タグ、機能 (`tools`、`thinking`、`vision`)、考えない既定 (`NoThink`)、メモリ、説明 | `chat.Catalog` | 来る (1 段) |
| 重みの形式、部分ごとの形式 (`format_for`)、GPTQ の符号 | `Arch.Format`、`IntelFormat`、`Formatter`、`-format-for`、`-codes` | 来る (`builds`、2 段) |
| アーキテクチャの設定 (`attn`、`conv`、`post-scale`、Gemma 4 の `logits`) | `Arch.Params` の既定 | 来る。`builds` の `params: {attn: grouped}`。名前と取れる値はコードが決め、知らない名前はエラー |
| MoE の動かし方 (AMD の専門家ごとの行の上限、経路の選択をデバイスでするか、共有の専門家の分け方) | `tools/amd/iron_*.py` の引数、`internal/llm/iron` | 来る。`npu: amd-xdna2` の `builds` の `params` に。書き出しの道具にそのまま渡す |
| MTP の層があるか | `mtpArch` がチェックポイントを開いて調べる | 来る。`mtp: true` をモデルに書く (チェックポイントを落とす前に分かる)。書いていなければ今と同じに調べる |
| エンコーダ (視覚の 512×512、音声の 30 秒) | `chat.encoders` | 来る。モデルの `encoders` に大きさと長さを |
| どのトークナイザーのファイルを読むか | `chat.TokenizerFiles` (PLaMo 3 だけ `tokenizer.jsonl`) | 来る。ファミリーか モデルの `tokenizer: {files: [...]}`。省けば `tokenizer.json` |
| トークナイザーの種類 (BPE、byte fallback、区切りの正規表現) | `tokenizer.Load` が `tokenizer.json` から判別 | 来ない。ファイルの中身で決まる。新しい種類 (SentencePiece の unigram など) は Go の実装を足す |
| 会話の型 (ツールの呼び出しの書き方と読み方、`<think>`) | `chat.TemplateFor(arch)` (Go) | 選ぶのだけ来る。`template: qwen35` のように、Go にある型の名前で。Jinja の `chat_template` をそのまま動かすのは別の大きな仕事 (Jinja の実装と、ツールの呼び出しの読み方の対応づけ) なので、ここでは持たない |
| 止めるトークン、特別なトークンの id | 会話の型 (`Template.Stops`) とトークナイザー | 足すのだけ来る。`stops: ["<|im_end|>"]`。型にあるものに加える |
| Neural Engine 向けの書き換え (silu を exp で、サンドイッチ正規化の吸収、ソフトキャップをホストで) | アーキテクチャのコード | 来ない。計算そのもの。切り替えが `Params` にあるものだけ、上の行のとおり来る |
| 変換し直しが要る変更の番号 (`Arch.Revision`) | アーキテクチャのコード | 来ない。コードの変更についてくるもの |
| 会話の KV の 1 トークンあたりの大きさ (メモリの見積もり) | MODEL.md の手書き | 来ない。`config.json` から計算できる。MODEL.md の表を作るときに計算する |

### 12.1 docs/new-model-checklist.md の変わるところ

手順の中身 (測り方、目安の値、試して効かなかったもの) は変わらない。変わるのは、決めた値の**書き先**と、記録の一部。

| 節 | 今 | 変わったあと |
|---|---|---|
| 1. 机上の見積もり | | 「10.1 節の確かめ (アーキテクチャ、トークナイザー、会話の型) が通れば、移植は要らず 5 節から」を足す |
| 5.4 形式の決め方 | `Formatter` (config の `Format()`) で決める | `builds` の `size` のルールで決める (6.1 節) |
| 5.3 校正 | `hwnpu-chat --codes` | `builds` の `codes` に、取得元と同じ書き方で |
| 4.1 AMD | `iron_export_lm.py` の引数 (`--mlp q6` など) を手で | `npu: amd-xdna2` の `builds` に書き、ビルドツールが渡す |
| 9. 記録 | design.md と improvements | 加えて `models.yaml` (`size`、`memory`、コミットと sha256)。MODEL.md は定義ファイルと食い違えば `go test` が落ちる |
| 新しい節 | | 「手元か PR か」(10.3 節) と、skill の使い方 |

英語版 (new-model-checklist.md) と日本語版 (new-model-checklist.ja.md) の両方を、2 段 (ビルド設定を移す) を入れるときに直す。それまでは今の手順のまま。

### 12.2 構造の型を選ぶ

構造 (forward の計算) は、今と同じく Go に書いた**型**のどれかを選ぶ。複雑な構造は型の中にハードコードのまま置き、定義ファイルからは選ぶのと、型が公開する設定 (`Params`) を変えるのだけにする。

```yaml
models:
  my-distill-1b:
    like: LFM3-1.7B
    arch: lfm3              # 構造の型。省けば config.json の model_type と architectures から判別 (今の llm.Detect)
    from: team-share:my-distill-1b
```

- 型は今の `llm.Archs()` の一覧 (`hwnpu-dev convert -archs`): `lfm3`、`gemma4`、`qwen35`、`plamo3`、`bert`、`plamo`、`whisper`、エンコーダ類
- `arch` を書くのは、蒸留や独自の学習で `model_type` が元と違う名前になったとき (`MyDistillForCausalLM` など)。中身が同じ構造なら、型を名指しすれば通る。違う構造なら、型の `LoadConfig` が知らない設定でエラーにする (10.1 節)
- 型の中身 (Gemma 4 の層ごとの埋め込み、Qwen3.5 の Gated DeltaNet、LFM3 の短い畳み込み、MoE の経路、Neural Engine 向けの書き換え) は定義ファイルに出さない。出すと組み合わせが爆発し、正しさを確かめられなくなる

**型を増やす候補: 素直なデコーダ (Llama 型)**。今は型ごとに 1 つの系統だけを受けている。Llama、Mistral、Qwen2 / Qwen3 (密)、Phi など、正規化、GQA の注意、SwiGLU の MLP の繰り返しで済む系統は、違いが `config.json` の設定 (RoPE の種類、窓、QK の正規化、バイアスの有無) に収まる。これを 1 つの型 `decoder` にまとめれば、蒸留モデルの多く (小さいモデルの蒸留先によく使われる系統) が、コードを足さずに定義ファイルだけで使える。

- 設定の組み合わせごとに、fp64 の正解実装とのテストを持つ (port-architecture の合格条件と同じ)
- 受ける設定は一覧で持ち、外れたものはエラー。「たぶん動く」で通さない
- 今の型 (lfm3 など) をこの型に寄せることはしない。それぞれの Neural Engine 向けの書き換えが型ごとに違うため

`arch` の選択は 1 段で入れる。`decoder` 型も計画に入れる (8 節の 7 段)。port-architecture の skill の合格条件で作る。

**Anima の Qwen3 を取り出して元にする**。Qwen3-0.6B はすでに Anima-Turbo (MODEL.md の画像生成の行) のテキストエンコーダとして動いている。

- Apple: `internal/imagegen/anima/mil/text.go` の `qwenLayer`。RMS 正規化、Q と K の正規化 (`q_norm`、`k_norm`)、RoPE (θ = 1e6)、GQA (16 / 8 ヘッド)、SwiGLU。ただし大きさは定数 (`qwenLayers = 28` など) で、KV キャッシュも出力の行列もない (512 行を一度に読んで隠れ状態を出すだけ)
- AMD: `tools/amd/iron_anima.py` の `Qwen3Layer` (IRON)

取り出し方:

1. `internal/llm/decoder` に型 `decoder` を作る。`qwenLayer` を元に、大きさを `config.json` から読み、ほかの型と同じく KV キャッシュ、塊 (`paged`)、最後の行だけ、出力の行列を持たせる。`model_type` は最初 `qwen3` を受け、続けて `llama` (Q と K の正規化なし、Llama 3 の RoPE の拡張、出力の行列を埋め込みと共有) と `mistral` (窓) を受ける
2. Anima のテキストエンコーダを `decoder` 型に寄せる (`NoHead` で隠れ状態を出す、今の `Start` / `Layers` の仕組み)。Anima の画像が変わらないこと (今のテストの値と同じ) を、取り出しが正しいことの確かめに使う
3. AMD: `Qwen3Layer` を `iron_export_lm.py` から使えるようにし、`decoder` 型のチャットを XDNA 2 でも書き出せるようにする (Qwen3.5 の 0.8B / 2B と同じ道)

**最初に確かめるモデル: Qwen3-0.6B** (`Qwen/Qwen3-0.6B`)。理由:

- MODEL.md にすでにある (Anima の中)。重みも、Anima の正解の値も手元にある
- Anima が使うのは base (`qwen_3_06b_base`)、チャットは instruct (`Qwen/Qwen3-0.6B`)。同じ構造で、テストの正解を 2 通り持てる
- 用途の表 (`chat` の picks) には入れない。`qwen3:0.6b` としてファミリーのタグで選べるようにし、蒸留モデル (Qwen3 を元にしたものが多い) の土台を確かめるのが目的。MODEL.md の 8 GB の欄には、一番小さなダウンロードとして並べる
- **入れた (2026-11-09、design.md 16.127)**: 型の名前は今は `qwen3` (`internal/llm/qwen3`)。全層が注意なので注意で分けると遅く、分けない 1 本 (文脈 4096、`llm.Unpaged`) が既定。Anima のテキストエンコーダをこの型に寄せること (2.) と AMD (3.) はまだ

**会話の型: ChatML + JSON のツールの呼び出し (Hermes 式) を足す**。Qwen3 のツールの呼び出しは `<tool_call>{"name": ..., "arguments": ...}</tool_call>` の JSON で、Qwen3.5 の XML 式 (`<function=...>`) とは違う。同じ型を SmolLM などの ChatML の系統でも使う。Llama 3 と Mistral の型は、それぞれのモデルを受けるときに足す。Jinja の実行はしない (12 節)。

## 13. 決めたこと (2026-11-09)

| 項目 | 決めたこと |
|---|---|
| 形式 | YAML (`goccy/go-yaml`)。コメントで値の理由を残す |
| 組み込みの置き場所 | リポジトリの直下の `models.yaml` |
| ユーザーのファイル | `~/.config/hwnpu/models.yaml`、Windows は `%LOCALAPPDATA%\hwnpu\models.yaml`。差分だけを書く |
| 版が違うとき | 組み込みのファイルは厳しく読む (知らないキー、知らない名前はエラー、`go test` で落ちる)。ユーザーのファイルの「知らないモデルへの上書き」と、`version` が自分より新しいときの知らないキーは、警告を出して読み飛ばす。hwnpu を更新しても、古い hwnpu に戻しても、ユーザーのファイルで起動できなくならないため |
| NPU の名前 | 世代で: `apple-m1`…、`amd-xdna1` / `amd-xdna2`、`intel-npu3` / `npu4` / `npu5` |
| S3 | `aws` コマンド。`endpoint`、`profile`、`region` を設定できる |
| 固定 | コミットとファイルごとの sha256 |
| 読み込みの順 | 手元 → ネイティブの変換済み → 手元で変換 (今と同じ) → 使えない。書き戻しはしない |
| 運用 | ロック、オフライン、積極的なキャッシュの削除 |
| ほかのコマンド | 同じ表から選ぶ |
| 構造 | Go の型から選ぶ (`arch`)。素直なデコーダの型 `decoder` を、Anima の Qwen3 から取り出して作る |
| 会話の型 | Go で足す (ChatML + JSON のツールの呼び出しから)。Jinja は実行しない |
| ライセンス | 新しく足すモデルは、手間のかからないもの (Apache-2.0、MIT) から選ぶ。`models.yaml` の `license` に書き、配布 (docs/prebuilt.md) はそれで決める |

### 13.1 ライセンスと `decoder` 型で確かめるモデル

- **Qwen3-0.6B** (`Qwen/Qwen3-0.6B`、Apache-2.0): 最初のモデル。Anima の中にあり、`model_type` は `qwen3`
- **SmolLM2-1.7B-Instruct** (`HuggingFaceTB/SmolLM2-1.7B-Instruct`、Apache-2.0): `model_type` が `llama` の構造を、Meta の Llama を使わずに確かめる。Q と K の正規化がない、出力の行列を埋め込みと共有する、ChatML の会話の型
- Meta の Llama (同意とコミュニティライセンスが要る) と、研究用のライセンスのモデルは足さない。`llama` の構造は SmolLM2 で受けるので、Apache-2.0 の蒸留モデル (Llama 型が多い) はそのまま使える
- 今あるモデルのライセンスは変えない (LFM3 は LFM Open License、Gemma 4 は Gemma の利用規約)。`license` に書き、配れるかを docs/prebuilt.md の表で決める
