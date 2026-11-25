# モデルのレシピ (`model-recipes.yaml`)

モデルの変換を、誰がどの機械で作っても同じものにするための定義。作るときにしか要らないことを、ここに書く。動かすときに要ることは models.yaml に置く。

- **models.yaml**: 動かすときに要るもの。モデルの取得元、メモリ、サイズ (`npu.*.sizes`)、固定した書き出し (`exports`)
- **model-recipes.yaml**: 作るときにだけ要るもの。どのチップ・世代向けに作るか、何本同時に作るか、通らないプログラムをどうするか、AMD の書き出しのスクリプトと引数、道具の版

どちらもリポジトリの直下にあって、バイナリに組み込まれる (`hwnpu.ModelsYAML`、`hwnpu.RecipesYAML`)。読むのは `internal/recipe`。

## 1. 作るコマンド

```sh
hwnpu-dev models package FLUX.2-klein-4B
```

- レシピがあるモデルは、ターゲットを書かなくてよい。Core AI の書き出しを作り、レシピのチップ (`archs`) 向けにプリビルドし、取得元 (`HWNPU_MODEL_SOURCE …?writeback=true`) に置く
- `MODEL/coreai-h14g` のようにチップを書くと、そのチップだけを作る。並列数と、実機に回すプログラムは、そのときもレシピのとおり
- チップによって書き出しが違うモデル (FLUX.2 klein は M1 だけ別: models.yaml の `npu.apple.generation_edits`) は、M1 として作った書き出しを h13 の種類 (h13g、h13s、…) だけに、M2 として作った書き出しをそのほかの種類だけにプリビルドする (`chat.TakesExportsOf`)。ほかのチップの書き出しのプリビルトは作らない。手元の書き出しを `hwnpu-dev prebuild EXPORT_DIR` で作るときは、この振り分けがないので `-arch` で種類を名指しする
- 手元の書き出しだけをプリビルドするときは `hwnpu-dev prebuild -put EXPORT_DIR`。どのモデルのレシピかは、書き出しの `entry.json` の `model` から決める (`-model NAME` で指定もできる)
- `-arch` と `-j` は、レシピより優先する。確かめのとき以外は書かない
- 作り直すとき (版が上がったときなど)、source (`main.hash`) の変わらない部品は、同じモデルのフォルダにある前の書き出しの prebuilt を流用して、コンパイルしない。流用するのは、記録 (`NAME.coreai.prebuilt/ARCH/build.json`) で同じ coreai-build、同じかより少ない同時数、端末に任せる決まりなし、NE に乗っている、端末で落ちた印なし、と分かるものだけ。記録のない前の prebuilt には `hwnpu-dev prebuild -record` で一度記録を足す (足すまでは、その場所では使うが、ほかの書き出しへは流用しない)。端末で落ちた印は `hwnpu-dev prebuild -mark` か `HWNPU_DEV=coreai.mark-prebuilt` で付ける ([prebuilt-reuse](prebuilt-reuse.md))
- AMD のレシピ (`amd`) があるモデルは、Linux と Windows ではターゲットを書かなくても、レシピの世代 (`generations`) の書き出し (`amd-xdna2`) を作る。Mac では AMD の分を、ほかの OS では Core AI のプリビルドの分を、一言出して飛ばす
- `MODEL/amd-xdna2` のように世代のターゲットを書くと、その世代だけを作る。hwnpu がまだ動かさない世代 (`amd-xdna1`) と、レシピが `skip` する世代は、理由を出して止まる

## 2. ファイルの形

```yaml
version: 1

tools:
  coreai-build: 3605.6.3        # coreai-build --version。違う版では止まる (--any-tools で作れる)

apple:                          # どのモデルにも効く既定
  archs: [h13g, h14g, h15g]     # プリビルドするチップ (h13g M1、h14g M2、h15g M3)
  jobs: 2                       # coreai-build を何本同時に動かすか

models:
  FLUX.2-klein-4B:              # models.yaml のモデルの名前
    apple:                      # archs と jobs は既定より優先し、prebuild は既定のあとに足す
      prebuild:
        - arch: h13g                                    # 1 つか並び。省くと全部
          export: "flux.2-klein-4b-8sizes-*"            # 書き出しの名前のパターン。省くと全部
          programs: [dit_single_*]                      # プログラムの名前のパターン (.aimodel なし)
          do: device                                    # プリビルドせず、端末のコンパイラに任せる
          why: "RegAlloc: failed to allocate tensors with fixed L2 hint"   # coreai-build が何をしたか
          checked: 2026-11-22                           # それを見た日
```

置き換え (`rewrite`) は、チップによって通らない演算を、どう書くかを決める:

```yaml
  MOSS-SoundEffect-v2.0:
    apple:
      rewrite:
        - arch: [h15g, h15c, h16g, h16s, h16c, h17g, h17s, h17c, h17d]  # 1 つか並び。省くと全部
          op: sin                       # 演算 (recipe.Ways にあるもの)
          as: native                    # 書き方。sin は series (既定) か native
          why: "..."
          checked: 2026-11-23
```

- 書き方の一覧と既定は `internal/recipe` の `Ways` (sin: `series` 範囲を縮めて級数、`native` 演算そのもの)。規則のないチップは既定
- 置き換えは変換のときに選ぶ (書き出しにはバイトコードしか残らない)。そこで、既定でない置き換えを、その演算を持つ書き出しの名前に入れる (`hwnpu-dac-r3-fp16-sin-native.coreai`)。チップごとに自分の書き出しを取ってくる
- `hwnpu-dev models package MODEL` は、置き換えの組ごとに 1 つずつ書き出しを作って置く (`Apple.RewriteArchs`)
- 動かすときは、その Mac のチップ (`coreai.Arch`) の規則で選ぶ。比べるときは `HWNPU_DEV=rewrite=sin:series` (または `sin:native`) で、その Mac の分だけ上書きできる
- 置き換えを使うモデルのコードは、`llm.Options.Rewrites` を読んで書き分け、書き出しの名前に `recipe.Suffix` を付ける (MOSS の DAC: `sfx.DACOps`)

- 知らないキー、知らない版、知らないチップの名前、知らない演算と書き方は、読むときに誤りにする
- `why` と `checked` は省けない。あとで Metal Toolchain が上がったときに、その規則がまだ要るかを確かめるため
- パターンは `path.Match` の形 (`*` は何文字でも)

### AMD (`amd`)

AMD の NPU の書き出しは、`tools/amd` のスクリプトを IRON の環境で動かして作る。どのスクリプトを、どの引数で動かすかをレシピに書く。前は models.yaml の `amd_export` (チャットのモデル) と Go のコード (画像、深さ、ラベル、検出など) に分かれていた。

```yaml
amd:                            # どのモデルにも効く既定
  generations: [xdna2]          # 作る世代: xdna1 (Phoenix、Hawk Point: IRON の npu1)、xdna2 (Strix、Krackan: npu2)
  extra:
    xdna1: [--device, npu1]     # その世代で、どのスクリプトにも足す引数 (どれも --device を取り、既定は npu2)

models:
  gpt-oss-20b:
    amd:
      script: iron_export_lm.py                                  # tools/amd のスクリプト
      args: [gptoss, "{in}", "{out}", --max-seq-len, "8192", --prefill-chunk, "256", --expert-places, "16", --host-route]
      variants:                                                  # models.yaml の variants と同じ GB のキー
        24: [gptoss, "{in}", "{out}", --max-seq-len, "8192", --prefill-chunk, "256"]
      extra:
        xdna1: [--cols, "4"]                                     # 既定の extra のあとに足す
  FLUX.2-klein-4B:
    amd:
      script: iron_flux2klein.py
      args: ["{in}", "{out}", --size, "{sizes}"]
      generations: [xdna1, xdna2]                                # 既定より優先
      skip:                                                      # 作らない世代と、その理由
        - generation: xdna1
          why: "..."
          checked: 2026-11-23
```

- 置き場所 (`{...}`) は 1 つの引数まるごとで、動かすコードが入れる: `{in}` コードが用意したもの (チェックポイントのフォルダ、ONNX のファイル、Anima のフォルダ)、`{out}` 書き出しのフォルダ、`{sizes}` 大きさの並び (models.yaml の `npu.amd.sizes`、Anima は `aspect.Of(16)`、Supertonic は `supertonic.Sizes`)、`{size}` (検出と姿勢の入力の大きさ)、`{outputs}` (コードが ONNX から選んだ出力)、`{text}` (MOSS の文のエンコーダーの形式)、`{dit}` (Anima の DiT のファイル)。コードが入れない置き場所を書くと、作るときに止まる
- スクリプトの名前は、動かすコードが読む書き出しのもの (`iron_whisper.py` の書き出しを読むのは Whisper のコード)。違うスクリプトを書くと、作るときに止まる
- `iron_export_lm.py` の args のうち `{in}` と `{out}` を除いたものが、そのまま models.yaml の `amd_export` だったもの。書き出しの名前 (`-q4`、`-q6`、`-p16`) と取得元のキーになる。動かす側もこれを読む (`chat.Model.AMDExport`)
- ほかのスクリプトの書き出しのキーは、前と同じで、コードの版 (`ExportVersion`) と、FLUX.2 klein、FLUX.3 klein、MOSS は args (`{in}` `{out}` を除く) も入る。手で作った書き出し (Anima、DA3、MoGe、SigLIP2、YuNet) は名前だけ。キーの中身は前と同じにしたので、取得元の書き出しとピンはそのまま使える
- `like` のモデル (自分の定義ファイルのものも) は、自分のレシピがなければ、似せたモデルのレシピで作る
- 世代ごとに別のターゲット (`amd-xdna1`、`amd-xdna2`) に置く。hwnpu が動かすのは今は `amd-xdna2` だけ (`recipe.Built`) なので、xdna1 のレシピは読んで確かめるが、作らない

#### 版 (`version`)

```yaml
  LFM3-1.7B:
    amd:
      version: 1        # レシピを変えたら上げる
      script: iron_export_lm.py
      ...
```

- `version` は、そのレシピが作るものの版。取得元には `vN` のフォルダで置く (`converted/MODEL/amd-xdna2/EXPORT/v1`)。前は models.yaml の `targets: {amd-xdna2: {version: N}}` だった。組み込みの models.yaml には書かない (自分の定義ファイルに書けば、そちらが優先)
- **レシピか道具を変えたら版を上げる**。版の中身は `internal/recipe/testdata/amd-versions.txt` に残してある: レシピの指紋 (`Fingerprint`) と、そのスクリプトが読む `tools/amd` のファイルの hash (`ToolsHash`: スクリプトと、それが import するモジュール、全部のカーネル。`iron_export_lm.py` はそのアーキの分、`LMHash`)。同じ版のままどちらかが変わると `go test ./internal/recipe` が落ちる。版を上げたら `go test ./internal/recipe -run TestAMDVersions -update` で書き直す
- **ピンはその版のもの**。ピンのフォルダの版 (`vN`) がモデルの版と違うと、そのピンは取らない (`chat.Pin.OfVersion`)。版を上げると、前のピンの書き出しではなく、その版のものを作るか取る。組み込みのピンがどれもその版のものであることは `go test ./internal/chat` (TestPinsOfVersion) が見る。レシピより前にピンした `v0-HASH` は版 1 のものとみなす
- 版を上げたら、作り直して置き、ピンし直す (`hwnpu-dev models package MODEL/amd-xdna2`、`hwnpu-dev models pin MODEL`)。手元で確かめるときは、手元の前の書き出し (`~/.cache/hwnpu/entries/amd-xdna2/EXPORT`) を消してから作り直す
- `vN` は道具の hash によらず取られる (キーの比べから道具を除く)。道具が変わったことは、上の `ToolsHash` で版を上げさせて知らせる
- まだ見ないもの: コードや models.yaml から入る値 (`{sizes}` の大きさ、`{size}`、`{outputs}`)。これを変えたときは、手で版を上げる
- 版 1 は、models.yaml の `amd_export` と Go のコードからレシピに移したときのもの (2026-11-23)。作るものは同じ (作り直してバイトまで同じことを見た) なので、作り直さず、取得元のピンした書き出しを `v1` にコピーして版だけ書き換え、ピンを `v1` に移す。2026-11-23 は 12 個 (LFM3 の 4 つ、Gemma 4 の 2 つ、Qwen3.5 の 3 つ、gpt-oss-20b の 2 つ、Qwen3.6-35B-A3B の q4) を移したところで NAS が書き込みを受けなくなった。残りの 34 個のピンは前の `v0-…` のまま (ピンはフォルダの名前によらず中身の digest で取るので、そのままで動く)

#### 仕上げ (`finish`)

```yaml
  LFM3-1.7B:
    amd:
      version: 2
      finish: [{tokenizer: [tokenizer.json]}]   # 書き出しにした後で、こうする
  Anima-Turbo:
    amd:
      finish: [{tokenizer: [text/qwen_tokenizer.json, text/t5_tokenizer.json]}]
```

- スクリプトや変換が書き出しを作った後、置く (`out.part` を `out` にする、`entry.json` を書く) 前に、ビルドがする手順。順に、一つの手順に一つの種類 (`internal/finish`、`recipe.Step`)
- `tokenizer`: 書き出しの中のトークナイザーのファイル (書き出しの中のパス) を、それぞれ隣のバイナリのファイル (`tokenizer.tok`、`qwen_tokenizer.tok`) にして、元のファイルを消す。実行するときは JSON を読まずに、バイナリのファイルを mmap してそのまま引く (Gemma 4 の 32 MB の tokenizer.json で、読み込み 480 ms・194 MB → 0.6 ms・ほぼ 0)。バイナリのファイルは元と同じに符号化することを、書いた後で確かめる
- `remove`: 書き出しの中の、実行するときに読まないファイル (書き出しの中のパス) を消す。ないものは消えたものとする。スクリプトが自分のために書くものや、前の手順が読むもの (`tokenizer_config.json`: PLaMo の `tokenizer` の手順が読み、バイナリのファイルが値を持つ) に。`iron_export_lm.py` のチャットのモデルは `finish: [{tokenizer: [tokenizer.json]}, {remove: [tokenizer_config.json]}]` (版 3)。スクリプトを変えず (`iron_export_lm.py` はほとんどのスクリプトが読むので、変えると全部の版が上がる) にレシピで決める
- ツールのチェック用のデータ (`manifest.json`・`model.json` の隣の `check/`: `iron_export_lm.py` が必ず書く `check/words.json`、`--check` の入出力) は、エントリに入れない (`store.Describe`)。手元のテストは作ったフォルダから読む
- 書き出しに、`finish` が挙げていないトークナイザーのファイル (`tokenizer.json`、`NAME_tokenizer.json`、`tokenizer.jsonl`) が残っていたら、ビルドは止まる。挙げ忘れたモデルが JSON のまま出ていかないように
- `finish` を変えるのはレシピを変えること: 指紋 (`Fingerprint`) に入るので、版を上げないと `go test ./internal/recipe` が落ちる。手順の種類ごとの版 (`recipe.StepVersions`、tokenizer はバイナリの形の版) も指紋に入る
- 前の版の書き出しを作り直さずに新しい版にするには: 手元の書き出しのフォルダで `hwnpu-dev sources import --finish DIR` (仕上げをして、今の版として置く)、それから `hwnpu-dev models pin MODEL`。Core AI の書き出し (`models/MODEL/NAME.coreai`) も同じコマンドで置ける (ピンはない)
- Apple (`models.NAME.apple.finish`) も同じ形。Core AI の書き出しを置く前 (`chat.FinishCoreAI`) にする。AMD と同じに、トークナイザーのテキストが残った Core AI の書き出し (finish のないモデル、モデルの分からないもの) は置かない
- 書き出しにない (元のチェックポイントから読む: Apple、Intel、CPU のチャット、TTS、Whisper、sfx、d1) トークナイザーは、取得元のエントリ `tokenizer` (`converted/NAME/any/tokenizer/v0-HASH`、キーはリポジトリとコミットとファイルとバイナリの形の版) のバイナリのファイルを取ってきて使い、tokenizer.json は落とさない (`chat.PrepareTokenizer`)。取得元にないときは JSON を落としてここでバイナリにし、書き戻す取得元に置く。前もって全部置くには `hwnpu-dev sources tokenizers`
- それ以外にテキストのまま読んだトークナイザー (手で渡したファイルなど) は、初めに読んだときにキャッシュの根にバイナリのファイル (`KEY.tok`) を残し、次からはそれを mmap する

### Apple の版 (`version`、`code`、`convert`)

Core AI の書き出しは Go のコードが作る。どの値で作るかを `convert` に、どのコードが書くかを `code` に、その版を `version` に書く。AMD の `script`・`args`・`version` と同じ考え。

```yaml
apple:
  code: [internal/mlpackage, internal/coreai/ailower, ...]   # どの変換も書くのに使うコード
models:
  flux-3-image-klein:
    apple:
      version: 1                # convert か code を変えたら上げる
      code: [internal/imagegen/flux3klein, internal/imagegen/flux3klein/mil, ...]
      convert:                  # 前はコードの定数だった値。コードの名前で書く
        text_format: lut8
        dit_format: lut8
        text_group: 4
        single_group: 5
```

- `version` は、そのモデルの Core AI の書き出しの版。取得元には `vN` のフォルダで置く (`converted/MODEL/coreai/EXPORT/v1`)。プリビルドは書き出しのバイト列で引くので、版によらない
- **版 1 は、レシピに移す前と同じもの** (2026-11-24)。書き出しの名前も中身も前と同じ。取得元の前の書き出し (`v0-HASH`) は、同じ鍵なら版 1 として取る (AMD と同じ `Unversioned`)。取ったマシンが `v1` に置き直すので、コピーは要らない
- **convert かコードを変えたら版を上げる**。`internal/recipe/testdata/apple-versions.txt` に、モデル、版、`convert` の指紋 (`AppleFingerprint`)、`code` の hash (`CodeHash`) を残してある。版が同じままどちらかが変わると `go test ./internal/recipe` が落ちる。版を上げたら `go test ./internal/recipe -run TestAppleVersions -update` で書き直す
- `CodeHash` は、`code` の各パッケージの Go のファイル (テストを除く) を、コメントを除いて gofmt の形にしたものの hash。コメントや並べ方を直しただけでは変わらない。`.go` で終わる名前は、そのファイルだけ (変換のほかのこともするパッケージ: `internal/llm` の書き出しの部分、`internal/chat/model.go`)。`!FILE` は、そのパッケージのうち動かすときだけ読むファイルを除く (`internal/llm/experts/host.go`、`internal/llm/lfm3vl/preprocess.go`)
- **作り直してバイトまで同じなら、版を上げなくてよい**。共通のコード (`apple.code`) はどのモデルも使うので、そこを直すと全部のモデルの test が落ちる。`tools/coreai/same-exports.sh BASE MODEL...` で作り直して同じことを見たら (変える前の commit と今の hwnpu-dev で、別々のキャッシュに `models package MODEL/coreai` を作り、名前と、`metadata.json` の `creationDate` のほかのバイトを比べる)、版はそのままで `-update` だけしてよい。PR にそう書く
- 版を上げると、手元の書き出しも作り直す。書き出しの `entry.json` の版がレシピの版と違えば、なかったものとして取り直すか作り直す (`llm.StaleExport`)。名前は変わらなくてよい
- `convert` の値を読むのは Core AI の書き出しを作るときだけ。Core ML や Intel の書き出し、`hwnpu-image convert` で手で作るときは、コードの既定のまま。動かすときにも要る値 (models.yaml の `npu.apple` の大きさと形式、`settings`) は models.yaml に残す。`settings` で変えたものはレシピより先に使う
- 今 `convert` があるのは、画像のモデル (FLUX.2 klein、FLUX.3 klein、Anima: 形式とプログラムの段の数)、視覚と音声のモデル (検出、姿勢、分割、SigLIP 2、DA3、MoGe、YuNet、PP-OCR、SwiftF0、Basic Pitch: 重みの形式 `format`; Magenta: `format` と量子化 `quant`; Whisper: `quant`)
- 視覚と音声の `format` は書き出しの名前にも入る (`rtmdet-…-r1-fp16-640`)。動かすときも名前を作るのでレシピを読む (組み込みのレシピ)。書き出しの場所からモデルを引くので、動かしながら作る大きさ (DA3、MoGe、YuNet、PP-OCR) も同じ値になる
- MOSS の文のエンコーダーと DiT の形式、プログラムの分け方、fp16 に残す層は、AMD の書き出しと動かすときも読むので、コードに残す (`code` の hash が見る)。TTS (LFM3-Audio-2B-TTS) と Supertonic (Core ML で書く) もまだコード

## 3. 通らないプログラムがあったとき

`hwnpu-dev models package` と `hwnpu-dev prebuild` は、レシピにないプログラムが落ちると止まる。

- 落ちたプログラムがある書き出しは、そのチップ向けのものを取得元に置かない。人によって置かれるものが変わらないようにするため。ほかのチップのものは置く
- coreai-build のエラーは出力ではなく os_log に出る。そこで、落ちた coreai-build のプロセスのログ (`/usr/bin/log show --predicate 'processID == PID'`) から、理由の行を拾って表示する
- 表示の最後に、そのプログラムを端末に回すときのレシピの規則を示す。プログラムの作り方を直すか、その規則を model-recipes.yaml に書き足して PR にする
- 2 本同時のコンパイルで NPU から外れたものは、Build が最後に 1 本ずつ作り直してから判断する (外した Chroma1 の double で見た)
- 同じ source の prebuilt が端末で落ちた印 (`build.json` の `device.status`) のある部品は、作り直しても同じものになるので作らず、同じように止まって端末に回す規則を示す ([prebuilt-reuse](prebuilt-reuse.md) 3節)

## 4. 実機に回る仕組み

- `do: device` のプログラムはプリビルドしない。端末がそのプログラムのプリビルドを探しても見つからないので、端末のコンパイラがコンパイルする (初回だけ。そのあとは Core AI のキャッシュから)
- VAE (`vae*`)、文のエンコーダー (`text*`)、名前が `_device` で終わるプログラムは、レシピとは別に、いつも端末に回る (`coreai.LeftToDevice`)。動かすときにも使う決まりなので、コードに残している

## 5. いまのレシピの中身 (2026-11-25)

| モデル | M1 (h13g) | M2 (h14g) | M3 (h15g) |
|---|---|---|---|
| FLUX.2 klein 4B | 全部プリビルド。ただし書き出しは M1 用 (1024² の参照 1 枚を 512 で読む。8192 キー以上の関数は h13g のプリビルトが壊れる: [known-issues](known-issues.md) 13 節) | 全部プリビルド | 全部プリビルド |
| MOSS-SoundEffect v2.0 (`-lut8` も) | sin は級数 | sin は級数 | sin はそのまま (`-sin-native` の書き出し) |
| FLUX.3 klein | 全部プリビルド | 全部プリビルド | 全部プリビルド |

FLUX.3 klein (Apple の版 2) は、規則なしで 3 つのチップとも通った (2026-11-25、M3 の上の coreai-build 3605.6.3)。DiT の 9 本 (`ctx`、`dit_image_0`、`dit_text_0`、`dit_single_*` の 6 本) をプリビルドし、文のエンコーダーと VAE は端末に回る (`LeftToDevice`)。単ブロックの 1 本 (5 ブロック) は M3 で 6〜7 分、M2 で 9〜12 分、M1 で 18〜27 分。M1 のコンパイルされた区間は M2 の 1.5 倍 (74 MB と 49.5 MB)。M1 の実機では、NAS の書き出し (9.2 GB) と h13g のプリビルド (0.15 GB) を取ってきて、プリビルドの失敗 (`the prebuilt program failed`) なしに全部 Neural Engine で動いた。512² は 1 ステップ 4.0〜4.2 秒、1024² は 16.8〜17.2 秒 (16 ステップと VAE 9.3 秒で 4 分 45 秒)。どちらも縞、雑音、継ぎ目はない。初回は 49 分 33 秒 (取得 7 分 25 秒、残りは端末に回る文のエンコーダーと VAE のコンパイルと仕上げ。FLUX.2 のプリビルドを同時に動かしていたので遅めに出ている)。

MOSS の DAC の sin は、coreai-build で 5 つの段を見た。sin をそのまま書くと、h13g、h14g、h14c (M2 Pro・Max) は 1 段が Neural Engine の 8〜9 個の区間に分かれる (sin がその間で外に出る)。h15g/c、h16g/s/c、h17g/s/c/d は 1 区間に収まる (h15s/m/d と h17p は coreai-build が落ちるので見ていない)。M3 では、この書き出しを取ってきて音を作れた。級数はどのチップでも 1 区間。sin の前の clip は、sin とまとめられて (NEFUSED_EW: EW: MAX) h15g でも通らないので、そのまま書くときは clip を外す (clip は丸めのためのもの)。音声のプリビルドは置かない (M1 で遅いか壊れていた) ので、DAC は全部端末に回す規則にしてある。

どれも、落ちるまでに 1 本 1〜3 時間かかっていた。レシピに書いたので、もう試さない。

## 6. まだレシピにないもの (次の PR)

- **ほかの置き換え**: 2 節の `rewrite` は、いまは MOSS の DAC の sin だけ。ほかはコードに直接書いてある (softmax を exp と逆数で、x / c を 1 / c 倍で、sigmoid と tanh を exp で、conv_transpose を polyphase で、など)。M1 と M2 以降で違うものから、書き方を `Ways` に足して移す。sin の表引きも、そのときに書き方として足す
- **変換の設定 (Apple)**: 画像のモデルのほかは、値がまだコードにある。次に視覚と音声 (検出、姿勢、深度、OCR、MOSS、Magenta、Whisper、TTS)、そのあと言語モデル (prefill の長さ、区切りの段) を `convert` に移す。MOSS の DiT の段の数や形式は、動かすときと AMD の書き出しも読むので、manifest に書いてから移す
- **動かすときにも読む値**: 量子化の形式 (`npu.apple.formats`)、FLUX.2 klein の編集の組み合わせなど。models.yaml に残す
- **AMD の xdna1**: レシピは書けるが、hwnpu が `amd-xdna1` の書き出しを読んで動かすところ (`.amd-npu1` の名前、`cache.AMDTarget` の世代ごとの切り替え、`iron_export_lm.py` の xclbin を段ごとにした書き出しの読み込み) がまだない。xdna1 の機械で確かめてから `recipe.Built` に足す
- **AMD の書き出しのキー**: 手で作った書き出し (Anima、DA3、SigLIP2 など) のキーはいまも名前だけで、ほかの多くもコードの版だけ。レシピの args を変えても、ピンのないものは前の書き出しを取ってしまう。args をキーに入れる (取得元の書き出しを作り直してピンし直すことになる)
- **道具の版 (AMD)**: mlir-aie と Python の版 (`tools.iron`)
- **Core AI の書き出しの固定**: `hwnpu-dev models pin` は AMD だけ

## 7. 確かめ方

- `go test ./internal/recipe`: 組み込みのレシピが読めること、M1 / M2 / M3 で規則が分かれること、AMD のコマンドの組み立て (置き場所、variants、世代の extra、skip)、AMD のレシピが変わったら版も上がっていること (`testdata/amd-versions.txt`)、Apple の `convert` か `code` が変わったら版も上がっていること (`testdata/apple-versions.txt`)
- `go test ./cmd/hwnpu-dev/modelscmd`: Core AI の書き出しのあるモデルに、どれも `apple.version` があること
- `tools/coreai/same-exports.sh BASE MODEL...`: 変える前の commit と今のコードで書き出しを作って比べる。同じなら版を上げずに `-update` してよい。作るのは M1 の書き出し。`CHIP=M2` を付けると、ほかのチップの書き出しで比べる (M1 だけ書き出しが違うモデル)
- `go test ./internal/chat`: レシピの名前が models.yaml のモデルであること、`iron_export_lm.py` のレシピが `AMDExport` になること (`like` と variants も)
- `go test ./internal/prep`: AMD の NPU で動かすモデルに、コードが読むスクリプトのレシピがあり、置き場所がコードの入れるものであること。コマンドが前のコードのものと同じこと
- 作ったプリビルドが端末で仕上がるかは、オフラインではわからない。そのチップの Mac で Core AI のキャッシュを空にして描き、「the prebuilt program failed」が出ないこと、DiT の各部品が数十秒で開くことを見る ([known-issues](known-issues.md) 12 節)
