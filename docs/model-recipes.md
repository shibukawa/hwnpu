# モデルのレシピ (`model-recipes.yaml`)

モデルの変換を、誰がどの機械で作っても同じものにするための定義。作るときにしか要らないことを、ここに書く。動かすときに要ることは models.yaml に置く。

- **models.yaml**: 動かすときに要るもの。モデルの取得元、メモリ、サイズ (`npu.*.sizes`)、固定した書き出し (`exports`)
- **model-recipes.yaml**: 作るときにだけ要るもの。どのチップ向けに作るか、何本同時に作るか、通らないプログラムをどうするか、道具の版

どちらもリポジトリの直下にあって、バイナリに組み込まれる (`hwnpu.ModelsYAML`、`hwnpu.RecipesYAML`)。読むのは `internal/recipe`。

## 1. 作るコマンド

```sh
hwnpu-dev models package FLUX.2-klein-4B
```

- レシピがあるモデルは、ターゲットを書かなくてよい。Core AI の書き出しを作り、レシピのチップ (`archs`) 向けにプリビルドし、取得元 (`HWNPU_MODEL_SOURCE …?writeback=true`) に置く
- `MODEL/coreai-h14g` のようにチップを書くと、そのチップだけを作る。並列数と、実機に回すプログラムは、そのときもレシピのとおり
- 手元の書き出しだけをプリビルドするときは `hwnpu-dev prebuild -put EXPORT_DIR`。どのモデルのレシピかは、書き出しの `entry.json` の `model` から決める (`-model NAME` で指定もできる)
- `-arch` と `-j` は、レシピより優先する。確かめのとき以外は書かない

## 2. ファイルの形

```yaml
version: 1

tools:
  coreai-build: 3605.6.3        # coreai-build --version。違う版では止まる (--any-tools で作れる)

apple:                          # どのモデルにも効く既定
  archs: [h13g, h14g, h15g]     # プリビルドするチップ (h13g M1、h14g M2、h15g M3)
  jobs: 2                       # coreai-build を何本同時に動かすか

models:
  Chroma1-Flash:                # models.yaml のモデルの名前
    apple:                      # archs と jobs は既定より優先し、prebuild は既定のあとに足す
      prebuild:
        - arch: h14g                                    # 1 つか並び。省くと全部
          export: "chroma1-flash-8sizes-*"              # 書き出しの名前のパターン。省くと全部
          programs: [dit_single_0, dit_single_8, dit_single_16]   # プログラムの名前のパターン (.aimodel なし)
          do: device                                    # プリビルドせず、端末のコンパイラに任せる
          why: "left off the Neural Engine (no compiled region), alone too"   # coreai-build が何をしたか
          checked: 2026-11-23                           # それを見た日
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

## 3. 通らないプログラムがあったとき

`hwnpu-dev models package` と `hwnpu-dev prebuild` は、レシピにないプログラムが落ちると止まる。

- 落ちたプログラムがある書き出しは、そのチップ向けのものを取得元に置かない。人によって置かれるものが変わらないようにするため。ほかのチップのものは置く
- coreai-build のエラーは出力ではなく os_log に出る。そこで、落ちた coreai-build のプロセスのログ (`/usr/bin/log show --predicate 'processID == PID'`) から、理由の行を拾って表示する
- 表示の最後に、そのプログラムを端末に回すときのレシピの規則を示す。プログラムの作り方を直すか、その規則を model-recipes.yaml に書き足して PR にする
- 2 本同時のコンパイルで NPU から外れたものは、Build が最後に 1 本ずつ作り直してから判断する (Chroma の double で見た)

## 4. 実機に回る仕組み

- `do: device` のプログラムはプリビルドしない。端末がそのプログラムのプリビルドを探しても見つからないので、端末のコンパイラがコンパイルする (初回だけ。そのあとは Core AI のキャッシュから)
- VAE (`vae*`)、文のエンコーダー (`text*`)、名前が `_device` で終わるプログラムは、レシピとは別に、いつも端末に回る (`coreai.LeftToDevice`)。動かすときにも使う決まりなので、コードに残している

## 5. いまのレシピの中身 (2026-11-23)

| モデル | M1 (h13g) | M2 (h14g) | M3 (h15g) |
|---|---|---|---|
| FLUX.2 klein 4B | `dit_single_*` を端末で (RegAlloc の L2 の割り付けに失敗) | 全部プリビルド | 全部プリビルド |
| Chroma1-Flash / HD | `dit_single_*` を端末で | `dit_single_0/8/16` を端末で | 全部プリビルド |
| MOSS-SoundEffect v2.0 (`-lut8` も) | sin は級数 | sin は級数 | sin はそのまま (`-sin-native` の書き出し) |

MOSS の DAC の sin は、coreai-build で 5 つの段を見た。sin をそのまま書くと、h13g、h14g、h14c (M2 Pro・Max) は 1 段が Neural Engine の 8〜9 個の区間に分かれる (sin がその間で外に出る)。h15g/c、h16g/s/c、h17g/s/c/d は 1 区間に収まる (h15s/m/d と h17p は coreai-build が落ちるので見ていない)。M3 では、この書き出しを取ってきて音を作れた。級数はどのチップでも 1 区間。sin の前の clip は、sin とまとめられて (NEFUSED_EW: EW: MAX) h15g でも通らないので、そのまま書くときは clip を外す (clip は丸めのためのもの)。音声のプリビルドは置かない (M1 で遅いか壊れていた) ので、DAC は全部端末に回す規則にしてある。

どれも、落ちるまでに 1 本 1〜3 時間かかっていた。レシピに書いたので、もう試さない。

## 6. まだレシピにないもの (次の PR)

- **ほかの置き換え**: 2 節の `rewrite` は、いまは MOSS の DAC の sin だけ。ほかはコードに直接書いてある (softmax を exp と逆数で、x / c を 1 / c 倍で、sigmoid と tanh を exp で、conv_transpose を polyphase で、など)。M1 と M2 以降で違うものから、書き方を `Ways` に足して移す。sin の表引きも、そのときに書き方として足す
- **変換の設定**: 量子化の形式 (`npu.apple.formats`)、FLUX.2 klein の編集の組み合わせなど。いまは models.yaml とコードにある
- **AMD**: `iron_*.py` のスクリプトと引数。いまは models.yaml の `amd_export` (チャットのモデル) と Go のコード (画像、深さ、ラベルなど)、md のコマンドの例にある。次の形を考えている:

  ```yaml
  models:
    Anima-Turbo:
      amd:
        script: iron_anima.py
        args: [--size, "{sizes}", --dit, "{file}"]   # {sizes} は models.yaml の npu.amd.sizes から
  ```

  手で作った書き出し (Anima、Chroma、DA3、SigLIP2 など) のキーは、いまは名前だけ。引数を変えても名前が同じなので、レシピの引数をキーに入れる
- **道具の版 (AMD)**: mlir-aie と Python の版 (`tools.iron`)
- **Core AI の書き出しの固定**: `hwnpu-dev models pin` は AMD だけ

## 7. 確かめ方

- `go test ./internal/recipe`: 組み込みのレシピが読めること、名前が models.yaml のモデルであること、M1 / M2 / M3 で規則が分かれること
- 作ったプリビルドが端末で仕上がるかは、オフラインではわからない。そのチップの Mac で Core AI のキャッシュを空にして描き、「the prebuilt program failed」が出ないこと、DiT の各部品が数十秒で開くことを見る ([known-issues](known-issues.md) 12 節)
