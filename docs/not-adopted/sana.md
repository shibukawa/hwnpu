# Sana (Sana 600M、SANA 1.5 1.6B、Sana-Sprint)

## 状態

- **削除済み (コードなし)**。
- Sana 600M と SANA 1.5 1.6B は 2026-11-18 に外しました (#222、`bb5da8d5`)。Qwen-Image と一緒です。
- Sana-Sprint 0.6B と 1.6B は、それより前の 2026-11-11 に外しました (#109、`011e94fb`)。

## モデル

| モデル | リポジトリ | ライセンス | 大きさ |
|---|---|---|---|
| Sana 600M | `Efficient-Large-Model/Sana_600M_1024px_diffusers` | Apache-2.0 | DiT 0.6B、文は Gemma 2 2B |
| SANA 1.5 1.6B | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` | Apache-2.0 | DiT 1.6B、文は Gemma 2 2B |
| Sana-Sprint 0.6B / 1.6B | `Efficient-Large-Model/Sana_Sprint_*` | Apache-2.0 | 2 ステップに蒸留した版 |

- 文の符号器は Gemma 2 2B です。そのため、Gemma の利用規約と禁止用途の方針 (性的な内容は不可) が付いてきます。
- VAE は DC-AE (潜在は 32 分の 1) です。自己注意は線形の注意です。

## どの config で動かしていたか

最後に models.yaml にあったのは `2d5e46f6` です (`git show 2d5e46f6:models.yaml` の `Sana-600M:` と `SANA1.5-1.6B:`)。

| | Sana-600M | SANA1.5-1.6B |
|---|---|---|
| family / tag | `sana` / `0.6b` | `sana` / `1.6b` |
| machine | 8 (8 GB の `image` の既定) | 16 |
| memory (GB) | AMD 2.4、Apple 4.4 | AMD 2.3、Apple 4.5 |
| bytes | AMD 11.5e9、Apple 8.9e9 | AMD 16.5e9、Apple 9.7e9 |
| ステップ | 20、CFG (`--guidance` 既定 4.5、`--negative`) | 同じ |

- `image` の用途の `picks` で、8 GB の既定が Sana-600M でした (`{machine: 8, model: Sana-600M}`)。今は FLUX.2 klein 4B です。
- Apple の形式 (Core AI):
  - Gemma 2 は行ごとの int8 です。グループつきの int8 にすると CPU に落ちました。
  - DiT は 8 ビット (int8 / lut8) です。
  - 交差注意の K と V は、caption のプログラムで fp16 で計算していました。
- AMD の書き出し: `iron_sana.py --size` で 8 つのアスペクト比を 1 つにまとめていました (`sana-600m-aspects.amd-npu2` は 11 GB、1.6B は 16 GB)。

## 実装状況 (外した時点)

**AMD (Ryzen AI 9 365)**
- テキスト (Gemma 2、q8)、caption、DiT、VAE が全部 NPU で動きました。
- 600M は 20 ステップで1枚 35 秒です (1:1s は 19 秒)。最大 2.03〜2.23 GiB でした。
- 1.6B は1枚 1 分 01〜02 秒、最大 2.03〜2.14 GiB でした。
- CPU の参照と比べた潜在のコサインは次のとおりです。
  - Sprint 0.6B: 0.997
  - Sprint 1.6B: 0.963
  - 600M: 0.983
  - 1.6B: 0.984

**Apple (Core ML、のちに Core AI)**
- Core ML の時点で、diffusers の fp32 と同じ雑音から比べた結果です。

| モデル | 潜在のコサイン | 画像の PSNR |
|---|---|---|
| Sprint 0.6B | 0.986 | 26.0 dB |
| Sprint 1.6B | 0.990 | 27.0 dB |
| 600M | 0.998 | 32.7 dB |
| 1.6B | 0.984 | 22.6 dB |

- Core AI では 512² と 1024² を1つの書き出しにまとめていました。600M は 4.0 GB で、どの部品も `all on the NPU` でした。
- 512² は、コンパイル済みなら1枚 7.0 秒でした (M1)。
- VAE の初回のコンパイルは長く、M1 で 21 分かかりました。
- `coreai-build` (プリビルド) は、そのままでは 27 本のうち 19 本を ANE に載せませんでした。下の手当てで 34 本が全部通りました (M2、h14g)。ただ、空のキャッシュから NAS の書き出しとプリビルドを使う通しの確認はしていません。

**Intel**
- 画像生成はしていません (Intel の画像は Anima だけです)。

## ロジックが残っているコミット

- **最後にコードがあったコミット: `2d5e46f6d26bc8c073e9be07b7d650f6b899ac7c`** (#222 の親)。
  - Go: `internal/imagegen/sana/` (変換、パイプライン、`progEngine` と `ironEngine`、スケジューラ)、`internal/imagegen/sana/mil/` (Gemma 2、caption、DiT、VAE の MIL)、`cmd/hwnpu-image/sana.go`
  - AMD: `tools/amd/iron_sana.py`、`tools/amd/iron_dwconv.py` (深さ方向の畳み込み)
  - 参照: `tools/ref/ref_sana.py`
  - 記録: `docs/design.md` の 16.119、16.122、16.138、16.139、16.145、16.148、16.152、16.160、16.189、16.224、16.233、16.240
- **Sana-Sprint が最後にあったコミット: `326f46a615f9e84f5ae34109540e1705c0464146`** (`011e94fb` の親)。SCM の刻み (`SCMTimesteps`) と guidance の埋め込みが入っています。
- **今も main に残っているもの:**
  - IRON の部品は `tools/amd/iron_dit.py` に移しました (`Checkpoint`、`qparam`、`VaeStage` など。Anima、FLUX.2 klein が使っています)。
  - AMD のカーネル (`tools/amd/kernels/dwconv3.cc` など) も残しています。チャットの書き出しの鍵 (`amdtools.LMHash`) が全部のカーネルを含むので、消すとチャットの書き出しが全部作り直しになるからです。
  - `mil.repeatChannels` (チャンネルの繰り返しを 0/1 の 1 × 1 の畳み込みで書く) は MIL の共通部分に残っています。

## 問題と外した理由

- **外した理由:** Qwen-Image をやめたときに一緒にやめました。8 GB の枠は FLUX.2 klein 4B が取りました。FLUX.2 klein のほうが絵がよく、編集もでき、M1 の 1024² で 4.9 GB に収まります。
- **Sana-Sprint を外した理由:** 2 ステップで速い (1024² で 12〜32 秒) のですが、絵の質が Sana よりかなり低かったからです。
- **見つかった問題** (今のモデルにも当てはまるものは [image-apple.md](../design/image-apple.md) の「Lessons kept from dropped models」に残っています):
  - IRON の MHA はスコアを bf16 で持ちます。600M の交差注意はスコアが約 1750 で、行の中の差が数単位しかなく、CFG で崩れました。鍵から先頭の鍵を引いて直しました。
  - ANE の深さ方向の畳み込みは 4096 チャンネルまでです。5760 チャンネルは CPU に落ちました。
  - 時刻の埋め込みを ANE で計算すると 1 % ずれました。ホストの float32 で計算しました。
  - Gemma 2 の 128 行の文の関数で、ANE が 1〜2 分止まったあと `Request cancelled` になりました。そのあとは暇になるまで、どのプロセスでも失敗し続けました。原因は分かっていません。
  - Core AI では、1024² の `vae_s0` が `empty reset layer candidates in SS cost model` で GPU に回りました。チャンネルの繰り返し (`tile`) を 0/1 の 1 × 1 の畳み込みにして直しました。
  - `coreai-build` のための手当てが要りました。
    - DiT の出力に `× 1` を入れる (`redefinition of symbol named 'hwnpu_N'` への対策)
    - VAE の norm の最大は、チャンネルを最後の軸に移してから取る
    - 1 段に 1 ブロック
    - multiscale の群の 1 × 1 の畳み込みを、ブロック対角の密な 1 × 1 にする

## 再開するときの手順と注意

1. `2d5e46f6` から `internal/imagegen/sana`、`cmd/hwnpu-image/sana.go`、`tools/amd/iron_sana.py`、`iron_dwconv.py`、`tools/ref/ref_sana.py` を戻します。models.yaml の2つの項目も戻します。
2. 今の作りに合わせて直す必要があります。外したあとに入った仕組みは次のとおりです。
   - 8 サイズを1つの書き出しにまとめる (`internal/imagegen/aspect`)
   - model-recipes.yaml の `apple` と `amd` (`version`、`code`、`convert`)
   - hwnpud がキャッシュを一手に書く仕組み
   - `iron_dit.py` に移した共通部品
3. Gemma 2 の利用規約が文の符号器に付くことを、models.yaml の `license` に書きます。
4. 8 GB の枠に戻すなら、FLUX.2 klein と絵とメモリを比べてからにします。
