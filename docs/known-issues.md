# 既知の問題と回避策: 根本で直ったときにすること

作成: 2026-11-19。対象: Apple の Core AI (macOS 27.2.1 26B131) と `coreai-build` (Metal Toolchain 27.2.47.0、coreai-build 3605.6.3)。

hwnpu が今どう回避しているか、Apple の側で直ったときに hwnpu で何をするか、を問題ごとにまとめる。調べた経緯と数値は [docs/design/README.md](design/README.md) の各節にある。Apple への報告 (Feedback Assistant) の番号も書いた。未報告のものの下書きと再現の手順は [apple-feedback/](apple-feedback/README.md)。

OS や Metal Toolchain を上げたら、まず最後の「確かめ方」で各問題がまだあるかを見て、直ったものだけをこの文書の手順で外す。

## 一覧

| # | 問題 | 報告 | 今の回避 | 直ったら |
|---|---|---|---|---|
| 1 | prebuilt の仕上げが機械全体で 10 秒に 1 本 | FB25624129 | なし (prebuilt は大きな部品に効くので置く) | 測り直して、画像以外のモデルにも置くか決める |
| 2 | 隣り合う同じ lut の表で `coreai-build` が ANE をあきらめる | FB25624151 (追記あり) | 表の 1 つを 1 ulp ずらす (`mlpackage.DistinctNeighbours`) | 外さなくてよい。外すなら FormatVersion を上げて全部作り直す |
| 3 | `coreai-build` が通した `.aimodelc` が端末で仕上がらない (`vae_s0` 1024²) | FB25624294 | 端末のコンパイルに戻る (自動、約 20 秒) | 何もしなくてよい (置いた prebuilt のまま効く) |
| 4 | prebuilt から仕上げたプログラムの 1 回の呼び出しが遅い | FB25624318 | なし (8〜16 %、M1) | 何もしなくてよい。チャットのモデルの判断に入れる |
| 5 | ANE のコンパイルと prebuilt の仕上げが機械全体で 1 本ずつ | FB25624259 | なし | 読み込みを並べるか測る |
| 6 | 22 種類のうち 9 種類で `coreai-build` が落ちる | FB25624336 (追記あり) | 通る 12 種類だけ `-arch` で名指しして作る | その種類を作って置く |
| 7 | h18g (M6) で ANE に乗らない | FB25625077 (FB25624336 にも) | h18g は作らない (端末でコンパイル) | h18g を作って置く |
| 8 | ディスクが少ないと、仕上げたばかりのプログラムを macOS が消す | 未報告 (下書き [purged-programs-silent-recompile.md](apple-feedback/purged-programs-silent-recompile.md)) | 25 GB 以上空ける (人の運用) | 何もしなくてよい |
| 9 | 画像以外のモデルでは prebuilt が遅い (チャットには止まるものもある) | 報告済み ([apple-coreai](design/apple-coreai.md)) | 画像生成のモデルにだけ prebuilt を置く | 測り直して、置くなら作って置く |
| 10 | M1 (と M2: `sin`) の ANE のコンパイラが `sin` と、まとめた要素ごとの演算の一部を受け付けない | 未報告 (下書き [sin-off-ane-m1-m2.md](apple-feedback/sin-off-ane-m1-m2.md)) | MOSS の DAC の snake を級数で、段 3 を 4500 フレームに | 何もしなくてよい |
| 11 | `coreai-build` が、端末のコンパイラが ANE に乗せるプログラムを GPU に回す ("NEFUSED_EW: EW: MAX") | 未報告 (下書き [offline-stricter-than-device.md](apple-feedback/offline-stricter-than-device.md)) | オフラインでも通る形で書く (DAC r3) | 何もしなくてよい。r2 の形に戻す理由はない |
| 12 | h15g (M3) 向けの `.aimodelc` が、DMA のキャッシュの指定で端末の検査に落ちる (FLUX.2 klein 512² の attention) | 未報告 (下書き [h15g-dma-cache-hint.md](apple-feedback/h15g-dma-cache-hint.md)) | 1536 行を 768 行ずつ 2 回に分ける (`flux2klein` の `headChunk`、ExportVersion 4) | 何もしなくてよい |
| 13 | h13g (M1) 向けの 8192 キー以上の attention の `.aimodelc` が、M1 で雑音を計算するか、仕上げのサービスを落とす (FLUX.2 klein の 1024 の参照 1 枚の編集) | 未報告 (下書き [h13g-attention-8192-keys.md](apple-feedback/h13g-attention-8192-keys.md)) | M1 の書き出しにはその関数を入れず、参照を 512 で読む (models.yaml の `generation_edits`) | M1 の上書きを消す |
| 14 | 1024² の VAE の段の ANE のコンパイルが 1 段最大 15 分、ディスクを最大約 50 GB 使う (Anima) | 未報告 (下書き [compile-disk-and-time.md](apple-feedback/compile-disk-and-time.md)) | なし (初回は 70 GB 以上空ける) | VAE の prebuilt を M2 以降で使うか測り直す (#336) |
| 15 | 仕上げのサービスが落ちるたびに、launchd がその再起動を待たせる時間が伸びる (10 秒 → 80 秒 → … → 20 分)。再起動まで戻らない | 未報告 (下書き [finisher-crash-throttle.md](apple-feedback/finisher-crash-throttle.md)) | サービスを落とす prebuilt を置かない。測る前に今の待ちを読む | 1 節と 9 節を測り直す |
| 16 | h13g (M1) 向けの `.aimodelc` で、活性化のタスクが、ほかの活性化と分け合うカーネルの外を読む。M1 では最初の呼び出しが終わらず (100 秒で打ち切り。その間ほかのプロセスの呼び出しも止まる)、関数が 2 つ以上だと仕上げのサービスが落ちる (Qwen3.5、LFM3、Gemma 4 のチャット。13 も同じと見られる) | 未報告 (下書き [h13g-kernel-overrun.md](apple-feedback/h13g-kernel-overrun.md)) | 活性化を同じ並びで書く (`blocks.WideAct` など。[design/apple-neural-engine.ja.md](design/apple-neural-engine.ja.md) の「活性」)。`hwnpu-dev prebuild` は、そのタスクのあるプログラムを置かない (`llirtasks -check`) | 何もしなくてよい (そろえた書き方に害はない) |
| 17 | 16 と同じ原因で、Gated DeltaNet の層を持つ関数が 2 つ以上のプログラムの prebuilt が、仕上げのサービスを落とす (M1) | 未報告 (16 の下書きに含む) | 16 と同じ | 何もしなくてよい |
| 18 | 定数を左に置いた行列積が、prebuilt から仕上がらない ("Kernel evaluation failed"、M1) | 未報告 (16 の下書きに添える) | 定数を右に置く (`paged` の mass プログラム) | 何もしなくてよい |

## 1. 仕上げの 10 秒の間隔 (FB25624129)

- **何が**: `.aimodelc` を端末で仕上げる `ANELargeModelCompilerService` は 1 本ごとに終わり ("Exiting … to avoid VA fragmentation")、launchd が次の起動を 10 秒待たせる ("service spawn deferred by 10 seconds due to throttle")。小さな部品でも 1 本 10.2 秒かかる。[docs/design/image-apple.md](design/image-apple.md), [docs/design/quantization.md](design/quantization.md) のころの調べ
- **今**: 回避はない。prebuilt が効くのは、1 部品の端末のコンパイルが数分かかる画像のモデル (FLUX.2 klein、Anima) だけ。部品の多い小さなプログラムは、10 秒ずつ積み上がって負ける (画像以外は 9 節)
- 10 秒は下限で、いつもそうとは限らない: このサービスが落ちると間隔が伸びる (15 節)
- **直ったら**: FLUX.2 klein の文の符号器 (prebuilt で 112 秒、端末で 81 秒、[distribution](design/distribution.md)) と、9 節の表のモデルを同じ測り方で測り直す。端末のコンパイルより速くなれば、9 節の手順で置く

## 2. 隣り合う同じ lut の表 (FB25624151)

- **何が**: `coreai-build` の ANE のコンパイラは、`lut_to_dense` の重みの出力 256 行ごとの区切りで、先頭から続く同じ表の数を「表を共有する行の数」と取り違え、次の組が違うと "EmitGlobalBackedCoalescedLutTensor: per-cout group 1 is not content-uniform" で落ちて GPU に回す。どの種類、lut4/6/8、どの組の大きさでも起きる。端末のコンパイラは正しい。FLUX.2 klein の `dit_single_0` に 1 組あった。[docs/design/quantization.md](design/quantization.md)
- **今**: lut を書くとき (`mlpackage.Builder.LUT`)、前の組と同じ表の値を 1 つだけ 1 ulp ずらす (`mlpackage.DistinctNeighbours`)。これで書くものが変わったので `ailower.FormatVersion` を 2 にした
- **直ったら**: 外さなくてよい。ずれは表 1 つの値 1 つで、絵も答えも変わらなかった。外すのは、回避が別の問題を起こしたときだけ。外すなら:
  1. `DistinctNeighbours` の呼び出しを除き、`ailower.FormatVersion` を 3 に上げる (書くものが変わるので、取得元の鍵も変わる)
  2. 6 節の「作り直し」をする (書き出し、prebuilt、チャットのモデルのすべて)
  3. 直った Metal Toolchain を、prebuilt を作る機械の最低限として docs に書く (古い `coreai-build` では同じ部品がまた GPU に回る)
- 確かめ方: `tools/coreai/lutfake -dup N` (合成の 1 層) は `Builder.LUT` を通るので、今は再現しない。再現には `DistinctNeighbours` を外した hwnpu で書くか、Apple への追記に添えた再現 (lut8 の 256 × 256 の 1 層。リポジトリにはない) を使う。モデルにその並びがあるかは `tools/coreai/lutscan -tile 16` で見る

## 3. 端末で仕上がらない `.aimodelc` (FB25624294)

- **何が**: FLUX.2 klein の `vae_s0` の 1024² の関数は、`coreai-build` は通るのに、端末の仕上げで "Too many kernel bytes" から `failedToSpecialize` になる
- **今**: hwnpu は失敗を見て、その部品だけ端末でコンパイルする ("the prebuilt program failed (failedToSpecialize), compiled on the device")。約 20 秒余計にかかる
- **直ったら**: 何もしなくてよい。置いてある prebuilt のまま、その行が出なくなる

## 4. prebuilt から仕上げたプログラムの呼び出しが遅い (FB25624318)

- **何が**: prebuilt から仕上げたものは LLIR の道 (`useANELLIR` が true) を通り、中間のバッファが大きい (VAE の段で 549 MB、端末のコンパイルでは 411 MB)。1 回の呼び出しが M1 で 8〜16 %、M2 の Chroma1 (外した) で 3〜5 % 遅い
- **今**: 回避はない。初回の短縮 (FLUX.2 klein の M1 で 40 分 → 8 分) の方が大きいので置いている
- **直ったら**: 何もしなくてよい。チャットのモデルに置くかの判断 (9 節) では、呼び出しの速さが変わらないことも条件になる

## 5. コンパイルと仕上げが 1 本ずつ (FB25624259)

- **何が**: aned は ANE のコンパイルも prebuilt の仕上げも、機械全体で 1 本ずつ回す。hwnpu が並べて頼んでも待つ
- **今**: 回避はない (並べても速くならないので、読み込みは順に頼んでいる)
- **直ったら**: 部品を並べて開く (Core AI の読み込みの先回り、#171) と初回が縮むか測る。縮むなら、並べる数を決めて既定にする

## 6. 9 種類で `coreai-build` が落ちる (FB25624336)

- **何が**: `--preferred-compute neural-engine` で h13c/d/s、h14d/s、h15d/m/s、h17p の 9 種類が SIGSEGV (`ZinIrCompilerParameters::PropogateGlobalOptionsToPerProcedureOptions`) で落ちる。`--architecture` を省くと最初の h13c で落ちて全体が止まる。[docs/design/apple-coreai.md](design/apple-coreai.md), [docs/design/quantization.md](design/quantization.md)
- **今**: prebuilt は通る 12 種類 (h13g、h14c/g、h15c/g、h16c/g/s、h17c/d/g/s) だけ、`-arch` で名指しして 1 種類ずつ作る。落ちる種類の Mac (M1 Pro/Max/Ultra、M2 Pro/Ultra、M3 Pro/Max/Ultra、A18) は端末でコンパイルする
- **直ったら**: その種類の prebuilt を作って置く。書き出しは変わらないので、prebuilt だけ:

  ```sh
  # 1 種類ずつ作り、できたらすぐ取得元に置く (12 種類を一度に作ると、最後まで何も置かれない)
  export HWNPU_MODEL_SOURCE='https://…/llm?writeback=true'
  for a in h13c h13d h13s h14d h14s h15d h15m h15s h17p; do
    hwnpu-dev models package ~/Library/Caches/hwnpu/models/FLUX.2-klein/flux.2-klein-4b-8sizes-4d02f40a-5edits-fa7f0afe-lut8-lut8-v4.coreai/coreai-$a
  done
  ```

  Anima も同じに (書き出しが取得元にあるもの)。全種類が通るようになれば `--arch all` でよい
- **作り直し** (書き出しが変わったとき。2 節、FormatVersion を上げたとき): 書き出しを作って置き (`hwnpu-image` か `tools/coreai/kleinonly prepare` を取得元の書き戻しありで)、その書き出しに上の prebuild をする。チャットのモデルは `hwnpu-chat` を書き戻しありで 1 回ずつ動かす (9 節)。2026-11-19 の作り直しの記録は [docs/design/distribution.md](design/distribution.md)
- 確かめ方: `xcrun coreai-build compile X.aimodel --platform macOS --architecture h13c --preferred-compute neural-engine --output out/` が落ちず、`out/…/manifest.plist` に `mps.fullyPlacedOnANE` があること。失敗の理由は stderr でなく os_log (プロセス `coreai-build`、ANEHostCompiler) に出る

## 7. h18g (M6) で ANE に乗らない (FB25625077、FB25624336)

- **何が**: h18g は落ちないが、どのモデルも "failed to verify module" で ANE をあきらめ、GPU だけの `.aimodelc` を書いて 0 で終わる
- **今**: h18g の prebuilt は作らない (GPU だけのものを置くと、M6 で ANE を使わなくなる)。M6 は端末でコンパイルする
- **直ったら**: 6 節と同じに `--arch h18g` で作って置く。置く前に、`manifest.plist` に `mps.fullyPlacedOnANE` があることを必ず確かめる (GPU に回っても終了コードは 0)

## 8. ディスクが少ないと仕上げたプログラムが消える (未報告)

- **何が**: 空きが少ないと macOS の `deleted` が消せるものを消す (CD_NEAR_LOW_DISK_EVENT)。Core AI のキャッシュと aned の仕上げたプログラムも消える。仕上げた直後に消えると aned が `Model load failed … isPreCompiledModel=1` を出し、Core AI が OS の中で端末のコンパイルに切り替える。M1 で初回の最初のステップが 5〜15 分かかった。22 GB 空いていても消し始めた。[docs/design/distribution.md](design/distribution.md)
- **今**: hwnpu では直せない (失敗が見えない)。初回を測るとき、使うときは 25 GB 以上空ける
- **直ったら**: 何もしなくてよい。Core AI のキャッシュの方針を `.persistent` にすれば ([docs/design/apple-coreai.md](design/apple-coreai.md)) Core AI の分は消えなくなるが、aned の分は残るので、それだけでは足りない

## 9. 画像以外のモデルに prebuilt を置かない ([apple-coreai](design/apple-coreai.md))

- **何が**: 画像生成以外のモデルは部品が小さく、端末のコンパイルが短い (長くて数分)。1 つめの prebuilt の決まった費用と 1 節の間隔で負ける
  - チャット: LFM3-1.7B は端末で 2 分 17 秒、prebuilt で 8 分 12 秒。Qwen3.5-4B は prebuilt で読めたが、最初の呼び出しから答えなかった
  - M1 (2026-11-26): Qwen3.5-0.8B は端末で 7 分 5 秒〜7 分 11 秒。prebuilt は読み込みが 3 分 56 秒 (3 部品は端末のコンパイルに戻った) で、最初の呼び出しが失敗した。Neural Engine の上で呼び出しが終わらなくなっていた (16 節)。Qwen3.5-4B (M3) の「答えなかった」が同じものかは分からない (16 節のタスクは h13g 向けのプログラムにしかない)
  - M1 (2026-11-27): 16 節と 18 節を直した書き出しでは、`model.part1` (関数 2 つ) が prebuilt から 8.2 秒で仕上がって動いた。モデル全体の初回の時間はまだ測っていない (15 節の間隔が 20 分に伸びたままで、再起動するまで 1 部品 20 分かかる)
  - 「prebuilt の方が遅い」のは、1 節の間隔が 10 秒より伸びていたせいかもしれない (15 節: 仕上げのサービスが落ちると伸び、再起動まで戻らない)。測るときは今の間隔を先に読む
  - 解析と音声 (2026-11-22、M3、h15g、空のキャッシュから 1 回目の読み込み):

    | モデル | 端末でコンパイル | prebuilt |
    |---|---|---|
    | YuNet | 2.3 秒 | 15 秒 |
    | DA3-BASE | 3 分 00 秒 | 4 分 45 秒 |
    | whisper-large-v3-turbo | 53 秒 | 5 分 27 秒 |
    | LFM3-Audio-2B-TTS | 45 秒 | 測れず (ディスクが尽きて ANE のコンパイラが落ちた、8 節) |

  - `coreai-build` で作れないものもある: basic-pitch と RTMO-s は ANE に何も乗らない、magenta-realtime-2-small の復号器は `coreai-build` が落ちる。PP-OCR、EfficientSAM、MoGe-2 は書き出しが最初の読み込みでできるので、先に端末で一度読まないと作れない
  - M1、M2 では測っていない
- **今**: 画像生成のモデル (FLUX.2 klein、FLUX.3 klein、Anima) にだけ prebuilt を置く。model-recipes.yaml の `apple.prebuilt: true` がその印で、`hwnpu-dev models package MODEL` と `hwnpu-dev prebuild EXPORT` は、チップを名指さなければ印のあるモデルだけをプリビルドする。ほかは書き出し (`.aimodel`) だけを取得元に置く。Mac は自分の種類の prebuilt があれば自動で取りにいくので、置くと初回が遅くなる (2026-11-22 に h15g で作った 17 モデル分は消した)
- **直ったら** (1 節、4 節、Qwen3.5-4B の止まりが直ったら): [apple-coreai](design/apple-coreai.md) と同じ測り方 (署名の識別子を変えたコピーで空のキャッシュから) で、LFM3-1.7B、plamo-3-translate-8b、Qwen3.5-4B と上の表のモデルを測る。速くなっていれば、その書き出しにも 6 節の prebuild をする (書き出しは `~/Library/Caches/hwnpu/models/NAME/` の下)

## 10. M1 の ANE のコンパイラが受け付けない演算 (未報告)

- **何が**: Core AI で M1 (h13) の ANE のコンパイラは、`sin`、要素ごとの演算にまとめた clip ("NEFUSED_EW: EW: MAX")、定数倍した入力どうしの引き算 ("NEFUSED_EW: KernelPreScales") を受け付けず、プログラムごと GPU に回す。MOSS の DAC の段 3 は 1 回 4998 フレームで "unexpected failure" になる。M3 の端末のコンパイラは通す。[docs/design/audio-models.md](design/audio-models.md)。ほかの乗らない演算の一覧は [docs/design/apple-coreml.md](design/apple-coreml.md)
- **今**: DAC の snake の sin を π を単位にした級数で計算し、段 3 を 4500 フレームにした (`DACRevision` 3)。M1 でも M3 でも同じ書き出し
- **直ったら**: 何もしなくてよい (級数の値は CPU の参照と 0.0085 で、sin と同じ)
- 確かめ方: `sin` だけのプログラムを `coreai-build --architecture h13g` で作り、`mps.fullyPlacedOnANE` があるか

## 11. オフラインのコンパイラの方が厳しい (未報告)

- **何が**: 前の MOSS の DAC (r2、snake に `sin` と clip) の段 4 は、M3 の端末では全部 ANE に乗っていた ([docs/design/audio-models.md](design/audio-models.md))。しかし `coreai-build` では h15g、h16g、h17g のどれでも "Unsupported NE layer "NEFUSED_EW": EW: MAX" になり、GPU だけの `.aimodelc` を書いて 0 で終わる。先頭から 46 演算なら乗り、47 演算目 (畳み込み → reshape → チャンネルごとの mul の後の clip) を足すと落ちる。この 4 つだけのプログラムは乗り、`sin` を `mul(y, y)` にした段も乗る。前の `sin` と clip との組み合わせで起きる。[docs/design/audio-models.md](design/audio-models.md)
- **今**: DAC は、どの機種でもオフラインで通る形 (r3) にした。prebuilt を作るなら、オフラインで乗る形で書く
- **直ったら**: 何もしなくてよい
- 確かめ方: 段 4 の先頭 47 演算 (`aicheck merge` を `HWNPU_DEV=ailower.cut=47` で) を h15g で作り、`mps.fullyPlacedOnANE` があるか

## 12. h15g の DMA のキャッシュの指定が端末の検査に落ちる (未報告)

- **何が**: FLUX.2 klein の DiT (8 つの大きさの関数を持つプログラム) の h15g 向けの prebuilt が、M3 (macOS 27.2.1) で全部 `failedToSpecialize` になった。Core AI は関数が 1 つ仕上がらないとプログラムごと使わないので、DiT の 7 つの部品がどれも端末でコンパイルされた (1 つ 3〜19 分。初回が prebuilt のないときと同じ)。落ちていたのは 512² の関数だけ。端末の仕上げ (ANELargeModelCompilerService) が "tile_dma_src.dma_config2.cache_hint != ane_tile_dma_src_dma_config2_cache_hint_drop_v8"、"Instruction corresponding to oplayer … failed validation test"、"Invalid TD exists" を出す。M1 で作ったものとこの機械で作ったものは、日付と統計のファイルのほかはバイト単位で同じで、どちらも落ちる。`--min-deployment-version 27.2` でも `--expect-frequent-reshapes` でも同じ
- **原因の形**: attention の、1536 個のキーに対する 512 行のクエリの塊 (1536 行を `HeadChunk` の 1024 行ずつに分けた残り)。attention だけのプログラムでも起きる (1 ヘッド、キーを分けなくても)。768 行ずつ、1536 行まとめて、1024 個のキーに 512 行、3072 個のキーに 512 行は通る。1792 個のキーに 1280 行と 512 行も落ちる。attention だけでは 3968 行や 4032 行も落ちたが、DiT の中では通る (周りの演算とまとめ方で変わる)。どの形で起きるかは、オフラインのコンパイラの割り付け次第で、読めない
- **今**: 行が 1 塊より多く 2 塊より少ないとき、半分ずつ (64 の倍数) にする (`flux2klein/mil` の `headChunk`)。FLUX.2 klein で変わるのは 512² の文からの生成だけ (1536 行 → 768 + 768)。計算は同じ (行ごとの softmax)。書き出しの版を 4 にした (`ExportVersion`。3 までは編集の置き方)。M3 で、直した 512² の DiT の 7 部品は prebuilt から全部仕上がり、直す前のものは全部落ちた。編集の 5 通り (2560 行から) は変わらない
- **作り直し**: 書き出しの名前が `-v4` になるので、FLUX.2 klein の書き出しと、各種類の prebuilt (6 節) を作り直して取得元に置く。v3 までの prebuilt は使われなくなる
- **直ったら**: 何もしなくてよい
- まだ見ていないもの: h15g 以外の種類の端末での検査 (この機械は h15g だけ)
- 確かめ方: `coreai-build compile … --architecture h15g` で作った `.aimodelc` を M3 で開き (AIModel(contentsOf:))、`/usr/bin/log stream` に "cache_hint" が出るか。attention だけなら 1536 行、`HeadChunk` 1024 のもので足りる

## 13. h13g (M1) 向けの 8192 キー以上の attention が端末で壊れる (未報告)

- **何が**: FLUX.2 klein の「1024² を 1024 の参照 1 枚で編集」の関数 (`s1024x1024_ref1`、8704 キー。キーの部分ごとの softmax を log-sum-exp で合わせる形、`milpart.PieceKeys`) を `coreai-build` で h13g 向けに作ると、オフラインでは全部 Neural Engine に載るのに、M1 (macOS 27.2.1、Metal Toolchain 27.2.47、coreai-build 3605.6.3) で正しく動かない
  - 単ブロックのプログラム: 仕上がって全部 Neural Engine で動くが、出力が雑音になる。絵は一面の雑音で、参照は何も残らない
  - 二重ブロックのプログラム: 仕上げのサービス (ANELargeModelCompilerService) がその関数の変換の終わり近くで落ちる (SIGSEGV、`ZinAneHloConverter::ParseInstructions`。13 関数のうち 2 番目、38060 個中 37838 個目のタスク)
  - Core AI は関数が 1 つ仕上がらないとプログラムごと端末でコンパイルし直す。13 関数の `dit_double_0` で 54 分かかり、その間にコンパイラが約 137 GB を書いた
- **切り分け** (単ブロック 2 個、関数 1 つのプログラム。M1 の Neural Engine の出力を CPU のインタープリタと比べた):

  | 関数 | キー | コンパイル | 余弦 |
  |---|---|---|---|
  | `s1024x1024` | 4608 | `coreai-build` h13g | 0.999999 |
  | `s1024x1024_refh1` (M1 の書き出しの、参照 1 枚を 512 で読む関数) | 5632 | `coreai-build` h13g | 0.999999 |
  | `s1024x1024_refh2` | 6656 | `coreai-build` h13g | 0.999999 |
  | `s1024x1024_ref1` | 8704 | `coreai-build` h13g | −0.0005 (非有限値 37924 個) |
  | `s1024x1024_ref1` | 8704 | M1 の上 | 0.999999 |

  同じプログラムを Core AI の CPU で動かすと 0.999996 なので、書き出しは正しい。M2 (h14g) と M3 (h15g) では、同じ書き出しのプリビルトで絵が正しく出る
- **今**: M1 の書き出しには 8192 キー以上の関数を入れない。1024² の絵の参照が 1 枚のとき、M1 では参照を 512 で読む (models.yaml の `npu.apple.generation_edits`、5632 キー)。書き出しは M1 用に 1 つ (名前の編集の hash が違う)。ほかのチップは今までどおり
- **確かめた結果** (2026-11-25、M1): M1 用の書き出しの h13g のプリビルト 14 本は、M1 の上で全部仕上がった (`the prebuilt program failed` は 0 回、DiT の端末でのコンパイルも 0 回)。空の Core AI キャッシュからの初回は 18 分 15 秒。文からの生成 5 サイズ (512²、1024²、1280 × 704、704 × 1280、768 × 1024) と編集 5 通りの絵が正しく出た。本物の `dit_single_0` (13 関数) の 2 つの関数も、CPU との余弦が 0.999999 だった。時間は [design/image-apple.ja.md](design/image-apple.ja.md)
- **16 節と同じ原因と見られる** (2026-11-27): その書き出しの h13g のプログラムの 8704 キーの関数には、カーネルの外を読むタスクが 1296 個ある (単ブロックのプログラムの 37224 タスク中、二重ブロックの 38060 タスク中)。`(1, 512, 1)` の並びで 16 組の係数を読む活性化が、1 組の活性化と 128 バイトのカーネルを分け合っている。M1 用の書き出し (動く) には 1 つもない。二重ブロックで仕上げのサービスが落ちる所 (37838 個目) も、そのタスクのあとである。attention の部分の活性化の並びをそろえれば、M1 用の書き出しが要らなくなるかもしれない (試していない)
- **直ったら**: `generation_edits` の M1 を消せば、M1 もほかのチップと同じ書き出しを使う (M1 用の書き出しとプリビルトは使われなくなる)
- 確かめ方: `flux2klein` の `TestWriteSingleProbe` で関数 1 つのプログラムと CPU 用のパッケージを作る (`HWNPU_TEST_KLEIN_PROBE_REFS=1`、`_GROUP=0,2`、`_REF=…mlpackage`)。`hwnpu-dev prebuild -arch h13g` で作り、M1 で `go run ./tools/coreai/aicheck run -cpu -ref REF OUT#s1024x1024_ref1` の余弦を見る。Core AI のキャッシュに端末でコンパイルした同じプログラムが残っていると、プリビルトは使われない (実行ファイルの名前を変えるか、その項目を消す)

## 14. 1024² の VAE の段のコンパイルが長く、ディスクを大きく使う (未報告)

- **何が**: Anima の VAE の段 (832 KB、8 つの大きさの関数) を M2 の端末でコンパイルすると、`vae_s6.aimodel#s1024x1024` だけで 871 秒かかり、その間に空きが 99.1 GB から 48.2 GB に減った (コンパイルが終わると戻る。2026-11-25)。10 段で約 70 分 (1 段 1〜15 分)。FLUX.2 klein の VAE は 10 段で 104 秒
- 空きが足りないと "Failed to allocate memory for file backing" で GPU に回る。その前に空きが 15 GB ほどを切り、8 節の消去が起きる
- **今**: 回避はない。Anima の初回は 70 GB 以上空けて動かす
- **直ったら**: 何もしなくてよい。VAE の prebuilt を M2 以降で使うか (#336) は、この問題とは別に測る
- 確かめ方: `tools/coreai/fbrepro/load.sh "vae_s6.aimodel#s1024x1024"` で時間と空きの最小値を見る ([apple-feedback/repro.md](apple-feedback/repro.md))

## 15. 仕上げのサービスが落ちるたびに待ちが伸びる (未報告)

- **何が**: 1 節の 10 秒は、launchd が `ANELargeModelCompilerService` の起動どうしの間に置く間隔である (`service spawn deferred by N seconds due to throttle` の N は、前の起動からその間隔までの残り)。この間隔は、サービスが落ちると伸びる。2026-11-26 の M1 では、午後は 10 秒だったものが、サービスが何度か落ちたあとの夜には 80 秒になっていて、その夜さらに落ちるたびに 320 秒、640 秒、1200 秒になった。80 秒は、何も動かさなかった 4 時間半のあいだも変わらなかった。落ちた直後の 1 回だけはすぐ起動し、その次から新しい間隔を待つ
- 仕上げは機械全体で 1 本ずつ、この間隔で進む。間隔が 80 秒なら 17 部品のチャットのモデルは 23 分、1200 秒なら 1 部品 20 分かかる。どのアプリの prebuilt も同じに遅くなる
- 9 節の「prebuilt の方が遅い」と、[apple-coreai](design/apple-coreai.md) の「プロセスの最初の prebuilt に数分の決まった費用があるらしい」は、これで説明がつく (その日にサービスが落ちていたかは、当時の記録にない)
- サービスを落としたもの (どれも端末の仕上げのメモリ破壊。`ZinKernelBuilder::BuildKernelFromLLIRData` の中の malloc の検査、`ZinAneHloConverter::ParseInstructions`、MLIR の後片付けなど、落ちる場所は毎回違う): 17 節のプログラム、13 節の二重ブロック、h13g の 8 つの大きさの Chroma。17 節と 13 節のものは、カーネルの外を読むタスク (16 節) のあるプログラムである
- **今**: 回避はない。サービスを落とすと分かっている prebuilt は置かない (レシピの `prebuild` の決まりで端末に回す)。prebuilt の時間を測る前、切り分ける前に、今の間隔を読む。10 秒より長ければ、再起動してから測る
- **直ったら**: 1 節と 9 節を測り直す
- 確かめ方: `/usr/bin/log show --last 30m --predicate 'process == "launchd" AND eventMessage CONTAINS "deferred by"' | grep ANELargeModelCompilerService`。prebuilt を 2 本続けて開いたときの N と、2 本目が開くまでの時間を足したものが間隔

## 16. h13g 向けのプログラムの活性化が、分け合うカーネルの外を読む (未報告)

- **何が**: `coreai-build` (3605.6.3) の h13g 向けのコンパイラは、1 つの関数の中で定数が同じ活性化のカーネル (sigmoid と silu の exp の段と逆数の段、gelu の tanh など。1 組 128 バイト) を 1 つにまとめる。活性化のタスクが読む係数の組の数は、テンソルの並びで決まる
  - `Linear` の出力のように、行をバッチとして置いたテンソル: 1 組 (128 バイト)
  - 行やヘッドをチャネルとして置いたテンソル: チャネルごとに 1 組。8 行なら 8 組 (1024 バイト)、チャネルが 16 以上なら 16 組 (2048 バイト)

  まとめたカーネルの大きさは、それを使う最後のタスクに合わせられる。最後が 1 組のタスクだとカーネルは 128 バイトになり、その前の 8 組や 16 組のタスクはカーネルの外を読む
  - LLIR での見え方: `anehlo.kload` が読む `memref.view` の型が `memref<128>` なのに、その中の `anehlo.coeff_dma_offset_size` が (0, 128)、(128, 128)、…、(896, 128) を指す。`tools/coreai/llirtasks -check X.mlir.bc` がそのタスクを挙げる
  - h13g だけで起きる。直す前の同じ部品を h14g〜h17g 向けに作ると 1 つもない (3 モデルの `model.part2`)
- **M1 で起きること** (macOS 27.2.1、Metal Toolchain 27.2.47): オフラインでは全部 Neural Engine に載る。端末でコンパイルした同じプログラムは動く
  - 関数が 1 つのプログラム: 仕上がるが、呼ぶと終わらない。カーネルのログ (`AppleH11ANEInterface`) に `SyncRequest timed out after 100000 ms` が出て、呼び出しは 100 秒後に "Request cancelled" で返る
  - 関数が 2 つ以上のプログラム: 仕上げのサービスが落ちる (17 節)
- **止まったあと**: Neural Engine のファームウェアは止まったままで、止める指示 (`CSNE_CMD_STOP`) も 2 秒で時間切れになる。ドライバは、Neural Engine を使うプロセスがいなくなってから 5 秒後に電源を落とし、強制的に再起動する (`Forcing firmware timeout path for clean reboot`)。それまでは、**ほかのプロセスの呼び出しも 100 秒ずつ待って失敗する**。プログラムを開いたまま呼び直すプロセスや、呼び続けるシステムのデーモン (夜に `mediaanalysisd` がそうだった) がいると、再起動が始まらない
- **根拠** (2026-11-27): h13g のプログラムの LLIR を読んだ結果と、M1 での結果が全部合う
  - M1 で止まったもの、仕上げで落ちたものには、どれにもこのタスクがある: Qwen3.5-0.8B の `model.part1` (関数 2 つでは仕上げで落ち、1 つでは止まる) と `model.part2` (どの部品にも 1 関数に 2〜11 個ある)、Gated DeltaNet 3 層の検体、層 1 つの検体 (3 個)、層 0 を演算 294、297、301 で切った検体 (3 個ずつ)
  - M1 で動いたもの約 20 本には、どれにもない: 層 0 を演算 263、277、288 で切った検体、デルタ則だけの検体 8 本 (チャンク 8 で 8〜128 行、チャンク 16 で 16〜64 行)、softplus だけの検体 2 本、MLP の silu だけの検体、mass プログラム、Qwen3-0.6B の `model.part1` の 2 つの関数。M1 で試していない検体 20 本あまり (MLP を別の形で書いたものなど) にもない
  - 層 0 では、288 演算 (MLP の最初の射影まで) と 294 演算 (その silu まで) の間で変わる。288 では分け合うカーネルが 1024 バイト (最後に使うのが 8 組の `silu(z)`)、294 では MLP の silu (1 組) が最後に加わって 128 バイトになる
  - 手元の prebuilt 1331 本を読んだ。h14g 369 本と h15g 421 本には 1 つもない。h13g 541 本のうち、M1 で動いているもの (Anima、FLUX.3 klein、FLUX.2 klein の M1 用の書き出し) には 1 つもなく、あったのは Qwen3.5 のチャット、13 節の FLUX.2 klein の書き出し、古い FLUX.2 klein v1 の `vae_s0` だけ
  - 最小の再現は 16 タスクのプログラム (620 KB): sigmoid を `(1, 8, 16, 64)` に掛け、別の `Linear` の出力に silu を掛けて、積を取る。sigmoid の 2 つの段 (タスク 6 と 7) が、128 バイトのカーネルから 1024 バイトを読む。M1 で prebuilt から開くと、仕上がったあと最初の呼び出しが 100 秒で打ち切られた (カーネルのログに `Request timeout detected … elapsed: 100005 ms`)。sigmoid を `(1, 8, 1024)` に掛ける形にすると 1 つもない
- **どこにあったか** (直す前の書き出し、h13g):

  | モデル | 1 関数のタスク | 分け合っていたもの |
  |---|---|---|
  | Qwen3.5-0.8B の Gated DeltaNet の層 | 層ごとに 3 個 | ヘッドごとの `silu(z)` と 16 幅のゲートの sigmoid (8 組) と、畳み込みのあとの silu、MLP の silu (1 組) |
  | Qwen3.5-0.8B の attention の層 | 層ごとに 2 個 | ヘッドごとの出力のゲートの sigmoid (8 組) と MLP の silu (1 組) |
  | LFM3-1.7B (`q8` の `model.part2`〜`model.part6`) | 1 個 | ヘッドごとのゲートの sigmoid (16 組と 8 組) と MLP の silu (1 組) |
  | Gemma 4 E2B (全部品) | 1〜6 個 | MLP の gelu (6144 チャネル、16 組) と、層ごとの入力のゲートの gelu (1 組か 8 組。層によって違う) |
  | Qwen3-0.6B、plamo-3-translate-0.8b | 0 | |
- **今**:
  - 書き出しの側で活性化の並びをそろえた ([design/apple-neural-engine.ja.md](design/apple-neural-engine.ja.md) の「活性」)。チャットのモデルのレシピの版は 2。直した書き出しの h13g のプログラム 202 本 (Qwen3.5-0.8B 24、Qwen3.5-2B 24、Qwen3.5-4B 26、LFM3-1.7B 17、LFM3-VL-2B 18、Gemma 4 E2B 18、Gemma 4 E4B 18、plamo-3-translate-2b 18、Qwen3-1.7B 39) のどれにもこのタスクがない
  - M1 で確かめた (2026-11-27、直した書き出しの h13g の prebuilt。どれも関数 2 つで、全部 Neural Engine の上で動いた): Qwen3.5-0.8B の `model.part1` (Gated DeltaNet 3 層。直す前は仕上げのサービスが落ちた) は 8.2 秒で仕上がり、`sr-q8` が 5.45 ms、`q128-last` が 14.6 ms。`model.part2` (attention の層と Gated DeltaNet 3 層。直す前は止まった) は 9.4 秒で仕上がり、4.42 ms と 12.7 ms。直した LFM3-1.7B の `model.part2` (`q8` と `q128-last`) は 17.5 秒で仕上がり、4.56 ms と 11.3 ms。直した Gemma 4 E2B の `model.part2` (`q8` と `q128-kv`) は 14.1 秒で仕上がり、4.33 ms と 11.8 ms。Gated DeltaNet の層 1 つの検体は、CPU との余弦が出力で 0.999997、状態で 1.000000
  - 1 回の呼び出しの時間は、直す前と変わらない (端末でコンパイルして測った。[design/apple-neural-engine.ja.md](design/apple-neural-engine.ja.md) の「活性」の表)。prebuilt から仕上げた `model.part2` と端末でコンパイルしたものを続けて測ると、8 行は 3.96 ms どうし、128 行は 12.05 ms と 11.56 ms だった
  - `hwnpu-dev prebuild` (`hwnpu-dev models package` も) は、作ったプログラムの LLIR を読み、このタスクがあれば失敗にする。そのプログラムは置かれず、端末に任せる決まり (レシピの `prebuild`) を書くよう言われる。前に作ってある prebuilt も同じに読み、あれば作り直す
  - 置いてしまった場合の備え: prebuilt から仕上げたプログラムの最初の呼び出しが失敗したら、その機械はその束を使うのをやめる (記録の `stopped`、[prebuilt-reuse](prebuilt-reuse.md) 5 節)。その実行はエラーで終わり、次の実行が端末でコンパイルする。端末の印 `ok` は、開けたときでなく最初の呼び出しが返ったときに書く
- **prebuilt を確かめるとき**: まずオフラインで `llirtasks -check` を通す (数秒。Neural Engine は要らない)。実機では、署名の識別子を変えた写しで動かす (`codesign -f -s - -i NEW`。aned のコンパイル結果は識別子ごとで、Go の実行ファイルはどれも `a.out`)。1 回の試行は 1 つのプロセスで行い、失敗したら Neural Engine が戻るまで待つ (端末でコンパイルしたプログラムを新しいプロセスで動かして見る)。ほかのセッションが Neural Engine を使っていないことを `ps` でも見る
- **直ったら**: 何もしなくてよい (そろえた書き方は値を変えない)
- 確かめ方: `HWNPU_TEST_QWEN35_MLP_PROBE=share.mlpackage HWNPU_TEST_QWEN35_MLP_PROBE_SPEC=1024x256:share go test ./internal/llm/qwen35 -run TestWriteMLPProbe` で最小の再現を作り、`aicheck lower`、`coreai-build compile … --architecture h13g`、`llirtasks -check` の順にかける。2 個と出れば残っている。M1 で `aicheck run -scale 0.05 share.aimodel` (prebuilt を横に置いて) が 100 秒後に "the prebuilt program failed" で終われば、端末の側も直っていない

## 17. 関数が 2 つ以上の Gated DeltaNet のプログラムで仕上げのサービスが落ちる (未報告)

- **何が**: Qwen3.5-0.8B の `model.part1` (層 0〜2 と層 3 の attention の前段。関数は `sr-q8` と `q128-last`) の h13g の prebuilt を M1 で開くと、`ANELargeModelCompilerService` が 2 つめの関数 (`sr-q8`) の変換の途中で落ちる (1155 タスク中 1013 個目、4 回とも同じ所)。プロンプトの関数を 64 行にしても、デルタ則のチャンクを 16 行にしても落ちる。層 3〜7 を 8 つの関数にした検体も落ちた。関数を `sr-q8` 1 つにした `model.part1` は落ちずに仕上がり、16 節のとおり呼び出しが終わらない
- **原因は 16 節と同じ**: その関数で、カーネルの外を読むタスクは 103、294、295、448、639、640、793、984、985 個目の 9 つで、落ちるのは最後の 2 つのすぐあとである。落ち方はヒープの破壊 (`ZinKernelBuilder::BuildKernelFromLLIRData` の中の malloc の検査など) で、カーネルの外を読むことと合う。直した書き出しの `model.part1` は、同じ関数 2 つのまま 8.2 秒で仕上がった (2026-11-27)
- Core AI は `failedToSpecialize` を返し、hwnpu はその部品を端末でコンパイルする (今までどおり)。困るのは 15 節: 落ちるたびに、その機械のすべての prebuilt の仕上げが遅くなる
- **今**: 16 節と同じ
- **直ったら**: 何もしなくてよい

## 18. 定数を左に置いた行列積が仕上がらない (未報告)

- **何が**: `paged` の mass プログラムは、行の合計を「全部 1 の定数 × exp(scores)」の行列積で取っていた (定数が左)。`coreai-build` はこれを全部 Neural Engine に載せるが、M1 の仕上げは `Kernel evaluation failed: … mps.constant_… : memref<1x1x1x2x32x1xf16>` で失敗する (`failedToSpecialize`)。同じ「1 との積」でも、定数を右に置く attention のプログラムは仕上がる。16 節のタスクはこのプログラムにない (別の問題)
- **今**: 定数を右に置く (転置した exp に 1 の列を掛ける。`internal/llm/paged/mass.go`)。仕上がり、全部 Neural Engine で動き、CPU との余弦は 1.000000、速さは同じ (8 行で 1.23 ms どうし、128 行で 3.43 → 3.25 ms)
- **直ったら**: 何もしなくてよい
- 確かめ方: 直す前の mass プログラム (8 クエリ × 2 つの kv ヘッド、チャンク 8192) を h13g で作り、M1 で開く

## 確かめ方 (OS や Metal Toolchain を上げたとき)

- 1 節: 画像のモデルを prebuilt で開き、`log show --predicate 'process == "launchd" AND eventMessage CONTAINS "ANELargeModel"'` に "deferred by 10 seconds" が出るか
- 2 節: 2 節の再現を `coreai-build` で h13g 向けに作り、`mps.fullyPlacedOnANE` があるか
- 3 節: FLUX.2 klein 1024² の初回に "vae_s0 … the prebuilt program failed" が出るか
- 6、7 節: 6 節の確かめ方を、落ちる 9 種類と h18g で
- 9 節: [apple-coreai](design/apple-coreai.md) の表と 9 節の表を測り直す
- 10、11、12、14 節: 各節の確かめ方を ([apple-feedback/repro.md](apple-feedback/repro.md) の道具)
- 16 節 (17、13 節も): 16 節の最小の再現を h13g 向けに作り、`llirtasks -check` が 0 個と言うか。0 個なら、活性化の並びをそろえる書き方 (`blocks.WideAct` など) と 13 節の M1 用の書き出しは要らなくなっている

確かめた日と OS、Metal Toolchain の版を、この文書の各節に書き足す。
