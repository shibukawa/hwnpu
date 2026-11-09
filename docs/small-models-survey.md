# 1B 以下のモデルの候補 (音程、歌声、音楽生成、効果音、姿勢) — 2026-11-09 の調査

チャット以外の小さなモデルを足すための下調べ。移植はまだしていない。ライセンスは docs/model-table.md 13 節の方針 (新しく足すものは Apache-2.0 か MIT から) に、今回は重みの CC-BY 4.0 まで (帰属の表示が要る、商用可) を加えた範囲で選ぶ。非商用 (CC-BY-NC) と Stability AI Community License は外す。

数値とライセンスは各モデルカードと GitHub の README から。「未確認」は一次情報で確かめられなかったもの。

## 1. 一覧

| 用途 | 候補 | 大きさ | ライセンス (コード / 重み) | 構造 | hwnpu の既存の部品 | 判断 |
|---|---|---|---|---|---|---|
| 音声 → 音符 (MIDI) | **Basic Pitch** (Spotify, `spotify/basic-pitch`) | 約 20 MB (数十万パラメータ級、公式の数は未確認) | Apache-2.0 / Apache-2.0 | CQT の上の小さな CNN。出力は音の始まり、音符、ピッチの輪郭の 3 枚。22050 Hz、長さは窓で区切る。複数音、楽器を問わない | 音声の前処理 (`mel_golden.py` の照合の形)、hwnpu-transcribe の窓の切り方 | ◎ 最初の候補。CQT をホストか Neural Engine の畳み込みで作る、音符への組み立て (ホスト) を Go に移す |
| 音声 → 音符 (多楽器) | MT3 (Google Magenta) | T5-small 級 (約 60M、未確認) | Apache-2.0 / Apache-2.0 (未確認) | スペクトログラム → T5 のエンコーダ・デコーダ → MIDI のイベントのトークン | Whisper のエンコーダ・デコーダ (`internal/transcribe`) | ○ Basic Pitch の後。楽器ごとに分けたいとき |
| 音声 → ピッチ曲線 (F0) | **SwiftF0** (`lars76/swift-f0`) | 14,386 パラメータ | MIT (重みの別のライセンスの記載なし) | スペクトログラムの上の CNN。16 kHz、16 ms ごとに F0、有声の確からしさ、音量 | 同上 | ◎ 一番小さい。Neural Engine に載せるより CPU の方が速い可能性が高い (11 節の「小さなモデルを詰め物で ANE に」は CPU より遅かった)。測ってから決める |
| 〃 | FCPE (`CNChTu/FCPE`、torchfcpe) | 未確認 (小さい) | MIT / MIT | mel の上の Conformer 系 | Whisper のエンコーダの部品 | ○ 歌声の変換 (RVC、so-vits) で使われる。SwiftF0 で足りなければ |
| 〃 | RMVPE (`Dream-High/RMVPE`、RVC が配る重み) | 約 90M (ONNX の書き出しの値) | Apache-2.0 (元) / MIT (RVC の重み) | U-Net + 双方向 GRU。伴奏があっても声の F0 を取る | — (GRU は時刻の繰り返しで、Neural Engine に向かない) | △ 伴奏の上の歌の F0 が要るときだけ。GRU をホストに回す形を考える |
| 〃 | CREPE (torchcrepe) | full 約 2200 万、tiny もある | MIT / MIT (再配布者の記載、元のファイルは未確認) | 1024 サンプルの窓の CNN。1 秒 282 GFLOP (full) | — | △ SwiftF0 より重く、精度の差が小さい |
| 文 → 歌声 | **SoulX-Singer** (`Soul-AILab/SoulX-Singer`) | 公式の数なし。重みのファイル約 2.8 GB (fp32 なら約 0.7B、未確認) | Apache-2.0 / Apache-2.0 | 歌詞の音素と音符 (または F0) → 内容のエンコーダ、長さの調整 → フローマッチングの DiT で mel → ボコーダ。参照の声 (mel) で声質を真似る | Anima の DiT とオイラーのステップ、LFM3-Audio の出力側 (hwnpu-tts) | ○ ただし言語は中国語 (普通話、広東語) と英語で、**日本語の歌詞は受けない**。前処理 (分離、書き起こし) は別のリポジトリ。大きさとボコーダを確かめてから |
| 〃 | DiffSinger (OpenVPI) | 声のデータごと | Apache-2.0 (コード) / 声のデータは配布者ごと | 音響モデル (拡散) + ボコーダ (NSF-HiFiGAN) | — | △ 日本語の声のデータはあるが、ライセンスが声ごとにばらばら。1 つずつ確かめる必要がある |
| 音楽生成 | **Magenta RealTime 2 の small** (`google/magenta-realtime-2`) | 230M (base は 2.4B) | Apache-2.0 / **CC-BY 4.0** (+ 使い方の注意: 他者の権利を侵さない) | 3 つの部品: SpectroStream (48 kHz ステレオの音声のコーデック、25 Hz のトークン)、MusicCoCa (文と 16 kHz の音声の埋め込みで曲風を指定)、デコーダだけの Transformer (1 フレームずつ自己回帰)。入力は文、音声の例、MIDI (128 音の状態)。約 200 ms の遅れで流し続ける。受容野は約 20 秒 | Qwen3 / LFM3 の因果 LM と KV キャッシュ、LFM3-Audio のコーデックの復号 (hwnpu-tts) | ○ 230M は小さいが、部品が 3 つ (コーデックの符号化と復号、埋め込み、LM)。1 秒に 25 ステップ以上回す必要があり、M1 の速さを先に見積もる |
| 〃 | ACE-Step 1.5 | 標準 3.5B (LM 0.6B〜4B + DiT) | MIT / MIT | LM + 拡散の DiT + VAE | Anima の DiT | × 1B を超える (LM の 0.6B だけでは曲にならない) |
| 〃 | MusicGen-small | 300M | MIT / CC-BY-NC 4.0 | — | — | × 非商用 |
| 効果音 | Stable Audio Open Small / Stable Audio 3 Small SFX | 341M / 0.46〜0.6B | Stability AI Community License (年商 100 万ドル未満なら商用可) | 拡散の DiT + VAE | Anima | × 今回の範囲 (CC-BY 4.0 まで) の外 |
| 〃 | Woosh (Sony) | 未確認 | Apache-2.0 / CC-BY-NC 4.0 | — | — | × 非商用 |
| 〃 | **MOSS-SoundEffect v2.0** (`OpenMOSS-Team/MOSS-SoundEffect-v2.0`) | DiT 1.3B + 文の符号器 Qwen3-1.7B + DAC VAE | Apache-2.0 / Apache-2.0 | Wan 2.1 の DiT + flow matching。48 kHz モノラル、最長 30 秒。英語と中国語 | Anima の DiT、Qwen3 | ◎ 入れた (2026-11-10 に探し直して見つけた) |
| 〃 | MMAudio (`hkchengrex/MMAudio`) | small 約 157M | MIT / MIT。ただし文の特徴の Apple DFN5B-CLIP は研究専用 | flow matching の DiT + mel の VAE + BigVGAN | — | × 文の符号器の許諾が研究専用 |
| 〃 | MiDashengLM-Gen (Xiaomi) | 約 3B (ミラーの表示) | Apache-2.0 | Qwen3-1.7B + flow matching + Vocos、16 kHz。声、音楽、効果音を混ぜた場面 | — | △ 大きく、効果音専用でない |
| 人の姿勢 (骨格) | **RTMO** (OpenMMLab MMPose、`Xenova/RTMO-s/m/l`) | 8.9M / 12.4M / 19.1M | Apache-2.0 / Apache-2.0 (OpenMMLab の配布、変換者の記載) | 1 段 (人の検出なしで、箱と 17 点を一度に)。CSPNeXt の CNN、640 × 640 | LFM3-VL の画像の前処理 (`resize_golden.py`) | ◎ 検出器が要らず、CNN で Neural Engine に向く。後処理 (NMS、点の復号) はホスト |
| 〃 | ViTPose (`usyd-community/vitpose-base-simple` ほか) | base 85.9M (small から huge まで) | Apache-2.0 / Apache-2.0 | ViT + 簡単なヘッド。**人の箱が要る** (例は RT-DETR で検出してから) | LFM3-VL の視覚の塔 (ViT) | ○ transformers に `VitPoseForPoseEstimation` があり照合しやすい。ただし検出器 (RT-DETR など) も移植が要る |
| 〃 | DWPose / RTMW (全身 133 点、顔と手も) | 未確認 (t / s / m / l) | Apache-2.0 (変換者の記載) | 2 段 (検出器 + RTMPose 系) | — | ○ 手や顔の点が要るとき。RTMO の後 |

## 2. 進み具合 (2026-11-09)

| モデル | 状態 | コマンド | 記録 |
|---|---|---|---|
| SwiftF0 | ◎ 入れた。全部 NPU、41 秒を 0.24 秒、ONNX と有声 / 無声が全フレーム一致 | `hwnpu-pitch` | design.md 16.128 |
| RTMO-s | ◎ 入れた。全部 NPU (尾部はホスト)、1 枚約 0.1 秒 | `hwnpu-pose` | design.md 16.129 |
| Basic Pitch | ○ 入れた。参照と同じ音符、35 倍速。畳み込み 1 つと ReLU が CPU | `hwnpu-notes` | design.md 16.129 |
| Magenta RealTime 2 small | ◎ 入れた。言語モデルと SpectroStream の復号と符号化、MusicCoCa の文と音楽の符号器がプログラム (NPU)。文と録音からの曲調 (参照と同じ符号)、混ぜ合わせ、MIDI の音符、打楽器、録音の続き、ストリーミング | `hwnpu-music` | design.md 16.130、16.131 |
| MOSS-SoundEffect v2.0 | ◎ 入れた。文の符号器、DiT、DAC の復号がすべて NPU。16 GB 以上は文の符号器が fp16、8 GB は lut8 (5 層だけ fp16)。4 ステップで参照と同じ雑音からの波形のコサイン 0.99 | `hwnpu-sfx` | design.md 16.136 |
| SoulX-Singer | 未着手 (日本語の歌詞は受けない) | — | — |

ONNX だけで配られるモデルは `internal/onnx` (読み込み、ホストの評価、静的な部分の MIL への下ろし) で入れられる。TFLite で配られるモデルは `internal/tflite` (FlatBuffers を読んで `internal/onnx` のグラフにする) で入れられる。

## 3. 入れる順の案 (調査時)

1. **RTMO-s / m (姿勢)**: 1 段で検出器が要らず、CNN だけ、20M 以下。画像の前処理は LFM3-VL のものを使える。新しいコマンド (仮に `hwnpu-pose`) と、点を JSON で返す API
2. **Basic Pitch (音声 → MIDI)**: CNN で小さく、Apache-2.0。前処理の CQT と、音符への組み立てが仕事の大半。hwnpu-transcribe の隣に `-midi` か新しいコマンド
3. **SwiftF0 (F0)**: 14K パラメータ。まず CPU (Go の素朴な実装) と Neural Engine の両方で測る。Basic Pitch と同じ前処理の照合の道具を使える
4. **Magenta RealTime 2 small (音楽生成)**: 部品が 3 つで一番大きな仕事。SpectroStream の符号化と復号から照合する
5. **SoulX-Singer (歌声)**: 日本語を受けないので後回し。日本語の歌詞が要るなら、DiffSinger の日本語の声のデータのライセンスを 1 つずつ確かめる
6. 効果音: 範囲の中に候補がない。保留

## 4. 移植の前に決めること

- **コマンドと API の形**: 今のコマンドは用途ごと (hwnpu-transcribe、hwnpu-tts、hwnpu-image)。姿勢、MIDI、F0、音楽は新しいコマンドにするか、既存のもの (hwnpu-transcribe に `-midi` と `-f0`) に足すか。models.yaml の `kind` も増える (今は chat、embedding、transcribe、speech、decision)
- **CC-BY 4.0 の帰属の表示**: Magenta RT 2 を配る (docs/prebuilt.md) なら、帰属の表示の置き場所を決める。models.yaml に `license` を書く (13 節の方針)
- **小さすぎるモデル**: SwiftF0 のような数万パラメータのモデルは、Neural Engine の呼び出しの手間 (1 回数 ms) の方が計算より大きい。CPU の方が電力も小さければ、NPU に載せない (方針は「NPU だけで動かして電力を下げる」なので、測ってから)
- **確かめていないこと**: Basic Pitch、MT3、FCPE、DWPose のパラメータ数、SoulX-Singer の部品ごとの大きさとボコーダ、Magenta RT 2 の SpectroStream と MusicCoCa の大きさ、RTMO の重みの元の配布のライセンスの本文
