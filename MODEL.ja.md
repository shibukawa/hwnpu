# おすすめのモデル

hwnpu が動かすモデルを、用途と機械のメモリの量で選べるようにまとめました。数値は 2026-11-06 の main のものです。Apple のチャットのモデルは、Core AI (macOS 27) で 2026-11-15 に、画像のモデルは 2026-11-16 に M1 で測り直した値です。各 NPU が扱うデータ形式と、モデルの変換の仕方は [MODEL_ARCHITECTURE.ja.md](MODEL_ARCHITECTURE.ja.md) にあります。

English: [MODEL.md](MODEL.md)

## 用途とメモリで選ぶ

列は機械に積んだメモリの量です。OS とふだんのアプリの分を引いて、モデルに使える量の目安を 2 行目に書きました。NPU のバッファは固定されたメモリで、スワップに逃がせません。目安を超えると、ほかのアプリがスワップに追い出されます。

hwnpu-chat と hwnpu-chat agent は、この表から自分でモデルを選びます。用途の欄の名前 (`-m translate`、`-m chat`) で、機械のメモリの列のモデルを動かし、ダウンロードもそのモデルの分だけです。ファミリーの名前でも同じように選びます (`-m plamo3translate`、`-m gemma4`)。Docker のイメージのようにタグを付けると、メモリによらずそのモデルです。タグはモデルの大きさ (`-m plamo3translate:2b`、`-m chat:e4b`) か、表の列 (`-m chat:24g`、`-m plamo3translate:8g`。そのメモリの機械で動かすもの) です。この機械でどれが選ばれるかは `hwnpu-chat list` に出ます。`HWNPU_MEMORY=24` で 24 GB の機械と同じに選びます。hwnpu-audio transcribe (`-m transcribe`)、hwnpu-audio tts (`-m speech`)、hwnpu-chat systemone (`-m decision`) も同じように選びます。表は [models.yaml](models.yaml) (`categories` と、各モデルの `machine`) から来ていて、hwnpu に組み込まれています。自分のファイルで、モデルを足す、列のモデルを替える、あるモデルを別のモデルに差し替える (エイリアス) ができます。書くのは差分だけです ([docs/model-table.md](docs/model-table.md))。

<table>
<thead>
<tr>
<th rowspan="2">用途</th>
<th>8 GB</th>
<th>16 GB</th>
<th>24 GB</th>
<th>32 GB</th>
<th>48 GB</th>
<th>64 GB 以上</th>
</tr>
<tr>
<th>約 3 GB</th>
<th>約 9 GB</th>
<th>約 15 GB</th>
<th>約 22 GB</th>
<th>約 35 GB</th>
<th>約 50 GB</th>
</tr>
</thead>
<tbody>
<tr>
<th>チャット、コーディング<br><sub>hwnpu-chat、hwnpu-chat agent <code>-m chat</code></sub></th>
<td><b>LFM3-1.7B</b> (Apple 1.7 GB、AMD 2.2 GB)<br>速く、エージェント向け<br><b>Gemma 4 E2B</b> (AMD 2.4 GB、Apple 2.3 GB)<br>層ごとの埋め込みを含めて 5B の、ひと回り大きなモデル<br><b>Qwen3.5-2B</b> (<code>-m qwen3.5</code>、2.2〜2.4 GB)<br>コードも書く<br><b>Qwen3-1.7B</b> (<code>-m qwen3</code>、2.1 GB。Apple だけ)<br>プログラム 1 本で、長いプロンプトを読むのは遅い (<code>--prefill on</code> で速く、+0.3 GB)<br><b>Qwen3-0.6B</b> (<code>-m qwen3:0.6b</code>、M1 で 1.1 GB、M2 以降 1.6 GB、ダウンロード 1.5 GB)<br>一番小さなダウンロード。事実に弱い</td>
<td><b>Gemma 4 E2B</b> (AMD 2.4 GB、Apple 2.3 GB)<br>長い文脈 (131072 で +0.8 GB) も<br><b>Gemma 4 E4B</b> (AMD 4.3 GB、Apple 4.0 GB)<br>短い会話なら<br><b>Qwen3-1.7B</b> (<code>-m qwen3</code>、2.3 GB)<br>プロンプトを読むプログラムを別に持ち、読むのが 5 倍速い</td>
<td colspan="2"><b>Gemma 4 E4B</b> (AMD 4.3 GB、Apple 4.0 GB)<br>長い文脈 (131072 で +2.1 GB) も</td>
<td colspan="2"><b>Gemma 4 E4B</b> (AMD 4.3 GB、Apple 4.0 GB)<br><b>Qwen3.6-35B-A3B</b> (<code>-m qwen3.6</code>、AMD のみ、<code>-m chat</code> では選ばない。全部の専門家で 26.2 GB)<br>専門家の混合、35B のうち動くのは 3B。16 GB から、置く専門家を減らして (8.1 GB、遅くなる)</td>
</tr>
<tr>
<th>画像を読む<br><sub>hwnpu-chat <code>-m vision</code> (<code>image_url</code>、<code>/image</code>)</sub></th>
<td><b>LFM3-VL-2B</b> (2.6〜2.7 GB)</td>
<td colspan="5"><b>Gemma 4 E2B 画像と音声つき</b> (3.4 GB)<br><b>LFM3-VL-2B</b> (2.6〜2.7 GB)</td>
</tr>
<tr>
<th>文字起こし<br><sub>hwnpu-audio transcribe</sub></th>
<td colspan="6"><b>Whisper large-v3-turbo</b> (約 1.9 GB)<br>99 言語、タイムスタンプつき、30 秒を 0.6 秒で読む</td>
</tr>
<tr>
<th>音高 (F0)<br><sub>hwnpu-audio pitch</sub></th>
<td colspan="6"><b>SwiftF0</b> (約 0.06 GB)<br>声や楽器 1 つの音高を 16 ms ごとに、有声の確からしさつき。41 秒を 0.24 秒</td>
</tr>
<tr>
<th>音符 (MIDI)<br><sub>hwnpu-audio notes</sub></th>
<td colspan="6"><b>Basic Pitch</b> (約 0.07 GB)<br>どの楽器でも、和音も、ピッチベンドつきで MIDI に。実時間の 35 倍</td>
</tr>
<tr>
<th>人の姿勢<br><sub>hwnpu-image pose</sub></th>
<td colspan="6"><b>RTMO-s</b> (約 0.23 GB)<br>画像の人と 17 の関節点を 1 回で (人の検出器は不要)。1 枚約 0.1 秒</td>
</tr>
<tr>
<th>物の検出<br><sub>hwnpu-image detect</sub></th>
<td><b>RTMDet-s</b> (約 0.1 GB)<br>物の枠と COCO の 80 種類のラベル。M1 の Neural Engine で 1 枚 9 ms</td>
<td colspan="5"><b>RTMDet-m</b> (約 0.2 GB)<br>より正確 (box AP 49.1、s は 44.6)。1 枚 18 ms</td>
</tr>
<tr>
<th>文字 (OCR)<br><sub>hwnpu-image ocr</sub></th>
<td colspan="6"><b>PP-OCRv6 small</b> (約 0.45 GB)<br>文字の行と枠と中身、50 言語 (日本語、中国語、英語など)。M1 で論文 1 ページ 98 行が 0.5 秒</td>
</tr>
<tr>
<th>音楽生成<br><sub>hwnpu-audio generate</sub></th>
<td colspan="6"><b>Magenta RealTime 2 small</b> (約 1.0 GB)<br>文や録音で指定した曲調の 48 kHz ステレオの音楽。MIDI の音符を弾く、録音の続きを作る、できたそばから流すこともできる。実時間の約 1.4 倍の速さ。重みは CC-BY 4.0 (使うときは「Magenta RealTime 2 by Google DeepMind」と表示)</td>
</tr>
<tr>
<th>効果音<br><sub>hwnpu-audio generate -m sfx</sub></th>
<td><b>MOSS-SoundEffect v2.0</b> (約 2.2 GB。文の符号器は lut8、5 層だけ fp16)</td>
<td colspan="5"><b>MOSS-SoundEffect v2.0</b> (約 2.3 GB。文の符号器は fp16)<br>英語か中国語の説明から最長 30 秒の 48 kHz モノラルの効果音。M1 でカードの 100 ステップなら 10 秒に約 4 分 (<code>-steps 50</code> で半分)。Apache-2.0</td>
</tr>
<tr>
<th>音声を読む (音声についての質問)<br><sub>hwnpu-chat <code>-m audio</code> (<code>input_audio</code>、<code>/audio</code>)</sub></th>
<td><b>LFM3-Audio-2B</b> (2.6 GB)</td>
<td colspan="5"><b>Gemma 4 E2B 画像と音声つき</b> (3.4 GB)<br><b>LFM3-Omni-2B</b> (3.0〜3.1 GB、画像も)</td>
</tr>
<tr>
<th>翻訳<br><sub>hwnpu-chat <code>-m translate</code></sub></th>
<td><b>PLaMo 3 Translate 0.8B</b> (1.0 GB)<br>2B の 3 倍速、訳は 2B の言い換え程度の違い</td>
<td colspan="5"><b>PLaMo 3 Translate 8B</b> (Apple で 8.2 GB、AMD で約 6.5 GB)<br>訳語が 2B より的確。M3 で 10〜11 トークン/秒 (読み込み 4 秒)、M2 で 9〜10 (起動のたびに 35〜41 秒)<br>M1 では <b>PLaMo 3 Translate 2B</b> (3.2 GB) を選ぶ: 8B は 0.8 トークン/秒 (<code>-m plamo3translate:8b</code> なら 8B)</td>
</tr>
<tr>
<th>画像生成<br><sub>hwnpu-image</sub></th>
<td colspan="6"><b>FLUX.2 klein 4B</b> 1024 × 1024 (AMD 3.9 GB、Apple 4.9 GB。<code>--size 512</code> で AMD 1.9 GB、Apple 3.1 GB)<br>主力: Apache-2.0、4 ステップで M1 で 41 秒 (512 × 512 は 11 秒)、AMD で 29 秒。AMD では参照画像の編集と合成も (<code>--image</code>)<br>オプトイン (<code>HWNPU_MODEL_ALLOWLIST=+nsfw</code> で、<code>-m</code> で名前を指定): <b>Anima-Turbo</b> (Apple 4.8 GB、M1 で 48 秒。Danbooru のタグ)、<b>Anima-Aesthetic</b> (<code>-m anima</code>、25 ステップと CFG で約 4.5 倍の時間)</td>
</tr>
<tr>
<th>読み上げ<br><sub>hwnpu-audio tts</sub></th>
<td colspan="6"><b>LFM3-Audio-2B</b> (約 2.4 GB)</td>
</tr>
<tr>
<th>文の埋め込み、検索<br><sub>hwnpu-chat <code>-m embedding</code> (<code>/v1/embeddings</code>)</sub></th>
<td colspan="6"><b>PLaMo-Embedding-1B</b> (1.2〜1.3 GB)</td>
</tr>
<tr>
<th>状態の判定<br><sub>hwnpu-chat systemone</sub></th>
<td colspan="6"><b>d1-2B</b> (約 2.4 GB、画像も)<br><b>d1-Omni-2B</b> (約 2.8 GB、画像と音声)</td>
</tr>
<tr>
<th>いくつも同時に<br><sub>hwnpu-chat <code>serve -m A,B,…</code></sub></th>
<td>LFM3-1.7B + PLaMo 3 Translate 0.8B (Apple 2.7 GB、AMD 3.2 GB)</td>
<td>Gemma 4 E2B + PLaMo-Embedding-1B (3.5〜3.6 GB)<br>+ PLaMo 3 Translate 2B (AMD 6.3 GB、Apple 6.7 GB)</td>
<td>Gemma 4 E4B + PLaMo-Embedding-1B + PLaMo 3 Translate 2B (AMD 8.0 GB、Apple 8.4 GB)</td>
<td colspan="3">その上に Gemma 4 E2B 画像と音声つき、別のプロセスで FLUX.2 klein 4B (約 16〜17 GB)</td>
</tr>
</tbody>
</table>

- 画像生成 (`-m image`、hwnpu-image generate の既定) は、8 GB から FLUX.2 klein 4B です。頼まなくても NSFW の絵を描くことがある Anima と Chroma は、`HWNPU_MODEL_ALLOWLIST=+nsfw` でオプトインして名前で指定したとき (`-m anima-turbo`、`-m chroma`) だけ動きます。オプトインしても用途からは選びません。ライセンスでモデルを絞るときも同じ変数を使います ([docs/model-provenance.md](docs/model-provenance.md))
- 括弧の中は、起動して短い会話をしたときの想定のメモリです (次の節)。長い会話では KV キャッシュの分が増えます
- 32 GB を超えても、今は大きなモデルがありません。メモリの多い機械では、上の表の組み合わせを同時に動かすか、長い文脈に使います
- 8 GB の機械では、メモリの小さい方の動かし方になります (LFM3-1.7B は短い会話用の 1 本なし、Qwen3-1.7B はプロンプト用のプログラムなし)。メモリの多い機械でほかのプログラムと一緒に動かすときは、`hwnpu memory LFM3-1.7B 8` (または `hwnpu-chat --memory 8`) で、どのコマンドでもそのモデルは 8 GB の形で動きます ([docs/cache-sharing-design.md](docs/cache-sharing-design.md) 7.8)
- 小さな機械では、`--context` で文脈を短くすると KV キャッシュの分が減ります。hwnpu-chat は KV キャッシュが物理メモリの 4 分の 1 を超えるとき、文脈を自分で縮めます

## モデルごとのメモリ

「起動時」は、NPU のバッファ (重み、作業領域、既定の文脈の KV キャッシュ) とプロセス自身 (トークナイザー、ホストのバッファ。約 0.4 GB) の合計です。NPU のバッファはプロセスの RSS には出ないので、タスクマネージャーや `top` に出る値より大きくなります。

| モデル | 用途 | 起動時 | 会話で増える分 (KV キャッシュ) | 根拠 |
|---|---|---|---|---|
| LFM3-1.7B | チャット、コーディング | AMD 2.2 GB。Apple は 1.7 GB (Neural Engine に 1.36 GiB とプロセスの 0.23 GiB)。16 GB からの M2 以降は 3.1 GB (短い会話用の 1 本が重みをもう 1 組持つ: 2.64 GiB と 0.21 GiB) | 12 KiB/トークン (262144 で 3 GiB) | 実測 (AMD と M1。M2 以降の形は M1 に `--tier 2048` で読み込ませて) |
| Gemma 4 E2B | チャット、コーディング | AMD 2.4 GB (2026-11-17)、Apple 2.3 GB (M1 の Core AI: Neural Engine に 1.66 GiB。プロセスの分は Core AI では測っていない) | 6 KiB/トークン (131072 で 0.75 GiB) | 実測 |
| Gemma 4 E2B 画像と音声つき | 画像、音声 (1 回に 1 枚と 30 秒まで) | 3.4 GB (画像か音声が来るまでは 2.4 GB) | 6 KiB/トークン | 実測 |
| Gemma 4 E4B | チャット、コーディング | AMD 4.3 GB (2026-11-17)、Apple 4.0 GB | 16 KiB/トークン (131072 で 2 GiB) | 実測 |
| LFM3-VL-2B | 画像 (512 × 512) | AMD 2.6 GB、Apple 2.7 GB (M1: 画像を読んで Neural Engine に 2.24 GiB とプロセスの 0.25 GiB。文だけなら 2.6 GB。画像の符号器は初めから載る) | 12 KiB/トークン (128000 で 1.46 GiB) | 実測 (AMD と M1) |
| SwiftF0 | 音高 (F0) | 約 0.06 GB (M1: プロセスの footprint 58 MB。重み 14K) | なし (30 秒ずつ) | M1 で実測 |
| Basic Pitch | 音符 (MIDI) | 約 0.07 GB (M1: プロセスの footprint 75 MB。重み 36K) | なし (2 秒の窓ずつ) | M1 で実測 |
| RTMO-s | 人の姿勢 | 約 0.23 GB (M1: プロセスの footprint 232 MB。重み 9.9M を fp16) | なし (1 枚ずつ) | M1 で実測 |
| RTMDet-tiny / s / m | 物の検出 | 約 0.07 / 0.1 / 0.2 GB (M1: footprint 56 / 73 / 141 MB、Neural Engine に 11 / 21 / 56 MB) | なし (1 枚ずつ) | M1 で実測 |
| PP-OCRv6 small | 文字 (OCR) | 約 0.45 GB (M1: footprint 最大 425 MB。認識器の 7 つの幅のプログラムを読み込んだとき) | なし (1 枚ずつ) | M1 で実測 |
| Magenta RealTime 2 small | 音楽生成 | 約 1.0 GB (M1 の footprint: 文の曲調 0.97 GB、録音の曲調 0.55 GB、録音の続き 0.96 GB、符号を `-codes` で渡せば 0.42 GB) | なし (直前の 41 フレームの窓) | M1 で実測 |
| MOSS-SoundEffect v2.0 | 効果音 | 約 2.3 GB (M1 の footprint の最大は文の符号器が fp16 で 2.34 GB、lut8 で 2.17 GB。文の符号器のプログラム、fp16 で 2.8 GB か lut8 で 1.7 GB を先に動かして手放し、次に DiT の 1.4 GB (lut8) と復号の 0.2 GB) | なし (30 秒の潜在をまとめて) | M1 で実測 |
| Whisper large-v3-turbo | 文字起こし (長さの上限なし、30 秒ずつ) | 約 1.9 GB (M3: Neural Engine に 0.78 GiB とプロセスの 1.1 GB) | なし (30 秒ごとに 448 トークンまで) | M3 で実測 |
| LFM3-Audio-2B | 音声 (1 本 30 秒まで) | 2.6 GB (M1: 音声を読んで Neural Engine に 2.18 GiB とプロセスの 0.21 GiB。文だけなら 2.4 GB) | 12 KiB/トークン | 実測 (AMD と M1) |
| LFM3-Omni-2B | 画像と音声 | AMD 3.0 GB (画像か音声が来るまでは 2.2 GB)、Apple 3.1 GB (M1: 画像と音声を読んで Neural Engine に 2.63 GiB とプロセスの 0.25 GiB。2 つの符号器は初めから載り、文だけでも 2.8 GB) | 12 KiB/トークン | 実測 (AMD と M1) |
| PLaMo 3 Translate 2B | 翻訳 | AMD 2.7 GB、Apple 3.2 GB (M1: Neural Engine に 2.74 GiB とプロセスの 0.24 GiB) | 7 KiB/トークン (262144 で 1.75 GiB) | 実測 (AMD と M1) |
| PLaMo 3 Translate 8B | 翻訳 | Apple で約 8.2 GB (M1: 4 ビットの表で Neural Engine に 3.81 GiB とプロセスの最大 4.1 GB)。AMD で約 6.5 GB (q4、重みからの見積もり) | 9 KiB/トークン (262144 で 2.25 GiB) | M1、M2、M3 で実測 (Apple)、AMD は未実行 |
| PLaMo 3 Translate 0.8B | 翻訳 | 1.0 GB (M1: Neural Engine に 0.82 GiB とプロセスの 0.12 GiB) | 1 KiB/トークン (262144 で 0.28 GiB) | 実測 (AMD と M1) |
| PLaMo-Embedding-1B | 文の埋め込み | AMD 1.3 GB、Apple 1.2 GB (M1: Neural Engine に 1.05 GiB とプロセスの 0.03 GiB) | なし | AMD は重みから見込み、Apple は M1 で実測 |
| MoGe-2 ViT-S / B / L (AMD) | メートルの深度、点群、法線 (hwnpu-image depth) | 1.9 GB (S)、2.1 GB (B)、2.6 GB (L) | なし | 実測 |
| Depth Anything 3 Mono / Metric Large、Small、Base (AMD) | 深度推定 (hwnpu-image depth) | 1.4 GB (Large)、0.75 GB (Base)、0.5 GB (Small) | なし | 実測 |
| FLUX.2 klein 4B (1024 × 1024、AMD) | 画像生成、参照画像の編集 (hwnpu-image generate `-m flux2-klein`、`--image`) | 3.9 GB (NPU の最大 3.4 GiB は VAE の段、それにプロセスの 0.2 GiB。編集でも同じ。512 × 512 では 1.9 GB。文の符号器と DiT は 1 組ずつ読む) | なし | 実測 |
| FLUX.2 klein 4B (1024 × 1024、Apple) | 画像生成 (hwnpu-image generate `-m flux2-klein`。編集は AMD) | 4.9 GB (M1: VAE の段でプロセス 3.8 GiB、Neural Engine 0.8 GiB。Neural Engine は DiT のときの 2.3 GiB が最大。512 × 512 で 3.1 GB) | なし | 実測 (M1) |
| FLUX.3 klein (1024 × 1024、AMD) | 画像生成 (hwnpu-image generate `-m flux3-klein`。文からの生成) | 3.9 GB (NPU のバッファとプロセスの和の最大 3.6 GiB。512 × 512 で 2.7 GB。テキストエンコーダーと DiT は組ごとに読む) | なし | 実測 |
| FLUX.3 klein (1024 × 1024、Apple) | 画像生成 (hwnpu-image generate `-m flux3-klein`。文からの生成) | プロセスで最大 4.1 GB (M3。Neural Engine の分は未測定) | なし | 実測 (M3) |
| DA3-SMALL、DA3MONO-LARGE、MoGe-2 ViT-S (Apple) | 深度推定 (hwnpu-image depth) | 1 つの大きさのプログラム (fp16)。縦横比ごとに初めて使うときに変換 | なし | 未測定 |
| SigLIP 2 Base / Large (Apple) | ラベル付け、画像と文の埋め込み (hwnpu-image label、embed) | 約 0.9 / 3.3 GB (M3: プロセスの footprint 最大 0.45 / 1.8 GB と、Neural Engine に 0.36 / 1.24 GB。Core AI では 0.48 / 1.56 GB) | なし (1 枚、文 8 つずつ) | M3 で実測 |
| EfficientSAM-Ti / S (Apple) | 点や箱で指定する切り抜き (hwnpu-image segment) | プロセスの footprint 約 0.22 / 0.25 GB (M3。符号器は Neural Engine、復号器は CPU) | なし (1 枚ずつ) | M3 で実測 |
| Anima-Turbo、Anima-Aesthetic (1024 × 1024) | 画像生成 (オプトイン、`+nsfw`) | AMD 3.5 GB (NPU は VAE の段で最大 2.4 GiB、プロセス 0.9 GiB)、Apple 4.8 GB (M1: プロセス 3.1 GiB、Neural Engine 1.4 GiB) | なし | 実測 (AMD と M1) |
| LFM3-Audio-2B (読み上げ) | 読み上げ | 約 2.4 GB | 12 KiB/トークン | 重みから見込み |
| d1-2B、d1-Omni-2B | 状態の判定 | 約 2.4 GB、約 2.8 GB | 短い入力のみ | 重みから見込み |
| Qwen3-0.6B | チャット (事実に弱い) | M1 で 1.1 GB (Neural Engine に 0.65 GiB とプロセスの 0.35 GiB)、M2 以降は 1.6 GB (短い会話用の 1 本を足して 1.23 GiB と 0.26 GiB) | 126 KiB/トークン (文脈 40960 で 4.9 GiB。16 GB の機械では 32768 に縮めて 3.9 GiB) | M1 で実測 (M2 以降の形は `--tier 1024` で) |
| Qwen3-1.7B | チャット (日本語の事実に弱い) | 16 GB 未満は 2.1 GB、16 GB からは 2.3 GB (M1: 8 ビットの表の 1 本で Neural Engine に 1.64 GiB とプロセスの 0.23 GiB。16 GB からはプロンプトを読む関数を足して 1.80 GiB と 0.36 GiB) | 126 KiB/トークン (文脈 40960 で 4.9 GiB。16 GB の機械では 32768 に縮めて 3.9 GiB) | M1 で実測 |
| Qwen3.5-0.8B | チャット (日本語の事実に弱い) | 1.5 GB (AMD は MLP が 6 ビット。Apple は 8 ビットの表で、Neural Engine に 0.99 GiB とプロセスの 0.39 GiB)。16 GB からの M2 以降は 2.3 GB (短い会話用の 1 本を足して 1.84 GiB と 0.31 GiB) | 14 KiB/トークン (262144 で 3.38 GiB) | 実測 (AMD と M1。M2 以降の形は `--tier 2048` で) |
| Qwen3.5-2B | チャット、コード | AMD 2.4 GB (MLP は 6 ビット)、Apple 2.2 GB (M1: 6 ビットの表で Neural Engine に 1.67 GiB とプロセスの 0.39 GiB) | 14 KiB/トークン (262144 で 3.38 GiB) | 実測 (AMD と M1) |
| Qwen3.5-4B | チャット、コード | 約 4.2 GB (8 ビットの重み 3.9 GiB) | 32 KiB/トークン (262144 で 8 GiB。16 GB では 114688 トークン) | M3 の Neural Engine で動作を確認。メモリは重みから見込み |
| Qwen2.5-Coder-7B-Instruct (Apple) | コード、チャット | 約 4.4 GB (M1: GPTQ で選んだ 4 ビットの表で Neural Engine に 4.0 GB、プロセスに 0.35 GB) | 63 KiB/トークン (32768 で 1.97 GiB) | M1 で測定 (0.8 トークン/秒: Neural Engine の 2.8 GiB を超えるので M1 では選ばない)。M2 以降は未測定 |
| gpt-oss-20b | チャット、コード、推論 | 16 GB 以上の機械 (8 GB は不可)。AMD の 16 GB: 8.3 GB (NPU に 6.90 GiB を固定: 層の 32 専門家のうち 16 個を置き、ほかは選ばれたときにディスクから読む、次の層は先読み。プロセス 0.8 GB)。14.4 トークン/秒。AMD の 24 GB 以上: 14.5 GB (NPU に 13.0 GiB を固定: 専門家は全部で MXFP4 のまま、注意は 8 ビット、出力層は 4 ビット)。19〜20 トークン/秒。Apple (Core AI、M2 以降) の 16 GB: 約 6.4 GB (専門家のうち約 4.3 GB を固定し、ほかはチェックポイントを割り付け。Neural Engine に 8 と 6 ビットの表で 1.44 GiB、プロセス 0.5 GB。部品からの見積もり)。24 GB 以上は 12.2 GB (専門家を全部固定)。16 GB の M3 で 17〜19 トークン/秒 | AMD 48 KiB/トークン (8192 で 0.38 GiB)、Apple 27 KiB/トークン (131072 で 3.38 GiB) | Ryzen AI 9 365 と M3 (16 GB) で測定 |
| Qwen3.6-35B-A3B (AMD のみ) | チャット、コード、推論 | 16 GB 以上の機械。16 GB では 8.1 GB (NPU に固定 6.8 GB: 各層の 256 個の専門家のうち 32 個を置き、ほかは選ばれたときにディスクから読む。プロセス約 1 GB)、4.2 トークン/秒、1173 トークンのプロンプト 29 秒。24 GB から 16.1 GB (128 個を置く)、5.1 トークン/秒、19 秒。48 GB から 26.2 GB (全部)、5.8 トークン/秒、14 秒。専門家と MLP と出力層は 4 ビット、ミキサーは 8 ビット | 40 層のうち 30 層は再帰 (状態は一定)、AMD の文脈は 4096 | Ryzen AI 9 365 で測った (16 GB と 24 GB は、その機械と同じ読み方で) |

- 重みは 4 ビットと 8 ビットを混ぜた形です (MLP と出力の行列が 4 ビット、注意の射影が 8 ビット。画像生成は DiT が 8 ビット)。Qwen3.5-0.8B と 2B は 4 ビットでは長い英文で崩れるので、MLP と出力の行列を 6 ビットにします
- 「実測」は、NPU に固定されたメモリを Ryzen AI 9 365 で測った値に、プロセスの分を足したものです。モデルの重みの大きさで決まります
- Apple の Neural Engine (Core AI、macOS 27) の重みは 4、6、8 ビットの表です (LFM3-1.7B、Gemma 4、Qwen3.5-2B は 6 ビット、LFM3 の VL、Audio、Omni、d1、PLaMo 3 Translate の 2B と 0.8B、Qwen3、ほかの Qwen3.5 は 8 ビット、PLaMo 3 Translate 8B は 4 ビット)。生成用とプロンプトを読む用のプログラムは同じ `.aimodel` の関数なので、重みは Neural Engine に 1 回分だけ載ります。Core ML の頃は 2 本がそれぞれ重みを持ち、M3 で LFM3-1.7B は Neural Engine に 2.43 GiB、PLaMo 3 Translate 0.8B は合わせて約 2.4 GB、M1 で Qwen3-0.6B は約 3.9 GB でした
- M2 以降の Mac では、重みの 2 倍が 2.5 GiB に収まるモデル (LFM3-1.7B、Qwen3-0.6B、Qwen3.5-0.8B) は、短い会話用に注意で分けない 1 本も読み込み (`--tier`)、これが重みをもう 1 組持ちます。M1 は読み込みません (分けた形の方が速い)。8 GB の機械では、LFM3-1.7B と Qwen3.5-0.8B はこの 1 本を読み込みません (models.yaml の `tier: none` と `variants`。LFM3-1.7B は 3.1 GB になり、8 GB の機械の目安を越えるため)。`--tier auto` で読み込み、`--tier none` で 16 GB からの機械でも読み込みません
- 「M1」と書いた値は 16 GB の M1 で 2026-11-15 に測りました (docs/design.md 16.221)。2 回目からの起動で 200 トークンの答えを出す間、プロセスの footprint と Neural Engine の分を 0.1 秒ごとに見た、和の最大です。初めて動かすときは、変換のためにプロセスが一時的に大きくなります (最大で Qwen3-0.6B 2.0 GiB、PLaMo 3 Translate 0.8B 2.3、2B 2.7、LFM3 の VL、Audio、Omni 2.8〜2.9、Qwen3.5-0.8B 3.0、Qwen3.5-2B 6.0 GiB)。Gemma 4 E4B、Qwen3.5-4B、PLaMo 3 Translate 8B は Core AI では測り直していません
- 画像のモデルの「M1」は、同じ測り方で 2026-11-16 に測りました (docs/design.md 16.228)。2 回目からの起動で 1 枚を描く間の和の最大で、1024 × 1024 では VAE の最後の段が最大です。models.yaml の Chroma1 の値は測り直していません (Core ML の頃の、プロセスだけの値)
- Core AI では、長い会話の 2 段の注意のために 8 個ごとの平均も持つので、会話で増える分は 8 分の 9 になります (LFM3-1.7B で 13.5 KiB/トークン、262144 で 3.4 GiB。Gemma 4 E2B で 6.75 KiB)
- Neural Engine の分はそのドライバーのカーネルのメモリで、どのプロセスの RSS にも footprint にも出ません (アクティビティモニタや `top` には見えません)。hwnpu-chat は読み込んだ後にその量を表示します。その場でコンパイルしたプログラムは、最初の呼び出しで Neural Engine に載ります
- Qwen3-0.6B と Qwen3-1.7B も、Core AI ではほかのチャットのモデルと同じく注意で分けて動かします (28 層すべてが注意。M1 で Qwen3-0.6B は 32 トークン/秒、Qwen3-1.7B は 22)。文脈は 40960 で、キャッシュは会話とともに増え、メモリの 4 分の 1 を超える分は文脈を縮めます (16 GB で 32768)。Core ML の頃は 4096 のキャッシュを持つ 1 本のプログラムで、Qwen3-0.6B は約 3.9 GB、Qwen3-1.7B は 4.5 GB (16 GB からは 7.6 GB) でした。16 GB 未満の機械では、Qwen3-1.7B は生成のプログラムがプロンプトも 1 回 8 トークンずつ読む 1 本で動かします (models.yaml の `prefill: off` と `variants`)。Core ML で測ると、新しい 1822 トークンのプロンプトを読むのに 16.8 秒、プロンプト用のプログラムでは 3.4 秒でした (`--prefill on` でいつでも)。Core AI ではプロンプト用のプログラムは同じ重みの関数で、足しても Neural Engine に 0.16 GiB、合わせて 0.3 GB しか増えません。AMD の NPU 向けの書き出しはまだありません
- 数えていないもの: Gemma 4 の層ごとの埋め込みの表 (E2B で int8 の 2.4 GB。2026-11-17 より前に変換した bf16 のものは 4.7 GB。docs/improvements.md 12.7) と、トークンの埋め込みの表 (0.5〜0.75 GB) は、ファイルのまま置いて、使う行だけを読みます。OS のページキャッシュに入りますが、メモリが足りなければ捨てられます
- 同じモデルの複数の書き出し (文だけと画像と音声つきなど) は、ディスクの上で同じファイルを共有できます (Btrfs、XFS、APFS、ReFS)。メモリの上では共有しないので、同時に動かせば別々に数えます
