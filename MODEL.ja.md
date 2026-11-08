# おすすめのモデル

hwnpu が動かすモデルを、用途と機械のメモリの量で選べるようにまとめました。数値は 2026-11-06 の main のものです。各 NPU が扱うデータ形式と、モデルの変換の仕方は [MODEL_ARCHITECTURE.ja.md](MODEL_ARCHITECTURE.ja.md) にあります。

English: [MODEL.md](MODEL.md)

## 用途とメモリで選ぶ

列は機械に積んだメモリの量です。OS とふだんのアプリの分を引いて、モデルに使える量の目安を 2 行目に書きました。NPU のバッファは固定されたメモリで、スワップに逃がせません。目安を超えると、ほかのアプリがスワップに追い出されます。

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
<th>チャット、コーディング<br><sub>hwnpu-chat、hwnpu-agent</sub></th>
<td><b>LFM3-1.7B</b> (2.2 GB)<br>速く、エージェント向け<br><b>Gemma 4 E2B</b> (2.3 GB)<br>層ごとの埋め込みを含めて 5B の、ひと回り大きなモデル</td>
<td><b>Gemma 4 E2B</b> (2.3 GB)<br>長い文脈 (131072 で +0.8 GB) も<br><b>Gemma 4 E4B</b> (4.0 GB)<br>短い会話なら</td>
<td colspan="2"><b>Gemma 4 E4B</b> (4.0 GB)<br>長い文脈 (131072 で +2.1 GB) も</td>
<td colspan="2"><b>Gemma 4 E4B</b> (4.0 GB)<br><i>予定: Qwen3.6-35B-A3B (MoE、4 ビットで約 25 GB)</i></td>
</tr>
<tr>
<th>画像を読む<br><sub>hwnpu-chat (<code>image_url</code>、<code>/image</code>)</sub></th>
<td><b>LFM3-VL-2B</b> (2.6 GB)</td>
<td colspan="5"><b>Gemma 4 E2B 画像と音声つき</b> (3.4 GB)<br><b>LFM3-VL-2B</b> (2.6 GB)</td>
</tr>
<tr>
<th>文字起こし<br><sub>hwnpu-transcribe</sub></th>
<td colspan="6"><b>Whisper large-v3-turbo</b> (約 1.9 GB)<br>99 言語、タイムスタンプつき、30 秒を 0.6 秒で読む</td>
</tr>
<tr>
<th>音声を読む (音声についての質問)<br><sub>hwnpu-chat (<code>input_audio</code>、<code>/audio</code>)</sub></th>
<td><b>LFM3-Audio-2B</b> (2.6 GB)</td>
<td colspan="5"><b>Gemma 4 E2B 画像と音声つき</b> (3.4 GB)<br><b>LFM3-Omni-2B</b> (3.0 GB、画像も)</td>
</tr>
<tr>
<th>翻訳<br><sub>hwnpu-chat <code>-use plamo-3-translate-2b</code>、<code>-use plamo-3-translate-0.8b</code></sub></th>
<td><b>PLaMo 3 Translate 0.8B</b> (約 2.4 GB)<br>2B の 3 倍速、訳は 2B の言い換え程度の違い</td>
<td colspan="5"><b>PLaMo 3 Translate 2B</b> (2.7 GB)</td>
</tr>
<tr>
<th>画像生成<br><sub>hwnpu-image</sub></th>
<td>△ <b>Anima-Turbo</b> (4.1 GB)<br>ほかのアプリを閉じて</td>
<td colspan="5"><b>Anima-Turbo</b> 1024 × 1024 (4.1 GB)</td>
</tr>
<tr>
<th>読み上げ<br><sub>hwnpu-tts</sub></th>
<td colspan="6"><b>LFM3-Audio-2B</b> (約 2.4 GB)</td>
</tr>
<tr>
<th>文の埋め込み、検索<br><sub>hwnpu-chat <code>/v1/embeddings</code></sub></th>
<td colspan="6"><b>PLaMo-Embedding-1B</b> (1.3 GB)</td>
</tr>
<tr>
<th>状態の判定<br><sub>hwnpu-systemone</sub></th>
<td colspan="6"><b>d1-2B</b> (約 2.4 GB、画像も)<br><b>d1-Omni-2B</b> (約 2.8 GB、画像と音声)</td>
</tr>
<tr>
<th>いくつも同時に<br><sub>hwnpu-chat <code>-listen -use A,B,…</code></sub></th>
<td>1 つずつ</td>
<td>Gemma 4 E2B + PLaMo-Embedding-1B (3.6 GB)<br>+ PLaMo 3 Translate 2B (6.3 GB)</td>
<td>Gemma 4 E4B + PLaMo-Embedding-1B + PLaMo 3 Translate 2B (8.0 GB)</td>
<td colspan="3">その上に Gemma 4 E2B 画像と音声つき、別のプロセスで Anima-Turbo (約 15.5 GB)</td>
</tr>
</tbody>
</table>

- 括弧の中は、起動して短い会話をしたときの想定のメモリです (次の節)。長い会話では KV キャッシュの分が増えます
- 32 GB を超えても、今は大きなモデルがありません。メモリの多い機械では、上の表の組み合わせを同時に動かすか、長い文脈に使います
- 小さな機械では、`-context` で文脈を短くすると KV キャッシュの分が減ります。hwnpu-chat は KV キャッシュが物理メモリの 4 分の 1 を超えるとき、文脈を自分で縮めます

## モデルごとのメモリ

「起動時」は、NPU のバッファ (重み、作業領域、既定の文脈の KV キャッシュ) とプロセス自身 (トークナイザー、ホストのバッファ。約 0.4 GB) の合計です。NPU のバッファはプロセスの RSS には出ないので、タスクマネージャーや `top` に出る値より大きくなります。

| モデル | 用途 | 起動時 | 会話で増える分 (KV キャッシュ) | 根拠 |
|---|---|---|---|---|
| LFM3-1.7B | チャット、コーディング | 2.2 GB (Apple の Neural Engine では約 3.0〜3.3 GB) | 12 KiB/トークン (262144 で 3 GiB) | 実測 (AMD と M3) |
| Gemma 4 E2B | チャット、コーディング | 2.3 GB | 6 KiB/トークン (131072 で 0.75 GiB) | 実測 |
| Gemma 4 E2B 画像と音声つき | 画像、音声 (1 回に 1 枚と 30 秒まで) | 3.4 GB (画像か音声が来るまでは 2.4 GB) | 6 KiB/トークン | 実測 |
| Gemma 4 E4B | チャット、コーディング | 4.0 GB | 16 KiB/トークン (131072 で 2 GiB) | 実測 |
| LFM3-VL-2B | 画像 (512 × 512) | 2.6 GB | 12 KiB/トークン (128000 で 1.46 GiB) | 実測 |
| Whisper large-v3-turbo | 文字起こし (長さの上限なし、30 秒ずつ) | 約 1.9 GB (M3: Neural Engine に 0.78 GiB とプロセスの 1.1 GB) | なし (30 秒ごとに 448 トークンまで) | M3 で実測 |
| LFM3-Audio-2B | 音声 (1 本 30 秒まで) | 2.6 GB | 12 KiB/トークン | 実測 |
| LFM3-Omni-2B | 画像と音声 | 3.0 GB (画像か音声が来るまでは 2.2 GB) | 12 KiB/トークン | 実測 |
| PLaMo 3 Translate 2B | 翻訳 | 2.7 GB | 7 KiB/トークン (262144 で 1.75 GiB) | 実測 |
| PLaMo 3 Translate 0.8B | 翻訳 | 1.0 GB (M3: Neural Engine に 1.38 GiB とプロセスの 1.0 GB で約 2.4 GB) | 1 KiB/トークン (262144 で 0.28 GiB) | 実測 (AMD と M3) |
| PLaMo-Embedding-1B | 文の埋め込み | 1.3 GB | なし | 重みから見込み |
| Anima-Turbo (1024 × 1024) | 画像生成 | 1.9 GB (VAE の段の最大。DiT はブロックの重みを 1 組ずつ読む) | なし | 実測 |
| LFM3-Audio-2B (読み上げ) | 読み上げ | 約 2.4 GB | 12 KiB/トークン | 重みから見込み |
| d1-2B、d1-Omni-2B | 状態の判定 | 約 2.4 GB、約 2.8 GB | 短い入力のみ | 重みから見込み |
| Qwen3.5-0.8B | チャット (日本語の事実に弱い) | 1.5 GB (MLP は 6 ビット。M3: Neural Engine に 1.47 GiB とプロセスの 1.1 GB で約 2.6 GB) | 14 KiB/トークン (262144 で 3.38 GiB) | 実測 (AMD と M3) |
| Qwen3.5-2B | チャット、コード | 2.4 GB (MLP は 6 ビット。M3: Neural Engine に 6 ビットの表の 2 本で 2.67 GiB とプロセスの 1.5 GB で約 4.2 GB。`-prefill off` で 1.32 GiB) | 14 KiB/トークン (262144 で 3.38 GiB) | 実測 (AMD と M3) |
| Qwen3.5-4B | チャット、コード | 約 4.2 GB (8 ビットの重み 3.9 GiB) | 32 KiB/トークン (262144 で 8 GiB。16 GB では 114688 トークン) | M3 の Neural Engine で動作を確認。メモリは重みから見込み |
| Qwen3.6-35B-A3B (予定) | チャット、コード | 約 25 GB | — | 4 ビットの重みからの見込み。移植していない |

- 重みは 4 ビットと 8 ビットを混ぜた形です (MLP と出力の行列が 4 ビット、注意の射影が 8 ビット。画像生成は DiT が 8 ビット)。Qwen3.5-0.8B と 2B は 4 ビットでは長い英文で崩れるので、MLP と出力の行列を 6 ビットにします
- 「実測」は、NPU に固定されたメモリを Ryzen AI 9 365 で測った値に、プロセスの分を足したものです。モデルの重みの大きさで決まります
- Apple の Neural Engine ではもっと多くなります。重みは 6 ビットと 8 ビットの表 (LFM3-1.7B と Qwen3.5-2B は 6 ビット、LFM3 の VL、Audio、Omni、d1 と PLaMo 3、ほかの Qwen3.5 は 8 ビット) で、hwnpu-chat は生成用とプロンプトを読む用の 2 本のプログラムを読み込み、それぞれが重みを持ちます (Gemma 4 E4B、PLaMo 3 Translate 2B、Qwen3.5-4B は 1 本)。M3 で測ると (2026-11-07)、LFM3-1.7B は Neural Engine に 2.43 GiB (1.21 GiB のプログラム 2 本)、それにプロセスの 0.5〜0.9 GB、PLaMo 3 Translate 0.8B は 2 本で 1.38 GiB、それにプロセスの 1.0 GB でした。ほかのモデルは Apple ではまだ測っていませんが、同じく表より多くなる見込みです
- Neural Engine の分はそのドライバーのカーネルのメモリで、どのプロセスの RSS にも footprint にも出ません (アクティビティモニタや `top` には見えません)。hwnpu-chat は読み込んだ後にその量を表示します。その場でコンパイルしたプログラムは、最初の呼び出しで Neural Engine に載ります
- 数えていないもの: Gemma 4 の層ごとの埋め込みの表 (E2B で 4.4 GB) と、トークンの埋め込みの表 (0.5〜0.75 GB) は、ファイルのまま置いて、使う行だけを読みます。OS のページキャッシュに入りますが、メモリが足りなければ捨てられます
- 同じモデルの複数の書き出し (文だけと画像と音声つきなど) は、ディスクの上で同じファイルを共有できます (Btrfs、XFS、APFS、ReFS)。メモリの上では共有しないので、同時に動かせば別々に数えます
