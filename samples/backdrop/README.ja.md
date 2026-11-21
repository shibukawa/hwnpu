# backdrop

[English](README.md) · [サンプルの一覧](../README.ja.md)

カメラの映像で、自分の後ろをぼかします。絵や単色に差し替えることもできます。ビデオ通話の背景と同じものです。フレームごとに人物と背景を分け (MediaPipe Selfie Segmentation)、窓はフレームとそのマットを重ねて描きます (Ebitengine の Kage のシェーダー)。

```
ネットワーク (動き続ける): camera → cutout → steady → view   (MediaPipe Selfie Segmentation と、プログラムの処理)
窓:                      1 秒に 60 回、ネットワークが最後に出したフレームとマットで描く
```

モデルは CPU の 1 スレッドで動きます。Ryzen AI 9 365 で 1280×720 のフレーム 1 枚が 5.9 ms なので、カメラの全フレームにマットがつきます。重みは 0.46 MB で、NPU に載せても得にならない小ささです。

## 動かし方

```bash
cd samples
go run ./backdrop models prepare               # モデルのダウンロード (0.5 MB)
go run ./backdrop                              # 窓: カメラ、後ろをぼかす
go run ./backdrop --behind beach.jpg           # 後ろを絵に
go run ./backdrop --behind "#00aa44"           # 後ろを単色に (グリーンバック)
go run ./backdrop --frames a.png,b.png         # カメラの代わりに絵をフレームとして (カメラのない機械)
```

| コマンド | 何をするか |
|---|---|
| `backdrop` | 窓。`--behind blur\|FILE\|#RRGGBB`、`--blur 0.02` (ぼかしの届く範囲。フレームの幅に対する割合)、`--steady 0.5` (前のフレームのマットをどれだけ残すか)、`--no-mirror`、`--size 1280x720`、`--for 30s` |
| `backdrop cameras` | カメラの一覧 (番号つき)。`--device 2` で 2 番目 |
| `backdrop models`、`backdrop status` | モデルと要るもの、この機械の NPU |

窓では、B で後ろを切り替え (ぼかし、絵または色、そのまま)、M でマットを表示、上下でぼかしの強さ、Q で終了です。

## 後ろのぼかし方

フレームをそのままぼかすと、自分の色がまわりににじみます。そこで、フレームに「人物でない割合」(1 − マット) の重みをかけ、4 分の 1 の大きさに縮めてぼかし、ぼかした重みで割ります。混ざるのは後ろの色だけです。マットはモデルの大きさ (256×256) で、GPU で引き伸ばします。

## 要るもの

- モデル: MediaPipe Selfie Segmentation (Google、Apache-2.0、0.5 MB)。`-m` の指定はなく、定義の用途 `cutout` を使います。
- CPU で動かすための ONNX Runtime の共有ライブラリ。システムのもの (Ubuntu の `libonnxruntime1.23` は見つけます) か、[リリース](https://github.com/microsoft/onnxruntime/releases)の `libonnxruntime.so` を `HWNPU_ONNXRUNTIME` で指します。
- カメラの前の人物のためのモデルです。人物のいないフレームでは目立つものを人物として返し、広がった髪や指のような細いものは追えません。
- 確認は AMD の機械で、絵をフレームとして渡し、ソフトウェア描画の画面で行いました。実際のカメラ、実際の画面、Mac では試していません。
- 画面や Ebitengine のライブラリのない機械では `-tags headless` でビルドします (`cameras`、`models`、`status` は動きます)。
