# NPU 最適化コード・モデル公式リポジトリ一覧

各ハードウェアベンダー（Apple / Qualcomm / AMD / Intel）が公式に提供している、NPU 向けモデル最適化レシピ・サンプルコード・変換ツール等のリポジトリまとめ。

## 比較表

| ベンダー | NPU 名称 | モデル最適化レシピ / サンプル集 | 量子化・変換・最適化ツール |
| :--- | :--- | :--- | :--- |
| **Apple** | Apple Neural Engine (ANE) | [`apple/coreai-models`](https://github.com/apple/coreai-models) | [`apple/coreai-optimization`](https://github.com/apple/coreai-optimization), Core AI (`coreai-torch`) |
| **Qualcomm** | Hexagon NPU (HTP) | [`qualcomm/ai-hub-models`](https://github.com/qualcomm/ai-hub-models), [`qualcomm/ai-hub-apps`](https://github.com/qualcomm/ai-hub-apps) | Qualcomm AI Hub (`qai-hub`) / QNN |
| **AMD** | Ryzen AI NPU (XDNA) | [`amd/RyzenAI-SW`](https://github.com/amd/RyzenAI-SW), [`ROCm/FastFlowLM`](https://github.com/ROCm/FastFlowLM) | [`amd/Quark`](https://github.com/amd/Quark), [`amd/IRON`](https://github.com/amd/IRON), [`huggingface/optimum-amd`](https://github.com/huggingface/optimum-amd) |
| **Intel** | Intel AI Boost NPU | [`openvinotoolkit/openvino_notebooks`](https://github.com/openvinotoolkit/openvino_notebooks), [`openvinotoolkit/openvino.genai`](https://github.com/openvinotoolkit/openvino.genai) | [`openvinotoolkit/nncf`](https://github.com/openvinotoolkit/nncf), [`huggingface/optimum-intel`](https://github.com/huggingface/optimum-intel) |

---

## 1. Apple（Apple Neural Engine / ANE）

* **[apple/coreai-models](https://github.com/apple/coreai-models)**
  * Core AI フレームワーク向けに、言語・画像・音声などのオープンソースモデルを `.aimodel` 形式へ最適化・エクスポートするための Python レシピ集および Swift ランタイムユーティリティ（`CoreAILanguageModel` 等）。
* **[apple/coreai-optimization](https://github.com/apple/coreai-optimization)**
  * Apple Silicon（ANE / GPU / CPU）上での Core AI 実行向けに、PyTorch モデルの量子化・パレット化などの圧縮・最適化を行う Apple 公式 Python ライブラリ。

---

## 2. Qualcomm（Snapdragon / Hexagon NPU）

構造・思想ともに `apple/coreai-models` に最も近いエコシステム。

* **[qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models)**（旧 `quic/ai-hub-models`）
  * Vision / Audio / LLM / 画像生成などの主要モデルについて、Snapdragon NPU（HTP）向けに最適化・量子化・エクスポートするための Python レシピ集（`qai_hub_models`）。NPU 向けにモデル構造を書き換える具体的な PyTorch 最適化コードが豊富。
* **[qualcomm/ai-hub-apps](https://github.com/qualcomm/ai-hub-apps)**
  * 最適化したモデルを Android / Windows アプリに組み込むためのエンドツーエンドのサンプルアプリ集。
* **[Qualcomm AI Hub](https://aihub.qualcomm.com/)**
  * Web 上のモデルカタログ。クラウド上の実機でのプロファイリングや、変換済みモデルのダウンロードが可能。

---

## 3. AMD（Ryzen AI / XDNA NPU）

サンプル・レシピ集、量子化ツール、NPU 特化ランタイム、低レイヤー開発ツールに分かれている。

* **[amd/RyzenAI-SW](https://github.com/amd/RyzenAI-SW)**
  * Ryzen AI NPU 向けの公式デモ・チュートリアル集。CNN（ResNet / YOLOv8 等）や LLM / VLM を量子化し、ONNX Runtime（Vitis AI EP / ONNX Runtime GenAI）経由で NPU 実行するサンプルを収録。
* **[amd/Quark](https://github.com/amd/Quark)**
  * PyTorch / ONNX モデルを NPU 向け（INT8 / INT4 等）に量子化・最適化するための公式ツールキット。
* **[ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM)**
  * AMD Ryzen AI NPU（XDNA / XDNA2）に特化して高度に最適化された LLM 推論ランタイム。
* **[amd/IRON](https://github.com/amd/IRON)** / **[amd/Triton-XDNA](https://github.com/amd/Triton-XDNA)**
  * MLIR-AIE や Triton ベースで、NPU 向けカスタムカーネルを低レイヤーから記述・チューニングするための公式リポジトリ。
* **[huggingface/optimum-amd](https://github.com/huggingface/optimum-amd)** / **[Hugging Face AMD 公式](https://huggingface.co/amd)**
  * Hugging Face モデルを Ryzen AI 向けに量子化・実行するための公式インテグレーションおよび最適化済みモデルコレクション。

---

## 4. Intel（Core Ultra / Intel AI Boost NPU）

OpenVINO エコシステムを中心に、サンプル集と最適化ツールを提供している。

* **[openvinotoolkit/openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks)**
  * Intel（OpenVINO チーム）公式の Jupyter Notebook 集。LLM、画像生成、YOLO、Whisper など多数のモデルについて、変換・量子化から NPU 実行（`device="NPU"`）までの実践的なサンプルを収録。
* **[openvinotoolkit/openvino.genai](https://github.com/openvinotoolkit/openvino.genai)**
  * LLM や VLM などの生成 AI モデルを NPU 等で高速に推論するための高レベル C++ / Python API とサンプル。
* **[openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf)**
  * NPU 向けの量子化（INT8 / INT4 等）やモデル圧縮を行う公式フレームワーク（Neural Network Compression Framework）。
* **[huggingface/optimum-intel](https://github.com/huggingface/optimum-intel)** / **[Hugging Face OpenVINO 公式](https://huggingface.co/OpenVINO)**
  * Hugging Face モデルを OpenVINO IR 形式へエクスポートし、NPU 上で推論するための公式ライブラリおよび最適化済みモデルコレクション。
* **[intel/intel-npu-acceleration-library](https://github.com/intel/intel-npu-acceleration-library)**
  * OpenVINO を介さず、PyTorch から直接 Intel NPU を叩くための軽量ライブラリ。

---

## 最適化コードを読む際のポイント

* **モデル構造の書き換えテクニック（グラフ・オペレータ単位の最適化）** を参照したい場合は、各モデルの PyTorch 実装がまとまっている **[qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models)** が特に参考になる。
* **NPU のハードウェア特性に踏み込んだ低レイヤー最適化** を参照したい場合は、**[amd/IRON](https://github.com/amd/IRON)** や **[ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM)** が参考になる。
