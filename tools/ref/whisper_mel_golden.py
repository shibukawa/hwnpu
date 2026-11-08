"""Golden log mel frames, for checking internal/llm/whisper's Features.

    python whisper_mel_golden.py OUT.json

Computes the features of transformers' WhisperFeatureExtractor (128 mel bins, as
whisper-large-v3 and its turbo) for a 0.4 second chirp, and writes the clip, its first 64 frames
and the last frame (the zeros the clip is padded with to 30 seconds: every frame past the clip is
the same). Needs transformers and numpy.
"""
import argparse
import json

import numpy as np
from transformers import WhisperFeatureExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    a = ap.parse_args()
    rng = np.random.default_rng(5)
    t = np.arange(6400) / 16000
    clip = (0.4 * np.sin(2 * np.pi * (300 + 2000 * t) * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    fe = WhisperFeatureExtractor(feature_size=128)
    f = fe(clip, sampling_rate=16000, return_tensors="np").input_features[0].T  # frames x mels
    json.dump({"samples": clip.tolist(), "frames": f[:64].tolist(), "last": f[-1].tolist(), "count": len(f)}, open(a.out, "w"))


if __name__ == "__main__":
    main()
