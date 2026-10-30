"""Golden log mel frames, for checking internal/llm/lfm3audio's Features.

    python mel_golden.py OUT.json

Computes the features of LFM3-Audio's feature extractor as llama.cpp's
mtmd_audio_preprocessor_lfm3a does (16 kHz, n_fft 400, a periodic Hann window of 400, hop 160,
the clip padded with zeros on the right so that the last frame takes its last sample, the power
spectrum, 128 Slaney mel filters from 0 to 8 kHz, log10 with a floor of 1e-12, then (v + 4) / 4)
with librosa's filter bank and STFT, for two clips (one shorter than a window), and writes the
clips and the frames. Needs librosa and numpy.
"""
import argparse
import json

import librosa
import numpy as np


def features(y):
    win, hop = 400, 160
    n = len(y)
    padded = win if n <= win else n + (hop - (n - win) % hop) % hop
    y = np.pad(y, (0, padded - n))
    power = np.abs(librosa.stft(y, n_fft=400, hop_length=hop, win_length=win, window="hann", center=False)) ** 2
    mel = librosa.filters.mel(sr=16000, n_fft=400, n_mels=128, fmin=0, fmax=8000, htk=False, norm="slaney") @ power
    return ((np.log10(np.maximum(mel, 1e-12)) + 4) / 4).T  # frames x mels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    t = np.arange(4111) / 16000
    clip = 0.4 * np.sin(2 * np.pi * (200 + 1500 * t) * t) + 0.05 * rng.standard_normal(len(t))
    short = 0.3 * np.sin(2 * np.pi * 440 * np.arange(250) / 16000)
    cases = []
    for name, y in (("chirp", clip), ("short", short)):
        y = y.astype(np.float32)
        cases.append({"name": name, "samples": y.tolist(), "frames": features(y.astype(np.float64)).astype(np.float32).tolist()})
    with open(a.out, "w") as f:
        json.dump({"cases": cases}, f)
        f.write("\n")


if __name__ == "__main__":
    main()
