"""Reference pitch of SwiftF0 (lars76/swift-f0), for checking internal/pitch.

    python ref_swiftf0.py MODEL.onnx OUT_PREFIX [--seconds 3] [--wav FILE] [--fmin HZ --fmax HZ]

Runs swift_f0/model.onnx with onnx's reference evaluator (onnxruntime need not be installed) as
SwiftF0.detect does: windows of 1875 frames with 11 frames before and 10 after, the pitch decoded
from the graph's logits with the graph's own tail (written out in numpy, since the reference
evaluator does not take its GatherElements), confidence 0 for frames whose hop peaks under 1e-3,
and the loudness of core.py. The clip is a test signal (a glide, a vibrato and noise, with a
silent gap) unless --wav (16 kHz mono) is given. Writes, little endian:

    OUT_PREFIX.audio.bin     float32 (samples)
    OUT_PREFIX.features.bin  float32 (3, 128, frames) of the first window: the CNN's input
    OUT_PREFIX.logits.bin    float32 (96, frames) of the first window: 95 pitch logits, voicing
    OUT_PREFIX.pitch.bin     float64 (3, frames of the clip): pitch, confidence, loudness
    OUT_PREFIX.json          sizes and the band

Needs numpy and onnx.
"""
import argparse
import json
import wave

import numpy as np
import onnx
from onnx import helper
from onnx.reference import ReferenceEvaluator

HOP, WINDOW, LEFT, AHEAD = 256, 1875, 11, 10


def body(path):
    """The graph up to the logits: features v352, pitch logits v429 (1, T, 95), voicing v437."""
    m = onnx.load(path)
    last = max(i for i, n in enumerate(m.graph.node) if "v437" in n.output)
    nodes = list(m.graph.node)[: last + 1]
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    del m.graph.output[:]
    for n in ("v352", "v429", "v437"):
        m.graph.output.append(helper.make_tensor_value_info(n, onnx.TensorProto.UNDEFINED, None))
    ins = [i for i in m.graph.input if i.name == "audio"]
    del m.graph.input[:]
    m.graph.input.extend(ins)
    init = {t.name: onnx.numpy_helper.to_array(t) for t in m.graph.initializer}
    return ReferenceEvaluator(m), init["v438"], init["v455"]


def decode(logits, voicing, hz, loghz, fmin, fmax):
    """The graph's tail: logits (T, 95), voicing (T,)."""
    inband = (hz >= fmin) & (hz <= fmax)
    masked = np.where(inband, logits, -np.inf)
    best = masked.argmax(-1)
    padded = np.pad(masked, ((0, 0), (1, 1)), constant_values=-np.inf)
    idx = best[:, None] + np.array([-1, 0, 1])
    three = np.take_along_axis(padded, idx + 1, -1).astype(np.float64)
    p = np.exp(three - three.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    pitch = np.exp((p * loghz[np.clip(idx, 0, 94)]).sum(-1))
    sig = (1 / (1 + np.exp(-voicing.astype(np.float32)))).astype(np.float32)
    conf = np.where(inband[logits.argmax(-1)], sig, 0).astype(np.float64)
    return pitch, conf


def run(ev, hz, loghz, audio, fmin, fmax):
    feats, logits, voicing = ev.run(None, {"audio": audio[None, :]})
    pitch, conf = decode(logits[0], voicing[0], hz, loghz, fmin, fmax)
    n = len(conf)
    hops = audio[: n * HOP].reshape(n, HOP) if len(audio) >= HOP else audio[None, :]
    conf[np.abs(hops).max(axis=1) < 1e-3] = 0.0
    power = (hops.astype(np.float64) ** 2).sum(1)
    loud = 20 * np.log10(np.maximum(np.sqrt((np.concatenate(([0.0], power[:-1])) + power) / 512), 1e-7))
    return feats[0], logits[0], voicing[0], pitch, conf, loud


def signal(seconds, sr=16000):
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(0)
    f = 110 * 2 ** (3 * t / seconds) * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
    x = 0.4 * np.sin(2 * np.pi * np.cumsum(f) / sr) + 0.1 * np.sin(4 * np.pi * np.cumsum(f) / sr)
    x += 0.005 * rng.standard_normal(len(t))
    x[int(len(x) * 0.4) : int(len(x) * 0.5)] = 0  # digital silence
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=3)
    ap.add_argument("--wav")
    ap.add_argument("--fmin", type=float, default=46.875)
    ap.add_argument("--fmax", type=float, default=2093.75)
    a = ap.parse_args()
    if a.wav:
        with wave.open(a.wav) as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    else:
        audio = signal(a.seconds)
    ev, hz, loghz = body(a.model)
    n = max(1, len(audio) // HOP)
    parts, first = [], None
    for start in range(0, n, WINDOW):
        end = min(start + WINDOW, n)
        left = max(0, start - LEFT)
        win = audio[left * HOP : (end + AHEAD) * HOP] if end < n else audio[left * HOP :]
        feats, logits, voicing, pitch, conf, loud = run(ev, hz, loghz, win, a.fmin, a.fmax)
        if first is None:
            first = (feats, np.concatenate([logits.T, voicing[None]], 0))
        parts.append(np.stack([pitch, conf, loud])[:, start - left : end - left])
    out = np.concatenate(parts, 1)
    audio.tofile(a.out + ".audio.bin")
    first[0].astype(np.float32).tofile(a.out + ".features.bin")
    first[1].astype(np.float32).tofile(a.out + ".logits.bin")
    out.astype(np.float64).tofile(a.out + ".pitch.bin")
    meta = {"samples": len(audio), "frames": out.shape[1], "window_frames": int(first[0].shape[-1]), "fmin": a.fmin, "fmax": a.fmax}
    with open(a.out + ".json", "w") as f:
        json.dump(meta, f)
    voiced = out[1] >= 0.5
    print(f"{a.out}: {out.shape[1]} frames, {voiced.sum()} voiced, median pitch of the voiced {np.median(out[0][voiced]):.1f} Hz")


if __name__ == "__main__":
    main()
