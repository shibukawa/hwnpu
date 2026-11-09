"""Reference notes of Basic Pitch (spotify/basic-pitch), for checking internal/notes.

    python ref_basic_pitch.py NMP.onnx NOTE_CREATION.py OUT_PREFIX [--wav FILE] [--seconds 5]

Runs nmp.onnx with onnx's reference evaluator (onnxruntime need not be installed) on the windows
basic_pitch.inference.run_inference cuts (22050 Hz, 43844 samples, 30 frames of overlap, half of
it as silence before the audio), unwraps the outputs as it does, and makes the notes with
basic-pitch's own note_creation.py (given as a file; librosa and pretty_midi, which it imports,
are replaced by the two conversions it uses), with predict's defaults: onset 0.5, frame 0.3,
notes of 127.7 ms and over, the melodia trick, pitch bends. The clip is a test melody with a
chord unless --wav (22050 Hz mono, 16 bit) is given. Writes, little endian:

    OUT_PREFIX.audio.bin    float32 (samples)
    OUT_PREFIX.cut.bin      float32 (172, 309) of the first window: the graph's normalized log CQT
    OUT_PREFIX.window.bin   float32 (172, 440) of the first window: note, onset and contour
    OUT_PREFIX.output.bin   float32 (frames, 440): the unwrapped note, onset and contour
    OUT_PREFIX.json         the notes (start s, end s, pitch, amplitude, bends) and sizes

Needs numpy, scipy and onnx.
"""
import argparse
import importlib.util
import json
import sys
import types
import wave

import numpy as np
import onnx
from onnx import helper
from onnx.reference import ReferenceEvaluator

SR, HOP, N_SAMPLES, OVERLAP_FRAMES = 22050, 256, 43844, 30
CUT = None  # found in the graph: the output of normalized_log's div_no_nan


def stubs():
    librosa = types.ModuleType("librosa")
    librosa.midi_to_hz = lambda m: 440.0 * 2.0 ** ((np.asarray(m) - 69.0) / 12.0)
    librosa.hz_to_midi = lambda f: 12 * (np.log2(np.asarray(f)) - np.log2(440.0)) + 69
    core = types.ModuleType("librosa.core")
    core.frames_to_time = lambda frames, sr, hop_length: np.asarray(frames) * hop_length / sr
    librosa.core = core
    sys.modules["librosa"], sys.modules["librosa.core"] = librosa, core
    pm = types.ModuleType("pretty_midi")
    pm.PrettyMIDI = object
    sys.modules["pretty_midi"] = pm
    for name in ("resampy", "mir_eval"):
        sys.modules[name] = types.ModuleType(name)
    consts = types.ModuleType("basic_pitch.constants")
    consts.AUDIO_SAMPLE_RATE, consts.FFT_HOP, consts.AUDIO_N_SAMPLES = SR, HOP, N_SAMPLES
    consts.ANNOTATIONS_FPS, consts.ANNOT_N_FRAMES = SR // HOP, 2 * (SR // HOP)
    consts.CONTOURS_BINS_PER_SEMITONE, consts.ANNOTATIONS_BASE_FREQUENCY = 3, 27.5
    consts.N_FREQ_BINS_CONTOURS, consts.ANNOTATIONS_N_SEMITONES = 264, 88
    pkg = types.ModuleType("basic_pitch")
    pkg.constants = consts
    sys.modules["basic_pitch"], sys.modules["basic_pitch.constants"] = pkg, consts


def melody(seconds):
    t = np.arange(int(SR * seconds)) / SR
    x = np.zeros_like(t)
    notes = [(60, 0.0, 0.5), (62, 0.5, 1.0), (64, 1.0, 1.5), (65, 1.5, 2.0), (67, 2.0, 3.0), (72, 3.0, 3.5)]
    notes += [(48, 3.5, seconds), (52, 3.5, seconds), (55, 3.5, seconds)]  # a chord
    for p, a, b in notes:
        f = 440 * 2 ** ((p - 69) / 12)
        env = np.clip((t - a) * 50, 0, 1) * np.clip((b - t) * 50, 0, 1) * np.exp(-2 * np.clip(t - a, 0, None))
        for h, amp in ((1, 0.5), (2, 0.25), (3, 0.12)):
            x += amp * env * np.sin(2 * np.pi * f * h * t)
    return (0.3 * x).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("note_creation")
    ap.add_argument("out")
    ap.add_argument("--wav")
    ap.add_argument("--seconds", type=float, default=5)
    a = ap.parse_args()
    if a.wav:
        with wave.open(a.wav) as w:
            assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    else:
        audio = melody(a.seconds)

    m = onnx.load(a.model)
    cut = next(o for n in m.graph.node for o in n.output if n.op_type == "Where" and "normalized_log" in o)
    m.graph.output.append(helper.make_tensor_value_info(cut, onnx.TensorProto.UNDEFINED, None))
    ev = ReferenceEvaluator(m)
    names = [o.name for o in m.graph.output]

    overlap = OVERLAP_FRAMES * HOP
    hop = N_SAMPLES - overlap
    padded = np.concatenate([np.zeros(overlap // 2, np.float32), audio])
    outs, first = [], None
    for i in range(0, len(padded), hop):
        win = padded[i : i + N_SAMPLES]
        win = np.pad(win, (0, N_SAMPLES - len(win)))
        res = dict(zip(names, ev.run(None, {m.graph.input[0].name: win[None, :, None]})))
        y = np.concatenate([res["StatefulPartitionedCall:1"][0], res["StatefulPartitionedCall:2"][0], res["StatefulPartitionedCall:0"][0]], -1)
        if first is None:
            first = (res[cut][0], y)
        outs.append(y)
    n_olap = OVERLAP_FRAMES // 2
    out = np.concatenate([o[n_olap:-n_olap] for o in outs])
    n_frames = int(len(audio) / hop * (2 * (SR // HOP) - OVERLAP_FRAMES))
    out = out[:n_frames]

    stubs()
    spec = importlib.util.spec_from_file_location("note_creation", a.note_creation)
    nc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nc)
    min_len = int(np.round(127.7 / 1000 * (SR / HOP)))
    notes = nc.output_to_notes_polyphonic(out[:, :88].copy(), out[:, 88:176].copy(), onset_thresh=0.5, frame_thresh=0.3,
                                          min_note_len=min_len, infer_onsets=True, max_freq=None, min_freq=None, melodia_trick=True)
    notes = nc.get_pitch_bends(out[:, 176:], notes)
    times = nc.model_frames_to_time(out.shape[0])
    events = [(float(times[s]), float(times[e]), int(p), float(amp), [int(b) for b in bends]) for s, e, p, amp, bends in notes]

    audio.tofile(a.out + ".audio.bin")
    first[0].astype(np.float32).tofile(a.out + ".cut.bin")
    first[1].astype(np.float32).tofile(a.out + ".window.bin")
    out.astype(np.float32).tofile(a.out + ".output.bin")
    with open(a.out + ".json", "w") as f:
        json.dump({"samples": len(audio), "frames": out.shape[0], "cut": cut, "notes": events}, f)
    print(f"{a.out}: {out.shape[0]} frames, {len(events)} notes:", sorted((round(s, 2), p) for s, _, p, _, _ in events))


if __name__ == "__main__":
    main()
