#!/usr/bin/env python3
"""Export a language model IRON compiles (an iron.lm.CausalLM) for hwnpu.

Writes, into a directory:

  manifest.json    the model's shape, the arena's banks, and each version (the decode
                   step, the prompt chunk): its commands, arguments, per-command
                   scratchpad words, carried values, logits and input encoders
  decode.elf       the decode step: one token in, the logits after it out
  prompt.elf       a prompt chunk of up to prefill_chunk tokens, the last one's logits out
  x_*.xclbin       on NPU1 (no full ELF; --device npu1 --cols 4) or with --per-step: each
  stream_*.json    version is an xclbin per step instead, the xclbins, the programs of the
  <v>_insts.bin    steps' per-call instruction streams (iron_stream) and the static ones
  bank_<kind>.bin  the arena's banks as the device holds them: every weight in place,
                   zeros elsewhere (the caches and the other states start at zero)
  gather_*.bin     the control word templates of the prompt's embedding gathers
  check/           with --check: a prompt and the logits after it and after a few
                   greedy steps (IRON's graph on the host with random weights)

hwnpu writes a call's values and reads its logits; it never uses the token the device
draws. Run in IRON's virtual environment; no NPU is needed.

  python tools/amd/iron_export_lm.py gemma4 ~/models/gemma-4-E2B-it OUT --max-seq-len 4096

See docs/amd-npu-models.md.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iron_export import _carried, _form, _place, bind_device  # noqa: E402

MANIFEST_VERSION = 1

MODELS = {
    # Gemma 4 whose prompt chunks run the KV-shared layers on their last row alone
    # (iron_gemma4.py); gemma4full is IRON's own, every row through every layer.
    "gemma4": ("iron_gemma4", "Gemma4Tail"),
    "gemma4full": ("iron.applications.gemma4.model", "Gemma4"),
    # Gemma 4 with its vision and audio towers (one image and one clip a prompt).
    "gemma4mm": ("iron_gemma4", "Gemma4TailMultimodal"),
    "gemma4mmfull": ("iron.applications.gemma4.model", "Gemma4Multimodal"),
    # LFM3 (LFM3-1.7B): an application kept in hwnpu (iron_lfm3.py), IRON has none.
    "lfm3": ("iron_lfm3", "Lfm3"),
    # LFM3-VL (LFM3-VL-2B): LFM3 with its vision tower, for one image size (iron_lfm3vl.py).
    "lfm3vl": ("iron_lfm3vl", "Lfm3Vl"),
    # LFM3-Audio (LFM3-Audio-2B): LFM3 with its audio encoder, audio in (iron_lfm3audio.py).
    "lfm3audio": ("iron_lfm3audio", "Lfm3Audio"),
    # LFM3-Omni (LFM3-Omni-2B): LFM3 with both towers, images and audio in (iron_lfm3omni.py).
    "lfm3omni": ("iron_lfm3omni", "Lfm3Omni"),
    # PLaMo 3 (pfnet/plamo-3-translate-2b): an application kept in hwnpu (iron_plamo3.py).
    "plamo3": ("iron_plamo3", "Plamo3"),
    # Qwen3.5 (qwen3_5): Gated DeltaNet (iron_gdn.py) and gated attention (iron_qwen35.py).
    "qwen35": ("iron_qwen35", "Qwen35"),
    # Qwen3.6-35B-A3B (qwen3_5_moe): Qwen3.5 with experts routed on the device (iron_moe.py).
    "qwen35moe": ("iron_qwen35", "Qwen35Moe"),
    # gpt-oss (openai/gpt-oss-20b): sink attention and MXFP4 experts kept exact (iron_gptoss.py).
    "gptoss": ("iron_gptoss", "GptOss"),
}


def _params_by_command(path) -> dict:
    """The parameters each command reads, from the lowering's params.txt: by command
    name, the parameter names (each command has a scratchpad of its own)."""
    if path is None:
        return {}
    lines = [ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()]
    i = 0
    nparams = int(lines[i][0])
    i += 1 + nparams
    njoint = int(lines[i][0])
    i += 1 + njoint
    ncommands = int(lines[i][0])
    i += 1
    out = {}
    for _ in range(ncommands):
        name, count, *names = lines[i]
        assert int(count) == len(names), lines[i]
        out[name] = names
        i += 1
    return out


def _encoders(compiled, place=None) -> list:
    """The host-side encoders of the version's inputs: the embedding gathers of the prompt's
    token ids (IRON's GatherRows.control_words)."""
    seq = compiled.sequence
    out = []
    for name, held in compiled.recording.encoders.items():
        start = 0
        for e in held:
            op = getattr(e.encode, "__self__", None)
            if op is None or type(op).__name__ not in ("Gather", "GatherRows") or not hasattr(op, "template"):
                raise SystemExit(f"input {name}: encoder {e.encode} is not a row gather")
            template, slots = op.template
            from iron.common.image.step import BufferRef

            kind, offset, length = (place or _place)(seq, BufferRef(e.buffer))
            out.append(dict(
                input=name, start_word=start, words=int(e.words), table=e.buffer,
                table_argument=kind, table_offset=offset,
                rows=int(op.rows), table_rows=int(op.table_rows),
                stride=int(op.row * np.dtype(op.dtype).itemsize), pad_bytes=int(op.pad_bytes),
                template=np.asarray(template, np.uint32), slots=[int(s) for s in slots],
                bound=None if e.bound is None else _form(e.bound),
            ))
            start += int(e.words)
    return out


def _derived(compiled) -> dict:
    """By symbol, the operator and the derived value a word without a closed form is, with
    the Affine forms of the extents the operator's derivation reads (IRON's _words)."""
    rec = compiled.recording
    out, seen = {}, set()
    for b in rec.bindings:
        if id(b.op) in seen or b.member.bounds is None:
            continue
        seen.add(id(b.op))
        op = b.op.resolved()
        counts = {e.member.name: e.expression for e in rec.bindings if e.op is b.op and e.member.bounds is not None}
        for name in sorted(op._per_call_derived() - op.bound_values.keys()):
            out.setdefault(op.value(name).symbol, (op, name, counts))
    return out


MHA_CONSTANTS = ("B_q", "B_kv", "num_pipelines", "packed", "causal", "window", "seq_len", "kv_len", "seq_pad", "num_heads", "num_KV_heads")


def _derived_entry(w, derived: dict, version: str) -> dict:
    """A word with no closed form: an MHA derivation by its constants and extents, or else a
    table over a prompt chunk's rows (the words of the row counts)."""
    op, name, counts = derived[w.symbol]
    if type(op).__name__ == "MHA":
        const = {}
        for k in MHA_CONSTANTS:
            v = getattr(op, k)
            const[k] = None if v is None else (bool(v) if isinstance(v, (bool, np.bool_)) else int(v))
        return dict(derived=dict(op="MHA", name=name, const=const, extents={k: _form(e) for k, e in counts.items()}))
    if version != "prompt":
        raise SystemExit(f"word {w.symbol} ({type(op).__name__}.{name}) has no form")
    C, L = _CONFIG["prefill_chunk"], _CONFIG["max_seq_len"]

    def at(start, rows):
        return int(w(dict(token=0, position=start + rows - 1, start=start, rows=rows, rope_start=start)))

    for s in (1, 100, L - C):
        for r in (1, 7, C):
            if at(s, r) != at(0, r):
                raise SystemExit(f"word {w.symbol} ({type(op).__name__}.{name}) follows more than a chunk's rows")
    return dict(table=dict(value="rows", lo=1, values=[at(0, r) for r in range(1, C + 1)]))


_CONFIG = {}


_TOWER = {}


def _vision_grid(compiled) -> dict:
    """Every scratchpad word of the vision tower for each image size it can run at, as hwnpu
    feeds it: the image resized on the host to the processor's size (out_height by
    out_width, hk by wk blocks of pool * patch pixels, hk * wk soft tokens at most the
    budget), so the tower's own resampling is 1:1. Keyed by (hk, wk)."""
    tower, c = _TOWER["vision"], _TOWER["vision"].config
    side = c.pooling_kernel_size * c.patch_size
    addr = set(compiled.recording.addresses)
    # Words of a buffer's address are computed at run time from their forms.
    gridded = [w for w in compiled.words if not (w.form is not None and w.form.value.name in addr)]
    pairs, values = [], {w.symbol: [] for w in gridded}
    zero = {x.name: 0 for x in compiled.recording.values}
    name = _TOWER["vision_input"]
    for hk in range(1, c.image_tokens + 1):
        for wk in range(1, c.image_tokens // hk + 1):
            oh, ow = hk * side, wk * side
            v = dict(zero)
            own = dict(n=oh // c.patch_size * (ow // c.patch_size), hk=hk, wk=wk, walked=tower.walked(hk, wk),
                       height=oh, width=ow, out_height=oh, out_width=ow)
            v.update({f"{name}_{k}": x for k, x in own.items()})
            pairs.append([hk, wk])
            for w in gridded:
                values[w.symbol].append(int(w(v)))
    return dict(side=side, pairs=pairs, values=values)


def _audio_grid(compiled) -> dict:
    """Every scratchpad word of the audio tower for each count of valid feature frames, which
    the clip's length decides and n and tokens follow from. Keyed by frames."""
    tower, c = _TOWER["audio"], _TOWER["audio"].config
    addr = set(compiled.recording.addresses)
    gridded = [w for w in compiled.words if not (w.form is not None and w.form.value.name in addr)]
    keys, values = [], {w.symbol: [] for w in gridded}
    zero = {x.name: 0 for x in compiled.recording.values}
    name = _TOWER["audio_input"]
    if _TOWER.get("audio_kind") == "lfm3audio":
        # LFM3-Audio's encoder: the valid rows of its three stride-2 stages
        for frames in range(1, c.max_frames + 1):
            n1 = -(-frames // 2)
            n2 = -(-n1 // 2)
            v = dict(zero)
            v.update({f"{name}_n1": n1, f"{name}_n2": n2, f"{name}_tokens": -(-n2 // 2)})
            keys.append(frames)
            for w in gridded:
                values[w.symbol].append(int(w(v)))
        return dict(frames=keys, values=values)
    for frames in range(1, 4 * c.max_tokens + 1):
        tokens = c.tokens(frames)
        if tokens > c.max_tokens:
            break
        v = dict(zero)
        v.update({f"{name}_n": -(-frames // 2), f"{name}_frames": frames, f"{name}_tokens": tokens})
        keys.append(frames)
        for w in gridded:
            values[w.symbol].append(int(w(v)))
    return dict(frames=keys, values=values)


def _word_entry(w, derived: dict, version: str) -> dict:
    """How a word follows from the call's values: an Affine form, a linear one, or a
    derivation (_derived_entry)."""
    if w.form is not None:
        return dict(form=_form(w.form))
    if w.linear is not None:
        # ceil(v * ratio + offset), both fractions
        lin = w.linear
        return dict(linear=dict(value=lin.value, num=lin.ratio.numerator, den=lin.ratio.denominator,
                                off_num=lin.offset.numerator, off_den=lin.offset.denominator))
    return _derived_entry(w, derived, version)


def _version_kind(compiled) -> str:
    names = {h.name for h in compiled.recording.inputs}
    return "prompt" if names & {"ids", "placed"} else "tower" if names else "decode"


def _addresses(compiled, place=None) -> list:
    """Per-call values computed from a buffer's device address (a gather of rows the graph
    computed, not an input): base_lo and base_hi of the table's address."""
    from iron.common.image.step import BufferRef

    rec, seq = compiled.recording, compiled.sequence
    addresses = []
    for name, (buffer, address_words, member) in rec.addresses.items():
        op = getattr(address_words, "__self__", None)
        if op is None or not hasattr(op, "low_bits") or member not in ("base_lo", "base_hi"):
            raise SystemExit(f"address value {name}: {address_words} {member} is not known")
        kind, offset, length = (place or _place)(seq, BufferRef(buffer))
        addresses.append(dict(name=name, buffer=buffer, argument=kind, offset=offset, low_bits=int(op.low_bits), member=member))
    return addresses


def _place_steps(seq, ref):
    """_place for an xclbin per step (IRON's PerStepCallable): a resident (a weight, a cache)
    is in the arena's banks as in a full ELF, but a transient is in a pool of the version's
    own, whose offsets start again from zero: its banks are the arguments pool, pool_1..."""
    from iron.common.image.allocator import BANK

    kind, offset, length = ref.place(seq.layout)
    if kind != "scratch":
        return kind, int(offset), int(length)
    if ref.root in seq.residents:
        return _place(seq, ref)
    k = offset // BANK
    if length and (offset + length - 1) // BANK != k:
        raise SystemExit(f"transient {ref.root} crosses a bank of the pool")
    return ("pool" if k == 0 else f"pool_{k}"), int(offset - k * BANK), int(length)


def _pool_arguments(seq) -> list:
    """The pool's banks an xclbin per step keeps its transients in, as (kind, bytes)."""
    from iron.common.image.allocator import BANK
    from iron.common.image.step import BufferRef

    size = 0
    for root, (kind, offset, length) in seq.layout.items():
        if kind == "scratch" and root not in seq.residents:
            size = max(size, offset + length)
    return [("pool" if k == 0 else f"pool_{k}", min(BANK, size - k * BANK)) for k in range((size + BANK - 1) // BANK)]


def _io(compiled, place=None) -> dict:
    """The version's buffers, arguments, values, carried values, outputs, inputs and encoders,
    which a full ELF and an xclbin per step describe alike (place: _place_steps for the
    latter)."""
    from iron.common.image.step import BufferRef

    place = place or _place
    rec, seq = compiled.recording, compiled.sequence
    outputs = []
    for h in rec.returned:
        kind, offset, length = place(seq, h.ref())
        outputs.append(dict(name=h.name, shape=[int(d) for d in h.shape], dtype=str(np.dtype(seq.buffer_dtype(h.ref()))),
                            argument=kind, offset=offset, bytes=length))
    inputs = []
    for h in rec.inputs:
        kind, offset, length = place(seq, h.ref())
        inputs.append(dict(name=h.name, shape=[int(d) for d in h.shape], dtype=str(np.dtype(h.dtype)),
                           argument=kind, offset=offset, bytes=length))
    buffers = {}
    for root in seq.layout:
        kind, offset, length = place(seq, BufferRef(root))
        buffers[root] = dict(argument=kind, offset=offset, bytes=length, dtype=str(np.dtype(seq.buffer_dtype(BufferRef(root)))))
    return dict(
        buffers=buffers,
        arguments=[dict(index=i, kind=k, bytes=int(n)) for i, (k, n) in enumerate(
            list(seq.buffer_sizes.arguments().items()) + (_pool_arguments(seq) if place is _place_steps else []))],
        values=[v.name for v in rec.values],
        # An xclbin per step carries nothing on the device: every value is a dispatch-time
        # scalar the host gives each call.
        carried=_carried(compiled) if compiled.artifacts.full_elf else None,
        outputs=outputs,
        inputs=inputs,
        encoders=_encoders(compiled, place),
        addresses=_addresses(compiled, place),
        plan=compiled.plan.report(rec.name),
    )


def steps_manifest(compiled, files: dict) -> dict:
    """The manifest of a version packaged as an xclbin per step (NPU1, which has no full
    ELF): each step's xclbin and kernel, its instruction stream (static, or a program of the
    step's dispatch-time scalars: iron_stream), and its buffers in order. ``files`` gathers
    what export writes: xclbins and programs by content, static streams in one file."""
    import hashlib

    import iron_stream

    seq = compiled.sequence
    words = {w.symbol: w for w in compiled.words}
    derived = _derived(compiled)
    version = _version_kind(compiled)
    files.setdefault("words", {})[version] = words
    used, steps = {}, []
    for st in seq.steps:
        k = seq.packaging.links[id(st.op)].kernel
        xb = Path(k.xclbin_path).read_bytes()
        xname = f"x_{hashlib.sha256(xb).hexdigest()[:16]}.xclbin"
        files["xclbins"][xname] = xb
        step = dict(xclbin=xname, kernel=k.kernel_name,
                    buffers=[dict(zip(("argument", "offset", "bytes"), _place_steps(seq, r))) for r in st.buffers])
        if k.dispatch_params:
            cpp = (Path(k.dispatch_lib_path).parent / "dispatch_gen.cpp").read_text()
            pname = f"stream_{hashlib.sha256(cpp.encode()).hexdigest()[:16]}.json"
            if pname not in files["programs"]:
                files["programs"][pname] = (iron_stream.parse(cpp), k)
            step["program"] = pname
            step["scalars"] = list(k.dispatch_params)
            for n in k.dispatch_params:
                if n not in words:
                    raise SystemExit(f"dispatch scalar {n} is no word of the version")
                used[n] = words[n]
            files["checks"].append((pname, k, list(k.dispatch_params), version))
        else:
            insts = np.asarray(k.instructions(), np.uint32)
            step["insts"] = [files["insts_words"], int(insts.size)]
            files["insts"].append(insts)
            files["insts_words"] += int(insts.size)
        steps.append(step)
    table = {}
    for n, w in used.items():
        table[n] = dict(name=n, **_word_entry(w, derived, version))
    m = _io(compiled, _place_steps)
    m.update(steps=steps, words=table)
    return m


def version_manifest(compiled, elf_name: str) -> dict:
    """The manifest of a version packaged as one full ELF (NPU2)."""
    from iron.common.image.callable import FullELFCallable
    from iron.common.image.compose import Composition

    art = compiled.artifacts
    seq = compiled.sequence
    commands = [
        f"{FullELFCallable.device_name}:{Composition.sequence_name(k)}"
        for k in range(len(seq.packaging.commands) + 1)
    ]
    words = {w.symbol: w for w in compiled.words}
    derived = _derived(compiled)
    missing = []
    inputs = {h.name for h in compiled.recording.inputs}
    grid = (_vision_grid(compiled) if _TOWER.get("vision_input") in inputs
            else _audio_grid(compiled) if _TOWER.get("audio_input") in inputs else None)
    version = _version_kind(compiled)
    table = {}
    for p in art.parameter_table:
        w = words.get(p.name)
        if w is None:
            raise SystemExit(f"parameter {p.name} is no word of the version")
        entry = dict(name=p.name, index=int(p.index), dtype=p.dtype, kind=p.kind)
        if grid is not None and p.name in grid["values"]:
            entry["grid"] = True  # looked up in the version's grid
        else:
            try:
                entry.update(_word_entry(w, derived, version))
            except SystemExit as e:
                op, dname, counts = derived[w.symbol]
                missing.append(f"{type(op).__name__}.{dname} from {sorted(counts)}: {e}")
                continue
        table[p.name] = entry
    if missing:
        raise SystemExit("words without a form:\n  " + "\n  ".join(sorted(set(missing))))
    members = _params_by_command(art.params)
    kernels = []
    for name in commands:
        names = members.get(name, list(table) if len(commands) == 1 else [])
        kernels.append(dict(name=name, parameters=[table[n] for n in names]))
    m = _io(compiled)
    m.update(elf=elf_name, kernels=kernels, grid=grid)
    return m


def host_steps(compiled) -> list:
    """The version's host steps hwnpu runs itself (a step's fn with an ``hwnpu`` dict: Qwen3.6's
    prompt experts, iron_qwen35.MoEHost): before which command, what it is, and the placement,
    shape and type of what it reads and writes. Any other host step is refused."""
    rec, seq = compiled.recording, compiled.sequence
    edges = list(compiled.plan.commands)
    out = []
    for h in rec.host:
        info = getattr(h.fn, "hwnpu", None)
        if info is None and not h.reads and not h.writes:
            continue  # a command's end alone (iron_qwen35's decode step: on_host(lambda: None))
        if info is None:
            raise SystemExit(f"{rec.name}: a host step ({h.fn!r}) hwnpu does not know")
        if h.at not in edges:
            raise SystemExit(f"{rec.name}: host step {info['kind']} at step {h.at}, no command's start")

        def region(x):
            kind, offset, length = _place(seq, x.ref())
            return dict(name=x.name, argument=kind, offset=int(offset), bytes=int(length),
                        shape=[int(d) for d in x.shape], dtype=str(np.dtype(x.dtype)))

        out.append(dict(command=edges.index(h.at) + 1, **info,
                        reads=[region(x) for x in h.reads], writes=[region(x) for x in h.writes]))
    return out


def compile_versions(model, full_elf=None):
    """The decode step and the prompt chunk, compiled into one arena as load_versions does.
    full_elf False packages each as an xclbin per step (NPU1's only packaging; on NPU2, to
    check that path on a machine without NPU1)."""
    import aie.utils as aie_utils

    dev = aie_utils.get_current_device() if hasattr(aie_utils, "get_current_device") else None
    versions, decode = {}, None
    for shapes in model.shapes():
        prompt = "ids" in shapes or "placed" in shapes
        v = model.compile(dev=dev, feeds=decode if prompt else None, full_elf=full_elf, **shapes)
        if not shapes and v.emit is not None:  # only a looping full ELF feeds the prompt (IRON's load_versions)
            decode = v
        verify = "ids" in shapes and shapes["ids"][0][0] == getattr(model, "verify_rows", 0)
        name = ("verify" if verify else "placed" if "placed" in shapes else "prompt" if "ids" in shapes
                else "vision" if "rgb" in shapes else "audio" if "wave" in shapes
                else "decode" if not shapes else None)
        if name is None or name in versions:
            raise SystemExit(f"a version on {sorted(shapes)} is not known")
        versions[name] = v
    return versions


def write_banks(versions, banks: dict, out: Path) -> dict:
    """Write each arena bank holding weights, the weights at their offsets."""
    images = {}
    seen = set()
    for v in versions.values():
        seq = v.sequence
        for key, (tensor, handle) in v.recording.weights.items():
            # Every version's weights: a version may hold its own copy of one, at a place of
            # its own, under a name another version gives a different weight.
            kind, offset, length = _place(seq, handle.ref())
            if kind not in banks:
                raise SystemExit(f"weight {handle.name} is in {kind}, not an arena bank")
            if kind not in images:
                images[kind] = np.lib.format.open_memmap(out / f"bank_{kind}.npy.tmp", mode="w+", dtype=np.uint8, shape=(banks[kind],))
            b = np.ascontiguousarray(np.asarray(tensor).astype(np.dtype(seq.buffer_dtype(handle.ref())))).view(np.uint8).reshape(-1)
            if b.size > length:
                raise SystemExit(f"weight {handle.name}: {b.size} bytes in a buffer of {length}")
            images[kind][offset : offset + b.size] = b
    names = {}
    for kind, img in images.items():
        img.flush()
        name = f"bank_{kind}.bin"
        # The raw bytes, without the .npy header.
        with open(out / name, "wb") as f:
            f.write(memoryview(img))
        del img
        (out / f"bank_{kind}.npy.tmp").unlink()
        names[kind] = name
    return names


SHIFT = {"core": 2, "repeat": 16, "addr": 0}


def write_words(versions, man: dict, out: Path) -> None:
    """The scratchpad words IRON computes for a few calls, as each command's slot -> word
    (shifted as written), in check/words.json: what hwnpu must write for the same values."""
    L, C = _CONFIG["max_seq_len"], _CONFIG["prefill_chunk"]
    calls = {
        "decode": [dict(token=7, position=p, start=0, rows=1, rope_start=0) for p in (0, 1, 19, 100, L - 1)],
        "prompt": [dict(token=7, position=s + r - 1, start=s, rows=r, rope_start=s) for s, r in ((0, 20), (0, C), (100, 37), (C, C))],
    }
    res = {}
    for name, v in versions.items():
        if name not in calls or "kernels" not in man["versions"][name]:
            continue  # an xclbin per step has no scratchpad: write_steps checks its streams
        words = {w.symbol: w for w in v.words}
        res[name] = []
        for values in calls[name]:
            values = {x.name: 0 for x in v.recording.values} | values  # towers idle
            per_cmd = []
            for k in man["versions"][name]["kernels"]:
                slots = {}
                for p in k["parameters"]:
                    x = int(words[p["name"]](values))
                    slots[str(p["index"])] = (x << SHIFT[p["kind"]]) & 0xFFFFFFFF
                per_cmd.append(slots)
            res[name].append(dict(values=values, commands=per_cmd))
    d = out / "check"
    d.mkdir(exist_ok=True)
    (d / "words.json").write_text(json.dumps(res) + "\n")


def write_steps(files: dict, out: Path) -> None:
    """The files of the versions packaged as an xclbin per step: their xclbins, and their
    stream programs, each checked against the dispatch library mlir-aie built for it over
    calls across the context (check/streams.json keeps a few for hwnpu to check)."""
    import iron_stream

    for name, b in files["xclbins"].items():
        (out / name).write_bytes(b)
    L, C = _CONFIG["max_seq_len"], _CONFIG["prefill_chunk"]
    calls = {
        "decode": [dict(token=7, position=p, start=0, rows=1, rope_start=0)
                   for p in (0, 1, 2, 63, 64, 100, 1000, C - 1, C, 2 * C - 1, 2 * C, L - 96, L - 1)],
        "prompt": [dict(token=7, position=s + r - 1, start=s, rows=r, rope_start=s)
                   for s, r in ((0, 1), (0, 20), (0, C), (100, 37), (C, C), (L - C, C), (L - 1, 1), (L - 40, 40))],
    }
    checked, kept = set(), []
    for pname, k, scalars, version in files["checks"]:
        prog = files["programs"][pname][0]
        key = (pname, tuple(scalars))
        if key in checked:
            continue
        checked.add(key)
        for values in calls.get(version, [{}]):
            sc = [int(files["words"][version][n](values)) for n in scalars]
            try:
                want = [int(x) for x in k.instructions(**dict(zip(scalars, sc)))]
            except Exception:  # the dispatch library refused the call
                want = None
            try:
                got = iron_stream.run(prog, sc)
            except iron_stream.Refused:
                got = None
            if got != want:
                raise SystemExit(f"stream program {pname} differs from its dispatch library at {values}")
            if len(kept) < 64 and want is not None:
                kept.append(dict(program=pname, scalars=sc, stream=want))
    for pname, (prog, _) in files["programs"].items():
        (out / pname).write_text(json.dumps(prog, separators=(",", ":")) + "\n")
    if kept:
        d = out / "check"
        d.mkdir(exist_ok=True)
        (d / "streams.json").write_text(json.dumps(kept, separators=(",", ":")) + "\n")


def export(model, versions, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    c = model.text.config if hasattr(model, "towers") else model.config
    vm = {}
    files = dict(xclbins={}, programs={}, checks=[], insts=[], insts_words=0)
    for name, v in versions.items():
        if v.artifacts.full_elf:
            shutil.copyfile(v.artifacts.image, out / f"{name}.elf")
            vm[name] = version_manifest(v, f"{name}.elf")
            if v.recording.host:
                vm[name]["host"] = host_steps(v)
        else:
            files.update(insts=[], insts_words=0)
            vm[name] = steps_manifest(v, files)
            if files["insts"]:
                fn = f"{name}_insts.bin"
                np.concatenate(files["insts"]).astype("<u4").tofile(out / fn)
                vm[name]["insts"] = fn
        for i, e in enumerate(vm[name]["encoders"]):
            fn = f"gather_{name}_{i}.bin"
            e.pop("template").tofile(out / fn)
            e["template"] = fn
    # The arena's banks: every argument that is neither the input, the output nor the
    # feedback, as big as the largest version needs (all versions are placed in one arena).
    banks = {}
    for m in vm.values():
        for a in m["arguments"]:
            if a["kind"] not in ("input", "output", "feedback") and not a["kind"].startswith("pool"):
                banks[a["kind"]] = max(banks.get(a["kind"], 0), a["bytes"])
    conv_pairs = model.hwnpu_conv_pairs() if hasattr(model, "hwnpu_conv_pairs") else []
    for e in conv_pairs:
        # the convolution's own state is the prompt chunk's alone: a resident, at the same place
        # in every version, listed in the decode step's too (the checkpoints and hwnpu read it)
        for k in ("state", "conv"):
            have = [m["buffers"][e[k]] for m in vm.values() if e[k] in m["buffers"]]
            if not have or any(h != have[0] for h in have):
                raise SystemExit(f"conv_pairs: buffer {e[k]} not one resident in the versions")
            for m in vm.values():
                m["buffers"].setdefault(e[k], have[0])
    if hasattr(model, "program_rows"):
        # Qwen35Moe: the commands whose scratchpad the device writes (iron_moe.py), and the
        # routing programs that write them, known once the lowering has placed their words.
        vm["decode"]["fed"] = model.program_rows(vm["decode"]["kernels"])
    images = write_banks(versions, banks, out)
    write_steps(files, out)
    for n in range(1, 3 * c.prefill_chunk, max(1, c.prefill_chunk // 3)):
        if (model.text if hasattr(model, "towers") else model).rope_row(n, 0) != 0:
            raise SystemExit("rope_row is not the chunk's start: not supported yet")
    mm = None
    kind = getattr(model, "hwnpu_kind", None)
    if kind == "lfm3omni":
        # LFM3-Omni: LFM3-VL's vision tower and LFM3-Audio's encoder, each run between the
        # vision markers
        mc = model.config
        v, au = mc.vision, mc.audio
        it, at = int(mc.image_token_id), int(mc.audio_token_id)
        start, end = int(mc.vision_start_token_id), int(mc.vision_end_token_id)
        mm = dict(
            kind="lfm3omni",
            image_token_id=it, audio_token_id=at,
            boi_token_id=start, eoi_token_id=end, boa_token_id=start, eoa_token_id=end,
            placeholders=[int(t) for t in model.at],
            vision=dict(width=int(v.width), height=int(v.height), patch_size=int(v.patch_size),
                        image_tokens=int(v.tokens), patches=int(v.patches), patch_dim=int(v.patch_dim)),
            audio=dict(sample_rate=16000, hop=160, frame=400, max_frames=int(au.max_frames), mels=int(au.mels),
                       max_tokens=int(au.max_tokens), max_samples=int(au.max_frames) * 160),
            tower_values={str(t): [f"{model.tower_inputs[t]}_{x}" for x in model.tower_values[t]] for t in model.towers},
        )
        c = model.text.config
        model_name, rope_model, recurrent = type(model.text).__name__ + "Omni", model.text, model.text.recurrent
    elif kind == "lfm3audio":
        # LFM3-Audio: the audio encoder, its input the clip's log-mel features; a clip's run of
        # placeholders has no markers around it
        mc = model.config
        au = mc.audio
        (token,) = model.towers
        mm = dict(
            kind="lfm3audio",
            image_token_id=-1, audio_token_id=int(mc.audio_token_id),
            boi_token_id=-1, eoi_token_id=-1, boa_token_id=-1, eoa_token_id=-1,
            placeholders=[int(t) for t in model.at],
            audio=dict(sample_rate=16000, hop=160, frame=400, max_frames=int(au.max_frames), mels=int(au.mels),
                       max_tokens=int(au.max_tokens), max_samples=int(au.max_frames) * 160),
            tower_values={str(token): [f"{model.tower_inputs[token]}_{x}" for x in model.tower_values[token]]},
        )
        c = model.text.config
        model_name, rope_model, recurrent = type(model.text).__name__ + "Audio", model.text, model.text.recurrent
    elif kind == "lfm3vl":
        # LFM3-VL: one vision tower for one image size, its input the image's patches
        mc = model.config
        v = mc.vision
        mm = dict(
            kind="lfm3vl",
            image_token_id=int(mc.image_token_id), audio_token_id=-1,
            boi_token_id=int(mc.vision_start_token_id), eoi_token_id=int(mc.vision_end_token_id),
            boa_token_id=-1, eoa_token_id=-1,
            placeholders=[int(t) for t in model.at],
            vision=dict(width=int(v.width), height=int(v.height), patch_size=int(v.patch_size),
                        image_tokens=int(v.tokens), patches=int(v.patches), patch_dim=int(v.patch_dim)),
            tower_values={str(t): [] for t in model.towers},
        )
        c = model.text.config
        model_name, rope_model, recurrent = type(model.text).__name__ + "Vl", model.text, model.text.recurrent
    elif hasattr(model, "towers"):
        mc = model.config
        v, au = mc.vision, mc.audio
        text = model.text
        mm = dict(
            image_token_id=int(mc.image_token_id), audio_token_id=int(mc.audio_token_id),
            boi_token_id=int(mc.boi_token_id), eoi_token_id=int(mc.eoi_token_id),
            boa_token_id=int(mc.boa_token_id), eoa_token_id=int(mc.eoa_token_id),
            placeholders=[int(t) for t in model.at],  # the order of the placed version's before
            vision=dict(
                pooling_kernel_size=int(v.pooling_kernel_size), patch_size=int(v.patch_size),
                image_tokens=int(v.image_tokens), patches=int(v.patches), largest=[int(x) for x in v.largest],
                chunk=int(v.chunk),
                image_chunks=int(model.towers[mc.image_token_id].processor.resample.image_chunks),
                groups=[[int(g[0]), int(g[1])] for g in model.towers[mc.image_token_id].groups],
            ),
            audio=dict(sample_rate=int(au.sample_rate), frame=int(au.frame), hop=int(au.hop),
                       max_samples=int(au.max_samples), max_tokens=int(au.max_tokens)),
            tower_values={str(t): [f"{model.tower_inputs[t]}_{x}" for x in model.tower_values[t]] for t in model.towers},
        )
        c = text.config
        model_name, rope_model, recurrent = _name(text), text, text.recurrent
    else:
        model_name, rope_model, recurrent = _name(model), model, model.recurrent
    m = dict(
        version=MANIFEST_VERSION,
        model=model_name,
        multimodal=mm,
        config=dict(
            vocab_size=int(c.vocab_size), hidden_size=int(c.hidden_size), max_seq_len=int(c.max_seq_len),
            prefill_chunk=int(c.prefill_chunk), num_hidden_layers=int(c.num_hidden_layers),
        ),
        rope_start="start",
        # Whether the state restarts anywhere: no ring caches, no recurrent layers
        # (CausalLM.recurrent). Only then is a position all a sequence needs to go back.
        restartable=not recurrent,
        # The recurrent layers' states (IRON's Checkpoints, LFM3's short convolutions): a slot
        # per prompt-chunk boundary, slot k the state after the positions up to k * chunk run
        # so far. A host that copies the slot a position reads goes back to it.
        checkpoints=_checkpoints(vm["decode"]["buffers"], recurrent, c),
        # Qwen3.6's folded Gated DeltaNet: the convolution's rows in two homes, which the host
        # copies between around a prompt chunk (iron_qwen35's hwnpu_conv_pairs)
        conv_pairs=conv_pairs or None,
        # The ring caches (a sliding window's, when 3 rings are under max_seq_len), which no
        # copy of a slot brings back.
        rings=sum(1 for r in recurrent if not _is_checkpoints(r)),
        # The KV caches' rows (positions), by buffer: max_seq_len, or more for iron_window's
        # sliding layers (a prompt chunk's rows past the context). A position's row is its bytes
        # over these.
        cache_rows=_cache_rows(model, vm["decode"]["buffers"]),
        banks={k: dict(bytes=int(n), image=images.get(k)) for k, n in banks.items()},
        versions=vm,
    )
    # Tables the host reads rows of into a device buffer before each call (iron_hostple)
    import iron_hostple

    host_tables = iron_hostple.write(model, vm["decode"]["buffers"], out)
    if host_tables:
        m["host_tables"] = host_tables
    (out / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    write_words(versions, m, out)
    return m


def _cache_rows(model, buffers) -> dict:
    """By KV cache buffer (".cache." in its name), its rows: the first dimension of the state the
    name leads to from the model (model.layers.4.cache.keys). Names that lead nowhere are left
    out, and hwnpu takes max_seq_len rows for them."""
    out = {}
    for name in buffers:
        if ".cache." not in name:
            continue
        obj = model
        for part in name.split("."):
            obj = obj[int(part)] if part.isdigit() and hasattr(obj, "__getitem__") else getattr(obj, part, None)
            if obj is None:
                break
        shape = getattr(obj, "shape", None)
        if shape:
            out[name] = int(shape[0])
    return out


def _name(model) -> str:
    """The model's name in the manifest, which picks hwnpu's chat format: a variant's
    ``hwnpu_name`` (Gemma4Tail is Gemma4), else its class's."""
    return getattr(model, "hwnpu_name", type(model).__name__)


def _is_checkpoints(state) -> bool:
    from iron.lm import Checkpoints

    return isinstance(state, Checkpoints)


def _checkpoints(buffers: dict, recurrent, c) -> list:
    """The buffers of the recurrent states, ``*.slots`` in the layout, one per Checkpoints."""
    names = sorted(n for n in buffers if n.endswith(".slots"))
    count = sum(1 for r in recurrent if _is_checkpoints(r))
    if len(names) != count:
        raise SystemExit(f"{count} recurrent states, {len(names)} buffers named *.slots: {names}")
    slots = (int(c.max_seq_len) - 1) // int(c.prefill_chunk) + 2
    out = []
    for n in names:
        if buffers[n]["bytes"] % slots:
            raise SystemExit(f"{n}: {buffers[n]['bytes']} bytes are no {slots} slots")
        out.append(dict(buffer=n, slots=slots, chunk=int(c.prefill_chunk)))
    return out


def write_check(Model, args, out: Path, prompt_tokens: int, steps: int, rng: np.random.Generator) -> None:
    """A prompt and the logits after it and after each greedy step, from IRON's graph run
    on the host with the same random weights. (Model.cpu_reference() would do it, but for
    Gemma 4 it walks the checkpoint's tree twice; a model built with OnHost mixed in does
    not.)"""
    from iron.lm.model import OnHost

    if args.random_weights is None:
        raise SystemExit("--check needs --random-weights: real weights are judged by Hugging Face's logits")
    Host = type("Host", (OnHost, Model), {})
    ref = Host.from_checkpoint(args.checkpoint, random_weights=args.random_weights, max_seq_len=args.max_seq_len)
    model = ref
    tokens = [int(t) for t in rng.integers(0, model.config.vocab_size, prompt_tokens)]
    d = out / "check"
    d.mkdir(exist_ok=True)
    calls = []
    history = list(tokens)
    for k in range(steps + 1):
        logits = np.asarray(ref.logits(history), np.float32).reshape(-1)[: model.config.vocab_size]
        fn = f"logits_{k}.bin"
        logits.tofile(d / fn)
        calls.append(dict(tokens=len(history), logits=fn, top=int(np.argmax(logits))))
        history.append(int(np.argmax(logits)))
    (d / "check.json").write_text(json.dumps(dict(prompt=tokens, calls=calls)) + "\n")


def prompt_attention_profile(model, pipelines: int) -> None:
    """Give the prompt chunk's attention (MHA at prefill_chunk rows) ``pipelines`` pipelines, as
    IRON's Llama profile does (iron/applications/llama/profiles/npu2.json: 8 at its chunk, 4 in
    decode). Without a profile MHA takes its default, 1: Gemma 4 E2B's chunk of 1024 spent 60 %
    of its time in attention (design.md 16.107). The decode step's attention is left as it is."""
    from iron.common.declare.profile import Profile
    from iron.operators.mha import MHA

    if model.profile is not None:
        raise SystemExit(f"--mha-pipelines: {type(model).__name__} has a profile of its own")
    C = int(model.config.prefill_chunk)
    # MHA takes whole rounds of 64 rows on each pipeline (a chunk of 256: 4 at most), and an even
    # number above 6
    while pipelines > 1 and (C % (64 * pipelines) or (pipelines > 6 and pipelines % 2)):
        pipelines -= 1
    if pipelines > 1:
        p = Profile()
        p.add(MHA, seq_pad=C, num_pipelines=pipelines)
        model.profile = p
    print(f"prompt attention: {pipelines} pipelines at {C} rows")


def install_command_per_step(out: Path) -> None:
    """Cut a prompt chunk's full ELF into a command a step (each runs on its own, as the plan's
    commands queue them), and write each command's operator into out/steps.json, so that
    HWNPU_IRON_TIMING's times say which operator they are. The decode step (no input tensor
    beyond its token) is left as it is."""
    import dataclasses

    from iron.common.graph import compiled

    base = compiled.plan
    steps = {}

    def plan(dev, recording, *args, **kwargs):
        p = base(dev, recording, *args, **kwargs)
        if p.full_elf and len(recording.steps) > 1 and recording.inputs:
            p = dataclasses.replace(p, commands=tuple(range(1, len(recording.steps))))
            steps[recording.name] = [type(s.op).__name__ + " " + str(getattr(s.op, "name", "")) for s in recording.steps]
            out.mkdir(parents=True, exist_ok=True)
            (out / "steps.json").write_text(json.dumps(steps, indent=1))
        return p

    compiled.plan = plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("checkpoint", help="the Hugging Face directory (config.json, weights, tokenizer)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--prefill-chunk", type=int, default=None,
                    help="tokens of a prompt chunk (default 512 for qwen35, IRON's own for qwen35moe, 1024 for the "
                         "others: design.md 16.54, 16.108)")
    ap.add_argument("--random-weights", type=int, default=None, help="weights drawn at this seed")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--per-step", action="store_true", help="package each version as an xclbin per step (as on NPU1, which has no full ELF)")
    ap.add_argument("--check", type=int, default=0, metavar="TOKENS", help="write a check of a prompt of TOKENS tokens")
    ap.add_argument("--steps", type=int, default=3, help="greedy steps of the check")
    ap.add_argument("--check-only", action="store_true", help="write the check alone, into an export made before")
    ap.add_argument("--q4", action="store_true",
                    help="Gemma 4, PLaMo 3 and LFM3: the MLP's projections and the head in 4 bits (q4nx, iron_q4), "
                         "the token embeddings on the host; the attention's in --attn, LFM3's convolutions' in --conv")
    ap.add_argument("--attn", choices=("q8", "q4", "bf16"), default="q8",
                    help="with --q4, the attention's projections: q8 (8.5 bits a weight), q4 or bf16")
    ap.add_argument("--conv", choices=("q8", "q4", "bf16"), default="q8",
                    help="with --q4, LFM3's short convolutions' projections (in_proj, out_proj)")
    ap.add_argument("--mlp", choices=("q4", "q6", "q8"), default="q4",
                    help="with --q4, Qwen3.5's MLP: q4nx, or q6 (6.5 bits a weight) for the small models (0.8B, "
                         "2B), whose 4-bit MLP is most of the error")
    ap.add_argument("--head", choices=("q4", "q6", "q8", "bf16"), default="q4",
                    help="with --q4, Qwen3.5's head (tied to the embeddings)")
    ap.add_argument("--q4-codes", type=Path, default=None,
                    help="with --q4, the codes q4_gptq.py chose against the activations (GPTQ): its weights take "
                         "them instead of rounding to the nearest level, the others are rounded")
    ap.add_argument("--device-ple", action="store_true",
                    help="Gemma 4: keep the per-layer token table on the device (IRON's own gathers) instead of "
                         "on the host (iron_hostple: 4.4 GiB less pinned memory for E2B)")
    ap.add_argument("--verify", type=int, default=0, metavar="ROWS",
                    help="add the version checking drafts: ROWS tokens (the next and the drafts) a call, "
                         "every row's logits out (iron_verify.py)")
    ap.add_argument("--mha-pipelines", type=int, default=None,
                    help="the pipelines of a prompt chunk's attention (MHA's num_pipelines; default 8, 1 for "
                         "qwen35moe, not measured with more; 1 is IRON's default without a profile, which left a "
                         "chunk's attention on one pipeline)")
    ap.add_argument("--command-per-step", action="store_true",
                    help="for measuring: the prompt chunk a command per step (HWNPU_IRON_TIMING times each), "
                         "the steps' operators by command in OUT/steps.json")
    ap.add_argument("--scan-window", action="store_true",
                    help="IRON's sliding-window attention, which reads the keys from position 0 and masks "
                         "those out of the window (by default it reads only the window's: iron_window)")
    a = ap.parse_args(argv)
    bind_device(a.device, a.cols)
    import importlib

    if not a.scan_window:
        import iron_window

        iron_window.install()

    if a.model.startswith("gemma4") and not a.device_ple:
        import iron_hostple

        if a.model.endswith("full"):
            iron_hostple.install(q4=a.q4, attn=a.attn)
        else:
            import iron_gemma4

            iron_hostple.install(q4=a.q4, attn=a.attn, words=iron_gemma4.SCRATCHPAD_WORDS)
            iron_gemma4.install_tail_boundary()
    elif a.model == "plamo3" and a.q4:
        import iron_plamo3

        iron_plamo3.install(attn=a.attn)
    elif a.model.startswith("lfm3") and a.q4:
        import iron_lfm3

        iron_lfm3.install_q4(attn=a.attn, conv=a.conv)
    elif a.model.startswith("qwen35") and a.q4:
        import iron_qwen35

        iron_qwen35.install(attn=a.attn, mlp=a.mlp, head=a.head)
    elif a.model == "gptoss":
        import iron_gptoss

        # the experts are mx4 (the checkpoint's MXFP4) whatever --q4 says
        iron_gptoss.install(attn=a.attn, head=a.head)
    elif a.q4:
        raise SystemExit(f"--q4: not built for {a.model}")
    if a.model == "qwen35":
        import iron_qwen35

        iron_qwen35.install_dense()
    if a.q4_codes:
        if not a.q4:
            raise SystemExit("--q4-codes: with --q4")
        import iron_q4

        print(f"{a.q4_codes}: codes of {iron_q4.load_codes(a.q4_codes)} weights")

    module, cls = MODELS[a.model]
    Model = getattr(importlib.import_module(module), cls)
    if a.prefill_chunk is None and a.model != "qwen35moe":
        # A chunk of 1024: short prompts as fast as 256, long ones as 2048 (LFM3: 30 tokens 536 ->
        # 409 ms against 2048). Gated DeltaNet's operator and projections run the chunk's rows
        # whatever the call's, so Qwen3.5 takes 512 (300 tokens 911 ms at 256, 664 at 512).
        a.prefill_chunk = 512 if a.model == "qwen35" else 1024
    if a.prefill_chunk:
        import dataclasses

        base = Model.default_config.__func__

        def default_config(cls, weights, _base=base, _chunk=a.prefill_chunk):
            return dataclasses.replace(_base(cls, weights), prefill_chunk=_chunk)

        Model.default_config = classmethod(default_config)
    if a.verify:
        import iron_verify

        iron_verify.install(Model, a.verify)
    if a.check_only:
        write_check(Model, a, a.out, a.check, a.steps, np.random.default_rng(1))
        return 0
    if a.command_per_step:  # last: it overrides the cuts the installs above make
        install_command_per_step(a.out)
    model = Model.from_checkpoint(a.checkpoint, random_weights=a.random_weights, max_seq_len=a.max_seq_len)
    pipelines = a.mha_pipelines if a.mha_pipelines is not None else (1 if a.model in ("qwen35moe", "gptoss") else 8)
    if pipelines > 1:
        prompt_attention_profile(model, pipelines)
    if a.q4_codes:
        import iron_q4

        used = iron_q4.CODES_USED
        print(f"--q4-codes: {used['hit']} weights took the chosen codes, {used['miss']} were rounded")
    if getattr(model, "hwnpu_kind", None) in ("lfm3audio", "lfm3omni"):  # its words by grid of frames, as Gemma 4's audio
        at = model.config.audio_token_id
        _TOWER.update(audio=model.towers[at], audio_input=model.tower_inputs[at], audio_kind="lfm3audio")
    if hasattr(model, "towers") and not getattr(model, "hwnpu_kind", None):  # Gemma 4's towers take words by grid
        it = model.config.image_token_id
        at = model.config.audio_token_id
        _TOWER.update(vision=model.towers[it], vision_input=model.tower_inputs[it],
                      audio=model.towers[at], audio_input=model.tower_inputs[at])
    c = model.text.config if hasattr(model, "towers") else model.config
    _CONFIG.update(max_seq_len=int(c.max_seq_len), prefill_chunk=int(c.prefill_chunk), vocab_size=int(c.vocab_size))
    versions = compile_versions(model, full_elf=False if a.per_step else None)
    m = export(model, versions, a.out)
    # What hwnpu-chat reads beside the model: the tokenizer and the chat format.
    for name in ("tokenizer.json", "tokenizer.jsonl", "tokenizer_config.json", "config.json", "chat_template.jinja", "generation_config.json",
                 "processor_config.json"):
        src = Path(a.checkpoint) / name
        if src.is_file():
            shutil.copyfile(src, a.out / name)
    for name, v in m["versions"].items():
        shape = f"{len(v['kernels'])} commands" if "kernels" in v else f"{len(v['steps'])} steps on {len({s['xclbin'] for s in v['steps']})} xclbins"
        print(f"{name}: {shape}, arguments {[(x['kind'], x['bytes']) for x in v['arguments']]}")
    if a.check:
        write_check(Model, a, a.out, a.check, a.steps, np.random.default_rng(1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
