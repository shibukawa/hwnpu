#!/usr/bin/env python3
"""Export a language model IRON compiles (an iron.lm.CausalLM) for hwnpu.

Writes, into a directory:

  manifest.json    the model's shape, the arena's banks, and each version (the decode
                   step, the prompt chunk): its commands, arguments, per-command
                   scratchpad words, carried values, logits and input encoders
  decode.elf       the decode step: one token in, the logits after it out
  prompt.elf       a prompt chunk of up to prefill_chunk tokens, the last one's logits out
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
    "gemma4": ("iron.applications.gemma4.model", "Gemma4"),
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


def _encoders(compiled) -> list:
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

            kind, offset, length = _place(seq, BufferRef(e.buffer))
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


def version_manifest(compiled, elf_name: str) -> dict:
    from iron.common.image.callable import FullELFCallable
    from iron.common.image.compose import Composition
    from iron.common.image.step import BufferRef

    art = compiled.artifacts
    if not art.full_elf:
        raise SystemExit("the version was packaged as an xclbin per step; hwnpu runs full ELFs only")
    rec, seq = compiled.recording, compiled.sequence
    if rec.addresses:
        raise SystemExit("words computed from buffer addresses are not supported yet")
    commands = [
        f"{FullELFCallable.device_name}:{Composition.sequence_name(k)}"
        for k in range(len(seq.packaging.commands) + 1)
    ]
    words = {w.symbol: w for w in compiled.words}
    derived = _derived(compiled)
    version = "prompt" if compiled.recording.inputs else "decode"
    table = {}
    for p in art.parameter_table:
        w = words.get(p.name)
        if w is None:
            raise SystemExit(f"parameter {p.name} is no word of the version")
        entry = dict(name=p.name, index=int(p.index), dtype=p.dtype, kind=p.kind)
        if w.form is not None:
            entry["form"] = _form(w.form)
        elif w.linear is not None:
            # ceil(v * ratio + offset), both fractions
            lin = w.linear
            entry["linear"] = dict(value=lin.value, num=lin.ratio.numerator, den=lin.ratio.denominator,
                                   off_num=lin.offset.numerator, off_den=lin.offset.denominator)
        else:
            entry.update(_derived_entry(w, derived, version))
        table[p.name] = entry
    members = _params_by_command(art.params)
    kernels = []
    for name in commands:
        names = members.get(name, list(table) if len(commands) == 1 else [])
        kernels.append(dict(name=name, parameters=[table[n] for n in names]))
    outputs = []
    for h in rec.returned:
        kind, offset, length = _place(seq, h.ref())
        outputs.append(dict(name=h.name, shape=[int(d) for d in h.shape], dtype=str(np.dtype(seq.buffer_dtype(h.ref()))),
                            argument=kind, offset=offset, bytes=length))
    inputs = []
    for h in rec.inputs:
        kind, offset, length = _place(seq, h.ref())
        inputs.append(dict(name=h.name, shape=[int(d) for d in h.shape], dtype=str(np.dtype(h.dtype)),
                           argument=kind, offset=offset, bytes=length))
    buffers = {}
    for root in seq.layout:
        kind, offset, length = _place(seq, BufferRef(root))
        buffers[root] = dict(argument=kind, offset=offset, bytes=length, dtype=str(np.dtype(seq.buffer_dtype(BufferRef(root)))))
    return dict(
        elf=elf_name,
        kernels=kernels,
        buffers=buffers,
        arguments=[dict(index=i, kind=k, bytes=int(n)) for i, (k, n) in enumerate(seq.buffer_sizes.arguments().items())],
        values=[v.name for v in rec.values],
        carried=_carried(compiled),
        outputs=outputs,
        inputs=inputs,
        encoders=_encoders(compiled),
        plan=compiled.plan.report(rec.name),
    )


def compile_versions(model):
    """The decode step and the prompt chunk, compiled into one arena as load_versions does."""
    import aie.utils as aie_utils

    dev = aie_utils.get_current_device() if hasattr(aie_utils, "get_current_device") else None
    versions, decode = {}, None
    for shapes in model.shapes():
        prompt = "ids" in shapes
        v = model.compile(dev=dev, feeds=decode if prompt else None, full_elf=None, **shapes)
        if not shapes:
            decode = v
        versions["prompt" if prompt else "decode"] = v
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
        words = {w.symbol: w for w in v.words}
        res[name] = []
        for values in calls[name]:
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


def export(model, versions, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    c = model.config
    vm = {}
    for name, v in versions.items():
        shutil.copyfile(v.artifacts.image, out / f"{name}.elf")
        vm[name] = version_manifest(v, f"{name}.elf")
        for i, e in enumerate(vm[name]["encoders"]):
            fn = f"gather_{name}_{i}.bin"
            e.pop("template").tofile(out / fn)
            e["template"] = fn
    # The arena's banks: every argument that is neither the input, the output nor the
    # feedback, as big as the largest version needs (all versions are placed in one arena).
    banks = {}
    for m in vm.values():
        for a in m["arguments"]:
            if a["kind"] not in ("input", "output", "feedback"):
                banks[a["kind"]] = max(banks.get(a["kind"], 0), a["bytes"])
    images = write_banks(versions, banks, out)
    for n in range(1, 3 * c.prefill_chunk, max(1, c.prefill_chunk // 3)):
        if model.rope_row(n, 0) != 0:
            raise SystemExit("rope_row is not the chunk's start: not supported yet")
    m = dict(
        version=MANIFEST_VERSION,
        model=type(model).__name__,
        config=dict(
            vocab_size=int(c.vocab_size), hidden_size=int(c.hidden_size), max_seq_len=int(c.max_seq_len),
            prefill_chunk=int(c.prefill_chunk), num_hidden_layers=int(c.num_hidden_layers),
        ),
        rope_start="start",
        # Whether the state restarts anywhere: no ring caches, no recurrent layers
        # (CausalLM.recurrent). Only then is a position all a sequence needs to go back.
        restartable=not model.recurrent,
        banks={k: dict(bytes=int(n), image=images.get(k)) for k, n in banks.items()},
        versions=vm,
    )
    (out / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    write_words(versions, m, out)
    return m


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("checkpoint", help="the Hugging Face directory (config.json, weights, tokenizer)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--prefill-chunk", type=int, default=None, help="tokens of a prompt chunk (IRON's default 2048)")
    ap.add_argument("--random-weights", type=int, default=None, help="weights drawn at this seed")
    ap.add_argument("--device", default="npu2")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--check", type=int, default=0, metavar="TOKENS", help="write a check of a prompt of TOKENS tokens")
    ap.add_argument("--steps", type=int, default=3, help="greedy steps of the check")
    ap.add_argument("--check-only", action="store_true", help="write the check alone, into an export made before")
    a = ap.parse_args(argv)
    bind_device(a.device, a.cols)
    import importlib

    module, cls = MODELS[a.model]
    Model = getattr(importlib.import_module(module), cls)
    if a.prefill_chunk:
        import dataclasses

        base = Model.default_config.__func__

        def default_config(cls, weights, _base=base, _chunk=a.prefill_chunk):
            return dataclasses.replace(_base(cls, weights), prefill_chunk=_chunk)

        Model.default_config = classmethod(default_config)
    if a.check_only:
        write_check(Model, a, a.out, a.check, a.steps, np.random.default_rng(1))
        return 0
    model = Model.from_checkpoint(a.checkpoint, random_weights=a.random_weights, max_seq_len=a.max_seq_len)
    c = model.config
    _CONFIG.update(max_seq_len=int(c.max_seq_len), prefill_chunk=int(c.prefill_chunk), vocab_size=int(c.vocab_size))
    versions = compile_versions(model)
    m = export(model, versions, a.out)
    # What hwnpu-chat reads beside the model: the tokenizer and the chat format.
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json", "chat_template.jinja", "generation_config.json"):
        src = Path(a.checkpoint) / name
        if src.is_file():
            shutil.copyfile(src, a.out / name)
    for name, v in m["versions"].items():
        print(f"{name}: {len(v['kernels'])} commands, arguments {[(x['kind'], x['bytes']) for x in v['arguments']]}")
    if a.check:
        write_check(Model, a, a.out, a.check, a.steps, np.random.default_rng(1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
