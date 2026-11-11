"""Load a Hugging Face model whose float32 weights do not fit in memory: built on the meta device,
its small tensors read at once and each of its repeated layers read just before it runs and dropped
after (forward hooks). One layer's weights are held at a time, so a 7B model in float32 runs in a
few GB, reading its checkpoint once a forward pass.

    model = streamed(lambda: Cls(config), ckpt_dir, ["transformer_blocks"], torch.float32)

ckpt_dir holds model.safetensors, or the shards of a *.safetensors.index.json. Tensors under
``skip`` prefixes (a vision tower the text path never runs) are left on the meta device. With
``stats`` each streamed layer's largest output magnitude is appended to it (name, value), for
choosing fp16 scales.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open


class Checkpoint:
    def __init__(self, d: Path):
        idx = sorted(d.glob("*.safetensors.index.json"))
        if idx:
            wm = json.loads(idx[0].read_text())["weight_map"]
            files = sorted(set(wm.values()))
        else:
            files = [p.name for p in sorted(d.glob("*.safetensors"))]
            wm = None
        self.handles = {f: safe_open(str(d / f), "pt") for f in files}
        if wm is None:
            wm = {k: f for f, h in self.handles.items() for k in h.keys()}
        self.index = wm

    def get(self, name: str) -> torch.Tensor:
        return self.handles[self.index[name]].get_tensor(name)

    def names(self):
        return self.index.keys()


def _set(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    *path, leaf = name.split(".")
    m = module
    for p in path:
        m = getattr(m, p)
    old = getattr(m, leaf)
    if isinstance(old, torch.nn.Parameter):
        setattr(m, leaf, torch.nn.Parameter(value, requires_grad=False))
    else:
        setattr(m, leaf, value)


def streamed(build, ckpt_dir: Path, layer_lists: list, dtype, rename=lambda n: n, skip=(), stats=None):
    """The model ``build()`` makes (on the meta device), its weights from ckpt_dir (checkpoint names
    through ``rename`` give the module's names), the layers of each ModuleList named in
    layer_lists streamed."""
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = build()
    model.eval()
    ck = Checkpoint(ckpt_dir)
    by_module = {rename(n): n for n in ck.names()}
    streamed_prefixes = []
    for ln in layer_lists:
        lst = model.get_submodule(ln)
        for i, layer in enumerate(lst):
            prefix = f"{ln}.{i}."
            streamed_prefixes.append(prefix)
            names = [k for k in by_module if k.startswith(prefix)]

            def pre(mod, args, kwargs, prefix=prefix, names=names):
                for k in names:
                    _set(mod, k[len(prefix):], ck.get(by_module[k]).to(dtype))
                return None

            def post(mod, args, kwargs, out, prefix=prefix):
                if stats is not None:
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    stats.append((prefix.rstrip("."), float(o.detach().abs().max())))
                mod.to_empty(device="meta")
                return out

            layer.register_forward_pre_hook(pre, with_kwargs=True)
            layer.register_forward_hook(post, with_kwargs=True)
    for k, ckname in by_module.items():
        if any(k.startswith(p) for p in streamed_prefixes) or any(k.startswith(s) for s in skip):
            continue
        try:
            model.get_parameter(k)
        except AttributeError:
            try:
                model.get_buffer(k)
            except AttributeError:
                continue
        _set(model, k, ck.get(ckname).to(dtype))
    # buffers (rotary tables) are made on the CPU by the modules themselves (include_buffers=False)
    return model
