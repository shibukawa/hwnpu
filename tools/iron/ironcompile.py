"""Compile an @iron.jit design from a file for npu2: load(path) then compile(fn, out, **kwargs)."""
import importlib.util
import os
import sys

from aie.iron.device import NPU2
from aie.utils.hostruntime import set_current_device


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(mod)
    return mod


def compile(fn, out, **kwargs):
    set_current_device(NPU2())
    os.makedirs(out, exist_ok=True)
    os.chdir(out)
    fn.specialize(**kwargs).compile(
        xclbin_path=os.path.join(out, "final.xclbin"),
        inst_path=os.path.join(out, "insts.bin"),
    )
    print("built", out)
