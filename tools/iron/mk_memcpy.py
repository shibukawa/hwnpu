"""mk_memcpy.py SRC OUT N: the tuned 8-column memcpy (getting_started/00_memcpy) moving N int32."""
import sys

import numpy as np
from ironcompile import compile, load

src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
mod = load(src, "memcpy00")
ty = np.ndarray[(n,), np.dtype[np.int32]]
compile(mod.my_memcpy, out, input0=ty, output=ty, size=n, xfr_dtype=np.int32)
