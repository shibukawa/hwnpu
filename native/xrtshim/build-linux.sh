#!/bin/sh
# Builds libxrtshim.so against libxrt_coreutil, with the XRT headers in include/ (Linux, for
# testing the shim: hwnpu talks to the amdxdna driver itself on Linux). XRT_ROOT is where XRT
# is installed (default /usr; an unpacked package works). uuid/uuid.h comes from uuid-dev.
set -e
cd "$(dirname "$0")"
XRT_ROOT=${XRT_ROOT:-/usr}
LIBDIR=$XRT_ROOT/lib/x86_64-linux-gnu
[ -d "$LIBDIR" ] || LIBDIR=$XRT_ROOT/lib
OUT=${OUT:-libxrtshim.so}
${CXX:-g++} -std=c++17 -O2 -fPIC -shared -fvisibility=hidden -Wall -Wextra \
	-isystem include -isystem "$XRT_ROOT/include" xrtshim.cpp -o "$OUT" \
	-L"$LIBDIR" -l:libxrt_coreutil.so.2 -Wl,-rpath,"$LIBDIR"
echo "$OUT"
