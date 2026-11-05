#!/bin/bash
# bench.sh: build an IRON design for npu2, run it with ironbench and check the result.
#
#   bench.sh gemm NAME DTIN DTOUT M K N m k n COLS EMU [RUNS]
#   bench.sh bfp  NAME M K N m k n COLS [RUNS]
#   bench.sh memcpy NAME N_INT32 [RUNS]
#   bench.sh dma NAME read|write N_INT32 COLS CHANS LINE [RUNS]
#   bench.sh q4deq NAME TILES COLS CHANS [RUNS]
#   bench.sh q4gemv NAME M K COLS ROWS [RUNS]
#   bench.sh lfm3conv NAME MODEL_DIR REF_PREFIX [RUNS]   (layer 0 of LFM3-1.7B, one decode step)
#
# Prints one JSON line. Needs (see README.md): an mlir-aie venv on PATH, PEANO_INSTALL_DIR,
# MLIR_AIE_SRC (an mlir-aie checkout of the same version), AIE_XCLBINUTIL (xclbinutil or
# hwnpu-xclbin), IRONBENCH (the ironbench binary) and OUT (a directory for builds).
set -u
here=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$here
kind=$1 name=$2
shift 2
out=$OUT/${kind}_$name
fail() { echo "{\"name\":\"$name\",\"error\":\"$1\"}"; exit 1; }

case $kind in
gemm)
	[ -f "$out/final.xclbin" ] || python3 "$here/gemm.py" build "$out" "${@:1:10}" > "$out.log" 2>&1 || fail "build failed, see $out.log"
	runs=${11:-20}
	;;
bfp)
	[ -f "$out/final.xclbin" ] || python3 "$here/gemm_bfp.py" build "$out" "${@:1:7}" > "$out.log" 2>&1 || fail "build failed, see $out.log"
	runs=${8:-10}
	;;
memcpy)
	n=$1
	[ -f "$out/final.xclbin" ] || python3 "$here/mk_memcpy.py" "$MLIR_AIE_SRC/programming_examples/getting_started/00_memcpy/memcpy.py" "$out" "$n" > "$out.log" 2>&1 || fail "build failed, see $out.log"
	head -c $((n * 4)) /dev/urandom > "$out/A.bin"
	r=$("$IRONBENCH" -xclbin "$out/final.xclbin" -insts "$out/insts.bin" -in "$out/A.bin" -out $((n * 4)) -save "$out/C.bin" -runs "${2:-20}" 2>&1) || fail "$r"
	cmp -s "$out/A.bin" "$out/C.bin" && ok=true || ok=false
	echo "{\"name\":\"$name\",\"kind\":\"memcpy\",\"bytes\":$((n * 4)),\"run\":$r,\"check\":{\"ok\":$ok}}"
	exit 0
	;;
dma)
	mode=$1 n=$2
	[ -f "$out/final.xclbin" ] || python3 "$here/dma_bw.py" "$out" "$mode" "$n" "$3" "$4" "$5" > "$out.log" 2>&1 || fail "build failed, see $out.log"
	if [ "$mode" = read ]; then
		[ -f "$out/A.bin" ] || head -c $((n * 4)) /dev/urandom > "$out/A.bin"
		r=$("$IRONBENCH" -xclbin "$out/final.xclbin" -insts "$out/insts.bin" -in "$out/A.bin" -runs "${6:-20}" 2>&1) || fail "$r"
	else
		r=$("$IRONBENCH" -xclbin "$out/final.xclbin" -insts "$out/insts.bin" -out $((n * 4)) -runs "${6:-20}" 2>&1) || fail "$r"
	fi
	echo "{\"name\":\"$name\",\"kind\":\"dma\",\"mode\":\"$mode\",\"bytes\":$((n * 4)),\"cols\":$3,\"chans\":$4,\"run\":$r}"
	exit 0
	;;
q4deq)
	[ -f "$out/final.xclbin" ] || python3 "$here/q4deq.py" build "$out" "$1" "$2" "$3" > "$out.log" 2>&1 || fail "build failed, see $out.log"
	csize=$(python3 "$here/q4deq.py" data "$out") || fail "data failed"
	r=$("$IRONBENCH" -xclbin "$out/final.xclbin" -insts "$out/insts.bin" -in "$out/A.bin" -out "$csize" -save "$out/C.bin" -runs "${4:-20}" 2>&1) || fail "$r"
	c=$(python3 "$here/q4deq.py" check "$out" 2>&1 | tail -1)
	echo "{\"name\":\"$name\",\"kind\":\"q4deq\",\"params\":$(cat "$out/params.json"),\"run\":$r,\"check\":$c}"
	exit 0
	;;
q4gemv)
	[ -f "$out/final.xclbin" ] || python3 "$here/q4gemv.py" build "$out" "$1" "$2" "$3" "$4" > "$out.log" 2>&1 || fail "build failed, see $out.log"
	ys=$(python3 "$here/q4gemv.py" data "$out") || fail "data failed"
	r=$("$IRONBENCH" -xclbin "$out/final.xclbin" -insts "$out/insts.bin" -in "$out/W.bin,$out/x.bin" -out "$ys" -save "$out/y.bin" -runs "${5:-20}" 2>&1) || fail "$r"
	c=$(python3 "$here/q4gemv.py" check "$out" 2>&1 | tail -1)
	echo "{\"name\":\"$name\",\"kind\":\"q4gemv\",\"params\":$(cat "$out/params.json"),\"run\":$r,\"check\":$c}"
	exit 0
	;;
lfm3conv)
	[ -f "$out/final.xclbin" ] || python3 "$here/lfm3conv.py" build "$out" > "$out.log" 2>&1 || fail "build failed, see $out.log"
	ss=$(python3 "$here/lfm3conv.py" data "$out" "$1" "$2") || fail "data failed"
	r=$("$IRONBENCH" -xclbin "$out/final.xclbin" -insts "$out/insts.bin" -in "$out/x.bin,$out/w.bin,$out/state.bin,$out/norms.bin" -out "$ss" -save "$out/scr.bin" -runs "${3:-20}" 2>&1) || fail "$r"
	c=$(python3 "$here/lfm3conv.py" check "$out" "$1" 2>&1 | tail -1)
	echo "{\"name\":\"$name\",\"kind\":\"lfm3conv\",\"run\":$r,\"check\":$c}"
	exit 0
	;;
*) fail "unknown kind $kind" ;;
esac

script=$here/gemm.py
[ "$kind" = bfp ] && script=$here/gemm_bfp.py
csize=$(python3 "$script" data "$out") || fail "data failed"
r=$("$IRONBENCH" -xclbin "$out/final.xclbin" -insts "$out/insts.bin" -in "$out/A.bin,$out/B.bin" -out "$csize" -save "$out/C.bin" -runs "$runs" 2>&1) || fail "$r"
c=$(python3 "$script" check "$out" 2>&1 | tail -1)
echo "{\"name\":\"$name\",\"kind\":\"$kind\",\"params\":$(cat "$out/params.json"),\"run\":$r,\"check\":$c}"
