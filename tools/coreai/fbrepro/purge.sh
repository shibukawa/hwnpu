#!/bin/bash
# purge.sh PROGRAM.aimodel[#FUNCTION] [KEEP_GB]: what happens to a compiled program when the disk is
# nearly full (docs/apple-feedback/purged-programs-silent-recompile.md).
#
#   1. load the program (compiled on the device; Core AI's entry and aned's program made)
#   2. fill the disk until KEEP_GB (default 11) are left, hold it 5 minutes, free it
#   3. load it again with the same binary: Core AI's entry is still there; aned logs
#      "Model load failed ... isPreCompiledModel=1" and the program compiles again
#
# It fills the disk on purpose: run it on a test machine, with nothing else writing.
set -u
prog=$1
keep=${2:-11}
aicheck=${AICHECK:-./aicheck}
dir=$(mktemp -d /tmp/fbrepro-purge.XXXXXX)
cp "$aicheck" "$dir/aicheck"
codesign -s - -f -i "hwnpu.fbrepro.purge.$(date +%s)" "$dir/aicheck" 2>/dev/null
load() {
  local t0=$(date +%s)
  HWNPU_DAEMON=off "$dir/aicheck" run -calls 1 "$prog" > "$dir/$1.log" 2>&1
  echo "$1: $(( $(date +%s) - t0 ))s  $(tail -1 "$dir/$1.log")"
}
start=$(date "+%Y-%m-%d %H:%M:%S")
load first
load second   # cached: should be fast
free=$(df -k "$HOME" | awk 'NR==2{print int($4/1048576)}')
fill=$(( free - keep ))
if [ "$fill" -gt 0 ]; then
  echo "filling ${fill} GB (free ${free} GB -> ${keep} GB) for 5 minutes"
  mkfile -n "${fill}g" "$dir/fill" 2>/dev/null || dd if=/dev/zero of="$dir/fill" bs=1m count=$(( fill * 1024 )) 2>/dev/null
  sleep 300
  rm -f "$dir/fill"
fi
load third    # after the purge
/usr/bin/log show --start "$start" --predicate '(process == "aned" AND eventMessage CONTAINS "isPreCompiledModel=1") OR (process == "deleted" AND eventMessage CONTAINS "deleted_helper took" AND NOT eventMessage CONTAINS "purged 0 bytes")' --style compact > "$dir/log.txt" 2>&1
echo "purges: $(grep -c "deleted_helper" "$dir/log.txt"), aned load failures: $(grep -c "Model load failed" "$dir/log.txt") (log in $dir/log.txt)"
