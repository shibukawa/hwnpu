#!/bin/bash
# load.sh PROGRAM.aimodel[#FUNCTION]: load a program on the Neural Engine with no Core AI or aned
# cache for it (a copy of aicheck signed with a new identifier), timing the load and sampling the
# free disk every 5 s. aicheck is built from tools/coreai/aicheck (AICHECK, default ./aicheck).
set -u
prog=$1
aicheck=${AICHECK:-./aicheck}
dir=$(mktemp -d /tmp/fbrepro-load.XXXXXX)
cp "$aicheck" "$dir/aicheck"
codesign -s - -f -i "bonsaiai.fbrepro.$(date +%s).$$" "$dir/aicheck" 2>/dev/null
( while :; do echo "$(date +%T) $(df -k "$HOME" | awk 'NR==2{printf "%.1f", $4/1048576}')"; sleep 5; done ) > "$dir/disk.log" &
sampler=$!
t0=$(date +%s)
BONSAIAI_DAEMON=off "$dir/aicheck" run -calls 1 "$prog" > "$dir/run.log" 2>&1
rc=$?
kill $sampler
echo "rc=$rc wall=$(( $(date +%s) - t0 ))s"
tail -1 "$dir/run.log"
echo "free disk GB: start $(head -1 "$dir/disk.log" | cut -d' ' -f2), lowest $(sort -k2 -n "$dir/disk.log" | head -1 | cut -d' ' -f2), end $(tail -1 "$dir/disk.log" | cut -d' ' -f2) (samples in $dir/disk.log)"
