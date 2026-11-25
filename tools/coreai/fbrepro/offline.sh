#!/bin/bash
# offline.sh ARCH PROGRAM.aimodel... : coreai-build each for ARCH; whether it is all on the Neural Engine
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
arch=$1; shift
for p in "$@"; do
  out=$(mktemp -d)
  start=$(date "+%Y-%m-%d %H:%M:%S")
  xcrun coreai-build compile "$p" --platform macOS --architecture "$arch" --preferred-compute neural-engine --output "$out" > "$out.log" 2>&1
  rc=$?
  m=$(find "$out" -name manifest.plist | head -1)
  if [ -n "$m" ] && plutil -p "$m" | grep -q fullyPlacedOnANE; then ane="ANE"; else ane="NOT on ANE"; fi
  why=$(/usr/bin/log show --start "$start" --predicate 'process == "coreai-build" AND (messageType == 16 OR eventMessage CONTAINS "Unsupported" OR eventMessage CONTAINS "NEFUSED")' --style compact 2>/dev/null | grep -oE "Unsupported NE layer[^;]*|NEFUSED_EW[^,;]*|failed[^;]{0,80}" | sort -u | head -3 | tr '\n' ';')
  printf "%-6s %-26s exit=%s %-11s %s\n" "$arch" "$(basename $p)" "$rc" "$ane" "$why"
  rm -rf "$out" "$out.log"
done
