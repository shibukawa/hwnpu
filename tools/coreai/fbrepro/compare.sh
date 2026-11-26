#!/bin/bash
# compare.sh [-a ARCHS] PROGRAM... : compile each program offline with coreai-build of each Xcode in
# XCODES (default: every /Applications/Xcode*.app) for each architecture, and print a row each:
# the Xcode, coreai-build's version, the architecture, the program, the exit status, whether it is
# all on the Neural Engine (mps.fullyPlacedOnANE), the seconds, the disk it took at most (GB) and a
# hash of what it wrote for the Neural Engine (the .aimodelc without its weights and dates: whether
# a new toolchain writes other code for the same program).
#
#   XCODES="/Applications/Xcode.app /Applications/Xcode-beta.app" compare.sh -a h13g,h14g,h15g sin.aimodel
set -u
archs="h13g,h14g,h15g,h16g,h17g,h18g"
if [ "${1:-}" = "-a" ]; then archs=$2; shift 2; fi
xcodes=${XCODES:-$(ls -d /Applications/Xcode*.app)}
freek() { df -k "$HOME" | awk 'NR==2{print $4}'; }
printf "xcode\tcoreai-build\tarch\tprogram\texit\tane\tseconds\tdisk_gb\tcode_hash\n"
for x in $xcodes; do
  export DEVELOPER_DIR=$x/Contents/Developer
  ver=$(xcrun coreai-build --version 2>/dev/null | awk '{print $NF}')
  [ -n "$ver" ] || { echo "$(basename $x): no coreai-build (xcodebuild -downloadComponent MetalToolchain)" >&2; continue; }
  for a in ${archs//,/ }; do
    for p in "$@"; do
      out=$(mktemp -d /tmp/fbrepro-cmp.XXXXXX)
      start=$(freek); low=$start
      ( while :; do f=$(freek); echo "$f"; sleep 1; done ) > "$out.disk" & s=$!
      t0=$(date +%s)
      xcrun coreai-build compile "$p" --platform macOS --architecture "$a" --preferred-compute neural-engine --output "$out/o" > "$out.log" 2>&1
      rc=$?; secs=$(( $(date +%s) - t0 ))
      kill $s 2>/dev/null; wait $s 2>/dev/null
      low=$(sort -n "$out.disk" | head -1); [ -n "$low" ] || low=$start
      m=$(find "$out/o" -name manifest.plist 2>/dev/null | head -1)
      if [ -n "$m" ] && plutil -p "$m" | grep -q fullyPlacedOnANE; then ane=yes; else ane=no; fi
      # the compiled code: every file but the weights (resources.bin), dates and statistics
      code=$(cd "$out/o" 2>/dev/null && find . -type f ! -name "resources.bin" ! -name "*.plist" ! -name "stats.json" ! -name "metadata.json" -print0 | sort -z | xargs -0 cat 2>/dev/null | shasum | cut -c1-12)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.1f\t%s\n" "$(basename $x .app)" "$ver" "$a" "$(basename $p)" "$rc" "$ane" "$secs" "$(echo "($start - $low)/1048576" | bc -l)" "$code"
      rm -rf "$out" "$out.disk" "$out.log"
    done
  done
done
