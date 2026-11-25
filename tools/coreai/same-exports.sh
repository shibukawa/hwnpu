#!/bin/bash
# same-exports.sh BASE MODEL[/coreai]...: makes the models' Core AI exports with the bonsaiai-dev of
# BASE (a commit) and with this checkout's, each into a cache of its own, and says whether they
# are the same: the same names, the same bytes but metadata.json's creationDate (and the caches'
# own paths in a JSON file). When they are,
# a change of the code (apple.code) or of apple.convert needs no new version: go test
# ./internal/recipe -run TestAppleVersions -update (docs/model-recipes.md 2).
#
# The checkpoints are taken from this machine's cache (original/, linked; what is missing comes
# from Hugging Face into it); no export is fetched or put in the sources. The programs are not
# prebuilt. WORK (default a new temporary directory)
# keeps the caches: 2 x the exports' size on the disk; they are removed when the two are the same.
# CHIP (default M1) is the chip whose exports are made (bonsaiai-dev models package --chip): M2 for
# the other chips' where the M1 has an export of its own (a recipe's rewrite for h13g).
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,10p' "$0"; exit 2; }
base=$1; shift
root=$(git rev-parse --show-toplevel)
work=${WORK:-$(mktemp -d)}
mine=${BONSAIAI_CACHE:-$HOME/Library/Caches/bonsaiai}
models=()
for m in "$@"; do models+=("${m%/coreai}/coreai"); done

mkdir -p "$work/bin-base" "$work/bin-new"
git -C "$root" worktree add -q --detach "$work/src-base" "$base"
trap 'git -C "$root" worktree remove --force "$work/src-base" 2>/dev/null || true' EXIT
(cd "$work/src-base" && go build -o "$work/bin-base/" ./cmd/bonsaiai-dev)
(cd "$root" && go build -o "$work/bin-new/" ./cmd/bonsaiai-dev)

unset BONSAIAI_MODEL_SOURCE
# originals missing here come from Hugging Face into this machine's original/ (linked)
for side in base new; do
	export BONSAIAI_CACHE="$work/cache-$side"
	mkdir -p "$BONSAIAI_CACHE"
	ln -sfn "$mine/original" "$BONSAIAI_CACHE/original"
	echo "== $side ($([ $side = base ] && echo "$base" || echo this checkout)): ${models[*]}"
	"$work/bin-$side/bonsaiai-dev" models package "${models[@]}" --chip "${CHIP:-M1}" --memory 16
done

# the same exports, file for file, but metadata.json's creationDate
same=1
a="$work/cache-base/models" b="$work/cache-new/models"
diff <(cd "$a" && find . -name '*.coreai' -prune -print | sort) <(cd "$b" && find . -name '*.coreai' -prune -print | sort) || same=0
while IFS= read -r f; do
	if [ ! -f "$b/$f" ]; then
		echo "only in $base: $f"; same=0
	elif [ "$(basename "$f")" = metadata.json ]; then
		cmp -s <(grep -v creationDate "$a/$f") <(grep -v creationDate "$b/$f") || { echo "differs: $f"; same=0; }
	elif [ "${f##*.}" = json ]; then
		# the caches' own paths (DA3's and MoGe's checkpoint) aside
		cmp -s <(sed "s#$work/cache-base#CACHE#g" "$a/$f") <(sed "s#$work/cache-new#CACHE#g" "$b/$f") || { echo "differs: $f"; same=0; }
	elif ! cmp -s "$a/$f" "$b/$f"; then
		echo "differs: $f"; same=0
	fi
done < <(cd "$a" && find . -path '*.coreai/*' -type f ! -name '*.done' ! -name entry.json | sort)
while IFS= read -r f; do
	[ -f "$a/$f" ] || { echo "only in this checkout: $f"; same=0; }
done < <(cd "$b" && find . -path '*.coreai/*' -type f ! -name '*.done' ! -name entry.json | sort)
if [ $same = 1 ]; then
	echo "the same: $(cd "$a" && find . -name '*.coreai' -prune | wc -l | tr -d ' ') exports"
	rm -rf "$work/cache-base" "$work/cache-new"
else
	echo "not the same (the caches are in $work)"
	exit 1
fi
