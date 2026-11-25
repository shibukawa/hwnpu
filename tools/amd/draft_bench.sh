#!/bin/bash
# draft_bench.sh OUT MODEL...: what checking drafts brings on AMD's NPU (a model exported with
# --verify: Qwen3.5-0.8B, Qwen3.5-2B). Runs bonsaiai-chat on each prompt of draft_prompts with no
# drafts (BONSAIAI_DEV=iron.drafts=0), every draft checked (BONSAIAI_DEV=draft.policy=0) and the drafts that
# pay (the default, chat.draftPolicy), in turn, ROUNDS (2) times, N (300) tokens at temperature 0,
# and keeps each run's answer, messages and calls (BONSAIAI_DEV=draft.trace) in OUT. A line a run; then
# draft_analyze.py OUT reads them. Run from the repository, with nothing else on the NPU: the
# calls' times in the traces show a run that was disturbed (docs/design.md 16.284).
set -u
out=${1:?out}
shift
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
bin=$(mktemp -d)/bonsaiai-chat
go build -o "$bin" ./cmd/bonsaiai-chat || exit 1
for m in "$@"; do
  for r in $(seq "${ROUNDS:-2}"); do
    for p in "$here"/draft_prompts/*.txt; do
      name=$(basename "$p" .txt)
      for mode in none always policy; do
        f="$out/$m.$name.$mode.$r"
        case $mode in
          none) env=(BONSAIAI_DEV=iron.drafts=0) ;;
          always) env=(BONSAIAI_DEV=draft.policy=0) ;;
          policy) env=(BONSAIAI_DEV=draft.policy=1) ;;
        esac
        rm -f "$f.trace"
        env "${env[@]}" BONSAIAI_DEV=draft.trace="$f.trace" "$bin" chat -m "$m" -n "${N:-300}" --temp 0 --no-think --prompt-cache off "$(cat "$p")" >"$f.out" 2>"$f.err" </dev/null
        echo "$m $name $mode $r $(grep -a -o '\[prompt.*' "$f.err" | tail -1)" | tee -a "$out/runs.txt"
      done
    done
  done
done
