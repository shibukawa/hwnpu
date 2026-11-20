#!/bin/bash
# collect.sh MODEL OUT [TOKENS]: runs hwnpu-chat on every prompt of prompts.tsv (a category and a
# prompt a line, tab between) and adds the counts of the experts each layer ran, the prompt's and
# the answer's, to OUT/CATEGORY.bin (experts.Usage). analyze.py reads them. gpt-oss: on a Mac its
# experts on the host, on AMD's NPU an IRON export (MODEL a directory; iron.Model.CountExperts).
# Qwen3.6-35B-A3B's experts run on the host too. The application's own counts are left alone
# (--experts-key off). Run from the repository; about 30 s a prompt for gpt-oss-20b on an M3,
# 26 s on a Ryzen AI 9 365 (110 prompts: 48 min) and 45 s for Qwen3.6-35B-A3B. The line of a
# prompt's times is taken wherever it is in a line: a model cut off while it thinks ends its text
# without a newline.
set -u
model=${1:?model}
out=${2:?out}
tokens=${3:-400}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
bin=$(mktemp -d)/hwnpu-chat
go build -o "$bin" ./cmd/hwnpu-chat || exit 1
i=0
while IFS=$'\t' read -r cat prompt; do
  [ -z "$cat" ] && continue
  case $cat in \#*) continue ;; esac
  i=$((i + 1))
  line=$(HWNPU_MOE_DEBUG="usage=$out/$cat.bin" "$bin" chat -m "$model" --experts-key off -n "$tokens" --seed "$i" --prompt-cache off "$prompt" 2>&1 >/dev/null </dev/null | grep -a -o -E 'hwnpu-chat: error.*|\[prompt [0-9]+ tokens.*' | tail -1)
  echo "$i $cat $line"
done < "$here/prompts.tsv"
