// Command draftvocab ranks a tokenizer's tokens by how often they occur in some texts, for the
// drafts of speculative decoding from a reduced output layer (FR-Spec): the multi-token
// prediction layer of Qwen3.5 guesses among the most frequent tokens alone (qwen35-mtp's
// draft-vocab), which makes its output layer, most of its weights, several times smaller.
//
//	go run ./tools/draftvocab -tokenizer ~/Library/Caches/hwnpu/models/Qwen3.5-2B/tokenizer.json \
//	    -o internal/llm/qwen35/draft_vocab.bin -n 65536 ~/models/calib/wiki40b-ja-calib.txt \
//	    ~/models/calib/wikitext2-train.txt $(git ls-files '*.go' '*.md' '*.py')
//
// The output is the ids of the n most frequent tokens, most frequent first, as little-endian
// uint32. The tokenizer's control tokens (the chat format's) come first. Each file counts up to
// -max bytes, so that one large text does not decide the order alone.
package main

import (
	"bufio"
	"cmp"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

func main() {
	tok := flag.String("tokenizer", "", "tokenizer.json")
	out := flag.String("o", "", "output file")
	n := flag.Int("n", 65536, "tokens to keep")
	maxBytes := flag.Int64("max", 24<<20, "bytes read of each file at most")
	flag.Parse()
	if *tok == "" || *out == "" || flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: draftvocab -tokenizer tokenizer.json -o out.bin [-n N] FILE...")
		os.Exit(2)
	}
	t, err := tokenizer.Load(*tok)
	if err != nil {
		fail(err)
	}
	counts := make([]int64, t.VocabSize())
	var total int64
	for _, path := range flag.Args() {
		f, err := os.Open(path)
		if err != nil {
			fail(err)
		}
		// a line (a paragraph of the corpora, a line of code) at a time, newlines kept
		r := bufio.NewReader(io.LimitReader(f, *maxBytes))
		var sb strings.Builder
		flush := func() {
			for _, id := range t.Encode(sb.String()) {
				counts[id]++
				total++
			}
			sb.Reset()
		}
		for {
			line, err := r.ReadString('\n')
			sb.WriteString(line)
			if sb.Len() > 4096 || err != nil {
				flush()
			}
			if err != nil {
				break
			}
		}
		f.Close()
	}
	ids := make([]int32, len(counts))
	for i := range ids {
		ids[i] = int32(i)
	}
	slices.SortStableFunc(ids, func(a, b int32) int { return cmp.Compare(counts[b], counts[a]) })
	var ranked []int32
	seen := map[int32]bool{}
	for _, c := range t.Control() {
		if id, ok := t.ID(c); ok && !seen[id] {
			ranked, seen[id] = append(ranked, id), true
		}
	}
	for _, id := range ids {
		if len(ranked) >= *n || counts[id] == 0 {
			break
		}
		if !seen[id] {
			ranked, seen[id] = append(ranked, id), true
		}
	}
	buf := make([]byte, 4*len(ranked))
	for i, id := range ranked {
		binary.LittleEndian.PutUint32(buf[4*i:], uint32(id))
	}
	if err := os.WriteFile(*out, buf, 0o644); err != nil {
		fail(err)
	}
	var covered int64
	for _, id := range ranked {
		covered += counts[id]
	}
	for _, k := range []int{8192, 16384, 32768, 65536} {
		if k > len(ranked) {
			break
		}
		var c int64
		for _, id := range ranked[:k] {
			c += counts[id]
		}
		fmt.Printf("top %d: %.2f%% of the %d tokens counted\n", k, 100*float64(c)/float64(total), total)
	}
	fmt.Printf("%d tokens of %d seen, %d kept (%.2f%%)\n", countSeen(counts), len(counts), len(ranked), 100*float64(covered)/float64(total))
}

func countSeen(c []int64) int {
	n := 0
	for _, v := range c {
		if v > 0 {
			n++
		}
	}
	return n
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "draftvocab:", err)
	os.Exit(1)
}
