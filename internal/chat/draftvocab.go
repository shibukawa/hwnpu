package chat

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"unicode"
	"unicode/utf8"

	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// The multi-token prediction layer's output layer is the model's, over its whole vocabulary:
// Qwen3.5-4B's 248320 rows were 11.4 of the layer's 13.6 ms a call, twice for every step of the
// model (docs/design.md 16.192). Its drafts are checked by the model, so a part of the
// vocabulary loses only the drafts of the tokens outside it, and the answers stay the same (as
// FR-Spec's drafts over the frequent tokens). draftVocab is that part.

// draftVocabRule names the rule of draftVocab in the programs' names: another rule converts anew.
const draftVocabRule = "dv2"

// draftVocabLow is the lowest token IDs draftVocab keeps: byte-level BPE gives the earlier
// merges, the frequent ones, the lower IDs. They cover 95 % of English text and 96 % of Go code
// with Qwen3.5's tokenizer, but its Japanese tokens are spread over the whole vocabulary (65 % at
// 32768, 81 % at 131072), and so are some of markdown's.
const draftVocabLow = 32768

// draftVocabFile is the file next to the layer's program that lists its rows (draftVocab), read
// back to map its logits to tokens.
const draftVocabFile = "draft-vocab.json"

// draftVocab is the part of the vocabulary the drafts are of: the IDs under draftVocabLow, the
// tokens with kana, those of one or two kanji and those of ASCII symbols and spaces alone
// (74262 of Qwen3.5's 248320: 97 % of Japanese text, 96 % of English and of Go code, all of an
// answer of code and Japanese), in increasing order, padded with the next IDs to a multiple of 16
// (the lookup tables' rows). nil when that is not smaller than half the vocabulary. Kanji alone
// (dv1, 43618 rows) left out words such as 説明 and 計算 and the code fence, and a draft missed
// ends the drafts after it: the drafts taken fell from 82 to 67 of an answer, slower than the
// whole vocabulary.
func draftVocab(tok *tokenizer.Tokenizer) []int32 {
	n := tok.VocabSize()
	var ids []int32
	for id := 0; id < n; id++ {
		if id < draftVocabLow || draftToken(tok.Decode([]int32{int32(id)}, false)) {
			ids = append(ids, int32(id))
		}
	}
	for next := int32(draftVocabLow); len(ids)%16 != 0 && int(next) < n; next++ {
		if i := searchID(ids, next); i == len(ids) || ids[i] != next {
			ids = append(ids[:i], append([]int32{next}, ids[i:]...)...)
		}
	}
	if len(ids)%16 != 0 || len(ids)*2 > n {
		return nil
	}
	return ids
}

// draftToken reports whether a token's text is one draftVocab keeps beyond the low IDs: kana in
// it, one or two kanji, or ASCII symbols and spaces alone (markdown's and code's: the code
// fence's ``` is above the low IDs).
func draftToken(s string) bool {
	han, sym := 0, 0
	for _, r := range s {
		switch {
		case unicode.In(r, unicode.Hiragana, unicode.Katakana):
			return true
		case unicode.Is(unicode.Han, r):
			han++
		case r < utf8.RuneSelf && !unicode.IsLetter(r) && !unicode.IsDigit(r):
			sym++
		}
	}
	n := utf8.RuneCountInString(s)
	return n > 0 && (han == n && n <= 2 || sym == n)
}

// searchID is the index of the first of the increasing ids not below id.
func searchID(ids []int32, id int32) int {
	lo, hi := 0, len(ids)
	for lo < hi {
		mid := (lo + hi) / 2
		if ids[mid] < id {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// writeDraftVocab writes ids into the export dir (draftVocabFile).
func writeDraftVocab(dir string, ids []int32) error {
	b, err := json.Marshal(ids)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, draftVocabFile), b, 0o644)
}

// readDraftVocab reads the rows of the program at path (a manifest or a program in an export),
// nil when it has none (its logits are of the whole vocabulary).
func readDraftVocab(path string) ([]int32, error) {
	b, err := os.ReadFile(filepath.Join(filepath.Dir(path), draftVocabFile))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var ids []int32
	if err := json.Unmarshal(b, &ids); err != nil {
		return nil, fmt.Errorf("chat: %s: %w", draftVocabFile, err)
	}
	return ids, nil
}
