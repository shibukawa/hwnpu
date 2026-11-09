package tokenizer

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Unigram is a Hugging Face tokenizer.json of a Unigram model as T5's (sentencepiece converted):
// the Precompiled normalizer (sentencepiece's charsmap), WhitespaceSplit then Metaspace ("▁"
// prepended to every word), the most likely segmentation of each word (Viterbi over the
// pieces' log probabilities), and the post-processor's tokens after the text (T5's </s>).
type Unigram struct {
	pieces  map[string]int32
	scores  []float64
	maxLen  int // the longest piece in bytes
	unk     int32
	unkCost float64
	after   []int32 // what the post-processor puts after a text
	norm    *charsmap
	space   string // Metaspace's replacement
	// bytes are the <0xNN> pieces when an unknown character becomes its UTF-8 bytes
	// (sentencepiece's byte fallback), or nil.
	bytes []int32
}

// LoadUnigram reads a Unigram tokenizer.json.
func LoadUnigram(path string) (*Unigram, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	u, err := parseUnigram(raw)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	return u, nil
}

func parseUnigram(raw []byte) (*Unigram, error) {
	var f struct {
		Normalizer struct {
			Type     string `json:"type"`
			Charsmap string `json:"precompiled_charsmap"`
		} `json:"normalizer"`
		PreTokenizer struct {
			Type          string `json:"type"`
			Pretokenizers []struct {
				Type          string `json:"type"`
				Replacement   string `json:"replacement"`
				PrependScheme string `json:"prepend_scheme"`
				Split         bool   `json:"split"`
			} `json:"pretokenizers"`
		} `json:"pre_tokenizer"`
		PostProcessor struct {
			Type   string `json:"type"`
			Single []struct {
				Sequence *struct {
					ID string `json:"id"`
				} `json:"Sequence"`
				SpecialToken *struct {
					ID string `json:"id"`
				} `json:"SpecialToken"`
			} `json:"single"`
			SpecialTokens map[string]struct {
				IDs []int32 `json:"ids"`
			} `json:"special_tokens"`
		} `json:"post_processor"`
		Model struct {
			Type         string               `json:"type"`
			UnkID        *int32               `json:"unk_id"`
			Vocab        [][2]json.RawMessage `json:"vocab"`
			ByteFallback bool                 `json:"byte_fallback"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, err
	}
	if f.Model.Type != "Unigram" {
		return nil, fmt.Errorf("model %s is not Unigram", f.Model.Type)
	}
	if f.Model.ByteFallback {
		return nil, errors.New("Unigram with byte fallback is not supported")
	}
	if f.Model.UnkID == nil {
		return nil, errors.New("Unigram without unk_id is not supported")
	}
	u := &Unigram{pieces: map[string]int32{}, unk: *f.Model.UnkID, space: "▁"}
	minScore := math.Inf(1)
	for i, v := range f.Model.Vocab {
		var piece string
		var score float64
		if err := json.Unmarshal(v[0], &piece); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(v[1], &score); err != nil {
			return nil, err
		}
		if _, ok := u.pieces[piece]; !ok {
			u.pieces[piece] = int32(i)
		}
		u.scores = append(u.scores, score)
		u.maxLen = max(u.maxLen, len(piece))
		minScore = min(minScore, score)
	}
	// Hugging Face's Unigram: an unknown character costs the lowest score less 10.
	u.unkCost = minScore - 10
	switch f.Normalizer.Type {
	case "Precompiled":
		cm, err := parseCharsmap(f.Normalizer.Charsmap)
		if err != nil {
			return nil, err
		}
		u.norm = cm
	case "":
	default:
		return nil, fmt.Errorf("normalizer %s is not supported; only Precompiled", f.Normalizer.Type)
	}
	pt := f.PreTokenizer.Pretokenizers
	if f.PreTokenizer.Type != "Sequence" || len(pt) != 2 || pt[0].Type != "WhitespaceSplit" || pt[1].Type != "Metaspace" ||
		pt[1].PrependScheme != "always" || !pt[1].Split {
		return nil, errors.New("pre-tokenizer is not supported; only WhitespaceSplit then Metaspace (prepend always, split)")
	}
	u.space = pt[1].Replacement
	pp := f.PostProcessor
	if pp.Type != "TemplateProcessing" || len(pp.Single) == 0 || pp.Single[0].Sequence == nil {
		return nil, errors.New("post-processor is not supported; only a template of the text then special tokens")
	}
	for _, s := range pp.Single[1:] {
		if s.SpecialToken == nil {
			return nil, errors.New("post-processor is not supported; only special tokens after the text")
		}
		st, ok := pp.SpecialTokens[s.SpecialToken.ID]
		if !ok {
			return nil, fmt.Errorf("post-processor: no special token %s", s.SpecialToken.ID)
		}
		u.after = append(u.after, st.IDs...)
	}
	return u, nil
}

// Encode returns text's token ids with the post-processor's after them, as transformers'
// tokenizer(text).input_ids. Added tokens written in the text (<extra_id_0>) are not split out.
func (u *Unigram) Encode(text string) []int32 {
	if u.norm != nil {
		text = u.norm.normalize(text)
	}
	var ids []int32
	for _, w := range strings.FieldsFunc(text, unicode.IsSpace) {
		ids = u.word(ids, u.space+strings.ReplaceAll(w, " ", u.space))
	}
	return append(ids, u.after...)
}

// word appends the most likely segmentation of w: Viterbi over its byte positions, an unknown
// character one token of unkCost, consecutive unknowns fused (Hugging Face's fuse_unk).
func (u *Unigram) word(ids []int32, w string) []int32 {
	n := len(w)
	best := make([]float64, n+1)
	from := make([]int, n+1)
	tok := make([]int32, n+1)
	for i := 1; i <= n; i++ {
		best[i] = math.Inf(-1)
	}
	for i := 0; i < n; i++ {
		if math.IsInf(best[i], -1) || !utf8.RuneStart(w[i]) {
			continue
		}
		found := false
		for j := i + 1; j <= min(n, i+u.maxLen); j++ {
			if j < n && !utf8.RuneStart(w[j]) {
				continue
			}
			id, ok := u.pieces[w[i:j]]
			if !ok {
				continue
			}
			found = found || j == i+runeLen(w[i:])
			if s := best[i] + u.scores[id]; s > best[j] {
				best[j], from[j], tok[j] = s, i, id
			}
		}
		if !found { // the character alone is no piece: unknown
			j := i + runeLen(w[i:])
			if s := best[i] + u.unkCost; s > best[j] {
				best[j], from[j], tok[j] = s, i, u.unk
			}
		}
	}
	var rev []int32
	var ends []int
	for j := n; j > 0; j = from[j] {
		rev = append(rev, tok[j])
		ends = append(ends, j)
	}
	for k := len(rev) - 1; k >= 0; k-- {
		if rev[k] == u.unk && u.bytes != nil {
			for _, c := range []byte(w[from[ends[k]]:ends[k]]) {
				ids = append(ids, u.bytes[c])
			}
			continue
		}
		if rev[k] == u.unk && len(ids) > 0 && ids[len(ids)-1] == u.unk {
			continue
		}
		ids = append(ids, rev[k])
	}
	return ids
}

func runeLen(s string) int {
	_, n := utf8.DecodeRuneInString(s)
	return n
}

// charsmap is sentencepiece's precompiled normalization: a Darts double-array trie from byte
// strings to offsets in a blob of NUL-terminated replacements.
type charsmap struct {
	trie []uint32
	repl []byte
}

func parseCharsmap(b64 string) (*charsmap, error) {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, fmt.Errorf("precompiled charsmap: %w", err)
	}
	return charsmapFrom(raw)
}

// charsmapFrom reads a precompiled charsmap's bytes.
func charsmapFrom(raw []byte) (*charsmap, error) {
	if len(raw) < 4 {
		return nil, errors.New("precompiled charsmap: too short")
	}
	size := int(binary.LittleEndian.Uint32(raw))
	if size%4 != 0 || 4+size > len(raw) {
		return nil, errors.New("precompiled charsmap: bad trie size")
	}
	cm := &charsmap{trie: make([]uint32, size/4), repl: raw[4+size:]}
	for i := range cm.trie {
		cm.trie[i] = binary.LittleEndian.Uint32(raw[4+4*i:])
	}
	return cm, nil
}

// first returns the replacement of the shortest prefix of s the trie holds (Darts'
// commonPrefixSearch's first result, which Hugging Face's spm_precompiled takes).
func (c *charsmap) first(s string) (string, bool) {
	unit := func(p uint32) uint32 {
		if int(p) >= len(c.trie) {
			return 0
		}
		return c.trie[p]
	}
	offset := func(u uint32) uint32 { return (u >> 10) << ((u & (1 << 9)) >> 6) }
	hasLeaf := func(u uint32) bool { return (u>>8)&1 == 1 }
	label := func(u uint32) uint32 { return u & ((1 << 31) | 0xFF) }
	pos := offset(unit(0))
	for i := 0; i < len(s); i++ {
		ch := uint32(s[i])
		if ch == 0 {
			break
		}
		pos ^= ch
		u := unit(pos)
		if label(u) != ch {
			break
		}
		pos ^= offset(u)
		if hasLeaf(u) {
			value := int(unit(pos) & ((1 << 31) - 1))
			if value >= len(c.repl) {
				return "", false
			}
			end := value
			for end < len(c.repl) && c.repl[end] != 0 {
				end++
			}
			return string(c.repl[value:end]), true
		}
	}
	return "", false
}

// normalize applies the charsmap as Hugging Face's Precompiled normalizer does: a grapheme
// (here: a character with the combining marks after it) of under 6 bytes replaced whole when
// a prefix of it is in the trie, else each character replaced when it is.
func (c *charsmap) normalize(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		j := i + runeLen(s[i:])
		for j < len(s) {
			r, n := utf8.DecodeRuneInString(s[j:])
			if !unicode.Is(unicode.M, r) {
				break
			}
			j += n
		}
		g := s[i:j]
		if r, ok := c.first(g); ok && len(g) < 6 {
			b.WriteString(r)
		} else {
			for k := 0; k < len(g); {
				n := runeLen(g[k:])
				if r, ok := c.first(g[k : k+n]); ok {
					b.WriteString(r)
				} else {
					b.WriteString(g[k : k+n])
				}
				k += n
			}
		}
		i = j
	}
	return b.String()
}
