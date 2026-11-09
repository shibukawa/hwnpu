package tokenizer

import (
	"errors"
	"fmt"
	"math"
	"os"
)

// LoadSentencePieceUnigram reads a SentencePiece Unigram model (a spm.model) as a Unigram: the
// normal and user-defined pieces, the precompiled normalization (nmt_nfkc's charsmap), a ▁
// before every word (add_dummy_prefix, extra whitespace removed) and, when the model was trained
// with it, byte fallback. Nothing is put after the text; control pieces are not matched in it.
func LoadSentencePieceUnigram(path string) (*Unigram, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	u, err := parseSentencePieceUnigram(raw)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %s: %w", path, err)
	}
	return u, nil
}

func parseSentencePieceUnigram(raw []byte) (*Unigram, error) {
	u := &Unigram{pieces: map[string]int32{}, unk: -1, space: "▁"}
	modelType, byteFallback := 1, false
	dummyPrefix, removeSpaces, escapeSpaces := true, true, true
	var cm []byte
	var bytes [256]int32
	nbytes := 0
	minScore := math.Inf(1)
	err := protoFields(raw, func(f int, v uint64, b []byte) error {
		switch f {
		case 1:
			piece, score, typ := "", float32(0), uint64(spNormal)
			if err := protoFields(b, func(f int, v uint64, b []byte) error {
				switch f {
				case 1:
					piece = string(b)
				case 2:
					score = math.Float32frombits(uint32(v))
				case 3:
					typ = v
				}
				return nil
			}); err != nil {
				return err
			}
			id := int32(len(u.scores))
			u.scores = append(u.scores, float64(score))
			switch typ {
			case spNormal, spUserDefined:
				if _, ok := u.pieces[piece]; !ok {
					u.pieces[piece] = id
				}
				u.maxLen = max(u.maxLen, len(piece))
				if typ == spNormal {
					minScore = min(minScore, float64(score))
				}
			case spUnknown:
				u.unk = id
			case spByte:
				var c byte
				if _, err := fmt.Sscanf(piece, "<0x%02X>", &c); err != nil {
					return fmt.Errorf("byte piece %q", piece)
				}
				bytes[c] = id
				nbytes++
			}
		case 2:
			return protoFields(b, func(f int, v uint64, b []byte) error {
				switch f {
				case 3:
					modelType = int(v)
				case 35:
					byteFallback = v != 0
				}
				return nil
			})
		case 3:
			return protoFields(b, func(f int, v uint64, b []byte) error {
				switch f {
				case 2:
					cm = b
				case 3:
					dummyPrefix = v != 0
				case 4:
					removeSpaces = v != 0
				case 5:
					escapeSpaces = v != 0
				}
				return nil
			})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	switch {
	case modelType != 1:
		return nil, fmt.Errorf("model type %d is not Unigram (1)", modelType)
	case u.unk < 0:
		return nil, errors.New("no unknown piece")
	case !dummyPrefix || !removeSpaces || !escapeSpaces:
		return nil, errors.New("only add_dummy_prefix, remove_extra_whitespaces and escape_whitespaces are supported")
	case byteFallback && nbytes != 256:
		return nil, fmt.Errorf("byte fallback with %d byte pieces", nbytes)
	}
	if byteFallback {
		u.bytes = bytes[:]
	}
	// sentencepiece's kUnkPenalty: an unknown character costs the lowest score less 10.
	u.unkCost = minScore - 10
	if len(cm) > 0 {
		c, err := charsmapFrom(cm)
		if err != nil {
			return nil, err
		}
		u.norm = c
	}
	return u, nil
}
