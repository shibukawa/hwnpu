package transcribe

import (
	"math"
	"slices"
	"testing"
	"time"
)

// A small vocabulary: text 0..9, the end 10, other specials 11..13, <|notimestamps|> 14,
// timestamps from 15 (<|0.00|>) to 29.
const (
	tsEOS   = 10
	tsBegin = 15
	tsVocab = 30
)

func flat(v float32) []float32 {
	l := make([]float32, tsVocab)
	for i := range l {
		l[i] = v
	}
	return l
}

// textFirst is logits of 5 for the tokens before the timestamps and 0 for these: the text far
// likelier than the timestamps together (log 15 < 5), so that the last rule leaves the others'
// result.
func textFirst() []float32 {
	l := flat(0)
	for i := 0; i < tsBegin; i++ {
		l[i] = 5
	}
	return l
}

// allowed lists the tokens left after the rules.
func allowed(l []float32) []int32 {
	var out []int32
	for i, v := range l {
		if !math.IsInf(float64(v), -1) {
			out = append(out, int32(i))
		}
	}
	return out
}

func span(from, to int32) []int32 {
	var out []int32
	for i := from; i < to; i++ {
		out = append(out, i)
	}
	return out
}

// The rules of transformers' WhisperTimeStampLogitsProcessor, case by case.
func TestTimestampRules(t *testing.T) {
	for _, c := range []struct {
		name      string
		generated []int32
		logits    []float32
		want      []int32
	}{
		// the first token: a timestamp of at most maxInitial (3) steps
		{"first", nil, flat(0), span(tsBegin, tsBegin+4)},
		// after one timestamp (a pair's first, at the start): text, the end, other specials
		{"after the opening timestamp", []int32{tsBegin}, textFirst(), span(0, tsBegin-1)},
		// text after a timestamp: anything but <|notimestamps|> and timestamps before the last
		// (the opening one excluded too: the next must be past it)
		{"text", []int32{tsBegin + 2, 4}, textFirst(), append(span(0, tsBegin-1), span(tsBegin+3, tsVocab)...)},
		// a timestamp after text closes the segment: another timestamp (from the same one) or the end
		{"closing timestamp", []int32{tsBegin, 4, tsBegin + 5}, textFirst(), append([]int32{tsEOS, 11, 12, 13}, span(tsBegin+5, tsVocab)...)},
		// a pair: text next
		{"after a pair", []int32{tsBegin, 4, tsBegin + 5, tsBegin + 5}, textFirst(), span(0, tsBegin-1)},
		// the timestamps' sum wins over equal logits: a timestamp past the last
		{"flat logits after text", []int32{tsBegin + 2, 4}, flat(0), span(tsBegin+3, tsVocab)},
	} {
		l := slices.Clone(c.logits)
		timestampRules(l, c.generated, tsBegin, tsEOS, 3)
		if got := allowed(l); !slices.Equal(got, c.want) {
			t.Errorf("%s: allowed %v, want %v", c.name, got, c.want)
		}
	}
	// when the timestamps together are likelier than the likeliest text token, a timestamp: here
	// text 4 has 2 and each of the 13 timestamps after 16 has 0 (e^2 = 7.4 < 13)
	l := flat(float32(math.Inf(-1)))
	l[4] = 2
	for i := tsBegin + 2; i < tsVocab; i++ {
		l[i] = 0
	}
	timestampRules(l, []int32{tsBegin + 1, 3}, tsBegin, tsEOS, 3)
	if got := allowed(l); !slices.Equal(got, span(tsBegin+2, tsVocab)) {
		t.Errorf("timestamps likelier: allowed %v", got)
	}
	l = flat(float32(math.Inf(-1)))
	l[4] = 3 // e^3 = 20 > 13: text may win
	for i := tsBegin + 2; i < tsVocab; i++ {
		l[i] = 0
	}
	timestampRules(l, []int32{tsBegin + 1, 3}, tsBegin, tsEOS, 3)
	if got := allowed(l); !slices.Contains(got, 4) {
		t.Errorf("text likelier: allowed %v", got)
	}
}

// segments cuts as transformers' _retrieve_segment: at pairs of timestamps, the next window
// from the last complete segment's end, or after the window when a segment is left open.
func TestSegments(t *testing.T) {
	ts := func(steps int) int32 { return tsBegin + int32(steps) }
	at := func(steps int) time.Duration { return time.Duration(steps) * TimestampStep }
	for _, c := range []struct {
		name    string
		tokens  []int32
		want    []Segment
		advance int
	}{
		{"pairs, the last segment unfinished", []int32{ts(0), 1, 2, ts(5), ts(6), 3, ts(9), ts(10), 4},
			[]Segment{{0, at(5), []int32{1, 2}}, {at(6), at(9), []int32{3}}}, 9 * framesPerTimestamp},
		{"pairs, a single timestamp at the end", []int32{ts(0), 1, ts(5), ts(6), 3, ts(9)},
			[]Segment{{0, at(5), []int32{1}}, {at(6), at(9), []int32{3}}}, 3000},
		{"no pair", []int32{ts(0), 1, 2, ts(7)}, []Segment{{0, at(7), []int32{1, 2}}}, 3000},
		{"no timestamp past the start", []int32{ts(0), 1, 2}, []Segment{{0, 30 * time.Second, []int32{1, 2}}}, 3000},
		{"nothing", nil, nil, 3000},
		// a pair at 0 with nothing before: the window starts again past it, never stuck
		{"a pair at the start", []int32{ts(0), ts(0), 5}, nil, 3000},
	} {
		segs, adv := segments(c.tokens, tsBegin, 3000)
		if adv != c.advance {
			t.Errorf("%s: advance %d, want %d", c.name, adv, c.advance)
		}
		if len(segs) != len(c.want) {
			t.Errorf("%s: segments %v, want %v", c.name, segs, c.want)
			continue
		}
		for i := range segs {
			if segs[i].Start != c.want[i].Start || segs[i].End != c.want[i].End || !slices.Equal(segs[i].Tokens, c.want[i].Tokens) {
				t.Errorf("%s: segment %d %v, want %v", c.name, i, segs[i], c.want[i])
			}
		}
	}
	// the last window's frames: the advance never passes them
	if _, adv := segments([]int32{ts(0), 1, ts(14), ts(14), 2}, tsBegin, 20); adv != 20 {
		t.Errorf("advance %d past the window's 20 frames", adv)
	}
}
