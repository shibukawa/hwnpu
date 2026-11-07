package transcribe

import (
	"math"
	"time"
)

// TimestampStep is the time of a timestamp token: <|0.00|>, <|0.02|>, ... after
// <|notimestamps|>.
const TimestampStep = 20 * time.Millisecond

// framesPerTimestamp is the mel frames (10 ms) of a timestamp step.
const framesPerTimestamp = 2

// timestampRules masks logits as transformers' WhisperTimeStampLogitsProcessor does, for the
// next token after the generated ones (without the prompt). Timestamp tokens are begin and up:
//
//   - <|notimestamps|> (begin - 1) is never generated
//   - the first token is a timestamp, at most maxInitial steps in
//   - timestamps come in pairs: after a pair (or a timestamp first), text or the end follows;
//     after a timestamp that follows text, another timestamp or the end
//   - timestamps do not go back: a timestamp after text is at least the last one, one that
//     opens a segment past it
//   - when the timestamps together are likelier than the likeliest text token, a timestamp
func timestampRules(logits []float32, generated []int32, begin, eos int32, maxInitial int) {
	inf := float32(math.Inf(-1))
	logits[begin-1] = inf
	n := len(generated)
	if n == 0 {
		for i := int32(0); i < begin; i++ {
			logits[i] = inf
		}
		for i := int(begin) + maxInitial + 1; i < len(logits); i++ {
			logits[i] = inf
		}
	}
	last := n >= 1 && generated[n-1] >= begin
	penultimate := n < 2 || generated[n-2] >= begin
	if last {
		if penultimate {
			for i := int(begin); i < len(logits); i++ {
				logits[i] = inf
			}
		} else {
			for i := int32(0); i < eos; i++ {
				logits[i] = inf
			}
		}
	}
	for i := n - 1; i >= 0; i-- {
		if ts := generated[i]; ts >= begin {
			from := ts + 1
			if last && !penultimate {
				from = ts
			}
			for j := begin; j < from && int(j) < len(logits); j++ {
				logits[j] = inf
			}
			break
		}
	}
	// the log of the timestamps' summed probability against the likeliest text token's
	top := float32(math.Inf(-1))
	for _, v := range logits {
		top = max(top, v)
	}
	if math.IsInf(float64(top), -1) {
		return
	}
	var ts float64
	for i, v := range logits {
		if int32(i) >= begin {
			ts += math.Exp(float64(v - top))
		}
	}
	text := float32(math.Inf(-1))
	for i := int32(0); i < begin; i++ {
		text = max(text, logits[i])
	}
	if ts > 0 && math.Log(ts) > float64(text-top) {
		for i := int32(0); i < begin; i++ {
			logits[i] = inf
		}
	}
}

// Segment is a stretch of speech between two timestamps.
type Segment struct {
	Start, End time.Duration
	Tokens     []int32 // the text's, without the timestamps
}

// segments cuts a window's generated tokens (with timestamps from begin up) into segments, as
// OpenAI's transcribe and transformers' long form generation do, and returns how many mel frames
// the next window starts after this one's start: up to the last complete segment's end, or the
// whole window of frames when the tokens end with a segment left open by a single timestamp, or
// have no pair of timestamps at all (the window as one segment, whose end is its last timestamp
// when there is one past 0).
func segments(tokens []int32, begin int32, frames int) ([]Segment, int) {
	isTS := func(i int) bool { return tokens[i] >= begin }
	at := func(id int32) time.Duration { return time.Duration(id-begin) * TimestampStep }
	n := len(tokens)
	single := n >= 2 && !isTS(n-2) && isTS(n-1)
	var cuts []int // where consecutive timestamps meet: after the first of the pair
	for i := 1; i < n; i++ {
		if isTS(i-1) && isTS(i) {
			cuts = append(cuts, i)
		}
	}
	text := func(from, to int) []int32 {
		var out []int32
		for _, id := range tokens[from:to] {
			if id < begin {
				out = append(out, id)
			}
		}
		return out
	}
	if len(cuts) == 0 {
		end := time.Duration(frames) * 10 * time.Millisecond
		for i := n - 1; i >= 0; i-- {
			if isTS(i) && tokens[i] != begin {
				end = at(tokens[i])
				break
			}
		}
		if t := text(0, n); len(t) > 0 {
			return []Segment{{Start: 0, End: end, Tokens: t}}, frames
		}
		return nil, frames
	}
	if single {
		cuts = append(cuts, n)
	}
	var segs []Segment
	prev := 0
	for _, c := range cuts {
		part := tokens[prev:c]
		s := Segment{Tokens: text(prev, c)}
		if len(part) > 0 && part[0] >= begin {
			s.Start = at(part[0])
		}
		if len(part) > 0 && part[len(part)-1] >= begin {
			s.End = at(part[len(part)-1])
		}
		if len(s.Tokens) > 0 {
			segs = append(segs, s)
		}
		prev = c
	}
	if single {
		return segs, frames
	}
	advance := int(tokens[prev-1]-begin) * framesPerTimestamp
	if advance <= 0 {
		advance = frames // not stuck at the window's start
	}
	return segs, min(advance, frames)
}
