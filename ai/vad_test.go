package ai

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"
	"time"
)

// clip is 16 kHz audio of parts: a tone of amplitude amp (0 for the background noise alone) for
// a duration each.
func clip(parts ...struct {
	d   time.Duration
	amp float64
}) []float32 {
	rng := rand.New(rand.NewPCG(1, 2))
	var out []float32
	for _, p := range parts {
		n := int(p.d.Seconds() * TranscribeRate)
		for i := 0; i < n; i++ {
			v := 0.002*(rng.Float64()*2-1) + p.amp*math.Sin(2*math.Pi*220*float64(len(out))/TranscribeRate)
			out = append(out, float32(v))
		}
	}
	return out
}

type part = struct {
	d   time.Duration
	amp float64
}

func near(a, b time.Duration) bool { return (a - b).Abs() <= 30*time.Millisecond }

func TestVADSegments(t *testing.T) {
	pcm := clip(part{time.Second, 0}, part{time.Second, 0.3}, part{time.Second, 0},
		part{100 * time.Millisecond, 0.5}, // a click: shorter than MinSpeech
		part{time.Second, 0}, part{800 * time.Millisecond, 0.2}, part{300 * time.Millisecond, 0})
	v := NewVAD(VADOptions{})
	segs := v.Segments(pcm)
	if len(segs) != 2 {
		t.Fatalf("segments %v, want 2", segs)
	}
	// with 200 ms of padding on each side
	want := []Span{{800 * time.Millisecond, 2200 * time.Millisecond}, {3900 * time.Millisecond, 5100 * time.Millisecond}}
	for i, s := range segs {
		if !near(s.Start, want[i].Start) || !near(s.End, want[i].End) {
			t.Errorf("segment %d is %v, want about %v", i, s, want[i])
		}
	}
}

func TestVADStreamMatchesBatch(t *testing.T) {
	pcm := clip(part{500 * time.Millisecond, 0}, part{1500 * time.Millisecond, 0.3}, part{time.Second, 0}, part{time.Second, 0.1}, part{time.Second, 0})
	v := NewVAD(VADOptions{})
	batch, err := Collect(v.Detect(context.Background(), Of(pcm, 0)))
	if err != nil {
		t.Fatal(err)
	}
	// the same audio in chunks of odd sizes, as a microphone gives them
	rng := rand.New(rand.NewPCG(3, 4))
	stream := func(yield func(Chunk[[]float32], error) bool) {
		for i := 0; i < len(pcm); {
			n := min(len(pcm)-i, 1+rng.IntN(1000))
			if !yield(Chunk[[]float32]{Data: pcm[i : i+n]}, nil) {
				return
			}
			i += n
		}
	}
	streamed, err := Collect(v.Detect(context.Background(), stream))
	if err != nil {
		t.Fatal(err)
	}
	flat := func(cs []Chunk[[]float32]) (data []float32, finals []time.Duration) {
		for _, c := range cs {
			if len(data) > 0 || len(c.Data) > 0 {
				data = append(data, c.Data...)
			}
			if c.Final {
				finals = append(finals, c.Start+c.Len)
			}
		}
		return
	}
	bd, bf := flat(batch)
	sd, sf := flat(streamed)
	if len(bd) != len(sd) || len(bf) != len(sf) || len(bf) != 2 {
		t.Fatalf("batch %d samples, %d utterances; stream %d samples, %d utterances", len(bd), len(bf), len(sd), len(sf))
	}
	for i := range bd {
		if bd[i] != sd[i] {
			t.Fatalf("sample %d differs", i)
		}
	}
	// each utterance's chunks are contiguous, and their times are the input's
	for _, cs := range [][]Chunk[[]float32]{batch, streamed} {
		var next time.Duration = -1
		for _, c := range cs {
			if next >= 0 && c.Start != next {
				t.Fatalf("a chunk starts at %v, the one before ended at %v", c.Start, next)
			}
			next = c.Start + c.Len
			if c.Final {
				next = -1
			}
			start := int(c.Start.Seconds()*TranscribeRate + 0.5)
			if len(c.Data) > 0 && c.Data[0] != pcm[start] {
				t.Fatalf("the chunk at %v does not hold the input's samples there", c.Start)
			}
		}
	}
}

func TestVADMaxSpeech(t *testing.T) {
	pcm := clip(part{300 * time.Millisecond, 0}, part{5 * time.Second, 0.3}, part{time.Second, 0})
	segs := NewVAD(VADOptions{MaxSpeech: 2 * time.Second}).Segments(pcm)
	if len(segs) < 3 {
		t.Fatalf("segments %v: 5 s of speech in at most 2 s each", segs)
	}
	for _, s := range segs {
		if s.End-s.Start > 2*time.Second+50*time.Millisecond {
			t.Errorf("segment %v is longer than MaxSpeech", s)
		}
	}
}
