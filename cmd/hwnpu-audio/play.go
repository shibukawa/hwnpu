package main

import (
	"context"
	"encoding/binary"
	"io"
	"math"
	"sync"
	"time"

	"github.com/ebitengine/oto/v3"

	"github.com/shibukawa/hwnpu/internal/music"
)

// speaker plays the music through the default audio output as it is generated (oto, which on
// macOS needs no cgo): the samples go into a buffer the player reads, and it starts once a second
// is buffered, so that the next second is generated before the first has been heard.
type speaker struct {
	mu      sync.Mutex
	buf     []byte
	closed  bool
	started bool
	player  *oto.Player
	gain    float64 // what the samples are multiplied by: 0.5, the WAV's (music), or 1 (sound effects)
}

// prebuffer is how much is buffered before the music starts.
const prebuffer = 4 * music.SampleRate // bytes of int16 stereo: one second

var otoContext struct {
	once sync.Once
	ctx  *oto.Context
	err  error
}

func newSpeaker() (*speaker, error) {
	otoContext.once.Do(func() {
		var ready chan struct{}
		otoContext.ctx, ready, otoContext.err = oto.NewContext(&oto.NewContextOptions{
			SampleRate:   music.SampleRate,
			ChannelCount: 2,
			Format:       oto.FormatSignedInt16LE,
		})
		if otoContext.err == nil {
			<-ready
		}
	})
	if otoContext.err != nil {
		return nil, otoContext.err
	}
	s := &speaker{gain: 0.5}
	s.player = otoContext.ctx.NewPlayer(s)
	return s, nil
}

// write queues interleaved stereo samples (at the gain of the WAV).
func (s *speaker) write(pcm []float32) {
	b := make([]byte, 2*len(pcm))
	for i, v := range pcm {
		x := math.Round(math.Max(-1, math.Min(1, s.gain*float64(v))) * 32767)
		binary.LittleEndian.PutUint16(b[2*i:], uint16(int16(x)))
	}
	s.mu.Lock()
	s.buf = append(s.buf, b...)
	start := !s.started && len(s.buf) >= prebuffer
	s.started = s.started || start
	s.mu.Unlock()
	if start {
		s.player.Play()
	}
}

// Read gives the player what is buffered: nothing while the next second is generated (oto's
// mixer reads again a millisecond later), io.EOF when the music has ended.
func (s *speaker) Read(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.buf) == 0 {
		if s.closed {
			return 0, io.EOF
		}
		return 0, nil
	}
	n := copy(p, s.buf)
	s.buf = s.buf[n:]
	return n, nil
}

// finish says the music has ended and waits until it has been heard, or ctx ends.
func (s *speaker) finish(ctx context.Context) error {
	// a quarter of a second of silence after the music, and a wait for the device's buffer after
	// the player has read it all, so that the music's last moments are heard
	s.write(make([]float32, 2*music.SampleRate/4))
	s.mu.Lock()
	s.closed = true
	start := !s.started
	s.started = true
	s.mu.Unlock()
	if start {
		s.player.Play()
	}
	for s.player.IsPlaying() {
		select {
		case <-ctx.Done():
			s.player.Pause()
			return ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
	select {
	case <-ctx.Done():
	case <-time.After(300 * time.Millisecond):
	}
	return s.player.Close()
}
