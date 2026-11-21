package capture

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/ebitengine/oto/v3"
)

// Speaker plays sound on the system's default output through oto (github.com/ebitengine/oto,
// without cgo): the clips given to Play one after another, each as soon as the one before it
// has ended, while the program goes on.
type Speaker struct {
	ctx   *oto.Context
	rate  int
	clips chan []float32
	done  sync.WaitGroup
}

var (
	otoOnce sync.Once
	otoCtx  *oto.Context
	otoRate int
	otoErr  error
)

// OpenSpeaker opens the output for mono sound of rate samples a second. A process has one
// output, of the rate it was first opened at.
func OpenSpeaker(rate int) (*Speaker, error) {
	otoOnce.Do(func() {
		var ready chan struct{}
		otoCtx, ready, otoErr = oto.NewContext(&oto.NewContextOptions{SampleRate: rate, ChannelCount: 1, Format: oto.FormatFloat32LE})
		if otoErr == nil {
			<-ready
			otoRate = rate
		}
	})
	if otoErr != nil {
		return nil, fmt.Errorf("the speaker: %w", otoErr)
	}
	if rate != otoRate {
		return nil, fmt.Errorf("the speaker: opened at %d Hz, not %d", otoRate, rate)
	}
	s := &Speaker{ctx: otoCtx, rate: rate, clips: make(chan []float32, 64)}
	s.done.Add(1)
	go s.play()
	return s, nil
}

// play plays the clips in turn until Wait closes their channel.
func (s *Speaker) play() {
	defer s.done.Done()
	for pcm := range s.clips {
		b := make([]byte, 4*len(pcm))
		for i, v := range pcm {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
		}
		p := s.ctx.NewPlayer(bytes.NewReader(b))
		p.Play()
		// playing ends when the clip is read; what the player and the output still hold is
		// played after
		for p.IsPlaying() || p.BufferedSize() > 0 {
			time.Sleep(10 * time.Millisecond)
		}
		p.Close()
	}
}

// Play adds a clip to be played after those before it. It does not wait for the sound.
func (s *Speaker) Play(pcm []float32) {
	if len(pcm) > 0 {
		s.clips <- append([]float32(nil), pcm...)
	}
}

// Wait waits until every clip given to Play has been played. The speaker is not used after.
func (s *Speaker) Wait() {
	// a little silence last, so that the output's own buffer plays the last words out before
	// the program ends
	s.clips <- make([]float32, s.rate*3/10)
	close(s.clips)
	s.done.Wait()
}
