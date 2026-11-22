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
// has ended, while the program goes on. Stop cuts what is playing and what waits.
type Speaker struct {
	ctx   *oto.Context
	rate  int
	clips chan clip
	done  sync.WaitGroup

	mu      sync.Mutex
	gen     int         // counted up by Stop: the clips given before it are not played
	pending int         // clips given and not played to their end
	cur     *oto.Player // the clip playing, nil between clips
	idle    time.Time   // when the last clip ended (zero while one waits or plays)
}

// clip is a clip to play, with the Stop count it was given at.
type clip struct {
	pcm []float32
	gen int
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
	s := &Speaker{ctx: otoCtx, rate: rate, clips: make(chan clip, 64), idle: time.Now()}
	s.done.Add(1)
	go s.play()
	return s, nil
}

// Rate is the samples a second the speaker plays.
func (s *Speaker) Rate() int { return s.rate }

// play plays the clips in turn until Wait closes their channel.
func (s *Speaker) play() {
	defer s.done.Done()
	for c := range s.clips {
		s.mu.Lock()
		stale := c.gen != s.gen
		var p *oto.Player
		if !stale {
			b := make([]byte, 4*len(c.pcm))
			for i, v := range c.pcm {
				binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
			}
			p = s.ctx.NewPlayer(bytes.NewReader(b))
			s.cur = p
			p.Play()
		}
		s.mu.Unlock()
		if p != nil {
			// playing ends when the clip is read; what the player and the output still hold is
			// played after. Stop pauses the player, which ends the wait.
			for p.IsPlaying() || p.BufferedSize() > 0 {
				s.mu.Lock()
				cut := c.gen != s.gen
				s.mu.Unlock()
				if cut {
					break
				}
				time.Sleep(10 * time.Millisecond)
			}
			p.Close()
		}
		s.mu.Lock()
		s.cur = nil
		s.pending--
		if s.pending == 0 {
			s.idle = time.Now()
		}
		s.mu.Unlock()
	}
}

// Play adds a clip to be played after those before it. It does not wait for the sound.
func (s *Speaker) Play(pcm []float32) {
	if len(pcm) == 0 {
		return
	}
	s.mu.Lock()
	s.pending++
	s.idle = time.Time{}
	c := clip{append([]float32(nil), pcm...), s.gen}
	s.mu.Unlock()
	s.clips <- c
}

// Stop cuts the clip that is playing and drops those that wait. Clips given after it play.
func (s *Speaker) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.gen++
	if s.cur != nil {
		s.cur.Pause()
	}
}

// Busy reports whether a clip plays or waits to.
func (s *Speaker) Busy() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pending > 0
}

// Idle is how long the speaker has had nothing to play (0 while it plays).
func (s *Speaker) Idle() time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.pending > 0 {
		return 0
	}
	return time.Since(s.idle)
}

// Wait waits until every clip given to Play has been played. The speaker is not used after.
func (s *Speaker) Wait() {
	// a little silence last, so that the output's own buffer plays the last words out before
	// the program ends
	s.Play(make([]float32, s.rate*3/10))
	close(s.clips)
	s.done.Wait()
}
