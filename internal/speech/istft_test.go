package speech

import (
	"math"
	"math/rand/v2"
	"testing"
)

// refISTFT is the inverse STFT as first written (a division per term, one output at a time):
// Frame and Frames must give the same samples to the bit.
type refISTFT struct{ *ISTFT }

func (t refISTFT) frame(polar []float32) []float32 {
	bins := t.nfft/2 + 1
	for j := 0; j < bins; j++ {
		mag := math.Exp(float64(polar[j]))
		ph := float64(polar[bins+j])
		t.re[j], t.im[j] = mag*math.Cos(ph), mag*math.Sin(ph)
	}
	for j := 1; j < bins-1; j++ {
		t.re[t.nfft-j], t.im[t.nfft-j] = t.re[j], -t.im[j]
	}
	for n := 0; n < t.nfft; n++ {
		var acc float64
		for k := 0; k < t.nfft; k++ {
			idx := (k * n) % t.nfft
			acc += t.re[k]*t.cos[idx] - t.im[k]*t.sin[idx]
		}
		t.td[n] = acc / float64(t.nfft)
	}
	return t.add(t.td)
}

func TestISTFTReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, c := range []struct{ nfft, hop, frames int }{{16, 4, 9}, {15, 5, 7}, {1280, 320, 13}, {1281, 320, 3}, {6, 2, 1}} {
		w := 2 * (c.nfft/2 + 1)
		stft := make([]float32, c.frames*w)
		for i := range stft {
			stft[i] = float32(rng.Float64()*6 - 3)
		}
		ref := refISTFT{NewISTFT(c.nfft, c.hop)}
		var want []float32
		for f := range c.frames {
			want = append(want, ref.frame(stft[f*w:(f+1)*w])...)
		}
		want = append(want, ref.Flush()...)
		one := NewISTFT(c.nfft, c.hop)
		var got []float32
		for f := range c.frames {
			got = append(got, one.Frame(stft[f*w:(f+1)*w])...)
		}
		got = append(got, one.Flush()...)
		all := NewISTFT(c.nfft, c.hop)
		par := append(all.Frames(stft), all.Flush()...)
		for name, g := range map[string][]float32{"Frame": got, "Frames": par} {
			if len(g) != len(want) {
				t.Fatalf("nfft %d: %s gave %d samples, want %d", c.nfft, name, len(g), len(want))
			}
			for i := range g {
				if math.Float32bits(g[i]) != math.Float32bits(want[i]) {
					t.Fatalf("nfft %d: %s sample %d is %g, want %g", c.nfft, name, i, g[i], want[i])
				}
			}
		}
	}
}

func BenchmarkISTFT(b *testing.B) {
	const nfft, hop, frames = 1280, 320, 75 // one second at 24 kHz
	w := 2 * (nfft/2 + 1)
	stft := make([]float32, frames*w)
	for i := range stft {
		stft[i] = float32(i%7) * 0.1
	}
	b.Run("Frame", func(b *testing.B) {
		for range b.N {
			t := NewISTFT(nfft, hop)
			for f := range frames {
				t.Frame(stft[f*w : (f+1)*w])
			}
		}
	})
	b.Run("Frames", func(b *testing.B) {
		for range b.N {
			NewISTFT(nfft, hop).Frames(stft)
		}
	})
}
