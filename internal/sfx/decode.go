package sfx

import (
	"context"
	"fmt"
)

// Program is a loaded program (runner.Program): fp16 inputs by name, output y.
type Program interface {
	Run(ctx context.Context, in map[string][]float32) ([]float32, error)
}

// DecodeLatents runs the DAC decoder's stages over the latents z (LatentDim x frames, channel
// major) of a clip of clip frames (the reference decodes all 30 seconds, 1500 frames, and cuts
// the samples it wants out of them: the frames past clip are zeros to the decoder) and returns
// the clip's first samples samples. Each stage runs in calls of its In frames: a call gives the
// frames between its halos, and the next call starts where they end. Only the frames the
// samples wanted depend on are computed, so a short sound costs less than 30 seconds.
func DecodeLatents(ctx context.Context, stages []Program, z []float32, frames, clip, samples int) ([]float32, error) {
	st := DACStages()
	if len(stages) != len(st) {
		return nil, fmt.Errorf("sfx: %d decoder stages, want %d", len(stages), len(st))
	}
	if len(z) != LatentDim*frames {
		return nil, fmt.Errorf("sfx: %d latent values for %d frames", len(z), frames)
	}
	// the frames each stage gives (the last, samples), back from the end
	need := make([]int, len(st)+1) // need[i] the frames stage i takes, need[len] the samples
	need[len(st)] = samples
	clipAt := make([]int, len(st)+1) // the clip's frames at each stage's input rate
	clipAt[0] = clip
	for i, s := range st {
		clipAt[i+1] = clipAt[i] * s.Rate
	}
	for i := len(st) - 1; i >= 0; i-- {
		s := st[i]
		need[i] = min(clipAt[i], (need[i+1]+s.Rate-1)/s.Rate+s.Halo)
	}
	x, have := z, frames
	for i, s := range st {
		out, err := runStage(ctx, stages[i], s, x, have, clipAt[i], need[i+1])
		if err != nil {
			return nil, fmt.Errorf("sfx: decoder stage %d: %w", i, err)
		}
		x, have = out, need[i+1]
	}
	return x[:samples], nil
}

// runStage runs a stage over x (s.Channels x have frames; frames past have are zeros) for its
// first want output frames (s.Out x want).
func runStage(ctx context.Context, p Program, s DACStage, x []float32, have, clip, want int) ([]float32, error) {
	core := s.In - 2*s.Halo
	if core <= 0 {
		return nil, fmt.Errorf("halo %d over %d frames", s.Halo, s.In)
	}
	outN := s.In * s.Rate
	y := make([]float32, s.Out*want)
	in := map[string][]float32{
		"x":  make([]float32, s.Channels*s.In),
		"m":  make([]float32, s.In),
		"mo": make([]float32, outN),
		"mg": {Magic},
	}
	for c0 := 0; c0*s.Rate < want; c0 += core {
		lo := c0 - s.Halo // the call's first input frame
		xs := in["x"]
		clear(xs)
		for c := range s.Channels {
			row := x[c*have : (c+1)*have]
			for t := range s.In {
				if g := lo + t; g >= 0 && g < have {
					xs[c*s.In+t] = row[g]
				}
			}
		}
		for t := range s.In {
			in["m"][t] = inside(lo+t, clip)
		}
		for t := range outN {
			in["mo"][t] = inside(lo*s.Rate+t, clip*s.Rate)
		}
		out, err := p.Run(ctx, in)
		if err != nil {
			return nil, err
		}
		if len(out) != s.Out*outN {
			return nil, fmt.Errorf("output of %d values, want %d", len(out), s.Out*outN)
		}
		// the frames between the halos: c0 * Rate on
		from := s.Halo * s.Rate
		for c := range s.Out {
			for t := range core * s.Rate {
				g := c0*s.Rate + t
				if g >= want {
					break
				}
				y[c*want+g] = out[c*outN+from+t]
			}
		}
	}
	return y, nil
}

func inside(t, n int) float32 {
	if t >= 0 && t < n {
		return 1
	}
	return 0
}
