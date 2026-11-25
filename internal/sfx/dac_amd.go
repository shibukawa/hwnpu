package sfx

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/llm/iron"
	"github.com/shibukawa/bonsaiai/internal/vecops"
)

// On AMD's NPU the decoder's stages are the graphs of tools/amd/iron_moss_dac.py (dac-N/ of the
// export): each over its own frames a call (time-major, bf16), without the Core ML stages' masks.
// A graph's convolutions pad with zeros, which is what the masks give only at the clip's edges:
// so a call's window either starts at the clip's first frame or the host throws away the frames
// a halo reads past its start, and likewise at its end (decodeAMD).

// amdStageMeta is the meta of a stage's graph.
type amdStageMeta struct {
	Stage    int `json:"stage"`
	Frames   int `json:"frames"` // input frames a call
	Rate     int `json:"rate"`
	Channels int `json:"channels"`
	Out      int `json:"out"` // the output's columns (the last stage's: its one channel in 64)
	Halo     int `json:"halo"`
}

type amdStage struct {
	g    *iron.Graph
	m    amdStageMeta
	in   []byte
	name string
	// calls and took are the calls of the graph and their time, for the log
	calls int
	took  time.Duration
}

// amdDAC is the decoder's stages on AMD's NPU.
type amdDAC struct{ st []*amdStage }

func openAMDDAC(ctx context.Context, dev *bonsaiai.Device, dirs []string) (*amdDAC, error) {
	d := &amdDAC{}
	st := DACStages()
	if len(dirs) != len(st) {
		return nil, fmt.Errorf("sfx: %d decoder stages, want %d", len(dirs), len(st))
	}
	for i, dir := range dirs {
		g, err := iron.OpenGraph(ctx, dev, dir)
		if err != nil {
			d.Close()
			return nil, fmt.Errorf("sfx: the decoder's stage %d: %w", i, err)
		}
		s := &amdStage{g: g, name: fmt.Sprintf("dac-%d", i)}
		d.st = append(d.st, s)
		if err := json.Unmarshal(g.Manifest.Meta, &s.m); err != nil {
			d.Close()
			return nil, fmt.Errorf("sfx: %s: meta: %w", dir, err)
		}
		want := st[i]
		if s.m.Stage != i || s.m.Rate != want.Rate || s.m.Channels != want.Channels || s.m.Halo < want.Halo ||
			(i < len(st)-1 && s.m.Out != want.Out) || s.m.Frames <= 2*s.m.Halo {
			d.Close()
			return nil, fmt.Errorf("sfx: %s: stage %+v, not internal/sfx's %+v", dir, s.m, want)
		}
		ins, outs := g.Inputs(), g.Outputs()
		if len(ins) != 1 || len(outs) != 1 || ins[0].Bytes != 2*s.m.Frames*s.m.Channels || outs[0].Bytes != 2*s.m.Frames*s.m.Rate*s.m.Out {
			d.Close()
			return nil, fmt.Errorf("sfx: %s: the graph's input and output are not the meta's", dir)
		}
		s.in = make([]byte, ins[0].Bytes)
	}
	return d, nil
}

func (d *amdDAC) Close() error {
	for _, s := range d.st {
		s.g.Close()
	}
	d.st = nil
	return nil
}

// Decode is DecodeLatents on AMD's NPU: the latents x (frames rows of LatentDim, time-major).
func (d *amdDAC) Decode(ctx context.Context, x []float32, frames, clip, samples int) ([]float32, error) {
	st := DACStages()
	if len(x) != LatentDim*frames {
		return nil, fmt.Errorf("sfx: %d latent values for %d frames", len(x), frames)
	}
	need := make([]int, len(st)+1)
	need[len(st)] = samples
	clipAt := make([]int, len(st)+1)
	clipAt[0] = clip
	for i, s := range st {
		clipAt[i+1] = clipAt[i] * s.Rate
	}
	for i := len(st) - 1; i >= 0; i-- {
		need[i] = min(clipAt[i], (need[i+1]+st[i].Rate-1)/st[i].Rate+d.st[i].m.Halo)
	}
	cur := make([]byte, 2*len(x))
	for i, v := range x {
		binary.LittleEndian.PutUint16(cur[2*i:], bf16Even(v))
	}
	have := frames
	for i, s := range d.st {
		out, err := s.run(ctx, cur, have, clipAt[i], need[i+1])
		if err != nil {
			return nil, fmt.Errorf("sfx: decoder stage %d: %w", i, err)
		}
		cur, have = out, need[i+1]
	}
	// the output layer's convolution in column 0: its tanh
	s := d.st[len(d.st)-1]
	y := make([]float32, samples)
	row := make([]float32, 1)
	for t := range samples {
		vecops.BF16ToF32(row, cur[2*t*s.m.Out:])
		y[t] = float32(math.Tanh(float64(row[0])))
	}
	return y, nil
}

// run runs the stage over x (have rows of its channels, bf16; the frames past have zeros) for
// its first want output frames (rows of s.m.Out), in windows of its frames: each window starts
// at the clip's first frame or a halo before the first frame it gives, and ends at the clip's
// end or gives all but its last halo.
func (s *amdStage) run(ctx context.Context, x []byte, have, clip, want int) ([]byte, error) {
	m := s.m
	if clip < m.Frames {
		return nil, fmt.Errorf("a clip of %d frames under the graph's %d", clip, m.Frames)
	}
	rowIn, rowOut := 2*m.Channels, 2*m.Out
	y := make([]byte, want*rowOut)
	for done := 0; done < want; {
		c := done / m.Rate // done is a multiple of the rate
		lo := max(0, c-m.Halo)
		lo = min(lo, clip-m.Frames)
		hi := lo + m.Frames
		end := hi - m.Halo // the input frames the window gives
		if hi == clip {
			end = clip
		}
		if end <= c || (lo > 0 && lo+m.Halo > c) {
			return nil, fmt.Errorf("a window [%d, %d) for frame %d", lo, hi, c)
		}
		clear(s.in)
		if lo < have {
			copy(s.in, x[lo*rowIn:min(hi, have)*rowIn])
		}
		if err := s.g.SetInput(s.g.Inputs()[0].Name, s.in); err != nil {
			return nil, err
		}
		t0 := time.Now()
		if err := s.g.Run(ctx, nil); err != nil {
			return nil, fmt.Errorf("%s: %w", s.name, err)
		}
		s.calls++
		s.took += time.Since(t0)
		out, err := s.g.Output(s.g.Outputs()[0].Name)
		if err != nil {
			return nil, err
		}
		next := min(want, end*m.Rate)
		copy(y[done*rowOut:next*rowOut], out[(done-lo*m.Rate)*rowOut:])
		done = next
	}
	return y, nil
}

// bf16Even is x in bf16, to the nearest even.
func bf16Even(x float32) uint16 {
	bits := math.Float32bits(x)
	if bits&0x7fffffff > 0x7f800000 {
		return uint16(bits>>16) | 0x40
	}
	return uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
}
