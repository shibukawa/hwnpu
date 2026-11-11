package pitch

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/onnx"
	"github.com/shibukawa/hwnpu/internal/onnxrt"
)

// cpuRunner runs SwiftF0's CNN from the ONNX file with ONNX Runtime: the graph cut from its
// features (Model.Features' values) to the pitch logits and the voicing logit. It runs a window
// at its own length, so it needs no mask: the graph's convolutions pad the clip's ends with zeros
// as the program's masks do.
type cpuRunner struct {
	s                *onnxrt.Session
	feat, pit, voice string
}

// cutValues are the graph's values the CNN begins and ends with: the first convolution's input
// (the features, (1, 3, 128, frames)), the logits the band's Where masks ((1, 95, frames)) and
// the voicing logit the confidence's Sigmoid takes ((1, frames)).
func cutValues(g *onnx.Graph) (feat, pit, voice string, err error) {
	prod := map[string]*onnx.Node{}
	for _, n := range g.Nodes {
		for _, o := range n.Out {
			prod[o] = n
		}
	}
	for _, n := range g.Nodes {
		switch {
		case n.Op == "Conv" && feat == "":
			feat = n.In[0]
		case n.Op == "Where" && len(n.In) == 3 && prod[n.In[0]] != nil && prod[n.In[0]].Op == "And":
			pit = n.In[1]
		case n.Op == "Where" && len(n.Out) == 1 && n.Out[0] == "confidence":
			if s := prod[n.In[1]]; s != nil && s.Op == "Sigmoid" {
				voice = s.In[0]
			}
		}
	}
	if feat == "" || pit == "" || voice == "" {
		return "", "", "", fmt.Errorf("pitch: no SwiftF0 CNN in the graph (features %q, logits %q, voicing %q)", feat, pit, voice)
	}
	return feat, pit, voice, nil
}

// OpenCPU loads model.onnx (Prepare's model) into ONNX Runtime on threads threads of the CPU (0
// or 1: the calling thread); the features and the decoding stay on the host.
func OpenCPU(model string, threads int) (*Detector, error) {
	m, err := Load(model)
	if err != nil {
		return nil, err
	}
	g, err := onnx.Load(model)
	if err != nil {
		return nil, err
	}
	feat, pit, voice, err := cutValues(g)
	if err != nil {
		return nil, err
	}
	raw, err := onnxrt.ReadModel(model)
	if err != nil {
		return nil, err
	}
	if raw, err = onnxrt.Cut(raw, []string{feat}, []string{pit, voice}); err != nil {
		return nil, err
	}
	rt, err := onnxrt.Load()
	if err != nil {
		return nil, err
	}
	s, err := rt.Open(raw, onnxrt.Options{Threads: threads})
	if err != nil {
		return nil, err
	}
	return &Detector{Model: m, cpu: &cpuRunner{s: s, feat: feat, pit: pit, voice: voice}}, nil
}

// run is a Runner: in (1, 3, 128, ProgramFrames) with the window's frames first, mask 1 for them.
func (r *cpuRunner) run(in, mask []float32) ([]float32, error) {
	frames := 0
	for _, v := range mask {
		if v != 0 {
			frames++
		}
	}
	x := make([]float32, 3*Bands*frames)
	for c := range 3 * Bands {
		copy(x[c*frames:(c+1)*frames], in[c*ProgramFrames:])
	}
	out, err := r.s.Run(map[string]onnxrt.Tensor{r.feat: {Shape: []int64{1, 3, Bands, int64(frames)}, Data: x}})
	if err != nil {
		return nil, fmt.Errorf("pitch: %w", err)
	}
	p, v := out[r.pit], out[r.voice]
	if len(p.Data) != Bins*frames || len(v.Data) != frames {
		return nil, fmt.Errorf("pitch: the CNN gives %v and %v for %d frames", p.Shape, v.Shape, frames)
	}
	y := make([]float32, (Bins+1)*ProgramFrames)
	if n := len(p.Shape); n >= 2 && p.Shape[n-1] == Bins && (p.Shape[n-2] == int64(frames) || frames == 1) {
		// (1, frames, 95): the graph lays the logits out frame by frame for its ArgMax
		for t := range frames {
			for b := range Bins {
				y[b*ProgramFrames+t] = p.Data[t*Bins+b]
			}
		}
	} else {
		for b := range Bins {
			copy(y[b*ProgramFrames:], p.Data[b*frames:(b+1)*frames])
		}
	}
	copy(y[Bins*ProgramFrames:], v.Data)
	return y, nil
}
