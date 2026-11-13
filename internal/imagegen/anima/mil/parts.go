package mil

import "github.com/shibukawa/hwnpu/internal/imagegen/milpart"

// DitScale is what the DiT's residual stream is divided by in its programs: its largest value
// measured (2.7e5, blocks 12 to 20 at 512 x 512, design.md 16.69) becomes 8.4e3, and the
// largest gated branch (2.3e5) 7.3e3.
const DitScale = 32

// Part is one program of Anima (milpart.Part).
type Part = milpart.Part

type graph = milpart.Graph

func part(name string, body func(g *graph)) *Part {
	return &Part{Name: name, Model: "anima", Body: body}
}
