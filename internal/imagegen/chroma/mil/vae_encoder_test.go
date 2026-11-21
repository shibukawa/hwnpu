package mil

import (
	"fmt"
	"testing"
)

// The encoder's stages follow one another (each stage's input is what the one before gives),
// hold every module of diffusers' Encoder once, and are as many at every size (an export's
// parts do not depend on the size).
func TestVaeEncStages(t *testing.T) {
	v := Vae{Channels: []int{128, 256, 512, 512}, Layers: 2, Groups: 32, Latent: 32, Quant: true}
	n := -1
	for _, sz := range [][2]int{{512, 512}, {1024, 1024}, {1152, 768}, {704, 1280}} {
		st := VaeEncStages(v, sz[0], sz[1])
		if n >= 0 && len(st) != n {
			t.Errorf("%v: %d stages, %d at another size", sz, len(st), n)
		}
		n = len(st)
		C, H, W := 3, sz[1], sz[0]
		seen := map[string]bool{}
		for i, s := range st {
			if s.C != C || s.H != H || s.W != W {
				t.Fatalf("%v: stage %d takes (%d, %d, %d), the one before gives (%d, %d, %d)", sz, i, s.C, s.H, s.W, C, H, W)
			}
			for _, op := range s.Ops {
				seen[op.Kind+" "+op.Path] = true
				switch op.Kind {
				case "in", "res":
					C = op.Out
				case "down":
					H, W = (H+1)/2, (W+1)/2
				case "head":
					C = v.Latent
				}
			}
		}
		if C != v.Latent || H != sz[1]/8 || W != sz[0]/8 {
			t.Errorf("%v: the encoder gives (%d, %d, %d)", sz, C, H, W)
		}
		var want []string
		want = append(want, "in vae.encoder.conv_in")
		for b := 0; b < 4; b++ {
			for r := 0; r < 2; r++ {
				want = append(want, fmt.Sprintf("res vae.encoder.down_blocks.%d.resnets.%d", b, r))
			}
			if b < 3 {
				want = append(want, fmt.Sprintf("down vae.encoder.down_blocks.%d.downsamplers.0.conv", b))
			}
		}
		want = append(want, "res vae.encoder.mid_block.resnets.0", "attn vae.encoder.mid_block.attentions.0", "res vae.encoder.mid_block.resnets.1", "head ")
		for _, w := range want {
			if !seen[w] {
				t.Errorf("%v: no %s", sz, w)
			}
		}
		if len(seen) != len(want) {
			t.Errorf("%v: %d modules, want %d", sz, len(seen), len(want))
		}
	}
	t.Logf("%d stages", n)
}
