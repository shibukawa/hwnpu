package mil

import (
	"fmt"
	"testing"
)

func TestTextGroups(t *testing.T) {
	for _, c := range []struct {
		layers, per int
		want        string
	}{
		{26, 6, "[[0 6] [6 5] [11 5] [16 5] [21 5]]"},
		{26, 13, "[[0 13] [13 13]]"},
		{4, 6, "[[0 4]]"},
	} {
		if got := fmt.Sprint(TextGroups(c.layers, c.per)); got != c.want {
			t.Errorf("%d by %d: %s, want %s", c.layers, c.per, got, c.want)
		}
	}
}

// TestVaeStages checks that each stage's input is the previous stage's output: the channels and
// the size after its upsamplings; and that Core AI's stages run the same modules in the same
// order.
func TestVaeStages(t *testing.T) {
	var ops [2][]VaeOp
	for k, coreAI := range []bool{false, true} {
		st := VaeStages(1024, 768, coreAI)
		C, H, W := 32, 768/32, 1024/32
		for i, s := range st {
			if s.C != C || s.H != H || s.W != W {
				t.Fatalf("coreAI %v: stage %d takes %dx%dx%d, the one before gives %dx%dx%d", coreAI, i, s.C, s.H, s.W, C, H, W)
			}
			for _, op := range s.Ops {
				ops[k] = append(ops[k], op)
				switch op.Kind {
				case "in":
					C = VaeChannels[len(VaeChannels)-1]
				case "up":
					if op.In != C {
						t.Fatalf("stage %d: %s takes %d channels of %d", i, op.Path, op.In, C)
					}
					C, H, W = op.Out, 2*H, 2*W
				case "head":
					C = 3
				}
			}
		}
		if C != 3 || H != 768 || W != 1024 {
			t.Errorf("coreAI %v: the decoder gives %dx%dx%d", coreAI, C, H, W)
		}
	}
	if fmt.Sprint(ops[0]) != fmt.Sprint(ops[1]) {
		t.Errorf("Core AI's stages run\n%v\nnot\n%v", ops[1], ops[0])
	}
}
