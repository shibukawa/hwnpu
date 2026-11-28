package tensorops

import (
	"slices"
	"testing"
)

func TestPermuteAndTile(t *testing.T) {
	src := []float32{1, 2, 3, 4, 5, 6} // 2x3
	outShape, srcSt := TransposePlan([]int{2, 3}, []int{1, 0})
	dst := make([]float32, Numel(outShape))
	Permute(dst, src, 0, outShape, srcSt)
	if !slices.Equal(outShape, []int{3, 2}) || !slices.Equal(dst, []float32{1, 4, 2, 5, 3, 6}) {
		t.Fatalf("got shape=%v dst=%v", outShape, dst)
	}

	tiled := make([]float32, 4)
	Tile(tiled, []float32{7, 8}, []int{1, 2}, []int{2, 2})
	if !slices.Equal(tiled, []float32{7, 8, 7, 8}) {
		t.Fatalf("got tiled=%v", tiled)
	}
}

func TestConcatAndPad(t *testing.T) {
	got := Concat(make([]int, 0, 6), [][]int{{1, 2, 3, 4}, {5, 6}}, []int{2, 1}, 2, 1)
	if !slices.Equal(got, []int{1, 2, 5, 3, 4, 6}) {
		t.Fatalf("concat=%v", got)
	}

	pads := []int{1, 1}
	sh := PadShape([]int{3}, pads)
	dst := make([]int, Numel(sh))
	Pad(dst, []int{10, 20, 30}, []int{3}, pads, "reflect", 0)
	if !slices.Equal(dst, []int{20, 10, 20, 30, 20}) {
		t.Fatalf("pad reflect=%v", dst)
	}
	Pad(dst, []int{10, 20, 30}, []int{3}, pads, "replicate", 0)
	if !slices.Equal(dst, []int{10, 10, 20, 30, 30}) {
		t.Fatalf("pad replicate=%v", dst)
	}
	Pad(dst, []int{10, 20, 30}, []int{3}, pads, "constant", -1)
	if !slices.Equal(dst, []int{-1, 10, 20, 30, -1}) {
		t.Fatalf("pad constant=%v", dst)
	}
}
