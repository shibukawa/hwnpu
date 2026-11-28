package binview

import (
	"slices"
	"testing"
)

func TestEncodeDecodeAndView(t *testing.T) {
	in := []float32{1.5, -2.25, 0, 42.125}
	b := EncodeLE(in)
	got := DecodeLE[float32](b)
	if !slices.Equal(got, in) {
		t.Fatalf("got %v, want %v", got, in)
	}

	aligned := make([]uint64, 4)
	bytes := EncodeLE([]int32{10, -20, 30, -40})
	buf := EncodeLE(aligned)
	copy(buf, bytes)
	if v, ok := NativeLEView[int32](buf[:len(bytes)]); NativeLE && (!ok || !slices.Equal(v, []int32{10, -20, 30, -40})) {
		t.Fatalf("view ok=%v v=%v", ok, v)
	}
	if _, ok := NativeLEView[int32](buf[1 : 1+len(bytes)]); ok {
		t.Fatal("expected unaligned slice to fail NativeLEView")
	}
}

func TestConvert(t *testing.T) {
	got := Convert[int64]([]int{1, -2, 3})
	if !slices.Equal(got, []int64{1, -2, 3}) {
		t.Fatalf("got %v", got)
	}
}
