package vecops

import (
	"math/rand"
	"testing"
)

func TestDotI8(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 7, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 80, 96, 112, 127, 128, 191, 1024, 2560} {
		a := make([]int8, n)
		b := make([]int8, n)
		var want int32
		for i := range n {
			a[i] = int8(rng.Intn(256) - 128)
			b[i] = int8(rng.Intn(256) - 128)
			want += int32(a[i]) * int32(b[i])
		}
		if got := DotI8(a, b); got != want {
			t.Fatalf("n=%d: DotI8=%d, want %d", n, got, want)
		}
	}
	// Extremes: -128 * -128 = +16384, and mixed -128 pairs that catch VPMADDUBSW saturation
	// or VPSIGNB -128 wrap-around bugs.
	for _, pair := range [][2]int8{{-128, -128}, {-128, -1}, {-1, -128}, {-128, 127}, {127, -128}, {127, 127}} {
		for _, n := range []int{16, 32, 64, 128, 143} {
			a := make([]int8, n)
			b := make([]int8, n)
			for i := range n {
				a[i], b[i] = pair[0], pair[1]
			}
			want := int32(n) * int32(pair[0]) * int32(pair[1])
			if got := DotI8(a, b); got != want {
				t.Fatalf("extremes (%d,%d) n=%d: DotI8=%d, want %d", pair[0], pair[1], n, got, want)
			}
		}
	}
}

func BenchmarkDotI8(b *testing.B) {
	a := make([]int8, 1024)
	c := make([]int8, 1024)
	for i := range a {
		a[i] = int8(i%251 - 125)
		c[i] = int8((i*7)%251 - 125)
	}
	for _, kern := range []bool{false, true} {
		b.Run(map[bool]string{false: "go", true: "simd"}[kern], func(b *testing.B) {
			saved := haveDotI8
			haveDotI8 = haveDotI8 && kern
			defer func() { haveDotI8 = saved }()
			b.SetBytes(int64(len(a)))
			var sink int32
			for b.Loop() {
				sink += DotI8(a, c)
			}
			_ = sink
		})
	}
}
