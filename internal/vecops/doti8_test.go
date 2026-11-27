package vecops

import (
	"math/rand"
	"testing"
)

func TestDotI8(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 7, 15, 16, 63, 64, 65, 127, 128, 191, 1024, 2560} {
		a := make([]int8, n)
		b := make([]int8, n)
		var want int32
		for i := range n {
			a[i] = int8(rng.Intn(255) - 127)
			b[i] = int8(rng.Intn(255) - 127)
			want += int32(a[i]) * int32(b[i])
		}
		if got := DotI8(a, b); got != want {
			t.Fatalf("n=%d: DotI8=%d, want %d", n, got, want)
		}
	}
	// Extremes (-128 * -128 = +16384 per element)
	a := make([]int8, 128)
	b := make([]int8, 128)
	for i := range a {
		a[i], b[i] = -128, -128
	}
	if got, want := DotI8(a, b), int32(128*16384); got != want {
		t.Fatalf("extremes: DotI8=%d, want %d", got, want)
	}
}

func BenchmarkDotI8(b *testing.B) {
	a := make([]int8, 1024)
	c := make([]int8, 1024)
	for i := range a {
		a[i] = int8(i%251 - 125)
		c[i] = int8((i*7)%251 - 125)
	}
	b.SetBytes(int64(len(a)))
	var sink int32
	for b.Loop() {
		sink += DotI8(a, c)
	}
	_ = sink
}
