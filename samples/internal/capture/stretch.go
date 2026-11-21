package capture

import "math"

// Stretch changes the speed of speech without changing its pitch (WSOLA): speed 1.5 says it in
// two thirds of the time, 0.8 more slowly. Windows of 30 ms are laid out every 15 ms of the
// output, each taken from about where the input is at that time, moved by up to 10 ms to where it
// best continues the window before (so that the voice's periods line up and no beat is heard).
func Stretch(x []float32, rate int, speed float64) []float32 {
	if speed <= 0 || math.Abs(speed-1) < 1e-3 || len(x) == 0 {
		return x
	}
	n := rate * 30 / 1000 // a window
	hs := n / 2           // the output's hop
	tol := rate * 10 / 1000
	if len(x) < 2*n {
		return x
	}
	// silence after the end, so that the last windows reach the end of the speech
	x = append(append(make([]float32, 0, len(x)+2*n), x...), make([]float32, 2*n)...)
	win := make([]float32, n)
	for i := range win {
		win[i] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	outLen := int(float64(len(x)-2*n) / speed)
	out := make([]float32, outLen+n)
	norm := make([]float32, outLen+n)
	prev := 0 // where the window before was taken from
	for k := 0; ; k++ {
		at := k * hs // in the output
		if at >= outLen {
			break
		}
		pos := 0
		if k > 0 {
			// the input where the output is now, moved to continue the window before best
			want := int(float64(at) * speed)
			next := prev + hs // what follows the window before in the input
			lo, hi := max(want-tol, 0), min(want+tol, len(x)-n)
			if lo > hi || next+n > len(x) {
				break
			}
			best, bestPos := math.Inf(-1), lo
			for p := lo; p <= hi; p++ {
				var c float64
				for i := 0; i < n; i += 2 { // every other sample: half the work, the same choice
					c += float64(x[p+i]) * float64(x[next+i])
				}
				if c > best {
					best, bestPos = c, p
				}
			}
			pos = bestPos
		}
		for i := 0; i < n && pos+i < len(x); i++ {
			out[at+i] += x[pos+i] * win[i]
			norm[at+i] += win[i]
		}
		prev = pos
	}
	for i := range out {
		if norm[i] > 1e-3 {
			out[i] /= norm[i]
		}
	}
	return out[:outLen]
}
