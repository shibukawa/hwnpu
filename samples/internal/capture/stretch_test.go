package capture

import (
	"math"
	"testing"
)

// A stretched tone takes the time its speed says and keeps its pitch (as many zero crossings a
// second), with no gaps or jumps in its level.
func TestStretch(t *testing.T) {
	const rate, f = 24000, 220.0
	x := make([]float32, 2*rate)
	for i := range x {
		x[i] = float32(0.5 * math.Sin(2*math.Pi*f*float64(i)/rate))
	}
	for _, speed := range []float64{0.75, 1.25, 1.5, 2} {
		y := Stretch(x, rate, speed)
		if want := int(float64(len(x)) / speed); len(y) != want {
			t.Errorf("speed %g: %d samples, want %d", speed, len(y), want)
		}
		body := y[rate/20 : len(y)-rate/20] // past the first and last windows
		cross := 0
		for i := 1; i < len(body); i++ {
			if (body[i-1] < 0) != (body[i] < 0) {
				cross++
			}
		}
		got := float64(cross) / 2 / (float64(len(body)) / rate)
		if math.Abs(got-f) > 3 {
			t.Errorf("speed %g: a tone of %.1f Hz, want %.0f", speed, got, f)
		}
		peak := 0.0
		for _, v := range body {
			peak = math.Max(peak, math.Abs(float64(v)))
		}
		if peak < 0.45 || peak > 0.55 {
			t.Errorf("speed %g: peak %.2f, want about 0.5", speed, peak)
		}
	}
	if y := Stretch(x, rate, 1); len(y) != len(x) {
		t.Error("speed 1 changed the sound")
	}
}

// The end of the speech is kept: a sound that goes on to the last sample is still there in the
// last 20 ms of what is stretched (the last words were cut off before).
func TestStretchKeepsTheEnd(t *testing.T) {
	const rate = 24000
	x := make([]float32, rate)
	for i := range x {
		x[i] = float32(0.5 * math.Sin(2*math.Pi*300*float64(i)/rate))
	}
	for _, speed := range []float64{0.75, 1.3, 2} {
		y := Stretch(x, rate, speed)
		tail := y[len(y)-rate/50:]
		peak := 0.0
		for _, v := range tail {
			peak = math.Max(peak, math.Abs(float64(v)))
		}
		if peak < 0.3 {
			t.Errorf("speed %g: the last 20 ms peak at %.2f (the end cut off)", speed, peak)
		}
	}
}
