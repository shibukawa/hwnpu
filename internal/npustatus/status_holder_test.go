package npustatus

import "testing"

func TestNPUHolder(t *testing.T) {
	ps := func(m ...int64) []Process {
		var out []Process
		for _, x := range m {
			out = append(out, Process{NPUMemory: x})
		}
		return out
	}
	for _, c := range []struct {
		ps   []Process
		want int
	}{
		{ps(0, 0), -1},
		{ps(140e3, 1.7e9, 39e3), 1}, // the input method and Siri beside a model
		{ps(1e9, 0.5e9), -1},        // two models: neither
		{ps(0, 5e6), 1},
	} {
		if got := npuHolder(c.ps); got != c.want {
			t.Errorf("%v: %d, want %d", c.ps, got, c.want)
		}
	}
}
