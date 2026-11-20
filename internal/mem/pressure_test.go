package mem

import (
	"runtime"
	"testing"
)

func TestPressureNow(t *testing.T) {
	p, ok := PressureNow()
	if !ok {
		if runtime.GOOS == "darwin" {
			t.Fatal("kern.memorystatus_level not read")
		}
		t.Skip("not read here")
	}
	if p.Free < 0 || p.Free > 100 || p.Level < 1 {
		t.Fatalf("%+v", p)
	}
	t.Logf("%+v", p)
}
