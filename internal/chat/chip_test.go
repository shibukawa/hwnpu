package chat

import (
	"os"
	"strings"
	"testing"
)

// TestMain picks models as on a chip that no definition leaves out, whatever this machine is;
// TestPickNotOn sets the chip itself.
func TestMain(m *testing.M) {
	chipName = func() string { return "" }
	os.Exit(m.Run())
}

func TestChipOf(t *testing.T) {
	for brand, want := range map[string]string{
		"Apple M1": "M1", "Apple M1 Pro": "M1", "Apple M2": "M2", "Apple M3 Max": "M3", "Apple M4": "M4",
		"Intel(R) Core(TM) Ultra 5 135U": "", "Apple M9": "", "": "",
	} {
		if got := chipOf(brand); got != want {
			t.Errorf("chipOf(%q) = %q, want %q", brand, got, want)
		}
	}
}

// PLaMo 3 Translate 8B is not picked on an M1 (not_on), by its family or a use: the 2B is. Its
// tag still runs it.
func TestPickNotOn(t *testing.T) {
	defer func(f func() string) { chipName = f }(chipName)
	for _, c := range []struct {
		chip, name string
		machine    int
		want       string
	}{
		{"M1", "plamo3translate", 16, "plamo-3-translate-2b"},
		{"M1", "plamo3translate", 64, "plamo-3-translate-2b"},
		{"M1", "translate", 16, "plamo-3-translate-2b"},
		{"M1", "translate", 64, "plamo-3-translate-2b"},
		{"M1", "translate", 8, "plamo-3-translate-0.8b"},
		{"M1", "plamo3translate:16g", 8, "plamo-3-translate-2b"},
		{"M1", "plamo3translate:8b", 16, "plamo-3-translate-8b"},
		{"M2", "plamo3translate", 16, "plamo-3-translate-8b"},
		{"M3", "translate", 16, "plamo-3-translate-8b"},
	} {
		chipName = func() string { return c.chip }
		m, why, ok, err := Pick(c.name, c.machine)
		if err != nil || !ok || m.Name != c.want {
			t.Errorf("%s: Pick(%q, %d) = %s (%s), %v; want %s", c.chip, c.name, c.machine, m.Name, why, err, c.want)
		}
		if c.chip == "M1" && c.name == "plamo3translate" && !strings.Contains(why, "not plamo3translate:8b on an M1") {
			t.Errorf("why: %s", why)
		}
	}
}
