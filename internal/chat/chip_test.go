package chat

import (
	"os"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/catalog"
)

// TestMain picks models as on a chip that no definition leaves out, whatever this machine is
// (TestPickNotOn sets the chip itself), and keeps what the tests download out of this machine's
// cache (a test of its own sets HWNPU_CACHE again).
func TestMain(m *testing.M) {
	catalog.ChipName = func() string { return "" }
	dir, err := os.MkdirTemp("", "hwnpu-chat-test")
	if err != nil {
		panic(err)
	}
	os.Setenv("HWNPU_CACHE", dir)
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// cacheHere gives the test a cache of its own (HWNPU_CACHE): a machine of its own.
func cacheHere(t *testing.T) string {
	t.Helper()
	d := t.TempDir()
	t.Setenv("HWNPU_CACHE", d)
	clearCommits()
	t.Cleanup(clearCommits)
	return d
}

// PLaMo 3 Translate 8B is not picked on an M1 (not_on), by its family or a use: the 2B is. Its
// tag still runs it.
func TestPickNotOn(t *testing.T) {
	defer func(f func() string) { catalog.ChipName = f }(catalog.ChipName)
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
		// the family has no other: still picked, and Prepare refuses it there (a MoE model)
		{"M1", "lfm2", 16, "LFM2-8B-A1B"},
		{"M2", "lfm2", 16, "LFM2-8B-A1B"},
	} {
		catalog.ChipName = func() string { return c.chip }
		m, why, ok, err := Pick(c.name, c.machine)
		if err != nil || !ok || m.Name != c.want {
			t.Errorf("%s: Pick(%q, %d) = %s (%s), %v; want %s", c.chip, c.name, c.machine, m.Name, why, err, c.want)
		}
		if c.chip == "M1" && c.name == "plamo3translate" && !strings.Contains(why, "not plamo3translate:8b on an M1") {
			t.Errorf("why: %s", why)
		}
	}
}

// An export made as a chip converts is prebuilt for that chip's architectures of the Neural
// Engine, the M2's standing for the later chips' too.
func TestTakesExportsOf(t *testing.T) {
	for _, c := range []struct {
		arch, chip string
		want       bool
	}{
		{"h13g", "M1", true}, {"h13s", "M1", true}, {"h13g", "M2", false},
		{"h14g", "M2", true}, {"h14g", "M1", false},
		{"h15g", "M2", true}, {"h15g", "M1", false}, {"h15g", "M3", true},
		{"h18g", "M2", true}, {"h17p", "M1", false},
	} {
		if got := TakesExportsOf(c.arch, c.chip); got != c.want {
			t.Errorf("%s takes the exports of an %s: %v, want %v", c.arch, c.chip, got, c.want)
		}
	}
}
