package iron

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestWordsMatchIRON compares the scratchpad words this package writes with the ones IRON
// computed at export time for the same calls (check/words.json of HWNPU_IRON_EXPORT).
func TestWordsMatchIRON(t *testing.T) {
	dir := os.Getenv("HWNPU_IRON_EXPORT")
	if dir == "" {
		t.Skip("HWNPU_IRON_EXPORT not set")
	}
	m, err := ReadManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "check", "words.json"))
	if err != nil {
		t.Fatal(err)
	}
	var calls map[string][]struct {
		Values   map[string]int64    `json:"values"`
		Commands []map[string]uint32 `json:"commands"`
	}
	if err := json.Unmarshal(b, &calls); err != nil {
		t.Fatal(err)
	}
	bad := 0
	for name, cs := range calls {
		v := m.Versions[name]
		for _, c := range cs {
			for k, want := range c.Commands {
				sp := make([]byte, 4*64)
				if err := writeWords(sp, v.Kernels[k].Parameters, c.Values); err != nil {
					t.Errorf("%s %v command %d: %v", name, c.Values, k, err)
					continue
				}
				for slot, w := range want {
					var i int
					json.Unmarshal([]byte(slot), &i)
					if got := binary.LittleEndian.Uint32(sp[4*i:]); got != w {
						bad++
						if bad < 20 {
							for _, p := range v.Kernels[k].Parameters {
								if p.Index == i {
									t.Errorf("%s %v command %d slot %d (%s): %d, IRON %d", name, c.Values, k, i, p.Name, got, w)
								}
							}
						}
					}
				}
			}
		}
	}
	if bad > 0 {
		t.Fatalf("%d words differ", bad)
	}
}
