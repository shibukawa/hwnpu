package iron

import "testing"

// A sliding layer of an iron_window export holds a prompt chunk's rows past the context
// (cache_rows): its row is its bytes over those, not over max_seq_len.
func TestCachesRows(t *testing.T) {
	const L, chunk, row = 12288, 2048, 1024
	bufs := map[string]Placement{
		"model.layers.2.cache.keys": {Argument: "scratch", Bytes: L * row},
		"model.layers.4.cache.keys": {Argument: "scratch", Offset: L * row, Bytes: (L + chunk) * row},
	}
	m := &Model{Manifest: &Manifest{}, decode: &version{m: &Version{Buffers: bufs}}}
	m.Manifest.Config.MaxSeqLen = L
	if _, err := m.caches(); err == nil {
		t.Fatal("a cache of more rows than the context passed without cache_rows")
	}
	m.Manifest.CacheRows = map[string]int{"model.layers.4.cache.keys": L + chunk}
	cs, err := m.caches()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cs {
		if c.row != row {
			t.Errorf("%s: row %d bytes, want %d", c.name, c.row, row)
		}
	}
	if got := m.RowBytes(); got != 2*row {
		t.Errorf("RowBytes %d, want %d", got, 2*row)
	}
}
