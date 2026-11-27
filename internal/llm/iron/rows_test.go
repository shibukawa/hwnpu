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

// A decode step's array of fewer rows than the keys is a ring, when the manifest says so: a
// position's row is its place modulo the ring, so after the rows before a position are joined
// the ring holds the last of them, and a step's rows split back to their positions.
func TestKVRing(t *testing.T) {
	const L, ring, row = 1024, 128, 16
	bufs := map[string]Placement{
		"model.layers.0.cache.keys":   {Argument: "scratch", Bytes: L * row},
		"model.layers.0.cache.values": {Argument: "scratch", Offset: L * row, Bytes: L * row},
		"model.layers.0.cache.kv":     {Argument: "scratch", Offset: 2 * L * row, Bytes: ring * 2 * row},
		"model.layers.1.cache.keys":   {Argument: "scratch", Offset: 3 * L * row, Bytes: L * row},
		"model.layers.1.cache.values": {Argument: "scratch", Offset: 4 * L * row, Bytes: L * row},
		"model.layers.1.cache.kv":     {Argument: "scratch", Offset: 5 * L * row, Bytes: L * 2 * row},
	}
	m := &Model{Manifest: &Manifest{}, decode: &version{m: &Version{Buffers: bufs}}}
	m.Manifest.Config.MaxSeqLen = L
	if _, err := m.caches(); err == nil {
		t.Fatal("a kv of fewer rows than its keys passed without cache_rows")
	}
	m.Manifest.CacheRows = map[string]int{"model.layers.0.cache.kv": ring}
	if _, err := m.caches(); err != nil {
		t.Fatal(err)
	}
	if !m.rings || len(m.joins) != 2 {
		t.Fatalf("rings %v, %d joins", m.rings, len(m.joins))
	}
	for i := range m.joins {
		j := &m.joins[i]
		want := 0
		if j.kv.name == "model.layers.0.cache.kv" {
			want = ring
		}
		if j.ring != want {
			t.Fatalf("%s: ring %d, want %d", j.kv.name, j.ring, want)
		}
		if j.ring == 0 {
			continue
		}
		keys, values := make([]byte, L*row), make([]byte, L*row)
		for p := 0; p < L; p++ {
			for k := 0; k < row; k++ {
				keys[p*row+k], values[p*row+k] = byte(p), byte(p>>8)|0x80
			}
		}
		kv := make([]byte, ring*2*row)
		// the rows before position 300: the ring holds 172 .. 299
		j.joinRows(kv, keys, values, 300-ring, 300)
		for p := 300 - ring; p < 300; p++ {
			s := p % ring
			if kv[2*s*row] != byte(p) || kv[(2*s+1)*row] != byte(p>>8)|0x80 {
				t.Fatalf("position %d is not in slot %d", p, s)
			}
		}
		// steps at 300 .. 309 write their slots; split back, they are the positions' rows
		for p := 300; p < 310; p++ {
			s := p % ring
			for k := 0; k < row; k++ {
				kv[2*s*row+k], kv[(2*s+1)*row+k] = 0x55, 0x66
			}
		}
		j.splitRows(kv, keys, values, 300, 310)
		for p := 290; p < 320; p++ {
			wantK, wantV := byte(p), byte(p>>8)|0x80
			if p >= 300 && p < 310 {
				wantK, wantV = 0x55, 0x66
			}
			if keys[p*row] != wantK || values[p*row+row-1] != wantV {
				t.Fatalf("position %d after the split: %#x %#x", p, keys[p*row], values[p*row+row-1])
			}
		}
	}
}

// The check of drafts' scratch rows (iron_verify.py's rows_o, named under the layer's cache) are
// no cache of the sequence: the decode step and the prompt chunks do not have them.
func TestCachesSkipVerifyScratch(t *testing.T) {
	const L, row = 4096, 4096
	seq := map[string]Placement{
		"model.layers.3.cache.keys":   {Argument: "scratch", Bytes: L * row},
		"model.layers.3.cache.values": {Argument: "scratch", Offset: L * row, Bytes: L * row},
	}
	ver := map[string]Placement{"model.layers.3.cache.rows_o": {Argument: "scratch", Offset: 2 * L * row, Bytes: 3 * row}}
	for n, p := range seq {
		ver[n] = p
	}
	m := &Model{Manifest: &Manifest{CacheRows: map[string]int{"model.layers.3.cache.rows_o": 3}},
		decode: &version{m: &Version{Buffers: seq}}, verify: &version{m: &Version{Buffers: ver}}}
	m.Manifest.Config.MaxSeqLen = L
	cs, err := m.caches()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cs {
		if c.name == "model.layers.3.cache.rows_o" {
			t.Errorf("the check's scratch %s taken for a cache", c.name)
		}
	}
	if len(cs) != 2 || m.RowBytes() != 2*row {
		t.Errorf("%d caches, RowBytes %d; want 2 and %d", len(cs), m.RowBytes(), 2*row)
	}
}
