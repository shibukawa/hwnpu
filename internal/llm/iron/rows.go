package iron

import (
	"fmt"
	"sort"
	"strings"
)

// kvCache is one layer's keys or values: (max_seq_len, groups, head_dim), a row per position.
type kvCache struct {
	name string
	p    Placement
	row  int // bytes per position
}

// caches lists the KV caches, by name: the buffers whose name holds ".cache.". Every version
// places them alike (one arena); Open checks it.
func (m *Model) caches() ([]kvCache, error) {
	if m.kv != nil {
		return m.kv, nil
	}
	L := m.Manifest.Config.MaxSeqLen
	var out []kvCache
	for name, p := range m.decode.m.Buffers {
		if !strings.Contains(name, ".cache.") {
			continue
		}
		if p.Bytes%L != 0 {
			return nil, fmt.Errorf("iron: cache %s of %d bytes is no whole rows of %d positions", name, p.Bytes, L)
		}
		for _, v := range []*version{m.prompt, m.placed} {
			if v == nil {
				continue
			}
			if q, ok := v.m.Buffers[name]; !ok || q != p {
				return nil, fmt.Errorf("iron: cache %s is placed differently by a prompt chunk", name)
			}
		}
		out = append(out, kvCache{name, p, p.Bytes / L})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	m.kv = out
	return out, nil
}

// RowBytes is the bytes of the KV caches' rows of one position.
func (m *Model) RowBytes() int {
	cs, err := m.caches()
	if err != nil {
		return 0
	}
	n := 0
	for _, c := range cs {
		n += c.row
	}
	return n
}

// ExportRows copies the KV caches' rows of positions [from, to): position by position, the
// caches by name. The caches are written by position, so these rows are the whole state of
// the sequence before to (a model with recurrent state is refused at export).
func (m *Model) ExportRows(from, to int) ([]byte, error) {
	cs, err := m.caches()
	if err != nil {
		return nil, err
	}
	if from < 0 || to < from || to > m.Manifest.Config.MaxSeqLen {
		return nil, fmt.Errorf("iron: rows [%d, %d) of %d positions", from, to, m.Manifest.Config.MaxSeqLen)
	}
	rb := m.RowBytes()
	out := make([]byte, (to-from)*rb)
	o := 0
	for _, c := range cs {
		buf := m.buffer(m.decode, c.p.Argument)
		if err := buf.SyncFromDevice(c.p.Offset+from*c.row, (to-from)*c.row); err != nil {
			return nil, err
		}
		src := buf.Bytes()[c.p.Offset:]
		for p := from; p < to; p++ {
			copy(out[(p-from)*rb+o:], src[p*c.row:(p+1)*c.row])
		}
		o += c.row
	}
	return out, nil
}

// ImportRows writes rows (ExportRows's layout) at positions from on.
func (m *Model) ImportRows(from int, rows []byte) error {
	cs, err := m.caches()
	if err != nil {
		return err
	}
	rb := m.RowBytes()
	if rb == 0 || len(rows)%rb != 0 {
		return fmt.Errorf("iron: %d bytes are no whole rows of %d", len(rows), rb)
	}
	n := len(rows) / rb
	if from < 0 || from+n > m.Manifest.Config.MaxSeqLen {
		return fmt.Errorf("iron: %d rows from %d past %d positions", n, from, m.Manifest.Config.MaxSeqLen)
	}
	o := 0
	for _, c := range cs {
		buf := m.buffer(m.decode, c.p.Argument)
		dst := buf.Bytes()[c.p.Offset:]
		for i := 0; i < n; i++ {
			copy(dst[(from+i)*c.row:(from+i+1)*c.row], rows[i*rb+o:i*rb+o+c.row])
		}
		if err := buf.SyncToDevice(c.p.Offset+from*c.row, n*c.row); err != nil {
			return err
		}
		o += c.row
	}
	return nil
}
