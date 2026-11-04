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
// the sequence before to, together with StateAt for a model with recurrent states.
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

// Resumable reports whether a sequence can go back to any position: the state is the KV caches
// written by position (Restartable), or besides them recurrent states whose slot a position
// reads can be copied (StateAt), with no ring caches.
func (m *Model) Resumable() bool {
	if m.Manifest.Restartable {
		return true
	}
	if len(m.Manifest.Checkpoints) == 0 || m.Manifest.Rings > 0 {
		return false
	}
	// every version places the states alike (one arena), as the caches
	for _, c := range m.Manifest.Checkpoints {
		p, ok := m.decode.m.Buffers[c.Buffer]
		if !ok || c.Slots < 1 || c.Chunk < 1 || p.Bytes%c.Slots != 0 {
			return false
		}
		for _, v := range []*version{m.prompt, m.placed} {
			if v != nil && v.m.Buffers[c.Buffer] != p {
				return false
			}
		}
	}
	return true
}

// stateSlot is the slot of a recurrent state that a call starting at pos reads: 0 before the
// sequence, else the one its last chunk boundary's calls write (IRON's Checkpoints.at).
func stateSlot(c Checkpoint, pos int) int {
	if pos == 0 {
		return 0
	}
	return (pos-1)/c.Chunk + 1
}

// StateAt copies what a call starting at pos reads of the recurrent states, in their manifest
// order: with the rows before pos, it is the whole state of the sequence there. It is good while
// the positions before pos are not run again; the calls after pos change only that slot and
// later ones. Empty for a model without recurrent states.
func (m *Model) StateAt(pos int) ([]byte, error) {
	var out []byte
	for _, c := range m.Manifest.Checkpoints {
		p, ok := m.decode.m.Buffers[c.Buffer]
		if !ok {
			return nil, fmt.Errorf("iron: no buffer %s of a recurrent state", c.Buffer)
		}
		size := p.Bytes / c.Slots
		off := p.Offset + stateSlot(c, pos)*size
		buf := m.buffer(m.decode, p.Argument)
		if err := buf.SyncFromDevice(off, size); err != nil {
			return nil, err
		}
		out = append(out, buf.Bytes()[off:off+size]...)
	}
	return out, nil
}

// SetStateAt writes back what StateAt copied at pos, so that the next call can start there.
func (m *Model) SetStateAt(pos int, state []byte) error {
	o := 0
	for _, c := range m.Manifest.Checkpoints {
		p, ok := m.decode.m.Buffers[c.Buffer]
		if !ok {
			return fmt.Errorf("iron: no buffer %s of a recurrent state", c.Buffer)
		}
		size := p.Bytes / c.Slots
		if o+size > len(state) {
			return fmt.Errorf("iron: %d bytes of state, the recurrent states take more", len(state))
		}
		off := p.Offset + stateSlot(c, pos)*size
		buf := m.buffer(m.decode, p.Argument)
		copy(buf.Bytes()[off:off+size], state[o:o+size])
		if err := buf.SyncToDevice(off, size); err != nil {
			return err
		}
		o += size
	}
	if o != len(state) {
		return fmt.Errorf("iron: %d bytes of state, the recurrent states take %d", len(state), o)
	}
	return nil
}
