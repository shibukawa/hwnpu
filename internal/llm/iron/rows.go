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
	joined := map[string]*kvJoin{}
	// by name over every version (a decode step on one array of keys and values, joinKV's, does
	// not name the keys and values a prompt chunk writes): placed alike where several name it
	placed := map[string]Placement{}
	for _, v := range []*version{m.decode, m.prompt, m.placed, m.verify} {
		if v == nil {
			continue
		}
		for name, p := range v.m.Buffers {
			if !strings.Contains(name, ".cache.") {
				continue
			}
			if q, ok := placed[name]; ok && q != p {
				return nil, fmt.Errorf("iron: cache %s is placed differently by a prompt chunk", name)
			}
			placed[name] = p
		}
	}
	for name, p := range placed {
		if prefix, ok := strings.CutSuffix(name, ".kv"); ok {
			// the decode step's own copy of the layer's keys and values (joinKV): not a cache
			// of the sequence's state, which keys and values hold
			j := &kvJoin{}
			j.kv = kvCache{name: name, p: p}
			joined[prefix] = j
			continue
		}
		rows := L
		if r, ok := m.Manifest.CacheRows[name]; ok && r > 0 {
			rows = r
		}
		if p.Bytes%rows != 0 {
			return nil, fmt.Errorf("iron: cache %s of %d bytes is no whole rows of %d positions", name, p.Bytes, rows)
		}
		out = append(out, kvCache{name, p, p.Bytes / rows})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	m.joins = nil
	for prefix, j := range joined {
		for _, c := range out {
			switch c.name {
			case prefix + ".keys":
				j.keys = c
			case prefix + ".values":
				j.values = c
			}
		}
		if j.keys.row == 0 || j.keys.row != j.values.row || j.kv.p.Bytes%(2*j.keys.row) != 0 ||
			j.kv.p.Bytes/(2*j.keys.row) < j.keys.p.Bytes/j.keys.row {
			return nil, fmt.Errorf("iron: %s.kv is no rows of its keys and values", prefix)
		}
		j.kv.row = 2 * j.keys.row
		m.joins = append(m.joins, *j)
	}
	m.kv = out
	return out, nil
}

// kvJoin is a layer's keys and values, and the decode step's array of both (kv: a row the
// position's keys then its values, which tools/amd/iron_gptoss_layer.py's attention reads a
// group's key and value of together). A prompt chunk writes keys and values, a decode step kv:
// joinKV copies a chunk's rows into kv after it, splitKV a decode step's rows back before
// anything reads keys and values (a prompt chunk, ExportRows).
type kvJoin struct {
	kv, keys, values kvCache
}

// joinKV copies positions [from, to) of every joined layer's keys and values into its kv.
func (m *Model) joinKV(from, to int) error {
	if _, err := m.caches(); err != nil || len(m.joins) == 0 || to <= from {
		return err
	}
	for _, j := range m.joins {
		kb, vb, jb := m.buffer(m.decode, j.keys.p.Argument), m.buffer(m.decode, j.values.p.Argument), m.buffer(m.decode, j.kv.p.Argument)
		r := j.keys.row
		if err := kb.SyncFromDevice(j.keys.p.Offset+from*r, (to-from)*r); err != nil {
			return err
		}
		if err := vb.SyncFromDevice(j.values.p.Offset+from*r, (to-from)*r); err != nil {
			return err
		}
		k, v, dst := kb.Bytes()[j.keys.p.Offset:], vb.Bytes()[j.values.p.Offset:], jb.Bytes()[j.kv.p.Offset:]
		for p := from; p < to; p++ {
			copy(dst[2*p*r:(2*p+1)*r], k[p*r:(p+1)*r])
			copy(dst[(2*p+1)*r:(2*p+2)*r], v[p*r:(p+1)*r])
		}
		if err := jb.SyncToDevice(j.kv.p.Offset+2*from*r, 2*(to-from)*r); err != nil {
			return err
		}
	}
	return nil
}

// splitKV copies the rows the decode steps wrote (m.stepped) from every kv back into its keys
// and values.
func (m *Model) splitKV() error {
	from, to := m.stepped[0], m.stepped[1]
	m.stepped = [2]int{}
	if _, err := m.caches(); err != nil || len(m.joins) == 0 || to <= from {
		return err
	}
	for _, j := range m.joins {
		kb, vb, jb := m.buffer(m.decode, j.keys.p.Argument), m.buffer(m.decode, j.values.p.Argument), m.buffer(m.decode, j.kv.p.Argument)
		r := j.keys.row
		if err := jb.SyncFromDevice(j.kv.p.Offset+2*from*r, 2*(to-from)*r); err != nil {
			return err
		}
		k, v, src := kb.Bytes()[j.keys.p.Offset:], vb.Bytes()[j.values.p.Offset:], jb.Bytes()[j.kv.p.Offset:]
		for p := from; p < to; p++ {
			copy(k[p*r:(p+1)*r], src[2*p*r:(2*p+1)*r])
			copy(v[p*r:(p+1)*r], src[(2*p+1)*r:(2*p+2)*r])
		}
		if err := kb.SyncToDevice(j.keys.p.Offset+from*r, (to-from)*r); err != nil {
			return err
		}
		if err := vb.SyncToDevice(j.values.p.Offset+from*r, (to-from)*r); err != nil {
			return err
		}
	}
	return nil
}

// stepAt notes that a decode step wrote position's row of the joined layers' kv.
func (m *Model) stepAt(position int) {
	if m.stepped[1] <= m.stepped[0] {
		m.stepped = [2]int{position, position + 1}
		return
	}
	m.stepped[0], m.stepped[1] = min(m.stepped[0], position), max(m.stepped[1], position+1)
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
	if m.released {
		return nil, errReleased
	}
	cs, err := m.caches()
	if err != nil {
		return nil, err
	}
	if from < 0 || to < from || to > m.Manifest.Config.MaxSeqLen {
		return nil, fmt.Errorf("iron: rows [%d, %d) of %d positions", from, to, m.Manifest.Config.MaxSeqLen)
	}
	if err := m.splitKV(); err != nil {
		return nil, err
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
	if m.released {
		return errReleased
	}
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
	return m.joinKV(from, from+n)
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
		for _, v := range []*version{m.prompt, m.placed, m.verify} {
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
	if m.released {
		return nil, errReleased
	}
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
	if m.released {
		return errReleased
	}
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
