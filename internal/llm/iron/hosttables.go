package iron

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
)

// HostTable is a table of rows by token kept on the host (tools/amd/iron_hostple.py: Gemma 4's
// per-layer token table, 4.4 GiB for E2B, which the device would hold pinned) instead of in
// the arena. Before each call the host writes the rows of the call's tokens into Buffer, a
// device buffer of Layers parts, each Slots rows of PartBytes: part l of the table row of the
// call's token r goes to row r of part l. The file holds Rows rows of Layers * PartBytes.
type HostTable struct {
	Buffer    string `json:"buffer"`
	File      string `json:"file"`
	Rows      int    `json:"rows"`
	Layers    int    `json:"layers"`
	PartBytes int    `json:"part_bytes"`
	Slots     int    `json:"slots"`
}

type hostTable struct {
	t   HostTable
	f   *os.File
	buf *hwnpu.Buffer
	off int // the buffer's offset in buf
	row []byte
}

// openHostTables opens the tables' files and finds their buffers (in the decode step's
// placement: every version shares the arena).
func (m *Model) openHostTables() error {
	for _, t := range m.Manifest.HostTables {
		p, ok := m.decode.m.Buffers[t.Buffer]
		if !ok {
			return fmt.Errorf("iron: host table %s: no buffer %s", t.File, t.Buffer)
		}
		if t.Layers <= 0 || t.PartBytes <= 0 || t.Slots < m.Manifest.Config.PrefillChunk || p.Bytes < t.Layers*t.Slots*t.PartBytes {
			return fmt.Errorf("iron: host table %s: %d parts of %d rows of %d bytes in a buffer of %d", t.File, t.Layers, t.Slots, t.PartBytes, p.Bytes)
		}
		f, err := os.Open(filepath.Join(m.dir, t.File))
		if err != nil {
			return fmt.Errorf("iron: host table: %w", err)
		}
		ht := &hostTable{t: t, f: f, buf: m.buffer(m.decode, p.Argument), off: p.Offset, row: make([]byte, t.Layers*t.PartBytes)}
		m.hostTables = append(m.hostTables, ht)
		if fi, err := f.Stat(); err != nil || fi.Size() < int64(t.Rows)*int64(len(ht.row)) {
			return fmt.Errorf("iron: host table %s is shorter than %d rows of %d bytes", t.File, t.Rows, len(ht.row))
		}
	}
	return nil
}

func (m *Model) closeHostTables() {
	for _, t := range m.hostTables {
		t.f.Close()
	}
	m.hostTables = nil
}

// fillHostTables writes the rows of ids (a call's tokens, row 0 first) into the tables' buffers.
func (m *Model) fillHostTables(ids []int32) error {
	for _, t := range m.hostTables {
		if len(ids) > t.t.Slots {
			return fmt.Errorf("iron: %d tokens for host table %s of %d rows", len(ids), t.t.File, t.t.Slots)
		}
		dst := t.buf.Bytes()
		pb := t.t.PartBytes
		part := t.t.Slots * pb
		for r, id := range ids {
			if id < 0 || int(id) >= t.t.Rows {
				return fmt.Errorf("iron: token %d outside host table %s of %d rows", id, t.t.File, t.t.Rows)
			}
			if _, err := t.f.ReadAt(t.row, int64(id)*int64(len(t.row))); err != nil {
				return fmt.Errorf("iron: host table %s: %w", t.t.File, err)
			}
			for l := 0; l < t.t.Layers; l++ {
				copy(dst[t.off+l*part+r*pb:][:pb], t.row[l*pb:(l+1)*pb])
			}
		}
		for l := 0; l < t.t.Layers; l++ {
			if err := t.buf.SyncToDevice(t.off+l*part, len(ids)*pb); err != nil {
				return err
			}
		}
	}
	return nil
}
