package iron

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"

	"github.com/shibukawa/bonsaiai/internal/vecops"
)

// HostTable is a table of rows by token kept on the host (tools/amd/iron_hostple.py: Gemma 4's
// per-layer token table, 4.4 GiB for E2B, which the device would hold pinned) instead of in
// the arena. Before each call the host writes the rows of the call's tokens into Buffer, a
// device buffer of Layers parts, each Slots rows of PartBytes: part l of the table row of the
// call's token r goes to row r of part l. The file holds Rows rows of Layers parts, each as
// Format stores it.
type HostTable struct {
	Buffer    string `json:"buffer"`
	File      string `json:"file"`
	Rows      int    `json:"rows"`
	Layers    int    `json:"layers"`
	PartBytes int    `json:"part_bytes"`
	Slots     int    `json:"slots"`
	// Format is how the file stores a part: "" the buffer's PartBytes of bf16 as they are;
	// HostTableInt8 an fp16 scale followed by PartBytes/2 int8 values, expanded to bf16 when a
	// row is written (Gemma 4's per-layer embeddings: half the file, KL 0.00005 from bf16,
	// docs/design/performance.md). Older exports have no format.
	Format string `json:"format,omitempty"`
}

// EmbedTable is the buffer of the host table of the token embeddings (tools/amd/iron_hostple.py's
// EMBED): the rows a call reads in place of a gather on the device.
const EmbedTable = "bonsaiai_embed_rows"

// HostTableInt8 is the HostTable.Format of a part stored as a scale and int8 values.
const HostTableInt8 = "int8"

// fileBytes is the bytes of a part in the file.
func (t HostTable) fileBytes() int {
	if t.Format == HostTableInt8 {
		return 2 + t.PartBytes/2
	}
	return t.PartBytes
}

// expandInt8 writes into dst (bf16 values) the part src stores as an fp16 scale and int8 values.
func expandInt8(dst, src []byte) {
	s := vecops.F16Value(binary.LittleEndian.Uint16(src))
	for j, q := range src[2:] {
		binary.LittleEndian.PutUint16(dst[2*j:], bf16(float32(int8(q))*s))
	}
}

type hostTable struct {
	t   HostTable
	f   *os.File
	arg string // the argument (bank) the buffer is in: its device buffer is the model's of the day
	off int    // the buffer's offset in it
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
		if t.Format != "" && t.Format != HostTableInt8 {
			return fmt.Errorf("iron: host table %s: format %q (this build reads %q)", t.File, t.Format, HostTableInt8)
		}
		if t.Layers <= 0 || t.PartBytes <= 0 || t.PartBytes%2 != 0 || t.Slots < m.Manifest.Config.PrefillChunk || p.Bytes < t.Layers*t.Slots*t.PartBytes {
			return fmt.Errorf("iron: host table %s: %d parts of %d rows of %d bytes in a buffer of %d", t.File, t.Layers, t.Slots, t.PartBytes, p.Bytes)
		}
		f, err := os.Open(filepath.Join(m.dir, t.File))
		if err != nil {
			return fmt.Errorf("iron: host table: %w", err)
		}
		ht := &hostTable{t: t, f: f, arg: p.Argument, off: p.Offset, row: make([]byte, t.Layers*t.fileBytes())}
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
		buf := m.buffer(m.decode, t.arg)
		dst := buf.Bytes()
		pb, fb := t.t.PartBytes, t.t.fileBytes()
		part := t.t.Slots * pb
		if m.embedRow != nil && t.t.Buffer == EmbedTable {
			// StepRow's row in place of the token's
			if t.t.Layers != 1 || len(ids) != 1 || len(m.embedRow) != pb {
				return fmt.Errorf("iron: a row of %d bytes for host table %s of %d parts of %d bytes", len(m.embedRow), t.t.File, t.t.Layers, pb)
			}
			copy(dst[t.off:t.off+pb], m.embedRow)
			if err := buf.SyncToDevice(t.off, pb); err != nil {
				return err
			}
			continue
		}
		for r, id := range ids {
			if id < 0 || int(id) >= t.t.Rows {
				return fmt.Errorf("iron: token %d outside host table %s of %d rows", id, t.t.File, t.t.Rows)
			}
			if _, err := t.f.ReadAt(t.row, int64(id)*int64(len(t.row))); err != nil {
				return fmt.Errorf("iron: host table %s: %w", t.t.File, err)
			}
			for l := 0; l < t.t.Layers; l++ {
				d, src := dst[t.off+l*part+r*pb:][:pb], t.row[l*fb:(l+1)*fb]
				if t.t.Format == HostTableInt8 {
					expandInt8(d, src)
				} else {
					copy(d, src)
				}
			}
		}
		for l := 0; l < t.t.Layers; l++ {
			if err := buf.SyncToDevice(t.off+l*part, len(ids)*pb); err != nil {
				return err
			}
		}
	}
	return nil
}

// StepRow runs the decode step at position on row (bf16, the model's hidden size) in place of a
// token's embedding: an export whose token embeddings are the host's (EmbedTable, --q4), which
// reads a row the host writes. LFM3-Audio's speech feeds the sum of an audio frame's codes'
// feedback rows this way (internal/speech).
func (m *Model) StepRow(ctx context.Context, row []byte, position int) ([]float32, error) {
	found := false
	for _, t := range m.hostTables {
		found = found || t.t.Buffer == EmbedTable
	}
	if !found {
		return nil, fmt.Errorf("iron: the export reads its token embeddings on the device (no host table %s): export it with --q4", EmbedTable)
	}
	m.embedRow = row
	defer func() { m.embedRow = nil }()
	return m.Step(ctx, 0, position)
}

// EmbedRow is row id of the export's host table of the token embeddings (EmbedTable, a --q4
// export's embed.bin), as float32: what a tied head's row of id is too (a d1 export reads the
// logits of its answer labels so, from the hidden state its prompt chunk gives).
func (m *Model) EmbedRow(id int32) ([]float32, error) {
	for _, t := range m.hostTables {
		if t.t.Buffer != EmbedTable {
			continue
		}
		if t.t.Layers != 1 || t.t.Format != "" {
			return nil, fmt.Errorf("iron: host table %s is not one bf16 row a token", t.t.File)
		}
		if id < 0 || int(id) >= t.t.Rows {
			return nil, fmt.Errorf("iron: token %d outside host table %s of %d rows", id, t.t.File, t.t.Rows)
		}
		b := make([]byte, t.t.PartBytes)
		if _, err := t.f.ReadAt(b, int64(id)*int64(len(b))); err != nil {
			return nil, fmt.Errorf("iron: host table %s: %w", t.t.File, err)
		}
		out := make([]float32, len(b)/2)
		vecops.BF16ToF32(out, b)
		return out, nil
	}
	return nil, fmt.Errorf("iron: the export has no host table of the token embeddings (%s): export it with --q4", EmbedTable)
}
