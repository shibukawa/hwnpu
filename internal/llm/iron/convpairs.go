package iron

import "fmt"

// ConvPair is a folded Gated DeltaNet layer's two homes of its short convolution's last Rows
// rows (tools/amd/iron_qwen35.py's hwnpu_conv_pairs): the decode step's, the pair of floats
// ahead of each of Columns runs of the state (Run floats each; each of the pair's two halves a
// copy of the column's rows, Seg bf16 a row, packed, zero after), and the prompt chunk's, the
// convolution's own state (Rows by Columns * Seg bf16). Both are checkpoints of slots alike.
type ConvPair struct {
	State   string `json:"state"`
	Conv    string `json:"conv"`
	Columns int    `json:"columns"`
	Seg     int    `json:"seg"`
	Rows    int    `json:"rows"`
	Pair    int    `json:"pair"`
	Run     int    `json:"run"`
}

// slotBytes is checkpoint buffer name's slot s in v: its bytes, synced from the device.
func (m *Model) slotBytes(v *version, name string, s int) ([]byte, Region, error) {
	p, ok := v.m.Buffers[name]
	if !ok {
		return nil, Region{}, fmt.Errorf("iron: no buffer %s", name)
	}
	slots := 0
	for _, c := range m.Manifest.Checkpoints {
		if c.Buffer == name {
			slots = c.Slots
		}
	}
	if slots == 0 {
		return nil, Region{}, fmt.Errorf("iron: %s is no checkpoint", name)
	}
	size := p.Bytes / slots
	r := Region{Name: name, Argument: p.Argument, Offset: p.Offset + s*size, Bytes: size}
	b, err := m.viewRegion(v, r, size)
	return b, r, err
}

// convPairsIn copies each layer's rows before a prompt chunk starting at start from the state's
// pairs (the slot the chunk reads) into the convolution's state (that slot), where the chunk's
// convolution reads them.
func (m *Model) convPairsIn(v *version, start int) error {
	for _, e := range m.Manifest.ConvPairs {
		s := stateSlot(Checkpoint{Chunk: m.Manifest.Config.PrefillChunk}, start)
		st, _, err := m.slotBytes(v, e.State, s)
		if err != nil {
			return err
		}
		cv, cr, err := m.slotBytes(v, e.Conv, s)
		if err != nil {
			return err
		}
		W := e.Columns * e.Seg
		if 4*e.Columns*e.Run > len(st) || 2*e.Rows*W > len(cv) {
			return fmt.Errorf("iron: conv pairs of %s: slots of %d and %d bytes", e.State, len(st), len(cv))
		}
		for c := 0; c < e.Columns; c++ {
			half := st[4*c*e.Run:]
			for k := 0; k < e.Rows; k++ {
				copy(cv[2*(k*W+c*e.Seg):2*(k*W+(c+1)*e.Seg)], half[2*k*e.Seg:2*(k+1)*e.Seg])
			}
		}
		if err := m.buffer(v, cr.Argument).SyncToDevice(cr.Offset, 2*e.Rows*W); err != nil {
			return err
		}
	}
	return nil
}

// convPairsOut copies each layer's rows ending with a prompt chunk (start, rows) from the
// convolution's state (the slot the chunk wrote) into the state's pairs (that slot), where the
// decode step's operator reads them: both halves of each column's pair, zero past the rows.
func (m *Model) convPairsOut(v *version, start, rows int) error {
	for _, e := range m.Manifest.ConvPairs {
		s := (start+rows-1)/m.Manifest.Config.PrefillChunk + 1
		cv, _, err := m.slotBytes(v, e.Conv, s)
		if err != nil {
			return err
		}
		st, sr, err := m.slotBytes(v, e.State, s)
		if err != nil {
			return err
		}
		W, halfBytes := e.Columns*e.Seg, 2*e.Pair // a half is Pair/2 floats, Pair bf16
		for c := 0; c < e.Columns; c++ {
			for h := 0; h < 2; h++ {
				half := st[4*c*e.Run+h*halfBytes : 4*c*e.Run+(h+1)*halfBytes]
				clear(half)
				for k := 0; k < e.Rows; k++ {
					copy(half[2*k*e.Seg:2*(k+1)*e.Seg], cv[2*(k*W+c*e.Seg):2*(k*W+(c+1)*e.Seg)])
				}
			}
			off := sr.Offset + 4*c*e.Run
			if err := m.buffer(v, sr.Argument).SyncToDevice(off, 4*e.Pair); err != nil {
				return err
			}
		}
	}
	return nil
}
