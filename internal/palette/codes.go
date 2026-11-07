package palette

import (
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
)

// Codes are lookup tables and indices chosen against the activations (GPTQ, by
// tools/amd/q4_gptq.py with --fmt-q4, --fmt-q8 and --head lutN), for the weights they were
// chosen for: rounding each weight to the nearest entry of its table keeps the weight error
// small, not the layer's output error. A directory holds index.json (by key: the format, the
// shape and the rows a table) and, for each weight, KEY.idx (an index a weight, uint8, row
// major) and KEY.tab (the tables, rows/group by 2^bits float32 of fp16 values, little endian).
// The key is WeightKey of the weight as the checkpoint holds it.
type Codes struct {
	dir   string
	index map[string]codeEntry
	mu    sync.Mutex
	hits  int
}

type codeEntry struct {
	Name  string `json:"name"`
	Fmt   string `json:"fmt"`
	Shape []int  `json:"shape"`
	Group int    `json:"group"`
}

// LoadCodes reads the index of a codes directory; the codes of other formats (q4_gptq.py's
// q4nx blocks for AMD) are left out.
func LoadCodes(dir string) (*Codes, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "index.json"))
	if err != nil {
		return nil, err
	}
	var all map[string]codeEntry
	if err := json.Unmarshal(raw, &all); err != nil {
		return nil, fmt.Errorf("palette: %s: %v", dir, err)
	}
	c := &Codes{dir: dir, index: map[string]codeEntry{}}
	for k, e := range all {
		if lutBits(e.Fmt) != 0 {
			c.index[k] = e
		}
	}
	return c, nil
}

// Len is the number of weights with codes.
func (c *Codes) Len() int { return len(c.index) }

// Hits is the number of weights Lookup has found codes for.
func (c *Codes) Hits() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits
}

func lutBits(format string) int {
	switch format {
	case "lut4":
		return 4
	case "lut6":
		return 6
	case "lut8":
		return 8
	}
	return 0
}

// WeightKey names a rows x cols weight among codes: the SHA-1 of its shape as Python writes a
// tuple, "(rows, cols)", and its values rounded to bf16 (to nearest even), little endian. It is
// iron_q4.weight_key's, so a weight loaded from the checkpoint has the same key in both.
func WeightKey(w []float32, rows, cols int) string {
	h := sha1.New()
	fmt.Fprintf(h, "(%d, %d)", rows, cols)
	buf := make([]byte, 2*len(w))
	for i, v := range w {
		binary.LittleEndian.PutUint16(buf[2*i:], bf16(v))
	}
	h.Write(buf)
	return hex.EncodeToString(h.Sum(nil))
}

func bf16(f float32) uint16 {
	x := math.Float32bits(f)
	if x&0x7fffffff > 0x7f800000 { // NaN
		return uint16(x>>16) | 0x40
	}
	return uint16((x + 0x7fff + (x>>16)&1) >> 16)
}

// Lookup returns the palettized weight for w (rows x cols, as the checkpoint holds it) when the
// codes have it in bits wide tables of group rows by whole rows, with the tables multiplied by
// scale (what the graph multiplies the weight by: a table scales with its weights). ok is false
// when there are none, or for another layout; the caller then quantizes w itself.
func (c *Codes) Lookup(w []float32, rows, cols, group, colBlock, bits int, scale float32) (LUT4, bool, error) {
	if c == nil || len(c.index) == 0 {
		return LUT4{}, false, nil
	}
	if colBlock > 0 && colBlock != cols {
		return LUT4{}, false, nil
	}
	key := WeightKey(w, rows, cols)
	e, ok := c.index[key]
	if !ok || lutBits(e.Fmt) != bits || e.Group != group || len(e.Shape) != 2 || e.Shape[0] != rows || e.Shape[1] != cols {
		return LUT4{}, false, nil
	}
	k := 1 << bits
	idx, err := os.ReadFile(filepath.Join(c.dir, key+".idx"))
	if err != nil {
		return LUT4{}, false, err
	}
	tab, err := os.ReadFile(filepath.Join(c.dir, key+".tab"))
	if err != nil {
		return LUT4{}, false, err
	}
	if len(idx) != rows*cols || len(tab) != rows/group*k*4 {
		return LUT4{}, false, fmt.Errorf("palette: codes of %s (%s): %d indices and %d table bytes for %d x %d in %s",
			e.Name, key, len(idx), len(tab), rows, cols, e.Fmt)
	}
	for i, v := range idx {
		if int(v) >= k {
			return LUT4{}, false, fmt.Errorf("palette: codes of %s: index %d at %d of a %d entry table", e.Name, v, i, k)
		}
	}
	out := LUT4{Rows: rows, Cols: cols, Group: group, ColBlock: cols, Bits: bits, LUT: make([]uint16, rows/group*k)}
	for i := range out.LUT {
		out.LUT[i] = f16(math.Float32frombits(binary.LittleEndian.Uint32(tab[4*i:])) * scale)
	}
	out.Indices = pack(idx, bits)
	c.mu.Lock()
	c.hits++
	c.mu.Unlock()
	return out, true, nil
}
