package palette

import (
	"cmp"
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
// major) and KEY.tab (the tables, rows/group by 2^bits float32 of fp16 values, little endian;
// with col_block, rows/group by cols/col_block of them, in row major order). The key is WeightKey
// of the weight as the checkpoint holds it. CodesWriter writes them from Go (tools/coreai/lutfit).
//
// A weight of format int8 (a scale a row, tools/ref/gptoss_attn_gptq.py) has its integers in KEY.idx
// (int8, row major) and its scales in KEY.tab (rows float32 of fp16 values); LookupLinear reads it.
type Codes struct {
	dir   string
	index map[string]codeEntry
	mu    sync.Mutex
	hits  int
}

type codeEntry struct {
	Name     string `json:"name"`
	Fmt      string `json:"fmt"`
	Shape    []int  `json:"shape"`
	Group    int    `json:"group"`
	ColBlock int    `json:"col_block,omitempty"` // columns a table (0: whole rows)
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
		if lutBits(e.Fmt) != 0 || e.Fmt == "int8" {
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
// codes have it in bits wide tables of group rows, with the tables multiplied by scale (what the
// graph multiplies the weight by: a table scales with its weights). The codes' column blocks are
// taken unless colBlock asks for others (0 takes any). ok is false when there are none, or for
// another layout; the caller then quantizes w itself.
func (c *Codes) Lookup(w []float32, rows, cols, group, colBlock, bits int, scale float32) (LUT4, bool, error) {
	if c == nil || len(c.index) == 0 {
		return LUT4{}, false, nil
	}
	key := WeightKey(w, rows, cols)
	e, ok := c.index[key]
	if !ok || lutBits(e.Fmt) != bits || e.Group != group || len(e.Shape) != 2 || e.Shape[0] != rows || e.Shape[1] != cols {
		return LUT4{}, false, nil
	}
	cb := cmp.Or(e.ColBlock, cols)
	if (colBlock > 0 && colBlock != cb) || cols%cb != 0 {
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
	if len(idx) != rows*cols || len(tab) != rows/group*(cols/cb)*k*4 {
		return LUT4{}, false, fmt.Errorf("palette: codes of %s (%s): %d indices and %d table bytes for %d x %d in %s",
			e.Name, key, len(idx), len(tab), rows, cols, e.Fmt)
	}
	for i, v := range idx {
		if int(v) >= k {
			return LUT4{}, false, fmt.Errorf("palette: codes of %s: index %d at %d of a %d entry table", e.Name, v, i, k)
		}
	}
	out := LUT4{Rows: rows, Cols: cols, Group: group, ColBlock: cb, Bits: bits, LUT: make([]uint16, rows/group*(cols/cb)*k)}
	for i := range out.LUT {
		out.LUT[i] = f16(math.Float32frombits(binary.LittleEndian.Uint32(tab[4*i:])) * scale)
	}
	out.Indices = pack(idx, bits)
	c.mu.Lock()
	c.hits++
	c.mu.Unlock()
	return out, true, nil
}

// LookupLinear returns the int8 weight (a scale a row) for w (rows x cols, as the checkpoint
// holds it) when the codes have it, with the scales multiplied by scale (what the graph multiplies
// the weight by). ok is false when there are none; the caller then quantizes w itself.
func (c *Codes) LookupLinear(w []float32, rows, cols int, scale float32) (Linear, bool, error) {
	if c == nil || len(c.index) == 0 {
		return Linear{}, false, nil
	}
	key := WeightKey(w, rows, cols)
	e, ok := c.index[key]
	if !ok || e.Fmt != "int8" || len(e.Shape) != 2 || e.Shape[0] != rows || e.Shape[1] != cols {
		return Linear{}, false, nil
	}
	data, err := os.ReadFile(filepath.Join(c.dir, key+".idx"))
	if err != nil {
		return Linear{}, false, err
	}
	tab, err := os.ReadFile(filepath.Join(c.dir, key+".tab"))
	if err != nil {
		return Linear{}, false, err
	}
	if len(data) != rows*cols || len(tab) != rows*4 {
		return Linear{}, false, fmt.Errorf("palette: codes of %s (%s): %d integers and %d scale bytes for %d x %d in int8",
			e.Name, key, len(data), len(tab), rows, cols)
	}
	for i, v := range data {
		if int8(v) == -128 {
			return Linear{}, false, fmt.Errorf("palette: codes of %s: -128 at %d (int8 codes are -127..127)", e.Name, i)
		}
	}
	out := Linear{Rows: rows, Cols: cols, ColBlock: cols, Bits: 8, Data: data, Scales: make([]uint16, rows)}
	for r := range out.Scales {
		out.Scales[r] = f16(math.Float32frombits(binary.LittleEndian.Uint32(tab[4*r:])) * scale)
	}
	c.mu.Lock()
	c.hits++
	c.mu.Unlock()
	return out, true, nil
}

// CodesWriter writes a codes directory (Codes) from Go: the tables and indices of weights
// palettized here, such as with QuantizeWeighted.
type CodesWriter struct {
	dir   string
	index map[string]codeEntry
}

// NewCodesWriter starts a codes directory at dir, which it makes.
func NewCodesWriter(dir string) (*CodesWriter, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &CodesWriter{dir: dir, index: map[string]codeEntry{}}, nil
}

// Add writes q, the palettized form of the weight w (rows x cols, as the checkpoint holds it),
// under the weight's key; name is what the index calls it.
func (cw *CodesWriter) Add(name string, w []float32, q LUT4) error {
	format := fmt.Sprintf("lut%d", q.Bits)
	if lutBits(format) == 0 {
		return fmt.Errorf("palette: codes of %s: %d bit tables", name, q.Bits)
	}
	key := WeightKey(w, q.Rows, q.Cols)
	idx := make([]byte, q.Rows*q.Cols)
	for i := range idx {
		idx[i] = byte(q.index(i))
	}
	tab := make([]byte, 4*len(q.LUT))
	for i, h := range q.LUT {
		binary.LittleEndian.PutUint32(tab[4*i:], math.Float32bits(f16val(h)))
	}
	if err := os.WriteFile(filepath.Join(cw.dir, key+".idx"), idx, 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(cw.dir, key+".tab"), tab, 0o644); err != nil {
		return err
	}
	e := codeEntry{Name: name, Fmt: format, Shape: []int{q.Rows, q.Cols}, Group: q.Group}
	if q.ColBlock != q.Cols {
		e.ColBlock = q.ColBlock
	}
	cw.index[key] = e
	return nil
}

// Close writes the index.
func (cw *CodesWriter) Close() error {
	b, err := json.MarshalIndent(cw.index, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(cw.dir, "index.json"), append(b, '\n'), 0o644)
}
