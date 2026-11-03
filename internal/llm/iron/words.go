package iron

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
)

// Affine is IRON's integer expression in one per-call value v:
//
//	clamp(((v*scale + bias) >> down)*mul + add, lo, hi) % mod + post
//
// with >> flooring, a missing bound none and mod a power of two.
type Affine struct {
	Value string `json:"value"`
	Scale int64  `json:"scale"`
	Bias  int64  `json:"bias"`
	Down  uint   `json:"down"`
	Mul   int64  `json:"mul"`
	Add   int64  `json:"add"`
	Lo    *int64 `json:"lo"`
	Hi    *int64 `json:"hi"`
	Mod   *int64 `json:"mod"`
	Post  int64  `json:"post"`
}

// Eval evaluates the expression at values.
func (f *Affine) Eval(values map[string]int64) (int64, error) {
	v, ok := values[f.Value]
	if !ok {
		return 0, fmt.Errorf("iron: no per-call value %s", f.Value)
	}
	x := ((v*f.Scale+f.Bias)>>f.Down)*f.Mul + f.Add // >> on int64 floors
	if f.Lo != nil && x < *f.Lo {
		x = *f.Lo
	}
	if f.Hi != nil && x > *f.Hi {
		x = *f.Hi
	}
	if f.Mod != nil {
		x &= *f.Mod - 1 // a power of two: floors as Python's %
	}
	return x + f.Post, nil
}

// Linear is a word as ceil(v*num/den + off_num/off_den).
type Linear struct {
	Value  string `json:"value"`
	Num    int64  `json:"num"`
	Den    int64  `json:"den"`
	OffNum int64  `json:"off_num"`
	OffDen int64  `json:"off_den"`
}

func ceilDiv(a, b int64) int64 {
	q := a / b
	if (a%b != 0) && ((a < 0) == (b < 0)) {
		q++
	}
	return q
}

// Eval evaluates the expression at values.
func (l *Linear) Eval(values map[string]int64) (int64, error) {
	v, ok := values[l.Value]
	if !ok {
		return 0, fmt.Errorf("iron: no per-call value %s", l.Value)
	}
	return ceilDiv(v*l.Num*l.OffDen+l.OffNum*l.Den, l.Den*l.OffDen), nil
}

// Table is a word tabulated over one per-call value from Lo.
type Table struct {
	Value  string  `json:"value"`
	Lo     int64   `json:"lo"`
	Values []int64 `json:"values"`
	Const  bool    `json:"const"`
}

// Eval looks the word up.
func (t *Table) Eval(values map[string]int64) (int64, error) {
	if t.Const {
		return t.Values[0], nil
	}
	v, ok := values[t.Value]
	if !ok {
		return 0, fmt.Errorf("iron: no per-call value %s", t.Value)
	}
	i := v - t.Lo
	if i < 0 || i >= int64(len(t.Values)) {
		return 0, fmt.Errorf("iron: %s = %d is outside the table [%d, %d]", t.Value, v, t.Lo, t.Lo+int64(len(t.Values))-1)
	}
	return t.Values[i], nil
}

// Num is an operator's constant: a number, or a boolean as 0 and 1.
type Num int64

func (n *Num) UnmarshalJSON(b []byte) error {
	switch string(b) {
	case "true":
		*n = 1
	case "false":
		*n = 0
	default:
		var v int64
		if err := json.Unmarshal(b, &v); err != nil {
			return err
		}
		*n = Num(v)
	}
	return nil
}

// Derived is a value an operator derives from its per-call extents. Only MHA's are known.
type Derived struct {
	Op      string             `json:"op"`
	Name    string             `json:"name"`
	Const   map[string]*Num    `json:"const"` // nil: None
	Extents map[string]*Affine `json:"extents"`
}

func (d *Derived) c(name string) (int64, error) {
	v, ok := d.Const[name]
	if !ok {
		return 0, fmt.Errorf("iron: %s.%s: no constant %s", d.Op, d.Name, name)
	}
	if v == nil {
		return 0, nil
	}
	return int64(*v), nil
}

// Eval computes MHA's per-call derived values (iron/operators/mha.py) from its extents: the
// query rows (valid) and the keys (kv_valid).
func (d *Derived) Eval(values map[string]int64) (int64, error) {
	if d.Op != "MHA" {
		return 0, fmt.Errorf("iron: derived values of %s are not known", d.Op)
	}
	var cs [11]int64
	for i, n := range []string{"B_q", "B_kv", "num_pipelines", "packed", "causal", "window", "seq_len", "kv_len", "seq_pad", "num_heads", "num_KV_heads"} {
		v, err := d.c(n)
		if err != nil {
			return 0, err
		}
		cs[i] = v
	}
	bq, bkv, pipes, packed, causal, window, seqLen, kvLen, seqPad, heads, kvHeads := cs[0], cs[1], cs[2], cs[3] != 0, cs[4] != 0, cs[5] != 0, cs[6], cs[7], cs[8], cs[9], cs[10]
	qTokens := seqLen
	if e := d.Extents["valid"]; e != nil {
		v, err := e.Eval(values)
		if err != nil {
			return 0, err
		}
		qTokens = v
	}
	kvTokens := kvLen - seqPad + qTokens
	if e := d.Extents["kv_valid"]; e != nil {
		v, err := e.Eval(values)
		if err != nil {
			return 0, err
		}
		kvTokens = v
	}
	var firstQuery int64
	if causal || window {
		firstQuery = kvTokens - qTokens
		if firstQuery < 0 {
			return 0, fmt.Errorf("iron: MHA: %d queries are more than the %d keys", qTokens, kvTokens)
		}
	}
	fqb := firstQuery / bq
	if packed {
		fqb = ceilDiv(kvTokens, bq)
	}
	switch d.Name {
	case "heads":
		if packed {
			return kvHeads / pipes, nil
		}
		return heads, nil
	case "q_blocks_per_pipeline":
		return ceilDiv(seqPad, bq*pipes), nil
	case "q_blocks_valid":
		return ceilDiv(qTokens, bq*pipes), nil
	case "kv_blocks":
		return ceilDiv(kvTokens, bkv), nil
	case "s_q":
		if packed {
			return (fqb + pipes) * bq, nil
		}
		return fqb*bq + qTokens, nil
	case "s_kv":
		return kvTokens, nil
	case "q_start":
		return fqb, nil
	case "q_offset":
		if packed {
			return 0, nil
		}
		return firstQuery % bq, nil
	}
	return 0, fmt.Errorf("iron: MHA's derived value %s is not known", d.Name)
}

// Param is one scratchpad word: its slot, its kind and how it follows from the call's values.
type Param struct {
	Name    string   `json:"name"`
	Index   int      `json:"index"`
	DType   string   `json:"dtype"`
	Kind    string   `json:"kind"` // core: held shifted left by 2; addr: as it is
	Form    *Affine  `json:"form"`
	Linear  *Linear  `json:"linear"`
	Table   *Table   `json:"table"`
	Derived *Derived `json:"derived"`
	Grid    bool     `json:"grid"` // looked up in the version's Grid
}

// Eval computes the word's value (before the kind's shift).
func (p *Param) Eval(values map[string]int64) (int64, error) {
	switch {
	case p.Form != nil:
		return p.Form.Eval(values)
	case p.Linear != nil:
		return p.Linear.Eval(values)
	case p.Table != nil:
		return p.Table.Eval(values)
	case p.Derived != nil:
		return p.Derived.Eval(values)
	}
	return 0, fmt.Errorf("iron: parameter %s has no expression", p.Name)
}

// writeWords writes a command's scratchpad words. A word a core reads is held shifted left by
// 2 (the firmware drops the low bits), a repeat count by 16, an address as it is
// (mlir-aie's ParameterScratchpad).
func writeWords(sp []byte, params []Param, values map[string]int64) error {
	return writeGridWords(sp, params, values, nil, -1)
}

// writeGridWords is writeWords with the words of a grid at index row.
func writeGridWords(sp []byte, params []Param, values map[string]int64, g *Grid, row int) error {
	for i := range params {
		p := &params[i]
		var v int64
		var err error
		if p.Grid {
			var col []int64
			if g != nil {
				col = g.Values[p.Name]
			}
			if row < 0 || row >= len(col) {
				return fmt.Errorf("iron: parameter %s is in no grid row %d", p.Name, row)
			}
			v = col[row]
		} else {
			v, err = p.Eval(values)
		}
		if err != nil {
			return fmt.Errorf("%s: %w", p.Name, err)
		}
		switch p.Kind {
		case "core":
			v <<= 2
		case "repeat":
			v <<= 16
		case "addr":
		default:
			return fmt.Errorf("iron: parameter %s of kind %q", p.Name, p.Kind)
		}
		if 4*p.Index+4 > len(sp) {
			return fmt.Errorf("iron: parameter %s at word %d: the scratchpad has %d bytes", p.Name, p.Index, len(sp))
		}
		binary.LittleEndian.PutUint32(sp[4*p.Index:], uint32(v))
	}
	return nil
}
