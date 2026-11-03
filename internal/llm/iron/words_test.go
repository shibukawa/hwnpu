package iron

import (
	"encoding/binary"
	"testing"
)

func i64(v int64) *int64 { return &v }

func TestAffine(t *testing.T) {
	// ceil(n / 2048) as IRON writes it: -((-n) >> 11)
	ceil := Affine{Value: "n", Scale: -1, Down: 11, Mul: -1}
	for n, want := range map[int64]int64{0: 0, 1: 1, 2048: 1, 2049: 2, 4096: 2} {
		if got, _ := ceil.Eval(map[string]int64{"n": n}); got != want {
			t.Errorf("ceil(%d/2048) = %d, want %d", n, got, want)
		}
	}
	// a ring's row: (p + 3) % 8 + 1, clamped before the wrap
	ring := Affine{Value: "p", Scale: 1, Bias: 3, Mul: 1, Mod: i64(8), Post: 1}
	if got, _ := ring.Eval(map[string]int64{"p": 6}); got != 2 {
		t.Errorf("ring %d", got)
	}
	clamp := Affine{Value: "p", Scale: 2, Mul: 1, Lo: i64(4), Hi: i64(10)}
	for p, want := range map[int64]int64{0: 4, 3: 6, 9: 10} {
		if got, _ := clamp.Eval(map[string]int64{"p": p}); got != want {
			t.Errorf("clamp(%d) = %d, want %d", p, got, want)
		}
	}
	if _, err := clamp.Eval(map[string]int64{}); err == nil {
		t.Error("a missing value was accepted")
	}
}

func TestLinear(t *testing.T) {
	// ceil(rows * 3/16 + 1/2)
	l := Linear{Value: "rows", Num: 3, Den: 16, OffNum: 1, OffDen: 2}
	for rows, want := range map[int64]int64{0: 1, 1: 1, 2: 1, 3: 2, 16: 4, 2048: 385} {
		if got, _ := l.Eval(map[string]int64{"rows": rows}); got != want {
			t.Errorf("rows %d: %d, want %d", rows, got, want)
		}
	}
	if ceilDiv(-5, 2) != -2 || ceilDiv(5, 2) != 3 || ceilDiv(4, 2) != 2 {
		t.Error("ceilDiv")
	}
}

func TestMHADerived(t *testing.T) {
	n := func(v int64) *Num { x := Num(v); return &x }
	c := map[string]*Num{"B_q": n(16), "B_kv": n(64), "num_pipelines": n(4), "packed": n(0), "causal": n(1), "window": nil,
		"seq_len": n(2048), "kv_len": n(4096), "seq_pad": n(2048), "num_heads": n(8), "num_KV_heads": n(1)}
	ext := map[string]*Affine{"valid": {Value: "rows", Scale: 1, Mul: 1}, "kv_valid": {Value: "position", Scale: 1, Bias: 1, Mul: 1}}
	values := map[string]int64{"rows": 37, "position": 136} // a chunk of 37 rows from 100
	for name, want := range map[string]int64{
		"kv_blocks": 3, "s_kv": 137, "q_start": 6, "q_offset": 4, "s_q": 6*16 + 37, "q_blocks_valid": 1, "q_blocks_per_pipeline": 32, "heads": 8,
	} {
		d := Derived{Op: "MHA", Name: name, Const: c, Extents: ext}
		got, err := d.Eval(values)
		if err != nil || got != want {
			t.Errorf("%s = %d (%v), want %d", name, got, err, want)
		}
	}
	d := Derived{Op: "MHA", Name: "q_start", Const: c, Extents: ext}
	if _, err := d.Eval(map[string]int64{"rows": 10, "position": 3}); err == nil {
		t.Error("more queries than keys were accepted")
	}
}

func TestWriteWords(t *testing.T) {
	sp := make([]byte, 16)
	params := []Param{
		{Name: "n", Index: 0, Kind: "core", Form: &Affine{Value: "n", Scale: 1, Mul: 1}},
		{Name: "off", Index: 2, Kind: "addr", Form: &Affine{Value: "n", Scale: 4, Mul: 1}},
	}
	if err := writeWords(sp, params, map[string]int64{"n": 5}); err != nil {
		t.Fatal(err)
	}
	if binary.LittleEndian.Uint32(sp) != 20 || binary.LittleEndian.Uint32(sp[8:]) != 20 {
		t.Errorf("%x", sp)
	}
	params[1].Index = 4
	if err := writeWords(sp, params, map[string]int64{"n": 5}); err == nil {
		t.Error("a slot past the scratchpad was accepted")
	}
}
