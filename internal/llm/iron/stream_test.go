package iron

import (
	"encoding/json"
	"errors"
	"os"
	"testing"
)

// TestStream runs the program of a design's per-call instruction stream (IRON's ReLU on
// x[:n], for NPU1) against the streams mlir-aie's dispatch library built for the same
// scalars (testdata, from tools/amd/iron_stream.py's check).
func TestStream(t *testing.T) {
	b, err := os.ReadFile("testdata/stream-relu-npu1.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Program Program
		Samples []struct {
			Scalars []int64
			Stream  []uint32
			Refused bool
		}
	}
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	if err := f.Program.check(); err != nil {
		t.Fatal(err)
	}
	var buf []uint32
	for _, s := range f.Samples {
		got, err := f.Program.Run(buf, s.Scalars)
		if s.Refused {
			var r *RefusedError
			if !errors.As(err, &r) {
				t.Errorf("%v: %v, want a refusal", s.Scalars, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%v: %v", s.Scalars, err)
		}
		if len(got) != len(s.Stream) {
			t.Fatalf("%v: %d words, want %d", s.Scalars, len(got), len(s.Stream))
		}
		for i := range got {
			if got[i] != s.Stream[i] {
				t.Fatalf("%v: word %d is %#x, want %#x", s.Scalars, i, got[i], s.Stream[i])
			}
		}
		buf = got
	}
}

func TestStreamConversions(t *testing.T) {
	for _, c := range []struct {
		op     string
		x      int64
		tx     ctype
		y      int64
		ty     ctype
		want   int64
		result ctype
	}{
		{"+", -1, tI32, 1, tU32, 0, tU32},                   // -1 as uint32 + 1 wraps
		{"<", -1, tI32, 1, tU32, 0, tBool},                  // -1 converts to 0xffffffff
		{"<", -1, tI64, 1, tU32, 1, tBool},                  // uint32 fits in int64
		{"/", -7, tI32, 2, tI32, -3, tI32},                  // truncates
		{"%", -7, tI32, 2, tI32, -1, tI32},                  //
		{">>", -8, tI32, 1, tI32, -4, tI32},                 // arithmetic
		{">>", 0xfffffff8, tU32, 1, tU32, 0x7ffffffc, tU32}, // logical
		{"*", 0x10000, tU32, 0x10000, tU32, 0, tU32},        // wraps
		{"-", 0, tU64, 1, tU64, -1, tU64},                   // 0xffff...ffff held as -1
		{"<=", -1, tU64, 0, tU64, 0, tBool},                 // unsigned compare
	} {
		got, err := binop(c.op, c.x, c.tx, c.y, c.ty)
		if err != nil {
			t.Fatal(err)
		}
		if got = conv(got, c.result); got != c.want {
			t.Errorf("%d %s %d: %d, want %d", c.x, c.op, c.y, got, c.want)
		}
	}
}
