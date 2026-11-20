package decode

import (
	"slices"
	"testing"
)

// carried gives the fake program what StepChunks needs: the attention's output carried from
// stage 0 to stage 1.
func carried(p Program) Program {
	p.Carry = func(i int) []byte { return slices.Clone(p.In["attn_o_1"]) }
	p.Restore = func(i int, b []byte) { copy(p.In["attn_o_1"], b) }
	p.OutStage = map[string]int{"y": 1, "attn_q_1": 0, "kv_k_1_new": 0, "kv_v_1_new": 0}
	return p
}

// Chunks read a stage at a time (StepChunks) give the attention the cache gives read a chunk at
// a time: the last chunk's output is exact, and the next steps see the same cache.
func TestStepChunks(t *testing.T) {
	const n, q = 300, 4
	tab := newTables(n, 21)
	var out []float32
	s, err := New(carried(fakeProgram(q, n, &tab, &out, false)), fakeModel{})
	if err != nil {
		t.Fatal(err)
	}
	rows := func(from, to int) [][]float32 {
		var r [][]float32
		for p := from; p < to; p++ {
			r = append(r, []float32{float32(p)})
		}
		return r
	}
	// 5 chunks of 4 and a last of 3, from position 0, then 2 chunks more from 23
	var chunks [][][]float32
	for c := 0; c < 5; c++ {
		chunks = append(chunks, rows(4*c, 4*c+4))
	}
	chunks = append(chunks, rows(20, 23))
	if st, ok := s.layerMajor(chunks); !ok || st == nil {
		t.Fatal("the program is not read a stage at a time")
	}
	if _, _, err := s.StepChunks(chunks, nil); err != nil {
		t.Fatal(err)
	}
	if s.Pos() != 23 {
		t.Fatalf("position %d after 23 tokens", s.Pos())
	}
	if e := relErr(out, exactAttention(tab, 20, 3, q), q, 3); e > 2e-3 {
		t.Fatalf("the last chunk's attention is %g off", e)
	}
	if _, _, err := s.StepChunks([][][]float32{rows(23, 27), rows(27, 31)}, nil); err != nil {
		t.Fatal(err)
	}
	if e := relErr(out, exactAttention(tab, 27, 4, q), q, 4); e > 2e-3 {
		t.Fatalf("after more chunks the attention is %g off", e)
	}
	// one token after them, by Step: the cache the chunks wrote
	if _, err := s.Step(rows(31, 32)); err != nil {
		t.Fatal(err)
	}
	if e := relErr(out, exactAttention(tab, 31, 1, q), q, 1); e > 2e-3 {
		t.Fatalf("a step after the chunks attends %g off", e)
	}
}

// A mark StepChunks takes after a chunk inside the call is a point the session can go back to:
// after Rewind to it, the next step attends as after reading only that far.
func TestStepChunksMarks(t *testing.T) {
	const n, q = 300, 4
	tab := newTables(n, 22)
	var out []float32
	s, err := New(carried(fakeProgram(q, n, &tab, &out, false)), fakeModel{})
	if err != nil {
		t.Fatal(err)
	}
	rows := func(from, to int) [][]float32 {
		var r [][]float32
		for p := from; p < to; p++ {
			r = append(r, []float32{float32(p)})
		}
		return r
	}
	chunks := [][][]float32{rows(0, 4), rows(4, 8), rows(8, 12), rows(12, 15)}
	_, marks, err := s.StepChunks(chunks, []bool{false, true, false, false})
	if err != nil {
		t.Fatal(err)
	}
	if len(marks) != 1 || marks[0].Mark.Pos() != 8 || len(marks[0].Logits) == 0 {
		t.Fatalf("marks %+v, want one at 8 with its logits", marks)
	}
	s.Rewind(marks[0].Mark)
	if _, err := s.Step(rows(8, 10)); err != nil {
		t.Fatal(err)
	}
	if e := relErr(out, exactAttention(tab, 8, 2, q), q, 2); e > 2e-3 {
		t.Fatalf("after going back to the mark the attention is %g off", e)
	}
}
