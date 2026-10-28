package ert

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"testing"
)

// The golden header words are produced from the C bit fields of struct ert_start_kernel_cmd by
// testdata/abi/xdna_abi.c.
func TestHeaderMatchesCBitfields(t *testing.T) {
	raw, err := os.ReadFile("../../testdata/abi/xdna_abi.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Headers []uint32 `json:"ert_headers"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	got := []uint32{
		Header(StateNew, 0, 5, OpStartCU, TypeCU),
		Header(StateCompleted, 3, 2047, 31, 15),
		Header(StateCompleted, 0, 1, OpStartNPU, TypeCU),
		Header(0, 0, 0, 0, 0),
	}
	for i := range got {
		if got[i] != g.Headers[i] {
			t.Errorf("header %d: Go %#x, C %#x", i, got[i], g.Headers[i])
		}
	}
	if StateOf(g.Headers[2]) != StateCompleted {
		t.Error("StateOf")
	}
}

func TestStartCU(t *testing.T) {
	buf := make([]byte, 64)
	regmap := []byte{1, 0, 0, 0, 2, 0, 0, 0}
	n, err := StartCU(buf, 2, regmap)
	if err != nil || n != 16 {
		t.Fatalf("%d %v", n, err)
	}
	h := binary.LittleEndian.Uint32(buf)
	if StateOf(h) != StateNew || (h>>12)&0x7ff != 3 || (h>>23)&0x1f != OpStartCU || h>>28 != TypeCU {
		t.Fatalf("header %#x", h)
	}
	if binary.LittleEndian.Uint32(buf[4:]) != 4 || buf[8] != 1 || buf[12] != 2 {
		t.Fatalf("% x", buf[:16])
	}
	if _, err := StartCU(make([]byte, 8), 0, regmap); err == nil {
		t.Fatal("small buffer accepted")
	}
	if !StateCompleted.Terminal() || StateRunning.Terminal() || StateSubmitted.Terminal() || !StateError.Terminal() {
		t.Fatal("Terminal")
	}
}
