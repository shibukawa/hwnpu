package iron

import (
	"encoding/binary"
	"math"
	"testing"
)

// A part stored as an fp16 scale and int8 values expands to bf16(q * scale), as
// tools/amd/iron_hostple.py's ple_int8 rounds it.
func TestExpandInt8(t *testing.T) {
	// scale 0.25 in fp16 (0x3400), values -127, -1, 0, 3, 127
	src := []byte{0x00, 0x34, 0x81, 0xff, 0x00, 0x03, 0x7f}
	dst := make([]byte, 2*(len(src)-2))
	expandInt8(dst, src)
	want := []float32{-31.75, -0.25, 0, 0.75, 31.75}
	for j, w := range want {
		got := math.Float32frombits(uint32(binary.LittleEndian.Uint16(dst[2*j:])) << 16)
		if got != w {
			t.Errorf("value %d: %v, want %v", j, got, w)
		}
	}
	if (HostTable{PartBytes: 512, Format: HostTableInt8}).fileBytes() != 258 || (HostTable{PartBytes: 512}).fileBytes() != 512 {
		t.Error("fileBytes")
	}
}
