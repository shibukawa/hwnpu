package safetensors

import (
	"bytes"
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"
)

func TestMappedFloat32RangeMatchesReadAt(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.safetensors")

	const rows, cols = 64, 128
	f32 := make([]byte, rows*cols*4)
	for i := range rows * cols {
		binary.LittleEndian.PutUint32(f32[4*i:], math.Float32bits(float32(i%97)-48.5))
	}
	i8 := make([]byte, rows*cols)
	for i := range i8 {
		i8[i] = byte(int8((i % 25) - 12))
	}
	scales := make([]byte, rows*2*2) // 2 groups per row, F16 1.0 (0x3c00)
	for i := range rows * 2 {
		binary.LittleEndian.PutUint16(scales[2*i:], 0x3c00)
	}
	err := Write(path, []Out{
		{Name: "emb", DType: "F32", Shape: []int{rows, cols}, Size: int64(len(f32)), Data: bytes.NewReader(f32)},
		{Name: "ple", DType: "I8", Shape: []int{rows, cols}, Size: int64(len(i8)), Data: bytes.NewReader(i8)},
		{Name: "ple" + ScaleSuffix, DType: "F16", Shape: []int{rows, 2}, Size: int64(len(scales)), Data: bytes.NewReader(scales)},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}

	plain, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer plain.Close()

	mapped, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()
	if err := mapped.PrefetchAsync(1 << 20); err != nil {
		t.Fatal(err)
	}

	for _, name := range []string{"emb", "ple"} {
		for _, r := range []int{0, 7, 31, 63} {
			want, err := plain.Float32Range(name, r*cols, (r+1)*cols)
			if err != nil {
				t.Fatal(err)
			}
			got, err := mapped.Float32Range(name, r*cols, (r+1)*cols)
			if err != nil {
				t.Fatal(err)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s row %d[%d]: got %v, want %v", name, r, i, got[i], want[i])
				}
			}
		}
	}
}

func BenchmarkFloat32Range(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "emb.safetensors")
	const rows, cols = 4096, 1024
	bf16 := make([]byte, rows*cols*2)
	for i := range rows * cols {
		binary.LittleEndian.PutUint16(bf16[2*i:], 0x3f80) // 1.0 in bf16
	}
	if err := Write(path, []Out{
		{Name: "emb", DType: "BF16", Shape: []int{rows, cols}, Size: int64(len(bf16)), Data: bytes.NewReader(bf16)},
	}, nil); err != nil {
		b.Fatal(err)
	}
	plain, err := Open(path)
	if err != nil {
		b.Fatal(err)
	}
	defer plain.Close()
	mapped, err := Open(path)
	if err != nil {
		b.Fatal(err)
	}
	defer mapped.Close()
	if err := mapped.MapFile(); err != nil {
		b.Fatal(err)
	}
	b.Run("ReadAt", func(b *testing.B) {
		for i := 0; b.Loop(); i++ {
			r := (i * 97) % rows
			if _, err := plain.Float32Range("emb", r*cols, (r+1)*cols); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("Mapped", func(b *testing.B) {
		for i := 0; b.Loop(); i++ {
			r := (i * 97) % rows
			if _, err := mapped.Float32Range("emb", r*cols, (r+1)*cols); err != nil {
				b.Fatal(err)
			}
		}
	})
}
