package chat

import (
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

type fakeRowSource struct {
	dtype    string
	headDim  int
	rowBytes int
	data     []byte
}

func (f *fakeRowSource) RowBytes() int             { return f.rowBytes }
func (f *fakeRowSource) RowFormat() (string, int)  { return f.dtype, f.headDim }
func (f *fakeRowSource) ExportRows(from, to int) ([]byte, error) {
	return f.data[from*f.rowBytes : to*f.rowBytes], nil
}

func TestKVQRoundTripF16AndBF16(t *testing.T) {
	rng := rand.New(rand.NewPCG(101, 202))
	for _, dtype := range []string{"f16", "bf16"} {
		for _, dim := range []int{64, 128} {
			const nvecs = 320 // > 256 to exercise parallel workers too
			orig := make([]float32, nvecs*dim)
			for i := range orig {
				v := float32(rng.NormFloat64())
				if i%dim == 0 {
					v *= 12 // strong channel outlier as seen in real attention keys
				}
				orig[i] = v
			}
			raw := make([]byte, len(orig)*2)
			encodeKVQElems(raw, orig, dtype)
			ref := make([]float32, len(orig))
			decodeKVQElems(ref, raw, dtype)

			for _, bits := range []int{5, 6} {
				comp, err := kvqCompress(raw, dtype, dim, bits)
				if err != nil {
					t.Fatalf("%s dim=%d kv%d compress: %v", dtype, dim, bits, err)
				}
				dec, err := kvqDecompress(comp, dtype, dim, bits)
				if err != nil {
					t.Fatalf("%s dim=%d kv%d decompress: %v", dtype, dim, bits, err)
				}
				got := make([]float32, len(orig))
				decodeKVQElems(got, dec, dtype)
				var num, den float64
				for i := range ref {
					d := float64(ref[i] - got[i])
					num += d * d
					den += float64(ref[i]) * float64(ref[i])
				}
				relMSE := num / den
				maxMSE := 0.004
				if bits == 6 {
					maxMSE = 0.0012
				}
				t.Logf("%s dim=%d kv%d: %d -> %d B (%.1f%%), relMSE=%.5f (relRMS=%.4f)",
					dtype, dim, bits, len(raw), len(comp), 100*float64(len(comp))/float64(len(raw)), relMSE, math.Sqrt(relMSE))
				if relMSE > maxMSE {
					t.Fatalf("%s dim=%d kv%d: relMSE %.5f > %.5f", dtype, dim, bits, relMSE, maxMSE)
				}
			}
		}
	}
}

func TestStoreKVQPutAppendReload(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 9))
	for _, tc := range []struct {
		name    string
		dtype   string
		headDim int
		wantDim int
	}{
		{"apple_f16_d128", "f16", 128, 128},
		{"apple_f16_d64", "f16", 64, 64},
		{"amd_bf16_d64", "bf16", 64, 64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const rows = 32
			const heads = 4
			rowVals := heads * tc.headDim
			rowBytes := rowVals * 2
			vals := make([]float32, rows*rowVals)
			for i := range vals {
				vals[i] = float32(rng.NormFloat64())
			}
			raw := make([]byte, len(vals)*2)
			encodeKVQElems(raw, vals, tc.dtype)
			src := &fakeRowSource{dtype: tc.dtype, headDim: tc.headDim, rowBytes: rowBytes, data: raw}

			dir := t.TempDir()
			st, err := openStore(dir)
			if err != nil {
				t.Fatal(err)
			}
			toks := make([]int32, rows)
			for i := range toks {
				toks[i] = int32(i + 1)
			}
			// Put first 20 rows, then append the remaining 12 rows.
			e1, err := st.put(toks[:20], src, []storedMark{{Pos: 20}}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if e1.KVQBits != storeKVQBits || e1.KVQDim != tc.wantDim || e1.KVQDType != tc.dtype {
				t.Fatalf("got kvq=(%d, %d, %q), want (%d, %d, %q)", e1.KVQBits, e1.KVQDim, e1.KVQDType, storeKVQBits, tc.wantDim, tc.dtype)
			}
			e2, err := st.put(toks, src, []storedMark{{Pos: rows}}, e1)
			if err != nil {
				t.Fatal(err)
			}
			if e2 != e1 {
				t.Fatalf("expected append to reuse base entry")
			}
			fi, err := os.Stat(filepath.Join(e2.dir, "rows.bin"))
			if err != nil {
				t.Fatal(err)
			}
			wantSize := int64(rows * e2.diskRowBytes())
			if fi.Size() != wantSize || fi.Size() >= int64(rows*rowBytes)/2 {
				t.Fatalf("rows.bin size=%d, want %d (< 50%% of %d)", fi.Size(), wantSize, rows*rowBytes)
			}

			// Re-open store from disk (simulating restart) and verify decoded rows match closely.
			st2, err := openStore(dir)
			if err != nil || len(st2.entries) != 1 {
				t.Fatalf("reopen: len=%d err=%v", len(st2.entries), err)
			}
			gotBytes, _, _, err := st2.load(st2.entries[0], rows)
			if err != nil {
				t.Fatal(err)
			}
			ref := make([]float32, len(vals))
			got := make([]float32, len(vals))
			decodeKVQElems(ref, raw, tc.dtype)
			decodeKVQElems(got, gotBytes, tc.dtype)
			var num, den float64
			for i := range ref {
				d := float64(ref[i] - got[i])
				num += d * d
				den += float64(ref[i]) * float64(ref[i])
			}
			if relMSE := num / den; relMSE > 0.0012 {
				t.Fatalf("relMSE=%.5f too high", relMSE)
			}
		})
	}
}
