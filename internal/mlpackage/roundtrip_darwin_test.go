package mlpackage

import (
	"context"
	"encoding/binary"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/types"
)

// Builds a small program (a palettized embedding table, a gather and a palettized linear) with
// this package alone, loads it through the Core ML backend and compares the Neural Engine's
// result with a float32 computation of the same program.
func TestRoundTrip(t *testing.T) {
	ctx := context.Background()
	infos, err := coreml.Provider{}.Enumerate(ctx)
	if err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	const vocab, dim, out, seq, group = 64, 128, 96, 8, 16
	rng := rand.New(rand.NewSource(1))
	lutOf := func(rows int) []uint16 {
		l := make([]uint16, rows/group*16)
		for i := range l {
			l[i] = Float16Bits(float32(rng.NormFloat64()) * 0.1)
		}
		return l
	}
	idxOf := func(n int) []byte {
		p := make([]byte, n/2)
		for i := range p {
			p[i] = byte(rng.Intn(16)) | byte(rng.Intn(16))<<4
		}
		return p
	}
	dense := func(rows, cols int, idx []byte, lut []uint16) []float32 {
		d := make([]float32, rows*cols)
		for i := range d {
			q := idx[i/2] >> (4 * uint(i%2)) & 15
			d[i] = Float16Value(lut[i/cols/group*16+int(q)])
		}
		return d
	}
	tblIdx, tblLut := idxOf(vocab*dim), lutOf(vocab)
	wIdx, wLut := idxOf(out*dim), lutOf(out)

	dir := filepath.Join(t.TempDir(), "rt.mlpackage")
	pkg, err := Create(dir)
	if err != nil {
		t.Fatal(err)
	}
	b := NewBuilder("CoreML8", pkg.Blobs)
	ids := b.Input("ids", Int32, 1, seq)
	table, err := b.LUT4("table", []int{vocab, dim}, []int{group, dim}, tblIdx, tblLut)
	if err != nil {
		t.Fatal(err)
	}
	emb := b.Op("gather", map[string]any{"x": table, "indices": ids, "axis": b.ScalarI32(0), "validate_indices": b.ScalarBool(false)}, Float16, 1, seq, dim)
	w, err := b.LUT4("w", []int{out, dim}, []int{group, dim}, wIdx, wLut)
	if err != nil {
		t.Fatal(err)
	}
	y := b.NamedOp("linear", "y", map[string]any{"x": emb, "weight": w}, Float16, 1, seq, out)
	prog := b.Finish(y)
	if err := pkg.Finish(prog, 9, []Feature{{"ids", Int32, []int{1, seq}}}, []Feature{{"y", Float16, []int{1, seq, out}}}); err != nil {
		t.Fatal(err)
	}

	dev, err := coreml.Provider{}.Open(ctx, infos[0], backend.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	p := g.Placement()
	t.Logf("placement: verified=%v npu ops=%d elsewhere=%v", p.Verified, p.NPUOps, p.Ops)
	req, err := g.NewRequest()
	if err != nil {
		t.Fatal(err)
	}
	defer req.Close()
	bufs := map[string]backend.Buffer{}
	for _, a := range g.Args() {
		buf, err := dev.AllocBuffer(a.Bytes, types.BufferDefault)
		if err != nil {
			t.Fatal(err)
		}
		defer buf.Close()
		if err := req.Bind(a.Index, buf); err != nil {
			t.Fatal(err)
		}
		bufs[a.Name] = buf
	}
	tokens := make([]int32, seq)
	for i := range tokens {
		tokens[i] = int32(rng.Intn(vocab))
		binary.LittleEndian.PutUint32(bufs["ids"].Bytes()[4*i:], uint32(tokens[i]))
	}
	f, err := req.Submit()
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Wait(ctx); err != nil {
		t.Fatal(err)
	}

	tbl, wd := dense(vocab, dim, tblIdx, tblLut), dense(out, dim, wIdx, wLut)
	raw := bufs["y"].Bytes()
	var maxErr, maxAbs float64
	for s := 0; s < seq; s++ {
		for o := 0; o < out; o++ {
			var acc float32
			for k := 0; k < dim; k++ {
				acc += tbl[int(tokens[s])*dim+k] * wd[o*dim+k]
			}
			got := Float16Value(binary.LittleEndian.Uint16(raw[2*(s*out+o):]))
			maxErr = math.Max(maxErr, math.Abs(float64(got-acc)))
			maxAbs = math.Max(maxAbs, math.Abs(float64(acc)))
		}
	}
	t.Logf("max |err| %.4g, max |y| %.4g", maxErr, maxAbs)
	if maxErr > 0.02*maxAbs+1e-3 {
		t.Fatalf("Neural Engine result differs from the reference: max |err| %.4g", maxErr)
	}
}
