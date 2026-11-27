package coreai

import (
	"context"
	"encoding/binary"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/coreai/ailower"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// An input's buffer that is an array of Core AI's (graph.AllocInput) is run as it is, with what
// the host wrote into it since the last run: its values changed in place between runs, an array
// of one input bound to another of its shape, a buffer of the host's bound after an array and
// an array after that, an array freed while the program still holds it, and an array of another
// shape (copied as a buffer of the host's). y = 2 a + b, z = 2 w.
func TestInputArrays(t *testing.T) {
	infos, err := Provider{}.Enumerate(context.Background())
	if err != nil || len(infos) == 0 {
		t.Skip("no Core AI:", err)
	}
	t.Setenv("HWNPU_CACHE", t.TempDir())
	dir := filepath.Join(t.TempDir(), "m.mlpackage")
	pkg, err := mlpackage.Create(dir)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	f16 := mlpackage.Float16
	shape := []int{1, 4, 256, 64}
	a := b.Input("a", f16, shape...)
	c := b.Input("b", f16, shape...)
	two := b.Op("mul", map[string]any{"x": a, "y": b.ScalarF16(2)}, f16, shape...)
	y := b.NamedOp("add", "y", map[string]any{"x": two, "y": c}, f16, shape...)
	// as many bytes in another shape
	other := []int{1, 64, 4, 256}
	w := b.Input("w", f16, other...) // (the builder names its constants c)
	z := b.NamedOp("add", "z", map[string]any{"x": w, "y": w}, f16, other...)
	feat := func(name string) mlpackage.Feature { return mlpackage.Feature{Name: name, DType: f16, Shape: shape} }
	wf, zf := mlpackage.Feature{Name: "w", DType: f16, Shape: other}, mlpackage.Feature{Name: "z", DType: f16, Shape: other}
	if err := pkg.Finish(b.Finish(y, z), 9, []mlpackage.Feature{feat("a"), feat("b"), wf}, []mlpackage.Feature{feat("y"), zf}); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	dev, err := Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{CacheDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	req, err := g.NewRequest()
	if err != nil {
		t.Fatal(err)
	}
	arg := map[string]types.ArgInfo{}
	for _, x := range g.Args() {
		arg[x.Name] = x
	}
	n := arg["a"].Bytes / 2
	array := func(name string) backend.Buffer {
		buf, err := g.(backend.InputAllocator).AllocInput(arg[name].Index)
		if err != nil {
			t.Fatal(err)
		}
		if buf == nil {
			t.Skip("the shim has no arrays for the inputs")
		}
		if len(buf.Bytes()) != arg[name].Bytes {
			t.Fatalf("an array of %d bytes for %s of %d", len(buf.Bytes()), name, arg[name].Bytes)
		}
		return buf
	}
	plain := func() backend.Buffer {
		buf, err := dev.AllocBuffer(arg["a"].Bytes, 0)
		if err != nil {
			t.Fatal(err)
		}
		return buf
	}
	fill := func(buf backend.Buffer, base float32) {
		for i := range n {
			binary.LittleEndian.PutUint16(buf.Bytes()[2*i:], mlpackage.Float16Bits(base+float32(i%97)/8))
		}
	}
	out, zout := plain(), plain()
	defer out.Close()
	defer zout.Close()
	req.Bind(arg["y"].Index, out)
	req.Bind(arg["z"].Index, zout)
	win := array("a") // an array of a's shape for w, of another: copied
	defer win.Close()
	fill(win, 20)
	req.Bind(arg["w"].Index, win)
	run := func(what string, a, b backend.Buffer, abase, bbase float32) {
		t.Helper()
		req.Bind(arg["a"].Index, a)
		req.Bind(arg["b"].Index, b)
		f, err := req.Submit()
		if err == nil {
			err = f.Wait(ctx)
		}
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		for i := range n {
			want := 2*(abase+float32(i%97)/8) + bbase + float32(i%97)/8
			got := mlpackage.Float16Value(binary.LittleEndian.Uint16(out.Bytes()[2*i:]))
			if math.Abs(float64(got-want)) > 0.02 {
				t.Fatalf("%s: y[%d] = %g, want %g", what, i, got, want)
			}
			wv := mlpackage.Float16Value(binary.LittleEndian.Uint16(win.Bytes()[2*i:]))
			if got := mlpackage.Float16Value(binary.LittleEndian.Uint16(zout.Bytes()[2*i:])); got != 2*wv {
				t.Fatalf("%s: z[%d] = %g, want 2 w = %g", what, i, got, 2*wv)
			}
		}
	}
	a1, a2, b1 := array("a"), array("a"), array("b")
	defer b1.Close()
	fill(a1, 1)
	fill(a2, -3)
	fill(b1, 5)
	run("arrays", a1, b1, 1, 5)
	fill(a1, 7) // in place, between two runs
	fill(win, 21)
	run("an array written again", a1, b1, 7, 5)
	fill(b1, -2)
	run("the other written again", a1, b1, 7, -2)
	run("another array of the input", a2, b1, -3, -2)
	run("the first again", a1, b1, 7, -2)
	run("an input's array for the other", b1, a2, -2, -3)
	p := plain()
	defer p.Close()
	fill(p, 11)
	run("a buffer of the host's after an array", p, b1, 11, -2)
	fill(p, 12)
	run("the buffer written again", p, b1, 12, -2)
	fill(a1, 9)
	run("an array after the buffer", a1, b1, 9, -2)
	// freed while the program holds it from its last run: the next run takes another
	if err := a1.Close(); err != nil {
		t.Fatal(err)
	}
	a3 := array("a")
	defer a3.Close()
	fill(a3, 4)
	run("an array after one freed", a3, b1, 4, -2)
	if err := a2.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestOutputArrays checks output zero-copy (AllocOutput via outputViews), chaining one graph's
// output array directly into another's input, input/output aliasing, output/output aliasing,
// official AIModelCache hasEntry, and verifies zero CoW reallocations occurred.
func TestOutputArrays(t *testing.T) {
	infos, err := Provider{}.Enumerate(context.Background())
	if err != nil || len(infos) == 0 {
		t.Skip("no Core AI:", err)
	}
	t.Setenv("HWNPU_CACHE", t.TempDir())
	shape := []int{1, 4, 256, 64}
	makePkg := func(name string, scale float32) string {
		dir := filepath.Join(t.TempDir(), name+".mlpackage")
		pkg, err := mlpackage.Create(dir)
		if err != nil {
			t.Fatal(err)
		}
		b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
		f16 := mlpackage.Float16
		x := b.Input("x", f16, shape...)
		y := b.NamedOp("mul", "y", map[string]any{"x": x, "y": b.ScalarF16(scale)}, f16, shape...)
		z := b.NamedOp("add", "z", map[string]any{"x": x, "y": b.ScalarF16(1)}, f16, shape...)
		feat := func(n string) mlpackage.Feature { return mlpackage.Feature{Name: n, DType: f16, Shape: shape} }
		if err := pkg.Finish(b.Finish(y, z), 9, []mlpackage.Feature{feat("x")}, []mlpackage.Feature{feat("y"), feat("z")}); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	ctx := context.Background()
	cacheDir := t.TempDir()
	dev, err := Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{CacheDir: cacheDir})
	if err != nil {
		t.Fatal(err)
	}
	p1 := makePkg("p1", 2)
	p2 := makePkg("p2", 3)

	g1, err := dev.Compile(ctx, types.GraphSource{Path: p1})
	if err != nil {
		t.Fatal(err)
	}
	defer g1.Close()
	g2, err := dev.Compile(ctx, types.GraphSource{Path: p2})
	if err != nil {
		t.Fatal(err)
	}
	defer g2.Close()

	// Verify official AIModelCache check reports cached entry after Compile.
	aimodelDir, _, _, err := resolve(cacheDir, p1)
	if err != nil {
		t.Fatal(err)
	}
	if !hasEntry(aimodelDir, kindNeuralEngine) {
		t.Errorf("hasEntry(%s) = false after Compile, want true", aimodelDir)
	}
	if hasEntry(filepath.Join(t.TempDir(), "missing.aimodel"), kindNeuralEngine) {
		t.Errorf("hasEntry on nonexistent asset = true, want false")
	}

	argOf := func(g backend.Graph) map[string]types.ArgInfo {
		m := map[string]types.ArgInfo{}
		for _, a := range g.Args() {
			m[a.Name] = a
		}
		return m
	}
	a1, a2 := argOf(g1), argOf(g2)
	n := a1["x"].Bytes / 2

	req1, err := g1.NewRequest()
	if err != nil {
		t.Fatal(err)
	}
	req2, err := g2.NewRequest()
	if err != nil {
		t.Fatal(err)
	}

	inBuf, err := g1.(backend.InputAllocator).AllocInput(a1["x"].Index)
	if err != nil || inBuf == nil {
		t.Fatalf("AllocInput: %v %v", inBuf, err)
	}
	defer inBuf.Close()
	midBuf, err := g1.(backend.OutputAllocator).AllocOutput(a1["y"].Index)
	if err != nil || midBuf == nil {
		t.Fatalf("AllocOutput y: %v %v", midBuf, err)
	}
	defer midBuf.Close()
	z1Buf, err := g1.(backend.OutputAllocator).AllocOutput(a1["z"].Index)
	if err != nil || z1Buf == nil {
		t.Fatalf("AllocOutput z1: %v %v", z1Buf, err)
	}
	defer z1Buf.Close()
	outBuf, err := g2.(backend.OutputAllocator).AllocOutput(a2["y"].Index)
	if err != nil || outBuf == nil {
		t.Fatalf("AllocOutput y2: %v %v", outBuf, err)
	}
	defer outBuf.Close()
	z2Buf, err := g2.(backend.OutputAllocator).AllocOutput(a2["z"].Index)
	if err != nil || z2Buf == nil {
		t.Fatalf("AllocOutput z2: %v %v", z2Buf, err)
	}
	defer z2Buf.Close()

	fill := func(buf backend.Buffer, base float32) {
		for i := range n {
			binary.LittleEndian.PutUint16(buf.Bytes()[2*i:], mlpackage.Float16Bits(base+float32(i%31)/4))
		}
	}
	submit := func(req backend.Request) {
		t.Helper()
		f, err := req.Submit()
		if err == nil {
			err = f.Wait(ctx)
		}
		if err != nil {
			t.Fatal(err)
		}
	}

	movesBefore := caiCowMoves()

	// 1. Chained zero-copy: inBuf -> g1 (x2) -> midBuf -> g2 (x3) -> outBuf, repeated across runs.
	req1.Bind(a1["x"].Index, inBuf)
	req1.Bind(a1["y"].Index, midBuf)
	req1.Bind(a1["z"].Index, z1Buf)
	req2.Bind(a2["x"].Index, midBuf)
	req2.Bind(a2["y"].Index, outBuf)
	req2.Bind(a2["z"].Index, z2Buf)
	for step, base := range []float32{1.0, -2.5, 4.25} {
		fill(inBuf, base)
		submit(req1)
		submit(req2)
		for i := range n {
			x := base + float32(i%31)/4
			gotMid := mlpackage.Float16Value(binary.LittleEndian.Uint16(midBuf.Bytes()[2*i:]))
			gotOut := mlpackage.Float16Value(binary.LittleEndian.Uint16(outBuf.Bytes()[2*i:]))
			gotZ2 := mlpackage.Float16Value(binary.LittleEndian.Uint16(z2Buf.Bytes()[2*i:]))
			if math.Abs(float64(gotMid-2*x)) > 0.02 {
				t.Fatalf("step %d mid[%d] = %g, want %g", step, i, gotMid, 2*x)
			}
			if math.Abs(float64(gotOut-6*x)) > 0.05 {
				t.Fatalf("step %d out[%d] = %g, want %g", step, i, gotOut, 6*x)
			}
			if math.Abs(float64(gotZ2-(2*x+1))) > 0.02 {
				t.Fatalf("step %d z2[%d] = %g, want %g", step, i, gotZ2, 2*x+1)
			}
		}
	}

	// 2. Aliasing safety: bind the exact same lent buffer to both input x and output y.
	fill(inBuf, 3.0)
	req1.Bind(a1["x"].Index, inBuf)
	req1.Bind(a1["y"].Index, inBuf)
	req1.Bind(a1["z"].Index, z1Buf)
	submit(req1)
	for i := range n {
		x := 3.0 + float32(i%31)/4
		gotY := mlpackage.Float16Value(binary.LittleEndian.Uint16(inBuf.Bytes()[2*i:]))
		gotZ := mlpackage.Float16Value(binary.LittleEndian.Uint16(z1Buf.Bytes()[2*i:]))
		if math.Abs(float64(gotY-2*x)) > 0.02 || math.Abs(float64(gotZ-(x+1))) > 0.02 {
			t.Fatalf("in-place alias [%d]: y=%g z=%g, want %g %g", i, gotY, gotZ, 2*x, x+1)
		}
	}

	// 3. Output/output aliasing: bind midBuf to both y and z (must not crash on Swift exclusivity).
	fill(inBuf, 2.0)
	req1.Bind(a1["x"].Index, inBuf)
	req1.Bind(a1["y"].Index, midBuf)
	req1.Bind(a1["z"].Index, midBuf)
	submit(req1)

	if moves := caiCowMoves() - movesBefore; moves != 0 {
		t.Errorf("CoW storage moved %d times during zero-copy outputViews, want 0", moves)
	}
}

// TestAssetInspect verifies AIModelAsset.isValid(at:) and AIModelAsset.summary(includingStatistics:false)
// without loading weights or triggering Core AI compilation.
func TestAssetInspect(t *testing.T) {
	infos, err := Provider{}.Enumerate(context.Background())
	if err != nil || len(infos) == 0 {
		t.Skip("no Core AI:", err)
	}
	emptyDir := t.TempDir()
	if AssetValid(emptyDir) {
		t.Errorf("AssetValid(%s) = true, want false", emptyDir)
	}
	if _, err := AssetFunctions(emptyDir); err == nil {
		t.Errorf("AssetFunctions(%s) succeeded on empty dir, want error", emptyDir)
	}
	makePkg := func(name string, scale float32) string {
		dir := filepath.Join(t.TempDir(), name+".mlpackage")
		pkg, err := mlpackage.Create(dir)
		if err != nil {
			t.Fatal(err)
		}
		b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
		f16 := mlpackage.Float16
		x := b.Input("x", f16, 1, 8)
		y := b.NamedOp("mul", "y", map[string]any{"x": x, "y": b.ScalarF16(scale)}, f16, 1, 8)
		ins := []mlpackage.Feature{{Name: "x", DType: f16, Shape: []int{1, 8}}}
		outs := []mlpackage.Feature{{Name: "y", DType: f16, Shape: []int{1, 8}}}
		if err := pkg.Finish(b.Finish(y), 9, ins, outs); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	p1, p2 := makePkg("a", 2), makePkg("b", 3)
	prog, err := ailower.LowerFunctions([]string{p1, p2}, []string{"fn_a", "fn_b"})
	if err != nil {
		t.Fatal(err)
	}
	assetDir := filepath.Join(t.TempDir(), "multi.aimodel")
	if err := prog.WriteAsset(assetDir); err != nil {
		t.Fatal(err)
	}
	if !AssetValid(assetDir) {
		t.Fatalf("AssetValid(%s) = false, want true", assetDir)
	}
	fns, err := AssetFunctions(assetDir)
	if err != nil {
		t.Fatalf("AssetFunctions: %v", err)
	}
	slices.Sort(fns)
	if want := []string{"fn_a", "fn_b"}; !slices.Equal(fns, want) {
		t.Errorf("AssetFunctions = %v, want %v", fns, want)
	}
}

// BenchmarkOutputZeroCopy compares AllocOutput (outputViews zero-copy) vs plain host buffers
// on a large 8 MiB fp16 tensor (similar to 1024x1024 VAE stages or 8x256K logits).
func BenchmarkOutputZeroCopy(b *testing.B) {
	infos, err := Provider{}.Enumerate(context.Background())
	if err != nil || len(infos) == 0 {
		b.Skip("no Core AI:", err)
	}
	shape := []int{1, 64, 256, 256} // 4M fp16 elements = 8 MiB
	dir := filepath.Join(b.TempDir(), "big.mlpackage")
	pkg, err := mlpackage.Create(dir)
	if err != nil {
		b.Fatal(err)
	}
	mb := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	f16 := mlpackage.Float16
	x := mb.Input("x", f16, shape...)
	y := mb.NamedOp("add", "y", map[string]any{"x": x, "y": mb.ScalarF16(1)}, f16, shape...)
	feat := func(n string) mlpackage.Feature { return mlpackage.Feature{Name: n, DType: f16, Shape: shape} }
	if err := pkg.Finish(mb.Finish(y), 9, []mlpackage.Feature{feat("x")}, []mlpackage.Feature{feat("y")}); err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	dev, err := Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{CacheDir: b.TempDir()})
	if err != nil {
		b.Fatal(err)
	}
	g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
	if err != nil {
		b.Fatal(err)
	}
	defer g.Close()
	arg := map[string]types.ArgInfo{}
	for _, a := range g.Args() {
		arg[a.Name] = a
	}
	inArr, err := g.(backend.InputAllocator).AllocInput(arg["x"].Index)
	if err != nil || inArr == nil {
		b.Fatal(err)
	}
	defer inArr.Close()
	outArr, err := g.(backend.OutputAllocator).AllocOutput(arg["y"].Index)
	if err != nil || outArr == nil {
		b.Fatal(err)
	}
	defer outArr.Close()
	outHost, err := dev.AllocBuffer(arg["y"].Bytes, 0)
	if err != nil {
		b.Fatal(err)
	}
	defer outHost.Close()

	run := func(b *testing.B, out backend.Buffer) {
		req, err := g.NewRequest()
		if err != nil {
			b.Fatal(err)
		}
		req.Bind(arg["x"].Index, inArr)
		req.Bind(arg["y"].Index, out)
		// warmup
		for range 5 {
			f, err := req.Submit()
			if err != nil || f.Wait(ctx) != nil {
				b.Fatal(err)
			}
		}
		b.ResetTimer()
		for b.Loop() {
			f, err := req.Submit()
			if err != nil {
				b.Fatal(err)
			}
			if err := f.Wait(ctx); err != nil {
				b.Fatal(err)
			}
		}
	}
	b.Run("ZeroCopyOutputArray_8MiB", func(b *testing.B) { run(b, outArr) })
	b.Run("HostBufferOutput_8MiB", func(b *testing.B) { run(b, outHost) })
}
