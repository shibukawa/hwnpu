package coreai

import (
	"context"
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
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
