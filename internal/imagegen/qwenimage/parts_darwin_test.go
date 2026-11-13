package qwenimage

import (
	"cmp"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/imagegen/qwenimage/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

func readF32s(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func cosine(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}

// TestPartsAgainstReference checks a Core ML export (HWNPU_QWENIMAGE_EXPORT) part by part against
// tools/ref/ref_qwenimage21.py's float32 intermediates (HWNPU_QWENIMAGE_REF: the prefix of
// PREFIX.ids.i32, .txtin.f32 (txt_in of the text rows), .dit_in0.f32, .dit_out0.f32, written from
// its npz): the tokens, the text rows after txt_in, and the first step's velocity from the
// reference's text rows and from the export's.
func TestPartsAgainstReference(t *testing.T) {
	dir, ref := os.Getenv("HWNPU_QWENIMAGE_EXPORT"), os.Getenv("HWNPU_QWENIMAGE_REF")
	if dir == "" || ref == "" {
		t.Skip("HWNPU_QWENIMAGE_EXPORT and HWNPU_QWENIMAGE_REF not set")
	}
	ctx := context.Background()
	var dev *hwnpu.Device
	if os.Getenv("HWNPU_QWENIMAGE_CPU") == "" {
		if infos, err := (coreml.Provider{}).Enumerate(ctx); err != nil || len(infos) == 0 {
			t.Skip("no Neural Engine")
		}
		cache, _ := os.UserCacheDir()
		d, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(cache, "hwnpu")))
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		dev = d
	}
	p, err := OpenWith(ctx, dev, dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	e := p.prog
	prompt := os.Getenv("HWNPU_QWENIMAGE_PROMPT")
	if prompt == "" {
		prompt = "A red fox sitting in a snowy forest at dawn, soft light"
	}
	ids := p.Tokens(prompt)
	b, err := os.ReadFile(ref + ".ids.i32")
	if err != nil {
		t.Fatal(err)
	}
	same := len(b)/4 == len(ids)
	for i := 0; same && i < len(ids); i++ {
		same = int32(binary.LittleEndian.Uint32(b[4*i:])) == ids[i]
	}
	if !same {
		t.Fatalf("tokens %v differ from the reference's", ids)
	}
	E, TR := p.Dit.Hidden, p.Dit.TextRows
	T := len(ids) - p.drop
	// the text rows after txt_in: the first dit_text's input, kept by a run of the text alone
	want := readF32s(t, ref+".txtin.f32")
	refOnly := os.Getenv("HWNPU_QWENIMAGE_REF_TEXT") != "" // the DiT from the reference's text rows alone
	var got []float32
	if !refOnly {
		got = textRows(t, ctx, e, ids)
	} else {
		got = append([]float32(nil), want[:T*E]...)
		for j := range got {
			got[j] /= mil.DitScale
		}
	}
	for j := range got {
		got[j] *= mil.DitScale
	}
	norm := func(v []float32) float64 { return math.Sqrt(dot(v, v)) }
	for i := 0; i < T; i++ {
		g, w := got[i*E:(i+1)*E], want[i*E:(i+1)*E]
		t.Logf("text row %d (token %d): cosine %.5f, norm %.1f (reference %.1f)", i, ids[p.drop+i], cosine(g, w), norm(g), norm(w))
	}
	t.Logf("text rows: cosine %.5f", cosine(got[:T*E], want[:T*E]))
	if os.Getenv("HWNPU_QWENIMAGE_TEXT_ONLY") != "" {
		return
	}
	// the reference's text rows through the DiT's text pass
	refCond := Cond{Rows: T}
	{
		x := make([]float32, TR*E)
		for i := range want {
			x[i] = want[i] / mil.DitScale
		}
		mod0, _ := e.time.modulation(0)
		for gi, name := range e.m.DitText {
			count := min(e.m.Dit.Group, e.m.Dit.Layers-gi*e.m.Dit.Group)
			err := e.each(ctx, name, func(pr runner.Program) error {
				y, err := pr.Run(ctx, map[string][]float32{"x": x, "mod": mod0})
				if err != nil {
					return err
				}
				x = y[:TR*E]
				var tk, tv []float32
				for j := 0; j < count; j++ {
					tk = append(tk, y[(1+2*j)*TR*E:(2+2*j)*TR*E]...)
					tv = append(tv, y[(2+2*j)*TR*E:(3+2*j)*TR*E]...)
				}
				refCond.ptk, refCond.ptv = append(refCond.ptk, tk), append(refCond.ptv, tv)
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	var conds []Cond
	if !refOnly {
		if conds, err = e.encode(ctx, p.embed, [][]int32{ids}); err != nil {
			t.Fatal(err)
		}
	}
	if err := e.openDiT(ctx); err != nil {
		t.Fatal(err)
	}
	x0, v0 := readF32s(t, ref+".dit_in0.f32"), readF32s(t, ref+".dit_out0.f32")
	for _, c := range []struct {
		what string
		cd   Cond
	}{{"the reference's text", refCond}, {"the export's text", Cond{}}} {
		if c.cd.ptk == nil {
			if refOnly {
				continue
			}
			c.cd = conds[0]
		}
		vs, err := e.velocity(ctx, x0, 1, []Cond{c.cd}, []imageInputs{e.imageInputs(c.cd)})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("velocity at sigma 1 from %s: cosine %.5f", c.what, cosine(vs[0], v0))
	}
}

// textRows runs the text encoder's programs on ids: the DiT's text rows (txt_in's, divided by
// DitScale), after the system message's.
func textRows(t *testing.T, ctx context.Context, e *progEngine, ids []int32) []float32 {
	t.Helper()
	m := e.m
	E := m.Text.Hidden
	f, err := os.Open(filepath.Join(e.dir, "text", "embed.bin"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	x := make([]float32, m.Text.Rows*E)
	row := make([]byte, 2*E)
	for i, id := range ids {
		if _, err := f.ReadAt(row, int64(id)*int64(2*E)); err != nil {
			t.Fatal(err)
		}
		for j := 0; j < E; j++ {
			x[i*E+j] = fromBF16(binary.LittleEndian.Uint16(row[2*j:])) / mil.TextScale
		}
	}
	for _, name := range m.TextIn {
		err := e.each(ctx, name, func(p runner.Program) error {
			y, err := p.Run(ctx, map[string][]float32{"x": x})
			x = y
			return err
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	return x[m.Drop*E : len(ids)*E]
}

// TestTextGroupsBisect runs each of the text encoder's groups (HWNPU_QWENIMAGE_EXPORT) on the
// Neural Engine and on Core ML's CPU interpreter from the interpreter's previous output, and logs
// the cosine of the prompt's rows: the group where the Neural Engine drifts.
func TestTextGroupsBisect(t *testing.T) {
	dir := os.Getenv("HWNPU_QWENIMAGE_EXPORT")
	if dir == "" {
		t.Skip("HWNPU_QWENIMAGE_EXPORT not set")
	}
	ctx := context.Background()
	if infos, err := (coreml.Provider{}).Enumerate(ctx); err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	cache, _ := os.UserCacheDir()
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(cache, "hwnpu")))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	p, err := OpenWith(ctx, nil, dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	ane := &progEngine{m: p.prog.m, dir: dir, dev: dev}
	ids := p.Tokens("A red fox sitting in a snowy forest at dawn, soft light")
	m := p.prog.m
	E := m.Text.Hidden
	x := make([]float32, m.Text.Rows*E)
	row := make([]byte, 2*E)
	for i, id := range ids {
		if _, err := p.embed.ReadAt(row, int64(id)*int64(2*E)); err != nil {
			t.Fatal(err)
		}
		for j := 0; j < E; j++ {
			x[i*E+j] = fromBF16(binary.LittleEndian.Uint16(row[2*j:])) / mil.TextScale
		}
	}
	run := func(e *progEngine, name string, in []float32) []float32 {
		var y []float32
		if err := e.each(ctx, name, func(pr runner.Program) error {
			var err error
			y, err = pr.Run(ctx, map[string][]float32{"x": in})
			return err
		}); err != nil {
			t.Fatal(err)
		}
		return y
	}
	n := len(ids)
	for _, name := range m.TextIn {
		cpu := run(p.prog, name, x)
		npu := run(ane, name, x)
		worst, wi := 1.0, 0
		for i := 0; i < n; i++ {
			if c := cosine(npu[i*E:(i+1)*E], cpu[i*E:(i+1)*E]); c < worst {
				worst, wi = c, i
			}
		}
		t.Logf("%s: the prompt's rows %.5f, the worst row %d (token %d) %.5f", name, cosine(npu[:n*E], cpu[:n*E]), wi, ids[wi], worst)
		x = cpu
	}
}

// TestTextLayersBisect writes Qwen3-VL-8B's layers HWNPU_QWENIMAGE_LAYERS (4..7) one program each
// (HWNPU_QWENIMAGE_ROOT's text encoder, int8) into HWNPU_QWENIMAGE_OUT, and runs them from the CPU
// interpreter's previous output on both: each layer's cosine and its non-finite values.
func TestTextLayersBisect(t *testing.T) {
	dir, root, out := os.Getenv("HWNPU_QWENIMAGE_EXPORT"), os.Getenv("HWNPU_QWENIMAGE_ROOT"), os.Getenv("HWNPU_QWENIMAGE_OUT")
	if dir == "" || root == "" || out == "" {
		t.Skip("HWNPU_QWENIMAGE_EXPORT, HWNPU_QWENIMAGE_ROOT and HWNPU_QWENIMAGE_OUT not set")
	}
	ctx := context.Background()
	cache, _ := os.UserCacheDir()
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(cache, "hwnpu")))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	p, err := OpenWith(ctx, nil, dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	text := filepath.Join(filepath.Dir(root), TextDir)
	tc, err := mil.ReadText(text)
	if err != nil {
		t.Fatal(err)
	}
	dc, err := mil.ReadDit(root, 512, 512)
	if err != nil {
		t.Fatal(err)
	}
	w, err := mil.OpenWeights(root, text)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	ids := p.Tokens("A red fox sitting in a snowy forest at dawn, soft light")
	m := p.prog.m
	E := m.Text.Hidden
	x := make([]float32, m.Text.Rows*E)
	row := make([]byte, 2*E)
	for i, id := range ids {
		if _, err := p.embed.ReadAt(row, int64(id)*int64(2*E)); err != nil {
			t.Fatal(err)
		}
		for j := 0; j < E; j++ {
			x[i*E+j] = fromBF16(binary.LittleEndian.Uint16(row[2*j:])) / mil.TextScale
		}
	}
	run := func(d *hwnpu.Device, path string, in []float32) []float32 {
		pr, err := runner.Open(ctx, d, path)
		if err != nil {
			t.Fatal(err)
		}
		defer pr.Close()
		y, err := pr.Run(ctx, map[string][]float32{"x": in})
		if err != nil {
			t.Fatal(err)
		}
		return y
	}
	// the first group's layers on the CPU, from the export
	if err := p.prog.each(ctx, m.TextIn[0], func(pr runner.Program) error {
		y, err := pr.Run(ctx, map[string][]float32{"x": x})
		x = y
		return err
	}); err != nil {
		t.Fatal(err)
	}
	n := len(ids)
	for i := 4; i < 8; i++ {
		format := cmp.Or(os.Getenv("HWNPU_QWENIMAGE_TEXT_FORMAT"), "int8")
		path := filepath.Join(out, fmt.Sprintf("text_layer_%d_%s.mlpackage", i, format))
		if _, err := os.Stat(path); err != nil {
			if _, err := llm.WritePart(path, mil.TextGroup(tc, dc, i, 1, false), w, llm.Options{Backend: llm.BackendCoreML, Format: format}, nil); err != nil {
				t.Fatal(err)
			}
		}
		cpu := run(nil, path, x)
		npu := run(dev, path, x)
		bad, worst, wi := 0, 1.0, 0
		for r := 0; r < n; r++ {
			for _, v := range npu[r*E : (r+1)*E] {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					bad++
				}
			}
			if c := cosine(npu[r*E:(r+1)*E], cpu[r*E:(r+1)*E]); c < worst {
				worst, wi = c, r
			}
		}
		var big float32
		for _, v := range cpu[:n*E] {
			big = max(big, float32(math.Abs(float64(v))))
		}
		t.Logf("layer %d: %d non-finite values on the NPU, worst row %d (token %d) %.5f, the CPU's largest %.0f", i, bad, wi, ids[wi], worst, big)
		x = cpu
	}
}

// TestGroupFormats writes the DiT's first group of image blocks (HWNPU_QWENIMAGE_ROOT) in fp16 and
// in each of HWNPU_QWENIMAGE_FORMATS (format or format:quant, comma separated) into
// HWNPU_QWENIMAGE_OUT, and runs them on the Neural Engine on the first step's input of the reference
// (HWNPU_QWENIMAGE_REF's dit_in0, the export's text keys and values of HWNPU_QWENIMAGE_EXPORT): each
// format's output against fp16's.
func TestGroupFormats(t *testing.T) {
	dir, root, out, ref := os.Getenv("HWNPU_QWENIMAGE_EXPORT"), os.Getenv("HWNPU_QWENIMAGE_ROOT"), os.Getenv("HWNPU_QWENIMAGE_OUT"), os.Getenv("HWNPU_QWENIMAGE_REF")
	if dir == "" || root == "" || out == "" || ref == "" {
		t.Skip("HWNPU_QWENIMAGE_EXPORT, _ROOT, _OUT and _REF not set")
	}
	ctx := context.Background()
	cache, _ := os.UserCacheDir()
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(cache, "hwnpu")))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	p, err := OpenWith(ctx, dev, dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	e := p.prog
	ids := p.Tokens("A red fox sitting in a snowy forest at dawn, soft light")
	T := len(ids) - p.drop
	E, TR := p.Dit.Hidden, p.Dit.TextRows
	x := make([]float32, TR*E)
	for i, v := range readF32s(t, ref+".txtin.f32") {
		x[i] = v / mil.DitScale
	}
	mod0, _ := e.time.modulation(0)
	var tk, tv []float32
	if err := e.each(ctx, e.m.DitText[0], func(pr runner.Program) error {
		y, err := pr.Run(ctx, map[string][]float32{"x": x, "mod": mod0})
		for j := 0; j < e.m.Dit.Group; j++ {
			tk = append(tk, y[(1+2*j)*TR*E:(2+2*j)*TR*E]...)
			tv = append(tv, y[(2+2*j)*TR*E:(3+2*j)*TR*E]...)
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
	ii := e.imageInputs(Cond{Rows: T})
	mod, _ := e.time.modulation(1)
	in := map[string][]float32{"x": readF32s(t, ref+".dit_in0.f32"), "mod": mod, "cos": ii.cos, "sin": ii.sin, "tk": tk, "tv": tv, "kbias": ii.kbias}
	c, err := mil.ReadDit(root, p.Dit.Width, p.Dit.Height)
	if err != nil {
		t.Fatal(err)
	}
	w, err := mil.OpenWeights(root, "")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	run := func(spec string) []float32 {
		format, quant, _ := strings.Cut(spec, ":")
		path := filepath.Join(out, "g0-"+strings.ReplaceAll(spec, ":", "-")+".mlpackage")
		if _, err := os.Stat(path); err != nil {
			t0 := time.Now()
			o := llm.Options{Backend: llm.BackendCoreML, Format: format, Quant: quant}
			if f, rest, ok := strings.Cut(format, "+"); ok { // lut4+attn.to_=lut8: the rest as FormatFor
				o.Format, o.FormatFor = f, strings.Split(rest, "+")
			}
			if _, err := llm.WritePart(path, mil.Blocks(c, 0, e.m.Dit.Group, mil.QChunk(c, 0), false), w, o, nil); err != nil {
				t.Fatal(err)
			}
			t.Logf("%s written in %.0f s", spec, time.Since(t0).Seconds())
		}
		pr, err := runner.Open(ctx, dev, path)
		if err != nil {
			t.Fatal(err)
		}
		defer pr.Close()
		y, err := pr.Run(ctx, in)
		if err != nil {
			t.Fatal(err)
		}
		return y
	}
	base := run("fp16")
	for _, spec := range strings.Split(cmp.Or(os.Getenv("HWNPU_QWENIMAGE_FORMATS"), "lut4,lut4:kmeans,lut6,lut8"), ",") {
		y := run(spec)
		var num, den float64
		for i := range y {
			d := float64(y[i] - base[i])
			num += d * d
			den += float64(base[i]) * float64(base[i])
		}
		t.Logf("%s: cosine %.5f, relative RMS error %.4f", spec, cosine(y, base), math.Sqrt(num/den))
	}
}

// TestParallelLoad loads the DiT's image blocks of HWNPU_QWENIMAGE_EXPORT HWNPU_QWENIMAGE_PAR at a
// time (1, 2, 4): whether the Neural Engine's compiles of programs its cache has dropped overlap.
func TestParallelLoad(t *testing.T) {
	dir := os.Getenv("HWNPU_QWENIMAGE_EXPORT")
	if dir == "" {
		t.Skip("HWNPU_QWENIMAGE_EXPORT not set")
	}
	ctx := context.Background()
	cache, _ := os.UserCacheDir()
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(cache, "hwnpu")))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	m, err := ReadProgramManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, par := range strings.Split(cmp.Or(os.Getenv("HWNPU_QWENIMAGE_PAR"), "1,2,4"), ",") {
		n := 0
		fmt.Sscan(par, &n)
		// the text encoder's programs first, so that the Neural Engine's cache drops the blocks
		t1 := time.Now()
		for _, name := range m.TextIn {
			p, err := runner.Open(ctx, dev, filepath.Join(dir, name))
			if err != nil {
				t.Fatal(err)
			}
			p.Close()
		}
		t.Logf("the text's programs loaded in %.1f s", time.Since(t1).Seconds())
		t0 := time.Now()
		progs := make([]runner.Program, len(m.Blocks))
		sem := make(chan struct{}, n)
		errs := make(chan error, len(m.Blocks))
		for i, name := range m.Blocks {
			sem <- struct{}{}
			go func() {
				defer func() { <-sem }()
				p, err := runner.Open(ctx, dev, filepath.Join(dir, name))
				progs[i] = p
				errs <- err
			}()
		}
		for range m.Blocks {
			if err := <-errs; err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("%d at a time: %d programs in %.1f s", n, len(m.Blocks), time.Since(t0).Seconds())
		for _, p := range progs {
			p.Close()
		}
	}
}

// TestTiledVae runs each tiled VAE stage of an export (HWNPU_QWENIMAGE_EXPORT) on its tiles and
// the same stage of an export of the same size without tiles (HWNPU_QWENIMAGE_UNTILED, an export
// of version 1) whole, on Core ML, from the same input: the tiles give what the whole stage gives.
func TestTiledVae(t *testing.T) {
	dir, whole := os.Getenv("HWNPU_QWENIMAGE_EXPORT"), os.Getenv("HWNPU_QWENIMAGE_UNTILED")
	if dir == "" || whole == "" {
		t.Skip("HWNPU_QWENIMAGE_EXPORT and HWNPU_QWENIMAGE_UNTILED not set")
	}
	ctx := context.Background()
	if infos, err := (coreml.Provider{}).Enumerate(ctx); err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	m, err := ReadProgramManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	cache, _ := os.UserCacheDir()
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementStrict), hwnpu.WithCacheDir(filepath.Join(cache, "hwnpu")))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	n := 0
	for i, tl := range m.VaeTiles {
		if !tl.tiled() {
			continue
		}
		n++
		x := make([]float32, tl.C*tl.H*tl.W)
		for j := range x { // smooth, as a decoder's activations: a few cycles across the picture
			r, c := j/tl.W%tl.H, j%tl.W
			x[j] = float32(math.Sin(float64(r)*0.21+float64(j/(tl.H*tl.W))) * math.Cos(float64(c)*0.17))
		}
		run := func(path string, tiled bool) []float32 {
			p, err := runner.Open(ctx, dev, path)
			if err != nil {
				t.Fatal(err)
			}
			defer p.Close()
			var y []float32
			if tiled {
				y, err = runTiled(ctx, p, x, tl)
			} else {
				y, err = p.Run(ctx, map[string][]float32{"x": x})
			}
			if err != nil {
				t.Fatal(err)
			}
			return y
		}
		t0 := time.Now()
		a := run(filepath.Join(whole, m.Vae[i]), false)
		ta := time.Since(t0)
		t0 = time.Now()
		b := run(filepath.Join(dir, m.Vae[i]), true)
		tb := time.Since(t0)
		var worst float64
		for j := range a {
			worst = max(worst, math.Abs(float64(a[j]-b[j])))
		}
		t.Logf("%s: %dx%d tiles of %dx%d: cosine %.6f, largest difference %.4g (whole %s, tiles %s, with the loads)",
			m.Vae[i], tl.Rows.N, tl.Cols.N, tl.Rows.Window(), tl.Cols.Window(), cosine(a, b), worst, ta.Round(time.Millisecond), tb.Round(time.Millisecond))
		if c := cosine(a, b); c < 0.9999 {
			t.Errorf("%s: cosine %.6f", m.Vae[i], c)
		}
	}
	if n == 0 {
		t.Errorf("%s: no tiled stage", dir)
	}
}
