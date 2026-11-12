package chroma

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/imagegen/chroma/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// The tests here check a Core ML export's programs on the Neural Engine against the float32
// references tools/ref/ref_chroma_parts.py and ref_chroma_lean.py write, as raw float32 files
// (NAME.f32) in HWNPU_CHROMA_REF. HWNPU_CHROMA_EXPORT is the export; HWNPU_CHROMA_CPU=1 runs the
// CPU interpreter instead of the device.

func refDir(t *testing.T) (export, ref string) {
	export, ref = os.Getenv("HWNPU_CHROMA_EXPORT"), os.Getenv("HWNPU_CHROMA_REF")
	if export == "" || ref == "" {
		t.Skip("HWNPU_CHROMA_EXPORT and HWNPU_CHROMA_REF name an export and a reference")
	}
	return export, ref
}

func readRef(t *testing.T, dir, name string) []float32 {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, name+".f32"))
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func testDevice(t *testing.T) *hwnpu.Device {
	if os.Getenv("HWNPU_CHROMA_CPU") == "1" {
		return nil
	}
	cache := ""
	if d, err := os.UserCacheDir(); err == nil {
		cache = filepath.Join(d, "hwnpu")
	}
	dev, _, err := llm.OpenDevice(context.Background(), llm.BackendCoreML, cache, hwnpu.WithPlacement(hwnpu.PlacementStrict))
	if err != nil {
		t.Skip(err)
	}
	t.Cleanup(func() { dev.Close() })
	return dev
}

// compare is the cosine, the relative RMS error and the largest error of got against want.
func compare(got, want []float32) (cos, rel, maxErr float64) {
	var gw, gg, ww, ee float64
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		gw += g * w
		gg += g * g
		ww += w * w
		e := g - w
		ee += e * e
		maxErr = max(maxErr, math.Abs(e))
	}
	return gw / math.Sqrt(gg*ww+1e-300), math.Sqrt(ee / (ww + 1e-300)), maxErr
}

func psnr(got, want []float32) float64 {
	var se float64
	for i := range want {
		clamp := func(v float32) float64 { return math.Max(-1, math.Min(1, float64(v))) }
		e := (clamp(got[i]) - clamp(want[i])) / 2
		se += e * e
	}
	return 10 * math.Log10(1/(se/float64(len(want))))
}

func runProgram(t *testing.T, dev *hwnpu.Device, path string, in map[string][]float32) []float32 {
	t.Helper()
	t0 := time.Now()
	p, err := runner.Open(context.Background(), dev, path)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if dev != nil && !p.OnNPU() {
		t.Errorf("%s is not all on the NPU", filepath.Base(path))
	}
	t1 := time.Now()
	y, err := p.Run(context.Background(), in)
	if err != nil {
		t.Fatalf("%s: %v", filepath.Base(path), err)
	}
	t.Logf("%s: loaded in %.1f s, ran in %.2f s", filepath.Base(path), t1.Sub(t0).Seconds(), time.Since(t1).Seconds())
	return y
}

// TestVAEOnDevice decodes ref_chroma_parts.py vae's latents (latents.f32) and compares the
// picture with its image.f32.
func TestVAEOnDevice(t *testing.T) {
	export, ref := refDir(t)
	dev := testDevice(t)
	m, err := ReadProgramManifest(export)
	if err != nil {
		t.Fatal(err)
	}
	x := readRef(t, ref, "latents")
	want := readRef(t, ref, "image")
	for i, name := range m.Vae {
		x = runProgram(t, dev, filepath.Join(export, name), map[string][]float32{"x": x})
		var bad int
		for _, v := range x {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				bad++
			}
		}
		if bad > 0 {
			t.Fatalf("stage %d: %d values not finite", i, bad)
		}
	}
	cos, rel, me := compare(x, want)
	t.Logf("picture: cosine %.6f, rel RMS %.4f, max |err| %.3f, PSNR %.1f dB", cos, rel, me, psnr(x, want))
	if cos < 0.999 {
		t.Errorf("cosine %.6f", cos)
	}
}

// refInt is a reference's int scalar (keys.f32).
func refInt(t *testing.T, dir, name string) int {
	return int(readRef(t, dir, name)[0])
}

func scaled(x []float32, s float32) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * s
	}
	return out
}

// TestDiTOnDevice runs the export's text, approximator and DiT programs on the first DiT call of
// ref_chroma_lean.py --dump (its npz as raw files): each program on the reference's input, its
// output against the reference's (the image's rows and the prompt's).
func TestDiTOnDevice(t *testing.T) {
	export, ref := refDir(t)
	dev := testDevice(t)
	m, err := ReadProgramManifest(export)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Dit
	E, N, T := c.Hidden, c.Tokens(), c.TextRows
	keys := refInt(t, ref, "keys")
	only := os.Getenv("HWNPU_CHROMA_PART")
	check := func(what string, got, want []float32, rows ...[2]int) {
		var g, w []float32
		for _, r := range rows {
			g = append(g, got[r[0]*E:r[1]*E]...)
			w = append(w, want[r[0]*E:r[1]*E]...)
		}
		cos, rel, me := compare(g, w)
		t.Logf("%s: cosine %.6f, rel RMS %.4f, max |err| %.4g", what, cos, rel, me)
		if cos < 0.999 {
			t.Errorf("%s: cosine %.6f", what, cos)
		}
	}
	if only == "" || only == "text" {
		// T5's rows: the embedding rows of the ids, then each group
		ids := readRef(t, ref, "ids")
		emb, err := os.Open(filepath.Join(export, "text", "embed.bin"))
		if err != nil {
			t.Fatal(err)
		}
		H := m.T5.Hidden
		x := make([]float32, T*H)
		b := make([]byte, 2*H)
		for i := 0; i < T; i++ {
			if _, err := emb.ReadAt(b, int64(ids[i])*int64(2*H)); err != nil {
				t.Fatal(err)
			}
			copy(x[i*H:], scaled(fromBF16s(b), m.EmbedScale))
		}
		emb.Close()
		mask := maskBias(keys-1, T)
		for _, name := range m.Text {
			x = runProgram(t, dev, filepath.Join(export, name), map[string][]float32{"x": x, "mask": mask})
		}
		got, want := scaled(x, mil.DitScale), readRef(t, ref, "ctx_emb")
		var rows []string
		for i := 0; i < keys; i++ {
			c, _, _ := compare(got[i*E:(i+1)*E], want[i*E:(i+1)*E])
			rows = append(rows, fmt.Sprintf("%.4f", c))
		}
		t.Logf("rows' cosines: %v", rows)
		check("text rows (context_embedder)", got, want, [2]int{0, keys})
	}
	mods := readRef(t, ref, "mods")
	call := 0 // the call ref_chroma_lean.py --dump-call dumped
	if _, err := os.Stat(filepath.Join(ref, "dump_call.f32")); err == nil {
		call = refInt(t, ref, "dump_call")
	}
	if only == "" || only == "mods" {
		sigma := float64(readRef(t, ref, "sigma")[call])
		got := runProgram(t, dev, filepath.Join(export, m.Mods), map[string][]float32{"x": ModInput(1000*sigma, c.Mods, c.Mods, c.ApproxIn)})
		check("mods", got, mods, [2]int{0, c.Mods})
	}
	kb := maskBias(c.Rows(), c.Rows())
	for i := N + keys; i < c.Rows(); i++ {
		kb[i] = -1e4
	}
	both := [][2]int{{0, N}, {N, N + keys}}
	row := func(rs ...int) []float32 {
		var out []float32
		for _, r := range rs {
			out = append(out, mods[r*E:(r+1)*E]...)
		}
		return out
	}
	if only == "" || only == "doubles" {
		for gi, g := range m.Doubles {
			var rs []int
			for b := g.First; b < g.First+g.Count; b++ {
				for _, text := range []int{0, 1} {
					r := 3*c.Singles + 6*b + text*6*c.Doubles
					rs = append(rs, r, r+1, r+2, r+3, r+4, r+5)
				}
			}
			in := map[string][]float32{"mods": row(rs...), "kbias": kb}
			if gi == 0 {
				in["x"] = readRef(t, ref, "dit_in")[call*N*c.Latent : (call+1)*N*c.Latent]
				in["ctx"] = scaled(readRef(t, ref, "ctx_emb"), 1.0/mil.DitScale)
			} else {
				in["x"] = scaled(readRef(t, ref, fmt.Sprintf("d%d", g.First-1)), 1.0/mil.DitScale)
			}
			got := runProgram(t, dev, filepath.Join(export, g.Name), in)
			check(g.Name, scaled(got, mil.DitScale), readRef(t, ref, fmt.Sprintf("d%d", g.First+g.Count-1)), both...)
		}
	}
	if only == "" || only == "singles" {
		for gi, g := range m.Singles {
			if os.Getenv("HWNPU_CHROMA_LAST") == "1" && gi != len(m.Singles)-1 {
				continue
			}
			var rs []int
			for b := g.First; b < g.First+g.Count; b++ {
				rs = append(rs, 3*b, 3*b+1, 3*b+2)
			}
			last := gi == len(m.Singles)-1
			if last {
				rs = append(rs, c.Mods-2, c.Mods-1)
			}
			prev := fmt.Sprintf("d%d", c.Doubles-1)
			if g.First > 0 {
				prev = fmt.Sprintf("s%d", g.First-1)
			}
			got := runProgram(t, dev, filepath.Join(export, g.Name), map[string][]float32{"x": scaled(readRef(t, ref, prev), 1.0/mil.DitScale), "mods": row(rs...), "kbias": kb})
			if last {
				cos, rel, me := compare(got, readRef(t, ref, "dit_out")[call*N*c.Latent:(call+1)*N*c.Latent])
				t.Logf("%s (the velocity): cosine %.6f, rel RMS %.4f, max |err| %.4g", g.Name, cos, rel, me)
				if cos < 0.999 {
					t.Errorf("velocity: cosine %.6f", cos)
				}
			} else {
				check(g.Name, scaled(got, mil.DitScale), readRef(t, ref, fmt.Sprintf("s%d", g.First+g.Count-1)), both...)
			}
		}
	}
}

// TestLoadOnDevice loads each program of the export (HWNPU_CHROMA_EXPORT) and runs it twice on
// small inputs: its compile, load and call times, and whether it is all on the NPU.
func TestLoadOnDevice(t *testing.T) {
	export := os.Getenv("HWNPU_CHROMA_EXPORT")
	if export == "" {
		t.Skip("HWNPU_CHROMA_EXPORT names an export")
	}
	dev := testDevice(t)
	m, err := ReadProgramManifest(export)
	if err != nil {
		t.Fatal(err)
	}
	names := append(append([]string(nil), m.Text...), m.Mods)
	for _, g := range append(append([]Group(nil), m.Doubles...), m.Singles...) {
		names = append(names, g.Name)
	}
	if only := os.Getenv("HWNPU_CHROMA_PART"); only != "" {
		names = []string{only}
	}
	for _, name := range names {
		t0 := time.Now()
		p, err := runner.Open(context.Background(), dev, filepath.Join(export, name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		in := map[string][]float32{}
		for k, shape := range p.Inputs() {
			n := 1
			for _, d := range shape {
				n *= d
			}
			v := make([]float32, n)
			for i := range v {
				v[i] = float32((i*7919)%2003)/2003 - 0.5
			}
			if k == "kbias" || k == "mask" {
				clear(v)
			}
			in[k] = v
		}
		t1 := time.Now()
		var d []time.Duration
		for range 2 {
			t2 := time.Now()
			y, err := p.Run(context.Background(), in)
			if err != nil {
				p.Close()
				t.Fatalf("%s: %v", name, err)
			}
			d = append(d, time.Since(t2))
			for _, v := range y {
				if math.IsNaN(float64(v)) {
					t.Errorf("%s: NaN", name)
					break
				}
			}
		}
		t.Logf("%s: loaded in %.1f s (%v), calls %.2f s, %.2f s", name, t1.Sub(t0).Seconds(), p.OnNPU(), d[0].Seconds(), d[1].Seconds())
		p.Close()
	}
}

// TestTextFirstGroup runs the first text program alone against t5l4.f32 (T5's residual stream
// after 4 layers, transformers' T5Block in float32) over the prompt's rows.
func TestTextFirstGroup(t *testing.T) {
	export, ref := refDir(t)
	dev := testDevice(t)
	m, err := ReadProgramManifest(export)
	if err != nil {
		t.Fatal(err)
	}
	T, H := m.Dit.TextRows, m.T5.Hidden
	keys := refInt(t, ref, "keys")
	ids := readRef(t, ref, "ids")
	emb, _ := os.Open(filepath.Join(export, "text", "embed.bin"))
	defer emb.Close()
	x := make([]float32, T*H)
	b := make([]byte, 2*H)
	for i := 0; i < T; i++ {
		emb.ReadAt(b, int64(ids[i])*int64(2*H))
		copy(x[i*H:], scaled(fromBF16s(b), m.EmbedScale))
	}
	y := runProgram(t, dev, filepath.Join(export, m.Text[0]), map[string][]float32{"x": x, "mask": maskBias(keys-1, T)})
	want := readRef(t, ref, "t5l4")
	for _, n := range []int{1, 4, keys - 1} {
		cos, rel, _ := compare(scaled(y[:n*H], mil.T5Scale), want[:n*H])
		t.Logf("rows 0..%d: cosine %.6f rel %.4f", n, cos, rel)
	}
}

// TestTextGroupsDeviceVsCPU runs each text program on the device and on the CPU interpreter on
// the same input (the CPU's chain), over the prompt's rows.
func TestTextGroupsDeviceVsCPU(t *testing.T) {
	export, ref := refDir(t)
	dev := testDevice(t)
	m, err := ReadProgramManifest(export)
	if err != nil {
		t.Fatal(err)
	}
	T, H := m.Dit.TextRows, m.T5.Hidden
	keys := refInt(t, ref, "keys")
	ids := readRef(t, ref, "ids")
	emb, _ := os.Open(filepath.Join(export, "text", "embed.bin"))
	defer emb.Close()
	x := make([]float32, T*H)
	b := make([]byte, 2*H)
	for i := 0; i < T; i++ {
		emb.ReadAt(b, int64(ids[i])*int64(2*H))
		copy(x[i*H:], scaled(fromBF16s(b), m.EmbedScale))
	}
	mask := maskBias(keys-1, T)
	chain := os.Getenv("HWNPU_CHROMA_CHAIN") == "1"
	xn := x
	for _, name := range m.Text {
		in := map[string][]float32{"x": x, "mask": mask}
		cpu := runProgram(t, nil, filepath.Join(export, name), in)
		if chain {
			in = map[string][]float32{"x": xn, "mask": mask}
		}
		npu := runProgram(t, dev, filepath.Join(export, name), in)
		if chain && name == m.Text[len(m.Text)-1] {
			cpu2 := runProgram(t, nil, filepath.Join(export, name), in)
			n := (keys - 1) * len(cpu) / T
			c1, _, _ := compare(cpu2[:n], cpu[:n])
			c2, _, _ := compare(npu[:n], cpu2[:n])
			t.Logf("%s on the device's input: the CPU against the CPU's chain %.6f, the device against the CPU %.6f", name, c1, c2)
		}
		xn = npu
		if len(cpu) == T*H {
			// the channels under 50 in every prompt row
			var a, b []float32
			for i := 0; i < keys-1; i++ {
				for j := 0; j < H; j++ {
					if v := cpu[i*H+j]; math.Abs(float64(v)) < 50 {
						a, b = append(a, npu[i*H+j]), append(b, v)
					}
				}
			}
			c, r, _ := compare(a, b)
			t.Logf("%s: the small values (%d): cosine %.6f rel %.4f", name, len(a), c, r)
		}
		n := (keys - 1) * len(cpu) / T
		cos, rel, me := compare(npu[:n], cpu[:n])
		var big float64
		for _, v := range cpu[:n] {
			big = max(big, math.Abs(float64(v)))
		}
		t.Logf("%s: device against CPU cosine %.6f rel %.4f max |err| %.4g (|y| max %.4g)", name, cos, rel, me, big)
		x = cpu
	}
}

// TestCallsOnDevice runs each DiT call of the reference (its input and sigma, the reference's
// text rows) through the export's DiT and compares the velocities.
func TestCallsOnDevice(t *testing.T) {
	export, ref := refDir(t)
	dev := testDevice(t)
	p, err := Open(context.Background(), dev, export, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	c := p.C
	sig := readRef(t, ref, "sigma")
	in, out := readRef(t, ref, "dit_in"), readRef(t, ref, "dit_out")
	n := c.Tokens() * c.Latent
	var s []float64
	for _, v := range sig {
		s = append(s, float64(v))
	}
	e := p.eng.(*progEngine)
	if err := e.openDiT(context.Background(), s); err != nil {
		t.Fatal(err)
	}
	// the text rows: the reference's (ref_chroma_lean.py --dump), else the export's T5 on the
	// reference's tokens, and the negative prompt's for the guided calls (each after its prompt's)
	var cond, neg Cond
	encode := func(ids []float32, keys int) Cond {
		T, H := c.TextRows, e.m.T5.Hidden
		emb, err := os.Open(filepath.Join(export, "text", "embed.bin"))
		if err != nil {
			t.Fatal(err)
		}
		defer emb.Close()
		x := make([]float32, T*H)
		b := make([]byte, 2*H)
		for i := 0; i < T; i++ {
			if _, err := emb.ReadAt(b, int64(ids[i])*int64(2*H)); err != nil {
				t.Fatal(err)
			}
			copy(x[i*H:], scaled(fromBF16s(b), e.m.EmbedScale))
		}
		cs, err := e.encode(context.Background(), [][]float32{x}, []int{keys - 1})
		if err != nil {
			t.Fatal(err)
		}
		cs[0].Keys = keys
		return cs[0]
	}
	if _, err := os.Stat(filepath.Join(ref, "ctx_emb.f32")); err == nil {
		cond = Cond{rows: scaled(readRef(t, ref, "ctx_emb"), 1.0/mil.DitScale), Keys: refInt(t, ref, "keys")}
	} else {
		cond = encode(readRef(t, ref, "ids"), refInt(t, ref, "keys"))
	}
	guided := false
	if _, err := os.Stat(filepath.Join(ref, "neg_ids.f32")); err == nil {
		guided = true
		neg = encode(readRef(t, ref, "neg_ids"), refInt(t, ref, "neg_keys"))
	}
	only := -1
	if v := os.Getenv("HWNPU_CHROMA_CALL"); v != "" {
		fmt.Sscan(v, &only)
	}
	for k := range s {
		if only >= 0 && k != only {
			continue
		}
		cd := cond
		if guided && k > 0 && s[k] == s[k-1] && k%2 == 1 {
			cd = neg
		}
		vs, err := e.velocity(context.Background(), []Cond{cd}, in[k*n:(k+1)*n], s[k])
		if err != nil {
			t.Fatal(err)
		}
		cos, rel, _ := compare(vs[0], out[k*n:(k+1)*n])
		t.Logf("call %d, sigma %.4f: cosine %.6f rel %.4f", k, s[k], cos, rel)
	}
}
