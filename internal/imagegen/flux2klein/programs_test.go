package flux2klein

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein/mil"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// fakeProgram records its inputs and gives y of out values: its first input's first value plus
// add (the text groups: a value that tells the groups apart).
type fakeProgram struct {
	name string
	out  int
	add  float32
	log  *[]string
}

func (p *fakeProgram) Inputs() map[string][]int    { return nil }
func (p *fakeProgram) Output() []int               { return []int{p.out} }
func (p *fakeProgram) Metadata() map[string]string { return nil }
func (p *fakeProgram) OnNPU() bool                 { return true }
func (p *fakeProgram) Close() error                { return nil }

func (p *fakeProgram) Run(_ context.Context, in map[string][]float32) ([]float32, error) {
	var keys []string
	for _, k := range []string{"x", "ctx", "mods", "kbias"} {
		if v, ok := in[k]; ok {
			keys = append(keys, fmt.Sprintf("%s:%d", k, len(v)))
		}
	}
	*p.log = append(*p.log, p.name+" "+strings.Join(keys, " "))
	y := make([]float32, p.out)
	for i := range y {
		y[i] = in["x"][0] + p.add
	}
	return y, nil
}

func smallManifest() *ProgramManifest {
	m := &ProgramManifest{Version: ExportVersion,
		Dit:  Config{Width: 32, Height: 32, TextRows: 4, Hidden: 2, Latent: 3, Doubles: 3, Singles: 4},
		Text: TextConfig{Rows: 4, Hidden: 2}}
	for i, tap := range []int{-1, 0, 1, -1, 2} {
		m.TextGroups = append(m.TextGroups, TextProgram{Name: fmt.Sprintf("text_%d", i), Tap: tap})
	}
	m.Ctx = "ctx"
	m.Doubles = []Group{{"dit_double_0", 0, 2}, {"dit_double_2", 2, 1}}
	m.Singles = []Group{{"dit_single_0", 0, 2}, {"dit_single_2", 2, 2}}
	return m
}

// The text groups run in turn on every prompt, and the hidden states the taps name go to
// context_embedder side by side; a step runs the DiT's groups in order with the modulation rows
// of their kind.
func TestEngineFlow(t *testing.T) {
	m := smallManifest()
	var log []string
	c := m.Dit
	E, N := c.Hidden, c.Tokens()
	R := N + c.TextRows
	open := func(_ context.Context, name string) (runner.Program, error) {
		p := &fakeProgram{name: name, log: &log}
		switch {
		case strings.HasPrefix(name, "text_"):
			p.out = m.Text.Rows * m.Text.Hidden
			fmt.Sscanf(name, "text_%f", &p.add)
			p.add = 10 * (p.add + 1)
		case name == "ctx":
			p.out = c.TextRows * E
		case name == "dit_single_2":
			p.out = N * c.Latent
		default:
			p.out = R * E
		}
		return p, nil
	}
	te := &timeEmbed{E: E, w: [][]float32{make([]float32, E*4), make([]float32, E*E), make([]float32, 6*E*E),
		make([]float32, 6*E*E), make([]float32, 3*E*E), make([]float32, 2*E*E)}, b: make([][]float32, 6)}
	e := &progEngine{m: m, loader: open, logf: func(string, ...any) {}, time: te}
	xs := [][]float32{{1, 0, 0, 0, 0, 0, 0, 0}, {2, 0, 0, 0, 0, 0, 0, 0}}
	rows, err := e.encode(context.Background(), xs, []int{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	// ctx's x: the taps (groups 1, 2 and 4: 1 + 10 + 20, + 30, + 40 + 50) side by side, its first
	// value the first tap's
	if len(rows) != 2 || rows[0][0] != 31 || rows[1][0] != 32 {
		t.Errorf("text rows %v", rows)
	}
	want := []string{"text_0 x:8 kbias:4", "text_0 x:8 kbias:4", "text_1 x:8 kbias:4", "text_1 x:8 kbias:4"}
	if strings.Join(log[:4], "; ") != strings.Join(want, "; ") || log[len(log)-1] != "ctx x:24" {
		t.Errorf("text calls %v", log)
	}
	log = nil
	if err := e.openDiT(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	log = nil
	vs, err := e.velocities(context.Background(), rows[:1], [][]float32{make([]float32, N*c.Latent)}, nil, nil, 0.5)
	v := vs[0]
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != N*c.Latent {
		t.Errorf("%d values of velocity", len(v))
	}
	want = []string{
		fmt.Sprintf("dit_double_0 x:%d ctx:%d mods:%d", N*c.Latent, c.TextRows*E, mil.ModSingle*E),
		fmt.Sprintf("dit_double_2 x:%d mods:%d", R*E, mil.ModSingle*E),
		fmt.Sprintf("dit_single_0 x:%d mods:%d", R*E, 3*E),
		fmt.Sprintf("dit_single_2 x:%d mods:%d", R*E, 5*E),
	}
	if strings.Join(log, "; ") != strings.Join(want, "; ") {
		t.Errorf("a step's calls:\n%s\nwant\n%s", strings.Join(log, "\n"), strings.Join(want, "\n"))
	}
	// two pictures: each group runs both before the next group
	log = nil
	vs, err = e.velocities(context.Background(), rows, [][]float32{make([]float32, N*c.Latent), make([]float32, N*c.Latent)}, nil, nil, 0.5)
	if err != nil || len(vs) != 2 {
		t.Fatalf("%d velocities, %v", len(vs), err)
	}
	if len(log) != 8 || !strings.HasPrefix(log[0], "dit_double_0") || !strings.HasPrefix(log[1], "dit_double_0") ||
		!strings.HasPrefix(log[6], "dit_single_2") || !strings.HasPrefix(log[7], "dit_single_2") {
		t.Errorf("two pictures' calls:\n%s", strings.Join(log, "\n"))
	}
	e.closeDiT()
}

// An export of 512² and 1024² is named for both and gives each size's functions.
func TestSizes(t *testing.T) {
	if n := ExportName("FLUX.2-klein-4B", ConvertOptions{Width: 1024}); n != "flux.2-klein-4b-512+1024-lut8-lut8-v2.coreai" {
		t.Errorf("export name %s", n)
	}
	if n := ExportName("FLUX.2-klein-4B", ConvertOptions{Width: 768}); n != "flux.2-klein-4b-768x768-lut8-lut8-v2.coreai" {
		t.Errorf("export name of another size %s", n)
	}
	// models.yaml npu.apple's 8: one export, named by their count and a hash of the list
	eight := [][2]int{{1024, 1024}, {512, 512}, {768, 1152}, {768, 1024}, {704, 1280}, {1152, 768}, {1024, 768}, {1280, 704}}
	if n := ExportName("FLUX.2-klein-4B", ConvertOptions{Width: 1024, Height: 1024, Sizes: eight}); n != "flux.2-klein-4b-8sizes-4d02f40a-lut8-lut8-v2.coreai" {
		t.Errorf("export name of 8 sizes %s", n)
	}
	m := &ProgramManifest{Dit: Config{Width: 1024, Height: 1024}, Vae: []string{"vae_s0.aimodel#s1024x1024"},
		Sizes: []SizeParts{{Width: 512, Height: 512, Vae: []string{"vae_s0.aimodel#s512x512"}}, {Width: 1024, Height: 1024, Vae: []string{"vae_s0.aimodel#s1024x1024"}}}}
	if err := m.At(512, 512); err != nil || m.Dit.Width != 512 || m.Vae[0] != "vae_s0.aimodel#s512x512" {
		t.Errorf("at 512: %v %+v", err, m)
	}
	if err := m.At(768, 768); err == nil {
		t.Error("768 found in an export of 512 and 1024")
	}
}

// With the edit's programs a step with a reference runs their groups (from their directory),
// the reference's patches after the picture's in the first group's x; a step without one runs
// the export's again; without them a reference is refused.
func TestEngineEdit(t *testing.T) {
	m := smallManifest()
	var log []string
	c := m.Dit
	E, N := c.Hidden, c.Tokens()
	refN := N // one reference of the picture's size
	R := N + refN + c.TextRows
	open := func(_ context.Context, name string) (runner.Program, error) {
		p := &fakeProgram{name: name, log: &log}
		base := name[strings.LastIndex(name, "/")+1:]
		switch {
		case base == "dit_single_2" || base == "dit_single_2_ref1":
			p.out = N * c.Latent
		case strings.HasSuffix(base, "_ref1"):
			p.out = R * E
		default:
			p.out = (N + c.TextRows) * E
		}
		return p, nil
	}
	te := &timeEmbed{E: E, w: [][]float32{make([]float32, E*4), make([]float32, E*E), make([]float32, 6*E*E),
		make([]float32, 6*E*E), make([]float32, 3*E*E), make([]float32, 2*E*E)}, b: make([][]float32, 6)}
	e := &progEngine{m: m, loader: open, logf: func(string, ...any) {}, time: te}
	rows := [][]float32{make([]float32, c.TextRows*E)}
	x := [][]float32{make([]float32, N*c.Latent)}
	ref := make([]float32, refN*c.Latent)
	if err := e.openDiT(context.Background(), []int{1}); err == nil {
		t.Error("a reference without the edit's programs")
	}
	e.edit, e.editDir = &EditSize{Width: 32, Height: 32, Steps: []EditStep{{Refs: []int{1},
		Doubles: []Group{{"dit_double_0_ref1", 0, 3}}, Singles: []Group{{"dit_single_0_ref1", 0, 2}, {"dit_single_2_ref1", 2, 2}}}}}, "/edit"
	if err := e.openDiT(context.Background(), []int{1}); err != nil {
		t.Fatal(err)
	}
	log = nil
	if _, err := e.velocities(context.Background(), rows, x, ref, []int{1}, 0.5); err != nil {
		t.Fatal(err)
	}
	want := []string{
		fmt.Sprintf("/edit/dit_double_0_ref1 x:%d ctx:%d mods:%d", (N+refN)*c.Latent, c.TextRows*E, mil.ModSingle*E),
		fmt.Sprintf("/edit/dit_single_0_ref1 x:%d mods:%d", R*E, 3*E),
		fmt.Sprintf("/edit/dit_single_2_ref1 x:%d mods:%d", R*E, 5*E),
	}
	if strings.Join(log, "; ") != strings.Join(want, "; ") {
		t.Errorf("an edit's step:\n%s\nwant\n%s", strings.Join(log, "\n"), strings.Join(want, "\n"))
	}
	// text to image again: the export's groups, the edit's ring closed
	if err := e.openDiT(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	log = nil
	if _, err := e.velocities(context.Background(), rows, x, nil, nil, 0.5); err != nil {
		t.Fatal(err)
	}
	if len(log) != 4 || !strings.HasPrefix(log[0], "dit_double_0 x:") {
		t.Errorf("text to image after an edit:\n%s", strings.Join(log, "\n"))
	}
	if _, ok := e.edit.Step([]int{2}); ok {
		t.Error("a step of half a picture's sides in an export of the picture's size")
	}
	e.closeDiT()
}

// The edit's programs are named for their sizes and references, apart from the export's.
func TestEditExportName(t *testing.T) {
	if n := EditExportName("FLUX.2-klein-4B", EditOptions{}); n != "flux.2-klein-4b-edit-512+1024-ref1-lut8-v1.coreai" {
		t.Errorf("edit export name %s", n)
	}
	if n := EditExportName("FLUX.2-klein-4B", EditOptions{Sizes: [][2]int{{512, 512}}, Refs: [][]int{{1}, {2}}}); n != "flux.2-klein-4b-edit-512x512-ref1+refh1-lut8-v1.coreai" {
		t.Errorf("edit export name %s", n)
	}
}

// TestEditPatterns: the patterns' sizes and references, each references' sizes, and a name apart
// from every size with every references.
func TestEditPatterns(t *testing.T) {
	o := EditOptions{Patterns: EditPatterns}
	o.defaults()
	if !slices.Equal(o.Sizes, [][2]int{{1024, 1024}, {512, 512}}) || len(o.Refs) != 4 {
		t.Fatalf("sizes %v, refs %v", o.Sizes, o.Refs)
	}
	for _, c := range []struct {
		refs []int
		want [][2]int
	}{{[]int{1}, [][2]int{{1024, 1024}, {512, 512}}}, {[]int{1, 2}, [][2]int{{1024, 1024}}}, {[]int{1, 1}, [][2]int{{512, 512}}}, {[]int{1, 2, 2}, [][2]int{{1024, 1024}}}} {
		if got := o.sizesFor(c.refs); !slices.Equal(got, c.want) {
			t.Errorf("sizes of %v: %v, want %v", c.refs, got, c.want)
		}
	}
	n := EditExportName("FLUX.2-klein-4B", EditOptions{Patterns: EditPatterns})
	if !strings.HasPrefix(n, "flux.2-klein-4b-edit-5edits-") || !strings.HasSuffix(n, "-lut8-v1.coreai") {
		t.Errorf("edit export name %s", n)
	}
	if n == EditExportName("FLUX.2-klein-4B", EditOptions{Sizes: o.Sizes, Refs: o.Refs}) {
		t.Error("the patterns' name is every size's with every references'")
	}
}
