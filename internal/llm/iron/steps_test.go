package iron

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/amd"
	"github.com/shibukawa/hwnpu/internal/xdna"
	"github.com/shibukawa/hwnpu/internal/xdna/xdnasim"
)

const simBackend = "xdna-sim-iron"

func init() {
	backend.Register(amd.Provider{
		ProviderName: simBackend,
		Nodes: func() ([]amd.Node, error) {
			return []amd.Node{{Path: "/dev/accel/accel9", Device: "0x1502", Revision: "0x00"}}, nil
		},
		OpenDriver: func(string) (xdna.Driver, error) {
			sim := xdnasim.New()
			sim.Meta.Cols = 4 // npu1 (Phoenix) has 4 columns for applications
			sim.Run = xdnasim.RunPassthrough
			return sim, nil
		},
		Hwmon: "-",
	})
}

// writeStepsExport writes an export of two versions packaged as an xclbin per step on the
// simulated passthrough design: the decode step copies the bank's first 512 bytes to its
// middle (a static stream), then 2*position+2 bytes of those to the logits (a per-call
// stream built from the scalar n).
func writeStepsExport(t *testing.T) string {
	dir := t.TempDir()
	write := func(name string, b []byte) {
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("x_sim.xclbin", xdnasim.PassthroughXclbin())
	prog := map[string]any{
		"abi": []string{"int32_t"}, "regs": 3, "msgs": []string{"n must be positive"},
		"ops": []any{
			[]any{"k", 1, "i32", 0}, []any{"bin", 2, "b", ">", 0, 1}, []any{"chk", 2, 0},
			[]any{"k", 1, "u32", xdnasim.PassthroughMagic}, []any{"w32", 1, 0}, []any{"hdr", 1, 3, 6, 4, 1},
		},
	}
	pb, _ := json.Marshal(prog)
	write("stream_sim.json", pb)
	insts := binary.LittleEndian.AppendUint32(nil, 0xC0FFEE01)
	write("decode_insts.bin", binary.LittleEndian.AppendUint32(insts, 512))
	bank := make([]byte, 4096)
	for i := 0; i < 256; i++ {
		binary.LittleEndian.PutUint16(bank[2*i:], uint16(math.Float32bits(float32(i+1))>>16))
	}
	write("bank_scratch.bin", bank)
	logits := Handle{Name: "logits", Shape: []int{64}, DType: "bfloat16", Argument: "output", Bytes: 128}
	values := []string{"token", "position", "start", "rows", "rope_start"}
	copyStep := Step{Xclbin: "x_sim.xclbin", Kernel: xdnasim.PassthroughKernel, Insts: []int{0, 2},
		Buffers: []StepBuffer{{"scratch", 0, 512}, {"scratch", 3072, 512}, {"scratch", 1024, 512}}}
	m := map[string]any{
		"version": 1, "model": "Sim",
		"config":     map[string]int{"vocab_size": 64, "hidden_size": 1, "max_seq_len": 16, "prefill_chunk": 4, "num_hidden_layers": 1},
		"rope_start": "start", "restartable": true,
		"banks": map[string]Bank{"scratch": {Bytes: 4096, Image: "bank_scratch.bin"}},
		"versions": map[string]any{
			"decode": map[string]any{
				"insts": "decode_insts.bin",
				"steps": []Step{copyStep, {Xclbin: "x_sim.xclbin", Kernel: xdnasim.PassthroughKernel, Program: "stream_sim.json",
					Scalars: []string{"n"}, Buffers: []StepBuffer{{"scratch", 1024, 512}, {"scratch", 3072, 512}, {"output", 0, 128}}}},
				"words":     map[string]Param{"n": {Name: "n", Form: &Affine{Value: "position", Scale: 1, Mul: 2, Add: 2}}},
				"arguments": []Argument{{Index: 0, Kind: "output", Bytes: 128}, {Index: 1, Kind: "scratch", Bytes: 4096}},
				"values":    values, "outputs": []Handle{logits},
			},
			"prompt": map[string]any{
				"insts":     "decode_insts.bin",
				"steps":     []Step{copyStep},
				"arguments": []Argument{{Index: 0, Kind: "input", Bytes: 16}, {Index: 1, Kind: "output", Bytes: 128}, {Index: 2, Kind: "scratch", Bytes: 4096}},
				"values":    values, "outputs": []Handle{logits},
				"inputs": []Handle{{Name: "ids", Shape: []int{4}, DType: "int32", Argument: "input", Bytes: 16}},
			},
		},
	}
	mb, err := json.MarshalIndent(m, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	write("manifest.json", mb)
	return dir
}

func TestSteps(t *testing.T) {
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend(simBackend))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	m, err := Open(ctx, dev, writeStepsExport(t), t.Logf)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, pos := range []int{3, 10} {
		logits, err := m.Step(ctx, 7, pos)
		if err != nil {
			t.Fatal(err)
		}
		for i, x := range logits {
			want := float32(0)
			if i <= pos {
				want = float32(i + 1)
			}
			if pos == 10 || i <= pos { // the first call's tail is overwritten by the second
				if x != want {
					t.Fatalf("position %d: logit %d = %v, want %v", pos, i, x, want)
				}
			}
		}
	}
	// The per-call program refuses n <= 0 (here: never, n = 2*position+2), and the prompt
	// version runs its static step.
	if _, err := m.Prompt(ctx, []int32{1, 2, 3}, 0); err != nil {
		t.Fatal(err)
	}
}

// TestExportStreams checks an export packaged as an xclbin per step (HWNPU_IRON_EXPORT, an
// export directory): every stream check/streams.json keeps, which mlir-aie's dispatch library
// built, is what the Go interpreter builds from the same scalars. No NPU is needed.
func TestExportStreams(t *testing.T) {
	dir := os.Getenv("HWNPU_IRON_EXPORT")
	if dir == "" {
		t.Skip("set HWNPU_IRON_EXPORT to an export with check/streams.json")
	}
	b, err := os.ReadFile(filepath.Join(dir, "check", "streams.json"))
	if err != nil {
		t.Skip(err)
	}
	var checks []struct {
		Program string
		Scalars []int64
		Stream  []uint32
	}
	if err := json.Unmarshal(b, &checks); err != nil {
		t.Fatal(err)
	}
	progs := map[string]*Program{}
	for _, c := range checks {
		p, ok := progs[c.Program]
		if !ok {
			if p, err = ReadProgram(filepath.Join(dir, c.Program)); err != nil {
				t.Fatal(err)
			}
			progs[c.Program] = p
		}
		got, err := p.Run(nil, c.Scalars)
		if err != nil {
			t.Fatalf("%s %v: %v", c.Program, c.Scalars, err)
		}
		if len(got) != len(c.Stream) {
			t.Fatalf("%s %v: %d words, want %d", c.Program, c.Scalars, len(got), len(c.Stream))
		}
		for i := range got {
			if got[i] != c.Stream[i] {
				t.Fatalf("%s %v: word %d is %#x, want %#x", c.Program, c.Scalars, i, got[i], c.Stream[i])
			}
		}
	}
	t.Logf("%d streams of %d programs", len(checks), len(progs))
}

// TestExportLoads loads an export packaged as an xclbin per step (HWNPU_IRON_EXPORT) on the
// simulated NPU of 4 columns, as on NPU1: its xclbins become hardware contexts and its steps'
// runs are made, with every buffer bound. The simulated design runs nothing of the model.
func TestExportLoads(t *testing.T) {
	dir := os.Getenv("HWNPU_IRON_EXPORT")
	if dir == "" {
		t.Skip("set HWNPU_IRON_EXPORT to an export")
	}
	ctx := context.Background()
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend(simBackend))
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	m, err := Open(ctx, dev, dir, t.Logf)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	t.Logf("%d xclbins", len(m.xclbins))
	values, err := m.complete(m.decode, callValues(7, 100, 0, 1))
	if err != nil {
		t.Fatal(err)
	}
	if err := m.buildStreams(m.decode, values); err != nil {
		t.Fatal(err)
	}
	C := m.Manifest.Config.PrefillChunk
	if values, err = m.complete(m.prompt, callValues(7, 2*C-1, C, C)); err != nil {
		t.Fatal(err)
	}
	if err := m.buildStreams(m.prompt, values); err != nil {
		t.Fatal(err)
	}
}
