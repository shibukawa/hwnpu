package music

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// On AMD's NPU SpectroStream's decoder runs as the graph of tools/amd/iron_magenta.py (an export
// NAME.amd-npu2 with decoder/), the language model, MusicCoCa and the encoder as Core ML packages
// on the CPU interpreter.

// AMDTool is the IRON exporter of SpectroStream's decoder (tools/amd, built into hwnpu).
const AMDTool = "iron_magenta.py"

// AMDExportVersion is the version of PrepareAMD's exports (a key of the shared layers:
// chat.PinnedRecipe), raised by hand when what PrepareAMD passes AMDTool changes.
const AMDExportVersion = 1

// AMDExportName is the name of the export for AMD's NPU: NAME.amd-npu2 in lower case.
func AMDExportName(name string) string { return strings.ToLower(name) + ".amd-npu2" }

// ironMeta is model.json of an export of tools/amd/iron_magenta.py.
type ironMeta struct {
	Kind    string `json:"kind"`
	Frames  int    `json:"frames"`   // the window's most frames
	Rows    int    `json:"rows"`     // the input's rows
	TimeOut int    `json:"time_out"` // the output's STFT frames
	Bins    int    `json:"bins"`
}

func readIronMeta(dir string) (*ironMeta, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	var m ironMeta
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("music: %s: model.json: %w", dir, err)
	}
	return &m, nil
}

// IsIronExport reports whether dir is an export of tools/amd/iron_magenta.py.
func IsIronExport(dir string) bool {
	m, err := readIronMeta(dir)
	if err != nil || m.Kind != "magenta-rt2" {
		return false
	}
	_, err = os.Stat(filepath.Join(dir, "decoder", "manifest.json"))
	return err == nil
}

// PrepareAMD makes the export of SpectroStream's decoder of m for AMD's NPU ready and returns its
// directory: here (cache.ExportRoots; one made by hand goes in the sources written back to),
// else from the shared layers, else it downloads the checkpoint and exports it with IRON in
// IRON's environment (chat.IRONEnv) into the cache's entries/amd-xdna2, and puts it in the
// sources written back to.
func PrepareAMD(ctx context.Context, m chat.Model, log io.Writer) (string, error) {
	if log == nil {
		log = io.Discard
	}
	name := AMDExportName(m.Name)
	if d := cache.FindExport(name, IsIronExport); d != "" {
		chat.ShareMadeByHand(m, d, log)
		return d, nil
	}
	out := filepath.Join(chat.AMDRoot(), name)
	imported := []chat.SharedRecipe{{Kind: chat.ImportKind, Recipe: chat.ImportedRecipe(m, cache.AMDTarget, name)}}
	err := chat.SharedExportOr(ctx, chat.ExportKind(AMDTool), chat.PinnedRecipe(m, AMDTool, AMDExportVersion), imported, out, log, func() error {
		python, err := chat.IRONPython(ctx)
		if err != nil {
			return fmt.Errorf("%s has no export for AMD's NPU here (%s) or in the sources, and making one needs IRON (amd/IRON with mlir-aie and Peano, Python 3.12): %w. Install it as INSTALL_AMD.md says and set %s to its virtual environment or to a script that sets it up, or put an export made elsewhere in a source (hwnpu-dev models import)",
				m.Name, out, err, chat.IRONEnv)
		}
		fmt.Fprintf(log, "%s: no export of SpectroStream's decoder for AMD's NPU here or in the sources; exporting it with IRON\n", m.Name)
		if err := chat.FetchFiles(ctx, []chat.File{{Repo: m.Repo, Name: Files[0]}}, log); err != nil {
			return err
		}
		return chat.RunIRON(ctx, python, m.Name, AMDTool, []string{chat.RepoDir(m.Repo), chat.ExportOut}, out, log)
	})
	if err != nil {
		return "", err
	}
	if !IsIronExport(out) {
		return "", fmt.Errorf("music: %s is no whole export of %s", out, AMDTool)
	}
	return out, nil
}

// OpenAMD loads the programs of p as Open does on the CPU interpreter, SpectroStream's decoder
// the graph of the export p.IronDecoder on amd, AMD's NPU.
func OpenAMD(ctx context.Context, amd *hwnpu.Device, p Paths) (*Generator, error) {
	if p.IronDecoder == "" {
		return nil, fmt.Errorf("music: no export of SpectroStream's decoder for AMD's NPU")
	}
	meta, err := readIronMeta(p.IronDecoder)
	if err != nil {
		return nil, err
	}
	gr, err := iron.OpenGraph(ctx, amd, filepath.Join(p.IronDecoder, "decoder"))
	if err != nil {
		return nil, fmt.Errorf("music: SpectroStream's decoder: %w", err)
	}
	d := &ironDecoder{g: gr, m: meta}
	q := p
	q.Decoder = "" // opened here
	// (openOn closes the step, the graph with it, when it fails)
	return openOn(ctx, nil, q, "", &step{name: "spectrostream-decoder", onNPU: true, closer: d, run: d.run})
}

// ironDecoder runs the decoder's graph as the Core ML program's step: x (256, W) channels first
// and its masks in, y (2, 2, 4 W, 480) out, divided by DecoderOutScale. The graph has no masks:
// its rows start at the window's first frame of the clip (the frames before it are the program's
// masked ones, which its causal convolutions read as zeros: the grids' zero rows are the same).
type ironDecoder struct {
	g  *iron.Graph
	m  *ironMeta
	in []byte
	o  []float32
}

func (d *ironDecoder) Close() error { return d.g.Close() }

func (d *ironDecoder) run(in map[string][]float32) (map[string][]float32, error) {
	x, mask := in["x"], in["mask"]
	W := len(mask)
	if len(x) != EmbeddingSize*W {
		return nil, fmt.Errorf("music: the decoder's x of %d values for %d frames", len(x), W)
	}
	i0 := 0
	for i0 < W && mask[i0] == 0 {
		i0++
	}
	n := W - i0
	if n > d.m.Frames {
		return nil, fmt.Errorf("music: %d frames for the decoder's graph of %d", n, d.m.Frames)
	}
	if d.in == nil {
		d.in = make([]byte, 2*d.m.Rows*EmbeddingSize)
	}
	clear(d.in)
	row := make([]float32, EmbeddingSize)
	for r := range n {
		for c := range EmbeddingSize {
			row[c] = x[c*W+i0+r]
		}
		putBF16(d.in[2*r*EmbeddingSize:], row)
	}
	if err := d.g.SetInput(d.g.Inputs()[0].Name, d.in); err != nil {
		return nil, err
	}
	if err := d.g.Run(context.Background(), nil); err != nil {
		return nil, fmt.Errorf("music: SpectroStream's decoder: %w", err)
	}
	K := timeStride * W // the program's STFT frames
	y := make([]float32, 4*K*stftBins)
	B := d.m.Bins
	for b, name := range []string{"out0", "out1"} {
		out, err := d.g.Output(name)
		if err != nil {
			return nil, err
		}
		cols := len(out) / 2 / (d.m.TimeOut * B)
		if cols < 2 || B != stftBins {
			return nil, fmt.Errorf("music: the decoder's output %s of %d bytes", name, len(out))
		}
		if len(d.o) < len(out)/2 {
			d.o = make([]float32, len(out)/2)
		}
		vecops.BF16ToF32(d.o[:len(out)/2], out)
		for k := timeStride * i0; k < K && k-timeStride*i0 < d.m.TimeOut; k++ {
			src := (k - timeStride*i0) * B * cols
			for c := range 2 {
				dst := ((b*2+c)*K + k) * stftBins
				for bin := range stftBins {
					y[dst+bin] = d.o[src+bin*cols+c] / DecoderOutScale
				}
			}
		}
	}
	return map[string][]float32{"y": y}, nil
}

// putBF16 writes xs as bf16 (to the nearest even) into b.
func putBF16(b []byte, xs []float32) {
	for i, v := range xs {
		bits := math.Float32bits(v)
		var h uint16
		if bits&0x7fffffff > 0x7f800000 {
			h = uint16(bits>>16) | 0x40
		} else {
			h = uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
		}
		b[2*i], b[2*i+1] = byte(h), byte(h>>8)
	}
}
