package chat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	shared "github.com/shibukawa/hwnpu/internal/store"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// PLaMo-Embedding on AMD's NPU is an export of tools/amd/iron_plamo_embed.py: a graph a sequence
// length (sNNN, 128 and 1024), each the encoder over its rows (the token embeddings in, a valid
// row count per call, the last hidden state out), their weights at the same places of one bank,
// which the graphs share (one pinned copy); the token embedding table (embed.bin) and the
// tokenizer the host reads. The host pools as the programs' (meanPool).

// EmbedAMDTool is the script of the export.
const EmbedAMDTool = "iron_plamo_embed.py"

// embedAMDModel is the export's model.json.
type embedAMDModel struct {
	Kind   string `json:"kind"`
	Seqs   []int  `json:"seqs"`
	Hidden int    `json:"hidden"`
}

func readEmbedAMD(dir string) (embedAMDModel, error) {
	var m embedAMDModel
	b, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return m, err
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return m, fmt.Errorf("chat: %s: %w", dir, err)
	}
	if m.Kind != "plamo-embed" || len(m.Seqs) == 0 {
		return m, fmt.Errorf("chat: %s is not an export of tools/amd/%s", dir, EmbedAMDTool)
	}
	return m, nil
}

// EmbedAMDDir is where model m's export for AMD's NPU is kept: NAME.amd-npu2 in lower case.
func EmbedAMDDir(m Model) string {
	return filepath.Join(AMDRoot(), strings.ToLower(m.Name)+".amd-npu2")
}

// FindEmbedAMDExport is embedding model m's export for AMD's NPU here, "" for none.
func FindEmbedAMDExport(m Model) string {
	return cache.FindExport(filepath.Base(EmbedAMDDir(m)), func(d string) bool { _, err := readEmbedAMD(d); return err == nil })
}

// PrepareEmbedderAMD makes embedding model m's export for AMD's NPU ready: here (cache.ExportRoots),
// else from the sources (by its recipe, or its name), else made with IRON from the checkpoint.
func PrepareEmbedderAMD(ctx context.Context, m Model, log io.Writer) (string, error) {
	out := EmbedAMDDir(m)
	if d := FindEmbedAMDExport(m); d != "" {
		return d, nil
	}
	var recipe map[string]string
	if len(shared.Sources()) > 0 {
		recipe = ExportRecipe(ctx, m, EmbedAMDTool, nil)
	}
	imported := []SharedRecipe{{Kind: ImportKind, Recipe: ImportedRecipe(m, cache.AMDTarget, filepath.Base(out))}}
	err := SharedExportOr(ctx, ExportKind(EmbedAMDTool), recipe, imported, out, log, func() error {
		python, err := ironPython(ctx)
		if err != nil {
			return fmt.Errorf("%s has no export for AMD's NPU here (%s) or in the sources, and making one needs IRON (amd/IRON with mlir-aie and Peano, Python 3.12): %w. Install it as INSTALL_AMD.md says and set %s to its virtual environment",
				m.Name, out, err, IRONEnv)
		}
		for _, f := range []string{"config.json", "tokenizer.model"} {
			if _, err := fetch(ctx, m.Source(), f, log); err != nil {
				return err
			}
		}
		if _, err := FetchWeights(ctx, m.Source(), log); err != nil {
			return err
		}
		return RunIRON(ctx, python, m.Name, EmbedAMDTool, []string{RepoDir(m.Source()), ExportOut}, out, log)
	})
	if err != nil {
		return "", err
	}
	if _, err := readEmbedAMD(out); err != nil {
		return "", err
	}
	return out, nil
}

// amdEmbed runs an export of EmbedAMDTool.
type amdEmbed struct {
	dev    *hwnpu.Device
	graphs []amdEmbedGraph // shortest first
	emb    *os.File        // embed.bin: a bf16 row a token
	shared []*hwnpu.Buffer // the banks the graphs share
	work   []int64         // where each's work area starts (its image's end: the weights')
	cur    int             // the graph run last (-1: none)
}

type amdEmbedGraph struct {
	seq int
	g   *iron.Graph
}

// LoadEmbedderAMD loads an export of EmbedAMDTool for AMD's NPU.
func LoadEmbedderAMD(ctx context.Context, dir, name string, log io.Writer) (*Embedder, error) {
	logf := func(format string, args ...any) {
		if log != nil {
			fmt.Fprintf(log, format+"\n", args...)
		}
	}
	mj, err := readEmbedAMD(dir)
	if err != nil {
		return nil, err
	}
	a, model, err := llm.LoadConfig(filepath.Join(dir, "config.json"), "")
	if err != nil {
		return nil, err
	}
	if a.Name != "plamo-bi" {
		return nil, fmt.Errorf("chat: %s is not an embedding model hwnpu-chat runs (only PLaMo-Embedding)", a.Title)
	}
	if model.Hidden() != mj.Hidden {
		return nil, fmt.Errorf("chat: %s: graphs of %d for a model of %d", dir, mj.Hidden, model.Hidden())
	}
	tok, err := tokenizer.LoadSentencePiece(filepath.Join(dir, "tokenizer.model"))
	if err != nil {
		return nil, err
	}
	e, err := newEmbedder(tok, model)
	if err != nil {
		return nil, err
	}
	e.Name = name
	if e.Name == "" {
		e.Name = strings.TrimSuffix(filepath.Base(dir), ".amd-npu2")
	}
	if err := llm.LockNPU(ctx); err != nil {
		return nil, err
	}
	x := &amdEmbed{cur: -1}
	e.amd = x
	if x.dev, err = hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD)); err != nil {
		e.Close()
		return nil, err
	}
	if x.emb, err = os.Open(filepath.Join(dir, "embed.bin")); err != nil {
		e.Close()
		return nil, err
	}
	iron.ShareExport(dir, func(f string, a ...any) { logf("  "+f, a...) })
	seqs := slices.Clone(mj.Seqs)
	slices.Sort(seqs)
	// the longest first: its banks (as large as any's) are lent to the others, whose weights
	// are at the same places (one pinned copy)
	var lend map[string]*hwnpu.Buffer
	for i := len(seqs) - 1; i >= 0; i-- {
		g, err := iron.OpenGraphWith(ctx, x.dev, filepath.Join(dir, fmt.Sprintf("s%d", seqs[i])), iron.GraphOptions{Buffers: lend})
		if err != nil {
			e.Close()
			return nil, err
		}
		x.graphs = append([]amdEmbedGraph{{seqs[i], g}}, x.graphs...)
		if lend == nil {
			lend = map[string]*hwnpu.Buffer{}
			for kind := range g.Manifest.Banks {
				n, err := g.ImageBytes(kind)
				if err != nil {
					e.Close()
					return nil, err
				}
				lend[kind] = g.Bank(kind)
				x.shared, x.work = append(x.shared, g.Bank(kind)), append(x.work, n)
			}
		}
	}
	logf("ready: embeddings of %d dimensions, up to %d tokens, on AMD's NPU (graphs of %v tokens)", e.Dim(), e.MaxTokens(), seqs)
	return e, nil
}

// maxTokens is the longest graph's tokens.
func (x *amdEmbed) maxTokens() int { return x.graphs[len(x.graphs)-1].seq }

// run reads ids on the shortest graph that holds them and returns the mean of the last hidden
// state over ids[skip:], scaled to length 1.
func (x *amdEmbed) run(ctx context.Context, ids []int32, skip, d int) ([]float32, error) {
	i := slices.IndexFunc(x.graphs, func(g amdEmbedGraph) bool { return g.seq >= len(ids) })
	if i < 0 {
		return nil, fmt.Errorf("%w: %d tokens, at most %d", ErrTooLong, len(ids), x.maxTokens())
	}
	if x.cur >= 0 && x.cur != i {
		// another length's work areas, in the shared banks past the weights: zero again, as a
		// graph finds them first
		for k, b := range x.shared {
			w := x.work[k]
			iron.ClearParallel(b.Bytes()[w:])
			if err := b.SyncToDevice(int(w), int(int64(b.Size())-w)); err != nil {
				return nil, err
			}
		}
	}
	x.cur = i
	g := x.graphs[i].g
	in := make([]byte, len(ids)*d*2)
	for r, id := range ids {
		if _, err := x.emb.ReadAt(in[r*d*2:(r+1)*d*2], int64(id)*int64(d)*2); err != nil {
			return nil, fmt.Errorf("chat: embed.bin: token %d: %w", id, err)
		}
	}
	if err := g.SetInput("x", in); err != nil {
		return nil, err
	}
	if err := g.Run(ctx, map[string]int64{"nk": int64(len(ids))}); err != nil {
		return nil, err
	}
	outs := g.Outputs()
	if len(outs) != 1 {
		return nil, errors.New("chat: the graph has not one output")
	}
	y, err := g.Output(outs[0].Name)
	if err != nil {
		return nil, err
	}
	h := make([]float32, len(ids)*d)
	vecops.BF16ToF32(h, y[:2*len(h)])
	return meanPool(h, d, skip), nil
}

func (x *amdEmbed) close() {
	for _, g := range x.graphs {
		g.g.Close() // shortest first, the longest last: the others borrow its banks
	}
	x.graphs = nil
	if x.emb != nil {
		x.emb.Close()
		x.emb = nil
	}
	if x.dev != nil {
		x.dev.Close()
		x.dev = nil
	}
}
