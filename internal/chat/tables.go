package chat

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/safetensors"
)

// TablesFile is the file in a model's directory that holds the tables the host reads rows of
// while it runs the model (llm.HostTables: the token embedding, Gemma 4's per-layer embeddings),
// copied out of the checkpoint as stored. Running the converted programs reads it instead of the
// checkpoint (docs/model-fit.md T5), so the checkpoint is only needed to convert.
const TablesFile = "tables.safetensors"

// WeightsIn is the file a model's directory gives Config.Weights: its tables when they are there,
// else the checkpoint.
func WeightsIn(dir string) string {
	if p := filepath.Join(dir, TablesFile); exists(p) {
		return p
	}
	return filepath.Join(dir, "model.safetensors")
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// hasTables reports whether path is a tables file with all of names.
func hasTables(path string, names []string) bool {
	f, err := safetensors.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	for _, n := range names {
		if _, ok := f.Lookup(n); !ok {
			return false
		}
	}
	return true
}

// writeTables makes dir's tables file for the model unless it is there with all its tables, and
// returns its path. checkpoint gives the checkpoint's path, downloading it if need be.
func writeTables(dir string, a *llm.Arch, model llm.Model, checkpoint func() (string, error), log io.Writer) (string, error) {
	names := llm.HostTables(model)
	if len(names) == 0 {
		return "", fmt.Errorf("chat: %s has no tables the host reads", a.Name)
	}
	return Tables(filepath.Join(dir, TablesFile), a, names, checkpoint, log)
}

// Tables writes the named tensors of the checkpoint to out (llm.WriteTables) unless out has them
// all, and returns out: the tables a command reads rows of while it runs converted programs, so
// that it runs without the checkpoint (docs/model-fit.md T5). checkpoint gives the checkpoint's
// path, downloading it if need be; a opens it.
func Tables(out string, a *llm.Arch, names []string, checkpoint func() (string, error), log io.Writer) (string, error) {
	if hasTables(out, names) {
		return out, nil
	}
	path, err := checkpoint()
	if err != nil {
		return "", err
	}
	w, err := llm.OpenWeights(path, a)
	if err != nil {
		return "", err
	}
	defer w.Close()
	if err := llm.WriteTables(w, names, out); err != nil {
		return "", err
	}
	if log != nil {
		size := int64(0)
		if fi, err := os.Stat(out); err == nil {
			size = fi.Size()
		}
		fmt.Fprintf(log, "wrote %s (%.2f GB, %d tables): the converted programs run without the checkpoint, which is only needed to convert again\n", out, float64(size)/1e9, len(names))
	}
	return out, nil
}

// Converted reports whether a program was converted to out (ConvertProgram, Prepare): its
// NAME.done is there.
func Converted(out string) bool { return converted(out) }

// Program is what ConvertProgram writes: the program of Arch (empty: the one config.json names)
// in Format (empty: the architecture's), with the architecture's settings Params, Seq tokens per
// call and a KV cache of Cache tokens (0: none).
type Program struct {
	Arch, Format string
	Params       map[string]string
	Seq, Cache   int
	// Backend is the program format (llm.Options.Backend): Core ML (empty) or OpenVINO IR for
	// Intel's NPU, which takes a lookup table format as its integer one (llm.IntelFormat).
	Backend string
}

// ConvertProgram converts a program for the Neural Engine (or Intel's NPU, p.Backend) as Prepare
// does (the converter's defaults, the tables fitted as for the chat models) unless it was
// converted to out, and returns its path (an .mlpackage or .xml, or the parts.json of one split).
func ConvertProgram(ctx context.Context, config, weights, out string, p Program, log io.Writer) (string, error) {
	format := p.Format
	if p.Backend == llm.BackendIntel && format != "" {
		format = llm.IntelFormat(format)
	}
	return convert(ctx, config, weights, out, program{backend: p.Backend, arch: p.Arch, format: format, params: p.Params, seq: p.Seq, cache: p.Cache}, log)
}

// ProgramFile is where a program named name for backend goes in a model's directory dir: an
// .mlpackage next to the checkpoint for Core ML, an .xml in dir/intel for Intel's NPU (made
// here), so that one directory serves both.
func ProgramFile(dir, name, backend string) (string, error) {
	d, err := programDir(dir, backend)
	if err != nil {
		return "", err
	}
	return filepath.Join(d, name+programExt(backend)), nil
}

// converted reports whether convert finished writing out (its NAME.done is there).
func converted(out string) bool {
	return exists(out[:len(out)-len(filepath.Ext(out))] + ".done")
}
