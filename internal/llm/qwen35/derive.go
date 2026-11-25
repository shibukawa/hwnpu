package qwen35

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/bonsaiai/internal/mxfp4"
	"github.com/shibukawa/bonsaiai/internal/safetensors"
	"github.com/shibukawa/bonsaiai/internal/vecops"
)

// Source reads the files of a checkpoint: small ones whole and ranges of the shards, so that a
// checkpoint far larger than the disk's free space (Qwen3.6-35B-A3B's 72 GB) is read a layer at a
// time from where it is (DirSource, or Hugging Face with HTTPSource in cmd/bonsaiai-dev/mxfp4). A
// file that is not there is an error wrapping fs.ErrNotExist.
type Source interface {
	ReadFile(name string) ([]byte, error)
	ReadRange(name string, off, n int64) ([]byte, error)
}

// DirSource is a checkpoint in a directory.
type DirSource string

func (d DirSource) ReadFile(name string) ([]byte, error) {
	return os.ReadFile(filepath.Join(string(d), name))
}

func (d DirSource) ReadRange(name string, off, n int64) ([]byte, error) {
	f, err := os.Open(filepath.Join(string(d), name))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b := make([]byte, n)
	if _, err := f.ReadAt(b, off); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	return b, nil
}

// DeriveFiles are the files besides config.json and the weights that DeriveMXFP4 copies when the
// source has them: what bonsaiai reads to tokenize and chat.
var DeriveFiles = []string{"tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "LICENSE"}

// DeriveKey is the key DeriveMXFP4 adds to config.json: what it did, and from what.
const DeriveKey = "bonsaiai_mxfp4"

// DeriveMXFP4 writes into out a checkpoint of a Qwen3.5-MoE model (qwen3_5_moe) whose experts
// are MXFP4, which the host runs as they are (MoE.HostExperts, llm.HostRouted): the release's
// experts are bf16, 64 GB of Qwen3.6-35B-A3B's 72, more than a 16 GB machine can map, and as
// MXFP4 they are 17 GB. A layer's experts gate_up_proj (experts, 2 inter, hidden) and down_proj
// (experts, hidden, inter) become gpt-oss's tensors (experts.MXFP4): the gate's and the up
// projection's rows interleaved (the gate the even rows), each row in blocks of 32 values with
// a power of two scale (mxfp4.Encode). Every other tensor of the text model is copied as it is;
// the vision encoder (model.visual.) and the multi-token prediction layer (mtp.) are left out.
//
// It writes a file a layer (model-layer-NN.safetensors) and one of the rest, each whole or not
// at all, and skips the files there already: a run cut off goes on where it stopped. The index,
// config.json (with DeriveKey) and DeriveFiles come last.
func DeriveMXFP4(src Source, from, out string, log io.Writer) error {
	raw, err := src.ReadFile("config.json")
	if err != nil {
		return err
	}
	c, err := ParseMoE(raw)
	if err != nil {
		return err
	}
	shards := map[string][]string{} // a source file's tensors
	if idx, err := src.ReadFile("model.safetensors.index.json"); err == nil {
		var ix struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(idx, &ix); err != nil {
			return fmt.Errorf("qwen35: model.safetensors.index.json: %w", err)
		}
		for name, f := range ix.WeightMap {
			shards[f] = append(shards[f], name)
		}
	} else if errors.Is(err, fs.ErrNotExist) {
		shards["model.safetensors"] = nil // its header lists them
	} else {
		return err
	}
	headers := map[string]*remoteHeader{}
	var fileNames []string
	for f := range shards {
		fileNames = append(fileNames, f)
	}
	sort.Strings(fileNames)
	for _, f := range fileNames {
		h, err := readHeader(src, f)
		if err != nil {
			return err
		}
		headers[f] = h
	}
	// the text model's tensors by the file they go to
	layerRe := regexp.MustCompile(`^` + regexp.QuoteMeta(c.prefix()) + `layers\.(\d+)\.`)
	groups := map[string][]tensorAt{}
	for _, f := range fileNames {
		for name, t := range headers[f].tensors {
			if !strings.HasPrefix(name, c.prefix()) && name != "lm_head.weight" || strings.HasPrefix(name, "model.visual.") || strings.HasPrefix(name, "mtp.") {
				continue
			}
			g := "model-rest.safetensors"
			if m := layerRe.FindStringSubmatch(name); m != nil {
				var l int
				fmt.Sscan(m[1], &l)
				g = fmt.Sprintf("model-layer-%02d.safetensors", l)
			}
			groups[g] = append(groups[g], tensorAt{name: name, file: f, header: headers[f], t: t})
		}
	}
	if len(groups) == 0 {
		return fmt.Errorf("qwen35: no tensors of the text model under %s", c.prefix())
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	var outNames []string
	for g := range groups {
		outNames = append(outNames, g)
	}
	sort.Strings(outNames)
	weightMap := map[string]string{}
	var total int64
	start := time.Now()
	for i, g := range outNames {
		ts := groups[g]
		sort.Slice(ts, func(a, b int) bool { return ts[a].name < ts[b].name })
		path := filepath.Join(out, g)
		if f, err := safetensors.Open(path); err == nil {
			for _, n := range f.Names() {
				weightMap[n] = g
			}
			f.Close()
			if st, err := os.Stat(path); err == nil {
				total += st.Size()
			}
			fmt.Fprintf(log, "%s: there already (%d of %d)\n", g, i+1, len(outNames))
			continue
		}
		t0 := time.Now()
		outs, err := c.deriveGroup(src, ts)
		if err != nil {
			return fmt.Errorf("qwen35: %s: %w", g, err)
		}
		var size int64
		for _, o := range outs {
			weightMap[o.Name] = g
			size += o.Size
		}
		if err := safetensors.Write(path, outs, map[string]string{"format": "pt"}); err != nil {
			return err
		}
		total += size
		fmt.Fprintf(log, "%s: %d tensors, %.2f GB in %.0f s (%d of %d, %.0f min so far)\n", g, len(outs), float64(size)/1e9, time.Since(t0).Seconds(), i+1, len(outNames), time.Since(start).Minutes())
	}
	idx, err := json.MarshalIndent(map[string]any{"metadata": map[string]any{"total_size": total}, "weight_map": weightMap}, "", "  ")
	if err != nil {
		return err
	}
	if err := writeFile(filepath.Join(out, "model.safetensors.index.json"), idx); err != nil {
		return err
	}
	for _, f := range DeriveFiles {
		b, err := src.ReadFile(f)
		if errors.Is(err, fs.ErrNotExist) {
			continue
		}
		if err != nil {
			return err
		}
		if err := writeFile(filepath.Join(out, f), b); err != nil {
			return err
		}
	}
	var cfg map[string]any
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return err
	}
	cfg[DeriveKey] = map[string]any{"from": from, "experts": "mxfp4 (gate and up interleaved, the gate the even rows; bonsaiai-dev mxfp4)", "dropped": []string{"model.visual.", "mtp."}}
	cb, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	// config.json last: a directory with it is whole
	if err := writeFile(filepath.Join(out, "config.json"), cb); err != nil {
		return err
	}
	fmt.Fprintf(log, "%s: %.2f GB in %d files, %.0f min\n", out, float64(total)/1e9, len(outNames), time.Since(start).Minutes())
	return nil
}

func writeFile(path string, b []byte) error {
	if err := os.WriteFile(path+".tmp", b, 0o644); err != nil {
		return err
	}
	return os.Rename(path+".tmp", path)
}

// IsDerived reports whether config.json (raw) is of a checkpoint DeriveMXFP4 wrote.
func IsDerived(raw []byte) bool {
	var m map[string]json.RawMessage
	if json.Unmarshal(raw, &m) != nil {
		return false
	}
	_, ok := m[DeriveKey]
	return ok
}

type remoteHeader struct {
	base    int64 // where the data begins
	tensors map[string]remoteTensor
}

type remoteTensor struct {
	DType   string  `json:"dtype"`
	Shape   []int   `json:"shape"`
	Offsets []int64 `json:"data_offsets"`
}

type tensorAt struct {
	name   string
	file   string
	header *remoteHeader
	t      remoteTensor
}

func readHeader(src Source, file string) (*remoteHeader, error) {
	b, err := src.ReadRange(file, 0, 8)
	if err != nil {
		return nil, err
	}
	n := int64(binary.LittleEndian.Uint64(b))
	if n <= 0 || n > 100<<20 {
		return nil, fmt.Errorf("qwen35: %s: a header of %d bytes", file, n)
	}
	hb, err := src.ReadRange(file, 8, n)
	if err != nil {
		return nil, err
	}
	var all map[string]json.RawMessage
	if err := json.Unmarshal(bytes.TrimRight(hb, " \x00"), &all); err != nil {
		return nil, fmt.Errorf("qwen35: %s's header: %w", file, err)
	}
	h := &remoteHeader{base: 8 + n, tensors: map[string]remoteTensor{}}
	for name, v := range all {
		if name == "__metadata__" {
			continue
		}
		var t remoteTensor
		if err := json.Unmarshal(v, &t); err != nil || len(t.Offsets) != 2 {
			return nil, fmt.Errorf("qwen35: %s: tensor %s: %v", file, name, err)
		}
		h.tensors[name] = t
	}
	return h, nil
}

// read reads bytes [off, off+n) of a tensor's data.
func (t tensorAt) read(src Source, off, n int64) ([]byte, error) {
	return src.ReadRange(t.file, t.header.base+t.t.Offsets[0]+off, n)
}

// deriveGroup is the tensors of one output file: the experts in MXFP4, the others as they are.
func (c *MoE) deriveGroup(src Source, ts []tensorAt) ([]safetensors.Out, error) {
	var outs []safetensors.Out
	add := func(name, dtype string, shape []int, b []byte) {
		outs = append(outs, safetensors.Out{Name: name, DType: dtype, Shape: shape, Size: int64(len(b)), Data: bytes.NewReader(b)})
	}
	E, H, I := c.NumExperts, c.HiddenSize, c.MoEIntermediate
	for _, t := range ts {
		switch {
		case strings.HasSuffix(t.name, ".mlp.experts.gate_up_proj"):
			if !slices.Equal(t.t.Shape, []int{E, 2 * I, H}) {
				return nil, fmt.Errorf("%s is %v, want [%d %d %d]", t.name, t.t.Shape, E, 2*I, H)
			}
			w, e, err := c.encodeExperts(src, t, 2*I, H, true)
			if err != nil {
				return nil, err
			}
			add(t.name+"_blocks", "U8", []int{E, 2 * I, H / 32, 16}, w)
			add(t.name+"_scales", "U8", []int{E, 2 * I, H / 32}, e)
		case strings.HasSuffix(t.name, ".mlp.experts.down_proj"):
			if !slices.Equal(t.t.Shape, []int{E, H, I}) {
				return nil, fmt.Errorf("%s is %v, want [%d %d %d]", t.name, t.t.Shape, E, H, I)
			}
			w, e, err := c.encodeExperts(src, t, H, I, false)
			if err != nil {
				return nil, err
			}
			add(t.name+"_blocks", "U8", []int{E, H, I / 32, 16}, w)
			add(t.name+"_scales", "U8", []int{E, H, I / 32}, e)
		default:
			b, err := t.read(src, 0, t.t.Offsets[1]-t.t.Offsets[0])
			if err != nil {
				return nil, err
			}
			add(t.name, t.t.DType, t.t.Shape, b)
		}
	}
	return outs, nil
}

// encodeExperts reads an experts tensor (experts, rows, cols) in pieces of a few experts, several
// at a time, and encodes each row in MXFP4; with gateUp the rows are reordered so that the gate's
// row j (row j) and the up projection's (row rows/2 + j) are rows 2j and 2j+1.
func (c *MoE) encodeExperts(src Source, t tensorAt, rows, cols int, gateUp bool) (w, e []byte, err error) {
	E := c.NumExperts
	es := dtypeSize(t.t.DType)
	if es == 0 {
		return nil, nil, fmt.Errorf("%s: dtype %s", t.name, t.t.DType)
	}
	per := int64(rows) * int64(cols) * int64(es) // an expert's bytes
	const piece = 8                              // experts a read
	w, e = make([]byte, E*rows*cols/2), make([]byte, E*rows*cols/32)
	type job struct{ first, n int }
	jobs := make(chan job)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error
	workers := 4 // reads at a time: a range of Hugging Face's comes at about 20 MB/s
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				b, err := t.read(src, int64(j.first)*per, int64(j.n)*per)
				if err != nil {
					mu.Lock()
					firstErr = cmpErr(firstErr, err)
					mu.Unlock()
					continue
				}
				encodePiece(b, t.t.DType, j.first, j.n, rows, cols, gateUp, w, e)
			}
		}()
	}
	for first := 0; first < E; first += piece {
		jobs <- job{first, min(piece, E-first)}
	}
	close(jobs)
	wg.Wait()
	return w, e, firstErr
}

func cmpErr(a, b error) error {
	if a != nil {
		return a
	}
	return b
}

// encodePiece encodes experts first.. of n from their raw values b over the CPU's cores.
func encodePiece(b []byte, dtype string, first, n, rows, cols int, gateUp bool, w, e []byte) {
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())
	for k := range n {
		for r := range rows {
			sem <- struct{}{}
			wg.Add(1)
			go func(k, r int) {
				defer func() { <-sem; wg.Done() }()
				row := make([]float32, cols)
				decodeRow(row, b, dtype, (k*rows+r)*cols)
				dst := r
				if gateUp {
					if half := rows / 2; r < half {
						dst = 2 * r
					} else {
						dst = 2*(r-half) + 1
					}
				}
				o := ((first+k)*rows + dst) * cols
				mxfp4.Encode(w[o/2:(o+cols)/2], e[o/32:(o+cols)/32], row)
			}(k, r)
		}
	}
	wg.Wait()
}

func dtypeSize(dt string) int {
	switch dt {
	case "BF16", "F16":
		return 2
	case "F32":
		return 4
	}
	return 0
}

// decodeRow reads len(dst) values from element at of raw.
func decodeRow(dst []float32, raw []byte, dtype string, at int) {
	switch dtype {
	case "BF16":
		for i := range dst {
			dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[2*(at+i):])) << 16)
		}
	case "F16":
		vecops.F16ToF32(dst, raw[2*at:2*(at+len(dst))])
	case "F32":
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*(at+i):]))
		}
	}
}
