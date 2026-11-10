// Package ocr reads the text in a picture with PaddleOCR's PP-OCRv6 (Apache-2.0, the ONNX files
// PaddlePaddle publishes): a detector (DB, differentiable binarization) finds the lines' boxes,
// and a recognizer (SVTR with a CTC head, 50 languages: Chinese, Japanese, English, ...) reads
// each line.
//
// Both networks run on the NPU at fixed sizes: the detector at the size of Sizes the picture is
// scaled down least to fit (the picture at the top left, zeros past it), the recognizer on a
// few lines at a time, 48 pixels high and as wide as the narrowest of Widths that holds them. A size's program is converted the first time a picture takes it. The host
// does what depends on the picture: the binarized map's connected regions, their rotated boxes
// and scores, the boxes' growth (unclip), the lines' crops, and the CTC decoding.
package ocr

import (
	"cmp"
	"context"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/goccy/go-yaml"
	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/onnx"
	"github.com/shibukawa/hwnpu/internal/vecops"
	"github.com/shibukawa/hwnpu/internal/vision/detect"
)

// Revision counts the changes to the programs that call for converting them again.
const Revision = 1

// Height is the recognizer's line height.
const Height = 48

// The files of each repository.
const (
	ModelFile  = "inference.onnx"
	ConfigFile = "inference.yml"
)

// Sizes are the detector's sizes (W x H, multiples of 32) by default.
var Sizes = [][2]int{{960, 960}, {1280, 736}, {736, 1280}}

// Widths are the recognizer's widths by default: its 320 (a ratio of 6.7) and wider for longer
// lines, each at most half again the one before; a line wider still is narrowed to the last.
// The steps matter: the recognizer attends over the whole width, and what it reads changes
// with the zeros after a line ("SUB TOTAL" 230 pixels wide reads as SUBTOTAL padded to 640).
// PaddleOCR pads a batch to its widest line.
var Widths = []int{320, 480, 640, 960, 1280, 1920, 2560}

// Batch is the most lines the recognizer reads in one call by default, Columns the most
// columns of them (fewer lines a call for the wider: 4 at 1280, its output 24 MB).
const (
	Batch   = 8
	Columns = 5120
)

// Source is a repository and its files' sha256 by name (pinned when given).
type Source struct {
	Repo   string
	SHA256 map[string]string
}

// Setup says where the models are and what to download.
type Setup struct {
	// Dir holds the repositories' files and the programs, a directory for each.
	Dir      string
	Det, Rec Source
	Backend  string
	Log      io.Writer
}

// Line is a line of text found: its box (four corners clockwise from the top left, in the
// picture's pixels), its text and the recognizer's mean confidence.
type Line struct {
	Box   [4][2]float32 `json:"box"`
	Text  string        `json:"text"`
	Score float32       `json:"score"`
}

// Reader holds the models.
type Reader struct {
	// Thresh is the probability that marks a pixel as text, BoxThresh the least mean
	// probability of a box, Unclip how much a box grows (its area times Unclip over its
	// perimeter on each side), MinScore the least recognition score of a line kept (PP-OCRv6's
	// inference.yml: 0.2, 0.45, 1.4; PaddleOCR's pipeline drops lines under 0.0 by default).
	Thresh, BoxThresh, Unclip, MinScore float32
	// Sizes are the detector's sizes, Widths the recognizer's, Batch and Columns its most lines
	// and columns a call.
	Sizes          [][2]int
	Widths         []int
	Batch, Columns int
	// Size, when set, is the size every picture is detected at.
	Size [2]int
	// Times are the last picture's: the detector (resize, program, boxes), the crops, the
	// recognizer's programs and the decoding.
	Times struct{ Resize, Detect, Boxes, Crop, Recognize, Decode time.Duration }
	// Calls is how many recognizer calls the last picture took.
	Calls int
	// Logf, when set, is told what is converted.
	Logf func(format string, args ...any)

	s        Setup
	dev      *hwnpu.Device
	chars    []string // the CTC classes: blank, the dictionary, a space
	detModel string
	recModel string
	mu       sync.Mutex
	det      map[[2]int]runner.Program
	rec      map[int]runner.Program
	onNPU    bool
}

// Open downloads the repositories' files into s.Dir unless they are there and reads the
// recognizer's dictionary. The programs are converted and loaded onto dev (nil: the CPU
// interpreter) when a picture first takes their size.
func Open(ctx context.Context, dev *hwnpu.Device, s Setup) (*Reader, error) {
	log := cmp.Or[io.Writer](s.Log, io.Discard)
	s.Log = log
	r := &Reader{Thresh: 0.2, BoxThresh: 0.45, Unclip: 1.4, Sizes: Sizes, Widths: Widths, Batch: Batch, Columns: Columns,
		s: s, dev: dev, det: map[[2]int]runner.Program{}, rec: map[int]runner.Program{}, onNPU: dev != nil}
	var err error
	if r.detModel, err = fetch(ctx, s.Dir, s.Det, ModelFile, log); err != nil {
		return nil, err
	}
	detCfg, err := fetch(ctx, s.Dir, s.Det, ConfigFile, log)
	if err != nil {
		return nil, err
	}
	if r.recModel, err = fetch(ctx, s.Dir, s.Rec, ModelFile, log); err != nil {
		return nil, err
	}
	recCfg, err := fetch(ctx, s.Dir, s.Rec, ConfigFile, log)
	if err != nil {
		return nil, err
	}
	var dc struct {
		PostProcess struct {
			Thresh      *float32 `yaml:"thresh"`
			BoxThresh   *float32 `yaml:"box_thresh"`
			UnclipRatio *float32 `yaml:"unclip_ratio"`
		} `yaml:"PostProcess"`
	}
	if err := readYAML(detCfg, &dc); err != nil {
		return nil, err
	}
	if p := dc.PostProcess; p.Thresh != nil && p.BoxThresh != nil && p.UnclipRatio != nil {
		r.Thresh, r.BoxThresh, r.Unclip = *p.Thresh, *p.BoxThresh, *p.UnclipRatio
	}
	var rc struct {
		PostProcess struct {
			Name  string   `yaml:"name"`
			Chars []string `yaml:"character_dict"`
		} `yaml:"PostProcess"`
	}
	if err := readYAML(recCfg, &rc); err != nil {
		return nil, err
	}
	if rc.PostProcess.Name != "CTCLabelDecode" || len(rc.PostProcess.Chars) == 0 {
		return nil, fmt.Errorf("ocr: %s has no CTC dictionary", recCfg)
	}
	r.chars = append(append([]string{""}, rc.PostProcess.Chars...), " ")
	return r, nil
}

func readYAML(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := yaml.Unmarshal(b, v); err != nil {
		return fmt.Errorf("ocr: %s: %w", path, err)
	}
	return nil
}

// fetch downloads a repository's file into dir/NAME (the repository's name) and checks its
// sha256 when pinned.
func fetch(ctx context.Context, dir string, s Source, file string, log io.Writer) (string, error) {
	path := filepath.Join(repoDir(dir, s), file)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	if err := chat.Fetch(ctx, s.Repo, file, path, log); err != nil {
		return "", err
	}
	if want := s.SHA256[file]; want != "" {
		if err := detect.CheckSHA(path, want); err != nil {
			return "", fmt.Errorf("ocr: %w", err)
		}
	}
	return path, nil
}

func repoDir(dir string, s Source) string {
	repo := s.Repo
	if i := slices.Index([]byte(repo), '@'); i >= 0 {
		repo = repo[:i]
	}
	return filepath.Join(dir, filepath.Base(repo))
}

// OnNPU reports whether every program loaded so far runs all on the NPU.
func (r *Reader) OnNPU() bool { return r.onNPU }

// Close releases the programs.
func (r *Reader) Close() error {
	var err error
	for _, p := range r.det {
		err = cmp.Or(err, p.Close())
	}
	for _, p := range r.rec {
		err = cmp.Or(err, p.Close())
	}
	clear(r.det)
	clear(r.rec)
	return err
}

// program converts (unless converted) and loads the program of model at the input shape, in
// the repository's directory as NAME-rREV-fp16-SHAPE.
func (r *Reader) program(ctx context.Context, src Source, model, name string, shape []int, describe string) (runner.Program, error) {
	ext := ".mlpackage"
	if r.s.Backend == llm.BackendIntel {
		ext = ".xml"
	}
	tag := fmt.Sprintf("%dx%d", shape[3], shape[2])
	if shape[0] != 1 {
		tag = fmt.Sprintf("%dx%dx%d", shape[0], shape[3], shape[2])
	}
	path := filepath.Join(repoDir(r.s.Dir, src), fmt.Sprintf("%s-r%d-fp16-%s%s", name, Revision, tag, ext))
	if _, err := os.Stat(path + ".done"); err != nil {
		g, err := onnx.Load(model)
		if err != nil {
			return nil, err
		}
		p := NewProgram(g, shape)
		bar := console.NewBar(r.s.Log, "converting "+filepath.Base(path))
		os.RemoveAll(path)
		_, err = llm.WritePart(path, p, nil, llm.Options{Format: "fp16", Backend: r.s.Backend}, func(pkg *mlpackage.Package) {
			pkg.Description = fmt.Sprintf("PP-OCR %s from %s, %s, fp16 (hwnpu)", describe, src.Repo, tag)
			pkg.Metadata[llm.MetaArch] = "ppocr-" + name
		})
		if err != nil {
			bar.Done("failed")
			return nil, err
		}
		bar.Done(console.Duration(bar.Elapsed()))
		if err := os.WriteFile(path+".done", nil, 0o644); err != nil {
			return nil, err
		}
	}
	t0 := time.Now()
	p, err := runner.Open(ctx, r.dev, path)
	if err != nil {
		return nil, err
	}
	r.onNPU = r.onNPU && p.OnNPU()
	if r.Logf != nil {
		where := "on the CPU interpreter"
		if r.dev != nil {
			where = "all on the NPU"
			if !p.OnNPU() {
				where = "NOT all on the NPU"
			}
		}
		r.Logf("loaded %s in %.1f s, %s", filepath.Base(path), time.Since(t0).Seconds(), where)
	}
	return p, nil
}

func (r *Reader) detector(ctx context.Context, W, H int) (runner.Program, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if p := r.det[[2]int{W, H}]; p != nil {
		return p, nil
	}
	p, err := r.program(ctx, r.s.Det, r.detModel, "det", []int{1, 3, H, W}, "text detector")
	if err != nil {
		return nil, err
	}
	r.det[[2]int{W, H}] = p
	return p, nil
}

func (r *Reader) recognizer(ctx context.Context, W int) (runner.Program, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if p := r.rec[W]; p != nil {
		return p, nil
	}
	b := r.batch(W)
	p, err := r.program(ctx, r.s.Rec, r.recModel, "rec", []int{b, 3, Height, W}, "text recognizer")
	if err != nil {
		return nil, err
	}
	// (batch, steps, classes), or (batch, steps, parts, classes / parts) for classes past the
	// Neural Engine's dimension: the same order
	if out := p.Output(); len(out) < 3 || out[0] != b || onnx.Len(out[2:]) != len(r.chars) {
		p.Close()
		return nil, fmt.Errorf("ocr: the recognizer gives %v, not (%d, steps, %d) for its dictionary", out, b, len(r.chars))
	}
	r.rec[W] = p
	return p, nil
}

// NewProgram is the program of a graph (the detector or the recognizer) at an input shape.
// A graph that ends in a softmax (the recognizer's, over its 18710 classes: past the Neural
// Engine's 16384 a dimension, Core ML runs it on the CPU) gives the logits before it: Decode
// takes the most likely class and its probability from them.
func NewProgram(g *onnx.Graph, shape []int) *onnx.Program {
	prod := map[string]*onnx.Node{}
	for _, n := range g.Nodes {
		prod[n.Out[0]] = n
	}
	out := g.Outputs[0].Name
	if n := prod[out]; n != nil && n.Op == "Softmax" {
		out = n.In[0]
		// and before the Identities Paddle's export puts between
		for n := prod[out]; n != nil && n.Op == "Identity"; n = prod[out] {
			out = n.In[0]
		}
	}
	return &onnx.Program{G: g, Inputs: []onnx.ValueInfo{{Name: g.Inputs[0].Name, Shape: shape}}, Outputs: []string{out}, Axis: -1, ExpSigmoid: true}
}

// batch is how many lines the recognizer reads a call at width W.
func (r *Reader) batch(W int) int { return max(1, min(r.Batch, r.Columns/W)) }

// width is the narrowest of the recognizer's widths that holds a line of cw columns (the widest
// for a wider one).
func (r *Reader) width(cw int) int {
	for _, w := range r.Widths {
		if w >= cw {
			return w
		}
	}
	return r.Widths[len(r.Widths)-1]
}

// Choose is the detector's size for a w x h picture: the one it is scaled down least to fit,
// the smallest of those.
func (r *Reader) Choose(w, h int) (int, int) {
	if r.Size != [2]int{} {
		return r.Size[0], r.Size[1]
	}
	best, bw, bh := -1.0, 0, 0
	for _, s := range r.Sizes {
		sc := min(1, float64(s[0])/float64(w), float64(s[1])/float64(h))
		if sc > best+1e-9 || (math.Abs(sc-best) <= 1e-9 && s[0]*s[1] < bw*bh) {
			best, bw, bh = sc, s[0], s[1]
		}
	}
	return bw, bh
}

// DetMean and DetStd normalize the detector's BGR pixels (0 to 1): ImageNet's RGB values,
// applied to B, G, R in this order as PaddleOCR does.
var (
	DetMean = [3]float32{0.485, 0.456, 0.406}
	DetStd  = [3]float32{0.229, 0.224, 0.225}
)

// DetInput is the detector's input for a picture's RGB pixels (w x h) at W x H: scaled by
// scale to fit (not enlarged), at the top left, BGR normalized by DetMean and DetStd, zeros past
// it, channels first.
func DetInput(rgb []uint8, w, h, W, H int) (x []float32, scale float64, nw, nh int) {
	scale = min(1, float64(W)/float64(w), float64(H)/float64(h))
	nw, nh = max(1, min(W, int(float64(w)*scale+0.5))), max(1, min(H, int(float64(h)*scale+0.5)))
	px := lfm3vl.ResizeRGB(rgb, w, h, nw, nh, "bilinear")
	x = make([]float32, 3*W*H)
	for c := range 3 { // c: B, G, R
		plane := x[c*W*H:]
		src := 2 - c
		k, b := 1/(255*DetStd[c]), -DetMean[c]/DetStd[c]
		for y := range nh {
			row, in := plane[y*W:y*W+nw], px[y*nw*3:]
			for i := range row {
				row[i] = float32(in[3*i+src])*k + b
			}
		}
	}
	return x, scale, nw, nh
}

// Read finds and reads the lines of text in a picture, in reading order (top to bottom, left to
// right).
func (r *Reader) Read(ctx context.Context, img image.Image) ([]Line, error) {
	t0 := time.Now()
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	rgb := lfm3vl.RGB(img)
	W, H := r.Choose(w, h)
	x, scale, nw, nh := DetInput(rgb, w, h, W, H)
	t1 := time.Now()
	det, err := r.detector(ctx, W, H)
	if err != nil {
		return nil, err
	}
	t1l := time.Now()
	prob, err := det.Run(ctx, map[string][]float32{"x": x})
	if err != nil {
		return nil, err
	}
	t2 := time.Now()
	boxes := r.Boxes(prob, W, H, nw, nh)
	for i := range boxes {
		for j := range 4 {
			boxes[i][j][0] = float32(math.Max(0, math.Min(float64(w), float64(boxes[i][j][0])/scale)))
			boxes[i][j][1] = float32(math.Max(0, math.Min(float64(h), float64(boxes[i][j][1])/scale)))
		}
	}
	SortBoxes(boxes)
	t3 := time.Now()
	lines, err := r.Recognize(ctx, rgb, w, h, boxes)
	if err != nil {
		return nil, err
	}
	r.Times.Resize, r.Times.Detect, r.Times.Boxes = t1.Sub(t0), t2.Sub(t1l), t3.Sub(t2)
	var kept []Line
	for _, l := range lines {
		if l.Text != "" && l.Score >= r.MinScore {
			kept = append(kept, l)
		}
	}
	return kept, nil
}

// Recognize reads the lines in the boxes of a picture's RGB pixels (w x h).
func (r *Reader) Recognize(ctx context.Context, rgb []uint8, w, h int, boxes [][4][2]float32) ([]Line, error) {
	t0 := time.Now()
	type crop struct {
		i   int
		px  []uint8 // RGB, Height rows
		len int     // its width
	}
	crops := make([]crop, len(boxes))
	parallel(len(boxes), func(i int) {
		px, cw := Crop(rgb, w, h, boxes[i], slices.Max(r.Widths))
		crops[i] = crop{i, px, cw}
	})
	r.Times.Crop = time.Since(t0)
	// lines by width, so that a batch takes the narrowest program it can (PaddleOCR sorts by
	// the ratio too)
	order := make([]int, len(crops))
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(a, b int) int { return cmp.Compare(crops[a].len, crops[b].len) })
	lines := make([]Line, len(boxes))
	var tRec, tDec time.Duration
	var wg sync.WaitGroup
	var decMu sync.Mutex
	r.Calls = 0
	for at := 0; at < len(order); {
		// the next lines of one width, as many as a call takes
		W := r.width(crops[order[at]].len)
		end := at + 1
		for end < len(order) && end-at < r.batch(W) && r.width(crops[order[end]].len) == W {
			end++
		}
		batch := order[at:end]
		at = end
		p, err := r.recognizer(ctx, W)
		if err != nil {
			wg.Wait()
			return nil, err
		}
		x := make([]float32, r.batch(W)*3*Height*W)
		for k, i := range batch {
			RecInput(x[k*3*Height*W:(k+1)*3*Height*W], crops[i].px, crops[i].len, W)
		}
		s := time.Now()
		y, err := runner.F16(ctx, p, map[string][]float32{"x": x})
		if err != nil {
			wg.Wait()
			return nil, err
		}
		r.Calls++
		tRec += time.Since(s)
		// decode while the next call runs: the output is the program's until its next run
		y = slices.Clone(y)
		out := p.Output()
		steps := out[1]
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := time.Now()
			for k, i := range batch {
				// the steps the line covers (the network takes W / steps columns a step), and
				// one more: past them is the zeros' padding
				n := min(steps, (crops[i].len*steps+W-1)/W+1)
				text, score := r.Decode(y[2*k*steps*len(r.chars):][:2*n*len(r.chars)], n)
				lines[i] = Line{Box: boxes[i], Text: text, Score: score}
			}
			decMu.Lock()
			tDec += time.Since(s)
			decMu.Unlock()
		}()
	}
	wg.Wait()
	r.Times.Recognize, r.Times.Decode = tRec, tDec
	return lines, nil
}

// RecInput writes a line's crop (RGB, Height rows of cw pixels) into the recognizer's input x
// (3 x Height x W): BGR as (v / 255 - 0.5) / 0.5, zeros past it.
func RecInput(x []float32, px []uint8, cw, W int) {
	for c := range 3 {
		plane := x[c*Height*W:]
		src := 2 - c
		for y := range Height {
			row, in := plane[y*W:y*W+min(cw, W)], px[y*cw*3:]
			for i := range row {
				row[i] = float32(in[3*i+src])/127.5 - 1
			}
		}
	}
}

// Decode is CTC's greedy decoding of a line's logits (steps x classes, fp16 little-endian): the
// most likely class at each step, repeats merged and blanks dropped, and the mean probability
// (the softmax's) of the characters kept.
func (r *Reader) Decode(logits []byte, steps int) (string, float32) {
	n := len(r.chars)
	all := f16s(logits)
	// each step's most likely class, the steps in parts on the CPU's cores
	bests := make([]int, steps)
	const part = 16
	parallel((steps+part-1)/part, func(j int) {
		for t := j * part; t < min(steps, (j+1)*part); t++ {
			best, bk := 0, uint16(0)
			for c, h := range all[t*n : (t+1)*n] {
				if k := sortKey(h); k > bk {
					best, bk = c, k
				}
			}
			bests[t] = best
		}
	})
	var text []byte
	var sum float64
	kept, prev := 0, 0
	for t, best := range bests {
		row := all[t*n : (t+1)*n]
		if best != 0 && best != prev {
			text = append(text, r.chars[best]...)
			// the softmax's largest value, 1 / sum(exp(v - max)), over the values within 16 of
			// the largest (the others add under 2e-3 over 18710 classes)
			bv := vecops.F16Value(row[best])
			least := sortKey(vecops.F16Bits(bv - 16))
			var z float64
			for _, h := range row {
				if sortKey(h) > least {
					z += math.Exp(float64(vecops.F16Value(h) - bv))
				}
			}
			sum += 1 / z
			kept++
		}
		prev = best
	}
	if kept == 0 {
		return "", 0
	}
	return string(text), float32(sum / float64(kept))
}

// sortKey is fp16's bits made to sort as unsigned integers do: a negative value's bits flipped,
// the others' sign bit set (no NaNs come out of the network).
func sortKey(h uint16) uint16 {
	return h ^ (uint16(int16(h)>>15) | 0x8000)
}

// f16s are fp16 values' little-endian bytes as uint16s (the machines hwnpu runs on are
// little-endian).
func f16s(b []byte) []uint16 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*uint16)(unsafe.Pointer(&b[0])), len(b)/2)
}

// parallel calls f(0), ..., f(n-1) on the CPU's cores.
func parallel(n int, f func(i int)) {
	workers := min(n, runtime.GOMAXPROCS(0))
	var next atomic.Int64
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1) - 1)
				if i >= n {
					return
				}
				f(i)
			}
		}()
	}
	wg.Wait()
}
