package vision

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/vision/ocr"
)

// OCR is the use of a model that reads the text in pictures (PP-OCR: a text detector and a
// recognizer of 50 languages) for an app: a model, a family or a use of the definitions ("ocr",
// "ppocr:small").
func OCR(name string) ai.Use[*TextReader] {
	return ai.Define[*TextReader](ocrRole, name)
}

// ocrSetup is where the detector and the recognizer of model m come from: the recognizer's
// repository, and the detector's in models.yaml's with (detector), its files' sha256 under
// detector/.
func ocrSetup(m chat.Model, log io.Writer) (ocr.Setup, error) {
	det := m.With["detector"]
	if det == "" {
		return ocr.Setup{}, fmt.Errorf("%s: models.yaml gives no detector (with: {detector: hf:OWNER/NAME})", shown(m.Name))
	}
	detSHA, recSHA := map[string]string{}, map[string]string{}
	for f, d := range m.SHA256 {
		if rest, ok := strings.CutPrefix(f, "detector/"); ok {
			detSHA[rest] = d
		} else {
			recSHA[f] = d
		}
	}
	return ocr.Setup{Dir: chat.ModelsDir(), Det: ocr.Source{Repo: det, SHA256: detSHA}, Rec: ocr.Source{Repo: m.Source(), SHA256: recSHA}, Log: log}, nil
}

// The definition of an OCR model may give the settings threads and cpu, and the device
// (common.go). On an NPU each size of the detector's and the recognizer's programs is converted
// the first time a picture (a line) takes it.
var ocrRole = &airole.Role{
	Name:  "ocr",
	Kinds: []chat.Kind{chat.KindOCR},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		m := e.Model
		s, err := ocrSetup(m, e.Log)
		if err != nil {
			return f, err
		}
		ort, builtin := imagedev.CPURoute(e.Ctx, m, imagedev.DeviceFlag(m), interpreter(m), threads(m), quiet)
		if !ort && !interpreter(m) && !builtin && e.NPU.Format == "" {
			return f, errors.New("this machine's NPU runs neither Core AI programs nor OpenVINO IR (--device cpu runs PP-OCR on the CPU)")
		}
		// the repositories' files (opening the reader downloads them; its programs are made later)
		s.CPU = true
		r, err := ocr.Open(e.Ctx, nil, s)
		if err != nil {
			return f, err
		}
		r.Close()
		f.Set("models", s.Dir)
		return f, nil
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		m := e.Def
		s, err := ocrSetup(m, e.Log)
		if err != nil {
			return nil, err
		}
		n, interp := threads(m), interpreter(m)
		ort, builtin := imagedev.CPURoute(ctx, named(m), imagedev.DeviceFlag(m), interp, n, logf(e.Log))
		cpu := interp || builtin
		var (
			dev      *hwnpu.Device
			closeDev = func() {}
		)
		if !ort {
			if dev, s.Backend, closeDev, err = imagedev.OpenProgramDevice(ctx, cpu, e.Cache, e.NPU.Device, "PP-OCR"); err != nil {
				return nil, err
			}
		}
		s.Dir, s.CPU, s.Threads = f.Path("models"), ort, n
		r, err := ocr.Open(ctx, dev, s)
		if err != nil {
			closeDev()
			return nil, err
		}
		r.Logf = logf(e.Log)
		t := &TextReader{r: r, closeDev: closeDev, name: e.Model, device: "cpu"}
		switch {
		case ort:
			t.where = fmt.Sprintf("on the CPU (ONNX Runtime, %d threads)", n)
		case interp:
			t.where = imagedev.Placement(true, false)
		case builtin:
			t.where = "on the CPU (hwnpu's own executor)"
		default:
			t.device, t.where = "npu", "on the NPU"
		}
		return t, nil
	},
}

// TextLine is a line of text found in a picture: its box (four corners clockwise from the top
// left, in the picture's pixels), what it says, and the recognizer's mean confidence (0 to 1).
type TextLine struct {
	Box   [4]Point
	Text  string
	Score float64
}

// Text is the text read in a picture: its lines in reading order (top to bottom, left to
// right), and what reading them took.
type Text struct {
	Lines []TextLine
	// Size is the detector's input size the picture ran at (W x H).
	Size [2]int
	// Took is how long it all took: Resize, Detect and Boxes the detector's resizing, network
	// and boxes, Crop the lines' crops, Recognize the recognizer's network in Calls calls,
	// Decode its text decoded.
	Took, Resize, Detect, Boxes, Crop, Recognize, Decode time.Duration
	Calls                                                int
}

// OCROptions are how the text of a picture is read.
type OCROptions struct {
	// MinScore is the least recognition score of a line kept (0 keeps every line read).
	MinScore float32
	// Size is the detector's input size, multiples of 32 (default: of 960 x 960, 1280 x 736 and
	// 736 x 1280 the one the picture is scaled down least to fit). Larger finds smaller text.
	Size [2]int
}

// TextReader reads the text in pictures. It runs one picture at a time; others wait.
type TextReader struct {
	mu            sync.Mutex
	r             *ocr.Reader
	closeDev      func()
	name          string
	device, where string
}

// Model is the name of the model in the definitions.
func (t *TextReader) Model() string { return t.name }

// Device is where the networks run: npu or cpu.
func (t *TextReader) Device() string { return t.device }

// Where says where the networks run, for a log.
func (t *TextReader) Where() string { return t.where }

// Read reads the lines of text in a picture.
func (t *TextReader) Read(ctx context.Context, img image.Image, o OCROptions) (*Text, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if s := o.Size; s != [2]int{} && (s[0] <= 0 || s[1] <= 0 || s[0]%32 != 0 || s[1]%32 != 0) {
		return nil, fmt.Errorf("vision: the detector's size %dx%d is not of multiples of 32", s[0], s[1])
	}
	t.r.MinScore, t.r.Size = o.MinScore, o.Size
	t0 := time.Now()
	lines, err := t.r.Read(ctx, img)
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	W, H := t.r.Choose(b.Dx(), b.Dy())
	tm := t.r.Times
	out := &Text{Lines: []TextLine{}, Size: [2]int{W, H}, Took: time.Since(t0), Resize: tm.Resize, Detect: tm.Detect, Boxes: tm.Boxes,
		Crop: tm.Crop, Recognize: tm.Recognize, Decode: tm.Decode, Calls: t.r.Calls}
	x0, y0 := float64(b.Min.X), float64(b.Min.Y)
	for _, l := range lines {
		tl := TextLine{Text: l.Text, Score: float64(l.Score)}
		for i, c := range l.Box {
			tl.Box[i] = Point{float64(c[0]) + x0, float64(c[1]) + y0}
		}
		out.Lines = append(out.Lines, tl)
	}
	return out, nil
}

// Close unloads the models. An app leaves it to the Runtime (its Close, or the use's Release).
func (t *TextReader) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	err := t.r.Close()
	t.closeDev()
	return err
}
