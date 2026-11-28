package vision

import (
	"context"
	"image"
	"io"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
)

// OCR is the use of a model that reads the text in pictures (PP-OCR: a text detector and a
// recognizer of 50 languages) for an app: a model, a family or a use of the definitions ("ocr",
// "ppocr:small").
func OCR(name string) ai.Use[*TextReader] {
	return ai.Define[*TextReader](ocrRole, name)
}

// The definition of an OCR model may give the settings threads and cpu, and the device
// (common.go). On an NPU each size of the detector's and the recognizer's programs is converted
// the first time a picture (a line) takes it. ocrRole's Prepare and Load are the engine's
// (ai/vision/engine).
var ocrRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &TextReader{p: airpc.RemoteProxy[textReaderImpl](h)} },
	Name:   "ocr",
	Kinds:  []catalog.Kind{catalog.KindOCR},
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
	p airpc.Proxy[textReaderImpl]
}

// textReaderImpl is an OCR model of the engine in this process.
type textReaderImpl interface {
	Model() string
	Device() string
	Where() string
	Read(ctx context.Context, img image.Image, o OCROptions) (*Text, error)
	Close() error
}

// Model is the name of the model in the definitions.
func (t *TextReader) Model() string { return airpc.ProxyValue(t.p, "Model", textReaderImpl.Model) }

// Device is where the networks run: npu or cpu.
func (t *TextReader) Device() string { return airpc.ProxyValue(t.p, "Device", textReaderImpl.Device) }

// Where says where the networks run, for a log.
func (t *TextReader) Where() string { return airpc.ProxyValue(t.p, "Where", textReaderImpl.Where) }

// Read reads the lines of text in a picture.
func (t *TextReader) Read(ctx context.Context, img image.Image, o OCROptions) (*Text, error) {
	return airpc.ProxyCall1(t.p, ctx, "Read", func(l textReaderImpl) (*Text, error) {
		return l.Read(ctx, img, o)
	}, img, o)
}

// Close unloads the models. An app leaves it to the Runtime (its Close, or the use's Release).
func (t *TextReader) Close() error { return t.p.Close() }

func init() {
	airole.Register(ocrRole)
	bridge.Roles[ocrRole.Name] = ocrRole
	bridge.Wrap[ocrRole.Name] = func(l any) io.Closer { return &TextReader{p: airpc.LocalProxy(l.(textReaderImpl))} }
}
