package prep

import (
	"fmt"
	"io"
	"strings"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/vision/depth"
	"github.com/shibukawa/hwnpu/internal/vision/face"
	"github.com/shibukawa/hwnpu/internal/vision/moge"
	"github.com/shibukawa/hwnpu/internal/vision/ocr"
	"github.com/shibukawa/hwnpu/internal/vision/segment"
)

// depth: DA3's and MoGe-2's export with every size (depth.Sizes, moge.Aspects), which
// hwnpu-image depth converts as a picture first takes its aspect.
func init() {
	Register(&Preparer{Name: "depth", Kinds: []chat.Kind{chat.KindDepth}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		m := e.Model
		var dir string
		var err error
		if m.Family == "moge" {
			if dir, err = moge.Prepare(e.Ctx, moge.Setup{Dir: chat.ModelsDir(), Repo: m.Repo, Name: m.Name, Backend: llm.BackendCoreAI, Log: e.Log}); err == nil {
				err = moge.PrepareSizes(chat.WithModel(e.Ctx, m), dir, logf(e.Log))
			}
		} else if dir, err = depth.Prepare(e.Ctx, depth.Setup{Dir: chat.ModelsDir(), Repo: m.Repo, Name: m.Name, Backend: llm.BackendCoreAI, Log: e.Log}); err == nil {
			err = depth.PrepareSizes(chat.WithModel(e.Ctx, m), dir, logf(e.Log))
		}
		f.Set("export", dir)
		return err
	})})
	// face-detect: YuNet's program of each size (face.CPUSizes), converted as a picture first
	// takes it
	Register(&Preparer{Name: "faces", Kinds: []chat.Kind{chat.KindFace}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		path, err := chat.Fetch(e.Ctx, e.Model.Source(), "face_detection_yunet_2023mar.onnx", e.Log)
		if err != nil {
			return err
		}
		f.Set("programs", chat.ModelDir(e.Model))
		return face.PreparePrograms(chat.WithModel(e.Ctx, e.Model), llm.BackendCoreAI, path, chat.ModelDir(e.Model))
	})})
	// ocr: PP-OCR's detector of each size and recognizer of each width (ocr.Sizes, ocr.Widths)
	Register(&Preparer{Name: "ocr", Kinds: []chat.Kind{chat.KindOCR}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		m := e.Model
		det := m.With["detector"]
		if det == "" {
			return fmt.Errorf("models.yaml gives no detector (with: {detector: hf:OWNER/NAME})")
		}
		detSHA, recSHA := map[string]string{}, map[string]string{}
		for file, d := range m.SHA256 {
			if rest, ok := strings.CutPrefix(file, "detector/"); ok {
				detSHA[rest] = d
			} else {
				recSHA[file] = d
			}
		}
		r, err := ocr.Open(e.Ctx, nil, ocr.Setup{Dir: chat.ModelsDir(), Det: ocr.Source{Repo: det, SHA256: detSHA}, Rec: ocr.Source{Repo: m.Source(), SHA256: recSHA}, Backend: llm.BackendCoreAI, Log: e.Log})
		if err != nil {
			return err
		}
		return r.PreparePrograms(chat.WithModel(e.Ctx, m))
	})})
	// segment: EfficientSAM's encoder (the decoder runs on the host from its ONNX file)
	Register(&Preparer{Name: "segment", Kinds: []chat.Kind{chat.KindSegment}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		m := e.Model
		var encFile, decFile string
		for file := range m.SHA256 {
			switch {
			case strings.HasSuffix(file, "_encoder.onnx"):
				encFile = file
			case strings.HasSuffix(file, "_decoder.onnx"):
				decFile = file
			}
		}
		if encFile == "" || decFile == "" {
			return fmt.Errorf("models.yaml names no *_encoder.onnx and *_decoder.onnx in its sha256")
		}
		setup := segment.Setup{Dir: chat.ModelDir(m), Repo: m.Source(), Encoder: encFile, Decoder: decFile, SHA256: m.SHA256, Log: e.Log}
		enc, _, err := segment.Fetch(e.Ctx, setup)
		if err != nil {
			return err
		}
		program, err := segment.ConvertEncoder(chat.WithModel(e.Ctx, m), llm.BackendCoreAI, enc, setup.Dir, e.Log)
		f.Set("program", program)
		return err
	})})
}

// logf writes lines to w (nil for nowhere).
func logf(w io.Writer) func(string, ...any) {
	return func(format string, args ...any) {
		if w != nil {
			fmt.Fprintf(w, format+"\n", args...)
		}
	}
}
