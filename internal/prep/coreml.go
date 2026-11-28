package prep

import (
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/notes"
	"github.com/shibukawa/hwnpu/internal/pitch"
	"github.com/shibukawa/hwnpu/internal/pose"
	"github.com/shibukawa/hwnpu/internal/vision/detect"
	"github.com/shibukawa/hwnpu/internal/vision/face"
	"github.com/shibukawa/hwnpu/internal/vision/segment"
)

// The Core ML packages of the models the CPU runs converted (cache.CoreMLTarget): what hwnpu's
// own executor runs where the definitions put the model on the CPU (models.yaml's device: cpu),
// made from the ONNX file once here and put in the sources, so that such a machine takes them in
// place of the ONNX file and its conversion. What a run reads besides a program goes in its package
// as a part file (onnx.PartExt): RTMO's host part, Basic Pitch's front end, SwiftF0's filter
// banks, EfficientSAM's decoder, PP-OCR's settings; YuNet's outputs' shapes are in its packages'
// metadata, RTMDet reads its program alone.
func init() {
	coreml := []string{cache.CoreMLTarget}
	Register(&Preparer{Name: "detect", Kinds: []chat.Kind{chat.KindDetect}, Targets: coreml, Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		file, err := onlyFile(e.Model, ".onnx")
		if err != nil {
			return f, err
		}
		model, program, err := detect.Prepare(e.Ctx, detect.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), File: file, SHA256: e.Model.SHA256[file], Backend: llm.BackendCoreML, Log: e.Log})
		f.Set("model", model)
		f.Set("program", program)
		return f, err
	}})
	Register(&Preparer{Name: "pose", Kinds: []chat.Kind{chat.KindPose}, Targets: coreml, Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		model, program, err := pose.Prepare(e.Ctx, pose.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), SHA256: e.Model.SHA256[pose.ModelFile], Backend: llm.BackendCoreML, Log: e.Log})
		f.Set("model", model)
		f.Set("program", program)
		return f, err
	}})
	Register(&Preparer{Name: "pitch", Kinds: []chat.Kind{chat.KindPitch}, Targets: coreml, Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		model, program, err := pitch.Prepare(e.Ctx, pitch.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), Backend: llm.BackendCoreML, Log: e.Log})
		f.Set("model", model)
		f.Set("program", program)
		return f, err
	}})
	Register(&Preparer{Name: "notes", Kinds: []chat.Kind{chat.KindNotes}, Targets: coreml, Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		model, program, err := notes.Prepare(e.Ctx, notes.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), SHA256: e.Model.SHA256[notes.ModelFile], Backend: llm.BackendCoreML, Log: e.Log})
		f.Set("model", model)
		f.Set("program", program)
		return f, err
	}})
	Register(&Preparer{Name: "segment", Kinds: []chat.Kind{chat.KindSegment}, Targets: coreml, Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		setup, err := segmentSetup(e)
		if err != nil {
			return f, err
		}
		program, err := segment.PrepareCPU(chat.WithModel(e.Ctx, e.Model), setup, setup.Dir)
		f.Set("program", program)
		return f, err
	}})
	Register(&Preparer{Name: "ocr", Kinds: []chat.Kind{chat.KindOCR}, Targets: coreml, Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		return f, prepareOCR(e, llm.BackendCoreML)
	}})
	Register(&Preparer{Name: "faces", Kinds: []chat.Kind{chat.KindFace}, Targets: coreml, Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		dir := chat.ModelDir(e.Model)
		model, err := face.PreparePrograms(e.Ctx, llm.BackendCoreML, func() (string, error) {
			return chat.Fetch(e.Ctx, e.Model.Source(), imagedev.YuNetFile, e.Log)
		}, dir, face.CPUSizes)
		f.Set("model", model)
		f.Set("programs", dir)
		return f, err
	}})
}
