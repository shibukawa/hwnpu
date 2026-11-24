package prep

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/bonsaiai/internal/airole"
	"github.com/shibukawa/bonsaiai/internal/cache"
	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/internal/imagegen/aspect"
	"github.com/shibukawa/bonsaiai/internal/imagegen/flux2klein"
	"github.com/shibukawa/bonsaiai/internal/imagegen/flux3klein"
	"github.com/shibukawa/bonsaiai/internal/music"
	"github.com/shibukawa/bonsaiai/internal/pose"
	"github.com/shibukawa/bonsaiai/internal/sfx"
	"github.com/shibukawa/bonsaiai/internal/vision/detect"
	"github.com/shibukawa/bonsaiai/internal/vision/ironcnn"
	"github.com/shibukawa/bonsaiai/internal/vision/segment"
)

// The exports for AMD's NPU (amd-xdna2) of the kinds the library's roles do not make: those
// their command makes with IRON on its first run (detect, pose, segment, sfx, music's decoder,
// FLUX.3 klein), Anima's (made with its script of tools/amd: handmade.go), and the
// kinds that run on the CPU beside AMD's NPU (nothing to make).
func init() {
	amd := []string{cache.AMDTarget}
	Register(&Preparer{Name: "detect", Kinds: []chat.Kind{chat.KindDetect}, Targets: amd, Prepare: onAMDNPU(func(e airole.PrepareEnv, f *airole.Files) error {
		file, err := onlyFile(e.Model, ".onnx")
		if err != nil {
			return err
		}
		setup := detect.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), File: file, SHA256: e.Model.SHA256[file], Log: e.Log}
		dir, err := ironcnn.FirstRunExport(e.Ctx, e.Model, "iron_onnxcnn.py", detect.ExportVersion, readCNN,
			func() (string, error) { return detect.Fetch(e.Ctx, setup) },
			func(python []string, model, dir string) error {
				return detect.ExportNPU(e.Ctx, python, e.Model.Name, model, dir, e.Log)
			}, e.Log)
		f.Set("export", dir)
		return err
	})})
	Register(&Preparer{Name: "pose", Kinds: []chat.Kind{chat.KindPose}, Targets: amd, Prepare: onAMDNPU(func(e airole.PrepareEnv, f *airole.Files) error {
		setup := pose.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), SHA256: e.Model.SHA256[pose.ModelFile], Log: e.Log}
		dir, err := ironcnn.FirstRunExport(e.Ctx, e.Model, "iron_onnxcnn.py", pose.ExportVersion, readCNN,
			func() (string, error) { return pose.Fetch(e.Ctx, setup) },
			func(python []string, model, dir string) error {
				return pose.ExportNPU(e.Ctx, python, e.Model.Name, model, dir, e.Log)
			}, e.Log)
		f.Set("export", dir)
		return err
	})})
	Register(&Preparer{Name: "segment", Kinds: []chat.Kind{chat.KindSegment}, Targets: amd, Prepare: onAMDNPU(func(e airole.PrepareEnv, f *airole.Files) error {
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
		setup := segment.Setup{Dir: chat.ModelDir(m), Repo: m.Source(), Encoder: encFile, Decoder: decFile, SHA256: m.SHA256, Log: e.Log}
		have := func(dir string) error { _, err := os.Stat(filepath.Join(dir, "model.json")); return err }
		dir, err := ironcnn.FirstRunExport(e.Ctx, m, "iron_efficientsam.py", segment.ExportVersion, have,
			func() (string, error) { enc, _, err := segment.Fetch(e.Ctx, setup); return enc, err },
			func(python []string, model, dir string) error {
				return segment.ExportNPU(e.Ctx, python, m.Name, model, dir, e.Log)
			}, e.Log)
		f.Set("export", dir)
		return err
	})})
	Register(&Preparer{Name: "sfx", Kinds: []chat.Kind{chat.KindSFX}, Targets: amd, Prepare: amdOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		// the text encoder's format of a machine of each memory (sfx.Formats)
		machines := []int{e.Model.Machine}
		if e.Model.Machine == 0 {
			machines = []int{8, 16}
		}
		for _, gb := range machines {
			text, _ := sfx.Formats(gb)
			p, err := sfx.PrepareAMD(chat.WithModel(e.Ctx, e.Model), e.Model, text, e.Log)
			if err != nil {
				return err
			}
			f.Set(fmt.Sprintf("dir-%dg", gb), p.Dir)
		}
		return nil
	})})
	Register(&Preparer{Name: "music", Kinds: []chat.Kind{chat.KindMusic}, Targets: amd, Prepare: amdOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		// SpectroStream's decoder; the language model and MusicCoCa run on the CPU interpreter
		dir, err := music.PrepareAMD(chat.WithModel(e.Ctx, e.Model), e.Model, e.Log)
		f.Set("decoder", dir)
		return err
	})})
	Register(&Preparer{Name: "image", Kinds: []chat.Kind{chat.KindImage}, Targets: amd, Prepare: amdOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		m := e.Model
		ctx := chat.WithModel(e.Ctx, m)
		var dir string
		var err error
		switch {
		case strings.HasPrefix(m.Family, "flux2-klein"), strings.HasPrefix(m.Family, "flux3-klein"):
			// one export of every size of models.yaml npu.amd
			as, aerr := flux2klein.NPUAspects(m, "amd", flux2klein.NPUGeneration("amd"))
			if aerr != nil {
				return aerr
			}
			if strings.HasPrefix(m.Family, "flux2-klein") {
				dir, err = flux2klein.PrepareAMD(ctx, m, aspect.SizesOf(as), as[0].Width, as[0].Height, e.Log)
			} else {
				dir, err = flux3klein.PrepareAMD(ctx, m, aspect.SizesOf(as), as[0].Width, as[0].Height, e.Log)
			}
		case strings.HasPrefix(m.Family, "anima"):
			dir, err = imageHandMade(e)
		default:
			err = fmt.Errorf("no image model of bonsaiai-image generate of the family %q", m.Family)
		}
		f.Set("export", dir)
		return err
	})})
	Register(&Preparer{Name: "decision", Kinds: []chat.Kind{chat.KindDecision}, Targets: amd, Prepare: amdOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		// d1: an export of tools/amd/iron_export_lm.py d1 (its recipe, model-recipes.yaml), as a chat model's
		if len(e.Model.AMDExport) == 0 {
			fmt.Fprintf(e.Log, "%s: no recipe of iron_export_lm.py: nothing to make for AMD's NPU\n", e.Model.Name)
			return nil
		}
		dir, err := chat.AMDExportOf(chat.WithModel(e.Ctx, e.Model), e.Model, e.Log)
		f.Set("export", dir)
		return err
	})})
	Register(&Preparer{Name: "embed", Kinds: []chat.Kind{chat.KindEmbed}, Targets: amd, Prepare: amdOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		// PLaMo-Embedding: an export of tools/amd/iron_plamo_embed.py
		dir, err := chat.PrepareEmbedderAMD(chat.WithModel(e.Ctx, e.Model), e.Model, e.Log)
		f.Set("export", dir)
		return err
	})})
	// the kinds with no graph for AMD's NPU: they run on the CPU beside it (ONNX Runtime, or the
	// CPU interpreter on Core ML packages made where they run)
	Register(&Preparer{Name: "cpu", Kinds: []chat.Kind{chat.KindOCR, chat.KindPitch, chat.KindNotes}, Targets: amd,
		Prepare: func(e airole.PrepareEnv) (airole.Files, error) {
			fmt.Fprintf(e.Log, "%s: runs on the CPU beside AMD's NPU: nothing to make for it\n", e.Model.Name)
			return airole.Files{}, nil
		}})
}

// readCNN says whether dir holds an export of tools/amd/iron_onnxcnn.py.
func readCNN(dir string) error { _, err := ironcnn.Read(dir); return err }

// amdOnly is a Prepare that makes the export for AMD's NPU with prepare.
func amdOnly(prepare func(airole.PrepareEnv, *airole.Files) error) func(airole.PrepareEnv) (airole.Files, error) {
	return func(e airole.PrepareEnv) (airole.Files, error) {
		var f airole.Files
		if !e.NPU.AMD() {
			return f, notYet(e.Model, e.NPU.Target)
		}
		if err := prepare(e, &f); err != nil {
			return f, fmt.Errorf("%s: %w", e.Model.Name, err)
		}
		return f, nil
	}
}

// onAMDNPU is amdOnly for a model the definitions may run on the CPU beside AMD's NPU
// (models.yaml device: {amd: cpu}): nothing is made for one of those.
func onAMDNPU(prepare func(airole.PrepareEnv, *airole.Files) error) func(airole.PrepareEnv) (airole.Files, error) {
	return amdOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		if device, _, err := chat.ChooseDevice(e.Model, "amd", ""); err != nil || device != "npu" {
			fmt.Fprintf(e.Log, "%s: runs on the CPU beside AMD's NPU (models.yaml device): nothing to make for it\n", e.Model.Name)
			return nil
		}
		return prepare(e, f)
	})
}
