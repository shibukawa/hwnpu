package prep

import (
	"cmp"
	"fmt"
	"maps"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/music"
	"github.com/shibukawa/hwnpu/internal/notes"
	"github.com/shibukawa/hwnpu/internal/pitch"
	"github.com/shibukawa/hwnpu/internal/pose"
	"github.com/shibukawa/hwnpu/internal/recipe"
	"github.com/shibukawa/hwnpu/internal/sfx"
	"github.com/shibukawa/hwnpu/internal/systemone"
	"github.com/shibukawa/hwnpu/internal/vision/detect"
	"github.com/shibukawa/hwnpu/internal/vision/siglip"
)

// The kinds whose programs are of one shape, converted in Go from the model's files: what their
// command's first run converts for Core AI (hwnpu-image detect, pose, label and embed;
// hwnpu-audio pitch, notes, generate and generate -m sfx; hwnpu-chat search and systemone).
func init() {
	Register(&Preparer{Name: "detect", Kinds: []chat.Kind{chat.KindDetect}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		file, err := onlyFile(e.Model, ".onnx")
		if err != nil {
			return err
		}
		model, program, err := detect.Prepare(e.Ctx, detect.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), File: file, SHA256: e.Model.SHA256[file], Backend: llm.BackendCoreAI, Log: e.Log})
		f.Set("model", model)
		f.Set("program", program)
		return err
	})})
	Register(&Preparer{Name: "pose", Kinds: []chat.Kind{chat.KindPose}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		model, program, err := pose.Prepare(e.Ctx, pose.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), SHA256: e.Model.SHA256[pose.ModelFile], Backend: llm.BackendCoreAI, Log: e.Log})
		f.Set("model", model)
		f.Set("program", program)
		return err
	})})
	Register(&Preparer{Name: "pitch", Kinds: []chat.Kind{chat.KindPitch}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		model, program, err := pitch.Prepare(e.Ctx, pitch.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), Backend: llm.BackendCoreAI, Log: e.Log})
		f.Set("model", model)
		f.Set("program", program)
		return err
	})})
	Register(&Preparer{Name: "notes", Kinds: []chat.Kind{chat.KindNotes}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		model, program, err := notes.Prepare(e.Ctx, notes.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), SHA256: e.Model.SHA256[notes.ModelFile], Backend: llm.BackendCoreAI, Log: e.Log})
		f.Set("model", model)
		f.Set("program", program)
		return err
	})})
	Register(&Preparer{Name: "label", Kinds: []chat.Kind{chat.KindLabel}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		dir, err := siglip.Prepare(chat.WithModel(e.Ctx, e.Model), siglip.Setup{Dir: chat.ModelsDir(), Repo: e.Model.Source(), Name: e.Model.Name, Format: llm.BackendCoreAI, Log: e.Log})
		f.Set("export", dir)
		return err
	})})
	Register(&Preparer{Name: "embed", Kinds: []chat.Kind{chat.KindEmbed}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		c, err := chat.PrepareEmbedder(e.Ctx, chat.EmbedSetup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), Name: e.Model.Name, Seqs: chat.DefaultEmbedSeqs, Backend: llm.BackendCoreAI, Log: e.Log})
		f.Set("programs", c.Packages...)
		return err
	})})
	Register(&Preparer{Name: "decision", Kinds: []chat.Kind{chat.KindDecision}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		c, err := systemone.Prepare(e.Ctx, systemone.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), Seqs: systemone.DefaultSeqs, Backend: llm.BackendCoreAI, Log: e.Log})
		f.Set("programs", c.Packages...)
		return err
	})})
	Register(&Preparer{Name: "music", Kinds: []chat.Kind{chat.KindMusic}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		// every part: styles from texts and from audio, and continuing a clip
		p, err := music.Prepare(e.Ctx, music.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), Backend: llm.BackendCoreAI, Log: e.Log, Text: true, AudioStyle: true, Encoder: true})
		f.Set("programs", p.Temporal, p.Depth, p.Decoder, p.Encoder, p.TextProgram, p.MusicProgram)
		return err
	})})
	Register(&Preparer{Name: "sfx", Kinds: []chat.Kind{chat.KindSFX}, Prepare: coreAIOnly(func(e airole.PrepareEnv, f *airole.Files) error {
		// the formats a machine of each memory takes (sfx.Formats), when the definitions pin none
		machines := []int{e.Model.Machine}
		if e.Model.Machine == 0 {
			machines = []int{8, 16}
		}
		// and the decoder's export of each of the recipe's rewrites (model-recipes.yaml), for
		// the chips of each
		recipes, err := recipe.Builtin()
		if err != nil {
			return err
		}
		for _, gb := range machines {
			for _, arch := range recipes.AppleOf(e.Model.Name).RewriteArchs() {
				p, err := sfx.Prepare(e.Ctx, sfx.Setup{Dir: chat.ModelDir(e.Model), Repo: e.Model.Source(), Machine: gb, Backend: llm.BackendCoreAI,
					Model: e.Model.Name, Arch: arch, Log: e.Log})
				if err != nil {
					return err
				}
				f.Set(fmt.Sprintf("dir-%dg", gb), p.Dir)
			}
		}
		return nil
	})})
}

// coreAIOnly is a Prepare that makes the Core AI programs with prepare, and for another target
// says it is not made yet.
func coreAIOnly(prepare func(airole.PrepareEnv, *airole.Files) error) func(airole.PrepareEnv) (airole.Files, error) {
	return func(e airole.PrepareEnv) (airole.Files, error) {
		var f airole.Files
		if e.NPU.Format != llm.BackendCoreAI {
			return f, notYet(e.Model, cmp.Or(e.NPU.Target, e.NPU.Format))
		}
		if err := prepare(e, &f); err != nil {
			return f, fmt.Errorf("%s: %w", e.Model.Name, err)
		}
		return f, nil
	}
}

// onlyFile is the one file of extension ext the definitions pin of m (hwnpu-image's).
func onlyFile(m chat.Model, ext string) (string, error) {
	var files []string
	for f := range maps.Keys(m.SHA256) {
		if strings.EqualFold(filepath.Ext(f), ext) && !strings.Contains(f, "/") {
			files = append(files, f)
		}
	}
	if len(files) != 1 {
		return "", fmt.Errorf("models.yaml's sha256 must name its one %s file (it names %d)", ext, len(files))
	}
	slices.Sort(files)
	return files[0], nil
}
