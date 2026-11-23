package prep

import (
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux3klein"
	"github.com/shibukawa/hwnpu/internal/music"
	"github.com/shibukawa/hwnpu/internal/recipe"
	"github.com/shibukawa/hwnpu/internal/sfx"
	"github.com/shibukawa/hwnpu/internal/speech"
	"github.com/shibukawa/hwnpu/internal/supertonic"
	"github.com/shibukawa/hwnpu/internal/transcribe"
	"github.com/shibukawa/hwnpu/internal/vision/ironcnn"
	"github.com/shibukawa/hwnpu/internal/vision/segment"
)

// The scripts of tools/amd whose exports hwnpu reads, by kind: a model of the kind with an
// export for AMD's NPU must have a recipe of one.
var amdScripts = map[chat.Kind][]string{
	chat.KindTranscribe: {transcribe.AMDTool},
	chat.KindSpeech:     {speech.AMDTool, supertonic.AMDTool},
	chat.KindMusic:      {music.AMDTool},
	chat.KindSFX:        {sfx.AMDTool},
	chat.KindImage:      {flux2klein.AMDTool, flux3klein.AMDTool, "iron_anima.py"},
	chat.KindDetect:     {ironcnn.Tool},
	chat.KindPose:       {ironcnn.Tool},
	chat.KindSegment:    {segment.Tool},
	chat.KindDepth:      {"iron_da3.py", "iron_moge.py"},
	chat.KindLabel:      {"iron_siglip2.py"},
	chat.KindFace:       {"iron_yunet.py"},
	chat.KindEmbed:      {chat.EmbedAMDTool},
}

// every value a kind's code gives a recipe's placeholders, for the check
var allVars = map[string][]string{recipe.In: {"/in"}, recipe.Sizes: {"1024"}, recipe.Size: {"640x640"},
	recipe.Outputs: {"a"}, recipe.Text: {"fp16"}, recipe.DiT: {"dit.safetensors"}}

// Every model of the definitions that hwnpu makes an export for AMD's NPU of has a recipe of a
// script its code reads, and the recipe's placeholders are values its code gives.
func TestAMDRecipes(t *testing.T) {
	for _, m := range chat.Catalog {
		want, ok := amdScripts[m.Kind]
		if !ok {
			continue // a language model: its recipe gives AMDExport (internal/chat)
		}
		if m.Kind == chat.KindEmbed && !strings.HasPrefix(m.Family, "plamo-embedding") {
			continue
		}
		c, err := chat.AMDCommand(m, allVars, want...)
		if err != nil {
			t.Errorf("%s: %v", m.Name, err)
			continue
		}
		if !strings.Contains(strings.Join(c.Args, " "), chat.ExportOut) {
			t.Errorf("%s: %v has no %s", m.Name, c.Args, chat.ExportOut)
		}
	}
}

// The commands are those the code ran before the recipes, so that the exports are the same
// (and their keys: the args of those keyed by them).
func TestAMDRecipesAsBefore(t *testing.T) {
	for name, c := range map[string]struct {
		vars map[string][]string
		want string
	}{
		"RTMDet-m":                   {map[string][]string{recipe.In: {"m.onnx"}, recipe.Size: {"640x640"}, recipe.Outputs: {"a", "b"}}, "m.onnx {out} --size 640x640 --outputs a b"},
		"RTMO-s":                     {map[string][]string{recipe.In: {"m.onnx"}, recipe.Size: {"640x640"}, recipe.Outputs: {"a"}}, "m.onnx {out} --size 640x640 --mean 0 0 0 --std 1 1 1 --silu op --precision HIGH --dw-precision HIGH --outputs a"},
		"MOSS-SoundEffect-v2.0-lut8": {map[string][]string{recipe.In: {"/r"}, recipe.Text: {"lut8"}}, "/r {out} --text lut8"},
		"Anima-Turbo":                {map[string][]string{recipe.In: {"/root"}, recipe.Sizes: {"1024", "832x1248"}, recipe.DiT: {"d.safetensors"}}, "/root {out} --size 1024 832x1248 --dit d.safetensors"},
		"supertonic-3":               {map[string][]string{recipe.In: {"/r/onnx"}, recipe.Sizes: {"64x32"}}, "/r/onnx {out} --sizes 64x32"},
		"DA3-SMALL":                  {chat.InVar("/r"), "/r {out}"},
		"moge-2-vitl":                {chat.InVar("/r"), "/r {out}"},
		"EfficientSAM-S":             {chat.InVar("e.onnx"), "e.onnx {out}"},
	} {
		m, _ := chat.Lookup(name)
		got, err := chat.AMDCommand(m, c.vars)
		if err != nil || strings.Join(got.Args, " ") != c.want {
			t.Errorf("%s: %v %v, want %s", name, got.Args, err, c.want)
		}
	}
}
