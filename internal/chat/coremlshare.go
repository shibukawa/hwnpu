package chat

import (
	"context"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// CoreMLKind is the kind of entry of a Core ML package (NAME.mlpackage) in the shared layers:
// the program of a model the CPU runs converted from its ONNX file (hwnpu's own executor,
// internal/milcpu), made on any OS. A machine that takes it from a source needs neither the
// ONNX file nor its conversion: what the run reads besides the program is in the package too
// (part files, onnx.PartExt).
const CoreMLKind = "coreml"

// CoreMLTarget is the target of the Core ML packages (cache.CoreMLTarget), its version the
// model's Core AI one (the recipe's apple.version: the same code writes both).
const CoreMLTarget = cache.CoreMLTarget

// CoreMLKinds are the kinds of the models converted from ONNX files whose CPU runs their Core ML
// packages (hwnpu's own executor) with what the run reads beside them in part files: their
// packages go in the sources as their target (hwnpu-dev models package MODEL/coreml).
var CoreMLKinds = []Kind{KindFace, KindDetect, KindPose, KindSegment, KindOCR, KindPitch, KindNotes}

// isCoreML reports whether path is a Core ML package (NAME.mlpackage) the shared layers take.
func isCoreML(path string) bool { return filepath.Ext(path) == ".mlpackage" }

// coreMLRecipe is what the Core ML package at pkg is made from, the key of the shared layers: the
// model's files as coreAIRecipe's and the package's name (the revisions and the format it is
// converted with). nil when the model is not known.
func coreMLRecipe(ctx context.Context, pkg string) (Model, map[string]string) {
	m, r := modelRecipe(ctx, pkg)
	if r == nil {
		return Model{}, nil
	}
	// a model an app made of a model of the definitions (NAME+HASH: hwnpu-image --device cpu)
	// converts the same files: its packages are the definitions' model's
	if base, _, ok := strings.Cut(m.Name, "+"); ok {
		if b, found := Lookup(base); found {
			m = b
		}
	}
	r["kind"], r["target"], r["export"] = CoreMLKind, CoreMLTarget, filepath.Base(pkg)
	return m, r
}

// fetchCoreML makes the Core ML package pkg from the first source that has it. Unlike a Core AI
// export, nothing is put early on a miss: the package is written whole by the conversion, then
// put (putCoreML).
func fetchCoreML(ctx context.Context, pkg string) bool {
	m, r := coreMLRecipe(ctx, pkg)
	if r == nil {
		return false
	}
	return fetchFrom(ctx, placeOf(CoreMLKind, m, r["target"], r["export"], r), pkg, labelLog{sharedLog, filepath.Base(pkg), r["export"]})
}

// putCoreML puts the Core ML package made here in the layers that are written.
func putCoreML(ctx context.Context, pkg string) {
	m, r := coreMLRecipe(ctx, pkg)
	putShared(CoreMLKind, m, r, pkg, r["export"])
}
