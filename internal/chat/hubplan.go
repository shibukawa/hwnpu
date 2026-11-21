package chat

import (
	"context"
	"fmt"
	"maps"
	"path"
	"strings"

	"github.com/shibukawa/hwnpu/internal/coreai/ailower"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// What hwnpu-dev hf plan and push look for in the sources: the entries a model's conversions
// have now, as the commands would look for them (docs/prebuilt.md 4.1). The exports for AMD's
// NPU are known by their recipes (ExportRecipe, PinnedRecipe); the Core AI exports, their
// prebuilt programs and the tables by the names the sources hold them under, each taken only
// when it is what this hwnpu would make of the model now.

// Takes reports whether got, found at want's place, is what want describes: the same key, or
// for a version that is published (vN) the same recipe but for the tools' hash.
func Takes(want, got shared.Entry) bool { return place{want}.takes(got) }

// CoreAIWant is the entry of the Core AI export of model m named export in the sources
// (NAME.coreai, or NAME.coreai/SIZE for one of several sizes), as the commands look for it now;
// false when neither the digests of the model's files nor their commit are known.
func CoreAIWant(ctx context.Context, m Model, export string) (shared.Entry, bool) {
	_, r := modelRecipe(WithModel(ctx, m), "")
	if r == nil {
		return shared.Entry{}, false
	}
	r["kind"], r["target"], r["export"], r["lowering"] = CoreAIKind, CoreAITarget, export, fmt.Sprint(ailower.FormatVersion)
	return placeOf(CoreAIKind, m, CoreAITarget, export, r).entry, true
}

// TablesWant is the entry of model m's tables that got (an entry of tables found in a source)
// would be if made now: the same tables of the same architecture, of the model's files and the
// tables' version of this hwnpu. false when got is not tables, or the model's files are not known.
func TablesWant(ctx context.Context, m Model, got shared.Entry) (shared.Entry, bool) {
	if got.Kind != TablesKind {
		return shared.Entry{}, false
	}
	_, r := modelRecipe(WithModel(ctx, m), "")
	if r == nil {
		return shared.Entry{}, false
	}
	r["kind"], r["version"] = TablesKind, fmt.Sprint(tablesVersion)
	for _, k := range []string{"arch", "tables", "int8"} {
		if v, ok := got.Recipe[k]; ok {
			r[k] = v
		}
	}
	return placeOf(TablesKind, m, "", path.Base(path.Dir(got.Path)), r).entry, true
}

// PrebuiltWant is the entry of the prebuilt programs for the architecture arch (h14g) of the
// Core AI export of model m named export whose programs' digest (coreai.ProgramsDigest) is
// digest.
func PrebuiltWant(m Model, export, arch, digest string) shared.Entry {
	r := map[string]string{"kind": PrebuiltKind, "export": export, "arch": arch, "programs": digest, "format": fmt.Sprint(prebuiltFormat)}
	return placeOf(PrebuiltKind, m, CoreAITarget+"-"+arch, export, r).entry
}

// IsPrebuilt reports whether e is an entry of prebuilt programs this hwnpu reads (of its
// format), and gives their export and architecture.
func IsPrebuilt(e shared.Entry) (export, arch string, ok bool) {
	if e.Kind != PrebuiltKind || e.Recipe["format"] != fmt.Sprint(prebuiltFormat) {
		return "", "", false
	}
	return e.Recipe["export"], e.Recipe["arch"], true
}

// Unversioned is where the entry want would be found with no version: v0-HASH of its key, which
// is the same for every version (the version is not in the recipe). An entry there with want's
// key is what want's version holds, byte for byte.
func Unversioned(want shared.Entry) string {
	return path.Join(path.Dir(want.Path), shared.VersionDir(0, false, want.Key))
}

// Released is want as it goes to its published version: e (found at Unversioned, of want's key)
// with want's version and place.
func Released(want, e shared.Entry) shared.Entry {
	e.Version, e.Develop, e.Path = want.Version, want.Develop, want.Path
	e.Recipe = maps.Clone(e.Recipe)
	return e
}

// HubCommitPinned reports whether model m's files are of a commit that does not move: its
// revision is a commit, or the definitions pin every file it reads (sha256).
func HubCommitPinned(m Model) bool {
	return commitRE.MatchString(strings.ToLower(m.Revision)) || len(m.SHA256) > 0
}

// SourceCommit is the commit model m's files come from (its revision when that is a commit,
// else the sources' lock, else Hugging Face's main); "" when none is known.
func SourceCommit(ctx context.Context, m Model) string { return sourceCommit(ctx, m) }
