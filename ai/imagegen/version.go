package imagegen

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strconv"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/imagegen/anima"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux2klein"
	"github.com/shibukawa/hwnpu/internal/imagegen/flux3klein"
)

// coreAIVersion is the version in the name of a Core AI export Prepare makes (NAME-vN.coreai,
// its -device.coreai beside it too).
var coreAIVersion = regexp.MustCompile(`-v(\d+)(?:-device)?\.coreai$`)

// valueGiven is the files' value that says the export is the definition's (models.yaml programs:
// export), not one Prepare made.
const valueGiven = "export_given"

// exportVersions is the version of the Core AI export each family's Prepare makes now.
var exportVersions = map[string]int{
	famKlein: flux2klein.ExportVersion,
	famFlux3: flux3klein.ExportVersion,
	famAnima: anima.CoreAIExportVersion,
}

// currentExport is the role's Usable: a store's record of a Core AI export of another version
// than its family's Prepare makes now counts as not prepared, so that Prepare takes the export of
// this version (from the sources, or converted) instead of the load failing on the old layout
// (FLUX.2 klein's v6 after v7, 2026-11-27: "is of layout 6, this hwnpu reads 7") or running the
// old programs (Anima's). ai.Store.find does not look at an export's version itself. Exports by
// path (models.yaml programs: export) and AMD's and Intel's are left as they are.
func currentExport(f airole.Files) error {
	want, ok := exportVersions[f.Values["family"]]
	if !ok || f.Values[valueGiven] != "" {
		return nil
	}
	for _, key := range []string{"export", "device"} {
		dir := f.Path(key)
		if dir == "" {
			continue
		}
		m := coreAIVersion.FindStringSubmatch(filepath.Base(dir))
		if m == nil {
			continue
		}
		if v, _ := strconv.Atoi(m[1]); v != want {
			return fmt.Errorf("its export %s is of version %d, and this hwnpu makes %d", filepath.Base(dir), v, want)
		}
	}
	return nil
}
