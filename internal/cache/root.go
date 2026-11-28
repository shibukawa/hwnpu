package cache

import (
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

// RootEnv moves the root of everything hwnpu keeps (Root) elsewhere: an external disk, a
// machine's shared disk (docs/cache-sharing-design.md 7.3).
const RootEnv = "HWNPU_CACHE"

// Root is the one directory hwnpu writes what it keeps under: hwnpu in the OS's cache directory
// that does not roam (~/Library/Caches on macOS, $XDG_CACHE_HOME or ~/.cache on Linux,
// %LOCALAPPDATA% on Windows), or RootEnv. Under it (docs/cache-sharing-design.md 7.3):
//
//	models/     the models' checkpoints and the programs converted from them
//	entries/    exports made with IRON for AMD's NPU (TARGET/NAME), with their entry.json
//	iron/       IRON's own cache of compiled kernels (NPU_CACHE_HOME)
//	layers/     what the shared layers hold here: the index of what was put in them
//	locks/      the locks of models and files being prepared
//	work/       what is being written (on the same file system, for rename and reflinks)
//	prompts/    the prompt store; history/ the terminal's input history
//
// and the compile caches' entries (KEY.mlmodelc, KEY.blob) at its top, as before.
func Root() string {
	if d := setRoot.Load(); d != nil {
		return *d
	}
	if d := os.Getenv(RootEnv); d != "" {
		return d
	}
	if testing.Testing() {
		return testRoot() // go test never writes the machine's cache
	}
	d, err := os.UserCacheDir()
	if err != nil {
		d = os.TempDir()
	}
	return filepath.Join(d, "hwnpu")
}

var setRoot atomic.Pointer[string]

// SetRoot moves Root to dir for this process, over RootEnv ("" for RootEnv's again): an
// application that keeps hwnpu's files where it says (modelstore.Options.Root).
func SetRoot(dir string) {
	if dir == "" {
		setRoot.Store(nil)
		return
	}
	setRoot.Store(&dir)
}

// testRoot is Root under go test without RootEnv: a directory of the process's own.
var testRoot = sync.OnceValue(func() string {
	d, err := os.MkdirTemp("", "hwnpu-test-cache")
	if err != nil {
		panic(err)
	}
	return d
})

// Sub is a directory under Root.
func Sub(elem ...string) string {
	return filepath.Join(append([]string{Root()}, elem...)...)
}

// AMDTarget is the target (docs/model-table.md 4.1) of the exports for AMD's NPU hwnpu makes:
// XDNA 2 (IRON's npu2).
const AMDTarget = "amd-xdna2"

// CoreAITarget is the target of the Core AI exports (.aimodel, MLIR bytecode): the same on
// every Mac; what is compiled for a chip goes as its own target, CoreAITarget-ARCH.
const CoreAITarget = "coreai"

// CoreMLTarget is the target of the Core ML packages (.mlpackage) of the models the CPU runs
// converted (hwnpu's own executor, internal/milcpu): made from their ONNX files, the same on
// every OS, so that a machine takes them in place of the ONNX file and its conversion.
const CoreMLTarget = "coreml"

// Entries is where the exports for a target are kept: entries/TARGET under Root.
func Entries(target string) string { return Sub("entries", target) }

// LegacyExports is ~/models, where exports for AMD's NPU were kept before Root held them (and
// where the exporters run by hand still write): read, never written (docs/cache-sharing-design.md 7.3).
func LegacyExports() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, "models")
}

// ExportRoots are the directories an export for AMD's NPU is looked for in: Entries(AMDTarget),
// then LegacyExports.
func ExportRoots() []string {
	roots := []string{Entries(AMDTarget)}
	if l := LegacyExports(); l != "" {
		roots = append(roots, l)
	}
	return roots
}

// ExportStale says whether an export found by name is not the one the definitions pin (an
// export fetched or made by hwnpu, its entry.json of other bytes than models.yaml's exports pin):
// FindExport passes it by, and the export is fetched again at the pin. The chat package sets it.
var ExportStale = func(dir string) bool { return false }

// FindExport is the first of ExportRoots/name that have says holds an export (and ExportStale
// does not refuse), "" for none.
func FindExport(name string, have func(dir string) bool) string {
	for _, r := range ExportRoots() {
		if d := filepath.Join(r, name); have(d) && !ExportStale(d) {
			Touch(d) // used now (LRU)
			return d
		}
	}
	return ""
}
