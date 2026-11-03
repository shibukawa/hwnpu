//go:build windows

package amdxrt

import (
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/dl"
)

// preloadXRT loads xrt_coreutil.dll by its full path before the shim, so that the shim's
// import of it resolves to the loaded module: AMD's NPU driver installs XRT outside the DLL
// search path (System32\AMD, or its driver store folder). XRTEnv names it first. Nothing is
// loaded when none is found: the shim's own load then reports the missing DLL.
func preloadXRT() {
	for _, p := range xrtCandidates() {
		if _, err := os.Stat(p); err != nil {
			continue
		}
		if _, err := dl.Open(p); err == nil {
			return
		}
	}
}

func xrtCandidates() []string {
	var c []string
	if p := os.Getenv(XRTEnv); p != "" {
		c = append(c, p)
	}
	root := os.Getenv("SystemRoot")
	if root == "" {
		root = `C:\Windows`
	}
	c = append(c, filepath.Join(root, "System32", "AMD", "xrt_coreutil.dll"))
	found, _ := filepath.Glob(filepath.Join(root, "System32", "DriverStore", "FileRepository", "*", "xrt_coreutil.dll"))
	return append(c, found...)
}
