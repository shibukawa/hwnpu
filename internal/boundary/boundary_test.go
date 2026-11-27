// Package boundary checks that the packages of hwnpu's clients stay without the engine
// (docs/design/hwnpud.md, "Clients without the engine"): what they import must not reach the
// packages that convert and run the models.
package boundary

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

const module = "github.com/shibukawa/hwnpu"

// light are the packages a client links, which must not reach engine; the list grows as the
// clients are taken apart from the engine.
var light = []string{
	"internal/daemon",
	"internal/envfile",
	"internal/airpc",
	"internal/catalog",
	"internal/airole",
	"internal/aieng",
	"modelstore",
	"ai",
}

// engine are the packages of the engine: the conversions, the runtimes and the backends.
var engine = []string{
	"internal/llm",
	"internal/backend",
	"internal/coreai",
	"internal/mlpackage",
	"internal/imagegen",
	"internal/chat",
	"internal/aihost",
	"internal/daemon/engine",
	"ai/engine",
}

func TestLightPackages(t *testing.T) {
	gobin := filepath.Join(runtime.GOROOT(), "bin", "go")
	for _, p := range light {
		out, err := exec.Command(gobin, "list", "-deps", module+"/"+p).CombinedOutput()
		if err != nil {
			t.Fatalf("go list %s: %v\n%s", p, err, out)
		}
		for _, dep := range strings.Fields(string(out)) {
			rel, ok := strings.CutPrefix(dep, module+"/")
			if !ok {
				continue
			}
			for _, e := range engine {
				if rel == e || strings.HasPrefix(rel, e+"/") {
					t.Errorf("%s reaches %s, a package of the engine", p, rel)
				}
			}
		}
	}
}
