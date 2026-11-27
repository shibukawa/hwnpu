package catalog

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/envfile"
)

// ModelsUsage is the usage of the -models flag of the commands that pick models.
const ModelsUsage = "definition files read over the built-in models.yaml, the machine's (/etc/hwnpu/models.yaml) and yours (~/.config/hwnpu/models.yaml, %LOCALAPPDATA%\\hwnpu\\models.yaml on Windows, HWNPU_MODELS), comma separated: models, uses and aliases of your own (docs/model-table.md)"

// PolicyError is the error of a model the policy refuses.
type PolicyError struct {
	Model string
	Why   string // what refuses it
}

func (e *PolicyError) Error() string {
	return fmt.Sprintf("chat: %s is not allowed by the model policy (%s, policy of models.yaml): %s", e.Model, envfile.PolicyEnv, e.Why)
}

// Definitions is a definition file given as its text, not a path: Name names it in messages
// and in a model's Files.
type Definitions struct {
	Name string
	YAML []byte
}

// Category is a use that names the model recommended for it on the machine (MODEL.md's rows):
// -use translate. Picks go by the machine's memory: the one of the largest Machine the machine
// has. CATEGORY:TAG picks the model of that tag among them (translate:0.8b).
type Category struct {
	Name    string
	Aliases []string
	About   string
	Picks   []CategoryPick
}

// CategoryPick is a model of a category and the least memory of a machine, in GB, it is
// recommended for there.
type CategoryPick struct {
	Machine int
	Model   string // its name in Catalog
}

// MemoryTag reads a tag of memory, N g or N gb (8g, 16GB): the model a machine of N GB gets.
func MemoryTag(tag string) (int, bool) {
	t := tagOf(tag)
	num, ok := strings.CutSuffix(t, "gb")
	if !ok {
		if num, ok = strings.CutSuffix(t, "g"); !ok {
			return 0, false
		}
	}
	n, err := strconv.Atoi(strings.TrimSpace(num)) // 8 g, 8 GB too
	return n, err == nil && n > 0
}

// tagOf is a tag as tags are compared: lower case, dots kept (0.8b and 08b are not the same).
func tagOf(s string) string { return strings.ToLower(strings.TrimSpace(s)) }

// TableVersion is the version of the definition files this hwnpu reads (models.yaml's version).
const TableVersion = 1

// LicenseEnv is the variable of the licenses accepted without asking: their ids, or all.
const LicenseEnv = "HWNPU_ACCEPT_LICENSE"
