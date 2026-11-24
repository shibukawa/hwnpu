// Package prep is what bonsaiai-dev models package makes a model with, for the kinds the library's
// roles (airole: ai/text, ai/speech, ai/vision, ai/imagegen) do not make, or make in part: each
// preparer makes every program a command of the kind would convert for the target on any machine
// (every size, every form), as its first run there does (the same functions, the same names, so
// that the command finds them in the sources). No device is opened.
package prep

import (
	"fmt"
	"slices"

	"github.com/shibukawa/bonsaiai/internal/airole"
	"github.com/shibukawa/bonsaiai/internal/cache"
	"github.com/shibukawa/bonsaiai/internal/chat"
)

// Preparer makes the models of its kinds ready for its targets (airole.PrepareEnv: the NPU).
type Preparer struct {
	Name  string
	Kinds []chat.Kind
	// Targets are those it makes (cache.CoreAITarget, cache.AMDTarget); none: Core AI's alone.
	Targets []string
	Prepare func(airole.PrepareEnv) (airole.Files, error)
}

var preparers []*Preparer

// Register adds p; the first registered for a kind is For's.
func Register(p *Preparer) { preparers = append(preparers, p) }

// For is the preparer of model m's kind for target, nil when none is registered (the library's
// role then: airole.For).
func For(m chat.Model, target string) *Preparer {
	for _, p := range preparers {
		ts := p.Targets
		if len(ts) == 0 {
			ts = []string{cache.CoreAITarget}
		}
		if slices.Contains(p.Kinds, m.Kind) && slices.Contains(ts, target) {
			return p
		}
	}
	return nil
}

// notYet is the error of a target a preparer does not make yet.
func notYet(m chat.Model, target string) error {
	return fmt.Errorf("prep: %s (%s) for %s is not made by bonsaiai-dev models package yet", m.Name, m.Kind, target)
}
