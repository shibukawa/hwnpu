package catalog

import (
	"fmt"
	"strings"
)

// policyKeys are the keys of a rule of the model policy (internal/chat's), with whether they take
// a value.
var policyKeys = map[string]bool{"license": true, "nsfw": false}

// Rule is an item of the model policy.
type Rule struct {
	Allow bool   // + (else -)
	Key   string // a key of policyKeys
	Value string // in lower case
	From  string // where it was written (PolicyEnv, a definition file)
}

func (r Rule) String() string {
	s := "-"
	if r.Allow {
		s = "+"
	}
	s += r.Key
	if r.Value != "" {
		s += ":" + r.Value
	}
	return s
}

// ParsePolicy reads the items of a policy written in from (+license:ID, -license:CLASS, +nsfw).
func ParsePolicy(items []string, from string) ([]Rule, error) {
	var rules []Rule
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		bad := func(why string) error {
			return fmt.Errorf("chat: %s: %q %s (+license:ID, -license:CLASS or +nsfw: docs/model-provenance.md 2)", from, item, why)
		}
		var r Rule
		switch item[0] {
		case '+':
			r.Allow = true
		case '-':
		default:
			return nil, bad("starts with neither + nor -")
		}
		key, value, valued := strings.Cut(strings.ToLower(item[1:]), ":")
		r.Key, r.Value, r.From = key, value, from
		takes, ok := policyKeys[key]
		switch {
		case !ok:
			return nil, bad("has no key hwnpu knows")
		case !takes && valued:
			return nil, bad("takes no value")
		case takes && value == "":
			return nil, bad("has no value")
		}
		rules = append(rules, r)
	}
	return rules, nil
}
