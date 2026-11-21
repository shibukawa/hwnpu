package chat

import (
	"fmt"
	"os"
	"slices"
	"strings"
)

// The model policy (docs/model-provenance.md 2): which models may be picked or run, by their
// license and what they may make (content). Rules come from the policy of the user's definition
// files and from PolicyEnv, items separated by commas:
//
//	HWNPU_MODEL_ALLOWLIST=-license:non-commercial,+nsfw
//
// + allows and - refuses; a - wins over a +. A +license makes the licenses a list of what is
// allowed: a model none of its +s match is refused, as is one of no license known. Models of
// content nsfw are refused unless +nsfw.

// PolicyEnv is the environment's model policy (and a .env file's).
const PolicyEnv = "HWNPU_MODEL_ALLOWLIST"

// LicenseClasses are a license's classes (License.Class): what it allows of commercial use.
var LicenseClasses = []string{
	"open",           // allowed (apache-2.0, mit, cc-by-4.0)
	"conditional",    // allowed under conditions (a cap of revenue, a registration)
	"non-commercial", // the weights are not for commercial use (even when what they make is)
}

// Contents are what a model's content may say (Model.Content).
var Contents = []string{
	"nsfw", // makes NSFW pictures without being asked (Anima, even with "safe" in the prompt)
}

// policyKeys are the keys of a rule, with whether they take a value.
var policyKeys = map[string]bool{"license": true, "nsfw": false}

// Rule is an item of the policy.
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

// Policy are the rules of the model policy, set by LoadModels.
var Policy []Rule

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

// PolicyError is the error of a model the policy refuses.
type PolicyError struct {
	Model string
	Why   string // what refuses it
}

func (e *PolicyError) Error() string {
	return fmt.Sprintf("chat: %s is not allowed by the model policy (%s, policy of models.yaml): %s", e.Model, PolicyEnv, e.Why)
}

// Allowed is nil when the policy allows m, else a *PolicyError that says why.
func Allowed(m Model) error {
	if why := refused(Policy, m); why != "" {
		return &PolicyError{Model: m.Name, Why: why}
	}
	return nil
}

// refused is why rules refuse m, "" when they allow it.
func refused(rules []Rule, m Model) string {
	of := func(key string) (allow, deny []Rule) {
		for _, r := range rules {
			if r.Key != key {
				continue
			}
			if r.Allow {
				allow = append(allow, r)
			} else {
				deny = append(deny, r)
			}
		}
		return allow, deny
	}
	// nsfw: refused unless allowed, and a - wins
	if slices.Contains(m.Content, "nsfw") {
		allow, deny := of("nsfw")
		if len(deny) > 0 {
			return fmt.Sprintf("it may make NSFW pictures unasked (%s in %s)", deny[0], deny[0].From)
		}
		if len(allow) == 0 {
			return fmt.Sprintf("it may make NSFW pictures unasked; +nsfw in %s allows it", PolicyEnv)
		}
	}
	allow, deny := of("license")
	values := slices.DeleteFunc([]string{strings.ToLower(m.License.ID), m.License.Class}, func(v string) bool { return v == "" })
	for _, r := range deny {
		if slices.Contains(values, r.Value) {
			return fmt.Sprintf("its license is %s (%s in %s)", licenseOf(m.License), r, r.From)
		}
	}
	if len(allow) > 0 && !slices.ContainsFunc(allow, func(r Rule) bool { return slices.Contains(values, r.Value) }) {
		if m.License.ID == "" {
			return fmt.Sprintf("its license is not known (%s)", joinRules(allow))
		}
		return fmt.Sprintf("its license is %s, not of those allowed (%s)", licenseOf(m.License), joinRules(allow))
	}
	return ""
}

func licenseOf(l License) string {
	if l.Class == "" {
		return l.ID
	}
	return l.ID + ", " + l.Class
}

func joinRules(rs []Rule) string {
	var s []string
	for _, r := range rs {
		s = append(s, r.String())
	}
	return strings.Join(s, ",")
}

// policyFromEnv is the rules of PolicyEnv.
func policyFromEnv() ([]Rule, error) {
	return ParsePolicy(strings.Split(os.Getenv(PolicyEnv), ","), PolicyEnv)
}

// AllowedAlso is Allowed with a client's own policy too (its PolicyEnv, items separated by
// commas): hwnpud runs a model that its own policy allows and so does the client's, the rules of
// the definition files with the client's items. A client narrows hwnpud's policy, never widens
// it. items "" is no policy of the client's.
func AllowedAlso(m Model, items string) error {
	if err := Allowed(m); err != nil {
		return err
	}
	if strings.TrimSpace(items) == "" {
		return nil
	}
	theirs, err := ParsePolicy(strings.Split(items, ","), "the client's "+PolicyEnv)
	if err != nil {
		return err
	}
	rules := slices.DeleteFunc(slices.Clone(Policy), func(r Rule) bool { return r.From == PolicyEnv })
	if why := refused(append(rules, theirs...), m); why != "" {
		return &PolicyError{Model: m.Name, Why: why}
	}
	return nil
}
