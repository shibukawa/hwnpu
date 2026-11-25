package daemon

import (
	"fmt"
	"net/url"
	"os"
	"slices"
	"sort"
	"strings"

	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/internal/envfile"
)

// The settings of where the models are and where they come from are bonsaiaid's: one cache that
// every client shares, one set of sources. bonsaiaid reads them from the .env files
// (~/.config/bonsaiai/.env, the machine's) and from the environment it was started with, never
// from a client's: a client of other settings would make a second copy of the models, or fetch
// them from elsewhere. What is a client's to choose (the model, its memory, the model policy's
// narrowing: LoadRequest) goes with each request.

// settingNames are the variables of bonsaiaid's side that Settings reports and a client compares
// with its own.
var settingNames = []string{
	"BONSAIAI_CACHE",
	"BONSAIAI_CACHE_MIN_FREE",
	"BONSAIAI_CACHE_MAX_SIZE",
	"BONSAIAI_MODEL_HUGGING_FACE",
	"BONSAIAI_MODELS",
	"BONSAIAI_BACKEND",
	"BONSAIAI_DEVICE",
	"BONSAIAI_MEMORY",
	"BONSAIAI_ACCEPT_LICENSE",
	"BONSAIAI_COREAI_PREBUILT",
	"BONSAIAI_COREAI_UNITS",
	"BONSAIAI_WEBDAV_USER",
	"HF_ENDPOINT",
	chat.PolicyEnv,
}

// clientNames are the settings of settingNames a client may have of its own: the model policy.
// bonsaiaid's is its own, from the .env files; a client's goes with each request and narrows it
// (LoadRequest.Allowlist), so it is reported but not compared.
var clientNames = []string{chat.PolicyEnv}

// settingPrefixes are beginnings of the names of such variables (BONSAIAI_MODEL_SOURCE_1, ...).
var settingPrefixes = []string{"BONSAIAI_MODEL_SOURCE"}

// secretNames are the credentials among them: reported as set or not, never their values.
var secretNames = []string{"BONSAIAI_WEBDAV_PASSWORD", "HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"}

// Secret is the value Settings gives a credential that is set.
const Secret = "(set)"

// Settings are bonsaiaid's settings in env (KEY=VALUE): the variables of settingNames and
// settingPrefixes, with the passwords of URLs taken out, and secretNames as Secret.
func Settings(env []string) map[string]string {
	out := map[string]string{}
	for _, kv := range env {
		k, v, _ := strings.Cut(kv, "=")
		switch {
		case slices.Contains(secretNames, k):
			if v != "" {
				out[k] = Secret
			}
		case slices.Contains(settingNames, k):
			out[k] = v
		case slices.ContainsFunc(settingPrefixes, func(p string) bool { return strings.HasPrefix(k, p) }):
			out[k] = redactURL(v)
		}
	}
	return out
}

// Shown is how a variable's value is shown: Secret for a credential, a URL without its password.
func Shown(k, v string) string {
	if slices.Contains(secretNames, k) {
		return Secret
	}
	return redactURL(v)
}

func redactURL(v string) string {
	u, err := url.Parse(v)
	if err != nil || u.User == nil {
		return v
	}
	return u.Redacted()
}

// Mismatch is a setting of a client that bonsaiaid does not have: bonsaiaid's is used.
type Mismatch struct {
	Name  string
	Here  string // "" unset
	There string // bonsaiaid's, "" unset
}

func (m Mismatch) String() string {
	show := func(v string) string {
		if v == "" {
			return "unset"
		}
		return v
	}
	return fmt.Sprintf("%s is %s here, %s in bonsaiaid", m.Name, show(m.Here), show(m.There))
}

// Compare is how the settings here differ from bonsaiaid's, by name: those a client may have of
// its own (the model policy) are not compared.
func Compare(here, there map[string]string) []Mismatch {
	var out []Mismatch
	for _, k := range union(here, there) {
		if here[k] != there[k] && !slices.Contains(clientNames, k) {
			out = append(out, Mismatch{Name: k, Here: here[k], There: there[k]})
		}
	}
	return out
}

func union(a, b map[string]string) []string {
	var keys []string
	for k := range a {
		keys = append(keys, k)
	}
	for k := range b {
		if _, ok := a[k]; !ok {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	return keys
}

// watched are the variables of env an engine is started again for when they change: all of
// bonsaiai's, Hugging Face's and AWS's (envfile.Prefixes), not only those Settings reports, but not
// bonsaiaid's own (BONSAIAI_DAEMON*), which the supervisor reads when it starts. The values stay in
// the supervisor: only the names are told.
func watched(env []string) map[string]string {
	out := map[string]string{}
	for _, kv := range env {
		k, v, _ := strings.Cut(kv, "=")
		if strings.HasPrefix(k, "BONSAIAI_DAEMON") {
			continue
		}
		if slices.ContainsFunc(envfile.Prefixes, func(p string) bool { return strings.HasPrefix(k, p) }) {
			out[k] = v
		}
	}
	return out
}

// changed are the names whose values differ between two sets of variables.
func changed(a, b map[string]string) []string {
	var out []string
	for _, k := range union(a, b) {
		va, oka := a[k]
		vb, okb := b[k]
		if va != vb || oka != okb {
			out = append(out, k)
		}
	}
	return out
}

// localSettings are this process's settings (its environment and the .env files it read).
func localSettings() map[string]string { return Settings(os.Environ()) }
