package envfile

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
)

// Key is a setting by the name bonsaiai config gives it (model-allowlist), and the variable of the
// .env files that holds it (BONSAIAI_MODEL_ALLOWLIST): the files keep the variables, which every
// process reads, and the users use the keys.
type Key struct {
	Name string
	Env  string
	// Sep separates the items of a list (model-allowlist: ","); "" for a single value.
	Sep string
	// Sign is the sign of a list's items in the variable that two keys share: model-allowlist's
	// are its +items, model-denylist's its -items (the policy's +nsfw, -license:non-commercial).
	// The key's items are written without it.
	Sign string
	// Numbered keeps a list's items in variables of their own, in order: Env, Env_1, Env_2, ...
	// (the sources: model-source).
	Numbered bool
	Help     string
}

// List says whether the key holds a list of items.
func (k Key) List() bool { return k.Sep != "" || k.Numbered }

// Owns says whether the variable env is (one of) the key's.
func (k Key) Owns(env string) bool {
	if env == k.Env {
		return true
	}
	_, ok := k.number(env)
	return ok
}

// number is the number of a Numbered key's variable Env_N.
func (k Key) number(env string) (int, bool) {
	if !k.Numbered {
		return 0, false
	}
	rest, ok := strings.CutPrefix(env, k.Env+"_")
	if !ok {
		return 0, false
	}
	n, err := strconv.Atoi(rest)
	return n, err == nil && n >= 0
}

// Get is the key's items in the variables vars (a single value's is one item): set is false when
// it has none.
func (k Key) Get(vars map[string]string) (items []string, set bool) {
	switch {
	case k.Numbered:
		type num struct {
			n int
			v string
		}
		var ns []num
		for name, v := range vars {
			if strings.TrimSpace(v) == "" {
				continue
			}
			if name == k.Env {
				ns = append(ns, num{-1, v})
			} else if n, ok := k.number(name); ok {
				ns = append(ns, num{n, v})
			}
		}
		slices.SortFunc(ns, func(a, b num) int { return a.n - b.n })
		for _, n := range ns {
			items = append(items, strings.TrimSpace(n.v))
		}
	case k.List():
		items = k.Items(vars[k.Env])
	default:
		if v, ok := vars[k.Env]; ok {
			items = []string{v}
		}
	}
	return items, len(items) > 0
}

// GetIn is the key's items in the file p.
func GetIn(p string, k Key) (items []string, set bool, err error) {
	vars, _, err := Read(p)
	m := map[string]string{}
	for _, v := range vars {
		if _, seen := m[v.Name]; !seen {
			m[v.Name] = v.Value
		}
	}
	items, set = k.Get(m)
	return items, set, err
}

// SetIn writes the key's items to the file p (none removes the key): a single value's line, a
// list's variable with the items of the other sign kept, a Numbered key's variables numbered again
// from Env, where the first of them was.
func SetIn(p string, k Key, items []string) error {
	switch {
	case k.Numbered:
		var lines []string
		for i, it := range items {
			name := k.Env
			if i > 0 {
				name += "_" + strconv.Itoa(i)
			}
			lines = append(lines, name+"="+quote(it))
		}
		return rewrite(p, func(old []string) []string {
			var out []string
			done := false
			for _, l := range old {
				if name, _, ok, _ := parseLine(l); ok && k.Owns(name) {
					if !done {
						out = append(out, lines...)
						done = true
					}
					continue
				}
				out = append(out, l)
			}
			if !done {
				out = append(out, lines...)
			}
			return out
		})
	case k.List():
		v, _, err := Lookup(p, k.Env)
		if err != nil {
			return err
		}
		if v = k.With(v, items); v != "" {
			return Set(p, k.Env, v)
		}
	case len(items) == 1:
		return Set(p, k.Env, items[0])
	case len(items) > 1:
		return fmt.Errorf("%s holds one value", k.Name)
	}
	_, err := Unset(p, k.Env)
	return err
}

// Items are the key's items of the variable's value v: those of its Sign, without it.
func (k Key) Items(v string) []string {
	var out []string
	for _, s := range k.all(v) {
		if k.Sign == "" {
			out = append(out, s)
		} else if t, ok := strings.CutPrefix(s, k.Sign); ok {
			out = append(out, t)
		}
	}
	return out
}

// With is the variable's value v with the key's items replaced by items: the items of the other
// sign are kept, before them.
func (k Key) With(v string, items []string) string {
	var out []string
	if k.Sign != "" {
		for _, s := range k.all(v) {
			if !strings.HasPrefix(s, k.Sign) {
				out = append(out, s)
			}
		}
	}
	for _, it := range items {
		out = append(out, k.Sign+it)
	}
	return strings.Join(out, k.Sep)
}

func (k Key) all(v string) []string {
	var out []string
	for _, s := range strings.Split(v, k.Sep) {
		if s = strings.TrimSpace(s); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// Keys are the settings bonsaiai config knows.
var Keys = []Key{
	{Name: "model-source", Env: "BONSAIAI_MODEL_SOURCE", Numbered: true, Help: "where the models are looked for before Hugging Face, in order: URLs (file:///..., https://USER@host/dav, s3://BUCKET/PREFIX; ?writeback=true puts there what is made here)"},
	{Name: "model-allowlist", Env: "BONSAIAI_MODEL_ALLOWLIST", Sep: ",", Sign: "+", Help: "what the model policy allows: nsfw (the models that may make NSFW pictures), license:open (only these licenses or classes), ... (docs/model-provenance.md)"},
	{Name: "model-denylist", Env: "BONSAIAI_MODEL_ALLOWLIST", Sep: ",", Sign: "-", Help: "what the model policy refuses: license:non-commercial, license:conditional, license:ID, ... (wins over model-allowlist)"},
	{Name: "model-hugging-face", Env: "BONSAIAI_MODEL_HUGGING_FACE", Help: "false: never fetch from Hugging Face"},
	{Name: "accept-license", Env: "BONSAIAI_ACCEPT_LICENSE", Sep: ",", Help: "licenses accepted without asking: their ids, or all"},
	{Name: "models", Env: "BONSAIAI_MODELS", Sep: string(os.PathListSeparator), Help: "more definition files of models (models.yaml)"},
	{Name: "cache", Env: "BONSAIAI_CACHE", Help: "where the models are kept (default: bonsaiai in the user cache directory)"},
	{Name: "cache-min-free", Env: "BONSAIAI_CACHE_MIN_FREE", Help: "the disk's space downloads leave free (10GB, 5%, 0)"},
	{Name: "cache-max-size", Env: "BONSAIAI_CACHE_MAX_SIZE", Help: "the most the cache holds (200GB)"},
	{Name: "memory", Env: "BONSAIAI_MEMORY", Help: "the machine's memory in GB that models are picked by"},
	{Name: "backend", Env: "BONSAIAI_BACKEND", Help: "the backend to run on"},
	{Name: "device", Env: "BONSAIAI_DEVICE", Help: "the device to run on"},
	{Name: "coreai-prebuilt", Env: "BONSAIAI_COREAI_PREBUILT", Help: "off: compile Core AI's programs on the device even when prebuilt ones are there"},
	{Name: "coreai-units", Env: "BONSAIAI_COREAI_UNITS", Help: "cpu: run Core AI's programs off the Neural Engine"},
	{Name: "daemon-idle", Env: "BONSAIAI_DAEMON_IDLE", Help: "how long bonsaiaid keeps the models without a request (5m; 0 always)"},
	{Name: "webdav-user", Env: "BONSAIAI_WEBDAV_USER", Help: "the WebDAV source's user"},
	{Name: "webdav-password", Env: "BONSAIAI_WEBDAV_PASSWORD", Help: "the WebDAV source's password"},
	{Name: "hf-token", Env: "HF_TOKEN", Help: "Hugging Face's token"},
	{Name: "hf-endpoint", Env: "HF_ENDPOINT", Help: "another Hugging Face endpoint (a mirror)"},
	{Name: "aws-profile", Env: "AWS_PROFILE", Help: "the AWS profile of an s3:// source"},
	{Name: "aws-access-key-id", Env: "AWS_ACCESS_KEY_ID", Help: "the AWS key of an s3:// source"},
	{Name: "aws-secret-access-key", Env: "AWS_SECRET_ACCESS_KEY", Help: "the AWS secret of an s3:// source"},
}

// KeyByName is the key of the name bonsaiai config gives it.
func KeyByName(name string) (Key, bool) {
	for _, k := range Keys {
		if k.Name == name {
			return k, true
		}
	}
	return Key{}, false
}

// KeyOf is the (first) key whose variable is env.
func KeyOf(env string) (Key, bool) {
	for _, k := range Keys {
		if k.Owns(env) {
			return k, true
		}
	}
	return Key{}, false
}

// KeysOf are the keys whose variable is env (model-allowlist and model-denylist share one).
func KeysOf(env string) []Key {
	var out []Key
	for _, k := range Keys {
		if k.Owns(env) {
			out = append(out, k)
		}
	}
	return out
}

// Display is the name a variable is shown by: its key's, else itself.
func Display(env string) string {
	if k, ok := KeyOf(env); ok {
		return k.Name
	}
	return env
}
