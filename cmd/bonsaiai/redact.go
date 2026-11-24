package main

import (
	"net/url"
	"os"
	"os/user"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	shared "github.com/shibukawa/bonsaiai/internal/store"
)

// redactor hides in an issue what tells who and where the person is, and secrets: the home
// folder (~), the user's and the machine's names, the addresses of the sources (and any URL's
// host but the public ones), passwords in URLs, tokens and e-mail addresses.
type redactor struct {
	words [][2]string // replaced as they are, the longest first
	names [][2]string // replaced as whole words
}

var (
	urlUser   = regexp.MustCompile(`([A-Za-z][A-Za-z0-9+.-]*://)[^/\s@]+@`)
	urlHost   = regexp.MustCompile(`([A-Za-z][A-Za-z0-9+.-]*://)([^/\s?#"'<>]+)`)
	secrets   = regexp.MustCompile(`(?i)\b(token|password|passwd|secret|api[_-]?key|access[_-]?key)(["']?\s*[=:]\s*["']?)[^\s"'&]+`)
	hfToken   = regexp.MustCompile(`\bhf_[A-Za-z0-9]{8,}`)
	awsKey    = regexp.MustCompile(`\b(AKIA|ASIA)[0-9A-Z]{16}\b`)
	bearer    = regexp.MustCompile(`(?i)\bbearer\s+\S+`)
	email     = regexp.MustCompile(`[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+`)
	publicURL = []string{"huggingface.co", "hf.co", "github.com", "cdn-lfs.huggingface.co", "cas-bridge.xethub.hf.co", "pypi.org", "localhost", "127.0.0.1"}
)

func newRedactor() redactor {
	var r redactor
	if home, err := os.UserHomeDir(); err == nil && len(home) > 1 {
		r.words = append(r.words, [2]string{home, "~"}, [2]string{filepath.ToSlash(home), "~"})
		r.addName(filepath.Base(home), "USER")
	}
	if u, err := user.Current(); err == nil {
		name := u.Username
		if i := strings.LastIndexAny(name, `\`); i >= 0 {
			name = name[i+1:] // DOMAIN\user
		}
		r.addName(name, "USER")
		for _, n := range strings.Fields(u.Name) {
			r.addName(n, "USER")
		}
	}
	if h, err := os.Hostname(); err == nil {
		r.addName(h, "HOST")
		short, _, _ := strings.Cut(h, ".")
		r.addName(short, "HOST")
	}
	n := 0
	for _, kv := range os.Environ() {
		k, v, _ := strings.Cut(kv, "=")
		if k != shared.SourceEnv && !strings.HasPrefix(k, shared.SourceEnv+"_") || v == "" {
			continue
		}
		n++
		if u, err := url.Parse(v); err == nil && u.Host != "" {
			r.words = append(r.words, [2]string{u.Host, "source-host-" + strconv.Itoa(n)})
			r.addName(u.Hostname(), "source-host-"+strconv.Itoa(n))
		} else if u != nil && u.Path != "" {
			r.words = append(r.words, [2]string{u.Path, "SOURCE-FOLDER"})
		}
	}
	slices.SortFunc(r.words, func(a, b [2]string) int { return len(b[0]) - len(a[0]) })
	return r
}

// addName hides a name as a whole word, unless it is too short or too common to be one.
func (r *redactor) addName(name, as string) {
	if len(name) < 3 || slices.Contains([]string{"root", "user", "admin", "home", "localhost", "runner"}, strings.ToLower(name)) {
		return
	}
	r.names = append(r.names, [2]string{name, as})
}

// hide returns s with what r hides replaced.
func (r redactor) hide(s string) string {
	s = urlUser.ReplaceAllString(s, "${1}***@")
	for _, w := range r.words {
		s = strings.ReplaceAll(s, w[0], w[1])
	}
	for _, n := range r.names {
		re := regexp.MustCompile(`(?i)(^|[^A-Za-z0-9])` + regexp.QuoteMeta(n[0]) + `($|[^A-Za-z0-9])`)
		for re.MatchString(s) { // overlapping matches ("user user")
			s = re.ReplaceAllString(s, "${1}"+n[1]+"${2}")
		}
	}
	s = urlHost.ReplaceAllStringFunc(s, func(m string) string {
		sub := urlHost.FindStringSubmatch(m)
		host := strings.ToLower(sub[2])
		h := host
		if i := strings.LastIndex(h, ":"); i >= 0 && !strings.Contains(h[i:], "]") {
			h = h[:i]
		}
		if slices.Contains(publicURL, h) || strings.HasSuffix(h, ".huggingface.co") || strings.HasSuffix(h, ".hf.co") ||
			strings.HasPrefix(host, "source-host-") || host == "***@" {
			return m
		}
		return sub[1] + "HOST"
	})
	s = secrets.ReplaceAllString(s, "${1}${2}***")
	s = hfToken.ReplaceAllString(s, "hf_***")
	s = awsKey.ReplaceAllString(s, "AKIA***")
	s = bearer.ReplaceAllString(s, "Bearer ***")
	s = email.ReplaceAllString(s, "***@***")
	return s
}

// args are a command's arguments for an issue: a long text with spaces (a prompt) is replaced
// by its length.
func (r redactor) args(a []string) string {
	out := make([]string, len(a))
	for i, s := range a {
		switch n := len([]rune(s)); {
		case n > 40 && strings.ContainsAny(s, " \n") && !strings.HasPrefix(s, "-"):
			s = "<text of " + strconv.Itoa(n) + " characters>"
		case strings.ContainsAny(s, " \t\n\"'"):
			s = "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
		}
		out[i] = r.hide(s)
	}
	return strings.Join(out, " ")
}
