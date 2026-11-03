package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// Result is one hit of a search.
type Result struct {
	Title, URL, Snippet string
}

// Searcher is the search tool's backend: Wikipedia, SearXNG or Brave.
type Searcher interface {
	// Search returns up to n results for the query, the best first.
	Search(ctx context.Context, query string, n int) ([]Result, error)
	// About says what is searched, for the tool's declaration: "the web", "Wikipedia".
	About() string
}

// Format writes results as the model reads them: numbered, the title, the URL and the
// snippet of each.
func Format(results []Result) string {
	var sb strings.Builder
	for i, r := range results {
		if i > 0 {
			sb.WriteString("\n\n")
		}
		fmt.Fprintf(&sb, "%d. %s\n%s", i+1, strings.TrimSpace(r.Title), readable(r.URL))
		if s := clean(r.Snippet); s != "" {
			sb.WriteString("\n" + s)
		}
	}
	return sb.String()
}

// maxSnippet is the most characters of a snippet the model is given.
const maxSnippet = 400

// clean strips the tags of a snippet, joins its lines and cuts it at maxSnippet.
func clean(s string) string {
	s = html.UnescapeString(tagRe.ReplaceAllString(s, ""))
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > maxSnippet {
		s = string(r[:maxSnippet]) + "…"
	}
	return s
}

var tagRe = regexp.MustCompile(`<[^>]*>`)

// userAgent names the agent to the sites it asks, as Wikipedia's policy wants.
const userAgent = "hwnpu-agent (https://github.com/shibukawa/hwnpu)"

// get fetches a URL with the query and decodes its JSON answer into v.
func get(ctx context.Context, client *http.Client, u string, q url.Values, header http.Header, v any) error {
	if client == nil {
		client = &http.Client{Timeout: 20 * time.Second}
	}
	if len(q) > 0 {
		sep := "?"
		if strings.Contains(u, "?") {
			sep = "&"
		}
		u += sep + q.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Accept", "application/json")
	for k, vs := range header {
		req.Header[k] = vs
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s", req.URL.Host, resp.Status)
	}
	if err := json.Unmarshal(body, v); err != nil {
		return fmt.Errorf("%s: %w", req.URL.Host, err)
	}
	return nil
}

// Wikipedia searches Wikipedia's articles (the MediaWiki API, no key): the hits of the search
// with the opening sentences of each article. It knows named things (富士山, 信濃川, a novel,
// a person) and little else: a question in words finds lists and the words' own articles.
type Wikipedia struct {
	// Lang is the edition, "ja" or "en"; "" picks ja for a query with kana or kanji, and en
	// otherwise.
	Lang string
	// URL is the api.php to ask instead of the edition's (tests).
	URL    string
	Client *http.Client
}

func (w *Wikipedia) About() string {
	return "Wikipedia, whose articles are found by the names of their subjects (a mountain, a river, a novel, a person)"
}

func (w *Wikipedia) Search(ctx context.Context, query string, n int) ([]Result, error) {
	// the model quotes a title (『こころ』 作者), which the search takes literally
	query = strings.Trim(strings.Map(func(r rune) rune {
		if strings.ContainsRune("『』「」\"'“”", r) {
			return ' '
		}
		return r
	}, query), " ")
	query = strings.Join(strings.Fields(query), " ")
	out, err := w.search(ctx, query, n)
	if err == nil && len(out) == 0 {
		// nothing for the words together: the first is the name, often
		if first, rest, ok := strings.Cut(query, " "); ok && rest != "" {
			out, err = w.search(ctx, first, n)
		}
	}
	return out, err
}

func (w *Wikipedia) search(ctx context.Context, query string, n int) ([]Result, error) {
	lang := w.Lang
	if lang == "" {
		lang = "en"
		if japanese(query) {
			lang = "ja"
		}
	}
	api := w.URL
	if api == "" {
		api = "https://" + lang + ".wikipedia.org/w/api.php"
	}
	var hits struct {
		Query struct {
			Search []struct {
				Title   string `json:"title"`
				PageID  int    `json:"pageid"`
				Snippet string `json:"snippet"`
			} `json:"search"`
		} `json:"query"`
	}
	q := url.Values{"action": {"query"}, "list": {"search"}, "srsearch": {query}, "srlimit": {strconv.Itoa(n)}, "format": {"json"}, "utf8": {"1"}}
	if err := get(ctx, w.Client, api, q, nil, &hits); err != nil {
		return nil, err
	}
	var out []Result
	var ids []string
	for _, h := range hits.Query.Search {
		// the title as it is, not percent-encoded: the model copies the URL into its answer,
		// and an encoded one comes out mangled
		out = append(out, Result{Title: h.Title, URL: "https://" + lang + ".wikipedia.org/wiki/" + strings.ReplaceAll(h.Title, " ", "_"), Snippet: h.Snippet})
		ids = append(ids, strconv.Itoa(h.PageID))
	}
	if len(out) == 0 {
		return nil, nil
	}
	// the opening of each article says what it is (the height of a mountain, the author of
	// a novel), where the search's snippet shows where the words matched
	var pages struct {
		Query struct {
			Pages map[string]struct {
				Extract string `json:"extract"`
			} `json:"pages"`
		} `json:"query"`
	}
	q = url.Values{"action": {"query"}, "prop": {"extracts"}, "exintro": {"1"}, "explaintext": {"1"}, "exsentences": {"2"}, "pageids": {strings.Join(ids, "|")}, "format": {"json"}, "utf8": {"1"}}
	if err := get(ctx, w.Client, api, q, nil, &pages); err == nil {
		for i, id := range ids {
			if p, ok := pages.Query.Pages[id]; ok && strings.TrimSpace(p.Extract) != "" {
				out[i].Snippet = p.Extract
			}
		}
	}
	return out, nil
}

// japanese reports whether the text has kana or kanji.
func japanese(s string) bool {
	for _, r := range s {
		if unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r) || unicode.Is(unicode.Han, r) {
			return true
		}
	}
	return false
}

// SearXNG searches the web through a SearXNG instance (searxng.org), such as one run at
// home: docker run -p 8888:8080 searxng/searxng, with json among search.formats in its
// settings.yml.
type SearXNG struct {
	URL    string // the instance, such as http://localhost:8888
	Client *http.Client
}

func (s *SearXNG) About() string { return "the web" }

func (s *SearXNG) Search(ctx context.Context, query string, n int) ([]Result, error) {
	var res struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	q := url.Values{"q": {query}, "format": {"json"}}
	if err := get(ctx, s.Client, strings.TrimRight(s.URL, "/")+"/search", q, nil, &res); err != nil {
		return nil, err
	}
	var out []Result
	for _, r := range res.Results {
		if len(out) == n {
			break
		}
		out = append(out, Result{Title: r.Title, URL: r.URL, Snippet: r.Content})
	}
	return out, nil
}

// Brave searches the web with the Brave Search API (api.search.brave.com), which takes a
// subscription key.
type Brave struct {
	Key    string
	URL    string // the endpoint to ask instead of Brave's (tests)
	Client *http.Client
}

func (b *Brave) About() string { return "the web" }

func (b *Brave) Search(ctx context.Context, query string, n int) ([]Result, error) {
	if b.Key == "" {
		return nil, errors.New("brave: no API key")
	}
	api := b.URL
	if api == "" {
		api = "https://api.search.brave.com/res/v1/web/search"
	}
	var res struct {
		Web struct {
			Results []struct {
				Title       string `json:"title"`
				URL         string `json:"url"`
				Description string `json:"description"`
			} `json:"results"`
		} `json:"web"`
	}
	q := url.Values{"q": {query}, "count": {strconv.Itoa(n)}}
	if err := get(ctx, b.Client, api, q, http.Header{"X-Subscription-Token": {b.Key}}, &res); err != nil {
		return nil, err
	}
	var out []Result
	for _, r := range res.Web.Results {
		out = append(out, Result{Title: r.Title, URL: r.URL, Snippet: r.Description})
	}
	return out, nil
}

// NewSearcher picks the backend -search names: "wikipedia", "brave" (the key is
// BRAVE_API_KEY), or the URL of a SearXNG instance.
func NewSearcher(name, braveKey string) (Searcher, error) {
	switch {
	case name == "" || strings.EqualFold(name, "wikipedia"):
		return &Wikipedia{}, nil
	case strings.EqualFold(name, "brave"):
		if braveKey == "" {
			return nil, errors.New("-search brave needs the key in BRAVE_API_KEY")
		}
		return &Brave{Key: braveKey}, nil
	case strings.HasPrefix(name, "http://") || strings.HasPrefix(name, "https://"):
		return &SearXNG{URL: name}, nil
	}
	return nil, fmt.Errorf("-search %q: wikipedia, brave, or the URL of a SearXNG instance", name)
}

// readable decodes the percent-encoding of a URL's path when it decodes to text, since the
// model copies a URL into a fetch call or an answer, and gets a long encoded one wrong
// (%E5%AD%A2%E6%BD+ad%3E%3D for a Wikipedia title). Go's client encodes it again on fetch. A
// path that decodes to a space or a delimiter stays encoded.
func readable(u string) string {
	if !strings.Contains(u, "%") {
		return u
	}
	p, err := url.Parse(u)
	if err != nil {
		return u
	}
	path, err := url.PathUnescape(p.EscapedPath())
	if err != nil || !utf8.ValidString(path) || strings.ContainsAny(path, " ?#\x00") {
		return u
	}
	s := p.Scheme + "://" + p.Host + path
	if p.RawQuery != "" {
		s += "?" + p.RawQuery
	}
	if p.Fragment != "" {
		s += "#" + p.EscapedFragment()
	}
	return s
}
