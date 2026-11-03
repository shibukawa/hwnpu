package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A Wikipedia search is two requests: the hits, then the openings of their articles, which
// replace the hits' snippets.
func TestWikipedia(t *testing.T) {
	var asked []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked = append(asked, r.URL.RawQuery)
		if r.Header.Get("User-Agent") != userAgent {
			t.Errorf("user agent %q", r.Header.Get("User-Agent"))
		}
		q := r.URL.Query()
		switch q.Get("list") + q.Get("prop") {
		case "search":
			if q.Get("srsearch") == "存在しない 語" {
				json.NewEncoder(w).Encode(map[string]any{"query": map[string]any{"search": []any{}}})
				return
			}
			if q.Get("srsearch") != "富士山 高さ" && q.Get("srsearch") != "存在しない" || q.Get("srlimit") != "2" {
				t.Errorf("search %v", q)
			}
			json.NewEncoder(w).Encode(map[string]any{"query": map[string]any{"search": []map[string]any{
				{"title": "富士山", "pageid": 5149, "snippet": `<span class="searchmatch">富士山</span>本宮浅間大社`},
				{"title": "高さ", "pageid": 156381, "snippet": "垂直方向の長さ"},
			}}})
		case "extracts":
			if q.Get("pageids") != "5149|156381" {
				t.Errorf("extracts %v", q)
			}
			json.NewEncoder(w).Encode(map[string]any{"query": map[string]any{"pages": map[string]any{
				"5149": map[string]any{"extract": "富士山（ふじさん）は、日本の活火山である。標高は3775.56 m。"},
			}}})
		default:
			t.Errorf("request %v", q)
		}
	}))
	defer srv.Close()
	wk := &Wikipedia{URL: srv.URL, Lang: "ja"}
	res, err := wk.Search(context.Background(), "『富士山』 高さ", 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []Result{
		{"富士山", "https://ja.wikipedia.org/wiki/富士山", "富士山（ふじさん）は、日本の活火山である。標高は3775.56 m。"},
		{"高さ", "https://ja.wikipedia.org/wiki/高さ", "垂直方向の長さ"},
	}
	if len(res) != 2 || res[0] != want[0] || res[1] != want[1] {
		t.Errorf("got %+v, want %+v", res, want)
	}
	if len(asked) != 2 {
		t.Errorf("%d requests: %v", len(asked), asked)
	}
	// nothing for the words together: the first word alone is tried
	if res, err := wk.Search(context.Background(), "存在しない 語", 2); err != nil || len(res) != 2 || len(asked) != 5 {
		t.Errorf("the retry: %d results, %v, %d requests", len(res), err, len(asked))
	}
	got := Format(res)
	if !strings.HasPrefix(got, "1. 富士山\nhttps://ja.wikipedia.org/wiki/富士山\n富士山（ふじさん）は") || !strings.Contains(got, "\n\n2. 高さ\n") {
		t.Errorf("formatted:\n%s", got)
	}
	for u, want := range map[string]string{
		"https://ja.wikipedia.org/wiki/%E5%AF%8C%E5%A3%AB%E5%B1%B1":      "https://ja.wikipedia.org/wiki/富士山",
		"https://example.com/a%20b?q=%E3%81%82":                          "https://example.com/a%20b?q=%E3%81%82",
		"https://ja.wikipedia.org/wiki/%E5%B7%9D?x=1#%E6%A6%82%E8%A6%81": "https://ja.wikipedia.org/wiki/川?x=1#%E6%A6%82%E8%A6%81",
		"https://example.com/plain":                                      "https://example.com/plain",
		"https://example.com/%FF%FE":                                     "https://example.com/%FF%FE",
	} {
		if got := readable(u); got != want {
			t.Errorf("readable(%q) = %q, want %q", u, got, want)
		}
	}
	for q, ja := range map[string]bool{"富士山 高さ": true, "longest river Japan": false, "ｶﾀｶﾅ": true, "": false} {
		if japanese(q) != ja {
			t.Errorf("japanese(%q) = %v", q, !ja)
		}
	}
	if s := clean(`a  <b>bold</b>&amp;<span class="x">\n</span> b`); s != "a bold&\\n b" {
		t.Errorf("clean: %q", s)
	}
	if s := clean(strings.Repeat("あ", maxSnippet+1)); len([]rune(s)) != maxSnippet+1 || !strings.HasSuffix(s, "…") {
		t.Errorf("clean did not cut: %d runes", len([]rune(s)))
	}
}

func TestSearXNGAndBrave(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		switch r.URL.Path {
		case "/search":
			if q.Get("q") != "日本一長い川" || q.Get("format") != "json" {
				t.Errorf("searxng %v", q)
			}
			json.NewEncoder(w).Encode(map[string]any{"results": []map[string]any{
				{"title": "信濃川 - Wikipedia", "url": "https://ja.wikipedia.org/wiki/信濃川", "content": "全長367 km"},
				{"title": "利根川", "url": "https://example.com/tone", "content": "322 km"},
				{"title": "石狩川", "url": "https://example.com/ishikari", "content": "268 km"},
			}})
		case "/brave":
			if r.Header.Get("X-Subscription-Token") != "k" || q.Get("q") != "日本一長い川" || q.Get("count") != "2" {
				t.Errorf("brave %v %v", r.Header, q)
			}
			json.NewEncoder(w).Encode(map[string]any{"web": map[string]any{"results": []map[string]any{
				{"title": "信濃川", "url": "https://example.com/shinano", "description": "367 km"},
			}}})
		case "/down/search":
			http.Error(w, "nope", http.StatusServiceUnavailable)
		}
	}))
	defer srv.Close()
	res, err := (&SearXNG{URL: srv.URL + "/"}).Search(context.Background(), "日本一長い川", 2)
	if err != nil || len(res) != 2 || res[0].Title != "信濃川 - Wikipedia" || res[0].Snippet != "全長367 km" || res[1].Title != "利根川" {
		t.Errorf("searxng: %+v, %v", res, err)
	}
	res, err = (&Brave{Key: "k", URL: srv.URL + "/brave"}).Search(context.Background(), "日本一長い川", 2)
	if err != nil || len(res) != 1 || res[0] != (Result{"信濃川", "https://example.com/shinano", "367 km"}) {
		t.Errorf("brave: %+v, %v", res, err)
	}
	if _, err := (&Brave{URL: srv.URL + "/brave"}).Search(context.Background(), "x", 1); err == nil {
		t.Error("brave without a key searched")
	}
	if _, err := (&SearXNG{URL: srv.URL + "/down"}).Search(context.Background(), "x", 1); err == nil || !strings.Contains(err.Error(), "503") {
		t.Errorf("a server error: %v", err)
	}
	for _, c := range []struct {
		name, key string
		want      string
		fails     bool
	}{
		{"", "", "*agent.Wikipedia", false}, {"Wikipedia", "", "*agent.Wikipedia", false},
		{"brave", "k", "*agent.Brave", false}, {"brave", "", "", true},
		{"http://localhost:8888", "", "*agent.SearXNG", false}, {"bing", "", "", true},
	} {
		s, err := NewSearcher(c.name, c.key)
		if (err != nil) != c.fails || err == nil && typeName(s) != c.want {
			t.Errorf("NewSearcher(%q): %T, %v", c.name, s, err)
		}
	}
}

func typeName(v any) string {
	switch v.(type) {
	case *Wikipedia:
		return "*agent.Wikipedia"
	case *Brave:
		return "*agent.Brave"
	case *SearXNG:
		return "*agent.SearXNG"
	}
	return ""
}
