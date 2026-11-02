package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestFetch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/page":
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Write([]byte(`<html><head><title>t</title><style>p{}</style></head><body>
<script>var x = "<p>no</p>";</script><h1>信濃川</h1><!-- c -->
<p>信濃川（しなのがわ）は、<b>全長367 km</b>で日本一長い川。</p>

<div><ul><li>長野県</li><li>新潟県</li></ul></div><p>&amp; &lt;end&gt;</p></body></html>`))
		case "/text":
			w.Header().Set("Content-Type", "text/plain")
			w.Write([]byte(strings.Repeat("あ", 100)))
		case "/bin":
			w.Write([]byte{0xff, 0xfe, 0x00, 0x01})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	got, err := Fetch(context.Background(), nil, srv.URL+"/page", 6000)
	if err != nil || got != "信濃川\n信濃川（しなのがわ）は、全長367 kmで日本一長い川。\n長野県\n新潟県\n& <end>" {
		t.Errorf("page: %q, %v", got, err)
	}
	got, err = Fetch(context.Background(), nil, srv.URL+"/text", 60)
	if err != nil || got != strings.Repeat("あ", 60)+"\n[... 40 more characters omitted]" {
		t.Errorf("text: %q, %v", got, err)
	}
	if _, err := Fetch(context.Background(), nil, srv.URL+"/bin", 60); err == nil {
		t.Error("a binary page was returned")
	}
	if _, err := Fetch(context.Background(), nil, srv.URL+"/none", 60); err == nil || !strings.Contains(err.Error(), "404") {
		t.Errorf("404: %v", err)
	}
	c := Config{FetchChars: 60}
	if got := c.fetch(context.Background(), map[string]any{"url": "wiki/x"}); !strings.HasPrefix(got, "url must begin") {
		t.Errorf("a relative url: %q", got)
	}
	if got := c.fetch(context.Background(), map[string]any{"url": srv.URL + "/none"}); !strings.HasPrefix(got, "The page could not be read: ") {
		t.Errorf("a missing page: %q", got)
	}
	if got := c.fetch(context.Background(), map[string]any{"url": srv.URL + "/page"}); !strings.HasPrefix(got, "信濃川\n") {
		t.Errorf("the tool: %q", got)
	}
	// with search and fetch only, the prompt is the assistant's
	s, err := New(&script{}, Config{Dir: t.TempDir(), Tools: []string{"search", "fetch"}, Search: &fakeSearch{}})
	if err != nil || !strings.HasPrefix(s.System(), SearchSystem) || !strings.Contains(s.System(), "Use fetch") || len(s.Tools()) != 2 {
		t.Errorf("the prompt: %v\n%s", err, s.System())
	}
}
