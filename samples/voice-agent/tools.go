package main

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/shibukawa/bonsaiai/ai/agent"
)

// Tools are the functions the model may call: the time, timers, the weather (Open-Meteo), the
// news (an RSS feed, NHK's by default) and Wikipedia. What each returns is short Japanese text
// for the model to say in its own words. The ones that go to the network are Slow: the
// assistant says a word of waiting before they run.
type Tools struct {
	Timers *Timers
	// Place is the weather's place when a request names none.
	Place string
	// Feed is the news feed's URL, with %s for the category (NHK's cat0, cat1, ...).
	Feed string
	// Client is the HTTP client of the network tools.
	Client *http.Client
	// now is the clock (time.Now; a test's).
	now func() time.Time
	// places are the places found, so that a place is looked up once
	mu     sync.Mutex
	places map[string]place
}

// userAgent is what the tools tell the services they ask: Wikipedia asks a client to name
// itself.
const userAgent = "bonsaiai-voice-agent-sample/0.1 (https://github.com/shibukawa/bonsaiai)"

// List is the tools, for the agent.
func (t *Tools) List() []agent.Tool {
	now := time.Now
	if t.now != nil {
		now = t.now
	}
	return []agent.Tool{
		{Name: "get_time", Description: "今の日付、曜日、時刻を返す。",
			Run: func(ctx context.Context, a agent.Args) (string, error) {
				n := now()
				return fmt.Sprintf("%d年%s、%s", n.Year(), spokenDate(n), spokenTime(n)), nil
			}},
		{Name: "set_timer", Description: "タイマーをセットする。時間になると音で知らせる。",
			Parameters: `{"type": "object", "properties": {"seconds": {"type": "integer", "description": "秒数。3分なら180"}, "label": {"type": "string", "description": "何のタイマーか (なければ空)"}}, "required": ["seconds"]}`,
			Run: func(ctx context.Context, a agent.Args) (string, error) {
				secs := a.Int("seconds")
				if secs < 1 {
					return "seconds は 1 以上の整数で。", nil
				}
				return t.Timers.Set(time.Duration(secs)*time.Second, a.String("label")).Name() + "をセットしました。", nil
			}},
		{Name: "cancel_timer", Description: "タイマーを止める。label がなければ全部止める。",
			Parameters: `{"type": "object", "properties": {"label": {"type": "string"}}}`,
			Run: func(ctx context.Context, a agent.Args) (string, error) {
				stopped := t.Timers.Cancel(a.String("label"))
				if len(stopped) == 0 {
					return "止めるタイマーはありませんでした。", nil
				}
				return fmt.Sprintf("タイマーを%d個止めました。", len(stopped)), nil
			}},
		{Name: "list_timers", Description: "セットしてあるタイマーと残り時間を返す。",
			Run: func(ctx context.Context, a agent.Args) (string, error) { return t.Timers.Describe(), nil }},
		{Name: "get_weather", Description: "天気予報を返す。今の天気と今日、明日、明後日の予報。", Slow: true,
			Parameters: `{"type": "object", "properties": {"place": {"type": "string", "description": "地名 (市や都道府県)。言われなければ空"}}}`,
			Run:        func(ctx context.Context, a agent.Args) (string, error) { return t.weather(ctx, a.String("place")) }},
		{Name: "get_news", Description: "最新のニュースの見出しを返す。", Slow: true,
			Parameters: `{"type": "object", "properties": {"category": {"type": "string", "enum": ["主要", "社会", "科学", "政治", "経済", "国際", "スポーツ", "文化"]}}}`,
			Run:        func(ctx context.Context, a agent.Args) (string, error) { return t.news(ctx, a.String("category")) }},
		{Name: "search", Description: "Wikipedia で調べて、いちばん合う項目の要約を返す。人、場所、ものごと、出来事の事実を答える前に使う。", Slow: true,
			Parameters: `{"type": "object", "properties": {"query": {"type": "string", "description": "調べる言葉 (名詞)"}}, "required": ["query"]}`,
			Run:        func(ctx context.Context, a agent.Args) (string, error) { return t.search(ctx, a.String("query")) }},
	}
}

// get reads a URL's body (up to 2 MB).
func (t *Tools) get(ctx context.Context, u string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	c := t.Client
	if c == nil {
		c = http.DefaultClient
	}
	resp, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: %s", u, resp.Status)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 2<<20))
}

// place is a place found by Open-Meteo's geocoding.
type place struct {
	Name, Admin string
	Lat, Lon    float64
	Population  int
}

// locate finds a place. Open-Meteo's geocoding knows Japanese names with their suffix (大阪市,
// 東京都, 北海道) better than without (大阪 is not found), and has small places of the same name
// (横浜 is a town in Aomori), so the name is asked with each suffix and the most populous place
// is taken.
func (t *Tools) locate(ctx context.Context, name string) (place, error) {
	t.mu.Lock()
	if p, ok := t.places[name]; ok {
		t.mu.Unlock()
		return p, nil
	}
	t.mu.Unlock()
	names := []string{name}
	if last, _ := utf8.DecodeLastRuneInString(name); !strings.ContainsRune("市都府県区町村道", last) {
		names = append(names, name+"市", name+"都", name+"府", name+"県", name+"区")
	}
	var (
		wg   sync.WaitGroup
		mu   sync.Mutex
		best place
		errs []error
	)
	for _, n := range names {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := url.Values{"name": {n}, "count": {"5"}, "language": {"ja"}, "format": {"json"}}
			body, err := t.get(ctx, "https://geocoding-api.open-meteo.com/v1/search?"+q.Encode())
			if err != nil {
				mu.Lock()
				errs = append(errs, err)
				mu.Unlock()
				return
			}
			var r struct {
				Results []struct {
					Name       string  `json:"name"`
					Admin1     string  `json:"admin1"`
					Latitude   float64 `json:"latitude"`
					Longitude  float64 `json:"longitude"`
					Population int     `json:"population"`
				} `json:"results"`
			}
			if json.Unmarshal(body, &r) != nil {
				return
			}
			mu.Lock()
			defer mu.Unlock()
			for _, x := range r.Results {
				if best.Name == "" || x.Population > best.Population {
					best = place{x.Name, x.Admin1, x.Latitude, x.Longitude, x.Population}
				}
			}
		}()
	}
	wg.Wait()
	if best.Name == "" {
		if len(errs) == len(names) {
			return place{}, errs[0]
		}
		return place{}, fmt.Errorf("%s という場所が見つかりません", name)
	}
	t.mu.Lock()
	if t.places == nil {
		t.places = map[string]place{}
	}
	t.places[name] = best
	t.mu.Unlock()
	return best, nil
}

// weather is the weather of a place: now, and today, tomorrow and the day after.
func (t *Tools) weather(ctx context.Context, name string) (string, error) {
	if name == "" {
		name = t.Place
	}
	p, err := t.locate(ctx, name)
	if err != nil {
		return "", err
	}
	q := url.Values{
		"latitude": {fmt.Sprint(p.Lat)}, "longitude": {fmt.Sprint(p.Lon)}, "timezone": {"auto"}, "forecast_days": {"3"},
		"current": {"temperature_2m,weather_code"},
		"daily":   {"weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"},
	}
	body, err := t.get(ctx, "https://api.open-meteo.com/v1/forecast?"+q.Encode())
	if err != nil {
		return "", err
	}
	var r struct {
		Current struct {
			Temperature float64 `json:"temperature_2m"`
			Code        int     `json:"weather_code"`
		} `json:"current"`
		Daily struct {
			Code []int     `json:"weather_code"`
			Max  []float64 `json:"temperature_2m_max"`
			Min  []float64 `json:"temperature_2m_min"`
			Rain []float64 `json:"precipitation_probability_max"`
		} `json:"daily"`
	}
	if err := json.Unmarshal(body, &r); err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s。今は%s、%.0f度。", p.Name, weatherWord(r.Current.Code), r.Current.Temperature)
	for i, day := range []string{"今日", "明日", "明後日"} {
		if i >= len(r.Daily.Code) || i >= len(r.Daily.Max) || i >= len(r.Daily.Min) {
			break
		}
		fmt.Fprintf(&b, "%sは%s、最高%.0f度、最低%.0f度", day, weatherWord(r.Daily.Code[i]), r.Daily.Max[i], r.Daily.Min[i])
		if i < len(r.Daily.Rain) {
			fmt.Fprintf(&b, "、降水確率%.0f%%", r.Daily.Rain[i])
		}
		b.WriteString("。")
	}
	return b.String(), nil
}

// weatherWord is a WMO weather code in a word.
func weatherWord(code int) string {
	switch {
	case code == 0:
		return "晴れ"
	case code <= 2:
		return "晴れ時々曇り"
	case code == 3:
		return "曇り"
	case code <= 48:
		return "霧"
	case code <= 57:
		return "霧雨"
	case code <= 67:
		return "雨"
	case code <= 77:
		return "雪"
	case code <= 82:
		return "にわか雨"
	case code <= 86:
		return "にわか雪"
	}
	return "雷雨"
}

// newsCategories are NHK's feeds by category.
var newsCategories = map[string]string{"主要": "cat0", "社会": "cat1", "文化": "cat2", "科学": "cat3", "政治": "cat4", "経済": "cat5", "国際": "cat6", "スポーツ": "cat7"}

// news is the latest headlines of a category, the first sentence of each.
func (t *Tools) news(ctx context.Context, category string) (string, error) {
	cat, ok := newsCategories[category]
	if !ok {
		cat = "cat0"
	}
	u := t.Feed
	if strings.Contains(u, "%s") {
		u = fmt.Sprintf(u, cat)
	}
	body, err := t.get(ctx, u)
	if err != nil {
		return "", err
	}
	var rss struct {
		Items []struct {
			Title       string `xml:"title"`
			Description string `xml:"description"`
		} `xml:"channel>item"`
	}
	if err := xml.Unmarshal(body, &rss); err != nil {
		return "", err
	}
	if len(rss.Items) == 0 {
		return "", errors.New("ニュースがありません")
	}
	var b strings.Builder
	for i, it := range rss.Items[:min(5, len(rss.Items))] {
		fmt.Fprintf(&b, "%d. %s: %s\n", i+1, strings.TrimSpace(it.Title), firstSentence(it.Description, 80))
	}
	return b.String(), nil
}

// search is the summary of Wikipedia's article that best matches the query.
func (t *Tools) search(ctx context.Context, query string) (string, error) {
	if query == "" {
		return "", errors.New("query が空です")
	}
	q := url.Values{"action": {"query"}, "list": {"search"}, "srsearch": {query}, "srlimit": {"1"}, "format": {"json"}, "utf8": {"1"}}
	body, err := t.get(ctx, "https://ja.wikipedia.org/w/api.php?"+q.Encode())
	if err != nil {
		return "", err
	}
	var r struct {
		Query struct {
			Search []struct {
				Title string `json:"title"`
			} `json:"search"`
		} `json:"query"`
	}
	if err := json.Unmarshal(body, &r); err != nil {
		return "", err
	}
	if len(r.Query.Search) == 0 {
		return "", fmt.Errorf("%s は Wikipedia に見つかりません", query)
	}
	title := r.Query.Search[0].Title
	body, err = t.get(ctx, "https://ja.wikipedia.org/api/rest_v1/page/summary/"+url.PathEscape(title))
	if err != nil {
		return "", err
	}
	var s struct {
		Extract string `json:"extract"`
	}
	if err := json.Unmarshal(body, &s); err != nil {
		return "", err
	}
	return title + ": " + cut(s.Extract, 400), nil
}

var tags = regexp.MustCompile(`<[^>]*>`)

// firstSentence is a text's first sentence (its tags left out), at most n letters.
func firstSentence(s string, n int) string {
	s = strings.TrimSpace(tags.ReplaceAllString(s, ""))
	if i := strings.Index(s, "。"); i >= 0 {
		s = s[:i+len("。")]
	}
	return cut(s, n)
}

// cut is s cut to n letters.
func cut(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	return string([]rune(s)[:n]) + "…"
}
