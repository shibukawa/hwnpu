package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/ai/agent"
)

func TestNormalizeNumbers(t *testing.T) {
	for in, want := range map[string]string{
		"三分":    "3分",
		"二十五秒":  "25秒",
		"一時間半":  "1時間半",
		"十分":    "10分",
		"百二十秒":  "120秒",
		"３分":    "3分",
		"一緒に一番": "一緒に一番",
	} {
		if got := normalizeNumbers(in); got != want {
			t.Errorf("normalizeNumbers(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestShortcuts(t *testing.T) {
	ts := NewTimers()
	defer ts.Cancel("")
	now := time.Date(2026, 11, 22, 15, 4, 0, 0, time.Local)
	s := &Shortcuts{Timers: ts, Reset: func() {}, now: func() time.Time { return now }}
	for _, c := range []struct{ in, name, say string }{
		{"今何時？", "time", "午後3時4分です。"},
		{"今日は何曜日？", "date", "11月22日、日曜日です。"},
		{"3分のタイマー", "timer", "3分のタイマーをセットしました。"},
		{"ラーメンの三分タイマーをセットして", "timer", "ラーメンのタイマーをセットしました。"},
		{"1時間半後に教えて", "timer", "1時間30分のタイマーをセットしました。"},
		{"2分30秒測って", "timer", "2分30秒のタイマーをセットしました。"},
		{"タイマーあと何分？", "left", ""},
		{"ラーメンのタイマーを止めて", "cancel", "ラーメンのタイマーを止めました。"},
		{"タイマーを全部止めて", "cancel", "タイマーを3個止めました。"},
		{"会話をリセットして", "reset", "はい、ここまでの話は忘れました。"},
	} {
		a, ok := s.Match(c.in)
		if !ok || a.Name != c.name {
			t.Errorf("Match(%q) = %q, %v; want %q", c.in, a.Name, ok, c.name)
			continue
		}
		if say := a.Run(); c.say != "" && say != c.say {
			t.Errorf("Match(%q) says %q, want %q", c.in, say, c.say)
		}
	}
	for _, in := range []string{"明日の天気は？", "タイマーって何？", "今日のニュース"} {
		if a, ok := s.Match(in); ok {
			t.Errorf("Match(%q) = %q; want the model", in, a.Name)
		}
	}
}

func TestSpokenTime(t *testing.T) {
	for _, c := range []struct {
		h, m int
		want string
	}{{0, 5, "午前0時5分"}, {9, 0, "午前9時ちょうど"}, {12, 30, "午後0時30分"}, {23, 59, "午後11時59分"}} {
		if got := spokenTime(time.Date(2026, 1, 1, c.h, c.m, 0, 0, time.Local)); got != c.want {
			t.Errorf("spokenTime(%d:%02d) = %q, want %q", c.h, c.m, got, c.want)
		}
	}
	if got := spokenDuration(90 * time.Minute); got != "1時間30分" {
		t.Errorf("spokenDuration = %q", got)
	}
}

func TestTimers(t *testing.T) {
	ts := NewTimers()
	rang := make(chan *Timer, 1)
	ts.Ring = func(t *Timer) { rang <- t }
	ts.Set(time.Hour, "洗濯")
	short := ts.Set(20*time.Millisecond, "")
	select {
	case got := <-rang:
		if got != short {
			t.Errorf("rang %v, want the short one", got.Name())
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the timer did not ring")
	}
	if l := ts.List(); len(l) != 1 || l[0].Label != "洗濯" {
		t.Errorf("left %v", l)
	}
	if got := ts.Describe(); !strings.HasPrefix(got, "洗濯のタイマーがあと59分") {
		t.Errorf("Describe() = %q", got)
	}
	if n := len(ts.Cancel("")); n != 1 {
		t.Errorf("cancelled %d", n)
	}
}

// fakeWeb answers the tools' requests by the host and path.
type fakeWeb map[string]string

func (f fakeWeb) RoundTrip(r *http.Request) (*http.Response, error) {
	for prefix, body := range f {
		if strings.HasPrefix(r.URL.Host+r.URL.Path, prefix) {
			if prefix == "geocoding-api.open-meteo.com/v1/search" && r.URL.Query().Get("name") != "大阪市" {
				body = `{}`
			}
			return &http.Response{StatusCode: 200, Status: "200 OK", Body: io.NopCloser(strings.NewReader(body)), Header: http.Header{}}, nil
		}
	}
	return &http.Response{StatusCode: 404, Status: "404 Not Found", Body: io.NopCloser(strings.NewReader("")), Header: http.Header{}}, nil
}

func TestWeatherAndNews(t *testing.T) {
	web := fakeWeb{
		"geocoding-api.open-meteo.com/v1/search": `{"results": [{"name": "大阪市", "admin1": "大阪府", "latitude": 34.69, "longitude": 135.5, "population": 2753862}]}`,
		"api.open-meteo.com/v1/forecast": `{"current": {"temperature_2m": 9.2, "weather_code": 0},
			"daily": {"weather_code": [1, 61, 3], "temperature_2m_max": [15.1, 12, 14], "temperature_2m_min": [6, 8, 7], "precipitation_probability_max": [0, 80, 20]}}`,
		"news.example/cat7.xml": `<rss><channel><item><title>大の里 優勝</title><description>大相撲九州場所は千秋楽。大の里が…</description></item></channel></rss>`,
	}
	tools := &Tools{Timers: NewTimers(), Place: "東京", Feed: "https://news.example/%s.xml", Client: &http.Client{Transport: web}}
	call := func(name, args string) string {
		for _, t := range tools.List() {
			if t.Name == name {
				var a agent.Args
				json.Unmarshal([]byte(args), &a)
				res, err := t.Run(context.Background(), a)
				if err != nil {
					return "error: " + err.Error()
				}
				return res
			}
		}
		return "no tool " + name
	}
	got := call("get_weather", `{"place": "大阪"}`)
	want := "大阪市。今は晴れ、9度。今日は晴れ時々曇り、最高15度、最低6度、降水確率0%。明日は雨、最高12度、最低8度、降水確率80%。明後日は曇り、最高14度、最低7度、降水確率20%。"
	if got != want {
		t.Errorf("get_weather = %q\nwant %q", got, want)
	}
	if got := call("get_news", `{"category": "スポーツ"}`); got != "1. 大の里 優勝: 大相撲九州場所は千秋楽。\n" {
		t.Errorf("get_news = %q", got)
	}
	if got := call("set_timer", `{"seconds": 180}`); got != "3分のタイマーをセットしました。" {
		t.Errorf("set_timer = %q", got)
	}
	tools.Timers.Cancel("")
}
