package main

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/bonsaiai/ai/agent/voice"
)

// Shortcuts answer the requests that need no model at once: the time and the date, setting,
// stopping and asking after timers, forgetting the conversation. The words are matched as
// Whisper writes them (digits, or kanji numbers), so "3分のタイマー" and "ラーメンの三分タイマー
// をセットして" are both a timer of 3 minutes. Anything else goes to the model.
type Shortcuts struct {
	Timers *Timers
	// Reset forgets the conversation (nil: not offered).
	Reset func()
	// now is the clock (time.Now; a test's).
	now func() time.Time
}

// Action is what a shortcut does: Run makes its change, when the request is confirmed, and
// returns the answer to speak.
type Action struct {
	Name string // for the log: time, date, timer, cancel, left, reset
	Run  func() string
}

// Func is the shortcuts as a voice.Shortcut.
func (s *Shortcuts) Func() voice.Shortcut {
	return func(request string) (func() string, bool) {
		a, ok := s.Match(request)
		return a.Run, ok
	}
}

var (
	timerWords  = regexp.MustCompile(`タイマー|測って|計って|はかって|(後|あと)(で|に)?(教えて|知らせて|呼んで)|経ったら|たったら`)
	cancelWords = regexp.MustCompile(`止め|とめ|消し|消して|けし|キャンセル|解除|ストップ|やめ|オフ`)
	leftWords   = regexp.MustCompile(`あと|残り|のこり|何分|なんぷん|どれくらい|確認|いくつ`)
	timeWords   = regexp.MustCompile(`何時|なんじ|時刻|今の時間|いまの時間|今、?時間`)
	dateWords   = regexp.MustCompile(`何日|なんにち|何曜|なんよう|日付|曜日`)
	resetWords  = regexp.MustCompile(`(会話|話|はなし)を?(リセット|忘れて|わすれて)|リセットして`)
	durationRE  = regexp.MustCompile(`(\d+)\s*(時間|分|秒)(半)?`)
	labelRE     = regexp.MustCompile(`^(.+?)(用)?の\s*\d`)
)

// Match is the shortcut of a request, if one answers it.
func (s *Shortcuts) Match(request string) (Action, bool) {
	now := time.Now
	if s.now != nil {
		now = s.now
	}
	r := normalizeNumbers(request)
	switch {
	case s.Reset != nil && resetWords.MatchString(r):
		return Action{Name: "reset", Run: func() string { s.Reset(); return "はい、ここまでの話は忘れました。" }}, true
	case s.Timers != nil && strings.Contains(r, "タイマー") && cancelWords.MatchString(r):
		return s.cancel(r), true
	case s.Timers != nil && strings.Contains(r, "タイマー") && leftWords.MatchString(r) && !durationRE.MatchString(r):
		return Action{Name: "left", Run: s.Timers.Describe}, true
	case s.Timers != nil && timerWords.MatchString(r):
		d := parseDuration(r)
		if d <= 0 {
			return Action{}, false
		}
		label := ""
		if m := labelRE.FindStringSubmatch(r); m != nil && !strings.ContainsAny(m[1], "0123456789") {
			label = strings.TrimSpace(m[1])
		}
		name := (&Timer{Label: label, For: d}).Name()
		return Action{Name: "timer", Run: func() string { s.Timers.Set(d, label); return name + "をセットしました。" }}, true
	case timeWords.MatchString(r):
		return Action{Name: "time", Run: func() string { return spokenTime(now()) + "です。" }}, true
	case dateWords.MatchString(r):
		return Action{Name: "date", Run: func() string { return spokenDate(now()) + "です。" }}, true
	}
	return Action{}, false
}

// cancel is the action that stops timers: those of the label the request names
// ("ラーメンのタイマーを止めて"), or all of them.
func (s *Shortcuts) cancel(r string) Action {
	label := ""
	if i := strings.Index(r, "のタイマー"); i > 0 {
		label = strings.TrimSpace(r[:i])
	}
	return Action{Name: "cancel", Run: func() string {
		stopped := s.Timers.Cancel(label)
		if len(stopped) == 0 && label != "" {
			stopped = s.Timers.Cancel("")
		}
		switch len(stopped) {
		case 0:
			return "止めるタイマーはありません。"
		case 1:
			return stopped[0].Name() + "を止めました。"
		}
		return fmt.Sprintf("タイマーを%d個止めました。", len(stopped))
	}}
}

// parseDuration is the duration a request says ("1時間半", "2分30秒", "90秒"), 0 for none.
func parseDuration(r string) time.Duration {
	var d time.Duration
	for _, m := range durationRE.FindAllStringSubmatch(r, -1) {
		n, err := strconv.Atoi(m[1])
		if err != nil {
			continue
		}
		unit := map[string]time.Duration{"時間": time.Hour, "分": time.Minute, "秒": time.Second}[m[2]]
		d += time.Duration(n) * unit
		if m[3] != "" {
			d += unit / 2
		}
	}
	return d
}

// spokenTime is the time as it is said: "午後3時15分", "午前9時ちょうど".
func spokenTime(t time.Time) string {
	ampm := "午前"
	h := t.Hour()
	if h >= 12 {
		ampm, h = "午後", h-12
	}
	if t.Minute() == 0 {
		return fmt.Sprintf("%s%d時ちょうど", ampm, h)
	}
	return fmt.Sprintf("%s%d時%d分", ampm, h, t.Minute())
}

// spokenDate is the date as it is said: "11月22日、日曜日".
func spokenDate(t time.Time) string {
	days := []string{"日", "月", "火", "水", "木", "金", "土"}
	return fmt.Sprintf("%d月%d日、%s曜日", int(t.Month()), t.Day(), days[t.Weekday()])
}

// normalizeNumbers writes the numbers of a text in ASCII digits: full-width digits, and kanji
// numbers up to the thousands (三十 → 30, 二十五 → 25, 百 → 100).
func normalizeNumbers(s string) string {
	digit := map[rune]int{'〇': 0, '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
	unit := map[rune]int{'十': 10, '百': 100, '千': 1000}
	var b strings.Builder
	rs := []rune(s)
	for i := 0; i < len(rs); {
		r := rs[i]
		if r >= '０' && r <= '９' {
			b.WriteRune('0' + r - '０')
			i++
			continue
		}
		_, isDigit := digit[r]
		_, isUnit := unit[r]
		if !isDigit && !isUnit {
			b.WriteRune(r)
			i++
			continue
		}
		// a kanji number: digits and units, read to their end
		j := i
		for j < len(rs) {
			_, d := digit[rs[j]]
			_, u := unit[rs[j]]
			if !d && !u {
				break
			}
			j++
		}
		// a lone 一 in a word (一緒, 一番) is left alone unless a unit or a counter follows
		word := rs[i:j]
		if len(word) == 1 && isDigit && (j >= len(rs) || !strings.ContainsRune("時分秒", rs[j])) {
			b.WriteRune(r)
			i++
			continue
		}
		total, cur := 0, 0
		for _, c := range word {
			if d, ok := digit[c]; ok {
				cur = cur*10 + d
				continue
			}
			total += max(cur, 1) * unit[c]
			cur = 0
		}
		b.WriteString(strconv.Itoa(total + cur))
		i = j
	}
	return b.String()
}
