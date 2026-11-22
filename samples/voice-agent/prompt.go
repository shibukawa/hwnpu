package main

import "strings"

// system is the assistant's system prompt: answers to be heard, short, in spoken Japanese.
// It does not change from turn to turn (the date and time are a tool), so the model's cache
// keeps it, and the tools' declarations after it, read once.
const system = `あなたは音声アシスタントです。答えは声で読み上げられ、耳で聞かれます。
- 日本語の話し言葉で、ふつうは1文か2文、短く答える。前置きや繰り返しはしない。
- 箇条書き、記号、絵文字、マークダウン、URL は使わない。数字は読みやすく書く。
- 時刻と日付は get_time、タイマーは set_timer、cancel_timer、list_timers、天気は get_weather、ニュースは get_news を使う。
- 人、場所、ものごと、出来事について事実を聞かれたら、推測せず search で調べてから答える。
- ニュースは見出しを3つまで、それぞれ短く言う。
- 自分の名前は言わない。`

// Profile is what the assistant knows of its user from the start: it goes into the system
// prompt, which does not change while the assistant runs (and stays in the model's cache).
type Profile struct {
	// Name is how the user is called ("" for not at all).
	Name string
	// Place is where the user is: the weather's place when a request names none.
	Place string
	// About is anything else the user tells about themselves (their family, their work, the
	// news they care for), in their words.
	About string
}

// prompt is the system prompt with the profile.
func (p Profile) prompt() string {
	var b strings.Builder
	b.WriteString(system)
	if p.Name != "" || p.Place != "" || p.About != "" {
		b.WriteString("\n\nユーザーについて:")
		if p.Name != "" {
			b.WriteString("\n- 呼び方: " + p.Name)
		}
		if p.Place != "" {
			b.WriteString("\n- 住んでいる所: " + p.Place + " (天気で場所を言われなければここ)")
		}
		if p.About != "" {
			b.WriteString("\n- " + p.About)
		}
	}
	return b.String()
}
