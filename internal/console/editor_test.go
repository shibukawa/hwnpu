package console

import (
	"bufio"
	"io"
	"path/filepath"
	"strings"
	"testing"
)

// fake is an editor reading keys from a string, drawing to a discarded terminal of cols columns.
func fake(keys string, cols int, history ...string) *Editor {
	return &Editor{in: bufio.NewReader(strings.NewReader(keys)), out: io.Discard, edit: true, cols: func() int { return cols }, history: history}
}

func TestEditLine(t *testing.T) {
	const (
		up, down, left, right = "\x1b[A", "\x1b[B", "\x1b[D", "\x1b[C"
		home, end, del        = "\x1b[H", "\x1b[F", "\x1b[3~"
	)
	for _, c := range []struct {
		name, keys string
		history    []string
		want       string
	}{
		{"typed", "hello\r", nil, "hello"},
		{"Japanese", "会議は月曜日\r", nil, "会議は月曜日"},
		{"backspace", "helx\x7flo\r", nil, "hello"},
		{"left and insert", "hllo" + left + left + left + "e\r", nil, "hello"},
		{"home, end, delete", "xhello" + home + del + end + "!\r", nil, "hello!"},
		{"Ctrl-A, Ctrl-K", "hello world\x01\x06\x06\x06\x06\x06\x0b\r", nil, "hello"},
		{"Ctrl-W", "hello world\x17\r", nil, "hello "},
		{"Ctrl-U", "hello world" + left + left + left + left + left + "\x15\r", nil, "world"},
		{"up", up + "\r", []string{"first", "second"}, "second"},
		{"up twice, edit", up + up + "!\r", []string{"first", "second"}, "first!"},
		{"up past the oldest", up + up + up + "\r", []string{"first", "second"}, "first"},
		{"up and down back to the line", "draft" + up + up + down + down + "\r", []string{"first", "second"}, "draft"},
		{"continued", "one\\\rtwo\r", nil, "one\ntwo"},
		{"pasted lines", "\x1b[200~line 1\rline 2\x1b[201~\r", nil, "line 1\nline 2"},
		{"Ctrl-C clears", "junk\x03hello\r", nil, "hello"},
		{"Ctrl-D deletes", "hxello" + home + right + "\x04\r", nil, "hello"},
	} {
		got, err := fake(c.keys, 20, c.history...).editLine("> ", c.history)
		if err != nil || got != c.want {
			t.Errorf("%s: %q, %v; want %q", c.name, got, err, c.want)
		}
	}
	for _, keys := range []string{"\x04", "\x03", ""} {
		if _, err := fake(keys, 20).editLine("> ", nil); err != io.EOF {
			t.Errorf("%q on an empty line: %v, want io.EOF", keys, err)
		}
	}
}

// Wide characters take two columns, and one that does not fit at the end of a row goes to the
// next; a row filled to its end leaves the cursor at the start of the next.
func TestLayout(t *testing.T) {
	l := layout(2, []rune("abc"), 5) // "> abc" fills the row
	if r, c := l.at(3); r != 0 || c != 5 {
		t.Errorf("after abc: row %d column %d, want 0 5", r, c)
	}
	l = layout(2, []rune("日本語"), 5) // "> 日" then 本 does not fit in the last column
	for i, want := range [][2]int{{0, 2}, {1, 0}, {1, 2}, {1, 4}} {
		if r, c := l.at(i); r != want[0] || c != want[1] {
			t.Errorf("before %d: row %d column %d, want %v", i, r, c, want)
		}
	}
	if w := visibleWidth("\x1b[1m> \x1b[0m"); w != 2 {
		t.Errorf("the prompt's width %d, want 2", w)
	}
	if cellWidth('あ') != 2 || cellWidth('Ａ') != 2 || cellWidth('a') != 1 || cellWidth('́') != 0 {
		t.Error("cell widths")
	}
}

// The history keeps lines across editors through its file, without repeating the last.
func TestHistoryFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history")
	e := fake("", 20)
	e.file = path
	for _, l := range []string{"one", "two\nlines", "two\nlines", "three"} {
		e.remember(l)
	}
	got := readHistory(path)
	if strings.Join(got, "|") != "one|two\nlines|three" {
		t.Errorf("history %q", got)
	}
}

func TestHighlight(t *testing.T) {
	got := Highlight(" 97.7s, 4 parts, NOT all on the NPU / all on the NPU")
	want := " 97.7s, 4 parts, " + red + "NOT all on the NPU" + gray + " / " + green + "all on the NPU" + gray
	if got != want {
		t.Errorf("Highlight = %q, want %q", got, want)
	}
}
