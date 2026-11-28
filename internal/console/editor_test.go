package console

import (
	"bufio"
	"bytes"
	"io"
	"path/filepath"
	"slices"
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

// Select moves with the arrows (and Ctrl-P, Ctrl-N, k, j, Home, End) from the row given and
// chooses with Enter; Esc alone, q, Ctrl-C and the end of the input leave without one.
func TestSelect(t *testing.T) {
	const up, down, home, end = "\x1b[A", "\x1b[B", "\x1b[H", "\x1b[F"
	rows := []string{"zero", "one", "two", "three", "four"}
	for _, c := range []struct {
		name, keys string
		cur, want  int
	}{
		{"Enter keeps the row", "\r", 1, 1},
		{"down twice", down + down + "\r", 0, 2},
		{"up past the first", up + up + "\r", 1, 0},
		{"down past the last", down + down + down + "\r", 3, 4},
		{"Ctrl-N, j, Ctrl-P, k", "\x0ej\x0ej\x10k\r", 0, 2},
		{"End, Home", end + up + "\r", 0, 3},
		{"Home", home + "\r", 4, 0},
		{"Esc alone", down + "\x1b", 0, -1},
		{"q", down + "q", 0, -1},
		{"Ctrl-C", "\x03", 2, -1},
		{"the input ends", down, 0, -1},
		{"a row out of range", "\r", 9, 4},
	} {
		e := fake(c.keys, 40)
		if got := e.selectRow("HEAD", rows, c.cur, nil); got != c.want {
			t.Errorf("%s: %d, want %d", c.name, got, c.want)
		}
	}
	// not a terminal: the caller writes the rows
	if i, err := NewPlainEditor(strings.NewReader("\r"), io.Discard).Select("", rows, 0, nil); i != -1 || err != ErrNoTerminal {
		t.Errorf("on a pipe: %d, %v", i, err)
	}
}

// Of more rows than the terminal has, Select shows those around the one marked, each cut to the
// terminal's width, and erases what it drew when it returns.
func TestSelectDraws(t *testing.T) {
	var out bytes.Buffer
	rows := []string{"zero", "one", "two", "three", "a row that is longer than the terminal is wide"}
	e := &Editor{in: bufio.NewReader(strings.NewReader("\x1b[B\x1b[B\x1b[B\x1b[B\r")), out: &out, edit: true,
		cols: func() int { return 20 }, lines: func() int { return 5 }} // the headings, 2 rows and the keys
	if got := e.selectRow("HEAD", rows, 0, nil); got != 4 {
		t.Fatalf("chose %d, want 4", got)
	}
	frames := strings.Split(out.String(), "\x1b[J")
	first, last := frames[0], frames[len(frames)-2]
	if !strings.Contains(first, "> zero") || !strings.Contains(first, "  one") || strings.Contains(first, "two") || !strings.Contains(first, "1-2 of 5") {
		t.Errorf("the first frame: %q", first)
	}
	if strings.Contains(last, "zero") || !strings.Contains(last, "  three") || !strings.Contains(last, "> a row that is lo…") || !strings.Contains(last, "4-5 of 5") {
		t.Errorf("the last frame: %q", last)
	}
	for _, line := range strings.Split(strings.TrimSuffix(last, "\r\x1b[3A"), "\r\n") {
		if w := visibleWidth(line); w > 19 {
			t.Errorf("a line of %d columns on a terminal of 20: %q", w, line)
		}
	}
	// 4 lines drawn: back to the first and erased from there, at each redraw and at the end
	if !strings.HasSuffix(out.String(), "\r\x1b[3A\x1b[J") {
		t.Errorf("not erased at the end: %q", out.String()[max(0, out.Len()-20):])
	}
}

// On a terminal of colors, Select paints each row as it is cut and marks the one chosen in bold
// around its paint; without colors it paints nothing.
func TestSelectPaints(t *testing.T) {
	rows := []string{"zero tag", "a row that is longer than the terminal is wide"}
	var painted []string
	paint := func(row string) string {
		painted = append(painted, row)
		return strings.Replace(row, "tag", "\x1b[36mtag\x1b[39m", 1)
	}
	var out bytes.Buffer
	e := &Editor{in: bufio.NewReader(strings.NewReader("\r")), out: &out, edit: true, color: true, cols: func() int { return 20 }}
	if got := e.selectRow("", rows, 0, paint); got != 0 {
		t.Fatalf("chose %d", got)
	}
	frame := strings.Split(out.String(), "\x1b[J")[0]
	if !strings.Contains(frame, "\x1b[1m> zero \x1b[36mtag\x1b[39m\x1b[0m") {
		t.Errorf("the row chosen: %q", frame)
	}
	if !slices.Equal(painted, []string{"zero tag", "a row that is lo…"}) {
		t.Errorf("painted %q", painted)
	}
	for _, line := range strings.Split(frame, "\r\n") {
		line, _, _ = strings.Cut(line, "\r") // the last: the cursor back to the first
		if w := visibleWidth(line); w > 19 {
			t.Errorf("a line of %d columns on a terminal of 20: %q", w, line)
		}
	}
	painted = nil
	e = &Editor{in: bufio.NewReader(strings.NewReader("\r")), out: io.Discard, edit: true, cols: func() int { return 20 }}
	if e.selectRow("", rows, 0, paint); painted != nil {
		t.Errorf("painted %q on a terminal without colors", painted)
	}
}
