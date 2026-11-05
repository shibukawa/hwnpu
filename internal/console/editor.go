package console

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/width"
)

// maxHistory is how many entries a history file keeps.
const maxHistory = 1000

// Editor reads lines from standard input. On a terminal it edits them: the arrows move and go
// through the history (↑ ↓, Ctrl-P, Ctrl-N), Home, End, Ctrl-A, Ctrl-E, Ctrl-K, Ctrl-U and Ctrl-W
// as in a shell; Ctrl-C clears the line (and quits on an empty one), Ctrl-D quits on an empty
// one. A pasted text comes in whole, its line breaks included (bracketed paste), and a line ending
// in \ continues on the next; a line break shows as ⏎. Japanese and other wide characters take two
// columns. Elsewhere (a pipe), it reads lines as they come, a line ending in \ continuing.
type Editor struct {
	in   *bufio.Reader
	out  io.Writer
	edit bool // standard input and output are a terminal
	cols func() int

	file    string // the history file, "" for none
	history []string
	row     int // the row of the cursor below the prompt's, as last drawn
}

// NewEditor returns an editor of standard input with the history called name (kept in the user
// cache directory, hwnpu/history/NAME; "" for none).
func NewEditor(name string) *Editor {
	e := &Editor{in: bufio.NewReader(os.Stdin), out: os.Stdout, edit: Terminal(os.Stdin) && Terminal(os.Stdout), cols: func() int { return columns(os.Stdout) }}
	if name != "" {
		if d, err := os.UserCacheDir(); err == nil {
			e.file = filepath.Join(d, "hwnpu", "history", name)
			e.history = readHistory(e.file)
		}
	}
	return e
}

// NewPlainEditor returns an editor that reads lines from in as from a pipe, without editing or a
// history file, and prompts on out.
func NewPlainEditor(in io.Reader, out io.Writer) *Editor {
	return &Editor{in: bufio.NewReader(in), out: out, cols: func() int { return 80 }}
}

// Read prints the prompt and reads a line, kept in the history. It returns io.EOF at the end of
// the input (Ctrl-D, or Ctrl-C on an empty line).
func (e *Editor) Read(prompt string) (string, error) {
	line, err := e.read(prompt, e.history)
	if err == nil && strings.TrimSpace(line) != "" {
		e.remember(line)
	}
	return line, err
}

// Ask is Read without the history: an answer to a question (y/N).
func (e *Editor) Ask(prompt string) (string, error) {
	return e.read(prompt, nil)
}

func (e *Editor) read(prompt string, history []string) (string, error) {
	if !e.edit {
		return e.readPlain(prompt)
	}
	restore, err := makeRaw(int(os.Stdin.Fd()))
	if err != nil {
		return e.readPlain(prompt)
	}
	io.WriteString(e.out, "\x1b[?2004h") // bracketed paste
	defer func() {
		io.WriteString(e.out, "\x1b[?2004l")
		restore()
	}()
	return e.editLine(prompt, history)
}

// readPlain reads lines without editing, those ending in \ joined to the next.
func (e *Editor) readPlain(prompt string) (string, error) {
	io.WriteString(e.out, prompt)
	var lines []string
	for {
		line, err := e.in.ReadString('\n')
		if err != nil && line == "" {
			io.WriteString(e.out, "\n")
			return "", io.EOF
		}
		line = strings.TrimRight(line, "\r\n")
		if strings.HasSuffix(line, `\`) {
			lines = append(lines, strings.TrimSuffix(line, `\`))
			io.WriteString(e.out, ". ")
			continue
		}
		return strings.Join(append(lines, line), "\n"), nil
	}
}

// errEnd ends editLine at the end of the input.
var errEnd = errors.New("end")

// editLine edits a line on the terminal in raw mode.
func (e *Editor) editLine(prompt string, history []string) (string, error) {
	var buf []rune
	pos := 0
	at := len(history) // the history entry shown; len(history) is the line being written
	pending := ""      // that line, while an entry is shown
	paste := false
	e.row = 0
	e.draw(prompt, buf, pos)
	set := func(s string) {
		buf, pos = []rune(s), len([]rune(s))
	}
	insert := func(r ...rune) {
		buf = append(buf[:pos], append(append([]rune{}, r...), buf[pos:]...)...)
		pos += len(r)
	}
	for {
		r, _, err := e.in.ReadRune()
		if err != nil {
			e.finish(prompt, buf)
			return "", io.EOF
		}
		if paste {
			switch {
			case r == 0x1b:
				if seq := e.escape(); seq == "[201~" {
					paste = false
				}
			case r == '\r' || r == '\n':
				insert('\n')
			case r == '\t' || r >= 0x20 && r != 0x7f:
				insert(r)
			}
			e.draw(prompt, buf, pos)
			continue
		}
		switch r {
		case '\r', '\n':
			if pos == len(buf) && pos > 0 && buf[pos-1] == '\\' {
				buf[pos-1] = '\n' // the line goes on
				break
			}
			e.finish(prompt, buf)
			return string(buf), nil
		case 0x03: // Ctrl-C
			if len(buf) == 0 {
				e.finish(prompt, buf)
				return "", io.EOF
			}
			e.finish(prompt, buf)
			io.WriteString(e.out, "^C\r\n")
			buf, pos, at = nil, 0, len(history)
			e.row = 0
		case 0x04: // Ctrl-D
			if len(buf) == 0 {
				e.finish(prompt, buf)
				return "", io.EOF
			}
			if pos < len(buf) {
				buf = append(buf[:pos], buf[pos+1:]...)
			}
		case 0x7f, 0x08: // backspace
			if pos > 0 {
				buf = append(buf[:pos-1], buf[pos:]...)
				pos--
			}
		case 0x01: // Ctrl-A
			pos = 0
		case 0x05: // Ctrl-E
			pos = len(buf)
		case 0x02: // Ctrl-B
			pos = max(pos-1, 0)
		case 0x06: // Ctrl-F
			pos = min(pos+1, len(buf))
		case 0x0b: // Ctrl-K
			buf = buf[:pos]
		case 0x15: // Ctrl-U
			buf, pos = buf[pos:], 0
		case 0x17: // Ctrl-W
			start := wordStart(buf, pos)
			buf, pos = append(buf[:start], buf[pos:]...), start
		case 0x10: // Ctrl-P
			at, pending = e.older(history, at, pending, buf, set)
		case 0x0e: // Ctrl-N
			at = e.newer(history, at, pending, set)
		case 0x0c: // Ctrl-L
			io.WriteString(e.out, "\x1b[H\x1b[2J")
			e.row = 0
		case 0x1b:
			switch e.escape() {
			case "[A", "OA":
				at, pending = e.older(history, at, pending, buf, set)
			case "[B", "OB":
				at = e.newer(history, at, pending, set)
			case "[C", "OC":
				pos = min(pos+1, len(buf))
			case "[D", "OD":
				pos = max(pos-1, 0)
			case "[H", "OH", "[1~", "[7~":
				pos = 0
			case "[F", "OF", "[4~", "[8~":
				pos = len(buf)
			case "[3~": // delete
				if pos < len(buf) {
					buf = append(buf[:pos], buf[pos+1:]...)
				}
			case "[200~":
				paste = true
			case "b", "[1;3D", "[1;5D": // a word back
				pos = wordStart(buf, pos)
			case "f", "[1;3C", "[1;5C": // a word on
				pos = wordEnd(buf, pos)
			case "\x7f": // Alt-Backspace
				start := wordStart(buf, pos)
				buf, pos = append(buf[:start], buf[pos:]...), start
			}
		default:
			if r == '\t' || r >= 0x20 {
				insert(r)
			}
		}
		e.draw(prompt, buf, pos)
	}
}

// older shows the history entry before the one shown, keeping the line being written.
func (e *Editor) older(history []string, at int, pending string, buf []rune, set func(string)) (int, string) {
	if at == 0 {
		return at, pending
	}
	if at == len(history) {
		pending = string(buf)
	}
	at--
	set(history[at])
	return at, pending
}

// newer shows the history entry after the one shown, or the line being written.
func (e *Editor) newer(history []string, at int, pending string, set func(string)) int {
	if at >= len(history) {
		return at
	}
	at++
	if at == len(history) {
		set(pending)
	} else {
		set(history[at])
	}
	return at
}

// escape reads the rest of an escape sequence after ESC: "[" and its parameters and final byte
// (CSI), "O" and a letter (SS3), or one character (Alt with a key).
func (e *Editor) escape() string {
	r, _, err := e.in.ReadRune()
	if err != nil {
		return ""
	}
	switch r {
	case '[':
		seq := []rune{'['}
		for {
			c, _, err := e.in.ReadRune()
			if err != nil {
				return string(seq)
			}
			seq = append(seq, c)
			if c >= 0x40 && c <= 0x7e {
				return string(seq)
			}
		}
	case 'O':
		c, _, _ := e.in.ReadRune()
		return "O" + string(c)
	}
	return string(r)
}

func wordStart(buf []rune, pos int) int {
	for pos > 0 && unicode.IsSpace(buf[pos-1]) {
		pos--
	}
	for pos > 0 && !unicode.IsSpace(buf[pos-1]) {
		pos--
	}
	return pos
}

func wordEnd(buf []rune, pos int) int {
	for pos < len(buf) && unicode.IsSpace(buf[pos]) {
		pos++
	}
	for pos < len(buf) && !unicode.IsSpace(buf[pos]) {
		pos++
	}
	return pos
}

// finish draws the line with the cursor at its end and moves to the next row.
func (e *Editor) finish(prompt string, buf []rune) {
	e.draw(prompt, buf, len(buf))
	io.WriteString(e.out, "\r\n")
	e.row = 0
}

// draw redraws the prompt and the line from the prompt's row, wrapped at the terminal's width,
// and puts the cursor at pos.
func (e *Editor) draw(prompt string, buf []rune, pos int) {
	var sb strings.Builder
	sb.WriteString("\r")
	if e.row > 0 {
		fmt.Fprintf(&sb, "\x1b[%dA", e.row)
	}
	sb.WriteString("\x1b[J")
	sb.WriteString(prompt)
	for _, r := range buf {
		sb.WriteString(shown(r))
	}
	l := layout(visibleWidth(prompt), buf, e.cols())
	endRow, endCol := l.at(len(buf))
	if endCol == l.cols {
		sb.WriteString("\n") // past the last column: the terminal waits to wrap; go to the next row
		endRow++
	}
	row, col := l.at(pos)
	if col == l.cols {
		row, col = row+1, 0
	}
	if up := endRow - row; up > 0 {
		fmt.Fprintf(&sb, "\x1b[%dA", up)
	}
	sb.WriteString("\r")
	if col > 0 {
		fmt.Fprintf(&sb, "\x1b[%dC", col)
	}
	e.row = row
	io.WriteString(e.out, sb.String())
}

// shown is how a character of the line is drawn: a line break as ⏎, a tab as →.
func shown(r rune) string {
	switch r {
	case '\n':
		return "⏎"
	case '\t':
		return "→"
	}
	return string(r)
}

// cellWidth is the columns a character takes on the terminal: 2 for East Asian wide and full
// width ones, 0 for combining marks, 1 otherwise (⏎ and → for a line break and a tab).
func cellWidth(r rune) int {
	switch {
	case r == '\n' || r == '\t':
		return 1
	case unicode.Is(unicode.Mn, r) || unicode.Is(unicode.Me, r) || r == 0x200d:
		return 0
	}
	switch width.LookupRune(r).Kind() {
	case width.EastAsianWide, width.EastAsianFullwidth:
		return 2
	}
	return 1
}

// visibleWidth is the columns a prompt takes, its escape sequences left out.
func visibleWidth(s string) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] == 0x1b && i+1 < len(s) && s[i+1] == '[' {
			i += 2
			for i < len(s) && !(s[i] >= 0x40 && s[i] <= 0x7e) {
				i++
			}
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		n += cellWidth(r)
		i += size - 1
	}
	return n
}

// lineLayout is where each character of a line goes on a terminal of cols columns: a character
// that does not fit in what is left of a row goes to the next.
type lineLayout struct {
	cols      int
	rows, col []int // of the position before each character, and after the last
}

func layout(prompt int, buf []rune, cols int) lineLayout {
	if cols <= 0 {
		cols = 80
	}
	l := lineLayout{cols: cols}
	row, col := prompt/cols, prompt%cols
	if prompt > 0 && col == 0 {
		row, col = row-1, cols // the prompt fills its last row
	}
	for _, r := range buf {
		w := cellWidth(r)
		if col+w > cols {
			row, col = row+1, 0
		}
		l.rows, l.col = append(l.rows, row), append(l.col, col)
		col += w
	}
	l.rows, l.col = append(l.rows, row), append(l.col, col)
	return l
}

// at is the row and column of the cursor before character i (after the last for len(buf)); a
// column of cols is past the end of the row.
func (l lineLayout) at(i int) (int, int) { return l.rows[i], l.col[i] }

// remember adds a line to the history (not twice in a row) and to its file.
func (e *Editor) remember(line string) {
	if n := len(e.history); n > 0 && e.history[n-1] == line {
		return
	}
	e.history = append(e.history, line)
	if len(e.history) > maxHistory {
		e.history = e.history[len(e.history)-maxHistory:]
	}
	if e.file == "" {
		return
	}
	if err := os.MkdirAll(filepath.Dir(e.file), 0o700); err != nil {
		return
	}
	var sb strings.Builder
	for _, h := range e.history {
		b, _ := json.Marshal(h) // one line each, its line breaks escaped
		sb.Write(b)
		sb.WriteByte('\n')
	}
	os.WriteFile(e.file, []byte(sb.String()), 0o600)
}

// readHistory reads a history file: a JSON string a line.
func readHistory(path string) []string {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []string
	for _, line := range strings.Split(string(b), "\n") {
		var s string
		if json.Unmarshal([]byte(line), &s) == nil && s != "" {
			out = append(out, s)
		}
	}
	if len(out) > maxHistory {
		out = out[len(out)-maxHistory:]
	}
	return out
}
