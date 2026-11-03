package chat

import (
	"bytes"
	"crypto/sha256"
	"encoding/gob"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm/decode"
)

// A store keeps prompts read earlier on disk, so that a prompt that begins like one of them is
// not read again after the engine restarts: an agent's system message and tools (7K tokens for
// OpenCode, 13 to 17 seconds on an M3) come back from disk instead. An entry is one sequence:
// its tokens, the rows of the longest caches of all its positions (decode.Session.ExportRows,
// the same in every tier) and marks (decode.Mark, the rings and the windows, with the logits)
// every markEvery tokens and at its end. A prompt sharing a beginning with an entry comes back at
// the last mark inside the shared beginning and reads the rest. The entries of a model are in a
// directory of their own (storeKey: the name and the programs), each a directory of
// meta.json, rows.bin and mark-POS.bin.
type store struct {
	dir     string
	entries []*entry
}

// entry is one kept sequence. Tokens are all of it; Rows is how many positions rows.bin holds
// (the length of Tokens, or fewer after a failed write); Marks are the positions with a mark
// file, ascending.
type entry struct {
	dir      string
	Tokens   []int32 `json:"tokens"`
	Rows     int     `json:"rows"`
	Marks    []int   `json:"marks"`
	RowBytes int     `json:"row_bytes"`
	Used     int64   `json:"used"` // unix nanoseconds of the last use
}

// storedMark is a mark file.
type storedMark struct {
	Pos    int
	Bufs   map[string][]byte
	Logits []float32
}

var (
	markEvery = 512  // tokens between the marks of an entry
	minStored = 1024 // shorter sequences are not kept: they read in a second
	maxStored = 4    // entries per model; the least recently used goes
)

// storeKey names the directory of a model's entries: the rows depend on the model and on its
// conversion, so the name, the packages and their weights' sizes and times go into it.
func storeKey(name string, packages []string, weightBytes []int64) string {
	h := sha256.New()
	fmt.Fprintln(h, name)
	for _, p := range packages {
		fmt.Fprintln(h, filepath.Base(p))
		if fi, err := os.Stat(p); err == nil {
			fmt.Fprintln(h, fi.ModTime().UnixNano())
		}
	}
	for _, b := range weightBytes {
		fmt.Fprintln(h, b)
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// openStore opens the entries in dir, making it if it does not exist. Entries that cannot be
// read are left out.
func openStore(dir string) (*store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	st := &store{dir: dir}
	ds, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, d := range ds {
		if !d.IsDir() {
			continue
		}
		e := &entry{dir: filepath.Join(dir, d.Name())}
		b, err := os.ReadFile(filepath.Join(e.dir, "meta.json"))
		if err != nil || json.Unmarshal(b, e) != nil || e.RowBytes <= 0 {
			continue
		}
		if fi, err := os.Stat(filepath.Join(e.dir, "rows.bin")); err != nil || fi.Size() < int64(e.Rows)*int64(e.RowBytes) {
			continue
		}
		st.entries = append(st.entries, e)
	}
	return st, nil
}

// find returns the entry and the mark position that bring prompt furthest without reading: the
// last mark inside the longest beginning that prompt shares with an entry, whose rows the entry
// holds. 0 and nil for none.
func (st *store) find(prompt []int32) (*entry, int) {
	var best *entry
	bestPos := 0
	for _, e := range st.entries {
		n := 0
		for n < len(e.Tokens) && n < len(prompt) && e.Tokens[n] == prompt[n] {
			n++
		}
		for _, p := range e.Marks {
			if p <= n && p <= e.Rows && p > 0 && (p > bestPos || p == bestPos && e.Used > best.Used) {
				best, bestPos = e, p
			}
		}
	}
	return best, bestPos
}

// load reads the rows before pos and the mark at pos of an entry.
func (st *store) load(e *entry, pos int) (rows []byte, m *decode.Mark, logits []float32, err error) {
	if !slices.Contains(e.Marks, pos) || pos > e.Rows {
		return nil, nil, nil, fmt.Errorf("chat: no mark at %d in %s", pos, e.dir)
	}
	f, err := os.Open(filepath.Join(e.dir, "rows.bin"))
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()
	rows = make([]byte, pos*e.RowBytes)
	if _, err := io.ReadFull(f, rows); err != nil {
		return nil, nil, nil, fmt.Errorf("chat: %s: rows: %w", e.dir, err)
	}
	sm, err := readMark(filepath.Join(e.dir, markFile(pos)))
	if err != nil {
		return nil, nil, nil, err
	}
	if sm.Pos != pos {
		return nil, nil, nil, fmt.Errorf("chat: %s: the mark file of %d is at %d", e.dir, pos, sm.Pos)
	}
	e.Used = time.Now().UnixNano()
	if err := e.writeMeta(); err != nil {
		return nil, nil, nil, err
	}
	return rows, decode.NewMark(sm.Pos, sm.Bufs), sm.Logits, nil
}

func markFile(pos int) string { return "mark-" + strconv.Itoa(pos) + ".bin" }

func readMark(path string) (*storedMark, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var sm storedMark
	if err := gob.NewDecoder(f).Decode(&sm); err != nil {
		return nil, fmt.Errorf("chat: %s: %w", path, err)
	}
	return &sm, nil
}

func writeMark(path string, sm storedMark) error {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(sm); err != nil {
		return err
	}
	return writeFile(path, buf.Bytes())
}

// writeFile writes a file through a temporary name, so that a reader never sees a part of it.
func writeFile(path string, b []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (e *entry) writeMeta() error {
	b, err := json.Marshal(e)
	if err != nil {
		return err
	}
	return writeFile(filepath.Join(e.dir, "meta.json"), b)
}

// put keeps the sequence of tokens that the session holds, with marks taken while reading it
// (the last at its end). base is the entry the sequence came from or extended last (nil for
// none): when the tokens go on from base's, base gets the new rows and marks; otherwise a new
// entry is made, with base's marks inside the shared beginning. It returns the entry.
// rowSource is what the store reads a sequence's rows from: decode.Session, ironSession.
type rowSource interface {
	RowBytes() int
	ExportRows(from, to int) ([]byte, error)
}

func (st *store) put(tokens []int32, sess rowSource, marks []storedMark, base *entry) (*entry, error) {
	rb := sess.RowBytes()
	if base != nil && base.RowBytes != rb {
		base = nil
	}
	shared := 0
	if base != nil {
		for shared < len(base.Tokens) && shared < len(tokens) && base.Tokens[shared] == tokens[shared] {
			shared++
		}
	}
	if base != nil && shared == len(base.Tokens) && base.Rows == len(base.Tokens) {
		// goes on from base: append the rows and the marks past it
		if err := base.append(tokens, sess, marks); err != nil {
			return nil, err
		}
		return base, nil
	}
	if base != nil && shared == len(tokens) && base.Rows >= len(tokens) {
		// a beginning of base: add the marks it lacks
		for _, m := range marks {
			if m.Pos <= shared && !slices.Contains(base.Marks, m.Pos) {
				if err := writeMark(filepath.Join(base.dir, markFile(m.Pos)), m); err != nil {
					return nil, err
				}
				base.Marks = append(base.Marks, m.Pos)
			}
		}
		slices.Sort(base.Marks)
		base.Used = time.Now().UnixNano()
		return base, base.writeMeta()
	}
	e := &entry{dir: filepath.Join(st.dir, strconv.FormatInt(time.Now().UnixNano(), 10)), RowBytes: rb}
	if err := os.MkdirAll(e.dir, 0o755); err != nil {
		return nil, err
	}
	if base != nil {
		// base's marks inside the shared beginning, before the sequence's own
		for _, p := range base.Marks {
			if p <= shared && p <= base.Rows {
				if err := copyFile(filepath.Join(base.dir, markFile(p)), filepath.Join(e.dir, markFile(p))); err == nil {
					e.Marks = append(e.Marks, p)
				}
			}
		}
	}
	if err := e.append(tokens, sess, marks); err != nil {
		os.RemoveAll(e.dir)
		return nil, err
	}
	st.entries = append(st.entries, e)
	// the least recently used go
	for len(st.entries) > maxStored {
		old := 0
		for i, o := range st.entries {
			if o.Used < st.entries[old].Used {
				old = i
			}
		}
		os.RemoveAll(st.entries[old].dir)
		st.entries = slices.Delete(st.entries, old, old+1)
	}
	return e, nil
}

// append adds the rows of positions Rows to len(tokens) and the marks past Rows to the entry,
// whose tokens tokens extend.
func (e *entry) append(tokens []int32, sess rowSource, marks []storedMark) error {
	path := filepath.Join(e.dir, "rows.bin")
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Seek(int64(e.Rows)*int64(e.RowBytes), io.SeekStart); err != nil {
		f.Close()
		return err
	}
	const batch = 1024
	for from := e.Rows; from < len(tokens); from += batch {
		to := min(from+batch, len(tokens))
		rows, err := sess.ExportRows(from, to)
		if err != nil {
			f.Close()
			return err
		}
		if _, err := f.Write(rows); err != nil {
			f.Close()
			return err
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	e.Rows = max(e.Rows, len(tokens))
	for _, m := range marks {
		if m.Pos > len(tokens) || slices.Contains(e.Marks, m.Pos) {
			continue
		}
		if err := writeMark(filepath.Join(e.dir, markFile(m.Pos)), m); err != nil {
			return err
		}
		e.Marks = append(e.Marks, m.Pos)
	}
	slices.Sort(e.Marks)
	e.Tokens = slices.Clone(tokens)
	e.Used = time.Now().UnixNano()
	return e.writeMeta()
}

func copyFile(from, to string) error {
	b, err := os.ReadFile(from)
	if err != nil {
		return err
	}
	return writeFile(to, b)
}

// remove forgets an entry whose files went bad.
func (st *store) remove(e *entry) {
	os.RemoveAll(e.dir)
	st.entries = slices.DeleteFunc(st.entries, func(o *entry) bool { return o == e })
}
