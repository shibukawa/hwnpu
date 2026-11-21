// Package envfile reads hwnpu's settings from .env files (docs/cache-sharing-design.md 7.4) into the
// environment when a program starts: .env.local and .env of the current directory, of the
// user's settings (~/.config/hwnpu, %LOCALAPPDATA%\hwnpu) and of the machine's (/etc/hwnpu,
// /Library/Application Support/hwnpu, %ProgramData%\hwnpu), in that order. A variable the
// environment sets, or a file before sets, wins. Only hwnpu's, Hugging Face's and AWS's
// variables are read (HWNPU_, HF_, AWS_): a project's own .env changes nothing else.
//
// A file has a variable a line: NAME=VALUE, with an optional "export " before it; '#' starts a
// comment; a VALUE in single quotes is taken as written, one in double quotes may have \n, \",
// \\ in it.
package envfile

import (
	"bufio"
	"cmp"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// Prefixes are the beginnings of the variables a file may set.
var Prefixes = []string{"HWNPU_", "HF_", "AWS_"}

// Loaded are the files read, in order, and what each set.
var Loaded []File

// File is a file read and the variables it set.
type File struct {
	Path string
	Set  []string
	Err  error
}

func init() {
	if !testing.Testing() { // a test sets what it needs itself
		Load()
	}
}

// Dirs are the folders whose .env.local and .env are read, in order.
func Dirs() []string {
	var dirs []string
	if wd, err := os.Getwd(); err == nil {
		dirs = append(dirs, wd)
	}
	if d := UserDir(); d != "" {
		dirs = append(dirs, d)
	}
	return append(dirs, MachineDir())
}

// UserDir is the folder of the user's settings (where models.yaml is).
func UserDir() string {
	if runtime.GOOS == "windows" {
		if d := os.Getenv("LOCALAPPDATA"); d != "" {
			return filepath.Join(d, "hwnpu")
		}
	}
	if d := os.Getenv("XDG_CONFIG_HOME"); d != "" {
		return filepath.Join(d, "hwnpu")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".config", "hwnpu")
}

// MachineDir is the folder of the machine's settings, which its administrator puts there.
func MachineDir() string {
	switch runtime.GOOS {
	case "windows":
		return filepath.Join(cmp.Or(os.Getenv("ProgramData"), `C:\ProgramData`), "hwnpu")
	case "darwin":
		return "/Library/Application Support/hwnpu"
	}
	return "/etc/hwnpu"
}

// Load reads the files of Dirs.
func Load() {
	seen := map[string]bool{}
	for _, d := range Dirs() {
		for _, n := range []string{".env.local", ".env"} {
			p := filepath.Join(d, n)
			if seen[p] {
				continue
			}
			seen[p] = true
			if f, ok := LoadFile(p); ok {
				Loaded = append(Loaded, f)
			}
		}
	}
}

// LoadFile sets the variables of the file p that the environment does not set; ok is false
// when there is no such file.
func LoadFile(p string) (File, bool) {
	f := File{Path: p}
	ok, err := readFile(p, func(k, v string) {
		if _, set := os.LookupEnv(k); set {
			return
		}
		os.Setenv(k, v)
		f.Set = append(f.Set, k)
	})
	f.Err = err
	return f, ok
}

// readFile calls set for each of hwnpu's variables of the file p in order; ok is false when there
// is no such file.
func readFile(p string, set func(k, v string)) (ok bool, err error) {
	in, err := os.Open(p)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return true, err
	}
	defer in.Close()
	sc := bufio.NewScanner(in)
	n := 0
	for sc.Scan() {
		n++
		k, v, ok, err := parseLine(sc.Text())
		if err != nil {
			return true, fmt.Errorf("%s:%d: %w", p, n, err)
		}
		if ok && ours(k) {
			set(k, v)
		}
	}
	return true, sc.Err()
}

// Original is the environment the process was started with: os.Environ without what the files
// set when it started.
func Original() []string {
	fromFiles := map[string]bool{}
	for _, f := range Loaded {
		for _, k := range f.Set {
			fromFiles[k] = true
		}
	}
	var out []string
	for _, kv := range os.Environ() {
		if k, _, _ := strings.Cut(kv, "="); !fromFiles[k] {
			out = append(out, kv)
		}
	}
	return out
}

// Environ is the environment a process started now with base would have after reading the files
// of Dirs as they are now (hwnpud gives it to its engine, so that a file changed is read
// without restarting hwnpud), and the files read. Under go test the files are not read, as Load.
func Environ(base []string) (env []string, files []string, err error) {
	if testing.Testing() {
		return slices.Clone(base), nil, nil
	}
	return environIn(base, Dirs())
}

func environIn(base, dirs []string) (env []string, files []string, err error) {
	env = slices.Clone(base)
	have := map[string]bool{}
	for _, kv := range base {
		k, _, _ := strings.Cut(kv, "=")
		have[k] = true
	}
	seen := map[string]bool{}
	for _, d := range dirs {
		for _, n := range []string{".env.local", ".env"} {
			p := filepath.Join(d, n)
			if seen[p] {
				continue
			}
			seen[p] = true
			ok, ferr := readFile(p, func(k, v string) {
				if !have[k] {
					have[k] = true
					env = append(env, k+"="+v)
				}
			})
			if ok {
				files = append(files, p)
			}
			if ferr != nil && err == nil {
				err = ferr
			}
		}
	}
	return env, files, err
}

func ours(k string) bool {
	for _, p := range Prefixes {
		if strings.HasPrefix(k, p) {
			return true
		}
	}
	return false
}

// parseLine reads a line of a file: ok is false for a blank line or a comment.
func parseLine(line string) (key, value string, ok bool, err error) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return "", "", false, nil
	}
	line = strings.TrimPrefix(line, "export ")
	k, v, found := strings.Cut(line, "=")
	k = strings.TrimSpace(k)
	if !found || k == "" || strings.ContainsAny(k, " \t") {
		return "", "", false, fmt.Errorf("%q is not NAME=VALUE", line)
	}
	v = strings.TrimSpace(v)
	switch {
	case strings.HasPrefix(v, "'"):
		end := strings.Index(v[1:], "'")
		if end < 0 {
			return "", "", false, fmt.Errorf("%s: no closing '", k)
		}
		return k, v[1 : 1+end], true, nil
	case strings.HasPrefix(v, `"`):
		var b strings.Builder
		for i := 1; i < len(v); i++ {
			switch c := v[i]; {
			case c == '"':
				return k, b.String(), true, nil
			case c == '\\' && i+1 < len(v):
				i++
				switch v[i] {
				case 'n':
					b.WriteByte('\n')
				default:
					b.WriteByte(v[i])
				}
			default:
				b.WriteByte(c)
			}
		}
		return "", "", false, fmt.Errorf(`%s: no closing "`, k)
	}
	if i := strings.Index(v, " #"); i >= 0 {
		v = strings.TrimSpace(v[:i])
	}
	return k, v, true, nil
}
