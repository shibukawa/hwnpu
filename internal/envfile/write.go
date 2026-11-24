package envfile

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// UserFile is the user's .env, which bonsaiai config writes and bonsaiaid reads.
func UserFile() string {
	if d := UserDir(); d != "" {
		return filepath.Join(d, ".env")
	}
	return ""
}

// Var is a variable of a file.
type Var struct{ Name, Value string }

// Read is the variables of bonsaiai's that the file p sets, in order (the first of a name is the one
// used); ok is false when there is no such file.
func Read(p string) (vars []Var, ok bool, err error) {
	ok, err = readFile(p, func(k, v string) { vars = append(vars, Var{k, v}) })
	return vars, ok, err
}

// Lookup is what the file p sets name to (the first line of it).
func Lookup(p, name string) (value string, set bool, err error) {
	vars, _, err := Read(p)
	for _, v := range vars {
		if v.Name == name {
			return v.Value, true, err
		}
	}
	return "", false, err
}

// CheckName says whether name is a variable a file may set (Prefixes, a name of a shell).
func CheckName(name string) error {
	if !validName.MatchString(name) {
		return fmt.Errorf("%q is not a variable's name", name)
	}
	if !ours(name) {
		return fmt.Errorf("%s: the .env files set only %s variables", name, strings.Join(Prefixes, "*, ")+"*")
	}
	return nil
}

var (
	validName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	bare      = regexp.MustCompile(`^[A-Za-z0-9_./:@,+=%~-]*$`)
)

// Set writes name=value to the file p: in place of its line (the others of the same name are
// removed), else at its end. The rest of the file (comments, the other variables) is kept as it
// is. A new file is made only for this user (0600): it may hold credentials.
func Set(p, name, value string) error {
	if err := CheckName(name); err != nil {
		return err
	}
	if strings.ContainsAny(value, "\r\x00") {
		return fmt.Errorf("%s: a value of a .env file has no carriage return or NUL", name)
	}
	line := name + "=" + quote(value)
	return rewrite(p, func(lines []string) []string {
		var out []string
		done := false
		for _, l := range lines {
			if k, _, ok, _ := parseLine(l); ok && k == name {
				if !done {
					out = append(out, line)
					done = true
				}
				continue
			}
			out = append(out, l)
		}
		if !done {
			out = append(out, line)
		}
		return out
	})
}

// Unset removes the lines of name from the file p; removed is false when it had none.
func Unset(p, name string) (removed bool, err error) {
	if _, err := os.Stat(p); os.IsNotExist(err) {
		return false, nil
	}
	err = rewrite(p, func(lines []string) []string {
		var out []string
		for _, l := range lines {
			if k, _, ok, _ := parseLine(l); ok && k == name {
				removed = true
				continue
			}
			out = append(out, l)
		}
		return out
	})
	return removed, err
}

// quote is value as a file has it: as is when it has nothing a reader would take apart, else in
// double quotes.
func quote(v string) string {
	if bare.MatchString(v) {
		return v
	}
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`)
	return `"` + r.Replace(v) + `"`
}

// rewrite replaces the lines of the file p (none when there is no such file) with edit's, by a
// file beside it renamed over it: a reader sees the old or the new, never half.
func rewrite(p string, edit func([]string) []string) error {
	mode := os.FileMode(0o600)
	b, err := os.ReadFile(p)
	switch {
	case err == nil:
		if fi, err := os.Stat(p); err == nil {
			mode = fi.Mode().Perm()
		}
	case os.IsNotExist(err):
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
	default:
		return err
	}
	var lines []string
	if s := strings.TrimSuffix(string(b), "\n"); s != "" || len(b) > 0 {
		lines = strings.Split(s, "\n")
	}
	var out bytes.Buffer
	for _, l := range edit(lines) {
		out.WriteString(l)
		out.WriteByte('\n')
	}
	tmp, err := os.CreateTemp(filepath.Dir(p), ".env-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(out.Bytes()); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Chmod(mode); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), p)
}
