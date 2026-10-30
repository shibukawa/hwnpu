package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// Names are the tools, in the order they are offered: read, edit, write, bash.
var Names = []string{"read", "edit", "write", "bash"}

// Tool is one of the agent's tools: its declaration for the model, the rule it adds to the
// system prompt, and what it does.
type Tool struct {
	Name string
	// Rule is the line the tool adds to the rules of the system prompt ("" for none).
	Rule string
	// Decl is the function declaration sent with the request.
	Decl chat.Tool
	// Writes says that the tool changes files or runs commands: Config.Allow is asked first.
	Writes bool
	// Describe says what a call does in a few words (the command, the path), for the screen.
	Describe func(args map[string]any) string
	// Run does the call and returns what the model is told, also when it failed.
	Run func(ctx context.Context, args map[string]any) string
}

// buildTools builds the tools named in the config, in the order of Names.
func (s *Session) buildTools() ([]*Tool, error) {
	names := s.cfg.Tools
	if len(names) == 0 {
		names = Names
	}
	var out []*Tool
	for _, want := range Names {
		for _, n := range names {
			if n == want {
				out = append(out, s.tool(want))
				break
			}
		}
	}
	for _, n := range names {
		found := false
		for _, t := range out {
			found = found || t.Name == n
		}
		if !found {
			return nil, fmt.Errorf("agent: no tool %q; the tools are %s", n, strings.Join(Names, ", "))
		}
	}
	return out, nil
}

func (s *Session) tool(name string) *Tool {
	c := &s.cfg
	switch name {
	case "read":
		return &Tool{
			Name: "read",
			Rule: "Read a file before you edit it (edit and write want that).",
			Decl: decl(fmt.Sprintf(`{"name": "read", "description": "Read a text file. Returns up to %d lines; use offset and limit for the rest.",
 "parameters": {"type": "object", "properties": {
  "path": {"type": "string", "description": "Path of the file, relative to the working directory"},
  "offset": {"type": "integer", "description": "First line to read, counted from 1"},
  "limit": {"type": "integer", "description": "Number of lines to read"}},
  "required": ["path"]}}`, c.ReadLines)),
			Describe: func(args map[string]any) string {
				s := str(args, "path")
				if o := intArg(args, "offset", 0); o > 1 {
					s += fmt.Sprintf(" from line %d", o)
				}
				return s
			},
			Run: func(ctx context.Context, args map[string]any) string { return s.read(args) },
		}
	case "edit":
		return &Tool{
			Name:   "edit",
			Rule:   "Use edit to change an existing file: old is copied exactly from the file and occurs once.",
			Writes: true,
			Decl: decl(`{"name": "edit", "description": "Replace one passage of a file. old must match the file exactly and occur once.",
 "parameters": {"type": "object", "properties": {
  "path": {"type": "string", "description": "Path of the file, relative to the working directory"},
  "old": {"type": "string", "description": "Text to replace, copied exactly from the file"},
  "new": {"type": "string", "description": "Text that replaces it"}},
  "required": ["path", "old", "new"]}}`),
			Describe: func(args map[string]any) string { return str(args, "path") },
			Run:      func(ctx context.Context, args map[string]any) string { return s.edit(args) },
		}
	case "write":
		return &Tool{
			Name:   "write",
			Rule:   "Use write only for a new file or a complete rewrite.",
			Writes: true,
			Decl: decl(`{"name": "write", "description": "Create a file, or replace its whole content. Parent directories are created.",
 "parameters": {"type": "object", "properties": {
  "path": {"type": "string", "description": "Path of the file, relative to the working directory"},
  "content": {"type": "string", "description": "The whole content of the file"}},
  "required": ["path", "content"]}}`),
			Describe: func(args map[string]any) string { return str(args, "path") },
			Run:      func(ctx context.Context, args map[string]any) string { return s.write(args) },
		}
	case "bash":
		return &Tool{
			Name:   "bash",
			Rule:   "Use bash for ls, grep, build and test commands, and check your work with it. Commands run in the current directory; do not cd.",
			Writes: true,
			Decl: decl(fmt.Sprintf(`{"name": "bash", "description": "Run a shell command in the working directory and return its output (stdout and stderr, the last %d lines).",
 "parameters": {"type": "object", "properties": {
  "command": {"type": "string", "description": "The command"},
  "timeout": {"type": "integer", "description": "Seconds before the command is killed (default %d)"}},
  "required": ["command"]}}`, c.TailLines, int(c.Timeout.Seconds()))),
			Describe: func(args map[string]any) string { return str(args, "command") },
			Run:      func(ctx context.Context, args map[string]any) string { return c.bash(ctx, args) },
		}
	}
	panic("agent: unknown tool " + name)
}

func decl(s string) chat.Tool {
	var t chat.Tool
	if err := json.Unmarshal([]byte(s), &t); err != nil {
		panic(err)
	}
	return t
}

func str(args map[string]any, key string) string {
	switch v := args[key].(type) {
	case string:
		return v
	case nil:
		return ""
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

// intArg reads a number the model may have written as a number or as text.
func intArg(args map[string]any, key string, def int) int {
	switch v := args[key].(type) {
	case float64:
		return int(v)
	case string:
		var n int
		if _, err := fmt.Sscanf(strings.TrimSpace(v), "%d", &n); err == nil {
			return n
		}
	}
	return def
}

// resolve finds the file a call names: relative to the working directory, or absolute. A
// small model writes /main.go for main.go; such a path outside the directory is taken as
// relative when that file exists there (or, for a file to create, its directory).
func (c *Config) resolve(path string, create bool) string {
	if !filepath.IsAbs(path) {
		return filepath.Join(c.Dir, path)
	}
	abs := filepath.Clean(path)
	if c.inside(abs) {
		return abs
	}
	rel := filepath.Join(c.Dir, strings.TrimLeft(path, `/\`))
	if _, err := os.Stat(rel); err == nil {
		return rel
	}
	if st, err := os.Stat(filepath.Dir(rel)); create && err == nil && st.IsDir() {
		return rel
	}
	return abs
}

// outside tells the model to use a relative path.
func outside(path string) string {
	return fmt.Sprintf("%s is outside the working directory; use a path relative to it, like main.go.", path)
}

// inside reports whether the path is in the working directory.
func (c *Config) inside(abs string) bool {
	rel, err := filepath.Rel(c.Dir, abs)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func (s *Session) read(args map[string]any) string {
	c := &s.cfg
	path := str(args, "path")
	if path == "" {
		return "path is required."
	}
	abs := c.resolve(path, false)
	info, err := os.Stat(abs)
	switch {
	case errors.Is(err, os.ErrNotExist):
		return c.missing(path)
	case err != nil:
		return err.Error()
	case info.IsDir():
		return fmt.Sprintf("%s is a directory; list it with bash (ls).", path)
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return err.Error()
	}
	s.seen[abs] = true
	if bytes.IndexByte(data[:min(len(data), 8192)], 0) >= 0 {
		return fmt.Sprintf("%s is a binary file.", path)
	}
	if len(data) == 0 {
		return "(empty file)"
	}
	lines := strings.SplitAfter(string(data), "\n")
	if lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	offset := max(intArg(args, "offset", 1), 1)
	limit := intArg(args, "limit", c.ReadLines)
	if limit <= 0 || limit > c.ReadLines {
		limit = c.ReadLines
	}
	if offset > len(lines) {
		return fmt.Sprintf("%s has %d lines; offset %d is past the end.", path, len(lines), offset)
	}
	end, size := offset-1, 0
	for end < len(lines) && end-(offset-1) < limit {
		if size += len(lines[end]); size > c.ReadBytes && end > offset-1 {
			break
		}
		end++
	}
	out := strings.TrimSuffix(strings.Join(lines[offset-1:end], ""), "\n")
	if len(out) > c.ReadBytes {
		out = out[:c.ReadBytes] + "…"
	}
	switch {
	case end < len(lines):
		out += fmt.Sprintf("\n[Lines %d-%d of %d. Use offset=%d to read on.]", offset, end, len(lines), end+1)
	case offset > 1:
		out += fmt.Sprintf("\n[Lines %d-%d of %d.]", offset, end, len(lines))
	}
	return out
}

func (s *Session) write(args map[string]any) string {
	c := &s.cfg
	path, content := str(args, "path"), str(args, "content")
	if path == "" {
		return "path is required."
	}
	abs := c.resolve(path, true)
	if !c.inside(abs) {
		return outside(path)
	}
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return err.Error()
	}
	_, err := os.Stat(abs)
	existed := err == nil
	if existed && !s.seen[abs] {
		return fmt.Sprintf("%s exists and you have not read it. Use edit to change it, or read it first.", path)
	}
	if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
		return err.Error()
	}
	s.seen[abs] = true
	if existed {
		return fmt.Sprintf("Replaced %s (%d bytes).", path, len(content))
	}
	return fmt.Sprintf("Wrote %s (%d bytes).", path, len(content))
}

func (s *Session) edit(args map[string]any) string {
	c := &s.cfg
	path, old, new := str(args, "path"), str(args, "old"), str(args, "new")
	switch {
	case path == "":
		return "path is required."
	case old == "":
		return "old is empty: copy the text to replace from the file."
	}
	abs := c.resolve(path, false)
	if !c.inside(abs) {
		return outside(path)
	}
	info, err := os.Stat(abs)
	if errors.Is(err, os.ErrNotExist) {
		return c.missing(path) + " Use write to create a file."
	} else if err != nil {
		return err.Error()
	}
	if !s.seen[abs] {
		return fmt.Sprintf("Read %s before editing it.", path)
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return err.Error()
	}
	text := string(data)
	n := strings.Count(text, old)
	if n == 0 {
		// the blank lines around old are the model's, not the file's
		if o := strings.Trim(old, "\r\n"); o != "" && o != old && strings.Count(text, o) == 1 {
			old, new, n = o, strings.Trim(new, "\r\n"), 1
		}
	}
	switch {
	case n == 0:
		return fmt.Sprintf("old was not found in %s. Read the file and copy the text exactly, with its indentation.", path)
	case n > 1:
		return fmt.Sprintf("old occurs %d times in %s. Include more of the lines around it so that it occurs once.", n, path)
	}
	i := strings.Index(text, old)
	line := 1 + strings.Count(text[:i], "\n")
	if err := os.WriteFile(abs, []byte(text[:i]+new+text[i+len(old):]), info.Mode().Perm()); err != nil {
		return err.Error()
	}
	return fmt.Sprintf("Edited %s at line %d.", path, line)
}

func (c *Config) bash(ctx context.Context, args map[string]any) string {
	command := str(args, "command")
	if strings.TrimSpace(command) == "" {
		return "command is required."
	}
	if m := cdRe.FindStringSubmatch(command); m != nil {
		// a small model writes cd /tmp && go run main.go for a directory it has not seen
		dir := m[2]
		if home, err := os.UserHomeDir(); err == nil && strings.HasPrefix(dir, "~") {
			dir = home + dir[1:]
		}
		if abs := c.resolve(dir, false); !c.inside(abs) {
			return fmt.Sprintf("Commands run in the current directory, where the files are; do not cd to %s. Run: %s", m[2], strings.TrimSpace(command[len(m[0]):]))
		}
	}
	timeout := c.Timeout
	if n := intArg(args, "timeout", 0); n > 0 {
		timeout = time.Duration(n) * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	name, flag := shell()
	cmd := exec.CommandContext(ctx, name, flag, command)
	cmd.Dir = c.Dir
	var buf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &buf, &buf
	setGroup(cmd)
	cmd.Cancel = func() error { return killGroup(cmd) }
	cmd.WaitDelay = time.Second
	err := cmd.Run()
	// the directory's absolute path is long and a small model copies it wrong: it is "."
	out, omitted := tail(strings.ReplaceAll(buf.String(), c.Dir, "."), c.TailLines, c.TailBytes)
	if omitted > 0 {
		out = fmt.Sprintf("[... %d lines omitted]\n", omitted) + out
	}
	if strings.TrimSpace(out) == "" {
		out = "(no output)"
	}
	var exit *exec.ExitError
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		out += fmt.Sprintf("\n[killed after %d seconds]", int(timeout.Seconds()))
	case errors.As(err, &exit):
		out += fmt.Sprintf("\n[exit status %d]", exit.ExitCode())
	case err != nil:
		out += fmt.Sprintf("\n[%v]", err)
	}
	return out
}

// missing says that a file does not exist, and what the working directory holds.
func (c *Config) missing(path string) string {
	if files := listing(c.Dir); files != "" {
		return fmt.Sprintf("%s does not exist. The files here are: %s.", path, files)
	}
	return fmt.Sprintf("%s does not exist.", path)
}

// Allowed reports whether a command matches one of the patterns: a pattern is the command
// with * for any text, such as "go *", "ls*" or "git status". Each part of a command joined
// with &&, ||, ; or | must match, so that "ls; rm -rf ." is not let through by "ls*".
func Allowed(command string, patterns []string) bool {
	if len(patterns) == 0 {
		return false
	}
	parts := splitRe.Split(command, -1)
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		ok := false
		for _, p := range patterns {
			if glob(strings.TrimSpace(p), part) {
				ok = true
				break
			}
		}
		if !ok {
			return false
		}
	}
	return true
}

// splitRe separates the commands of a line.
var splitRe = regexp.MustCompile(`&&|\|\|?|;|\n`)

// glob matches s against a pattern whose * stands for any text.
func glob(pattern, s string) bool {
	pieces := strings.Split(pattern, "*")
	if len(pieces) == 1 {
		return pattern == s
	}
	if !strings.HasPrefix(s, pieces[0]) {
		return false
	}
	s = s[len(pieces[0]):]
	for _, piece := range pieces[1 : len(pieces)-1] {
		i := strings.Index(s, piece)
		if i < 0 {
			return false
		}
		s = s[i+len(piece):]
	}
	return strings.HasSuffix(s, pieces[len(pieces)-1])
}

// cdRe matches a command that begins by changing the directory: cd DIR && ..., cd "DIR"; ...
var cdRe = regexp.MustCompile(`^\s*cd\s+("?)([^\s"&;|]+)"?\s*(&&|;)\s*`)

// tail keeps the last lines of an output, at most maxLines and about maxBytes, and says how
// many lines it dropped.
func tail(s string, maxLines, maxBytes int) (string, int) {
	s = strings.TrimRight(s, "\n")
	if s == "" {
		return "", 0
	}
	lines := strings.Split(s, "\n")
	keep, size := 0, 0
	for i := len(lines) - 1; i >= 0 && keep < maxLines; i-- {
		if size += len(lines[i]) + min(keep, 1); size > maxBytes && keep > 0 {
			break
		}
		keep++
	}
	out := strings.Join(lines[len(lines)-keep:], "\n")
	if len(out) > maxBytes {
		out = "…" + out[len(out)-maxBytes:]
	}
	return out, len(lines) - keep
}
