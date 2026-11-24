package daemon

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// Unit is the name of systemd's user units: Unit+".socket" listens, Unit+".service" runs the
// supervisor at the first connection.
const Unit = "bonsaiaid"

// unitDir is where the user's own units are.
func unitDir() string {
	if d := os.Getenv("XDG_CONFIG_HOME"); d != "" {
		return filepath.Join(d, "systemd", "user")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "systemd", "user")
}

// ServiceFile is the definition of the installed service (the .service unit).
func ServiceFile() string { return filepath.Join(unitDir(), Unit+".service") }

func socketFile() string { return filepath.Join(unitDir(), Unit+".socket") }

// LogWhere is where the service's log is: systemd's journal.
func LogWhere() string { return "journalctl --user -u " + Unit }

// unitPath quotes a path for a unit file: in double quotes, with its backslashes, quotes and
// specifiers (%) escaped.
func unitPath(p string) string {
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, `%`, `%%`)
	return `"` + r.Replace(p) + `"`
}

// SocketUnit is bonsaiaid.socket: systemd listens on socket (mode 0600) and passes it to the
// service under SocketName.
func SocketUnit(socket string) []byte {
	return []byte(`[Unit]
Description=bonsaiaid's socket: bonsaiai's NPU and models (docs/design/bonsaiaid.md)

[Socket]
ListenStream=` + strings.ReplaceAll(socket, "%", "%%") + `
SocketMode=0600
DirectoryMode=0700
FileDescriptorName=` + SocketName + `

[Install]
WantedBy=sockets.target
`)
}

// ServiceUnit is bonsaiaid.service, which runs exe's supervisor. The supervisor stops the engine
// itself (KillMode=mixed: SIGTERM to it alone, then SIGKILL to what is left), and it exits when
// idle: systemd starts it again at the next connection.
func ServiceUnit(exe string) []byte {
	return []byte(`[Unit]
Description=bonsaiaid: bonsaiai's NPU and models (docs/design/bonsaiaid.md)
Requires=` + Unit + `.socket
After=` + Unit + `.socket

[Service]
ExecStart=` + unitPath(exe) + ` serve
KillMode=mixed
TimeoutStopSec=45
`)
}

// Install writes the user units for exe and starts the socket: systemd listens on it and starts
// the supervisor at the first connection. An installed one is replaced.
func Install(exe string, out io.Writer) error {
	if exe == "" {
		var err error
		if exe, err = os.Executable(); err != nil {
			return err
		}
	}
	exe, err := filepath.EvalSymlinks(exe)
	if err != nil {
		return err
	}
	if strings.Contains(exe, string(filepath.Separator)+"go-build") {
		return fmt.Errorf("%s is a go run build, gone when it exits: go install ./cmd/bonsaiaid, then bonsaiai service install", exe)
	}
	if _, err := exec.LookPath("systemctl"); err != nil {
		return fmt.Errorf("bonsaiai service install needs systemd's user units (systemctl): %w; run it in the foreground with bonsaiai service start", err)
	}
	if os.Getenv("XDG_RUNTIME_DIR") == "" {
		return fmt.Errorf("bonsaiai service install: XDG_RUNTIME_DIR is unset (no systemd user session here); run it in the foreground with bonsaiai service start")
	}
	socket := DefaultSocketPath()
	if err := os.MkdirAll(unitDir(), 0o755); err != nil {
		return err
	}
	uninstallQuiet()
	os.Remove(socket) // systemd binds it anew
	if err := os.WriteFile(socketFile(), SocketUnit(socket), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(ServiceFile(), ServiceUnit(exe), 0o644); err != nil {
		return err
	}
	for _, args := range [][]string{{"daemon-reload"}, {"enable", "--now", Unit + ".socket"}} {
		if err := systemctl(args...); err != nil {
			return err
		}
	}
	fmt.Fprintf(out, "installed %s.socket: systemd starts %s when a client connects to %s (log: %s)\n", Unit, exe, socket, LogWhere())
	return nil
}

// Uninstall stops the units and removes them and the socket.
func Uninstall(out io.Writer) error {
	if _, err := os.Stat(ServiceFile()); os.IsNotExist(err) {
		if _, err := os.Stat(socketFile()); os.IsNotExist(err) {
			fmt.Fprintf(out, "%s is not installed\n", Unit)
			return nil
		}
	}
	uninstallQuiet()
	for _, f := range []string{socketFile(), ServiceFile()} {
		if err := os.Remove(f); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	systemctl("daemon-reload")
	os.Remove(DefaultSocketPath())
	os.Remove(EngineSocketPath(DefaultSocketPath()))
	fmt.Fprintf(out, "uninstalled %s\n", Unit)
	return nil
}

// Installed is the executable of the installed service, "" when there is none.
func Installed() string {
	b, err := os.ReadFile(ServiceFile())
	if err != nil {
		return ""
	}
	sc := bufio.NewScanner(bytes.NewReader(b))
	for sc.Scan() {
		if v, ok := strings.CutPrefix(strings.TrimSpace(sc.Text()), "ExecStart="); ok {
			return unquoteUnit(v)
		}
	}
	return ""
}

// unquoteUnit is the first word of a unit's command line, unquoted.
func unquoteUnit(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, `"`) {
		w, _, _ := strings.Cut(s, " ")
		return strings.ReplaceAll(w, "%%", "%")
	}
	var b strings.Builder
	for i := 1; i < len(s); i++ {
		switch c := s[i]; {
		case c == '\\' && i+1 < len(s):
			i++
			b.WriteByte(s[i])
		case c == '"':
			return strings.ReplaceAll(b.String(), "%%", "%")
		default:
			b.WriteByte(c)
		}
	}
	return ""
}

// ServiceEnviron is the environment systemd's user manager gives its services (the
// environment.d files and what was imported into it), nil when it cannot be read.
func ServiceEnviron() []string {
	b, err := exec.Command("systemctl", "--user", "show-environment").Output()
	if err != nil {
		return nil
	}
	var out []string
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if k, v, ok := strings.Cut(line, "="); ok {
			out = append(out, k+"="+unquoteShell(v))
		}
	}
	return out
}

// unquoteShell is a value as show-environment prints it: as is, or in $'...' with backslash
// escapes when it has characters a shell would take apart.
func unquoteShell(v string) string {
	if !strings.HasPrefix(v, "$'") || !strings.HasSuffix(v, "'") || len(v) < 3 {
		return v
	}
	v = v[2 : len(v)-1]
	var b strings.Builder
	for i := 0; i < len(v); i++ {
		if v[i] != '\\' || i+1 == len(v) {
			b.WriteByte(v[i])
			continue
		}
		i++
		switch c := v[i]; c {
		case 'n':
			b.WriteByte('\n')
		case 't':
			b.WriteByte('\t')
		case 'x':
			if i+2 < len(v) {
				if n, err := strconv.ParseUint(v[i+1:i+3], 16, 8); err == nil {
					b.WriteByte(byte(n))
					i += 2
					continue
				}
			}
			b.WriteByte(c)
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

func uninstallQuiet() {
	exec.Command("systemctl", "--user", "disable", "--now", Unit+".socket").Run()
	exec.Command("systemctl", "--user", "stop", Unit+".service").Run()
}

func systemctl(args ...string) error {
	b, err := exec.Command("systemctl", append([]string{"--user"}, args...)...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("systemctl --user %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(string(b)))
	}
	return nil
}
