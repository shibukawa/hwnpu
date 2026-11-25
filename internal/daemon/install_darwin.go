package daemon

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ServiceFile is the definition of the installed service (the LaunchAgent's plist).
func ServiceFile() string { return PlistPath() }

// LogPath is where the service writes its log.
func LogPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "Library", "Logs", "bonsaiaid.log")
}

// LogWhere is where the service's log is.
func LogWhere() string { return LogPath() }

// ServiceEnviron is what launchd gives the service beyond the .env files: nothing of the shell's.
func ServiceEnviron() []string { return nil }

// PlistPath is the LaunchAgent's definition.
func PlistPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "Library", "LaunchAgents", Label+".plist")
}

// Plist is the LaunchAgent that runs exe's supervisor when a client connects to socket, with its
// log in logPath.
func Plist(exe, socket, logPath string) []byte {
	esc := func(s string) string {
		var b bytes.Buffer
		xml.EscapeText(&b, []byte(s))
		return b.String()
	}
	return []byte(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>` + Label + `</string>
	<key>ProgramArguments</key>
	<array>
		<string>` + esc(exe) + `</string>
		<string>serve</string>
	</array>
	<key>Sockets</key>
	<dict>
		<key>` + SocketName + `</key>
		<dict>
			<key>SockPathName</key>
			<string>` + esc(socket) + `</string>
			<key>SockPathMode</key>
			<integer>384</integer>
		</dict>
	</dict>
	<key>ProcessType</key>
	<string>Interactive</string>
	<key>StandardOutPath</key>
	<string>` + esc(logPath) + `</string>
	<key>StandardErrorPath</key>
	<string>` + esc(logPath) + `</string>
</dict>
</plist>
`)
}

// Install writes the LaunchAgent for exe and loads it: launchd listens on the socket and starts
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
	socket := DefaultSocketPath()
	for _, dir := range []string{filepath.Dir(PlistPath()), filepath.Dir(socket), filepath.Dir(LogPath())} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	if err := os.Chmod(filepath.Dir(socket), 0o700); err != nil {
		return err
	}
	uninstallQuiet()
	os.Remove(socket) // launchd binds it anew
	if err := os.WriteFile(PlistPath(), Plist(exe, socket, LogPath()), 0o644); err != nil {
		return err
	}
	if b, err := exec.Command("launchctl", "bootstrap", domain(), PlistPath()).CombinedOutput(); err != nil {
		return fmt.Errorf("launchctl bootstrap %s: %v: %s", PlistPath(), err, strings.TrimSpace(string(b)))
	}
	// launchd binds the socket elsewhere and moves it to its path after bootstrap returns: a client
	// right after install would find no socket
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(50 * time.Millisecond) {
		if _, err := os.Stat(socket); err == nil {
			break
		}
	}
	fmt.Fprintf(out, "installed %s: launchd starts %s when a client connects to %s (log: %s)\n", Label, exe, socket, LogPath())
	return nil
}

// Uninstall unloads the LaunchAgent and removes its definition and socket.
func Uninstall(out io.Writer) error {
	if _, err := os.Stat(PlistPath()); os.IsNotExist(err) {
		fmt.Fprintf(out, "%s is not installed\n", Label)
		return nil
	}
	uninstallQuiet()
	if err := os.Remove(PlistPath()); err != nil {
		return err
	}
	os.Remove(DefaultSocketPath())
	os.Remove(EngineSocketPath(DefaultSocketPath()))
	fmt.Fprintf(out, "uninstalled %s\n", Label)
	return nil
}

// Installed is the executable of the installed LaunchAgent, "" when there is none.
func Installed() string {
	b, err := os.ReadFile(PlistPath())
	if err != nil {
		return ""
	}
	s := string(b)
	i := strings.Index(s, "<key>ProgramArguments</key>")
	if i < 0 {
		return ""
	}
	s = s[i:]
	a, b2 := strings.Index(s, "<string>"), strings.Index(s, "</string>")
	if a < 0 || b2 < a {
		return ""
	}
	var v struct {
		S string `xml:",chardata"`
	}
	if xml.Unmarshal([]byte("<x>"+s[a+len("<string>"):b2]+"</x>"), &v) != nil {
		return ""
	}
	return v.S
}

// uninstallQuiet unloads the LaunchAgent and waits until launchd has let it go: bootout returns
// before a running bonsaiaid has exited, and a bootstrap before then fails (5: Input/output error).
func uninstallQuiet() {
	exec.Command("launchctl", "bootout", domain()+"/"+Label).Run()
	for deadline := time.Now().Add(45 * time.Second); time.Now().Before(deadline); time.Sleep(100 * time.Millisecond) {
		if exec.Command("launchctl", "print", domain()+"/"+Label).Run() != nil {
			return
		}
	}
}

func domain() string { return "gui/" + strconv.Itoa(os.Getuid()) }
