package chat

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/catalog"
)

// LicenseEnv accepts licenses without asking: their ids, comma separated, or "all" (for scripts
// and servers, where nothing can be asked).
const LicenseEnv = catalog.LicenseEnv

// LicensesFile is where the accepted licenses are kept: licenses.json beside the user's
// models.yaml (UserModelsFile).
func LicensesFile() string {
	p := UserModelsFile()
	if p == "" {
		return ""
	}
	return filepath.Join(filepath.Dir(p), "licenses.json")
}

// accepted is LicensesFile: by license id, when it was accepted and for which model first.
type accepted map[string]struct {
	At    time.Time `json:"at"`
	Model string    `json:"model"`
}

func readAccepted(path string) accepted {
	a := accepted{}
	if b, err := os.ReadFile(path); err == nil {
		json.Unmarshal(b, &a) // a broken file asks again
	}
	return a
}

// LicenseError is AcceptLicense's error when the license was not accepted.
type LicenseError struct {
	Model   string
	License License
}

func (e *LicenseError) Error() string {
	return fmt.Sprintf("%s is under the %s license (%s)%s; accept it with --accept-license, or %s=%s",
		e.Model, e.License.ID, e.License.URL, terms(e.License), LicenseEnv, e.License.ID)
}

func terms(l License) string {
	if l.Terms == "" {
		return ""
	}
	return ": " + l.Terms
}

// AcceptLicense returns nil when m's license needs no acceptance or has been accepted: before,
// by LicenseEnv, by yes (--accept-license), or now, asked on out and answered on in when ask
// (a terminal). An acceptance is kept in LicensesFile; otherwise a *LicenseError says how to
// accept it.
func AcceptLicense(m Model, yes, ask bool, in io.Reader, out io.Writer) error {
	l := m.License
	if !l.Accept {
		return nil
	}
	path := LicensesFile()
	a := readAccepted(path)
	if _, ok := a[l.ID]; ok {
		return nil
	}
	env := strings.Split(os.Getenv(LicenseEnv), ",")
	for i := range env {
		env[i] = strings.TrimSpace(env[i])
	}
	if slices.Contains(env, "all") || slices.Contains(env, l.ID) {
		return nil // accepted for this run only: the environment says it again next time
	}
	if !yes && ask {
		fmt.Fprintf(out, "%s is under the %s license%s.\n  %s\nAccept it? [y/N] ", m.Name, l.ID, terms(l), l.URL)
		line, _ := bufio.NewReader(in).ReadString('\n')
		yes = strings.EqualFold(strings.TrimSpace(line), "y") || strings.EqualFold(strings.TrimSpace(line), "yes")
	}
	if !yes {
		return &LicenseError{Model: m.Name, License: l}
	}
	if path == "" {
		return errors.New("chat: no configuration directory to keep the license's acceptance in")
	}
	e := a[l.ID]
	e.At, e.Model = time.Now().UTC().Truncate(time.Second), m.Name
	a[l.ID] = e
	b, err := json.MarshalIndent(a, "", " ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
		return fmt.Errorf("chat: keeping the license's acceptance: %w", err)
	}
	fmt.Fprintf(out, "accepted the %s license (kept in %s)\n", l.ID, path)
	return nil
}
