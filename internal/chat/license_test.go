package chat

import (
	"bytes"
	"errors"
	"os"
	"strings"
	"testing"
)

func TestAcceptLicense(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", os.Getenv("XDG_CONFIG_HOME"))
	t.Setenv(LicenseEnv, "")
	m := Model{Name: "M", License: License{ID: "research", URL: "https://example.com/LICENSE", Terms: "not commercial", Accept: true}}
	var out bytes.Buffer

	if err := AcceptLicense(Model{Name: "free", License: License{ID: "apache-2.0"}}, false, false, nil, &out); err != nil {
		t.Fatalf("a license without accept: %v", err)
	}
	// nothing to ask on: the error says how to accept
	err := AcceptLicense(m, false, false, nil, &out)
	var le *LicenseError
	if !errors.As(err, &le) || !strings.Contains(err.Error(), "--accept-license") || !strings.Contains(err.Error(), LicenseEnv+"=research") {
		t.Fatalf("not accepted: %v", err)
	}
	// asked, and declined
	if err := AcceptLicense(m, false, true, strings.NewReader("n\n"), &out); !errors.As(err, &le) {
		t.Fatalf("declined: %v", err)
	}
	// the environment accepts for the run only
	t.Setenv(LicenseEnv, "other, research")
	if err := AcceptLicense(m, false, false, nil, &out); err != nil {
		t.Fatalf("by %s: %v", LicenseEnv, err)
	}
	t.Setenv(LicenseEnv, "")
	if err := AcceptLicense(m, false, false, nil, &out); err == nil {
		t.Fatal("the environment's acceptance was kept")
	}
	// asked, and accepted: kept, and not asked again (another model of the license too)
	if err := AcceptLicense(m, false, true, strings.NewReader("y\n"), &out); err != nil {
		t.Fatalf("accepted: %v", err)
	}
	if _, err := os.Stat(LicensesFile()); err != nil {
		t.Fatalf("not kept: %v", err)
	}
	m2 := m
	m2.Name = "M2"
	if err := AcceptLicense(m2, false, false, nil, &out); err != nil {
		t.Fatalf("asked again: %v", err)
	}
}

func TestLicenseInTable(t *testing.T) {
	yaml := `version: 1
models:
  Research-Model:
    kind: image
    from: hf:owner/research-model
    license: {id: research, url: https://example.com/LICENSE, terms: "not commercial", accept: true}
  Broken:
    kind: image
    from: hf:owner/broken
    license: {accept: true}
`
	tb := newTable(nil)
	if err := tb.add("test.yaml", []byte(yaml), false); err != nil {
		t.Fatal(err)
	}
	_, _, _, err := tb.build()
	if err == nil || !strings.Contains(err.Error(), "Broken") {
		t.Fatalf("a license without its id: %v", err)
	}
}
