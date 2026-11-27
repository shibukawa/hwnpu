package airole

import (
	"strings"
	"testing"
)

// A role refuses a setting its Load does not read, naming those it takes.
func TestCheckSettings(t *testing.T) {
	r := &Role{Name: "chat", LoadSettings: []string{"idle", "sparse"}}
	if err := r.CheckSettings(map[string]string{"idle": "5m", "sparse": "off"}); err != nil {
		t.Fatal(err)
	}
	err := r.CheckSettings(map[string]string{"idle": "5m", "reserve": "8GB"})
	if err == nil || !strings.Contains(err.Error(), "reserve") || !strings.Contains(err.Error(), "idle, sparse") {
		t.Fatalf("an unknown setting: %v", err)
	}
	if err := (&Role{Name: "faces"}).CheckSettings(map[string]string{"x": "1"}); err == nil || !strings.Contains(err.Error(), "none") {
		t.Fatalf("a role of no settings: %v", err)
	}
}
