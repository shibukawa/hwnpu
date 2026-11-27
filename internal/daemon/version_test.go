package daemon

import (
	"strings"
	"testing"
)

// A client says when hwnpud's definitions are not its own, and stops before sending a setting
// hwnpud's engine does not take.
func TestVersionNote(t *testing.T) {
	defer func(b [][]byte) { Builtin = b }(Builtin)
	Builtin = [][]byte{[]byte("models"), []byte("recipes")}
	mine := Definitions()
	if mine == "" || len(mine) != 12 {
		t.Fatalf("Definitions: %q", mine)
	}
	for _, c := range []struct {
		defs, want string
	}{{mine, ""}, {"", "older than this client"}, {"0123456789ab", "other model definitions"}} {
		if got := VersionNote(Hello{Build: "b", Definitions: c.defs}); c.want == "" && got != "" || !strings.Contains(got, c.want) {
			t.Errorf("definitions %q: %q, want %q", c.defs, got, c.want)
		}
	}
	Builtin = nil
	if got := VersionNote(Hello{Build: "b"}); got != "" {
		t.Errorf("a process without definitions: %q", got)
	}
}

func TestHelloCheckSettings(t *testing.T) {
	defer func(b [][]byte) { Builtin = b }(Builtin)
	Builtin = [][]byte{[]byte("models"), []byte("recipes")}
	h := Hello{Build: "b", Definitions: Definitions(), Roles: map[string][]string{"chat": {"idle", "sparse"}}}
	if err := h.CheckSettings(map[string]map[string]string{"chat": {"idle": "5m"}, "speak": {"x": "1"}}); err != nil {
		t.Fatal(err) // speak: a role hwnpud does not tell
	}
	// the same version: a name of no setting, not hwnpud's age
	err := h.CheckSettings(map[string]map[string]string{"chat": {"reserve_moe_memroy": "8GB"}})
	if err == nil || !strings.Contains(err.Error(), "reserve_moe_memroy of chat (it takes idle, sparse)") || strings.Contains(err.Error(), "version") {
		t.Fatalf("a typo: %v", err)
	}
	h.Definitions = "0123456789ab"
	err = h.CheckSettings(map[string]map[string]string{"chat": {"reserve_moe_memory": "8GB"}})
	if err == nil || !strings.Contains(err.Error(), "not this client's version") {
		t.Fatalf("another version: %v", err)
	}
}
