package chat

import (
	"errors"
	"io"
	"strings"
	"testing"
)

// setPolicy sets the model policy of items (comma separated) for the test.
func setPolicy(t *testing.T, items string) {
	t.Helper()
	rules, err := ParsePolicy(strings.Split(items, ","), PolicyEnv)
	if err != nil {
		t.Fatal(err)
	}
	old := Policy
	Policy = rules
	t.Cleanup(func() { Policy = old })
}

func TestParsePolicy(t *testing.T) {
	rules, err := ParsePolicy([]string{" -license:Non-Commercial", "+nsfw", "", "+license:apache-2.0"}, "test")
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, r := range rules {
		got = append(got, r.String())
	}
	if s := strings.Join(got, ","); s != "-license:non-commercial,+nsfw,+license:apache-2.0" {
		t.Errorf("rules %s", s)
	}
	for _, bad := range []string{"license:mit", "-colour:red", "+nsfw:yes", "-license:", "-license"} {
		if _, err := ParsePolicy([]string{bad}, "test"); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
}

// TestBuiltinLicenses checks that every model of the built-in definitions says its license, with
// its class (docs/model-provenance.md).
func TestBuiltinLicenses(t *testing.T) {
	for _, m := range Catalog {
		if m.License.ID == "" || m.License.URL == "" || m.License.Class == "" {
			t.Errorf("%s: license %+v needs an id, a url and a class", m.Name, m.License)
		}
	}
}

func TestPolicyPicks(t *testing.T) {
	for _, tc := range []struct {
		policy, name string
		machine      int
		want         string // the model picked, or "error"
	}{
		// NSFW models run only with +nsfw, and the image's pick is not NSFW (FLUX.2 klein from
		// 8 GB): Anima runs when it is named
		{"", "image", 8, "FLUX.2-klein-4B"},
		{"", "image", 16, "FLUX.2-klein-4B"},
		{"+nsfw", "image", 8, "FLUX.2-klein-4B"},
		{"+nsfw", "image", 16, "FLUX.2-klein-4B"},
		{"+nsfw", "anima-turbo", 8, "Anima-Turbo"},
		{"", "anima", 8, "error"},
		{"+nsfw,-nsfw", "image", 8, "FLUX.2-klein-4B"},
		// license: an id or a class; a use takes another of its kind when its own are refused
		{"-license:conditional", "chat", 8, "Qwen3.5-2B"}, // LFM3 caps the revenue: an open model of 8 GB
		{"-license:conditional", "chat", 16, "gemma-4-E2B-it"},
		{"+nsfw,-license:non-commercial", "image", 8, "FLUX.2-klein-4B"},
		{"-license:lfm-open-1.0", "speech", 8, "error"},
		{"+license:open", "translate", 8, "error"},
		{"+license:open,+license:plamo-community-license", "translate", 8, "plamo-3-translate-0.8b"},
		{"-license:apache-2.0", "image", 8, "error"}, // the others are NSFW (Anima) or Apache-2.0 too (Chroma1)
	} {
		setPolicy(t, tc.policy)
		m, why, ok, err := Pick(tc.name, tc.machine)
		switch {
		case !ok:
			t.Errorf("%s: Pick(%s, %d) found nothing", tc.policy, tc.name, tc.machine)
		case tc.want == "error" && err == nil:
			t.Errorf("%s: Pick(%s, %d) = %s, want an error", tc.policy, tc.name, tc.machine, m.Name)
		case tc.want != "error" && (err != nil || m.Name != tc.want):
			t.Errorf("%s: Pick(%s, %d) = %s, %v; want %s", tc.policy, tc.name, tc.machine, m.Name, err, tc.want)
		case tc.want != "error" && tc.policy != "" && tc.want != picked(t, tc.name, tc.machine) && !strings.Contains(why, "refused by the model policy"):
			t.Errorf("%s: Pick(%s, %d): why %q does not say what was refused", tc.policy, tc.name, tc.machine, why)
		}
	}
}

// picked is the model Pick picks with no policy but +nsfw.
func picked(t *testing.T, name string, machine int) string {
	old := Policy
	defer func() { Policy = old }()
	Policy, _ = ParsePolicy([]string{"+nsfw"}, "test")
	m, _, _, _ := Pick(name, machine)
	return m.Name
}

// TestPolicyNamed checks that a model named (-m, -use, a request) is refused with why, not
// replaced, and that a repository's model, of no license known, passes a - and not a +.
func TestPolicyNamed(t *testing.T) {
	setPolicy(t, "-license:conditional")
	_, err := ResolveFor("LFM3-1.7B", KindChat, io.Discard)
	var pe *PolicyError
	if !errors.As(err, &pe) || !strings.Contains(err.Error(), "lfm-open-1.0, conditional") {
		t.Errorf("LFM3-1.7B: %v", err)
	}
	if _, err := ResolveFor("someone/some-model", KindImage, io.Discard); err != nil {
		t.Errorf("a repository under -license:conditional: %v", err)
	}
	setPolicy(t, "+license:open")
	if _, err := ResolveFor("someone/some-model", KindImage, io.Discard); !errors.As(err, &pe) {
		t.Errorf("a repository under +license:open: %v", err)
	}
	if _, err := ResolveFor("gemma-4-E2B-it", KindChat, io.Discard); err != nil {
		t.Errorf("gemma-4-E2B-it under +license:open: %v", err)
	}
}

// TestPolicyFiles checks the policy of a user's file, read before the environment's, and that
// the built-in definitions may have none.
func TestPolicyFiles(t *testing.T) {
	if _, err := userFiles(t, "version: 1\npolicy: [-license:conditional]\n", "version: 1\npolicy: [+nsfw, -license:mit]\n"); err != nil {
		t.Fatal(err)
	}
	if got := joinRules(Policy); got != "-license:conditional,+nsfw,-license:mit" {
		t.Errorf("policy %s", got)
	}
	if _, err := userFiles(t, "version: 1\npolicy: [license:mit]\n"); err == nil {
		t.Error("an item without + or - was read")
	}
	if err := newTable(nil).add(Builtin, []byte("version: 1\npolicy: [+nsfw]\n"), true); err == nil {
		t.Error("the built-in definitions had a policy")
	}
	t.Setenv(PolicyEnv, "+nsfw,-license:mit")
	if err := LoadModels(nil, true, nil); err != nil {
		t.Fatal(err)
	}
	if got := joinRules(Policy); got != "+nsfw,-license:mit" {
		t.Errorf("policy of %s: %s", PolicyEnv, got)
	}
}

// TestAllowedAlso checks that a client's policy narrows hwnpud's and does not widen it.
func TestAllowedAlso(t *testing.T) {
	m := Model{Name: "x", License: License{ID: "lfm-open-1.0", Class: "conditional"}}
	nsfw := Model{Name: "y", License: License{ID: "apache-2.0", Class: "open"}, Content: []string{"nsfw"}}
	setPolicy(t, "+nsfw")
	if err := AllowedAlso(m, ""); err != nil {
		t.Errorf("no policy of the client's: %v", err)
	}
	var pe *PolicyError
	if err := AllowedAlso(m, "-license:conditional"); !errors.As(err, &pe) || !strings.Contains(err.Error(), "the client's") {
		t.Errorf("the client's -license:conditional: %v", err)
	}
	if err := AllowedAlso(nsfw, "+nsfw"); err != nil {
		t.Errorf("both +nsfw: %v", err)
	}
	if err := AllowedAlso(nsfw, "-license:mit"); !errors.As(err, &pe) {
		t.Errorf("the client without +nsfw: %v", err)
	}
	setPolicy(t, "-license:conditional")
	if err := AllowedAlso(m, "+license:lfm-open-1.0"); !errors.As(err, &pe) {
		t.Errorf("the client's + over hwnpud's -: %v", err)
	}
	if err := AllowedAlso(m, "bad"); err == nil {
		t.Error("an item without + or -")
	}
}
