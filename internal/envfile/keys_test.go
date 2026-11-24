package envfile

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestKeysShareThePolicy(t *testing.T) {
	allow, _ := KeyByName("model-allowlist")
	deny, _ := KeyByName("model-denylist")
	if allow.Env != deny.Env || len(KeysOf(allow.Env)) != 2 {
		t.Fatalf("%+v %+v", allow, deny)
	}
	v := "+nsfw, -license:non-commercial,+license:open"
	if a, d := allow.Items(v), deny.Items(v); fmt.Sprint(a) != "[nsfw license:open]" || fmt.Sprint(d) != "[license:non-commercial]" {
		t.Fatalf("items %q %q", a, d)
	}
	v = deny.With(v, []string{"license:non-commercial", "license:conditional"})
	if v != "+nsfw,+license:open,-license:non-commercial,-license:conditional" {
		t.Fatalf("with deny: %q", v)
	}
	if v = allow.With(v, nil); v != "-license:non-commercial,-license:conditional" {
		t.Fatalf("without allow: %q", v)
	}
	if v = deny.With(v, nil); v != "" {
		t.Fatalf("without either: %q", v)
	}
}

func TestNumberedSources(t *testing.T) {
	src, _ := KeyByName("model-source")
	if !src.List() || !src.Owns("BONSAIAI_MODEL_SOURCE_2") || src.Owns("BONSAIAI_MODEL_SOURCES") {
		t.Fatalf("%+v", src)
	}
	p := filepath.Join(t.TempDir(), ".env")
	os.WriteFile(p, []byte("# mine\nBONSAIAI_MODEL_SOURCE_10=s3://c\nBONSAIAI_CACHE=/x\nBONSAIAI_MODEL_SOURCE=file:///a\nBONSAIAI_MODEL_SOURCE_2=https://b\n"), 0o600)
	items, set, err := GetIn(p, src)
	if err != nil || !set || fmt.Sprint(items) != "[file:///a https://b s3://c]" {
		t.Fatalf("items %q %v %v", items, set, err)
	}
	if err := SetIn(p, src, []string{"https://b", "s3://c"}); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(p)
	if want := "# mine\nBONSAIAI_MODEL_SOURCE=https://b\nBONSAIAI_MODEL_SOURCE_1=s3://c\nBONSAIAI_CACHE=/x\n"; string(b) != want {
		t.Fatalf("file:\n%s\nwant:\n%s", b, want)
	}
	if err := SetIn(p, src, nil); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(p); string(b) != "# mine\nBONSAIAI_CACHE=/x\n" {
		t.Fatalf("after removing all: %q", b)
	}
}
