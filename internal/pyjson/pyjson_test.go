package pyjson

import (
	"strings"
	"testing"
)

// Write and WriteIndent agree with Python's json.dumps(v, ensure_ascii=False[, indent=2]).
// Numbers keep the text they were read with, which Python would write in its own form.
func TestWrite(t *testing.T) {
	v, err := Parse([]byte(`{"b": [1, 2.5, {}], "a": {"x": null, "y": []}, "名": "a\"b\n\u0001"}`))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := v.String(), `{"b": [1, 2.5, {}], "a": {"x": null, "y": []}, "名": "a\"b\n\u0001"}`; got != want {
		t.Errorf("Write\n%s\nwant\n%s", got, want)
	}
	var sb strings.Builder
	v.WriteIndent(&sb, 2)
	want := "{\n  \"b\": [\n    1,\n    2.5,\n    {}\n  ],\n  \"a\": {\n    \"x\": null,\n    \"y\": []\n  },\n  \"名\": \"a\\\"b\\n\\u0001\"\n}"
	if sb.String() != want {
		t.Errorf("WriteIndent\n%s\nwant\n%s", sb.String(), want)
	}
}

func TestTruthy(t *testing.T) {
	for js, want := range map[string]bool{`null`: false, `false`: false, `""`: false, `0`: false, `0.0`: false, `-0e3`: false,
		`[]`: false, `{}`: false, `true`: true, `"0"`: true, `1e-9`: true, `[0]`: true, `{"a": null}`: true} {
		v, err := Parse([]byte(js))
		if err != nil {
			t.Fatal(err)
		}
		if v.Truthy() != want {
			t.Errorf("%s: truthy %v", js, !want)
		}
	}
	var missing *Value
	if missing.Truthy() || !missing.IsNull() {
		t.Error("a missing value is not null")
	}
}
