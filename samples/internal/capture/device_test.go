package capture

import "testing"

func TestPick(t *testing.T) {
	list := func() ([]string, error) { return []string{"BuiltInMicrophoneDevice", "AppleUSBAudioEngine:x:2"}, nil }
	for _, c := range []struct{ in, want string }{{"", ""}, {"1", "BuiltInMicrophoneDevice"}, {"2", "AppleUSBAudioEngine:x:2"}, {"AppleUSBAudioEngine:x:2", "AppleUSBAudioEngine:x:2"}} {
		if got, err := pick(c.in, list); err != nil || got != c.want {
			t.Errorf("pick(%q) = %q, %v; want %q", c.in, got, err, c.want)
		}
	}
	for _, in := range []string{"0", "3"} {
		if _, err := pick(in, list); err == nil {
			t.Errorf("pick(%q): no error for a number past the list", in)
		}
	}
}
