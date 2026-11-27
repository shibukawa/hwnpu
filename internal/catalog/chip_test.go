package catalog

import "testing"

func TestChipOf(t *testing.T) {
	for brand, want := range map[string]string{
		"Apple M1": "M1", "Apple M1 Pro": "M1", "Apple M2": "M2", "Apple M3 Max": "M3", "Apple M4": "M4",
		"Intel(R) Core(TM) Ultra 5 135U": "", "Apple M9": "", "": "",
	} {
		if got := chipOf(brand); got != want {
			t.Errorf("chipOf(%q) = %q, want %q", brand, got, want)
		}
	}
}
