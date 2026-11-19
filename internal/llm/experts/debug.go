package experts

import (
	"os"
	"strings"
)

// DebugOpt reads an item of HWNPU_MOE_DEBUG, a list of items separated by commas, each NAME or
// NAME=VALUE: "time" (or "1") prints what the experts took; "usage=FILE" adds the counts of the
// experts each layer ran to FILE (Usage) when the experts close; "prefault=0" leaves their pages
// to the first runs, "prefault=wait" has every page in before the first run.
func DebugOpt(name string) (string, bool) {
	for _, it := range strings.Split(os.Getenv("HWNPU_MOE_DEBUG"), ",") {
		k, v, _ := strings.Cut(strings.TrimSpace(it), "=")
		if k == "1" {
			k = "time"
		}
		if k == name {
			return v, true
		}
	}
	return "", false
}

// DebugTime is whether HWNPU_MOE_DEBUG asks for times.
func DebugTime() bool {
	_, ok := DebugOpt("time")
	return ok
}
