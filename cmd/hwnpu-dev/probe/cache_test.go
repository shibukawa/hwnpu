package probe

import (
	"strings"
	"testing"
)

func TestCacheHitsAndReport(t *testing.T) {
	r := cacheResult{Format: "fp16", Layers: 4, ProgramBytes: 1 << 29, Programs: 2, TotalBytes: 1 << 30, Hold: "close", Order: "same", HitS: 1}
	r.Rounds = []cacheRound{
		{Exe: "hwnpu-dev", Order: []int{0, 1}, Passes: [][]cacheLoad{{{Program: 0, LoadS: 30}, {Program: 1, LoadS: 25}}}},
		{Exe: "hwnpu-dev", Order: []int{0, 1}, Passes: [][]cacheLoad{{{Program: 0, LoadS: 0.1}, {Program: 1, LoadS: 0.1, RunS: 0.05}}}},
	}
	if !r.warm() {
		t.Error("every round after the first hit: warm")
	}
	if got := r.hits(r.Rounds[0].Passes[0]); got != 0 {
		t.Errorf("first round hits %d, want 0", got)
	}
	cold := r
	cold.TotalBytes, cold.Programs = 2<<30, 4
	cold.Rounds = append([]cacheRound(nil), r.Rounds...)
	cold.Rounds = append(cold.Rounds, cacheRound{Exe: "other", Passes: [][]cacheLoad{{{Program: 0, LoadS: 0.1}, {Program: 1, LoadS: 40}}}})
	if !cold.warm() {
		t.Error("a miss under another executable's name does not count against warm")
	}
	cold.Rounds[1] = cacheRound{Exe: "hwnpu-dev", Order: []int{0, 1}, Passes: [][]cacheLoad{{{Program: 0, LoadS: 0.1}, {Program: 1, LoadS: 12}}}}
	if cold.warm() {
		t.Error("a second round with a miss: not warm")
	}
	var sb strings.Builder
	cacheMarkdown(&sb, []cacheResult{r, cold})
	md := sb.String()
	for _, want := range []string{"0/2, 2/2", "1/2 (other)", "warm up to 1.00 GiB, not at 2.00 GiB", "[1] 12.0*", "[1] 40.0*"} {
		if !strings.Contains(md, want) {
			t.Errorf("report lacks %q:\n%s", want, md)
		}
	}
}

func TestCacheOptsParse(t *testing.T) {
	o := cacheOpts{format: "fp16", rounds: 3, passes: 1, holds: []string{"close"}, orders: []string{"same"}}
	if err := o.parse("0.5,1", "4,1,2", ""); err != nil {
		t.Fatal(err)
	}
	if o.totals[0] != 1 || o.totals[2] != 4 {
		t.Errorf("totals %v, want ascending", o.totals)
	}
	o.format = "int4"
	if err := o.parse("1", "1", ""); err == nil {
		t.Error("int4 does not run on the Neural Engine")
	}
	o.format, o.keep = "lut4", true
	if err := o.parse("1", "1", ""); err == nil {
		t.Error("-cache-keep without -dir")
	}
}
