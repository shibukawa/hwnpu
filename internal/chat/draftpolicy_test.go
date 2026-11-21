package chat

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// lookupDraft's drafts follow the latest earlier occurrence of the sequence's last 4 tokens (else
// 3, 2, 1), and its match is how many tokens that occurrence repeats, up to 32.
func TestLookupDraftMatch(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for range 3000 {
		seq := make([]int32, 1+rng.IntN(80))
		for i := range seq {
			seq[i] = int32(rng.IntN(3))
		}
		k := 1 + rng.IntN(7)
		last := len(seq) - 1
		end := -1 // the earlier occurrence's last token
	find:
		for n := min(4, last); n >= 1; n-- {
			for e := last - 1; e >= n-1; e-- {
				if slices.Equal(seq[e-n+1:e+1], seq[last-n+1:]) {
					end = e
					break find
				}
			}
		}
		var want []int32
		m := 0
		if end >= 0 {
			want = seq[end+1 : min(end+1+k, len(seq))]
			for m < 32 && end-m >= 0 && seq[end-m] == seq[last-m] {
				m++
			}
		}
		if got, n := lookupDraft(seq, k); !slices.Equal(got, want) || n != m {
			t.Fatalf("%v: drafts %v behind a match of %d, want %v behind %d", seq, got, n, want, m)
		}
		if got := LookupDraft(seq, k); !slices.Equal(got, want) {
			t.Fatalf("%v: LookupDraft %v, want %v", seq, got, want)
		}
	}
	seq := []int32{1, 2, 3, 4, 5, 6, 9, 1, 2, 3, 4, 5, 6}
	if d, m := lookupDraft(seq, 2); !slices.Equal(d, []int32{9, 1}) || m != 6 {
		t.Fatalf("drafts %v behind a match of %d, want [9 1] behind 6", d, m)
	}
	if d, m := lookupDraft([]int32{7, 8, 9}, 2); d != nil || m != 0 {
		t.Fatalf("drafts %v behind a match of %d with nothing repeated", d, m)
	}
}

func TestDraftPolicy(t *testing.T) {
	p := &draftPolicy{ratio: draftCheckSteps}
	if p.take(1, 2) {
		t.Fatal("a match of 1 token is checked before anything is known")
	}
	if !p.take(8, 2) {
		t.Fatal("a match of 8 tokens is not checked")
	}
	// drafts behind short matches turn out right: the policy comes to check them
	for range 50 {
		p.leave(1, []int32{7, 8})
		p.saw(7)
		p.saw(8)
	}
	if len(p.open) != 0 || !p.take(1, 2) {
		t.Fatalf("after 50 right drafts: %d left open, %.2f right", len(p.open), p.q(0))
	}
	// and stops when they turn out wrong
	for range 200 {
		p.leave(1, []int32{7, 8})
		p.saw(9)
	}
	if p.take(1, 2) {
		t.Fatalf("after 200 wrong drafts: %.2f right", p.q(0))
	}
	// a check that kept one of two drafts counts one right and the end of a run
	p = &draftPolicy{ratio: draftCheckSteps}
	p.checked(5, 2, 1, 0)
	if b := draftBucket(5); p.ok[b] != 1 || p.bad[b] != 1 {
		t.Fatalf("one of two drafts kept: %.0f right, %.0f ended", p.ok[b], p.bad[b])
	}
	// checking every draft (HWNPU_DRAFT_POLICY=0)
	if p = (&draftPolicy{ratio: draftCheckSteps, always: true}); !p.take(1, 2) {
		t.Fatal("always: a draft is left unchecked")
	}
}
