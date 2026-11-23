package airpc

import (
	"context"
	"testing"
)

// chatOptions has pointers whose zero values mean something (text.ChatOptions' Think false).
type chatOptions struct {
	Think  *bool
	Inner  *inner
	Absent *bool
}

type inner struct{ N *int }

func (s *service) Think(ctx context.Context, o chatOptions, p *int) string {
	out := ""
	switch {
	case o.Think == nil:
		out += "think unset"
	case *o.Think:
		out += "think"
	default:
		out += "no think"
	}
	if o.Inner != nil && o.Inner.N != nil && *o.Inner.N == 0 {
		out += ", inner zero"
	}
	if o.Absent != nil {
		out += ", absent set"
	}
	if p != nil && *p == 0 {
		out += ", p zero"
	}
	return out
}

// TestPointersToZero: a pointer to a zero value is sent as one, not as nil (gob's own).
func TestPointersToZero(t *testing.T) {
	h, _ := pair(t)
	no, zero, z := false, 0, 0
	if s := Value[string](h, "Think", chatOptions{Think: &no, Inner: &inner{N: &zero}}, &z); s != "no think, inner zero, p zero" {
		t.Errorf("pointers to zero values: %q", s)
	}
	if s := Value[string](h, "Think", chatOptions{}, (*int)(nil)); s != "think unset" {
		t.Errorf("nil pointers: %q", s)
	}
}
