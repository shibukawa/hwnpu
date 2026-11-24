package sourcescmd

import (
	"testing"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/bonsaiai/cmd/bonsaiai-dev/devopt"
)

// TestParse: bonsaiai-dev sources alone lists them; its commands parse with the definitions' flag.
func TestParse(t *testing.T) {
	var c struct {
		Sources Cmd `cmd:""`
	}
	p, err := kong.New(&c, kong.Vars(devopt.Vars))
	if err != nil {
		t.Fatal(err)
	}
	k, err := p.Parse([]string{"sources"})
	if err != nil || k.Command() != "sources list" {
		t.Errorf("sources: %v, %v", k, err)
	}
	for _, args := range [][]string{{"sources", "ls", "1", "--original"}, {"sources", "--models", "x.yaml", "gc", "1", "--model", "a"}, {"sources", "import", "."}, {"sources", "update"}, {"sources", "verify", "."}} {
		if _, err := p.Parse(args); err != nil {
			t.Errorf("%v: %v", args, err)
		}
	}
}
