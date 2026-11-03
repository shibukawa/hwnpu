package amdxrt

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/types"
)

func TestFindWatts(t *testing.T) {
	for _, c := range []struct {
		js   string
		want float64
		ok   bool
	}{
		{`{"power_consumption_watts":"3.25","power_consumption_max_watts":"20"}`, 3.25, true},
		{`{"electrical":{"power_rails":[],"power_consumption_watts":1.5}}`, 1.5, true},
		{`{"electrical":{"power_rails":[]}}`, 0, false},
		{`{"power_consumption_watts":"N/A"}`, 0, false},
	} {
		var v any
		if err := json.Unmarshal([]byte(c.js), &v); err != nil {
			t.Fatal(err)
		}
		got, ok := findWatts(v)
		if ok != c.ok || got != c.want {
			t.Errorf("%s: %v %v, want %v %v", c.js, got, ok, c.want, c.ok)
		}
	}
}

func TestNotListedUnlessAsked(t *testing.T) {
	if listing() {
		t.Skip("listing is on (Windows, or HWNPU_BACKEND=xrt)")
	}
	infos, err := Provider{}.Enumerate(context.Background())
	if err != nil || len(infos) != 0 {
		t.Fatalf("%v %v", infos, err)
	}
}

// TestDevice runs on an AMD NPU through XRT with the shim built (native/xrtshim): set
// HWNPU_XRTSHIM to it.
func TestDevice(t *testing.T) {
	if os.Getenv(ShimEnv) == "" {
		t.Skip("set " + ShimEnv + " to a built xrtshim")
	}
	Enable()
	ctx := context.Background()
	infos, err := Provider{}.Enumerate(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) == 0 {
		t.Skip("no device")
	}
	t.Logf("%+v", infos[0])
	d, err := Provider{}.Open(ctx, infos[0], backend.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	for _, kind := range []types.BufferKind{types.BufferDefault, types.BufferInstructions} {
		b, err := d.AllocBuffer(8192, kind)
		if err != nil {
			t.Fatal(err)
		}
		for i := range b.Bytes() {
			b.Bytes()[i] = byte(i)
		}
		if err := b.SyncToDevice(0, 8192); err != nil {
			t.Fatal(err)
		}
		if err := b.SyncFromDevice(4096, 4096); err != nil {
			t.Fatal(err)
		}
		if b.Bytes()[5000] != byte(5000%256) {
			t.Errorf("kind %d: contents lost", kind)
		}
		if a, ok := b.(interface{ DeviceAddress() uint64 }); ok && kind == types.BufferDefault && a.DeviceAddress() == 0 {
			t.Error("no device address")
		}
		b.Close()
	}
}
