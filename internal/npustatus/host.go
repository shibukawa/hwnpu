package npustatus

import (
	"context"
	"errors"
	"os"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/aieng"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/cachecmd"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/daemon"
	"github.com/shibukawa/hwnpu/internal/types"
	"github.com/shibukawa/hwnpu/modelstore"
)

// BackendEnv names the environment variable that picks the device the commands run on
// (internal/llm's BackendEnv).
const BackendEnv = "HWNPU_BACKEND"

// Host is what hwnpud's engine serves of hwnpu status on a session (internal/aihost), beside
// cachecmd.Host's CacheState: what needs the NPU opened.
type Host interface {
	// NPUDevices opens the NPUs hwnpud's engine finds (hwnpu status --wake).
	NPUDevices(ctx context.Context) (Devices, error)
}

// Devices are the NPUs an engine found, and the error of looking for them ("" for none).
type Devices struct {
	List []types.DeviceInfo
	Err  string
}

// LocalDevices are the NPUs the engine of this process finds (hwnpud's: Host.NPUDevices).
func LocalDevices(ctx context.Context) Devices {
	m, err := aieng.GetMachine()
	if err != nil {
		return Devices{Err: err.Error()}
	}
	d, err := m.Enumerate(ctx)
	out := Devices{List: d}
	if err != nil {
		out.Err = err.Error()
	}
	return out
}

// enumerate asks hwnpud for the NPUs, or with HWNPU_DAEMON=off this process's engine.
func enumerate(ctx context.Context) ([]types.DeviceInfo, error) {
	c, err := daemon.Use(ctx, daemon.DefaultMode(), console.Log(os.Stderr))
	if err != nil {
		return nil, err
	}
	var d Devices
	if c == nil {
		d = LocalDevices(ctx)
	} else {
		s, err := c.DialAI(ctx)
		if err != nil {
			return nil, err
		}
		defer s.Close()
		h := airpc.Root[Host](s)
		if d, err = airpc.Call1[Devices](h, ctx, "NPUDevices"); err != nil {
			return nil, err
		}
	}
	if d.Err != "" {
		return d.List, errors.New(d.Err)
	}
	return d.List, nil
}

// backendVendor is the vendor HWNPU_BACKEND's value b names (as internal/chat's HereVendor reads
// it): a backend of Intel's programs (intel, levelzero, openvino) intel, coreml or coreai apple;
// "" for another or none.
func backendVendor(b string) string {
	switch strings.TrimSpace(b) {
	case "intel", "levelzero", "openvino":
		return "intel"
	case "coreml", "coreai":
		return "apple"
	}
	return ""
}

// vendorFound is the NPU the commands run on once the devices are opened (internal/chat's
// HereVendor): HWNPU_BACKEND's, else the first vendor of devs hwnpu runs on.
func vendorFound(devs []types.DeviceInfo) string {
	if v := backendVendor(os.Getenv(BackendEnv)); v != "" {
		return v
	}
	for _, d := range devs {
		if v := d.Vendor.String(); slices.Contains(catalog.Vendors, v) {
			return v
		}
	}
	return ""
}

// OpenView is the cache that status shows: hwnpud's, or with HWNPU_DAEMON=off the store o opens
// in this process (its own settings, not this shell's). close ends hwnpud's session.
func OpenView(ctx context.Context, o modelstore.Options) (v StoreView, close func(), err error) {
	rv, closeRemote, err := cachecmd.RemoteStore(ctx)
	if err != nil {
		return nil, nil, err
	}
	if rv != nil {
		return rv, closeRemote, nil
	}
	s, err := modelstore.Open(o)
	if err != nil {
		return nil, nil, err
	}
	return View{s}, func() {}, nil
}
