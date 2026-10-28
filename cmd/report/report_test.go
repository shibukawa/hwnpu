package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/amd"
	"github.com/shibukawa/hwnpu/internal/xdna"
	"github.com/shibukawa/hwnpu/internal/xdna/xdnasim"
)

func init() {
	backend.Register(amd.Provider{
		ProviderName: "xdna-sim",
		Nodes:        func() ([]amd.Node, error) { return []amd.Node{{Path: "/dev/accel/accel9", Device: "0x1502"}}, nil },
		OpenDriver: func(string) (xdna.Driver, error) {
			sim := xdnasim.New()
			sim.Run = xdnasim.RunPassthrough
			return sim, nil
		},
	})
}

// The AMD path of the report runs against the simulated NPU, with the design given as files
// exactly like -xclbin / -insts (insts.txt format).
func TestReportAMD(t *testing.T) {
	dir := t.TempDir()
	xclbinPath := filepath.Join(dir, "final.xclbin")
	instsPath := filepath.Join(dir, "insts.txt")
	os.WriteFile(xclbinPath, xdnasim.PassthroughXclbin(), 0o644)
	ins := xdnasim.PassthroughInstructions(16384)
	var txt strings.Builder
	for i := 0; i < len(ins); i += 4 {
		fmt.Fprintf(&txt, "%08x\n", binary.LittleEndian.Uint32(ins[i:]))
	}
	os.WriteFile(instsPath, []byte(txt.String()), 0o644)
	amdDesign.xclbin, amdDesign.insts, amdDesign.bytes, amdDesign.kernel = xclbinPath, instsPath, 16384, "MLIR_AIE"
	defer func() { amdDesign.xclbin = "" }()

	rep := run(context.Background(), 3, nil)
	var sim *DeviceResult
	for i := range rep.Devices {
		if rep.Devices[i].Info.Backend == "xdna-sim" {
			sim = &rep.Devices[i]
		}
	}
	if sim == nil {
		t.Fatal("simulated device missing from the report")
	}
	if len(sim.Checks) != 4 {
		t.Fatalf("checks %+v", sim.Checks)
	}
	for _, c := range sim.Checks {
		if c.Status != Pass {
			t.Errorf("%s: %s %s", c.Name, c.Status, c.Detail)
		}
	}
	if k := sim.Checks[3]; k.Runs != 3 || k.EnergyScope != "npu" {
		t.Errorf("kernel check %+v", k)
	}
	md := markdown(rep)
	if !strings.Contains(md, "IRON passthrough kernel") || !strings.Contains(md, "AMD NPU npu1") {
		t.Errorf("markdown:\n%s", md)
	}
}
