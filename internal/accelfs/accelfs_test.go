package accelfs

import (
	"os"
	"path/filepath"
	"testing"
)

func TestListAt(t *testing.T) {
	root := t.TempDir()
	mk := func(name, driver, vendor string) {
		dev := filepath.Join(root, "class", name, "device")
		if err := os.MkdirAll(dev, 0o755); err != nil {
			t.Fatal(err)
		}
		drv := filepath.Join(root, "drivers", driver)
		os.MkdirAll(drv, 0o755)
		if err := os.Symlink(drv, filepath.Join(dev, "driver")); err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(dev, "vendor"), []byte(vendor+"\n"), 0o644)
	}
	mk("accel1", DriverAMD, "0x1022")
	mk("accel0", DriverIntel, "0x8086")
	nodes, err := ListAt(filepath.Join(root, "class"), "/dev/accel")
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes) != 2 || nodes[0].Path != "/dev/accel/accel0" || nodes[0].Driver != DriverIntel || nodes[0].Vendor != "0x8086" {
		t.Fatalf("%+v", nodes)
	}
	if got := ByDriver(nodes, DriverAMD); len(got) != 1 || got[0].Name != "accel1" {
		t.Fatalf("%+v", got)
	}
	if n, err := ListAt(filepath.Join(root, "missing"), "/dev/accel"); err != nil || n != nil {
		t.Fatalf("missing root: %v %v", n, err)
	}
}
